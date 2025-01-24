import triton
import torch
import triton.language as tl
import pytest
from typing import Any, Dict, Optional
import os
import json
import functools
import argparse
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # This goes one level up from fused-moe/
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from utils.benchmark_utils import get_available_models, get_model_configs  # noqa: E402

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


def torch_moe(a, b, c, topk_ids, topk_weights, routed_weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
              a_scale, b_scale, b_scale_int4, dtype, fp8, int4):
    E, N, K = b.shape
    M, topk, _ = c.shape
    c = c.reshape(-1, c.shape[2])

    if int4:
        b_dequant = torch_int4_to_fp8_dequant(b, b_scale_int4)
        b = b_dequant

    if fp8:
        a = a.to(dtype)

    for e in range(E):
        token_ids = (topk_ids == e).any(dim=-1)
        flat_topk_ids = topk_ids.view(-1)
        flat_token_ids = torch.arange(topk_ids.numel(), device=topk_ids.device)
        c_token_ids = flat_token_ids[flat_topk_ids == e]

        b_e = b[e]
        a_e = a[token_ids, :]

        if fp8:
            b_e = b_e.to(dtype)

        acc = torch.matmul(a_e, b_e.T)
        if routed_weight:
            acc = acc * topk_weights.view(-1)[c_token_ids].unsqueeze(-1)

        if fp8:
            acc = (acc * a_scale * b_scale[e]).to(dtype)

        c[c_token_ids, :] = acc

    c = c.reshape(M, topk, N)

    return c


def torch_int4_to_fp8_dequant(qweights,  # E, N, K/8
                              scales,  #E, N
                              ):
    E, N, K8 = qweights.shape

    eweights = torch.unsqueeze(qweights, 3)
    eweights = torch.broadcast_to(eweights, (E, N, K8, 8))
    weights = torch.reshape(eweights, (E, N, K8 * 8))

    reverse_order_tensor = ((torch.arange(0, 2) * 4)[None, :] + torch.arange(0, 4)[:, None]).reshape(8)

    shifts = reverse_order_tensor * 4
    shifts = torch.broadcast_to(shifts[None, :], (K8 * N, 8))  #(K8*N,8)
    shifts = torch.reshape(shifts, (N, K8 * 8))  #(N,K)
    shifts = torch.unsqueeze(shifts, 0)
    shifts = torch.broadcast_to(shifts, (E, N, K8 * 8)).to(qweights.device)

    weights = ((weights >> shifts) & 0xF)  #(E, N, K)
    weights = torch.where(weights >= 8, weights - 16, weights)

    scales = torch.broadcast_to(scales[:, :, None], (E, N, K8 * 8))
    dweights = weights * scales
    dweights = dweights.to(dtype=torch.float8_e4m3fnuz)

    return dweights


@triton.jit
def int4_to_fp8_dequant(qweights,  # quantized matrix, K/8 x N
                        scales,  # scales, per channel (N,)
                        K8: tl.constexpr,  #K/8
                        N: tl.constexpr):
    #qweights = qweights.trans(1, 0)  #(N,K8)
    #qweights = tl.interleave(qweights, qweights)
    #qweights = tl.interleave(qweights, qweights)
    #weights = tl.interleave(qweights, qweights).trans(1, 0)  #(K,N)

    eweights = tl.expand_dims(qweights, 1)
    eweights = tl.broadcast_to(eweights, (K8, 8, N)) 
    weights = tl.reshape(eweights, (K8*8, N))

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8,1)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in weights
    shifts = reverse_order_tensor * 4 
    shifts = tl.broadcast_to(shifts[None, :], (K8, 8, N))  #(K8*N,8)
    shifts = tl.reshape(shifts, (K8 * 8, N))  #(K,N)

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    weights = ((weights >> shifts) & 0xF)  #(K,N)
    weights = tl.where(weights >= 8, weights - 16, weights)

    scales = tl.broadcast_to(scales[:], (K8 * 8, N))
    dweights = weights * scales
    dweights = dweights.to(tl.float8e4b8)

    return dweights


@triton.jit
def moe_gemm_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, b_scale_int4_ptr, topk_weights_ptr,
                    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N: tl.constexpr, K: tl.constexpr,
                    EM: tl.constexpr, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn,
                    stride_cm, stride_cn, stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn, stride_bsie,
                    stride_bsin, group_n, group_k, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
                    top_k: tl.constexpr, compute_type: tl.constexpr, use_fp8_w8a8: tl.constexpr,
                    #use_int8_w8a16: tl.constexpr,
                    even_Ks: tl.constexpr, use_int4_w: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = (offs_token >= 0) & (offs_token < EM)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_8 = tl.arange(0, BLOCK_SIZE_K // 8)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if use_int4_w:
        #load 1/8 th elements in B in the K direction
        b_ptrs = (b_ptr + off_experts * stride_be + (offs_k_8[:, None] * stride_bk + offs_bn[None, :] * stride_bn))
    else:
        #load regular
        b_ptrs = (b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    if use_fp8_w8a8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_experts)
    if use_int4_w:
        #load b int4 scale
        b_scale_int4_ptrs = (b_scale_int4_ptr + off_experts * stride_bsie + offs_bn[None, :] * stride_bsin)
        b_scale_int4 = tl.load(b_scale_int4_ptrs)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            if use_int4_w:
                #size (BLOCK_SIZE_K /8, BLOCK_SIZE_N)
                b_int4 = tl.load(b_ptrs)
                b = int4_to_fp8_dequant(b_int4, b_scale_int4, b_int4.shape[0], b_int4.shape[1])
            else:
                b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            if use_int4_w:
                #size (BLOCK_SIZE_K /8, BLOCK_SIZE_N)
                b_int4 = tl.load(b_ptrs, mask=offs_k_8[:, None] < (K - k) * BLOCK_SIZE_K / 8, other=0.0)
                b = int4_to_fp8_dequant(b_int4, b_scale_int4, b_int4.shape[0], b_int4.shape[1])
            else:
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.
        if use_fp8_w8a8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w:
            b_ptrs += BLOCK_SIZE_K // 8 * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_fp8_w8a8:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, top_k: int, block_size: int,
                          sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                          num_tokens_post_pad: torch.Tensor) -> None:
    M, top_k = topk_ids.shape

    expert_to_tokens = [[] for _ in range(num_experts)]
    # For each token, for each selected expert, we append (token_id, expert)
    for token_id in range(M):
        for j in range(top_k):
            e_id = topk_ids[token_id, j].item()
            expert_to_tokens[e_id].append(token_id * top_k + j)

    # Reorder tokens block by block, padding if needed
    reordered_token_ids = []
    reordered_expert_ids = []

    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)

        n_blocks = ((num_tokens + block_size - 1) // block_size)
        # If not a multiple of block_size, pad up to the next multiple
        padded_size = n_blocks * block_size

        # Reorder all actual tokens for expert e_id
        reordered_token_ids.extend(tokens_for_expert)
        # reordered_expert_ids.extend([e_id]*num_tokens)
        reordered_expert_ids.extend([e_id] * n_blocks)

        # Pad with dummy token_id = -1 (or any sentinel), if needed
        if padded_size > num_tokens:
            pad_count = padded_size - num_tokens
            reordered_token_ids.extend([-1] * pad_count)

    token_length = len(reordered_token_ids)
    expert_length = len(reordered_expert_ids)

    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=sorted_token_ids.dtype,
                                                   device=sorted_token_ids.device)
    expert_ids[:expert_length] = torch.tensor(reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device)

    # Fill remainder with -1 if these arrays are bigger than total_length
    if token_length < sorted_token_ids.numel():
        sorted_token_ids[token_length:] = -1
    if expert_length < expert_ids.numel():
        expert_ids[expert_length:] = -1

    num_tokens_post_pad.fill_(token_length)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    top_k = topk_ids.shape[1]
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1), ), dtype=torch.int32,
                             device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size(topk_ids, num_experts, top_k, block_size, sorted_ids, expert_ids, num_tokens_post_pad)

    return sorted_ids, expert_ids, num_tokens_post_pad


def get_config_dtype_str(dtype: torch.dtype, use_int8_w8a16: Optional[bool] = False,
                         use_fp8_w8a8: Optional[bool] = False):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def get_config_file_name(dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"device_name={device_name}{dtype_selector}.json"


@functools.lru_cache
def get_moe_configs(dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(dtype)

    config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # If a configuration has been found, return it
            return {key: val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None


def get_default_config(
    M: int,
    E: int,
    is_marlin: bool,
) -> Dict[str, int]:
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}
    return config


def try_get_optimal_moe_config(
    E: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
):
    configs = get_moe_configs(dtype)

    if configs:
        if configs:
            if M < M_THRESHOLD_SMALL:
                config = configs["small_M"]
            elif M < M_THRESHOLD_MEDIUM:
                config = configs["medium_M"]
            else:
                config = configs["large_M"]
    else:
        # Else use the default config
        config = get_default_config(M, E, is_marlin)

    return config


def moe_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor,
             b_scale_int4: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
             sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor,
             config) -> None:
    # TODO shard M dim

    _, top_k = topk_ids.shape

    EM = sorted_token_ids.shape[0]
    _, N, K = b.shape
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    EVEN_K = K % config["BLOCK_SIZE_K"] == 0

    use_fp8 = True if a_scale is not None and b_scale is not None else False
    use_int4 = True if b_scale_int4 is not None else False

    moe_gemm_kernel[grid](
        a, b, c, a_scale, b_scale, b_scale_int4, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, N,
        K * 8 if use_int4 else K, EM, topk_ids.numel(), a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
        c.stride(1), c.stride(2), a_scale.stride(0) if a_scale is not None and a_scale.ndim == 2 else 0,
        a_scale.stride(1) if a_scale is not None and a_scale.ndim == 2 else 0,
        b_scale.stride(0) if b_scale is not None and b_scale.ndim >= 2 else 0,
        b_scale.stride(2) if b_scale is not None and b_scale.ndim == 3 else 0,
        b_scale.stride(1) if b_scale is not None and b_scale.ndim >= 2 else 0,
        b_scale_int4.stride(0) if b_scale_int4 is not None else 0,
        b_scale_int4.stride(1) if b_scale_int4 is not None else 0, 0, 0, MUL_ROUTED_WEIGHT=topk_weights is not None,
        top_k=top_k, even_Ks=EVEN_K, compute_type=tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16,
        use_fp8_w8a8=use_fp8, use_int4_w=use_int4, **config)
    return c


def input_helper(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, dtype, fp8: bool, int4: bool):
    if fp8:
        a = torch.randn((M, K), dtype=dtype, device='cuda')
        a = a.to(torch.float8_e4m3fnuz)
        if int4:
            b = torch.randint(0, 1000000, (E, N, K // 8), dtype=torch.int32, device='cuda')
        else:
            b = torch.rand((E, N, K), dtype=dtype, device='cuda')
            b = b.to(torch.float8_e4m3fnuz)
    else:
        b = torch.randn((E, N, K), dtype=dtype, device='cuda')
        a = torch.randn((M, K), dtype=dtype, device='cuda')
    c = torch.zeros((M, top_k, N), dtype=dtype, device='cuda')

    if fp8:
        a_scale = torch.randn((1), dtype=torch.float32, device='cuda')
        b_scale = torch.randn((E), dtype=torch.float32, device='cuda')
    else:
        a_scale = None
        b_scale = None
    if int4:
        b_scale_int4 = torch.randn((E, N), dtype=torch.float32, device='cuda')
    else:
        b_scale_int4 = None

    values = torch.randn(M, E, dtype=dtype, device='cuda')

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    config_dtype = None
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        E,
        config_dtype,
    )
    config = get_config_func(M)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    if not routed_weight:
        return a, b, c, None, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale, b_scale_int4

    return a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale, b_scale_int4


@pytest.mark.parametrize("M, N, K, top_k, E", [(64, 14336, 4096, 2, 8), (16, 14336, 1, 2, 4), (4, 4, 8, 1, 2),
                                               (1, 14336, 128, 2, 4), (3, 14336, 128, 2, 4), (16, 14336, 128, 1, 4),
                                               (16, 14336, 128, 1, 1), (64, 7186, 128, 2, 8), (64, 3584, 128, 2, 8),
                                               (64, 1792, 128, 2, 8), (64, 64, 128, 2, 8), (1, 1024, 16384, 1, 2)])
@pytest.mark.parametrize('routed_weight', [True, False])
@pytest.mark.parametrize('fp8, int4', [(True, True), (True, False), (False, False)])
def test_correctness(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, fp8: bool, int4: bool,
                     dtype=torch.float16):
    #torch.manual_seed(20)
    if int4:
        if K < 8 or K % 8 != 0:
            pytest.skip("INT4 packed data requires K >=8 and K % 8 == 0"
                        )  #This is because in case int4, we have 8 int4 packed into 1 int32 value.
    a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale, b_scale_int4 = input_helper(
        M, N, K, top_k, E, routed_weight=routed_weight, dtype=dtype, fp8=fp8, int4=int4)

    tri_out = moe_gemm(a, b, c, a_scale, b_scale, b_scale_int4, topk_weights, topk_ids, sorted_token_ids, expert_ids,
                       num_tokens_post_padded, config)

    ref_out = torch.empty_like(c)
    ref_out = torch_moe(a, b, ref_out, topk_ids, topk_weights, routed_weight, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, a_scale, b_scale, b_scale_int4, dtype, fp8, int4)

    # Validate correctness
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


def get_configs():
    configs = [
        {"M": 64, "N": 256, "K": 128, "E": 8, "top_k": 2},
        {"M": 64, "N": 1792, "K": 1024, "E": 8, "top_k": 2},
        {"M": 64, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 128, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 1024, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 4096, "N": 7168, "K": 4096, "E": 8, "top_k": 2},
        {"M": 64, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 128, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 256, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 512, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 1024, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 2048, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
        {"M": 4096, "N": 14336, "K": 4096, "E": 8, "top_k": 2},
    ]
    return configs


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, model_families=["mistral"], model=args.model)
    moe_configs = []
    M = args.M if args.M else 4096  # check size
    # M, K, N, E, top_k

    for model_name, config in configs.items():
        N1 = config["intermediate_size"]
        K1 = config["hidden_size"]

        N2 = config["hidden_size"]
        K2 = config["intermediate_size"] // 2

        E = 8
        top_k = 2
        moe_configs.append((model_name, M, N1, K1, E, top_k))
        moe_configs.append((model_name, M, N2, K2, E, top_k))

    return moe_configs


def run_benchmark(custom, args):
    routed_weight = args.routed_weight
    dtype = arg_to_torch_dtype[args.dtype]
    fp8 = args.fp8
    int4 = args.int4
    x_names = ['M', 'N', 'K', 'E', 'top_k']
    if custom:
        assert args.M and args.N and args.K and args.E and args.top_k, \
            "Please provide M, N, K, E, top_k for custom runs."
        x_vals_list = [(args.M, args.N, args.K, args.E, args.top_k)]
    else:
        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ['model', 'M', 'N', 'K', 'E', 'top_k']
        else:
            configs = get_configs()
            x_vals_list = [(cfg['M'], cfg['N'], cfg['K'], cfg['E'], cfg['top_k']) for cfg in configs]

    line_names = ['Time (ms)', 'TFLOPS', 'Bandwidth (GB/s)']
    line_vals = ['time', 'tflops', 'bandwidth']

    benchmark = triton.testing.Benchmark(
        x_names=x_names, x_vals=x_vals_list, line_arg='metric', line_vals=line_vals, line_names=line_names,
        styles=[('red', '-'), ('blue', '-'), ('yellow', '-')], ylabel='ms / TFLOPS / GB/s',
        plot_name='moe-gemm-benchmark', args={'dtype': dtype, 'routed_weight': routed_weight, 'fp8': fp8, 'int4': int4})

    @triton.testing.perf_report([benchmark])
    def bench_moe_gemm(M, N, K, E, top_k, dtype, routed_weight, metric, fp8, int4, model=None):
        # metric will be either 'time'/'tflops' or 'bandwidth'
        a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale, b_scale_int4 = input_helper(
            M, N, K, top_k, E, routed_weight=routed_weight, dtype=dtype, fp8=fp8, int4=int4)

        # (M, K) * (top_k, N, K) -> (M, top_k, N). 2 for multiplication and accumulation
        flops = 2.0 * M * top_k * K * N
        # The weight is applied on the gemm product which has the shape of (M, top_k, N)
        if routed_weight:
            flops += M * top_k * N

        bytes_ = torch.tensor([], dtype=dtype).element_size()
        # (M, K) memory load for A (E,  N,  K) for B not (top_k,  N,  K) because we are in total bringing in all expert matrices into the chip from memory. It's just that not all multiply the same A.
        mem_read = (M * K) * bytes_
        if int4:
            mem_read +=  E * N * (K // 8) * torch.tensor([], dtype=torch.int32).element_size() 
        else:
            mem_read += E * N * K * bytes_

        # Memory write for the gemm product
        mem_write = (M * top_k * N) * bytes_
        mem = mem_read + mem_write
        fn = lambda: moe_gemm(a, b, c, a_scale, b_scale, b_scale_int4, topk_weights, topk_ids, sorted_token_ids,
                              expert_ids, num_tokens_post_padded, config)
        ms = triton.testing.do_bench(fn)

        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        tflops = flops / ms * 1e-9

        # Return exactly one scalar depending on which metric is active
        if metric == 'time':
            return ms
        elif metric == 'tflops':
            return tflops
        elif metric == 'bandwidth':
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_moe_gemm.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MoE GEMM",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")
    available_models = get_available_models(model_families=["mistral"])  # Dynamically load model names
    model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +
                  "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument("-K", type=int, default=0, help="K dimension")
    parser.add_argument("-N", type=int, default=0, help="N dimension")
    parser.add_argument("-E", type=int, default=0, help="Number of experts")
    parser.add_argument("-top_k", type=int, default=0, help="top_k experts per token")
    parser.add_argument("-routed_weight", action='store_true', default=False)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-fp8", type=bool, default=False)  #fp8 weights and activation
    parser.add_argument("-int4", type=bool, default=False)  #int4 weights and fp8 activation. Must have fp8=True also
    args = parser.parse_args()
    return args


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def main():
    args = parse_args()
    custom_config = False
    # If user provides all M,K,N,E,top_k we consider it custom
    if args.M and args.K and args.N and args.E and args.top_k:
        custom_config = True
    run_benchmark(custom_config, args)


if __name__ == '__main__':
    sys.exit(main())
