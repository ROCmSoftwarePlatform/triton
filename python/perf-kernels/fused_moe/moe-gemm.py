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


@triton.jit
def moe_gemm_kernel(
    A,
    B,
    Out,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    EM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    # Here we assume that valid tokens are in the range [0, M).
    token_mask = (offs_token >= 0) & (offs_token < EM)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Masking ensures we don't load from invalid tokens or indices
        a = tl.load(a_ptrs, mask=(token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = Out + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(out_ptrs, accumulator.to(Out.dtype.element_ty), mask=c_mask)


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


def moe_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
             sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor,
             config) -> None:
    # TODO shard M dim
    _, top_k = topk_ids.shape

    EM = num_tokens_post_padded.item()
    _, N, K = b.shape
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    moe_gemm_kernel[grid](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(1),
                          c.stride(2), top_k, topk_weights, sorted_token_ids, expert_ids, EM, N, K,
                          MUL_ROUTED_WEIGHT=topk_weights is not None, **config)
    return c


def input_helper(M: int, K: int, N: int, top_k: int, E: int, routed_weight: bool, dtype):
    a = torch.randn((M, K), dtype=dtype, device='cuda')
    b = torch.randn((E, N, K), dtype=dtype, device='cuda')
    c = torch.zeros((M, top_k, N), dtype=dtype, device='cuda')

    values = torch.randn(M, E, device='cuda')

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
        return a, b, c, None, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config

    return a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config


@pytest.mark.parametrize("M, K, N, top_k, E", [
    (64, 4096, 14336, 2, 8),
    (16, 1, 14336, 2, 4),
    (1, 128, 14336, 2, 4),
    (16, 128, 14336, 1, 4),
    (16, 128, 14336, 1, 1),
    (64, 128, 7186, 2, 8),
    (64, 128, 3584, 2, 8),
    (64, 128, 1792, 2, 8),
    (64, 128, 64, 2, 8),
])
@pytest.mark.parametrize('routed_weight', [True, False])
def test_correctness(M: int, K: int, N: int, top_k: int, E: int, routed_weight: bool, dtype=torch.float16):
    torch.manual_seed(20)
    a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config = input_helper(
        M, K, N, top_k, E, routed_weight=routed_weight, dtype=dtype)

    tri_out = moe_gemm(a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config)

    ref_out = torch.empty_like(c)
    # Repeat a -> (M, top_k, K)
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
    # (M, top_k, N, K)
    b_indexed = b[topk_ids]
    ref_out = torch.einsum("mek,menk->men", a_expanded, b_indexed)
    if routed_weight:
        ref_out *= topk_weights.unsqueeze(-1)

    # Validate correctness
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)


def get_configs():
    configs = [
        {"M": 64, "K": 128, "N": 256, "E": 8, "top_k": 2},
        {"M": 64, "K": 1024, "N": 1792, "E": 8, "top_k": 2},
        {"M": 64, "K": 4096, "N": 7168, "E": 8, "top_k": 2},
        {"M": 128, "K": 4096, "N": 7168, "E": 8, "top_k": 2},
        {"M": 1024, "K": 4096, "N": 7168, "E": 8, "top_k": 2},
        {"M": 4096, "K": 4096, "N": 7168, "E": 8, "top_k": 2},
        {"M": 64, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
        {"M": 128, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
        {"M": 256, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
        {"M": 512, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
        {"M": 1024, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
        {"M": 2048, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
        {"M": 4096, "K": 4096, "N": 14336, "E": 8, "top_k": 2},
    ]
    return configs


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, model_families=["mistral"], model=args.model)
    fa_configs = []
    M = args.M if args.M else 1024  # check size
    # M, K, N, E, top_k

    for model_name, config in configs.items():
        N = config["intermediate_size"]
        K = config["hidden_size"]

        E = 8
        top_k = 2
        fa_configs.append((model_name, M, K, N, E, top_k))

    return fa_configs


def run_benchmark(custom, args):
    print_time = args.return_time
    routed_weight = args.routed_weight
    dtype = arg_to_torch_dtype[args.dtype]
    x_names = ['M', 'K', 'N', 'E', 'top_k']
    if custom:
        assert args.M and args.K and args.N and args.E and args.top_k, \
            "Please provide M, K, N, E, top_k for custom runs."
        x_vals_list = [(args.M, args.K, args.N, args.E, args.top_k)]
    else:
        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ['model', 'M', 'K', 'N', 'E', 'top_k']
        else:
            configs = get_configs()
            x_vals_list = [(cfg['M'], cfg['K'], cfg['N'], cfg['E'], cfg['top_k']) for cfg in configs]

    line_names = ['Time (ms)', 'Bandwidth (GB/s)'] if print_time else ['TFLOPS', 'Bandwidth (GB/s)']

    if print_time:
        # We'll have 2 lines: 'time' and 'bandwidth'
        line_vals = ['time', 'bandwidth']
        line_names = ['Time (ms)', 'Bandwidth (GB/s)']
    else:
        line_vals = ['tflops', 'bandwidth']
        line_names = ['TFLOPS', 'Bandwidth (GB/s)']

    benchmark = triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='metric',  # <--- important
                                         line_vals=line_vals,  # <--- a list of 2 metrics
                                         line_names=line_names,  # <--- matching 2 metrics
                                         styles=[('red', '-'),
                                                 ('blue', '-')], ylabel='ms / TFLOPS / GB/s',  # or a more generic label
                                         plot_name='moe-gemm-benchmark',
                                         args={'dtype': dtype, 'routed_weight': routed_weight})

    @triton.testing.perf_report([benchmark])
    def bench_moe_gemm(M, K, N, E, top_k, dtype, routed_weight, metric, model=None):
        # metric will be either 'time'/'tflops' or 'bandwidth'
        a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config = input_helper(
            M, K, N, top_k, E, routed_weight=routed_weight, dtype=dtype)

        flops = 2.0 * M * top_k * K * N
        if routed_weight:
            flops += M * top_k * N

        bytes_ = torch.tensor([], dtype=dtype).element_size()
        mem_read = (M * K + E * N * K) * bytes_
        mem_write = (M * top_k * N) * bytes_
        mem = mem_read + mem_write
        fn = lambda: moe_gemm(a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
                              config)
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
    parser.add_argument("-return_time", action='store_true', default=False, help='Return time instead of TFLOPs')
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
