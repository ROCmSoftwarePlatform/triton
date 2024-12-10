import triton
import torch
import triton.language as tl
import pytest
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ("gfx1030", "gfx1100", "gfx1101",
                                                                                   "gfx1102", "gfx1200", "gfx1201")

def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']


def get_rdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']


def get_autotune_configs():
    if is_rdna():
        return get_rdna_autotune_configs()
    elif is_cdna():
        return get_cdna_autotune_configs()
    else:
        raise ValueError("Unknown Device Type")
@triton.jit
def moe_gemm_kernel(
    A, B, Out,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    top_k: tl.constexpr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    # MUL_ROUTED_WEIGHT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)

    # Generate token_mask: True if token is valid, False otherwise
    # Here we assume that valid tokens are in the range [0, M).
    token_mask = (offs_token >= 0) & (offs_token < M)

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

    # Load the MoE weights using the token_mask to avoid invalid memory accesses.
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
    accumulator = accumulator * moe_weight[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = Out + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=c_mask)


def moe_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                            topk_weights: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor, top_k: int) -> None:
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(b.shape[1], META['BLOCK_SIZE_N']), )

    M, K = a.shape
    K, N = b.shape
    moe_gemm_kernel[grid](
        a,
        b,
        c,
        a.stride(0), a.stride(0),
        b.stride(0), b.stride(0),
        c.stride(0), c.stride(0),
        top_k,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        M, N, K
    )


def input_helper(M: int, K: int, N: int, top_k: int, E: int):
    """
    Parameters:
    - M: number of tokens after sorting and routing (this may be total tokens * top_k if each token is duplicated for each expert)
    - K: input feature dimension
    - N: output feature dimension
    - top_k: number of experts per token
    - E: number of experts

    Returns:
    (a, b, c, topk_weights, sorted_token_ids, expert_ids, top_k)
    ready to be passed into moe_gemm.
    """

    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')
    c = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    topk_weights = torch.rand((M,), dtype=torch.float32, device='cuda')

    sorted_token_ids = torch.randperm(M, device='cuda', dtype=torch.int64)

    expert_ids = torch.randint(low=0, high=E, size=(M,), device='cuda', dtype=torch.int64)

    return a, b, c, topk_weights, sorted_token_ids, expert_ids, top_k

@pytest.mark.parametrize(
    "M,K,N,top_k,E",
    [
        (128, 64, 256, 2, 4),
        (64, 32, 128, 1, 2),
        (256, 128, 512, 2, 8)
    ]
)
def test_correctness(M: int, K: int, N: int, top_k: int, E: int):
    a, b, c, topk_weights, sorted_token_ids, expert_ids, top_k = input_helper(M, K, N, top_k, E)

    tri_out = moe_gemm(a, b, c, topk_weights, sorted_token_ids, expert_ids, top_k)

    K_original = a.shape[1]
    assert b.shape[0] == E * K_original, "b's first dimension must be E*K_original."

    ref_out = torch.empty_like(c)
    for i in range(M):
        A_index = (sorted_token_ids[i].item() // top_k)
        expert_id = expert_ids[i].item()
        A_i = a[A_index]
        B_e = b[expert_id*K_original:(expert_id+1)*K_original, :]

        # Compute the expert-transformed token
        # (K_original,) @ (K_original, N) → (N,)
        out_i = (A_i @ B_e) * topk_weights[i]
        ref_out[i, :] = out_i

    # Validate correctness
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)
