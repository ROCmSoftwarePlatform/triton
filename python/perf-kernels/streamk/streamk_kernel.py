import triton
import triton.language as tl

@triton.jit()
def get_tiles_config(M, N, K, num_sms,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    total_blocks_M = tl.cdiv(M, BLOCK_SIZE_M)
    total_blocks_N = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)

    total_tiles = total_blocks_M * total_blocks_N
    if num_sms > 0:  # Stream-K
        total_tiles_streamk = total_tiles 
        total_iters_streamk = total_tiles_streamk * iters_per_tile
        # iterations related to full waves
        total_full_tiles_streamk = total_iters_streamk // num_sms
        # iterations related to last (partial) wave
        total_partial_tiles_streamk = total_iters_streamk % num_sms

    else:  # all tiles are computed using classical blocking
        total_tiles_streamk = 0
        total_full_tiles_streamk = 0
        total_partial_tiles_streamk = 0
        total_iters_streamk = 0

    return iters_per_tile, total_tiles_streamk, total_full_tiles_streamk, total_partial_tiles_streamk, total_iters_streamk

@triton.jit()
def streamk_gemm(
        A, B, C,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    iters_per_tile, total_tiles_streamk, total_full_tiles_streamk, total_partial_tiles_streamk, total_iters_streamk = get_tiles_config(M, N, K, num_sms, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)
    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        # where are we in the grid
        tile_id = start_iter // iters_per_tile
        if GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            a = tl.load(A_BASE)
            b = tl.load(B_BASE)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if remainder ==0 and end_iter % iters_per_tile ==0:
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn  # compute inside the if/else to avoid spilling!
            tl.store(C_, acc)
        else:
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn  # compute inside the if/else to avoid spilling!
            tl.atomic_add(C_, acc)

        start_iter = end_iter
