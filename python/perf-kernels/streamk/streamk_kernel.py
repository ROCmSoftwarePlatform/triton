import triton
import triton.language as tl

@triton.jit()
def get_tiles_config(M, N, K, total_programs_streamk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    total_blocks_M = tl.cdiv(M, BLOCK_M)
    total_blocks_N = tl.cdiv(N, BLOCK_N)
    iters_per_tile = tl.cdiv(K, BLOCK_K)

    total_tiles = total_blocks_M * total_blocks_N
    if total_programs_streamk > 0:  # Stream-K
        total_tiles_streamk = total_tiles 
        total_iters_streamk = total_tiles_streamk * iters_per_tile
        # iterations related to full waves
        total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
        # iterations related to last (partial) wave
        total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

    else:  # all tiles are computed using classical blocking
        total_tiles_streamk = 0
        total_full_tiles_streamk = 0
        total_partial_tiles_streamk = 0
        total_iters_streamk = 0

    return iters_per_tile, total_tiles_streamk, total_full_tiles_streamk, total_partial_tiles_streamk, total_iters_streamk

@triton.jit()
def streamk_gemm(
        A, B, C,
        M, N, K, total_programs_streamk,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        ACC_TYPE: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    iters_per_tile, total_tiles_streamk, total_full_tiles_streamk, total_partial_tiles_streamk, total_iters_streamk = get_tiles_config(M, N, K, total_programs_streamk, BLOCK_M, BLOCK_N, BLOCK_K)

    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)
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

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_K * stride_bk * remainder
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for current_iter in range(start_iter, end_iter):
            a = tl.load(A_BASE)
            b = tl.load(B_BASE)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_K * stride_ak
            B_BASE += BLOCK_K * stride_bk

        if remainder ==0 and end_iter % iters_per_tile ==0:
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn  # compute inside the if/else to avoid spilling!
        #    mask = (rm < M)[:, None] & (rn < N)[None, :]
        #    tl.store(C_, acc, mask=mask)
            tl.store(C_, acc)
        else:
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn  # compute inside the if/else to avoid spilling!
        #    mask = (rm < M)[:, None] & (rn < N)[None, :]
        #    tl.atomic_add(C_, acc, mask=mask)
            tl.atomic_add(C_, acc)

        start_iter = end_iter
