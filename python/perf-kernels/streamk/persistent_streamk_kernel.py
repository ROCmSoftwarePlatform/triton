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
        total_full_tiles_pcu = total_tiles // num_sms
        total_streamk_tiles = total_tiles % num_sms
        total_full_tiles = total_tiles - total_streamk_tiles
        total_streamk_iters = total_streamk_tiles * iters_per_tile
        # iterations related to full waves
        streamk_iters_pcu = total_streamk_iters // num_sms
        # iterations related to last (partial) wave
        streamk_remainder_iters = total_streamk_iters % num_sms

    else:  # all tiles are computed using classical blocking
        total_full_tiles = total_tiles
        total_streamk_tiles = 0
        streamk_iters_pcu = 0
        streamk_remainder_iters = 0
        total_streamk_iters = 0

    return iters_per_tile, total_full_tiles, total_streamk_tiles, streamk_iters_pcu, streamk_remainder_iters

@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit()
def persistent_streamk_gemm(
        A, B, C, P, locks,
        M, N, K, num_sms,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    iters_per_tile, total_full_tiles, total_streamk_tiles, streamk_iters_pcu, streamk_remainder_iters = get_tiles_config(M, N, K, num_sms, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
#    EVEN_K = K % BLOCK_SIZE_K == 0

    for tile_id in range(pid, total_full_tiles, num_sms):
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

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        acc = acc * 0.0
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                a = tl.load(A_BASE, mask=rk[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(B_BASE, mask=rk[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, acc, mask=mask)

    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(pid + 1, streamk_remainder_iters)
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

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
        rk = tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        acc = acc * 0.0
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile
        if start_iter == tile_iter:
            tile_iter_end = tile_iter + iters_per_tile
            next_pid = pid + 1
            end = end_iter
            while (end < tile_iter_end and next_pid < num_sms):
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
             #   while tl.load(locks + next_pid) == 0:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc1 = tl.load(P_)
                acc += acc1
          #      if next_pid < streamk_remainder_iters:
          #          end += streamk_iters_pcu + 1
          #      else:
          #          end += streamk_iters_pcu 
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)

                next_pid += 1

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))%M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))%N
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, acc, mask=mask)
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N +  rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc)
            tl.atomic_xchg(locks + pid, 1)

        start_iter = end_iter
