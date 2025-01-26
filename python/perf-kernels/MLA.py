"""
Multihead Latent Attention
"""

import argparse
import pytest
import sys
import torch

import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def print_gpu(prefix, val=None):
    if (tl.program_id(0) == 0) and ((tl.program_id(1) == 0) and (tl.program_id(2) == 0)):
        if val is not None:
            tl.device_print(prefix, val)
        else:
            tl.device_print(prefix)


# acc, l_i, m_i, q_nope, q_pe, wkv_b, kv_ptrs, k_pe_ptrs, bias_ptrs
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q_pe, kv_ptrs, k_pe_ptrs, kv_offs_k, kv_offs_c, q_nope_wkv_b_ptrs1,
                    stride_q_nope_wkv_b_k, stride_kv_k, stride_kv_n, stride_k_pe_n, actual_seqlen_k, block_min,
                    block_max, offs_n_causal, n_extra_tokens, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K: tl.constexpr, EVEN_K: tl.constexpr,
                    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr, MASK_STEPS: tl.constexpr, QK_SCALE: tl.constexpr):

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None

        # k_pe_offs_k = None if not PADDED_ROPE_HEAD else tl.arange(0, BLOCK_RMODEL)
        k_pe = load_fn(k_pe_ptrs, None, k_offs_n, None, actual_seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # -- compute qk ----
        # normally in FA: qk += (tl.dot(q, k) * QK_SCALE)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            #if EVEN_K:
            q_nope_wkv_b_k = tl.load(q_nope_wkv_b_ptrs1 + k * BLOCK_K * stride_q_nope_wkv_b_k)
            # kv_k = tl.load(kv_ptrs + k * BLOCK_K * stride_kv_k)
            kv_k = load_fn(kv_ptrs + kv_offs_k + k * BLOCK_K * stride_kv_k, None, k_offs_n, None, actual_seqlen_k)
            # else:
            #
            #
            qk += tl.dot(q_nope_wkv_b_k, kv_k)

        qk += tl.dot(q_pe, k_pe)
        qk *= QK_SCALE

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        l_ij = tl.sum(p, 1)

        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        # leave wkv_b gemm outside

        kv = load_fn(kv_ptrs + kv_offs_c, None, k_offs_n, None, actual_seqlen_k)
        acc += tl.dot(p.to(kv.type.element_ty), kv.trans())

        # Or do wkv_b gemm inside
        # for k in range(0, tl.cdiv(K, BLOCK_K)):
        #     #if EVEN_K:
        #     wkv_b_k_2 = tl.load(wkv_b_ptrs2 + k * BLOCK_K * stride_wkv_b_k)
        #     # kv_k = tl.load(kv_ptrs + k * BLOCK_K * stride_kv_k)
        #     kv_k = load_fn(kv_ptrs + kv_offs_k + k * BLOCK_K * stride_kv_k, None, k_offs_n, None, actual_seqlen_k)
        #     # else:
        #     #
        #     #
        #     v_k = tl.dot(wkv_b_k_2, kv_k).trans() # values k subset

        #     acc += tl.dot(p.to(v_k.type.element_ty), v_k)

        kv_ptrs += BLOCK_N * stride_kv_n
        k_pe_ptrs += BLOCK_N * stride_k_pe_n

    return acc, l_i, m_i


@triton.jit
def splitKdot(a, b, acc, M: tl.constexpr, N: tl.constexpr, split_k: tl.constexpr):
    a = a.reshape((M, -1, 2))
    b = b.reshape((N, -1, 2))

    a1, a2 = a.split()
    b1, b2 = b.split()

    if split_k == 0:
        acc += tl.dot(a1, b1) + tl.dot(a2, b2)
        return 0
    else:
        splitKdot(a1, b1, acc, M, N, split_k - 1)
        splitKdot(a2, b2, acc, M, N, split_k - 1)

    return 0


def get_cdna_autotune_configs():
    return [
        # triton.Config(
        #     {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'waves_per_eu': 2, 'GRID_CU_MULTIP':2},
        #     num_stages=1, num_warps=4),
        # triton.Config(
        #     {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'waves_per_eu': 2, 'GRID_CU_MULTIP': 2},
        #     num_stages=1, num_warps=4),
        # triton.Config(
        #     {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'waves_per_eu': 1, 'GRID_CU_MULTIP': 2},
        #     num_stages=1, num_warps=4),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'waves_per_eu': 2, 'GRID_CU_MULTIP': 2},
            num_stages=1, num_warps=4),
    ], [
        'IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'kv_lora_rank', "qk_nope_head_dim",
        "qk_rope_head_dim", "v_head_dim", 'VARLEN', 'HQ', 'HK'
    ]


autotune_configs, autotune_keys = get_cdna_autotune_configs()


@triton.autotune(
    configs=autotune_configs,
    key=autotune_keys,
    use_cuda_graph=True,
)
@triton.heuristics({
    'EVEN_K': lambda args: args['kv_lora_rank'] % args['BLOCK_K'] == 0,
})
@triton.jit
def attn_fwd(X, Q_NOPE, Q_PE, KV, K_PE, 
             WKV_A, WKV_B, WQ, WO,
             GEMM_ACC, SM_SCALE: tl.constexpr, L, Out,
             stride_x_b, stride_x_s, stride_x_dim, # strides for X: bs(dim)
             stride_q_nope_b, stride_q_nope_h, stride_q_nope_s, stride_q_nope_d,  # strides for Q_NOPE: bhsd
             stride_q_pe_b, stride_q_pe_h, stride_q_pe_s, stride_q_pe_r,  # strides for Q_PE: bhsr
             stride_kv_b, stride_kv_t, stride_kv_c,  # strides for KV: btc
             stride_k_pe_b, stride_k_pe_t, stride_k_pe_r,  # strides for K_PE: btr
             stride_o_b, stride_o_s, stride_o_dim,  # strides for O: bhsd
             stride_wkv_a_dim, stride_wkv_a_d,
             stride_wkv_b_h, stride_wkv_b_c, stride_wkv_b_d,  # strides for WKV_B: h(d+d)c
             stride_wq_h, stride_wq_dim, stride_wq_d,
             stride_wo_h, stride_wo_d, stride_wo_dim,
             stride_gemm_acc_b, stride_gemm_acc_h, stride_gemm_acc_s, stride_gemm_acc_c, 
             kv_lora_rank: tl.constexpr, qk_nope_head_dim: tl.constexpr, qk_rope_head_dim: tl.constexpr, v_head_dim: tl.constexpr, dim: tl.constexpr,
             PERSISTENT: tl.constexpr, PERSISTENT_DYNAMIC: tl.constexpr, atomic_counter, NUM_CU: tl.constexpr,
             GRID_CU_MULTIP: tl.constexpr, B: tl.constexpr, HQ: tl.constexpr, HK: tl.constexpr,
             MAX_SEQLENS_Q: tl.constexpr, MAX_SEQLENS_K: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr):

    assert EVEN_K, "Assumes EVEN_K"

    if PERSISTENT:  # if persistent, kernel loops over multiple tiles
        NUM_WG = NUM_CU * GRID_CU_MULTIP  # number of workgroups launched
        num_tiles_per_head = tl.cdiv(MAX_SEQLENS_Q, BLOCK_M)  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * HQ  # times the number of heads
        num_tiles_total = num_tiles_per_sample * B  # times the number of samples
        if PERSISTENT_DYNAMIC:
            tile_id = atomic_counter.atomic_add(1)  # retuns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)
    else:  # standard, kernel processes only one tile
        tile_id = 0
        num_tiles_total = 1

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT

        if PERSISTENT:
            # tile id basically tells us the Q block we are handling
            off_z = tile_id // num_tiles_per_sample  # at which batch sample are we
            off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head  # at which head are we inside the sample
            start_m = tile_id % num_tiles_per_sample % num_tiles_per_head  # at which tile are we inside the head
        else:
            start_m = tl.program_id(0)
            off_h_q = tl.program_id(1)
            off_z = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_q_d = tl.arange(0, qk_nope_head_dim)
        offs_v_d = tl.arange(0, v_head_dim)
        offs_wkv_b_k = tl.arange(0, qk_nope_head_dim)
        offs_wkv_b_v = tl.arange(qk_nope_head_dim, qk_nope_head_dim + v_head_dim)
        offs_c = tl.arange(0, kv_lora_rank)
        offs_r = tl.arange(0, qk_rope_head_dim)
        offs_k = tl.arange(0, BLOCK_K)


        continue_condition = True  # as we can't have return statements inside a while loop in Triton

        cu_seqlens_q_start = 0
        cu_seqlens_kv_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

        if continue_condition:
            # Now we compute whether we need to exit early due to causal masking.
            # This is because for seqlen_q > seqlen_k, M rows of the attn scores
            # are completely masked, resulting in 0s written to the output, and
            # inf written to LSE. We don't need to do any GEMMs in this case.
            # This block of code determines what N is, and if this WG is operating
            # on those M rows.
            n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
            if (IS_CAUSAL):
                # If seqlen_q == seqlen_k, the attn scores are a square matrix.
                # If seqlen_q != seqlen_k, attn scores are rectangular which means
                # the causal mask boundary is bottom right aligned, and ends at either
                # the top edge (seqlen_q < seqlen_k) or left edge.
                # This captures the decrease in n_blocks if we have a rectangular attn matrix
                n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
                # This is what adjusts the block_max for the current WG, only
                # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
                n_blocks = min(n_blocks, n_blocks_seqlen)
                # If we have no blocks after adjusting for seqlen deltas, this WG is part of
                # the blocks that are all 0. We exit early.
                if n_blocks <= 0:
                    o_offset = Out + off_z * stride_ob + off_h_q * stride_oh + cu_seqlens_q_start * stride_os
                    o_ptrs = o_offset + offs_m[:, None] * stride_os + offs_v_d[None, :] * stride_od
                    acc = tl.zeros([BLOCK_M, v_head_dim], dtype=Out.type.element_ty)
                    o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to([BLOCK_M, v_head_dim])
                    # We still need to write 0s to the result
                    tl.store(o_ptrs, acc, mask=o_ptrs_mask)
                    # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
                    # statically known.
                    l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
                    # We store inf to LSE, not -inf because in the bwd pass, we subtract this
                    # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
                    l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
                    l_ptrs_mask = offs_m < MAX_SEQLENS_Q
                    tl.store(l_ptrs, l, mask=l_ptrs_mask)
                    # TODO: Should dropout and return encoded softmax be handled here too?
                    continue_condition = False
                    # return

            if continue_condition:

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N



                # Compute pointers for all the tensors used in this kernel.
                q_offset = Q_NOPE + off_z * stride_q_nope_b + off_h_q * stride_q_nope_h + cu_seqlens_q_start * stride_q_nope_s
                q_ptrs = q_offset + offs_m[:, None] * stride_q_nope_s + offs_q_d[None, :] * stride_q_nope_d

                kv_offset = KV + off_z * stride_kv_b + cu_seqlens_kv_start * stride_kv_t
                kv_ptrs = kv_offset + offs_n[None, :] * stride_kv_t

                kv_offs_k = offs_k[:, None] * stride_kv_c
                kv_offs_c = offs_c[:, None] * stride_kv_c

                # pointers for position embeddings
                k_pe_offset = K_PE + off_z * stride_k_pe_b + cu_seqlens_kv_start * stride_k_pe_r
                k_pe_ptrs = k_pe_offset + offs_r[:, None] * stride_k_pe_r + offs_n[None, :] * stride_k_pe_t

                q_pe_offset = Q_PE + off_z * stride_q_pe_b + off_h_q * stride_q_pe_h
                q_pe_ptrs = q_pe_offset + offs_m[:, None] * stride_q_pe_s + offs_r[None, :] * stride_q_pe_r
                
                
                # load X
                x_offset = X + off_z * stride_x_b
                x_ptrs = x_offset + offs_m[:,None] * stride_x_s + offs_k[None,:] * stride_x_dim

                # project q_nope, q_pe
                wq_offset = WQ + off_h_q * stride_wq_h
                wq_ptrs = wq_offset + offs_k[:, None] * stride_wq_dim

                # Q_NOPE and Q_PE could be done with single WQ_A but since qk_nope_head_dim is like 128 and qk_rope_head_dim is like 64, and tl.load requires pow2 pointer sizes....
                # Q_NOPE
                q_nope = tl.zeros((BLOCK_M, qk_nope_head_dim), dtype=tl.float32)
                offsets_nope = offs_q_d[None, :] * stride_wq_d
                for k in range(0, tl.cdiv(dim, BLOCK_K)):
                    #if EVEN_K:
                    wq_k = tl.load(wq_ptrs + offsets_nope + k * BLOCK_K * stride_wq_dim)
                    x_k = tl.load(x_ptrs + k * BLOCK_K * stride_x_dim)
                    # else:
                    #
                    #
                    q_nope += tl.dot(x_k, wq_k)

                tl.store(q_ptrs, q_nope.to(tl.float32))
                
                # Q_PE
                q_pe = tl.zeros((BLOCK_M, qk_rope_head_dim), dtype=tl.float32)
                offsets_rope = tl.arange(qk_nope_head_dim, qk_nope_head_dim + qk_rope_head_dim)[None,:] * stride_wq_d


                for k in range(0, tl.cdiv(dim, BLOCK_K)):
                    #if EVEN_K:
                    wq_k = tl.load(wq_ptrs + offsets_rope + k * BLOCK_K * stride_wq_dim)
                    x_k = tl.load(x_ptrs + k * BLOCK_K * stride_x_dim)
                    # else:
                    #
                    #
                    q_pe += tl.dot(x_k, wq_k)
 
                tl.store(q_pe_ptrs, q_pe.to(tl.float32))


                # project kv, k_pe These really shouldnt be here since we are not parallelizing among keys, but queries
                # but if seqlen k == seqlen q and causal, we might do something interesting
                # if causal, we only need kv[:wg] (kv[wg:] can be still not computed). wg here is basically the m_i so at which token we are going
                # But I dunno if this projection of x --> kv is that heavy we should start going too complex over

                # wkv_a_offset = WKV_A + off_h_q * stride_wkv_b_h
                # wkv_a_ptrs = wkv_a_offset + offs_k[:, None] * stride_wkv_a_dim + offs_k[None, :] * stride_wkv_a_d                
                

                # k_blocks = tl.cdiv(kv_lora_rank + qk_rope_head_dim, BLOCK_K)
                # for k in range(0, k_blocks):
                #     kv_k = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
                #     for k2 in range(0, tl.cdiv(dim, BLOCK_K)):
                #         #if EVEN_K:
                #         wkv_a_k = tl.load(wkv_a_ptrs + k2 * BLOCK_K * stride_wkv_a_dim)
                #         x_k = tl.load(x_ptrs + k2 * BLOCK_K * stride_x_dim)
                #         # else:
                #         #
                #         #
                #         kv_k += tl.dot(x_k, wkv_a_k)
                #         # TODO: store back to some accumulator tensor in parts
                    
                #     if k_blocks - k > (qk_rope_head_dim // BLOCK_K): # KV
                #         tl.store(kv_ptrs + kv_offs_k + k * BLOCK_K * stride_kv_c, kv_k.trans())
                #     else: # K_PE
                #         tl.store(k_pe_ptrs + (k_blocks - k) * BLOCK_K * stride_k_pe_r, kv_k.trans())

                

                # weight matrix:
                wkv_b_offset = WKV_B + off_h_q * stride_wkv_b_h
                wkv_b_ptrs1 = wkv_b_offset + offs_wkv_b_k[:, None] * stride_wkv_b_d + offs_k[None, :] * stride_wkv_b_c
                wkv_b_ptrs2 = wkv_b_offset + offs_wkv_b_v[None, :] * stride_wkv_b_d + offs_k[:, None] * stride_wkv_b_c

                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, kv_lora_rank], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                q_pe_ptrs_mask = offs_m[:, None] < seqlen_q

                q_nope = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
                q_pe = tl.load(q_pe_ptrs, mask=q_pe_ptrs_mask, other=0.0)

                q_wkv_b_offset = GEMM_ACC + off_z * stride_gemm_acc_b + off_h_q * stride_gemm_acc_h + cu_seqlens_q_start * stride_gemm_acc_s
                q_wkv_b_ptrs = q_wkv_b_offset + offs_m[:,None] * stride_gemm_acc_s + offs_k[None, :] * stride_gemm_acc_c
                
                # wish we could do this... But out of LDS
                # q_wkv_b_ptrs_whole = q_wkv_b_offset + offs_m[:,
                #                                        None] * stride_gemm_acc_s + offs_c[None, :] * stride_gemm_acc_c

                # wkv_b_ptrs1_whole = wkv_b_offset + offs_wkv_b_qk[:, None] * stride_wkv_b_d + offs_k[None, :] * stride_wkv_b_c
                # wkv_b_1 = tl.load(wkv_b_ptrs1_whole)
                # q_nope = tl.dot(q_nope, wkv_b_1)
                
                for k in range(0, tl.cdiv(kv_lora_rank, BLOCK_K)):
                    #if EVEN_K:
                    wkv_b_k_1 = tl.load(wkv_b_ptrs1 + k * BLOCK_K * stride_wkv_b_c)
                    # else:
                    #
                    #
                    q_wkv_b_k = tl.dot(q_nope, wkv_b_k_1).to(q_nope.type.element_ty)
                    # TODO: store back to some accumulator tensor in parts
                    tl.store(q_wkv_b_ptrs + k * BLOCK_K * stride_gemm_acc_c, q_wkv_b_k)

                tl.debug_barrier()
                # Here we compute how many full and masked blocks we have.
                padded_block_k = n_extra_tokens != 0
                is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
                if IS_CAUSAL:
                    # There are always at least BLOCK_M // BLOCK_N masked blocks.
                    # Additionally there might be one more due to dissimilar seqlens.
                    masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
                else:
                    # Padding on Q does not need to be masked in the FA loop.
                    masked_blocks = padded_block_k
                # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
                # In this case we might exceed n_blocks so pick the min.
                masked_blocks = min(masked_blocks, n_blocks)
                n_full_blocks = n_blocks - masked_blocks
                block_min = 0
                block_max = n_blocks * BLOCK_N
                # Compute for full blocks. Here we set causal to false regardless of its actual
                # value because there is no masking. Similarly we do not need padding.

                if n_full_blocks > 0:
                    block_max = (n_blocks - masked_blocks) * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_pe, kv_ptrs, k_pe_ptrs, kv_offs_k, kv_offs_c,
                                                    q_wkv_b_ptrs, stride_gemm_acc_c, stride_kv_c, stride_kv_t,
                                                    stride_k_pe_t, seqlen_k,
                                                    # _, _, offs_n_causal, n_extra_tokens, _
                                                    block_min, block_max, 0, 0,
                                                    # IS_CAUSAL, ....
                                                    False, BLOCK_M, BLOCK_N, BLOCK_K, kv_lora_rank, EVEN_K, offs_m,
                                                    offs_n,
                                                    # MASK_STEPS, ...
                                                    False, QK_SCALE)
                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
                # Remaining blocks, if any, are full / not masked.
                if (masked_blocks > 0):
                    if IS_CAUSAL:
                        offs_n_causal = offs_n + (seqlen_q - seqlen_k)
                    else:
                        offs_n_causal = 0
                    kv_ptrs += n_full_blocks * BLOCK_N * stride_kv_t
                    k_pe_ptrs += n_full_blocks * BLOCK_N * stride_k_pe_t
                    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_pe, kv_ptrs, k_pe_ptrs, kv_offs_k, kv_offs_c,
                                                    q_wkv_b_ptrs, stride_gemm_acc_c, stride_kv_c, stride_kv_t,
                                                    stride_k_pe_t, seqlen_k, block_min, block_max, offs_n_causal,
                                                    n_extra_tokens, IS_CAUSAL, BLOCK_M, BLOCK_N, BLOCK_K, kv_lora_rank,
                                                    EVEN_K, offs_m, offs_n,
                                                    # MASK_STEPS, ...
                                                    True, QK_SCALE)

                # epilogue
                # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
                l_recip = 1 / l_i[:, None]
                acc = acc * l_recip
            
                acc = acc.to(q_nope.type.element_ty)

                # we can reuse the q_nope_wkv_b tensor from before to store the acc
                acc_ptrs = q_wkv_b_offset + offs_m[:, None] * stride_gemm_acc_s + offs_c[None, :] * stride_gemm_acc_c
                tl.store(acc_ptrs, acc)

                tl.debug_barrier()

                acc = tl.zeros([BLOCK_M, v_head_dim], dtype=tl.float32)

                acc_k_ptrs = q_wkv_b_offset + offs_m[:, None] * stride_gemm_acc_s + offs_k[None, :] * stride_gemm_acc_c

                for k in range(0, tl.cdiv(kv_lora_rank, BLOCK_K)):
                    #if EVEN_K:
                    wkv_b_k_2 = tl.load(wkv_b_ptrs2 + k * BLOCK_K * stride_wkv_b_c)
                    acc_k = tl.load(acc_k_ptrs + k * BLOCK_K * stride_gemm_acc_c)
                    # else:
                    #
                    #
                    acc += tl.dot(acc_k, wkv_b_k_2)
                
                # write back O
                o_offset = Out + off_z * stride_o_b 
                o_ptrs = o_offset + offs_m[:, None] * stride_o_s + offs_k[None, :] * stride_o_dim
                
                wo_offset = WO + off_h_q * stride_wo_h
                wo_ptrs = wo_offset + offs_v_d[:, None] * stride_wo_d + offs_k[None, :] * stride_wo_dim
                
                # so basically o has shape bxsx(dim)
                # it would be a result of attn_output.flatten(hd) (bxhxsxd->bxsx(hd))  * wo ((hd)x(dim))
                # Naively this would be unfusable to this kernel as we dont have values from other heads here
                # But my idea is that we can calculate this head's contribution and add it as a atomic add
                # This can however be a bottleneck as we cant continue calculating the other attention_outputs
                # if it ends up hanging on the atomic add
                                
                for k in range(0, tl.cdiv(dim, BLOCK_K)):
                    wo_k = tl.load(wo_ptrs + k * BLOCK_K * stride_wo_dim)
                    # o = tl.load(o_ptrs + k * BLOCK_K * stride_o_dim)
                    o = tl.dot(acc, wo_k).to(q_nope.type.element_ty)
                    # tl.store(o_ptrs + k * BLOCK_K * stride_o_dim, o) 
                    tl.atomic_add(o_ptrs + k * BLOCK_K * stride_o_dim, o)

                # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
                # then we have one block with a row of all NaNs which come from computing
                # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
                # and store 0s where there are NaNs as these rows should've been zeroed out.
                end_m_idx = (start_m + 1) * BLOCK_M
                start_m_idx = start_m * BLOCK_M
                causal_start_idx = seqlen_q - seqlen_k
                acc = acc.to(Out.type.element_ty)
                if IS_CAUSAL:
                    if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
                        out_mask_boundary = tl.full((v_head_dim, ), causal_start_idx, dtype=tl.int32)
                        mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
                        out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
                        z = 0.0
                        acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
                # write back LSE
                l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
                # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
                # This is only true for the last M block. For others, overflow_size will be -ve
                overflow_size = end_m_idx - seqlen_q

                assert overflow_size <= 0, "No overflow allowed"

                if overflow_size > 0:
                    boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
                    l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                else:
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i))

                

                # tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = atomic_counter.atomic_add(1)
            else:
                tile_id += NUM_WG
        else:
            tile_id = num_tiles_total  # break after single tile


def get_shape_from_layout(q_nope, q_pe, kv, k_pe, wkv_b, layout):
    if layout == 'bhsd':
        kv_lora_rank = wkv_b.shape[-1]
        qk_nope_head_dim = q_nope.shape[-1]
        qk_rope_head_dim = q_pe.shape[-1]
        v_head_dim = wkv_b.shape[-2] - qk_nope_head_dim
        batch, nheads_q, _, head_size = q_nope.shape
        nheads_k = kv.shape[1]
    else:
        assert False, "Got unsupported layout."
    return batch, nheads_q, nheads_k, qk_nope_head_dim, v_head_dim, kv_lora_rank, qk_rope_head_dim


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout(x, q, q_pe, kv, k_pe, o, wkv_a, wkv_b, wq, wo, layout):
    if layout == 'bhsd':
        x_strides = (x.stride(0), x.stride(1), x.stride(2))
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        q_pe_strides = (q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), q_pe.stride(3))
        kv_strides = (kv.stride(0), kv.stride(1), kv.stride(2))
        k_pe_strides = (k_pe.stride(0), k_pe.stride(1), k_pe.stride(2))
        # o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2))
        wkv_a_strides = (wkv_a.stride(0), wkv_a.stride(1))
        wkv_b_strides = (wkv_b.stride(0), wkv_b.stride(1), wkv_b.stride(2))
        wq_strides = (wq.stride(0), wq.stride(1), wq.stride(2))
        wo_strides = (wo.stride(0), wo.stride(1), wo.stride(2))
    else:
        assert False, 'Got unsupported layout.'

    return x_strides, q_strides, q_pe_strides, kv_strides, k_pe_strides, o_strides, wkv_a_strides, wkv_b_strides, wq_strides, wo_strides


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, wkv_a, wkv_b, wq, wo, qk_nope_head_dim, v_head_dim, kv_lora_rank, qk_rope_head_dim, nheads, causal=False, sm_scale=1.0, layout="bhsd"):

        
        batch, seqlen_q, dim = x.shape
        nheads_q = nheads
        nheads_k = nheads
        seqlen_k = seqlen_q
        
        # q = torch.einsum("hzd,bsz->bhsd", wq, x)

        # q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

        kv = torch.einsum("zc,btz->btc", wkv_a, x)

        kv, k_pe = torch.split(kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        
        # o = torch.empty_like(q_nope)

        # assert layout == "bhsd"
        # batch, nheads_q, nheads_k, qk_nope_head_dim, v_head_dim, kv_lora_rank, qk_rope_head_dim = get_shape_from_layout(
        #     q_nope, q_pe, kv, k_pe, wkv_b, layout)

        # q_nope_strides, q_pe_strides, kv_strides, k_pe_strides, o_strides, wkv_b_strides = get_strides_from_layout(
        #     q_nope, q_pe, kv, k_pe, o, wkv_b, layout)

    

        # seqlen_q = q_nope.shape[-2]
        # seqlen_k = kv.shape[1]

        assert seqlen_q == seqlen_k, "varlen not implemented"

        M = torch.empty((batch, nheads_q, seqlen_q), device=x.device, dtype=torch.float32)

        # number of compute units available
        NUM_CU = torch.cuda.get_device_properties("cuda").multi_processor_count

        persistent = causal

        if persistent:
            grid = lambda META: (min(NUM_CU * META['GRID_CU_MULTIP'],
                                     triton.cdiv(seqlen_q, META['BLOCK_M']) * nheads_q * batch), )
        else:
            grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), nheads_q, batch)

        atomic_counter = torch.zeros([1], device=x.device, dtype=torch.int32)

        # accumulation matrix for the gemms. Allows loading and storing in parts
        # q_nope * wkv_b -> gemm_acc
        # acc -> gemm_acc
        gemm_acc = torch.empty((batch, nheads_q, seqlen_q, kv_lora_rank), dtype=x.dtype, device=x.device)
        gemm_acc_strides = (gemm_acc.stride(0), gemm_acc.stride(1), gemm_acc.stride(2), gemm_acc.stride(3))

        q_nope = torch.empty((batch, nheads_q, seqlen_q, qk_nope_head_dim), dtype=x.dtype, device=x.device)
        q_pe = torch.empty((batch, nheads_q, seqlen_q, qk_rope_head_dim), dtype=x.dtype, device=x.device)
        # kv = torch.empty((batch, seqlen_k, kv_lora_rank), dtype=x.dtype, device=x.device)
        # k_pe = torch.empty((batch, seqlen_k, qk_rope_head_dim), dtype=x.dtype, device=x.device)

        o = torch.empty_like(x)
        x_strides, q_nope_strides, q_pe_strides, kv_strides, k_pe_strides, o_strides, wkv_a_strides, wkv_b_strides, wq_strides, wo_strides = get_strides_from_layout(x,
            q_nope, q_pe, kv, k_pe, o, wkv_a, wkv_b, wq, wo, layout)

        attn_fwd[grid](x, q_nope, q_pe, kv, k_pe, wkv_a, wkv_b, wq, wo, gemm_acc, sm_scale, M, o, *x_strides, *q_nope_strides, *q_pe_strides,
                       *kv_strides, *k_pe_strides, *o_strides, *wkv_a_strides, *wkv_b_strides, *wq_strides, *wo_strides, *gemm_acc_strides, kv_lora_rank,
                       qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dim, HQ=nheads_q, HK=nheads_k, MAX_SEQLENS_Q=seqlen_q,
                       MAX_SEQLENS_K=seqlen_k, IS_CAUSAL=causal, PERSISTENT=causal, PERSISTENT_DYNAMIC=causal,
                       NUM_CU=NUM_CU, atomic_counter=atomic_counter, B=batch)

        # o = torch.einsum("hdz,bhsd->bsz", wo, o)

        return o, None, attn_fwd.best_config


attention = _attention.apply


def input_helper_MLA(dim, H, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dtype, layout,
                     requires_grad=False):
    torch.manual_seed(20)

    wkv_a_tensor_shape = (dim, kv_lora_rank + qk_rope_head_dim)
    wkv_b_tensor_shape = (H, kv_lora_rank, qk_nope_head_dim + v_head_dim)

    wq_tensor_shape = (H, dim, qk_nope_head_dim + qk_rope_head_dim)
    wo_tensor_shape = (H, v_head_dim, dim)

    wkv_a = torch.randn(wkv_a_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    wkv_b = torch.randn(wkv_b_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    wq = torch.randn(wq_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    wo = torch.randn(wo_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)

    sm_scale = qk_nope_head_dim**-0.5

    return  wkv_a, wkv_b, wq, wo, sm_scale



@pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim', [
    (8, 16, 128, 2048, 512, 128, 64, 128),
])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('layout', ['bhsd'])
@pytest.mark.parametrize('ref_impl', ['naive', 'absorb'])
def test_op_fwd(B, H, S, dim, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,  causal, layout,
                dtype=torch.float32, ref_impl="naive"):
    import time
    torch.manual_seed(20)
    
    x = torch.randn(B, S, dim, dtype=dtype, device="cuda", requires_grad=False)

    wkv_a, wkv_b, wq, wo, sm_scale = input_helper_MLA(dim, H, kv_lora_rank, qk_nope_head_dim,
                                                                  qk_rope_head_dim, v_head_dim, dtype, layout)

    # warmup (let autotuning happen)
    
    attention(x, wkv_a, wkv_b, wq, wo, qk_nope_head_dim, v_head_dim, kv_lora_rank, qk_rope_head_dim, H, causal, sm_scale)

    # triton implementation
    torch.cuda.synchronize()
    start = time.time()
    tri_out, _, _ = attention(x, wkv_a, wkv_b, wq, wo, qk_nope_head_dim, v_head_dim, kv_lora_rank, qk_rope_head_dim, H, causal, sm_scale)
    torch.cuda.synchronize()
    print(f"time for triton: {time.time()-start}")


    torch.cuda.synchronize()
    start = time.time()
    # ref implementation
    q = torch.einsum("hzd,bsz->bhsd", wq, x)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    kv = torch.einsum("zc,btz->btc", wkv_a, x)
    kv, k_pe = torch.split(kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    
    if ref_impl == "naive":
        q = torch.cat([q_nope, q_pe], dim=-1)
        # kv = self.wkv_b(self.kv_norm(kv))
        # wkv_b = wkv_b.view(H, -1, kv_lora_rank)
        kv = torch.einsum("hcd,btc->bhtd", wkv_b, kv)
        kv = kv.view(B, H, S, qk_nope_head_dim + v_head_dim)
        k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, H, -1, -1)], dim=-1)
        scores = torch.einsum("bhsd,bhtd->bhst", q, k) * sm_scale
    else:
        # wkv_b = wkv_b.view(H, -1, kv_lora_rank)
        q_nope = torch.einsum("bhsd,hcd->bhsc", q_nope, wkv_b[:, :,:qk_nope_head_dim])
        scores = (torch.einsum("bhsc,btc->bhst", q_nope, kv) +
                  torch.einsum("bhsr,btr->bhst", q_pe, k_pe.squeeze(1))) * sm_scale

    if causal:
        mask = torch.full((S, S), float("-inf"), device=q_nope.device).triu_(1)
        scores += mask.unsqueeze(0)

    scores = scores.softmax(dim=-1).type_as(q_nope)

    if ref_impl == "naive":
        x = torch.einsum("bhst,bhtd->bhsd", scores, v)
    else:
        x = torch.einsum("bhst,btc->bhsc", scores, kv)
        x = torch.einsum("bhsc,hcd->bhsd", x, wkv_b[:, :, -v_head_dim:])


    ref_out = torch.einsum("hdz,bhsd->bsz", wo, x)

    tri_out *= 0.5

    torch.cuda.synchronize()
    print(f"time for ref: {time.time()-start}")
    print(f"ref out type {ref_out.dtype}")
    print(f"tri out type {tri_out.dtype}")

    print(f"ref_out (first 10): {ref_out.flatten()[:10]}")
    print(f"tri_out (first 10): {tri_out.flatten()[:10]}")

    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


def supported_layouts():
    layouts = \
        'bhsd: Q, K, V are individual tensors of [batch, num_heads, seqlen_q/k, head_size]' \
        'bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]' \
        'thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]' \
        'This layout is sometimes called "varlen" or "grouped" layout.'
    return layouts


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument("-equal_seqlens", action='store_true', default=False,
                        help='If specified, each context within the thd layout' \
                            ' has same seqlen as sq and sk')
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", action='store_true', default=False)
    parser.add_argument("-int8", action='store_true', default=False)
    parser.add_argument("-quantize_p", action='store_true', default=False)
    parser.add_argument("-int8_kv", action='store_true', default=False)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-return_time", action='store_true', default=False)
    parser.add_argument("-layout", type=str, default='bhsd', help=supported_layouts())
    parser.add_argument(
        "-persistent", nargs='?', const='fixed', choices=['fixed', 'dynamic'], default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.")
    return parser.parse_args()


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def main():
    # args = parse_args()
    test_op_fwd(8, 16, 128, 2048, 512, 128, 64, 128, causal=False, layout="bhsd", dtype=torch.float32, ref_impl="absorb")


if __name__ == '__main__':
    sys.exit(main())
