"""
Multihead Latent Attention
"""

import argparse
import subprocess
import pytest
import sys
import torch

import triton
import triton.language as tl


class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    persistent = None
    num_contexts = 0
    varlen = False
    int8 = False
    layout = None
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def set_persistent(self, persistent):
        self.persistent = persistent

    def set_int8_params(self, q_descale, k_descale, v_descale, p_scale, p_descale):
        self.int8 = True
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale
        self.p_scale = p_scale
        self.p_descale = p_descale
        self.use_p_scale = (p_scale is not None) and (p_descale is not None) and (v_descale is not None)
        self.int8_kv = (q_descale is None) and (k_descale is not None) and (v_descale is not None)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax


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
def _attn_fwd_inner(acc, l_i, m_i, q_nope, q_pe, kv_ptrs, k_pe_ptrs, bias_ptrs, 
                    wkv_b_ptrs1, wkv_b_ptrs2, stride_kv_k, stride_wkv_b_k,
                    stride_kv_n, stride_k_pe_n, stride_bn, start_m,
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, q_descale,
                    k_descale, v_descale, p_scale, IS_CAUSAL: tl.constexpr, 
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K: tl.constexpr, EVEN_K: tl.constexpr,
                    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, 
                    QK_SCALE: tl.constexpr, INT8_GEMM: tl.constexpr,
                    USE_P_SCALE: tl.constexpr, INT8_KV: tl.constexpr, LATENT_ATTENTION: tl.constexpr):

    assert LATENT_ATTENTION, "This kernel is only planned to work with the latent attention"
    
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
            wkv_b_k_1 = tl.load(wkv_b_ptrs1 + k * BLOCK_K * stride_wkv_b_k)
            # kv_k = tl.load(kv_ptrs + k * BLOCK_K * stride_kv_k)
            kv_k = load_fn(kv_ptrs + k * BLOCK_K * stride_kv_k, None, k_offs_n, None, actual_seqlen_k)
            # else:
            #     
            #     
            k_k = tl.dot(wkv_b_k_1, kv_k) # keys subset
          
            qk += tl.dot(q_nope.to(k_k.type.element_ty), k_k)
            

        qk += tl.dot(q_pe, k_pe)
        qk *= QK_SCALE

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_sm_ptrs, tl.where(keep, p, -p).to(encoded_sm_ptrs.type.element_ty))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_sm_ptrs, p.to(encoded_sm_ptrs.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            #if EVEN_K:
            wkv_b_k_2 = tl.load(wkv_b_ptrs2 + k * BLOCK_K * stride_wkv_b_k)
            # kv_k = tl.load(kv_ptrs + k * BLOCK_K * stride_kv_k)
            kv_k = load_fn(kv_ptrs + k * BLOCK_K * stride_kv_k, None, k_offs_n, None, actual_seqlen_k)
            # else:
            #     
            # 
            v_k = tl.dot(wkv_b_k_2, kv_k).trans() # values k subset
            
            acc += tl.dot(p.to(v_k.type.element_ty), v_k)

        kv_ptrs += BLOCK_N * stride_kv_n
        k_pe_ptrs += BLOCK_N * stride_k_pe_n
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N


    return acc, l_i, m_i


def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K',
        'kv_lora_rank', "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        'VARLEN', 'HQ', 'HK']



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
def attn_fwd(Q_NOPE, Q_PE, KV, K_PE, WKV_B, bias, SM_SCALE: tl.constexpr, L, Out,
            stride_q_nope_b, stride_q_nope_h, stride_q_nope_s, stride_q_nope_d,  # strides for Q_NOPE: bhsd
            stride_q_pe_b, stride_q_pe_h, stride_q_pe_s, stride_q_pe_r,  # strides for Q_PE: bhsr
            stride_kv_b, stride_kv_t, stride_kv_c,  # strides for KV: btc
            stride_k_pe_b, stride_k_pe_t, stride_k_pe_r,  # strides for K_PE: btr
            stride_ob, stride_oh, stride_os, stride_od, # strides for O: bhsd
            stride_wkv_b_h, stride_wkv_b_d, stride_wkv_b_c, # strides for WKV_B: h(d+d)c  
            stride_bz, stride_bh, stride_bm, stride_bn, 
            stride_az, stride_ah, 
            kv_lora_rank: tl.constexpr, qk_nope_head_dim: tl.constexpr, qk_rope_head_dim: tl.constexpr, v_head_dim: tl.constexpr,
            Q_descale, K_descale, P_scale, P_descale, V_descale, cu_seqlens_q, cu_seqlens_k,
            dropout_p, philox_seed, PERSISTENT: tl.constexpr, PERSISTENT_DYNAMIC: tl.constexpr, atomic_counter,
            NUM_CU: tl.constexpr, GRID_CU_MULTIP: tl.constexpr, B: tl.constexpr, philox_offset_base, encoded_softmax,
            alibi_slopes, HQ: tl.constexpr, HK: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr, MAX_SEQLENS_K: tl.constexpr,
            VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, 
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
            PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
            RETURN_ENCODED_SOFTMAX: tl.constexpr, USE_ALIBI: tl.constexpr, INT8: tl.constexpr,
            USE_P_SCALE: tl.constexpr, INT8_KV: tl.constexpr, LATENT_ATTENTION: tl.constexpr):

    assert LATENT_ATTENTION, "This kernel is meant to work with latent attention only"
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
        offs_wkv_b_qk = tl.arange(0, qk_nope_head_dim)
        offs_wkv_b_v = tl.arange(qk_nope_head_dim, qk_nope_head_dim + v_head_dim)
        # offs_c = tl.arange(0, kv_lora_rank)
        offs_r = tl.arange(0, qk_rope_head_dim)
        offs_k = tl.arange(0, BLOCK_K)

        continue_condition = True  # as we can't have return statements inside a while loop in Triton

        if VARLEN:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0). Some seqlens might be too
            # small for all start_m so for those we return early.
            if start_m * BLOCK_M > seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_kv_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_kv_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_kv_end - cu_seqlens_kv_start
        else:
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
                    o_ptrs = o_offset + offs_m[:, None] * stride_os + offs_q_d[None, :] * stride_od
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
                # If MQA / GQA, set the K and V head offsets appropriately.
                GROUP_SIZE: tl.constexpr = HQ // HK
                if GROUP_SIZE != 1:
                    off_h_k = off_h_q // GROUP_SIZE
                else:
                    off_h_k = off_h_q

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N


                # Compute pointers for all the tensors used in this kernel.
                q_offset = Q_NOPE + off_z * stride_q_nope_b + off_h_q * stride_q_nope_h + cu_seqlens_q_start * stride_q_nope_s
                q_ptrs = q_offset + offs_m[:, None] * stride_q_nope_s + offs_q_d[None, :] * stride_q_nope_d

                kv_offset = KV + off_z * stride_kv_b + cu_seqlens_kv_start * stride_kv_t
                kv_ptrs = kv_offset + offs_k[:, None] * stride_kv_c + offs_n[None, :] * stride_kv_t
                
                # pointers for position embeddings
                k_pe_offset = K_PE + off_z * stride_k_pe_b + cu_seqlens_kv_start * stride_k_pe_r
                k_pe_ptrs = k_pe_offset + offs_r[:, None] * stride_k_pe_r + offs_n[None, :] * stride_k_pe_t
                
                q_pe_offset = Q_PE + off_z * stride_q_pe_b + off_h_q * stride_q_pe_h
                q_pe_ptrs = q_pe_offset + offs_m[:, None] * stride_q_pe_s + offs_r[None, :] * stride_q_pe_r
                
                # weight matrix:                
                wkv_b_offset = WKV_B + off_h_q * stride_wkv_b_h 
                wkv_b_ptrs1 = wkv_b_offset + offs_wkv_b_qk[:, None] * stride_wkv_b_d + offs_k[None, :] * stride_wkv_b_c
                wkv_b_ptrs2 = wkv_b_offset + offs_wkv_b_v[:, None] * stride_wkv_b_d + offs_k[None, :] * stride_wkv_b_c

                # Compute pointers for all the scale tensors used in this kernel.
                INT8_GEMM: tl.constexpr = INT8 & (not INT8_KV)
                if INT8:
                    k_descale_ptrs = K_descale + off_h_k
                    v_descale_ptrs = V_descale + off_h_k
                    if not INT8_KV:
                        q_descale_ptrs = Q_descale + off_h_q
                    if USE_P_SCALE:
                        p_scale_ptrs = P_scale + off_h_q
                        p_descale_ptrs = P_descale + off_h_q

                if USE_BIAS:
                    # Note: this might get large enough to overflow on some configs
                    bias_offset = off_h_q * stride_bh
                    bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
                else:
                    bias_ptrs = None

                if USE_ALIBI:
                    a_offset = off_z * stride_az + off_h_q * stride_ah
                    alibi_slope = tl.load(alibi_slopes + a_offset)
                else:
                    alibi_slope = None

                if ENABLE_DROPOUT:
                    off_hz = off_z * HQ + off_h_q
                    batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
                else:
                    batch_philox_offset = 0
                # We can ask to return the dropout mask without actually doing any dropout. In
                # this case, we return an invalid pointer so indicate the mask is not valid.
                if RETURN_ENCODED_SOFTMAX:
                    encoded_sm_base = encoded_softmax + off_h_q * seqlen_q * seqlen_k
                    encoded_sm_ptrs = encoded_sm_base + offs_m[:, None] * seqlen_k + offs_n[None, :]
                else:
                    encoded_sm_ptrs = None
                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, v_head_dim], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                q_pe_ptrs_mask = offs_m[:, None] < seqlen_q

                q_nope = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
                q_pe = tl.load(q_pe_ptrs, mask=q_pe_ptrs_mask, other=0.0)
        

                if INT8:
                    k_descale = tl.load(k_descale_ptrs)
                    v_descale = tl.load(v_descale_ptrs)
                    if not INT8_KV:
                        q_descale = tl.load(q_descale_ptrs)
                    else:
                        q_descale = None
                    if USE_P_SCALE:
                        p_scale = tl.load(p_scale_ptrs)
                        p_descale = tl.load(p_descale_ptrs)
                    else:
                        p_scale = None
                        p_descale = None
                else:
                    q_descale = None
                    k_descale = None
                    v_descale = None
                    p_scale = None
                    p_descale = None
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
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc, l_i, m_i, q_nope, q_pe, kv_ptrs, k_pe_ptrs, bias_ptrs, 
                        wkv_b_ptrs1, wkv_b_ptrs2, stride_kv_c, stride_wkv_b_c,
                        stride_kv_t, stride_k_pe_t, 
                        stride_bn, start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs,
                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                        block_min, block_max, 0, 0, 0, alibi_slope, q_descale, k_descale, v_descale, p_scale,
                        # IS_CAUSAL, ....
                        False, BLOCK_M, BLOCK_N,
                        BLOCK_K, kv_lora_rank, EVEN_K,
                        offs_m, offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, QK_SCALE, INT8_GEMM, USE_P_SCALE, INT8_KV,
                        True)
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
                    if USE_BIAS:
                        bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                    if RETURN_ENCODED_SOFTMAX:
                        encoded_sm_ptrs += n_full_blocks * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc, l_i, m_i, q_nope, q_pe, kv_ptrs, k_pe_ptrs, bias_ptrs, 
                        wkv_b_ptrs1, wkv_b_ptrs2, stride_kv_c, stride_wkv_b_c,
                        stride_kv_t, stride_k_pe_t, 
                        stride_bn, start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs,
                        block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, q_descale,
                        k_descale, v_descale, p_scale, IS_CAUSAL, BLOCK_M, BLOCK_N,
                        BLOCK_K, kv_lora_rank, EVEN_K,
                        offs_m, offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, QK_SCALE, INT8_GEMM, USE_P_SCALE, INT8_KV,
                        True)

                if INT8 and not INT8_KV:
                    if USE_P_SCALE:
                        acc *= p_descale
                    acc *= v_descale
                
                # epilogue
                # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
                l_recip = 1 / l_i[:, None]
                acc = acc * l_recip

                if ENABLE_DROPOUT:
                    acc = acc / (1 - dropout_p)
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
                if overflow_size > 0:
                    boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
                    l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                else:
                    tl.store(l_ptrs, m_i + tl.math.log2(l_i))

                # write back O
                o_offset = Out + off_z * stride_ob + off_h_q * stride_oh + cu_seqlens_q_start * stride_os
                o_ptrs = o_offset + offs_m[:, None] * stride_os + offs_v_d[None, :] * stride_od
                o_ptrs_mask = tl.full([BLOCK_M, v_head_dim], 1, dtype=tl.int1)

                if overflow_size > 0:
                    o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
                
                tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = atomic_counter.atomic_add(1)
            else:
                tile_id += NUM_WG
        else:
            tile_id = num_tiles_total  # break after single tile



def get_shape_from_layout(q, k, metadata):
    if metadata.layout == 'thd':
        nheads_q, nheads_k = q.shape[1], k.shape[1]
        head_size = q.shape[-1]
        batch = metadata.num_contexts
    elif metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    else:
        assert False, "Got unsupported layout."
    return batch, nheads_q, nheads_k, head_size


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout(q, k, v, o, metadata):
    if metadata.layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif metadata.layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif metadata.layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout_latent(q, q_pe, kv, k_pe, o, wkv_b, metadata):
    if metadata.layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        q_pe_strides = (q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), q_pe.stride(3))
        kv_strides = (kv.stride(0), kv.stride(1), kv.stride(2))
        k_pe_strides = (k_pe.stride(0), k_pe.stride(1), k_pe.stride(2))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
        wkv_b_strides = (wkv_b.stride(0), wkv_b.stride(1), wkv_b.stride(2))
    else:
        assert False, 'Got unsupported layout.'

    return q_strides, q_pe_strides, kv_strides, k_pe_strides, o_strides, wkv_b_strides


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_nope, q_pe, kv, k_pe, o, wkv_b, metadata: MetaData):
        # NOTE: a large bias tensor leads to overflow during pointer arithmetic
        if (metadata.bias is not None):
            assert (metadata.bias.numel() < 2**31)

        if o is None:
            if not metadata.int8:
                o = torch.empty_like(q_nope, dtype=kv.dtype)
            else:
                o = torch.empty_like(q_nope, dtype=torch.float16)

        assert metadata.layout=="bhsd"
        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q_nope, kv, metadata)

        q_nope_strides, q_pe_strides, kv_strides, k_pe_strides, o_strides, wkv_b_strides = get_strides_from_layout_latent(
            q_nope, q_pe, kv, k_pe, o, wkv_b, metadata)

        kv_lora_rank = wkv_b.shape[-1]
        qk_nope_head_dim = q_nope.shape[-1]
        qk_rope_head_dim = q_pe.shape[-1] 
        v_head_dim = wkv_b.shape[-2] - qk_nope_head_dim


        M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q_nope.device, dtype=torch.float32)

        # Seed the RNG so we get reproducible results for testing.
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        assert metadata.bias is None
        bias_strides = (0, 0, 0, 0)

        assert metadata.alibi_slopes is None
        alibi_strides = (0, 0)

        assert metadata.dropout_p <= 0.0


        q_descale = k_descale = p_scale = p_descale = v_descale = None

        # number of compute units available
        NUM_CU = torch.cuda.get_device_properties("cuda").multi_processor_count

        if metadata.persistent is not None:
            grid = lambda META: (min(NUM_CU * META['GRID_CU_MULTIP'],
                                     triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']) * nheads_q * batch), )
        else:
            grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), nheads_q, batch)

        atomic_counter = torch.zeros([1], device=q_nope.device, dtype=torch.int32)

        attn_fwd[grid](
            q_nope, q_pe, kv, k_pe, wkv_b, metadata.bias, metadata.sm_scale, M, o, *q_nope_strides, *q_pe_strides, *kv_strides,
            *k_pe_strides, *o_strides, *wkv_b_strides, *bias_strides, *alibi_strides, 
            kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            q_descale, k_descale, p_scale, p_descale, v_descale, metadata.cu_seqlens_q,
            metadata.cu_seqlens_k, dropout_p=metadata.dropout_p, philox_seed=philox_seed,
            philox_offset_base=philox_offset, encoded_softmax=None, alibi_slopes=metadata.alibi_slopes,
            HQ=nheads_q, HK=nheads_k, MAX_SEQLENS_Q=metadata.max_seqlens_q, MAX_SEQLENS_K=metadata.max_seqlens_k,
            IS_CAUSAL=metadata.causal, VARLEN=metadata.varlen,
            USE_BIAS=False if metadata.bias is None else True,
            USE_ALIBI=False if metadata.alibi_slopes is None else True, ENABLE_DROPOUT=metadata.dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax, INT8=metadata.int8, USE_P_SCALE=metadata.int8
            and metadata.use_p_scale, INT8_KV=metadata.int8 and metadata.int8_kv, PERSISTENT=metadata.persistent
            is not None, PERSISTENT_DYNAMIC=metadata.persistent == "dynamic", NUM_CU=NUM_CU, LATENT_ATTENTION=True,
            atomic_counter=atomic_counter, B=batch)

        return o, None, attn_fwd.best_config


attention = _attention.apply

def input_helper_MLA(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dtype, layout, requires_grad=False):
    torch.manual_seed(20)

    q_nope_tensor_shape = (B, H, S, qk_nope_head_dim)
    q_pe_tensor_shape = (B, H, S, qk_rope_head_dim)
    kv_tensor_shape = (B, S, kv_lora_rank)
    k_pe_tensor_shape = (B, 1, S, qk_rope_head_dim)
    v_tensor_shape = (B, H, S, v_head_dim)

    wkv_b_tensor_shape = (H, qk_nope_head_dim + v_head_dim, kv_lora_rank)

    q_nope = torch.randn(q_nope_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    q_pe = torch.randn(q_pe_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    kv = torch.randn(kv_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    k_pe = torch.randn(k_pe_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    v = torch.randn(v_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)

    wkv_b = torch.randn(wkv_b_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad)

    sm_scale = qk_nope_head_dim**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = S
    input_metadata.max_seqlens_k = S
    input_metadata.layout = layout
    return q_nope, q_pe, kv, k_pe, v, wkv_b, input_metadata


def sanity_check(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, causal, use_alibi, layout, dtype=torch.float32):
    torch.manual_seed(20)
    q_nope, q_pe, kv, k_pe, v, wkv_b, input_metadata = input_helper_MLA(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dtype, layout)
    
    for x, x_name in zip([q_nope, q_pe, kv, k_pe, v, wkv_b], ["q_nope", "q_pe", "kv", "k_pe", "v", "wkv_b"]):
        print(f"{x_name}: {x.shape}")

    # naive implementation    
    q = torch.cat([q_nope, q_pe], dim=-1)
    kv_proj = torch.einsum("hdc,btc->bhtd", wkv_b, kv)
    kv_proj = kv_proj.view(B, H, S, qk_nope_head_dim + v_head_dim)
    k_nope, v = torch.split(kv_proj, [qk_nope_head_dim, v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)
    scores = torch.einsum("bhsd,bhtd->bhst", q, k) # * input_metadata.sm_scale
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(q_nope)
    naive_out = torch.einsum("bhst,bhtd->bhsd", scores, v)
    print(f"naive_out (first 10): {naive_out.flatten()[:10]}")

    # absorb implementation
    wkv_b = wkv_b.view(H, -1, kv_lora_rank)
    q_nope = torch.einsum("bhsd,hdc->bhsc", q_nope, wkv_b[:, :qk_nope_head_dim])
    scores = (torch.einsum("bhsc,btc->bhst", q_nope, kv) +
                torch.einsum("bhsr,btr->bhst", q_pe, k_pe.squeeze(1)))  # * input_metadata.sm_scale
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(q_nope)
    x = torch.einsum("bhst,btc->bhsc", scores, kv)
    absorb_out = torch.einsum("bhsc,hdc->bhsd", x, wkv_b[:, -v_head_dim:])
    print(f"absorb_out (first 10): {absorb_out.flatten()[:10]}")

    torch.testing.assert_close(naive_out, absorb_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim', [
    (8, 16, 128, 512, 128, 64, 128),
])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('use_alibi', [False])
@pytest.mark.parametrize('layout', ['bhsd'])
@pytest.mark.parametrize('ref_impl', ['naive', 'absorb'])
def test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, causal, use_alibi, layout, dtype=torch.float32, ref_impl="naive"):
    import time
    torch.manual_seed(20)
    q_nope, q_pe, kv, k_pe, v, wkv_b, input_metadata = input_helper_MLA(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dtype, layout)
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q_nope)

    for x, x_name in zip([q_nope, q_pe, kv, k_pe, v, wkv_b], ["q_nope", "q_pe", "kv", "k_pe", "v", "wkv_b"]):
        print(f"{x_name}: {x.shape}")

    # warmup (let autotuning happen)
    attention(q_nope, q_pe, kv, k_pe.squeeze(1), o, wkv_b, input_metadata)

    # triton implementation
    torch.cuda.synchronize()
    start = time.time()
    tri_out, _, _ = attention(q_nope, q_pe, kv, k_pe.squeeze(1), o, wkv_b, input_metadata)
    torch.cuda.synchronize()
    print(f"time for triton: {time.time()-start}")

    torch.cuda.synchronize()
    start = time.time()
    # ref implementation    
    if ref_impl=="naive":
        q = torch.cat([q_nope, q_pe], dim=-1)
        # kv = self.wkv_b(self.kv_norm(kv))
        kv = torch.einsum("hdc,btc->bhtd", wkv_b, kv)
        kv = kv.view(B, H, S, qk_nope_head_dim + v_head_dim)
        k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)
        scores = torch.einsum("bhsd,bhtd->bhst", q, k) * input_metadata.sm_scale
    else:
        wkv_b = wkv_b.view(H, -1, kv_lora_rank)
        q_nope = torch.einsum("bhsd,hdc->bhsc", q_nope, wkv_b[:, :qk_nope_head_dim])
        scores = (torch.einsum("bhsc,btc->bhst", q_nope, kv) +
                    torch.einsum("bhsr,btr->bhst", q_pe, k_pe.squeeze(1))) * input_metadata.sm_scale

    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(q_nope)

    if ref_impl=="naive":
        ref_out = torch.einsum("bhst,bhtd->bhsd", scores, v)
    else:
        x = torch.einsum("bhst,btc->bhsc", scores, kv)
        ref_out = torch.einsum("bhsc,hdc->bhsd", x, wkv_b[:, -v_head_dim:])

    torch.cuda.synchronize()
    print(f"time for ref: {time.time()-start}")


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
    args = parse_args()
    custom_config = False
    assert args.layout == 'thd' or not args.equal_seqlens, \
           "Equal sequence lengths arg must be used with the thd layout."
    if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
        custom_config = True
        assert args.b and args.hq and args.sq and args.d, \
               "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    assert args.dtype in arg_to_torch_dtype, \
           "Only fp16, bf16 and f32 types currently supported."

    # sanity_check(8, 16, 128, 512, 128, 64, 128, False, False, "bhsd")

    test_op_fwd(8, 16, 128, 512, 128, 64, 128, False, False, "bhsd", ref_impl="naive")


if __name__ == '__main__':
    sys.exit(main())
