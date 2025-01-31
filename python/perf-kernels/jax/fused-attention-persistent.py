# Copyright 2024 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flash attention example."""
import functools

import jax
from jax import random
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
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

    #def check_args(self, q, k, v, o):
    def check_args(self, q, k, v):
        assert q.ndim == k.ndim and q.ndim == v.ndim

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        if self.varlen:
            assert q.ndim == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
        else:
            assert q.ndim == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        if self.int8:
            if self.int8_kv:
                assert v.dtype == k.dtype and k.dtype == jnp.int8
                assert q.dtype != k.dtype
                assert (self.v_descale is not None) and (self.k_descale is not None)
            else:
                assert q.dtype == k.dtype and q.dtype == v.dtype and q.dtype == jnp.int8
                assert (self.q_descale is not None) and (self.k_descale is not None) and (self.v_descale is not None)
                if self.use_p_scale:
                    assert (self.p_scale is not None) and (self.p_descale is not None)
        else:
            assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        #assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen

def _strides(shape):
  size = np.prod(shape)
  for s in shape:
    size = size // s
    yield int(size)


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
def _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, start_m,
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, q_descale,
                    k_descale, v_descale, p_scale, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, OFFS_M: tl.constexpr, OFFS_N: tl.constexpr,
                    PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
                    RETURN_ENCODED_SOFTMAX: tl.constexpr, PADDED_HEAD: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr,
                    QK_SCALE: tl.constexpr, INT8_GEMM: tl.constexpr, USE_P_SCALE: tl.constexpr, INT8_KV: tl.constexpr):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
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
        if INT8_GEMM:
            qk += ((((tl.dot(q, k).to(tl.float32) * q_descale)) * k_descale) * QK_SCALE)
        else:
            if INT8_KV:
                k = (k * k_descale).to(q.type.element_ty)
            qk += (tl.dot(q, k) * QK_SCALE)

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += (bias * 1.44269504089)

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk += (alibi_block * 1.44269504089)  # scale factor of log2(e)

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
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if INT8_GEMM:
            if USE_P_SCALE:
                p = (p * p_scale).to(tl.int8)
                # They are all int8
                acc += tl.dot(p, v)
            else:
                # v is in int8 but p is not, we want the gemm in p's type
                acc += tl.dot(p, v.to(p.type.element_ty))
        else:
            if INT8_KV:
                v = (v * v_descale).to(p.type.element_ty)
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i

@triton.jit
def attn_fwd(Q, K, V, L, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
             stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh,
             stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah, Out, bias, 
             SM_SCALE: tl.constexpr, Q_descale, K_descale, P_scale, P_descale, V_descale, cu_seqlens_q, cu_seqlens_k, 
             dropout_p, philox_seed, PERSISTENT: tl.constexpr, PERSISTENT_DYNAMIC: tl.constexpr, atomic_counter, 
             NUM_CU: tl.constexpr, GRID_CU_MULTIP: tl.constexpr, B: tl.constexpr, philox_offset_base, encoded_softmax, 
             alibi_slopes, HQ: tl.constexpr, HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, USE_ALIBI: tl.constexpr,
             INT8: tl.constexpr, USE_P_SCALE: tl.constexpr, INT8_KV: tl.constexpr):

    
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
        offs_d = tl.arange(0, BLOCK_DMODEL)

        continue_condition = True  # as we can't have return statements inside while loop in Triton

        if VARLEN:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0). Some seqlens might be too
            # small for all start_m so for those we return early.
            if start_m * BLOCK_M > seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        else:
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
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
                    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
                    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
                    o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to([BLOCK_M, BLOCK_DMODEL])
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
                PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

                # Compute pointers for all the tensors used in this kernel.
                q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
                q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
                k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
                v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
                v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
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
                acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                if PADDED_HEAD:
                    q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

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
                    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk,
                                                    stride_bn, start_m, seqlen_k, seqlen_q, dropout_p, philox_seed,
                                                    batch_philox_offset, encoded_sm_ptrs,
                                                    # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                                                    block_min, block_max, 0, 0, 0, alibi_slope, q_descale, k_descale,
                                                    v_descale, p_scale,
                                                    # IS_CAUSAL, ....
                                                    False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                                    # _, MASK_STEPS, ...
                                                    PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX,
                                                    PADDED_HEAD, ACTUAL_BLOCK_DMODEL, QK_SCALE, INT8_GEMM, USE_P_SCALE,
                                                    INT8_KV)
                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
                # Remaining blocks, if any, are full / not masked.
                if (masked_blocks > 0):
                    if IS_CAUSAL:
                        offs_n_causal = offs_n + (seqlen_q - seqlen_k)
                    else:
                        offs_n_causal = 0
                    k_ptrs += n_full_blocks * BLOCK_N * stride_kn
                    v_ptrs += n_full_blocks * BLOCK_N * stride_vk
                    if USE_BIAS:
                        bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                    if RETURN_ENCODED_SOFTMAX:
                        encoded_sm_ptrs += n_full_blocks * BLOCK_N
                    acc, l_i, m_i = _attn_fwd_inner(
                        acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, start_m, seqlen_k,
                        seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs, block_min, block_max,
                        offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, q_descale, k_descale, v_descale,
                        p_scale, IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD, ACTUAL_BLOCK_DMODEL,
                        QK_SCALE, INT8_GEMM, USE_P_SCALE, INT8_KV)

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
                        out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
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
                o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
                o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
                o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
                if overflow_size > 0:
                    o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
                if PADDED_HEAD:
                    o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = atomic_counter.atomic_add(1)
            else:
                tile_id += NUM_WG
        else:
            tile_id = num_tiles_total

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
#def get_strides_from_layout(q, k, v, o, metadata):
def get_strides_from_layout(q, k, v, metadata):
    if metadata.layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        #o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif metadata.layout == 'bhsd':
        q_strides = jt.strides_from_shape(q.shape)
        k_strides = jt.strides_from_shape(k.shape)
        v_strides = jt.strides_from_shape(v.shape)
        #o_strides = jt.strides_from_shape(o.shape)
    elif metadata.layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        #o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides#, o_strides

def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, requires_grad=True):
    q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, 'Got unsupported tensor layout'
    q = random.normal(q_key, q_tensor_shape, dtype=jnp.float16)
    k = random.normal(k_key, k_tensor_shape, dtype=jnp.float16)
    v = random.normal(v_key, k_tensor_shape, dtype=jnp.float16)

    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata

@functools.partial(jax.jit, static_argnames=["metadata"])
def _attention_fwd(q: jnp.ndarray, k: jnp.ndarray,
        v: jnp.ndarray, metadata: MetaData) -> jnp.ndarray:
        #v: jnp.ndarray, o: jnp.ndarray, metadata: MetaData) -> jnp.ndarray:
  # NOTE: a large bias tensor leads to overflow during pointer arithmetic
  if (metadata.bias is not None):
      assert (metadata.bias.numel() < 2**31)

  metadata.check_args(q, k, v)

  batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
  q_strides, k_strides, v_strides = get_strides_from_layout(q, k, v, metadata)

  # Get closest power of 2 over or equal to 32.
  padded_d_model = 1 << (head_size - 1).bit_length()
  # Smallest head_dim supported is 16. If smaller, the tile in the
  # kernel is padded - there is no padding in memory for any dims.
  padded_d_model = max(padded_d_model, 16)

  encoded_softmax = None

  M = jnp.empty((batch, nheads_q, metadata.max_seqlens_q), dtype=jnp.float32)

  # Seed the RNG so we get reproducible results for testing.
  philox_seed = 0x1BF52
  philox_offset = 0x1D4B42

  if metadata.bias is not None:
      bias_strides = (metadata.bias.stride(0), metadata.bias.stride(1), metadata.bias.stride(2),
                      metadata.bias.stride(3))
  else:
      bias_strides = (0, 0, 0, 0)

  if metadata.alibi_slopes is not None:
      alibi_strides = (metadata.alibi_slopes.stride(0), metadata.alibi_slopes.stride(1))
  else:
      alibi_strides = (0, 0)

  if metadata.int8:
      q_descale, k_descale, p_scale, p_descale, v_descale = metadata.q_descale, metadata.k_descale, metadata.p_scale, metadata.p_descale, metadata.v_descale
  else:
      q_descale = k_descale = p_scale = p_descale = v_descale = None

  # number of compute units available
  # Hard code some hyper parameters
  NUM_CU = 304
  block_m=128
  block_n=128
  preload_v=False
  grid_cu_multip=2

  if metadata.persistent is not None:
      grid = lambda META: (min(NUM_CU * grid_cu_multip,
                     triton.cdiv(metadata.max_seqlens_q, block_m) * nheads_q * batch), )
  else:
      grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, block_m), nheads_q, batch)

  out_shape = jax.ShapeDtypeStruct(shape=(2, q.shape[0], q.shape[1], q.shape[2], q.shape[3]), dtype=q.dtype)

  atomic_counter = jnp.zeros([1], dtype=jnp.int32)

  metaparams = dict(
      bias=metadata.bias,
      SM_SCALE= metadata.sm_scale,
      Q_descale=q_descale,
      K_descale=k_descale, 
      P_scale=p_scale, 
      P_descale=p_descale, 
      V_descale=v_descale, 
      cu_seqlens_q=metadata.cu_seqlens_q, 
      cu_seqlens_k=metadata.cu_seqlens_k, 
      dropout_p=metadata.dropout_p, 
      philox_seed=philox_seed,
      PERSISTENT=metadata.persistent is not None,
      PERSISTENT_DYNAMIC=metadata.persistent == "dynamic",
      atomic_counter=atomic_counter,
      NUM_CU=NUM_CU,
      GRID_CU_MULTIP=grid_cu_multip,
      B=batch,
      philox_offset_base=philox_offset,
      encoded_softmax=encoded_softmax,
      alibi_slopes=metadata.alibi_slopes,
      HQ=nheads_q, 
      HK=nheads_k, 
      ACTUAL_BLOCK_DMODEL=head_size, 
      MAX_SEQLENS_Q=metadata.max_seqlens_q,
      MAX_SEQLENS_K=metadata.max_seqlens_k, 
      VARLEN=metadata.varlen, 
      IS_CAUSAL=metadata.causal, 
      BLOCK_M=block_m,
      BLOCK_DMODEL=padded_d_model,
      BLOCK_N=block_n, 
      PRE_LOAD_V=preload_v, 
      USE_BIAS=False if metadata.bias is None else True,
      ENABLE_DROPOUT=metadata.dropout_p > 0.0, 
      RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax, 
      USE_ALIBI=False if metadata.alibi_slopes is None else True,
      INT8=metadata.int8, 
      USE_P_SCALE=metadata.int8 and metadata.use_p_scale, 
      INT8_KV=metadata.int8 and metadata.int8_kv)

  output1, _ = jt.triton_call(q, k, v, M, *q_strides, *k_strides, 
                      *v_strides, *q_strides, *bias_strides, *alibi_strides,
                      kernel=attn_fwd, out_shape=out_shape, grid=grid, num_warps=4, num_stages=1, **metaparams)
  
  return output1

def main(unused_argv):

  #Random config
  dtype=jnp.float16
  layout="bhsd"
  causal=False
  use_alibi=False
  persistent="fixed"
  #Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD = 4, 48, 48, 1, 1, 128
  Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD = 4, 48, 48, 1001, 990, 64

  q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
  input_metadata.set_persistent("fixed")
  o = jnp.empty_like(q)

  # Do Triton
  print("Triton")
  print(jax.jit(_attention_fwd,static_argnums=3)(q, k, v, input_metadata))

  # Do JAX
  # Replicate K and V if using MQA/GQA
  if HQ != HK:
      k = jnp.reshape(k, (k.shape[0], k.shape[1], -1, k.shape[2], k.shape[3]))
      k = jnp.expand_dims(k, (-1, -1, HQ // HK, -1, -1))
      k = jnp.reshape(k, (k.shape[0], -1, k.shape[2], k.shape[3]))
      v = jnp.reshape(v, (v.shape[0], v.shape[1], -1, v.shape[2], v.shape[3])).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

  scores = jnp.einsum('bhqd,bhkd->bhqk', q, k).astype("float32") * input_metadata.sm_scale
  #if causal:
 #     mask = jnp.tril(torch.ones(N_CTX_Q, N_CTX_K), diagonal=N_CTX_K - N_CTX_Q)
 #     scores[:, :, mask == 0] = float("-inf")
 # if use_alibi:
 #     scores += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)
#
  p = jax.nn.softmax(scores, axis=-1)
#  if causal:
#      # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
#      # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
#      # this by converting the NaNs to 0s, which is what they should be out of the softmax.
#      nan_mask = jnp.isnan(p)
#      p[nan_mask == 1] = 0
  ref_out = jnp.einsum('bhqk,bhkd->bhqd', p.astype("float16"), v)

  print("JAX")
  print(ref_out)

if __name__ == "__main__":
  from absl import app
  app.run(main)
