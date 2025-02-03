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
    num_contexts = 0
    varlen = False
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

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
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        #assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0


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

@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0, 1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0, ), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1, ), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor

@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


def compute_alibi_tensor(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, actual_seqlen_k, actual_seqlen_q, dropout_p,
                    philox_seed, batch_philox_offset, encoded_softmax_block_ptr, block_min, block_max, offs_n_causal,
                    masked_blocks, n_extra_tokens, bias_ptr, alibi_slope, IS_CAUSAL: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, OFFS_M: tl.constexpr,
                    OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, PADDED_HEAD: tl.constexpr):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        k = load_fn(K_block_ptr, PADDED_HEAD, MASK_STEPS and (n_extra_tokens != 0), "zero")
        if PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
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
        qk += tl.dot(q, k)
        if bias_ptr is not None:
            bias = load_fn(bias_ptr, False, MASK_STEPS and (n_extra_tokens != 0), "zero")
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
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
def attn_fwd(Q, K, V, L, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz,
             stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh,
             stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah, Out, bias, 
             SM_SCALE: tl.constexpr, cu_seqlens_q, cu_seqlens_k, 
             dropout_p, philox_seed, philox_offset_base, encoded_softmax, 
             alibi_slopes, HQ: tl.constexpr, HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, BIAS_TYPE: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, USE_ALIBI: tl.constexpr,
             BATCH_SIZE: tl.constexpr):

    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

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
            o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
            O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(seqlen_q, BLOCK_DMODEL),
                                            strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
                                            block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            # We still need to write 0s to the result
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
            l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
            # We store inf to LSE, not -inf because in the bwd pass, we subtract this
            # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
            l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
            tl.store(l_ptrs, l)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    # need_padding = False
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        # need_padding = True
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        # need_padding = True
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
                                    strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N),
                                    order=(0, 1))
    v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
                                    strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL),
                                    order=(1, 0))
    if BIAS_TYPE != 0:
        b_offset = off_h_q * stride_bh  # Note: this might get large enough to overflow on some configs
        bias_ptr = tl.make_block_ptr(
            base=bias + b_offset,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        bias_ptr = None

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
    # TODO: Fix encoded softmax. It currently uses just h_q in the base offset.
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(base=encoded_softmax + off_h_q * seqlen_q * seqlen_k,
                                                      shape=(seqlen_q, seqlen_k), strides=(seqlen_k, 1),
                                                      offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N),
                                                      order=(1, 0))
    else:
        encoded_softmax_block_ptr = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = SM_SCALE * 1.44269504089
    # Q is loaded once at the beginning and shared by all N blocks.
    q = load_fn(Q_block_ptr, True, PADDED_HEAD, "zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

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
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, seqlen_k, seqlen_q,
                                        dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                                        block_min, block_max, 0, 0, 0, bias_ptr, alibi_slope,
                                        # IS_CAUSAL, ....
                                        False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks * BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks * BLOCK_N, 0))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, n_full_blocks * BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, n_full_blocks))
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, seqlen_k, seqlen_q,
                                        dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                                        block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, bias_ptr,
                                        alibi_slope, IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD)
    # epilogue
    acc = acc / l_i[:, None]
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
        # This is a > check because mask being 0 blocks the store.
        l_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    # write back O
    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
                                    strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    # Need boundary check on this to make sure the padding from the
    # Q and KV tensors in both dims are not part of what we store back.
    # TODO: Do the boundary check optionally.
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))
    

def get_shape_from_layout(q, k, metadata):
    batch, nheads_q, _, head_size = q.shape
    nheads_k = k.shape[1]
    return batch, nheads_q, nheads_k, head_size


# TODO: This can probably optimized to have fewer lines of code.
#def get_strides_from_layout(q, k, v, o, metadata):
def get_strides_from_layout(q, k, v, metadata):
    q_strides = jt.strides_from_shape(q.shape)
    k_strides = jt.strides_from_shape(k.shape)
    v_strides = jt.strides_from_shape(v.shape)
    #o_strides = jt.strides_from_shape(o.shape)
    return q_strides, k_strides, v_strides#, o_strides

def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, requires_grad=True):
    q_key, k_key, v_key = random.split(random.PRNGKey(0), 3)

    q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
    k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)

    q = random.normal(q_key, q_tensor_shape, dtype=jnp.float16)
    k = random.normal(k_key, k_tensor_shape, dtype=jnp.float16)
    v = random.normal(v_key, k_tensor_shape, dtype=jnp.float16)

    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
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

  # number of compute units available
  # Hard code some hyper parameters
  block_m=128
  block_n=128
  preload_v=False

  grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, block_m), nheads_q, batch)

  out_shape = jax.ShapeDtypeStruct(shape=(2, q.shape[0], q.shape[1], q.shape[2], q.shape[3]), dtype=q.dtype)

  atomic_counter = jnp.zeros([1], dtype=jnp.int32)

  metaparams = dict(
      bias=metadata.bias,
      SM_SCALE=metadata.sm_scale,
      cu_seqlens_q=metadata.cu_seqlens_q, 
      cu_seqlens_k=metadata.cu_seqlens_k, 
      dropout_p=metadata.dropout_p, 
      philox_seed=philox_seed,
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
      BIAS_TYPE=False if metadata.bias is None else True,
      ENABLE_DROPOUT=metadata.dropout_p > 0.0, 
      RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax, 
      USE_ALIBI=False if metadata.alibi_slopes is None else True,
      BATCH_SIZE=batch)

  output1, _ = jt.triton_call(q, k, v, M, *q_strides, *k_strides, 
                      *v_strides, *q_strides, *bias_strides, *alibi_strides,
                      kernel=attn_fwd, out_shape=out_shape, grid=grid, 
                      num_warps=4, num_stages=1, **metaparams)
  
  return output1

def main(unused_argv):

  #Random config
  dtype=jnp.float16
  causal=False
  use_alibi=False

  # Hardcode config
  #Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD = 4, 48, 48, 1, 1, 128
  Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD = 1, 32, 32, 1024, 1024, 128

  q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype)
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
