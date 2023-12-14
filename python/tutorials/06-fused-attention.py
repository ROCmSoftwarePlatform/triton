"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16
TORCH_HAS_FP8 = False
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2')
TORCH_HAS_FP8E5FNUZ = hasattr(torch, 'float8_e5m2fnuz')
if TORCH_HAS_FP8E5:
    torch_dtype:tl.constexpr = torch.float8_e5m2
    TORCH_HAS_FP8 = True
if TORCH_HAS_FP8E5FNUZ:
    torch_dtype:tl.constexpr = torch.float8_e5m2fnuz
    TORCH_HAS_FP8 = True

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
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m,
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1: # "Solid" blocks of Causal masks
        lo, hi = 0, min(seqlen_k, start_m * BLOCK_M)
    elif STAGE == 2: # "Semi-solid", or "Transition" block of Causal mask
        # Must use BLOCK_M, because the starting position of semi-solid block
        # is determined by start_m * BLOCK_M
        lo, hi = start_m * BLOCK_M, min(seqlen_k, start_m * BLOCK_M + BLOCK_M)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
    else: # causal = False
        lo, hi = 0, seqlen_k
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        if STAGE == 1 or STAGE == 3:
            start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # Note about the conflicts of Flash attention algorithm and PyTorch's CUDA implementation
        # PyTorch needs to return softmax(qk) (dropout mask encoded in sign bits)
        # While Flash attention paper computer the dropout AFTER exp2(qk- m_ij)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty)) # FIXME: This is correct code
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty)) # FIXME: This is correct code
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

# We don't run auto-tuning everytime to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
@triton.autotune(
   configs=[
       triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=8),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=8),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4), # Fallback for dropout
   ],
   key=['seqlen_q', 'seqlen_k', 'STAGE', 'BLOCK_DMODEL'],
)


@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_hz * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and attn_fwd_inner gets 3 as its STAGE
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_hz * seqlen_q * seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_q, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            4 - STAGE, offs_m, offs_n,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX)
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_q, seqlen_k,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            2, offs_m, offs_n,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    m_ptrs = M + off_hz * seqlen_q + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    o_offset = off_hz * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         NewDO, Delta,  #
                         BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(O + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_hz = tl.program_id(1)
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointers to Q, K, V
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_hz * stride_vh
    VT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    do_offset = q_offset
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * seqlen_q
    l_ptrs = L + off_hz * seqlen_q
    qk_scale = sm_scale * 1.44269504
    # load k and v: they will stay in SRAM throughout
    k = tl.load(K_block_ptr)
    k = (k * qk_scale).to(K_block_ptr.type.element_ty)
    vt = tl.load(VT_block_ptr)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    lo = start_m if CAUSAL else 0
    hi = seqlen_q
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)

    dV = (QK)^T dO

    dV1 = qk11 dO1 + qk21 dO2 = q1 k1 dO1 + q2 k1 dO2
    dV2 = qk12 dO1 + qk22 dO2 = q1 k2 dO1 + q2 k2 dO2
                                ~~~~~ = 0
    start_m: select k and dV
    start_n: select q and dO
    '''
    # loop over q (seqlen_q, dhead), do (seqlen_q, d_head)
    for start_n in range(lo, hi, BLOCK_M):
        offs_m_curr = offs_n[:, None] + start_n
        # -- load q, do --
        q = tl.load(Q_block_ptr)
        do = tl.load(DO_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, k) # BLOCK_M x BLOCK_N
        if CAUSAL:
            qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_n * seqlen_k + start_m
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            # CAVEAT: do NOT update p, ds needs the original p
            dv += tl.dot(tl.where(tl.trans(keep), tl.trans(p) / (1 - dropout_p), 0.0).to(Q.dtype.element_ty), do)
        else:
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di)
        # compute dk
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    # initialize pointers to output
    DK_block_ptr = tl.make_block_ptr(
        base=DK + k_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.type.element_ty))
    tl.store(DV_block_ptr, dv.to(DK.type.element_ty))

@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointers to Q, K, V
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_hz * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * seqlen_q
    l_ptrs = L + off_hz * seqlen_q
    qk_scale = sm_scale * 1.44269504
    # load q and do: they will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    do = tl.load(DO_block_ptr)
    Di = tl.load(D_ptrs + offs_m)
    l_i = tl.load(l_ptrs + offs_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = start_m + BLOCK_M if CAUSAL else seqlen_k
    batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)
    '''
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        kt = tl.load(K_block_ptr)
        vt = tl.load(V_block_ptr)
        # -- compute qk ----
        qk = tl.dot(q, kt)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        dq += tl.dot(ds.to(Q.type.element_ty), tl.trans(kt))
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + q_offset,
        shape=(seqlen_q, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(DQ_block_ptr.type.element_ty))

empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p=0.0, return_encoded_softmax=False, split_kernel=True):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        o = torch.empty_like(q, dtype=v.dtype)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8
            # Tuning for H100
            if torch.cuda.get_device_capability()[0] == 9:
                num_warps = 8
                num_stages = 7 if Lk >= 64 else 3

        stage = 3 if causal else 1
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[0] * q.shape[1],
            1
        )
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=_attention.DEBUG_MASK_DTYPE)
        else:
            encoded_softmax = None

        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            BLOCK_DMODEL=Lk,
            STAGE=stage,
            ENABLE_DROPOUT=dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
        )

        ## restore the grid for bwd kernel
        best_config = _attn_fwd.get_best_config()
        block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.split_kernel = split_kernel
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        return o, encoded_softmax

    @staticmethod
    def backward(ctx, do, _):
        # configuration is not supported
        assert ctx.split_kernel
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        delta = torch.empty_like(L)
        do_scaled = torch.empty_like(do)
        _attn_bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,  #
            do_scaled, delta,  #
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,  #
        )
        dq = torch.zeros_like(q)
        _bwd_kernel_dk_dv[(triton.cdiv(q.shape[2], BLOCK), ctx.grid[1])](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset_base=ctx.philox_offset,
            # debug_mask=debug_mask,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            CAUSAL=ctx.causal,
            num_warps=4,
            num_stages=1,
            ENABLE_DROPOUT=ctx.dropout_p > 0.0,
        )
        DQ_BLOCK_M = min(seqlen_q, BLOCK)
        _bwd_kernel_dq[(triton.cdiv(q.shape[2], DQ_BLOCK_M), q.shape[0] * q.shape[1])](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset_base=ctx.philox_offset,
            BLOCK_M=DQ_BLOCK_M, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            CAUSAL=ctx.causal,
            num_warps=4, waves_per_eu=1,
            num_stages=1,
            ENABLE_DROPOUT=ctx.dropout_p > 0.0,
        )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None, None, None, None

attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (4, 48, 1024, 128),
                          (4, 48, 2048, 128),
                          (4, 48, 4096, 128),
                          #(4, 48, 8192, 64),
                          #(4, 48, 16384, 64)
                          ])
@pytest.mark.parametrize('causal', [False, True])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    if TORCH_HAS_FP8:
        q = q.to(torch_dtype)
        k = k.to(torch_dtype)
    sm_scale = 0.5
    dout = torch.randn_like(q, dtype=torch.float16)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q.half(), k.transpose(2, 3).half()) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (1, 16, 8192, 64),
                          ])
def test_op_bwd(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    causal = True
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()

    sm_scale = 0.5
    split_kernel = True
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, split_kernel)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        torch.testing.assert_close(ref_dv, tri_dv, atol=5e-2, rtol=0)
    torch.testing.assert_close(ref_dk, tri_dk, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=5e-2, rtol=1e-2)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd', 'bwd']:
    for D_HEAD in [128, 64]:
        if mode == 'bwd' and D_HEAD == 128:
            continue
        for causal in [False, True]:
            if mode == 'bwd' and causal == False:
                continue
            configs.append(triton.testing.Benchmark(
                x_names=['BATCH', 'H','N_CTX'],
                x_vals=[(16, 16, 1024),
                        (8, 16, 2048),
                        (4, 16, 4096),
                        (2, 16, 8192),
                        (1, 16, 16384),
                        (4, 48, 1024),
                        (4, 48, 2048),
                        (4, 48, 4096),
                        (4, 48, 8192),
                        (4, 48, 16384),
                        ],
                line_arg='provider',
                line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
                styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-{mode}-d{D_HEAD}-causal={causal}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': torch.float16,
                    'mode': mode,
                    'causal': causal})
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    split_kernel = True
    return_encoded_softmax = False # This is for UT, must disable for performance
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if mode == "fwd" and TORCH_HAS_FP8:
            q = q.to(torch_dtype)
            k = k.to(torch_dtype)
        sm_scale = 1.3
        dropout_p = 0.0 # TODO: benchmark with dropout
        fn = lambda: attention(q, k, v, causal, sm_scale, dropout_p, return_encoded_softmax, split_kernel)
        if mode == 'bwd':
            o, _ = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path=".", print_data=True)
