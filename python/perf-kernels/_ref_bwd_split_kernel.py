#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""
import triton
import triton.language as tl
from _ref_bwd_kernel_common import bwd_kernel_dk_dv_common, bwd_kernel_dq_db_common
from masked_load_store import mstore2d

# TODO: Remove Unused 'Out' Argument from kernels below
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, B, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,   # set num_seqlens to zero to ignore cu_seqlens_q/k
    max_seqlen_q, # and use max_seqlen_q/k for all seqlen_q/k
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index, for varlen it indicates index in cu_seqlens_q/k
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    off_zh = off_z * num_h + off_h * 1
    if num_seqlens > 0:
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_m >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        batch_index = 0
    elif num_seqlens == 0:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        batch_index = off_z
    else: # < 0 for padded seqlen
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_m >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # Varlen, but padded to Rank 4 tensor
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z

    # Initialize pointers to Q, K, V
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    q_offset = off_h * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_h * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_h * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    VT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    do_offset = off_h * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(0, start_m),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    # pointer to row-wise quantities in value-like data
    # Shape (batch, num_heads, max_seqlen_q)
    # In varlen cases, batch == len(cu_seqlens_q) - 1).
    # Hence off_z plays the same role in varlen/non-varlen
    D_ptrs = D + off_zh * max_seqlen_q
    l_ptrs = L + off_zh * max_seqlen_q

    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
    else:
        batch_philox_offset = 0

    dk_offset = off_h * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
    DK_block_ptr = tl.make_block_ptr(
        base=DK + dk_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_dkn, stride_dkk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    dv_offset = off_h * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk
    DV_block_ptr = tl.make_block_ptr(
        base=DV + dv_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_dvk, stride_dvn),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    dk, dv = bwd_kernel_dk_dv_common(
        Q_block_ptr, KT_block_ptr, VT_block_ptr, B_block_ptr,
        sm_scale, DO_block_ptr,
        DK_block_ptr, DV_block_ptr,
        l_ptrs,
        D_ptrs,
        seqlen_q, seqlen_k,
        start_m,
        head_dim,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        max_seqlen_k,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD,
        BIAS_TYPE)
    mstore2d(dk,
             BLOCK_N,
             BLOCK_DMODEL,
             o_base=DK + dk_offset,
             o_start_row=start_m,
             o_start_col=0,
             o_rows=seqlen_k,
             o_cols=head_dim,
             stride_row=stride_dkn,
             stride_col=stride_dkk)
    mstore2d(dv,
             BLOCK_N,
             BLOCK_DMODEL,
             o_base=DV + dv_offset,
             o_start_row=start_m,
             o_start_col=0,
             o_rows=seqlen_k,
             o_cols=head_dim,
             stride_row=stride_dvk,
             stride_col=stride_dvn)

@triton.jit
def bwd_kernel_dq(
    Q, K, V, B, sm_scale, Out, DO,
    DQ, DB,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,   # set num_seqlens to zero to ignore cu_seqlens_q/k
    max_seqlen_q, # and use max_seqlen_q/k for all seqlen_q/k
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    off_zh = off_z * num_h + off_h * 1
    if num_seqlens > 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0
    elif num_seqlens == 0:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        batch_index = off_z
    else: # < 0 for padded seqlen
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        # Varlen, but padded to Rank 4 tensor
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z
    # Initialize pointers to Q, K, V
    q_offset = off_h * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_h * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_h * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    do_offset = off_h * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * max_seqlen_q
    l_ptrs = L + off_zh * max_seqlen_q
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
    else:
        batch_philox_offset = 0

    # initialize pointers to output
    dq_offset = batch_index * stride_dqz + off_h * stride_dqh + cu_seqlens_q_start * stride_dqm
    # tl.device_print('batch_index ', batch_index)
    # tl.device_print('off_h ', off_h)
    # tl.device_print('cu_seqlens_q_start ', cu_seqlens_q_start)
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + dq_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    store_db = True
    if BIAS_TYPE == 0:
        B_block_ptr = 0
        DB_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
        if (stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0:
            store_db = False
        # Still have to make one even if no_db = False
        # due to a limit of Triton: runtime branches must have identical data types.
        DB_block_ptr = tl.make_block_ptr(
                base=DB + off_h * stride_dbh + batch_index * stride_dbz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_dbm, stride_dbn),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    dq = bwd_kernel_dq_db_common(
        Q_block_ptr, K_block_ptr, V_block_ptr, B_block_ptr,
        sm_scale, DO_block_ptr,
        DQ_block_ptr, DB_block_ptr, store_db,
        l_ptrs,
        D_ptrs,
        seqlen_q, seqlen_k,
        start_m,
        head_dim,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        max_seqlen_k,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD,
        BIAS_TYPE)
    dq_ptrs, dq_masks = mstore2d(dq,
             BLOCK_M,
             BLOCK_DMODEL,
             o_base=DQ + dq_offset,
             o_start_row=start_m,
             o_start_col=0,
             o_rows=seqlen_q,
             o_cols=head_dim,
             stride_row=stride_dqm,
             stride_col=stride_dqk)
    # tl.device_print('dq_offset ', dq_offset)
    # tl.device_print('stride_dqm ', stride_dqm)
    # tl.device_print('stride_dqk ', stride_dqk)
    # tl.device_print('head_dim ', head_dim)
    # tl.device_print('dq_ptrs ', dq_ptrs)
    # tl.device_print('dq_masks ', dq_masks)
