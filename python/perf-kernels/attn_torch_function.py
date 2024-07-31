#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import torch
import triton
import triton.language as tl

from flash_attention import (
    attn_fwd as tuned_attn_fwd,
    _attn_bwd_preprocess as bare_bwd_preprocess,
)
from _ref_bwd_split_kernel import (
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq
)

VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET = 0x1D4B42

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

TRITON_CONFIG_LIST_FWD = [
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=4),
   ]

TRITON_CONFIG_LIST_BWD_SIZED = [
       triton.Config({'waves_per_eu': 0}, num_stages=1, num_warps=4),
       triton.Config({'waves_per_eu': 1}, num_stages=1, num_warps=4),
       triton.Config({'waves_per_eu': 2}, num_stages=1, num_warps=4),
       triton.Config({'waves_per_eu': 3}, num_stages=1, num_warps=4),
       triton.Config({'waves_per_eu': 4}, num_stages=1, num_warps=4),
]

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_SIZED,
   key=['max_seqlen_q', 'max_seqlen_k'],
)
@triton.jit
def sized_tuned_bwd_kernel_dk_dv(
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
    num_seqlens,
    max_seqlen_q,
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
    bare_bwd_kernel_dk_dv(
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
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            dropout_p,
            philox_seed,
            philox_offset_base,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD=PADDED_HEAD,
            BIAS_TYPE=BIAS_TYPE,
            )

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_SIZED,
   key=['max_seqlen_q', 'max_seqlen_k'],
)
@triton.jit
def sized_tuned_bwd_kernel_dq(
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
    num_seqlens,
    max_seqlen_q,
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
    bare_bwd_kernel_dq(Q, K, V, B, sm_scale, Out, DO,
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
        num_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        head_dim,
        dropout_p,
        philox_seed,
        philox_offset_base,
        BLOCK_M, BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD=PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
        )

class _attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, b, causal, sm_scale, dropout_p, return_encoded_softmax,
                autotune=False, return_autotune=False):
        dtype = q.dtype
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        head_dim_rounded = 2 ** (Lk - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != Lk
        num_head_q = q.shape[1]
        num_head_k = q.shape[2]
        max_seqlen_q = q.shape[2]
        max_seqlen_k = k.shape[2]
        o = torch.zeros_like(q)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[1],
            q.shape[0],
        )
        null_tensor = torch.empty((0), device=q.device, dtype=torch.int32)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if return_encoded_softmax:
            encoded_softmax = torch.ones((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=q.dtype)
        else:
            encoded_softmax = None
        if False or VERBOSE:
            print(f'{q.shape=}')
            print(f'{k.shape=}')
            print(f'{v.shape=}')
            print(f'{o.shape=}')
            print(f'{q.data_ptr()=:x}')
            print(f'{k.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{M.data_ptr()=:x}')
            print(f'{o.data_ptr()=:x}')
            print(f'max_seqlen_q={q.shape[2]}')
            print(f'max_seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_seed = DEFAULT_PHILOX_SEED
        philox_offset = DEFAULT_PHILOX_OFFSET
        if b is None:
            b = torch.empty((0,0,0,0), device=q.device, dtype=q.dtype)
            BIAS_TYPE = 0
            USE_BIAS = False
        else:
            BIAS_TYPE = 1
            USE_BIAS = True

        use_small_block = dropout_p > 0.0 or BIAS_TYPE != 0 or return_encoded_softmax
        use_medium_block = False # reserved
        if use_small_block:
            BLOCK_M = 64
            BLOCK_N = 32
        elif use_medium_block:
            BLOCK_M = 64
            BLOCK_N = 64
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if dtype == torch.float32:
            BLOCK_M //= 2

        alibi_strides = (0, 0)
        tuned_attn_fwd[grid](
            q, k, v, b, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            b.stride(0), b.stride(1), b.stride(2), b.stride(3),
            *alibi_strides,
            cu_seqlens_q=null_tensor,
            cu_seqlens_k=null_tensor,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset_base=philox_offset,
            encoded_softmax=encoded_softmax,
            alibi_slopes=null_tensor,
            HQ=num_head_q,
            HK=num_head_k,
            ACTUAL_BLOCK_DMODEL=Lk,
            MAX_SEQLENS_Q=q.shape[2],
            MAX_SEQLENS_K=k.shape[2],
            IS_CAUSAL=causal,
            VARLEN=False,
            BLOCK_DMODEL=head_dim_rounded,
            USE_BIAS=USE_BIAS,
            USE_ALIBI=False,
            ENABLE_DROPOUT=dropout_p > 0.0,
            RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
        )

        ctx.autotune = autotune
        ctx.return_autotune = return_autotune
        if autotune and return_autotune:
            ## restore the grid for bwd kernel
            best_config = tuned_attn_fwd.get_best_config()
            tuning_result = copy.deepcopy(best_config)
            block_m = int(best_config.kwargs['BLOCK_M'])
            """
            # print(f'{best_config=}')
            # print(f'{dir(best_config)=}')
            # print(f'{str(best_config)=}')
            print("Best config")
            for key, value in best_config.kwargs.items():
                print('\t', key, '=', value)
            print(f'{str(best_config)=}')
            # block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
            block_m = int(best_config.kwargs['BLOCK_M'])
            print(f'{block_m=}')
            BATCH = q.shape[0]
            N_HEADS = q.shape[1]
            D_HEAD = q.shape[3]
            inputs = {
                'Q.shape' : list(q.shape),
                'Q.dtype' : str(q.dtype),
                'N_HEADS' : N_HEADS,
                'D_HEAD' : D_HEAD,
                'max_seqlen_q' : max_seqlen_q,
                'max_seqlen_k' : max_seqlen_k,
                'CAUSAL' : causal,
                'RETURN_ENCODED_SOFTMAX': encoded_softmax is not None,
                'BLOCK_DMODEL' : Lk,
                'ENABLE_DROPOUT' : dropout_p > 0.0,
            }
            tuned_kernel = dict(best_config.kwargs)
            compiler_options = {
                'num_warps' : best_config.num_warps,
                'num_stages': best_config.num_stages,
            }
            tuning_result = {
                'kernel_name' : 'attn_fwd',
                'inputs' : inputs,
                'tuned_kernel' : tuned_kernel,
                'compiler_options' : compiler_options,
            }
            """
        else:
            tuning_result = None
            block_m = min(128, q.shape[2], k.shape[2])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[1], q.shape[0])
        # print(f'{M=}')
        # print(f'{M.shape=}')
        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.head_dim = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.bias_type = BIAS_TYPE
        ctx.tuning_result = [('attn_fwd', tuning_result)] if tuning_result is not None else None
        if ctx.tuning_result is not None:
            for kernel_name, best in ctx.tuning_result:
                print(f'{kernel_name=} {best.kwargs=} {best.num_warps=} {best.num_stages=}')
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward(ctx, do, _, fwd_tuning_result):
        q, k, v, b, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv and Lk == ctx.head_dim
        head_dim_rounded = 2 ** (ctx.head_dim - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != ctx.head_dim

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b)
        delta = torch.empty_like(L)
        null_tensor = torch.empty((0), device=q.device, dtype=torch.int32)
        num_head_q = q.shape[1]
        num_head_k = q.shape[2]
        max_seqlen_q = q.shape[2]
        max_seqlen_k = k.shape[2]
        MAX_BLOCK = 64 if ctx.dropout_p == 0 else 16
        # BLOCK = min(max_seqlen_q, max_seqlen_k, q.shape[-1], MAX_BLOCK)
        # BLOCK = BLOCK if is_supported_by_tl_dot(max_seqlen_q) and is_supported_by_tl_dot(max_seqlen_k) else 1
        if not ctx.autotune:
            BLOCK = 16 # FIXME: Variable block size
        else:
            BLOCK = 128
        return_autotune = ctx.tuning_result is not None

        grid_prep = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
        bare_bwd_preprocess[grid_prep](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            max_seqlen_q,
            Lk,
            BLOCK_M=BLOCK, D_HEAD=head_dim_rounded,
        )
        if False or VERBOSE:
            print(f'{q.shape=} {q.stride()=}')
            print(f'{k.shape=} {k.stride()=}')
            print(f'{v.shape=} {v.stride()=}')
            print(f'{o.shape=} {o.stride()=}')
            print(f'{dq.shape=} {dq.stride()=}')
            print(f'{dk.shape=} {dk.stride()=}')
            print(f'{dv.shape=} {dv.stride()=}')
            print(f'{do.shape=} {do.stride()=}')
            print(f'{L=} {L.shape=}')
            print(f'{delta=}')
            print(f'{BLOCK=}')

        use_small_block = ctx.dropout_p > 0.0
        use_medium_block = ctx.bias_type != 0
        if use_small_block:
            # DQ_BLOCK_M = min(max_seqlen_q, BLOCK)
            BLOCK_M = 32
            BLOCK_N = 16
        elif use_medium_block:
            BLOCK_M = 64
            BLOCK_N = 32
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if q.dtype == torch.float32:
            BLOCK_M = max(16, BLOCK_M // 2)
            BLOCK_N = max(16, BLOCK_N // 2)
        # debug_mask = torch.zeros((q.shape[0], q.shape[1], max_seqlen_q, max_seqlen_k), device=q.device, dtype=ctx.encoded_softmax.dtype)
        grid_dk_dv = lambda META: (
            triton.cdiv(max_seqlen_k, META['BLOCK_N']),
            q.shape[1],
            q.shape[0],
        )
        stride_dbz, stride_dbh, stride_dbm, stride_dbn = db.stride()
        if db.numel() == 0 or not b.requires_grad:
            # Passing all zeros to indicate no elements
            stride_dbz, stride_dbh, stride_dbm, stride_dbn = 0,0,0,0
        else:
            db.fill_(float('nan'))
        print(f'backward {ctx.bias_type=} {ctx.autotune=} {BLOCK_M=} {BLOCK_N=} {stride_dbz=} {stride_dbh=} {stride_dbm=} {stride_dbn=}')
        if k.requires_grad and v.requires_grad:
            if ctx.autotune:
                sized_tuned_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                if return_autotune:
                    dkdv_best_config = copy.deepcopy(sized_tuned_bwd_kernel_dk_dv.get_best_config())
                    # BLOCK_M/N are missing with sized_tuned_bwd_kernel_*
                    dkdv_best_config.kwargs['BLOCK_M'] = BLOCK_M
                    dkdv_best_config.kwargs['BLOCK_N'] = BLOCK_N
                    tuning_result = copy.deepcopy(dkdv_best_config)
                    """
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'max_seqlen_q': max_seqlen_q,
                        'max_seqlen_k': max_seqlen_k,
                        'head_dim' : ctx.BLOCK_DMODEL,
                        'BLOCK_DMODEL' : head_dim_rounded,
                        'CAUSAL'  : ctx.causal,
                        'ENABLE_DROPOUT' : ctx.dropout_p > 0.0,
                    }
                    tuned_kernel = dict(dkdv_best_config.kwargs)
                    compiler_options = {
                        'num_warps' : dkdv_best_config.num_warps,
                        'num_stages': dkdv_best_config.num_stages,
                    }
                    tuning_result = {
                        'kernel_name' : 'bwd_kernel_dk_dv',
                        'inputs' : inputs,
                        'tuned_kernel' : tuned_kernel,
                        'compiler_options' : compiler_options,
                    }
                    """
                    ctx.tuning_result.append(('bwd_kernel_dk_dv', tuning_result))
                    print(f'{id(ctx.tuning_result)=}')
            else:
                bare_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    # debug_mask=debug_mask,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    num_warps=4,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
        print(f"{dq.stride()=}", flush=True)
        print(f"{dq.data_ptr()=:x}", flush=True)
        print(f"{dk.stride()=}", flush=True)
        print(f"{dk.data_ptr()=:x}", flush=True)
        # mask_allclose = torch.allclose(debug_mask < 0, ctx.encoded_softmax < 0)
        if False:
            mask_allclose = torch.allclose(torch.abs(debug_mask), torch.abs(ctx.encoded_softmax)) # Stores QK
            if not mask_allclose:
                torch.set_printoptions(linewidth=200, threshold=2000)
                import sys
                print(f'bwd mask: {torch.abs(debug_mask[:,:,:2,16:])}')
                print(f'fwd mask: {torch.abs(ctx.encoded_softmax[:,:,:2,16:])}')
                print(f'Full bwd mask: {debug_mask[0,0]}')
                print(f'Full fwd mask: {ctx.encoded_softmax[0,0]}')
                print(f'Full mask div: {debug_mask[0,0] / ctx.encoded_softmax[0,0]}')
                print(f'Full dv: {dv}')
                if max_seqlen_q == 32:
                    print(f'2nd block bwd mask: {debug_mask[0,0, 16:]}')
                    print(f'2nd block fwd mask: {ctx.encoded_softmax[0,0, 16:]}')
            # print(f'Full q: {q}', file=sys.stderr)
            # assert mask_allclose
        grid_dq = lambda META: (
            triton.cdiv(max_seqlen_q, META['BLOCK_M']),
            q.shape[1],
            q.shape[0],
        )
        if q.requires_grad:
            if ctx.autotune:
                sized_tuned_bwd_kernel_dq[grid_dq](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                if return_autotune:
                    dq_best_config = copy.deepcopy(sized_tuned_bwd_kernel_dq.get_best_config())
                    # BLOCK_M/N are missing with sized_tuned_bwd_kernel_*
                    dq_best_config.kwargs['BLOCK_M'] = BLOCK_M
                    dq_best_config.kwargs['BLOCK_N'] = BLOCK_N
                    tuning_result = dq_best_config
                    """
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'max_seqlen_q': max_seqlen_q,
                        'max_seqlen_k': max_seqlen_k,
                        'head_dim' : ctx.BLOCK_DMODEL,
                        'BLOCK_DMODEL' : head_dim_rounded,
                        'CAUSAL'  : ctx.causal,
                        'ENABLE_DROPOUT' : ctx.dropout_p > 0.0,
                    }
                    tuned_kernel = dict(dq_best_config.kwargs)
                    compiler_options = {
                        'num_warps' : dq_best_config.num_warps,
                        'num_stages': dq_best_config.num_stages,
                    }
                    tuning_result = {
                        'kernel_name' : 'bwd_kernel_dq',
                        'inputs' : inputs,
                        'tuned_kernel' : tuned_kernel,
                        'compiler_options' : compiler_options,
                    }
                    """
                    ctx.tuning_result.append(('bwd_kernel_dq', tuning_result))
            else:
                bare_bwd_kernel_dq[grid_dq](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    num_warps=4, waves_per_eu=1,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None if db.numel() == 0 else db, None, None, None, None, None, None, None

attention = _attention.apply
