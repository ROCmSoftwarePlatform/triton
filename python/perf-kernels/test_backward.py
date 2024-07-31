#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os

from attn_torch_function import attention
from _common_test import SdpaContext, SdpaParams

def _make_block_eyes(q, base=1.0, inc=0.0):
    dhead = q.shape[-1]
    seqlen = q.shape[2]
    assert seqlen % dhead == 0
    scale = base
    for i in range(0, seqlen, dhead):
        q[:, :, i:i+dhead, :] = torch.eye(dhead, device=q.device, dtype=q.dtype) * scale
        scale += inc

def RP(x):
    rounded = 2 ** (x - 1).bit_length()
    return max(16, rounded)

'''
Flash Attention is batch operator that evaluates sm(QK')V
Q = batch_size x ... x seqlen_q x head_size
K = batch_size x ... x seqlen_k x head_size
    => K' = batch_size x ... x head_size x seqlen_k
V = batch_size x ... x seqlen_k x head_size
sm(.) = softmax(.)
The output size is
batch_size x ... x seqlen_q x head_size

Note: In Flash V2 API the ... is denoted as "num_heads", serving as uniformly sized sequences
but in PyTorch API it does not present at all
'''

def _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if causal and seqlen_q != seqlen_k:
        pytest.skip("PyTorch's Flash V2 does not accept casual=True when seqlen_q != seqlen_k. Skipping")
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    SKIP_DK_DV = False
    SKIP_DQ = False
    SKIP_DB = True if bias_type is None else False
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)

    return_encoded_softmax = True
    q, k, v, b = ctx.dev_tensors
    # autotune = True
    # # triton implementation
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, return_encoded_softmax, USE_AUTOTUNE)
    dropout_mask = encoded_softmax >= 0
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)
    dout = torch.rand_like(tri_out)
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff = ctx.validate_with_reference(tri_out, ctx.dout_tensors)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        print(f'{tri_out[0, 0, :4, :4]=}')
        print(f'{ref_out[0, 0, :4, :4]=}')
    assert is_allclose, 'Forward pass {is_allclose=}'

    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=dtype)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{dropout_mask.shape=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{torch.isnan(ref_dv).any()=}')
        '''
        any_nan = torch.isnan(ref_dv).any()
        if any_nan:
            torch.set_printoptions(linewidth=200)
            print(f'{q=}')
            print(f'{k=}')
            print(f'{v=}')
            print(f'{dropout_p=}')
            print(f'{causal=}')
            print(f'{sm_scale=}')
        '''
        if seqlen_q <= 16 or True:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dk[0,0]=}')
            print(f'{ref_dk[0,0]=}')
            print(f'{tri_dv[0,0]=}')
            print(f'{ref_dv[0,0]=}')
            # print(f'{tri_dq[0,0]=}')
            # print(f'{ref_dq[0,0]=}')

    if dv_allclose and not dk_allclose:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

    if dk_allclose and dv_allclose and not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

    if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
        print(f'{err_idx=}')
        print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')
    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}'
    print(f'{adiff=} {grads_adiff=}')

# @pytest.mark.parametrize('BATCH', [1])
# @pytest.mark.parametrize('N_HEADS', [1])
@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [1, 4])
# @pytest.mark.parametrize('D_HEAD', [16, 32, 64, 128, 256])
# Irregular-only PyTorch set
# @pytest.mark.parametrize('D_HEAD', [8, 21, 72, 96, 160, 192, 203])
# @pytest.mark.parametrize('seqlen_q', [1, 4, 32, 128, 256, 512, 1024, 7, 394, 250, 399, 511, 1019])
# @pytest.mark.parametrize('seqlen_k', [1, 4, 32, 128, 256, 512, 1024, 3, 217, 339, 313, 491, 988])
# PyTorch set
@pytest.mark.parametrize('D_HEAD', [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256])
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# Currently debugging
# @pytest.mark.parametrize('D_HEAD', range(8,128+1,4))
# @pytest.mark.parametrize('D_HEAD', [84,92,108, 203] + list(range(128, 256+1, 4)))
# @pytest.mark.parametrize('D_HEAD', [84,203])
# @pytest.mark.parametrize('D_HEAD', range(8,64+1,4))
# @pytest.mark.parametrize('seqlen_q', [128, 2048, 4096])
# @pytest.mark.parametrize('seqlen_k', [128, 2048, 4096])
# Minimal set
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
# @pytest.mark.parametrize('sm_scale', [1.2])
# @pytest.mark.parametrize('storage_flip', [False])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('D_HEAD', [16,32,64,128,256])
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [128, 113])
# @pytest.mark.parametrize('seqlen_k', [128, 79])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_bwd_with_matrix_bias(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main2():
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-4-1
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-1-4
    # False-1.2-dtype0-0.0-False-4-4-72-1-4
    # BATCH = 8
    # D_HEAD = 32
    # N_HEADS = 8
    # seqlen_q = 16
    # seqlen_k = 16
    BATCH = 4
    D_HEAD = 1
    N_HEADS = 8
    seqlen_q = 256
    seqlen_k = 4
    causal = False
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main():
    BATCH = 1
    D_HEAD = 80
    '''
    N_HEADS = 2
    seqlens_q = 6432
    seqlens_k = 6432
    '''
    N_HEADS = 6432
    seqlens_q = 2
    seqlens_k = 2
    causal = False
    sm_scale = 1.2
    dropout_p = 0.5
    dtype = torch.bfloat16
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    main2()
