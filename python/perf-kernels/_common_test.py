import os
from typing import List, Tuple, Optional
from collections import namedtuple
import numpy as np
import torch

def _reference_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        if dropout_mask is not None:
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    av = attn_weight @ value
    return av, attn_weight

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()

def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    # Low precision may yield NAN due to numerical instability
    # See https://github.com/pytorch/pytorch/issues/116176 for a real-world example.
    # Section 3 in https://arxiv.org/abs/2112.05682v3 explains how accelerated
    # SDPA does not suffer from it.
    deviation = torch.nan_to_num(true_value - computed_value)
    atol = torch.abs(deviation).max().item()
    return atol

def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol

SdpaParams = namedtuple('SdpaParams', ['causal', 'sm_scale', 'dropout_p', 'dropout_mask'])

class SdpaContext(object):
    TENSOR_NAMES = ('q', 'k', 'v', 'b')

    def __init__(self, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                 bias_type=None, storage_flip=None, device='cuda'):
        qdims = (BATCH, N_HEADS, seqlen_q, D_HEAD)
        kdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
        vdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
        bdims = (seqlen_q, seqlen_k)
        if storage_flip is not None:
            order = [0,1,2,3]
            x, y = storage_flip
            order[x], order[y] = order[y], order[x]
            i, j, k, l = order
            qdims = (qdims[i], qdims[j], qdims[k], qdims[l])
            kdims = (kdims[i], kdims[j], kdims[k], kdims[l])
            vdims = (vdims[i], vdims[j], vdims[k], vdims[l])
            # bdims = (bdims[1], bdims[0])
        # q = torch.empty(qdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        # k = torch.empty(kdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        # v = torch.empty(vdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        q = torch.rand(*qdims, dtype=dtype, device=device)
        k = torch.rand(*kdims, dtype=dtype, device=device)
        v = torch.rand(*vdims, dtype=dtype, device=device)
        if bias_type is None:
            b = None
        elif bias_type == 'matrix':
            # b = torch.empty(bdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
            b = torch.rand(*bdims, dtype=dtype, device=device)
            b = b.expand(BATCH, N_HEADS, b.shape[0], b.shape[1])
        else:
            assert False, f'Unsupported bias_type {bias_type}'
        if storage_flip is not None:
            x, y = storage_flip
            q = torch.transpose(q, x, y)
            k = torch.transpose(k, x, y)
            v = torch.transpose(v, x, y)
            '''
            # No need to support flipped storage
            # attn_mask.stride(-1) is assumed to be 1 in PyTorch
            if b is not None:
                b = torch.transpose(b, 2, 3)
                print(f'{b.stride()=}')
            '''
        self.dev_tensors = (q, k, v, b)
        # self.FUDGE_FACTORS = (4, 2, 2, 2)  # Matches the order of self.dev_tensors
        self.OUT_FUDGE_FACTOR = 3

    @property
    def dtype(self):
        return self.dev_tensors[0].dtype

    @property
    def seqlen_k(self):
        q, k, v, b = self.dev_tensors
        seqlen_k = k.shape[2]
        return seqlen_k

    @property
    def ref_device(self):
        return self.ref_tensors[0].device

    @staticmethod
    def clone_tensor(t, dtype, device=None):
        if t is None:
            return None
        return t.clone().detach().to(dtype=dtype, device=device).requires_grad_(t.requires_grad)

    @staticmethod
    def clone_tensor_tuple(in_tensors, dtype, device=None):
        return tuple([SdpaContext.clone_tensor(t, dtype=dtype, device=device) for t in in_tensors])

    def create_ref_inputs(self):
        ref_device_option = os.getenv('AOTRITON_REF_DEVICE_OPTION', default='default')
        if ref_device_option == 'default':
            seqlen_k = self.seqlen_k
            '''
            Shader _ZN2at6native12_GLOBAL__N_119cunn_SoftMaxForwardILi2EdddNS1_22SoftMaxForwardEpilogueEEEvPT2_PKT0_i causes Segfault
            for Case test_op_bwd[False-0.0-dtype2-0.0-False-587-64-8-4-4], but cannot be reproduced by running this individual UT.
            Avoiding running it on GPU for now
            '''
            if seqlen_k == 587:
                ref_device = 'cpu'
            else:
                ref_device = 'cuda'
        elif ref_device_option == 'cuda':
            ref_device = 'cuda'
        elif ref_device_option == 'cpu':
            ref_device = 'cpu'
        else:
            assert False, f'Unknown ref_device_option value {ref_device_option}. Allowed choices "default" "cpu" "cuda"'
        self.create_ref_inputs_with_device(ref_device)

    def create_ref_inputs_with_device(self, ref_device):
        dtype = self.dtype
        hp_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        self.ref_tensors = self.clone_tensor_tuple(self.dev_tensors, dtype=hp_dtype, device=ref_device)
        self.lp_ref_tensors = self.clone_tensor_tuple(self.dev_tensors, dtype=dtype, device=ref_device)

    @staticmethod
    def _require_grads(tensors, skip_dq=False, skip_dk_dv=False, skip_db=False):
        q, k, v, b = tensors
        if not skip_dq:
            q.requires_grad_()
        if not skip_dk_dv:
            k.requires_grad_()
            v.requires_grad_()
        if not skip_db:
            assert b is not None
            b.requires_grad_()

    def set_require_grads(self, skip_dq=False, skip_dk_dv=False, skip_db=False):
        self._require_grads(self.dev_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)
        self._require_grads(self.ref_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)
        self._require_grads(self.lp_ref_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)

    def _compute_fudge_factors(self, p : SdpaParams):
        ref_q, ref_k, ref_v, ref_b = self.ref_tensors
        dtype = self.dtype
        seqlen_k = self.seqlen_k
        seqlen_k_fudge_factor = 1.0 if seqlen_k < 1024 else 2.0
        dropout_fudge_factor = 1.0 if p.dropout_p == 0.0 else 2.0
        query_fudge_factor = 8 * dropout_fudge_factor * seqlen_k_fudge_factor  # TODO: Investigate why grad_q needs larger tolerances
        key_fudge_factor = 8 * dropout_fudge_factor
        value_fudge_factor = 7
        bias_fudge_factor = 12
        return (query_fudge_factor, key_fudge_factor, value_fudge_factor, bias_fudge_factor)

    @staticmethod
    def _compute_ref_forward(ref_tensors, p : SdpaParams):
        ref_q, ref_k, ref_v, ref_b = ref_tensors
        dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=ref_q.device)
        # _scaled_dot_product_attention_math seems also working for nested tensor
        ref_out, ref_mask = torch.ops.aten._scaled_dot_product_attention_math(ref_q, ref_k, ref_v,
                                                                    dropout_p=p.dropout_p,
                                                                    is_causal=p.causal,
                                                                    attn_mask=ref_b,
                                                                    scale=p.sm_scale,
                                                                    dropout_mask=dropout_mask)
        return (ref_out, ref_mask)

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        self.refout_tensors = self._compute_ref_forward(self.ref_tensors, p)
        self.lp_refout_tensors = self._compute_ref_forward(self.lp_ref_tensors, p)
        return self.lp_refout_tensors

    @staticmethod
    def _compute_backward(in_tensors, out, dout):
        q, k, v, b = in_tensors
        out.backward(dout.to(device=out.device, dtype=out.dtype))
        dq, q.grad = None if not q.requires_grad else q.grad.clone(), None
        dk, k.grad = None if not k.requires_grad else k.grad.clone(), None
        dv, v.grad = None if not v.requires_grad else v.grad.clone(), None
        if b is None or not b.requires_grad:
            db = None
        else:
            db, b.grad = b.grad.clone(), None
        return (dq, dk, dv, db)

    # Note: this follows pytorch's testing approach and expects low precision dout
    def compute_backward(self, out, dout):
        self.dref_tensors = self._compute_backward(self.ref_tensors, self.refout_tensors[0], dout)
        self.lp_dref_tensors = self._compute_backward(self.lp_ref_tensors, self.lp_refout_tensors[0], dout)
        self.dout_tensors = self._compute_backward(self.dev_tensors, out, dout)

    @staticmethod
    def _validate(out, ref, lp_ref, fudge_factor, tname):
        if out is None and ref is None:
            return True, float('nan')
        atol, rtol = get_tolerances(ref, lp_ref, fudge_factor)
        assert out is not None, f'd{tname} is none'
        assert ref is not None, f'd{tname}_ref is none'
        # print(f'{out=}')
        # print(f'{ref=}')
        x = out.to(device=ref.device)
        y = ref.to(out.dtype)
        max_adiff = float(torch.max(torch.abs(x - y)))
        return torch.allclose(x, y, atol=atol, rtol=rtol), max_adiff

    def validate_with_reference(self, out, grads):
        out_allclose, out_adiff = self._validate(out, self.refout_tensors[0], self.lp_refout_tensors[0], self.OUT_FUDGE_FACTOR, 'out')
        grads_allclose = []
        grads_adiff = []
        for grad, ref, lp_ref, fudge_factor, tname in zip(grads, self.dref_tensors, self.lp_dref_tensors, self.fudge_factors, self.TENSOR_NAMES):
            allclose, adiff = self._validate(grad, ref, lp_ref, fudge_factor, tname)
            grads_allclose.append(allclose)
            grads_adiff.append(adiff)
        return out_allclose, out_adiff, grads_allclose, grads_adiff

class VarlenSdpaContext(SdpaContext):
    TENSOR_NAMES = ('q', 'k', 'v', 'b')

    @staticmethod
    def _rng_varlen_tensor(num_heads, seqlens, head_dim, dtype, device, packed=False):
        # Note: do NOT use nested tensor here
        #       PyTorch's UT can use nested tensor because PyTorch preprocessed
        #       the input nested tensors before sending them to its SDPA
        #       backends (See sdpa_nested_preprocessing in
        #       aten/src/ATen/native/nested/cuda/NestedTensorTransformerUtils.cpp)
        #
        #       AOTriton works in the same level as PyTorch's SDPA backends and
        #       its UT should generate the tensor in the preprocessed format directly.
        dims = (1, np.sum(seqlens), num_heads, head_dim)
        return torch.rand(*dims, dtype=dtype, device=device).transpose(1, 2)
        '''
        def _size(seqlen):
            return (seqlen, num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

        return torch.nested.nested_tensor([
            torch.rand(_size(seqlen), device=device, dtype=dtype, requires_grad=True)
            for seqlen in seqlens])
        '''

    def __init__(self, N_HEADS, D_HEAD, seqlens_q, seqlens_k, dtype, device='cuda'):
        q  = self._rng_varlen_tensor(N_HEADS, seqlens_q, D_HEAD, dtype, device)
        k  = self._rng_varlen_tensor(N_HEADS, seqlens_k, D_HEAD, dtype, device)
        v  = self._rng_varlen_tensor(N_HEADS, seqlens_k, D_HEAD, dtype, device)
        b = None
        self.dev_tensors = (q, k, v, b)
        self.OUT_FUDGE_FACTOR = 3
        self._seqlens_q = np.array(seqlens_q)
        self._seqlens_k = np.array(seqlens_k)

    # Not perfect but fits our needs.
    @property
    def seqlen_k(self):
        return np.max(self._seqlens_q)

    @staticmethod
    def _compute_ref_forward_varlen(ref_tensors, seqlens_q, seqlens_k, p : SdpaParams):
        packed_ref_q, packed_ref_k, packed_ref_v, _ = ref_tensors
        packed_dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=packed_ref_q.device)
        ref_out_array = []
        ref_mask_array = []
        seqlen_q_start = 0
        seqlen_k_start = 0
        for i, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k)):
            ref_q = packed_ref_q[0, :, seqlen_q_start:seqlen_q_start+seqlen_q, :]
            ref_k = packed_ref_k[0, :, seqlen_k_start:seqlen_k_start+seqlen_k, :]
            ref_v = packed_ref_v[0, :, seqlen_k_start:seqlen_k_start+seqlen_k, :]
            dropout_mask = packed_dropout_mask[i, :, :, :] if packed_dropout_mask is not None else None
            print(f'REF {seqlen_q_start=} {seqlen_q_start+seqlen_q=} {ref_q.shape=} {ref_k.shape=} {ref_v.shape=}')
            print(f'REF {ref_q.stride()=}')
            if dropout_mask is not None:
                print(f'REF {packed_dropout_mask.shape=}')
                print(f'REF {dropout_mask.shape=}')
                dropout_mask = dropout_mask[:, :seqlen_q, :seqlen_k]  # Trim to actual seqlen
                print(f'REF CLAMPED {dropout_mask.shape=}')
                # print(f'REF {dropout_mask=}')
            ref_out, ref_mask = torch.ops.aten._scaled_dot_product_attention_math(ref_q, ref_k, ref_v,
                                                                        dropout_p=p.dropout_p,
                                                                        is_causal=p.causal,
                                                                        scale=p.sm_scale,
                                                                        dropout_mask=dropout_mask)
            ref_out_array.append(ref_out)
            ref_mask_array.append(ref_mask)
            seqlen_q_start += seqlen_q
            seqlen_k_start += seqlen_k
        ref_out = torch.cat(ref_out_array, dim=1).unsqueeze(dim=0)
        return ref_out, None

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        self.refout_tensors = self._compute_ref_forward_varlen(self.ref_tensors, self._seqlens_q, self._seqlens_k, p)
        self.lp_refout_tensors = self._compute_ref_forward_varlen(self.lp_ref_tensors, self._seqlens_q, self._seqlens_k, p)
        return self.lp_refout_tensors

