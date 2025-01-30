# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

import triton
import triton.language as tl

import sys
import torch
import pytest

import argparse

from utils.rotary_embedding import DeepseekScalingRotaryEmbedding

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


is_hip_ = is_hip()

logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
logger.warning("The following error message 'operation scheduled before its operands' can be ignored.")


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(Q_NOPE, Q_PE,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
                    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
                    W_KC,  # c x h x d
                    cos_sin_cache, # max_seq_len x (rotary_dim * 2)
                    positions, # sequence positions
                    sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
                    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank + 1)
                    stride_req_to_tokens_b,
                    stride_q_nope_b, stride_q_nope_h, stride_q_pe_b, stride_q_pe_h, stride_buf_kbs, stride_mid_ob, stride_mid_oh,
                    stride_mid_os, stride_w_kc_h, stride_w_kc_d, stride_w_kc_c,
                    stride_cos_sin_cache_s,
                    stride_positions_b,
                    Q_descale, W_KC_descale,
                    rotary_dim: tl.constexpr,
                    kv_lora_rank: tl.constexpr,
                    qk_nope_head_dim: tl.constexpr, qk_rope_head_dim: tl.constexpr, kv_group_num: tl.constexpr,
                    BLOCK_D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
                    NUM_KV_SPLITS: tl.constexpr, logit_cap: tl.constexpr, ROPE_FUSED: tl.constexpr, USE_FP8: tl.constexpr):

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_C)
    offs_q_r = tl.arange(0, BLOCK_R)  # to get the q_pe
    offs_k_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe
    # For tl.dot to meet dim requirement
    offs_i = tl.arange(0, 16)

    mask_d = offs_d < qk_nope_head_dim
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    if USE_FP8:
        off_q = cur_batch * stride_q_nope_b + cur_head * stride_q_nope_h + offs_d[None, :] + offs_i[:, None]
        q = tl.load(Q_NOPE + off_q, mask=(mask_d[None, :] & offs_i[:, None] < 1), other=0.0)
    else:
        off_q = cur_batch * stride_q_nope_b + cur_head * stride_q_nope_h + offs_d
        q = tl.load(Q_NOPE + off_q, mask=mask_d, other=0.0)

    off_q_pe = cur_batch * stride_q_pe_b + cur_head * stride_q_pe_h + offs_q_r
    mask_q_r = offs_q_r < qk_rope_head_dim
    mask_c = offs_c < kv_lora_rank
    mask_k_r = offs_k_r < kv_lora_rank + qk_rope_head_dim

    q_pe = tl.load(Q_PE + off_q_pe, mask=mask_q_r, other=0.0)

    # apply rotary embedding for q_pe
    if ROPE_FUSED:
        seqlen_k = NUM_KV_SPLITS * BLOCK_N
        pos_q = tl.load(positions + cur_batch * stride_positions_b + seqlen_k) # its the last
        offs_rotary = tl.arange(0, rotary_dim)
        cos_q = tl.load(cos_sin_cache + pos_q * stride_cos_sin_cache_s + offs_rotary)
        sin_q = tl.load(cos_sin_cache + pos_q * stride_cos_sin_cache_s + offs_rotary + rotary_dim)
        # neox style
        q_pe_1, q_pe_2 = q_pe.reshape(qk_rope_head_dim//2, 2).split()
        q_pe_rot = tl.join(-q_pe_2, q_pe_1).reshape(qk_rope_head_dim)
        q_pe = q_pe * cos_q + q_pe_rot * sin_q

    w_kc_offset = W_KC + cur_kv_head * stride_w_kc_h
    w_kc_ptrs = w_kc_offset + offs_d[:, None] * stride_w_kc_d + offs_c[None, :] * stride_w_kc_c
    mask_w_kc = (offs_d[:, None] < qk_nope_head_dim) & (mask_c[None, :])

    w_kc = tl.load(w_kc_ptrs, mask=mask_w_kc, other=0.0)

    # this doesn't work with fp8
    if USE_FP8:
        q = tl.dot(q, w_kc)

        q = tl.sum(q, 0)
    else:
        q = tl.sum(q[:, None] * w_kc, 0)  # 1 x c

    if USE_FP8:
        q = q.to(K_Buffer.type.element_ty)
        q *= Q_descale
        q *= W_KC_descale

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_kv = (kv_loc[:, None] * stride_buf_kbs + offs_c[None, :])
            offs_buf_k_pe = (kv_loc[:, None] * stride_buf_kbs + offs_k_r[None, :])

            kv = tl.load(
                K_Buffer + offs_buf_kv,
                mask=(offs_n[:, None] < split_kv_end) & (mask_c[None, :]),
                other=0.0,
            )  # the shared latent tensor for keys and values

            k_pe = tl.load(
                K_Buffer + offs_buf_k_pe,
                mask=(offs_n[:, None] < split_kv_end) & (mask_k_r[None, :]),
                other=0.0,
            )  # positional embedding part of keys

            if ROPE_FUSED:
                # apply rotary embedding for k_pe
                pos = tl.load(positions + cur_batch * stride_positions_b + offs_n[:, None])
                offs_rotary = tl.arange(0, rotary_dim)
                cos = tl.load(cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary[None, :])
                sin = tl.load(cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary[None, :] + rotary_dim)
                # neox style
                k_pe_1, k_pe_2 = k_pe.reshape(BLOCK_N, qk_rope_head_dim//2, 2).split()
                k_pe_rot = tl.join(-k_pe_2, k_pe_1).reshape(BLOCK_N, qk_rope_head_dim)
                k_pe = k_pe * cos + k_pe_rot * sin

            qk = tl.sum(q[None, :] * kv, 1)  # ((1 x c) * (BLOCK_N x c)).sum(1) = (BLOCK_N)
            qk += tl.sum(q_pe[None, :] * k_pe, 1)  # ((1 x r) * (BLOCK_N x r)).sum(1) = (BLOCK_N)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * kv, 0)  # ((BLOCK_N x 1) * (BLOCK_N x c)).sum(0) = 1 x c

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        # acc: 1 x c

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + offs_c)

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_c),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os +
                        kv_lora_rank)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q_nope,
    q_rope,
    kv_cache,
    att_out,
    w_kc,
    cos_sin_cache, positions, rotary_dim,
    q_descale, # scalar value
    w_kc_descale, # scalar value
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    logit_cap,
    fuse_rope,
    USE_FP8
):
    BLOCK = 64
    NUM_KV_SPLITS = num_kv_splits

    batch, head_num = B_req_idx.shape[0], q_nope.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = 1  # q.shape[1] // kv_cache.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    kv_lora_rank = w_kc.shape[-1]
    qk_nope_head_dim = w_kc.shape[1]
    qk_rope_head_dim = kv_cache.shape[-1] - kv_lora_rank

    BLOCK_D = triton.next_power_of_2(qk_nope_head_dim)
    BLOCK_C = triton.next_power_of_2(kv_lora_rank)
    BLOCK_R = triton.next_power_of_2(qk_rope_head_dim)


    _fwd_kernel_stage1[grid](q_nope, q_rope, kv_cache, w_kc, cos_sin_cache, positions, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen, att_out,
                             Req_to_tokens.stride(0), q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1), kv_cache.stride(0), att_out.stride(0),
                             att_out.stride(1), att_out.stride(2), w_kc.stride(0), w_kc.stride(1), w_kc.stride(2),
                             cos_sin_cache.stride(0), positions.stride(0),
                             q_descale, w_kc_descale, rotary_dim, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, kv_group_num=kv_group_num,
                             BLOCK_D=BLOCK_D, BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R, BLOCK_N=BLOCK,
                             NUM_KV_SPLITS=NUM_KV_SPLITS, logit_cap=logit_cap, USE_FP8=USE_FP8, num_warps=num_warps, num_stages=2, ROPE_FUSED=fuse_rope)


@triton.jit
def _fwd_kernel_stage2(Mid_O, W_VC,  # hdc
                       O, B_Seqlen, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_obs, stride_oh, stride_w_vch,
                       stride_w_vcc, stride_w_vcd, W_VC_descale, NUM_KV_SPLITS: tl.constexpr,
                       kv_lora_rank: tl.constexpr,  # we assume lora (low rank dim c) is pow of 2 and its the actual c
                       BLOCK_SIZE_N: tl.constexpr,  # we split lora dim for inner loop ??
                       BLOCK_DV: tl.constexpr,  # head_dim of v rounded to the nearest power of 2
                       BLOCK_C: tl.constexpr,  # lora rounded to the nearest power of 2
                       Lv: tl.constexpr,  # The actual head_dim of v
                       USE_FP8: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_c = tl.arange(0, BLOCK_C)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_c = offs_c < kv_lora_rank

    e_sum = 0.0
    e_max = -float("inf")
    if USE_FP8:
        acc = tl.zeros([BLOCK_C, 16], dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_c
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + kv_lora_rank
    offs_w_kv = cur_head * stride_w_vch + offs_n[:, None] * stride_w_vcd + offs_c[None, :] * stride_w_vcc
    w_kv_prts = W_VC + offs_w_kv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            # No more mask for this one as lora is pow of 2
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    acc = acc / e_sum  # c = 512

    if USE_FP8:
        acc_min = acc.min(0)
        acc_max = acc.max(0)
        amax = tl.max(tl.abs(acc_min), acc_max)

        FP8_MAX = 448.0

        scale = FP8_MAX / amax
        acc_de_scale = amax / -FP8_MAX
        # acc_scaled = acc * scale
        # acc_scaled = tl.clip(acc_scaled, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
        acc = (acc * scale).to(tl.float8e4nv)

    result = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    for n in range(0, tl.cdiv(BLOCK_DV, BLOCK_SIZE_N)):
        mask_v = offs_n[:, None] + n * BLOCK_SIZE_N < Lv
        mask_out = offs_n + n * BLOCK_SIZE_N < Lv
        w_vc = tl.load(w_kv_prts, mask=(mask_v & mask_c[None, :]), other=0.0)  # dc, head is parallelized (128, 512)

        if  USE_FP8:
            result = tl.dot(w_vc, acc)
        else:
            result = tl.sum(w_vc * acc[None, :], 1)


        if USE_FP8:
            result = result.to(O.type.element_ty)
            result *= acc_de_scale
            result *= W_VC_descale

        w_kv_prts += BLOCK_SIZE_N * stride_w_vcd

        offs_out = cur_batch * stride_obs + cur_head * stride_oh + offs_n + n * BLOCK_SIZE_N

        if USE_FP8:
            tl.store(
                O + offs_out[:, None] + tl.arange(0, 16)[None, :],
                result,
                mask=mask_out[:, None] & tl.arange(0, 16)[None, :] < 1,
            )
        else:
            tl.store(
                O + offs_out,
                result,
                mask=mask_out,
            )


# qk_nope_head_dim=v_head_dim=d
# w_kv has shape (c , ((d * 2) * num_heads)) its unpacked to w_kc and w_vc, along the d * 2 dim
# the output has shape
def _decode_softmax_reducev_fwd(
    logits,  # bhsc, c is the lora dim there's logit at the end of c dim
    w_vc,  # hdc each work group loads 512(c) * 128(d)
    q,
    o,
    w_vc_descale,
    Lv,  # head dim of v
    b_seq_len,
    num_kv_splits,
    USE_FP8
):
    batch, head_num = q.shape[0], q.shape[1]
    # hcd
    kv_lora_rank = w_vc.shape[1]
    # Lv = v_buffer.shape[-1],should be compressed c dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    # TODO tune this !!!!! tiling on the head_dim_v
    BLOCK_SIZE_N = 8

    BLOCK_C = triton.next_power_of_2(kv_lora_rank)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    # grid = lambda META: (batch, head_num, triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']))
    _fwd_kernel_stage2[grid](
        logits,
        w_vc,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        w_vc.stride(0),
        w_vc.stride(1),
        w_vc.stride(2),
        w_vc_descale,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        kv_lora_rank=kv_lora_rank,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_C=BLOCK_C,
        Lv=Lv,
        USE_FP8=USE_FP8,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    kv_cache,
    w_kc,
    w_vc,
    cos_sin_cache,
    positions,
    rotary_dim,
    v_head_dim,
    o,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
    fuse_rope=False,
    use_fp8=False,
):

    kv_lora_rank = w_kc.shape[-1]
    qk_nope_head_dim = w_kc.shape[1]
    qk_rope_head_dim = kv_cache.shape[-1] - kv_lora_rank
    q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    q_nope, q_nope_descale, w_kc, w_kc_descale, w_vc, w_vc_descale = quantize_input_fp8(q_nope, w_kc, w_vc, use_fp8)


    _decode_att_m_fwd(q_nope, q_rope, kv_cache, attn_logits, w_kc, cos_sin_cache, positions, rotary_dim, q_nope_descale, w_kc_descale, req_to_token, b_req_idx, b_seq_len, num_kv_splits, sm_scale,
                      logit_cap, fuse_rope, use_fp8)

    _decode_softmax_reducev_fwd(attn_logits, w_vc, q, o, w_vc_descale, v_head_dim, b_seq_len, num_kv_splits, use_fp8)


def attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap):
    from utils.sglang_ref import decode_attention_fwd_normal as decode_attention_fwd_normal_ref
    B, H = q_input.shape[0], q_input.shape[1]
    kv_lora_rank = v_input.shape[-1]
    device = q_input.device

    o = torch.empty((*q_input.shape[:-1], v_input.shape[-1]), dtype=q_input.dtype, device=q_input.device)
    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, device=device)
    decode_attention_fwd_normal_ref(q_input, k_input, v_input, o, Req_to_tokens, B_req_idx, B_Seqlen, attn_logits,
                                    num_kv_splits, sm_scale, logit_cap)
    return o, attn_logits


def input_helper(B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, device, rope_base=10, rope_max_seq_len=16324, rope_scaling=1.0):
    Req_to_tokens = torch.arange(B * S).reshape(B, S).to(device)
    B_req_idx = torch.arange(B).to(device)
    B_Seqlen = torch.full((B, ), S).to(device)

    q = torch.randn(B, H, D + qk_rope_head_dim, device=device)
    kv_cache = torch.randn(B * S, kv_lora_rank + qk_rope_head_dim, device=device)
    # v = k[:,:kv_lora_rank]

    att_out = torch.empty(B, H, D, device=device)
    attn_logits = torch.randn(B, H, num_kv_splits, kv_lora_rank + 1, device=device)

    w_kc = torch.randn(H, D, kv_lora_rank, device=device)
    w_vc = torch.randn(H, kv_lora_rank, D, device=device)


    rotary_dim = qk_rope_head_dim
    rotary_emb = DeepseekScalingRotaryEmbedding(
            qk_rope_head_dim,
            rotary_dim,
            rope_max_seq_len,
            rope_base,
            True,
            rope_scaling,
            q.dtype,
            device=device,
        )

    positions = torch.arange(0, S + 1).unsqueeze(0).repeat(B, 1).to(q.device) # k positions and q position as last

    return Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions


def input_to_float8(x, dtype=torch.float8_e4m3fnuz):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal().item()


def quantize_input_fp8(q, w_kc, w_vc, use_fp8):
    q_descale = w_kc_descale = w_vc_descale = None

    if use_fp8:
        q, q_descale = input_to_float8(q)
        w_kc, w_kc_descale = input_to_float8(w_kc)
        w_vc, w_vc_descale = input_to_float8(w_vc)

    return q, q_descale, w_kc, w_kc_descale, w_vc, w_vc_descale

@pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim', [
    (8, 16, 128, 512, 128, 64),
    (8, 16, 128, 511, 128, 64),
    (8, 16, 1024, 512, 128, 64),
])
@pytest.mark.parametrize('fuse_rope', [False, True])
@pytest.mark.parametrize('use_fp8', [False])
def test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, fuse_rope, use_fp8, num_kv_splits=2, sm_scale=1.0, logit_cap=0.0,
                device="cuda"):
    torch.manual_seed(0)
    torch.set_default_device(device)
    D = qk_nope_head_dim
    Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions = input_helper(
        B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, device)

    # Initialize additional parameters

    decode_attention_fwd_normal(
                    q,
                    kv_cache,
                    w_kc,
                    w_vc,
                    rotary_emb.cos_sin_cache,
                    positions,
                    rotary_dim,
                    D,
                    att_out,
                    Req_to_tokens,
                    B_req_idx,
                    B_Seqlen,
                    attn_logits,
                    num_kv_splits,
                    sm_scale,
                    logit_cap=0.0,
                    fuse_rope=fuse_rope,
                    use_fp8=use_fp8
                )

    tri_output, tri_logits = att_out, attn_logits  # .flatten(1,2)

    # reference
    ref_output, ref_logits = ref_impl(q, kv_cache, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, rope_fused=fuse_rope, use_fp8=use_fp8, device="cuda")

    print("first 10 logits:")
    print(f"ref: {ref_logits.flatten()[:10]}")
    print(f"tri: {tri_logits.flatten()[:10]}")
    torch.testing.assert_close(ref_logits, tri_logits, atol=1e-2, rtol=1e-2)
    print("attn_logits from stage 1 matches with ref")

    print("first 10 outputs:")
    print(f"ref: {ref_output.flatten()[:10]}")
    print(f"tri: {tri_output.flatten()[:10]}")
    torch.testing.assert_close(ref_output, tri_output, atol=1e-2, rtol=1e-2)
    print("attn_output from stage 2 matches with ref")


def ref_impl(q, kv_cache, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap, rotary_emb, positions, rope_fused=False, use_fp8=False,
             device="cuda"):

    B, H = q.shape[0], q.shape[1]

    kv_lora_rank = w_kc.shape[-1]
    qk_nope_head_dim = w_kc.shape[1]
    qk_rope_head_dim = kv_cache.shape[-1] - kv_lora_rank

    q_input = torch.empty(B, H, kv_lora_rank + qk_rope_head_dim).to(device)

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    q_nope, q_nope_descale, w_kc, w_kc_descale, w_vc, w_vc_descale = quantize_input_fp8(q_nope, w_kc, w_vc, use_fp8)

    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    if use_fp8:
        q_nope_out *= q_nope_descale
        q_nope_out *= w_kc_descale

    q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)

    latent_cache = kv_cache
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank:]

    if rope_fused:
        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)

    q_input[..., kv_lora_rank:] = q_pe
    k_input[..., kv_lora_rank:] = k_pe

    attn_output, attn_logits_ref = attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen,
                                            num_kv_splits, sm_scale, logit_cap)

    attn_output = attn_output.view(-1, H, kv_lora_rank)

    if use_fp8:
        attn_output, attn_output_descale = input_to_float8(attn_output)

    attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), w_vc)

    if use_fp8:
        attn_bmm_output *= attn_output_descale
        attn_bmm_output *= w_vc_descale

    ref_output = attn_bmm_output.transpose(0, 1)  #  # .flatten(1, 2)

    return ref_output, attn_logits_ref
    # ref_output, _ = self.o_proj(attn_output)


def benchmark(args):

    configs = []

    x_vals_list = [(8, 16, 1024, 512, 128, 64, 2), (8, 16, 4096, 512, 128, 64, 2), (8, 16, 8192, 512, 128, 64, 2)]
    x_names = ["B", "H", "S", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "num_kv_splits"]
    line_vals = ["ref", "fused"]
    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': "cuda"}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_kv_splits, sm_scale, logit_cap, device,
                  provider):
        warmup = 25
        rep = 100

        D = qk_nope_head_dim
        
        Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions = input_helper(
            B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, device)
        
        if "fused" in provider:
            fn = lambda: {
                decode_attention_fwd_normal(
                    q,
                    kv_cache,
                    w_kc,
                    w_vc,
                    rotary_emb.cos_sin_cache,
                    positions,
                    rotary_dim,
                    D,
                    att_out,
                    Req_to_tokens,
                    B_req_idx,
                    B_Seqlen,
                    attn_logits,
                    num_kv_splits,
                    sm_scale,
                    logit_cap=0.0,
                    fuse_rope=args.fuse_rope
                )
            }

        if "ref" in provider:
            fn = lambda: ref_impl(q, kv_cache, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, rope_fused=args.fuse_rope, device="cuda")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-fuse_rope", action='store_true', default=False, help="Test fusing rope inside kernel.")
    return parser.parse_args()


def main():
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    args = parse_args()
    benchmark(args)


if __name__ == '__main__':
    sys.exit(main())
