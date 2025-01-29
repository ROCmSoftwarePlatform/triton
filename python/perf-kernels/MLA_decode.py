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
def _fwd_kernel_stage1(Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
                       K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
                       W_KC,  # c x h x d
                       sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
                       Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank + 1)
                       stride_req_to_tokens_b, stride_qbs, stride_qh, stride_buf_kbs, stride_mid_ob, stride_mid_oh,
                       stride_mid_os, stride_w_kc_c, stride_w_kc_h, stride_w_kc_d, kv_lora_rank: tl.constexpr,
                       qk_nope_head_dim: tl.constexpr, qk_rope_head_dim: tl.constexpr, kv_group_num: tl.constexpr,
                       BLOCK_D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
                       NUM_KV_SPLITS: tl.constexpr, logit_cap: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_C)
    offs_q_r = tl.arange(qk_nope_head_dim, qk_nope_head_dim + BLOCK_R)  # to get the q_pe
    offs_k_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

    mask_d = offs_d < qk_nope_head_dim
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    off_q_pe = cur_batch * stride_qbs + cur_head * stride_qh + offs_q_r
    mask_q_r = offs_q_r < qk_nope_head_dim + qk_rope_head_dim
    mask_c = offs_c < kv_lora_rank
    mask_k_r = offs_k_r < kv_lora_rank + qk_rope_head_dim

    q_pe = tl.load(Q + off_q_pe, mask=mask_q_r, other=0.0)

    w_kc_offset = W_KC + cur_kv_head * stride_w_kc_h
    w_kc_ptrs = w_kc_offset + offs_d[:, None] * stride_w_kc_d + offs_c[None, :] * stride_w_kc_c
    mask_w_kc = (offs_d[:, None] < qk_nope_head_dim) & (mask_c[None, :])

    w_kc = tl.load(w_kc_ptrs, mask=mask_w_kc, other=0.0)

    q = tl.sum(q[:, None] * w_kc, 0)  # 1 x c

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([kv_lora_rank], dtype=tl.float32)

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
    q,
    k_buffer,
    att_out,
    w_kc,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 64
    NUM_KV_SPLITS = num_kv_splits

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = 1  # q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    kv_lora_rank = w_kc.shape[0]
    qk_nope_head_dim = w_kc.shape[1] // head_num
    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank

    BLOCK_D = triton.next_power_of_2(qk_nope_head_dim)
    BLOCK_C = triton.next_power_of_2(kv_lora_rank)
    BLOCK_R = triton.next_power_of_2(qk_rope_head_dim)

    w_kc = w_kc.view(kv_lora_rank, head_num, qk_nope_head_dim)

    _fwd_kernel_stage1[grid](q, k_buffer, w_kc, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen, att_out,
                             Req_to_tokens.stride(0), q.stride(0), q.stride(1), k_buffer.stride(0), att_out.stride(0),
                             att_out.stride(1), att_out.stride(2), w_kc.stride(0), w_kc.stride(1), w_kc.stride(2),
                             kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, kv_group_num=kv_group_num,
                             BLOCK_D=BLOCK_D, BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R, BLOCK_N=BLOCK,
                             NUM_KV_SPLITS=NUM_KV_SPLITS, logit_cap=logit_cap, num_warps=num_warps, num_stages=2)

@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    W_VC, # hdc
    O,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_w_vcc,
    stride_w_vch,
    stride_w_vcd,
    NUM_KV_SPLITS: tl.constexpr,
    LORA: tl.constexpr, # we assume lora (low rank dim c) is pow of 2 and its the actual c
    BLOCK_SIZE_N: tl.constexpr, # we split lora dim for inner loop ??
    BLOCK_DV: tl.constexpr, # head_dim of v rounded to the nearest power of 2
    Lv: tl.constexpr, # The actual head_dim of v
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    offs_c = tl.arange(0, LORA)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # TODO check this
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([LORA], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_c
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + LORA
    offs_w_kv = cur_head * stride_w_vch + offs_n[:, None] * stride_w_vcd + offs_c[None, :] * stride_w_vcc
    w_kv_prts = W_VC + offs_w_kv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            # No more mask for this one as lora is pow of 2
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    acc = acc / e_sum # c = 512

    result = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

    for n in range(0, tl.cdiv(BLOCK_DV, BLOCK_SIZE_N)):
        mask_v = offs_n[:, None] + n * BLOCK_SIZE_N < Lv
        mask_out = offs_n + n * BLOCK_SIZE_N < Lv
        w_vc = tl.load(w_kv_prts, mask=mask_v, other=0.0) # dc, head is parallelized (128, 512)

        result = tl.sum(w_vc * acc[None, :], 1)

        w_kv_prts += BLOCK_SIZE_N * stride_w_vcd

        offs_out = cur_batch * stride_obs + cur_head * stride_oh + offs_n + n * BLOCK_SIZE_N
        tl.store(
            O + offs_out,
            result,
            mask=mask_out,
        )

# qk_nope_head_dim=v_head_dim=d
# w_kv has shape (c , ((d * 2) * num_heads)) its unpacked to w_kc and w_vc, along the d * 2 dim
# the output has shape
def _decode_softmax_reducev_fwd(
    logits, # bhsc, c is the lora dim there's logit at the end of c dim
    w_vc, # hdc each work group loads 512(c) * 128(d)
    q,
    o,
    Lv, # head dim of v
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    # Lv = v_buffer.shape[-1],should be compressed c dim
    # TODO check the BLOCK_DV here
    BLOCK_DV = triton.next_power_of_2(Lv)
    # tiling on the head_dim_v
    BLOCK_K = 8

    kv_lora_rank = w_vc.shape[0]

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    # grid = lambda META: (batch, head_num, triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']))
    w_vc = w_vc.view(kv_lora_rank, head_num, Lv)

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
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        LORA=kv_lora_rank,
        BLOCK_SIZE_N=BLOCK_K,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    kv_cache,
    w_kc,
    w_vc,
    v_head_dim,
    o,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        q,
        kv_cache,
        attn_logits,
        w_kc,
        req_to_token,
        b_req_idx,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        logit_cap
    )

    _decode_softmax_reducev_fwd(attn_logits, w_vc, q, o, v_head_dim, b_seq_len, num_kv_splits)

def attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap):
    from utils.sglang_ref import decode_attention_fwd_normal as decode_attention_fwd_normal_ref
    B, H = q_input.shape[0], q_input.shape[1]
    kv_lora_rank = v_input.shape[-1]
    device = q_input.device

    o = torch.empty((*q_input.shape[:-1], v_input.shape[-1]), dtype=q_input.dtype, device=q_input.device)
    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, device=device)
    decode_attention_fwd_normal_ref(q_input, k_input, v_input, o, Req_to_tokens, B_req_idx, B_Seqlen, attn_logits, num_kv_splits, sm_scale, logit_cap)
    return o, attn_logits


@pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim', [
    (8, 16, 128, 512, 128, 64),
    (8, 16, 1024, 512, 128, 64),
])
def test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_kv_splits=2, sm_scale=1.0, logit_cap=0.0, device="cuda"):
    torch.manual_seed(0)
    D = qk_nope_head_dim

    Req_to_tokens = torch.arange(B * S).reshape(B, S).to(device)
    B_req_idx = torch.arange(B).to(device)
    B_Seqlen = torch.full((B, ), S).to(device)

    q = torch.randn(B, H, qk_nope_head_dim + qk_rope_head_dim, device=device)
    kv_cache = torch.randn(B * S, kv_lora_rank + qk_rope_head_dim, device=device)
    # v = k[:,:kv_lora_rank]

    att_out = torch.empty(B, H, D, device=device)
    attn_logits = torch.randn(B, H, num_kv_splits, kv_lora_rank + 1, device=device)

    w_kc = torch.randn(kv_lora_rank, H * qk_nope_head_dim, device=device)
    w_vc = torch.randn(kv_lora_rank, H * qk_nope_head_dim, device=device)

    # Initialize additional parameters

    decode_attention_fwd_normal(
        q,
        kv_cache,
        w_kc,
        w_vc,
        D,
        att_out,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        attn_logits,
        num_kv_splits,
        sm_scale,
        logit_cap=0.0,
    )

    tri_output = att_out # .flatten(1,2)

    # reference
    q_input = torch.empty(
            B, H, kv_lora_rank + qk_rope_head_dim
        ).to(device)

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    w_kc = w_kc.view(kv_lora_rank, H, qk_nope_head_dim).permute((1,2,0))

    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., : kv_lora_rank] = q_nope_out.transpose(0, 1)

    latent_cache = kv_cache
    v_input = latent_cache[..., : kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., : kv_lora_rank] = v_input
    k_pe = k_input[..., kv_lora_rank :]

    # q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    q_input[..., kv_lora_rank :] = q_pe
    k_input[..., kv_lora_rank :] = k_pe

    attn_output, attn_logits_ref = attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap)

    print("first 10 logits:")
    print(f"ref: {attn_logits_ref.flatten()[:10]}")
    print(f"tri: {attn_logits.flatten()[:10]}")
    torch.testing.assert_close(attn_logits_ref, attn_logits, atol=1e-2, rtol=1e-2)
    print("attn_logits are correct")

    attn_output = attn_output.view(-1, H, kv_lora_rank)

    w_vc = w_vc.view(kv_lora_rank, H, qk_nope_head_dim).permute((1,0,2))

    attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), w_vc)

    ref_output = attn_bmm_output.transpose(0, 1) # .flatten(1, 2)
    # ref_output, _ = self.o_proj(attn_output)
    # print(ref_output.shape)
    # print(tri_output.shape)

    print("first 10 outputs:")
    print(f"ref: {ref_output.flatten()[:10]}")
    print(f"tri: {tri_output.flatten()[:10]}")
    torch.testing.assert_close(ref_output, tri_output, atol=1e-2, rtol=1e-2)
    print("attn_output are correct")



def main():
    torch.manual_seed(0)
    B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim = 8, 16, 1024, 512, 128, 64
    num_kv_splits = 2
    sm_scale = 1.0
    logit_cap = 0.0
    device = "cuda"

    test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_kv_splits, sm_scale, logit_cap, device)


if __name__ == '__main__':
    sys.exit(main())
