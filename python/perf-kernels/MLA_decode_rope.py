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
from utils.sglang_ref import _decode_softmax_reducev_fwd, decode_attention_fwd_grouped

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


is_hip_ = is_hip()

logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
logger.warning("The following error message 'operation scheduled before its operands' can be ignored.")

fp8_e4m3fnuz_max =torch.finfo(torch.float8_e4m3fnuz).max

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1



@triton.jit
def _fwd_grouped_kernel_stage1_rope(Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
                    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
                    cos_sin_cache, # max_seq_len x (rotary_dim * 2)
                    positions, # sequence positions
                    sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
                    Att_Out, # b x h x NUM_KV_SPLITS x (kv_lora_rank + 1)
                    stride_req_to_tokens_b,
                    stride_qb, stride_qh, stride_buf_kbs, stride_mid_ob, stride_mid_oh,
                    stride_mid_os,
                    stride_cos_sin_cache_s,
                    stride_positions_b,
                    rotary_dim: tl.constexpr,
                    kv_lora_rank: tl.constexpr,
                    qk_rope_head_dim: tl.constexpr, kv_group_num: tl.constexpr,
                    q_head_num: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
                    NUM_KV_SPLITS: tl.constexpr, logit_cap: tl.constexpr):

    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_c = tl.arange(0, BLOCK_C)
    offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

    # q_pe = tl.zeros([BLOCK_H, BLOCK_R], dtype=tl.float32)
    # # [BLOCK_R // 2]
    # offs_r = tl.arange(0, BLOCK_R) - (BLOCK_R // 2) % BLOCK_R
    # offs_q_r1 =  kv_lora_rank + tl.arange(0, BLOCK_R // 2)
    # offs_q_r2 =  kv_lora_rank + tl.arange(BLOCK_R // 2, BLOCK_R)

    # mask = [-1, 1]
    # q_pe = q_pe * mask

    off_q_pe = (cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :])
    mask_c = offs_c < kv_lora_rank
    mask_qk_r = offs_qk_r < kv_lora_rank + qk_rope_head_dim

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)

    # 0, 2, 4,.... 1, 3, 5...
    q_pe = tl.load(Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # apply rotary embedding for q_pe, and k_pe (last token per batch of K_PE)
    LAST_SPLIT = split_kv_end == cur_batch_seq_len
    last_token_pe_sum = tl.zeros([BLOCK_H, 1], dtype=q_pe.dtype)

    offs_rotary = tl.arange(0, rotary_dim//2)
    pos = tl.load(positions + cur_batch * stride_positions_b)

    cos = tl.load(cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary)
    sin = tl.load(cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary + rotary_dim)
    # neox style
    cos = tl.join(cos, cos).reshape(qk_rope_head_dim)
    sin = tl.join(sin, sin).reshape(qk_rope_head_dim)

    q_pe_1, q_pe_2 = q_pe.reshape(qk_rope_head_dim//2, 2).split()
    q_pe_rot = tl.join(-q_pe_2, q_pe_1).reshape(qk_rope_head_dim)
    q_pe = q_pe * cos + q_pe_rot * sin

    # we only apply to the last token in the K_PE
    # if LAST_SPLIT:
    #     # debug assert
    #     if (cur_batch==0 and cur_head==0) and split_kv_id < NUM_KV_SPLITS - 1:
    #             tl.device_assert(False, "Only last split should compute k_pe")

    #     kv_loc = tl.load(
    #         Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + cur_batch_seq_len - 1
    #     )
    #     offs_buf_k_pe = kv_loc * stride_buf_kbs + offs_k_r[None, :]
    #     k_pe = tl.load(K_Buffer + offs_buf_k_pe)
    #     k_pe_1, k_pe_2 = k_pe.reshape(qk_rope_head_dim//2, 2).split()
    #     k_pe_rot = tl.join(-k_pe_2, k_pe_1).reshape(qk_rope_head_dim)
    #     k_pe = k_pe * cos + k_pe_rot * sin
    #     # TODO: we need to save in the cache the rope'd k_pe token
    #     # tl.store(K_Buffer + offs_buf_k_pe, k_pe)
    #     last_token_pe_sum = tl.sum(q_pe[None, :] * k_pe, 1)


    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_kv = (kv_loc[None, :] * stride_buf_kbs + offs_c[:, None])
            offs_buf_k_pe = (kv_loc[None, :] * stride_buf_kbs + offs_qk_r[:, None])

            kv = tl.load(
                K_Buffer + offs_buf_kv,
                mask=(offs_n[None, :] < split_kv_end) & (mask_c[:, None]),
                other=0.0,
            )  # the shared latent tensor for keys and values

            k_pe = tl.load(
                K_Buffer + offs_buf_k_pe,
                mask=(offs_n[None, :] < split_kv_end) & (mask_qk_r[:, None]),
                other=0.0,
            ) # positional embedding part of keys

            # (16, 64) x (64, 32)
            # dot product of rope parts
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))

            # if LAST_SPLIT:
            #     qk = tl.where(offs_n[None, :] < split_kv_end - 1, qk, last_token_pe_sum.to(qk.type.element_ty))

            # (16, 512) x (512, 32)
            # dot product of nope parts
            qk += tl.dot(q, kv)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # (16, 32) x (32, 512)
            acc += tl.dot(p.to(kv.dtype), kv.trans())

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os + offs_c[None, :])

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_c[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os +
                        kv_lora_rank)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_att_m_fwd(
    q,
    kv_cache,
    att_out,
    kv_lora_rank, # c
    cos_sin_cache,
    positions,
    rotary_dim,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    logit_cap
):
    BLOCK = 32
    qk_rope_head_dim = kv_cache.shape[-1] - kv_lora_rank
    batch, head_num = B_req_idx.shape[0], q.shape[1]
    # we view kv_cache as one head
    kv_group_num = q.shape[1]

    BLOCK_C = triton.next_power_of_2(kv_lora_rank)
    BLOCK_R = triton.next_power_of_2(qk_rope_head_dim)

    # [TODO] work around shmem limit on MI3xx
    if is_hip_ and kv_lora_rank + qk_rope_head_dim >= 576:
        BLOCK = 16

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
    # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
    extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_grouped_kernel_stage1_rope[grid](q, kv_cache, cos_sin_cache, positions, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen, att_out,
                             Req_to_tokens.stride(0), q.stride(0), q.stride(1), kv_cache.stride(0), att_out.stride(0),
                             att_out.stride(1), att_out.stride(2),
                             cos_sin_cache.stride(0), positions.stride(0),
                             rotary_dim, kv_lora_rank, qk_rope_head_dim, kv_group_num=kv_group_num,
                             q_head_num=head_num, BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R, BLOCK_N=BLOCK, BLOCK_H=BLOCK_H,
                             NUM_KV_SPLITS=NUM_KV_SPLITS, logit_cap=logit_cap, num_warps=4, num_stages=1, **extra_kargs)

def decode_attention_fwd_grouped_rope(
    q,
    kv_cache,
    kv_lora_rank,
    cos_sin_cache,
    positions,
    rotary_dim,
    o,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):

    _decode_att_m_fwd(q, kv_cache, attn_logits, kv_lora_rank, cos_sin_cache, positions, rotary_dim, req_to_token, b_req_idx, b_seq_len, num_kv_splits, sm_scale,
                      logit_cap)

    _decode_softmax_reducev_fwd(attn_logits, q, o, kv_cache[..., kv_lora_rank:], b_seq_len, num_kv_splits)



def attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap):
    B, H = q_input.shape[0], q_input.shape[1]
    kv_lora_rank = v_input.shape[-1]
    device = q_input.device

    o = torch.empty((*q_input.shape[:-1], v_input.shape[-1]), dtype=q_input.dtype, device=q_input.device)
    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, dtype=q_input.dtype, device=device)
    decode_attention_fwd_grouped(q_input, k_input, v_input, o, Req_to_tokens, B_req_idx, B_Seqlen, attn_logits,
                                    num_kv_splits, sm_scale, logit_cap)
    return o, attn_logits


def input_helper(B, H, S, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device, rope_base=10, rope_max_seq_len=16324, rope_scaling=1.0):
    Req_to_tokens = torch.arange(B * S, device=device).reshape(B, S)
    B_req_idx = torch.arange(B, device=device)
    B_Seqlen = torch.full((B, ), S, device=device)

    q = torch.randn(B, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(B * S, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    # v = k[:,:kv_lora_rank]

    att_out = torch.empty(B, H, kv_lora_rank, dtype=dtype, device=device)
    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, dtype=dtype, device=device)

    # w = torch.randn(H, D, kv_lora_rank * 2, dtype=dtype, device=device)

    # w_kc, w_vc = torch.split(w, kv_lora_rank, dim=2)
    # w_vc = w_vc.permute([0, 2, 1])

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

    positions = torch.tensor([S], device=device).unsqueeze(0).repeat(B, 1) # k positions and q position as last

    return Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, rotary_dim, rotary_emb, positions


@pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_rope_head_dim', [
    (8, 128, 2048, 512, 64),
])
# @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op_fwd(B, H, S, kv_lora_rank, qk_rope_head_dim, dtype, num_kv_splits=2, sm_scale=1.0, logit_cap=0.0,
                device="cuda"):
    torch.manual_seed(0)

    Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, rotary_dim, rotary_emb, positions = input_helper(
        B, H, S, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device)

    # Initialize additional parameters

    decode_attention_fwd_grouped_rope(
                    q,
                    kv_cache,
                    kv_lora_rank,
                    rotary_emb.cos_sin_cache,
                    positions,
                    rotary_dim,
                    att_out,
                    Req_to_tokens,
                    B_req_idx,
                    B_Seqlen,
                    attn_logits,
                    num_kv_splits,
                    sm_scale,
                    logit_cap=0.0,
                )

    tri_output, tri_logits = att_out, attn_logits  # .flatten(1,2)

    # reference
    k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
    ref_output, ref_logits = ref_compute(q, k_input, v_input, kv_lora_rank, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, device="cuda")


    print("first 10 logits:")
    # print(f"ref: {ref_logits[:,:,-1].flatten()[:]}") # to debug the rope, check last split
    # print(f"tri: {tri_logits[:,:,-1].flatten()[:]}")
    print(f"ref: {ref_logits[3, 86, 1, 300:310]}") # to debug the rope, check last split
    print(f"tri: {tri_logits[3, 86, 1, 300:310]}")
    torch.testing.assert_close(ref_logits, tri_logits, atol=1e-2, rtol=1e-2)
    print("attn_logits from stage 1 matches with ref")

    # print("first 10 outputs:")
    # print(f"ref: {ref_output.flatten()[:10]}")
    # print(f"tri: {tri_output.flatten()[:10]}")
    # torch.testing.assert_close(ref_output, tri_output, atol=1e-2, rtol=1e-2)
    # print("attn_output from stage 2 matches with ref")

def ref_preprocess(kv_cache, kv_lora_rank):
    latent_cache = kv_cache
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    return k_input, v_input

def ref_compute(q, k_input, v_input, kv_lora_rank, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap, rotary_emb, positions,
             device="cuda"):

    B, H = q.shape[0], q.shape[1]
    S = B_Seqlen[0].item()

    qk_rope_head_dim = k_input.shape[-1] - kv_lora_rank

    q_input = torch.empty(B, H, kv_lora_rank + qk_rope_head_dim, dtype=q.dtype).to(device)
    q_nope_out, q_pe = q.split([kv_lora_rank, qk_rope_head_dim], dim=-1)

    # ROPE
    k_pe_t = k_input.view(B,1,S,-1)[:,:,-1:,kv_lora_rank:]
    q_pe, k_pe_t = rotary_emb(positions, q_pe.unsqueeze(2), k_pe_t)
    q_pe = q_pe.squeeze()
    k_input.view(B,1,S,-1)[:,:,-1:,kv_lora_rank:] = k_pe_t

    q_input[..., :kv_lora_rank] = q_nope_out
    q_input[..., kv_lora_rank:] = q_pe

    attn_output, attn_logits_ref = attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen,
                                            num_kv_splits, sm_scale, logit_cap)

    attn_output = attn_output.view(-1, H, kv_lora_rank)

    return attn_output, attn_logits_ref


def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    configs = []

    x_vals_list = [(1, 128, 2048, 512, 128, 64, 16)]
    x_names = ["B", "H", "S", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "num_kv_splits"]
    line_vals = ["ref", "fused"]
    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_kv_splits, sm_scale, logit_cap, device,
                  provider):
        warmup = 2
        rep = 2

        D = qk_nope_head_dim

        Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions = input_helper(
            B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device)

        if "fused" in provider:
            fn = lambda: {
                decode_attention_fwd_grouped_rope(
                    q,
                    kv_cache,
                    kv_lora_rank,
                    rotary_emb.cos_sin_cache,
                    positions,
                    rotary_dim,
                    att_out,
                    Req_to_tokens,
                    B_req_idx,
                    B_Seqlen,
                    attn_logits,
                    num_kv_splits,
                    sm_scale,
                    logit_cap=0.0,
                )
            }

        if "ref" in provider:
            k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
            ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                        logit_cap, rotary_emb, positions, device="cuda")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-device", default='cuda')
    return parser.parse_args()

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def main():
    torch.manual_seed(0)
    args = parse_args()
    torch.set_default_device(args.device)
    benchmark(args)


if __name__ == '__main__':
    sys.exit(main())
