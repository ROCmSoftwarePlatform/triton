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
import time

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


fp8_e4m3fnuz_max = torch.finfo(torch.float8_e4m3fnuz).max


def attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap, persistent=False):
    
    if persistent:
        from utils.sglang_ref import decode_attention_fwd_grouped_persistent as decode_attention_fwd_ref
    else:
        from utils.sglang_ref import decode_attention_fwd_grouped as decode_attention_fwd_ref

    
    B, H = q_input.shape[0], q_input.shape[1]
    kv_lora_rank = v_input.shape[-1]
    device = q_input.device

    o = torch.empty((*q_input.shape[:-1], v_input.shape[-1]), dtype=q_input.dtype, device=q_input.device)
    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, dtype=q_input.dtype, device=device)
    decode_attention_fwd_ref(q_input, k_input, v_input, o, Req_to_tokens, B_req_idx, B_Seqlen, attn_logits,
                                    num_kv_splits, sm_scale, logit_cap)
    return o, attn_logits


def input_helper(B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device, rope_base=10, rope_max_seq_len=16324, rope_scaling=1.0):
    Req_to_tokens = torch.arange(B * S, device=device).reshape(B, S)
    B_req_idx = torch.arange(B, device=device)
    B_Seqlen = torch.full((B, ), S, device=device)

    q = torch.randn(B, H, D + qk_rope_head_dim, dtype=dtype, device=device)
    kv_cache = torch.randn(B * S, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)
    # v = k[:,:kv_lora_rank]

    att_out = torch.empty(B, H, D, dtype=dtype, device=device)
    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, dtype=dtype, device=device)

    w_kc = torch.randn(H, D, kv_lora_rank, dtype=dtype, device=device)
    w_vc = torch.randn(H, kv_lora_rank, D, dtype=dtype, device=device)


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

    return Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions


def input_to_float8(x, dtype=torch.float8_e4m3fnuz):
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = fp8_e4m3fnuz_max / amax
    x_scl_sat = (x * scale).clamp(min=-fp8_e4m3fnuz_max, max=fp8_e4m3fnuz_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal().item()


def quantize_input_fp8(q, w_kc, w_vc, use_fp8):
    q_descale = w_kc_descale = w_vc_descale = None

    if use_fp8:
        q, q_descale = input_to_float8(q)
        w_kc, w_kc_descale = input_to_float8(w_kc)
        w_vc, w_vc_descale = input_to_float8(w_vc)

    return q, q_descale, w_kc, w_kc_descale, w_vc, w_vc_descale

@pytest.mark.parametrize('B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim', [
    (32, 16, 2048, 512, 128, 64),
])
@pytest.mark.parametrize('fuse_rope', [False])
@pytest.mark.parametrize('use_fp8', [False])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, fuse_rope, use_fp8, dtype, num_kv_splits=32, sm_scale=1.0, logit_cap=0.0,
                device="cuda"):
    torch.manual_seed(0)

    D = qk_nope_head_dim
    Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions = input_helper(
        B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device)


    k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)

    tri_output, tri_logits = ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions,  rope_fused=fuse_rope, use_fp8=use_fp8, device="cuda", persistent=True)

    # reference
    
    ref_output, ref_logits = ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions,  rope_fused=fuse_rope, use_fp8=use_fp8, device="cuda")

    print("first 10 logits:")
    print(f"ref: {ref_logits[:,:,-1].flatten()[:]}") # to debug the rope, check last split
    print(f"tri: {tri_logits[:,:,-1].flatten()[:]}")
    torch.testing.assert_close(ref_logits, tri_logits, atol=1e-2, rtol=1e-2)
    print("attn_logits from stage 1 matches with ref")

    print("first 10 outputs:")
    print(f"ref: {ref_output.flatten()[:10]}")
    print(f"tri: {tri_output.flatten()[:10]}")
    torch.testing.assert_close(ref_output, tri_output, atol=1e-2, rtol=1e-2)
    print("attn_output from stage 2 matches with ref")

def ref_preprocess(kv_cache, kv_lora_rank):
    latent_cache = kv_cache
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    return k_input, v_input

def ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap, rotary_emb, positions, rope_fused=False, use_fp8=False,
             device="cuda", persistent=False):

    B, H = q.shape[0], q.shape[1]
    S = B_Seqlen[0].item()
    kv_lora_rank = w_kc.shape[-1]
    qk_nope_head_dim = w_kc.shape[1]
    qk_rope_head_dim = k_input.shape[-1] - kv_lora_rank

    q_input = torch.empty(B, H, kv_lora_rank + qk_rope_head_dim, dtype=q.dtype).to(device)
    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope, q_nope_descale, w_kc, w_kc_descale, w_vc, w_vc_descale = quantize_input_fp8(q_nope, w_kc, w_vc, use_fp8)
    
    # torch.cuda.synchronize()
    # start_t = time.time()

    if use_fp8:
        q_nope_out = torch.bmm(q_nope.transpose(0, 1).float(), w_kc.float())
        q_nope_out *= q_nope_descale
        q_nope_out *= w_kc_descale
        q_nope_out = q_nope_out.to(q.dtype)
    else:
        q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)

    q_input[..., :kv_lora_rank] = q_nope_out.transpose(0, 1)

    if rope_fused:
        k_pe_t = k_input.view(B,1,S,-1)[:,:,-1:,kv_lora_rank:]
        q_pe, k_pe_t = rotary_emb(positions, q_pe.unsqueeze(2), k_pe_t)
        q_pe = q_pe.squeeze()
        k_input.view(B,1,S,-1)[:,:,-1:,kv_lora_rank:] = k_pe_t

    q_input[..., kv_lora_rank:] = q_pe

    attn_output, attn_logits_ref = attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen,
                                            num_kv_splits, sm_scale, logit_cap, persistent=persistent)

    attn_output = attn_output.view(-1, H, kv_lora_rank)

    if use_fp8:
        attn_output, attn_output_descale = input_to_float8(attn_output)

        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1).float(), w_vc.float())
        attn_bmm_output *= attn_output_descale
        attn_bmm_output *= w_vc_descale
        attn_bmm_output = attn_bmm_output.to(q.dtype)
    else:
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), w_vc)

    
    # torch.cuda.synchronize()
    # print(f"time for ref {time.time() - start_t}")


    ref_output = attn_bmm_output.transpose(0, 1)  #  # .flatten(1, 2)

    return ref_output, attn_logits_ref


def benchmark(args):
    fuse_rope = args.fuse_rope
    fp8_gemm = args.fp8_gemm
    dtype = arg_to_torch_dtype[args.dtype]
    configs = []


    x_vals_list = [(32, 16, 2048, 512, 128, 64, 32)]
    x_names = ["B", "H", "S", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "num_kv_splits"]
    line_vals = ["ref", "persistent"]
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


        if "persistent" in provider:
            k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
            fn = lambda: ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, rope_fused=fuse_rope, use_fp8=fp8_gemm, device="cuda", persistent=True)

        if "ref" in provider:
            k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
            fn = lambda: ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, rope_fused=fuse_rope, use_fp8=fp8_gemm, device="cuda")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-fuse_rope", action='store_true', default=False, help="Test fusing rope inside kernel.")
    parser.add_argument("-fp8_gemm", action='store_true', default=False, help="Enable the fp8 gemm")
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
