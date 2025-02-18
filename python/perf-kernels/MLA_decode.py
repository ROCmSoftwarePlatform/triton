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
        from utils.sglang_ref import _decode_grouped_persistent_att_m_fwd as decode_attention_fwd
    else:
        from utils.sglang_ref import _decode_grouped_att_m_fwd as decode_attention_fwd

    B, H = q_input.shape[0], q_input.shape[1]
    kv_lora_rank = v_input.shape[-1]
    device = q_input.device

    attn_logits = torch.empty(B, H, num_kv_splits, kv_lora_rank + 1, dtype=q_input.dtype, device=device)
    decode_attention_fwd(q_input, k_input, v_input, attn_logits, Req_to_tokens, B_req_idx, B_Seqlen,
                                    num_kv_splits, sm_scale, logit_cap)
    
    return attn_logits


def input_helper(B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device, rope_base=10, rope_max_seq_len=16324, rope_scaling=1.0):
    Req_to_tokens = torch.arange(B * S, device=device).reshape(B, S)
    B_req_idx = torch.arange(B, device=device)
    B_Seqlen = torch.full((B, ), S, device=device)

    q = torch.randn(B, H, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device) # w_kc fused
    kv_cache = torch.randn(B * S, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device)

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
@pytest.mark.parametrize('dtype', [torch.bfloat16])
def test_op_fwd(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, dtype, num_kv_splits=32, sm_scale=1.0, logit_cap=0.0,
                device="cuda"):
    torch.manual_seed(0)

    D = qk_nope_head_dim
    Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions = input_helper(
        B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device)

    k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)

    tri_logits = ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, device="cuda", persistent=True)

    # reference
    ref_logits = ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, device="cuda")

    print("first 10 logits:")
    print(f"ref: {ref_logits[:,:,-1].flatten()[:]}") # to debug the rope, check last split
    print(f"tri: {tri_logits[:,:,-1].flatten()[:]}")
    torch.testing.assert_close(ref_logits, tri_logits, atol=1e-2, rtol=1e-2)
    print("attn_logits from stage 1 matches with ref")
    # stage 2 is shared

def ref_preprocess(kv_cache, kv_lora_rank):
    latent_cache = kv_cache
    v_input = latent_cache[..., :kv_lora_rank]
    v_input = v_input.contiguous().unsqueeze(1)
    k_input = latent_cache.unsqueeze(1)
    k_input[..., :kv_lora_rank] = v_input
    return k_input, v_input

def ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale, logit_cap, rotary_emb, positions, device="cuda", persistent=False):
    q_input = q
    attn_logits = attn_mqa(q_input, k_input, v_input, Req_to_tokens, B_req_idx, B_Seqlen,
                                            num_kv_splits, sm_scale, logit_cap, persistent=persistent)
    return attn_logits


def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []
    x_vals_list = [
                    (1, 16, 2048, 512, 128, 64, 128),
                    (2, 16, 2048, 512, 128, 64, 64), 
                    (4, 16, 2048, 512, 128, 64, 64),
                    (8, 16, 2048, 512, 128, 64, 64),
                    (16, 16, 2048, 512, 128, 64, 32),
                    (32, 16, 2048, 512, 128, 64, 32),
                    (64, 16, 2048, 512, 128, 64, 32),
                    (128, 16, 2048, 512, 128, 64, 32),
                    ]
    x_names = ["B", "H", "S", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "num_kv_splits"]

    line_vals = [ "persistent", "ref"] 
    if args.persistent:
        line_vals = ["persistent"]
    elif args.ref:
        line_vals = ["ref"]

    plot_name = "MLA-decode"

    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=line_vals,
                                 line_names=line_vals, styles=[('red', '-'), ('green', '-')], ylabel='ms',
                                 plot_name=plot_name, args={'sm_scale': 1.0, 'logit_cap': 0.0, 'device': args.device}))

    @triton.testing.perf_report(configs)
    def bench_MLA(B, H, S, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, num_kv_splits, sm_scale, logit_cap, device,
                  provider):
        warmup = 25
        rep = 100

        D = qk_nope_head_dim

        Req_to_tokens, B_req_idx, B_Seqlen, q, kv_cache, att_out, attn_logits, w_kc, w_vc, rotary_dim, rotary_emb, positions = input_helper(
            B, H, S, D, kv_lora_rank, qk_rope_head_dim, num_kv_splits, dtype, device)

        if "persistent" in provider:
            k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
            fn = lambda: ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, device="cuda", persistent=True)

        if "ref" in provider:
            k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
            fn = lambda: ref_compute(q, k_input, v_input, w_kc, w_vc, Req_to_tokens, B_req_idx, B_Seqlen, num_kv_splits, sm_scale,
                                  logit_cap, rotary_emb, positions, device="cuda")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path=".", print_data=True, show_plots=False)
    return x_vals_list, x_names, line_vals



arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype", default='bf16')
    parser.add_argument("-device", default='cuda')
    parser.add_argument("-persistent", action="store_true", default=False)
    parser.add_argument("-ref", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    return parser.parse_args()

arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

import re
from prettytable import PrettyTable

def parse_vgpr_usage(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Extract VGPR-related information
    vgpr_info = []
    table_lines = []
    in_table = False

    for line in lines:
        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())
        # Detect start of table
        if re.match(r"^\s*MLA-decode:", line):
            in_table = True
            # table_lines.append(line.strip())
        elif in_table:
            table_lines.append(line.strip())

    # Print extracted information
    print("\n".join(vgpr_info))

    table = PrettyTable()
    table.field_names = table_lines[0].split()
    [table.add_row(line.split()[1:]) for line in table_lines[1:]]

    print(table)


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)

import sys
import time
import re
import os
import tempfile

def print_vgpr(args):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        output_file = temp_file.name

        # Redirect stdout and stderr to the temporary file
        sys.stdout = temp_file
        sys.stderr = temp_file
        
        os.environ["AMDGCN_ENABLE_DUMP"] = "1"
        # os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        run_bench(args)  # Run the benchmark
        
        sys.stdout.flush()
        sys.stderr.flush()

    # Restore stdout and stderr to normal
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    time.sleep(0.5)  # Ensure everything is written before reading

    # Parse and print relevant output
    parse_vgpr_usage(output_file)

    # Remove the temporary file
    os.unlink(output_file)

def main():
    args = parse_args()
    if args.print_vgpr:
        print_vgpr(args)
        return 0
    
    run_bench(args)


if __name__ == "__main__":
    main()