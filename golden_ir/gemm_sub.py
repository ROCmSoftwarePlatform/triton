#!/opt/conda/envs/py_3.10/bin/python3

# Taken from Ravil
# https://github.com/ROCm/triton/blob/ravil/kernel-sub/python/perf-kernels/tools/kernel-sub/gemm-ex.py

import torch
import triton
import triton.language as tl
import time
import argparse
import os
import yaml
from pathlib import Path
from triton._C.libtriton import ir


from triton.runtime import driver
from triton.backends import backends
from triton.compiler.compiler import *


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", action='store_true', help='run benchmark')
parser.add_argument("-t", "--type", choices=['fp16', 'fp32'],
                    default="fp16",
                    help="data type under the test")
parser.add_argument("--dump-ir", choices=['ttir', 'ttgir','llir', 'amdgcn'],
                    default="amdgcn",
                    help="dump IR format")
parser.add_argument("-p", "--prefix", type=str, default=None, help="prefix for the dumped files")
parser.add_argument("-f", "--file", type=str, default=None, help="load gemm parameters from file")
parser.add_argument("-s", "--sub", type=str, default=None, help="substitution file")
parser.add_argument("-m", "--use-mask", action='store_true', help='use masked load/store')
parser.add_argument("--trans-b", action='store_true', help='tanspose B matrix')
parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
args = parser.parse_args()

curr_dir = os.path.dirname(os.path.abspath(__file__))

perf = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

print(f"MASKING load/store: {'enabled' if args.use_mask else 'disabled'}")
print(f"MATRIX B TRANSPOSED: {'true' if args.trans_b else 'false'}")

gemm_config = None
if args.file:
    # read GEMM config from a file
    if not os.path.exists(args.file):
        raise RuntimeError(f'cannot open `{args.file}`')
    with open(args.file, 'r') as file:
        gemm_config = yaml.safe_load(file)
else:
   raise RuntimeError('no config file has been provided')


assert(gemm_config != None)
if not 'tuning' in gemm_config.keys():
   raise RuntimeError('tuning field is not provided in the config')
tuning_config = gemm_config['tuning']


sub_file = None
if args.sub:
  if not os.path.exists(args.sub):
    raise RuntimeError(f'cannot open `{args.sub}`')
  with open(args.sub, 'r') as file:
    sub_file = file.readlines()
    sub_file = ''.join(sub_file)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,
                  M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, stride_bias,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  SPLIT_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr,
                  BIAS: tl.constexpr,
                  EVEN_K: tl.constexpr,
                  GRID_MN: tl.constexpr,
                  NUM_XCDS: tl.constexpr,
                  USE_MASK: tl.constexpr
                  ):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_bias > 0)

    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = GRID_MN // NUM_XCDS
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        pid = xcd * pids_per_xcd + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if BIAS:
        bias_ptrs = bias_ptr + offs_am * stride_bias
        bias = tl.load(bias_ptrs, mask=offs_am < M if USE_MASK else None, other=0.0)
        #bias = tl.load(bias_ptrs, mask=None, other=0.0)
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in tl.range(0, tl.cdiv((K-256), BLOCK_SIZE_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    c = accumulator.to(c_ptr.type.element_ty)
    if BIAS:
        c += bias[:, None]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask if USE_MASK else None)
        #tl.store(c_ptrs, c, mask=None)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask if USE_MASK else None)
        #tl.atomic_add(c_ptrs, c, mask=None)


runner = None
kernel_args = None
kernel_kwargs = None


def generate_kerenel_args(a, b, c, bias, use_bias=False):
  M, K = a.shape
  K, N = b.shape
  stride_bias = N

  kernel_args = (a, b, c, bias,
                 M, N, K,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1),
                 stride_bias)

  kernel_kwargs = {"BLOCK_SIZE_M": tuning_config["BLOCK_SIZE_M"],
                   "BLOCK_SIZE_N": tuning_config["BLOCK_SIZE_N"],
                   "BLOCK_SIZE_K": tuning_config["BLOCK_SIZE_K"],
                   "GROUP_SIZE_M": tuning_config["GROUP_SIZE_M"],
                   "SPLIT_K" : 1,
                   "EVEN_K": True,
                   "NUM_XCDS": 8,
                   "GRID_MN": triton.cdiv(M, tuning_config["BLOCK_SIZE_M"]) * triton.cdiv(N, tuning_config["BLOCK_SIZE_N"]),
                   "BIAS" : use_bias,
                   "USE_MASK": True if args.use_mask else False,
                   "waves_per_eu": tuning_config["waves_per_eu"],
                   "num_warps": tuning_config["num_warps"],
                   "num_stages": tuning_config["num_stages"],
                   "num_ctas": tuning_config["num_ctas"],
                   "instruction_sched_variant": 0}

  if tuning_config:
    if tuning_config['matrix_instr_nonkdim']:
      kernel_kwargs['matrix_instr_nonkdim'] = tuning_config['matrix_instr_nonkdim']
    if tuning_config['kpack']:
      kernel_kwargs['kpack'] = tuning_config['kpack']
    if tuning_config['instruction_sched_variant']:
      kernel_kwargs['instruction_sched_variant'] = tuning_config['instruction_sched_variant']

  return kernel_args, kernel_kwargs


def matmul(a, b, c, bias, use_bias=False):
  M, _ = a.shape
  _, N = b.shape

  grid = (triton.cdiv(M, tuning_config['BLOCK_SIZE_M']) * triton.cdiv(N, tuning_config['BLOCK_SIZE_N']), 1, 1)

  global runner
  global kernel_args
  global kernel_kwargs

  runner = matmul_kernel[grid]
  kernel_args, kernel_kwargs = generate_kerenel_args(a, b, c, bias, use_bias)

  handle = runner(*kernel_args, **kernel_kwargs)

  if args.sub:
    path = Path(args.sub)
    target = driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options(
       {"num_warps": tuning_config["num_warps"],
        "waves_per_eu": tuning_config["waves_per_eu"],
        "num_ctas": tuning_config["num_ctas"],
        "warp_size": 64}
    )
    metadata = handle.metadata._asdict()
    #options.num_warps = int(tuning_config["num_warps"])

    enabled_next_stage = False
    llir_src = None
    if path.suffix == ".ttgir":
      print("Found TTGIR: ", path)
      context = ir.context()
      ir.load_dialects(context)
      backend.load_dialects(context)
      mod = ir.parse_mlir_module(args.sub, context)
      mod.context = context
      llir_src = backend.make_llir(mod, metadata, options)
      handle.asm['llir'] = llir_src
      enabled_next_stage = True

    amdgcn_src = None
    if path.suffix == ".llir" or enabled_next_stage:
      print("Found LLIR: ", path)
      llir_src = llir_src if llir_src else sub_file
      amdgcn_src = backend.make_amdgcn(llir_src, metadata, options)
      enabled_next_stage = True

    amdgcn_src = amdgcn_src if amdgcn_src else sub_file
    handle.asm['amdgcn'] = amdgcn_src
    hasco_src = backend.make_hsaco(amdgcn_src, metadata, options)
    handle.kernel = hasco_src
    handle.module = None
    def substituted_kernel(*args, **kwargs):
      handle[grid](*args)
    runner = substituted_kernel

  if args.verbose:
    print(handle.asm.keys())

  print(handle.metadata.name)

  if args.dump_ir:
    filename = f"matmul_kernel.{args.dump_ir}"
    filename = f"{args.prefix}-{filename}" if args.prefix else filename
    with open(os.path.join(curr_dir, filename), "w") as file:
       file.write(handle.asm[args.dump_ir])


def generate_tensor(M, N, K, tested_dtype):
  a = torch.randn((M, K), device='cuda', dtype=tested_dtype)

  if args.trans_b:
    b = torch.randn((N, K), device='cuda', dtype=tested_dtype)
    b = b.T
  else:
    b = torch.randn((K, N), device='cuda', dtype=tested_dtype)

  c = torch.empty((M, N), device='cuda', dtype=tested_dtype)
  bias = torch.empty((M, N), device='cuda', dtype=tested_dtype)
  return a, b, c, bias


M = gemm_config['M']
N = gemm_config['N']
K = gemm_config['K']

if args.type == 'fp16':
    tested_dtype=torch.float16
elif args.type == 'fp32':
    tested_dtype=torch.float32
else:
    raise RuntimeError(f'`{args.type}` unsupported data type')

a, b, c, bias = generate_tensor(M, N, K, tested_dtype)

use_bias = False
matmul(a, b, c, bias, use_bias)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
#num_repeats = 300
num_repeats = 1
for _ in range(num_repeats):
  runner(*kernel_args, **kernel_kwargs)
end_event.record()
torch.cuda.synchronize()
estimate_ms = start_event.elapsed_time(end_event) / num_repeats
tflops = perf(estimate_ms, M, N, K)
print(f'perf: {tflops} TFLOP/s')


a, b, c, bias = generate_tensor(M, N, K, tested_dtype)
kernel_args, kernel_kwargs = generate_kerenel_args(a, b, c, bias, use_bias)
runner(*kernel_args, **kernel_kwargs)


torch_output = torch.matmul(a, b)
if torch.allclose(c, torch_output, atol=0.125, rtol=0.01):
  print("✅ Triton and Torch match")
else:
  print("❌ Triton and Torch differ")
  print("TORCH:")
  print(torch_output.cpu())
  print("-"*80)
  print("TRITON:")
  print(c.cpu())
