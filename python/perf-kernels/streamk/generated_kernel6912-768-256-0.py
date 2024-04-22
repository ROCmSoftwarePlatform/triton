import torch
import triton
import triton.language as tl
import argparse
import sys
import multiprocessing
from tune_gemm import gen_input




@triton.jit
def matmul_kernel_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    if SPLIT_K == 1:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
    else:
        offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)



def matmul_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16(a, b, c, M, N, K, am, ak, bk, bn, cm, cn, warmup=False):
    grid = triton.cdiv(M, 64) * triton.cdiv(N, 64), 1
    #print(f'config: matmul_kernel_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16', flush=True)
    if warmup:
        matmul_kernel_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16.warmup(
            torch.float16, torch.float16, torch.float16,
            M, N, K,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = 64,
            BLOCK_SIZE_N = 64,
            BLOCK_SIZE_K = 64,
            GROUP_SIZE_M = 4,
            SPLIT_K = 1,
            num_warps = 4,
            num_stages = 0,
            waves_per_eu = 0,
            matrix_instr_nonkdim = 16,
            kpack = 2,
            grid=(1,)
        )
        return None
    else:
        matmul_kernel_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16[grid](
            a, b, c,
            M, N, K,
            am, ak, bk, bn, cm, cn,
            BLOCK_SIZE_M = 64,
            BLOCK_SIZE_N = 64,
            BLOCK_SIZE_K = 64,
            GROUP_SIZE_M = 4,
            SPLIT_K = 1,
            num_warps = 4,
            num_stages = 0,
            waves_per_eu = 0,
            matrix_instr_nonkdim = 16,
            kpack = 2
        )
        return c

def try_config_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16(M, N, K, am, ak, bk, bn, cm, cn):
    try:
        matmul_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16(None, None, None, M, N, K, am, ak, bk, bn, cm, cn, True)
        return True
    except Exception as e:
        print(f'invalid config(compilation): M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16: ', e, flush=True)
        return False

def test_gemm(M, N, K, num_threads):
    thread_pool = multiprocessing.Pool(processes=num_threads)
    a, a_fp16 = gen_input(M, K, 'fp16', False, 1, 'randn', device='cuda')
    b, b_fp16 = gen_input(K, N, 'fp16', True, 2, 'randn', device='cuda')
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    task_args = (M, N, K,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1))

    if num_threads > 1:
        results = []
        config_names = []

        results += [thread_pool.apply_async(try_config_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16, args=task_args)]
        config_names += ['M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16']

        failed_configs = []
        for i in range(len(results)):
            results[i].wait()
            res = results[i].get()
            if not res:
                failed_configs += [config_names[i]]
        thread_pool.close()
        thread_pool.join()
        with open("generated_kernel6912-768-256-0.py.failed_configs", "w") as f:
            for cfg in failed_configs:
                f.write(cfg + "\n")
    else:
        try:
            with open("generated_kernel6912-768-256-0.py.failed_configs", "r") as f:
                failed_configs = [cfg.strip() for cfg in f.readlines()]
        except Exception:
            failed_configs = []
        
        if 'M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16' not in failed_configs:
            for i in range(1000):
                d = matmul_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16(a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        return d

def main():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,)
    parser.add_argument("-n", type=int, default=1, help='number of threads')
    args = parser.parse_args()
    numThreads = args.n
    test_gemm(6912, 768, 256, numThreads)

    M=6912
    N=768
    K=256
    a, a_fp16 = gen_input(M, K, 'fp16', False, 1, 'randn', device='cuda')
    b, b_fp16 = gen_input(K, N, 'fp16', True, 2, 'randn', device='cuda')
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    am=a.stride(0)
    ak=a.stride(1)
    bn=b.stride(1)
    bk=b.stride(0)
    cm=c.stride(0)
    cn=c.stride(1)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda:  matmul_M6912_N768_K256_BM64_BN64_BK64_GM4_SK1_nW4_nS0_EU0_kP2_mfma16(a, b, c, M, N, K, am, ak, bk, bn, cm, cn, warmup=False), quantiles=quantiles)

    perf_flops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    out_str = f'TFLOPS: {perf_flops(min_ms)} time(ns): {min_ms * 1000000}'
    print(out_str)

if __name__ == '__main__':
   sys.exit(main())
