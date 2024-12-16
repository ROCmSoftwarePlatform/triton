"""
Matrix Multiplication Tuning Scripts, Changed from the tutorial example "python/tutorials/03-matrix-multiplication.py"
"""

import torch

import triton
import triton.language as tl
import argparse
import sys
import yaml
import os
import subprocess

from matmul_kernel import matmul_kernel, gen_input


# global flag to indicate whether using the full tuing space
tuning_full_space = False


# pruned some unreasonable config
def prune_configs(configs, named_args, **kwargs):
    # call only for full tuning space
    if not tuning_full_space:
        return configs

    SIZE_M = named_args["a_ptr"].shape[0]
    SIZE_N = named_args["b_ptr"].shape[1]
    SIZE_K = named_args["a_ptr"].shape[1]

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K =\
            kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"]
        SPLIT_K = kw["SPLIT_K"]
        if SIZE_M <=32 and BLOCK_M != 32:
            continue
        if SIZE_N <=32 and BLOCK_N != 32:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(SIZE_M, SIZE_N, SIZE_K):
            continue
        pruned_configs.append(config)

    return pruned_configs


def get_full_tuning_configs(use_split_k):
    configs = []
    if not tuning_full_space:
        return configs

    block_mn_range = [32, 64, 128]
    block_k_range = [32, 64]
    split_k_range = [1, 2, 4, 5, 8, 10]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m, 'SPLIT_K': split_k}, num_stages=2, num_warps=num_warps))

    return configs


def get_default_tuning_configs():
    configs= [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'SPLIT_K': 1, 'matrix_instr_nonkdim': 16, 'waves_per_eu': 0}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 1, 'SPLIT_K': 8}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 1, 'SPLIT_K': 10}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 1, 'SPLIT_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 1, 'SPLIT_K': 10}, num_stages=2, num_warps=1),
    ]

    return configs

kernel_heuristic = triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})(matmul_kernel)

kernel = triton.autotune(
    configs= get_full_tuning_configs(True) if tuning_full_space else get_default_tuning_configs(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': prune_configs,
        'perf_model': None,
        "top_k": None
    },
    use_cuda_graph=True,
)(kernel_heuristic)


def matmul(a, b, c):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    grid_splitK = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K']
    )
    kernel[grid_splitK](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )


def test_correctness(M, N, K, datatype, col_a, col_b, device = 'cuda'):
    a, a_f16 = gen_input(M, K, datatype, col_a, 1, device)
    b, b_f16 = gen_input(K, N, datatype, col_b, 2, device)

    triton_output = torch.zeros((M, N), dtype=a.dtype, device=a.device)
    matmul(a, b, triton_output)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    rtol = 0 if torch.version.hip is None else 1e-2
    size_str = f'size, (M: {M}, N: {N}, K: {K})'
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print(f'✅ Triton and Torch match for {size_str}')
    else:
        print(f'❌ Triton and Torch differ for {size_str}')


def run_speed(M, N, K, datatype, col_a, col_b, use_rocprof, provider, device = 'cuda'):
    a, a_f16 = gen_input(M, K, datatype, col_a, 1, device)
    b, b_f16 = gen_input(K, N, datatype, col_b, 2, device)

    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c), quantiles=quantiles)
    return min_ms

def run_bash_command(commandstring):
    #print( commandstring )
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()


def parse_args(print_help=False):
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("--col_a", action='store_true', default=False, help='input A matrix is column-major format')
    parser.add_argument("--col_b", action='store_true', default=False, help='input B matrix is column-major format')
    parser.add_argument("-dtype", type=str, default='fp16', help="Input data type, default is fp16")
    parser.add_argument("--specify_type", action='store_true', default=False, help="Whether user specify data type, default false")
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    parser.add_argument("--rocprof", action='store_true', default=False, help='Use rocprof to measure kernel time, default uses do_bench()!')
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    args = parser.parse_args()

    if print_help:
        parser.print_help()
        return None

    return args

def main():
    args = parse_args()
    dtype = args.dtype
    use_rocprof = args.rocprof
    verbose = args.v
    col_a = args.col_a
    col_b = args.col_b

    mnks = []
    if args.gemm_size_file:
        matrix_size_file = args.gemm_size_file
        if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
            print(f"Matrix size file: {matrix_size_file} does not exist!")
            parse_args(True)
            sys.exit(1)

        with open(matrix_size_file) as file:
            matrix_sizes = yaml.safe_load(file)

        for sizes in matrix_sizes:
            M = sizes['M']
            N = sizes['N']
            K = sizes['K']
            mnks.append((M, N, K))
    else:
        M = args.m
        N = args.n
        K = args.k
        if M == 0 or N == 0 or K == 0:
            print(f"Input matrix size: (M {M}, N {N}, K {K}) contains dim size 0!")
            parse_args(True)
            sys.exit(1)
        mnks = [(M, N, K)]

    for (m, n, k) in mnks:
        min_ms = run_speed(m, n, k, dtype, col_a, col_b, use_rocprof, 'triton')

        # function to compute flops
        perf_flops = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

        if args.compare:
            test_correctness(m, n, k, dtype)
        best_config = kernel.best_config

        if use_rocprof:
            dtype_str = 'fp16' if (not args.specify_type) else args.dtype 
            block_m = best_config.kwargs['BLOCK_M']
            block_n = best_config.kwargs['BLOCK_N']
            block_k = best_config.kwargs['BLOCK_K']
            group_m = best_config.kwargs['GROUP_M']
            split_k = best_config.kwargs['SPLIT_K']
            num_warps = best_config.num_warps
            num_stages = best_config.num_stages
            driver = 'rocprof_matmul.py'
            TRITON_DIR = os.getenv('TRITON_DIR')
            if TRITON_DIR is not None:
                driver = os.path.join(TRITON_DIR, 'scripts/amd/gemm', driver)
            run_cmd = f'python {driver} -m {m} -n {n} -k {k} \
                        -block_m {block_m} -block_n {block_n} -block_k {block_k} \
                        -group_m {group_m} -split_k {split_k} -num_warps {num_warps} \
                        -num_stages {num_stages} -dtype {dtype_str}'
            if col_a:
                run_cmd = f'{run_cmd} -col_a'
            if col_b:
                run_cmd = f'{run_cmd} -col_b'

            prof_cmd = f'rocprof --stats {run_cmd}'
            run_bash_command(prof_cmd)

            parse_result_cmd = f'sed -n \'/matmul_kernel/p\' results.stats.csv | awk -F \',\' \'{{print $4}}\''
            parse_outputs = run_bash_command(parse_result_cmd)
            min_ms = int(parse_outputs[0]) / 1000000

        out_str = f'SIZE: {m},{n},{k} '
        # print best config
        if verbose:
            out_str += f'  best_config: ({best_config}),   '
        out_str += f'TFLOPS: {perf_flops(min_ms)} time(ns): {min_ms * 1000000}'
        print(out_str)


if __name__ == '__main__':
    sys.exit(main())
