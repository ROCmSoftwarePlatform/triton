import argparse
import sys
import yaml
import os
import pandas as pd

from utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-dtype_a",
                        type=str,
                        default='fp16',
                        help="matrix a element data type")
    parser.add_argument("-dtype_b",
                        type=str,
                        default='fp16',
                        help="matrix b element data type")
    parser.add_argument("-dtype_c",
                        type=str,
                        default='fp16',
                        help="output element data type")
    parser.add_argument("--gemm_size_file",
                        type=str,
                        default="",
                        help='yaml file to indicate matrix size')
    parser.add_argument("--triton_result",
                        type=str,
                        default="",
                        help='yaml file to load (for benchmarking) or '
                        'store (for tuning) triton results')
    parser.add_argument("--tune_hipblaslt",
                        action='store_true',
                        default=False,
                        help='Run tuning with hipblaslt')
    parser.add_argument("--tune_triton",
                        action='store_true',
                        default=False,
                        help='Run tuning with triton')
    parser.add_argument("--hipblaslt_result",
                        type=str,
                        default="",
                        help='csv file to load (if not tuning hipblaslt) or '
                        'store (if tuning hipblaslt) hipblaslt tuning results')

    args = parser.parse_args()
    return args


def run_hipblaslt_bench(hipblaslt_bench, M, N, K, transA, transB, dtype):
    ITER = 10
    WARMUP = 100
    dtype = 'f16_r' if dtype == "fp16" else 'f8_r'
    hipBLASLt_bench_args = f"-f matmul -r {dtype} -m {M} -n {N} -k {K}"
    hipBLASLt_bench_args += f" --transA {transA} --transB {transB}"
    hipBLASLt_bench_args += f" --compute_type f32_r --algo_method all"
    hipBLASLt_bench_args += f" -i {ITER} -j {WARMUP} --print_kernel_info"
    SED_WINNER = "sed -n '/Winner:/, $p'"

    print(f"Tuning hipblaslt with {hipBLASLt_bench_args}", flush=True)

    winner = run_bash_command(
        f"HIP_FORCE_DEV_KERNARG=1 {hipblaslt_bench} {hipBLASLt_bench_args} | {SED_WINNER}"
    )

    for line in winner:
        line = line.decode('utf-8')

        if "Solution index" in line:
            winner_index = int(line.split(':', 1)[1].strip())
        if "kernel name" in line:
            kernel_name = line.split(':', 1)[1].strip()
        if f"{M},{N},{K}" in line:
            tflops = int(line.split(',')[-2].strip()) / 1000
            us = float(line.split(',')[-1].strip())

    return winner_index, kernel_name, tflops, us


def run_triton_tuning(input, output, dtype_a):
    print(f"Tuning gemm sizes from {input} with Triton", flush=True)
    run_bash_command(
        f"./tune_gemm.py --gemm_size_file {input} -dtype_a {dtype_a} -dtype_b {dtype_a} --ngpus 8 --jobs 32 --o {output} --rotating_tensor 512",
        False)


def run_triton_bench(input, dtype_a):
    if not os.path.exists(input):
        print(f"{input} does not exist, please run tuning first")
        sys.exit(1)
    print(f"Benchmarking gemms from {input} with Triton", flush=True)
    triton_output = run_bash_command(
        f"./tune_gemm.py --gemm_size_file {input} -dtype_a {dtype_a} -dtype_b {dtype_a} --benchmark --rotating_tensor 512"
    )

    data = []
    for line in triton_output:
        line = line.decode('utf-8')

        if "Benchmarking" in line or "trans" in line:
            continue

        items = line.split()
        trans = items[0].strip()
        M = items[1].strip()
        N = items[2].strip()
        K = items[3].strip()
        tflops = items[4].strip()
        us = items[5].strip()

        data.append([trans, int(M), int(N), int(K), float(tflops), float(us)])

    return pd.DataFrame(data, columns=['trans', 'M', 'N', 'K', 'tflops', 'us'])


def main():
    args = parse_args()
    gemm_size_file = args.gemm_size_file
    hipblaslt_csv = args.hipblaslt_result
    triton_yaml = args.triton_result

    if not gemm_size_file:
        print("Need to provide gemm size file: --i filename")
        sys.exit(1)

    if not triton_yaml:
        print(
            "Need to provide triton result filename: --triton_result filename.yaml"
        )
        sys.exit(1)

    if not hipblaslt_csv:
        print(
            "Need to provide hipblaslt result filename: --hipblaslt_result filename.csv"
        )
        sys.exit(1)

    # Get element type
    dtype_a = args.dtype_a
    dtype_b = args.dtype_b
    dtype_c = args.dtype_c

    if args.tune_triton:
        run_triton_tuning(gemm_size_file, triton_yaml, dtype_a)

    df_triton = run_triton_bench(triton_yaml, dtype_a)

    if args.tune_hipblaslt:
        with open(gemm_size_file) as inFile:
            matrix_sizes = yaml.safe_load(inFile)

        mnks = []
        for item in matrix_sizes:
            M = item['M']
            N = item['N']
            K = item['K']
            transA = item['rowMajorA']
            transB = item['rowMajorB']
            mnks.append((M, N, K, transA, transB))

        hipblaslt_ROOT_DIR = os.environ.get('HIPBLASLT_ROOT')
        if not hipblaslt_ROOT_DIR:
            print("Need to provide hipblaslt root dir: HIPBLASLT_ROOT")
            sys.exit(1)
        hipblaslt_bench = os.path.join(hipblaslt_ROOT_DIR,
                                       "build/clients/staging",
                                       "hipblaslt-bench")

        hipblaslt_data = []

        for (M, N, K, transA, transB) in mnks:
            if not (transA == 'T' and transB == 'N'):
                ## It seems hipblaslt does not support TT case?
                continue
            winner_index, kernel_name, tflops, us = run_hipblaslt_bench(
                hipblaslt_bench, M, N, K, transA, transB, dtype_a)
            hipblaslt_data.append([
                f"{transA}{transB}", M, N, K, tflops, us, winner_index,
                kernel_name
            ])

        df_hipblaslt = pd.DataFrame(hipblaslt_data,
                                    columns=[
                                        'trans', 'M', 'N', 'K', 'tflops', 'us',
                                        'winner_idx', 'kernel_name'
                                    ])
        df_hipblaslt.to_csv(hipblaslt_csv, index=False)
    else:
        if not os.path.exists(hipblaslt_csv):
            print(f"{hipblaslt_csv} does not exist, please run tuning first")
            sys.exit(1)
        df_hipblaslt = pd.read_csv(hipblaslt_csv)

    df_merged = pd.merge(df_triton,
                         df_hipblaslt,
                         on=['trans', 'M', 'N', 'K'],
                         how='left',
                         suffixes=('_triton', '_hipblaslt'))


    print(df_merged[[
        'trans', 'M', 'N', 'K', 'tflops_triton', 'tflops_hipblaslt'
    ]])


if __name__ == '__main__':
    sys.exit(main())
