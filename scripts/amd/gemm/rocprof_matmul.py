#!/usr/bin/env python3
import argparse
import sys

import torch
import triton
import triton.language as tl

from matmul_kernel import matmul_kernel, gen_input


kernel = triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})(matmul_kernel)


def matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    _, N = b.shape

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
        BLOCK_M = block_m, 
        BLOCK_N = block_n, 
        BLOCK_K = block_k,
        GROUP_M = group_m,
        SPLIT_K = split_k,
        num_warps = num_warps,
        num_stages = num_stages,
    )


def test_gemm(M, N, K, col_a, col_b, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, dtype):
    a, a_f16 = gen_input(M, K, dtype, col_a, 1, 'cuda')
    b, b_f16 = gen_input(K, N, dtype, col_b, 2, 'cuda')

    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    matmul(a, b, c, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages)

    return c


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="test gemm tuning",
        description="Tuning infra for triton gemm",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-col_a", action='store_true', default=False, help='input A matrix is column-major format')
    parser.add_argument("-col_b", action='store_true', default=False, help='input B matrix is column-major format')
    parser.add_argument("-block_m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-block_n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-block_k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-group_m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-split_k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-num_warps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-num_stages", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-dtype", type=str, default='fp16', help="Input/output data type")
    parsed_args = parser.parse_args(args)

    dtype = parsed_args.dtype

    M = parsed_args.m
    N = parsed_args.n
    K = parsed_args.k
    block_m = parsed_args.block_m
    block_n = parsed_args.block_n
    block_k = parsed_args.block_k
    group_m = parsed_args.group_m
    split_k = parsed_args.split_k
    num_warps = parsed_args.num_warps
    num_stages = parsed_args.num_stages
    col_a = parsed_args.col_a
    col_b = parsed_args.col_b

    test_gemm(M, N, K, col_a, col_b, block_m, block_n, block_k, group_m, split_k, num_warps, num_stages, dtype)


if __name__ == '__main__':
    sys.exit(main())
