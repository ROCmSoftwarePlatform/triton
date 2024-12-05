import torch

import triton
import triton.language as tl
import sys
import argparse
import pytest
import re


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 32, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
    use_cuda_graph=True,
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
})
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    scale,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
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

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    if APPLY_SCALE:
        accumulator = accumulator * scale
    c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, c, a_scale, b_scale, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    apply_scale = a_scale is not None and b_scale is not None
    scale = a_scale * b_scale if apply_scale else None
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        scale,
        APPLY_SCALE=apply_scale,
        ACTIVATION=activation,
    )


name_to_torch_types = {
    'int8': torch.int8,
    'int32': torch.int32,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp8e5': torch.float8_e5m2fnuz,
    'fp8e4': torch.float8_e4m3fnuz,
}

dtype_max = {
    torch.float8_e5m2fnuz: 57344,
    torch.float8_e4m3fnuz: 240,
    torch.int8: 127,
}

def dtype_is_8_bit(dtype):
    return (dtype is torch.float8_e5m2fnuz) or \
           (dtype is torch.float8_e4m3fnuz) or \
           (dtype is torch.int8)

def gen_input(M, N, dtype, needTrans, seed, device='cuda'):
    torch.manual_seed(seed)

    if needTrans:
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T
    else:
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    scale = None
    if dtype_is_8_bit(dtype):
        max_val = torch.max(torch.abs(raw_data))
        scale = max_val / dtype_max[dtype]
        raw_data = raw_data / scale
    
    input = raw_data.to(dtype)
    input_f32 = input.to(torch.float32)

    return input, input_f32, scale


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., rocBLAS).
def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]

    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536)]
    # x_vals = [(1024, 1024, 1024)]

    return x_vals


@pytest.mark.parametrize("M, N, K, in_dtype, out_dtype, col_a, col_b", [
    (*shape, in_dtype, out_dtype, col_a, col_b)
    for shape in get_x_vals()
    for in_dtype, out_dtype in [('fp16', 'fp16'), 
                                ('bf16', 'bf16'), 
                                ('fp32','fp32'), 
                                ('fp8e4','fp16'), ('fp8e5', 'fp16'),
                                ('int8', 'int8'),
                                ('int8', 'int32')
                                ]
    # Only test k-major tensors because
    # 1. This is the most preformant config and the current focus
    # 2. Other case does not work with num_stages=0 (TODO (zhanglx))
    for col_a in [True, False]
    for col_b in [True, False]
])
def test_correctness(M, N, K, col_a, col_b, in_dtype, out_dtype):
    torch_in_dtype = name_to_torch_types[in_dtype]
    a, a_fp32, a_scale = gen_input(M, K, torch_in_dtype, col_a, 1, device='cuda')
    b, b_fp32, b_scale = gen_input(K, N, torch_in_dtype, col_b, 2, device='cuda')
    # Allocates output.
    torch_out_dtype = name_to_torch_types[out_dtype]
    c = torch.empty((M, N), device=a.device, dtype=torch_out_dtype)
    if dtype_is_8_bit(torch_in_dtype):
        matmul(a, b, c, a_scale.item(), b_scale.item(), activation="")
        torch_output = torch.matmul(a_fp32, b_fp32)
        torch_output = torch_output * a_scale * b_scale
    else:
        matmul(a, b, c, a_scale=None, b_scale=None, activation="")
        torch_output = torch.matmul(a.to(torch_in_dtype), b.to(torch_in_dtype))
    if out_dtype == 'int8':
        torch.testing.assert_close(c.to(torch.float32), torch_output.to(torch.int8).to(torch.float32), atol=1e-3, rtol=1e-2)
    else:
        torch.testing.assert_close(c, torch_output.to(torch_out_dtype), atol=5e-3, rtol=1e-2)


# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.


def get_type(provider):
    res = re.findall(r'\(.*?\)', provider)
    return res[0][1:-1]


inout_dtype = {
    'int8': torch.int8,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp8e4': torch.float16,
    'fp8e5': torch.float16,
}


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=get_x_vals(),
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=[
            'rocblas(fp16)', 'rocblas(bf16)', 'triton(fp16)', 'triton(bf16)', 'triton(int8)', 'triton(fp8e4)',
            'triton(fp8e5)'
        ],
        # Label name for the lines
        line_names=[
            "rocBLAS.Fp16", "rocBLAS.Bf16", "Triton.Fp16", "Triton.Bf16", "Triton.Int8", "Triton.Fp8E4", "Triton.Fp8E5"
        ],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    in_dtype = name_to_torch_types[get_type(provider)]
    out_dtype = in_dtype

    quantiles = [0.5, 0.2, 0.8]
    if 'rocblas' in provider:
        a = torch.randn((M, K), dtype=in_dtype, device='cuda')
        b = torch.randn((K, N), dtype=in_dtype, device='cuda')

        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    else:  # triton, different data types
        assert "triton" in provider
        a, _, a_scale = gen_input(M, K, in_dtype, False, 1, device='cuda')
        b, _, b_scale = gen_input(K, N, in_dtype, True, 2, device='cuda')
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=out_dtype)

        if dtype_is_8_bit(in_dtype):
            a_scale = a_scale.item()
            b_scale = b_scale.item()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c, a_scale, b_scale, activation=""), quantiles=quantiles)
        global verbose
        if verbose:
            print(f'SIZE: {M},{N},{K}   Best tuning config: ({matmul_kernel.best_config()})')
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="GEMM tutorial example",
        allow_abbrev=False,
    )

    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    args = parser.parse_args()

    return args


def main():
    # assign to a global verbose var to indicate whether print
    # best tuning config
    global verbose
    args = parse_args()
    verbose = args.v
    benchmark.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    sys.exit(main())
