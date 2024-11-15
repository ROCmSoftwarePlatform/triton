#!/opt/conda/envs/py_3.10/bin/python

import argparse
import torch
import sys
import pytest

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=16, num_stages=1),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

@triton.jit
def rms_kernel(output_ptr, input_ptr, g_ptr, input_row_stride, output_row_stride, n_cols, epsilon,
               BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # Each program instance handles one row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    sum_squares = tl.zeros([1], dtype=tl.float32)

    # Pointers to the start of the row
    row_input_ptr = input_ptr + row_idx * input_row_stride
    row_output_ptr = output_ptr + row_idx * output_row_stride

    # Accumulate sum of squares
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + col_offsets
        mask = cols < n_cols
        input_ptrs = row_input_ptr + cols
        input_ptrs = tl.multiple_of(input_ptrs, (16, ))
#        row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        sum_squares += tl.sum(x * x, axis=0)

    # Compute normalization factor
    mean_square = sum_squares / n_cols
    norm_factor = tl.rsqrt(mean_square + epsilon)

    # Normalize and write output
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + col_offsets
        mask = cols < n_cols
        input_ptrs = row_input_ptr + cols
        input_ptrs = tl.multiple_of(input_ptrs, (16, ))
        g_ptrs = g_ptr + cols
        output_ptrs = row_output_ptr + cols
#        output_ptrs = tl.multiple_of(output_ptrs, (16, ))
        x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
#        x = tl.load(input_ptrs, mask=mask, other=0.0)
        g = tl.load(g_ptrs, mask=mask, other=0.0)
        rms_norm = x * norm_factor * g
        tl.store(output_ptrs, rms_norm, mask=mask)

def triton_rmsnorm(x, g, epsilon=1e-6):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 65536
#    BLOCK_SIZE = 32768
#    BLOCK_SIZE = 16384

    y = torch.empty_like(x, device=x.device)

    grid = (n_rows,)
    kk=rms_kernel[grid](
        y, x, g, x.stride(0), y.stride(0), n_cols, epsilon, BLOCK_SIZE,
        num_warps=8, num_stages=2
    )

    return y

def torch_rmsnorm(x, g):
    M, N = x.shape
    if hasattr(torch.nn, 'RMSNorm'):
        rms_norm = torch.nn.RMSNorm(N, device='cuda')
        return rms_norm(x)
    else:
        rms = torch.sqrt(torch.sum(x * x, dim=-1) * 1/N)
        rms_norm = torch.div(x, rms.unsqueeze(1).repeat(1, N)) * g
        return rms_norm


@pytest.mark.parametrize('M, N', [
    (1, 4),
    (2, 10),
    (8192, 4096),
    (4096, 8192),
    (1, 8192),
    (873, 1245),
])
def test_rmsnorm(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    g = torch.ones((1, N), device='cuda')
    y_triton = triton_rmsnorm(x, g)

    y_torch = torch_rmsnorm(x, g)

    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def run_benchmark(args):
    config = []
    if (args.M_benchmark):
        val = args.M_start
        x_vals_list = []
        while val <= args.M_end:
            x_vals_list.append(val)
            val *= args.M_step
        mn_args = {'N': args.N_start}
        plot_name = str("rmsnorm-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                        "-" + str(args.M_end) + "-" + str(args.M_step))
        x_names = ['M']
    else:
        x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
        mn_args = {'M': args.M_start}
        x_names = ['N']
        plot_name = str("rmsnorm-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                        "-" + str(args.N_end) + "-" + str(args.N_step))

    dtype = arg_to_torch_dtype[args.dtype]

    print(plot_name)
    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=["Triton", "Torch"],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name=plot_name,
            args=mn_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        g = torch.ones((1, N), device='cuda')
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, g))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: triton_rmsnorm(x, g))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RMSNorm",
        allow_abbrev=False,
    )

    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)  #This is multiplicative step
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="8192", type=int)
    parser.add_argument('-Ns', "--N_step", default="1024", type=int)
    parser.add_argument('-Ne', "--N_end", default="32768", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_benchmark:
        x = torch.randn(args.M_start, args.N_start, device='cuda')
        g = torch.ones((1, args.N_start), device='cuda')
        triton_rmsnorm(x, g)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
