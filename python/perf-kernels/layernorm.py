import argparse
import sys
import pytest

import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=1),
        triton.Config({'waves_per_eu': 2}, num_warps=1),
        triton.Config({'waves_per_eu': 2}, num_warps=2),
        triton.Config({'waves_per_eu': 1}, num_warps=4),
        triton.Config({'waves_per_eu': 2}, num_warps=4),
        triton.Config({'waves_per_eu': 2}, num_warps=8),
    ]


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def layernorm_kernel_blocked_impl(x_ptr, y_ptr, w_ptr, b_ptr, x_row_stride, y_row_stride, n_rows, n_cols, eps,
                                  BLOCK_SIZE: tl.constexpr):

    tl.assume(x_row_stride > 0)
    tl.assume(y_row_stride > 0)
    #program id
    row = tl.program_id(0)
    tl.assume(row > 0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    #calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in tl.range(0, loop_num_l, num_stages=3):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  #Unmasked loads
        _mean += x_block

    #For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)
    _mean += x_block

    mean = tl.sum(_mean, axis=0) / n_cols

    #variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in tl.range(0, loop_num_l, num_stages=3):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  #Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    #For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    #Normalize and store
    loop_num_l = loop_num
    for b in tl.range(0, loop_num_l, num_stages=3):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        tl.store(y_ptr_start + col_offsets, y_block)

    #For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def layernorm_kernel_impl(x_ptr, y_ptr, w_ptr, b_ptr, x_row_stride, y_row_stride, n_rows, n_cols, eps,
                          BLOCK_SIZE: tl.constexpr):

    tl.assume(x_row_stride > 0)
    tl.assume(y_row_stride > 0)
    #program id
    row = tl.program_id(0)
    tl.assume(row > 0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)
    col_offs = tl.arange(0, BLOCK_SIZE)

    #calculate mean
    x_ptrs = x_ptr_start + col_offs
    mask = col_offs < n_cols
    x_block = tl.load(x_ptrs, cache_modifier=".cg", mask=mask, other=0.0).to(tl.float32)  #Unmasked loads
    mean = tl.sum(x_block, axis=0) / n_cols
    _x_block = tl.where(mask, x_block - mean, 0.0)
    var = tl.sum(_x_block * _x_block, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    w_block = tl.load(w_ptr + col_offs, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offs, mask=mask, other=0.0)
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offs, y_block, mask=mask)


def layernorm(x, y, w, b, eps=1e-5):
    n_rows, n_cols = x.shape

    grid = lambda meta: (n_rows, )
    if n_cols <= 8192:
        layernorm_kernel_impl[grid](x, y, w, b, x.stride(0), y.stride(0), n_rows, n_cols, eps,
                                    BLOCK_SIZE=triton.next_power_of_2(n_cols))
    else:
        layernorm_kernel_blocked_impl[grid](x, y, w, b, x.stride(0), y.stride(0), n_rows, n_cols, eps, BLOCK_SIZE=2048)

    return y


def torch_layernorm(x, w, b):
    M, N = x.shape
    w_shape = (N, )
    y_torch = torch.nn.functional.layer_norm(x, w_shape, w, b, eps=1e-5)
    return y_torch


def run_layernorm(M, N):
    print(f"Running Layernorm on shape ({M},{N})")
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y = torch.empty_like(x)
    w_shape = (N, )
    w = torch.rand(w_shape, device='cuda')
    b = torch.rand(w_shape, device='cuda')
    y_triton = layernorm(x, y, w, b)

    return y_triton


#pytest
@pytest.mark.parametrize('M, N', [(1823, 781), (2, 128), (1, 4), (128, 2), (1, 128), (8192, 8192), (4096, 8192),
                                  (359, 1), (1, 359), (1, 131072), (1, 89999)])
def test_layernorm(M, N, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y = torch.empty_like(x)
    w_shape = (N, )
    w = torch.rand(w_shape, device='cuda')
    b = torch.rand(w_shape, device='cuda')
    y_triton = layernorm(x, y, w, b, eps)
    y_torch = torch.nn.functional.layer_norm(x, w_shape, w, b, eps)

    assert torch.allclose(y_triton, y_torch, rtol=1e-05, atol=1e-06)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def run_benchmark(args):
    config = []
    sweep_m = args.M_step != 0
    sweep_n = args.N_step != 0
    x_vals_list = []
    if (sweep_m):
        val = args.M_start
        x_vals_list = []
        while val <= args.M_end:
            x_vals_list.append(val)
            val *= args.M_step
        mn_args = {'N': args.N_start}
        plot_name = str("layernorm-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                        "-" + str(args.M_end) + "-" + str(args.M_step))
        x_names = ['M']
    elif (sweep_n):
        x_vals_list = [i for i in range(args.N_start, args.N_end + 1, args.N_step)]
        mn_args = {'M': args.M_start}
        plot_name = str("layernorm-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                        "-" + str(args.N_end) + "-" + str(args.N_step))
        x_names = ['N']
    else:
        x_vals_list.append(args.N_start)
        x_names = ['N']
        mn_args = {'M': args.M_start}
        plot_name = str("layernorm-performance" + "_M" + str(args.M_start) + "_N" + str(args.N_start))
    dtype = arg_to_torch_dtype[args.dtype]

    print(plot_name)
    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=[
                "Triton",
                "Torch",
            ],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name=plot_name,
            args=mn_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        y = torch.empty_like(x)
        w_shape = (N, )
        w = torch.rand(w_shape, device='cuda', dtype=dtype)
        b = torch.rand(w_shape, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_layernorm(x, w, b))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: layernorm(x, y, w, b))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Layernorm",
        allow_abbrev=False,
    )

    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="0", type=int)
    parser.add_argument('-Me', "--M_end", default="0", type=int)

    parser.add_argument('-N', "--N_start", default="65536", type=int)
    parser.add_argument('-Ns', "--N_step", default="0", type=int)
    parser.add_argument('-Ne', "--N_end", default="0", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_benchmark:
        run_layernorm(args.M_start, args.N_start)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
