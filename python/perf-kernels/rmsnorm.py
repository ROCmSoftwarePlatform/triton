import argparse
import torch
import sys
import pytest
from itertools import product

import triton
import triton.language as tl
from utils.benchmark_utils import get_available_models, get_model_configs


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [4, 8, 16])]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def rms_kernel(output_ptr, input_ptr, g_ptr, rsigma_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
               ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr,
               NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # as older version Triton doesn't support tl.assume and BUFF OPS, comment out for now
    # tl.assume(input_row_stride >= 0)
    # tl.assume(output_row_stride >= 0)
    # tl.assume(row_start >= 0)

    if USE_BLOCKED:

        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # sum_squares: tl.float32 = 0.
            # however, with type promoting rule in triton, sum_squares should be always fp32 with below init
            sum_squares = 0.
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            rsigma_output_ptr = rsigma_ptr + row_idx
            tl.store(rsigma_output_ptr, norm_factor)

            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


def triton_rmsnorm(x, y, g, rsigma, n_rows, n_cols, ZERO_CENTERED_GAMMA, blk_size, USE_BLOCKED, NUM_PRGMS,
                   epsilon=1e-6):
    grid = lambda meta: (NUM_PRGMS, )
    rms_kernel[grid](y, x, g, rsigma, x.stride(0), y.stride(0), n_rows, n_cols, epsilon, ZERO_CENTERED_GAMMA, blk_size,
                     USE_BLOCKED, NUM_PRGMS)

    return y, rsigma


def torch_rmsnorm(x, g, ZERO_CENTERED_GAMMA, out_dtype=torch.float16, epsilon=1e-6):
    M, N = x.shape
    # cast to float32 as the triton kernel
    x_f32 = x.float()
    g_f32 = g.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N)
    rsigma = 1.0 / rms
    if (ZERO_CENTERED_GAMMA):
        g_f32 += 1
    rms_norm_f32 = x_f32 * rsigma.unsqueeze(1) * g_f32
    rms_norm = rms_norm_f32.to(out_dtype)
    return rms_norm, rsigma


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


@pytest.mark.parametrize("in_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("out_dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize('ZERO_CENTERED_GAMMA', [True, False])
@pytest.mark.parametrize('M, N', [
    (1, 4),
    (2, 10),
    (8192, 4096),
    (4096, 8192),
    (1, 31744),
    (3, 65536),
    (873, 1245),
])
def test_rmsnorm(M, N, ZERO_CENTERED_GAMMA, in_dtype_str, out_dtype_str):
    in_dtype = arg_to_torch_dtype[in_dtype_str]
    out_dtype = arg_to_torch_dtype[out_dtype_str]
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda', dtype=in_dtype)
    y = torch.zeros_like(x, device='cuda', dtype=out_dtype)
    rsigma = torch.empty((M, ), device='cuda', dtype=torch.float32)

    n_rows, n_cols = x.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())
    g = torch.ones((1, N), device='cuda', dtype=in_dtype)

    y_triton, rsigma_triton = triton_rmsnorm(x, y, g, rsigma, n_rows, n_cols, ZERO_CENTERED_GAMMA, blk_size,
                                             USE_BLOCKED, NUM_PRGMS)

    y_torch, rsigma_torch = torch_rmsnorm(x, g, ZERO_CENTERED_GAMMA, out_dtype)

    if out_dtype in (torch.float16, torch.bfloat16):
        atol, rtol = 1e-3, 1e-2
    else:
        # float32 typically can be tighter
        atol, rtol = 1e-5, 1e-5

    assert y_triton.dtype == out_dtype, f"y_triton has dtype={y_triton.dtype}, expected {out_dtype}"
    assert y_torch.dtype == out_dtype, f"y_torch has dtype={y_torch.dtype}, expected {out_dtype}"

    assert torch.allclose(y_triton, y_torch, atol=atol, rtol=rtol), \
        f"Mismatch in 'y' (in={in_dtype_str}, out={out_dtype_str})"
    assert torch.allclose(rsigma_triton, rsigma_torch, atol=atol, rtol=rtol), \
        f"Mismatch in 'rsigma' (in={in_dtype_str}, out={out_dtype_str})"


#Benchmark
def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, model_families=["llama3"], model=args.model)

    x_vals_list = []
    batch_size = args.b if args.b else 1

    for model_name, config in configs.items():
        seq_len = args.sq if args.sq else 4096
        x_vals_list.append((model_name, batch_size * seq_len, config["hidden_size"]))

    return x_vals_list


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

    if args.model:
        assert not args.M_benchmark, \
            "Trying to provide both -model benchmark and M_benchmark is not supported!"
        x_names = ['model', 'M', 'N']
        mn_args = {}
        plot_name = str("rmsnorm-performance_" + args.dtype)
        x_vals_list = model_benchmark_configs(args)

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
    def benchmark(M, N, provider, model=None):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        y = torch.zeros_like(x, device='cuda')
        rsigma = torch.empty((M, ), device='cuda', dtype=torch.float32)
        n_rows, n_cols = x.shape
        MAX_FUSED_SIZE = 65536 // x.element_size()
        blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
        USE_BLOCKED = n_cols > blk_size
        NUM_PRGMS = min(n_rows, get_num_sms())
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        g = torch.ones((1, N), device='cuda')
        ZERO_CENTERED_GAMMA = False
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, g, ZERO_CENTERED_GAMMA))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: triton_rmsnorm(x, y, g, rsigma, n_rows, n_cols, ZERO_CENTERED_GAMMA,
                                                                blk_size, USE_BLOCKED, NUM_PRGMS))
            global verbose
            if verbose:
                print(f'SIZE: {N} Best tuning config: ({rms_kernel.best_config})')
                print(f'time: {ms}')
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark RMSNorm",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")

    available_models = get_available_models(model_families=["llama3"])  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +
        "]. Use 'all' to benchmark all models. Not providing runs the default benchmark script with custom configs.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument('-b', type=int, default=0, help="Batch size used together with model.")
    parser.add_argument('-sq', type=int, default=0, help="Sequence length used together with model.")
    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)  #This is multiplicative step
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="8192", type=int)
    parser.add_argument('-Ns', "--N_step", default="1024", type=int)
    parser.add_argument('-Ne', "--N_end", default="32768", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")

    return parser.parse_args()


def main():
    args = parse_args()
    global verbose
    if args.no_benchmark:
        x = torch.randn(args.M_start, args.N_start, device='cuda')
        y = torch.zeros_like(x, device='cuda')
        rsigma = torch.empty((args.M_start, ), device='cuda', dtype=torch.float32)
        n_rows, n_cols = x.shape
        MAX_FUSED_SIZE = 65536 // x.element_size()
        blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
        USE_BLOCKED = n_cols > blk_size
        NUM_PRGMS = min(n_rows, get_num_sms())
        g = torch.ones((1, args.N_start), device='cuda')
        ZERO_CENTERED_GAMMA = True
        triton_rmsnorm(x, y, g, rsigma, n_rows, n_cols, ZERO_CENTERED_GAMMA, blk_size, USE_BLOCKED, NUM_PRGMS)
    else:
        verbose = args.v
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
