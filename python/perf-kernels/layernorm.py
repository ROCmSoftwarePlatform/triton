import argparse
import sys
import pytest

import torch
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


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def layernorm_kernel(x_ptr, 
                     y_ptr, 
                     w_ptr, 
                     b_ptr, 
                     mean_ptr, 
                     rstd_ptr, 
                     x_row_stride, 
                     y_row_stride, 
                     n_rows, 
                     n_cols, 
                     eps,
                     BLOCK_SIZE: tl.constexpr):

    #program id
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    #calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
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
    for b in range(0, loop_num_l):
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

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    #Normalize and store
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
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


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             NUM_ROWS: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = NUM_ROWS // tile_num
    if pid < NUM_ROWS % tile_num:
        rows_per_tile += 1

    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    row = pid
    for _ in range(0, rows_per_tile):
        x_ptrs = X + row * stride
        dy_ptrs = DY + row * stride
        dx_ptrs = DX + row * stride
        dw_ptrs = DW + pid * N + cols
        db_ptrs = DB + pid * N + cols
        # Load data to SRAM
        x = tl.load(x_ptrs + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(dy_ptrs + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(dx_ptrs + cols, dx, mask=mask)
        # Accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        partial_dw += tl.load(dw_ptrs, mask=mask)
        partial_db += tl.load(db_ptrs, mask=mask)
        tl.store(dw_ptrs, partial_dw, mask=mask)
        tl.store(db_ptrs, partial_db, mask=mask)
        row += tile_num


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


class LayerNorm(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        layernorm_kernel[(M, )](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0), y.stride(0), M, N, eps, BLOCK_SIZE
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y


    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        N = w.shape[0]
        x_arg = x.reshape(-1, x.shape[-1])
        M = x_arg.shape[0]
        tile_num = min(256, M // 4)
        # allocate output
        _dw = torch.zeros((tile_num, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((tile_num, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        M, N = x_arg.shape
        grid_bwd = (tile_num,)

        _layer_norm_bwd_dx_fused[grid_bwd](  #
            dx, dy, _dw, _db, x, w, m, v,  #
            x_arg.stride(0), N,  #
            NUM_ROWS=M,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            num_warps=ctx.num_warps)
        grid_reduce = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid_reduce](
            _dw, _db, dw, db, min(tile_num, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layernorm = LayerNorm
# def layernorm(x, w, b, eps=1e-5):
#     n_rows, n_cols = x.shape

#     MAX_FUSED_SIZE = 65536 // x.element_size()
#     BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
#     y = torch.empty_like(x)

#     num_programs = n_rows

#     grid = lambda meta: (num_programs, )
#     layernorm_kernel[grid](x, y, w, b, x.stride(0), y.stride(0), n_rows, n_cols, eps, BLOCK_SIZE)

    # return y


def torch_layernorm(x, w, b):
    M, N = x.shape
    w_shape = (N, )
    y_torch = torch.nn.functional.layer_norm(x, w_shape, w, b, eps=1e-5)
    return y_torch


def run_layernorm(M, N):
    print(f"Running Layernorm on shape ({M},{N})")
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    w_shape = (N, )
    w = torch.rand(w_shape, device='cuda')
    b = torch.rand(w_shape, device='cuda')
    y_triton = layernorm(x, w, b)

    return y_triton


#pytest
@pytest.mark.parametrize('M, N', [(1823, 781), (2, 128), (1, 4), (128, 2), (1, 128), (8192, 8192), (4096, 8192),
                                  (359, 1), (1, 359), (1, 131072), (1, 89999)])
def test_layernorm(M, N, eps=1e-5):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    w_shape = (N, )
    w = torch.rand(w_shape, device='cuda')
    b = torch.rand(w_shape, device='cuda')
    y_triton = layernorm(x, w, b, eps)
    y_torch = torch.nn.functional.layer_norm(x, w_shape, w, b, eps)

    assert torch.allclose(y_triton, y_torch, rtol=1e-05, atol=1e-06)


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
        plot_name = str("layernorm-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                        "-" + str(args.M_end) + "-" + str(args.M_step))
        x_names = ['M']
    else:
        x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
        mn_args = {'M': args.M_start}
        plot_name = str("layernorm-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                        "-" + str(args.N_end) + "-" + str(args.N_step))
        x_names = ['N']
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
        w_shape = (N, )
        w = torch.rand(w_shape, device='cuda', dtype=dtype)
        b = torch.rand(w_shape, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch_layernorm(x, w, b))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: layernorm(x, w, b))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Layernorm",
        allow_abbrev=False,
    )

    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="1024", type=int)
    parser.add_argument('-Ns', "--N_step", default="2048", type=int)
    parser.add_argument('-Ne', "--N_end", default="65536", type=int)

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
