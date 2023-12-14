"""
Simple test to measure achieved HBM bandwidth.
This kernel moves N bytes of data from one region in HBM to another, using Triton.
"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    input_ptr,  # *Pointer* to input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements: tl.constexpr,  # Total elements to move.
    BLOCK_SIZE: tl.constexpr,  # Elements to load / store per iteration
    vector_size: tl.constexpr, # Size of the entire vector being moved.

):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    
    lo = pid * n_elements
    hi = lo + n_elements
    for idx in range(lo, hi, BLOCK_SIZE):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        #mask = offsets < vector_size
        in_vals = tl.load(input_ptr + offsets)
        tl.store(output_ptr + offsets, in_vals)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def copy(x: torch.Tensor, wgs=512):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda
    vector_size = output.numel()
    BLOCK_SIZE = 4096
    grid = (wgs, 1, 1)
    # Each WG will move these many elements
    n_elements = triton.cdiv(vector_size, wgs)
    copy_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, vector_size=vector_size, num_warps=1)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)
size = 2**30
#x = torch.rand(size, device='cuda')
#output_torch = x
#output_triton = copy(x)
#print(output_torch)
#print(output_triton)
#print(
#    f'The maximum difference between torch and triton is '
#    f'{torch.max(torch.abs(output_torch - output_triton))}'
#)

# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.

configs = []
for wgs in [2 ** i for i in range(0, 12)]:
    configs.append(triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**30
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=f'wgs={wgs}',  # Name for the plot. Used also as a file name for saving the plot.
        args={'wgs':wgs},  # Values for function arguments not in `x_names` and `y_name`.
    )
    )

@triton.testing.perf_report(configs)
def benchmark(size, provider, wgs):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.clone(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: copy(x, wgs), quantiles=quantiles)
    # 8 because 4 bytes from load, 4 from store.
    gbps = lambda ms: 8 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
