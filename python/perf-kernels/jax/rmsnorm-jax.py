import argparse
import jax
import jax.numpy as jnp
from jax import random
from jax import grad, jit
import numpy as np
import flax
from flax import nnx
import sys
import pytest
from itertools import product

import jax_triton as jt
import triton
import triton.language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_num_sms():
    return 304

#@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def rms_kernel(input_ptr, g_ptr, output_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
               ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr, NUM_PRGMS: tl.constexpr):
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

            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

def triton_rmsnorm(x: jnp.ndarray, y: jnp.ndarray, g: jnp.ndarray, rows: int, cols: int,
                   gamma: bool, blk_size: int, use_blk: bool, num_p: int, 
                   epsilon=1e-6)-> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    x_strides = jt.strides_from_shape(x.shape)
    y_strides = jt.strides_from_shape(y.shape)

    return jt.triton_call(
      x,
      g,
      input_row_stride=x_strides[0],
      output_row_stride=y_strides[0],
      n_rows=rows,
      n_cols=cols,
      epsilon=1e-6,
      ZERO_CENTERED_GAMMA=gamma,
      BLOCK_SIZE=blk_size,
      USE_BLOCKED=use_blk,
      NUM_PRGMS=num_p,
      kernel=rms_kernel,
      out_shape=out_shape,
      grid=lambda meta: (num_p, ),
    )

def jax_rmsnorm(x, g, doFlax: bool) -> jnp.ndarray:
    M, N = x.shape
    eps=1e-8
    if doFlax:
        if hasattr(flax.nnx, 'RMSNorm'):
            print("Flax rmsnorm:")
            layer = nnx.RMSNorm(num_features=N, rngs=nnx.Rngs(0))
            return layer(x)
    else:
        print("JAX rmsnorm:")
        return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)

arg_to_jax_dtype = {'fp16': jnp.float16, 'bf16': jnp.bfloat16, 'fp32': jnp.float32}

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

    parser.add_argument('-d', "--dtype", default="fp32")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")

    return parser.parse_args()


def main():
    args = parse_args()
    global verbose
    dtype = arg_to_jax_dtype[args.dtype]
    k1 = random.PRNGKey(10)
    x = random.normal(k1, (args.M_start, args.N_start), arg_to_jax_dtype[args.dtype])
    print("Input:")
    print(x)
    y = jnp.zeros_like(x)
    n_rows, n_cols = x.shape
    MAX_FUSED_SIZE = 65536 // x.itemsize
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())
    g = jnp.ones((1, args.N_start))
    ZERO_CENTERED_GAMMA = False
    print("Triton RMSNorm: ")
    print(jax.jit(triton_rmsnorm, static_argnums=[3,4,5,6,7,8])
                 (x, y, g, n_rows, n_cols, ZERO_CENTERED_GAMMA, 
                  blk_size, USE_BLOCKED, NUM_PRGMS
                  ).block_until_ready())
    print(jax_rmsnorm(x, g, True))
    print(jax_rmsnorm(x, g, False))
    print("numpy rmsnorm")
    print(x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6))


if __name__ == "__main__":
    sys.exit(main())
