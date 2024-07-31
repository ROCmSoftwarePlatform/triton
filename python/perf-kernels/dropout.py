import triton
import triton.language as tl

@triton.jit
def dropout_offsets(philox_seed, philox_offset, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]

@triton.jit
def dropout_rng(philox_seed, philox_offset, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, m, n, stride)
    # if tl.program_id(2) == 1:
    #     rng_keep = rng_output > -1.0
    # else:
    #     rng_keep = rng_output > dropout_p
    rng_keep = rng_output > dropout_p
    return rng_keep
