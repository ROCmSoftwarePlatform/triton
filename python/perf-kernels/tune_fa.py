"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import argparse
import pytest
import torch
import sys

import triton
import triton.language as tl

# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8:tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

def generate_one_fa_kernel_from_config(Batch, H, N_Ctx, D_Head, block_m, block_n, pre_load_v):
    attn_fwd_str = f"""
@triton.jit
def _attn_fwd_BLOCKM{block_m}_BLOCKN{block_n}_Preloadv{pre_load_v}(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(0, 1)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    # it's even better to multiply the qk_scale and convert to f16
    # than doing it inside the loop
    # So conversion is quite cheap
    q = (q * qk_scale).to(q.dtype)
    lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        if pre_load_v:
            v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        #qk = (qk * qk_scale)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    acc = acc / l_i[:, None]
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    """
    return attn_fwd_str

def generate_wrapper(tuning_parms):
    dri_str = """
name_to_torch_types = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': float8
}

def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal=False, dtype='fp16'):
    if dtype == 'fp8' and not TORCH_HAS_FP8E4:
        sys.exit("fp8 is not available")
    init_dtype = torch.float16 if dtype != 'bf16' else torch.bfloat16
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q, dtype=v.dtype)
    waves_per_eu = 2
    num_warps = 4
    num_stages = 1
    slice_k_tile = 32
    kpack = 1

    M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    """
    dri_str += '\n'
    for tp in tuning_parms:
        block_m = tp[0]
        block_n = tp[1]
        pre_load_v = tp[2]
        dri_str += f"""
    for i in range(100):
        grid = ( triton.cdiv(q.shape[2], {block_m}), q.shape[0] * q.shape[1], 1)
        _attn_fwd_BLOCKM{block_m}_BLOCKN{block_n}_Preloadv{pre_load_v}[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            N_CTX=q.shape[2],
            BLOCK_DMODEL=Lk,
            BLOCK_M = {block_m},
            BLOCK_N = {block_n},
            waves_per_eu = waves_per_eu,
            num_warps = num_warps,
            num_stages = num_stages,
            pre_load_v = {pre_load_v},
            slice_k_tile = slice_k_tile,
            kpack = kpack,
        )
        """

    return dri_str 

def generate_main(Batch, H, N_Ctx, D_Head):
    main_str = f"""
def main():
    bench_flash_attention({Batch}, {H}, {N_Ctx}, {D_Head})

if __name__ == '__main__':
    sys.exit(main())
    """

def generate_fa_kernel(Batch, H, N_Ctx, D_Head):
    # create the kernel file
    file_name = f"{Batch}_{H}_{N_Ctx}_{D_Head}.py"
    f_kernel = open("./generated_fa_kernel"+file_name, 'w')

    # import string
    import_str = """import pytest
import torch
import sys

import triton
import triton.language as tl
# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8:tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
"""
    
    f_kernel.write(import_str + '\n')
    
    # generate kernels with tuning parameters
    tuning_parms = []
    block_m_range = [16, 32]
    block_n_range = [16, 32]
    pre_load_v_range = [True, False]

    for block_m in block_m_range:
        for block_n in block_n_range:
            for pre_load_v in pre_load_v_range:
                tuning_parms.append((block_m, block_n, pre_load_v))
                kernel_str = generate_one_fa_kernel_from_config(Batch, H, N_Ctx, D_Head, block_m, block_n, pre_load_v)
                f_kernel.write(kernel_str + "\n") 
   
    # generate the driver
    dri_str = generate_wrapper(tuning_parms)
    f_kernel.write(dri_str + "\n") 

    main_str = f"""
def main():
    bench_flash_attention({Batch}, {H}, {N_Ctx}, {D_Head})

if __name__ == '__main__':
    sys.exit(main())
    """
    f_kernel.write(main_str+'\n')

def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a flash attention kernel",
        allow_abbrev=False,
    )

    parser.add_argument("-b", type=int, default=16, help='batch')
    parser.add_argument("-H", type=int, default=16)
    parser.add_argument("-n_ctx", type=int, default=1024)
    parser.add_argument("-d_head", type=int, default=128)
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--verbose", action='store_true', default=False, help="enables time_breakdown and additional logging messages")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use for kernel compilation and post processing")
    parser.add_argument("--jobs", type=int, default=1, help="number of generated files")
    parser.add_argument("--iters", type=int, default=1000, help="number of iterations")
    parser.add_argument("--datatype", type=str, default='fp16', help="element type")
    parser.add_argument("--no_warmup", action='store_true', default=False, help="Do not call the warmup kernel")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    keepTmp = args.keep
    jobs = args.jobs
    iters = args.iters
    skipWarmup = args.no_warmup

    # Get element type
    dtype = args.datatype

    mnks = []
    # TODO: make it more robust to get user input
    batch = args.b
    #h = args.h
    h = 16
    n_ctx = args.n_ctx
    d_head = args.d_head
    generate_fa_kernel(batch, h, n_ctx, d_head)

if __name__ == '__main__':
    sys.exit(main())
