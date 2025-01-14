# AMD Perf Kernels

This directory contains customized/tuned/experimental kernels for AMD Instinct series GPUs.
Please make sure your Triton compiler is v2.1 or later, and is from the OpenAI Triton repository
[here](https://github.com/openai/triton). To install Triton, please see
[these](https://github.com/openai/triton/tree/main?tab=readme-ov-file#install-from-source) instructions.

## `06-fused-attention-transV.py`

This script is a copy of `tutorials/06-fused-attention.py` with the following
two changes:

- Tensor V is transposed in the way that seqlen/N_CTX dimension becomes the
fastest changing (a.k.a. leading or least strided) dimension.
This script produces better performance than `tutorials/06-fused-attention.py`
since it has better LDS access efficiency for tensor V.
Note that in the future, we'll improve the LDS access efficiency for
non-transposed tensor V, i.e. head dimension is the fastest changing dimension.
- Only fwd kernel is benchmarked.

## `06-fused-attention-fwd-transV.py`

This script is used to produce the best performance for fwd kernel.
It is a copy of `06-fused-attention-transV.py` with the following
changes:

- All bwd kernels are removed.
- Storing `m` at the end of the fwd kernel is removed.
- Autotuner is removed. All parameters for D=64 ad D=128 are pre-tuned
on MI250X and hard coded.

Note that this script is also used to benchmark FA performance with 2 GCDs.
Check the [2GCD benchmark script](https://github.com/ROCmSoftwarePlatform/triton/blob/triton-mlir/scripts/amd/benchmark_flash_attention.py) for more details.

## `flash-attention.py`

This script contains the Flash Attention kernel with the following support

- Arbitrary Q and KV sequence lengths, and arbitrary head sizes
- Autoregressive or "causal" masking
- Flash Attention v2 with variable sequence lengths
- Multi and Grouped Query attention
- ALiBi bias
- Matrix bias
- Persistent kernels. Useful when the sequence lengths are up to a moderate length and especially when doing causal attention.
- Int8 quantization

These are currently supported for the forward kernel only.

INT8 Quantization Support

1. <em>q_descale</em>, <em>k_descale</em>, and <em>v_descale</em> provided:
   - The first QK GEMM runs in INT8, then the output is dequantized to the specified <em>dtype</em>.
   - The second PV GEMM runs in the specified <em>dtype</em>.

2. <em>q_descale</em>, <em>k_descale</em>, <em>p_descale</em>, and <em>v_descale</em> provided:
   - Both the first and second GEMM operations run in INT8.
   - The results are dequantized to the specified <em>dtype</em> after both GEMMs.

3. Only <em>k_descale</em> and <em>v_descale</em> provided:
   - K and V are dequantized before the first and second GEMM operations, respectively.
   - Both GEMMs run in the specified <em>dtype</em>.

Note: The softmax operation is always performed in <em>fp32</em>.


## `06-attention-decode.py`

This contains the Flash Decoding kernel.

## `hbm-bw-test.py`

This is a script that measures HBM bandwidth performance on your device.

## `03-matrix-multiplication-all-types.py`

This script contains the GEMM kernel that supports int8, int32, fp16,
fp32, bf16 and f8 (both e5m2 and e4m3) datatypes.

## `03-matrix-multiplication-stream-k.py`

This script contains the GEMM kernel that implements [stream-k](https://arxiv.org/abs/2301.03598)

## `multreduce_matmul_kernel.py`

Kernel that implements GEMM with explicit multiply-reduce instructions for small block sizes. Such
small block sizes aren't natively supported by `tl.dot` operator.

Despite being numerically correct, this kernel performed worse than a corresponding GEMM kernel that
used `tl.dot` with minimum block size equal to $16$.

## `softmax.py`

Kernel that implements Softmax over a row of tensor.

## `rmsnorm.py`

Kernel that implements RMS Norm over a row of tensor.

## `layernorm.py`
Kernel that implements Layer Normalization over a row on tensor


## `MLA.py`

Kernel that implements Multihead Latent Attention.
For the first version the idea is to just fuse the absorbed gemms into flash attention.
In pseudocode:

```python
"""
Q_NOPE: shape bhsd, query tokens with no positional embedding applied
Q_PE: shape bhsr, query tokens with positional embedding applied
KV: shape btc, latent representation for keys and values
K_PE: shape btr, latent representation for keys, positional embedding applied
WKV_B: shape h(d+d)c, projection matrix for KV

b: batch size, h: num heads, s: query sequence length, t: key/value sequence length,
d/r/c: query/pos.emb./latent repr. head dim.

I'm currently running with: b=2, h=16, s=128, t=128, d=64, r=32, c=256. And with:
triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
"""
@triton.jit
def attn_fwd(Q_NOPE, Q_PE, KV, K_PE, WKV_B, ...):
   q_nope_ptrs = ... # of size BLOCK_M x d
   kv_ptrs = ... # of size c x BLOCK_N
   k_pe_ptrs = ... # of size r x BLOCK_N
   q_pe_ptrs = ... # of size BLOCK_M x r
   wkv_b_ptrs1 = ... # of size d x c
   wkv_b_ptrs2 = ... # of size d x c

   q_nope = tl.load(q_nope_ptrs)
   q_pe = tl.load(q_pe_ptrs)
   wkv_b1 = tl.load(wkv_b_ptrs1) # WKV_B[:, :self.qk_nope_head_dim], absorbed into q_nope
   wkv_b = tl.load(wkv_b_ptrs2) # WKV_B[:, -self.v_head_dim:], needed in _attn_fwd_inner

   q_nope = tl.dot(q_nope, wkv_b1) # absorbtion
   acc, l_i, m_i = _attn_fwd_inner(
                        acc, l_i, m_i, q_nope, q_pe, wkv_b, kv_ptrs, k_pe_ptrs, ...)
   
   l_recip = 1 / l_i[:, None]
   acc = acc * l_recip
   tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q_nope, q_pe, wkv_b, kv_ptrs, k_pe_ptrs, ...):
   for start_n in range(block_min, block_max, BLOCK_N):
      kv = load_fn(kv_ptrs, ...)
      k_pe = load_fn(k_pe_ptrs, ...)

      qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      # -- compute qk ----
      qk += tl.dot(q_nope.to(kv.type.element_ty), kv)
      qk += tl.dot(q_pe, k_pe)
      qk *= QK_SCALE

      # softmax
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk = qk - m_ij[:, None]
      p = tl.math.exp2(qk)

      l_ij = tl.sum(p, 1)

      alpha = tl.math.exp2(m_i - m_ij)
      acc = acc * alpha[:, None]

      # -- update m_i and l_i
      l_i = l_i * alpha + l_ij
      m_i = m_ij

      v = tl.dot(wkv_b, kv).trans() # not the actual v, but helps to think
      acc += tl.dot(p.to(v.type.element_ty), v)

      kv_ptrs += BLOCK_N * stride_kv_n
      k_pe_ptrs += BLOCK_N * stride_k_pe_n
   
   return acc, l_i, m_i
```

Launch of the kernel:

```python
class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q_NOPE, Q_PE, KV, K_PE, O, WKV_B, metadata: MetaData):
      M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)
      grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), nheads_q, batch)
      attn_fwd[grid](
         q, q_pe, kv, k_pe, wkv_b, metadata.bias, metadata.sm_scale, M, o, ...)
      
      return o, _, _

```

And then how the kernel gets called:

```python
global attn_impl

def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
   """
   Forward pass for the Multi-Headed Attention Layer (MLA).

   Args:
      x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
      start_pos (int): Starting position in the sequence for caching.
      freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
      mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

   Returns:
      torch.Tensor: Output tensor with the same shape as the input.
   """
   bsz, seqlen, _ = x.size()
   end_pos = start_pos + seqlen
   if self.q_lora_rank == 0:
      q = self.wq(x)
   else:
      q = self.wq_b(self.q_norm(self.wq_a(x)))
   q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
   q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
   q_pe = apply_rotary_emb(q_pe, freqs_cis)
   kv = self.wkv_a(x)
   kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
   k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

   if attn_impl == "flash":
      # "absorb" and Flash Attention fused
      wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
      wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
      kv = self.kv_norm(kv)
      x = flash_attention(q_nope, q_pe, kv, k_pe.squeeze(2), wkv_b, self.qk_nope_head_dim, self.v_head_dim, self.softmax_scale)
   else:
      if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
      else: # absorb
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(
               self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                     torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

      if mask is not None:
            scores += mask.unsqueeze(1)
      scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
      if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
      else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
   x = self.wo(x.flatten(2))

   return x

def flash_attention(q_nope, q_pe, kv, k_pe, wkv_b, qk_nope_head_dim, v_head_dim, sm_scale):
    # q comes in as bshd, we assume bhsd inside kernel. So permute for now.
    q_nope = q_nope.permute((0, 2, 1, 3))
    q_pe = q_pe.permute((0, 2, 1, 3))
    o = torch.zeros((*q_nope.shape[:-1], v_head_dim), dtype=q_nope.dtype)
    Z, H, N, D = q_nope.shape
    _, _, _, input_metadata = input_helper(Z, H, H, N, N, D, q_nope.dtype, "bhsd", requires_grad=False)
    input_metadata.qk_nope_head_dim = qk_nope_head_dim
    input_metadata.v_head_dim = v_head_dim
    input_metadata.sm_scale = sm_scale
    o, _, _ = attention(q_nope, q_pe, kv, k_pe, o, wkv_b, input_metadata)
    return o.permute((0, 2, 1, 3))  # permute back to bshd

```