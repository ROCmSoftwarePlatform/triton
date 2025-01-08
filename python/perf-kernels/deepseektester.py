import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from MLA import attention, input_helper



# Define ModelArgs for MLA
@dataclass
class ModelArgs:
    attn_impl: str = "naive"
    dim: int = 2048
    n_heads: int = 16
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    max_batch_size: int = 8
    max_seq_len: int = 4096
    dtype: str = "bf16"

# Define MLA with Flash Attention integration
class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Query projection
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # Key-Value projection
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

        self.attn_impl = args.attn_impl

        # Caches for attention
        if self.attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.qk_head_dim))
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.v_head_dim))
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank))
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        attn_impl = self.attn_impl
        
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Compute queries
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # Compute keys and values
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if attn_impl == "flash":
            # Flash Attention integration
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
            kv = self.kv_norm(kv)
            kv = kv.view(bsz, seqlen, self.n_heads, -1)
            x = flash_attention(q_nope, q_pe, kv, k_pe.squeeze(2), wkv_b)
        else:
            if attn_impl == "naive":
                q = torch.cat([q_nope, q_pe], dim=-1)
                kv = self.wkv_b(self.kv_norm(kv))
                kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
                self.k_cache[:bsz, start_pos:end_pos] = k
                self.v_cache[:bsz, start_pos:end_pos] = v
                scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            else:
                wkv_b = self.wkv_b.weight
                wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
                # print(q_nope.shape)
                # print(wkv_b[:, :self.qk_nope_head_dim].shape)
                print("q, q_pe, kv, k_pe, wkv_b", q_nope.shape, q_pe.shape, kv.shape, k_pe.squeeze(2).shape, wkv_b.shape)

                q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                # print("q nope after")
                # print(q_nope.shape)
                self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
                # print("kv shape")
                # print(self.kv_cache[:bsz, :end_pos].shape)


                scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

            if mask is not None:
                scores += mask.unsqueeze(1)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

            if attn_impl == "naive":
                x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            else:
                # print("scores shape")
                # print(scores.shape)
                # print("kv shape")
                # print(self.kv_cache[:bsz, :end_pos].shape)
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        if attn_impl == "flash":
            torch.cuda.synchronize()
            print("here")
            print("x shape")
            print(x.shape)
        x = self.wo(x.flatten(2))
        return x


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


# Dummy Flash Attention implementation (replace with your actual implementation)
def flash_attention(q_nope, q_pe, kv, k_pe, wkv_b):
    # Replace this with your actual Flash Attention implementation
    q_nope = q_nope.permute((0,2,1,3))
    q_pe = q_pe.permute((0,2,1,3))
    kv = kv.permute((0,2,1,3))
    # k_pe = k_pe.permute((0,2,1,3))
    o = torch.zeros_like(q_nope)
    Z, H, N, D = q_nope.shape
    print("Z, H, N, D: ", Z, H, N, D)
    _, _, _, input_metadata = input_helper(Z, H, H, N, N, D, q_nope.dtype, "bhsd", requires_grad=False)
    attention(q_nope, q_pe, kv, k_pe, o, wkv_b, input_metadata)
    return o # torch.randn_like(q_nope)  # Dummy output


def test_mla_forward():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = ModelArgs()
    bsz, seqlen = 2, 128
    x = torch.randn(bsz, seqlen, args.dim, device="cuda", dtype=torch.float)
    freqs_cis = torch.randn(seqlen, args.qk_rope_head_dim // 2, 1, device="cuda", dtype=torch.float)

    # Run naive implementation
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    args.attn_impl = "absorb"
    mla_naive = MLA(args).to("cuda").float()
    output_naive = mla_naive(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    # Run absorb implementation
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    args.attn_impl = "flash"
    mla_absorb = MLA(args).to("cuda").float()
    output_absorb = mla_absorb(x, start_pos=0, freqs_cis=freqs_cis, mask=None)

    # Compute the difference between the outputs
    diff = torch.abs(output_naive - output_absorb).mean().item()
    print(f"Mean absolute difference between naive and absorb: {diff}")

    # Assert that the difference is within tolerance
    tolerance = 0.02
    assert diff <= tolerance, f"Difference {diff} exceeds tolerance {tolerance}"

    print("Test passed: Differences are within tolerance.")


if __name__ == "__main__":
    test_mla_forward()