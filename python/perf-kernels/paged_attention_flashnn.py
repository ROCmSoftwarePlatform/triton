import sys
import argparse
import pytest
import random
import math

import triton
import triton.language as tl
import torch

#This code is derived from sglang and FLASHNN projects
#https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

#From VLLM 
#_SEQ_PARTITION_SIZE = 512 #CUDA
#_SEQ_PARTITION_SIZE = 1024 #HIP
_SEQ_PARTITION_SIZE = 16 #HIP


def paged_attention(
    output: torch.Tensor,  #[num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor, #[num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor, #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor, #[num_blks, num_kv_heads, kv_blk_sz, head_sz]
    seq_lens: torch.Tensor, #[num_seqs]
    block_tables: torch.Tensor, #[num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    num_seq_partitions: int = 0, #TODO use this below
    alibi_slopes: torch.Tensor = None,
) -> None:

    #get num_seqs, num_kv_heads, kv_blk_sz, head_sz and query_grp_sz
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads

    max_num_partitions = ((max_seq_len + _SEQ_PARTITION_SIZE - 1) //
                              _SEQ_PARTITION_SIZE)
    
    #print(f"max_num_partitions={max_num_partitions}")
    #max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    #print(f"max_num_partitions_pow2={max_num_partitions_pow2}")
    #print(f"max_num_partitions={max_num_partitions}")

    #print(f"max_num_partitions={max_num_partitions}")
    use_v1 = (max_seq_len <= 8192 and (max_num_partitions == 1 or num_seqs * num_q_heads > 512))
    kv_cache_dtype="auto"
    k_scale = 1.0
    v_scale = 1.0
    if use_v1:
        paged_attn_decode_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            attn_scale,
            block_tables,
            seq_lens,
            kv_blk_sz,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale
        )
    else:
        paged_attn_decode_v2(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            attn_scale,
            block_tables,
            seq_lens,
            kv_blk_sz,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            max_num_partitions,
        )

def paged_attn_decode_v1(
    output,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens,
    block_size,
    max_seq_len,
    alibi_slopes,
    kv_cache_dtype,
    k_scale,
    v_scale,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0 
):
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    #print(f"kv_blk_sz={kv_blk_sz}")
    #print(f"kv_blk_sz_pow2={kv_blk_sz_pow2}")
    #print(f"head_sz={head_sz}")
    #print(f"head_sz_pow2={head_sz_pow2}")
    #MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel[grid](
                                output,
                                query,
                                key_cache,
                                value_cache,
                                scale,
                                block_tables,
                                seq_lens,
                                alibi_slopes,
                                query.stride(0),
                                query.stride(1),
                                output.stride(0),
                                output.stride(1),
                                output.stride(2),
                                key_cache.stride(0),
                                key_cache.stride(1),
                                key_cache.stride(2),
                                block_tables.stride(0),
                                KV_BLK_SZ=kv_blk_sz,
                                KV_BLK_SZ_POW2=kv_blk_sz_pow2,
                                HEAD_SZ=head_sz,
                                HEAD_SZ_POW2=head_sz_pow2,
                                QUERY_GRP_SZ=query_grp_sz,
                                MAX_SEQ_LEN_POW2=max_seq_len)
    #GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:    
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        #print(f"HEAD_SZ={head_sz}")
        #print(f"HEAD_SZ_POW2={head_sz_pow2}")
        #print(f"QUERY_GRP_SZ={query_grp_sz}")
        #print(f"QUERY_GRP_SZ_POW2={query_grp_sz_pow2}")
        #print(f"KV_BLK_SZ={kv_blk_sz}")
        #print(f"KV_BLK_SZ_POW2={kv_blk_sz_pow2}")
        _paged_attn_decode_v1_w_dot_kernel[grid](
                output,
                query,
                key_cache,
                value_cache,
                seq_lens,
                block_tables,
                scale,
                block_tables.stride(0),
                block_tables.stride(1),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                HEAD_SZ=head_sz,
                HEAD_SZ_POW2=head_sz_pow2,
                QUERY_GRP_SZ=query_grp_sz,
                QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
                KV_BLK_SZ=kv_blk_sz,
                KV_BLK_SZ_POW2=kv_blk_sz
        )

@triton.jit
def _paged_attn_decode_v1_wo_dot_kernel(
    out, #[num_seqs, num_kv_heads * query_gr_sz, head_sz]
    q_ptr, #[num_seqs, num_kv_heads * query_gr_sz, head_sz]
    k_cache_ptr,#[num_blks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr, #[num_blks, num_kv_heads, kv_blk_sz, head_sz]
    scale,
    blk_tables_ptr, #[num_seqs, max_num_blocks_per_seq]
    seq_lens, #[num_seqs]
    alibi_slopes,
    stride_q_s,
    stride_q_h,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    stride_k_b, 
    stride_k_nh,
    stride_k_kb,
    stride_bt_s,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    #get head_idx, kv_head_idx
    head_idx = tl.program_id(axis=0)
    #kv_head_idx = head_idx // QUERY_GRP_SZ
    seq_idx = tl.program_id(axis=1)

    dtype = q_ptr.dtype.element_ty

    #seq_len
    seq_len = tl.load(seq_lens + seq_idx)

    if seq_idx >= seq_len:
        return

    #set start and ending blocks
    num_ctx_blocks = tl.cdiv(seq_len, KV_BLK_SZ)

    #alibi slopes calculation
    if alibi_slopes is None:
        alibi_slope = 0.0
    else:
        alibi_slope = tl.load(alibi_slopes + head_idx)
    
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)

    #load q [1, HEAD_SZ_POW2]
    q = tl.load(q_ptr + seq_idx*stride_q_s + head_idx*stride_q_h + head_sz_offs)
    q = (q*scale).to(dtype) 
    #tl.device_print("q",q)

    qkv = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)
    qk_max = float("-inf")
    exp_sum = 0.0
    kv_offs = (
        head_idx * stride_k_nh + blk_offs[:, None]*stride_k_kb + head_sz_offs[None, :]
    )
    blk_start_ptr = blk_tables_ptr + seq_idx * stride_bt_s
    
    #loop unroll. calculate qkv
    #loop over blocks
    for blk_idx in range(num_ctx_blocks):
        #get k block indices
        kv_offs_0 = tl.load(blk_start_ptr + blk_idx + 0) 
        kv_blk_offs = kv_offs_0 * stride_k_b + kv_offs 
        kv_mask = (blk_offs[:, None] < (seq_len - blk_idx * KV_BLK_SZ)) & (blk_offs[:, None] < KV_BLK_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
        #load k [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k_0 = tl.load(k_cache_ptr + kv_blk_offs,mask=kv_mask).to(dtype)
        #load v [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v_0 = tl.load(v_cache_ptr + kv_blk_offs,mask=kv_mask).to(dtype)

        #tl.device_print("k_0", k_0)
        #calculate qk #[KV_BLK_SZ_POW2]
        _qk_0 = tl.sum((q[None, :]*k_0).to(tl.float32), axis=1)
        _qk_0 = tl.where(blk_idx*KV_BLK_SZ + blk_offs < seq_len, _qk_0, float("-inf")) 
        #tl.device_print("_qk_0", _qk_0)

        #add alibi slopes
        #if alibi_slope is not None:
        #    _qk_0 += alibi_slope *((blk_idx + 0) * KV_BLK_SZ + kv_blk_offs - seq_len + 1)
        #Find max
        _qk_max = tl.maximum(tl.max(_qk_0, axis=0), qk_max)
        #tl.device_print("_qk_max", _qk_max)

        #Calculate exp
        exp_tmp = tl.exp(_qk_0 - _qk_max)
        _exp_sum = exp_sum * tl.exp(qk_max - _qk_max) + tl.sum(exp_tmp, axis=0)
        qkv_sum_tmp = (tl.exp(_qk_0[:, None] - _qk_max)).to(v_cache_ptr.dtype.element_ty) * v_0
        qkv = (qkv * (exp_sum * tl.exp(qk_max - _qk_max)) + qkv_sum_tmp) / _exp_sum
        qk_max = _qk_max
        exp_sum = _exp_sum
        #tl.device_print("qk_max", qk_max)
        #tl.device_print("exp_sum", exp_sum)
        #tl.device_print("qkv", qkv)

    #store out
    offs_out = (
        seq_idx * stride_o_s
        + head_idx * stride_o_nh
        + head_sz_offs
    )
    #tl.device_print("head_sz_offs", head_sz_offs)
    out_mask = head_sz_offs < HEAD_SZ
    tl.store(out + offs_out, tl.sum(qkv, axis=0), mask= out_mask)


@triton.jit
def _paged_attn_decode_v1_w_dot_kernel(
    out_ptr, #[num_seqs, num_kv_heads * query_gr_sz, head_sz]
    q_ptr, #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr, #[num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr, #[num_blocks, num_kv_heads, kv_blk_sz, head_sz]
    ctx_lens_ptr, #[num_seqs]
    blk_tables_ptrs, #[num_seqs, max_num_blks_per_seq]
    attn_scale,
    stride_blk_tables_s,
    stride_blk_tables_nb,
    stride_q_s,
    stride_q_nh,
    stride_q_hs,
    stride_k_s,
    stride_k_nh,
    stride_k_b,
    stride_k_hs,
    stride_o_s,
    stride_o_nh,
    stride_o_hs,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    log2e: tl.constexpr = 1.4426950408889634

    dtype = q_ptr.dtype.element_ty

    ctx_len = tl.load(ctx_lens_ptr + seq_idx)
    num_kv_blks = tl.cdiv(ctx_len, KV_BLK_SZ).to(tl.int32)
    
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    kv_offs = (kv_head_idx * stride_k_nh 
        + blk_offs[:, None] * stride_k_b 
        + head_offs[None, :] * stride_k_hs)
    
    q_offs = (seq_idx * stride_q_s 
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh 
        + head_offs[None, :] * stride_q_hs)
    
    grp_mask = q_grp_offs[:, None] < QUERY_GRP_SZ
    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ]
    q = tl.load(q_ptr + q_offs, mask=grp_mask, other=0.0)
    q = (q * attn_scale).to(dtype)
    #tl.device_print("q",q)

    max_logit_i = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum_i = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)

    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_blk_tables_s
    # TODO: loop peeling
    for b in range(num_kv_blks):
        blk_num = tl.load(blk_tables_start_ptr + b*stride_blk_tables_nb)

        kv_blk_offs = blk_num * stride_k_s + kv_offs
        mask_offset = b * KV_BLK_SZ + blk_offs
        kv_mask = (mask_offset[:, None] < ctx_len) & (blk_offs[:, None] < KV_BLK_SZ_POW2) & (head_offs[None, :] < HEAD_SZ_POW2)

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0).to(dtype)
        #tl.device_print("k",k)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        #qk = tl.where(mask_offset < ctx_len, qk, float("-inf"))

        qk = tl.where((q_grp_offs[:, None] < QUERY_GRP_SZ) & (mask_offset[None, :] < ctx_len), qk, float("-inf"))
        
        max_logit_i_new = tl.maximum(max_logit_i, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit_i - max_logit_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ, HEAD_SZ]
        v = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0).to(dtype)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)
        
        exp_sum_i = exp_sum_i * alpha + tl.sum(p, axis=1)
        max_logit_i = max_logit_i_new
    acc = acc / exp_sum_i[:, None]


    out_offs = (seq_idx * stride_o_s 
                + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_nh 
                + head_offs[None, :])

    out_mask = q_grp_offs[:, None] < QUERY_GRP_SZ
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def paged_attn_decode_v2(
    output,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    seq_lens,
    block_size,
    max_seq_len,
    alibi_slopes,
    kv_cache_dtype,
    k_scale,
    v_scale,
    max_num_partitions,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0 
):
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    #max_num_partitions = max_logits.shape[2]
    #max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2**math.ceil(math.log2(max_num_partitions))
    #print(f"max_num_partitions={max_num_partitions}")
    #print(f"max_num_partitions_pow2={max_num_partitions_pow2}")
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    #MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(size=shape_info, dtype=torch.float32, device=output.device)
        max_logits = torch.empty(size=shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty((*shape_info, head_sz), dtype=output.dtype, device=output.device)
        _paged_attn_decode_v2_wo_dot_kernel[grid](
                                exp_sums,
                                max_logits,
                                tmp_output,
                                query,
                                key_cache,
                                value_cache,
                                scale,
                                block_tables,
                                seq_lens,
                                alibi_slopes,
                                query.stride(0),
                                query.stride(1),
                                tmp_output.stride(0),
                                tmp_output.stride(1),
                                tmp_output.stride(2),
                                key_cache.stride(0),
                                key_cache.stride(1),
                                key_cache.stride(2),
                                exp_sums.stride(0),
                                exp_sums.stride(1),
                                block_tables.stride(0),
                                block_tables.stride(1),
                                KV_BLK_SZ=kv_blk_sz,
                                KV_BLK_SZ_POW2=kv_blk_sz_pow2,
                                HEAD_SZ=head_sz,
                                HEAD_SZ_POW2=head_sz_pow2,
                                QUERY_GRP_SZ=query_grp_sz,
                                SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
                                MAX_NUM_BLKS_PER_SEQ=block_tables.shape[1],
                                MAX_SEQ_LEN_POW2=max_seq_len)
        #print(f"max_logits={max_logits}") 
        #print(f"exp_sums={exp_sums}") 
        #print(f"tmp_output={tmp_output}") 
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel[grid](
                output,
                exp_sums,
                max_logits,
                tmp_output,
                seq_lens,
                exp_sums.stride(0),
                exp_sums.stride(1),
                output.stride(0),
                output.stride(1),
                tmp_output.stride(0),
                tmp_output.stride(1),
                tmp_output.stride(2),
                HEAD_SZ=head_sz,
                HEAD_SZ_POW2=head_sz_pow2,
                SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
                MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
                MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2)
        )
    #GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(*shape_info, head_sz, dtype=output.dtype, device=output.device)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:    
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        #print(f"query_grp_sz={query_grp_sz}")
        #print(f"query_grp_sz_pow2={query_grp_sz_pow2}")
        _paged_attn_decode_v2_w_dot_kernel[grid](
            max_logits,
            exp_sums,
            tmp_output,
            query,
            key_cache,
            value_cache,
            seq_lens,
            block_tables,
            scale,
            block_tables.stride(0), 
            block_tables.stride(1), 
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            max_logits.stride(0),
            max_logits.stride(1),
            max_logits.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            tmp_output.stride(4),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE
        )
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel[grid](
            output,
            max_logits,
            exp_sums,
            tmp_output,
            seq_lens,
            max_logits.stride(0),
            max_logits.stride(1),
            max_logits.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            HEAD_SZ=head_sz,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions))
        )


@triton.jit
def _paged_attn_decode_v2_wo_dot_kernel(
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    scale,
    blk_tables_ptr,
    ctx_lens_ptr,
    alibi_slopes,
    stride_q_s,
    stride_q_h,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_k_s, 
    stride_k_nh,
    stride_k_b,
    stride_exp_s,
    stride_exp_h,
    stride_blk_tables_s,
    stride_blk_tables_nb,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_BLKS_PER_SEQ: tl.constexpr,
    MAX_SEQ_LEN_POW2: tl.constexpr,
):
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(0)
    ctx_part_idx = tl.program_id(2)
    kv_head_idx = head_idx // QUERY_GRP_SZ

    log2e: tl.constexpr = 1.4426950408889634
    dtype = q_ptr.dtype.element_ty

    ctx_len = tl.load(ctx_lens_ptr + seq_idx)

    if ctx_part_idx * SEQ_PARTITION_SZ >= ctx_len:
        return

    ctx_start_idx = ctx_part_idx * SEQ_PARTITION_SZ
    ctx_end_idx = tl.minimum(ctx_start_idx + SEQ_PARTITION_SZ, ctx_len)

    #This conversion is needed. Otherwise, this causes a compilation error with mismatch types in the loop below
    #num_kv_blks = tl.cdiv(ctx_end_idx - ctx_start_idx, KV_BLK_SZ).to(tl.int32)
    num_kv_blks = tl.cdiv(ctx_end_idx - ctx_start_idx, KV_BLK_SZ)
    
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)

    kv_offs = (kv_head_idx * stride_k_nh 
        + blk_offs[:, None] * stride_k_b 
        + head_offs[None, :])
    
    q_offs = (seq_idx * stride_q_s 
        + head_idx * stride_q_h 
        + head_offs)
    
    # load q[HEAD_SZ]
    q = tl.load(q_ptr + q_offs)
    q = (q * scale).to(dtype)
    #tl.device_print("q",q)

    max_logit_i = float("-inf")
    exp_sum_i = 0.0
    acc = tl.zeros([KV_BLK_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)

    kv_blk_start = ctx_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptr + seq_idx * stride_blk_tables_s
    for b in range(num_kv_blks):
        blk_idx = kv_blk_start + b
        blk_num = tl.load(blk_tables_start_ptr + blk_idx*stride_blk_tables_nb)

        kv_blk_offs = blk_num * stride_k_s + kv_offs
        mask_offset = blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = (mask_offset[:, None] < ctx_len) & (blk_offs[:, None] < KV_BLK_SZ) & (head_offs[None, :] < HEAD_SZ)

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)

        #tl.device_print("k",k)
        # qk: [KV_BLK_SZ_POW2]
        qk = tl.sum((q[None, :]* k).to(tl.float32), axis=1)
        
        qk = tl.where(mask_offset < ctx_len, qk, float("-inf")) 
        #tl.device_print("qk",qk)

        max_logit_i_new = tl.maximum(max_logit_i, tl.max(qk, axis=0))   #_qk_max

        # p: [KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_i_new) * log2e)
        alpha = tl.math.exp2((max_logit_i - max_logit_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0)

        # acc: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        acc += p[:, None] * v
        
        exp_sum_i = exp_sum_i * alpha + tl.sum(p, axis=0)
        max_logit_i = max_logit_i_new
    acc = acc / exp_sum_i

    max_logits_offs = (
            seq_idx * stride_exp_s
        +   head_idx * stride_exp_h
        +   ctx_part_idx
    )
    tl.store(max_logits_ptr + max_logits_offs, max_logit_i)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum_i)

    logits_offs = (seq_idx * stride_logits_s
        + head_idx * stride_logits_h
        + ctx_part_idx * stride_logits_p
        + head_offs
    )
    logits_mask = head_offs < HEAD_SZ
    tl.store(logits_ptr + logits_offs, tl.sum(acc,axis=0), mask=logits_mask) 

@triton.jit
def _paged_attn_decode_v2_wo_dot_reduce_kernel(
    out,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    context_lens,
    stride_exp_sums_n,
    stride_exp_sums_h,
    stride_out_n,
    stride_out_h,
    stride_logits_n,
    stride_logits_h,
    stride_logits_b,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    #get seq_idx, head_idx, seq_len
    head_idx = tl.program_id(axis=0)
    seq_idx = tl.program_id(axis=1)

    ctx_len = tl.load(context_lens + seq_idx)

    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    seq_part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    #tl.device_print("seq_part_offs", seq_part_offs)
    #tl.device_print("MAX_NUM_SEQ_PARTITIONS",MAX_NUM_SEQ_PARTITIONS)
    #tl.device_print("MAX_NUM_SEQ_PARTITIONS_POW2",MAX_NUM_SEQ_PARTITIONS_POW2)

    #offs_logit = seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h

    max_logit = float("-inf")
    acc = tl.zeros([HEAD_SZ], dtype=tl.float32)
    global_exp_sum = tl.zeros([1], dtype=tl.float32)

    #load max_logits [MAX_NUM_SEQ_PARTITIONS_POW2]
    max_logits_offs = seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h + seq_part_offs
    max_logits_mask = seq_part_offs < MAX_NUM_SEQ_PARTITIONS
    max_logits = tl.load(
        max_logits_ptr + max_logits_offs,
        mask=max_logits_mask,
        other=float("-inf"),
    )

    #find max_logit
    max_logit = tl.max(max_logits, axis=0)
    #tl.device_print("max_logits", max_logits)
    #tl.device_print("max_logit", max_logit)

    #load exp_sum [MAX_NUM_SEQ_PARTITIONS_POW2]
    exp_sums_offs = seq_idx * stride_exp_sums_n + head_idx * stride_exp_sums_h + seq_part_offs
    exp_sums_mask = seq_part_offs < MAX_NUM_SEQ_PARTITIONS
    exp_sums = tl.load(
        exp_sums_ptr + exp_sums_offs,
        mask=exp_sums_mask,
        other=0.0,
    )

    #rescaled_exp_sum and global_exp_sum
    # [MAX_NUM_SEQ_PARTITIONS_POW2]
    #tl.device_print("exp_sums", exp_sums)
    rescaled_exp_sum = exp_sums * tl.exp(max_logits - max_logit) 
    #tl.device_print("rescaled_exp_sum", rescaled_exp_sum)
    global_exp_sum += tl.sum(rescaled_exp_sum, axis=0)
    rescaled_exp_sum /= global_exp_sum
    #tl.device_print("global_exp_sum", global_exp_sum)
    #load logits
    logits_offs = ( seq_idx * stride_logits_n 
        + head_idx * stride_logits_h 
        + seq_part_offs[:, None] * stride_logits_b
        + head_sz_offs[None, :]
    )
    logits_mask =  seq_part_offs[:, None] < MAX_NUM_SEQ_PARTITIONS

    logits = tl.load(logits_ptr + logits_offs, mask=logits_mask, other=0.0)
    acc += tl.sum(logits * rescaled_exp_sum[:, None], axis=0)

    #store the final output
    #inv_sum = 1.0/ (global_exp_sum + 1e-6)
    out_ptr = seq_idx * stride_out_n + head_idx * stride_out_h + head_sz_offs
    out_mask = (head_sz_offs < HEAD_SZ)
    tl.store(out + out_ptr, acc, mask=out_mask)

@triton.jit
def _paged_attn_decode_v2_w_dot_kernel(
    max_logits_ptr, #[num_seqs, num_kv_heads, max_seq_partitions, query_grp_sz]
    exp_sums_ptr, #[num_seqs, num_kv_heads, max_seq_partitions, query_grp_sz]
    logits_ptr, #[num_seqs, num_kv_heads, max_seq_partitions, query_grp_sz, head_sz]
    q_ptr, #[num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr, #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
    v_cache_ptr, #[num_seqs, num_kv_heads, kv_blk_sz, head_sz]
    ctx_lens_ptr, #[num_seqs]
    blk_tables_ptrs, #[num_seqs, max_num_blks_per_seq]
    attn_scale,
    stride_blk_tables_s,
    stride_blk_tables_nb,
    stride_q_s,
    stride_q_nh,
    stride_q_hs,
    stride_k_s,
    stride_k_nh,
    stride_k_b,
    stride_k_hs,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_logits_hs,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ:tl.constexpr
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    ctx_part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    log2e: tl.constexpr = 1.4426950408889634
    dtype = q_ptr.dtype.element_ty

    ctx_len = tl.load(ctx_lens_ptr + seq_idx)

    ctx_start_idx = ctx_part_idx * SEQ_PARTITION_SZ
    if ctx_start_idx >= ctx_len:
        return

    ctx_end_idx = tl.minimum(ctx_start_idx + SEQ_PARTITION_SZ, ctx_len)

    #This conversion is needed. Otherwise, this causes a compilation error with mismatch types in the loop below
    num_kv_blks = tl.cdiv(ctx_end_idx - ctx_start_idx, KV_BLK_SZ).to(tl.int32)
    
    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)

    kv_offs = (kv_head_idx * stride_k_nh 
        + blk_offs[:, None] * stride_k_b 
        + head_offs[None, :] * stride_k_hs)
    
    q_offs = (seq_idx * stride_q_s 
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh 
        + head_offs[None, :] * stride_q_hs)
    
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_offs[None, :] < HEAD_SZ)
    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = (q * attn_scale).to(dtype)

    max_logit_i = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32) + float("-inf")
    exp_sum_i = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    acc = tl.zeros([QUERY_GRP_SZ_POW2, HEAD_SZ_POW2], dtype=tl.float32)

    kv_blk_start = ctx_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_blk_tables_s
    for b in range(num_kv_blks):
        blk_idx = kv_blk_start + b
        blk_num = tl.load(blk_tables_start_ptr + blk_idx*stride_blk_tables_nb)

        kv_blk_offs = blk_num * stride_k_s + kv_offs
        mask_offset = blk_idx * KV_BLK_SZ + blk_offs
        kv_mask = (mask_offset[:, None] < ctx_len) & (blk_offs[:, None] < KV_BLK_SZ) & (head_offs[None, :] < HEAD_SZ)

        # load k[KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        k = tl.load(k_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0).to(dtype)

        # qk: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        qk = tl.where(mask_offset < ctx_len, qk, float("-inf"))

        max_logit_i_new = tl.maximum(max_logit_i, tl.max(qk, axis=1))

        # p: [QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((max_logit_i - max_logit_i_new) * log2e)
        acc *= alpha[:, None]

        # v: [KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v = tl.load(v_cache_ptr + kv_blk_offs, mask=kv_mask, other=0.0).to(dtype)

        p = p.to(v.dtype)
        acc += tl.dot(p, v, out_dtype=tl.float32)
        
        exp_sum_i = exp_sum_i * alpha + tl.sum(p, axis=1)
        max_logit_i = max_logit_i_new
    acc = acc / exp_sum_i[:, None]

    max_logits_offs = (
            seq_idx * stride_max_logits_s
        +   kv_head_idx * stride_max_logits_nh
        +   ctx_part_idx * stride_max_logits_p
        +   q_grp_offs
    )
    m_grp_mask =  q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs, max_logit_i, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs, exp_sum_i, mask=m_grp_mask)

    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        ctx_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_offs[None, :] 
    )

    tl.store(logits_ptr + logits_offs, acc, mask=q_mask)


@triton.jit
def _paged_attn_decode_v2_w_dot_reduce_kernel(
    out_ptr, # [num_seqs, num_kv_heads, query_grp_sz, head_sz]
    max_logits_ptr, # [num_seqs, num_kv_heads, max_seq_partitions, query_grp_sz]
    exp_sums_ptr, # [num_seqs, num_kv_heads, max_seq_partitions, query_grp_sz]
    logits_ptrs, # [num_seqs, num_kv_heads, max_seq_partitions, query_grp_sz, head_sz]
    ctx_lens_ptr, #[num_seqs]
    stride_max_logits_s,
    stride_max_logits_h,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    stride_o_s,
    stride_o_h,
    stride_o_g,
    HEAD_SZ: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    ctx_len = tl.load(ctx_lens_ptr + seq_idx)

    part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ)
    #group_head_offs = tl.arange(0, QUERY_GRP_SZ_POW2[:, None]) * HEAD_SZ + tl.arange(0, HEAD_SZ)[None, :]
    #group_mask = tl.arange(0, QUERY_GRP_SZ_POW2)[:, None] < QUERY_GRP_SZ

    #get global max logit
    max_logits_offs = (seq_idx*stride_max_logits_s 
                + kv_head_idx*stride_max_logits_h 
                + part_offs[:, None]*stride_max_logits_p 
                + q_grp_offs[None, :])
    ms_mask = (part_offs[:, None] < MAX_NUM_SEQ_PARTITIONS) & (q_grp_offs[None, :] < QUERY_GRP_SZ)
    # max_logits: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    max_logits = tl.load(max_logits_ptr + max_logits_offs, mask=ms_mask, other=float("-inf"))
    # max_logit: [QUERY_GRP_SZ_POW2]
    ml = tl.max(max_logits, axis=0)

    #Rescale the exp sums and compute the global sum
    # exp_sums: [MAX_NUM_SEQ_PARTITIONS, QUERY_GRP_SZ_POW2]
    exp_sums = tl.load(exp_sums_ptr + max_logits_offs, mask=ms_mask, other=0.0)
    exp_sums *= tl.exp(max_logits - ml[None, :])

    # exp_sum: [QUERY_GRP_SZ_POW2]
    exp_sum = tl.sum(exp_sums, axis=0) 

    # p: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    p = exp_sums / exp_sum[None, :]
    p = tl.reshape(p, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2, 1))

    #logits_offset
    logits_offset = (seq_idx * stride_logits_s  
                + kv_head_idx * stride_logits_h 
                + part_offs[:, None, None] * stride_logits_p 
                + q_grp_offs[None, :, None] * stride_logits_g
                + head_offs[None, None, :]
    )
    #load logits
    logits_mask = (part_offs[:, None] < MAX_NUM_SEQ_PARTITIONS) & (q_grp_offs[None, :] < QUERY_GRP_SZ)
    logits = tl.load(logits_ptrs + logits_offset, mask=logits_mask[:, :, None], other=0.0)

    #out: [QUERY_GRP_SZ_POW2, HEAD_SZ]
    out = tl.sum((logits * p).to(tl.float32), axis=0)

    #store output
    out_offs = (seq_idx * stride_o_s
            + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_h
            +  head_offs[None, :] 
    )
    tl.store(out_ptr + out_offs, out, mask=(q_grp_offs[:, None] < QUERY_GRP_SZ))

def paged_attention_reference(
    output,  #[num_seqs, num_q_heads, head_sz]
    query,  #[num_seqs, num_kv_heads, head_sz], 
    k_cache, #[num_seqs, num_kv_heads, head_sz/x, blk_sz, x]
    v_cache, #[num_seqs, num_kv_heads, head_sz, blk_sz]
    blk_tables, #[num_seq, max_num_blks_per_seq]
    ctx_lens, #[num_seqs]
) -> None:
    num_q_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    head_sz = v_cache.shape[2]
    kv_blk_sz = v_cache.shape[3]

    #print(f"ref: query={query}")
    #print(f"ref: blk_tables={blk_tables}")
    num_seqs = query.shape[0]
    for s in range(num_seqs):
        q = query[s].unsqueeze(0)
        blk_tbl = blk_tables[s]
        ctx_len = ctx_lens[s]

        keys = []
        values = []
        for j in range(ctx_len):
            blk_number = int(blk_tbl[j // kv_blk_sz])
            blk_offset = j % kv_blk_sz

            k = k_cache[blk_number, :, :, blk_offset, :]
            k = k.reshape(num_kv_heads, head_sz)
            if q_grp_sz != 1:
                k = k.repeat_interleave(q_grp_sz, 0)
            keys.append(k)

            v = v_cache[blk_number, :, :, blk_offset]
            if q_grp_sz != 1:
                v = v.repeat_interleave(q_grp_sz, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        #print(f"q.shape={q.shape}")
        #print(f"ref{s} key.shape={keys.shape}")
        #print(f"ref{s} values.shape={values.shape}")

        scale = 1.0 / (head_sz**0.5)
        q = q * scale
        #print(f"ref{s}: q={q}")
        attn = torch.einsum("qhd,khd->hqk", q, keys)
        #print(f"ref{s}: qk={attn}")
        attn = torch.softmax(attn, dim=-1)
        #print(f"ref{s}: sm(qk)={attn}")
        out = torch.einsum("hqk, khd->qhd", attn,values)
        #print(f"ref{s}: out={out.shape}")
        out = out.view(num_q_heads, head_sz)
        output[s].copy_(out, non_blocking=True)



@pytest.mark.parametrize('B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, dtype', [
            #basic
            (1, 1, 1, 1, 1, 1, torch.float32),
            (1, 1, 1, 1, 1, 10, torch.float32),

            #H_Q and H_KV > 1
            (1, 4, 4, 1, 1, 1, torch.float32),
            (1, 4, 4, 1, 1, 10, torch.float32), #FAILED mistmatch

            #Head_dim > 1
            (1, 1, 1, 8, 1, 1, torch.float32),
            (1, 1, 1, 8, 1, 10, torch.float32), 

            #H_Q and H_KV > 1 and Head_dim > 1
            (1, 4, 4, 8, 1, 1, torch.float32),
            (1, 4, 4, 8, 1, 10, torch.float32), #abort
            
            #H_Q and H_KV > 1 and Head_dim > 1 and KV_BLK_SZ > 1
            (1, 1, 1, 1, 1, 1, torch.float32),
            (1, 1, 1, 1, 4, 4, torch.float32), 
            (1, 1, 1, 1, 4, 8, torch.float32), 
            (1, 1, 1, 1, 4, 10, torch.float32),
            (1, 1, 1, 1, 16, 1, torch.float32), 
            (1, 1, 1, 1, 16, 8, torch.float32), 
            (1, 1, 1, 1, 16, 16, torch.float32), 
            (1, 1, 1, 1, 16, 32, torch.float32), 
            (1, 1, 1, 1, 16, 30, torch.float32), 
            (1, 1, 1, 1, 16, 56, torch.float32), 
            (1, 1, 1, 1, 16, 128, torch.float32), 

            #GQA Basic
            (1, 2, 1, 16, 16, 1, torch.float32),
            (1, 2, 1, 16, 16, 10, torch.float32),
           
            #GQA Basic
            (1, 4, 2, 16, 16, 1, torch.float32),
            (1, 4, 2, 16, 16, 10, torch.float32),
            (1, 4, 2, 128, 16, 1, torch.float32),
            (1, 4, 2, 128, 16, 10, torch.float32),
            (1, 6, 2, 128, 16, 1, torch.float32),
            (1, 6, 2, 128, 16, 10, torch.float32),
            (1, 6, 2, 128, 16, 16, torch.float32),
            (1, 6, 2, 128, 16, 30, torch.float32),
            (1, 6, 2, 128, 16, 32, torch.float32),
            (1, 6, 2, 128, 16, 48, torch.float32),
            (1, 6, 2, 128, 16, 56, torch.float32),
            (1, 6, 2, 128, 16, 64, torch.float32),
            (1, 6, 2, 128, 16, 128, torch.float32),
            (1, 8, 2, 128, 16, 128, torch.float32),
            ])
def test_paged_attn(B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN, dtype):
    torch.set_printoptions(threshold=100000)
    num_blocks = 4

    query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")
    #query = torch.tensor([[[0.3058]]], dtype=dtype, device="cuda")
    x = min(D, 16 // torch.tensor([], dtype=dtype).element_size())
    #print(f"x={x}")
    key_cache = torch.randn(num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=dtype, device="cuda")
    value_cache = torch.randn(num_blocks, H_KV, D, KV_BLK_SZ, dtype=dtype, device="cuda")

    ''' 
    key_cache=torch.tensor(
       [[[[[-0.4553],
           [ 0.2596],
           [ 0.3202],
           [ 0.8652]]]],
        [[[[ 0.8952],
           [ 0.4952],
           [ 0.2480],
           [ 0.8029]]]],
        [[[[ 0.0860],
           [ 1.1698],
           [ 0.0634],
           [ 0.5278]]]],
        [[[[ 0.2714],
           [ 0.8718],
           [-0.8499],
           [ 1.2258]]]]],dtype=dtype, device='cuda:0')
    value_cache=torch.tensor([[[[-1.3628,  0.3893,  0.8341, -1.0391]]],
        [[[ 0.9113,  0.8691,  0.4231, -0.2141]]],
        [[[ 0.3029,  0.1333,  0.6171,  0.7549]]],
        [[[ 1.3372, -0.1279,  0.5395,  0.1134]]]],dtype=dtype, device='cuda:0')
    '''
    #print(f"key_cache={key_cache}")
    #print(f"value_cache={value_cache}")
    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()

    context_lens = torch.full((B, ), SEQ_LEN, device="cuda")
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ
    #print(f"context_lens={context_lens}")
    #print(f"max_num_blks_per_seq={max_num_blks_per_seq}")

    block_tables = []
    for i in range(B):
        block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")
    #block_tables = torch.tensor([[0, 2, 3, 1, 0, 1]], device='cuda:0', dtype=torch.int32)
    #block_tables = torch.tensor([[2]], device='cuda:0', dtype=torch.int32)
    attn_scale = 1.0 / (D**0.5)
    #print(f"q.shape={query.shape}")
    #print(f"q={query}")
    #print(f"key_cache_tri.shape={key_cache_tri.shape}")
    #print(f"key_cache_tri={key_cache_tri}")
    #print(f"value_cache_tri.shape={value_cache_tri.shape}")
    #print(f"value_cache_tri={value_cache_tri}")
    #print(f"block_tables.shape={block_tables.shape}")
    #print(f"block_tables={block_tables}")

    torch_output = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
    paged_attention_reference(torch_output, query, key_cache, value_cache, block_tables, context_lens)
    print(f"torch_output={torch_output}")


    triton_output = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
    paged_attention(triton_output, query, key_cache_tri, value_cache_tri, context_lens, block_tables, attn_scale, max_context_len)
    print(f"triton_output={triton_output}")

    assert torch.allclose(triton_output, torch_output, rtol=1e-02, atol=1e-02)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def run_benchmark(args):
    config = []
    val = args.batch_sz_start
    x_vals_list = []
    while val <= args.batch_sz_end:
        x_vals_list.append(val)
        val *= args.batch_sz_step
    pa_args = {'H_Q': args.num_q_heads, "Q_GRP_SZ": args.q_grp_sz, "D": args.head_dim}
    plot_name = str("pagedattn-performance_" + args.dtype + "_NUM_Q_HEADS-" + str(args.num_q_heads) + "_Q_GRP_SZ-" +
                    str(args.q_grp_sz) + "_B-" + str(args.batch_sz_start) + "-" + str(args.batch_sz_end) + "-" +
                    str(args.batch_sz_step) + "_HEAD_DIM-" + str(args.head_dim))
    x_names = ['B']
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
            args=pa_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(B, H_Q, Q_GRP_SZ, D, provider):
        seq_len = 10
        total_tokens = B * seq_len
        H_KV = H_Q // Q_GRP_SZ
        print(f"B={B}, H_Q={H_Q}, H_KV={H_KV}, Q_GRP_SZ={Q_GRP_SZ} D={D}")
        sm_scale = 1.0 / (D**0.5)
        #print(f"benchmark: sm_scale={sm_scale}")
        quantiles = [0.5, 0.2, 0.8]

        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

        req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
        b_req_idx = torch.arange(B, device="cuda")
        b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
        b_seq_len = torch.full((B, ), seq_len, device="cuda")
        attn_logits = torch.empty((H_Q, D), dtype=dtype, device="cuda")

        output = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: paged_attention_reference(q, k_buffer, v_buffer, output, req_to_token, b_req_idx, b_seq_len),
                warmup=20,
                rep=100,
                quantiles=quantiles,
            )
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: decode_attention_fwd(q, k_buffer, v_buffer, output, req_to_token, b_req_idx, b_start_loc,
                                             b_seq_len, attn_logits, total_tokens, sm_scale),
                warmup=20,
                rep=100,
                quantiles=quantiles,
            )

        def ms2us(ms):
            return ms * 1000

        return ms2us(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark PagedAttention",
        allow_abbrev=False,
    )

    parser.add_argument('-b', "--batch_sz_start", default="1", type=int)
    parser.add_argument('-bs', "--batch_sz_step", default="2", type=int)
    parser.add_argument('-be', "--batch_sz_end", default="64", type=int)
    parser.add_argument('-qg', "--q_grp_sz", default="1", type=int)
    parser.add_argument('-qh', "--num_q_heads", default="16", type=int)  #num_kv_heads determined from q_grp_sz
    parser.add_argument('-hd', "--head_dim", default="128", type=int)
    parser.add_argument('-hds', "--head_dim_step", default="2", type=int)
    parser.add_argument('-hde', "--head_dim_end", default="4096", type=int)
    parser.add_argument('-d', "--dtype", default="fp16")

    return parser.parse_args()


def main():
    #args = parse_args()
    #run_benchmark(args)
    #test_paged_attn(2, 2, 2, 1, 1, 10)
    #B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN
    #test_paged_attn(1, 1, 1, 1, 1, 1, torch.float32)
    #test_paged_attn(1, 1, 1, 1, 1, 1)
    #test_paged_attn(1, 1, 1, 1, 16, 32, torch.float32)
    #test_paged_attn(1, 2, 2, 1, 1, 5, torch.float32)
    test_paged_attn(1, 2, 1, 16, 16, 10, torch.float32), #FAILED mistmatch
if __name__ == "__main__":
    sys.exit(main())
