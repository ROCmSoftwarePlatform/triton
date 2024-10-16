import triton
import triton.language as tl
import torch
import sys
import argparse
import pytest

#This code is derived from sglang and FLASHNN projects
#https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
#https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,  #[num_seqs/batch,num_q_heads, D=head_sz/head_dim]
    K_Buffer,  #[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
    V_Buffer,  #[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    token_blk_logits,  #(num_seqs, num_q_heads, max_len_in_batch/BLOCK, head_sz)
    exp_sums,  #(num_seqs, num_q_heads, max_len_in_batch/BLOCK)
    max_logits,  #(num_seqs, num_q_heads, max_len_in_batch/BLOCK)
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_token_blk_logits_bs,
    stride_token_blk_logits_h,
    stride_token_blk_logits_t,
    stride_exp_sums_bs,
    stride_exp_sums_h,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
):
    cur_batch = tl.program_id(0)  #cur_seq
    cur_head = tl.program_id(1)  #cur head
    cur_token_blk_in_batch = tl.program_id(2)  #token block within seq_len

    reduce_dtype = token_blk_logits.dtype.element_ty

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    #cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    offs_n = cur_token_blk_in_batch * BLOCK_N + tl.arange(0, BLOCK_N)

    #load q ([1xhead_sz/BLOCK_DMODEL])
    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + offs_q).to(reduce_dtype)

    #tl.device_print("q", q)
    #load k_loc from req_to_tokens ie do logical to physical block translation
    k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n, mask=offs_n
                    < cur_batch_seq_len, other=0)
    #offs_k
    offs_buf_k = (k_loc[:, None] * stride_buf_kbs + cur_head * stride_buf_kh + offs_d[None, :])
    #load k [BLOCK_N, BLOCK_DMODEL]
    k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n[:, None] < cur_batch_seq_len) * (offs_d[None, :] < BLOCK_DMODEL),
                other=0.0).to(reduce_dtype)

    #tl.device_print("k", k)
    #calculate qk [BLOCK_N]
    qk = tl.sum((q[None, :] * k), axis=1)
    qk *= sm_scale
    qk = tl.where(offs_n < cur_batch_seq_len, qk, float("-inf"))
    #tl.device_print("qk", qk)

    if logit_cap > 0:
        qk = logit_cap * tanh(qk / logit_cap)

    #find max
    m = tl.max(qk, 0)
    #tl.device_print("m", m)

    #calculate p  [BLOCK_N]
    p = tl.exp(qk - m)
    #tl.device_print("p", p)

    #sum p
    exp_sum = tl.sum(p, 0)
    #tl.device_print("exp_sum", exp_sum)

    #load v_index from req_to_tokens ie do logical to physical block translation
    v_loc = tl.load(Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b + offs_n, mask=offs_n
                    < cur_batch_seq_len, other=0)
    #load v [BLOCK_N, BLOCK_DMODEL]
    offs_buf_v = v_loc[:, None] * stride_buf_kbs + cur_head * stride_buf_kh + offs_d[None, :]
    v = tl.load(V_Buffer + offs_buf_v, mask=(offs_n[:, None] < cur_batch_seq_len) & (offs_d[None, :] < BLOCK_DMODEL),
                other=0.0)
    #tl.device_print("v", v)
    #Calculate qkv[1, BLOCK_DMODEL]. elementwise multiply p[:, None] * v.
    qkv = tl.sum(p[:, None] * v, 0)
    #tl.device_print("qkv", qkv)

    att_value = qkv / exp_sum

    #store exp_sum and max_logits
    offs_exp = cur_batch * stride_exp_sums_bs + cur_head * stride_exp_sums_h + cur_token_blk_in_batch
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, m)

    #store logits
    off_token_blk_logit = cur_batch * stride_token_blk_logits_bs + cur_head * stride_token_blk_logits_h + cur_token_blk_in_batch * stride_token_blk_logits_t + offs_d
    tl.store(token_blk_logits + off_token_blk_logit, att_value, mask=(offs_d < BLOCK_DMODEL))


@triton.jit
def _fwd_kernel_stage2(out, exp_sums,  #[num_batch, num_q_heads, MAX_NUM_TOKEN_BLKS]
                       max_logits,  #[num_batch, num_q_heads, MAX_NUM_TOKEN_BLKS]
                       token_blk_logits,  #[num_batch, num_q_heads, MAX_NUM_TOKEN_BLKS]
                       b_seq_len, stride_obs, stride_oh, stride_exp_sums_bs, stride_exp_sums_h,
                       stride_token_blk_logits_bs, stride_token_blk_logits_h, stride_token_blk_logits_t,
                       BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_POW2: tl.constexpr, BLOCK_N: tl.constexpr,
                       NUM_TOKEN_BLKS: tl.constexpr, NUM_TOKEN_BLKS_POW2: tl.constexpr):

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    offs_tkn = tl.arange(0, NUM_TOKEN_BLKS_POW2)

    #load max_logits for each blk. [NUM_TOKEN_BLKS_POW2]
    ml_start_ptr = max_logits + cur_batch * stride_exp_sums_bs + cur_head * stride_exp_sums_h
    ml = tl.load(ml_start_ptr + offs_tkn, mask=offs_tkn < NUM_TOKEN_BLKS, other=float("-inf"))
    #tl.device_print("ml", ml)

    #calculate global_max_logit [1]
    max_logit = tl.max(ml, axis=0)
    #tl.device_print("max_logit", max_logit)

    #load exp_sums [NUM_TOKEN_BLKS_POW2]
    exp_sum_start_ptr = exp_sums + cur_batch * stride_exp_sums_bs + cur_head * stride_exp_sums_h
    exp_sum = tl.load(exp_sum_start_ptr + offs_tkn, mask=offs_tkn < NUM_TOKEN_BLKS, other=0.0)
    #tl.device_print("exp_sum", exp_sum)

    #rescale each sum and calculate global exp sum
    rescale_factor = tl.exp(ml - max_logit)  #[NUM_TOKEN_BLKS_POW2]
    rescaled_exp_sum = exp_sum * rescale_factor  #[NUM_TOKEN_BLKS_POW2]
    global_exp_sum = tl.sum(rescaled_exp_sum, axis=0)  #[1]

    #load attn_logits
    token_blk_logits_start_ptr = token_blk_logits + cur_batch * stride_token_blk_logits_bs + cur_head * stride_token_blk_logits_h
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    off_token_blk_logit = offs_tkn[:, None] * stride_token_blk_logits_t + offs_d[None, :]
    mask = (offs_tkn[:, None] < NUM_TOKEN_BLKS) & offs_d[None, :] < BLOCK_DMODEL
    attn_logits = tl.load(token_blk_logits_start_ptr + off_token_blk_logit, mask=mask, other=0.0)

    #rescale logits and add all attn_logits [BLOCK_DMODEL]
    acc += tl.sum(attn_logits * rescaled_exp_sum[:, None], axis=0)

    #for token_blk in tl.range(0, NUM_TOKEN_BLKS):
    #    offs_n = token_blk*stride_token_blk_logits_t +  tl.arange(0, BLOCK_DMODEL_POW2)
    #    attn_logit = tl.load(token_blk_logits_start_ptr + offs_n, mask=offs_n < BLOCK_DMODEL, other=0.0)
    #    attn_logit = attn_logit* rescale_factor[token_blk]
    #    acc += attn_logit

    #divide by global_sum
    acc /= (global_exp_sum + 1e-6)
    #tl.device_print("acc", acc)

    #write out output
    offs_o = cur_batch * stride_obs + cur_head * stride_oh + tl.arange(0, BLOCK_DMODEL_POW2)
    tl.store(out + offs_o, acc, mask=(offs_d < BLOCK_DMODEL))


def _decode_reduce_fwd(out, token_blk_logits, exp_sums, max_logits, req_to_token, b_req_idx, b_start_loc, b_seq_len,
                       BLOCK_N, NUM_TOKEN_BLKS, NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL, BLOCK_DMODEL_POW2):
    #BLOCK = 64
    batch, head = b_seq_len.shape[0], token_blk_logits.shape[1]

    #BLOCK_DMODEL = token_blk_logits.shape[-1]
    #BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    grid = (batch, head, 1)
    _fwd_kernel_stage2[grid](out, exp_sums, max_logits, token_blk_logits, b_seq_len, out.stride(0), out.stride(1),
                             exp_sums.stride(0), exp_sums.stride(1), token_blk_logits.stride(0),
                             token_blk_logits.stride(1), token_blk_logits.stride(2), BLOCK_DMODEL=BLOCK_DMODEL,
                             BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2, BLOCK_N=BLOCK_N, NUM_TOKEN_BLKS=NUM_TOKEN_BLKS,
                             NUM_TOKEN_BLKS_POW2=NUM_TOKEN_BLKS_POW2)


def _decode_att_m_fwd(q, k_buffer, v_buffer, token_blk_logits, exp_sums, max_logits, Req_to_tokens, B_req_idx,
                      B_Start_Loc, B_Seqlen, max_len_in_batch, sm_scale, logit_cap, BLOCK_N, NUM_TOKEN_BLKS,
                      NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL, BLOCK_DMODEL_POW2):
    #BLOCK_DMODEL = k_buffer.shape[-1]  #head_sz/head_dim
    #BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    #set up grid (batch, num_q_heads, max_len_in_match/BLOCK)
    num_batch, num_q_heads = B_req_idx.shape[0], q.shape[1]
    #num_token_blks = triton.cdiv(max_len_in_batch, BLOCK_N)

    grid = (num_batch, num_q_heads, NUM_TOKEN_BLKS)

    #print(f"_decode_attn_m_fwd: sm_scale={sm_scale}")
    _fwd_kernel_stage1[grid](q, k_buffer, v_buffer, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
                             token_blk_logits, exp_sums, max_logits, Req_to_tokens.stride(0), q.stride(0), q.stride(1),
                             k_buffer.stride(0), k_buffer.stride(1), token_blk_logits.stride(0),
                             token_blk_logits.stride(1), token_blk_logits.stride(2), exp_sums.stride(0),
                             exp_sums.stride(1), BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
                             BLOCK_N=BLOCK_N, logit_cap=logit_cap)


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    token_blk_logits,
    exp_sums,
    max_logits,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_token_blk_logits_bs,
    stride_token_blk_logits_h,
    stride_token_blk_logits_t,
    stride_token_blk_logits_g,
    stride_exp_sums_bs,
    stride_exp_sums_h,
    stride_exp_sums_t,
    q_grp_sz: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_token_blk_in_batch = tl.program_id(2)

    reduce_dtype = token_blk_logits.dtype.element_ty
    cur_q_heads = cur_kv_head * q_grp_sz + tl.arange(0, BLOCK_H)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    #cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = cur_token_blk_in_batch * BLOCK_N + tl.arange(0, BLOCK_N)

    #load q [BLOCK_H, BLOCK_DMODEL]
    offs_q = cur_batch * stride_qbs + cur_q_heads[:, None] * stride_qh + offs_d[None, :]
    mask = (offs_h[:, None] < q_grp_sz) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(Q + offs_q, mask=mask, other=0.0).to(reduce_dtype)
    #tl.device_print("q", q)

    #load k_loc from req_to_tokens ie do logical to physical block translation
    k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n, mask=offs_n
                    < cur_batch_seq_len, other=0)

    #offs_k
    offs_buf_k = (k_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None])
    #load kt [BLOCK_DMODEL, BLOCK_N]
    k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n[None, :] < cur_batch_seq_len) * (offs_d[:, None] < BLOCK_DMODEL),
                other=0.0).to(reduce_dtype)
    #tl.device_print("k", k)

    #qk using tl.dot[BLOCK_H, BLOCK_N]
    qk = tl.dot(q, k, out_dtype=tl.float32)
    qk *= sm_scale
    if logit_cap > 0:
        qk = logit_cap * tanh(qk / logit_cap)

    qk = tl.where(offs_n < cur_batch_seq_len, qk, float("-inf"))

    #tl.device_print("qk", qk)
    #find max [BLOCK_H]
    m = tl.max(qk, 1)

    #calculate p [BLOCK_H, BLOCK_N]
    p = tl.exp(qk - m[:, None])

    #sum p [BLOCK_H]
    exp_sum = tl.sum(p, 1)

    #load v_index from req_to_tokens ie do logical to physical block translation
    v_loc = tl.load(Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b + offs_n, mask=offs_n
                    < cur_batch_seq_len, other=0)
    #load v [BLOCK_N, BLOCK_DMODEL]
    offs_buf_v = v_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]
    v = tl.load(V_buffer + offs_buf_v, mask=(offs_n[:, None] < cur_batch_seq_len) & (offs_d[None, :] < BLOCK_DMODEL),
                other=0.0).to(reduce_dtype)

    #calculate qkv and attn [BLOCK_H, BLOCK_DMODEL]
    qkv = tl.dot(p, v)
    attn = qkv / exp_sum[:, None]
    #tl.device_print("attn", attn)

    #store exp_sum and max_logits
    offs_exp = cur_batch * stride_exp_sums_bs + cur_kv_head * stride_exp_sums_h + cur_token_blk_in_batch * stride_exp_sums_t
    offs_exp += offs_h
    tl.store(exp_sums + offs_exp, exp_sum)
    tl.store(max_logits + offs_exp, m)

    #store logits
    off_token_blk_logit = cur_batch * stride_token_blk_logits_bs + cur_kv_head * stride_token_blk_logits_h
    off_token_blk_logit += cur_token_blk_in_batch * stride_token_blk_logits_t + (
        offs_h)[:, None] * stride_token_blk_logits_g + offs_d[None, :]
    m = (offs_h[:, None] < q_grp_sz) & (offs_d[None, :] < BLOCK_DMODEL)
    #tl.device_print("m",m)
    tl.store(token_blk_logits + off_token_blk_logit, attn, mask=m)


@triton.jit
def _fwd_grouped_kernel_stage2(
    out,
    exp_sums,
    max_logits,
    token_blk_logits,
    stride_obs,
    stride_oh,
    stride_exp_sums_bs,
    stride_exp_sums_kh,
    stride_exp_sums_t,
    stride_token_blk_logits_bs,
    stride_token_blk_logits_h,
    stride_token_blk_logits_t,
    stride_token_blk_logits_g,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TOKEN_BLKS: tl.constexpr,
    MAX_NUM_TOKEN_BLKS: tl.constexpr,
    Q_GRP_SZ: tl.constexpr,
    Q_GRP_SZ_PADDED: tl.constexpr,
):

    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    #cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    #cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_tkn = tl.arange(0, MAX_NUM_TOKEN_BLKS)
    offs_q_grp = tl.arange(0, Q_GRP_SZ_PADDED)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    #load max_logits for each blk.[NUM_TOKEN_BLKS, Q_GRP_SZ]
    offs_exp = cur_batch * stride_exp_sums_bs + cur_kv_head * stride_exp_sums_kh + offs_tkn[:,
                                                                                            None] * stride_exp_sums_t + offs_q_grp[
                                                                                                None, :]
    mask_exp = (offs_tkn[:, None] < NUM_TOKEN_BLKS) & (offs_q_grp[None, :] < Q_GRP_SZ)
    ml = tl.load(max_logits + offs_exp, mask=mask_exp, other=0.0)
    #tl.device_print("ml", ml)

    #calculate global_max_logit [Q_GRP_SZ]
    max_logit = tl.max(ml, axis=0)
    #tl.device_print("max_logit", max_logit)

    #load exp_sums[NUM_TOKENS_BLKS, Q_GRP_SZ]
    exp_sum = tl.load(exp_sums + offs_exp, mask=mask_exp, other=0.0)

    #rescale each sum and calculate global exp sum
    rescale_factor = tl.exp(ml - max_logit[None, :])  #[NUM_TOKEN_BLKS, Q_GRP_SZ]
    rescaled_exp_sum = exp_sum * rescale_factor  #[NUM_TOKEN_BLKS, Q_GRP_SZ]
    global_exp_sum = tl.sum(rescaled_exp_sum, axis=0)  #[Q_GRP_SZ]
    inv = rescaled_exp_sum / global_exp_sum[None, :]  #[NUM_TOKEN_BLKS, Q_GRP_SZ]
    inv = tl.reshape(inv, (MAX_NUM_TOKEN_BLKS, Q_GRP_SZ_PADDED, 1))

    #load attn_logits [NUM_TOKEN_BLKS, Q_GRP_SZ, BLOCK_DMODEL]
    token_blk_logits_start_ptr = token_blk_logits + cur_batch * stride_token_blk_logits_bs + cur_kv_head * stride_token_blk_logits_h
    offs_l = offs_tkn[:, None, None] * stride_token_blk_logits_t + offs_q_grp[
        None, :, None] * stride_token_blk_logits_g + offs_d[None, None, :]
    mask_l = (offs_tkn[:, None, None] < NUM_TOKEN_BLKS) & (offs_q_grp[None, :, None]
                                                           < Q_GRP_SZ) & (offs_d[None, None, :] < BLOCK_DMODEL)
    logits = tl.load(token_blk_logits_start_ptr + offs_l, mask=mask_l, other=0.0)
    #tl.device_print("logits", logits)

    #rescale logits and add all logits [Q_GRP_SZ, BLOCK_DMODEL]
    logits = tl.sum((logits * inv).to(tl.float32), axis=0)
    #tl.device_print("logits", logits)

    #write out output
    off_o = cur_batch * stride_obs + cur_kv_head * Q_GRP_SZ * stride_oh
    off_o += offs_q_grp[:, None] * stride_oh + offs_d[None, :]
    mask_o = (offs_q_grp[:, None] < Q_GRP_SZ) & (offs_d[None, :] < BLOCK_DMODEL)
    #tl.device_print("mask_o", mask_o)
    tl.store(out + off_o, logits, mask=mask_o)


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    token_blk_logits,
    exp_sums,
    max_logits,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
    BLOCK_N,
    NUM_TOKEN_BLKS,
    NUM_TOKEN_BLKS_POW2,
    BLOCK_DMODEL,
    BLOCK_DMODEL_POW2,
    Q_GRP_SZ,
):
    #BLOCK_DMODEL = k_buffer.shape[-1]  #head_sz/head_dim
    #BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)
    if Q_GRP_SZ < 16:
        BLOCK_H = 16
    else:
        BLOCK_H = triton.next_power_of_2(Q_GRP_SZ)

    #set up grid
    num_batch, num_kv_heads = B_req_idx.shape[0], k_buffer.shape[1]
    grid = (num_batch, num_kv_heads, NUM_TOKEN_BLKS)

    _fwd_grouped_kernel_stage1[grid](q, k_buffer, v_buffer, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc,
                                     B_Seqlen, token_blk_logits, exp_sums, max_logits, Req_to_tokens.stride(0),
                                     q.stride(0), q.stride(1), k_buffer.stride(0), k_buffer.stride(1),
                                     token_blk_logits.stride(0), token_blk_logits.stride(1), token_blk_logits.stride(2),
                                     token_blk_logits.stride(3), exp_sums.stride(0), exp_sums.stride(1),
                                     exp_sums.stride(2), q_grp_sz=Q_GRP_SZ, num_kv_heads=num_kv_heads,
                                     BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2, BLOCK_N=BLOCK_N,
                                     BLOCK_H=BLOCK_H, logit_cap=logit_cap)


def _decode_grouped_reduce_fwd(out, token_blk_logits, exp_sums, max_logits, req_to_token, b_req_idx, b_start_loc,
                               b_seq_len, BLOCK_N, NUM_TOKEN_BLKS, NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
                               Q_GRP_SZ):
    batch, num_kv_heads = b_seq_len.shape[0], token_blk_logits.shape[1]

    BLOCK_H = max(16, triton.next_power_of_2(Q_GRP_SZ))

    grid = (batch, num_kv_heads, 1)
    _fwd_grouped_kernel_stage2[grid](out, exp_sums, max_logits, token_blk_logits, out.stride(0), out.stride(1),
                                     exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
                                     token_blk_logits.stride(0), token_blk_logits.stride(1), token_blk_logits.stride(2),
                                     token_blk_logits.stride(3), BLOCK_DMODEL=BLOCK_DMODEL,
                                     BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2, BLOCK_N=BLOCK_N,
                                     NUM_TOKEN_BLKS=NUM_TOKEN_BLKS, MAX_NUM_TOKEN_BLKS=NUM_TOKEN_BLKS_POW2,
                                     Q_GRP_SZ=Q_GRP_SZ, Q_GRP_SZ_PADDED=BLOCK_H)


def decode_attention_fwd(
    q,  #[num_seqs/batch,num_q_heads, D=head_sz/head_dim]
    k_buffer,  #[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
    v_buffer,  #[total_tokens,num_kv_heads, D=head_sz/head_dim]
    o,  #[, num_q_heads, D=head_sz/head_dim]
    req_to_token,  #[num_seq/batch, seq_len for each seq]
    b_req_idx,  #[num_seq],
    b_start_loc,  #[num_seq], start token location in k/v_buffer for this seq/batch
    b_seq_len,  #[num_seq], seq length for this batch/seq
    attn_logits,  # #[num_head, total_num_tokens]
    max_len_in_batch,
    sm_scale,
    logit_cap=0.0,
):
    num_q_heads = q.shape[1]
    num_kv_heads = k_buffer.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    num_batch = b_req_idx.shape[0]

    BLOCK_N = 16
    NUM_TOKEN_BLKS = triton.cdiv(max_len_in_batch, BLOCK_N)
    NUM_TOKEN_BLKS_POW2 = triton.next_power_of_2(NUM_TOKEN_BLKS)

    BLOCK_DMODEL = q.shape[2]
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(BLOCK_DMODEL)

    #print(f"decode_attention_fwd: sm_scale={sm_scale}")
    if q_grp_sz == 1:
        # MHA
        exp_sums = torch.empty((num_batch, num_q_heads, NUM_TOKEN_BLKS), device="cuda")
        max_logits = torch.empty((num_batch, num_q_heads, NUM_TOKEN_BLKS), device="cuda")
        token_blk_logits = torch.empty((num_batch, num_q_heads, NUM_TOKEN_BLKS, BLOCK_DMODEL), device="cuda")
        _decode_att_m_fwd(q, k_buffer, v_buffer, token_blk_logits, exp_sums, max_logits, req_to_token, b_req_idx,
                          b_start_loc, b_seq_len, max_len_in_batch, sm_scale, logit_cap, BLOCK_N, NUM_TOKEN_BLKS,
                          NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL, BLOCK_DMODEL_POW2)
        #print(f"token_blk_logits={token_blk_logits}")
        _decode_reduce_fwd(o, token_blk_logits, exp_sums, max_logits, req_to_token, b_req_idx, b_start_loc, b_seq_len,
                           BLOCK_N, NUM_TOKEN_BLKS, NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL, BLOCK_DMODEL_POW2)
    # GQA
    else:
        exp_sums = torch.empty((num_batch, num_kv_heads, NUM_TOKEN_BLKS, q_grp_sz), device="cuda")
        max_logits = torch.empty((num_batch, num_kv_heads, NUM_TOKEN_BLKS, q_grp_sz), device="cuda")
        token_blk_logits = torch.empty((num_batch, num_kv_heads, NUM_TOKEN_BLKS, q_grp_sz, BLOCK_DMODEL), device="cuda")
        _decode_grouped_att_m_fwd(q, k_buffer, v_buffer, token_blk_logits, exp_sums, max_logits, req_to_token,
                                  b_req_idx, b_start_loc, b_seq_len, max_len_in_batch, sm_scale, logit_cap, BLOCK_N,
                                  NUM_TOKEN_BLKS, NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL, BLOCK_DMODEL_POW2, q_grp_sz)
        #print(f"token_blk_logits={token_blk_logits}")
        _decode_grouped_reduce_fwd(o, token_blk_logits, exp_sums, max_logits, req_to_token, b_req_idx, b_start_loc,
                                   b_seq_len, BLOCK_N, NUM_TOKEN_BLKS, NUM_TOKEN_BLKS_POW2, BLOCK_DMODEL,
                                   BLOCK_DMODEL_POW2, q_grp_sz)


def paged_attention_reference(
        q,  #[num_seqs/batch,num_q_heads, D=head_sz/head_dim]
        k_buffer,  #[total_tokens,num_kv_heads, D=head_sz/head_dim], # total_tokens=sum of seq_len for each seq
        v_buffer,  #[total_tokens,num_kv_heads, D=head_sz/head_dim]
        output,  #[, num_q_heads, D=head_sz/head_dim]
        req_to_tokens,  #[num_seq/batch, seq_len for each seq]
        b_req_idx,  #[num_seq],
        seq_lens: torch.Tensor,  # [num_seq]
) -> None:
    num_q_heads = q.shape[1]
    num_kv_heads = k_buffer.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    BLOCK_DMODEL = v_buffer.shape[2]

    num_batch = q.shape[0]
    for b in range(num_batch):
        #Get q
        query = q[b].unsqueeze(0)  # [1, q_heads, head_size]
        seq_len = int(seq_lens[b])

        keys = []
        values = []
        #read all k and v locations and append them to a list
        req_idx = b_req_idx[b]
        for j in range(seq_len):
            k_loc = req_to_tokens[req_idx, j]
            k = k_buffer[k_loc].reshape(num_kv_heads, BLOCK_DMODEL)
            if q_grp_sz != 1:
                k = k.repeat_interleave(q_grp_sz, 0)
            keys.append(k)

            v_loc = req_to_tokens[req_idx, j]
            v = v_buffer[v_loc].reshape(num_kv_heads, BLOCK_DMODEL)
            if q_grp_sz != 1:
                v = v.repeat_interleave(q_grp_sz, 0)
            values.append(v)

        #stack k and v
        keys = torch.stack(keys, dim=0)  # [seq_len, num_kv_heads, BLOCK_DMODEL]
        values = torch.stack(values, dim=0)

        scale = 1.0 / (BLOCK_DMODEL**0.5)

        #print(f"ref: scale={scale}")
        query = query * scale
        #print(f"ref: query={query}")
        attn = torch.einsum("qhd,khd->hqk", query, keys)
        #print(f"ref: qk={attn}")
        attn = torch.softmax(attn, dim=-1)
        #print(f"ref: sm(qk)={attn}")
        out = torch.einsum("hqk,khd->qhd", attn, values)
        #print(f"ref: sm(qk)v={out}")
        out = out.view(num_q_heads, BLOCK_DMODEL)
        output[b].copy_(out, non_blocking=True)


@pytest.mark.parametrize('B, H_Q, H_KV, D', [(1, 1, 1, 16), (1, 4, 4, 16), (4, 4, 4, 16), (4, 16, 16, 64),
                                             (1, 2, 1, 16), (2, 2, 1, 16), (1, 8, 2, 16), (2, 8, 2, 16), (2, 16, 4, 16),
                                             (4, 16, 4, 16), (4, 64, 4, 32), (32, 128, 2, 128), (64, 128, 2, 256),
                                             (64, 128, 128, 16), (3, 4, 4, 32), (5, 64, 16, 1024)])
def test_decode_attn_fwd(B, H_Q, H_KV, D):
    torch.set_printoptions(threshold=100000)
    dtype = torch.float16
    seq_len = 5
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D**0.5)

    #print(f"test_decode_attn_fwd: sm_scale={sm_scale}")
    q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    #print(f"q={q}")
    #print(f"k_buffer={k_buffer}")
    #print(f"v_buffer={v_buffer}")

    req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
    b_req_idx = torch.arange(B, device="cuda")
    b_start_loc = torch.arange(0, total_tokens, seq_len, device="cuda")
    b_seq_len = torch.full((B, ), seq_len, device="cuda")
    attn_logits = torch.empty((H_Q, D), dtype=dtype, device="cuda")

    torch_output = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
    paged_attention_reference(q, k_buffer, v_buffer, torch_output, req_to_token, b_req_idx, b_seq_len)
    #print(f"torch_output={torch_output}")

    triton_output = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")
    decode_attention_fwd(q, k_buffer, v_buffer, triton_output, req_to_token, b_req_idx, b_start_loc, b_seq_len,
                         attn_logits, seq_len, sm_scale)
    #print(f"triton_output={triton_output}")

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
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
