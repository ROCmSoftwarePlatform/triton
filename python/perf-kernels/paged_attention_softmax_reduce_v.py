import triton
import triton.language as tl
import torch
import sys
import argparse
import pytest

#This code is derived from sglang project
#https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/decode_attention.py


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    att_stride_h,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    reduce_dtype = Att_Out.dtype.element_ty
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark).to(reduce_dtype)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (k_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :])
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n_new[:, None] < cur_batch_end_index) & (offs_d[None, :] < Lk),
            other=0.0,
        ).to(reduce_dtype)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale

        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)

        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)


@triton.jit
def _fwd_kernel_stage2(
    logits,
    V_Buffer,
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    stride_logic_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_obs,
    stride_oh,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(
            Req_to_tokens + cur_batch_req_idx * stride_req_to_token_b + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        qk = tl.load(
            logits + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n),
            mask=start_n + offs_n < cur_batch_seq_len,
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_buf_vbs, mask=(offs_d[None, :] < Lv))
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_d < Lv))


def _decode_att_m_fwd(
    q,
    k_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK))
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )


def _decode_softmax_reducev_fwd(
    logits,
    v_buffer,
    o,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logits.shape[0]
    grid = (batch, head, 1)
    kv_group_num = logits.shape[0] // v_buffer.shape[1]

    num_warps = 1

    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lv)

    _fwd_kernel_stage2[grid](
        logits,
        v_buffer,
        o,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        logits.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        o.stride(0),
        o.stride(1),
        req_to_tokens.stride(0),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    att_stride_h,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    start_n = tl.program_id(2)

    reduce_dtype = Att_Out.dtype.element_ty
    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :])

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + offs_q + start_mark, mask=(mask_h[:, None]) & (offs_d[None, :] < Lk)).to(reduce_dtype)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        offs_buf_k = (k_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None])
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(offs_n_new[None, :] < cur_batch_end_index) & (offs_d[:, None] < Lk),
            other=0.0,
        ).to(reduce_dtype)
        qk = tl.dot(q, k)
        if BLOCK_DPE > 0:
            qpe = tl.load(Q + off_qpe + start_mark, mask=mask_h[:, None]).to(reduce_dtype)
            offs_buf_kpe = (k_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None])
            kpe = tl.load(
                K_Buffer + offs_buf_kpe,
                mask=offs_n_new[None, :] < cur_batch_end_index,
                other=0.0,
            ).to(reduce_dtype)
            qk += tl.dot(qpe, kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        offs_o = cur_head[:, None] * att_stride_h + (cur_batch_in_all_start_index + offs_n[None, :])

        tl.store(
            Att_Out + offs_o,
            qk,
            mask=mask_h[:, None] & (offs_n_new[None, :] < cur_batch_end_index),
        )


@triton.jit
def _fwd_grouped_kernel_stage2(
    logits,
    V_Buffer,
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    stride_logic_h,
    stride_buf_vbs,
    stride_buf_vh,
    stride_obs,
    stride_oh,
    stride_req_to_token_b,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    cur_head = cur_kv_head * kv_group_num + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_kv_head + 1) * kv_group_num
    mask_h = mask_h & (cur_head < q_head_num)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(
            Req_to_tokens + cur_batch_req_idx * stride_req_to_token_b + (start_n + offs_n),
            mask=(start_n + offs_n) < cur_batch_seq_len,
            other=0,
        )

        offs_qk = cur_head[:, None] * stride_logic_h + (cur_batch_start_loc + start_n + offs_n[None, :])

        qk = tl.load(
            logits + offs_qk,
            mask=mask_h[:, None] & (start_n + offs_n[None, :] < cur_batch_seq_len),
            other=float("-inf"),
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        e_sum = e_sum * old_scale + tl.sum(p, 1)
        v = tl.load(v_ptrs + v_index[:, None] * stride_buf_vbs, mask=(offs_d[None, :] < Lv))
        p = p.to(v.dtype)
        acc = acc * old_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max

    acc = acc / e_sum[:, None]
    off_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(mask_h[:, None]) & (offs_d[None, :] < Lv))


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    max_len_in_batch,
    sm_scale,
    logit_cap,
):
    BLOCK = 64
    Lk = k_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0

    batch, head_num = B_req_idx.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        triton.cdiv(max_len_in_batch, BLOCK),
    )

    num_warps = 4

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Start_Loc,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        att_out.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
    )


def _decode_grouped_softmax_reducev_fwd(
    logits,
    v_buffer,
    o,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    b_seq_len,
):
    BLOCK = 128
    batch, head_num = b_seq_len.shape[0], logits.shape[0]
    kv_group_num = logits.shape[0] // v_buffer.shape[1]
    BLOCK_H = max(16, triton.next_power_of_2(kv_group_num))
    grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), 1)

    num_warps = 8

    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lv)

    _fwd_grouped_kernel_stage2[grid](
        logits,
        v_buffer,
        o,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        logits.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        o.stride(0),
        o.stride(1),
        req_to_tokens.stride(0),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        Lv=Lv,
        num_warps=num_warps,
        num_stages=1,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    attn_logits,
    max_len_in_batch,
    sm_scale,
    logit_cap=0.0,
):
    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        _decode_att_m_fwd(
            q,
            k_buffer,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
        )
        _decode_softmax_reducev_fwd(
            attn_logits,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
        )
    else:
        # GQA/MQA/MLA
        _decode_grouped_att_m_fwd(
            q,
            k_buffer,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            max_len_in_batch,
            sm_scale,
            logit_cap,
        )
        _decode_grouped_softmax_reducev_fwd(
            attn_logits,
            v_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_start_loc,
            b_seq_len,
        )


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
    plot_name = str("pagedattn-softmax_reduce_v-perf_" + args.dtype + "_NUM_Q_HEADS-" + str(args.num_q_heads) +
                    "_Q_GRP_SZ-" + str(args.q_grp_sz) + "_B-" + str(args.batch_sz_start) + "-" +
                    str(args.batch_sz_end) + "-" + str(args.batch_sz_step) + "_HEAD_DIM-" + str(args.head_dim))
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
