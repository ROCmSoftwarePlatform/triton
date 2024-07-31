#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl

@triton.jit
def mstore2d(
        registers,
        REG_ROWS : tl.constexpr,
        REG_COLS : tl.constexpr,
        o_base,
        o_start_row,
        o_start_col,
        o_rows,
        o_cols,
        stride_row,
        stride_col,
):
    off_rows = tl.arange(0, REG_ROWS) + o_start_row
    off_cols = tl.arange(0, REG_COLS) + o_start_col
    o_ptrs = o_base + off_rows[:, None] * stride_row + off_cols[None, :] * stride_col
    o_ptrs_mask = tl.full([REG_ROWS, REG_COLS], 1, dtype=tl.int1)
    row_overflow = o_start_row + REG_ROWS - o_rows
    if row_overflow > 0:
        o_ptrs_mask = o_ptrs_mask & (off_rows[:, None] < o_rows)
    col_overflow = o_start_col + REG_COLS - o_cols
    if col_overflow > 0:
        o_ptrs_mask = o_ptrs_mask & (off_cols[None, :] < o_cols)
    tl.store(o_ptrs, registers, mask=o_ptrs_mask)
    return o_ptrs, o_ptrs_mask
