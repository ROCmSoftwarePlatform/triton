#!/usr/bin/bash

TRITON_ALWAYS_COMPILE=1 python tune_gemm.py \
  --gemm_size_file config.yaml \
  --iters 300 \
  --benchmark

