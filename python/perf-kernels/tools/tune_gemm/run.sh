#!/usr/bin/bash

python tune_gemm.py \
  --gemm_size_file config.yaml \
  --iters 1 \
  --benchmark

