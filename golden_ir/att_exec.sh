#!/bin/bash

#TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_CACHE_DIR=triton_cache python gemm_sub.py \
TRITON_ALWAYS_COMPILE=1 python gemm_sub.py \
  --trans-b \
  --file config.yaml \
  --sub versions/5d_loop_begin.llir \
  --verbose

