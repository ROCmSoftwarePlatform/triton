#!/bin/bash

#TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_CACHE_DIR=triton_cache python gemm_sub.py \
python gemm_sub.py \
  --trans-b \
  --file config.yaml \
  --sub versions/4d_loop_end.llir \
  --verbose

