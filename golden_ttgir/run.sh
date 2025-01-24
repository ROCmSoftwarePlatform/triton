#!/bin/bash

rm -rf ~/.triton/cache/


TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_ALWAYS_COMPILE=1 HIP_VISIBLE_DEVICES=0 python gemm_sub.py \
  --trans-b \
  --sub irp/10.ttgir \
  --file config.yaml \
  --verbose

#  --sub ir/7.ttgir \
#  --use-mask \
# remember groupm of llir must match grid launch

#  --use-mask \
#  --verbose
#  --dump-ir amdgcn \
# TRITON_HIP_USE_BLOCK_PINGPONG=1
# MLIR_ENABLE_DUMP=1
# TRITON_CACHE_DIR=triton_cache

