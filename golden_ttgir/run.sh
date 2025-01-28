#!/bin/bash

rm -rf ~/.triton/cache/

GPU=3
#sudo rocm-smi --setperfdeterminism 1400 -d ${GPU}

#TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_ALWAYS_COMPILE=1 HIP_VISIBLE_DEVICES=${GPU} gdb -ex r --args python gemm_sub.py \
TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_ALWAYS_COMPILE=1 HIP_VISIBLE_DEVICES=${GPU} python gemm_sub.py \
  --trans-b \
  --file config.yaml \
  --sub irp/16.ttgir \
  --verbose

#  --sub irp/14.ttgir \
#  --sub irp/10.ttgir \
#sudo rocm-smi --resetperfdeterminism -d ${GPU}

#  --sub ir/7.ttgir \
#  --use-mask \
# remember groupm of llir must match grid launch

#  --use-mask \
#  --verbose
#  --dump-ir amdgcn \
# TRITON_HIP_USE_BLOCK_PINGPONG=1
# MLIR_ENABLE_DUMP=1
# TRITON_CACHE_DIR=triton_cache

