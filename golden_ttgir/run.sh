#!/bin/bash

rm -rf ~/.triton/cache/

TRITON_ALWAYS_COMPILE=1 python gemm_sub.py \
  --trans-b \
  --file config.yaml \
  --dump-ir amdgcn \
  --verbose

#  --use-mask \
#  --sub versions/2d_global_loads.llir \
#  --sub versions/3_regions.llir \
#  --sub versions/2d_global_loads.llir \
# remember groupm of llir must match grid launch

#  --use-mask \
#  --verbose
#  --dump-ir amdgcn \
# TRITON_HIP_USE_BLOCK_PINGPONG=1
# MLIR_ENABLE_DUMP=1
# TRITON_CACHE_DIR=triton_cache

