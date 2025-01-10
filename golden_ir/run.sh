#!/bin/bash

rm -rf triton_cache

TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_CACHE_DIR=triton_cache python gemm_sub.py \
  --trans-b \
  --file config.yaml \
  --sub versions/2d_global_loads.llir \
  --verbose

#  --sub versions/3_regions.llir \
#  --sub versions/2d_global_loads.llir \
# remember groupm of llir must match grid launch

#  --sub ir/mfma_order_sb_nomask_orig.ttgir \    # =1 575
#  --sub ir/mfma_order_sb_nomask_orig.llir \     # =1 575 passes

#  --use-mask \
#  --verbose
#  --dump-ir amdgcn \
# TRITON_HIP_USE_BLOCK_PINGPONG=1
# MLIR_ENABLE_DUMP=1

