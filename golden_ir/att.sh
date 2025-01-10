#!/usr/bin/bash

# works without sub and without att

#TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_CACHE_DIR=triton_cache rocprofv2 \
AMDGCN=/home/dtanner/repos/rocm_triton/golden_ir/matmul_kernel.amdgcn
rocprofv2 \
  -d att_2d \
  -i att.txt \
  --plugin att ${AMDGCN} \
  --mode file \
  python gemm_sub.py \
  --trans-b \
  --file config.yaml \
  --sub versions/2d_global_loads.llir \
  --verbose

#  --sub versions/2d_global_loads.llir \
