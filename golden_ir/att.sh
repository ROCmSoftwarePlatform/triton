#!/usr/bin/bash

rm -rf triton_cache
# works without sub and without att

#TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_CACHE_DIR=triton_cache rocprofv2 \
AMDGCN=/home/dtanner/repos/rocm_triton/golden_ir/matmul_kernel.amdgcn
rocprofv2 \
  -d att_3a \
  -i att.txt \
  --plugin att ${AMDGCN} \
  --mode file \
  ./att_exec.sh

#  python gemm_sub.py \
#    --trans-b \
#    --file config.yaml \
#    --sub versions/3_regions.llir \
#    --verbose

#  --mode file \
#  --sub versions/3_regions.llir \
#  --sub versions/2d_global_loads.llir \
