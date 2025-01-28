#!/usr/bin/bash

rm -rf triton_cache
rm -rf ~/.triton/cache

#TRITON_CACHE_DIR=triton_cache rocprofv2 \

TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_ALWAYS_COMPILE=1 rocprofv2 \
  -d att_irp_11_4x2 \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./run.sh

