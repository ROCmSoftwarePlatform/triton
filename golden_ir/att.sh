#!/usr/bin/bash

rm -rf triton_cache
rm -rf ~/.triton/cache

#TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS=1 TRITON_CACHE_DIR=triton_cache rocprofv2 \

TRITON_ALWAYS_COMPILE=1 rocprofv2 \
  -d att_6d_local_prefetch \
  -i att.txt \
  --plugin att auto \
  --mode file \
  ./att_exec.sh

