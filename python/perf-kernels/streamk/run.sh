#!/usr/bin/bash

python tune_streamk.py \
    --gemm_size_file config.yaml \
    --benchmark \
    -col_b \
    --ngpus 1 \
    --jobs 1

