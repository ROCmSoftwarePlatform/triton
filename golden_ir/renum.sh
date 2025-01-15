#!/usr/bin/bash

INPUT="matmul/6c_local_prefetch"
OUTPUT="matmul/6d_local_prefetch"
python renumber_llir.py /home/dtanner/repos/rocm_triton/golden_ir/${INPUT}.llir
mv /home/dtanner/repos/rocm_triton/golden_ir/${INPUT}.llir.renumbered \
  /home/dtanner/repos/rocm_triton/golden_ir/${OUTPUT}.llir 
