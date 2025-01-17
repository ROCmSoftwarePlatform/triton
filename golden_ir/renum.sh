#!/usr/bin/bash

INPUT="matmul/8k"
OUTPUT="matmul/8l"
python renumber_llir.py /home/dtanner/repos/rocm_triton/golden_ir/${INPUT}.llir
mv /home/dtanner/repos/rocm_triton/golden_ir/${INPUT}.llir.renumbered \
  /home/dtanner/repos/rocm_triton/golden_ir/${OUTPUT}.llir 
