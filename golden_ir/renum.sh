#!/usr/bin/bash

INPUT="matmul/7d"
OUTPUT="matmul/7f"
python renumber_llir.py /home/dtanner/repos/rocm_triton/golden_ir/${INPUT}.llir
mv /home/dtanner/repos/rocm_triton/golden_ir/${INPUT}.llir.renumbered \
  /home/dtanner/repos/rocm_triton/golden_ir/${OUTPUT}.llir 
