#!/usr/bin/bash

INPUT="4_loop_end"
OUTPUT="4b_loop_end"
python renumber_llir.py /home/dtanner/repos/rocm_triton/golden_ir/versions/${INPUT}.llir
mv /home/dtanner/repos/rocm_triton/golden_ir/versions/${INPUT}.llir.renumbered \
  /home/dtanner/repos/rocm_triton/golden_ir/versions/${OUTPUT}.llir 
