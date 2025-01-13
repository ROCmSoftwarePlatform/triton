#!/usr/bin/bash

INPUT="5c_loop_begin"
OUTPUT="5d_loop_begin"
python renumber_llir.py /home/dtanner/repos/rocm_triton/golden_ir/versions/${INPUT}.llir
mv /home/dtanner/repos/rocm_triton/golden_ir/versions/${INPUT}.llir.renumbered \
  /home/dtanner/repos/rocm_triton/golden_ir/versions/${OUTPUT}.llir 
