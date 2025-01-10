#!/usr/bin/bash

INPUT="2c"
OUTPUT="2d"
python renumber_llir.py /home/dtanner/repos/rocm_triton/golden_ir/versions/${INPUT}_global_loads.llir
mv /home/dtanner/repos/rocm_triton/golden_ir/versions/${INPUT}_global_loads.llir.renumbered \
  /home/dtanner/repos/rocm_triton/golden_ir/versions/${OUTPUT}_global_loads.llir 
