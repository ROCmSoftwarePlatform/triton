#!/bin/bash

set -e

pushd .

git clone https://github.com/dfukalov/ROCm-Device-Libs.git
cd ROCm-Device-Libs
git apply ../cuda2gcn.patch
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=$HOME/.triton/llvm/LLVM-14.0.0-Linux
make -j

popd
cp ROCm-Device-Libs/build/amdgcn/bitcode/cuda2gcn.bc .
rm -rf ROCm-Device-Libs
