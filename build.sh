#!/usr/bin/bash
llvm_install_path=/home/dtanner/repos/triton/usr/llvm

# build and install Triton
export REL_WITH_DEB_INFO=true
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_BUILD_WITH_CCACHE=true


LLVM_INCLUDE_DIRS=${llvm_install_path}/include \
  LLVM_LIBRARY_DIR=${llvm_install_path}/lib \
  LLVM_SYSPATH=${llvm_install_path} \
  pip install -e python -v

########################################
# Build with default llvm
# pip install -e python


