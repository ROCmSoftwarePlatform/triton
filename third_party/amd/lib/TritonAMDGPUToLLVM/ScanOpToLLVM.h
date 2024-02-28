#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_SCAN_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_SCAN_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD{
void populateScanOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);
}

#endif
