#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
// #include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
// #include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
// #include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
// #include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
// #include "triton/Conversion/TritonGPUToLLVM/Utility.h"
// #include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/TritonGPU/IR/Dialect.h"
// #include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

namespace {
struct TritonAMDGPUMembarAnalysis
    : public mlir::TritonAMDGPUMembarAnalysisBase<TritonAMDGPUMembarAnalysis> {

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Allocate shared memory and set barrier
    mlir::ModuleAllocation allocation(mod);
    mlir::ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();
  }
};
} // namespace

namespace mlir {
std::unique_ptr<Pass> createTritonAMDGPUMembarAnalysisPass() {
  return std::make_unique<TritonAMDGPUMembarAnalysis>();
}
} // namespace mlir
