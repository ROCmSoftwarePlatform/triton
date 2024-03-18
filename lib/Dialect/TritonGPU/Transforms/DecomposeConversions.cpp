#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUDecomposeConversionsPass
    : public TritonGPUDecomposeConversionsBase<
          TritonGPUDecomposeConversionsPass> {
public:
  TritonGPUDecomposeConversionsPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcEncoding = srcType.getEncoding();
      if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>())
        return;
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (!dstDotOp)
        return;
      if (auto srcMmaEncoding =
              srcEncoding.dyn_cast<triton::gpu::MmaEncodingAttr>()) {

        if (srcMmaEncoding.getVersionMajor() == 1 ||
            (srcMmaEncoding.getWarpsPerCTA()[1] == 1 &&
             dstDotOp.getParent() == srcMmaEncoding))
          return;
      }
#ifdef USE_ROCM
      if (auto srcMfmaEncoding =
              srcEncoding.dyn_cast<triton::gpu::MfmaEncodingAttr>()) {

        if (srcMfmaEncoding.getWarpsPerCTA()[1] == 1 &&
            srcMfmaEncoding.getIsTransposed() &&
            dstDotOp.getParent() == srcMfmaEncoding)
          return;
        auto dotOperandEncoding =
            dstDotOp.getParent().dyn_cast<triton::gpu::MfmaEncodingAttr>();
        if (dotOperandEncoding) {
          if (srcMfmaEncoding.getMDim() == 4 &&
              srcMfmaEncoding.getNDim() == 64 &&
              srcMfmaEncoding.getIsTransposed() &&
              dstDotOp.getKWidth() == 4 &&
              dstDotOp.getOpIdx() == 0 &&
              dotOperandEncoding.getMDim() == 4 &&
              dotOperandEncoding.getNDim() == 4 &&
              srcMfmaEncoding.getWarpsPerCTA()[1] == 1)
            return;
          if (srcMfmaEncoding.getMDim() == 64 &&
              srcMfmaEncoding.getNDim() == 4 &&
              !srcMfmaEncoding.getIsTransposed() &&
              dstDotOp.getKWidth() == 4 &&
              dstDotOp.getOpIdx() == 0 &&
              dotOperandEncoding.getMDim() == 4 &&
              dotOperandEncoding.getNDim() == 4 &&
              srcMfmaEncoding.getWarpsPerCTA()[0] == 1)
            return;
          if (srcMfmaEncoding.getWarpsPerCTA()[1] == 1 &&
              dstDotOp.getOpIdx() == 0 &&
              dstDotOp.getKWidth() == 4 &&
              (srcMfmaEncoding.getMDim() == 32 ||
              srcMfmaEncoding.getMDim() == 16) &&
              srcMfmaEncoding.getIsTransposed())
            return;
        }
      }
#endif
      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(),
              triton::gpu::getOrder(srcEncoding),
              triton::gpu::getCTALayout(srcEncoding),
              srcType.getElementType()));
      auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getOperand());
      auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUDecomposeConversionsPass() {
  return std::make_unique<TritonGPUDecomposeConversionsPass>();
}
