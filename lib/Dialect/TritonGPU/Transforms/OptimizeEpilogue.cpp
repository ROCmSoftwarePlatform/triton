#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;

namespace {

// convert(val) : mma -> blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> mma
// convert(mask) : blocked -> mma
// tt.store(ptr, val, mask, ...) : mma
//
// Store with mma layout directly
class BypassEpilogueSMEM : public mlir::RewritePattern {

public:
  explicit BypassEpilogueSMEM(mlir::MLIRContext *context)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    Value ptr, val, mask; 
    RankedTensorType ptrType, valType;
    triton::gpu::ConvertLayoutOp cvtOp;

    if (auto stOp = dyn_cast<triton::StoreOp>(op)){
       ptr = stOp.getPtr();
       val = stOp.getValue();
       mask = stOp.getMask();
    } else if (auto atomicRMWOp =  dyn_cast<triton::AtomicRMWOp>(op)) {
      ptr = atomicRMWOp.getPtr();
      val = atomicRMWOp.getVal();
      mask = atomicRMWOp.getMask();
    } else {
      return mlir::failure();
    }

    ptrType = ptr.getType().dyn_cast<RankedTensorType>();
    valType = val.getType().dyn_cast<RankedTensorType>();
    if (!ptrType || !valType ||
        !ptrType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>() ||
        !valType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>())
      return mlir::failure();

    cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(val.getDefiningOp());
    if (!cvtOp)
      return mlir::failure();

    auto encoding =
        cvtOp.getSrc().getType().cast<RankedTensorType>().getEncoding();

#ifdef USE_ROCM
    if (!encoding.isa<triton::gpu::MfmaEncodingAttr>())
      return mlir::failure();
#else
    if (!encoding.isa<triton::gpu::MmaEncodingAttr>())
      return mlir::failure();
#endif

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    auto newEncoding =
        cvtOp.getOperand().getType().cast<RankedTensorType>().getEncoding();

    auto newVal = cvtOp.getOperand();

    auto newPtrType = RankedTensorType::get(
        ptrType.getShape(), ptrType.getElementType(), newEncoding);
    Value newPtr = rewriter.create<triton::gpu::ConvertLayoutOp>(
        ptr.getLoc(), newPtrType, ptr);

    Value newMask = mask;
    if (mask) {
      auto maskType = mask.getType().dyn_cast<RankedTensorType>();
      auto newMaskType = RankedTensorType::get(
          maskType.getShape(), maskType.getElementType(), newEncoding);
      newMask = rewriter.create<triton::gpu::ConvertLayoutOp>(
          mask.getLoc(), newMaskType, mask);
    }

    if (auto stOp = dyn_cast<triton::StoreOp>(op)) {
       rewriter.replaceOpWithNewOp<triton::StoreOp>(
           stOp, newPtr, newVal, newMask, stOp.getCache(), stOp.getEvict());
    } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
       auto result = atomicRMWOp.getResult();
       auto resultType = result.getType().dyn_cast<RankedTensorType>();
       auto newResultType = RankedTensorType::get(
         resultType.getShape(), resultType.getElementType(), newEncoding);
       rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
           atomicRMWOp, newResultType, atomicRMWOp.getAtomicRmwOpAttr(), newPtr, newVal, newMask, atomicRMWOp.getSemAttr(), atomicRMWOp.getScopeAttr());
    }

    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeEpiloguePass
    : public TritonGPUOptimizeEpilogueBase<TritonGPUOptimizeEpiloguePass> {

public:
  TritonGPUOptimizeEpiloguePass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassEpilogueSMEM>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeEpiloguePass() {
  return std::make_unique<TritonGPUOptimizeEpiloguePass>();
}
