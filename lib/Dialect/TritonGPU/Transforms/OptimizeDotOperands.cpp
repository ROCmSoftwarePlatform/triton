#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SharedEncodingAttr;
using triton::gpu::SliceEncodingAttr;

// convert(trans(convert(arg)))
// x = convert_layout arg: #distributed -> #shared_x
// y = trans x: #shared_x -> #shared_y
// z = convert_layout y: #shared_y -> #dot_operand
class ConvertTransConvert : public mlir::RewritePattern {

public:
  ConvertTransConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto tmpOp =
        dyn_cast_or_null<triton::TransOp>(dstOp.getSrc().getDefiningOp());
    if (!tmpOp)
      return mlir::failure();
    auto srcOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        tmpOp.getSrc().getDefiningOp());
    if (!srcOp)
      return mlir::failure();
    auto arg = srcOp.getSrc();
    auto X = tmpOp.getSrc();
    // types
    auto argType = arg.getType().cast<RankedTensorType>();
    auto XType = X.getType().cast<RankedTensorType>();
    auto ZType = dstOp.getResult().getType().cast<RankedTensorType>();
    // encodings
    auto argEncoding = argType.getEncoding();
    auto XEncoding =
        XType.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto ZEncoding =
        ZType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    if (!ZEncoding)
      return mlir::failure();
    // new X encoding
    auto newXOrder = triton::gpu::getOrder(argEncoding);
    auto newXEncoding = triton::gpu::SharedEncodingAttr::get(
        getContext(), ZEncoding, XType.getShape(), newXOrder,
        XType.getElementType());
    auto newXType = RankedTensorType::get(XType.getShape(),
                                          XType.getElementType(), newXEncoding);
    if (XEncoding == newXEncoding)
      return mlir::failure();

    auto newX = rewriter.create<triton::gpu::ConvertLayoutOp>(srcOp.getLoc(),
                                                              newXType, arg);
    auto newY = rewriter.create<triton::TransOp>(tmpOp.getLoc(), newX);
    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(dstOp, ZType,
                                                              newY);
    return mlir::success();
  }
};

// convert(layout_preserving_op(x), dot_operand)
// -> layout_preserving_op(convert(x, dot_operand))
class MoveOpAfterLayoutConversion : public mlir::RewritePattern {
public:
  MoveOpAfterLayoutConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    // conversion should be dependent on a load
    // and all operations between the load and the conversion
    // should be layout preserving
    SetVector<Operation *> slice;
    getBackwardSlice(op, &slice);
    int loadIdx = -1;
    bool checkOp = false;
    for (int i = 0; i < slice.size(); i++) {
      Operation *currOp = *(slice.begin() + i);
      if (currOp->getParentRegion() != op->getParentRegion())
        continue;
      if (isa<triton::LoadOp>(currOp))
        checkOp = true;
      else if (checkOp) {
        if (!isa<triton::FpToFpOp, triton::BitcastOp>(currOp) &&
            currOp->getDialect()->getTypeID() !=
                mlir::TypeID::get<arith::ArithDialect>())
          return mlir::failure();
      }
    }
    if (!checkOp)
      return mlir::failure();

    auto cvtTy = cvt.getType().cast<RankedTensorType>();
    auto cvtArgOp = cvt.getSrc().getDefiningOp();
    if (!cvtArgOp || cvtArgOp->getNumOperands() == 0)
      return mlir::failure();
    // only consider custom conversions or arith ops
    if (!isa<triton::FpToFpOp, triton::BitcastOp>(cvtArgOp) &&
        cvtArgOp->getDialect()->getTypeID() !=
            mlir::TypeID::get<arith::ArithDialect>())
      return mlir::failure();
    // only considers conversions to dot operand
    if (!cvtTy.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
      return mlir::failure();
    auto argTy = cvtArgOp->getOperand(0).getType().cast<RankedTensorType>();
    auto retTy = cvtArgOp->getResult(0).getType().cast<RankedTensorType>();
    if (!argTy || !retTy)
      return mlir::failure();
    Type newRetTy = RankedTensorType::get(
        retTy.getShape(), retTy.getElementType(), cvtTy.getEncoding());
    Type newCvtTy = RankedTensorType::get(
        retTy.getShape(), argTy.getElementType(), cvtTy.getEncoding());
    int numArgs = cvtArgOp->getNumOperands();
    SmallVector<triton::gpu::ConvertLayoutOp> newCvts(numArgs);
    for (int i = 0; i < numArgs; i++)
      newCvts[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
          cvt.getLoc(), newCvtTy, cvtArgOp->getOperand(i));
    auto newRet = rewriter.clone(*cvtArgOp);
    for (int i = 0; i < numArgs; i++)
      newRet->setOperand(i, newCvts[i]);
    newRet->getResult(0).setType(newRetTy);
    rewriter.replaceOp(op, newRet->getResults());
    return mlir::success();
  }
};

#ifdef USE_ROCM
// Following pattern searches for mfma DotOp with ConvertOps as arguments
// chains of convert layouts with operands which
class SimplifyMFMADotOpConversions : public mlir::RewritePattern {
public:
  SimplifyMFMADotOpConversions(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto candidate = cast<triton::gpu::ConvertLayoutOp>(op);
    auto dstType = cast<RankedTensorType>(candidate.getResult().getType());
    auto dotOpEnc = dyn_cast_or_null<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
    if (!dotOpEnc)
      return mlir::failure();
    auto mfmaEnc = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(dotOpEnc.getParent());
    if (!mfmaEnc)
      return mlir::failure();
    auto loc = candidate.getLoc();

    auto [earlySrc, dist, needTranspose] = earliestMFMACompatibleValue(candidate.getResult());

    if (dist > 1) {
      rewriter.setInsertionPoint(candidate);
      auto newCvt = adjustLayout(rewriter, loc, earlySrc, dstType, needTranspose);
      rewriter.replaceOp(candidate, newCvt);
      return mlir::success();
    }

    return mlir::failure();
  }

private:

  // Insert no-op operations to adjust compatible layout
  static Value adjustLayout(mlir::PatternRewriter &rewriter, mlir::Location loc, Value src, Type targetType, bool transpose) {
    assert(isa<RankedTensorType>(targetType));
    auto srcType = cast<RankedTensorType>(src.getType());
    if (srcType == targetType)
      return src;
    if (transpose) {
      auto srcEncoding = cast<triton::gpu::MfmaEncodingAttr>(srcType.getEncoding());
      auto ctx = srcEncoding.getContext();
      auto nonKDim = srcEncoding.getNonKDim();
      auto warps = srcEncoding.getWarpsPerCTA();
      auto trans = srcEncoding.getIsTransposed();
      auto tSrcEncoding = triton::gpu::MfmaEncodingAttr::get(ctx, nonKDim, warps, !trans);

      auto shape = srcType.getShape();
      auto dtype = srcType.getElementType();
      auto tSrcType = RankedTensorType::get(shape, dtype, tSrcEncoding);
      src = rewriter.create<triton::ViewOp>(loc, tSrcType, src);
    }
    auto convert = rewriter.create<triton::gpu::ConvertLayoutOp>(loc, targetType, src);
    return convert.getResult();
  }

  // trace chain of layout conversions and find earliest compatible value,
  // which can be used as a dot operand
  // returns tuple: found value, distance to this value (0 if returned value equal to input val)
  // and flag which means we need to transpose value.
  static std::tuple<mlir::Value, int, bool> earliestMFMACompatibleValue(mlir::Value val) {
    auto dstType = cast<RankedTensorType>(val.getType());
    mlir::Operation *op;
    bool isTransposed = false;

    // Components of best found result
    Value earliestDef = val;
    int earliestDist = 0;
    bool earliestTransposed = false;

    for (int dist = 0; ; ++dist) {
      auto valType = cast<RankedTensorType>(val.getType());
      auto mfmaLayout = dyn_cast_or_null<triton::gpu::MfmaEncodingAttr>(valType.getEncoding());
      if (mfmaLayout && isMfmaToDotShortcut(valType, dstType, isTransposed)) {
        earliestDef = val;
        earliestDist = dist;
        earliestTransposed = isTransposed;
      }
      op = val.getDefiningOp();
      auto convertOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(op);
      auto transposeOp = dyn_cast_or_null<triton::TransOp>(op);
      if (!convertOp && !transposeOp)
        return {earliestDef, earliestDist, earliestTransposed};
      if (transposeOp)
        isTransposed = !isTransposed;
      val = op->getOperand(0);
    }
  }
};

#endif // USE_ROCM

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  TritonGPUOptimizeDotOperandsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::PassManager pm(m.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    auto ret = pm.run(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertTransConvert>(context);
    patterns.add<MoveOpAfterLayoutConversion>(context);
#ifdef USE_ROCM
    patterns.add<SimplifyMFMADotOpConversions>(context);
#endif
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
    if (fixupLoops(m).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeDotOperandsPass() {
  return std::make_unique<TritonGPUOptimizeDotOperandsPass>();
}
