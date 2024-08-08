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
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
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
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = triton::gpu;

// // convert(val) : mma -> blocked
// // tt.store(ptr, val, mask, ...) : blocked
// // ==>
// // convert(ptr) : blocked -> mma
// // convert(mask) : blocked -> mma
// // tt.store(ptr, val, mask, ...) : mma
// //
// // Store with mma layout directly

// static Type getNewType(Type type, Attribute encoding) {
//   RankedTensorType tensorType = typecast<RankedTensorType>();
//   return RankedTensorType::get(tensorType.getShape(),
//                                tensorType.getElementType(), encoding);
// }

// void convertLayout(Attribute encoding, Operation *op) {
//   OpBuilder builder(op);
//   // Convert operands
//   // For load/store with tensor pointers, we don't have to change the
//   // operands' type, we do this by changing the outputs' type of
//   // `make_tensor_ptr`
//   SmallVector<Value, 4> newArgs;
//   for (auto operand : op->getOperands()) {
//     auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
//     if (tensorType &&
//         !tensorType.getEncoding().isa<ttg::SharedEncodingAttr>()) {
//       Type newType = getNewType(tensorType, encoding);
//       newArgs.push_back(
//           builder.create<ttg::ConvertLayoutOp>(op->getLoc(), newType, operand));
//     } else {
//       newArgs.push_back(operand);
//     }
//   }

//   // Convert output types
//   SmallVector<Type, 4> newTypes;
//   for (auto t : op->getResultTypes()) {
//     bool isAsync = isa<ttg::InsertSliceAsyncOp>(op);
//     newTypes.push_back(isAsync ? t : getNewType(t, encoding));
//   }

//   // Construct new op with the new encoding
//   Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
//                                     newArgs, newTypes, op->getAttrs());

//   // Cast the results back to the original layout
//   for (size_t i = 0; i < op->getNumResults(); i++) {
//     Value newResult = newOp->getResult(i);
//     if (newTypes[i] != op->getResultTypes()[i]) {
//       newResult = builder.create<ttg::ConvertLayoutOp>(
//           op->getLoc(), op->getResult(i).getType(), newResult);
//     }
//     op->getResult(i).replaceAllUsesWith(newResult);
//   }
//   op->erase();
// }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = cast<RankedTensorType>(type);
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  void convertLayout(Attribute encoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType &&
          !isa<triton::gpu::SharedEncodingAttr>(tensorType.getEncoding())) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

triton::LoadOp getLoadInst(Operation *op, ModuleOp &mod) {
  SmallVector<triton::LoadOp> loadOpsVec;

  mod.walk([&](triton::LoadOp loadOp) {
    SetVector<Operation *> forwardSlices;
    getForwardSlice((Operation *)loadOp, &forwardSlices);
    if (std::find(forwardSlices.begin(), forwardSlices.end(), op) !=
        forwardSlices.end()) {
      loadOpsVec.push_back(loadOp);
    }
  });

  // Currently, we expect the dot operand to depend only on one tensor
  // from global memory (applicable for dot ops that don't depend on other dot
  // ops). This condition can be lifted if necessary.
  // assert(loadOpsVec.size() == 1);
  llvm::outs() << loadOpsVec.size()  << "\n";
  return loadOpsVec[2];
}

// class BypassLDSForDotLayout : public mlir::RewritePattern {

// public:
//   explicit BypassLDSForDotLayout(mlir::MLIRContext *context)
//       : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(), 1, context) {}
//   mlir::LogicalResult
//   matchAndRewrite(mlir::Operation *op,
//                   mlir::PatternRewriter &rewriter) const override {

//     auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
//     auto mod = op->getParentOfType<ModuleOp>();

//     static int counter = 0;
//     if(counter > 0){
//       return mlir::failure();
//     }

//     if (!cvtOp)
//       return mlir::failure();

//     auto srcType = cast<RankedTensorType>(cvtOp.getOperand().getType());
//     auto dstType = cast<RankedTensorType>(cvtOp.getType());
//     auto srcBlocked =
//         dyn_cast<triton::gpu::BlockedEncodingAttr>(srcType.getEncoding());
//     auto dstDotOp =
//         dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());



//     if (!(srcBlocked && dstDotOp)) {
//       return mlir::failure();
//     }


//     if(dstDotOp.getOpIdx() != 1){
//       return mlir::failure();
//     }

//     if(srcBlocked.getOrder()[1] == 0){
//       return mlir::failure();
//     }

    
    

//     SmallVector<unsigned> newWarpsPerCTA(2, 4);
//     SmallVector<unsigned> newSizePerThread(2, 4);
//     SmallVector<unsigned> newThreadsPerWarp(2, 4);
//     SmallVector<unsigned> newOrder(2, 4);
//     newOrder[0] = 1;
//     newOrder[1] = 0;
//     newThreadsPerWarp[0] = 4;
//     newThreadsPerWarp[1] = 16;
//     newSizePerThread[0] = 8;
//     newSizePerThread[1] = 1;
//     newWarpsPerCTA[0] = 1;
//     newWarpsPerCTA[1] = 8;
//     auto newBlockedEncoding = triton::gpu::BlockedEncodingAttr::get(
//         mod.getContext(), newSizePerThread, newThreadsPerWarp, newWarpsPerCTA,
//         newOrder, srcBlocked.getCTALayout());

//     auto loadInst = getLoadInst(cvtOp, mod);
//     convertLayout(newBlockedEncoding, (Operation *)loadInst);
//     if (failed(mlir::verify(mod))) {
//       assert(false);
//     }
//     counter+= 1;
//     return mlir::success();
//   }
// };





class BypassLDSForDotLayout : public mlir::RewritePattern {

public:
  explicit BypassLDSForDotLayout(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::LoadOp::getOperationName(), 1, context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    auto cvtOp = dyn_cast<triton::LoadOp>(op);


  if (op->getResult(0).hasOneUse()) {
    OpBuilder builder(op);

    Operation *userOp = *op->getResult(0).user_begin();

    // Check if the user is of type LocalAllocOp
    if (isa<triton::gpu::LocalAllocOp>(userOp)) {
      Value allocResult = userOp->getResult(0);

      llvm::outs() << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
      // Check if the result of LocalAllocOp has exactly one user
        Operation *loadUserOp = *allocResult.user_begin();

        auto userIter = allocResult.getUsers().begin();
        std::advance(userIter, 1); // Move to the second user
        Operation *secondUserOp = *userIter;



        // Check if this user is of type LocalLoadOp
        if (isa<triton::gpu::LocalLoadOp>(loadUserOp)) {
          // Get the result type of the LocalLoadOp
          auto loadResultType = dyn_cast<RankedTensorType>(loadUserOp->getResult(0).getType());

          // Dump the type
          loadResultType.dump();

          auto dotOperandLayout =
              dyn_cast<triton::gpu::DotOperandEncodingAttr>(loadResultType.getEncoding());

      llvm::outs() << "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n";

          if(!dotOperandLayout){
            return mlir::failure();
          }
      llvm::outs() << "ccccccccccccccccccccccccccccccccccccccccc\n";

          if(dotOperandLayout.getOpIdx() == 0){
            return mlir::failure();
          }
      llvm::outs() << "ddddddddddddddddddddddddddddddddddddddddd\n";

          auto newCvt = builder.create<triton::gpu::ConvertLayoutOp>(
              op->getLoc(), loadResultType, op->getResult(0));

          rewriter.replaceAllUsesWith(userOp->getResult(0), newCvt.getResult());

      llvm::outs() << "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n";

        if (secondUserOp->getNumOperands() == 2) {
          Value firstOperand = secondUserOp->getOperand(0);
          Operation *firstOperandOp = firstOperand.getDefiningOp();

          auto newCvt2 = builder.create<triton::gpu::ConvertLayoutOp>(op->getLoc(), loadResultType, firstOperandOp->getResult(0));
      llvm::outs() << "fffffffffffffffffffffffffffffffffffffff\n";
          rewriter.replaceAllUsesWith(secondUserOp, newCvt2.getResult());

        }

          return mlir::success();
      }
    }
  }
    // Operation *localAlloc = cvtOp->getUsers(0);


    auto mod = op->getParentOfType<ModuleOp>();



    return mlir::failure();
  }
};



class TritonAMDGPUBypassLDSForDotLayoutPass
    : public TritonAMDGPUBypassLDSForDotLayoutBase<TritonAMDGPUBypassLDSForDotLayoutPass> {

public:
  TritonAMDGPUBypassLDSForDotLayoutPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    // int *a = nullptr;
    // *a = 4;
    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassLDSForDotLayout>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUBypassLDSForDotLayout() {
  return std::make_unique<TritonAMDGPUBypassLDSForDotLayoutPass>();
}
