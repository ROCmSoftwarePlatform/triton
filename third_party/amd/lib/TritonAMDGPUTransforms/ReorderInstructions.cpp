#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <deque>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

static bool isLocalLoadOrDotLayoutConversion(Operation *op) {
  if (isa<ttg::LocalLoadOp>(op))
    return true;
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
    return isa<ttg::DotOperandEncodingAttr>(cvt.getType().getEncoding());
  return false;
}

// Check if the operation opInsideLoop is inside any scf::ForOp and
// opOutsideLoop is not inside the same loop.
bool isCrossLoopBoundary(mlir::Operation *opInsideLoop,
                         mlir::Operation *opOutsideLoop) {
  scf::ForOp parentForOp = opInsideLoop->getParentOfType<scf::ForOp>();
  return parentForOp && !parentForOp->isAncestor(opOutsideLoop);
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  Operation *getFirstUse(Operation *op) {
    std::vector<Operation *> users;
    for (auto user : op->getUsers()) {
      if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
        users.push_back(ancestor);
    }
    auto minOpIt = std::min_element(users.begin(), users.end(),
                                    [](mlir::Operation *a, mlir::Operation *b) {
                                      return a->isBeforeInBlock(b);
                                    });
    return minOpIt != users.end() ? *minOpIt : nullptr;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Sink shared memory loads and layout conversions into loops to decrease
    // register pressure when possible.
    DenseMap<Operation *, Operation *> opToMove;
    m.walk([&](Operation *op) {
      if (!isLocalLoadOrDotLayoutConversion(op))
        return;
      if (!op->hasOneUse())
        return;
      Operation *user = *op->getUsers().begin();
      if (user->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, user});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    opToMove.clear();

    // Adjust the placement of LDS writes and reads to immediately follow the
    // definition of their operands in case where LDS write is in the
    // loop but it's operand is not. This is a heuristic for optimizing fused
    // attention by hoisting Q tensor LDS read/write operations outside of the
    // loop, as Q is a loop invariant and can be loaded once before entering the
    // loop.
    // There are two possible patterns for this adjustment depending on
    // whether the write to LDS is performed using an optional `local_alloc`
    // argument or a `local_store` instruction.
    //
    // clang-format off
    //
    // 1) %1 = some_op ... (typically a load or an operation that scales the tensor after loading)
    //    %2 = local_alloc %1
    //    %3 = local_load %2
    //
    // 2) %1 = some_op ...
    //    %2 = local_alloc
    //    %3 = local_store %1, %2
    //    %4 = local_load %2
    //
    // clang-format on
    m.walk([&](ttg::LocalLoadOp localLoad) {
      auto localAlloc = localLoad.getSrc().getDefiningOp<ttg::LocalAllocOp>();
      if (!localAlloc)
        return;

      // Case when localAlloc has operands
      if (localAlloc->getNumOperands() == 1) {
        if (!localAlloc->hasOneUse())
          return;

        auto srcTensorOp = localAlloc->getOperand(0).getDefiningOp();
        // Check if localAlloc is in the loop but it's src tensor defining op is
        // outside of it.
        if (!srcTensorOp || !isCrossLoopBoundary(localAlloc, srcTensorOp)) {
          return;
        }

        localAlloc->moveAfter(srcTensorOp);
        localLoad->moveAfter(localAlloc);
        return;
      }

      // Case when localAlloc has no operands
      assert(localAlloc->getNumOperands() < 1);
      auto allocVal = localAlloc->getResult(0);

      // Check if the localAlloc has exactly two uses (localStore and localLoad)
      int numUses = std::distance(allocVal.use_begin(), allocVal.use_end());
      if (numUses != 2)
        return;

      // localStore comes before localLoad in block.
      Operation *localStore = getFirstUse(localAlloc);
      if (!isa<ttg::LocalStoreOp>(localStore))
        return;

      auto srcTensorOp = localStore->getOperand(0).getDefiningOp();
      // Check if localStore is in the loop but it's src tensor defining op is
      // outside of it.
      if (!srcTensorOp || !isCrossLoopBoundary(localStore, srcTensorOp)) {
        return;
      }

      localAlloc->moveAfter(srcTensorOp);
      localStore->moveAfter(localAlloc);
      localLoad->moveAfter(localStore);
    });

    // Sink conversion after the last dealloc but before the first use ancestor
    // in its block. This helps to avoid unnecessary shared memory allocation.
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto curr = mlir::Block::iterator(op);
      for (; &*curr != getFirstUse(op); curr++)
        if (isa<triton::gpu::LocalDeallocOp>(&*curr))
          op->moveAfter(&*curr);
    });

    // Move transpositions just after their definition.
    m.walk([&](triton::TransOp op) {
      if (Operation *argOp = op.getSrc().getDefiningOp())
        op->moveAfter(argOp);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
