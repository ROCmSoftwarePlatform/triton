#include "TritonAMDGPUToLLVM/Passes.h"

#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_SCHEDGROUPBARRIERSINSERTION
#define GEN_PASS_DEF_SCHEDGROUPBARRIERSLOWERING
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
enum class InstructionMaskEnum : int64_t {
  NONE = 0x0000000,
  VALU = 0x00000002,
  SALU = 0x00000004,
  MFMA = 0x00000008,
  ALL_VMEM = 0x00000010,
  VMEM_READ = 0x00000020,
  VMEM_WRITE = 0x00000040,
  ALL_DS = 0x00000080,
  DS_READ = 0x00000100,
  DS_WRITE = 0x00000200
};

const bool modifyScheduling{false};
// const bool modifyScheduling{true};

void buildSchedGroupBarrier(PatternRewriter &builder,
                            InstructionMaskEnum maskValue, int sizeValue,
                            int groupIdValue) {
  MLIRContext *ctx = builder.getContext();
  Location loc = builder.getUnknownLoc();
  auto intrinsicName = StringAttr::get(ctx, "llvm.amdgcn.sched.group.barrier");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Type i32 = builder.getI32Type();
  auto mask = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(i32, static_cast<int64_t>(maskValue)));
  auto size = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(i32, sizeValue));
  auto groupId = builder.create<LLVM::ConstantOp>(
      loc, builder.getIntegerAttr(i32, groupIdValue));
  builder.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                        ValueRange{mask, size, groupId},
                                        defaultFlags);
}

Operation *generatedSchedBarrier(PatternRewriter &rewriter,
                                 InstructionMaskEnum maskValue) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = rewriter.getUnknownLoc();
  auto intrinsicName = StringAttr::get(ctx, "llvm.amdgcn.sched.barrier");
  LLVM::FastmathFlagsAttr defaultFlags{};
  Type i32 = rewriter.getI32Type();
  auto mask = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32, static_cast<int64_t>(maskValue)));
  return rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange{}, intrinsicName,
                                                ValueRange{mask}, defaultFlags);
}

struct SchedGroupBarriersRewriter
    : public OpRewritePattern<triton::gpu::GroupSched> {
  using OpRewritePattern<triton::gpu::GroupSched>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::gpu::GroupSched schedBarrier,
                                PatternRewriter &rewriter) const override {

    Block *block = schedBarrier->getBlock();

    size_t numGlbLoads = 0;
    block->walk([&numGlbLoads](LLVM::CallOp callOp) {
      StringRef calleeName = callOp.getCallee().value();
      if (calleeName.contains("__predicated_load_vector"))
        ++numGlbLoads;
    });

    size_t numDsReads = 0;
    block->walk([&numDsReads](LLVM::LoadOp op) {
      auto operandType = op.getOperand().getType();
      if (auto ptr = llvm::dyn_cast<LLVM::LLVMPointerType>(operandType))
        if (ptr.getAddressSpace() == 3)
          ++numDsReads;
    });

    size_t numDsWrites = 0;
    block->walk([&numDsWrites](LLVM::StoreOp op) {
      auto operandType = op.getOperand(1).getType();
      if (auto ptr = llvm::dyn_cast<LLVM::LLVMPointerType>(operandType))
        if (ptr.getAddressSpace() == 3)
          ++numDsWrites;
    });

    size_t numMfmas = 0;
    block->walk([&numMfmas](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      if (opName.contains("mfma"))
        ++numMfmas;
    });

    llvm::dbgs() << "group scheduling info: ["
                 << "numGlbLoads: " << numGlbLoads << ", "
                 << "numDsReads: " << numDsReads << ", "
                 << "numDsWrites: " << numDsWrites << ", "
                 << "numMfmas: " << numMfmas << "]\n";

    size_t barrierCounter{0};
    block->walk([&barrierCounter, &rewriter](ROCDL::BarrierOp op) {
      if (barrierCounter == 1) {
        rewriter.setInsertionPointAfter(op);
        return WalkResult::interrupt();
      }
      ++barrierCounter;
      return WalkResult::advance();
    });

    // rewriter.setInsertionPointToStart(block);
    auto op = generatedSchedBarrier(rewriter, InstructionMaskEnum::NONE);

    rewriter.setInsertionPointAfter(schedBarrier);
    const size_t numIssues = numGlbLoads;
    for (size_t i = 0; i < numIssues; ++i) {
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_READ,
                             numDsReads / numIssues, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::DS_WRITE,
                             numDsWrites / numIssues, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA, 1, 0);
      buildSchedGroupBarrier(rewriter, InstructionMaskEnum::MFMA,
                             (numMfmas / numIssues) - 3, 0);
    }
    op = generatedSchedBarrier(rewriter, InstructionMaskEnum::NONE);
    rewriter.eraseOp(schedBarrier);
    return mlir::success();
  }
};

struct SchedGroupBarriersLowering
    : public triton::impl::SchedGroupBarriersLoweringBase<
          SchedGroupBarriersLowering> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    if (!modifyScheduling)
      return;

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<triton::gpu::GroupSched>();

    RewritePatternSet patterns(ctx);
    patterns.add<SchedGroupBarriersRewriter>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct SchedGroupBarriersInsertion
    : public triton::impl::SchedGroupBarriersInsertionBase<
          SchedGroupBarriersInsertion> {

  void insertPlaceholder(mlir::OpBuilder &builder, triton::DotOp dot) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(dot);
    Location loc = builder.getUnknownLoc();
    builder.create<triton::gpu::GroupSched>(loc);
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    if (!modifyScheduling)
      return;

    mlir::OpBuilder builder(ctx);
    mod.walk(
        [this, &builder](triton::DotOp op) { insertPlaceholder(builder, op); });
  }
};
} // namespace

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>>
createSchedGroupBarriersLoweringPass() {
  return std::make_unique<SchedGroupBarriersLowering>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createSchedGroupBarriersInsertionPass() {
  return std::make_unique<SchedGroupBarriersInsertion>();
}
} // namespace triton
} // namespace mlir
