#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToRockPass.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "TypeConverter.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "triton/Dialect/Rock/IR/Rock.h"
#include "triton/Dialect/Rock/IR/TransformMapBuilder.h"
#include "triton/Dialect/Rock/Passes.h"
#include "triton/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "triton/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "triton/Dialect/Rock/utility/builderUtils.h"
#include "triton/Dialect/Rock/utility/loweringUtils.h"
#include "triton/Dialect/Rock/utility/math.h"
#include "triton/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "dot-to-blockwise-gemm"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::arith;
using namespace mlir::rock;

#define printVec(vec)                                                          \
  llvm::outs() << #vec ": ";                                                   \
  for (int i = 0; i < vec.size() - 1; i++) {                                   \
    llvm::outs() << vec[i] << " ";                                             \
  }                                                                            \
  llvm::outs() << vec[vec.size() - 1] << "\n";

namespace mlir {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOROCK
#include "triton/Conversion/Passes.h.inc"
#include "triton/Dialect/Rock/Passes.h.inc"
} // namespace mlir

static bool isConvertLDSToDotOp(Operation *op) {
  bool result = false;
  if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    auto srcTy = convertOp.getSrc().getType().cast<RankedTensorType>();
    auto dstTy = convertOp.getResult().getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (srcLayout.isa<triton::gpu::LDSEncodingAttr>() &&
        dstLayout.isa<triton::gpu::DotOperandEncodingAttr>())
      result = true;
  }
  return result;
}

static Type obtainAccumulatorType(OpBuilder &b, Type &elementType,
                                  Type &destType) {
  // Determine the type used on VGPR to act as accumulator.
  // f32: f32.
  // f16, bf16: f32 to prevent overflow from happening.
  // i16 : i16.
  // i8: i32, since we have an i32 output
  Type accumulatorType = destType;
  if (elementType.isF16() || elementType.isBF16()) {
    accumulatorType = b.getF32Type();
  } else if (elementType.isInteger(8)) {
    accumulatorType = b.getI32Type();
  }
  return accumulatorType;
}

namespace {
struct DotOpRewritePattern : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value matA = op.getA(), matB = op.getB();
    Operation *matADefiningOp = matA.getDefiningOp();
    Operation *matBDefiningOp = matB.getDefiningOp();
    bool dotInLoop = false;

    // Check if operands come from LDS
    // TODO (stage 3): integrate xdlops_gemm_v2 if operands come from
    // registers directly
    if ((matADefiningOp && !isConvertLDSToDotOp(matADefiningOp)) ||
        (matBDefiningOp && !isConvertLDSToDotOp(matBDefiningOp))) {
      LLVM_DEBUG(llvm::dbgs() << "operand a or b does not come from LDS\n");
      return failure();
    }

    if (!matADefiningOp || !matBDefiningOp) {
      llvm::outs() << "matA does not have a defining op\n";
      dotInLoop = true;
      return failure();
    }

    auto aShape = matA.getType().cast<RankedTensorType>().getShape();
    auto bShape = matB.getType().cast<RankedTensorType>().getShape();
    auto cShape = op.getC().getType().cast<RankedTensorType>().getShape();
    auto elementType = matA.getType().cast<RankedTensorType>().getElementType();
    uint32_t mPerBlock = aShape[0];
    uint32_t kPerBlock = aShape[1];
    uint32_t nPerBlock = bShape[1];

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    // The GPU kernel is launched with blockSize = 32*numWarps
    // TODO: make it less confusing when working on AMDGPUs
    // TODO: enable warpSize = 64
    uint32_t numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    uint32_t blockSize = 64 * numWarps;

    constexpr int64_t waveSize = 64;
    uint32_t numWaves = blockSize / waveSize;

    // Tuning parameters
    // TODO: make kpack come from a single source of truth. E.g. tuning
    // parameter when launching the kernel
    int64_t kpack = 4;
    int64_t kpacksPerBlock = kPerBlock / kpack;
    // TODO: in rocMLIR, blockSize is derived from [m|n]PerBlock, [m|n]PerWave,
    // and waveSize. In triton, blockSize and [m|n]PerBlock are given.
    // Therefore, mPerWave and nPerWave need to be derived.
    // TODO: this is a duplicate from BlockedToMFMA in the
    // tritongpu-accelerate-matmul pass. We should unify them in the future.
    uint32_t mPerWave = std::min<uint32_t>(32, mPerBlock);
    uint32_t nPerWave = mPerBlock * nPerBlock / numWarps / mPerWave;
    if (mPerBlock * nPerBlock / (mPerWave * nPerWave) != numWaves)
      return emitError(loc) << "Need to pick another [m|n]PerWave!\n";
    // When numWarps are too small, we need larger mPerWave and nPerWave
    if (nPerWave > nPerBlock) {
      nPerWave = nPerBlock;
      mPerWave = mPerBlock * nPerBlock / numWarps / nPerWave;
    }
    int64_t nWaves = nPerBlock / nPerWave;

    LLVM_DEBUG(llvm::dbgs() << "mPerBlock:      " << mPerBlock << "\n"
                            << "kPerBlock:      " << kPerBlock << "\n"
                            << "nPerBlock:      " << nPerBlock << "\n"
                            << "blockSize:      " << blockSize << "\n"
                            << "kpack:          " << kpack << "\n"
                            << "kpacksPerBlock: " << kpacksPerBlock << "\n"
                            << "mPerWave:       " << mPerWave << "\n"
                            << "nPerWave:       " << nPerWave << "\n"
                            << "nWaves:         " << nWaves << "\n"
                            << "numWarps:       " << numWarps << "\n"
                            << "numWaves:       " << numWaves << "\n");

    // Prepare LDS buffer
    // TODO: Check the size of LDS buffer
    int64_t ldsBlockASize = mPerBlock * kPerBlock;
    int64_t ldsBlockBSize = nPerBlock * kPerBlock;
    int64_t ldsBlockSize = ldsBlockASize + ldsBlockBSize;
    // if (ldsBlockSize * sizeof(float) > 64 *1024)
    //     return failure();
    auto workgroupMemoryAddressSpace =
        rewriter.getAttr<mlir::gpu::AddressSpaceAttr>(
            mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
    // Convert LDS buffer for tile A
    auto aConvertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(matADefiningOp);
    auto ldsAMemRefType = MemRefType::get(
        {ldsBlockASize}, elementType, AffineMap{}, workgroupMemoryAddressSpace);
    auto ldsBufferA = rewriter.create<triton::gpu::TensorToMemRefOp>(
        loc, ldsAMemRefType, aConvertOp.getSrc());
    // Convert LDS buffer for tile B
    auto bConvertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(matBDefiningOp);
    auto ldsBMemRefType = MemRefType::get(
        {ldsBlockBSize}, elementType, AffineMap{}, workgroupMemoryAddressSpace);
    auto ldsBufferB = rewriter.create<triton::gpu::TensorToMemRefOp>(
        loc, ldsBMemRefType, bConvertOp.getSrc());

    auto waveSizeConstantOp = rewriter.create<ConstantIndexOp>(loc, waveSize);
    auto mPerWaveConstantOp = rewriter.create<ConstantIndexOp>(loc, mPerWave);
    auto nPerWaveConstantOp = rewriter.create<ConstantIndexOp>(loc, nPerWave);
    auto nWavesConstantOp = rewriter.create<ConstantIndexOp>(loc, nWaves);

    // Get current workitem ID.
    auto tid = rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());

    // Mfma instruction group selection.
    auto maybeMfmaInsnGroup =
        MfmaInsnGroup::select(elementType, mPerWave, nPerWave);
    if (failed(maybeMfmaInsnGroup)) {
      return emitError(loc) << "Failed to select xdlops instruction group.\n";
    }
    MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
    if (!mfmaGroup.isCoherentWithK(kpack, kPerBlock)) {
      return emitError(loc)
             << "Mfma instruction group selection is not compatible with k.\n";
    }
    // TODO: Check if we really need these parameters
    int64_t mRepeats = mfmaGroup.getMRepeats(mPerWave);
    int64_t nRepeats = mfmaGroup.getNRepeats(nPerWave);
    auto imms = mfmaGroup.getImms();

    int64_t nResultVectors = imms.size() * mRepeats * nRepeats;

    VectorType vectorType = mfmaGroup.getRetType();
    MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
    int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
    int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;

    // Logic to setup blockwise_gemm_v2 parameters.
    //
    // Original C++ logic:
    // index_t mMyWaveOffsetA;
    // index_t mMyWaveOffsetB;
    // const index_t waveId   = tid / waveSize;
    // const index_t waveId_m = waveId / nWaves;
    // const index_t waveId_n = waveId % nWaves;
    // mMyWaveOffsetA = waveId_m * mPerWave;
    // mMyWaveOffsetB = waveId_n * nPerWave;
    auto waveId = rewriter.create<DivUIOp>(loc, tid, waveSizeConstantOp);
    auto waveId_m = rewriter.create<DivUIOp>(loc, waveId, nWavesConstantOp);
    auto waveId_n = rewriter.create<RemUIOp>(loc, waveId, nWavesConstantOp);
    Value mMyWaveOffsetA, mMyWaveOffsetB;
    mMyWaveOffsetA = rewriter.create<MulIOp>(loc, waveId_m, mPerWaveConstantOp);
    mMyWaveOffsetB = rewriter.create<MulIOp>(loc, waveId_n, nPerWaveConstantOp);

    // Logic to setup buffers A and B for blockwise_gemm_v2.
    bool isKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);
    int64_t arrayASize =
        (!isKReduction) ? (kpacksPerBlock * mRepeats)
                        : (kpacksPerBlock / inputSpansPerMfmaIn * mRepeats);
    int64_t arrayBSize =
        (!isKReduction) ? (kpacksPerBlock * nRepeats)
                        : (kpacksPerBlock / inputSpansPerMfmaIn * nRepeats);
    Type arrayAType, arrayBType;
    auto privateMemoryAddressSpace =
        rewriter.getAttr<mlir::gpu::AddressSpaceAttr>(
            mlir::gpu::GPUDialect::getPrivateAddressSpace());
    if (kpack > 1) {
      arrayAType =
          MemRefType::get({arrayASize}, VectorType::get({kpack}, elementType),
                          AffineMap{}, privateMemoryAddressSpace);
      arrayBType =
          MemRefType::get({arrayBSize}, VectorType::get({kpack}, elementType),
                          AffineMap{}, privateMemoryAddressSpace);
    } else {
      arrayAType = MemRefType::get({arrayASize}, elementType, AffineMap{},
                                   privateMemoryAddressSpace);
      arrayBType = MemRefType::get({arrayBSize}, elementType, AffineMap{},
                                   privateMemoryAddressSpace);
    }
    auto arrayA = rewriter.create<GpuAllocOp>(loc, arrayAType);
    auto arrayB = rewriter.create<GpuAllocOp>(loc, arrayBType);

    LLVM_DEBUG(llvm::dbgs()
               << "ldsBlockSize:        " << ldsBlockSize << "\n"
               << "mRepeats:            " << mRepeats << "\n"
               << "nRepeats:            " << nRepeats << "\n"
               << "nResultVectors:      " << nResultVectors << "\n"
               << "inputSpansPerMfmaIn: " << inputSpansPerMfmaIn << "\n"
               << "blocksInOutRegs:     " << blocksInOutRegs << "\n"
               << "arrayASize:          " << arrayASize << "\n"
               << "arrayBSize:          " << arrayBSize << "\n"
               << "isKReduction:        " << isKReduction << "\n");

    // Logic to allocate 0-initialized vectors for C.
    Type destType = elementType;
    Type accumulatorType =
        obtainAccumulatorType(rewriter, elementType, destType);
    VectorType accumulatorVectorType =
        vectorType.cloneWith({}, accumulatorType);
    MemRefType regCAllocType =
        MemRefType::get(nResultVectors, accumulatorVectorType, AffineMap{},
                        /*memorySpace=*/privateMemoryAddressSpace);
    Value regCAllocOp = rewriter.create<rock::GpuAllocOp>(loc, regCAllocType);
    Value zeroConstantCOp = createZeroConstantOp(rewriter, loc, vectorType);
    rewriter.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);

    // emit blockwise_gemm_v2
    BlockwiseGemmV2Op blockwiseGemmV2Op;
    int64_t ldsBlockAOffset = 0;
    int64_t ldsBlockBOffset = 0;
    XdlopsGemmParamsAttr gemmParams = rewriter.getAttr<XdlopsGemmParamsAttr>(
        kpacksPerBlock, mPerBlock, nPerBlock, kpack, mPerWave, nPerWave,
        /*forceUnroll*/ true);
    blockwiseGemmV2Op = rewriter.create<BlockwiseGemmV2Op>(
        loc, ldsBufferA, ldsBufferB, rewriter.getIndexAttr(ldsBlockAOffset),
        rewriter.getIndexAttr(ldsBlockAOffset), mMyWaveOffsetA, mMyWaveOffsetB,
        arrayA, arrayB, regCAllocOp, rewriter.getI32IntegerAttr(blockSize),
        gemmParams);

    // insert triton_gpu.memref_to_tensor to convert the result of
    // blockwise_gemm to a tensor with the #mfma encoding
    uint32_t elemsPerThread = vectorType.getShape()[0] * nResultVectors;
    Attribute dEnc = op.getD().getType().cast<RankedTensorType>().getEncoding();
    if (!dEnc.isa<MfmaEncodingAttr>())
      return emitError(loc) << "Result of dot must be MfmaEncodingAttr.\n";
    auto dEncMfma = dEnc.cast<MfmaEncodingAttr>();
    RankedTensorType toTensorType =
        RankedTensorType::get(cShape, accumulatorType, dEncMfma);

    LLVM_DEBUG(llvm::dbgs() << "elemsPerThread: " << elemsPerThread << "\n");
    Value toTensorOp = rewriter.create<triton::gpu::MemRefToTensorOp>(
        loc, toTensorType, regCAllocOp);

    op.replaceAllUsesWith(toTensorOp);

    // Erase tt.dot and the two convert_layout ops that convert
    // #shared to #dot_op for a and b
    rewriter.eraseOp(op);
    rewriter.eraseOp(matADefiningOp);
    rewriter.eraseOp(matBDefiningOp);
    return success();
  }

private:
};
} // end anonymous namespace

namespace {

class ConvertTritonGPUToRock
    : public impl::ConvertTritonGPUToRockBase<ConvertTritonGPUToRock> {

public:
  using impl::ConvertTritonGPUToRockBase<
      ConvertTritonGPUToRock>::ConvertTritonGPUToRockBase;
  explicit ConvertTritonGPUToRock(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    /* preprocess */
    decomposeMmaToDotOperand(mod, numWarps);
    decomposeBlockedToDotOperand(mod);
    if (failed(decomposeInsertSliceAsyncOp(mod)))
      return signalPassFailure();

    processDotInLoop(mod);
    // new step 1
    // replace tt.dot with rock.blockwise_gemm_v2
    ConversionTarget target(*context);
    target.addIllegalOp<triton::DotOp>();
    target.addLegalDialect<
        arith::ArithDialect, rock::RockDialect, memref::MemRefDialect,
        AffineDialect, triton::gpu::TritonGPUDialect, vector::VectorDialect>();

    RewritePatternSet patterns(context);
    patterns.add<DotOpRewritePattern>(context);
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  int computeCapability{};

  // LogicalResult
  // matchAndRewriteDotInLoop(triton::DotOp op,
  //                        mlir::PatternRewriter &rewriter) const {

  LogicalResult processDotInLoop(ModuleOp mod) const {

    mod.walk([&](triton::DotOp op) -> void {
      OpBuilder rewriter(op);

      auto loc = op.getLoc();
      Value matA = op.getA(), matB = op.getB(), matC = op.getC();
      // If matA has a defining op, then tt.dot is not in a loop.
      // Let the pattern rewriter handle this case.
      if (matA.getDefiningOp())
        return;
      Attribute dEnc =
          op.getD().getType().cast<RankedTensorType>().getEncoding();
      if (!dEnc.isa<MfmaEncodingAttr>()) {
        emitError(loc) << "Result of dot must be MfmaEncodingAttr.\n";
        return;
      }

      auto aShape = matA.getType().cast<RankedTensorType>().getShape();
      auto bShape = matB.getType().cast<RankedTensorType>().getShape();
      auto cShape = op.getC().getType().cast<RankedTensorType>().getShape();
      auto elementType =
          matA.getType().cast<RankedTensorType>().getElementType();
      uint32_t mPerBlock = aShape[0];
      uint32_t kPerBlock = aShape[1];
      uint32_t nPerBlock = bShape[1];

      // The GPU kernel is launched with blockSize = 32*numWarps
      // TODO: make it less confusing when working on AMDGPUs
      // TODO: enable warpSize = 64
      uint32_t numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      uint32_t blockSize = 64 * numWarps;

      constexpr int64_t waveSize = 64;
      uint32_t numWaves = blockSize / waveSize;

      // Tuning parameters
      // TODO: make kpack come from a single source of truth. E.g. tuning
      // parameter when launching the kernel
      int64_t kpack = 4;
      int64_t kpacksPerBlock = kPerBlock / kpack;
      // TODO: in rocMLIR, blockSize is derived from [m|n]PerBlock,
      // [m|n]PerWave, and waveSize. In triton, blockSize and [m|n]PerBlock are
      // given. Therefore, mPerWave and nPerWave need to be derived.
      // TODO: this is a duplicate from BlockedToMFMA in the
      // tritongpu-accelerate-matmul pass. We should unify them in the future.
      uint32_t mPerWave = std::min<uint32_t>(32, mPerBlock);
      uint32_t nPerWave = mPerBlock * nPerBlock / numWarps / mPerWave;
      if (mPerBlock * nPerBlock / (mPerWave * nPerWave) != numWaves) {
        emitError(loc) << "Need to pick another [m|n]PerWave!\n";
        return;
      }
      // When numWarps are too small, we need larger mPerWave and nPerWave
      if (nPerWave > nPerBlock) {
        nPerWave = nPerBlock;
        mPerWave = mPerBlock * nPerBlock / numWarps / nPerWave;
      }
      int64_t nWaves = nPerBlock / nPerWave;

      LLVM_DEBUG(llvm::dbgs() << "mPerBlock:      " << mPerBlock << "\n"
                              << "kPerBlock:      " << kPerBlock << "\n"
                              << "nPerBlock:      " << nPerBlock << "\n"
                              << "blockSize:      " << blockSize << "\n"
                              << "kpack:          " << kpack << "\n"
                              << "kpacksPerBlock: " << kpacksPerBlock << "\n"
                              << "mPerWave:       " << mPerWave << "\n"
                              << "nPerWave:       " << nPerWave << "\n"
                              << "nWaves:         " << nWaves << "\n"
                              << "numWarps:       " << numWarps << "\n"
                              << "numWaves:       " << numWaves << "\n");

      // Mfma instruction group selection.
      auto maybeMfmaInsnGroup =
          MfmaInsnGroup::select(elementType, mPerWave, nPerWave);
      if (failed(maybeMfmaInsnGroup)) {
        emitError(loc) << "Failed to select xdlops instruction group.\n";
        return;
      }
      MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
      if (!mfmaGroup.isCoherentWithK(kpack, kPerBlock)) {
        emitError(loc) << "Mfma instruction group selection is not "
                          "compatible with k.\n";
        return;
      }
      int64_t mRepeats = mfmaGroup.getMRepeats(mPerWave);
      int64_t nRepeats = mfmaGroup.getNRepeats(nPerWave);
      auto imms = mfmaGroup.getImms();
      int64_t nResultVectors = imms.size() * mRepeats * nRepeats;

      VectorType vectorType = mfmaGroup.getRetType();
      MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
      int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
      int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;

      // Logic to setup buffers A and B for blockwise_gemm_v2.
      bool isKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);
      int64_t arrayASize =
          (!isKReduction) ? (kpacksPerBlock * mRepeats)
                          : (kpacksPerBlock / inputSpansPerMfmaIn * mRepeats);
      int64_t arrayBSize =
          (!isKReduction) ? (kpacksPerBlock * nRepeats)
                          : (kpacksPerBlock / inputSpansPerMfmaIn * nRepeats);
      Type arrayAType, arrayBType;
      auto privateMemoryAddressSpace =
          rewriter.getAttr<mlir::gpu::AddressSpaceAttr>(
              mlir::gpu::GPUDialect::getPrivateAddressSpace());
      if (kpack > 1) {
        arrayAType =
            MemRefType::get({arrayASize}, VectorType::get({kpack}, elementType),
                            AffineMap{}, privateMemoryAddressSpace);
        arrayBType =
            MemRefType::get({arrayBSize}, VectorType::get({kpack}, elementType),
                            AffineMap{}, privateMemoryAddressSpace);
      } else {
        arrayAType = MemRefType::get({arrayASize}, elementType, AffineMap{},
                                     privateMemoryAddressSpace);
        arrayBType = MemRefType::get({arrayBSize}, elementType, AffineMap{},
                                     privateMemoryAddressSpace);
      }
      // Logic to allocate 0-initialized vectors for C.
      Type destType = elementType;
      Type accumulatorType =
          obtainAccumulatorType(rewriter, elementType, destType);
      VectorType accumulatorVectorType =
          vectorType.cloneWith({}, accumulatorType);
      MemRefType regCAllocType =
          MemRefType::get(nResultVectors, accumulatorVectorType, AffineMap{},
                          /*memorySpace=*/privateMemoryAddressSpace);

      // In this case, tt.dot's operands all come from block arguments
      auto dotOpA = matA.dyn_cast<BlockArgument>();
      auto dotOpB = matB.dyn_cast<BlockArgument>();
      auto dotOpC = matC.dyn_cast<BlockArgument>();
      uint dotOpAIdx = 0;
      uint dotOpBIdx = 0;
      uint accIdx = 0;
      if (dotOpA && dotOpB) {
        dotOpAIdx = dotOpA.getArgNumber();
        dotOpBIdx = dotOpB.getArgNumber();
        accIdx = dotOpC.getArgNumber();
      } else {
        emitError(loc) << "Invalid block arguments!\n";
        return;
      }

      // Obtain the head of the for loop
      auto loopBB = dotOpA.getOwner();
      int cnt = 0;
      // Before processing tt.dot, we first process the two predeccors of the
      // loop head block:
      // 1. The entry block
      // 2. The current block, which is the block that tt.dot resides in
      uint bufferCIdx = 0;
      uint extractTensorAIdx = 0;
      uint extractTensorBIdx = 0;
      Type extractTensorATy, extractTensorBTy;
      Value mMyWaveOffsetA, mMyWaveOffsetB;
      Value arrayA, arrayB;
      for (auto it = loopBB->pred_begin(), e = loopBB->pred_end(); it != e;
           ++it) {
        auto predBB = *it;
        bool isEntryBB = predBB->isEntryBlock();

        // Obtain the cf.br op
        auto brOp = predBB->getTerminator();

        // Now obtain the defining op of dotOpA and dotOpB
        auto matADefiningOp = brOp->getOperand(dotOpAIdx).getDefiningOp();
        auto matBDefiningOp = brOp->getOperand(dotOpBIdx).getDefiningOp();
        if (!(isConvertLDSToDotOp(matADefiningOp) &&
              isConvertLDSToDotOp(matBDefiningOp))) {
          LLVM_DEBUG(llvm::dbgs() << "operand a or b does not come from LDS\n");
          emitError(loc) << "operand a or b does not come from LDS\n";
          return;
        }
        // Obtain the convert_layout #lds -> #dot_op
        auto cvtOpA = dyn_cast<triton::gpu::ConvertLayoutOp>(matADefiningOp);
        auto cvtOpB = dyn_cast<triton::gpu::ConvertLayoutOp>(matBDefiningOp);
        // Obtain the extract_slice op
        auto extractSliceOpA = dyn_cast<triton::gpu::ExtractSliceOp>(
            cvtOpA.getSrc().getDefiningOp());
        auto extractSliceOpB = dyn_cast<triton::gpu::ExtractSliceOp>(
            cvtOpB.getSrc().getDefiningOp());
        if (!extractSliceOpA || !extractSliceOpB) {
          emitError(loc) << "Failed to find the extract_slice op.\n";
          return;
        }
        // Save their type since we need to add them into the block argument
        // list of loopBB
        extractTensorATy = extractSliceOpA.getResult().getType();
        extractTensorBTy = extractSliceOpB.getResult().getType();
        // Obtain the insert_slice op
        // Note that insert_slice_async should be decomposed to
        // load and insert_slice already
        auto insertSliceOpA = dyn_cast<triton::gpu::InsertSliceOp>(
            extractSliceOpA.getSource().getDefiningOp());
        auto insertSliceOpB = dyn_cast<triton::gpu::InsertSliceOp>(
            extractSliceOpB.getSource().getDefiningOp());
        if (!insertSliceOpA || !insertSliceOpB) {
          emitError(loc) << "Failed to find insert_slice op.\n";
          return;
        }

        // Changes only in the entry block:
        // 1. Parameter computations: tid, mPerWave, nPerWave, waveId,
        //    waveId_m, waveId_n, mMyWaveOffsetA, mMyWaveOffsetB
        // 2. Allocate bufferA, bufferB, and bufferC
        // 3. Zero initialize bufferC
        // 4. Insert %bufferC as the last operand in cf.br
        uint brOperandInsertIdx = brOp->getNumOperands();
        if (isEntryBB) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointAfter(cvtOpB);
          auto _waveSize = rewriter.create<ConstantIndexOp>(loc, waveSize);
          auto _mPerWave = rewriter.create<ConstantIndexOp>(loc, mPerWave);
          auto _nPerWave = rewriter.create<ConstantIndexOp>(loc, nPerWave);
          auto _nWaves = rewriter.create<ConstantIndexOp>(loc, nWaves);
          auto _tid =
              rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());
          auto _waveId = rewriter.create<DivUIOp>(loc, _tid, _waveSize);
          auto _waveId_m = rewriter.create<DivUIOp>(loc, _waveId, _nWaves);
          auto _waveId_n = rewriter.create<RemUIOp>(loc, _waveId, _nWaves);

          mMyWaveOffsetA = rewriter.create<MulIOp>(loc, _waveId_m, _mPerWave);
          mMyWaveOffsetB = rewriter.create<MulIOp>(loc, _waveId_n, _nPerWave);

          arrayA = rewriter.create<GpuAllocOp>(loc, arrayAType);
          arrayB = rewriter.create<GpuAllocOp>(loc, arrayBType);
          Value regCAllocOp =
              rewriter.create<rock::GpuAllocOp>(loc, regCAllocType);
          Value zeroConstantCOp =
              createZeroConstantOp(rewriter, loc, vectorType);
          rewriter.create<FillOp>(loc, regCAllocOp, zeroConstantCOp);
          brOp->insertOperands(brOperandInsertIdx, {regCAllocOp});
        }

        // Changes common in both entry block and current block
        // 1. erase two convert_layout #lds -> #dot_op
        // 2. in cf.br
        //    - remove %DotOpA and %DotOpB
        //    - remove %acc
        //    - add %extractTensorA and %extractTensorB
        // Note that we can only add %bufferC in the cf.br operand list of
        // the current block later
        brOp->eraseOperands(dotOpAIdx, 2);
        brOp->eraseOperand(accIdx);
        cvtOpA.erase();
        cvtOpB.erase();
        // because we removed 3 operands from cf.br
        extractTensorAIdx = brOperandInsertIdx - 3;
        extractTensorBIdx = extractTensorAIdx + 1;
        // In the case of entry block, %bufferC is already inserted.
        // brOperandInsertIdx - 3 is the position before %bufferC
        brOp->insertOperands(brOperandInsertIdx - 3,
                             {extractSliceOpA, extractSliceOpB});
      }

      // Changes in the current block
      // 1. Insert two tensor_to_memref to prepare LDSBufferA and LDSBufferB
      // 2. Insert blockwise_gemm_v2
      // 3. Add %bufferC at the end of cf.br's operand list
      // 4. erase tt.dot, and erase %dotOpA and %dotOpB in loopBB blockArg list

      // Step 1: insert tensor_to_memref
      auto LdsTensorA =
          loopBB->addArgument(extractTensorATy, rewriter.getUnknownLoc());
      auto LdsTensorB =
          loopBB->addArgument(extractTensorBTy, rewriter.getUnknownLoc());
      auto workgroupMemoryAddressSpace =
          rewriter.getAttr<mlir::gpu::AddressSpaceAttr>(
              mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
      auto LDSTensorAShape =
          LdsTensorA.getType().cast<RankedTensorType>().getShape();
      auto LDSTensorBShape =
          LdsTensorB.getType().cast<RankedTensorType>().getShape();
      int64_t ldsBlockASize = product<long int>(LDSTensorAShape);
      int64_t ldsBlockBSize = product<long int>(LDSTensorBShape);
      auto ldsAMemRefType =
          MemRefType::get({ldsBlockASize}, elementType, AffineMap{},
                          workgroupMemoryAddressSpace);
      auto ldsBMemRefType =
          MemRefType::get({ldsBlockBSize}, elementType, AffineMap{},
                          workgroupMemoryAddressSpace);
      auto ldsBufferA = rewriter.create<triton::gpu::TensorToMemRefOp>(
          loc, ldsAMemRefType, LdsTensorA);
      auto ldsBufferB = rewriter.create<triton::gpu::TensorToMemRefOp>(
          loc, ldsBMemRefType, LdsTensorB);

      // Step 2: insert blockwise_gemm_v2
      auto regCAllocOp =
          loopBB->addArgument(regCAllocType, rewriter.getUnknownLoc());
      BlockwiseGemmV2Op blockwiseGemmV2Op;
      int64_t ldsBlockAOffset = 0;
      int64_t ldsBlockBOffset = 0;
      XdlopsGemmParamsAttr gemmParams = rewriter.getAttr<XdlopsGemmParamsAttr>(
          kpacksPerBlock, mPerBlock, nPerBlock, kpack, mPerWave, nPerWave,
          /*forceUnroll*/ true);
      blockwiseGemmV2Op = rewriter.create<BlockwiseGemmV2Op>(
          loc, ldsBufferA, ldsBufferB, rewriter.getIndexAttr(ldsBlockAOffset),
          rewriter.getIndexAttr(ldsBlockAOffset), mMyWaveOffsetA,
          mMyWaveOffsetB, arrayA, arrayB, regCAllocOp,
          rewriter.getI32IntegerAttr(blockSize), gemmParams);

      // Step 3: Add %bufferC in the cf.br's operand list
      auto brOp = op->getBlock()->getTerminator();
      brOp->insertOperands(brOp->getNumOperands(), {regCAllocOp});

      // Step 4: erase tt.dot
      //         erase %dotOpA and %dotOpB fro the block argument list of loopBB
      op.erase();
      loopBB->eraseArguments(dotOpAIdx, 2);

      // In the block after the loop
      // Insert memref_to_tensor
      if (!dotOpC.hasOneUse()) {
        emitError(loc) << "dotOpC has more than one users!\n";
        return;
      }
      auto cvtOpC = dotOpC.getUsers().begin();
      auto dEncMfma = dEnc.cast<MfmaEncodingAttr>();
      RankedTensorType toTensorType =
          RankedTensorType::get(cShape, accumulatorType, dEncMfma);
      rewriter.setInsertionPoint(*cvtOpC);
      Value toTensorOp = rewriter.create<triton::gpu::MemRefToTensorOp>(
          cvtOpC->getLoc(), toTensorType, regCAllocOp);

      dotOpC.replaceAllUsesWith(toTensorOp);
      loopBB->eraseArgument(accIdx);
    });
    return success();
  }

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps) const {
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcMma, dstDotOp)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) const {
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                getOrder(srcBlocked), srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  LogicalResult decomposeInsertSliceAsyncOp(ModuleOp mod) const {
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AxisInfoAnalysis *axisInfoAnalysis = solver->load<AxisInfoAnalysis>();
    if (failed(solver->initializeAndRun(mod)))
      return failure();
    // TODO(Keren): This is a hacky knob that may cause performance regression
    // when decomposition has been performed. We should remove this knob once we
    // have thorough analysis on async wait. Currently, we decompose
    // `insert_slice_async` into `load` and `insert_slice` without knowing which
    // `async_wait` is responsible for the `insert_slice_async`. To guarantee
    // correctness, we blindly set the `async_wait` to wait for all async ops.
    //
    // There are two options to improve this:
    // 1. We can perform a dataflow analysis to find the `async_wait` that is
    // responsible for the `insert_slice_async` in the backend.
    // 2. We can modify the pipeline to perform the decomposition before the
    // `async_wait` is inserted. However, it is also risky because we don't know
    // the correct vectorized shape yet in the pipeline pass. Making the
    // pipeline pass aware of the vectorization could introduce additional
    // dependencies on the AxisInfoAnalysis and the Coalesce analysis.
    bool decomposed = false;
    // insert_slice_async %src, %dst, %idx, %mask, %other
    // =>
    // %tmp = load %src, %mask, %other
    // %res = insert_slice %tmp into %dst[%idx]
    mod.walk([&](triton::gpu::InsertSliceAsyncOp insertSliceAsyncOp) -> void {
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.getSrc();
      auto dst = insertSliceAsyncOp.getDst();
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resLDSLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::LDSEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis->getPtrContiguity(src);
      unsigned outVec = resLDSLayout.getKpack();
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
#ifndef USE_ROCM
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth))
        return;
#endif

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.getSrc(),
          insertSliceAsyncOp.getMask(), insertSliceAsyncOp.getOther(),
          insertSliceAsyncOp.getCache(), insertSliceAsyncOp.getEvict(),
          insertSliceAsyncOp.getIsVolatile());

      // insert_slice
      auto axis = insertSliceAsyncOp.getAxis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      offsets[axis] = insertSliceAsyncOp.getIndex();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<triton::gpu::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.getDst(),
          offsets, sizes, strides);

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(computeCapability))
        asyncCommitGroupOp.erase();
      OpBuilder builder(asyncCommitGroupOp);
      builder.create<mlir::amdgpu::LDSBarrierOp>(asyncCommitGroupOp.getLoc());
      asyncCommitGroupOp.erase();
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
#ifdef USE_ROCM
      assert(decomposed &&
             "AsyncWait is not supported for ROCM and should be removed");
      asyncWaitOp.erase();
#else
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        asyncWaitOp.erase();
      }
#endif
    });
    return success();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToRockPass(int computeCapability) {
  return std::make_unique<::ConvertTritonGPUToRock>(computeCapability);
}

} // namespace triton
} // namespace mlir
