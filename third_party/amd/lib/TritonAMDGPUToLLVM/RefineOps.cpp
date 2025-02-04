#include "TritonAMDGPUToLLVM/Passes.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
// #include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
// #include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
// #include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
// #include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// #include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
// #include "triton/Analysis/Allocation.h"
// #include "triton/Analysis/AxisInfo.h"
// #include "triton/Analysis/Membar.h"
// #include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
// #include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/MfmaGroup.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUREFINEOPS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// TODO: take the implementation from `ReorderInstructions.cpp`
static SmallVector<scf::ForOp> getLeafForOps(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> allOps;
  funcOp->walk([&](scf::ForOp forOp) { allOps.push_back(forOp); });

  SmallVector<scf::ForOp> leafOps;
  for (scf::ForOp forOp : allOps) {
    auto searchResult = forOp.getBody()->walk(
        [](scf::ForOp) { return WalkResult::interrupt(); });
    if (!searchResult.wasInterrupted())
      leafOps.push_back(forOp);
  }
  return leafOps;
}

SmallVector<Value> createOffset(llvm::ArrayRef<Value> valueOffset,
                                llvm::ArrayRef<int32_t> intOffset,
                                OpBuilder &rewriter, Location loc) {
  SmallVector<Value> values;
  for (auto item : valueOffset) {
    values.push_back(item);
  }

  for (auto item : intOffset) {
    Value value = rewriter.create<arith::ConstantIntOp>(loc, item, 32);
    values.push_back(value);
  }
  return values;
}

enum InstructionKindMask {
  NONE =        0x0000,
  ALL_ALU =     0x0001,
  VALU =        0x0002,
  SALU =        0x0004,
  MFMA =        0x0008,
  ALL_VMEM =    0x0010,
  VMEM_READ =   0x0020,
  VMEM_WRITE =  0x0040,
  ALL_DS =      0x0080,
  DS_READ =     0x0100,
  DS_WRITE =    0x0200,
  TRANSCEND =   0x0400
};

int32_t mfmaMask = NONE
    // | InstructionKindMask::ALL_ALU
    | InstructionKindMask::VALU
    | InstructionKindMask::SALU
    // | InstructionKindMask::MFMA
    | InstructionKindMask::ALL_VMEM
    | InstructionKindMask::VMEM_READ
    | InstructionKindMask::VMEM_WRITE
    | InstructionKindMask::ALL_DS
    | InstructionKindMask::DS_READ
    | InstructionKindMask::DS_WRITE
    | InstructionKindMask::TRANSCEND;

int32_t dsReadMask = NONE
    | InstructionKindMask::ALL_ALU
    | InstructionKindMask::VALU
    | InstructionKindMask::SALU
    | InstructionKindMask::MFMA
    | InstructionKindMask::ALL_VMEM
    | InstructionKindMask::VMEM_READ
    | InstructionKindMask::VMEM_WRITE
    // | InstructionKindMask::ALL_DS
    // | InstructionKindMask::DS_READ
    | InstructionKindMask::DS_WRITE
    | InstructionKindMask::TRANSCEND;

void createSchedBarrier(OpBuilder &rewriter, Location loc,
                        int32_t maskValue) {
  const char *intrinsicName = "llvm.amdgcn.sched.barrier";
  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, TypeRange{},
                                  ValueRange{mask});
}

class CoordinateAux {
public:
  CoordinateAux(llvm::ArrayRef<int64_t> layout) : layout(layout) {
    bounds.resize(layout.size());
    std::exclusive_scan(layout.rbegin(), layout.rend(), bounds.begin(), 1,
                        std::multiplies<>());
  }

  SmallVector<int64_t> map(int64_t index) {
    SmallVector<int64_t> coords(bounds.size(), 0);
    for (size_t i = 1; i < bounds.size(); ++i) {
      size_t d = bounds.size() - i;
      coords[d] = index / bounds[d];
      index = index % bounds[d];
    }
    coords[0] = index;
    std::reverse(coords.begin(), coords.end());
    return coords;
  }

private:
  llvm::ArrayRef<int64_t> layout;
  std::vector<int> bounds;
};

inline bool isRowMajor(::llvm::ArrayRef<unsigned> order) {
  auto rank = order.size();
  return order[rank - 1] == 0;
}

/*
 TODO - this needs to be MUCH more official.
*/
unsigned calcCyclesPerMfma(AMDMfmaEncodingAttr mfmaLayout, DotOp dotOp) {
  // Get mfma op type.
  Value a = dotOp.getA();
  Value b = dotOp.getB();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto elemTyA = aTensorTy.getElementType();
  auto elemTyB = bTensorTy.getElementType();
  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  auto mfmaVersion = mfmaLayout.getVersionMajor();
  bool allowXF32 =
      dotOp.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
  auto maybeMfmaInsn = MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB,
                                              mfmaVersion, allowXF32);
  if (failed(maybeMfmaInsn))
    llvm::report_fatal_error("No match found in MFMA database\n");
  // Estimate rate of mfma op type.
  unsigned maxBitWidth = std::max(maybeMfmaInsn->getElementTypeA().getIntOrFloatBitWidth(),
                                  maybeMfmaInsn->getElementTypeB().getIntOrFloatBitWidth());
  // Estimate throughput as fma's per cycle.
  unsigned opsPerCycle;
  if (maxBitWidth <= 8) { // fp8, bf8, i8
    opsPerCycle = 512;
  } else if (maxBitWidth <= 16) { // fp16, bf16
    opsPerCycle = 256;
  } else if (maxBitWidth <= 32) { // fp32
    opsPerCycle = 128;
  } else {
    opsPerCycle = 64; // fp64
  }
  // total floating point mfmas
  int64_t totalOps = maybeMfmaInsn->getMDim()
      * maybeMfmaInsn->getNDim()
      * maybeMfmaInsn->getKDim();
  unsigned cyclesPerMfma = static_cast<unsigned>(totalOps / opsPerCycle);
  llvm::outs() << maybeMfmaInsn->getInsnName() << " = " << cyclesPerMfma << " cycles\n";
  return cyclesPerMfma;
}

/*
Calculate how many mfmas are in a rep, e.g. 1x1x2.
// TODO - is there a more direct method for this?
*/
SmallVector<unsigned, 3> calcMfmasPerRep(
    SmallVector<int64_t> ctaTile,
    SmallVector<unsigned> warpsPerCta,
    SmallVector<int64_t> numReps,
    SmallVector<unsigned> mfmaShape
) {
  // Tile size per warp.
  SmallVector<int64_t, 3> warpTile = {
    ctaTile[0] / warpsPerCta[0],
    ctaTile[1] / warpsPerCta[1],
    ctaTile[2],
  };
  // Tile size per rep.
  SmallVector<int64_t, 3> repTile = {
    warpTile[0] / numReps[0],
    warpTile[1] / numReps[1],
    warpTile[2] / numReps[2],
  };
  SmallVector<unsigned, 3> mfmasPerRep = {
    static_cast<unsigned>(repTile[0] / mfmaShape[0]),
    static_cast<unsigned>(repTile[1] / mfmaShape[1]),
    static_cast<unsigned>(repTile[2] / mfmaShape[2])};
  return mfmasPerRep;
}

// TODO(guacamoleo) - we also need to detect whether the a or b operands of a tile are already loaded
// into registers, e.g. for FA, and we want to choose the tile shape (small improvement)
// and tile order (rows vs colums which will be a large improvement).

/*
 Returns the dot tile size (in number of reps, not number of mfmas);
 typical sizes when mfmasPerRep = 1x1x2 for b128 and localLoadIssueRate=32 for b128
 2x2 for fp16 (128 mfma cycles per tile) and
 4x4 for fp8 (256 mfma cycles per tile).
 
 Tile size is chosen so that
 (1) A tile's worth of local_load_a or local_load_b can 
     can be issued during a tile's worth of mfmas, and
 (2) A tile's worth of mfmas hides the ds_read data latency.

 Args:
  - mfmasPerRep - 3D shape of number of mfmas in decomposed dot, e.g. 1x1x2.
  - preferLargerM - prefer M > N if not square.
  - cyclesPerMfma - how many cycles does mfma take in total.
  - localLoadIssueRate - cycles between issuing consecutive ds_reads to not overrun hardware queues.
      This is estimated to be b128 -> 32 cycles, b64 -> 16 cycles, b32 -> 8 cycles.
  - localLoadDataLatency - cycles between issuing ds_read and waiting for data; rounded up to pow2.

  Notes:

  The intended scheduling of tiles-of-reps-of-mfmas is 
  --------------------------------
  local_load_a[n]
  MFMA_Tile[n-3] // a can be loaded during tile
  --------------------------------
  local_load_b[n]
  MFMA_Tile[n-2] // b can be loaded during tile
  --------------------------------
  MFMA_Tile[n-1] // load data latency hiding
  --------------------------------
  MFMA_Tile[n] // all a,b data is ready by the first mfma of tile.
  --------------------------------

  This can be further refined if the data latency becomes much larger than
  the issue rate; in this case we can remove the condition that one tile
  hides all the data latency (which could make the tiles huge and waste registers),
  and intead local load issue rate is the only criteria and we retroactively calculate
  how many tiles are needed to hide the data latency.
*/
SmallVector<unsigned, 2> calcDotTileSize(
  SmallVector<unsigned, 3> mfmasPerRep, // = 16x16x64 / 16x16x32 = 1x1x2; served by 1 a,b load
  bool preferLargerM,
  unsigned cyclesPerMfma = 8,
  unsigned localLoadIssueRate = 32,
  unsigned localLoadDataLatency = 128
) {
  SmallVector<unsigned, 2> tileSize = {1, 1};
  int64_t numMfmas = tileSize[0]*tileSize[1]*mfmasPerRep[0]*mfmasPerRep[1]*mfmasPerRep[2];
  int64_t mfmaCycles = numMfmas * cyclesPerMfma;
  int64_t numLoads = std::max(tileSize[0]*mfmasPerRep[0], tileSize[1]*mfmasPerRep[1]);
  int64_t loadIssueCycles = numLoads * localLoadIssueRate;
  while (mfmaCycles < loadIssueCycles || mfmaCycles < localLoadDataLatency) {
    if (tileSize[0]*mfmasPerRep[0] < tileSize[1]*mfmasPerRep[1] || preferLargerM) {
      tileSize[0] *= 2;
    } else {
      tileSize[1] *= 2;
    }
    numMfmas = tileSize[0]*tileSize[1]*mfmasPerRep[0]*mfmasPerRep[1]*mfmasPerRep[2];
    mfmaCycles = numMfmas * cyclesPerMfma;
    numLoads = tileSize[0]*mfmasPerRep[0]+tileSize[1]*mfmasPerRep[1];
    loadIssueCycles = numLoads * localLoadIssueRate;
  };

  return tileSize;
}

/*
  DotTiling creates tiles of mfmas while they are decomposed from a dot operation.
  A tile of mfmas is a set of mfmas that will be co-scheduled
  because they use the same A,B operands; co-scheduling mfmas with same operands
  allows finer control over prefetching from LDS and register usage for these operands.
  Args:
   - inputNumRepM - total number of [decomposed] dot ops along m.
   - inputNumRepN - total number of [decomposed] dot ops along n.
   - inputTileSizeM - number of [decomposed] dot ops along m per tile.
   - inputTileSizeN - number of [decomposed] dot ops along n per tile.
   - inputOuterLoopM - should be set to (warpTileM >= warpTileN). True means m should be
       outer loop of mfma ops so that inner loop is smaller dimension which leads to smallest
       number of registers carrying A,B operands.
  E.g. numRep = 8x4, tileSize=2x2.
*/
struct DotTileOrder {
  const int numRepM;
  const int numRepN;
  const int tileSizeM;
  const int tileSizeN;
  const int numTilesM;
  const int numTilesN;
  bool outerTileM;
  int tileSizeOuter;
  int tileSizeInner;
  int numTilesOuter;
  int numTilesInner;

  explicit DotTileOrder(int inputNumRepM, int inputNumRepN,
                     int inputTileSizeM, int inputTileSizeN,
                     bool inputOuterLoopM)
      : numRepM(inputNumRepM),
        numRepN(inputNumRepN),
        tileSizeM(inputTileSizeM),
        tileSizeN(inputTileSizeN),
        numTilesM(numRepM / tileSizeM),
        numTilesN(numRepN / tileSizeN),
        outerTileM(inputOuterLoopM) {
    // Num mfmas must evenly divide into tiles.
    assert(numTilesM * tileSizeM == numRepM);
    assert(numTilesN * tileSizeN == numRepN);
    // Assign M and N to be outer vs inner tile loop.
    if (outerTileM) {
      // M is tile of outer loop.
      tileSizeOuter = tileSizeM;
      tileSizeInner = tileSizeN;
      numTilesOuter = numTilesM;
      numTilesInner = numTilesN;
    } else {
      // N is tile of outer loop.
      tileSizeOuter = tileSizeN;
      tileSizeInner = tileSizeM;
      numTilesOuter = numTilesN;
      numTilesInner = numTilesM;
    }
  }
  int getTileSizeM() const { return tileSizeM; }
  int getTileSizeN() const { return tileSizeN; }
  int getNumTilesOuter() const { return numTilesOuter; }
  int getNumTilesInner() const { return numTilesInner; }
  int getTileStartM(int tileOuterIdx, int tileInnerIdx) const {
    if (outerTileM) {
      return tileOuterIdx * tileSizeOuter; // M is outer tile loop.
    } else {
      return tileInnerIdx * tileSizeInner; // M is inner tile loop.
    }
  }
  int getTileStartN(int tileOuterIdx, int tileInnerIdx) const {
    if (outerTileM) {
      return tileInnerIdx * tileSizeInner; // N is inner tile loop.
    } else {
      return tileOuterIdx * tileSizeOuter; // N is outer tile loop.
    }
  }
};

struct DotOpMFMAConverter {
  AMDMfmaEncodingAttr mfmaLayout;
  OpBuilder &rewriter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConverter(AMDMfmaEncodingAttr mfmaLayout,
                              OpBuilder &rewriter, Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter), loc(loc),
        ctx(mfmaLayout.getContext()) {}

  LogicalResult convert(DotOp dotOp, DotOpAdaptor adaptor) const {
    llvm::outs() << "DotOpMFMAConverter::convert()\n";

    InputPrecisionAttr precisionAttr = dotOp.getInputPrecisionAttr();
    SmallVector<unsigned> warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    Value c = dotOp.getC();
    Value d = dotOp.getD();

    auto localLoadA = cast<ttg::LocalLoadOp>(a.getDefiningOp());
    auto localLoadB = cast<ttg::LocalLoadOp>(b.getDefiningOp());

    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto cTensorTy = cast<RankedTensorType>(c.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());

    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();
    auto elemTyC = cTensorTy.getElementType();
    auto elemTyD = dTensorTy.getElementType();

    auto encodeA = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto encodeB = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    auto encodeC = cast<AMDMfmaEncodingAttr>(cTensorTy.getEncoding());
    auto encodeD = cast<AMDMfmaEncodingAttr>(dTensorTy.getEncoding());

    auto shapeA = aTensorTy.getShape();
    auto shapeB = bTensorTy.getShape();
    auto shapeC = cTensorTy.getShape();
    auto shapeD = dTensorTy.getShape();

    int kWidth = encodeA.getKWidth();
    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);
    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto memDescA = localLoadA->getOperand(0);
    auto memDescTypeA = cast<ttg::MemDescType>(memDescA.getType());
    auto memDescEncodingA =
        cast<triton::gpu::SharedEncodingAttr>(memDescTypeA.getEncoding());

    auto memDescB = localLoadB->getOperand(0);
    auto memDescTypeB = cast<ttg::MemDescType>(memDescB.getType());
    auto memDescEncodingB =
        cast<triton::gpu::SharedEncodingAttr>(memDescTypeB.getEncoding());

    // TODO: adjusting `numRepX` using `isRowMajor` is a work around.
    // it needs to be fixed in the future.
    auto aa = cast<triton::gpu::SharedEncodingAttr>(memDescTypeA.getEncoding())
                  .getOrder();
    //const auto numRepM = isRowMajor(memDescEncodingA.getOrder()) ? repA[1] : 1;
    //const auto numRepN = isRowMajor(memDescEncodingB.getOrder()) ? 1 : repB[2];
    const auto numRepM = repA[1];
    const auto numRepN = repB[2];
    const auto numRepK = repA[2];
    const auto numRepB = repA[0];
    SmallVector<int64_t> numRepShape = {numRepM , numRepN , numRepK };

    SmallVector<int64_t> refinedShapeA = {shapeA[0] / numRepM, shapeA[1] / numRepK};
    SmallVector<int64_t> refinedShapeB = {shapeB[0] / numRepK, shapeB[1] / numRepN};
    SmallVector<int64_t> refinedShapeCD = {shapeC[0] / numRepM, shapeC[1] / numRepN};

    // Calculate mfmas per rep.
    SmallVector<int64_t> ctaTile = {shapeC[0], shapeC[1], shapeA[1]};
    SmallVector<int64_t> warpTile = {
      shapeC[0] / warpsPerCTA[0],
      shapeC[1] / warpsPerCTA[1],
      shapeA[1],
      };
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    bool allowXF32 =
        dotOp.getInputPrecision() == InputPrecision::TF32 && mfmaVersion == 3;
    auto maybeMfmaInsn = MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB,
                                              mfmaVersion, allowXF32);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");
    SmallVector<unsigned> mfmaShape = {
      maybeMfmaInsn->getMDim(),
      maybeMfmaInsn->getNDim(),
      maybeMfmaInsn->getKDim()};

    auto mfmasPerRep = calcMfmasPerRep(
        ctaTile,
        warpsPerCTA,
        numRepShape,
        mfmaShape);
    llvm::outs() << "mfmasPerRep: "
      << mfmasPerRep[0] << "x"
      << mfmasPerRep[1] << "x"
      << mfmasPerRep[2] << "\n";

    // Calculate Tiling
    unsigned cyclesPerMfma = calcCyclesPerMfma(mfmaLayout, dotOp);
    // Prefer tile to be skinny along inner loop dimension to minimize registers.
    const bool preferOuterLoopM (warpTile[0] >= warpTile[1]);
    const bool preferTileLargerM = !preferOuterLoopM;
    // Calculate tile size (in reps per tile).
    auto tileSize = calcDotTileSize(mfmasPerRep, preferTileLargerM, cyclesPerMfma);

    llvm::outs() << "mfmasPerTile: "
        << tileSize[0] * mfmasPerRep[0] << "x"
        << tileSize[1] * mfmasPerRep[1] << "x"
        << 1 * mfmasPerRep[2] << "\n";
    const int tileSizeM = tileSize[0];
    const int tileSizeN = tileSize[1];
    const DotTileOrder dotTileOrder(numRepM, numRepN, tileSizeM, tileSizeM, preferOuterLoopM);

    /*
    Storing tiling information.
    mfma->opd->ds_read
    ds_read

    store
     - tileM, tileN, tileK, tileIdx, scheduleTileIdx (for mfmas it is actual, for memory ops it is where it should be scheduled)
     - how many of a,b local loaded from lds?

    (1) examine local_loads
      - a: m, ?, k, id
      - b: ?, n, k, id
    (2) query which tile it must be fetched right before.
      - a: m, 0, k, id
      - b: 0, n, k, id
    (3) calculate which tile it should be scheduled amongst.
      - a: m, 0, k, id
      - b: 0, n, k, id
    (4) iterate through mfmas looking for first one with the above info.
    */

    /*
     * Emit refined loads.
     */
    auto refinedTensorTypeA =
        RankedTensorType::get(refinedShapeA, elemTyA, encodeA);
    auto refinedTensorTypeB =
        RankedTensorType::get(refinedShapeB, elemTyB, encodeB);
    auto refinedTensorTypeC =
        RankedTensorType::get(refinedShapeCD, elemTyC, encodeC);
    auto refinedTensorTypeD =
        RankedTensorType::get(refinedShapeCD, elemTyD, encodeD);

    auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);

    constexpr bool mutableMemory = true;
    auto subviewTypeA = ttg::MemDescType::get(
        refinedShapeA, memDescTypeA.getElementType(),
        memDescTypeA.getEncoding(), sharedMemorySpace, mutableMemory);

    auto subviewTypeB = ttg::MemDescType::get(
        refinedShapeB, memDescTypeB.getElementType(),
        memDescTypeB.getEncoding(), sharedMemorySpace, mutableMemory);

    constexpr int M = 0;
    constexpr int N = 1;
    constexpr int K = 2;
    SmallVector<int64_t> elementsPerSlice = {refinedShapeCD[0], // M
                                             refinedShapeCD[1], // N
                                             refinedShapeA[1]}; // K
    llvm::outs() << "elementsPerSlice: "
        << elementsPerSlice[0] << "x"
        << elementsPerSlice[1] << "x"
        << elementsPerSlice[2] << "\n";
    rewriter.setInsertionPointAfter(localLoadA);
    SmallVector<SmallVector<ttg::LocalLoadOp>> subtilesA;
    unsigned tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      SmallVector<ttg::LocalLoadOp> subtilesK;
      for (int32_t i = 0; i < numRepM; ++i) {
        if (i % tileSizeM == 0) {
          llvm::outs() << "tile[m=" << i/tileSizeM << ", k=" << k << "]\n";
        }
        llvm::outs() << " - loadA[" << tileIdx << "] = " << i << ", " << k << "\n";
        tileIdx++;
        int32_t shiftM = i * elementsPerSlice[M];
        int32_t shiftK = k * elementsPerSlice[K];
        auto offset = createOffset({}, {shiftM, shiftK}, rewriter, loc);
        auto viewLoadA = rewriter.create<ttg::MemDescSubviewOp>(loc, subviewTypeA,
                                                                memDescA, offset);
        auto refinedLoadA =
            rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorTypeA, viewLoadA);
        /*
        triton::amdgpu::DotTileAttr dotTileAttr;
        dotTileAttr.m = i;
        dotTileAttr.n = 0;
        dotTileAttr.k = k;
        dotTileAttr.s = 12345;
        refinedLoadA.setAttr("DotTile", dotTileAttr);
        */
        subtilesK.push_back(refinedLoadA);
        if (false) {
          // Use sched.barrier to enforce the relative order of ds_reads.
          createSchedBarrier(rewriter, loc, dsReadMask);
        }
      }
      subtilesA.push_back(subtilesK);
    }
/*
  static DotTileAttr get(::mlir::MLIRContext *context, unsigned m, unsigned n, unsigned k, unsigned serial);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getM() const;
  unsigned getN() const;
  unsigned getK() const;
  unsigned getSerial() const;
*/

    rewriter.setInsertionPointAfter(localLoadB);
    SmallVector<SmallVector<ttg::LocalLoadOp>> subtilesB;
    tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      SmallVector<ttg::LocalLoadOp> subtilesK;
      for (int32_t j = 0; j < numRepN; ++j) {
       if (j % tileSizeN == 0) {
          llvm::outs() << "tile[n=" << j/tileSizeN << ", k=" << k << "]\n";
        }
        llvm::outs() << "loadB[" << tileIdx << "] = " << j << ", " << k << "\n";
        tileIdx++;
        int32_t shiftN = j * elementsPerSlice[N];
        int32_t shiftK = k * elementsPerSlice[K];
        auto offset = createOffset({}, {shiftK, shiftN}, rewriter, loc);
        auto viewLoadB = rewriter.create<ttg::MemDescSubviewOp>(loc, subviewTypeB,
                                                                memDescB, offset);
        auto refinedLoadB =
            rewriter.create<ttg::LocalLoadOp>(loc, refinedTensorTypeB, viewLoadB);
        subtilesK.push_back(refinedLoadB);
        if (false) {
          // Use sched.barrier to enforce the relative order of ds_reads.
          createSchedBarrier(rewriter, loc, dsReadMask);
        }
      }
      subtilesB.push_back(subtilesK);
    }

    /*
     * Emit refined dots.
     */
    rewriter.setInsertionPointAfter(dotOp);
    auto dotAttrs = dotOp->getAttrs();

    SmallVector<Value> refinedDotValues;
    // get 1st c opds
    for (int tileOuterIdx = 0; tileOuterIdx < dotTileOrder.getNumTilesOuter(); ++tileOuterIdx) {
      for (int tileInnerIdx = 0; tileInnerIdx < dotTileOrder.getNumTilesInner(); ++tileInnerIdx) {
        const int tileStartM = dotTileOrder.getTileStartM(tileOuterIdx, tileInnerIdx);
        const int tileStartN = dotTileOrder.getTileStartN(tileOuterIdx, tileInnerIdx);
        for (int m = tileStartM; m < tileStartM + tileSizeM; ++m) {
          for (int n = tileStartN; n < tileStartN + tileSizeN; ++n) {
            SmallVector<int64_t> offset = {m * elementsPerSlice[M],
                                           n * elementsPerSlice[N]};
            auto refinedTensorC = rewriter.create<triton::amdgpu::ExtractSliceOp>(
                loc, Type{refinedTensorTypeC}, Value{c}, offset);
            refinedDotValues.push_back(refinedTensorC);
          }
        }
      }
    }
    // dots
    tileIdx = 0;
    for (int32_t k = 0; k < numRepK; ++k) {
      for (int tileOuterIdx = 0; tileOuterIdx < dotTileOrder.getNumTilesOuter(); ++tileOuterIdx) {
        for (int tileInnerIdx = 0; tileInnerIdx < dotTileOrder.getNumTilesInner(); ++tileInnerIdx) {
          llvm::outs() << "tile[o=" << tileOuterIdx << ", i=" << tileInnerIdx << ", k=" << k << "]\n";
          const int tileStartM = dotTileOrder.getTileStartM(tileOuterIdx, tileInnerIdx);
          const int tileStartN = dotTileOrder.getTileStartN(tileOuterIdx, tileInnerIdx);
          // begin new tile
          for (int m = tileStartM; m < tileStartM + tileSizeM; ++m) {
            for (int n = tileStartN; n < tileStartN + tileSizeN; ++n) {
              llvm::outs() << " - dot[" << tileIdx << "] = " << m << ", " << n << ", " << k << "\n";
              tileIdx++;
              auto refinedTensorA = subtilesA[k][m];
              auto refinedTensorB = subtilesB[k][n];
              refinedDotValues[int32_t(m*numRepN+n)] = rewriter.create<tt::DotOp>(
                  loc, refinedTensorTypeD,
                  ValueRange{refinedTensorA, refinedTensorB,
                  refinedDotValues[int32_t(m*numRepN+n)]}, dotAttrs);

              if (false) {
                // Use sched.barrier to enforce the relative order of mfmas.
                createSchedBarrier(rewriter, loc, mfmaMask);
              }
            }
          }
        }
      }
    }

    auto concatDims = DenseI64ArrayAttr::get(ctx, {numRepM, numRepN});
    auto joinedDotsResult = rewriter.create<triton::amdgpu::ConcatOp>(
        loc, dTensorTy, refinedDotValues, concatDims);

    d.replaceAllUsesWith(joinedDotsResult);

    // Note: dangling localLoadA or/and localLoadB (if exist)
    // should be removed by the dead code elimination pass
    dotOp.erase();
    return success();
  }
};

inline RankedTensorType rankedTType(Value tensor) {
  return cast<RankedTensorType>(tensor.getType());
};

LogicalResult rewriteMFMA(OpBuilder &rewriter, triton::DotOp op) {
  if (!(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
        isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()))) {
    LDBG("Both $a and %b should be DotOperand layout");
    return failure();
  }

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  if (!isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding())) {
    LDBG("Currently, we only support $c with a mfma layout");
    return failure();
  }

  if (!(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
        cTensorTy.getShape()[1] == dTensorTy.getShape()[1])) {
    LDBG("DotOp's $c operand should pass the same number of values as $d");
    return failure();
  }

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConverter converter(mfmaLayout, rewriter, loc);
  return converter.convert(op, DotOpAdaptor(op));
}

struct RefinedBlock {
  RefinedBlock(ArrayRef<int64_t> shape, Type elemType,
               BlockedEncodingAttr encoding)
      : encoding(encoding), elemType(elemType) {
    auto ctaOrder = encoding.getCTAOrder();
    auto warpsPerCTA = encoding.getWarpsPerCTA();
    auto threadsPerWarp = encoding.getThreadsPerWarp();
    auto sizePerThread = encoding.getSizePerThread();

    numDims = warpsPerCTA.size();
    elementsPerWorkGroup.resize(numDims);
    numPerDims.resize(numDims);
    refinedShape.resize(numDims);
    for (size_t dim = 0; dim < numDims; ++dim) {
      elementsPerWorkGroup[dim] =
          sizePerThread[dim] * threadsPerWarp[dim] * warpsPerCTA[dim];
      numPerDims[dim] = shape[dim] / elementsPerWorkGroup[dim];
    }

    tensorType =
        RankedTensorType::get(elementsPerWorkGroup, elemType, encoding);
  }

  BlockedEncodingAttr encoding;
  Type elemType;
  SmallVector<int64_t> elementsPerWorkGroup;
  SmallVector<int64_t> numPerDims;
  SmallVector<int64_t> refinedShape;
  size_t numDims;
  RankedTensorType tensorType;
};

LogicalResult rewriteLoadOp(OpBuilder &rewriter, triton::LoadOp loadOp) {
  auto ctx = loadOp->getContext();
  auto loc = loadOp.getLoc();

  Value origSrc = loadOp->getOperand(0);
  Value origResult = loadOp.getResult();
  Type origResultType = loadOp.getResult().getType();
  auto origPtrs = rankedTType(origSrc);
  auto origShape = origPtrs.getShape();
  auto elemType = origPtrs.getElementType();
  auto encoding = dyn_cast<BlockedEncodingAttr>(origPtrs.getEncoding());
  if (encoding == nullptr)
    return failure();

  RefinedBlock refinedBlock(origShape, elemType, encoding);

  rewriter.setInsertionPointAfter(loadOp);
  SmallVector<Value> refinedTensors;

  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  auto boundaryCheck = loadOp.getBoundaryCheck();
  auto padding = loadOp.getPadding();
  auto cache = loadOp.getCache();
  auto evict = loadOp.getEvict();
  auto isVolatile = loadOp.getIsVolatile();
  ArrayRef<NamedAttribute> attrs = loadOp->getAttrs();

  CoordinateAux aux(refinedBlock.numPerDims);
  const auto numSubTiles =
      std::accumulate(refinedBlock.numPerDims.begin(),
                      refinedBlock.numPerDims.end(), 1, std::multiplies<>());

  for (size_t counter = 0; counter < numSubTiles; ++counter) {
    auto coords = aux.map(counter);
    SmallVector<int64_t> offset(refinedBlock.numDims, 0);
    for (auto [dim, coord] : llvm::enumerate(coords)) {
      offset[dim] = coord * refinedBlock.elementsPerWorkGroup[dim];
    }

    auto slice = rewriter.create<triton::amdgpu::ExtractSliceOp>(
        loc, Type{refinedBlock.tensorType}, Value{origSrc}, offset);

    auto refinedTensor =
        rewriter.create<triton::LoadOp>(loc, slice, mask, other, boundaryCheck,
                                        padding, cache, evict, isVolatile);
    refinedTensors.push_back(refinedTensor);
  }

  auto concatDims = DenseI64ArrayAttr::get(ctx, refinedBlock.numPerDims);
  auto joinedResult = rewriter.create<triton::amdgpu::ConcatOp>(
      loc, origResultType, refinedTensors, concatDims);

  origResult.replaceAllUsesWith(joinedResult);
  return success();
}

LogicalResult rewriteLocalStoreOp(OpBuilder &rewriter,
                                  triton::gpu::LocalStoreOp loadStoreOp) {
  auto ctx = loadStoreOp->getContext();
  auto loc = loadStoreOp.getLoc();

  Value origSrc = loadStoreOp->getOperand(0);
  auto origMemViewOp =
      cast<ttg::MemDescSubviewOp>(loadStoreOp->getOperand(1).getDefiningOp());
  Value origMemView = origMemViewOp->getOperand(0);

  auto origSrcType = rankedTType(origSrc);
  auto blockEncoding = dyn_cast<BlockedEncodingAttr>(origSrcType.getEncoding());
  if (blockEncoding == nullptr)
    return failure();

  auto origMemViewType = cast<ttg::MemDescType>(origMemView.getType());
  auto sharedEncoding =
      cast<triton::gpu::SharedEncodingAttr>(origMemViewType.getEncoding());
  if (sharedEncoding == nullptr)
    return failure();

  RefinedBlock refinedBlock(origSrcType.getShape(),
                            origSrcType.getElementType(), blockEncoding);

  // auto origMemViewType = rankedTType(origMemView);
  // auto shape = ptrs.getShape();
  // auto elemType = ptrs.getElementType();
  // auto encoding = dyn_cast<BlockedEncodingAttr>(ptrs.getEncoding());
  // if (encoding == nullptr)
  //   return failure();
  //
  //
  llvm::outs() << "applying \n";
  return success();
}

struct TritonAMDGPURefineOps
    : public triton::impl::TritonAMDGPURefineOpsBase<TritonAMDGPURefineOps> {
  explicit TritonAMDGPURefineOps(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::triton::AMD::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == mlir::triton::AMD::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    mod->walk([&](triton::FuncOp funcOp) {
      SmallVector<scf::ForOp> forOps = getLeafForOps(funcOp);
      for (auto forOp : forOps) {

        forOp.walk([&](triton::DotOp dotOp) {
          OpBuilder rewriter(dotOp->getContext());

          // TODO: extend to WMMA instructions
          if (failed(rewriteMFMA(rewriter, dotOp))) {
            LDBG("failed to refine tt.dotOp: " << *dotOp);
          }
        });

        forOp->walk([&](triton::LoadOp loadOp) {
          OpBuilder rewriter(loadOp->getContext());
          if (loadOp->getNumOperands() == 1)
            if (failed(rewriteLoadOp(rewriter, loadOp))) {
              LDBG("failed to refine tt.loadOp: " << *loadOp);
            }
        });

        forOp->walk([&](triton::gpu::LocalStoreOp storeOp) {
          OpBuilder rewriter(storeOp->getContext());
          if (storeOp->getNumOperands() == 2)
            if (failed(rewriteLocalStoreOp(rewriter, storeOp))) {
              LDBG("failed to refine ttg.localLoadOp: " << *storeOp);
            }
        });
      }
    });
  }

private:
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPURefineOpsPass(StringRef targetArch) {
  return std::make_unique<TritonAMDGPURefineOps>(targetArch);
}

} // namespace mlir::triton
