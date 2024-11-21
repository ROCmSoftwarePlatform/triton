/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "../PatternTritonGPUOpToLLVM.h"
#include "../TritonAMDGPUToLLVM/SchedInstructions.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

using ValueTable = std::map<std::array<int, 3>, Value>;

// Bitmask that encodes instruction types for LLVM AMD scheduling hints.
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

void createSchedBarrier(PatternRewriter &rewriter, Location loc,
                        int32_t maskValue) {
  const char *intrinsicName = "llvm.amdgcn.sched.barrier";
  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, TypeRange{},
                                  ValueRange{mask});
}

void createSchedGroupBarrier(PatternRewriter &rewriter, Location loc,
                             int32_t maskValue, int sizeValue,
                             int groupIdValue) {
  const char *intrinsicName = "llvm.amdgcn.sched.group.barrier";

  Value mask =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(maskValue));
  Value size =
      LLVM::createConstantI32(loc, rewriter, static_cast<int32_t>(sizeValue));
  Value groupId = LLVM::createConstantI32(loc, rewriter,
                                          static_cast<int32_t>(groupIdValue));

  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName, TypeRange{},
                                  ValueRange{mask, size, groupId});
};

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  Value generateMFMAOp(StringRef mfmaInsnName, Value valA, Value valB,
                       Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    OperationState loweredOp(loc, mfmaInsnName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if ((mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64))
      return 1;
    assert(mDim == nDim);
    switch (mDim) {
    case 32:
    case 16:
      return 1;
      break;
    case 4:
      assert(elementType.getIntOrFloatBitWidth() <= 32 &&
             "fp64 is not supported yet");
      assert(elementType.getIntOrFloatBitWidth() != 8 ||
             elementType.isInteger(8) && "fp8 is not supported yet");
      return 16;
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int warpSize = 64;
    int subBlockSize = warpSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = and_(laneId, i32_val(warpSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < warpSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = add(accScalar[i], other_acc);
          else
            accScalar[i] = fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = i32_val(0);
      else
        zero = f32_val(0.0);
      auto cond = icmp_ult(laneId, i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = select(cond, accScalar[i], zero);
    }

    Value reducedAcc = undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc = insert_element(vecTy, reducedAcc, accScalar[i], i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every warp in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the warp. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  /*
  This class helps with re-ordering mfmas into tiles to improve data-locality
  and therefore register allocation in the backend.
  The effect of tiling should be fewer registers used for A,B operands
  between the ds_reads and mfmas.
  However, this is not a robust solution since
  (1) pre-RA scheduling is bidirectional so the ds_reads can be issued
      from the top-down in a conflicting order from the mfmas' order bottom-up,
  (2) the ds_reads aren't ordered to match the tiles which impacts (1),
  (3) the pre-RA scheduler can still move the ds_reads too early
      or too late relative to their respective mfmas.
  */
  struct DotTiling {
    const int numRepM;
    const int numRepN;
    // The tile with fewer reps should be inner loop (=true) to minimize register lifetimes.
    const int preferMoreOuterTiles;
    const int tileSizeM;
    const int tileSizeN;
    const int numTilesM;
    const int numTilesN;
    const int moreTilesN;
    bool outerTileN;
    int tileSizeOuter;
    int tileSizeInner;
    int numTilesOuter;
    int numTilesInner;

    explicit DotTiling(int inputNumRepM, int inputNumRepN,
                       bool inputPreferMoreOuterTiles = true)
        : numRepM(inputNumRepM),
          numRepN(inputNumRepN),
          preferMoreOuterTiles(inputPreferMoreOuterTiles),
#if 1
          // 2x2 tile
          tileSizeM((numRepM % 2 == 0) ? 2 : 1),
          tileSizeN((numRepN % 2 == 0) ? 2 : 1),
#else
          // half x half
          tileSizeM((numRepM % 2 == 0 && numRepM >= 2) ? numRepM/2 : 1),
          tileSizeN((numRepN % 2 == 0 && numRepN >= 2) ? numRepN/2 : 1),
#endif
          numTilesM(numRepM / tileSizeM),
          numTilesN(numRepN / tileSizeN),
          moreTilesN(numTilesN > numTilesM),
          outerTileN(preferMoreOuterTiles == moreTilesN) {
      // Num mfmas must evenly divide into tiles.
      assert(numTilesM * tileSizeM == numRepM);
      assert(numTilesN * tileSizeN == numRepN);
      // Assign M and N to be outer vs inner tile loop.
      if (outerTileN) {
        // N is tile of outer loop.
        tileSizeOuter = tileSizeN;
        tileSizeInner = tileSizeM;
        numTilesOuter = numTilesN;
        numTilesInner = numTilesM;
      } else {
        // M is tile of outer loop.
        tileSizeOuter = tileSizeM;
        tileSizeInner = tileSizeN;
        numTilesOuter = numTilesM;
        numTilesInner = numTilesN;
      }
    }
    int getTileSizeM() const { return tileSizeM; }
    int getTileSizeN() const { return tileSizeN; }
    int getNumTilesOuter() const { return numTilesOuter; }
    int getNumTilesInner() const { return numTilesInner; }
    int getTileStartM(int tileOuterIdx, int tileInnerIdx) const {
      if (outerTileN) {
        return tileInnerIdx * tileSizeInner; // M is inner tile loop.
      } else {
        return tileOuterIdx * tileSizeOuter; // M is outer tile loop.
      }
    }
    int getTileStartN(int tileOuterIdx, int tileInnerIdx) const {
      if (outerTileN) {
        return tileOuterIdx * tileSizeOuter; // N is outer tile loop.
      } else {
        return tileInnerIdx * tileSizeInner; // N is inner tile loop.
      }
    }
  }; // DotTiling

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = cast<RankedTensorType>(a.getType());
    auto bTensorTy = cast<RankedTensorType>(b.getType());
    auto dTensorTy = cast<RankedTensorType>(d.getType());
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    StringRef mfmaInsnName;
    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    mfmaInsnName = maybeMfmaInsn->getInsnName();
    unsigned kBase = maybeMfmaInsn->getKBase();

    auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
    int kWidth = aEncoding.getKWidth();
    auto rank = aTensorTy.getShape().size();

    auto repA = mfmaLayout.getRepForOperand(aTensorTy.getShape(), kWidth, 0);
    auto repB = mfmaLayout.getRepForOperand(bTensorTy.getShape(), kWidth, 1);

    assert(repA[2] == repB[1]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[1];
    auto numRepN = repB[2];
    auto numRepK = repA[2];
    auto numRepB = repA[0];
    assert(repA[0] == repB[0]);

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepB, numRepM, numRepK, kWidth, kBase,
        aTensorTy.getElementType());
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepB, numRepN, numRepK, kWidth, kBase,
        aTensorTy.getElementType());

    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);

    const DotTiling dotTiling(numRepM, numRepN);
    const int tileSizeM = dotTiling.getTileSizeM();
    const int tileSizeN = dotTiling.getTileSizeN();
    for (int k = 0; k < numRepK; k++) {
      for (int tileOuterIdx = 0; tileOuterIdx < dotTiling.getNumTilesOuter(); ++tileOuterIdx) {
        for (int tileInnerIdx = 0; tileInnerIdx < dotTiling.getNumTilesInner(); ++tileInnerIdx) {
          const int tileStartM = dotTiling.getTileStartM(tileOuterIdx, tileInnerIdx);
          const int tileStartN = dotTiling.getTileStartN(tileOuterIdx, tileInnerIdx);

          for (int b = 0; b < numRepB; ++b) {
            for (int m = tileStartM; m < tileStartM + tileSizeM; ++m) {
              for (int n = tileStartN; n < tileStartN + tileSizeN; ++n) {
                Value acc = undef(vecTy);
                for (unsigned v = 0; v < elemsPerVec; ++v) {
                  acc = insert_element(
                      vecTy, acc,
                      fc[b * numRepM * numRepN * elemsPerVec +
                        m * numRepN * elemsPerVec + n * elemsPerVec + v],
                      i32_val(v));
                }
                acc = zeroAuxiliarBlocks(subBlocks, acc);
                for (int kPack = 0; kPack < kWidth / kBase; ++kPack) {
                  acc =
                      mfmaLayout.getIsTransposed()
                          ? generateMFMAOp(mfmaInsnName, operandB[kPack][{b, n, k}],
                                           operandA[kPack][{b, m, k}], acc)
                          : generateMFMAOp(mfmaInsnName, operandA[kPack][{b, m, k}],
                                           operandB[kPack][{b, n, k}], acc);
                  if (triton::tools::getBoolEnv("TRITON_MFMA_TILE_ENABLE_SCHED_BARRIERS")) {
                    // Use sched.barrier to enforce the ordering of mfmas.
                    // However, there is a bug in llvm's sched.barrier solver,
                    // so these barriers may lead to incorrect code; requires
                    // https://github.com/llvm/llvm-project/pull/112781 to fix.
                    // Mask every instruction type except for MFMA to only
                    // enforce the relative order of mfmas.
                    int32_t mfmaMask = InstructionKindMask::VALU
                        | InstructionKindMask::SALU
                        | InstructionKindMask::ALL_VMEM
                        | InstructionKindMask::VMEM_READ
                        | InstructionKindMask::VMEM_WRITE
                        | InstructionKindMask::ALL_DS
                        | InstructionKindMask::DS_READ
                        | InstructionKindMask::DS_WRITE
                        | InstructionKindMask::TRANSCEND;
                    createSchedBarrier(rewriter, loc, mfmaMask);
                  }
                } // kPack
                acc = reduceSubBlocks(subBlocks, acc);
                for (unsigned v = 0; v < elemsPerVec; ++v) {
                  fc[b * numRepM * numRepN * elemsPerVec + m * numRepN * elemsPerVec +
                    n * elemsPerVec + v] =
                      extract_element(dstElemTy, acc, i32_val(v));
                }
              } // n
            } // m
          } // b
        } // inner tile
      } // outer tile
    } // k

    /*
    Sched.Group.Barriers
    This is temporary code to avoid global_loads and ds_write latencies from being significant issues,
    so that the ds_read and mfmas can be more clearly analyzed. These work for TN 256x256x64 and nW=4, 8.
    */
    int group_id = 0;
    int nW = 8; // num waves / workgroup.
    if (false) {
      // global_load early.
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::VMEM_READ, 4*(8/nW), group_id);
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA,     96*(8/nW), group_id);
    }
    // ds_writes late and interleaved.
    if (false) {
      group_id++;
      createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA,     112*(8/nW), group_id);
      for (int w = 0; w < 8*(8/nW); ++w) {
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::MFMA,     1, group_id);
        createSchedGroupBarrier(rewriter, loc, InstructionKindMask::DS_WRITE, 1, group_id);
      }
    }

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

    Type elemtTy = elemTyA;
    const size_t mmaCount =
        numRepB * numRepM * numRepN * numRepK * kWidth / kBase;
    setNumGeneratedMMAs(op, mmaCount, maybeMfmaInsn->getMDim(),
                        maybeMfmaInsn->getNDim(), maybeMfmaInsn->getKDim(),
                        elemtTy);

    rewriter.replaceOp(op, res);

    return success();
  }

  /**
   * @brief extract vector from rawElems based on kWidth and kBase
   * rawElems is a vector of kWidth elements. We need to prepare vector(s) of
   * kBase elements for each mfma instruction
   */
  SmallVector<Value> extractOperands(Value rawElems, int kWidth, int kBase,
                                     Type type) const {
    int kpack = kWidth / kBase;
    SmallVector<Value> results;
    auto vecTy = vec_ty(type, kBase);
    if (type.isBF16())
      vecTy = vec_ty(i16_ty, kBase);
    for (int k = 0; k < kpack; ++k) {
      Value vec = undef(vecTy);
      for (int elemId = 0; elemId < kBase; ++elemId) {
        auto val = extract_element(type, rawElems, i32_val(elemId + k * kBase));
        if (type.isBF16()) {
          // rocdl.mfma.f32.32x32x8bf16.1k calls for input of i16 type
          auto cast = bitcast(val, i16_ty);
          vec = insert_element(vecTy, vec, cast, i32_val(elemId));
        } else
          vec = insert_element(vecTy, vec, val, i32_val(elemId));
      }
      if (type.getIntOrFloatBitWidth() == 8) {
        if (4 == kBase)
          // This is for int8 on pre- MI300 GPUs
          results.push_back(bitcast(vec, i32_ty));
        if (8 == kBase)
          results.push_back(bitcast(vec, i64_ty));
      } else
        results.push_back(vec);
    }
    return results;
  }

  /**
   * @brief Converts dot operand structure to value table and converts types
   * appropriate for mfma instructions
   */
  SmallVector<ValueTable>
  getValuesFromDotOperandLayoutStruct(Value value, int batch, int n0, int n1,
                                      int kWidth, int kBase, Type type) const {
    auto elems = unpackLLElements(loc, value, rewriter);
    int kpack = kWidth / kBase;
    SmallVector<ValueTable> dotOpVals(kpack);
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < n0; i++) {
        for (int j = 0; j < n1; j++) {
          Type elemTy = typeConverter->convertType(type);
          Type ty = vec_ty(elemTy, kWidth);
          Value rawElems = undef(ty);
          for (int k = 0; k < kWidth; ++k) {
            rawElems = insert_element(
                ty, rawElems,
                elems[kWidth * n1 * n0 * b + kWidth * n1 * i + kWidth * j + k],
                i32_val(k));
          }

          Value convertedElems;
          if (type.isF32()) {
            for (int k = 0; k < kpack; ++k)
              dotOpVals[k][{b, i, j}] =
                  extract_element(type, rawElems, i32_val(k));
          } else {
            SmallVector<Value> vals;
            if (type.getIntOrFloatBitWidth() == 8) {
              vals = extractOperands(rawElems, kWidth, kBase, i8_ty);
            } else if (type.isBF16()) {
              vals = extractOperands(rawElems, kWidth, kBase, bf16_ty);
            } else {
              assert(type.isF16() && "Unsupported data type");
              vals = extractOperands(rawElems, kWidth, kBase, f16_ty);
            }
            for (int k = 0; k < kpack; ++k) {
              dotOpVals[k][{b, i, j}] = vals[k];
            }
          }
        }
      }
    }
    return dotOpVals;
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDMfmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}
} // namespace mlir::triton::AMD
