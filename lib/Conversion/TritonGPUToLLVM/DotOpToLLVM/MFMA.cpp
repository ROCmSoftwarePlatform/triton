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
#ifdef USE_ROCM

#include "../DotOpToLLVM.h"
#include "../Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MfmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

// mapping from touple <kpack-rep, non-k-rep, k-rep> to vector of values
// vector contains single element for MFMA32, MFMA16 and MFMA4 layouts
// for MFMA 4x64 and 64x4 layouts there are 16 vectors for one of the arguments,
// because each repetition in these layouts requires 16 mfma operations
using ValueTable = std::map<std::tuple<unsigned, unsigned, unsigned>,
                            llvm::SmallVector<Value>>;

struct DotOpMFMAConversionHelper {
  MfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(
      MfmaEncodingAttr mfmaLayout, ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  /**
   * @param mfmaInsnName
   * @param valA
   * @param valB
   * @param valC
   * @param cbsz Control Broadcast Size modifier
   * @param abid A-matrix Broadcast Identifier
   * @param blgp B-matrix Lane Group Pattern modifier
   */
  Value generateMFMAOp(StringRef mfmaInsnName, Value valA, Value valB,
                       Value valC, int cbsz = 0, int abid = 0,
                       int blgp = 0) const {
    assert(cbsz >= 0 && cbsz <= 4);
    assert(abid >= 0 && abid <= 15);
    assert(blgp >= 0 && blgp <= 7);
    auto resType = valC.getType();
    Value zeroVal = i32_val(0);
    Value cbszFlag = cbsz != 0 ? i32_val(cbsz) : zeroVal;
    Value abidFlag = abid != 0 ? i32_val(abid) : zeroVal;
    Value blgpFlag = blgp != 0 ? i32_val(blgp) : zeroVal;
    OperationState loweredOp(loc, mfmaInsnName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, cbszFlag, abidFlag, blgpFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  Value broadcastGroup(Value val, int groupId, int numGroups) const {
    constexpr int waveSize = 64;
    const int groupSize = waveSize / numGroups;

    Value lane = getThreadId();
    // Multiply by 4, because permute requires offset in bytes
    Value laneOffset = mul(urem(lane, i32_val(groupSize)), i32_val(4));
    Value permuteAddr = add(laneOffset, i32_val(groupId * groupSize * 4));
    Type valType = val.getType();
    Value broadcasted;
    if (valType.isInteger(32))
      broadcasted = rewriter.create<ROCDL::DsBpermuteOp>(loc, val.getType(),
                                                         permuteAddr, val);
    if (valType.isF32()) {
      val = bitcast(val, i32_ty);
      broadcasted = rewriter.create<ROCDL::DsBpermuteOp>(loc, val.getType(),
                                                         permuteAddr, val);
      broadcasted = bitcast(broadcasted, f32_ty);
    }
    if (valType.isa<VectorType>()) {
      auto vecTy = valType.dyn_cast<VectorType>();
      auto vecBitSize = vecTy.getElementType().getIntOrFloatBitWidth() *
                        vecTy.getNumElements();
      const int int32VecSize = vecBitSize / 32;

      Type int32VecTy = vec_ty(i32_ty, int32VecSize);
      Value int32Val = bitcast(val, int32VecTy);
      Value int32Broadcasted = undef(int32VecTy);
      for (int i = 0; i < int32VecSize; ++i) {
        Value int32Chunk = extract_element(i32_ty, int32Val, i32_val(i));
        Value broadcastedChunk = rewriter.create<ROCDL::DsBpermuteOp>(
            loc, i32_ty, permuteAddr, int32Chunk);
        int32Broadcasted = insert_element(int32VecTy, int32Broadcasted,
                                          broadcastedChunk, i32_val(i));
      }
      broadcasted = bitcast(int32Broadcasted, valType);
    }
    assert(broadcasted);
    return broadcasted;
  }

  Value generateMFMATile(StringRef mfmaInsnName, SmallVector<Value> valA,
                         SmallVector<Value> valB, Value valC, int mDim,
                         int nDim, bool transpose) const {

    Value acc;
    if (mDim == nDim) {
      assert(valA.size() == 1 && valB.size() == 1);
      acc = transpose ? generateMFMAOp(mfmaInsnName, valB[0], valA[0], valC)
                      : generateMFMAOp(mfmaInsnName, valA[0], valB[0], valC);
    }
    if (mDim == 4 && nDim == 64 || mDim == 64 && nDim == 4) {
      // broadcast selected kRep A operand matrix to all A matrices(2^4=16)
      constexpr int broadcastCtrl = 4;
      constexpr int numRepeats = 16;
      acc = valC;
      for (int kRep = 0; kRep < numRepeats; kRep++) {
        if (mDim == 4 && !transpose) {
          assert(valA.size() == 1 && valB.size() == 16);
          acc = generateMFMAOp(mfmaInsnName, valA[0], valB[kRep], acc,
                               broadcastCtrl, kRep);
        }
        if (mDim == 4 && transpose) {
          assert(valA.size() == 1 && valB.size() == 16);
          Value broadcastValA = broadcastGroup(valA[0], kRep, numRepeats);
          acc = generateMFMAOp(mfmaInsnName, valB[kRep], broadcastValA, acc);
        }
        if (nDim == 4 && !transpose) {
          assert(valA.size() == 16 && valB.size() == 1);
          Value broadcastValB = broadcastGroup(valB[0], kRep, numRepeats);
          acc = generateMFMAOp(mfmaInsnName, valA[kRep], broadcastValB, acc);
        }
        if (nDim == 4 && transpose) {
          assert(valA.size() == 16 && valB.size() == 1);
          acc = generateMFMAOp(mfmaInsnName, valB[0], valA[kRep], acc,
                               broadcastCtrl, kRep);
        }
      }
    }
    return acc;
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if (mDim == 64 && nDim == 4 || mDim == 4 && nDim == 64)
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
    constexpr int waveSize = 64;
    int subBlockSize = waveSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = and_(laneId, i32_val(waveSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < waveSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              mlir::LLVM::shflSync(loc, rewriter, accScalar[i], subBlockSize);
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

  /// @brief MFMA 4x4 is computes 16 matrix mupliplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every wave in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the wave. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

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
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    StringRef mfmaInsnName;
    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");

    mfmaInsnName = (*maybeMfmaInsn).getInsnName();
    unsigned kBaseA = (*maybeMfmaInsn).getKBaseA();
    unsigned kBaseB = (*maybeMfmaInsn).getKBaseB();

    auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();

    auto kWidthA = aEncoding.getKWidth();
    auto kWidthB = bEncoding.getKWidth();

    auto repA = aEncoding.getMFMARep(aTensorTy.getShape());
    auto repB = bEncoding.getMFMARep(bTensorTy.getShape());

    assert(repA[1] == repB[0]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[0];
    auto numRepN = repB[1];
    auto numRepK = repA[1];

    auto operandA = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepM, numRepK, kWidthA, kBaseA, aTensorTy.getElementType());
    auto operandB = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepN, numRepK, kWidthB, kBaseB, aTensorTy.getElementType());

    auto dstElemTy = dTensorTy.getElementType();
    auto fc =
        typeConverter->unpackLLElements(loc, loadedC, rewriter, dstElemTy);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          acc = insert_element(
              vecTy, acc, fc[m * numRepN * elemsPerVec + n * elemsPerVec + v],
              i32_val(v));
        }

        acc = zeroAuxiliarBlocks(subBlocks, acc);
        for (size_t k = 0; k < numRepK; k++)
          for (int kpack = 0; kpack < kWidthA / kBaseA; ++kpack)
            acc = generateMFMATile(mfmaInsnName, operandA[{kpack, m, k}],
                                   operandB[{kpack, n, k}], acc, mDim, nDim,
                                   mfmaLayout.getIsTransposed());
        acc = reduceSubBlocks(subBlocks, acc);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          fc[m * numRepN * elemsPerVec + n * elemsPerVec + v] =
              extract_element(dstElemTy, acc, i32_val(v));
        }
      }
    }

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

  /**
   * @brief extract vector from rawElems based on kWidth and kBase
   * rawElems is a vector of kWidth elements. We need to prepare vector(s) of
   * kBase elements for each mfma instruction
   *
   * @param rawElems vector of "raw" elements for one mfma tile
   * @param k id in k-pack
   * @param kPack size of k-pack
   * @param numIntrinsics number of operands we need to extract
   * @param type type mfma intrinsic requires
   *
   * @return elements converted for one repetition
   */
  SmallVector<Value> extractOperands(Value rawElems, int k, int kPack,
                                     int numIntrinsics, Type type) const {
    assert(numIntrinsics == 1 || numIntrinsics == 16);
    auto rawTy = rawElems.getType().cast<VectorType>();
    auto rawElemTy = rawTy.getElementType();
    // number of elements required by one mfma intrinsic
    int intrinsicK = rawTy.getNumElements() / numIntrinsics / kPack;
    int kBase = rawTy.getNumElements() / kPack;

    SmallVector<Value> results;
    // extract needed elements in original dtype
    auto typedVecTy = vec_ty(rawElemTy, intrinsicK);
    for (int intrinsic = 0; intrinsic < numIntrinsics; ++intrinsic) {
      Value typedVec = undef(typedVecTy);
      for (int elemId = 0; elemId < intrinsicK; ++elemId) {
        int elemOff = elemId + intrinsic * intrinsicK + k * kBase;
        auto val = extract_element(rawElemTy, rawElems, i32_val(elemOff));
        typedVec = insert_element(typedVecTy, typedVec, val, i32_val(elemId));
      }
      Value castedVec = bitcast(typedVec, type);
      results.push_back(castedVec);
    }
    return results;
  }

  /**
   * @brief Converts dot operand structure to value table and converts types
   * appropriate for mfma instructions
   */
  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 int kWidth, int kBase,
                                                 Type type) const {
    auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
    int kpack = kWidth / kBase;
    // "Wide operand" means that this operand is for mfma 4x64 layout
    // This operand is 64x64 for fp16, bf16 and int8 data types and
    // 16x64 for fp32
    bool wideOperand = kWidth >= 16;
    // How many rocdl intrinsics will process one tile
    int numIntrinsics = wideOperand ? 16 : 1;
    int intrinsicKWidth = wideOperand ? kBase / numIntrinsics : kBase;
    Type intrinsicDType;
    if (type.isF32())
      intrinsicDType = f32_ty;
    if (type.getIntOrFloatBitWidth() == 8)
      intrinsicDType = rewriter.getIntegerType(intrinsicKWidth * 8);
    if (type.isBF16())
      intrinsicDType = vec_ty(i16_ty, intrinsicKWidth);
    if (type.isF16())
      intrinsicDType = vec_ty(f16_ty, intrinsicKWidth);
    assert(intrinsicDType);

    ValueTable dotOpVals;
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        auto rawElems = elems[n1 * i + j];
        for (int k = 0; k < kpack; k++) {
          SmallVector<Value> vals = extractOperands(
              rawElems, k, kpack, numIntrinsics, intrinsicDType);
          assert(vals.size() == numIntrinsics);
          dotOpVals[{k, i, j}] = vals;
        }
      }
    }
    return dotOpVals;
  }
};

} // namespace

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return tensor.getType().cast<RankedTensorType>();
  };

  assert(rankedTType(op.getA()).getEncoding().isa<DotOperandEncodingAttr>() &&
         rankedTType(op.getB()).getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(cTensorTy.getEncoding().isa<MfmaEncodingAttr>() &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<MfmaEncodingAttr>();

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

#endif // ifdef USE_ROCM
