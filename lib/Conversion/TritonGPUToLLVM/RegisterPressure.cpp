#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Liveness.h"

#include <numeric>

#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {

namespace triton {

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"

// filter out type info with 
// %s/reg_cost = \(\d*\) : i32}.*$/register_cost = \1/

class TritonGPURegisterPressurePass
    : public TritonGPURegisterPressureBase<
          TritonGPURegisterPressurePass> {
public:
  TritonGPURegisterPressurePass() = default;

  int estimateTypeByteSize(mlir::Type type) {
    int numElems = 0;
    if (auto vectorTy = type.dyn_cast<VectorType>())
      return vectorTy.getNumElements() * estimateTypeByteSize(vectorTy.getElementType());
    if (auto structTy = type.dyn_cast<LLVM::LLVMStructType>()){
      int totalSize = 0;
      for (auto subTy: structTy.getBody())
        totalSize += estimateTypeByteSize(subTy);
      return totalSize;
    }
    if (auto ptrTy = type.dyn_cast<mlir::triton::PointerType>())
      return ptrTy.getAddressSpace() == 3 ? 4 : 8;
    if (auto ptrTy = type.dyn_cast<mlir::LLVM::LLVMPointerType>())
      return ptrTy.getAddressSpace() == 3 ? 4 : 8;
    return type.getIntOrFloatBitWidth() / 8;
  }

  int estimateTensorRegisterUsage(mlir::RankedTensorType type) {
    auto encoding = type.getEncoding();
    if (encoding.isa<mlir::triton::gpu::SharedEncodingAttr>())
      return 0;
    int elemsPerThread = 0;
    if (auto dotOpLayout = encoding.dyn_cast<mlir::triton::gpu::DotOperandEncodingAttr>()) {
      if (auto mfmaParent = dotOpLayout.getParent().dyn_cast<mlir::triton::gpu::MfmaEncodingAttr>()) {
        auto rep = dotOpLayout.getMFMARep(type.getShape());
        elemsPerThread = rep[0] * rep[1] * dotOpLayout.getKWidth();
      } else {
        assert(false);
      }
    } else {
      auto elemsShape = mlir::triton::gpu::getElemsPerThread(type);
      elemsPerThread = std::reduce(elemsShape.begin(), elemsShape.end(), 1, std::multiplies<unsigned>());
    }
    auto elemTypeSize = estimateTypeByteSize(type.getElementType());
    return std::max(1, elemsPerThread * elemTypeSize / 4);
  }

  int estimateScalarRegisterUsage(mlir::Type type) {
    return estimateTypeByteSize(type) / 4;
  }

  bool isTTGIR(ModuleOp m) {
    auto &ops = m.getBody()->getOperations();
    auto pred = [](const mlir::Operation &op){return isa<const mlir::triton::FuncOp>(op);};
    return std::any_of(ops.begin(), ops.end(), pred);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    bool isTTG = isTTGIR(m);

    Liveness liveness(m);

    std::map<mlir::Operation *, int> registersAlive;
    std::map<mlir::Operation *, int> registersDefined;

    int opId = 0;
    std::map<mlir::Operation *, int> opToId;
    std::map<mlir::Operation *, std::vector<int>> affectedIds;

    // assign IDs to operations
    m.walk([&](mlir::Operation *op) {
      opToId[op] = opId;
      auto idAttr = StringAttr::get(context, "opId" + std::to_string(opId));
      op->setDiscardableAttr("reg_usage.op_id", idAttr);
      opId++;
    }
    );
    // gather liveness info
    m.walk([&](mlir::Operation *op) {
      int totalRegs = 0;
      for (auto res: op->getResults()) {
        auto liveOperations = liveness.resolveLiveness(res);
        int registers = 0;
        if (auto t = res.getType().dyn_cast<RankedTensorType>())
          registers = estimateTensorRegisterUsage(t);
        if (!isTTG)
           registers = estimateScalarRegisterUsage(res.getType());
        totalRegs += registers;
        for (auto liveOp: liveOperations) {
          registersAlive[liveOp] += registers;
          int affectedOpId = opToId[liveOp];
          affectedIds[op].push_back(affectedOpId);
        }
        if (isa<triton::DotOp>(op)) {
          for (auto liveOps: liveOperations)
            llvm::errs() << liveOps;
        }

      }
      registersDefined[op] += totalRegs;
    }
    );
    // set register usage info and affected lists
    for (auto item: registersAlive) {
      auto op = item.first;
      auto regsAliveAttr = IntegerAttr::get(mlir::IntegerType::get(context, 32), item.second);
      op->setDiscardableAttr("reg_usage.regs_alive", regsAliveAttr);
      auto regsDefinedAttr = IntegerAttr::get(mlir::IntegerType::get(context, 32), registersDefined.at(item.first));
      op->setDiscardableAttr("reg_usage.regs_defined", regsDefinedAttr);
      std::string affectedData = "";
      for (int affectedId: affectedIds[op])
        affectedData += "opId" + std::to_string(affectedId) + ";";
      auto affectedIdsAttr = StringAttr::get(context, affectedData);
      op->setDiscardableAttr("reg_usage.affecting_ops", affectedIdsAttr);
    }
  }
};

std::unique_ptr<Pass> createTritonGPURegisterPressurePass() {
  return std::make_unique<TritonGPURegisterPressurePass>();
}

} // namespace triton

} // namespace mlir
