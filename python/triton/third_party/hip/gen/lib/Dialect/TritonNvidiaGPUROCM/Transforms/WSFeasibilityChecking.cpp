/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "triton/Dialect/TritonGPUROCM/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPUROCM/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPUROCM/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPUROCM/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPUROCM/Transforms/Passes.h.inc"

namespace mlir {

namespace ttng = triton_rocm::nvidia_gpu;

namespace {

class TritonGPUROCMWSFeasibilityCheckingPass
    : public TritonGPUROCMWSFeasibilityCheckingBase<
          TritonGPUROCMWSFeasibilityCheckingPass> {
public:
  TritonGPUROCMWSFeasibilityCheckingPass() = default;
  TritonGPUROCMWSFeasibilityCheckingPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int wsSupported = isWSSupported(mod, this->computeCapability);
    auto i32_ty = IntegerType::get(mod->getContext(), 32);
    mod->setAttr(ttng::TritonNvidiaGPUROCMDialect::getWSSupportedAttrName(),
                 IntegerAttr::get(i32_ty, llvm::APInt(32, wsSupported)));
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonNvidiaGPUROCMWSFeasibilityCheckingPass(int computeCapability) {
  return std::make_unique<TritonGPUROCMWSFeasibilityCheckingPass>(
      computeCapability);
}

} // namespace mlir
