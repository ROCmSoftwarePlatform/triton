#pragma once

#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "testcase.hpp"

using ADataType = ck::half_t;
using BDataType = ck::half_t;
using CDataType = ck::half_t;

using F16 = ck::half_t;
using F32 = float;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

template <typename DeviceGemmInstance> struct Driver {
  static void launchKernel(real *matA, real *matB, real *matC,
                           const TestCase::Config &config) {

    auto gemm = DeviceGemmInstance{};
    std::cout << gemm.GetTypeString() << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    auto invoker = gemm.MakeInvoker();
    double aveTime = 0.0f;

    size_t strideA = config.transA ? config.m : config.k;
    size_t strideB = config.transB ? config.k : config.n;
    size_t strideC = config.n;

    auto argument = gemm.MakeArgument(
        static_cast<ADataType *>(matA), static_cast<BDataType *>(matB),
        static_cast<CDataType *>(matC), config.m, config.n, config.k, strideA,
        strideB, strideC, config.kbatch, AElementOp{}, BElementOp{},
        CElementOp{});

    if (!gemm.IsSupportedArgument(argument)) {
      std::cerr << gemm.GetTypeString() << " does not support this problem"
                << std::endl;
      return;
    }

    StreamConfig streamConfig{
        /*stream_id_=*/nullptr,
        /*time_kernel_=*/true,  config.logLevel,
        config.coldNumIters,    config.numRepeat,
        config.flushCache,      config.rotatingCount,
    };

    // time in milli seconds
    aveTime = invoker.Run(argument, streamConfig);

    double flops = 2.0 * (config.m * config.n * config.k);
    flops /= (aveTime * 1e9);

    std::cout << "TFLOP/s: " << flops << "\n";
  }
};
