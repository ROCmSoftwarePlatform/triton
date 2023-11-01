// RUN: triton-opt %s -split-input-file --tritongpu-remove-layout-conversions -canonicalize | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: remove_layout_multiple_outputs
  tt.func public @remove_layout_multiple_outputs(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %second_reduce_input = arith.constant dense<9223372036854775807> : tensor<256x256xi64, #blocked>
    %load_mask = arith.constant dense<1>: tensor<1x256xi1, #blocked>
    %store_mask = arith.constant dense<1>: tensor<256xi1, #blocked1>
    %default_load_val = arith.constant dense<0.000000e+00> : tensor<256x256xf16, #blocked>
    %70 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<256x256x!tt.ptr<f16, 1>, #blocked>
    %76 = tt.broadcast %load_mask : (tensor<1x256xi1, #blocked>) -> tensor<256x256xi1, #blocked>
    %87 = tt.load %70, %76, %default_load_val {cache = 1 : i32, evict = 2 : i32, isVolatile = false} : tensor<256x256xf16, #blocked>
    %88 = triton_gpu.convert_layout %87 : (tensor<256x256xf16, #blocked>) -> tensor<256x256xf16, #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>>
    %89 = arith.extf %87 : tensor<256x256xf16, #blocked> to tensor<256x256xf32, #blocked>
    %108:2 = "tt.reduce"(%89, %second_reduce_input) <{axis = 1 : i32}> ({
    ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: i64):
      tt.reduce.return %arg7, %arg6 : f32, i64
    }) : (tensor<256x256xf32, #blocked>, tensor<256x256xi64, #blocked>) -> (tensor<256xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>)
    %111 = tt.splat %arg1 : (!tt.ptr<i64, 1>) -> tensor<256x!tt.ptr<i64, 1>, #blocked1>
    %110 = triton_gpu.convert_layout %108#1 : (tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> (tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>)
    %112 = tt.view %110 : (tensor<256xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<256xi64, #blocked1>
    tt.store %111, %112, %store_mask {cache = 1 : i32, evict = 1 : i32} : tensor<256xi64, #blocked1>
    tt.return
  }
}
