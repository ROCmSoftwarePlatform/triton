#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mfma = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [4, 64], isTransposed = true}>
#mfma1 = #triton_gpu.mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [4, 4], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_fwd_kernel_splitK_0d1d2d34d5d6de7de8de9de10de11c12de13de14de15de16c17de18de19de20de21c22de23de24de25c26de27de28de29c30e31de32de33de(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg21: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg22: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg23: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg24: i32 {tt.max_divisibility = 8 : i32}, %arg25: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg26: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg27: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    %c64_i64 = arith.constant 64 : i64
    %cst_1 = arith.constant dense<0> : tensor<16x1xi64, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<16x1xi64, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_3 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #mfma>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #mfma1>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %arg27 : i32
    %4 = arith.addi %2, %c1_i32 : i32
    %5 = arith.muli %4, %arg27 : i32
    %6 = arith.minsi %5, %arg26 : i32
    %7 = tt.addptr %arg0, %c0_i32 : !tt.ptr<f16, 1>, i32
    %8 = arith.muli %1, %arg6 : i32
    %9 = tt.addptr %7, %8 : !tt.ptr<f16, 1>, i32
    %10 = tt.addptr %9, %c0_i32 : !tt.ptr<f16, 1>, i32
    %11 = arith.muli %0, %c16_i32 : i32
    %12 = arith.extsi %arg25 : i32 to i64
    %13 = arith.extsi %arg7 : i32 to i64
    %14 = arith.extsi %11 : i32 to i64
    %15 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f16, 1>, i32
    %16 = arith.muli %1, %arg10 : i32
    %17 = tt.addptr %15, %16 : !tt.ptr<f16, 1>, i32
    %18 = tt.addptr %17, %c0_i32 : !tt.ptr<f16, 1>, i32
    %19 = tt.addptr %18, %c0_i32 : !tt.ptr<f16, 1>, i32
    %20 = arith.extsi %arg11 : i32 to i64
    %21 = arith.extsi %3 : i32 to i64
    %22 = tt.addptr %arg2, %c0_i32 : !tt.ptr<f16, 1>, i32
    %23 = arith.muli %1, %arg14 : i32
    %24 = tt.addptr %22, %23 : !tt.ptr<f16, 1>, i32
    %25 = tt.addptr %24, %c0_i32 : !tt.ptr<f16, 1>, i32
    %26 = tt.addptr %25, %c0_i32 : !tt.ptr<f16, 1>, i32
    %27 = arith.extsi %arg15 : i32 to i64
    %28 = arith.mulf %arg3, %cst_3 : f32
    %29 = tt.splat %14 : (i64) -> tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %30 = tt.splat %14 : (i64) -> tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %31 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %33 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked2>
    %34 = arith.extsi %31 : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %35 = arith.extsi %32 : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %36 = arith.addi %29, %34 : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %37 = arith.addi %30, %35 : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %38 = tt.expand_dims %36 {axis = 1 : i32} : (tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xi64, #blocked1>
    %39 = tt.expand_dims %37 {axis = 1 : i32} : (tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16x1xi64, #blocked>
    %40 = tt.splat %13 : (i64) -> tensor<16x1xi64, #blocked1>
    %41 = arith.muli %38, %40 : tensor<16x1xi64, #blocked1>
    %42 = tt.splat %10 : (!tt.ptr<f16, 1>) -> tensor<16x1x!tt.ptr<f16, 1>, #blocked1>
    %43 = tt.addptr %42, %41 : tensor<16x1x!tt.ptr<f16, 1>, #blocked1>, tensor<16x1xi64, #blocked1>
    %44 = tt.broadcast %43 : (tensor<16x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<16x128x!tt.ptr<f16, 1>, #blocked1>
    %45 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %46 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %47 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %48 = arith.extsi %45 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %49 = arith.extsi %46 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %50 = arith.extsi %47 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %51 = tt.expand_dims %48 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi64, #blocked1>
    %52 = tt.expand_dims %49 {axis = 0 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi64, #blocked>
    %53 = tt.broadcast %51 : (tensor<1x128xi64, #blocked1>) -> tensor<16x128xi64, #blocked1>
    %54 = tt.broadcast %52 : (tensor<1x128xi64, #blocked>) -> tensor<16x128xi64, #blocked>
    %55 = tt.addptr %44, %53 : tensor<16x128x!tt.ptr<f16, 1>, #blocked1>, tensor<16x128xi64, #blocked1>
    %56 = arith.cmpi sge, %38, %cst_2 : tensor<16x1xi64, #blocked1>
    %57 = arith.cmpi sge, %39, %cst_1 : tensor<16x1xi64, #blocked>
    %58 = tt.splat %12 : (i64) -> tensor<16x1xi64, #blocked1>
    %59 = tt.splat %12 : (i64) -> tensor<16x1xi64, #blocked>
    %60 = arith.cmpi slt, %38, %58 : tensor<16x1xi64, #blocked1>
    %61 = arith.cmpi slt, %39, %59 : tensor<16x1xi64, #blocked>
    %62 = arith.andi %56, %60 : tensor<16x1xi1, #blocked1>
    %63 = arith.andi %57, %61 : tensor<16x1xi1, #blocked>
    %64 = tt.broadcast %62 : (tensor<16x1xi1, #blocked1>) -> tensor<16x128xi1, #blocked1>
    %65 = tt.broadcast %63 : (tensor<16x1xi1, #blocked>) -> tensor<16x128xi1, #blocked>
    %66 = tt.load %55, %64 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x128xf16, #blocked1>
    %67 = tt.splat %28 : (f32) -> tensor<16x128xf32, #blocked1>
    %68 = arith.extf %66 : tensor<16x128xf16, #blocked1> to tensor<16x128xf32, #blocked1>
    %69 = arith.mulf %68, %67 : tensor<16x128xf32, #blocked1>
    %70 = arith.truncf %69 : tensor<16x128xf32, #blocked1> to tensor<16x128xf16, #blocked1>
    %71 = triton_gpu.convert_layout %70 : (tensor<16x128xf16, #blocked1>) -> tensor<16x128xf16, #shared>
    %72 = triton_gpu.convert_layout %71 : (tensor<16x128xf16, #shared>) -> tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
    %73 = tt.expand_dims %50 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi64, #blocked3>
    %74 = tt.splat %19 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked3>
    %75 = tt.addptr %74, %73 : tensor<128x1x!tt.ptr<f16, 1>, #blocked3>, tensor<128x1xi64, #blocked3>
    %76 = tt.broadcast %75 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked3>) -> tensor<128x64x!tt.ptr<f16, 1>, #blocked3>
    %77 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %78 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %79 = arith.extsi %77 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %80 = arith.extsi %78 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %81 = tt.splat %20 : (i64) -> tensor<1x64xi64, #blocked3>
    %82 = tt.splat %27 : (i64) -> tensor<64x1xi64, #blocked1>
    %83 = tt.splat %26 : (!tt.ptr<f16, 1>) -> tensor<64x1x!tt.ptr<f16, 1>, #blocked1>
    %84 = tt.broadcast %51 : (tensor<1x128xi64, #blocked1>) -> tensor<64x128xi64, #blocked1>
    %85:6 = scf.for %arg28 = %3 to %6 step %c64_i32 iter_args(%arg29 = %cst_0, %arg30 = %cst, %arg31 = %cst_5, %arg32 = %21, %arg33 = %21, %arg34 = %cst) -> (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<16x128xf32, #mfma1>, i64, i64, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>)  : i32 {
      %109 = tt.splat %arg32 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
      %110 = arith.addi %109, %79 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
      %111 = tt.expand_dims %110 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x64xi64, #blocked3>
      %112 = arith.muli %111, %81 : tensor<1x64xi64, #blocked3>
      %113 = tt.broadcast %112 : (tensor<1x64xi64, #blocked3>) -> tensor<128x64xi64, #blocked3>
      %114 = tt.addptr %76, %113 : tensor<128x64x!tt.ptr<f16, 1>, #blocked3>, tensor<128x64xi64, #blocked3>
      %115 = tt.load %114 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked3>
      %116 = triton_gpu.convert_layout %115 : (tensor<128x64xf16, #blocked3>) -> tensor<128x64xf16, #shared1>
      %117 = triton_gpu.convert_layout %116 : (tensor<128x64xf16, #shared1>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %118 = tt.splat %arg33 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %119 = arith.addi %118, %80 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %120 = tt.expand_dims %119 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi64, #blocked1>
      %121 = arith.muli %120, %82 : tensor<64x1xi64, #blocked1>
      %122 = tt.addptr %83, %121 : tensor<64x1x!tt.ptr<f16, 1>, #blocked1>, tensor<64x1xi64, #blocked1>
      %123 = tt.broadcast %122 : (tensor<64x1x!tt.ptr<f16, 1>, #blocked1>) -> tensor<64x128x!tt.ptr<f16, 1>, #blocked1>
      %124 = tt.addptr %123, %84 : tensor<64x128x!tt.ptr<f16, 1>, #blocked1>, tensor<64x128xi64, #blocked1>
      %125 = tt.load %124 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128xf16, #blocked1>
      %126 = triton_gpu.convert_layout %125 : (tensor<64x128xf16, #blocked1>) -> tensor<64x128xf16, #shared2>
      %127 = triton_gpu.convert_layout %126 : (tensor<64x128xf16, #shared2>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma1, kWidth = 4}>>
      %128 = tt.dot %72, %117, %cst_4 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<16x64xf32, #mfma>
      %129 = "tt.reduce"(%128) <{axis = 1 : i32}> ({
      ^bb0(%arg35: f32, %arg36: f32):
        %152 = arith.maximumf %arg35, %arg36 : f32
        tt.reduce.return %152 : f32
      }) : (tensor<16x64xf32, #mfma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %130 = arith.maximumf %arg34, %129 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %131 = arith.maximumf %arg30, %129 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %132 = arith.subf %arg34, %130 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %133 = arith.subf %arg30, %131 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %134 = tt.extern_elementwise %132 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %135 = tt.extern_elementwise %133 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %136 = tt.expand_dims %131 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<16x1xf32, #mfma>
      %137 = tt.broadcast %136 : (tensor<16x1xf32, #mfma>) -> tensor<16x64xf32, #mfma>
      %138 = arith.subf %128, %137 : tensor<16x64xf32, #mfma>
      %139 = tt.extern_elementwise %138 {libname = "libdevice", libpath = "/triton/python/triton/language/../third_party/hip/lib/bitcode/cuda2gcn.bc", pure = true, symbol = "__nv_exp2f"} : (tensor<16x64xf32, #mfma>) -> tensor<16x64xf32, #mfma>
      %140 = arith.mulf %arg29, %135 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %141 = "tt.reduce"(%139) <{axis = 1 : i32}> ({
      ^bb0(%arg35: f32, %arg36: f32):
        %152 = arith.addf %arg35, %arg36 : f32
        tt.reduce.return %152 : f32
      }) : (tensor<16x64xf32, #mfma>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %142 = arith.addf %140, %141 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
      %143 = arith.truncf %139 : tensor<16x64xf32, #mfma> to tensor<16x64xf16, #mfma>
      %144 = triton_gpu.convert_layout %143 : (tensor<16x64xf16, #mfma>) -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma1, kWidth = 4}>>
      %145 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<16x1xf32, #mfma>
      %146 = triton_gpu.convert_layout %145 : (tensor<16x1xf32, #mfma>) -> tensor<16x1xf32, #mfma1>
      %147 = tt.broadcast %146 : (tensor<16x1xf32, #mfma1>) -> tensor<16x128xf32, #mfma1>
      %148 = arith.mulf %arg31, %147 : tensor<16x128xf32, #mfma1>
      %149 = tt.dot %144, %127, %148 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma1, kWidth = 4}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma1, kWidth = 4}>> -> tensor<16x128xf32, #mfma1>
      %150 = arith.addi %arg32, %c64_i64 : i64
      %151 = arith.addi %arg33, %c64_i64 : i64
      scf.yield %142, %131, %149, %150, %151, %130 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>, tensor<16x128xf32, #mfma1>, i64, i64, tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>
    }
    %86 = arith.muli %1, %arg18 : i32
    %87 = tt.addptr %arg4, %86 : !tt.ptr<f32, 1>, i32
    %88 = arith.muli %2, %arg19 : i32
    %89 = tt.addptr %87, %88 : !tt.ptr<f32, 1>, i32
    %90 = arith.extsi %arg20 : i32 to i64
    %91 = tt.splat %90 : (i64) -> tensor<16x1xi64, #blocked>
    %92 = arith.muli %39, %91 : tensor<16x1xi64, #blocked>
    %93 = tt.splat %89 : (!tt.ptr<f32, 1>) -> tensor<16x1x!tt.ptr<f32, 1>, #blocked>
    %94 = tt.addptr %93, %92 : tensor<16x1x!tt.ptr<f32, 1>, #blocked>, tensor<16x1xi64, #blocked>
    %95 = tt.broadcast %94 : (tensor<16x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<16x128x!tt.ptr<f32, 1>, #blocked>
    %96 = tt.addptr %95, %54 : tensor<16x128x!tt.ptr<f32, 1>, #blocked>, tensor<16x128xi64, #blocked>
    %97 = triton_gpu.convert_layout %85#2 : (tensor<16x128xf32, #mfma1>) -> tensor<16x128xf32, #blocked>
    tt.store %96, %97, %65 {cache = 1 : i32, evict = 1 : i32} : tensor<16x128xf32, #blocked>
    %98 = arith.muli %1, %arg21 : i32
    %99 = tt.addptr %arg5, %98 : !tt.ptr<f32, 1>, i32
    %100 = arith.muli %2, %arg23 : i32
    %101 = tt.addptr %99, %100 : !tt.ptr<f32, 1>, i32
    %102 = tt.addptr %101, %11 : !tt.ptr<f32, 1>, i32
    %103 = tt.splat %102 : (!tt.ptr<f32, 1>) -> tensor<16x!tt.ptr<f32, 1>, #blocked2>
    %104 = tt.addptr %103, %33 : tensor<16x!tt.ptr<f32, 1>, #blocked2>, tensor<16xi32, #blocked2>
    %105 = triton_gpu.convert_layout %85#1 : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<16xf32, #blocked2>
    tt.store %104, %105 {cache = 1 : i32, evict = 1 : i32} : tensor<16xf32, #blocked2>
    %106 = tt.splat %arg22 : (i32) -> tensor<16xi32, #blocked2>
    %107 = tt.addptr %104, %106 : tensor<16x!tt.ptr<f32, 1>, #blocked2>, tensor<16xi32, #blocked2>
    %108 = triton_gpu.convert_layout %85#0 : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #mfma}>>) -> tensor<16xf32, #blocked2>
    tt.store %107, %108 {cache = 1 : i32, evict = 1 : i32} : tensor<16xf32, #blocked2>
    tt.return
  }
}
