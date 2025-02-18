#include "common.hpp"
#include "testcase.hpp"

// Insert your GEMM kernel here: using DeviceGemmInstance = ... ;
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor,
        ck::tensor_layout::gemm::RowMajor, _Float16, _Float16, _Float16, float,
        _Float16, ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::device::GemmSpecialization::Default, 256, 128,
        128, 64, 8, 8, 32, 32, 2, 2, ck::Sequence<8, 32, 1>,
        ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2, 8, 8, false,
        ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 2,
        8, 8, false, 1, 1, ck::Sequence<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Interwave,
        ck::BlockGemmPipelineVersion::v1>;

void TestCase::launchKernel(real *matA, real *matB, real *matC,
                            const TestCase::Config &config) {
  Driver<DeviceGemmInstance>::launchKernel(matA, matB, matC, config);
}
