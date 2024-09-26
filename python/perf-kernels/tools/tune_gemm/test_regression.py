import tune_gemm

import json
import pytest
import warnings


@pytest.mark.parametrize('config', [
    # M // BLOCK_M * N // BLOCK_N % 304 == 0
    # 1 workgroup / CU
    {
        'M': 4864, 'N': 4096, 'K': 4096, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4160, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4224, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4288, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    # 1 workgroup / CU masked loadK
    {
        'M': 4864, 'N': 4096, 'K': 4097, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4098, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4100, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4104, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 4096, 'K': 4112, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },

    # 2 workgroups / CU
    {
        'M': 4864, 'N': 8192, 'K': 4096, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 8192, 'K': 4160, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 8192, 'K': 8192, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
    {
        'M': 4864, 'N': 8192, 'K': 8256, 'rowMajorA': 'T', 'rowMajorB': 'N', 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256,
        'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1, 'num_warps': 8, 'num_stages': 0, 'waves_per_eu': 0,
        'matrix_instr_nonkdim': 16, 'kpack': 2
    },
], ids=lambda val: f"Config: {val}")
class TestRegression:

    @classmethod
    def setup_class(self):
        self.test_results = []
        try:
            with open('gemm-performance-report-reference.json', 'r') as ref_file:
                self.reference_data = json.load(ref_file)
        except FileNotFoundError:
            warnings.warn("No reference file found. There will be no regression detected!")
            self.reference_data = []

    @classmethod
    def teardown_class(self):
        with open('gemm-performance-report.json', 'w') as out_file:
            json.dump(self.test_results, out_file)

    def test_matmul_performance_regression(self, config):
        # Get GPU ids
        gpus = [0]
        jobs = 1

        # Get element type
        dtype_a = 'fp16'
        dtype_b = 'fp16'
        dtype_c = 'fp16'

        rotating_buffer_size = 0
        bias_vector = False
        icache_flush = False
        iters = 200
        benchmark = True
        init_type = 'randn'
        num_threads = 32
        skipWarmup = False
        verbose_level = 0

        M, N, K, col_a, col_b, runConfig = tune_gemm.process_item(config)

        # Before tuning starts, clear cache and previously generated kernel files
        tune_gemm.run_bash_command("rm -rf ~/.triton/cache")
        tune_gemm.run_bash_command(f"rm -rf {tune_gemm.get_filename_myKernels()}")

        # Append new configs into the tuning space
        tune_gemm.generate_matmul_kernels([runConfig])

        bias_size = M if bias_vector else 0
        minTime, bestConfig, compile_time, profile_time, post_time = tune_gemm.tune_gemm_config(
            M, N, K, col_a, col_b, dtype_a, dtype_b, dtype_c, init_type, [runConfig], benchmark, jobs, iters,
            skipWarmup=skipWarmup, num_threads=num_threads, gpus=gpus, verbose=verbose_level,
            rotating_buffer_size=rotating_buffer_size, bias_size=bias_size, icache_flush=icache_flush)

        # post processing the numbers
        perf_tflops = lambda us: 2 * M * N * K * 1e-12 / (us * 1e-6)
        tri_tflops = perf_tflops(minTime)

        # Add to global results
        self.test_results.append({'config': config, 'tflops': float(tri_tflops)})

        # Look for reference run

        reference_run = None
        for run in self.reference_data:
            if run['config'] == config:
                reference_run = run
                break

        if reference_run is not None:
            performance_ratio = tri_tflops / reference_run['tflops']
            slowdown_threshold = 0.97
            assert performance_ratio > slowdown_threshold, f'Performance regressed by {(100.0 * (1.0 - performance_ratio)):.2f}% (threshold={((1.0 - slowdown_threshold) * 100.0 ):.2f}%)'
        else:
            warnings.warn(f"No reference file found. There will be no regression detected for config = {config}")
