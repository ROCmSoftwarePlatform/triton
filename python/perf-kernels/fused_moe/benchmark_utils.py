from typing import TypedDict, List, Optional
from itertools import product
import json
import torch
import os


def get_config_dtype_str(dtype: torch.dtype, use_int8_w8a16: Optional[bool] = False,
                         use_fp8_w8a8: Optional[bool] = False):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def get_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def get_tuning_space(use_fp16):
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    if not use_fp16:
        block_k_range.remove(16)  # BLOCK_K=16 not supported for fp8
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    num_stage_range = [1, 2]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32] if use_fp16 else []
    kpack_range = [1, 2] if use_fp16 else []

    param_ranges = {
        "BLOCK_SIZE_M": block_mn_range,
        "BLOCK_SIZE_N": block_mn_range,
        "BLOCK_SIZE_K": block_k_range,
        "GROUP_SIZE_M": group_m_range,
        "num_warps": num_warps_range,
        "num_stages": num_stage_range,
        "waves_per_eu": waves_per_eu_range,
    }
    if use_fp16:
        param_ranges["matrix_instr_nonkdim"] = matrix_instr_nonkdim_range
        param_ranges["kpack"] = kpack_range

    return param_ranges


def prune_configs(M, N, K, configs, is_fp16=True):
    pruned_configs = []
    elemBytes_a = 2 if is_fp16 else 1
    elemBytes_b = 2 if is_fp16 else 1

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")

        if is_fp16:
            matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
            if matrix_instr_nonkdim > mfma:
                continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = config.get("SPLIT_K", 1)
        GROUP_M = config.get("GROUP_SIZE_M")
        if is_fp16:
            if (matrix_instr_nonkdim > BLOCK_SIZE_M or matrix_instr_nonkdim > BLOCK_SIZE_N):
                continue
            if (matrix_instr_nonkdim >= M and matrix_instr_nonkdim != BLOCK_SIZE_M):
                continue
            if (matrix_instr_nonkdim >= N and matrix_instr_nonkdim != BLOCK_SIZE_N):
                continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b)
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 128 or BLOCK_SIZE_N < 128:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def merge_unique_dicts(list1, list2):
    result = []
    combined_list = list1.copy()
    combined_list.extend(list2)
    for dictionary in combined_list:
        if dictionary not in result:
            result.append(dictionary)
    return result


def prune_search_space(num_tokens, shard_intermediate_size, hidden_size, search_space, is_fp16):
    N1, K1 = shard_intermediate_size, hidden_size

    pruned_space_1 = prune_configs(num_tokens * 2, N1, K1, search_space, is_fp16)
    # NOTE, we are only tunning thr gemm here so only one pass of moe
    # pruned_space_2 = prune_configs(num_tokens * 8, N2, K2, search_space,
    #                                is_fp16)
    # search_space = merge_unique_dicts(pruned_space_1, pruned_space_2)
    return pruned_space_1


def update_configs(M: int, config: BenchmarkConfig, num_experts: int, shard_intermediate_size: int, hidden_size: int,
                   topk: int, dtype: torch.dtype, use_fp8_w8a8: bool, use_int8_w8a16: bool) -> None:
    dtype_str = get_config_dtype_str(dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8)

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    # NOTE, we are only tunning thr gemm here so no // 2
    # filename = get_config_file_name(num_experts, shard_intermediate_size // 2,
    #                                 dtype_str)

    filename = get_config_file_name(num_experts, shard_intermediate_size, dtype_str)
    print(f"Best config: {config}")
    print(f"Writing best config to {filename}...")

    config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", filename)
    # 1) Read the existing JSON file if it exists
    old_configs = {}
    if os.path.isfile(config_file_path):
        with open(config_file_path, "r") as f:
            try:
                old_configs = json.load(f)
            except json.JSONDecodeError:
                # If the file is empty or corrupt, we just ignore it
                old_configs = {}

    # 2) Update existing data with new configs
    #    If they share any keys, the new 'configs' will overwrite
    old_configs[str(M)] = config
    # old_configs[configs.keys()[0]] = configs[configs.keys()[0]]

    # 3) Write back to the same file
    with open(config_file_path, "w") as f:
        json.dump(old_configs, f, indent=2)
        f.write("\n")


def get_tuning_configs(M, N, K, use_fp16):
    param_ranges = get_tuning_space(use_fp16)
    configs: List[BenchmarkConfig] = []

    keys, values = zip(*param_ranges.items())
    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        configs.append(config)

    configs = prune_search_space(num_tokens=M, shard_intermediate_size=N, hidden_size=K, search_space=configs,
                                 is_fp16=use_fp16)

    return configs
