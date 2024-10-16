import numpy as np
from statistics import mean
import argparse
import sys
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-d", type=str, default="", help='*_ui dir')
    parser.add_argument("-se", type=int, default=0, help="")
    parser.add_argument("-sm", type=int, default=0, help="")
    parser.add_argument("-sl", type=int, default=0, help="")
    parser.add_argument("-wv", type=int, default=-1, help="")

    args = parser.parse_args()
    return args


def parse_trace(code_fullname, trace_fullname):
    instr0_clk, bar1_clk, bar2_clk, bar3_clk, instr9_clk, mfma_dsRead_cnt, mfma_dsWrite_cnt, incomplete = gen_all_clk(
        code_fullname, trace_fullname)
    if incomplete:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, incomplete
    pro, loop, epi, iter_clk = gen_coarse_clk(instr0_clk, bar1_clk, bar3_clk, instr9_clk)
    bar1_lat, bar2_lat = gen_fine_clk(bar1_clk, bar2_clk, bar3_clk)

    lat1, lat2, lat_sum, idle1, idle2 = print_loop_eff(bar1_lat, bar2_lat, mfma_dsRead_cnt, mfma_dsWrite_cnt)
    return pro, loop, epi, iter_clk, lat1, lat2, lat_sum, idle1, idle2, incomplete


def print_list(myList):
    for i in range(len(myList)):
        print(myList[i])


def gen_all_clk(code_fullname, trace_fullname):
    if not os.path.isfile(trace_fullname):
        print(f"trace file not found {trace_fullname}")
        return

    marker_to_line = dict()
    marker_to_line['firstInstr'] = 2
    marker_barrier = list()

    ## Read code.json to get instruction idx
    with open(code_fullname) as code_f:
        code_data = json.load(code_f)
        code_list = code_data['code']

        found_1st_barrier = False
        mfma_dsRead_cnt = 0
        mfma_cnt_total = 0
        should_cnt = False
        ## Find the s_barriers
        for i in range(len(code_list)):
            if "s_barrier" in code_list[i][0]:
                marker_barrier.append(code_list[i])
                if not found_1st_barrier:
                    ## This is barrier1
                    found_1st_barrier = True
                    should_cnt = True
                else:
                    ## This is barrier2 or barrier3
                    should_cnt = False
            if "mfma" in code_list[i][0] and should_cnt:
                mfma_dsRead_cnt += 1
            if "mfma" in code_list[i][0]:
                mfma_cnt_total += 1

    ## /= 2 because the last iteration of local_load and tt.dot
    ## is peeled off by stream-pipeliner
    mfma_cnt_total /= 2
    mfma_dsWrite_cnt = mfma_cnt_total - mfma_dsRead_cnt

    if len(marker_barrier) != 3:
        print(f"Not 3 barriers?? Found {len(marker_barrier)}")
        exit(0)
    marker_to_line['barrier_before_ds_read'] = marker_barrier[0][2]
    marker_to_line['instrAfterBarrier1'] = marker_barrier[0][2] + 1
    marker_to_line['barrier_before_ds_write'] = marker_barrier[1][2]
    marker_to_line['instrAfterBarrier2'] = marker_barrier[1][2] + 1
    marker_to_line['barrier_after_loop'] = marker_barrier[2][2]
    marker_to_line['instrAfterBarrier3'] = marker_barrier[2][2] + 1

    instrAfterBarrier1_clk = list()
    instrAfterBarrier2_clk = list()
    instrAfterBarrier3_clk = 0
    firstInstr_clk = 0
    lastInstr_clk = 0

    ## Read trace to get clk info for the markers
    with open(trace_fullname) as trace_f:
        trace_data = json.load(trace_f)
        trace_list = trace_data['wave']['instructions']

        for i in range(len(trace_list)):
            ## Capture the clk for the first instruction in the kernel
            if trace_list[i][-1] == marker_to_line['firstInstr']:
                firstInstr_clk = trace_list[i][0]
            ## Capture barrier1
            if trace_list[i][-1] == marker_to_line['instrAfterBarrier1']:
                instrAfterBarrier1_clk.append(trace_list[i][0])
            ## Capture barrier2
            if trace_list[i][-1] == marker_to_line['instrAfterBarrier2']:
                instrAfterBarrier2_clk.append(trace_list[i][0])
            ## Capture barrier3
            if trace_list[i][-1] == marker_to_line['instrAfterBarrier3']:
                instrAfterBarrier3_clk = trace_list[i][0]
        lastInstr_clk = trace_list[-1][0]

    incomplete = False
    if len(instrAfterBarrier1_clk) != len(instrAfterBarrier2_clk):
        print("different length of instrAfterBarrier1_clk and instrAfterBarrier2_clk")
        incomplete = True

    len1 = len(instrAfterBarrier1_clk)
    len2 = len(instrAfterBarrier2_clk)
    len3 = instrAfterBarrier3_clk

    if len1 == 0 or len2 == 0 or len3 == 0:
        incomplete = True

    #print(f"{firstInstr_clk}")
    #print(f"{instrAfterBarrier1_clk}")
    #print(f"{instrAfterBarrier2_clk}")
    #print(f"{instrAfterBarrier3_clk}")
    #print(f"{lastInstr_clk}")
    #print(f"{mfma_dsRead_cnt}")
    #print(f"{mfma_dsWrite_cnt}")
    #print(f"{incomplete}")

    return firstInstr_clk, instrAfterBarrier1_clk, instrAfterBarrier2_clk, instrAfterBarrier3_clk, lastInstr_clk, mfma_dsRead_cnt, int(mfma_dsWrite_cnt), incomplete


def gen_coarse_clk(instr0_clk, bar1_clk, bar3_clk, instr9_clk):
    prologue = bar1_clk[0] - instr0_clk
    loop = bar3_clk - bar1_clk[0]
    epilogue = instr9_clk - bar3_clk
    clk_per_iter = loop / len(bar1_clk)
    return prologue, loop, epilogue, clk_per_iter


def gen_max_wid(code_fullname):
    code_f = open(code_fullname)
    code_data = json.load(code_f)
    num_wv = code_data['code'][2][-2]
    return int(num_wv / 8)


def gen_fine_clk(bar1_clk, bar2_clk, bar3_clk):
    bar1_lat = list()
    bar2_lat = list()
    for i in range(len(bar1_clk)):
        bar1_lat.append(bar2_clk[i] - bar1_clk[i])
        if i + 1 == len(bar1_clk):
            bar2_lat.append(bar3_clk - bar2_clk[i])
        else:
            bar2_lat.append(bar1_clk[i + 1] - bar2_clk[i])

    return bar1_lat, bar2_lat


def list_to_stat(myList):
    ave = mean(myList)
    maxVal = max(myList)
    minVal = min(myList)
    stdVal = np.std(myList)
    return int(ave), maxVal, minVal, stdVal


def print_loop_eff(list1, list2, cnt1, cnt2):
    if len(list1) != len(list2):
        print("lists do not have the same length!!")
        exit(0)

    ave1, max1, min1, stddev1 = list_to_stat(list1)
    ave2, max2, min2, stddev2 = list_to_stat(list2)

    return ave1, ave2, ave1 + ave2, ave1 - cnt1 * 2 * 16, ave2 - cnt2 * 2 * 16


def calc_global_store_cycles(code_fullname):
    f = open(code_fullname)
    data = json.load(f)
    idx = 0
    saw_store = False
    total_mem = 0
    total_hitcnt = 0
    vmcnt_cnt = 0
    total_iss = 0
    total_iss_cnt = 0
    for i in data["code"]:
        if "global_store" in i[0]:
            global_store_name = i[0].split()[0]
            total_iss += i[-1] / i[-2]
            total_iss_cnt += i[-2]
            saw_store = True
            idx += 1
        if saw_store and "vmcnt(0)" in i[0]:
            hitcnt = i[-2]
            total_mem += i[-1] / i[-2]
            total_hitcnt += hitcnt
            vmcnt_cnt += 1

    print(f"{idx} {global_store_name}  {vmcnt_cnt} vmcnt(0)")
    print(f"total cycles: {total_iss+total_mem:.0f} = {total_iss:.0f}(iss) + {total_mem:.0f}(mem)")
    print(f"{total_iss/idx:.1f} issue cycles per {global_store_name}")
    print(f"{total_mem/vmcnt_cnt:.1f} cycles per vmcnt(0)")


def main():
    args = parse_args()
    trace_dir = args.d
    se = args.se
    sm = args.sm
    sl = args.sl
    wv = args.wv

    code_filename = "code.json"
    code_fullname = os.path.join(trace_dir, code_filename)
    trace_filename = f"se{se}_sm{sm}_sl{sl}_wv{wv}.json"
    trace_fullname = os.path.join(trace_dir, trace_filename)
    maxwid = gen_max_wid(code_fullname)

    print("wid,prologue,loop,epilogue,iter_clk,lat1,lat2,iter_lat,idle1,idle2")
    epi_total = 0
    epi_1st = 0
    total = 0
    flag = False
    cnt = 0
    for wid in range(maxwid):
        if wv != -1 and wid != wv:
            continue
        trace_filename = f"se{se}_sm{sm}_sl{sl}_wv{wid}.json"
        trace_fullname = os.path.join(trace_dir, trace_filename)
        if not os.path.isfile(trace_fullname):
            #print(f"trace file not found {trace_fullname}")
            return
        pro, loop, epi, iter_clk, lat1, lat2, lat_sum, idle1, idle2, incomplete = parse_trace(code_fullname, trace_fullname)
        if incomplete:
            continue
        print(f"{wid},{pro},{loop},{epi},{iter_clk:.0f},{lat1},{lat2},{lat_sum},{idle1},{idle2}")
        if not flag:
            epi_1st = epi
            flag = False

        if epi > 2 * epi_1st:
            continue
        epi_total += epi
        total += epi + pro + loop
        cnt += 1

    if cnt == 0:
        exit(0)
    print(f"averaged epilogue cycles: {epi_total / cnt:.0f}")
    print(f"averaged total cycles: {total / cnt:.0f}")
    print(f"global_store info (averaged for all {maxwid} waves):")
    calc_global_store_cycles(code_fullname)


if __name__ == '__main__':
    sys.exit(main())
