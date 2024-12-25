import argparse
import sys
import os
import subprocess


def draw_dot_layout_cmd(M, N, K, mfmaNonKDim, warpsPerCTA, trans, kWidth, kGroup, dtype_a, dtype_b, mfma_inst_str,
                        kpack):
    elemSmall = 0.04
    elemLarge = 0.16
    elemPerThread = kWidth * kGroup
    if elemPerThread == 16:
        ratio = 0.8
    elif elemPerThread == 32:
        ratio = 0.6
    else:
        ratio = 1
    elemWidth = elemLarge * ratio

    scaleLabel = 0.7 if (kWidth == 4 or (kWidth == 8 and mfmaNonKDim == 32)) else 1

    outType = 'i32' if dtype_a == 'i8' else 'f32'

    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{{elemSmall}}}
    \\def\\elemW{{\\elem}}
    \\coordinate (C TL) at (0,0);
    \\drawDot{{{M}}}{{{N}}}{{{K}}}{{{mfmaNonKDim}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{trans}}}{{{kWidth}}}{{{kGroup}}}

    \\coordinate (C TL) at ($(C TL)+({N}*\elem+32*\elem, 0)$);
    \\def\\mfmaTrans{{{trans}}}

    %% Draw zoomed in view of mfma
    \\def\\scaleLabel{{{scaleLabel}}}
    \\pgfmathsetmacro{{\\oldElem}}{{\\elem}}
    \\def\\elem{{{elemLarge}}}
    \\def\\elemW{{{elemWidth}}}
    \\pgfmathsetmacro{{\\gap}}{{\\elem*5}}
    \\pgfmathsetmacro{{\\nonTrans}}{{1-\\mfmaTrans}}
    \\pgfmathsetmacro{{\\groups}}{{64/{mfmaNonKDim}}}
    \\coordinate (C TL) at ($(C TL)+(.5*\\gap+1.2*\\nonTrans*\\gap+\\groups*{kWidth}*{kGroup}*\\elemW, -{M}*\\oldElem+{mfmaNonKDim}*\\elem)$);
    \\coordinate (mfma instr) at ($(C TL)+(-.5*\\gap-0.6*\\nonTrans*\\gap-0.4*\\mfmaTrans*\\gap, 1.5*\\gap+.5*\\mfmaTrans*\\gap)$);
    \\node [scale=\scaleLabel, above left, align=left, draw=black, fill=white] at (mfma instr) {{{mfma_inst_str}}};
    \\drawMFMAInstr{{{mfmaNonKDim}}}{{{kWidth}}}{{{kGroup}}}{{\\mfmaTrans}}{{{dtype_a}}}{{{dtype_b}}}{{{outType}}}

  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, sizePerThread, threadsPerWarp, warpsPerCTA, order):
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{0.06}}
    \\coordinate (TL) at (0,0);
    \\def\\dimColName{{{dim0Name}}}
    \\def\\dimRowName{{{dim1Name}}}
    \\drawBlockedTensor{{{dim0}}}{{{dim1}}}{{{sizePerThread[0]}}}{{{sizePerThread[1]}}}{{{threadsPerWarp[0]}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{order[0]}}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_lds_access_cmd(M, K, kWidth, ldsLayout, ldsAccess, sizePerThread, threadsPerWarp):
    if ldsLayout == 'swizzle':
        hasSwizzle = 1
    elif ldsLayout == 'padding':
        hasSwizzle = 2
    else:
        hasSwizzle = 0

    if ldsAccess == 'read':
        accessMode = 1
    elif ldsAccess == 'write':
        accessMode = 2
    else:
        accessMode = 0

    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\M{{{M}}}
    \\def\\K{{{K}}}
    \\def\\vec{{{kWidth}}}
    \\def\\hasSwizzle{{{hasSwizzle}}}
    \\def\\accessMode{{{accessMode}}}

    \\def\\sizePerThreadK{{{sizePerThread[1]}}}
    \\def\\sizePerThreadM{{{sizePerThread[0]}}}
    \\def\\threadsPerWarpK{{{threadsPerWarp[1]}}}

    \\def\\elem{{0.18}}
    \\coordinate (TL) at (0,0);
    \\drawTensorLayoutGlobalMem
    \\coordinate (TL) at ($(TL)+(0, -24*\\elem-10*\\elem)$);
    \\drawLDSLayoutTritonSwizzling{{\\hasSwizzle}}{{\\accessMode}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def draw_wmma_instr_cmd(waveSize):
    wmma_mode = 0 if waveSize == 32 else 1
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\coordinate (C TL) at (0,0);
    \\def\\elem{{0.25}}
    \\drawWMMAInstr{{{wmma_mode}}}{{1}}
  \\end{{tikzpicture}}
\\end{{document}}'''


matrixFormatTable = {'fp8': 0, 'bf8': 1, 'fp6': 2, 'bf6': 3, 'f4': 4}


def matrixFormat(dtype_a, dtype_b):
    '''
    return CBSZ and BLGP according to data types
    b000: E4M3(FP8)
    b001: E5M2(BF8)
    b010: E2M3(FP6)
    b011: E3M2(BF6)
    b100: E2M1(FP4)
    '''
    return matrixFormatTable[dtype_a], matrixFormatTable[dtype_b]


def isType4Or6Bit(dtype):
    return dtype == 'fp6' or dtype == 'bf6' or dtype == 'f4'


def isType8BitFloat(dtype):
    return dtype == 'fp8' or dtype == 'bf8'


def isType16Bit(dtype):
    return dtype == 'bf16' or dtype == 'fp16'


def isMixedPrecType(dtype):
    return isType8BitFloat(dtype) or isType4Or6Bit(dtype)


def isMixedPrecBtwF8AndF4OrF6(dtype_a, dtype_b):
    return (isType8BitFloat(dtype_a) and isType4Or6Bit(dtype_b)) or (isType8BitFloat(dtype_b)
                                                                     and isType4Or6Bit(dtype_a))


def checkMfmaValidity(mfmaNonKDim, kWidth, kGroup, dtype_a, dtype_b, trans):
    ## Check input types
    ## Mixed precision is only allowed within f8, f6 and f4
    assert (isMixedPrecType(dtype_a) and isMixedPrecType(dtype_b)) or (
        dtype_a == dtype_b), f"Cannot do mixed precision mfma with {dtype_a} and {dtype_b}"
    '''
    Check mfma size according to data types
    * refers to newly added instructions on MI350
    Both dtyes are f4 or fp6 or bf6
      *mfma_f32_16x16x128_f8f6f4: kWidth = 32, kGroup = 1
      *mfma_f32_32x32x64_f8f6f4: kWidth = 32, kGroup = 1
    One dtype is fp8 or bf8
      When the other operand is f4, fp6, or bf6
        *mfma_f32_16x16x128_f8f6f4: kWidth = 16, kGroup = 2
        *mfma_f32_32x32x64_f8f6f4: kWidth = 16, kGroup = 2
      When the other operand is fp8 or bf8
        *mfma_f32_16x16x128_f8f6f4: kWidth = 16, kGroup = 2
        mfma_f32_16x16x32_fp8/bf8_fp8/bf8: kWidth = 16, kGroup = 1, kpack=2
        mfma_f32_16x16x32_fp8/bf8_fp8/bf8: kWidth = 8, kGroup = 1
        *mfma_f32_32x32x64_f8f6f4: kWidth = 16, kGroup = 2
        mfma_f32_32x32x16_fp8/bf8_fp8/bf8: kWidth = 16, kGroup = 1, kpack=2
        mfma_f32_32x32x16_fp8/bf8_fp8/bf8: kWidth = 8, kGroup = 1
    Both dtypes are bf16 or bf16
        *mfma_f32_16x16x32_f16/bf16: kWidth = 8, kGroup = 1
        mfma_f32_16x16x16_f16/bf16: kWidth = 4, kGroup = 1
        *mfma_f32_32x32x16_f16/bf16: kWidth = 8, kGroup = 1
        mfma_f32_32x32x8_f16/bf16: kWidth = 4, kGroup = 1
    Both types are i8
        *mfma_i32_16x16x64_i8: kWidth = 16, kGroup = 1
        mfma_i32_16x16x32_i8: kWidth = 8, kGroup = 1
        *mfma_i32_32x32x32_i8: kWidth = 16, kGroup = 1
        mfma_i32_32x32x16_i8: kWidth = 8, kGroup = 1

    Return mfma instruction name and kpack
    '''
    kDim = 64 / mfmaNonKDim * kWidth * kGroup
    ## Both dtyes are f4 or fp6 or bf6
    if isType4Or6Bit(dtype_a) and isType4Or6Bit(dtype_b):
        assert kWidth == 32 and kGroup == 1, f"Only kWidth=32 and kGroup=1 is supported for {dtype_a} x {dtype_b}"
        kpack = 1
        CBSZ = matrixFormatTable[dtype_b] if trans else matrixFormatTable[dtype_a]
        BLGP = matrixFormatTable[dtype_a] if trans else matrixFormatTable[dtype_b]
        return f"mfma_f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_f8f6f4", kpack, CBSZ, BLGP

    ## Both dtypes are fp8 or bf8
    if isType8BitFloat(dtype_a) and isType8BitFloat(dtype_b):
        assert (kWidth == 8 and kGroup == 1) or (
            kWidth == 16), f"Not a valid mfma instruction for {dtype_a} x {dtype_b} with {kWidth=} and {kGroup=}"
        kpack = 2 if (kWidth == 16 and kGroup == 1) else 1
        if kGroup == 2:
            suffix = "f8f6f4"
            CBSZ = matrixFormatTable[dtype_b] if trans else matrixFormatTable[dtype_a]
            BLGP = matrixFormatTable[dtype_a] if trans else matrixFormatTable[dtype_b]
        else:
            suffix = f"{dtype_b}_{dtype_a}" if trans else f"{dtype_a}_{dtype_b}"
            CBSZ = -1
            BLGP = -1
        kDim = kDim / 2 if kpack == 2 else kDim
        return f"mfma_f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{suffix}", kpack, CBSZ, BLGP

    ## Both types are fp16 or bf16
    if isType16Bit(dtype_a) and isType16Bit(dtype_b):
        assert (
            kWidth == 8 or kWidth == 4
        ) and kGroup == 1, f"Not a valid mfma instruction for {dtype_a} x {dtype_b} with {kWidth=} and {kGroup=}"
        kpack = 1
        CBSZ = -1
        BLGP = -1
        return f"mfma_f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtype_a}", kpack, CBSZ, BLGP

    ## Both types are i8
    if dtype_a == 'i8' and dtype_b == 'i8':
        assert (
            kWidth == 16 or kWidth == 8
        ) and kGroup == 1, f"Not a valid mfma instruction for {dtype_a} x {dtype_b} with {kWidth=} and {kGroup=}"
        kpack = 1
        CBSZ = -1
        BLGP = -1
        return f"mfma_i32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtype_a}", kpack, CBSZ, BLGP

    assert False, "Mixed precision between fp8/bf8 and fp6/bf6/f4 not supported in this mode"


def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
    return proc.stdout.splitlines()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Draw triton layouts",
        allow_abbrev=False,
    )
    ## tensor shapes
    parser.add_argument("-tensorShape", type=int, nargs=2, default=(128, 64),
                        help='2D tensor shape in the form of dim0,dim1')
    parser.add_argument("-dotShape", type=int, nargs=3, default=(32, 128, 64), help='Dot op shape in the form of M,N,K')
    parser.add_argument("-plot", type=str, default="blocked", choices=['blocked', 'dot', 'wmma', 'lds'],
                        help='choose plot mode')
    parser.add_argument("-dim0", type=str, default="M", help='tensor dim0 name')
    parser.add_argument("-dim1", type=str, default="K", help='tensor dim1 name')
    ## blocked layout parameters
    parser.add_argument("-sizePerThread", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-threadsPerWarp", type=int, nargs=2, default=(16, 4))
    parser.add_argument("-warpsPerCTA", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-order", type=int, nargs=2, default=(1, 0))
    ## dot layout parameters
    parser.add_argument("-nonKDim", type=int, default=16, choices=[16, 32], help='mfma instruction dim')
    parser.add_argument("-kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements per thread')
    parser.add_argument("-kGroup", type=int, default=1, choices=[1, 2],
                        help='total number of elements / kWidth per mfma instruction')
    parser.add_argument("-dtype_a", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of operand A')
    parser.add_argument("-dtype_b", type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp8', 'bf8', 'fp6', 'bf6', 'f4',
                                 'i8'], help='element type of operand B')
    ## LDS access parameters
    parser.add_argument("-lds_layout", type=str, default="none", choices=['swizzle', 'padding', 'none'],
                        help='choose the LDS data layout')
    parser.add_argument("-lds_access", type=str, default="none", choices=['read', 'write', 'none'],
                        help='choose LDS access mode')
    ## wmma instruction layout parameter
    parser.add_argument("-wave_size", type=int, default=32, choices=[32, 64], help='choose the wmma instruction mode')

    parser.add_argument("-o", type=str, default="myplot", help='output pdf file name (without surfix)')
    parser.add_argument("-mfmaTrans", action='store_true', default=False, help='If set, then use mfma.trans layout')
    parser.add_argument("-keep", action='store_true', default=False, help='If set, keep the generated .tex file')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    dotShape = args.dotShape
    M = dotShape[0]
    N = dotShape[1]
    K = dotShape[2]
    tShape = args.tensorShape
    dim0 = tShape[0]
    dim1 = tShape[1]
    dim0Name = args.dim0
    dim1Name = args.dim1
    plot_mode = args.plot
    mfmaNonKDim = args.nonKDim
    kWidth = args.kWidth
    kGroup = args.kGroup
    dtype_a = args.dtype_a
    dtype_b = args.dtype_b
    trans = 1 if args.mfmaTrans else 0
    ofilename = args.o
    keepSrc = args.keep

    ldsLayout = args.lds_layout
    ldsAccess = args.lds_access

    waveSize = args.wave_size

    sizePerThread = args.sizePerThread
    threadsPerWarp = args.threadsPerWarp
    warpsPerCTA = args.warpsPerCTA
    order = args.order

    CTAShape = []
    if plot_mode == 'blocked':
        print(f"Plotting tensor {dim0Name}={dim0},{dim1Name}={dim1} with blocked layout:")
        print(f"{sizePerThread=}", end=" ")
        print(f"{threadsPerWarp=}", end=" ")
        print(f"{warpsPerCTA=}", end=" ")
        print(f"{order=}", end=" ")
        CTAShape.append(sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0])
        CTAShape.append(sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1])
        print(f"CTAShape={CTAShape}")
        assert dim0 != 0 and CTAShape[0] <= dim0 and dim0 % CTAShape[0] == 0, "bad tensor dimension " + dim0Name
        assert dim1 != 0 and CTAShape[1] <= dim1 and dim1 % CTAShape[1] == 0, "bad tensor dimension " + dim1Name

    if plot_mode == 'dot':
        CTAShape.append(mfmaNonKDim * warpsPerCTA[0])
        CTAShape.append(mfmaNonKDim * warpsPerCTA[1])
        print(f"Plotting dot operation with shapes=M{M}-N{N}-K{K},{kWidth=},{kGroup=},{warpsPerCTA=},{CTAShape=}")
        assert M != 0 and CTAShape[0] <= M and M % CTAShape[0] == 0, "bad tensor dimension M"
        assert N != 0 and CTAShape[1] <= N and N % CTAShape[1] == 0, "bad tensor dimension N"
        if isMixedPrecBtwF8AndF4OrF6(dtype_a, dtype_b):
            ## In the case of mixed precision between 8-bit and 4 or 6-bit,
            ## ignore kWidth and kGroup since inA and inB have different kWidth and kGroup values
            kDim = 128
            assert K != 0 and K % kDim == 0, f"one mfma instruction requires {kDim:.0f} elements along k dim but BLOCK_K = {K}"
            kpack = 1
            CBSZ = matrixFormatTable[dtype_b] if trans else matrixFormatTable[dtype_a]
            BLGP = matrixFormatTable[dtype_a] if trans else matrixFormatTable[dtype_b]
            mfma_inst_str = f"mfma_f32_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_f8f6f4"
        else:
            kDim = kWidth * kGroup * 64 / mfmaNonKDim
            assert K != 0 and K % kDim == 0, f"one mfma instruction requires {kDim:.0f} elements along k dim but BLOCK_K = {K}"
            mfma_inst_str, kpack, CBSZ, BLGP = checkMfmaValidity(mfmaNonKDim, kWidth, kGroup, dtype_a, dtype_b, trans)
        flag = '' if CBSZ == -1 else f" with {CBSZ=},{BLGP=}"
        print(f"MFMA: {mfma_inst_str} x {kpack}{flag}", end="")
        mfma_inst_str = mfma_inst_str.replace("_", "\\_")
        mfma_inst_str = mfma_inst_str + flag
        if kpack == 2:
            mfma_inst_str = mfma_inst_str + " $\\times$ 2"
        if ((dtype_a == 'fp16' or dtype_a == 'bf16') and kWidth == 8) or (dtype_a == 'i8' and kWidth == 16):
            kDim = 64 / mfmaNonKDim * kWidth / 2
            outType = "i32" if dtype_a == 'i8' else "f32"
            old_instr = f"mfma_{outType}_{mfmaNonKDim}x{mfmaNonKDim}x{kDim:.0f}_{dtype_a}"
            print(f" or {old_instr} x 2")
            old_instr = old_instr.replace("_", "\\_")
            mfma_inst_str = mfma_inst_str + " or\\\\" + old_instr + "$\\times$2"
        else:
            print("")

    if plot_mode == 'lds':
        print(f"Plotting LDS access for tensor M={M},K={K} with vec={kWidth}")
        if ldsAccess == 'write':
            print(f"sizePerThread={sizePerThread}, threadsPerWarp={threadsPerWarp}")

    with open("myplot.tex", 'w') as f_plot:
        with open("preamble.tex") as file:
            preamble = file.read()

        draw_blockedLayout_str = draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, sizePerThread, threadsPerWarp,
                                                         warpsPerCTA, order)

        draw_dotLayout_str = draw_dot_layout_cmd(M, N, K, mfmaNonKDim, warpsPerCTA, trans, kWidth, kGroup, dtype_a,
                                                 dtype_b, mfma_inst_str, kpack)

        draw_lds_str = draw_lds_access_cmd(M, K, kWidth, ldsLayout, ldsAccess, sizePerThread, threadsPerWarp)

        draw_wmma_str = draw_wmma_instr_cmd(waveSize)

        f_plot.write(preamble)
        if plot_mode == 'blocked':
            f_plot.write("\input{blockedLayout}\n")
            f_plot.write(draw_blockedLayout_str)
        elif plot_mode == 'dot':
            f_plot.write("\input{dotLayout}\n")
            f_plot.write(draw_dotLayout_str)
        elif plot_mode == 'lds':
            f_plot.write("\input{ldsLayout}\n")
            f_plot.write(draw_lds_str)
        elif plot_mode == 'wmma':
            f_plot.write("\input{wmmaLayout}\n")
            f_plot.write(draw_wmma_str)

    run_bash_command(f"pdflatex -jobname {ofilename} myplot.tex")
    print(f"plot saved in {ofilename}.pdf")

    ## Remove au files
    os.remove(f"{ofilename}.aux")
    os.remove(f"{ofilename}.log")
    if not keepSrc:
        os.remove("myplot.tex")
        run_bash_command("rm -rf ./auto")


if __name__ == '__main__':
    sys.exit(main())
