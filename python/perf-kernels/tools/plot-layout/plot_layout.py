import argparse
import sys
import os
import subprocess


def draw_dot_layout_cmd(M, N, K, mfmaNonKDim, warpsPerCTA, trans, kWidth):
    elemSmall = 0.04
    elemLarge = 0.16
    if kWidth == 16:
        ratio = 0.8
    elif kWidth == 32:
        ratio = 0.6
    else:
        ratio = 1
    elemWidth = elemLarge * ratio

    scaleLabel = 0.7 if kWidth == 4 else 1

    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\def\\elem{{{elemSmall}}}
    \\coordinate (C TL) at (0,0);
    \\drawDot{{{M}}}{{{N}}}{{{K}}}{{{mfmaNonKDim}}}{{{warpsPerCTA[0]}}}{{{warpsPerCTA[1]}}}{{{trans}}}{{{kWidth}}}

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
    \\coordinate (C TL) at ($(C TL)+(.5*\\gap+1.2*\\nonTrans*\\gap+\\groups*{kWidth}*\\elemW, -{M}*\\oldElem+{mfmaNonKDim}*\\elem)$);
    \\drawMFMAInstr{{{mfmaNonKDim}}}{{{kWidth}}}{{\\mfmaTrans}}

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
    parser.add_argument("-nonKDim", type=int, default=32, choices=[16, 32], help='mfma instruction dim')
    parser.add_argument("-dim0", type=str, default="M", help='tensor dim0 name')
    parser.add_argument("-dim1", type=str, default="K", help='tensor dim1 name')
    ## blocked layout parameters
    parser.add_argument("-sizePerThread", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-threadsPerWarp", type=int, nargs=2, default=(16, 4))
    parser.add_argument("-warpsPerCTA", type=int, nargs=2, default=(1, 4))
    parser.add_argument("-order", type=int, nargs=2, default=(1, 0))
    ## LDS access parameters
    parser.add_argument("-kWidth", type=int, default=4, choices=[4, 8, 16, 32],
                        help='number of contiguous elements per thread')
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

    if plot_mode == 'dot':
        mfma_inst_str = "mfma_32x32" if mfmaNonKDim == 32 else "mfma_16x16"
        mfma_trans_str = ".trans" if trans else ""
        print(f"Plotting dot operation with shapes {M=},{N=},{K=}")
        print("MFMA: " + mfma_inst_str + mfma_trans_str + f" {kWidth=}", end=" ")
        print(f"{warpsPerCTA=}", end=" ")
        CTAShape.append(mfmaNonKDim * warpsPerCTA[0])
        CTAShape.append(mfmaNonKDim * warpsPerCTA[1])

    if plot_mode == 'blocked' or plot_mode == 'dot':
        print(f"CTAShape={CTAShape}")

    if plot_mode == 'blocked':
        assert dim0 != 0 and CTAShape[0] <= dim0 and dim0 % CTAShape[0] == 0, "bad tensor dimension " + dim0Name
        assert dim1 != 0 and CTAShape[1] <= dim1 and dim1 % CTAShape[1] == 0, "bad tensor dimension " + dim1Name

    if plot_mode == 'dot':
        assert M != 0 and CTAShape[0] <= M and M % CTAShape[0] == 0, "bad tensor dimension M"
        assert N != 0 and CTAShape[1] <= N and N % CTAShape[1] == 0, "bad tensor dimension N"
        assert K != 0 and K % (2 * kWidth) == 0, "bad tensor dimension K"

    if plot_mode == 'lds':
        print(f"Plotting LDS access for tensor M={M},K={K} with vec={kWidth}")
        if ldsAccess == 'write':
            print(f"sizePerThread={sizePerThread}, threadsPerWarp={threadsPerWarp}")

    with open("myplot.tex", 'w') as f_plot:
        with open("preamble.tex") as file:
            preamble = file.read()

        draw_blockedLayout_str = draw_blocked_layout_cmd(dim0, dim1, dim0Name, dim1Name, sizePerThread, threadsPerWarp,
                                                         warpsPerCTA, order)

        draw_dotLayout_str = draw_dot_layout_cmd(M, N, K, mfmaNonKDim, warpsPerCTA, trans, kWidth)

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
