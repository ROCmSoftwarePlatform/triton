# Plot script for triton layouts

This script is used to draw triton layouts in the context of matmul.
Here is the help info from the script.

```bash
>$ python3 plot_layout.py -h
usage: Draw triton layouts [-h] [-tensorShape TENSORSHAPE TENSORSHAPE] [-dotShape DOTSHAPE DOTSHAPE DOTSHAPE] [-plot {blocked,dot,wmma,lds}] [-dim0 DIM0] [-dim1 DIM1] [-sizePerThread SIZEPERTHREAD SIZEPERTHREAD] [-threadsPerWarp THREADSPERWARP THREADSPERWARP]
                           [-warpsPerCTA WARPSPERCTA WARPSPERCTA] [-order ORDER ORDER] [-nonKDim {16,32}] [-kWidth {4,8,16,32}] [-kGroup {1,2}] [-lds_layout {swizzle,padding,none}] [-lds_access {read,write,none}] [-wave_size {32,64}] [-o O] [-mfmaTrans] [-keep]

options:
  -h, --help            show this help message and exit
  -tensorShape TENSORSHAPE TENSORSHAPE
                        2D tensor shape in the form of dim0,dim1
  -dotShape DOTSHAPE DOTSHAPE DOTSHAPE
                        Dot op shape in the form of M,N,K
  -plot {blocked,dot,wmma,lds}
                        choose plot mode
  -dim0 DIM0            tensor dim0 name
  -dim1 DIM1            tensor dim1 name
  -sizePerThread SIZEPERTHREAD SIZEPERTHREAD
  -threadsPerWarp THREADSPERWARP THREADSPERWARP
  -warpsPerCTA WARPSPERCTA WARPSPERCTA
  -order ORDER ORDER
  -nonKDim {16,32}      mfma instruction dim
  -kWidth {4,8,16,32}   number of contiguous elements per thread
  -kGroup {1,2}         total number of elements / kWidth per mfma instruction
  -lds_layout {swizzle,padding,none}
                        choose the LDS data layout
  -lds_access {read,write,none}
                        choose LDS access mode
  -wave_size {32,64}    choose the wmma instruction mode
  -o O                  output pdf file name (without surfix)
  -mfmaTrans            If set, then use mfma.trans layout
  -keep                 If set, keep the generated .tex file
```

## Installation
This script does not require torch or triton to be installed. The only package
it depends on is latex. On Ubuntu, do
```bash
sudo apt install texlive-full
```

## Draw blocked layout (`-plot blocked`)

Examples:
```bash
python3 plot_layout.py -plot blocked -tensorShape 128 64 -sizePerThread 1 8 -threadsPerWarp 8 8 -warpsPerCTA 4 1
python3 plot_layout.py -plot blocked -tensorShape 16 64 -sizePerThread 1 8 -threadsPerWarp 16 4 -warpsPerCTA 1 2
python3 plot_layout.py -plot blocked -tensorShape 32 64 -sizePerThread 8 1 -threadsPerWarp 4 16 -warpsPerCTA 1 2 -order 0 1
```

Blocked layouts are used during global load. It is used to describe the layout of the tensor
for pointers and results.
We can provide tensor shape (`-tensorShape dim0 dim1`) and blocked layout parameters (
`-sizePerThread x y`, `-threadsPerWarp x y`, and `-warpsPerCTA x y`).
We can also provide the order of the tensor as `-order x y` to control which dim
is the fastest changing dimension.

Notes
- The script does not support the case when threads are loading elements that are
  out of the boundary of the tensor dimensions. This means
  - For dim0: sizePerThread[0] * threadsPerWarps[0] * warpsPerCTA[0] <= dim0
  - For dim1: sizePerThread[1] * threadsPerWarps[1] * warpsPerCTA[1] <= dim1


## Draw mfma operand and result layouts (`-plot dot`)

Examples:
```bash
python3 plot_layout.py -plot dot -dotShape 128 128 64 -warpsPerCTA 2 4 -nonKDim 32 -kWidth 4
python3 plot_layout.py -plot dot -dotShape 128 128 64 -warpsPerCTA 2 4 -nonKDim 32 -kWidth 8
python3 plot_layout.py -plot dot -dotShape 128 128 64 -warpsPerCTA 2 4 -nonKDim 32 -kWidth 8 -mfmaTrans
python3 plot_layout.py -plot dot -dotShape 128 128 64 -warpsPerCTA 2 4 -nonKDim 16 -kWidth 8
python3 plot_layout.py -plot dot -dotShape 128 128 64 -warpsPerCTA 2 4 -nonKDim 16 -kWidth 16
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -nonKDim 16 -kWidth 16 -kGroup 2
```

This mode draws two graphs:
1. The layout of the whole tile for tile A, B, and C
2. The layout of a single mfma block, operands and results of one or more mfma
   instructions that share the same accumulating VGPRs.
   This view has thread distributions among tensor elements.

Knobs
- `-kWidth`: the number of elements that will be loaded into one thread at once
- `-kGroup`: total number of elements / kWidth for on mfma instruction.
   This is 1 for all mfma instructions except for mfma_f32_16x16x128_f8f6f4 and mfma_f32_32x32x64_f8f6f4
   with fp8 input types (CBSZ=0 or 1 and/or BLGP=0 or 1)
- `-nonKDim`: 16 ot 32, which is used to control the mfma instruction size
- `-mfmaTrans`: if set, the transposed mfma layout will be plotted.

Notes
- The layout shows the mapping from the threads/wave to the elements in the
  original tensor. It does not care if the elements are re-arranged in LDS, like
  swizzling to avoid bank conflicts.
- The script does not allow settings for data type or k dim of the mfma instruction.
  This can be controled by the `-kWidth` flag.
  - For example, if we want `mfma_32x32x8xf16`, we can set `-nonKDim 32` and `-kWidth 4`.
  - If we want `mfma_32x32x16xf8`, we can set `-nonKDim 32` and `-kWidth 8`.


## Draw LDS access (`-plot lds`)

Examples:
```bash
python3 plot_layout.py -plot lds -lds_layout none -lds_access none -shape 128 128 64 -kWidth 8
```

Knobs
- `kWidth` here means the vector size when accessing LDS
- Three options for `-lds_layout`:
  - `none`: no swizzling, no padding
  - `padding`: padding at every 128B
  - `swizzling`: apply the swizzling pattern, which is derived from tensor shape and kWidth.
- Three options for `-lds_access`:
  - `none`: do not plot access pattern
  - `read`: plot accessed elements during ds_read
  - `write`: plot accessed elements during ds_write. Note that this needs some infomation from
    global load. Therefore, we need to provide `-sizePerThread` and `-threadsPerWarp`.

Notes
- This mode is rarely used. If you have any questions, please contact Lixun Zhang directly.
