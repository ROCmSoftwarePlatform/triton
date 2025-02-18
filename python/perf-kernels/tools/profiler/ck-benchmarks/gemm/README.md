## Install

```bash
mkdir build && cd build
CK_PATH=$(realpath <CK install directory>) CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ cmake ..
make VERBOSE=1 -j4
```

## Example

```bash
./ck-gemm-runner -m 4864 -n 2048 -k 4160
```
