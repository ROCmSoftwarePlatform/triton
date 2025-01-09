import triton
import triton.language as tl
import torch

import numpy as np


def to_numpy(x):
    return x.cpu().numpy()


@triton.jit
def kernel(X, stride_xm, stride_xk, Y, stride_yk, stride_yn, Z, stride_zm, stride_zn, BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
    Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
    Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
    x = tl.load(Xs)
    y = tl.load(Ys)
    z = tl.dot(x, y)
    tl.store(Zs, z)


def permute_weight(x: torch.Tensor, numWarps) -> torch.Tensor:
    B = x.shape[0]
    N = x.shape[1]
    K = x.shape[2]
    x_ = x.clone()

    mfmaNSize = 16
    kWidth = 8
    numKGroups = 4

    kRepeats = K // numKGroups // kWidth
    nRepeats = N // numWarps // mfmaNSize
    #            0     1        2           3         4         5          6
    #            B  NNNNNNNN  NNNNNNNN  NNNNNNNNN  KKKKKKKK  KKKKKKKKKK  KKKKKK
    x_ = x_.view(B, nRepeats, numWarps, mfmaNSize, kRepeats, numKGroups, kWidth)

    x_ = x_.permute(0, 1, 4, 2, 5, 3, 6)

    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2])
    return x_


M = 128
N = 128
K = 128

x = torch.zeros((M, K), dtype=torch.float16, device="cuda")
for i in range(M):
    x[i, i] = 1
y = torch.zeros((1, N, K), dtype=torch.float16, device="cuda")

for i in range(K):
    for j in range(N):
        y[0, j, i] = i + j * K

ref = torch.matmul(x, y.permute([0, 2, 1]).reshape(K, N))

numWarps = 4
y = permute_weight(y, numWarps)

np.set_printoptions(threshold=100_000)
# print(to_numpy(y.reshape(N, K).to(torch.int32)))

z = torch.zeros((M, N), dtype=torch.float32, device="cuda")

kernel[(1, 1, 1)](x, x.stride(0), x.stride(1), y, y.stride(2), y.stride(1), z, z.stride(0), z.stride(1), M, N, K,
                  enable_moe_lds_bypass=True, num_warps=numWarps, matrix_instr_nonkdim=16, kpack=2)

print(to_numpy(z.reshape(N, M).to(torch.int32)))

np.testing.assert_allclose(to_numpy(ref), to_numpy(z))
