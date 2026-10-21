# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal check: are shared atomicMin (result unused) ordered before plain stores after __syncthreads?"""

from torch.utils.cpp_extension import load_inline

cuda = r"""
#include <torch/extension.h>
__global__ void k(int* out, int n, int fence) {
  extern __shared__ int a[];
  for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] = 0x7fffffff;
  __syncthreads();
  for (int p = threadIdx.x; p < 2 * n; p += blockDim.x) {
    atomicMin(&a[(p * 2654435761u) % n], p);
  }
  if (fence) __threadfence_block();
  __syncthreads();
  for (int i = (threadIdx.x * 37) % blockDim.x; i < n; i += blockDim.x) {
    a[i] = (1 << 30) | i;
  }
  __syncthreads();
  int bad = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) bad += a[i] != ((1 << 30) | i);
  atomicAdd(out, bad);
}
torch::Tensor run(int blocks, int n, int fence) {
  auto out = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, n * 4);
  k<<<blocks, 1024, n * 4>>>(out.data_ptr<int>(), n, fence);
  return out;
}
"""
cpp = "torch::Tensor run(int blocks, int n, int fence);"
mod = load_inline(
    "atom_order", cpp, cuda_sources=cuda, functions=["run"], extra_cuda_cflags=["-O3"]
)
for fence in (0, 1):
    bad = sum(int(mod.run(512, 8192, fence).item()) for _ in range(2000))
    print(f"fence={fence}: late atomics observed in {bad} words over 2000 launches")
