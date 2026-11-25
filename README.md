# macrograd-cuda

CUDA kernels for the ops in [macrograd](https://github.com/arun0786-co/macrograd), written from scratch. Every kernel is checked against a CPU reference (macrograd computes the same op in plain Python) and timed against cuBLAS or PyTorch on the same GPU.

macrograd shows how each op and its gradient work. This repo makes those ops fast on a GPU and measures how fast they are.

## Plan

### 1. Kernels

Each kernel gets a CPU reference, a correctness test and a timing.

- [x] vector add
- [x] elementwise ops with their backward: relu, tanh, sigmoid, gelu
- [x] sum reduction: naive, shared memory, warp shuffle
- [ ] softmax per row
- [ ] matmul: naive
- [ ] matmul: tiled with shared memory
- [ ] matmul: register blocking

### 2. Train an MLP on MNIST

Forward and backward written by hand for each layer, no autograd.

- [ ] linear layer forward and backward
- [ ] cross entropy forward and backward
- [ ] SGD and Adam update kernels
- [ ] full training loop on the GPU, matching macrograd's loss on a small model with the same weights

### 3. Results

- [ ] GFLOPS for each matmul version vs cuBLAS at a few sizes
- [ ] Nsight Compute numbers that explain each speedup (memory throughput, occupancy)
- [ ] MNIST time per epoch and test accuracy

## Layout

```
kernels/   one .cu file per op
tests/     correctness checks against the CPU reference
bench/     timing
mnist/     training
```

## Running

CUDA needs an NVIDIA GPU, so this doesn't run on a Mac. Use Google Colab (Runtime > Change runtime type > T4 GPU), Kaggle, or a cluster. Colab already has `nvcc`.

```
!nvidia-smi
!git clone https://github.com/arun0786-co/macrograd-cuda.git
%cd macrograd-cuda
!nvcc -O3 -arch=native kernels/vector_add.cu -o run && ./run
```

## Results

All on an NVIDIA GeForce GTX 1080 Ti with CUDA 12.4, built with `nvcc -O3 -arch=native`. Times are the mean of 100 launches after a warmup.

### Vector add

n = 16,777,223 floats (2^24 + 7, so the last block is only partly used). Each launch reads `a` and `b` and writes `c`, 12 bytes per element.

| | Time | Bandwidth | Max diff vs CPU |
|---|---|---|---|
| vector add | 0.555 ms | 362.5 GB/s | 0 |

### Elementwise ops

Same n. Forward reads `x` and writes `y`, 8 bytes per element. Backward adds `dy * f'(x)` into `dx`, reading one of `x` or `y` (tanh and sigmoid get their slope from `y`), `dy` and `dx` and writing `dx`, 16 bytes per element. The error is the largest `|gpu - ref| / (1 + |ref|)` against a CPU version in double precision. Inputs are uniform in [-6, 6] plus 0, tiny values and values up to ±100.

| Op | Forward | Backward | Max error (forward / backward) |
|---|---|---|---|
| relu | 0.371 ms, 361.5 GB/s | 0.741 ms, 362.5 GB/s | 0 / 3.0e-8 |
| tanh | 0.372 ms, 361.2 GB/s | 0.743 ms, 361.2 GB/s | 7.0e-8 / 1.2e-7 |
| sigmoid | 0.371 ms, 361.4 GB/s | 0.741 ms, 362.4 GB/s | 5.0e-8 / 8.4e-8 |
| gelu | 0.371 ms, 361.8 GB/s | 0.740 ms, 362.8 GB/s | 8.3e-8 / 5.3e-7 |
| `cudaMemcpy` device to device | 0.378 ms, 354.8 GB/s | | |

All four run at the same speed as vector add and a plain copy, so they're limited by memory, not by the math: gelu's tanh and cubic cost nothing you can measure. The 1080 Ti is rated at 484 GB/s, and these reach about 75% of that. Making them faster would mean moving fewer bytes, for example by doing the activation in the same kernel as the matmul before it.

### Sum reduction

Same n, uniform in [0, 1), so the sum is about 8.4 million. Each version reads every float once, 4 bytes per element. The error is `|gpu - ref| / sum(|x|)` against a CPU sum in double precision. "Same bits" means 6 runs gave exactly the same float.

| Kernel | Time | Bandwidth | Error | Same bits |
|---|---|---|---|---|
| atomic: every thread does `atomicAdd` into one float | 27.4 ms | 2.4 GB/s | 1.3e-4 | no |
| shared memory: a tree per block of 256, run again on the partials (4 launches) | 0.520 ms | 129.1 GB/s | 3.2e-9 | yes |
| warp shuffle: grid-stride loop, `__shfl_down_sync` in each warp, 224 blocks then 1 | 0.191 ms | 351.0 GB/s | 3.2e-9 | yes |
| `cub::DeviceReduce::Sum` | 0.186 ms | 361.2 GB/s | 3.2e-9 | yes |

- The atomic version is about 140 times slower than the others because 16.7 million adds to one address happen one at a time. It is also the least accurate: once the total is in the millions, a float can only change in steps of 0.5 or 1, so each small value added loses its low bits. The order of the adds changes from run to run, and so does the answer.
- The shared memory version adds in a fixed tree, so it is accurate and gives the same answer every time. It reads only 256 floats per block before a long chain of `__syncthreads` steps, and it needs four launches, so it reaches about a third of the memory bandwidth.
- The warp shuffle version has each thread add about 290 floats in a register first, then the warp adds its 32 sums by passing registers, with no shared memory until the last 8 values. It runs at 97% of CUB's speed and at the same speed as a plain copy, so it is limited by memory.

The test also checks sizes from 1 to 65,793 (around the block size and the edges of each pass), and runs the shuffle version with 1 and 1024 blocks so every warp of its second pass gets data. I put 9 deliberate bugs into copies of the file and 8 fail the test. The one that passes makes the shared memory version write its partials into the buffer it is reading from. That is a race, but on this GPU the blocks run in order and read their values before a later block overwrites them, so the answer comes out right anyway.

## References

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), Simon Boehm
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), NVIDIA
- [llm.c](https://github.com/karpathy/llm.c), GPT-2 training in plain C and CUDA
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
