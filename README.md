# macrograd-cuda

CUDA kernels for the ops in [macrograd](https://github.com/arun0786-co/macrograd), written from scratch. Every kernel is checked against a CPU reference (macrograd computes the same op in plain Python) and timed against cuBLAS or PyTorch on the same GPU.

macrograd shows how each op and its gradient work. This repo makes those ops fast on a GPU and measures how fast they are.

## Plan

### 1. Kernels

Each kernel gets a CPU reference, a correctness test and a timing.

- [ ] vector add
- [ ] elementwise ops with their backward: relu, tanh, sigmoid, gelu
- [ ] sum reduction: naive, shared memory, warp shuffle
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
!nvcc -O3 -arch=native kernels/<file>.cu -o run && ./run
```

## Results

Nothing measured yet. Each table will list the GPU, CUDA version and problem sizes.

## References

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), Simon Boehm
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), NVIDIA
- [llm.c](https://github.com/karpathy/llm.c), GPT-2 training in plain C and CUDA
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
