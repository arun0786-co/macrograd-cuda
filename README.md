# macrograd-cuda

CUDA kernels for the ops in [macrograd](https://github.com/arun0786-co/macrograd), written from scratch. Every kernel is checked against a CPU reference (macrograd computes the same op in plain Python) and timed against cuBLAS or PyTorch on the same GPU.

macrograd shows how each op and its gradient work. This repo makes those ops fast on a GPU and measures how fast they are.

## Plan

### 1. Kernels

Each kernel gets a CPU reference, a correctness test and a timing.

- [x] vector add
- [x] elementwise ops with their backward: relu, tanh, sigmoid, gelu
- [x] sum reduction: naive, shared memory, warp shuffle
- [x] softmax per row, with its backward
- [x] matmul: naive
- [x] matmul: tiled with shared memory
- [x] matmul: register blocking

### 2. Train an MLP on MNIST

Forward and backward written by hand for each layer, no autograd.

- [x] linear layer forward and backward
- [x] cross entropy forward and backward
- [x] SGD and Adam update kernels
- [x] full training loop on the GPU, matching macrograd's loss on a small model with the same weights

### 3. Results

- [x] GFLOPS for each matmul version vs cuBLAS at a few sizes
- [ ] Nsight Compute numbers that explain each speedup (memory throughput, occupancy). The cluster GPU doesn't allow the performance counters, so for now there are the compiler's and the occupancy calculator's numbers instead
- [x] MNIST time per epoch and test accuracy

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

All on an NVIDIA GeForce GTX 1080 Ti with CUDA 12.4 (12.9 for register blocking), built with `nvcc -O3 -arch=native`. Times are the mean of 100 launches after a warmup.

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

### Softmax per row

The same 2^24 floats as a tall, a square and a wide matrix, softmax over each row. Forward reads `x` and writes `y`, 8 bytes per value; backward adds `y * (dy - sum(dy * y))` into `dx`, reading `y`, `dy` and `dx` and writing `dx`, 16 bytes per value. GB/s counts only those bytes, so a kernel that reads a row more than once gets no credit for it. Inputs are uniform in [-10, 10], with some rows shifted by +1000 or -1000 and some with every third value set to -inf like an attention mask. The forward error is the largest `|gpu - ref| / ref` against a CPU version in double precision; the backward error is relative to the size of the terms that went into each value.

| Kernel | 65536 x 256 | 4096 x 4096 | 256 x 65536 | Max error |
|---|---|---|---|---|
| naive: one thread per row, three passes | 20.7 ms, 6.5 GB/s | 3.15 ms, 42.6 GB/s | 49.0 ms, 2.7 GB/s | 5.9e-5 |
| block: one block of 256 per row, three passes | 0.646 ms, 207.8 GB/s | 0.769 ms, 174.6 GB/s | 0.999 ms, 134.4 GB/s | 1.4e-6 |
| online: one block per row, max and sum in one pass | 1.174 ms, 114.4 GB/s | 0.580 ms, 231.3 GB/s | 0.789 ms, 170.2 GB/s | 1.4e-6 |
| backward: one block per row | 0.745 ms, 360.2 GB/s | 1.138 ms, 235.9 GB/s | 1.216 ms, 220.9 GB/s | 1.4e-7 |
| `cudaMemcpy` device to device | 0.378 ms, 354.8 GB/s | | | |

- The naive version is slow at every shape. With 256 rows it has only 256 threads, one block on one of the 28 SMs. With 65536 rows it has enough threads, but neighbouring threads read floats a whole row apart, so each load of a warp touches 32 places in memory. Its error is the largest because one thread adds a whole row into one running float: 5.9e-5 at 65536 columns, against 2.2e-6 at 256.
- The online version reads each row twice instead of three times, and takes 25% less time than the block version on the square matrix and 21% less on the wide one. On rows of 256 it is slower: each thread has a single value, so there is no pass to save, and combining two (max, sum) pairs costs two `exp` calls at every shuffle step. Across three runs the block version took 0.65 to 0.72 ms on this shape and the online version 1.11 to 1.17 ms.
- On the wide matrix, at most 8 blocks of 256 threads fit on an SM, so 224 of the 256 rows run at once and the last 32 run in a second round with most of the GPU idle.
- The backward at 256 columns runs at copy speed. A row is 1 KB, so its second pass over `y` and `dy` comes from cache. On longer rows it drops to about 230 GB/s.
- No forward version reaches copy speed. The next step would be to keep the row in registers or shared memory, so `x` is read from memory once.

The test also checks the backward formula against finite differences, and 30 small shapes: 1 to 4097 columns (around a warp and a block) and 1 to 257 rows (around the naive version's block). I put 14 deliberate bugs into copies of the file and 13 fail the test. The one that passes removes the `__syncthreads` before a block reduction writes its per-warp results, so a warp can overwrite a value another warp is still reading from the previous reduction. That is a race, but it didn't show up in these runs on this GPU.

### Matmul: naive

`C = A @ B` for square matrices, one thread per value of `C`, blocks of 32 x 32 threads. GFLOPS counts a multiply and an add for each of the `n^3` terms. Inputs are uniform in [-1, 1]. The error is `|gpu - ref| / sum_k |A[i][k] B[k][j]|` against a CPU matmul in double precision, over every value up to 1024 x 1024 and over 4100 sampled values above that. The last column is the time of `cublasSgemm` divided by the kernel's time.

| Kernel | 256 | 512 | 1024 | 2048 | 4096 | 4096, of cuBLAS |
|---|---|---|---|---|---|---|
| rows: `threadIdx.x` picks the row of `C` | 48.8 GFLOPS | 59.8 | 66.4 | 75.0 | 52.5 (2617 ms) | 0.5% |
| cols: `threadIdx.x` picks the column of `C` | 277.9 | 346.0 | 370.8 | 487.8 | 468.9 (293 ms) | 4.8% |
| `cublasSgemm` | 3245 | 4428 | 9044 | 10129 | 9849 (13.95 ms) | 100% |

The largest error was 2.9e-7, for cuBLAS too.

- The two kernels do the same arithmetic in the same order. The only difference is which thread gets which value of `C`. With `threadIdx.x` on the row, the 32 threads of a warp read `A` from 32 rows `n` floats apart at every step of `k`, and write `C` the same way. With it on the column, they read 32 neighbouring floats of `B`, which come in one transaction, and all want the same float of `A`, which the hardware sends to all of them at once. That alone makes it 5.6 to 8.9 times faster.
- Even so, it gets less than 5% of cuBLAS. Every multiply-add needs two loads, and nothing a thread loads is reused by the threads next to it except through the cache. The tiled version loads a tile of `A` and of `B` into shared memory once and has the whole block use it.
- cuBLAS reaches 10.1 TFLOPS at 2048, about 90% of the 11.3 TFLOPS the 1080 Ti is rated for at its boost clock. At 256 it gets a third of that, because there isn't enough work to keep all 28 SMs busy.

The test also checks 10 small and odd shapes, from 1 x 1 x 1 to 1000 x 777 x 513, including shapes where the last block is only partly used, single rows and columns, and `K = 1`. `C` is filled with NaN first, so a value that never gets written fails the test. I put 15 deliberate bugs into copies of the file (wrong strides, skipped rows, columns or steps of `k`, swapped grid dimensions) and all 15 fail the test.

### Matmul: tiled with shared memory

Same setup as the naive matmul. Each block of T x T threads walks along `K` one tile at a time: every thread loads one float of a T x T tile of `A` and one of `B` into shared memory, the block waits at `__syncthreads`, then every thread does T multiply-adds from shared memory. Each float loaded from global memory is used T times instead of once.

| Kernel | 256 | 512 | 1024 | 2048 | 4096 | 4096, of cuBLAS |
|---|---|---|---|---|---|---|
| naive, columns across the warp | 279.3 GFLOPS | 348.5 | 409.8 | 476.2 | 457.6 (300 ms) | 4.8% |
| tiled, T = 16 | 939.1 | 1271.7 | 1270.2 | 1395.1 | 1404.8 (97.8 ms) | 14.7% |
| tiled, T = 32 | 904.2 | 1219.3 | 1310.2 | 1598.9 | 1627.9 (84.4 ms) | 17.0% |
| `cublasSgemm` | 3310 | 4490 | 7652 | 9162 | 9586 (14.3 ms) | 100% |

The largest error was 2.9e-7, the same as the naive version, since the adds happen in the same order.

- Tiling with T = 32 makes it 3.6 times faster than the naive version at 4096.
- T = 16 is a little faster up to 512. A 256 x 256 matrix has only 64 tiles of 32 x 32 for the 28 SMs, against 256 tiles of 16 x 16. From 1024 up there is work for every SM either way, and T = 32 wins because each float it loads is used 32 times instead of 16.
- It is still at 17% of cuBLAS. Each multiply-add now reads two floats from shared memory instead of global memory, so shared memory is probably the next limit. The register blocking version has each thread compute several values of `C`, so one float read from shared memory feeds several multiply-adds.

I put 10 deliberate bugs into copies of the file and all 10 fail the test. Two of them only did so after I changed the test. Each removes one bounds check along `K`. That alone gives the right answer, because the other tile still has a 0 there and the product is 0, but it reads past the end of `A` or `B`. The memory there held leftover numbers, and a leftover number times 0 is 0. Now the test fills the buffers with NaN before copying `A` and `B` in, so a read past the end gives NaN times 0, which is NaN, and fails.

### Matmul: register blocking

Same setup as the other matmuls, with CUDA 12.9. Each thread makes several values of `C` and keeps them in registers, so a float read from shared memory feeds several multiply-adds. Both versions walk along `K` 8 columns at a time. These are from one of three runs.

| Kernel | 256 | 512 | 1024 | 2048 | 4096 | 4096, of cuBLAS |
|---|---|---|---|---|---|---|
| tiled, T = 32 | 904.2 GFLOPS | 1221.1 | 1308.8 | 1660.5 | 1641.5 (83.7 ms) | 16.8% |
| 1d: 8 values down a column per thread, blocks of 64 x 64 | 781.8 | 1246.3 | 1819.3 | 2426.6 | 2569.7 (53.5 ms) | 26.3% |
| 2d: 8 x 8 values per thread, blocks of 128 x 128 | 267.3 | 952.4 | 1498.6 | 2913.4 | 3637.3 (37.8 ms) | 37.3% |
| `cublasSgemm` | 3330 | 4512 | 7697 | 9218 | 9763 (14.1 ms) | 100% |

The largest error was 2.9e-7 over the square sizes. The three runs agree within 4% at 4096, but the third was up to 27% faster than the second at 256 to 1024 for every kernel, cuBLAS included, so compare kernels within a run.

- At 4096 the 1d version is 1.6 times faster than the tiled one and the 2d version 2.2 times. The tiled version reads two floats from shared memory per multiply-add, the 1d version 9 for 8 and the 2d version 16 for 64.
- The 2d version is the slowest of the three up to 512. Its blocks are large: a 256 x 256 `C` has only 4 of them and a 512 x 512 one has 16, so most or many of the 28 SMs have nothing to do. It passes the 1d version at 2048.
- The 1d kernel needs `__launch_bounds__`. Left alone it uses 68 registers per thread, so only one block of 512 threads fits on an SM, and it gets 1447 GFLOPS at 4096. Capped at 64 registers, with no spills, two blocks fit.
- It is still at 37% of cuBLAS.

The test also checks that nothing is written up to 128 rows past the end of `C`. I added that after a bug that removed the 2d kernel's row check passed. I put 17 deliberate bugs into copies of the file and 14 fail the test. Two of the three that pass remove a bounds check on the rows of `A` or the columns of `B`. Those values only go into values of `C` outside the matrix, which are never written, though the kernel does read past the end of `A` or into the next row of `B`. The third removes the last `__syncthreads` of the 2d loop, so a thread can load the next tile while another still reads this one. That is a race, but it didn't show up in these runs on this GPU.

### Matmul: what each kernel asks of the GPU

Nsight Compute needs the GPU's performance counters, and the cluster only lets admins read them (`ERR_NVGPUCTRPERM`). `bench/matmul_resources.cu` gets what it can without them: each kernel's registers, shared memory and spills from the compiler, how many blocks fit on an SM from the occupancy calculator, its time at 4096, and from its design, how many floats it must read from global memory. The GPU has 28 SMs, 64 warps, 65,536 registers and 96 KB of shared memory per SM, 11.3 TFLOPS at 1.58 GHz and 484 GB/s.

| Kernel | Threads per block | Registers per thread | Shared memory per block | Blocks per SM | Occupancy | ms at 4096 | % of peak FLOPS | Global reads | Global reads per second |
|---|---|---|---|---|---|---|---|---|---|
| naive rows | 1024 | 32 | 0 | 2 | 100% | 2617.6 | 0.5% | 549.8 GB | 210 GB/s |
| naive cols | 1024 | 32 | 0 | 2 | 100% | 297.8 | 4.1% | 549.8 GB | 1846 GB/s |
| tiled 16 | 256 | 27 | 2 KB | 8 | 100% | 98.8 | 12.3% | 34.4 GB | 348 GB/s |
| tiled 32 | 1024 | 28 | 8 KB | 2 | 100% | 83.6 | 14.5% | 17.2 GB | 206 GB/s |
| 1d, 64 x 64 blocks, 8 per thread | 512 | 64 | 4 KB | 2 | 50% | 54.3 | 22.3% | 8.6 GB | 158 GB/s |
| 2d, 128 x 128 blocks, 8 x 8 per thread | 256 | 120 | 8 KB | 2 | 25% | 38.1 | 31.8% | 4.3 GB | 113 GB/s |
| `cublasSgemm` | | | | | | 15.0 | 80.6% | | |

None of them spill registers. "Global reads" is what the design asks for: 2 floats per multiply-add for the naive kernels, `2 / T` for tiles of `T`, and `1 / BM + 1 / BN` for blocks of `BM x BN`. It isn't what reaches DRAM, which only a counter could tell.

- The naive cols kernel asks for 1846 GB/s, almost 4 times what DRAM can give, so the caches serve most of its reads. A warp's 32 threads share one row of `A`, so they all read the same float of it, and their 32 floats of `B` sit next to each other in memory. The rows kernel asks for the same bytes at a ninth of the speed, because its warps read 32 floats a whole row apart.
- Tiled 16 needs 348 GB/s, 72% of DRAM, so it's close to memory bound. Tiled 32 halves the bytes but is only 18% faster. Its limit is the 2 shared memory loads per multiply-add.
- Register blocking cuts the shared memory loads per multiply-add from 2 to 1.125 (1d) and 0.25 (2d), and the time falls with them, while occupancy falls from 100% to 50% and 25%. A 2d thread keeps 64 values of `C` plus 16 of `A` and `B` in its 120 registers, so only 512 threads fit on an SM. Few warps are enough when each one has 64 independent multiply-adds per step to keep the cores busy, so occupancy isn't what limits these kernels.
- cuBLAS is 2.5 times faster than the 2d kernel. The usual next steps (Boehm's article) are vectorized 128-bit loads, storing the `A` tile transposed so a thread reads its 8 values in one go, and double buffering the tiles. The counters would show which of these matters on this GPU.

### Linear layer

`kernels/linear.cu` does `y = x @ W + b` and its backward, `dx += dy @ W^T`, `dW += x^T @ dy` and `db +=` the column sums of `dy`. All three matmuls use the 2d blocked kernel. It now takes flags for an `A` or `B` stored transposed, so `W^T` and `x^T` are read in place, and at the end it can add a bias or add into what `C` already holds. `db` is a column sum with 32 x 8 threads per 32 columns. The same layer is also built from `cublasSgemm`, for comparison.

It is checked three ways:

- `tests/make_linear_fixture.py` runs one forward and backward through macrograd's `linear` and saves the numbers. The GPU results are within 2.0e-7 of macrograd's, for both versions.
- Eleven shapes against a CPU version in double, every value checked: one row, one input or one output, shapes that don't fill a 128 block or a step of 8 along the inner dimension, and the MNIST shapes. The gradients start at random values, so the test sees that they are added to.
- Nothing is written in the 1024 floats after the end of any output.

| Rows x in -> out | Blocked forward | Blocked backward | cuBLAS forward | cuBLAS backward |
|---|---|---|---|---|
| 64 x 784 -> 128 | 0.255 ms (50 GFLOPS) | 0.095 ms (270) | 0.014 ms (925) | 0.014 ms (1864) |
| 1024 x 784 -> 128 | 0.392 ms (525) | 0.601 ms (684) | 0.042 ms (4914) | 0.087 ms (4715) |
| 1024 x 1024 -> 1024 | 1.21 ms (1771) | 2.34 ms (1832) | 0.250 ms (8582) | 0.486 ms (8833) |
| 4096 x 4096 -> 4096 | 37.6 ms (3656) | 76.7 ms (3582) | 12.7 ms (10846) | 25.0 ms (10997) |

The backward does two matmuls of the same size as the forward, so it counts twice the flops. A first run was 10% to 28% slower at the three smaller sizes for both versions, cuBLAS included; the table is from the second run.

- At 4096 the blocked layer gets the same speed as the plain blocked matmul, 3.6 TFLOPS forward and backward, about a third of cuBLAS. Reading `W` and `x` transposed in place costs nothing measurable.
- At MNIST sizes it is 18 times slower than cuBLAS going forward. A batch of 64 with 128 outputs is a single 128 x 128 block, so one of the 28 SMs does all of it and walks through the 784 inputs alone. The backward is faster than the forward here because its two matmuls have 7 blocks each. cuBLAS picks smaller tiles or splits the inner dimension for shapes like these. The training loop will need one of those.
- The loads of a transposed `A` are arranged so neighbouring threads read neighbouring floats. Loading it the same way as a plain `A` gave the same times (76.8 ms against 76.7 at 4096), so that arrangement doesn't matter here.

I put 15 deliberate bugs into copies of the file and 12 fail the test. The three that pass are the same kinds that passed for the blocked matmul: one removes the column check when loading `B`, which only changes values of `C` that are never written; one removes the `__syncthreads` at the end of the main loop, a race that didn't show on this GPU; and one is the load arrangement above, which only changes speed.

### Cross entropy

`kernels/cross_entropy.cu` is softmax plus negative log likelihood with integer labels, the mean over rows, and its backward `dloss / R * (softmax(x) - one_hot(label))`, added into `dx`. The forward goes over each row once: the online max and sum from the softmax, then `log(d) + (m - x[label])`, where `m - x` stays exact even for logits near 1000. It saves `m` and `1 / d` for each row, so the backward makes the softmax without another pass. A second small kernel takes the mean in a fixed order, so the loss is the same on every run. A label outside the row gives a NaN loss.

Rows are given to threads three ways: a thread per row, a warp per row, or a block of 256 per row. Same 2^24 logits as different shapes. Forward GB/s counts 4 bytes per logit and backward 12 (read `x` and `dx`, write `dx`):

| Rows x classes | Thread per row | Warp per row | Block per row | Backward |
|---|---|---|---|---|
| 1677721 x 10 | 1.644 ms, 40.8 GB/s | 3.427 ms, 19.6 | 21.09 ms, 3.2 | 0.622 ms, 323.9 |
| 524288 x 32 | 1.215 ms, 55.2 | 0.937 ms, 71.6 | 6.548 ms, 10.2 | 0.578 ms, 348.4 |
| 262144 x 64 | 1.613 ms, 41.6 | 0.523 ms, 128.3 | 3.416 ms, 19.6 | 0.570 ms, 353.5 |
| 65536 x 256 | 3.566 ms, 18.8 | 0.237 ms, 283.0 | 1.176 ms, 57.1 | 0.573 ms, 351.4 |
| 16384 x 1024 | 1.442 ms, 46.5 | 0.202 ms, 331.6 | 0.364 ms, 184.2 | 0.599 ms, 336.2 |
| 4096 x 4096 | 1.411 ms, 47.5 | 0.227 ms, 296.1 | 0.223 ms, 300.6 | 0.585 ms, 344.4 |
| 333 x 50257 | 16.64 ms, 4.0 | 0.534 ms, 125.3 | 0.266 ms, 251.8 | 0.567 ms, 354.4 |

`cudaMemcpy` device to device does 355.8 GB/s. `cross_entropy()` uses a thread per row up to 16 classes, a warp per row up to 1024, and a block per row above that.

- A warp per row is the best from 32 to 1024 classes and close to copy speed at 1024. At 10 classes it's the slowest of the useful versions: 22 of the 32 lanes have nothing, and combining two (max, sum) pairs at each of the 5 shuffle steps costs two `exp` calls, which is more work than the row itself. A thread per row is twice as fast there.
- At 50257 classes there are only 333 rows, so a warp per row runs 333 warps, about 12 per SM. A block per row has 8 times as many threads on the job and is twice as fast.
- The 10-class forward is still far from copy speed. The mean is part of it: one block adds 1.68M row losses, which took 0.43 ms of the 1.64 alone. At an MNIST batch of 64 rows the forward kernel takes 2.8 µs and the mean 3.7 µs, so both are about the cost of a launch.
- The backward runs at 324 to 354 GB/s at every shape, since it is one thread per logit.

The test checks the backward formula against finite differences of the CPU loss (4.9e-11), and 71 shapes against a CPU version in double, with every version: 1 to 2500 rows and 1 to 5000 classes, around a warp, a block and the 1024 switch. Rows are uniform in [-10, 10], some shifted by +1000 or -1000 and some with every fourth logit at -inf, and the labels include the first and last column. The largest error was 3.6e-7 for the loss, relative to `1 + |m|`, and 5.0e-8 for `dx`. I put 18 deliberate bugs into copies of the file and 17 fail the test. The one that passes removes the `__syncthreads` between the steps of the mean's tree sum, a race that didn't show on this GPU.

### SGD and Adam

`kernels/optim.cu` has one-thread-per-weight steps for SGD with momentum and weight decay, and for Adam and AdamW (Adam is AdamW with no weight decay), with the same formulas as macrograd's `optim.py`. The CPU works out the numbers that only depend on the settings and the step, like Adam's bias corrections `1 / (1 - beta^t)`, in double once per step and passes them in. `1 - beta2` is one of those numbers. 0.999 isn't exact in float, and `1.0f - 0.999f` is 0.00099998713, 1.3e-5 away from 0.001. Computed on the GPU, it moved every Adam step by about 6e-6 of its size, 100 times a float rounding, and the test caught it.

- Five steps of each optimizer against macrograd on the same weights and gradients, with one gradient of 1e-7 so Adam's `eps` matters (`tests/make_optim_fixture.py`): largest differences 9.7e-8 for SGD, 1.0e-7 for Adam and 1.3e-7 for AdamW.
- One step against a CPU version in double from random weights, gradients and optimizer state, on 1 to 2^20 + 3 weights and at Adam steps 1, 2 and 1000: largest error 1.3e-7 for SGD and 2.3e-7 for AdamW, relative to the size of what went into each value.
- Nothing is written past the end.

| Weights | SGD | AdamW |
|---|---|---|
| 101,770 (the MNIST MLP) | 0.0031 ms, 665 GB/s | 0.0046 ms, 624 GB/s |
| 2^24 | 0.969 ms, 346 GB/s | 1.436 ms, 327 GB/s |

SGD moves 20 bytes per weight (reads the weight, gradient and velocity, writes the weight and velocity) and AdamW 28. `cudaMemcpy` does 357 GB/s. At 2^24 weights the steps run at 97% and 92% of copy speed, so there is little left to gain. The MNIST-sized steps go faster than memory allows because the 2 to 3 MB they touch stays in the 2.75 MB L2 cache between repeats. In a training loop the forward and backward run in between and push it out.

I put 16 deliberate bugs into copies of the file and 15 fail the test. The one that passes reads `m[i]` back right after writing it, which gives the same number. Two bugs that update one weight past the end passed at first: the memory there was NaN, and an update of NaN writes NaN back. The test now fills the space after the end with 3.0 instead.

### Training an MLP on MNIST

`mnist/train_mlp.cu` trains `linear(784, 128) -> relu -> linear(128, 10) -> cross entropy` on MNIST with the kernels above, all on the GPU. The backward is written out layer by layer, and one SGD launch updates all 101,770 weights, since they sit in one buffer. The images go to the GPU once, and each step gathers its batch there. The kernel files are included whole, each in its own namespace, with `MACROGRAD_NO_MAIN` so their tests are left out, so each file still builds and tests on its own. `python3 mnist/get_mnist.py` downloads the data into `data/` first.

`./build/train_mlp --check` runs the first 20 steps against macrograd: the same model, weights made from a hash of their index (whole numbers times 2^-24 or 2^-22, exact in float and double), batches of 32 in file order, and SGD with lr 0.1 and momentum 0.9 (`tests/make_mnist_fixture.py`). The losses go from 2.2886341 to 1.0357997 on the GPU and to 1.0357998 in macrograd, and the largest difference over the 20 steps is 2.1e-7 with the blocked matmul and 2.3e-7 with cuBLAS.

The full run trains 10 epochs from macrograd's init (uniform in `+-1 / sqrt(inputs)`, biases 0), batch 64, SGD with lr 0.05 and momentum 0.9, once with the blocked matmul and once with cuBLAS from the same starting weights and the same shuffles:

| Matmuls | Time per epoch | Time per step | Test accuracy after 1 / 5 / 10 epochs | Test loss after 10 |
|---|---|---|---|---|
| blocked | 402 ms | 429 µs | 95.73% / 97.38% / 97.67% | 0.084 |
| cuBLAS | 64 ms | 68 µs | 95.80% / 97.54% / 97.91% | 0.079 |

- The two runs drift apart from the first epoch, though they start from the same weights and see the same batches: the sums inside the matmuls are added in a different order, and training makes the small differences grow. Both reach about 97.8%, and 99.7% on the training set, so the model is overfitting by epoch 10.
- The blocked version is 6 times slower, and it's the matmuls. A batch of 64 gives the first layer's forward a single 128 x 128 block, which walks all 784 inputs alone, and its weight gradient 7 blocks, so most of the 28 SMs sit idle (see the linear layer above). cuBLAS picks kernels for shapes like these.
- macrograd takes about 0.7 seconds for one step of 32 images, so an epoch there is about 20 minutes. The GPU does a step of 64 in 68 µs with cuBLAS.

## References

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), Simon Boehm
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), NVIDIA
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Milakov and Gimelshein
- [llm.c](https://github.com/karpathy/llm.c), GPT-2 training in plain C and CUDA
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
