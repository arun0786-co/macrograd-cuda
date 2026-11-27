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

### Transpose

`kernels/transpose.cu` turns `in (R, C)` into `out (C, R)` three ways, all on 32 x 32 tiles with blocks of 32 x 8 threads that move 4 values each. Naive reads along rows of `in` and writes down columns of `out`. Tiled goes through shared memory, so both the reads and the writes run along rows. Padded is tiled with the tile 33 floats wide, so reading down a column of it touches 32 different banks instead of one. Every value is checked exactly on 12 shapes, from 1 x 1 to 4097 x 3, with nothing written past the end. GB/s counts 8 bytes per value.

| Rows x cols | Naive | Tiled | Padded |
|---|---|---|---|
| 4096 x 4096 | 1.109 ms, 121.1 GB/s | 0.411 ms, 326.8 | 0.388 ms, 346.1 |
| 16384 x 1024 | 1.002 ms, 133.9 | 0.399 ms, 336.3 | 0.392 ms, 342.8 |
| 1024 x 16384 | 0.882 ms, 152.2 | 0.419 ms, 320.2 | 0.425 ms, 315.7 |
| 4000 x 4000 | 0.895 ms, 143.0 | 0.416 ms, 307.7 | 0.410 ms, 312.0 |

`cudaMemcpy` does 356.1 GB/s.

- Going through shared memory makes it 2.1 to 2.7 times faster, and the padded version reaches 97% of copy speed at 4096 x 4096.
- The padding itself does little: 6% at 4096 x 4096, 1 to 2% at the others, and 1% slower at 1024 x 16384. A warp's column read has 32-way bank conflicts without it, but the kernel waits on DRAM, so shared memory has time to spare.

I put 7 deliberate bugs into copies of the file and 6 fail the test. The one that passes removes the column check when loading the tile, which only fills tile slots that go to rows of `out` past the end, which are never written.

### Layer norm

`kernels/layer_norm.cu` is macrograd's `layer_norm` with `gamma` and `beta`: `y = gamma * (x - mean) / sqrt(var + eps) + beta` over each row, and its backward, `dx += rstd * (g - mean(g) - x_hat * mean(g * x_hat))` with `g = dy * gamma`, plus `dgamma` and `dbeta` summed over the rows. Rows go to a warp up to 1024 columns and to a block of 256 above that. The forward saves each row's mean and `1 / std` for the backward. `dgamma` and `dbeta` are column sums: each block takes 32 columns and one part of the rows, and a second kernel adds the parts in a fixed order, so a tall matrix still has enough blocks.

The first version worked out the mean as a float, and for a row near 1000 a float mean is off by up to 3e-5, which every `x - mean` in the row inherited. Now each row has its first value `x_0` subtracted first, which is exact for nearby floats, and the kernels only hold `mean(x - x_0)`. The largest error on `y` went from 2.6e-5 to 4.5e-7.

- Against macrograd on the same numbers, with one constant row (variance 0, so `1 / std` is 316): within 2.3e-5 on values up to 323, which is 7e-8 of their size.
- 46 shapes against a CPU version in double, from 1 x 1 to 20000 x 16 and 1000 x 3000, with rows shifted by 1000 and constant rows, gradients that start at random values, and nothing written past the end of `y`. Largest errors relative to the size of what went in: 4.5e-7 for `y`, 1.4e-7 for `dx`, 2.2e-7 for `dgamma` and 1.4e-7 for `dbeta`.

| Rows x cols | Forward | `dx` | `dgamma`, `dbeta` |
|---|---|---|---|
| 65536 x 256 | 0.962 ms, 139.5 GB/s | 1.681 ms, 159.7 GB/s | 0.440 ms, 304.7 GB/s |
| 16384 x 1024 | 0.762 ms, 176.1 | 1.172 ms, 229.0 | 0.411 ms, 326.6 |
| 4096 x 4096 | 0.648 ms, 207.2 | 1.148 ms, 233.7 | 0.416 ms, 322.6 |
| 1024 x 16384 | 0.773 ms, 173.6 | 1.145 ms, 234.5 | 0.456 ms, 294.1 |

GB/s counts each value once: 8 bytes for the forward (read `x`, write `y`), 16 for `dx` (read `x`, `dy`, `dx`, write `dx`) and 8 for the parameter gradients (read `x` and `dy`). `cudaMemcpy` does 356 GB/s.

- The forward reads each row three times (sum, variance, write) and `dx` twice, so neither gets near copy speed. Keeping the row in registers (a warp's 32 lanes hold 1024 floats in 32 registers each) would read it once.
- The parameter gradients read everything once and run at 83 to 92% of copy speed.

I put 15 deliberate bugs into copies of the file and 14 fail the test. The one that passes uses a warp per row for every width, which is still right, just slower. One more passed at first: without the check for rows past the end, the forward wrote past the end of `y`, which the test didn't look at. It does now.

### Prefix sum (cumsum)

`kernels/cumsum.cu` is macrograd's `cumsum` along the last axis, `out[k] = x[0] + ... + x[k]` on each row, and its backward, the same scan run from the end of the row and added into `dx`. Each thread scans 4 values of its own, the threads' totals are scanned across the block (5 `__shfl_up_sync` steps in each warp, then the same on the warps' totals), and each thread adds the total of everything before it. Two ways to lay it out:

1. Rows: one block per row walks the row 1024 values at a time and carries the total along.
2. Tiles, for a few long rows: one kernel sums each tile of 1024, a second scans the tiles' sums, and a third scans each tile again starting from the sum of the tiles before it. It reads `x` twice, but every SM has work.

`cumsum()` picks rows from 2 rows per SM up, and tiles below that. The test runs every version forward and backward on 13 shapes, from 1 x 1 to 1 x 3000001 and 7 x 100000. With whole numbers from -3 to 3 every partial sum is exact in float, and the GPU matches the CPU bit for bit. With random floats the largest error is 1.2e-7, relative to the sum of `|x|` so far.

| Rows x cols | Rows | Tiles | Backward (auto) |
|---|---|---|---|
| 1 x 16777216 | 22.6 ms, 5.9 GB/s | 0.698 ms, 192.2 GB/s | 0.815 ms, 246.9 GB/s |
| 16 x 1048576 | 1.301 ms, 103.1 | 0.643 ms, 208.8 | 0.793 ms, 253.8 |
| 64 x 262144 | 0.505 ms, 265.7 | 0.644 ms, 208.4 | 0.733 ms, 274.8 |
| 4096 x 4096 | 0.419 ms, 320.3 | 0.680 ms, 197.4 | 0.580 ms, 346.8 |
| 65536 x 256 | 0.496 ms, 270.5 | 1.223 ms, 109.7 | 0.725 ms, 277.6 |

GB/s counts 8 bytes a value for the forward and 12 for the backward, which also reads `dx`. `cub::DeviceScan::InclusiveSum` does one row of 16777216 in 0.380 ms, 353.1 GB/s, and `cudaMemcpy` does 356.0 GB/s.

- One block per row on a single row of 16M puts the whole job on one SM: 22.6 ms. The tiles version spreads it over all of them and is 32 times faster.
- cub is still 1.8 times faster than the tiles on one long row: it reads `x` once. Each block waits for the block before it to publish its total, and looks back over several earlier blocks so the wait is short ("decoupled look-back", Merrill and Garland, 2016).
- With many rows, a block per row runs at 270 to 320 GB/s and the tiles version only adds work.

I put 13 deliberate bugs into copies of the file and 12 fail the test. The one that passes removes the `__syncthreads` before the block scan writes its per-warp totals, so a warp can overwrite a total another warp is still reading from the previous call. That is a race, and it didn't show up on this GPU.

### Attention, naive

`kernels/attention.cu` is macrograd's `scaled_dot_product_attention` forward, `softmax(q @ k^T / sqrt(d) + mask) @ v`, for `BH` heads at once. With `causal`, query `i` sees keys up to `T_k - T_q + i`, as macrograd does when there are fewer queries than keys. It runs in three steps that each go through memory: the score matrix `S = q @ k^T` for each head with the linear layer's matmul (reading `k` transposed in place), one kernel that scales, masks and takes the softmax of each row of `S`, and `P @ v` for each head. The same three steps with `cublasSgemmStridedBatched`, all heads in one call, are there to compare against.

- Against macrograd on 4 heads with 5 queries and 7 keys, with and without the mask: within 6.9e-8.
- 8 shapes against a CPU version in double, from 1 query and 1 key to 300 queries, 1000 keys and heads of 128, with and without the mask: largest error 7.5e-7, relative to `sum_j p_j |v_j|`.
- A row where every key past the mask scores far higher than the visible ones, up to 200 against 0. The softmax's max has to leave the masked scores out, and be taken after scaling. Otherwise every `exp` in the row underflows to 0 and the row is 0 / 0. Two deliberate bugs that did exactly this passed the other tests, since softmax gives the same answer for any shift as long as nothing underflows.

16 heads of size 64 with the causal mask. GFLOPS counts the two matmuls, `4 T^2 d` per head, including the masked half:

| T | Naive | cuBLAS matmuls | Score matrix | Softmax alone |
|---|---|---|---|---|
| 256 | 0.633 ms, 424 GFLOPS | 0.140 ms, 1915 | 4.2 MB | 0.056 ms |
| 1024 | 3.158 ms, 1360 | 1.309 ms, 3281 | 67 MB | 0.335 ms |
| 2048 | 11.85 ms, 1450 | 4.457 ms, 3855 | 268 MB | 1.20 ms |
| 4096 | 43.32 ms, 1586 | 20.69 ms, 3322 | 1074 MB | 6.25 ms |

- The score matrix grows with `T^2`: a GB at 4096 for 16 heads, which is written by the first matmul, read and written by the softmax, and read again by the second.
- The softmax alone is 30% of the cuBLAS version's time at 4096. The rest is the matmuls, and cuBLAS does them at 3.3 to 3.9 TFLOPS, well below its 10 for square matrices: one of the two products has an inner dimension of only 64.
- The naive version launches 2 matmuls per head, and its kernel isn't tuned for an inner dimension of 64 either, so it's 2.1 to 4.5 times slower than cuBLAS.

I put 8 deliberate bugs into copies of the file. Five failed the first tests, and the two above failed once the large-score test was added. The one that passes removes a `__syncthreads` in the block sum, a race that didn't show up on this GPU.

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

### Matmul: 128-bit loads and a transposed A tile

A third version in `kernels/matmul_blocked.cu` keeps the 2d kernel's 128 x 128 blocks and 8 x 8 values per thread, but moves 4 floats at a time: every global load, every shared memory read and the final store are `float4`. The `A` tile is also stored transposed in shared memory, so a thread's 8 values of `A` for one `k` sit next to each other and take 2 reads instead of 8. `float4` needs 16-byte aligned addresses, so it only runs when `K` and `N` are multiples of 4, and falls back to the 2d kernel otherwise. The test has 8 shapes that take the `float4` path, some with `M`, `N` or `K` that don't fill a block or a step of 8.

| Kernel | 256 | 512 | 1024 | 2048 | 4096 | 4096, of cuBLAS |
|---|---|---|---|---|---|---|
| 2d | 262.4 GFLOPS | 933.2 | 1486.1 | 3247.4 | 3615.1 (38.0 ms) | 37.1% |
| 2d, float4 | 570.3 | 2059.9 | 3708.9 | 6622.9 | 6984.9 (19.7 ms) | 71.6% |
| `cublasSgemm` | 3257 | 4410 | 7587 | 9961 | 9749 (14.1 ms) | 100% |

From the second of two runs, which agreed within 2% except cuBLAS at 2048 (2.07 then 1.73 ms). The largest error was 2.9e-7.

- The `float4` version is 1.9 to 2.5 times faster than the 2d one at every size, and reaches 72% of cuBLAS at 4096. That's a lot for the same arithmetic, which says the 2d kernel was limited by the number of load instructions, not by memory or math: per `k` a thread now issues 4 shared memory reads instead of 16.
- It is still slow at 256, where 4 blocks of 128 x 128 leave 24 SMs idle. The split-K layouts from the linear layer would help there.

I put 13 deliberate bugs into the new kernel and 9 fail the test. Two of the four that pass remove the bounds check on the rows of `A` or the columns of `B`, which only changes values of `C` that are never written, as in the other matmuls. One removes the `__syncthreads` at the end of the loop, a race that didn't show up. One gives the threads a different but complete mapping of the `B` tile, which isn't a bug.

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
- Tiled 16 needs 348 GB/s, 72% of DRAM, so it's close to memory bound. Tiled 32 halves the bytes but is only 18% faster, which points at something else: both read 2 floats from shared memory for every multiply-add.
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

### Matmul: small shapes, split K

The 2d kernel's 128 x 128 blocks are too big for small batches. For a batch of 64 with 128 outputs the first MNIST layer is one block, so one of the 28 SMs does the whole forward. `kernels/linear.cu` now picks a layout by shape. It uses 64 x 64 blocks (4 x 4 values per thread) unless the matmul is big. If those still give fewer than 2 blocks per SM, it also splits `K` into parts: separate blocks work out each part's sum into a scratch buffer, and a second kernel adds the parts in a fixed order and then the bias or what `C` held. `bench/matmul_plan_sweep.cu` timed the choices (ms, `M x N x K`, `C = A @ B`):

| M x N x K | 128 x 128 | 64 x 64 | 64 x 64, 2 parts | 4 parts | 8 parts | 16 parts |
|---|---|---|---|---|---|---|
| 64 x 128 x 784 | 0.321 | 0.147 | 0.078 | 0.043 | 0.026 | 0.018 |
| 784 x 128 x 64 | 0.047 | 0.017 | 0.018 | 0.025 | 0.038 | 0.038 |
| 64 x 784 x 128 | 0.064 | 0.028 | 0.019 | 0.017 | 0.025 | 0.037 |
| 1024 x 128 x 784 | 0.468 | 0.200 | 0.123 | 0.130 | 0.130 | 0.156 |
| 1024 x 784 x 128 | 0.150 | 0.098 | 0.132 | 0.166 | 0.258 | 0.462 |
| 512 x 512 x 512 | 0.259 | 0.118 | 0.129 | 0.135 | 0.154 | 0.200 |
| 1024 x 1024 x 1024 | 1.175 | 0.867 | 0.828 | 0.788 | 0.827 | 1.018 |
| 2048 x 2048 x 2048 | 5.214 | 5.538 | 5.588 | 5.805 | 6.319 | 6.108 |
| 1024 x 4096 x 1024 | 2.663 | 2.915 | 3.072 | 3.360 | 3.593 | 3.926 |
| 4096 x 4096 x 4096 | 38.28 | 43.08 | 43.29 | 44.51 | 47.09 | 52.39 |
| 256 x 256 x 4096 | 1.855 | 0.810 | 0.417 | 0.227 | 0.240 | 0.253 |
| 128 x 128 x 8192 | 3.691 | 1.641 | 0.834 | 0.430 | 0.242 | 0.137 |

- The 128 x 128 blocks only win from 256 of them up. Two of them fit on an SM, and at 1024 x 1024 x 1024 the 64 blocks fill the GPU once and then leave most of it idle for a second round. So they're used from 8 blocks per SM (224) up.
- Splitting `K` pays off when the 64 x 64 blocks are fewer than 2 per SM: 128 x 128 x 8192 goes from 1.64 ms to 0.14 ms with 16 parts. Splitting a matmul that already has enough blocks makes it slower, since every part writes a whole `M x N` partial sum. The rule picks enough parts to make 2 blocks per SM, each part at least 32 long.

The layer now runs at these speeds (ms; the fixed column is the 128 x 128 kernel from before):

| Rows x in -> out | Fixed forward | Forward | Fixed backward | Backward | cuBLAS forward | cuBLAS backward |
|---|---|---|---|---|---|---|
| 64 x 784 -> 128 | 0.332 | 0.018 | 0.123 | 0.039 | 0.018 | 0.017 |
| 64 x 128 -> 10 | 0.068 | 0.010 | 0.074 | 0.022 | 0.007 | 0.011 |
| 1024 x 784 -> 128 | 0.480 | 0.119 | 0.714 | 0.233 | 0.047 | 0.096 |
| 1024 x 1024 -> 1024 | 1.280 | 0.881 | 2.432 | 1.931 | 0.250 | 0.484 |
| 4096 x 4096 -> 4096 | 38.57 | 38.63 | 78.21 | 78.41 | 12.75 | 24.74 |

The first MNIST layer's forward is 18 times faster and level with cuBLAS. Training MNIST with these kernels went from 402 ms per epoch to 85.5 ms (91 µs per step), against 64 ms with cuBLAS. The check against macrograd still passes (1.8e-7), and test accuracy after 10 epochs is 97.75%.

The test now runs every shape three ways: the fixed kernel, the automatic choice, and a forced split into up to 4 parts with 64 x 64 blocks, so the split code is checked on every shape, not only the ones that pick it. I put 11 deliberate bugs into the new code and 5 fail the test. The other 6 don't change any result. Two load up to `K` instead of the end of the block's part, but parts are always whole tile steps long, so no tile crosses one. One doesn't round the parts to whole tile steps, and one doesn't recount the parts, which only makes uneven or empty parts, and an empty part adds 0. One runs the bias step on split launches, which get no bias. One changes the load loop's stride so some floats are loaded twice, with the same values.

### MNIST: a CUDA graph for the training step

A training step is about 15 launches. A CUDA graph records the launches of one step once, and then the whole step goes to the GPU in one call. A recorded graph replays the same arguments every time, so the batch number lives on the GPU: the gather kernel reads it and a one-thread kernel moves it on at the end of the step. Recording needs a stream other than the legacy default one, so `mnist/train_mlp.cu` now builds with `--default-stream per-thread` and gives cuBLAS the same stream. The graph gives exactly the same test loss as plain launches, which the program checks.

| Matmuls | Plain launches | CUDA graph |
|---|---|---|
| blocked | 85.0 ms per epoch | 84.0 ms |
| cuBLAS | 63.8 ms | 66.9 ms |

(epoch 10 of each run; the first epoch of a run with plain launches is up to 20% slower.)

The graph saves 1% with the blocked matmuls and costs 5% with cuBLAS. The CPU wasn't what the step waited for: it queues launches faster than the GPU runs them. Each part of a step timed alone on the GPU (µs, batch 64):

| Part | Blocked | cuBLAS |
|---|---|---|
| gather the batch | 3.7 | 3.6 |
| layer 1 forward | 18.2 | 17.4 |
| relu | 1.7 | 1.8 |
| layer 2 forward | 10.3 | 7.1 |
| loss (2 kernels) | 7.4 | 7.7 |
| zero the gradients (4 memsets) and loss backward | 11.4 | 11.3 |
| layer 2 backward and relu backward | 23.6 | 12.6 |
| layer 1 backward | 21.2 | 10.8 |
| SGD | 3.1 | 3.1 |
| sum | 100.8 | 75.3 |

The parts add up to a bit more than a real step (90 and 68 µs), since each was timed alone. A kernel that does almost nothing still takes 2 to 4 µs on the GPU itself: the relu on 8192 values takes 1.7, and the 4 memsets and one small kernel take 11.4. So at this size the step is set by how many kernels it has, which a graph doesn't change. Doing more in each kernel would: relu inside the first layer's matmul, the loss and its backward in one kernel, and gradients that the first write overwrites instead of adding to zeroed memory.

## References

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM), Simon Boehm
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), NVIDIA
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Milakov and Gimelshein
- [llm.c](https://github.com/karpathy/llm.c), GPT-2 training in plain C and CUDA
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
