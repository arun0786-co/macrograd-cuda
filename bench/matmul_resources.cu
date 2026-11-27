// what each matmul kernel asks of the GPU, read from the compiled kernels and the CUDA
// occupancy calculator, without the hardware counters Nsight Compute needs (they are
// admin-only on the cluster this was run on). for every kernel:
//
// - registers and shared memory per thread or block, and local memory (register spills)
// - how many of its blocks fit on one SM, and so how many warps the SM can switch between
//   while some wait on memory (the occupancy, out of 64 warps on this GPU)
// - its time at 4096, and from the kernel's design, how many bytes it must read from
//   global memory and how many shared memory loads it makes per multiply-add
//
// the global bytes are what each design asks for, not what reaches DRAM: the naive
// kernels ask for far more than DRAM could deliver in the time, so the caches serve
// most of it.
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 bench/matmul_resources.cu -lcublas -o build/matmul_resources && ./build/matmul_resources

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "../kernels/common.h"

#define MACROGRAD_NO_MAIN
namespace naive {
#include "../kernels/matmul_naive.cu"
}
namespace tiled {
#include "../kernels/matmul_tiled.cu"
}
namespace blocked {
#include "../kernels/matmul_blocked.cu"
}

typedef void (*Launch)(const float *, const float *, float *, int, int, int);

struct Kernel {
    const char *name;
    const void *func;  // the __global__ function, for its attributes
    int threads;       // threads per block
    Launch launch;
    // global floats read per multiply-add, from how the kernel reuses its loads:
    // 2 for the naive kernels (one of A, one of B), 2 / T for tiles of T, and
    // 1 / BN + 1 / BM for blocks of BM x BN
    double global_per_fma;
    // shared memory loads per multiply-add in one thread: 2 for the tiles, (TM + 1) / TM
    // for 1d blocking and (TM + TN) / (TM * TN) for 2d
    double shared_per_fma;
};

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // 128 FP32 cores per SM on Pascal, 2 flops per multiply-add
    double peak_flops = 2.0 * 128 * prop.multiProcessorCount * prop.clockRate * 1e3;
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8;
    printf("gpu: %s, %d SMs, %d max warps per SM, %d KB shared memory per SM, %d registers per SM\n", prop.name,
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor / 32, (int)(prop.sharedMemPerMultiprocessor / 1024),
           prop.regsPerMultiprocessor);
    printf("peak: %.1f TFLOPS fp32 at %.2f GHz, %.0f GB/s memory\n\n", peak_flops / 1e12, prop.clockRate / 1e6,
           peak_bw / 1e9);

    CUBLAS_CHECK(cublasCreate(&naive::g_cublas));
    tiled::g_cublas = blocked::g_cublas = naive::g_cublas;

    Kernel kernels[] = {
        {"naive rows", (const void *)naive::matmul_rows, 32 * 32, naive::launch_rows, 2.0, 0.0},
        {"naive cols", (const void *)naive::matmul_cols, 32 * 32, naive::launch_cols, 2.0, 0.0},
        {"tiled 16", (const void *)tiled::matmul_tiled<16>, 16 * 16, tiled::launch_tiled<16>, 2.0 / 16, 2.0},
        {"tiled 32", (const void *)tiled::matmul_tiled<32>, 32 * 32, tiled::launch_tiled<32>, 2.0 / 32, 2.0},
        {"1d 64x64, TM 8", (const void *)blocked::matmul_1d<64, 64, 8, 8>, 512, blocked::launch_1d, 2.0 / 64, 9.0 / 8},
        {"2d 128x128, 8x8", (const void *)blocked::matmul_2d<128, 128, 8, 8, 8>, 256, blocked::launch_2d, 2.0 / 128,
         16.0 / 64},
    };

    const int n = 4096;
    float *A, *B, *C;
    size_t bytes = (size_t)n * n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    std::vector<float> h((size_t)n * n);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto &v : h) v = u(rng);
    CUDA_CHECK(cudaMemcpy(A, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, h.data(), bytes, cudaMemcpyHostToDevice));

    double fma = (double)n * n * n;
    printf("%-16s %5s %5s %7s %6s %7s %6s %10s %9s %9s %10s %9s\n", "kernel", "thr", "regs", "shared", "spill",
           "blk/SM", "occ", "ms @4096", "TFLOPS", "of peak", "global GB", "GB/s");
    for (auto &k : kernels) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, k.func));
        int per_sm;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.func, k.threads, 0));
        double occupancy = (double)per_sm * k.threads / prop.maxThreadsPerMultiProcessor;
        float ms = blocked::time_ms([&] { k.launch(A, B, C, n, n, n); });
        double global_gb = fma * k.global_per_fma * 4 / 1e9;
        double tflops = 2 * fma / (ms * 1e-3) / 1e12;
        printf("%-16s %5d %5d %6zuB %5zuB %7d %5.0f%% %10.2f %9.2f %8.1f%% %10.1f %9.0f\n", k.name, k.threads,
               attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes, per_sm, 100 * occupancy, ms, tflops,
               100 * tflops * 1e12 / peak_flops, global_gb, global_gb / (ms * 1e-3));
    }
    float ms = blocked::time_ms([&] { blocked::launch_cublas(A, B, C, n, n, n); });
    printf("%-16s %75.2f %9.2f %8.1f%%\n", "cublas", ms, 2 * fma / (ms * 1e-3) / 1e12,
           100 * 2 * fma / (ms * 1e-3) / peak_flops);

    printf("\nshared memory loads per multiply-add: tiled 2, 1d %.3f, 2d %.3f\n", 9.0 / 8, 16.0 / 64);
    CUBLAS_CHECK(cublasDestroy(naive::g_cublas));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return 0;
}
