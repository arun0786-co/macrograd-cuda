// times the two tile sizes of kernels/linear.cu, and the small one with K split into 1 to
// 16 parts, on the mnist shapes and some square and long-K ones. the rule in matmul()'s
// AUTO plan comes from this. build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 bench/matmul_plan_sweep.cu -lcublas -o build/matmul_plan_sweep && ./build/matmul_plan_sweep
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "../kernels/common.h"
#define MACROGRAD_NO_MAIN
namespace lin {
#include "../kernels/linear.cu"
}
using namespace lin;
int main() {
    float *A, *B, *C;
    size_t cap = (size_t)1 << 24;
    CUDA_CHECK(cudaMalloc(&A, cap * 4)); CUDA_CHECK(cudaMalloc(&B, cap * 4)); CUDA_CHECK(cudaMalloc(&C, cap * 4));
    CUDA_CHECK(cudaMemset(A, 0, cap * 4)); CUDA_CHECK(cudaMemset(B, 0, cap * 4));
    int shapes[][3] = {{64,128,784},{784,128,64},{64,784,128},{1024,128,784},{784,128,1024},{1024,784,128},{512,512,512},{1024,1024,1024},{2048,2048,2048},{1024,4096,1024},{4096,4096,4096},{256,256,4096},{128,128,8192}};
    printf("%-18s %8s %8s %8s %8s %8s %8s\n", "M N K", "big", "small1", "small2", "small4", "small8", "small16");
    for (auto &s : shapes) {
        int M = s[0], N = s[1], K = s[2];
        printf("%5d %5d %5d   ", M, N, K);
        float t = time_ms([&] { run<Big, false, false>(A, B, C, nullptr, false, M, N, K, 1); });
        printf(" %8.4f", t);
        for (int sp : {1, 2, 4, 8, 16}) {
            float t2 = time_ms([&] { run<Small, false, false>(A, B, C, nullptr, false, M, N, K, sp); });
            printf(" %8.4f", t2);
        }
        printf("\n");
    }
    int per_big, per_small;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_big, matmul_2d<Big, false, false>, 256, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_small, matmul_2d<Small, false, false>, 256, 0);
    printf("blocks per SM: big %d small %d\n", per_big, per_small);
}
