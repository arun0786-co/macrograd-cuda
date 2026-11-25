// matmul, tiled with shared memory: C = A @ B with A (M, K), B (K, N), C (M, N), all
// stored row by row, one thread per value of C as in matmul_naive.cu.
//
// in the naive version every thread reads a whole row of A and a whole column of B from
// global memory, and the 32 x 32 threads of a block read the same 32 rows of A and the
// same 32 columns of B over and over: 2 loads for every multiply-add. here the block
// walks along K one T x T tile at a time:
//
//    1. every thread loads one float of the A tile and one of the B tile into shared memory
//    2. __syncthreads, so the whole tile is there before anyone reads it
//    3. every thread does T multiply-adds from shared memory: its row of the A tile
//       against its column of the B tile
//    4. __syncthreads again, so nobody overwrites the tile while others still read it
//
// so each float loaded from global memory is used T times, once by every thread in its row
// (or column) of the block. values past the edge of A or B are loaded as 0, which adds
// nothing to the sum, so any M, N, K works. the adds happen in the same order as in the
// naive version, k = 0, 1, 2, ...
//
// in step 3 the 32 threads of a warp have the same row and neighbouring columns: they all
// read the same float of the A tile, which shared memory sends to all of them at once, and
// 32 neighbouring floats of the B tile, which sit in 32 different banks. so neither read
// has bank conflicts.
//
// T = 16 and T = 32 are both timed, against the naive column version and cublasSgemm.
//
// build and run: nvcc -O3 -arch=native kernels/matmul_tiled.cu -lcublas -o run && ./run

#include <cmath>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "common.h"

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "%s:%d cublas error %d\n", __FILE__, __LINE__, (int)st); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

const int NAIVE = 32;  // the naive version's blocks are 32 x 32 threads

// the naive column version from matmul_naive.cu, for comparison
__global__ void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int j = blockIdx.x * NAIVE + threadIdx.x;
    int i = blockIdx.y * NAIVE + threadIdx.y;
    if (i < M && j < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
        }
        C[(size_t)i * N + j] = acc;
    }
}

// blocks of T x T threads, each block makes a T x T piece of C
template <int T>
__global__ void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.y * T + ty;  // row of C
    int j = blockIdx.x * T + tx;  // column of C
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        // this thread's float of each tile: A[i][k0 + tx] and B[k0 + ty][j]. neighbouring
        // threads load neighbouring floats of a row, so both loads are coalesced
        As[ty][tx] = i < M && k0 + tx < K ? A[(size_t)i * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = k0 + ty < K && j < N ? B[(size_t)(k0 + ty) * N + j] : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    // threads past the edge still had to load their part of the tiles, so they only stop here
    if (i < M && j < N) {
        C[(size_t)i * N + j] = acc;
    }
}

// ---- launches ----

cublasHandle_t g_cublas;

void launch_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 grid((N + NAIVE - 1) / NAIVE, (M + NAIVE - 1) / NAIVE);
    matmul_naive<<<grid, dim3(NAIVE, NAIVE)>>>(A, B, C, M, N, K);
}

template <int T>
void launch_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);
    matmul_tiled<T><<<grid, dim3(T, T)>>>(A, B, C, M, N, K);
}

// cuBLAS stores matrices column by column. a row-major (M, N) matrix is the same bytes as
// a column-major (N, M) one, its transpose, so ask cuBLAS for C^T = B^T @ A^T
void launch_cublas(const float *A, const float *B, float *C, int M, int N, int K) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
}

typedef void (*Matmul)(const float *, const float *, float *, int, int, int);
const int KERNELS = 4;
const char *NAMES[] = {"naive", "tiled16", "tiled32", "cublas"};
const Matmul MATMULS[] = {launch_naive, launch_tiled<16>, launch_tiled<32>, launch_cublas};

template <typename F>
float time_ms(F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    int iters = ms > 50.0f ? 3 : ms > 5.0f ? 10 : 50;  // slow kernels get fewer repeats
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

// ---- CPU side ----

// one value of C in double, and sum_k |A[i][k] * B[k][j]|, the size of what went into it.
// the error is measured against that, since a float sum of K terms can only be as exact
// as its largest terms allow, even when they cancel to something small
void ref_entry(const std::vector<float> &A, const std::vector<float> &B, int N, int K, int i, int j,
               double *value, double *scale) {
    double v = 0.0, s = 0.0;
    for (int k = 0; k < K; k++) {
        double p = (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
        v += p;
        s += fabs(p);
    }
    *value = v;
    *scale = s;
}

// checks every value of C when there are at most `full` of them, otherwise a fixed sample
// of 4096 plus the four corners. nan counts as infinitely wrong
double max_error(const std::vector<float> &A, const std::vector<float> &B, const std::vector<float> &C,
                 int M, int N, int K, std::mt19937 &rng) {
    std::vector<std::pair<int, int>> where;
    if ((size_t)M * N <= (1 << 20)) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                where.push_back({i, j});
            }
        }
    } else {
        where = {{0, 0}, {0, N - 1}, {M - 1, 0}, {M - 1, N - 1}};
        std::uniform_int_distribution<int> ri(0, M - 1), rj(0, N - 1);
        for (int s = 0; s < 4096; s++) {
            where.push_back({ri(rng), rj(rng)});
        }
    }
    double worst = 0.0;
    for (auto [i, j] : where) {
        double v, s;
        ref_entry(A, B, N, K, i, j, &v, &s);
        double e = fabs(C[(size_t)i * N + j] - v) / fmax(s, 1e-30);
        if (std::isnan(e)) {
            e = INFINITY;
        }
        worst = fmax(worst, e);
    }
    return worst;
}

const int MAX_DIM = 4096;  // the buffers hold up to MAX_DIM x MAX_DIM floats

struct Buffers {
    float *A, *B, *C;
};

// runs every version on one shape, checks it, and with timing on prints a row each
bool run_shape(int M, int N, int K, Buffers d, std::mt19937 &rng, bool timing) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (auto &v : A) v = uniform(rng);
    for (auto &v : B) v = uniform(rng);
    // everything past the end of A and B is nan, so a kernel that reads past the edge gets
    // nan * 0 = nan instead of a leftover number times 0, and fails the check
    CUDA_CHECK(cudaMemset(d.A, 0xff, (size_t)MAX_DIM * MAX_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.B, 0xff, (size_t)MAX_DIM * MAX_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d.A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    bool ok = true;
    float cublas_ms = 0.0f;
    for (int k = KERNELS - 1; k >= 0; k--) {  // cublas first, so the others can be compared to it
        CUDA_CHECK(cudaMemset(d.C, 0xff, C.size() * sizeof(float)));  // nan, so a missed write shows
        MATMULS[k](d.A, d.B, d.C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C.data(), d.C, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double err = max_error(A, B, C, M, N, K, rng);
        // a float sum of K terms is off by up to about K roundings in the worst case, but
        // the roundings mostly cancel, so a few times sqrt(K) of them is plenty
        double tol = 8.0 * sqrt((double)K) * 6e-8;
        bool good = err < tol;
        ok = ok && good;
        if (timing) {
            float ms = time_ms([&] { MATMULS[k](d.A, d.B, d.C, M, N, K); });
            if (k == KERNELS - 1) {
                cublas_ms = ms;
            }
            printf("  %-8s %-10.2e %-6s %-10.3f %-9.1f %5.1f%%\n", NAMES[k], err, good ? "ok" : "WRONG", ms,
                   2.0 * M * N * K / (ms * 1e-3) / 1e9, 100.0 * cublas_ms / ms);
        } else if (!good) {
            printf("  shape %d x %d x %d, %s: error %.2e (tol %.2e)\n", M, N, K, NAMES[k], err, tol);
        }
    }
    return ok;
}

int main() {
    Buffers d;
    CUDA_CHECK(cudaMalloc(&d.A, (size_t)MAX_DIM * MAX_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.B, (size_t)MAX_DIM * MAX_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.C, (size_t)MAX_DIM * MAX_DIM * sizeof(float)));
    CUBLAS_CHECK(cublasCreate(&g_cublas));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    // shapes that don't fill the last block, thin ones, and K = 1
    bool small_ok = true;
    int shapes[][3] = {{1, 1, 1}, {1, 1, 100}, {7, 5, 3}, {31, 33, 1}, {33, 31, 64}, {32, 32, 32},
                       {100, 1, 50}, {1, 100, 50}, {257, 129, 65}, {1000, 777, 513}};
    for (auto &s : shapes) {
        small_ok = run_shape(s[0], s[1], s[2], d, rng, false) && small_ok;
    }
    printf("small shapes: %s\n", small_ok ? "ok" : "WRONG");
    bool ok = small_ok;

    // square sizes. GFLOPS counts one multiply and one add per (i, j, k): 2 M N K
    for (int n : {256, 512, 1024, 2048, 4096}) {
        printf("\n%d x %d x %d\n", n, n, n);
        printf("  %-8s %-10s %-6s %-10s %-9s %s\n", "kernel", "max error", "check", "ms", "GFLOPS",
               "of cublas");
        ok = run_shape(n, n, n, d, rng, true) && ok;
    }

    CUBLAS_CHECK(cublasDestroy(g_cublas));
    CUDA_CHECK(cudaFree(d.A));
    CUDA_CHECK(cudaFree(d.B));
    CUDA_CHECK(cudaFree(d.C));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
