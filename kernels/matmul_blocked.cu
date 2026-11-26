// matmul with register blocking: C = A @ B with A (M, K), B (K, N), C (M, N), all
// stored row by row.
//
// in the tiled version (matmul_tiled.cu) each thread makes one value of C, and every
// multiply-add reads two floats from shared memory. shared memory is fast, but not as fast
// as registers, and the multiply-add itself is one instruction, so the loads cost more than
// the math. here each thread makes several values of C and keeps them in registers, so a
// float read from shared memory feeds several multiply-adds:
//
// 1. 1d blocking: blocks of 64 x 64 values of C, 512 threads, each thread makes TM = 8
//    values down one column. for every k it reads one float of the B tile into a register
//    and uses it for all 8 rows: 9 shared memory reads for 8 multiply-adds instead of 16.
// 2. 2d blocking: blocks of 128 x 128 values of C, 256 threads, each thread makes an
//    8 x 8 square. for every k it reads 8 floats of the A tile and 8 of the B tile into
//    registers and does all 64 multiply-adds between them: 16 reads for 64 multiply-adds.
//
// both walk along K BK = 8 columns at a time. the tiles are loaded into shared memory by
// all the threads together, values past the edge of A or B loaded as 0, so any M, N, K
// works, and the adds happen in the same order as in the other versions, k = 0, 1, 2, ...
// the sizes are the ones Simon Boehm uses for these two steps (see the references).
//
// timed against the 32 x 32 tiled version and cublasSgemm.
//
// build and run: nvcc -O3 -arch=native kernels/matmul_blocked.cu -lcublas -o run && ./run

#include <algorithm>
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

// the 32 x 32 tiled version from matmul_tiled.cu, for comparison
const int T = 32;

__global__ void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.y * T + ty;
    int j = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = i < M && k0 + tx < K ? A[(size_t)i * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = k0 + ty < K && j < N ? B[(size_t)(k0 + ty) * N + j] : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (i < M && j < N) {
        C[(size_t)i * N + j] = acc;
    }
}

// loads the (BM x BK) tile of A starting at (row0, k0) and the (BK x BN) tile of B starting
// at (k0, col0) into shared memory, `threads` threads together, 0 past the edges.
// neighbouring threads load neighbouring floats of a row
template <int BM, int BN, int BK>
__device__ void load_tiles(const float *A, const float *B, float *As, float *Bs, int M, int N, int K,
                           int row0, int col0, int k0, int threads) {
    for (int idx = threadIdx.x; idx < BM * BK; idx += threads) {
        int r = row0 + idx / BK, c = k0 + idx % BK;
        As[idx] = r < M && c < K ? A[(size_t)r * K + c] : 0.0f;
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += threads) {
        int r = k0 + idx / BN, c = col0 + idx % BN;
        Bs[idx] = r < K && c < N ? B[(size_t)r * N + c] : 0.0f;
    }
}

// ---- 1. 1d blocking: TM values down a column per thread ----

// __launch_bounds__(threads, 2) tells the compiler to fit 2 blocks on an SM. left alone
// it uses 68 registers a thread, and 512 threads x 68 is more than half the SM's 65,536
// registers, so only one block fits and the SM has 512 threads to switch between while
// they wait on shared memory. capped at 64 (no spills) two blocks fit, and at 4096 it goes
// from 1460 to 2561 GFLOPS
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__(BM / TM * BN, 2) matmul_1d(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];  // As[r * BK + k]
    __shared__ float Bs[BK * BN];  // Bs[k * BN + c]
    const int threads = BM / TM * BN;
    // this thread's column of the block, and which group of TM rows
    int tcol = threadIdx.x % BN, trow = threadIdx.x / BN;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tiles<BM, BN, BK>(A, B, As, Bs, M, N, K, row0, col0, k0, threads);
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            // one read of B, used for all TM rows. the 32 threads of a warp have
            // neighbouring columns, so this read has no bank conflicts, and they all
            // read the same float of A, which shared memory sends to all of them at once
            float b = Bs[k * BN + tcol];
            for (int t = 0; t < TM; t++) {
                acc[t] += As[(trow * TM + t) * BK + k] * b;
            }
        }
        __syncthreads();
    }
    for (int t = 0; t < TM; t++) {
        int r = row0 + trow * TM + t, c = col0 + tcol;
        if (r < M && c < N) {
            C[(size_t)r * N + c] = acc[t];
        }
    }
}

// ---- 2. 2d blocking: a TM x TN square per thread ----

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_2d(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int threads = (BM / TM) * (BN / TN);
    // this thread's square starts at row trow * TM and column tcol * TN of the block
    int tcol = threadIdx.x % (BN / TN), trow = threadIdx.x / (BN / TN);
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tiles<BM, BN, BK>(A, B, As, Bs, M, N, K, row0, col0, k0, threads);
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            // TM + TN reads from shared memory, then TM * TN multiply-adds from registers.
            // the loops have fixed sizes, so the compiler unrolls them and a, b and acc
            // stay in registers
            for (int i = 0; i < TM; i++) {
                a[i] = As[(trow * TM + i) * BK + k];
            }
            for (int j = 0; j < TN; j++) {
                b[j] = Bs[k * BN + tcol * TN + j];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int r = row0 + trow * TM + i, c = col0 + tcol * TN + j;
            if (r < M && c < N) {
                C[(size_t)r * N + c] = acc[i][j];
            }
        }
    }
}

// ---- launches ----

cublasHandle_t g_cublas;

void launch_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);
    matmul_tiled<<<grid, dim3(T, T)>>>(A, B, C, M, N, K);
}

void launch_1d(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_1d<BM, BN, BK, TM><<<grid, BM / TM * BN>>>(A, B, C, M, N, K);
}

void launch_2d(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_2d<BM, BN, BK, TM, TN><<<grid, (BM / TM) * (BN / TN)>>>(A, B, C, M, N, K);
}

// cuBLAS stores matrices column by column. a row-major (M, N) matrix is the same bytes as
// a column-major (N, M) one, its transpose, so ask cuBLAS for C^T = B^T @ A^T
void launch_cublas(const float *A, const float *B, float *C, int M, int N, int K) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
}

typedef void (*Matmul)(const float *, const float *, float *, int, int, int);
const int KERNELS = 4;
const char *NAMES[] = {"tiled", "1d", "2d", "cublas"};
const Matmul MATMULS[] = {launch_tiled, launch_1d, launch_2d, launch_cublas};

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
    // up to 128 rows after the end of C, where a block that forgets its row check would write
    std::vector<float> past(std::min((size_t)128 * N, (size_t)MAX_DIM * MAX_DIM - C.size()));
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
        CUDA_CHECK(cudaMemset(d.C, 0xff, (C.size() + past.size()) * sizeof(float)));  // nan, so a missed write shows
        MATMULS[k](d.A, d.B, d.C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(C.data(), d.C, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool inside = true;  // still nan after the end of C
        if (!past.empty()) {
            CUDA_CHECK(cudaMemcpy(past.data(), d.C + C.size(), past.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (float v : past) {
                inside = inside && std::isnan(v);
            }
        }
        double err = max_error(A, B, C, M, N, K, rng);
        // a float sum of K terms is off by up to about K roundings in the worst case, but
        // the roundings mostly cancel, so a few times sqrt(K) of them is plenty
        double tol = 8.0 * sqrt((double)K) * 6e-8;
        bool good = err < tol && inside;
        ok = ok && good;
        if (timing) {
            float ms = time_ms([&] { MATMULS[k](d.A, d.B, d.C, M, N, K); });
            if (k == KERNELS - 1) {
                cublas_ms = ms;
            }
            printf("  %-8s %-10.2e %-6s %-10.3f %-9.1f %5.1f%%\n", NAMES[k], err, good ? "ok" : "WRONG", ms,
                   2.0 * M * N * K / (ms * 1e-3) / 1e9, 100.0 * cublas_ms / ms);
        } else if (!good) {
            printf("  shape %d x %d x %d, %s: error %.2e (tol %.2e)%s\n", M, N, K, NAMES[k], err, tol,
                   inside ? "" : ", wrote past the end of C");
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

    // shapes that don't fill the last block, thin ones, K = 1, and shapes around the
    // 64 and 128 blocks and BK = 8
    bool small_ok = true;
    int shapes[][3] = {{1, 1, 1}, {1, 1, 100}, {7, 5, 3}, {31, 33, 1}, {33, 31, 64}, {32, 32, 32},
                       {100, 1, 50}, {1, 100, 50}, {257, 129, 65}, {1000, 777, 513},
                       {64, 64, 8}, {65, 63, 9}, {128, 128, 7}, {129, 127, 17}, {200, 300, 8}};
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
