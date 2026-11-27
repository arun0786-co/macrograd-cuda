// a linear layer, y = x @ W + b, and its backward, for x (R, I), W (I, O), b (O,) and
// y (R, O), all stored row by row (R rows of the batch, I inputs, O outputs).
//
// with dy = dL/dy, the gradient that arrives at y:
//   dx = dy @ W^T     each input feeds every output through its row of W
//   dW = x^T @ dy     W[i][o] meets x[r][i] once in every row r
//   db = sum of dy over the rows
// the backward adds into dx, dW and db instead of overwriting them, like the elementwise
// kernels, since a value used twice gets its gradient from both places.
//
// all three matmuls use the 2d register blocked kernel from matmul_blocked.cu, the fastest
// one here, with two changes:
// 1. TA and TB say that A or B is stored transposed. dx needs W^T and dW needs x^T, and
//    reading them in place saves writing a transposed copy first. the loads then step down
//    the stored matrix so that neighbouring threads still read neighbouring floats
// 2. at the end each value of C can add a bias (forward) or add into what C already holds
//    (backward)
// db is a column sum, done by its own kernel.
//
// checked against a CPU version in double and against macrograd's linear on the same
// numbers (tests/linear_macrograd.txt, from tests/make_linear_fixture.py), and timed
// against the same passes done with cublasSgemm.
//
// build and run from the repo root:
//   nvcc -O3 -arch=native kernels/linear.cu -lcublas -o build/linear && ./build/linear

#include <algorithm>
#include <cmath>
#include <fstream>
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

// ---- the matmul: C (M, N) = A (M, K) @ B (K, N) ----

const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
const int THREADS = (BM / TM) * (BN / TN);

// A(r, c) is A[r * K + c], or A[c * M + r] when A is stored transposed, as a (K, M) matrix.
// B(r, c) is B[r * N + c], or B[c * K + r] when B is stored as (N, K).
// the tile index idx is split so that neighbouring threads load neighbouring floats of
// the stored matrix, down a row of it either way
template <bool TA, bool TB>
__device__ void load_tiles(const float *A, const float *B, float *As, float *Bs, int M, int N, int K,
                           int row0, int col0, int k0) {
    for (int idx = threadIdx.x; idx < BM * BK; idx += THREADS) {
        int r = TA ? idx % BM : idx / BK, c = TA ? idx / BM : idx % BK;
        int gr = row0 + r, gc = k0 + c;
        float v = 0.0f;
        if (gr < M && gc < K) {
            v = TA ? A[(size_t)gc * M + gr] : A[(size_t)gr * K + gc];
        }
        As[r * BK + c] = v;
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += THREADS) {
        int r = TB ? idx % BK : idx / BN, c = TB ? idx / BK : idx % BN;
        int gr = k0 + r, gc = col0 + c;
        float v = 0.0f;
        if (gr < K && gc < N) {
            v = TB ? B[(size_t)gc * K + gr] : B[(size_t)gr * N + gc];
        }
        Bs[r * BN + c] = v;
    }
}

// each thread makes a TM x TN square of C in registers, as in matmul_blocked.cu. then
//   C = A @ B + bias          when bias is given (one value per column)
//   C = C + A @ B             when accumulate is set
template <bool TA, bool TB>
__global__ void matmul_2d(const float *A, const float *B, float *C, const float *bias, bool accumulate,
                          int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    int tcol = threadIdx.x % (BN / TN), trow = threadIdx.x / (BN / TN);
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    float a[TM], b[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tiles<TA, TB>(A, B, As, Bs, M, N, K, row0, col0, k0);
        __syncthreads();
        for (int k = 0; k < BK; k++) {
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
                float v = acc[i][j];
                if (bias) {
                    v += bias[c];
                }
                if (accumulate) {
                    v += C[(size_t)r * N + c];
                }
                C[(size_t)r * N + c] = v;
            }
        }
    }
}

template <bool TA, bool TB>
void matmul(const float *A, const float *B, float *C, const float *bias, bool accumulate, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_2d<TA, TB><<<grid, THREADS>>>(A, B, C, bias, accumulate, M, N, K);
}

// ---- db: column sums ----

// db[o] += sum over r of dy[r][o]. a block covers 32 columns with 32 x 8 threads: the 32
// threads of a warp take neighbouring columns, so each row is read in one go, and the 8
// warps take every 8th row. then the 8 partial sums of each column are added in shared
// memory, always in the same order, so the result doesn't change from run to run
__global__ void column_sum(const float *dy, float *db, int R, int O) {
    __shared__ float part[8][32];
    int c = blockIdx.x * 32 + threadIdx.x;
    float s = 0.0f;
    if (c < O) {
        for (int r = threadIdx.y; r < R; r += 8) {
            s += dy[(size_t)r * O + c];
        }
    }
    part[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && c < O) {
        float total = 0.0f;
        for (int w = 0; w < 8; w++) {
            total += part[w][threadIdx.x];
        }
        db[c] += total;
    }
}

void launch_column_sum(const float *dy, float *db, int R, int O) {
    column_sum<<<(O + 31) / 32, dim3(32, 8)>>>(dy, db, R, O);
}

// ---- the layer ----

// y = x @ W + b
void linear_forward(const float *x, const float *W, const float *b, float *y, int R, int I, int O) {
    matmul<false, false>(x, W, y, b, false, R, O, I);
}

// dx += dy @ W^T, dW += x^T @ dy, db += column sums of dy. dx can be null when the
// input needs no gradient, like the images going into a first layer
void linear_backward(const float *x, const float *W, const float *dy, float *dx, float *dW, float *db,
                     int R, int I, int O) {
    // dy is (R, O) and W^T is W (I, O) read as (O, I)
    if (dx) {
        matmul<false, true>(dy, W, dx, nullptr, true, R, I, O);
    }
    // x^T is x (R, I) read as (I, R), and dy is (R, O)
    matmul<true, false>(x, dy, dW, nullptr, true, I, O, R);
    launch_column_sum(dy, db, R, O);
}

// the same with cublasSgemm. cuBLAS stores matrices column by column, so a row-major
// (m, n) matrix is the column-major (n, m) one, its transpose: ask for C^T = B^T @ A^T

cublasHandle_t g_cublas;

__global__ void add_bias(float *y, const float *b, int R, int O) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)R * O) {
        y[i] += b[i % O];
    }
}

void cublas_forward(const float *x, const float *W, const float *b, float *y, int R, int I, int O) {
    const float one = 1.0f, zero = 0.0f;
    // y^T (O, R) = W^T (O, I) @ x^T (I, R)
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, O, R, I, &one, W, O, x, I, &zero, y, O));
    add_bias<<<(unsigned)(((size_t)R * O + 255) / 256), 256>>>(y, b, R, O);
}

void cublas_backward(const float *x, const float *W, const float *dy, float *dx, float *dW, float *db,
                     int R, int I, int O) {
    const float one = 1.0f;
    // dx^T (I, R) += W (I, O) @ dy^T (O, R). the stored W is W^T in column-major, so transpose it
    if (dx) {
        CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, I, R, O, &one, W, O, dy, O, &one, dx, I));
    }
    // dW^T (O, I) += dy^T (O, R) @ x (R, I). the stored x is x^T in column-major, so transpose it
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, O, I, R, &one, dy, O, x, I, &one, dW, O));
    launch_column_sum(dy, db, R, O);
}

typedef void (*Forward)(const float *, const float *, const float *, float *, int, int, int);
typedef void (*Backward)(const float *, const float *, const float *, float *, float *, float *, int, int, int);
const int VERSIONS = 2;
const char *NAMES[] = {"blocked", "cublas"};
const Forward FORWARDS[] = {linear_forward, cublas_forward};
const Backward BACKWARDS[] = {linear_backward, cublas_backward};

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
    int iters = ms > 50.0f ? 3 : ms > 5.0f ? 10 : 100;
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

// what the GPU should hold after a forward and a backward, in double, with the size of
// what went into each value: the sum of |term| over everything added. the error is
// measured against that, since a float sum is only as exact as its largest terms allow
struct Expected {
    std::vector<double> y, dx, dW, db, y_s, dx_s, dW_s, db_s;
};

Expected reference(const std::vector<float> &x, const std::vector<float> &W, const std::vector<float> &b,
                   const std::vector<float> &dy, const std::vector<float> &dx0, const std::vector<float> &dW0,
                   const std::vector<float> &db0, int R, int I, int O) {
    Expected e;
    e.y.assign((size_t)R * O, 0.0);
    e.y_s.assign((size_t)R * O, 0.0);
    e.dx.assign(dx0.begin(), dx0.end());
    e.dW.assign(dW0.begin(), dW0.end());
    e.db.assign(db0.begin(), db0.end());
    for (size_t n = 0; n < e.dx.size(); n++) e.dx_s.push_back(fabs(e.dx[n]));
    for (size_t n = 0; n < e.dW.size(); n++) e.dW_s.push_back(fabs(e.dW[n]));
    for (size_t n = 0; n < e.db.size(); n++) e.db_s.push_back(fabs(e.db[n]));
    for (int r = 0; r < R; r++) {
        for (int o = 0; o < O; o++) {
            size_t ro = (size_t)r * O + o;
            e.y[ro] = b[o];
            e.y_s[ro] = fabs(b[o]);
            e.db[o] += dy[ro];
            e.db_s[o] += fabs(dy[ro]);
            for (int i = 0; i < I; i++) {
                size_t ri = (size_t)r * I + i, io = (size_t)i * O + o;
                double fwd = (double)x[ri] * W[io];
                e.y[ro] += fwd;
                e.y_s[ro] += fabs(fwd);
                double gx = (double)dy[ro] * W[io];
                e.dx[ri] += gx;
                e.dx_s[ri] += fabs(gx);
                double gw = (double)x[ri] * dy[ro];
                e.dW[io] += gw;
                e.dW_s[io] += fabs(gw);
            }
        }
    }
    return e;
}

double max_error(const std::vector<float> &got, const std::vector<double> &want, const std::vector<double> &scale) {
    double worst = 0.0;
    for (size_t n = 0; n < got.size(); n++) {
        double e = fabs(got[n] - want[n]) / fmax(scale[n], 1e-30);
        worst = fmax(worst, std::isnan(e) ? INFINITY : e);
    }
    return worst;
}

const size_t CAP = (size_t)1 << 24;  // every buffer holds this many floats
const size_t PAST = 1024;            // floats after each output checked for stray writes

struct Buffers {
    float *x, *W, *b, *dy, *y, *dx, *dW, *db;
};

// fills the buffer with nan, then copies v to its start, so a read past the end of v
// gets nan and a write past it can be seen
void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemset(d, 0xff, CAP * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// copies n floats back, and checks the PAST floats after them are still nan
std::vector<float> download(const float *d, size_t n, bool *untouched) {
    std::vector<float> v(n + PAST);
    CUDA_CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t k = n; k < v.size(); k++) {
        *untouched = *untouched && std::isnan(v[k]);
    }
    v.resize(n);
    return v;
}

std::vector<float> random_vector(size_t n, std::mt19937 &rng) {
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto &a : v) a = uniform(rng);
    return v;
}

// runs a forward and a backward with every version on one shape and checks all four
// outputs. the gradients start at random values, so the check sees that they're added to
bool check_shape(int R, int I, int O, Buffers d, std::mt19937 &rng) {
    auto x = random_vector((size_t)R * I, rng), W = random_vector((size_t)I * O, rng);
    auto b = random_vector(O, rng), dy = random_vector((size_t)R * O, rng);
    auto dx0 = random_vector((size_t)R * I, rng), dW0 = random_vector((size_t)I * O, rng);
    auto db0 = random_vector(O, rng);
    Expected e = reference(x, W, b, dy, dx0, dW0, db0, R, I, O);
    // the largest sum has max(I, O, R) terms, plus one for the bias or starting value
    double tol = 8.0 * sqrt((double)std::max({I, O, R}) + 1) * 6e-8;
    bool ok = true;
    for (int v = 0; v < VERSIONS; v++) {
        upload(d.x, x);
        upload(d.W, W);
        upload(d.b, b);
        upload(d.dy, dy);
        upload(d.dx, dx0);
        upload(d.dW, dW0);
        upload(d.db, db0);
        CUDA_CHECK(cudaMemset(d.y, 0xff, CAP * sizeof(float)));
        FORWARDS[v](d.x, d.W, d.b, d.y, R, I, O);
        BACKWARDS[v](d.x, d.W, d.dy, d.dx, d.dW, d.db, R, I, O);
        CUDA_CHECK(cudaGetLastError());
        bool untouched = true;
        double errs[4] = {
            max_error(download(d.y, (size_t)R * O, &untouched), e.y, e.y_s),
            max_error(download(d.dx, (size_t)R * I, &untouched), e.dx, e.dx_s),
            max_error(download(d.dW, (size_t)I * O, &untouched), e.dW, e.dW_s),
            max_error(download(d.db, O, &untouched), e.db, e.db_s),
        };
        const char *what[] = {"y", "dx", "dW", "db"};
        for (int k = 0; k < 4; k++) {
            if (!(errs[k] < tol)) {
                printf("  %d x %d -> %d, %s: %s error %.2e (tol %.2e)\n", R, I, O, NAMES[v], what[k], errs[k], tol);
                ok = false;
            }
        }
        if (!untouched) {
            printf("  %d x %d -> %d, %s: wrote past the end of an output\n", R, I, O, NAMES[v]);
            ok = false;
        }
    }
    return ok;
}

// the same numbers through macrograd's linear (pure python, in double): the GPU's float
// results should match them to float precision
bool check_macrograd(const char *path, Buffers d) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    int R, I, O;
    f >> R >> I >> O;
    size_t sizes[] = {(size_t)R * I, (size_t)I * O, (size_t)O, (size_t)R * O,
                      (size_t)R * O, (size_t)R * I, (size_t)I * O, (size_t)O};
    std::vector<double> arr[8];
    for (int k = 0; k < 8; k++) {
        arr[k].resize(sizes[k]);
        for (auto &v : arr[k]) f >> v;
    }
    if (!f) {
        printf("macrograd: couldn't read %s\n", path);
        return false;
    }
    auto as_float = [](const std::vector<double> &v) { return std::vector<float>(v.begin(), v.end()); };
    bool ok = true;
    for (int v = 0; v < VERSIONS; v++) {
        upload(d.x, as_float(arr[0]));
        upload(d.W, as_float(arr[1]));
        upload(d.b, as_float(arr[2]));
        upload(d.dy, as_float(arr[3]));
        upload(d.dx, std::vector<float>(sizes[5], 0.0f));  // macrograd's grads start at 0
        upload(d.dW, std::vector<float>(sizes[6], 0.0f));
        upload(d.db, std::vector<float>(sizes[7], 0.0f));
        FORWARDS[v](d.x, d.W, d.b, d.y, R, I, O);
        BACKWARDS[v](d.x, d.W, d.dy, d.dx, d.dW, d.db, R, I, O);
        CUDA_CHECK(cudaGetLastError());
        float *outs[] = {d.y, d.dx, d.dW, d.db};
        double worst = 0.0;
        for (int k = 0; k < 4; k++) {
            bool untouched = true;
            std::vector<float> got = download(outs[k], sizes[4 + k], &untouched);
            for (size_t n = 0; n < got.size(); n++) {
                double e = fabs(got[n] - arr[4 + k][n]);
                worst = fmax(worst, std::isnan(e) ? INFINITY : e);
            }
        }
        // the inputs are about 1 and at most 10 terms are added, so a few float roundings
        bool good = worst < 1e-6;
        printf("macrograd (%d x %d -> %d), %s: largest difference %.2e, %s\n", R, I, O, NAMES[v], worst,
               good ? "ok" : "WRONG");
        ok = ok && good;
    }
    return ok;
}

// the test. a program that includes this file for its kernels defines
// MACROGRAD_NO_MAIN first, like mnist/train_mlp.cu
#ifndef MACROGRAD_NO_MAIN
int main() {
    Buffers d;
    float **all[] = {&d.x, &d.W, &d.b, &d.dy, &d.y, &d.dx, &d.dW, &d.db};
    for (auto p : all) {
        CUDA_CHECK(cudaMalloc(p, CAP * sizeof(float)));
    }
    CUBLAS_CHECK(cublasCreate(&g_cublas));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/linear_macrograd.txt", d);

    // one row, one input or one output, shapes that don't fill a 128 block or an 8 step
    // of the inner dimension, and the mnist shapes. every value is checked
    int shapes[][3] = {{1, 1, 1}, {1, 5, 3}, {7, 1, 9}, {9, 7, 1}, {3, 100, 2}, {64, 784, 128},
                       {64, 128, 10}, {129, 127, 9}, {200, 33, 257}, {8, 8, 8}, {130, 17, 300}};
    bool shapes_ok = true;
    for (auto &s : shapes) {
        shapes_ok = check_shape(s[0], s[1], s[2], d, rng) && shapes_ok;
    }
    printf("shapes: %s\n", shapes_ok ? "ok" : "WRONG");
    ok = ok && shapes_ok;

    // timing. each pass is matmuls of 2 R I O flops: one forward, two backward
    printf("\n%-20s %-8s %-11s %-9s %-11s %-9s\n", "rows x in -> out", "version", "forward ms", "GFLOPS",
           "backward ms", "GFLOPS");
    int timed[][3] = {{64, 784, 128}, {1024, 784, 128}, {1024, 1024, 1024}, {4096, 4096, 4096}};
    for (auto &s : timed) {
        int R = s[0], I = s[1], O = s[2];
        double flops = 2.0 * R * I * O;
        for (int v = 0; v < VERSIONS; v++) {
            float f_ms = time_ms([&] { FORWARDS[v](d.x, d.W, d.b, d.y, R, I, O); });
            float b_ms = time_ms([&] { BACKWARDS[v](d.x, d.W, d.dy, d.dx, d.dW, d.db, R, I, O); });
            char shape[32];
            snprintf(shape, sizeof shape, "%d x %d -> %d", R, I, O);
            printf("%-20s %-8s %-11.4f %-9.1f %-11.4f %-9.1f\n", v == 0 ? shape : "", NAMES[v], f_ms,
                   flops / (f_ms * 1e-3) / 1e9, b_ms, 2 * flops / (b_ms * 1e-3) / 1e9);
        }
    }

    CUBLAS_CHECK(cublasDestroy(g_cublas));
    for (auto p : all) {
        CUDA_CHECK(cudaFree(*p));
    }
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
