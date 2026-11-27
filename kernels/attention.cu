// attention forward, the way macrograd's scaled_dot_product_attention computes it, for
// q (BH, T_q, d), k (BH, T_k, d) and v (BH, T_k, d_v), BH heads stored one after another,
// each row by row:
//
//    out = softmax(q @ k^T / sqrt(d) + mask) @ v
//
// with causal, query i may only look at keys up to T_k - T_q + i: the queries are the last
// T_q positions, as when generating with cached keys, and with T_q = T_k it's the usual
// lower triangle. the other scores get -inf, so softmax gives them weight 0.
//
// this is the naive version, three steps that each go through memory:
// 1. S = q @ k^T for each head, a (T_q, T_k) matrix, with the matmul from linear.cu (it
//    reads k transposed in place)
// 2. one kernel scales, masks and takes the softmax of every row of S, in place: one
//    block per row, max, sum, write
// 3. out = P @ v for each head, with the same matmul
// the score matrix takes BH * T_q * T_k floats: 1 GB for 16 heads at T = 4096, and it is
// written and read twice. the fused version (attention_fused.cu) never stores it.
//
// timed against the same three steps with cublasSgemmStridedBatched for 1 and 3, which
// does all the heads in one call.
//
// checked against a CPU version in double and against macrograd on the same numbers
// (tests/attention_macrograd.txt, from tests/make_attention_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/attention.cu -lcublas -o build/attention && ./build/attention

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "common.h"

#ifndef MACROGRAD_NO_MAIN
#define MACROGRAD_NO_MAIN
#define ATTENTION_OWNS_MAIN
#endif
namespace lin {
#include "linear.cu"
}
#ifdef ATTENTION_OWNS_MAIN
#undef MACROGRAD_NO_MAIN
#endif

const int THREADS = 256;

__device__ float block_max(float v) {
    __shared__ float w[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    __syncthreads();
    if (threadIdx.x % 32 == 0) w[threadIdx.x / 32] = v;
    __syncthreads();
    v = threadIdx.x % 32 < THREADS / 32 ? w[threadIdx.x % 32] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float w[THREADS / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) w[threadIdx.x / 32] = v;
    __syncthreads();
    v = threadIdx.x % 32 < THREADS / 32 ? w[threadIdx.x % 32] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// row r of S is query i = r % T_q of head r / T_q. masked scores are left out of the max
// and the sum, and written as exactly 0
__global__ void scale_mask_softmax(float *S, int T_q, int T_k, float scale, bool causal) {
    int i = blockIdx.x % T_q;
    float *row = S + (size_t)blockIdx.x * T_k;
    int last = causal ? T_k - T_q + i : T_k - 1;  // the last key this query sees
    float m = -INFINITY;
    for (int j = threadIdx.x; j <= last; j += THREADS) m = fmaxf(m, row[j] * scale);
    m = block_max(m);
    float s = 0.0f;
    for (int j = threadIdx.x; j <= last; j += THREADS) s += expf(row[j] * scale - m);
    s = block_sum(s);
    for (int j = threadIdx.x; j < T_k; j += THREADS) row[j] = j <= last ? expf(row[j] * scale - m) / s : 0.0f;
}

// S holds BH * T_q * T_k floats of scratch
void attention_naive(const float *q, const float *k, const float *v, float *S, float *out, int BH, int T_q, int T_k,
                     int d, int dv, bool causal) {
    for (int h = 0; h < BH; h++) {
        // S_h (T_q, T_k) = q_h (T_q, d) @ k_h^T, with k_h stored (T_k, d)
        lin::matmul<false, true>(q + (size_t)h * T_q * d, k + (size_t)h * T_k * d, S + (size_t)h * T_q * T_k,
                                 nullptr, false, T_q, T_k, d);
    }
    scale_mask_softmax<<<BH * T_q, THREADS>>>(S, T_q, T_k, 1.0f / sqrtf((float)d), causal);
    for (int h = 0; h < BH; h++) {
        lin::matmul<false, false>(S + (size_t)h * T_q * T_k, v + (size_t)h * T_k * dv, out + (size_t)h * T_q * dv,
                                  nullptr, false, T_q, dv, T_k);
    }
}

// the same with cuBLAS for the two matmuls, all heads in one call each. cuBLAS is
// column-major: a row-major (m, n) matrix is its (n, m) transpose, so S^T = k @ q^T and
// out^T = v^T @ P^T
void attention_cublas(const float *q, const float *k, const float *v, float *S, float *out, int BH, int T_q, int T_k,
                      int d, int dv, bool causal) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(lin::g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, T_k, T_q, d, &one, k, d,
                                           (long long)T_k * d, q, d, (long long)T_q * d, &zero, S, T_k,
                                           (long long)T_q * T_k, BH));
    scale_mask_softmax<<<BH * T_q, THREADS>>>(S, T_q, T_k, 1.0f / sqrtf((float)d), causal);
    CUBLAS_CHECK(cublasSgemmStridedBatched(lin::g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, dv, T_q, T_k, &one, v, dv,
                                           (long long)T_k * dv, S, T_k, (long long)T_q * T_k, &zero, out, dv,
                                           (long long)T_q * dv, BH));
}

typedef void (*Attention)(const float *, const float *, const float *, float *, float *, int, int, int, int, int, bool);
const int VERSIONS = 2;
const char *NAMES[] = {"naive", "cublas"};
const Attention ATTENTIONS[] = {attention_naive, attention_cublas};

// ---- CPU side ----

// out and, for the error, the size of what went into each value: sum_j p_j |v_j|
void cpu_attention(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v, int BH,
                   int T_q, int T_k, int d, int dv, bool causal, std::vector<double> &out, std::vector<double> &size) {
    out.assign((size_t)BH * T_q * dv, 0.0);
    size.assign(out.size(), 0.0);
    std::vector<double> p(T_k);
    for (int h = 0; h < BH; h++) {
        for (int i = 0; i < T_q; i++) {
            int last = causal ? T_k - T_q + i : T_k - 1;
            double m = -INFINITY;
            for (int j = 0; j <= last; j++) {
                double s = 0.0;
                for (int c = 0; c < d; c++) s += (double)q[((size_t)h * T_q + i) * d + c] * k[((size_t)h * T_k + j) * d + c];
                p[j] = s / sqrt((double)d);
                m = fmax(m, p[j]);
            }
            double total = 0.0;
            for (int j = 0; j <= last; j++) total += (p[j] = exp(p[j] - m));
            for (int j = 0; j <= last; j++) {
                for (int c = 0; c < dv; c++) {
                    double vv = v[((size_t)h * T_k + j) * dv + c];
                    out[((size_t)h * T_q + i) * dv + c] += p[j] / total * vv;
                    size[((size_t)h * T_q + i) * dv + c] += p[j] / total * fabs(vv);
                }
            }
        }
    }
}

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
    int iters = ms > 20.0f ? 3 : 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

struct Device {
    float *q, *k, *v, *S, *out;
};

std::vector<float> random_vector(size_t n, std::mt19937 &rng, float range) {
    std::uniform_real_distribution<float> u(-range, range);
    std::vector<float> x(n);
    for (auto &a : x) a = u(rng);
    return x;
}

void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// every version, with and without the mask, against the CPU. q and k in [-2, 2], so
// some rows of scores are spread out and their softmax is close to one-hot
bool check_shape(int BH, int T_q, int T_k, int d, int dv, Device dev, std::mt19937 &rng, double *worst) {
    auto q = random_vector((size_t)BH * T_q * d, rng, 2.0f), k = random_vector((size_t)BH * T_k * d, rng, 2.0f);
    auto v = random_vector((size_t)BH * T_k * dv, rng, 1.0f);
    upload(dev.q, q);
    upload(dev.k, k);
    upload(dev.v, v);
    bool ok = true;
    for (bool causal : {false, true}) {
        std::vector<double> want, size;
        cpu_attention(q, k, v, BH, T_q, T_k, d, dv, causal, want, size);
        for (int ver = 0; ver < VERSIONS; ver++) {
            CUDA_CHECK(cudaMemset(dev.out, 0xff, want.size() * sizeof(float)));
            ATTENTIONS[ver](dev.q, dev.k, dev.v, dev.S, dev.out, BH, T_q, T_k, d, dv, causal);
            CUDA_CHECK(cudaGetLastError());
            std::vector<float> got(want.size());
            CUDA_CHECK(cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double e = 0.0;
            for (size_t i = 0; i < got.size(); i++) {
                double x = fabs(got[i] - want[i]) / fmax(size[i], 1e-30);
                e = fmax(e, std::isnan(x) ? INFINITY : x);
            }
            *worst = fmax(*worst, e);
            // the scores have errors of a few roundings times their size (up to 4 d / sqrt(d)),
            // which exp turns into relative errors of the weights
            double tol = 2e-6 * sqrt((double)d) + 1e-6 * sqrt((double)T_k);
            if (!(e < tol)) {
                printf("  %d heads, %d x %d, d %d, dv %d, %s%s: error %.2e (tol %.2e)\n", BH, T_q, T_k, d, dv,
                       NAMES[ver], causal ? ", causal" : "", e, tol);
                ok = false;
            }
        }
    }
    return ok;
}

bool check_macrograd(const char *path, Device dev) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    int BH, T_q, T_k, d, dv;
    f >> BH >> T_q >> T_k >> d >> dv;
    size_t sizes[] = {(size_t)BH * T_q * d, (size_t)BH * T_k * d, (size_t)BH * T_k * dv, (size_t)BH * T_q * dv,
                      (size_t)BH * T_q * dv};
    std::vector<double> arr[5];
    for (int i = 0; i < 5; i++) {
        arr[i].resize(sizes[i]);
        for (auto &x : arr[i]) f >> x;
    }
    if (!f) {
        printf("macrograd: couldn't read %s\n", path);
        return false;
    }
    auto as_float = [](const std::vector<double> &x) { return std::vector<float>(x.begin(), x.end()); };
    upload(dev.q, as_float(arr[0]));
    upload(dev.k, as_float(arr[1]));
    upload(dev.v, as_float(arr[2]));
    bool ok = true;
    for (int ver = 0; ver < VERSIONS; ver++) {
        double w = 0.0;
        for (int causal = 0; causal < 2; causal++) {
            ATTENTIONS[ver](dev.q, dev.k, dev.v, dev.S, dev.out, BH, T_q, T_k, d, dv, causal);
            std::vector<float> got(sizes[3]);
            CUDA_CHECK(cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < got.size(); i++) {
                double e = fabs(got[i] - arr[3 + causal][i]);
                w = fmax(w, std::isnan(e) ? INFINITY : e);
            }
        }
        bool good = w < 1e-6;
        printf("macrograd (%d heads, %d queries, %d keys), %s: largest difference %.2e, %s\n", BH, T_q, T_k,
               NAMES[ver], w, good ? "ok" : "WRONG");
        ok = ok && good;
    }
    return ok;
}

// the mask has to come before the max: here every later key scores far higher than the
// ones a query may see, about 200 at the end of the row against 0 at its start. a max
// taken over the masked scores too, or before scaling, is then so large that every exp
// it shifts underflows to 0 and the row becomes 0 / 0. with d = 64 the scale is 1/8
bool check_large_scores(Device dev) {
    const int T = 16, d = 64, dv = 4;
    std::vector<float> q((size_t)T * d, 1.0f), k((size_t)T * d), v((size_t)T * dv);
    for (int j = 0; j < T; j++) {
        for (int c = 0; c < d; c++) k[(size_t)j * d + c] = 1.6f * j;  // score of key j: 1.6 j * 64 / 8 = 12.8 j
        for (int c = 0; c < dv; c++) v[(size_t)j * dv + c] = (float)(j + c);
    }
    upload(dev.q, q);
    upload(dev.k, k);
    upload(dev.v, v);
    std::vector<double> want, size;
    cpu_attention(q, k, v, 1, T, T, d, dv, true, want, size);
    bool ok = true;
    for (int ver = 0; ver < VERSIONS; ver++) {
        ATTENTIONS[ver](dev.q, dev.k, dev.v, dev.S, dev.out, 1, T, T, d, dv, true);
        std::vector<float> got(want.size());
        CUDA_CHECK(cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double e = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            double x = fabs(got[i] - want[i]) / fmax(size[i], 1e-30);
            e = fmax(e, std::isnan(x) ? INFINITY : x);
        }
        if (!(e < 1e-5)) {
            printf("  large scores, %s: error %.2e\n", NAMES[ver], e);
            ok = false;
        }
    }
    return ok;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    Device dev;
    const size_t big = (size_t)1 << 24, scores = (size_t)16 * 4096 * 4096;
    CUDA_CHECK(cudaMalloc(&dev.q, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.k, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.v, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.out, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.S, scores * sizeof(float)));
    CUBLAS_CHECK(cublasCreate(&lin::g_cublas));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/attention_macrograd.txt", dev);
    // one query and one key, fewer queries than keys, rows longer than a block of 256,
    // head sizes from 1 to 128
    double worst = 0.0;
    bool shapes_ok = true;
    int shapes[][5] = {{1, 1, 1, 1, 1}, {1, 1, 9, 4, 3}, {3, 7, 7, 8, 8}, {2, 5, 300, 16, 5}, {4, 64, 64, 64, 64},
                       {2, 129, 257, 32, 48}, {1, 300, 1000, 128, 128}, {8, 100, 100, 1, 2}};
    for (auto &s : shapes) {
        shapes_ok = check_shape(s[0], s[1], s[2], s[3], s[4], dev, rng, &worst) && shapes_ok;
    }
    printf("shapes: %s, largest error %.2e\n", shapes_ok ? "ok" : "WRONG", worst);
    bool large_ok = check_large_scores(dev);
    printf("large scores past the mask: %s\n", large_ok ? "ok" : "WRONG");
    ok = ok && shapes_ok && large_ok;

    // 16 heads of size 64 with the causal mask. flops: the two matmuls, 2 T^2 d each;
    // the masked half is still computed
    CUDA_CHECK(cudaMemset(dev.q, 0, big * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.k, 0, big * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.v, 0, big * sizeof(float)));
    printf("\n%-8s %-10s %-12s %-12s %-12s %-12s\n", "T", "version", "ms", "GFLOPS", "scores MB", "softmax ms");
    const int BH = 16, d = 64;
    for (int T : {256, 1024, 2048, 4096}) {
        for (int ver = 0; ver < VERSIONS; ver++) {
            float ms = time_ms([&] { ATTENTIONS[ver](dev.q, dev.k, dev.v, dev.S, dev.out, BH, T, T, d, d, true); });
            float sm = time_ms([&] { scale_mask_softmax<<<BH * T, THREADS>>>(dev.S, T, T, 0.125f, true); });
            printf("%-8d %-10s %-12.3f %-12.1f %-12.1f %-12.3f\n", T, NAMES[ver], ms,
                   4.0 * BH * T * T * d / (ms * 1e-3) / 1e9, (double)BH * T * T * 4 / 1e6, sm);
        }
    }

    CUBLAS_CHECK(cublasDestroy(lin::g_cublas));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
