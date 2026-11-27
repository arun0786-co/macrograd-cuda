// attention forward in one kernel that never writes the score matrix to memory, the idea
// of flash attention (Dao et al., 2022): softmax(q @ k^T / sqrt(d) + mask) @ v with the
// same shapes and causal mask as attention.cu, for head sizes d = d_v of 16, 32, 64 or 128.
//
// each block takes BR = 64 queries of one head, one thread per query. the thread keeps
// its query row and its output row in registers and walks the keys a tile of BC at a
// time. the block loads the tile's keys and values into shared memory together, then each
// thread:
// 1. works out its BC scores, keeping them in shared memory, and their max
// 2. moves its running max m to max(m, tile max), scaling its output row and running sum
//    l by exp(m_old - m_new): everything added so far was weighted relative to m_old
// 3. adds exp(s_j - m) * v_j for the tile's keys to the output row and exp(s_j - m) to l
// at the end the output row is divided by l. this is the online softmax from the softmax
// kernels, carried through the product with v. with causal, a tile whose keys are all
// past every query of the block is skipped, so about half the work goes.
//
// the score matrix of the naive version is BH * T_q * T_k floats; here nothing but the
// output goes to memory, and k and v are read once per block of 64 queries.
//
// checked against the CPU version from attention.cu and timed against both versions there.
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/attention_fused.cu -lcublas -o build/attention_fused && ./build/attention_fused

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "common.h"

#ifndef MACROGRAD_NO_MAIN
#define MACROGRAD_NO_MAIN
#define FUSED_OWNS_MAIN
#endif
namespace naive {
#include "attention.cu"
}
#ifdef FUSED_OWNS_MAIN
#undef MACROGRAD_NO_MAIN
#endif

const int BR = 64;  // queries per block, one thread each

// keys per tile: 64 for small heads, 32 from d = 64 up, so the tiles of k and v and the
// scores fit in the 48 KB of shared memory a block can have (at d = 64 with 64 keys
// they'd take 49,408 bytes)
template <int D>
__host__ __device__ constexpr int tile_keys() {
    return D <= 32 ? 64 : 32;
}

template <int D>
__global__ void __launch_bounds__(BR) attention_fused_kernel(const float *q, const float *k, const float *v,
                                                             float *out, int T_q, int T_k, float scale, bool causal) {
    constexpr int BC = tile_keys<D>();
    __shared__ float ks[BC][D], vs[BC][D];
    __shared__ float s[BR][BC + 1];  // one row of scores per thread, padded off one bank
    int h = blockIdx.y, i0 = blockIdx.x * BR, i = i0 + threadIdx.x;
    const float *qh = q + (size_t)h * T_q * D, *kh = k + (size_t)h * T_k * D, *vh = v + (size_t)h * T_k * D;
    bool active = i < T_q;
    int shift = T_k - T_q;
    // the last key this query sees, and the last any query of the block sees
    int last = causal ? shift + i : T_k - 1;
    int block_last = causal ? min(T_k - 1, shift + i0 + BR - 1) : T_k - 1;

    float qr[D], o[D];
    for (int c = 0; c < D; c++) {
        qr[c] = active ? qh[(size_t)i * D + c] * scale : 0.0f;  // scaled once here
        o[c] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;

    for (int j0 = 0; j0 <= block_last; j0 += BC) {
        // the tile's keys and values, BR threads loading BC * D floats of each, 0 past the end
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int j = j0 + idx / D, c = idx % D;
            ks[idx / D][c] = j < T_k ? kh[(size_t)j * D + c] : 0.0f;
            vs[idx / D][c] = j < T_k ? vh[(size_t)j * D + c] : 0.0f;
        }
        __syncthreads();
        // 1. scores and their max. a key this query can't see scores -inf
        float tile_max = -INFINITY;
        for (int jj = 0; jj < BC; jj++) {
            int j = j0 + jj;
            float sc = -INFINITY;
            if (active && j <= last && j < T_k) {
                sc = 0.0f;
                for (int c = 0; c < D; c++) sc += qr[c] * ks[jj][c];
            }
            s[threadIdx.x][jj] = sc;
            tile_max = fmaxf(tile_max, sc);
        }
        // 2 and 3, only if this tile has a key this query sees: otherwise exp(-inf - -inf)
        // would be nan
        if (tile_max > -INFINITY) {
            float m_new = fmaxf(m, tile_max);
            float shrink = expf(m - m_new);  // 0 on the first tile, where m = -inf
            l *= shrink;
            for (int c = 0; c < D; c++) o[c] *= shrink;
            for (int jj = 0; jj < BC; jj++) {
                float p = expf(s[threadIdx.x][jj] - m_new);  // 0 for -inf
                l += p;
                for (int c = 0; c < D; c++) o[c] += p * vs[jj][c];
            }
            m = m_new;
        }
        __syncthreads();  // before the next tile overwrites ks and vs
    }
    if (active) {
        float *oh = out + ((size_t)h * T_q + i) * D;
        for (int c = 0; c < D; c++) oh[c] = o[c] / l;
    }
}

template <int D>
void launch_fused(const float *q, const float *k, const float *v, float *out, int BH, int T_q, int T_k, bool causal) {
    dim3 grid((T_q + BR - 1) / BR, BH);
    attention_fused_kernel<D><<<grid, BR>>>(q, k, v, out, T_q, T_k, 1.0f / sqrtf((float)D), causal);
}

// same arguments as the versions in attention.cu; S isn't used. d must equal dv
void attention_fused(const float *q, const float *k, const float *v, float *, float *out, int BH, int T_q, int T_k,
                     int d, int dv, bool causal) {
    if (d != dv) {
        fprintf(stderr, "attention_fused needs d == dv, got %d and %d\n", d, dv);
        exit(1);
    }
    switch (d) {
        case 16: launch_fused<16>(q, k, v, out, BH, T_q, T_k, causal); break;
        case 32: launch_fused<32>(q, k, v, out, BH, T_q, T_k, causal); break;
        case 64: launch_fused<64>(q, k, v, out, BH, T_q, T_k, causal); break;
        case 128: launch_fused<128>(q, k, v, out, BH, T_q, T_k, causal); break;
        default: fprintf(stderr, "attention_fused has head sizes 16, 32, 64 and 128, not %d\n", d); exit(1);
    }
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    naive::Device dev;
    const size_t big = (size_t)1 << 24, scores = (size_t)16 * 4096 * 4096;
    CUDA_CHECK(cudaMalloc(&dev.q, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.k, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.v, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.out, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.S, scores * sizeof(float)));
    CUBLAS_CHECK(cublasCreate(&naive::lin::g_cublas));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    // every head size, with and without the mask: one query, fewer queries than keys,
    // sizes that don't fill a block of 64 queries or a tile of keys, and long rows
    bool ok = true;
    double worst = 0.0;
    int shapes[][4] = {{1, 1, 1, 16}, {2, 1, 9, 32}, {3, 7, 7, 64}, {2, 5, 300, 16}, {4, 64, 64, 64},
                       {2, 129, 257, 32}, {1, 300, 1000, 128}, {2, 65, 65, 128}, {1, 1000, 1000, 64}};
    for (auto &sh : shapes) {
        int BH = sh[0], T_q = sh[1], T_k = sh[2], d = sh[3];
        auto q = naive::random_vector((size_t)BH * T_q * d, rng, 2.0f), k = naive::random_vector((size_t)BH * T_k * d, rng, 2.0f);
        auto v = naive::random_vector((size_t)BH * T_k * d, rng, 1.0f);
        naive::upload(dev.q, q);
        naive::upload(dev.k, k);
        naive::upload(dev.v, v);
        for (bool causal : {false, true}) {
            std::vector<double> want, size;
            naive::cpu_attention(q, k, v, BH, T_q, T_k, d, d, causal, want, size);
            CUDA_CHECK(cudaMemset(dev.out, 0xff, (want.size() + 4096) * sizeof(float)));
            attention_fused(dev.q, dev.k, dev.v, nullptr, dev.out, BH, T_q, T_k, d, d, causal);
            CUDA_CHECK(cudaGetLastError());
            std::vector<float> got(want.size() + 4096);
            CUDA_CHECK(cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double e = 0.0;
            for (size_t n = 0; n < want.size(); n++) {
                double x = fabs(got[n] - want[n]) / fmax(size[n], 1e-30);
                e = fmax(e, std::isnan(x) ? INFINITY : x);
            }
            bool past = true;
            for (size_t n = want.size(); n < got.size(); n++) past = past && std::isnan(got[n]);
            worst = fmax(worst, e);
            double tol = 2e-6 * sqrt((double)d) + 1e-6 * sqrt((double)T_k);
            if (!(e < tol) || !past) {
                printf("  %d heads, %d x %d, d %d%s: error %.2e (tol %.2e)%s\n", BH, T_q, T_k, d,
                       causal ? ", causal" : "", e, tol, past ? "" : ", wrote past the end");
                ok = false;
            }
        }
    }
    printf("shapes: %s, largest error %.2e\n", ok ? "ok" : "WRONG", worst);

    // the large-score row from attention.cu: the mask has to come before the max
    {
        const int T = 16, d = 64;
        std::vector<float> q((size_t)T * d, 1.0f), k((size_t)T * d), v((size_t)T * d);
        for (int j = 0; j < T; j++) {
            for (int c = 0; c < d; c++) {
                k[(size_t)j * d + c] = 1.6f * j;
                v[(size_t)j * d + c] = (float)(j + c % 4);
            }
        }
        naive::upload(dev.q, q);
        naive::upload(dev.k, k);
        naive::upload(dev.v, v);
        std::vector<double> want, size;
        naive::cpu_attention(q, k, v, 1, T, T, d, d, true, want, size);
        attention_fused(dev.q, dev.k, dev.v, nullptr, dev.out, 1, T, T, d, d, true);
        std::vector<float> got(want.size());
        CUDA_CHECK(cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double e = 0.0;
        for (size_t n = 0; n < got.size(); n++) {
            double x = fabs(got[n] - want[n]) / fmax(size[n], 1e-30);
            e = fmax(e, std::isnan(x) ? INFINITY : x);
        }
        bool good = e < 1e-5;
        printf("large scores past the mask: %s\n", good ? "ok" : "WRONG");
        ok = ok && good;
    }

    // 16 heads, causal, against the two versions from attention.cu
    CUDA_CHECK(cudaMemset(dev.q, 0, big * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.k, 0, big * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.v, 0, big * sizeof(float)));
    printf("\n%-6s %-6s %-12s %-12s %-12s %-14s\n", "d", "T", "naive ms", "cublas ms", "fused ms", "scores MB");
    const int BH = 16;
    for (int d : {32, 64, 128}) {
        for (int T : {256, 1024, 2048, 4096}) {
            float a = naive::time_ms([&] { naive::attention_naive(dev.q, dev.k, dev.v, dev.S, dev.out, BH, T, T, d, d, true); });
            float b = naive::time_ms([&] { naive::attention_cublas(dev.q, dev.k, dev.v, dev.S, dev.out, BH, T, T, d, d, true); });
            float c = naive::time_ms([&] { attention_fused(dev.q, dev.k, dev.v, nullptr, dev.out, BH, T, T, d, d, true); });
            printf("%-6d %-6d %-12.3f %-12.3f %-12.3f %-14.1f\n", d, T, a, b, c, (double)BH * T * T * 4 / 1e6);
        }
    }
    cudaFuncAttributes attr;
    for (int d : {32, 64, 128}) {
        const void *f = d == 32 ? (const void *)attention_fused_kernel<32> : d == 64 ? (const void *)attention_fused_kernel<64>
                                                                                     : (const void *)attention_fused_kernel<128>;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, f));
        int per_sm;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, f, BR, 0));
        printf("d %d: %d registers, %zu B spilled, %zu B shared, %d blocks of %d per SM\n", d, attr.numRegs,
               attr.localSizeBytes, attr.sharedSizeBytes, per_sm, BR);
    }

    CUBLAS_CHECK(cublasDestroy(naive::lin::g_cublas));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
