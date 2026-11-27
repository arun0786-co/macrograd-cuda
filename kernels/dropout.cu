// dropout, like macrograd's nn.dropout: each value is kept with probability 1 - p and
// scaled by 1 / (1 - p), or set to 0, so the expected output is the input:
//
//    out = x * keep / (1 - p),   dx += dy * keep / (1 - p)
//
// macrograd keeps a value when random() > p; here random() is u = (top 24 bits of a
// random 32-bit number) / 2^24. p = 1 drops everything (macrograd doesn't allow it).
//
// the random numbers come from Philox-4x32-10 (Salmon et al., "Parallel random numbers:
// as easy as 1, 2, 3", 2011), a counter-based generator: four 32-bit outputs are a fixed
// scrambling of a 128-bit counter and a 64-bit key, ten rounds of multiplies and xors. there
// is no state to carry from one number to the next, so value i of the mask is just
// philox(counter = i / 4, key = seed)[i % 4], which any thread can work out on its own,
// and the backward makes the same mask again from the seed instead of reading a stored
// one. the same function runs on the CPU, so the test knows the exact mask.
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/dropout.cu -o build/dropout && ./build/dropout

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "common.h"

const int THREADS = 256;

struct U4 {
    uint32_t x[4];
};

// one round: two 32 x 32 -> 64 bit multiplies by fixed odd numbers, their high halves
// xored with the other words and the key, and the words swapped about
__host__ __device__ inline U4 philox_round(U4 c, uint32_t k0, uint32_t k1) {
    const uint64_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    uint64_t p0 = M0 * c.x[0], p1 = M1 * c.x[2];
    U4 out;
    out.x[0] = (uint32_t)(p1 >> 32) ^ c.x[1] ^ k0;
    out.x[1] = (uint32_t)p1;
    out.x[2] = (uint32_t)(p0 >> 32) ^ c.x[3] ^ k1;
    out.x[3] = (uint32_t)p0;
    return out;
}

// ten rounds, the key moving by fixed steps between them (the golden ratio and sqrt(3) - 1
// as 32-bit fractions)
__host__ __device__ inline U4 philox4x32_10(U4 c, uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; r++) {
        c = philox_round(c, k0, k1);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

// the 4 random numbers for values 4g .. 4g + 3
__host__ __device__ inline U4 random4(uint64_t g, uint64_t seed) {
    U4 c = {{(uint32_t)g, (uint32_t)(g >> 32), 0u, 0u}};
    return philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
}

__host__ __device__ inline bool keep(uint32_t r, float p) {
    return (float)(r >> 8) * (1.0f / 16777216.0f) > p;
}

// each thread does 4 values, one call to philox. backward is the same with ADD: dx += ...
template <bool ADD>
__global__ void dropout_kernel(const float *in, float *out, size_t n, float p, float scale, uint64_t seed) {
    uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * g >= n) {
        return;
    }
    U4 r = random4(g, seed);
    for (int k = 0; k < 4; k++) {
        size_t i = 4 * g + k;
        if (i < n) {
            float v = keep(r.x[k], p) ? in[i] * scale : 0.0f;
            out[i] = ADD ? out[i] + v : v;
        }
    }
}

float scale_for(float p) { return p >= 1.0f ? 0.0f : 1.0f / (1.0f - p); }

void dropout_forward(const float *x, float *out, size_t n, float p, uint64_t seed) {
    size_t groups = (n + 3) / 4;
    dropout_kernel<false><<<(unsigned)((groups + THREADS - 1) / THREADS), THREADS>>>(x, out, n, p, scale_for(p), seed);
}

// the seed has to be the forward's
void dropout_backward(const float *dy, float *dx, size_t n, float p, uint64_t seed) {
    size_t groups = (n + 3) / 4;
    dropout_kernel<true><<<(unsigned)((groups + THREADS - 1) / THREADS), THREADS>>>(dy, dx, n, p, scale_for(p), seed);
}

template <typename F>
float time_ms(F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 50; it++) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / 50;
}

// ---- tests ----

const size_t CAP = ((size_t)1 << 24) + 4096, PAST = 4096;

// the gpu output against the cpu's, bit for bit: the same philox, the same float ops
bool check(size_t n, float p, uint64_t seed, float *d_x, float *d_out, std::mt19937 &rng, double *kept_out) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(n), dx0(n), want(n), want_dx(n);
    for (auto &v : x) v = u(rng);
    for (auto &v : dx0) v = u(rng);
    float scale = scale_for(p);
    size_t kept = 0;
    for (size_t i = 0; i < n; i++) {
        bool k = keep(random4(i / 4, seed).x[i % 4], p);
        kept += k;
        want[i] = k ? x[i] * scale : 0.0f;
        want_dx[i] = dx0[i] + (k ? x[i] * scale : 0.0f);  // x plays dy in the backward
    }
    *kept_out = (double)kept / n;
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (int pass = 0; pass < 2; pass++) {
        CUDA_CHECK(cudaMemset(d_out, 0xff, (n + PAST) * sizeof(float)));
        if (pass == 0) {
            dropout_forward(d_x, d_out, n, p, seed);
        } else {
            CUDA_CHECK(cudaMemcpy(d_out, dx0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            dropout_backward(d_x, d_out, n, p, seed);
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> got(n + PAST);
        CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const std::vector<float> &w = pass == 0 ? want : want_dx;
        size_t wrong = 0;
        for (size_t i = 0; i < n; i++) wrong += memcmp(&got[i], &w[i], sizeof(float)) != 0;
        for (size_t i = n; i < got.size(); i++) wrong += !std::isnan(got[i]);
        if (wrong) {
            printf("  n %zu, p %.2f, seed %llu, %s: %zu wrong values\n", n, p, (unsigned long long)seed,
                   pass == 0 ? "forward" : "backward", wrong);
            ok = false;
        }
    }
    return ok;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, CAP * sizeof(float)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    // the three philox4x32 10 known-answer tests in Random123's tests/kat_vectors, the
    // reference implementation (github.com/DEShawResearch/random123)
    U4 zero = philox4x32_10({{0, 0, 0, 0}}, 0, 0);
    U4 ones = philox4x32_10({{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}}, 0xffffffffu, 0xffffffffu);
    U4 pi = philox4x32_10({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, 0xa4093822u, 0x299f31d0u);
    uint32_t want[3][4] = {{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u},
                           {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu},
                           {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}};
    bool ok = memcmp(zero.x, want[0], 16) == 0 && memcmp(ones.x, want[1], 16) == 0 && memcmp(pi.x, want[2], 16) == 0;
    printf("philox known answers: %s (zeros: %08x %08x %08x %08x)\n", ok ? "ok" : "WRONG", zero.x[0], zero.x[1],
           zero.x[2], zero.x[3]);

    std::mt19937 rng(0);
    bool exact = true;
    double kept;
    for (size_t n : {(size_t)1, (size_t)3, (size_t)4, (size_t)1001, (size_t)1 << 20}) {
        for (float p : {0.0f, 0.1f, 0.5f, 0.9f, 1.0f}) {
            exact = check(n, p, 1234 + n, d_x, d_out, rng, &kept) && exact;
        }
    }
    printf("gpu equals cpu, forward and backward: %s\n", exact ? "ok" : "WRONG");
    ok = ok && exact;

    // how many are kept, on 2^24 values: within 5 standard deviations of 1 - p
    bool fractions = true;
    for (float p : {0.1f, 0.5f, 0.9f}) {
        size_t n = (size_t)1 << 24;
        check(n, p, 99, d_x, d_out, rng, &kept);
        double sd = sqrt(p * (1 - p) / n);
        bool good = fabs(kept - (1 - p)) < 5 * sd;
        printf("p %.1f: kept %.6f, expected %.6f +- %.6f, %s\n", p, kept, 1 - p, sd, good ? "ok" : "WRONG");
        fractions = fractions && good;
    }
    // two seeds give different masks
    {
        size_t n = 1 << 20, same = 0;
        for (size_t i = 0; i < n; i++) {
            same += keep(random4(i / 4, 1).x[i % 4], 0.5f) == keep(random4(i / 4, 2).x[i % 4], 0.5f);
        }
        bool good = fabs((double)same / n - 0.5) < 0.01;
        printf("seeds 1 and 2 agree on %.4f of a p = 0.5 mask (about 0.5 if unrelated): %s\n", (double)same / n,
               good ? "ok" : "WRONG");
        fractions = fractions && good;
    }
    ok = ok && fractions;

    // forward reads x and writes out, 8 bytes a value; backward reads dy and dx and writes dx, 12
    size_t n = (size_t)1 << 24;
    float f = time_ms([&] { dropout_forward(d_x, d_out, n, 0.1f, 7); });
    float b = time_ms([&] { dropout_backward(d_x, d_out, n, 0.1f, 7); });
    float c = time_ms([&] { CUDA_CHECK(cudaMemcpy(d_out, d_x, n * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("\n2^24 values: forward %.3f ms %.1f GB/s, backward %.3f ms %.1f GB/s, memcpy %.3f ms %.1f GB/s\n", f,
           8.0 * n / (f * 1e-3) / 1e9, b, 12.0 * n / (b * 1e-3) / 1e9, c, 8.0 * n / (c * 1e-3) / 1e9);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
