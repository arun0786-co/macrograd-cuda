// running total along each row of x (R, C), stored row by row, like macrograd's
// cumsum(axis=-1): out[r][k] = x[r][0] + ... + x[r][k]. its backward is a running total
// from the other end, dx[r][j] += dy[r][j] + dy[r][j+1] + ... + dy[r][C-1], since x[j]
// went into every out[k] with k >= j. the kernels take REVERSE for that and ADD for
// adding into the output instead of overwriting it.
//
// a scan looks serial, since each total needs the one before. it splits up like this:
// each thread scans 4 values of its own, the threads' totals are scanned across the
// block, and each thread adds the total of everything before it. inside a warp that
// middle step is 5 __shfl_up_sync steps (Hillis-Steele: after step s every lane holds the
// sum of the 2^s lanes up to it); across the warps of a block it's the same again on the
// warps' totals in shared memory.
//
// 1. rows: one block of 256 per row walks the row 1024 values at a time and carries the
//    total from one piece to the next. fine when there are enough rows to fill the GPU
// 2. tiles: for a few long rows. three kernels: each block sums one tile of 1024, the
//    tiles' sums are scanned (with kernel 1, they're a short row per row), and each block
//    scans its tile again starting from the sum of the tiles before it. it reads x twice,
//    but every SM has work
//
// cumsum() picks rows when there are at least 2 rows per SM, tiles otherwise. checked
// exactly on whole numbers (their sums are exact in float up to 2^24), against double on
// random floats, and timed against cub::DeviceScan::InclusiveSum on one long row.
//
// build and run: nvcc -O3 -arch=native -std=c++17 kernels/cumsum.cu -o build/cumsum && ./build/cumsum

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <cub/cub.cuh>
#include "common.h"

const int THREADS = 256, PER_THREAD = 4, TILE = THREADS * PER_THREAD;

// where the k-th value of a row sits: from the start, or from the end for the backward
template <bool REVERSE>
__device__ size_t at(int k, int C) {
    return REVERSE ? (size_t)(C - 1 - k) : (size_t)k;
}

// inclusive scan of one value per thread across the block, and the block's total.
// blockDim.x == THREADS
__device__ float block_scan(float v, float *total) {
    __shared__ float warps[THREADS / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) {
            v += n;
        }
    }
    __syncthreads();  // an earlier call may still be reading warps[]
    if (lane == 31) {
        warps[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        float w = lane < THREADS / 32 ? warps[lane] : 0.0f;
        for (int offset = 1; offset < 32; offset <<= 1) {
            float n = __shfl_up_sync(0xffffffff, w, offset);
            if (lane >= offset) {
                w += n;
            }
        }
        if (lane < THREADS / 32) {
            warps[lane] = w;
        }
    }
    __syncthreads();
    *total = warps[THREADS / 32 - 1];
    return v + (warp > 0 ? warps[warp - 1] : 0.0f);
}

// scans this block's TILE values of a row from position base, adding start to all of
// them, and returns the tile's total. thread t has positions base + 4t .. base + 4t + 3
template <bool REVERSE, bool ADD>
__device__ float scan_tile(const float *in, float *out, int C, int base, float start) {
    float v[PER_THREAD];
    int k0 = base + threadIdx.x * PER_THREAD;
    for (int i = 0; i < PER_THREAD; i++) {
        v[i] = k0 + i < C ? in[at<REVERSE>(k0 + i, C)] : 0.0f;
    }
    for (int i = 1; i < PER_THREAD; i++) {
        v[i] += v[i - 1];
    }
    float total;
    float before = block_scan(v[PER_THREAD - 1], &total) - v[PER_THREAD - 1] + start;
    for (int i = 0; i < PER_THREAD; i++) {
        if (k0 + i < C) {
            size_t p = at<REVERSE>(k0 + i, C);
            out[p] = (ADD ? out[p] : 0.0f) + v[i] + before;
        }
    }
    return total;
}

// ---- 1. a block per row ----

template <bool REVERSE, bool ADD>
__global__ void scan_rows(const float *x, float *out, int C) {
    const float *in = x + (size_t)blockIdx.x * C;
    float *o = out + (size_t)blockIdx.x * C;
    float carry = 0.0f;
    for (int base = 0; base < C; base += TILE) {
        carry += scan_tile<REVERSE, ADD>(in, o, C, base, carry);
    }
}

// ---- 2. tiles ----

// sums[r][t] = the sum of tile t of row r (in REVERSE order, tile 0 is at the row's end)
template <bool REVERSE>
__global__ void tile_sums(const float *x, float *sums, int C, int tiles) {
    int r = blockIdx.x, t = blockIdx.y;
    const float *in = x + (size_t)r * C;
    float s = 0.0f;
    int k0 = t * TILE + threadIdx.x * PER_THREAD;
    for (int i = 0; i < PER_THREAD; i++) {
        s += k0 + i < C ? in[at<REVERSE>(k0 + i, C)] : 0.0f;
    }
    float total;
    block_scan(s, &total);
    if (threadIdx.x == 0) {
        sums[(size_t)r * tiles + t] = total;
    }
}

// each tile scanned again, starting from the total of the tiles before it: the scanned
// sums (inclusive) minus the tile's own sum
template <bool REVERSE, bool ADD>
__global__ void scan_tiles(const float *x, float *out, const float *sums, const float *scanned, int C, int tiles) {
    int r = blockIdx.x, t = blockIdx.y;
    size_t i = (size_t)r * tiles + t;
    float start = scanned[i] - sums[i];
    scan_tile<REVERSE, ADD>(x + (size_t)r * C, out + (size_t)r * C, C, t * TILE, start);
}

// ---- launches ----

float *g_sums = nullptr;
size_t g_sums_size = 0;

template <bool REVERSE, bool ADD>
void cumsum_rows(const float *x, float *out, int R, int C) {
    scan_rows<REVERSE, ADD><<<R, THREADS>>>(x, out, C);
}

template <bool REVERSE, bool ADD>
void cumsum_tiles(const float *x, float *out, int R, int C) {
    int tiles = (C + TILE - 1) / TILE;
    size_t n = (size_t)R * tiles;
    if (2 * n > g_sums_size) {
        if (g_sums) {
            CUDA_CHECK(cudaFree(g_sums));
        }
        CUDA_CHECK(cudaMalloc(&g_sums, 2 * n * sizeof(float)));
        g_sums_size = 2 * n;
    }
    float *sums = g_sums, *scanned = g_sums + n;
    // rows on x, which goes up to 2^31 - 1, and tiles on y, up to 65535 (rows of 67M)
    dim3 grid(R, tiles);
    tile_sums<REVERSE><<<grid, THREADS>>>(x, sums, C, tiles);
    // the sums are already in the order of the scan, so they're scanned forwards
    scan_rows<false, false><<<R, THREADS>>>(sums, scanned, tiles);
    scan_tiles<REVERSE, ADD><<<grid, THREADS>>>(x, out, sums, scanned, C, tiles);
}

int cs_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    }
    return sms;
}

template <bool REVERSE, bool ADD>
void cumsum_auto(const float *x, float *out, int R, int C) {
    if (R >= 2 * cs_sm_count() || C <= TILE) {
        cumsum_rows<REVERSE, ADD>(x, out, R, C);
    } else {
        cumsum_tiles<REVERSE, ADD>(x, out, R, C);
    }
}

// out = cumsum(x) along rows, and its backward dx += reverse cumsum(dy)
void cumsum(const float *x, float *out, int R, int C) { cumsum_auto<false, false>(x, out, R, C); }
void cumsum_backward(const float *dy, float *dx, int R, int C) { cumsum_auto<true, true>(dy, dx, R, C); }

typedef void (*Launch)(const float *, float *, int, int);
const int VERSIONS = 3;
const char *NAMES[] = {"rows", "tiles", "auto"};
const Launch FORWARDS[] = {cumsum_rows<false, false>, cumsum_tiles<false, false>, cumsum};
const Launch BACKWARDS[] = {cumsum_rows<true, true>, cumsum_tiles<true, true>, cumsum_backward};

template <typename F>
float time_ms(F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 20; it++) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / 20;
}

// ---- tests ----

const size_t CAP = ((size_t)1 << 24) + 4096, PAST = 4096;

// whole numbers in [-3, 3]: every partial sum is a whole number below 2^24, which float
// holds exactly, so the gpu has to match the cpu bit for bit whatever order it adds in.
// random floats: error against double, relative to the sum of |x| up to that point
bool check_shape(int R, int C, bool whole, float *d_in, float *d_out, std::mt19937 &rng, double *worst) {
    std::vector<float> x((size_t)R * C), dx0((size_t)R * C);
    std::uniform_int_distribution<int> small(-3, 3);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto &v : x) v = whole ? (float)small(rng) : u(rng);
    for (auto &v : dx0) v = whole ? (float)small(rng) : u(rng);
    std::vector<double> fwd(x.size()), fwd_s(x.size()), bwd(x.size()), bwd_s(x.size());
    for (int r = 0; r < R; r++) {
        double t = 0.0, a = 0.0;
        for (int k = 0; k < C; k++) {
            size_t i = (size_t)r * C + k;
            t += x[i];
            a += fabs(x[i]);
            fwd[i] = t;
            fwd_s[i] = a;
        }
        t = 0.0, a = 0.0;
        for (int k = C - 1; k >= 0; k--) {
            size_t i = (size_t)r * C + k;
            t += x[i];  // x plays dy here
            a += fabs(x[i]);
            bwd[i] = dx0[i] + t;
            bwd_s[i] = fabs(dx0[i]) + a;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_in, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (int v = 0; v < VERSIONS; v++) {
        for (int pass = 0; pass < 2; pass++) {
            // nan everywhere, then dx0 for the backward, which adds into it
            CUDA_CHECK(cudaMemset(d_out, 0xff, CAP * sizeof(float)));
            if (pass == 1) {
                CUDA_CHECK(cudaMemcpy(d_out, dx0.data(), dx0.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
            (pass == 0 ? FORWARDS : BACKWARDS)[v](d_in, d_out, R, C);
            CUDA_CHECK(cudaGetLastError());
            std::vector<float> got(x.size() + PAST);
            CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const std::vector<double> &want = pass == 0 ? fwd : bwd, &scale = pass == 0 ? fwd_s : bwd_s;
            double e = 0.0;
            for (size_t i = 0; i < x.size(); i++) {
                double d = fabs(got[i] - want[i]);
                if (!whole) {
                    d /= fmax(scale[i], 1e-30);
                }
                e = fmax(e, std::isnan(d) ? INFINITY : d);
            }
            bool past_ok = true;
            for (size_t i = x.size(); i < got.size(); i++) {
                past_ok = past_ok && std::isnan(got[i]);
            }
            // whole numbers exactly; floats within a few roundings per level of the scan
            double tol = whole ? 0.0 : 1e-6;
            if (!(e <= tol) || !past_ok) {
                printf("  %d x %d %s, %s %s: error %.2e%s\n", R, C, whole ? "whole" : "floats", NAMES[v],
                       pass == 0 ? "forward" : "backward", e, past_ok ? "" : ", wrote past the end");
                ok = false;
            }
            if (!whole) {
                *worst = fmax(*worst, e);
            }
        }
    }
    return ok;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, CAP * sizeof(float)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    // one value, around a warp, a block's 1024 and several tiles, and long rows
    bool ok = true;
    double worst = 0.0;
    int shapes[][2] = {{1, 1}, {1, 31}, {1, 33}, {1, 1023}, {1, 1024}, {1, 1025}, {1, 5000}, {3, 4097},
                       {200, 33}, {1000, 3}, {7, 100000}, {1, 3000001}, {130, 2049}};
    for (auto &s : shapes) {
        for (bool whole : {true, false}) {
            ok = check_shape(s[0], s[1], whole, d_in, d_out, rng, &worst) && ok;
        }
    }
    printf("shapes: %s, largest float error %.2e\n", ok ? "ok" : "WRONG", worst);

    // timing: 8 bytes a value (read x, write out); the tiles version reads x twice
    CUDA_CHECK(cudaMemset(d_in, 0, CAP * sizeof(float)));
    printf("\n%-16s %-20s %-20s %-20s\n", "rows x cols", "rows", "tiles", "backward (auto)");
    int timed[][2] = {{1, 1 << 24}, {16, 1 << 20}, {64, 1 << 18}, {4096, 4096}, {65536, 256}};
    for (auto &s : timed) {
        int R = s[0], C = s[1];
        double n = (double)R * C;
        printf("%6d x %-8d", R, C);
        for (int v = 0; v < 2; v++) {
            float ms = time_ms([&] { FORWARDS[v](d_in, d_out, R, C); });
            printf(" %8.3f ms %6.1f GB/s", ms, 8 * n / (ms * 1e-3) / 1e9);
        }
        float ms = time_ms([&] { cumsum_backward(d_in, d_out, R, C); });
        printf(" %8.3f ms %6.1f GB/s\n", ms, 12 * n / (ms * 1e-3) / 1e9);  // backward also reads dx
    }
    int n = 1 << 24;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_in, d_out, n);
    void *temp;
    CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
    float ms = time_ms([&] { cub::DeviceScan::InclusiveSum(temp, temp_bytes, d_in, d_out, n); });
    printf("cub InclusiveSum, 1 x %d: %.3f ms, %.1f GB/s\n", n, ms, 8.0 * n / (ms * 1e-3) / 1e9);
    ms = time_ms([&] { CUDA_CHECK(cudaMemcpy(d_out, d_in, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("memcpy, %d floats: %.3f ms, %.1f GB/s\n", n, ms, 8.0 * n / (ms * 1e-3) / 1e9);

    CUDA_CHECK(cudaFree(temp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
