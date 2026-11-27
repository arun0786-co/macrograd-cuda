// out-of-place transpose: out (C, R) = in (R, C)^T, both stored row by row. no arithmetic,
// so the only question is how well the memory accesses line up. a warp's 32 loads or
// stores are fast when they touch 32 neighbouring floats (coalesced: a few 128-byte
// transactions) and slow when they're a whole row apart (32 transactions).
//
// every version works on 32 x 32 tiles with blocks of 32 x 8 threads, each thread moving
// 4 values, a common layout (Harris, "An Efficient Matrix Transpose in CUDA C/C++"):
//
// 1. naive: each thread reads in[r][c] and writes out[c][r]. a warp's reads go along a
//    row of in, but its writes go down a column of out, C floats apart
// 2. tiled: the block reads its tile into shared memory along rows, and writes it out
//    along rows of out, reading the tile down its columns. both global accesses are
//    coalesced. but shared memory has 32 banks, a float at index i sits in bank i % 32,
//    and a column of a 32-wide tile is all one bank, so the 32 reads of a warp take turns
// 3. padded: the tile is 32 x 33. the extra column moves each row one bank over, so a
//    column of the tile touches all 32 banks
//
// checked exactly against the CPU (a transpose only moves floats), with odd shapes so
// tiles hang over the edges, and timed against cudaMemcpy.
//
// build and run: nvcc -O3 -arch=native kernels/transpose.cu -o build/transpose && ./build/transpose

#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include "common.h"

const int TILE = 32, ROWS = 8;  // blocks of TILE x ROWS threads, each covering TILE / ROWS rows

__global__ void transpose_naive(const float *in, float *out, int R, int C) {
    int c = blockIdx.x * TILE + threadIdx.x;
    for (int k = 0; k < TILE; k += ROWS) {
        int r = blockIdx.y * TILE + threadIdx.y + k;
        if (r < R && c < C) {
            out[(size_t)c * R + r] = in[(size_t)r * C + c];
        }
    }
}

template <int PAD>
__global__ void transpose_tiled(const float *in, float *out, int R, int C) {
    __shared__ float tile[TILE][TILE + PAD];
    int c = blockIdx.x * TILE + threadIdx.x;
    for (int k = 0; k < TILE; k += ROWS) {
        int r = blockIdx.y * TILE + threadIdx.y + k;
        if (r < R && c < C) {
            tile[threadIdx.y + k][threadIdx.x] = in[(size_t)r * C + c];
        }
    }
    __syncthreads();
    // the block's tile of out starts at row blockIdx.x * TILE and column blockIdx.y * TILE.
    // threadIdx.x now runs along a row of out, which is down a column of the tile
    int oc = blockIdx.y * TILE + threadIdx.x;
    for (int k = 0; k < TILE; k += ROWS) {
        int orow = blockIdx.x * TILE + threadIdx.y + k;
        if (orow < C && oc < R) {
            out[(size_t)orow * R + oc] = tile[threadIdx.x][threadIdx.y + k];
        }
    }
}

typedef void (*Launch)(const float *, float *, int, int);

void launch_naive(const float *in, float *out, int R, int C) {
    dim3 grid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    transpose_naive<<<grid, dim3(TILE, ROWS)>>>(in, out, R, C);
}

template <int PAD>
void launch_tiled(const float *in, float *out, int R, int C) {
    dim3 grid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    transpose_tiled<PAD><<<grid, dim3(TILE, ROWS)>>>(in, out, R, C);
}

const int VERSIONS = 3;
const char *NAMES[] = {"naive", "tiled", "padded"};
const Launch LAUNCHES[] = {launch_naive, launch_tiled<0>, launch_tiled<1>};

template <typename F>
float time_ms(F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; it++) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / 100;
}

// the gpu result must be exactly the cpu one, and the PAST floats after it untouched
const size_t CAP = ((size_t)1 << 24) + 4096, PAST = 4096;

bool check_shape(int R, int C, float *d_in, float *d_out, std::mt19937 &rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> in((size_t)R * C);
    for (auto &v : in) v = u(rng);
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (int v = 0; v < VERSIONS; v++) {
        CUDA_CHECK(cudaMemset(d_out, 0xff, CAP * sizeof(float)));  // nan, so a missed write shows
        LAUNCHES[v](d_in, d_out, R, C);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> out(in.size() + PAST);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        size_t wrong = 0;
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                // != is false for nan, so compare the bits
                float got = out[(size_t)c * R + r], want = in[(size_t)r * C + c];
                wrong += memcmp(&got, &want, sizeof(float)) != 0;
            }
        }
        for (size_t k = in.size(); k < out.size(); k++) {
            wrong += !std::isnan(out[k]);
        }
        if (wrong) {
            printf("  %d x %d, %s: %zu wrong values\n", R, C, NAMES[v], wrong);
            ok = false;
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

    // one value, one row or column, sizes around a tile, and odd ones
    bool ok = true;
    int shapes[][2] = {{1, 1}, {1, 100}, {100, 1}, {31, 33}, {32, 32}, {33, 31}, {64, 8}, {8, 64},
                       {1000, 777}, {777, 1000}, {4097, 3}, {3, 4097}};
    for (auto &s : shapes) {
        ok = check_shape(s[0], s[1], d_in, d_out, rng) && ok;
    }
    printf("shapes: %s\n", ok ? "ok" : "WRONG");

    // each value is read once and written once: 8 bytes
    printf("\n%-14s %-8s %-10s %-10s\n", "rows x cols", "version", "ms", "GB/s");
    int timed[][2] = {{4096, 4096}, {16384, 1024}, {1024, 16384}, {4000, 4000}};
    for (auto &s : timed) {
        int R = s[0], C = s[1];
        char shape[32];
        snprintf(shape, sizeof shape, "%d x %d", R, C);
        for (int v = 0; v < VERSIONS; v++) {
            float ms = time_ms([&] { LAUNCHES[v](d_in, d_out, R, C); });
            printf("%-14s %-8s %-10.4f %-10.1f\n", v == 0 ? shape : "", NAMES[v], ms, 8.0 * R * C / (ms * 1e-3) / 1e9);
        }
    }
    size_t n = (size_t)1 << 24;
    float ms = time_ms([&] { CUDA_CHECK(cudaMemcpy(d_out, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("%-14s %-8s %-10.4f %-10.1f\n", "2^24 floats", "memcpy", ms, 8.0 * n / (ms * 1e-3) / 1e9);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
