// sum reduction: one float out of n floats, three ways, each faster than the last.
//
// 1. atomic: every thread adds its value straight into one global float with atomicAdd.
//    simple, but all n adds land on the same address one after another, and the running
//    total gets big, so every small value added to it loses its low bits.
// 2. shared memory: each block adds up its 256 values as a tree in shared memory
//    (128 pairs, then 64, ..., then 1) and writes one partial sum. the partials are
//    reduced again by the same kernel until one value is left.
// 3. warp shuffle: a fixed number of blocks, each thread first adds up many values in a
//    register with a grid-stride loop. then the 32 threads of a warp add their sums by
//    passing registers to each other with __shfl_down_sync, without shared memory or
//    __syncthreads. one more small launch adds the block partials.
//
// all of them are checked against a sum in double precision on the CPU, and timed against
// cub::DeviceReduce::Sum, the reduction that ships with CUDA.
//
// build and run: nvcc -O3 -arch=native kernels/reduction.cu -o run && ./run

#include <cmath>
#include <cstring>
#include <random>
#include <cub/cub.cuh>
#include "common.h"

const int THREADS = 256;

// ---- 1. atomic ----

__global__ void sum_atomic(const float *x, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(out, x[i]);
    }
}

// ---- 2. shared memory tree ----

// sums x[blockIdx.x * 256 ... + 255] into partial[blockIdx.x]. each step halves the
// number of active threads, and thread t adds the value s places to its right. the
// active threads stay packed at the front (sequential addressing), so whole warps go
// idle together instead of every warp having a few threads left doing work.
__global__ void sum_shared(const float *x, float *partial, int n) {
    __shared__ float sh[THREADS];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    sh[t] = i < n ? x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            sh[t] += sh[t + s];
        }
        // every thread has to see this step's sums before the next step reads them
        __syncthreads();
    }
    if (t == 0) {
        partial[blockIdx.x] = sh[0];
    }
}

// ---- 3. warp shuffle ----

// after this, lane 0 holds the sum of all 32 lanes' v. each step every lane adds the
// value from the lane `offset` above it: 16, 8, 4, 2, 1.
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// one partial sum per block. the grid is fixed and smaller than n, so each thread first
// walks the array in steps of the whole grid's size, which keeps neighbouring threads
// reading neighbouring floats. needs blockDim.x to be a multiple of 32, at most 1024.
__global__ void sum_shuffle(const float *x, float *partial, int n) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        v += x[i];
    }
    v = warp_sum(v);

    // lane 0 of each warp now has its warp's sum. the first warp adds those up
    __shared__ float warps[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        warps[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < blockDim.x / 32 ? warps[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) {
            partial[blockIdx.x] = v;
        }
    }
}

// ---- launches. each one leaves the sum in out[0] ----

// scratch space for partial sums, set up in main
float *g_a, *g_b;
void *g_cub_tmp;
size_t g_cub_bytes;
int g_shuffle_blocks;

void launch_atomic(const float *x, float *out, int n) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float)));
    sum_atomic<<<(n + THREADS - 1) / THREADS, THREADS>>>(x, out, n);
}

void launch_shared(const float *x, float *out, int n) {
    // 2^24 + 7 values -> 65,537 partials -> 257 -> 2 -> 1, swapping between two buffers
    const float *in = x;
    float *bufs[2] = {g_a, g_b};
    int pass = 0;
    while (true) {
        int blocks = (n + THREADS - 1) / THREADS;
        float *dst = blocks == 1 ? out : bufs[pass % 2];
        sum_shared<<<blocks, THREADS>>>(in, dst, n);
        if (blocks == 1) {
            break;
        }
        in = dst;
        n = blocks;
        pass++;
    }
}

void launch_shuffle(const float *x, float *out, int n) {
    sum_shuffle<<<g_shuffle_blocks, THREADS>>>(x, g_a, n);
    sum_shuffle<<<1, 1024>>>(g_a, out, g_shuffle_blocks);
}

void launch_cub(const float *x, float *out, int n) {
    CUDA_CHECK(cub::DeviceReduce::Sum(g_cub_tmp, g_cub_bytes, x, out, n));
}

float time_ms(void (*launch)(const float *, float *, int), const float *x, float *out, int n) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch(x, out, n);  // warmup
    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        launch(x, out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

float run_once(void (*launch)(const float *, float *, int), const float *x, float *out, int n) {
    launch(x, out, n);
    CUDA_CHECK(cudaGetLastError());
    float got;
    CUDA_CHECK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));
    return got;
}

// error relative to the sum of |x|, so it means the same thing for sums near 0
bool check(const char *name, void (*launch)(const float *, float *, int), const float *d_x,
           float *d_out, int n, double ref, double abs_sum, double tol) {
    float got = run_once(launch, d_x, d_out, n);
    // run it a few more times: the tree versions add in a fixed order, so they should
    // give the same bits every time. the atomic one adds in whatever order threads arrive
    bool same = true;
    for (int r = 0; r < 5; r++) {
        float again = run_once(launch, d_x, d_out, n);
        same = same && memcmp(&again, &got, sizeof(float)) == 0;
    }
    double err = fabs(got - ref) / abs_sum;
    float ms = time_ms(launch, d_x, d_out, n);
    bool ok = err < tol;
    printf("%-10s %-16.8g %-10.2e %-6s %-10s %-9.4f %-8.1f\n", name, got, err, ok ? "ok" : "WRONG",
           same ? "yes" : "no", ms, 4.0 * n / (ms * 1e-3) / 1e9);
    return ok;
}

bool test_small_sizes(const float *d_x, float *d_out, const float *h_x) {
    // sizes around the block size and the edges of each pass, against the CPU
    int sizes[] = {1, 2, 31, 32, 33, 255, 256, 257, 1000, 65536, 65537, 65793};
    void (*launches[])(const float *, float *, int) = {launch_atomic, launch_shared, launch_shuffle};
    double tols[] = {1e-4, 1e-6, 1e-6};  // atomic adds in one running total, see main
    bool ok = true;
    for (int n : sizes) {
        double ref = 0.0, abs_sum = 0.0;
        for (int i = 0; i < n; i++) {
            ref += h_x[i];
            abs_sum += fabs(h_x[i]);
        }
        for (int k = 0; k < 3; k++) {
            float got = run_once(launches[k], d_x, d_out, n);
            if (fabs(got - ref) / abs_sum > tols[k]) {
                printf("small size %d, kernel %d: got %.8g, want %.8g\n", n, k, got, ref);
                ok = false;
            }
        }
    }
    // the shuffle version's second pass sums one partial per block with up to 1024
    // threads. at the default block count only a few of its 32 warps get data, so also
    // try 1 block (no real second pass) and 1024 blocks (every warp gets data)
    int default_blocks = g_shuffle_blocks;
    for (int blocks : {1, 1024}) {
        g_shuffle_blocks = blocks;
        for (int n : {1000, 65793, 1 << 20}) {
            double ref = 0.0, abs_sum = 0.0;
            for (int i = 0; i < n; i++) {
                ref += h_x[i];
                abs_sum += fabs(h_x[i]);
            }
            float got = run_once(launch_shuffle, d_x, d_out, n);
            if (fabs(got - ref) / abs_sum > 1e-6) {
                printf("shuffle with %d blocks, size %d: got %.8g, want %.8g\n", blocks, n, got, ref);
                ok = false;
            }
        }
    }
    g_shuffle_blocks = default_blocks;
    return ok;
}

int main() {
    int n = (1 << 24) + 7;
    size_t bytes = n * sizeof(float);
    float *x = (float *)malloc(bytes);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        x[i] = uniform(rng);
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // enough blocks to fill every SM a few times over, few enough that the second pass
    // fits in one block of 1024 threads
    g_shuffle_blocks = prop.multiProcessorCount * 8;
    if (g_shuffle_blocks > 1024) {
        g_shuffle_blocks = 1024;
    }
    int max_partials = (n + THREADS - 1) / THREADS;
    CUDA_CHECK(cudaMalloc(&g_a, max_partials * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_b, max_partials * sizeof(float)));
    g_cub_tmp = nullptr;
    CUDA_CHECK(cub::DeviceReduce::Sum(g_cub_tmp, g_cub_bytes, d_x, d_out, n));
    CUDA_CHECK(cudaMalloc(&g_cub_tmp, g_cub_bytes));

    printf("gpu: %s, %d SMs, n = %d, shuffle blocks = %d\n", prop.name, prop.multiProcessorCount, n,
           g_shuffle_blocks);
    bool small_ok = test_small_sizes(d_x, d_out, x);
    printf("small sizes: %s\n", small_ok ? "ok" : "WRONG");

    double ref = 0.0, abs_sum = 0.0;
    for (int i = 0; i < n; i++) {
        ref += x[i];
        abs_sum += fabs(x[i]);
    }
    printf("cpu sum in double: %.8f\n\n", ref);

    // a float holds about 7 digits, so every add into a running total near 8 million
    // rounds to the nearest 0.5 or 1. the atomic version does 16 million of those adds
    // into one number and gets a looser tolerance; the tree versions keep each partial
    // sum small and should be within a few float roundings
    printf("%-10s %-16s %-10s %-6s %-10s %-9s %-8s\n", "kernel", "sum", "rel err", "check",
           "same bits", "ms", "GB/s");
    bool ok = small_ok;
    ok = check("atomic", launch_atomic, d_x, d_out, n, ref, abs_sum, 1e-2) && ok;
    ok = check("shared", launch_shared, d_x, d_out, n, ref, abs_sum, 1e-6) && ok;
    ok = check("shuffle", launch_shuffle, d_x, d_out, n, ref, abs_sum, 1e-6) && ok;
    ok = check("cub", launch_cub, d_x, d_out, n, ref, abs_sum, 1e-6) && ok;

    // a plain copy of the same bytes, as a ceiling for what reading memory can do
    float *d_copy;
    CUDA_CHECK(cudaMalloc(&d_copy, bytes));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemcpy(d_copy, d_x, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; it++) {
        CUDA_CHECK(cudaMemcpy(d_copy, d_x, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float copy_ms;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start, stop));
    copy_ms /= 100;
    printf("\ncudaMemcpy device to device: %.4f ms, %.1f GB/s (reads and writes, 8 bytes per float)\n",
           copy_ms, 2.0 * bytes / (copy_ms * 1e-3) / 1e9);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_copy));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(g_a));
    CUDA_CHECK(cudaFree(g_b));
    CUDA_CHECK(cudaFree(g_cub_tmp));
    free(x);
    return ok ? 0 : 1;
}
