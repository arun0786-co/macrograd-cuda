// softmax over each row of a (rows, cols) matrix stored row by row, three ways, plus the
// backward.
//
//    y[r][i] = exp(x[r][i] - m) / sum_j exp(x[r][j] - m),   m = max_j x[r][j]
//
// subtracting the row max doesn't change the answer (it cancels between top and bottom),
// but it keeps exp from overflowing: exp(89) is already past the largest float.
//
// 1. naive: one thread per row. each thread walks its row three times: max, sum, write.
//    neighbouring threads read floats `cols` apart, so a warp's 32 loads land in 32
//    different places in memory, and when there are few rows most of the GPU sits idle.
// 2. block per row: 256 threads share a row, so a warp reads 32 neighbouring floats at
//    once. the max and the sum are block reductions (warp shuffle, then shared memory
//    across the warps). still three passes over the row.
// 3. online: max and sum in one pass (Milakov and Gimelshein, "Online normalizer
//    calculation for softmax", 2018). keep a running max m and a running sum d of
//    exp(x - m); when a bigger value arrives, scale d by exp(m_old - m_new) first. two
//    passes over the row: one to read, one to write.
//
// backward: dy is the gradient of the loss with respect to y. since
// dy_j/dx_i = y_j * ([i == j] - y_i),
//
//    dx_i = sum_j dy_j * y_j * ([i == j] - y_i) = y_i * (dy_i - sum_j dy_j * y_j)
//
// one block per row: a block sum for the dot product, then the update, added into dx
// like the elementwise backward kernels.
//
// all of it is checked against a CPU version in double precision, and the backward
// formula against finite differences.
//
// build and run: nvcc -O3 -arch=native kernels/softmax.cu -o run && ./run

#include <cmath>
#include <random>
#include <vector>
#include "common.h"

const int THREADS = 256;

struct Max {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Sum {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// every lane ends up with the result of all 32 lanes. __shfl_xor_sync swaps values
// between lanes whose numbers differ in one bit, so after 5 swaps (16, 8, 4, 2, 1) each
// lane has combined all 32, unlike __shfl_down_sync where only lane 0 has the total.
template <typename Op>
__device__ float warp_reduce(float v, Op op) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = op(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

// every thread of the block gets the result. blockDim.x must be a multiple of 32.
template <typename Op>
__device__ float block_reduce(float v, Op op, float identity) {
    __shared__ float warps[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_reduce(v, op);
    // an earlier call may still be reading warps[]
    __syncthreads();
    if (lane == 0) {
        warps[warp] = v;
    }
    __syncthreads();
    // every warp combines the per-warp results, so every thread has the answer
    v = lane < blockDim.x / 32 ? warps[lane] : identity;
    return warp_reduce(v, op);
}

// ---- 1. naive: one thread per row ----

__global__ void softmax_naive(const float *x, float *y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float *xr = x + (size_t)r * cols;
    float *yr = y + (size_t)r * cols;
    float m = -INFINITY;
    for (int i = 0; i < cols; i++) {
        m = fmaxf(m, xr[i]);
    }
    float s = 0.0f;
    for (int i = 0; i < cols; i++) {
        s += expf(xr[i] - m);
    }
    for (int i = 0; i < cols; i++) {
        yr[i] = expf(xr[i] - m) / s;
    }
}

// ---- 2. one block per row, three passes ----

__global__ void softmax_block(const float *x, float *y, int cols) {
    const float *xr = x + (size_t)blockIdx.x * cols;
    float *yr = y + (size_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        m = fmaxf(m, xr[i]);
    }
    m = block_reduce(m, Max(), -INFINITY);
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        s += expf(xr[i] - m);
    }
    s = block_reduce(s, Sum(), 0.0f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        yr[i] = expf(xr[i] - m) / s;
    }
}

// ---- 3. one block per row, online max and sum ----

// m is the largest value seen so far and d is the sum of exp(value - m) over them.
// m = -inf, d = 0 means nothing seen yet.
struct MD {
    float m, d;
};

// two partial results put together: move both sums onto the larger max
__device__ MD md_merge(MD a, MD b) {
    float m = fmaxf(a.m, b.m);
    // if one side is empty its d is 0 and exp(-inf) is 0, so the sum below is right. if
    // both are, exp(-inf - -inf) would be nan
    if (m == -INFINITY) {
        return a;
    }
    return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

__device__ MD warp_reduce_md(MD v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        MD other = {__shfl_xor_sync(0xffffffff, v.m, offset), __shfl_xor_sync(0xffffffff, v.d, offset)};
        v = md_merge(v, other);
    }
    return v;
}

__device__ MD block_reduce_md(MD v) {
    __shared__ MD warps[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_reduce_md(v);
    __syncthreads();
    if (lane == 0) {
        warps[warp] = v;
    }
    __syncthreads();
    v = lane < blockDim.x / 32 ? warps[lane] : MD{-INFINITY, 0.0f};
    return warp_reduce_md(v);
}

__global__ void softmax_online(const float *x, float *y, int cols) {
    const float *xr = x + (size_t)blockIdx.x * cols;
    float *yr = y + (size_t)blockIdx.x * cols;
    MD md = {-INFINITY, 0.0f};
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = xr[i];
        if (v > md.m) {
            // a new max: the old sum was relative to the old max, so scale it down
            md.d = md.d * expf(md.m - v) + 1.0f;
            md.m = v;
        } else if (v != -INFINITY) {
            // a masked value (-inf) adds exp(-inf) = 0. skipping it also avoids
            // exp(-inf - -inf) = nan while m is still -inf
            md.d += expf(v - md.m);
        }
    }
    md = block_reduce_md(md);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        yr[i] = expf(xr[i] - md.m) / md.d;
    }
}

// ---- backward: one block per row ----

__global__ void softmax_backward(const float *y, const float *dy, float *dx, int cols) {
    size_t row = (size_t)blockIdx.x * cols;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dot += dy[row + i] * y[row + i];
    }
    dot = block_reduce(dot, Sum(), 0.0f);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dx[row + i] += y[row + i] * (dy[row + i] - dot);
    }
}

// ---- launches ----

typedef void (*Forward)(const float *, float *, int, int);

void launch_naive(const float *x, float *y, int rows, int cols) {
    softmax_naive<<<(rows + THREADS - 1) / THREADS, THREADS>>>(x, y, rows, cols);
}

void launch_block(const float *x, float *y, int rows, int cols) {
    softmax_block<<<rows, THREADS>>>(x, y, cols);
}

void launch_online(const float *x, float *y, int rows, int cols) {
    softmax_online<<<rows, THREADS>>>(x, y, cols);
}

void launch_backward(const float *y, const float *dy, float *dx, int rows, int cols) {
    softmax_backward<<<rows, THREADS>>>(y, dy, dx, cols);
}

const char *NAMES[] = {"naive", "block", "online"};
const Forward FORWARDS[] = {launch_naive, launch_block, launch_online};

// mean time of one launch. slow kernels get fewer repeats
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
    int iters = ms > 10.0f ? 5 : 100;
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

// uniform in [-10, 10], and every fourth row of each kind: shifted up by 1000 (exp would
// overflow without the max), shifted down by 1000 (exp would underflow to 0), or with
// every third value masked to -inf like a causal attention mask. the first value of a
// row is never masked, so no row is all -inf.
void make_input(std::vector<float> &x, int rows, int cols, std::mt19937 &rng) {
    std::uniform_real_distribution<float> uniform(-10.0f, 10.0f);
    x.resize((size_t)rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int i = 0; i < cols; i++) {
            float v = uniform(rng);
            if (r % 4 == 1) {
                v += 1000.0f;
            } else if (r % 4 == 2 && i % 3 == 1) {
                v = -INFINITY;
            } else if (r % 4 == 3) {
                v -= 1000.0f;
            }
            x[(size_t)r * cols + i] = v;
        }
    }
}

void softmax_ref(const float *x, double *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * cols;
        double *yr = y + (size_t)r * cols;
        double m = -INFINITY, s = 0.0;
        for (int i = 0; i < cols; i++) {
            m = fmax(m, (double)xr[i]);
        }
        for (int i = 0; i < cols; i++) {
            s += exp(xr[i] - m);
        }
        for (int i = 0; i < cols; i++) {
            yr[i] = exp(xr[i] - m) / s;
        }
    }
}

// dx = dx0 + y * (dy - dot). scale is the size of what went into each value,
// |dx0| + y * (|dy| + sum_j |dy_j y_j|), so the error is relative to the numbers the GPU
// actually added, even where dy - dot cancels to almost nothing
void backward_ref(const float *y, const float *dy, const float *dx0, double *dx, double *scale,
                  int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        size_t row = (size_t)r * cols;
        double dot = 0.0, abs_dot = 0.0;
        for (int i = 0; i < cols; i++) {
            dot += (double)dy[row + i] * y[row + i];
            abs_dot += fabs((double)dy[row + i] * y[row + i]);
        }
        for (int i = 0; i < cols; i++) {
            dx[row + i] = dx0[row + i] + y[row + i] * (dy[row + i] - dot);
            scale[row + i] = fabs(dx0[row + i]) + y[row + i] * (fabs(dy[row + i]) + abs_dot);
        }
    }
}

// largest |got - want| / scale. nan counts as infinitely wrong
double max_error(const float *got, const double *want, const double *scale, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(got[i] - want[i]) / fmax(scale[i], 1e-30);
        if (std::isnan(e)) {
            e = INFINITY;
        }
        worst = fmax(worst, e);
    }
    return worst;
}

// checks the backward formula itself: for L = sum_j dy_j * softmax(x)_j, compare
// y_i * (dy_i - dot) with (L(x + h e_i) - L(x - h e_i)) / 2h, all in double
bool check_backward_formula() {
    const int C = 7;
    double x[C] = {0.3, -1.2, 2.0, 0.0, 5.5, -3.1, 1.7};
    double dy[C] = {0.5, -0.25, 1.0, 2.0, -1.5, 0.75, 0.1};
    auto loss = [&](const double *v) {
        double m = -INFINITY, s = 0.0, l = 0.0;
        for (int i = 0; i < C; i++) m = fmax(m, v[i]);
        for (int i = 0; i < C; i++) s += exp(v[i] - m);
        for (int i = 0; i < C; i++) l += dy[i] * exp(v[i] - m) / s;
        return l;
    };
    double y[C], dot = 0.0, s = 0.0;
    for (int i = 0; i < C; i++) s += exp(x[i] - 5.5);
    for (int i = 0; i < C; i++) {
        y[i] = exp(x[i] - 5.5) / s;
        dot += dy[i] * y[i];
    }
    double worst = 0.0, h = 1e-5;
    for (int i = 0; i < C; i++) {
        double plus[C], minus[C];
        for (int j = 0; j < C; j++) plus[j] = minus[j] = x[j];
        plus[i] += h;
        minus[i] -= h;
        double numeric = (loss(plus) - loss(minus)) / (2 * h);
        worst = fmax(worst, fabs(numeric - y[i] * (dy[i] - dot)));
    }
    printf("backward formula vs finite differences: max diff %.1e\n", worst);
    return worst < 1e-9;
}

struct Buffers {
    float *x, *y, *dy, *dx;
};

// runs every forward and the backward on one shape and checks them. with timing on, also
// prints a row per kernel. the backward starts from dx0 = 0 when timed, otherwise from
// random values to check that it adds into dx
bool run_shape(int rows, int cols, Buffers d, std::mt19937 &rng, bool timing) {
    size_t n = (size_t)rows * cols;
    std::vector<float> x, y(n), dy(n), dx0(n), got(n);
    make_input(x, rows, cols, rng);
    std::vector<double> ref(n), dx_ref(n), scale(n);
    softmax_ref(x.data(), ref.data(), rows, cols);
    CUDA_CHECK(cudaMemcpy(d.x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    bool ok = true;
    // the naive version adds a whole row into one running float, so its rounding errors
    // pile up with the row length (5.9e-5 at 65536 columns). the block versions split the
    // sum over 256 threads and a tree and stay near 1e-6
    double tols[] = {1e-4, 1e-5, 1e-5};
    for (int k = 0; k < 3; k++) {
        CUDA_CHECK(cudaMemset(d.y, 0xff, n * sizeof(float)));  // nan, so a missed write shows
        FORWARDS[k](d.x, d.y, rows, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(got.data(), d.y, n * sizeof(float), cudaMemcpyDeviceToHost));
        // relative to the value itself, so masked values must come out exactly 0
        double err = max_error(got.data(), ref.data(), ref.data(), n);
        bool good = err < tols[k];
        ok = ok && good;
        if (timing) {
            float ms = time_ms([&] { FORWARDS[k](d.x, d.y, rows, cols); });
            printf("  %-9s %-10.2e %-6s %-10.4f %-8.1f\n", NAMES[k], err, good ? "ok" : "WRONG", ms,
                   8.0 * n / (ms * 1e-3) / 1e9);
        } else if (!good) {
            printf("  shape %d x %d, %s: error %.2e\n", rows, cols, NAMES[k], err);
        }
    }

    // backward on y from the CPU (rounded to float), so it doesn't depend on the forward
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        y[i] = (float)ref[i];
        dy[i] = uniform(rng);
        dx0[i] = timing ? 0.0f : uniform(rng);
    }
    backward_ref(y.data(), dy.data(), dx0.data(), dx_ref.data(), scale.data(), rows, cols);
    CUDA_CHECK(cudaMemcpy(d.y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.dy, dy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.dx, dx0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    launch_backward(d.y, d.dy, d.dx, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(got.data(), d.dx, n * sizeof(float), cudaMemcpyDeviceToHost));
    double err = max_error(got.data(), dx_ref.data(), scale.data(), n);
    bool good = err < 1e-5;
    ok = ok && good;
    if (timing) {
        // reads y, dy and dx and writes dx: 16 bytes per value
        float ms = time_ms([&] { launch_backward(d.y, d.dy, d.dx, rows, cols); });
        printf("  %-9s %-10.2e %-6s %-10.4f %-8.1f\n", "backward", err, good ? "ok" : "WRONG", ms,
               16.0 * n / (ms * 1e-3) / 1e9);
    } else if (!good) {
        printf("  shape %d x %d, backward: error %.2e\n", rows, cols, err);
    }
    return ok;
}

int main() {
    const size_t max_n = 1 << 24;
    Buffers d;
    CUDA_CHECK(cudaMalloc(&d.x, max_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.y, max_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dy, max_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dx, max_n * sizeof(float)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_backward_formula();

    // widths around a warp and a block, and row counts around the naive version's block
    bool small_ok = true;
    for (int cols : {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097}) {
        for (int rows : {1, 4, 257}) {
            small_ok = run_shape(rows, cols, d, rng, false) && small_ok;
        }
    }
    printf("small shapes: %s\n", small_ok ? "ok" : "WRONG");
    ok = ok && small_ok;

    // the same 2^24 values as one tall, one square and one wide matrix. forward reads x
    // and writes y, 8 bytes per value
    int shapes[3][2] = {{65536, 256}, {4096, 4096}, {256, 65536}};
    for (auto [rows, cols] : shapes) {
        printf("\n%d rows x %d cols\n", rows, cols);
        printf("  %-9s %-10s %-6s %-10s %-8s\n", "kernel", "max error", "check", "ms", "GB/s");
        ok = run_shape(rows, cols, d, rng, true) && ok;
    }

    // a plain copy of the same bytes, as a ceiling for what moving memory can do
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemcpy(d.y, d.x, max_n * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; it++) {
        CUDA_CHECK(cudaMemcpy(d.y, d.x, max_n * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float copy_ms;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start, stop));
    copy_ms /= 100;
    printf("\ncudaMemcpy device to device: %.4f ms, %.1f GB/s\n", copy_ms,
           8.0 * max_n / (copy_ms * 1e-3) / 1e9);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d.x));
    CUDA_CHECK(cudaFree(d.y));
    CUDA_CHECK(cudaFree(d.dy));
    CUDA_CHECK(cudaFree(d.dx));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
