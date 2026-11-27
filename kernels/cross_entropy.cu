// cross entropy with integer labels, for logits x (R, C) stored row by row and one label
// per row: softmax over each row, then the mean of -log(probability of the label):
//
//    loss = 1/R * sum_r ( log sum_j exp(x[r][j]) - x[r][label_r] )
//
// with m_r the row max and d_r = sum_j exp(x[r][j] - m_r), the row's term is
// log(d_r) + m_r - x[r][label_r]. subtracting m keeps exp from overflowing, and m - x is
// a difference of two nearby floats, so it stays exact even when the logits are 1000.
//
// the backward of the mean loss, times the gradient dloss that arrives at it, is
//
//    dx[r][j] += dloss / R * (softmax(x)[r][j] - [j == label_r])
//
// forward is one kernel: the online max and sum from softmax.cu, one pass over the row,
// then the row's loss. it saves m_r and 1 / d_r so the backward can make the softmax
// without going over the row again: softmax(x)[r][j] = exp(x[r][j] - m_r) / d_r. a second
// small kernel takes the mean of the row losses in a fixed order, so the loss is the
// same on every run. three ways to give rows to threads:
//
// 1. thread per row: each thread walks its own row twice, for the max and then the sum.
//    for short rows, like the 10 classes of mnist, where most lanes of a warp would have
//    nothing to do and the shuffles would cost more than the row
// 2. warp per row: 32 threads share a row, 8 rows to a block of 256. no shared memory or
//    __syncthreads
// 3. block per row: 256 threads share a row, for long rows like a 50,000 token vocabulary
//
// cross_entropy() picks one by the number of classes, with cutoffs from the timings in
// the readme.
// a label outside 0..C-1 gives a nan loss instead of reading outside the row.
//
// checked against a CPU version in double, and the backward formula against finite
// differences of the CPU loss.
//
// build and run: nvcc -O3 -arch=native kernels/cross_entropy.cu -o build/cross_entropy && ./build/cross_entropy

#include <cmath>
#include <random>
#include <vector>
#include "common.h"

const int THREADS = 256;

// m is the largest value seen so far and d is the sum of exp(value - m) over them.
// m = -inf, d = 0 means nothing seen yet
struct MD {
    float m, d;
};

// two partial results put together: move both sums onto the larger max
__device__ MD md_merge(MD a, MD b) {
    float m = fmaxf(a.m, b.m);
    // both empty: exp(-inf - -inf) would be nan
    if (m == -INFINITY) {
        return a;
    }
    return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

// every lane ends up with the result of all 32
__device__ MD warp_reduce_md(MD v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        MD other = {__shfl_xor_sync(0xffffffff, v.m, offset), __shfl_xor_sync(0xffffffff, v.d, offset)};
        v = md_merge(v, other);
    }
    return v;
}

// every thread of the block gets the result. blockDim.x must be a multiple of 32
__device__ MD block_reduce_md(MD v) {
    __shared__ MD warps[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_reduce_md(v);
    if (lane == 0) {
        warps[warp] = v;
    }
    __syncthreads();
    v = lane < blockDim.x / 32 ? warps[lane] : MD{-INFINITY, 0.0f};
    return warp_reduce_md(v);
}

// what each thread adds to the row's (m, d), going through every `step`-th value from `first`.
// a running max and sum: a bigger value moves the sum onto it first
__device__ MD row_part(const float *xr, int C, int first, int step) {
    MD v = {-INFINITY, 0.0f};
    for (int j = first; j < C; j += step) {
        v = md_merge(v, {xr[j], 1.0f});
    }
    return v;
}

// the row's loss from its (m, d), and m and 1 / d saved for the backward
__device__ void finish_row(const float *xr, int C, int label, MD v, float *row_loss, float2 *stats, int r) {
    float x_label = label >= 0 && label < C ? xr[label] : NAN;
    row_loss[r] = logf(v.d) + (v.m - x_label);
    stats[r] = make_float2(v.m, 1.0f / v.d);
}

// ---- 1. thread per row ----

__global__ void ce_forward_thread(const float *x, const int *labels, float *row_loss, float2 *stats, int R, int C) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) {
        return;
    }
    const float *xr = x + (size_t)r * C;
    float m = -INFINITY;
    for (int j = 0; j < C; j++) {
        m = fmaxf(m, xr[j]);
    }
    float d = 0.0f;
    for (int j = 0; j < C; j++) {
        d += expf(xr[j] - m);
    }
    finish_row(xr, C, labels[r], {m, d}, row_loss, stats, r);
}

// ---- 2. warp per row ----

__global__ void ce_forward_warp(const float *x, const int *labels, float *row_loss, float2 *stats, int R, int C) {
    int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    // a whole warp has the same r, so it leaves together and the shuffles below see all 32 lanes
    if (r >= R) {
        return;
    }
    const float *xr = x + (size_t)r * C;
    MD v = warp_reduce_md(row_part(xr, C, lane, 32));
    if (lane == 0) {
        finish_row(xr, C, labels[r], v, row_loss, stats, r);
    }
}

// ---- 3. block per row ----

__global__ void ce_forward_block(const float *x, const int *labels, float *row_loss, float2 *stats, int C) {
    int r = blockIdx.x;
    const float *xr = x + (size_t)r * C;
    MD v = block_reduce_md(row_part(xr, C, threadIdx.x, blockDim.x));
    if (threadIdx.x == 0) {
        finish_row(xr, C, labels[r], v, row_loss, stats, r);
    }
}

// ---- the mean ----

// one block of 1024: thread t adds rows t, t + 1024, ..., then the 1024 partial sums are
// added in a tree. the order never changes, so neither does the result
__global__ void mean_rows(const float *row_loss, float *loss, int R) {
    __shared__ float part[1024];
    float s = 0.0f;
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        s += row_loss[r];
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            part[threadIdx.x] += part[threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *loss = part[0] / R;
    }
}

// ---- backward ----

// one thread per value. dloss is on the GPU, so the loss can come from a later kernel
// without a copy back to the CPU
__global__ void ce_backward(const float *x, const int *labels, const float2 *stats, const float *dloss, float *dx,
                            int R, int C) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)R * C) {
        return;
    }
    int r = i / C, j = i % C;
    float2 s = stats[r];
    float p = expf(x[i] - s.x) * s.y;
    dx[i] += *dloss / R * (p - (j == labels[r] ? 1.0f : 0.0f));
}

// ---- launches ----

// row_loss holds R floats and stats R float2s, both reused by the backward
void forward_thread(const float *x, const int *labels, float *row_loss, float2 *stats, float *loss, int R, int C) {
    ce_forward_thread<<<(R + THREADS - 1) / THREADS, THREADS>>>(x, labels, row_loss, stats, R, C);
    mean_rows<<<1, 1024>>>(row_loss, loss, R);
}

void forward_warp(const float *x, const int *labels, float *row_loss, float2 *stats, float *loss, int R, int C) {
    int rows_per_block = THREADS / 32;
    ce_forward_warp<<<(R + rows_per_block - 1) / rows_per_block, THREADS>>>(x, labels, row_loss, stats, R, C);
    mean_rows<<<1, 1024>>>(row_loss, loss, R);
}

void forward_block(const float *x, const int *labels, float *row_loss, float2 *stats, float *loss, int R, int C) {
    ce_forward_block<<<R, THREADS>>>(x, labels, row_loss, stats, C);
    mean_rows<<<1, 1024>>>(row_loss, loss, R);
}

const int THREAD_MAX_C = 16, WARP_MAX_C = 1024;

void cross_entropy(const float *x, const int *labels, float *row_loss, float2 *stats, float *loss, int R, int C) {
    if (C <= THREAD_MAX_C) {
        forward_thread(x, labels, row_loss, stats, loss, R, C);
    } else if (C <= WARP_MAX_C) {
        forward_warp(x, labels, row_loss, stats, loss, R, C);
    } else {
        forward_block(x, labels, row_loss, stats, loss, R, C);
    }
}

void cross_entropy_backward(const float *x, const int *labels, const float2 *stats, const float *dloss, float *dx,
                            int R, int C) {
    size_t n = (size_t)R * C;
    ce_backward<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS>>>(x, labels, stats, dloss, dx, R, C);
}

typedef void (*Forward)(const float *, const int *, float *, float2 *, float *, int, int);
const int FORWARDS_N = 4;
const char *NAMES[] = {"thread", "warp", "block", "chosen"};
const Forward FORWARDS[] = {forward_thread, forward_warp, forward_block, cross_entropy};

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

// ---- CPU side ----

// the loss of each row in double, with x[r][label] given separately so the finite
// difference check can move one value. a bad label gives nan, as on the GPU
double cpu_row_loss(const float *xr, int C, int label, double nudge_at = -1, double nudge = 0.0) {
    if (label < 0 || label >= C) {
        return NAN;
    }
    auto val = [&](int j) { return (double)xr[j] + (j == nudge_at ? nudge : 0.0); };
    double m = -INFINITY;
    for (int j = 0; j < C; j++) m = fmax(m, val(j));
    double d = 0.0;
    for (int j = 0; j < C; j++) d += exp(val(j) - m);
    return log(d) + m - val(label);
}

struct Case {
    std::vector<float> x, dx0;
    std::vector<int> labels;
};

// uniform in [-10, 10]. every 5th row is shifted by +1000 and every 7th by -1000, and
// every 3rd row has every 4th value set to -inf like an attention mask, except at the
// label. row 0 has its label at column 0 and row 1 at the last column
Case make_case(int R, int C, std::mt19937 &rng) {
    std::uniform_real_distribution<float> uniform(-10.0f, 10.0f), small(-1.0f, 1.0f);
    std::uniform_int_distribution<int> label(0, C - 1);
    Case c;
    c.x.resize((size_t)R * C);
    c.dx0.resize((size_t)R * C);
    c.labels.resize(R);
    for (int r = 0; r < R; r++) {
        c.labels[r] = r == 0 ? 0 : r == 1 ? C - 1 : label(rng);
        float shift = r % 5 == 4 ? 1000.0f : r % 7 == 6 ? -1000.0f : 0.0f;
        for (int j = 0; j < C; j++) {
            float v = uniform(rng) + shift;
            if (r % 3 == 2 && j % 4 == 1 && j != c.labels[r]) {
                v = -INFINITY;
            }
            c.x[(size_t)r * C + j] = v;
        }
    }
    for (auto &v : c.dx0) v = small(rng);
    return c;
}

struct Device {
    float *x, *dx, *row_loss, *loss, *dloss;
    float2 *stats;
    int *labels;
};

const size_t CAP = (size_t)1 << 24;

// one shape, every forward version and the backward after each, against the CPU.
// errors are relative to the size of what was added: 1 + |m| per row for the loss, since
// log d + (m - x) loses about that much to rounding, and |dx0| + p + 1 for dx
bool check_shape(int R, int C, Device d, std::mt19937 &rng, double *worst_loss, double *worst_dx) {
    Case c = make_case(R, C, rng);
    double want = 0.0, scale = 0.0;
    std::vector<double> m(R), dsum(R);
    for (int r = 0; r < R; r++) {
        const float *xr = &c.x[(size_t)r * C];
        double mr = -INFINITY;
        for (int j = 0; j < C; j++) mr = fmax(mr, (double)xr[j]);
        double ds = 0.0;
        for (int j = 0; j < C; j++) ds += exp(xr[j] - mr);
        m[r] = mr;
        dsum[r] = ds;
        want += cpu_row_loss(xr, C, c.labels[r]);
        scale += 1.0 + fabs(mr);
    }
    want /= R;
    scale /= R;
    const double dloss = 0.75;  // not 1, so a backward that ignores it fails

    CUDA_CHECK(cudaMemcpy(d.x, c.x.data(), c.x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.labels, c.labels.data(), R * sizeof(int), cudaMemcpyHostToDevice));
    float dl = (float)dloss;
    CUDA_CHECK(cudaMemcpy(d.dloss, &dl, sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (int v = 0; v < FORWARDS_N; v++) {
        CUDA_CHECK(cudaMemset(d.stats, 0xff, R * sizeof(float2)));
        CUDA_CHECK(cudaMemset(d.loss, 0xff, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d.dx, c.dx0.data(), c.dx0.size() * sizeof(float), cudaMemcpyHostToDevice));
        FORWARDS[v](d.x, d.labels, d.row_loss, d.stats, d.loss, R, C);
        cross_entropy_backward(d.x, d.labels, d.stats, d.dloss, d.dx, R, C);
        CUDA_CHECK(cudaGetLastError());
        float got;
        CUDA_CHECK(cudaMemcpy(&got, d.loss, sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<float> dx(c.x.size());
        CUDA_CHECK(cudaMemcpy(dx.data(), d.dx, dx.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double el = fabs(got - want) / scale;
        el = std::isnan(el) ? INFINITY : el;
        double ed = 0.0;
        for (int r = 0; r < R; r++) {
            for (int j = 0; j < C; j++) {
                size_t i = (size_t)r * C + j;
                double p = exp(c.x[i] - m[r]) / dsum[r];
                double g = c.dx0[i] + dloss / R * (p - (j == c.labels[r] ? 1.0 : 0.0));
                double e = fabs(dx[i] - g) / (fabs(c.dx0[i]) + p + 1.0);
                ed = fmax(ed, std::isnan(e) ? INFINITY : e);
            }
        }
        *worst_loss = fmax(*worst_loss, el);
        *worst_dx = fmax(*worst_dx, ed);
        if (!(el < 2e-6 && ed < 2e-6)) {
            printf("  %d x %d, %s: loss %.9g want %.9g (error %.2e), dx error %.2e\n", R, C, NAMES[v], got, want, el, ed);
            ok = false;
        }
    }
    return ok;
}

// labels outside the row give a nan loss, not a number read from the next row
bool check_bad_labels(Device d) {
    int R = 4, C = 10;
    std::vector<float> x(R * C, 1.0f);
    bool ok = true;
    for (int bad : {-1, C, C + 5}) {
        std::vector<int> labels = {3, bad, 0, 9};
        CUDA_CHECK(cudaMemcpy(d.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.labels, labels.data(), R * sizeof(int), cudaMemcpyHostToDevice));
        for (int v = 0; v < FORWARDS_N; v++) {
            FORWARDS[v](d.x, d.labels, d.row_loss, d.stats, d.loss, R, C);
            float got;
            CUDA_CHECK(cudaMemcpy(&got, d.loss, sizeof(float), cudaMemcpyDeviceToHost));
            if (!std::isnan(got)) {
                printf("  label %d, %s: loss %g, not nan\n", bad, NAMES[v], got);
                ok = false;
            }
        }
    }
    return ok;
}

// the backward formula, in double: p - one_hot against (loss(x + h) - loss(x - h)) / 2h
double check_formula(std::mt19937 &rng) {
    int R = 3, C = 7;
    Case c = make_case(R, C, rng);
    for (auto &v : c.x) v = std::isinf(v) ? v : fmodf(v, 10.0f);  // no 1000 shift, h has to matter
    double worst = 0.0, h = 1e-5;
    for (int r = 0; r < R; r++) {
        const float *xr = &c.x[(size_t)r * C];
        double mr = -INFINITY, ds = 0.0;
        for (int j = 0; j < C; j++) mr = fmax(mr, (double)xr[j]);
        for (int j = 0; j < C; j++) ds += exp(xr[j] - mr);
        for (int j = 0; j < C; j++) {
            if (std::isinf(xr[j])) continue;
            double numeric = (cpu_row_loss(xr, C, c.labels[r], j, h) - cpu_row_loss(xr, C, c.labels[r], j, -h)) / (2 * h);
            double formula = exp(xr[j] - mr) / ds - (j == c.labels[r] ? 1.0 : 0.0);
            worst = fmax(worst, fabs(numeric - formula));
        }
    }
    return worst;
}

int main() {
    Device d;
    CUDA_CHECK(cudaMalloc(&d.x, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dx, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.row_loss, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.stats, CAP * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d.labels, CAP * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dloss, sizeof(float)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    double fd = check_formula(rng);
    bool ok = fd < 1e-8;
    printf("backward formula vs finite differences: max diff %.1e %s\n", fd, fd < 1e-8 ? "ok" : "WRONG");

    // around a warp, a block and the 1024 switch, one row, and more rows than the mean's 1024 threads
    double worst_loss = 0.0, worst_dx = 0.0;
    bool shapes_ok = true;
    for (int R : {1, 2, 7, 33, 300, 2500}) {
        for (int C : {1, 2, 10, 31, 32, 33, 255, 256, 257, 1000, 1025, 5000}) {
            if ((size_t)R * C > 3000000) continue;
            shapes_ok = check_shape(R, C, d, rng, &worst_loss, &worst_dx) && shapes_ok;
        }
    }
    printf("shapes: %s, largest loss error %.2e, dx error %.2e\n", shapes_ok ? "ok" : "WRONG", worst_loss, worst_dx);
    bool bad_ok = check_bad_labels(d);
    printf("bad labels give nan: %s\n", bad_ok ? "ok" : "WRONG");
    ok = ok && shapes_ok && bad_ok;

    // timing on 2^24 logits in three shapes. forward reads each logit once, 4 bytes;
    // backward reads x and dx and writes dx, 12 bytes. the labels and per-row numbers
    // aren't counted
    CUDA_CHECK(cudaMemset(d.x, 0, CAP * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.labels, 0, CAP * sizeof(int)));
    float dl = 1.0f;
    CUDA_CHECK(cudaMemcpy(d.dloss, &dl, sizeof(float), cudaMemcpyHostToDevice));
    printf("\n%-16s %-10s %-10s %-10s\n", "rows x classes", "version", "ms", "GB/s");
    int shapes[][2] = {{1677721, 10}, {524288, 32}, {262144, 64}, {65536, 256}, {16384, 1024}, {4096, 4096}, {333, 50257}};
    for (auto &s : shapes) {
        int R = s[0], C = s[1];
        double n = (double)R * C;
        char shape[32];
        snprintf(shape, sizeof shape, "%d x %d", R, C);
        for (int v = 0; v < 3; v++) {
            float ms = time_ms([&] { FORWARDS[v](d.x, d.labels, d.row_loss, d.stats, d.loss, R, C); });
            printf("%-16s %-10s %-10.3f %-10.1f\n", v == 0 ? shape : "", NAMES[v], ms, 4 * n / (ms * 1e-3) / 1e9);
        }
        float ms = time_ms([&] { cross_entropy_backward(d.x, d.labels, d.stats, d.dloss, d.dx, R, C); });
        printf("%-16s %-10s %-10.3f %-10.1f\n", "", "backward", ms, 12 * n / (ms * 1e-3) / 1e9);
    }
    float ms = time_ms([&] { CUDA_CHECK(cudaMemcpy(d.dx, d.x, CAP * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("%-16s %-10s %-10.3f %-10.1f\n", "2^24 floats", "memcpy", ms, 8.0 * CAP / (ms * 1e-3) / 1e9);

    CUDA_CHECK(cudaFree(d.x));
    CUDA_CHECK(cudaFree(d.dx));
    CUDA_CHECK(cudaFree(d.row_loss));
    CUDA_CHECK(cudaFree(d.stats));
    CUDA_CHECK(cudaFree(d.labels));
    CUDA_CHECK(cudaFree(d.loss));
    CUDA_CHECK(cudaFree(d.dloss));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
