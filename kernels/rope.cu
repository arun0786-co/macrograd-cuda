// rotary position embedding (Su et al., 2021) like macrograd's rotary_embedding, for x of
// shape (..., T, d) stored row by row: row r is position start + r % T, and each pair of
// features (2i, 2i + 1) is turned by the angle p * w_i, w_i = base^(-2i / d):
//
//    out[2i]     = x[2i] cos - x[2i + 1] sin
//    out[2i + 1] = x[2i] sin + x[2i + 1] cos
//
// the backward turns each pair's gradient back by the same angle, and adds it into dx:
//
//    dx[2i]     += dy[2i] cos + dy[2i + 1] sin
//    dx[2i + 1] += -dy[2i] sin + dy[2i + 1] cos
//
// the angles get large: at position 100,000 the fastest pair has turned 100,000 radians,
// and a float angle there is off by up to 0.004. so the CPU works out cos and sin in
// double for the T positions and d / 2 speeds once, and the kernels read them from a
// (T, d / 2) table of float2. one thread per pair.
//
// checked against a CPU version in double and against macrograd on the same numbers
// (tests/rope_macrograd.txt, from tests/make_rope_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/rope.cu -o build/rope && ./build/rope

#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#include "common.h"

const int THREADS = 256;

// (cos, sin) of position start + t and speed i at [t * d / 2 + i], in double, then float
std::vector<float2> rope_table(int T, int d, double base, int start) {
    std::vector<float2> table((size_t)T * d / 2);
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < d / 2; i++) {
            double angle = (double)(start + t) * pow(base, -2.0 * i / d);
            table[(size_t)t * d / 2 + i] = make_float2((float)cos(angle), (float)sin(angle));
        }
    }
    return table;
}

template <bool BACKWARD>
__global__ void rope_kernel(const float *in, float *out, const float2 *table, size_t rows, int T, int d) {
    size_t pair = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = d / 2;
    if (pair >= rows * half) {
        return;
    }
    size_t r = pair / half;
    int i = pair % half;
    float2 cs = table[(r % T) * half + i];
    float a = in[2 * pair], b = in[2 * pair + 1];
    if (BACKWARD) {
        out[2 * pair] += a * cs.x + b * cs.y;
        out[2 * pair + 1] += -a * cs.y + b * cs.x;
    } else {
        out[2 * pair] = a * cs.x - b * cs.y;
        out[2 * pair + 1] = a * cs.y + b * cs.x;
    }
}

// table from rope_table(T, d, base, start), already on the GPU
void rope_forward(const float *x, float *out, const float2 *table, size_t rows, int T, int d) {
    size_t pairs = rows * (d / 2);
    rope_kernel<false><<<(unsigned)((pairs + THREADS - 1) / THREADS), THREADS>>>(x, out, table, rows, T, d);
}

void rope_backward(const float *dy, float *dx, const float2 *table, size_t rows, int T, int d) {
    size_t pairs = rows * (d / 2);
    rope_kernel<true><<<(unsigned)((pairs + THREADS - 1) / THREADS), THREADS>>>(dy, dx, table, rows, T, d);
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

struct Device {
    float *x, *out;
    float2 *table;
};

// forward and backward against the formulas in double with exact angles; errors relative
// to |a| + |b| of the pair (plus |dx0| for the backward). a float cos or sin is off by up to
// 6e-8, and at position p a float angle would be off by 6e-8 * p, which the test would see
bool check_shape(size_t rows, int T, int d, double base, int start, Device dv, std::mt19937 &rng, double *worst) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    size_t n = rows * d;
    std::vector<float> x(n), dx0(n);
    for (auto &v : x) v = u(rng);
    for (auto &v : dx0) v = u(rng);
    std::vector<double> fwd(n), bwd(n), size(n), bsize(n);
    for (size_t r = 0; r < rows; r++) {
        for (int i = 0; i < d / 2; i++) {
            double angle = (double)(start + r % T) * pow(base, -2.0 * i / d), c = cos(angle), s = sin(angle);
            size_t k = r * d + 2 * i;
            double a = x[k], b = x[k + 1];
            fwd[k] = a * c - b * s, fwd[k + 1] = a * s + b * c;
            bwd[k] = dx0[k] + a * c + b * s, bwd[k + 1] = dx0[k + 1] - a * s + b * c;
            size[k] = size[k + 1] = fabs(a) + fabs(b);
            bsize[k] = size[k] + fabs(dx0[k]), bsize[k + 1] = size[k] + fabs(dx0[k + 1]);
        }
    }
    std::vector<float2> table = rope_table(T, d, base, start);
    CUDA_CHECK(cudaMemcpy(dv.table, table.data(), table.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    bool ok = true;
    for (int pass = 0; pass < 2; pass++) {
        CUDA_CHECK(cudaMemset(dv.out, 0xff, (n + PAST) * sizeof(float)));
        if (pass == 0) {
            rope_forward(dv.x, dv.out, dv.table, rows, T, d);
        } else {
            CUDA_CHECK(cudaMemcpy(dv.out, dx0.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            rope_backward(dv.x, dv.out, dv.table, rows, T, d);  // x plays dy
        }
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> got(n + PAST);
        CUDA_CHECK(cudaMemcpy(got.data(), dv.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const std::vector<double> &want = pass == 0 ? fwd : bwd, &sc = pass == 0 ? size : bsize;
        double e = 0.0;
        for (size_t i = 0; i < n; i++) {
            double xx = fabs(got[i] - want[i]) / fmax(sc[i], 1e-30);
            e = fmax(e, std::isnan(xx) ? INFINITY : xx);
        }
        bool past = true;
        for (size_t i = n; i < got.size(); i++) past = past && std::isnan(got[i]);
        *worst = fmax(*worst, e);
        if (!(e < 3e-7) || !past) {
            printf("  rows %zu, T %d, d %d, start %d, %s: error %.2e%s\n", rows, T, d, start,
                   pass == 0 ? "forward" : "backward", e, past ? "" : ", wrote past the end");
            ok = false;
        }
    }
    return ok;
}

bool check_macrograd(const char *path, Device dv) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    size_t rows;
    int T, d, start;
    double base;
    f >> rows >> T >> d >> start >> base;
    size_t n = rows * d;
    std::vector<double> arr[4];
    for (int k = 0; k < 4; k++) {
        arr[k].resize(n);
        for (auto &v : arr[k]) f >> v;
    }
    if (!f) {
        printf("macrograd: couldn't read %s\n", path);
        return false;
    }
    std::vector<float2> table = rope_table(T, d, base, start);
    CUDA_CHECK(cudaMemcpy(dv.table, table.data(), table.size() * sizeof(float2), cudaMemcpyHostToDevice));
    std::vector<float> x(arr[0].begin(), arr[0].end()), dy(arr[1].begin(), arr[1].end());
    CUDA_CHECK(cudaMemcpy(dv.x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    rope_forward(dv.x, dv.out, dv.table, rows, T, d);
    std::vector<float> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), dv.out, n * sizeof(float), cudaMemcpyDeviceToHost));
    double w = 0.0;
    for (size_t i = 0; i < n; i++) w = fmax(w, fabs(got[i] - arr[2][i]));
    CUDA_CHECK(cudaMemcpy(dv.x, dy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dv.out, 0, n * sizeof(float)));
    rope_backward(dv.x, dv.out, dv.table, rows, T, d);
    CUDA_CHECK(cudaMemcpy(got.data(), dv.out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) w = fmax(w, std::isnan(got[i]) ? INFINITY : fabs(got[i] - arr[3][i]));
    bool good = w < 1e-6;
    printf("macrograd (%zu rows, T %d, d %d, from position %d): largest difference %.2e, %s\n", rows, T, d, start, w,
           good ? "ok" : "WRONG");
    return good;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    Device dv;
    CUDA_CHECK(cudaMalloc(&dv.x, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv.out, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv.table, (size_t)(1 << 22) * sizeof(float2)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/rope_macrograd.txt", dv);
    // rows, T, d, base, start: one pair, heads of 2 to 128, many rows per position, a
    // late start where float angles would be visibly wrong
    double worst = 0.0;
    bool shapes_ok = true;
    struct S { size_t rows; int T, d; double base; int start; } shapes[] = {
        {1, 1, 2, 10000.0, 0}, {7, 7, 2, 10000.0, 0}, {48, 16, 8, 10000.0, 3}, {4096, 128, 64, 10000.0, 0},
        {1000, 100, 128, 500.0, 0}, {512, 512, 32, 10000.0, 100000}};
    for (auto &s : shapes) shapes_ok = check_shape(s.rows, s.T, s.d, s.base, s.start, dv, rng, &worst) && shapes_ok;
    printf("shapes: %s, largest error %.2e\n", shapes_ok ? "ok" : "WRONG", worst);
    // a turn and its backward undo each other: rope_backward(rope_forward(x)) = x
    {
        size_t rows = 4096;
        int T = 256, d = 64;
        std::vector<float> x(rows * d);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto &v : x) v = u(rng);
        std::vector<float2> table = rope_table(T, d, 10000.0, 7);
        CUDA_CHECK(cudaMemcpy(dv.table, table.data(), table.size() * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        float *back = dv.x + x.size();
        CUDA_CHECK(cudaMemset(back, 0, x.size() * sizeof(float)));
        rope_forward(dv.x, dv.out, dv.table, rows, T, d);
        rope_backward(dv.out, back, dv.table, rows, T, d);
        std::vector<float> got(x.size());
        CUDA_CHECK(cudaMemcpy(got.data(), back, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double e = 0.0;
        for (size_t i = 0; i < x.size(); i++) e = fmax(e, fabs(got[i] - x[i]));
        bool good = e < 1e-6;
        printf("backward undoes forward: largest difference %.2e, %s\n", e, good ? "ok" : "WRONG");
        ok = ok && good;
    }
    ok = ok && shapes_ok;

    // 16 heads, 1024 positions, d 64, 64 sequences: 2^26 / 4 values
    size_t rows = (size_t)1 << 18;
    int T = 1024, d = 64;
    std::vector<float2> table = rope_table(T, d, 10000.0, 0);
    CUDA_CHECK(cudaMemcpy(dv.table, table.data(), table.size() * sizeof(float2), cudaMemcpyHostToDevice));
    size_t n = rows * d;
    float f = time_ms([&] { rope_forward(dv.x, dv.out, dv.table, rows, T, d); });
    float b = time_ms([&] { rope_backward(dv.x, dv.out, dv.table, rows, T, d); });
    float c = time_ms([&] { CUDA_CHECK(cudaMemcpy(dv.out, dv.x, n * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("\n%zu values: forward %.3f ms %.1f GB/s, backward %.3f ms %.1f GB/s, memcpy %.3f ms %.1f GB/s\n", n, f,
           8.0 * n / (f * 1e-3) / 1e9, b, 12.0 * n / (b * 1e-3) / 1e9, c, 8.0 * n / (c * 1e-3) / 1e9);

    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
