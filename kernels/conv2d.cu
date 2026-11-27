// conv2d forward like macrograd's tensor.conv2d: images x (N, C, H, W), filters w (O, C,
// KH, KW), no bias, with stride and zero padding:
//
//    out[n][o][i][j] = sum over c, a, b of x[n][c][i*stride + a - padding][j*stride + b - padding] * w[o][c][a][b]
//
// output size OH = (H + 2 padding - KH) / stride + 1, and the same for OW. two ways:
//
// 1. direct: one thread per output value, looping over the C * KH * KW taps. the threads
//    of a warp take neighbouring j, so they share n, o and i: they read the same filter
//    value at each step, and neighbouring image values
// 2. im2col: write every patch of every image out as a column of a matrix cols, with
//    C * KH * KW rows and N * OH * OW columns. then one matmul, w (O, C*KH*KW) @ cols, gives
//    every output at once, and a last kernel moves it from (O, N, OH*OW) to (N, O, OH*OW).
//    the matmul is the linear layer's, so it runs at matmul speed, but cols holds each
//    image value up to KH * KW times
//
// checked against a CPU version in double and against macrograd on the same numbers
// (tests/conv2d_macrograd.txt, from tests/make_conv2d_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/conv2d.cu -lcublas -o build/conv2d && ./build/conv2d

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "common.h"

#ifndef MACROGRAD_NO_MAIN
#define MACROGRAD_NO_MAIN
#define CONV_OWNS_MAIN
#endif
namespace lin {
#include "linear.cu"
}
#ifdef CONV_OWNS_MAIN
#undef MACROGRAD_NO_MAIN
#endif

const int THREADS = 256;

struct Conv {
    int N, C, H, W, O, KH, KW, stride, padding;
    int OH() const { return (H + 2 * padding - KH) / stride + 1; }
    int OW() const { return (W + 2 * padding - KW) / stride + 1; }
};

// ---- 1. direct ----

__global__ void conv_direct_kernel(const float *x, const float *w, float *out, Conv p, int OH, int OW) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)p.N * p.O * OH * OW;
    if (idx >= total) {
        return;
    }
    int j = idx % OW, i = (idx / OW) % OH, o = (idx / ((size_t)OW * OH)) % p.O, n = idx / ((size_t)OW * OH * p.O);
    float s = 0.0f;
    for (int c = 0; c < p.C; c++) {
        const float *xc = x + ((size_t)n * p.C + c) * p.H * p.W;
        const float *wc = w + ((size_t)o * p.C + c) * p.KH * p.KW;
        for (int a = 0; a < p.KH; a++) {
            int y = i * p.stride + a - p.padding;
            if (y < 0 || y >= p.H) continue;
            for (int b = 0; b < p.KW; b++) {
                int xx = j * p.stride + b - p.padding;
                if (xx < 0 || xx >= p.W) continue;
                s += xc[(size_t)y * p.W + xx] * wc[a * p.KW + b];
            }
        }
    }
    out[idx] = s;
}

void conv_direct(const float *x, const float *w, float *out, float *, Conv p) {
    size_t total = (size_t)p.N * p.O * p.OH() * p.OW();
    conv_direct_kernel<<<(unsigned)((total + THREADS - 1) / THREADS), THREADS>>>(x, w, out, p, p.OH(), p.OW());
}

// ---- 2. im2col ----

// cols[(c*KH + a)*KW + b][n*OH*OW + i*OW + j] = the image value that tap (c, a, b) of
// output (n, i, j) reads, 0 in the padding. one thread per value of cols; neighbouring
// threads write neighbouring columns
__global__ void im2col_kernel(const float *x, float *cols, Conv p, int OH, int OW) {
    size_t P = (size_t)p.N * OH * OW;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * p.C * p.KH * p.KW) {
        return;
    }
    size_t col = idx % P;
    int row = idx / P;
    int b = row % p.KW, a = (row / p.KW) % p.KH, c = row / (p.KW * p.KH);
    int j = col % OW, i = (col / OW) % OH, n = col / ((size_t)OW * OH);
    int y = i * p.stride + a - p.padding, xx = j * p.stride + b - p.padding;
    cols[idx] = y >= 0 && y < p.H && xx >= 0 && xx < p.W ? x[(((size_t)n * p.C + c) * p.H + y) * p.W + xx] : 0.0f;
}

// tmp is (O, N, OH*OW); out is (N, O, OH*OW)
__global__ void to_nchw(const float *tmp, float *out, int N, int O, int HW) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)N * O * HW) {
        int s = idx % HW, o = (idx / HW) % O, n = idx / ((size_t)HW * O);
        out[idx] = tmp[((size_t)o * N + n) * HW + s];
    }
}

size_t im2col_scratch(Conv p) {
    size_t P = (size_t)p.N * p.OH() * p.OW();
    return P * p.C * p.KH * p.KW + (size_t)p.O * P;  // cols, then the matmul's output
}

// scratch holds im2col_scratch(p) floats
void conv_im2col(const float *x, const float *w, float *out, float *scratch, Conv p) {
    int OH = p.OH(), OW = p.OW(), K = p.C * p.KH * p.KW;
    size_t P = (size_t)p.N * OH * OW;
    float *cols = scratch, *tmp = scratch + P * K;
    size_t n = P * K;
    im2col_kernel<<<(unsigned)((n + THREADS - 1) / THREADS), THREADS>>>(x, cols, p, OH, OW);
    // w is (O, K) row by row, cols is (K, P): tmp (O, P) = w @ cols
    lin::matmul<false, false>(w, cols, tmp, nullptr, false, p.O, (int)P, K);
    size_t m = (size_t)p.O * P;
    to_nchw<<<(unsigned)((m + THREADS - 1) / THREADS), THREADS>>>(tmp, out, p.N, p.O, OH * OW);
}

typedef void (*ConvFn)(const float *, const float *, float *, float *, Conv);
const int VERSIONS = 2;
const char *NAMES[] = {"direct", "im2col"};
const ConvFn CONVS[] = {conv_direct, conv_im2col};

template <typename F>
float time_ms(F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 10; it++) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / 10;
}

// ---- tests ----

struct Device {
    float *x, *w, *out, *scratch;
};

const size_t CAP = (size_t)1 << 24, SCRATCH = (size_t)1 << 27, PAST = 4096;

// the CPU in double, and for the error the sum of |x * w| over each output's taps
void cpu_conv(const std::vector<float> &x, const std::vector<float> &w, Conv p, std::vector<double> &out,
              std::vector<double> &size) {
    int OH = p.OH(), OW = p.OW();
    out.assign((size_t)p.N * p.O * OH * OW, 0.0);
    size.assign(out.size(), 0.0);
    for (int n = 0; n < p.N; n++)
        for (int o = 0; o < p.O; o++)
            for (int i = 0; i < OH; i++)
                for (int j = 0; j < OW; j++) {
                    double s = 0.0, a_s = 0.0;
                    for (int c = 0; c < p.C; c++)
                        for (int a = 0; a < p.KH; a++)
                            for (int b = 0; b < p.KW; b++) {
                                int y = i * p.stride + a - p.padding, xx = j * p.stride + b - p.padding;
                                if (y < 0 || y >= p.H || xx < 0 || xx >= p.W) continue;
                                double t = (double)x[(((size_t)n * p.C + c) * p.H + y) * p.W + xx] *
                                           w[(((size_t)o * p.C + c) * p.KH + a) * p.KW + b];
                                s += t;
                                a_s += fabs(t);
                            }
                    size_t idx = (((size_t)n * p.O + o) * OH + i) * OW + j;
                    out[idx] = s;
                    size[idx] = a_s;
                }
}

void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

bool check_shape(Conv p, Device d, std::mt19937 &rng, double *worst) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x((size_t)p.N * p.C * p.H * p.W), w((size_t)p.O * p.C * p.KH * p.KW);
    for (auto &v : x) v = u(rng);
    for (auto &v : w) v = u(rng);
    std::vector<double> want, size;
    cpu_conv(x, w, p, want, size);
    // nan after the end of x and w, so reading past them shows
    CUDA_CHECK(cudaMemset(d.x, 0xff, (x.size() + PAST) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.w, 0xff, (w.size() + PAST) * sizeof(float)));
    upload(d.x, x);
    upload(d.w, w);
    bool ok = true;
    for (int v = 0; v < VERSIONS; v++) {
        CUDA_CHECK(cudaMemset(d.out, 0xff, (want.size() + PAST) * sizeof(float)));
        CONVS[v](d.x, d.w, d.out, d.scratch, p);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> got(want.size() + PAST);
        CUDA_CHECK(cudaMemcpy(got.data(), d.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double e = 0.0;
        for (size_t i = 0; i < want.size(); i++) {
            double x_ = fabs(got[i] - want[i]) / fmax(size[i], 1e-30);
            e = fmax(e, std::isnan(x_) ? INFINITY : x_);
        }
        bool past = true;
        for (size_t i = want.size(); i < got.size(); i++) past = past && std::isnan(got[i]);
        *worst = fmax(*worst, e);
        double tol = 8.0 * sqrt((double)p.C * p.KH * p.KW) * 6e-8;
        if (!(e < tol) || !past) {
            printf("  N %d C %d %dx%d O %d k %dx%d s %d p %d, %s: error %.2e (tol %.2e)%s\n", p.N, p.C, p.H, p.W, p.O,
                   p.KH, p.KW, p.stride, p.padding, NAMES[v], e, tol, past ? "" : ", wrote past the end");
            ok = false;
        }
    }
    return ok;
}

bool check_macrograd(const char *path, Device d) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    Conv p;
    f >> p.N >> p.C >> p.H >> p.W >> p.O >> p.KH >> p.KW;
    std::vector<double> x((size_t)p.N * p.C * p.H * p.W), w((size_t)p.O * p.C * p.KH * p.KW);
    for (auto &v : x) f >> v;
    for (auto &v : w) f >> v;
    upload(d.x, std::vector<float>(x.begin(), x.end()));
    upload(d.w, std::vector<float>(w.begin(), w.end()));
    bool ok = true;
    for (int setting = 0; setting < 2; setting++) {
        f >> p.stride >> p.padding;
        std::vector<double> want((size_t)p.N * p.O * p.OH() * p.OW());
        for (auto &v : want) f >> v;
        if (!f) {
            printf("macrograd: couldn't read %s\n", path);
            return false;
        }
        for (int v = 0; v < VERSIONS; v++) {
            CONVS[v](d.x, d.w, d.out, d.scratch, p);
            std::vector<float> got(want.size());
            CUDA_CHECK(cudaMemcpy(got.data(), d.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double w_ = 0.0;
            for (size_t i = 0; i < got.size(); i++) w_ = fmax(w_, std::isnan(got[i]) ? INFINITY : fabs(got[i] - want[i]));
            bool good = w_ < 1e-6;
            printf("macrograd (stride %d, padding %d), %s: largest difference %.2e, %s\n", p.stride, p.padding,
                   NAMES[v], w_, good ? "ok" : "WRONG");
            ok = ok && good;
        }
    }
    return ok;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    Device d;
    CUDA_CHECK(cudaMalloc(&d.x, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.w, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.out, CAP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.scratch, SCRATCH * sizeof(float)));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/conv2d_macrograd.txt", d);
    // N C H W O KH KW stride padding: 1x1 filters, a filter as big as the image, stride
    // bigger than the filter, wide padding, odd sizes, many channels
    Conv shapes[] = {{1, 1, 1, 1, 1, 1, 1, 1, 0},  {1, 1, 5, 5, 1, 5, 5, 1, 0},   {2, 3, 7, 6, 4, 3, 3, 1, 1},
                     {2, 3, 7, 6, 4, 3, 3, 2, 0},  {3, 2, 9, 9, 5, 2, 2, 3, 0},   {1, 4, 8, 5, 3, 3, 5, 1, 2},
                     {4, 16, 12, 12, 8, 1, 1, 1, 0}, {2, 64, 10, 10, 32, 3, 3, 1, 1}, {8, 3, 28, 28, 16, 5, 5, 2, 2}};
    double worst = 0.0;
    bool shapes_ok = true;
    for (auto &p : shapes) shapes_ok = check_shape(p, d, rng, &worst) && shapes_ok;
    printf("shapes: %s, largest error %.2e\n", shapes_ok ? "ok" : "WRONG", worst);
    ok = ok && shapes_ok;

    // GFLOPS counts 2 flops per tap of every output, padding taps included
    CUDA_CHECK(cudaMemset(d.x, 0, CAP * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.w, 0, CAP * sizeof(float)));
    printf("\n%-34s %-20s %-20s %-12s\n", "N C HxW -> O, k, pad", "direct", "im2col", "cols MB");
    Conv timed[] = {{64, 1, 28, 28, 32, 3, 3, 1, 1}, {64, 32, 28, 28, 64, 3, 3, 1, 1}, {64, 64, 32, 32, 64, 3, 3, 1, 1},
                    {32, 128, 28, 28, 128, 3, 3, 1, 1}, {64, 256, 14, 14, 256, 1, 1, 1, 0}};
    for (auto &p : timed) {
        double flops = 2.0 * p.N * p.O * p.OH() * p.OW() * p.C * p.KH * p.KW;
        float a = time_ms([&] { conv_direct(d.x, d.w, d.out, d.scratch, p); });
        float b = time_ms([&] { conv_im2col(d.x, d.w, d.out, d.scratch, p); });
        char name[64];
        snprintf(name, sizeof name, "%d %d %dx%d -> %d, %dx%d, %d", p.N, p.C, p.H, p.W, p.O, p.KH, p.KW, p.padding);
        size_t P = (size_t)p.N * p.OH() * p.OW();
        printf("%-34s %7.3f ms %6.0f GF  %7.3f ms %6.0f GF  %8.1f\n", name, a, flops / (a * 1e-3) / 1e9, b,
               flops / (b * 1e-3) / 1e9, P * p.C * p.KH * p.KW * 4 / 1e6);
    }

    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
