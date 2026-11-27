// rms norm over the last dim of x (R, C), stored row by row, with gamma (C,), the same as
// macrograd's nn.rms_norm and the norm LLaMA uses:
//
//    rstd = 1 / sqrt(mean(x^2) + eps),   x_hat = x * rstd,   y = gamma * x_hat
//
// backward, with dy the gradient at y and g = dy * gamma:
//
//    dx_j += rstd * (g_j - x_hat_j * mean(g * x_hat)),   dgamma += sum over rows of dy * x_hat
//
// it's layer_norm.cu without the mean: no beta, no mean(g) term, and since a sum of
// squares has nothing to cancel, no shift either. the same layouts: a warp per row up to
// 1024 columns and a block per row above, rstd saved for the backward, and dgamma summed
// over parts of the rows that a second kernel adds in a fixed order.
//
// checked against a CPU version in double and against macrograd on the same numbers
// (tests/rms_norm_macrograd.txt, from tests/make_rms_norm_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/rms_norm.cu -o build/rms_norm && ./build/rms_norm

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include "common.h"

const int THREADS = 256;

// the sum over a warp (GROUP = 32) or the whole block (GROUP = 256), which every thread of it gets
template <int GROUP>
__device__ float group_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (GROUP == 32) {
        return v;
    }
    __shared__ float warps[GROUP / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __syncthreads();  // an earlier call may still be reading warps[]
    if (lane == 0) {
        warps[warp] = v;
    }
    __syncthreads();
    v = lane < GROUP / 32 ? warps[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int GROUP>
__device__ void my_row(int *r, int *lane) {
    if (GROUP == 32) {
        *r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        *lane = threadIdx.x % 32;
    } else {
        *r = blockIdx.x;
        *lane = threadIdx.x;
    }
}

template <int GROUP>
__global__ void rms_forward(const float *x, const float *gamma, float *y, float *rstd_out, int R, int C, float eps) {
    int r, lane;
    my_row<GROUP>(&r, &lane);
    if (r >= R) {
        return;  // a whole warp has the same r
    }
    const float *xr = x + (size_t)r * C;
    float ss = 0.0f;
    for (int j = lane; j < C; j += GROUP) {
        ss += xr[j] * xr[j];
    }
    float rstd = 1.0f / sqrtf(group_sum<GROUP>(ss) / C + eps);
    float *yr = y + (size_t)r * C;
    for (int j = lane; j < C; j += GROUP) {
        yr[j] = gamma[j] * (xr[j] * rstd);
    }
    if (lane == 0) {
        rstd_out[r] = rstd;
    }
}

template <int GROUP>
__global__ void rms_backward_dx(const float *x, const float *gamma, const float *dy, const float *rstd, float *dx,
                                int R, int C) {
    int r, lane;
    my_row<GROUP>(&r, &lane);
    if (r >= R) {
        return;
    }
    const float *xr = x + (size_t)r * C, *dyr = dy + (size_t)r * C;
    float rs = rstd[r];
    float sgx = 0.0f;
    for (int j = lane; j < C; j += GROUP) {
        sgx += dyr[j] * gamma[j] * (xr[j] * rs);
    }
    float mean_gx = group_sum<GROUP>(sgx) / C;
    float *dxr = dx + (size_t)r * C;
    for (int j = lane; j < C; j += GROUP) {
        dxr[j] += rs * (dyr[j] * gamma[j] - (xr[j] * rs) * mean_gx);
    }
}

// part p of the rows into parts[p][c]: the part's sum of dy * x_hat. 32 x 8 threads cover
// 32 columns, the warps every 8th row, their sums added in a fixed order
__global__ void rms_gamma_parts(const float *x, const float *dy, const float *rstd, float *parts, int R, int C,
                                int rows_per_part) {
    __shared__ float sg[8][32];
    int c = blockIdx.x * 32 + threadIdx.x;
    int r0 = blockIdx.y * rows_per_part, r1 = min(R, r0 + rows_per_part);
    float g = 0.0f;
    if (c < C) {
        for (int r = r0 + threadIdx.y; r < r1; r += 8) {
            g += dy[(size_t)r * C + c] * (x[(size_t)r * C + c] * rstd[r]);
        }
    }
    sg[threadIdx.y][threadIdx.x] = g;
    __syncthreads();
    if (threadIdx.y == 0 && c < C) {
        float t = 0.0f;
        for (int w = 0; w < 8; w++) {
            t += sg[w][threadIdx.x];
        }
        parts[(size_t)blockIdx.y * C + c] = t;
    }
}

__global__ void rms_gamma_add(const float *parts, int count, float *dgamma, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        float g = 0.0f;
        for (int p = 0; p < count; p++) {
            g += parts[(size_t)p * C + c];
        }
        dgamma[c] += g;
    }
}

// ---- launches ----

const int WARP_MAX_C = 1024;

void rms_norm_forward(const float *x, const float *gamma, float *y, float *rstd, int R, int C, float eps = 1e-5f) {
    if (C <= WARP_MAX_C) {
        rms_forward<32><<<(R + THREADS / 32 - 1) / (THREADS / 32), THREADS>>>(x, gamma, y, rstd, R, C, eps);
    } else {
        rms_forward<THREADS><<<R, THREADS>>>(x, gamma, y, rstd, R, C, eps);
    }
}

void rms_norm_dx(const float *x, const float *gamma, const float *dy, const float *rstd, float *dx, int R, int C) {
    if (C <= WARP_MAX_C) {
        rms_backward_dx<32><<<(R + THREADS / 32 - 1) / (THREADS / 32), THREADS>>>(x, gamma, dy, rstd, dx, R, C);
    } else {
        rms_backward_dx<THREADS><<<R, THREADS>>>(x, gamma, dy, rstd, dx, R, C);
    }
}

float *g_rms_parts = nullptr;
size_t g_rms_parts_size = 0;

int rms_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    }
    return sms;
}

// enough parts of the rows to give every SM about 8 blocks, each part at least 64 rows
void rms_norm_dgamma(const float *x, const float *dy, const float *rstd, float *dgamma, int R, int C) {
    int col_blocks = (C + 31) / 32;
    int want = std::max(1, 8 * rms_sm_count() / col_blocks);
    int count = std::max(1, std::min(want, R / 64));
    int rows_per_part = (R + count - 1) / count;
    count = (R + rows_per_part - 1) / rows_per_part;
    size_t need = (size_t)count * C;
    if (need > g_rms_parts_size) {
        if (g_rms_parts) {
            CUDA_CHECK(cudaFree(g_rms_parts));
        }
        CUDA_CHECK(cudaMalloc(&g_rms_parts, need * sizeof(float)));
        g_rms_parts_size = need;
    }
    rms_gamma_parts<<<dim3(col_blocks, count), dim3(32, 8)>>>(x, dy, rstd, g_rms_parts, R, C, rows_per_part);
    rms_gamma_add<<<(C + THREADS - 1) / THREADS, THREADS>>>(g_rms_parts, count, dgamma, C);
}

void rms_norm_backward(const float *x, const float *gamma, const float *dy, const float *rstd, float *dx,
                       float *dgamma, int R, int C) {
    rms_norm_dx(x, gamma, dy, rstd, dx, R, C);
    rms_norm_dgamma(x, dy, rstd, dgamma, R, C);
}

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

// ---- tests ----

const size_t CAP = (size_t)1 << 24, PAST = 4096;

struct Device {
    float *x, *gamma, *dy, *y, *dx, *dgamma, *rstd;
};

void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> download(const float *d, size_t n) {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

double worst(const std::vector<float> &got, const std::vector<double> &want, const std::vector<double> &scale) {
    double w = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        double e = fabs(got[i] - want[i]) / fmax(scale[i], 1e-30);
        w = fmax(w, std::isnan(e) ? INFINITY : e);
    }
    return w;
}

double g_worst[3];  // largest y, dx and dgamma errors over all shapes

// rows uniform in [-2, 2]; every 4th shifted by 1000, every 7th all zero. the gradients
// start at random values, so the test sees that they're added to. errors are relative to
// the size of what went into each value
bool check_shape(int R, int C, Device d, std::mt19937 &rng) {
    std::uniform_real_distribution<float> u(-2.0f, 2.0f), pos(0.5f, 1.5f), small(-1.0f, 1.0f);
    std::vector<float> x((size_t)R * C), gamma(C), dy((size_t)R * C), dx0((size_t)R * C), dg0(C);
    for (int r = 0; r < R; r++) {
        float shift = r % 4 == 3 ? 1000.0f : 0.0f;
        for (int j = 0; j < C; j++) {
            x[(size_t)r * C + j] = r % 7 == 6 ? 0.0f : u(rng) + shift;
        }
    }
    for (int j = 0; j < C; j++) {
        gamma[j] = pos(rng);
        dg0[j] = small(rng);
    }
    for (auto &v : dy) v = small(rng);
    for (auto &v : dx0) v = small(rng);

    const double eps = 1e-5;
    std::vector<double> y((size_t)R * C), y_s((size_t)R * C), dx((size_t)R * C), dx_s((size_t)R * C);
    std::vector<double> dg(dg0.begin(), dg0.end()), dg_s(C);
    for (int j = 0; j < C; j++) dg_s[j] = fabs(dg0[j]);
    for (int r = 0; r < R; r++) {
        const float *xr = &x[(size_t)r * C];
        double ms = 0.0;
        for (int j = 0; j < C; j++) ms += (double)xr[j] * xr[j];
        double rstd = 1.0 / sqrt(ms / C + eps);
        double mgx = 0.0, magx = 0.0;
        for (int j = 0; j < C; j++) {
            size_t i = (size_t)r * C + j;
            double xh = xr[j] * rstd, g = (double)dy[i] * gamma[j];
            y[i] = gamma[j] * xh;
            y_s[i] = fabs(y[i]) + 1e-30;
            mgx += g * xh;
            magx += fabs(g * xh);
            dg[j] += dy[i] * xh;
            dg_s[j] += fabs(dy[i] * xh);
        }
        mgx /= C;
        magx /= C;
        for (int j = 0; j < C; j++) {
            size_t i = (size_t)r * C + j;
            double xh = xr[j] * rstd, g = (double)dy[i] * gamma[j];
            dx[i] = dx0[i] + rstd * (g - xh * mgx);
            dx_s[i] = fabs(dx0[i]) + rstd * (fabs(g) + fabs(xh) * magx);
        }
    }

    upload(d.x, x);
    upload(d.gamma, gamma);
    upload(d.dy, dy);
    upload(d.dx, dx0);
    upload(d.dgamma, dg0);
    CUDA_CHECK(cudaMemset(d.y, 0xff, (x.size() + PAST) * sizeof(float)));
    rms_norm_forward(d.x, d.gamma, d.y, d.rstd, R, C);
    rms_norm_backward(d.x, d.gamma, d.dy, d.rstd, d.dx, d.dgamma, R, C);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> got_y = download(d.y, x.size() + PAST);
    double e[3] = {worst(got_y, y, y_s), worst(download(d.dx, x.size()), dx, dx_s),
                   worst(download(d.dgamma, C), dg, dg_s)};
    bool ok = true;
    for (size_t i = x.size(); i < got_y.size(); i++) {
        if (!std::isnan(got_y[i])) {
            printf("  %d x %d: wrote past the end of y\n", R, C);
            ok = false;
            break;
        }
    }
    const char *what[] = {"y", "dx", "dgamma"};
    double tol = 1e-6 * sqrt((double)std::max(R, C) + 16);
    for (int k = 0; k < 3; k++) {
        g_worst[k] = fmax(g_worst[k], e[k]);
        if (!(e[k] < tol)) {
            printf("  %d x %d: %s error %.2e (tol %.2e)\n", R, C, what[k], e[k], tol);
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
    int R, C;
    f >> R >> C;
    size_t sizes[] = {(size_t)R * C, (size_t)C, (size_t)R * C, (size_t)R * C, (size_t)R * C, (size_t)C};
    std::vector<double> arr[6];
    for (int k = 0; k < 6; k++) {
        arr[k].resize(sizes[k]);
        for (auto &v : arr[k]) f >> v;
    }
    if (!f) {
        printf("macrograd: couldn't read %s\n", path);
        return false;
    }
    auto as_float = [](const std::vector<double> &v) { return std::vector<float>(v.begin(), v.end()); };
    upload(d.x, as_float(arr[0]));
    upload(d.gamma, as_float(arr[1]));
    upload(d.dy, as_float(arr[2]));
    upload(d.dx, std::vector<float>(sizes[4], 0.0f));
    upload(d.dgamma, std::vector<float>(C, 0.0f));
    rms_norm_forward(d.x, d.gamma, d.y, d.rstd, R, C);
    rms_norm_backward(d.x, d.gamma, d.dy, d.rstd, d.dx, d.dgamma, R, C);
    CUDA_CHECK(cudaGetLastError());
    float *outs[] = {d.y, d.dx, d.dgamma};
    double w = 0.0, biggest = 0.0;
    for (int k = 0; k < 3; k++) {
        std::vector<float> got = download(outs[k], sizes[3 + k]);
        for (size_t i = 0; i < got.size(); i++) {
            double e = fabs(got[i] - arr[3 + k][i]);
            w = fmax(w, std::isnan(e) ? INFINITY : e);
            biggest = fmax(biggest, fabs(arr[3 + k][i]));
        }
    }
    bool good = w < 1e-6 * biggest;
    printf("macrograd (%d x %d): largest difference %.2e, largest value %.2f, %s\n", R, C, w, biggest,
           good ? "ok" : "WRONG");
    return good;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    Device d;
    float **bufs[] = {&d.x, &d.dy, &d.y, &d.dx};
    for (auto p : bufs) {
        CUDA_CHECK(cudaMalloc(p, CAP * sizeof(float)));
    }
    float **small[] = {&d.gamma, &d.dgamma, &d.rstd};
    for (auto p : small) {
        CUDA_CHECK(cudaMalloc(p, (1 << 20) * sizeof(float)));
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/rms_norm_macrograd.txt", d);
    bool shapes_ok = true;
    for (int R : {1, 3, 9, 70, 1000}) {
        for (int C : {1, 2, 31, 32, 33, 100, 1024, 1025, 3000}) {
            shapes_ok = check_shape(R, C, d, rng) && shapes_ok;
        }
    }
    shapes_ok = check_shape(20000, 16, d, rng) && shapes_ok;
    printf("shapes: %s, largest errors y %.2e, dx %.2e, dgamma %.2e\n", shapes_ok ? "ok" : "WRONG", g_worst[0],
           g_worst[1], g_worst[2]);
    ok = ok && shapes_ok;

    // forward 8 bytes a value, dx 16, dgamma 8, as for layer norm
    CUDA_CHECK(cudaMemset(d.x, 0, CAP * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.dy, 0, CAP * sizeof(float)));
    printf("\n%-16s %-20s %-20s %-20s\n", "rows x cols", "forward", "dx", "dgamma");
    int timed[][2] = {{65536, 256}, {16384, 1024}, {4096, 4096}, {1024, 16384}};
    for (auto &s : timed) {
        int R = s[0], C = s[1];
        double n = (double)R * C;
        float f = time_ms([&] { rms_norm_forward(d.x, d.gamma, d.y, d.rstd, R, C); });
        float b = time_ms([&] { rms_norm_dx(d.x, d.gamma, d.dy, d.rstd, d.dx, R, C); });
        float p = time_ms([&] { rms_norm_dgamma(d.x, d.dy, d.rstd, d.dgamma, R, C); });
        printf("%5d x %-8d %7.3f ms %5.1f GB/s  %7.3f ms %5.1f GB/s  %7.3f ms %5.1f GB/s\n", R, C, f,
               8 * n / (f * 1e-3) / 1e9, b, 16 * n / (b * 1e-3) / 1e9, p, 8 * n / (p * 1e-3) / 1e9);
    }

    for (auto p : bufs) CUDA_CHECK(cudaFree(*p));
    for (auto p : small) CUDA_CHECK(cudaFree(*p));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
