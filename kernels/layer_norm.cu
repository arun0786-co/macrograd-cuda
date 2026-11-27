// layer norm over the last dim of x (R, C), stored row by row, with gamma and beta (C,),
// the same as macrograd's nn.layer_norm:
//
//    mean = sum_j x_j / C,  var = sum_j (x_j - mean)^2 / C,  rstd = 1 / sqrt(var + eps)
//    x_hat = (x - mean) * rstd,   y = gamma * x_hat + beta
//
// backward, with dy the gradient at y and g = dy * gamma the gradient at x_hat:
//
//    dx_j += rstd * (g_j - mean(g) - x_hat_j * mean(g * x_hat))
//    dgamma += sum over rows of dy * x_hat,   dbeta += sum over rows of dy
//
// moving x_j changes x_hat_j directly, but also the row's mean (which moves every x_hat)
// and its variance (which rescales every x_hat); the two means are those two effects.
//
// forward and dx go one row per warp up to 1024 columns, and one row per block of 256
// above that, like cross_entropy.cu. the forward saves mean and rstd for each row, so
// the backward doesn't work them out again.
//
// the mean is a sum of C floats, and a float sum of values near 1000 loses about 1000 *
// 6e-8 per add, which next to a spread of 1 is a big error. so each row first has its
// first value x_0 subtracted, which is exact for nearby floats: the kernels work with
// m = mean(x - x_0) and x - mean = (x - x_0) - m, and never hold the mean itself, which as
// a float near 1000 would be off by up to 3e-5. the saved "mean" is that m. the variance
// is a second pass over ((x - x_0) - m)^2, which doesn't lose anything to cancellation
// the way mean(x^2) - mean^2 would.
//
// dgamma and dbeta are column sums over all the rows. a block covers 32 columns and one
// part of the rows, writes its sums, and a second kernel adds the parts in a fixed order,
// so a tall matrix still has enough blocks and the result is the same on every run.
//
// checked against a CPU version in double and against macrograd on the same numbers
// (tests/layer_norm_macrograd.txt, from tests/make_layer_norm_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/layer_norm.cu -o build/layer_norm && ./build/layer_norm

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include "common.h"

const int THREADS = 256;

// the sum over a group of GROUP threads, which every thread of the group gets: a warp
// (GROUP = 32) or the whole block (GROUP = blockDim.x = 256)
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

// which row this thread works on, and its place in the row's group
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
__global__ void ln_forward(const float *x, const float *gamma, const float *beta, float *y, float *mean_out,
                           float *rstd_out, int R, int C, float eps) {
    int r, lane;
    my_row<GROUP>(&r, &lane);
    // a whole warp has the same r, so it leaves together and the shuffles see all 32 lanes
    if (r >= R) {
        return;
    }
    const float *xr = x + (size_t)r * C;
    float shift = xr[0];
    float s = 0.0f;
    for (int j = lane; j < C; j += GROUP) {
        s += xr[j] - shift;
    }
    float mean = group_sum<GROUP>(s) / C;  // of x - x_0
    float ss = 0.0f;
    for (int j = lane; j < C; j += GROUP) {
        float d = (xr[j] - shift) - mean;
        ss += d * d;
    }
    float rstd = 1.0f / sqrtf(group_sum<GROUP>(ss) / C + eps);
    float *yr = y + (size_t)r * C;
    for (int j = lane; j < C; j += GROUP) {
        yr[j] = gamma[j] * (((xr[j] - shift) - mean) * rstd) + beta[j];
    }
    if (lane == 0) {
        mean_out[r] = mean;
        rstd_out[r] = rstd;
    }
}

template <int GROUP>
__global__ void ln_backward_dx(const float *x, const float *gamma, const float *dy, const float *mean,
                               const float *rstd, float *dx, int R, int C) {
    int r, lane;
    my_row<GROUP>(&r, &lane);
    if (r >= R) {
        return;
    }
    const float *xr = x + (size_t)r * C, *dyr = dy + (size_t)r * C;
    float shift = xr[0], m = mean[r], rs = rstd[r];
    float sg = 0.0f, sgx = 0.0f;
    for (int j = lane; j < C; j += GROUP) {
        float g = dyr[j] * gamma[j];
        sg += g;
        sgx += g * (((xr[j] - shift) - m) * rs);
    }
    float mean_g = group_sum<GROUP>(sg) / C;
    float mean_gx = group_sum<GROUP>(sgx) / C;
    float *dxr = dx + (size_t)r * C;
    for (int j = lane; j < C; j += GROUP) {
        float g = dyr[j] * gamma[j];
        float x_hat = ((xr[j] - shift) - m) * rs;
        dxr[j] += rs * (g - mean_g - x_hat * mean_gx);
    }
}

// part p of the rows: rows p * rows_per_part to the next part. 32 x 8 threads cover 32
// columns: the warps take every 8th row and their sums are added in a fixed order.
// parts[p][0][c] gets the part's sum of dy * x_hat and parts[p][1][c] its sum of dy
__global__ void ln_param_parts(const float *x, const float *dy, const float *mean, const float *rstd, float *parts,
                               int R, int C, int rows_per_part) {
    __shared__ float sg[8][32], sb[8][32];
    int c = blockIdx.x * 32 + threadIdx.x;
    int r0 = blockIdx.y * rows_per_part, r1 = min(R, r0 + rows_per_part);
    float g = 0.0f, b = 0.0f;
    if (c < C) {
        for (int r = r0 + threadIdx.y; r < r1; r += 8) {
            float d = dy[(size_t)r * C + c];
            g += d * (((x[(size_t)r * C + c] - x[(size_t)r * C]) - mean[r]) * rstd[r]);
            b += d;
        }
    }
    sg[threadIdx.y][threadIdx.x] = g;
    sb[threadIdx.y][threadIdx.x] = b;
    __syncthreads();
    if (threadIdx.y == 0 && c < C) {
        float tg = 0.0f, tb = 0.0f;
        for (int w = 0; w < 8; w++) {
            tg += sg[w][threadIdx.x];
            tb += sb[w][threadIdx.x];
        }
        parts[(size_t)blockIdx.y * 2 * C + c] = tg;
        parts[(size_t)blockIdx.y * 2 * C + C + c] = tb;
    }
}

__global__ void ln_param_add(const float *parts, int count, float *dgamma, float *dbeta, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        float g = 0.0f, b = 0.0f;
        for (int p = 0; p < count; p++) {
            g += parts[(size_t)p * 2 * C + c];
            b += parts[(size_t)p * 2 * C + C + c];
        }
        dgamma[c] += g;
        dbeta[c] += b;
    }
}

// ---- launches ----

const int WARP_MAX_C = 1024;

// mean and rstd hold R floats each, for the backward. mean is the mean of x - x_0, the
// row's first value, not of x
void layer_norm_forward(const float *x, const float *gamma, const float *beta, float *y, float *mean, float *rstd,
                        int R, int C, float eps = 1e-5f) {
    if (C <= WARP_MAX_C) {
        ln_forward<32><<<(R + THREADS / 32 - 1) / (THREADS / 32), THREADS>>>(x, gamma, beta, y, mean, rstd, R, C, eps);
    } else {
        ln_forward<THREADS><<<R, THREADS>>>(x, gamma, beta, y, mean, rstd, R, C, eps);
    }
}

void layer_norm_dx(const float *x, const float *gamma, const float *dy, const float *mean, const float *rstd,
                   float *dx, int R, int C) {
    if (C <= WARP_MAX_C) {
        ln_backward_dx<32><<<(R + THREADS / 32 - 1) / (THREADS / 32), THREADS>>>(x, gamma, dy, mean, rstd, dx, R, C);
    } else {
        ln_backward_dx<THREADS><<<R, THREADS>>>(x, gamma, dy, mean, rstd, dx, R, C);
    }
}

// scratch for the parts of dgamma and dbeta, grown when needed
float *g_ln_parts = nullptr;
size_t g_ln_parts_size = 0;

int ln_sm_count() {
    static int sms = 0;
    if (sms == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    }
    return sms;
}

// enough parts of the rows to give every SM about 8 blocks, each part at least 64 rows
void layer_norm_param_grads(const float *x, const float *dy, const float *mean, const float *rstd, float *dgamma,
                            float *dbeta, int R, int C) {
    int col_blocks = (C + 31) / 32;
    int want = std::max(1, 8 * ln_sm_count() / col_blocks);
    int count = std::max(1, std::min(want, R / 64));
    int rows_per_part = (R + count - 1) / count;
    count = (R + rows_per_part - 1) / rows_per_part;
    size_t need = (size_t)count * 2 * C;
    if (need > g_ln_parts_size) {
        if (g_ln_parts) {
            CUDA_CHECK(cudaFree(g_ln_parts));
        }
        CUDA_CHECK(cudaMalloc(&g_ln_parts, need * sizeof(float)));
        g_ln_parts_size = need;
    }
    ln_param_parts<<<dim3(col_blocks, count), dim3(32, 8)>>>(x, dy, mean, rstd, g_ln_parts, R, C, rows_per_part);
    ln_param_add<<<(C + THREADS - 1) / THREADS, THREADS>>>(g_ln_parts, count, dgamma, dbeta, C);
}

void layer_norm_backward(const float *x, const float *gamma, const float *dy, const float *mean, const float *rstd,
                         float *dx, float *dgamma, float *dbeta, int R, int C) {
    layer_norm_dx(x, gamma, dy, mean, rstd, dx, R, C);
    layer_norm_param_grads(x, dy, mean, rstd, dgamma, dbeta, R, C);
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

const size_t CAP = (size_t)1 << 24;

struct Device {
    float *x, *gamma, *beta, *dy, *y, *dx, *dgamma, *dbeta, *mean, *rstd;
};

void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> download(const float *d, size_t n) {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

// largest |got - want| / scale; nan is infinitely wrong
double worst(const std::vector<float> &got, const std::vector<double> &want, const std::vector<double> &scale) {
    double w = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        double e = fabs(got[i] - want[i]) / fmax(scale[i], 1e-30);
        w = fmax(w, std::isnan(e) ? INFINITY : e);
    }
    return w;
}

double g_worst[4];  // largest y, dx, dgamma and dbeta errors over all shapes

// rows uniform in [-2, 2]; every 4th shifted by 1000, every 7th constant (variance 0).
// the gradients start at random values, so the test sees that they're added to.
// errors are relative to the size of what went into each value
bool check_shape(int R, int C, Device d, std::mt19937 &rng) {
    std::uniform_real_distribution<float> u(-2.0f, 2.0f), pos(0.5f, 1.5f), small(-1.0f, 1.0f);
    std::vector<float> x((size_t)R * C), gamma(C), beta(C), dy((size_t)R * C), dx0((size_t)R * C), dg0(C), db0(C);
    for (int r = 0; r < R; r++) {
        float shift = r % 4 == 3 ? 1000.0f : 0.0f;
        float constant = u(rng);
        for (int j = 0; j < C; j++) {
            x[(size_t)r * C + j] = r % 7 == 6 ? constant : u(rng) + shift;
        }
    }
    for (int j = 0; j < C; j++) {
        gamma[j] = pos(rng);
        beta[j] = small(rng);
        dg0[j] = small(rng);
        db0[j] = small(rng);
    }
    for (auto &v : dy) v = small(rng);
    for (auto &v : dx0) v = small(rng);

    const double eps = 1e-5;
    std::vector<double> y((size_t)R * C), y_s((size_t)R * C), dx((size_t)R * C), dx_s((size_t)R * C);
    std::vector<double> dg(dg0.begin(), dg0.end()), db(db0.begin(), db0.end()), dg_s(C), db_s(C);
    for (int j = 0; j < C; j++) {
        dg_s[j] = fabs(dg0[j]);
        db_s[j] = fabs(db0[j]);
    }
    for (int r = 0; r < R; r++) {
        const float *xr = &x[(size_t)r * C];
        double mean = 0.0, var = 0.0;
        for (int j = 0; j < C; j++) mean += xr[j];
        mean /= C;
        for (int j = 0; j < C; j++) var += (xr[j] - mean) * (xr[j] - mean);
        var /= C;
        double rstd = 1.0 / sqrt(var + eps);
        double mg = 0.0, mgx = 0.0, mag = 0.0, magx = 0.0;
        for (int j = 0; j < C; j++) {
            size_t i = (size_t)r * C + j;
            double xh = (xr[j] - mean) * rstd, g = (double)dy[i] * gamma[j];
            y[i] = gamma[j] * xh + beta[j];
            // the size of what goes into x_hat, after x_0 is taken off
            double xh_size = (fabs(xr[j] - xr[0]) + fabs(mean - xr[0])) * rstd;
            y_s[i] = gamma[j] * xh_size + fabs(beta[j]);
            mg += g;
            mgx += g * xh;
            mag += fabs(g);
            magx += fabs(g) * xh_size;
            dg[j] += dy[i] * xh;
            dg_s[j] += fabs(dy[i]) * xh_size;
            db[j] += dy[i];
            db_s[j] += fabs(dy[i]);
        }
        mg /= C;
        mgx /= C;
        mag /= C;
        magx /= C;
        for (int j = 0; j < C; j++) {
            size_t i = (size_t)r * C + j;
            double xh = (xr[j] - mean) * rstd, g = (double)dy[i] * gamma[j];
            double xh_size = (fabs(xr[j] - xr[0]) + fabs(mean - xr[0])) * rstd;
            dx[i] = dx0[i] + rstd * (g - mg - xh * mgx);
            dx_s[i] = fabs(dx0[i]) + rstd * (fabs(g) + mag + xh_size * magx);
        }
    }

    upload(d.x, x);
    upload(d.gamma, gamma);
    upload(d.beta, beta);
    upload(d.dy, dy);
    upload(d.dx, dx0);
    upload(d.dgamma, dg0);
    upload(d.dbeta, db0);
    // nan, here and for PAST floats after the end, so a missed write or a stray one shows
    const size_t PAST = 4096;
    CUDA_CHECK(cudaMemset(d.y, 0xff, (x.size() + PAST) * sizeof(float)));
    layer_norm_forward(d.x, d.gamma, d.beta, d.y, d.mean, d.rstd, R, C);
    layer_norm_backward(d.x, d.gamma, d.dy, d.mean, d.rstd, d.dx, d.dgamma, d.dbeta, R, C);
    CUDA_CHECK(cudaGetLastError());
    double e[4] = {worst(download(d.y, x.size()), y, y_s), worst(download(d.dx, x.size()), dx, dx_s),
                   worst(download(d.dgamma, C), dg, dg_s), worst(download(d.dbeta, C), db, db_s)};
    bool ok = true;
    std::vector<float> after = download(d.y, x.size() + PAST);
    for (size_t i = x.size(); i < after.size(); i++) {
        if (!std::isnan(after[i])) {
            printf("  %d x %d: wrote past the end of y\n", R, C);
            ok = false;
            break;
        }
    }
    const char *what[] = {"y", "dx", "dgamma", "dbeta"};
    // a few float roundings per term, and sums of up to max(R, C) terms
    double tol = 1e-6 * sqrt((double)std::max(R, C) + 16);
    for (int k = 0; k < 4; k++) {
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
    size_t sizes[] = {(size_t)R * C, (size_t)C, (size_t)C, (size_t)R * C, (size_t)R * C, (size_t)R * C, (size_t)C, (size_t)C};
    std::vector<double> arr[8];
    for (int k = 0; k < 8; k++) {
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
    upload(d.beta, as_float(arr[2]));
    upload(d.dy, as_float(arr[3]));
    upload(d.dx, std::vector<float>(sizes[5], 0.0f));
    upload(d.dgamma, std::vector<float>(C, 0.0f));
    upload(d.dbeta, std::vector<float>(C, 0.0f));
    layer_norm_forward(d.x, d.gamma, d.beta, d.y, d.mean, d.rstd, R, C);
    layer_norm_backward(d.x, d.gamma, d.dy, d.mean, d.rstd, d.dx, d.dgamma, d.dbeta, R, C);
    CUDA_CHECK(cudaGetLastError());
    float *outs[] = {d.y, d.dx, d.dgamma, d.dbeta};
    double w = 0.0, biggest = 0.0;
    for (int k = 0; k < 4; k++) {
        std::vector<float> got = download(outs[k], sizes[4 + k]);
        for (size_t i = 0; i < got.size(); i++) {
            double e = fabs(got[i] - arr[4 + k][i]);
            w = fmax(w, std::isnan(e) ? INFINITY : e);
            biggest = fmax(biggest, fabs(arr[4 + k][i]));
        }
    }
    // the constant row's rstd is 1 / sqrt(1e-5) = 316, so its dx is in the hundreds and a
    // float rounding of it is about 2e-5; compare against the largest value
    bool good = w < 1e-6 * biggest;
    printf("macrograd (%d x %d): largest difference %.2e, largest value %.1f, %s\n", R, C, w, biggest,
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
    float **small[] = {&d.gamma, &d.beta, &d.dgamma, &d.dbeta, &d.mean, &d.rstd};
    for (auto p : small) {
        CUDA_CHECK(cudaMalloc(p, (1 << 20) * sizeof(float)));
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/layer_norm_macrograd.txt", d);
    // one column, around a warp, the 1024 switch and a block, tall and wide
    bool shapes_ok = true;
    for (int R : {1, 3, 9, 70, 1000}) {
        for (int C : {1, 2, 31, 32, 33, 100, 1024, 1025, 3000}) {
            shapes_ok = check_shape(R, C, d, rng) && shapes_ok;
        }
    }
    shapes_ok = check_shape(20000, 16, d, rng) && shapes_ok;
    printf("shapes: %s, largest errors y %.2e, dx %.2e, dgamma %.2e, dbeta %.2e\n", shapes_ok ? "ok" : "WRONG",
           g_worst[0], g_worst[1], g_worst[2], g_worst[3]);
    ok = ok && shapes_ok;

    // forward reads x and writes y, 8 bytes a value; dx reads x, dy and dx and writes dx,
    // 16; the parameter gradients read x and dy, 8. gamma, beta, mean and rstd aren't counted
    CUDA_CHECK(cudaMemset(d.x, 0, CAP * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.dy, 0, CAP * sizeof(float)));
    printf("\n%-16s %-20s %-20s %-20s\n", "rows x cols", "forward", "dx", "dgamma, dbeta");
    int timed[][2] = {{65536, 256}, {16384, 1024}, {4096, 4096}, {1024, 16384}};
    for (auto &s : timed) {
        int R = s[0], C = s[1];
        double n = (double)R * C;
        float f = time_ms([&] { layer_norm_forward(d.x, d.gamma, d.beta, d.y, d.mean, d.rstd, R, C); });
        float b = time_ms([&] { layer_norm_dx(d.x, d.gamma, d.dy, d.mean, d.rstd, d.dx, R, C); });
        float p = time_ms([&] { layer_norm_param_grads(d.x, d.dy, d.mean, d.rstd, d.dgamma, d.dbeta, R, C); });
        printf("%5d x %-8d %7.3f ms %5.1f GB/s  %7.3f ms %5.1f GB/s  %7.3f ms %5.1f GB/s\n", R, C, f,
               8 * n / (f * 1e-3) / 1e9, b, 16 * n / (b * 1e-3) / 1e9, p, 8 * n / (p * 1e-3) / 1e9);
    }
    size_t n = CAP;
    float ms = time_ms([&] { CUDA_CHECK(cudaMemcpy(d.y, d.x, n * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("memcpy %.3f ms, %.1f GB/s\n", ms, 8.0 * n / (ms * 1e-3) / 1e9);

    for (auto p : bufs) CUDA_CHECK(cudaFree(*p));
    for (auto p : small) CUDA_CHECK(cudaFree(*p));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
