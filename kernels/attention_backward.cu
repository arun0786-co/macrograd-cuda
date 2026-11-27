// the backward of attention, out = P @ v with P = softmax(q @ k^T * s + mask) and
// s = 1 / sqrt(d), for the same shapes and causal mask as attention.cu. with dO the
// gradient at out:
//
//    dv = P^T @ dO                       each value row is used with weight P[i][j]
//    dP = dO @ v^T
//    dS = P * (dP - rowsum(dP * P))      softmax's backward, row by row
//    dq = dS @ k * s,   dk = dS^T @ q * s
//
// masked scores have P = 0, so they get dS = 0 and send nothing back. this is the naive
// version: it makes P again with the forward from attention.cu (one score matrix of
// BH * T_q * T_k floats), dP is a second one, and the four products are the linear
// layer's matmul, which reads the transposes in place. dq, dk and dv are added into, like
// the other backward kernels.
//
// checked three ways: the formula against finite differences of the forward in double,
// the kernels against a CPU version in double, and both against macrograd's backward on
// the same numbers (tests/attention_backward_macrograd.txt, from
// tests/make_attention_backward_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 kernels/attention_backward.cu -lcublas -o build/attention_backward && ./build/attention_backward

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include "common.h"

#ifndef MACROGRAD_NO_MAIN
#define MACROGRAD_NO_MAIN
#define BACKWARD_OWNS_MAIN
#endif
namespace naive {
#include "attention.cu"
}
#ifdef BACKWARD_OWNS_MAIN
#undef MACROGRAD_NO_MAIN
#endif

namespace lin = naive::lin;
const int THREADS = 256;

// dS = P * (dP - rowsum(dP * P)) * s, in place in dP, one block per row
__global__ void softmax_backward_rows(const float *P, float *dP, int T_k, float s) {
    const float *p = P + (size_t)blockIdx.x * T_k;
    float *dp = dP + (size_t)blockIdx.x * T_k;
    float dot = 0.0f;
    for (int j = threadIdx.x; j < T_k; j += THREADS) dot += dp[j] * p[j];
    dot = naive::block_sum(dot);
    for (int j = threadIdx.x; j < T_k; j += THREADS) dp[j] = p[j] * (dp[j] - dot) * s;
}

// P and dP hold BH * T_q * T_k floats each; out holds BH * T_q * dv (the forward's output,
// which the backward doesn't need but the forward writes)
void attention_backward(const float *q, const float *k, const float *v, const float *dO, float *dq, float *dk,
                        float *dv, float *P, float *dP, float *out, int BH, int T_q, int T_k, int d, int dvs,
                        bool causal) {
    naive::attention_naive(q, k, v, P, out, BH, T_q, T_k, d, dvs, causal);  // leaves P in P
    float s = 1.0f / sqrtf((float)d);
    for (int h = 0; h < BH; h++) {
        const float *Ph = P + (size_t)h * T_q * T_k, *dOh = dO + (size_t)h * T_q * dvs;
        // dv_h (T_k, dv) += P_h^T (read from P_h, T_q x T_k) @ dO_h (T_q, dv)
        lin::matmul<true, false>(Ph, dOh, dv + (size_t)h * T_k * dvs, nullptr, true, T_k, dvs, T_q);
        // dP_h (T_q, T_k) = dO_h (T_q, dv) @ v_h^T, v_h stored (T_k, dv)
        lin::matmul<false, true>(dOh, v + (size_t)h * T_k * dvs, dP + (size_t)h * T_q * T_k, nullptr, false, T_q,
                                 T_k, dvs);
    }
    softmax_backward_rows<<<BH * T_q, THREADS>>>(P, dP, T_k, s);
    for (int h = 0; h < BH; h++) {
        const float *dSh = dP + (size_t)h * T_q * T_k;
        // dq_h (T_q, d) += dS_h (T_q, T_k) @ k_h (T_k, d)
        lin::matmul<false, false>(dSh, k + (size_t)h * T_k * d, dq + (size_t)h * T_q * d, nullptr, true, T_q, d, T_k);
        // dk_h (T_k, d) += dS_h^T @ q_h (T_q, d)
        lin::matmul<true, false>(dSh, q + (size_t)h * T_q * d, dk + (size_t)h * T_k * d, nullptr, true, T_k, d, T_q);
    }
}

// ---- CPU side ----

// the forward in double, for finite differences: sum(out * dO)
double cpu_loss(const std::vector<double> &q, const std::vector<double> &k, const std::vector<double> &v,
                const std::vector<double> &dO, int BH, int T_q, int T_k, int d, int dvs, bool causal) {
    double loss = 0.0;
    std::vector<double> p(T_k);
    for (int h = 0; h < BH; h++)
        for (int i = 0; i < T_q; i++) {
            int last = causal ? T_k - T_q + i : T_k - 1;
            double m = -INFINITY, tot = 0.0;
            for (int j = 0; j <= last; j++) {
                double sc = 0.0;
                for (int c = 0; c < d; c++) sc += q[((size_t)h * T_q + i) * d + c] * k[((size_t)h * T_k + j) * d + c];
                p[j] = sc / sqrt((double)d);
                m = fmax(m, p[j]);
            }
            for (int j = 0; j <= last; j++) tot += (p[j] = exp(p[j] - m));
            for (int j = 0; j <= last; j++)
                for (int c = 0; c < dvs; c++)
                    loss += p[j] / tot * v[((size_t)h * T_k + j) * dvs + c] * dO[((size_t)h * T_q + i) * dvs + c];
        }
    return loss;
}

// the formulas in double: dq, dk, dv, and for the error the size of what went into each
void cpu_backward(const std::vector<double> &q, const std::vector<double> &k, const std::vector<double> &v,
                  const std::vector<double> &dO, int BH, int T_q, int T_k, int d, int dvs, bool causal,
                  std::vector<double> &dq, std::vector<double> &dk, std::vector<double> &dv,
                  std::vector<double> &sq, std::vector<double> &sk, std::vector<double> &sv) {
    dq.assign(q.size(), 0.0), dk.assign(k.size(), 0.0), dv.assign(v.size(), 0.0);
    sq.assign(q.size(), 0.0), sk.assign(k.size(), 0.0), sv.assign(v.size(), 0.0);
    double s = 1.0 / sqrt((double)d);
    std::vector<double> p(T_k), dp(T_k);
    for (int h = 0; h < BH; h++)
        for (int i = 0; i < T_q; i++) {
            int last = causal ? T_k - T_q + i : T_k - 1;
            const double *qi = &q[((size_t)h * T_q + i) * d], *dOi = &dO[((size_t)h * T_q + i) * dvs];
            double m = -INFINITY, tot = 0.0;
            for (int j = 0; j <= last; j++) {
                double sc = 0.0;
                for (int c = 0; c < d; c++) sc += qi[c] * k[((size_t)h * T_k + j) * d + c];
                p[j] = sc * s;
                m = fmax(m, p[j]);
            }
            for (int j = 0; j <= last; j++) tot += (p[j] = exp(p[j] - m));
            double dot = 0.0, adot = 0.0;
            for (int j = 0; j <= last; j++) {
                p[j] /= tot;
                dp[j] = 0.0;
                for (int c = 0; c < dvs; c++) dp[j] += dOi[c] * v[((size_t)h * T_k + j) * dvs + c];
                dot += dp[j] * p[j];
                adot += fabs(dp[j]) * p[j];
                for (int c = 0; c < dvs; c++) {
                    dv[((size_t)h * T_k + j) * dvs + c] += p[j] * dOi[c];
                    sv[((size_t)h * T_k + j) * dvs + c] += p[j] * fabs(dOi[c]);
                }
            }
            for (int j = 0; j <= last; j++) {
                double ds = p[j] * (dp[j] - dot) * s, ads = p[j] * (fabs(dp[j]) + adot) * s;
                for (int c = 0; c < d; c++) {
                    dq[((size_t)h * T_q + i) * d + c] += ds * k[((size_t)h * T_k + j) * d + c];
                    sq[((size_t)h * T_q + i) * d + c] += ads * fabs(k[((size_t)h * T_k + j) * d + c]);
                    dk[((size_t)h * T_k + j) * d + c] += ds * qi[c];
                    sk[((size_t)h * T_k + j) * d + c] += ads * fabs(qi[c]);
                }
            }
        }
}

// the backward formulas against finite differences of the loss, all in double
double check_formulas(std::mt19937 &rng) {
    const int BH = 2, T_q = 3, T_k = 5, d = 4, dvs = 3;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> q(BH * T_q * d), k(BH * T_k * d), v(BH * T_k * dvs), dO(BH * T_q * dvs);
    for (auto *vec : {&q, &k, &v, &dO}) for (auto &x : *vec) x = u(rng);
    double worst = 0.0;
    for (bool causal : {false, true}) {
        std::vector<double> dq, dk, dv, sq, sk, sv;
        cpu_backward(q, k, v, dO, BH, T_q, T_k, d, dvs, causal, dq, dk, dv, sq, sk, sv);
        std::vector<double> *inputs[] = {&q, &k, &v}, *grads[] = {&dq, &dk, &dv};
        for (int t = 0; t < 3; t++) {
            for (size_t i = 0; i < inputs[t]->size(); i++) {
                double old = (*inputs[t])[i], h = 1e-6;
                (*inputs[t])[i] = old + h;
                double up = cpu_loss(q, k, v, dO, BH, T_q, T_k, d, dvs, causal);
                (*inputs[t])[i] = old - h;
                double down = cpu_loss(q, k, v, dO, BH, T_q, T_k, d, dvs, causal);
                (*inputs[t])[i] = old;
                worst = fmax(worst, fabs((up - down) / (2 * h) - (*grads[t])[i]));
            }
        }
    }
    return worst;
}

std::vector<float> to_float(const std::vector<double> &v) { return std::vector<float>(v.begin(), v.end()); }

struct Buffers {
    float *q, *k, *v, *dO, *dq, *dk, *dv, *P, *dP, *out;
};

void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> download(const float *d, size_t n) {
    std::vector<float> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

// the gradients start at random values, so the test sees they're added to
bool check_shape(int BH, int T_q, int T_k, int d, int dvs, bool causal, Buffers b, std::mt19937 &rng, double *worst) {
    std::uniform_real_distribution<double> u(-1.0, 1.0), wide(-2.0, 2.0);
    std::vector<double> q((size_t)BH * T_q * d), k((size_t)BH * T_k * d), v((size_t)BH * T_k * dvs),
        dO((size_t)BH * T_q * dvs);
    for (auto &x : q) x = (float)wide(rng);
    for (auto &x : k) x = (float)wide(rng);
    for (auto &x : v) x = (float)u(rng);
    for (auto &x : dO) x = (float)u(rng);
    std::vector<double> dq, dk, dv, sq, sk, sv;
    cpu_backward(q, k, v, dO, BH, T_q, T_k, d, dvs, causal, dq, dk, dv, sq, sk, sv);
    std::vector<float> start[3];
    std::vector<double> *grads[] = {&dq, &dk, &dv}, *sizes[] = {&sq, &sk, &sv};
    for (int t = 0; t < 3; t++) {
        start[t].resize(grads[t]->size());
        for (size_t i = 0; i < start[t].size(); i++) {
            start[t][i] = (float)u(rng);
            (*grads[t])[i] += start[t][i];
            (*sizes[t])[i] += fabs(start[t][i]);
        }
    }
    upload(b.q, to_float(q)), upload(b.k, to_float(k)), upload(b.v, to_float(v)), upload(b.dO, to_float(dO));
    upload(b.dq, start[0]), upload(b.dk, start[1]), upload(b.dv, start[2]);
    attention_backward(b.q, b.k, b.v, b.dO, b.dq, b.dk, b.dv, b.P, b.dP, b.out, BH, T_q, T_k, d, dvs, causal);
    CUDA_CHECK(cudaGetLastError());
    float *outs[] = {b.dq, b.dk, b.dv};
    const char *names[] = {"dq", "dk", "dv"};
    bool ok = true;
    // scores a few float roundings off, which exp passes on to P; sums of up to T_k terms
    double tol = 2e-6 * sqrt((double)d) + 1e-6 * sqrt((double)std::max(T_q, T_k));
    for (int t = 0; t < 3; t++) {
        std::vector<float> got = download(outs[t], grads[t]->size());
        double e = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            double x = fabs(got[i] - (*grads[t])[i]) / fmax((*sizes[t])[i], 1e-30);
            e = fmax(e, std::isnan(x) ? INFINITY : x);
        }
        *worst = fmax(*worst, e);
        if (!(e < tol)) {
            printf("  %d heads, %d x %d, d %d dv %d%s, %s: error %.2e (tol %.2e)\n", BH, T_q, T_k, d, dvs,
                   causal ? ", causal" : "", names[t], e, tol);
            ok = false;
        }
    }
    return ok;
}

bool check_macrograd(const char *path, Buffers b) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    int BH, T_q, T_k, d, dvs;
    f >> BH >> T_q >> T_k >> d >> dvs;
    size_t sizes[] = {(size_t)BH * T_q * d, (size_t)BH * T_k * d, (size_t)BH * T_k * dvs, (size_t)BH * T_q * dvs,
                      (size_t)BH * T_q * d, (size_t)BH * T_k * d, (size_t)BH * T_k * dvs};
    std::vector<double> arr[7];
    for (int i = 0; i < 7; i++) {
        arr[i].resize(sizes[i]);
        for (auto &x : arr[i]) f >> x;
    }
    if (!f) {
        printf("macrograd: couldn't read %s\n", path);
        return false;
    }
    upload(b.q, to_float(arr[0])), upload(b.k, to_float(arr[1])), upload(b.v, to_float(arr[2]));
    upload(b.dO, to_float(arr[3]));
    float *outs[] = {b.dq, b.dk, b.dv};
    for (int t = 0; t < 3; t++) upload(outs[t], std::vector<float>(sizes[4 + t], 0.0f));
    attention_backward(b.q, b.k, b.v, b.dO, b.dq, b.dk, b.dv, b.P, b.dP, b.out, BH, T_q, T_k, d, dvs, true);
    double w = 0.0;
    for (int t = 0; t < 3; t++) {
        std::vector<float> got = download(outs[t], sizes[4 + t]);
        for (size_t i = 0; i < got.size(); i++) w = fmax(w, std::isnan(got[i]) ? INFINITY : fabs(got[i] - arr[4 + t][i]));
    }
    bool good = w < 1e-6;
    printf("macrograd (%d heads, %d queries, %d keys, causal): largest difference %.2e, %s\n", BH, T_q, T_k, w,
           good ? "ok" : "WRONG");
    return good;
}

#ifndef MACROGRAD_NO_MAIN
int main() {
    Buffers b;
    const size_t big = (size_t)1 << 24, scores = (size_t)16 * 2048 * 2048;
    float **bufs[] = {&b.q, &b.k, &b.v, &b.dO, &b.dq, &b.dk, &b.dv, &b.out};
    for (auto p : bufs) CUDA_CHECK(cudaMalloc(p, big * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.P, scores * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.dP, scores * sizeof(float)));
    CUBLAS_CHECK(cublasCreate(&lin::g_cublas));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    double fd = check_formulas(rng);
    bool ok = fd < 1e-8;
    printf("backward formulas vs finite differences: largest difference %.1e, %s\n", fd, ok ? "ok" : "WRONG");
    ok = check_macrograd("tests/attention_backward_macrograd.txt", b) && ok;

    double worst = 0.0;
    bool shapes_ok = true;
    int shapes[][5] = {{1, 1, 1, 1, 1}, {1, 1, 9, 4, 3}, {3, 7, 7, 8, 8}, {2, 5, 300, 16, 5}, {4, 64, 64, 64, 64},
                       {2, 129, 257, 32, 48}, {1, 300, 1000, 128, 64}};
    for (auto &s : shapes)
        for (bool causal : {false, true})
            shapes_ok = check_shape(s[0], s[1], s[2], s[3], s[4], causal, b, rng, &worst) && shapes_ok;
    printf("shapes: %s, largest error %.2e\n", shapes_ok ? "ok" : "WRONG", worst);
    ok = ok && shapes_ok;

    // 16 heads of 64, causal: the backward (which runs the forward again) against the forward
    printf("\n%-8s %-14s %-14s\n", "T", "forward ms", "backward ms");
    const int BH = 16, d = 64;
    for (int T : {256, 1024, 2048}) {
        float f_ = naive::time_ms([&] { naive::attention_naive(b.q, b.k, b.v, b.P, b.out, BH, T, T, d, d, true); });
        float bw = naive::time_ms([&] {
            attention_backward(b.q, b.k, b.v, b.dO, b.dq, b.dk, b.dv, b.P, b.dP, b.out, BH, T, T, d, d, true);
        });
        printf("%-8d %-14.3f %-14.3f\n", T, f_, bw);
    }

    CUBLAS_CHECK(cublasDestroy(lin::g_cublas));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
