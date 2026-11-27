// optimizer steps for SGD with momentum and Adam(W), one thread per weight, with the same
// formulas as macrograd's optim.py. every weight is updated on its own, so these are
// elementwise kernels: each reads a few floats per weight, does a few flops and writes
// them back, and memory bandwidth sets the speed.
//
// SGD, with velocity v starting at 0:
//    g = grad + weight_decay * w
//    v = momentum * v + g
//    w = w - lr * v
//
// Adam, with m and v starting at 0 and t the step number, counting from 1:
//    m = beta1 * m + (1 - beta1) * grad
//    v = beta2 * v + (1 - beta2) * grad^2
//    w = w - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
// m and v start at 0, so for the first steps they are too small. dividing by
// 1 - beta^t corrects that. the CPU works out 1 / (1 - beta^t) in double once per step
// and passes it in, so the kernel doesn't need t.
//
// AdamW first shrinks every weight by (1 - lr * weight_decay), then takes the Adam step.
// with weight_decay = 0 it's Adam.
//
// checked against a CPU version in double for one step, and against macrograd's SGD, Adam
// and AdamW for five steps on the same gradients (tests/optim_macrograd.txt, from
// tests/make_optim_fixture.py).
//
// build and run from the repo root:
//   nvcc -O3 -arch=native kernels/optim.cu -o build/optim && ./build/optim

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "common.h"

const int THREADS = 256;

// a grid-stride loop: a fixed number of blocks walks over any n, so a launch for a
// hundred million weights doesn't need a hundred million threads
__global__ void sgd_step(float *w, const float *grad, float *vel, float lr, float momentum, float weight_decay,
                         size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float wi = w[i];
        float g = grad[i] + weight_decay * wi;
        float v = momentum * vel[i] + g;
        vel[i] = v;
        w[i] = wi - lr * v;
    }
}

// the numbers that only depend on the settings and t, worked out on the CPU in double.
// 1 - beta2 is one of them: 0.999 isn't exact in float, and 1.0f - 0.999f is 0.00099998713,
// 1.3e-5 away from 0.001. in v and in the bias correction that moves every step by about
// 6e-6 of its size, 100 times a float rounding
struct AdamConsts {
    float lr, beta1, one_minus_beta1, beta2, one_minus_beta2, eps, shrink, correct1, correct2;
};

__global__ void adamw_step(float *w, const float *grad, float *m, float *v, AdamConsts k, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float g = grad[i];
        float mi = k.beta1 * m[i] + k.one_minus_beta1 * g;
        float vi = k.beta2 * v[i] + k.one_minus_beta2 * g * g;
        m[i] = mi;
        v[i] = vi;
        float wi = w[i] * k.shrink;
        w[i] = wi - k.lr * (mi * k.correct1) / (sqrtf(vi * k.correct2) + k.eps);
    }
}

// enough blocks to fill the GPU several times over, and no more than the weights need
int blocks_for(size_t n) {
    static int sms = 0;
    if (sms == 0) {
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    }
    size_t needed = (n + THREADS - 1) / THREADS;
    return (int)std::min(needed, (size_t)sms * 8);
}

struct SGD {
    float lr, momentum, weight_decay;
    void step(float *w, const float *grad, float *vel, size_t n) {
        sgd_step<<<blocks_for(n), THREADS>>>(w, grad, vel, lr, momentum, weight_decay, n);
    }
};

struct AdamW {
    double lr, beta1, beta2, eps, weight_decay;
    int t = 0;
    // step() counts the step and updates one tensor. with several tensors, add 1 to t
    // once per step and call launch() for each, so they all use the same t
    void step(float *w, const float *grad, float *m, float *v, size_t n) {
        t++;
        launch(w, grad, m, v, n);
    }
    void launch(float *w, const float *grad, float *m, float *v, size_t n) const {
        AdamConsts k = {(float)lr, (float)beta1, (float)(1.0 - beta1), (float)beta2, (float)(1.0 - beta2),
                        (float)eps, (float)(1.0 - lr * weight_decay), (float)(1.0 / (1.0 - pow(beta1, t))),
                        (float)(1.0 / (1.0 - pow(beta2, t)))};
        adamw_step<<<blocks_for(n), THREADS>>>(w, grad, m, v, k, n);
    }
};

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

const size_t CAP = ((size_t)1 << 24) + 1024;
const size_t PAST = 1024;  // floats after the end that must stay untouched

struct Device {
    float *w, *grad, *a, *b;  // a and b: SGD's velocity, or Adam's m and v
};

// a value no update makes from itself: an update of a weight past the end reads this, gives
// some other number and writes it, and the check sees it. nan wouldn't do, since an update
// of nan writes nan back
const float MARK = 3.0f;

// copies v to the start of the buffer and MARK into the PAST floats after it
void upload(float *d, const std::vector<float> &v) {
    std::vector<float> padded(v);
    padded.resize(v.size() + PAST, MARK);
    CUDA_CHECK(cudaMemcpy(d, padded.data(), padded.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> download(const float *d, size_t n, bool *untouched) {
    std::vector<float> v(n + PAST);
    CUDA_CHECK(cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t k = n; k < v.size(); k++) {
        *untouched = *untouched && v[k] == MARK;
    }
    v.resize(n);
    return v;
}

// largest |got - want| / scale, with scale 1 when none is given
double max_diff(const std::vector<float> &got, const std::vector<double> &want, const std::vector<double> *scale = nullptr) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        double e = fabs(got[i] - want[i]) / (scale ? (*scale)[i] : 1.0);
        worst = fmax(worst, std::isnan(e) ? INFINITY : e);
    }
    return worst;
}

// one step from random weights, gradients and optimizer state, against the same formulas
// in double. the weights, gradients and m are in [-1, 1] and v in [0, 1], so a float
// rounding is about 6e-8 and 1e-6 leaves room for a few. the new weight is measured
// against the size of what went into it: |w| plus the step worked out from the sizes of
// the terms of the velocity or of m. where v is near 0, Adam's step m / sqrt(v) can be 10
// or more, and where the terms of m cancel, m is small but its rounding isn't
double g_worst_sgd = 0.0, g_worst_adam = 0.0;  // largest errors over all the one-step checks

bool check_one_step(size_t n, int t, Device d, std::mt19937 &rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), pos(0.0f, 1.0f);
    std::vector<float> w(n), g(n), a(n), b(n);
    for (size_t i = 0; i < n; i++) {
        w[i] = u(rng);
        g[i] = u(rng);
        a[i] = u(rng);
        b[i] = pos(rng);
    }
    bool ok = true;
    const double tol = 1e-6;

    // SGD: velocity in a
    SGD sgd{0.1f, 0.9f, 0.01f};
    upload(d.w, w);
    upload(d.grad, g);
    upload(d.a, a);
    sgd.step(d.w, d.grad, d.a, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> want_w(n), want_a(n), want_b(n), scale(n);
    for (size_t i = 0; i < n; i++) {
        double gi = (double)g[i] + (double)sgd.weight_decay * w[i];
        want_a[i] = (double)sgd.momentum * a[i] + gi;
        want_w[i] = w[i] - (double)sgd.lr * want_a[i];
        scale[i] = fabs(w[i]) + sgd.lr * (sgd.momentum * fabs(a[i]) + fabs(g[i]) + sgd.weight_decay * fabs(w[i]));
    }
    bool untouched = true;
    double ew = max_diff(download(d.w, n, &untouched), want_w, &scale);
    double ea = max_diff(download(d.a, n, &untouched), want_a);
    g_worst_sgd = fmax(g_worst_sgd, fmax(ew, ea));
    bool good = ew < tol && ea < tol && untouched;
    if (!good) {
        printf("  sgd, n = %zu: w error %.2e, velocity error %.2e%s\n", n, ew, ea, untouched ? "" : ", wrote past the end");
    }
    ok = ok && good;

    // AdamW at step t, with m in a and v in b
    AdamW adam{0.01, 0.9, 0.999, 1e-8, 0.1};
    adam.t = t - 1;
    upload(d.w, w);
    upload(d.a, a);
    upload(d.b, b);
    adam.step(d.w, d.grad, d.a, d.b, n);
    CUDA_CHECK(cudaGetLastError());
    double c1 = 1.0 / (1.0 - pow(0.9, t)), c2 = 1.0 / (1.0 - pow(0.999, t));
    for (size_t i = 0; i < n; i++) {
        double b1 = adam.beta1, b2 = adam.beta2, lr = adam.lr;
        want_a[i] = b1 * a[i] + (1 - b1) * g[i];
        want_b[i] = b2 * b[i] + (1 - b2) * (double)g[i] * g[i];
        double wi = w[i] * (1 - lr * adam.weight_decay);
        want_w[i] = wi - lr * (want_a[i] * c1) / (sqrt(want_b[i] * c2) + adam.eps);
        double m_size = b1 * fabs(a[i]) + (1 - b1) * fabs(g[i]);
        scale[i] = fabs(wi) + lr * (m_size * c1) / (sqrt(want_b[i] * c2) + adam.eps);
    }
    untouched = true;
    ew = max_diff(download(d.w, n, &untouched), want_w, &scale);
    ea = max_diff(download(d.a, n, &untouched), want_a);
    double eb = max_diff(download(d.b, n, &untouched), want_b);
    g_worst_adam = fmax(g_worst_adam, fmax(ew, fmax(ea, eb)));
    good = ew < tol && ea < tol && eb < tol && untouched;
    if (!good) {
        printf("  adamw, n = %zu, t = %d: w error %.2e, m error %.2e, v error %.2e%s\n", n, t, ew, ea, eb,
               untouched ? "" : ", wrote past the end");
    }
    return ok && good;
}

std::vector<double> read_values(std::ifstream &f, size_t n) {
    std::vector<double> v(n);
    for (auto &x : v) f >> x;
    return v;
}

// five steps of each optimizer against macrograd, step by step
bool check_macrograd(const char *path, Device d) {
    std::ifstream f(path);
    if (!f) {
        printf("macrograd: %s not found, run from the repo root\n", path);
        return false;
    }
    size_t n;
    int steps;
    f >> n >> steps;
    std::vector<double> w0 = read_values(f, n);
    std::vector<std::vector<double>> grads;
    for (int s = 0; s < steps; s++) grads.push_back(read_values(f, n));
    auto as_float = [](const std::vector<double> &v) { return std::vector<float>(v.begin(), v.end()); };
    bool ok = true;
    for (int k = 0; k < 3; k++) {
        std::string name;
        double lr, beta1, beta2, eps, wd;
        f >> name >> lr >> beta1 >> beta2 >> eps >> wd;
        upload(d.w, as_float(w0));
        upload(d.a, std::vector<float>(n, 0.0f));
        upload(d.b, std::vector<float>(n, 0.0f));
        SGD sgd{(float)lr, (float)beta1, (float)wd};
        AdamW adam{lr, beta1, beta2, eps, wd};
        double worst = 0.0;
        for (int s = 0; s < steps; s++) {
            std::vector<double> want = read_values(f, n);
            upload(d.grad, as_float(grads[s]));
            if (name == "sgd") {
                sgd.step(d.w, d.grad, d.a, n);
            } else {
                adam.step(d.w, d.grad, d.a, d.b, n);
            }
            CUDA_CHECK(cudaGetLastError());
            bool untouched = true;
            worst = fmax(worst, max_diff(download(d.w, n, &untouched), want));
        }
        if (!f) {
            printf("macrograd: couldn't read %s\n", path);
            return false;
        }
        // the weights are about 1 and the steps are small, so a handful of float roundings
        bool good = worst < 1e-6;
        printf("macrograd %-5s, %d steps on %zu weights: largest difference %.2e, %s\n", name.c_str(), steps, n, worst,
               good ? "ok" : "WRONG");
        ok = ok && good;
    }
    return ok;
}

// the test. a program that includes this file for its kernels defines
// MACROGRAD_NO_MAIN first, like mnist/train_mlp.cu
#ifndef MACROGRAD_NO_MAIN
int main() {
    Device d;
    float **all[] = {&d.w, &d.grad, &d.a, &d.b};
    for (auto p : all) {
        CUDA_CHECK(cudaMalloc(p, CAP * sizeof(float)));
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::mt19937 rng(0);

    bool ok = check_macrograd("tests/optim_macrograd.txt", d);

    // one weight, less than a block, not a multiple of a block, more weights than the
    // grid has threads (so the grid-stride loop goes round), and Adam's first step and a late one
    bool steps_ok = true;
    for (size_t n : {(size_t)1, (size_t)200, (size_t)257, (size_t)101770, ((size_t)1 << 20) + 3}) {
        for (int t : {1, 2, 1000}) {
            steps_ok = check_one_step(n, t, d, rng) && steps_ok;
        }
    }
    printf("one step against double: %s, largest error sgd %.2e, adamw %.2e\n", steps_ok ? "ok" : "WRONG",
           g_worst_sgd, g_worst_adam);
    ok = ok && steps_ok;

    // timing. SGD reads w, grad and velocity and writes w and velocity, 20 bytes a
    // weight; AdamW reads w, grad, m and v and writes w, m and v, 28 bytes
    for (auto p : all) {
        CUDA_CHECK(cudaMemset(*p, 0, CAP * sizeof(float)));
    }
    printf("\n%-10s %-8s %-10s %-10s\n", "weights", "step", "ms", "GB/s");
    for (size_t n : {(size_t)101770, (size_t)1 << 24}) {
        SGD sgd{0.1f, 0.9f, 0.01f};
        AdamW adam{0.01, 0.9, 0.999, 1e-8, 0.1};
        float ms = time_ms([&] { sgd.step(d.w, d.grad, d.a, n); });
        printf("%-10zu %-8s %-10.4f %-10.1f\n", n, "sgd", ms, 20.0 * n / (ms * 1e-3) / 1e9);
        ms = time_ms([&] { adam.step(d.w, d.grad, d.a, d.b, n); });
        printf("%-10s %-8s %-10.4f %-10.1f\n", "", "adamw", ms, 28.0 * n / (ms * 1e-3) / 1e9);
    }
    size_t n = (size_t)1 << 24;
    float ms = time_ms([&] { CUDA_CHECK(cudaMemcpy(d.a, d.w, n * sizeof(float), cudaMemcpyDeviceToDevice)); });
    printf("%-10zu %-8s %-10.4f %-10.1f\n", n, "memcpy", ms, 8.0 * n / (ms * 1e-3) / 1e9);

    for (auto p : all) {
        CUDA_CHECK(cudaFree(*p));
    }
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
#endif
