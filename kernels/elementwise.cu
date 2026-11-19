// elementwise activations and their backward: relu, tanh, sigmoid, gelu
// one thread per element, like vector_add.
//   forward:  y[i] = f(x[i])
//   backward: dx[i] += dy[i] * f'(x[i])
// backward adds into dx instead of overwriting it, like macrograd's backward, so a tensor
// that feeds two ops ends up with the sum of both grads.
// the slopes of tanh and sigmoid are cheapest from the output y:
//   tanh'(x) = 1 - y^2      sigmoid'(x) = y * (1 - y)
// while relu and gelu need x. so the backward kernel reads one input, x or y, whichever
// the op says, plus dy and dx.
// each op is a struct with a forward and a slope function, and one kernel template runs
// any of them, so the four ops share the indexing and the bounds check.
//
// the check compares against a CPU version in double precision, written separately from
// the GPU code, and first checks that version's slopes against finite differences.
//
// build and run: nvcc -O3 -arch=native kernels/elementwise.cu -o run && ./run

#include <cmath>
#include <random>
#include <vector>
#include "common.h"

struct Relu {
    // max(x, 0). the slope is 1 for x > 0 and 0 otherwise, 0 at x = 0 too, as in macrograd
    static constexpr bool slope_from_output = false;
    __device__ static float forward(float x) { return x > 0.0f ? x : 0.0f; }
    __device__ static float slope(float x) { return x > 0.0f ? 1.0f : 0.0f; }
};

struct Tanh {
    static constexpr bool slope_from_output = true;
    __device__ static float forward(float x) { return tanhf(x); }
    __device__ static float slope(float y) { return 1.0f - y * y; }
};

struct Sigmoid {
    // 1 / (1 + e^-x). for very negative x, e^-x overflows to inf and the result is 0,
    // which is the right limit, so this form needs no special case in float
    static constexpr bool slope_from_output = true;
    __device__ static float forward(float x) { return 1.0f / (1.0f + expf(-x)); }
    __device__ static float slope(float y) { return y * (1.0f - y); }
};

struct Gelu {
    // the tanh approximation, the same one macrograd uses:
    //   gelu(x) = 0.5 * x * (1 + t),  t = tanh(u),  u = sqrt(2/pi) * (x + 0.044715 * x^3)
    //   gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * du/dx,  du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    // the slope can't be had from y alone, so it works out t again from x
    static constexpr bool slope_from_output = false;
    static constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    __device__ static float forward(float x) {
        float t = tanhf(k * (x + 0.044715f * x * x * x));
        return 0.5f * x * (1.0f + t);
    }
    __device__ static float slope(float x) {
        float t = tanhf(k * (x + 0.044715f * x * x * x));
        float du_dx = k * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du_dx;
    }
};

template <typename Op>
__global__ void forward_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = Op::forward(x[i]);
    }
}

// in is x or y, depending on Op::slope_from_output
template <typename Op>
__global__ void backward_kernel(const float *in, const float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] += dy[i] * Op::slope(in[i]);
    }
}

// the CPU version, in double, from the formulas rather than from the structs above
double relu_ref(double x) { return x > 0.0 ? x : 0.0; }
double relu_slope_ref(double x) { return x > 0.0 ? 1.0 : 0.0; }
double tanh_ref(double x) { return std::tanh(x); }
double tanh_slope_ref(double x) { double t = std::tanh(x); return 1.0 - t * t; }
double sigmoid_ref(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double sigmoid_slope_ref(double x) { double s = sigmoid_ref(x); return s * (1.0 - s); }
double gelu_ref(double x) {
    const double k = std::sqrt(2.0 / M_PI);
    return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
}
double gelu_slope_ref(double x) {
    const double k = std::sqrt(2.0 / M_PI);
    double t = std::tanh(k * (x + 0.044715 * x * x * x));
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3.0 * 0.044715 * x * x);
}

// central differences in double: (f(x + h) - f(x - h)) / 2h is off by about h^2 * f''' / 6,
// so with h = 1e-5 a right slope agrees to far better than 1e-6
bool slopes_match_finite_differences(double (*f)(double), double (*slope)(double)) {
    const double h = 1e-5;
    for (double x = -8.0; x <= 8.0; x += 0.0625) {
        if (std::fabs(x) < 2 * h) continue;  // relu has a kink at 0
        double numeric = (f(x + h) - f(x - h)) / (2 * h);
        if (std::fabs(numeric - slope(x)) > 1e-6) {
            printf("  slope at x = %g: formula %.9f, finite difference %.9f\n", x, slope(x), numeric);
            return false;
        }
    }
    return true;
}

// error of a GPU value against the double reference, relative for big values and absolute
// for small ones
double err(float got, double want) {
    return std::fabs((double)got - want) / (1.0 + std::fabs(want));
}

float time_ms(void (*launch)(const float *, const float *, float *, int, int), const float *a,
              const float *b, float *c, int n, int blocks) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch(a, b, c, n, blocks);  // warmup
    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        launch(a, b, c, n, blocks);
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

const int THREADS = 256;

template <typename Op>
void launch_forward(const float *x, const float *, float *y, int n, int blocks) {
    forward_kernel<Op><<<blocks, THREADS>>>(x, y, n);
}

template <typename Op>
void launch_backward(const float *in, const float *dy, float *dx, int n, int blocks) {
    backward_kernel<Op><<<blocks, THREADS>>>(in, dy, dx, n);
}

// runs one op forward and backward, checks both against the reference and times them.
// returns true if everything matched
template <typename Op>
bool check(const char *name, double (*f)(double), double (*slope)(double),
           const std::vector<float> &x, const std::vector<float> &dy, const std::vector<float> &dx0,
           float *d_x, float *d_y, float *d_dy, float *d_dx) {
    int n = (int)x.size();
    size_t bytes = n * sizeof(float);
    int blocks = (n + THREADS - 1) / THREADS;
    bool fd_ok = slopes_match_finite_differences(f, slope);

    forward_kernel<Op><<<blocks, THREADS>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_dx, dx0.data(), bytes, cudaMemcpyHostToDevice));
    const float *d_in = Op::slope_from_output ? d_y : d_x;
    backward_kernel<Op><<<blocks, THREADS>>>(d_in, d_dy, d_dx, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> y(n), dx(n);
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, bytes, cudaMemcpyDeviceToHost));
    double fwd_err = 0.0, bwd_err = 0.0;
    for (int i = 0; i < n; i++) {
        fwd_err = std::fmax(fwd_err, err(y[i], f(x[i])));
        bwd_err = std::fmax(bwd_err, err(dx[i], (double)dx0[i] + (double)dy[i] * slope(x[i])));
    }

    // forward reads x and writes y: 8 bytes per element. backward reads in, dy and dx and
    // writes dx: 16 bytes per element. the timed backward runs keep adding into dx, which
    // doesn't change the memory traffic
    float fwd_ms = time_ms(launch_forward<Op>, d_x, nullptr, d_y, n, blocks);
    float bwd_ms = time_ms(launch_backward<Op>, d_in, d_dy, d_dx, n, blocks);
    bool ok = fd_ok && fwd_err < 1e-5 && bwd_err < 1e-5;
    printf("%-8s %-10s %-10.2e %-10.2e %-8.4f %-9.1f %-8.4f %-9.1f\n", name, fd_ok ? "ok" : "WRONG",
           fwd_err, bwd_err, fwd_ms, 2.0 * bytes / (fwd_ms * 1e-3) / 1e9, bwd_ms,
           4.0 * bytes / (bwd_ms * 1e-3) / 1e9);
    return ok;
}

int main() {
    // not a multiple of the block size, so the i < n check gets used
    int n = (1 << 24) + 7;
    size_t bytes = n * sizeof(float);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> wide(-6.0f, 6.0f), unit(-1.0f, 1.0f);
    std::vector<float> x(n), dy(n), dx0(n);
    for (int i = 0; i < n; i++) {
        x[i] = wide(rng);
        dy[i] = unit(rng);
        dx0[i] = unit(rng);  // backward adds to this
    }
    // exactly 0 (relu's kink), tiny values, and values where tanh, sigmoid and gelu flatten out
    float special[] = {0.0f, -0.0f, 1e-7f, -1e-7f, 1e-3f, -1e-3f, 10.0f, -10.0f, 30.0f, -30.0f, 100.0f, -100.0f};
    for (int i = 0; i < (int)(sizeof(special) / sizeof(float)); i++) {
        x[i] = special[i];
    }

    float *d_x, *d_y, *d_dy, *d_dx;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_dy, bytes));
    CUDA_CHECK(cudaMalloc(&d_dx, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dy, dy.data(), bytes, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, n = %d\n\n", prop.name, n);
    printf("%-8s %-10s %-10s %-10s %-8s %-9s %-8s %-9s\n", "op", "fd check", "fwd err", "bwd err",
           "fwd ms", "fwd GB/s", "bwd ms", "bwd GB/s");

    bool ok = true;
    ok &= check<Relu>("relu", relu_ref, relu_slope_ref, x, dy, dx0, d_x, d_y, d_dy, d_dx);
    ok &= check<Tanh>("tanh", tanh_ref, tanh_slope_ref, x, dy, dx0, d_x, d_y, d_dy, d_dx);
    ok &= check<Sigmoid>("sigmoid", sigmoid_ref, sigmoid_slope_ref, x, dy, dx0, d_x, d_y, d_dy, d_dx);
    ok &= check<Gelu>("gelu", gelu_ref, gelu_slope_ref, x, dy, dx0, d_x, d_y, d_dy, d_dx);

    // the fastest a kernel that reads one array and writes one can go: a device-to-device copy
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemcpy(d_y, d_x, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; it++) {
        CUDA_CHECK(cudaMemcpy(d_y, d_x, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 100;
    printf("\ncudaMemcpy device to device: %.4f ms, %.1f GB/s\n", ms, 2.0 * bytes / (ms * 1e-3) / 1e9);
    printf("%s\n", ok ? "all match" : "MISMATCH");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_dx));
    return ok ? 0 : 1;
}
