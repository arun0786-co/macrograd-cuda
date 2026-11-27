// trains an MLP on mnist with the kernels from kernels/, everything on the GPU:
//
//    x (B, 784) -> linear(784, 128) -> relu -> linear(128, 10) -> cross entropy
//
// one step: gather the batch's images and labels, forward, backward written out by hand
// layer by layer (no autograd), then one SGD step over all the weights at once. the
// weights, gradients and velocities each sit in one buffer (W1, b1, W2, b2 one after
// another), so one optimizer launch updates all 101,770 of them.
//
// the kernel files are included whole, each inside its own namespace so names like
// THREADS don't clash, with MACROGRAD_NO_MAIN so their tests are left out. the standard
// headers and common.h come first, so the includes inside the files find them already
// read and skip them.
//
//   ./build/train_mlp           trains for EPOCHS epochs with the blocked matmul, then
//                               again with cuBLAS, each once with plain launches and once
//                               with a CUDA graph, and prints time and test accuracy
//   ./build/train_mlp --check   runs the first 20 steps from fixed weights and compares
//                               the losses with macrograd's (tests/mnist_macrograd.txt)
//
// a step is about 15 kernel launches, and the CPU spends a few microseconds on each. a CUDA
// graph records the launches of one step once, and then the whole step goes to the GPU in
// one call. only work on a stream that isn't the legacy default stream can be recorded, so
// this builds with --default-stream per-thread, which gives every <<<>>> launch without a
// stream, and every async call with stream 0, the thread's own stream instead.
//
// needs data/mnist (python3 mnist/get_mnist.py). build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 --default-stream per-thread mnist/train_mlp.cu -lcublas -o build/train_mlp && ./build/train_mlp

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "../kernels/common.h"

#define MACROGRAD_NO_MAIN
namespace ew {
#include "../kernels/elementwise.cu"
}
namespace lin {
#include "../kernels/linear.cu"
}
namespace xent {
#include "../kernels/cross_entropy.cu"
}
namespace opt {
#include "../kernels/optim.cu"
}

const int IN = 784, HIDDEN = 128, CLASSES = 10;
const int BATCH = 64, EPOCHS = 10;
const float LR = 0.05f, MOMENTUM = 0.9f;
const int THREADS = 256;

// ---- data ----

struct Split {
    std::vector<float> images;  // n x 784, pixels / 255
    std::vector<int> labels;
    int n;
};

Split read_split(const std::string &images, const std::string &labels) {
    std::ifstream fi(images, std::ios::binary), fl(labels, std::ios::binary);
    if (!fi || !fl) {
        fprintf(stderr, "can't open %s, run python3 mnist/get_mnist.py from the repo root\n", images.c_str());
        exit(1);
    }
    std::vector<unsigned char> pix((std::istreambuf_iterator<char>(fi)), {});
    std::vector<unsigned char> lab((std::istreambuf_iterator<char>(fl)), {});
    Split s;
    s.n = (int)lab.size() - 8;  // the label file has an 8 byte header, the image file 16
    if ((size_t)s.n * IN + 16 != pix.size()) {
        fprintf(stderr, "%s and %s don't match\n", images.c_str(), labels.c_str());
        exit(1);
    }
    s.images.resize((size_t)s.n * IN);
    for (size_t i = 0; i < s.images.size(); i++) {
        s.images[i] = pix[16 + i] / 255.0f;
    }
    s.labels.assign(lab.begin() + 8, lab.end());
    return s;
}

// ---- small kernels this program needs ----

// the batch's rows of the image matrix and their labels, in the order idx gives
__global__ void gather(const float *images, const int *labels, const int *idx, float *xb, int *yb, int B) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)B * IN) {
        int r = i / IN, j = i % IN;
        xb[i] = images[(size_t)idx[r] * IN + j];
        if (j == 0) {
            yb[r] = labels[idx[r]];
        }
    }
}

// gather for the batch at step *step of the shuffled order. a recorded graph replays the
// same arguments every time, so the step number lives on the GPU and next_step moves it
__global__ void gather_step(const float *images, const int *labels, const int *idx, const int *step, float *xb,
                            int *yb, int B) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)B * IN) {
        const int *batch = idx + (size_t)*step * B;
        int r = i / IN, j = i % IN;
        xb[i] = images[(size_t)batch[r] * IN + j];
        if (j == 0) {
            yb[r] = labels[batch[r]];
        }
    }
}

__global__ void next_step(int *step) { *step += 1; }

// one thread per row: is the largest score at the label? ties go to the lower class
__global__ void count_correct(const float *logits, const int *labels, int *correct, int B) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < B) {
        int best = 0;
        for (int c = 1; c < CLASSES; c++) {
            if (logits[r * CLASSES + c] > logits[r * CLASSES + best]) {
                best = c;
            }
        }
        if (best == labels[r]) {
            atomicAdd(correct, 1);
        }
    }
}

int blocks(size_t n) { return (int)((n + THREADS - 1) / THREADS); }

// the loss's backward without the loss: training doesn't look at the loss every step, so
// the fused step skips the per-row losses and their mean, and goes straight from the
// logits to dlogits = (softmax - one_hot) / B. one thread per row, the same arithmetic as
// the thread-per-row kernels in cross_entropy.cu, so the result is the same to the bit
__global__ void softmax_minus_onehot(const float *logits, const int *labels, float *dlogits, int B) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < B) {
        const float *x = logits + r * CLASSES;
        float m = -INFINITY;
        for (int c = 0; c < CLASSES; c++) m = fmaxf(m, x[c]);
        float d = 0.0f;
        for (int c = 0; c < CLASSES; c++) d += expf(x[c] - m);
        // rounded step by step with the _rn intrinsics, as in cross_entropy.cu's backward:
        // left to itself the compiler fused p's multiply with the subtraction here but not
        // there, and 4 of the 640 values of the first batch came out one bit apart
        float inv_d = 1.0f / d, scale = 1.0f / B;
        for (int c = 0; c < CLASSES; c++) {
            float p = __fmul_rn(expf(x[c] - m), inv_d);
            dlogits[r * CLASSES + c] = __fmul_rn(scale, __fsub_rn(p, c == labels[r] ? 1.0f : 0.0f));
        }
    }
}

// ---- the model ----

const size_t W1 = 0, B1 = W1 + (size_t)IN * HIDDEN, W2 = B1 + HIDDEN, B2 = W2 + (size_t)HIDDEN * CLASSES,
             PARAMS = B2 + CLASSES;

struct Net {
    float *params, *grads, *vel;
    // activations and their gradients for one batch
    float *xb, *h_pre, *h, *logits, *dlogits, *dh, *dh_pre, *row_loss, *loss, *dloss;
    float2 *stats;
    int *yb, *idx, *correct, *step_number;
    bool cublas, fused = false;

    void forward(int B) {
        if (cublas) {
            lin::cublas_forward(xb, params + W1, params + B1, h_pre, B, IN, HIDDEN);
        } else {
            lin::linear_forward(xb, params + W1, params + B1, h_pre, B, IN, HIDDEN);
        }
        ew::forward_kernel<ew::Relu><<<blocks((size_t)B * HIDDEN), THREADS>>>(h_pre, h, B * HIDDEN);
        if (cublas) {
            lin::cublas_forward(h, params + W2, params + B2, logits, B, HIDDEN, CLASSES);
        } else {
            lin::linear_forward(h, params + W2, params + B2, logits, B, HIDDEN, CLASSES);
        }
        xent::cross_entropy(logits, yb, row_loss, stats, loss, B, CLASSES);
    }

    // the backward adds into the gradients, so they start at 0 every step
    void backward(int B) {
        CUDA_CHECK(cudaMemsetAsync(grads, 0, PARAMS * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(dlogits, 0, (size_t)B * CLASSES * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(dh, 0, (size_t)B * HIDDEN * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(dh_pre, 0, (size_t)B * HIDDEN * sizeof(float)));
        xent::cross_entropy_backward(logits, yb, stats, dloss, dlogits, B, CLASSES);
        auto backward_layer = cublas ? lin::cublas_backward : lin::linear_backward;
        backward_layer(h, params + W2, dlogits, dh, grads + W2, grads + B2, B, HIDDEN, CLASSES);
        ew::backward_kernel<ew::Relu><<<blocks((size_t)B * HIDDEN), THREADS>>>(h_pre, dh, dh_pre, B * HIDDEN);
        // the images need no gradient, so dx is skipped
        backward_layer(xb, params + W1, dh_pre, nullptr, grads + W1, grads + B1, B, IN, HIDDEN);
    }

    // the same step in fewer kernels, with the blocked matmuls only:
    // - relu inside the first layer's matmul, so h is written once and h_pre not at all
    // - relu's backward inside the matmul that makes dh: zero where h <= 0. relu's slope
    //   can be read from its output, h > 0 exactly where h_pre > 0
    // - the loss's backward without the loss (softmax_minus_onehot)
    // - every gradient written instead of added to memory zeroed first, so no memsets
    // each value is worked out with the same operations in the same order as in forward()
    // and backward(): adding to 0 and multiplying by a slope of 1 change nothing, so the
    // weights come out the same to the bit
    void forward_backward_fused(int B) {
        lin::matmul<false, false>(xb, params + W1, h, params + B1, false, B, HIDDEN, IN, lin::AUTO, {true, nullptr});
        lin::linear_forward(h, params + W2, params + B2, logits, B, HIDDEN, CLASSES);
        softmax_minus_onehot<<<blocks(B), THREADS>>>(logits, yb, dlogits, B);
        lin::matmul<false, true>(dlogits, params + W2, dh_pre, nullptr, false, B, HIDDEN, CLASSES, lin::AUTO, {false, h});
        lin::matmul<true, false>(h, dlogits, grads + W2, nullptr, false, HIDDEN, CLASSES, B);
        lin::launch_column_sum(dlogits, grads + B2, B, CLASSES, false);
        lin::matmul<true, false>(xb, dh_pre, grads + W1, nullptr, false, IN, HIDDEN, B);
        lin::launch_column_sum(dh_pre, grads + B1, B, HIDDEN, false);
    }

    void forward_backward(int B) {
        if (fused) {
            forward_backward_fused(B);
        } else {
            forward(B);
            backward(B);
        }
    }

    void step(const float *images, const int *labels, const int *batch_idx, int B, opt::SGD &sgd) {
        gather<<<blocks((size_t)B * IN), THREADS>>>(images, labels, batch_idx, xb, yb, B);
        forward_backward(B);
        sgd.step(params, grads, vel, PARAMS);
    }

    // a full batch of BATCH, recorded into a graph: the batch comes from step_number
    cudaGraphExec_t record_step(const float *images, const int *labels, opt::SGD &sgd) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));
        gather_step<<<blocks((size_t)BATCH * IN), THREADS>>>(images, labels, idx, step_number, xb, yb, BATCH);
        forward_backward(BATCH);
        sgd.step(params, grads, vel, PARAMS);
        next_step<<<1, 1>>>(step_number);
        CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        return exec;
    }
};

Net make_net() {
    Net n;
    CUDA_CHECK(cudaMalloc(&n.params, PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&n.grads, PARAMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&n.vel, PARAMS * sizeof(float)));
    const size_t rows = 1000;  // the largest batch: test batches of 1000
    float **acts[] = {&n.xb, &n.h_pre, &n.h, &n.logits, &n.dlogits, &n.dh, &n.dh_pre, &n.row_loss};
    size_t sizes[] = {rows * IN, rows * HIDDEN, rows * HIDDEN, rows * CLASSES, rows * CLASSES, rows * HIDDEN,
                      rows * HIDDEN, rows};
    for (int k = 0; k < 8; k++) {
        CUDA_CHECK(cudaMalloc(acts[k], sizes[k] * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&n.loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&n.dloss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&n.stats, rows * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&n.yb, rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&n.idx, 60000 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&n.correct, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&n.step_number, sizeof(int)));
    float one = 1.0f;  // the loss is the output, so its gradient is 1
    CUDA_CHECK(cudaMemcpy(n.dloss, &one, sizeof(float), cudaMemcpyHostToDevice));
    return n;
}

void upload(float *d, const std::vector<float> &v) {
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// macrograd's linear init: weights uniform in +-1 / sqrt(inputs), biases 0
std::vector<float> random_init(std::mt19937 &rng) {
    std::vector<float> p(PARAMS, 0.0f);
    std::uniform_real_distribution<float> u1(-1.0f / sqrtf(IN), 1.0f / sqrtf(IN)), u2(-1.0f / sqrtf(HIDDEN), 1.0f / sqrtf(HIDDEN));
    for (size_t i = W1; i < B1; i++) p[i] = u1(rng);
    for (size_t i = W2; i < B2; i++) p[i] = u2(rng);
    return p;
}

// the weights tests/make_mnist_fixture.py gives macrograd: a hash of the index, every
// value a whole number times 2^-shift, so float holds them exactly
std::vector<float> hashed_init() {
    std::vector<float> p(PARAMS, 0.0f);
    auto fill = [&](size_t start, size_t n, uint32_t seed, int shift) {
        for (size_t i = 0; i < n; i++) {
            uint32_t u = ((uint32_t)i * 2654435761u + seed * 40503u) >> 12;
            p[start + i] = ldexpf((float)((int)u - (1 << 19)), -shift);
        }
    };
    fill(W1, (size_t)IN * HIDDEN, 1, 24);
    fill(W2, (size_t)HIDDEN * CLASSES, 2, 22);
    return p;
}

// mean loss and accuracy on a split, in batches of 1000
void evaluate(Net &net, const float *images, const int *labels, int n, double *loss, double *acc) {
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    CUDA_CHECK(cudaMemcpy(net.idx, all.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(net.correct, 0, sizeof(int)));
    double total = 0.0;
    for (int first = 0; first < n; first += 1000) {
        int B = std::min(1000, n - first);
        gather<<<blocks((size_t)B * IN), THREADS>>>(images, labels, net.idx + first, net.xb, net.yb, B);
        net.forward(B);
        count_correct<<<blocks(B), THREADS>>>(net.logits, net.yb, net.correct, B);
        float l;
        CUDA_CHECK(cudaMemcpy(&l, net.loss, sizeof(float), cudaMemcpyDeviceToHost));
        total += (double)l * B;
    }
    int correct;
    CUDA_CHECK(cudaMemcpy(&correct, net.correct, sizeof(int), cudaMemcpyDeviceToHost));
    *loss = total / n;
    *acc = (double)correct / n;
}

struct DeviceSplit {
    float *images;
    int *labels;
    int n;
};

DeviceSplit to_device(const Split &s) {
    DeviceSplit d{nullptr, nullptr, s.n};
    CUDA_CHECK(cudaMalloc(&d.images, s.images.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.labels, s.n * sizeof(int)));
    upload(d.images, s.images);
    CUDA_CHECK(cudaMemcpy(d.labels, s.labels.data(), s.n * sizeof(int), cudaMemcpyHostToDevice));
    return d;
}

// the first 20 steps from the hashed weights on batches of 32 in file order, with
// macrograd's settings, against macrograd's losses
bool check(Net &net, const DeviceSplit &train) {
    std::ifstream f("tests/mnist_macrograd.txt");
    if (!f) {
        printf("tests/mnist_macrograd.txt not found, run from the repo root\n");
        return false;
    }
    std::vector<double> want;
    int step;
    double l;
    while (f >> step >> l) want.push_back(l);
    bool ok = true;
    for (bool cublas : {false, true}) {
        net.cublas = cublas;
        upload(net.params, hashed_init());
        CUDA_CHECK(cudaMemset(net.vel, 0, PARAMS * sizeof(float)));
        std::vector<int> order(want.size() * 32);
        std::iota(order.begin(), order.end(), 0);
        CUDA_CHECK(cudaMemcpy(net.idx, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
        opt::SGD sgd{0.1f, 0.9f, 0.0f};
        double worst = 0.0;
        for (size_t s = 0; s < want.size(); s++) {
            net.step(train.images, train.labels, net.idx + s * 32, 32, sgd);
            float got;
            CUDA_CHECK(cudaMemcpy(&got, net.loss, sizeof(float), cudaMemcpyDeviceToHost));
            worst = fmax(worst, std::isnan(got) ? INFINITY : fabs(got - want[s]));
            if (s == 0 || s + 1 == want.size()) {
                printf("  %s step %2zu: loss %.7f, macrograd %.7f\n", cublas ? "cublas " : "blocked", s, got, want[s]);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        // float against double over 20 steps of training: differences grow a little each step
        bool good = worst < 1e-5;
        printf("%s: %zu steps, largest loss difference %.2e, %s\n", cublas ? "cublas " : "blocked", want.size(), worst,
               good ? "ok" : "WRONG");
        ok = ok && good;
    }
    return ok;
}

// returns the test loss after the last epoch, so the runs with and without a graph can
// be compared
double train_once(Net &net, const DeviceSplit &trainset, const DeviceSplit &testset, bool cublas, bool graph,
                  bool fused) {
    net.cublas = cublas;
    net.fused = fused;
    std::mt19937 rng(0);
    opt::SGD sgd{LR, MOMENTUM, 0.0f};
    std::vector<int> order(trainset.n);
    std::iota(order.begin(), order.end(), 0);
    // one step before the real start, so the scratch buffers and cuBLAS's workspace are
    // allocated now: a recorded graph can't allocate. its weight changes are thrown away
    CUDA_CHECK(cudaMemcpy(net.idx, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
    net.step(trainset.images, trainset.labels, net.idx, BATCH, sgd);
    CUDA_CHECK(cudaDeviceSynchronize());
    upload(net.params, random_init(rng));
    CUDA_CHECK(cudaMemset(net.vel, 0, PARAMS * sizeof(float)));
    cudaGraphExec_t exec = nullptr;
    if (graph) {
        exec = net.record_step(trainset.images, trainset.labels, sgd);
    }
    printf("\n%s matmuls, %s%s, batch %d, SGD lr %.2f momentum %.1f\n", cublas ? "cuBLAS" : "blocked",
           graph ? "CUDA graph" : "plain launches", fused ? ", fused" : "", BATCH, LR, MOMENTUM);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double total_ms = 0.0, test_loss = 0.0;
    int full = trainset.n / BATCH;  // 937 full batches, then one of 32
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        std::shuffle(order.begin(), order.end(), rng);
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyAsync(net.idx, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
        if (graph) {
            CUDA_CHECK(cudaMemsetAsync(net.step_number, 0, sizeof(int)));
            for (int s = 0; s < full; s++) {
                CUDA_CHECK(cudaGraphLaunch(exec, cudaStreamPerThread));
            }
        } else {
            for (int s = 0; s < full; s++) {
                net.step(trainset.images, trainset.labels, net.idx + s * BATCH, BATCH, sgd);
            }
        }
        // the last, smaller batch is launched the plain way in both
        int first = full * BATCH;
        if (first < trainset.n) {
            net.step(trainset.images, trainset.labels, net.idx + first, trainset.n - first, sgd);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        double train_loss, train_acc, test_acc;
        evaluate(net, trainset.images, trainset.labels, trainset.n, &train_loss, &train_acc);
        evaluate(net, testset.images, testset.labels, testset.n, &test_loss, &test_acc);
        printf("epoch %2d | %6.1f ms | train loss %.4f acc %.2f%% | test loss %.4f acc %.2f%%\n", epoch, ms, train_loss,
               100 * train_acc, test_loss, 100 * test_acc);
    }
    int steps = (trainset.n + BATCH - 1) / BATCH;
    printf("%.1f ms per epoch, %.1f us per step\n", total_ms / EPOCHS, 1000 * total_ms / EPOCHS / steps);
    if (exec) {
        CUDA_CHECK(cudaGraphExecDestroy(exec));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return test_loss;
}

// how long each part of a step takes on the GPU, each part repeated on its own, so the
// CPU's launch cost doesn't count unless the GPU waits for it
void breakdown(Net &net, const DeviceSplit &trainset) {
    const int B = BATCH;
    std::vector<int> order(trainset.n);
    std::iota(order.begin(), order.end(), 0);
    CUDA_CHECK(cudaMemcpy(net.idx, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::mt19937 rng(0);
    upload(net.params, random_init(rng));
    opt::SGD sgd{LR, MOMENTUM, 0.0f};
    net.step(trainset.images, trainset.labels, net.idx, B, sgd);  // fills every buffer
    for (bool cublas : {false, true}) {
        net.cublas = cublas;
        auto fwd = cublas ? lin::cublas_forward : lin::linear_forward;
        auto bwd = cublas ? lin::cublas_backward : lin::linear_backward;
        float t[9];
        t[0] = lin::time_ms([&] { gather<<<blocks((size_t)B * IN), THREADS>>>(trainset.images, trainset.labels, net.idx, net.xb, net.yb, B); });
        t[1] = lin::time_ms([&] { fwd(net.xb, net.params + W1, net.params + B1, net.h_pre, B, IN, HIDDEN); });
        t[2] = lin::time_ms([&] { ew::forward_kernel<ew::Relu><<<blocks((size_t)B * HIDDEN), THREADS>>>(net.h_pre, net.h, B * HIDDEN); });
        t[3] = lin::time_ms([&] { fwd(net.h, net.params + W2, net.params + B2, net.logits, B, HIDDEN, CLASSES); });
        t[4] = lin::time_ms([&] { xent::cross_entropy(net.logits, net.yb, net.row_loss, net.stats, net.loss, B, CLASSES); });
        t[5] = lin::time_ms([&] {
            CUDA_CHECK(cudaMemsetAsync(net.grads, 0, PARAMS * sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(net.dlogits, 0, (size_t)B * CLASSES * sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(net.dh, 0, (size_t)B * HIDDEN * sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(net.dh_pre, 0, (size_t)B * HIDDEN * sizeof(float)));
            xent::cross_entropy_backward(net.logits, net.yb, net.stats, net.dloss, net.dlogits, B, CLASSES);
        });
        t[6] = lin::time_ms([&] {
            bwd(net.h, net.params + W2, net.dlogits, net.dh, net.grads + W2, net.grads + B2, B, HIDDEN, CLASSES);
            ew::backward_kernel<ew::Relu><<<blocks((size_t)B * HIDDEN), THREADS>>>(net.h_pre, net.dh, net.dh_pre, B * HIDDEN);
        });
        t[7] = lin::time_ms([&] { bwd(net.xb, net.params + W1, net.dh_pre, nullptr, net.grads + W1, net.grads + B1, B, IN, HIDDEN); });
        t[8] = lin::time_ms([&] { sgd.step(net.params, net.grads, net.vel, PARAMS); });
        const char *what[] = {"gather", "layer 1 forward", "relu", "layer 2 forward", "loss", "zero grads + loss backward",
                              "layer 2 backward + relu backward", "layer 1 backward", "sgd"};
        printf("\n%s matmuls, one step of %d, each part alone:\n", cublas ? "cuBLAS" : "blocked", B);
        float total = 0;
        for (int k = 0; k < 9; k++) {
            printf("  %-34s %7.1f us\n", what[k], 1000 * t[k]);
            total += t[k];
        }
        printf("  %-34s %7.1f us\n", "sum", 1000 * total);
    }
}

// the graph replays the same kernels in the same order, and the fused step does the same
// arithmetic, so with the blocked matmuls all four runs have to give exactly the same
// weights, and so the same test loss to the last bit. with cuBLAS, the two runs too
bool train(Net &net, const DeviceSplit &trainset, const DeviceSplit &testset) {
    double blocked[4], cublas[2];
    int n = 0;
    for (bool fused : {false, true}) {
        for (bool graph : {false, true}) {
            blocked[n++] = train_once(net, trainset, testset, false, graph, fused);
        }
    }
    cublas[0] = train_once(net, trainset, testset, true, false, false);
    cublas[1] = train_once(net, trainset, testset, true, true, false);
    bool same = blocked[1] == blocked[0] && blocked[2] == blocked[0] && blocked[3] == blocked[0];
    printf("\nblocked: same test loss with and without the graph and the fusing: %s\n", same ? "yes" : "NO");
    bool same_cublas = cublas[0] == cublas[1];
    printf("cuBLAS: same test loss with and without the graph: %s\n", same_cublas ? "yes" : "NO");
    return same && same_cublas;
}

int main(int argc, char **argv) {
    bool check_only = argc > 1 && strcmp(argv[1], "--check") == 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    CUBLAS_CHECK(cublasCreate(&lin::g_cublas));
    // cuBLAS has to use the same stream as the kernels, or its calls wouldn't be recorded
    CUBLAS_CHECK(cublasSetStream(lin::g_cublas, cudaStreamPerThread));
    Split tr = read_split("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
    Split te = read_split("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    printf("%d training and %d test images\n", tr.n, te.n);
    DeviceSplit trainset = to_device(tr), testset = to_device(te);
    Net net = make_net();

    bool ok = check(net, trainset);
    if (!check_only) {
        breakdown(net, trainset);
        ok = train(net, trainset, testset) && ok;
    }
    CUBLAS_CHECK(cublasDestroy(lin::g_cublas));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
