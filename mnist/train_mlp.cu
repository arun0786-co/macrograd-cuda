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
//                               again with cuBLAS, and prints time and test accuracy
//   ./build/train_mlp --check   runs the first 20 steps from fixed weights and compares
//                               the losses with macrograd's (tests/mnist_macrograd.txt)
//
// needs data/mnist (python3 mnist/get_mnist.py). build and run from the repo root:
//   nvcc -O3 -arch=native -std=c++17 mnist/train_mlp.cu -lcublas -o build/train_mlp && ./build/train_mlp

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

// ---- the model ----

const size_t W1 = 0, B1 = W1 + (size_t)IN * HIDDEN, W2 = B1 + HIDDEN, B2 = W2 + (size_t)HIDDEN * CLASSES,
             PARAMS = B2 + CLASSES;

struct Net {
    float *params, *grads, *vel;
    // activations and their gradients for one batch
    float *xb, *h_pre, *h, *logits, *dlogits, *dh, *dh_pre, *row_loss, *loss, *dloss;
    float2 *stats;
    int *yb, *idx, *correct;
    bool cublas;

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

    void step(const float *images, const int *labels, const int *batch_idx, int B, opt::SGD &sgd) {
        gather<<<blocks((size_t)B * IN), THREADS>>>(images, labels, batch_idx, xb, yb, B);
        forward(B);
        backward(B);
        sgd.step(params, grads, vel, PARAMS);
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

void train(Net &net, const DeviceSplit &trainset, const DeviceSplit &testset) {
    for (bool cublas : {false, true}) {
        net.cublas = cublas;
        std::mt19937 rng(0);
        upload(net.params, random_init(rng));
        CUDA_CHECK(cudaMemset(net.vel, 0, PARAMS * sizeof(float)));
        opt::SGD sgd{LR, MOMENTUM, 0.0f};
        std::vector<int> order(trainset.n);
        std::iota(order.begin(), order.end(), 0);
        printf("\n%s matmuls, batch %d, SGD lr %.2f momentum %.1f\n", cublas ? "cuBLAS" : "blocked", BATCH, LR, MOMENTUM);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0.0;
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            std::shuffle(order.begin(), order.end(), rng);
            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpyAsync(net.idx, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice));
            // the last batch is smaller: 60000 = 937 * 64 + 32
            for (int first = 0; first < trainset.n; first += BATCH) {
                net.step(trainset.images, trainset.labels, net.idx + first, std::min(BATCH, trainset.n - first), sgd);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
            double train_loss, train_acc, test_loss, test_acc;
            evaluate(net, trainset.images, trainset.labels, trainset.n, &train_loss, &train_acc);
            evaluate(net, testset.images, testset.labels, testset.n, &test_loss, &test_acc);
            printf("epoch %2d | %6.1f ms | train loss %.4f acc %.2f%% | test loss %.4f acc %.2f%%\n", epoch, ms,
                   train_loss, 100 * train_acc, test_loss, 100 * test_acc);
        }
        int steps = (trainset.n + BATCH - 1) / BATCH;
        printf("%.1f ms per epoch, %.1f us per step\n", total_ms / EPOCHS, 1000 * total_ms / EPOCHS / steps);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

int main(int argc, char **argv) {
    bool check_only = argc > 1 && strcmp(argv[1], "--check") == 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    CUBLAS_CHECK(cublasCreate(&lin::g_cublas));
    Split tr = read_split("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte");
    Split te = read_split("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte");
    printf("%d training and %d test images\n", tr.n, te.n);
    DeviceSplit trainset = to_device(tr), testset = to_device(te);
    Net net = make_net();

    bool ok = check(net, trainset);
    if (!check_only) {
        train(net, trainset, testset);
    }
    CUBLAS_CHECK(cublasDestroy(lin::g_cublas));
    printf("\n%s\n", ok ? "all ok" : "SOMETHING IS WRONG");
    return ok ? 0 : 1;
}
