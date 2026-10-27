// vector add: c[i] = a[i] + b[i]
// one thread per element. each thread works out its global index from its block and
// its position inside the block. the last block can run past the end of the arrays,
// so threads with i >= n do nothing.
//
// build and run: nvcc -O3 -arch=native kernels/vector_add.cu -o run && ./run

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // not a multiple of the block size, so the i < n check gets used
    int n = (1 << 24) + 7;
    size_t bytes = n * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    float *c_ref = (float *)malloc(bytes);
    srand(0);
    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        b[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
    vector_add_cpu(a, b, c_ref, n);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    // one float add per element on both sides, so the results should match exactly
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        max_diff = fmaxf(max_diff, fabsf(c[i] - c_ref[i]));
    }
    printf("n = %d, max diff vs cpu = %g\n", n, max_diff);

    // timing: the first launch above was the warmup. each launch reads a and b and
    // writes c, so it moves 3 * bytes through memory.
    int iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; it++) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;
    printf("%.4f ms per launch, %.1f GB/s\n", ms, 3.0 * bytes / (ms * 1e-3) / 1e9);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s\n", prop.name);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(a);
    free(b);
    free(c);
    free(c_ref);
    return max_diff == 0.0f ? 0 : 1;
}
