// shared by every kernel file. nvcc finds it next to the .cu file, so each file still
// builds on its own with one nvcc command.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA calls return an error code instead of throwing. this prints where the call was and
// what went wrong, then stops, so a failed malloc or launch can't go unnoticed.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
