// Minimal stand-in for PyTorch's c10 header so FlashAttention-2 builds without libtorch.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#define C10_CUDA_CHECK(expr)                                                                   \
    do {                                                                                       \
        cudaError_t err__ = (expr);                                                            \
        if (err__ != cudaSuccess)                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
    } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
