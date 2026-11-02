// Shared device helpers for the krea2 CUDA backend.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>

typedef __nv_bfloat16 bf16;

#define KREA_API extern "C"

// dtype codes shared with Rust (src/cuda.rs)
enum DType { DT_F32 = 0, DT_F16 = 1, DT_BF16 = 2 };

__device__ __forceinline__ float ldf(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float ldf(const bf16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float ldf(const half* p, int64_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void stf(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ void stf(bf16* p, int64_t i, float v) { p[i] = __float2bfloat16_rn(v); }
__device__ __forceinline__ void stf(half* p, int64_t i, float v) { p[i] = __float2half_rn(v); }

// Round a float to the precision of T (emulates PyTorch materializing a T tensor).
template <typename T> __device__ __forceinline__ float rnd(float v);
template <> __device__ __forceinline__ float rnd<float>(float v) { return v; }
template <> __device__ __forceinline__ float rnd<bf16>(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
template <> __device__ __forceinline__ float rnd<half>(float v) { return __half2float(__float2half_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Block-wide reductions; `red` must hold >= 33 floats of shared memory.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < nw ? red[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) red[32] = t;
    }
    __syncthreads();
    return red[32];
}
__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < nw ? red[lane] : 0.f;
        t = warp_max(t);
        if (lane == 0) red[32] = t;
    }
    __syncthreads();
    return red[32];
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }
// tanh that stays accurate near zero, as comfy-kitchen's quantizer computes it
__device__ __forceinline__ float stable_tanh(float x) {
    const float t = expm1f(-2.f * fabsf(x));
    return copysignf(-t / (t + 2.f), x);
}
// MOSS Image's soft-capped gated activation in one float expression
__device__ __forceinline__ float situ_f(float hidden, float gate) {
    return 4.f * stable_tanh(gate / 4.f) / (1.f + expf(-gate)) * (25.f * stable_tanh(hidden / 25.f));
}
__device__ __forceinline__ float gelu_f(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ float gelu_tanh_f(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x * x * x)));
}

// Row of the regular Hadamard block H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
// dotted with (x0..x3); same left-to-right order as comfy-kitchen so results match bit-for-bit.
__device__ __forceinline__ float h4_row(int d, float x0, float x1, float x2, float x3) {
    switch (d) {
        case 0: return x0 + x1 + x2 - x3;
        case 1: return x0 + x1 - x2 + x3;
        case 2: return x0 - x1 + x2 + x3;
        default: return -x0 + x1 + x2 + x3;
    }
}

// In-place ConvRot rotation (normalized H256 = (1/16) H4^(x4)) of every 256-wide group of a
// shared-memory row of K floats. H256 is symmetric and orthogonal, so it is its own inverse.
__device__ __forceinline__ void fht256_rows_smem(float* row, int K) {
    const int nbf = K >> 2;  // butterflies of 4 per stage
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        const int s = 1 << (2 * stage);
        for (int b = threadIdx.x; b < nbf; b += blockDim.x) {
            const int g = b >> 6, r = b & 63;
            const int i0 = (g << 8) + (r / s) * (s << 2) + (r % s);
            const float x0 = row[i0], x1 = row[i0 + s], x2 = row[i0 + 2 * s], x3 = row[i0 + 3 * s];
            row[i0] = 0.5f * h4_row(0, x0, x1, x2, x3);
            row[i0 + s] = 0.5f * h4_row(1, x0, x1, x2, x3);
            row[i0 + 2 * s] = 0.5f * h4_row(2, x0, x1, x2, x3);
            row[i0 + 3 * s] = 0.5f * h4_row(3, x0, x1, x2, x3);
        }
        __syncthreads();
    }
}

#define KCHECK() do { cudaError_t e__ = cudaGetLastError(); if (e__ != cudaSuccess) return (int)e__; } while (0)
