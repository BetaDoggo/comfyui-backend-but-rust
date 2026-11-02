// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// W6A8 requantize in one launch: raw bf16 weight -> ConvRot -> packed codes + fp8 s_rel + f32
// s_channel. comfy-kitchen's w4a8_gemm.cu, its 6-bit path.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace {

__device__ __forceinline__ float rq_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T rq_from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 rq_from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ uint32_t rq_pcg(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    uint32_t w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}
// uniform in [0,1) keyed by a global element index + seed
__device__ __forceinline__ float rq_uniform(int64_t idx, uint64_t seed) {
    uint32_t h = rq_pcg(static_cast<uint32_t>(idx) ^ static_cast<uint32_t>(seed)
                        ^ static_cast<uint32_t>(seed >> 32));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float rq_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide max, returned on every thread.
__device__ __forceinline__ float rq_block_max(float v, float* warp_max) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = rq_warp_max(v);
    if (lane == 0) warp_max[wid] = v;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = (blockDim.x + 31) >> 5;
        float t = (lane < nwarps) ? warp_max[lane] : 0.0f;
        t = rq_warp_max(t);
        if (lane == 0) warp_max[0] = t;
    }
    __syncthreads();
    return warp_max[0];
}

// Rotated row in smem, in the weight dtype, 2 elements of padding per group (bank-conflict free).
template <int LOG2G>
__device__ __forceinline__ int rq_padded_index(int i) { return i + 2 * (i >> LOG2G); }
template <int G> struct RqLog2 { static constexpr int value = G == 16 ? 4 : G == 32 ? 5 : 6; };

// int8 value of 6-bit level index i (0..62) at group step s: eager's _grid_levels, on the fly.
__device__ __forceinline__ float w6_level(int i, float s) {
    return fminf(127.0f, fmaxf(-127.0f, nearbyintf(static_cast<float>(i - 31) * s)));
}
// First level index >= t (searchsorted): closed-form estimate, fixup loops make it exact.
__device__ __forceinline__ int w6_lower_bound(float t, float s, float inv_s) {
    int i = __float2int_rn((ceilf(t) - 0.5f) * inv_s) + 31;
    i = min(max(i, 0), 63);
    while (i > 0 && w6_level(i - 1, s) >= t) --i;
    while (i < 63 && w6_level(i, s) < t) ++i;
    return i;
}
// Eager _assign_grid: nearer bracket around t (lower on ties), or an SR draw in the bracket.
template <bool SR>
__device__ __forceinline__ int w6_assign(float t, float s, float inv_s, int64_t idx, uint64_t seed) {
    const int pos = w6_lower_bound(t, s, inv_s);
    if constexpr (SR) {
        const int l = min(max(pos - 1, 0), 61);
        const float level_lo = w6_level(l, s);
        const float thr = level_lo + rq_uniform(idx, seed) * (w6_level(l + 1, s) - level_lo);
        return min(l + (t > thr ? 1 : 0), 62);
    }
    const int l = min(max(pos - 1, 0), 62), h = min(pos, 62);
    return (fabsf(t - w6_level(h, s)) < fabsf(t - w6_level(l, s))) ? h : l;
}

// One W6A8 row (uniform levels, fp8 s_rel, no scale search) from the padded smem row.
template <typename InputType, int G>
__device__ __forceinline__ void w6_quantize_row(
    const InputType* row_buf, int K, int64_t elem_base,
    int8_t* __restrict__ packed_row,   // [3K/4]: nibble plane then 2-bit plane
    uint8_t* __restrict__ s_rel_row,   // [K/G] e4m3 bits
    float* __restrict__ s_channel_out, // [1]
    float* gscale,                     // smem [K/G]
    float* warp_max,                   // smem [32]
    bool stochastic, uint64_t seed)
{
    auto src = [&](int i) { return rq_to_float(row_buf[rq_padded_index<RqLog2<G>::value>(i)]); };
    const int groups = K / G;

    // --- Phase 1: per-group step (2 ALS rounds); row amax of the decoded weight ---
    float thr_amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * G;
        float w[G];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < G; ++i) { w[i] = src(base + i); amax = fmaxf(amax, fabsf(w[i])); }
        float gs = fmaxf(amax / 31.0f, 1e-8f);
        float inv = 1.0f / gs;
        #pragma unroll
        for (int it = 0; it < 2; ++it) {       // eager _ALS_ITERS
            float num = 0.0f, den = 0.0f;
            #pragma unroll
            for (int i = 0; i < G; ++i) {
                const float q = fminf(31.0f, fmaxf(-31.0f, nearbyintf(w[i] * inv)));
                num += w[i] * q; den += q * q;
            }
            gs = fmaxf(num / fmaxf(den, 1e-8f), 1e-8f);
            inv = 1.0f / gs;
        }
        gscale[g] = gs;
        #pragma unroll
        for (int i = 0; i < G; ++i)
            thr_amax = fmaxf(thr_amax, fabsf(fminf(31.0f, fmaxf(-31.0f, nearbyintf(w[i] * inv))) * gs));
    }
    const float sc = fmaxf(rq_block_max(thr_amax, warp_max) / 127.0f, 1e-8f);
    const float inv_sc = 1.0f / sc;
    if (threadIdx.x == 0) *s_channel_out = sc;

    // --- Phase 2: fp8 s_rel -> grid assignment -> code q+32 -> two-plane pack ---
    int8_t* hi_plane = packed_row + K / 2;
    auto phase2 = [&](auto sr_tag) {
        constexpr bool SR = decltype(sr_tag)::value;
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint8_t srel_bits = __nv_cvt_float_to_fp8(gscale[g] / sc, __NV_SATFINITE, __NV_E4M3);
            s_rel_row[g] = srel_bits;
            const float srel_r = __half2float(__nv_cvt_fp8_to_halfraw(srel_bits, __NV_E4M3));
            const float inv_srel = 1.0f / fmaxf(srel_r, 1e-30f);
            const int base = g * G;
            int8_t* lo = packed_row + g * (G / 2);
            int8_t* hi = hi_plane + g * (G / 4);
            #pragma unroll
            for (int p = 0; p < G / 4; ++p) {
                const int c = 4 * p;
                int u[4];
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                    u[k] = w6_assign<SR>(src(base + c + k) * inv_sc, srel_r, inv_srel, elem_base + base + c + k, seed) + 1;
                lo[2 * p] = static_cast<int8_t>((u[0] & 0xF) | ((u[1] & 0xF) << 4));
                lo[2 * p + 1] = static_cast<int8_t>((u[2] & 0xF) | ((u[3] & 0xF) << 4));
                hi[p] = static_cast<int8_t>(((u[0] >> 4) & 3) | (((u[1] >> 4) & 3) << 2) | (((u[2] >> 4) & 3) << 4) | (((u[3] >> 4) & 3) << 6));
            }
        }
    };
    if (stochastic) phase2(std::true_type{}); else phase2(std::false_type{});
}

// Row k of H4, same association as int8_linear.cu's h4_row_dot (bit-identical).
__device__ __forceinline__ float rq_h4_row(int k, float x0, float x1, float x2, float x3) {
    return 0.5f * ((((k == 3 ? -x0 : x0) + (k == 2 ? -x1 : x1)) + (k == 1 ? -x2 : x2)) + (k == 0 ? -x3 : x3));
}

// Radix-4 stage pairing register bit RB with the lane bit selected by LX.
template <int RB, int LX>
__device__ __forceinline__ void rq_fht_mixed(float (&v)[32], int lane_bit) {
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        if (r & (1 << RB)) continue;
        const int r1 = r | (1 << RB);
        const float a0 = v[r], a1 = v[r1];
        const float p0 = __shfl_xor_sync(0xffffffffu, a0, LX);
        const float p1 = __shfl_xor_sync(0xffffffffu, a1, LX);
        const float x0 = lane_bit ? p0 : a0, x1 = lane_bit ? p1 : a1;
        const float x2 = lane_bit ? a0 : p0, x3 = lane_bit ? a1 : p1;
        v[r] = rq_h4_row(2 * lane_bit, x0, x1, x2, x3);
        v[r1] = rq_h4_row(2 * lane_bit + 1, x0, x1, x2, x3);
    }
}

// ConvRot256 of one group in registers: 8 lanes x 32 values. Index bits 0,1,2,4,6 are register
// bits, 3,5,7 lane bits. Same stage order and arithmetic as rotate_int8_convrot_weight; the
// rounded result goes to the padded smem row.
template <typename InputType, int LOG2G>
__device__ __forceinline__ void rq_rotate_group_regs(const InputType* __restrict__ xg, InputType* row_buf,
                                                     int e_base, int lane8, bool store)
{
    const int b3 = lane8 & 1, b5 = (lane8 >> 1) & 1, b7 = lane8 >> 2;
    float v[32];   // r = b0 | b1<<1 | b2<<2 | b4<<3 | b6<<4
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int off = 8 * b3 + 16 * (q & 1) + 32 * b5 + 64 * (q >> 1) + 128 * b7;
        const uint4 raw = *reinterpret_cast<const uint4*>(xg + off);   // 8 x 16-bit
        const InputType* e = reinterpret_cast<const InputType*>(&raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[q * 8 + i] = rq_to_float(e[i]);
    }
    #pragma unroll
    for (int base = 0; base < 32; base += 4) {       // stage 0: bits (0,1), register-only
        const float x0 = v[base], x1 = v[base + 1], x2 = v[base + 2], x3 = v[base + 3];
        v[base] = 0.5f * (x0 + x1 + x2 - x3);
        v[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        v[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        v[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    }
    rq_fht_mixed<2, 1>(v, b3);                        // bits (2,3)
    rq_fht_mixed<3, 2>(v, b5);                        // bits (4,5)
    rq_fht_mixed<4, 4>(v, b7);                        // bits (6,7)
    if (!store) return;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int off = 8 * b3 + 16 * (q & 1) + 32 * b5 + 64 * (q >> 1) + 128 * b7;
        InputType* dst = row_buf + rq_padded_index<LOG2G>(e_base + off);   // even index: 4-byte pairs
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            InputType pair[2] = {rq_from_float<InputType>(v[q * 8 + i]), rq_from_float<InputType>(v[q * 8 + i + 1])};
            *reinterpret_cast<uint32_t*>(dst + i) = *reinterpret_cast<const uint32_t*>(pair);
        }
    }
}

template <typename InputType, int G>
__global__ void __launch_bounds__(256) requant_kernel(const InputType* __restrict__ weight, int8_t* __restrict__ packed, uint8_t* __restrict__ s_rel, float* __restrict__ s_channel,
    int K, bool stochastic, uint64_t seed) {
    constexpr int LOG2G = RqLog2<G>::value;
    __shared__ float warp_max[32];
    extern __shared__ float smem[];
    float* gscale = smem;
    InputType* row_buf = reinterpret_cast<InputType*>(gscale + (K >> LOG2G));
    const int64_t row = blockIdx.x;
    const InputType* wrow = weight + row * K;
    const int clusters = blockDim.x / 8, cluster = threadIdx.x / 8, lane8 = threadIdx.x % 8;
    const int n_groups = K / 256;
    for (int it = 0; it < (n_groups + clusters - 1) / clusters; ++it) {
        const int g = it * clusters + cluster;
        const bool active = g < n_groups;
        rq_rotate_group_regs<InputType, LOG2G>(wrow + (active ? g : 0) * 256, row_buf, g * 256, lane8, active);
    }
    __syncthreads();
    w6_quantize_row<InputType, G>(row_buf, K, row * K, packed + row * (K * 3 / 4), s_rel + row * (K / G), s_channel + row, gscale, warp_max, stochastic, seed);
}

}  // namespace

// weight bf16 [N, K] -> packed [N, 3K/4], s_rel [N, K/32], s_channel [N]
extern "C" int gpu_w6_requant(const void* weight, void* packed, void* s_rel, void* s_channel, int64_t N, int K, uint64_t seed, void* stream) {
    if (K % 256) return -1;
    using B = __nv_bfloat16;
    const size_t shmem = (K / 32) * sizeof(float) + (static_cast<size_t>(K) + 2 * (K / 32)) * 2;
    auto kern = requant_kernel<B, 32>;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem)) != cudaSuccess) return -2;
    kern<<<static_cast<unsigned>(N), 256, shmem, (cudaStream_t)stream>>>(static_cast<const B*>(weight), static_cast<int8_t*>(packed), static_cast<uint8_t*>(s_rel),
        static_cast<float*>(s_channel), K, true, seed);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
