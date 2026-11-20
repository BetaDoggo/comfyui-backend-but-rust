// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Optimized INT8 per-thread quantization for Q and K (SageAttention).
//
// All tensors assumed contiguous [B, H, L, D] (HND layout).
// Two-kernel launch: Q and K in separate kernels for better I-cache
// utilization, with a fused fallback for non-standard tile configs.
//   Q path: warp-per-group, single-pass, vectorized float2 loads / int32
//   stores. K path: warp-per-group, single-pass, vectorized loads / stores.
//   No __syncthreads anywhere – pure warp-level reductions.
//
// Block / warp tile sizes and alignment are template parameters so the
// compiler can constant-fold address arithmetic (divisions, modulos) and
// eliminate dead scalar-fallback code when C is a multiple of 4.
//
#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

using comfy::quant_int8_rcp;
using comfy::store4_i8;
using comfy::warp_reduce_fmax;

namespace {

template <typename T>
struct VectorLoader4;

template <>
struct VectorLoader4<half> {
  __forceinline__ __device__ static void load(const half* ptr, float* out) {
    float2 raw = __ldg(reinterpret_cast<const float2 *>(ptr));
    const half *vals = reinterpret_cast<const half *>(&raw);
    out[0] = static_cast<float>(vals[0]);
    out[1] = static_cast<float>(vals[1]);
    out[2] = static_cast<float>(vals[2]);
    out[3] = static_cast<float>(vals[3]);
  }
};

template <>
struct VectorLoader4<nv_bfloat16> {
  __forceinline__ __device__ static void load(const nv_bfloat16* ptr, float* out) {
    float2 raw = __ldg(reinterpret_cast<const float2 *>(ptr));
    const nv_bfloat16 *vals = reinterpret_cast<const nv_bfloat16 *>(&raw);
    out[0] = static_cast<float>(vals[0]);
    out[1] = static_cast<float>(vals[1]);
    out[2] = static_cast<float>(vals[2]);
    out[3] = static_cast<float>(vals[3]);
  }
};

template <>
struct VectorLoader4<float> {
  __forceinline__ __device__ static void load(const float* ptr, float* out) {
    float4 raw = __ldg(reinterpret_cast<const float4 *>(ptr));
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
  }
};

__forceinline__ __device__ void convrot4(float *values) {
  const float x0 = values[0];
  const float x1 = values[1];
  const float x2 = values[2];
  const float x3 = values[3];
  const float a0 = x0 + x1;
  const float a1 = x0 - x1;
  const float a2 = x2 + x3;
  const float a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}

// A fixed random diagonal makes the following Hadamard a randomized
// orthogonal transform instead of aligning every row to the same structured
// basis. Q and K use the same signs, so their exact dot product is unchanged.
// Flip the IEEE sign bit directly: this is exact and avoids an FP multiply.
__forceinline__ __device__ void apply_convrot_sign128(float *values,
                                                      const int lane) {
  constexpr uint32_t signs_0 = 0x1035997bu;
  constexpr uint32_t signs_1 = 0x8087f5eeu;
  constexpr uint32_t signs_2 = 0xee2e4e1au;
  constexpr uint32_t signs_3 = 0x71132418u;
  const uint32_t signs =
      lane < 8    ? signs_0
      : lane < 16 ? signs_1
      : lane < 24 ? signs_2
                  : signs_3;
  const int shift = (lane & 7) * 4;
#pragma unroll
  for (int channel = 0; channel < 4; ++channel) {
    const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
    values[channel] =
        __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
  }
}

// Apply a normalized Walsh-Hadamard H64 to one 64-channel half-warp group.
// H4 covers the four adjacent channels owned by each lane; four shuffle
// butterflies cover the remaining 16-lane dimension. This uses half as many
// shuffles as evaluating H4 x H4 x H4 directly.
__forceinline__ __device__ void convrot64(float *values) {
  convrot4(values);
  const int half_lane = threadIdx.x & 15;
  const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;

#pragma unroll
  for (int bit = 1; bit < 16; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(mask, values[c], bit, 16);
      values[c] =
          (half_lane & bit) ? other - values[c] : values[c] + other;
    }
  }

#pragma unroll
  for (int c = 0; c < 4; ++c)
    values[c] *= 0.25f;
}

// Apply a normalized Walsh-Hadamard H128 across all four-channel warp groups.
__forceinline__ __device__ void convrot128_plain(float *values) {
  convrot4(values);
  const int lane = threadIdx.x & 31;

#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(0xffffffffu, values[c], bit);
      values[c] = (lane & bit) ? other - values[c] : values[c] + other;
    }
  }

#pragma unroll
  for (int c = 0; c < 4; ++c)
    values[c] *= 0.1767766952966369f;
}

__forceinline__ __device__ void convrot128(float *values) {
  apply_convrot_sign128(values, threadIdx.x & 31);
  convrot128_plain(values);
}

// ---------------------------------------------------------------------------
// Optional per-head RMSNorm, weight and RoPE ahead of quantization, for head
// dim 128 in bf16. It reproduces gpu_qk_norm_rope (csrc/kernels.cu) bit for
// bit, so the quantizer sees exactly the bf16 values that pass would have
// written: the same per-lane sums in the same order, the same butterfly, the
// same roundings.
//
// NM picks the variant: 0 none, 1 interleaved-pair RoPE (rope_mode 1), 2
// split-half RoPE (rope_mode 2), 3 no RoPE, 4 no RoPE with rope_mode 2's
// lane order for the sum of squares.
// ---------------------------------------------------------------------------
struct QkNormRope {
  const float *qw;   // [128] or null
  const float *kw;   // [128] or null
  const float *rope; // [L][64] x (cos, sin), 16-byte aligned; null for NM 3, 4
  float eps;
};

template <int NM> constexpr bool kHasRope = NM == 1 || NM == 2;
template <int NM> constexpr bool kSplitHalves = NM == 2 || NM == 4;

__forceinline__ __device__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// This lane's four weights (channels 4 * lane ..), or ones without a weight:
// multiplying by one is exact.
__forceinline__ __device__ float4 norm_weights(const float *w) {
  if (!w)
    return make_float4(1.f, 1.f, 1.f, 1.f);
  const int ch = (threadIdx.x & 31) << 2;
  return make_float4(__ldg(&w[ch]), __ldg(&w[ch + 1]), __ldg(&w[ch + 2]),
                     __ldg(&w[ch + 3]));
}

// The rope table entries (cos, sin pairs) this lane needs for `row`.
template <int NM>
__forceinline__ __device__ void rope_entries(const QkNormRope &p, const int row,
                                             float4 &r0, float4 &r1) {
  const int lane = threadIdx.x & 31;
  const float4 *r = reinterpret_cast<const float4 *>(p.rope + (int64_t)row * 128);
  if constexpr (NM == 1) {
    r0 = __ldg(&r[lane]);
  } else if constexpr (NM == 2) {
    r0 = __ldg(&r[2 * (lane & 15)]);
    r1 = __ldg(&r[2 * (lane & 15) + 1]);
  }
}

// values: channels 4 * lane .. 4 * lane + 3 of one row; w from norm_weights,
// r0 and r1 from rope_entries. Every lane of the warp must take part, so call
// it outside divergent branches: a row of zeros comes out as zeros.
template <int NM>
__forceinline__ __device__ void norm_rope128(float *values, const float4 w,
                                             const float eps, const float4 r0,
                                             const float4 r1) {
  const int lane = threadIdx.x & 31;
  float x[4];
  if constexpr (kSplitHalves<NM>) {
    // gpu_qk_norm_rope's lane L holds channels 2L, 2L + 1, 64 + 2L, 65 + 2L
    const int src = lane >> 1;
    const bool hi = lane & 1;
    float a[4], b[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      a[c] = __shfl_sync(0xffffffffu, values[c], src);
      b[c] = __shfl_sync(0xffffffffu, values[c], src + 16);
    }
    x[0] = hi ? a[2] : a[0];
    x[1] = hi ? a[3] : a[1];
    x[2] = hi ? b[2] : b[0];
    x[3] = hi ? b[3] : b[1];
  } else {
#pragma unroll
    for (int c = 0; c < 4; ++c)
      x[c] = values[c];
  }
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c)
    ss += x[c] * x[c];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rstd = rsqrtf(ss / 128.f + eps);
  values[0] = round_bf16(values[0] * rstd * w.x);
  values[1] = round_bf16(values[1] * rstd * w.y);
  values[2] = round_bf16(values[2] * rstd * w.z);
  values[3] = round_bf16(values[3] * rstd * w.w);
  if constexpr (NM == 1) {
    const float cs[2][2] = {{r0.x, r0.y}, {r0.z, r0.w}};
#pragma unroll
    for (int pi = 0; pi < 2; ++pi) {
      const float c = cs[pi][0], s = cs[pi][1];
      const float a = values[2 * pi], b = values[2 * pi + 1];
      values[2 * pi] = c * a - s * b;
      values[2 * pi + 1] = s * a + c * b;
    }
  } else if constexpr (NM == 2) {
    const float cs[4][2] = {{r0.x, r0.y}, {r0.z, r0.w}, {r1.x, r1.y}, {r1.z, r1.w}};
    const bool first = lane < 16;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float other = __shfl_xor_sync(0xffffffffu, values[k], 16);
      const float c = cs[k][0], s = cs[k][1];
      if (first) {
        const float a = values[k], b = other;
        values[k] = a * c - b * s;
      } else {
        const float a = other, b = values[k];
        values[k] = b * c + a * s;
      }
    }
  }
  if constexpr (kHasRope<NM>) {
#pragma unroll
    for (int c = 0; c < 4; ++c)
      values[c] = round_bf16(values[c]);
  }
}

template <int NM>
__forceinline__ __device__ void norm_rope128(float *values, const float4 w,
                                             const QkNormRope &p,
                                             const int row) {
  float4 r0 = make_float4(0.f, 0.f, 0.f, 0.f), r1 = r0;
  if constexpr (kHasRope<NM>)
    rope_entries<NM>(p, row, r0, r1);
  norm_rope128<NM>(values, w, p.eps, r0, r1);
}

// ---------------------------------------------------------------------------
// Q processing device function
// ---------------------------------------------------------------------------
#pragma nv_diag_suppress 1056
template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__forceinline__ __device__ void
process_q(const T *__restrict__ in, int8_t *__restrict__ out,
          float *__restrict__ sc_buf, const int oblk, const int L,
          const int C, const int64_t stride_n) {
  constexpr int NSUB = BLKQ / WARPQ;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int otld = wid * 2 + g;
    const int base = (oblk / NSUB) * BLKQ + (oblk % NSUB) * WARPQ + otld;

    float v[CHANNEL_TILES * NR * 4];
    float mx = 0.f;

#pragma unroll
    for (int i = 0; i < CHANNEL_TILES * NR * 4; ++i)
      v[i] = 0.f;

#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int tile_base = tile * NR * 4;
      if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
            VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                                 fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
          }
        }
      } else if (ch < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              v[vi + c] =
                  (ch + c < C)
                      ? static_cast<float>(
                            __ldg(&in[(int64_t)n * stride_n + ch + c]))
                      : 0.f;
              mx = fmaxf(mx, fabsf(v[vi + c]));
            }
          }
        }
      }
    }

    if constexpr (ROTATION != 0) {
#pragma unroll
      for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          if constexpr (ROTATION == 128)
            convrot128(&v[(tile * NR + j) * 4]);
          else if constexpr (ROTATION == 129)
            convrot128_plain(&v[(tile * NR + j) * 4]);
          else if constexpr (ROTATION == 64)
            convrot64(&v[(tile * NR + j) * 4]);
          else
            convrot4(&v[(tile * NR + j) * 4]);
        }
      }
      mx = 0.f;
#pragma unroll
      for (int j = 0; j < CHANNEL_TILES * NR * 4; ++j)
        mx = fmaxf(mx, fabsf(v[j]));
    }

    mx = warp_reduce_fmax(mx);
    const float sc = mx / 127.f + 1e-7f;
    const float inv_sc = 1.f / sc;

    if (lane == 0)
      sc_buf[oblk * 8 + otld] = sc;

#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int tile_base = tile * NR * 4;
      if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
            store4_i8(&out[(int64_t)n * C + ch],
                      quant_int8_rcp(v[vi], inv_sc),
                      quant_int8_rcp(v[vi + 1], inv_sc),
                      quant_int8_rcp(v[vi + 2], inv_sc),
                      quant_int8_rcp(v[vi + 3], inv_sc));
          }
        }
      } else if (ch < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (ch + c < C)
                out[(int64_t)n * C + ch + c] =
                    quant_int8_rcp(v[vi + c], inv_sc);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// K processing device function
//
// When anchor_index is nonnegative, subtracts that key vector from every key.
// The shift is exactly softmax-invariant and happens before abs-max reduction,
// rotation, and quantization.
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int CHANNEL_TILES, int ROTATION,
          bool ALIGNED4>
__forceinline__ __device__ void
process_k(const T *__restrict__ in, int8_t *__restrict__ out,
          float *__restrict__ sc_buf, const int oblk, const int L, const int C,
          const int anchor_index, const int64_t stride_n) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int otld = wid;

  float bias[CHANNEL_TILES * 4];
#pragma unroll
  for (int i = 0; i < CHANNEL_TILES * 4; ++i)
    bias[i] = 0.f;

  if (anchor_index >= 0) {
#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int64_t anchor_offset =
          (int64_t)anchor_index * stride_n + ch;
      if (ALIGNED4 || ch + 3 < C) {
        VectorLoader4<T>::load(&in[anchor_offset], &bias[tile * 4]);
      } else if (ch < C) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          bias[tile * 4 + c] =
              (ch + c < C)
                  ? static_cast<float>(__ldg(&in[anchor_offset + c]))
                  : 0.f;
      }
    }
  }

  float v[CHANNEL_TILES * 2 * NL * 4];
  float mx = 0.f;

#pragma unroll
  for (int i = 0; i < CHANNEL_TILES * 2 * NL * 4; ++i)
    v[i] = 0.f;

#pragma unroll
  for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
    const int ch = tile * 128 + (lane << 2);
    const int tile_base = tile * 2 * NL * 4;
    if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
            VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
#pragma unroll
            for (int c = 0; c < 4; ++c)
              v[vi + c] -= bias[tile * 4 + c];
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                                 fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
          }
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              v[vi + c] =
                  (ch + c < C)
                      ? static_cast<float>(
                            __ldg(&in[(int64_t)n * stride_n + ch + c])) -
                            bias[tile * 4 + c]
                      : 0.f;
              mx = fmaxf(mx, fabsf(v[vi + c]));
            }
          }
        }
      }
    }
  }

  if constexpr (ROTATION != 0) {
#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
#pragma unroll
      for (int j = 0; j < 2 * NL; ++j) {
        if constexpr (ROTATION == 128)
          convrot128(&v[(tile * 2 * NL + j) * 4]);
        else if constexpr (ROTATION == 129)
          convrot128_plain(&v[(tile * 2 * NL + j) * 4]);
        else if constexpr (ROTATION == 64)
          convrot64(&v[(tile * 2 * NL + j) * 4]);
        else
          convrot4(&v[(tile * 2 * NL + j) * 4]);
      }
    }
    mx = 0.f;
#pragma unroll
    for (int j = 0; j < CHANNEL_TILES * 2 * NL * 4; ++j)
      mx = fmaxf(mx, fabsf(v[j]));
  }

  mx = warp_reduce_fmax(mx);
  const float sc = mx / 127.f + 1e-7f;
  const float inv_sc = 1.f / sc;

  if (lane == 0)
    sc_buf[oblk * 4 + otld] = sc;

#pragma unroll
  for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
    const int ch = tile * 128 + (lane << 2);
    const int tile_base = tile * 2 * NL * 4;
    if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
            store4_i8(&out[(int64_t)n * C + ch],
                      quant_int8_rcp(v[vi], inv_sc),
                      quant_int8_rcp(v[vi + 1], inv_sc),
                      quant_int8_rcp(v[vi + 2], inv_sc),
                      quant_int8_rcp(v[vi + 3], inv_sc));
          }
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (ch + c < C)
                out[(int64_t)n * C + ch + c] =
                    quant_int8_rcp(v[vi + c], inv_sc);
            }
          }
        }
      }
    }
  }
}
#pragma nv_diag_default 1056

// ---------------------------------------------------------------------------
// Model-independent K stabilization detector
//
// Samples nine evenly spaced keys per (batch, head), selects the sampled key
// that minimizes residual energy, and enables centering only when that anchor
// reduces sampled energy without increasing sampled abs-max by more than
// 12.5%. The output is an absolute sequence index, or -1 when the original K
// range is preferable. No host synchronization is required.
// ---------------------------------------------------------------------------
constexpr int CENTER_DETECT_THREADS = 128;
constexpr int CENTER_SAMPLES = 9;
constexpr int CENTER_MAX_CHANNELS = 256;

template <typename T, int NM = 0>
__global__ __launch_bounds__(CENTER_DETECT_THREADS) void detect_k_anchor(
    const T *__restrict__ k_in, int *__restrict__ anchor_indices, const int Lk,
    const int C, const int H_kv, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n,
    const QkNormRope norm = QkNormRope{}) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t bh_offset =
      (int64_t)b * stride_b + (int64_t)h * stride_h;

  __shared__ float samples[CENTER_SAMPLES * CENTER_MAX_CHANNELS];
  __shared__ float warp_original_energy[4];
  __shared__ float warp_original_max[4];
  __shared__ float warp_candidate_distance[CENTER_SAMPLES][4];
  __shared__ float warp_best_energy[4];
  __shared__ float warp_best_max[4];
  __shared__ int selected_candidate;

  for (int index = tid; index < CENTER_SAMPLES * C;
       index += CENTER_DETECT_THREADS) {
    const int sample = index / C;
    const int channel = index - sample * C;
    const int row = sample * (Lk - 1) / (CENTER_SAMPLES - 1);
    samples[index] = static_cast<float>(
        __ldg(&k_in[bh_offset + (int64_t)row * stride_n + channel]));
  }
  __syncthreads();
  if constexpr (NM != 0) {
    // C is 128: warp w normalizes sampled rows w, w + 4, w + 8
    const float4 w = norm_weights(norm.kw);
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
      const int sample = min(warp + pass * 4, CENTER_SAMPLES - 1);
      float *row_values = &samples[sample * C + (lane << 2)];
      float v[4] = {row_values[0], row_values[1], row_values[2], row_values[3]};
      norm_rope128<NM>(v, w, norm, sample * (Lk - 1) / (CENTER_SAMPLES - 1));
      __syncwarp();
      if (warp + pass * 4 < CENTER_SAMPLES) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          row_values[c] = v[c];
      }
    }
    __syncthreads();
  }

  float original_energy = 0.f;
  float original_max = 0.f;
  float candidate_distance[CENTER_SAMPLES];
#pragma unroll
  for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
    candidate_distance[candidate] = 0.f;
  }

  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    float channel_sum = 0.f;
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float value = samples[sample * C + channel];
      original_energy = fmaf(value, value, original_energy);
      original_max = fmaxf(original_max, fabsf(value));
      channel_sum += value;
    }
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      const float distance =
          CENTER_SAMPLES * samples[candidate * C + channel] - channel_sum;
      candidate_distance[candidate] =
          fmaf(distance, distance, candidate_distance[candidate]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    original_energy +=
        __shfl_down_sync(0xffffffffu, original_energy, offset);
    original_max = fmaxf(
        original_max, __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      candidate_distance[candidate] += __shfl_down_sync(
          0xffffffffu, candidate_distance[candidate], offset);
    }
  }

  if (lane == 0) {
    warp_original_energy[warp] = original_energy;
    warp_original_max[warp] = original_max;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      warp_candidate_distance[candidate][warp] =
          candidate_distance[candidate];
    }
  }
  __syncthreads();

  if (tid == 0) {
    int best_candidate = 0;
    float best_distance = 3.402823466e+38F;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      float distance = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        distance += warp_candidate_distance[candidate][w];
      }
      if (distance < best_distance) {
        best_candidate = candidate;
        best_distance = distance;
      }
    }
    selected_candidate = best_candidate;
  }
  __syncthreads();

  float best_energy = 0.f;
  float best_max = 0.f;
  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    const float anchor = samples[selected_candidate * C + channel];
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float residual = samples[sample * C + channel] - anchor;
      best_energy = fmaf(residual, residual, best_energy);
      best_max = fmaxf(best_max, fabsf(residual));
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
    best_max = fmaxf(
        best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
  }
  if (lane == 0) {
    warp_best_energy[warp] = best_energy;
    warp_best_max[warp] = best_max;
  }
  __syncthreads();

  if (tid == 0) {
    float total_original_energy = 0.f;
    float total_original_max = 0.f;
    float total_best_energy = 0.f;
    float total_best_max = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      total_original_energy += warp_original_energy[w];
      total_original_max =
          fmaxf(total_original_max, warp_original_max[w]);
      total_best_energy += warp_best_energy[w];
      total_best_max = fmaxf(total_best_max, warp_best_max[w]);
    }

    const bool improves_range =
        total_best_energy < total_original_energy &&
        total_best_max <= total_original_max * 1.125f;
    anchor_indices[b * H_kv + h] =
        improves_range
            ? selected_candidate * (Lk - 1) / (CENTER_SAMPLES - 1)
            : -1;
  }
}

// ---------------------------------------------------------------------------
// Standalone Q kernel
// ---------------------------------------------------------------------------
template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_q_kernel(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const int Lq, const int C, const int H_q,
    const int q_sc_per_h, const int64_t stride_b, const int64_t stride_h,
    const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
  const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
  process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
      q_in + in_bh, q_out + out_bh, q_sb + sbh, oblk, Lq, C, stride_n);
}

// ---------------------------------------------------------------------------
// Standalone K kernel
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int CHANNEL_TILES, int ROTATION,
          bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_k_kernel(
    const T *__restrict__ k_in, int8_t *__restrict__ k_out,
    float *__restrict__ k_sb, const int *__restrict__ anchor_indices,
    const int Lk, const int C,
    const int H_kv, const int k_sc_per_h, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
  const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
  const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
  process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
      k_in + in_bh, k_out + out_bh, k_sb + sbh, oblk, Lk, C, anchor_index,
      stride_n);
}

// ---------------------------------------------------------------------------
// Q for head dim 128 with one row group per warp (8 warps per block): the same
// output as process_q with its two groups per warp, with more warps in flight.
// ---------------------------------------------------------------------------
template <typename T, int ROTATION, int NM>
__global__ __launch_bounds__(256) void quant_q128_kernel(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const int Lq, const int H_q,
    const int q_sc_per_h, const int64_t stride_b, const int64_t stride_h,
    const int64_t stride_n, const QkNormRope norm) {
  constexpr int NR = 4;
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int otld = threadIdx.x >> 5;
  const int ch = lane << 2;
  const int base = (oblk >> 2) * 128 + (oblk & 3) * 32 + otld;
  const T *in = q_in + (int64_t)b * stride_b + (int64_t)h * stride_h;
  int8_t *out = q_out + ((int64_t)b * H_q + h) * Lq * 128;
  float *sc_buf = q_sb + ((int64_t)b * H_q + h) * q_sc_per_h;
  float4 wq;
  if constexpr (NM != 0)
    wq = norm_weights(norm.qw);
  float v[NR * 4];
#pragma unroll
  for (int i = 0; i < NR * 4; ++i)
    v[i] = 0.f;
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    const int n = base + j * 8;
    if (n < Lq)
      VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[j * 4]);
    if constexpr (NM != 0)
      norm_rope128<NM>(&v[j * 4], wq, norm, min(n, Lq - 1));
  }
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    if constexpr (ROTATION == 128)
      convrot128(&v[j * 4]);
    else if constexpr (ROTATION == 129)
      convrot128_plain(&v[j * 4]);
    else if constexpr (ROTATION == 64)
      convrot64(&v[j * 4]);
    else
      convrot4(&v[j * 4]);
  }
  float mx = 0.f;
#pragma unroll
  for (int i = 0; i < NR * 4; ++i)
    mx = fmaxf(mx, fabsf(v[i]));
  mx = warp_reduce_fmax(mx);
  const float sc = mx / 127.f + 1e-7f;
  const float inv_sc = 1.f / sc;
  if (lane == 0)
    sc_buf[oblk * 8 + otld] = sc;
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    const int n = base + j * 8;
    if (n < Lq)
      store4_i8(&out[(int64_t)n * 128 + ch], quant_int8_rcp(v[j * 4], inv_sc),
                quant_int8_rcp(v[j * 4 + 1], inv_sc),
                quant_int8_rcp(v[j * 4 + 2], inv_sc),
                quant_int8_rcp(v[j * 4 + 3], inv_sc));
  }
}

// ---------------------------------------------------------------------------
// K for head dim 128 with four warps per row group (16 warps per block): each
// warp holds a quarter of process_k's rows, so nothing spills, and the group's
// abs-max is combined through shared memory. The same output as process_k.
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int ROTATION, int NM>
__global__ __launch_bounds__(512) void quant_k128_kernel(
    const T *__restrict__ k_in, int8_t *__restrict__ k_out,
    float *__restrict__ k_sb, const int *__restrict__ anchor_indices,
    const int L, const int H_kv, const int k_sc_per_h,
    const int64_t stride_b, const int64_t stride_h, const int64_t stride_n,
    const QkNormRope norm) {
  constexpr int C = 128, SPLIT = 4, NJ = NL / SPLIT;
  static_assert(NL % SPLIT == 0, "rows must split evenly");
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int otld = wid / SPLIT, part = wid % SPLIT;
  const T *in = k_in + (int64_t)b * stride_b + (int64_t)h * stride_h;
  int8_t *out = k_out + ((int64_t)b * H_kv + h) * L * C;
  float *sc_buf = k_sb + ((int64_t)b * H_kv + h) * k_sc_per_h;
  const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
  const int ch = lane << 2;
  float4 wk;
  if constexpr (NM != 0)
    wk = norm_weights(norm.kw);

  float bias[4] = {0.f, 0.f, 0.f, 0.f};
  if (anchor_index >= 0)
    VectorLoader4<T>::load(&in[(int64_t)anchor_index * stride_n + ch], bias);
  if constexpr (NM != 0)
    norm_rope128<NM>(bias, wk, norm, max(anchor_index, 0));

  float v[NJ * 2 * 4];
#pragma unroll
  for (int i = 0; i < NJ * 2 * 4; ++i)
    v[i] = 0.f;
#pragma unroll
  for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int n = oblk * WARPK + (part + jj * SPLIT) * 8 + otld * 2 + p;
      const int vi = (jj * 2 + p) * 4;
      if (n < L)
        VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
      if constexpr (NM != 0)
        norm_rope128<NM>(&v[vi], wk, norm, min(n, L - 1));
      if (n < L) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          v[vi + c] -= bias[c];
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NJ * 2; ++j) {
    if constexpr (ROTATION == 128)
      convrot128(&v[j * 4]);
    else if constexpr (ROTATION == 129)
      convrot128_plain(&v[j * 4]);
    else if constexpr (ROTATION == 64)
      convrot64(&v[j * 4]);
    else
      convrot4(&v[j * 4]);
  }
  float mx = 0.f;
#pragma unroll
  for (int j = 0; j < NJ * 2 * 4; ++j)
    mx = fmaxf(mx, fabsf(v[j]));
  mx = warp_reduce_fmax(mx);
  __shared__ float part_max[16];
  if (lane == 0)
    part_max[wid] = mx;
  __syncthreads();
  mx = fmaxf(fmaxf(part_max[otld * SPLIT], part_max[otld * SPLIT + 1]),
             fmaxf(part_max[otld * SPLIT + 2], part_max[otld * SPLIT + 3]));
  const float sc = mx / 127.f + 1e-7f;
  const float inv_sc = 1.f / sc;
  if (part == 0 && lane == 0)
    sc_buf[oblk * 4 + otld] = sc;
#pragma unroll
  for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int n = oblk * WARPK + (part + jj * SPLIT) * 8 + otld * 2 + p;
      const int vi = (jj * 2 + p) * 4;
      if (n < L)
        store4_i8(&out[(int64_t)n * C + ch], quant_int8_rcp(v[vi], inv_sc),
                  quant_int8_rcp(v[vi + 1], inv_sc),
                  quant_int8_rcp(v[vi + 2], inv_sc),
                  quant_int8_rcp(v[vi + 3], inv_sc));
    }
  }
}

// ---------------------------------------------------------------------------
// Fused Q+K kernel – fallback for non-standard tile configs.
// blockIdx.x < q_oblk_count  →  Q path
// blockIdx.x >= q_oblk_count →  K path
// ---------------------------------------------------------------------------
template <typename T, int NR, int NL, int BLKQ, int WARPQ, int BLKK, int WARPK,
          int CHANNEL_TILES, int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 3) void quant_qk_fused(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const T *__restrict__ k_in,
    int8_t *__restrict__ k_out, float *__restrict__ k_sb,
    const int *__restrict__ anchor_indices,
    const int Lq, const int Lk, const int C, const int q_oblk_count,
    const int H_q, const int H_kv, const int q_sc_per_h, const int k_sc_per_h,
    const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_n, const int64_t k_stride_b,
    const int64_t k_stride_h, const int64_t k_stride_n) {
  const int h = blockIdx.y, b = blockIdx.z;

  if (blockIdx.x < (unsigned)q_oblk_count) {
    if (h >= H_q)
      return;
    const int64_t in_bh = (int64_t)b * q_stride_b + (int64_t)h * q_stride_h;
    const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
    const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
    process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
        q_in + in_bh, q_out + out_bh, q_sb + sbh, blockIdx.x, Lq, C,
        q_stride_n);
  } else {
    if (h >= H_kv)
      return;
    const int64_t in_bh = (int64_t)b * k_stride_b + (int64_t)h * k_stride_h;
    const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
    const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
    const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
    process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
        k_in + in_bh, k_out + out_bh, k_sb + sbh,
        (int)blockIdx.x - q_oblk_count, Lk, C, anchor_index, k_stride_n);
  }
}

// Head dim 128: Q in quant_q128_kernel, K in quant_k128_kernel. NMK: K's own mode (0 when K is normalized already).
template <typename T, int NL, int BK, int ROT, int NM, int NMK = NM>
void launch_qk128(const void *q, void *q_int8, void *q_scale, const void *k,
                  void *k_int8, void *k_scale, int *anchors, int B, int H_q,
                  int Lq, int H_kv, int Lk, int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
                  int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
                  const QkNormRope &norm, cudaStream_t stream) {
  const int q_oblk = (Lq + 127) / 128 * 4;
  const int k_oblk = (Lk + BK - 1) / BK;
  quant_q128_kernel<T, ROT, NM><<<dim3(q_oblk, H_q, B), 256, 0, stream>>>(
      (const T *)q, (int8_t *)q_int8, (float *)q_scale, Lq, H_q, q_oblk * 8,
      q_stride_b, q_stride_h, q_stride_n, norm);
  quant_k128_kernel<T, NL, BK, ROT, NMK>
      <<<dim3(k_oblk, H_kv, B), 512, 0, stream>>>(
          (const T *)k, (int8_t *)k_int8, (float *)k_scale, anchors, Lk, H_kv,
          k_oblk * 4, k_stride_b, k_stride_h, k_stride_n, norm);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("quant_qk kernel launch failed: ") +
                             cudaGetErrorString(error));
}

} // namespace

extern "C" void launch_quant_qk_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ,
    int WARPQ, int BLKK, int WARPK, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, void *anchor_indices,
    cudaStream_t stream) {
  if (C != 64 && C != 128 && C != 256) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: padded head_dim must be 64, 128, or 256");
  }
  const int expected_warpq = C == 256 ? 16 : 32;
  if (BLKQ != 128 || WARPQ != expected_warpq ||
      (BLKK != 64 && BLKK != 128) || WARPK != BLKK ||
      (C == 64 && BLKK != 64)) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: unsupported block/warp configuration");
  }
  if (!anchor_indices) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: anchor_indices scratch is required");
  }
  // An extent of one is exempt: its index is always zero, so the stride never
  // reaches an address, and PyTorch rewrites such a stride to 1 on the way
  // through DLPack (ATen/DLConvertor.cpp, gh-83069) on 2.9 and older, which no
  // caller can prevent.
  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const size_t vector_size = 4 * element_size;
  const auto stride_ok = [](int64_t stride, int extent) {
    return extent < 2 || (stride > 0 && stride % 4 == 0);
  };
  const auto is_vector_aligned = [vector_size, &stride_ok](const void *ptr, int64_t stride_b,
                                                            int64_t stride_h, int64_t stride_n,
                                                            int extent_b, int extent_h,
                                                            int extent_n) {
    return reinterpret_cast<uintptr_t>(ptr) % vector_size == 0 &&
           stride_ok(stride_b, extent_b) && stride_ok(stride_h, extent_h) &&
           stride_ok(stride_n, extent_n);
  };
  if (!is_vector_aligned(q, q_stride_b, q_stride_h, q_stride_n, B, H_q, Lq) ||
      !is_vector_aligned(k, k_stride_b, k_stride_h, k_stride_n, B, H_kv, Lk)) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: Q/K base pointers and B/H/N strides must preserve 4-element alignment");
  }

  const int q_oblk = (Lq + BLKQ - 1) / BLKQ * (BLKQ / WARPQ);
  const int k_oblk = (Lk + BLKK - 1) / BLKK * (BLKK / WARPK);
  const int q_sc_per_h = q_oblk * 8;
  const int k_sc_per_h = k_oblk * 4;
  auto *anchor_ptr = static_cast<int *>(anchor_indices);
  dim3 gd(H_kv, B);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    detect_k_anchor<T><<<gd, CENTER_DETECT_THREADS, 0, stream>>>(
        (const T *)k, anchor_ptr, Lk, C, H_kv, k_stride_b, k_stride_h,
        k_stride_n);
  });
  cudaError_t anchor_error = cudaGetLastError();
  if (anchor_error != cudaSuccess) {
    throw std::runtime_error(std::string("detect_k_anchor kernel launch failed: ") +
                             cudaGetErrorString(anchor_error));
  }

#define LAUNCH_SPLIT(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    dim3 gq(q_oblk, H_q, B);                                                   \
    quant_q_kernel<T, NR, BQ, WQ, CT, ROT, A4>                                 \
        <<<gq, 128, 0, stream>>>((const T *)q, (int8_t *)q_int8,               \
                                 (float *)q_scale, Lq, C, H_q, q_sc_per_h,     \
                                 q_stride_b, q_stride_h, q_stride_n);           \
    cudaError_t q_error = cudaGetLastError();                                   \
    if (q_error != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("quant_q kernel launch failed: ") + \
                               cudaGetErrorString(q_error));                    \
    dim3 gk(k_oblk, H_kv, B);                                                  \
    quant_k_kernel<T, NL, WK, CT, ROT, A4>                                     \
        <<<gk, 128, 0, stream>>>(                                              \
        (const T *)k, (int8_t *)k_int8, (float *)k_scale, anchor_ptr, Lk, C,   \
        H_kv, k_sc_per_h, k_stride_b, k_stride_h, k_stride_n);                 \
    cudaError_t k_error = cudaGetLastError();                                   \
    if (k_error != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("quant_k kernel launch failed: ") + \
                               cudaGetErrorString(k_error));                    \
  } while (0)

#define LAUNCH_FUSED(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    const int H_max = H_q > H_kv ? H_q : H_kv;                                 \
    dim3 g(q_oblk + k_oblk, H_max, B);                                         \
    quant_qk_fused<T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4>                     \
        <<<g, 128, 0, stream>>>(                                                \
        (const T *)q, (int8_t *)q_int8, (float *)q_scale, (const T *)k,        \
        (int8_t *)k_int8, (float *)k_scale, anchor_ptr, Lq, Lk, C,             \
        q_oblk, H_q, H_kv, q_sc_per_h, k_sc_per_h, q_stride_b, q_stride_h,     \
        q_stride_n, k_stride_b, k_stride_h, k_stride_n);                       \
    cudaError_t qk_error = cudaGetLastError();                                  \
    if (qk_error != cudaSuccess)                                                \
      throw std::runtime_error(                                                 \
          std::string("quant_qk fused kernel launch failed: ") +              \
          cudaGetErrorString(qk_error));                                        \
  } while (0)

#define LAUNCH_C64(T, ROT)                                                      \
  LAUNCH_FUSED(T, 4, 8, 128, 32, 64, 64, 1, ROT, false)

// Head dim 128 takes separate Q and K kernels with more warps per row group.
#define LAUNCH_C128(T, NL, BK, ROT) launch_qk128<T, NL, BK, ROT, 0>(q, q_int8, q_scale, k, k_int8, k_scale, anchor_ptr, B, H_q, Lq, H_kv, Lk, q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h, k_stride_n, QkNormRope{}, stream)

#define LAUNCH_C128_CTA64(T, ROT) LAUNCH_C128(T, 8, 64, ROT)

#define LAUNCH_C128_CTA128(T, ROT) LAUNCH_C128(T, 16, 128, ROT)

#define LAUNCH_C256_CTA64(T, ROT)                                              \
  LAUNCH_SPLIT(T, 2, 8, 128, 16, 64, 64, 2, ROT, true)

#define LAUNCH_C256_CTA128(T, ROT)                                             \
  LAUNCH_SPLIT(T, 2, 16, 128, 16, 128, 128, 2, ROT, true)

// Dispatch head dimension and CTA shape before rotation. The previous
// LAUNCH_SELECTED macro put runtime shape checks inside every rotation/dtype
// branch, which made NVCC instantiate all eight tile configurations at every
// call site even though only these five configurations can reach the kernel.
#define DISPATCH_C64(T, LAUNCHER)                                              \
  do {                                                                         \
    if (Lk <= 256) {                                                           \
      LAUNCHER(T, 4);                                                          \
    } else {                                                                   \
      LAUNCHER(T, 64);                                                         \
    }                                                                          \
  } while (0)

#define DISPATCH_C128(T, LAUNCHER)                                             \
  do {                                                                         \
    if (Lk <= 256) {                                                           \
      LAUNCHER(T, 4);                                                          \
    } else {                                                                   \
      LAUNCHER(T, 128);                                                        \
    }                                                                          \
  } while (0)

#define DISPATCH_C256(T, LAUNCHER)                                             \
  do {                                                                         \
    if (Lk <= 256) {                                                           \
      LAUNCHER(T, 4);                                                          \
    } else {                                                                   \
      LAUNCHER(T, 129);                                                        \
    }                                                                          \
  } while (0)

#define DO(T)                                                                  \
  do {                                                                         \
    if (C == 64) {                                                             \
      DISPATCH_C64(T, LAUNCH_C64);                                             \
    } else if (C == 128) {                                                     \
      if (BLKK == 128) {                                                       \
        DISPATCH_C128(T, LAUNCH_C128_CTA128);                                  \
      } else {                                                                 \
        DISPATCH_C128(T, LAUNCH_C128_CTA64);                                   \
      }                                                                        \
    } else if (BLKK == 128) {                                                  \
      DISPATCH_C256(T, LAUNCH_C256_CTA128);                                    \
    } else {                                                                   \
      DISPATCH_C256(T, LAUNCH_C256_CTA64);                                     \
    }                                                                          \
  } while (0)

  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] { DO(T); });

#undef LAUNCH_SPLIT
#undef LAUNCH_FUSED
#undef LAUNCH_C64
#undef LAUNCH_C128_CTA64
#undef LAUNCH_C128_CTA128
#undef LAUNCH_C128
#undef LAUNCH_C256_CTA64
#undef LAUNCH_C256_CTA128
#undef DISPATCH_C64
#undef DISPATCH_C128
#undef DISPATCH_C256
#undef DO
}

// launch_quant_qk_norm_rope_int8's kernels: Q in mode NM, K in mode NMK.
template <int NM, int NMK>
static void launch_norm128(const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8, void *k_scale, int *anchors,
                           int B, int H_q, int Lq, int H_kv, int Lk, int BLKK, int64_t q_stride_b, int64_t q_stride_h,
                           int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
                           const QkNormRope &norm, cudaStream_t stream) {
  using T = nv_bfloat16;
  detect_k_anchor<T, NMK><<<dim3(H_kv, B), CENTER_DETECT_THREADS, 0, stream>>>(
      (const T *)k, anchors, Lk, 128, H_kv, k_stride_b, k_stride_h, k_stride_n, norm);
#define NORM128(NL, BK, ROT) launch_qk128<T, NL, BK, ROT, NM, NMK>(q, q_int8, q_scale, k, k_int8, k_scale, anchors, B, H_q, Lq, H_kv, Lk, q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h, k_stride_n, norm, stream)
  if (BLKK == 128) {
    if (Lk <= 256) NORM128(16, 128, 4); else NORM128(16, 128, 128);
  } else {
    if (Lk <= 256) NORM128(8, 64, 4); else NORM128(8, 64, 128);
  }
#undef NORM128
}

// launch_quant_qk_per_thread_int8 for bf16 head dim 128, with Q and K first
// taken through gpu_qk_norm_rope's per-head RMSNorm, weight and RoPE
// (rope_mode 1 or 2; a null table for none). Q and K in memory are left as
// they are. A null kw takes K as it is (normalized already), as the plain
// quantizer does.
extern "C" void launch_quant_qk_norm_rope_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int BLKK,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
    void *anchor_indices, const float *qw, const float *kw, const float *rope,
    float eps, int rope_mode, cudaStream_t stream) {
  using T = nv_bfloat16;
  if (BLKK != 64 && BLKK != 128)
    throw std::runtime_error("quant_qk_norm_rope_int8: unsupported key block");
  if (rope_mode < 0 || rope_mode > 2)
    throw std::runtime_error("quant_qk_norm_rope_int8: rope_mode must be 0, 1 or 2");
  if (reinterpret_cast<uintptr_t>(rope) % 16 != 0)
    throw std::runtime_error("quant_qk_norm_rope_int8: the rope table must be 16-byte aligned");
  const auto ok = [](const void *ptr, int64_t sb, int64_t sh, int64_t sn) {
    return reinterpret_cast<uintptr_t>(ptr) % 8 == 0 && sb % 4 == 0 &&
           sh % 4 == 0 && sn % 4 == 0;
  };
  if (!ok(q, q_stride_b, q_stride_h, q_stride_n) ||
      !ok(k, k_stride_b, k_stride_h, k_stride_n))
    throw std::runtime_error(
        "quant_qk_norm_rope_int8: Q/K must keep 4-element alignment");
  const QkNormRope norm{qw, kw, rope, eps};
  // gpu_qk_norm_rope's lane order follows rope_mode even without a table
  const int nm = rope && rope_mode != 0 ? rope_mode : (rope_mode == 2 ? 4 : 3);
  auto *anchors = static_cast<int *>(anchor_indices);
  const auto go = [&](auto nm_tag) {
    constexpr int NM = decltype(nm_tag)::value;
    if (kw)
      launch_norm128<NM, NM>(q, q_int8, q_scale, k, k_int8, k_scale, anchors, B, H_q, Lq, H_kv, Lk, BLKK, q_stride_b, q_stride_h,
                             q_stride_n, k_stride_b, k_stride_h, k_stride_n, norm, stream);
    else
      launch_norm128<NM, 0>(q, q_int8, q_scale, k, k_int8, k_scale, anchors, B, H_q, Lq, H_kv, Lk, BLKK, q_stride_b, q_stride_h,
                            q_stride_n, k_stride_b, k_stride_h, k_stride_n, norm, stream);
  };
  switch (nm) {
  case 1: go(std::integral_constant<int, 1>{}); break;
  case 2: go(std::integral_constant<int, 2>{}); break;
  case 3: go(std::integral_constant<int, 3>{}); break;
  default: go(std::integral_constant<int, 4>{}); break;
  }
}

namespace {
// Fixed H128 rotations and no sequence-wide centering: a sealed page never changes.
__global__ void quant_k_decode_pages(
    const nv_bfloat16* k, int8_t* out, float* scales, const int64_t* position,
    int B, int H, int capacity, int pages, int page_size, bool initialize) {
  const int b = blockIdx.z / H, h = blockIdx.z % H;
  const int length = static_cast<int>(position[0]);
  const int page = initialize ? blockIdx.y : max(0, (length - 1) / page_size - 1) + blockIdx.y;
  if (page >= pages || page * page_size >= length) return;
  const int live = min(page_size, length - page * page_size);
  const int64_t dst = (static_cast<int64_t>(b) * pages + page) * H + h;
  const int source_row = initialize ? page * page_size : page * page_size % capacity;
  process_k<nv_bfloat16, 8, 64, 2, 129, true>(
      k + ((static_cast<int64_t>(b) * H + h) * capacity + source_row) * 256,
      out + dst * page_size * 256, scales + dst * (page_size / 64) * 4,
      blockIdx.x, live, 256, -1, 256);
}
}

extern "C" void launch_int8_decode_quant_q(
    const void* q, int8_t* out, float* scales, int B, int H, int rows, cudaStream_t stream) {
  quant_q_kernel<nv_bfloat16, 2, 128, 16, 2, 129, true><<<dim3(8, H, B), 128, 0, stream>>>(
      static_cast<const nv_bfloat16*>(q), out, scales, rows, 256, H, 64,
      static_cast<int64_t>(H) * rows * 256, static_cast<int64_t>(rows) * 256, 256);
}

extern "C" void launch_int8_decode_quant_k(
    const void* k, int8_t* out, float* scales, const int64_t* position,
    int B, int H, int capacity, int pages, int page_size, bool initialize, cudaStream_t stream) {
  quant_k_decode_pages<<<dim3(page_size / 64, initialize ? pages : 2, B * H), 128, 0, stream>>>(
      static_cast<const nv_bfloat16*>(k), out, scales, position, B, H, capacity, pages, page_size, initialize);
}
