// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Fused V -> signed INT8 quantization with per-channel scaling.
// Output layout [B,H,D,padded_N] with INT8 MMA 16-element permutation and
// per-row scale.
//
// One thread block per (b, h, d_tile) with D_TILE=8 d-channels. Short K uses
// 128 threads so it does not launch idle warps; longer K uses 512 threads.
// Threads cooperate along N with vectorized 128-bit loads, warp-shuffle absmax
// reduction, and reverse Pass 2 iteration for L2 cache reuse. Pass 1 uses 4×
// manual unroll for memory-level parallelism at low occupancy.
//
// Pass 2 iterates over linear source indices (coalesced 128-bit reads) and
// applies the inverse permutation to compute the destination write index,
// avoiding the scattered-read pattern of iterating over destination indices.

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int kDTile = 8;

// INT8 MMA 16-element inverse permutation.
// Forward: fwd(j) maps bit pattern {b3,b2,b1,b0} → {b1,b3,b2,b0}.
// Inverse: inv(s) maps {b3,b2,b1,b0} → {b2,b0,b3,b1}.
__device__ __forceinline__ int inv_perm16(int w) {
  return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) |
         (((w >> 2) & 1) << 3);
}

template <typename T>
__device__ __forceinline__ void load_tile(const T *ptr, float *out_vals) {
  if constexpr (std::is_same_v<T, float>) {
    float4 v0 = *reinterpret_cast<const float4 *>(ptr);
    float4 v1 = *reinterpret_cast<const float4 *>(ptr + 4);
    out_vals[0] = v0.x;
    out_vals[1] = v0.y;
    out_vals[2] = v0.z;
    out_vals[3] = v0.w;
    out_vals[4] = v1.x;
    out_vals[5] = v1.y;
    out_vals[6] = v1.z;
    out_vals[7] = v1.w;
  } else {
    const T *loaded = comfy::load_f16x8(ptr);
#pragma unroll
    for (int i = 0; i < kDTile; ++i) {
      out_vals[i] = static_cast<float>(loaded[i]);
    }
  }
}

template <typename T, int THREADS, bool paged = false>
__global__ void
quant_v_int8_kernel(const T *__restrict__ v, int8_t *__restrict__ out,
                   float *__restrict__ scale_out, int N, int padded_N, int H,
                   int D, int64_t sb, int64_t sh, int64_t sn,
                   int pages = 0, const int64_t* position = nullptr, bool initialize = false) {
  const int d_tiles = D / kDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kDTile;

  if constexpr (paged) {
    const int length = static_cast<int>(position[0]);
    const int page = initialize ? blockIdx.y : max(0, (length - 1) / padded_N - 1) + blockIdx.y;
    if (page >= pages || page * padded_N >= length) return;
    N = min(padded_N, length - page * padded_N);
    const int source_row = initialize ? page * padded_N : page * padded_N % (sh / sn);
    v += static_cast<int64_t>(source_row) * sn;
    const int64_t offset = (static_cast<int64_t>(b) * (pages - 1) + page) * H * D;
    out += offset * padded_N;
    scale_out += offset;
  }
  const T *base = v + b * sb + h * sh + d0;
  constexpr int WARPS = THREADS / 32;

  // ── Pass 1: per-channel absmax (threads cooperate over N) ──────────
  float mx[kDTile];
#pragma unroll
  for (int i = 0; i < kDTile; ++i)
    mx[i] = 0.f;

  // 4× unrolled: issue 4 independent 128-bit loads per iteration
  // so the memory controller can overlap them (critical at ≤50% occupancy).
  int n = threadIdx.x;
  const int N_body = N - 3 * THREADS;
  for (; n < N_body; n += 4 * THREADS) {
    float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
    load_tile(base + (int64_t)n * sn, t0);
    load_tile(base + (int64_t)(n + THREADS) * sn, t1);
    load_tile(base + (int64_t)(n + 2 * THREADS) * sn, t2);
    load_tile(base + (int64_t)(n + 3 * THREADS) * sn, t3);
#pragma unroll
    for (int di = 0; di < kDTile; ++di) {
      float a = fabsf(t0[di]);
      float b = fabsf(t1[di]);
      float c = fabsf(t2[di]);
      float d = fabsf(t3[di]);
      mx[di] = fmaxf(mx[di], fmaxf(fmaxf(a, b), fmaxf(c, d)));
    }
  }
  for (; n < N; n += THREADS) {
    float tmp[kDTile];
    load_tile(base + (int64_t)n * sn, tmp);
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      mx[di] = fmaxf(mx[di], fabsf(tmp[di]));
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

#pragma unroll
  for (int di = 0; di < kDTile; ++di)
    mx[di] = comfy::warp_reduce_fmax(mx[di]);

  // Cross-warp reduction
  __shared__ float warp_mx[kDTile][WARPS];
  __shared__ float inv_sc_sh[kDTile];

  if (lane == 0) {
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      warp_mx[di][warp] = mx[di];
  }
  __syncthreads();

  if (threadIdx.x < kDTile) {
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w)
      val = fmaxf(val, warp_mx[threadIdx.x][w]);

    float sc = fmaxf(val * (1.f / 127.f), 1e-12f);
    scale_out[(b * H + h) * D + d0 + threadIdx.x] = sc;
    inv_sc_sh[threadIdx.x] = 1.f / sc;
  }
  __syncthreads();

  float inv_sc[kDTile];
#pragma unroll
  for (int di = 0; di < kDTile; ++di)
    inv_sc[di] = inv_sc_sh[di];

  // ── Pass 2: quantize + permute (reverse for L2 reuse) ──────────────
  // Iterate over LINEAR source indices (coalesced reads from v) and compute
  // the permuted destination index for the INT8 MMA layout.
  const int64_t out_row = static_cast<int64_t>((b * H + h) * D + d0);

  for (int src = N - 1 - threadIdx.x; src >= 0; src -= THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | inv_perm16(w);

    float tmp[kDTile];
    load_tile(base + (int64_t)src * sn, tmp);
#pragma unroll
    for (int di = 0; di < kDTile; ++di) {
      out[(out_row + di) * padded_N + dst] =
          comfy::float_to_int8_rn(tmp[di] * inv_sc[di]);
    }
  }

  // Zero-fill padding region [N, padded_N) with the same inverse permutation
  // so that the layout matches the eager reference for partial 16-element groups.
  for (int src = N + threadIdx.x; src < padded_N; src += THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | inv_perm16(w);
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      out[(out_row + di) * padded_N + dst] = 0;
  }
}

// ---------------------------------------------------------------------------
// Head dim 128 in two coalesced passes. quant_v_int8_kernel gives each block
// eight channels, so it reads 16 bytes per row at the row stride, twice; here
// blocks take whole 256-byte rows.
//
// v128_absmax: each block takes 128 rows of one (b, h) and folds its
// per-channel abs-max into absmax[(b * H + h) * 128 + d] (zeroed first) with an
// integer atomicMax, which orders non-negative floats as their bit patterns do.
// v128_quant: each block takes 64 rows, quantizes them with the finished
// scales and writes them through shared memory as 16-byte stores. The output
// is quant_v_int8_kernel's: the same abs-max (it doesn't depend on order), the
// same scale and rounding, the same permutation and zero padding.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __launch_bounds__(256) void v128_absmax_kernel(
    const T *__restrict__ v, unsigned *__restrict__ absmax, int N, int H,
    int64_t sb, int64_t sh, int64_t sn) {
  const int h = blockIdx.y, b = blockIdx.z;
  const int cg = threadIdx.x & 15;   // channels cg * 8 ..
  const int lane_row = threadIdx.x >> 4;
  const T *base = v + b * sb + h * sh + cg * 8;
  float mx[kDTile];
#pragma unroll
  for (int i = 0; i < kDTile; ++i)
    mx[i] = 0.f;
  const int r0 = blockIdx.x * 128;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int n = r0 + r * 16 + lane_row;
    if (n < N) {
      float t[kDTile];
      load_tile(base + (int64_t)n * sn, t);
#pragma unroll
      for (int di = 0; di < kDTile; ++di)
        mx[di] = fmaxf(mx[di], fabsf(t[di]));
    }
  }
  // lanes l and l ^ 16 hold the same channels
#pragma unroll
  for (int di = 0; di < kDTile; ++di)
    mx[di] = fmaxf(mx[di], __shfl_xor_sync(0xffffffffu, mx[di], 16));
  __shared__ float part[8][128];
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) < 16) {
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      part[warp][cg * 8 + di] = mx[di];
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
      val = fmaxf(val, part[w][threadIdx.x]);
    atomicMax(&absmax[(b * H + h) * 128 + threadIdx.x], __float_as_uint(val));
  }
}

template <typename T>
__global__ __launch_bounds__(256) void v128_quant_kernel(
    const T *__restrict__ v, const unsigned *__restrict__ absmax,
    int8_t *__restrict__ out, float *__restrict__ scale_out, int N,
    int padded_N, int H, int64_t sb, int64_t sh, int64_t sn) {
  constexpr int D = 128, ROWS = 64;
  const int h = blockIdx.y, b = blockIdx.z;
  const int bh = b * H + h;
  __shared__ float inv_sc_sh[D];
  // [permuted row][channel]: rows written as 8-byte runs, read back as
  // channel columns
  __shared__ __align__(16) int8_t tile[ROWS][D];
  if (threadIdx.x < D) {
    const float val = __uint_as_float(absmax[bh * D + threadIdx.x]);
    const float sc = fmaxf(val * (1.f / 127.f), 1e-12f);
    if (blockIdx.x == 0)
      scale_out[bh * D + threadIdx.x] = sc;
    inv_sc_sh[threadIdx.x] = 1.f / sc;
  }
  __syncthreads();
  const int cg = threadIdx.x & 15;
  const int lane_row = threadIdx.x >> 4;
  const T *base = v + b * sb + h * sh + cg * 8;
  const int r0 = blockIdx.x * ROWS;
  float inv_sc[kDTile];
#pragma unroll
  for (int di = 0; di < kDTile; ++di)
    inv_sc[di] = inv_sc_sh[cg * 8 + di];
#pragma unroll
  for (int r = 0; r < ROWS / 16; ++r) {
    const int local = r * 16 + lane_row;
    const int src = r0 + local;
    const int dst = (local & ~15) | inv_perm16(local & 15);
    uint32_t w[2] = {0u, 0u};
    if (src < N) {
      float t[kDTile];
      load_tile(base + (int64_t)src * sn, t);
#pragma unroll
      for (int di = 0; di < kDTile; ++di)
        w[di >> 2] |= (static_cast<uint32_t>(static_cast<uint8_t>(
                          comfy::float_to_int8_rn(t[di] * inv_sc[di]))))
                      << ((di & 3) * 8);
    }
    *reinterpret_cast<uint2 *>(&tile[dst][cg * 8]) = make_uint2(w[0], w[1]);
  }
  __syncthreads();
  // channel ch, rows half * 32 .. + 31: one 32-byte run of the output row
  const int ch = threadIdx.x & 127, half = threadIdx.x >> 7;
  uint32_t packed[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    packed[i] = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      packed[i] |= static_cast<uint32_t>(static_cast<uint8_t>(
                       tile[half * 32 + i * 4 + k][ch]))
                   << (k * 8);
  }
  uint4 *dst = reinterpret_cast<uint4 *>(
      &out[((int64_t)bh * D + ch) * padded_N + r0 + half * 32]);
  dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}

} // namespace

extern "C" void launch_quant_v_int8_kernel(const void *v, void *out, void *scale,
                                          int B, int H, int N, int D,
                                          int padded_N, int64_t sb, int64_t sh,
                                          int64_t sn, int input_dtype_code,
                                          cudaStream_t stream) {
  // An extent of one forms no offset, and PyTorch rewrites such a stride to 1 on
  // the way through DLPack (ATen/DLConvertor.cpp, gh-83069) on 2.9 and older, so
  // policing it would reject a tensor the caller has no way to hand over
  // differently.
  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const auto stride_ok = [element_size](int64_t stride, int extent) {
    return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
  };
  if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(sb, B) ||
      !stride_ok(sh, H) || !stride_ok(sn, N)) {
    throw std::runtime_error(
        "quant_v_int8: V base pointer and B/H/N strides must be 16-byte aligned");
  }
  if (D <= 0 || D % kDTile != 0) {
    throw std::runtime_error("quant_v_int8: head_dim must be a positive multiple of 8");
  }

  const int blocks = B * H * (D / kDTile);

  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (N <= 256) {
      quant_v_int8_kernel<T, 128><<<blocks, 128, 0, stream>>>(
          static_cast<const T *>(v), static_cast<int8_t *>(out),
          static_cast<float *>(scale), N, padded_N, H, D, sb, sh, sn);
    } else {
      quant_v_int8_kernel<T, 512><<<blocks, 512, 0, stream>>>(
          static_cast<const T *>(v), static_cast<int8_t *>(out),
          static_cast<float *>(scale), N, padded_N, H, D, sb, sh, sn);
    }
  });

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("quant_v_int8 kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}

// launch_quant_v_int8_kernel for 16-bit V at head dim 128, padded_N a
// multiple of 64: the same output in two coalesced passes. absmax: B * H * 128
// scratch words.
extern "C" void launch_quant_v128_int8(const void *v, void *out, void *scale,
                                       void *absmax, int B, int H, int N,
                                       int padded_N, int64_t sb, int64_t sh,
                                       int64_t sn, int input_dtype_code,
                                       cudaStream_t stream) {
  if (input_dtype_code != comfy::DTYPE_CODE_BFLOAT16 &&
      input_dtype_code != comfy::DTYPE_CODE_FLOAT16)
    throw std::runtime_error("quant_v128_int8: V must be fp16 or bf16");
  if (padded_N % 64 != 0 || padded_N < N)
    throw std::runtime_error("quant_v128_int8: padded_N must be a multiple of 64");
  if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || (B > 1 && sb % 8 != 0) ||
      (H > 1 && sh % 8 != 0) || (N > 1 && sn % 8 != 0) ||
      reinterpret_cast<uintptr_t>(out) % 16 != 0)
    throw std::runtime_error("quant_v128_int8: V and its strides must be 16-byte aligned");
  cudaMemsetAsync(absmax, 0, (size_t)B * H * 128 * sizeof(unsigned), stream);
  const auto go = [&](auto tag) {
    using T = decltype(tag);
    v128_absmax_kernel<T><<<dim3((N + 127) / 128, H, B), 256, 0, stream>>>(
        static_cast<const T *>(v), static_cast<unsigned *>(absmax), N, H, sb,
        sh, sn);
    v128_quant_kernel<T><<<dim3(padded_N / 64, H, B), 256, 0, stream>>>(
        static_cast<const T *>(v), static_cast<const unsigned *>(absmax),
        static_cast<int8_t *>(out), static_cast<float *>(scale), N, padded_N,
        H, sb, sh, sn);
  };
  if (input_dtype_code == comfy::DTYPE_CODE_BFLOAT16)
    go(__nv_bfloat16{});
  else
    go(half{});
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("quant_v128_int8 launch failed: ") +
                             cudaGetErrorString(error));
}

extern "C" void launch_int8_decode_quant_v(
    const void* v, int8_t* out, float* scales, const int64_t* position,
    int B, int H, int capacity, int pages, int page_size, bool initialize, cudaStream_t stream) {
  quant_v_int8_kernel<__nv_bfloat16, 128, true><<<dim3(B * H * 32, initialize ? pages : 2), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(v), out, scales, page_size, page_size, H, 256,
      static_cast<int64_t>(H) * capacity * 256, static_cast<int64_t>(capacity) * 256, 256,
      pages, position, initialize);
}
