// MiniMax H3: its weight formats, split-half RoPE on part of a head, and the audio decoder's 1-D ops.
#include "common.cuh"
#include <cuda_fp8.h>

__device__ __forceinline__ float e4m3(uint8_t v) { return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)); }

// ---------------------------------------------------------------------------------------------
// w6a8: rows of K 6-bit codes, a nibble plane of K/2 bytes then a K/4 byte plane of top bits, with
// an fp8 scale per G columns. Decodes to the int8 grid the GEMM reads: round((c - 32) * s_rel), +-127.
// ---------------------------------------------------------------------------------------------
__global__ void w6_decode_kernel(const uint8_t* q, const uint8_t* s_rel, int8_t* out, int64_t N, int K, int G) {
    const int64_t per = K / 16;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * per) return;
    const int64_t n = i / per;
    const int c = (int)(i % per);
    const uint8_t* row = q + n * (K * 3 / 4);
    const uint2 lo = *reinterpret_cast<const uint2*>(row + c * 8);
    const unsigned hi = *reinterpret_cast<const unsigned*>(row + K / 2 + c * 4);
    const float s = e4m3(s_rel[n * (K / G) + c * 16 / G]);
    char4 o[4];
    int8_t* b = reinterpret_cast<int8_t*>(o);
    const unsigned words[2] = {lo.x, lo.y};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
            const int oo = w * 4 + bi;
            const unsigned byte = (words[w] >> (bi * 8)) & 0xFF;
            const unsigned c0 = (byte & 0xF) | (((hi >> (4 * oo)) & 3u) << 4);
            const unsigned c1 = ((byte >> 4) & 0xF) | (((hi >> (4 * oo + 2)) & 3u) << 4);
            b[2 * oo] = (int8_t)fminf(127.f, fmaxf(-127.f, nearbyintf(((float)c0 - 32.f) * s)));
            b[2 * oo + 1] = (int8_t)fminf(127.f, fmaxf(-127.f, nearbyintf(((float)c1 - 32.f) * s)));
        }
    }
    *reinterpret_cast<uint4*>(out + n * K + c * 16) = *reinterpret_cast<uint4*>(o);
}

// q [N, 3K/4], s_rel [N, K/G] -> out int8 [N, K]; the row scales stay as stored.
GPU_API int gpu_w6_decode(const void* q, const void* s_rel, void* out, int64_t N, int K, int G, void* stream) {
    if (K % 32 || G % 16) return -1;
    const int64_t n = N * (K / 16);
    w6_decode_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint8_t*)q, (const uint8_t*)s_rel, (int8_t*)out, N, K, G);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// nvfp4: 4-bit E2M1 values, high nibble first, with an fp8 scale per 16 columns in cuBLAS's tiled
// layout and one scale for the tensor. Written as f32 holding bf16 values, as ComfyUI dequantizes
// them; `pre` multiplies the columns, in place of scaling the activation (ModelOpt AWQ).
// ---------------------------------------------------------------------------------------------
__device__ __constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__global__ void nvfp4_kernel(const uint8_t* q, const uint8_t* scales, float scale2, const bf16* pre, float* out, int64_t N, int K) {
    const int per = K / 16;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * per) return;
    const int64_t n = i / per;
    const int c = (int)(i % per);
    const int64_t cblocks = (per + 3) / 4;
    const int rr = (int)(n % 128);
    const int64_t tile = (n / 128) * cblocks + c / 4;
    const int64_t flat = tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + c % 4;
    const float total = rnd<bf16>(scale2 * rnd<bf16>(e4m3(scales[flat])));
    const uint8_t* row = q + n * (K / 2) + c * 8;
    float* o = out + n * K + c * 16;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const uint8_t v = row[b];
        float a = rnd<bf16>(kE2M1[v >> 4] * total), d = rnd<bf16>(kE2M1[v & 15] * total);
        if (pre) {
            a *= __bfloat162float(pre[c * 16 + 2 * b]);
            d *= __bfloat162float(pre[c * 16 + 2 * b + 1]);
        }
        o[2 * b] = a;
        o[2 * b + 1] = d;
    }
}

GPU_API int gpu_nvfp4_dequant(const void* q, const void* scales, float scale2, const void* pre, void* out, int64_t N, int K, void* stream) {
    if (K % 16) return -1;
    const int64_t n = N * (K / 16);
    nvfp4_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint8_t*)q, (const uint8_t*)scales, scale2, (const bf16*)pre, (float*)out, N, K);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Per-head RMSNorm with a weight, then split-half RoPE on the first 2*pairs dims, bf16 in place.
// x [L, H, d] with rows `stride` and heads `hs` elements apart; rope [L][pairs][(cos, sin)]; w may be null.
// ---------------------------------------------------------------------------------------------
__global__ void rope_bf16_kernel(bf16* x, int64_t stride, int64_t hs, const float* w, float eps, const float* rope, int H, int d, int pairs) {
    __shared__ float red[33];
    const int t = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    bf16* v = x + (int64_t)t * stride + (int64_t)h * hs;
    const float e = __bfloat162float(v[i]);
    const float rstd = rsqrtf(block_sum(e * e, red) / (float)d + eps);
    const float wi = w ? w[i] : 1.f;
    if (i < pairs) {
        const float* r = rope + ((int64_t)t * pairs + i) * 2;
        const float a = rnd<bf16>(e * rstd * wi), b = rnd<bf16>(__bfloat162float(v[i + pairs]) * rstd * (w ? w[i + pairs] : 1.f));
        v[i] = __float2bfloat16_rn(a * r[0] - b * r[1]);
        v[i + pairs] = __float2bfloat16_rn(b * r[0] + a * r[1]);
    } else if (i >= 2 * pairs) {
        v[i] = __float2bfloat16_rn(e * rstd * wi);
    }
}
GPU_API int gpu_rope_bf16(void* x, int64_t stride, int64_t hs, const float* w, float eps, const float* rope, int L, int H, int d, int pairs, void* stream) {
    rope_bf16_kernel<<<L * H, d, 0, (cudaStream_t)stream>>>((bf16*)x, stride, hs, w, eps, rope, H, d, pairs);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 1-D ops of the audio decoder, f32, [B, C, T] with a row per (batch, channel)
// ---------------------------------------------------------------------------------------------
// out = conv1d(x, w [Co, Ci, K]) + bias, with `pad` on both sides (zeros) and a dilation
__global__ void conv1d_kernel(const float* x, const float* w, const float* bias, float* out, int B, int Ci, int Co, int T, int K, int dil, int pad) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * Co * T) return;
    const int t = (int)(i % T);
    const int64_t bc = i / T;
    const int c = (int)(bc % Co), b = (int)(bc / Co);
    const float* xb = x + (int64_t)b * Ci * T;
    const float* wc = w + (int64_t)c * Ci * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int s = t - pad + k * dil;
        if (s < 0 || s >= T) continue;
        const float* wk = wc + k;
        for (int ci = 0; ci < Ci; ++ci) acc += wk[(int64_t)ci * K] * xb[(int64_t)ci * T + s];
    }
    out[i] = acc + (bias ? bias[c] : 0.f);
}
GPU_API int gpu_conv1d(const void* x, const void* w, const void* bias, void* out, int B, int Ci, int Co, int T, int K, int dil, int pad, void* stream) {
    const int64_t n = (int64_t)B * Co * T;
    conv1d_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const float*)x, (const float*)w, (const float*)bias, (float*)out, B, Ci, Co, T, K, dil, pad);
    KCHECK();
    return 0;
}

// out = conv_transpose1d(x, w [Ci, Co, K], stride u, padding pad) + bias; Tout = (T - 1) * u + K - 2 * pad.
// (a transposed conv's weight has the input channels first)
__global__ void conv_t1d_kernel(const float* x, const float* w, const float* bias, float* out, int B, int Ci, int Co, int T, int Tout, int K, int u, int pad) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)B * Co * Tout) return;
    const int to = (int)(i % Tout);
    const int64_t bc = i / Tout;
    const int c = (int)(bc % Co), b = (int)(bc / Co);
    const float* xb = x + (int64_t)b * Ci * T;
    const int full = to + pad;
    float acc = 0.f;
    for (int k = full % u; k < K; k += u) {
        const int s = (full - k) / u;
        if (s < 0 || s >= T) continue;
        for (int ci = 0; ci < Ci; ++ci) acc += w[((int64_t)ci * Co + c) * K + k] * xb[(int64_t)ci * T + s];
    }
    out[i] = acc + (bias ? bias[c] : 0.f);
}
GPU_API int gpu_conv_t1d(const void* x, const void* w, const void* bias, void* out, int B, int Ci, int Co, int T, int K, int u, int pad, void* stream) {
    const int tout = (T - 1) * u + K - 2 * pad;
    const int64_t n = (int64_t)B * Co * tout;
    conv_t1d_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const float*)x, (const float*)w, (const float*)bias, (float*)out, B, Ci, Co, T, tout, K, u, pad);
    KCHECK();
    return 0;
}

// Activation1d in one pass, up 2x, snake beta, down 2x, per channel with 12-tap filters:
// x + sin^2(alpha x) / beta on the upsampled signal, replicate padding on both filters.
__global__ void aa_snake_kernel(const float* x, const float* alpha, const float* beta, const float* up, const float* down, float* tmp, int64_t n, int C, int T, int mode) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mode == 0) {
        // upsampled and activated, [rows, 2T]
        const int64_t N = 2 * (n / T) * T;
        if (i >= N) return;
        const int t2 = (int)(i % (2 * T));
        const int64_t r = i / (2 * T);
        const int c = (int)(r % C);
        const float* row = x + r * T;
        // conv_transpose with stride 2 of the replicate-padded input, tap k at sample (t2 + 6 + 2*5 - k) / 2 - 5
        float acc = 0.f;
#pragma unroll
        for (int k = (t2 + 5 + 10) & 1; k < 12; k += 2) {
            const int s = (t2 + 5 + 10 - k) / 2 - 5;
            acc += up[k] * row[s < 0 ? 0 : (s >= T ? T - 1 : s)];
        }
        acc *= 2.f;
        const float a = expf(alpha[c]), b = expf(beta[c]);
        const float sn = sinf(a * acc);
        tmp[i] = sn * sn / (b + 1e-9f) + acc;
    } else {
        if (i >= n) return;
        const int t = (int)(i % T);
        const int64_t r = i / T;
        const float* row = tmp + r * 2 * T;
        const int T2 = 2 * T;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 12; ++k) {
            const int s = 2 * t - 5 + k;
            acc += down[k] * row[s < 0 ? 0 : (s >= T2 ? T2 - 1 : s)];
        }
        const_cast<float*>(x)[i] = acc;
    }
}
// x [rows, T] in place, rows = B * C; alpha, beta [C]; up, down [12]; tmp [rows, 2T]
GPU_API int gpu_aa_snake(void* x, const void* alpha, const void* beta, const void* up, const void* down, void* tmp, int B, int C, int T, void* stream) {
    const int64_t n = (int64_t)B * C * T;
    cudaStream_t st = (cudaStream_t)stream;
    aa_snake_kernel<<<(unsigned)((2 * n + 255) / 256), 256, 0, st>>>((const float*)x, (const float*)alpha, (const float*)beta, (const float*)up, (const float*)down, (float*)tmp, n, C, T, 0);
    aa_snake_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>((const float*)x, (const float*)alpha, (const float*)beta, (const float*)up, (const float*)down, (float*)tmp, n, C, T, 1);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The VAE encoder on one frame, NHWC f16 like ComfyUI's fp16 run: a causal conv on one frame is a 2D conv with the
// kernel's last temporal tap, its spatial padding reflects, and GroupNorm takes its statistics per frame.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ int reflect_at(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

// 3x3 patches of output rows [y0, y0 + rows): out[(y - y0) * Wo + x, (ky * 3 + kx) * C + c]. Stride 1 pads one pixel
// on every side; stride 2 pads one on the right and bottom only (Downsample3D), both by reflection.
__global__ void h3_im2col3_kernel(const __half* in, int H, int W, int C, int stride, int y0, int rows, __half* out) {
    const int Wo = stride == 1 ? W : W / 2, pad = stride == 1 ? 1 : 0;
    const int64_t per = 9 * (int64_t)C;
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)rows * Wo * per) return;
    const int64_t p = t / per;
    const int rest = (int)(t % per);
    const int k = rest / C, c = rest % C;
    const int y = y0 + (int)(p / Wo), x = (int)(p % Wo);
    const int sy = reflect_at(y * stride + k / 3 - pad, H), sx = reflect_at(x * stride + k % 3 - pad, W);
    out[t] = in[((int64_t)sy * W + sx) * C + c];
}

GPU_API int gpu_h3_im2col3(const void* in, int H, int W, int C, int stride, int y0, int rows, void* out, void* stream) {
    const int Wo = stride == 1 ? W : W / 2;
    const int64_t total = (int64_t)rows * Wo * 9 * C;
    h3_im2col3_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const __half*)in, H, W, C, stride, y0, rows, (__half*)out);
    KCHECK();
    return 0;
}

// sums and sums of squares per group, in double: acc [2 * G]
__global__ void h3_gn_stats_kernel(const __half* x, int64_t P, int C, int G, double* acc) {
    const int g = blockIdx.y, cpg = C / G;
    double s = 0.0, q = 0.0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < P * cpg; i += (int64_t)gridDim.x * blockDim.x) {
        const double v = __half2float(x[(i / cpg) * C + g * cpg + i % cpg]);
        s += v;
        q += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
        q += __shfl_xor_sync(0xffffffff, q, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&acc[g], s);
        atomicAdd(&acc[G + g], q);
    }
}

// the normed value rounds to f16, then SiLU of it rounds again, as two fp16 ops do
__global__ void h3_gn_apply_kernel(const __half* x, __half* out, int64_t P, int C, int G, const double* acc, const __half* w, const __half* b, float eps,
                                   int silu) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * C) return;
    const int c = (int)(i % C), g = c / (C / G);
    const double n = (double)P * (C / G);
    const double mean = acc[g] / n;
    const float var = (float)fmax(acc[G + g] / n - mean * mean, 0.0);
    const float v = __half2float(__float2half_rn((__half2float(x[i]) - (float)mean) * rsqrtf(var + eps) * __half2float(w[c]) + __half2float(b[c])));
    out[i] = __float2half_rn(silu ? silu_f(v) : v);
}

// GroupNorm (then SiLU) over one NHWC f16 image [P, C]; scratch is 2 * G doubles.
GPU_API int gpu_h3_group_norm_f16(const void* x, void* out, const void* w, const void* b, int64_t P, int C, int G, float eps, int silu, double* scratch,
                                  void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (cudaMemsetAsync(scratch, 0, 2 * G * sizeof(double), st) != cudaSuccess) return -2;
    h3_gn_stats_kernel<<<dim3(128, G), 256, 0, st>>>((const __half*)x, P, C, G, scratch);
    h3_gn_apply_kernel<<<(unsigned)((P * C + 255) / 256), 256, 0, st>>>((const __half*)x, (__half*)out, P, C, G, scratch, (const __half*)w, (const __half*)b, eps,
                                                                        silu);
    KCHECK();
    return 0;
}
