// Custom kernels for the Krea 2, Anima and MOSS Image pipelines (DiTs, Qwen text encoders, VAE decoders).
//
// Where ComfyUI materializes an intermediate tensor in the activation dtype (bf16 for the DiT),
// the kernels round to that dtype at the same point (rnd<T>) so results track ComfyUI closely.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Row prologue -> optional ConvRot rotation -> per-row int8 quantization.
// Produces the A operand (+ per-row scales) of the fused int8 GEMM.
// ---------------------------------------------------------------------------------------------
// LN_MOD: LayerNorm then addcmul(shift, n, 1 + scale) (Trellis 2), one rounding after the norm and one after the
// addcmul. LN_AFFINE: LayerNorm with weight and bias, rounded once; LN_AFFINE_SILU then rounds silu of that.
enum { PRO_PLAIN = 0, PRO_RMS = 1, PRO_GATE = 2, PRO_SWIGLU = 3, PRO_LAYERNORM = 4, PRO_SILU = 5, PRO_GELU = 6, PRO_ADALN = 7,
       PRO_SITU = 8, PRO_LN_MOD = 9, PRO_LN_AFFINE = 10, PRO_GELU_TANH = 11, PRO_LN_AFFINE_SILU = 12 };

template <int PRO> __device__ __forceinline__ constexpr bool is_ln() {
    return PRO == PRO_LAYERNORM || PRO == PRO_LN_MOD || PRO == PRO_LN_AFFINE || PRO == PRO_LN_AFFINE_SILU;
}

struct QuantParams {
    const void* x;       // PLAIN/RMS: input row; GATE: attention output; SWIGLU: [gate | up] row
    int64_t x_stride;    // elements between rows of x
    const void* g;       // GATE: gate logits
    int64_t g_stride;
    int64_t up_offset;   // SWIGLU: column of `up` inside the x row; SITU: of `gate`
    const float* norm_w; // RMS weight (zero-centered offset already applied); null = 1
    float eps;
    const void* mod_scale;  // RMS/LAYERNORM modulation (T): v = (1 + scale) * v + shift; null = none
    const void* mod_shift;
    int8_t* q;           // [M, K]
    float* scales;       // [M]
    int K;
    int rotate;
    float* out_f;        // if set: write the (rotated) fp32 row here instead of quantizing
    const float* norm_b; // LN_AFFINE bias
};

// One normalized element after the prologue's norm, for the LayerNorm variants.
template <typename T, int PRO>
__device__ __forceinline__ float ln_finish(float n, int c, const QuantParams& p, float ms, float sh) {
    if (PRO == PRO_LN_MOD) return rnd<T>(fmaf(rnd<T>(n), rnd<T>(1.f + ms), sh));
    if (PRO == PRO_LN_AFFINE) return rnd<T>(fmaf(n, p.norm_w[c], p.norm_b[c]));
    if (PRO == PRO_LN_AFFINE_SILU) return rnd<T>(silu_f(rnd<T>(fmaf(n, p.norm_w[c], p.norm_b[c]))));
    return n;
}

template <typename T> __device__ __forceinline__ float finite_max();
template <> __device__ __forceinline__ float finite_max<float>() { return FLT_MAX; }
template <> __device__ __forceinline__ float finite_max<bf16>() { return 3.38953139e38f; }

// Vectorized 8-element loads (16-byte aligned rows).
__device__ __forceinline__ void load8(const bf16* p, float* v) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}
__device__ __forceinline__ void load8(const float* p, float* v) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// One block per row; warp w owns 256-wide groups w, w + nwarps, ... (at most MAXG). Each lane
// holds 8 consecutive elements of each of its groups in registers; the 256-point transform runs in
// a per-warp shared buffer with warp-level syncs only.
constexpr int kMaxGroupsPerWarp = 4;

template <typename T, int PRO>
__global__ void __launch_bounds__(1024) quant_rows_kernel(QuantParams p) {
    extern __shared__ float wbuf[];  // [nwarps][256]
    __shared__ float red[33];
    constexpr int G = kMaxGroupsPerWarp;
    const int K = p.K;
    const int ngroups = K >> 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const int64_t m = blockIdx.x;
    const T* x = (const T*)p.x + m * p.x_stride;
    float v[G][8];

    // prologue -> v
    float ss = 0.f;
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
        const int g = warp + gi * nwarps;
        if (g >= ngroups) break;
        const int c = (g << 8) + lane * 8;
        if (PRO == PRO_GATE) {
            float a[8], gg[8];
            load8(x + c, a);
            load8((const T*)p.g + m * p.g_stride + c, gg);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[gi][j] = rnd<T>(a[j] * rnd<T>(sigmoid_f(gg[j])));
        } else if (PRO == PRO_SWIGLU) {
            float a[8], u[8];
            load8(x + c, a);
            load8(x + p.up_offset + c, u);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[gi][j] = rnd<T>(rnd<T>(silu_f(a[j])) * u[j]);
        } else if (PRO == PRO_SITU) {
            float a[8], u[8];
            load8(x + c, a);
            load8(x + p.up_offset + c, u);
#pragma unroll
            for (int j = 0; j < 8; ++j) v[gi][j] = situ_f(a[j], u[j]);
        } else {
            load8(x + c, v[gi]);
            if (PRO == PRO_RMS || PRO == PRO_ADALN) {
#pragma unroll
                for (int j = 0; j < 8; ++j) ss += v[gi][j] * v[gi][j];
            } else if (is_ln<PRO>()) {
#pragma unroll
                for (int j = 0; j < 8; ++j) ss += v[gi][j];
            } else if (PRO == PRO_SILU) {
#pragma unroll
                for (int j = 0; j < 8; ++j) v[gi][j] = rnd<T>(silu_f(v[gi][j]));
            } else if (PRO == PRO_GELU) {
#pragma unroll
                for (int j = 0; j < 8; ++j) v[gi][j] = rnd<T>(gelu_f(v[gi][j]));
            } else if (PRO == PRO_GELU_TANH) {
#pragma unroll
                for (int j = 0; j < 8; ++j) v[gi][j] = rnd<T>(gelu_tanh_f(v[gi][j]));
            }
        }
    }
    if (is_ln<PRO>()) {
        const float mean = block_sum(ss, red) / (float)K;
        ss = 0.f;
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
            if (warp + gi * nwarps >= ngroups) break;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                v[gi][j] -= mean;
                ss += v[gi][j] * v[gi][j];
            }
        }
    }
    if (PRO == PRO_RMS || is_ln<PRO>() || PRO == PRO_ADALN) {
        ss = block_sum(ss, red);
        const float rstd = rsqrtf(ss / (float)K + p.eps);
        const T* ms = (const T*)p.mod_scale;
        const T* sh = (const T*)p.mod_shift;
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
            const int g = warp + gi * nwarps;
            if (g >= ngroups) break;
            const int c = (g << 8) + lane * 8;
            float msv[8], shv[8];
            if (ms) {
                load8(ms + c, msv);
                load8(sh + c, shv);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float n = v[gi][j] * rstd;
                if (PRO == PRO_ADALN) {  // one rounding, like the fused rms_adaln
                    v[gi][j] = rnd<T>(n * (1.f + msv[j]) + shv[j]);
                    continue;
                }
                if (PRO != PRO_LAYERNORM && PRO != PRO_RMS) {
                    v[gi][j] = ln_finish<T, PRO>(n, c + j, p, ms ? msv[j] : 0.f, ms ? shv[j] : 0.f);
                    continue;
                }
                if (p.norm_w) n *= p.norm_w[c + j];
                n = rnd<T>(n);
                if (ms) {
                    const float a = rnd<T>(1.f + msv[j]);
                    n = rnd<T>(a * n);
                    n = rnd<T>(n + shv[j]);
                }
                v[gi][j] = n;
            }
        }
    }

    // ConvRot: 4 radix-4 stages (strides 1, 4, 16, 64) per group, same op order as comfy-kitchen
    float amax = 0.f;
    float* wb = wbuf + warp * 256;
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
        const int g = warp + gi * nwarps;
        if (g >= ngroups) break;
        if (p.rotate) {
#pragma unroll
            for (int j = 0; j < 8; ++j) wb[lane * 8 + j] = v[gi][j];
            __syncwarp();
#pragma unroll
            for (int stage = 0; stage < 4; ++stage) {
                const int s = 1 << (2 * stage);
#pragma unroll
                for (int bb = 0; bb < 2; ++bb) {
                    const int r = lane * 2 + bb;
                    const int i0 = (r / s) * (s << 2) + (r % s);
                    const float x0 = wb[i0], x1 = wb[i0 + s], x2 = wb[i0 + 2 * s], x3 = wb[i0 + 3 * s];
                    wb[i0] = 0.5f * h4_row(0, x0, x1, x2, x3);
                    wb[i0 + s] = 0.5f * h4_row(1, x0, x1, x2, x3);
                    wb[i0 + 2 * s] = 0.5f * h4_row(2, x0, x1, x2, x3);
                    wb[i0 + 3 * s] = 0.5f * h4_row(3, x0, x1, x2, x3);
                }
                __syncwarp();
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) v[gi][j] = wb[lane * 8 + j];
            __syncwarp();
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[gi][j]));
    }

    if (p.out_f) {
        float* o = p.out_f + m * (int64_t)K;
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
            const int g = warp + gi * nwarps;
            if (g >= ngroups) break;
            const int c = (g << 8) + lane * 8;
            *reinterpret_cast<float4*>(o + c) = make_float4(v[gi][0], v[gi][1], v[gi][2], v[gi][3]);
            *reinterpret_cast<float4*>(o + c + 4) = make_float4(v[gi][4], v[gi][5], v[gi][6], v[gi][7]);
        }
        return;
    }

    amax = block_max(amax, red);
    const float scale = fmaxf(fminf(amax, finite_max<T>()) * (1.f / 127.f), 1.0e-30f);
    if (threadIdx.x == 0) p.scales[m] = scale;
    const float scale_t = rnd<T>(scale);
    int8_t* q = p.q + m * (int64_t)K;
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
        const int g = warp + gi * nwarps;
        if (g >= ngroups) break;
        const int c = (g << 8) + lane * 8;
        char4 o[2];
        int8_t* ob = reinterpret_cast<int8_t*>(o);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float t = rnd<T>(rnd<T>(v[gi][j]) / scale_t);
            ob[j] = (int8_t)fminf(127.f, fmaxf(-128.f, nearbyintf(t)));
        }
        *reinterpret_cast<uint2*>(q + c) = *reinterpret_cast<uint2*>(o);
    }
}

template <typename T, int PRO>
static int launch_quant(const QuantParams& p, int M, cudaStream_t st) {
    const int ngroups = p.K / 256;
    const int warps = (ngroups + kMaxGroupsPerWarp - 1) / kMaxGroupsPerWarp;
    if (warps > 32) return -2;
    quant_rows_kernel<T, PRO><<<M, warps * 32, p.rotate ? warps * 256 * sizeof(float) : 0, st>>>(p);
    KCHECK();
    return 0;
}

GPU_API int gpu_quant_rows(int dtype, int prologue, const void* x, int64_t x_stride, const void* g,
                             int64_t g_stride, int64_t up_offset, const float* norm_w, float eps,
                             const void* mod_scale, const void* mod_shift, int8_t* q, float* scales,
                             int M, int K, int rotate, void* stream) {
    if ((K % 256) != 0) return -1;
    QuantParams p{x, x_stride, g, g_stride, up_offset, norm_w, eps, mod_scale, mod_shift, q, scales, K, rotate, nullptr};
    cudaStream_t st = (cudaStream_t)stream;
#define QDISPATCH(T)                                                   \
    switch (prologue) {                                                \
        case PRO_PLAIN: return launch_quant<T, PRO_PLAIN>(p, M, st);   \
        case PRO_RMS: return launch_quant<T, PRO_RMS>(p, M, st);       \
        case PRO_GATE: return launch_quant<T, PRO_GATE>(p, M, st);     \
        case PRO_SWIGLU: return launch_quant<T, PRO_SWIGLU>(p, M, st); \
        case PRO_LAYERNORM: return launch_quant<T, PRO_LAYERNORM>(p, M, st); \
        case PRO_SILU: return launch_quant<T, PRO_SILU>(p, M, st);     \
        case PRO_GELU: return launch_quant<T, PRO_GELU>(p, M, st);     \
        case PRO_ADALN: return launch_quant<T, PRO_ADALN>(p, M, st);   \
        case PRO_SITU: return launch_quant<T, PRO_SITU>(p, M, st);     \
        case PRO_LN_MOD: return launch_quant<T, PRO_LN_MOD>(p, M, st); \
        case PRO_LN_AFFINE: return launch_quant<T, PRO_LN_AFFINE>(p, M, st); \
        case PRO_GELU_TANH: return launch_quant<T, PRO_GELU_TANH>(p, M, st); \
        default: return -3;                                            \
    }
    if (dtype == DT_BF16) { QDISPATCH(bf16) }
    if (dtype == DT_F32) { QDISPATCH(float) }
#undef QDISPATCH
    return -4;
}

// gpu_quant_rows with a LayerNorm bias, for PRO_LN_AFFINE.
GPU_API int gpu_quant_rows_b(int dtype, int prologue, const void* x, int64_t x_stride, const float* norm_w, const float* norm_b,
                               float eps, const void* mod_scale, const void* mod_shift, int8_t* q, float* scales, int M, int K,
                               int rotate, void* stream) {
    if ((K % 256) != 0) return -1;
    QuantParams p{x, x_stride, nullptr, 0, 0, norm_w, eps, mod_scale, mod_shift, q, scales, K, rotate, nullptr, norm_b};
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype != DT_BF16) return -4;
    switch (prologue) {
        case PRO_LN_MOD: return launch_quant<bf16, PRO_LN_MOD>(p, M, st);
        case PRO_LN_AFFINE: return launch_quant<bf16, PRO_LN_AFFINE>(p, M, st);
        default: return -3;
    }
}

// Same prologues + ConvRot rotation, fp32 output (weight-only int8 path: x_rot @ (q*s)^T).
GPU_API int gpu_rotate_rows_f32(int prologue, const float* x, int64_t x_stride, int64_t up_offset,
                                  const float* norm_w, float eps, float* out, int M, int K, void* stream) {
    if ((K % 256) != 0) return -1;
    QuantParams p{x, x_stride, nullptr, 0, up_offset, norm_w, eps, nullptr, nullptr, nullptr, nullptr, K, 1, out};
    cudaStream_t st = (cudaStream_t)stream;
    switch (prologue) {
        case PRO_PLAIN: return launch_quant<float, PRO_PLAIN>(p, M, st);
        case PRO_RMS: return launch_quant<float, PRO_RMS>(p, M, st);
        case PRO_SWIGLU: return launch_quant<float, PRO_SWIGLU>(p, M, st);
        default: return -3;
    }
}

// int8 [N, K] * scale[n] -> fp32 [N, K]
__global__ void dequant_i8_kernel(const int8_t* q, const float* s, float* out, int64_t N, int64_t K) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= N * K) return;
    const char4 v = *reinterpret_cast<const char4*>(q + i);
    const float sc = s[i / K];
    *reinterpret_cast<float4*>(out + i) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
}
GPU_API int gpu_dequant_i8(const int8_t* q, const float* s, float* out, int64_t N, int64_t K, void* stream) {
    if (K % 4) return -1;
    const int64_t n4 = N * K / 4;
    dequant_i8_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, (cudaStream_t)stream>>>(q, s, out, N, K);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Same prologues without quantization: writes the activation (T) for a bf16/f32 GEMM.
// ---------------------------------------------------------------------------------------------
template <typename T, int PRO>
__global__ void prologue_rows_kernel(QuantParams p, T* out, int64_t out_stride) {
    __shared__ float red[33];
    const int K = p.K;
    const int64_t m = blockIdx.x;
    const T* x = (const T*)p.x + m * p.x_stride;
    T* o = out + m * out_stride;
    if (PRO == PRO_RMS || is_ln<PRO>() || PRO == PRO_ADALN) {
        float mean = 0.f;
        if (is_ln<PRO>()) {
            for (int k = threadIdx.x; k < K; k += blockDim.x) mean += ldf(x, k);
            mean = block_sum(mean, red) / (float)K;
        }
        float ss = 0.f;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float v = ldf(x, k) - mean;
            ss += v * v;
        }
        ss = block_sum(ss, red);
        const float rstd = rsqrtf(ss / (float)K + p.eps);
        const T* ms = (const T*)p.mod_scale;
        const T* sh = (const T*)p.mod_shift;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            float n = (ldf(x, k) - mean) * rstd;
            if (PRO == PRO_ADALN) {
                stf(o, k, n * (1.f + ldf(ms, k)) + ldf(sh, k));
                continue;
            }
            if (PRO != PRO_LAYERNORM && PRO != PRO_RMS) {
                stf(o, k, ln_finish<T, PRO>(n, k, p, ms ? ldf(ms, k) : 0.f, ms ? ldf(sh, k) : 0.f));
                continue;
            }
            if (p.norm_w) n *= p.norm_w[k];
            n = rnd<T>(n);
            if (ms) {
                const float a = rnd<T>(1.f + ldf(ms, k));
                n = rnd<T>(a * n);
                n = rnd<T>(n + ldf(sh, k));
            }
            stf(o, k, n);
        }
    } else if (PRO == PRO_GATE) {
        const T* g = (const T*)p.g + m * p.g_stride;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float s = rnd<T>(sigmoid_f(ldf(g, k)));
            stf(o, k, ldf(x, k) * s);
        }
    } else if (PRO == PRO_SWIGLU) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float s = rnd<T>(silu_f(ldf(x, k)));
            stf(o, k, s * ldf(x, p.up_offset + k));
        }
    } else if (PRO == PRO_SILU) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) stf(o, k, silu_f(ldf(x, k)));
    } else if (PRO == PRO_GELU) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) stf(o, k, gelu_f(ldf(x, k)));
    } else if (PRO == PRO_GELU_TANH) {
        for (int k = threadIdx.x; k < K; k += blockDim.x) stf(o, k, gelu_tanh_f(ldf(x, k)));
    } else if (PRO == PRO_SITU) {
        // the eager op chain: every intermediate tensor is rounded
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float hidden = ldf(x, k), gate = ldf(x, p.up_offset + k);
            const float capped = rnd<T>(4.f * rnd<T>(stable_tanh(rnd<T>(gate / 4.f))));
            const float g = rnd<T>(capped * rnd<T>(sigmoid_f(gate)));
            const float h = rnd<T>(25.f * rnd<T>(stable_tanh(rnd<T>(hidden / 25.f))));
            stf(o, k, g * h);
        }
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) stf(o, k, ldf(x, k));
    }
}

GPU_API int gpu_prologue_rows(int dtype, int prologue, const void* x, int64_t x_stride, const void* g,
                                int64_t g_stride, int64_t up_offset, const float* norm_w, float eps,
                                const void* mod_scale, const void* mod_shift, void* out, int64_t out_stride,
                                int M, int K, void* stream) {
    QuantParams p{x, x_stride, g, g_stride, up_offset, norm_w, eps, mod_scale, mod_shift, nullptr, nullptr, K, 0, nullptr};
    cudaStream_t st = (cudaStream_t)stream;
    const int threads = K >= 2048 ? 512 : 256;
#define PDISPATCH(T)                                                                                       \
    switch (prologue) {                                                                                    \
        case PRO_PLAIN: prologue_rows_kernel<T, PRO_PLAIN><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;   \
        case PRO_RMS: prologue_rows_kernel<T, PRO_RMS><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;       \
        case PRO_GATE: prologue_rows_kernel<T, PRO_GATE><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;     \
        case PRO_SWIGLU: prologue_rows_kernel<T, PRO_SWIGLU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        case PRO_LAYERNORM: prologue_rows_kernel<T, PRO_LAYERNORM><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        case PRO_SILU: prologue_rows_kernel<T, PRO_SILU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;     \
        case PRO_GELU: prologue_rows_kernel<T, PRO_GELU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;     \
        case PRO_ADALN: prologue_rows_kernel<T, PRO_ADALN><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;   \
        case PRO_SITU: prologue_rows_kernel<T, PRO_SITU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;     \
        default: return -3;                                                                                \
    }
    if (dtype == DT_BF16) { PDISPATCH(bf16) }
    else if (dtype == DT_F32) { PDISPATCH(float) }
    else return -4;
#undef PDISPATCH
    KCHECK();
    return 0;
}

// gpu_prologue_rows for the LayerNorm variants with a bias, and the plain ones, in bf16, f16 or f32.
GPU_API int gpu_prologue_rows_b(int dtype, int prologue, const void* x, int64_t x_stride, const float* norm_w,
                                  const float* norm_b, float eps, const void* mod_scale, const void* mod_shift, void* out,
                                  int64_t out_stride, int M, int K, void* stream) {
    QuantParams p{x, x_stride, nullptr, 0, 0, norm_w, eps, mod_scale, mod_shift, nullptr, nullptr, K, 0, nullptr, norm_b};
    cudaStream_t st = (cudaStream_t)stream;
    const int threads = K >= 2048 ? 512 : K >= 256 ? 256 : 64;
#define BDISPATCH(T)                                                                                       \
    switch (prologue) {                                                                                    \
        case PRO_PLAIN: prologue_rows_kernel<T, PRO_PLAIN><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;   \
        case PRO_LAYERNORM: prologue_rows_kernel<T, PRO_LAYERNORM><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        case PRO_LN_MOD: prologue_rows_kernel<T, PRO_LN_MOD><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        case PRO_LN_AFFINE: prologue_rows_kernel<T, PRO_LN_AFFINE><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        case PRO_LN_AFFINE_SILU: prologue_rows_kernel<T, PRO_LN_AFFINE_SILU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        case PRO_SILU: prologue_rows_kernel<T, PRO_SILU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;     \
        case PRO_GELU: prologue_rows_kernel<T, PRO_GELU><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break;     \
        case PRO_GELU_TANH: prologue_rows_kernel<T, PRO_GELU_TANH><<<M, threads, 0, st>>>(p, (T*)out, out_stride); break; \
        default: return -3;                                                                                \
    }
    if (dtype == DT_BF16) { BDISPATCH(bf16) }
    else if (dtype == DT_F16) { BDISPATCH(half) }
    else if (dtype == DT_F32) { BDISPATCH(float) }
    else return -4;
#undef BDISPATCH
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Per-head RMSNorm of q and k (head dim 32 * EPL) + rotary embedding, in place.
// rope_mode 1: interleaved pairs (Flux/Krea2 EmbedND), 2: split-half (Llama/Qwen/Cosmos).
// rope table: [L][D/2] x (cos, sin) float.
// ---------------------------------------------------------------------------------------------
template <typename T, int EPL>
__global__ void qk_norm_rope_kernel(T* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk,
                                    const float* qw, const float* kw, float eps, const float* rope,
                                    int rope_mode, int L) {
    constexpr int D = 32 * EPL, PP = EPL / 2;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int heads = hq + hk;
    if (warp >= L * heads) return;
    const int t = warp / heads, h = warp % heads;
    const bool isq = h < hq;
    T* x = buf + (int64_t)t * row_stride + (isq ? q_off + h * D : k_off + (h - hq) * D);
    const float* w = isq ? qw : kw;
    // element indices owned by this lane
    int idx[EPL];
#pragma unroll
    for (int i = 0; i < PP; ++i) {
        if (rope_mode == 2) {
            idx[i] = PP * lane + i;
            idx[PP + i] = D / 2 + PP * lane + i;
        } else {
            idx[i] = EPL * lane + i;
            idx[PP + i] = EPL * lane + PP + i;
        }
    }
    float v[EPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        v[i] = ldf(x, idx[i]);
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / (float)D + eps);
#pragma unroll
    for (int i = 0; i < EPL; ++i) v[i] = rnd<T>(v[i] * rstd * (w ? w[idx[i]] : 1.f));
    if (rope && rope_mode == 1) {
        const float* r = rope + (int64_t)t * D;
#pragma unroll
        for (int pi = 0; pi < PP; ++pi) {
            const int j = PP * lane + pi;
            const float c = r[2 * j], s = r[2 * j + 1];
            const float a = v[2 * pi], b = v[2 * pi + 1];
            v[2 * pi] = c * a - s * b;
            v[2 * pi + 1] = s * a + c * b;
        }
    } else if (rope && rope_mode == 2) {
        const float* r = rope + (int64_t)t * D;
#pragma unroll
        for (int pi = 0; pi < PP; ++pi) {
            const int j = PP * lane + pi;
            const float c = r[2 * j], s = r[2 * j + 1];
            const float a = v[pi], b = v[PP + pi];
            v[pi] = a * c - b * s;
            v[PP + pi] = b * c + a * s;
        }
    }
#pragma unroll
    for (int i = 0; i < EPL; ++i) stf(x, idx[i], v[i]);
}

GPU_API int gpu_qk_norm_rope(int dtype, int head_dim, void* buf, int64_t row_stride, int q_off, int k_off, int hq,
                               int hk, const float* qw, const float* kw, float eps, const float* rope, int rope_mode,
                               int L, void* stream) {
    const int64_t warps = (int64_t)L * (hq + hk);
    const int threads = 256;
    const int blocks = (int)((warps * 32 + threads - 1) / threads);
    cudaStream_t st = (cudaStream_t)stream;
#define QKN(T, EPL) qk_norm_rope_kernel<T, EPL><<<blocks, threads, 0, st>>>((T*)buf, row_stride, q_off, k_off, hq, hk, qw, kw, eps, rope, rope_mode, L)
    if (dtype == DT_BF16 && head_dim == 128) QKN(bf16, 4);
    else if (dtype == DT_BF16 && head_dim == 64) QKN(bf16, 2);
    else if (dtype == DT_F32 && head_dim == 128) QKN(float, 4);
    else return -4;
#undef QKN
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Small exact attention, used for text encoders and the Anima LLM adapter (short sequences).
// grid (lq, hq), one thread per head-dim element. q: [lq, hq*d], k and v: [lk, hk*d], out: [lq, hq*d].
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void attn_small_kernel(const T* q, int64_t q_stride, const T* kp, const T* vp, int64_t kv_stride, int hq,
                                  int hk, int lk, T* out, int64_t out_stride, float scale, int causal, int d) {
    extern __shared__ float sm[];
    // blockDim.x is d rounded up to whole warps (the reductions need them), so threads past d only help with the scores
    float* qs = sm;       // d
    float* p = sm + d;    // lk
    __shared__ float red[33];
    const int i = blockIdx.x, h = blockIdx.y, e0 = threadIdx.x;
    const int kh = h / (hq / hk);
    if (e0 < d) qs[e0] = ldf(q, (int64_t)i * q_stride + h * d + e0);
    __syncthreads();
    const int n = causal ? i + 1 : lk;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const T* kr = kp + (int64_t)j * kv_stride + kh * d;
        float s = 0.f;
        for (int e = 0; e < d; ++e) s += qs[e] * ldf(kr, e);
        s *= scale;
        p[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = expf(p[j] - mx);
        p[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    if (e0 >= d) return;
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += p[j] * ldf(vp, (int64_t)j * kv_stride + kh * d + e0);
    stf(out, (int64_t)i * out_stride + h * d + e0, acc / sum);
}

GPU_API int gpu_attn_small(int dtype, const void* q, int64_t q_stride, const void* k, const void* v,
                             int64_t kv_stride, int hq, int hk, int d, int lq, int lk, void* out, int64_t out_stride,
                             float scale, int causal, void* stream) {
    dim3 grid(lq, hq);
    const size_t smem = ((size_t)d + lk) * sizeof(float);
    cudaStream_t st = (cudaStream_t)stream;
#define ATTN(T)                                                                                                 \
    if (smem > 48 * 1024) cudaFuncSetAttribute(attn_small_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
    attn_small_kernel<T><<<grid, (d + 31) / 32 * 32, smem, st>>>((const T*)q, q_stride, (const T*)k, (const T*)v, kv_stride, hq, hk, lk, \
                                                (T*)out, out_stride, scale, causal, d)
    if (dtype == DT_F32) { ATTN(float); }
    else if (dtype == DT_BF16) { ATTN(bf16); }
    else return -4;
#undef ATTN
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// INT8 ConvRot embedding rows -> float rows: x = H256 (q * s)   (H256 is its own inverse)
// ---------------------------------------------------------------------------------------------
__global__ void embed_dequant_kernel(const int8_t* q, const float* s, float* out, int K) {
    extern __shared__ float row[];
    const int64_t m = blockIdx.x;
    const float sc = s[m];
    for (int k = threadIdx.x; k < K; k += blockDim.x) row[k] = (float)q[m * K + k] * sc;
    __syncthreads();
    fht256_rows_smem(row, K);
    // ComfyUI returns the dequantized rows in the table's original dtype (bf16)
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[m * K + k] = rnd<bf16>(row[k]);
}

GPU_API int gpu_embed_dequant(const int8_t* q, const float* s, float* out, int M, int K, int rotate, void* stream) {
    if (!rotate) return -1;
    embed_dequant_kernel<<<M, 256, K * sizeof(float), (cudaStream_t)stream>>>(q, s, out, K);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Elementwise helpers (bf16 unless noted)
// ---------------------------------------------------------------------------------------------
__global__ void gelu_tanh_kernel(bf16* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16_rn(gelu_tanh_f(__bfloat162float(x[i])));
}
GPU_API int gpu_gelu_tanh_bf16(void* x, int64_t n, void* stream) {
    gelu_tanh_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((bf16*)x, n);
    KCHECK();
    return 0;
}

// x[r, c] = rnd(x[r, c] + rnd(gate[c] * y[r, c]))   (gate may be null)
__global__ void add_gated_kernel(bf16* x, int64_t x_stride, const bf16* y, int64_t y_stride, const bf16* gate,
                                 int rows, int cols) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    float v = __bfloat162float(y[(int64_t)r * y_stride + c]);
    if (gate) v = rnd<bf16>(__bfloat162float(gate[c]) * v);
    bf16* px = x + (int64_t)r * x_stride + c;
    *px = __float2bfloat16_rn(__bfloat162float(*px) + v);
}
GPU_API int gpu_add_gated_bf16(void* x, int64_t x_stride, const void* y, int64_t y_stride, const void* gate,
                                 int rows, int cols, void* stream) {
    const int64_t n = (int64_t)rows * cols;
    add_gated_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (bf16*)x, x_stride, (const bf16*)y, y_stride, (const bf16*)gate, rows, cols);
    KCHECK();
    return 0;
}

// out[i] = rnd(a[i] + rnd(b[i]))  with b float or bf16 (modulation vectors)
__global__ void add_vec_kernel(bf16* out, const bf16* a, const void* b, int b_is_f32, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float bv = b_is_f32 ? rnd<bf16>(((const float*)b)[i]) : __bfloat162float(((const bf16*)b)[i]);
    out[i] = __float2bfloat16_rn(__bfloat162float(a[i]) + bv);
}
GPU_API int gpu_add_vec_bf16(void* out, const void* a, const void* b, int b_is_f32, int n, void* stream) {
    add_vec_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((bf16*)out, (const bf16*)a, b, b_is_f32, n);
    KCHECK();
    return 0;
}

__global__ void f32_to_bf16_kernel(const float* in, bf16* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}
GPU_API int gpu_f32_to_bf16(const float* in, void* out, int64_t n, void* stream) {
    f32_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, (bf16*)out, n);
    KCHECK();
    return 0;
}

// GPU threads torch.randn splits a large tensor across (multiProcessorCount * blocks per SM * 256).
GPU_API int gpu_randn_threads() {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return -1;
    return p.multiProcessorCount * (p.maxThreadsPerMultiProcessor / 256) * 256;
}

__global__ void bf16_to_f32_kernel(const bf16* in, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}
GPU_API int gpu_bf16_to_f32(const void* in, float* out, int64_t n, void* stream) {
    bf16_to_f32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)in, out, n);
    KCHECK();
    return 0;
}

// out[i] = c[0] * x[0][i] + ... + c[nx-1] * x[nx-1][i]; out may alias any input. Every sampler
// update and CFG is one or two of these.
struct LincombArgs {
    const float* x[4];
    float c[4];
};
__global__ void lincomb_kernel(LincombArgs a, int nx, float* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.f;
    for (int j = 0; j < nx; ++j) v += a.c[j] * a.x[j][i];
    out[i] = v;
}
// RoPE on f32 rows of H heads of d elements, the first half of each head paired with the second half.
// rope: [L][d/2][(cos, sin)], for vision attention, which has no q/k norm.
__global__ void rope_rows_kernel(float* x, int64_t stride, const float* rope, int H, int d) {
    const int64_t l = blockIdx.x;
    const int h = blockIdx.y;
    float* row = x + l * stride + (int64_t)h * d;
    const float* r = rope + l * d;
    const int half = d / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        const float a = row[j], b = row[half + j], c = r[2 * j], s = r[2 * j + 1];
        row[j] = a * c - b * s;
        row[half + j] = b * c + a * s;
    }
}
GPU_API int gpu_rope_rows(float* x, int64_t stride, const float* rope, int L, int H, int d, void* stream) {
    rope_rows_kernel<<<dim3(L, H), 32, 0, (cudaStream_t)stream>>>(x, stride, rope, H, d);
    KCHECK();
    return 0;
}

// x = gelu(x) on f32, with torch's tanh approximation or the exact erf form.
__global__ void gelu_f32_kernel(float* x, int64_t n, int approximate) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = approximate ? gelu_tanh_f(x[i]) : gelu_f(x[i]);
}
GPU_API int gpu_gelu_f32(float* x, int64_t n, int approximate, void* stream) {
    gelu_f32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, n, approximate);
    KCHECK();
    return 0;
}

// The samples of a stride 2 conv from its stride 1 output: x [H, W, C] -> out [H/2, W/2, C], taking
// odd rows and columns, whose windows start on even positions.
__global__ void take_odd_kernel(const bf16* x, bf16* out, int H, int W, int C) {
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ow = W / 2;
    if (p >= (H / 2) * ow) return;
    const int64_t y = p / ow, xx = p % ow;
    const int64_t src = (((2 * y + 1) * W) + 2 * xx + 1) * C;
    for (int c = 0; c < C; ++c) out[p * C + c] = x[src + c];
}
GPU_API int gpu_take_odd(const void* x, void* out, int H, int W, int C, void* stream) {
    const int64_t n = (int64_t)(H / 2) * (W / 2);
    take_odd_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, (bf16*)out, H, W, C);
    KCHECK();
    return 0;
}

GPU_API int gpu_lincomb(float* out, int64_t n, int nx, const float* const* x, const float* c, void* stream) {
    if (nx < 1 || nx > 4) return -1;
    LincombArgs a{};
    for (int j = 0; j < nx; ++j) { a.x[j] = x[j]; a.c[j] = c[j]; }
    lincomb_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(a, nx, out, n);
    KCHECK();
    return 0;
}

// Krea2 txtfusion projector: out[l, d] = sum_n x[l*nl + n, d] * w[n]   (Linear(nl -> 1), no bias)
__global__ void layer_project_kernel(const bf16* x, const bf16* w, bf16* out, int L, int nl, int D) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)L * D) return;
    const int l = (int)(i / D), d = (int)(i % D);
    float acc = 0.f;
    for (int n = 0; n < nl; ++n) acc += __bfloat162float(x[((int64_t)l * nl + n) * D + d]) * __bfloat162float(w[n]);
    out[i] = __float2bfloat16_rn(acc);
}
GPU_API int gpu_layer_project(const void* x, const void* w, void* out, int L, int nl, int D, void* stream) {
    const int64_t n = (int64_t)L * D;
    layer_project_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const bf16*)x, (const bf16*)w, (bf16*)out, L, nl, D);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Krea2 patchify / unpatchify + Euler step
// latent: float [C, H, W]; tokens: bf16 [(H/p)*(W/p), C*p*p] in (c, ph, pw) order.
// ---------------------------------------------------------------------------------------------
__global__ void patchify_kernel(const float* lat, bf16* tok, int C, int H, int W, int p) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t)C * H * W;
    if (i >= n) return;
    const int c = (int)(i / ((int64_t)H * W));
    const int y = (int)((i / W) % H), xx = (int)(i % W);
    const int ty = y / p, tx = xx / p, py = y % p, px = xx % p;
    const int64_t t = (int64_t)ty * (W / p) + tx;
    const int f = (c * p + py) * p + px;
    tok[t * (C * p * p) + f] = __float2bfloat16_rn(lat[i]);
}
GPU_API int gpu_patchify(const float* lat, void* tok, int C, int H, int W, int p, void* stream) {
    const int64_t n = (int64_t)C * H * W;
    patchify_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lat, (bf16*)tok, C, H, W, p);
    KCHECK();
    return 0;
}

// x += v * dt using ComfyUI's arithmetic: denoised = x - v*sigma; d = (x - denoised)/sigma; x = x + d*dt
__global__ void unpatch_euler_kernel(const bf16* tok, int64_t tok_stride, float* lat, int C, int H, int W, int p,
                                     float sigma, float sigma_next, float* denoised_out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t)C * H * W;
    if (i >= n) return;
    const int c = (int)(i / ((int64_t)H * W));
    const int y = (int)((i / W) % H), xx = (int)(i % W);
    const int ty = y / p, tx = xx / p, py = y % p, px = xx % p;
    const int64_t t = (int64_t)ty * (W / p) + tx;
    const int f = (c * p + py) * p + px;
    const float v = __bfloat162float(tok[t * tok_stride + f]);
    const float x = lat[i];
    const float den = x - v * sigma;
    if (denoised_out) denoised_out[i] = den;
    const float d = (x - den) / sigma;
    lat[i] = x + d * (sigma_next - sigma);
}
GPU_API int gpu_unpatch_euler(const void* tok, int64_t tok_stride, float* lat, int C, int H, int W, int p,
                                float sigma, float sigma_next, float* denoised_out, void* stream) {
    const int64_t n = (int64_t)C * H * W;
    unpatch_euler_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const bf16*)tok, tok_stride, lat, C, H, W, p, sigma, sigma_next, denoised_out);
    KCHECK();
    return 0;
}

// Anima output tokens to denoised latent: den = x - v * sigma. Tokens are bf16 [(H/p)*(W/p), p*p*C]
// in (ph, pw, c) order (Cosmos unpatchify), unlike the (c, ph, pw) order of the input side.
__global__ void unpatch_denoised_kernel(const bf16* tok, int64_t tok_stride, const float* lat, float* den, int C, int H,
                                        int W, int p, float sigma) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t)C * H * W;
    if (i >= n) return;
    const int c = (int)(i / ((int64_t)H * W));
    const int y = (int)((i / W) % H), xx = (int)(i % W);
    const int ty = y / p, tx = xx / p, py = y % p, px = xx % p;
    const int64_t t = (int64_t)ty * (W / p) + tx;
    const int f = (py * p + px) * C + c;
    den[i] = lat[i] - __bfloat162float(tok[t * tok_stride + f]) * sigma;
}
GPU_API int gpu_unpatch_denoised(const void* tok, int64_t tok_stride, const float* lat, float* den, int C, int H,
                                   int W, int p, float sigma, void* stream) {
    const int64_t n = (int64_t)C * H * W;
    unpatch_denoised_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const bf16*)tok, tok_stride, lat, den, C, H, W, p, sigma);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Wan VAE decoder helpers (NHWC bf16)
// ---------------------------------------------------------------------------------------------
// latent float [C, H, W] -> NHWC bf16 with process_latent_out: x * std[c] + mean[c]
__global__ void vae_latent_in_kernel(const float* lat, const float* mean, const float* stdv, bf16* out, int C,
                                     int HW) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)C * HW) return;
    const int c = (int)(i / HW), s = (int)(i % HW);
    out[(int64_t)s * C + c] = __float2bfloat16_rn(lat[i] * stdv[c] + mean[c]);
}
GPU_API int gpu_vae_latent_in(const float* lat, const float* mean, const float* stdv, void* out, int C, int HW,
                                void* stream) {
    const int64_t n = (int64_t)C * HW;
    vae_latent_in_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lat, mean, stdv, (bf16*)out, C, HW);
    KCHECK();
    return 0;
}

// Wan RMS_norm: F.normalize(x, dim=C) * sqrt(C) * gamma, then optional SiLU. One warp per pixel.
template <int VPT>
__global__ void l2norm_silu_kernel(const bf16* x, bf16* out, const bf16* gamma, int64_t P, int C, int silu) {
    const int64_t pix = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (pix >= P) return;
    const bf16* xr = x + pix * C;
    float v[VPT];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int c = lane + 32 * i;
        v[i] = c < C ? __bfloat162float(xr[c]) : 0.f;
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float nrm = fmaxf(rnd<bf16>(sqrtf(ss)), 1e-12f);
    const float sc = sqrtf((float)C);
    bf16* o = out + pix * C;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int c = lane + 32 * i;
        if (c < C) {
            float y = rnd<bf16>(v[i] / nrm);
            y = rnd<bf16>(y * sc);
            y = rnd<bf16>(y * __bfloat162float(gamma[c]));
            if (silu) y = silu_f(y);
            o[c] = __float2bfloat16_rn(y);
        }
    }
}
GPU_API int gpu_l2norm_silu(const void* x, void* out, const void* gamma, int64_t P, int C, int silu, void* stream) {
    const int threads = 256;
    const unsigned blocks = (unsigned)((P * 32 + threads - 1) / threads);
    cudaStream_t st = (cudaStream_t)stream;
    if (C <= 128)
        l2norm_silu_kernel<4><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else if (C <= 256)
        l2norm_silu_kernel<8><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else if (C <= 512)
        l2norm_silu_kernel<16><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else if (C <= 768)
        l2norm_silu_kernel<24><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else if (C <= 1152)
        l2norm_silu_kernel<36><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else
        return -1;
    KCHECK();
    return 0;
}

// nearest 2x upsample, NHWC
__global__ void upsample2x_kernel(const bf16* x, bf16* out, int H, int W, int C) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = (int64_t)4 * H * W * C;
    if (i >= n) return;
    const int c = (int)(i % C);
    const int64_t p = i / C;
    const int ox = (int)(p % (2 * W)), oy = (int)(p / (2 * W));
    out[i] = x[((int64_t)(oy >> 1) * W + (ox >> 1)) * C + c];
}
GPU_API int gpu_upsample2x(const void* x, void* out, int H, int W, int C, void* stream) {
    const int64_t n = (int64_t)4 * H * W * C;
    upsample2x_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, (bf16*)out, H, W, C);
    KCHECK();
    return 0;
}

// Row softmax: in float [rows, n] (pre-scale applied here), out bf16 [rows, n]
__global__ void softmax_rows_kernel(const float* s, bf16* p, int n, float scale) {
    __shared__ float red[33];
    const int64_t r = blockIdx.x;
    const float* sr = s + r * n;
    bf16* pr = p + r * n;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, sr[j] * scale);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) sum += expf(sr[j] * scale - mx);
    sum = block_sum(sum, red);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) pr[j] = __float2bfloat16_rn(expf(sr[j] * scale - mx) * inv);
}
GPU_API int gpu_softmax_rows(const float* s, void* p, int rows, int n, float scale, void* stream) {
    softmax_rows_kernel<<<rows, 1024, 0, (cudaStream_t)stream>>>(s, (bf16*)p, n, scale);
    KCHECK();
    return 0;
}

// Final VAE output: NHWC bf16 [P, cstride] (first 3 channels) -> RGB8 like ComfyUI SaveImage
__global__ void to_rgb8_kernel(const bf16* x, int cstride, uint8_t* out, int64_t P) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * 3) return;
    const int64_t p = i / 3;
    const int c = (int)(i % 3);
    float v = __bfloat162float(x[p * cstride + c]);
    v = fminf(fmaxf((v + 1.f) / 2.f, 0.f), 1.f);
    v = fminf(fmaxf(255.f * v, 0.f), 255.f);
    out[i] = (uint8_t)v;
}
GPU_API int gpu_to_rgb8(const void* x, int cstride, uint8_t* out, int64_t P, void* stream) {
    to_rgb8_kernel<<<(unsigned)((P * 3 + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, cstride, out, P);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// MOSS Image
// ---------------------------------------------------------------------------------------------
// weight * (1 + scale) - 1: the norm weight folded into the scale of a fused rms_adaln.
__global__ void adaln_weight_kernel(const bf16* w, const bf16* sc, bf16* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = rnd<bf16>(1.f + __bfloat162float(sc[i]));
    out[i] = __float2bfloat16_rn(rnd<bf16>(__bfloat162float(w[i]) * a) - 1.f);
}
GPU_API int gpu_adaln_weight(const void* w, const void* sc, void* out, int n, void* stream) {
    adaln_weight_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((const bf16*)w, (const bf16*)sc, (bf16*)out, n);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Qwen3.5 GatedDeltaNet (fp32)
// ---------------------------------------------------------------------------------------------
// silu(conv1d(x)): causal depthwise FIR. x, out [L, C]; w [C, 4]
__global__ void delta_conv_kernel(const float* x, const bf16* w, float* out, int L, int C) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)L * C) return;
    const int t = (int)(i / C), c = (int)(i % C);
    const bf16* wc = w + (int64_t)c * 4;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int s = t + k - 3;
        if (s >= 0) acc += __bfloat162float(wc[k]) * x[(int64_t)s * C + c];
    }
    out[i] = silu_f(acc);
}
GPU_API int gpu_delta_conv(const float* x, const void* w, float* out, int L, int C, void* stream) {
    const int64_t n = (int64_t)L * C;
    delta_conv_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, (const bf16*)w, out, L, C);
    KCHECK();
    return 0;
}

// L2-normalizes the d-wide heads of q and k in place (query | key halves of a [L, 2*H*d] row).
__global__ void delta_normalize_kernel(float* x, int64_t stride, int H, int d) {
    __shared__ float red[33];
    const int64_t r = blockIdx.x;
    const int t = (int)(r / (2 * H)), h = (int)(r % (2 * H));
    float* v = x + (int64_t)t * stride + (int64_t)h * d;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) ss += v[i] * v[i];
    ss = block_sum(ss, red);
    const float n = fmaxf(sqrtf(ss), 1e-12f);
    for (int i = threadIdx.x; i < d; i += blockDim.x) v[i] /= n;
}

// The delta rule as the plain recursion its chunked form computes: one block per value head, one
// thread per state column, state in shared memory.
__global__ void delta_rule_kernel(const float* mix, const float* a, const float* b, const float* dt, const float* alog,
                                  float* out, int L, int Hv, int Hk, int d, float scale) {
    extern __shared__ float sm[];  // state [d][d] | k | q
    float* kq = sm + d * d;
    const int h = blockIdx.x, c = threadIdx.x;
    const int hk = h / (Hv / Hk);
    const int64_t kdim = (int64_t)Hk * d, row = 2 * kdim + (int64_t)Hv * d;
    for (int k = 0; k < d; ++k) sm[k * d + c] = 0.f;
    const float decay = -expf(alog[h]);
    for (int s = 0; s < L; ++s) {
        const float* m = mix + (int64_t)s * row;
        kq[c] = m[kdim + (int64_t)hk * d + c];
        kq[d + c] = m[(int64_t)hk * d + c] * scale;
        __syncthreads();
        const float aa = a[(int64_t)s * Hv + h] + dt[h];
        const float g = expf(decay * (aa > 20.f ? aa : log1pf(expf(aa))));
        const float beta = sigmoid_f(b[(int64_t)s * Hv + h]);
        float kvm = 0.f;
        for (int k = 0; k < d; ++k) {
            const float sv = sm[k * d + c] * g;
            sm[k * d + c] = sv;
            kvm = fmaf(kq[k], sv, kvm);
        }
        const float delta = (m[2 * kdim + (int64_t)h * d + c] - kvm) * beta;
        float o = 0.f;
        for (int k = 0; k < d; ++k) {
            const float sv = fmaf(kq[k], delta, sm[k * d + c]);
            sm[k * d + c] = sv;
            o = fmaf(kq[d + k], sv, o);
        }
        out[((int64_t)s * Hv + h) * d + c] = o;
        __syncthreads();
    }
}
// mix [L, q | k | v] is normalized in place. a, b [L, Hv]; dt, alog [Hv]; out [L, Hv, d].
GPU_API int gpu_delta_rule(float* mix, const float* a, const float* b, const float* dt, const float* alog, float* out,
                             int L, int Hv, int Hk, int d, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    const int64_t row = 2 * (int64_t)Hk * d + (int64_t)Hv * d;
    delta_normalize_kernel<<<L * 2 * Hk, d, 0, st>>>(mix, row, Hk, d);
    const size_t smem = (size_t)(d * d + 2 * d) * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(delta_rule_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    delta_rule_kernel<<<Hv, d, smem, st>>>(mix, a, b, dt, alog, out, L, Hv, Hk, d, 1.f / sqrtf((float)d));
    KCHECK();
    return 0;
}

__global__ void mul_silu_kernel(float* x, const float* z, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= silu_f(z[i]);
}
GPU_API int gpu_mul_silu(float* x, const float* z, int64_t n, void* stream) {
    mul_silu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, z, n);
    KCHECK();
    return 0;
}

// RMSNorm of d-wide heads, then split-half RoPE on their first 2*pairs dims. x [L, H, d] with rows
// `stride` elements apart, w [d], rope [L][pairs][(cos, sin)].
__global__ void partial_rope_kernel(float* x, int64_t stride, const float* w, float eps, const float* rope, int H, int d,
                                    int pairs) {
    __shared__ float red[33];
    const int t = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    float* v = x + (int64_t)t * stride + (int64_t)h * d;
    const float e = v[i];
    const float rstd = rsqrtf(block_sum(e * e, red) / (float)d + eps);
    if (i < pairs) {
        const float* r = rope + ((int64_t)t * pairs + i) * 2;
        const float a = e * rstd * w[i], b = v[i + pairs] * rstd * w[i + pairs];
        v[i] = a * r[0] - b * r[1];
        v[i + pairs] = b * r[0] + a * r[1];
    } else if (i >= 2 * pairs) {
        v[i] = e * rstd * w[i];
    }
}
GPU_API int gpu_partial_rope(float* x, int64_t stride, const float* w, float eps, const float* rope, int L, int H, int d,
                               int pairs, void* stream) {
    partial_rope_kernel<<<L * H, d, 0, (cudaStream_t)stream>>>(x, stride, w, eps, rope, H, d, pairs);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Flux 2 VAE decoder helpers (NHWC bf16)
// ---------------------------------------------------------------------------------------------
// latent float [64*2, H, W] -> z * s + m, unpacked to NHWC bf16 [2H, 2W, C/4]
__global__ void flux2_latent_in_kernel(const float* lat, const float* mean, const float* stdv, bf16* out, int C, int H, int W) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)C * H * W) return;
    const int cc = (int)(i / ((int64_t)H * W));
    const int y = (int)((i / W) % H), xx = (int)(i % W);
    const float v = rnd<bf16>(rnd<bf16>(rnd<bf16>(lat[i]) * stdv[cc]) + mean[cc]);
    const int c = cc / 4, py = (cc >> 1) & 1, px = cc & 1;
    const int oy = 2 * y + py, ox = 2 * xx + px;
    out[(((int64_t)oy * 2 * W + ox) * (C / 4)) + c] = __float2bfloat16_rn(v);
}
GPU_API int gpu_flux2_latent_in(const float* lat, const float* mean, const float* stdv, void* out, int C, int H, int W,
                                  void* stream) {
    const int64_t n = (int64_t)C * H * W;
    flux2_latent_in_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lat, mean, stdv, (bf16*)out, C, H, W);
    KCHECK();
    return 0;
}

// GroupNorm statistics of x [P, C]: stats[g] = (mean, 1 / sqrt(var + eps))
__global__ void group_stats_kernel(const bf16* x, float2* stats, int64_t P, int C, int groups, float eps) {
    __shared__ float red[33];
    const int g = blockIdx.x, per = C / groups;
    float sum = 0.f, ss = 0.f;
    for (int64_t p = threadIdx.x; p < P; p += blockDim.x) {
        const bf16* r = x + p * C + g * per;
#pragma unroll 4
        for (int c = 0; c < per; ++c) {
            const float v = __bfloat162float(r[c]);
            sum += v;
            ss += v * v;
        }
    }
    const float nv = 1.f / ((float)P * per);
    const float mean = block_sum(sum, red) * nv;
    const float var = block_sum(ss, red) * nv - mean * mean;
    if (threadIdx.x == 0) stats[g] = make_float2(mean, rsqrtf(fmaxf(var, 0.f) + eps));
}

__global__ void group_apply_kernel(const bf16* x, bf16* out, const bf16* w, const bf16* b, const float2* stats, int64_t n, int C,
                                   int per, int silu) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int)(i % C);
    const float2 st = stats[c / per];
    const float y = (__bfloat162float(x[i]) - st.x) * st.y * __bfloat162float(w[c]) + __bfloat162float(b[c]);
    out[i] = __float2bfloat16_rn(silu ? silu_f(y) : y);
}

// GroupNorm, then optional SiLU. x, out [P, C]; stats is 8 * groups bytes of scratch.
GPU_API int gpu_group_norm(const void* x, void* out, const void* w, const void* b, void* stats, int64_t P, int C, int groups,
                             float eps, int silu, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    group_stats_kernel<<<groups, 1024, 0, st>>>((const bf16*)x, (float2*)stats, P, C, groups, eps);
    const int64_t n = P * C;
    group_apply_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)w, (const bf16*)b,
                                                                     (const float2*)stats, n, C, C / groups, silu);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// ConvRot int8 weights <-> bf16 or f32, to bake LoRAs. The same arithmetic as comfy-kitchen's
// dequantize_int8_convrot_weight and quantize_int8_rowwise_convrot64 with stochastic rounding.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float lora_load(float v) { return v; }
__device__ __forceinline__ float lora_load(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void lora_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void lora_store(bf16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ float noisy_sum(float scaled, float noise, float*) { return scaled + noise; }
__device__ __forceinline__ float noisy_sum(float scaled, float noise, bf16*) {
    return __bfloat162float(__hadd(__float2bfloat16_rn(scaled), __float2bfloat16_rn(noise)));
}

template <typename T>
__global__ void convrot_dequant_kernel(const int8_t* q, const float* s, T* out, int K) {
    extern __shared__ float row[];
    const int64_t m = blockIdx.x;
    const float sc = s[m];
    for (int k = threadIdx.x; k < K; k += blockDim.x) row[k] = (float)q[m * K + k] * sc;
    __syncthreads();
    fht256_rows_smem(row, K);
    for (int k = threadIdx.x; k < K; k += blockDim.x) lora_store(&out[m * K + k], row[k]);
}

__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) {
    const uint32_t state = x * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

template <typename T>
__global__ void convrot_requant_kernel(const T* x, int8_t* q, float* scales, int K, uint64_t seed) {
    extern __shared__ float smem[];
    float* red = smem;
    float* row = smem + 33;
    const int64_t m = blockIdx.x;
    for (int k = threadIdx.x; k < K; k += blockDim.x) row[k] = lora_load(x[m * K + k]);
    __syncthreads();
    fht256_rows_smem(row, K);
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(row[k]));
    amax = block_max(amax, red);
    const float scale = fmaxf(fminf(amax, finite_max<T>()) * (1.f / 127.f), 1.0e-30f);
    if (threadIdx.x == 0) scales[m] = scale;
    const float scale_t = rnd<T>(scale);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float scaled = rnd<T>(rnd<T>(row[k]) / scale_t);
        const uint64_t key = (uint64_t)(m * K + k) + seed;
        const uint32_t folded = (uint32_t)key ^ (uint32_t)(key >> 32);
        const float v = floorf(noisy_sum(scaled, (float)(pcg_hash(folded) >> 8) * 0x1.0p-24f, (T*)nullptr));
        q[m * K + k] = (int8_t)fminf(127.f, fmaxf(-128.f, v));
    }
}

static int convrot_threads(int K) { return K >= 6144 ? 1024 : 256; }

// out = the weight in its original basis; q [N, K], s [N]
GPU_API int gpu_convrot_dequant(int dtype, const void* q, const void* s, void* out, int64_t N, int K, void* stream) {
    const int smem = K * sizeof(float);
    cudaFuncSetAttribute(convrot_dequant_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(convrot_dequant_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (dtype == DT_F32) {
        convrot_dequant_kernel<float><<<(unsigned)N, convrot_threads(K), smem, (cudaStream_t)stream>>>((const int8_t*)q, (const float*)s, (float*)out, K);
    } else {
        convrot_dequant_kernel<bf16><<<(unsigned)N, convrot_threads(K), smem, (cudaStream_t)stream>>>((const int8_t*)q, (const float*)s, (bf16*)out, K);
    }
    KCHECK();
    return 0;
}

// x [N, K] in its original basis -> rotated q [N, K] and row scales s [N]
GPU_API int gpu_convrot_requant(int dtype, const void* x, void* q, void* s, int64_t N, int K, uint64_t seed, void* stream) {
    const int smem = (K + 33) * sizeof(float);
    cudaFuncSetAttribute(convrot_requant_kernel<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(convrot_requant_kernel<bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (dtype == DT_F32) {
        convrot_requant_kernel<float><<<(unsigned)N, convrot_threads(K), smem, (cudaStream_t)stream>>>((const float*)x, (int8_t*)q, (float*)s, K, seed);
    } else {
        convrot_requant_kernel<bf16><<<(unsigned)N, convrot_threads(K), smem, (cudaStream_t)stream>>>((const bf16*)x, (int8_t*)q, (float*)s, K, seed);
    }
    KCHECK();
    return 0;
}

template <typename T>
__global__ void lora_add_kernel(T* w, const T* d, float scale, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lora_store(&w[i], lora_load(w[i]) + rnd<T>(lora_load(d[i]) * scale));
}

// w += scale * d, every intermediate in dtype as torch computes it
GPU_API int gpu_lora_add(int dtype, void* w, const void* d, float scale, int64_t n, void* stream) {
    if (dtype == DT_F32) {
        lora_add_kernel<float><<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((float*)w, (const float*)d, scale, n);
    } else {
        lora_add_kernel<bf16><<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((bf16*)w, (const bf16*)d, scale, n);
    }
    KCHECK();
    return 0;
}
