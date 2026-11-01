// Custom kernels for the krea2 pipeline (DiT, Qwen3-VL text encoder, Wan VAE decoder).
//
// Where ComfyUI materializes an intermediate tensor in the activation dtype (bf16 for the DiT),
// the kernels round to that dtype at the same point (rnd<T>) so results track ComfyUI closely.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Row prologue -> optional ConvRot rotation -> per-row int8 quantization.
// Produces the A operand (+ per-row scales) of the fused int8 GEMM.
// ---------------------------------------------------------------------------------------------
enum { PRO_PLAIN = 0, PRO_RMS = 1, PRO_GATE = 2, PRO_SWIGLU = 3 };

struct QuantParams {
    const void* x;       // PLAIN/RMS: input row; GATE: attention output; SWIGLU: [gate | up] row
    int64_t x_stride;    // elements between rows of x
    const void* g;       // GATE: gate logits
    int64_t g_stride;
    int64_t up_offset;   // SWIGLU: column of `up` inside the x row
    const float* norm_w; // RMS weight (zero-centered offset already applied); null = 1
    float eps;
    const void* mod_scale;  // RMS modulation (T): v = (1 + scale) * v + shift; null = none
    const void* mod_shift;
    int8_t* q;           // [M, K]
    float* scales;       // [M]
    int K;
    int rotate;
    float* out_f;        // if set: write the (rotated) fp32 row here instead of quantizing
};

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
        } else {
            load8(x + c, v[gi]);
            if (PRO == PRO_RMS) {
#pragma unroll
                for (int j = 0; j < 8; ++j) ss += v[gi][j] * v[gi][j];
            }
        }
    }
    if (PRO == PRO_RMS) {
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

KREA_API int krea_quant_rows(int dtype, int prologue, const void* x, int64_t x_stride, const void* g,
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
        default: return -3;                                            \
    }
    if (dtype == DT_BF16) { QDISPATCH(bf16) }
    if (dtype == DT_F32) { QDISPATCH(float) }
#undef QDISPATCH
    return -4;
}

// Same prologues + ConvRot rotation, fp32 output (weight-only int8 path: x_rot @ (q*s)^T).
KREA_API int krea_rotate_rows_f32(int prologue, const float* x, int64_t x_stride, int64_t up_offset,
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
KREA_API int krea_dequant_i8(const int8_t* q, const float* s, float* out, int64_t N, int64_t K, void* stream) {
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
    if (PRO == PRO_RMS) {
        float ss = 0.f;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float v = ldf(x, k);
            ss += v * v;
        }
        ss = block_sum(ss, red);
        const float rstd = rsqrtf(ss / (float)K + p.eps);
        const T* ms = (const T*)p.mod_scale;
        const T* sh = (const T*)p.mod_shift;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            float n = ldf(x, k) * rstd;
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
    } else {
        for (int k = threadIdx.x; k < K; k += blockDim.x) stf(o, k, ldf(x, k));
    }
}

KREA_API int krea_prologue_rows(int dtype, int prologue, const void* x, int64_t x_stride, const void* g,
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
        default: return -3;                                                                                \
    }
    if (dtype == DT_BF16) { PDISPATCH(bf16) }
    else if (dtype == DT_F32) { PDISPATCH(float) }
    else return -4;
#undef PDISPATCH
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Per-head RMSNorm of q and k (head dim 128) + rotary embedding, in place.
// rope_mode 1: interleaved pairs (Flux/Krea2 EmbedND), 2: split-half (Llama/Qwen).
// rope table: [L][64] x (cos, sin) float.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void qk_norm_rope_kernel(T* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk,
                                    const float* qw, const float* kw, float eps, const float* rope,
                                    int rope_mode, int L) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int heads = hq + hk;
    if (warp >= L * heads) return;
    const int t = warp / heads, h = warp % heads;
    const bool isq = h < hq;
    T* x = buf + (int64_t)t * row_stride + (isq ? q_off + h * 128 : k_off + (h - hq) * 128);
    const float* w = isq ? qw : kw;
    // element indices owned by this lane
    int idx[4];
    if (rope_mode == 2) {
        idx[0] = 2 * lane; idx[1] = 2 * lane + 1; idx[2] = 64 + 2 * lane; idx[3] = 65 + 2 * lane;
    } else {
        idx[0] = 4 * lane; idx[1] = 4 * lane + 1; idx[2] = 4 * lane + 2; idx[3] = 4 * lane + 3;
    }
    float v[4];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = ldf(x, idx[i]);
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / 128.f + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = rnd<T>(v[i] * rstd * (w ? w[idx[i]] : 1.f));
    if (rope && rope_mode == 1) {
        const float* r = rope + (int64_t)t * 128;
#pragma unroll
        for (int pi = 0; pi < 2; ++pi) {
            const int j = 2 * lane + pi;
            const float c = r[2 * j], s = r[2 * j + 1];
            const float a = v[2 * pi], b = v[2 * pi + 1];
            v[2 * pi] = c * a - s * b;
            v[2 * pi + 1] = s * a + c * b;
        }
    } else if (rope && rope_mode == 2) {
        const float* r = rope + (int64_t)t * 128;
#pragma unroll
        for (int pi = 0; pi < 2; ++pi) {
            const int j = 2 * lane + pi;
            const float c = r[2 * j], s = r[2 * j + 1];
            const float a = v[pi], b = v[2 + pi];
            v[pi] = a * c - b * s;
            v[2 + pi] = b * c + a * s;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) stf(x, idx[i], v[i]);
}

KREA_API int krea_qk_norm_rope(int dtype, void* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk,
                               const float* qw, const float* kw, float eps, const float* rope, int rope_mode,
                               int L, void* stream) {
    const int64_t warps = (int64_t)L * (hq + hk);
    const int threads = 256;
    const int blocks = (int)((warps * 32 + threads - 1) / threads);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_BF16)
        qk_norm_rope_kernel<bf16><<<blocks, threads, 0, st>>>((bf16*)buf, row_stride, q_off, k_off, hq, hk, qw, kw, eps, rope, rope_mode, L);
    else if (dtype == DT_F32)
        qk_norm_rope_kernel<float><<<blocks, threads, 0, st>>>((float*)buf, row_stride, q_off, k_off, hq, hk, qw, kw, eps, rope, rope_mode, L);
    else
        return -4;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Small exact attention (fp32), used for the text encoder (causal, GQA, short sequences).
// grid (L, hq), block 128 (one thread per head-dim element). out: [L, hq*128].
// ---------------------------------------------------------------------------------------------
__global__ void attn_small_f32_kernel(const float* qkv, int64_t row_stride, int q_off, int k_off, int v_off,
                                      int hq, int hk, int L, float* out, int64_t out_stride, float scale,
                                      int causal) {
    extern __shared__ float sm[];
    float* qs = sm;          // 128
    float* p = sm + 128;     // L
    __shared__ float red[33];
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int kh = h / (hq / hk);
    qs[d] = qkv[(int64_t)i * row_stride + q_off + h * 128 + d];
    __syncthreads();
    const int n = causal ? i + 1 : L;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float* kr = qkv + (int64_t)j * row_stride + k_off + kh * 128;
        float s = 0.f;
        for (int e = 0; e < 128; ++e) s += qs[e] * kr[e];
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
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += p[j] * qkv[(int64_t)j * row_stride + v_off + kh * 128 + d];
    out[(int64_t)i * out_stride + h * 128 + d] = acc / sum;
}

KREA_API int krea_attn_small_f32(const float* qkv, int64_t row_stride, int q_off, int k_off, int v_off, int hq,
                                 int hk, int L, float* out, int64_t out_stride, float scale, int causal,
                                 void* stream) {
    dim3 grid(L, hq);
    const size_t smem = (128 + (size_t)L) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(attn_small_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    attn_small_f32_kernel<<<grid, 128, smem, (cudaStream_t)stream>>>(qkv, row_stride, q_off, k_off, v_off, hq, hk, L,
                                                                     out, out_stride, scale, causal);
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

KREA_API int krea_embed_dequant(const int8_t* q, const float* s, float* out, int M, int K, int rotate, void* stream) {
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
KREA_API int krea_gelu_tanh_bf16(void* x, int64_t n, void* stream) {
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
KREA_API int krea_add_gated_bf16(void* x, int64_t x_stride, const void* y, int64_t y_stride, const void* gate,
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
KREA_API int krea_add_vec_bf16(void* out, const void* a, const void* b, int b_is_f32, int n, void* stream) {
    add_vec_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((bf16*)out, (const bf16*)a, b, b_is_f32, n);
    KCHECK();
    return 0;
}

__global__ void f32_to_bf16_kernel(const float* in, bf16* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}
KREA_API int krea_f32_to_bf16(const float* in, void* out, int64_t n, void* stream) {
    f32_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, (bf16*)out, n);
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
KREA_API int krea_layer_project(const void* x, const void* w, void* out, int L, int nl, int D, void* stream) {
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
KREA_API int krea_patchify(const float* lat, void* tok, int C, int H, int W, int p, void* stream) {
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
KREA_API int krea_unpatch_euler(const void* tok, int64_t tok_stride, float* lat, int C, int H, int W, int p,
                                float sigma, float sigma_next, float* denoised_out, void* stream) {
    const int64_t n = (int64_t)C * H * W;
    unpatch_euler_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const bf16*)tok, tok_stride, lat, C, H, W, p, sigma, sigma_next, denoised_out);
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
KREA_API int krea_vae_latent_in(const float* lat, const float* mean, const float* stdv, void* out, int C, int HW,
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
KREA_API int krea_l2norm_silu(const void* x, void* out, const void* gamma, int64_t P, int C, int silu, void* stream) {
    const int threads = 256;
    const unsigned blocks = (unsigned)((P * 32 + threads - 1) / threads);
    cudaStream_t st = (cudaStream_t)stream;
    if (C <= 128)
        l2norm_silu_kernel<4><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else if (C <= 256)
        l2norm_silu_kernel<8><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
    else if (C <= 512)
        l2norm_silu_kernel<16><<<blocks, threads, 0, st>>>((const bf16*)x, (bf16*)out, (const bf16*)gamma, P, C, silu);
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
KREA_API int krea_upsample2x(const void* x, void* out, int H, int W, int C, void* stream) {
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
KREA_API int krea_softmax_rows(const float* s, void* p, int rows, int n, float scale, void* stream) {
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
KREA_API int krea_to_rgb8(const void* x, int cstride, uint8_t* out, int64_t P, void* stream) {
    to_rgb8_kernel<<<(unsigned)((P * 3 + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, cstride, out, P);
    KCHECK();
    return 0;
}
