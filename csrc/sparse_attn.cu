// Block-sparse attention: Sol-Attn and SLA-style top-k over comfy-kitchen's kernels (vendor/sage/sol_attn*.cu), as
// ComfyUI's BlockSparseAttention node runs them. q/k/v are post-rope bf16 [T, H, 128] with any row and head stride,
// batch 1; the output is contiguous [T, H, 128].
//
// The top-k threshold replicates comfy-kitchen's `_topk_threshold` (backends/cuda/__init__.py): block means of q and k,
// k centred over blocks, both quantized to int8 per block, and the k-th largest pooled score of each query block.
#include "common.cuh"

#include <cmath>
#include <exception>

extern "C" int sol_attn_plan(int batch, int seq_len, int num_heads, int n_tok, int64_t* out, int cap);
extern "C" void launch_sol_attn(const void* q, const void* k, const void* v, void* out, void* workspace, int batch, int seq_len,
                                int num_heads, int head_dim, int elem, float tau, float scale, const void* key_bias,
                                const void* ext_threshold, const void* blen, int tail, int sink_start, int sink_end,
                                int sink_q_start, int sink_q_end, int64_t qs_b, int64_t qs_t, int64_t qs_h, int64_t ks_b,
                                int64_t ks_t, int64_t ks_h, int64_t vs_b, int64_t vs_t, int64_t vs_h, int n_tok,
                                cudaStream_t stream);

namespace {

constexpr int HD = 128;
constexpr int BLK = 64;

bool valid_shape(int t, int h, int n_tok) {
    return t > 0 && h > 0 && (t + BLK - 1) / BLK <= 65535 && h <= 65535 && n_tok >= 0 && n_tok <= 256 && n_tok % BLK == 0;
}

// Block means of q and k over each 64-row block (the last one over its rows): [H, NB, 128] f32.
__global__ void sol_means(const bf16* q, const bf16* k, int64_t rs, int64_t hs, int t, int nb, float* qm, float* km) {
    const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int r0 = b * BLK, r1 = min(t, r0 + BLK);
    float sq = 0.f, sk = 0.f;
    for (int r = r0; r < r1; ++r) {
        sq += __bfloat162float(q[r * rs + h * hs + d]);
        sk += __bfloat162float(k[r * rs + h * hs + d]);
    }
    const float n = (float)(r1 - r0);
    const int64_t o = ((int64_t)h * nb + b) * HD + d;
    qm[o] = sq / n;
    km[o] = sk / n;
}

// The block means of k, centred by their mean over blocks.
__global__ void sol_center(float* km, int nb) {
    const int h = blockIdx.x, d = threadIdx.x;
    float* p = km + (int64_t)h * nb * HD + d;
    float s = 0.f;
    for (int b = 0; b < nb; ++b) s += p[(int64_t)b * HD];
    const float mean = s / (float)nb;
    for (int b = 0; b < nb; ++b) p[(int64_t)b * HD] -= mean;
}

// x / (absmax / 127) rounded half to even into int8, one 128-wide row per CTA, as `torch.round(x / scale).clamp_(-127, 127)`.
__global__ void sol_quant(const float* x, int8_t* x8, float* xs, float floor_scale) {
    __shared__ float red[33];
    const int64_t row = (int64_t)blockIdx.y * gridDim.x + blockIdx.x;
    const float v = x[row * HD + threadIdx.x];
    const float sc = fmaxf(block_max(fabsf(v), red) / 127.0f, floor_scale);
    x8[row * HD + threadIdx.x] = (int8_t)fminf(fmaxf(rintf(__fdiv_rn(v, sc)), -127.f), 127.f);
    if (threadIdx.x == 0) xs[row] = sc;
}

__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key_value(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// One query block per CTA: its pooled score against every key block (sinks -inf), then the kk-th largest, backed off
// a few ulps as comfy-kitchen does.
__global__ void sol_topk(const int8_t* c8, const float* cs, const int8_t* k8, const float* ks, int nb, int kk, float log2s,
                         int s0, int s1, float* thr) {
    extern __shared__ uint32_t keys[];
    __shared__ int c_row[HD / 4];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t sel[2];
    const int qb = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int64_t qrow = (int64_t)h * nb + qb;
    if (kk == 0) {
        if (tid == 0) thr[qrow] = INFINITY;
        return;
    }
    if (tid < HD / 4) c_row[tid] = reinterpret_cast<const int*>(c8 + qrow * HD)[tid];
    __syncthreads();
    const float row_scale = cs[qrow] * log2s;
    for (int j = tid; j < nb; j += blockDim.x) {
        float s;
        if (j >= s0 && j < s1) {
            s = -INFINITY;
        } else {
            const int4* kr = reinterpret_cast<const int4*>(k8 + ((int64_t)h * nb + j) * HD);
            int acc = 0;
#pragma unroll
            for (int w = 0; w < HD / 16; ++w) {
                const int4 v = kr[w];
                acc = __dp4a(c_row[w * 4 + 0], v.x, acc);
                acc = __dp4a(c_row[w * 4 + 1], v.y, acc);
                acc = __dp4a(c_row[w * 4 + 2], v.z, acc);
                acc = __dp4a(c_row[w * 4 + 3], v.w, acc);
            }
            s = __fmul_rn(__fmul_rn((float)acc, row_scale), ks[(int64_t)h * nb + j]);
        }
        keys[j] = order_key(s);
    }
    // radix select of the kk-th largest key, 8 bits at a time from the top
    uint32_t prefix = 0, mask = 0;
    int want = kk;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int j = tid; j < nb; j += blockDim.x) {
            const uint32_t key = keys[j];
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, bin = 255;
            for (; bin > 0; --bin) {
                if (acc + (int)hist[bin] >= want) break;
                acc += hist[bin];
            }
            sel[0] = bin;
            sel[1] = want - acc;
        }
        __syncthreads();
        prefix |= sel[0] << shift;
        mask |= 255u << shift;
        want = (int)sel[1];
        __syncthreads();
    }
    if (tid == 0) {
        const float kth = key_value(prefix);
        thr[qrow] = kth - fabsf(kth) * 1e-5f;
    }
}

}  // namespace

// Workspace bytes for gpu_sol_attn, or -1 for a shape the kernels can't take.
GPU_API int64_t gpu_sol_attn_workspace(int t, int h, int n_tok) {
    if (!valid_shape(t, h, n_tok)) return -1;
    int64_t v[48];
    const int n = sol_attn_plan(1, t, h, n_tok, v, 48);
    if (n > 48) return -1;
    return v[n - 1];
}

// Scratch bytes for gpu_sol_topk_threshold.
GPU_API int64_t gpu_sol_topk_workspace(int t, int h) {
    const int64_t nb = (t + BLK - 1) / BLK;
    return 2 * h * nb * HD * 4 + 2 * h * nb * HD + 2 * h * nb * 4 + 1024;
}

// Per-query-block thresholds [H, NB] f32 that keep `kk` key blocks besides the sinks [s0, s1).
GPU_API int gpu_sol_topk_threshold(const void* q, const void* k, int64_t rs, int64_t hs, int t, int h, int kk, float scale,
                                   int s0, int s1, float* thr, void* scratch, void* stream) {
    const int nb = (t + BLK - 1) / BLK;
    const size_t smem = (size_t)nb * 4;
    if (smem > 96 * 1024) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    char* w = (char*)scratch;
    const size_t fl = (size_t)h * nb * HD * 4, i8 = (size_t)h * nb * HD, sc = (size_t)h * nb * 4;
    float* qm = (float*)w;
    float* km = (float*)(w + fl);
    int8_t* c8 = (int8_t*)(w + 2 * fl);
    int8_t* k8 = (int8_t*)(w + 2 * fl + i8);
    float* cs = (float*)(w + 2 * fl + 2 * i8);
    float* ks = (float*)(w + 2 * fl + 2 * i8 + sc);
    sol_means<<<dim3(nb, h), HD, 0, st>>>((const bf16*)q, (const bf16*)k, rs, hs, t, nb, qm, km);
    sol_center<<<h, HD, 0, st>>>(km, nb);
    sol_quant<<<dim3(nb, h), HD, 0, st>>>(qm, c8, cs, 1e-8f);
    sol_quant<<<dim3(nb, h), HD, 0, st>>>(km, k8, ks, 1e-12f);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(sol_topk, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        configured = true;
    }
    sol_topk<<<dim3(nb, h), 256, smem, st>>>(c8, cs, k8, ks, nb, kk, scale * 1.4426950408889634f, s0, s1, thr);
    KCHECK();
    return 0;
}

// Sol-Attn over post-rope q/k/v (B = 1). `threshold` (from gpu_sol_topk_threshold) selects top-k routing instead of
// tau's. Sinks: key blocks [s0, s1) are exact for every query, query blocks [q0, q1) attend everything exactly.
GPU_API int gpu_sol_attn(const void* q, const void* k, const void* v, void* out, void* ws, int t, int h, int64_t rs,
                         int64_t hs, float tau, float scale, const float* threshold, int s0, int s1, int q0, int q1,
                         int n_tok, void* stream) {
    if (!valid_shape(t, h, n_tok) || rs % 8 || hs % 8 || (uintptr_t)q % 16 || (uintptr_t)k % 16 || (uintptr_t)v % 16) return -1;
    try {
        launch_sol_attn(q, k, v, out, ws, 1, t, h, HD, 0, tau, scale, nullptr, threshold, nullptr, 1, s0, s1, q0, q1,
                        rs * t, rs, hs, rs * t, rs, hs, rs * t, rs, hs, n_tok, (cudaStream_t)stream);
    } catch (const std::exception&) {
        return -2;
    }
    KCHECK();
    return 0;
}
