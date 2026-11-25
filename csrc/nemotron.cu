// Nemotron-Labs-Image: kernels for decoding the Nemotron-H hybrid (Mamba-2, attention and MoE layers) a few rows at a
// time, and for its image decoder. Activations are f32; weights bf16, or nvfp4 for the routed experts.
#include "common.cuh"
#include <cuda_fp8.h>

namespace {

__device__ __forceinline__ float fp8_e4m3(uint8_t v) { return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)); }
__device__ __constant__ float kFp4[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float relu2(float v) {
    v = fmaxf(v, 0.f);
    return v * v;
}

// ---------------------------------------------------------------------------------------------
// out[r, n] (+)= sum_k W[n, k] x[r, k] for R rows: a warp per output column, reading its weight row once for all rows.
// K a multiple of 8, rows of W and x 16-byte aligned. act 1 squares the ReLU (relu2).
// ---------------------------------------------------------------------------------------------
template <int R>
__global__ void __launch_bounds__(256) gemv_bf16_kernel(const bf16* __restrict__ W, int64_t ldw, const float* __restrict__ x, int64_t ldx,
                                                        float* out, int64_t ldo, int N, int K, int accumulate, int act) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const bf16* w = W + (int64_t)n * ldw;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int k = lane * 8; k < K; k += 256) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(w + k));
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
        float wf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(p[i]);
            wf[2 * i] = f.x;
            wf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float4 a = *reinterpret_cast<const float4*>(x + r * ldx + k);
            const float4 b = *reinterpret_cast<const float4*>(x + r * ldx + k + 4);
            acc[r] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = warp_sum(acc[r]);
        if (lane == 0) {
            if (act == 1) v = relu2(v);
            float* o = out + r * ldo + n;
            *o = accumulate ? *o + v : v;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// nvfp4 GEMV over a list of jobs, each one expert's weight and one row: out[job.orow, n] = act(W[n, :] . x[job.xrow, :]).
// W: U8 [N, K/2], element 2i in the low nibble; scales F8_E4M3 [N, K/16] row major; value = e2m1 * scale * scale2.
// ---------------------------------------------------------------------------------------------
struct Fp4Job {
    const uint8_t* q;
    const uint8_t* sc;
    float scale2;
    int xrow;
    int orow;
    int pad;
};

__global__ void __launch_bounds__(256) gemv_fp4_kernel(const Fp4Job* __restrict__ jobs, const float* __restrict__ x, int64_t ldx, float* out,
                                                       int64_t ldo, int N, int K, int act) {
    extern __shared__ float xs[];
    __shared__ float lut[16];
    const Fp4Job j = jobs[blockIdx.y];
    if (threadIdx.x < 16) lut[threadIdx.x] = kFp4[threadIdx.x];
    for (int k = threadIdx.x; k < K; k += 256) xs[k] = x[(int64_t)j.xrow * ldx + k];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const uint8_t* q = j.q + (int64_t)n * (K / 2);
    const uint8_t* sc = j.sc + (int64_t)n * (K / 16);
    float acc = 0.f;
    // a 16-byte load holds 32 values, two scale blocks
    for (int c = lane; c < K / 32; c += 32) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(q + c * 16));
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
        const float* xv = xs + c * 32;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const float s = fp8_e4m3(sc[c * 2 + half]);
            float part = 0.f;
#pragma unroll
            for (int wi = 0; wi < 2; ++wi) {
                const uint32_t wd = words[half * 2 + wi];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t byte = (wd >> (b * 8)) & 0xFF;
                    const int e = half * 16 + wi * 8 + b * 2;
                    part += lut[byte & 15] * xv[e] + lut[byte >> 4] * xv[e + 1];
                }
            }
            acc += part * s;
        }
    }
    float v = warp_sum(acc);
    if (lane == 0) {
        v *= j.scale2;
        if (act == 1) v = relu2(v);
        out[(int64_t)j.orow * ldo + n] = v;
    }
}

// ---------------------------------------------------------------------------------------------
// Mamba-2's depthwise causal conv1d (kernel 4) over T new tokens of each sequence, then SiLU. `state` keeps each
// channel's last three inputs, oldest first. Rows of `in` and `out` are sequence-major: s * T + t.
// ---------------------------------------------------------------------------------------------
__global__ void mamba_conv_kernel(const float* in, int64_t ld_in, float* state, const float* w, const float* b, float* out, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, s = blockIdx.y;
    if (c >= C) return;
    float* st = state + ((int64_t)s * C + c) * 3;
    float h0 = st[0], h1 = st[1], h2 = st[2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], bias = b[c];
    for (int t = 0; t < T; ++t) {
        const int64_t row = (int64_t)s * T + t;
        const float xin = in[row * ld_in + c];
        const float y = bias + w0 * h0 + w1 * h1 + w2 * h2 + w3 * xin;
        out[row * C + c] = silu_f(y);
        h0 = h1;
        h1 = h2;
        h2 = xin;
    }
    st[0] = h0;
    st[1] = h1;
    st[2] = h2;
}

// ---------------------------------------------------------------------------------------------
// Mamba-2's selective scan, one token at a time, for a block of one head of one sequence. P = 64 head channels,
// N = 128 state values; a thread holds 32 state values of one channel. xbc: [rows, H*P + 2*G*N] after the conv;
// dt raw at column dt_col of `zx`. y = C . state + D x.
// ---------------------------------------------------------------------------------------------
constexpr int MP = 64, MN = 128;

__global__ void __launch_bounds__(256) mamba_scan_kernel(const float* xbc, int64_t ld_xbc, const float* zx, int64_t ldzx, int dt_col, float* ssm,
                                                         const float* A_log, const float* Dp, const float* dt_bias, float* y, int64_t ldy, int T,
                                                         int H, int G) {
    __shared__ float Bs[MN], Cs[MN];
    const int h = blockIdx.x, s = blockIdx.y, g = h / (H / G);
    const int p = threadIdx.x >> 2, nq = threadIdx.x & 3;
    float* st = ssm + (((int64_t)s * H + h) * MP + p) * MN + nq * 32;
    float state[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = st[i];
    const float A = -expf(A_log[h]), D = Dp[h], bias = dt_bias[h];
    const int bcol = H * MP + g * MN, ccol = H * MP + G * MN + g * MN;
    for (int t = 0; t < T; ++t) {
        const int64_t row = (int64_t)s * T + t;
        __syncthreads();
        if (threadIdx.x < MN) Bs[threadIdx.x] = xbc[row * ld_xbc + bcol + threadIdx.x];
        else Cs[threadIdx.x - MN] = xbc[row * ld_xbc + ccol + threadIdx.x - MN];
        __syncthreads();
        float dt = zx[row * ldzx + dt_col + h] + bias;
        dt = dt > 20.f ? dt : log1pf(expf(dt));
        const float dA = expf(dt * A);
        const float xv = xbc[row * ld_xbc + h * MP + p];
        const float dx = dt * xv;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            state[i] = state[i] * dA + dx * Bs[nq * 32 + i];
            acc += state[i] * Cs[nq * 32 + i];
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (nq == 0) y[row * ldy + h * MP + p] = acc + D * xv;
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) st[i] = state[i];
}

// ---------------------------------------------------------------------------------------------
// Mamba-2's gated RMSNorm (norm_before_gate=False): v = y * silu(z), normalized over groups of `gs` channels.
// A block per (row, group).
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) gated_norm_kernel(const float* y, int64_t ldy, const float* z, int64_t ldz, const float* w, float eps, float* out,
                                                         int64_t ldo, int gs) {
    extern __shared__ float vs[];
    __shared__ float red[8];
    const int64_t r = blockIdx.y;
    const int c0 = blockIdx.x * gs;
    float ss = 0.f;
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        const float v = y[r * ldy + c0 + i] * silu_f(z[r * ldz + c0 + i]);
        vs[i] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float rstd = rsqrtf(red[0] / gs + eps);
    for (int i = threadIdx.x; i < gs; i += blockDim.x) out[r * ldo + c0 + i] = vs[i] * rstd * w[c0 + i];
}

// ---------------------------------------------------------------------------------------------
// The MoE router of a row: sigmoid(x W^T); the top `topk` of scores + bias are chosen, weighted by their scores
// normalized to sum 1, times `scale`. W f32 [E, K], E <= 256.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) router_kernel(const float* x, int64_t ldx, const float* W, const float* bias, int E, int K, int topk, float scale,
                                                     int* idx, float* wts) {
    extern __shared__ float xs[];
    __shared__ float scores[256];
    const int64_t r = blockIdx.x;
    for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = x[r * ldx + k];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int e = warp; e < E; e += 8) {
        const float* w = W + (int64_t)e * K;
        float acc = 0.f;
        for (int k = lane; k < K; k += 32) acc += w[k] * xs[k];
        acc = warp_sum(acc);
        if (lane == 0) scores[e] = 1.f / (1.f + expf(-acc));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int chosen[16];
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) {
            int best = -1;
            float bv = -INFINITY;
            for (int e = 0; e < E; ++e) {
                bool used = false;
                for (int i = 0; i < j; ++i) used |= chosen[i] == e;
                const float v = scores[e] + bias[e];
                if (!used && v > bv) {
                    bv = v;
                    best = e;
                }
            }
            chosen[j] = best;
            sum += scores[best];
        }
        for (int j = 0; j < topk; ++j) {
            idx[r * topk + j] = chosen[j];
            wts[r * topk + j] = scores[chosen[j]] / (sum + 1e-20f) * scale;
        }
    }
}

// x[r, :] += shared[r, :] + sum_j w[r, j] * y[r * topk + j, :]
__global__ void moe_combine_kernel(float* x, int64_t ldx, const float* shared, const float* y, const float* wts, int topk, int D) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t r = blockIdx.y;
    if (c >= D) return;
    float v = shared[r * D + c];
    for (int j = 0; j < topk; ++j) v += wts[r * topk + j] * y[(r * topk + j) * D + c];
    x[r * ldx + c] += v;
}

// ---------------------------------------------------------------------------------------------
// Sampling: the Gumbel-max trick, argmax(logit / temperature + Gumbel noise), which draws from softmax(logit /
// temperature); temperature 0 takes the argmax. The noise is a hash of (seed, step, column), so a run repeats.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__global__ void __launch_bounds__(1024) sample_kernel(const float* logits, int64_t ld, int V, float temperature, const uint64_t* seeds, uint64_t step,
                                                      int* out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int64_t r = blockIdx.x;
    const uint64_t key = mix64(seeds[r] ^ mix64(step + 0x9e3779b97f4a7c15ull));
    float best = -INFINITY;
    int besti = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float v = logits[r * ld + i];
        if (temperature > 0.f) {
            const uint64_t h = mix64(key ^ ((uint64_t)i * 0xd1b54a32d192ed03ull));
            // u in (0, 1)
            const float u = ((float)(h >> 40) + 0.5f) * (1.f / 16777216.f);
            v = v / temperature - logf(-logf(u));
        }
        if (v > best || (v == best && i < besti)) {
            best = v;
            besti = i;
        }
    }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
        if (ov > best || (ov == best && oi < besti)) {
            best = ov;
            besti = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        bv[warp] = best;
        bi[warp] = besti;
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        best = lane < nw ? bv[lane] : -INFINITY;
        besti = lane < nw ? bi[lane] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
            if (ov > best || (ov == best && oi < besti)) {
                best = ov;
                besti = oi;
            }
        }
        if (lane == 0) out[r] = besti;
    }
}

// ---------------------------------------------------------------------------------------------
// Attention of a few query rows against the cached keys: a block per (row, head), 128 threads for 128-wide heads. Row i
// reads the first min(lk, i + offset) keys.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128) attn_rows_kernel(const float* q, int64_t q_stride, const float* kp, const float* vp, int64_t kv_stride, int hq,
                                                        int hk, int lk, float* out, int64_t out_stride, float scale, int offset) {
    extern __shared__ float p[];
    __shared__ float qs[128];
    __shared__ float red[4];
    const int i = blockIdx.x, h = blockIdx.y, e = threadIdx.x, warp = e >> 5, lane = e & 31;
    const int kh = h / (hq / hk);
    qs[e] = q[(int64_t)i * q_stride + h * 128 + e];
    __syncthreads();
    const int n = min(lk, i + offset);
    float mx = -INFINITY;
    for (int j = e; j < n; j += 128) {
        const float* kr = kp + (int64_t)j * kv_stride + kh * 128;
        float s = 0.f;
        for (int c = 0; c < 128; c += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(kr + c);
            s += qs[c] * k4.x + qs[c + 1] * k4.y + qs[c + 2] * k4.z + qs[c + 3] * k4.w;
        }
        s *= scale;
        p[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = warp_max(mx);
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float sum = 0.f;
    for (int j = e; j < n; j += 128) {
        const float x = expf(p[j] - mx);
        p[j] = x;
        sum += x;
    }
    sum = warp_sum(sum);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    sum = red[0] + red[1] + red[2] + red[3];
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += p[j] * vp[(int64_t)j * kv_stride + kh * 128 + e];
    out[(int64_t)i * out_stride + h * 128 + e] = acc / sum;
}

// ---------------------------------------------------------------------------------------------
// Image decoder (L24-SQ ViT)
// ---------------------------------------------------------------------------------------------
// out[i, :] = codebook[idx[i], :] / max(|codebook[idx[i], :]|, 1e-12)
__global__ void codebook_lookup_kernel(const float* codebook, const int* idx, int n, int d, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* c = codebook + (int64_t)idx[i] * d;
    float ss = 0.f;
    for (int e = 0; e < d; ++e) ss += c[e] * c[e];
    const float inv = 1.f / fmaxf(sqrtf(ss), 1e-12f);
    for (int e = 0; e < d; ++e) out[(int64_t)i * d + e] = c[e] * inv;
}

// 2D RoPE on q and k ([gh * gw, heads * 64] rows of stride `ld`, in place): each 64-wide head's first 32 values turn by
// the token's row, the last 32 by its column, each half as rotate_half pairs (i, i + 16) at frequency i.
__global__ void rope2d_kernel(float* x, int64_t ld, int gw, int heads, float theta, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // one thread per (token, head, half, pair)
    const int pair = (int)(i % 16);
    const int half = (int)((i / 16) % 2);
    const int head = (int)((i / 32) % heads);
    const int64_t tok = i / (32 * heads);
    const int pos = half == 0 ? (int)(tok / gw) : (int)(tok % gw);
    const float inv = powf(theta, -(float)(pair * 2) / 32.f);
    float sn, cs;
    sincosf((float)pos * inv, &sn, &cs);
    float* p = x + tok * ld + head * 64 + half * 32;
    const float a = p[pair], b = p[pair + 16];
    p[pair] = a * cs - b * sn;
    p[pair + 16] = b * cs + a * sn;
}

__global__ void tanh_kernel(float* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

// Tokens [gh * gw, 3 * 16 * 16] laid out (c, sh, sw) to RGB8 pixels [gh * 16][gw * 16][3]: round((clamp(v) + 1) / 2 * 255).
__global__ void unpatch_rgb8_kernel(const float* tok, int gh, int gw, uint8_t* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int W = gw * 16;
    const int64_t n = (int64_t)gh * 16 * W * 3;
    if (i >= n) return;
    const int c = (int)(i % 3);
    const int x = (int)((i / 3) % W);
    const int y = (int)(i / 3 / W);
    const int64_t t = (int64_t)(y / 16) * gw + x / 16;
    const float v = fminf(fmaxf(tok[t * 768 + c * 256 + (y % 16) * 16 + x % 16], -1.f), 1.f);
    out[i] = (uint8_t)rintf((v + 1.f) * 0.5f * 255.f);
}

}  // namespace

GPU_API int gpu_nm_gemv_bf16(const void* W, int64_t ldw, const float* x, int64_t ldx, float* out, int64_t ldo, int rows, int N, int K, int accumulate,
                             int act, void* stream) {
    if (K % 8 || rows < 1 || rows > 8 || ldx % 4 || ldw % 8) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const dim3 grid((N + 7) / 8);
    const bf16* w = (const bf16*)W;
    switch (rows) {
#define CASE(R) case R: gemv_bf16_kernel<R><<<grid, 256, 0, s>>>(w, ldw, x, ldx, out, ldo, N, K, accumulate, act); break;
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    }
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_gemv_fp4(const void* jobs, int njobs, const float* x, int64_t ldx, float* out, int64_t ldo, int N, int K, int act, void* stream) {
    if (K % 32 || njobs < 1 || sizeof(Fp4Job) != 32) return -1;
    const dim3 grid((N + 7) / 8, njobs);
    const size_t smem = (size_t)K * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(gemv_fp4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    gemv_fp4_kernel<<<grid, 256, smem, (cudaStream_t)stream>>>((const Fp4Job*)jobs, x, ldx, out, ldo, N, K, act);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_mamba_conv(const float* in, int64_t ld_in, float* state, const float* w, const float* b, float* out, int nseq, int T, int C,
                              void* stream) {
    const dim3 grid((C + 255) / 256, nseq);
    mamba_conv_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(in, ld_in, state, w, b, out, T, C);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_mamba_scan(const float* xbc, int64_t ld_xbc, const float* zx, int64_t ldzx, int dt_col, float* ssm, const float* A_log, const float* D,
                              const float* dt_bias, float* y, int64_t ldy, int nseq, int T, int H, int P, int N, int G, void* stream) {
    if (P != MP || N != MN || H % G) return -1;
    const dim3 grid(H, nseq);
    mamba_scan_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(xbc, ld_xbc, zx, ldzx, dt_col, ssm, A_log, D, dt_bias, y, ldy, T, H, G);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_gated_norm(const float* y, int64_t ldy, const float* z, int64_t ldz, const float* w, float eps, float* out, int64_t ldo, int rows,
                              int C, int gs, void* stream) {
    if (C % gs) return -1;
    const dim3 grid(C / gs, rows);
    const size_t smem = (size_t)gs * sizeof(float);
    gated_norm_kernel<<<grid, 256, smem, (cudaStream_t)stream>>>(y, ldy, z, ldz, w, eps, out, ldo, gs);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_router(const float* x, int64_t ldx, const float* W, const float* bias, int rows, int E, int K, int topk, float scale, int* idx,
                          float* wts, void* stream) {
    if (E > 256 || topk > 16) return -1;
    const size_t smem = (size_t)K * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(router_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    router_kernel<<<rows, 256, smem, (cudaStream_t)stream>>>(x, ldx, W, bias, E, K, topk, scale, idx, wts);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_moe_combine(float* x, int64_t ldx, const float* shared, const float* y, const float* wts, int rows, int topk, int D, void* stream) {
    const dim3 grid((D + 255) / 256, rows);
    moe_combine_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(x, ldx, shared, y, wts, topk, D);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_sample(const float* logits, int64_t ld, int rows, int V, float temperature, const uint64_t* seeds, uint64_t step, int* out,
                          void* stream) {
    sample_kernel<<<rows, 1024, 0, (cudaStream_t)stream>>>(logits, ld, V, temperature, seeds, step, out);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_attn_rows(const float* q, int64_t q_stride, const float* k, const float* v, int64_t kv_stride, int hq, int hk, int d, int lq, int lk,
                             float* out, int64_t out_stride, float scale, int offset, void* stream) {
    if (d != 128 || hq % hk || kv_stride % 4) return -1;
    const size_t smem = (size_t)lk * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(attn_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attn_rows_kernel<<<dim3(lq, hq), 128, smem, (cudaStream_t)stream>>>(q, q_stride, k, v, kv_stride, hq, hk, lk, out, out_stride, scale, offset);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_codebook_lookup(const float* codebook, const int* idx, int n, int d, float* out, void* stream) {
    codebook_lookup_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(codebook, idx, n, d, out);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_rope2d(float* x, int64_t ld, int gh, int gw, int heads, float theta, void* stream) {
    const int64_t n = (int64_t)gh * gw * heads * 32;
    rope2d_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, ld, gw, heads, theta, n);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_tanh(float* x, int64_t n, void* stream) {
    tanh_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, n);
    KCHECK();
    return 0;
}

GPU_API int gpu_nm_unpatch_rgb8(const float* tok, int gh, int gw, uint8_t* out, void* stream) {
    const int64_t n = (int64_t)gh * gw * 256 * 3;
    unpatch_rgb8_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(tok, gh, gw, out);
    KCHECK();
    return 0;
}
