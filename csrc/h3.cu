// MiniMax H3: its weight formats, split-half RoPE on part of a head, and the audio decoder's 1-D ops.
#include "common.cuh"
#include <cuda_fp8.h>
#include <algorithm>

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
// out[M, N] = A[M, K] . W^T (+ beta * out), f32, with W decoded from nvfp4 on the fly, to the values
// gpu_nvfp4_dequant writes: for short prompts, where writing and reading back the f32 weights costs far more than the
// products. W may be up to three weights side by side in N (q, k and v; gate and up), each its own tensor. Tiles of
// BM rows by 64 columns, 128 threads of TM x 4; split along K into `ws`, summed in order by nvfp4_splits_kernel.
// ---------------------------------------------------------------------------------------------
struct NvWeights {
    const uint8_t* q[3];
    const uint8_t* sc[3];
    const bf16* pre[3];
    float scale2[3];
    int n[3];
    int count;
};

template <int BM>
__global__ void __launch_bounds__(128) nvfp4_gemm_kernel(const float* __restrict__ A, int64_t lda, int M, int K, NvWeights wt, float* out, int64_t ldd,
                                                         float beta, float* ws, int64_t ntot, int kc) {
    constexpr int BN = 64, BK = 32, TM = BM / 8;
    __shared__ __align__(16) float As[BK][BM];
    __shared__ __align__(16) float Ws[BK][BN];
    __shared__ float lut[16];
    const int t = threadIdx.x, tm = t >> 4, tn = t & 15;
    if (t < 16) lut[t] = kE2M1[t];
    const int col0 = blockIdx.x * BN;
    int si = 0, base = 0;
    while (si < wt.count - 1 && col0 >= base + wt.n[si]) base += wt.n[si++];
    const uint8_t* sc = wt.sc[si];
    const bf16* pre = wt.pre[si];
    const float scale2 = wt.scale2[si];
    const int m0 = blockIdx.y * BM, k0 = blockIdx.z * kc, k1 = min(K, k0 + kc);
    // this thread decodes 16 values of weight row `n` per tile, the tile's first or second group
    const int wn = t & 63, wg = t >> 6, n = col0 - base + wn;
    const uint8_t* qrow = wt.q[si] + (int64_t)n * (K / 2);
    const int64_t cblocks = (K / 16 + 3) / 4, tbase = (int64_t)(n / 128) * cblocks;
    const int rr = n % 128, sub = (rr % 32) * 16 + (rr / 32) * 4;
    float acc[TM][4];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    __syncthreads();
    for (int kb = k0; kb < k1; kb += BK) {
#pragma unroll
        for (int i = t; i < BM * BK; i += 128) {
            const int mm = i / BK, kk = i % BK, m = m0 + mm;
            As[kk][mm] = m < M ? A[(int64_t)m * lda + kb + kk] : 0.f;
        }
        {
            const int c = kb / 16 + wg;
            const uint2 qv = *reinterpret_cast<const uint2*>(qrow + c * 8);
            const float total = rnd<bf16>(scale2 * rnd<bf16>(e4m3(sc[(tbase + c / 4) * 512 + sub + c % 4])));
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned v = ((b < 4 ? qv.x : qv.y) >> ((b & 3) * 8)) & 0xFF;
                float a = rnd<bf16>(lut[v >> 4] * total), d = rnd<bf16>(lut[v & 15] * total);
                if (pre) {
                    a *= __bfloat162float(pre[c * 16 + 2 * b]);
                    d *= __bfloat162float(pre[c * 16 + 2 * b + 1]);
                }
                Ws[wg * 16 + 2 * b][wn] = a;
                Ws[wg * 16 + 2 * b + 1][wn] = d;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; ++kk) {
            const float4 w = *reinterpret_cast<const float4*>(&Ws[kk][tn * 4]);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float a = As[kk][tm * TM + i];
                acc[i][0] = fmaf(a, w.x, acc[i][0]);
                acc[i][1] = fmaf(a, w.y, acc[i][1]);
                acc[i][2] = fmaf(a, w.z, acc[i][2]);
                acc[i][3] = fmaf(a, w.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int m = m0 + tm * TM + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t col = col0 + tn * 4 + j;
            if (ws) {
                ws[((int64_t)blockIdx.z * M + m) * ntot + col] = acc[i][j];
            } else {
                float* o = out + (int64_t)m * ldd + col;
                *o = beta != 0.f ? acc[i][j] + beta * *o : acc[i][j];
            }
        }
    }
}

__global__ void nvfp4_splits_kernel(const float* ws, int splits, int M, int64_t ntot, float* out, int64_t ldd, float beta) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * ntot) return;
    const int64_t m = i / ntot, col = i % ntot;
    float s = ws[i];
    for (int z = 1; z < splits; ++z) s += ws[(int64_t)z * M * ntot + i];
    float* o = out + m * ldd + col;
    *o = beta != 0.f ? s + beta * *o : s;
}

// `count` weights of `n[i]` rows (multiples of 64) by K (a multiple of 32); `ws` takes splits x M x sum(n) floats
GPU_API int gpu_nvfp4_gemm(const float* A, int64_t lda, int M, int K, const void* const* q, const void* const* sc, const void* const* pre,
                           const float* scale2, const int* n, int count, float* out, int64_t ldd, float beta, float* ws, int64_t ws_bytes, void* stream) {
    if (count < 1 || count > 3 || K % 32 || M < 1) return -1;
    NvWeights wt{};
    int64_t ntot = 0;
    for (int i = 0; i < count; ++i) {
        if (n[i] % 64) return -1;
        wt.q[i] = (const uint8_t*)q[i];
        wt.sc[i] = (const uint8_t*)sc[i];
        wt.pre[i] = (const bf16*)pre[i];
        wt.scale2[i] = scale2[i];
        wt.n[i] = n[i];
        ntot += n[i];
    }
    wt.count = count;
    const int bm = M <= 16 ? 16 : M <= 32 ? 32 : 64;
    const int mt = (M + bm - 1) / bm;
    // enough blocks to cover the SMs about four times, each K split at least 512 wide
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int64_t tiles = (ntot / 64) * mt;
    int splits = (int)std::min<int64_t>(std::max<int64_t>(1, (4 * sms + tiles - 1) / tiles), K / 512);
    while (splits > 1 && (int64_t)splits * M * ntot * 4 > ws_bytes) --splits;
    if (splits < 1) splits = 1;
    const int kc = ((K + splits - 1) / splits + 31) / 32 * 32;
    splits = (K + kc - 1) / kc;
    float* w = splits > 1 ? ws : nullptr;
    const dim3 grid((unsigned)(ntot / 64), (unsigned)mt, (unsigned)splits);
    cudaStream_t s = (cudaStream_t)stream;
    if (bm == 16) nvfp4_gemm_kernel<16><<<grid, 128, 0, s>>>(A, lda, M, K, wt, out, ldd, beta, w, ntot, kc);
    else if (bm == 32) nvfp4_gemm_kernel<32><<<grid, 128, 0, s>>>(A, lda, M, K, wt, out, ldd, beta, w, ntot, kc);
    else nvfp4_gemm_kernel<64><<<grid, 128, 0, s>>>(A, lda, M, K, wt, out, ldd, beta, w, ntot, kc);
    KCHECK();
    if (w) {
        const int64_t total = (int64_t)M * ntot;
        nvfp4_splits_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(w, splits, M, ntot, out, ldd, beta);
        KCHECK();
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Per-head RMSNorm with a weight, then split-half RoPE on the first 2*pairs dims, bf16 in place.
// x [L, H, d] with rows `stride` and heads `hs` elements apart; rope [L][pairs][(cos, sin)]; w may be null.
// ---------------------------------------------------------------------------------------------
// A warp per head, its d / 32 values strided by 32 over the lanes, as the old block of d threads held them: the sum
// of squares reduces in the same order, so the results are the same bits. `ew` stages the head for the pair partners.
__global__ void rope_bf16_kernel(bf16* x, int64_t stride, int64_t hs, const float* w, float eps, const float* rope, int L, int H, int d, int pairs) {
    __shared__ float ew[8][128];
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5;
    const int64_t head = (int64_t)blockIdx.x * (blockDim.x >> 5) + wi;
    if (head >= (int64_t)L * H) return;
    const int t = (int)(head / H), h = (int)(head % H), nw = d >> 5;
    bf16* v = x + (int64_t)t * stride + (int64_t)h * hs;
    float* e = ew[wi];
    float part[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j < nw) {
            const float a = __bfloat162float(v[j * 32 + lane]);
            e[j * 32 + lane] = a;
            part[j] = warp_sum(a * a);
        }
    }
    // block_sum's second stage: the warps' sums in the first lanes, the rest zero
    float tot = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (j < nw && lane == j) tot = part[j];
    const float rstd = rsqrtf(warp_sum(tot) / (float)d + eps);
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j >= nw) continue;
        const int i = j * 32 + lane;
        if (i < 2 * pairs) {
            const int p = i < pairs ? i : i - pairs;
            const float* r = rope + ((int64_t)t * pairs + p) * 2;
            const float a = rnd<bf16>(e[p] * rstd * (w ? w[p] : 1.f)), b = rnd<bf16>(e[p + pairs] * rstd * (w ? w[p + pairs] : 1.f));
            if (i < pairs) v[i] = __float2bfloat16_rn(a * r[0] - b * r[1]);
            else v[i] = __float2bfloat16_rn(b * r[0] + a * r[1]);
        } else {
            v[i] = __float2bfloat16_rn(e[i] * rstd * (w ? w[i] : 1.f));
        }
    }
}
GPU_API int gpu_rope_bf16(void* x, int64_t stride, int64_t hs, const float* w, float eps, const float* rope, int L, int H, int d, int pairs, void* stream) {
    if (d % 32 || d > 128) return -1;
    const int64_t heads = (int64_t)L * H;
    rope_bf16_kernel<<<(unsigned)((heads + 7) / 8), 256, 0, (cudaStream_t)stream>>>((bf16*)x, stride, hs, w, eps, rope, L, H, d, pairs);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The DWD pixel head's patches of one native block to frames: `out` bf16 holds the first latent's `hw` tokens of
// 3x16x16 values, then the other latents' tokens of 3x4x16x16 (c, t, y, x). Each pixel goes through the head's color
// codec, std * v + mean to [-1, 1] then to 0..1, into fp16 `clip` [3, frames, 16h, 16w] at frame f0.
// ---------------------------------------------------------------------------------------------
__global__ void dwd_unpatch_kernel(const bf16* out, int h, int w, int bf, half* clip, int frames, int f0, float m0, float m1, float m2,
                                   float s0, float s1, float s2) {
    const int H = h * 16, W = w * 16, hw = h * w;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)3 * bf * H * W) return;
    const int X = (int)(i % W), Y = (int)((i / W) % H);
    const int fb = (int)((i / ((int64_t)W * H)) % bf), c = (int)(i / ((int64_t)W * H * bf));
    const int tok = (Y >> 4) * w + (X >> 4), py = Y & 15, px = X & 15;
    int64_t src;
    if (fb == 0) {
        src = (int64_t)tok * 768 + (c * 16 + py) * 16 + px;
    } else {
        const int l = (fb - 1) >> 2, pt = (fb - 1) & 3;
        src = (int64_t)hw * 768 + ((int64_t)l * hw + tok) * 3072 + ((c * 4 + pt) * 16 + py) * 16 + px;
    }
    const float mean = c == 0 ? m0 : c == 1 ? m1 : m2, sd = c == 0 ? s0 : c == 1 ? s1 : s2;
    // unfused, as the host computed it
    const float v = fminf(1.f, fmaxf(-1.f, __fadd_rn(__fmul_rn(__bfloat162float(out[src]), sd), mean)));
    clip[(((int64_t)c * frames + f0 + fb) * H + Y) * W + X] = __float2half_rn(__fmul_rn(__fadd_rn(v, 1.f), 0.5f));
}
GPU_API int gpu_dwd_unpatch(const void* out, int h, int w, int bf, void* clip, int frames, int f0, const float* mean, const float* sd, void* stream) {
    const int64_t n = (int64_t)3 * bf * h * 16 * w * 16;
    dwd_unpatch_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)out, h, w, bf, (half*)clip, frames, f0, mean[0], mean[1], mean[2], sd[0], sd[1],
                                                                                    sd[2]);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// x[r] += gate[r] * y[r], bf16 rows of `cols`, each row's gate the row `sel[r]` of `gates`: a residual GEMM's epilogue
// (gpu_int8_gemm_residual) after a plain GEMM of every row, when the rows' gates differ. Rounded as that epilogue rounds:
// y is the plain GEMM's bf16 output, the product and the sum stay f32, then bf16.
// ---------------------------------------------------------------------------------------------
__global__ void add_gated_rows_kernel(bf16* x, const bf16* y, const bf16* gates, const int* sel, int64_t rows, int cols) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= rows * cols) return;
    const int64_t r = i / cols;
    const int c = (int)(i % cols);
    const uint4 yv = *reinterpret_cast<const uint4*>(y + i);
    const uint4 gv = *reinterpret_cast<const uint4*>(gates + (int64_t)sel[r] * cols + c);
    uint4 xv = *reinterpret_cast<const uint4*>(x + i);
    const bf16* yb = reinterpret_cast<const bf16*>(&yv);
    const bf16* gb = reinterpret_cast<const bf16*>(&gv);
    bf16* xb = reinterpret_cast<bf16*>(&xv);
#pragma unroll
    for (int j = 0; j < 8; ++j) xb[j] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(__bfloat162float(gb[j]), __bfloat162float(yb[j])), __bfloat162float(xb[j])));
    *reinterpret_cast<uint4*>(x + i) = xv;
}
GPU_API int gpu_add_gated_rows_bf16(void* x, const void* y, const void* gates, const int* sel, int rows, int cols, void* stream) {
    if (cols % 8) return -1;
    const int64_t n = (int64_t)rows * cols / 8;
    add_gated_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((bf16*)x, (const bf16*)y, (const bf16*)gates, sel, rows, cols);
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
// conv1d_kernel in tiles of 32 output channels by 64 samples, staged through shared memory, for the wide layers. Each
// output sums in the same order, taps outer and input channels inner, so the results are the same bits.
__global__ void __launch_bounds__(256) conv1d_tiled_kernel(const float* x, const float* w, const float* bias, float* out, int Ci, int Co, int T, int K, int dil,
                                                           int pad) {
    constexpr int CT = 32, TT = 64, CC = 16;
    __shared__ float xs[CC][TT];
    __shared__ float wsm[CT][CC + 1];
    const int tid = threadIdx.x, cy = tid >> 5, tx = tid & 31;
    const int t0 = blockIdx.x * TT, c0 = blockIdx.y * CT, b = blockIdx.z;
    const float* xb = x + (int64_t)b * Ci * T;
    float acc[4][2] = {};
    for (int k = 0; k < K; ++k) {
        const int shift = k * dil - pad;
        // a tap past either end adds nothing, as the plain kernel skips it
        bool in[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int s = t0 + tx + 32 * j + shift;
            in[j] = s >= 0 && s < T;
        }
        for (int ci0 = 0; ci0 < Ci; ci0 += CC) {
            __syncthreads();
            for (int i = tid; i < CC * TT; i += 256) {
                const int ci = i / TT, tt = i % TT, s = t0 + tt + shift;
                xs[ci][tt] = (ci0 + ci < Ci && s >= 0 && s < T) ? xb[(int64_t)(ci0 + ci) * T + s] : 0.f;
            }
            for (int i = tid; i < CT * CC; i += 256) {
                const int cc = i / CC, ci = i % CC;
                wsm[cc][ci] = ci0 + ci < Ci ? w[((int64_t)(c0 + cc) * Ci + ci0 + ci) * K + k] : 0.f;
            }
            __syncthreads();
            const int n = min(CC, Ci - ci0);
            for (int ci = 0; ci < n; ++ci) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float wv = wsm[cy * 4 + i][ci];
#pragma unroll
                    for (int j = 0; j < 2; ++j)
                        if (in[j]) acc[i][j] = fmaf(wv, xs[ci][tx + 32 * j], acc[i][j]);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = c0 + cy * 4 + i;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = t0 + tx + 32 * j;
            if (t < T) out[((int64_t)b * Co + c) * T + t] = acc[i][j] + (bias ? bias[c] : 0.f);
        }
    }
}
GPU_API int gpu_conv1d(const void* x, const void* w, const void* bias, void* out, int B, int Ci, int Co, int T, int K, int dil, int pad, void* stream) {
    if (Co % 32 == 0) {
        const dim3 grid((unsigned)((T + 63) / 64), (unsigned)(Co / 32), (unsigned)B);
        conv1d_tiled_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const float*)x, (const float*)w, (const float*)bias, (float*)out, Ci, Co, T, K, dil, pad);
        KCHECK();
        return 0;
    }
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

// 3x3x3 causal patches of output frame `to`, rows [y0, y0 + rows), from NHWC frames [T][H][W][C]:
// out[(y - y0) * Wo + x, ((kt * 3 + ky) * 3 + kx) * C + c]. Tap kt reads frame to * tstride + kt - 2, zero before the
// first frame (the causal front padding); the spatial padding is im2col3's.
__global__ void h3_im2col3t_kernel(const __half* in, int H, int W, int C, int stride, int tstride, int to, int y0, int rows, __half* out) {
    const int Wo = stride == 1 ? W : W / 2, pad = stride == 1 ? 1 : 0;
    const int64_t per = 27 * (int64_t)C;
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)rows * Wo * per) return;
    const int64_t p = t / per;
    const int rest = (int)(t % per);
    const int k = rest / C, c = rest % C;
    const int f = to * tstride + k / 9 - 2, ks = k % 9;
    if (f < 0) {
        out[t] = __float2half(0.f);
        return;
    }
    const int y = y0 + (int)(p / Wo), x = (int)(p % Wo);
    const int sy = reflect_at(y * stride + ks / 3 - pad, H), sx = reflect_at(x * stride + ks % 3 - pad, W);
    out[t] = in[(((int64_t)f * H + sy) * W + sx) * C + c];
}

GPU_API int gpu_h3_im2col3t(const void* in, int H, int W, int C, int stride, int tstride, int to, int y0, int rows, void* out, void* stream) {
    const int Wo = stride == 1 ? W : W / 2;
    const int64_t total = (int64_t)rows * Wo * 27 * C;
    h3_im2col3t_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const __half*)in, H, W, C, stride, tstride, to, y0, rows,
                                                                                         (__half*)out);
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
