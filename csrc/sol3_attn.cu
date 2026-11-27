// Sol³ sparse attention (Fan et al., "Sol³: Accelerating Video Diffusion for Free via Hierarchical Sparse Attention
// Routing", arXiv 2611.32562), ported from the SM120 path of sol3_attn (github.com/FofGofx/sol3 at 21a509a,
// src/sol3_attn/backends/sm120, Apache-2.0): the public `sol3_attn(q, k, v, prefix_tokens=...)` call, whose route and
// joint kernel the FastH3 profile tunes. Bidirectional bf16 attention, head dim 128, batch 1. q/k/v are post-rope
// [T, H, 128] with any row and head stride; the output has its own row stride.
//
// For each 64-row query block i and 64-row key block j, from block means (fp32 q, bf16 k and v):
//   proxy = mean_q(i) . mean_k(j) * scale * log2e
//   exact  when proxy > th1(i) (the proxy's mean plus 1.9 sigma over all key blocks, from the key means' moments), j is
//          next to i, or j overlaps the protected prefix: every row of the block is attended exactly
//   pure   when the block's risk (proxy + log2(rows * rms(v)) + the key variance's second-order term) is below the mean
//          risk of row i: the block is represented by its mean key and value
//   mixed  otherwise: the tokens of every mixed block whose score against mean_q(i) is above mu + 1.125 sigma, both over
//          all tokens of the row's mixed blocks, are attended exactly, and the rest by the block's mean
// The joint kernel runs the representatives first (each weighted by the rows it stands for), adds the first-order
// k-v covariance term, then the exact rows in 64-row chunks, in the order upstream compacts them. Prefix query rows
// attend everything densely, through the caller's own attention, as sol3_attn's `dense` callback lets a host do.
#include "common.cuh"
#include "sol_layout.cuh"

#include <cmath>
#include <cstdint>

namespace sol3 {

constexpr int D = 128;
constexpr int BLK = 64;
constexpr float LOG2E = 1.4426950408889634f;
// upstream's FastH3 route: beta1, beta3 and the covariance coefficient (beta2 is 0)
constexpr float BETA1 = 1.9f;
constexpr float BETA3 = 1.125f;
constexpr float GAMMA = 1.0f;
constexpr int MU_SPLIT = 32;   // key blocks per CTA of the token route
constexpr int COV_SPLIT = 32;  // key blocks per CTA of the covariance
constexpr int CQ = 32;         // query blocks per CTA of the classification
constexpr int CJ = 64;         // key blocks per tile of the classification
constexpr int CP = D + 1;      // its padded smem row

// ---------------------------------------------------------------------------------------------
// Tensor-core helpers (sm_80+): bf16 mma.m16n8k16 with float accumulators, ldmatrix, cp.async.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// 16 bytes from global to shared, zero-filled when `valid` is false.
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// A 64 x 128 bf16 tile in smem: row r's 16-byte chunk c sits at chunk c ^ (r & 7).
__device__ __forceinline__ int swz(int r, int c) { return r * D + ((c ^ (r & 7)) << 3); }

// S = Q.K^T for a warp's 16 query rows against a 64-row tile.
__device__ __forceinline__ void tile_qk(const bf16* sk, const uint32_t (&qa)[D / 16][4], int lane, float (&s)[BLK / 8][4]) {
#pragma unroll
    for (int nt = 0; nt < BLK / 8; ++nt) {
        s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ks += 2) {
            uint32_t b[4];
            const int j = nt * 8 + (lane & 7);
            ldsm_x4(b, sk + swz(j, 2 * ks + (lane >> 3)));
            mma(s[nt], qa[ks], b[0], b[1]);
            mma(s[nt], qa[ks + 1], b[2], b[3]);
        }
    }
}

// acc += P.V for a warp's 16 rows: P in A-fragment form over the tile's 64 rows, V a 64 x 128 tile.
__device__ __forceinline__ void tile_pv(const bf16* sv, const uint32_t (&pa)[BLK / 16][4], int lane, float (&acc)[D / 8][4]) {
#pragma unroll
    for (int kk = 0; kk < BLK / 16; ++kk) {
#pragma unroll
        for (int dn = 0; dn < D / 8; dn += 2) {
            uint32_t b[4];
            const int j = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldsm_x4_t(b, sv + swz(j, dn + (lane >> 4)));
            mma(acc[dn], pa[kk], b[0], b[1]);
            mma(acc[dn + 1], pa[kk], b[2], b[3]);
        }
    }
}

// exp2 as Triton's tl.exp2 lowers it.
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// One online-softmax step over a tile of scores already in log2 units (-inf for masked columns): rescales acc and l,
// and leaves P in A-fragment form for acc += P.V. As upstream: the running max is taken over the tile,
// alpha = exp2(m - m_new), and P is rounded to bf16 for P.V while l sums it in f32. Each lane's l covers its columns.
__device__ __forceinline__ void softmax_p(float (&s)[BLK / 8][4], float (&m)[2], float (&l)[2], float (&acc)[D / 8][4],
                                          uint32_t (&pa)[BLK / 16][4]) {
    float tmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < BLK / 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) tmax[e >> 1] = fmaxf(tmax[e >> 1], s[nt][e]);
    float alpha[2], safe[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 1));
        tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 2));
        const float mn = fmaxf(m[r], tmax[r]);
        safe[r] = mn == -INFINITY ? 0.f : mn;
        alpha[r] = ex2(m[r] - safe[r]);
        m[r] = mn;
        l[r] *= alpha[r];
    }
#pragma unroll
    for (int nt = 0; nt < BLK / 8; ++nt) {
        float pv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            pv[e] = ex2(s[nt][e] - safe[e >> 1]);
            l[e >> 1] += pv[e];
        }
        pa[nt >> 1][(nt & 1) * 2 + 0] = pack(pv[0], pv[1]);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack(pv[2], pv[3]);
    }
#pragma unroll
    for (int dn = 0; dn < D / 8; ++dn) {
        acc[dn][0] *= alpha[0];
        acc[dn][1] *= alpha[0];
        acc[dn][2] *= alpha[1];
        acc[dn][3] *= alpha[1];
    }
}

// A fragments of 16 bf16 rows (row a = lane / 4, row b = a + 8 of the warp's 16), zero where not `ok`.
__device__ __forceinline__ void load_qa(uint32_t (&qa)[D / 16][4], const bf16* pa, const bf16* pb, bool oka, bool okb, int lane) {
    const int cq = (lane & 3) * 2;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
        const int c = ks * 16 + cq;
        qa[ks][0] = oka ? *reinterpret_cast<const uint32_t*>(pa + c) : 0u;
        qa[ks][1] = okb ? *reinterpret_cast<const uint32_t*>(pb + c) : 0u;
        qa[ks][2] = oka ? *reinterpret_cast<const uint32_t*>(pa + c + 8) : 0u;
        qa[ks][3] = okb ? *reinterpret_cast<const uint32_t*>(pb + c + 8) : 0u;
    }
}

// ---------------------------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------------------------

// pool_qkv: per 64-row block, the mean of q (f32), of k and v (bf16), the per-channel variance of k, and
// log2(rows * rms of v over the block and channels). One CTA per (block, head), a thread per channel.
__global__ void __launch_bounds__(D) pool_kernel(const bf16* q, const bf16* k, const bf16* v, int64_t rs, int64_t hs, int T, int N,
                                                  float* qb, bf16* kc, bf16* vm, float* kvar, float* cst, unsigned* vamax) {
    __shared__ float red[33];
    const int j = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int r0 = j * BLK, n = min(BLK, T - r0);
    const int64_t base = (int64_t)r0 * rs + (int64_t)h * hs + d;
    float sq = 0.f, sk = 0.f, sv = 0.f;
    for (int r = 0; r < n; ++r) {
        const int64_t o = base + (int64_t)r * rs;
        sq += __bfloat162float(q[o]);
        sk += __bfloat162float(k[o]);
        sv += __bfloat162float(v[o]);
    }
    const float fn = (float)n;
    const float qm = sq / fn, km = sk / fn, vmean = sv / fn;
    float kv2 = 0.f, vv2 = 0.f;
    for (int r = 0; r < n; ++r) {
        const int64_t o = base + (int64_t)r * rs;
        const float dk = __bfloat162float(k[o]) - km, dv = __bfloat162float(v[o]) - vmean;
        kv2 += dk * dk;
        vv2 += dv * dv;
    }
    if (vamax) {
        float av = 0.f;
        for (int r = 0; r < n; ++r) av = fmaxf(av, fabsf(__bfloat162float(v[base + (int64_t)r * rs])));
        // non-negative floats order as their bits
        atomicMax(vamax + h * D + d, __float_as_uint(av));
    }
    const int64_t o = ((int64_t)h * N + j) * D + d;
    qb[o] = qm;
    kc[o] = __float2bfloat16_rn(km);
    vm[o] = __float2bfloat16_rn(vmean);
    kvar[o] = kv2 / fn;
    const float tot = block_sum(vv2, red);
    if (d == 0) {
        const float rms = sqrtf(fmaxf(tot / (fn * D), 0.f));
        cst[(int64_t)h * N + j] = log2f(fn * fmaxf(rms, 1e-6f));
    }
}

// moments_fast: per head and channel, the mean and variance of the key block means over blocks.
__global__ void __launch_bounds__(D) moments_kernel(const bf16* kc, int N, float* mom) {
    const int h = blockIdx.x, d = threadIdx.x;
    float sm = 0.f, ss = 0.f;
    for (int j = 0; j < N; ++j) {
        const float x = __bfloat162float(kc[((int64_t)h * N + j) * D + d]);
        sm += x;
        ss += x * x;
    }
    const float mean = sm / N;
    mom[h * 2 * D + d] = mean;
    mom[h * 2 * D + D + d] = fmaxf(ss / N - mean * mean, 0.f);
}

// risk_pool_class + finish_class: each (query block, key block) pair as exact (-1), pure (-2) or mixed (0).
// CQ query blocks per CTA, 8 threads per query block, each over every 8th key block of a CJ-wide tile. The first pass
// takes the mean risk of each row (th2), the second recomputes the same scores and classifies.
__global__ void __launch_bounds__(256) classify_kernel(const float* qb, const bf16* kc, const float* kvar, const float* cst,
                                                        const float* mom, int N, int prefix, int s1, float scale, int8_t* flags) {
    extern __shared__ float smf[];
    float* sq = smf;               // [CQ][CP]
    float* skc = sq + CQ * CP;     // [CJ][CP]
    float* skv = skc + CJ * CP;    // [CJ][CP]
    float* scc = skv + CJ * CP;    // [CJ]
    float* sth1 = scc + CJ;        // [CQ]
    float* sth2 = sth1 + CQ;       // [CQ]
    const int h = blockIdx.y, q0 = blockIdx.x * CQ, tid = threadIdx.x;
    const float* hm = mom + h * 2 * D;
    for (int i = tid; i < CQ * D; i += blockDim.x) {
        const int r = i / D, d = i % D;
        sq[r * CP + d] = q0 + r < N ? qb[((int64_t)h * N + q0 + r) * D + d] : 0.f;
    }
    __syncthreads();
    const float s = scale * LOG2E;
    const int r = tid / 8, jl = tid % 8;
    {
        float mu = 0.f, var = 0.f;
        for (int d = jl; d < D; d += 8) {
            const float x = sq[r * CP + d];
            mu += x * hm[d];
            var += x * x * hm[D + d];
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            mu += __shfl_xor_sync(0xffffffff, mu, o);
            var += __shfl_xor_sync(0xffffffff, var, o);
        }
        mu *= s;
        var *= s * s;
        if (jl == 0) sth1[r] = mu + BETA1 * sqrtf(fmaxf(var, 0.f) + 1e-6f);
    }
    const int qi = q0 + r;
    const bool active = qi < N && (qi + 1) * BLK > prefix;
    const float half_s2 = 0.5f * scale * scale;
    float rsum = 0.f;
    for (int pass = 0; pass < 2; ++pass) {
        for (int jt = 0; jt < N; jt += CJ) {
            __syncthreads();
            for (int i = tid; i < CJ * D; i += blockDim.x) {
                const int jj = i / D, d = i % D;
                const bool ok = jt + jj < N;
                const int64_t o = ((int64_t)h * N + jt + jj) * D + d;
                skc[jj * CP + d] = ok ? __bfloat162float(kc[o]) : 0.f;
                skv[jj * CP + d] = ok ? kvar[o] : 0.f;
            }
            for (int i = tid; i < CJ; i += blockDim.x) scc[i] = jt + i < N ? cst[(int64_t)h * N + jt + i] : 0.f;
            __syncthreads();
            float p[CJ / 8], dl[CJ / 8];
#pragma unroll
            for (int u = 0; u < CJ / 8; ++u) p[u] = dl[u] = 0.f;
            for (int d = 0; d < D; ++d) {
                const float x = sq[r * CP + d], x2 = x * x;
#pragma unroll
                for (int u = 0; u < CJ / 8; ++u) {
                    const int jj = jl + 8 * u;
                    p[u] += x * skc[jj * CP + d];
                    dl[u] += x2 * skv[jj * CP + d];
                }
            }
#pragma unroll
            for (int u = 0; u < CJ / 8; ++u) {
                const int jj = jl + 8 * u, j = jt + jj;
                if (j >= N) continue;
                const float proxy = p[u] * s;
                const float risk = proxy + scc[jj] + fmaxf(half_s2 * dl[u], 0.f) * LOG2E;
                if (pass == 0) {
                    rsum += risk;
                } else if (qi < N) {
                    const bool full = active && (proxy > sth1[r] || abs(qi - j) <= 1 || j < s1);
                    const bool pure = !full && active && risk < sth2[r];
                    flags[((int64_t)h * N + qi) * N + j] = (int8_t)(full ? -1 : pure ? -2 : 0);
                }
            }
        }
        if (pass == 0) {
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) rsum += __shfl_xor_sync(0xffffffff, rsum, o);
            if (jl == 0) sth2[r] = rsum / N;
        }
    }
}

// h3_mu_parallel: scores of the bf16-rounded query block means against every token of the row's mixed blocks.
// SELECT = false: their count, mean and M2 over this CTA's key blocks, combined by Chan's formula.
// SELECT = true: the same scores against the row's cut; the mixed pair's flag becomes the count, with the token bits.
// Grid (query tiles of 64 blocks, key splits of MU_SPLIT blocks, heads); a warp per 16 query blocks.
template <bool SELECT>
__global__ void __launch_bounds__(128) route_kernel(const float* qb, const bf16* k, int64_t rs, int64_t hs, int T, int N, int prefix,
                                                     float scale, int S, int8_t* flags, uint64_t* bits, float* part, const float* cut) {
    __shared__ __align__(16) bf16 sk[BLK * D];
    const int h = blockIdx.z, split = blockIdx.y, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    const int ra = blockIdx.x * BLK + warp * 16 + g, rb = ra + 8;
    const bool acta = ra < N && (ra + 1) * BLK > prefix, actb = rb < N && (rb + 1) * BLK > prefix;
    uint32_t qa[D / 16][4];
    {
        const float* pa = qb + ((int64_t)h * N + min(ra, N - 1)) * D;
        const float* pb = qb + ((int64_t)h * N + min(rb, N - 1)) * D;
        const int cq = qd * 2;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
            const int c = ks * 16 + cq;
            qa[ks][0] = ra < N ? pack(pa[c], pa[c + 1]) : 0u;
            qa[ks][1] = rb < N ? pack(pb[c], pb[c + 1]) : 0u;
            qa[ks][2] = ra < N ? pack(pa[c + 8], pa[c + 9]) : 0u;
            qa[ks][3] = rb < N ? pack(pb[c + 8], pb[c + 9]) : 0u;
        }
    }
    const float cuta = SELECT && ra < N ? cut[(int64_t)h * N + ra] : INFINITY;
    const float cutb = SELECT && rb < N ? cut[(int64_t)h * N + rb] : INFINITY;
    float cnt[2] = {0.f, 0.f}, mean[2] = {0.f, 0.f}, m2[2] = {0.f, 0.f};
    const int j0 = split * MU_SPLIT, j1 = min(N, j0 + MU_SPLIT);
    int8_t* fa_row = flags + ((int64_t)h * N + ra) * N;
    int8_t* fb_row = flags + ((int64_t)h * N + rb) * N;
    for (int j = j0; j < j1; ++j) {
        const int fa = acta ? fa_row[j] : -2, fb = actb ? fb_row[j] : -2;
        if (!__syncthreads_or(fa >= 0 || fb >= 0)) continue;
        for (int i = tid; i < BLK * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8), tok = j * BLK + row;
            const bool ok = tok < T;
            cp16(sk + swz(row, c), k + (int64_t)(ok ? tok : 0) * rs + (int64_t)h * hs + c * 8, ok);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        float sc[BLK / 8][4];
        tile_qk(sk, qa, lane, sc);
        __syncthreads();
        const int f[2] = {fa, fb};
        if (!SELECT) {
            float tc[2] = {0.f, 0.f}, ts[2] = {0.f, 0.f};
#pragma unroll
            for (int nt = 0; nt < BLK / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int col = nt * 8 + qd * 2 + (e & 1), rr = e >> 1;
                    sc[nt][e] = sc[nt][e] * scale * LOG2E;
                    if (f[rr] >= 0 && j * BLK + col < T) {
                        tc[rr] += 1.f;
                        ts[rr] += sc[nt][e];
                    }
                }
            float tm[2], tq[2] = {0.f, 0.f};
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                tc[rr] += __shfl_xor_sync(0xffffffff, tc[rr], 1);
                tc[rr] += __shfl_xor_sync(0xffffffff, tc[rr], 2);
                ts[rr] += __shfl_xor_sync(0xffffffff, ts[rr], 1);
                ts[rr] += __shfl_xor_sync(0xffffffff, ts[rr], 2);
                tm[rr] = ts[rr] / fmaxf(tc[rr], 1.f);
            }
#pragma unroll
            for (int nt = 0; nt < BLK / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int col = nt * 8 + qd * 2 + (e & 1), rr = e >> 1;
                    if (f[rr] >= 0 && j * BLK + col < T) {
                        const float dx = sc[nt][e] - tm[rr];
                        tq[rr] += dx * dx;
                    }
                }
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                tq[rr] += __shfl_xor_sync(0xffffffff, tq[rr], 1);
                tq[rr] += __shfl_xor_sync(0xffffffff, tq[rr], 2);
                if (tc[rr] > 0.f) {
                    const float tot = cnt[rr] + tc[rr], delta = tm[rr] - mean[rr];
                    mean[rr] += delta * (tc[rr] / tot);
                    m2[rr] += tq[rr] + delta * delta * cnt[rr] * tc[rr] / tot;
                    cnt[rr] = tot;
                }
            }
        } else {
            uint64_t mask[2] = {0ull, 0ull};
            const float cr[2] = {cuta, cutb};
#pragma unroll
            for (int nt = 0; nt < BLK / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int col = nt * 8 + qd * 2 + (e & 1), rr = e >> 1;
                    const float x = sc[nt][e] * scale * LOG2E;
                    if (f[rr] >= 0 && j * BLK + col < T && x > cr[rr]) mask[rr] |= 1ull << col;
                }
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                mask[rr] |= __shfl_xor_sync(0xffffffff, mask[rr], 1);
                mask[rr] |= __shfl_xor_sync(0xffffffff, mask[rr], 2);
            }
            // each pair belongs to this CTA alone, and its flag was read above
            if (qd == 0) {
                if (fa >= 0) {
                    fa_row[j] = (int8_t)__popcll(mask[0]);
                    bits[((int64_t)h * N + ra) * N + j] = mask[0];
                }
                if (fb >= 0) {
                    fb_row[j] = (int8_t)__popcll(mask[1]);
                    bits[((int64_t)h * N + rb) * N + j] = mask[1];
                }
            }
        }
    }
    if (!SELECT && qd == 0) {
        const int rows[2] = {ra, rb};
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            if (rows[rr] >= N) continue;
            float* p = part + (((int64_t)h * N + rows[rr]) * S + split) * 3;
            p[0] = cnt[rr];
            p[1] = mean[rr];
            p[2] = m2[rr];
        }
    }
}

// The row's cut, mu + 1.125 sigma over all its mixed tokens, from the splits' partial moments; +inf with none.
__global__ void cut_kernel(const float* part, int S, int rows, float* cut) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float* p = part + (int64_t)i * S * 3;
    float amount = 0.f, wsum = 0.f;
    for (int s = 0; s < S; ++s) {
        amount += p[s * 3];
        wsum += p[s * 3] * p[s * 3 + 1];
    }
    const float mu = wsum / fmaxf(amount, 1.f);
    float var = 0.f;
    for (int s = 0; s < S; ++s) {
        const float dm = p[s * 3 + 1] - mu;
        var += p[s * 3 + 2] + p[s * 3] * dm * dm;
    }
    var /= fmaxf(amount, 1.f);
    const float sigma = sqrtf(fmaxf(var, 0.f) + 1e-6f);
    cut[i] = amount > 0.f ? mu + BETA3 * sigma : INFINITY;
}

// sol3_kv_covariance: sum over the key blocks after the prefix's of (k - block mean of k)^T v, bf16 operands,
// f32 accumulation; a partial per split of COV_SPLIT blocks. 8 warps, each 16 rows of the 128 x 128 result.
__global__ void __launch_bounds__(256) cov_kernel(const bf16* k, const bf16* v, int64_t rs, int64_t hs, int T, int N, int first,
                                                   int CS, float* covp) {
    __shared__ __align__(16) bf16 sk[BLK * D];
    __shared__ __align__(16) bf16 sv[BLK * D];
    __shared__ float skm[D];
    const int h = blockIdx.y, split = blockIdx.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;
    float acc[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    const int j0 = max(first, split * COV_SPLIT), j1 = min(N, (split + 1) * COV_SPLIT);
    for (int j = j0; j < j1; ++j) {
        const int n = min(BLK, T - j * BLK);
        for (int i = tid; i < BLK * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8), tok = j * BLK + row;
            const bool ok = row < n;
            const int64_t o = (int64_t)(ok ? tok : 0) * rs + (int64_t)h * hs + c * 8;
            cp16(sk + swz(row, c), k + o, ok);
            cp16(sv + swz(row, c), v + o, ok);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        if (tid < D) {
            float sum = 0.f;
            for (int r = 0; r < n; ++r) sum += __bfloat162float(sk[swz(r, tid >> 3) + (tid & 7)]);
            skm[tid] = sum / (float)n;
        }
        __syncthreads();
        for (int i = tid; i < BLK * D; i += blockDim.x) {
            const int row = i / D, d = i % D;
            bf16* p = sk + swz(row, d >> 3) + (d & 7);
            *p = row < n ? __float2bfloat16_rn(__bfloat162float(*p) - skm[d]) : __float2bfloat16_rn(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BLK / 16; ++kk) {
            // A (16 channel rows x 16 tokens) from the centred k tile stored [token][channel]
            uint32_t a[4];
            {
                const int mi = lane >> 3;
                const int trow = kk * 16 + (mi >> 1) * 8 + (lane & 7);
                ldsm_x4_t(a, sk + swz(trow, (warp * 16 + (mi & 1) * 8) >> 3));
            }
#pragma unroll
            for (int dn = 0; dn < D / 8; dn += 2) {
                uint32_t b[4];
                const int jr = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldsm_x4_t(b, sv + swz(jr, dn + (lane >> 4)));
                mma(acc[dn], a, b[0], b[1]);
                mma(acc[dn + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
    }
    float* out = covp + ((int64_t)h * CS + split) * D * D;
#pragma unroll
    for (int dn = 0; dn < D / 8; ++dn)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = warp * 16 + g + (e >> 1) * 8, col = dn * 8 + qd * 2 + (e & 1);
            out[row * D + col] = acc[dn][e];
        }
}

// The covariance over the whole sequence length, in bf16.
__global__ void cov_reduce_kernel(const float* covp, int CS, int T, bf16* cov) {
    const int h = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D * D) return;
    float s = 0.f;
    for (int c = 0; c < CS; ++c) s += covp[((int64_t)h * CS + c) * D * D + i];
    cov[(int64_t)h * D * D + i] = __float2bfloat16_rn(s / (float)T);
}

// The position of the set bit of rank `rank` (from 0) in x.
__device__ __forceinline__ int select64(uint64_t x, int rank) {
    int pos = 0;
#pragma unroll
    for (int w = 32; w >= 1; w >>= 1) {
        const int c = __popcll(x & ((1ull << w) - 1ull));
        if (rank >= c) {
            rank -= c;
            x >>= w;
            pos += w;
        }
    }
    return pos;
}

// The exact tokens of each (query block, key block) pair as exclusive prefix sums along the row, and each row's
// total: a whole exact block's rows, a mixed block's selected tokens, none for a pure one.
__global__ void __launch_bounds__(256) offsets_kernel(const int8_t* flags, int T, int N, int* off, int* tot) {
    __shared__ int wsum[8];
    const int qi = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int8_t* frow = flags + ((int64_t)h * N + qi) * N;
    int* orow = off + ((int64_t)h * N + qi) * N;
    auto count = [&](int j) {
        const int f = frow[j];
        return f == -1 ? min(BLK, T - j * BLK) : max(f, 0);
    };
    const int per = (N + 255) / 256, j0 = tid * per, j1 = min(N, j0 + per);
    int local = 0;
    for (int j = j0; j < j1; ++j) local += count(j);
    int x = local;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int v = lane < 8 ? wsum[lane] : 0;
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += y;
        }
        if (lane < 8) wsum[lane] = v;
    }
    __syncthreads();
    int run = x - local + (warp > 0 ? wsum[warp - 1] : 0);
    for (int j = j0; j < j1; ++j) {
        orow[j] = run;
        run += count(j);
    }
    if (tid == 255) tot[(int64_t)h * N + qi] = run;
}

// ---------------------------------------------------------------------------------------------
// The int8 exact branch's operands, quantized once per call
// ---------------------------------------------------------------------------------------------
// The V scale of channel d: its absmax over the sequence, as Sage's per-channel V.
__device__ __forceinline__ float v_scale(const unsigned* vamax, int d) { return fmaxf(__uint_as_float(vamax[d]) / 127.f, 1e-8f); }

// k less the mean of the key block means (the route's own moments), per-token absmax scale, channels in
// sol::perm_d order; v by channel. A warp per token, a lane per 4 channels.
__global__ void __launch_bounds__(256) quant_kv_kernel(const bf16* k, const bf16* v, int64_t rs, int64_t hs, int T, const float* mom,
                                                        const unsigned* vamax, int8_t* k8, float* ks, int8_t* v8) {
    const int h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x * 8 + warp;
    if (t >= T) return;
    const int c = lane * 4;
    const int64_t src = (int64_t)t * rs + (int64_t)h * hs + c;
    const uint2 kw = *reinterpret_cast<const uint2*>(k + src), vw = *reinterpret_cast<const uint2*>(v + src);
    const bf16* kb = reinterpret_cast<const bf16*>(&kw);
    const bf16* vb = reinterpret_cast<const bf16*>(&vw);
    const float* kmean = mom + h * 2 * D;
    float x[4], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = __bfloat162float(kb[i]) - kmean[c + i];
        amax = fmaxf(amax, fabsf(x[i]));
    }
    amax = warp_max(amax);
    const float sc = fmaxf(amax / 127.f, 1e-8f), inv = 1.f / sc;
    uint32_t kq = 0u, vq = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kq |= (uint32_t)(uint8_t)sol::q8(x[i], inv) << (8 * i);
        vq |= (uint32_t)(uint8_t)sol::q8(__bfloat162float(vb[i]), 1.f / v_scale(vamax + h * D, c + i)) << (8 * i);
    }
    const int64_t row = ((int64_t)h * T + t) * D;
    *reinterpret_cast<uint32_t*>(k8 + row + sol::perm_d(c)) = kq;
    *reinterpret_cast<uint32_t*>(v8 + row + c) = vq;
    if (lane == 0) ks[(int64_t)h * T + t] = sc;
}

// ---------------------------------------------------------------------------------------------
// Joint attention: one CTA per (64-row query block, head), a warp per 16 query rows
// ---------------------------------------------------------------------------------------------
// K tiles are double-buffered and V single-buffered, so two CTAs fit an SM: each step waits for its K tile, takes
// the scores and P while V lands, then multiplies by V and starts the next V load.
constexpr int ATTN_SMEM = 3 * BLK * D * 2 + 2 * BLK * 4 + 4 * 4 + 2 * BLK;

// INT8: the exact rows run as int8 Q.K^T and u8 P . int8 V (Sage's scheme, through comfy-kitchen's Sol-Attn tile
// code), from k8/ks/v8 and the V scales. K is centred there, so the representatives' scores take the same per-row
// shift, q . kmean, and the softmax sees one consistent set of scores.
template <bool INT8>
__global__ void __launch_bounds__(128, 2) attn_kernel(const bf16* q, const bf16* k, const bf16* v, bf16* out, int64_t rs, int64_t hs,
                                                       int64_t ors, int T, int N, int prefix, const bf16* kc, const bf16* vm,
                                                       const int8_t* flags, const uint64_t* bits, const bf16* cov, float scale_log2,
                                                       float cov_scale, const int8_t* k8, const float* ks, const int8_t* v8,
                                                       const unsigned* vamax, const float* mom, const int* off, const int* tot) {
    extern __shared__ __align__(16) unsigned char raw[];
    bf16* sk = reinterpret_cast<bf16*>(raw);               // [2][64][128]
    bf16* sv = sk + 2 * BLK * D;                           // [64][128]
    int* sids = reinterpret_cast<int*>(sv + BLK * D);      // [2][64]
    int* sn = sids + 2 * BLK;                              // [2] (+2 pad)
    int8_t* sft = reinterpret_cast<int8_t*>(sn + 4);       // [2][64]: a representative tile's flags
    const int qi = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if ((qi + 1) * BLK <= prefix) return;
    const int warp = tid >> 5, lane = tid & 31, qd = lane & 3;
    const int8_t* frow = flags + ((int64_t)h * N + qi) * N;
    const uint64_t* brow = bits + ((int64_t)h * N + qi) * N;

    const int ra = qi * BLK + warp * 16 + (lane >> 2), rb = ra + 8;
    uint32_t qa[D / 16][4];
    load_qa(qa, q + (int64_t)min(ra, T - 1) * rs + (int64_t)h * hs, q + (int64_t)min(rb, T - 1) * rs + (int64_t)h * hs, ra < T, rb < T, lane);
    float acc[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    uint32_t pa[BLK / 16][4];
    float s[BLK / 8][4];

    // INT8: q per row, absmax-scaled, in the A layout of m16n8k32 (lane qd: channels kc*32 + qd*4.. and +16), and the
    // row's q . kmean
    uint32_t qa8[sol::TILE_KC][4];
    float qsc[2] = {0.f, 0.f}, qkm[2] = {0.f, 0.f};
    if constexpr (INT8) {
        const float* kmean = mom + h * 2 * D;
        const int rows[2] = {ra, rb};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const bool ok = rows[r] < T;
            const bf16* pr = q + (int64_t)min(rows[r], T - 1) * rs + (int64_t)h * hs;
            float amax = 0.f, dot = 0.f;
#pragma unroll
            for (int kc4 = 0; kc4 < sol::TILE_KC; ++kc4)
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int c = kc4 * 32 + half * 16 + qd * 4;
                    const uint2 w = *reinterpret_cast<const uint2*>(pr + c);
                    const bf16* b = reinterpret_cast<const bf16*>(&w);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float x = ok ? __bfloat162float(b[i]) : 0.f;
                        amax = fmaxf(amax, fabsf(x));
                        dot += x * kmean[c + i];
                    }
                }
#pragma unroll
            for (int o = 1; o <= 2; o <<= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
                dot += __shfl_xor_sync(0xffffffff, dot, o);
            }
            const float sc = fmaxf(amax / 127.f, 1e-8f), inv = 1.f / sc;
            qsc[r] = sc * scale_log2;
            qkm[r] = dot * scale_log2;
#pragma unroll
            for (int kc4 = 0; kc4 < sol::TILE_KC; ++kc4)
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int c = kc4 * 32 + half * 16 + qd * 4;
                    const uint2 w = *reinterpret_cast<const uint2*>(pr + c);
                    const bf16* b = reinterpret_cast<const bf16*>(&w);
                    uint32_t word = 0u;
#pragma unroll
                    for (int i = 0; i < 4; ++i) word |= (uint32_t)(uint8_t)(ok ? sol::q8(__bfloat162float(b[i]), inv) : 0) << (8 * i);
                    qa8[kc4][r + 2 * half] = word;
                }
        }
    }

    // representatives: every block's mean key and value, weighted by the rows it stands for
    const bf16* kch = kc + (int64_t)h * N * D;
    const bf16* vmh = vm + (int64_t)h * N * D;
    auto rep_k = [&](int t) {
        const int buf = t & 1;
        for (int i = tid; i < BLK * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8), j = t * BLK + row;
            cp16(sk + buf * BLK * D + swz(row, c), kch + (int64_t)(j < N ? j : 0) * D + c * 8, j < N);
        }
        if (tid < BLK) sft[buf * BLK + tid] = t * BLK + tid < N ? frow[t * BLK + tid] : (int8_t)-1;
    };
    auto rep_v = [&](int t) {
        for (int i = tid; i < BLK * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8), j = t * BLK + row;
            cp16(sv + swz(row, c), vmh + (int64_t)(j < N ? j : 0) * D + c * 8, j < N);
        }
    };
    const int nrep = (N + BLK - 1) / BLK;
    rep_k(0);
    cp_commit();
    rep_v(0);
    cp_commit();
    for (int t = 0; t < nrep; ++t) {
        if (t + 1 < nrep) rep_k(t + 1);
        cp_commit();
        cp_wait<2>();
        __syncthreads();
        tile_qk(sk + (t & 1) * BLK * D, qa, lane, s);
        const int8_t* ft = sft + (t & 1) * BLK;
#pragma unroll
        for (int nt = 0; nt < BLK / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = nt * 8 + qd * 2 + (e & 1), j = t * BLK + col;
                const int f = ft[col];
                const int rem = j < N && f != -1 ? min(BLK, T - j * BLK) - max(f, 0) : 0;
                s[nt][e] = rem > 0 ? s[nt][e] * scale_log2 - (INT8 ? qkm[e >> 1] : 0.f) + log2f((float)rem) : -INFINITY;
            }
        softmax_p(s, m, l, acc, pa);
        cp_wait<1>();
        __syncthreads();
        tile_pv(sv, pa, lane, acc);
        __syncthreads();
        if (t + 1 < nrep) rep_v(t + 1);
        cp_commit();
    }

    // first-order covariance: acc += l * (q . cov) / sqrt(128), over both K buffers as one 128-row tile
    {
        const bf16* ch = cov + (int64_t)h * D * D;
        for (int i = tid; i < D * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8);
            cp16(sk + swz(row, c), ch + row * D + c * 8, true);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        // each lane holds the sum over its own columns; the term weighs the whole row's represented mass
        float lrow[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lrow[r] = l[r] + __shfl_xor_sync(0xffffffff, l[r], 1);
            lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], 2);
        }
#pragma unroll
        for (int dn = 0; dn < D / 8; dn += 2) {
            float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                uint32_t b[4];
                const int j = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldsm_x4_t(b, sk + swz(j, dn + (lane >> 4)));
                mma(c0, qa[kk], b[0], b[1]);
                mma(c1, qa[kk], b[2], b[3]);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[dn][e] += lrow[e >> 1] * (c0[e] * cov_scale);
                acc[dn + 1][e] += lrow[e >> 1] * (c1[e] * cov_scale);
            }
        }
        __syncthreads();
    }

    // exact rows, compacted into 64-row chunks in block order: whole exact blocks and the selected tokens of mixed ones,
    // as upstream compacts them. Row r of a chunk is position p of the row's list: the last key block whose prefix sum is
    // at most p, and the token of rank p - off[j] in it. Lanes 0 to 15 of each warp find one row each.
    const int* orow = off + ((int64_t)h * N + qi) * N;
    const int total = tot[(int64_t)h * N + qi];
    int chunk = 0;
    auto build = [&](int buf) {
        const int base = chunk++ * BLK, n = max(0, min(BLK, total - base));
        const int r = warp * 16 + lane;
        if (lane < 16 && r < n) {
            const int p = base + r;
            int lo = 0, hi = N - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (orow[mid] <= p) lo = mid;
                else hi = mid - 1;
            }
            const int rank = p - orow[lo];
            sids[buf * BLK + r] = lo * BLK + (frow[lo] == -1 ? rank : select64(brow[lo], rank));
        }
        if (tid == 0) sn[buf] = n;
    };
    auto ex_k = [&](int buf) {
        const int n = sn[buf];
        for (int i = tid; i < BLK * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8);
            const bool ok = row < n;
            cp16(sk + buf * BLK * D + swz(row, c), k + (int64_t)(ok ? sids[buf * BLK + row] : 0) * rs + (int64_t)h * hs + c * 8, ok);
        }
    };
    auto ex_v = [&](int buf) {
        const int n = sn[buf];
        for (int i = tid; i < BLK * (D / 8); i += blockDim.x) {
            const int row = i / (D / 8), c = i % (D / 8);
            const bool ok = row < n;
            cp16(sv + swz(row, c), v + (int64_t)(ok ? sids[buf * BLK + row] : 0) * rs + (int64_t)h * hs + c * 8, ok);
        }
    };
    if constexpr (INT8) {
        // the representatives' state in the int8 branch's units: P scaled by 255, o by 1 / (V scale), l the whole row's
        const int g = lane >> 2;
        const unsigned* vah = vamax + h * D;
        float lrow[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            lrow[r] = l[r] + __shfl_xor_sync(0xffffffff, l[r], 1);
            lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], 2);
        }
        float m_r[2] = {m[0], m[1]}, c_r[2] = {m[0], m[1]};
        l[0] = lrow[0] * 255.f;
        l[1] = lrow[1] * 255.f;
#pragma unroll
        for (int dn = 0; dn < D / 8; ++dn)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[dn][e] *= 255.f / v_scale(vah, dn * 8 + qd * 2 + (e & 1));

        int8_t* sk8 = reinterpret_cast<int8_t*>(raw);     // [2][64][128], rows in sol::perm_key order
        int8_t* svr = sk8 + 2 * BLK * D;                    // [64][128] gathered V rows, 16-byte chunks XOR (row / 4)
        int8_t* svt = svr + BLK * D;                        // [128][64] V^T, keys in sol::perm_d order
        float2* sksb = reinterpret_cast<float2*>(svt + D * BLK);  // [2][64] (k scale, bias) by stored row
        const int8_t* k8h = k8 + (int64_t)h * T * D;
        const int8_t* v8h = v8 + (int64_t)h * T * D;
        const float* ksh = ks + (int64_t)h * T;
        auto ex_k8 = [&](int buf) {
            const int n = sn[buf];
            for (int i = tid; i < BLK * (D / 16); i += blockDim.x) {
                const int p = i / (D / 16), c16 = i % (D / 16), j = sol::perm_key(p);
                const bool ok = j < n;
                cp16(sk8 + buf * BLK * D + p * D + ((c16 ^ sol::swz_k(p)) << 4), k8h + (int64_t)(ok ? sids[buf * BLK + j] : 0) * D + c16 * 16, ok);
            }
            if (tid < BLK) {
                const int j = sol::perm_key(tid);
                sksb[buf * BLK + tid] = j < n ? make_float2(ksh[sids[buf * BLK + j]], 0.f) : make_float2(0.f, sol::NEG);
            }
        };
        auto ex_v8 = [&](int buf) {
            const int n = sn[buf];
            for (int i = tid; i < BLK * (D / 16); i += blockDim.x) {
                const int j = i / (D / 16), c16 = i % (D / 16);
                const bool ok = j < n;
                cp16(svr + j * D + ((c16 ^ ((j >> 2) & 7)) << 4), v8h + (int64_t)(ok ? sids[buf * BLK + j] : 0) * D + c16 * 16, ok);
            }
        };
        // V rows to V^T, 4 keys x 4 channels per step through byte permutes
        auto transpose = [&]() {
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int j4 = lane & 15, c4 = it * 8 + warp * 2 + (lane >> 4);
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = j4 * 4 + i;
                    w[i] = *reinterpret_cast<const uint32_t*>(svr + r * D + (((c4 >> 2) ^ (j4 & 7)) << 4) + (c4 & 3) * 4);
                }
                const uint32_t t0 = __byte_perm(w[0], w[1], 0x5140), t1 = __byte_perm(w[2], w[3], 0x5140);
                const uint32_t t2 = __byte_perm(w[0], w[1], 0x7362), t3 = __byte_perm(w[2], w[3], 0x7362);
                const uint32_t u[4] = {__byte_perm(t0, t1, 0x5410), __byte_perm(t0, t1, 0x7632), __byte_perm(t2, t3, 0x5410),
                                       __byte_perm(t2, t3, 0x7632)};
                const int kp = sol::perm_d(j4 * 4);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int C = c4 * 4 + c;
                    *reinterpret_cast<uint32_t*>(svt + C * BLK + (((kp >> 4) ^ sol::swz_v(C)) << 4) + (kp & 15)) = u[c];
                }
            }
        };
        build(0);
        __syncthreads();
        ex_k8(0);
        cp_commit();
        ex_v8(0);
        cp_commit();
        for (int t = 0;; ++t) {
            const int n = sn[t & 1];
            if (n == 0) break;
            build((t + 1) & 1);
            __syncthreads();
            ex_k8((t + 1) & 1);
            cp_commit();
            cp_wait<1>();
            __syncthreads();
            transpose();
            __syncthreads();
            ex_v8((t + 1) & 1);
            cp_commit();
            int32_t s_acc[sol::TILE_NKT][4];
            sol::tile_qk(sk8 + (t & 1) * BLK * D, qa8, g, qd, s_acc);
            float bmax[2] = {m_r[0] - 20.f, m_r[1] - 20.f};
            bool has[2] = {true, true};
            sol::tile_scores<true>(s_acc, sksb + (t & 1) * BLK, qsc, qd, s, bmax);
            sol::tile_softmax_pv<false>(s, bmax, has, svt, g, qd, m_r, l, c_r, acc);
        }
    } else {
        build(0);
        __syncthreads();
        ex_k(0);
        cp_commit();
        ex_v(0);
        cp_commit();
        for (int t = 0;; ++t) {
            const int n = sn[t & 1];
            if (n == 0) break;
            build((t + 1) & 1);
            __syncthreads();
            ex_k((t + 1) & 1);
            cp_commit();
            cp_wait<2>();
            __syncthreads();
            tile_qk(sk + (t & 1) * BLK * D, qa, lane, s);
    #pragma unroll
            for (int nt = 0; nt < BLK / 8; ++nt)
    #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int col = nt * 8 + qd * 2 + (e & 1);
                    s[nt][e] = col < n ? s[nt][e] * scale_log2 : -INFINITY;
                }
            softmax_p(s, m, l, acc, pa);
            cp_wait<1>();
            __syncthreads();
            tile_pv(sv, pa, lane, acc);
            __syncthreads();
            ex_v((t + 1) & 1);
            cp_commit();
        }
    }
    cp_wait<0>();
    const int rows[2] = {ra, rb};
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = rows[rr];
        float lr = l[rr] + __shfl_xor_sync(0xffffffff, l[rr], 1);
        lr += __shfl_xor_sync(0xffffffff, lr, 2);
        if (r >= T || r < prefix) continue;
        bf16* o = out + (int64_t)r * ors + (int64_t)h * D + qd * 2;
        if constexpr (INT8) {
            // the int8 branch keeps l as the whole row's sum, in its P units
            const float inv = 1.f / l[rr];
            const unsigned* vah = vamax + h * D;
#pragma unroll
            for (int dn = 0; dn < D / 8; ++dn) {
                const int c = dn * 8 + qd * 2;
                *reinterpret_cast<__nv_bfloat162*>(o + dn * 8) =
                    __floats2bfloat162_rn(acc[dn][rr * 2] * inv * v_scale(vah, c), acc[dn][rr * 2 + 1] * inv * v_scale(vah, c + 1));
            }
        } else {
            const float inv = 1.f / lr;
#pragma unroll
            for (int dn = 0; dn < D / 8; ++dn)
                *reinterpret_cast<__nv_bfloat162*>(o + dn * 8) = __floats2bfloat162_rn(acc[dn][rr * 2] * inv, acc[dn][rr * 2 + 1] * inv);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------------------------
struct Layout {
    int64_t qb, kc, vm, kvar, cst, mom, cut, part, flags, bits, off, tot, covp, cov, k8, ks, v8, vamax, total;
};

int64_t up256(int64_t x) { return (x + 255) / 256 * 256; }

Layout layout(int t, int h, int hc, bool int8) {
    const int64_t n = (t + BLK - 1) / BLK, S = (n + MU_SPLIT - 1) / MU_SPLIT, CS = (n + COV_SPLIT - 1) / COV_SPLIT;
    Layout l{};
    int64_t o = 0;
    auto take = [&](int64_t bytes) {
        const int64_t at = o;
        o += up256(bytes);
        return at;
    };
    l.qb = take(hc * n * D * 4);
    l.kc = take(hc * n * D * 2);
    l.vm = take(hc * n * D * 2);
    l.kvar = take(hc * n * D * 4);
    l.cst = take(hc * n * 4);
    l.mom = take(hc * 2 * D * 4);
    l.cut = take(hc * n * 4);
    l.part = take(hc * n * S * 3 * 4);
    l.flags = take(hc * n * n);
    l.bits = take(hc * n * n * 8);
    l.off = take(hc * n * n * 4);
    l.tot = take(hc * n * 4);
    l.covp = take(hc * CS * D * D * 4);
    l.cov = take(hc * D * D * 2);
    if (int8) {
        l.k8 = take(hc * t * D);
        l.ks = take(hc * t * 4);
        l.v8 = take(hc * t * D);
        l.vamax = take(hc * D * 4);
    }
    l.total = o;
    return l;
}

size_t classify_smem() { return (size_t)((CQ + 2 * CJ) * CP + CJ + 2 * CQ) * 4; }

}  // namespace sol3
using namespace sol3;

// Workspace bytes for `t` tokens and `h` heads, run `hc` heads at a time, with the int8 exact branch's operands when
// `int8` is set; -1 for a shape it can't take.
GPU_API int64_t gpu_sol3_workspace(int t, int h, int hc, int int8) {
    if (t <= 0 || h <= 0 || hc <= 0 || hc > h || (t + BLK - 1) / BLK > 65535) return -1;
    return layout(t, h, hc, int8 != 0).total;
}

// Where the route of the first head chunk sits in the workspace, for tests: the flags (int8 [hc][N][N]: -1 exact,
// -2 pure, else the selected token count), the token bits (u64, same shape) and the cuts (f32 [hc][N]).
GPU_API int gpu_sol3_route_offsets(int t, int h, int hc, int int8, int64_t* out) {
    if (gpu_sol3_workspace(t, h, hc, int8) < 0) return -1;
    const Layout L = layout(t, h, hc, int8 != 0);
    out[0] = L.flags;
    out[1] = L.bits;
    out[2] = L.cut;
    return 0;
}

// Sol³ attention over post-rope q/k/v (head dim 128, batch 1, element strides `rs` per row and `hs` per head) into
// `out` (row stride `ors`, head h at column h * 128). The first `prefix` rows are the protected conditioning: every
// query attends their keys and values exactly. Their own query rows are left to the caller, which attends them densely
// (sol3_attn's `dense` callback). `int8` runs the exact rows in int8 (the workspace must have been sized for it);
// otherwise everything is bf16, as upstream.
GPU_API int gpu_sol3_attn(const void* q, const void* k, const void* v, void* out, void* ws, int t, int h, int hc, int64_t rs,
                          int64_t hs, int64_t ors, int prefix, int int8, void* stream) {
    if (gpu_sol3_workspace(t, h, hc, int8) < 0 || prefix < 0 || prefix > t) return -1;
    if (prefix == t) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    const bool i8 = int8 != 0;
    const Layout L = layout(t, h, hc, i8);
    unsigned char* w = (unsigned char*)ws;
    const double scale_d = 1.0 / std::sqrt((double)D);
    const float scale = (float)scale_d;
    const float scale_log2 = (float)(scale_d * 1.4426950408889634);
    const float cov_scale = (float)scale_d * GAMMA;
    const int N = (t + BLK - 1) / BLK, S = (N + MU_SPLIT - 1) / MU_SPLIT, CS = (N + COV_SPLIT - 1) / COV_SPLIT;
    const int s1 = (prefix + BLK - 1) / BLK;
    static bool configured = false;
    const size_t csm = classify_smem();
    if (!configured) {
        if (cudaFuncSetAttribute(attn_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM) != cudaSuccess) return -2;
        if (cudaFuncSetAttribute(attn_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_SMEM) != cudaSuccess) return -2;
        if (cudaFuncSetAttribute(classify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)csm) != cudaSuccess) return -2;
        configured = true;
    }
    float* qb = (float*)(w + L.qb);
    bf16* kc = (bf16*)(w + L.kc);
    bf16* vm = (bf16*)(w + L.vm);
    float* kvar = (float*)(w + L.kvar);
    float* cst = (float*)(w + L.cst);
    float* mom = (float*)(w + L.mom);
    float* cut = (float*)(w + L.cut);
    float* part = (float*)(w + L.part);
    int8_t* flags = (int8_t*)(w + L.flags);
    uint64_t* bits = (uint64_t*)(w + L.bits);
    int* off = (int*)(w + L.off);
    int* tot = (int*)(w + L.tot);
    float* covp = (float*)(w + L.covp);
    bf16* cov = (bf16*)(w + L.cov);
    int8_t* k8 = i8 ? (int8_t*)(w + L.k8) : nullptr;
    float* ks = i8 ? (float*)(w + L.ks) : nullptr;
    int8_t* v8 = i8 ? (int8_t*)(w + L.v8) : nullptr;
    unsigned* vamax = i8 ? (unsigned*)(w + L.vamax) : nullptr;
    for (int h0 = 0; h0 < h; h0 += hc) {
        const int nh = std::min(hc, h - h0);
        const bf16* qh = (const bf16*)q + (int64_t)h0 * hs;
        const bf16* kh = (const bf16*)k + (int64_t)h0 * hs;
        const bf16* vh = (const bf16*)v + (int64_t)h0 * hs;
        bf16* oh = (bf16*)out + (int64_t)h0 * D;
        if (i8 && cudaMemsetAsync(vamax, 0, (size_t)nh * D * 4, st) != cudaSuccess) return -3;
        pool_kernel<<<dim3(N, nh), D, 0, st>>>(qh, kh, vh, rs, hs, t, N, qb, kc, vm, kvar, cst, vamax);
        moments_kernel<<<nh, D, 0, st>>>(kc, N, mom);
        if (i8) quant_kv_kernel<<<dim3((t + 7) / 8, nh), 256, 0, st>>>(kh, vh, rs, hs, t, mom, vamax, k8, ks, v8);
        classify_kernel<<<dim3((N + CQ - 1) / CQ, nh), 256, csm, st>>>(qb, kc, kvar, cst, mom, N, prefix, s1, scale, flags);
        const dim3 rg((N + BLK - 1) / BLK, S, nh);
        route_kernel<false><<<rg, 128, 0, st>>>(qb, kh, rs, hs, t, N, prefix, scale, S, flags, bits, part, cut);
        cut_kernel<<<(nh * N + 255) / 256, 256, 0, st>>>(part, S, nh * N, cut);
        route_kernel<true><<<rg, 128, 0, st>>>(qb, kh, rs, hs, t, N, prefix, scale, S, flags, bits, part, cut);
        offsets_kernel<<<dim3(N, nh), 256, 0, st>>>(flags, t, N, off, tot);
        cov_kernel<<<dim3(CS, nh), 256, 0, st>>>(kh, vh, rs, hs, t, N, s1, CS, covp);
        cov_reduce_kernel<<<dim3(D * D / 256, nh), 256, 0, st>>>(covp, CS, t, cov);
        if (i8)
            attn_kernel<true><<<dim3(N, nh), 128, ATTN_SMEM, st>>>(qh, kh, vh, oh, rs, hs, ors, t, N, prefix, kc, vm, flags, bits, cov,
                                                                  scale_log2, cov_scale, k8, ks, v8, vamax, mom, off, tot);
        else
            attn_kernel<false><<<dim3(N, nh), 128, ATTN_SMEM, st>>>(qh, kh, vh, oh, rs, hs, ors, t, N, prefix, kc, vm, flags, bits, cov,
                                                                   scale_log2, cov_scale, nullptr, nullptr, nullptr, nullptr, mom, off, tot);
        KCHECK();
    }
    return 0;
}
