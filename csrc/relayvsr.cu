// RelayVSR (comfy.ldm.wan.model_vsr): the sliding-window attention both of its networks use, and the small steps around
// them that ComfyUI runs as separate tensor operations.
//
// Window attention follows comfy-kitchen's `window_attention`: query plane t attends key plane t + Tk - Tq and the
// `temporal_window - 1` before it; in key plane p a query at (y, x) sees the (wh, ww) keys centred on it, shifted inward
// at the border when `clamp` is set, cut off by it otherwise. Scores and the softmax are float32, the probabilities are
// rounded to the activation type for the product with v, as a flash attention kernel does.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Tensor-core helpers (sm_80+): mma.m16n8k16 with float accumulators, ldmatrix, cp.async.
// ---------------------------------------------------------------------------------------------
template <typename T> struct MmaOp;
template <> struct MmaOp<bf16> {
    static __device__ __forceinline__ void run(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
        __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    }
};
template <> struct MmaOp<half> {
    static __device__ __forceinline__ void run(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    static __device__ __forceinline__ uint32_t pack(float lo, float hi) {
        __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    }
};

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

// ---------------------------------------------------------------------------------------------
// Window attention
// ---------------------------------------------------------------------------------------------
constexpr int WA_MAXP = 4;   // key planes one query plane attends
constexpr int WA_ROWS = 4;   // query rows per block, one per warp
constexpr int WA_COLS = 16;  // query columns per warp: one m16 tile
constexpr int WA_KT = 64;    // keys per tile, a run along one key row

struct WinArgs {
    const void* q;
    void* o;
    int64_t q_rs, q_ps, o_rs, o_ps;  // row and plane strides (elements)
    // explicit planes (n > 0): every query plane attends these, in order
    const void* k[WA_MAXP];
    const void* v[WA_MAXP];
    int64_t krs[WA_MAXP], vrs[WA_MAXP];
    int wh[WA_MAXP], ww[WA_MAXP];
    int n;
    // uniform planes (n == 0): plane p at kb + p * k_ps, all with one window
    const void* kb;
    const void* vb;
    int64_t k_ps, v_ps;
    int uwh, uww, tk, tq, tw;
    int64_t k_rs, v_rs;
    int H, W, heads, kv_heads, clamp;
    float scale_log2;
};

// Start of the window along one axis of `size` for position i.
__device__ __forceinline__ int win_lo(int i, int win, int size, int clamp) {
    if (clamp) {
        const int w = min(win, size);
        return min(max(i - w / 2, 0), size - w);
    }
    return i - win / 2;
}
__device__ __forceinline__ int win_len(int win, int size, int clamp) { return clamp ? min(win, size) : win; }

template <typename T, int HD>
__global__ void __launch_bounds__(128) win_attn_kernel(WinArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* sk = reinterpret_cast<T*>(smem_raw);  // [2][KT][HD] then v [2][KT][HD]
    T* sv = sk + 2 * WA_KT * HD;
    constexpr int CH = HD / 8;  // 16-byte chunks per row
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int H = a.H, W = a.W;
    const int tiles_x = (W + WA_COLS - 1) / WA_COLS;
    const int qy0 = (blockIdx.x / tiles_x) * WA_ROWS, qx0 = (blockIdx.x % tiles_x) * WA_COLS;
    const int head = blockIdx.y, tplane = blockIdx.z;
    const int kvh = head / (a.heads / a.kv_heads);
    const int qy = qy0 + warp;
    const int qy_last = min(qy0 + WA_ROWS, H) - 1, qx_last = min(qx0 + WA_COLS, W) - 1;

    // the key planes of this query plane
    int np;
    const T* kp[WA_MAXP];
    const T* vp[WA_MAXP];
    int pwh[WA_MAXP], pww[WA_MAXP];
    int64_t pks[WA_MAXP], pvs[WA_MAXP];
    if (a.n > 0) {
        np = a.n;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) {
            if (i < np) {
                kp[i] = (const T*)a.k[i];
                vp[i] = (const T*)a.v[i];
                pks[i] = a.krs[i];
                pvs[i] = a.vrs[i];
                pwh[i] = a.wh[i];
                pww[i] = a.ww[i];
            }
        }
    } else {
        const int last = tplane + a.tk - a.tq;
        const int first = max(0, last - a.tw + 1);
        np = last - first + 1;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) {
            if (i < np) {
                kp[i] = (const T*)a.kb + (int64_t)(first + i) * a.k_ps;
                vp[i] = (const T*)a.vb + (int64_t)(first + i) * a.v_ps;
                pks[i] = a.k_rs;
                pvs[i] = a.v_rs;
                pwh[i] = a.uwh;
                pww[i] = a.uww;
            }
        }
    }
    // per plane: the rows and columns any query of the block reads, and the key tiles covering them
    int ry0[WA_MAXP], ry1[WA_MAXP], rx0[WA_MAXP], rx1[WA_MAXP], ntiles[WA_MAXP], nch[WA_MAXP];
    int total = 0;
#pragma unroll
    for (int i = 0; i < WA_MAXP; ++i) {
        if (i >= np) break;
        const int lh = win_len(pwh[i], H, a.clamp), lw = win_len(pww[i], W, a.clamp);
        ry0[i] = max(0, win_lo(qy0, pwh[i], H, a.clamp));
        ry1[i] = min(H, win_lo(qy_last, pwh[i], H, a.clamp) + lh);
        rx0[i] = max(0, win_lo(qx0, pww[i], W, a.clamp));
        rx1[i] = min(W, win_lo(qx_last, pww[i], W, a.clamp) + lw);
        nch[i] = (rx1[i] - rx0[i] + WA_KT - 1) / WA_KT;
        ntiles[i] = max(0, ry1[i] - ry0[i]) * nch[i];
        total += ntiles[i];
    }
    // which tile is the i-th: (plane, key row, first key column)
    auto tile = [&](int t, int& p, int& ky, int& kx0) {
        p = 0;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) {
            if (i < np - 1 && t >= ntiles[i]) {
                t -= ntiles[i];
                p = i + 1;
            } else {
                break;
            }
        }
        int c = 0;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) if (i == p) c = nch[i];
        ky = t / c;
        kx0 = t % c;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) {
            if (i == p) {
                ky += ry0[i];
                kx0 = rx0[i] + kx0 * WA_KT;
            }
        }
    };
    auto load = [&](int t, int buf) {
        int p, ky, kx0;
        tile(t, p, ky, kx0);
        const T* kpl = nullptr;
        const T* vpl = nullptr;
        int x1 = 0;
        int64_t ks = 0, vs = 0;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) {
            if (i == p) {
                kpl = kp[i];
                vpl = vp[i];
                ks = pks[i];
                vs = pvs[i];
                x1 = rx1[i];
            }
        }
        T* dk = sk + buf * WA_KT * HD;
        T* dv = sv + buf * WA_KT * HD;
        for (int i = threadIdx.x; i < WA_KT * CH; i += blockDim.x) {
            const int j = i / CH, c = i % CH;
            const int kx = kx0 + j;
            const bool ok = kx < x1;
            const int64_t tok = (int64_t)ky * W + (ok ? kx : 0);
            const int sw = (c ^ (j & 7)) * 8;
            cp16(dk + j * HD + sw, kpl + tok * ks + kvh * HD + c * 8, ok);
            cp16(dv + j * HD + sw, vpl + tok * vs + kvh * HD + c * 8, ok);
        }
    };

    // q fragments: rows lane / 4 and lane / 4 + 8 of the warp's 16 columns
    const int r0 = lane >> 2, cq = (lane & 3) * 2;
    const int xa = qx0 + r0, xb = qx0 + r0 + 8;
    const bool row_ok = qy < H;
    uint32_t qa[HD / 16][4];
    {
        const T* qb = (const T*)a.q + (int64_t)tplane * a.q_ps + head * HD;
        const bool oka = row_ok && xa < W, okb = row_ok && xb < W;
        const T* pa = qb + ((int64_t)qy * W + xa) * a.q_rs;
        const T* pb = qb + ((int64_t)qy * W + xb) * a.q_rs;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ++ks) {
            const int c = ks * 16 + cq;
            qa[ks][0] = oka ? *reinterpret_cast<const uint32_t*>(pa + c) : 0u;
            qa[ks][1] = okb ? *reinterpret_cast<const uint32_t*>(pb + c) : 0u;
            qa[ks][2] = oka ? *reinterpret_cast<const uint32_t*>(pa + c + 8) : 0u;
            qa[ks][3] = okb ? *reinterpret_cast<const uint32_t*>(pb + c + 8) : 0u;
        }
    }
    float oacc[HD / 8][4];
#pragma unroll
    for (int i = 0; i < HD / 8; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

    if (total > 0) {
        load(0, 0);
        cp_commit();
    }
    for (int t = 0; t < total; ++t) {
        if (t + 1 < total) load(t + 1, (t + 1) & 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        int p, ky, kx0;
        tile(t, p, ky, kx0);
        int wh = 0, ww = 0;
#pragma unroll
        for (int i = 0; i < WA_MAXP; ++i) {
            if (i == p) {
                wh = pwh[i];
                ww = pww[i];
            }
        }
        const int ylo = win_lo(qy, wh, H, a.clamp);
        const bool use = row_ok && ky >= ylo && ky < ylo + win_len(wh, H, a.clamp);
        if (use) {
            const T* bk = sk + (t & 1) * WA_KT * HD;
            const T* bv = sv + (t & 1) * WA_KT * HD;
            float s[WA_KT / 8][4];
#pragma unroll
            for (int nt = 0; nt < WA_KT / 8; ++nt) {
                s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
                for (int ks = 0; ks < HD / 16; ks += 2) {
                    uint32_t b[4];
                    const int j = nt * 8 + (lane & 7);
                    const int c = 2 * ks + (lane >> 3);
                    ldsm_x4(b, bk + j * HD + ((c ^ (j & 7)) * 8));
                    MmaOp<T>::run(s[nt], qa[ks], b[0], b[1]);
                    MmaOp<T>::run(s[nt], qa[ks + 1], b[2], b[3]);
                }
            }
            // mask to each query's window and the grid
            const int lw = win_len(ww, W, a.clamp);
            const int la = win_lo(xa, ww, W, a.clamp), lb = win_lo(xb, ww, W, a.clamp);
            float tmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int nt = 0; nt < WA_KT / 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int kx = kx0 + nt * 8 + cq + (e & 1);
                    const int lo = e < 2 ? la : lb;
                    const bool ok = kx < W && kx >= lo && kx < lo + lw;
                    s[nt][e] = ok ? s[nt][e] * a.scale_log2 : -INFINITY;
                    tmax[e >> 1] = fmaxf(tmax[e >> 1], s[nt][e]);
                }
            }
            float alpha[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 1));
                tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffff, tmax[r], 2));
                const float mn = fmaxf(mrow[r], tmax[r]);
                alpha[r] = mn == -INFINITY ? 1.f : exp2f(mrow[r] - mn);
                mrow[r] = mn;
                lrow[r] *= alpha[r];
            }
            // tile nt's accumulators are (row a, keys 0-7 of a k step) and (row b, ...) for even nt, keys 8-15 for odd:
            // exactly the A fragment of the product with v
            uint32_t pa[WA_KT / 16][4];
#pragma unroll
            for (int nt = 0; nt < WA_KT / 8; ++nt) {
                float pv[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float m = mrow[e >> 1];
                    pv[e] = m == -INFINITY ? 0.f : exp2f(s[nt][e] - m);
                    lrow[e >> 1] += pv[e];
                }
                pa[nt >> 1][(nt & 1) * 2 + 0] = MmaOp<T>::pack(pv[0], pv[1]);
                pa[nt >> 1][(nt & 1) * 2 + 1] = MmaOp<T>::pack(pv[2], pv[3]);
            }
#pragma unroll
            for (int dn = 0; dn < HD / 8; ++dn) {
                oacc[dn][0] *= alpha[0];
                oacc[dn][1] *= alpha[0];
                oacc[dn][2] *= alpha[1];
                oacc[dn][3] *= alpha[1];
            }
#pragma unroll
            for (int kk = 0; kk < WA_KT / 16; ++kk) {
#pragma unroll
                for (int dn = 0; dn < HD / 8; dn += 2) {
                    uint32_t b[4];
                    const int j = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int c = dn + (lane >> 4);
                    ldsm_x4_t(b, bv + j * HD + ((c ^ (j & 7)) * 8));
                    MmaOp<T>::run(oacc[dn], pa[kk], b[0], b[1]);
                    MmaOp<T>::run(oacc[dn + 1], pa[kk], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    if (!row_ok) return;
    float l[2] = {lrow[0], lrow[1]};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
        l[r] = l[r] > 0.f ? 1.f / l[r] : 0.f;
    }
    T* ob = (T*)a.o + (int64_t)tplane * a.o_ps + head * HD;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int x = r ? xb : xa;
        if (x >= W) continue;
        T* orow = ob + ((int64_t)qy * W + x) * a.o_rs;
#pragma unroll
        for (int dn = 0; dn < HD / 8; ++dn) {
            const uint32_t v = MmaOp<T>::pack(oacc[dn][2 * r] * l[r], oacc[dn][2 * r + 1] * l[r]);
            *reinterpret_cast<uint32_t*>(orow + dn * 8 + cq) = v;
        }
    }
}

GPU_API int gpu_win_attn(int dtype, int hd, const WinArgs* args, float scale, void* stream) {
    WinArgs a = *args;
    a.scale_log2 = scale * 1.4426950408889634f;
    if (a.n > WA_MAXP || (a.n == 0 && a.tw > WA_MAXP) || a.heads % a.kv_heads != 0) return -1;
    if (a.n > 0) a.tq = 1;
    const int tiles = ((a.H + WA_ROWS - 1) / WA_ROWS) * ((a.W + WA_COLS - 1) / WA_COLS);
    const dim3 grid(tiles, a.heads, a.tq);
    cudaStream_t st = (cudaStream_t)stream;
#define WA_LAUNCH(T, D)                                                                                         \
    {                                                                                                           \
        const int smem = 4 * WA_KT * D * (int)sizeof(T);                                                        \
        static bool set = false;                                                                                \
        if (!set) {                                                                                             \
            cudaFuncSetAttribute(win_attn_kernel<T, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);     \
            set = true;                                                                                         \
        }                                                                                                       \
        win_attn_kernel<T, D><<<grid, 128, smem, st>>>(a);                                                      \
        KCHECK();                                                                                               \
        return 0;                                                                                               \
    }
    if (dtype == DT_BF16 && hd == 128) WA_LAUNCH(bf16, 128)
    if (dtype == DT_BF16 && hd == 64) WA_LAUNCH(bf16, 64)
    if (dtype == DT_F16 && hd == 128) WA_LAUNCH(half, 128)
    if (dtype == DT_F16 && hd == 64) WA_LAUNCH(half, 64)
#undef WA_LAUNCH
    return -2;
}

// ---------------------------------------------------------------------------------------------
// RMSNorm over `k`-wide groups of each row (`groups` of them, back to back), optionally times a float weight, rounded
// to T; then plus `add` (T [k]), rounded again. out may alias x.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void relay_rms_kernel(const T* x, int64_t x_rs, T* out, int64_t out_rs, const float* w, const T* add, float eps, int k,
                                 int groups) {
    __shared__ float red[33];
    const int64_t row = blockIdx.x / groups;
    const int g = blockIdx.x % groups;
    const T* xr = x + row * x_rs + (int64_t)g * k;
    T* orow = out + row * out_rs + (int64_t)g * k;
    float ss = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const float v = ldf(xr, i);
        ss += v * v;
    }
    const float rstd = rsqrtf(block_sum(ss, red) / (float)k + eps);
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float v = ldf(xr, i) * rstd;
        if (w) v *= w[i];
        v = rnd<T>(v);
        if (add) v = rnd<T>(v + ldf(add, i));
        stf(orow, i, v);
    }
}

GPU_API int gpu_relay_rms(int dtype, const void* x, int64_t x_rs, void* out, int64_t out_rs, const float* w, const void* add, float eps,
                          int64_t rows, int k, int groups, void* stream) {
    if (rows == 0) return 0;
    const unsigned blocks = (unsigned)(rows * groups);
    const int threads = k >= 1024 ? 256 : 64;
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_BF16) relay_rms_kernel<bf16><<<blocks, threads, 0, st>>>((const bf16*)x, x_rs, (bf16*)out, out_rs, w, (const bf16*)add, eps, k, groups);
    else if (dtype == DT_F16) relay_rms_kernel<half><<<blocks, threads, 0, st>>>((const half*)x, x_rs, (half*)out, out_rs, w, (const half*)add, eps, k, groups);
    else return -1;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Flux-style RoPE (`apply_rope1`) over the heads of each row, three axes: pairs [0, pt) from the plane's row of `tt`,
// [pt, pt + ph) from the row's `th`, the rest from the column's `tw`. Tables are float (cos, sin) pairs. Row r is
// plane r / (H W), row (r / W) % H, column r % W. out may alias x.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void relay_rope_kernel(const T* x, int64_t x_rs, T* out, int64_t out_rs, int heads, int hd, int64_t rows, int H, int W,
                                  const float* tt, const float* th, const float* tw, int pt, int ph) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = hd / 2;
    if (i >= rows * heads * pairs) return;
    const int j = (int)(i % pairs);
    const int64_t rh = i / pairs;
    const int h = (int)(rh % heads);
    const int64_t r = rh / heads;
    const int64_t plane = r / ((int64_t)H * W);
    const int y = (int)((r / W) % H), xx = (int)(r % W);
    const float* cs;
    if (j < pt) cs = tt + (plane * pt + j) * 2;
    else if (j < pt + ph) cs = th + ((int64_t)y * ph + (j - pt)) * 2;
    else cs = tw + ((int64_t)xx * (pairs - pt - ph) + (j - pt - ph)) * 2;
    const T* src = x + r * x_rs + h * hd + 2 * j;
    const float x0 = ldf(src, 0), x1 = ldf(src, 1);
    T* dst = out + r * out_rs + h * hd + 2 * j;
    stf(dst, 0, cs[0] * x0 - cs[1] * x1);
    stf(dst, 1, cs[1] * x0 + cs[0] * x1);
}

GPU_API int gpu_relay_rope(int dtype, const void* x, int64_t x_rs, void* out, int64_t out_rs, int heads, int hd, int64_t rows, int H, int W,
                           const float* tt, const float* th, const float* tw, int pt, int ph, void* stream) {
    const int64_t n = rows * heads * (hd / 2);
    if (n == 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_BF16) relay_rope_kernel<bf16><<<g, 256, 0, st>>>((const bf16*)x, x_rs, (bf16*)out, out_rs, heads, hd, rows, H, W, tt, th, tw, pt, ph);
    else if (dtype == DT_F16) relay_rope_kernel<half><<<g, 256, 0, st>>>((const half*)x, x_rs, (half*)out, out_rs, heads, hd, rows, H, W, tt, th, tw, pt, ph);
    else return -1;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The keyframe model's cache: bf16 rows of 128 (one head of one token) as int8 with a bf16 scale each, as
// `RelayVSRSelfAttention` stores them, and back.
// ---------------------------------------------------------------------------------------------
__global__ void relay_kv_quant_kernel(const bf16* src, int64_t src_rs, int8_t* q, bf16* s, int64_t rows, int heads) {
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows * heads) return;
    const int64_t r = warp / heads;
    const int h = (int)(warp % heads);
    const bf16* x = src + r * src_rs + h * 128;
    float v[4];
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = __bfloat162float(x[lane * 4 + j]);
        m = fmaxf(m, fabsf(v[j]));
    }
    m = warp_max(m);
    // amax / 127 in bf16, clamped to bf16's smallest normal
    const float scb = fmaxf(rnd<bf16>(m / 127.f), 1.17549435e-38f);
    if (lane == 0) s[warp] = __float2bfloat16_rn(scb);
    int8_t* o = q + warp * 128;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[lane * 4 + j] = (int8_t)fminf(127.f, fmaxf(-127.f, nearbyintf(v[j] / scb)));
}

__global__ void relay_kv_dequant_kernel(const int8_t* q, const bf16* s, bf16* dst, int64_t dst_rs, int64_t rows, int heads) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * 128) return;
    const int64_t rh = i / 128;
    const int c = (int)(i % 128);
    const int64_t r = rh / heads;
    const int h = (int)(rh % heads);
    dst[r * dst_rs + h * 128 + c] = __float2bfloat16_rn((float)q[i] * __bfloat162float(s[rh]));
}

GPU_API int gpu_relay_kv_quant(const void* src, int64_t src_rs, void* q, void* s, int64_t rows, int heads, void* stream) {
    const int64_t warps = rows * heads;
    if (warps == 0) return 0;
    relay_kv_quant_kernel<<<(unsigned)((warps * 32 + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)src, src_rs, (int8_t*)q, (bf16*)s, rows,
                                                                                                    heads);
    KCHECK();
    return 0;
}

GPU_API int gpu_relay_kv_dequant(const void* q, const void* s, void* dst, int64_t dst_rs, int64_t rows, int heads, void* stream) {
    const int64_t n = rows * heads * 128;
    if (n == 0) return 0;
    relay_kv_dequant_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const int8_t*)q, (const bf16*)s, (bf16*)dst, dst_rs, rows,
                                                                                            heads);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The keyframe model's low-quality input: a frame (f32 [h][w][c], 0..1) as `lq_pixels` (x * 2 - 1, bf16), replicate-padded
// to [hp][wp], bilinearly upsampled `up` times (align_corners=False, bf16), then pixel-unshuffled by `r`:
// out [(hp up / r) (wp up / r)][3 r r] bf16, column c r r + i r + j.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float lq_px(const float* src, int h, int w, int c, int y, int x, int ch) {
    y = min(y, h - 1);
    x = min(x, w - 1);
    return rnd<bf16>(src[((int64_t)y * w + x) * c + ch] * 2.f - 1.f);
}

__global__ void relay_lq_unshuffle_kernel(const float* src, int h, int w, int c, int hp, int wp, int up, int r, bf16* out) {
    const int ow = wp * up / r, oh = hp * up / r;
    const int cols = 3 * r * r;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)oh * ow * cols) return;
    const int col = (int)(i % cols);
    const int64_t tok = i / cols;
    const int ty = (int)(tok / ow), tx = (int)(tok % ow);
    const int ch = col / (r * r), ii = (col / r) % r, jj = col % r;
    const int Y = ty * r + ii, X = tx * r + jj;
    float v;
    if (up == 1) {
        v = lq_px(src, h, w, c, Y, X, ch);
    } else {
        const float s = 1.f / (float)up;
        const float ry = fmaxf(s * (Y + 0.5f) - 0.5f, 0.f), rx = fmaxf(s * (X + 0.5f) - 0.5f, 0.f);
        const int y0 = (int)ry, x0 = (int)rx;
        const int y1 = y0 + (y0 < hp - 1 ? 1 : 0), x1 = x0 + (x0 < wp - 1 ? 1 : 0);
        const float ly = ry - y0, lx = rx - x0;
        const float a = lq_px(src, h, w, c, y0, x0, ch), b = lq_px(src, h, w, c, y0, x1, ch);
        const float cc = lq_px(src, h, w, c, y1, x0, ch), d = lq_px(src, h, w, c, y1, x1, ch);
        v = rnd<bf16>((1.f - ly) * ((1.f - lx) * a + lx * b) + ly * ((1.f - lx) * cc + lx * d));
    }
    out[i] = __float2bfloat16_rn(v);
}

GPU_API int gpu_relay_lq_unshuffle(const float* src, int h, int w, int c, int hp, int wp, int up, int r, void* out, void* stream) {
    const int64_t n = (int64_t)(hp * up / r) * (wp * up / r) * 3 * r * r;
    if (n == 0) return 0;
    relay_lq_unshuffle_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(src, h, w, c, hp, wp, up, r, (bf16*)out);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// 3x3 im2col, padding 1, of an NHWC map: out [oh ow][ld], column (ky 3 + kx) c + ch, zeros past 9 c.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void relay_im2col3_kernel(const T* in, int H, int W, int C, int stride, T* out, int ld, int oh, int ow) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)oh * ow * ld) return;
    const int col = (int)(i % ld);
    const int64_t pix = i / ld;
    const int y = (int)(pix / ow), x = (int)(pix % ow);
    float v = 0.f;
    if (col < 9 * C) {
        const int tap = col / C, ch = col % C;
        const int sy = y * stride - 1 + tap / 3, sx = x * stride - 1 + tap % 3;
        if (sy >= 0 && sy < H && sx >= 0 && sx < W) v = ldf(in, ((int64_t)sy * W + sx) * C + ch);
    }
    stf(out, i, v);
}

GPU_API int gpu_relay_im2col3(int dtype, const void* in, int H, int W, int C, int stride, void* out, int ld, void* stream) {
    const int oh = (H - 1) / stride + 1, ow = (W - 1) / stride + 1;
    const int64_t n = (int64_t)oh * ow * ld;
    if (n == 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_BF16) relay_im2col3_kernel<bf16><<<g, 256, 0, st>>>((const bf16*)in, H, W, C, stride, (bf16*)out, ld, oh, ow);
    else if (dtype == DT_F16) relay_im2col3_kernel<half><<<g, 256, 0, st>>>((const half*)in, H, W, C, stride, (half*)out, ld, oh, ow);
    else return -1;
    KCHECK();
    return 0;
}

// in-place SiLU, rounded to T
template <typename T>
__global__ void relay_silu_kernel(T* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(x, i, silu_f(ldf(x, i)));
}

GPU_API int gpu_relay_silu(int dtype, void* x, int64_t n, void* stream) {
    if (n == 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_BF16) relay_silu_kernel<bf16><<<g, 256, 0, st>>>((bf16*)x, n);
    else if (dtype == DT_F16) relay_silu_kernel<half><<<g, 256, 0, st>>>((half*)x, n);
    else return -1;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The keyframe model's patch embedding input: latent f32 [C][F][lh][lw], frames f0.. f0 + nf, as f32 rows of 2x2 patches
// [nf (lh/2) (lw/2)][4 C], column c 4 + i 2 + j (the Conv3d's weight order). The values are rounded to bf16 first: the
// model gets the latent in its dtype and the embedding upcasts it.
// ---------------------------------------------------------------------------------------------
__global__ void relay_patchify_kernel(const float* lat, int C, int F, int f0, int nf, int lh, int lw, float* out) {
    const int th = lh / 2, tw = lw / 2;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)nf * th * tw * 4 * C) return;
    const int col = (int)(i % (4 * C));
    const int64_t tok = i / (4 * C);
    const int f = (int)(tok / ((int64_t)th * tw)), y = (int)((tok / tw) % th), x = (int)(tok % tw);
    const int c = col / 4, pi = (col / 2) % 2, pj = col % 2;
    out[i] = rnd<bf16>(lat[(((int64_t)c * F + f0 + f) * lh + 2 * y + pi) * lw + 2 * x + pj]);
}

GPU_API int gpu_relay_patchify(const float* lat, int C, int F, int f0, int nf, int lh, int lw, float* out, void* stream) {
    const int64_t n = (int64_t)nf * (lh / 2) * (lw / 2) * 4 * C;
    if (n == 0) return 0;
    relay_patchify_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lat, C, F, f0, nf, lh, lw, out);
    KCHECK();
    return 0;
}

// out = bf16(lq + bf16(pe)): the projected low-quality frames plus the patch embedding, as `RelayVSRWanModel` sums them.
__global__ void relay_embed_add_kernel(const float* pe, const bf16* lq, bf16* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(__bfloat162float(lq[i]) + rnd<bf16>(pe[i]));
}

GPU_API int gpu_relay_embed_add(const float* pe, const void* lq, void* out, int64_t n, void* stream) {
    if (n == 0) return 0;
    relay_embed_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(pe, (const bf16*)lq, (bf16*)out, n);
    KCHECK();
    return 0;
}

// The head's patches (bf16 [nf (lh/2) (lw/2)][ld], column (i 2 + j) C + c) back to the latent grid, and the denoised
// latent x - sigma * out into den (f32 [C][F][lh][lw]) for frames f0.. f0 + nf.
__global__ void relay_unpatch_denoise_kernel(const bf16* tok, int ld, const float* x, float* den, int C, int F, int f0, int nf, int lh, int lw,
                                             float sigma) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)C * nf * lh * lw) return;
    const int xx = (int)(i % lw);
    const int y = (int)((i / lw) % lh);
    const int f = (int)((i / ((int64_t)lw * lh)) % nf);
    const int c = (int)(i / ((int64_t)lw * lh * nf));
    const int64_t t = ((int64_t)f * (lh / 2) + y / 2) * (lw / 2) + xx / 2;
    const float v = __bfloat162float(tok[t * ld + ((y % 2) * 2 + xx % 2) * C + c]);
    const int64_t o = (((int64_t)c * F + f0 + f) * lh + y) * lw + xx;
    den[o] = x[o] - v * sigma;
}

GPU_API int gpu_relay_unpatch_denoise(const void* tok, int ld, const float* x, float* den, int C, int F, int f0, int nf, int lh, int lw, float sigma,
                                      void* stream) {
    const int64_t n = (int64_t)C * nf * lh * lw;
    if (n == 0) return 0;
    relay_unpatch_denoise_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)tok, ld, x, den, C, F, f0, nf, lh, lw,
                                                                                                 sigma);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The Flash decoder's inputs: one latent plane (f32 [C][F][h][w], plane f) as T rows [h w][C]; and one low-quality frame
// (f32 [h][w][c], 0..1) as `lr` = x * 2 - 1, replicate-padded to [hp][wp]: f32 NHWC [hp][wp][3] and the same in T.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void relay_plane_rows_kernel(const float* lat, int C, int F, int f, int h, int w, T* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)h * w * C) return;
    const int c = (int)(i % C);
    const int64_t p = i / C;
    stf(out, i, lat[((int64_t)c * F + f) * h * w + p]);
}

GPU_API int gpu_relay_plane_rows(int dtype, const float* lat, int C, int F, int f, int h, int w, void* out, void* stream) {
    const int64_t n = (int64_t)h * w * C;
    if (n == 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_F16) relay_plane_rows_kernel<half><<<g, 256, 0, st>>>(lat, C, F, f, h, w, (half*)out);
    else if (dtype == DT_BF16) relay_plane_rows_kernel<bf16><<<g, 256, 0, st>>>(lat, C, F, f, h, w, (bf16*)out);
    else return -1;
    KCHECK();
    return 0;
}

template <typename T>
__global__ void relay_lr_frame_kernel(const float* src, int h, int w, int c, int hp, int wp, float* out_f, T* out_t) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)hp * wp * 3) return;
    const int ch = (int)(i % 3);
    const int64_t p = i / 3;
    const int y = min((int)(p / wp), h - 1), x = min((int)(p % wp), w - 1);
    const float v = src[((int64_t)y * w + x) * c + ch] * 2.f - 1.f;
    out_f[i] = v;
    stf(out_t, i, v);
}

GPU_API int gpu_relay_lr_frame(int dtype, const float* src, int h, int w, int c, int hp, int wp, float* out_f, void* out_t, void* stream) {
    const int64_t n = (int64_t)hp * wp * 3;
    if (n == 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_F16) relay_lr_frame_kernel<half><<<g, 256, 0, st>>>(src, h, w, c, hp, wp, out_f, (half*)out_t);
    else if (dtype == DT_BF16) relay_lr_frame_kernel<bf16><<<g, 256, 0, st>>>(src, h, w, c, hp, wp, out_f, (bf16*)out_t);
    else return -1;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The Flash decoder's output: bicubic upscale of the f32 frame (NHWC [hp][wp][3], align_corners=False, A = -0.75, taps
// clamped to the frame) plus the residual (T [h w][768], pixel-shuffled by 16 as channel c 256 + i 16 + j), the residual
// first scaled by `detail` in T unless it is 1; then (v + 1) / 2 clamped to 0..1, into f32 [oh][ow][3] (oh, ow at most
// the upscaled frame).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float cubic1(float x, float A) { return ((A + 2.f) * x - (A + 3.f)) * x * x + 1.f; }
__device__ __forceinline__ float cubic2(float x, float A) { return ((A * x - 5.f * A) * x + 8.f * A) * x - 4.f * A; }
__device__ __forceinline__ void cubic_coeffs(float t, float* c) {
    const float A = -0.75f;
    c[0] = cubic2(t + 1.f, A);
    c[1] = cubic1(t, A);
    c[2] = cubic1(1.f - t, A);
    c[3] = cubic2(2.f - t, A);
}

template <typename T>
__global__ void relay_flash_out_kernel(const T* res, const float* lr, int hp, int wp, int up, float detail, float* out, int oh, int ow, int lw) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)oh * ow * 3) return;
    const int ch = (int)(i % 3);
    const int64_t p = i / 3;
    const int Y = (int)(p / ow), X = (int)(p % ow);
    float r = ldf(res, ((int64_t)(Y / 16) * lw + X / 16) * 768 + ch * 256 + (Y % 16) * 16 + (X % 16));
    if (detail != 1.f) r = rnd<T>(r * detail);
    const float s = 1.f / (float)up;
    const float ry = s * (Y + 0.5f) - 0.5f, rx = s * (X + 0.5f) - 0.5f;
    const int y0 = (int)floorf(ry), x0 = (int)floorf(rx);
    float cy[4], cx[4];
    cubic_coeffs(ry - y0, cy);
    cubic_coeffs(rx - x0, cx);
    float rows[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int yy = min(max(y0 - 1 + a, 0), hp - 1);
        float v[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int xx = min(max(x0 - 1 + b, 0), wp - 1);
            v[b] = lr[((int64_t)yy * wp + xx) * 3 + ch];
        }
        rows[a] = v[0] * cx[0] + v[1] * cx[1] + v[2] * cx[2] + v[3] * cx[3];
    }
    const float upv = rows[0] * cy[0] + rows[1] * cy[1] + rows[2] * cy[2] + rows[3] * cy[3];
    out[i] = fminf(fmaxf((upv + r + 1.f) / 2.f, 0.f), 1.f);
}

GPU_API int gpu_relay_flash_out(int dtype, const void* res, const float* lr, int hp, int wp, int up, float detail, float* out, int oh, int ow, int lw,
                                void* stream) {
    const int64_t n = (int64_t)oh * ow * 3;
    if (n == 0) return 0;
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_F16) relay_flash_out_kernel<half><<<g, 256, 0, st>>>((const half*)res, lr, hp, wp, up, detail, out, oh, ow, lw);
    else if (dtype == DT_BF16) relay_flash_out_kernel<bf16><<<g, 256, 0, st>>>((const bf16*)res, lr, hp, wp, up, detail, out, oh, ow, lw);
    else return -1;
    KCHECK();
    return 0;
}
