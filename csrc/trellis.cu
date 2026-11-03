// Kernels for Trellis 2 / Pixal3D: the sparse flow transformer, the sparse convolution VAEs, the NAF feature
// upsampler and the image-to-voxel feature projection.
//
// Rounding follows ComfyUI's tensors as for the other models: rnd<T> where it materializes a T tensor.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Per-head RMSNorm with a weight per head (MultiHeadRMSNorm, gamma [heads, 128]) and interleaved RoPE.
// F.rms_norm rounds to the input dtype, then the gamma multiply rounds again; eps is float's epsilon.
// rope: [L][64][(cos, sin)] or null. One warp per (token, head).
// ---------------------------------------------------------------------------------------------
__global__ void qk_rms_heads_kernel(bf16* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk, const float* qw,
                                    const float* kw, float eps, const float* rope, int L) {
    constexpr int D = 128, EPL = 4;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int heads = hq + hk;
    if (warp >= L * heads) return;
    const int t = warp / heads, h = warp % heads;
    const bool isq = h < hq;
    const int hh = isq ? h : h - hq;
    bf16* x = buf + (int64_t)t * row_stride + (isq ? q_off : k_off) + hh * D;
    const float* w = (isq ? qw : kw) + hh * D;
    float v[EPL];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        v[i] = __bfloat162float(x[lane * EPL + i]);
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / (float)D + eps);
#pragma unroll
    for (int i = 0; i < EPL; ++i) v[i] = rnd<bf16>(rnd<bf16>(v[i] * rstd) * w[lane * EPL + i]);
    if (rope) {
        const float* r = rope + (int64_t)t * D;
#pragma unroll
        for (int pi = 0; pi < EPL / 2; ++pi) {
            const int j = lane * (EPL / 2) + pi;
            const float c = r[2 * j], s = r[2 * j + 1];
            const float a = v[2 * pi], b = v[2 * pi + 1];
            v[2 * pi] = c * a - s * b;
            v[2 * pi + 1] = s * a + c * b;
        }
    }
#pragma unroll
    for (int i = 0; i < EPL; ++i) x[lane * EPL + i] = __float2bfloat16_rn(v[i]);
}

GPU_API int gpu_qk_rms_heads(void* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk, const float* qw,
                               const float* kw, float eps, const float* rope, int L, void* stream) {
    const int64_t warps = (int64_t)L * (hq + hk);
    const int threads = 256;
    qk_rms_heads_kernel<<<(unsigned)((warps * 32 + threads - 1) / threads), threads, 0, (cudaStream_t)stream>>>(
        (bf16*)buf, row_stride, q_off, k_off, hq, hk, qw, kw, eps, rope, L);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Sparse voxel coordinates: an open-addressing hash from (b, x, y, z) to the row index, and the 27
// submanifold neighbours of each voxel in flexgemm's order (x offset outermost, z innermost).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long voxel_key(int b, int x, int y, int z, int X, int Y, int Z) {
    return (((unsigned long long)b * X + x) * Y + y) * Z + z;
}

__global__ void hash_insert_kernel(const int* coords, int n, int X, int Y, int Z, unsigned long long* keys, int* vals,
                                   unsigned cap_mask) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int* c = coords + 4 * (int64_t)i;
    const unsigned long long key = voxel_key(c[0], c[1], c[2], c[3], X, Y, Z);
    unsigned slot = (unsigned)mix64(key) & cap_mask;
    while (true) {
        const unsigned long long prev = atomicCAS(&keys[slot], ~0ull, key);
        if (prev == ~0ull || prev == key) {
            vals[slot] = i;
            return;
        }
        slot = (slot + 1) & cap_mask;
    }
}

__device__ __forceinline__ int hash_find(const unsigned long long* keys, const int* vals, unsigned cap_mask,
                                         unsigned long long key) {
    unsigned slot = (unsigned)mix64(key) & cap_mask;
    while (true) {
        const unsigned long long k = keys[slot];
        if (k == key) return vals[slot];
        if (k == ~0ull) return -1;
        slot = (slot + 1) & cap_mask;
    }
}

GPU_API int gpu_voxel_hash(const int* coords, int n, int X, int Y, int Z, unsigned long long* keys, int* vals, int cap,
                             void* stream) {
    if (cap & (cap - 1)) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    if (cudaMemsetAsync(keys, 0xff, (size_t)cap * sizeof(unsigned long long), st) != cudaSuccess) return -2;
    if (n > 0) hash_insert_kernel<<<(n + 255) / 256, 256, 0, st>>>(coords, n, X, Y, Z, keys, vals, (unsigned)cap - 1);
    KCHECK();
    return 0;
}

__global__ void voxel_neighbors_kernel(const int* coords, int n, int X, int Y, int Z, const unsigned long long* keys,
                                       const int* vals, unsigned cap_mask, int* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)n * 27) return;
    const int i = (int)(t / 27), v = (int)(t % 27);
    const int* c = coords + 4 * (int64_t)i;
    const int x = c[1] + v / 9 - 1, y = c[2] + (v / 3) % 3 - 1, z = c[3] + v % 3 - 1;
    int r = -1;
    if (x >= 0 && x < X && y >= 0 && y < Y && z >= 0 && z < Z) r = hash_find(keys, vals, cap_mask, voxel_key(c[0], x, y, z, X, Y, Z));
    out[t] = r;
}

GPU_API int gpu_voxel_neighbors(const int* coords, int n, int X, int Y, int Z, const unsigned long long* keys, const int* vals,
                                  int cap, int* out, void* stream) {
    const int64_t total = (int64_t)n * 27;
    if (total > 0)
        voxel_neighbors_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(coords, n, X, Y, Z, keys, vals,
                                                                                                  (unsigned)cap - 1, out);
    KCHECK();
    return 0;
}

// Look up arbitrary voxels: out[i] = row of query i, or -1.
__global__ void voxel_lookup_kernel(const int* q, int n, int X, int Y, int Z, const unsigned long long* keys, const int* vals,
                                    unsigned cap_mask, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int* c = q + 4 * (int64_t)i;
    int r = -1;
    if (c[1] >= 0 && c[1] < X && c[2] >= 0 && c[2] < Y && c[3] >= 0 && c[3] < Z)
        r = hash_find(keys, vals, cap_mask, voxel_key(c[0], c[1], c[2], c[3], X, Y, Z));
    out[i] = r;
}

GPU_API int gpu_voxel_lookup(const int* q, int n, int X, int Y, int Z, const unsigned long long* keys, const int* vals, int cap,
                               int* out, void* stream) {
    if (n > 0) voxel_lookup_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(q, n, X, Y, Z, keys, vals, (unsigned)cap - 1, out);
    KCHECK();
    return 0;
}

// im2col of a submanifold conv: out[r, v * C + c] = x[nbr[r0 + r][v], c], or 0 for a missing neighbour.
// 2-byte elements, C a multiple of 8.
__global__ void sparse_gather_kernel(const uint4* x, int c8, const int* nbr, int r0, int rows, uint4* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per_row = 27 * (int64_t)c8;
    if (t >= rows * per_row) return;
    const int64_t r = t / per_row;
    const int rest = (int)(t % per_row);
    const int v = rest / c8, c = rest % c8;
    const int j = nbr[(r0 + r) * 27 + v];
    out[t] = j >= 0 ? x[(int64_t)j * c8 + c] : make_uint4(0, 0, 0, 0);
}

GPU_API int gpu_sparse_gather(const void* x, int C, const int* nbr, int r0, int rows, void* out, void* stream) {
    if (C % 8) return -1;
    const int64_t total = (int64_t)rows * 27 * (C / 8);
    if (total > 0)
        sparse_gather_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint4*)x, C / 8, nbr, r0,
                                                                                                rows, (uint4*)out);
    KCHECK();
    return 0;
}

// out[i] = x[idx[i]] for rows of `bytes` bytes (a multiple of 2).
__global__ void gather_rows_kernel(const uint16_t* x, int64_t w, const int* idx, int n, uint16_t* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * w) return;
    const int64_t i = t / w, c = t % w;
    out[t] = x[(int64_t)idx[i] * w + c];
}

GPU_API int gpu_gather_rows(const void* x, int64_t bytes, const int* idx, int n, void* out, void* stream) {
    if (bytes % 2) return -1;
    const int64_t total = (int64_t)n * (bytes / 2);
    if (total > 0)
        gather_rows_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint16_t*)x, bytes / 2, idx,
                                                                                              n, (uint16_t*)out);
    KCHECK();
    return 0;
}

// The C2S skip: h[m, c * rep + r] += skip[m, c], in f16.
__global__ void skip_add_kernel(half* h, const half* skip, int64_t m, int cs, int rep) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= m * cs * rep) return;
    const int64_t row = t / ((int64_t)cs * rep);
    const int c = (int)((t / rep) % cs);
    h[t] = __float2half_rn(__half2float(h[t]) + __half2float(skip[row * cs + c]));
}

GPU_API int gpu_skip_add_f16(void* h, const void* skip, int64_t m, int cs, int rep, void* stream) {
    const int64_t total = m * cs * rep;
    if (total > 0)
        skip_add_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((half*)h, (const half*)skip, m, cs, rep);
    KCHECK();
    return 0;
}

// x = x + y, f16, both [n].
__global__ void add_f16_kernel(half* x, const half* y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half_rn(__half2float(x[i]) + __half2float(y[i]));
}

GPU_API int gpu_add_f16(void* x, const void* y, int64_t n, void* stream) {
    if (n > 0) add_f16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((half*)x, (const half*)y, n);
    KCHECK();
    return 0;
}

// pixel_shuffle_3d on a dense grid of voxel rows (x-major): in [R^3, C * 8], out [(2R)^3, C],
// out voxel (2x + i, 2y + j, 2z + k) channel c = in voxel (x, y, z) channel c * 8 + i * 4 + j * 2 + k.
__global__ void pixel_shuffle3d_kernel(const half* in, half* out, int R, int C) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int R2 = 2 * R;
    if (t >= (int64_t)R2 * R2 * R2 * C) return;
    const int c = (int)(t % C);
    const int64_t vox = t / C;
    const int oz = (int)(vox % R2), oy = (int)((vox / R2) % R2), ox = (int)(vox / ((int64_t)R2 * R2));
    const int64_t src = (((int64_t)(ox >> 1) * R + (oy >> 1)) * R + (oz >> 1)) * (C * 8) + c * 8 + (ox & 1) * 4 + (oy & 1) * 2 + (oz & 1);
    out[t] = in[src];
}

GPU_API int gpu_pixel_shuffle3d_f16(const void* in, void* out, int R, int C, void* stream) {
    const int64_t total = (int64_t)8 * R * R * R * C;
    pixel_shuffle3d_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)in, (half*)out, R, C);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Dtype casts and small f32 helpers for DINOv3.
// ---------------------------------------------------------------------------------------------
template <typename A, typename B>
__global__ void cast_kernel(const A* in, B* out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(out, i, ldf(in, i));
}

GPU_API int gpu_cast(int in_dt, int out_dt, const void* in, void* out, int64_t n, void* stream) {
    const unsigned g = (unsigned)((n + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (n <= 0) return 0;
#define CAST(A, B) cast_kernel<A, B><<<g, 256, 0, st>>>((const A*)in, (B*)out, n)
    if (in_dt == DT_F32 && out_dt == DT_F16) CAST(float, half);
    else if (in_dt == DT_F16 && out_dt == DT_F32) CAST(half, float);
    else if (in_dt == DT_F32 && out_dt == DT_BF16) CAST(float, bf16);
    else if (in_dt == DT_BF16 && out_dt == DT_F32) CAST(bf16, float);
    else if (in_dt == DT_F16 && out_dt == DT_BF16) CAST(half, bf16);
    else if (in_dt == DT_BF16 && out_dt == DT_F16) CAST(bf16, half);
    else return -4;
#undef CAST
    KCHECK();
    return 0;
}

// block_max for values of any sign (common.cuh's pads unused lanes with 0, which suits absolute maxima only).
__device__ __forceinline__ float block_max_signed(float v, float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < nw ? red[lane] : -INFINITY;
        t = warp_max(t);
        if (lane == 0) red[32] = t;
    }
    __syncthreads();
    return red[32];
}

// Row softmax in place, f32: x[r] = softmax(x[r] * scale).
__global__ void softmax_f32_kernel(float* x, int n, float scale) {
    __shared__ float red[33];
    float* row = x + (int64_t)blockIdx.x * n;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, row[j] * scale);
    mx = block_max_signed(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = expf(row[j] * scale - mx);
        row[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] *= inv;
}

GPU_API int gpu_softmax_f32(float* x, int rows, int n, float scale, void* stream) {
    softmax_f32_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(x, n, scale);
    KCHECK();
    return 0;
}

// x[r, c] += s[c] * y[r, c], f32 (LayerScale then the residual add).
__global__ void scaled_add_f32_kernel(float* x, const float* y, const float* s, int64_t n, int cols) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i] * s[i % cols];
}

GPU_API int gpu_scaled_add_f32(float* x, const float* y, const float* s, int64_t rows, int cols, void* stream) {
    const int64_t n = rows * cols;
    scaled_add_f32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, y, s, n, cols);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// NAF: f32 NHWC image encoder helpers, then the neighbourhood cross-attention.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ int reflect(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

// out[(y - y0) * W + x, (ky * 3 + kx) * C + c] = in[reflect(y + ky - 1), reflect(x + kx - 1), c]
__global__ void im2col3_reflect_kernel(const float* in, int H, int W, int C, int y0, int rows, float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = 9 * (int64_t)C;
    if (t >= (int64_t)rows * W * per) return;
    const int64_t p = t / per;
    const int rest = (int)(t % per);
    const int k = rest / C, c = rest % C;
    const int y = y0 + (int)(p / W), x = (int)(p % W);
    const int sy = reflect(y + k / 3 - 1, H), sx = reflect(x + k % 3 - 1, W);
    out[t] = in[((int64_t)sy * W + sx) * C + c];
}

GPU_API int gpu_im2col3_reflect(const float* in, int H, int W, int C, int y0, int rows, float* out, void* stream) {
    const int64_t total = (int64_t)rows * W * 9 * C;
    im2col3_reflect_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, H, W, C, y0, rows, out);
    KCHECK();
    return 0;
}

// GroupNorm over NHWC f32 (one image), then optional SiLU. stats: float2 [groups].
__global__ void gn_mean_kernel(const float* x, int64_t P, int C, int G, double* acc) {
    const int g = blockIdx.y, cpg = C / G;
    double s = 0.0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < P * cpg; i += (int64_t)gridDim.x * blockDim.x)
        s += x[(i / cpg) * C + g * cpg + i % cpg];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&acc[g], s);
}

__global__ void gn_var_kernel(const float* x, int64_t P, int C, int G, const double* mean_acc, double* acc) {
    const int g = blockIdx.y, cpg = C / G;
    const double mean = mean_acc[g] / ((double)P * cpg);
    double s = 0.0;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < P * cpg; i += (int64_t)gridDim.x * blockDim.x) {
        const double d = x[(i / cpg) * C + g * cpg + i % cpg] - mean;
        s += d * d;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(&acc[G + g], s);
}

// The NAF runs in f16 in ComfyUI: with `silu`, the norm and the activation are each rounded to f16.
__global__ void gn_apply_kernel(float* x, int64_t P, int C, int G, const double* acc, const float* w, const float* b, float eps,
                                int silu) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * C) return;
    const int c = (int)(i % C), g = c / (C / G);
    const double n = (double)P * (C / G);
    const float mean = (float)(acc[g] / n), var = (float)(acc[G + g] / n);
    float v = rnd<half>((x[i] - mean) * rsqrtf(var + eps) * w[c] + b[c]);
    x[i] = silu ? rnd<half>(silu_f(v)) : v;
}

// x = f16(x) in place: where the f16 reference materializes a tensor.
__global__ void round_f16_kernel(float* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = rnd<half>(x[i]);
}

GPU_API int gpu_round_f16(float* x, int64_t n, void* stream) {
    if (n > 0) round_f16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, n);
    KCHECK();
    return 0;
}

GPU_API int gpu_group_norm_nhwc_f32(float* x, int64_t P, int C, int G, const float* w, const float* b, float eps, int silu,
                                      double* scratch, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (cudaMemsetAsync(scratch, 0, 2 * G * sizeof(double), st) != cudaSuccess) return -2;
    const dim3 grid(256, G);
    gn_mean_kernel<<<grid, 256, 0, st>>>(x, P, C, G, scratch);
    gn_var_kernel<<<grid, 256, 0, st>>>(x, P, C, G, scratch, scratch);
    gn_apply_kernel<<<(unsigned)((P * C + 255) / 256), 256, 0, st>>>(x, P, C, G, scratch, w, b, eps, silu);
    KCHECK();
    return 0;
}

// adaptive_avg_pool2d by an integer factor, NHWC f32: out [H/f, W/f, C].
__global__ void avg_pool_kernel(const float* in, int H, int W, int C, int f, float* out) {
    const int Ho = H / f, Wo = W / f;
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)Ho * Wo * C) return;
    const int c = (int)(t % C);
    const int64_t p = t / C;
    const int ox = (int)(p % Wo), oy = (int)(p / Wo);
    float s = 0.f;
    for (int dy = 0; dy < f; ++dy)
        for (int dx = 0; dx < f; ++dx) s += in[((int64_t)(oy * f + dy) * W + ox * f + dx) * C + c];
    out[t] = s / (float)(f * f);
}

GPU_API int gpu_avg_pool_nhwc_f32(const float* in, int H, int W, int C, int f, float* out, void* stream) {
    const int64_t total = (int64_t)(H / f) * (W / f) * C;
    avg_pool_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, H, W, C, f, out);
    KCHECK();
    return 0;
}

// NAF's 2D RoPE on NHWC f32 (4 heads of 64): angle j < 32 is 2 pi * coord[j / 16] / periods[j % 16], with coords the
// pixel centres in [-1, 1] (h first); each head's first half pairs with its second.
__global__ void naf_rope_kernel(float* x, int H, int W, int C, const float* periods) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = C / 2;
    if (t >= (int64_t)H * W * pairs) return;
    const int64_t p = t / pairs;
    const int q = (int)(t % pairs), head = q / 32, j = q % 32;
    const int py = (int)(p / W), px = (int)(p % W);
    const float coord = j < 16 ? ((float)py + 0.5f) / (float)H * 2.f - 1.f : ((float)px + 0.5f) / (float)W * 2.f - 1.f;
    const float ang = 2.f * 3.14159265358979323846f * coord / periods[j % 16];
    float* row = x + p * C + head * 64;
    const float c = cosf(ang), s = sinf(ang);
    const float a = row[j], b = row[32 + j];
    row[j] = a * c - b * s;
    row[32 + j] = b * c + a * s;
}

GPU_API int gpu_naf_rope(float* x, int H, int W, int C, const float* periods, void* stream) {
    const int64_t total = (int64_t)H * W * (C / 2);
    naf_rope_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, H, W, C, periods);
    KCHECK();
    return 0;
}

// Neighbourhood attention of NAF (na2d_pure): every HR query of one LR cell attends to the same 9x9 window of the
// zero-padded LR grid. q [H*W, 4*64] f16, k [h*w, 4*64] f16, v [h*w, 4*dv] f16, out [H*W, 4*dv] f16.
// One block per (LR cell, head), one thread per HR query of the cell. Scores and softmax in f32, probabilities
// rounded to f16 before the f16 matmul with v, like the reference.
constexpr int NA_K = 9, NA_KK = NA_K * NA_K, NA_DQK = 64;

__global__ void naf_attn_kernel(const half* q, const half* k, const half* v, half* out, int H, int W, int h, int w, int dv) {
    extern __shared__ unsigned char sm_raw[];
    float* ks = reinterpret_cast<float*>(sm_raw);                 // [81][64]
    half* vs = reinterpret_cast<half*>(ks + NA_KK * NA_DQK);       // [81][dv]
    const int cell = blockIdx.x, head = blockIdx.y;
    const int cr = cell / w, cc = cell % w;
    const int Dh = H / h, Dw = W / w;
    const int nh = 4;
    for (int i = threadIdx.x; i < NA_KK * NA_DQK; i += blockDim.x) {
        const int j = i / NA_DQK, d = i % NA_DQK;
        const int r = cr + j / NA_K - NA_K / 2, c = cc + j % NA_K - NA_K / 2;
        ks[i] = (r >= 0 && r < h && c >= 0 && c < w) ? __half2float(k[((int64_t)r * w + c) * (nh * NA_DQK) + head * NA_DQK + d]) : 0.f;
    }
    for (int i = threadIdx.x; i < NA_KK * dv; i += blockDim.x) {
        const int j = i / dv, d = i % dv;
        const int r = cr + j / NA_K - NA_K / 2, c = cc + j % NA_K - NA_K / 2;
        vs[i] = (r >= 0 && r < h && c >= 0 && c < w) ? v[((int64_t)r * w + c) * (nh * dv) + head * dv + d] : __float2half(0.f);
    }
    __syncthreads();
    if (threadIdx.x >= Dh * Dw) return;
    const int y = cr * Dh + threadIdx.x / Dw, x = cc * Dw + threadIdx.x % Dw;
    const int64_t pix = (int64_t)y * W + x;
    float qv[NA_DQK];
#pragma unroll
    for (int d = 0; d < NA_DQK; ++d) qv[d] = __half2float(q[pix * (nh * NA_DQK) + head * NA_DQK + d]);
    float p[NA_KK];
    float mx = -INFINITY;
    for (int j = 0; j < NA_KK; ++j) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < NA_DQK; ++d) s += qv[d] * ks[j * NA_DQK + d];
        p[j] = s * 0.125f;
        mx = fmaxf(mx, p[j]);
    }
    float sum = 0.f;
    for (int j = 0; j < NA_KK; ++j) {
        p[j] = expf(p[j] - mx);
        sum += p[j];
    }
    for (int j = 0; j < NA_KK; ++j) p[j] = rnd<half>(p[j] / sum);
    half* o = out + pix * (nh * dv) + head * dv;
    for (int d = 0; d < dv; ++d) {
        float acc = 0.f;
        for (int j = 0; j < NA_KK; ++j) acc += p[j] * __half2float(vs[j * dv + d]);
        o[d] = __float2half_rn(acc);
    }
}

GPU_API int gpu_naf_attn(const void* q, const void* k, const void* v, void* out, int H, int W, int h, int w, int dv, void* stream) {
    const int threads = (H / h) * (W / w);
    if (threads > 1024 || H % h || W % w) return -1;
    const size_t smem = NA_KK * NA_DQK * sizeof(float) + (size_t)NA_KK * dv * sizeof(half);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(naf_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        configured = true;
    }
    naf_attn_kernel<<<dim3(h * w, 4), ((threads + 31) / 32) * 32, smem, (cudaStream_t)stream>>>(
        (const half*)q, (const half*)k, (const half*)v, (half*)out, H, W, h, w, dv);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Pixal3D back-projection: F.grid_sample(map, grid, bilinear, padding 'border', align_corners=False) of an NHWC
// feature map at N points, each rounded to the map's dtype and then to bf16 (the model's dtype), written to
// out[n * out_stride + col0 + c]. uv: [N][2] (x, y) in normalized device coordinates.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void grid_sample_kernel(const T* map, int H, int W, int C, const float* uv, int n, bf16* out, int64_t out_stride,
                                   int col0) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)n * C) return;
    const int64_t i = t / C;
    const int c = (int)(t % C);
    float ix = ((uv[2 * i] + 1.f) * W - 1.f) / 2.f;
    float iy = ((uv[2 * i + 1] + 1.f) * H - 1.f) / 2.f;
    ix = fminf((float)(W - 1), fmaxf(ix, 0.f));
    iy = fminf((float)(H - 1), fmaxf(iy, 0.f));
    const int x0 = (int)floorf(ix), y0 = (int)floorf(iy), x1 = x0 + 1, y1 = y0 + 1;
    const float nw = (x1 - ix) * (y1 - iy), ne = (ix - x0) * (y1 - iy), sw = (x1 - ix) * (iy - y0), se = (ix - x0) * (iy - y0);
    float acc = 0.f;
    auto at = [&](int y, int x) { return ldf(map, ((int64_t)y * W + x) * C + c); };
    acc += at(y0, x0) * nw;
    if (x1 < W) acc += at(y0, x1) * ne;
    if (y1 < H) acc += at(y1, x0) * sw;
    if (x1 < W && y1 < H) acc += at(y1, x1) * se;
    out[i * out_stride + col0 + c] = __float2bfloat16_rn(rnd<T>(acc));
}

GPU_API int gpu_grid_sample_bf16(int map_dt, const void* map, int H, int W, int C, const float* uv, int n, void* out,
                                   int64_t out_stride, int col0, void* stream) {
    const int64_t total = (int64_t)n * C;
    if (total == 0) return 0;
    const unsigned g = (unsigned)((total + 255) / 256);
    cudaStream_t st = (cudaStream_t)stream;
    if (map_dt == DT_F32) grid_sample_kernel<float><<<g, 256, 0, st>>>((const float*)map, H, W, C, uv, n, (bf16*)out, out_stride, col0);
    else if (map_dt == DT_F16) grid_sample_kernel<half><<<g, 256, 0, st>>>((const half*)map, H, W, C, uv, n, (bf16*)out, out_stride, col0);
    else return -4;
    KCHECK();
    return 0;
}
