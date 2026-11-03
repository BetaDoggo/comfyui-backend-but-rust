// Kernels for BiRefNet (background removal): a Swin-L backbone with window attention, a decoder of deformable
// convolutions, and the bilinear resizes between them. Everything is f32 and channel-last (NHWC), like ComfyUI
// runs it (its manual_cast ops compute in the input's dtype).
#include "common.cuh"

constexpr int WS = 12;           // window size
constexpr int WN = WS * WS;      // tokens per window
constexpr int HD = 32;           // head dim of every Swin stage

// ---------------------------------------------------------------------------------------------
// im2col of a zero-padded K x K convolution (stride 1) for output rows y0 .. y0 + rows, in (ky, kx, c) order.
// ---------------------------------------------------------------------------------------------
__global__ void im2col_zero_kernel(const float* in, int H, int W, int C, int K, int pad, int y0, int rows, float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)rows * W * K * K * C) return;
    int64_t r = t;
    const int c = (int)(r % C);
    r /= C;
    const int tap = (int)(r % (K * K));
    r /= K * K;
    const int x = (int)(r % W);
    const int yr = (int)(r / W);
    const int sy = y0 + yr + tap / K - pad, sx = x + tap % K - pad;
    out[t] = (sy >= 0 && sy < H && sx >= 0 && sx < W) ? in[((int64_t)sy * W + sx) * C + c] : 0.f;
}

GPU_API int gpu_im2col_zero(const float* in, int H, int W, int C, int K, int pad, int y0, int rows, float* out, void* stream) {
    const int64_t total = (int64_t)rows * W * K * K * C;
    im2col_zero_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, H, W, C, K, pad, y0, rows, out);
    KCHECK();
    return 0;
}

// Non-overlapping P x P patches of an [H, W, C] image as rows [(H / P) * (W / P), P * P * C], in (ky, kx, c) order.
__global__ void patchify_kernel(const float* in, int H, int W, int C, int P, float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int Wo = W / P;
    if (t >= (int64_t)(H / P) * Wo * P * P * C) return;
    int64_t r = t;
    const int c = (int)(r % C);
    r /= C;
    const int tap = (int)(r % (P * P));
    r /= P * P;
    const int px = (int)(r % Wo), py = (int)(r / Wo);
    out[t] = in[((int64_t)(py * P + tap / P) * W + px * P + tap % P) * C + c];
}

GPU_API int gpu_patchify_nhwc(const float* in, int H, int W, int C, int P, float* out, void* stream) {
    const int64_t total = (int64_t)H * W * C;
    patchify_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, H, W, C, P, out);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// torchvision's deform_conv2d (one offset group, stride 1): columns for output rows y0 .. y0 + rows, in (tap, c)
// order. `om` is [H * W, 3 * K * K]: per tap a (dy, dx) pair, then the K * K raw modulator logits (mask = 2 sigmoid).
// ---------------------------------------------------------------------------------------------
__global__ void deform_im2col_kernel(const float* in, const float* om, int H, int W, int C, int K, int pad, int y0, int rows,
                                     float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int KK = K * K;
    if (t >= (int64_t)rows * W * KK * C) return;
    int64_t r = t;
    const int c = (int)(r % C);
    r /= C;
    const int tap = (int)(r % KK);
    r /= KK;
    const int x = (int)(r % W);
    const int y = y0 + (int)(r / W);
    const float* o = om + ((int64_t)y * W + x) * 3 * KK;
    const float h = (float)(y - pad + tap / K) + o[2 * tap];
    const float w = (float)(x - pad + tap % K) + o[2 * tap + 1];
    const float m = 2.f * (1.f / (1.f + expf(-o[2 * KK + tap])));
    float val = 0.f;
    if (h > -1.f && w > -1.f && h < (float)H && w < (float)W) {
        const int hl = (int)floorf(h), wl = (int)floorf(w);
        const int hh = hl + 1, wh = wl + 1;
        const float lh = h - (float)hl, lw = w - (float)wl;
        const float hh_ = 1.f - lh, hw_ = 1.f - lw;
        float v1 = 0.f, v2 = 0.f, v3 = 0.f, v4 = 0.f;
        if (hl >= 0 && wl >= 0) v1 = in[((int64_t)hl * W + wl) * C + c];
        if (hl >= 0 && wh <= W - 1) v2 = in[((int64_t)hl * W + wh) * C + c];
        if (hh <= H - 1 && wl >= 0) v3 = in[((int64_t)hh * W + wl) * C + c];
        if (hh <= H - 1 && wh <= W - 1) v4 = in[((int64_t)hh * W + wh) * C + c];
        val = hh_ * hw_ * v1 + hh_ * lw * v2 + lh * hw_ * v3 + lh * lw * v4;
    }
    out[t] = m * val;
}

GPU_API int gpu_deform_im2col(const float* in, const float* om, int H, int W, int C, int K, int pad, int y0, int rows, float* out,
                              void* stream) {
    const int64_t total = (int64_t)rows * W * K * K * C;
    deform_im2col_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, om, H, W, C, K, pad, y0, rows, out);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Swin windows. x is the normed [H, W, C] map; the windows are rows [nW * 144, C], zero where the padding to
// (Hp, Wp) is, after rolling by -shift.
// ---------------------------------------------------------------------------------------------
__global__ void window_gather_kernel(const float* x, int H, int W, int C, int Hp, int Wp, int shift, float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int nwx = Wp / WS;
    if (t >= (int64_t)(Hp / WS) * nwx * WN * C) return;
    int64_t r = t;
    const int c = (int)(r % C);
    r /= C;
    const int tok = (int)(r % WN);
    const int win = (int)(r / WN);
    const int ys = (win / nwx) * WS + tok / WS, xs = (win % nwx) * WS + tok % WS;
    const int sy = (ys + shift) % Hp, sx = (xs + shift) % Wp;
    out[t] = (sy < H && sx < W) ? x[((int64_t)sy * W + sx) * C + c] : 0.f;
}

GPU_API int gpu_window_gather(const float* x, int H, int W, int C, int Hp, int Wp, int shift, float* out, void* stream) {
    const int64_t total = (int64_t)(Hp / WS) * (Wp / WS) * WN * C;
    window_gather_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, H, W, C, Hp, Wp, shift, out);
    KCHECK();
    return 0;
}

// x[y, x] += windows[...] after rolling back by +shift and dropping the padding.
__global__ void window_scatter_add_kernel(float* x, const float* win_rows, int H, int W, int C, int Hp, int Wp, int shift) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)H * W * C) return;
    const int c = (int)(t % C);
    const int64_t p = t / C;
    const int y = (int)(p / W), xx = (int)(p % W);
    const int ys = (y - shift + Hp) % Hp, xs = (xx - shift + Wp) % Wp;
    const int win = (ys / WS) * (Wp / WS) + xs / WS;
    const int tok = (ys % WS) * WS + xs % WS;
    x[t] += win_rows[((int64_t)win * WN + tok) * C + c];
}

GPU_API int gpu_window_scatter_add(float* x, const float* win_rows, int H, int W, int C, int Hp, int Wp, int shift, void* stream) {
    const int64_t total = (int64_t)H * W * C;
    window_scatter_add_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, win_rows, H, W, C, Hp, Wp, shift);
    KCHECK();
    return 0;
}

__device__ __forceinline__ int region(int c, int size, int shift) { return c < size - WS ? 0 : (c < size - shift ? 1 : 2); }

// Window attention with the relative position bias and, for shifted blocks, the -100 mask between regions.
// One block per (window, head), one thread per query; q is scaled before the product like torch. qkv: [nW * 144, 3C].
__global__ void window_attn_kernel(const float* qkv, const float* table, float* out, int C, int heads, int Hp, int Wp, int shift,
                                   float scale) {
    __shared__ float ks[WN * HD], vs[WN * HD];
    const int win = blockIdx.x, head = blockIdx.y, i = threadIdx.x;
    const int64_t row0 = (int64_t)win * WN;
    for (int e = threadIdx.x; e < WN * HD; e += blockDim.x) {
        const int j = e / HD, d = e % HD;
        const float* p = qkv + (row0 + j) * 3 * C + head * HD + d;
        ks[e] = p[C];
        vs[e] = p[2 * C];
    }
    __syncthreads();
    float q[HD];
    const float* qp = qkv + (row0 + i) * 3 * C + head * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) q[d] = qp[d] * scale;

    const int nwx = Wp / WS;
    const int yi = i / WS, xi = i % WS;
    int ri = 0;
    if (shift > 0) ri = 3 * region((win / nwx) * WS + yi, Hp, shift) + region((win % nwx) * WS + xi, Wp, shift);
    auto score = [&](int j) {
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < HD; ++d) s += q[d] * ks[j * HD + d];
        const int yj = j / WS, xj = j % WS;
        s += table[((yi - yj + WS - 1) * (2 * WS - 1) + (xi - xj + WS - 1)) * heads + head];
        if (shift > 0) {
            const int rj = 3 * region((win / nwx) * WS + yj, Hp, shift) + region((win % nwx) * WS + xj, Wp, shift);
            if (ri != rj) s += -100.f;
        }
        return s;
    };
    float mx = -3.4e38f;
    for (int j = 0; j < WN; ++j) mx = fmaxf(mx, score(j));
    float acc[HD];
#pragma unroll
    for (int d = 0; d < HD; ++d) acc[d] = 0.f;
    float sum = 0.f;
    for (int j = 0; j < WN; ++j) {
        const float p = expf(score(j) - mx);
        sum += p;
#pragma unroll
        for (int d = 0; d < HD; ++d) acc[d] += p * vs[j * HD + d];
    }
    const float inv = 1.f / sum;
    float* o = out + (row0 + i) * C + head * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) o[d] = acc[d] * inv;
}

GPU_API int gpu_window_attn(const float* qkv, const float* table, float* out, int nW, int C, int heads, int Hp, int Wp, int shift,
                            void* stream) {
    if (C != heads * HD) return -1;
    window_attn_kernel<<<dim3(nW, heads), WN, 0, (cudaStream_t)stream>>>(qkv, table, out, C, heads, Hp, Wp, shift, 1.f / sqrtf((float)HD));
    KCHECK();
    return 0;
}

// PatchMerging's gather: [H, W, C] -> [(H / 2) * (W / 2), 4C], the quadrants in the order (0,0), (1,0), (0,1), (1,1).
__global__ void patch_merge_kernel(const float* x, int H, int W, int C, float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int W2 = W / 2;
    if (t >= (int64_t)(H / 2) * W2 * 4 * C) return;
    int64_t r = t;
    const int c = (int)(r % C);
    r /= C;
    const int q = (int)(r % 4);
    r /= 4;
    const int x2 = (int)(r % W2), y2 = (int)(r / W2);
    const int sy = 2 * y2 + (q & 1), sx = 2 * x2 + (q >> 1);
    out[t] = x[((int64_t)sy * W + sx) * C + c];
}

GPU_API int gpu_patch_merge(const float* x, int H, int W, int C, float* out, void* stream) {
    const int64_t total = (int64_t)H * W * C;
    patch_merge_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, H, W, C, out);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Decoder pieces.
// ---------------------------------------------------------------------------------------------
// BiRefNet's get_patches_batch: the input image [Hf, Hf, C] cut into G x G patches of P x P (G = Hf / P), stacked as
// channels, patch (row r, column col) at (col * G + r) * C: out [P * P, G * G * C].
__global__ void split_patches_kernel(const float* x, int Hf, int C, int P, float* out) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int G = Hf / P;
    const int64_t nc = (int64_t)G * G * C;
    if (t >= (int64_t)P * P * nc) return;
    const int64_t pix = t / nc;
    const int ch = (int)(t % nc);
    const int c = ch % C, pi = ch / C;
    const int r = pi % G, col = pi / G;
    const int py = (int)(pix / P), px = (int)(pix % P);
    out[t] = x[((int64_t)(r * P + py) * Hf + col * P + px) * C + c];
}

GPU_API int gpu_split_patches(const float* x, int Hf, int C, int P, float* out, void* stream) {
    const int64_t total = (int64_t)Hf * Hf * C;
    split_patches_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, Hf, C, P, out);
    KCHECK();
    return 0;
}

// F.interpolate(mode="bilinear", align_corners=True) on NHWC f32, with torch's operation order.
__global__ void resize_ac_kernel(const float* in, int H, int W, int C, float* out, int64_t ldo, int OH, int OW) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)OH * OW * C) return;
    int64_t r = t;
    const int c = (int)(r % C);
    r /= C;
    const int ox = (int)(r % OW), oy = (int)(r / OW);
    const float sh = OH > 1 ? (float)(H - 1) / (float)(OH - 1) : 0.f;
    const float sw = OW > 1 ? (float)(W - 1) / (float)(OW - 1) : 0.f;
    const float fy = sh * (float)oy, fx = sw * (float)ox;
    const int y0 = (int)fy, x0 = (int)fx;
    const int y1 = y0 + (y0 < H - 1 ? 1 : 0), x1 = x0 + (x0 < W - 1 ? 1 : 0);
    const float ly = fy - (float)y0, lx = fx - (float)x0;
    const float hy = 1.f - ly, hx = 1.f - lx;
    auto at = [&](int y, int x) { return in[((int64_t)y * W + x) * C + c]; };
    out[((int64_t)oy * OW + ox) * ldo + c] = hy * (hx * at(y0, x0) + lx * at(y0, x1)) + ly * (hx * at(y1, x0) + lx * at(y1, x1));
}

// `out` has a row stride of `ldo` floats, so a result can be written into the channels of a concatenation.
GPU_API int gpu_resize_bilinear_ac(const float* in, int H, int W, int C, float* out, int64_t ldo, int OH, int OW, void* stream) {
    const int64_t total = (int64_t)OH * OW * C;
    resize_ac_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(in, H, W, C, out, ldo, OH, OW);
    KCHECK();
    return 0;
}

// x[rows, 0 .. c) (row stride ld) = act(x): mode 0 ReLU, 1 sigmoid.
__global__ void act_kernel(float* x, int64_t ld, int c, int64_t rows, int mode) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * c) return;
    float* p = x + (t / c) * ld + t % c;
    *p = mode == 0 ? fmaxf(*p, 0.f) : 1.f / (1.f + expf(-*p));
}

GPU_API int gpu_act_rows(float* x, int64_t ld, int c, int64_t rows, int mode, void* stream) {
    const int64_t total = rows * c;
    act_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, ld, c, rows, mode);
    KCHECK();
    return 0;
}

// x[r, :] *= s[r]
__global__ void mul_rows_kernel(float* x, const float* s, int c, int64_t rows) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * c) return;
    x[t] *= s[t / c];
}

GPU_API int gpu_mul_rows(float* x, const float* s, int c, int64_t rows, void* stream) {
    const int64_t total = rows * c;
    mul_rows_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, s, c, rows);
    KCHECK();
    return 0;
}

__global__ void add_kernel(float* x, const float* y, int64_t n) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) x[t] += y[t];
}

GPU_API int gpu_add_f32(float* x, const float* y, int64_t n, void* stream) {
    add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, y, n);
    KCHECK();
    return 0;
}

// out[r, 0 .. c) = v[0 .. c) for every row (row stride ld): the global-pool branch upsampled from 1 x 1.
__global__ void bcast_rows_kernel(const float* v, int c, float* out, int64_t ld, int64_t rows) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < rows * c) out[(t / c) * ld + t % c] = v[t % c];
}

GPU_API int gpu_bcast_rows(const float* v, int c, float* out, int64_t ld, int64_t rows, void* stream) {
    const int64_t total = rows * c;
    bcast_rows_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(v, c, out, ld, rows);
    KCHECK();
    return 0;
}
