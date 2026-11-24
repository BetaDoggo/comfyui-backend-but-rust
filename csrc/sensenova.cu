// Kernels for SenseNova U1.6: the split q/k norms and three-part RoPE of its attention, the 2D interleaved RoPE of its
// patch embedding, the reshuffles around its two embedding convs and its pixel head, and the denoised image the head's
// prediction makes. Rounding follows ComfyUI's (comfy/ldm/sensenova/model.py), which runs the generation tower in bf16.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Attention._project after the projections: each head of q and k is two halves of 64, each RMS-normed with its own
// weight (q_norm / q_norm_hw, k_norm / k_norm_hw), then the first half takes a split-half RoPE over its 64 values and
// the second's two quarters one each over 32. `rope` is [L][64][2] (cos, sin) per row: 32 pairs for the first half,
// 16 for the first quarter of the second, 16 for its other quarter. One warp per (row, head), four values a lane: the
// lane's own in each quarter of the head.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void sn_qk_kernel(T* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk, const float* qn,
                             const float* qn_hw, const float* kn, const float* kn_hw, float eps, const float* rope, int L) {
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (int64_t)L * (hq + hk)) return;
    const int row = (int)(warp / (hq + hk)), head = (int)(warp % (hq + hk));
    const bool is_q = head < hq;
    T* p = buf + row * row_stride + (is_q ? q_off + head * 128 : k_off + (head - hq) * 128);
    const float* w_t = is_q ? qn : kn;
    const float* w_hw = is_q ? qn_hw : kn_hw;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = ldf(p, lane + 32 * j);
    // the two RMS norms, each over 64 values
    const float inv_t = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1]) / 64.f + eps);
    const float inv_hw = rsqrtf(warp_sum(x[2] * x[2] + x[3] * x[3]) / 64.f + eps);
    x[0] = rnd<T>(rnd<T>(x[0] * inv_t) * w_t[lane]);
    x[1] = rnd<T>(rnd<T>(x[1] * inv_t) * w_t[lane + 32]);
    x[2] = rnd<T>(rnd<T>(x[2] * inv_hw) * w_hw[lane]);
    x[3] = rnd<T>(rnd<T>(x[3] * inv_hw) * w_hw[lane + 32]);
    const float* r = rope + (int64_t)row * 128;
    // x * cos + rotate_half(x) * sin, each product rounded, as in eager PyTorch
    auto rot = [](float v, float other, float c, float s, bool second) {
        const float rh = second ? other : -other;
        return rnd<T>(rnd<T>(v * c) + rnd<T>(rh * s));
    };
    // first half: pair (lane, lane + 32)
    {
        const float c = rnd<T>(r[lane * 2]), s = rnd<T>(r[lane * 2 + 1]);
        const float a = x[0], b = x[1];
        x[0] = rot(a, b, c, s, false);
        x[1] = rot(b, a, c, s, true);
    }
    // second half: quarters of 32, pair (lane, lane ^ 16) within each
    const int i = lane & 15;
    const bool second = lane >= 16;
#pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        const float other = __shfl_xor_sync(0xffffffff, x[2 + qq], 16);
        const float c = rnd<T>(r[(32 + qq * 16 + i) * 2]), s = rnd<T>(r[(32 + qq * 16 + i) * 2 + 1]);
        x[2 + qq] = rot(x[2 + qq], other, c, s, second);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) stf(p, lane + 32 * j, x[j]);
}

GPU_API int gpu_sn_qk(int dtype, void* buf, int64_t row_stride, int q_off, int k_off, int hq, int hk, const float* qn,
                      const float* qn_hw, const float* kn, const float* kn_hw, float eps, const float* rope, int L, void* stream) {
    const int64_t warps = (int64_t)L * (hq + hk);
    if (warps == 0) return 0;
    const int threads = 256;
    const unsigned blocks = (unsigned)((warps * 32 + threads - 1) / threads);
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_BF16) sn_qk_kernel<bf16><<<blocks, threads, 0, st>>>((bf16*)buf, row_stride, q_off, k_off, hq, hk, qn, qn_hw, kn, kn_hw, eps, rope, L);
    else if (dtype == DT_F32) sn_qk_kernel<float><<<blocks, threads, 0, st>>>((float*)buf, row_stride, q_off, k_off, hq, hk, qn, qn_hw, kn, kn_hw, eps, rope, L);
    else return -4;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// VisionEmbeddings between its convs: x bf16 [gh * gw, C] (the patch conv's output, rows in raster order) takes GELU,
// rounded, then an interleaved RoPE in float32: pairs (2i, 2i + 1) of the first C / 2 channels turn by the column, of
// the others by the row, at angle position * freqs[i]. `freqs` has C / 4 values.
// ---------------------------------------------------------------------------------------------
__global__ void sn_gelu_rope2d_kernel(bf16* x, int gw, int C, const float* freqs, int64_t pairs) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= pairs) return;
    const int half = C / 2, per_row = C / 2;
    const int64_t row = t / per_row;
    const int pr = (int)(t % per_row);
    const bool y_part = pr >= half / 2;
    const int i = y_part ? pr - half / 2 : pr;
    const int pos = y_part ? (int)(row / gw) : (int)(row % gw);
    bf16* p = x + row * C + (y_part ? half : 0) + 2 * i;
    const float a = rnd<bf16>(gelu_f(__bfloat162float(p[0]))), b = rnd<bf16>(gelu_f(__bfloat162float(p[1])));
    float s, c;
    sincosf((float)pos * freqs[i], &s, &c);
    p[0] = __float2bfloat16_rn(c * a + -s * b);
    p[1] = __float2bfloat16_rn(s * a + c * b);
}

GPU_API int gpu_sn_gelu_rope2d(void* x, int gh, int gw, int C, const float* freqs, void* stream) {
    if (C % 4) return -1;
    const int64_t pairs = (int64_t)gh * gw * C / 2;
    if (pairs) sn_gelu_rope2d_kernel<<<(unsigned)((pairs + 255) / 256), 256, 0, (cudaStream_t)stream>>>((bf16*)x, gw, C, freqs, pairs);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The input of a 2x2, stride 2 conv as GEMM rows: x bf16 [H, W, C] -> out [(H / 2) * (W / 2), C * 4], column
// c * 4 + ky * 2 + kx, the order of a [Cout, C, 2, 2] weight.
// ---------------------------------------------------------------------------------------------
__global__ void sn_space_to_depth2_kernel(const bf16* x, bf16* out, int H, int W, int C) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = H / 2, ow = W / 2;
    if (i >= (int64_t)oh * ow * C * 4) return;
    const int col = (int)(i % (C * 4));
    const int64_t o = i / (C * 4);
    const int c = col / 4, ky = (col / 2) & 1, kx = col & 1;
    const int oy = (int)(o / ow), ox = (int)(o % ow);
    out[i] = x[((int64_t)(oy * 2 + ky) * W + ox * 2 + kx) * C + c];
}

GPU_API int gpu_sn_space_to_depth2(const void* x, void* out, int H, int W, int C, void* stream) {
    if (H % 2 || W % 2) return -1;
    const int64_t n = (int64_t)H * W * C;
    if (n) sn_space_to_depth2_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, (bf16*)out, H, W, C);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// torch's PixelShuffle(r) on channel-last bf16: in [H, W, C * r * r] -> out [H * r, W * r, C], output (Y, X, c) from
// input channel c * r * r + (Y % r) * r + X % r.
// ---------------------------------------------------------------------------------------------
__global__ void sn_pixel_shuffle_kernel(const bf16* in, bf16* out, int H, int W, int C, int r) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int OW = W * r;
    if (i >= (int64_t)H * r * OW * C) return;
    const int c = (int)(i % C);
    const int64_t px = i / C;
    const int X = (int)(px % OW), Y = (int)(px / OW);
    out[i] = in[((int64_t)(Y / r) * W + X / r) * C * r * r + (int64_t)c * r * r + (Y % r) * r + X % r];
}

GPU_API int gpu_sn_pixel_shuffle(const void* in, void* out, int H, int W, int C, int r, void* stream) {
    const int64_t n = (int64_t)H * W * C * r * r;
    if (n) sn_pixel_shuffle_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)in, (bf16*)out, H, W, C, r);
    KCHECK();
    return 0;
}

// nn.GELU (erf) in place, bf16
__global__ void sn_gelu_kernel(bf16* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16_rn(gelu_f(__bfloat162float(x[i])));
}

GPU_API int gpu_sn_gelu(void* x, int64_t n, void* stream) {
    if (n) sn_gelu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((bf16*)x, n);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The head's last PixelShuffle(8) and what ComfyUI makes of it: the prediction pred [3, H, W] is read from the conv's
// output y bf16 [H / 8, W / 8, 192]; the model returns velocity = (x - pred) / denom, the difference in bf16 (x the
// input rounded to bf16) and the division by the float32 denominator in float32, and the sampler's denoised is
// x - velocity * sigma in float32. x and den are planar float32 [3, H, W].
// ---------------------------------------------------------------------------------------------
__global__ void sn_denoise_kernel(const bf16* y, const float* x, float* den, int H, int W, float sigma, float denom) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)3 * H * W) return;
    const int c = (int)(i / ((int64_t)H * W));
    const int Y = (int)((i / W) % H), X = (int)(i % W);
    const float pred = __bfloat162float(y[((int64_t)(Y / 8) * (W / 8) + X / 8) * 192 + c * 64 + (Y % 8) * 8 + X % 8]);
    const float xv = x[i];
    const float v = rnd<bf16>(rnd<bf16>(xv) - pred) / denom;
    den[i] = xv - v * sigma;
}

GPU_API int gpu_sn_denoise(const void* y, const float* x, float* den, int H, int W, float sigma, float denom, void* stream) {
    if (H % 8 || W % 8) return -1;
    const int64_t n = (int64_t)3 * H * W;
    if (n) sn_denoise_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)y, x, den, H, W, sigma, denom);
    KCHECK();
    return 0;
}
