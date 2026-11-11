// SeedVR2: the DiT's patch embedding input, the window attention's way back to token order, and the VAE
// decoder's pixel shuffle.
#include "common.cuh"

// x and cond: float [16, H, W] (the noisy latent and the VAE's, at model scale). tok: bf16 [(H/2)*(W/2), ld] in
// (ph, pw, c) order over 33 channels: x, cond, then the all-ones mask; columns 132..ld are zeroed.
__global__ void seedvr_patch_in_kernel(const float* x, const float* cond, bf16* tok, int H, int W, int ld) {
    const int64_t t = blockIdx.x;
    const int wt = W / 2;
    const int ty = (int)(t / wt), tx = (int)(t % wt);
    for (int f = threadIdx.x; f < ld; f += blockDim.x) {
        float v = 0.f;
        if (f < 132) {
            const int p = f / 33, c = f % 33;
            const int y = ty * 2 + p / 2, xx = tx * 2 + p % 2;
            const int64_t at = (int64_t)y * W + xx;
            if (c < 16) v = x[(int64_t)c * H * W + at];
            else if (c < 32) v = cond[(int64_t)(c - 16) * H * W + at];
            else v = 1.f;
        }
        tok[t * ld + f] = __float2bfloat16_rn(v);
    }
}

GPU_API int gpu_seedvr_patch_in(const float* x, const float* cond, void* tok, int H, int W, int ld, void* stream) {
    if (H % 2 || W % 2 || ld < 132) return -1;
    const int64_t n = (int64_t)(H / 2) * (W / 2);
    if (n > 0) seedvr_patch_in_kernel<<<(unsigned)n, 128, 0, (cudaStream_t)stream>>>(x, cond, (bf16*)tok, H, W, ld);
    KCHECK();
    return 0;
}

// out[r] = mean of in[src[k]] for k in offs[r] .. offs[r + 1]: a token's attention output from the windows it was in
// (once for an image token, every window for a text token). bf16 rows of C, accumulated in float, rounded once.
__global__ void seedvr_unwindow_kernel(const bf16* in, int C, const int* src, const int* offs, bf16* out) {
    const int r = blockIdx.x;
    const int k0 = offs[r], k1 = offs[r + 1];
    const float inv = 1.f / (float)(k1 - k0);
    for (int c = (blockIdx.y * blockDim.x + threadIdx.x) * 2; c < C; c += gridDim.y * blockDim.x * 2) {
        float a = 0.f, b = 0.f;
        for (int k = k0; k < k1; ++k) {
            const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(in + (int64_t)src[k] * C + c));
            a += v.x;
            b += v.y;
        }
        if (k1 - k0 > 1) {
            a *= inv;
            b *= inv;
        }
        *reinterpret_cast<__nv_bfloat162*>(out + (int64_t)r * C + c) = __floats2bfloat162_rn(a, b);
    }
}

GPU_API int gpu_seedvr_unwindow(const void* in, int C, const int* src, const int* offs, int rows, void* out, void* stream) {
    if (C % 2) return -1;
    if (rows > 0) {
        dim3 grid(rows, (C + 511) / 512);
        seedvr_unwindow_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const bf16*)in, C, src, offs, (bf16*)out);
    }
    KCHECK();
    return 0;
}

// in: bf16 [H, W, 4C] whose channel (i * 2 + j) * C + c goes to out [2H, 2W, C] at (2y + i, 2x + j, c).
__global__ void pixel_shuffle2_kernel(const bf16* in, bf16* out, int H, int W, int C) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int C8 = C / 8;
    const int64_t n = (int64_t)H * 2 * W * 2 * C8;
    if (i >= n) return;
    const int c8 = (int)(i % C8);
    const int64_t px = i / C8;
    const int ox = (int)(px % (2 * W)), oy = (int)(px / (2 * W));
    const int y = oy / 2, x = ox / 2, ph = ((oy & 1) * 2 + (ox & 1));
    const uint4 v = *reinterpret_cast<const uint4*>(in + (((int64_t)y * W + x) * 4 + ph) * C + c8 * 8);
    *reinterpret_cast<uint4*>(out + px * C + c8 * 8) = v;
}

GPU_API int gpu_pixel_shuffle2_bf16(const void* in, void* out, int H, int W, int C, void* stream) {
    if (C % 8) return -1;
    const int64_t n = (int64_t)H * 2 * W * 2 * (C / 8);
    if (n > 0) pixel_shuffle2_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)in, (bf16*)out, H, W, C);
    KCHECK();
    return 0;
}
