// Kernels for Qwen Image 2.1: the shortcuts of its VAE (a Wan 2.2 layout with one frame), the attention of the
// text and reference tokens that come before the image, and small elementwise helpers. bf16, channel-last.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// DupUp3D of one frame, the decoder's shortcut. x [H, W, CIN] -> out [2H, 2W, COUT]. The channels are repeated
// to COUT * ft * 4 and read as (COUT, ft, 2, 2); with two frames only the second one is kept (first_chunk).
// ---------------------------------------------------------------------------------------------
__global__ void dup_up_kernel(const bf16* x, bf16* out, int H, int W, int cin, int cout, int ft) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)4 * H * W * cout) return;
    const int o = (int)(i % cout);
    const int64_t p = i / cout;
    const int ox = (int)(p % (2 * W)), oy = (int)(p / (2 * W));
    const int a = oy & 1, b = ox & 1, t = ft == 2 ? 1 : 0;
    const int c = ((o * ft + t) * 2 + a) * 2 + b;
    const int repeats = cout * ft * 4 / cin;
    out[i] = x[((int64_t)(oy >> 1) * W + (ox >> 1)) * cin + c / repeats];
}

GPU_API int gpu_dup_up(const void* x, void* out, int H, int W, int cin, int cout, int ft, void* stream) {
    if ((cout * ft * 4) % cin) return -1;
    const int64_t n = (int64_t)4 * H * W * cout;
    dup_up_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, (bf16*)out, H, W, cin, cout, ft);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// AvgDown3D of one frame, the encoder's shortcut. x [H, W, CIN] -> out [H / fs, W / fs, COUT]. The channels
// gain (ft, fs, fs) from the space and time, and groups of them are averaged. With two time steps the first is the
// zero frame that pads the single image, so it counts in the mean but adds nothing.
// ---------------------------------------------------------------------------------------------
__global__ void avg_down_kernel(const bf16* x, bf16* out, int H, int W, int cin, int cout, int ft, int fs) {
    const int ow = W / fs, oh = H / fs;
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)oh * ow * cout) return;
    const int o = (int)(i % cout);
    const int64_t p = i / cout;
    const int px = (int)(p % ow), py = (int)(p / ow);
    const int factor = ft * fs * fs, gs = cin * factor / cout;
    float sum = 0.f;
    for (int g = 0; g < gs; ++g) {
        const int c2 = o * gs + g;
        const int c = c2 / factor, r = c2 % factor;
        const int t = r / (fs * fs), a = (r / fs) % fs, b = r % fs;
        if (ft == 2 && t == 0) continue;
        sum += __bfloat162float(x[((int64_t)(py * fs + a) * W + px * fs + b) * cin + c]);
    }
    out[i] = __float2bfloat16_rn(sum / (float)gs);
}

GPU_API int gpu_avg_down(const void* x, void* out, int H, int W, int cin, int cout, int ft, int fs, void* stream) {
    if ((cin * ft * fs * fs) % cout) return -1;
    const int64_t n = (int64_t)(H / fs) * (W / fs) * cout;
    avg_down_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, (bf16*)out, H, W, cin, cout, ft, fs);
    KCHECK();
    return 0;
}

// NHWC bf16 [P, cstride] (first 4 channels) -> RGBA8 like ComfyUI saves a decoded image.
__global__ void to_rgba8_kernel(const bf16* x, int cstride, uint8_t* out, int64_t P) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * 4) return;
    const int64_t p = i / 4;
    const int c = (int)(i % 4);
    float v = __bfloat162float(x[p * cstride + c]);
    v = fminf(fmaxf((v + 1.f) / 2.f, 0.f), 1.f);
    v = fminf(fmaxf(255.f * v, 0.f), 255.f);
    out[i] = (uint8_t)v;
}

GPU_API int gpu_to_rgba8(const void* x, int cstride, uint8_t* out, int64_t P, void* stream) {
    to_rgba8_kernel<<<(unsigned)((P * 4 + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const bf16*)x, cstride, out, P);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Exact attention of a run of tokens that see everything before them, and of each other causally: query i of
// the run attends to keys 0 .. offset + i. q [lq, h * 128], k and v [lk, h * 128] (bf16, rows `*_stride` apart).
// ---------------------------------------------------------------------------------------------
__global__ void attn_prefix_kernel(const bf16* q, int64_t q_stride, const bf16* kp, const bf16* vp, int64_t kv_stride, int offset,
                                   bf16* out, int64_t out_stride, float scale) {
    extern __shared__ float sm[];
    constexpr int D = 128;
    float* qs = sm;       // D
    float* p = sm + D;    // keys
    __shared__ float red[33];
    const int i = blockIdx.x, h = blockIdx.y, e0 = threadIdx.x;
    qs[e0] = __bfloat162float(q[(int64_t)i * q_stride + h * D + e0]);
    __syncthreads();
    const int n = offset + i + 1;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const bf16* kr = kp + (int64_t)j * kv_stride + h * D;
        float s = 0.f;
        for (int e = 0; e < D; ++e) s += qs[e] * __bfloat162float(kr[e]);
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
    for (int j = 0; j < n; ++j) acc += p[j] * __bfloat162float(vp[(int64_t)j * kv_stride + h * D + e0]);
    out[(int64_t)i * out_stride + h * D + e0] = __float2bfloat16_rn(acc / sum);
}

GPU_API int gpu_attn_prefix(const void* q, int64_t q_stride, const void* k, const void* v, int64_t kv_stride, int h, int lq, int offset,
                              void* out, int64_t out_stride, float scale, void* stream) {
    dim3 grid(lq, h);
    const size_t smem = ((size_t)128 + offset + lq) * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(attn_prefix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attn_prefix_kernel<<<grid, 128, smem, (cudaStream_t)stream>>>((const bf16*)q, q_stride, (const bf16*)k, (const bf16*)v, kv_stride, offset,
                                                                  (bf16*)out, out_stride, scale);
    KCHECK();
    return 0;
}

// x = tanh(x), bf16
__global__ void tanh_bf16_kernel(bf16* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16_rn(tanhf(__bfloat162float(x[i])));
}

GPU_API int gpu_tanh_bf16(void* x, int64_t n, void* stream) {
    tanh_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((bf16*)x, n);
    KCHECK();
    return 0;
}
