// Kernels for TripoSplat / Idol (image to gaussian splat): the fp16 element-wise ops ComfyUI runs eagerly around the
// flow model's and the octree gaussian decoder's matmuls.
//
// Rounding follows ComfyUI's fp16 tensors: rnd<half> where it materializes one.
#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// x = x + y * gate (torch.addcmul in fp16, one rounding), or x + y without a gate. gate: [cols] fp16.
// ---------------------------------------------------------------------------------------------
__global__ void ts_add_kernel(half* x, int64_t x_stride, const half* y, int64_t y_stride, const half* gate, int64_t rows,
                              int cols) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int64_t r = i / cols;
    const int c = (int)(i % cols);
    const float a = __half2float(x[r * x_stride + c]), b = __half2float(y[r * y_stride + c]);
    x[r * x_stride + c] = __float2half_rn(gate ? fmaf(b, __half2float(gate[c]), a) : a + b);
}

GPU_API int gpu_ts_add_f16(void* x, int64_t x_stride, const void* y, int64_t y_stride, const void* gate, int64_t rows,
                             int cols, void* stream) {
    const int64_t n = rows * cols;
    if (n == 0) return 0;
    ts_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((half*)x, x_stride, (const half*)y, y_stride,
                                                                             (const half*)gate, rows, cols);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// RePo3DRotaryEmbedding + MultiHeadRMSNorm on q and k in place, head dim 64, one warp per (token, head, q|k):
// the rotation angles are delta[axis] * freq * pi from the per-head 3D offsets the RePo layer predicts (fp16
// [L, heads * 3]); the pair (2i, 2i+1) rotates in f32 and rounds to fp16 (flux's apply_rope), then rms_norm (eps,
// rounded) and the fp16 gamma multiply (rounded). delta null: the norm only. freqs: f32 [32], the 10 + 10 + 12
// frequencies of the three axes.
// ---------------------------------------------------------------------------------------------
__global__ void ts_repo_qk_kernel(half* buf, int64_t row_stride, int k_off, const half* delta, int64_t delta_stride,
                                  const float* freqs, const half* qw, const half* kw, float eps, int L, int H, int do_k) {
    constexpr int D = 64;
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int per_tok = H * (do_k ? 2 : 1);
    if (warp >= (int64_t)L * per_tok) return;
    const int t = (int)(warp / per_tok), j = (int)(warp % per_tok);
    const bool isk = j >= H;
    const int h = isk ? j - H : j;
    half* x = buf + (int64_t)t * row_stride + (isk ? k_off : 0) + h * D;
    // lane owns pair `lane`: elements 2 lane, 2 lane + 1
    float a = __half2float(x[2 * lane]), b = __half2float(x[2 * lane + 1]);
    if (delta) {
        const int axis = lane < 10 ? 0 : lane < 20 ? 1 : 2;
        const float d = __half2float(delta[(int64_t)t * delta_stride + h * 3 + axis]);
        const float ang = __fmul_rn(__fmul_rn(d, freqs[lane]), 3.14159265358979323846f);
        const float c = cosf(ang), s = sinf(ang);
        const float ra = __fadd_rn(__fmul_rn(c, a), __fmul_rn(-s, b));
        const float rb = __fadd_rn(__fmul_rn(s, a), __fmul_rn(c, b));
        a = rnd<half>(ra);
        b = rnd<half>(rb);
    }
    const float ss = warp_sum(a * a + b * b);
    const float rstd = rsqrtf(ss / (float)D + eps);
    const half* w = (isk ? kw : qw) + h * D;
    x[2 * lane] = __float2half_rn(rnd<half>(a * rstd) * __half2float(w[2 * lane]));
    x[2 * lane + 1] = __float2half_rn(rnd<half>(b * rstd) * __half2float(w[2 * lane + 1]));
}

GPU_API int gpu_ts_repo_qk(void* buf, int64_t row_stride, int k_off, const void* delta, int64_t delta_stride,
                             const float* freqs, const void* qw, const void* kw, float eps, int L, int H, int do_k,
                             void* stream) {
    const int64_t warps = (int64_t)L * H * (do_k ? 2 : 1);
    if (warps == 0) return 0;
    const int threads = 256;
    ts_repo_qk_kernel<<<(unsigned)((warps * 32 + threads - 1) / threads), threads, 0, (cudaStream_t)stream>>>(
        (half*)buf, row_stride, k_off, (const half*)delta, delta_stride, freqs, (const half*)qw, (const half*)kw, eps, L, H,
        do_k);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// out = silu(x[:, :cols]) * x[:, cols:2 cols], each op rounded to fp16 (RePo's gate * content).
// ---------------------------------------------------------------------------------------------
__global__ void ts_silu_mul_kernel(const half* x, int64_t x_stride, half* out, int64_t out_stride, int64_t rows, int cols) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int64_t r = i / cols;
    const int c = (int)(i % cols);
    const float g = rnd<half>(silu_f(__half2float(x[r * x_stride + c])));
    out[r * out_stride + c] = __float2half_rn(g * __half2float(x[r * x_stride + cols + c]));
}

GPU_API int gpu_ts_silu_mul_f16(const void* x, int64_t x_stride, void* out, int64_t out_stride, int64_t rows, int cols,
                                  void* stream) {
    const int64_t n = rows * cols;
    if (n == 0) return 0;
    ts_silu_mul_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)x, x_stride, (half*)out,
                                                                                  out_stride, rows, cols);
    KCHECK();
    return 0;
}
