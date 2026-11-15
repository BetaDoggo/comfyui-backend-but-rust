// SD 1.x and SDXL: the fp16 UNet's convolutions, norms and activations, the float32 CLIP text encoders' pieces, and the
// int8 activation quantizer of the UNet's int8 linears.
#include "common.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------------------------
// conv2d NHWC fp16, 'same' padding (R / 2), stride 1 or 2, fp32 accumulate, bias broadcast via zero-stride C.
// ---------------------------------------------------------------------------------------------
namespace {

template <int TBM, int TBN, int TBK, int WM, int WN, int Stages, cutlass::conv::StrideSupport Stride>
struct ConvF16 {
    using Element = cutlass::half_t;
    using Layout = cutlass::layout::TensorNHWC;
    using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Element, Layout, Element, Layout, Element, Layout, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, TBK>, cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized, Stride>::Kernel;
    using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    static int run(const void* x, const void* w, const void* bias, void* out, int N, int H, int W, int C, int Cout, int R,
                   int stride, cudaStream_t stream) {
        const int pad = R / 2;
        const int OH = (H + 2 * pad - R) / stride + 1, OW = (W + 2 * pad - R) / stride + 1;
        cutlass::conv::Conv2dProblemSize problem(cutlass::Tensor4DCoord(N, H, W, C), cutlass::Tensor4DCoord(Cout, R, R, C),
                                                 cutlass::Tensor4DCoord(pad, pad, pad, pad), cutlass::MatrixCoord(stride, stride),
                                                 cutlass::MatrixCoord(1, 1), cutlass::Tensor4DCoord(N, OH, OW, Cout),
                                                 cutlass::conv::Mode::kCrossCorrelation, 1);
        using TensorRef = cutlass::TensorRef<Element, Layout>;
        TensorRef a((Element*)x, Layout::packed({N, H, W, C}));
        TensorRef b((Element*)w, Layout::packed({Cout, R, R, C}));
        TensorRef d((Element*)out, Layout::packed({N, OH, OW, Cout}));
        TensorRef c((Element*)bias, Layout(Layout::Stride(0)));
        typename Conv::Arguments args{problem, a, b, bias ? c : d, d, {1.f, bias ? 1.f : 0.f}};
        Conv op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return -300;
        const size_t ws = op.get_workspace_size(args);
        void* wsp = nullptr;
        if (ws && cudaMallocAsync(&wsp, ws, stream) != cudaSuccess) return -301;
        cutlass::Status s = op.initialize(args, wsp, stream);
        if (s == cutlass::Status::kSuccess) s = op(stream);
        if (wsp) cudaFreeAsync(wsp, stream);
        return s == cutlass::Status::kSuccess ? 0 : -302;
    }
};

}  // namespace

// x [N,H,W,C], w [Cout,R,R,C], bias [Cout] (nullable), out [N,H/stride,W/stride,Cout]; C and Cout multiples of 8.
GPU_API int gpu_conv2d_f16(const void* x, const void* w, const void* bias, void* out, int N, int H, int W, int C, int Cout, int R,
                           int stride, void* stream) {
    if (C % 8 || Cout % 8) return -310;
    cudaStream_t st = (cudaStream_t)stream;
    using cutlass::conv::StrideSupport;
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    // the output pixels and channels as a GEMM: 128x128 tiles unless they would leave SMs idle (the 8x8 and 16x16 levels)
    const int64_t m = (int64_t)N * (H / stride) * (W / stride);
    const bool small = (m + 127) / 128 * ((Cout + 127) / 128) < 2 * sms;
    int r;
    if (stride == 1) {
        if (small) r = ConvF16<64, 64, 32, 32, 32, 4, StrideSupport::kUnity>::run(x, w, bias, out, N, H, W, C, Cout, R, 1, st);
        else if (Cout >= 128) r = ConvF16<128, 128, 32, 64, 64, 4, StrideSupport::kUnity>::run(x, w, bias, out, N, H, W, C, Cout, R, 1, st);
        else r = ConvF16<128, 64, 32, 64, 32, 4, StrideSupport::kUnity>::run(x, w, bias, out, N, H, W, C, Cout, R, 1, st);
    } else if (stride == 2) {
        if (small) r = ConvF16<64, 64, 32, 32, 32, 4, StrideSupport::kStrided>::run(x, w, bias, out, N, H, W, C, Cout, R, 2, st);
        else r = ConvF16<128, 128, 32, 64, 64, 4, StrideSupport::kStrided>::run(x, w, bias, out, N, H, W, C, Cout, R, 2, st);
    } else {
        return -311;
    }
    if (r) return r;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// GroupNorm of fp16 x [N, P, C] per image, then optional SiLU, as torch computes it for half tensors: float statistics,
// the affine result rounded to half, SiLU of that rounded again.
// ---------------------------------------------------------------------------------------------
__global__ void sd_group_sums_kernel(const half* x, double* sums, int64_t P, int C, int G, int splits) {
    const int ng = blockIdx.x, split = blockIdx.y;
    const int n = ng / G, g = ng % G, per = C / G;
    const half* base = x + (int64_t)n * P * C + g * per;
    const int64_t p0 = P * split / splits, p1 = P * (split + 1) / splits;
    float s = 0.f, ss = 0.f;
    const int64_t count = (p1 - p0) * per;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        const int64_t p = p0 + i / per;
        const float v = __half2float(base[p * C + i % per]);
        s += v;
        ss += v * v;
    }
    __shared__ float red[33];
    s = block_sum(s, red);
    ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        atomicAdd(&sums[ng * 2], (double)s);
        atomicAdd(&sums[ng * 2 + 1], (double)ss);
    }
}

// Each group's (mean, rstd) in float, written over the first half of its pair of sums.
__global__ void sd_group_stats_kernel(double* sums, double cnt, float eps, int groups) {
    const int ng = blockIdx.x * blockDim.x + threadIdx.x;
    if (ng >= groups) return;
    const double mean = sums[ng * 2] / cnt;
    const double var = fmax(sums[ng * 2 + 1] / cnt - mean * mean, 0.0);
    ((float2*)sums)[ng * 2] = make_float2((float)mean, (float)(1.0 / sqrt(var + (double)eps)));
}

__global__ void sd_group_apply_kernel(const half* x, half* out, const float* w, const float* b, const double* sums, int64_t P, int C,
                                      int G, int silu, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % C);
    const int n = (int)(i / (P * C));
    const int ng = n * G + c / (C / G);
    const float2 st = ((const float2*)sums)[ng * 2];
    float y = rnd<half>((__half2float(x[i]) - st.x) * st.y * w[c] + b[c]);
    if (silu) y = silu_f(y);
    out[i] = __float2half_rn(y);
}

// sums: 16 * N * G bytes of scratch.
GPU_API int gpu_sd_group_norm(const void* x, void* out, const float* w, const float* b, void* sums, int N, int64_t P, int C, int G,
                              float eps, int silu, void* stream) {
    if (C % G) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    cudaMemsetAsync(sums, 0, (size_t)N * G * 2 * sizeof(double), st);
    const int64_t want = P * (C / G) / 4096;
    const int splits = want < 1 ? 1 : want > 64 ? 64 : (int)want;
    sd_group_sums_kernel<<<dim3(N * G, splits), 256, 0, st>>>((const half*)x, (double*)sums, P, C, G, splits);
    sd_group_stats_kernel<<<(N * G + 255) / 256, 256, 0, st>>>((double*)sums, (double)P * (C / G), eps, N * G);
    const int64_t total = (int64_t)N * P * C;
    sd_group_apply_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>((const half*)x, (half*)out, w, b, (const double*)sums, P, C, G,
                                                                          silu, total);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// LayerNorm of rows [M, K] (half or float) with float weight and bias, out in the input's type.
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ void sd_layer_norm_kernel(const T* x, int64_t x_stride, T* out, int64_t out_stride, const float* w, const float* b, int K,
                                     float eps) {
    __shared__ float red[33];
    const T* r = x + blockIdx.x * x_stride;
    float s = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) s += ldf(r, k);
    const float mean = block_sum(s, red) / K;
    float ss = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float d = ldf(r, k) - mean;
        ss += d * d;
    }
    const float rstd = rsqrtf(block_sum(ss, red) / K + eps);
    T* o = out + blockIdx.x * out_stride;
    for (int k = threadIdx.x; k < K; k += blockDim.x) stf(o, k, (ldf(r, k) - mean) * rstd * w[k] + b[k]);
}

GPU_API int gpu_sd_layer_norm(int dtype, const void* x, int64_t x_stride, void* out, int64_t out_stride, const float* w, const float* b,
                              int M, int K, float eps, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (dtype == DT_F16) sd_layer_norm_kernel<half><<<M, 256, 0, st>>>((const half*)x, x_stride, (half*)out, out_stride, w, b, K, eps);
    else if (dtype == DT_F32) sd_layer_norm_kernel<float><<<M, 256, 0, st>>>((const float*)x, x_stride, (float*)out, out_stride, w, b, K, eps);
    else return -4;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------------------------
// GEGLU: in [M, 2I] (value, then gate) -> out [M, I] = value * gelu(gate), each half op rounded
__global__ void sd_geglu_kernel(const half* in, half* out, int64_t M, int I) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * I) return;
    const int64_t m = i / I;
    const int c = (int)(i % I);
    const float a = __half2float(in[m * 2 * I + c]);
    const float g = rnd<half>(gelu_f(__half2float(in[m * 2 * I + I + c])));
    out[i] = __float2half_rn(a * g);
}
GPU_API int gpu_sd_geglu(const void* in, void* out, int64_t M, int I, void* stream) {
    const int64_t n = M * I;
    sd_geglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)in, (half*)out, M, I);
    KCHECK();
    return 0;
}

// x = x + y, in half
__global__ void sd_add_kernel(half* x, const half* y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half_rn(__half2float(x[i]) + __half2float(y[i]));
}
GPU_API int gpu_sd_add(void* x, const void* y, int64_t n, void* stream) {
    sd_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((half*)x, (const half*)y, n);
    KCHECK();
    return 0;
}

// Nearest upsample of x [N, H, W, C] to out [N, OH, OW, C], OH 2H or 2H - 1 (the skip connection's size, as ComfyUI's
// Upsample takes it). torch's nearest reads floor(o * H / OH), which is o / 2 for both sizes.
__global__ void sd_upsample_kernel(const half* x, half* out, int H, int W, int OH, int OW, int C, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % C);
    int64_t p = i / C;
    const int ox = (int)(p % OW);
    p /= OW;
    const int oy = (int)(p % OH);
    const int64_t n = p / OH;
    out[i] = x[((n * H + (oy >> 1)) * W + (ox >> 1)) * C + c];
}
GPU_API int gpu_sd_upsample(const void* x, void* out, int N, int H, int W, int OH, int OW, int C, void* stream) {
    if (OH != 2 * H && OH != 2 * H - 1) return -1;
    if (OW != 2 * W && OW != 2 * W - 1) return -1;
    const int64_t total = (int64_t)N * OH * OW * C;
    sd_upsample_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)x, (half*)out, H, W, OH, OW, C,
                                                                                          total);
    KCHECK();
    return 0;
}

// x [N, P, C] += emb [N, C] (each image's embedding, broadcast over its pixels), in half
__global__ void sd_add_emb_kernel(half* x, const half* emb, int64_t P, int C, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t n = i / (P * C);
    const int c = (int)(i % C);
    x[i] = __float2half_rn(__half2float(x[i]) + __half2float(emb[n * C + c]));
}
GPU_API int gpu_sd_add_emb(void* x, const void* emb, int N, int64_t P, int C, void* stream) {
    const int64_t total = (int64_t)N * P * C;
    sd_add_emb_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((half*)x, (const half*)emb, P, C, total);
    KCHECK();
    return 0;
}

// SiLU in place, in half
__global__ void sd_silu_kernel(half* x, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half_rn(silu_f(__half2float(x[i])));
}
GPU_API int gpu_sd_silu(void* x, int64_t n, void* stream) {
    sd_silu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((half*)x, n);
    KCHECK();
    return 0;
}

// CLIP's activations in place, float32: 0 quick_gelu, 1 gelu
__global__ void sd_clip_act_kernel(float* x, int64_t n, int mode) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = mode == 0 ? v * (1.f / (1.f + expf(-1.702f * v))) : gelu_f(v);
}
GPU_API int gpu_sd_clip_act(float* x, int64_t n, int mode, void* stream) {
    sd_clip_act_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, n, mode);
    KCHECK();
    return 0;
}

// CLIP embeddings: out [L, D] f32 = token[ids[l]] + position[l], both tables half
__global__ void sd_clip_embed_kernel(const half* tok, const half* pos, const int* ids, float* out, int D) {
    const int l = blockIdx.x;
    const int64_t t = ids[l];
    for (int d = threadIdx.x; d < D; d += blockDim.x) out[(int64_t)l * D + d] = __half2float(tok[t * D + d]) + __half2float(pos[(int64_t)l * D + d]);
}
GPU_API int gpu_sd_clip_embed(const void* tok, const void* pos, const int* ids, float* out, int L, int D, void* stream) {
    sd_clip_embed_kernel<<<L, 256, 0, (cudaStream_t)stream>>>((const half*)tok, (const half*)pos, ids, out, D);
    KCHECK();
    return 0;
}

// The UNet's input: latent f32 [N, C, H, W] / sqrt(sigma^2 + 1), as half NHWC [N, H, W, cs] with channels past C zero.
__global__ void sd_unet_in_kernel(const float* lat, half* out, int C, int64_t HW, int cs, const float* sigmas, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % cs);
    const int64_t p = (i / cs) % HW, n = i / (cs * HW);
    float v = 0.f;
    if (c < C) {
        const float s = sigmas[n];
        v = lat[(n * C + c) * HW + p] / sqrtf(s * s + 1.f);
    }
    out[i] = __float2half_rn(v);
}
GPU_API int gpu_sd_unet_in(const float* lat, void* out, int N, int C, int64_t HW, int cs, const float* sigmas, void* stream) {
    const int64_t total = (int64_t)N * HW * cs;
    sd_unet_in_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lat, (half*)out, C, HW, cs, sigmas, total);
    KCHECK();
    return 0;
}

// The prediction from the UNet's output y (half NHWC, cs channels): x - y * sigma (eps), or v-prediction's
// x / (sigma^2 + 1) - y * sigma / sqrt(sigma^2 + 1), in float32 NCHW.
__global__ void sd_unet_out_kernel(const half* y, const float* x, float* out, int C, int64_t HW, int cs, const float* sigmas, int v,
                                   int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t p = i % HW, c = (i / HW) % C, n = i / (HW * C);
    const float s = sigmas[n];
    const float m = __half2float(y[(n * HW + p) * cs + c]);
    out[i] = v ? x[i] * 1.f / (s * s + 1.f) - m * s * 1.f / sqrtf(s * s + 1.f) : x[i] - m * s;
}
GPU_API int gpu_sd_unet_out(const void* y, const float* x, float* out, int N, int C, int64_t HW, int cs, const float* sigmas, int v,
                            void* stream) {
    const int64_t total = (int64_t)N * C * HW;
    sd_unet_out_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)y, x, out, C, HW, cs, sigmas, v, total);
    KCHECK();
    return 0;
}

// Heads d wide, at a row stride, into heads dp wide (zero padded) for the int8 attention: dst [L, H, dp].
__global__ void sd_pad_heads_kernel(const half* src, int64_t stride, half* dst, int H, int d, int dp, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int j = (int)(i % dp);
    const int64_t lh = i / dp;
    const int h = (int)(lh % H);
    const int64_t l = lh / H;
    dst[i] = j < d ? src[l * stride + (int64_t)h * d + j] : __float2half_rn(0.f);
}
GPU_API int gpu_sd_pad_heads(const void* src, int64_t stride, void* dst, int64_t L, int H, int d, int dp, void* stream) {
    const int64_t total = L * H * dp;
    sd_pad_heads_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)src, stride, (half*)dst, H, d, dp, total);
    KCHECK();
    return 0;
}

// [L, H, dp] -> [L, H * d], dropping the padding
__global__ void sd_unpad_heads_kernel(const half* src, half* dst, int H, int d, int dp, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int j = (int)(i % d);
    const int64_t lh = i / d;
    dst[i] = src[lh * dp + j];
}
GPU_API int gpu_sd_unpad_heads(const void* src, void* dst, int64_t L, int H, int d, int dp, void* stream) {
    const int64_t total = L * H * d;
    sd_unpad_heads_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const half*)src, (half*)dst, H, d, dp, total);
    KCHECK();
    return 0;
}

// Row softmax of float scores [rows, n] times scale, to half probabilities
__global__ void sd_softmax_kernel(const float* s, half* p, int n, float scale) {
    __shared__ float red[33];
    const float* sr = s + (int64_t)blockIdx.x * n;
    half* pr = p + (int64_t)blockIdx.x * n;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, sr[j] * scale);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) sum += expf(sr[j] * scale - mx);
    sum = block_sum(sum, red);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) pr[j] = __float2half_rn(expf(sr[j] * scale - mx) * inv);
}
GPU_API int gpu_sd_softmax(const float* s, void* p, int rows, int n, float scale, void* stream) {
    sd_softmax_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(s, (half*)p, n, scale);
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// int8 per-row quantization of half rows [M, K] (activations, or weights at load), optionally ConvRot-rotated in groups
// of `rotate` (64 or 256) first: scale = absmax / 127 (kept in float), q = round(x / scale) in half, as the int8 linears'
// quantizer.
// ---------------------------------------------------------------------------------------------
// The normalized Hadamard transform of each group of 64 in a row held in shared memory: fht256_rows_smem's butterflies,
// three stages instead of four.
__device__ __forceinline__ void fht64_rows_smem(float* row, int K) {
    const int nbf = K >> 2;
#pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        const int s = 1 << (2 * stage);
        for (int b = threadIdx.x; b < nbf; b += blockDim.x) {
            const int g = b >> 4, r = b & 15;
            const int i0 = (g << 6) + (r / s) * (s << 2) + (r % s);
            const float x0 = row[i0], x1 = row[i0 + s], x2 = row[i0 + 2 * s], x3 = row[i0 + 3 * s];
            row[i0] = 0.5f * h4_row(0, x0, x1, x2, x3);
            row[i0 + s] = 0.5f * h4_row(1, x0, x1, x2, x3);
            row[i0 + 2 * s] = 0.5f * h4_row(2, x0, x1, x2, x3);
            row[i0 + 3 * s] = 0.5f * h4_row(3, x0, x1, x2, x3);
        }
        __syncthreads();
    }
}

__global__ void sd_quant_rows_kernel(const half* x, int64_t x_stride, int8_t* q, float* scales, int K, int rotate) {
    extern __shared__ float row[];
    __shared__ float red[33];
    const int64_t m = blockIdx.x;
    const half* r = x + m * x_stride;
    for (int k = threadIdx.x; k < K; k += blockDim.x) row[k] = __half2float(r[k]);
    __syncthreads();
    if (rotate == 256) fht256_rows_smem(row, K);
    else if (rotate == 64) fht64_rows_smem(row, K);
    float mx = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(rnd<half>(row[k])));
    mx = block_max(mx, red);
    const float scale = fmaxf(mx, 1e-30f) / 127.f;
    if (threadIdx.x == 0) scales[m] = scale;
    // a scale below half's range divides in float
    const float scale_t = rnd<half>(scale) > 0.f ? rnd<half>(scale) : scale;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float t = rnd<half>(rnd<half>(row[k]) / scale_t);
        q[m * K + k] = (int8_t)fminf(127.f, fmaxf(-128.f, nearbyintf(t)));
    }
}
GPU_API int gpu_sd_quant_rows(const void* x, int64_t x_stride, int8_t* q, float* scales, int M, int K, int rotate, void* stream) {
    if ((rotate != 0 && rotate != 64 && rotate != 256) || (rotate && K % rotate)) return -1;
    const size_t smem = (size_t)K * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(sd_quant_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sd_quant_rows_kernel<<<M, 256, smem, (cudaStream_t)stream>>>((const half*)x, x_stride, q, scales, K, rotate);
    KCHECK();
    return 0;
}

// The VAE decoder's input: the sampler's latent f32 [C, HW] / scale, as bf16 NHWC [HW, cs] with channels past C zero.
__global__ void sd_vae_in_kernel(const float* lat, bf16* out, int C, int64_t HW, int cs, float scale) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW * cs) return;
    const int c = (int)(i % cs);
    const int64_t p = i / cs;
    out[i] = __float2bfloat16_rn(c < C ? lat[c * HW + p] / scale : 0.f);
}
GPU_API int gpu_sd_vae_in(const float* lat, void* out, int C, int64_t HW, int cs, float scale, void* stream) {
    const int64_t n = HW * cs;
    sd_vae_in_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(lat, (bf16*)out, C, HW, cs, scale);
    KCHECK();
    return 0;
}
