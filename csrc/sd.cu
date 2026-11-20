// SD 1.x and SDXL: the fp16 UNet's convolutions, norms and activations, the float32 CLIP text encoders' pieces, and the
// int8 activation quantizer of the UNet's int8 linears.
#include "common.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"

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

// The epilogue of a conv whose output joins a residual: half(half(acc + bias) + residual), the conv's output rounded as
// ConvF16's epilogue rounds it, then the residual added as gpu_sd_add adds it.
template <int Count>
class ConvF16BiasResidual {
public:
    static bool const kIsSingleSource = true;
    using ElementOutput = cutlass::half_t;
    using ElementC = ElementOutput;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementVector = ElementOutput;
    static int const kElementsPerAccess = Count;
    static int const kCount = Count;
    using FragmentAccumulator = cutlass::Array<float, Count>;
    using FragmentCompute = cutlass::Array<float, Count>;
    using FragmentC = cutlass::Array<ElementOutput, Count>;
    using FragmentOutput = FragmentC;
    using ElementZ = ElementOutput;
    using ElementT = ElementOutput;
    using FragmentZ = FragmentC;
    using FragmentT = FragmentC;
    static bool const kIsHeavy = false;
    static bool const kStoreZ = true;
    static bool const kStoreT = false;
    struct Params {
        CUTLASS_HOST_DEVICE Params() {}
    };
    CUTLASS_HOST_DEVICE ConvF16BiasResidual(Params const&) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_HOST_DEVICE void operator()(FragmentOutput& z, FragmentOutput&, FragmentAccumulator const& ab, FragmentC const& resid,
                                        FragmentCompute const& bias) const {
        cutlass::NumericConverter<ElementOutput, float, cutlass::FloatRoundStyle::round_to_nearest> to_half;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Count; ++i) {
            const float conv = float(to_half(ab[i] + bias[i]));
            z[i] = to_half(conv + float(resid[i]));
        }
    }
    CUTLASS_HOST_DEVICE void operator()(FragmentOutput&, FragmentOutput&, FragmentAccumulator const&, FragmentCompute const&) const {}
};

// ConvF16 at stride 1 with the residual epilogue: the same mainloop, so the same sums
template <int TBM, int TBN, int TBK, int WM, int WN, int Stages>
struct ConvF16Residual {
    using Element = cutlass::half_t;
    using Layout = cutlass::layout::TensorNHWC;
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
        Element, Layout, Element, Layout, Element, Layout, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, TBK>, cutlass::gemm::GemmShape<16, 8, 16>,
        ConvF16BiasResidual<8>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized, cutlass::conv::StrideSupport::kUnity>::Kernel;
    using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    static int run(const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int H, int W, int C, int Cout,
                   int R, cudaStream_t stream) {
        const int pad = R / 2;
        cutlass::conv::Conv2dProblemSize problem(cutlass::Tensor4DCoord(N, H, W, C), cutlass::Tensor4DCoord(Cout, R, R, C),
                                                 cutlass::Tensor4DCoord(pad, pad, pad, pad), cutlass::MatrixCoord(1, 1),
                                                 cutlass::MatrixCoord(1, 1), cutlass::Tensor4DCoord(N, H, W, Cout),
                                                 cutlass::conv::Mode::kCrossCorrelation, 1);
        using TensorRef = cutlass::TensorRef<Element, Layout>;
        TensorRef a((Element*)x, Layout::packed({N, H, W, C}));
        TensorRef b((Element*)w, Layout::packed({Cout, R, R, C}));
        TensorRef c((Element*)resid, Layout::packed({N, H, W, Cout}));
        TensorRef d((Element*)out, Layout::packed({N, H, W, Cout}));
        typename Conv::Arguments args(problem, a, b, c, d, {}, cutlass::conv::SplitKMode::kSerial, const_cast<void*>(bias), nullptr, 0, Cout);
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

// gpu_conv2d_f16 at stride 1, then gpu_sd_add of `resid`, in one pass, rounded as those two round. bias is required; out
// must not alias resid or x.
GPU_API int gpu_conv2d_f16_residual(const void* x, const void* w, const void* bias, const void* resid, void* out, int N, int H, int W,
                                    int C, int Cout, int R, void* stream) {
    if (C % 8 || Cout % 8) return -310;
    if (!bias) return -311;
    cudaStream_t st = (cudaStream_t)stream;
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    // gpu_conv2d_f16's tiles
    const int64_t m = (int64_t)N * H * W;
    const bool small = (m + 127) / 128 * ((Cout + 127) / 128) < 2 * sms;
    int r;
    if (small) r = ConvF16Residual<64, 64, 32, 32, 32, 4>::run(x, w, bias, resid, out, N, H, W, C, Cout, R, st);
    else if (Cout >= 128) r = ConvF16Residual<128, 128, 32, 64, 64, 4>::run(x, w, bias, resid, out, N, H, W, C, Cout, R, st);
    else r = ConvF16Residual<128, 64, 32, 64, 32, 4>::run(x, w, bias, resid, out, N, H, W, C, Cout, R, st);
    if (r) return r;
    KCHECK();
    return 0;
}

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
// x, or half(x + add[n, c]) when add [N, C] is given: a residual block's first conv output with its timestep embedding
__device__ __forceinline__ float sd_gn_in(const half* x, const half* add, int64_t i, int n, int c, int C) {
    const float v = __half2float(x[i]);
    return add ? rnd<half>(v + __half2float(add[(int64_t)n * C + c])) : v;
}

// A group's (mean, rstd) in float from its sums s and ss, written over the first half of its pair of sums.
__device__ __forceinline__ void sd_group_stat(double* sums, int ng, double s, double ss, double cnt, float eps) {
    const double mean = s / cnt;
    const double var = fmax(ss / cnt - mean * mean, 0.0);
    ((float2*)sums)[ng * 2] = make_float2((float)mean, (float)(1.0 / sqrt(var + (double)eps)));
}

// The last block to finish turns every group's sums into statistics; `done` counts the finished blocks.
__global__ void sd_group_sums_kernel(const half* x, const half* add, double* sums, unsigned* done, int64_t P, int C, int G, int splits,
                                     double cnt, float eps) {
    const int ng = blockIdx.x, split = blockIdx.y;
    const int n = ng / G, g = ng % G, per = C / G;
    const int64_t base = (int64_t)n * P * C + g * per;
    const int64_t p0 = P * split / splits, p1 = P * (split + 1) / splits;
    float s = 0.f, ss = 0.f;
    const int64_t count = (p1 - p0) * per;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        const int64_t p = p0 + i / per;
        const int c = (int)(i % per);
        const float v = sd_gn_in(x, add, base + p * C + c, n, g * per + c, C);
        s += v;
        ss += v * v;
    }
    __shared__ float red[33];
    __shared__ bool last;
    s = block_sum(s, red);
    ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        atomicAdd(&sums[ng * 2], (double)s);
        atomicAdd(&sums[ng * 2 + 1], (double)ss);
        __threadfence();
        last = atomicAdd(done, 1u) == gridDim.x * gridDim.y - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) {
        // the other blocks' atomics are in L2; read past L1
        const volatile double* vs = sums;
        sd_group_stat(sums, i, vs[i * 2], vs[i * 2 + 1], cnt, eps);
    }
}

__global__ void sd_group_apply_kernel(const half* x, const half* add, half* out, const float* w, const float* b, const double* sums,
                                      int64_t P, int C, int G, int silu, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % C);
    const int n = (int)(i / (P * C));
    const int ng = n * G + c / (C / G);
    const float2 st = ((const float2*)sums)[ng * 2];
    float y = rnd<half>((sd_gn_in(x, add, i, n, c, C) - st.x) * st.y * w[c] + b[c]);
    if (silu) y = silu_f(y);
    out[i] = __float2half_rn(y);
}

// sd_group_apply_kernel over 8 channels of a pixel per thread, for C a multiple of 8 and 16-byte aligned x, add and out
__global__ void sd_group_apply_v8_kernel(const half* x, const half* add, half* out, const float* w, const float* b, const double* sums,
                                         int64_t P, int C, int G, int silu, int64_t total8) {
    const int64_t i8 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i8 >= total8) return;
    const int64_t i0 = i8 * 8;
    const int c0 = (int)(i0 % C);
    const int n = (int)(i0 / (P * C));
    uint4 raw = *(const uint4*)(x + i0);
    const half* xv = (const half*)&raw;
    uint4 araw = make_uint4(0u, 0u, 0u, 0u);
    if (add) araw = *(const uint4*)(add + (int64_t)n * C + c0);
    const half* av = (const half*)&araw;
    uint4 oraw;
    half* ov = (half*)&oraw;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = c0 + j;
        const int ng = n * G + c / (C / G);
        const float2 st = ((const float2*)sums)[ng * 2];
        const float v = __half2float(xv[j]);
        const float in = add ? rnd<half>(v + __half2float(av[j])) : v;
        float y = rnd<half>((in - st.x) * st.y * w[c] + b[c]);
        if (silu) y = silu_f(y);
        ov[j] = __float2half_rn(y);
    }
    *(uint4*)(out + i0) = oraw;
}

// sums: 16 * N * G + 16 bytes of scratch. add (nullable) [N, C] is added to x first, rounded to half.
GPU_API int gpu_sd_group_norm(const void* x, const void* add, void* out, const float* w, const float* b, void* sums, int N, int64_t P,
                              int C, int G, float eps, int silu, void* stream) {
    if (C % G) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    cudaMemsetAsync(sums, 0, (size_t)N * G * 2 * sizeof(double) + 16, st);
    const int64_t want = P * (C / G) / 4096;
    const int splits = want < 1 ? 1 : want > 64 ? 64 : (int)want;
    unsigned* done = (unsigned*)((double*)sums + (size_t)N * G * 2);
    sd_group_sums_kernel<<<dim3(N * G, splits), 256, 0, st>>>((const half*)x, (const half*)add, (double*)sums, done, P, C, G, splits,
                                                             (double)P * (C / G), eps);
    const int64_t total = (int64_t)N * P * C;
    if (C % 8 == 0 && (uintptr_t)x % 16 == 0 && (uintptr_t)out % 16 == 0 && (uintptr_t)add % 16 == 0) {
        sd_group_apply_v8_kernel<<<(unsigned)((total / 8 + 255) / 256), 256, 0, st>>>((const half*)x, (const half*)add, (half*)out, w, b,
                                                                                     (const double*)sums, P, C, G, silu, total / 8);
    } else {
        sd_group_apply_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>((const half*)x, (const half*)add, (half*)out, w, b,
                                                                              (const double*)sums, P, C, G, silu, total);
    }
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
// The same copy 16 bytes at a time, for d, dp and the stride multiples of 8 and 16-byte aligned rows
__global__ void sd_pad_heads_v8_kernel(const uint4* src, int64_t stride8, uint4* dst, int H, int d8, int dp8, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int j = (int)(i % dp8);
    const int64_t lh = i / dp8;
    const int h = (int)(lh % H);
    const int64_t l = lh / H;
    dst[i] = j < d8 ? src[l * stride8 + (int64_t)h * d8 + j] : make_uint4(0u, 0u, 0u, 0u);
}
GPU_API int gpu_sd_pad_heads(const void* src, int64_t stride, void* dst, int64_t L, int H, int d, int dp, void* stream) {
    if (d % 8 == 0 && dp % 8 == 0 && stride % 8 == 0 && (uintptr_t)src % 16 == 0 && (uintptr_t)dst % 16 == 0) {
        const int64_t total = L * H * (dp / 8);
        sd_pad_heads_v8_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint4*)src, stride / 8, (uint4*)dst,
                                                                                                  H, d / 8, dp / 8, total);
        KCHECK();
        return 0;
    }
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

// sd_quant_rows of a row that another kernel would have written in half first: the row is computed as that kernel
// computes it, rounded to half, then quantized as sd_quant_rows quantizes it. Blocks of 256 threads, as both kernels use,
// so every sum runs in their order.
enum { SDQ_LN = 1, SDQ_GEGLU = 2, SDQ_UNPAD = 3, SDQ_GN = 4 };
struct SdQuantSrc {
    const half* x;
    int64_t x_stride;    // LN: the row stride
    const float* w;      // LN, GN: weight and bias
    const float* b;
    float eps;           // LN
    int heads, d, dp;    // UNPAD: heads of d values, each stored dp wide
    const double* sums;  // GN: the statistics gpu_sd_group_norm's sums pass leaves
    int groups;          // GN
    int64_t P;           // GN: pixels per image
};

template <int MODE>
__global__ void sd_quant_src_kernel(SdQuantSrc p, int8_t* q, float* scales, int K, int rotate) {
    extern __shared__ float row[];
    __shared__ float red[33];
    const int64_t m = blockIdx.x;
    if (MODE == SDQ_LN) {
        // sd_layer_norm_kernel
        const half* r = p.x + m * p.x_stride;
        float s = 0.f;
        for (int k = threadIdx.x; k < K; k += blockDim.x) s += ldf(r, k);
        const float mean = block_sum(s, red) / K;
        float ss = 0.f;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float d = ldf(r, k) - mean;
            ss += d * d;
        }
        const float rstd = rsqrtf(block_sum(ss, red) / K + p.eps);
        for (int k = threadIdx.x; k < K; k += blockDim.x) row[k] = __half2float(__float2half_rn((ldf(r, k) - mean) * rstd * p.w[k] + p.b[k]));
    } else if (MODE == SDQ_GEGLU) {
        // sd_geglu_kernel: [M, 2K] value then gate
        const half* r = p.x + m * 2 * K;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            const float a = __half2float(r[k]);
            const float g = rnd<half>(gelu_f(__half2float(r[K + k])));
            row[k] = __half2float(__float2half_rn(a * g));
        }
    } else if (MODE == SDQ_UNPAD) {
        const half* r = p.x + m * p.heads * p.dp;
        for (int k = threadIdx.x; k < K; k += blockDim.x) row[k] = __half2float(r[(k / p.d) * p.dp + k % p.d]);
    } else {
        // sd_group_apply_kernel without SiLU: row m is a pixel of image m / P
        const half* r = p.x + m * K;
        const int n = (int)(m / p.P);
        for (int c = threadIdx.x; c < K; c += blockDim.x) {
            const int ng = n * p.groups + c / (K / p.groups);
            const float2 st = ((const float2*)p.sums)[ng * 2];
            const float y = rnd<half>((__half2float(r[c]) - st.x) * st.y * p.w[c] + p.b[c]);
            row[c] = __half2float(__float2half_rn(y));
        }
    }
    __syncthreads();
    // sd_quant_rows_kernel from here
    if (rotate == 256) fht256_rows_smem(row, K);
    else if (rotate == 64) fht64_rows_smem(row, K);
    float mx = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(rnd<half>(row[k])));
    mx = block_max(mx, red);
    const float scale = fmaxf(mx, 1e-30f) / 127.f;
    if (threadIdx.x == 0) scales[m] = scale;
    const float scale_t = rnd<half>(scale) > 0.f ? rnd<half>(scale) : scale;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float t = rnd<half>(rnd<half>(row[k]) / scale_t);
        q[m * K + k] = (int8_t)fminf(127.f, fmaxf(-128.f, nearbyintf(t)));
    }
}

template <int MODE>
static int sd_quant_src(const SdQuantSrc& src, int8_t* q, float* scales, int M, int K, int rotate, cudaStream_t st) {
    if ((rotate != 0 && rotate != 64 && rotate != 256) || (rotate && K % rotate)) return -1;
    const size_t smem = (size_t)K * sizeof(float);
    if (smem > 48 * 1024) cudaFuncSetAttribute(sd_quant_src_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    sd_quant_src_kernel<MODE><<<M, 256, smem, st>>>(src, q, scales, K, rotate);
    KCHECK();
    return 0;
}

// gpu_sd_layer_norm (half) then gpu_sd_quant_rows
GPU_API int gpu_sd_ln_quant_rows(const void* x, int64_t x_stride, const float* w, const float* b, float eps, int8_t* q, float* scales,
                                 int M, int K, int rotate, void* stream) {
    SdQuantSrc p{};
    p.x = (const half*)x, p.x_stride = x_stride, p.w = w, p.b = b, p.eps = eps;
    return sd_quant_src<SDQ_LN>(p, q, scales, M, K, rotate, (cudaStream_t)stream);
}
// gpu_sd_geglu of in [M, 2I] then gpu_sd_quant_rows of the [M, I] result
GPU_API int gpu_sd_geglu_quant_rows(const void* in, int8_t* q, float* scales, int M, int I, int rotate, void* stream) {
    SdQuantSrc p{};
    p.x = (const half*)in;
    return sd_quant_src<SDQ_GEGLU>(p, q, scales, M, I, rotate, (cudaStream_t)stream);
}
// gpu_sd_unpad_heads of src [M, H, dp] then gpu_sd_quant_rows of the [M, H * d] result
GPU_API int gpu_sd_unpad_quant_rows(const void* src, int8_t* q, float* scales, int M, int H, int d, int dp, int rotate, void* stream) {
    SdQuantSrc p{};
    p.x = (const half*)src, p.heads = H, p.d = d, p.dp = dp;
    return sd_quant_src<SDQ_UNPAD>(p, q, scales, M, H * d, rotate, (cudaStream_t)stream);
}

// gpu_sd_group_norm without SiLU, then gpu_sd_quant_rows of the [N * P, C] result. sums: as gpu_sd_group_norm's.
GPU_API int gpu_sd_group_norm_quant_rows(const void* x, const float* w, const float* b, void* sums, int N, int64_t P, int C, int G,
                                         float eps, int8_t* q, float* scales, int rotate, void* stream) {
    if (C % G) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    cudaMemsetAsync(sums, 0, (size_t)N * G * 2 * sizeof(double) + 16, st);
    const int64_t want = P * (C / G) / 4096;
    const int splits = want < 1 ? 1 : want > 64 ? 64 : (int)want;
    unsigned* done = (unsigned*)((double*)sums + (size_t)N * G * 2);
    sd_group_sums_kernel<<<dim3(N * G, splits), 256, 0, st>>>((const half*)x, nullptr, (double*)sums, done, P, C, G, splits,
                                                             (double)P * (C / G), eps);
    SdQuantSrc p{};
    p.x = (const half*)x, p.w = w, p.b = b, p.sums = (const double*)sums, p.groups = G, p.P = P;
    return sd_quant_src<SDQ_GN>(p, q, scales, (int)(N * P), C, rotate, st);
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
