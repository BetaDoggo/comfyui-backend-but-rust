// Dense GEMMs: cuBLASLt (bf16 inputs, fp32 accumulate) and CUTLASS implicit-GEMM conv2d (NHWC bf16).
#include "common.cuh"
#include <cublasLt.h>
#include <mutex>
#include <unordered_map>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------------------------
// cuBLASLt
// ---------------------------------------------------------------------------------------------
namespace {

cublasLtHandle_t g_lt = nullptr;
void* g_ws = nullptr;
size_t g_ws_size = 0;
std::mutex g_mu;

struct PlanKey {
    int64_t m, n, k, lda, ldb, ldd;
    int transb, in_dt, out_dt, has_bias, beta_nz, pad;
    bool operator==(const PlanKey& o) const { return memcmp(this, &o, sizeof(PlanKey)) == 0; }
};
struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        const uint64_t* p = reinterpret_cast<const uint64_t*>(&k);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(PlanKey) / 8; ++i) h = (h ^ p[i]) * 1099511628211ull;
        return (size_t)h;
    }
};
struct Plan {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
};
std::unordered_map<PlanKey, Plan, PlanKeyHash> g_plans;

}  // namespace

GPU_API int gpu_blas_init(size_t workspace_bytes) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_lt) return 0;
    if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
    if (cudaMalloc(&g_ws, workspace_bytes) != cudaSuccess) return -2;
    g_ws_size = workspace_bytes;
    return 0;
}

// Row-major D[M,N] = alpha * A[M,K] . op(B) (+ bias[N]) (+ beta * D)
//   transb = 1: B is [N,K] (a Linear weight), transb = 0: B is [K,N].
// A, B bf16, f16 or f32 (in_dt, fp32 is computed without TF32); D likewise (out_dt); bias in the output dtype.
GPU_API int gpu_gemm(const void* A, int64_t lda, const void* B, int64_t ldb, void* D, int64_t ldd, const void* bias,
                       int64_t M, int64_t N, int64_t K, int transb, int in_dt, int out_dt, float alpha, float beta,
                       void* stream) {
    if (!g_lt) return -100;
    // Column-major view: D^T[N,M] = op(B)^T-ish[N,K] x A^T[K,M]
    PlanKey key{N, M, K, lda, ldb, ldd, transb, in_dt, out_dt, bias != nullptr, beta != 0.f, 0};
    Plan plan;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_plans.find(key);
        if (it == g_plans.end()) {
            auto cuda_dt = [](int d) { return d == DT_F32 ? CUDA_R_32F : d == DT_F16 ? CUDA_R_16F : CUDA_R_16BF; };
            const cudaDataType_t dt = cuda_dt(out_dt);
            const cudaDataType_t it = cuda_dt(in_dt);
            cublasLtMatmulDescCreate(&plan.desc, in_dt == DT_F32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F);
            cublasOperation_t ta = transb ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
            cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
            if (bias) {
                cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
                cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
                cudaDataType_t bdt = dt;
                cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));
            }
            // "A" of the col-major problem is B; "B" is A.
            if (transb) cublasLtMatrixLayoutCreate(&plan.a, it, K, N, ldb);
            else cublasLtMatrixLayoutCreate(&plan.a, it, N, K, ldb);
            cublasLtMatrixLayoutCreate(&plan.b, it, K, M, lda);
            cublasLtMatrixLayoutCreate(&plan.c, dt, N, M, ldd);
            cublasLtMatmulPreference_t pref;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws_size, sizeof(g_ws_size));
            cublasLtMatmulHeuristicResult_t res[4];
            int got = 0;
            const cublasStatus_t s =
                cublasLtMatmulAlgoGetHeuristic(g_lt, plan.desc, plan.a, plan.b, plan.c, plan.c, pref, 4, res, &got);
            cublasLtMatmulPreferenceDestroy(pref);
            if (s != CUBLAS_STATUS_SUCCESS || got == 0) return -101;
            plan.algo = res[0].algo;
            g_plans.emplace(key, plan);
        } else {
            plan = it->second;
        }
    }
    if (bias) {
        const void* bp = bias;
        cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp));
    }
    const cublasStatus_t s = cublasLtMatmul(g_lt, plan.desc, &alpha, B, plan.a, A, plan.b, &beta, D, plan.c, D, plan.c,
                                            &plan.algo, g_ws, g_ws_size, (cudaStream_t)stream);
    return s == CUBLAS_STATUS_SUCCESS ? 0 : -(int)s - 1000;
}

// ---------------------------------------------------------------------------------------------
// INT8 GEMM with fused dequant epilogue (vendored comfy-kitchen CUTLASS kernels)
// ---------------------------------------------------------------------------------------------
extern "C" bool launch_cutlass_int8_dequant(const void* A, const void* B, const void* xs, const void* ws,
                                            const void* bias, void* D, int64_t M, int64_t N, int64_t K,
                                            int out_dtype_code, cudaStream_t stream);
extern "C" bool launch_cutlass_int8_dequant_residual(const void* A, const void* B, const void* xs, const void* ws,
                                                     const void* bias, const void* rscale, const void* resid, void* D,
                                                     int64_t M, int64_t N, int64_t K, int out_dtype_code,
                                                     cudaStream_t stream);

// D[M,N] = (A[M,K] . W[N,K]^T) * xs[m] * ws[n] + bias[n]
GPU_API int gpu_int8_gemm(const int8_t* A, const int8_t* W, const float* xs, const float* ws, const void* bias,
                            void* D, int64_t M, int64_t N, int64_t K, int out_dt, void* stream) {
    if (!launch_cutlass_int8_dequant(A, W, xs, ws, bias, D, M, N, K, out_dt, (cudaStream_t)stream)) return -200;
    KCHECK();
    return 0;
}

// D[M,N] = resid[M,N] + rscale[n] * ((A . W^T) * xs[m] * ws[n] + bias[n])   (D may alias resid)
GPU_API int gpu_int8_gemm_residual(const int8_t* A, const int8_t* W, const float* xs, const float* ws,
                                     const void* bias, const void* rscale, const void* resid, void* D, int64_t M,
                                     int64_t N, int64_t K, int out_dt, void* stream) {
    if (!launch_cutlass_int8_dequant_residual(A, W, xs, ws, bias, rscale, resid, D, M, N, K, out_dt,
                                              (cudaStream_t)stream))
        return -201;
    KCHECK();
    return 0;
}

// ---------------------------------------------------------------------------------------------
// conv2d NHWC bf16 (stride 1, 'same' padding), fp32 accumulate, bias broadcast via zero-stride C.
// ---------------------------------------------------------------------------------------------
namespace {

template <int TBM, int TBN, int TBK, int WM, int WN, int Stages>
struct ConvFprop {
    using ElementA = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;
    using Layout = cutlass::layout::TensorNHWC;
    using Epilogue = cutlass::epilogue::thread::LinearCombination<ElementOut, 8, float, float>;
    using Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementA, Layout, ElementA, Layout, ElementOut, Layout, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<TBM, TBN, TBK>, cutlass::gemm::GemmShape<WM, WN, TBK>,
        cutlass::gemm::GemmShape<16, 8, 16>, Epilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages, cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kUnity>::Kernel;
    using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    static int run(const void* x, const void* w, const void* bias, void* out, int N, int H, int W, int C, int Cout,
                   int R, cudaStream_t stream) {
        const int pad = R / 2;
        cutlass::conv::Conv2dProblemSize problem(cutlass::Tensor4DCoord(N, H, W, C),
                                                 cutlass::Tensor4DCoord(Cout, R, R, C),
                                                 cutlass::Tensor4DCoord(pad, pad, pad, pad), cutlass::MatrixCoord(1, 1),
                                                 cutlass::MatrixCoord(1, 1), cutlass::Tensor4DCoord(N, H, W, Cout),
                                                 cutlass::conv::Mode::kCrossCorrelation, 1);
        using TensorRef = cutlass::TensorRef<cutlass::bfloat16_t, Layout>;
        TensorRef a((cutlass::bfloat16_t*)x, Layout::packed({N, H, W, C}));
        TensorRef b((cutlass::bfloat16_t*)w, Layout::packed({Cout, R, R, C}));
        TensorRef d((cutlass::bfloat16_t*)out, Layout::packed({N, H, W, Cout}));
        TensorRef c((cutlass::bfloat16_t*)bias, Layout(Layout::Stride(0)));
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

// x [N,H,W,C], w [Cout,R,R,C], bias [Cout] (nullable), out [N,H,W,Cout]; C and Cout multiples of 8.
GPU_API int gpu_conv2d(const void* x, const void* w, const void* bias, void* out, int N, int H, int W, int C,
                         int Cout, int R, void* stream) {
    if (C % 8 || Cout % 8) return -310;
    cudaStream_t st = (cudaStream_t)stream;
    int r;
    if (Cout >= 128) r = ConvFprop<128, 128, 32, 64, 64, 4>::run(x, w, bias, out, N, H, W, C, Cout, R, st);
    else r = ConvFprop<128, 64, 32, 64, 32, 4>::run(x, w, bias, out, N, H, W, C, Cout, R, st);
    if (r) return r;
    KCHECK();
    return 0;
}
