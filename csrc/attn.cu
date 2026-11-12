// FlashAttention-2 forward (bf16, head dim 128 or 64, no dropout/mask) wrapped behind a C ABI.
// Only the kernel variants this pipeline needs are instantiated.
#include "common.cuh"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

using namespace FLASH_NAMESPACE;

namespace gpu_fa {

template <typename Traits, bool EvenMN>
void launch_variant(Flash_fwd_params& params, cudaStream_t stream) {
    constexpr size_t smem = Traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    // <Traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>
    auto kernel = &flash_fwd_kernel<Traits, false, false, false, false, EvenMN, true, false, false>;
    static bool configured = false;
    if (!configured) {
        if (smem >= 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    kernel<<<grid, Traits::kNThreads, smem, stream>>>(params);
}

template <typename Traits>
void launch_traits(Flash_fwd_params& params, cudaStream_t stream) {
    const bool even = params.seqlen_k % Traits::kBlockN == 0 && params.seqlen_q % Traits::kBlockM == 0;
    if (even) launch_variant<Traits, true>(params, stream);
    else launch_variant<Traits, false>(params, stream);
}

using TraitsA = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using TraitsB = Flash_fwd_kernel_traits<128, 128, 32, 4, false, false, cutlass::bfloat16_t>;
// head dim 64, the tiles FlashAttention's own launcher picks for it
using Traits64 = Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t>;

int round_up(int x, int m) { return (x + m - 1) / m * m; }

}  // namespace gpu_fa
using namespace gpu_fa;

// q/k/v/o: bf16, layout [b, seq, heads, d] with arbitrary batch/row/head strides (elements), d 128 or 64.
// lse: float scratch [b, h, seqlen_q]. variant 0: 128x64 tiles, 1: 128x32 tiles (head dim 128 only).
GPU_API int gpu_flash_attn_d(const void* q, const void* k, const void* v, void* o, float* lse, int b, int sq, int sk,
                               int h, int hk, int d, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                               int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                               int64_t o_hs, float scale, int variant, void* stream) {
    if (h % hk != 0) return -1;
    if (d != 128 && d != 64) return -2;
    Flash_fwd_params p;
    memset(&p, 0, sizeof(p));
    p.is_bf16 = true;
    p.q_ptr = const_cast<void*>(q);
    p.k_ptr = const_cast<void*>(k);
    p.v_ptr = const_cast<void*>(v);
    p.o_ptr = o;
    p.q_batch_stride = q_bs; p.q_row_stride = q_rs; p.q_head_stride = q_hs;
    p.k_batch_stride = k_bs; p.k_row_stride = k_rs; p.k_head_stride = k_hs;
    p.v_batch_stride = v_bs; p.v_row_stride = v_rs; p.v_head_stride = v_hs;
    p.o_batch_stride = o_bs; p.o_row_stride = o_rs; p.o_head_stride = o_hs;
    p.h = h;
    p.h_k = hk;
    p.h_h_k_ratio = h / hk;
    p.softmax_lse_ptr = lse;
    p.b = b;
    p.seqlen_q = sq;
    p.seqlen_k = sk;
    p.d = d;
    p.d_rounded = d;
    p.seqlen_q_rounded = round_up(sq, 128);
    p.seqlen_k_rounded = round_up(sk, 128);
    p.scale_softmax = scale;
    p.scale_softmax_log2 = scale * 1.4426950408889634f;
    p.p_dropout = 1.f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = scale;
    p.window_size_left = -1;
    p.window_size_right = -1;
    p.softcap = 0.f;
    p.is_causal = false;
    p.is_seqlens_k_cumulative = true;
    p.num_splits = 1;
    cudaStream_t st = (cudaStream_t)stream;
    if (d == 64) launch_traits<Traits64>(p, st);
    else if (variant == 1) launch_traits<TraitsB>(p, st);
    else launch_traits<TraitsA>(p, st);
    KCHECK();
    return 0;
}

GPU_API int gpu_flash_attn(const void* q, const void* k, const void* v, void* o, float* lse, int b, int sq, int sk,
                             int h, int hk, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                             int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                             int64_t o_hs, float scale, int variant, void* stream) {
    return gpu_flash_attn_d(q, k, v, o, lse, b, sq, sk, h, hk, 128, q_bs, q_rs, q_hs, k_bs, k_rs, k_hs, v_bs, v_rs, v_hs, o_bs, o_rs,
                            o_hs, scale, variant, stream);
}
