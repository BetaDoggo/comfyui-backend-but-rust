// Optional INT8 attention (comfy-kitchen's SageAttention-derived kernel, vendor/sage).
// Q/K are block-Hadamard rotated and quantized per thread-tile, V per channel; softmax state is
// fp32. Faster than FlashAttention-2 bf16 on GeForce cards, at slightly lower precision.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

extern "C" void launch_quant_qk_per_thread_int8(const void* q, void* q_int8, void* q_scale, const void* k,
    void* k_int8, void* k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ, int WARPQ, int BLKK,
    int WARPK, int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, void* anchor_indices, cudaStream_t stream);
extern "C" void launch_quant_v_int8_kernel(const void* v, void* out, void* scale, int B, int H, int N, int D,
    int padded_N, int64_t sb, int64_t sh, int64_t sn, int input_dtype_code, cudaStream_t stream);
extern "C" void launch_sage_attn_kernel(const void* q, const void* k, const void* v, void* o, const void* q_scale,
    const void* k_scale, const void* v_scale, const void* mask, int64_t mask_stride_b, int64_t mask_stride_h,
    int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code, int cta_k, int batch_size, int qo_len,
    int kv_len, int num_qo_heads, int num_kv_heads, int head_dim, int64_t stride_bz_q, int stride_seq_q,
    int64_t stride_h_q, int64_t stride_bz_k, int stride_seq_k, int64_t stride_h_k, int64_t stride_bz_v,
    int64_t stride_h_v, int stride_d_v, int64_t stride_bz_o, int stride_seq_o, int64_t stride_h_o, float sm_scale,
    int output_dtype_code, cudaStream_t stream, const float* mask_tile_bias);

static int cta_k_for(int lk, int d) { return lk > 1024 && d != 64 ? 128 : 64; }

// Scratch bytes needed for b batches, h/hk heads, lengths lq/lk, head dim 128.
extern "C" int64_t gpu_int8_attn_workspace_d(int b, int h, int hk, int lq, int lk, int head_dim) {
    const int ck = cta_k_for(lk, head_dim);
    const int64_t plk = (int64_t)(lk + ck - 1) / ck * ck;
    const int64_t d = head_dim;
    int64_t n = 0;
    auto add = [&](int64_t bytes) { n += (bytes + 255) / 256 * 256; };
    add((int64_t)b * h * lq * d);                                  // q int8
    add((int64_t)b * hk * lk * d);                                 // k int8
    add((int64_t)b * h * ((lq + 127) / 128) * 32 * 4);            // q scale
    add((int64_t)b * hk * ((lk + ck - 1) / ck) * 4 * 4);          // k scale
    add((int64_t)b * hk * d * plk);                                // v int8 (transposed, padded)
    add((int64_t)b * hk * d * 4);                                  // v scale
    add((int64_t)b * hk * 4);                                      // anchor indices
    return n;
}

extern "C" int64_t gpu_int8_attn_workspace(int b, int h, int hk, int lq, int lk) {
    return gpu_int8_attn_workspace_d(b, h, hk, lq, lk, 128);
}

// q/k/v: bf16 [b, seq, heads, D] with element strides; o: bf16, same convention.
extern "C" int gpu_int8_attn_d(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk,
                              int h, int hk, int D, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                              int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                              int64_t o_hs, float scale, void* stream) {
    try {
        cudaStream_t st = (cudaStream_t)stream;
        const int ck = cta_k_for(lk, D);
        const int plk = (lk + ck - 1) / ck * ck;
        char* p = (char*)ws;
        auto take = [&](int64_t bytes) { char* r = p; p += (bytes + 255) / 256 * 256; return r; };
        int8_t* qi = (int8_t*)take((int64_t)b * h * lq * D);
        int8_t* ki = (int8_t*)take((int64_t)b * hk * lk * D);
        float* qs = (float*)take((int64_t)b * h * ((lq + 127) / 128) * 32 * 4);
        float* ks = (float*)take((int64_t)b * hk * ((lk + ck - 1) / ck) * 4 * 4);
        int8_t* vi = (int8_t*)take((int64_t)b * hk * D * plk);
        float* vs = (float*)take((int64_t)b * hk * D * 4);
        void* anchors = take((int64_t)b * hk * 4);
        // inputs are [B, H, L, D] views: (batch, head, seq) strides
        launch_quant_qk_per_thread_int8(q, qi, qs, k, ki, ks, b, h, lq, hk, lk, D, 128, 32, ck, ck, q_bs, q_hs, q_rs,
                                        k_bs, k_hs, k_rs, 2, anchors, st);
        launch_quant_v_int8_kernel(v, vi, vs, b, hk, lk, D, plk, v_bs, v_hs, v_rs, 2, st);
        const int64_t qh = (int64_t)lq * D, kh = (int64_t)lk * D, vh = (int64_t)D * plk;
        launch_sage_attn_kernel(qi, ki, vi, o, qs, ks, vs, nullptr, 0, 0, 0, 0, 0, ck, b, lq, lk, h, hk, D, h * qh, D,
                                qh, hk * kh, D, kh, hk * vh, vh, plk, o_bs, (int)o_rs, o_hs, scale, 2, st, nullptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "int8 attention: %s\n", e.what());
        return -400;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -401;
}

extern "C" int gpu_int8_attn(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk,
                              int h, int hk, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                              int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                              int64_t o_hs, float scale, void* stream) {
    return gpu_int8_attn_d(q, k, v, o, ws, b, lq, lk, h, hk, 128, q_bs, q_rs, q_hs, k_bs, k_rs, k_hs, v_bs, v_rs, v_hs, o_bs, o_rs, o_hs, scale, stream);
}
