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
extern "C" void launch_quant_qk_norm_rope_int8(const void* q, void* q_int8, void* q_scale, const void* k, void* k_int8,
    void* k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int BLKK, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n, void* anchor_indices,
    const float* qw, const float* kw, const float* rope, float eps, int rope_mode, cudaStream_t stream);
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
// The quantizer's query warp: 16 rows at head dim 256, 32 otherwise; each warp keeps 8 scales.
static int warp_q(int d) { return d == 256 ? 16 : 32; }
static int q_scales(int64_t d) { return 128 / warp_q((int)d) * 8; }

// Scratch bytes needed for b batches, h/hk heads, lengths lq/lk, head dim 128.
extern "C" int64_t gpu_int8_attn_workspace_d(int b, int h, int hk, int lq, int lk, int head_dim) {
    const int ck = cta_k_for(lk, head_dim);
    const int64_t plk = (int64_t)(lk + ck - 1) / ck * ck;
    const int64_t d = head_dim;
    int64_t n = 0;
    auto add = [&](int64_t bytes) { n += (bytes + 255) / 256 * 256; };
    add((int64_t)b * h * lq * d);                                  // q int8
    add((int64_t)b * hk * lk * d);                                 // k int8
    add((int64_t)b * h * ((lq + 127) / 128) * q_scales(d) * 4);   // q scale
    add((int64_t)b * hk * ((lk + ck - 1) / ck) * 4 * 4);          // k scale
    add((int64_t)b * hk * d * plk);                                // v int8 (transposed, padded)
    add((int64_t)b * hk * d * 4);                                  // v scale
    add((int64_t)b * hk * 4);                                      // anchor indices
    return n;
}

extern "C" int64_t gpu_int8_attn_workspace(int b, int h, int hk, int lq, int lk) {
    return gpu_int8_attn_workspace_d(b, h, hk, lq, lk, 128);
}

// Per-head RMSNorm and RoPE for q and k, applied on the way into the quantizer (gpu_qk_norm_rope's arguments).
struct QkNorm {
    const float* qw;
    const float* kw;
    const float* rope;
    float eps;
    int rope_mode;
};

static int int8_attn(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk, int h, int hk,
                     int D, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs, int64_t k_hs, int64_t v_bs,
                     int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs, int64_t o_hs, float scale, int dt, void* stream,
                     const QkNorm* norm = nullptr);

// q/k/v: bf16 [b, seq, heads, D] with element strides; o: bf16, same convention.
extern "C" int gpu_int8_attn_d(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk,
                              int h, int hk, int D, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                              int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                              int64_t o_hs, float scale, void* stream) {
    return int8_attn(q, k, v, o, ws, b, lq, lk, h, hk, D, q_bs, q_rs, q_hs, k_bs, k_rs, k_hs, v_bs, v_rs, v_hs, o_bs, o_rs, o_hs,
                     scale, 2, stream);
}

// gpu_int8_attn_d with q/k/v/o in fp16 (TripoSplat).
extern "C" int gpu_int8_attn_f16(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk,
                                int h, int hk, int D, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                                int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                                int64_t o_hs, float scale, void* stream) {
    return int8_attn(q, k, v, o, ws, b, lq, lk, h, hk, D, q_bs, q_rs, q_hs, k_bs, k_rs, k_hs, v_bs, v_rs, v_hs, o_bs, o_rs, o_hs,
                     scale, 1, stream);
}

// dt: the sage dtype code of q/k/v and o, 1 fp16 or 2 bf16
static int int8_attn(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk, int h, int hk,
                     int D, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs, int64_t k_hs, int64_t v_bs,
                     int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs, int64_t o_hs, float scale, int dt, void* stream,
                     const QkNorm* norm) {
    try {
        cudaStream_t st = (cudaStream_t)stream;
        const int ck = cta_k_for(lk, D);
        const int plk = (lk + ck - 1) / ck * ck;
        char* p = (char*)ws;
        auto take = [&](int64_t bytes) { char* r = p; p += (bytes + 255) / 256 * 256; return r; };
        int8_t* qi = (int8_t*)take((int64_t)b * h * lq * D);
        int8_t* ki = (int8_t*)take((int64_t)b * hk * lk * D);
        float* qs = (float*)take((int64_t)b * h * ((lq + 127) / 128) * q_scales(D) * 4);
        float* ks = (float*)take((int64_t)b * hk * ((lk + ck - 1) / ck) * 4 * 4);
        int8_t* vi = (int8_t*)take((int64_t)b * hk * D * plk);
        float* vs = (float*)take((int64_t)b * hk * D * 4);
        void* anchors = take((int64_t)b * hk * 4);
        // inputs are [B, H, L, D] views: (batch, head, seq) strides
        if (norm) {
            if (D != 128 || dt != 2) throw std::runtime_error("fused q/k norm needs bf16 head dim 128");
            launch_quant_qk_norm_rope_int8(q, qi, qs, k, ki, ks, b, h, lq, hk, lk, ck, q_bs, q_hs, q_rs, k_bs, k_hs, k_rs,
                                           anchors, norm->qw, norm->kw, norm->rope, norm->eps, norm->rope_mode, st);
        } else {
            launch_quant_qk_per_thread_int8(q, qi, qs, k, ki, ks, b, h, lq, hk, lk, D, 128, warp_q(D), ck, ck, q_bs, q_hs,
                                            q_rs, k_bs, k_hs, k_rs, dt, anchors, st);
        }
        launch_quant_v_int8_kernel(v, vi, vs, b, hk, lk, D, plk, v_bs, v_hs, v_rs, dt, st);
        const int64_t qh = (int64_t)lq * D, kh = (int64_t)lk * D, vh = (int64_t)D * plk;
        launch_sage_attn_kernel(qi, ki, vi, o, qs, ks, vs, nullptr, 0, 0, 0, 0, 0, ck, b, lq, lk, h, hk, D, h * qh, D,
                                qh, hk * kh, D, kh, hk * vh, vh, plk, o_bs, (int)o_rs, o_hs, scale, dt, st, nullptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "int8 attention: %s\n", e.what());
        return -400;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -401;
}

// gpu_int8_attn with q and k taken through gpu_qk_norm_rope(qw, kw, eps, rope, rope_mode) first, in the quantizer. The
// result is bit-identical to running gpu_qk_norm_rope and then gpu_int8_attn, and q and k in memory are left unnormalized.
// Head dim 128, bf16.
extern "C" int gpu_int8_attn_norm_rope(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq,
                                      int lk, int h, int hk, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs,
                                      int64_t k_rs, int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs,
                                      int64_t o_rs, int64_t o_hs, float scale, const float* qw, const float* kw,
                                      float eps, const float* rope, int rope_mode, void* stream) {
    const QkNorm norm{qw, kw, rope, eps, rope_mode};
    return int8_attn(q, k, v, o, ws, b, lq, lk, h, hk, 128, q_bs, q_rs, q_hs, k_bs, k_rs, k_hs, v_bs, v_rs, v_hs, o_bs, o_rs,
                     o_hs, scale, 2, stream, &norm);
}

extern "C" void launch_sage_prepare_key_mask(const void* mask, float* packed, int batch, int heads, int length, int64_t stride_b,
    int64_t stride_h, int64_t stride_k, int dtype_code, cudaStream_t stream);

// sage's prepared key mask row for lk keys: each key's bias, then one value per 128-key tile
static int64_t key_mask_width(int lk) {
    const int64_t tiles = (lk + 127) / 128;
    return (tiles * 129 + 3) / 4 * 4;
}

extern "C" int64_t gpu_int8_attn_key_bias_workspace(int b, int h, int hk, int lq, int lk) {
    return gpu_int8_attn_workspace_d(b, h, hk, lq, lk, 128) + (key_mask_width(lk) * 4 + 255) / 256 * 256;
}

// gpu_int8_attn with bias (f32 [lk], the same for every batch item, query and head) added to each key's logits, as an
// additive attention mask. Long sequences take sage's prepared key mask, short ones its per-key mask.
extern "C" int gpu_int8_attn_key_bias(const void* q, const void* k, const void* v, void* o, void* ws, int b, int lq, int lk,
                                     int h, int hk, int64_t q_bs, int64_t q_rs, int64_t q_hs, int64_t k_bs, int64_t k_rs,
                                     int64_t k_hs, int64_t v_bs, int64_t v_rs, int64_t v_hs, int64_t o_bs, int64_t o_rs,
                                     int64_t o_hs, float scale, const float* bias, void* stream) {
    const int D = 128, dt = 2;
    try {
        cudaStream_t st = (cudaStream_t)stream;
        const int ck = cta_k_for(lk, D);
        const int plk = (lk + ck - 1) / ck * ck;
        char* p = (char*)ws;
        auto take = [&](int64_t bytes) { char* r = p; p += (bytes + 255) / 256 * 256; return r; };
        int8_t* qi = (int8_t*)take((int64_t)b * h * lq * D);
        int8_t* ki = (int8_t*)take((int64_t)b * hk * lk * D);
        float* qs = (float*)take((int64_t)b * h * ((lq + 127) / 128) * q_scales(D) * 4);
        float* ks = (float*)take((int64_t)b * hk * ((lk + ck - 1) / ck) * 4 * 4);
        int8_t* vi = (int8_t*)take((int64_t)b * hk * D * plk);
        float* vs = (float*)take((int64_t)b * hk * D * 4);
        void* anchors = take((int64_t)b * hk * 4);
        float* packed = (float*)take(key_mask_width(lk) * 4);
        launch_quant_qk_per_thread_int8(q, qi, qs, k, ki, ks, b, h, lq, hk, lk, D, 128, warp_q(D), ck, ck, q_bs, q_hs, q_rs,
                                        k_bs, k_hs, k_rs, dt, anchors, st);
        launch_quant_v_int8_kernel(v, vi, vs, b, hk, lk, D, plk, v_bs, v_hs, v_rs, dt, st);
        const int64_t qh = (int64_t)lq * D, kh = (int64_t)lk * D, vh = (int64_t)D * plk;
        // the mask's (batch, head, query, key) strides: one row for everything
        const void* mask = bias;
        const float* tile_bias = nullptr;
        if (ck == 128) {
            launch_sage_prepare_key_mask(bias, packed, 1, 1, lk, 0, 0, 1, 0, st);
            mask = packed;
            tile_bias = packed + (lk + 127) / 128 * 128;
        }
        launch_sage_attn_kernel(qi, ki, vi, o, qs, ks, vs, mask, 0, 0, 0, 1, 0, ck, b, lq, lk, h, hk, D, h * qh, D,
                                qh, hk * kh, D, kh, hk * vh, vh, plk, o_bs, (int)o_rs, o_hs, scale, dt, st, tile_bias);
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
