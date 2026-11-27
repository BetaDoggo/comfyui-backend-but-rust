//! Declarations of the C ABI exported by csrc/*.cu, plus checked wrappers.
#![allow(clippy::too_many_arguments)]

use crate::cuda::Stream;
use anyhow::{bail, Result};
use std::ffi::c_void;

pub type P = *const c_void;
pub type PM = *mut c_void;

pub const DT_F32: i32 = 0;
pub const DT_F16: i32 = 1;
pub const DT_BF16: i32 = 2;

pub const PRO_PLAIN: i32 = 0;
pub const PRO_RMS: i32 = 1;
pub const PRO_GATE: i32 = 2;
pub const PRO_SWIGLU: i32 = 3;
pub const PRO_LAYERNORM: i32 = 4;
pub const PRO_SILU: i32 = 5;
pub const PRO_GELU: i32 = 6;
pub const PRO_ADALN: i32 = 7;
pub const PRO_SITU: i32 = 8;
pub const PRO_LN_MOD: i32 = 9;
pub const PRO_LN_AFFINE: i32 = 10;
pub const PRO_GELU_TANH: i32 = 11;
pub const PRO_LN_AFFINE_SILU: i32 = 12;
/// `PRO_SITU` with hardware tanh and fast exponentials, `gpu_quant_rows` only (`--lossy`)
pub const PRO_SITU_FAST: i32 = 13;

extern "C" {
    pub fn gpu_blas_init(workspace: usize) -> i32;
    pub fn gpu_gemm(a: P, lda: i64, b: P, ldb: i64, d: PM, ldd: i64, bias: P, m: i64, n: i64, k: i64, transb: i32,
        in_dt: i32, out_dt: i32, alpha: f32, beta: f32, s: PM) -> i32;
    pub fn gpu_rotate_rows_f32(prologue: i32, x: P, x_stride: i64, up_offset: i64, norm_w: P, eps: f32, out: PM,
        m: i32, k: i32, s: PM) -> i32;
    pub fn gpu_dequant_i8(q: P, s_: P, out: PM, n: i64, k: i64, s: PM) -> i32;
    pub fn gpu_int8_gemm(a: P, w: P, xs: P, ws: P, bias: P, d: PM, m: i64, n: i64, k: i64, out_dt: i32, s: PM) -> i32;
    pub fn gpu_int8_gemm_residual(a: P, w: P, xs: P, ws: P, bias: P, rscale: P, resid: P, d: PM, m: i64, n: i64,
        k: i64, out_dt: i32, s: PM) -> i32;
    pub fn gpu_conv2d(x: P, w: P, bias: P, out: PM, n: i32, h: i32, w_: i32, c: i32, cout: i32, r: i32, s: PM) -> i32;
    /// `gpu_conv2d` with `resid` added to its output (rounded as `gpu_add_gated_bf16` rounds); `out` aliases neither input.
    pub fn gpu_conv2d_residual(x: P, w: P, bias: P, resid: P, out: PM, n: i32, h: i32, w_: i32, c: i32, cout: i32, r: i32, s: PM) -> i32;
    pub fn gpu_flash_attn(q: P, k: P, v: P, o: PM, lse: PM, b: i32, sq: i32, sk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, variant: i32, s: PM) -> i32;
    /// `gpu_flash_attn` for heads `d` wide, 128 or 64.
    pub fn gpu_flash_attn_d(q: P, k: P, v: P, o: PM, lse: PM, b: i32, sq: i32, sk: i32, h: i32, hk: i32, d: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, variant: i32, s: PM) -> i32;

    // block-sparse attention (csrc/sparse_attn.cu over comfy-kitchen's Sol-Attn)
    pub fn gpu_sol_attn_workspace(t: i32, h: i32, n_tok: i32) -> i64;
    pub fn gpu_sol_topk_workspace(t: i32, h: i32) -> i64;
    pub fn gpu_sol_topk_threshold(q: P, k: P, rs: i64, hs: i64, t: i32, h: i32, kk: i32, scale: f32, s0: i32, s1: i32,
        thr: PM, scratch: PM, s: PM) -> i32;
    pub fn gpu_sol_attn(q: P, k: P, v: P, out: PM, ws: PM, t: i32, h: i32, rs: i64, hs: i64, tau: f32, scale: f32, thr: P,
        s0: i32, s1: i32, q0: i32, q1: i32, n_tok: i32, s: PM) -> i32;
    // Sol³ sparse attention (csrc/sol3_attn.cu)
    pub fn gpu_sol3_workspace(t: i32, h: i32, hc: i32, int8: i32) -> i64;
    pub fn gpu_sol3_route_offsets(t: i32, h: i32, hc: i32, int8: i32, out: *mut i64) -> i32;
    pub fn gpu_sol3_attn(q: P, k: P, v: P, out: PM, ws: PM, t: i32, h: i32, hc: i32, rs: i64, hs: i64, ors: i64, prefix: i32,
        int8: i32, s: PM) -> i32;

    pub fn gpu_int8_attn_workspace(b: i32, h: i32, hk: i32, lq: i32, lk: i32) -> i64;
    pub fn gpu_int8_attn(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, s: PM) -> i32;
    /// `gpu_int8_attn` with q and k taken through `gpu_qk_norm_rope` inside the quantizer, bit-identical to running it
    /// first. q and k in memory are left as they are. bf16, head dim 128.
    pub fn gpu_int8_attn_norm_rope(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, qw: P, kw: P, eps: f32, rope: P, rope_mode: i32, s: PM) -> i32;
    /// `gpu_int8_attn` over `lk` keys plus `sink` more whose keys and values are zero, counted in the denominator only
    /// (`--lossy`). A non-null `qw` normalizes q first, as `gpu_int8_attn_norm_rope` with a null `kw`.
    pub fn gpu_int8_attn_sink(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, sink: f32, qw: P, eps: f32, s: PM) -> i32;
    pub fn gpu_quant_rows(dtype: i32, prologue: i32, x: P, x_stride: i64, g: P, g_stride: i64, up_offset: i64,
        norm_w: P, eps: f32, mod_scale: P, mod_shift: P, q: PM, scales: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_prologue_rows(dtype: i32, prologue: i32, x: P, x_stride: i64, g: P, g_stride: i64, up_offset: i64,
        norm_w: P, eps: f32, mod_scale: P, mod_shift: P, out: PM, out_stride: i64, m: i32, k: i32, s: PM) -> i32;
    pub fn gpu_qk_norm_rope(dtype: i32, head_dim: i32, buf: PM, row_stride: i64, q_off: i32, k_off: i32, hq: i32,
        hk: i32, qw: P, kw: P, eps: f32, rope: P, rope_mode: i32, l: i32, s: PM) -> i32;
    pub fn gpu_attn_small(dtype: i32, q: P, q_stride: i64, k: P, v: P, kv_stride: i64, hq: i32, hk: i32, d: i32,
        lq: i32, lk: i32, out: PM, out_stride: i64, scale: f32, causal: i32, s: PM) -> i32;
    /// `gpu_attn_small` through its one-block-per-row kernel at any length, to check the tiled one
    pub fn gpu_attn_small_ref(dtype: i32, q: P, q_stride: i64, k: P, v: P, kv_stride: i64, hq: i32, hk: i32, d: i32,
        lq: i32, lk: i32, out: PM, out_stride: i64, scale: f32, causal: i32, s: PM) -> i32;
    pub fn gpu_embed_dequant(q: P, s_: P, out: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_gelu_tanh_bf16(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_add_gated_bf16(x: PM, x_stride: i64, y: P, y_stride: i64, gate: P, rows: i32, cols: i32, s: PM) -> i32;
    pub fn gpu_add_vec_bf16(out: PM, a: P, b: P, b_is_f32: i32, n: i32, s: PM) -> i32;
    /// `gpu_add_vec_bf16` with bf16 `b` over `rows` rows of `a` into rows of `out`, strides in elements
    pub fn gpu_add_vec_rows_bf16(out: PM, out_stride: i64, a: P, a_stride: i64, b: P, rows: i32, n: i32, s: PM) -> i32;
    pub fn gpu_steer_record(x: P, rows: i32, cols: i32, sum: PM, norm: PM, s: PM) -> i32;
    pub fn gpu_steer_add(x: PM, rows: i32, cols: i32, dir: P, amount: f32, preserve: i32, s: PM) -> i32;
    pub fn gpu_zero_f32(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_colstats(x: P, rows: i32, cols: i32, stats: PM, s: PM) -> i32;
    pub fn gpu_adain(x: PM, rows: i32, cols: i32, tstats: P, trows: f32, rstats: P, rrows: f32, lambda: f32, s: PM) -> i32;
    pub fn gpu_steer_axpy(x: PM, xs: i64, a: P, b: P, ys: i64, rows: i32, cols: i32, alpha: f32, s: PM) -> i32;
    pub fn gpu_f32_to_bf16(inp: P, out: PM, n: i64, s: PM) -> i32;
    pub fn gpu_bf16_to_f32(inp: P, out: PM, n: i64, s: PM) -> i32;
    pub fn gpu_randn_threads() -> i32;
    pub fn gpu_lincomb(out: PM, n: i64, nx: i32, x: *const P, c: *const f32, s: PM) -> i32;
    pub fn gpu_unpatch_denoised(tok: P, tok_stride: i64, lat: P, den: PM, c: i32, h: i32, w: i32, p: i32, sigma: f32,
        s: PM) -> i32;
    pub fn gpu_layer_project(x: P, w: P, out: PM, l: i32, nl: i32, d: i32, s: PM) -> i32;
    pub fn gpu_patchify(lat: P, tok: PM, c: i32, h: i32, w: i32, p: i32, s: PM) -> i32;
    pub fn gpu_unpatch_euler(tok: P, tok_stride: i64, lat: PM, c: i32, h: i32, w: i32, p: i32, sigma: f32,
        sigma_next: f32, denoised: PM, s: PM) -> i32;
    pub fn gpu_vae_latent_in(lat: P, mean: P, std: P, out: PM, c: i32, hw: i32, s: PM) -> i32;
    pub fn gpu_l2norm_silu(x: P, out: PM, gamma: P, p: i64, c: i32, silu: i32, s: PM) -> i32;
    pub fn gpu_upsample2x(x: P, out: PM, h: i32, w: i32, c: i32, s: PM) -> i32;
    pub fn gpu_softmax_rows(s_: P, p: PM, rows: i32, n: i32, scale: f32, s: PM) -> i32;
    pub fn gpu_to_rgb8(x: P, cstride: i32, out: PM, p: i64, s: PM) -> i32;
    pub fn gpu_adaln_weight(w: P, sc: P, out: PM, n: i32, s: PM) -> i32;
    pub fn gpu_delta_conv(x: P, w: P, out: PM, l: i32, c: i32, s: PM) -> i32;
    pub fn gpu_delta_rule(mix: PM, a: P, b: P, dt: P, alog: P, out: PM, l: i32, hv: i32, hk: i32, d: i32, s: PM) -> i32;
    pub fn gpu_mul_silu(x: PM, z: P, n: i64, s: PM) -> i32;
    pub fn gpu_rope_rows(x: PM, stride: i64, rope: P, l: i32, h: i32, d: i32, s: PM) -> i32;
    pub fn gpu_gelu_f32(x: PM, n: i64, approximate: i32, s: PM) -> i32;
    pub fn gpu_take_odd(x: P, out: PM, h: i32, w: i32, c: i32, s: PM) -> i32;
    pub fn gpu_partial_rope(x: PM, stride: i64, w: P, eps: f32, rope: P, l: i32, h: i32, d: i32, pairs: i32, s: PM) -> i32;
    pub fn gpu_flux2_latent_in(lat: P, mean: P, std: P, out: PM, c: i32, h: i32, w: i32, s: PM) -> i32;
    pub fn gpu_convrot_dequant(dtype: i32, q: P, s_: P, out: PM, n: i64, k: i32, s: PM) -> i32;
    pub fn gpu_convrot_requant(dtype: i32, x: P, q: PM, s_: PM, n: i64, k: i32, seed: u64, s: PM) -> i32;
    /// SenseNova's q/k: two 64-wide RMS norms per 128-wide head, then RoPE over 64, 32 and 32 values from `rope`
    /// ([l][64][(cos, sin)]). In place, bf16 or f32.
    pub fn gpu_sn_qk(dtype: i32, buf: PM, row_stride: i64, q_off: i32, k_off: i32, hq: i32, hk: i32, qn: P, qn_hw: P, kn: P, kn_hw: P,
        eps: f32, rope: P, l: i32, s: PM) -> i32;
    pub fn gpu_sn_gelu_rope2d(x: PM, gh: i32, gw: i32, c: i32, freqs: P, s: PM) -> i32;
    pub fn gpu_sn_space_to_depth2(x: P, out: PM, h: i32, w: i32, c: i32, s: PM) -> i32;
    pub fn gpu_sn_pixel_shuffle(inp: P, out: PM, h: i32, w: i32, c: i32, r: i32, s: PM) -> i32;
    pub fn gpu_sn_gelu(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_sn_denoise(y: P, x: P, den: PM, h: i32, w: i32, sigma: f32, denom: f32, s: PM) -> i32;
    pub fn gpu_lora_add(dtype: i32, w: PM, d: P, scale: f32, n: i64, s: PM) -> i32;
    pub fn gpu_w6_decode(q: P, s_rel: P, out: PM, n: i64, k: i32, g: i32, s: PM) -> i32;
    pub fn gpu_add_gated_rows_bf16(x: PM, y: P, gates: P, sel: P, rows: i32, cols: i32, s: PM) -> i32;
    pub fn gpu_dwd_unpatch(out: P, h: i32, w: i32, bf: i32, clip: PM, frames: i32, f0: i32, mean: *const f32, sd: *const f32, s: PM) -> i32;
    pub fn gpu_int8_attn_f16(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, d: i32, q_bs: i64, q_rs: i64,
        q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64, o_hs: i64, scale: f32, s: PM) -> i32;
    // TripoSplat / Idol (csrc/triposplat.cu)
    pub fn gpu_ts_add_f16(x: PM, x_stride: i64, y: P, y_stride: i64, gate: P, rows: i64, cols: i32, s: PM) -> i32;
    pub fn gpu_ts_repo_qk(buf: PM, row_stride: i64, k_off: i32, delta: P, delta_stride: i64, freqs: P, qw: P, kw: P, eps: f32, l: i32,
        h: i32, do_k: i32, s: PM) -> i32;
    pub fn gpu_ts_silu_mul_f16(x: P, x_stride: i64, out: PM, out_stride: i64, rows: i64, cols: i32, s: PM) -> i32;
    pub fn gpu_int8_attn_workspace_d(b: i32, h: i32, hk: i32, lq: i32, lk: i32, d: i32) -> i64;
    pub fn gpu_int8_attn_d(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, d: i32, q_bs: i64, q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64,
        v_hs: i64, o_bs: i64, o_rs: i64, o_hs: i64, scale: f32, s: PM) -> i32;
    pub fn gpu_w6_requant(w: P, q: PM, s_rel: PM, s_channel: PM, n: i64, k: i32, seed: u64, s: PM) -> i32;
    pub fn gpu_nvfp4_dequant(q: P, scales: P, scale2: f32, pre: P, out: PM, n: i64, k: i32, s: PM) -> i32;
    pub fn gpu_nvfp4_gemm(a: P, lda: i64, m: i32, k: i32, q: *const P, sc: *const P, pre: *const P, scale2: *const f32, n: *const i32, count: i32, out: PM, ldd: i64,
        beta: f32, ws: PM, ws_bytes: i64, s: PM) -> i32;
    pub fn gpu_rope_bf16(x: PM, stride: i64, hs: i64, w: P, eps: f32, rope: P, l: i32, h: i32, d: i32, pairs: i32, s: PM) -> i32;
    pub fn gpu_conv1d(x: P, w: P, bias: P, out: PM, b: i32, ci: i32, co: i32, t: i32, k: i32, dil: i32, pad: i32, s: PM) -> i32;
    pub fn gpu_conv_t1d(x: P, w: P, bias: P, out: PM, b: i32, ci: i32, co: i32, t: i32, k: i32, u: i32, pad: i32, s: PM) -> i32;
    pub fn gpu_aa_snake(x: PM, alpha: P, beta: P, up: P, down: P, tmp: PM, b: i32, c: i32, t: i32, s: PM) -> i32;
    pub fn gpu_group_norm(x: P, out: PM, w: P, b: P, stats: PM, p: i64, c: i32, groups: i32, eps: f32, silu: i32, s: PM) -> i32;
    pub fn gpu_h3_im2col3(inp: P, h: i32, w: i32, c: i32, stride: i32, y0: i32, rows: i32, out: PM, s: PM) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn gpu_h3_im2col3t(inp: P, h: i32, w: i32, c: i32, stride: i32, tstride: i32, to: i32, y0: i32, rows: i32, out: PM, s: PM) -> i32;
    pub fn gpu_h3_group_norm_f16(x: P, out: PM, w: P, b: P, p: i64, c: i32, g: i32, eps: f32, silu: i32, scratch: PM, s: PM) -> i32;

    // Trellis 2 / Pixal3D (csrc/trellis.cu and the LayerNorm prologues of kernels.cu)
    pub fn gpu_quant_rows_b(dtype: i32, prologue: i32, x: P, x_stride: i64, norm_w: P, norm_b: P, eps: f32, mod_scale: P,
        mod_shift: P, q: PM, scales: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_prologue_rows_b(dtype: i32, prologue: i32, x: P, x_stride: i64, norm_w: P, norm_b: P, eps: f32, mod_scale: P,
        mod_shift: P, out: PM, out_stride: i64, m: i32, k: i32, s: PM) -> i32;
    pub fn gpu_qk_rms_heads(buf: PM, row_stride: i64, q_off: i32, k_off: i32, hq: i32, hk: i32, qw: P, kw: P, eps: f32, rope: P,
        l: i32, s: PM) -> i32;
    pub fn gpu_voxel_hash(coords: P, n: i32, x: i32, y: i32, z: i32, keys: PM, vals: PM, cap: i32, s: PM) -> i32;
    pub fn gpu_voxel_neighbors(coords: P, n: i32, x: i32, y: i32, z: i32, keys: P, vals: P, cap: i32, out: PM, s: PM) -> i32;
    pub fn gpu_voxel_lookup(q: P, n: i32, x: i32, y: i32, z: i32, keys: P, vals: P, cap: i32, out: PM, s: PM) -> i32;
    pub fn gpu_sparse_gather(x: P, c: i32, nbr: P, r0: i32, rows: i32, out: PM, s: PM) -> i32;
    pub fn gpu_gather_rows(x: P, bytes: i64, idx: P, n: i32, out: PM, s: PM) -> i32;
    pub fn gpu_scatter_rows(x: P, bytes: i64, idx: P, n: i32, out: PM, s: PM) -> i32;
    pub fn gpu_skip_add_f16(h: PM, skip: P, m: i64, cs: i32, rep: i32, s: PM) -> i32;
    pub fn gpu_add_f16(x: PM, y: P, n: i64, s: PM) -> i32;
    pub fn gpu_pixel_shuffle3d_f16(inp: P, out: PM, r: i32, c: i32, s: PM) -> i32;
    pub fn gpu_cast(in_dt: i32, out_dt: i32, inp: P, out: PM, n: i64, s: PM) -> i32;
    pub fn gpu_softmax_f32(x: PM, rows: i32, n: i32, scale: f32, s: PM) -> i32;
    pub fn gpu_scaled_add_f32(x: PM, y: P, sc: P, rows: i64, cols: i32, s: PM) -> i32;
    pub fn gpu_im2col3_reflect(inp: P, h: i32, w: i32, c: i32, y0: i32, rows: i32, out: PM, s: PM) -> i32;
    pub fn gpu_round_f16(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_group_norm_nhwc_f32(x: PM, p: i64, c: i32, g: i32, w: P, b: P, eps: f32, silu: i32, scratch: PM, s: PM) -> i32;
    pub fn gpu_avg_pool_nhwc_f32(inp: P, h: i32, w: i32, c: i32, f: i32, out: PM, s: PM) -> i32;
    pub fn gpu_naf_rope(x: PM, h: i32, w: i32, c: i32, periods: P, s: PM) -> i32;
    pub fn gpu_naf_attn(q: P, k: P, v: P, out: PM, hh: i32, ww: i32, h: i32, w: i32, dv: i32, s: PM) -> i32;
    pub fn gpu_grid_sample_bf16(map_dt: i32, map: P, h: i32, w: i32, c: i32, uv: P, n: i32, out: PM, out_stride: i64, col0: i32,
        s: PM) -> i32;

    // BiRefNet (csrc/birefnet.cu)
    pub fn gpu_im2col_zero(inp: P, h: i32, w: i32, c: i32, k: i32, pad: i32, y0: i32, rows: i32, out: PM, s: PM) -> i32;
    pub fn gpu_patchify_nhwc(inp: P, h: i32, w: i32, c: i32, p: i32, out: PM, s: PM) -> i32;
    pub fn gpu_deform_im2col(inp: P, om: P, h: i32, w: i32, c: i32, k: i32, pad: i32, y0: i32, rows: i32, out: PM, s: PM) -> i32;
    pub fn gpu_window_gather(x: P, h: i32, w: i32, c: i32, hp: i32, wp: i32, shift: i32, out: PM, s: PM) -> i32;
    pub fn gpu_window_scatter_add(x: PM, win_rows: P, h: i32, w: i32, c: i32, hp: i32, wp: i32, shift: i32, s: PM) -> i32;
    pub fn gpu_window_attn(qkv: P, table: P, out: PM, nw: i32, c: i32, heads: i32, hp: i32, wp: i32, shift: i32, s: PM) -> i32;
    pub fn gpu_patch_merge(x: P, h: i32, w: i32, c: i32, out: PM, s: PM) -> i32;
    pub fn gpu_split_patches(x: P, hf: i32, c: i32, p: i32, out: PM, s: PM) -> i32;
    pub fn gpu_resize_bilinear_ac(inp: P, h: i32, w: i32, c: i32, out: PM, ldo: i64, oh: i32, ow: i32, s: PM) -> i32;
    pub fn gpu_act_rows(x: PM, ld: i64, c: i32, rows: i64, mode: i32, s: PM) -> i32;
    pub fn gpu_mul_rows(x: PM, sc: P, c: i32, rows: i64, s: PM) -> i32;
    pub fn gpu_add_f32(x: PM, y: P, n: i64, s: PM) -> i32;
    pub fn gpu_bcast_rows(v: P, c: i32, out: PM, ld: i64, rows: i64, s: PM) -> i32;

    // Qwen Image 2.1 (csrc/qwen21.cu)
    pub fn gpu_dup_up(x: P, out: PM, h: i32, w: i32, cin: i32, cout: i32, ft: i32, s: PM) -> i32;
    pub fn gpu_avg_down(x: P, out: PM, h: i32, w: i32, cin: i32, cout: i32, ft: i32, fs: i32, s: PM) -> i32;
    pub fn gpu_to_rgba8(x: P, cstride: i32, out: PM, p: i64, s: PM) -> i32;
    pub fn gpu_attn_prefix(q: P, q_stride: i64, k: P, v: P, kv_stride: i64, h: i32, lq: i32, offset: i32, out: PM, out_stride: i64,
        scale: f32, s: PM) -> i32;
    pub fn gpu_tanh_bf16(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_scale_bf16(x: PM, n: i64, scale: f32, s: PM) -> i32;
    pub fn gpu_attn_merge(groups: *const MergeGroups, n: i32, out: PM, b: i32, sq: i32, h: i32, d: i32, s: PM) -> i32;
    /// Krea 2's attention token weights: `x[rows[r] * stride + c] *= factors[r]` in bf16 (`rows` i32, `factors` f32).
    pub fn gpu_scale_rows_bf16(x: PM, stride: i64, rows: P, factors: P, n: i32, cols: i32, s: PM) -> i32;
    /// `gpu_int8_attn` with `bias` (f32 [lk], every image's) added to each key's logits.
    pub fn gpu_int8_attn_key_bias_workspace(b: i32, h: i32, hk: i32, lq: i32, lk: i32) -> i64;
    pub fn gpu_int8_attn_key_bias(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, bias: P, s: PM) -> i32;

    // SeedVR2 (csrc/seedvr.cu)
    pub fn gpu_seedvr_patch_in(x: P, cond: P, tok: PM, h: i32, w: i32, ld: i32, s: PM) -> i32;
    pub fn gpu_seedvr_unwindow(x: P, c: i32, src: P, offs: P, rows: i32, out: PM, s: PM) -> i32;
    pub fn gpu_seedvr_stack3(a: P, b: P, c: P, out: PM, px: i64, ch: i32, s: PM) -> i32;
    pub fn gpu_pixel_shuffle2_bf16(x: P, out: PM, h: i32, w: i32, c: i32, s: PM) -> i32;

    // RelayVSR (csrc/relayvsr.cu)
    /// Sliding-window attention, see [`WinArgs`]; `dtype` bf16 or f16, heads `hd` 64 or 128 wide.
    pub fn gpu_win_attn(dtype: i32, hd: i32, args: *const WinArgs, scale: f32, s: PM) -> i32;
    pub fn gpu_relay_rms(dtype: i32, x: P, x_rs: i64, out: PM, out_rs: i64, w: P, add: P, eps: f32, rows: i64, k: i32, groups: i32, s: PM) -> i32;
    pub fn gpu_relay_rope(dtype: i32, x: P, x_rs: i64, out: PM, out_rs: i64, heads: i32, hd: i32, rows: i64, h: i32, w: i32, tt: P, th: P, tw: P,
        pt: i32, ph: i32, s: PM) -> i32;
    pub fn gpu_relay_kv_quant(src: P, src_rs: i64, q: PM, sc: PM, rows: i64, heads: i32, s: PM) -> i32;
    pub fn gpu_relay_kv_dequant(q: P, sc: P, dst: PM, dst_rs: i64, rows: i64, heads: i32, s: PM) -> i32;
    pub fn gpu_relay_lq_unshuffle(src: P, h: i32, w: i32, c: i32, hp: i32, wp: i32, up: i32, r: i32, out: PM, s: PM) -> i32;
    pub fn gpu_relay_im2col3(dtype: i32, inp: P, h: i32, w: i32, c: i32, stride: i32, out: PM, ld: i32, s: PM) -> i32;
    pub fn gpu_relay_silu(dtype: i32, x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_relay_patchify(lat: P, c: i32, f: i32, f0: i32, nf: i32, lh: i32, lw: i32, out: PM, s: PM) -> i32;
    pub fn gpu_relay_embed_add(pe: P, lq: P, out: PM, n: i64, s: PM) -> i32;
    pub fn gpu_relay_unpatch_denoise(tok: P, ld: i32, x: P, den: PM, c: i32, f: i32, f0: i32, nf: i32, lh: i32, lw: i32, sigma: f32, s: PM) -> i32;
    pub fn gpu_relay_plane_rows(dtype: i32, lat: P, c: i32, f: i32, plane: i32, h: i32, w: i32, out: PM, s: PM) -> i32;
    pub fn gpu_relay_lr_frame(dtype: i32, src: P, h: i32, w: i32, c: i32, hp: i32, wp: i32, out_f: PM, out_t: PM, s: PM) -> i32;
    pub fn gpu_relay_flash_out(dtype: i32, res: P, lr: P, hp: i32, wp: i32, up: i32, detail: f32, out: PM, oh: i32, ow: i32, lw: i32, s: PM) -> i32;

    // SD 1.x and SDXL (csrc/sd.cu): fp16 UNet, float32 CLIP
    pub fn gpu_conv2d_f16(x: P, w: P, bias: P, out: PM, n: i32, h: i32, w_: i32, c: i32, cout: i32, r: i32, stride: i32, s: PM) -> i32;
    /// `gpu_conv2d_f16` at stride 1 with `resid` added as `gpu_sd_add` adds it; `out` aliases neither input.
    pub fn gpu_conv2d_f16_residual(x: P, w: P, bias: P, resid: P, out: PM, n: i32, h: i32, w_: i32, c: i32, cout: i32, r: i32, s: PM) -> i32;
    /// `add` (nullable, [n, c]) is added to `x` first, rounded to fp16; `sums` holds 16 * n * g + 16 bytes.
    pub fn gpu_sd_group_norm(x: P, add: P, out: PM, w: P, b: P, sums: PM, n: i32, p: i64, c: i32, g: i32, eps: f32, silu: i32, s: PM)
        -> i32;
    pub fn gpu_sd_layer_norm(dtype: i32, x: P, x_stride: i64, out: PM, out_stride: i64, w: P, b: P, m: i32, k: i32, eps: f32, s: PM) -> i32;
    pub fn gpu_sd_geglu(inp: P, out: PM, m: i64, i: i32, s: PM) -> i32;
    pub fn gpu_sd_add(x: PM, y: P, n: i64, s: PM) -> i32;
    pub fn gpu_sd_upsample(x: P, out: PM, n: i32, h: i32, w: i32, oh: i32, ow: i32, c: i32, s: PM) -> i32;
    pub fn gpu_sd_add_emb(x: PM, emb: P, n: i32, p: i64, c: i32, s: PM) -> i32;
    pub fn gpu_sd_silu(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_sd_clip_act(x: PM, n: i64, mode: i32, s: PM) -> i32;
    pub fn gpu_sd_clip_embed(tok: P, pos: P, ids: P, out: PM, l: i32, d: i32, s: PM) -> i32;
    pub fn gpu_sd_unet_in(lat: P, out: PM, n: i32, c: i32, hw: i64, cs: i32, sigmas: P, s: PM) -> i32;
    pub fn gpu_sd_unet_out(y: P, x: P, out: PM, n: i32, c: i32, hw: i64, cs: i32, sigmas: P, v: i32, s: PM) -> i32;
    pub fn gpu_sd_pad_heads(src: P, stride: i64, dst: PM, l: i64, h: i32, d: i32, dp: i32, s: PM) -> i32;
    pub fn gpu_sd_unpad_heads(src: P, dst: PM, l: i64, h: i32, d: i32, dp: i32, s: PM) -> i32;
    pub fn gpu_sd_softmax(scores: P, p: PM, rows: i32, n: i32, scale: f32, s: PM) -> i32;
    pub fn gpu_sd_vae_in(lat: P, out: PM, c: i32, hw: i64, cs: i32, scale: f32, s: PM) -> i32;
    pub fn gpu_sd_quant_rows(x: P, x_stride: i64, q: PM, scales: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    /// `gpu_sd_quant_rows` of what `gpu_sd_layer_norm`, `gpu_sd_geglu`, `gpu_sd_unpad_heads` or `gpu_sd_group_norm` (without SiLU)
    /// would write, without writing it
    pub fn gpu_sd_ln_quant_rows(x: P, x_stride: i64, w: P, b: P, eps: f32, q: PM, scales: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_sd_geglu_quant_rows(inp: P, q: PM, scales: PM, m: i32, i: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_sd_unpad_quant_rows(src: P, q: PM, scales: PM, m: i32, h: i32, d: i32, dp: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_sd_group_norm_quant_rows(x: P, w: P, b: P, sums: PM, n: i32, p: i64, c: i32, g: i32, eps: f32, q: PM, scales: PM, rotate: i32,
        s: PM) -> i32;
}

/// Key planes one query plane of [`gpu_win_attn`] attends, at most.
pub const WIN_PLANES: usize = 4;

/// [`gpu_win_attn`]'s arguments (`WinArgs` in csrc/relayvsr.cu). Query plane `t` attends key plane `t + tk - tq` and the
/// `tw - 1` before it, each at `kb + p * k_ps` with one window (`n == 0`), or the `n` planes given, each with its own
/// window and strides, for one query plane. Strides are in elements; `scale_log2` is filled in by the launcher.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WinArgs {
    pub q: P,
    pub o: PM,
    pub q_rs: i64,
    pub q_ps: i64,
    pub o_rs: i64,
    pub o_ps: i64,
    pub k: [P; WIN_PLANES],
    pub v: [P; WIN_PLANES],
    pub krs: [i64; WIN_PLANES],
    pub vrs: [i64; WIN_PLANES],
    pub wh: [i32; WIN_PLANES],
    pub ww: [i32; WIN_PLANES],
    pub n: i32,
    pub kb: P,
    pub vb: P,
    pub k_ps: i64,
    pub v_ps: i64,
    pub uwh: i32,
    pub uww: i32,
    pub tk: i32,
    pub tq: i32,
    pub tw: i32,
    pub k_rs: i64,
    pub v_rs: i64,
    pub h: i32,
    pub w: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub clamp: i32,
    pub scale_log2: f32,
}

impl WinArgs {
    /// No planes yet, for a `h` x `w` grid.
    pub fn new(q: P, q_rs: i64, o: PM, o_rs: i64, h: usize, w: usize, heads: usize, kv_heads: usize, clamp: bool) -> Self {
        WinArgs {
            q, o, q_rs, q_ps: 0, o_rs, o_ps: 0,
            k: [null(); WIN_PLANES], v: [null(); WIN_PLANES], krs: [0; WIN_PLANES], vrs: [0; WIN_PLANES], wh: [0; WIN_PLANES], ww: [0; WIN_PLANES], n: 0,
            kb: null(), vb: null(), k_ps: 0, v_ps: 0, uwh: 0, uww: 0, tk: 0, tq: 1, tw: 1, k_rs: 0, v_rs: 0,
            h: h as i32, w: w as i32, heads: heads as i32, kv_heads: kv_heads as i32, clamp: clamp as i32, scale_log2: 0.0,
        }
    }

    /// One more explicit key plane.
    pub fn plane(mut self, k: P, krs: i64, v: P, vrs: i64, window: (usize, usize)) -> Self {
        let i = self.n as usize;
        assert!(i < WIN_PLANES, "at most {WIN_PLANES} key planes");
        self.k[i] = k;
        self.v[i] = v;
        self.krs[i] = krs;
        self.vrs[i] = vrs;
        self.wh[i] = window.0 as i32;
        self.ww[i] = window.1 as i32;
        self.n += 1;
        self
    }
}

/// Most key groups [`gpu_attn_merge`] takes.
pub const MERGE_GROUPS: usize = 8;

/// `gpu_attn_merge`'s groups: each one's attention output (bf16 [b, sq, h, d]), its log-sum-exp (f32 [b, h, sq]) and
/// the bias on its keys' logits.
#[repr(C)]
pub struct MergeGroups {
    pub o: [P; MERGE_GROUPS],
    pub lse: [P; MERGE_GROUPS],
    pub bias: [f32; MERGE_GROUPS],
}

/// Turn a launcher return code into an error naming the call.
pub fn ok(code: i32, what: &str) -> Result<()> {
    if code != 0 {
        if code > 0 {
            bail!("{what} failed: {}", crate::cuda::err_str(code));
        }
        bail!("{what} failed with code {code}");
    }
    Ok(())
}

macro_rules! k {
    ($f:ident ( $($a:expr),* $(,)? )) => {
        $crate::ffi::ok(unsafe { $crate::ffi::$f($($a),*) }, stringify!($f))
    };
}
pub(crate) use k;

pub fn s(st: &Stream) -> PM {
    st.raw()
}

pub fn null() -> P {
    std::ptr::null()
}

/// Byte offset into a device pointer.
pub fn off(p: PM, bytes: usize) -> PM {
    unsafe { (p as *mut u8).add(bytes) as PM }
}
