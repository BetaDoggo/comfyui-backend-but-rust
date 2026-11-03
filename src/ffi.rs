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
    pub fn gpu_flash_attn(q: P, k: P, v: P, o: PM, lse: PM, b: i32, sq: i32, sk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, variant: i32, s: PM) -> i32;

    pub fn gpu_int8_attn_workspace(b: i32, h: i32, hk: i32, lq: i32, lk: i32) -> i64;
    pub fn gpu_int8_attn(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, s: PM) -> i32;
    pub fn gpu_quant_rows(dtype: i32, prologue: i32, x: P, x_stride: i64, g: P, g_stride: i64, up_offset: i64,
        norm_w: P, eps: f32, mod_scale: P, mod_shift: P, q: PM, scales: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_prologue_rows(dtype: i32, prologue: i32, x: P, x_stride: i64, g: P, g_stride: i64, up_offset: i64,
        norm_w: P, eps: f32, mod_scale: P, mod_shift: P, out: PM, out_stride: i64, m: i32, k: i32, s: PM) -> i32;
    pub fn gpu_qk_norm_rope(dtype: i32, head_dim: i32, buf: PM, row_stride: i64, q_off: i32, k_off: i32, hq: i32,
        hk: i32, qw: P, kw: P, eps: f32, rope: P, rope_mode: i32, l: i32, s: PM) -> i32;
    pub fn gpu_attn_small(dtype: i32, q: P, q_stride: i64, k: P, v: P, kv_stride: i64, hq: i32, hk: i32, d: i32,
        lq: i32, lk: i32, out: PM, out_stride: i64, scale: f32, causal: i32, s: PM) -> i32;
    pub fn gpu_embed_dequant(q: P, s_: P, out: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn gpu_gelu_tanh_bf16(x: PM, n: i64, s: PM) -> i32;
    pub fn gpu_add_gated_bf16(x: PM, x_stride: i64, y: P, y_stride: i64, gate: P, rows: i32, cols: i32, s: PM) -> i32;
    pub fn gpu_add_vec_bf16(out: PM, a: P, b: P, b_is_f32: i32, n: i32, s: PM) -> i32;
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
    pub fn gpu_lora_add(dtype: i32, w: PM, d: P, scale: f32, n: i64, s: PM) -> i32;
    pub fn gpu_w6_decode(q: P, s_rel: P, out: PM, n: i64, k: i32, g: i32, s: PM) -> i32;
    pub fn gpu_int8_attn_workspace_d(b: i32, h: i32, hk: i32, lq: i32, lk: i32, d: i32) -> i64;
    pub fn gpu_int8_attn_d(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, d: i32, q_bs: i64, q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64,
        v_hs: i64, o_bs: i64, o_rs: i64, o_hs: i64, scale: f32, s: PM) -> i32;
    pub fn gpu_w6_requant(w: P, q: PM, s_rel: PM, s_channel: PM, n: i64, k: i32, seed: u64, s: PM) -> i32;
    pub fn gpu_nvfp4_dequant(q: P, scales: P, scale2: f32, pre: P, out: PM, n: i64, k: i32, s: PM) -> i32;
    pub fn gpu_rope_bf16(x: PM, stride: i64, hs: i64, w: P, eps: f32, rope: P, l: i32, h: i32, d: i32, pairs: i32, s: PM) -> i32;
    pub fn gpu_conv1d(x: P, w: P, bias: P, out: PM, b: i32, ci: i32, co: i32, t: i32, k: i32, dil: i32, pad: i32, s: PM) -> i32;
    pub fn gpu_conv_t1d(x: P, w: P, bias: P, out: PM, b: i32, ci: i32, co: i32, t: i32, k: i32, u: i32, pad: i32, s: PM) -> i32;
    pub fn gpu_aa_snake(x: PM, alpha: P, beta: P, up: P, down: P, tmp: PM, b: i32, c: i32, t: i32, s: PM) -> i32;
    pub fn gpu_group_norm(x: P, out: PM, w: P, b: P, stats: PM, p: i64, c: i32, groups: i32, eps: f32, silu: i32, s: PM) -> i32;

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
