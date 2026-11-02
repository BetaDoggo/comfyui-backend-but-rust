//! Declarations of the C ABI exported by csrc/*.cu, plus checked wrappers.
#![allow(clippy::too_many_arguments)]

use crate::cuda::Stream;
use anyhow::{bail, Result};
use std::ffi::c_void;

pub type P = *const c_void;
pub type PM = *mut c_void;

pub const DT_F32: i32 = 0;
pub const DT_BF16: i32 = 2;

pub const PRO_PLAIN: i32 = 0;
pub const PRO_RMS: i32 = 1;
pub const PRO_GATE: i32 = 2;
pub const PRO_SWIGLU: i32 = 3;
pub const PRO_LAYERNORM: i32 = 4;
pub const PRO_SILU: i32 = 5;
pub const PRO_GELU: i32 = 6;

extern "C" {
    pub fn krea_blas_init(workspace: usize) -> i32;
    pub fn krea_gemm(a: P, lda: i64, b: P, ldb: i64, d: PM, ldd: i64, bias: P, m: i64, n: i64, k: i64, transb: i32,
        in_dt: i32, out_dt: i32, alpha: f32, beta: f32, s: PM) -> i32;
    pub fn krea_rotate_rows_f32(prologue: i32, x: P, x_stride: i64, up_offset: i64, norm_w: P, eps: f32, out: PM,
        m: i32, k: i32, s: PM) -> i32;
    pub fn krea_dequant_i8(q: P, s_: P, out: PM, n: i64, k: i64, s: PM) -> i32;
    pub fn krea_int8_gemm(a: P, w: P, xs: P, ws: P, bias: P, d: PM, m: i64, n: i64, k: i64, out_dt: i32, s: PM) -> i32;
    pub fn krea_int8_gemm_residual(a: P, w: P, xs: P, ws: P, bias: P, rscale: P, resid: P, d: PM, m: i64, n: i64,
        k: i64, out_dt: i32, s: PM) -> i32;
    pub fn krea_conv2d(x: P, w: P, bias: P, out: PM, n: i32, h: i32, w_: i32, c: i32, cout: i32, r: i32, s: PM) -> i32;
    pub fn krea_flash_attn(q: P, k: P, v: P, o: PM, lse: PM, b: i32, sq: i32, sk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, variant: i32, s: PM) -> i32;

    pub fn krea_int8_attn_workspace(b: i32, h: i32, hk: i32, lq: i32, lk: i32) -> i64;
    pub fn krea_int8_attn(q: P, k: P, v: P, o: PM, ws: PM, b: i32, lq: i32, lk: i32, h: i32, hk: i32, q_bs: i64,
        q_rs: i64, q_hs: i64, k_bs: i64, k_rs: i64, k_hs: i64, v_bs: i64, v_rs: i64, v_hs: i64, o_bs: i64, o_rs: i64,
        o_hs: i64, scale: f32, s: PM) -> i32;
    pub fn krea_quant_rows(dtype: i32, prologue: i32, x: P, x_stride: i64, g: P, g_stride: i64, up_offset: i64,
        norm_w: P, eps: f32, mod_scale: P, mod_shift: P, q: PM, scales: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn krea_prologue_rows(dtype: i32, prologue: i32, x: P, x_stride: i64, g: P, g_stride: i64, up_offset: i64,
        norm_w: P, eps: f32, mod_scale: P, mod_shift: P, out: PM, out_stride: i64, m: i32, k: i32, s: PM) -> i32;
    pub fn krea_qk_norm_rope(dtype: i32, head_dim: i32, buf: PM, row_stride: i64, q_off: i32, k_off: i32, hq: i32,
        hk: i32, qw: P, kw: P, eps: f32, rope: P, rope_mode: i32, l: i32, s: PM) -> i32;
    pub fn krea_attn_small(dtype: i32, q: P, q_stride: i64, k: P, v: P, kv_stride: i64, hq: i32, hk: i32, d: i32,
        lq: i32, lk: i32, out: PM, out_stride: i64, scale: f32, causal: i32, s: PM) -> i32;
    pub fn krea_embed_dequant(q: P, s_: P, out: PM, m: i32, k: i32, rotate: i32, s: PM) -> i32;
    pub fn krea_gelu_tanh_bf16(x: PM, n: i64, s: PM) -> i32;
    pub fn krea_add_gated_bf16(x: PM, x_stride: i64, y: P, y_stride: i64, gate: P, rows: i32, cols: i32, s: PM) -> i32;
    pub fn krea_add_vec_bf16(out: PM, a: P, b: P, b_is_f32: i32, n: i32, s: PM) -> i32;
    pub fn krea_f32_to_bf16(inp: P, out: PM, n: i64, s: PM) -> i32;
    pub fn krea_bf16_to_f32(inp: P, out: PM, n: i64, s: PM) -> i32;
    pub fn krea_randn_threads() -> i32;
    pub fn krea_lincomb(out: PM, n: i64, nx: i32, x: *const P, c: *const f32, s: PM) -> i32;
    pub fn krea_unpatch_denoised(tok: P, tok_stride: i64, lat: P, den: PM, c: i32, h: i32, w: i32, p: i32, sigma: f32,
        s: PM) -> i32;
    pub fn krea_layer_project(x: P, w: P, out: PM, l: i32, nl: i32, d: i32, s: PM) -> i32;
    pub fn krea_patchify(lat: P, tok: PM, c: i32, h: i32, w: i32, p: i32, s: PM) -> i32;
    pub fn krea_unpatch_euler(tok: P, tok_stride: i64, lat: PM, c: i32, h: i32, w: i32, p: i32, sigma: f32,
        sigma_next: f32, denoised: PM, s: PM) -> i32;
    pub fn krea_vae_latent_in(lat: P, mean: P, std: P, out: PM, c: i32, hw: i32, s: PM) -> i32;
    pub fn krea_l2norm_silu(x: P, out: PM, gamma: P, p: i64, c: i32, silu: i32, s: PM) -> i32;
    pub fn krea_upsample2x(x: P, out: PM, h: i32, w: i32, c: i32, s: PM) -> i32;
    pub fn krea_softmax_rows(s_: P, p: PM, rows: i32, n: i32, scale: f32, s: PM) -> i32;
    pub fn krea_to_rgb8(x: P, cstride: i32, out: PM, p: i64, s: PM) -> i32;
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
