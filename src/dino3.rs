//! DINOv3 image encoders, in f32 like ComfyUI runs them: Pixal3D's ViT-L/16 with the NAF feature upsampler bundled
//! in the same file (f16 in ComfyUI, emulated by rounding to f16 where it materializes a tensor), and TripoSplat's
//! ViT-H/16+ (SwiGLU MLP), whose bf16 weights ComfyUI holds in f16.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, DT_F16, DT_F32, PM, PRO_LAYERNORM, PRO_LN_AFFINE, PRO_SWIGLU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};

/// ViT-L's width (the NAF's value channels).
const D: usize = 1024;
const HD: usize = 64;
const PATCH: usize = 16;

/// The shape of one DINOv3 checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cfg {
    d: usize,
    heads: usize,
    mlp: usize,
    layers: usize,
    /// SwiGLU MLP (gate_proj, up_proj) instead of GELU
    gated: bool,
    /// bf16 weights, which ComfyUI rounds to f16 and the GEMMs widen to f32 a layer at a time
    bf16: bool,
}

const VIT_L: Cfg = Cfg { d: 1024, heads: 16, mlp: 4096, layers: 24, gated: false, bf16: false };
const VIT_H: Cfg = Cfg { d: 1280, heads: 20, mlp: 5120, layers: 32, gated: true, bf16: true };
/// CLS and four registers.
pub const PREFIX: usize = 5;
const EPS: f32 = 1e-5;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
// NAF
const NAF_DIM: usize = 256;
const NAF_HALF: usize = 128;
const NAF_GROUPS: usize = 8;

/// True if `st` holds a DINOv3 ViT-L with NAF weights.
pub fn is_dino3_naf(has: impl Fn(&str) -> bool) -> bool {
    has("layer.23.layer_scale2.lambda1") && has("embeddings.register_tokens") && has("naf.image_encoder.rope.periods")
}

/// True if `st` holds a DINOv3 ViT-H/16+ (TripoSplat's `dino_v3_vit_h`).
pub fn is_dino3_vith(has: impl Fn(&str) -> bool) -> bool {
    has("layer.0.mlp.gate_proj.weight") && has("layer.31.norm1.weight") && has("embeddings.register_tokens")
}

/// One layer's weights as stored in the file (f32 for ViT-L, bf16 for ViT-H); every entry has the file's dtype, so
/// a bf16 layer widens to f32 as a whole, each offset doubling.
fn layer_layout(st: &SafeTensors, i: usize, c: Cfg) -> Result<Layout> {
    let p = |x: &str| format!("layer.{i}.{x}");
    let el = if c.bf16 { 2 } else { 4 };
    let mut l = Layout::new();
    l.raw(st, "qkv", &[p("attention.q_proj.weight"), p("attention.k_proj.weight"), p("attention.v_proj.weight")])?;
    // the key projection has no bias
    let mut bias = st.bytes(&p("attention.q_proj.bias"))?.to_vec();
    bias.extend(vec![0u8; c.d * el]);
    bias.extend_from_slice(st.bytes(&p("attention.v_proj.bias"))?);
    l.bytes("qkvb", bias);
    if c.gated {
        // [gate | up] in one GEMM, then silu(gate) * up
        l.raw(st, "up", &[p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")])?;
        l.raw(st, "upb", &[p("mlp.gate_proj.bias"), p("mlp.up_proj.bias")])?;
    } else {
        l.raw(st, "up", &[p("mlp.up_proj.weight")])?;
        l.raw(st, "upb", &[p("mlp.up_proj.bias")])?;
    }
    for (key, name) in [("o", "attention.o_proj.weight"), ("ob", "attention.o_proj.bias"), ("down", "mlp.down_proj.weight"),
        ("downb", "mlp.down_proj.bias"), ("n1w", "norm1.weight"), ("n1b", "norm1.bias"), ("n2w", "norm2.weight"),
        ("n2b", "norm2.bias"), ("ls1", "layer_scale1.lambda1"), ("ls2", "layer_scale2.lambda1")]
    {
        l.raw(st, key, &[p(name)])?;
    }
    Ok(l)
}

/// A tensor as f32 bytes, through f16 where ComfyUI holds it in f16.
fn f32_via(st: &SafeTensors, name: &str, f16: bool) -> Result<Vec<u8>> {
    let v = st.f32_vec(name, false)?;
    Ok(f32_bytes(&if f16 { v.into_iter().map(|x| half::f16::from_f32(x).to_f32()).collect() } else { v }))
}

/// The NAF weights, rounded to f16 as ComfyUI holds them, and its 3x3 kernels reordered to (out, ky, kx, in).
fn naf_layout(st: &SafeTensors) -> Result<Layout> {
    let f16w = |name: &str, reorder: bool| -> Result<Vec<u8>> {
        let info = st.info(name)?;
        let mut v: Vec<f32> = st.f32_vec(name, false)?.into_iter().map(|x| half::f16::from_f32(x).to_f32()).collect();
        if reorder && info.shape.len() == 4 && info.shape[2] == 3 {
            let (co, ci) = (info.shape[0], info.shape[1]);
            let mut r = vec![0f32; v.len()];
            for o in 0..co {
                for c in 0..ci {
                    for k in 0..9 {
                        r[(o * 9 + k) * ci + c] = v[(o * ci + c) * 9 + k];
                    }
                }
            }
            v = r;
        }
        Ok(f32_bytes(&v))
    };
    let mut l = Layout::new();
    for enc in ["encoder", "sem_encoder"] {
        let n = |x: &str| format!("naf.image_encoder.{enc}.{x}");
        l.bytes(&format!("{enc}.0.w"), f16w(&n("0.weight"), true)?);
        l.bytes(&format!("{enc}.0.b"), f16w(&n("0.bias"), false)?);
        for b in 1..=2 {
            for x in ["conv1", "conv2", "norm1", "norm2"] {
                l.bytes(&format!("{enc}.{b}.{x}.w"), f16w(&n(&format!("{b}.{x}.weight")), true)?);
                l.bytes(&format!("{enc}.{b}.{x}.b"), f16w(&n(&format!("{b}.{x}.bias")), false)?);
            }
        }
    }
    l.bytes("periods", f16w("naf.image_encoder.rope.periods", false)?);
    Ok(l)
}

/// One encode: the normalized final tokens (f32 [PREFIX + side * side, d]).
pub struct Tokens {
    pub buf: DevBuf,
    pub side: usize,
    /// width of a token
    pub d: usize,
}

impl Tokens {
    /// The patch grid (f32 [side * side, d], NHWC) starts after the prefix tokens.
    pub fn patches(&self) -> PM {
        self.buf.at(PREFIX * self.d * 4)
    }
}

pub struct Dino3 {
    cfg: Cfg,
    globals: DevBuf,
    gl: Layout,
    layers: Option<WeightSet>,
    /// ViT-L's NAF upsampler
    naf: Option<(DevBuf, Layout)>,
}

impl Dino3 {
    /// Pixal3D's ViT-L with its NAF upsampler.
    pub fn load(st: &SafeTensors, threads: usize, s0: &Stream) -> Result<Self> {
        if !is_dino3_naf(|k| st.has(k)) {
            bail!("not a DINOv3 ViT-L with NAF weights");
        }
        Self::load_cfg(st, VIT_L, threads, s0)
    }

    /// TripoSplat's ViT-H/16+.
    pub fn load_vith(st: &SafeTensors, threads: usize, s0: &Stream) -> Result<Self> {
        if !is_dino3_vith(|k| st.has(k)) {
            bail!("not a DINOv3 ViT-H/16+");
        }
        Self::load_cfg(st, VIT_H, threads, s0)
    }

    fn load_cfg(st: &SafeTensors, cfg: Cfg, threads: usize, s0: &Stream) -> Result<Self> {
        let mut gl = Layout::new();
        let mut globals_names = vec![("pw", "embeddings.patch_embeddings.weight"), ("pb", "embeddings.patch_embeddings.bias"),
            ("cls", "embeddings.cls_token"), ("reg", "embeddings.register_tokens")];
        if cfg.gated {
            globals_names.extend([("nw", "norm.weight"), ("nb", "norm.bias")]);
        }
        for (key, name) in globals_names {
            gl.bytes(key, f32_via(st, name, cfg.bf16)?);
        }
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        let naf = if cfg == VIT_L {
            let nl = naf_layout(st)?;
            let mut nh = vec![0u8; nl.size];
            nl.pack(st, &mut nh)?;
            let naf = DevBuf::alloc(nl.size)?;
            naf.upload(0, &nh, s0)?;
            Some((naf, nl))
        } else {
            None
        };
        s0.sync()?;
        let layouts = (0..cfg.layers).map(|i| layer_layout(st, i, cfg)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, usize::MAX, 0);
        let layers = WeightSet::load("dinov3", st, layouts, p, threads)?;
        Ok(Dino3 { cfg, globals, gl, layers: Some(layers), naf })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Frees the ViT's weights; the NAF stays.
    pub fn free_vit(&mut self) {
        self.layers = None;
    }

    fn n(&self, key: &str) -> PM {
        let (naf, nl) = self.naf.as_ref().expect("the NAF of DINOv3 ViT-L");
        naf.at(nl.off(key))
    }

    /// `_dinov3_encode`: `rgb` is the composite already resized to `size` (f32 [size, size, 3] in [0, 1]).
    pub fn encode(&mut self, rgb: &[f32], size: usize, st: &Stream) -> Result<Tokens> {
        self.encode_probe(rgb, size, st, None)
    }

    /// `encode`, handing `probe` the hidden states after the embedding and after each layer.
    pub fn encode_probe(&mut self, rgb: &[f32], size: usize, st: &Stream, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>)
        -> Result<Tokens> {
        if size % PATCH != 0 || rgb.len() != size * size * 3 {
            bail!("DINOv3 needs a square image whose side is a multiple of {PATCH}");
        }
        let side = size / PATCH;
        let np = side * side;
        let l = PREFIX + np;
        // patches in the conv weight's (channel, ky, kx) order
        let kk = 3 * PATCH * PATCH;
        let mut px = vec![0f32; np * kk];
        for py in 0..side {
            for pxi in 0..side {
                let row = &mut px[(py * side + pxi) * kk..][..kk];
                for c in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            let v = rgb[((py * PATCH + ky) * size + pxi * PATCH + kx) * 3 + c];
                            row[(c * PATCH + ky) * PATCH + kx] = (v - MEAN[c]) / STD[c];
                        }
                    }
                }
            }
        }
        let Cfg { d, heads, mlp, layers: layers_n, gated, bf16 } = self.cfg;
        let pxd = DevBuf::alloc_async(px.len() * 4, st)?;
        pxd.upload(0, &f32_bytes(&px), st)?;
        let x = DevBuf::alloc_async(l * d * 4, st)?;
        crate::cuda::memcpy_d2d(x.ptr(), self.g("cls"), d * 4, st)?;
        crate::cuda::memcpy_d2d(x.at(d * 4), self.g("reg"), 4 * d * 4, st)?;
        k!(gpu_gemm(pxd.ptr(), kk as i64, self.g("pw"), kk as i64, x.at(PREFIX * d * 4), d as i64, self.g("pb"), np as i64, d as i64,
            kk as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
        drop(pxd);
        if let Some(p) = probe.as_mut() {
            p("embed", &x, l * d);
        }

        // RoPE of the patches: 16 frequencies per axis, the first half of each head paired with the second
        let inv: Vec<f32> = (0..HD / 4).map(|i| 1.0 / 100f32.powf(i as f32 * (4.0 / HD as f32))).collect();
        let mut rope = Vec::with_capacity(np * HD);
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        for py in 0..side {
            for pxi in 0..side {
                let coord = [(py as f32 + 0.5) / side as f32 * 2.0 - 1.0, (pxi as f32 + 0.5) / side as f32 * 2.0 - 1.0];
                for c in coord {
                    for f in &inv {
                        let a = two_pi * c * f;
                        rope.extend_from_slice(&[a.cos(), a.sin()]);
                    }
                }
            }
        }
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &f32_bytes(&rope), st)?;

        let h = DevBuf::alloc_async(l * d * 4, st)?;
        let qkv = DevBuf::alloc_async(l * 3 * d * 4, st)?;
        let attn = DevBuf::alloc_async(l * d * 4, st)?;
        let y = DevBuf::alloc_async(l * mlp * if gated { 2 } else { 1 } * 4, st)?;
        let scores = DevBuf::alloc_async(l * l * 4, st)?;
        let (di, li) = (d as i64, l as i64);
        let gemm = |a: PM, lda: usize, w: PM, ldb: usize, bias: PM, out: PM, ldd: usize, m: usize, n: usize, kd: usize, transb: i32|
            -> Result<()> {
            k!(gpu_gemm(a, lda as i64, w, ldb as i64, out, ldd as i64, bias, m as i64, n as i64, kd as i64, transb, DT_F32, DT_F32, 1.0,
                0.0, s(st)))
        };
        let nil = std::ptr::null_mut();
        let Some(layers) = self.layers.as_mut() else { bail!("the DINOv3 weights were freed") };
        // f16 staging and the f32 copy of one layer, for bf16 weights
        let wide = DevBuf::alloc_async(if bf16 { layers.layouts.iter().map(|l| l.size).max().unwrap_or(0) * 3 } else { 1 }, st)?;
        for i in 0..layers_n {
            let base = layers.acquire(i, st)?;
            let lay = layers.layouts[i].clone();
            let base = if bf16 {
                // f16 values, as ComfyUI holds them, widened to f32: every offset doubles
                let n = lay.size / 2;
                k!(gpu_cast(DT_BF16, DT_F16, base, wide.ptr(), n as i64, s(st)))?;
                k!(gpu_cast(DT_F16, DT_F32, wide.ptr(), off(wide.ptr(), n * 2), n as i64, s(st)))?;
                off(wide.ptr(), n * 2)
            } else {
                base
            };
            let scale = if bf16 { 2 } else { 1 };
            let w = |key: &str| off(base, lay.off(key) * scale);
            k!(gpu_prologue_rows_b(DT_F32, PRO_LN_AFFINE, x.ptr(), di, w("n1w"), w("n1b"), EPS, null(), null(), h.ptr(), di, l as i32,
                d as i32, s(st)))?;
            gemm(h.ptr(), d, w("qkv"), d, w("qkvb"), qkv.ptr(), 3 * d, l, 3 * d, d, 1)?;
            if i == 0 {
                if let Some(p) = probe.as_mut() {
                    p("dbg ln1", &h, l * d);
                    p("dbg qkv", &qkv, l * 3 * d);
                }
            }
            for part in 0..2 {
                k!(gpu_rope_rows(off(qkv.ptr(), (PREFIX * 3 * d + part * d) * 4), (3 * d) as i64, rope_d.ptr(), np as i32, heads as i32,
                    HD as i32, s(st)))?;
            }
            for hh in 0..heads {
                let q = off(qkv.ptr(), hh * HD * 4);
                gemm(q, 3 * d, off(q, d * 4), 3 * d, nil, scores.ptr(), l, l, l, HD, 1)?;
                k!(gpu_softmax_f32(scores.ptr(), l as i32, l as i32, 1.0 / (HD as f32).sqrt(), s(st)))?;
                gemm(scores.ptr(), l, off(q, 2 * d * 4), 3 * d, nil, off(attn.ptr(), hh * HD * 4), d, l, HD, l, 0)?;
            }
            if i == 0 {
                if let Some(p) = probe.as_mut() {
                    p("dbg attn", &attn, l * d);
                }
            }
            gemm(attn.ptr(), d, w("o"), d, w("ob"), h.ptr(), d, l, d, d, 1)?;
            k!(gpu_scaled_add_f32(x.ptr(), h.ptr(), w("ls1"), li, d as i32, s(st)))?;
            k!(gpu_prologue_rows_b(DT_F32, PRO_LN_AFFINE, x.ptr(), di, w("n2w"), w("n2b"), EPS, null(), null(), h.ptr(), di, l as i32,
                d as i32, s(st)))?;
            if gated {
                gemm(h.ptr(), d, w("up"), d, w("upb"), y.ptr(), 2 * mlp, l, 2 * mlp, d, 1)?;
                k!(gpu_prologue_rows(DT_F32, PRO_SWIGLU, y.ptr(), (2 * mlp) as i64, null(), 0, mlp as i64, null(), 0.0, null(),
                    null(), y.ptr(), (2 * mlp) as i64, l as i32, mlp as i32, s(st)))?;
                gemm(y.ptr(), 2 * mlp, w("down"), mlp, w("downb"), h.ptr(), d, l, d, mlp, 1)?;
            } else {
                gemm(h.ptr(), d, w("up"), d, w("upb"), y.ptr(), mlp, l, mlp, d, 1)?;
                k!(gpu_gelu_f32(y.ptr(), (l * mlp) as i64, 0, s(st)))?;
                gemm(y.ptr(), mlp, w("down"), mlp, w("downb"), h.ptr(), d, l, d, mlp, 1)?;
            }
            k!(gpu_scaled_add_f32(x.ptr(), h.ptr(), w("ls2"), li, d as i32, s(st)))?;
            layers.release(i, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("layer{i}"), &x, l * d);
            }
        }
        let out = DevBuf::alloc_async(l * d * 4, st)?;
        if gated {
            // TripoSplat: the model's own norm, then a plain layer_norm on top (TripoSplatConditioning)
            k!(gpu_prologue_rows_b(DT_F32, PRO_LN_AFFINE, x.ptr(), di, self.g("nw"), self.g("nb"), EPS, null(), null(), h.ptr(), di,
                l as i32, d as i32, s(st)))?;
            k!(gpu_prologue_rows_b(DT_F32, PRO_LAYERNORM, h.ptr(), di, null(), null(), EPS, null(), null(), out.ptr(), di, l as i32,
                d as i32, s(st)))?;
        } else {
            // Pixal3D: skip_norm_elementwise
            k!(gpu_prologue_rows_b(DT_F32, PRO_LAYERNORM, x.ptr(), di, null(), null(), EPS, null(), null(), out.ptr(), di, l as i32,
                d as i32, s(st)))?;
        }
        Ok(Tokens { buf: out, side, d })
    }

    /// conv (1x1 or 3x3 reflect-padded) of an NHWC f32 image with f16-valued weights, rounded to f16.
    #[allow(clippy::too_many_arguments)]
    fn naf_conv(&self, x: &DevBuf, hw: usize, cin: usize, key: &str, three: bool, out: PM, out_stride: usize, st: &Stream) -> Result<()> {
        let (w, b) = (self.n(&format!("{key}.w")), self.n(&format!("{key}.b")));
        let p = hw * hw;
        if !three {
            k!(gpu_gemm(x.ptr(), cin as i64, w, cin as i64, out, out_stride as i64, b, p as i64, NAF_HALF as i64, cin as i64, 1, DT_F32,
                DT_F32, 1.0, 0.0, s(st)))?;
        } else {
            let kk = 9 * cin;
            let rows = ((256usize << 20) / (hw * kk * 4)).clamp(1, hw);
            let col = DevBuf::alloc_async(rows * hw * kk * 4, st)?;
            for y0 in (0..hw).step_by(rows) {
                let r = rows.min(hw - y0);
                k!(gpu_im2col3_reflect(x.ptr(), hw as i32, hw as i32, cin as i32, y0 as i32, r as i32, col.ptr(),
                    s(st)))?;
                k!(gpu_gemm(col.ptr(), kk as i64, w, kk as i64, off(out, y0 * hw * out_stride * 4), out_stride as i64, b, (r * hw) as i64,
                    NAF_HALF as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
            }
        }
        if out_stride == NAF_HALF {
            k!(gpu_round_f16(out, (p * NAF_HALF) as i64, s(st)))?;
        }
        Ok(())
    }

    /// `NAF.forward`: upsample `lr` (f32 [h * h, D] NHWC patches) to `out_size`^2 guided by `rgb` (the composite
    /// resized to `img_size`), as an f16 NHWC map [out_size^2, D].
    pub fn naf(&self, rgb: &[f32], img_size: usize, lr: PM, h: usize, out_size: usize, st: &Stream) -> Result<DevBuf> {
        if img_size > 4 * out_size || img_size % out_size != 0 || out_size % h != 0 {
            bail!("unsupported NAF sizes {img_size} -> {out_size} over {h}");
        }
        let p = img_size * img_size;
        let img: Vec<f32> = rgb.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        let x = DevBuf::alloc_async(p * 3 * 4, st)?;
        x.upload(0, &f32_bytes(&img), st)?;
        let cat = DevBuf::alloc_async(p * NAF_DIM * 4, st)?;
        let a = DevBuf::alloc_async(p * NAF_HALF * 4, st)?;
        let b = DevBuf::alloc_async(p * NAF_HALF * 4, st)?;
        let stats = DevBuf::alloc_async(2 * NAF_GROUPS * 8, st)?;
        for (half, enc, three) in [(0, "encoder", false), (1, "sem_encoder", true)] {
            self.naf_conv(&x, img_size, 3, &format!("{enc}.0"), three, a.ptr(), NAF_HALF, st)?;
            for blk in 1..=2 {
                let key = |n: &str| format!("{enc}.{blk}.{n}");
                for (norm, conv, src, dst) in [("norm1", "conv1", &a, &b), ("norm2", "conv2", &b, &a)] {
                    k!(gpu_group_norm_nhwc_f32(src.ptr(), p as i64, NAF_HALF as i32, NAF_GROUPS as i32, self.n(&format!("{}.w", key(norm))),
                        self.n(&format!("{}.b", key(norm))), EPS, 1, stats.ptr(), s(st)))?;
                    self.naf_conv(src, img_size, NAF_HALF, &key(conv), three, dst.ptr(), NAF_HALF, st)?;
                }
            }
            // the two stacks' channels side by side
            crate::cuda::memcpy_2d_d2d(off(cat.ptr(), half * NAF_HALF * 4), NAF_DIM * 4, a.ptr(), NAF_HALF * 4, NAF_HALF * 4, p, st)?;
        }
        drop((a, b));
        let f = img_size / out_size;
        let po = out_size * out_size;
        let q = if f == 1 {
            cat
        } else {
            let pooled = DevBuf::alloc_async(po * NAF_DIM * 4, st)?;
            k!(gpu_avg_pool_nhwc_f32(cat.ptr(), img_size as i32, img_size as i32, NAF_DIM as i32, f as i32,
                pooled.ptr(), s(st)))?;
            k!(gpu_round_f16(pooled.ptr(), (po * NAF_DIM) as i64, s(st)))?;
            pooled
        };
        k!(gpu_naf_rope(q.ptr(), out_size as i32, out_size as i32, NAF_DIM as i32, self.n("periods"), s(st)))?;
        k!(gpu_round_f16(q.ptr(), (po * NAF_DIM) as i64, s(st)))?;
        let kf = DevBuf::alloc_async(h * h * NAF_DIM * 4, st)?;
        k!(gpu_avg_pool_nhwc_f32(q.ptr(), out_size as i32, out_size as i32, NAF_DIM as i32, (out_size / h) as i32,
            kf.ptr(), s(st)))?;
        let q16 = DevBuf::alloc_async(po * NAF_DIM * 2, st)?;
        let k16 = DevBuf::alloc_async(h * h * NAF_DIM * 2, st)?;
        let v16 = DevBuf::alloc_async(h * h * D * 2, st)?;
        k!(gpu_cast(DT_F32, DT_F16, q.ptr(), q16.ptr(), (po * NAF_DIM) as i64, s(st)))?;
        k!(gpu_cast(DT_F32, DT_F16, kf.ptr(), k16.ptr(), (h * h * NAF_DIM) as i64, s(st)))?;
        k!(gpu_cast(DT_F32, DT_F16, lr, v16.ptr(), (h * h * D) as i64, s(st)))?;
        drop((q, kf));
        let out = DevBuf::alloc_async(po * D * 2, st)?;
        k!(gpu_naf_attn(q16.ptr(), k16.ptr(), v16.ptr(), out.ptr(), out_size as i32, out_size as i32, h as i32, h as i32, (D / 4) as i32,
            s(st)))?;
        Ok(out)
    }
}
