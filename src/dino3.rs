//! Pixal3D's image encoder: DINOv3 ViT-L/16 (f32, like ComfyUI runs it) and the NAF feature upsampler bundled in
//! the same file (f16 in ComfyUI, emulated by rounding to f16 where it materializes a tensor).

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F16, DT_F32, PM, PRO_LAYERNORM, PRO_LN_AFFINE};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};

const D: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const MLP: usize = 4096;
const LAYERS: usize = 24;
const PATCH: usize = 16;
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

fn layer_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let p = |x: &str| format!("layer.{i}.{x}");
    let mut l = Layout::new();
    l.raw(st, "qkv", &[p("attention.q_proj.weight"), p("attention.k_proj.weight"), p("attention.v_proj.weight")])?;
    // the key projection has no bias
    let mut bias = st.bytes(&p("attention.q_proj.bias"))?.to_vec();
    bias.extend(vec![0u8; D * 4]);
    bias.extend_from_slice(st.bytes(&p("attention.v_proj.bias"))?);
    l.bytes("qkvb", bias);
    for (key, name) in [("o", "attention.o_proj.weight"), ("ob", "attention.o_proj.bias"), ("up", "mlp.up_proj.weight"),
        ("upb", "mlp.up_proj.bias"), ("down", "mlp.down_proj.weight"), ("downb", "mlp.down_proj.bias"), ("n1w", "norm1.weight"),
        ("n1b", "norm1.bias"), ("n2w", "norm2.weight"), ("n2b", "norm2.bias"), ("ls1", "layer_scale1.lambda1"),
        ("ls2", "layer_scale2.lambda1")]
    {
        l.raw(st, key, &[p(name)])?;
    }
    Ok(l)
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

/// One encode: the normalized final tokens (f32 [PREFIX + side * side, D]).
pub struct Tokens {
    pub buf: DevBuf,
    pub side: usize,
}

impl Tokens {
    /// The patch grid (f32 [side * side, D], NHWC) starts after the prefix tokens.
    pub fn patches(&self) -> PM {
        self.buf.at(PREFIX * D * 4)
    }
}

pub struct Dino3 {
    globals: DevBuf,
    gl: Layout,
    layers: Option<WeightSet>,
    naf: DevBuf,
    nl: Layout,
}

impl Dino3 {
    pub fn load(st: &SafeTensors, threads: usize, s0: &Stream) -> Result<Self> {
        if !is_dino3_naf(|k| st.has(k)) {
            bail!("not a DINOv3 ViT-L with NAF weights");
        }
        let mut gl = Layout::new();
        for (key, name) in [("pw", "embeddings.patch_embeddings.weight"), ("pb", "embeddings.patch_embeddings.bias"),
            ("cls", "embeddings.cls_token"), ("reg", "embeddings.register_tokens")]
        {
            gl.raw(st, key, &[name.to_string()])?;
        }
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        let nl = naf_layout(st)?;
        let mut nh = vec![0u8; nl.size];
        nl.pack(st, &mut nh)?;
        let naf = DevBuf::alloc(nl.size)?;
        naf.upload(0, &nh, s0)?;
        s0.sync()?;
        let layouts = (0..LAYERS).map(|i| layer_layout(st, i)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, usize::MAX, 0);
        let layers = WeightSet::load("dinov3", st, layouts, p, threads)?;
        Ok(Dino3 { globals, gl, layers: Some(layers), naf, nl })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Frees the ViT's weights; the NAF stays.
    pub fn free_vit(&mut self) {
        self.layers = None;
    }

    fn n(&self, key: &str) -> PM {
        self.naf.at(self.nl.off(key))
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
        let pxd = DevBuf::alloc_async(px.len() * 4, st)?;
        pxd.upload(0, &f32_bytes(&px), st)?;
        let x = DevBuf::alloc_async(l * D * 4, st)?;
        crate::cuda::memcpy_d2d(x.ptr(), self.g("cls"), D * 4, st)?;
        crate::cuda::memcpy_d2d(x.at(D * 4), self.g("reg"), 4 * D * 4, st)?;
        k!(gpu_gemm(pxd.ptr(), kk as i64, self.g("pw"), kk as i64, x.at(PREFIX * D * 4), D as i64, self.g("pb"), np as i64, D as i64,
            kk as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
        drop(pxd);
        if let Some(p) = probe.as_mut() {
            p("embed", &x, l * D);
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

        let h = DevBuf::alloc_async(l * D * 4, st)?;
        let qkv = DevBuf::alloc_async(l * 3 * D * 4, st)?;
        let attn = DevBuf::alloc_async(l * D * 4, st)?;
        let y = DevBuf::alloc_async(l * MLP * 4, st)?;
        let scores = DevBuf::alloc_async(l * l * 4, st)?;
        let (di, li) = (D as i64, l as i64);
        let gemm = |a: PM, lda: usize, w: PM, ldb: usize, bias: PM, out: PM, ldd: usize, m: usize, n: usize, kd: usize, transb: i32|
            -> Result<()> {
            k!(gpu_gemm(a, lda as i64, w, ldb as i64, out, ldd as i64, bias, m as i64, n as i64, kd as i64, transb, DT_F32, DT_F32, 1.0,
                0.0, s(st)))
        };
        let nil = std::ptr::null_mut();
        let Some(layers) = self.layers.as_mut() else { bail!("the DINOv3 weights were freed") };
        for i in 0..LAYERS {
            let base = layers.acquire(i, st)?;
            let lay = layers.layouts[i].clone();
            let w = |key: &str| off(base, lay.off(key));
            k!(gpu_prologue_rows_b(DT_F32, PRO_LN_AFFINE, x.ptr(), di, w("n1w"), w("n1b"), EPS, null(), null(), h.ptr(), di, l as i32,
                D as i32, s(st)))?;
            gemm(h.ptr(), D, w("qkv"), D, w("qkvb"), qkv.ptr(), 3 * D, l, 3 * D, D, 1)?;
            if i == 0 {
                if let Some(p) = probe.as_mut() {
                    p("dbg ln1", &h, l * D);
                    p("dbg qkv", &qkv, l * 3 * D);
                }
            }
            for part in 0..2 {
                k!(gpu_rope_rows(off(qkv.ptr(), (PREFIX * 3 * D + part * D) * 4), (3 * D) as i64, rope_d.ptr(), np as i32, HEADS as i32,
                    HD as i32, s(st)))?;
            }
            for hh in 0..HEADS {
                let q = off(qkv.ptr(), hh * HD * 4);
                gemm(q, 3 * D, off(q, D * 4), 3 * D, nil, scores.ptr(), l, l, l, HD, 1)?;
                k!(gpu_softmax_f32(scores.ptr(), l as i32, l as i32, 1.0 / (HD as f32).sqrt(), s(st)))?;
                gemm(scores.ptr(), l, off(q, 2 * D * 4), 3 * D, nil, off(attn.ptr(), hh * HD * 4), D, l, HD, l, 0)?;
            }
            if i == 0 {
                if let Some(p) = probe.as_mut() {
                    p("dbg attn", &attn, l * D);
                }
            }
            gemm(attn.ptr(), D, w("o"), D, w("ob"), h.ptr(), D, l, D, D, 1)?;
            k!(gpu_scaled_add_f32(x.ptr(), h.ptr(), w("ls1"), li, D as i32, s(st)))?;
            k!(gpu_prologue_rows_b(DT_F32, PRO_LN_AFFINE, x.ptr(), di, w("n2w"), w("n2b"), EPS, null(), null(), h.ptr(), di, l as i32,
                D as i32, s(st)))?;
            gemm(h.ptr(), D, w("up"), D, w("upb"), y.ptr(), MLP, l, MLP, D, 1)?;
            k!(gpu_gelu_f32(y.ptr(), (l * MLP) as i64, 0, s(st)))?;
            gemm(y.ptr(), MLP, w("down"), MLP, w("downb"), h.ptr(), D, l, D, MLP, 1)?;
            k!(gpu_scaled_add_f32(x.ptr(), h.ptr(), w("ls2"), li, D as i32, s(st)))?;
            layers.release(i, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("layer{i}"), &x, l * D);
            }
        }
        let out = DevBuf::alloc_async(l * D * 4, st)?;
        k!(gpu_prologue_rows_b(DT_F32, PRO_LAYERNORM, x.ptr(), di, null(), null(), EPS, null(), null(), out.ptr(), di, l as i32, D as i32,
            s(st)))?;
        Ok(Tokens { buf: out, side })
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
