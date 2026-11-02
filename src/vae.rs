//! Wan 2.1 VAE decoder (qwen_image_vae) for single images, NHWC bf16.
//!
//! With one latent frame and no temporal cache, every CausalConv3d sees two zero frames of
//! padding before the current one, so it equals a 2D conv with the kernel's last temporal slice,
//! and the temporal upsampling convs are skipped (as in ComfyUI).

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, PM};
use crate::offload::Layout;
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};
use half::bf16;
use std::path::Path;

const LATENT_MEAN: [f32; 16] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
    -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
const LATENT_STD: [f32; 16] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
    2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160];
const OUT_PAD: usize = 8;

enum Op {
    Res { pre: String, cin: usize, cout: usize, shortcut: bool },
    Attn { pre: String, c: usize },
    Up { pre: String, cin: usize },
}

pub struct Vae {
    buf: DevBuf,
    lay: Layout,
    ops: Vec<Op>,
    zdim: usize,
    c0: usize,
}

/// Conv weight [Cout, Cin, (T,) R, R] -> KRSC bf16 (last temporal slice), Cout zero-padded to `cout_pad`.
pub(crate) fn conv_krsc(st: &SafeTensors, name: &str, cout_pad: usize) -> Result<(Vec<u8>, usize, usize, usize)> {
    let info = st.info(name)?.clone();
    let v = st.f32_vec(name, false)?;
    let (co, ci) = (info.shape[0], info.shape[1]);
    let (t, r) = match info.shape.len() {
        5 => (info.shape[2], info.shape[3]),
        4 => (1, info.shape[2]),
        _ => bail!("{name}: unexpected conv shape {:?}", info.shape),
    };
    let cop = cout_pad.max(co);
    let mut out = vec![bf16::ZERO; cop * r * r * ci];
    for o in 0..co {
        for c in 0..ci {
            for y in 0..r {
                for x in 0..r {
                    let src = (((o * ci + c) * t + (t - 1)) * r + y) * r + x;
                    out[((o * r + y) * r + x) * ci + c] = bf16::from_f32(v[src]);
                }
            }
        }
    }
    Ok((out.iter().flat_map(|b| b.to_le_bytes()).collect(), cop, ci, r))
}

pub(crate) fn bias_pad(st: &SafeTensors, name: &str, pad: usize) -> Result<Vec<u8>> {
    let mut v = st.f32_vec(name, false)?;
    v.resize(pad.max(v.len()), 0.0);
    Ok(v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect())
}

/// Single-head spatial self-attention, added to x [p, c] in place. `n` holds the normalized x and is
/// overwritten. `w` is the weight and bias of [q | k | v], then of the output projection.
pub(crate) fn single_head_attention(x: &DevBuf, n: &DevBuf, p: usize, c: usize, w: [PM; 4], st: &Stream) -> Result<()> {
    let qkv = DevBuf::alloc_async(p * 3 * c * 2, st)?;
    let (ci, c3) = (c as i64, 3 * c as i64);
    k!(krea_gemm(n.ptr(), ci, w[0], ci, qkv.ptr(), c3, w[1], p as i64, c3, ci, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
    // query rows per chunk: keep the f32 scores + bf16 probabilities under ~256 MB
    let chunk = ((256 << 20) / (p * 6)).clamp(64, 4096).min(p);
    let sbuf = DevBuf::alloc_async(chunk * p * 4, st)?;
    let pbuf = DevBuf::alloc_async(chunk * p * 2, st)?;
    let q = qkv.ptr();
    let (kp, vp) = (off(q, c * 2), off(q, 2 * c * 2));
    for r0 in (0..p).step_by(chunk) {
        let rows = chunk.min(p - r0);
        k!(krea_gemm(off(q, r0 * 3 * c * 2), c3, kp, c3, sbuf.ptr(), p as i64, null(), rows as i64, p as i64, ci, 1, DT_BF16,
            crate::ffi::DT_F32, 1.0, 0.0, s(st)))?;
        k!(krea_softmax_rows(sbuf.ptr(), pbuf.ptr(), rows as i32, p as i32, 1.0 / (c as f32).sqrt(), s(st)))?;
        k!(krea_gemm(pbuf.ptr(), p as i64, vp, c3, off(n.ptr(), r0 * c * 2), ci, null(), rows as i64, ci, p as i64, 0, DT_BF16, DT_BF16, 1.0,
            0.0, s(st)))?;
    }
    let y = DevBuf::alloc_async(p * c * 2, st)?;
    k!(krea_gemm(n.ptr(), ci, w[2], ci, y.ptr(), ci, w[3], p as i64, ci, ci, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
    k!(krea_add_gated_bf16(x.ptr(), ci, y.ptr(), ci, null(), p as i32, c as i32, s(st)))
}

impl Vae {
    pub fn load(path: &Path, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let mut lay = Layout::new();
        let conv = |lay: &mut Layout, key: &str, name: &str, pad: usize| -> Result<usize> {
            let (w, cop, _, _) = conv_krsc(&st, &format!("{name}.weight"), pad)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias_pad(&st, &format!("{name}.bias"), cop)?);
            Ok(cop)
        };
        let zdim = st.info("decoder.conv1.weight")?.shape[1];
        let c0 = st.info("decoder.conv1.weight")?.shape[0];
        conv(&mut lay, "conv2", "conv2", 0)?;
        conv(&mut lay, "conv1", "decoder.conv1", 0)?;
        let mut ops = Vec::new();
        let mut blocks: Vec<String> = (0..3).map(|i| format!("decoder.middle.{i}")).collect();
        blocks.extend((0..).map(|i| format!("decoder.upsamples.{i}")).take_while(|p| {
            st.has(&format!("{p}.residual.2.weight")) || st.has(&format!("{p}.resample.1.weight"))
        }));
        for p in blocks {
            if st.has(&format!("{p}.residual.2.weight")) {
                let cin = st.info(&format!("{p}.residual.2.weight"))?.shape[1];
                let cout = st.info(&format!("{p}.residual.2.weight"))?.shape[0];
                lay.bf16(&st, &format!("{p}.g0"), &[format!("{p}.residual.0.gamma")])?;
                lay.bf16(&st, &format!("{p}.g3"), &[format!("{p}.residual.3.gamma")])?;
                conv(&mut lay, &format!("{p}.c2"), &format!("{p}.residual.2"), 0)?;
                conv(&mut lay, &format!("{p}.c6"), &format!("{p}.residual.6"), 0)?;
                let shortcut = st.has(&format!("{p}.shortcut.weight"));
                if shortcut {
                    conv(&mut lay, &format!("{p}.sc"), &format!("{p}.shortcut"), 0)?;
                }
                ops.push(Op::Res { pre: p, cin, cout, shortcut });
            } else if st.has(&format!("{p}.to_qkv.weight")) {
                let c = st.info(&format!("{p}.proj.weight"))?.shape[0];
                lay.bf16(&st, &format!("{p}.g"), &[format!("{p}.norm.gamma")])?;
                lay.bf16(&st, &format!("{p}.qkv.w"), &[format!("{p}.to_qkv.weight")])?;
                lay.bf16(&st, &format!("{p}.qkv.b"), &[format!("{p}.to_qkv.bias")])?;
                lay.bf16(&st, &format!("{p}.proj.w"), &[format!("{p}.proj.weight")])?;
                lay.bf16(&st, &format!("{p}.proj.b"), &[format!("{p}.proj.bias")])?;
                ops.push(Op::Attn { pre: p, c });
            } else {
                let cin = st.info(&format!("{p}.resample.1.weight"))?.shape[1];
                conv(&mut lay, &format!("{p}.up"), &format!("{p}.resample.1"), 0)?;
                ops.push(Op::Up { pre: p, cin });
            }
        }
        lay.bf16(&st, "head.g", &["decoder.head.0.gamma".to_string()])?;
        conv(&mut lay, "head", "decoder.head.2", OUT_PAD)?;
        lay.bytes("mean", f32_bytes(&LATENT_MEAN));
        lay.bytes("std", f32_bytes(&LATENT_STD));
        let mut host = vec![0u8; lay.size];
        lay.pack(&st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Vae { buf, lay, ops, zdim, c0 })
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    fn conv(&self, key: &str, x: PM, out: PM, h: usize, w: usize, cin: usize, cout: usize, r: usize, st: &Stream)
        -> Result<()> {
        k!(krea_conv2d(x, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), out, 1, h as i32, w as i32,
            cin as i32, cout as i32, r as i32, s(st)))
    }

    /// Peak activation memory (bytes) for decoding an h x w latent.
    pub fn activation_bytes(h: usize, w: usize) -> usize {
        let full = h * 8 * w * 8;
        // x + h + residual temp at full res (96ch), upsample buffer (192ch at full res), attention chunk
        full * 96 * 2 * 3 + full * 192 * 2 + (512 << 20)
    }

    /// latent f32 [16, h, w] -> RGB8 [8h, 8w, 3]
    pub fn decode(&self, latent: &DevBuf, h: usize, w: usize, st: &Stream) -> Result<Vec<u8>> {
        let z = self.zdim;
        let alloc = |c: usize, hh: usize, ww: usize| DevBuf::alloc_async(hh * ww * c * 2, st);
        let zin = alloc(z, h, w)?;
        k!(krea_vae_latent_in(latent.ptr(), self.w("mean"), self.w("std"), zin.ptr(), z as i32, (h * w) as i32, s(st)))?;
        let z2 = alloc(z, h, w)?;
        self.conv("conv2", zin.ptr(), z2.ptr(), h, w, z, z, 1, st)?;
        let c0 = self.c0;
        let mut x = alloc(c0, h, w)?;
        self.conv("conv1", z2.ptr(), x.ptr(), h, w, z, c0, 3, st)?;
        drop((zin, z2));
        let (mut hh, mut ww, mut c) = (h, w, c0);

        for op in &self.ops {
            match op {
                Op::Res { pre, cin, cout, shortcut } => {
                    debug_assert_eq!(*cin, c);
                    let p = (hh * ww) as i64;
                    let t = alloc(*cin, hh, ww)?;
                    k!(krea_l2norm_silu(x.ptr(), t.ptr(), self.w(&format!("{pre}.g0")), p, *cin as i32, 1, s(st)))?;
                    let u = alloc(*cout, hh, ww)?;
                    self.conv(&format!("{pre}.c2"), t.ptr(), u.ptr(), hh, ww, *cin, *cout, 3, st)?;
                    drop(t);
                    let t2 = alloc(*cout, hh, ww)?;
                    k!(krea_l2norm_silu(u.ptr(), t2.ptr(), self.w(&format!("{pre}.g3")), p, *cout as i32, 1, s(st)))?;
                    self.conv(&format!("{pre}.c6"), t2.ptr(), u.ptr(), hh, ww, *cout, *cout, 3, st)?;
                    drop(t2);
                    if *shortcut {
                        let sc = alloc(*cout, hh, ww)?;
                        self.conv(&format!("{pre}.sc"), x.ptr(), sc.ptr(), hh, ww, *cin, *cout, 1, st)?;
                        k!(krea_add_gated_bf16(u.ptr(), *cout as i64, sc.ptr(), *cout as i64, null(), p as i32,
                            *cout as i32, s(st)))?;
                    } else {
                        k!(krea_add_gated_bf16(u.ptr(), *cout as i64, x.ptr(), *cout as i64, null(), p as i32,
                            *cout as i32, s(st)))?;
                    }
                    x = u;
                    c = *cout;
                }
                Op::Attn { pre, c: ca } => {
                    self.attention(pre, &x, hh * ww, *ca, st)?;
                }
                Op::Up { pre, cin } => {
                    let up = alloc(*cin, 2 * hh, 2 * ww)?;
                    k!(krea_upsample2x(x.ptr(), up.ptr(), hh as i32, ww as i32, *cin as i32, s(st)))?;
                    hh *= 2;
                    ww *= 2;
                    let co = cin / 2;
                    let y = alloc(co, hh, ww)?;
                    drop(x);
                    self.conv(&format!("{pre}.up"), up.ptr(), y.ptr(), hh, ww, *cin, co, 3, st)?;
                    x = y;
                    c = co;
                }
            }
        }
        let p = hh * ww;
        let t = alloc(c, hh, ww)?;
        k!(krea_l2norm_silu(x.ptr(), t.ptr(), self.w("head.g"), p as i64, c as i32, 1, s(st)))?;
        let o = alloc(OUT_PAD, hh, ww)?;
        self.conv("head", t.ptr(), o.ptr(), hh, ww, c, OUT_PAD, 3, st)?;
        let rgb = DevBuf::alloc_async(p * 3, st)?;
        k!(krea_to_rgb8(o.ptr(), OUT_PAD as i32, rgb.ptr(), p as i64, s(st)))?;
        let mut out = vec![0u8; p * 3];
        rgb.download(0, &mut out, st)?;
        Ok(out)
    }

    /// Single-head spatial self-attention (in place on x [p, c]).
    fn attention(&self, pre: &str, x: &DevBuf, p: usize, c: usize, st: &Stream) -> Result<()> {
        let n = DevBuf::alloc_async(p * c * 2, st)?;
        k!(krea_l2norm_silu(x.ptr(), n.ptr(), self.w(&format!("{pre}.g")), p as i64, c as i32, 0, s(st)))?;
        let w = |key: &str| self.w(&format!("{pre}.{key}"));
        single_head_attention(x, &n, p, c, [w("qkv.w"), w("qkv.b"), w("proj.w"), w("proj.b")], st)
    }
}

