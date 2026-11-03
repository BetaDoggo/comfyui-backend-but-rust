//! BiRefNet background removal (ComfyUI's `LoadBackgroundRemovalModel` + `RemoveBackground`): a Swin-L backbone run
//! on the image at 1024 and at 512, and a decoder of deformable convolutions that turns it into a foreground mask.
//! ComfyUI computes in float32 (its manual_cast ops follow the input's dtype), so this does too; the checkpoint's
//! float16 weights are widened, and every BatchNorm is folded into the convolution before it.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, P, PM, PRO_LN_AFFINE};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::Path;

/// The model's input side; ComfyUI's config fixes it at 1024.
pub const SIZE: usize = 1024;
const WS: usize = 12;
const DEPTHS: [usize; 4] = [2, 2, 18, 2];
const EMBED: usize = 192;
const LN_EPS: f32 = 1e-5;
const BN_EPS: f32 = 1e-5;
/// Largest im2col chunk, bytes.
const COL_BUDGET: usize = 256 << 20;

/// True if `has` finds BiRefNet's tensors (ComfyUI keys the loader on the same one).
pub fn is_birefnet(has: impl Fn(&str) -> bool) -> bool {
    has("bb.layers.1.blocks.0.attn.relative_position_index") && has("decoder.conv_out1.0.weight")
}

struct Weights {
    buf: DevBuf,
    off: HashMap<String, usize>,
}

impl Weights {
    fn p(&self, key: &str) -> P {
        match self.off.get(key) {
            Some(o) => off(self.buf.ptr(), *o),
            None => panic!("BiRefNet weight {key}"),
        }
    }
}

/// A convolution's weights (out, in, ky, kx) as rows of (ky, kx, in).
fn tap_major(w: &[f32], co: usize, ci: usize, kk: usize) -> Vec<f32> {
    let mut r = vec![0f32; w.len()];
    for o in 0..co {
        for c in 0..ci {
            for t in 0..kk {
                r[(o * kk + t) * ci + c] = w[(o * ci + c) * kk + t];
            }
        }
    }
    r
}

struct Loader<'a> {
    st: &'a SafeTensors,
    stream: &'a Stream,
    buf: DevBuf,
    at: usize,
    off: HashMap<String, usize>,
}

impl Loader<'_> {
    fn put(&mut self, key: &str, v: &[f32]) -> Result<()> {
        let bytes = f32_bytes(v);
        self.at = (self.at + 255) / 256 * 256;
        if self.at + bytes.len() > self.buf.size() {
            bail!("BiRefNet weights overflow their buffer at {key}");
        }
        self.buf.upload(self.at, &bytes, self.stream)?;
        self.off.insert(key.to_string(), self.at);
        self.at += bytes.len();
        Ok(())
    }

    fn vec(&self, name: &str) -> Result<Vec<f32>> {
        self.st.f32_vec(name, false)
    }

    /// Stores the tensor as it is.
    fn plain(&mut self, name: &str) -> Result<()> {
        let v = self.vec(name)?;
        self.put(name, &v)
    }

    /// `{key}.w` (out, ky, kx, in) and `{key}.b` of a convolution `conv` with the BatchNorm `bn` folded in.
    fn conv(&mut self, key: &str, conv: &str, bn: Option<&str>) -> Result<()> {
        let info = self.st.info(&format!("{conv}.weight"))?;
        let (co, ci, kk) = (info.shape[0], info.shape[1], info.shape[2] * info.shape[3]);
        let mut w = self.vec(&format!("{conv}.weight"))?;
        let mut b = if self.st.has(&format!("{conv}.bias")) { self.vec(&format!("{conv}.bias"))? } else { vec![0f32; co] };
        if let Some(bn) = bn {
            let (g, beta) = (self.vec(&format!("{bn}.weight"))?, self.vec(&format!("{bn}.bias"))?);
            let (mean, var) = (self.vec(&format!("{bn}.running_mean"))?, self.vec(&format!("{bn}.running_var"))?);
            for o in 0..co {
                let sc = g[o] / (var[o] + BN_EPS).sqrt();
                w[o * ci * kk..(o + 1) * ci * kk].iter_mut().for_each(|x| *x *= sc);
                b[o] = (b[o] - mean[o]) * sc + beta[o];
            }
        }
        self.put(&format!("{key}.w"), &tap_major(&w, co, ci, kk))?;
        self.put(&format!("{key}.b"), &b)
    }

    /// A deformable convolution with its BatchNorm: `{key}.om` (the offset and modulator convolutions stacked, offsets
    /// first) and `{key}.reg`.
    fn deform(&mut self, key: &str) -> Result<()> {
        let a = format!("{key}.atrous_conv");
        let info = self.st.info(&format!("{a}.offset_conv.weight"))?;
        let (ci, kk) = (info.shape[1], info.shape[2] * info.shape[3]);
        let mut w = tap_major(&self.vec(&format!("{a}.offset_conv.weight"))?, 2 * kk, ci, kk);
        w.extend(tap_major(&self.vec(&format!("{a}.modulator_conv.weight"))?, kk, ci, kk));
        let mut b = self.vec(&format!("{a}.offset_conv.bias"))?;
        b.extend(self.vec(&format!("{a}.modulator_conv.bias"))?);
        self.put(&format!("{key}.om.w"), &w)?;
        self.put(&format!("{key}.om.b"), &b)?;
        self.conv(&format!("{key}.reg"), &format!("{a}.regular_conv"), Some(&format!("{key}.bn")))
    }

    fn dec_blk(&mut self, b: &str) -> Result<()> {
        self.conv(&format!("{b}.conv_in"), &format!("{b}.conv_in"), Some(&format!("{b}.bn_in")))?;
        self.conv(&format!("{b}.conv_out"), &format!("{b}.conv_out"), Some(&format!("{b}.bn_out")))?;
        let a = format!("{b}.dec_att");
        self.deform(&format!("{a}.aspp1"))?;
        for i in 0..3 {
            self.deform(&format!("{a}.aspp_deforms.{i}"))?;
        }
        self.conv(&format!("{a}.pool"), &format!("{a}.global_avg_pool.1"), Some(&format!("{a}.global_avg_pool.2")))?;
        self.conv(&format!("{a}.conv1"), &format!("{a}.conv1"), Some(&format!("{a}.bn1")))
    }
}

/// One feature map: NHWC f32 [side * side, c].
struct Map {
    buf: DevBuf,
    side: usize,
    c: usize,
}

pub struct BiRefNet {
    w: Weights,
}

fn buf(floats: usize, st: &Stream) -> Result<DevBuf> {
    DevBuf::alloc_async(floats * 4, st)
}

#[allow(clippy::too_many_arguments)]
fn gemm(st: &Stream, a: P, lda: usize, w: P, bias: P, out: PM, ldo: usize, m: usize, n: usize, kd: usize) -> Result<()> {
    k!(gpu_gemm(a, lda as i64, w, kd as i64, out, ldo as i64, bias, m as i64, n as i64, kd as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))
}

fn relu(x: PM, ld: usize, c: usize, rows: usize, st: &Stream) -> Result<()> {
    k!(gpu_act_rows(x, ld as i64, c as i32, rows as i64, 0, s(st)))
}

fn resize(inp: P, side: usize, c: usize, out: PM, ldo: usize, to: usize, st: &Stream) -> Result<()> {
    k!(gpu_resize_bilinear_ac(inp, side as i32, side as i32, c as i32, out, ldo as i64, to as i32, to as i32, s(st)))
}

/// Copies `c` channels of a [rows, ld_src] map into columns `col ..` of a [rows, ld] one.
fn put_cols(dst: PM, ld: usize, col: usize, src: P, ld_src: usize, c: usize, rows: usize, st: &Stream) -> Result<()> {
    crate::cuda::memcpy_2d_d2d(off(dst, col * 4), ld * 4, src as PM, ld_src * 4, c * 4, rows, st)
}

impl BiRefNet {
    pub fn load(path: &Path, st: &Stream) -> Result<Self> {
        let f = SafeTensors::open(path)?;
        if !is_birefnet(|k| f.has(k)) {
            bail!("{} is not a BiRefNet checkpoint", path.display());
        }
        // an upper bound: every float tensor, widened, plus the alignment
        let total: usize = f.tensors.iter().filter(|(_, i)| i.dtype != crate::st::Dt::I64).map(|(_, i)| i.numel() * 4 + 512).sum();
        let mut l = Loader { st: &f, stream: st, buf: DevBuf::alloc(total + (1 << 20))?, at: 0, off: HashMap::new() };

        l.conv("bb.patch_embed.proj", "bb.patch_embed.proj", None)?;
        for n in ["bb.patch_embed.norm.weight", "bb.patch_embed.norm.bias"] {
            l.plain(n)?;
        }
        for (i, depth) in DEPTHS.iter().enumerate() {
            for j in 0..*depth {
                let b = format!("bb.layers.{i}.blocks.{j}");
                for n in ["norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.relative_position_bias_table", "attn.proj.weight",
                    "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
                {
                    l.plain(&format!("{b}.{n}"))?;
                }
            }
            if i < 3 {
                for n in ["norm.weight", "norm.bias", "reduction.weight"] {
                    l.plain(&format!("bb.layers.{i}.downsample.{n}"))?;
                }
            }
            for n in ["weight", "bias"] {
                l.plain(&format!("bb.norm{i}.{n}"))?;
            }
        }
        l.dec_blk("squeeze_module.0")?;
        for i in 1..=4 {
            l.dec_blk(&format!("decoder.decoder_block{i}"))?;
        }
        for i in 1..=5 {
            let b = format!("decoder.ipt_blk{i}");
            l.conv(&format!("{b}.conv1"), &format!("{b}.conv1"), None)?;
            l.conv(&format!("{b}.conv_out"), &format!("{b}.conv_out"), None)?;
        }
        for i in 2..=4 {
            l.conv(&format!("decoder.lateral_block{i}"), &format!("decoder.lateral_block{i}.conv"), None)?;
            l.conv(&format!("decoder.gdt_convs_{i}"), &format!("decoder.gdt_convs_{i}.0"), Some(&format!("decoder.gdt_convs_{i}.1")))?;
            l.conv(&format!("decoder.gdt_attn_{i}"), &format!("decoder.gdt_convs_attn_{i}.0"), None)?;
        }
        l.conv("decoder.conv_out1", "decoder.conv_out1.0", None)?;
        st.sync()?;
        let Loader { buf, off, .. } = l;
        Ok(BiRefNet { w: Weights { buf, off } })
    }

    /// `y = x W^T + b` for a stored 1x1 convolution or Linear.
    fn linear(&self, key: &str, x: P, rows: usize, cin: usize, cout: usize, out: PM, ldo: usize, st: &Stream) -> Result<()> {
        gemm(st, x, cin, self.w.p(&format!("{key}.w")), self.w.p(&format!("{key}.b")), out, ldo, rows, cout, cin)
    }

    /// A stride 1, zero-padded k x k convolution of an NHWC map, written to `out` with row stride `ldo`.
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, key: &str, x: P, side: usize, cin: usize, kern: usize, cout: usize, out: PM, ldo: usize, st: &Stream) -> Result<()> {
        if kern == 1 {
            return self.linear(key, x, side * side, cin, cout, out, ldo, st);
        }
        let kd = kern * kern * cin;
        let rows = (COL_BUDGET / (side * kd * 4)).clamp(1, side);
        let col = buf(rows * side * kd, st)?;
        for y0 in (0..side).step_by(rows) {
            let r = rows.min(side - y0);
            k!(gpu_im2col_zero(x, side as i32, side as i32, cin as i32, kern as i32, (kern / 2) as i32, y0 as i32, r as i32, col.ptr(), s(st)))?;
            gemm(st, col.ptr(), kd, self.w.p(&format!("{key}.w")), self.w.p(&format!("{key}.b")), off(out, y0 * side * ldo * 4), ldo, r * side, cout, kd)?;
        }
        Ok(())
    }

    fn layer_norm(&self, x: P, rows: usize, c: usize, name: &str, out: PM, st: &Stream) -> Result<()> {
        k!(gpu_prologue_rows_b(DT_F32, PRO_LN_AFFINE, x, c as i64, self.w.p(&format!("{name}.weight")), self.w.p(&format!("{name}.bias")), LN_EPS,
            null(), null(), out, c as i64, rows as i32, c as i32, s(st)))
    }

    /// The Swin-L backbone on an NHWC image [side, side, 3]: the four stages' normed outputs.
    fn swin(&self, img: P, side: usize, st: &Stream) -> Result<Vec<Map>> {
        let mut h = side / 4;
        let patches = buf(h * h * 48, st)?;
        k!(gpu_patchify_nhwc(img, side as i32, side as i32, 3, 4, patches.ptr(), s(st)))?;
        let mut x = buf(h * h * EMBED, st)?;
        gemm(st, patches.ptr(), 48, self.w.p("bb.patch_embed.proj.w"), self.w.p("bb.patch_embed.proj.b"), x.ptr(), EMBED, h * h, EMBED, 48)?;
        drop(patches);
        let t = buf(h * h * EMBED, st)?;
        self.layer_norm(x.ptr(), h * h, EMBED, "bb.patch_embed.norm", t.ptr(), st)?;
        x = t;
        let mut outs = Vec::new();
        for (i, depth) in DEPTHS.iter().enumerate() {
            let c = EMBED << i;
            let heads = 6 << i;
            let rows = h * h;
            let hp = h.div_ceil(WS) * WS;
            let nw = (hp / WS) * (hp / WS);
            let win = buf(nw * WS * WS * c, st)?;
            let qkv = buf(nw * WS * WS * 3 * c, st)?;
            let norm = buf(rows * c, st)?;
            for j in 0..*depth {
                let b = format!("bb.layers.{i}.blocks.{j}");
                let shift = if j % 2 == 1 { WS / 2 } else { 0 };
                self.layer_norm(x.ptr(), rows, c, &format!("{b}.norm1"), norm.ptr(), st)?;
                k!(gpu_window_gather(norm.ptr(), h as i32, h as i32, c as i32, hp as i32, hp as i32, shift as i32, win.ptr(), s(st)))?;
                gemm(st, win.ptr(), c, self.w.p(&format!("{b}.attn.qkv.weight")), self.w.p(&format!("{b}.attn.qkv.bias")), qkv.ptr(), 3 * c, nw * WS * WS, 3 * c, c)?;
                k!(gpu_window_attn(qkv.ptr(), self.w.p(&format!("{b}.attn.relative_position_bias_table")), win.ptr(), nw as i32, c as i32, heads as i32,
                    hp as i32, hp as i32, shift as i32, s(st)))?;
                // the projection of the windows, into the qkv buffer's first part
                gemm(st, win.ptr(), c, self.w.p(&format!("{b}.attn.proj.weight")), self.w.p(&format!("{b}.attn.proj.bias")), qkv.ptr(), c, nw * WS * WS, c, c)?;
                k!(gpu_window_scatter_add(x.ptr(), qkv.ptr(), h as i32, h as i32, c as i32, hp as i32, hp as i32, shift as i32, s(st)))?;
                self.layer_norm(x.ptr(), rows, c, &format!("{b}.norm2"), norm.ptr(), st)?;
                let y = buf(rows * 4 * c, st)?;
                gemm(st, norm.ptr(), c, self.w.p(&format!("{b}.mlp.fc1.weight")), self.w.p(&format!("{b}.mlp.fc1.bias")), y.ptr(), 4 * c, rows, 4 * c, c)?;
                k!(gpu_gelu_f32(y.ptr(), (rows * 4 * c) as i64, 0, s(st)))?;
                gemm(st, y.ptr(), 4 * c, self.w.p(&format!("{b}.mlp.fc2.weight")), self.w.p(&format!("{b}.mlp.fc2.bias")), norm.ptr(), c, rows, c, 4 * c)?;
                k!(gpu_add_f32(x.ptr(), norm.ptr(), (rows * c) as i64, s(st)))?;
            }
            drop((win, qkv));
            let out = buf(rows * c, st)?;
            self.layer_norm(x.ptr(), rows, c, &format!("bb.norm{i}"), out.ptr(), st)?;
            outs.push(Map { buf: out, side: h, c });
            if i < 3 {
                let h2 = h / 2;
                let merged = buf(h2 * h2 * 4 * c, st)?;
                k!(gpu_patch_merge(x.ptr(), h as i32, h as i32, c as i32, merged.ptr(), s(st)))?;
                let normed = buf(h2 * h2 * 4 * c, st)?;
                self.layer_norm(merged.ptr(), h2 * h2, 4 * c, &format!("bb.layers.{i}.downsample.norm"), normed.ptr(), st)?;
                let nx = buf(h2 * h2 * 2 * c, st)?;
                let rw = self.w.p(&format!("bb.layers.{i}.downsample.reduction.weight"));
                gemm(st, normed.ptr(), 4 * c, rw, null(), nx.ptr(), 2 * c, h2 * h2, 2 * c, 4 * c)?;
                x = nx;
                h = h2;
            }
        }
        Ok(outs)
    }

    /// ASPPDeformable on an NHWC map [side * side, 64].
    fn aspp(&self, a: &str, x: P, side: usize, st: &Stream) -> Result<DevBuf> {
        const C: usize = 64;
        const BR: usize = 256;
        let px = side * side;
        let cat = buf(px * BR * 5, st)?;
        for (i, (name, kern)) in [("aspp1", 1usize), ("aspp_deforms.0", 1), ("aspp_deforms.1", 3), ("aspp_deforms.2", 7)].into_iter().enumerate() {
            let key = format!("{a}.{name}");
            let kk = kern * kern;
            let om = buf(px * 3 * kk, st)?;
            self.conv(&format!("{key}.om"), x, side, C, kern, 3 * kk, om.ptr(), 3 * kk, st)?;
            let rows = (COL_BUDGET / (side * kk * C * 4)).clamp(1, side);
            let col = buf(rows * side * kk * C, st)?;
            let dst = off(cat.ptr(), i * BR * 4);
            for y0 in (0..side).step_by(rows) {
                let r = rows.min(side - y0);
                k!(gpu_deform_im2col(x, om.ptr(), side as i32, side as i32, C as i32, kern as i32, (kern / 2) as i32, y0 as i32, r as i32, col.ptr(), s(st)))?;
                gemm(st, col.ptr(), kk * C, self.w.p(&format!("{key}.reg.w")), self.w.p(&format!("{key}.reg.b")), off(dst, y0 * side * 5 * BR * 4), 5 * BR,
                    r * side, BR, kk * C)?;
            }
            relu(dst, 5 * BR, BR, px, st)?;
        }
        // the global branch: mean over the map, 1x1 conv, upsampled from 1 x 1
        let pooled = buf(C, st)?;
        k!(gpu_avg_pool_nhwc_f32(x, side as i32, side as i32, C as i32, side as i32, pooled.ptr(), s(st)))?;
        let v = buf(BR, st)?;
        self.linear(&format!("{a}.pool"), pooled.ptr(), 1, C, BR, v.ptr(), BR, st)?;
        relu(v.ptr(), BR, BR, 1, st)?;
        k!(gpu_bcast_rows(v.ptr(), BR as i32, off(cat.ptr(), 4 * BR * 4), (5 * BR) as i64, px as i64, s(st)))?;
        let out = buf(px * C, st)?;
        self.linear(&format!("{a}.conv1"), cat.ptr(), px, 5 * BR, C, out.ptr(), C, st)?;
        relu(out.ptr(), C, C, px, st)?;
        Ok(out)
    }

    /// BasicDecBlk: conv_in + BN + ReLU, the ASPP, conv_out + BN.
    fn dec_blk(&self, b: &str, x: P, side: usize, cin: usize, cout: usize, out: PM, ldo: usize, st: &Stream) -> Result<()> {
        let px = side * side;
        let t = buf(px * 64, st)?;
        self.conv(&format!("{b}.conv_in"), x, side, cin, 3, 64, t.ptr(), 64, st)?;
        relu(t.ptr(), 64, 64, px, st)?;
        let a = self.aspp(&format!("{b}.dec_att"), t.ptr(), side, st)?;
        drop(t);
        self.conv(&format!("{b}.conv_out"), a.ptr(), side, 64, 3, cout, out, ldo, st)
    }

    /// `ipt_blk`: the input image cut into patches of the map's size, stacked as channels, through two 3x3 convs, written
    /// to `out` (row stride `ldo`).
    fn ipt(&self, name: &str, img: P, side: usize, cout: usize, out: PM, ldo: usize, st: &Stream) -> Result<()> {
        let g = SIZE / side;
        let cin = g * g * 3;
        let patches;
        let src = if g == 1 {
            img
        } else {
            patches = buf(side * side * cin, st)?;
            k!(gpu_split_patches(img, SIZE as i32, 3, side as i32, patches.ptr(), s(st)))?;
            patches.ptr()
        };
        let t = buf(side * side * 64, st)?;
        self.conv(&format!("decoder.ipt_blk{name}.conv1"), src, side, cin, 3, 64, t.ptr(), 64, st)?;
        self.conv(&format!("decoder.ipt_blk{name}.conv_out"), t.ptr(), side, 64, 3, cout, out, ldo, st)
    }

    /// `p *= sigmoid(conv1x1(relu(bn(conv3x3(p)))))`
    fn gdt(&self, i: usize, p: &DevBuf, side: usize, c: usize, st: &Stream) -> Result<()> {
        let px = side * side;
        let g = buf(px * 16, st)?;
        self.conv(&format!("decoder.gdt_convs_{i}"), p.ptr(), side, c, 3, 16, g.ptr(), 16, st)?;
        relu(g.ptr(), 16, 16, px, st)?;
        let a = buf(px, st)?;
        self.linear(&format!("decoder.gdt_attn_{i}"), g.ptr(), px, 16, 1, a.ptr(), 1, st)?;
        k!(gpu_act_rows(a.ptr(), 1, 1, px as i64, 1, s(st)))?;
        k!(gpu_mul_rows(p.ptr(), a.ptr(), c as i32, px as i64, s(st)))
    }

    /// The mask logits [SIZE * SIZE] of an image (f32 NHWC [SIZE, SIZE, 3], values in [0, 1]); `probe` sees the encoder's
    /// maps and the logits.
    pub fn logits(&self, img: &[f32], st: &Stream, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<Vec<f32>> {
        if img.len() != SIZE * SIZE * 3 {
            bail!("BiRefNet takes a {SIZE}x{SIZE} image");
        }
        let x = buf(img.len(), st)?;
        x.upload(0, &f32_bytes(img), st)?;
        let half = SIZE / 2;
        let small = buf(half * half * 3, st)?;
        resize(x.ptr(), SIZE, 3, small.ptr(), 3, half, st)?;
        let big = self.swin(x.ptr(), SIZE, st)?;
        let low = self.swin(small.ptr(), half, st)?;
        drop(small);

        // forward_enc: each stage next to the 512 pass's, resized to match
        if let Some(p) = probe.as_mut() {
            for (i, (a, b)) in big.iter().zip(&low).enumerate() {
                p(&format!("big{i}"), &a.buf, a.side * a.side * a.c);
                p(&format!("small{i}"), &b.buf, b.side * b.side * b.c);
            }
        }
        let mut enc = Vec::new();
        for (a, b) in big.into_iter().zip(low) {
            let c = 2 * a.c;
            let m = buf(a.side * a.side * c, st)?;
            put_cols(m.ptr(), c, 0, a.buf.ptr(), a.c, a.c, a.side * a.side, st)?;
            resize(b.buf.ptr(), b.side, b.c, off(m.ptr(), a.c * 4), c, a.side, st)?;
            enc.push(Map { buf: m, side: a.side, c });
        }
        let top = enc[3].side;
        let tc: usize = enc.iter().map(|m| m.c).sum();
        let x4 = buf(top * top * tc, st)?;
        let mut col = 0;
        for m in &enc[..3] {
            resize(m.buf.ptr(), m.side, m.c, off(x4.ptr(), col * 4), tc, top, st)?;
            col += m.c;
        }
        put_cols(x4.ptr(), tc, col, enc[3].buf.ptr(), enc[3].c, enc[3].c, top * top, st)?;
        if let Some(p) = probe.as_mut() {
            for (i, m) in enc.iter().enumerate() {
                p(&format!("enc{i}"), &m.buf, m.side * m.side * m.c);
            }
            p("x4", &x4, top * top * tc);
        }
        let c4 = enc[3].c;
        let sq = buf(top * top * c4, st)?;
        self.dec_blk("squeeze_module.0", x4.ptr(), top, tc, c4, sq.ptr(), c4, st)?;
        drop(x4);
        if let Some(p) = probe.as_mut() {
            p("squeeze", &sq, top * top * c4);
        }
        let lateral = [&enc[2], &enc[1], &enc[0]];

        // the decoder, coarse to fine. Each level stacks the image's patches (ipt_blk5 to 2) onto the map and runs a
        // decoder block; all but the last then gate the result and add the next encoder map (the lateral blocks).
        let ch = [c4, c4 / 2, c4 / 4, c4 / 8];
        let ipt_names = ["5", "4", "3", "2"];
        let ipt_channels = [ch[0] / 8, ch[0] / 8, ch[1] / 8, ch[2] / 8];
        let block_out = [ch[1], ch[2], ch[3], ch[3] / 2];
        let mut p = sq;
        let mut side = top;
        let mut pc = c4;
        for lvl in 0..4 {
            let (ic, out_c) = (ipt_channels[lvl], block_out[lvl]);
            let in_c = pc + ic;
            let cat = buf(side * side * in_c, st)?;
            put_cols(cat.ptr(), in_c, 0, p.ptr(), pc, pc, side * side, st)?;
            self.ipt(ipt_names[lvl], x.ptr(), side, ic, off(cat.ptr(), pc * 4), in_c, st)?;
            let q = buf(side * side * out_c, st)?;
            self.dec_blk(&format!("decoder.decoder_block{}", 4 - lvl), cat.ptr(), side, in_c, out_c, q.ptr(), out_c, st)?;
            drop((cat, p));
            pc = out_c;
            if lvl == 3 {
                p = q;
                break;
            }
            self.gdt(4 - lvl, &q, side, pc, st)?;
            let m = lateral[lvl];
            let up = buf(m.side * m.side * pc, st)?;
            resize(q.ptr(), side, pc, up.ptr(), pc, m.side, st)?;
            let lt = buf(m.side * m.side * m.c, st)?;
            self.linear(&format!("decoder.lateral_block{}", 4 - lvl), m.buf.ptr(), m.side * m.side, m.c, m.c, lt.ptr(), m.c, st)?;
            k!(gpu_add_f32(up.ptr(), lt.ptr(), (m.side * m.side * pc) as i64, s(st)))?;
            p = up;
            side = m.side;
        }
        // decoder_block1 left the grid of x1; back to the image, with the full-size patches' convs beside it
        let last_c = pc + ch[3] / 8;
        let last = buf(SIZE * SIZE * last_c, st)?;
        resize(p.ptr(), side, pc, last.ptr(), last_c, SIZE, st)?;
        drop(p);
        self.ipt("1", x.ptr(), SIZE, ch[3] / 8, off(last.ptr(), pc * 4), last_c, st)?;
        let out = buf(SIZE * SIZE, st)?;
        self.linear("decoder.conv_out1", last.ptr(), SIZE * SIZE, last_c, 1, out.ptr(), 1, st)?;
        if let Some(pr) = probe.as_mut() {
            pr("logits", &out, SIZE * SIZE);
        }
        let mut h = vec![0u8; SIZE * SIZE * 4];
        out.download(0, &mut h, st)?;
        st.sync()?;
        Ok(h.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

// ---------------------------------------------------------------------------------------------
// Resizes around the model (torch's, on the CPU)
// ---------------------------------------------------------------------------------------------

/// torch's anti-aliased bicubic taps (a = -0.5) for each output index: first input index and normalized weights.
fn aa_taps(inp: usize, out: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = inp as f32 / out as f32;
    let (support, inv) = if scale >= 1.0 { (2.0 * scale, 1.0 / scale) } else { (2.0, 1.0) };
    let filter = |x: f32| {
        let (a, x) = (-0.5f32, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    (0..out)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).min(inp as i64) as usize).saturating_sub(xmin);
            let mut w: Vec<f32> = (0..xmax).map(|j| filter((j as f32 + xmin as f32 - center + 0.5) * inv)).collect();
            let total: f32 = w.iter().sum();
            w.iter_mut().for_each(|x| *x /= total);
            (xmin, w)
        })
        .collect()
}

/// `F.interpolate(x, size, mode="bicubic", antialias=True)` of an HWC f32 image.
fn resize_aa(src: &[f32], w: usize, h: usize, c: usize, nw: usize, nh: usize) -> Vec<f32> {
    let (tx, ty) = (aa_taps(w, nw), aa_taps(h, nh));
    let mut mid = vec![0f32; h * nw * c];
    for y in 0..h {
        for (x, (x0, ws)) in tx.iter().enumerate() {
            for ch in 0..c {
                mid[(y * nw + x) * c + ch] = ws.iter().enumerate().map(|(j, wt)| wt * src[(y * w + x0 + j) * c + ch]).sum();
            }
        }
    }
    let mut out = vec![0f32; nh * nw * c];
    for (y, (y0, ws)) in ty.iter().enumerate() {
        for x in 0..nw {
            for ch in 0..c {
                out[(y * nw + x) * c + ch] = ws.iter().enumerate().map(|(j, wt)| wt * mid[((y0 + j) * nw + x) * c + ch]).sum();
            }
        }
    }
    out
}

/// `F.interpolate(x, size, mode="bicubic")` (a = -0.75, align_corners=False) of one f32 plane.
fn resize_bicubic(src: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let taps = |inp: usize, out: usize| -> Vec<([usize; 4], [f32; 4])> {
        let scale = inp as f32 / out as f32;
        (0..out)
            .map(|i| {
                let at = (i as f32 + 0.5) * scale - 0.5;
                let f = at.floor();
                let t = at - f;
                let idx = [-1i64, 0, 1, 2].map(|d| (f as i64 + d).clamp(0, inp as i64 - 1) as usize);
                (idx, [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)])
            })
            .collect()
    };
    let (tx, ty) = (taps(w, nw), taps(h, nh));
    let mut mid = vec![0f32; h * nw];
    for y in 0..h {
        for (x, (idx, cf)) in tx.iter().enumerate() {
            mid[y * nw + x] = (0..4).map(|k| cf[k] * src[y * w + idx[k]]).sum();
        }
    }
    let mut out = vec![0f32; nh * nw];
    for (y, (idx, cf)) in ty.iter().enumerate() {
        for x in 0..nw {
            out[y * nw + x] = (0..4).map(|k| cf[k] * mid[idx[k] * nw + x]).sum();
        }
    }
    out
}

/// `clip_preprocess(image, size=1024, crop=False)` with ComfyUI's mean 0 and std 1: the image (f32 RGB [h, w, 3]) at
/// the model's size, rounded to 8 bits.
pub fn preprocess(rgb: &[f32], w: usize, h: usize) -> Vec<f32> {
    let img = if w == SIZE && h == SIZE { rgb.to_vec() } else { resize_aa(rgb, w, h, 3, SIZE, SIZE) };
    img.into_iter().map(|v| (255.0 * v).clamp(0.0, 255.0).round_ties_even() / 255.0).collect()
}

/// The foreground mask [h * w] (1 on the object) of an image: the model's logits resized bicubically, then sigmoid.
pub fn mask_from_logits(logits: &[f32], w: usize, h: usize) -> Vec<f32> {
    resize_bicubic(logits, SIZE, SIZE, w, h).into_iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

/// `RemoveBackground` on images (f32 RGB [h, w, 3] each): their foreground masks [h * w].
pub fn remove_backgrounds(model: &Path, frames: &[Vec<f32>], w: usize, h: usize) -> Result<Vec<Vec<f32>>> {
    if w == 0 || h == 0 || frames.iter().any(|f| f.len() != w * h * 3) {
        bail!("BiRefNet needs RGB images of {w}x{h}");
    }
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let st = Stream::new()?;
    let net = BiRefNet::load(model, &st)?;
    let masks = frames
        .iter()
        .map(|rgb| Ok(mask_from_logits(&net.logits(&preprocess(rgb, w, h), &st, None)?, w, h)))
        .collect::<Result<Vec<_>>>()?;
    drop(net);
    st.sync()?;
    crate::cuda::trim_pool()?;
    Ok(masks)
}

/// `RemoveBackground` on one image.
pub fn remove_background(model: &Path, rgb: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
    Ok(remove_backgrounds(model, &[rgb.to_vec()], w, h)?.remove(0))
}

// ---------------------------------------------------------------------------------------------
// Parity with ComfyUI (tools/birefnet_ref.py)
// ---------------------------------------------------------------------------------------------

/// Compares each stage with the dump of `tools/birefnet_ref.py`, which also fixes the input.
pub fn check(model: &Path, ref_path: &Path) -> Result<()> {
    use crate::pixal3d::report;
    let r = SafeTensors::open(ref_path)?;
    let image = r.f32_vec("image", false)?;
    let shape = &r.info("image")?.shape;
    let (h, w) = (shape[1], shape[2]);
    report("preprocess", &preprocess(&image, w, h), &r.f32_vec("pixel_values", false)?);
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let st = Stream::new()?;
    let t0 = std::time::Instant::now();
    let net = BiRefNet::load(model, &st)?;
    eprintln!("load {:.2} s", t0.elapsed().as_secs_f64());
    let mut seen: Vec<(String, Vec<f32>)> = Vec::new();
    let mut probe = |name: &str, b: &DevBuf, n: usize| {
        let mut h = vec![0u8; n * 4];
        if b.download(0, &mut h, &st).is_ok() && st.sync().is_ok() {
            seen.push((name.to_string(), h.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        }
    };
    let t1 = std::time::Instant::now();
    let logits = net.logits(&r.f32_vec("pixel_values", false)?, &st, Some(&mut probe))?;
    eprintln!("forward {:.2} s", t1.elapsed().as_secs_f64());
    for (name, v) in &seen {
        let theirs = match name.strip_prefix("big").or(name.strip_prefix("small")) {
            Some(i) => r.f32_vec(&format!("bb_{}_{i}", if name.starts_with("big") { "big" } else { "small" }), false)?,
            None if name == "squeeze" => r.f32_vec("squeeze", false)?,
            None => continue,
        };
        report(name, v, &theirs);
    }
    report("logits", &logits, &r.f32_vec("logits", false)?);
    report("mask", &mask_from_logits(&logits, w, h), &r.f32_vec("mask", false)?);
    Ok(())
}
