//! The KL-f8 VAE of SD 1.x and SDXL (4-channel latents at /8), NHWC bf16 as ComfyUI runs it: the decoder, and the encoder
//! when the file has one. It is the Flux 2 VAE's architecture with a 4-channel latent and no batch norm, so this follows
//! [`crate::flux2vae`].

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, s, PM};
use crate::offload::Layout;
use crate::st::SafeTensors;
use crate::vae::{bias_pad, single_head_attention, RGB_PAD};
use anyhow::{bail, Context, Result};
use half::bf16;
use std::path::Path;

const OUT_PAD: usize = 8;
/// The latent's channels, padded for the convolutions.
const Z_PAD: usize = 8;
const GROUPS: usize = 32;
const EPS: f32 = 1e-6;
const LEVELS: usize = 4;
const BLOCKS: usize = 3;
pub const Z: usize = 4;
/// `SD15.scale_factor` and `SDXL.scale_factor`: the sampler's latent is the VAE's times this.
pub const SD1_SCALE: f32 = 0.18215;
pub const SDXL_SCALE: f32 = 0.13025;

pub struct Vae {
    buf: DevBuf,
    lay: Layout,
    /// (out channels, in channels of the first block) of each level
    levels: Vec<(usize, usize)>,
    block_in: usize,
    encoder: Option<Encoder>,
    pub scale: f32,
}

struct Encoder {
    /// (out channels, in channels of the first block, blocks) of each level
    levels: Vec<(usize, usize, usize)>,
    block_in: usize,
}

/// The prefix of the VAE's keys in `st`: `first_stage_model.` in a checkpoint, none in a VAE file.
pub fn key_prefix(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    ["first_stage_model.", "vae.", ""].into_iter().find(|p| has(&format!("{p}decoder.conv_in.weight")) && has(&format!("{p}decoder.conv_out.weight")))
}

/// Conv weight [Cout, Cin, R, R] -> bf16 [cout_pad, R, R, cin_pad], zero padded.
fn conv_pad(st: &SafeTensors, name: &str, cin_pad: usize, cout_pad: usize) -> Result<(Vec<u8>, usize)> {
    let info = st.info(name)?.clone();
    if info.shape.len() != 4 {
        bail!("{name}: conv weight of shape {:?}", info.shape);
    }
    let v = st.f32_vec(name, false)?;
    let (co, ci, r) = (info.shape[0], info.shape[1], info.shape[2]);
    let (cop, cip) = (cout_pad.max(co), cin_pad.max(ci));
    let mut out = vec![bf16::ZERO; cop * r * r * cip];
    for o in 0..co {
        for c in 0..ci {
            for y in 0..r {
                for x in 0..r {
                    out[((o * r + y) * r + x) * cip + c] = bf16::from_f32(v[((o * ci + c) * r + y) * r + x]);
                }
            }
        }
    }
    Ok((out.iter().flat_map(|b| b.to_le_bytes()).collect(), cop))
}

impl Vae {
    /// From a VAE file or a checkpoint; `scale` is the family's latent scale factor.
    pub fn load(path: &Path, scale: f32, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        Self::from(&st, scale, s0)
    }

    pub fn from(st: &SafeTensors, scale: f32, s0: &Stream) -> Result<Self> {
        let p = key_prefix(|k| st.has(k)).context("no SD VAE in this file")?;
        if st.info(&format!("{p}decoder.conv_in.weight"))?.shape[1] != Z {
            bail!("the VAE's latent isn't {Z} channels");
        }
        let n = |x: &str| format!("{p}{x}");
        let mut lay = Layout::new();
        let conv = |lay: &mut Layout, key: &str, name: &str, cin_pad: usize, cout_pad: usize| -> Result<()> {
            let (w, cop) = conv_pad(st, &n(&format!("{name}.weight")), cin_pad, cout_pad)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias_pad(st, &n(&format!("{name}.bias")), cop)?);
            Ok(())
        };
        let norm = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            lay.bf16(st, &format!("{key}.w"), &[n(&format!("{name}.weight"))])?;
            lay.bf16(st, &format!("{key}.b"), &[n(&format!("{name}.bias"))])
        };
        let res = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            norm(lay, &format!("{key}.n1"), &format!("{name}.norm1"))?;
            norm(lay, &format!("{key}.n2"), &format!("{name}.norm2"))?;
            conv(lay, &format!("{key}.c1"), &format!("{name}.conv1"), 0, 0)?;
            conv(lay, &format!("{key}.c2"), &format!("{name}.conv2"), 0, 0)?;
            for sc in ["nin_shortcut", "conv_shortcut"] {
                if st.has(&n(&format!("{name}.{sc}.weight"))) {
                    conv(lay, &format!("{key}.sc"), &format!("{name}.{sc}"), 0, 0)?;
                }
            }
            Ok(())
        };
        let attention = |lay: &mut Layout, key: &str, side: &str, diffusers: bool| -> Result<()> {
            let a = |comfy: &str, diff: &str| if diffusers { format!("{side}.mid_block.attentions.0.{diff}") } else { format!("{side}.mid.attn_1.{comfy}") };
            norm(lay, &format!("{key}.n"), &a("norm", "group_norm"))?;
            let (q, k_, v, o) = (a("q", "to_q"), a("k", "to_k"), a("v", "to_v"), a("proj_out", "to_out.0"));
            lay.bf16(st, &format!("{key}.qkv.w"), &[n(&format!("{q}.weight")), n(&format!("{k_}.weight")), n(&format!("{v}.weight"))])?;
            lay.bf16(st, &format!("{key}.qkv.b"), &[n(&format!("{q}.bias")), n(&format!("{k_}.bias")), n(&format!("{v}.bias"))])?;
            lay.bf16(st, &format!("{key}.proj.w"), &[n(&format!("{o}.weight"))])?;
            lay.bf16(st, &format!("{key}.proj.b"), &[n(&format!("{o}.bias"))])
        };
        // ComfyUI's names or diffusers'
        let diffusers = st.has(&n("decoder.mid_block.resnets.0.conv1.weight"));
        let mid = |side: &str, i: usize| if diffusers { format!("{side}.mid_block.resnets.{i}") } else { format!("{side}.mid.block_{}", i + 1) };
        let up = |i: usize| if diffusers { format!("decoder.up_blocks.{}", LEVELS - 1 - i) } else { format!("decoder.up.{i}") };
        let up_block = |i: usize, b: usize| if diffusers { format!("{}.resnets.{b}", up(i)) } else { format!("{}.block.{b}", up(i)) };
        conv(&mut lay, "pq", "post_quant_conv", Z_PAD, Z_PAD)?;
        conv(&mut lay, "in", "decoder.conv_in", Z_PAD, 0)?;
        res(&mut lay, "m1", &mid("decoder", 0))?;
        res(&mut lay, "m2", &mid("decoder", 1))?;
        attention(&mut lay, "attn", "decoder", diffusers)?;
        let mut levels = vec![(0, 0); LEVELS];
        let block_in = st.info(&n("decoder.conv_in.weight"))?.shape[0];
        for i in (0..LEVELS).rev() {
            let first = st.info(&n(&format!("{}.conv1.weight", up_block(i, 0))))?;
            levels[i] = (first.shape[0], first.shape[1]);
            for b in 0..BLOCKS {
                res(&mut lay, &format!("u{i}.{b}"), &up_block(i, b))?;
            }
            if i != 0 {
                let name = if diffusers { format!("{}.upsamplers.0.conv", up(i)) } else { format!("{}.upsample.conv", up(i)) };
                conv(&mut lay, &format!("u{i}.up"), &name, 0, 0)?;
            }
        }
        norm(&mut lay, "out.n", if diffusers { "decoder.conv_norm_out" } else { "decoder.norm_out" })?;
        conv(&mut lay, "out", "decoder.conv_out", 0, OUT_PAD)?;

        let encoder = if st.has(&n("encoder.conv_in.weight")) {
            let down = |i: usize| if diffusers { format!("encoder.down_blocks.{i}") } else { format!("encoder.down.{i}") };
            let block = |i: usize, b: usize| if diffusers { format!("{}.resnets.{b}", down(i)) } else { format!("{}.block.{b}", down(i)) };
            conv(&mut lay, "e.in", "encoder.conv_in", RGB_PAD, 0)?;
            let block_in = st.info(&n("encoder.conv_in.weight"))?.shape[0];
            let mut levels = Vec::with_capacity(LEVELS);
            for i in 0..LEVELS {
                let blocks = (0..).take_while(|&b| st.has(&n(&format!("{}.conv1.weight", block(i, b))))).count();
                if blocks == 0 {
                    bail!("{} has no blocks", down(i));
                }
                let first = st.info(&n(&format!("{}.conv1.weight", block(i, 0))))?;
                levels.push((first.shape[0], first.shape[1], blocks));
                for b in 0..blocks {
                    res(&mut lay, &format!("e{i}.{b}"), &block(i, b))?;
                }
                if i != LEVELS - 1 {
                    let name = if diffusers { format!("{}.downsamplers.0.conv", down(i)) } else { format!("{}.downsample.conv", down(i)) };
                    conv(&mut lay, &format!("e{i}.down"), &name, 0, 0)?;
                }
            }
            res(&mut lay, "em1", &mid("encoder", 0))?;
            res(&mut lay, "em2", &mid("encoder", 1))?;
            attention(&mut lay, "eattn", "encoder", diffusers)?;
            norm(&mut lay, "eout.n", if diffusers { "encoder.conv_norm_out" } else { "encoder.norm_out" })?;
            conv(&mut lay, "eout", "encoder.conv_out", 0, 0)?;
            conv(&mut lay, "equant", "quant_conv", 0, 0)?;
            if st.info(&n("encoder.conv_out.weight"))?.shape[0] != 2 * Z {
                bail!("the encoder doesn't output {} channels", 2 * Z);
            }
            Some(Encoder { levels, block_in })
        } else {
            None
        };
        let mut host = vec![0u8; lay.size];
        lay.pack(st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Vae { buf, lay, levels, block_in, encoder, scale })
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    fn conv(&self, key: &str, x: PM, out: PM, h: usize, w: usize, cin: usize, cout: usize, r: usize, st: &Stream) -> Result<()> {
        crate::vae::conv2d(x, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), out, h, w, cin, cout, r, st)
    }

    fn norm(&self, key: &str, x: PM, out: PM, stats: PM, p: usize, c: usize, silu: bool, st: &Stream) -> Result<()> {
        k!(gpu_group_norm(x, out, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), stats, p as i64, c as i32, GROUPS as i32, EPS,
            silu as i32, s(st)))
    }

    /// Peak activation memory (bytes) for decoding an h x w latent.
    pub fn activation_bytes(h: usize, w: usize) -> usize {
        let full = h * 8 * w * 8;
        full * 128 * 2 * 3 + full * 256 * 2 + (512 << 20)
    }

    fn res(&self, key: &str, x: DevBuf, h: usize, w: usize, cin: usize, cout: usize, stats: PM, st: &Stream) -> Result<DevBuf> {
        let p = h * w;
        let alloc = |c: usize| DevBuf::alloc_async(p * c * 2, st);
        let t = alloc(cin)?;
        self.norm(&format!("{key}.n1"), x.ptr(), t.ptr(), stats, p, cin, true, st)?;
        let u = alloc(cout)?;
        self.conv(&format!("{key}.c1"), t.ptr(), u.ptr(), h, w, cin, cout, 3, st)?;
        drop(t);
        let t2 = alloc(cout)?;
        self.norm(&format!("{key}.n2"), u.ptr(), t2.ptr(), stats, p, cout, true, st)?;
        self.conv(&format!("{key}.c2"), t2.ptr(), u.ptr(), h, w, cout, cout, 3, st)?;
        drop(t2);
        let residual = if cin != cout {
            let sc = alloc(cout)?;
            self.conv(&format!("{key}.sc"), x.ptr(), sc.ptr(), h, w, cin, cout, 1, st)?;
            sc
        } else {
            x
        };
        k!(gpu_add_gated_bf16(u.ptr(), cout as i64, residual.ptr(), cout as i64, null(), p as i32, cout as i32, s(st)))?;
        Ok(u)
    }

    fn attention(&self, key: &str, x: &DevBuf, p: usize, c: usize, stats: PM, st: &Stream) -> Result<()> {
        let n = DevBuf::alloc_async(p * c * 2, st)?;
        self.norm(&format!("{key}.n"), x.ptr(), n.ptr(), stats, p, c, false, st)?;
        let w = |x: &str| self.w(&format!("{key}.{x}"));
        single_head_attention(x, &n, p, c, [w("qkv.w"), w("qkv.b"), w("proj.w"), w("proj.b")], st)
    }

    /// RGB f32 [h, w, 3] (0 to 1, both sides multiples of 8) -> the sampler's latent f32 [4, h/8, w/8]: the encoder's
    /// mean times the scale factor (`process_latent_in`).
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize, st: &Stream) -> Result<Vec<f32>> {
        let Some(e) = &self.encoder else { bail!("this VAE file has no encoder") };
        if h % 8 != 0 || w % 8 != 0 || h == 0 || w == 0 || rgb.len() != h * w * 3 {
            bail!("can't encode {w}x{h} from {} values", rgb.len());
        }
        let alloc = |c: usize, a: usize, b: usize| DevBuf::alloc_async(a * b * c * 2, st);
        let stats = DevBuf::alloc_async(GROUPS * 8, st)?;
        let sp = stats.ptr();
        let mut px = vec![bf16::ZERO; h * w * RGB_PAD];
        for (o, v) in px.chunks_exact_mut(RGB_PAD).zip(rgb.chunks_exact(3)) {
            for c in 0..3 {
                o[c] = bf16::from_f32(v[c] * 2.0 - 1.0);
            }
        }
        let input = alloc(RGB_PAD, h, w)?;
        input.upload(0, &px.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>(), st)?;
        let mut c = e.block_in;
        let mut x = alloc(c, h, w)?;
        self.conv("e.in", input.ptr(), x.ptr(), h, w, RGB_PAD, c, 3, st)?;
        drop(input);
        let (mut hh, mut ww) = (h, w);
        for (i, &(cout, cin, blocks)) in e.levels.iter().enumerate() {
            for b in 0..blocks {
                let from = if b == 0 { cin } else { cout };
                x = self.res(&format!("e{i}.{b}"), x, hh, ww, from, cout, sp, st)?;
                c = cout;
            }
            if i != LEVELS - 1 {
                // zero padding on the right and bottom, then a stride 2 conv: the stride 1 conv's odd samples
                let full = alloc(c, hh, ww)?;
                self.conv(&format!("e{i}.down"), x.ptr(), full.ptr(), hh, ww, c, c, 3, st)?;
                drop(x);
                let y = alloc(c, hh / 2, ww / 2)?;
                k!(gpu_take_odd(full.ptr(), y.ptr(), hh as i32, ww as i32, c as i32, s(st)))?;
                x = y;
                hh /= 2;
                ww /= 2;
            }
        }
        x = self.res("em1", x, hh, ww, c, c, sp, st)?;
        let p = hh * ww;
        self.attention("eattn", &x, p, c, sp, st)?;
        x = self.res("em2", x, hh, ww, c, c, sp, st)?;
        let t = alloc(c, hh, ww)?;
        self.norm("eout.n", x.ptr(), t.ptr(), sp, p, c, true, st)?;
        let moments = alloc(2 * Z, hh, ww)?;
        self.conv("eout", t.ptr(), moments.ptr(), hh, ww, c, 2 * Z, 3, st)?;
        let q = alloc(2 * Z, hh, ww)?;
        self.conv("equant", moments.ptr(), q.ptr(), hh, ww, 2 * Z, 2 * Z, 1, st)?;
        let mut bytes = vec![0u8; p * 2 * Z * 2];
        q.download(0, &mut bytes, st)?;
        // the mean is the first Z channels
        let mut out = vec![0f32; Z * p];
        for (i, px) in bytes.chunks_exact(2 * Z * 2).enumerate() {
            for ch in 0..Z {
                out[ch * p + i] = bf16::from_le_bytes([px[ch * 2], px[ch * 2 + 1]]).to_f32() * self.scale;
            }
        }
        Ok(out)
    }

    /// The sampler's latent f32 [4, h, w] -> RGB8 [8h, 8w, 3]
    pub fn decode(&self, latent: &DevBuf, h: usize, w: usize, st: &Stream) -> Result<Vec<u8>> {
        let (mut hh, mut ww) = (h, w);
        let alloc = |c: usize, a: usize, b: usize| DevBuf::alloc_async(a * b * c * 2, st);
        let stats = DevBuf::alloc_async(GROUPS * 8, st)?;
        let sp = stats.ptr();
        let z = alloc(Z_PAD, hh, ww)?;
        k!(gpu_sd_vae_in(latent.ptr(), z.ptr(), Z as i32, (h * w) as i64, Z_PAD as i32, self.scale, s(st)))?;
        let z2 = alloc(Z_PAD, hh, ww)?;
        self.conv("pq", z.ptr(), z2.ptr(), hh, ww, Z_PAD, Z_PAD, 1, st)?;
        let mut c = self.block_in;
        let mut x = alloc(c, hh, ww)?;
        self.conv("in", z2.ptr(), x.ptr(), hh, ww, Z_PAD, c, 3, st)?;
        drop((z, z2));
        x = self.res("m1", x, hh, ww, c, c, sp, st)?;
        self.attention("attn", &x, hh * ww, c, sp, st)?;
        x = self.res("m2", x, hh, ww, c, c, sp, st)?;
        for i in (0..LEVELS).rev() {
            let (cout, cin) = self.levels[i];
            for b in 0..BLOCKS {
                let from = if b == 0 { cin } else { cout };
                x = self.res(&format!("u{i}.{b}"), x, hh, ww, from, cout, sp, st)?;
                c = cout;
            }
            if i != 0 {
                let up = alloc(c, 2 * hh, 2 * ww)?;
                k!(gpu_upsample2x(x.ptr(), up.ptr(), hh as i32, ww as i32, c as i32, s(st)))?;
                hh *= 2;
                ww *= 2;
                let y = alloc(c, hh, ww)?;
                self.conv(&format!("u{i}.up"), up.ptr(), y.ptr(), hh, ww, c, c, 3, st)?;
                x = y;
            }
        }
        let p = hh * ww;
        let t = alloc(c, hh, ww)?;
        self.norm("out.n", x.ptr(), t.ptr(), sp, p, c, true, st)?;
        let o = alloc(OUT_PAD, hh, ww)?;
        self.conv("out", t.ptr(), o.ptr(), hh, ww, c, OUT_PAD, 3, st)?;
        let rgb = DevBuf::alloc_async(p * 3, st)?;
        k!(gpu_to_rgb8(o.ptr(), OUT_PAD as i32, rgb.ptr(), p as i64, s(st)))?;
        let mut out = vec![0u8; p * 3];
        rgb.download(0, &mut out, st)?;
        Ok(out)
    }
}
