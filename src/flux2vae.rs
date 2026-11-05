//! Flux 2 VAE for MOSS Image's 128-channel latents, NHWC bf16: the decoder, and the encoder when the file has one.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, s, PM};
use crate::offload::Layout;
use crate::st::{f32_bytes, SafeTensors};
use crate::vae::{bias_pad, conv_krsc, conv_krsc_rgb, single_head_attention, RGB_PAD};
use anyhow::{bail, Result};
use half::bf16;
use std::path::Path;

const OUT_PAD: usize = 8;
const GROUPS: usize = 32;
const EPS: f32 = 1e-6;
const BN_EPS: f32 = 1e-4;
const LEVELS: usize = 4;
const BLOCKS: usize = 3;
const Z: usize = 32;
pub const CH: usize = 128;

pub struct Vae {
    buf: DevBuf,
    lay: Layout,
    /// (out channels, in channels of the first block) of each level
    levels: Vec<(usize, usize)>,
    block_in: usize,
    encoder: Option<Encoder>,
    /// the batch norm's mean and standard deviation, for the encoder's output
    bn: (Vec<f32>, Vec<f32>),
}

struct Encoder {
    /// (out channels, in channels of the first block, blocks) of each level
    levels: Vec<(usize, usize, usize)>,
    block_in: usize,
}

impl Vae {
    pub fn load(path: &Path, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let mut lay = Layout::new();
        let conv = |lay: &mut Layout, key: &str, name: &str, pad: usize| -> Result<()> {
            let (w, cop, _, _) = conv_krsc(&st, &format!("{name}.weight"), pad)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias_pad(&st, &format!("{name}.bias"), cop)?);
            Ok(())
        };
        let norm = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            lay.bf16(&st, &format!("{key}.w"), &[format!("{name}.weight")])?;
            lay.bf16(&st, &format!("{key}.b"), &[format!("{name}.bias")])
        };
        let res = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            norm(lay, &format!("{key}.n1"), &format!("{name}.norm1"))?;
            norm(lay, &format!("{key}.n2"), &format!("{name}.norm2"))?;
            conv(lay, &format!("{key}.c1"), &format!("{name}.conv1"), 0)?;
            conv(lay, &format!("{key}.c2"), &format!("{name}.conv2"), 0)?;
            for sc in ["nin_shortcut", "conv_shortcut"] {
                if st.has(&format!("{name}.{sc}.weight")) {
                    conv(lay, &format!("{key}.sc"), &format!("{name}.{sc}"), 0)?;
                }
            }
            Ok(())
        };
        // Published copies of this VAE use either ComfyUI's names or diffusers'.
        let diffusers = st.has("decoder.mid_block.resnets.0.conv1.weight");
        let mid = |i: usize| if diffusers { format!("decoder.mid_block.resnets.{i}") } else { format!("decoder.mid.block_{}", i + 1) };
        let attn = |comfy: &str, diff: &str| {
            if diffusers { format!("decoder.mid_block.attentions.0.{diff}") } else { format!("decoder.mid.attn_1.{comfy}") }
        };
        let up = |i: usize| if diffusers { format!("decoder.up_blocks.{}", LEVELS - 1 - i) } else { format!("decoder.up.{i}") };
        let up_block = |i: usize, b: usize| if diffusers { format!("{}.resnets.{b}", up(i)) } else { format!("{}.block.{b}", up(i)) };
        let pq = if st.has("decoder.post_quant_conv.weight") { "decoder.post_quant_conv" } else { "post_quant_conv" };
        conv(&mut lay, "pq", pq, 0)?;
        conv(&mut lay, "in", "decoder.conv_in", 0)?;
        res(&mut lay, "m1", &mid(0))?;
        res(&mut lay, "m2", &mid(1))?;
        norm(&mut lay, "attn.n", &attn("norm", "group_norm"))?;
        let (q, k_, v, o) = (attn("q", "to_q"), attn("k", "to_k"), attn("v", "to_v"), attn("proj_out", "to_out.0"));
        lay.bf16(&st, "attn.qkv.w", &[format!("{q}.weight"), format!("{k_}.weight"), format!("{v}.weight")])?;
        lay.bf16(&st, "attn.qkv.b", &[format!("{q}.bias"), format!("{k_}.bias"), format!("{v}.bias")])?;
        lay.bf16(&st, "attn.proj.w", &[format!("{o}.weight")])?;
        lay.bf16(&st, "attn.proj.b", &[format!("{o}.bias")])?;
        let mut levels = vec![(0, 0); LEVELS];
        let block_in = st.info("decoder.conv_in.weight")?.shape[0];
        for i in (0..LEVELS).rev() {
            let first = st.info(&format!("{}.conv1.weight", up_block(i, 0)))?;
            levels[i] = (first.shape[0], first.shape[1]);
            for b in 0..BLOCKS {
                res(&mut lay, &format!("u{i}.{b}"), &up_block(i, b))?;
            }
            if i != 0 {
                let name = if diffusers { format!("{}.upsamplers.0.conv", up(i)) } else { format!("{}.upsample.conv", up(i)) };
                conv(&mut lay, &format!("u{i}.up"), &name, 0)?;
            }
        }
        let encoder = if st.has("encoder.conv_in.weight") { Some(Self::load_encoder(&st, &mut lay)?) } else { None };
        norm(&mut lay, "out.n", if diffusers { "decoder.conv_norm_out" } else { "decoder.norm_out" })?;
        conv(&mut lay, "out", "decoder.conv_out", OUT_PAD)?;
        let m = st.f32_vec("bn.running_mean", false)?;
        let sd: Vec<f32> = st.f32_vec("bn.running_var", false)?.iter().map(|v| (v + BN_EPS).sqrt()).collect();
        lay.bytes("mean", f32_bytes(&m));
        lay.bytes("std", f32_bytes(&sd));
        let mut host = vec![0u8; lay.size];
        lay.pack(&st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Vae { buf, lay, levels, block_in, encoder, bn: (m, sd) })
    }

    fn load_encoder(st: &SafeTensors, lay: &mut Layout) -> Result<Encoder> {
        let conv = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            let (w, cop, _, _) = conv_krsc(st, &format!("{name}.weight"), 0)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias_pad(st, &format!("{name}.bias"), cop)?);
            Ok(())
        };
        let norm = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            lay.bf16(st, &format!("{key}.w"), &[format!("{name}.weight")])?;
            lay.bf16(st, &format!("{key}.b"), &[format!("{name}.bias")])
        };
        let res = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            norm(lay, &format!("{key}.n1"), &format!("{name}.norm1"))?;
            norm(lay, &format!("{key}.n2"), &format!("{name}.norm2"))?;
            conv(lay, &format!("{key}.c1"), &format!("{name}.conv1"))?;
            conv(lay, &format!("{key}.c2"), &format!("{name}.conv2"))?;
            for sc in ["nin_shortcut", "conv_shortcut"] {
                if st.has(&format!("{name}.{sc}.weight")) {
                    conv(lay, &format!("{key}.sc"), &format!("{name}.{sc}"))?;
                }
            }
            Ok(())
        };
        // the same two naming schemes as the decoder
        let diffusers = st.has("encoder.mid_block.resnets.0.conv1.weight");
        let mid = |i: usize| if diffusers { format!("encoder.mid_block.resnets.{i}") } else { format!("encoder.mid.block_{}", i + 1) };
        let attn = |comfy: &str, diff: &str| {
            if diffusers { format!("encoder.mid_block.attentions.0.{diff}") } else { format!("encoder.mid.attn_1.{comfy}") }
        };
        let down = |i: usize| if diffusers { format!("encoder.down_blocks.{i}") } else { format!("encoder.down.{i}") };
        let block = |i: usize, b: usize| if diffusers { format!("{}.resnets.{b}", down(i)) } else { format!("{}.block.{b}", down(i)) };
        lay.bytes("e.in.w", conv_krsc_rgb(st, "encoder.conv_in.weight")?);
        lay.bytes("e.in.b", bias_pad(st, "encoder.conv_in.bias", 0)?);
        let block_in = st.info("encoder.conv_in.weight")?.shape[0];
        let mut levels = Vec::with_capacity(LEVELS);
        for i in 0..LEVELS {
            let blocks = (0..).take_while(|&b| st.has(&format!("{}.conv1.weight", block(i, b)))).count();
            if blocks == 0 {
                bail!("{} has no blocks", down(i));
            }
            let first = st.info(&format!("{}.conv1.weight", block(i, 0)))?;
            levels.push((first.shape[0], first.shape[1], blocks));
            for b in 0..blocks {
                res(lay, &format!("e{i}.{b}"), &block(i, b))?;
            }
            if i != LEVELS - 1 {
                let name = if diffusers { format!("{}.downsamplers.0.conv", down(i)) } else { format!("{}.downsample.conv", down(i)) };
                conv(lay, &format!("e{i}.down"), &name)?;
            }
        }
        res(lay, "em1", &mid(0))?;
        res(lay, "em2", &mid(1))?;
        norm(lay, "eattn.n", &attn("norm", "group_norm"))?;
        let (q, k_, v, o) = (attn("q", "to_q"), attn("k", "to_k"), attn("v", "to_v"), attn("proj_out", "to_out.0"));
        lay.bf16(st, "eattn.qkv.w", &[format!("{q}.weight"), format!("{k_}.weight"), format!("{v}.weight")])?;
        lay.bf16(st, "eattn.qkv.b", &[format!("{q}.bias"), format!("{k_}.bias"), format!("{v}.bias")])?;
        lay.bf16(st, "eattn.proj.w", &[format!("{o}.weight")])?;
        lay.bf16(st, "eattn.proj.b", &[format!("{o}.bias")])?;
        norm(lay, "eout.n", if diffusers { "encoder.conv_norm_out" } else { "encoder.norm_out" })?;
        conv(lay, "eout", "encoder.conv_out")?;
        let quant = ["encoder.quant_conv", "quant_conv"].into_iter().find(|n| st.has(&format!("{n}.weight"))).unwrap_or("quant_conv");
        conv(lay, "equant", quant)?;
        if st.info("encoder.conv_out.weight")?.shape[0] != 2 * Z {
            bail!("the encoder doesn't output {} channels", 2 * Z);
        }
        Ok(Encoder { levels, block_in })
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    fn conv(&self, key: &str, x: PM, out: PM, h: usize, w: usize, cin: usize, cout: usize, r: usize, st: &Stream) -> Result<()> {
        k!(gpu_conv2d(x, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), out, 1, h as i32, w as i32, cin as i32, cout as i32,
            r as i32, s(st)))
    }

    fn norm(&self, key: &str, x: PM, out: PM, stats: PM, p: usize, c: usize, silu: bool, st: &Stream) -> Result<()> {
        k!(gpu_group_norm(x, out, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), stats, p as i64, c as i32, GROUPS as i32, EPS,
            silu as i32, s(st)))
    }

    /// Peak activation memory (bytes) for decoding an h x w latent.
    pub fn activation_bytes(h: usize, w: usize) -> usize {
        let full = h * 16 * w * 16;
        full * CH * 2 * 3 + full * 512 * 2 / 2 + (512 << 20)
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

    /// RGB f32 [h, w, 3] (0 to 1, both sides multiples of 16) -> latent f32 [128, h/16, w/16]: the encoder's mean,
    /// patched 2x2 and batch-normalized, as ComfyUI's `AutoencodingEngine.encode` does for this VAE.
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize, st: &Stream) -> Result<Vec<f32>> {
        let Some(e) = &self.encoder else { bail!("this VAE file has no encoder") };
        if h % 16 != 0 || w % 16 != 0 || h == 0 || w == 0 || rgb.len() != h * w * 3 {
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
                debug_assert_eq!(from, c);
                x = self.res(&format!("e{i}.{b}"), x, hh, ww, from, cout, sp, st)?;
                c = cout;
            }
            if i != LEVELS - 1 {
                // zero padding on the right and bottom, then a stride 2 conv: a stride 1 conv pads the same zeros,
                // and holds the stride 2 samples at odd positions
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
        let n = alloc(c, hh, ww)?;
        self.norm("eattn.n", x.ptr(), n.ptr(), sp, p, c, false, st)?;
        single_head_attention(&x, &n, p, c, [self.w("eattn.qkv.w"), self.w("eattn.qkv.b"), self.w("eattn.proj.w"), self.w("eattn.proj.b")], st)?;
        drop(n);
        x = self.res("em2", x, hh, ww, c, c, sp, st)?;
        let t = alloc(c, hh, ww)?;
        self.norm("eout.n", x.ptr(), t.ptr(), sp, p, c, true, st)?;
        let moments = alloc(2 * Z, hh, ww)?;
        self.conv("eout", t.ptr(), moments.ptr(), hh, ww, c, 2 * Z, 3, st)?;
        let q = alloc(2 * Z, hh, ww)?;
        self.conv("equant", moments.ptr(), q.ptr(), hh, ww, 2 * Z, 2 * Z, 1, st)?;
        let mut bytes = vec![0u8; p * 2 * Z * 2];
        q.download(0, &mut bytes, st)?;
        // the mean is the first Z channels, and (c, 2i + pi, 2j + pj) goes to (4c + 2pi + pj, i, j)
        let (lh, lw) = (hh / 2, ww / 2);
        let mut out = vec![0f32; CH * lh * lw];
        for y in 0..hh {
            for xx in 0..ww {
                let at = (y * ww + xx) * 2 * Z;
                for ch in 0..Z {
                    let v = bf16::from_le_bytes([bytes[(at + ch) * 2], bytes[(at + ch) * 2 + 1]]).to_f32();
                    let k = ch * 4 + (y % 2) * 2 + xx % 2;
                    out[(k * lh + y / 2) * lw + xx / 2] = (v - self.bn.0[k]) / self.bn.1[k];
                }
            }
        }
        Ok(out)
    }

    /// latent f32 [128, h, w] -> RGB8 [16h, 16w, 3]
    pub fn decode(&self, latent: &DevBuf, h: usize, w: usize, st: &Stream) -> Result<Vec<u8>> {
        let (mut hh, mut ww) = (h * 2, w * 2);
        let alloc = |c: usize, a: usize, b: usize| DevBuf::alloc_async(a * b * c * 2, st);
        let stats = DevBuf::alloc_async(GROUPS * 8, st)?;
        let sp = stats.ptr();
        let z = alloc(Z, hh, ww)?;
        k!(gpu_flux2_latent_in(latent.ptr(), self.w("mean"), self.w("std"), z.ptr(), CH as i32, h as i32, w as i32, s(st)))?;
        let z2 = alloc(Z, hh, ww)?;
        self.conv("pq", z.ptr(), z2.ptr(), hh, ww, Z, Z, 1, st)?;
        let mut c = self.block_in;
        let mut x = alloc(c, hh, ww)?;
        self.conv("in", z2.ptr(), x.ptr(), hh, ww, Z, c, 3, st)?;
        drop((z, z2));

        x = self.res("m1", x, hh, ww, c, c, sp, st)?;
        let p = hh * ww;
        let n = alloc(c, hh, ww)?;
        self.norm("attn.n", x.ptr(), n.ptr(), sp, p, c, false, st)?;
        single_head_attention(&x, &n, p, c, [self.w("attn.qkv.w"), self.w("attn.qkv.b"), self.w("attn.proj.w"), self.w("attn.proj.b")], st)?;
        drop(n);
        x = self.res("m2", x, hh, ww, c, c, sp, st)?;

        for i in (0..LEVELS).rev() {
            let (cout, cin) = self.levels[i];
            for b in 0..BLOCKS {
                let from = if b == 0 { cin } else { cout };
                debug_assert_eq!(from, c);
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

/// Compare the encoder with a dump of `tools/flux2_vae_ref.py`, and both latents decoded against the picture.
pub fn check(model: &Path, reference: &Path) -> Result<()> {
    use crate::pixal3d::report;
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let r = SafeTensors::open(reference)?;
    let st = Stream::new()?;
    let vae = Vae::load(model, &st)?;
    let shape = r.info("image0")?.shape.clone();
    let (h, w) = (shape[1], shape[2]);
    let image = r.f32_vec("image0", false)?;
    let t0 = std::time::Instant::now();
    let z = vae.encode(&image, h, w, &st)?;
    st.sync()?;
    eprintln!("encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    let theirs = r.f32_vec("latent0", false)?;
    report("encode", &z, &theirs);
    let decode = |z: &[f32]| -> Result<Vec<f32>> {
        let d = DevBuf::alloc(z.len() * 4)?;
        d.upload(0, &f32_bytes(z), &st)?;
        let px = vae.decode(&d, h / 16, w / 16, &st)?;
        Ok(px.iter().map(|v| *v as f32 / 255.0).collect())
    };
    let (ours, comfy) = (decode(&z)?, decode(&theirs)?);
    let psnr = |a: &[f32], b: &[f32]| 10.0 * (a.len() as f64 / a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>()).log10();
    eprintln!("decoded here: our latent vs the picture {:.2} dB, ComfyUI's latent vs the picture {:.2} dB, ours vs ComfyUI's {:.2} dB",
        psnr(&ours, &image), psnr(&comfy, &image), psnr(&ours, &comfy));
    if r.has("back0") {
        eprintln!("ComfyUI's round trip vs the picture {:.2} dB", psnr(&r.f32_vec("back0", false)?, &image));
    }
    Ok(())
}
