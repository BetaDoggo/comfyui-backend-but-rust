//! SeedVR2's causal video VAE for single images, NHWC bf16.
//!
//! On one frame every causal conv sees its input repeated in front of it (ComfyUI pads the time axis with copies of the
//! first frame), so it is a 2D conv with the kernel's temporal taps summed; the temporal up and down samplers keep one
//! frame. The decoder's upsampler is a 1x1 expansion to four phases, shuffled into pixels, then a 3x3 conv.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, s, DT_BF16, PM};
use crate::offload::Layout;
use crate::st::{f32_bytes, SafeTensors};
use crate::vae::{single_head_attention, RGB_PAD};
use anyhow::{bail, Result};
use half::bf16;
use std::path::Path;

const OUT_PAD: usize = 8;
const GROUPS: usize = 32;
const EPS: f32 = 1e-6;
/// `BYTEDANCE_VAE_SCALING_FACTOR`: the sampler's latent is the encoder's mean times this.
pub const SCALE: f32 = 0.9152;
pub const Z: usize = 16;
const LEVELS: usize = 4;
/// Most elements of a conv's input or output in one `gpu_conv2d` call.
const CONV_ELEMENTS: usize = 1 << 29;

pub struct Vae {
    buf: DevBuf,
    lay: Layout,
    /// Encoder: (out channels, in channels, resnets) per level.
    down: Vec<(usize, usize, usize)>,
    /// Decoder: (out channels, in channels, resnets) per level, from the bottom.
    up: Vec<(usize, usize, usize)>,
    e_in: usize,
    d_in: usize,
}

/// Conv weight [Cout, Cin, T, R, R] -> KRSC bf16 with the temporal taps summed, Cout zero-padded to `cout_pad` and Cin
/// to `cin_pad`. Returns the weight and (padded Cout, Cin, R).
fn conv_summed(st: &SafeTensors, name: &str, cout_pad: usize, cin_pad: usize) -> Result<(Vec<u8>, usize, usize, usize)> {
    let info = st.info(name)?.clone();
    let v = st.f32_vec(name, false)?;
    let (co, ci) = (info.shape[0], info.shape[1]);
    let (t, r) = match info.shape.len() {
        5 => (info.shape[2], info.shape[3]),
        4 => (1, info.shape[2]),
        _ => bail!("{name}: unexpected conv shape {:?}", info.shape),
    };
    let (cop, cip) = (cout_pad.max(co), cin_pad.max(ci));
    let mut out = vec![bf16::ZERO; cop * r * r * cip];
    for o in 0..co {
        for c in 0..ci {
            for y in 0..r {
                for x in 0..r {
                    let sum: f32 = (0..t).map(|tt| v[(((o * ci + c) * t + tt) * r + y) * r + x]).sum();
                    out[((o * r + y) * r + x) * cip + c] = bf16::from_f32(sum);
                }
            }
        }
    }
    Ok((out.iter().flat_map(|b| b.to_le_bytes()).collect(), cop, ci, r))
}

fn bias(st: &SafeTensors, name: &str, pad: usize) -> Result<Vec<u8>> {
    let mut v = st.f32_vec(name, false)?;
    v.resize(pad.max(v.len()), 0.0);
    Ok(v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect())
}

impl Vae {
    pub fn load(path: &Path, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        if !crate::seedvr::is_seedvr2_vae(|k| st.has(k)) {
            bail!("{} is not SeedVR2's VAE", path.display());
        }
        let mut lay = Layout::new();
        let conv = |lay: &mut Layout, key: &str, name: &str, pad: usize, cin_pad: usize| -> Result<()> {
            let (w, cop, _, _) = conv_summed(&st, &format!("{name}.weight"), pad, cin_pad)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias(&st, &format!("{name}.bias"), cop)?);
            Ok(())
        };
        let norm = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            lay.bf16(&st, &format!("{key}.w"), &[format!("{name}.weight")])?;
            lay.bf16(&st, &format!("{key}.b"), &[format!("{name}.bias")])
        };
        let res = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            norm(lay, &format!("{key}.n1"), &format!("{name}.norm1"))?;
            norm(lay, &format!("{key}.n2"), &format!("{name}.norm2"))?;
            conv(lay, &format!("{key}.c1"), &format!("{name}.conv1"), 0, 0)?;
            conv(lay, &format!("{key}.c2"), &format!("{name}.conv2"), 0, 0)?;
            if st.has(&format!("{name}.conv_shortcut.weight")) {
                conv(lay, &format!("{key}.sc"), &format!("{name}.conv_shortcut"), 0, 0)?;
            }
            Ok(())
        };
        let attn = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            norm(lay, &format!("{key}.n"), &format!("{name}.group_norm"))?;
            lay.bf16(&st, &format!("{key}.qkv.w"), &[format!("{name}.to_q.weight"), format!("{name}.to_k.weight"), format!("{name}.to_v.weight")])?;
            lay.bf16(&st, &format!("{key}.qkv.b"), &[format!("{name}.to_q.bias"), format!("{name}.to_k.bias"), format!("{name}.to_v.bias")])?;
            lay.bf16(&st, &format!("{key}.proj.w"), &[format!("{name}.to_out.0.weight")])?;
            lay.bf16(&st, &format!("{key}.proj.b"), &[format!("{name}.to_out.0.bias")])
        };
        let count = |prefix: &str| (0..).take_while(|b| st.has(&format!("{prefix}.resnets.{b}.conv1.weight"))).count();

        // encoder
        conv(&mut lay, "e.in", "encoder.conv_in", 0, RGB_PAD)?;
        let e_in = st.info("encoder.conv_in.weight")?.shape[0];
        let mut down = Vec::with_capacity(LEVELS);
        for i in 0..LEVELS {
            let p = format!("encoder.down_blocks.{i}");
            let n = count(&p);
            if n == 0 {
                bail!("{p} has no resnets");
            }
            let first = st.info(&format!("{p}.resnets.0.conv1.weight"))?;
            down.push((first.shape[0], first.shape[1], n));
            for b in 0..n {
                res(&mut lay, &format!("e{i}.{b}"), &format!("{p}.resnets.{b}"))?;
            }
            if st.has(&format!("{p}.downsamplers.0.conv.weight")) != (i + 1 < LEVELS) {
                bail!("{p}: unexpected downsampler");
            }
            if i + 1 < LEVELS {
                conv(&mut lay, &format!("e{i}.down"), &format!("{p}.downsamplers.0.conv"), 0, 0)?;
            }
        }
        res(&mut lay, "em0", "encoder.mid_block.resnets.0")?;
        attn(&mut lay, "eattn", "encoder.mid_block.attentions.0")?;
        res(&mut lay, "em1", "encoder.mid_block.resnets.1")?;
        norm(&mut lay, "eout.n", "encoder.conv_norm_out")?;
        conv(&mut lay, "eout", "encoder.conv_out", 0, 0)?;
        if st.info("encoder.conv_out.weight")?.shape[0] != 2 * Z {
            bail!("the encoder doesn't output {} channels", 2 * Z);
        }

        // decoder
        conv(&mut lay, "d.in", "decoder.conv_in", 0, 0)?;
        let d_in = st.info("decoder.conv_in.weight")?.shape[0];
        res(&mut lay, "dm0", "decoder.mid_block.resnets.0")?;
        attn(&mut lay, "dattn", "decoder.mid_block.attentions.0")?;
        res(&mut lay, "dm1", "decoder.mid_block.resnets.1")?;
        let mut up = Vec::with_capacity(LEVELS);
        for i in 0..LEVELS {
            let p = format!("decoder.up_blocks.{i}");
            let n = count(&p);
            let first = st.info(&format!("{p}.resnets.0.conv1.weight"))?;
            let c = first.shape[0];
            up.push((c, first.shape[1], n));
            for b in 0..n {
                res(&mut lay, &format!("u{i}.{b}"), &format!("{p}.resnets.{b}"))?;
            }
            let u = format!("{p}.upsamplers.0");
            if st.has(&format!("{u}.upscale_conv.weight")) != (i + 1 < LEVELS) {
                bail!("{p}: unexpected upsampler");
            }
            if i + 1 < LEVELS {
                // the 1x1 expansion's channels are (i, j, t, c); the first frame keeps phase t = 0
                let w = st.f32_vec(&format!("{u}.upscale_conv.weight"), false)?;
                let b = st.f32_vec(&format!("{u}.upscale_conv.bias"), false)?;
                let rows = st.info(&format!("{u}.upscale_conv.weight"))?.shape[0];
                if rows % (4 * c) != 0 || w.len() != rows * c {
                    bail!("{u}.upscale_conv: {rows} outputs for {c} channels");
                }
                let tr = rows / (4 * c);
                let (mut ew, mut eb) = (Vec::with_capacity(4 * c * c), Vec::with_capacity(4 * c));
                for ph in 0..4 {
                    for ch in 0..c {
                        let k = (ph * tr) * c + ch;
                        ew.extend_from_slice(&w[k * c..(k + 1) * c]);
                        eb.push(b[k]);
                    }
                }
                let to_bf16 = |v: &[f32]| v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect::<Vec<u8>>();
                lay.bytes(&format!("u{i}.exp.w"), to_bf16(&ew));
                lay.bytes(&format!("u{i}.exp.b"), to_bf16(&eb));
                conv(&mut lay, &format!("u{i}.up"), &format!("{u}.conv"), 0, 0)?;
            }
        }
        norm(&mut lay, "dout.n", "decoder.conv_norm_out")?;
        conv(&mut lay, "dout", "decoder.conv_out", OUT_PAD, 0)?;
        lay.bytes("lat.mean", f32_bytes(&[0.0; Z]));
        lay.bytes("lat.std", f32_bytes(&[1.0 / SCALE; Z]));

        let mut host = vec![0u8; lay.size];
        lay.pack(&st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Vae { buf, lay, down, up, e_in, d_in })
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, key: &str, x: PM, out: PM, h: usize, w: usize, cin: usize, cout: usize, r: usize, st: &Stream) -> Result<()> {
        let (wk, bk) = (self.w(&format!("{key}.w")), self.w(&format!("{key}.b")));
        let run = |x: PM, out: PM, rows: usize| -> Result<()> {
            k!(gpu_conv2d(x, wk, bk, out, 1, rows as i32, w as i32, cin as i32, cout as i32, r as i32, s(st)))
        };
        // CUTLASS's conv indexes with 32 bits: a large picture goes through in bands of rows, each with a row of halo
        // on either side whose outputs are dropped
        let halo = r / 2;
        let band = (CONV_ELEMENTS / (w * cin.max(cout))).saturating_sub(2 * halo).max(1);
        if band >= h {
            return run(x, out, h);
        }
        let tmp = DevBuf::alloc_async((band + 2 * halo) * w * cout * 2, st)?;
        for y0 in (0..h).step_by(band) {
            let y1 = (y0 + band).min(h);
            let (a, b) = (y0.saturating_sub(halo), (y1 + halo).min(h));
            run(crate::ffi::off(x, a * w * cin * 2), tmp.ptr(), b - a)?;
            crate::cuda::memcpy_d2d(crate::ffi::off(out, y0 * w * cout * 2), crate::ffi::off(tmp.ptr(), (y0 - a) * w * cout * 2), (y1 - y0) * w * cout * 2,
                st)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(&self, key: &str, x: PM, out: PM, stats: PM, p: usize, c: usize, silu: bool, st: &Stream) -> Result<()> {
        k!(gpu_group_norm(x, out, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), stats, p as i64, c as i32, GROUPS as i32, EPS,
            silu as i32, s(st)))
    }

    #[allow(clippy::too_many_arguments)]
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

    fn mid(&self, pre: &str, x: DevBuf, h: usize, w: usize, c: usize, stats: PM, st: &Stream) -> Result<DevBuf> {
        let x = self.res(&format!("{pre}m0"), x, h, w, c, c, stats, st)?;
        let p = h * w;
        let n = DevBuf::alloc_async(p * c * 2, st)?;
        let a = format!("{pre}attn");
        self.norm(&format!("{a}.n"), x.ptr(), n.ptr(), stats, p, c, false, st)?;
        single_head_attention(&x, &n, p, c, [self.w(&format!("{a}.qkv.w")), self.w(&format!("{a}.qkv.b")), self.w(&format!("{a}.proj.w")),
            self.w(&format!("{a}.proj.b"))], st)?;
        drop(n);
        self.res(&format!("{pre}m1"), x, h, w, c, c, stats, st)
    }

    /// Peak activation memory (bytes) of encoding or decoding an `h` x `w` picture (`h`, `w` in pixels).
    pub fn activation_bytes(h: usize, w: usize) -> usize {
        let full = h * w;
        // the decoder's last upsampler: 256 channels at full size, its conv's output, and the next level's first resnet
        full * 256 * 2 * 2 + full * 128 * 2 * 2 + (512 << 20)
    }

    /// RGB f32 [h, w, 3] in 0..1 (both sides multiples of 8) -> the sampler's latent f32 [16, h/8, w/8]: the encoder's
    /// mean times [`SCALE`], as ComfyUI's `VAEEncode` gives it.
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize, st: &Stream) -> Result<Vec<f32>> {
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
        let mut c = self.e_in;
        let mut x = alloc(c, h, w)?;
        self.conv("e.in", input.ptr(), x.ptr(), h, w, RGB_PAD, c, 3, st)?;
        drop(input);
        let (mut hh, mut ww) = (h, w);
        for (i, &(cout, cin, blocks)) in self.down.iter().enumerate() {
            for b in 0..blocks {
                let from = if b == 0 { cin } else { cout };
                x = self.res(&format!("e{i}.{b}"), x, hh, ww, from, cout, sp, st)?;
                c = cout;
            }
            if i + 1 < LEVELS {
                // zero padding on the right and bottom, then a stride 2 conv: a stride 1 conv with symmetric padding
                // holds those samples at odd positions
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
        x = self.mid("e", x, hh, ww, c, sp, st)?;
        let p = hh * ww;
        let t = alloc(c, hh, ww)?;
        self.norm("eout.n", x.ptr(), t.ptr(), sp, p, c, true, st)?;
        drop(x);
        let moments = alloc(2 * Z, hh, ww)?;
        self.conv("eout", t.ptr(), moments.ptr(), hh, ww, c, 2 * Z, 3, st)?;
        let mut bytes = vec![0u8; p * 2 * Z * 2];
        moments.download(0, &mut bytes, st)?;
        let mut out = vec![0f32; Z * p];
        for (i, px) in bytes.chunks_exact(2 * Z * 2).enumerate() {
            for ch in 0..Z {
                out[ch * p + i] = bf16::from_le_bytes([px[ch * 2], px[ch * 2 + 1]]).to_f32() * SCALE;
            }
        }
        Ok(out)
    }

    /// The sampler's latent f32 [16, lh, lw] -> RGB f32 [8lh, 8lw, 3] in 0..1.
    pub fn decode(&self, latent: &DevBuf, lh: usize, lw: usize, st: &Stream) -> Result<Vec<f32>> {
        Ok(self.decode_raw(latent, lh, lw, st)?.into_iter().map(to_unit).collect())
    }

    /// The decoder's output before ComfyUI's `process_output`, about -1..1.
    fn decode_raw(&self, latent: &DevBuf, lh: usize, lw: usize, st: &Stream) -> Result<Vec<f32>> {
        let (mut hh, mut ww) = (lh, lw);
        let alloc = |c: usize, a: usize, b: usize| DevBuf::alloc_async(a * b * c * 2, st);
        let stats = DevBuf::alloc_async(GROUPS * 8, st)?;
        let sp = stats.ptr();
        let z = alloc(Z, hh, ww)?;
        k!(gpu_vae_latent_in(latent.ptr(), self.w("lat.mean"), self.w("lat.std"), z.ptr(), Z as i32, (hh * ww) as i32, s(st)))?;
        let mut c = self.d_in;
        let mut x = alloc(c, hh, ww)?;
        self.conv("d.in", z.ptr(), x.ptr(), hh, ww, Z, c, 3, st)?;
        drop(z);
        x = self.mid("d", x, hh, ww, c, sp, st)?;
        for (i, &(cout, cin, blocks)) in self.up.iter().enumerate() {
            for b in 0..blocks {
                let from = if b == 0 { cin } else { cout };
                x = self.res(&format!("u{i}.{b}"), x, hh, ww, from, cout, sp, st)?;
                c = cout;
            }
            if i + 1 < LEVELS {
                let p = hh * ww;
                let e = alloc(4 * c, hh, ww)?;
                k!(gpu_gemm(x.ptr(), c as i64, self.w(&format!("u{i}.exp.w")), c as i64, e.ptr(), (4 * c) as i64, self.w(&format!("u{i}.exp.b")),
                    p as i64, (4 * c) as i64, c as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
                drop(x);
                let shuffled = alloc(c, 2 * hh, 2 * ww)?;
                k!(gpu_pixel_shuffle2_bf16(e.ptr(), shuffled.ptr(), hh as i32, ww as i32, c as i32, s(st)))?;
                drop(e);
                hh *= 2;
                ww *= 2;
                let y = alloc(c, hh, ww)?;
                self.conv(&format!("u{i}.up"), shuffled.ptr(), y.ptr(), hh, ww, c, c, 3, st)?;
                x = y;
            }
        }
        let p = hh * ww;
        let t = alloc(c, hh, ww)?;
        self.norm("dout.n", x.ptr(), t.ptr(), sp, p, c, true, st)?;
        drop(x);
        let o = alloc(OUT_PAD, hh, ww)?;
        self.conv("dout", t.ptr(), o.ptr(), hh, ww, c, OUT_PAD, 3, st)?;
        let mut bytes = vec![0u8; p * OUT_PAD * 2];
        o.download(0, &mut bytes, st)?;
        let mut rgb = Vec::with_capacity(p * 3);
        for px in bytes.chunks_exact(OUT_PAD * 2) {
            for ch in 0..3 {
                rgb.push(bf16::from_le_bytes([px[ch * 2], px[ch * 2 + 1]]).to_f32());
            }
        }
        Ok(rgb)
    }

    /// [`Vae::decode`] of a host latent in square tiles of `tile` latent pixels, overlapping by a quarter, as ComfyUI's
    /// `decode_tiled` with `tiled_vae` does: each tile decoded alone, blended with cosine ramps over the overlaps in f32.
    pub fn decode_tiled(&self, latent: &[f32], lh: usize, lw: usize, tile: usize, st: &Stream) -> Result<Vec<f32>> {
        if latent.len() != Z * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), Z * lh * lw);
        }
        let tile = tile.max(2);
        // decode_tiled's overlap (tile // 4 latents) through tiled_vae's clamps
        let ov = (tile / 4).min(tile - 1);
        let mut blend = Blend::new(lh * 8, lw * 8);
        for (y0, y1, x0, x1) in tile_ranges(lh, lw, tile, ov) {
            let (th, tw) = (y1 - y0, x1 - x0);
            let mut sub = Vec::with_capacity(Z * th * tw);
            for c in 0..Z {
                for y in y0..y1 {
                    sub.extend_from_slice(&latent[(c * lh + y) * lw + x0..][..tw]);
                }
            }
            let d = DevBuf::alloc_async(sub.len() * 4, st)?;
            d.upload(0, &f32_bytes(&sub), st)?;
            let px = self.decode_raw(&d, th, tw, st)?;
            blend.add(&px, 3, (y0 * 8, y1 * 8, x0 * 8, x1 * 8), (y0 > 0, y1 < lh, x0 > 0, x1 < lw), ov * 8);
        }
        Ok(blend.finish().into_iter().map(to_unit).collect())
    }

    /// [`Vae::encode`] in square tiles of `tile` pixels overlapping by `overlap` (multiples of 8), as ComfyUI's
    /// `encode_tiled` with `tiled_vae` does; the latents blend with cosine ramps over the overlaps.
    pub fn encode_tiled(&self, rgb: &[f32], h: usize, w: usize, tile: usize, overlap: usize, st: &Stream) -> Result<Vec<f32>> {
        if h % 8 != 0 || w % 8 != 0 || tile % 8 != 0 || overlap % 8 != 0 || tile == 0 || rgb.len() != h * w * 3 {
            bail!("can't encode {w}x{h} in tiles of {tile} from {} values", rgb.len());
        }
        let ov = overlap.min(tile - 8);
        let (lh, lw) = (h / 8, w / 8);
        let mut blend = Blend::new(lh, lw);
        for (y0, y1, x0, x1) in tile_ranges(h, w, tile, ov) {
            let (th, tw) = (y1 - y0, x1 - x0);
            let mut sub = Vec::with_capacity(th * tw * 3);
            for y in y0..y1 {
                sub.extend_from_slice(&rgb[(y * w + x0) * 3..][..tw * 3]);
            }
            let z = self.encode(&sub, th, tw, st)?;
            // channel planes to pixel rows of 16, as the blend keeps them
            let (zh, zw) = (th / 8, tw / 8);
            let mut rows = vec![0f32; zh * zw * Z];
            for c in 0..Z {
                for i in 0..zh * zw {
                    rows[i * Z + c] = z[c * zh * zw + i];
                }
            }
            blend.add(&rows, Z, (y0 / 8, y1 / 8, x0 / 8, x1 / 8), (y0 > 0, y1 < h, x0 > 0, x1 < w), ov / 8);
        }
        let px = blend.finish();
        let p = lh * lw;
        let mut out = vec![0f32; Z * p];
        for i in 0..p {
            for c in 0..Z {
                out[c * p + i] = px[i * Z + c];
            }
        }
        Ok(out)
    }
}

/// ComfyUI's `process_output`: clamp((x + 1) / 2, 0, 1).
fn to_unit(v: f32) -> f32 {
    ((v + 1.0) / 2.0).clamp(0.0, 1.0)
}

/// `tiled_vae`'s tiles of an `h` x `w` grid: squares of `tile` stepping by `tile - ov`, rows outermost, skipping a last
/// row or column that would lie within the overlap. (y0, y1, x0, x1) each.
fn tile_ranges(h: usize, w: usize, tile: usize, ov: usize) -> Vec<(usize, usize, usize, usize)> {
    let stride = tile.saturating_sub(ov).max(1);
    if h <= tile && w <= tile {
        return vec![(0, h, 0, w)];
    }
    let mut out = Vec::new();
    for y in (0..h).step_by(stride) {
        let y1 = (y + tile).min(h);
        if y > 0 && y1 - y <= ov {
            continue;
        }
        for x in (0..w).step_by(stride) {
            let x1 = (x + tile).min(w);
            if x > 0 && x1 - x <= ov {
                continue;
            }
            out.push((y, y1, x, x1));
        }
    }
    out
}

/// The weighted sum and weights of overlapping tiles of `c`-channel pixel rows.
struct Blend {
    h: usize,
    w: usize,
    sum: Vec<f32>,
    count: Vec<f32>,
    c: usize,
}

impl Blend {
    fn new(h: usize, w: usize) -> Self {
        Blend { h, w, sum: Vec::new(), count: vec![0.0; h * w], c: 1 }
    }

    /// `0.5 - 0.5 cos(pi t)` over `n` points of 0..1, in f32 as torch computes it.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
                0.5 - 0.5 * (t * std::f32::consts::PI).cos()
            })
            .collect()
    }

    /// One tile `[ye - ys, xe - xs, c]` at (ys, ye, xs, xe); `inner` says which of its edges (top, bottom, left, right)
    /// are inside the picture, where it ramps over `blend` pixels (at most half the tile).
    #[allow(clippy::type_complexity)]
    fn add(&mut self, tile: &[f32], c: usize, (ys, ye, xs, xe): (usize, usize, usize, usize), inner: (bool, bool, bool, bool), blend: usize) {
        if self.sum.is_empty() {
            self.c = c;
            self.sum = vec![0.0; self.h * self.w * c];
        }
        let (top, bottom, left, right) = inner;
        let weights = |len: usize, lo: bool, hi: bool| {
            let n = blend.min(len / 2);
            let mut v = vec![1f32; len];
            if n > 0 {
                let r = Self::ramp(n);
                if lo {
                    v[..n].copy_from_slice(&r);
                }
                if hi {
                    for (j, rv) in r.iter().enumerate() {
                        v[len - n + j] = 1.0 - rv;
                    }
                }
            }
            v
        };
        let (th, tw) = (ye - ys, xe - xs);
        let (wy, wx) = (weights(th, top, bottom), weights(tw, left, right));
        for y in 0..th {
            for x in 0..tw {
                let wt = wy[y] * wx[x];
                let at = (ys + y) * self.w + xs + x;
                self.count[at] += wt;
                for ch in 0..c {
                    self.sum[at * c + ch] += tile[(y * tw + x) * c + ch] * wt;
                }
            }
        }
    }

    fn finish(self) -> Vec<f32> {
        let c = self.c;
        self.sum.iter().enumerate().map(|(i, v)| v / self.count[i / c].max(1e-6)).collect()
    }
}

/// ComfyUI's `preferred_decode_tile`: the largest square tile, in latent pixels, whose decode fits `free` bytes by
/// ComfyUI's own estimate (3150 bytes per output pixel and 384 MB), a multiple of 8 from 32 to 96.
pub fn preferred_tile(free: usize) -> usize {
    let budget = free as f64 * 0.6 - (384u64 << 20) as f64;
    if budget <= 0.0 {
        return 32;
    }
    let side = ((budget / 3150.0).max(1.0).sqrt() as usize) / 8;
    (side / 8 * 8).clamp(32, 96)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_like_tiled_vae() {
        // 256 latents in tiles of 96 overlapping by 24: steps of 72
        let t = tile_ranges(256, 256, 96, 24);
        let starts: Vec<usize> = t.iter().filter(|r| r.2 == 0).map(|r| r.0).collect();
        assert_eq!(starts, [0, 72, 144, 216]);
        assert_eq!(tile_ranges(64, 80, 96, 24), [(0, 64, 0, 80)]);
        assert_eq!(preferred_tile(12 << 30), 96);
        assert_eq!(preferred_tile(1 << 30), 32);
    }

    #[test]
    fn blend_of_equal_tiles_is_the_picture() {
        let (h, w) = (40, 56);
        let mut b = Blend::new(h, w);
        for (y0, y1, x0, x1) in tile_ranges(h, w, 24, 6) {
            let tile: Vec<f32> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (y * w + x) as f32)).collect();
            b.add(&tile, 1, (y0, y1, x0, x1), (y0 > 0, y1 < h, x0 > 0, x1 < w), 6);
        }
        for (i, v) in b.finish().iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-2, "{i}: {v}");
        }
    }
}
