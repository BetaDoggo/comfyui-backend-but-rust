//! Qwen Image 2.1's VAE for single images, NHWC bf16: the Wan 2.2 layout with a one-frame kernel, no patchify,
//! 64 latent channels at /16 and four image channels (RGBA).
//!
//! With one frame every CausalConv3d is a 2D conv, the temporal convs of the resamples are skipped (as in
//! ComfyUI), and each stage adds a shortcut that regroups channels instead of convolving: `AvgDown3D` in the encoder
//! and `DupUp3D` in the decoder.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, s, PM};
use crate::offload::Layout;
use crate::st::{f32_bytes, SafeTensors};
use crate::vae::{bias_pad, conv_krsc, single_head_attention};
use anyhow::{bail, Result};
use half::bf16;
use std::path::Path;

pub const CH: usize = 64;
/// Pixels per latent.
pub const SCALE: usize = 16;
/// ComfyUI's `QwenImage21` latent format: the sampler's latent is `(raw - mean) / std` per channel.
pub const LATENT_MEAN: [f32; CH] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub const LATENT_STD: [f32; CH] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
];
/// Channels of the image going in and out, padded to what the conv kernel takes.
const PAD: usize = 8;
const RES_BLOCKS: usize = 2;

/// True if `has` finds the keys of this VAE: the Wan 2.2 layout with a temporal kernel of one.
pub fn is_vae21(st: &SafeTensors) -> bool {
    st.has("decoder.upsamples.0.upsamples.0.residual.2.weight")
        && st.info("decoder.head.2.weight").is_ok_and(|i| i.shape.len() == 5 && i.shape[2] == 1)
}

struct Resample {
    /// Channels, and the time steps the shortcut regroups (2 where the file has a temporal conv).
    c: usize,
    ft: usize,
}

struct Stage {
    pre: String,
    cin: usize,
    cout: usize,
    /// (name, cin, cout, shortcut conv) per residual block
    res: Vec<(String, usize, usize, bool)>,
    resample: Option<Resample>,
}

enum Mid {
    Res(String, usize),
    Attn(String, usize),
}

pub struct Vae {
    buf: DevBuf,
    lay: Layout,
    encoder: bool,
    /// Decoder stages after the middle.
    up: Vec<Stage>,
    down: Vec<Stage>,
    dec_mid: Vec<Mid>,
    enc_mid: Vec<Mid>,
    c0: usize,
    e0: usize,
}

/// Conv weight [Cout, Cin, 1, R, R] -> KRSC bf16 with Cin zero-padded to `PAD`.
fn conv_krsc_in_padded(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    let (w, cout, ci, r) = conv_krsc(st, name, 0)?;
    let mut out = vec![0u8; cout * r * r * PAD * 2];
    for (i, px) in w.chunks_exact(ci * 2).enumerate() {
        out[i * PAD * 2..][..ci * 2].copy_from_slice(px);
    }
    Ok(out)
}

fn stages(st: &SafeTensors, side: &str, name: &str, blocks: usize) -> Result<Vec<Stage>> {
    let mut out = Vec::new();
    for i in 0.. {
        let pre = format!("{side}.{name}.{i}.{name}");
        let first = format!("{pre}.0.residual.2.weight");
        if !st.has(&first) {
            break;
        }
        let mut res = Vec::new();
        for j in 0..blocks {
            let p = format!("{pre}.{j}");
            let w = st.info(&format!("{p}.residual.2.weight"))?;
            res.push((p.clone(), w.shape[1], w.shape[0], st.has(&format!("{p}.shortcut.weight"))));
        }
        let rs = format!("{pre}.{blocks}");
        let resample = if st.has(&format!("{rs}.resample.1.weight")) {
            let c = st.info(&format!("{rs}.resample.1.weight"))?.shape[0];
            Some(Resample { c, ft: if st.has(&format!("{rs}.time_conv.weight")) { 2 } else { 1 } })
        } else {
            None
        };
        out.push(Stage { pre: format!("{side}.{name}.{i}"), cin: res[0].1, cout: res[blocks - 1].2, res, resample });
    }
    Ok(out)
}

fn middle(side: &str, c: usize) -> Vec<Mid> {
    vec![Mid::Res(format!("{side}.middle.0"), c), Mid::Attn(format!("{side}.middle.1"), c), Mid::Res(format!("{side}.middle.2"), c)]
}

impl Vae {
    pub fn load(path: &Path, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        if !is_vae21(&st) {
            bail!("{}: not a Qwen Image 2.1 VAE", path.display());
        }
        let mut lay = Layout::new();
        let conv = |lay: &mut Layout, key: &str, name: &str, pad: usize| -> Result<()> {
            let (w, cop, _, _) = conv_krsc(&st, &format!("{name}.weight"), pad)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias_pad(&st, &format!("{name}.bias"), cop)?);
            Ok(())
        };
        let res = |lay: &mut Layout, p: &str, shortcut: bool| -> Result<()> {
            lay.bf16(&st, &format!("{p}.g0"), &[format!("{p}.residual.0.gamma")])?;
            lay.bf16(&st, &format!("{p}.g3"), &[format!("{p}.residual.3.gamma")])?;
            conv(lay, &format!("{p}.c2"), &format!("{p}.residual.2"), 0)?;
            conv(lay, &format!("{p}.c6"), &format!("{p}.residual.6"), 0)?;
            if shortcut {
                conv(lay, &format!("{p}.sc"), &format!("{p}.shortcut"), 0)?;
            }
            Ok(())
        };
        let mid = |lay: &mut Layout, mids: &[Mid]| -> Result<()> {
            for m in mids {
                match m {
                    Mid::Res(p, _) => res(lay, p, false)?,
                    Mid::Attn(p, _) => {
                        lay.bf16(&st, &format!("{p}.g"), &[format!("{p}.norm.gamma")])?;
                        lay.bf16(&st, &format!("{p}.qkv.w"), &[format!("{p}.to_qkv.weight")])?;
                        lay.bf16(&st, &format!("{p}.qkv.b"), &[format!("{p}.to_qkv.bias")])?;
                        lay.bf16(&st, &format!("{p}.proj.w"), &[format!("{p}.proj.weight")])?;
                        lay.bf16(&st, &format!("{p}.proj.b"), &[format!("{p}.proj.bias")])?;
                    }
                }
            }
            Ok(())
        };
        let c0 = st.info("decoder.conv1.weight")?.shape[0];
        conv(&mut lay, "conv2", "conv2", 0)?;
        conv(&mut lay, "dec.conv1", "decoder.conv1", 0)?;
        let dec_mid = middle("decoder", c0);
        mid(&mut lay, &dec_mid)?;
        let up = stages(&st, "decoder", "upsamples", 3)?;
        for stage in &up {
            for (p, _, _, sc) in &stage.res {
                res(&mut lay, p, *sc)?;
            }
            if stage.resample.is_some() {
                conv(&mut lay, &format!("{}.conv", stage.pre), &format!("{}.upsamples.3.resample.1", stage.pre), 0)?;
            }
        }
        lay.bf16(&st, "dec.head.g", &["decoder.head.0.gamma".to_string()])?;
        conv(&mut lay, "dec.head", "decoder.head.2", PAD)?;

        let encoder = st.has("encoder.conv1.weight");
        let (mut down, mut enc_mid, mut e0) = (Vec::new(), Vec::new(), 0);
        if encoder {
            e0 = st.info("encoder.conv1.weight")?.shape[0];
            lay.bytes("enc.conv1.w", conv_krsc_in_padded(&st, "encoder.conv1.weight")?);
            lay.bytes("enc.conv1.b", bias_pad(&st, "encoder.conv1.bias", 0)?);
            down = stages(&st, "encoder", "downsamples", RES_BLOCKS)?;
            for stage in &down {
                for (p, _, _, sc) in &stage.res {
                    res(&mut lay, p, *sc)?;
                }
                if stage.resample.is_some() {
                    conv(&mut lay, &format!("{}.conv", stage.pre), &format!("{}.downsamples.2.resample.1", stage.pre), 0)?;
                }
            }
            let c = down.last().map_or(e0, |d| d.cout);
            enc_mid = middle("encoder", c);
            mid(&mut lay, &enc_mid)?;
            lay.bf16(&st, "enc.head.g", &["encoder.head.0.gamma".to_string()])?;
            conv(&mut lay, "enc.head", "encoder.head.2", 0)?;
            conv(&mut lay, "quant", "conv1", 0)?;
        }
        lay.bytes("mean", f32_bytes(&LATENT_MEAN));
        lay.bytes("std", f32_bytes(&LATENT_STD));
        let mut host = vec![0u8; lay.size];
        lay.pack(&st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Vae { buf, lay, encoder, up, down, dec_mid, enc_mid, c0, e0 })
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, key: &str, x: PM, out: PM, h: usize, w: usize, cin: usize, cout: usize, r: usize, st: &Stream) -> Result<()> {
        k!(gpu_conv2d(x, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), out, 1, h as i32, w as i32, cin as i32, cout as i32, r as i32,
            s(st)))
    }

    /// Peak activation memory (bytes) for decoding an h x w latent, or encoding the image it stands for.
    pub fn activation_bytes(h: usize, w: usize) -> usize {
        let full = h * SCALE * w * SCALE;
        // the last upsample: the stage's input, the nearest-neighbour copy, the conv output and the shortcut at 288 channels
        full * 288 * 2 * 3 + full * 144 * 2 * 2 + (512 << 20)
    }

    #[allow(clippy::too_many_arguments)]
    fn res(&self, pre: &str, x: &DevBuf, h: usize, w: usize, cin: usize, cout: usize, shortcut: bool, st: &Stream) -> Result<DevBuf> {
        let alloc = |c: usize| DevBuf::alloc_async(h * w * c * 2, st);
        let p = (h * w) as i64;
        let t = alloc(cin)?;
        k!(gpu_l2norm_silu(x.ptr(), t.ptr(), self.w(&format!("{pre}.g0")), p, cin as i32, 1, s(st)))?;
        let u = alloc(cout)?;
        self.conv(&format!("{pre}.c2"), t.ptr(), u.ptr(), h, w, cin, cout, 3, st)?;
        drop(t);
        let t2 = alloc(cout)?;
        k!(gpu_l2norm_silu(u.ptr(), t2.ptr(), self.w(&format!("{pre}.g3")), p, cout as i32, 1, s(st)))?;
        self.conv(&format!("{pre}.c6"), t2.ptr(), u.ptr(), h, w, cout, cout, 3, st)?;
        drop(t2);
        if shortcut {
            let sc = alloc(cout)?;
            self.conv(&format!("{pre}.sc"), x.ptr(), sc.ptr(), h, w, cin, cout, 1, st)?;
            k!(gpu_add_gated_bf16(u.ptr(), cout as i64, sc.ptr(), cout as i64, null(), p as i32, cout as i32, s(st)))?;
        } else {
            k!(gpu_add_gated_bf16(u.ptr(), cout as i64, x.ptr(), cout as i64, null(), p as i32, cout as i32, s(st)))?;
        }
        Ok(u)
    }

    fn attention(&self, pre: &str, x: &DevBuf, p: usize, c: usize, st: &Stream) -> Result<()> {
        let n = DevBuf::alloc_async(p * c * 2, st)?;
        k!(gpu_l2norm_silu(x.ptr(), n.ptr(), self.w(&format!("{pre}.g")), p as i64, c as i32, 0, s(st)))?;
        let w = |key: &str| self.w(&format!("{pre}.{key}"));
        single_head_attention(x, &n, p, c, [w("qkv.w"), w("qkv.b"), w("proj.w"), w("proj.b")], st)
    }

    fn middle(&self, mids: &[Mid], mut x: DevBuf, h: usize, w: usize, st: &Stream) -> Result<DevBuf> {
        for m in mids {
            match m {
                Mid::Res(p, c) => x = self.res(p, &x, h, w, *c, *c, false, st)?,
                Mid::Attn(p, c) => self.attention(p, &x, h * w, *c, st)?,
            }
        }
        Ok(x)
    }

    /// Latent f32 [64, h, w] at the VAE's scale (ComfyUI's `vae.decode` input) to the padded NHWC bf16 image
    /// [16h * 16w, 8] in -1 to 1. With `model_scale` the latent is the sampler's, and is scaled first.
    pub fn decode_raw(&self, latent: &DevBuf, h: usize, w: usize, model_scale: bool, st: &Stream) -> Result<DevBuf> {
        let alloc = |c: usize, hh: usize, ww: usize| DevBuf::alloc_async(hh * ww * c * 2, st);
        let zin = alloc(CH, h, w)?;
        if model_scale {
            k!(gpu_vae_latent_in(latent.ptr(), self.w("mean"), self.w("std"), zin.ptr(), CH as i32, (h * w) as i32, s(st)))?;
        } else {
            let (mean, std) = ([0f32; CH], [1f32; CH]);
            let (m, d) = (DevBuf::alloc_async(CH * 4, st)?, DevBuf::alloc_async(CH * 4, st)?);
            m.upload(0, &f32_bytes(&mean), st)?;
            d.upload(0, &f32_bytes(&std), st)?;
            k!(gpu_vae_latent_in(latent.ptr(), m.ptr(), d.ptr(), zin.ptr(), CH as i32, (h * w) as i32, s(st)))?;
        }
        let z2 = alloc(CH, h, w)?;
        self.conv("conv2", zin.ptr(), z2.ptr(), h, w, CH, CH, 1, st)?;
        let mut x = alloc(self.c0, h, w)?;
        self.conv("dec.conv1", z2.ptr(), x.ptr(), h, w, CH, self.c0, 3, st)?;
        drop((zin, z2));
        x = self.middle(&self.dec_mid, x, h, w, st)?;
        let (mut hh, mut ww) = (h, w);
        for stage in &self.up {
            let input = x;
            let mut main: Option<DevBuf> = None;
            for (p, cin, cout, sc) in &stage.res {
                let next = self.res(p, main.as_ref().unwrap_or(&input), hh, ww, *cin, *cout, *sc, st)?;
                main = Some(next);
            }
            let main = main.unwrap();
            let Some(rs) = &stage.resample else {
                x = main;
                continue;
            };
            let c = stage.cout;
            debug_assert_eq!(rs.c, c);
            let big = alloc(c, 2 * hh, 2 * ww)?;
            k!(gpu_upsample2x(main.ptr(), big.ptr(), hh as i32, ww as i32, c as i32, s(st)))?;
            drop(main);
            let y = alloc(c, 2 * hh, 2 * ww)?;
            self.conv(&format!("{}.conv", stage.pre), big.ptr(), y.ptr(), 2 * hh, 2 * ww, c, c, 3, st)?;
            drop(big);
            // the shortcut: the stage's input, regrouped onto the doubled grid
            let sc = alloc(c, 2 * hh, 2 * ww)?;
            k!(gpu_dup_up(input.ptr(), sc.ptr(), hh as i32, ww as i32, stage.cin as i32, c as i32, rs.ft as i32, s(st)))?;
            drop(input);
            k!(gpu_add_gated_bf16(y.ptr(), c as i64, sc.ptr(), c as i64, null(), (4 * hh * ww) as i32, c as i32, s(st)))?;
            x = y;
            hh *= 2;
            ww *= 2;
        }
        let p = hh * ww;
        let c = x_channels(&self.up, self.c0);
        let t = alloc(c, hh, ww)?;
        k!(gpu_l2norm_silu(x.ptr(), t.ptr(), self.w("dec.head.g"), p as i64, c as i32, 1, s(st)))?;
        drop(x);
        let o = alloc(PAD, hh, ww)?;
        self.conv("dec.head", t.ptr(), o.ptr(), hh, ww, c, PAD, 3, st)?;
        Ok(o)
    }

    /// Latent f32 [64, h, w] (the sampler's scale) to RGBA8 [16h, 16w, 4].
    pub fn decode(&self, latent: &DevBuf, h: usize, w: usize, st: &Stream) -> Result<Vec<u8>> {
        let o = self.decode_raw(latent, h, w, true, st)?;
        let p = h * SCALE * w * SCALE;
        let rgba = DevBuf::alloc_async(p * 4, st)?;
        k!(gpu_to_rgba8(o.ptr(), PAD as i32, rgba.ptr(), p as i64, s(st)))?;
        let mut out = vec![0u8; p * 4];
        rgba.download(0, &mut out, st)?;
        Ok(out)
    }

    /// RGBA f32 [h, w, 4], 0 to 1 (both sides multiples of 16), to the mean of the encoder's latent, f32 [64, h/16, w/16].
    /// `raw` leaves it at the VAE's scale, as ComfyUI's `vae.encode` returns it; otherwise it is the sampler's.
    pub fn encode(&self, rgba: &[f32], h: usize, w: usize, raw: bool, st: &Stream) -> Result<Vec<f32>> {
        if !self.encoder {
            bail!("this VAE file has no encoder");
        }
        if h % SCALE != 0 || w % SCALE != 0 {
            bail!("{w}x{h}: the encoder takes sides in multiples of {SCALE}");
        }
        let alloc = |c: usize, hh: usize, ww: usize| DevBuf::alloc_async(hh * ww * c * 2, st);
        let mut px = vec![bf16::ZERO; h * w * PAD];
        for (o, v) in px.chunks_exact_mut(PAD).zip(rgba.chunks_exact(4)) {
            for c in 0..4 {
                o[c] = bf16::from_f32(v[c] * 2.0 - 1.0);
            }
        }
        let input = alloc(PAD, h, w)?;
        input.upload(0, &px.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>(), st)?;
        let mut x = alloc(self.e0, h, w)?;
        self.conv("enc.conv1", input.ptr(), x.ptr(), h, w, PAD, self.e0, 3, st)?;
        drop(input);
        let (mut hh, mut ww) = (h, w);
        for stage in &self.down {
            let input = x;
            let mut main: Option<DevBuf> = None;
            for (p, cin, cout, sc) in &stage.res {
                main = Some(self.res(p, main.as_ref().unwrap_or(&input), hh, ww, *cin, *cout, *sc, st)?);
            }
            let mut main = main.unwrap();
            let c = stage.cout;
            let (ft, fs) = stage.resample.as_ref().map_or((1, 1), |r| (r.ft, 2));
            if stage.resample.is_some() {
                // a stride 1 conv pads the same zeros, and holds the stride 2 samples at odd positions
                let full = alloc(c, hh, ww)?;
                self.conv(&format!("{}.conv", stage.pre), main.ptr(), full.ptr(), hh, ww, c, c, 3, st)?;
                let y = alloc(c, hh / 2, ww / 2)?;
                k!(gpu_take_odd(full.ptr(), y.ptr(), hh as i32, ww as i32, c as i32, s(st)))?;
                main = y;
            }
            let sc = alloc(c, hh / fs, ww / fs)?;
            k!(gpu_avg_down(input.ptr(), sc.ptr(), hh as i32, ww as i32, stage.cin as i32, c as i32, ft as i32, fs as i32, s(st)))?;
            drop(input);
            hh /= fs;
            ww /= fs;
            k!(gpu_add_gated_bf16(main.ptr(), c as i64, sc.ptr(), c as i64, null(), (hh * ww) as i32, c as i32, s(st)))?;
            x = main;
        }
        x = self.middle(&self.enc_mid, x, hh, ww, st)?;
        let c = self.down.last().map_or(self.e0, |d| d.cout);
        let z2 = CH * 2;
        let t = alloc(c, hh, ww)?;
        k!(gpu_l2norm_silu(x.ptr(), t.ptr(), self.w("enc.head.g"), (hh * ww) as i64, c as i32, 1, s(st)))?;
        let moments = alloc(z2, hh, ww)?;
        self.conv("enc.head", t.ptr(), moments.ptr(), hh, ww, c, z2, 3, st)?;
        let q = alloc(z2, hh, ww)?;
        self.conv("quant", moments.ptr(), q.ptr(), hh, ww, z2, z2, 1, st)?;
        let mut bytes = vec![0u8; hh * ww * z2 * 2];
        q.download(0, &mut bytes, st)?;
        let value = |i: usize| bf16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32();
        let mut out = Vec::with_capacity(CH * hh * ww);
        for ch in 0..CH {
            for p in 0..hh * ww {
                let v = value(p * z2 + ch);
                out.push(if raw { v } else { (v - LATENT_MEAN[ch]) / LATENT_STD[ch] });
            }
        }
        Ok(out)
    }
}

/// Channels the decoder's last stage leaves.
fn x_channels(up: &[Stage], c0: usize) -> usize {
    up.last().map_or(c0, |s| s.cout)
}

/// Compares the decoder and the encoder with the dump of `tools/qwen21_ref.py --vae-only`.
pub fn check(model: &Path, reference: &Path) -> Result<()> {
    use crate::pixal3d::report;
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let r = SafeTensors::open(reference)?;
    let st = Stream::new()?;
    let t0 = std::time::Instant::now();
    let vae = Vae::load(model, &st)?;
    eprintln!("load {:.2} s", t0.elapsed().as_secs_f64());
    let shape = r.info("vae_in")?.shape.clone();
    let (h, w) = (shape[2], shape[3]);
    let z = DevBuf::alloc(CH * h * w * 4)?;
    z.upload(0, &f32_bytes(&r.f32_vec("vae_in", false)?), &st)?;
    let t1 = std::time::Instant::now();
    let o = vae.decode_raw(&z, h, w, false, &st)?;
    st.sync()?;
    eprintln!("decode {:.0} ms", t1.elapsed().as_secs_f64() * 1e3);
    let px = h * w * SCALE * SCALE;
    let mut bytes = vec![0u8; px * PAD * 2];
    o.download(0, &mut bytes, &st)?;
    st.sync()?;
    // ComfyUI's output: (x + 1) / 2, clamped, in RGBA order
    let ours: Vec<f32> = bytes.chunks_exact(PAD * 2).flat_map(|p| (0..4).map(|c| {
        ((bf16::from_le_bytes([p[c * 2], p[c * 2 + 1]]).to_f32() + 1.0) / 2.0).clamp(0.0, 1.0)
    }).collect::<Vec<_>>()).collect();
    report("decode", &ours, &r.f32_vec("vae_out", false)?);
    if r.has("image0") {
        let shape = r.info("image0")?.shape.clone();
        let (ih, iw) = (shape[1], shape[2]);
        let t2 = std::time::Instant::now();
        let z = vae.encode(&r.f32_vec("image0", false)?, ih, iw, true, &st)?;
        eprintln!("encode {:.0} ms", t2.elapsed().as_secs_f64() * 1e3);
        report("encode", &z, &r.f32_vec("ref_raw0", false)?);
        if r.has("ref_back0") {
            // what the difference costs: both latents decoded, against the picture and against ComfyUI's round trip
            let d = DevBuf::alloc(z.len() * 4)?;
            d.upload(0, &f32_bytes(&z), &st)?;
            let o = vae.decode_raw(&d, ih / SCALE, iw / SCALE, false, &st)?;
            st.sync()?;
            let mut bytes = vec![0u8; ih * iw * PAD * 2];
            o.download(0, &mut bytes, &st)?;
            st.sync()?;
            let back: Vec<f32> = bytes.chunks_exact(PAD * 2).flat_map(|p| (0..4).map(|c| {
                ((bf16::from_le_bytes([p[c * 2], p[c * 2 + 1]]).to_f32() + 1.0) / 2.0).clamp(0.0, 1.0)
            }).collect::<Vec<_>>()).collect();
            let image = r.f32_vec("image0", false)?;
            let theirs = r.f32_vec("ref_back0", false)?;
            let psnr = |a: &[f32], b: &[f32]| 10.0 * (a.len() as f64 / a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>()).log10();
            eprintln!("round trip PSNR: ours vs the picture {:.2} dB, ComfyUI's vs the picture {:.2} dB, ours vs ComfyUI's {:.2} dB",
                psnr(&back, &image), psnr(&theirs, &image), psnr(&back, &theirs));
            // the colour under transparent pixels is not kept by the VAE, so judge alpha and the opaque part apart
            let split = |a: &[f32], b: &[f32], alpha: bool| {
                let (mut x, mut y) = (Vec::new(), Vec::new());
                for ((p, q), r) in a.chunks_exact(4).zip(b.chunks_exact(4)).zip(image.chunks_exact(4)) {
                    if alpha {
                        x.push(p[3]);
                        y.push(q[3]);
                    } else if r[3] > 0.5 {
                        x.extend_from_slice(&p[..3]);
                        y.extend_from_slice(&q[..3]);
                    }
                }
                psnr(&x, &y)
            };
            eprintln!("ours vs ComfyUI's round trip: alpha {:.2} dB, colour where the picture is opaque {:.2} dB",
                split(&back, &theirs, true), split(&back, &theirs, false));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_statistics_cover_every_channel() {
        assert_eq!(LATENT_MEAN.len(), CH);
        assert!(LATENT_STD.iter().all(|s| *s > 2.0));
    }
}
