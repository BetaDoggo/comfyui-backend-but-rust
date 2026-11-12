//! Init image / Mask Image / inpainting nodes: image cropping, scaling and compositing (with
//! SwarmUI's color correction), init-image noise, inpaint VAE encoding, and the latent operations
//! behind Init Image Reset To Norm and Unsampler Prompt.

use std::sync::Arc;

use swarm_engine_api::{ImageBatch, InpaintConcat, Latent, LatentOp, Mask, PreviewMode, UnsampleRequest, VaeEncodeRequest};

use super::sampling::{SAMPLERS, SCHEDULERS};
use super::{input_err, NodeError, NodeOutput, NodeRegistry};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// Latent frames (`T` of a `[B, C, T, H, W]` video latent; 1 for images).
fn latent_frames(l: &Latent) -> usize {
    match &l.samples {
        swarm_engine_api::LatentSamples::Single(t) if t.shape.len() == 5 => t.shape[2],
        swarm_engine_api::LatentSamples::AudioVideo { video, .. } if video.shape.len() == 5 => video.shape[2],
        _ => 1,
    }
}

/// An image batch as plain `[B][H][W][C]` floats.
struct Px {
    count: usize,
    h: usize,
    w: usize,
    c: usize,
    data: Vec<f32>,
}

impl Px {
    fn of(b: &ImageBatch) -> Self {
        let mut data = Vec::with_capacity(b.frame_len() * b.count as usize);
        for i in 0..b.count {
            data.extend_from_slice(&b.frame_f32(i));
        }
        Self { count: b.count as usize, h: b.height as usize, w: b.width as usize, c: b.channels as usize, data }
    }

    fn at(&self, b: usize, y: usize, x: usize) -> usize {
        ((b * self.h + y) * self.w + x) * self.c
    }

    fn into_batch(self) -> ImageBatch {
        ImageBatch::from_f32(self.count as u32, self.h as u32, self.w as u32, self.c as u32, self.data)
    }
}

/// A mask resized (bilinear) to `w` x `h`, as `[B][H][W]` floats, with `count` frames (repeating as needed).
fn mask_plane(m: &Mask, w: usize, h: usize, count: usize) -> Result<Vec<f32>, NodeError> {
    let r = encode::resize_mask(m, w as u32, h as u32)?;
    let plane = w * h;
    Ok((0..count).flat_map(|b| r.data[(b % r.count as usize) * plane..(b % r.count as usize + 1) * plane].to_vec()).collect())
}

/// Python's `round()` (half to even).
fn py_round(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
}

// ------------------------------------------------------------------------------- color correction

fn rgb2hsv(r: f32, g: f32, b: f32) -> [f32; 3] {
    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;
    let h = if delta == 0.0 {
        0.0
    } else if cmax == r {
        ((g - b) / delta).rem_euclid(6.0)
    } else if cmax == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    } / 6.0;
    let s = if cmax == 0.0 { 0.0 } else { delta / cmax };
    [h, s, cmax]
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let [r, g, b] = match ((h * 6.0) as u8) % 6 {
        0 => [c, x, 0.0],
        1 => [x, c, 0.0],
        2 => [0.0, c, x],
        3 => [0.0, x, c],
        4 => [x, 0.0, c],
        _ => [c, 0.0, x],
    };
    [r + m, g + m, b + m]
}

/// SwarmUI's `linear_fit`: regress dest on source over the weighted (unmasked) pixels.
fn linear_fit(src: &[f32], dst: &[f32], weight: &[f32]) -> Vec<f32> {
    let total: f32 = weight.iter().sum();
    let sm = src.iter().zip(weight).map(|(s, w)| s * w).sum::<f32>() / total;
    let dm = dst.iter().zip(weight).map(|(d, w)| d * w).sum::<f32>() / total;
    let (mut num, mut den) = (0f32, 0f32);
    for i in 0..src.len() {
        let (sd, dd) = ((src[i] - sm) * weight[i], (dst[i] - dm) * weight[i]);
        num += sd * dd;
        den += sd * sd;
    }
    let m = if den != 0.0 { num / den } else { 1.0 };
    let b = dm - sm * m;
    src.iter().map(|s| (m * s + b).clamp(0.0, 1.0)).collect()
}

/// `color_correct_{uniform,linear,linear2}` over the composited region. `src`/`dst` are RGB
/// triples; `inverse` is 1 - mask. Only pixels the mask leaves completely untouched count.
fn color_correct(method: &str, src: &mut [[f32; 3]], dst: &[[f32; 3]], inverse: &[f32]) {
    let weight: Vec<f32> = inverse.iter().map(|v| ((v.clamp(0.0, 1.0) - 0.9999).clamp(0.0, 1.0)) * 10000.0).collect();
    let total: f32 = weight.iter().sum();
    if total <= 50.0 {
        return;
    }
    let sh: Vec<[f32; 3]> = src.iter().map(|p| rgb2hsv(p[0], p[1], p[2])).collect();
    let dh: Vec<[f32; 3]> = dst.iter().map(|p| rgb2hsv(p[0], p[1], p[2])).collect();
    let chan = |v: &[[f32; 3]], c: usize| v.iter().map(|p| p[c]).collect::<Vec<f32>>();
    let new_hsv: Vec<[f32; 3]> = match method {
        "Uniform" => {
            let mut diff = [0f32; 3];
            for c in 1..3 {
                diff[c] = (0..sh.len()).map(|i| (dh[i][c] - sh[i][c]) * weight[i]).sum::<f32>() / total;
            }
            sh.iter().map(|p| [p[0], (p[1] + diff[1]).clamp(0.0, 1.0), (p[2] + diff[2]).clamp(0.0, 1.0)]).collect()
        }
        "Linear" => {
            let s = linear_fit(&chan(&sh, 1), &chan(&dh, 1), &weight);
            let v = linear_fit(&chan(&sh, 2), &chan(&dh, 2), &weight);
            (0..sh.len()).map(|i| [sh[i][0], s[i], v[i]]).collect()
        }
        "Linear2" => {
            let svs: Vec<f32> = sh.iter().map(|p| p[1] * p[2]).collect();
            let svd: Vec<f32> = dh.iter().map(|p| p[1] * p[2]).collect();
            let sv = linear_fit(&svs, &svd, &weight);
            let v = linear_fit(&chan(&sh, 2), &chan(&dh, 2), &weight);
            (0..sh.len()).map(|i| [sh[i][0], if v[i] != 0.0 { (sv[i] / v[i]).clamp(0.0, 1.0) } else { 0.0 }, v[i]]).collect()
        }
        _ => return,
    };
    for (p, hsv) in src.iter_mut().zip(new_hsv) {
        *p = hsv2rgb(hsv[0], hsv[1], hsv[2]);
    }
}

// ------------------------------------------------------------------------------- compositing

/// ComfyUI's `composite` (multiplier 1) and SwarmUI's color-correcting variant: paste `source`
/// at (`x`, `y`) into `destination` through `mask` (resized to the source).
fn composite(dest: &ImageBatch, source: &ImageBatch, x: u32, y: u32, mask: Option<&Mask>, resize_source: bool, correction: Option<&str>) -> Result<ImageBatch, NodeError> {
    let source = if resize_source { encode::resize(source, dest.width, dest.height, "bilinear", "disabled")? } else { source.clone() };
    let mut d = Px::of(dest);
    // Alpha handling as in SwarmUI's node: a source alpha multiplies the mask; RGB is pasted, and an
    // RGBA destination keeps full opacity in the pasted area.
    let s = Px::of(&if source.channels == 4 && correction.is_none() && dest.channels == 4 { source.clone() } else { source.to_rgb() });
    let src_alpha = (source.channels == 4).then(|| source.alpha());
    let (left, top) = (x.min(dest.width) as usize, y.min(dest.height) as usize);
    let (vw, vh) = (s.w.min(d.w - left), s.h.min(d.h - top));
    let mut m = match mask {
        Some(m) => mask_plane(m, s.w, s.h, d.count)?,
        None => vec![1.0; d.count * s.w * s.h],
    };
    if let (Some(alpha), true) = (&src_alpha, s.c == 3) {
        let plane = s.w * s.h;
        for b in 0..d.count {
            for i in 0..plane {
                m[b * plane + i] *= alpha[(b % s.count) * plane + i];
            }
        }
    }
    for b in 0..d.count {
        let sb = b % s.count;
        // The visible region, as RGB triples, for color correction.
        let mut src_px: Vec<[f32; 3]> = Vec::with_capacity(vw * vh);
        let mut dst_px: Vec<[f32; 3]> = Vec::with_capacity(vw * vh);
        let mut inv: Vec<f32> = Vec::with_capacity(vw * vh);
        for yy in 0..vh {
            for xx in 0..vw {
                let si = s.at(sb, yy, xx);
                let di = d.at(b, top + yy, left + xx);
                src_px.push([s.data[si], s.data[si + 1.min(s.c - 1)], s.data[si + 2.min(s.c - 1)]]);
                dst_px.push([d.data[di], d.data[di + 1.min(d.c - 1)], d.data[di + 2.min(d.c - 1)]]);
                inv.push(1.0 - m[(b * s.h + yy) * s.w + xx]);
            }
        }
        if let Some(method) = correction {
            color_correct(method, &mut src_px, &dst_px, &inv);
        }
        for yy in 0..vh {
            for xx in 0..vw {
                let k = yy * vw + xx;
                let mv = 1.0 - inv[k];
                let di = d.at(b, top + yy, left + xx);
                let si = s.at(sb, yy, xx);
                for c in 0..d.c {
                    let sv = if c < 3 { src_px[k][c.min(2)] } else if s.c == 4 { s.data[si + 3] } else { 1.0 };
                    d.data[di + c] = mv * sv + (1.0 - mv) * d.data[di + c];
                }
            }
        }
    }
    Ok(d.into_batch())
}

/// The mask-greyed pixels VAEEncodeForInpaint / InpaintModelConditioning encode:
/// `(pixel - 0.5) * (1 - round(mask)) + 0.5` on the RGB channels.
fn grey_out(pixels: &ImageBatch, mask: &Mask) -> Result<ImageBatch, NodeError> {
    let mut p = Px::of(&pixels.to_rgb());
    let m = mask_plane(mask, p.w, p.h, p.count)?;
    for b in 0..p.count {
        for yy in 0..p.h {
            for xx in 0..p.w {
                let keep = 1.0 - m[(b * p.h + yy) * p.w + xx].round();
                let i = p.at(b, yy, xx);
                for c in 0..3 {
                    p.data[i + c] = (p.data[i + c] - 0.5) * keep + 0.5;
                }
            }
        }
    }
    Ok(p.into_batch())
}

/// Deterministic standard-normal samples (SplitMix64 + Box-Muller). SwarmImageNoise uses
/// `torch.randn` with the seed; this matches its distribution, not torch's exact values.
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (next().max(1e-12), next());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmImageCrop", "SwarmImageCrop", "SwarmUI/images")
                .describe("Crops an image to a specific region.")
                .input(link("image", "IMAGE"))
                .input(int_step("x", 0, 0, 8192, 8).tooltip("The x coordinate in pixels of the top left corner of the crop."))
                .input(int_step("y", 0, 0, 8192, 8).tooltip("The y coordinate in pixels of the top left corner of the crop."))
                .input(int_step("width", 512, 64, 8192, 8).tooltip("The width in pixels of the crop."))
                .input(int_step("height", 512, 64, 8192, 8).tooltip("The height in pixels of the crop."))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("image")?;
            let (x, y, w, h) = (i.int("x")?, i.int("y")?, i.int("width")?, i.int("height")?);
            if w <= 0 || h <= 0 {
                return Ok(NodeOutput::one(Value::Image(img.clone())));
            }
            let p = Px::of(img);
            let (x0, y0) = ((x.max(0) as usize).min(p.w), (y.max(0) as usize).min(p.h));
            let (x1, y1) = (((x + w).max(0) as usize).min(p.w), ((y + h).max(0) as usize).min(p.h));
            if x1 <= x0 || y1 <= y0 {
                return Err(input_err(format!("crop {w}x{h} at {x},{y} is outside the {}x{} image", p.w, p.h)));
            }
            let mut data = Vec::with_capacity(p.count * (y1 - y0) * (x1 - x0) * p.c);
            for b in 0..p.count {
                for yy in y0..y1 {
                    data.extend_from_slice(&p.data[p.at(b, yy, x0)..p.at(b, yy, x1 - 1) + p.c]);
                }
            }
            Ok(NodeOutput::one(Value::Image(ImageBatch::from_f32(p.count as u32, (y1 - y0) as u32, (x1 - x0) as u32, p.c as u32, data))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmImageScaleForMP", "SwarmImageScaleForMP", "SwarmUI/images")
                .describe("Scales an image to a target width and height, while keeping the aspect ratio.")
                .input(link("image", "IMAGE"))
                .input(int("width", 0, 0, 8192).tooltip("The target width of the image."))
                .input(int("height", 0, 0, 8192).tooltip("The target height of the image."))
                .input(boolean("can_shrink", true).tooltip("If true, the image can be shrunk to fit the target size, otherwise it will only be scaled up or left the same."))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("image")?;
            let target = (i.int("width")? * i.int("height")?) as f64;
            let scale = (target / (img.width as f64 * img.height as f64)).sqrt();
            if !i.boolean("can_shrink")? && scale < 1.0 {
                return Ok(NodeOutput::one(Value::Image(img.clone())));
            }
            let w = (py_round(img.width as f64 * scale / 64.0) * 64.0) as u32;
            let h = (py_round(img.height as f64 * scale / 64.0) * 64.0) as u32;
            Ok(NodeOutput::one(Value::Image(encode::resize(img, w.max(1), h.max(1), "bilinear", "disabled")?)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ImageCompositeMasked", "Image Composite Masked", "image/compositing")
                .input(link("destination", "IMAGE"))
                .input(link("source", "IMAGE"))
                .input(int_step("x", 0, 0, MAX_RESOLUTION, 1))
                .input(int_step("y", 0, 0, MAX_RESOLUTION, 1))
                .input(boolean("resize_source", false))
                .input(link("mask", "MASK").optional())
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let out = composite(i.image("destination")?, i.image("source")?, i.uint("x")?, i.uint("y")?, i.opt_mask("mask")?, i.boolean("resize_source")?, None)?;
            Ok(NodeOutput::one(Value::Image(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmImageCompositeMaskedColorCorrecting", "SwarmImageCompositeMaskedColorCorrecting", "SwarmUI/images")
                .describe("Works like ImageCompositeMasked, but does color correction for inpainted images (ie outside-the-mask areas are expected to be identical)")
                .input(link("destination", "IMAGE"))
                .input(link("source", "IMAGE"))
                .input(int_step("x", 0, 0, MAX_RESOLUTION, 1))
                .input(int_step("y", 0, 0, MAX_RESOLUTION, 1))
                .input(link("mask", "MASK"))
                .input(combo("correction_method", &["None", "Uniform", "Linear", "Linear2"]))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let method = i.string("correction_method")?;
            let out = composite(i.image("destination")?, i.image("source")?, i.uint("x")?, i.uint("y")?, Some(i.mask("mask")?), false, Some(method))?;
            Ok(NodeOutput::one(Value::Image(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmImageNoise", "SwarmImageNoise", "SwarmUI/images")
                .describe("Adds random noise to an image.")
                .input(link("image", "IMAGE"))
                .input(float("amount", 0.25, 0.0, 10.0, 0.01))
                .input(int("seed", 0, 0, MAX_SEED))
                .input(link("mask", "MASK").optional())
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let mut p = Px::of(i.image("image")?);
            let amount = i.float("amount")? as f32;
            let noise = gaussian(i.int("seed")? as u64, p.data.len());
            let m = match i.opt_mask("mask")? {
                Some(m) => Some(mask_plane(m, p.w, p.h, p.count)?),
                None => None,
            };
            for (k, v) in p.data.iter_mut().enumerate() {
                let scale = m.as_ref().map(|m| m[k / p.c]).unwrap_or(1.0);
                *v = (*v + noise[k] * amount * scale).clamp(0.0, 1.0);
            }
            Ok(NodeOutput::one(Value::Image(p.into_batch())))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("VAEEncodeForInpaint", "VAE Encode (for Inpainting)", "model/latent")
                .input(link("pixels", "IMAGE"))
                .input(link("vae", "VAE"))
                .input(link("mask", "MASK"))
                .input(int_step("grow_mask_by", 6, 0, 64, 1))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let (pixels, mask) = (i.image("pixels")?, i.mask("mask")?);
            let grow = i.uint("grow_mask_by")? as i64;
            let (w, h, n) = (pixels.width as i64, pixels.height as i64, pixels.count as usize);
            let m = mask_plane(mask, w as usize, h as usize, n)?;
            // conv2d(round(mask), ones(g, g), padding = ceil((g - 1) / 2)), clamped: a box dilation.
            let pad = (grow - 1 + 1) / 2;
            let grown: Vec<f32> = if grow == 0 {
                m.clone()
            } else {
                (0..n)
                    .flat_map(|b| {
                        let m = &m;
                        (0..h).flat_map(move |y| {
                            (0..w).map(move |x| {
                                let mut s = 0.0f32;
                                for dy in 0..grow {
                                    for dx in 0..grow {
                                        let (yy, xx) = (y - pad + dy, x - pad + dx);
                                        if yy >= 0 && xx >= 0 && yy < h && xx < w {
                                            s += m[(b * h as usize + yy as usize) * w as usize + xx as usize].round();
                                        }
                                    }
                                }
                                s.clamp(0.0, 1.0)
                            })
                        })
                    })
                    .collect()
            };
            let req = VaeEncodeRequest { vae: i.vae("vae")?.clone(), pixels: grey_out(pixels, mask)?, tiling: None };
            let mut latent = ctx.engine.vae_encode(&req)?;
            latent.noise_mask = Some(Mask { count: n as u32, height: h as u32, width: w as u32, data: Arc::new(grown.iter().map(|v| v.round()).collect()) });
            Ok(NodeOutput::one(Value::Latent(latent)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("InpaintModelConditioning", "InpaintModelConditioning", "model/conditioning")
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("vae", "VAE"))
                .input(link("pixels", "IMAGE"))
                .input(link("mask", "MASK"))
                .input(boolean("noise_mask", true).tooltip("Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."))
                .output("CONDITIONING", "positive")
                .output("CONDITIONING", "negative")
                .output("LATENT", "latent")
        },
        |i, ctx| {
            let (pixels, mask, vae) = (i.image("pixels")?, i.mask("mask")?, i.vae("vae")?);
            let pixel_mask = encode::resize_mask(mask, pixels.width, pixels.height)?;
            let concat = ctx.engine.vae_encode(&VaeEncodeRequest { vae: vae.clone(), pixels: grey_out(pixels, mask)?, tiling: None })?;
            let mut latent = ctx.engine.vae_encode(&VaeEncodeRequest { vae: vae.clone(), pixels: pixels.clone(), tiling: None })?;
            if i.boolean("noise_mask")? {
                latent.noise_mask = Some(pixel_mask.clone());
            }
            let inpaint = InpaintConcat { latent: concat, mask: pixel_mask };
            let pos = i.conditioning("positive")?.map_meta(|m| m.inpaint = Some(inpaint.clone()));
            let neg = i.conditioning("negative")?.map_meta(|m| m.inpaint = Some(inpaint.clone()));
            Ok(NodeOutput::values(vec![Value::Conditioning(pos), Value::Conditioning(neg), Value::Latent(latent)]))
        },
    );

    // ------------------------------------------------------------- latent math (engine-side)
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmLatentBlendMasked", "SwarmLatentBlendMasked", "SwarmUI/images")
                .describe("Blends two latent images together within a masked region.")
                .input(link("samples0", "LATENT"))
                .input(link("samples1", "LATENT"))
                .input(link("mask", "MASK"))
                .input(float("blend_factor", 0.5, 0.0, 1.0, 0.01).tooltip("The blend factor between the two samples. 0 means entirely use sample0, 1 means entirely sample1, 0.5 means 50/50 of each."))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let op = LatentOp::BlendMasked { a: i.latent("samples0")?.clone(), b: i.latent("samples1")?.clone(), mask: i.mask("mask")?.clone(), factor: i.float("blend_factor")? };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&op)?)))
        },
    );
    r.add(
        env,
        |_| NodeSchema::new("LatentAdd", "Add", "model/latent/advanced").input(link("samples1", "LATENT")).input(link("samples2", "LATENT")).output("LATENT", "LATENT"),
        |i, ctx| Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&LatentOp::Add { a: i.latent("samples1")?.clone(), b: i.latent("samples2")?.clone() })?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("LatentMultiply", "Multiply", "model/latent/advanced")
                .input(link("samples", "LATENT"))
                .input(float("multiplier", 1.0, -10.0, 10.0, 0.01))
                .output("LATENT", "LATENT")
        },
        |i, ctx| Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&LatentOp::Multiply { latent: i.latent("samples")?.clone(), factor: i.float("multiplier")? })?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("RepeatLatentBatch", "Repeat Latent Batch", "model/latent/batch")
                .input(link("samples", "LATENT"))
                .input(int("amount", 1, 1, 64))
                .output("LATENT", "LATENT")
        },
        |i, ctx| Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&LatentOp::RepeatBatch { latent: i.latent("samples")?.clone(), amount: i.uint("amount")? })?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("LatentFromBatch", "Get Latent From Batch", "model/latent/batch")
                .input(link("samples", "LATENT"))
                .input(int("batch_index", 0, -16384, 16384))
                .input(int("length", 1, 1, 64))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let latent = i.latent("samples")?;
            // Negative indices count from the end, as in Python.
            let n = latent.pixel.batch_size.max(1) as i64;
            let index = i.int("batch_index")?;
            let index = if index < 0 { (n + index).max(0) } else { index.min(n - 1) } as u32;
            Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&LatentOp::FromBatch { latent: latent.clone(), index, length: i.uint("length")? })?)))
        },
    );
    r.add(
        env,
        |_| NodeSchema::new("LatentBatch", "Batch Latents (DEPRECATED)", "model/latent/batch").input(link("samples1", "LATENT")).input(link("samples2", "LATENT")).output("LATENT", "LATENT"),
        |i, ctx| Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&LatentOp::Batch { a: i.latent("samples1")?.clone(), b: i.latent("samples2")?.clone() })?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("LatentUpscaleBy", "Upscale Latent By", "model/latent")
                .input(link("samples", "LATENT"))
                .input(combo("upscale_method", &["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]))
                .input(float("scale_by", 1.5, 0.01, 8.0, 0.01))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let op = LatentOp::UpscaleBy { latent: i.latent("samples")?.clone(), method: i.string("upscale_method")?.to_string(), scale: i.float("scale_by")? };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&op)?)))
        },
    );
    r.add(
        env,
        |_| NodeSchema::new("TrimVideoLatent", "Trim Video Latent", "model/latent").input(link("samples", "LATENT")).input(int("trim_amount", 0, 0, 99999)).output("LATENT", "LATENT"),
        |i, ctx| Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&LatentOp::TrimVideo { latent: i.latent("samples")?.clone(), trim_amount: i.uint("trim_amount")? })?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ReplaceVideoLatentFrames", "Replace Video Latent Frames", "model/latent/batch")
                .input(link("destination", "LATENT").tooltip("The destination latent where frames will be replaced."))
                .input(int_step("index", 0, -16384, 16384, 1).tooltip("The starting latent frame index in the destination latent where the source latent frames will be placed. Negative values count from the end."))
                .input(link("source", "LATENT").optional().tooltip("The source latent providing frames to insert into the destination latent. If not provided, the destination latent is returned unchanged."))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let destination = i.latent("destination")?;
            let Some(source) = i.opt_latent("source")? else { return Ok(NodeOutput::one(Value::Latent(destination.clone()))) };
            let (dest_frames, source_frames) = (latent_frames(destination) as i64, latent_frames(source) as i64);
            let mut index = i.int("index")?;
            if index < 0 {
                index += dest_frames;
            }
            // ComfyUI warns and returns the destination unchanged when the source doesn't fit.
            if index < 0 || index + source_frames > dest_frames {
                tracing::warn!("ReplaceVideoLatentFrames: {source_frames} frame(s) at {index} don't fit in {dest_frames}");
                return Ok(NodeOutput::one(Value::Latent(destination.clone())));
            }
            let op = LatentOp::ReplaceFrames { destination: destination.clone(), source: source.clone(), index: index as u32 };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&op)?)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("NormalizeVideoLatentStart", "NormalizeVideoLatentStart", "model/conditioning")
                .input(link("latent", "LATENT"))
                .input(int_step("start_frame_count", 4, 1, 16384, 1).tooltip("Number of latent frames to normalize, counted from the start"))
                .input(int_step("reference_frame_count", 5, 1, 16384, 1).tooltip("Number of latent frames after the start frames to use as reference"))
                .output("LATENT", "latent")
        },
        |i, ctx| {
            let latent = i.latent("latent")?;
            if latent_frames(latent) <= 1 {
                return Ok(NodeOutput::one(Value::Latent(latent.clone())));
            }
            let op = LatentOp::NormalizeStart { latent: latent.clone(), start_frame_count: i.uint("start_frame_count")?, reference_frame_count: i.uint("reference_frame_count")? };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.latent_op(&op)?)))
        },
    );
    r.add(
        env,
        |_| {
            let mut schedulers = vec!["turbo"];
            schedulers.extend_from_slice(SCHEDULERS);
            NodeSchema::new("SwarmUnsampler", "SwarmUnsampler", "SwarmUI/sampling")
                .describe("Runs sampling in reverse. The function of this is to create noise that matches an image, such that you can the run forward sampling with an altered version of the unsampling prompt to get a closely altered image. May not work on all models, may not work perfectly. Input values should largely match your Sampler inputs.")
                .input(link("model", "MODEL"))
                .input(int("steps", 20, 1, 10000))
                .input(combo("sampler_name", SAMPLERS))
                .input(combo("scheduler", &schedulers))
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("latent_image", "LATENT"))
                .input(int("start_at_step", 0, 0, 10000))
                .input(combo("previews", &["default", "none", "one"]))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let req = UnsampleRequest {
                model: i.model("model")?.clone(),
                positive: i.conditioning("positive")?.clone(),
                negative: i.conditioning("negative")?.clone(),
                latent: i.latent("latent_image")?.clone(),
                steps: i.uint("steps")?,
                start_at_step: i.uint("start_at_step")?,
                sampler: i.string("sampler_name")?.to_string(),
                scheduler: i.string("scheduler")?.to_string(),
                previews: PreviewMode::parse(i.string("previews")?),
            };
            let engine = ctx.engine;
            let out: Latent = engine.unsample(&req, ctx)?;
            Ok(NodeOutput::one(Value::Latent(Latent { empty: false, ..out })))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsv_roundtrip_and_rounding() {
        for rgb in [[0.2f32, 0.4, 0.6], [0.9, 0.1, 0.1], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]] {
            let [h, s, v] = rgb2hsv(rgb[0], rgb[1], rgb[2]);
            let back = hsv2rgb(h, s, v);
            for c in 0..3 {
                assert!((back[c] - rgb[c]).abs() < 1e-5, "{rgb:?} -> {back:?}");
            }
        }
        assert_eq!((py_round(2.5), py_round(3.5), py_round(2.4)), (2.0, 4.0, 2.0));
    }

    #[test]
    fn composite_pastes_through_mask() {
        let dest = ImageBatch::from_f32(1, 2, 2, 3, vec![0.0; 12]);
        let src = ImageBatch::from_f32(1, 2, 2, 3, vec![1.0; 12]);
        let mask = Mask { count: 1, height: 2, width: 2, data: Arc::new(vec![1.0, 0.0, 0.0, 1.0]) };
        let out = Px::of(&composite(&dest, &src, 0, 0, Some(&mask), false, Some("None")).unwrap());
        assert_eq!(out.data, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }
}
