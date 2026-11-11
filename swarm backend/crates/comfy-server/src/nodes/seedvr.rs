//! SeedVR2's nodes (`comfy_extras/nodes_seedvr.py`) as SwarmUI's SeedVR stage sends them: the padding before the VAE,
//! the conditioning, and the colour correction after it. Pixel work runs here; the model and its VAE are the engine's.

use swarm_engine_api::{CondMeta, Conditioning, ImageBatch, LatentFormat, LatentSamples, Opaque, SeedVr2Condition};

use super::{input_err, NodeError, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

const COLOR_METHODS: [&str; 4] = ["lab", "wavelet", "adain", "none"];

/// `_seedvr2_pad`: RGB only, clamped, zero-padded on the right and bottom to multiples of 16, and a clip's frames
/// padded with copies of the last to 4n + 1 (`cut_videos`). A batch is one clip, as in ComfyUI.
pub fn preprocess(img: &ImageBatch) -> Result<ImageBatch, NodeError> {
    if img.count == 0 {
        return Err(input_err("SeedVR2Preprocess expected at least one frame."));
    }
    if img.width.min(img.height) < 2 {
        return Err(input_err(format!("SeedVR2Preprocess: input shorter edge must be at least 2 pixels; got {}.", img.width.min(img.height))));
    }
    let (w, h, c) = (img.width as usize, img.height as usize, img.channels as usize);
    let (pw, ph) = (w.next_multiple_of(16), h.next_multiple_of(16));
    let t = img.count as usize;
    let frames = match t {
        1 => 1,
        2..=4 => 5,
        _ => t + (4 - (t - 1) % 4) % 4,
    };
    let mut out = vec![0f32; frames * ph * pw * 3];
    for f in 0..frames {
        let src = img.frame_f32(f.min(t - 1) as u32);
        let dst = &mut out[f * ph * pw * 3..];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3.min(c) {
                    dst[(y * pw + x) * 3 + ch] = src[(y * w + x) * c + ch].clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(ImageBatch::from_f32(frames as u32, ph as u32, pw as u32, 3, out))
}

/// Runs `f` over `items` on up to as many threads as the machine has.
fn par_each<T: Send>(items: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(items.len().max(1));
    let per = items.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for (k, chunk) in items.chunks_mut(per).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, item) in chunk.iter_mut().enumerate() {
                    f(k * per + j, item);
                }
            });
        }
    });
}

/// One channel plane, `h` rows of `w`.
struct Plane {
    w: usize,
    h: usize,
    v: Vec<f32>,
}

/// `wavelet_blur`: the 3x3 binomial kernel dilated by `radius` (at most a eighth of the short side), edges replicated.
/// The kernel is separable; rows first, then columns.
fn blur(p: &Plane, radius: usize) -> Vec<f32> {
    let r = radius.min((p.w.min(p.h) / 8).max(1));
    let (w, h) = (p.w, p.h);
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let row = &p.v[y * w..(y + 1) * w];
        let out = &mut tmp[y * w..(y + 1) * w];
        for x in 0..w {
            let (l, rr) = (x.saturating_sub(r), (x + r).min(w - 1));
            out[x] = 0.25 * row[l] + 0.5 * row[x] + 0.25 * row[rr];
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let (u, d) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = 0.25 * tmp[u * w + x] + 0.5 * tmp[y * w + x] + 0.25 * tmp[d * w + x];
        }
    }
    out
}

/// `wavelet_decomposition` over five levels: (high frequencies, low frequencies).
fn decompose(p: &Plane) -> (Vec<f32>, Vec<f32>) {
    let mut high = vec![0f32; p.v.len()];
    let mut image = Plane { w: p.w, h: p.h, v: p.v.clone() };
    for i in 0..5 {
        let low = blur(&image, 1 << i);
        for ((hv, iv), lv) in high.iter_mut().zip(&image.v).zip(&low) {
            *hv = (*hv + iv) - lv;
        }
        image.v = low;
    }
    (high, image.v)
}

/// `wavelet_reconstruction`: the content's high frequencies on the style's low ones, clamped to -1..1.
fn wavelet(content: &Plane, style: &Plane) -> Vec<f32> {
    let (high, _) = decompose(content);
    let (_, low) = decompose(style);
    high.iter().zip(&low).map(|(a, b)| (a + b).clamp(-1.0, 1.0)).collect()
}

const DELTA: f32 = 6.0 / 29.0;
const KAPPA: f32 = (29.0 / 3.0) * (29.0 / 3.0) * (29.0 / 3.0);
const XN: f32 = 0.95047;
const ZN: f32 = 1.08883;
const RGB_TO_XYZ: [[f32; 3]; 3] = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];
const XYZ_TO_RGB: [[f32; 3]; 3] = [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]];

fn rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let lin = rgb.map(|v| if v > 0.04045 { ((v + 0.055) / 1.055).powf(2.4) } else { v / 12.92 });
    let m = |r: [f32; 3]| r[0] * lin[0] + r[1] * lin[1] + r[2] * lin[2];
    let xyz = [m(RGB_TO_XYZ[0]) / XN, m(RGB_TO_XYZ[1]), m(RGB_TO_XYZ[2]) / ZN];
    let f = xyz.map(|v| if v > DELTA * DELTA * DELTA { v.powf(1.0 / 3.0) } else { (v * KAPPA + 16.0) / 116.0 });
    [f[1] * 116.0 - 16.0, (f[0] - f[1]) * 500.0, (f[1] - f[2]) * 200.0]
}

fn lab_to_rgb(lab: [f32; 3]) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    let inv = |f: f32| if f > DELTA { f * f * f } else { (f * 116.0 - 16.0) / KAPPA };
    let xyz = [inv(fx) * XN, inv(fy), inv(fz) * ZN];
    let m = |r: [f32; 3]| r[0] * xyz[0] + r[1] * xyz[1] + r[2] * xyz[2];
    [m(XYZ_TO_RGB[0]), m(XYZ_TO_RGB[1]), m(XYZ_TO_RGB[2])]
        .map(|v| if v > 0.0031308 { v.max(0.0).powf(1.0 / 2.4) * 1.055 - 0.055 } else { v * 12.92 }.clamp(0.0, 1.0))
}

/// `_histogram_matching_channel` for planes of equal size: each source value replaced by the reference value of the
/// same rank.
fn match_histogram(source: &[f32], reference: &[f32]) -> Vec<f32> {
    // an f32's bits made to sort as the value does, so the sorts compare integers in place
    let key = |v: f32| {
        let b = v.to_bits();
        if b >> 31 == 1 { !b } else { b | 1 << 31 }
    };
    let (mut order, mut sorted): (Vec<u64>, Vec<u32>) = std::thread::scope(|s| {
        let a = s.spawn(|| {
            let mut o: Vec<u64> = source.iter().enumerate().map(|(i, v)| (key(*v) as u64) << 32 | i as u64).collect();
            o.sort_unstable();
            o
        });
        let mut r: Vec<u32> = reference.iter().map(|v| key(*v)).collect();
        r.sort_unstable();
        (a.join().unwrap(), r)
    });
    let unkey = |k: u32| f32::from_bits(if k >> 31 == 1 { k & !(1 << 31) } else { !k });
    let mut out = vec![0f32; source.len()];
    for (rank, k) in order.iter_mut().enumerate() {
        out[(*k & 0xffff_ffff) as usize] = unkey(sorted[rank]);
    }
    sorted.clear();
    out
}

/// `f` over `0..n` in contiguous chunks on every thread, into one vector.
fn par_map<T: Send + Default + Clone>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut out = vec![T::default(); n];
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (k, chunk) in out.chunks_mut(per).enumerate() {
            let f = &f;
            s.spawn(move || chunk.iter_mut().enumerate().for_each(|(j, o)| *o = f(k * per + j)));
        }
    });
    out
}

/// `lab_color_transfer` (luminance weight 0.8) on one frame of planes in -1..1.
fn lab(content: &[Plane], style: &[Plane]) -> Vec<Vec<f32>> {
    let mut rebuilt: Vec<Vec<f32>> = vec![Vec::new(); 3];
    par_each(&mut rebuilt, |c, out| *out = wavelet(&content[c], &style[c]));
    let n = content[0].v.len();
    let to_lab = |planes: [&[f32]; 3]| -> Vec<[f32; 3]> {
        par_map(n, |i| rgb_to_lab([0, 1, 2].map(|c| ((planes[c][i] + 1.0) * 0.5).clamp(0.0, 1.0))))
    };
    let (cl, sl) = (to_lab([&rebuilt[0], &rebuilt[1], &rebuilt[2]]), to_lab([&style[0].v, &style[1].v, &style[2].v]));
    let mut matched: Vec<Vec<f32>> = vec![Vec::new(); 3];
    par_each(&mut matched, |c, out| {
        let src: Vec<f32> = cl.iter().map(|p| p[c]).collect();
        let reference: Vec<f32> = sl.iter().map(|p| p[c]).collect();
        *out = match_histogram(&src, &reference);
    });
    let rgb = par_map(n, |i| lab_to_rgb([cl[i][0] * 0.8 + matched[0][i] * (1.0 - 0.8), matched[1][i], matched[2][i]]));
    (0..3).map(|c| rgb.iter().map(|p| p[c] * 2.0 - 1.0).collect()).collect()
}

/// `adain_color_transfer`: each channel's mean and standard deviation matched to the style's, clamped to -1..1.
fn adain(content: &[Plane], style: &[Plane]) -> Vec<Vec<f32>> {
    let stats = |v: &[f32]| {
        let n = v.len() as f64;
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
        let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
        (mean as f32, ((var + 1e-5) as f32).sqrt())
    };
    (0..3)
        .map(|c| {
            let ((cm, cs), (sm, ss)) = (stats(&content[c].v), stats(&style[c].v));
            content[c].v.iter().map(|x| ((x - cm) / cs * ss + sm).clamp(-1.0, 1.0)).collect()
        })
        .collect()
}

/// `SeedVR2PostProcessing`: the decoded frames cropped to the picture they restore (the padding goes), their colours
/// matched to it, its alpha put back, and both sides cut to even.
pub fn postprocess(decoded: &ImageBatch, original: &ImageBatch, method: &str) -> Result<ImageBatch, NodeError> {
    if !COLOR_METHODS.contains(&method) {
        return Err(input_err(format!("SeedVR2PostProcessing: unknown color_correction_method '{method}'")));
    }
    let t = decoded.count.min(original.count) as usize;
    let (th, tw) = (decoded.height.min(original.height) as usize, decoded.width.min(original.width) as usize);
    // the reference is only resized when the decode came out smaller than it (it never does after SeedVR2Preprocess)
    let reference = if (original.height as usize, original.width as usize) != (th, tw) {
        crate::encode::resize(&original.to_rgb(), tw as u32, th as u32, "bicubic", "disabled")?
    } else {
        original.clone()
    };
    let alpha = original.has_alpha();
    let (h2, w2) = (th - th % 2, tw - tw % 2);
    let oc = if alpha { 4 } else { 3 };
    let mut out = Vec::with_capacity(t * h2 * w2 * oc);
    for f in 0..t {
        let dec = decoded.frame_f32(f as u32);
        let reff = reference.frame_f32(f as u32);
        let (dc, rc, dw, rw) = (decoded.channels as usize, reference.channels as usize, decoded.width as usize, reference.width as usize);
        let plane = |src: &[f32], c: usize, ch: usize, sw: usize, raw: bool| Plane {
            w: tw,
            h: th,
            v: (0..th).flat_map(|y| (0..tw).map(move |x| (y, x))).map(|(y, x)| {
                let v = src[(y * sw + x) * ch + c];
                if raw { v * 2.0 - 1.0 } else { v }
            }).collect(),
        };
        let channels: Vec<Vec<f32>> = if method == "none" {
            (0..3).map(|c| plane(&dec, c, dc, dw, false).v).collect()
        } else {
            let content: Vec<Plane> = (0..3).map(|c| plane(&dec, c, dc, dw, true)).collect();
            let style: Vec<Plane> = (0..3).map(|c| plane(&reff, c, rc, rw, true)).collect();
            let raw = match method {
                "lab" => lab(&content, &style),
                "wavelet" => {
                    let mut out: Vec<Vec<f32>> = vec![Vec::new(); 3];
                    par_each(&mut out, |c, o| *o = wavelet(&content[c], &style[c]));
                    out
                }
                _ => adain(&content, &style),
            };
            raw.into_iter().map(|v| v.into_iter().map(|x| ((x + 1.0) / 2.0).clamp(0.0, 1.0)).collect()).collect()
        };
        let orig = original.frame_f32(f as u32);
        let (ow, occ) = (original.width as usize, original.channels as usize);
        for y in 0..h2 {
            for x in 0..w2 {
                for c in 0..3 {
                    out.push(channels[c][y * tw + x]);
                }
                if alpha {
                    out.push(orig[(y * ow + x) * occ + 3]);
                }
            }
        }
    }
    Ok(ImageBatch::from_f32(t as u32, h2 as u32, w2 as u32, oc as u32, out))
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("SeedVR2Preprocess", "Pre-Process SeedVR2 Input", "image/pre-processors")
                .describe("Pad a resized image for SeedVR2 model. Alpha channel is dropped. The node Post-Process SeedVR2 Output re-applies it from the original resized image.")
                .input(link("resized_images", "IMAGE").tooltip("The resized image to process."))
                .output("IMAGE", "images")
        },
        |i, _| Ok(NodeOutput::one(Value::Image(preprocess(i.image("resized_images")?)?))),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SeedVR2Conditioning", "Apply SeedVR2 Conditioning", "model/conditioning")
                .describe("Build SeedVR2 positive/negative conditioning from a VAE latent.")
                .input(link("model", "MODEL").tooltip("The SeedVR2 model."))
                .input(link("vae_conditioning", "LATENT"))
                .output("CONDITIONING", "positive")
                .output("CONDITIONING", "negative")
        },
        |i, _| {
            i.model("model")?;
            let latent = i.latent("vae_conditioning")?;
            let LatentSamples::Single(t) = &latent.samples else {
                return Err(input_err("SeedVR2Conditioning expects a SeedVR2 VAE latent, not an audio/video one"));
            };
            if !matches!(t.format, LatentFormat::SeedVr2 | LatentFormat::EngineDefined) || t.shape.get(1) != Some(&16) {
                return Err(input_err(format!("SeedVR2Conditioning expects SeedVR2 VAE latents with 16 channels; got a {:?} latent of shape {:?}.", t.format, t.shape)));
            }
            let cond = |negative: bool| {
                let meta = CondMeta { seedvr2: Some(SeedVr2Condition { latent: latent.clone(), negative }), ..CondMeta::default() };
                Value::Conditioning(Conditioning::single(Opaque::none(), meta))
            };
            Ok(NodeOutput::values(vec![cond(false), cond(true)]))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SeedVR2PostProcessing", "Post-Process SeedVR2 Output", "image/post-processors")
                .describe("Align the generated image with the original resized image and apply color correction.")
                .input(link("images", "IMAGE").tooltip("The generated image to process."))
                .input(link("original_resized_images", "IMAGE").tooltip("The original resized image before pre-processing, used as reference."))
                .input(combo_default("color_correction_method", &COLOR_METHODS, "lab"))
                .output("IMAGE", "images")
        },
        |i, _| {
            let out = postprocess(i.image("images")?, i.image("original_resized_images")?, i.string("color_correction_method")?)?;
            Ok(NodeOutput::one(Value::Image(out)))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(count: u32, h: u32, w: u32, c: u32) -> ImageBatch {
        let n = (count * h * w * c) as usize;
        ImageBatch::from_f32(count, h, w, c, (0..n).map(|i| (i % 97) as f32 / 96.0).collect())
    }

    #[test]
    fn preprocess_pads_to_16_and_clips_to_4n_plus_1() {
        let p = preprocess(&image(1, 30, 17, 4)).unwrap();
        assert_eq!((p.count, p.height, p.width, p.channels), (1, 32, 32, 3));
        assert_eq!(p.frame_f32(0)[(31 * 32 + 31) * 3], 0.0);
        for (t, want) in [(2, 5), (4, 5), (5, 5), (6, 9), (9, 9), (10, 13)] {
            assert_eq!(preprocess(&image(t, 16, 16, 3)).unwrap().count, want, "{t} frames");
        }
    }

    #[test]
    fn postprocess_crops_the_padding_and_keeps_alpha() {
        let original = image(1, 30, 18, 4);
        let decoded = preprocess(&original).unwrap();
        for m in COLOR_METHODS {
            let out = postprocess(&decoded, &original, m).unwrap();
            assert_eq!((out.count, out.height, out.width, out.channels), (1, 30, 18, 4), "{m}");
        }
        // an image that is its own reference comes back the same through "none" and nearly so through "lab"
        let rgb = image(1, 64, 64, 3);
        let none = postprocess(&rgb, &rgb, "none").unwrap();
        assert_eq!(none.frame_f32(0), rgb.frame_f32(0));
    }

    #[test]
    fn lab_round_trips() {
        for rgb in [[0.2f32, 0.5, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.01, 0.6, 0.3]] {
            let back = lab_to_rgb(rgb_to_lab(rgb));
            for c in 0..3 {
                assert!((back[c] - rgb[c]).abs() < 1e-4, "{rgb:?} -> {back:?}");
            }
        }
    }
}
