//! [`LatentOp`]s on host latents, through [`diffusion::latent_ops`], with ComfyUI's bookkeeping of
//! each node: which input's `noise_mask` and `batch_index` the result keeps, and how batch ops
//! carry them. Image latents only; video and audio latents are refused.
//!
//! The engine keeps latents at the sampler's scale, but ComfyUI's `LATENT` values, which its nodes
//! compute on, are the VAE's raw output: Krea 2, Anima and Qwen Image 2.1 normalize per channel
//! only inside the sampler (`process_latent_in`). So each op runs on [`to_comfy`]'s values and
//! converts back. Scaling and slerp don't commute with that shift, and an empty latent is zero at
//! ComfyUI's scale, not the sampler's.

use std::borrow::Cow;
use std::sync::Arc;

use diffusion::latent_ops::{self, Shape};
use swarm_engine_api::*;

use super::{unsupported, HostLatent};

/// The per-channel `(mean, std)` between ComfyUI's scale and the sampler's: `None` where they're
/// the same (Flux 2 normalizes inside its VAE).
fn scale(format: LatentFormat) -> EngineResult<Option<(&'static [f32], &'static [f32])>> {
    match format {
        LatentFormat::Image16Ch => Ok(Some((&diffusion::vae::LATENT_MEAN, &diffusion::vae::LATENT_STD))),
        LatentFormat::QwenImage21 => Ok(Some((&diffusion::vae21::LATENT_MEAN, &diffusion::vae21::LATENT_STD))),
        LatentFormat::Flux2 => Ok(None),
        f => Err(unsupported(format!("latent operations on {f:?} latents"))),
    }
}

/// The engine's latent `[n, c, h, w]` at ComfyUI's scale (`process_latent_out`).
fn to_comfy(data: &[f32], [_, c, h, w]: Shape, format: LatentFormat) -> EngineResult<Vec<f32>> {
    let Some((mean, std)) = scale(format)? else { return Ok(data.to_vec()) };
    if mean.len() != c {
        return Err(unsupported(format!("a {c}-channel {format:?} latent")));
    }
    Ok(data.chunks_exact(h * w).enumerate().flat_map(|(i, plane)| plane.iter().map(move |v| v * std[i % c] + mean[i % c])).collect())
}

/// Back to the engine's scale (`process_latent_in`).
fn from_comfy(data: &mut [f32], [_, c, h, w]: Shape, format: LatentFormat) -> EngineResult<()> {
    if let Some((mean, std)) = scale(format)? {
        for (i, plane) in data.chunks_exact_mut(h * w).enumerate() {
            plane.iter_mut().for_each(|v| *v = (*v - mean[i % c]) / std[i % c]);
        }
    }
    Ok(())
}

/// The values of an image latent at ComfyUI's scale: sampled or encoded ones, or zeros for an
/// empty latent.
fn values(l: &Latent) -> EngineResult<(Cow<'_, [f32]>, Shape, LatentFormat)> {
    let LatentSamples::Single(t) = &l.samples else {
        return Err(unsupported("latent operations on audio/video latents"));
    };
    let Ok(shape) = Shape::try_from(t.shape.as_slice()) else {
        return Err(unsupported(format!("latent operations on latents of shape {:?}", t.shape)));
    };
    let data = match t.data.downcast_ref::<HostLatent>() {
        Some(h) => Cow::Owned(to_comfy(&h.data, shape, t.format)?),
        // Qwen Image 2.1's 4-channel placeholder is only ever empty
        None if t.data.is_none() => Cow::Owned(vec![0.0; shape.iter().product()]),
        None => return Err(unsupported("a latent this engine didn't make")),
    };
    Ok((data, shape, t.format))
}

/// `base` with new values at ComfyUI's scale. All zeros stays an empty latent, as ComfyUI's
/// samplers only test for zeros (`fix_empty_latent_channels`).
fn output(base: &Latent, format: LatentFormat, mut data: Vec<f32>, shape: Shape) -> EngineResult<Latent> {
    let empty = data.iter().all(|&v| v == 0.0);
    if !empty {
        from_comfy(&mut data, shape, format)?;
    }
    let data = if empty { Opaque::none() } else { Opaque::new(HostLatent { data }) };
    let mut out = base.with_samples(LatentSamples::Single(LatentTensor { format, shape: shape.to_vec(), data }));
    out.empty = empty;
    out.pixel.batch_size = shape[0] as u32;
    Ok(out)
}

/// Frames `range` of `mask`, cycled to `batch` frames first (ComfyUI's mask handling in the
/// batch nodes).
fn mask_frames(mask: &Mask, frames: impl Iterator<Item = usize>) -> Mask {
    let plane = (mask.width * mask.height) as usize;
    let n = mask.count.max(1) as usize;
    let data: Vec<f32> = frames.flat_map(|f| mask.data[(f % n) * plane..(f % n + 1) * plane].iter().copied()).collect();
    Mask { count: (data.len() / plane.max(1)) as u32, height: mask.height, width: mask.width, data: Arc::new(data) }
}

/// A SeedVR2 latent's values and shape `[B, C, T, H, W]` (a 4D one is one frame). Its engine and ComfyUI scales agree.
fn video(l: &Latent) -> EngineResult<(&[f32], [usize; 5])> {
    let LatentSamples::Single(t) = &l.samples else {
        return Err(unsupported("frame operations on audio/video latents"));
    };
    if t.format != LatentFormat::SeedVr2 {
        return Err(unsupported(format!("frame operations on {:?} latents", t.format)));
    }
    let shape = match *t.shape.as_slice() {
        [b, c, f, h, w] => [b, c, f, h, w],
        [b, c, h, w] => [b, c, 1, h, w],
        _ => return Err(unsupported(format!("frame operations on latents of shape {:?}", t.shape))),
    };
    let h = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("a latent this engine didn't make"))?;
    if h.data.len() != shape.iter().product::<usize>() {
        return Err(EngineError::InvalidInput(format!("a latent of {} values for shape {:?}", h.data.len(), t.shape)));
    }
    Ok((&h.data, shape))
}

/// `base` with a 5D SeedVR2 latent of `shape`, its pixel frames 4 (T - 1) + 1.
fn video_output(base: &Latent, data: Vec<f32>, shape: [usize; 5]) -> Latent {
    let t = LatentTensor { format: LatentFormat::SeedVr2, shape: shape.to_vec(), data: Opaque::new(HostLatent { data }) };
    let mut out = base.with_samples(LatentSamples::Single(t));
    out.empty = false;
    out.pixel.frames = (4 * (shape[2] - 1) + 1) as u32;
    out
}

/// `_seedvr2_chunk_crossfade_weights`: the previous chunk's weights over `n` overlapping frames, flat at 1 over the
/// first third, a Hann fall over the middle one, flat at 0 over the last.
pub fn crossfade_weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            let r = ((t - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0);
            0.5 + 0.5 * (std::f32::consts::PI * r).cos()
        })
        .collect()
}

/// `SeedVR2TemporalMerge` on chunks of `[planes][frames][hw]` values: each chunk starts `overlap` frames before the
/// previous one's end and fades in over the frames already filled. The merged values and their frames.
fn merge_chunks(chunks: &[(&[f32], usize)], planes: usize, hw: usize, overlap: usize) -> (Vec<f32>, usize) {
    let chunk_t = chunks[0].1;
    let step = chunk_t - overlap.min(chunk_t - 1);
    let total = step * (chunks.len() - 1) + chunks.last().map_or(0, |c| c.1);
    let mut out = vec![0f32; planes * total * hw];
    let mut filled = 0usize;
    for (i, &(x, t)) in chunks.iter().enumerate() {
        let start = i * step;
        let fade = filled.saturating_sub(start).min(t);
        let wts = crossfade_weights(fade);
        for p in 0..planes {
            for f in 0..t {
                let src = &x[(p * t + f) * hw..][..hw];
                let dst = &mut out[(p * total + start + f) * hw..][..hw];
                if f < fade {
                    let wp = wts[f];
                    dst.iter_mut().zip(src).for_each(|(d, s)| *d = *d * wp + *s * (1.0 - wp));
                } else {
                    dst.copy_from_slice(src);
                }
            }
        }
        filled = start + t;
    }
    (out, total)
}

fn checked(e: anyhow::Error) -> EngineError {
    unsupported(format!("{e:#}"))
}

pub fn latent_op(op: &LatentOp) -> EngineResult<Latent> {
    match op {
        LatentOp::Add { a, b } => {
            let ((x, sa, format), (y, sb, _)) = (values(a)?, values(b)?);
            output(a, format, latent_ops::add(&x, sa, &y, sb).map_err(checked)?, sa)
        }
        LatentOp::Multiply { latent, factor } => {
            let (x, s, format) = values(latent)?;
            output(latent, format, latent_ops::multiply(&x, *factor), s)
        }
        LatentOp::BlendMasked { a, b, mask, factor } => {
            if mask.count != 1 {
                return Err(unsupported(format!("blending latents with a mask of {} frames", mask.count)));
            }
            let ((x, sa, format), (y, sb, _)) = (values(a)?, values(b)?);
            let data = latent_ops::blend_masked(&x, sa, &y, sb, &mask.data, (mask.height as usize, mask.width as usize), *factor).map_err(checked)?;
            output(a, format, data, sa)
        }
        LatentOp::RepeatBatch { latent, amount } => {
            let (x, s, format) = values(latent)?;
            let amount = *amount as usize;
            let mut out = output(latent, format, latent_ops::repeat(&x, amount), [s[0] * amount, s[1], s[2], s[3]])?;
            // as in ComfyUI, the mask repeats as it was, not cycled to the batch first
            out.noise_mask = latent.noise_mask.as_ref().map(|m| if m.count > 1 { mask_frames(m, 0..m.count as usize * amount) } else { m.clone() });
            out.batch_index = latent.batch_index.as_ref().map(|b| {
                let offset = b.iter().max().unwrap_or(&0) - b.iter().min().unwrap_or(&0) + 1;
                (0..amount as i64).flat_map(|i| b.iter().map(move |x| x + i * offset)).collect()
            });
            Ok(out)
        }
        LatentOp::FromBatch { latent, index, length } => {
            let (x, s, format) = values(latent)?;
            let index = (*index as usize).min(s[0] - 1);
            let length = (*length as usize).min(s[0] - index);
            let item = s[1] * s[2] * s[3];
            let data = x[index * item..(index + length) * item].to_vec();
            let mut out = output(latent, format, data, [length, s[1], s[2], s[3]])?;
            out.noise_mask = latent.noise_mask.as_ref().map(|m| if m.count == 1 { m.clone() } else { mask_frames(m, index..index + length) });
            out.batch_index = Some(match &latent.batch_index {
                Some(b) => b[index.min(b.len())..(index + length).min(b.len())].to_vec(),
                None => (index as i64..(index + length) as i64).collect(),
            });
            Ok(out)
        }
        LatentOp::Batch { a, b } => {
            let ((x, sa, format), (y, sb, _)) = (values(a)?, values(b)?);
            let (data, shape) = latent_ops::batch(&x, sa, &y, sb).map_err(checked)?;
            let mut out = output(a, format, data, shape)?;
            let indices = |l: &Latent, n: usize| l.batch_index.clone().unwrap_or_else(|| (0..n as i64).collect());
            out.batch_index = Some([indices(a, sa[0]), indices(b, sb[0])].concat());
            Ok(out)
        }
        LatentOp::UpscaleBy { latent, method, scale } => {
            if !latent_ops::UPSCALE_METHODS.contains(&method.as_str()) {
                return Err(unsupported(format!("latent upscale method {method}")));
            }
            let (x, s, format) = values(latent)?;
            let (data, shape) = latent_ops::upscale_by(&x, s, method, *scale).map_err(checked)?;
            if shape[2] == 0 || shape[3] == 0 {
                return Err(EngineError::InvalidInput(format!("upscaling a {}x{} latent by {scale} leaves nothing", s[3], s[2])));
            }
            let mut out = output(latent, format, data, shape)?;
            out.pixel.width = (latent.pixel.width as usize * shape[3] / s[3]) as u32;
            out.pixel.height = (latent.pixel.height as usize * shape[2] / s[2]) as u32;
            Ok(out)
        }
        LatentOp::SliceFrames { latent, start, end } => {
            let (x, [b, c, t, h, w]) = video(latent)?;
            let (start, end) = (*start as usize, (*end as usize).min(t));
            if start >= end {
                return Err(EngineError::InvalidInput(format!("latent frames {start}..{end} of {t}")));
            }
            let (hw, n) = (h * w, end - start);
            let data: Vec<f32> = (0..b * c).flat_map(|bc| x[(bc * t + start) * hw..(bc * t + end) * hw].iter().copied()).collect();
            Ok(video_output(latent, data, [b, c, n, h, w]))
        }
        LatentOp::MergeChunks { chunks, overlap } => {
            let first = chunks.first().ok_or_else(|| EngineError::InvalidInput("no latents to merge".into()))?;
            let parts = chunks.iter().map(video).collect::<EngineResult<Vec<_>>>()?;
            let [b, c, chunk_t, h, w] = parts[0].1;
            for (i, (_, s)) in parts.iter().enumerate().skip(1) {
                if s[..2] != [b, c] || s[3..] != [h, w] {
                    return Err(EngineError::InvalidInput(format!("chunk {i} of shape {s:?} doesn't match chunk 0's {:?} outside the frames", parts[0].1)));
                }
                if i + 1 < parts.len() && s[2] != chunk_t {
                    return Err(EngineError::InvalidInput(format!("chunk {i} has {} latent frames but chunk 0 has {chunk_t}; only the last may be shorter", s[2])));
                }
            }
            if parts.len() == 1 {
                let mut out = first.clone();
                out.noise_mask = None;
                return Ok(out);
            }
            let (data, total) = merge_chunks(&parts.iter().map(|(x, s)| (*x, s[2])).collect::<Vec<_>>(), b * c, h * w, *overlap as usize);
            let mut merged = video_output(first, data, [b, c, total, h, w]);
            merged.noise_mask = None;
            Ok(merged)
        }
        other => {
            let name = format!("{other:?}");
            Err(unsupported(format!("latent operation {}", name.split([' ', '{', '(']).next().unwrap_or("?"))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent_in(format: LatentFormat, data: Vec<f32>, shape: Shape) -> Latent {
        let pixel = PixelInfo::image(shape[3] as u32 * 8, shape[2] as u32 * 8, shape[0] as u32);
        Latent::single(LatentTensor { format, shape: shape.to_vec(), data: Opaque::new(HostLatent { data }) }, pixel)
    }

    /// Flux 2's scale is ComfyUI's, so the bookkeeping tests can use any channel count.
    fn latent(data: Vec<f32>, shape: Shape) -> Latent {
        latent_in(LatentFormat::Flux2, data, shape)
    }

    /// The stored values, at the engine's scale.
    fn data(l: &Latent) -> Vec<f32> {
        let LatentSamples::Single(t) = &l.samples else { unreachable!() };
        t.data.downcast_ref::<HostLatent>().map(|h| h.data.clone()).unwrap_or_default()
    }

    fn empty(format: LatentFormat, shape: Shape) -> Latent {
        let mut l = latent_in(format, Vec::new(), shape);
        l.samples = LatentSamples::Single(LatentTensor { format, shape: shape.to_vec(), data: Opaque::none() });
        l.empty = true;
        l
    }

    fn clip(data: Vec<f32>, shape: [usize; 5]) -> Latent {
        let t = LatentTensor { format: LatentFormat::SeedVr2, shape: shape.to_vec(), data: Opaque::new(HostLatent { data }) };
        Latent::single(t, PixelInfo::image(shape[4] as u32 * 8, shape[3] as u32 * 8, 1))
    }

    #[test]
    fn seedvr2_chunks_slice_and_merge_like_comfy() {
        // two planes of 5 frames of one value each: plane * 100 + frame
        let x: Vec<f32> = (0..2).flat_map(|p| (0..5).map(move |f| (p * 100 + f) as f32)).collect();
        let whole = clip(x.clone(), [1, 2, 5, 1, 1]);
        let a = latent_op(&LatentOp::SliceFrames { latent: whole.clone(), start: 0, end: 4 }).unwrap();
        let b = latent_op(&LatentOp::SliceFrames { latent: whole, start: 2, end: 5 }).unwrap();
        assert_eq!(data(&b), [2.0, 3.0, 4.0, 102.0, 103.0, 104.0]);
        assert_eq!(a.pixel.frames, 13);
        // slices of one clip merge back to it whatever the fade
        let m = latent_op(&LatentOp::MergeChunks { chunks: vec![a.clone(), b], overlap: 2 }).unwrap();
        assert_eq!(data(&m), x);
        // ComfyUI's weights: a linspace over the fade, ramped on its middle third
        assert_eq!(crossfade_weights(1), [1.0]);
        assert_eq!(crossfade_weights(2), [1.0, 0.0]);
        let w = crossfade_weights(7);
        assert!((w[0] - 1.0).abs() < 1e-6 && (w[2] - 1.0).abs() < 1e-6 && (w[3] - 0.5).abs() < 1e-6 && w[4].abs() < 1e-6, "{w:?}");
        // a different chunk is blended in by those weights: two overlapping frames, [1, 0]
        let b2 = clip(vec![9.0; 6], [1, 2, 3, 1, 1]);
        let m = latent_op(&LatentOp::MergeChunks { chunks: vec![a, b2], overlap: 2 }).unwrap();
        assert_eq!(data(&m), [0.0, 1.0, 2.0, 9.0, 9.0, 100.0, 101.0, 102.0, 9.0, 9.0]);
    }

    #[test]
    fn ops_run_at_comfys_scale() {
        use diffusion::vae::{LATENT_MEAN as MEAN, LATENT_STD as STD};
        // Init Image Reset To Norm: empty * f + latent * (1 - f), on ComfyUI's raw latent
        let x: Vec<f32> = (0..16).map(|c| c as f32 * 0.1 - 0.7).collect();
        let encoded = latent_in(LatentFormat::Image16Ch, x.clone(), [1, 16, 1, 1]);
        let zero = latent_op(&LatentOp::Multiply { latent: empty(LatentFormat::Image16Ch, [1, 16, 1, 1]), factor: 0.3 }).unwrap();
        assert!(zero.empty);
        let kept = latent_op(&LatentOp::Multiply { latent: encoded, factor: 0.7 }).unwrap();
        let reset = latent_op(&LatentOp::Add { a: zero, b: kept }).unwrap();
        for (c, v) in data(&reset).iter().enumerate() {
            let want = ((x[c] * STD[c] + MEAN[c]) * 0.7 - MEAN[c]) / STD[c];
            assert!((v - want).abs() < 1e-5, "channel {c}: {v} vs {want}");
        }
    }

    fn ramp(shape: Shape) -> Latent {
        latent((1..=shape.iter().product::<usize>()).map(|v| v as f32).collect(), shape)
    }

    #[test]
    fn empty_latents_count_as_zeros_and_stay_empty() {
        let empty = empty(LatentFormat::Flux2, [2, 1, 2, 2]);
        let repeated = latent_op(&LatentOp::RepeatBatch { latent: empty.clone(), amount: 3 }).unwrap();
        assert!(repeated.empty);
        assert_eq!(repeated.pixel.batch_size, 6);
        let added = latent_op(&LatentOp::Add { a: empty, b: ramp([2, 1, 2, 2]) }).unwrap();
        assert!(!added.empty);
        assert_eq!(data(&added), (1..=8).map(|v| v as f32).collect::<Vec<_>>());
    }

    #[test]
    fn batch_nodes_keep_comfys_batch_index() {
        let l = ramp([3, 1, 1, 2]);
        let picked = latent_op(&LatentOp::FromBatch { latent: l.clone(), index: 1, length: 5 }).unwrap();
        assert_eq!(data(&picked), [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(picked.batch_index, Some(vec![1, 2]));
        let repeated = latent_op(&LatentOp::RepeatBatch { latent: picked, amount: 2 }).unwrap();
        assert_eq!(repeated.batch_index, Some(vec![1, 2, 3, 4]));
        let joined = latent_op(&LatentOp::Batch { a: repeated, b: l }).unwrap();
        assert_eq!(joined.batch_index, Some(vec![1, 2, 3, 4, 0, 1, 2]));
        assert_eq!(joined.pixel.batch_size, 7);
    }

    #[test]
    fn masks_follow_the_batch() {
        let mask = Mask { count: 2, height: 1, width: 1, data: Arc::new(vec![0.25, 0.75]) };
        let l = Latent { noise_mask: Some(mask), ..ramp([4, 1, 1, 1]) };
        let picked = latent_op(&LatentOp::FromBatch { latent: l.clone(), index: 2, length: 2 }).unwrap();
        assert_eq!(*picked.noise_mask.unwrap().data, [0.25, 0.75]);
        let repeated = latent_op(&LatentOp::RepeatBatch { latent: l, amount: 2 }).unwrap();
        assert_eq!(*repeated.noise_mask.unwrap().data, [0.25, 0.75, 0.25, 0.75]);
    }

    #[test]
    fn upscale_tracks_the_pixel_size() {
        let up = latent_op(&LatentOp::UpscaleBy { latent: ramp([1, 2, 4, 6]), method: "bislerp".into(), scale: 1.5 }).unwrap();
        let LatentSamples::Single(t) = &up.samples else { unreachable!() };
        assert_eq!(t.shape, [1, 2, 6, 9]);
        assert_eq!((up.pixel.width, up.pixel.height), (72, 48));
        let bad = latent_op(&LatentOp::UpscaleBy { latent: ramp([1, 2, 4, 6]), method: "lanczos".into(), scale: 2.0 });
        assert!(matches!(bad, Err(EngineError::Unsupported(_))));
    }

    #[test]
    fn video_latents_and_other_ops_are_refused() {
        let video = Latent::single(LatentTensor { format: LatentFormat::MiniMaxH3Video, shape: vec![1, 24, 2, 4, 4], data: Opaque::none() }, PixelInfo::image(64, 64, 1));
        assert!(matches!(latent_op(&LatentOp::Multiply { latent: video, factor: 2.0 }), Err(EngineError::Unsupported(_))));
        assert!(matches!(latent_op(&LatentOp::TrimVideo { latent: ramp([1, 1, 1, 1]), trim_amount: 1 }), Err(EngineError::Unsupported(_))));
    }
}
