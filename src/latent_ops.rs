//! ComfyUI's latent nodes on host latents `[n, c, h, w]`: `LatentAdd`, `LatentMultiply`,
//! `SwarmLatentBlendMasked`, `RepeatLatentBatch`, `LatentFromBatch`, `LatentBatch` and
//! `LatentUpscaleBy`, with torch's `interpolate` and ComfyUI's `common_upscale` and `bislerp`
//! written out so the results match (checked by `tests/latent_ops_match_comfy.rs`).

use anyhow::{bail, ensure, Result};

/// `[n, c, h, w]`
pub type Shape = [usize; 4];

/// `LatentUpscaleBy`'s methods.
pub const UPSCALE_METHODS: [&str; 5] = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"];

fn len(s: Shape) -> usize {
    s.iter().product()
}

/// torch's `area_pixel_compute_source_index` without `align_corners`, with the scale `in / out`
/// in f32 as on the CPU.
fn source_index(scale: f32, dst: usize, cubic: bool) -> f32 {
    let src = scale * (dst as f32 + 0.5) - 0.5;
    if !cubic && src < 0.0 { 0.0 } else { src }
}

/// One axis of a separable resize: for each output index, (input index, weight) pairs.
type Taps = Vec<Vec<(usize, f32)>>;

fn taps(input: usize, output: usize, method: &str) -> Taps {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|i| match method {
            "nearest-exact" => vec![((((i as f32 + 0.5) * scale).floor() as usize).min(input - 1), 1.0)],
            "bilinear" => {
                let real = source_index(scale, i, false);
                let index = (real.floor() as usize).min(input - 1);
                let lambda = (real - index as f32).clamp(0.0, 1.0);
                let next = if index < input - 1 { index + 1 } else { index };
                vec![(index, 1.0 - lambda), (next, lambda)]
            }
            "area" => {
                // adaptive_avg_pool2d
                let (start, end) = (i * input / output, ((i + 1) * input).div_ceil(output));
                let w = 1.0 / (end - start) as f32;
                (start..end).map(|k| (k, w)).collect()
            }
            "bicubic" => {
                let real = source_index(scale, i, true);
                let index = real.floor();
                let t = real - index;
                const A: f32 = -0.75;
                let near = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
                let far = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
                let weights = [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)];
                (0..4).map(|j| ((index as i64 - 1 + j as i64).clamp(0, input as i64 - 1) as usize, weights[j])).collect()
            }
            _ => unreachable!("checked by interpolate"),
        })
        .collect()
}

/// `torch.nn.functional.interpolate(x, size=(oh, ow), mode=method)` for `nearest-exact`,
/// `bilinear`, `area` and `bicubic`, and ComfyUI's `bislerp`.
pub fn interpolate(x: &[f32], s: Shape, oh: usize, ow: usize, method: &str) -> Result<Vec<f32>> {
    let [n, c, h, w] = s;
    ensure!(x.len() == len(s), "latent of {} values for shape {s:?}", x.len());
    ensure!(oh > 0 && ow > 0 && h > 0 && w > 0, "resizing {h}x{w} to {oh}x{ow}");
    if method == "bislerp" {
        return Ok(bislerp(x, s, oh, ow));
    }
    if !UPSCALE_METHODS.contains(&method) {
        bail!("upscale method {method}");
    }
    let (ty, tx) = (taps(h, oh, method), taps(w, ow, method));
    let mut out = Vec::with_capacity(n * c * oh * ow);
    let mut row = vec![0f32; ow];
    for plane in x.chunks_exact(h * w) {
        // along x for each input row is recomputed per output row: these latents are small
        for ys in &ty {
            row.iter_mut().for_each(|v| *v = 0.0);
            for &(y, wy) in ys {
                let line = &plane[y * w..(y + 1) * w];
                for (o, xs) in row.iter_mut().zip(&tx) {
                    *o += wy * xs.iter().map(|&(x, wx)| wx * line[x]).sum::<f32>();
                }
            }
            out.extend_from_slice(&row);
        }
    }
    Ok(out)
}

/// ComfyUI's `slerp` inside `bislerp`, on one pair of channel vectors.
fn slerp(b1: &[f32], b2: &[f32], r: f32, out: &mut [f32]) {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let (n1, n2) = (norm(b1), norm(b2));
    let unit = |v: f32, n: f32| if n == 0.0 { 0.0 } else { v / n };
    let dot: f32 = b1.iter().zip(b2).map(|(&a, &b)| unit(a, n1) * unit(b, n2)).sum();
    if dot > (1.0 - 1e-5) as f32 {
        out.copy_from_slice(b1);
        return;
    }
    if dot < (1e-5 - 1.0) as f32 {
        for ((o, a), b) in out.iter_mut().zip(b1).zip(b2) {
            *o = a * (1.0 - r) + b * r;
        }
        return;
    }
    let omega = dot.acos();
    let so = omega.sin();
    let (k1, k2) = (((1.0 - r) * omega).sin() / so, (r * omega).sin() / so);
    let scale = n1 * (1.0 - r) + n2 * r;
    for ((o, a), b) in out.iter_mut().zip(b1).zip(b2) {
        *o = (k1 * unit(*a, n1) + k2 * unit(*b, n2)) * scale;
    }
}

/// `generate_bilinear_data` of `bislerp`: per output index, the ratio and the two input indices.
fn bislerp_axis(old: usize, new: usize) -> Vec<(f32, usize, usize)> {
    let t = taps(old, new, "bilinear");
    let at = |v: &dyn Fn(usize) -> f32, taps: &[(usize, f32)]| taps.iter().map(|&(k, w)| w * v(k)).sum::<f32>();
    t.iter()
        .map(|taps| {
            let c1 = at(&|k| k as f32, taps);
            let c2 = at(&|k| if k == old - 1 { k as f32 } else { k as f32 + 1.0 }, taps);
            (c1 - c1.floor(), c1 as usize, c2 as usize)
        })
        .collect()
}

/// ComfyUI's `bislerp`: slerp of the channel vectors, along x then along y.
fn bislerp(x: &[f32], [n, c, h, w]: Shape, oh: usize, ow: usize) -> Vec<f32> {
    let (ax, ay) = (bislerp_axis(w, ow), bislerp_axis(h, oh));
    let gather = |src: &[f32], stride: usize, at: usize| -> Vec<f32> { (0..c).map(|ch| src[ch * stride + at]).collect() };
    let mut out = vec![0f32; n * c * oh * ow];
    let mut mid = vec![0f32; c * h * ow];
    let mut v = vec![0f32; c];
    for b in 0..n {
        let img = &x[b * c * h * w..(b + 1) * c * h * w];
        for y in 0..h {
            for (j, &(r, i1, i2)) in ax.iter().enumerate() {
                slerp(&gather(img, h * w, y * w + i1), &gather(img, h * w, y * w + i2), r, &mut v);
                for ch in 0..c {
                    mid[ch * h * ow + y * ow + j] = v[ch];
                }
            }
        }
        let dst = &mut out[b * c * oh * ow..(b + 1) * c * oh * ow];
        for (i, &(r, i1, i2)) in ay.iter().enumerate() {
            for j in 0..ow {
                slerp(&gather(&mid, h * ow, i1 * ow + j), &gather(&mid, h * ow, i2 * ow + j), r, &mut v);
                for ch in 0..c {
                    dst[ch * oh * ow + i * ow + j] = v[ch];
                }
            }
        }
    }
    out
}

/// ComfyUI's `common_upscale` to `ow x oh`, with `crop = "center"` or `"disabled"`.
pub fn common_upscale(x: &[f32], s: Shape, ow: usize, oh: usize, method: &str, center: bool) -> Result<Vec<f32>> {
    let [n, c, h, w] = s;
    if !center {
        return interpolate(x, s, oh, ow, method);
    }
    let (old_aspect, new_aspect) = (w as f64 / h as f64, ow as f64 / oh as f64);
    let (mut cx, mut cy) = (0, 0);
    if old_aspect > new_aspect {
        cx = ((w as f64 - w as f64 * (new_aspect / old_aspect)) / 2.0).round_ties_even() as usize;
    } else if old_aspect < new_aspect {
        cy = ((h as f64 - h as f64 * (old_aspect / new_aspect)) / 2.0).round_ties_even() as usize;
    }
    let (ch, cw) = (h - 2 * cy, w - 2 * cx);
    let mut cropped = Vec::with_capacity(n * c * ch * cw);
    for plane in x.chunks_exact(h * w) {
        for y in cy..cy + ch {
            cropped.extend_from_slice(&plane[y * w + cx..y * w + cx + cw]);
        }
    }
    interpolate(&cropped, [n, c, ch, cw], oh, ow, method)
}

/// `comfy.utils.repeat_to_batch_size`: cycle or cut the batch to `batch`.
pub fn repeat_to_batch(x: &[f32], s: Shape, batch: usize) -> Vec<f32> {
    let item = s[1] * s[2] * s[3];
    (0..batch).flat_map(|i| &x[(i % s[0]) * item..(i % s[0] + 1) * item]).copied().collect()
}

/// `reshape_latent_to`: `b` resized (bilinear, center crop) to `target`'s size when the rest of
/// its shape differs, then with `repeat_batch` cycled to `target`'s batch.
fn reshape_to(b: &[f32], sb: Shape, target: Shape, repeat_batch: bool) -> Result<(Vec<f32>, Shape)> {
    ensure!(sb[1] == target[1], "latents of {} and {} channels", target[1], sb[1]);
    let (mut b, mut sb) = (b.to_vec(), sb);
    if sb[1..] != target[1..] {
        b = common_upscale(&b, sb, target[3], target[2], "bilinear", true)?;
        sb = [sb[0], sb[1], target[2], target[3]];
    }
    if repeat_batch {
        b = repeat_to_batch(&b, sb, target[0]);
        sb[0] = target[0];
    }
    Ok((b, sb))
}

/// `LatentAdd`: `a + b`, `b` resized and cycled to `a`.
pub fn add(a: &[f32], sa: Shape, b: &[f32], sb: Shape) -> Result<Vec<f32>> {
    let (b, _) = reshape_to(b, sb, sa, true)?;
    Ok(a.iter().zip(&b).map(|(x, y)| x + y).collect())
}

/// `LatentMultiply`. The multiplier is a Python float applied to a float32 tensor.
pub fn multiply(x: &[f32], factor: f64) -> Vec<f32> {
    let f = factor as f32;
    x.iter().map(|v| v * f).collect()
}

/// `LatentBatch`: `a` then `b` (resized to `a`), along the batch.
pub fn batch(a: &[f32], sa: Shape, b: &[f32], sb: Shape) -> Result<(Vec<f32>, Shape)> {
    let (b, sb) = reshape_to(b, sb, sa, false)?;
    let mut out = a.to_vec();
    out.extend_from_slice(&b);
    Ok((out, [sa[0] + sb[0], sa[1], sa[2], sa[3]]))
}

/// `RepeatLatentBatch`: the whole batch `amount` times.
pub fn repeat(x: &[f32], amount: usize) -> Vec<f32> {
    x.repeat(amount)
}

/// `SwarmLatentBlendMasked`: `a * (1 - m * factor) + b * (m * factor)`, with `b` resized (bicubic)
/// to `a` when their shapes differ and the one-frame mask `[mh, mw]` resized (bicubic) to `a`.
/// `b` must have `a`'s batch or a batch of 1, as torch broadcasting needs.
pub fn blend_masked(a: &[f32], sa: Shape, b: &[f32], sb: Shape, mask: &[f32], (mh, mw): (usize, usize), factor: f64) -> Result<Vec<f32>> {
    let [n, c, h, w] = sa;
    ensure!(sb[1] == c, "latents of {c} and {} channels", sb[1]);
    ensure!(sb[0] == n || sb[0] == 1, "latent batches of {n} and {}", sb[0]);
    let b = if sb != sa { interpolate(b, sb, h, w, "bicubic")? } else { b.to_vec() };
    let m = interpolate(mask, [1, 1, mh, mw], h, w, "bicubic")?;
    let f = factor as f32;
    let item = c * h * w;
    let mut out = Vec::with_capacity(a.len());
    for (i, v) in a.iter().enumerate() {
        let k = m[i % (h * w)] * f;
        let bv = b[if sb[0] == 1 { i % item } else { i }];
        out.push(v * (1.0 - k) + bv * k);
    }
    Ok(out)
}

/// `LatentUpscaleBy`: the new size is the old one times `scale`, rounded (Python's `round`).
pub fn upscale_by(x: &[f32], s: Shape, method: &str, scale: f64) -> Result<(Vec<f32>, Shape)> {
    let ow = (s[3] as f64 * scale).round_ties_even() as usize;
    let oh = (s[2] as f64 * scale).round_ties_even() as usize;
    Ok((common_upscale(x, s, ow, oh, method, false)?, [s[0], s[1], oh, ow]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_is_identity() {
        let x: Vec<f32> = (0..2 * 3 * 4 * 5).map(|i| (i as f32 * 0.37).sin()).collect();
        for m in UPSCALE_METHODS {
            let y = interpolate(&x, [2, 3, 4, 5], 4, 5, m).unwrap();
            for (a, b) in x.iter().zip(&y) {
                assert!((a - b).abs() < 1e-6, "{m}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn batch_cycles_and_cuts() {
        let x = [1.0, 2.0, 3.0];
        assert_eq!(repeat_to_batch(&x, [3, 1, 1, 1], 5), [1.0, 2.0, 3.0, 1.0, 2.0]);
        assert_eq!(repeat_to_batch(&x, [3, 1, 1, 1], 2), [1.0, 2.0]);
    }
}
