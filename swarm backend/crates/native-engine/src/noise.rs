//! `SwarmKSampler`'s initial noise (`swarm_fixed_noise`), which differs from plain ComfyUI: every
//! batch image gets its own freshly seeded generator, and Variation Seed slerps per image.

use diffusion::sampling::{torch_randn, torch_randn_streams};
use swarm_engine_api::VariationSeed;

/// Noise of batch image `index`, `[channels, h, w]`.
pub fn swarm_noise(seed: u64, variation: Option<VariationSeed>, index: u64, channels: usize, h: usize, w: usize) -> Vec<f32> {
    let n = channels * h * w;
    match variation {
        None => torch_randn(seed.wrapping_add(index), n),
        Some(v) => slerp(v.strength as f32, &torch_randn(seed, n), &torch_randn(v.seed.wrapping_add(index), n), channels, h, w),
    }
}

/// MiniMax H3's noise, video then audio: both streams come from one generator, with Variation Seed
/// slerped frame by frame on the video and flat on the audio. `video` is `[channels, t, h, w]`.
pub fn h3_noise(seed: u64, variation: Option<VariationSeed>, (channels, t, h, w): (usize, usize, usize, usize), audio: usize) -> Vec<f32> {
    let video = channels * t * h * w;
    let sizes = [video, audio];
    let [low_video, low_audio] = <[Vec<f32>; 2]>::try_from(torch_randn_streams(seed, &sizes)).unwrap();
    let Some(v) = variation else { return [low_video, low_audio].concat() };
    let [high_video, high_audio] = <[Vec<f32>; 2]>::try_from(torch_randn_streams(v.seed, &sizes)).unwrap();
    let val = v.strength as f32;
    let mut out = vec![0f32; video + audio];
    for f in 0..t {
        // [C, T, H, W] to one frame's [C, H, W] and back
        let frame = |x: &[f32]| -> Vec<f32> { (0..channels).flat_map(|c| x[(c * t + f) * h * w..][..h * w].to_vec()).collect() };
        let mixed = slerp(val, &frame(&low_video), &frame(&high_video), channels, h, w);
        for c in 0..channels {
            out[(c * t + f) * h * w..][..h * w].copy_from_slice(&mixed[c * h * w..][..h * w]);
        }
    }
    out[video..].copy_from_slice(&slerp_flat(val, &low_audio, &high_audio));
    out
}

/// `SwarmKSampler.slerp_flat`: spherical interpolation of the whole tensor.
fn slerp_flat(val: f32, low: &[f32], high: &[f32]) -> Vec<f32> {
    let lerp = || low.iter().zip(high).map(|(l, r)| l + (r - l) * val).collect();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let (ln, hn) = (norm(low), norm(high));
    if ln == 0.0 || hn == 0.0 {
        return lerp();
    }
    let dot = (low.iter().zip(high).map(|(l, r)| *l as f64 * *r as f64).sum::<f64>() / (ln * hn)).clamp(-1.0, 1.0);
    if dot.abs() > 0.9995 {
        return lerp();
    }
    let omega = dot.acos();
    let so = omega.sin();
    let (a, b) = ((((1.0 - val as f64) * omega).sin() / so) as f32, ((val as f64 * omega).sin() / so) as f32);
    low.iter().zip(high).map(|(l, r)| a * l + b * r).collect()
}

/// `SwarmKSampler.slerp` on `[C, H, W]`: spherical interpolation of each column (`dim=1`).
fn slerp(val: f32, low: &[f32], high: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let at = |c: usize, y: usize, x: usize| (c * h + y) * w + x;
    let columns = channels * w;
    let mut dots = vec![0f32; columns];
    let mut low_norms = vec![0f32; columns];
    let mut high_norms = vec![0f32; columns];
    for c in 0..channels {
        for x in 0..w {
            let j = c * w + x;
            for y in 0..h {
                let (l, r) = (low[at(c, y, x)], high[at(c, y, x)]);
                low_norms[j] += l * l;
                high_norms[j] += r * r;
                dots[j] += l * r;
            }
            low_norms[j] = low_norms[j].sqrt();
            high_norms[j] = high_norms[j].sqrt();
            dots[j] /= low_norms[j] * high_norms[j];
        }
    }
    if dots.iter().sum::<f32>() / columns as f32 > 0.9995 {
        return low.iter().zip(high).map(|(l, r)| l * val + r * (1.0 - val)).collect();
    }
    let mut out = vec![0f32; low.len()];
    for c in 0..channels {
        for x in 0..w {
            let omega = dots[c * w + x].acos();
            let so = omega.sin();
            let (a, b) = (((1.0 - val) * omega).sin() / so, (val * omega).sin() / so);
            for y in 0..h {
                out[at(c, y, x)] = a * low[at(c, y, x)] + b * high[at(c, y, x)];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_images_use_consecutive_seeds() {
        assert_eq!(swarm_noise(10, None, 3, 16, 8, 8), torch_randn(13, 16 * 8 * 8));
    }

    #[test]
    fn h3_variation_matches_swarm() {
        // swarm_fixed_noise(5, nested zeros of [1, 24, 3, 4, 4] and [1, 32, 2, 8], 9, 0.25)
        let (video, audio) = (24 * 3 * 4 * 4, 32 * 2 * 8);
        let n = h3_noise(5, Some(VariationSeed { seed: 9, strength: 0.25 }), (24, 3, 4, 4), audio);
        let close = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| (p - q).abs() < 2e-5);
        assert!(close(&n[..4], &[1.205367, 0.107573, -1.296622, -1.708661]), "{:?}", &n[..4]);
        assert!(close(&n[video..video + 4], &[0.404534, -0.850518, -0.218806, 0.924675]), "{:?}", &n[video..video + 4]);
        let sum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
        assert!((sum(&n[..video]) - 913.362).abs() < 1e-2 && (sum(&n[video..]) - 389.127).abs() < 1e-2);
    }

    #[test]
    fn h3_variation_keeps_the_streams_of_each_seed() {
        let shape = (24, 3, 4, 4);
        let (video, audio) = (24 * 3 * 4 * 4, 32 * 2 * 8);
        let close = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-3);
        let plain = h3_noise(5, None, shape, audio);
        assert_eq!(plain, torch_randn_streams(5, &[video, audio]).concat());
        assert!(close(&h3_noise(5, Some(VariationSeed { seed: 9, strength: 0.0 }), shape, audio), &plain));
        let other = h3_noise(9, None, shape, audio);
        assert!(close(&h3_noise(5, Some(VariationSeed { seed: 9, strength: 1.0 }), shape, audio), &other));
    }

    #[test]
    fn slerp_endpoints_are_the_inputs() {
        let (a, b) = (torch_randn(1, 16 * 8 * 8), torch_randn(2, 16 * 8 * 8));
        let close = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-3);
        assert!(close(&slerp(0.0, &a, &b, 16, 8, 8), &a));
        assert!(close(&slerp(1.0, &a, &b, 16, 8, 8), &b));
    }
}
