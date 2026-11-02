//! `SwarmKSampler`'s initial noise (`swarm_fixed_noise`), which differs from plain ComfyUI: every
//! batch image gets its own freshly seeded generator, and Variation Seed slerps per image.

use diffusion::sampling::torch_randn;
use swarm_engine_api::VariationSeed;

/// Noise of batch image `index`, `[channels, h, w]`.
pub fn swarm_noise(seed: u64, variation: Option<VariationSeed>, index: u64, channels: usize, h: usize, w: usize) -> Vec<f32> {
    let n = channels * h * w;
    match variation {
        None => torch_randn(seed.wrapping_add(index), n),
        Some(v) => slerp(v.strength as f32, &torch_randn(seed, n), &torch_randn(v.seed.wrapping_add(index), n), channels, h, w),
    }
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
    fn slerp_endpoints_are_the_inputs() {
        let (a, b) = (torch_randn(1, 16 * 8 * 8), torch_randn(2, 16 * 8 * 8));
        let close = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-3);
        assert!(close(&slerp(0.0, &a, &b, 16, 8, 8), &a));
        assert!(close(&slerp(1.0, &a, &b, 16, 8, 8), &b));
    }
}
