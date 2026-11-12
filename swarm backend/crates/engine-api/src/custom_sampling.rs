//! ComfyUI's custom-sampler graph (`SamplerCustomAdvanced` and its inputs). SwarmUI builds it for
//! IP2P CFG 2 (SDXL edit models, OmniGen); custom workflows can use it with any model.
//!
//! The server owns the pieces that are plain data: `SIGMAS` lists (Karras schedules and
//! `SplitSigmas` are computed server-side), the noise seed and the sampler name. The engine
//! computes model-dependent schedules ([`crate::Engine::model_sigmas`]) and runs the sampler
//! ([`crate::Engine::sample_custom`]).

use crate::conditioning::Conditioning;
use crate::handles::ModelHandle;
use crate::latent::Latent;

/// `SIGMAS`: a noise schedule, highest first, normally ending in 0.
#[derive(Clone, Debug, PartialEq)]
pub struct Sigmas(pub Vec<f64>);

/// `NOISE`: `RandomNoise` (the initial noise from `seed`, like `KSampler`'s `noise_seed`).
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseSource {
    pub seed: u64,
}

/// `SAMPLER`: `KSamplerSelect` (a ComfyUI sampler name, e.g. `euler`, `dpmpp_2m`).
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerChoice {
    pub name: String,
}

/// `GUIDER`: how each step's prediction is built from the model and conditionings.
#[derive(Clone, Debug)]
pub enum Guider {
    /// `DualCFGGuider` (SwarmUI's IP2P CFG 2). `regular`:
    /// `neg + cfg_cond2_negative * (cond2 - neg) + cfg_conds * (cond1 - cond2)`; `nested`:
    /// `neg + cfg_cond2_negative * (cfg(cond1 over cond2, cfg_conds) - neg)`.
    DualCfg { model: ModelHandle, cond1: Conditioning, cond2: Conditioning, negative: Conditioning, cfg_conds: f64, cfg_cond2_negative: f64, nested: bool },
}

/// Model-dependent sigma schedules.
#[derive(Clone, Debug)]
pub enum ScheduleRequest {
    /// `BasicScheduler`: `steps + 1` sigmas of `scheduler` (`simple`, `karras`, `normal`, ...) over
    /// the model's sigma range. The server handles `denoise` by asking for more steps and keeping
    /// the tail, as ComfyUI does.
    Basic { model: ModelHandle, scheduler: String, steps: u32 },
    /// `SDTurboScheduler`: the model's sigmas at timesteps `999, 899, ..., 99`, starting at index
    /// `10 - floor(10 * denoise)`, taking `steps` of them, then 0.
    SdTurbo { model: ModelHandle, steps: u32, denoise: f64 },
}

/// `SamplerCustomAdvanced`.
#[derive(Clone, Debug)]
pub struct CustomSampleRequest {
    pub noise: NoiseSource,
    pub guider: Guider,
    pub sampler: SamplerChoice,
    pub sigmas: Sigmas,
    /// As with `SampleRequest`, may be an empty placeholder (see `Latent::conform_empty`).
    pub latent: Latent,
}

/// `SamplerCustomAdvanced`'s two outputs.
#[derive(Clone, Debug)]
pub struct CustomSampleOutput {
    /// The sampled latent (with leftover noise if the sigmas don't end at 0).
    pub output: Latent,
    /// The model's denoised prediction at the last step.
    pub denoised: Latent,
}

impl Sigmas {
    /// `KarrasScheduler` (k-diffusion `get_sigmas_karras`): `steps` sigmas, then 0.
    pub fn karras(steps: u32, sigma_min: f64, sigma_max: f64, rho: f64) -> Self {
        let (min_inv, max_inv) = (sigma_min.powf(1.0 / rho), sigma_max.powf(1.0 / rho));
        let n = steps as usize;
        let mut s: Vec<f64> = (0..n)
            .map(|i| {
                let ramp = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
                (max_inv + ramp * (min_inv - max_inv)).powf(rho)
            })
            .collect();
        s.push(0.0);
        Sigmas(s)
    }

    /// `SplitSigmas`: `(sigmas[..=step], sigmas[step..])`.
    pub fn split(&self, step: usize) -> (Sigmas, Sigmas) {
        let n = self.0.len();
        (Sigmas(self.0[..(step + 1).min(n)].to_vec()), Sigmas(self.0[step.min(n)..].to_vec()))
    }

    /// Steps this schedule runs (one fewer than its length).
    pub fn steps(&self) -> u32 {
        self.0.len().saturating_sub(1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn karras_matches_k_diffusion() {
        // comfy.k_diffusion.sampling.get_sigmas_karras(4, 0.0291675, 14.614642, 7.0)
        let s = Sigmas::karras(4, 0.0291675, 14.614642, 7.0);
        let want = [14.614642, 3.168_609_5, 0.446_920_6, 0.029_167_5, 0.0];
        assert_eq!(s.0.len(), 5);
        for (a, b) in s.0.iter().zip(want) {
            assert!((a - b).abs() < 1e-4, "{:?}", s.0);
        }
    }

    #[test]
    fn split_shares_the_boundary() {
        let (hi, lo) = Sigmas(vec![3.0, 2.0, 1.0, 0.0]).split(1);
        assert_eq!((hi.0, lo.0), (vec![3.0, 2.0], vec![2.0, 1.0, 0.0]));
    }
}
