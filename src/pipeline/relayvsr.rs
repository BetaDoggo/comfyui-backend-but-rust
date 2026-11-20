//! RelayVSR: the keyframe model and its Flash decoder.
//!
//! Like SeedVR2 they live beside whichever family is loaded: SwarmUI upscales another model's video in the same job.

use super::{run_sampler, upload, Generator, Sampling, Stats, Step, PLAIN};
use crate::cuda::{self, DevBuf};
use crate::{relayvsr, st};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Low-quality frames, f32 [count][height][width][channels] in 0..1.
pub struct Frames<'a> {
    pub data: &'a [f32],
    pub count: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

/// One clip's keyframes to sample.
pub struct RelayVsrRequest<'a> {
    /// The keyframes' low-quality frames (`RelayVSRConditioning`'s `images[keyframes]`).
    pub keyframes: Frames<'a>,
    /// The latent's size: [48, keyframes, lh, lw].
    pub lh: usize,
    pub lw: usize,
    pub sampling: Sampling,
    pub noise: Vec<f32>,
}

pub(super) struct Model {
    path: PathBuf,
    dit: relayvsr::Dit,
    st: st::SafeTensors,
}

impl Generator {
    /// Load the RelayVSR keyframe model unless `path` is loaded.
    pub fn ensure_relayvsr(&mut self, path: &Path) -> Result<()> {
        if self.relayvsr.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        let env = &self.env;
        env.st.sync()?;
        self.relayvsr = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        // under the legacy planner the blocks start in RAM: another family's model usually holds VRAM
        let budget = if env.comfy { usize::MAX } else { 0 };
        let mut dit = relayvsr::Dit::load(&dst, budget, env.slots, env.threads, &env.st)?;
        dit.int8_attn = env.int8_attn;
        eprintln!("RelayVSR x{} ready in {:.1}s", dit.factor, t0.elapsed().as_secs_f64());
        self.relayvsr = Some(Model { path: path.to_path_buf(), dit, st: dst });
        Ok(())
    }

    /// Load the RelayVSR Flash decoder unless `path` is loaded.
    pub fn ensure_relayvsr_vae(&mut self, path: &Path) -> Result<()> {
        if self.relayvsr_vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.relayvsr_vae = None;
        let t0 = Instant::now();
        let st = st::SafeTensors::open(path)?;
        let flash = relayvsr::Flash::load(&st, &self.env.st)?;
        eprintln!("RelayVSR Flash x{} decoder ready in {:.1}s", flash.factor, t0.elapsed().as_secs_f64());
        self.relayvsr_vae = Some((path.to_path_buf(), flash));
        Ok(())
    }

    /// The keyframe latent, sampled as ComfyUI's KSampler does (the template runs one euler step at CFG 1). `blocks`
    /// hears each block that finishes, of all the evaluations' blocks, and can stop the run.
    pub fn sample_relayvsr(&mut self, r: &RelayVsrRequest, mut progress: impl FnMut(Step) -> bool,
        mut blocks: impl FnMut(usize, usize) -> Result<()>) -> Result<(Vec<f32>, Stats)> {
        let k = &r.keyframes;
        let n = relayvsr::CH * k.count * r.lh * r.lw;
        if r.sampling.images != 1 {
            bail!("RelayVSR upscales one clip at a time");
        }
        if k.count == 0 || r.noise.len() != n || k.data.len() != k.count * k.height * k.width * k.channels {
            bail!("RelayVSR: {} noise values and {} frame values for {} keyframes", r.noise.len(), k.data.len(), k.count);
        }
        let need = self.relayvsr.as_ref().context("no RelayVSR model loaded")?.dit.activation_bytes(k.count, r.lh, r.lw);
        self.make_room(need)?;
        let env = &self.env;
        let m = self.relayvsr.as_mut().context("no RelayVSR model loaded")?;
        env.plan(&mut m.dit.blocks, &m.st, need)?;
        let st = &env.st;
        let lq = upload(k.data, st)?;
        let mut stats = Stats::default();
        let copied0 = m.dit.blocks.copied_bytes;
        let evals = r.sampling.sigmas.len().saturating_sub(1) * if r.sampling.uses_negative() { 2 } else { 1 };
        let latent = {
            let dit = &mut m.dit;
            let dims = (k.height, k.width, k.channels);
            let (count, lh, lw) = (k.count, r.lh, r.lw);
            let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, _side: super::Side, eval: usize| -> Result<()> {
                dit.predict(x, &lq, dims, count, lh, lw, sigma, out, st, |done, total| blocks(eval * total + done, evals.max(1) * total))
            };
            run_sampler(&r.sampling, &r.noise, [PLAIN, PLAIN], st, &mut stats, &mut model, &mut progress)
        };
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        if !env.comfy {
            env.st.sync()?;
            m.dit.blocks.set_resident(0, 1, &m.st)?;
            cuda::trim_pool()?;
        }
        Ok((latent?, stats))
    }

    /// `VSRDecode`: the keyframe latent (f32 [48][keyframes][lh][lw]) and the whole clip's low-quality frames to the
    /// upscaled clip, f32 [count][height * factor][width * factor][3]. `progress` hears each finished frame and can
    /// stop the decode.
    pub fn decode_relayvsr(&mut self, latent: &[f32], keyframes: usize, lh: usize, lw: usize, frames: &Frames, detail: f32,
        progress: impl FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        if !self.env.comfy {
            if let Some(m) = &mut self.relayvsr {
                m.dit.blocks.set_resident(0, 1, &m.st)?;
            }
        }
        let need = self.relayvsr_vae.as_ref().context("no RelayVSR Flash decoder loaded")?.1.activation_bytes(lh, lw);
        self.make_room(need)?;
        let st = &self.env.st;
        let flash = &self.relayvsr_vae.as_ref().context("no RelayVSR Flash decoder loaded")?.1;
        let t0 = Instant::now();
        let out = flash.decode(latent, keyframes, lh, lw, frames.data, (frames.count, frames.height, frames.width, frames.channels), detail, st,
            progress)?;
        eprintln!("RelayVSR Flash decode of {} frames {:.0} ms", frames.count, t0.elapsed().as_secs_f64() * 1e3);
        Ok(out)
    }

    /// The loaded keyframe model's upscale factor.
    pub fn relayvsr_factor(&self) -> Option<usize> {
        self.relayvsr.as_ref().map(|m| m.dit.factor)
    }

    /// The loaded Flash decoder's upscale factor.
    pub fn relayvsr_vae_factor(&self) -> Option<usize> {
        self.relayvsr_vae.as_ref().map(|(_, f)| f.factor)
    }

    pub(super) fn relayvsr_int8_attention(&mut self, on: bool) {
        if let Some(m) = &mut self.relayvsr {
            m.dit.int8_attn = on;
        }
    }

    /// RelayVSR's blocks out of VRAM (the legacy planner's idle release).
    pub(super) fn relayvsr_idle(&mut self) -> Result<()> {
        if let Some(m) = &mut self.relayvsr {
            m.dit.blocks.set_resident(0, 1, &m.st)?;
        }
        Ok(())
    }
}
