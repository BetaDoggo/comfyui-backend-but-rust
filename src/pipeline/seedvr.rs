//! SeedVR2: the one-step restoration DiT and its VAE.
//!
//! They live beside whichever family is loaded, not as one: SwarmUI runs SeedVR2 on another model's picture in the same
//! job, and entering a family drops the previous one. Under ComfyUI's memory management (the default) the other model's
//! weights only move to RAM to make room. Under `--legacy-offload` SeedVR2's blocks leave VRAM again after each run.

use super::{run_sampler, upload, Generator, Sampling, Stats, Step};
use crate::cuda::{self, DevBuf};
use crate::{seedvr, seedvr_vae, st};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One picture or clip to restore.
pub struct SeedVrRequest {
    /// Pixels, multiples of 16.
    pub width: usize,
    pub height: usize,
    /// Latent frames: 1 for a picture, n + 1 for a clip of 4n + 1 frames.
    pub frames: usize,
    /// One image.
    pub sampling: Sampling,
    /// f32 [16, frames, height / 8, width / 8], as the condition.
    pub noise: Vec<f32>,
    /// The latent from [`Generator::encode_seedvr`], f32 [16, frames, height / 8, width / 8]: the model's `condition`
    /// (`SeedVR2Conditioning`).
    pub condition: Vec<f32>,
}

pub(super) struct Model {
    path: PathBuf,
    dit: seedvr::Dit,
    st: st::SafeTensors,
}

/// SeedVR2's size check. `--max-side` doesn't bound it: its VAE tiles what doesn't fit, as ComfyUI's does, and its DiT
/// attends in windows, so its memory grows only linearly with the picture.
fn check_sides(width: usize, height: usize) -> Result<()> {
    if width % 16 != 0 || height % 16 != 0 || width == 0 || height == 0 {
        bail!("SeedVR2 needs both sides positive multiples of 16, got {width}x{height}");
    }
    Ok(())
}

/// Activation memory of one evaluation.
fn step_need(frames: usize, width: usize, height: usize, lt: usize) -> usize {
    let (lh, lw) = (height / 8, width / 8);
    seedvr::step_activation_bytes(frames * (lh / 2) * (lw / 2), lt, seedvr::window_count(frames, lh, lw))
}

impl Generator {
    /// Load the SeedVR2 DiT unless `path` is loaded.
    pub fn ensure_seedvr(&mut self, path: &Path) -> Result<()> {
        if self.seedvr.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        let env = &self.env;
        env.st.sync()?;
        self.seedvr = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = seedvr::globals_layout(&dst)?.size;
        // under the legacy planner the blocks start in RAM: another family's model usually holds VRAM
        let budget = if env.comfy { usize::MAX } else { 0 };
        let mut dit = seedvr::Dit::load(&dst, budget.saturating_sub(globals), env.slots, env.threads, &env.st)?;
        dit.int8_attn = env.int8_attn;
        eprintln!("SeedVR2 ready in {:.1}s", t0.elapsed().as_secs_f64());
        self.seedvr = Some(Model { path: path.to_path_buf(), dit, st: dst });
        Ok(())
    }

    /// Load SeedVR2's VAE unless `path` is loaded.
    pub fn ensure_seedvr_vae(&mut self, path: &Path) -> Result<()> {
        if self.seedvr_vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.seedvr_vae = None;
        let t0 = Instant::now();
        self.seedvr_vae = Some((path.to_path_buf(), seedvr_vae::Vae::load(path, &self.env.st)?));
        eprintln!("SeedVR2 VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// The restored latent, sampled as ComfyUI's KSampler does (SwarmUI sends one euler step at CFG 1). A clip is one
    /// sample over all its frames, as ComfyUI's.
    pub fn sample_seedvr(&mut self, r: &SeedVrRequest, mut progress: impl FnMut(Step) -> bool) -> Result<(Vec<f32>, Stats)> {
        check_sides(r.width, r.height)?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let n = seedvr::CH * r.frames * lh * lw;
        if r.sampling.images != 1 {
            bail!("SeedVR2 restores one picture or clip at a time");
        }
        if r.frames == 0 || r.noise.len() != n || r.condition.len() != n {
            bail!("noise and condition have {} and {} values, expected {n}", r.noise.len(), r.condition.len());
        }
        let need = step_need(r.frames, r.width, r.height, 64);
        // the loaded family's blocks give way first (legacy planner; ComfyUI's memory management does it as it goes)
        self.make_room(need)?;
        let env = &self.env;
        let m = self.seedvr.as_mut().context("no SeedVR2 model loaded")?;
        env.plan(&mut m.dit.blocks, &m.st, need)?;
        let st = &env.st;
        let cond = upload(&r.condition, st)?;
        let mut stats = Stats::default();
        let copied0 = m.dit.blocks.copied_bytes;
        let latent = {
            let dit = &mut m.dit;
            let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, _eval: usize| -> Result<()> {
                dit.predict(x, &cond, r.frames, lh, lw, sigma, negative, out, st, None)
            };
            run_sampler(&r.sampling, &r.noise, st, &mut stats, &mut model, &mut progress)
        };
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        if !env.comfy {
            env.st.sync()?;
            m.dit.blocks.set_resident(0, 1, &m.st)?;
            cuda::trim_pool()?;
        }
        Ok((latent?, stats))
    }

    /// RGB f32 [frames, height, width, 3] in 0..1 (sides multiples of 16) -> SeedVR2's latent f32 [16, latent frames,
    /// height / 8, width / 8] and its latent frames. One frame goes through the image path, more as one clip.
    pub fn encode_seedvr(&mut self, rgb: &[f32], frames: usize, width: usize, height: usize) -> Result<(Vec<f32>, usize)> {
        check_sides(width, height)?;
        if frames > 1 {
            let tile = self.clip_tile(width, height, false)?;
            let st = &self.env.st;
            let vae = &mut self.seedvr_vae.as_mut().context("no SeedVR2 VAE loaded")?.1;
            let t0 = Instant::now();
            // in pixels, overlapping by a quarter as the decode does
            let (z, lt) = vae.encode_clip(rgb, frames, height, width, tile.map(|t| (t * 8, t * 2)), st)?;
            eprintln!("SeedVR2 VAE encode of {frames} frames {:.0} ms{}", t0.elapsed().as_secs_f64() * 1e3,
                tile.map_or(String::new(), |t| format!(" in tiles of {}", t * 8)));
            return Ok((z, lt));
        }
        self.make_room(seedvr_vae::Vae::activation_bytes(height, width))?;
        let tile = self.vae_tile(width, height)?;
        let st = &self.env.st;
        let vae = &self.seedvr_vae.as_ref().context("no SeedVR2 VAE loaded")?.1;
        let t0 = Instant::now();
        let z = match tile {
            // in pixels, overlapping by a quarter as the decode does
            Some(t) => vae.encode_tiled(rgb, height, width, t * 8, t * 2, st)?,
            None => vae.encode(rgb, height, width, st)?,
        };
        eprintln!("SeedVR2 VAE encode {:.0} ms{}", t0.elapsed().as_secs_f64() * 1e3, tile.map_or(String::new(), |t| format!(" in tiles of {}", t * 8)));
        Ok((z, 1))
    }

    /// The most latent frames of a `width` x `height` clip that one SeedVR2 evaluation fits in the VRAM it may take (at
    /// least 1), for `SeedVR2TemporalChunk`'s `auto` mode. The blocks stream, so only their slots and the activations
    /// need room.
    pub fn seedvr_frames_that_fit(&self, width: usize, height: usize, frames: usize) -> Result<usize> {
        check_sides(width, height)?;
        // the DiT's globals, two streamed blocks, the modulations and the window geometry
        const BESIDE: usize = 1 << 30;
        let room = self.vae_room()?.saturating_sub(BESIDE);
        Ok((1..=frames).rev().find(|&f| step_need(f, width, height, 64) <= room).unwrap_or(1))
    }

    /// [`Generator::vae_tile`] for a clip: the causal convs keep two frames each, so it tiles sooner, with the largest
    /// tile whose clip fits.
    fn clip_tile(&mut self, width: usize, height: usize, decode: bool) -> Result<Option<usize>> {
        let vae = &self.seedvr_vae.as_ref().context("no SeedVR2 VAE loaded")?.1;
        let need = vae.clip_bytes(height, width, decode) + vae.temporal_bytes();
        self.make_room(need)?;
        if let Ok(v) = std::env::var("SEEDVR_VAE_TILE") {
            let n: usize = v.trim().parse().context("SEEDVR_VAE_TILE")?;
            return Ok((n > 0).then_some(n));
        }
        let room = self.vae_room()?;
        let vae = &self.seedvr_vae.as_ref().context("no SeedVR2 VAE loaded")?.1;
        let room = room.saturating_sub(vae.temporal_bytes());
        if vae.clip_bytes(height, width, decode) <= room {
            return Ok(None);
        }
        let side = (height.max(width) / 8).next_multiple_of(8);
        Ok(Some((2..side / 8).rev().map(|t| t * 8).find(|t| vae.clip_bytes(t * 8, t * 8, decode) <= room).unwrap_or(16)))
    }

    /// The VRAM the VAE may take: what is free, plus under ComfyUI's memory management the weights it would move to RAM.
    fn vae_room(&self) -> Result<usize> {
        self.env.st.sync()?;
        cuda::trim_pool()?;
        let mut free = cuda::device_free()?;
        if self.env.comfy {
            free += crate::comfy_memory::resident_bytes();
        }
        Ok(free.saturating_sub(self.env.reserve))
    }

    /// `None` when a `width` x `height` picture goes through the VAE whole, else the square tile (latent pixels) to take
    /// it in. As ComfyUI: tiles only when the whole doesn't fit, sized by `preferred_decode_tile`. `SEEDVR_VAE_TILE`
    /// forces a tile size, 0 none.
    fn vae_tile(&self, width: usize, height: usize) -> Result<Option<usize>> {
        if let Ok(v) = std::env::var("SEEDVR_VAE_TILE") {
            let n: usize = v.trim().parse().context("SEEDVR_VAE_TILE")?;
            return Ok((n > 0).then_some(n));
        }
        let room = self.vae_room()?;
        if seedvr_vae::Vae::activation_bytes(height, width) <= room {
            return Ok(None);
        }
        Ok(Some(seedvr_vae::preferred_tile(room)))
    }

    /// SeedVR2's latent f32 [16, frames, height / 8, width / 8] -> RGB f32 [out frames, height, width, 3] in 0..1 and its
    /// frames: one for a picture, 4 (frames - 1) + 1 for a clip.
    pub fn decode_seedvr(&mut self, latent: &[f32], frames: usize, width: usize, height: usize) -> Result<(Vec<f32>, usize)> {
        let (lh, lw) = (height / 8, width / 8);
        if latent.len() != seedvr::CH * frames * lh * lw || frames == 0 {
            bail!("latent has {} values, expected {}", latent.len(), seedvr::CH * frames * lh * lw);
        }
        if !self.env.comfy {
            if let Some(m) = &mut self.seedvr {
                m.dit.blocks.set_resident(0, 1, &m.st)?;
            }
        }
        if frames > 1 {
            let tile = self.clip_tile(width, height, true)?;
            let st = &self.env.st;
            let vae = &mut self.seedvr_vae.as_mut().context("no SeedVR2 VAE loaded")?.1;
            let t0 = Instant::now();
            let (rgb, n) = vae.decode_clip(latent, frames, lh, lw, tile, st)?;
            eprintln!("SeedVR2 VAE decode of {n} frames {:.0} ms{}", t0.elapsed().as_secs_f64() * 1e3,
                tile.map_or(String::new(), |t| format!(" in tiles of {}", t * 8)));
            return Ok((rgb, n));
        }
        self.make_room(seedvr_vae::Vae::activation_bytes(height, width))?;
        let tile = self.vae_tile(width, height)?;
        let st = &self.env.st;
        let vae = &self.seedvr_vae.as_ref().context("no SeedVR2 VAE loaded")?.1;
        let t0 = Instant::now();
        let rgb = match tile {
            Some(t) => vae.decode_tiled(latent, lh, lw, t, st)?,
            None => vae.decode(&upload(latent, st)?, lh, lw, st)?,
        };
        eprintln!("SeedVR2 VAE decode {:.0} ms{}", t0.elapsed().as_secs_f64() * 1e3, tile.map_or(String::new(), |t| format!(" in tiles of {}", t * 8)));
        Ok((rgb, 1))
    }

    pub(super) fn seedvr_int8_attention(&mut self, on: bool) {
        if let Some(m) = &mut self.seedvr {
            m.dit.int8_attn = on;
        }
    }

    /// SeedVR2's blocks out of VRAM (the legacy planner's idle release).
    pub(super) fn seedvr_idle(&mut self) -> Result<()> {
        if let Some(m) = &mut self.seedvr {
            m.dit.blocks.set_resident(0, 1, &m.st)?;
        }
        Ok(())
    }
}
