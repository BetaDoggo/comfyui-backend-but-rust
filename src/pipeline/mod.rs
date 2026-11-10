//! Loaded models and one generation, used by the CLI and the server.
//!
//! One model family is loaded at a time. [`Family`] holds it: the diffusion model, the encoders and decoders only that
//! family uses, and the prompts it has encoded. Each family's file keeps its caches next to whatever invalidates them,
//! and [`FamilyState`] is what the generator does without knowing which family is loaded. Adding a family means a file
//! here, a [`Kind`] and a [`Family`] variant; the compiler points at every match that needs the new arm.
//!
//! Two parts outlive a family: the Qwen3-VL text encoder with its vision tower ([`SharedTe`]), which Krea 2 and
//! Qwen Image 2.1 share, and the 16-channel VAE, which Krea 2 and Anima share.

mod anima;
mod h3;
mod krea;
mod moss;
mod qwen21;

pub use anima::{AnimaRequest, SnapRefRequest};
pub use h3::{h3_presentation, h3_tokens, image_pictures, ref2va_pictures, ref2va_size, resize as h3_resize, size_short_edge, H3Conds, H3Picture, H3Request};
pub use krea::KreaRequest;
pub use moss::{moss_tokens, MossRequest};
pub use qwen21::Qwen21Request;
pub use crate::samplers::Inpaint;

use crate::cuda::{self, DevBuf, Event, Stream};
use crate::samplers::{self, CfgTweaks, Eval, ModelSampling, Uncond};
use crate::vision::{self, PromptImage};
use crate::{ffi, lora, offload, st, te, vae};
use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const MB: usize = 1 << 20;

pub struct LoadOptions {
    pub dit: PathBuf,
    pub te: PathBuf,
    pub vae: PathBuf,
    /// Size the initial residency plan is made for (it is re-planned per request).
    pub max_width: usize,
    pub max_height: usize,
    /// Largest request accepted (width * height).
    pub max_pixels: usize,
    pub reserve_vram_mb: usize,
    pub resident_blocks: Option<usize>,
    pub slots: usize,
    pub threads: usize,
    pub attn_variant: i32,
    pub int8_attn: bool,
    /// `Some(headroom MB)`: ComfyUI's DynamicVRAM memory management ([`crate::comfy_memory`]), the default, keeping
    /// that much VRAM free (ComfyUI's `--reserve-vram`, 256 MB by default). `slots` is then its cast buffer count, and
    /// `resident_blocks` is refused. `None` is the legacy planner (`--legacy-offload`).
    pub comfy_memory: Option<usize>,
}

/// What the families share in a request: a sampler, its sigmas and the guidance.
#[derive(Clone, Debug)]
pub struct Sampling {
    /// A name from [`samplers::SAMPLERS`].
    pub sampler: String,
    pub model_sampling: ModelSampling,
    /// From [`samplers::sigmas`]; one step fewer than values.
    pub sigmas: Vec<f32>,
    pub cfg: f64,
    /// Seed of the sampler's own noise (ComfyUI's `noise_seed`).
    pub seed: u64,
    /// The first image's place in its batch: the sampler noise is one tensor for the whole batch.
    pub batch_index: usize,
    pub batch_size: usize,
    /// Images sampled from `batch_index`, together through the model. The noise, `latent` and the mask hold them one
    /// after another.
    pub images: usize,
    pub tweaks: CfgTweaks,
    /// The latent sampling starts from, at the sampler's scale (after `process_latent_in`); `None` is an empty latent.
    /// With `sigmas` starting below the model's highest this is img2img.
    pub latent: Option<Vec<f32>>,
    pub inpaint: Option<Inpaint>,
}

/// A prompt as ComfyUI's conditioning is built from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    /// `<break>` chunks, encoded apart and joined. Each is (text, weight) segments.
    pub chunks: Vec<Vec<(String, f32)>>,
    /// `ConditioningMultiply` and `ConditioningZeroOut`, on the text encoder's output.
    pub scale: f32,
}

impl Prompt {
    pub fn new(segments: Vec<(String, f32)>) -> Self {
        Prompt { chunks: vec![segments], scale: 1.0 }
    }

    pub fn text(text: &str) -> Self {
        Self::new(vec![(text.to_string(), 1.0)])
    }

    /// For the models whose prompts can't take `<break>`.
    fn segments(&self) -> Result<&[(String, f32)]> {
        match self.chunks.as_slice() {
            [segments] => Ok(segments),
            _ => bail!("this model takes one prompt chunk"),
        }
    }
}

/// One step of a run: `denoised` is the latest prediction of its images, one after another.
pub struct Step<'a> {
    pub index: usize,
    pub total: usize,
    /// The run's first image in the batch, and how many it samples together.
    pub first: usize,
    pub images: usize,
    pub denoised: &'a [f32],
}

/// Multiply the rows of an f32 [rows, width] buffer, each by its factor.
fn scale_rows(b: &DevBuf, width: usize, factors: &[f32], st: &Stream) -> Result<()> {
    if factors.iter().all(|f| *f == 1.0) {
        return Ok(());
    }
    let mut bytes = vec![0u8; factors.len() * width * 4];
    b.download(0, &mut bytes, st)?;
    for (row, f) in bytes.chunks_exact_mut(width * 4).zip(factors) {
        for v in row.chunks_exact_mut(4) {
            v.copy_from_slice(&(f32::from_le_bytes([v[0], v[1], v[2], v[3]]) * f).to_le_bytes());
        }
    }
    b.upload(0, &bytes, st)
}

impl Sampling {
    /// One image at `cfg` with the given sampler and scheduler.
    pub fn new(sampler: &str, scheduler: &str, model_sampling: ModelSampling, steps: usize, cfg: f64, seed: u64, latent: (usize, usize)) -> Result<Self> {
        let sigmas = samplers::sigmas(model_sampling, sampler, scheduler, steps, latent, None)?;
        Ok(Sampling {
            sampler: sampler.to_string(),
            model_sampling,
            sigmas,
            cfg,
            seed,
            batch_index: 0,
            batch_size: 1,
            images: 1,
            tweaks: CfgTweaks::default(),
            latent: None,
            inpaint: None,
        })
    }

    /// True when the negative prompt has to be encoded.
    pub fn uses_negative(&self) -> bool {
        (self.cfg - 1.0).abs() > 1e-6 || samplers::forces_uncond(&self.sampler)
    }
}

/// A reference latent, [channels, height, width] f32 at model scale: 16 channels for Krea 2, 64 for Qwen Image 2.1.
#[derive(Clone)]
pub struct Reference {
    pub latent: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub text_ms: f64,
    pub sample_ms: f64,
    pub vae_ms: f64,
    pub total_ms: f64,
    pub streamed_gb: f64,
}

/// Debug tensors: name -> (safetensors dtype, shape, bytes)
pub type Dump = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;

/// Returned (through `anyhow`) when the progress callback asks to stop.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// One run of a sampler. `model(x, out, sigma, negative, evaluation)` writes a denoised latent, and
/// is called twice per evaluation under CFG, always positive first.
fn run_sampler(s: &Sampling, noise: &[f32], st: &Stream, stats: &mut Stats,
    model: &mut dyn FnMut(&DevBuf, &DevBuf, f32, bool, usize) -> Result<()>, progress: &mut dyn FnMut(Step) -> bool) -> Result<Vec<f32>> {
    let threads = unsafe { ffi::gpu_randn_threads() };
    if threads <= 0 {
        bail!("could not read the GPU's core count");
    }
    let n = noise.len();
    if s.images == 0 || n % s.images != 0 {
        bail!("{n} noise values don't split into {} images", s.images);
    }
    let per = n / s.images;
    if s.latent.as_ref().is_some_and(|l| l.len() != n) || s.inpaint.as_ref().is_some_and(|i| i.mask.len() != n) {
        bail!("the latent or mask doesn't match the noise's {n} values");
    }
    let zeros;
    let latent = match &s.latent {
        Some(l) => l.as_slice(),
        None => {
            zeros = vec![0f32; n];
            &zeros
        }
    };
    let (x, den, und) = (DevBuf::alloc_async(n * 4, st)?, DevBuf::alloc_async(n * 4, st)?, DevBuf::alloc_async(n * 4, st)?);
    let (e0, e1) = (Event::timing()?, Event::timing()?);
    let cfg = s.cfg as f32;
    let guided = (s.cfg - 1.0).abs() > 1e-6;
    let mut evals = 0;
    let mut sample_ms = 0.0;
    let latest = RefCell::new(Vec::new());
    let mut eval = |v: &[f32], sigma: f32, mode: Uncond| -> Result<Eval> {
        e0.record(st)?;
        // KSamplerX0Inpaint: the model sees the kept region noised to this sigma
        let mask = s.inpaint.as_ref().map(|i| i.mask_at(s.model_sampling, &s.sigmas, sigma));
        let masked: Vec<f32>;
        let v = match &mask {
            Some(m) => {
                masked = Inpaint::input(m, v, sigma, noise, latent);
                &masked
            }
            None => v,
        };
        x.upload(0, &st::f32_bytes(v), st)?;
        model(&x, &den, sigma, false, evals)?;
        let negative = guided || mode == Uncond::Force;
        if negative {
            model(&x, &und, sigma, true, evals)?;
            if !s.tweaks.any() {
                let (p, c) = ([und.ptr() as ffi::P, den.ptr() as ffi::P], [1.0 - cfg, cfg]);
                ffi::k!(gpu_lincomb(den.ptr(), n as i64, 2, p.as_ptr(), c.as_ptr(), ffi::s(st)))?;
            }
        }
        let read = |b: &DevBuf| -> Result<Vec<f32>> {
            let mut bytes = vec![0u8; n * 4];
            b.download(0, &mut bytes, st)?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let (denoised, uncond) = if s.tweaks.any() {
            // the tweaks measure each image apart, as ComfyUI's do over the batch
            let raw = if negative { Some(read(&und)?) } else { None };
            let cond = read(&den)?;
            let (mut d, mut u) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for i in 0..s.images {
                let r = i * per..(i + 1) * per;
                let (di, ui) = s.tweaks.guide(&v[r.clone()], &cond[r.clone()], raw.as_ref().map(|u| u[r].to_vec()), cfg, sigma);
                d.extend(di);
                u.extend(ui);
            }
            (d, (negative && mode != Uncond::Skip).then_some(u))
        } else {
            (read(&den)?, if negative && mode != Uncond::Skip { Some(read(&und)?) } else { None })
        };
        // and its prediction there is the latent; the uncond a sampler reads stays the model's (post-CFG hooks)
        let denoised = match &mask {
            Some(m) => Inpaint::prediction(m, &denoised, latent),
            None => denoised,
        };
        latest.borrow_mut().clone_from(&denoised);
        e1.record(st)?;
        e1.sync()?;
        let ms = e1.elapsed_ms(&e0)? as f64;
        sample_ms += ms;
        evals += 1;
        eprintln!("evaluation {evals} sigma {sigma:.4}: {ms:.0} ms");
        Ok(Eval { denoised, uncond })
    };
    let (first, images) = (s.batch_index, s.images);
    let mut step = |index: usize, total: usize| {
        if progress(Step { index, total, first, images, denoised: &latest.borrow() }) { Ok(()) } else { Err(Cancelled.into()) }
    };
    let batch = samplers::Batch { index: s.batch_index, size: s.batch_size, images: s.images, threads: threads as usize };
    let out = samplers::sample(&s.sampler, s.model_sampling, &s.sigmas, noise, s.latent.as_deref(), s.seed, batch, &mut eval, &mut step)?;
    stats.sample_ms += sample_ms;
    eprintln!("sampling {:.2}s ({:.0} ms/evaluation)", sample_ms / 1e3, sample_ms / evals.max(1) as f64);
    if crate::comfy_memory::enabled() {
        // the sampler node is done
        crate::comfy_memory::log_residency();
        crate::comfy_memory::node_done()?;
    }
    Ok(out)
}

/// The images of `s` sampled in runs of at most `chunk`, each run's images together through the model, as
/// `run_sampler` does one run. `model` gets the run's image count last. `dpm_adaptive` picks its steps per run.
fn run_batches(s: &Sampling, noise: &[f32], chunk: usize, st: &Stream, stats: &mut Stats,
    model: &mut dyn FnMut(&DevBuf, &DevBuf, f32, bool, usize, usize) -> Result<()>, progress: &mut dyn FnMut(Step) -> bool) -> Result<Vec<f32>> {
    if s.images == 0 || noise.len() % s.images != 0 {
        bail!("{} noise values don't split into {} images", noise.len(), s.images);
    }
    let per = noise.len() / s.images;
    let chunk = chunk.clamp(1, s.images);
    let mut out = Vec::with_capacity(noise.len());
    for start in (0..s.images).step_by(chunk) {
        let k = chunk.min(s.images - start);
        let r = start * per..(start + k) * per;
        let run = Sampling {
            batch_index: s.batch_index + start,
            images: k,
            latent: s.latent.as_ref().map(|l| l[r.clone()].to_vec()),
            inpaint: s.inpaint.as_ref().map(|i| Inpaint { mask: i.mask[r.clone()].to_vec(), differential: i.differential }),
            ..s.clone()
        };
        if k < s.images {
            eprintln!("sampling images {} to {} of {} together", start + 1, start + k, s.images);
        }
        out.extend(run_sampler(&run, &noise[r], st, stats, &mut |x, o, sigma, neg, eval| model(x, o, sigma, neg, eval, k), progress)?);
    }
    Ok(out)
}

/// Token ids and per-token factors, one pair per `<break>` chunk.
type TokenKey = Vec<(Vec<u32>, Vec<f32>)>;

/// Encoded prompts kept in VRAM per family.
const CONTEXTS: usize = 4;
/// The most prompt images SwarmUI sends.
const VISION_CACHE: usize = 10;

/// Move the entry matching `hit` to the end of an LRU list (most recent last); false when there is none.
fn touch<T>(cache: &mut Vec<T>, hit: impl Fn(&T) -> bool) -> bool {
    match cache.iter().position(hit) {
        Some(i) => {
            let e = cache.remove(i);
            cache.push(e);
            true
        }
        None => false,
    }
}

/// Add to an LRU list, dropping the oldest past `cap`.
fn remember<T>(cache: &mut Vec<T>, entry: T, cap: usize) {
    cache.push(entry);
    if cache.len() > cap {
        cache.remove(0);
    }
}

/// Crop interleaved f32 pixels to a multiple of `multiple` on each side, evenly from both edges, as ComfyUI's VAE does.
/// Returns the pixels and their width and height.
fn crop(pixels: &[f32], width: usize, height: usize, channels: usize, multiple: usize) -> Result<(Vec<f32>, usize, usize)> {
    if pixels.len() != width * height * channels {
        bail!("image has {} values, expected {}", pixels.len(), width * height * channels);
    }
    let (cw, ch) = (width / multiple * multiple, height / multiple * multiple);
    if cw == 0 || ch == 0 {
        bail!("{width}x{height} is too small to encode");
    }
    let (x0, y0) = ((width - cw) / 2, (height - ch) / 2);
    let mut out = Vec::with_capacity(cw * ch * channels);
    for y in y0..y0 + ch {
        out.extend_from_slice(&pixels[(y * width + x0) * channels..][..cw * channels]);
    }
    Ok((out, cw, ch))
}

/// Upload a host f32 latent.
fn upload(latent: &[f32], st: &Stream) -> Result<DevBuf> {
    let d = DevBuf::alloc_async(latent.len() * 4, st)?;
    d.upload(0, &st::f32_bytes(latent), st)?;
    Ok(d)
}

/// Fails unless `loras` is empty: there is nothing loaded to bake them into.
fn nothing_for(loras: &[lora::Lora], what: &str) -> Result<()> {
    if !loras.is_empty() {
        bail!("no {what} loaded");
    }
    Ok(())
}

/// The stream and the memory settings every family plans with.
struct Env {
    st: Stream,
    max_pixels: usize,
    max_width: usize,
    max_height: usize,
    reserve: usize,
    resident_blocks: Option<usize>,
    slots: usize,
    threads: usize,
    attn_variant: i32,
    int8_attn: bool,
    /// Re-plan DiT block residency from live free VRAM (--legacy-offload, unless --resident-blocks is forced).
    dynamic: bool,
    /// ComfyUI's memory management decides residency as it runs (the default; off with `--legacy-offload`).
    comfy: bool,
}

impl Env {
    fn check_size(&self, width: usize, height: usize) -> Result<()> {
        if width % 16 != 0 || height % 16 != 0 || width == 0 || height == 0 {
            bail!("width and height must be positive multiples of 16");
        }
        if width * height > self.max_pixels {
            bail!("{width}x{height} exceeds the {} pixel limit", self.max_pixels);
        }
        Ok(())
    }

    /// Re-plan which DiT blocks live in VRAM from the memory free right now (other applications
    /// included), keeping `need` bytes for activations plus the reserve. Blocks are promoted from or
    /// demoted to pinned RAM; nothing is reloaded from disk. The stream is synchronized first.
    fn rebalance(&self, blocks: &mut offload::WeightSet, dst: &st::SafeTensors, need: usize, demote_only: bool) -> Result<()> {
        let (st, reserve) = (&self.st, self.reserve);
        st.sync()?;
        cuda::trim_pool()?;
        let free = cuda::device_free()?;
        let owned = blocks.vram_bytes();
        let avail = (free + owned).saturating_sub(need + reserve);
        let mut p = offload::plan(&blocks.layouts, avail, self.slots);
        if std::env::var_os("DEBUG_VRAM").is_some() {
            let (cf, _) = cuda::mem_info()?;
            eprintln!("rebalance: free {} MB (cuda {} MB), owned {} MB, need {} MB, avail {} MB -> {} resident",
                free / MB, cf / MB, owned / MB, need / MB, avail / MB, p.resident);
        }
        let n = blocks.len();
        let block = blocks.layouts.iter().map(|l| l.size).max().unwrap_or(0);
        if p.resident < n && avail < block * p.slots {
            // Not even full streaming fits (other applications hold the memory). Keep going with the
            // smallest footprint like ComfyUI does; the driver pages to system memory and it gets slower.
            p = offload::Plan { resident: 0, slots: 1 };
            eprintln!("warning: VRAM oversubscribed (free {} MB, activations {} MB, reserve {} MB); streaming every block through one slot, expect slowdowns", free / MB, need / MB, reserve / MB);
        }
        if demote_only && p.resident >= blocks.resident() {
            return Ok(());
        }
        let want_slots = if p.resident < n { p.slots } else { 0 };
        if p.resident != blocks.resident() || want_slots != blocks.slot_count() {
            blocks.set_resident(p.resident, p.slots, dst)?;
        }
        Ok(())
    }

    /// How many images fit through the diffusion model together, at `per_image` bytes of activations each next to
    /// `fixed` bytes, with the blocks streamed through the fewest slots when residency is dynamic. At least 1 and at
    /// most `batch`.
    fn batch_fit(&self, blocks: &offload::WeightSet, batch: usize, fixed: usize, per_image: usize) -> Result<usize> {
        if batch <= 1 {
            return Ok(1);
        }
        if self.comfy {
            // ComfyUI never splits a latent batch: what doesn't fit pushes weights out, then runs out of memory
            return Ok(batch);
        }
        self.st.sync()?;
        cuda::trim_pool()?;
        let free = cuda::device_free()?;
        let room = if self.dynamic {
            let block = blocks.layouts.iter().map(|l| l.size).max().unwrap_or(0);
            (free + blocks.vram_bytes()).saturating_sub(self.reserve + fixed + block * self.slots)
        } else {
            free.saturating_sub(self.reserve + fixed)
        };
        let fit = (room / per_image.max(1)).clamp(1, batch);
        if fit < batch {
            eprintln!("{fit} of {batch} images fit through the model at once ({} MB each, {} MB free)", per_image / MB, room / MB);
        }
        Ok(fit)
    }

    /// Before a request: re-plan residency for its activations, when residency is dynamic.
    fn plan(&self, blocks: &mut offload::WeightSet, dst: &st::SafeTensors, need: usize) -> Result<()> {
        if self.comfy {
            // the sampler loads its model: ComfyUI's load_models_gpu, with no estimate
            return blocks.prioritize();
        }
        if self.dynamic {
            self.rebalance(blocks, dst, need, false)?;
        }
        Ok(())
    }

    /// Mid-run: another application took VRAM, so demote blocks before the driver starts paging to
    /// shared system memory (which would be far slower than streaming). `step` is one step's activations.
    fn relieve(&self, blocks: &mut offload::WeightSet, dst: &st::SafeTensors, step: usize) -> Result<()> {
        if self.dynamic && cuda::device_free()? < self.reserve / 2 {
            eprintln!("VRAM pressure: free {} MB < {} MB", cuda::device_free()? / MB, self.reserve / 2 / MB);
            // NVML clamps free at 0 when oversubscribed, hiding the real shortfall: ask for an
            // extra reserve each time so blocks keep moving out until pressure clears
            self.rebalance(blocks, dst, step + self.reserve, true)?;
        }
        Ok(())
    }

    /// Before an encoder or decoder: demote blocks if `need` bytes don't fit next to the reserve.
    fn make_room(&self, blocks: &mut offload::WeightSet, dst: &st::SafeTensors, need: usize) -> Result<()> {
        if self.dynamic && cuda::device_free()? < need + self.reserve {
            self.rebalance(blocks, dst, need, true)?;
        }
        Ok(())
    }
}

/// Krea 2's and Qwen Image 2.1's text encoder, with the vision tower from the same file.
struct SharedTe {
    path: PathBuf,
    enc: te::TextEncoder,
    /// Loaded by the first prompt image.
    vision: Option<vision::Vision>,
    /// Latest images, most recent last.
    vision_cache: Vec<(PromptImage, Arc<vision::Embeds>)>,
}

impl SharedTe {
    /// The vision tower's tokens for `images`, through the cache.
    fn vision_embeds(&mut self, images: &[PromptImage], env: &Env, mut dump: Option<&mut Dump>) -> Result<Vec<Arc<vision::Embeds>>> {
        let st = &env.st;
        let mut out = Vec::with_capacity(images.len());
        for img in images {
            if touch(&mut self.vision_cache, |(k, _)| k == img) {
                out.push(self.vision_cache.last().unwrap().1.clone());
                continue;
            }
            if self.vision.is_none() {
                let t0 = Instant::now();
                self.vision = Some(vision::Vision::load(st::SafeTensors::open(&self.path)?, env.threads, st)?);
                eprintln!("vision tower ready in {:.1}s", t0.elapsed().as_secs_f64());
            }
            let t0 = Instant::now();
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                let mut v = vec![0u8; bytes];
                b.download(0, &mut v, st).unwrap();
                probes.push((name.to_string(), v));
            };
            let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
            let e = Arc::new(self.vision.as_mut().unwrap().encode(img, st, probe_ref)?);
            eprintln!("vision: {}x{}, {} tokens in {:.0} ms", img.width, img.height, e.tokens, t0.elapsed().as_secs_f64() * 1e3);
            if let Some(d) = dump.as_deref_mut() {
                for (name, v) in probes {
                    d.entry(name).or_insert(("F32".into(), vec![v.len() / 4], v));
                }
            }
            out.push(e.clone());
            remember(&mut self.vision_cache, (img.clone(), e), VISION_CACHE);
        }
        Ok(out)
    }
}

/// Bake text encoder LoRAs into the shared text encoder; true when the weights changed.
fn shared_te_loras(te: Option<&mut SharedTe>, loras: &[lora::Lora], st: &Stream) -> Result<bool> {
    match te {
        Some(t) => t.enc.apply_loras(loras, st),
        None => nothing_for(loras, "text encoder").map(|_| false),
    }
}

/// A model family, as [`Generator::enter`] switches between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Krea,
    Anima,
    Moss,
    Qwen21,
    H3,
}

impl Kind {
    /// Krea 2 and Qwen Image 2.1 share [`SharedTe`].
    fn uses_shared_te(self) -> bool {
        matches!(self, Kind::Krea | Kind::Qwen21)
    }
}

/// What the generator does to whichever family is loaded. Every cache a family keeps is invalidated
/// here or in its own `ensure_*` methods, next to the cache.
trait FamilyState {
    /// Bake `loras` into the diffusion model, in place of any others, and drop what was computed through the old weights.
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()>;
    /// The same for the text encoder. Krea 2 and Qwen Image 2.1 bake into `shared`; the others into their own.
    fn set_text_encoder_loras(&mut self, shared: Option<&mut SharedTe>, loras: &[lora::Lora], st: &Stream) -> Result<()>;
    /// Drop every encoded prompt.
    fn clear_prompts(&mut self);
    fn set_int8_attention(&mut self, on: bool);
    /// The diffusion model's streamed blocks and their file, to make room in VRAM.
    fn blocks(&mut self) -> Option<(&mut offload::WeightSet, &st::SafeTensors)>;
    /// Give the diffusion model's VRAM back while nothing runs; the next request takes it again.
    fn idle(&mut self) -> Result<()>;
}

/// The loaded family. Its parts are each optional: a family's VAE or text encoder can load before its diffusion model.
enum Family {
    None,
    Krea(krea::Krea),
    Anima(anima::Anima),
    Moss(moss::Moss),
    Qwen21(qwen21::Qwen21),
    H3(h3::H3),
}

impl Family {
    fn new(kind: Kind) -> Self {
        match kind {
            Kind::Krea => Family::Krea(Default::default()),
            Kind::Anima => Family::Anima(Default::default()),
            Kind::Moss => Family::Moss(Default::default()),
            Kind::Qwen21 => Family::Qwen21(Default::default()),
            Kind::H3 => Family::H3(Default::default()),
        }
    }

    fn kind(&self) -> Option<Kind> {
        Some(match self {
            Family::None => return None,
            Family::Krea(_) => Kind::Krea,
            Family::Anima(_) => Kind::Anima,
            Family::Moss(_) => Kind::Moss,
            Family::Qwen21(_) => Kind::Qwen21,
            Family::H3(_) => Kind::H3,
        })
    }

    fn state(&mut self) -> Option<&mut dyn FamilyState> {
        Some(match self {
            Family::None => return None,
            Family::Krea(f) => f,
            Family::Anima(f) => f,
            Family::Moss(f) => f,
            Family::Qwen21(f) => f,
            Family::H3(f) => f,
        })
    }
}

pub struct Generator {
    env: Env,
    te: Option<SharedTe>,
    vae: Option<(PathBuf, vae::Vae)>,
    family: Family,
}

impl Generator {
    /// Start the GPU runtime with no models loaded; the `ensure_*` methods load them.
    pub fn new(o: &LoadOptions) -> Result<Self> {
        cuda::init(0)?;
        ffi::ok(unsafe { ffi::gpu_blas_init(64 * MB) }, "cublasLt init")?;
        let st = Stream::new()?;
        let (_, total) = cuda::mem_info()?;
        eprintln!("GPU memory: {} MB free (device-wide) / {} MB total", cuda::device_free()? / MB, total / MB);
        if let Some(headroom) = o.comfy_memory {
            if o.resident_blocks.is_some() {
                bail!("--resident-blocks needs --legacy-offload: the default ComfyUI memory management decides residency as it runs");
            }
            crate::comfy_memory::enable(headroom * MB, o.slots)?;
        }
        let env = Env {
            st,
            max_pixels: o.max_pixels,
            max_width: o.max_width,
            max_height: o.max_height,
            reserve: o.reserve_vram_mb * MB,
            resident_blocks: o.resident_blocks,
            slots: o.slots,
            threads: o.threads,
            attn_variant: o.attn_variant,
            int8_attn: o.int8_attn,
            dynamic: o.resident_blocks.is_none() && o.comfy_memory.is_none(),
            comfy: o.comfy_memory.is_some(),
        };
        Ok(Generator { env, te: None, vae: None, family: Family::None })
    }

    /// The stream the models run on.
    pub fn stream(&self) -> &Stream {
        &self.env.st
    }

    /// Make `kind` the loaded family. Any other is dropped first, and the shared text encoder with it unless
    /// `kind` uses it.
    fn enter(&mut self, kind: Kind) -> Result<()> {
        if self.family.kind() == Some(kind) {
            return Ok(());
        }
        self.env.st.sync()?;
        self.family = Family::None;
        if !kind.uses_shared_te() {
            self.te = None;
        }
        self.family = Family::new(kind);
        cuda::trim_pool()
    }

    /// Load the 16-channel VAE (Krea 2 and Anima) unless `path` is already loaded.
    pub fn ensure_vae(&mut self, path: &Path) -> Result<()> {
        if self.vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.vae = None;
        let t0 = Instant::now();
        self.vae = Some((path.to_path_buf(), vae::Vae::load(path, &self.env.st)?));
        eprintln!("VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the Krea 2 / Qwen Image 2.1 text encoder unless `path` is already loaded (streamed from pinned RAM).
    pub fn ensure_te(&mut self, path: &Path) -> Result<()> {
        if self.te.as_ref().is_some_and(|t| t.path == path) {
            return Ok(());
        }
        self.te = None;
        if self.family.kind().is_some_and(Kind::uses_shared_te) {
            self.family.state().unwrap().clear_prompts();
        }
        let t0 = Instant::now();
        let enc = te::TextEncoder::load(path, 0, self.env.threads, &self.env.st)?;
        self.te = Some(SharedTe { path: path.to_path_buf(), enc, vision: None, vision_cache: Vec::new() });
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Bake `loras` into the loaded diffusion model, in place of any others. Prompts encoded through
    /// the old weights are dropped: their contexts come out of the diffusion model.
    pub fn set_loras(&mut self, loras: &[lora::Lora]) -> Result<()> {
        match self.family.state() {
            Some(f) => f.set_loras(loras, &self.env.st),
            None => nothing_for(loras, "diffusion model"),
        }
    }

    /// Bake `loras` into the loaded text encoder, in place of any others, at their clip strengths.
    /// A strength of 0 skips a LoRA, as ComfyUI's loader does. Encoded prompts are then dropped.
    pub fn set_text_encoder_loras(&mut self, loras: &[lora::Lora]) -> Result<()> {
        let loras: Vec<_> = loras.iter().filter(|l| l.strength != 0.0).cloned().collect();
        match self.family.state() {
            Some(f) => f.set_text_encoder_loras(self.te.as_mut(), &loras, &self.env.st),
            None => shared_te_loras(self.te.as_mut(), &loras, &self.env.st).map(|_| ()),
        }
    }

    /// Choose the DiT attention: INT8 (fastest) or bf16 FlashAttention. Takes effect on the next step.
    pub fn set_int8_attention(&mut self, on: bool) {
        self.env.int8_attn = on;
        if let Some(f) = self.family.state() {
            f.set_int8_attention(on);
        }
    }

    /// Drop every model and give their VRAM and pinned RAM back. `ensure_*` loads them again.
    pub fn unload(&mut self) -> Result<()> {
        self.env.st.sync()?;
        self.family = Family::None;
        self.te = None;
        self.vae = None;
        cuda::trim_pool()
    }

    /// Nothing is running: other applications get the VRAM, and the next generation takes its blocks
    /// back from RAM. Models keep their place, LoRAs and prompts. Forced residency is left alone, and under
    /// ComfyUI's memory management (the default) weights stay in VRAM, as ComfyUI leaves them, until something else
    /// needs the room.
    pub fn idle(&mut self) -> Result<()> {
        self.env.st.sync()?;
        if self.env.dynamic {
            if let Some(f) = self.family.state() {
                f.idle()?;
            }
        }
        if self.env.comfy {
            crate::comfy_memory::node_done()?;
        }
        cuda::trim_pool()
    }

    /// Move the loaded models' weights out of VRAM into RAM now, keeping the models, LoRAs and prompts loaded: the
    /// next generation copies them back instead of reading the files again. Under ComfyUI's memory management every
    /// streamed unit goes (diffusion model, text encoders, vision towers); under `--legacy-offload` the diffusion
    /// model's blocks do, as after five idle minutes, unless `--resident-blocks` fixes them in VRAM. Small weights
    /// outside the streamed units, the VAE and encoded prompts stay in VRAM.
    pub fn release_vram(&mut self) -> Result<()> {
        self.env.st.sync()?;
        let before = cuda::device_free()?;
        let forced = !self.env.comfy && !self.env.dynamic;
        if !forced {
            if let Some(f) = self.family.state() {
                f.idle()?;
            }
        }
        if self.env.comfy {
            crate::comfy_memory::release_all()?;
        }
        cuda::trim_pool()?;
        let after = cuda::device_free()?;
        eprintln!("released {} MB of VRAM to RAM ({} MB free){}", after.saturating_sub(before) / MB, after / MB,
            if forced { "; --resident-blocks keeps the diffusion blocks in VRAM" } else { "" });
        Ok(())
    }

    /// Before an encoder or decoder runs: demote the loaded diffusion model's blocks until `need` bytes fit.
    fn make_room(&mut self, need: usize) -> Result<()> {
        if let Some((blocks, dst)) = self.family.state().and_then(|f| f.blocks()) {
            self.env.make_room(blocks, dst, need)?;
        }
        Ok(())
    }

    /// VAE-encode RGB f32 [height, width, 3], 0 to 1, into a 16-channel reference latent. Pixels past a multiple
    /// of 8 are cropped evenly from each side, as ComfyUI's VAE does.
    pub fn encode_image(&mut self, rgb: &[f32], width: usize, height: usize) -> Result<Reference> {
        let (cropped, cw, ch) = crop(rgb, width, height, 3, 8)?;
        self.make_room(vae::Vae::activation_bytes(ch / 8, cw / 8))?;
        let t0 = Instant::now();
        let latent = self.vae.as_ref().context("no VAE loaded")?.1.encode(&cropped, ch, cw, &self.env.st)?;
        eprintln!("VAE encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        Ok(Reference { latent, height: ch / 8, width: cw / 8 })
    }

    /// Decode a host latent from [`Generator::sample_krea`] or [`Generator::sample_anima`] to RGB8 pixels.
    pub fn decode_latent(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / 8, width / 8);
        if latent.len() != crate::dit::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), crate::dit::CH * lh * lw);
        }
        // the decoder's activations may need memory freed since the request started
        self.make_room(vae::Vae::activation_bytes(lh, lw))?;
        let st = &self.env.st;
        let d = upload(latent, st)?;
        self.vae.as_ref().context("no VAE loaded")?.1.decode(&d, lh, lw, st)
    }
}

pub fn encode_png(w: usize, h: usize, rgb: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(rgb)?;
    }
    Ok(out)
}
