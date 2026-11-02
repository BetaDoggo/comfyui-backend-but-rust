//! Loaded models + one text-to-image generation, shared by the CLI and the web server.

use crate::cuda::{self, DevBuf, Event, Stream};
use crate::samplers::{self, Eval, ModelSampling, Uncond};
use crate::vision::{self, PromptImage};
use crate::{anima, dit, ffi, flux2vae, lora, moss, offload, qwen3, qwen35, sampling, st, te, tokenizer, vae};
use anyhow::{bail, Context, Result};
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
}

/// What the three families share in a request: a sampler, its sigmas and the guidance.
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
    /// This image's place in its batch: the sampler noise is one tensor for the whole batch.
    pub batch_index: usize,
    pub batch_size: usize,
}

impl Sampling {
    /// One image at `cfg` with the given sampler and scheduler.
    pub fn new(sampler: &str, scheduler: &str, model_sampling: ModelSampling, steps: usize, cfg: f64, seed: u64, latent: (usize, usize)) -> Result<Self> {
        let sigmas = samplers::sigmas(model_sampling, sampler, scheduler, steps, latent, None)?;
        Ok(Sampling { sampler: sampler.to_string(), model_sampling, sigmas, cfg, seed, batch_index: 0, batch_size: 1 })
    }

    /// True when the negative prompt has to be encoded.
    pub fn uses_negative(&self) -> bool {
        (self.cfg - 1.0).abs() > 1e-6 || samplers::forces_uncond(&self.sampler)
    }
}

pub struct Request {
    pub prompt: String,
    /// Only encoded when the sampling uses it.
    pub negative: String,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub sampling: Sampling,
    /// Use this noise instead of the seed (parity tests).
    pub noise: Option<Vec<f32>>,
    /// Prompt images, through the vision tower into the text encoder.
    pub images: Vec<PromptImage>,
    pub negative_images: Vec<PromptImage>,
    /// Reference latents of the positive prompt, see [`Generator::encode_image`].
    pub references: Vec<Reference>,
    /// ComfyUI's `index_timestep_zero` rather than `index`.
    pub timestep_zero: bool,
}

/// A reference latent, [16, height, width] f32 at model scale.
#[derive(Clone)]
pub struct Reference {
    pub latent: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

/// A prompt as the text encoder saw it.
#[derive(PartialEq)]
struct PromptKey {
    text: String,
    images: Vec<PromptImage>,
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

struct DitModel {
    path: PathBuf,
    dit: dit::Dit,
    st: st::SafeTensors,
    lora: lora::State,
}

pub struct AnimaRequest {
    /// (text, weight) segments, see [`tokenizer::token_weights`]
    pub positive: Vec<(String, f32)>,
    /// Only encoded when the sampling uses it.
    pub negative: Vec<(String, f32)>,
    pub width: usize,
    pub height: usize,
    pub sampling: Sampling,
    pub noise: Vec<f32>,
}

struct AnimaModel {
    path: PathBuf,
    dit: anima::Dit,
    st: st::SafeTensors,
    lora: lora::State,
}

pub struct MossRequest {
    pub positive: String,
    /// Only encoded when the sampling uses it.
    pub negative: String,
    /// `{}` marks the prompt; None is ComfyUI's own (the prompt is trimmed and the think block left open).
    pub template: Option<String>,
    pub width: usize,
    pub height: usize,
    pub sampling: Sampling,
    pub noise: Vec<f32>,
}

struct MossModel {
    path: PathBuf,
    dit: moss::Dit,
    st: st::SafeTensors,
    lora: lora::State,
}

/// One run of a sampler. `model(x, out, sigma, negative, evaluation)` writes a denoised latent, and
/// is called twice per evaluation under CFG, always positive first.
fn run_sampler(s: &Sampling, noise: &[f32], st: &Stream, stats: &mut Stats,
    model: &mut dyn FnMut(&DevBuf, &DevBuf, f32, bool, usize) -> Result<()>, progress: &mut dyn FnMut(usize, usize) -> bool) -> Result<Vec<f32>> {
    let threads = unsafe { ffi::krea_randn_threads() };
    if threads <= 0 {
        bail!("could not read the GPU's core count");
    }
    let n = noise.len();
    let (x, den, und) = (DevBuf::alloc_async(n * 4, st)?, DevBuf::alloc_async(n * 4, st)?, DevBuf::alloc_async(n * 4, st)?);
    let (e0, e1) = (Event::timing()?, Event::timing()?);
    let cfg = s.cfg as f32;
    let guided = (s.cfg - 1.0).abs() > 1e-6;
    let mut evals = 0;
    let mut sample_ms = 0.0;
    let mut eval = |v: &[f32], sigma: f32, mode: Uncond| -> Result<Eval> {
        e0.record(st)?;
        x.upload(0, &st::f32_bytes(v), st)?;
        model(&x, &den, sigma, false, evals)?;
        let negative = guided || mode == Uncond::Force;
        if negative {
            model(&x, &und, sigma, true, evals)?;
            let (p, c) = ([und.ptr() as ffi::P, den.ptr() as ffi::P], [1.0 - cfg, cfg]);
            ffi::k!(krea_lincomb(den.ptr(), n as i64, 2, p.as_ptr(), c.as_ptr(), ffi::s(st)))?;
        }
        let read = |b: &DevBuf| -> Result<Vec<f32>> {
            let mut bytes = vec![0u8; n * 4];
            b.download(0, &mut bytes, st)?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let denoised = read(&den)?;
        let uncond = if negative && mode != Uncond::Skip { Some(read(&und)?) } else { None };
        e1.record(st)?;
        e1.sync()?;
        let ms = e1.elapsed_ms(&e0)? as f64;
        sample_ms += ms;
        evals += 1;
        eprintln!("evaluation {evals} sigma {sigma:.4}: {ms:.0} ms");
        Ok(Eval { denoised, uncond })
    };
    let mut step = |i: usize, total: usize| if progress(i, total) { Ok(()) } else { Err(Cancelled.into()) };
    let batch = samplers::Batch { index: s.batch_index, size: s.batch_size, threads: threads as usize };
    let out = samplers::sample(&s.sampler, s.model_sampling, &s.sigmas, noise, s.seed, batch, &mut eval, &mut step)?;
    stats.sample_ms = sample_ms;
    eprintln!("sampling {:.2}s ({:.0} ms/evaluation)", sample_ms / 1e3, sample_ms / evals.max(1) as f64);
    Ok(out)
}

type AnimaPrompt = Vec<(String, f32)>;

pub struct Generator {
    st: Stream,
    te: Option<(PathBuf, te::TextEncoder)>,
    dit: Option<DitModel>,
    anima: Option<AnimaModel>,
    qwen3: Option<(PathBuf, qwen3::TextEncoder)>,
    tokenizers: Option<(tokenizer::Tokenizer, tokenizer::T5Tokenizer)>,
    /// Latest encoded prompts, most recent last.
    contexts: Vec<(AnimaPrompt, anima::Context)>,
    moss: Option<MossModel>,
    qwen35: Option<(PathBuf, qwen35::TextEncoder)>,
    moss_tokenizer: Option<tokenizer::Tokenizer>,
    /// Latest encoded token ids, most recent last.
    moss_contexts: Vec<(Vec<u32>, moss::Context)>,
    vae: Option<(PathBuf, vae::Vae)>,
    flux2: Option<(PathBuf, flux2vae::Vae)>,
    cached: Option<(PromptKey, te::Cond, DevBuf)>,
    cached_negative: Option<(PromptKey, te::Cond, DevBuf)>,
    /// Loaded from the text encoder's file by the first prompt image.
    vision: Option<vision::Vision>,
    /// Latest images, most recent last.
    vision_cache: Vec<(PromptImage, Arc<vision::Embeds>)>,
    pub max_pixels: usize,
    max_width: usize,
    max_height: usize,
    reserve: usize,
    resident_blocks: Option<usize>,
    slots: usize,
    threads: usize,
    attn_variant: i32,
    int8_attn: bool,
    /// Re-plan DiT block residency from live free VRAM (off when --resident-blocks is forced).
    dynamic: bool,
}

/// Re-plan which DiT blocks live in VRAM from the memory free right now (other applications
/// included), keeping `need` bytes for activations plus `reserve`. Blocks are promoted from or
/// demoted to pinned RAM; nothing is reloaded from disk. The stream is synchronized first.
fn rebalance(st: &Stream, blocks: &mut offload::WeightSet, dst: &st::SafeTensors, reserve: usize, slots: usize, need: usize,
    demote_only: bool) -> Result<()> {
    st.sync()?;
    cuda::trim_pool()?;
    let free = cuda::device_free()?;
    let owned = blocks.vram_bytes();
    let avail = (free + owned).saturating_sub(need + reserve);
    let mut p = offload::plan(&blocks.layouts, avail, slots);
    if std::env::var_os("KREA_DEBUG_VRAM").is_some() {
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
        eprintln!("warning: VRAM oversubscribed (free {} MB, activations {} MB, reserve {} MB); streaming every                    block through one slot, expect slowdowns", free / MB, need / MB, reserve / MB);
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

const LT_MAX: usize = 512; // text tokens budgeted for activation planning
const ANIMA_CONTEXTS: usize = 4; // encoded prompts kept in VRAM
const VISION_CACHE: usize = 10; // the most prompt images SwarmUI sends
/// MOSS Image latents are at /16 and the VAE's latent has no text budget; 300 tokens is a long prompt.
const MOSS_TEXT: usize = 300;

/// Peak activation memory of one request (denoising step or VAE decode, whichever is larger).
/// `extra` is the tokens of its prompt images and reference latents.
fn activation_need(width: usize, height: usize, extra: usize) -> usize {
    let (lh, lw) = (height / 8, width / 8);
    dit::step_activation_bytes(LT_MAX + extra + (lh / 2) * (lw / 2)).max(vae::Vae::activation_bytes(lh, lw))
}

fn moss_need(width: usize, height: usize) -> usize {
    let (lh, lw) = (height / 16, width / 16);
    moss::step_activation_bytes(lh * lw + MOSS_TEXT).max(flux2vae::Vae::activation_bytes(lh, lw))
}

fn anima_need(width: usize, height: usize) -> usize {
    let (lh, lw) = (height / 8, width / 8);
    anima::step_activation_bytes((lh / 2) * (lw / 2)).max(vae::Vae::activation_bytes(lh, lw))
}

impl Generator {
    /// Start the GPU runtime with no models loaded; the `ensure_*` methods load them.
    pub fn new(o: &LoadOptions) -> Result<Self> {
        cuda::init(0)?;
        ffi::ok(unsafe { ffi::krea_blas_init(64 * MB) }, "cublasLt init")?;
        let st0 = Stream::new()?;
        let (_, total) = cuda::mem_info()?;
        eprintln!("GPU memory: {} MB free (device-wide) / {} MB total", cuda::device_free()? / MB, total / MB);
        Ok(Generator { st: st0, te: None, dit: None, anima: None, qwen3: None, tokenizers: None, contexts: Vec::new(), moss: None, qwen35: None,
            moss_tokenizer: None, moss_contexts: Vec::new(), vae: None, flux2: None, cached: None, cached_negative: None, vision: None, vision_cache: Vec::new(),
            max_pixels: o.max_pixels,
            max_width: o.max_width, max_height: o.max_height, reserve: o.reserve_vram_mb * MB,
            resident_blocks: o.resident_blocks, slots: o.slots, threads: o.threads, attn_variant: o.attn_variant,
            int8_attn: o.int8_attn, dynamic: o.resident_blocks.is_none() })
    }

    pub fn load(o: &LoadOptions) -> Result<Self> {
        let t_start = Instant::now();
        let mut gen = Self::new(o)?;
        gen.ensure_vae(&o.vae)?;
        gen.ensure_te(&o.te)?;
        gen.ensure_dit(&o.dit)?;
        eprintln!("startup total {:.1}s", t_start.elapsed().as_secs_f64());
        Ok(gen)
    }

    /// Load the VAE unless `path` is already loaded (replaces any other VAE).
    pub fn ensure_vae(&mut self, path: &Path) -> Result<()> {
        if self.vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.vae = None;
        let t0 = Instant::now();
        self.vae = Some((path.to_path_buf(), vae::Vae::load(path, &self.st)?));
        eprintln!("VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the text encoder unless `path` is already loaded (streamed from pinned RAM).
    pub fn ensure_te(&mut self, path: &Path) -> Result<()> {
        if self.te.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.te = None;
        self.vision = None;
        self.vision_cache.clear();
        self.cached = None;
        self.cached_negative = None;
        let t0 = Instant::now();
        self.te = Some((path.to_path_buf(), te::TextEncoder::load(path, 0, self.threads, &self.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the DiT unless `path` is already loaded, planning block residency from what is free.
    pub fn ensure_dit(&mut self, path: &Path) -> Result<()> {
        if self.dit.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        self.drop_anima()?;
        self.drop_moss()?;
        self.dit = None;
        self.cached = None;
        self.cached_negative = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = dit::globals_layout(&dst)?.size;
        let txt_slots = 2 * dit::txt_layouts(&dst)?.iter().map(|l| l.size).max().unwrap_or(0);
        let te_slots = 2 * 110 * MB;
        let act = activation_need(self.max_width, self.max_height, 0) + txt_slots + te_slots;
        let free = cuda::device_free()?;
        let mut budget = free.saturating_sub(self.reserve + globals + act);
        if let Some(r) = self.resident_blocks {
            let one = dit::block_layout(&dst, 0)?.size;
            budget = if r >= dit::n_blocks(&dst) { r * one } else { (r + self.slots) * one };
        }
        let mut dit = dit::Dit::load(&dst, budget, self.slots, self.threads, &self.st)?;
        dit.attn_variant = self.attn_variant;
        dit.int8_attn = self.int8_attn;
        eprintln!("DiT ready in {:.1}s (activation reserve {} MB)", t0.elapsed().as_secs_f64(), act / MB);
        self.dit = Some(DitModel { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    fn drop_krea(&mut self) -> Result<()> {
        self.st.sync()?;
        self.cached = None;
        self.cached_negative = None;
        self.dit = None;
        self.te = None;
        self.vision = None;
        self.vision_cache.clear();
        cuda::trim_pool()
    }

    fn drop_anima(&mut self) -> Result<()> {
        self.st.sync()?;
        self.contexts.clear();
        self.anima = None;
        self.qwen3 = None;
        cuda::trim_pool()
    }

    fn drop_moss(&mut self) -> Result<()> {
        self.st.sync()?;
        self.moss_contexts.clear();
        self.moss = None;
        self.qwen35 = None;
        self.flux2 = None;
        cuda::trim_pool()
    }

    /// Load an Anima checkpoint unless `path` is loaded. One model family is loaded at a time.
    pub fn ensure_anima(&mut self, path: &Path) -> Result<()> {
        if self.anima.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        self.drop_krea()?;
        self.drop_moss()?;
        self.contexts.clear();
        self.anima = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = anima::globals_layout(&dst)?.size;
        let free = cuda::device_free()?;
        let budget = free.saturating_sub(self.reserve + globals + anima_need(self.max_width, self.max_height));
        let mut dit = anima::Dit::load(&dst, budget, self.slots, self.threads, &self.st)?;
        dit.attn_variant = self.attn_variant;
        dit.int8_attn = self.int8_attn;
        eprintln!("Anima ready in {:.1}s", t0.elapsed().as_secs_f64());
        self.anima = Some(AnimaModel { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Load Anima's Qwen3 text encoder unless `path` is loaded; in VRAM if it fits, else streamed.
    pub fn ensure_qwen3(&mut self, path: &Path) -> Result<()> {
        if self.qwen3.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.drop_krea()?;
        self.drop_moss()?;
        self.contexts.clear();
        self.qwen3 = None;
        let t0 = Instant::now();
        if self.tokenizers.is_none() {
            self.tokenizers = Some((tokenizer::Tokenizer::new()?, tokenizer::T5Tokenizer::new()?));
        }
        let size = qwen3::weight_bytes(&st::SafeTensors::open(path)?)?;
        let keep = self.reserve + anima_need(self.max_width, self.max_height);
        let budget = if cuda::device_free()? >= size + keep { size } else { 0 };
        self.qwen3 = Some((path.to_path_buf(), qwen3::TextEncoder::load(path, budget, self.threads, &self.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load a MOSS Image diffusion model unless `path` is loaded.
    pub fn ensure_moss(&mut self, path: &Path) -> Result<()> {
        if self.moss.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        self.drop_krea()?;
        self.drop_anima()?;
        self.moss_contexts.clear();
        self.moss = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = moss::globals_layout(&dst)?.size;
        let free = cuda::device_free()?;
        let budget = free.saturating_sub(self.reserve + globals + moss_need(self.max_width, self.max_height));
        let mut dit = moss::Dit::load(&dst, budget, self.slots, self.threads, &self.st)?;
        dit.attn_variant = self.attn_variant;
        dit.int8_attn = self.int8_attn;
        eprintln!("MOSS Image ready in {:.1}s", t0.elapsed().as_secs_f64());
        self.moss = Some(MossModel { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Load MOSS Image's Qwen3.5 text encoder unless `path` is loaded; streamed from pinned RAM,
    /// since it doesn't fit in VRAM with the blocks on most cards.
    pub fn ensure_qwen35(&mut self, path: &Path) -> Result<()> {
        if self.qwen35.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.drop_krea()?;
        self.drop_anima()?;
        self.moss_contexts.clear();
        self.qwen35 = None;
        let t0 = Instant::now();
        if self.moss_tokenizer.is_none() {
            self.moss_tokenizer = Some(tokenizer::Tokenizer::qwen35()?);
        }
        let size = qwen35::weight_bytes(&st::SafeTensors::open(path)?)?;
        let keep = self.reserve + moss_need(self.max_width, self.max_height);
        let budget = if cuda::device_free()? >= size + keep { size } else { 0 };
        self.qwen35 = Some((path.to_path_buf(), qwen35::TextEncoder::load(path, budget, self.threads, &self.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the Flux 2 VAE unless `path` is loaded.
    pub fn ensure_flux2_vae(&mut self, path: &Path) -> Result<()> {
        if self.flux2.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.flux2 = None;
        let t0 = Instant::now();
        self.flux2 = Some((path.to_path_buf(), flux2vae::Vae::load(path, &self.st)?));
        eprintln!("VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Token ids of a MOSS Image prompt, as ComfyUI's tokenizer builds them.
    fn moss_ids(&self, prompt: &str, template: Option<&str>) -> Result<Vec<u32>> {
        let t = self.moss_tokenizer.as_ref().context("no text encoder loaded")?;
        let text = match template {
            Some(tpl) => tpl.replacen("{}", prompt, 1),
            None => qwen35::TEMPLATE.replacen("{}", prompt.trim(), 1),
        };
        t.encode(&text)
    }

    /// Move the context of `ids` to the end of the cache, encoding it if needed.
    fn moss_context(&mut self, ids: &[u32], dump: Option<&mut Dump>) -> Result<()> {
        if let Some(i) = self.moss_contexts.iter().position(|(p, _)| p == ids) {
            let hit = self.moss_contexts.remove(i);
            self.moss_contexts.push(hit);
            return Ok(());
        }
        let st = &self.st;
        let m = self.moss.as_mut().context("no diffusion model loaded")?;
        let te = &mut self.qwen35.as_mut().context("no text encoder loaded")?.1;
        let t0 = Instant::now();
        let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
        let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
            let mut v = vec![0u8; bytes];
            b.download(0, &mut v, st).unwrap();
            probes.push((name.to_string(), v));
        };
        let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
        let hidden = te.encode(ids, st, probe_ref)?;
        let ctx = m.dit.encode_context(&hidden, ids.len(), st)?;
        st.sync()?;
        eprintln!("text: {} tokens in {:.0} ms", ids.len(), t0.elapsed().as_secs_f64() * 1e3);
        if let Some(d) = dump {
            d.insert("ids".into(), ("I64".into(), vec![ids.len()], ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect()));
            for (name, v) in probes {
                d.insert(name, ("F32".into(), vec![ids.len(), qwen35::D], v));
            }
            let mut b = vec![0u8; hidden.size()];
            hidden.download(0, &mut b, st)?;
            d.insert("cond".into(), ("F32".into(), vec![ids.len(), qwen35::D], b));
            let mut b = vec![0u8; ctx.ctx.size()];
            ctx.ctx.download(0, &mut b, st)?;
            d.insert("text_embedder_out".into(), ("BF16".into(), vec![ids.len(), 3072], b));
        }
        self.moss_contexts.push((ids.to_vec(), ctx));
        if self.moss_contexts.len() > ANIMA_CONTEXTS {
            self.moss_contexts.remove(0);
        }
        Ok(())
    }

    /// One MOSS Image latent, sampled as ComfyUI's KSampler does.
    pub fn sample_moss(&mut self, r: &MossRequest, mut progress: impl FnMut(usize, usize) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        if r.width % 16 != 0 || r.height % 16 != 0 || r.width == 0 || r.height == 0 {
            bail!("width and height must be positive multiples of 16");
        }
        if r.width * r.height > self.max_pixels {
            bail!("{}x{} exceeds the {} pixel limit", r.width, r.height, self.max_pixels);
        }
        let (lh, lw) = (r.height / 16, r.width / 16);
        let n = moss::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        if self.dynamic {
            let m = self.moss.as_mut().context("no diffusion model loaded")?;
            rebalance(&self.st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, moss_need(r.width, r.height), false)?;
        }
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let guided = r.sampling.uses_negative();
        let pos_ids = self.moss_ids(&r.positive, r.template.as_deref())?;
        let neg_ids = self.moss_ids(&r.negative, r.template.as_deref())?;
        if guided {
            self.moss_context(&neg_ids, None)?;
        }
        self.moss_context(&pos_ids, dump.as_deref_mut())?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        let st = &self.st;
        let m = self.moss.as_mut().unwrap();
        let find = |p: &[u32]| self.moss_contexts.iter().find(|(k, _)| k == p).map(|(_, c)| c).unwrap();
        let pos = find(&pos_ids);
        let neg = guided.then(|| find(&neg_ids));
        let rope = m.dit.rope(lh, lw, pos.len, st)?;
        let neg_rope = neg.map(|c| m.dit.rope(lh, lw, c.len, st)).transpose()?;
        let copied0 = m.dit.blocks.copied_bytes;
        let (dynamic, reserve, slots) = (self.dynamic, self.reserve, self.slots);
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, eval: usize| -> Result<()> {
            if negative {
                return m.dit.predict(x, lh, lw, neg.unwrap(), neg_rope.as_ref().unwrap(), sigma, out, st, None);
            }
            if dynamic && eval > 0 && cuda::device_free()? < reserve / 2 {
                eprintln!("VRAM pressure: free {} MB < {} MB", cuda::device_free()? / MB, reserve / 2 / MB);
                let need = moss::step_activation_bytes(lh * lw + MOSS_TEXT) + reserve;
                rebalance(st, &mut m.dit.blocks, &m.st, reserve, slots, need, true)?;
            }
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && matches!(name, "video_embedder_out" | "block0" | "block1" | "block41" | "final_tokens") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probes.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.predict(x, lh, lw, pos, &rope, sigma, out, st, probe_ref)?;
            }
            if let Some(dm) = dump.as_deref_mut() {
                for (name, v) in probes {
                    dm.insert(name, ("BF16".into(), vec![v.len() / 2], v));
                }
            }
            Ok(())
        };
        let latent = run_sampler(&r.sampling, &r.noise, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(dm) = dump {
            dm.insert("samples".into(), ("F32".into(), vec![moss::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }

    /// Decode a host MOSS Image latent ([128, H/16, W/16] f32) to RGB8 pixels.
    pub fn decode_moss(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / 16, width / 16);
        if latent.len() != moss::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), moss::CH * lh * lw);
        }
        let st = &self.st;
        if let Some(m) = self.moss.as_mut().filter(|_| self.dynamic) {
            let need = flux2vae::Vae::activation_bytes(lh, lw);
            if cuda::device_free()? < need + self.reserve {
                rebalance(st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, need, true)?;
            }
        }
        let d = DevBuf::alloc_async(latent.len() * 4, st)?;
        d.upload(0, &st::f32_bytes(latent), st)?;
        self.flux2.as_ref().context("no VAE loaded")?.1.decode(&d, lh, lw, st)
    }

    /// Move `prompt`'s context to the end of the cache, encoding it if needed.
    fn anima_context(&mut self, prompt: &[(String, f32)], dump: Option<&mut Dump>) -> Result<()> {
        if let Some(i) = self.contexts.iter().position(|(p, _)| p == prompt) {
            let hit = self.contexts.remove(i);
            self.contexts.push(hit);
            return Ok(());
        }
        let st = &self.st;
        let m = self.anima.as_mut().context("no diffusion model loaded")?;
        let te = &mut self.qwen3.as_mut().context("no text encoder loaded")?.1;
        let (tq, tt) = self.tokenizers.as_ref().context("no text encoder loaded")?;
        let t0 = Instant::now();
        let (qids, tids, weights) = tokenizer::anima_tokens(tq, tt, prompt)?;
        let hidden = te.encode(&qids, st)?;
        let ctx = m.dit.encode_context(&m.st, &hidden, qids.len(), &tids, &weights, st)?;
        st.sync()?;
        eprintln!("text: {} Qwen3 / {} T5 tokens in {:.0} ms", qids.len(), tids.len(), t0.elapsed().as_secs_f64() * 1e3);
        if let Some(d) = dump {
            let ids = |v: &[u32]| v.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect::<Vec<u8>>();
            d.insert("qwen_ids".into(), ("I64".into(), vec![qids.len()], ids(&qids)));
            d.insert("t5_ids".into(), ("I64".into(), vec![tids.len()], ids(&tids)));
            let mut b = vec![0u8; hidden.size()];
            hidden.download(0, &mut b, st)?;
            d.insert("cond".into(), ("F32".into(), vec![qids.len(), qwen3::D], b));
            let mut b = vec![0u8; ctx.ctx.size()];
            ctx.ctx.download(0, &mut b, st)?;
            d.insert("dit_in_ctx".into(), ("BF16".into(), vec![ctx.len, anima::CTX], b));
        }
        self.contexts.push((prompt.to_vec(), ctx));
        if self.contexts.len() > ANIMA_CONTEXTS {
            self.contexts.remove(0);
        }
        Ok(())
    }

    /// One Anima latent, sampled as ComfyUI's KSampler does.
    pub fn sample_anima(&mut self, r: &AnimaRequest, mut progress: impl FnMut(usize, usize) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        if r.width % 16 != 0 || r.height % 16 != 0 || r.width == 0 || r.height == 0 {
            bail!("width and height must be positive multiples of 16");
        }
        if r.width * r.height > self.max_pixels {
            bail!("{}x{} exceeds the {} pixel limit", r.width, r.height, self.max_pixels);
        }
        let (lh, lw) = (r.height / 8, r.width / 8);
        let n = dit::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        if self.dynamic {
            let m = self.anima.as_mut().context("no diffusion model loaded")?;
            rebalance(&self.st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, anima_need(r.width, r.height), false)?;
        }
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let guided = r.sampling.uses_negative();
        if guided {
            self.anima_context(&r.negative, None)?;
        }
        self.anima_context(&r.positive, dump.as_deref_mut())?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        let st = &self.st;
        let m = self.anima.as_mut().unwrap();
        let find = |p: &[(String, f32)]| self.contexts.iter().find(|(k, _)| k == p).map(|(_, c)| c).unwrap();
        let pos = find(&r.positive);
        let neg = guided.then(|| find(&r.negative));
        let rope = m.dit.rope(lh / 2, lw / 2, st)?;
        let copied0 = m.dit.blocks.copied_bytes;
        let (dynamic, reserve, slots) = (self.dynamic, self.reserve, self.slots);
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, eval: usize| -> Result<()> {
            if negative {
                return m.dit.predict(x, lh, lw, neg.unwrap(), &rope, sigma, out, st, None);
            }
            if dynamic && eval > 0 && cuda::device_free()? < reserve / 2 {
                eprintln!("VRAM pressure: free {} MB < {} MB", cuda::device_free()? / MB, reserve / 2 / MB);
                let need = anima::step_activation_bytes((lh / 2) * (lw / 2)) + reserve;
                rebalance(st, &mut m.dit.blocks, &m.st, reserve, slots, need, true)?;
            }
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && matches!(name, "x_embedder_out" | "block0" | "block27" | "final_tokens" | "block0_self_attn_raw") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probes.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.predict(x, lh, lw, pos, &rope, sigma, out, st, probe_ref)?;
            }
            if let Some(dm) = dump.as_deref_mut() {
                for (name, v) in probes {
                    dm.insert(name, ("BF16".into(), vec![v.len() / 2], v));
                }
            }
            Ok(())
        };
        let latent = run_sampler(&r.sampling, &r.noise, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(dm) = dump {
            dm.insert("samples".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }

    /// Bake `loras` into the loaded diffusion model, in place of any others. Prompts encoded through
    /// the old weights are dropped: their contexts come out of the diffusion model.
    pub fn set_loras(&mut self, loras: &[lora::Lora]) -> Result<()> {
        let st = &self.st;
        if let Some(m) = &mut self.dit {
            if lora::apply(lora::Arch::Krea2, &m.st, "", m.dit.stores(), &mut m.lora, loras, st)? {
                self.cached = None;
                self.cached_negative = None;
            }
        } else if let Some(m) = &mut self.anima {
            let prefix = m.dit.prefix();
            if lora::apply(lora::Arch::Anima, &m.st, &prefix, m.dit.stores(), &mut m.lora, loras, st)? {
                m.dit.clear_mods();
                self.contexts.clear();
            }
        } else if let Some(m) = &mut self.moss {
            if lora::apply(lora::Arch::Moss, &m.st, "", m.dit.stores(), &mut m.lora, loras, st)? {
                m.dit.clear_mods();
                self.moss_contexts.clear();
            }
        } else if !loras.is_empty() {
            bail!("no diffusion model loaded");
        }
        Ok(())
    }

    /// Choose the DiT attention: INT8 (fastest) or bf16 FlashAttention. Takes effect on the next step.
    pub fn set_int8_attention(&mut self, on: bool) {
        self.int8_attn = on;
        if let Some(m) = &mut self.dit {
            m.dit.int8_attn = on;
        }
        if let Some(m) = &mut self.anima {
            m.dit.int8_attn = on;
        }
        if let Some(m) = &mut self.moss {
            m.dit.int8_attn = on;
        }
    }

    /// Drop every model and give their VRAM and pinned RAM back. `ensure_*` loads them again.
    pub fn unload(&mut self) -> Result<()> {
        self.st.sync()?;
        self.cached = None;
        self.cached_negative = None;
        self.contexts.clear();
        self.dit = None;
        self.te = None;
        self.vision = None;
        self.vision_cache.clear();
        self.anima = None;
        self.qwen3 = None;
        self.moss_contexts.clear();
        self.moss = None;
        self.qwen35 = None;
        self.flux2 = None;
        self.vae = None;
        cuda::trim_pool()
    }

    /// Give back cached VRAM without unloading anything.
    pub fn trim(&mut self) -> Result<()> {
        self.st.sync()?;
        cuda::trim_pool()
    }

    /// Generate one image; returns RGB8 pixels. `progress(step, steps)` is called after each step
    /// and stops the run with [`Cancelled`] by returning false.
    pub fn generate(&mut self, r: &Request, progress: impl FnMut(usize, usize) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<u8>, Stats)> {
        let t_img = Instant::now();
        let (latent, mut stats) = self.denoise(r, progress, dump.as_deref_mut())?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let t0 = Instant::now();
        let rgb = self.decode_latent(&latent, r.width, r.height)?;
        stats.vae_ms = t0.elapsed().as_secs_f64() * 1e3;
        stats.total_ms = t_img.elapsed().as_secs_f64() * 1e3;
        eprintln!("VAE decode {:.0} ms, image done in {:.2}s", stats.vae_ms, stats.total_ms / 1e3);
        if let Some(d) = dump {
            d.insert("image_rgb8".into(), ("U8".into(), vec![lh * 8, lw * 8, 3], rgb.clone()));
        }
        Ok((rgb, stats))
    }

    /// Text encoding and denoising only; returns the final latent ([16, H/8, W/8] f32) on the host.
    pub fn sample_latent(&mut self, r: &Request, progress: impl FnMut(usize, usize) -> bool) -> Result<(Vec<f32>, Stats)> {
        self.denoise(r, progress, None)
    }

    /// The vision tower's tokens for `images`, through the cache.
    fn vision_embeds(&mut self, images: &[PromptImage], mut dump: Option<&mut Dump>) -> Result<Vec<Arc<vision::Embeds>>> {
        let mut out = Vec::with_capacity(images.len());
        for img in images {
            if let Some(i) = self.vision_cache.iter().position(|(k, _)| k == img) {
                let hit = self.vision_cache.remove(i);
                out.push(hit.1.clone());
                self.vision_cache.push(hit);
                continue;
            }
            if self.vision.is_none() {
                let t0 = Instant::now();
                let path = &self.te.as_ref().context("no text encoder loaded")?.0;
                self.vision = Some(vision::Vision::load(st::SafeTensors::open(path)?, self.threads, &self.st)?);
                eprintln!("vision tower ready in {:.1}s", t0.elapsed().as_secs_f64());
            }
            let t0 = Instant::now();
            let st = &self.st;
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
            self.vision_cache.push((img.clone(), e));
            if self.vision_cache.len() > VISION_CACHE {
                self.vision_cache.remove(0);
            }
        }
        Ok(out)
    }

    /// VAE-encode RGB f32 [height, width, 3], 0 to 1, into a reference latent. Pixels past a multiple
    /// of 8 are cropped evenly from each side, as ComfyUI's VAE does.
    pub fn encode_image(&mut self, rgb: &[f32], width: usize, height: usize) -> Result<Reference> {
        if rgb.len() != width * height * 3 {
            bail!("image has {} values, expected {}", rgb.len(), width * height * 3);
        }
        let (cw, ch) = (width / 8 * 8, height / 8 * 8);
        if cw == 0 || ch == 0 {
            bail!("{width}x{height} is too small to encode");
        }
        let (x0, y0) = ((width - cw) / 2, (height - ch) / 2);
        let mut cropped = Vec::with_capacity(cw * ch * 3);
        for y in y0..y0 + ch {
            cropped.extend_from_slice(&rgb[(y * width + x0) * 3..][..cw * 3]);
        }
        let st = &self.st;
        let need = vae::Vae::activation_bytes(ch / 8, cw / 8);
        if let Some(m) = self.dit.as_mut().filter(|_| self.dynamic) {
            if cuda::device_free()? < need + self.reserve {
                rebalance(st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, need, true)?;
            }
        }
        let t0 = Instant::now();
        let latent = self.vae.as_ref().context("no VAE loaded")?.1.encode(&cropped, ch, cw, st)?;
        eprintln!("VAE encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        Ok(Reference { latent, height: ch / 8, width: cw / 8 })
    }

    /// Decode a host latent from [`Generator::sample_latent`] to RGB8 pixels.
    pub fn decode_latent(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / 8, width / 8);
        if latent.len() != dit::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), dit::CH * lh * lw);
        }
        let d = DevBuf::alloc_async(latent.len() * 4, &self.st)?;
        d.upload(0, &st::f32_bytes(latent), &self.st)?;
        self.decode(&d, lh, lw)
    }

    fn denoise(&mut self, r: &Request, mut progress: impl FnMut(usize, usize) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        if r.width % 16 != 0 || r.height % 16 != 0 || r.width == 0 || r.height == 0 {
            bail!("width and height must be positive multiples of 16");
        }
        if r.width * r.height > self.max_pixels {
            bail!("{}x{} exceeds the {} pixel limit (--max-side)", r.width, r.height, self.max_pixels);
        }
        let guided = r.sampling.uses_negative();
        let t0 = Instant::now();
        let pos_vision = self.vision_embeds(&r.images, dump.as_deref_mut())?;
        let neg_vision = if guided { self.vision_embeds(&r.negative_images, None)? } else { Vec::new() };
        let vision_ms = t0.elapsed().as_secs_f64() * 1e3;
        let m = self.dit.as_mut().context("no diffusion model loaded")?;
        let te = &mut self.te.as_mut().context("no text encoder loaded")?.1;
        let ref_grids: Vec<(usize, usize)> = r.references.iter().map(|l| (l.height.div_ceil(dit::PATCH), l.width.div_ceil(dit::PATCH))).collect();
        let extra = pos_vision.iter().chain(&neg_vision).map(|e| e.tokens).max().unwrap_or(0) * r.images.len().max(r.negative_images.len())
            + ref_grids.iter().map(|(h, w)| h * w).sum::<usize>();
        let need = activation_need(r.width, r.height, extra);
        if self.dynamic {
            rebalance(&self.st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, need, false)?;
        }
        let st = &self.st;
        let mut stats = Stats::default();
        let (lh, lw) = (r.height / 8, r.width / 8);

        // text encoding (reused while the prompt is unchanged)
        let t0 = Instant::now();
        let refs: Vec<&vision::Embeds> = pos_vision.iter().map(|e| &**e).collect();
        let key = PromptKey { text: r.prompt.clone(), images: r.images.clone() };
        if self.cached.as_ref().map_or(true, |(p, _, _)| *p != key) {
            drop(self.cached.take());
            let text = if refs.is_empty() { r.prompt.clone() } else { te::image_prompt(&r.prompt, refs.len()) };
            let cond = te.encode_with_images(&text, &refs, st)?;
            let ctx = m.dit.encode_context(&cond.ctx, cond.len, st)?;
            st.sync()?;
            eprintln!("text: {} tokens ({} conditioning) in {:.0} ms", cond.token_ids.len(), cond.len,
                t0.elapsed().as_secs_f64() * 1e3);
            self.cached = Some((key, cond, ctx));
        }
        let key = PromptKey { text: r.negative.clone(), images: r.negative_images.clone() };
        if guided && self.cached_negative.as_ref().map_or(true, |(p, _, _)| *p != key) {
            drop(self.cached_negative.take());
            let refs: Vec<&vision::Embeds> = neg_vision.iter().map(|e| &**e).collect();
            let text = if refs.is_empty() { r.negative.clone() } else { te::image_prompt(&r.negative, refs.len()) };
            let cond = te.encode_with_images(&text, &refs, st)?;
            let ctx = m.dit.encode_context(&cond.ctx, cond.len, st)?;
            st.sync()?;
            self.cached_negative = Some((key, cond, ctx));
        }
        stats.text_ms = vision_ms + t0.elapsed().as_secs_f64() * 1e3;
        let (_, cond, ctx) = self.cached.as_ref().unwrap();
        if let Some(d) = dump.as_deref_mut() {
            let mut b = vec![0u8; cond.ctx.size()];
            cond.ctx.download(0, &mut b, st)?;
            d.insert("cond".into(), ("BF16".into(), vec![cond.len, dit::TXT_LAYERS * dit::TXT_D], b));
            let mut b = vec![0u8; ctx.size()];
            ctx.download(0, &mut b, st)?;
            d.insert("txtmlp_out".into(), ("BF16".into(), vec![cond.len, dit::F], b));
            let ids: Vec<u8> = cond.token_ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect();
            d.insert("token_ids".into(), ("I64".into(), vec![cond.token_ids.len()], ids));
        }

        // noise (same as ComfyUI's KSampler for the seed)
        let n = dit::CH * lh * lw;
        let noise = match &r.noise {
            Some(v) => v.clone(),
            None => sampling::torch_randn(r.seed, n),
        };
        if noise.len() != n {
            bail!("noise has {} values, expected {n}", noise.len());
        }
        let upload_rope = |lt: usize, grids: &[(usize, usize)]| -> Result<DevBuf> {
            let rope = dit::Dit::rope_table(lt, lh / 2, lw / 2, grids);
            let d = DevBuf::alloc_async(rope.len() * 4, st)?;
            d.upload(0, &st::f32_bytes(&rope), st)?;
            Ok(d)
        };
        let lt = cond.len;
        let rope_d = upload_rope(lt, &ref_grids)?;
        let latents: Vec<(Vec<f32>, usize, usize)> = r.references.iter().map(|l| (l.latent.clone(), l.height, l.width)).collect();
        let references = if latents.is_empty() { None } else { Some(m.dit.references(&latents, r.timestep_zero, st)?) };
        let negative = if guided { self.cached_negative.as_ref().map(|(_, c, ctx)| (ctx, c.len)) } else { None };
        let neg_rope = negative.map(|(_, len)| upload_rope(len, &[])).transpose()?;

        // denoise
        let copied0 = m.dit.blocks.copied_bytes;
        let (dynamic, reserve, slots) = (self.dynamic, self.reserve, self.slots);
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, neg: bool, eval: usize| -> Result<()> {
            if neg {
                let (c, len) = negative.unwrap();
                return m.dit.step(x, lh, lw, c, len, None, neg_rope.as_ref().unwrap(), sigma, sigma, st, Some(out), None);
            }
            // another application took VRAM mid-run: demote blocks before the driver starts
            // paging to shared system memory (which would be far slower than streaming)
            if dynamic && eval > 0 && cuda::device_free()? < reserve / 2 {
                eprintln!("VRAM pressure: free {} MB < {} MB", cuda::device_free()? / MB, reserve / 2 / MB);
                // NVML clamps free at 0 when oversubscribed, hiding the real shortfall: ask for an
                // extra reserve each time so blocks keep moving out until pressure clears
                let step_need = dit::step_activation_bytes(LT_MAX + extra + (lh / 2) * (lw / 2));
                rebalance(st, &mut m.dit.blocks, &m.st, reserve, slots, step_need + reserve, true)?;
            }
            let mut probe_store: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && (name == "block0" || name == "block27" || name == "dit_out_tokens") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probe_store.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> =
                    if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.step(x, lh, lw, ctx, lt, references.as_ref(), &rope_d, sigma, sigma, st, Some(out), probe_ref)?;
            }
            if let Some(d) = dump.as_deref_mut() {
                for (name, v) in probe_store {
                    let shape = if name == "dit_out_tokens" { vec![(lh / 2) * (lw / 2), 64] } else { vec![v.len() / 2 / dit::F, dit::F] };
                    d.insert(name, ("BF16".into(), shape, v));
                }
            }
            Ok(())
        };
        let latent = run_sampler(&r.sampling, &noise, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(d) = dump {
            d.insert("latent_final".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }

    fn decode(&mut self, latent: &DevBuf, lh: usize, lw: usize) -> Result<Vec<u8>> {
        let st = &self.st;
        // the decoder's activations may need memory freed since the request started
        if let Some(m) = self.dit.as_mut().filter(|_| self.dynamic) {
            if cuda::device_free()? < vae::Vae::activation_bytes(lh, lw) + self.reserve {
                rebalance(st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, vae::Vae::activation_bytes(lh, lw), true)?;
            }
        }
        if let Some(m) = self.anima.as_mut().filter(|_| self.dynamic) {
            if cuda::device_free()? < vae::Vae::activation_bytes(lh, lw) + self.reserve {
                rebalance(st, &mut m.dit.blocks, &m.st, self.reserve, self.slots, vae::Vae::activation_bytes(lh, lw), true)?;
            }
        }
        self.vae.as_ref().context("no VAE loaded")?.1.decode(latent, lh, lw, st)
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
