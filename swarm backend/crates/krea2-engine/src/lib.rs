//! [`Engine`] implementation on the krea2 pipeline: turbo text-to-image for the Krea 2 W8A8
//! (ConvRot int8) checkpoint, the same way ComfyUI samples it at CFG 1 with `euler` / `simple`.
//!
//! Text encoding is deferred to `sample`, so the negative prompt SwarmUI always sends costs nothing
//! at CFG 1. Everything the pipeline can't do is refused with [`EngineError::Unsupported`], which
//! SwarmUI shows to the user, instead of being ignored and producing a different image.

mod noise;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use krea2::pipeline::{Cancelled, Generator, LoadOptions, Request};
use swarm_engine_api::prompt::PromptLeafKind;
use swarm_engine_api::*;

/// Attention backends, default first. SwarmUI's "pytorch attention" maps to the first entry.
const ATTENTION: [&str; 2] = ["int8", "flash"];
/// ComfyUI's `Krea2.sampling_settings`, used when SwarmUI sends no Sigma Shift.
const DEFAULT_SHIFT: f64 = 1.15;

pub struct Krea2Options {
    /// VRAM (MB) to leave untouched for the desktop and other applications.
    pub reserve_vram_mb: usize,
    /// Largest side (pixels, at 1:1) of a generation.
    pub max_side: usize,
    pub threads: usize,
}

impl Default for Krea2Options {
    fn default() -> Self {
        Self { reserve_vram_mb: 512, max_side: 2048, threads: 6 }
    }
}

pub struct Krea2Engine {
    gen: Mutex<Generator>,
    default_shift: f64,
}

/// The prompt to encode, or why this conditioning can't be used. Only checked when a prompt is
/// really needed: at CFG 1 the negative is never read.
struct DeferredCond {
    text_encoder: PathBuf,
    prompt: Result<String, String>,
}

/// Sampled latents, `[B, 16, H/8, W/8]`.
struct HostLatent {
    data: Vec<f32>,
}

fn unsupported(what: impl Into<String>) -> EngineError {
    EngineError::Unsupported(what.into())
}

fn internal(e: anyhow::Error) -> EngineError {
    if e.is::<Cancelled>() {
        EngineError::Interrupted
    } else {
        EngineError::Internal(e)
    }
}

/// The safetensors header, as JSON.
fn read_header(path: &Path) -> anyhow::Result<serde_json::Value> {
    let mut f = std::fs::File::open(path)?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let mut h = vec![0u8; u64::from_le_bytes(n) as usize];
    f.read_exact(&mut h)?;
    Ok(serde_json::from_slice(&h)?)
}

/// Refuses other architectures and formats before any weights are touched.
fn check_dit(file: &ModelFile) -> EngineResult<()> {
    let h = read_header(&file.path).map_err(|e| unsupported(format!("{}: not a readable safetensors file ({e})", file.name)))?;
    // The keys SwarmUI's `isKrea2` tests.
    if ["txtfusion.projector.weight", "blocks.0.attn.gate.weight", "blocks.0.mod.lin"].iter().any(|k| h.get(k).is_none()) {
        return Err(unsupported(format!("{}: this backend only runs Krea 2 models", file.name)));
    }
    if h.get("blocks.0.attn.gate.weight_scale").is_none() {
        return Err(unsupported(format!("{}: this backend only runs W8A8 (ConvRot int8) Krea 2 checkpoints", file.name)));
    }
    Ok(())
}

/// The plain text of a prompt, or why it needs something the pipeline doesn't have.
fn prompt_text(p: &ParsedPrompt, steps: u32) -> Result<String, String> {
    if p.has_steps() {
        return Err("<alternate> and <fromto> prompts".into());
    }
    let chunks = p.chunks(0, steps.max(1));
    if chunks.len() != 1 {
        return Err("<break> in prompts".into());
    }
    let mut text = String::new();
    for leaf in &chunks[0] {
        if leaf.weight != 1.0 {
            return Err("prompt weights (<weight>)".into());
        }
        match &leaf.kind {
            PromptLeafKind::Text(t) => text.push_str(t),
            PromptLeafKind::Embed(_) => return Err("embeddings in prompts".into()),
            PromptLeafKind::LoraHook(_) => return Err("LoRA hooks in prompts".into()),
            PromptLeafKind::Break => unreachable!("chunks() splits on breaks"),
        }
    }
    Ok(text)
}

fn encode_request(req: &TextEncodeRequest) -> Result<String, String> {
    if !req.text_encoder.patches.is_empty() {
        return Err("LoRAs and CLIP Stop At Layer".into());
    }
    match &req.mode {
        TextEncodeMode::Plain { text } => Ok(text.clone()),
        TextEncodeMode::Swarm(s) => {
            if s.images.is_some() {
                return Err("Prompt Images".into());
            }
            if s.minimax_refs.is_some() || s.clip_vision.is_some() {
                return Err("image references".into());
            }
            if s.lora_hooks.as_ref().is_some_and(|h| !h.is_empty()) {
                return Err("LoRA hooks".into());
            }
            if s.guidance.is_some() {
                return Err("Flux Guidance Scale".into());
            }
            if !matches!(s.llama_template.as_deref(), None | Some("krea2")) {
                return Err(format!("prompt template {:?}", s.llama_template));
            }
            prompt_text(&s.prompt, s.steps)
        }
    }
}

/// The one plain prompt of a conditioning: anything regional or image-based is refused.
fn single_prompt(c: &Conditioning) -> EngineResult<(&Path, &str)> {
    let [e] = c.entries.as_slice() else {
        return Err(unsupported("regional prompts (more than one conditioning entry)"));
    };
    let m = &e.meta;
    let plain = m.start_percent == 0.0
        && m.end_percent == 1.0
        && m.multiplier == 1.0
        && !m.zeroed
        && m.mask.is_none()
        && m.reference_latents.is_empty()
        && m.gligen.is_empty()
        && m.inpaint.is_none()
        && m.embedding_ops.is_empty()
        && m.unclip.is_empty()
        && m.controls.is_empty();
    if !plain {
        return Err(unsupported("conditioning features (regions, references, ControlNet, style models, ...)"));
    }
    let cond = e.embedding.downcast_ref::<DeferredCond>().ok_or_else(|| unsupported("conditioning from another text encoder"))?;
    let text = cond.prompt.as_deref().map_err(|e| unsupported(e.clone()))?;
    Ok((&cond.text_encoder, text))
}

/// `ModelSamplingFlux`: `max_shift` applies at 2560 latent patches, `base_shift` at 256.
fn flux_shift(max_shift: f64, base_shift: f64, width: u32, height: u32) -> f64 {
    let mm = (max_shift - base_shift) / (2560.0 - 256.0);
    (width as f64 * height as f64 / (8.0 * 8.0 * 2.0 * 2.0)) * mm + base_shift - mm * 256.0
}

struct ModelSettings {
    shift: f64,
    int8_attention: bool,
}

fn model_settings(model: &ModelHandle, default_shift: f64) -> EngineResult<ModelSettings> {
    let mut s = ModelSettings { shift: default_shift, int8_attention: true };
    for p in &model.patches {
        match p {
            ModelPatch::SamplingFlux { max_shift, base_shift, width, height } => s.shift = flux_shift(*max_shift, *base_shift, *width, *height),
            ModelPatch::AttentionBackend { backend } => {
                s.int8_attention = match backend.as_str() {
                    "int8" => true,
                    "flash" => false,
                    b => return Err(unsupported(format!("attention backend {b}"))),
                }
            }
            // Weighted prompts are refused in encode_text; with none this patch has no effect.
            ModelPatch::AttnTokenWeights { .. } | ModelPatch::Compile { .. } => {}
            ModelPatch::Lora { .. } => return Err(unsupported("LoRAs")),
            other => {
                let name = format!("{other:?}");
                return Err(unsupported(format!("model option {}", name.split([' ', '{', '(']).next().unwrap_or("?"))));
            }
        }
    }
    Ok(s)
}

/// Everything in a sampler request beyond seeded turbo text-to-image is refused here.
fn check_sample(req: &SampleRequest) -> EngineResult<()> {
    if (req.cfg - 1.0).abs() > 1e-6 {
        return Err(unsupported(format!("CFG Scale {} (this backend runs Krea 2 Turbo: set CFG Scale to 1)", req.cfg)));
    }
    if req.sampler != "euler" {
        return Err(unsupported(format!("sampler {} (use euler)", req.sampler)));
    }
    if req.scheduler != "simple" {
        return Err(unsupported(format!("scheduler {} (use simple)", req.scheduler)));
    }
    if req.start_at_step != 0 || req.end_at_step < req.steps || !req.add_noise || req.denoise != 1.0 {
        return Err(unsupported("partial sampling (init images, refiners)"));
    }
    if !req.latent.empty || req.latent.noise_mask.is_some() {
        return Err(unsupported("init images and masks (no VAE encoder yet)"));
    }
    if req.sigma_min.is_some() || req.sigma_max.is_some() {
        return Err(unsupported("sigma min / max overrides"));
    }
    if req.tile_size.is_some() {
        return Err(unsupported("tiled sampling"));
    }
    Ok(())
}

impl Krea2Engine {
    pub fn new(o: &Krea2Options) -> anyhow::Result<Self> {
        let load = LoadOptions {
            dit: PathBuf::new(),
            te: PathBuf::new(),
            vae: PathBuf::new(),
            max_width: 1024,
            max_height: 1024,
            max_pixels: o.max_side * o.max_side,
            reserve_vram_mb: o.reserve_vram_mb,
            resident_blocks: None,
            slots: 2,
            threads: o.threads,
            attn_variant: 0,
            int8_attn: true,
        };
        Ok(Self { gen: Mutex::new(Generator::new(&load)?), default_shift: DEFAULT_SHIFT })
    }

    fn gen(&self) -> std::sync::MutexGuard<'_, Generator> {
        // A panic mid-generation leaves nothing the next prompt depends on.
        self.gen.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Engine for Krea2Engine {
    fn info(&self) -> EngineInfo {
        // Runs on server threads, possibly mid-generation: no generator lock here.
        let (vram_free, vram_total) = krea2::cuda::mem_info().unwrap_or((0, 0));
        EngineInfo {
            name: "krea2".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            devices: vec![DeviceInfo { name: "cuda:0".into(), kind: "cuda".into(), index: 0, vram_total: vram_total as u64, vram_free: vram_free as u64 }],
            ram_total: 0,
            ram_free: 0,
            attention_backends: ATTENTION.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn load_checkpoint(&self, req: &LoadCheckpoint) -> EngineResult<(ModelHandle, TextEncoderHandle, VaeHandle)> {
        // SwarmUI picks this loader for anything in its Stable-Diffusion folder, by folder alone.
        // The file holds only the diffusion model; SwarmUI adds its own CLIPLoader and VAELoader.
        check_dit(&req.file)?;
        self.gen().ensure_dit(&req.file.path).map_err(internal)?;
        Ok((
            ModelHandle::new(Some(req.file.clone()), Some("krea-2".into()), Opaque::none()),
            TextEncoderHandle { sources: vec![], encoder_type: "krea2".into(), device: "default".into(), engine: Opaque::none(), patches: vec![] },
            VaeHandle::new(None, Opaque::none()),
        ))
    }

    fn load_diffusion_model(&self, req: &LoadDiffusionModel) -> EngineResult<ModelHandle> {
        check_dit(&req.file)?;
        self.gen().ensure_dit(&req.file.path).map_err(internal)?;
        // `weight_dtype` doesn't apply: the file's int8 weights are used as stored.
        Ok(ModelHandle::new(Some(req.file.clone()), Some("krea-2".into()), Opaque::none()))
    }

    fn load_text_encoder(&self, req: &LoadTextEncoder) -> EngineResult<TextEncoderHandle> {
        let [file] = req.files.as_slice() else {
            return Err(unsupported("dual text encoders"));
        };
        if req.encoder_type != "krea2" {
            return Err(unsupported(format!("text encoder type {} (this backend only runs Krea 2)", req.encoder_type)));
        }
        self.gen().ensure_te(&file.path).map_err(internal)?;
        Ok(TextEncoderHandle { sources: req.files.clone(), encoder_type: req.encoder_type.clone(), device: req.device.clone(), engine: Opaque::none(), patches: vec![] })
    }

    fn load_vae(&self, req: &LoadVae) -> EngineResult<VaeHandle> {
        self.gen().ensure_vae(&req.file.path).map_err(internal)?;
        Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()))
    }

    fn encode_text(&self, req: &TextEncodeRequest) -> EngineResult<Conditioning> {
        let [file] = req.text_encoder.sources.as_slice() else {
            return Err(unsupported("this text encoder (Krea 2 needs one Qwen3-VL file)"));
        };
        let cond = DeferredCond { text_encoder: file.path.clone(), prompt: encode_request(req) };
        Ok(Conditioning::single(Opaque::new(cond), CondMeta::default()))
    }

    fn empty_latent(&self, req: &EmptyLatentRequest) -> EngineResult<Latent> {
        let EmptyLatentKind::Image16Ch { width, height } = req.kind else {
            return Err(unsupported(format!("latent {:?}", req.kind)));
        };
        let (b, h, w) = (req.batch_size as usize, height as usize / 8, width as usize / 8);
        Ok(Latent::single(
            LatentTensor { format: LatentFormat::Image16Ch, shape: vec![b, 16, h, w], data: Opaque::none() },
            PixelInfo::image(width, height, req.batch_size),
        ))
    }

    fn sample(&self, req: &SampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        check_sample(req)?;
        let (text_encoder, prompt) = single_prompt(&req.positive)?;
        let settings = model_settings(&req.model, self.default_shift)?;
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let LatentSamples::Single(tensor) = &req.latent.samples else {
            return Err(unsupported("audio/video latents"));
        };
        let &[batch, 16, lh, lw] = tensor.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", tensor.shape)));
        };
        let (width, height) = (lw * 8, lh * 8);
        let steps = req.steps as usize;

        let mut gen = self.gen();
        gen.ensure_dit(&file.path).map_err(internal)?;
        gen.ensure_te(text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);

        let mut data = Vec::with_capacity(batch * 16 * lh * lw);
        for i in 0..batch {
            let image = noise::swarm_noise(req.noise_seed, req.variation, i as u64, lh, lw);
            let r = Request { prompt: prompt.to_string(), seed: req.noise_seed, width, height, steps, shift: settings.shift, noise: Some(image) };
            let (latent, stats) = gen
                .sample_latent(&r, |step, _| {
                    progress.progress((i * steps + step) as u32, (batch * steps) as u32);
                    !progress.interrupted()
                })
                .map_err(internal)?;
            tracing::info!("krea2: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        Ok(req.latent.with_samples(LatentSamples::Single(LatentTensor {
            format: LatentFormat::Image16Ch,
            shape: tensor.shape.clone(),
            data: Opaque::new(HostLatent { data }),
        })))
    }

    fn vae_decode(&self, req: &VaeDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        if req.tiling.is_some() {
            return Err(unsupported("VAE tiling"));
        }
        if !req.vae.patches.is_empty() {
            return Err(unsupported("Seamless Tileable"));
        }
        let LatentSamples::Single(t) = &req.latent.samples else {
            return Err(unsupported("audio/video latents"));
        };
        let &[batch, 16, lh, lw] = t.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", t.shape)));
        };
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("decoding a latent this engine didn't sample"))?;
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE"))?;
        let (width, height) = (lw * 8, lh * 8);

        let mut gen = self.gen();
        gen.ensure_vae(&file.path).map_err(internal)?;
        let mut rgb = Vec::with_capacity(batch * height * width * 3);
        for (i, image) in latent.data.chunks_exact(16 * lh * lw).enumerate() {
            if progress.interrupted() {
                return Err(EngineError::Interrupted);
            }
            rgb.extend_from_slice(&gen.decode_latent(image, width, height).map_err(internal)?);
            progress.progress(i as u32 + 1, batch as u32);
        }
        Ok(ImageBatch::from_u8(batch as u32, height as u32, width as u32, 3, rgb))
    }

    fn vae_encode(&self, _req: &VaeEncodeRequest) -> EngineResult<Latent> {
        Err(unsupported("VAE encoding (init images, refiners, reference images)"))
    }

    fn vae_decode_audio(&self, _req: &VaeDecodeAudioRequest) -> EngineResult<AudioData> {
        Err(unsupported("audio"))
    }

    fn warmup(&self, req: &WarmupRequest) -> EngineResult<()> {
        let mut gen = self.gen();
        if let Some(f) = &req.model.source {
            gen.ensure_dit(&f.path).map_err(internal)?;
        }
        if let [f] = req.text_encoder.sources.as_slice() {
            gen.ensure_te(&f.path).map_err(internal)?;
        }
        if let Some(f) = &req.vae.source {
            gen.ensure_vae(&f.path).map_err(internal)?;
        }
        Ok(())
    }

    fn free_memory(&self, unload_models: bool, free_memory: bool) {
        let mut gen = self.gen();
        let r = if unload_models {
            gen.unload()
        } else if free_memory {
            gen.trim()
        } else {
            Ok(())
        };
        if let Err(e) = r {
            tracing::warn!("krea2: freeing memory failed: {e:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_prompts_keep_comfy_literal_parentheses() {
        // Qwen3-VL tokenizes with weights disabled, so `(x:1.2)` stays in the text.
        assert_eq!(prompt_text(&ParsedPrompt::parse("a (red:1.2) fox"), 8).unwrap(), "a (red:1.2) fox");
    }

    #[test]
    fn prompt_syntax_the_pipeline_cant_honor_is_refused() {
        for p in ["a <weight[1.3]:fox>", "a<break>b", "<alternate:a, b>", "<fromto[0.5]:a, b>", "<embed:x>"] {
            assert!(prompt_text(&ParsedPrompt::parse(p), 8).is_err(), "{p}");
        }
    }

    #[test]
    fn flux_shift_with_equal_bounds_is_that_shift() {
        assert!((flux_shift(1.15, 1.15, 1024, 1024) - 1.15).abs() < 1e-12);
        assert!((flux_shift(1.15, 1.15, 2048, 512) - 1.15).abs() < 1e-12);
    }

    #[test]
    fn flux_shift_matches_comfy_endpoints() {
        // 2560 and 256 patches are the points ModelSamplingFlux interpolates between.
        assert!((flux_shift(1.15, 0.5, 16 * 256, 16) - 0.5).abs() < 1e-12);
        assert!((flux_shift(1.15, 0.5, 16 * 2560, 16) - 1.15).abs() < 1e-12);
    }
}
