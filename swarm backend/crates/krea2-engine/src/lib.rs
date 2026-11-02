//! [`Engine`] implementation on the krea2 library: text-to-image for int8 ConvRot checkpoints of
//! Krea 2, Anima and MOSS Image, with every ComfyUI sampler and scheduler, any CFG and LoRAs. Each
//! file's architecture is read from its keys, so one backend serves all three.
//!
//! Text encoding is deferred to `sample`, so the negative prompt SwarmUI always sends costs nothing
//! at CFG 1. Everything the pipeline can't do is refused with [`EngineError::Unsupported`], which
//! SwarmUI shows to the user, instead of being ignored and producing a different image.

mod noise;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use krea2::pipeline::{AnimaRequest, Cancelled, Generator, LoadOptions, MossRequest, Request, Sampling};
use krea2::samplers::{self, ModelSampling, SigmaRange};
use krea2::lora::{self, Lora};
use krea2::tokenizer::token_weights;
use swarm_engine_api::prompt::PromptLeafKind;
use swarm_engine_api::*;

/// Attention backends, default first. SwarmUI's "pytorch attention" maps to the first entry.
const ATTENTION: [&str; 2] = ["int8", "flash"];
/// ComfyUI's `sampling_settings`, used when SwarmUI sends no Sigma Shift.
const KREA2_SHIFT: f64 = 1.15;
const ANIMA_SHIFT: f64 = 3.0;
const MOSS_SHIFT: f64 = 5.0;
/// [`TextEncoderHandle::encoder_type`] of Anima's Qwen3 0.6B; SwarmUI's own type is the generic
/// `stable_diffusion`.
const ANIMA_ENCODER: &str = "anima";
const MOSS_ENCODER: &str = "moss_image";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Krea2,
    Anima,
    Moss,
}

impl Family {
    /// (latent channels, pixels per latent)
    fn latent(self) -> (usize, usize) {
        if self == Family::Moss { (krea2::moss::CH, 16) } else { (16, 8) }
    }
}

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
}

/// The prompt to encode, or why this conditioning can't be used. Only checked when a prompt is
/// really needed: at CFG 1 the negative is never read.
struct DeferredCond {
    text_encoder: PathBuf,
    /// (text, weight) segments; Krea 2 and MOSS Image have exactly one, of weight 1.
    prompt: Result<Vec<(String, f32)>, String>,
    /// MOSS Image: the chat template, None for ComfyUI's own.
    template: Option<String>,
}

/// Sampled latents, `[B, 16, H/8, W/8]`, or `[B, 128, H/16, W/16]` for MOSS Image.
struct HostLatent {
    data: Vec<f32>,
}

fn unsupported(what: impl Into<String>) -> EngineError {
    EngineError::Unsupported(what.into())
}

fn internal(e: anyhow::Error) -> EngineError {
    if e.is::<Cancelled>() {
        return EngineError::Interrupted;
    }
    match e.chain().find_map(|c| c.downcast_ref::<lora::Unsupported>()) {
        Some(u) => unsupported(u.0.clone()),
        None => EngineError::Internal(e),
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

/// The file's architecture. Refuses other models and formats before any weights are touched.
fn check_dit(file: &ModelFile) -> EngineResult<Family> {
    let h = read_header(&file.path).map_err(|e| unsupported(format!("{}: not a readable safetensors file ({e})", file.name)))?;
    if let Some(p) = krea2::anima::key_prefix(|k| h.get(k).is_some()) {
        if h.get(format!("{p}blocks.0.mlp.layer1.weight_scale")).is_none() {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot Anima checkpoints", file.name)));
        }
        return Ok(Family::Anima);
    }
    if krea2::moss::is_moss(|k| h.get(k).is_some()) {
        if h.get("layers.0.proj.qkv.weight_scale").is_none() {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot MOSS Image checkpoints with fused q/k/v", file.name)));
        }
        return Ok(Family::Moss);
    }
    // The keys SwarmUI's `isKrea2` tests.
    if ["txtfusion.projector.weight", "blocks.0.attn.gate.weight", "blocks.0.mod.lin"].iter().any(|k| h.get(k).is_none()) {
        return Err(unsupported(format!("{}: this backend only runs Krea 2, Anima and MOSS Image models", file.name)));
    }
    if h.get("blocks.0.attn.gate.weight_scale").is_none() {
        return Err(unsupported(format!("{}: this backend only runs W8A8 (ConvRot int8) Krea 2 checkpoints", file.name)));
    }
    Ok(Family::Krea2)
}

/// Anima's `CLIPLoader` type is the generic `stable_diffusion`; this tells its encoder by shape.
fn is_qwen3_600m(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| h["model.layers.0.input_layernorm.weight"]["shape"][0] == 1024)
}

/// A prompt's (text, weight) segments, or why it needs something the pipeline doesn't have.
fn prompt_segments(p: &ParsedPrompt, steps: u32, family: Family) -> Result<Vec<(String, f32)>, String> {
    if p.has_steps() {
        return Err("<alternate> and <fromto> prompts".into());
    }
    let chunks = p.chunks(0, steps.max(1));
    if chunks.len() != 1 {
        return Err("<break> in prompts".into());
    }
    let mut text = String::new();
    let mut segments = Vec::new();
    for leaf in &chunks[0] {
        if family != Family::Anima && leaf.weight != 1.0 {
            return Err("prompt weights (<weight>)".into());
        }
        match &leaf.kind {
            PromptLeafKind::Text(t) if family == Family::Anima => segments.extend(token_weights(t, leaf.weight as f32)),
            PromptLeafKind::Text(t) => text.push_str(t),
            PromptLeafKind::Embed(_) => return Err("embeddings in prompts".into()),
            PromptLeafKind::LoraHook(_) => return Err("LoRA hooks in prompts".into()),
            PromptLeafKind::Break => unreachable!("chunks() splits on breaks"),
        }
    }
    if family != Family::Anima {
        segments.push((text, 1.0));
    }
    Ok(segments)
}

fn encode_request(req: &TextEncodeRequest, family: Family) -> Result<Vec<(String, f32)>, String> {
    for p in &req.text_encoder.patches {
        match p {
            TextEncoderPatch::Lora { file, .. } => {
                let h = read_header(&file.path).map_err(|e| format!("{}: not a readable safetensors file ({e})", file.name))?;
                let keys = h.as_object().map(|o| o.keys().collect::<Vec<_>>()).unwrap_or_default();
                if lora::has_text_encoder_keys(keys) {
                    return Err(format!("{}: LoRAs that patch the text encoder", file.name));
                }
            }
            TextEncoderPatch::Hooks { .. } => return Err("LoRA hooks".into()),
            TextEncoderPatch::StopAtLayer { .. } => return Err("CLIP Stop At Layer".into()),
        }
    }
    match &req.mode {
        // Qwen tokenizers run with weights disabled, Anima's T5 tokens take ComfyUI's emphasis.
        TextEncodeMode::Plain { text } if family != Family::Anima => Ok(vec![(text.clone(), 1.0)]),
        TextEncodeMode::Plain { text } => Ok(token_weights(text, 1.0)),
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
            prompt_segments(&s.prompt, s.steps, family)
        }
    }
}

/// The one plain prompt of a conditioning: anything regional or image-based is refused.
fn single_prompt(c: &Conditioning) -> EngineResult<(&DeferredCond, &[(String, f32)])> {
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
    let segments = cond.prompt.as_deref().map_err(|e| unsupported(e.clone()))?;
    Ok((cond, segments))
}

/// `ModelSamplingFlux`: `max_shift` applies at 2560 latent patches, `base_shift` at 256.
fn flux_shift(max_shift: f64, base_shift: f64, width: u32, height: u32) -> f64 {
    let mm = (max_shift - base_shift) / (2560.0 - 256.0);
    (width as f64 * height as f64 / (8.0 * 8.0 * 2.0 * 2.0)) * mm + base_shift - mm * 256.0
}

struct ModelSettings {
    model_sampling: ModelSampling,
    int8_attention: bool,
    loras: Vec<Lora>,
}

fn model_settings(model: &ModelHandle, family: Family) -> EngineResult<ModelSettings> {
    let model_sampling = match family {
        Family::Krea2 => ModelSampling::Flux { shift: KREA2_SHIFT },
        Family::Anima => ModelSampling::Flow { shift: ANIMA_SHIFT },
        Family::Moss => ModelSampling::Flow { shift: MOSS_SHIFT },
    };
    let mut s = ModelSettings { model_sampling, int8_attention: true, loras: Vec::new() };
    for p in &model.patches {
        match p {
            ModelPatch::SamplingFlux { max_shift, base_shift, width, height } if family == Family::Krea2 => {
                s.model_sampling = ModelSampling::Flux { shift: flux_shift(*max_shift, *base_shift, *width, *height) }
            }
            ModelPatch::SamplingAuraFlow { shift, sampling } if family != Family::Krea2 && sampling == "flow" => {
                s.model_sampling = ModelSampling::Flow { shift: *shift }
            }
            ModelPatch::AttentionBackend { backend } => {
                s.int8_attention = match backend.as_str() {
                    "int8" => true,
                    "flash" => false,
                    b => return Err(unsupported(format!("attention backend {b}"))),
                }
            }
            // Weighted prompts are refused in encode_text; with none this patch has no effect.
            ModelPatch::AttnTokenWeights { .. } | ModelPatch::Compile { .. } => {}
            ModelPatch::Lora { file, strength } => s.loras.push(Lora { path: file.path.clone(), strength: *strength }),
            other => {
                let name = format!("{other:?}");
                return Err(unsupported(format!("model option {}", name.split([' ', '{', '(']).next().unwrap_or("?"))));
            }
        }
    }
    Ok(s)
}

/// Everything in a sampler request beyond seeded text-to-image is refused here.
fn check_sample(req: &SampleRequest) -> EngineResult<()> {
    if !samplers::SAMPLERS.contains(&req.sampler.as_str()) {
        return Err(unsupported(format!("sampler {}", req.sampler)));
    }
    if !samplers::SCHEDULERS.contains(&req.scheduler.as_str()) {
        return Err(unsupported(format!("scheduler {}", req.scheduler)));
    }
    if req.start_at_step != 0 || req.end_at_step < req.steps || !req.add_noise || req.denoise != 1.0 {
        return Err(unsupported("partial sampling (init images, refiners)"));
    }
    if !req.latent.empty || req.latent.noise_mask.is_some() {
        return Err(unsupported("init images and masks (no VAE encoder yet)"));
    }
    if req.tile_size.is_some() {
        return Err(unsupported("tiled sampling"));
    }
    Ok(())
}

/// The sampling of image `index` of `batch`.
fn sampling(req: &SampleRequest, settings: &ModelSettings, (batch, lh, lw): (usize, usize, usize), index: usize) -> EngineResult<Sampling> {
    let range = req.sigma_min.zip(req.sigma_max).map(|(sigma_min, sigma_max)| SigmaRange { sigma_min, sigma_max, rho: req.rho });
    let sigmas = samplers::sigmas(settings.model_sampling, &req.sampler, &req.scheduler, req.steps as usize, (lw, lh), range).map_err(internal)?;
    Ok(Sampling {
        sampler: req.sampler.clone(),
        model_sampling: settings.model_sampling,
        sigmas,
        cfg: req.cfg,
        seed: req.noise_seed,
        batch_index: index,
        batch_size: batch,
    })
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
        Ok(Self { gen: Mutex::new(Generator::new(&load)?) })
    }

    fn gen(&self) -> std::sync::MutexGuard<'_, Generator> {
        // A panic mid-generation leaves nothing the next prompt depends on.
        self.gen.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load_model(&self, file: &ModelFile) -> EngineResult<ModelHandle> {
        // `weight_dtype` doesn't apply: the file's int8 weights are used as stored.
        let (family, name) = match check_dit(file)? {
            Family::Krea2 => (self.gen().ensure_dit(&file.path), "krea-2"),
            Family::Anima => (self.gen().ensure_anima(&file.path), "anima"),
            Family::Moss => (self.gen().ensure_moss(&file.path), "moss-image"),
        };
        family.map_err(internal)?;
        Ok(ModelHandle::new(Some(file.clone()), Some(name.into()), Opaque::none()))
    }

    fn sample_anima(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let (cond, positive) = single_prompt(&req.positive)?;
        let text_encoder = &cond.text_encoder;
        let first = sampling(req, settings, shape, 0)?;
        let negative = if first.uses_negative() { single_prompt(&req.negative)?.1 } else { &[] };
        let mut gen = self.gen();
        gen.ensure_anima(&file.path).map_err(internal)?;
        gen.ensure_qwen3(text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        let mut data = Vec::with_capacity(batch * 16 * lh * lw);
        for i in 0..batch {
            let r = AnimaRequest {
                positive: positive.to_vec(),
                negative: negative.to_vec(),
                width: lw * 8,
                height: lh * 8,
                sampling: sampling(req, settings, shape, i)?,
                noise: noise::swarm_noise(req.noise_seed, req.variation, i as u64, 16, lh, lw),
            };
            let (latent, stats) = gen
                .sample_anima(&r, |step, steps| {
                    progress.progress((i * steps + step) as u32, (batch * steps) as u32);
                    !progress.interrupted()
                }, None)
                .map_err(internal)?;
            tracing::info!("anima: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        Ok(data)
    }

    fn sample_moss(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let (cond, positive) = single_prompt(&req.positive)?;
        let negative = if sampling(req, settings, shape, 0)?.uses_negative() { single_prompt(&req.negative)?.1[0].0.clone() } else { String::new() };
        let mut gen = self.gen();
        gen.ensure_moss(&file.path).map_err(internal)?;
        gen.ensure_qwen35(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        let mut data = Vec::with_capacity(batch * krea2::moss::CH * lh * lw);
        for i in 0..batch {
            let r = MossRequest {
                positive: positive[0].0.clone(),
                negative: negative.clone(),
                template: cond.template.clone(),
                width: lw * 16,
                height: lh * 16,
                sampling: sampling(req, settings, shape, i)?,
                noise: noise::swarm_noise(req.noise_seed, req.variation, i as u64, krea2::moss::CH, lh, lw),
            };
            let (latent, stats) = gen
                .sample_moss(&r, |step, steps| {
                    progress.progress((i * steps + step) as u32, (batch * steps) as u32);
                    !progress.interrupted()
                }, None)
                .map_err(internal)?;
            tracing::info!("moss: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        Ok(data)
    }
}

/// True for the Flux 2 VAE, which MOSS Image uses.
fn is_flux2_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| h.get("bn.running_mean").is_some())
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
        Ok((
            self.load_model(&req.file)?,
            TextEncoderHandle { sources: vec![], encoder_type: "krea2".into(), device: "default".into(), engine: Opaque::none(), patches: vec![] },
            VaeHandle::new(None, Opaque::none()),
        ))
    }

    fn load_diffusion_model(&self, req: &LoadDiffusionModel) -> EngineResult<ModelHandle> {
        self.load_model(&req.file)
    }

    fn load_text_encoder(&self, req: &LoadTextEncoder) -> EngineResult<TextEncoderHandle> {
        let [file] = req.files.as_slice() else {
            return Err(unsupported("dual text encoders"));
        };
        let encoder_type = match req.encoder_type.as_str() {
            "krea2" => {
                self.gen().ensure_te(&file.path).map_err(internal)?;
                "krea2"
            }
            "stable_diffusion" if is_qwen3_600m(file) => {
                self.gen().ensure_qwen3(&file.path).map_err(internal)?;
                ANIMA_ENCODER
            }
            MOSS_ENCODER => {
                self.gen().ensure_qwen35(&file.path).map_err(internal)?;
                MOSS_ENCODER
            }
            t => return Err(unsupported(format!("text encoder type {t} (this backend only runs Krea 2, Anima and MOSS Image)"))),
        };
        Ok(TextEncoderHandle { sources: req.files.clone(), encoder_type: encoder_type.into(), device: req.device.clone(), engine: Opaque::none(), patches: vec![] })
    }

    fn load_vae(&self, req: &LoadVae) -> EngineResult<VaeHandle> {
        let mut gen = self.gen();
        if is_flux2_vae(&req.file) { gen.ensure_flux2_vae(&req.file.path) } else { gen.ensure_vae(&req.file.path) }.map_err(internal)?;
        Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()))
    }

    fn encode_text(&self, req: &TextEncodeRequest) -> EngineResult<Conditioning> {
        let [file] = req.text_encoder.sources.as_slice() else {
            return Err(unsupported("this text encoder (Krea 2 needs one Qwen3-VL file, Anima one Qwen3 file, MOSS Image one Qwen3.5 file)"));
        };
        let family = match req.text_encoder.encoder_type.as_str() {
            ANIMA_ENCODER => Family::Anima,
            MOSS_ENCODER => Family::Moss,
            _ => Family::Krea2,
        };
        // SwarmUI sends the Krea 2 template for MOSS Image as well.
        let template = match &req.mode {
            TextEncodeMode::Swarm(s) if s.llama_template.as_deref() == Some("krea2") => Some(krea2::te::TEMPLATE.to_string()),
            _ => None,
        };
        let cond = DeferredCond { text_encoder: file.path.clone(), prompt: encode_request(req, family), template };
        Ok(Conditioning::single(Opaque::new(cond), CondMeta::default()))
    }

    fn empty_latent(&self, req: &EmptyLatentRequest) -> EngineResult<Latent> {
        if let EmptyLatentKind::Flux2 { width, height } = req.kind {
            let (b, h, w) = (req.batch_size as usize, height as usize / 16, width as usize / 16);
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::Flux2, shape: vec![b, krea2::moss::CH, h, w], data: Opaque::none() },
                PixelInfo::image(width, height, req.batch_size),
            ));
        }
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
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let family = check_dit(file)?;
        check_sample(req)?;
        let settings = model_settings(&req.model, family)?;
        let LatentSamples::Single(tensor) = &req.latent.samples else {
            return Err(unsupported("audio/video latents"));
        };
        let &[batch, channels, lh, lw] = tensor.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", tensor.shape)));
        };
        if channels != family.latent().0 {
            return Err(unsupported(format!("latent shape {:?} for this model", tensor.shape)));
        }
        let finish = |data| {
            Ok(req.latent.with_samples(LatentSamples::Single(LatentTensor {
                format: tensor.format.clone(),
                shape: tensor.shape.clone(),
                data: Opaque::new(HostLatent { data }),
            })))
        };
        if family == Family::Anima {
            return finish(self.sample_anima(req, (batch, lh, lw), &settings, progress)?);
        }
        if family == Family::Moss {
            return finish(self.sample_moss(req, (batch, lh, lw), &settings, progress)?);
        }
        let (cond, prompt) = single_prompt(&req.positive)?;
        let (text_encoder, prompt) = (&cond.text_encoder, &prompt[0].0);
        let (width, height) = (lw * 8, lh * 8);
        let shape = (batch, lh, lw);
        let negative = if sampling(req, &settings, shape, 0)?.uses_negative() { single_prompt(&req.negative)?.1[0].0.clone() } else { String::new() };

        let mut gen = self.gen();
        gen.ensure_dit(&file.path).map_err(internal)?;
        gen.ensure_te(text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;

        let mut data = Vec::with_capacity(batch * 16 * lh * lw);
        for i in 0..batch {
            let image = noise::swarm_noise(req.noise_seed, req.variation, i as u64, 16, lh, lw);
            let r = Request { prompt: prompt.to_string(), negative: negative.clone(), seed: req.noise_seed, width, height,
                sampling: sampling(req, &settings, shape, i)?, noise: Some(image) };
            let (latent, stats) = gen
                .sample_latent(&r, |step, steps| {
                    progress.progress((i * steps + step) as u32, (batch * steps) as u32);
                    !progress.interrupted()
                })
                .map_err(internal)?;
            tracing::info!("krea2: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        finish(data)
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
        let &[batch, channels, lh, lw] = t.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", t.shape)));
        };
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("decoding a latent this engine didn't sample"))?;
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE"))?;
        let flux2 = is_flux2_vae(file);
        if flux2 != (channels == krea2::moss::CH) {
            return Err(unsupported(format!("this VAE on a {channels}-channel latent")));
        }
        let scale = if flux2 { 16 } else { 8 };
        let (width, height) = (lw * scale, lh * scale);

        let mut gen = self.gen();
        if flux2 { gen.ensure_flux2_vae(&file.path) } else { gen.ensure_vae(&file.path) }.map_err(internal)?;
        let mut rgb = Vec::with_capacity(batch * height * width * 3);
        for (i, image) in latent.data.chunks_exact(channels * lh * lw).enumerate() {
            if progress.interrupted() {
                return Err(EngineError::Interrupted);
            }
            let pixels = if flux2 { gen.decode_moss(image, width, height) } else { gen.decode_latent(image, width, height) };
            rgb.extend_from_slice(&pixels.map_err(internal)?);
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
        if let Some(f) = &req.model.source {
            self.load_model(f)?;
        }
        let mut gen = self.gen();
        if let [f] = req.text_encoder.sources.as_slice() {
            match req.text_encoder.encoder_type.as_str() {
                ANIMA_ENCODER => gen.ensure_qwen3(&f.path),
                MOSS_ENCODER => gen.ensure_qwen35(&f.path),
                _ => gen.ensure_te(&f.path),
            }
            .map_err(internal)?;
        }
        if let Some(f) = &req.vae.source {
            if is_flux2_vae(f) { gen.ensure_flux2_vae(&f.path) } else { gen.ensure_vae(&f.path) }.map_err(internal)?;
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
        let p = ParsedPrompt::parse("a (red:1.2) fox");
        assert_eq!(prompt_segments(&p, 8, Family::Krea2).unwrap(), [("a (red:1.2) fox".to_string(), 1.0)]);
    }

    #[test]
    fn prompt_syntax_the_pipeline_cant_honor_is_refused() {
        for p in ["a <weight[1.3]:fox>", "a<break>b", "<alternate:a, b>", "<fromto[0.5]:a, b>", "<embed:x>"] {
            assert!(prompt_segments(&ParsedPrompt::parse(p), 8, Family::Krea2).is_err(), "{p}");
        }
    }

    #[test]
    fn anima_takes_weights_from_both_syntaxes() {
        let segments = prompt_segments(&ParsedPrompt::parse("lake, <weight[1.3]:pine> (sky:1.2)"), 8, Family::Anima).unwrap();
        let weight = |t: &str| segments.iter().find(|(s, _)| s.contains(t)).unwrap().1;
        assert!((weight("pine") - 1.3).abs() < 1e-6);
        assert!((weight("sky") - 1.2).abs() < 1e-6);
        assert_eq!(weight("lake"), 1.0);
    }

    #[test]
    fn moss_prompts_are_one_literal_segment() {
        let p = ParsedPrompt::parse("a (red:1.2) lake");
        assert_eq!(prompt_segments(&p, 8, Family::Moss).unwrap(), [("a (red:1.2) lake".to_string(), 1.0)]);
        assert!(prompt_segments(&ParsedPrompt::parse("a <weight[1.3]:lake>"), 8, Family::Moss).is_err());
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
