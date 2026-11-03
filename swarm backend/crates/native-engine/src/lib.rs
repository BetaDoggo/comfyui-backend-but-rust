//! [`Engine`] implementation on the diffusion library: text-to-image for int8 ConvRot checkpoints of
//! Krea 2, Anima and MOSS Image and text-to-video with audio for MiniMax H3, with every ComfyUI
//! sampler and scheduler, any CFG and LoRAs, plus Krea 2's prompt images and reference latents. Each
//! file's architecture is read from its keys, so one backend serves them all.
//!
//! Text encoding is deferred to `sample`, so the negative prompt SwarmUI always sends costs nothing
//! at CFG 1. Everything the pipeline can't do is refused with [`EngineError::Unsupported`], which
//! SwarmUI shows to the user, instead of being ignored and producing a different image.

mod noise;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use diffusion::h3;
use diffusion::pipeline::{AnimaRequest, Cancelled, Generator, H3Request, LoadOptions, MossRequest, Prompt, Reference, Request, Sampling, Step};
use diffusion::pixal3d;
use diffusion::preview;
use diffusion::samplers::{self, CfgFunction, CfgTweaks, ModelSampling, SigmaRange};
use diffusion::lora::{self, Lora};
use diffusion::tokenizer::token_weights;
use diffusion::vision::PromptImage;
use swarm_engine_api::prompt::{PromptLeaf, PromptLeafKind};
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
const MINIMAX_ENCODER: &str = "minimax";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Krea2,
    Anima,
    Moss,
    H3,
}

impl Family {
    /// (latent channels, pixels per latent)
    fn latent(self) -> (usize, usize) {
        if self == Family::Moss { (diffusion::moss::CH, 16) } else { (16, 8) }
    }
}

pub struct NativeOptions {
    /// VRAM (MB) to leave untouched for the desktop and other applications.
    pub reserve_vram_mb: usize,
    /// Largest side (pixels, at 1:1) of a generation.
    pub max_side: usize,
    pub threads: usize,
    /// Send latent previews while sampling.
    pub live_previews: bool,
}

impl Default for NativeOptions {
    fn default() -> Self {
        Self { reserve_vram_mb: 512, max_side: 2048, threads: 6, live_previews: false }
    }
}

pub struct NativeEngine {
    gen: Mutex<Generator>,
    live_previews: bool,
    threads: usize,
}

/// The prompt to encode, or why this conditioning can't be used. Only checked when a prompt is
/// really needed: at CFG 1 the negative is never read.
struct DeferredCond {
    text_encoder: PathBuf,
    /// `<break>` chunks of (text, weight) segments; a Krea 2 chunk is one segment of weight 1.
    prompt: Result<Vec<Vec<(String, f32)>>, String>,
    /// Krea 2: prompt images for the vision tower.
    images: Vec<PromptImage>,
    /// `LoraLoader`'s clip half.
    loras: Vec<Lora>,
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
    if let Some(p) = diffusion::anima::key_prefix(|k| h.get(k).is_some()) {
        if h.get(format!("{p}blocks.0.mlp.layer1.weight_scale")).is_none() {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot Anima checkpoints", file.name)));
        }
        return Ok(Family::Anima);
    }
    if diffusion::moss::is_moss(|k| h.get(k).is_some()) {
        if h.get("layers.0.proj.qkv.weight_scale").is_none() {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot MOSS Image checkpoints with fused q/k/v", file.name)));
        }
        return Ok(Family::Moss);
    }
    if h3::is_h3(|k| h.get(k).is_some()) {
        if h.get("blocks.0.attn.qkv_proj.weight_s_rel").is_none() {
            return Err(unsupported(format!("{}: this backend only runs w6a8 MiniMax H3 checkpoints", file.name)));
        }
        return Ok(Family::H3);
    }
    // The keys SwarmUI's `isKrea2` tests.
    if ["txtfusion.projector.weight", "blocks.0.attn.gate.weight", "blocks.0.mod.lin"].iter().any(|k| h.get(k).is_none()) {
        return Err(unsupported(format!("{}: this backend only runs Krea 2, Anima, MOSS Image and MiniMax H3 models", file.name)));
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

/// A prompt's `<break>` chunks of (text, weight) segments, or why it needs something the pipeline
/// doesn't have.
fn prompt_chunks(p: &ParsedPrompt, steps: u32, family: Family) -> Result<Vec<Vec<(String, f32)>>, String> {
    if p.has_steps() {
        return Err("<alternate> and <fromto> prompts".into());
    }
    let chunks = p.chunks(0, steps.max(1));
    if chunks.len() != 1 && matches!(family, Family::Anima | Family::H3) {
        return Err("<break> in prompts".into());
    }
    chunks.iter().map(|leaves| chunk_segments(leaves, family)).collect()
}

fn chunk_segments(leaves: &[PromptLeaf], family: Family) -> Result<Vec<(String, f32)>, String> {
    let mut text = String::new();
    let mut segments = Vec::new();
    for leaf in leaves {
        if family == Family::Krea2 && leaf.weight != 1.0 {
            return Err("prompt weights (<weight>)".into());
        }
        match &leaf.kind {
            PromptLeafKind::Text(t) if family == Family::Anima => segments.extend(token_weights(t, leaf.weight as f32)),
            PromptLeafKind::Text(t) if family == Family::Krea2 => text.push_str(t),
            PromptLeafKind::Text(t) => segments.push((t.clone(), leaf.weight as f32)),
            PromptLeafKind::Embed(_) => return Err("embeddings in prompts".into()),
            PromptLeafKind::LoraHook(_) => return Err("LoRA hooks in prompts".into()),
            PromptLeafKind::Break => unreachable!("chunks() splits on breaks"),
        }
    }
    if family == Family::Krea2 {
        segments.push((text, 1.0));
    }
    Ok(segments)
}

fn encode_request(req: &TextEncodeRequest, family: Family) -> Result<Vec<Vec<(String, f32)>>, String> {
    for p in &req.text_encoder.patches {
        match p {
            TextEncoderPatch::Lora { .. } => {}
            TextEncoderPatch::Hooks { .. } => return Err("LoRA hooks".into()),
            TextEncoderPatch::StopAtLayer { .. } => return Err("CLIP Stop At Layer".into()),
        }
    }
    match &req.mode {
        // Qwen tokenizers run with weights disabled, Anima's T5 tokens take ComfyUI's emphasis.
        TextEncodeMode::Plain { text } if family == Family::Anima => Ok(vec![token_weights(text, 1.0)]),
        // MOSSImageTokenizer trims a plain prompt; SwarmTextEncodeAdvanced's leaves skip that.
        TextEncodeMode::Plain { text } if family == Family::Moss => Ok(vec![vec![(text.trim().to_string(), 1.0)]]),
        TextEncodeMode::Plain { text } => Ok(vec![vec![(text.clone(), 1.0)]]),
        TextEncodeMode::Swarm(s) => {
            if s.images.is_some() && family != Family::Krea2 {
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
            // The node only uses the template with images, and Krea 2's is the one its encoder has.
            if !matches!(s.llama_template.as_deref(), None | Some("krea2")) {
                return Err(format!("prompt template {:?}", s.llama_template));
            }
            let chunks = prompt_chunks(&s.prompt, s.steps, family)?;
            if chunks.len() != 1 && s.images.is_some() {
                return Err("<break> with Prompt Images".into());
            }
            Ok(chunks)
        }
    }
}

/// RGB frames of a batch, with alpha over white as SwarmTextEncodeAdvanced does.
fn prompt_images(b: &ImageBatch) -> Vec<PromptImage> {
    let channels = b.channels as usize;
    (0..b.count).map(|i| {
        let rgb = b.frame_f32(i).chunks_exact(channels).flat_map(|px| {
            let a = if channels == 4 { px[3] } else { 1.0 };
            let c = |v: usize| px[v.min(channels.min(3) - 1)] * a + 1.0 - a;
            [c(0), c(1), c(2)]
        }).collect();
        PromptImage { width: b.width as usize, height: b.height as usize, rgb }
    }).collect()
}

/// The one plain prompt of a conditioning: anything regional or image-based is refused, except
/// that Krea 2 reads reference latents with [`reference_latents`].
fn single_prompt(c: &Conditioning, family: Family) -> EngineResult<(&DeferredCond, Prompt)> {
    let [e] = c.entries.as_slice() else {
        return Err(unsupported("regional prompts (more than one conditioning entry)"));
    };
    let m = &e.meta;
    let plain = m.start_percent == 0.0
        && m.end_percent == 1.0
        && m.mask.is_none()
        && (family == Family::Krea2 || m.reference_latents.is_empty())
        && m.gligen.is_empty()
        && m.inpaint.is_none()
        && m.embedding_ops.is_empty()
        && m.unclip.is_empty()
        && m.controls.is_empty();
    if !plain {
        return Err(unsupported("conditioning features (regions, references, ControlNet, style models, ...)"));
    }
    let cond = e.embedding.downcast_ref::<DeferredCond>().ok_or_else(|| unsupported("conditioning from another text encoder"))?;
    let chunks = cond.prompt.clone().map_err(unsupported)?;
    Ok((cond, Prompt { chunks, scale: if m.zeroed { 0.0 } else { m.multiplier as f32 } }))
}

/// A Krea 2 conditioning's reference latents, and whether they use `index_timestep_zero`. Without a
/// method ComfyUI's model ignores them.
fn reference_latents(c: &Conditioning) -> EngineResult<(Vec<Reference>, bool)> {
    let [e] = c.entries.as_slice() else { return Ok((Vec::new(), false)) };
    let timestep_zero = match e.meta.reference_latents_method.as_deref() {
        None => return Ok((Vec::new(), false)),
        Some("index") => false,
        Some("index_timestep_zero") => true,
        Some(m) => return Err(unsupported(format!("reference latent method {m}"))),
    };
    let mut out = Vec::with_capacity(e.meta.reference_latents.len());
    for l in &e.meta.reference_latents {
        let LatentSamples::Single(t) = &l.samples else {
            return Err(unsupported("audio/video reference latents"));
        };
        let &[1, 16, height, width] = t.shape.as_slice() else {
            return Err(unsupported(format!("reference latent of shape {:?}", t.shape)));
        };
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("a reference latent this engine didn't encode"))?;
        out.push(Reference { latent: latent.data.clone(), height, width });
    }
    Ok((out, timestep_zero))
}

/// The positive and negative prompts share one baked text encoder.
fn check_text_encoder_loras(positive: &DeferredCond, negative: &DeferredCond) -> EngineResult<()> {
    if positive.loras != negative.loras {
        return Err(unsupported("different LoRAs on the positive and negative text encoders"));
    }
    Ok(())
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
    /// MiniMax H3's audio flow shift.
    audio_shift: f64,
    tweaks: CfgTweaks,
}

fn model_settings(model: &ModelHandle, family: Family) -> EngineResult<ModelSettings> {
    let model_sampling = match family {
        Family::Krea2 => ModelSampling::Flux { shift: KREA2_SHIFT },
        Family::Anima => ModelSampling::Flow { shift: ANIMA_SHIFT },
        Family::Moss => ModelSampling::Flow { shift: MOSS_SHIFT },
        Family::H3 => ModelSampling::Flow { shift: h3::SHIFT_VIDEO },
    };
    // H3's attention sink rows make int8's per-channel V scale lossy
    let mut s = ModelSettings { model_sampling, int8_attention: family != Family::H3, loras: Vec::new(), audio_shift: h3::SHIFT_AUDIO, tweaks: CfgTweaks::default() };
    for p in &model.patches {
        match p {
            ModelPatch::SamplingFlux { max_shift, base_shift, width, height } if family == Family::Krea2 => {
                s.model_sampling = ModelSampling::Flux { shift: flux_shift(*max_shift, *base_shift, *width, *height) }
            }
            ModelPatch::SamplingAuraFlow { shift, sampling } if family != Family::Krea2 && sampling == "flow" => {
                s.model_sampling = ModelSampling::Flow { shift: *shift }
            }
            ModelPatch::MiniMaxH3Shift { shift_video, shift_audio } if family == Family::H3 => {
                s.model_sampling = ModelSampling::Flow { shift: *shift_video };
                s.audio_shift = *shift_audio;
            }
            ModelPatch::AttentionBackend { backend } => {
                s.int8_attention = match backend.as_str() {
                    "int8" => true,
                    "flash" => false,
                    b => return Err(unsupported(format!("attention backend {b}"))),
                }
            }
            // H3's latent packs two streams, which ComfyUI's patches read as nested tensors.
            ModelPatch::RescaleCfg { multiplier } if family != Family::H3 => {
                s.tweaks.function = Some(CfgFunction::Rescale { multiplier: *multiplier as f32 })
            }
            ModelPatch::RenormCfg { cfg_trunc, renorm_cfg } if family != Family::H3 => {
                s.tweaks.function = Some(CfgFunction::Renorm { cfg_trunc: *cfg_trunc as f32, renorm_cfg: *renorm_cfg as f32 })
            }
            ModelPatch::CfgZeroStar if family != Family::H3 => s.tweaks.zero_star = true,
            ModelPatch::Tcfg if family != Family::H3 => s.tweaks.tangential = true,
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
        tweaks: settings.tweaks,
    })
}

/// Whether SwarmKSampler's `previews` shows image `index` of `batch`.
fn shows_preview(mode: PreviewMode, index: usize, batch: usize) -> bool {
    match mode {
        PreviewMode::None => false,
        PreviewMode::One => index == 0,
        PreviewMode::Second => index == 1 % batch,
        PreviewMode::Default | PreviewMode::Iterate | PreviewMode::Animate => true,
    }
}

/// Sends a step's progress within the whole batch, and its preview. False once cancelled.
fn report(mode: PreviewMode, progress: &mut dyn ProgressSink, (index, batch): (usize, usize), step: Step, image: impl FnOnce(&[f32]) -> (Vec<u8>, usize, usize)) -> bool {
    progress.progress((index * step.total + step.index) as u32, (batch * step.total) as u32);
    if shows_preview(mode, index, batch) {
        let (rgb, h, w) = image(step.denoised);
        progress.preview(PreviewImage::Rgb(ImageBatch::from_u8(1, h as u32, w as u32, 3, rgb)));
    }
    !progress.interrupted()
}

impl NativeEngine {
    pub fn new(o: &NativeOptions) -> anyhow::Result<Self> {
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
        Ok(Self { gen: Mutex::new(Generator::new(&load)?), live_previews: o.live_previews, threads: o.threads })
    }

    /// SwarmUI always asks for previews, so the server's option decides.
    fn preview_mode(&self, req: &SampleRequest) -> PreviewMode {
        if self.live_previews { req.previews } else { PreviewMode::None }
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
            Family::H3 => (self.gen().ensure_h3(&file.path), "minimax-h3"),
        };
        family.map_err(internal)?;
        Ok(ModelHandle::new(Some(file.clone()), Some(name.into()), Opaque::none()))
    }

    fn sample_anima(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let (cond, positive) = single_prompt(&req.positive, Family::Anima)?;
        let text_encoder = &cond.text_encoder;
        let first = sampling(req, settings, shape, 0)?;
        let negative = if first.uses_negative() {
            let (c, p) = single_prompt(&req.negative, Family::Anima)?;
            check_text_encoder_loras(cond, c)?;
            p
        } else {
            Prompt::new(Vec::new())
        };
        let mut gen = self.gen();
        gen.ensure_anima(&file.path).map_err(internal)?;
        gen.ensure_qwen3(text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;
        let mut data = Vec::with_capacity(batch * 16 * lh * lw);
        for i in 0..batch {
            let r = AnimaRequest {
                positive: positive.clone(),
                negative: negative.clone(),
                width: lw * 8,
                height: lh * 8,
                sampling: sampling(req, settings, shape, i)?,
                noise: noise::swarm_noise(req.noise_seed, req.variation, i as u64, 16, lh, lw),
            };
            let (latent, stats) = gen
                .sample_anima(&r, |step| report(mode, progress, (i, batch), step, |x0| (preview::wan21(x0, lh, lw), lh, lw)), None)
                .map_err(internal)?;
            tracing::info!("anima: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        Ok(data)
    }

    /// A text-to-video clip: returns the video and audio latents.
    fn sample_h3(&self, req: &SampleRequest, settings: &ModelSettings, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let LatentSamples::AudioVideo { video, audio } = &req.latent.samples else {
            return Err(unsupported("MiniMax H3 needs its joint audio/video latent"));
        };
        let (&[1, 24, t, lh, lw], &[1, 32, 2, audio_t]) = (video.shape.as_slice(), audio.shape.as_slice()) else {
            return Err(unsupported(format!("MiniMax H3 latents of shape {:?} and {:?}", video.shape, audio.shape)));
        };
        if t < 2 || lh % 2 != 0 || lw % 2 != 0 {
            return Err(unsupported("MiniMax H3 needs at least 2 video frames and a width and height in multiples of 32"));
        }
        if req.positive.entries.iter().any(|e| !e.meta.minimax_keyframes.is_empty()) {
            return Err(unsupported("first and last frame conditioning on MiniMax H3 (no VAE encoder yet)"));
        }
        let (cond, prompt) = single_prompt(&req.positive, Family::H3)?;
        let shape = h3::Shape { t, h: lh, w: lw, audio_t };
        let sampling = sampling(req, settings, (1, lh, lw), 0)?;
        let negative = if sampling.uses_negative() { single_prompt(&req.negative, Family::H3)?.1 } else { Prompt::new(Vec::new()) };
        let ModelSampling::Flow { shift } = settings.model_sampling else { unreachable!("H3 is a flow model") };
        let mut gen = self.gen();
        gen.ensure_h3(&file.path).map_err(internal)?;
        gen.ensure_h3_te(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        let noise = noise::h3_noise(req.noise_seed, req.variation, (24, t, lh, lw), shape.audio_len());
        let r = H3Request { prompt, negative, shape, sampling, shifts: (shift, settings.audio_shift), noise };
        let (mut latent, stats) = gen
            .sample_h3(&r, |step| report(mode, progress, (0, 1), step, |x0| (preview::h3(x0, t, lh, lw), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("h3: text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        let audio_data = latent.split_off(shape.video_len());
        Ok(req.latent.with_samples(LatentSamples::AudioVideo {
            video: LatentTensor { format: video.format, shape: video.shape.clone(), data: Opaque::new(HostLatent { data: latent }) },
            audio: LatentTensor { format: audio.format, shape: audio.shape.clone(), data: Opaque::new(HostLatent { data: audio_data }) },
        }))
    }

    fn decode_h3(&self, req: &VaeDecodeRequest, t: &LatentTensor) -> EngineResult<ImageBatch> {
        if req.tiling.as_ref().is_some_and(|v| v.tile_size != 256 || v.overlap != 64) {
            return Err(unsupported("VAE tile sizes other than 256 / 64 on MiniMax H3"));
        }
        let &[1, 24, lt, lh, lw] = t.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", t.shape)));
        };
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("decoding a latent this engine didn't sample"))?;
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE"))?;
        if !is_h3_vae(file) {
            return Err(unsupported("this VAE on a MiniMax H3 latent"));
        }
        let frames = (lt - 2) / 5 * 17 + 5;
        let shape = h3::Shape { t: lt, h: lh, w: lw, audio_t: 0 };
        let clip = self.gen().decode_h3(&file.path, &latent.data, shape, frames).map_err(internal)?;
        let (h, w) = (clip.h, clip.w);
        let mut rgb = vec![0u8; frames * h * w * 3];
        for c in 0..3 {
            for f in 0..frames {
                for i in 0..h * w {
                    rgb[(f * h * w + i) * 3 + c] = (clip.data[((c * frames + f) * h) * w + i] * 255.0).round() as u8;
                }
            }
        }
        Ok(ImageBatch::from_u8(frames as u32, h as u32, w as u32, 3, rgb))
    }

    fn sample_moss(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let (cond, positive) = single_prompt(&req.positive, Family::Moss)?;
        let negative = if sampling(req, settings, shape, 0)?.uses_negative() {
            let (c, p) = single_prompt(&req.negative, Family::Moss)?;
            check_text_encoder_loras(cond, c)?;
            p
        } else {
            Prompt::text("")
        };
        let mut gen = self.gen();
        gen.ensure_moss(&file.path).map_err(internal)?;
        gen.ensure_qwen35(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;
        let mut data = Vec::with_capacity(batch * diffusion::moss::CH * lh * lw);
        for i in 0..batch {
            let r = MossRequest {
                positive: positive.clone(),
                negative: negative.clone(),
                width: lw * 16,
                height: lh * 16,
                sampling: sampling(req, settings, shape, i)?,
                noise: noise::swarm_noise(req.noise_seed, req.variation, i as u64, diffusion::moss::CH, lh, lw),
            };
            let (latent, stats) = gen
                .sample_moss(&r, |step| report(mode, progress, (i, batch), step, |x0| (preview::flux2(x0, lh, lw), lh * 2, lw * 2)), None)
                .map_err(internal)?;
            tracing::info!("moss: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        Ok(data)
    }
}

/// Checks that each Pixal3D file is what its slot needs, before anything is unloaded for it.
fn check_pixal3d(req: &ImageTo3DRequest) -> EngineResult<()> {
    let has = |f: &ModelFile, what: &str, test: fn(&dyn Fn(&str) -> bool) -> bool| -> EngineResult<()> {
        let h = read_header(&f.path).map_err(|e| unsupported(format!("{}: not a readable safetensors file ({e})", f.name)))?;
        if test(&|k| h.get(k).is_some()) { Ok(()) } else { Err(unsupported(format!("{}: not {what}", f.name))) }
    };
    has(&req.diffusion_model, "a Pixal3D diffusion model", |h| diffusion::trellis::is_pixal3d(h))?;
    has(&req.clip_vision, "DINOv3 ViT-L with NAF weights (dino_v3_L_naf)", |h| diffusion::dino3::is_dino3_naf(h))?;
    has(&req.shape_vae, "the Trellis 2 shape VAE", |h| diffusion::trellis_vae::is_shape_vae(h))?;
    has(&req.texture_vae, "the Trellis 2 texture VAE", |h| diffusion::trellis_vae::is_texture_vae(h))?;
    if req.diffusion_model.path.to_string_lossy().contains("multiview") {
        return Err(unsupported("Pixal3D's multi-view checkpoint (this node takes one view)"));
    }
    Ok(())
}

/// The first image as f32 RGB and its object mask: the given one (resized views aren't supported), else alpha.
fn pixal3d_inputs(req: &ImageTo3DRequest) -> EngineResult<(Vec<f32>, Vec<f32>, usize, usize)> {
    let img = &req.image;
    let (w, h) = (img.width as usize, img.height as usize);
    let frame = img.frame_f32(0);
    let c = img.channels as usize;
    if c < 3 {
        return Err(unsupported("a greyscale image for Pixal3D"));
    }
    let rgb: Vec<f32> = frame.chunks_exact(c).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let mask = match &req.mask {
        Some(m) if (m.width as usize, m.height as usize) == (w, h) => m.data[..w * h].to_vec(),
        Some(m) => return Err(unsupported(format!("a {}x{} mask on a {w}x{h} image", m.width, m.height))),
        None if c == 4 => frame.chunks_exact(4).map(|p| p[3]).collect(),
        None => return Err(EngineError::InvalidInput("Pixal3D needs the object cut out: an image with transparency, or a mask".into())),
    };
    Ok((rgb, mask, w, h))
}

/// True for MiniMax H3's video or audio VAE, which load when they decode.
fn is_h3_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| h.get("decoder.transformer_blocks.0.scale1").is_some() || h.get("pre_block.attn.zero_k_bias").is_some())
}

/// True for the Flux 2 VAE, which MOSS Image uses.
fn is_flux2_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| h.get("bn.running_mean").is_some())
}

impl Engine for NativeEngine {
    fn info(&self) -> EngineInfo {
        // Runs on server threads, possibly mid-generation: no generator lock here.
        let (vram_free, vram_total) = diffusion::cuda::mem_info().unwrap_or((0, 0));
        EngineInfo {
            name: "native".into(),
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
            MINIMAX_ENCODER => {
                self.gen().ensure_h3_te(&file.path).map_err(internal)?;
                MINIMAX_ENCODER
            }
            t => return Err(unsupported(format!("text encoder type {t} (this backend only runs Krea 2, Anima, MOSS Image and MiniMax H3)"))),
        };
        Ok(TextEncoderHandle { sources: req.files.clone(), encoder_type: encoder_type.into(), device: req.device.clone(), engine: Opaque::none(), patches: vec![] })
    }

    fn load_vae(&self, req: &LoadVae) -> EngineResult<VaeHandle> {
        if is_h3_vae(&req.file) {
            return Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()));
        }
        let mut gen = self.gen();
        if is_flux2_vae(&req.file) { gen.ensure_flux2_vae(&req.file.path) } else { gen.ensure_vae(&req.file.path) }.map_err(internal)?;
        Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()))
    }

    fn encode_text(&self, req: &TextEncodeRequest) -> EngineResult<Conditioning> {
        let [file] = req.text_encoder.sources.as_slice() else {
            return Err(unsupported("this text encoder (Krea 2 needs one Qwen3-VL file, Anima one Qwen3 file, MOSS Image one Qwen3.5 file, MiniMax H3 one Qwen3-VL 32B file)"));
        };
        let family = match req.text_encoder.encoder_type.as_str() {
            ANIMA_ENCODER => Family::Anima,
            MOSS_ENCODER => Family::Moss,
            MINIMAX_ENCODER => Family::H3,
            _ => Family::Krea2,
        };
        let images = match &req.mode {
            TextEncodeMode::Swarm(s) if family == Family::Krea2 => s.images.as_ref().map(prompt_images).unwrap_or_default(),
            _ => Vec::new(),
        };
        let loras = req.text_encoder.patches.iter().filter_map(|p| match p {
            TextEncoderPatch::Lora { file, strength } => Some(Lora { path: file.path.clone(), strength: *strength }),
            _ => None,
        }).collect();
        let cond = DeferredCond { text_encoder: file.path.clone(), prompt: encode_request(req, family), images, loras };
        Ok(Conditioning::single(Opaque::new(cond), CondMeta::default()))
    }

    fn empty_latent(&self, req: &EmptyLatentRequest) -> EngineResult<Latent> {
        if let EmptyLatentKind::MiniMaxH3AudioVideo { width, height, length } = req.kind {
            let (frames, t, audio_t) = requests::minimax_h3::temporal_shape(length);
            let (h, w) = (height as usize / 16, width as usize / 16);
            let tensor = |format, shape| LatentTensor { format, shape, data: Opaque::none() };
            let samples = LatentSamples::AudioVideo {
                video: tensor(LatentFormat::MiniMaxH3Video, vec![1, 24, t as usize, h, w]),
                audio: tensor(LatentFormat::MiniMaxH3Audio, vec![1, 32, 2, audio_t as usize]),
            };
            let pixel = PixelInfo { frames, fps: Some(24.0), audio_seconds: Some(audio_t as f64 / 40.0), ..PixelInfo::image(width, height, 1) };
            return Ok(Latent { samples, pixel, noise_mask: None, audio_noise_mask: None, batch_index: None, empty: false });
        }
        if let EmptyLatentKind::Flux2 { width, height } = req.kind {
            let (b, h, w) = (req.batch_size as usize, height as usize / 16, width as usize / 16);
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::Flux2, shape: vec![b, diffusion::moss::CH, h, w], data: Opaque::none() },
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
        if family == Family::H3 {
            return self.sample_h3(req, &settings, progress);
        }
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
        let (cond, prompt) = single_prompt(&req.positive, family)?;
        let text_encoder = &cond.text_encoder;
        let (references, timestep_zero) = reference_latents(&req.positive)?;
        let (width, height) = (lw * 8, lh * 8);
        let shape = (batch, lh, lw);
        let (negative, negative_images) = if sampling(req, &settings, shape, 0)?.uses_negative() {
            let (c, p) = single_prompt(&req.negative, family)?;
            check_text_encoder_loras(cond, c)?;
            if !reference_latents(&req.negative)?.0.is_empty() {
                return Err(unsupported("reference latents on the negative prompt"));
            }
            (p, c.images.clone())
        } else {
            (Prompt::text(""), Vec::new())
        };

        let mode = self.preview_mode(req);
        let mut gen = self.gen();
        gen.ensure_dit(&file.path).map_err(internal)?;
        gen.ensure_te(text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;

        let mut data = Vec::with_capacity(batch * 16 * lh * lw);
        for i in 0..batch {
            let image = noise::swarm_noise(req.noise_seed, req.variation, i as u64, 16, lh, lw);
            let r = Request { prompt: prompt.clone(), negative: negative.clone(), seed: req.noise_seed, width, height,
                sampling: sampling(req, &settings, shape, i)?, noise: Some(image), images: cond.images.clone(),
                negative_images: negative_images.clone(), references: references.clone(), timestep_zero };
            let (latent, stats) = gen
                .sample_latent(&r, |step| report(mode, progress, (i, batch), step, |x0| (preview::wan21(x0, lh, lw), lh, lw)))
                .map_err(internal)?;
            tracing::info!("native: image {}/{batch}: text {:.0} ms, sampling {:.0} ms", i + 1, stats.text_ms, stats.sample_ms);
            data.extend_from_slice(&latent);
        }
        finish(data)
    }

    fn vae_decode(&self, req: &VaeDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        match &req.latent.samples {
            LatentSamples::AudioVideo { video, .. } => return self.decode_h3(req, video),
            LatentSamples::Single(t) if t.format == LatentFormat::MiniMaxH3Video => return self.decode_h3(req, t),
            _ => {}
        }
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
        if flux2 != (channels == diffusion::moss::CH) {
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

    fn vae_encode(&self, req: &VaeEncodeRequest) -> EngineResult<Latent> {
        if req.tiling.is_some() {
            return Err(unsupported("VAE tiling"));
        }
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE"))?;
        if is_flux2_vae(file) {
            return Err(unsupported("VAE encoding with the Flux 2 VAE"));
        }
        if is_h3_vae(file) {
            return Err(unsupported("VAE encoding with MiniMax H3's VAE"));
        }
        let mut gen = self.gen();
        gen.ensure_vae(&file.path).map_err(internal)?;
        let frames = prompt_images(&req.pixels);
        let (mut data, mut size) = (Vec::new(), (0, 0));
        for f in &frames {
            let r = gen.encode_image(&f.rgb, f.width, f.height).map_err(internal)?;
            data.extend_from_slice(&r.latent);
            size = (r.height, r.width);
        }
        let (lh, lw) = size;
        let tensor = LatentTensor { format: LatentFormat::Image16Ch, shape: vec![frames.len(), 16, lh, lw], data: Opaque::new(HostLatent { data }) };
        Ok(Latent::single(tensor, PixelInfo::image(lw as u32 * 8, lh as u32 * 8, frames.len() as u32)))
    }

    fn vae_decode_audio(&self, req: &VaeDecodeAudioRequest) -> EngineResult<AudioData> {
        let t = match &req.latent.samples {
            LatentSamples::AudioVideo { audio, .. } => audio,
            LatentSamples::Single(t) if t.format == LatentFormat::MiniMaxH3Audio => t,
            _ => return Err(unsupported("audio latents of other models")),
        };
        let &[1, 32, 2, audio_t] = t.shape.as_slice() else {
            return Err(unsupported(format!("audio latent shape {:?}", t.shape)));
        };
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("decoding a latent this engine didn't sample"))?;
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("audio without an audio VAE"))?;
        let mut wave = self.gen().decode_h3_audio(&file.path, &latent.data, audio_t).map_err(internal)?;
        let right = wave.split_off(wave.len() / 2);
        Ok(AudioData { sample_rate: 32000, channels: vec![Arc::new(wave), Arc::new(right)] })
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
                MINIMAX_ENCODER => gen.ensure_h3_te(&f.path),
                _ => gen.ensure_te(&f.path),
            }
            .map_err(internal)?;
        }
        if let Some(f) = req.vae.source.as_ref().filter(|f| !is_h3_vae(f)) {
            if is_flux2_vae(f) { gen.ensure_flux2_vae(&f.path) } else { gen.ensure_vae(&f.path) }.map_err(internal)?;
        }
        Ok(())
    }

    fn idle(&self) {
        if let Err(e) = self.gen().idle() {
            tracing::warn!("native: releasing VRAM failed: {e:#}");
        }
    }

    fn background_mask(&self, model: &AddonHandle, images: &ImageBatch) -> EngineResult<Mask> {
        let file = &model.source;
        let header = read_header(&file.path).map_err(|e| unsupported(format!("{}: not a readable safetensors file ({e})", file.name)))?;
        if !diffusion::birefnet::is_birefnet(|k| header.get(k).is_some()) {
            return Err(unsupported(format!("{}: not a BiRefNet background removal model", file.name)));
        }
        let c = images.channels as usize;
        if c < 3 {
            return Err(unsupported("background removal of a greyscale image"));
        }
        let frames: Vec<Vec<f32>> =
            (0..images.count).map(|f| images.frame_f32(f).chunks_exact(c).flat_map(|p| [p[0], p[1], p[2]]).collect()).collect();
        // the model takes what the GPU has free: the image models go, and load again when next used
        let mut gen = self.gen();
        gen.unload().map_err(internal)?;
        let masks = diffusion::birefnet::remove_backgrounds(&file.path, &frames, images.width as usize, images.height as usize).map_err(internal)?;
        drop(gen);
        Ok(Mask { count: images.count, height: images.height, width: images.width, data: Arc::new(masks.concat()) })
    }

    fn image_to_3d(&self, req: &ImageTo3DRequest, progress: &mut dyn ProgressSink) -> EngineResult<Vec<u8>> {
        check_pixal3d(req)?;
        let (rgb, mask, w, h) = pixal3d_inputs(req)?;
        if !(1024..=2048).contains(&req.resolution) || req.resolution % 128 != 0 {
            return Err(EngineError::InvalidInput("resolution must be 1024 to 2048 in steps of 128".into()));
        }
        let mut p = pixal3d::Params::example(req.seed);
        p.fov = req.fov;
        p.resolution = req.resolution as usize;
        (p.structure.steps, p.shape.steps, p.refine.steps, p.texture.steps) =
            (req.structure_steps as usize, req.shape_steps as usize, req.refine_steps as usize, req.texture_steps as usize);
        (p.structure.cfg, p.shape.cfg, p.refine.cfg) = (req.cfg, req.cfg, req.cfg);
        p.texture_enabled = req.texture;
        p.max_faces = req.max_faces as usize;
        let paths = pixal3d::Paths {
            dit: req.diffusion_model.path.clone(),
            dino: req.clip_vision.path.clone(),
            shape_vae: req.shape_vae.path.clone(),
            texture_vae: req.texture_vae.path.clone(),
        };
        // the pipeline needs the whole GPU: the image models go, and load again when next used
        let mut gen = self.gen();
        gen.unload().map_err(internal)?;
        let stages = if req.texture { 6 } else { 4 };
        let mut stage_index = 0;
        let mut last = String::new();
        let mut report = |stage: &str, i: usize, n: usize| -> anyhow::Result<()> {
            if stage != last {
                if !last.is_empty() {
                    stage_index += 1;
                }
                last = stage.to_string();
                tracing::info!("pixal3d: {stage}");
            }
            // one bar across the stages, each stage an equal share
            let frac = (stage_index.min(stages - 1) as f64 + i as f64 / n.max(1) as f64) / stages as f64;
            progress.progress((frac * 1000.0) as u32, 1000);
            if progress.interrupted() { Err(Cancelled.into()) } else { Ok(()) }
        };
        let mesh = pixal3d::run(&paths, &rgb, &mask, w, h, &p, self.threads, &mut report).map_err(internal)?;
        drop(gen);
        Ok(pixal3d::glb(&mesh))
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
            tracing::warn!("native: freeing memory failed: {e:#}");
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
        assert_eq!(prompt_chunks(&p, 8, Family::Krea2).unwrap(), [[("a (red:1.2) fox".to_string(), 1.0)]]);
    }

    #[test]
    fn prompt_syntax_the_pipeline_cant_honor_is_refused() {
        for p in ["a <weight[1.3]:fox>", "<alternate:a, b>", "<fromto[0.5]:a, b>", "<embed:x>"] {
            assert!(prompt_chunks(&ParsedPrompt::parse(p), 8, Family::Krea2).is_err(), "{p}");
        }
        for family in [Family::Anima, Family::H3] {
            assert!(prompt_chunks(&ParsedPrompt::parse("a<break>b"), 8, family).is_err());
        }
    }

    #[test]
    fn anima_takes_weights_from_both_syntaxes() {
        let chunks = prompt_chunks(&ParsedPrompt::parse("lake, <weight[1.3]:pine> (sky:1.2)"), 8, Family::Anima).unwrap();
        let weight = |t: &str| chunks[0].iter().find(|(s, _)| s.contains(t)).unwrap().1;
        assert!((weight("pine") - 1.3).abs() < 1e-6);
        assert!((weight("sky") - 1.2).abs() < 1e-6);
        assert_eq!(weight("lake"), 1.0);
    }

    #[test]
    fn moss_and_h3_weight_segments_and_keep_emphasis_literal() {
        for family in [Family::Moss, Family::H3] {
            let chunks = prompt_chunks(&ParsedPrompt::parse("a (red:1.2) <weight[1.5]:lake>"), 8, family).unwrap();
            assert_eq!(chunks, [[("a (red:1.2) ".to_string(), 1.0), ("lake".to_string(), 1.5)]]);
        }
    }

    #[test]
    fn breaks_become_chunks_on_krea2_and_moss() {
        for family in [Family::Krea2, Family::Moss] {
            let chunks = prompt_chunks(&ParsedPrompt::parse("a<break>b"), 8, family).unwrap();
            assert_eq!(chunks, [[("a".to_string(), 1.0)], [("b".to_string(), 1.0)]]);
        }
    }

    #[test]
    fn preview_modes_pick_images_like_swarm() {
        assert!(!shows_preview(PreviewMode::None, 0, 1));
        assert!(shows_preview(PreviewMode::One, 0, 2) && !shows_preview(PreviewMode::One, 1, 2));
        assert!(shows_preview(PreviewMode::Second, 1, 2) && shows_preview(PreviewMode::Second, 0, 1));
        assert!(shows_preview(PreviewMode::Default, 1, 2));
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
