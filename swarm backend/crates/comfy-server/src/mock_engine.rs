//! A stand-in [`Engine`] with no inference. It logs every typed request it receives
//! (so you can watch exactly what SwarmUI's inputs turn into), reports realistic
//! progress and previews, and returns deterministic placeholder pixels/audio of the
//! correct shape so the full SwarmUI round trip works.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use swarm_engine_api::requests::minimax_h3;
use swarm_engine_api::*;

pub struct MockEngine {
    step_delay: Duration,
    /// Claim every backlog node with a schema (see `Engine::supports_custom_node`).
    custom_nodes: bool,
}

impl MockEngine {
    pub fn new(step_delay_ms: u64) -> Self {
        Self { step_delay: Duration::from_millis(step_delay_ms), custom_nodes: true }
    }

    /// A mock that only answers the natively implemented nodes, like an engine that doesn't
    /// implement `custom_node` (what `/object_info` dumps for the backlog tools need).
    pub fn native_only(step_delay_ms: u64) -> Self {
        Self { custom_nodes: false, ..Self::new(step_delay_ms) }
    }
}

struct MockWeights {
    #[allow(dead_code)]
    name: String,
}

/// Stands in for "this VAE decodes to RGBA", which a real engine reads from the weights
/// (Qwen Image 2.1's decoder head has 4 output channels).
struct MockVae {
    rgba: bool,
}

/// Real latent layout per architecture, for conforming SwarmUI's placeholder empty latents.
fn arch_latent(arch: Option<&str>) -> Option<(LatentFormat, usize, u32)> {
    match arch? {
        "qwen-image-2.1" => Some((LatentFormat::QwenImage21, 64, 16)),
        "moss-image" => Some((LatentFormat::Flux2, 128, 16)),
        "krea-2" | "anima" => Some((LatentFormat::Image16Ch, 16, 8)),
        _ => None,
    }
}

/// What a mock latent "contains": enough to make decoded output vary with the inputs.
#[derive(Clone, Copy)]
struct MockLatent {
    seed: u64,
    cond: u64,
}

struct MockCond {
    hash: u64,
}

/// The ComfyUI type of a generic-node value.
fn node_value_type(v: &NodeValue) -> &str {
    match v {
        NodeValue::Model(_) => "MODEL",
        NodeValue::Clip(_) => "CLIP",
        NodeValue::Vae(_) => "VAE",
        NodeValue::Conditioning(_) => "CONDITIONING",
        NodeValue::Latent(_) => "LATENT",
        NodeValue::Image(_) => "IMAGE",
        NodeValue::Mask(_) => "MASK",
        NodeValue::Audio(_) => "AUDIO",
        NodeValue::Video(_) => "VIDEO",
        NodeValue::Addon(a) => a.kind.comfy_type(),
        NodeValue::ClipVisionOutput(_) => "CLIP_VISION_OUTPUT",
        NodeValue::File(_) => "FILE",
        NodeValue::Int(_) => "INT",
        NodeValue::Float(_) => "FLOAT",
        NodeValue::Str(_) => "STRING",
        NodeValue::Bool(_) => "BOOLEAN",
        NodeValue::Other { ty, .. } => ty,
    }
}

fn addon_kind(ty: &str) -> Option<AddonKind> {
    [AddonKind::Gligen, AddonKind::UpscaleModel, AddonKind::LatentUpscaleModel, AddonKind::ClipVision, AddonKind::ControlNet, AddonKind::StyleModel, AddonKind::ModelPatch]
        .into_iter()
        .find(|k| k.comfy_type() == ty)
}

fn hash_of(v: impl Hash) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// Guesses the architecture from a filename, the way a real engine would from tensor keys.
fn guess_arch(name: &str) -> Option<String> {
    let n = name.to_ascii_lowercase();
    if n.contains("krea") {
        Some("krea-2".into())
    } else if n.contains("anima") {
        Some("anima".into())
    } else if n.contains("minimax") || n.contains("h3") {
        Some("minimax-h3".into())
    } else if n.contains("moss") {
        Some("moss-image".into())
    } else if n.contains("qwen") && (n.contains("2.1") || n.contains("2_1") || n.contains("21")) {
        Some("qwen-image-2.1".into())
    } else {
        None
    }
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s));
    match i as i32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// A diagonal two-color gradient whose hues come from `key`; `phase` animates it.
fn render(width: u32, height: u32, count: u32, key: u64, animate: bool) -> ImageBatch {
    let h1 = (key % 360) as f32 / 360.0;
    let h2 = (h1 + 0.35) % 1.0;
    let (c1, c2) = (hsv(h1, 0.7, 0.95), hsv(h2, 0.8, 0.45));
    let mut data = Vec::with_capacity((count * width * height * 3) as usize);
    for f in 0..count {
        let phase = if animate && count > 1 { f as f32 / count as f32 } else { 0.0 };
        for y in 0..height {
            for x in 0..width {
                let t = ((x as f32 / width.max(1) as f32 + y as f32 / height.max(1) as f32) * 0.5 + phase).fract();
                let t = (t * std::f32::consts::TAU).sin() * 0.5 + 0.5;
                for c in 0..3 {
                    data.push(((c1[c] * t + c2[c] * (1.0 - t)) * 255.0) as u8);
                }
            }
        }
    }
    ImageBatch::from_u8(count, height, width, 3, data)
}

fn latent_batch(l: &Latent) -> usize {
    match &l.samples {
        LatentSamples::Single(t) => t.shape.first().copied().unwrap_or(1),
        LatentSamples::AudioVideo { video, .. } => video.shape.first().copied().unwrap_or(1),
    }
}

fn latent_shape(l: &Latent) -> Vec<usize> {
    match &l.samples {
        LatentSamples::Single(t) => t.shape.clone(),
        LatentSamples::AudioVideo { video, .. } => video.shape.clone(),
    }
}

fn latent_key(l: &Latent) -> u64 {
    let tensor = match &l.samples {
        LatentSamples::Single(t) => t,
        LatentSamples::AudioVideo { video, .. } => video,
    };
    tensor.data.downcast_ref::<MockLatent>().map(|m| hash_of((m.seed, m.cond))).unwrap_or(0)
}

impl Engine for MockEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "mock-engine".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            devices: vec![DeviceInfo { name: "mock (no inference)".into(), kind: "cpu".into(), index: 0, vram_total: 0, vram_free: 0 }],
            ram_total: 0,
            ram_free: 0,
            attention_backends: vec!["default".into(), "flash".into()],
        }
    }

    fn load_checkpoint(&self, req: &LoadCheckpoint) -> EngineResult<(ModelHandle, TextEncoderHandle, VaeHandle)> {
        tracing::info!("[mock] load_checkpoint {:?}", req.file.name);
        let w = || Opaque::new(MockWeights { name: req.file.name.clone() });
        Ok((
            ModelHandle::new(Some(req.file.clone()), guess_arch(&req.file.name), w()),
            TextEncoderHandle { sources: vec![req.file.clone()], encoder_type: "checkpoint".into(), device: "default".into(), engine: w(), patches: vec![] },
            VaeHandle::new(Some(req.file.clone()), w()),
        ))
    }

    fn load_diffusion_model(&self, req: &LoadDiffusionModel) -> EngineResult<ModelHandle> {
        tracing::info!("[mock] load_diffusion_model {:?} dtype={:?} arch={:?}", req.file.name, req.weight_dtype, guess_arch(&req.file.name));
        Ok(ModelHandle::new(Some(req.file.clone()), guess_arch(&req.file.name), Opaque::new(MockWeights { name: req.file.name.clone() })))
    }

    fn load_text_encoder(&self, req: &LoadTextEncoder) -> EngineResult<TextEncoderHandle> {
        tracing::info!("[mock] load_text_encoder {:?} type={} device={}", req.files.iter().map(|f| &f.name).collect::<Vec<_>>(), req.encoder_type, req.device);
        Ok(TextEncoderHandle { sources: req.files.clone(), encoder_type: req.encoder_type.clone(), device: req.device.clone(), engine: Opaque::none(), patches: vec![] })
    }

    fn load_vae(&self, req: &LoadVae) -> EngineResult<VaeHandle> {
        let rgba = guess_arch(&req.file.name).as_deref() == Some("qwen-image-2.1");
        tracing::info!("[mock] load_vae {:?} rgba={rgba}", req.file.name);
        Ok(VaeHandle::new(Some(req.file.clone()), Opaque::new(MockVae { rgba })))
    }

    fn encode_text(&self, req: &TextEncodeRequest) -> EngineResult<Conditioning> {
        let (key, guidance) = match &req.mode {
            TextEncodeMode::Plain { text } => {
                tracing::info!("[mock] encode_text plain type={} text={text:?}", req.text_encoder.encoder_type);
                (hash_of(text), None)
            }
            TextEncodeMode::Swarm(s) => {
                tracing::info!(
                    "[mock] encode_text swarm type={} template={:?} prompt={:?} steps={} size={}x{} images={:?} minimax_refs={} hooks={}",
                    req.text_encoder.encoder_type,
                    s.llama_template,
                    s.prompt.raw,
                    s.steps,
                    s.width,
                    s.height,
                    s.images,
                    s.minimax_refs.as_ref().map(|r| format!("{} img / {} vid / {} aud", r.images.len(), r.videos.len(), r.audios.len())).unwrap_or_else(|| "none".into()),
                    s.lora_hooks.as_ref().map(|h| h.len()).unwrap_or(0),
                );
                (hash_of(&s.prompt.raw), s.guidance)
            }
        };
        if !req.text_encoder.patches.is_empty() {
            tracing::info!("[mock]   text encoder patches: {:?}", req.text_encoder.patches);
        }
        Ok(Conditioning::single(Opaque::new(MockCond { hash: key }), CondMeta { guidance, ..Default::default() }))
    }

    fn empty_latent(&self, req: &EmptyLatentRequest) -> EngineResult<Latent> {
        tracing::info!("[mock] empty_latent {:?} batch={}", req.kind, req.batch_size);
        let b = req.batch_size as usize;
        let data = Opaque::new(MockLatent { seed: 0, cond: 0 });
        Ok(match req.kind {
            EmptyLatentKind::Image4Ch { width, height } => Latent::single(
                LatentTensor { format: LatentFormat::Image4Ch, shape: vec![b, 4, height as usize / 8, width as usize / 8], data },
                PixelInfo::image(width, height, req.batch_size),
            ),
            EmptyLatentKind::Image16Ch { width, height } => Latent::single(
                LatentTensor { format: LatentFormat::Image16Ch, shape: vec![b, 16, height as usize / 8, width as usize / 8], data },
                PixelInfo::image(width, height, req.batch_size),
            ),
            EmptyLatentKind::Flux2 { width, height } => Latent::single(
                LatentTensor { format: LatentFormat::Flux2, shape: vec![b, 128, height as usize / 16, width as usize / 16], data },
                PixelInfo::image(width, height, req.batch_size),
            ),
            EmptyLatentKind::MiniMaxH3AudioVideo { width, height, length } => {
                let (frames, lt, at) = minimax_h3::temporal_shape(length);
                Latent {
                    samples: LatentSamples::AudioVideo {
                        video: LatentTensor { format: LatentFormat::MiniMaxH3Video, shape: vec![1, 24, lt as usize, height as usize / 16, width as usize / 16], data: data.clone() },
                        audio: LatentTensor { format: LatentFormat::MiniMaxH3Audio, shape: vec![1, 32, 2, at as usize], data },
                    },
                    pixel: PixelInfo {
                        width,
                        height,
                        frames,
                        batch_size: 1,
                        fps: Some(minimax_h3::FPS as f64),
                        audio_seconds: Some(frames as f64 / minimax_h3::FPS as f64),
                    },
                    noise_mask: None,
                    audio_noise_mask: None,
                    batch_index: None,
                    empty: false,
                }
            }
        })
    }

    fn sample(&self, req: &SampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        let conformed = arch_latent(req.model.arch.as_deref()).and_then(|(f, c, d)| req.latent.conform_empty(f, c, d));
        if let Some(l) = &conformed {
            tracing::info!("[mock]   conformed empty latent {:?} -> {:?} ({}x{})", latent_shape(&req.latent), latent_shape(l), l.pixel.width, l.pixel.height);
        }
        let latent = conformed.as_ref().unwrap_or(&req.latent);
        let end = req.end_at_step.min(req.steps);
        let total = end.saturating_sub(req.start_at_step);
        tracing::info!(
            "[mock] sample arch={:?} seed={} steps={} ({}..{}) cfg={} sampler={} scheduler={} previews={:?} variation={:?} model_patches={:?}",
            req.model.arch,
            req.noise_seed,
            req.steps,
            req.start_at_step,
            end,
            req.cfg,
            req.sampler,
            req.scheduler,
            req.previews,
            req.variation,
            req.model.patches.iter().map(|p| match p {
                ModelPatch::AttnTokenWeights { .. } => "AttnTokenWeights".to_string(),
                other => format!("{other:?}"),
            }).collect::<Vec<_>>(),
        );
        if req.latent.noise_mask.is_some() || req.latent.audio_noise_mask.is_some() {
            tracing::info!("[mock]   noise masks: {:?} audio={:?}", req.latent.noise_mask, req.latent.audio_noise_mask);
        }
        for e in &req.positive.entries {
            if !e.meta.reference_latents.is_empty() || !e.meta.minimax_keyframes.is_empty() {
                tracing::info!(
                    "[mock]   positive cond: {} reference latent(s) method={:?}, {} H3 keyframe(s)",
                    e.meta.reference_latents.len(),
                    e.meta.reference_latents_method,
                    e.meta.minimax_keyframes.len()
                );
            }
        }
        let cond = req.positive.entries.first().and_then(|e| e.embedding.downcast_ref::<MockCond>()).map(|c| c.hash).unwrap_or(0);
        let out = MockLatent { seed: req.noise_seed, cond };
        let key = hash_of((out.seed, out.cond));
        let px = &latent.pixel;
        // Preview at a small size with the right aspect ratio.
        let scale = 128.0 / px.width.max(px.height).max(1) as f64;
        let (pw, ph) = (((px.width as f64 * scale) as u32).max(8), ((px.height as f64 * scale) as u32).max(8));
        for step in 0..total {
            if progress.interrupted() {
                return Err(EngineError::Interrupted);
            }
            std::thread::sleep(self.step_delay);
            progress.progress(step + 1, total);
            if req.previews != PreviewMode::None && (step % 4 == 3 || step + 1 == total) {
                // Fade in from grey to the final colors so previews visibly progress.
                let mut img = render(pw, ph, 1, key, false);
                let t = (step + 1) as f32 / total.max(1) as f32;
                if let PixelData::U8(d) = &img.data {
                    let faded: Vec<u8> = d.iter().map(|&v| (v as f32 * t + 128.0 * (1.0 - t)) as u8).collect();
                    img.data = PixelData::U8(Arc::new(faded));
                }
                progress.preview(PreviewImage::Rgb(img));
            }
        }
        let data = Opaque::new(out);
        let samples = match &latent.samples {
            LatentSamples::Single(t) => LatentSamples::Single(LatentTensor { data, ..t.clone() }),
            LatentSamples::AudioVideo { video, audio } => {
                LatentSamples::AudioVideo { video: LatentTensor { data: data.clone(), ..video.clone() }, audio: LatentTensor { data, ..audio.clone() } }
            }
        };
        Ok(latent.with_samples(samples))
    }

    fn vae_decode(&self, req: &VaeDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        let px = &req.latent.pixel;
        let count = px.frames.max(1) * px.batch_size.max(1);
        tracing::info!("[mock] vae_decode {}x{} x{} frame(s) tiling={:?} vae_patches={:?}", px.width, px.height, count, req.tiling, req.vae.patches);
        if progress.interrupted() {
            return Err(EngineError::Interrupted);
        }
        let img = render(px.width, px.height, count, latent_key(&req.latent), px.frames > 1);
        if !req.vae.engine.downcast_ref::<MockVae>().is_some_and(|v| v.rgba) {
            return Ok(img);
        }
        // Fade alpha out toward the right edge so transparency is visible in the saved PNG.
        let rgba = img.to_rgba();
        let PixelData::F32(d) = &rgba.data else { unreachable!() };
        let mut d = (**d).clone();
        let w = rgba.width.max(1);
        for (idx, px) in d.chunks_exact_mut(4).enumerate() {
            px[3] = 1.0 - (idx as u32 % w) as f32 / w as f32;
        }
        Ok(ImageBatch::from_f32(rgba.count, rgba.height, rgba.width, 4, d))
    }

    fn vae_encode(&self, req: &VaeEncodeRequest) -> EngineResult<Latent> {
        tracing::info!("[mock] vae_encode {:?} vae_patches={:?}", req.pixels, req.vae.patches);
        let p = &req.pixels;
        let data = Opaque::new(MockLatent { seed: hash_of(&p.frame_u8(0)[..p.frame_len().min(3000)]), cond: 0 });
        let tensor = LatentTensor { format: LatentFormat::EngineDefined, shape: vec![1, 16, p.height as usize / 8, p.width as usize / 8], data };
        Ok(Latent::single(tensor, PixelInfo { width: p.width, height: p.height, frames: p.count, batch_size: 1, fps: None, audio_seconds: None }))
    }

    fn vae_decode_audio(&self, req: &VaeDecodeAudioRequest) -> EngineResult<AudioData> {
        let px = &req.latent.pixel;
        let seconds = px.audio_seconds.unwrap_or_else(|| px.frames as f64 / px.fps.unwrap_or(24.0)).max(0.1);
        tracing::info!("[mock] vae_decode_audio {seconds:.2}s tiling={:?}", req.tiling);
        let rate = 44100u32;
        let freq = 220.0 + (latent_key(&req.latent) % 440) as f64;
        let n = (seconds * rate as f64) as usize;
        let tone: Vec<f32> = (0..n).map(|i| ((i as f64 / rate as f64 * freq * std::f64::consts::TAU).sin() * 0.2) as f32).collect();
        let tone = Arc::new(tone);
        Ok(AudioData { sample_rate: rate, channels: vec![tone.clone(), tone] })
    }

    fn segment(&self, req: &SegmentRequest) -> EngineResult<Mask> {
        // A box over the middle third of each image, so mask-driven crops and composites have something to do.
        let (images, what) = match req {
            SegmentRequest::ClipSeg { images, text, .. } => (images, format!("clipseg '{text}'")),
            SegmentRequest::Yolo { image, model, index, .. } => (image, format!("yolo {} #{index}", model.name)),
        };
        tracing::info!("[mock] segment {what} on {images:?}");
        let (w, h) = (images.width as usize, images.height as usize);
        let plane: Vec<f32> = (0..w * h).map(|p| if (h / 3..2 * h / 3).contains(&(p / w)) && (w / 3..2 * w / 3).contains(&(p % w)) { 1.0 } else { 0.0 }).collect();
        let data = (0..images.count).flat_map(|_| plane.clone()).collect();
        Ok(Mask { count: images.count, height: images.height, width: images.width, data: Arc::new(data) })
    }

    fn remove_background(&self, images: &ImageBatch) -> EngineResult<ImageBatch> {
        tracing::info!("[mock] remove_background {images:?}");
        // Keep the pixels; alpha fades from opaque at the centre to transparent at the edges.
        let rgba = images.to_rgba();
        let (w, h) = (rgba.width as f32, rgba.height as f32);
        let mut data = Vec::with_capacity(rgba.frame_len() * rgba.count as usize);
        for f in 0..rgba.count {
            for (k, px) in rgba.frame_f32(f).chunks(4).enumerate() {
                let (x, y) = ((k % rgba.width as usize) as f32 / w - 0.5, (k / rgba.width as usize) as f32 / h - 0.5);
                data.extend_from_slice(&[px[0], px[1], px[2], (1.0 - 2.0 * (x * x + y * y).sqrt()).clamp(0.0, 1.0)]);
            }
        }
        Ok(ImageBatch::from_f32(rgba.count, rgba.height, rgba.width, 4, data))
    }

    fn model_sigmas(&self, req: &ScheduleRequest) -> EngineResult<Sigmas> {
        // A flow-model style linear schedule from 1 to 0.
        let (steps, skip) = match req {
            ScheduleRequest::Basic { steps, .. } => (*steps, 0),
            ScheduleRequest::SdTurbo { denoise, .. } => (10, 10 - (10.0 * denoise) as u32),
        };
        let mut s: Vec<f64> = (0..=steps).map(|i| 1.0 - i as f64 / steps.max(1) as f64).collect();
        if let ScheduleRequest::SdTurbo { steps: n, .. } = req {
            s = s[skip as usize..(skip + n).min(10) as usize].to_vec();
            s.push(0.0);
        }
        tracing::info!("[mock] model_sigmas {req:?} -> {s:?}");
        Ok(Sigmas(s))
    }

    fn sample_custom(&self, req: &CustomSampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<CustomSampleOutput> {
        let Guider::DualCfg { model, cond1, negative, cfg_conds, .. } = &req.guider;
        tracing::info!("[mock] sample_custom sampler={} seed={} sigmas={:?} guider={:?}", req.sampler.name, req.noise.seed, req.sigmas.0, std::mem::discriminant(&req.guider));
        let steps = req.sigmas.steps();
        let plain = SampleRequest {
            model: model.clone(),
            model_negative: None,
            positive: cond1.clone(),
            negative: negative.clone(),
            latent: req.latent.clone(),
            noise_seed: req.noise.seed,
            add_noise: true,
            steps,
            start_at_step: 0,
            end_at_step: steps,
            return_with_leftover_noise: false,
            denoise: 1.0,
            cfg: *cfg_conds,
            sampler: req.sampler.name.clone(),
            scheduler: "custom".into(),
            variation: None,
            sigma_min: None,
            sigma_max: None,
            rho: 7.0,
            previews: PreviewMode::Default,
            tile_size: None,
        };
        let output = self.sample(&plain, progress)?;
        Ok(CustomSampleOutput { denoised: output.clone(), output })
    }

    fn supports_custom_node(&self, _class: &str) -> bool {
        // The mock fakes every node with a known schema, like everything else it does.
        self.custom_nodes
    }

    fn custom_node(&self, req: &CustomNodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<Vec<NodeValue>> {
        tracing::info!("[mock] custom_node {} inputs={:?} -> {:?}", req.class_type, req.inputs.keys().collect::<Vec<_>>(), req.output_types);
        if progress.interrupted() {
            return Err(EngineError::Interrupted);
        }
        let int = |k: &str, d: i64| match req.inputs.get(k) {
            Some(NodeValue::Int(v)) => *v,
            Some(NodeValue::Float(v)) => *v as i64,
            _ => d,
        };
        let files: Vec<&ModelFile> = req.inputs.values().filter_map(|v| if let NodeValue::File(f) = v { Some(f) } else { None }).collect();
        let key = hash_of((&req.class_type, req.inputs.keys().collect::<Vec<_>>()));
        // Inputs of a type, in a stable order (`positive` before `negative`, then by name).
        let of_type = |ty: &str| -> Vec<&NodeValue> {
            let mut v: Vec<(&String, &NodeValue)> = req.inputs.iter().filter(|(_, v)| node_value_type(v) == ty).collect();
            v.sort_by_key(|(k, _)| (k.as_str() != "positive", k.as_str() != "negative", k.as_str().to_string()));
            v.into_iter().map(|(_, v)| v).collect()
        };
        let mut seen: std::collections::HashMap<&str, usize> = Default::default();
        let mut out = Vec::new();
        for ty in &req.output_types {
            let n = seen.entry(ty.as_str()).or_default();
            let same = of_type(ty);
            let pass = same.get(*n).or(same.last()).map(|v| (*v).clone());
            *n += 1;
            if let Some(v) = pass {
                out.push(v);
                continue;
            }
            let (w, h) = (int("width", 1024).max(8) as u32, int("height", 1024).max(8) as u32);
            let (frames, batch) = (int("length", 1).max(1) as u32, int("batch_size", 1).max(1) as u32);
            let mock_file = |folder| ModelFile { folder, name: req.class_type.clone(), path: Default::default() };
            let file = files.first().map(|f| (*f).clone());
            out.push(match ty.as_str() {
                "MODEL" => {
                    let f = file.unwrap_or_else(|| mock_file(ModelFolder::DiffusionModels));
                    NodeValue::Model(ModelHandle::new(Some(f.clone()), guess_arch(&f.name), Opaque::new(MockWeights { name: f.name })))
                }
                "CLIP" => NodeValue::Clip(TextEncoderHandle {
                    sources: files.iter().map(|f| (*f).clone()).collect(),
                    encoder_type: req.class_type.to_lowercase(),
                    device: "default".into(),
                    engine: Opaque::none(),
                    patches: vec![],
                }),
                "VAE" => NodeValue::Vae(VaeHandle::new(file, Opaque::new(MockVae { rgba: false }))),
                "CONDITIONING" => NodeValue::Conditioning(Conditioning::single(Opaque::new(MockCond { hash: key }), CondMeta::default())),
                "LATENT" => {
                    let shape = if frames > 1 { vec![batch as usize, 16, ((frames - 1) / 4 + 1) as usize, (h / 8) as usize, (w / 8) as usize] } else { vec![batch as usize, 16, (h / 8) as usize, (w / 8) as usize] };
                    let pixel = PixelInfo { width: w, height: h, frames, batch_size: batch, fps: None, audio_seconds: None };
                    let mut l = Latent::single(LatentTensor { format: LatentFormat::EngineDefined, shape, data: Opaque::new(MockLatent { seed: key, cond: 0 }) }, pixel);
                    l.empty = req.class_type.starts_with("Empty");
                    NodeValue::Latent(l)
                }
                "IMAGE" => NodeValue::Image(render(w, h, frames * batch, key, frames > 1)),
                "MASK" => NodeValue::Mask(Mask { count: batch, height: h, width: w, data: Arc::new(vec![1.0; (batch * w * h) as usize]) }),
                "AUDIO" => NodeValue::Audio(AudioData { sample_rate: 44100, channels: vec![Arc::new(vec![0.0; 44100])] }),
                "CLIP_VISION_OUTPUT" => {
                    NodeValue::ClipVisionOutput(ClipVisionOutput { model: AddonHandle::new(AddonKind::ClipVision, mock_file(ModelFolder::ClipVision), Opaque::none()), data: Opaque::new(key) })
                }
                "INT" => NodeValue::Int(0),
                "FLOAT" => NodeValue::Float(0.0),
                "STRING" => NodeValue::Str(String::new()),
                "BOOLEAN" => NodeValue::Bool(false),
                other => match addon_kind(other) {
                    Some(kind) => NodeValue::Addon(AddonHandle::new(kind, file.unwrap_or_else(|| mock_file(ModelFolder::Checkpoints)), Opaque::none())),
                    None => NodeValue::Other { ty: other.to_string(), data: Opaque::new(key) },
                },
            });
        }
        Ok(out)
    }

    fn load_image_only_checkpoint(&self, req: &LoadCheckpoint) -> EngineResult<(ModelHandle, AddonHandle, VaeHandle)> {
        let (model, _, vae) = self.load_checkpoint(req)?;
        Ok((model, AddonHandle::new(AddonKind::ClipVision, req.file.clone(), Opaque::none()), vae))
    }

    fn clip_vision_encode(&self, req: &ClipVisionEncodeRequest) -> EngineResult<ClipVisionOutput> {
        tracing::info!("[mock] clip_vision_encode {} crop={} {:?}", req.model.source.name, req.crop, req.image);
        let key = hash_of((&req.model.source.name, req.image.width, req.image.height, req.image.count, &req.crop));
        Ok(ClipVisionOutput { model: req.model.clone(), data: Opaque::new(key) })
    }

    fn upscale_image(&self, req: &UpscaleImageRequest) -> EngineResult<ImageBatch> {
        // ESRGAN-style models are named by their factor ("4x_...", "2x_...").
        let name = &req.model.source.name;
        let factor = name.split(|c: char| !c.is_ascii_digit()).find_map(|s| s.parse::<u32>().ok()).filter(|f| (1..=8).contains(f)).unwrap_or(4);
        tracing::info!("[mock] upscale_image {name} x{factor} {:?}", req.image);
        crate::encode::resize(&req.image, req.image.width * factor, req.image.height * factor, "bilinear", "disabled").map_err(EngineError::Internal)
    }

    fn latent_op(&self, op: &LatentOp) -> EngineResult<Latent> {
        // The mock keeps no real tensors: track shapes and pixel sizes, and derive a new content key.
        fn with_shape(l: &Latent, f: impl Fn(&mut Vec<usize>), key: u64) -> Latent {
            let samples = match &l.samples {
                LatentSamples::Single(t) => {
                    let mut shape = t.shape.clone();
                    f(&mut shape);
                    LatentSamples::Single(LatentTensor { format: t.format, shape, data: Opaque::new(MockLatent { seed: key, cond: 0 }) })
                }
                other => other.clone(),
            };
            Latent { samples, empty: false, ..l.clone() }
        }
        let out = match op {
            LatentOp::Add { a, b } => with_shape(a, |_| {}, hash_of((latent_key(a), latent_key(b), "add"))),
            LatentOp::Multiply { latent, factor } => with_shape(latent, |_| {}, hash_of((latent_key(latent), factor.to_bits()))),
            LatentOp::BlendMasked { a, b, factor, .. } => with_shape(a, |_| {}, hash_of((latent_key(a), latent_key(b), factor.to_bits()))),
            LatentOp::RepeatBatch { latent, amount } => {
                let mut l = with_shape(latent, |s| s[0] *= *amount as usize, latent_key(latent));
                l.pixel.batch_size *= amount;
                l
            }
            LatentOp::FromBatch { latent, index, length } => {
                let n = latent.pixel.batch_size.max(1);
                let len = (*length).min(n.saturating_sub(*index)).max(1);
                let mut l = with_shape(latent, |s| s[0] = len as usize, hash_of((latent_key(latent), *index)));
                l.pixel.batch_size = len;
                l
            }
            LatentOp::Batch { a, b } => {
                let mut l = with_shape(a, |s| s[0] += latent_batch(b), hash_of((latent_key(a), latent_key(b))));
                l.pixel.batch_size += b.pixel.batch_size.max(1);
                l
            }
            LatentOp::UpscaleBy { latent, scale, .. } => {
                let mut l = with_shape(latent, |s| {
                    let n = s.len();
                    s[n - 2] = (s[n - 2] as f64 * scale).round() as usize;
                    s[n - 1] = (s[n - 1] as f64 * scale).round() as usize;
                }, hash_of((latent_key(latent), scale.to_bits())));
                l.pixel.width = (l.pixel.width as f64 * scale).round() as u32;
                l.pixel.height = (l.pixel.height as f64 * scale).round() as u32;
                l
            }
            LatentOp::ReplaceFrames { destination, source, .. } => {
                let mut l = with_shape(destination, |_| {}, hash_of((latent_key(destination), latent_key(source))));
                l.noise_mask = source.noise_mask.clone();
                l
            }
            LatentOp::NormalizeStart { latent, start_frame_count, .. } => with_shape(latent, |_| {}, hash_of((latent_key(latent), *start_frame_count))),
            LatentOp::PrependReference { reference, latent } => {
                let mut l = with_shape(latent, |s| s[0] += latent_batch(reference), hash_of((latent_key(reference), latent_key(latent))));
                l.pixel.batch_size += reference.pixel.batch_size.max(1);
                l
            }
            LatentOp::TrimVideo { latent, trim_amount } => with_shape(latent, |s| if s.len() == 5 { s[2] = s[2].saturating_sub(*trim_amount as usize).max(1) }, latent_key(latent)),
        };
        tracing::info!("[mock] latent_op {} -> {:?}", format!("{op:?}").split([' ', '{', '(']).next().unwrap_or("?"), latent_shape(&out));
        Ok(out)
    }

    fn unsample(&self, req: &UnsampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        tracing::info!("[mock] unsample steps={} start_at_step={} sampler={} scheduler={}", req.steps, req.start_at_step, req.sampler, req.scheduler);
        let total = req.steps.saturating_sub(req.start_at_step);
        for s in 0..total {
            if progress.interrupted() {
                return Err(EngineError::Interrupted);
            }
            std::thread::sleep(self.step_delay);
            progress.progress(s + 1, total);
        }
        let key = hash_of((latent_key(&req.latent), "unsampled"));
        let samples = match &req.latent.samples {
            LatentSamples::Single(t) => LatentSamples::Single(LatentTensor { data: Opaque::new(MockLatent { seed: key, cond: 0 }), ..t.clone() }),
            other => other.clone(),
        };
        Ok(Latent { samples, empty: false, ..req.latent.clone() })
    }

    fn vae_encode_audio(&self, req: &VaeEncodeAudioRequest) -> EngineResult<Latent> {
        let seconds = req.audio.duration_seconds();
        // MiniMax H3's audio VAE: [B, 32, 2, frames] at 40 latent frames per second.
        let frames = ((seconds * minimax_h3::AUDIO_LATENT_FPS).round() as usize).max(1);
        tracing::info!("[mock] vae_encode_audio {:?} -> {frames} latent frame(s)", req.audio);
        let first = req.audio.channels.first().map(|c| c.iter().take(4000).map(|s| (s * 1000.0) as i32).collect::<Vec<_>>()).unwrap_or_default();
        let data = Opaque::new(MockLatent { seed: hash_of(&first), cond: 0 });
        let tensor = LatentTensor { format: LatentFormat::MiniMaxH3Audio, shape: vec![1, 32, 2, frames], data };
        let pixel = PixelInfo { width: 0, height: 0, frames: 1, batch_size: 1, fps: None, audio_seconds: Some(seconds) };
        Ok(Latent::single(tensor, pixel))
    }

    fn audio_silence(&self, req: &AudioSilenceRequest) -> EngineResult<Latent> {
        let LatentSamples::Single(t) = &req.latent.samples else {
            return Err(EngineError::InvalidInput("audio_silence needs the audio half of the latent".into()));
        };
        let total = *t.shape.last().unwrap_or(&0);
        let frames_for = |s: f64| if s <= 0.0 { 0 } else { ((s * minimax_h3::AUDIO_LATENT_FPS).ceil() as usize).clamp(1, total) };
        let (pre, suf) = (frames_for(req.prefix_seconds), frames_for(req.suffix_seconds));
        // A 1 x frames mask over the time axis: 0 = keep the encoded silence, 1 = generate.
        let mut mask = vec![1.0f32; total];
        mask[..pre].iter_mut().for_each(|m| *m = 0.0);
        mask[total - suf..].iter_mut().for_each(|m| *m = 0.0);
        if let Some(old) = req.latent.noise_mask.as_ref().filter(|m| m.data.len() == total) {
            mask.iter_mut().zip(old.data.iter()).for_each(|(m, o)| *m *= o);
        }
        tracing::info!("[mock] audio_silence prefix={}s ({pre} frames) suffix={}s ({suf} frames) of {total}", req.prefix_seconds, req.suffix_seconds);
        let noise_mask = Mask { count: 1, height: 1, width: total as u32, data: Arc::new(mask) };
        Ok(Latent { noise_mask: Some(noise_mask), ..req.latent.clone() })
    }

    fn warmup(&self, req: &WarmupRequest) -> EngineResult<()> {
        tracing::info!("[mock] warmup model={:?} te={:?} vae={:?}", req.model.source.as_ref().map(|f| &f.name), req.text_encoder.encoder_type, req.vae.source.as_ref().map(|f| &f.name));
        Ok(())
    }

    fn free_memory(&self, unload_models: bool, free_memory: bool) {
        tracing::info!("[mock] free_memory unload_models={unload_models} free_memory={free_memory}");
    }
}
