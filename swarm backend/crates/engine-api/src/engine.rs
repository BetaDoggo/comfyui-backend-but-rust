//! The [`Engine`] trait: the heavy operations a native backend must provide.

use crate::conditioning::{ClipVisionOutput, Conditioning};
use crate::handles::{AddonHandle, ModelHandle, TextEncoderHandle, VaeHandle};
use crate::latent::Latent;
use crate::custom::{CustomNodeRequest, NodeValue};
use crate::custom_sampling::{CustomSampleOutput, CustomSampleRequest, ScheduleRequest, Sigmas};
use crate::media::{AudioData, ImageBatch, Mask};
use crate::requests::*;

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The user cancelled. The server reports this as `execution_interrupted`.
    #[error("interrupted")]
    Interrupted,
    /// The engine can't do this (yet). Shown to the SwarmUI user as the error message.
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

/// A preview image sent to SwarmUI while sampling.
pub enum PreviewImage {
    /// Already-encoded JPEG (the cheapest option; what ComfyUI sends).
    Jpeg(Vec<u8>),
    Png(Vec<u8>),
    /// Raw RGB frame(s); the server encodes them (animated previews use the first frame).
    Rgb(ImageBatch),
}

/// Live feedback channel for long operations. Implemented by the server.
pub trait ProgressSink: Send {
    /// Step progress for the current node (`value` of `max`). SwarmUI shows this as the
    /// per-image progress bar.
    fn progress(&mut self, value: u32, max: u32);
    /// A live preview image for the current node.
    fn preview(&mut self, image: PreviewImage);
    /// True once the user has asked to cancel. Long loops should poll this and return
    /// [`EngineError::Interrupted`].
    fn interrupted(&self) -> bool;
}

/// A no-op sink for callers that don't care about progress.
pub struct NullProgress;

impl ProgressSink for NullProgress {
    fn progress(&mut self, _: u32, _: u32) {}
    fn preview(&mut self, _: PreviewImage) {}
    fn interrupted(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    /// `"cuda"`, `"cpu"`, `"mps"`, ...
    pub kind: String,
    pub index: u32,
    pub vram_total: u64,
    pub vram_free: u64,
}

#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub name: String,
    pub version: String,
    pub devices: Vec<DeviceInfo>,
    pub ram_total: u64,
    pub ram_free: u64,
    /// Attention implementations the engine can switch between (`ModelAttentionBackend`).
    /// Listed in `/object_info`, which is where SwarmUI's "Model Attention Backend" dropdown
    /// comes from. The first entry is the default.
    pub attention_backends: Vec<String>,
}

/// A native generation engine.
///
/// All methods are called from a single dedicated worker thread, one prompt at a
/// time (ComfyUI semantics), so implementations don't need internal locking for
/// per-prompt state. Loader methods may be called repeatedly with the same file;
/// the server caches loader results between prompts, but engines should still keep
/// their own weight cache keyed by path.
pub trait Engine: Send + Sync + 'static {
    fn info(&self) -> EngineInfo;

    fn load_checkpoint(&self, req: &LoadCheckpoint) -> EngineResult<(ModelHandle, TextEncoderHandle, VaeHandle)>;
    fn load_diffusion_model(&self, req: &LoadDiffusionModel) -> EngineResult<ModelHandle>;
    fn load_text_encoder(&self, req: &LoadTextEncoder) -> EngineResult<TextEncoderHandle>;
    fn load_vae(&self, req: &LoadVae) -> EngineResult<VaeHandle>;

    /// `ImageOnlyCheckpointLoader` (SVD-style image-to-video checkpoints): the model, its
    /// CLIP-Vision encoder (an [`AddonKind::ClipVision`](crate::AddonKind) handle) and its VAE.
    /// Default: unsupported.
    fn load_image_only_checkpoint(&self, _req: &LoadCheckpoint) -> EngineResult<(ModelHandle, AddonHandle, VaeHandle)> {
        Err(EngineError::Unsupported("image-only checkpoints (ImageOnlyCheckpointLoader)".into()))
    }
    /// Add-on models (GLIGEN, upscalers, CLIP Vision, ControlNet, style models, ...).
    /// Default: a lazy handle holding just the file; load it when it's first used.
    fn load_addon(&self, req: &LoadAddon) -> EngineResult<AddonHandle> {
        Ok(AddonHandle::new(req.kind, req.file.clone(), crate::handles::Opaque::none()))
    }

    fn encode_text(&self, req: &TextEncodeRequest) -> EngineResult<Conditioning>;
    fn empty_latent(&self, req: &EmptyLatentRequest) -> EngineResult<Latent>;
    fn sample(&self, req: &SampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<Latent>;

    fn vae_decode(&self, req: &VaeDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch>;
    fn vae_encode(&self, req: &VaeEncodeRequest) -> EngineResult<Latent>;
    fn vae_decode_audio(&self, req: &VaeDecodeAudioRequest) -> EngineResult<AudioData>;

    /// See [`SegmentRequest`] (`<segment:>`, `<clear:>`). Default: unsupported.
    fn segment(&self, req: &SegmentRequest) -> EngineResult<Mask> {
        Err(EngineError::Unsupported(match req {
            SegmentRequest::ClipSeg { .. } => "CLIPSeg segmentation (SwarmClipSeg)".into(),
            SegmentRequest::Yolo { .. } => "YOLO detection (SwarmYoloDetection)".into(),
        }))
    }

    /// `SwarmRemBg` (Remove Background): the images with the background made transparent,
    /// returned as RGBA. The server derives the node's MASK output as `1 - alpha`. Default: unsupported.
    fn remove_background(&self, _images: &ImageBatch) -> EngineResult<ImageBatch> {
        Err(EngineError::Unsupported("background removal (SwarmRemBg)".into()))
    }

    /// `BasicScheduler` / `SDTurboScheduler`: a model-dependent sigma schedule (see
    /// [`ScheduleRequest`]). Default: unsupported.
    fn model_sigmas(&self, req: &ScheduleRequest) -> EngineResult<Sigmas> {
        Err(EngineError::Unsupported(match req {
            ScheduleRequest::Basic { .. } => "model sigma schedules (BasicScheduler)".into(),
            ScheduleRequest::SdTurbo { .. } => "SD-Turbo sigma schedules (SDTurboScheduler)".into(),
        }))
    }

    /// `SamplerCustomAdvanced`: sample along `sigmas` with the guider. Default: unsupported.
    fn sample_custom(&self, _req: &CustomSampleRequest, _progress: &mut dyn ProgressSink) -> EngineResult<CustomSampleOutput> {
        Err(EngineError::Unsupported("custom sampling (SamplerCustomAdvanced)".into()))
    }

    /// See [`ClipVisionEncodeRequest`]. Default: unsupported.
    fn clip_vision_encode(&self, _req: &ClipVisionEncodeRequest) -> EngineResult<ClipVisionOutput> {
        Err(EngineError::Unsupported("CLIP-Vision encoding (CLIPVisionEncode)".into()))
    }

    /// See [`UpscaleImageRequest`]. Default: unsupported.
    fn upscale_image(&self, _req: &UpscaleImageRequest) -> EngineResult<ImageBatch> {
        Err(EngineError::Unsupported("model upscaling (ImageUpscaleWithModel)".into()))
    }

    /// The generic hook (see [`crate::custom`]): any node class without a dedicated request,
    /// such as other model families' nodes and third-party node packs. Default: unsupported.
    fn custom_node(&self, req: &CustomNodeRequest, _progress: &mut dyn ProgressSink) -> EngineResult<Vec<NodeValue>> {
        Err(EngineError::Unsupported(format!("node '{}'", req.class_type)))
    }

    /// Whether the server should register `class` and route it to [`Engine::custom_node`]. Asked
    /// once at startup for every node without native support that has a known schema: other model
    /// families' stock nodes (Wan, LTX-V, Hunyuan, SD3, ...) and third-party packs (IP-Adapter,
    /// SAM2, RIFE, GGUF, ...). Registered nodes are listed in `/object_info`, which makes SwarmUI
    /// enable the matching features, so only claim what `custom_node` really handles. Default: none.
    fn supports_custom_node(&self, _class: &str) -> bool {
        false
    }

    /// See [`LatentOp`]. Default: unsupported.
    fn latent_op(&self, op: &LatentOp) -> EngineResult<Latent> {
        let name = format!("{op:?}");
        Err(EngineError::Unsupported(format!("latent operation {}", name.split([' ', '{', '(']).next().unwrap_or("?"))))
    }

    /// See [`UnsampleRequest`] (Unsampler Prompt). Default: unsupported.
    fn unsample(&self, _req: &UnsampleRequest, _progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        Err(EngineError::Unsupported("unsampling (SwarmUnsampler)".into()))
    }

    /// See [`VaeEncodeAudioRequest`] (Video Audio Input). Default: unsupported.
    fn vae_encode_audio(&self, _req: &VaeEncodeAudioRequest) -> EngineResult<Latent> {
        Err(EngineError::Unsupported("audio encoding (VAEEncodeAudio)".into()))
    }

    /// See [`AudioSilenceRequest`] (Audio Silent Prefix / Suffix Duration). Default: unsupported.
    fn audio_silence(&self, _req: &AudioSilenceRequest) -> EngineResult<Latent> {
        Err(EngineError::Unsupported("silent audio prefix/suffix (SwarmAudioSilentMaskPrefixSuffix)".into()))
    }

    /// See [`WarmupRequest`]. Default: nothing to do.
    fn warmup(&self, _req: &WarmupRequest) -> EngineResult<()> {
        Ok(())
    }

    /// `POST /free`. SwarmUI calls this from its "Free VRAM" buttons.
    fn free_memory(&self, _unload_models: bool, _free_memory: bool) {}
}
