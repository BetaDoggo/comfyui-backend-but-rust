//! Handles to engine-owned resources (loaded weights) plus the file references
//! SwarmUI used to request them.

use std::any::Any;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use crate::patches::{ModelPatch, TextEncoderPatch, VaePatch};

/// Engine-owned data that the server carries between nodes without looking inside.
///
/// The engine creates these (e.g. `Opaque::new(MyLoadedUnet { .. })`) and gets
/// them back on later calls, where it can `downcast_ref` to its own type.
#[derive(Clone, Default)]
pub struct Opaque(Option<Arc<dyn Any + Send + Sync>>);

impl Opaque {
    /// An empty value, for engines (or mocks) that don't need to store anything.
    pub fn none() -> Self {
        Self(None)
    }

    pub fn new<T: Any + Send + Sync>(value: T) -> Self {
        Self(Some(Arc::new(value)))
    }

    pub fn from_arc(value: Arc<dyn Any + Send + Sync>) -> Self {
        Self(Some(value))
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.as_deref().and_then(|v| v.downcast_ref::<T>())
    }

    pub fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

impl fmt::Debug for Opaque {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            None => f.write_str("Opaque(none)"),
            Some(_) => f.write_str("Opaque(..)"),
        }
    }
}

/// ComfyUI model folder categories. The server maps each of these to one or
/// more directories on disk (see the server's `--models-root` handling).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelFolder {
    Checkpoints,
    DiffusionModels,
    TextEncoders,
    Vae,
    Loras,
    Embeddings,
    UpscaleModels,
    LatentUpscaleModels,
    ClipVision,
    ControlNet,
    Gligen,
    StyleModels,
    /// `model_patches` (DiffSynth / LLLite style control patches, `ModelPatchLoader`).
    ModelPatches,
    /// SwarmUI's `yolov8` folder (`SwarmYoloDetection`).
    Yolov8,
    /// `ipadapter` (IP-Adapter models, `IPAdapterModelLoader`).
    IpAdapter,
    /// `tensorrt` (TensorRT engines, `TensorRTLoader`).
    TensorRt,
    /// `background_removal` (BiRefNet, `LoadBackgroundRemovalModel`).
    BackgroundRemoval,
}

impl ModelFolder {
    pub const ALL: [ModelFolder; 17] = [
        ModelFolder::Checkpoints,
        ModelFolder::DiffusionModels,
        ModelFolder::TextEncoders,
        ModelFolder::Vae,
        ModelFolder::Loras,
        ModelFolder::Embeddings,
        ModelFolder::UpscaleModels,
        ModelFolder::LatentUpscaleModels,
        ModelFolder::ClipVision,
        ModelFolder::ControlNet,
        ModelFolder::Gligen,
        ModelFolder::StyleModels,
        ModelFolder::ModelPatches,
        ModelFolder::Yolov8,
        ModelFolder::IpAdapter,
        ModelFolder::TensorRt,
        ModelFolder::BackgroundRemoval,
    ];

    /// The folder key ComfyUI uses in `folder_paths` (and `/models/{folder}`).
    pub fn comfy_key(self) -> &'static str {
        match self {
            ModelFolder::Checkpoints => "checkpoints",
            ModelFolder::DiffusionModels => "diffusion_models",
            ModelFolder::TextEncoders => "text_encoders",
            ModelFolder::Vae => "vae",
            ModelFolder::Loras => "loras",
            ModelFolder::Embeddings => "embeddings",
            ModelFolder::UpscaleModels => "upscale_models",
            ModelFolder::LatentUpscaleModels => "latent_upscale_models",
            ModelFolder::ClipVision => "clip_vision",
            ModelFolder::ControlNet => "controlnet",
            ModelFolder::Gligen => "gligen",
            ModelFolder::StyleModels => "style_models",
            ModelFolder::ModelPatches => "model_patches",
            ModelFolder::Yolov8 => "yolov8",
            ModelFolder::IpAdapter => "ipadapter",
            ModelFolder::TensorRt => "tensorrt",
            ModelFolder::BackgroundRemoval => "background_removal",
        }
    }

    pub fn from_comfy_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.comfy_key() == key)
    }
}

/// A model file as requested by SwarmUI and resolved by the server.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelFile {
    pub folder: ModelFolder,
    /// The name exactly as it appears in the workflow (forward slashes), e.g. `"Krea/krea2_fp8.safetensors"`.
    pub name: String,
    /// Absolute path on disk.
    pub path: PathBuf,
}

/// `UNETLoader.weight_dtype`. SwarmUI sends `fp8_e4m3fn` (or `_fast` on 30xx+)
/// by default for big models, and `default` for small dense ones such as Anima.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightDtype {
    Default,
    Fp8E4m3fn,
    Fp8E4m3fnFast,
    Fp8E5m2,
}

impl WeightDtype {
    pub const OPTIONS: [&'static str; 4] = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"];

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "default" => WeightDtype::Default,
            "fp8_e4m3fn" => WeightDtype::Fp8E4m3fn,
            "fp8_e4m3fn_fast" => WeightDtype::Fp8E4m3fnFast,
            "fp8_e5m2" => WeightDtype::Fp8E5m2,
            _ => return None,
        })
    }
}

/// A loaded diffusion model (`MODEL` in ComfyUI terms).
///
/// `patches` accumulates as the graph applies sampling shifts, LoRAs, etc. The
/// engine should treat `engine` as the immutable base weights and apply
/// `patches` (in order) when sampling. The server never mutates `engine`.
#[derive(Clone, Debug)]
pub struct ModelHandle {
    pub source: Option<ModelFile>,
    /// Architecture the engine detected on load, if any (e.g. `"krea-2"`, `"anima"`, `"minimax-h3"`).
    pub arch: Option<String>,
    pub engine: Opaque,
    pub patches: Vec<ModelPatch>,
}

impl ModelHandle {
    pub fn new(source: Option<ModelFile>, arch: Option<String>, engine: Opaque) -> Self {
        Self { source, arch, engine, patches: Vec::new() }
    }

    /// Returns a clone with one more patch applied (ComfyUI's `model.clone()` + patch idiom).
    pub fn with_patch(&self, patch: ModelPatch) -> Self {
        let mut m = self.clone();
        m.patches.push(patch);
        m
    }
}

/// A loaded text encoder (`CLIP` in ComfyUI terms, even when it is an LLM).
#[derive(Clone, Debug)]
pub struct TextEncoderHandle {
    pub sources: Vec<ModelFile>,
    /// The `CLIPLoader.type` value: `"krea2"` for Krea 2, `"stable_diffusion"` for Anima's
    /// Qwen3 0.6B, `"minimax"` for MiniMax H3's Qwen3-VL 32B, etc.
    pub encoder_type: String,
    pub device: String,
    pub engine: Opaque,
    pub patches: Vec<TextEncoderPatch>,
}

impl TextEncoderHandle {
    pub fn with_patch(&self, patch: TextEncoderPatch) -> Self {
        let mut c = self.clone();
        c.patches.push(patch);
        c
    }
}

/// Add-on models that other nodes consume (ComfyUI gives each its own link type).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddonKind {
    /// `GLIGENLoader` (`gligen/`).
    Gligen,
    /// `UpscaleModelLoader` (ESRGAN-style pixel upscalers, `upscale_models/`).
    UpscaleModel,
    /// `LatentUpscaleModelLoader` (`latent_upscale_models/`).
    LatentUpscaleModel,
    /// `CLIPVisionLoader` (`clip_vision/`).
    ClipVision,
    /// `ControlNetLoader` (`controlnet/`).
    ControlNet,
    /// `StyleModelLoader` (Flux Redux style models, `style_models/`).
    StyleModel,
    /// `ModelPatchLoader` (`model_patches/`: DiffSynth / LLLite control patches).
    ModelPatch,
    /// `LoadBackgroundRemovalModel` (`background_removal/`: BiRefNet).
    BackgroundRemoval,
}

impl AddonKind {
    /// The ComfyUI link type the loader outputs.
    pub fn comfy_type(self) -> &'static str {
        match self {
            AddonKind::Gligen => "GLIGEN",
            AddonKind::UpscaleModel => "UPSCALE_MODEL",
            AddonKind::LatentUpscaleModel => "LATENT_UPSCALE_MODEL",
            AddonKind::ClipVision => "CLIP_VISION",
            AddonKind::ControlNet => "CONTROL_NET",
            AddonKind::StyleModel => "STYLE_MODEL",
            AddonKind::ModelPatch => "MODEL_PATCH",
            AddonKind::BackgroundRemoval => "BACKGROUND_REMOVAL",
        }
    }
}

/// A loaded add-on model. `engine` is empty when the engine loads lazily (the default
/// [`crate::Engine::load_addon`]); the file reference is always there.
#[derive(Clone, Debug)]
pub struct AddonHandle {
    pub kind: AddonKind,
    pub source: ModelFile,
    pub engine: Opaque,
    /// `SetUnionControlNetType` (ControlNet Union Type) on a union ControlNet: one of `openpose`,
    /// `depth`, `hed/pidi/scribble/ted`, `canny/lineart/anime_lineart/mlsd`, `normal`, `segment`,
    /// `tile`, `repaint`. `None`: not set, or `auto`.
    pub control_type: Option<String>,
}

impl AddonHandle {
    pub fn new(kind: AddonKind, source: ModelFile, engine: Opaque) -> Self {
        Self { kind, source, engine, control_type: None }
    }
}

/// A loaded VAE (image, video, or audio).
#[derive(Clone, Debug)]
pub struct VaeHandle {
    /// `None` for ComfyUI's `pixel_space` VAE: an identity "VAE" for pixel-space models (Chroma
    /// Radiance, HiDream O1), where latents are the pixels themselves.
    pub source: Option<ModelFile>,
    pub engine: Opaque,
    pub patches: Vec<VaePatch>,
}

impl VaeHandle {
    pub fn new(source: Option<ModelFile>, engine: Opaque) -> Self {
        Self { source, engine, patches: Vec::new() }
    }

    pub fn with_patch(&self, patch: VaePatch) -> Self {
        let mut v = self.clone();
        v.patches.push(patch);
        v
    }
}
