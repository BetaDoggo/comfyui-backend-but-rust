//! One typed request per heavy engine operation. Every field here is something
//! SwarmUI can actually send for Krea 2, Anima, or MiniMax H3; see
//! `docs/INPUT_MAPPING.md` for which SwarmUI parameter feeds which field.

use crate::conditioning::{ClipVisionOutput, Conditioning};
use crate::handles::{AddonHandle, AddonKind, ModelFile, ModelHandle, TextEncoderHandle, VaeHandle, WeightDtype};
use crate::latent::Latent;
use crate::media::{AudioData, ImageBatch, Mask};
use crate::patches::LoraHook;
use crate::prompt::ParsedPrompt;

// ---------------------------------------------------------------- loading ---

/// `CheckpointLoaderSimple`: one file containing model + text encoder + VAE.
#[derive(Clone, Debug)]
pub struct LoadCheckpoint {
    pub file: ModelFile,
}

/// `UNETLoader`: diffusion-model-only file (how all three target models are distributed).
#[derive(Clone, Debug)]
pub struct LoadDiffusionModel {
    pub file: ModelFile,
    pub weight_dtype: WeightDtype,
}

/// `CLIPLoader` (one file) or `DualCLIPLoader` (two files).
#[derive(Clone, Debug)]
pub struct LoadTextEncoder {
    pub files: Vec<ModelFile>,
    /// Krea 2: `"krea2"` (Qwen3-VL 4B). Anima: `"stable_diffusion"` (Qwen3 0.6B).
    /// MiniMax H3: `"minimax"` (Qwen3-VL 32B MiniMax finetune). MOSS Image: `"moss_image"`
    /// (Qwen3.5 9B). Qwen Image 2.1: `"qwen_image"` (Qwen3-VL 8B; the same type string as
    /// Qwen Image 1, told apart by the file).
    pub encoder_type: String,
    /// `"default"` or `"cpu"`.
    pub device: String,
}

/// `VAELoader`. Also used for MiniMax H3's audio VAE (SwarmUI loads it through the plain loader).
#[derive(Clone, Debug)]
pub struct LoadVae {
    pub file: ModelFile,
}

/// `SwarmJustLoadTheModelPlease`: SwarmUI's "load model" button and model preloading
/// (it sends a workflow with `steps = 0` ending in this node). Engines should load
/// everything to the device and return.
#[derive(Clone, Debug)]
pub struct WarmupRequest {
    pub model: ModelHandle,
    pub text_encoder: TextEncoderHandle,
    pub vae: VaeHandle,
}

/// Arithmetic and reshaping on engine-owned latents. The server can't touch latent data, so every
/// latent-to-latent node that isn't sampling or VAE work arrives here as one typed operation.
#[derive(Clone, Debug)]
pub enum LatentOp {
    /// `LatentAdd`: `a + b` (`b` resized to `a`'s spatial size if needed). Init Image Reset To Norm.
    Add { a: Latent, b: Latent },
    /// `LatentMultiply`: `latent * factor`. Init Image Reset To Norm.
    Multiply { latent: Latent, factor: f64 },
    /// `SwarmLatentBlendMasked`: `a * (1 - mask * factor) + b * (mask * factor)`, with `b` and the
    /// mask resized (bicubic) to `a`. Init Image Reset To Norm with a mask.
    BlendMasked { a: Latent, b: Latent, mask: Mask, factor: f64 },
    /// `RepeatLatentBatch`: repeat the batch `amount` times (Batch Size with an init image).
    RepeatBatch { latent: Latent, amount: u32 },
    /// `LatentFromBatch`: `length` latents starting at `index`.
    FromBatch { latent: Latent, index: u32, length: u32 },
    /// `LatentBatch`: concatenate two batches (`b` resized to `a`).
    Batch { a: Latent, b: Latent },
    /// `LatentUpscaleBy`: resize by `scale` with `method` (`nearest-exact`, `bilinear`, `area`,
    /// `bicubic`, `bislerp`). Refiner Upscale Method = `latent-*`.
    UpscaleBy { latent: Latent, method: String, scale: f64 },
    /// `TrimVideoLatent`: drop the first `trim_amount` latent frames.
    TrimVideo { latent: Latent, trim_amount: u32 },
    /// `SwarmReferenceOnly`: `reference` resized (bilinear) to `latent`'s spatial size, then
    /// concatenated *before* `latent` along the batch. The server sets the noise mask.
    PrependReference { reference: Latent, latent: Latent },
    /// `ReplaceVideoLatentFrames` (Kandinsky 5 image-to-video): `destination` with latent frames
    /// `index..index + source frames` replaced by `source`'s. The server resolves negative indices
    /// and skips the op (returning `destination`) when the source doesn't fit, as ComfyUI does. As in
    /// ComfyUI, the result keeps `source`'s other latent fields (noise mask etc.).
    ReplaceFrames { destination: Latent, source: Latent, index: u32 },
    /// `NormalizeVideoLatentStart` (Kandinsky 5 image-to-video): adaptive mean/std normalization
    /// of the first `start_frame_count` latent frames against the next
    /// `min(reference_frame_count, frames - 1)`. No-op for single-frame latents.
    NormalizeStart { latent: Latent, start_frame_count: u32, reference_frame_count: u32 },
}

/// Segmentation models that turn an image into a mask (SwarmUI's `<segment:...>` and `<clear:...>`).
#[derive(Clone, Debug)]
pub enum SegmentRequest {
    /// `SwarmClipSeg` (`<segment:text>`, `<clear:text>`): CLIPSeg mask of what matches `text`,
    /// one mask per image. `threshold` is applied by the model as in SwarmUI's node.
    ClipSeg { images: ImageBatch, text: String, threshold: f64 },
    /// `SwarmYoloDetection` (`<segment:yolo-model,...>`): box mask of the `index`-th detection
    /// (0 = all) of `model` (an Ultralytics YOLOv8 file), optionally limited to comma-separated
    /// class names, ordered by `sort_order`, above `threshold` confidence.
    Yolo { image: ImageBatch, model: ModelFile, index: u32, class_filter: String, sort_order: String, threshold: f64 },
}

/// `ImageUpscaleWithModel` (Refiner Upscale Method = `model-*`, SeedVR pre-upscale).
#[derive(Clone, Debug)]
pub struct UpscaleImageRequest {
    pub model: AddonHandle,
    pub image: ImageBatch,
}

/// `SwarmUnsampler` (Unsampler Prompt): sample *backwards* from the init image towards noise,
/// over the reversed sigma schedule (`sigmas.flip(0) + 1e-4`), stopping `start_at_step` steps
/// short, so a forward pass with an edited prompt reproduces the image closely. Zero noise,
/// seed 0, respects the latent's noise mask.
#[derive(Clone, Debug)]
pub struct UnsampleRequest {
    pub model: ModelHandle,
    pub positive: Conditioning,
    pub negative: Conditioning,
    pub latent: Latent,
    pub steps: u32,
    pub start_at_step: u32,
    pub sampler: String,
    pub scheduler: String,
    pub previews: PreviewMode,
}

/// `GLIGENLoader`, `UpscaleModelLoader`, `CLIPVisionLoader`, `ControlNetLoader`, `StyleModelLoader`,
/// `LatentUpscaleModelLoader`, `ModelPatchLoader`.
#[derive(Clone, Debug)]
pub struct LoadAddon {
    pub kind: AddonKind,
    pub file: ModelFile,
}

// --------------------------------------------------------------- encoding ---

#[derive(Clone, Debug)]
pub struct TextEncodeRequest {
    pub text_encoder: TextEncoderHandle,
    pub mode: TextEncodeMode,
}

#[derive(Clone, Debug)]
pub enum TextEncodeMode {
    /// Vanilla `CLIPTextEncode`. SwarmUI uses this for Anima when the prompt has no Swarm
    /// tags and no parentheses. The text may still contain `(word:1.2)` emphasis.
    Plain { text: String },
    /// `SwarmTextEncodeAdvanced`. Krea 2 always uses this; Anima/H3 use it when the prompt
    /// has tags or parentheses, and H3 always uses it when references are attached.
    Swarm(SwarmTextEncode),
}

#[derive(Clone, Debug)]
pub struct SwarmTextEncode {
    pub prompt: ParsedPrompt,
    /// Total sampling steps, for resolving `<fromto>` / `<alternate>`.
    pub steps: u32,
    /// "Intended" size (SDXL-style size conditioning). Krea 2 sends the real resolution.
    pub width: u32,
    pub height: u32,
    pub target_width: u32,
    pub target_height: u32,
    /// Embedded guidance (Flux-style). `None` when SwarmUI sends `-1`.
    pub guidance: Option<f64>,
    /// Chat template to wrap the prompt with. Krea 2 and MOSS Image => `Some("krea2")`.
    /// Qwen Image 2.1 => `None` (encoder default).
    pub llama_template: Option<String>,
    /// Vision inputs for VL text encoders: SwarmUI "Prompt Images", batched with `ImageBatch`.
    /// Krea 2 / MOSS Image: up to 10. Qwen Image 2.1: up to 16 (also sent as reference latents).
    pub images: Option<ImageBatch>,
    /// MiniMax H3 `<Picture i>` / `<Video k>` / `<Audio j>` references.
    pub minimax_refs: Option<MiniMaxReferences>,
    /// LoRAs referenced by `<lora:...//hook=N//>` tags in the prompt (index N).
    pub lora_hooks: Option<Vec<LoraHook>>,
    /// An encoded image for LLaVA-style text encoders (Hunyuan Video image-to-video's
    /// `<image:...>` prompt; ComfyUI passes its `mm_projected` as `image_embeds`).
    pub clip_vision: Option<ClipVisionOutput>,
}

/// `CLIPVisionEncode`: encode an image with a CLIP-Vision model (prompt images for style
/// models, ReVision/unCLIP, and some image-to-video models).
#[derive(Clone, Debug)]
pub struct ClipVisionEncodeRequest {
    pub model: AddonHandle,
    pub image: ImageBatch,
    /// `center`: center-crop to a square before resizing to the model's input size;
    /// `none`: resize the whole image.
    pub crop: String,
}

/// Output of `SwarmMiniMaxH3CollectReferences`. The server collects the raw media;
/// encoding them (video VAE for images/videos, audio VAE for audio, Qwen3-VL for the
/// prompt side) is the engine's job inside `encode_text`.
#[derive(Clone, Debug)]
pub struct MiniMaxReferences {
    pub vae: VaeHandle,
    pub audio_vae: VaeHandle,
    pub width: u32,
    pub height: u32,
    /// Generation length in frames (refs videos get cropped to this).
    pub length: u32,
    /// `"match"` (scale refs down to the generation's pixel area) or `"max"` (2048px short edge).
    pub ref_image_size: String,
    /// Up to 9 reference images.
    pub images: Vec<ImageBatch>,
    /// Up to 3 reference videos (24 fps frames) with optional soundtracks.
    pub videos: Vec<(ImageBatch, Option<AudioData>)>,
    /// Up to 3 standalone audio references.
    pub audios: Vec<AudioData>,
}

// ---------------------------------------------------------------- latents ---

#[derive(Clone, Debug)]
pub struct EmptyLatentRequest {
    pub kind: EmptyLatentKind,
    pub batch_size: u32,
}

#[derive(Clone, Debug)]
pub enum EmptyLatentKind {
    /// `EmptyLatentImage`: `[B, 4, H/8, W/8]`. SwarmUI also uses this for Qwen Image 2.1, at half
    /// the real resolution, as a placeholder the sampler widens (see `Latent::conform_empty`).
    Image4Ch { width: u32, height: u32 },
    /// `EmptySD3LatentImage`: `[B, 16, H/8, W/8]` (Krea 2, Anima).
    Image16Ch { width: u32, height: u32 },
    /// `EmptyFlux2LatentImage`: `[B, 128, H/16, W/16]` (MOSS Image).
    Flux2 { width: u32, height: u32 },
    /// `SwarmEmptyMiniMaxH3LatentAV`: joint video `[1, 24, T, H/16, W/16]` + audio `[1, 32, 2, A]`.
    MiniMaxH3AudioVideo { width: u32, height: u32, length: u32 },
}

/// MiniMax H3 temporal layout helpers, matching `SwarmModels.py::temporal_shape`.
pub mod minimax_h3 {
    pub const FPS: u32 = 24;
    pub const AUDIO_LATENT_FPS: f64 = 40.0;

    /// Rounds a frame count up to the `17k + 5` grid (1 and 2 frames are special-cased).
    pub fn align_frames(n: u32) -> u32 {
        let mut n = n.max(5);
        while n % 17 != 5 {
            n += 1;
        }
        n
    }

    /// `(frame_count, video_latent_t, audio_latent_t)` for a requested length.
    pub fn temporal_shape(length: u32) -> (u32, u32, u32) {
        match length {
            1 => (1, 1, 2),
            2 => (2, 2, 3),
            _ => {
                let frames = align_frames(length);
                let latent_t = if frames <= 5 { 2 } else { ((frames - 5) / 17) * 5 + 2 };
                let audio_t = (frames as f64 / FPS as f64 * AUDIO_LATENT_FPS).round() as u32;
                (frames, latent_t, audio_t)
            }
        }
    }
}

// --------------------------------------------------------------- sampling ---

/// `SwarmKSampler` (what SwarmUI always uses when the Swarm nodes are advertised),
/// `KSamplerAdvanced`, or `KSampler`, normalized into one request.
#[derive(Clone, Debug)]
pub struct SampleRequest {
    pub model: ModelHandle,
    /// Optional separate model for the negative pass (SwarmUI "Negative Model" param).
    pub model_negative: Option<ModelHandle>,
    pub positive: Conditioning,
    pub negative: Conditioning,
    /// May be an empty placeholder with the wrong channel count; see `Latent::conform_empty`.
    pub latent: Latent,
    pub noise_seed: u64,
    /// Inject fresh noise at the start (false when continuing a refiner pass).
    pub add_noise: bool,
    pub steps: u32,
    /// Skip steps before this (img2img: `round(steps * (1 - creativity))`).
    pub start_at_step: u32,
    /// Stop after this step (10000 => run to the end).
    pub end_at_step: u32,
    pub return_with_leftover_noise: bool,
    /// `KSampler.denoise`; SwarmUI's samplers express this through start/end steps and send 1.0.
    pub denoise: f64,
    pub cfg: f64,
    /// ComfyUI sampler name. Model defaults SwarmUI picks when the user hasn't:
    /// Krea 2 / MOSS Image / Qwen Image 2.1 => `euler`, Anima => `er_sde`, MiniMax H3 => `res_multistep`.
    pub sampler: String,
    /// ComfyUI scheduler name. All five default to `simple`. Qwen Image 2.1 Turbo is meant to
    /// use `align_your_steps` (a fixed 8-step sigma list in `SwarmKSampler`).
    pub scheduler: String,
    pub variation: Option<VariationSeed>,
    /// Sigma overrides (`None` when SwarmUI sends -1).
    pub sigma_min: Option<f64>,
    pub sigma_max: Option<f64>,
    pub rho: f64,
    pub previews: PreviewMode,
    /// Tiled sampling (SwarmUI "Tile Sample" param) with this tile size.
    pub tile_size: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct VariationSeed {
    pub seed: u64,
    pub strength: f64,
}

/// `SwarmKSampler.previews`. Video models default to `Animate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewMode {
    Default,
    None,
    One,
    Second,
    Iterate,
    Animate,
}

impl PreviewMode {
    pub const OPTIONS: [&'static str; 6] = ["default", "none", "one", "second", "iterate", "animate"];

    pub fn parse(s: &str) -> Self {
        match s {
            "none" => PreviewMode::None,
            "one" => PreviewMode::One,
            "second" => PreviewMode::Second,
            "iterate" => PreviewMode::Iterate,
            "animate" => PreviewMode::Animate,
            _ => PreviewMode::Default,
        }
    }
}

// -------------------------------------------------------------------- VAE ---

#[derive(Clone, Copy, Debug)]
pub struct VaeTiling {
    pub tile_size: u32,
    pub overlap: u32,
    pub temporal_size: u32,
    pub temporal_overlap: u32,
}

/// `VAEDecode` / `VAEDecodeTiled`. For MiniMax H3 SwarmUI always uses tiled decode
/// (256 / 64, temporal 9999 = no temporal tiling) unless "model specific enhancements" is off.
///
/// Return RGBA (`channels = 4`) from alpha-capable VAEs (Qwen Image 2.1). The server saves it as
/// an RGBA PNG; SwarmUI adds `SplitImageWithAlpha` wherever it needs plain RGB.
#[derive(Clone, Debug)]
pub struct VaeDecodeRequest {
    pub vae: VaeHandle,
    pub latent: Latent,
    pub tiling: Option<VaeTiling>,
}

/// `VAEEncode` / `VAEEncodeTiled` (init images, reference latents, H3 keyframes).
/// `pixels` may be RGB or RGBA. RGBA VAEs (Qwen Image 2.1) get RGB from loaded images (pad alpha
/// with 1.0, as ComfyUI does) and RGBA when SwarmUI re-encodes a decode (refiner / upscale).
#[derive(Clone, Debug)]
pub struct VaeEncodeRequest {
    pub vae: VaeHandle,
    pub pixels: ImageBatch,
    pub tiling: Option<VaeTiling>,
}

/// `VAEDecodeAudio` (MiniMax H3 audio half, via its audio VAE) / `VAEDecodeAudioTiled`.
#[derive(Clone, Debug)]
pub struct VaeDecodeAudioRequest {
    pub vae: VaeHandle,
    pub latent: Latent,
    /// `VAEDecodeAudioTiled` (VAE Tile Size / Overlap on audio models): tile along the time axis.
    pub tiling: Option<AudioTiling>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioTiling {
    pub tile_size: u32,
    pub overlap: u32,
}

/// `VAEEncodeAudio`: encode a waveform with an audio VAE. SwarmUI uses it for Video Audio
/// Input on MiniMax H3: the clip (padded by `SwarmEnsureAudio` to at least 0.1 s, silence at
/// the *start*) becomes the audio half of the AV latent, with an all-zero noise mask so
/// sampling keeps it. The result's `pixel.audio_seconds` should be the clip's duration.
#[derive(Clone, Debug)]
pub struct VaeEncodeAudioRequest {
    pub vae: VaeHandle,
    pub audio: AudioData,
}

/// `SwarmAudioSilentMaskPrefixSuffix` (Audio Silent Prefix / Suffix Duration). On the audio
/// latent: encode `max(prefix, suffix)` seconds of silence (zeros, the VAE's channel count and
/// sample rate, padded up to a multiple of its hop length) with `vae`, copy the first latent
/// frames of that into the first `prefix` and last `suffix` seconds' worth of frames along the
/// time axis, and return the latent with a noise mask that is 0 there (keep) and 1 elsewhere,
/// merged with any existing `noise_mask`. Durations of 0 leave that end alone.
#[derive(Clone, Debug)]
pub struct AudioSilenceRequest {
    pub vae: VaeHandle,
    pub latent: Latent,
    pub prefix_seconds: f64,
    pub suffix_seconds: f64,
}
