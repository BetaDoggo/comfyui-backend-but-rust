//! The generic node hook: how nodes without a dedicated request type reach the engine.
//!
//! Model-family-specific nodes (Wan, LTX-V, Hunyuan, SD3, Stable Cascade, ...) and third-party
//! node packs (IP-Adapter, SAM2, frame interpolation, ...) are validated against their real
//! ComfyUI schemas by the server, then handed to [`crate::Engine::custom_node`] with their
//! resolved inputs. An engine implements the classes it supports and returns the node's outputs
//! in order; everything else stays "unsupported", which SwarmUI shows as a clear error.

use std::collections::BTreeMap;

use crate::conditioning::{ClipVisionOutput, Conditioning};
use crate::handles::{AddonHandle, ModelFile, ModelHandle, Opaque, TextEncoderHandle, VaeHandle};
use crate::latent::Latent;
use crate::media::{AudioData, ImageBatch, Mask, VideoClip};

/// A value on a graph link or widget, as the engine sees it.
#[derive(Clone, Debug)]
pub enum NodeValue {
    Model(ModelHandle),
    Clip(TextEncoderHandle),
    Vae(VaeHandle),
    Conditioning(Conditioning),
    Latent(Latent),
    Image(ImageBatch),
    Mask(Mask),
    Audio(AudioData),
    Video(VideoClip),
    /// GLIGEN, UPSCALE_MODEL, CLIP_VISION, CONTROL_NET, STYLE_MODEL, ...
    Addon(AddonHandle),
    ClipVisionOutput(ClipVisionOutput),
    /// A model-file combo (`ckpt_name`, `unet_name`, `ipadapter_file`, ...), resolved to the file.
    File(ModelFile),
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    /// Any other ComfyUI type (`BBOX`, `SAM2MODEL`, `IPADAPTER`, `HOOKS`, ...). Server-made payloads:
    /// `SIGMAS` = [`crate::Sigmas`], `NOISE` = [`crate::NoiseSource`], `SAMPLER` =
    /// [`crate::SamplerChoice`], `GUIDER` = [`crate::Guider`], `HOOKS` = `Vec<LoraHook>`,
    /// `MiniMaxReferences` = [`crate::MiniMaxReferences`], `EMBEDDING` = [`ModelFile`].
    /// Values the server creates itself document their payload type where they're made (e.g.
    /// `BBOX` from `SwarmSam2BBoxFromJson` is a `Vec<[f64; 4]>` of `[x1, y1, x2, y2]` pixels);
    /// values the engine returns from [`crate::Engine::custom_node`] are the engine's own.
    Other { ty: String, data: Opaque },
}

/// One node handed to [`crate::Engine::custom_node`].
#[derive(Clone, Debug)]
pub struct CustomNodeRequest {
    /// The ComfyUI class, e.g. `"WanImageToVideo"`.
    pub class_type: String,
    /// Every input that was provided or has a default, by name (Autogrow / DynamicCombo members
    /// as `"group.member"`).
    pub inputs: BTreeMap<String, NodeValue>,
    /// The node's declared output types; return one value per entry, in order.
    pub output_types: Vec<String>,
}
