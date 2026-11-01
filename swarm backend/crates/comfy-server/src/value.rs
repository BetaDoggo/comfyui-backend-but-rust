//! Runtime values that flow along graph links.

use swarm_engine_api::{
    AddonHandle, AudioData, ClipVisionOutput, Opaque, Conditioning, ImageBatch, Latent, LoraHook, Mask, MiniMaxReferences, ModelFile, ModelHandle, TextEncoderHandle, VaeHandle,
    VideoClip,
};

#[derive(Clone, Debug)]
pub enum Value {
    Model(ModelHandle),
    Clip(TextEncoderHandle),
    Vae(VaeHandle),
    Conditioning(Conditioning),
    Latent(Latent),
    Image(ImageBatch),
    Mask(Mask),
    Audio(AudioData),
    Video(VideoClip),
    Hooks(Vec<LoraHook>),
    MiniMaxRefs(MiniMaxReferences),
    Embedding(ModelFile),
    /// GLIGEN, UPSCALE_MODEL, CLIP_VISION, CONTROL_NET, STYLE_MODEL, ...
    Addon(AddonHandle),
    ClipVision(ClipVisionOutput),
    /// Any other ComfyUI type (`BBOX`, `SIGMAS`, engine-owned values from `custom_node`, ...).
    Other { ty: String, data: Opaque },
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl Value {
    /// The ComfyUI type name, for error messages.
    pub fn type_name(&self) -> &str {
        match self {
            Value::Model(_) => "MODEL",
            Value::Clip(_) => "CLIP",
            Value::Vae(_) => "VAE",
            Value::Conditioning(_) => "CONDITIONING",
            Value::Latent(_) => "LATENT",
            Value::Image(_) => "IMAGE",
            Value::Mask(_) => "MASK",
            Value::Audio(_) => "AUDIO",
            Value::Video(_) => "VIDEO",
            Value::Hooks(_) => "HOOKS",
            Value::MiniMaxRefs(_) => "MiniMaxReferences",
            Value::Embedding(_) => "EMBEDDING",
            Value::Addon(a) => a.kind.comfy_type(),
            Value::ClipVision(_) => "CLIP_VISION_OUTPUT",
            Value::Other { ty, .. } => ty,
            Value::Int(_) => "INT",
            Value::Float(_) => "FLOAT",
            Value::Str(_) => "STRING",
            Value::Bool(_) => "BOOLEAN",
        }
    }
}
