//! # swarm-engine-api
//!
//! The contract a native generation engine implements so it can sit behind the
//! ComfyUI-compatible server (`comfy-server`) that SwarmUI talks to.
//!
//! SwarmUI never talks to the engine directly. It sends ComfyUI "API format"
//! node graphs; the server validates them, walks the graph, and lowers each
//! node into one of the strongly typed requests in [`requests`]. Things that are
//! pure bookkeeping (model patches such as sigma shift or LoRA, conditioning
//! metadata, image batching/cropping) are resolved by the server and arrive at
//! the engine as plain data attached to the handles, so an engine only has to
//! implement the heavy operations listed on [`Engine`].
//!
//! Everything the engine owns (weights, tensors, embeddings) is carried around
//! by the server as [`Opaque`] values that only the engine knows how to
//! downcast. Pixels and audio samples cross the boundary as concrete buffers
//! ([`ImageBatch`], [`AudioData`]) because the server has to encode them for
//! SwarmUI.

pub mod conditioning;
pub mod custom;
pub mod custom_sampling;
pub mod engine;
pub mod handles;
pub mod latent;
pub mod media;
pub mod patches;
pub mod prompt;
pub mod requests;

pub use custom::{CustomNodeRequest, NodeValue};
pub use custom_sampling::{CustomSampleOutput, CustomSampleRequest, Guider, NoiseSource, SamplerChoice, ScheduleRequest, Sigmas};
pub use conditioning::{
    ClipVisionOutput, CondEntry, ControlHint, CondMask, CondMeta, Conditioning, EmbeddingOp, GligenBox, InpaintConcat, MiniMaxKeyframe, SeedVr2Condition, StyleStrength, UnclipInput,
};
pub use engine::{DeviceInfo, Engine, EngineError, EngineInfo, EngineResult, NullProgress, PreviewImage, ProgressSink};
pub use handles::{AddonHandle, AddonKind, ModelFile, ModelFolder, ModelHandle, Opaque, TextEncoderHandle, VaeHandle, WeightDtype};
pub use latent::{Latent, LatentFormat, LatentSamples, LatentTensor, PixelInfo};
pub use media::{AudioData, ImageBatch, Mask, PixelData, VideoClip};
pub use patches::{LoraHook, ModelPatch, SnapRefPatch, SparseAttention, SparseAttentionMethod, TextEncoderPatch, TileAxis, VaePatch};
pub use prompt::{ParsedPrompt, PromptLeaf, PromptNode};
pub use requests::*;
