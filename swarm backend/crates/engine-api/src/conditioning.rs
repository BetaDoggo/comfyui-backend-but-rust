//! ComfyUI `CONDITIONING`: a list of `[embedding_tensor, {metadata}]` pairs.
//! The embedding is engine-owned; the metadata keys the server knows how to set
//! from graph nodes are typed fields on [`CondMeta`].

use crate::handles::{AddonHandle, Opaque, VaeHandle};
use crate::latent::Latent;
use crate::media::{ImageBatch, Mask};

#[derive(Clone, Debug, Default)]
pub struct Conditioning {
    pub entries: Vec<CondEntry>,
}

#[derive(Clone, Debug)]
pub struct CondEntry {
    /// The encoded prompt (and pooled output, attention masks, etc.), engine-owned.
    pub embedding: Opaque,
    pub meta: CondMeta,
}

#[derive(Clone, Debug)]
pub struct CondMeta {
    /// Flux-style embedded guidance (`SwarmTextEncodeAdvanced.guidance`, -1 => unset).
    pub guidance: Option<f64>,
    /// Sampling-progress window this entry applies to (from `<fromto>` / `<alternate>` scheduling).
    pub start_percent: f64,
    pub end_percent: f64,
    /// `ReferenceLatent` nodes append VAE-encoded reference images here (Krea 2 edit-style refs).
    pub reference_latents: Vec<Latent>,
    /// `FluxKontextMultiReferenceLatentMethod` (Krea 2 sets this when refs are enabled):
    /// one of `offset`, `index`, `uxo/uno`, `index_timestep_zero`, ...
    pub reference_latents_method: Option<String>,
    /// MiniMax H3 image-to-video keyframes (`SwarmMiniMaxH3AddKeyframes`).
    pub minimax_keyframes: Vec<MiniMaxKeyframe>,
    pub minimax_frame_count: Option<u32>,
    /// `ConditioningMultiply` factor (applied to the embedding by the engine).
    pub multiplier: f64,
    /// `ConditioningZeroOut` was applied: engine should use zeros of the embedding's shape.
    pub zeroed: bool,
    /// `ConditioningSetMask` (regional prompting): this entry only applies inside `mask`.
    pub mask: Option<CondMask>,
    /// `GLIGENTextBoxApply`: GLIGEN boxes to apply alongside this entry.
    pub gligen: Vec<GligenBox>,
    /// `InpaintModelConditioning` (dedicated inpainting models): the VAE-encoded masked image and
    /// the mask, concatenated to the model input (ComfyUI's `concat_latent_image` / `concat_mask`).
    pub inpaint: Option<InpaintConcat>,
    /// Changes to the embedding tensor, in graph order (`StyleModelApply`, `ConditioningAverage`,
    /// `ConditioningConcat`). The engine builds the entry's embedding, then applies these in turn.
    pub embedding_ops: Vec<EmbeddingOp>,
    /// `unCLIPConditioning` (ReVision, SDXL only): CLIP-Vision image embeddings fed to unCLIP
    /// models' `adm` input (ComfyUI's `unclip_conditioning` list).
    pub unclip: Vec<UnclipInput>,
    /// `ControlNetApplyAdvanced` / `ControlNetInpaintingAliMamaApply` (ControlNet One/Two/Three):
    /// the control hints on this entry, oldest first (ComfyUI chains each onto the previous one).
    /// SwarmUI applies the same hints to the positive and negative prompts.
    pub controls: Vec<ControlHint>,
}

/// One applied ControlNet.
#[derive(Clone, Debug)]
pub struct ControlHint {
    /// The `CONTROL_NET` (with its union type, if `SetUnionControlNetType` set one).
    pub controlnet: AddonHandle,
    /// The control image (already preprocessed by the graph, if at all), RGB.
    pub image: ImageBatch,
    pub strength: f64,
    pub start_percent: f64,
    pub end_percent: f64,
    /// Passed by SwarmUI for SD3 / Flux / Flux 2 / Chroma / Qwen Image: ControlNets for those
    /// work on VAE-encoded hints.
    pub vae: Option<VaeHandle>,
    /// `ControlNetInpaintingAliMamaApply`: the inpaint mask (1 = repaint). For ControlNets that
    /// concatenate a mask, the engine zeroes the hint where the rounded, resized mask is 1 and
    /// concatenates `1 - mask`, as ComfyUI does.
    pub inpaint_mask: Option<Mask>,
}

/// A change to an entry's embedding tensor.
#[derive(Clone, Debug)]
pub enum EmbeddingOp {
    /// `StyleModelApply` (Use Style Model, Flux Redux style models): the style model turns the
    /// image embedding into extra tokens appended to the prompt's. `strength` scales those tokens
    /// (`Multiply`) or biases attention to them by `ln(strength)` (`AttnBias`).
    Style { model: AddonHandle, vision: ClipVisionOutput, strength: f64, strength_type: StyleStrength },
    /// `ConditioningAverage` (Style Model Merge Strength): `this * to_strength + from * (1 -
    /// to_strength)` for the embedding and the pooled output, with `from`'s embedding padded with
    /// zeros or truncated to this one's length. As in ComfyUI, only the first entry of the
    /// `conditioning_from` input is used.
    Average { from: Box<CondEntry>, to_strength: f64 },
    /// `ConditioningConcat`: append `from`'s embedding tokens to this one's (first entry only).
    Concat { from: Box<CondEntry> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleStrength {
    Multiply,
    AttnBias,
}

#[derive(Clone, Debug)]
pub struct UnclipInput {
    pub vision: ClipVisionOutput,
    pub strength: f64,
    pub noise_augmentation: f64,
}

/// `CLIP_VISION_OUTPUT`: an image encoded by a CLIP-Vision model (engine-owned embeddings;
/// ComfyUI's `image_embeds`, `last_hidden_state`, `mm_projected`, ...).
#[derive(Clone, Debug)]
pub struct ClipVisionOutput {
    pub model: AddonHandle,
    pub data: Opaque,
}

#[derive(Clone, Debug)]
pub struct InpaintConcat {
    /// The init image with the masked area set to 0.5 grey, VAE-encoded.
    pub latent: Latent,
    /// The (unrounded) pixel-space mask; the engine resizes it to the latent.
    pub mask: Mask,
}

/// A conditioning area mask. SwarmUI's `<region:>` / `<object:>` produce one entry per region
/// (strength `1 - Global Region Factor`), a background entry (the image minus all regions) and
/// the global prompt over the whole image (strength = Global Region Factor), combined into one
/// list. As in ComfyUI, the engine resizes `mask` to the latent and scales that entry's
/// prediction by `mask * strength` when combining conds.
#[derive(Clone, Debug)]
pub struct CondMask {
    pub mask: Mask,
    pub strength: f64,
    /// `set_cond_area = "mask bounds"`: crop the cond's area to the mask's bounding box
    /// (SwarmUI always sends `default`, i.e. false).
    pub set_area_to_bounds: bool,
}

/// One GLIGEN text box (SwarmUI's GLIGEN Model + `<region:>`/`<object:>`): the box text
/// encoded with the same text encoder, placed at a pixel rectangle.
#[derive(Clone, Debug)]
pub struct GligenBox {
    pub model: AddonHandle,
    pub text: Box<Conditioning>,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Default for CondMeta {
    fn default() -> Self {
        Self {
            guidance: None,
            start_percent: 0.0,
            end_percent: 1.0,
            reference_latents: Vec::new(),
            reference_latents_method: None,
            minimax_keyframes: Vec::new(),
            minimax_frame_count: None,
            multiplier: 1.0,
            zeroed: false,
            mask: None,
            gligen: Vec::new(),
            inpaint: None,
            embedding_ops: Vec::new(),
            unclip: Vec::new(),
            controls: Vec::new(),
        }
    }
}

/// One MiniMax H3 keyframe: an image VAE-encoded with the video VAE and pinned to a frame index.
#[derive(Clone, Debug)]
pub struct MiniMaxKeyframe {
    pub frame_index: u32,
    pub latent: Latent,
}

impl Conditioning {
    pub fn single(embedding: Opaque, meta: CondMeta) -> Self {
        Self { entries: vec![CondEntry { embedding, meta }] }
    }

    /// Applies `f` to every entry's metadata (ComfyUI's `conditioning_set_values`).
    pub fn map_meta(&self, mut f: impl FnMut(&mut CondMeta)) -> Self {
        let mut out = self.clone();
        for e in &mut out.entries {
            f(&mut e.meta);
        }
        out
    }
}
