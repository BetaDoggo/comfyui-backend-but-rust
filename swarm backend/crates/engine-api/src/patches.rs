//! Model / text-encoder modifications that ComfyUI expresses as separate nodes
//! which clone-and-patch a model. The server resolves these graph nodes into
//! plain data appended to the handle's `patches` list, in graph order, so the
//! engine sees the complete recipe at sample/encode time.

use crate::conditioning::Conditioning;
use crate::handles::{AddonHandle, ModelFile, VaeHandle};
use crate::media::{ImageBatch, Mask};

#[derive(Clone, Debug)]
pub enum ModelPatch {
    /// `LoraLoader` / `LoraLoaderModelOnly` (model half).
    Lora { file: ModelFile, strength: f32 },
    /// `ModelSamplingFlux`. SwarmUI emits this for Krea 2 only when the user sets
    /// Sigma Shift, with `max_shift == base_shift == shift`.
    SamplingFlux { max_shift: f64, base_shift: f64, width: u32, height: u32 },
    /// `ModelSamplingAuraFlow`. SwarmUI emits this for Anima only when the user sets Sigma Shift.
    SamplingAuraFlow { shift: f64, sampling: String },
    /// `ModelSamplingSD3` (not used by the three target models; kept for custom workflows).
    SamplingSd3 { shift: f64 },
    /// `MiniMaxH3SigmaShift`. Always emitted for MiniMax H3: video shift = Sigma Shift param
    /// (default 12), audio shift = 3.
    MiniMaxH3Shift { shift_video: f64, shift_audio: f64 },
    /// `MiniMaxH3ContextWindows` (Video Context Window): sample the clip in overlapping windows of
    /// `context_length` frames, `context_overlap` shared, blended by `fuse_method` (`pyramid`,
    /// `relative`, `flat` or `overlap-linear`). SwarmUI sends `pyramid`, FreeNoise on and
    /// `split_conds_to_windows` off. Lengths are in frames, as the node takes them.
    MiniMaxH3ContextWindows { context_length: u32, context_overlap: u32, fuse_method: String, freenoise: bool, split_conds_to_windows: bool },
    /// `SwarmAttnTokenWeights`: Krea 2 "model specific enhancement" that applies per-token
    /// prompt weights inside attention instead of scaling embeddings. Carries the
    /// positive conditioning whose weights should be applied.
    AttnTokenWeights { conditioning: Box<Conditioning> },

    // ------------------------------------------------ guidance tweaks (SwarmUI "Alternate Guidance")
    /// `FreeU` (`version: 1`) / `FreeU_V2` (`version: 2`): scale UNet backbone (`b*`) and skip (`s*`)
    /// features. SwarmUI's FreeU group; it only emits this when "[FreeU] Apply To" matches the stage.
    FreeU { version: u8, b1: f64, b2: f64, s1: f64, s2: f64 },
    /// `SelfAttentionGuidance` (Self-Attention Guidance Scale / Sigma Blur; Swarm default blur 2.0).
    SelfAttentionGuidance { scale: f64, blur_sigma: f64 },
    /// `PerturbedAttentionGuidance` (Perturbed-Attention Guidance Scale).
    PerturbedAttentionGuidance { scale: f64 },
    /// `RescaleCFG` (Rescale CFG Multiplier): rescale the CFG result towards the positive's std.
    RescaleCfg { multiplier: f64 },
    /// `RenormCFG` (Renorm CFG). SwarmUI always sends `cfg_trunc = 100` (apply at every timestep).
    RenormCfg { cfg_trunc: f64, renorm_cfg: f64 },
    /// `CFGZeroStar` (Use CFG Zero Star): optimized CFG scale + zeroing the first steps.
    CfgZeroStar,
    /// `TCFG` (Use TCFG): tangential damping of the unconditional prediction.
    Tcfg,
    /// `NAGuidance` (Normalized Attention Guidance Scale / Alpha / Tau; Swarm defaults 0.5 / 1.5).
    /// SwarmUI only emits it when the scale is > 0.
    NormalizedAttentionGuidance { scale: f64, alpha: f64, tau: f64 },
    /// `CFGNorm`. SwarmUI adds it automatically for some model classes (Lens: strength 1, `pre_cfg`).
    CfgNorm { strength: f64, pre_cfg: bool },
    /// `ModelNoiseScale`. SwarmUI adds it automatically for HiDream O1 (7.5).
    NoiseScale { noise_scale: f64 },
    /// `ModelSamplingDiscrete` (Override Prediction Type): `eps`, `v_prediction`, `lcm`, `x0`, ...
    SamplingDiscrete { sampling: String, zsnr: bool },
    /// `QwenImageDiffsynthControlnet` (a ControlNet model of class `*/control-diffpatch`, e.g.
    /// Qwen Image / Z-Image DiffSynth patches): `image` encoded with `vae` and injected through the
    /// `MODEL_PATCH` at `strength`, optionally limited to `mask`, between the given percents.
    DiffSynthControl { patch: AddonHandle, vae: VaeHandle, image: ImageBatch, strength: f64, mask: Option<Mask>, start_percent: f64, end_percent: f64 },
    /// `AnimaLLLiteApply` (an Anima ControlNet-LLLite `MODEL_PATCH`). SwarmUI passes the
    /// generation's inpaint mask, if any.
    AnimaLLLite { patch: AddonHandle, image: ImageBatch, strength: f64, mask: Option<Mask>, start_percent: f64, end_percent: f64 },
    /// `SnapRef` (ArcAnima's SnapRef node pack, which the SwarmUI-SnapRef extension emits): a reference adapter for
    /// Anima that takes pictures of a character. See [`SnapRefPatch`].
    SnapRef(Box<SnapRefPatch>),
    /// `SwarmReferenceOnly` (Use Reference Only): the latent batch is `[reference..., latent...]`
    /// (`batch` in total). In every self-attention, each sample's keys and values are extended
    /// with the queries of the first sample of its group of `batch`, so the generation attends to
    /// the reference.
    ReferenceOnly { batch: u32 },
    /// `HyperTile` (Refiner HyperTile): split attention into `tile_size` tiles, `swap_size` tile
    /// orderings, down to `max_depth` UNet levels (`scale_depth` scales the tile size per level).
    HyperTile { tile_size: u32, swap_size: u32, max_depth: u32, scale_depth: bool },
    /// `DifferentialDiffusion`: with a soft noise mask (or regional masks), each latent pixel's
    /// denoise strength follows the mask value instead of a hard 0/1 cut. SwarmUI enables it for
    /// regional prompting, Mask Image (Mask Behavior = Differential) and segment refining.
    DifferentialDiffusion { strength: f64 },
    /// `VideoLinearCFGGuidance` (Video Min CFG, Stable Video Diffusion): CFG ramps linearly
    /// across frames from `min_cfg` (first frame) to the sampler's cfg (last frame).
    VideoLinearCfg { min_cfg: f64 },

    // ------------------------------------------------ speed / runtime options
    /// `EasyCache` (EasyCache Mode / Threshold / Start / End): reuse model outputs between similar steps.
    EasyCache { reuse_threshold: f64, start_percent: f64, end_percent: f64, verbose: bool },
    /// `ModelAttentionBackend` (Model Attention Backend): one of `EngineInfo::attention_backends`.
    /// SwarmUI's own default ("pytorch attention") arrives as the first (default) entry of that list.
    AttentionBackend { backend: String },
    /// `BlockSparseAttention` (Use Sparse Attention).
    BlockSparseAttention(SparseAttention),
    /// `TorchCompileModel` (Torch Compile: `inductor` / `cudagraphs`). An engine may map this to
    /// its own graph compilation, or ignore it.
    Compile { backend: String },
    /// `SwarmModelTiling` (Seamless Tileable): circular padding in the diffusion model so the
    /// output tiles. Pairs with [`VaePatch::Tiling`].
    Tiling { axis: TileAxis },
}

/// The `SnapRef` node's inputs, resolved. The pictures are as the node takes them: one frame per input (every frame
/// with `use_batch`), pictures of weight 0 dropped; the engine shrinks each to `ref_megapixels` and VAE-encodes it.
#[derive(Clone, Debug)]
pub struct SnapRefPatch {
    /// The SnapRef weight file (`adapter`, from the LoRA folder).
    pub adapter: ModelFile,
    /// The VAE that encodes the pictures.
    pub vae: VaeHandle,
    /// `ref_caption` encoded by the node's CLIP: the reference stream's prompt.
    pub caption: Conditioning,
    /// Each picture (one frame) and its weight (`weight_1`..`weight_3`, a batch sharing its input's).
    pub references: Vec<(ImageBatch, f64)>,
    /// `cs` (character and style) or `char` (character only).
    pub mode: String,
    pub ref_in_uncond: bool,
    pub ref_megapixels: f64,
    /// `strength`: a log-scale bias on every reference key's attention logit (0 = as trained).
    pub strength: f64,
    /// Below 1 the negative prompt's evaluations read the references too, at `ln(1 - fidelity)`. Ignored with
    /// `ref_in_uncond` or reference guidance.
    pub fidelity: f64,
    /// `multi_layout` `stitch`: several pictures side by side as one reference, instead of one each.
    pub stitch: bool,
    /// `ref_guidance_on`: `ref_guidance`, `correction` (`apg`, `rescale`, `none`) and `ref_end`. SwarmUI never sends it.
    pub ref_guidance: Option<(f64, String, f64)>,
}

/// Which axes Seamless Tileable wraps around (`tile_axis`: `Both` / `X` / `Y`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileAxis {
    Both,
    X,
    Y,
}

impl TileAxis {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Both" => Some(Self::Both),
            "X" => Some(Self::X),
            "Y" => Some(Self::Y),
            _ => None,
        }
    }
}

/// `BlockSparseAttention` settings. SwarmUI's Use Sparse Attention values map to:
/// `sol` -> `SolAttn { tau: 1.3 }`, `topk` -> `Sla { keep_percent: 15 }`, `vsa` -> `Vsa { keep_percent: 10 }`
/// (with `start_percent` 0 and `extra_tokens` 0 for `vsa`, else 0.2 / 256).
#[derive(Clone, Debug)]
pub struct SparseAttention {
    pub method: SparseAttentionMethod,
    /// Attention stays dense before this fraction of sampling.
    pub start_percent: f64,
    pub end_percent: f64,
    /// Transformer blocks that always run dense, e.g. `"0, 1, 47-49"`.
    pub dense_blocks: String,
    /// Sequences shorter than this stay dense.
    pub min_tokens: u32,
    /// Extra top-scoring tokens each query block attends (ignored for VSA).
    pub extra_tokens: u32,
    /// MiniMax H3 only: `exact_kv`, `exact_kv_and_rows` or `off`.
    pub sink_conditioning: String,
    pub verbose: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SparseAttentionMethod {
    /// `sol-attn`: training-free adaptive threshold (in score-distribution sigmas).
    SolAttn { tau: f64 },
    /// `sla`: keep a fixed percentage of key blocks (for SLA-trained weights).
    Sla { keep_percent: f64 },
    /// `vsa`: video sparse attention (FastVideo-style weights).
    Vsa { keep_percent: f64 },
}

/// VAE modifications (`SwarmTileableVAE`), applied in order at encode/decode time.
#[derive(Clone, Debug)]
pub enum VaePatch {
    /// `SwarmTileableVAE` (Seamless Tileable): circular padding in the VAE.
    Tiling { axis: TileAxis },
}

#[derive(Clone, Debug)]
pub struct LoraHook {
    pub file: ModelFile,
    pub strength_model: f32,
    pub strength_clip: f32,
}

#[derive(Clone, Debug)]
pub enum TextEncoderPatch {
    /// `LoraLoader` (clip half).
    Lora { file: ModelFile, strength: f32 },
    /// `SetClipHooks` with a `CreateHookLora` chain.
    Hooks { hooks: Vec<LoraHook>, apply_to_conds: bool, schedule_clip: bool },
    /// `CLIPSetLastLayer` (CLIP Stop At Layer): use the output of this layer, counted from the
    /// end (`-1` = last, `-2` = penultimate, ...).
    StopAtLayer { layer: i32 },
}
