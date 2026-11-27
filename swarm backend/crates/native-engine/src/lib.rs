//! [`Engine`] implementation on the diffusion library: text-to-image for int8 ConvRot checkpoints of
//! Krea 2, Anima, MOSS Image, Qwen Image 2.1 and SenseNova U1.6, fp16 SD 1.x and SDXL checkpoints, and text-to-video with audio for MiniMax H3, with every ComfyUI
//! sampler and scheduler, any CFG and LoRAs, plus the prompt images and reference
//! latents of Krea 2 and of Qwen Image 2.1's edits, and MiniMax H3's reference pictures and keyframes. The image models also sample partially, from
//! a latent and with a noise mask: init images, refiners and inpainting. Each
//! file's architecture is read from its keys, so one backend serves them all.
//!
//! Text encoding is deferred to `sample`, so the negative prompt SwarmUI always sends costs nothing
//! at CFG 1. Everything the pipeline can't do is refused with [`EngineError::Unsupported`], which
//! SwarmUI shows to the user, instead of being ignored and producing a different image.

mod latent;
mod noise;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use diffusion::h3;
use diffusion::h3vae;
use diffusion::h3window;
use diffusion::pipeline::{
    AnimaRequest, Cancelled, Cond, Generator, H3Conds, H3Request, Inpaint, KreaEntry, KreaRequest, LoadOptions, MossRequest, Prompt, Qwen21Request, Reference, Region, Sampling, SdRequest,
    SenseNovaRequest, SeedVrRequest, SizeCond, SnapRefRequest, Step, StyleRequest,
};
use diffusion::nvvfx;
use diffusion::pixal3d;
use diffusion::triposplat;
use diffusion::preview;
use diffusion::samplers::{self, CfgFunction, CfgTweaks, ModelSampling, SigmaRange};
use diffusion::lora::{self, Lora};
use diffusion::tokenizer::token_weights;
use diffusion::vision::PromptImage;
use swarm_engine_api::prompt::{PromptLeaf, PromptLeafKind};
use swarm_engine_api::*;

/// Attention backends, default first. SwarmUI's "pytorch attention" maps to the first entry.
const ATTENTION: [&str; 2] = ["int8", "flash"];
/// A family's model sampling from [`diffusion::families`], ComfyUI's `sampling_settings`, used when SwarmUI sends no
/// Sigma Shift.
fn default_sampling(d: &diffusion::families::Defaults) -> ModelSampling {
    d.sampling.expect("a family whose sampling doesn't come from its checkpoint")
}
/// [`TextEncoderHandle::encoder_type`] of Krea 2's Qwen3-VL 4B, SwarmUI's own type.
const KREA_ENCODER: &str = "krea2";
/// [`TextEncoderHandle::encoder_type`] of Anima's Qwen3 0.6B; SwarmUI's own type is the generic
/// `stable_diffusion`.
const ANIMA_ENCODER: &str = "anima";
const MOSS_ENCODER: &str = "moss_image";
/// [`TextEncoderHandle::encoder_type`] of Qwen Image 2.1's Qwen3-VL 8B; SwarmUI's own type is `qwen_image`, which Qwen
/// Image 1 shares.
const QWEN21_ENCODER: &str = "qwen_image_2.1";
const MINIMAX_ENCODER: &str = "minimax";
/// [`TextEncoderHandle::encoder_type`] of a SenseNova checkpoint's text tower, which only `CheckpointLoaderSimple` hands
/// out.
const SENSENOVA_ENCODER: &str = "sensenova";
/// [`TextEncoderHandle::encoder_type`] of an SD 1.x or SDXL checkpoint's own CLIP text encoders, which only
/// `CheckpointLoaderSimple` hands out.
const SD1_ENCODER: &str = "sd1";
const SDXL_ENCODER: &str = "sdxl";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Krea2,
    Anima,
    Moss,
    H3,
    /// MiniMax-H3-Image: H3's transformer without audio, on its own sigma grid
    H3Image,
    Qwen21,
    /// SeedVR2 3B, the restoration model SwarmUI's SeedVR stage runs
    SeedVr2,
    /// RelayVSR's keyframe model (Wan 2.2 TI2V 5B), the video upscaler of the RelayVSR extension's stage
    RelayVsr,
    /// SD 1.x and SDXL base checkpoints, with their own CLIP and VAE
    Sd,
    /// SenseNova U1.6: its text tower is in the same file, and it samples pixels
    SenseNova,
}

impl Family {
    /// (latent channels, pixels per latent)
    fn latent(self) -> (usize, usize) {
        match self {
            Family::Moss => (diffusion::moss::CH, 16),
            Family::Qwen21 => (diffusion::qwen21::CH, diffusion::vae21::SCALE),
            Family::Sd => (diffusion::unet::CH, 8),
            Family::SenseNova => (3, 1),
            _ => (16, 8),
        }
    }

    /// The families whose text encoders take the prompt as one piece, with images.
    fn one_piece(self) -> bool {
        matches!(self, Family::Krea2 | Family::Qwen21)
    }

    /// The families that read reference latents.
    fn references(self) -> bool {
        matches!(self, Family::Krea2 | Family::Qwen21)
    }
}

/// Why a SeedVR2 latent batch is refused.
const SEEDVR_BATCH: &str = "SeedVR2 on a batch of more than one clip latent (a batch of pictures is one clip after SeedVR2Preprocess)";

/// The VRAM ComfyUI keeps free without `--reserve-vram`.
pub const COMFY_HEADROOM_MB: usize = diffusion::comfy_memory::DEFAULT_HEADROOM_MB;

pub struct NativeOptions {
    /// VRAM (MB) to leave untouched for the desktop and other applications.
    pub reserve_vram_mb: usize,
    /// `Some(headroom MB)`: ComfyUI's DynamicVRAM memory management, the default; `None` is the legacy planner
    /// (`--legacy-offload`).
    pub comfy_memory: Option<usize>,
    /// Largest side (pixels, at 1:1) of a generation.
    pub max_side: usize,
    pub threads: usize,
    /// Send latent previews while sampling.
    pub live_previews: bool,
    /// Quantize bf16 MOSS Image blocks to int8 ConvRot as they load (the default); false runs them in bf16.
    pub int8_weights: bool,
    /// Faster paths whose images aren't byte-identical to the default ones (`--lossy`, see `LoadOptions::lossy`).
    pub lossy: bool,
    /// Sol³ sparse attention on every MiniMax H3 generation (`--sol3`).
    pub sol3: Option<Sol3Options>,
}

/// MagoNodes' `MagoSol3Attention` inputs, applied to every MiniMax H3 model.
#[derive(Clone, Copy, Debug)]
pub struct Sol3Options {
    pub dense_steps: usize,
    pub dense_first_blocks: usize,
    pub min_tokens: usize,
}

impl Sol3Options {
    fn sparse(&self) -> h3::SparseAttn {
        h3::SparseAttn {
            method: h3::SparseMethod::Sol3 { dense_steps: self.dense_steps, dense_first_blocks: self.dense_first_blocks },
            // set from the schedule by `SparseAttn::for_schedule`
            sigma_start: f32::INFINITY,
            sigma_end: f32::NEG_INFINITY,
            min_tokens: self.min_tokens,
            dense_blocks: Vec::new(),
            sinks: h3::SparseSinks::ExactKvAndRows,
            extra_tokens: 0,
            verbose: true,
        }
    }
}

impl Default for NativeOptions {
    fn default() -> Self {
        Self { reserve_vram_mb: 512, comfy_memory: Some(COMFY_HEADROOM_MB), max_side: 2048, threads: 6, live_previews: false, int8_weights: true, lossy: false, sol3: None }
    }
}

pub struct NativeEngine {
    gen: Mutex<Generator>,
    live_previews: bool,
    threads: usize,
    /// `--sol3`: the sparse attention every MiniMax H3 generation runs.
    sol3: Option<h3::SparseAttn>,
    /// The last RTX Video Super Resolution effect, kept while its mode and sizes repeat: loading takes most of a second.
    vsr: Mutex<Option<nvvfx::SuperRes>>,
}

/// The VFX SDK's quality level for one pass.
fn vsr_mode(mode: NvidiaVsrMode) -> nvvfx::Mode {
    use nvvfx::Mode as M;
    use NvidiaVsrLevel as L;
    match mode {
        NvidiaVsrMode::Upscale(l) => match l {
            L::Low => M::Low,
            L::Medium => M::Medium,
            L::High => M::High,
            L::Ultra => M::Ultra,
        },
        NvidiaVsrMode::Deblur(l) => match l {
            L::Low => M::DeblurLow,
            L::Medium => M::DeblurMedium,
            L::High => M::DeblurHigh,
            L::Ultra => M::DeblurUltra,
        },
    }
}

/// What `ImageUpscaleWithModel` does with NVIDIA's model, which has no fixed factor: 4x at the best level, like the
/// usual ESRGAN models, so SwarmUI's resize to the target size afterwards only ever shrinks.
const VSR_MODEL_SCALE: u32 = 4;

/// RGB f32 `[3, frames, h, w]` in 0..1 to an image batch of `frames` images.
fn clip_images(clip: &h3vae::Clip, frames: usize) -> ImageBatch {
    let (h, w) = (clip.h, clip.w);
    let mut rgb = vec![0u8; frames * h * w * 3];
    for c in 0..3 {
        for f in 0..frames {
            for i in 0..h * w {
                rgb[(f * h * w + i) * 3 + c] = (clip.data[((c * clip.t + f) * h) * w + i] * 255.0).round() as u8;
            }
        }
    }
    ImageBatch::from_u8(frames as u32, h as u32, w as u32, 3, rgb)
}

/// The prompt to encode, or why this conditioning can't be used. Only checked when a prompt is
/// really needed: at CFG 1 the negative is never read.
struct DeferredCond {
    text_encoder: PathBuf,
    /// The prompt's pieces over sampling progress (one, unless it has `<alternate>` or `<fromto>`).
    prompt: Result<Vec<Piece>, String>,
    /// Krea 2, Qwen Image 2.1 and MiniMax H3: prompt images for the vision tower.
    images: Vec<PromptImage>,
    /// `LoraLoader`'s clip half.
    loras: Vec<Lora>,
    /// MiniMax H3: reference pictures, sized and encoded when the model they're for samples.
    minimax: Option<MiniMaxReferences>,
    /// SD 1.x and SDXL: `CLIPSetLastLayer`'s layer.
    clip_layer: Option<i32>,
    /// SDXL: the size conditioning `CLIPTextEncodeSDXL` or `SwarmTextEncodeAdvanced` set; `None` takes the latent's.
    sdxl_size: Option<SdxlSize>,
    /// SwarmTextEncodeAdvanced's prompt, which it tokenizes leaf by leaf (`combine_leaves(per_leaf=True)`).
    leaves: bool,
}

/// Sampled latents, `[B, 16, H/8, W/8]`, or `[B, 128, H/16, W/16]` for MOSS Image.
struct HostLatent {
    data: Vec<f32>,
    /// MiniMax H3 with a pi-H3 DWD adapter: the frames its pixel head decoded at the last step. Any video decode of
    /// this latent returns them, so SwarmUI's VAE Decode serves DWD runs as it serves the VAE's.
    dwd: Option<ImageBatch>,
}

impl HostLatent {
    fn new(data: Vec<f32>) -> Self {
        HostLatent { data, dwd: None }
    }
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

/// The safetensors header, as JSON. Family checks ask for the same files' headers on every request, so parsed headers
/// are kept, each until its file's size or modification time changes.
fn read_header(path: &Path) -> anyhow::Result<Arc<serde_json::Value>> {
    type Key = (PathBuf, u64, Option<std::time::SystemTime>);
    static CACHE: Mutex<Vec<(Key, Arc<serde_json::Value>)>> = Mutex::new(Vec::new());
    const KEEP: usize = 32;
    let mut f = std::fs::File::open(path)?;
    let meta = f.metadata()?;
    let key: Key = (path.to_path_buf(), meta.len(), meta.modified().ok());
    if let Some((_, h)) = CACHE.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key) {
        return Ok(h.clone());
    }
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let mut h = vec![0u8; u64::from_le_bytes(n) as usize];
    f.read_exact(&mut h)?;
    let h = Arc::new(serde_json::from_slice(&h)?);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.retain(|(k, _)| k.0 != key.0);
    if cache.len() >= KEEP {
        cache.remove(0);
    }
    cache.push((key, Arc::clone(&h)));
    Ok(h)
}

/// True for a MOSS Image file whose blocks are int8 ConvRot.
fn is_int8_moss(file: &ModelFile) -> EngineResult<bool> {
    Ok(read_header(&file.path).map_err(internal)?.get("layers.0.proj.qkv.weight_scale").is_some())
}

/// The file's architecture. Refuses other models and formats before any weights are touched.
fn check_dit(file: &ModelFile) -> EngineResult<Family> {
    if h3::is_hyperflow_diffusers(&file.path) {
        return Err(unsupported(format!("{}: {}", file.name, h3::HYPERFLOW_DIFFUSERS)));
    }
    if let Some(c) = h3::hyperflow_config(&file.path) {
        if c["quantization"]["method"] != "int8_convrot" {
            return Err(unsupported(format!("{}: this backend only runs the int8 ConvRot HyperFlow H3 Flash release", file.name)));
        }
        return Ok(Family::H3);
    }
    let h = read_header(&file.path).map_err(|e| unsupported(format!("{}: not a readable safetensors file ({e})", file.name)))?;
    if diffusion::seedvr::is_seedvr2_7b(|k| h.get(k).is_some()) {
        return Err(unsupported(format!("{}: SeedVR2 7B (this backend runs the 3B int8 ConvRot release)", file.name)));
    }
    if diffusion::seedvr::is_seedvr2(|k| h.get(k).is_some()) {
        if h.get("blocks.0.attn.proj_qkv.vid.weight_scale").is_none() || h["blocks.0.attn.proj_qkv.vid.weight"]["dtype"] != "I8" {
            return Err(unsupported(format!("{}: this backend only runs the int8 ConvRot SeedVR2 3B release", file.name)));
        }
        return Ok(Family::SeedVr2);
    }
    if let Some(p) = diffusion::relayvsr::key_prefix(|k| h.get(k).is_some()) {
        if h.get(format!("{p}blocks.0.self_attn.q.weight_scale")).is_none() || h[format!("{p}blocks.0.self_attn.q.weight")]["dtype"] != "I8" {
            return Err(unsupported(format!("{}: this backend only runs the int8 ConvRot RelayVSR releases", file.name)));
        }
        return Ok(Family::RelayVsr);
    }
    if let Some(p) = diffusion::qwen21::key_prefix(|k| h.get(k).is_some()) {
        if h.get(format!("{p}transformer_blocks.0.img_mlp.gate_up.weight_scale")).is_none() {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot Qwen Image 2.1 checkpoints", file.name)));
        }
        return Ok(Family::Qwen21);
    }
    if let Some(p) = diffusion::sensenova::key_prefix(|k| h.get(k).is_some()) {
        let q = &h[format!("{p}language_model.model.layers.0.self_attn.q_proj_mot_gen.weight")];
        if !p.is_empty() || q["dtype"] != "I8" || h.get(format!("{p}language_model.model.layers.0.self_attn.q_proj_mot_gen.weight_scale")).is_none() {
            return Err(unsupported(format!("{}: this backend only runs the int8 ConvRot SenseNova U1.6 release, as published", file.name)));
        }
        return Ok(Family::SenseNova);
    }
    if let Some(p) = diffusion::anima::key_prefix(|k| h.get(k).is_some()) {
        if h.get(format!("{p}blocks.0.mlp.layer1.weight_scale")).is_none() {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot Anima checkpoints", file.name)));
        }
        return Ok(Family::Anima);
    }
    if diffusion::moss::is_moss(|k| h.get(k).is_some()) {
        let bf16 = h.get("layers.0.proj.qkv.weight").and_then(|t| t.get("dtype")).is_some_and(|d| d == "BF16");
        if h.get("layers.0.proj.qkv.weight_scale").is_none() && !bf16 {
            return Err(unsupported(format!("{}: this backend only runs int8 ConvRot or bf16 MOSS Image checkpoints with fused q/k/v", file.name)));
        }
        return Ok(Family::Moss);
    }
    if diffusion::pipeline::is_sd(|k| h.get(k).is_some()) {
        sd_is_xl(file, &h)?;
        return Ok(Family::Sd);
    }
    if h3::is_h3_image(|k| h.get(k).is_some()) {
        if h["blocks.0.attn.qkv_proj.weight"]["dtype"] != "I8" || h.get("blocks.0.attn.qkv_proj.weight_scale").is_none() {
            return Err(unsupported(format!("{}: this backend only runs the int8 ConvRot MiniMax-H3-Image release", file.name)));
        }
        return Ok(Family::H3Image);
    }
    if h3::is_h3(|k| h.get(k).is_some()) {
        if h.get("blocks.0.attn.qkv_proj.weight_s_rel").is_none() {
            return Err(unsupported(format!("{}: this backend only runs w6a8 MiniMax H3 checkpoints", file.name)));
        }
        return Ok(Family::H3);
    }
    // The keys SwarmUI's `isKrea2` tests.
    if ["txtfusion.projector.weight", "blocks.0.attn.gate.weight", "blocks.0.mod.lin"].iter().any(|k| h.get(k).is_none()) {
        return Err(unsupported(format!("{}: this backend only runs Krea 2, Anima, MOSS Image, Qwen Image 2.1, SenseNova U1.6, SD 1.x, SDXL, MiniMax H3 and MiniMax-H3-Image models", file.name)));
    }
    if h.get("blocks.0.attn.gate.weight_scale").is_none() {
        return Err(unsupported(format!("{}: this backend only runs W8A8 (ConvRot int8) Krea 2 checkpoints", file.name)));
    }
    Ok(Family::Krea2)
}

/// Whether an SD checkpoint is SDXL (else SD 1.x), from its header: fp16, bf16 or fp32 weights, quantized to int8 here
/// unless the server runs with `--no-quant`. SD 2.x, the SDXL refiner, inpainting UNets and quantized files are refused.
fn sd_is_xl(file: &ModelFile, h: &serde_json::Value) -> EngineResult<bool> {
    let p = diffusion::unet::key_prefix(|k| h.get(k).is_some()).ok_or_else(|| unsupported(format!("{}: no SD UNet", file.name)))?;
    let conv_in = &h[format!("{p}input_blocks.0.0.weight")];
    let dtype = conv_in["dtype"].as_str().unwrap_or("?");
    if !matches!(dtype, "F16" | "BF16" | "F32") {
        return Err(unsupported(format!("{}: {dtype} SD weights (this backend runs fp16, bf16 or fp32 SD 1.x and SDXL checkpoints, and quantizes them to int8 itself)", file.name)));
    }
    if conv_in["shape"][1] != 4 {
        return Err(unsupported(format!("{}: a UNet that takes {} input channels (inpainting and instruct models)", file.name, conv_in["shape"][1])));
    }
    let keys = h.as_object().map(|o| o.iter()).into_iter().flatten();
    let context = keys.filter(|(k, _)| k.starts_with(p) && k.ends_with("attn2.to_k.weight")).find_map(|(_, t)| t["shape"][1].as_u64());
    let adm = h.get(format!("{p}label_emb.0.0.weight")).and_then(|t| t["shape"][1].as_u64());
    match (context, adm) {
        (Some(768), None) => Ok(false),
        (Some(2048), Some(2816)) => Ok(true),
        _ => Err(unsupported(format!("{}: this UNet (this backend runs SD 1.x and SDXL base models, not SD 2.x, the SDXL refiner or others)", file.name))),
    }
}

/// [`sd_is_xl`] of a file.
fn sd_file_is_xl(file: &ModelFile) -> EngineResult<bool> {
    sd_is_xl(file, &*read_header(&file.path).map_err(internal)?)
}

/// An SD 1.x or SDXL VAE, alone or in a checkpoint: the 4-channel latent's decoder.
fn is_sd_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| {
        diffusion::sdvae::key_prefix(|k| h.get(k).is_some()).is_some_and(|p| h[format!("{p}decoder.conv_in.weight")]["shape"][1] == 4)
    })
}

/// Qwen Image 2.1's `CLIPLoader` type is Qwen Image 1's `qwen_image`; this tells the Qwen3-VL 8B by its shape and its
/// vision tower.
fn is_qwen3vl_8b(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| {
        h["model.layers.0.input_layernorm.weight"]["shape"][0] == 4096 && h.get("model.visual.deepstack_merger_list.0.linear_fc1.weight").is_some()
    })
}

/// Qwen Image 2.1's VAE: the Wan 2.2 layout with a one-frame kernel.
fn is_qwen21_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| {
        h.get("decoder.upsamples.0.upsamples.0.residual.2.weight").is_some() && h["decoder.head.2.weight"]["shape"].as_array().is_some_and(|s| s.len() == 5 && s[2] == 1)
    })
}

/// Anima's `CLIPLoader` type is the generic `stable_diffusion`; this tells its encoder by shape.
fn is_qwen3_600m(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| h["model.layers.0.input_layernorm.weight"]["shape"][0] == 1024)
}

/// `<break>` chunks, each (text, weight) segments.
type Chunks = Vec<Vec<(String, f32)>>;

/// A prompt over a range of sampling progress: SwarmTextEncodeAdvanced's conditioning entries.
#[derive(Clone, Debug, PartialEq)]
struct Piece {
    start_percent: f64,
    end_percent: f64,
    chunks: Chunks,
}

impl Piece {
    fn whole(chunks: Chunks) -> Self {
        Piece { start_percent: 0.0, end_percent: 1.0, chunks }
    }
}

/// A prompt's pieces, as SwarmTextEncodeAdvanced builds them: one for the whole run, or with `<alternate>` and
/// `<fromto>` one per run of steps whose leaves are the same, its range widened by 0.001 on each side (the first from
/// -0.001, the last to 1). Or why it needs something the pipeline doesn't have.
fn prompt_pieces(p: &ParsedPrompt, steps: u32, family: Family) -> Result<Vec<Piece>, String> {
    let steps = steps.max(1);
    if !p.has_steps() {
        return Ok(vec![Piece::whole(prompt_chunks(&p.flatten(0, steps), family)?)]);
    }
    let mut pieces = Vec::new();
    let (mut last, mut start) = (p.flatten(0, steps), 0.0);
    for i in 1..steps {
        let perc = i as f64 / steps as f64;
        let leaves = p.flatten(i, steps);
        if leaves != last {
            pieces.push(Piece { start_percent: start - 0.001, end_percent: perc + 0.001, chunks: prompt_chunks(&last, family)? });
            (last, start) = (leaves, perc);
        }
    }
    pieces.push(Piece { start_percent: start - 0.001, end_percent: 1.0, chunks: prompt_chunks(&last, family)? });
    Ok(pieces)
}

/// One step's leaves as `<break>` chunks of (text, weight) segments, or why the family can't take them.
fn prompt_chunks(leaves: &[PromptLeaf], family: Family) -> Result<Chunks, String> {
    let mut chunks = vec![Vec::new()];
    for leaf in leaves {
        if leaf.kind == PromptLeafKind::Break {
            chunks.push(Vec::new());
        } else {
            chunks.last_mut().unwrap().push(leaf.clone());
        }
    }
    // Anima's joined conditioning keeps only the first chunk's T5 tokens, which carry the weights; ComfyUI weights a
    // later chunk's Qwen3 states instead
    if family == Family::Anima && chunks[1..].iter().any(|c| chunk_segments(c, family).is_ok_and(|s| s.iter().any(|(_, w)| *w != 1.0))) {
        return Err("prompt weights after a <break>".into());
    }
    // SwarmTextEncodeAdvanced joins the chunks' states but keeps the first chunk's token tags, and ComfyUI's model then
    // fails reading tags past them
    if chunks.len() != 1 && matches!(family, Family::H3 | Family::H3Image) {
        return Err("<break> on MiniMax H3 (ComfyUI's model fails on the joined chunks: their token tags cover only the first)".into());
    }
    chunks.iter().map(|leaves| chunk_segments(leaves, family)).collect()
}

fn chunk_segments(leaves: &[PromptLeaf], family: Family) -> Result<Vec<(String, f32)>, String> {
    let mut text = String::new();
    let mut segments = Vec::new();
    for leaf in leaves {
        match &leaf.kind {
            PromptLeafKind::Text(t) if family == Family::Anima => segments.extend(token_weights(t, leaf.weight as f32)),
            // SenseNova's tokenizer gives every token weight 1, and its prompts are keys and values ComfyUI never weights
            PromptLeafKind::Text(_) if family == Family::SenseNova && leaf.weight != 1.0 => return Err("prompt weights on SenseNova".into()),
            PromptLeafKind::Text(t) if family == Family::SenseNova => text.push_str(t),
            // one segment per leaf, tokenized apart and weighted by the pipeline, as SwarmTextEncodeAdvanced does for a tokenizer without weights
            PromptLeafKind::Text(t) if matches!(family, Family::Krea2 | Family::Qwen21) => segments.push((t.clone(), leaf.weight as f32)),
            PromptLeafKind::Text(t) if family.one_piece() => text.push_str(t),
            // SwarmTextEncodeAdvanced tokenizes each piece with weights disabled: parentheses are literal, escaped ones lose
            // their backslash
            PromptLeafKind::Text(t) if family == Family::Sd => segments.push((t.replace("\\)", ")").replace("\\(", "("), leaf.weight as f32)),
            PromptLeafKind::Text(t) => segments.push((t.clone(), leaf.weight as f32)),
            PromptLeafKind::Embed(_) => return Err("embeddings in prompts".into()),
            PromptLeafKind::LoraHook(_) => return Err("LoRA hooks in prompts".into()),
            PromptLeafKind::Break => unreachable!("chunks() splits on breaks"),
        }
    }
    if family == Family::SenseNova || family.one_piece() && !matches!(family, Family::Krea2 | Family::Qwen21) {
        segments.push((text, 1.0));
    }
    Ok(segments)
}

fn encode_request(req: &TextEncodeRequest, family: Family) -> Result<Vec<Piece>, String> {
    encode_chunks(req, family).map(|c| match c {
        Encoded::Whole(chunks) => vec![Piece::whole(chunks)],
        Encoded::Pieces(pieces) => pieces,
    })
}

/// A text encode's prompt: plain text is one piece; SwarmTextEncodeAdvanced's can have several.
enum Encoded {
    Whole(Chunks),
    Pieces(Vec<Piece>),
}

fn encode_chunks(req: &TextEncodeRequest, family: Family) -> Result<Encoded, String> {
    for p in &req.text_encoder.patches {
        match p {
            TextEncoderPatch::Lora { .. } => {}
            TextEncoderPatch::Hooks { .. } => return Err("LoRA hooks".into()),
            TextEncoderPatch::StopAtLayer { .. } if family == Family::Sd => {}
            TextEncoderPatch::StopAtLayer { .. } => return Err("CLIP Stop At Layer".into()),
        }
    }
    match &req.mode {
        // Qwen tokenizers run with weights disabled, Anima's T5 tokens take ComfyUI's emphasis.
        TextEncodeMode::Plain { text } if family == Family::Anima => Ok(Encoded::Whole(vec![token_weights(text, 1.0)])),
        // MOSSImageTokenizer trims a plain prompt; SwarmTextEncodeAdvanced's leaves skip that.
        TextEncodeMode::Plain { text } if family == Family::Moss => Ok(Encoded::Whole(vec![vec![(text.trim().to_string(), 1.0)]])),
        // CLIPTextEncode parses ComfyUI's (word:1.2) emphasis
        TextEncodeMode::Plain { text } if family == Family::Sd => Ok(Encoded::Whole(vec![token_weights(text, 1.0)])),
        TextEncodeMode::Sdxl { text_g, text_l, .. } if family == Family::Sd => {
            if text_g != text_l {
                return Err("different text_g and text_l prompts".into());
            }
            Ok(Encoded::Whole(vec![token_weights(text_g, 1.0)]))
        }
        TextEncodeMode::Sdxl { .. } => Err("CLIPTextEncodeSDXL on a model other than SDXL".into()),
        TextEncodeMode::Plain { text } => Ok(Encoded::Whole(vec![vec![(text.clone(), 1.0)]])),
        TextEncodeMode::MiniMax { text, .. } if family == Family::H3 => Ok(Encoded::Whole(vec![vec![(text.clone(), 1.0)]])),
        TextEncodeMode::MiniMax { .. } => Err("MiniMax H3 pictures on another text encoder".into()),
        TextEncodeMode::Swarm(s) => {
            if s.images.is_some() && (!family.one_piece() && family != Family::H3 || family == Family::SenseNova) {
                return Err("Prompt Images".into());
            }
            if s.minimax_refs.is_some() && family != Family::H3 || s.clip_vision.is_some() {
                return Err("image references".into());
            }
            if s.lora_hooks.as_ref().is_some_and(|h| !h.is_empty()) {
                return Err("LoRA hooks".into());
            }
            if s.guidance.is_some() {
                return Err("Flux Guidance Scale".into());
            }
            // The node only uses the template with images (SwarmUI sends `krea2` to MOSS Image too), and Krea 2's is
            // the one its encoder has. Qwen Image 2.1 builds its own.
            let template_used = s.images.is_some() && s.llama_template.is_some();
            // without `krea2` the node neither appends the images nor uses Krea 2's template
            if family == Family::Krea2 && s.images.is_some() && s.llama_template.as_deref() != Some("krea2") {
                return Err("Prompt Images without the krea2 prompt template".into());
            }
            if template_used && !(family == Family::Krea2 && s.llama_template.as_deref() == Some("krea2")) {
                return Err(format!("prompt template {:?}", s.llama_template));
            }
            let pieces = prompt_pieces(&s.prompt, s.steps, family)?;
            // Qwen Image 2.1's text encoder sees the images with every chunk, and the first chunk's image slots are kept
            if pieces.iter().any(|p| p.chunks.len() != 1) && s.images.is_some() && family != Family::Qwen21 {
                return Err("<break> with Prompt Images".into());
            }
            Ok(Encoded::Pieces(pieces))
        }
    }
}

/// The first frame of a batch as RGB, its alpha dropped as `image[..., :3]` does.
fn first_rgb(b: &ImageBatch) -> PromptImage {
    let channels = b.channels as usize;
    let rgb = b.frame_f32(0).chunks_exact(channels).flat_map(|px| [0, 1, 2].map(|c| px[c.min(channels - 1)])).collect();
    PromptImage { width: b.width as usize, height: b.height as usize, rgb }
}

/// Every frame of a batch as RGB, its alpha dropped as `frames[..., :3]` does.
fn rgb_frames(b: &ImageBatch) -> Vec<PromptImage> {
    let channels = b.channels as usize;
    (0..b.count)
        .map(|i| PromptImage {
            width: b.width as usize,
            height: b.height as usize,
            rgb: b.frame_f32(i).chunks_exact(channels).flat_map(|px| [0, 1, 2].map(|c| px[c.min(channels - 1)])).collect(),
        })
        .collect()
}

/// True for an H3 VAE file with the encoder, which both the video and the image VAE carry.
fn has_h3_encoder(file: &ModelFile) -> bool {
    is_h3_vae(file) && read_header(&file.path).is_ok_and(|h| h.get("encoder.conv_in.weight").is_some())
}

/// The reference pictures and keyframes of a MiniMax H3 prompt for a target of shape `sh`.
fn h3_conds(cond: &DeferredCond, meta: &CondMeta, family: Family, sh: &h3::Shape) -> EngineResult<H3Conds> {
    let mut out = H3Conds::default();
    if !cond.images.is_empty() {
        // SwarmTextEncodeAdvanced's images: shown to the text encoder as they are, with no reference latents
        if cond.minimax.is_some() {
            return Err(unsupported("Prompt Images and MiniMax H3 references at once"));
        }
        if family != Family::H3 {
            return Err(unsupported("Prompt Images on MiniMax-H3-Image (it takes MiniMaxH3ImageConditioning's pictures)"));
        }
        out.label = "<Picture {}>: ".into();
        out.pictures = cond.images.iter().map(|img| diffusion::pipeline::H3Picture { vision: img.clone(), latent: None }).collect();
    }
    if let Some(r) = &cond.minimax {
        if !r.audios.is_empty() || r.videos.iter().any(|(_, track)| track.is_some()) {
            return Err(unsupported("reference audio and reference video soundtracks on MiniMax H3 (pictures and silent videos only)"));
        }
        if r.ref_image_size != "match" && r.ref_image_size != "max" {
            return Err(unsupported(format!("MiniMax H3 ref_image_size {:?}", r.ref_image_size)));
        }
        let vae = match &r.vae {
            Some(v) => {
                let file = v.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE for MiniMax H3 reference pictures"))?;
                if !has_h3_encoder(file) {
                    return Err(unsupported(format!("{} for MiniMax H3 reference pictures (it takes an H3 VAE)", file.name)));
                }
                Some(file.path.clone())
            }
            None => None,
        };
        let images: Vec<PromptImage> = r.images.iter().filter(|b| b.count > 0).map(first_rgb).collect();
        (out.label, out.pictures) = match (family, r.style) {
            (Family::H3, MiniMaxRefStyle::Video) => {
                ("<Picture {}>: ".into(), diffusion::pipeline::ref2va_pictures(&images, (r.width as usize, r.height as usize), r.ref_image_size == "max", vae.is_some()))
            }
            (Family::H3Image, MiniMaxRefStyle::Image) => ("Picture {}: ".into(), diffusion::pipeline::image_pictures(&images, vae.is_some())),
            (Family::H3Image, _) => return Err(unsupported("MiniMax H3 video references on MiniMax-H3-Image: set the model's class to MiniMax H3 Image in SwarmUI")),
            _ => return Err(unsupported("MiniMaxH3ImageConditioning's pictures on the MiniMax H3 video model")),
        };
        // the videos are cut to the clip's length, as SwarmUI's node gives it
        let clip = diffusion::pipeline::h3_clip_frames(r.length.max(1) as usize);
        for (frames, _) in r.videos.iter().filter(|(b, _)| b.count > 0) {
            let v = diffusion::pipeline::ref2va_video(&rgb_frames(frames), clip, vae.is_some()).map_err(|e| unsupported(format!("{e:#}")))?;
            out.videos.push(v);
        }
        out.vae = vae;
    }
    // the clip's frames on the 17k + 5 grid, or one for a still
    let frames = if sh.t == 1 { 1 } else { (sh.t.max(2) - 2) / 5 * 17 + 5 };
    for k in &meta.minimax_keyframes {
        if family != Family::H3 {
            return Err(unsupported("keyframes on MiniMax-H3-Image"));
        }
        let LatentSamples::Single(t) = &k.latent.samples else { return Err(unsupported("an audio/video keyframe latent")) };
        if t.shape != [1, 24, 1, sh.h, sh.w] {
            return Err(unsupported(format!("a keyframe latent of shape {:?} for a {}x{} clip", t.shape, sh.w * 16, sh.h * 16)));
        }
        if k.frame_index as usize >= frames {
            return Err(unsupported(format!("a keyframe at frame {} of a {frames} frame clip", k.frame_index)));
        }
        let z = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("a keyframe latent this engine didn't encode"))?;
        out.keyframes.push((k.frame_index as usize, z.data.clone()));
    }
    Ok(out)
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
        && (family.references() || m.reference_latents.is_empty())
        && m.reference_images.is_empty()
        && m.gligen.is_empty()
        && m.inpaint.is_none()
        && m.embedding_ops.is_empty()
        && m.unclip.is_empty()
        && m.controls.is_empty();
    if !plain {
        return Err(unsupported("conditioning features (regions, references, ControlNet, style models, ...)"));
    }
    let cond = e.embedding.downcast_ref::<DeferredCond>().ok_or_else(|| unsupported("conditioning from another text encoder"))?;
    let chunks = match cond.prompt.clone().map_err(unsupported)?.as_slice() {
        [p] if p.start_percent == 0.0 && p.end_percent == 1.0 => p.chunks.clone(),
        _ => return Err(unsupported("<alternate> and <fromto> prompts on this model")),
    };
    Ok((cond, Prompt { chunks, scale: if m.zeroed { 0.0 } else { m.multiplier as f32 } }))
}

/// A side's conditioning entries, for the families that mix several as ComfyUI's `calc_cond_batch` does: SwarmUI's
/// `<alternate>` and `<fromto>` steps, `ConditioningSetTimestepRange` and regional prompts (`ConditioningSetMask`). Each
/// comes with its text encoder's conditioning and its metadata; masks are resized to the `(lh, lw)` latent.
fn conds(c: &Conditioning, family: Family, (lh, lw): (usize, usize)) -> EngineResult<Vec<(&DeferredCond, &CondMeta, Cond)>> {
    if c.entries.is_empty() {
        return Err(unsupported("a conditioning without entries"));
    }
    let mut out = Vec::new();
    for e in &c.entries {
        let m = &e.meta;
        let plain = (family.references() || m.reference_latents.is_empty())
            && (family == Family::SenseNova || m.reference_images.is_empty())
            && m.gligen.is_empty()
            && m.inpaint.is_none()
            && m.embedding_ops.is_empty()
            && m.unclip.is_empty()
            && m.controls.is_empty();
        if !plain {
            return Err(unsupported("conditioning features (references, ControlNet, style models, GLIGEN, ...)"));
        }
        let cond = e.embedding.downcast_ref::<DeferredCond>().ok_or_else(|| unsupported("conditioning from another text encoder"))?;
        let pieces = cond.prompt.clone().map_err(unsupported)?;
        let mask = m.mask.as_ref().map(|cm| region_mask(cm, lh, lw)).transpose()?;
        let scale = if m.zeroed { 0.0 } else { m.multiplier as f32 };
        // ConditioningSetTimestepRange sets the range of every entry it gets, a stepped prompt's too
        let ranged = m.start_percent != 0.0 || m.end_percent != 1.0;
        for p in pieces {
            let (start_percent, end_percent) = if ranged { (m.start_percent, m.end_percent) } else { (p.start_percent, p.end_percent) };
            let region = Region { start_percent, end_percent, mask: mask.clone() };
            out.push((cond, m, Cond { prompt: Prompt { chunks: p.chunks, scale }, region }));
        }
    }
    Ok(out)
}

/// `ConditioningSetMask`'s mask at the latent's size, times its strength: `resolve_areas_and_cond_masks_multidim`
/// resizes it bilinearly (no corner alignment, no antialiasing) as sampling begins, and `get_area_and_mult` scales it.
fn region_mask(cm: &CondMask, lh: usize, lw: usize) -> EngineResult<Vec<f32>> {
    if cm.set_area_to_bounds {
        return Err(unsupported("ConditioningSetMask's mask bounds area"));
    }
    if cm.mask.count != 1 {
        return Err(unsupported(format!("a batch of {} region masks", cm.mask.count)));
    }
    let shape = [1, 1, cm.mask.height as usize, cm.mask.width as usize];
    let mask = diffusion::latent_ops::common_upscale(&cm.mask.data, shape, lw, lh, "bilinear", false).map_err(internal)?;
    let strength = cm.strength as f32;
    Ok(mask.into_iter().map(|v| v * strength).collect())
}

/// The conditioning entries of a MiniMax H3 or MiniMax-H3-Image request on a `(lh, lw)` latent of shape `sh`: the
/// positive's, each with its pictures and keyframes, and the negative's when the sampling reads them, which take
/// neither. Returns the text encoder they share.
fn h3_entries<'a>(req: &'a SampleRequest, family: Family, sh: &h3::Shape, (lh, lw): (usize, usize), uses_negative: bool)
    -> EngineResult<(&'a DeferredCond, Vec<Cond>, Vec<H3Conds>, Vec<Cond>)> {
    let positive = conds(&req.positive, family, (lh, lw))?;
    let negative = if uses_negative { conds(&req.negative, family, (lh, lw))? } else { Vec::new() };
    // process_conds resizes masks against the packed audio/video latent's one dimension, which fails
    if family == Family::H3 && positive.iter().chain(&negative).any(|(_, m, _)| m.mask.is_some()) {
        return Err(unsupported("regional prompts (ConditioningSetMask) on MiniMax H3: ComfyUI can't fit a mask to its joint audio/video latent"));
    }
    let first = positive[0].0;
    for (cond, ..) in positive.iter().chain(&negative) {
        if cond.text_encoder != first.text_encoder {
            return Err(unsupported("MiniMax H3 conditioning entries from different text encoders"));
        }
        if !cond.loras.is_empty() {
            return Err(unsupported("LoRAs on MiniMax H3's text encoder"));
        }
    }
    if negative.iter().any(|(cond, ..)| cond.minimax.is_some() || !cond.images.is_empty()) {
        return Err(unsupported("pictures on the negative prompt of MiniMax H3"));
    }
    let mut pictures = Vec::with_capacity(positive.len());
    for (cond, meta, _) in &positive {
        pictures.push(h3_conds(cond, meta, family, sh)?);
    }
    let positive = positive.into_iter().map(|(.., c)| c).collect();
    Ok((first, positive, pictures, negative.into_iter().map(|(.., c)| c).collect()))
}

/// Refuses what H3's sampling can't combine with several conditioning entries or with pictures and keyframes.
fn h3_combinations(windows: Option<&h3window::ContextWindows>, sparse: bool, sh: &h3::Shape, positive: &[Cond], pictures: &[H3Conds], negative: &[Cond])
    -> EngineResult<()> {
    let windowed = windows.filter(|w| w.applies(sh));
    if windowed.is_some_and(|w| w.split_conds) && (positive.len() > 1 || negative.len() > 1) {
        return Err(unsupported("split_conds_to_windows with several conditioning entries (<alternate>, <fromto>, timestep ranges) on MiniMax H3"));
    }
    let conditioned = pictures.iter().any(|c| !c.keyframes.is_empty() || c.pictures.iter().any(|p| p.latent.is_some()));
    if conditioned && (windowed.is_some() || sparse) {
        return Err(unsupported("MiniMax H3 keyframes and reference pictures with context windows or sparse attention"));
    }
    Ok(())
}

/// One conditioning entry's reference latents (`ReferenceLatent` appends to every entry), in `family`'s latent.
fn entry_references(m: &CondMeta, family: Family) -> EngineResult<Vec<Reference>> {
    let mut out = Vec::with_capacity(m.reference_latents.len());
    for l in &m.reference_latents {
        let LatentSamples::Single(t) = &l.samples else {
            return Err(unsupported("audio/video reference latents"));
        };
        let &[1, channels, height, width] = t.shape.as_slice() else {
            return Err(unsupported(format!("reference latent of shape {:?}", t.shape)));
        };
        if channels != family.latent().0 {
            return Err(unsupported(format!("reference latent of shape {:?}", t.shape)));
        }
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("a reference latent this engine didn't encode"))?;
        out.push(Reference { latent: latent.data.clone(), height, width });
    }
    Ok(out)
}

/// A SenseNova entry's reference pictures, each frame of a batch one (`split_reference_batches`), RGB as
/// `preprocess_reference` takes it: the first three channels, or fewer repeated round.
fn sensenova_references(m: &CondMeta) -> EngineResult<Vec<diffusion::sensenova::Reference>> {
    let mut out = Vec::new();
    for b in &m.reference_images {
        if b.width < 16 || b.height < 16 {
            return Err(unsupported(format!("a {}x{} reference picture on SenseNova (16 pixels or more a side)", b.width, b.height)));
        }
        let c = b.channels as usize;
        for f in 0..b.count {
            let rgb = b.frame_f32(f).chunks_exact(c).flat_map(|px| [px[0], px[1 % c], px[2 % c]]).collect();
            out.push(diffusion::sensenova::Reference { width: b.width as usize, height: b.height as usize, rgb });
        }
    }
    Ok(out)
}

/// A Qwen Image 2.1 entry's reference latents. Its model has no reference method, so the method is ignored.
fn qwen21_references(m: &CondMeta) -> EngineResult<Vec<Reference>> {
    entry_references(m, Family::Qwen21)
}

/// Krea 2's reference latents of one conditioning entry, and whether they take `index_timestep_zero`.
fn krea_references(m: &CondMeta) -> EngineResult<(Vec<Reference>, bool)> {
    let timestep_zero = match m.reference_latents_method.as_deref() {
        // the model reads them only with a method, as ComfyUI's Krea 2 has no default one
        None => return Ok((Vec::new(), false)),
        Some("index") => false,
        Some("index_timestep_zero") => true,
        Some(method) => return Err(unsupported(format!("reference latent method {method}"))),
    };
    Ok((entry_references(m, Family::Krea2)?, timestep_zero))
}

/// `SwarmAttnTokenWeights` on Qwen Image 2.1: the first `<break>` chunk of its conditioning's first entry, whose weights
/// SwarmTextEncodeAdvanced recorded, and that entry's prompt image count.
fn qwen21_attn_token_weights(c: &Conditioning) -> EngineResult<Option<(Vec<(String, f32)>, usize)>> {
    let Some(e) = c.entries.first() else { return Ok(None) };
    let cond = e.embedding.downcast_ref::<DeferredCond>().ok_or_else(|| unsupported("SwarmAttnTokenWeights on conditioning from another text encoder"))?;
    let pieces = cond.prompt.clone().map_err(unsupported)?;
    Ok(pieces.into_iter().next().and_then(|p| p.chunks.into_iter().next()).map(|chunk| (chunk, cond.images.len())))
}

/// `SwarmAttnTokenWeights` on Krea 2: its conditioning's first entry, whose first chunk's `attn_token_weights` pairs the
/// pipeline works out. `None` when that entry has no weights, so the node leaves the model alone.
fn krea_attn_token_weights(c: &Conditioning, positive: &DeferredCond) -> EngineResult<Option<KreaEntry>> {
    let e = c.entries.first().ok_or_else(|| unsupported("SwarmAttnTokenWeights with an empty conditioning"))?;
    let cond = e.embedding.downcast_ref::<DeferredCond>().ok_or_else(|| unsupported("SwarmAttnTokenWeights on conditioning from another text encoder"))?;
    if cond.text_encoder != positive.text_encoder {
        return Err(unsupported("SwarmAttnTokenWeights on another text encoder's conditioning"));
    }
    let pieces = cond.prompt.clone().map_err(unsupported)?;
    let piece = pieces.into_iter().next().ok_or_else(|| unsupported("SwarmAttnTokenWeights with an empty prompt"))?;
    // only SwarmTextEncodeAdvanced records weights
    if !cond.leaves || piece.chunks.iter().flatten().all(|(_, w)| *w == 1.0) {
        return Ok(None);
    }
    let mut entry = KreaEntry::new(Prompt { chunks: piece.chunks, scale: 1.0 }, cond.images.clone());
    entry.leaves = true;
    Ok(Some(entry))
}

/// A picture's RGB as f32 [h][w][3]: the node takes the first three channels as they are.
fn rgb_f32(b: &ImageBatch) -> Vec<f32> {
    let c = b.channels as usize;
    b.frame_f32(0).chunks_exact(c).flat_map(|px| [px[0], px[1.min(c - 1)], px[2.min(c - 1)]]).collect()
}

/// The positive and negative prompts share one baked text encoder.
fn check_text_encoder_loras(positive: &DeferredCond, negative: &DeferredCond) -> EngineResult<()> {
    if positive.loras != negative.loras {
        return Err(unsupported("different LoRAs on the positive and negative text encoders"));
    }
    Ok(())
}

/// `SwarmAttnTokenWeights` on SD 1.x and SDXL. SwarmTextEncodeAdvanced only gives a prompt attention weights when its
/// tokenizer drops ComfyUI's `(x:2)` weights, and SD's CLIP keeps them, so a prompt from the checkpoint's CLIP has
/// none and the patch does nothing. Another text encoder's could have some, which ComfyUI's UNet can't take (its
/// `attn1_patch` returns three tensors, `attn1_token_weight_patch` a dict): refused.
fn sd_attn_token_weights(patch: Option<&Conditioning>, checkpoint: &Path) -> EngineResult<()> {
    let Some(e) = patch.and_then(|c| c.entries.first()) else {
        return Ok(());
    };
    match e.embedding.downcast_ref::<DeferredCond>() {
        Some(c) if c.text_encoder == checkpoint => Ok(()),
        _ => Err(unsupported("SwarmAttnTokenWeights with a prompt from another text encoder than the SD checkpoint's CLIP")),
    }
}

/// SD's entries of a side, each with the size conditioning of the prompt it came from: SDXL's `encode_adm` runs per
/// entry, on its own pooled output and sizes, and a prompt without sizes takes the latent's (`encode_model_conds`).
fn sd_entries(side: Vec<(&DeferredCond, &CondMeta, Cond)>, (lh, lw): (usize, usize)) -> (Vec<Cond>, Vec<SizeCond>) {
    let size = |c: &DeferredCond| match c.sdxl_size {
        Some(s) => SizeCond {
            width: s.width as f32,
            height: s.height as f32,
            crop_w: s.crop_w as f32,
            crop_h: s.crop_h as f32,
            target_width: s.target_width as f32,
            target_height: s.target_height as f32,
        },
        None => SizeCond::of(lw * 8, lh * 8),
    };
    side.into_iter().map(|(d, _, c)| (c, size(d))).unzip()
}

/// SwarmUI puts LoRAs that aren't confined to a section on every model it loads, SeedVR2's too, and ComfyUI skips the
/// keys that name nothing in it. So a LoRA for another model is skipped here too, and one that would patch SeedVR2 is
/// refused.
fn check_seedvr_loras(model: &Path, loras: &[Lora]) -> EngineResult<()> {
    if loras.is_empty() {
        return Ok(());
    }
    let st = diffusion::st::SafeTensors::open(model).map_err(internal)?;
    for l in loras {
        let name = l.path.file_name().unwrap_or(l.path.as_os_str()).to_string_lossy();
        let matched = lora::matched_modules(lora::Arch::SeedVr2, &st, "", &l.path).map_err(internal)?;
        if matched > 0 {
            return Err(unsupported(format!("LoRAs on SeedVR2 ({name} patches {matched} of its modules)")));
        }
        tracing::warn!("SeedVR2: skipping {name}, which patches none of its modules");
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
    /// `DifferentialDiffusion`'s strength, for noise masks.
    differential: Option<f32>,
    /// `BlockSparseAttention`, on H3.
    sparse: Option<h3::SparseAttn>,
    /// `MiniMaxH3ContextWindows`.
    windows: Option<h3window::ContextWindows>,
    /// `SnapRef`, on Anima.
    snapref: Option<Box<SnapRefPatch>>,
    /// `BrokeAnimaStyleReference`, on Anima and Krea 2.
    style: Option<Box<StyleRefPatch>>,
    /// SD: `ModelSamplingDiscrete`'s prediction type, in place of the checkpoint's.
    v_pred: Option<bool>,
    /// `SwarmAttnTokenWeights`: the conditioning whose first entry's per-token weights apply inside attention, on the
    /// models whose ComfyUI attention takes `attn1_patch`.
    attn_token_weights: Option<Conditioning>,
}

/// MiniMax-H3-Image's `MiniMaxH3ImageModelSampling`: the sigma grid its heads were distilled on, from the file.
fn h3_image_sampling(path: &Path) -> EngineResult<ModelSampling> {
    let st = diffusion::st::SafeTensors::open(path).map_err(internal)?;
    let grid: Vec<f32> = h3::image_sigmas(&st).map_err(|e| unsupported(e.to_string()))?.iter().map(|g| *g as f32).collect();
    Ok(ModelSampling::Grid(samplers::SigmaGrid::new(&grid).map_err(internal)?))
}

fn model_settings(model: &ModelHandle, family: Family) -> EngineResult<ModelSettings> {
    let model_sampling = match family {
        Family::H3Image => h3_image_sampling(&model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?.path)?,
        Family::Krea2 => default_sampling(&diffusion::families::KREA2),
        Family::Qwen21 => default_sampling(&diffusion::families::QWEN21),
        Family::Anima => default_sampling(&diffusion::families::ANIMA),
        Family::Moss => default_sampling(&diffusion::families::MOSS),
        Family::H3 => default_sampling(&diffusion::families::H3),
        // ComfyUI's SeedVR2 sampling_settings
        Family::SeedVr2 => ModelSampling::Flow { shift: 1.0 },
        // WAN22_RelayVSR keeps WAN21_T2V's sampling_settings
        Family::RelayVsr => ModelSampling::Flow { shift: 8.0 },
        Family::Sd => default_sampling(&diffusion::families::SD1),
        Family::SenseNova => ModelSampling::SenseNova { shift: diffusion::sensenova::SHIFT as f64 },
    };
    let mut s = ModelSettings { model_sampling, int8_attention: true,loras: Vec::new(), audio_shift: h3::SHIFT_AUDIO, tweaks: CfgTweaks::default(), differential: None, sparse: None, windows: None, snapref: None, style: None, v_pred: None, attn_token_weights: None };
    let mut sparse = None;
    for p in &model.patches {
        match p {
            ModelPatch::SamplingFlux { max_shift, base_shift, width, height } if matches!(family, Family::Krea2 | Family::Qwen21) => {
                s.model_sampling = ModelSampling::Flux { shift: flux_shift(*max_shift, *base_shift, *width, *height) }
            }
            ModelPatch::SenseNovaSampling { shift } if family == Family::SenseNova => s.model_sampling = ModelSampling::SenseNova { shift: *shift },
            ModelPatch::SamplingAuraFlow { shift, sampling } if !matches!(family, Family::Krea2 | Family::Sd | Family::SenseNova) && sampling == "flow" => {
                s.model_sampling = ModelSampling::Flow { shift: *shift }
            }
            ModelPatch::SamplingDiscrete { sampling, zsnr } if family == Family::Sd => {
                if *zsnr {
                    return Err(unsupported("ModelSamplingDiscrete's zsnr"));
                }
                s.v_pred = Some(match sampling.as_str() {
                    "eps" => false,
                    "v_prediction" => true,
                    other => return Err(unsupported(format!("ModelSamplingDiscrete sampling {other}"))),
                });
            }
            ModelPatch::MiniMaxH3Shift { shift_video, shift_audio } if family == Family::H3 => {
                s.model_sampling = ModelSampling::Flow { shift: *shift_video };
                s.audio_shift = *shift_audio;
            }
            ModelPatch::MiniMaxH3ContextWindows { context_length, context_overlap, fuse_method, freenoise, split_conds_to_windows } => {
                if family != Family::H3 {
                    return Err(unsupported("context windows on this model (only MiniMax H3 takes them)"));
                }
                // split_conds_to_windows only picks among several conditioning entries, refused in sample_h3
                let fuse = h3window::Fuse::parse(fuse_method).ok_or_else(|| EngineError::InvalidInput(format!("fuse_method {fuse_method}")))?;
                let mut w = h3window::ContextWindows::from_frames(*context_length as usize, *context_overlap as usize, fuse, *freenoise)
                    .map_err(|e| EngineError::InvalidInput(e.to_string()))?;
                w.split_conds = *split_conds_to_windows;
                s.windows = Some(w);
            }
            ModelPatch::BlockSparseAttention(sp) => {
                if let SparseAttentionMethod::Vsa { .. } = sp.method {
                    return Err(unsupported(
                        "VSA sparse attention: it needs FastH3's VSA-trained weights and kernel. Use Sol-Attn or Top-K (SLA) instead",
                    ));
                }
                if family != Family::H3 {
                    return Err(unsupported("sparse attention on this model (only MiniMax H3 takes it)"));
                }
                sparse = Some(sp.clone());
            }
            ModelPatch::AttentionBackend { backend } => {
                s.int8_attention = match backend.as_str() {
                    "int8" => true,
                    "flash" => false,
                    b => return Err(unsupported(format!("attention backend {b}"))),
                }
            }
            // H3's latent packs two streams, which ComfyUI's patches read as nested tensors. RescaleCFG's other branch,
            // for EPS and v-prediction models, isn't ported.
            ModelPatch::RescaleCfg { multiplier } if !matches!(family, Family::H3 | Family::Sd) => {
                s.tweaks.function = Some(CfgFunction::Rescale { multiplier: *multiplier as f32 })
            }
            ModelPatch::RenormCfg { cfg_trunc, renorm_cfg } if family != Family::H3 => {
                s.tweaks.function = Some(CfgFunction::Renorm { cfg_trunc: *cfg_trunc as f32, renorm_cfg: *renorm_cfg as f32 })
            }
            ModelPatch::CfgZeroStar if family != Family::H3 => s.tweaks.zero_star = true,
            ModelPatch::Tcfg if family != Family::H3 => s.tweaks.tangential = true,
            // only read with a noise mask, which H3 refuses
            ModelPatch::DifferentialDiffusion { strength } => s.differential = Some(*strength as f32),
            // Weighted prompts are refused in encode_text; with none this patch has no effect.
            ModelPatch::AttnTokenWeights { conditioning } => s.attn_token_weights = Some((**conditioning).clone()),
            ModelPatch::Compile { .. } => {}
            ModelPatch::Lora { file, strength } => s.loras.push(Lora { path: file.path.clone(), strength: *strength }),
            ModelPatch::SnapRef(p) => {
                if family != Family::Anima {
                    return Err(unsupported("SnapRef on this model (it is an Anima adapter)"));
                }
                if s.snapref.is_some() {
                    return Err(unsupported("more than one SnapRef node on a model"));
                }
                if p.ref_guidance.is_some() {
                    return Err(unsupported("SnapRef's ref_guidance_on (its extra model call per step)"));
                }
                s.snapref = Some(p.clone());
            }
            ModelPatch::StyleReference(p) => {
                if !matches!(family, Family::Anima | Family::Krea2) {
                    return Err(unsupported("Broke Anima Style Reference on this model (it works with Anima and Krea 2)"));
                }
                if s.style.is_some() {
                    return Err(unsupported("more than one Broke Anima Style Reference node on a model"));
                }
                s.style = Some(p.clone());
            }
            other => {
                let name = format!("{other:?}");
                return Err(unsupported(format!("model option {}", name.split([' ', '{', '(']).next().unwrap_or("?"))));
            }
        }
    }
    // after the loop: the window is in the model's sigmas, which a later shift patch may change
    s.sparse = sparse.map(|sp| sparse_attention(&sp, s.model_sampling)).transpose()?;
    Ok(s)
}

/// `BlockSparseAttention`'s settings for the H3 model, its window turned into sigmas as ComfyUI's `percent_to_sigma`.
fn sparse_attention(sp: &SparseAttention, ms: ModelSampling) -> EngineResult<h3::SparseAttn> {
    let method = match sp.method {
        SparseAttentionMethod::SolAttn { tau } => h3::SparseMethod::Sol { tau: tau as f32 },
        SparseAttentionMethod::Sla { keep_percent } => h3::SparseMethod::TopK { ratio: keep_percent / 100.0 },
        SparseAttentionMethod::Vsa { .. } => unreachable!("refused above"),
    };
    let sinks = match sp.sink_conditioning.as_str() {
        "off" => h3::SparseSinks::Off,
        "exact_kv" => h3::SparseSinks::ExactKv,
        "exact_kv_and_rows" => h3::SparseSinks::ExactKvAndRows,
        other => return Err(EngineError::InvalidInput(format!("sink_conditioning {other}"))),
    };
    if sp.extra_tokens > 256 || sp.extra_tokens % 64 != 0 {
        return Err(EngineError::InvalidInput(format!("extra_tokens {} (0 to 256, in steps of 64)", sp.extra_tokens)));
    }
    Ok(h3::SparseAttn {
        method,
        sigma_start: ms.percent_to_sigma(sp.start_percent),
        sigma_end: ms.percent_to_sigma(sp.end_percent),
        min_tokens: sp.min_tokens as usize,
        dense_blocks: block_list(&sp.dense_blocks),
        sinks,
        extra_tokens: sp.extra_tokens as usize,
        verbose: sp.verbose,
    })
}

/// ComfyUI's `parse_block_list`: the matches of `\d+\s*-\s*\d+|\d+`, so `"0, 1, 47-49"` is blocks 0, 1 and 47 to 49.
fn block_list(text: &str) -> Vec<usize> {
    let b = text.as_bytes();
    let number = |i: &mut usize| {
        let start = *i;
        while *i < b.len() && b[*i].is_ascii_digit() {
            *i += 1;
        }
        text[start..*i].parse::<usize>().unwrap_or(usize::MAX)
    };
    let skip_space = |i: &mut usize| {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    let (mut out, mut i) = (Vec::new(), 0);
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let first = number(&mut i);
        let mut j = i;
        skip_space(&mut j);
        if j < b.len() && b[j] == b'-' {
            j += 1;
            skip_space(&mut j);
            if j < b.len() && b[j].is_ascii_digit() {
                let last = number(&mut j);
                let (lo, hi) = (first.min(last), first.max(last));
                out.extend(lo..=hi.min(lo.saturating_add(1 << 16)));
                i = j;
                continue;
            }
        }
        out.push(first);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Everything in a sampler request the pipeline can't run is refused here.
fn check_sample(req: &SampleRequest) -> EngineResult<()> {
    if !samplers::SAMPLERS.contains(&req.sampler.as_str()) {
        return Err(unsupported(format!("sampler {}", req.sampler)));
    }
    if !samplers::SCHEDULERS.contains(&req.scheduler.as_str()) {
        return Err(unsupported(format!("scheduler {}", req.scheduler)));
    }
    if req.tile_size.is_some() {
        return Err(unsupported("tiled sampling"));
    }
    Ok(())
}

/// The sigmas a run takes: `KSampler.set_steps` for `denoise`, then the step range of `PatchedKSampler.sample`. `None`
/// when no step is left, and the latent comes back as it is.
fn schedule(req: &SampleRequest, settings: &ModelSettings, (lh, lw): (usize, usize)) -> EngineResult<Option<Vec<f32>>> {
    let range = req.sigma_min.zip(req.sigma_max).map(|(sigma_min, sigma_max)| SigmaRange { sigma_min, sigma_max, rho: req.rho });
    let schedule = |steps| samplers::sigmas(settings.model_sampling, &req.sampler, &req.scheduler, steps, (lw, lh), range);
    let (start, end) = (req.start_at_step as usize, req.end_at_step as usize);
    samplers::ksampler_sigmas(req.steps as usize, req.denoise, start, end, req.return_with_leftover_noise, schedule).map_err(internal)
}

/// `comfy.utils.reshape_mask`: a noise mask resized to the latent `[batch, channels, lh, lw]`, bilinearly.
fn denoise_mask(m: &Mask, format: LatentFormat, (batch, channels, lh, lw): (usize, usize, usize, usize)) -> EngineResult<Vec<f32>> {
    let (count, h, w) = (m.count as usize, m.height as usize, m.width as usize);
    if m.data.len() != count * h * w {
        return Err(EngineError::InvalidInput(format!("a noise mask of {} values for {count}x{h}x{w}", m.data.len())));
    }
    // ComfyUI's image latents of these models are 5D, where a batch of masks is resized across the batch
    if count > 1 && format != LatentFormat::Flux2 {
        return Err(unsupported("a batch of different noise masks on this model's latents"));
    }
    let resized = diffusion::latent_ops::interpolate(&m.data, [count, 1, h, w], lh, lw, "bilinear").map_err(internal)?;
    let planes = diffusion::latent_ops::repeat_to_batch(&resized, [count, 1, lh, lw], batch);
    Ok(planes.chunks_exact(lh * lw).flat_map(|p| std::iter::repeat_n(p, channels).flatten().copied()).collect())
}

/// What the images of a batch share: the sigmas, and where sampling starts.
struct Plan {
    sigmas: Vec<f32>,
    /// The batch's latent at the sampler's scale; `None` when empty.
    latent: Option<Vec<f32>>,
    /// The batch's noise mask, resized to the latent.
    mask: Option<Vec<f32>>,
    differential: Option<f32>,
}

impl Plan {
    fn new(sigmas: Vec<f32>, latent: &Latent, tensor: &LatentTensor, settings: &ModelSettings) -> EngineResult<Self> {
        let &[batch, channels, lh, lw] = tensor.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", tensor.shape)));
        };
        let values = match tensor.data.downcast_ref::<HostLatent>() {
            Some(h) if h.data.len() == batch * channels * lh * lw => Some(h.data.clone()),
            Some(h) => return Err(EngineError::InvalidInput(format!("a latent of {} values for shape {:?}", h.data.len(), tensor.shape))),
            None if tensor.data.is_none() => None,
            None => return Err(unsupported("a latent this engine didn't make")),
        };
        let mask = latent.noise_mask.as_ref().map(|m| denoise_mask(m, tensor.format, (batch, channels, lh, lw))).transpose()?;
        Ok(Plan { sigmas, latent: values, mask, differential: settings.differential })
    }

    /// The noise of the batch's images one after another: SwarmKSampler's, or zeros without `add_noise`.
    fn noise(&self, req: &SampleRequest, batch: usize, (channels, lh, lw): (usize, usize, usize)) -> Vec<f32> {
        if !req.add_noise {
            return vec![0.0; batch * channels * lh * lw];
        }
        (0..batch).flat_map(|i| noise::swarm_noise(req.noise_seed, req.variation, i as u64, channels, lh, lw)).collect()
    }
}

/// The sampling of a batch of `batch` images, all together.
fn sampling(req: &SampleRequest, settings: &ModelSettings, plan: &Plan, batch: usize) -> EngineResult<Sampling> {
    Ok(Sampling {
        sampler: req.sampler.clone(),
        model_sampling: settings.model_sampling,
        sigmas: plan.sigmas.clone(),
        cfg: req.cfg,
        seed: req.noise_seed,
        batch_index: 0,
        batch_size: batch,
        images: batch,
        tweaks: settings.tweaks,
        latent: plan.latent.clone(),
        inpaint: plan.mask.as_ref().map(|m| Inpaint { mask: m.clone(), differential: plan.differential }),
    })
}

/// Which image of a step over images `first..first + images` of `batch` SwarmKSampler's `previews` shows, if any.
fn preview_image(mode: PreviewMode, (first, images): (usize, usize), batch: usize, step: usize) -> Option<usize> {
    let shown = match mode {
        PreviewMode::None => return None,
        PreviewMode::One => 0,
        PreviewMode::Second => 1 % batch,
        // Iterate goes through the images a step at a time
        PreviewMode::Iterate => first + step % images,
        PreviewMode::Default | PreviewMode::Animate => first,
    };
    (first..first + images).contains(&shown).then(|| shown - first)
}

/// Sends a step's progress within the whole batch, and its preview. False once cancelled.
fn report(mode: PreviewMode, progress: &mut dyn ProgressSink, batch: usize, step: Step, image: impl FnOnce(&[f32]) -> (Vec<u8>, usize, usize)) -> bool {
    progress.progress((step.first * step.total + step.index * step.images) as u32, (batch * step.total) as u32);
    if let Some(i) = preview_image(mode, (step.first, step.images), batch, step.index) {
        let per = step.denoised.len() / step.images.max(1);
        if let Some(x0) = step.denoised.get(i * per..(i + 1) * per) {
            let (rgb, h, w) = image(x0);
            progress.preview(PreviewImage::Rgb(ImageBatch::from_u8(1, h as u32, w as u32, 3, rgb)));
        }
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
            int8_weights: o.int8_weights,
            lossy: o.lossy,
            comfy_memory: o.comfy_memory,
        };
        Ok(Self {
            gen: Mutex::new(Generator::new(&load)?),
            live_previews: o.live_previews,
            threads: o.threads,
            sol3: o.sol3.map(|s| s.sparse()),
            vsr: Mutex::new(None),
        })
    }

    /// Frees the cached RTX Video Super Resolution effect.
    fn drop_vsr(&self) {
        self.vsr.lock().unwrap_or_else(|e| e.into_inner()).take();
    }

    /// One RTX Video Super Resolution pass of `mode` over `images`, to `width` x `height`. If the effect can't load or
    /// run, which is most often the GPU being full, the image models leave VRAM and it's tried once more.
    fn run_vsr(&self, model: &Path, mode: nvvfx::Mode, images: &ImageBatch, (width, height): (u32, u32), progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        if images.channels != 3 {
            return Err(unsupported(format!("RTX Video Super Resolution of images with {} channels: it takes RGB", images.channels)));
        }
        if images.count == 0 {
            return Err(EngineError::InvalidInput("RTX Video Super Resolution without images".into()));
        }
        let input = (images.width as usize, images.height as usize);
        let output = (width as usize, height as usize);
        let mut slot = self.vsr.lock().unwrap_or_else(|e| e.into_inner());
        let attempt = |slot: &mut Option<nvvfx::SuperRes>, progress: &mut dyn ProgressSink| -> anyhow::Result<Option<Vec<u8>>> {
            if !slot.as_ref().is_some_and(|s| s.is(model, mode, input, output)) {
                // the old effect goes first, so its memory is free for the new one
                *slot = None;
                *slot = Some(nvvfx::SuperRes::new(model, mode, input, output)?);
            }
            let sr = slot.as_mut().expect("loaded above");
            let mut out = Vec::with_capacity(images.count as usize * output.0 * output.1 * 3);
            for f in 0..images.count {
                if progress.interrupted() {
                    return Ok(None);
                }
                let frame = match &images.data {
                    PixelData::U8(_) => sr.run(nvvfx::Frame::U8(&images.frame_u8(f)))?,
                    PixelData::F32(_) => sr.run(nvvfx::Frame::F32(&images.frame_f32(f)))?,
                };
                out.extend_from_slice(&frame);
                progress.progress(f + 1, images.count);
            }
            Ok(Some(out))
        };
        let out = match attempt(&mut slot, progress) {
            Ok(out) => out,
            Err(e) => {
                tracing::warn!("native: RTX Video Super Resolution failed ({e:#}); unloading the image models and trying again");
                *slot = None;
                self.gen().unload().map_err(internal)?;
                attempt(&mut slot, progress).map_err(internal)?
            }
        };
        let out = out.ok_or(EngineError::Interrupted)?;
        Ok(ImageBatch::from_u8(images.count, height, width, 3, out))
    }

    /// SeedVR2's restore of the picture or clip `SeedVR2Conditioning` carries: a clip (`[1, 16, T, H/8, W/8]`) is one
    /// sample over all its frames, as in ComfyUI.
    fn sample_seedvr(&self, req: &SampleRequest, settings: &ModelSettings, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        check_seedvr_loras(&file.path, &settings.loras)?;
        let LatentSamples::Single(tensor) = &req.latent.samples else {
            return Err(unsupported("SeedVR2 on an audio/video latent"));
        };
        let (batch, frames, lh, lw) = match *tensor.shape.as_slice() {
            [b, 16, lh, lw] => (b, 1, lh, lw),
            [b, 16, t, lh, lw] if t > 0 => (b, t, lh, lw),
            _ => return Err(unsupported(format!("SeedVR2 on a latent of shape {:?}", tensor.shape))),
        };
        if batch != 1 {
            return Err(unsupported(SEEDVR_BATCH));
        }
        if tensor.format != LatentFormat::SeedVr2 {
            return Err(unsupported("SeedVR2 on a latent its VAE didn't encode"));
        }
        if frames > 1 && req.latent.noise_mask.is_some() {
            return Err(unsupported("a noise mask on a SeedVR2 clip"));
        }
        if frames > 1 && req.variation.is_some() {
            return Err(unsupported("Variation Seed on a SeedVR2 clip"));
        }
        let condition = seedvr_condition(&req.positive, &tensor.shape)?.to_vec();
        let finish = |data| {
            let t = LatentTensor { format: LatentFormat::SeedVr2, shape: tensor.shape.clone(), data: Opaque::new(HostLatent::new(data)) };
            Ok(req.latent.with_samples(LatentSamples::Single(t)))
        };
        // [16, frames, lh, lw] in the order of a [16, frames * lh, lw] picture, which is how the sampler sees it
        let flat = LatentTensor { format: tensor.format, shape: vec![1, 16, frames * lh, lw], data: tensor.data.clone() };
        let Some(sigmas) = schedule(req, settings, (lh, lw))? else {
            let plan = Plan::new(Vec::new(), &req.latent, &flat, settings)?;
            return finish(plan.latent.unwrap_or_else(|| vec![0.0; 16 * frames * lh * lw]));
        };
        let plan = Plan::new(sigmas, &req.latent, &flat, settings)?;
        let sampling = sampling(req, settings, &plan, 1)?;
        if sampling.uses_negative() && seedvr_condition(&req.negative, &tensor.shape)? != condition.as_slice() {
            return Err(unsupported("different pictures on SeedVR2's positive and negative conditioning"));
        }
        let noise = plan.noise(req, 1, (16, frames * lh, lw));
        let r = SeedVrRequest { width: lw * 8, height: lh * 8, frames, noise, sampling, condition };
        let mut gen = self.gen();
        gen.ensure_seedvr(&file.path).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        // SeedVR2's latents have no preview colours in ComfyUI
        let (latent, stats) = gen.sample_seedvr(&r, |step| report(PreviewMode::None, progress, 1, step, |_| (Vec::new(), 0, 0))).map_err(internal)?;
        tracing::info!("seedvr2: {} latent frames of {}x{}: sampling {:.0} ms", r.frames, r.width, r.height, stats.sample_ms);
        finish(latent)
    }

    /// SeedVR2's VAE on a picture, or on a batch of frames as one clip (`[1, 16, T, H/8, W/8]`), as ComfyUI's.
    fn encode_seedvr(&self, file: &ModelFile, pixels: &ImageBatch) -> EngineResult<Latent> {
        let (w, h) = (pixels.width as usize, pixels.height as usize);
        if w % 16 != 0 || h % 16 != 0 {
            return Err(unsupported(format!("SeedVR2 VAE encode of a {w}x{h} picture (SeedVR2Preprocess pads to multiples of 16)")));
        }
        let frames = pixels.count as usize;
        let mut gen = self.gen();
        gen.ensure_seedvr_vae(&file.path).map_err(internal)?;
        let c = pixels.channels as usize;
        let rgb: Vec<f32> = (0..pixels.count)
            .flat_map(|f| pixels.frame_f32(f).chunks_exact(c).flat_map(|px| [px[0], px[1.min(c - 1)], px[2.min(c - 1)]]).collect::<Vec<_>>())
            .collect();
        let (data, lt) = gen.encode_seedvr(&rgb, frames, w, h).map_err(internal)?;
        let shape = if frames == 1 { vec![1, 16, h / 8, w / 8] } else { vec![1, 16, lt, h / 8, w / 8] };
        let tensor = LatentTensor { format: LatentFormat::SeedVr2, shape, data: Opaque::new(HostLatent::new(data)) };
        let mut info = PixelInfo::image(w as u32, h as u32, 1);
        info.frames = pixels.count;
        Ok(Latent::single(tensor, info))
    }

    /// RelayVSR's keyframe latent from the keyframes `RelayVSRConditioning` carries: one evaluation of the model over
    /// every keyframe, as in ComfyUI.
    fn sample_relayvsr(&self, req: &SampleRequest, settings: &ModelSettings, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        if !settings.loras.is_empty() {
            return Err(unsupported("LoRAs on RelayVSR"));
        }
        let LatentSamples::Single(tensor) = &req.latent.samples else {
            return Err(unsupported("RelayVSR on an audio/video latent"));
        };
        let &[1, ch, keys, lh, lw] = tensor.shape.as_slice() else {
            return Err(unsupported(format!("RelayVSR on a latent of shape {:?} (it upscales one clip from RelayVSRConditioning)", tensor.shape)));
        };
        if ch != diffusion::relayvsr::CH || tensor.format != LatentFormat::RelayVsr {
            return Err(unsupported("RelayVSR on a latent RelayVSRConditioning didn't make"));
        }
        if req.latent.noise_mask.is_some() {
            return Err(unsupported("a noise mask on RelayVSR"));
        }
        if req.variation.is_some() {
            return Err(unsupported("Variation Seed on RelayVSR"));
        }
        let cond = relayvsr_condition(&req.positive)?;
        let k = &cond.keyframes;
        if k.count as usize != keys {
            return Err(unsupported(format!("{} RelayVSR keyframes for a latent of {keys}", k.count)));
        }
        let finish = |data| {
            let t = LatentTensor { format: LatentFormat::RelayVsr, shape: tensor.shape.clone(), data: Opaque::new(HostLatent::new(data)) };
            Ok(req.latent.with_samples(LatentSamples::Single(t)))
        };
        // [48, keys, lh, lw] in the order of a [48, keys * lh, lw] picture, which is how the sampler sees it
        let flat = LatentTensor { format: tensor.format, shape: vec![1, ch, keys * lh, lw], data: tensor.data.clone() };
        let Some(sigmas) = schedule(req, settings, (lh, lw))? else {
            let plan = Plan::new(Vec::new(), &req.latent, &flat, settings)?;
            return finish(plan.latent.unwrap_or_else(|| vec![0.0; ch * keys * lh * lw]));
        };
        let plan = Plan::new(sigmas, &req.latent, &flat, settings)?;
        let sampling = sampling(req, settings, &plan, 1)?;
        // the model reads no prompt but the one stored in it, so CFG would need a negative that differs
        if sampling.uses_negative() {
            return Err(unsupported("CFG other than 1 on RelayVSR"));
        }
        let noise = plan.noise(req, 1, (ch, keys * lh, lw));
        let data = frames_f32(k);
        let r = diffusion::pipeline::RelayVsrRequest {
            keyframes: diffusion::pipeline::RelayVsrFrames { data: &data, count: keys, height: k.height as usize, width: k.width as usize, channels: k.channels as usize },
            lh,
            lw,
            sampling,
            noise,
        };
        let mut gen = self.gen();
        gen.ensure_relayvsr(&file.path).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        // one step: the bar follows the blocks; RelayVSR's latents have no preview colours in ComfyUI
        let progress = std::cell::RefCell::new(progress);
        let (latent, stats) = gen
            .sample_relayvsr(
                &r,
                |_| !progress.borrow().interrupted(),
                |done, total| {
                    let mut p = progress.borrow_mut();
                    p.progress(done as u32, total as u32);
                    if p.interrupted() { Err(Cancelled.into()) } else { Ok(()) }
                },
            )
            .map_err(internal)?;
        tracing::info!("relayvsr: {keys} keyframes of {}x{}: sampling {:.0} ms", k.width, k.height, stats.sample_ms);
        finish(latent)
    }

    fn decode_seedvr(&self, vae: &VaeHandle, t: &LatentTensor, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        let (frames, lh, lw) = match *t.shape.as_slice() {
            [1, 16, lh, lw] => (1, lh, lw),
            [1, 16, f, lh, lw] if f > 0 => (f, lh, lw),
            _ => return Err(unsupported(format!("SeedVR2 decode of a latent of shape {:?}", t.shape))),
        };
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("decoding a latent this engine didn't sample"))?;
        let file = vae.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE"))?;
        let mut gen = self.gen();
        gen.ensure_seedvr_vae(&file.path).map_err(internal)?;
        let (rgb, n) = gen.decode_seedvr(&latent.data, frames, lw * 8, lh * 8).map_err(internal)?;
        progress.progress(1, 1);
        Ok(ImageBatch::from_f32(n as u32, (lh * 8) as u32, (lw * 8) as u32, 3, rgb))
    }

    /// SwarmUI always asks for previews, so the server's option decides.
    fn preview_mode(&self, req: &SampleRequest) -> PreviewMode {
        if self.live_previews { req.previews } else { PreviewMode::None }
    }

    fn gen(&self) -> std::sync::MutexGuard<'_, Generator> {
        self.gen.lock().unwrap_or_else(|e| {
            // A job panicked while it held the generator, possibly halfway through updating a cache (modulation,
            // cross-attention k and v, baked LoRAs). Trust none of it: unload every model, and the next job loads
            // them fresh.
            let mut gen = e.into_inner();
            if let Err(e) = gen.unload() {
                tracing::warn!("native: unloading after a panic failed: {e:#}");
            }
            self.gen.clear_poison();
            gen
        })
    }

    fn load_model(&self, file: &ModelFile) -> EngineResult<ModelHandle> {
        // int8 weights are used as stored; `load_diffusion_model` refuses an fp8 cast of bf16 ones.
        let (family, name) = match check_dit(file)? {
            Family::Krea2 => (self.gen().ensure_krea(&file.path), "krea-2"),
            Family::Anima => (self.gen().ensure_anima(&file.path), "anima"),
            Family::Moss => (self.gen().ensure_moss(&file.path), "moss-image"),
            Family::H3 => (self.gen().ensure_h3(&file.path), "minimax-h3"),
            Family::H3Image => (self.gen().ensure_h3(&file.path), "minimax-h3-image"),
            Family::Qwen21 => (self.gen().ensure_qwen21(&file.path), "qwen-image-2.1"),
            Family::SenseNova => (self.gen().ensure_sensenova(&file.path), "sensenova-u1.5"),
            Family::SeedVr2 => (self.gen().ensure_seedvr(&file.path), "seedvr2"),
            Family::RelayVsr => (self.gen().ensure_relayvsr(&file.path), "relayvsr"),
            Family::Sd if sd_file_is_xl(file)? => (self.gen().ensure_sd(&file.path), "stable-diffusion-xl-v1-base"),
            Family::Sd => (self.gen().ensure_sd(&file.path), "stable-diffusion-v1"),
        };
        family.map_err(internal)?;
        Ok(ModelHandle::new(Some(file.clone()), Some(name.into()), Opaque::none()))
    }

    fn sample_krea(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, plan: &Plan, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let sampling = sampling(req, settings, plan, batch)?;
        let positive = conds(&req.positive, Family::Krea2, (lh, lw))?;
        let negative = if sampling.uses_negative() { conds(&req.negative, Family::Krea2, (lh, lw))? } else { Vec::new() };
        // one text encoder, with one set of LoRAs, encodes every entry
        let cond = positive[0].0;
        for (c, _, _) in positive.iter().chain(&negative) {
            if c.text_encoder != cond.text_encoder {
                return Err(unsupported("prompts from different text encoders"));
            }
            check_text_encoder_loras(cond, c)?;
        }
        let entry = |(c, m, cond): (&DeferredCond, &CondMeta, Cond)| -> EngineResult<KreaEntry> {
            let (references, timestep_zero) = krea_references(m)?;
            Ok(KreaEntry { cond, leaves: c.leaves, images: c.images.clone(), references, timestep_zero })
        };
        let positive = positive.into_iter().map(entry).collect::<EngineResult<Vec<_>>>()?;
        let negative = negative.into_iter().map(entry).collect::<EngineResult<Vec<_>>>()?;
        let attn_token_weights = settings.attn_token_weights.as_ref().map(|c| krea_attn_token_weights(c, cond)).transpose()?.flatten();
        let steps = sampling.sigmas.len().saturating_sub(1);
        let style = settings.style.as_deref().map(|p| self.style_request(p, cond, steps, Family::Krea2)).transpose()?;
        let mut gen = self.gen();
        gen.ensure_krea(&file.path).map_err(internal)?;
        gen.ensure_te(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;
        let r = KreaRequest {
            positive,
            negative,
            seed: req.noise_seed,
            width: lw * 8,
            height: lh * 8,
            sampling,
            noise: Some(plan.noise(req, batch, (16, lh, lw))),
            attn_token_weights,
            style,
        };
        let (latent, stats) = gen
            .sample_krea(&r, |step| report(mode, progress, batch, step, |x0| (preview::wan21(x0, lh, lw), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("krea2: {batch} image(s): text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        Ok(latent)
    }

    fn sample_anima(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, plan: &Plan, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let sampling = sampling(req, settings, plan, batch)?;
        // every entry of both sides, as calc_cond_batch mixes them; SnapRef and the style reference act on each of the
        // side's evaluations as their wrappers do on each model call
        let positive = conds(&req.positive, Family::Anima, (lh, lw))?;
        let cond = positive[0].0;
        let text_encoder = &cond.text_encoder;
        let negative = if sampling.uses_negative() { conds(&req.negative, Family::Anima, (lh, lw))? } else { Vec::new() };
        for (c, _, _) in positive.iter().chain(&negative) {
            if c.text_encoder != *text_encoder {
                return Err(unsupported("conditioning entries from different text encoders"));
            }
            check_text_encoder_loras(cond, c)?;
        }
        let positive: Vec<Cond> = positive.into_iter().map(|(_, _, c)| c).collect();
        // unread at CFG 1
        let negative: Vec<Cond> = if negative.is_empty() {
            vec![Cond::plain(Prompt::new(Vec::new()))]
        } else {
            negative.into_iter().map(|(_, _, c)| c).collect()
        };
        let snapref = settings.snapref.as_deref().map(|p| self.snapref_request(p, cond)).transpose()?;
        if snapref.is_some() && settings.style.is_some() {
            return Err(unsupported("SnapRef and Broke Anima Style Reference on the same model"));
        }
        let steps = sampling.sigmas.len().saturating_sub(1);
        let style = settings.style.as_deref().map(|p| self.style_request(p, cond, steps, Family::Anima)).transpose()?;
        let mut gen = self.gen();
        gen.ensure_anima(&file.path).map_err(internal)?;
        gen.ensure_qwen3(text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;
        let r = AnimaRequest { positive, negative, width: lw * 8, height: lh * 8, sampling, noise: plan.noise(req, batch, (16, lh, lw)), snapref, style };
        let (latent, stats) = gen
            .sample_anima(&r, |step| report(mode, progress, batch, step, |x0| (preview::wan21(x0, lh, lw), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("anima: {batch} image(s): text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        Ok(latent)
    }

    /// Qwen Image 2.1's VAE on `pixels`, RGB or RGBA: the encoder pads RGB with opaque alpha, like ComfyUI.
    /// The SnapRef node's pictures, encoded as it does (`_encode_image`, after `_stitch` with `multi_layout` stitch), and
    /// its settings. `prompt` is the positive's conditioning, whose text encoder the caption must share.
    fn snapref_request(&self, p: &SnapRefPatch, prompt: &DeferredCond) -> EngineResult<SnapRefRequest> {
        let (cc, caption) = single_prompt(&p.caption, Family::Anima)?;
        if cc.text_encoder != prompt.text_encoder {
            return Err(unsupported("a SnapRef CLIP other than the prompt's text encoder"));
        }
        check_text_encoder_loras(prompt, cc)?;
        // style needs an adapter with a style branch, which the engine checks when it loads the adapter
        if !matches!(p.mode.as_str(), "cs" | "char" | "style") {
            return Err(EngineError::InvalidInput(format!("SnapRef mode {}", p.mode)));
        }
        if !(0.25..=2.0).contains(&p.ref_megapixels) {
            return Err(EngineError::InvalidInput(format!("ref_megapixels {} (0.25 to 2)", p.ref_megapixels)));
        }
        let vae = p.vae.source.as_ref().ok_or_else(|| unsupported("SnapRef with the pixel-space VAE"))?;
        if is_qwen21_vae(vae) || is_flux2_vae(vae) || is_h3_vae(vae) {
            return Err(unsupported("SnapRef with a VAE other than Anima's"));
        }
        let mut pictures: Vec<(Vec<f32>, usize, usize)> =
            p.references.iter().map(|(img, _)| (rgb_f32(img), img.width as usize, img.height as usize)).collect();
        let mut weights: Vec<f64> = p.references.iter().map(|(_, w)| *w).collect();
        if p.stitch && pictures.len() > 1 {
            pictures = vec![diffusion::snapref::stitch(&pictures)];
            weights = vec![1.0];
        }
        let mut gen = self.gen();
        gen.ensure_vae(&vae.path).map_err(internal)?;
        let mut references = Vec::with_capacity(pictures.len());
        for (rgb, w, h) in &pictures {
            let (rgb, w, h) = diffusion::snapref::shrink(rgb, *w, *h, p.ref_megapixels);
            references.push(gen.encode_image(&rgb, w, h).map_err(internal)?);
        }
        Ok(SnapRefRequest {
            adapter: p.adapter.path.clone(),
            references,
            biases: weights.iter().map(|w| (p.strength + w.ln()) as f32).collect(),
            mode: p.mode.clone(),
            caption,
            in_uncond: p.ref_in_uncond,
            uncond_bias: (p.fidelity < 1.0 && !p.ref_in_uncond).then(|| (1.0 - p.fidelity).max(1e-6).ln() as f32),
        })
    }

    /// The `BrokeAnimaStyleReference` node's picture, encoded as it does (`_encode_image` at `reference_megapixels`), and its
    /// settings over `steps` sampling steps of `family` (Anima or Krea 2). `prompt` is the positive's conditioning, whose
    /// text encoder the node's caption must share.
    fn style_request(&self, p: &StyleRefPatch, prompt: &DeferredCond, steps: usize, family: Family) -> EngineResult<StyleRequest> {
        let (cc, caption) = single_prompt(&p.caption, family)?;
        if cc.text_encoder != prompt.text_encoder {
            return Err(unsupported("a Broke Anima Style Reference CLIP other than the prompt's text encoder"));
        }
        check_text_encoder_loras(prompt, cc)?;
        let vae = p.vae.source.as_ref().ok_or_else(|| unsupported("Broke Anima Style Reference with the pixel-space VAE"))?;
        if is_qwen21_vae(vae) || is_flux2_vae(vae) || is_h3_vae(vae) {
            return Err(unsupported("Broke Anima Style Reference with a VAE other than the Qwen Image VAE"));
        }
        if !StyleRequest::NODE_MEGAPIXELS_RANGE.contains(&p.reference_megapixels) {
            return Err(EngineError::InvalidInput(format!("reference_megapixels {} (0.5 to 4)", p.reference_megapixels)));
        }
        let (rgb, w, h) = diffusion::snapref::shrink(&rgb_f32(&p.image), p.image.width as usize, p.image.height as usize,
            p.reference_megapixels);
        let mut gen = self.gen();
        gen.ensure_vae(&vae.path).map_err(internal)?;
        let reference = gen.encode_image(&rgb, w, h).map_err(internal)?;
        let mut r = StyleRequest::node(reference, p.stats_match as f32, p.restyle, steps);
        r.caption = caption;
        Ok(r)
    }

    fn encode_qwen21(&self, file: &ModelFile, pixels: &ImageBatch) -> EngineResult<Latent> {
        let c = pixels.channels as usize;
        if c < 3 {
            return Err(unsupported("VAE encoding of a greyscale image"));
        }
        let (w, h) = (pixels.width as usize, pixels.height as usize);
        let mut gen = self.gen();
        gen.ensure_vae21(&file.path).map_err(internal)?;
        let (mut data, mut size) = (Vec::new(), (0, 0));
        for i in 0..pixels.count {
            let rgba: Vec<f32> = pixels.frame_f32(i).chunks_exact(c).flat_map(|px| [px[0], px[1], px[2], if c == 4 { px[3] } else { 1.0 }]).collect();
            let r = gen.encode_image21(&rgba, w, h).map_err(internal)?;
            data.extend_from_slice(&r.latent);
            size = (r.height, r.width);
        }
        let (lh, lw) = size;
        let tensor = LatentTensor {
            format: LatentFormat::QwenImage21,
            shape: vec![pixels.count as usize, diffusion::qwen21::CH, lh, lw],
            data: Opaque::new(HostLatent::new(data)),
        };
        let scale = diffusion::vae21::SCALE as u32;
        Ok(Latent::single(tensor, PixelInfo::image(lw as u32 * scale, lh as u32 * scale, pixels.count)))
    }

    /// MOSS Image's Flux 2 VAE on `pixels`. Alpha is dropped, as ComfyUI's VAEEncode does.
    fn encode_flux2(&self, file: &ModelFile, pixels: &ImageBatch) -> EngineResult<Latent> {
        let c = pixels.channels as usize;
        if c < 3 {
            return Err(unsupported("VAE encoding of a greyscale image"));
        }
        let (w, h) = (pixels.width as usize, pixels.height as usize);
        let mut gen = self.gen();
        gen.ensure_flux2_vae(&file.path).map_err(internal)?;
        let (mut data, mut size) = (Vec::new(), (0, 0));
        for i in 0..pixels.count {
            let rgb: Vec<f32> = pixels.frame_f32(i).chunks_exact(c).flat_map(|px| [px[0], px[1], px[2]]).collect();
            let r = gen.encode_moss(&rgb, w, h).map_err(internal)?;
            data.extend_from_slice(&r.latent);
            size = (r.height, r.width);
        }
        let (lh, lw) = size;
        let tensor = LatentTensor {
            format: LatentFormat::Flux2,
            shape: vec![pixels.count as usize, diffusion::moss::CH, lh, lw],
            data: Opaque::new(HostLatent::new(data)),
        };
        Ok(Latent::single(tensor, PixelInfo::image(lw as u32 * 16, lh as u32 * 16, pixels.count)))
    }

    /// Qwen Image 2.1's RGBA VAE: the latent `[batch, 64, lh, lw]` to `[batch, 16lh, 16lw, 4]`.
    fn decode_qwen21(&self, file: &ModelFile, batch: usize, channels: usize, (lh, lw): (usize, usize), latent: &[f32], progress: &mut dyn ProgressSink)
        -> EngineResult<ImageBatch> {
        if channels != diffusion::qwen21::CH {
            return Err(unsupported(format!("this VAE on a {channels}-channel latent")));
        }
        let scale = diffusion::vae21::SCALE;
        let (width, height) = (lw * scale, lh * scale);
        let mut gen = self.gen();
        gen.ensure_vae21(&file.path).map_err(internal)?;
        let mut rgba = Vec::with_capacity(batch * height * width * 4);
        for (i, image) in latent.chunks_exact(channels * lh * lw).enumerate() {
            if progress.interrupted() {
                return Err(EngineError::Interrupted);
            }
            rgba.extend_from_slice(&gen.decode_qwen21(image, width, height).map_err(internal)?);
            progress.progress(i as u32 + 1, batch as u32);
        }
        Ok(ImageBatch::from_u8(batch as u32, height as u32, width as u32, 4, rgba))
    }

    /// Qwen Image 2.1: text to image and edits. Prompt images go through the vision tower and, as the reference latents
    /// of the prompts, into the sequence. Each conditioning entry is evaluated with its own prompt, prompt images and
    /// reference latents.
    fn sample_qwen21(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, plan: &Plan, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let sampling = sampling(req, settings, plan, batch)?;
        let positive = conds(&req.positive, Family::Qwen21, (lh, lw))?;
        let negative = if sampling.uses_negative() { conds(&req.negative, Family::Qwen21, (lh, lw))? } else { Vec::new() };
        let first = positive[0].0;
        for (c, ..) in positive.iter().chain(&negative) {
            if c.text_encoder != first.text_encoder {
                return Err(unsupported("conditioning entries from different text encoders"));
            }
            check_text_encoder_loras(first, c)?;
        }
        let inputs = |side: &[(&DeferredCond, &CondMeta, Cond)]| -> EngineResult<(Vec<Vec<PromptImage>>, Vec<Vec<Reference>>)> {
            let images = side.iter().map(|(c, ..)| c.images.clone()).collect();
            let references = side.iter().map(|(_, m, _)| qwen21_references(m)).collect::<EngineResult<_>>()?;
            Ok((images, references))
        };
        let ((pos_images, pos_refs), (neg_images, neg_refs)) = (inputs(&positive)?, inputs(&negative)?);
        let attn_token_weights = settings.attn_token_weights.as_ref().map(qwen21_attn_token_weights).transpose()?.flatten();
        let mut gen = self.gen();
        gen.ensure_qwen21(&file.path).map_err(internal)?;
        gen.ensure_te(&first.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&first.loras).map_err(internal)?;
        let (width, height) = (lw * diffusion::vae21::SCALE, lh * diffusion::vae21::SCALE);
        let r = Qwen21Request {
            positive: positive.into_iter().map(|(.., c)| c).collect(),
            negative: negative.into_iter().map(|(.., c)| c).collect(),
            width,
            height,
            sampling,
            noise: plan.noise(req, batch, (diffusion::qwen21::CH, lh, lw)),
            images: [pos_images, neg_images],
            references: [pos_refs, neg_refs],
            attn_token_weights,
        };
        let (latent, stats) = gen
            .sample_qwen21(&r, |step| report(mode, progress, batch, step, |x0| (preview::qwen21(x0, lh, lw), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("qwen image 2.1: {batch} image(s): text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        Ok(latent)
    }

    /// SenseNova U1.6 text to image and edits: each conditioning entry's prompt, with its reference pictures, runs through
    /// the checkpoint's text tower once, and the generation tower samples the pixels against its keys and values.
    fn sample_sensenova(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, plan: &Plan, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, height, width) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let sampling = sampling(req, settings, plan, batch)?;
        let positive = conds(&req.positive, Family::SenseNova, (height, width))?;
        let negative = if sampling.uses_negative() { conds(&req.negative, Family::SenseNova, (height, width))? } else { Vec::new() };
        let first = positive[0].0;
        for (c, ..) in positive.iter().chain(&negative) {
            if c.text_encoder != first.text_encoder {
                return Err(unsupported("conditioning entries from different text encoders"));
            }
            if !c.loras.is_empty() {
                return Err(unsupported("LoRAs on SenseNova's text tower"));
            }
        }
        if settings.attn_token_weights.is_some() {
            return Err(unsupported("SwarmAttnTokenWeights on SenseNova"));
        }
        let refs = |side: &[(&DeferredCond, &CondMeta, Cond)]| side.iter().map(|(_, m, _)| sensenova_references(m)).collect::<EngineResult<Vec<_>>>();
        let (positive_refs, negative_refs) = (refs(&positive)?, refs(&negative)?);
        let mut gen = self.gen();
        gen.ensure_sensenova(&file.path).map_err(internal)?;
        gen.ensure_sensenova_te(&first.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        let r = SenseNovaRequest {
            text_encoder: first.text_encoder.clone(),
            positive: positive.into_iter().map(|(.., c)| c).collect(),
            negative: negative.into_iter().map(|(.., c)| c).collect(),
            positive_refs,
            negative_refs,
            width,
            height,
            sampling,
            noise: plan.noise(req, batch, (3, height, width)),
        };
        let (latent, stats) = gen
            .sample_sensenova(&r, |step| report(mode, progress, batch, step, |x0| (preview::pixels(x0, height, width, 8), height / 8, width / 8)))
            .map_err(internal)?;
        tracing::info!("sensenova: {batch} image(s): text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        Ok(latent)
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
        if t < 1 || lh % 2 != 0 || lw % 2 != 0 {
            return Err(unsupported("MiniMax H3 needs at least 1 video frame and a width and height in multiples of 32"));
        }
        if req.negative.entries.iter().any(|e| !e.meta.minimax_keyframes.is_empty()) {
            return Err(unsupported("keyframes on the negative prompt of MiniMax H3"));
        }
        let partial = req.start_at_step != 0 || req.end_at_step < req.steps || !req.add_noise || req.denoise != 1.0;
        if partial || req.latent.noise_mask.is_some() || req.latent.audio_noise_mask.is_some() || !video.data.is_none() || !audio.data.is_none() {
            return Err(unsupported("init videos, refiners and noise masks on MiniMax H3"));
        }
        if h3::hyperflow_config(&file.path).is_some() {
            if !h3::hyperflow_sampler_ok(&req.sampler) {
                return Err(unsupported(format!("the {} sampler on HyperFlow H3 Flash (only euler and res_multistep)", req.sampler)));
            }
            if !settings.loras.is_empty() {
                return Err(unsupported("LoRAs on HyperFlow H3 Flash (its layers are pruned and factorized)"));
            }
        }
        let shape = h3::Shape { t, h: lh, w: lw, audio_t };
        let sigmas = schedule(req, settings, (lh, lw))?.ok_or_else(|| EngineError::InvalidInput("no steps to sample".into()))?;
        let plan = Plan { sigmas, latent: None, mask: None, differential: None };
        let sampling = sampling(req, settings, &plan, 1)?;
        let (cond, positive, conds, negative) = h3_entries(req, Family::H3, &shape, (lh, lw), sampling.uses_negative())?;
        let refuses_conds = settings.sparse.as_ref().is_some_and(|sp| !sp.takes_conditions());
        h3_combinations(settings.windows.as_ref(), refuses_conds, &shape, &positive, &conds, &negative)?;
        let ModelSampling::Flow { shift } = settings.model_sampling else { unreachable!("H3 is a flow model") };
        let mut gen = self.gen();
        gen.ensure_h3(&file.path).map_err(internal)?;
        gen.ensure_h3_te(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        let mut noise = noise::h3_noise(req.noise_seed, req.variation, (24, t, lh, lw), shape.audio_len());
        if let Some(w) = &settings.windows {
            // the sampler's seed, as ComfyUI's sampler wrapper reads it
            w.freenoise(&mut noise, &shape, req.noise_seed);
        }
        let r = H3Request {
            positive,
            negative,
            shape,
            sampling,
            shifts: (shift, settings.audio_shift),
            noise,
            sparse: settings.sparse.clone(),
            windows: settings.windows.clone(),
            conds,
            seed: req.noise_seed,
        };
        let (mut latent, stats) = gen
            .sample_h3(&r, |step| report(mode, progress, 1, step, |x0| (preview::h3(x0, t, lh, lw), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("h3: text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        // the pixel head's frames ride with the video latent, for its decode
        let dwd = gen.take_h3_dwd_pixels().map(|clip| clip_images(&clip, clip.t));
        drop(gen);
        let audio_data = latent.split_off(shape.video_len());
        Ok(req.latent.with_samples(LatentSamples::AudioVideo {
            video: LatentTensor { format: video.format, shape: video.shape.clone(), data: Opaque::new(HostLatent { data: latent, dwd }) },
            audio: LatentTensor { format: audio.format, shape: audio.shape.clone(), data: Opaque::new(HostLatent::new(audio_data)) },
        }))
    }

    /// A MiniMax-H3-Image picture from `MiniMaxH3ImageConditioning`'s or `SwarmEmptyMiniMaxH3ImageLatent`'s latent.
    fn sample_h3_image(&self, req: &SampleRequest, settings: &ModelSettings, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let latent = &req.latent;
        let LatentSamples::Single(tensor) = &latent.samples else {
            return Err(unsupported("MiniMax-H3-Image with an audio/video latent: set the model's class to MiniMax H3 Image in SwarmUI"));
        };
        let &[1, 24, 1, lh, lw] = tensor.shape.as_slice() else {
            if let [b, 24, 1, _, _] = tensor.shape.as_slice() {
                return Err(unsupported(format!("batches of {b} on MiniMax-H3-Image (use Images instead of Batch Size)")));
            }
            return Err(unsupported(format!("MiniMax-H3-Image latents of shape {:?} (it takes SwarmEmptyMiniMaxH3ImageLatent's or MiniMaxH3ImageConditioning's)", tensor.shape)));
        };
        if lh % 2 != 0 || lw % 2 != 0 {
            return Err(unsupported("MiniMax-H3-Image needs a width and height in multiples of 32"));
        }
        let partial = req.start_at_step != 0 || req.end_at_step < req.steps || !req.add_noise || req.denoise != 1.0;
        if partial || latent.noise_mask.is_some() || !tensor.data.is_none() {
            return Err(unsupported("init images, refiners and noise masks on MiniMax-H3-Image"));
        }
        if !settings.loras.is_empty() {
            return Err(unsupported("LoRAs on MiniMax-H3-Image (not supported yet)"));
        }
        let shape = h3::Shape::image(lw * 16, lh * 16);
        let sigmas = schedule(req, settings, (lh, lw))?.ok_or_else(|| EngineError::InvalidInput("no steps to sample".into()))?;
        let plan = Plan { sigmas, latent: None, mask: None, differential: None };
        let sampling = sampling(req, settings, &plan, 1)?;
        // regional masks resize to the [1, 24, 1, h, w] latent and cover every channel, as the core's mix takes them
        let (cond, positive, conds, negative) = h3_entries(req, Family::H3Image, &shape, (lh, lw), sampling.uses_negative())?;
        let mut gen = self.gen();
        gen.ensure_h3(&file.path).map_err(internal)?;
        gen.ensure_h3_te(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        let noise = plan.noise(req, 1, (24, lh, lw));
        let r = H3Request { positive, negative, shape, sampling, shifts: (h3::SHIFT_VIDEO, h3::SHIFT_AUDIO), noise, sparse: None, windows: None, conds, seed: req.noise_seed };
        let (out, stats) = gen
            .sample_h3(&r, |step| report(mode, progress, 1, step, |x0| (preview::h3(x0, 1, lh, lw), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("h3 image: text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        let video = LatentTensor { format: tensor.format, shape: tensor.shape.clone(), data: Opaque::new(HostLatent::new(out)) };
        Ok(latent.with_samples(LatentSamples::Single(video)))
    }

    /// One picture through an H3 VAE's encoder (keyframes), cropped to multiples of 16 about its centre as
    /// `vae_encode_crop_pixels` does.
    fn encode_h3(&self, file: &ModelFile, pixels: &ImageBatch) -> EngineResult<Latent> {
        if !has_h3_encoder(file) {
            return Err(unsupported(format!("VAE encoding with {} (it has no encoder)", file.name)));
        }
        if pixels.count != 1 {
            return Err(unsupported("encoding more than one frame with MiniMax H3's VAE (only single pictures)"));
        }
        let img = first_rgb(pixels);
        let (w, h) = (img.width / 16 * 16, img.height / 16 * 16);
        if w == 0 || h == 0 {
            return Err(EngineError::InvalidInput(format!("a {}x{} picture for MiniMax H3's VAE", img.width, img.height)));
        }
        let (x0, y0) = ((img.width % 16) / 2, (img.height % 16) / 2);
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            rgb.extend_from_slice(&img.rgb[((y0 + y) * img.width + x0) * 3..][..w * 3]);
        }
        let z = self.gen().encode_h3(&file.path, &PromptImage { width: w, height: h, rgb }).map_err(internal)?;
        let tensor = LatentTensor { format: LatentFormat::MiniMaxH3Video, shape: vec![1, 24, 1, h / 16, w / 16], data: Opaque::new(HostLatent::new(z)) };
        Ok(Latent::single(tensor, PixelInfo::image(w as u32, h as u32, 1)))
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
        // one latent is a still (single image mode); otherwise the clip's length on the 17k + 5 grid
        let frames = if lt == 1 { 1 } else { (lt.max(2) - 2) / 5 * 17 + 5 };
        let shape = h3::Shape { t: lt, h: lh, w: lw, audio_t: 0 };
        let clip = self.gen().decode_h3(&file.path, &latent.data, shape, frames).map_err(internal)?;
        Ok(clip_images(&clip, frames))
    }

    fn sample_moss(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, plan: &Plan, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        // SwarmAttnTokenWeights needs nothing here: ComfyUI's MOSS Image attention never runs attn1_patch, so <weight>
        // is the hidden states' scaling alone
        let positive = conds(&req.positive, Family::Moss, (lh, lw))?;
        let cond = positive[0].0;
        let sampling = sampling(req, settings, plan, batch)?;
        let negative = if sampling.uses_negative() { conds(&req.negative, Family::Moss, (lh, lw))? } else { Vec::new() };
        for (c, _, _) in positive.iter().chain(&negative) {
            if c.text_encoder != cond.text_encoder {
                return Err(unsupported("prompts from different text encoders"));
            }
            check_text_encoder_loras(cond, c)?;
        }
        let entries = |side: Vec<(&DeferredCond, &CondMeta, Cond)>| side.into_iter().map(|(_, _, c)| c).collect::<Vec<_>>();
        let (positive, negative) = (entries(positive), entries(negative));
        let negative = if negative.is_empty() { vec![Cond::plain(Prompt::text(""))] } else { negative };
        let mut gen = self.gen();
        gen.ensure_moss(&file.path).map_err(internal)?;
        gen.ensure_qwen35(&cond.text_encoder).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;
        let r = MossRequest {
            positive,
            negative,
            width: lw * 16,
            height: lh * 16,
            sampling,
            noise: plan.noise(req, batch, (diffusion::moss::CH, lh, lw)),
        };
        let (latent, stats) = gen
            .sample_moss(&r, |step| report(mode, progress, batch, step, |x0| (preview::flux2(x0, lh, lw), lh * 2, lw * 2)), None)
            .map_err(internal)?;
        tracing::info!("moss: {batch} image(s): text {:.0} ms, sampling {:.0} ms", stats.text_ms, stats.sample_ms);
        Ok(latent)
    }

    /// SD 1.x and SDXL. Their latents pass between nodes raw, as in ComfyUI, and are scaled only for sampling.
    fn sample_sd(&self, req: &SampleRequest, shape: (usize, usize, usize), settings: &ModelSettings, plan: &Plan, progress: &mut dyn ProgressSink)
        -> EngineResult<Vec<f32>> {
        let (batch, lh, lw) = shape;
        let mode = self.preview_mode(req);
        let file = req.model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        let xl = sd_file_is_xl(file)?;
        let positive = conds(&req.positive, Family::Sd, (lh, lw))?;
        let cond = positive[0].0;
        let mut sampling = sampling(req, settings, plan, batch)?;
        let negative = if sampling.uses_negative() { conds(&req.negative, Family::Sd, (lh, lw))? } else { Vec::new() };
        for (c, _, _) in positive.iter().chain(&negative) {
            if c.text_encoder != file.path {
                return Err(unsupported("a CLIP from another checkpoint than the model's"));
            }
            if c.clip_layer != cond.clip_layer {
                return Err(unsupported("different CLIP Stop At Layer on the prompts"));
            }
            check_text_encoder_loras(cond, c)?;
        }
        sd_attn_token_weights(settings.attn_token_weights.as_ref(), &file.path)?;
        let scale = if xl { diffusion::sdvae::SDXL_SCALE } else { diffusion::sdvae::SD1_SCALE };
        // process_latent_in
        if let Some(l) = &mut sampling.latent {
            l.iter_mut().for_each(|v| *v *= scale);
        }
        let (positive, positive_size) = sd_entries(positive, (lh, lw));
        let (negative, negative_size) = sd_entries(negative, (lh, lw));
        let r = SdRequest {
            positive,
            negative,
            width: lw * 8,
            height: lh * 8,
            positive_size,
            negative_size,
            clip_layer: cond.clip_layer,
            v_pred: settings.v_pred,
            sampling,
            noise: plan.noise(req, batch, (diffusion::unet::CH, lh, lw)),
        };
        let mut gen = self.gen();
        gen.ensure_sd(&file.path).map_err(internal)?;
        gen.set_int8_attention(settings.int8_attention);
        gen.set_loras(&settings.loras).map_err(internal)?;
        gen.set_text_encoder_loras(&cond.loras).map_err(internal)?;
        let (mut latent, stats) = gen
            .sample_sd(&r, |step| report(mode, progress, batch, step, |x0| (preview::sd(x0, lh, lw, xl), lh, lw)), None)
            .map_err(internal)?;
        tracing::info!("{}: {batch} image(s): text {:.0} ms, sampling {:.0} ms", if xl { "sdxl" } else { "sd1" }, stats.text_ms, stats.sample_ms);
        // process_latent_out
        latent.iter_mut().for_each(|v| *v /= scale);
        Ok(latent)
    }

    /// An SD VAE on `pixels`, alpha dropped as ComfyUI's VAEEncode does: raw latents `[batch, 4, h/8, w/8]`.
    fn encode_sd(&self, file: &ModelFile, pixels: &ImageBatch) -> EngineResult<Latent> {
        let c = pixels.channels as usize;
        if c < 3 {
            return Err(unsupported("VAE encoding of a greyscale image"));
        }
        let (w, h) = (pixels.width as usize, pixels.height as usize);
        let mut gen = self.gen();
        gen.ensure_sd_vae(&file.path).map_err(internal)?;
        let scale = gen.sd_vae_scale().ok_or_else(|| unsupported("an SD VAE"))?;
        let (mut data, mut size) = (Vec::new(), (0, 0));
        for i in 0..pixels.count {
            let rgb: Vec<f32> = pixels.frame_f32(i).chunks_exact(c).flat_map(|px| [px[0], px[1], px[2]]).collect();
            let r = gen.encode_sd(&rgb, w, h).map_err(internal)?;
            data.extend(r.latent.iter().map(|v| v / scale));
            size = (r.height, r.width);
        }
        let (lh, lw) = size;
        let tensor = LatentTensor {
            format: LatentFormat::Image4Ch,
            shape: vec![pixels.count as usize, diffusion::unet::CH, lh, lw],
            data: Opaque::new(HostLatent::new(data)),
        };
        Ok(Latent::single(tensor, PixelInfo::image(lw as u32 * 8, lh as u32 * 8, pixels.count)))
    }

    /// An SD VAE on raw latents `[batch, 4, lh, lw]`.
    fn decode_sd(&self, file: &ModelFile, batch: usize, channels: usize, (lh, lw): (usize, usize), latent: &[f32], progress: &mut dyn ProgressSink)
        -> EngineResult<ImageBatch> {
        if channels != diffusion::unet::CH {
            return Err(unsupported(format!("an SD VAE on a {channels}-channel latent")));
        }
        let mut gen = self.gen();
        gen.ensure_sd_vae(&file.path).map_err(internal)?;
        let scale = gen.sd_vae_scale().ok_or_else(|| unsupported("an SD VAE"))?;
        let (width, height) = (lw * 8, lh * 8);
        let mut rgb = Vec::with_capacity(batch * height * width * 3);
        for (i, image) in latent.chunks_exact(channels * lh * lw).enumerate() {
            if progress.interrupted() {
                return Err(EngineError::Interrupted);
            }
            let scaled: Vec<f32> = image.iter().map(|v| v * scale).collect();
            rgb.extend_from_slice(&gen.decode_sd(&scaled, width, height).map_err(internal)?);
            progress.progress(i as u32 + 1, batch as u32);
        }
        Ok(ImageBatch::from_u8(batch as u32, height as u32, width as u32, 3, rgb))
    }
}

/// Whether `path` is a converted Nemotron-Labs-Image file, from its header alone.
fn nemotron_file(path: &std::path::Path) -> bool {
    diffusion::st::SafeTensors::open(path).is_ok_and(|s| s.metadata.get("nemotron_image").is_some_and(|f| f == diffusion::nemotron::FORMAT))
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

/// Checks that each TripoSplat file is what its slot needs, before anything is unloaded for it.
fn check_triposplat(req: &ImageToSplatRequest) -> EngineResult<()> {
    let has = |f: &ModelFile, what: &str, test: fn(&dyn Fn(&str) -> bool) -> bool| -> EngineResult<()> {
        let h = read_header(&f.path).map_err(|e| unsupported(format!("{}: not a readable safetensors file ({e})", f.name)))?;
        if test(&|k| h.get(k).is_some()) { Ok(()) } else { Err(unsupported(format!("{}: not {what}", f.name))) }
    };
    has(&req.diffusion_model, "a TripoSplat or Idol diffusion model", |h| triposplat::is_triposplat(h))?;
    has(&req.clip_vision, "DINOv3 ViT-H/16+ (dino_v3_vit_h)", |h| diffusion::dino3::is_dino3_vith(h))?;
    has(&req.vae, "the Flux 2 VAE with its encoder", |h| h("bn.running_mean") && h("encoder.conv_in.weight"))?;
    has(&req.decoder, "TripoSplat's octree gaussian decoder", |h| triposplat::is_decoder(h))?;
    Ok(())
}

/// True for MiniMax H3's video or audio VAE, which load when they decode.
fn is_h3_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| h.get("decoder.transformer_blocks.0.scale1").is_some() || h.get("pre_block.attn.zero_k_bias").is_some())
}

/// True for SeedVR2's VAE.
fn is_seedvr2_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| diffusion::seedvr::is_seedvr2_vae(|k| h.get(k).is_some()))
}

/// The `SeedVR2Conditioning` an entry list carries: its picture's or clip's latent, of the sampled latent's `shape`.
fn seedvr_condition<'a>(c: &'a Conditioning, shape: &[usize]) -> EngineResult<&'a [f32]> {
    let [entry] = c.entries.as_slice() else {
        return Err(unsupported("SeedVR2 with other conditioning than one SeedVR2Conditioning"));
    };
    let Some(cond) = &entry.meta.seedvr2 else {
        return Err(unsupported("SeedVR2 without SeedVR2Conditioning (it restores a picture and takes no prompt)"));
    };
    let m = &entry.meta;
    if m.multiplier != 1.0 || m.zeroed || m.mask.is_some() || !m.reference_latents.is_empty() || m.inpaint.is_some() || !m.controls.is_empty()
        || !m.embedding_ops.is_empty() || !m.gligen.is_empty() || !m.unclip.is_empty() || m.start_percent != 0.0 || m.end_percent != 1.0 {
        return Err(unsupported("conditioning changes on SeedVR2's conditioning"));
    }
    let LatentSamples::Single(t) = &cond.latent.samples else {
        return Err(unsupported("SeedVR2Conditioning of an audio/video latent"));
    };
    if t.shape != shape {
        return Err(unsupported(format!("a SeedVR2 condition of shape {:?} for a latent of shape {shape:?}", t.shape)));
    }
    let h = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("a SeedVR2 condition this engine didn't encode"))?;
    Ok(&h.data)
}

/// True for RelayVSR's Flash decoder.
fn is_relayvsr_vae(file: &ModelFile) -> bool {
    read_header(&file.path).is_ok_and(|h| diffusion::relayvsr::is_flash(|k| h.get(k).is_some()))
}

/// The `RelayVSRConditioning` an entry list carries.
fn relayvsr_condition(c: &Conditioning) -> EngineResult<&RelayVsrCondition> {
    let [entry] = c.entries.as_slice() else {
        return Err(unsupported("RelayVSR with other conditioning than one RelayVSRConditioning"));
    };
    let Some(cond) = &entry.meta.relayvsr else {
        return Err(unsupported("RelayVSR without RelayVSRConditioning (it upscales a video and takes no prompt)"));
    };
    let m = &entry.meta;
    if m.multiplier != 1.0 || m.zeroed || m.mask.is_some() || !m.reference_latents.is_empty() || m.inpaint.is_some() || !m.controls.is_empty()
        || !m.embedding_ops.is_empty() || !m.gligen.is_empty() || !m.unclip.is_empty() || m.start_percent != 0.0 || m.end_percent != 1.0
        || m.seedvr2.is_some() || !m.minimax_keyframes.is_empty() {
        return Err(unsupported("conditioning changes on RelayVSR's conditioning"));
    }
    Ok(cond)
}

/// A batch's pixels, f32 [count][height][width][channels].
fn frames_f32(b: &ImageBatch) -> Vec<f32> {
    (0..b.count).flat_map(|f| b.frame_f32(f).into_owned()).collect()
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
        if check_dit(&req.file)? == Family::Sd {
            // an SD checkpoint carries its CLIP and VAE, which load with the model
            let model = self.load_model(&req.file)?;
            let st = diffusion::st::SafeTensors::open(&req.file.path).map_err(internal)?;
            let encoder_type = if sd_file_is_xl(&req.file)? { SDXL_ENCODER } else { SD1_ENCODER };
            let sources = if diffusion::clip::has_clip_l(&st) { vec![req.file.clone()] } else { Vec::new() };
            let te = TextEncoderHandle { sources, encoder_type: encoder_type.into(), device: "default".into(), engine: Opaque::none(), patches: vec![] };
            let vae = VaeHandle::new(is_sd_vae(&req.file).then(|| req.file.clone()), Opaque::none());
            return Ok((model, te, vae));
        }
        if check_dit(&req.file)? == Family::SenseNova {
            // the text tower is in the file, and ComfyUI gives the checkpoint its pixel_space VAE
            let model = self.load_model(&req.file)?;
            let te = TextEncoderHandle { sources: vec![req.file.clone()], encoder_type: SENSENOVA_ENCODER.into(), device: "default".into(), engine: Opaque::none(), patches: vec![] };
            return Ok((model, te, VaeHandle::new(None, Opaque::none())));
        }
        // The other families' files hold only the diffusion model; SwarmUI adds its own CLIPLoader and VAELoader.
        Ok((
            self.load_model(&req.file)?,
            TextEncoderHandle { sources: vec![], encoder_type: String::new(), device: "default".into(), engine: Opaque::none(), patches: vec![] },
            VaeHandle::new(None, Opaque::none()),
        ))
    }

    fn load_diffusion_model(&self, req: &LoadDiffusionModel) -> EngineResult<ModelHandle> {
        // int8 files run as stored, but casting a bf16 file to fp8 isn't done here
        if req.weight_dtype != WeightDtype::Default && check_dit(&req.file)? == Family::Moss && !is_int8_moss(&req.file)? {
            return Err(unsupported(format!("{}: weight dtype {:?} on a bf16 MOSS Image file (choose default, which quantizes it to int8 unless the server has --no-quant)", req.file.name, req.weight_dtype)));
        }
        self.load_model(&req.file)
    }

    fn load_text_encoder(&self, req: &LoadTextEncoder) -> EngineResult<TextEncoderHandle> {
        let [file] = req.files.as_slice() else {
            return Err(unsupported("dual text encoders"));
        };
        let encoder_type = match req.encoder_type.as_str() {
            KREA_ENCODER => {
                self.gen().ensure_te(&file.path).map_err(internal)?;
                KREA_ENCODER
            }
            // Qwen Image 1's encoder (Qwen2.5-VL 7B) goes by the same type
            "qwen_image" if is_qwen3vl_8b(file) => {
                self.gen().ensure_te(&file.path).map_err(internal)?;
                QWEN21_ENCODER
            }
            "qwen_image" => return Err(unsupported("Qwen Image 1's text encoder (this backend runs Qwen Image 2.1)")),
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
            t => return Err(unsupported(format!("text encoder type {t} (this backend only runs Krea 2, Anima, MOSS Image, Qwen Image 2.1 and MiniMax H3)"))),
        };
        Ok(TextEncoderHandle { sources: req.files.clone(), encoder_type: encoder_type.into(), device: req.device.clone(), engine: Opaque::none(), patches: vec![] })
    }

    fn load_vae(&self, req: &LoadVae) -> EngineResult<VaeHandle> {
        if is_relayvsr_vae(&req.file) {
            self.gen().ensure_relayvsr_vae(&req.file.path).map_err(internal)?;
            return Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()));
        }
        if is_seedvr2_vae(&req.file) {
            self.gen().ensure_seedvr_vae(&req.file.path).map_err(internal)?;
            return Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()));
        }
        if is_h3_vae(&req.file) {
            return Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()));
        }
        let mut gen = self.gen();
        if is_qwen21_vae(&req.file) {
            gen.ensure_vae21(&req.file.path)
        } else if is_sd_vae(&req.file) {
            gen.ensure_sd_vae(&req.file.path)
        } else if is_flux2_vae(&req.file) {
            gen.ensure_flux2_vae(&req.file.path)
        } else {
            gen.ensure_vae(&req.file.path)
        }
        .map_err(internal)?;
        Ok(VaeHandle::new(Some(req.file.clone()), Opaque::none()))
    }

    fn encode_text(&self, req: &TextEncodeRequest) -> EngineResult<Conditioning> {
        let [file] = req.text_encoder.sources.as_slice() else {
            return Err(unsupported("this text encoder (Krea 2 needs one Qwen3-VL 4B file, Anima one Qwen3 file, MOSS Image one Qwen3.5 file, Qwen Image 2.1 one Qwen3-VL 8B file, MiniMax H3 one Qwen3-VL 32B file, SD 1.x and SDXL their checkpoint's CLIP)"));
        };
        let family = match req.text_encoder.encoder_type.as_str() {
            ANIMA_ENCODER => Family::Anima,
            MOSS_ENCODER => Family::Moss,
            MINIMAX_ENCODER => Family::H3,
            QWEN21_ENCODER => Family::Qwen21,
            KREA_ENCODER => Family::Krea2,
            SD1_ENCODER | SDXL_ENCODER => Family::Sd,
            SENSENOVA_ENCODER => Family::SenseNova,
            t => return Err(unsupported(format!("text encoder type {t}"))),
        };
        let images = match &req.mode {
            TextEncodeMode::Swarm(s) if family.one_piece() || family == Family::H3 => s.images.as_ref().map(prompt_images).unwrap_or_default(),
            _ => Vec::new(),
        };
        let loras = req.text_encoder.patches.iter().filter_map(|p| match p {
            TextEncoderPatch::Lora { file, strength } => Some(Lora { path: file.path.clone(), strength: *strength }),
            _ => None,
        }).collect();
        let minimax = match &req.mode {
            TextEncodeMode::Swarm(s) => s.minimax_refs.clone(),
            TextEncodeMode::MiniMax { refs, .. } => Some(refs.clone()),
            TextEncodeMode::Plain { .. } | TextEncodeMode::Sdxl { .. } => None,
        };
        let clip_layer = req.text_encoder.patches.iter().rev().find_map(|p| match p {
            TextEncoderPatch::StopAtLayer { layer } => Some(*layer),
            _ => None,
        });
        let sdxl_size = match &req.mode {
            TextEncodeMode::Sdxl { size, .. } => Some(*size),
            TextEncodeMode::Swarm(s) => Some(SdxlSize { width: s.width, height: s.height, crop_w: 0, crop_h: 0, target_width: s.target_width, target_height: s.target_height }),
            _ => None,
        };
        let leaves = matches!(req.mode, TextEncodeMode::Swarm(_));
        let cond = DeferredCond { text_encoder: file.path.clone(), prompt: encode_request(req, family), images, loras, minimax, clip_layer, sdxl_size, leaves };
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
        if let EmptyLatentKind::MiniMaxH3Image { width, height } = req.kind {
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::MiniMaxH3Video, shape: vec![req.batch_size as usize, 24, 1, height as usize / 16, width as usize / 16], data: Opaque::none() },
                PixelInfo::image(width, height, req.batch_size),
            ));
        }
        if let EmptyLatentKind::Flux2 { width, height } = req.kind {
            let (b, h, w) = (req.batch_size as usize, height as usize / 16, width as usize / 16);
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::Flux2, shape: vec![b, diffusion::moss::CH, h, w], data: Opaque::none() },
                PixelInfo::image(width, height, req.batch_size),
            ));
        }
        if let EmptyLatentKind::Pixel { width, height } = req.kind {
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::Pixel, shape: vec![req.batch_size as usize, 3, height as usize, width as usize], data: Opaque::none() },
                PixelInfo::image(width, height, req.batch_size),
            ));
        }
        // RelayVSRConditioning's zeros: one latent frame per keyframe, sampled as they are
        if let EmptyLatentKind::RelayVsr { keyframes, latent_height, latent_width, width, height, frames } = req.kind {
            let shape = vec![1, diffusion::relayvsr::CH, keyframes as usize, latent_height as usize, latent_width as usize];
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::RelayVsr, shape, data: Opaque::none() },
                PixelInfo { frames, ..PixelInfo::image(width, height, 1) },
            ));
        }
        // Qwen Image 2.1's placeholder, which `sample` widens to its 64 channels
        if let EmptyLatentKind::Image4Ch { width, height } = req.kind {
            let (b, h, w) = (req.batch_size as usize, height as usize / 8, width as usize / 8);
            return Ok(Latent::single(
                LatentTensor { format: LatentFormat::Image4Ch, shape: vec![b, 4, h, w], data: Opaque::none() },
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
        let mut settings = model_settings(&req.model, family)?;
        if let (Family::H3, Some(sol3)) = (family, &self.sol3) {
            if settings.sparse.is_some() {
                return Err(unsupported(
                    "a sparse attention node while this backend runs Sol³ on every MiniMax H3 generation (--sol3); turn off Use Sparse Attention",
                ));
            }
            settings.sparse = Some(sol3.clone());
        }
        if req.scheduler == "hyperflow" && h3::hyperflow_config(&file.path).is_none() {
            return Err(unsupported("the hyperflow scheduler on a model other than HyperFlow H3 Flash"));
        }
        if family == Family::H3 {
            return self.sample_h3(req, &settings, progress);
        }
        if family == Family::H3Image {
            return self.sample_h3_image(req, &settings, progress);
        }
        if family == Family::SeedVr2 {
            return self.sample_seedvr(req, &settings, progress);
        }
        if family == Family::RelayVsr {
            return self.sample_relayvsr(req, &settings, progress);
        }
        // SwarmUI sends Qwen Image 2.1 a half-size 4-channel placeholder, as ComfyUI widens it
        let widened = if family == Family::Qwen21 { req.latent.conform_empty(LatentFormat::QwenImage21, family.latent().0, family.latent().1 as u32) } else { None };
        let latent = widened.as_ref().unwrap_or(&req.latent);
        let LatentSamples::Single(tensor) = &latent.samples else {
            return Err(unsupported("audio/video latents"));
        };
        let &[batch, channels, lh, lw] = tensor.shape.as_slice() else {
            return Err(unsupported(format!("latent shape {:?}", tensor.shape)));
        };
        if channels != family.latent().0 {
            return Err(unsupported(format!("latent shape {:?} for this model", tensor.shape)));
        }
        let finish = |data| {
            Ok(latent.with_samples(LatentSamples::Single(LatentTensor {
                format: tensor.format.clone(),
                shape: tensor.shape.clone(),
                data: Opaque::new(HostLatent::new(data)),
            })))
        };
        let shape = (batch, lh, lw);
        let Some(sigmas) = schedule(req, &settings, (lh, lw))? else {
            // no step left: ComfyUI returns the latent it was given
            let plan = Plan::new(Vec::new(), latent, tensor, &settings)?;
            return finish(plan.latent.unwrap_or_else(|| vec![0.0; batch * channels * lh * lw]));
        };
        let plan = Plan::new(sigmas, latent, tensor, &settings)?;
        finish(match family {
            Family::Krea2 => self.sample_krea(req, shape, &settings, &plan, progress)?,
            Family::Anima => self.sample_anima(req, shape, &settings, &plan, progress)?,
            Family::Moss => self.sample_moss(req, shape, &settings, &plan, progress)?,
            Family::Qwen21 => self.sample_qwen21(req, shape, &settings, &plan, progress)?,
            Family::Sd => self.sample_sd(req, shape, &settings, &plan, progress)?,
            Family::SenseNova => self.sample_sensenova(req, shape, &settings, &plan, progress)?,
            Family::H3 | Family::H3Image | Family::SeedVr2 | Family::RelayVsr => unreachable!("H3, SeedVR2 and RelayVSR sample above"),
        })
    }

    fn vae_decode(&self, req: &VaeDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        // a pi-H3 DWD run's frames come from its adapter's pixel head, which decoded them at the last step: they take
        // the VAE's place, and tiling has nothing left to split
        let (LatentSamples::AudioVideo { video: t, .. } | LatentSamples::Single(t)) = &req.latent.samples;
        if let Some(frames) = t.data.downcast_ref::<HostLatent>().filter(|_| t.format == LatentFormat::MiniMaxH3Video).and_then(|d| d.dwd.as_ref()) {
            return Ok(frames.clone());
        }
        match &req.latent.samples {
            LatentSamples::AudioVideo { video, .. } => return self.decode_h3(req, video),
            LatentSamples::Single(t) if t.format == LatentFormat::MiniMaxH3Video => return self.decode_h3(req, t),
            _ => {}
        }
        if req.vae.source.as_ref().is_some_and(is_relayvsr_vae) {
            return Err(unsupported("VAE Decode with RelayVSR's Flash decoder (it decodes with VSR Decode, which reads the low-quality video too)"));
        }
        if matches!(&req.latent.samples, LatentSamples::Single(t) if t.format == LatentFormat::RelayVsr) {
            return Err(unsupported("VAE Decode of a RelayVSR latent (it decodes with VSR Decode and the RelayVSR Flash decoder)"));
        }
        let seedvr_vae = req.vae.source.as_ref().is_some_and(is_seedvr2_vae);
        if let LatentSamples::Single(t) = &req.latent.samples {
            if seedvr_vae != (t.format == LatentFormat::SeedVr2) {
                return Err(unsupported(if seedvr_vae { "SeedVR2's VAE on another model's latent" } else { "a SeedVR2 latent through another VAE" }));
            }
            if seedvr_vae {
                if req.tiling.is_some() {
                    return Err(unsupported("VAE tiling on SeedVR2"));
                }
                return self.decode_seedvr(&req.vae, t, progress);
            }
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
        if (t.format == LatentFormat::Pixel) != req.vae.source.is_none() {
            return Err(unsupported(if req.vae.source.is_none() {
                "the pixel-space VAE on another model's latent"
            } else {
                "a SenseNova pixel latent through a VAE (it decodes with its checkpoint's own pixel_space VAE)"
            }));
        }
        if t.format == LatentFormat::Pixel {
            // ComfyUI's pixel_space VAE: (x + 1) / 2, clamped
            let mut rgb = Vec::with_capacity(batch * lh * lw * 3);
            for image in latent.data.chunks_exact(channels * lh * lw) {
                rgb.extend_from_slice(&diffusion::sensenova::to_rgb8(image, lw, lh));
            }
            progress.progress(batch as u32, batch as u32);
            return Ok(ImageBatch::from_u8(batch as u32, lh as u32, lw as u32, 3, rgb));
        }
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("the pixel-space VAE"))?;
        if is_qwen21_vae(file) {
            return self.decode_qwen21(file, batch, channels, (lh, lw), &latent.data, progress);
        }
        if is_sd_vae(file) {
            return self.decode_sd(file, batch, channels, (lh, lw), &latent.data, progress);
        }
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
        if is_relayvsr_vae(file) {
            return Err(unsupported("VAE Encode with RelayVSR's Flash decoder, which only decodes"));
        }
        if is_seedvr2_vae(file) {
            return self.encode_seedvr(file, &req.pixels);
        }
        if is_qwen21_vae(file) {
            return self.encode_qwen21(file, &req.pixels);
        }
        if is_flux2_vae(file) {
            return self.encode_flux2(file, &req.pixels);
        }
        if is_sd_vae(file) {
            return self.encode_sd(file, &req.pixels);
        }
        if is_h3_vae(file) {
            return self.encode_h3(file, &req.pixels);
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
        let tensor = LatentTensor { format: LatentFormat::Image16Ch, shape: vec![frames.len(), 16, lh, lw], data: Opaque::new(HostLatent::new(data)) };
        Ok(Latent::single(tensor, PixelInfo::image(lw as u32 * 8, lh as u32 * 8, frames.len() as u32)))
    }

    fn latent_op(&self, op: &LatentOp) -> EngineResult<Latent> {
        latent::latent_op(op)
    }

    fn relayvsr_latent_stride(&self, model: &ModelHandle) -> EngineResult<u32> {
        let file = model.source.as_ref().ok_or_else(|| unsupported("a model without a source file"))?;
        if check_dit(file)? != Family::RelayVsr {
            return Err(unsupported(format!("RelayVSRConditioning on {}, which isn't a RelayVSR model", file.name)));
        }
        let h = read_header(&file.path).map_err(internal)?;
        let p = diffusion::relayvsr::key_prefix(|k| h.get(k).is_some()).ok_or_else(|| unsupported("not a RelayVSR model"))?;
        let cin = h[format!("{p}lq_proj_in.conv1.weight")]["shape"][1].as_u64().ok_or_else(|| unsupported("RelayVSR without lq_proj_in.conv1"))?;
        let factor = diffusion::relayvsr::factor_of(cin as usize).map_err(|e| unsupported(e.to_string()))?;
        Ok((16 / factor) as u32)
    }

    fn vsr_decode(&self, req: &VsrDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        let file = req.vae.source.as_ref().ok_or_else(|| unsupported("VSR Decode without a VAE file"))?;
        if !is_relayvsr_vae(file) {
            return Err(unsupported(format!("VSR Decode with {}, which isn't RelayVSR's Flash decoder", file.name)));
        }
        let LatentSamples::Single(t) = &req.latent.samples else {
            return Err(unsupported("VSR Decode of an audio/video latent"));
        };
        let &[1, ch, keys, lh, lw] = t.shape.as_slice() else {
            return Err(unsupported(format!("VSR Decode of a latent of shape {:?}", t.shape)));
        };
        if ch != diffusion::relayvsr::CH || t.format != LatentFormat::RelayVsr {
            return Err(unsupported("VSR Decode of a latent RelayVSR didn't sample"));
        }
        let latent = t.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("decoding a latent this engine didn't sample"))?;
        let img = &req.images;
        if img.count == 0 {
            return Err(EngineError::InvalidInput("VSR Decode without frames".into()));
        }
        let data = frames_f32(img);
        let frames = diffusion::pipeline::RelayVsrFrames {
            data: &data,
            count: img.count as usize,
            height: img.height as usize,
            width: img.width as usize,
            channels: img.channels as usize,
        };
        let mut gen = self.gen();
        gen.ensure_relayvsr_vae(&file.path).map_err(internal)?;
        let factor = gen.relayvsr_vae_factor().unwrap_or(1);
        let rgb = gen
            .decode_relayvsr(&latent.data, keys, lh, lw, &frames, req.detail as f32, |done, total| {
                progress.progress(done as u32, total as u32);
                if progress.interrupted() { Err(Cancelled.into()) } else { Ok(()) }
            })
            .map_err(internal)?;
        Ok(ImageBatch::from_f32(img.count, img.height * factor as u32, img.width * factor as u32, 3, rgb))
    }

    /// SeedVR2 clips: the most latent frames one evaluation fits, from this engine's own estimate.
    fn sample_frames_that_fit(&self, latent: &Latent) -> EngineResult<Option<usize>> {
        let LatentSamples::Single(t) = &latent.samples else { return Ok(None) };
        if t.format != LatentFormat::SeedVr2 {
            return Ok(None);
        }
        let (frames, lh, lw) = match *t.shape.as_slice() {
            [_, _, f, lh, lw] => (f, lh, lw),
            [_, _, lh, lw] => (1, lh, lw),
            _ => return Ok(None),
        };
        Ok(Some(self.gen().seedvr_frames_that_fit(lw * 8, lh * 8, frames).map_err(internal)?))
    }

    fn dwd_decode(&self, latent: &Latent) -> EngineResult<ImageBatch> {
        let video = match &latent.samples {
            LatentSamples::AudioVideo { video, .. } => video,
            LatentSamples::Single(t) if t.format == LatentFormat::MiniMaxH3Video => t,
            _ => return Err(unsupported("DWD Decode of a latent other than MiniMax H3's (only pi-H3's DWD adapters have a pixel head)")),
        };
        let data = video.data.downcast_ref::<HostLatent>().ok_or_else(|| unsupported("DWD Decode of a latent this engine didn't sample"))?;
        data.dwd.clone().ok_or_else(|| {
            EngineError::InvalidInput("no DWD frames for this latent: connect the output of a sampler whose model has a pi-H3 DWD adapter".into())
        })
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
                SD1_ENCODER | SDXL_ENCODER => gen.ensure_sd(&f.path),
                SENSENOVA_ENCODER => gen.ensure_sensenova_te(&f.path),
                // the 8B of Qwen Image 2.1 as ensure_te takes it too
                _ => gen.ensure_te(&f.path),
            }
            .map_err(internal)?;
        }
        if let Some(f) = req.vae.source.as_ref().filter(|f| !is_h3_vae(f)) {
            if is_relayvsr_vae(f) {
                gen.ensure_relayvsr_vae(&f.path)
            } else if is_seedvr2_vae(f) {
                gen.ensure_seedvr_vae(&f.path)
            } else if is_qwen21_vae(f) {
                gen.ensure_vae21(&f.path)
            } else if is_sd_vae(f) {
                gen.ensure_sd_vae(&f.path)
            } else if is_flux2_vae(f) {
                gen.ensure_flux2_vae(&f.path)
            } else {
                gen.ensure_vae(&f.path)
            }
            .map_err(internal)?;
        }
        Ok(())
    }

    fn nvidia_vsr(&self, req: &NvidiaVsrRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        if !nvvfx::is_model(&req.model.path) {
            return Err(unsupported(format!("RTX Video Super Resolution with {}, which isn't NVIDIA's {}", req.model.name, nvvfx::MODEL)));
        }
        let (w, h) = (req.images.width, req.images.height);
        match req.mode {
            NvidiaVsrMode::Deblur(_) if (req.width, req.height) != (w, h) => {
                return Err(EngineError::InvalidInput(format!("RTX Video Super Resolution deblurs at the image's size, {w}x{h}, not {}x{}", req.width, req.height)))
            }
            NvidiaVsrMode::Upscale(_) if req.width < w || req.height < h => {
                return Err(EngineError::InvalidInput(format!("RTX Video Super Resolution doesn't downscale: {w}x{h} to {}x{}", req.width, req.height)))
            }
            _ => {}
        }
        self.run_vsr(&req.model.path, vsr_mode(req.mode), &req.images, (req.width, req.height), progress)
    }

    fn upscale_image(&self, req: &UpscaleImageRequest) -> EngineResult<ImageBatch> {
        let file = &req.model.source;
        if !nvvfx::is_model(&file.path) {
            return Err(unsupported(format!("model upscaling with {}: the engine only runs NVIDIA's RTX Video Super Resolution ({})", file.name, nvvfx::MODEL)));
        }
        let img = &req.image;
        let size = (img.width * VSR_MODEL_SCALE, img.height * VSR_MODEL_SCALE);
        self.run_vsr(&file.path, nvvfx::Mode::Ultra, img, size, &mut NullProgress)
    }

    fn idle(&self) {
        self.drop_vsr();
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

    fn nemotron_image(&self, req: &NemotronImageRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        if !nemotron_file(&req.model.path) {
            return Err(EngineError::InvalidInput(format!("{} isn't a Nemotron-Labs-Image file from tools/nemotron_image_convert.py", req.model.name)));
        }
        if req.width % 16 != 0 || req.height % 16 != 0 || req.width == 0 || req.height == 0 {
            return Err(EngineError::InvalidInput("Nemotron-Labs-Image needs width and height positive multiples of 16".into()));
        }
        if !(1..=diffusion::nemotron::MAX_BATCH as u32).contains(&req.batch_size) {
            return Err(EngineError::InvalidInput(format!("Nemotron-Labs-Image samples 1 to {} images together", diffusion::nemotron::MAX_BATCH)));
        }
        if !(req.temperature >= 0.0 && req.temperature.is_finite()) {
            return Err(EngineError::InvalidInput("temperature must be 0 or more".into()));
        }
        let mut gen = self.gen();
        gen.ensure_nemotron(&req.model.path).map_err(internal)?;
        let r = diffusion::pipeline::NemotronRequest {
            prompt: req.prompt.clone(),
            width: req.width as usize,
            height: req.height as usize,
            seed: req.seed,
            images: req.batch_size as usize,
            temperature: req.temperature as f32,
        };
        let images = gen
            .generate_nemotron(&r, |done, of| {
                progress.progress(done as u32, of as u32);
                !progress.interrupted()
            })
            .map_err(internal)?;
        drop(gen);
        let (w, h) = (req.width, req.height);
        Ok(ImageBatch::from_u8(images.len() as u32, h, w, 3, images.concat()))
    }

    fn nemotron_load(&self, model: &ModelFile) -> EngineResult<()> {
        if !nemotron_file(&model.path) {
            return Err(EngineError::InvalidInput(format!("{} isn't a Nemotron-Labs-Image file from tools/nemotron_image_convert.py", model.name)));
        }
        self.gen().ensure_nemotron(&model.path).map_err(internal)
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

    fn image_to_splat(&self, req: &ImageToSplatRequest, progress: &mut dyn ProgressSink) -> EngineResult<Vec<u8>> {
        check_triposplat(req)?;
        let n = req.num_gaussians as usize;
        if !(triposplat::MIN_GAUSSIANS..=triposplat::MAX_GAUSSIANS).contains(&n) || n % 32 != 0 {
            return Err(EngineError::InvalidInput("num_gaussians must be a multiple of 32 from 32768 to 1048576".into()));
        }
        if req.steps == 0 {
            return Err(EngineError::InvalidInput("steps must be at least 1".into()));
        }
        let img = &req.image;
        let (w, h) = (img.width as usize, img.height as usize);
        let c = img.channels as usize;
        if c < 3 {
            return Err(unsupported("a greyscale image for TripoSplat"));
        }
        let frame = img.frame_f32(0);
        let rgb: Vec<f32> = frame.chunks_exact(c).flat_map(|p| [p[0], p[1], p[2]]).collect();
        let mask = match &req.mask {
            Some(m) if (m.width as usize, m.height as usize) == (w, h) => m.data[..w * h].to_vec(),
            Some(m) => return Err(unsupported(format!("a {}x{} mask on a {w}x{h} image", m.width, m.height))),
            None if c == 4 => frame.chunks_exact(4).map(|p| p[3]).collect(),
            None => return Err(EngineError::InvalidInput("TripoSplat needs the object cut out: an image with transparency, or a mask".into())),
        };
        let mut p = triposplat::Params::example(req.seed);
        (p.steps, p.cfg, p.num_gaussians) = (req.steps as usize, req.cfg, n);
        let paths = triposplat::Paths {
            dit: req.diffusion_model.path.clone(),
            dino: req.clip_vision.path.clone(),
            flux_vae: req.vae.path.clone(),
            decoder: req.decoder.path.clone(),
        };
        // the pipeline needs the whole GPU: the image models go, and load again when next used
        let mut gen = self.gen();
        gen.unload().map_err(internal)?;
        // one bar: the encode, the sampler's steps, then the decode's levels, by rough share of the time
        let shares = [("encode", 0.0, 0.1), ("sample", 0.1, 0.85), ("decode", 0.85, 1.0)];
        let mut last = String::new();
        let mut report = |stage: &str, i: usize, n: usize| -> anyhow::Result<()> {
            if stage != last {
                last = stage.to_string();
                tracing::info!("triposplat: {stage}");
            }
            let (_, a, b) = shares.iter().find(|s| s.0 == stage).copied().unwrap_or(("", 0.0, 1.0));
            let frac = a + (b - a) * i as f64 / n.max(1) as f64;
            progress.progress((frac * 1000.0) as u32, 1000);
            if progress.interrupted() { Err(Cancelled.into()) } else { Ok(()) }
        };
        let splat = triposplat::run(&paths, &rgb, &mask, w, h, &p, self.threads, &mut report).map_err(internal)?;
        drop(gen);
        Ok(triposplat::ply(&splat))
    }

    fn free_memory(&self, unload_models: bool, free_memory: bool) {
        if unload_models || free_memory {
            self.drop_vsr();
        }
        let mut gen = self.gen();
        let r = if unload_models {
            gen.unload()
        } else if free_memory {
            // weights leave VRAM for RAM; the models stay loaded
            gen.release_vram()
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

    /// Needs a CUDA GPU, but no checkpoint.
    #[test]
    fn a_panic_unloads_and_unpoisons_the_generator() {
        let engine = NativeEngine::new(&NativeOptions::default()).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _gen = engine.gen();
            panic!("a job failed while it held the generator");
        }));
        assert!(r.is_err() && engine.gen.is_poisoned());
        drop(engine.gen());
        assert!(!engine.gen.is_poisoned());
    }

    #[test]
    fn sparse_block_lists() {
        assert_eq!(block_list("0, 1, 47-49"), vec![0, 1, 47, 48, 49]);
        assert_eq!(block_list("5 - 3;9"), vec![3, 4, 5, 9]);
        assert_eq!(block_list(""), Vec::<usize>::new());
        assert_eq!(block_list("2-, x7"), vec![2, 7]);
    }

    #[test]
    fn plain_prompts_keep_comfy_literal_parentheses() {
        // Qwen3-VL tokenizes with weights disabled, so `(x:1.2)` stays in the text.
        let p = ParsedPrompt::parse("a (red:1.2) fox");
        assert_eq!(whole_chunks(&p, Family::Krea2).unwrap(), [[("a (red:1.2) fox".to_string(), 1.0)]]);
    }

    /// The chunks of a prompt without steps.
    fn whole_chunks(p: &ParsedPrompt, family: Family) -> Result<Chunks, String> {
        let pieces = prompt_pieces(p, 8, family)?;
        assert_eq!(pieces.len(), 1);
        assert_eq!((pieces[0].start_percent, pieces[0].end_percent), (0.0, 1.0));
        Ok(pieces[0].chunks.clone())
    }

    #[test]
    fn stepped_prompts_split_like_swarm_text_encode_advanced() {
        let ranges = |p: &str, steps| -> Vec<(f64, f64, String)> {
            prompt_pieces(&ParsedPrompt::parse(p), steps, Family::Moss).unwrap().into_iter()
                .map(|p| ((p.start_percent * 1e6).round() / 1e6, (p.end_percent * 1e6).round() / 1e6, p.chunks[0].iter().map(|(t, _)| t.as_str()).collect())).collect()
        };
        // a new entry wherever the step's leaves change, widened by 0.001 on each side
        assert_eq!(ranges("x <alternate:a, b>", 4), [
            (-0.001, 0.251, "x a".into()), (0.249, 0.501, "x b".into()), (0.499, 0.751, "x a".into()), (0.749, 1.0, "x b".into())
        ]);
        assert_eq!(ranges("<fromto[0.5]:a, b>", 10), [(-0.001, 0.501, "a".into()), (0.499, 1.0, "b".into())]);
        // fromto at a step count
        assert_eq!(ranges("<fromto[3]:a, b>", 10), [(-0.001, 0.301, "a".into()), (0.299, 1.0, "b".into())]);
    }

    #[test]
    fn prompt_syntax_the_pipeline_cant_honor_is_refused() {
        assert!(whole_chunks(&ParsedPrompt::parse("<embed:x>"), Family::Krea2).is_err());
        for family in [Family::H3] {
            assert!(whole_chunks(&ParsedPrompt::parse("a<break>b"), family).is_err());
        }
    }

    #[test]
    fn anima_breaks_are_chunks_with_weights_only_before_the_first() {
        let chunks = whole_chunks(&ParsedPrompt::parse("lake, <weight[1.3]:pine><break>sky"), Family::Anima).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!((chunks[0].iter().find(|(s, _)| s == "pine").unwrap().1 - 1.3).abs() < 1e-6);
        assert_eq!(chunks[1], [("sky".to_string(), 1.0)]);
        for p in ["lake<break><weight[1.3]:pine>", "lake<break>(sky:1.2)"] {
            assert!(whole_chunks(&ParsedPrompt::parse(p), Family::Anima).is_err(), "{p}");
        }
        // each stepped piece keeps its own chunks
        let pieces = prompt_pieces(&ParsedPrompt::parse("<alternate:lake, sea><break>pine"), 4, Family::Anima).unwrap();
        assert_eq!(pieces.len(), 4);
        assert!(pieces.iter().all(|p| p.chunks.len() == 2 && p.chunks[1] == [("pine".to_string(), 1.0)]));
    }

    #[test]
    fn h3_takes_stepped_prompts_and_refuses_what_it_cant_combine() {
        let pieces = prompt_pieces(&ParsedPrompt::parse("a lake <alternate:at dawn, at dusk>"), 4, Family::H3).unwrap();
        assert_eq!(pieces.len(), 4);
        assert!(prompt_pieces(&ParsedPrompt::parse("a lake<break>pines"), 4, Family::H3).unwrap_err().contains("token tags"));
        let shape = h3::Shape { t: 32, h: 4, w: 4, audio_t: 40 };
        let mut w = h3window::ContextWindows::from_frames(56, 22, h3window::Fuse::Pyramid, true).unwrap();
        assert!(w.applies(&shape));
        let one = vec![Cond::plain(Prompt::text("a lake"))];
        let two = vec![one[0].clone(), Cond { region: Region { start_percent: 0.5, end_percent: 1.0, mask: None }, ..one[0].clone() }];
        let (plain, plain2) = (vec![H3Conds::default()], vec![H3Conds::default(); 2]);
        // windows run every entry, as ComfyUI's handler mixes them per window
        assert!(h3_combinations(Some(&w), false, &shape, &two, &plain2, &one).is_ok());
        // split_conds_to_windows picks an entry per window, which only matters with several
        w.split_conds = true;
        assert!(h3_combinations(Some(&w), false, &shape, &two, &plain2, &one).is_err());
        assert!(h3_combinations(Some(&w), false, &shape, &one, &plain, &two).is_err());
        assert!(h3_combinations(Some(&w), false, &shape, &one, &plain, &one).is_ok());
        // short clips sample without windows
        assert!(h3_combinations(Some(&w), false, &h3::Shape { t: 7, ..shape }, &two, &plain2, &one).is_ok());
        let keyframe = vec![H3Conds { keyframes: vec![(0, vec![0.0; 24 * 16])], ..H3Conds::default() }];
        assert!(h3_combinations(Some(&w), false, &shape, &one, &keyframe, &one).is_err());
        assert!(h3_combinations(None, true, &shape, &one, &keyframe, &one).is_err());
        assert!(h3_combinations(None, false, &shape, &one, &keyframe, &one).is_ok());
    }

    #[test]
    fn anima_takes_weights_from_both_syntaxes() {
        let chunks = whole_chunks(&ParsedPrompt::parse("lake, <weight[1.3]:pine> (sky:1.2)"), Family::Anima).unwrap();
        let weight = |t: &str| chunks[0].iter().find(|(s, _)| s.contains(t)).unwrap().1;
        assert!((weight("pine") - 1.3).abs() < 1e-6);
        assert!((weight("sky") - 1.2).abs() < 1e-6);
        assert_eq!(weight("lake"), 1.0);
    }

    #[test]
    fn moss_and_h3_weight_segments_and_keep_emphasis_literal() {
        for family in [Family::Moss, Family::H3] {
            let chunks = whole_chunks(&ParsedPrompt::parse("a (red:1.2) <weight[1.5]:lake>"), family).unwrap();
            assert_eq!(chunks, [[("a (red:1.2) ".to_string(), 1.0), ("lake".to_string(), 1.5)]]);
        }
    }

    #[test]
    fn sd_swarm_pieces_are_literal_with_breaks() {
        // SwarmTextEncodeAdvanced tokenizes each piece with weights off: parentheses stay, escapes lose their backslash
        let chunks = whole_chunks(&ParsedPrompt::parse(r"a (red:1.2) \(b\) <weight[1.5]:lake><break>c"), Family::Sd).unwrap();
        assert_eq!(chunks, vec![vec![("a (red:1.2) (b) ".to_string(), 1.0), ("lake".to_string(), 1.5)], vec![("c".to_string(), 1.0)]]);
    }

    #[test]
    fn krea2_takes_a_segment_per_leaf_with_its_weight() {
        // SwarmTextEncodeAdvanced tokenizes each leaf apart; the pipeline weights them
        let chunks = whole_chunks(&ParsedPrompt::parse("a <weight[1.5]:red> boat"), Family::Krea2).unwrap();
        assert_eq!(chunks, [[("a ".to_string(), 1.0), ("red".to_string(), 1.5), (" boat".to_string(), 1.0)]]);
        let pieces = prompt_pieces(&ParsedPrompt::parse("lake <alternate:pine, oak>"), 4, Family::Krea2).unwrap();
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[1].chunks, [[("lake ".to_string(), 1.0), ("oak".to_string(), 1.0)]]);
    }

    #[test]
    fn breaks_become_chunks_on_krea2_and_moss() {
        for family in [Family::Krea2, Family::Moss] {
            let chunks = whole_chunks(&ParsedPrompt::parse("a<break>b"), family).unwrap();
            assert_eq!(chunks, [[("a".to_string(), 1.0)], [("b".to_string(), 1.0)]]);
        }
    }

    #[test]
    fn qwen21_prompts_keep_each_piece_and_its_weight() {
        let s = |t: &str, w: f32| (t.to_string(), w);
        let chunks = whole_chunks(&ParsedPrompt::parse("a, b"), Family::Qwen21).unwrap();
        assert_eq!(chunks, [[s("a, b", 1.0)]]);
        // SwarmUI's leaves, from its own parser on these prompts
        let chunks = whole_chunks(&ParsedPrompt::parse("lake, <weight[1.3]:pine <weight[2]:bark>> (sky:1.2)"), Family::Qwen21).unwrap();
        assert_eq!(chunks, [[s("lake, ", 1.0), s("pine ", 1.3), s("bark", 2.6), s(" (sky:1.2)", 1.0)]]);
        let chunks = whole_chunks(&ParsedPrompt::parse("hills<break><weight[0.5]:fog> and <weight[1.5]:rain>"), Family::Qwen21).unwrap();
        assert_eq!(chunks, vec![vec![s("hills", 1.0)], vec![s("fog", 0.5), s(" and ", 1.0), s("rain", 1.5)]]);
        // the tokenizer drops the backslashes, not the parser
        assert_eq!(whole_chunks(&ParsedPrompt::parse(r"a \(b\) c"), Family::Qwen21).unwrap(), [[s(r"a \(b\) c", 1.0)]]);
        assert_eq!(Family::Qwen21.latent(), (64, 16));
    }

    #[test]
    fn preview_modes_pick_images_like_swarm() {
        // (first, images) of the run, the batch, the step
        assert_eq!(preview_image(PreviewMode::None, (0, 1), 1, 0), None);
        assert_eq!(preview_image(PreviewMode::One, (0, 2), 2, 0), Some(0));
        assert_eq!(preview_image(PreviewMode::Second, (0, 2), 2, 0), Some(1));
        assert_eq!(preview_image(PreviewMode::Second, (0, 1), 1, 0), Some(0));
        // a run of the second half of the batch
        assert_eq!(preview_image(PreviewMode::One, (2, 2), 4, 0), None);
        assert_eq!(preview_image(PreviewMode::Default, (2, 2), 4, 0), Some(0));
        assert_eq!(preview_image(PreviewMode::Iterate, (0, 3), 3, 4), Some(1));
    }

    #[test]
    fn noise_masks_resize_to_every_channel_and_image() {
        let mask = Mask { count: 1, height: 2, width: 2, data: Arc::new(vec![0.0, 1.0, 0.0, 1.0]) };
        let m = denoise_mask(&mask, LatentFormat::Image16Ch, (2, 3, 2, 4)).unwrap();
        assert_eq!(m.len(), 2 * 3 * 2 * 4);
        // torch's bilinear, align_corners=False: [0, 0.25, 0.75, 1] across
        assert!(m.chunks_exact(4).all(|row| row == [0.0, 0.25, 0.75, 1.0]), "{m:?}");
        let two = Mask { count: 2, height: 1, width: 1, data: Arc::new(vec![0.0, 1.0]) };
        assert_eq!(denoise_mask(&two, LatentFormat::Flux2, (3, 1, 1, 1)).unwrap(), [0.0, 1.0, 0.0]);
        assert!(matches!(denoise_mask(&two, LatentFormat::Image16Ch, (2, 1, 1, 1)), Err(EngineError::Unsupported(_))));
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

    /// SwarmTextEncodeAdvanced's conditioning of `prompt` at 10 steps from `text_encoder`, as `encode_text` gives it.
    fn swarm_cond(prompt: &str, text_encoder: &str, family: Family, sdxl_size: Option<SdxlSize>) -> Conditioning {
        let prompt = prompt_pieces(&ParsedPrompt::parse(prompt), 10, family);
        let cond = DeferredCond { text_encoder: text_encoder.into(), prompt, images: Vec::new(), loras: Vec::new(), minimax: None, clip_layer: None, sdxl_size, leaves: true };
        Conditioning::single(Opaque::new(cond), CondMeta::default())
    }

    #[test]
    fn sd_entries_carry_their_own_size_conditioning() {
        let size = |width, height| SdxlSize { width, height, crop_w: 0, crop_h: 0, target_width: width, target_height: height };
        // a stepped prompt at 1024 and a region of another at 512x768, joined as ConditioningCombine does
        let stepped = swarm_cond("<fromto[0.5]:lake, pines>", "sdxl.safetensors", Family::Sd, Some(size(1024, 1024)));
        let mut region = swarm_cond("a stone bridge", "sdxl.safetensors", Family::Sd, Some(size(512, 768)));
        let mask = Mask { count: 1, height: 2, width: 2, data: Arc::new(vec![1.0, 0.0, 1.0, 0.0]) };
        region.entries[0].meta.mask = Some(CondMask { mask, strength: 0.5, set_area_to_bounds: false });
        // CLIPTextEncode sets no size: the latent's
        let plain = swarm_cond("fog", "sdxl.safetensors", Family::Sd, None);
        let all = Conditioning { entries: [stepped.entries, region.entries, plain.entries].concat() };
        let (entries, sizes) = sd_entries(conds(&all, Family::Sd, (4, 6)).unwrap(), (4, 6));
        let text = |c: &Cond| c.prompt.chunks[0][0].0.clone();
        assert_eq!(entries.iter().map(text).collect::<Vec<_>>(), ["lake", "pines", "a stone bridge", "fog"]);
        assert_eq!((entries[0].region.start_percent, entries[1].region.end_percent), (-0.001, 1.0));
        assert!(entries[2].region.mask.as_ref().is_some_and(|m| m.len() == 4 * 6) && entries[3].region.mask.is_none());
        let wh = |s: &SizeCond| (s.width, s.height, s.target_width, s.target_height);
        assert_eq!(sizes.iter().map(wh).collect::<Vec<_>>(),
            [(1024.0, 1024.0, 1024.0, 1024.0), (1024.0, 1024.0, 1024.0, 1024.0), (512.0, 768.0, 512.0, 768.0), (48.0, 32.0, 48.0, 32.0)]);
    }

    #[test]
    fn sd_takes_attention_weights_only_from_its_own_clip() {
        let checkpoint = Path::new("sdxl.safetensors");
        assert!(sd_attn_token_weights(None, checkpoint).is_ok());
        // SD's CLIP keeps (x:2) weights, so SwarmTextEncodeAdvanced gives its prompts no attention weights
        let own = swarm_cond("lake, <weight[1.5]:pines>", "sdxl.safetensors", Family::Sd, None);
        assert!(sd_attn_token_weights(Some(&own), checkpoint).is_ok());
        let other = swarm_cond("lake, <weight[1.5]:pines>", "qwen35.safetensors", Family::Moss, None);
        assert!(matches!(sd_attn_token_weights(Some(&other), checkpoint), Err(EngineError::Unsupported(_))));
    }
}
