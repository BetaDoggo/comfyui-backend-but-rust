//! diffusion: Rust + CUDA reimplementation of ComfyUI's generation, for quantized checkpoints of Krea 2, Anima, MOSS Image and MiniMax H3.

use anyhow::{bail, Context, Result};
use clap::Parser;
use diffusion::samplers::ModelSampling;
use diffusion::pipeline::Prompt;
use diffusion::{pipeline, st};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;


#[derive(Parser, Debug, Clone)]
#[command(version, about = "Int8 text-to-image for Krea 2, Anima and MOSS Image, ComfyUI-compatible")]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "a red fox sitting in a snowy forest, golden hour, photograph")]
    prompt: String,
    /// Seed (same noise as ComfyUI's KSampler for the same seed)
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long, default_value_t = 1024)]
    height: usize,
    #[arg(long, default_value_t = 8)]
    steps: usize,
    /// ModelSamplingFlux shift
    #[arg(long, default_value_t = 1.15)]
    shift: f64,
    /// File with one prompt per line (models are loaded once for all of them)
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Images per prompt (seed, seed+1, ...)
    #[arg(short = 'n', long, default_value_t = 1)]
    count: usize,
    /// Images of a prompt sampled together, through the model at once. Each keeps its seed's noise; the sampler's own
    /// noise is one tensor for the batch, as in ComfyUI
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Output path; with several images, "_<i>" is inserted before the extension
    #[arg(short, long, default_value = "output.png")]
    out: PathBuf,
    /// Folder holding Stable-diffusion/, clip/ and VAE/ (default: $MODELS_ROOT, else ./models)
    #[arg(long)]
    models: Option<PathBuf>,
    /// Diffusion model, instead of the default file in the models folder
    #[arg(long)]
    dit: Option<PathBuf>,
    /// Text encoder, instead of the default file in the models folder
    #[arg(long)]
    te: Option<PathBuf>,
    /// VAE, instead of the default file in the models folder
    #[arg(long)]
    vae: Option<PathBuf>,
    /// VRAM (MB) to leave untouched for the desktop / other apps (default 256, as ComfyUI's --reserve-vram; 512 with
    /// --legacy-offload)
    #[arg(long)]
    reserve_vram: Option<usize>,
    /// Force the number of DiT blocks kept resident in VRAM (needs --legacy-offload)
    #[arg(long)]
    resident_blocks: Option<usize>,
    /// Cast buffers for streamed blocks (the streaming ring's slots with --legacy-offload)
    #[arg(long, default_value_t = 2)]
    slots: usize,
    /// Plan block residency from activation estimates, our older offloading, instead of the default ComfyUI
    /// DynamicVRAM policy (weights fault into VRAM as they run while there is room, activations push them back out)
    #[arg(long)]
    legacy_offload: bool,
    /// The default now; kept so existing command lines still work
    #[arg(long, hide = true, conflicts_with = "legacy_offload")]
    comfy_memory_mode: bool,
    /// Loader threads
    #[arg(long, default_value_t = 6)]
    threads: usize,
    /// FlashAttention tile variant (0: 128x64, 1: 128x32)
    #[arg(long, default_value_t = 0)]
    attn_variant: i32,
    /// Attention for the DiT blocks: "int8" (SageAttention-style INT8 attention from comfy-kitchen,
    /// fastest) or "flash" (bf16 FlashAttention-2, matches ComfyUI's default numerics most closely).
    /// Default int8 for every model
    #[arg(long)]
    attention: Option<String>,
    /// Use the noise stored in a ComfyUI reference dump (tools/ref_dump.py) instead of the seed
    #[arg(long)]
    ref_noise: Option<PathBuf>,
    /// Write intermediate tensors (safetensors) for parity checks
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Run Anima instead of Krea 2 (--dit and --te then default to the Anima files, --shift to 3)
    #[arg(long)]
    anima: bool,
    /// Anima: record each block's mean image token and mean token norm, per step, averaged over every image of the
    /// run, to this safetensors file (activation steering, see docs/STEERING.md)
    #[arg(long)]
    steer_record: Option<PathBuf>,
    /// Anima: steer the image tokens with this vector (tools/steer_build.py) or map (tools/steer_fit.py) file.
    /// Repeatable as FILE[:ALPHA[:STEPS[:BLOCKS]]], each part defaulting to --steer-alpha, --steer-steps, --steer-blocks
    #[arg(long)]
    steer: Vec<String>,
    /// Steering strength, in multiples of the vector file's `norm` (tools/steer_build.py: the recorded difference)
    #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
    steer_alpha: f32,
    /// Blocks steered, "a-b" with b exclusive (default every block)
    #[arg(long)]
    steer_blocks: Option<String>,
    /// Steps steered, "a-b" with b exclusive, counted from 0
    #[arg(long, default_value = "0-3")]
    steer_steps: String,
    /// Don't rescale steered tokens back to their norm
    #[arg(long)]
    steer_no_preserve: bool,
    /// Anima: start from this picture (PNG, resized to --width x --height) instead of empty noise: img2img
    #[arg(long)]
    init_image: Option<PathBuf>,
    /// With --init-image: how much of the schedule runs, as ComfyUI's KSampler denoise
    #[arg(long, default_value_t = 1.0)]
    denoise: f64,
    /// A training-free style reference picture (PNG), see docs/STEERING.md. Anima: read through shared self-attention
    /// and/or --style-adain. Krea 2: residual AdaIN only, at --style-adain (the node's 0.6; --style-steps 2-8 is its restyle)
    #[arg(long)]
    style_image: Option<PathBuf>,
    /// Weight on the reference's attention logits (1 = as the image's own tokens)
    #[arg(long, default_value_t = 1.0)]
    style_strength: f32,
    /// The reference pass's prompt (default: the node's for the model family)
    #[arg(long)]
    style_caption: Option<String>,
    /// Steps that read the reference, "a-b" with b exclusive
    #[arg(long, default_value = "0-8")]
    style_steps: String,
    /// Blocks that read the reference, "a-b" with b exclusive (default every block)
    #[arg(long)]
    style_blocks: Option<String>,
    /// The reference's RoPE frame: 0 overlays it on the target's positions, higher sets it apart in time
    #[arg(long, default_value_t = 0.0)]
    style_frame: f32,
    /// Don't read the reference's keys and values in self-attention (with --style-adain: statistics only)
    #[arg(long)]
    style_no_attention: bool,
    /// Residual AdaIN towards the reference's per-channel statistics, 0 to 1 (0 = off)
    #[arg(long, default_value_t = 0.0)]
    style_adain: f32,
    /// Blocks AdaIN runs after, "a-b" with b exclusive (default every block)
    #[arg(long)]
    style_adain_blocks: Option<String>,
    /// Shuffle the reference in square tiles of this many pixels (a multiple of 16; 0 = off): texture stays, objects don't
    #[arg(long, default_value_t = 0)]
    style_shuffle: usize,
    /// The reference is shrunk (never enlarged) to this many megapixels before encoding
    #[arg(long, default_value_t = 1.0)]
    style_megapixels: f64,
    /// Anima: concept steering in cross-attention, towards this prompt and away from --concept-null
    #[arg(long)]
    concept: Option<String>,
    #[arg(long, default_value = "")]
    concept_null: String,
    /// Concept steering's strength
    #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
    concept_alpha: f32,
    /// Concept steering's steps, "a-b" with b exclusive
    #[arg(long, default_value = "0-8")]
    concept_steps: String,
    /// Concept steering's blocks, "a-b" with b exclusive (default every block)
    #[arg(long)]
    concept_blocks: Option<String>,
    /// Anima: keep every --steer-token-stride-th image token of --steer-token-blocks at --steer-token-steps whole, to
    /// this safetensors file (for tools/steer_fit.py)
    #[arg(long)]
    steer_tokens: Option<PathBuf>,
    /// Blocks whose tokens are kept, comma separated
    #[arg(long, default_value = "8,13,18")]
    steer_token_blocks: String,
    /// Steps whose tokens are kept, "a-b" with b exclusive
    #[arg(long, default_value = "0-4")]
    steer_token_steps: String,
    #[arg(long, default_value_t = 33)]
    steer_token_stride: usize,
    /// Run MOSS Image instead of Krea 2 (--dit, --te and --vae then default to the MOSS files, --shift to 5)
    #[arg(long)]
    moss: bool,
    /// Run an SD 1.x checkpoint (--dit defaults to SD 1.5; its text encoder and VAE come from the same file unless
    /// --vae is given). --steps defaults to 20 and --cfg to 7
    #[arg(long)]
    sd: bool,
    /// Run an SDXL checkpoint, as --sd does (--dit defaults to SDXL base 1.0)
    #[arg(long)]
    sdxl: bool,
    /// SD 1.x and SDXL: CLIPSetLastLayer's stop_at_clip_layer, -1 for the last layer, -2 for "clip skip 2"
    #[arg(long, allow_hyphen_values = true)]
    clip_layer: Option<i32>,
    /// Run a bf16 MOSS Image checkpoint in bf16. By default its blocks are quantized to int8 ConvRot as it loads, for
    /// speed (in RAM; the file is only read)
    #[arg(long)]
    no_quant: bool,
    /// Faster paths whose images aren't byte-identical to the default: Anima's int8 cross-attention skips the prompt's
    /// zero padding (exact in the math), and MOSS Image's int8 blocks use a fast SiTU activation
    #[arg(long)]
    lossy: bool,
    /// Run MiniMax H3 (text to video with audio); --dit and --te then default to its files
    #[arg(long)]
    h3: bool,
    /// Run MiniMax-H3-Image (text to image); --dit, --te and --vae then default to its files, --steps to its 6
    #[arg(long)]
    h3_image: bool,
    /// Run Nemotron-Labs-Image (autoregressive text to image); --dit then defaults to its converted file. Takes --prompt,
    /// --prompt-file, --seed, --width, --height (multiples of 16), --count, --batch (up to 8) and --temperature
    #[arg(long)]
    nemotron: bool,
    /// Nemotron-Labs-Image: softmax temperature of the image tokens (the model card's 0.9); 0 takes the likeliest token
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    /// MiniMax H3: block-sparse attention, "sol" (Sol-Attn's adaptive threshold) or "sla" (top-k), as ComfyUI's
    /// BlockSparseAttention node
    #[arg(long)]
    sparse: Option<String>,
    /// Sol-Attn's threshold: higher is sparser
    #[arg(long, default_value_t = 1.3)]
    sparse_tau: f32,
    /// Top-k: percent of key blocks each query block keeps
    #[arg(long, default_value_t = 10.0)]
    sparse_keep: f64,
    /// Sparse attention starts this far into the schedule
    #[arg(long, default_value_t = 0.2)]
    sparse_start: f64,
    /// Sequences shorter than this stay dense
    #[arg(long, default_value_t = 12288)]
    sparse_min_tokens: usize,
    /// Extra top-scoring tokens each query block attends: 0 to 256, in steps of 64
    #[arg(long, default_value_t = 256)]
    sparse_extra_tokens: usize,
    /// MiniMax H3: sample in context windows of LENGTH,OVERLAP frames (pyramid blend, FreeNoise),
    /// as SwarmUI's Video Context Window, e.g. 124,39
    #[arg(long)]
    context_windows: Option<String>,
    /// MiniMax H3: frames at 24 fps, snapped up to the model's 17k + 5 grid
    #[arg(long, default_value_t = 5)]
    frames: usize,
    /// Negative prompt (used when --cfg isn't 1, and by the cfg_pp samplers)
    #[arg(long, default_value = "")]
    negative: String,
    /// CFG scale
    #[arg(long, default_value_t = 1.0)]
    cfg: f64,
    /// Any ComfyUI sampler, default er_sde for Anima, euler otherwise
    #[arg(long)]
    sampler: Option<String>,
    /// Any ComfyUI scheduler
    #[arg(long, default_value = "simple")]
    scheduler: String,
    /// LoRA file, `path`, `path:strength` or `path:strength:clip_strength`, in order (repeatable)
    #[arg(long)]
    lora: Vec<String>,
    /// Krea 2: PNG for the text encoder, at its own size; SwarmUI scales these to about 384x384. Anima with
    /// --snapref: a picture of the character. MiniMax-H3-Image: a picture to edit or refer to, <Picture n> in the
    /// prompt. MiniMax H3: a reference picture, <Picture n>, scaled to the clip's area (repeatable)
    #[arg(long)]
    image: Vec<PathBuf>,
    /// Anima: a SnapRef weight file, the reference adapter that takes --image as pictures of the character
    #[arg(long)]
    snapref: Option<PathBuf>,
    /// SnapRef: "cs" (character and style), "char" (character only) or "style" (art style only; needs an adapter with a
    /// style branch, such as v2s)
    #[arg(long, default_value = "cs")]
    snapref_mode: String,
    /// SnapRef: how much the pictures count, 1 as trained
    #[arg(long, default_value_t = 1.0)]
    snapref_strength: f32,
    /// SnapRef: below 1 the negative prompt reads the pictures too, weaker, so the prompt is followed more freely
    #[arg(long, default_value_t = 1.0)]
    snapref_fidelity: f32,
    /// SnapRef: the pictures are shrunk (never enlarged) to this many megapixels
    #[arg(long, default_value_t = 1.0)]
    snapref_megapixels: f64,
    /// Krea 2: PNG for a reference latent, for edit LoRAs (repeatable)
    #[arg(long)]
    ref_image: Vec<PathBuf>,
    /// MiniMax H3: PNG anchored as the clip's first frame, stretched to its size
    #[arg(long)]
    first_frame: Option<PathBuf>,
    /// MiniMax H3: PNG anchored as the clip's last frame, stretched to its size
    #[arg(long)]
    last_frame: Option<PathBuf>,
    /// MiniMax H3: a reference video (anything ffmpeg reads, taken at 24 fps), `<Video k>` in the prompt (repeatable)
    #[arg(long)]
    ref_video: Vec<PathBuf>,
    /// Krea 2: how reference latents are used, "index" or "index_timestep_zero"
    #[arg(long, default_value = "index")]
    ref_method: String,
    /// Pixal3D: an RGBA PNG whose alpha masks the object, turned into a vertex-coloured mesh written to --output (GLB)
    #[arg(long)]
    pixal3d: Option<PathBuf>,
    /// Pixal3D: horizontal field of view of the photo, in degrees
    #[arg(long, default_value_t = 49.13)]
    fov: f64,
    /// Pixal3D: voxel resolution of the refined shape, 1024 to 2048 in steps of 128
    #[arg(long, default_value_t = 1536)]
    resolution: usize,
    /// Pixal3D: skip the texture stage (grey mesh)
    #[arg(long)]
    no_texture: bool,
    /// Pixal3D: simplify the mesh to at most this many triangles, 0 to keep the full one
    #[arg(long, default_value_t = 700_000)]
    max_faces: usize,
    /// Pixal3D: the DINOv3 + NAF file (--vae is the shape VAE)
    #[arg(long)]
    clip_vision: Option<PathBuf>,
    /// Pixal3D: the texture VAE
    #[arg(long)]
    texture_vae: Option<PathBuf>,
    /// Pixal3D: compare every stage with a dump of tools/pixal3d_ref.py instead of generating
    #[arg(long)]
    pixal3d_check: Option<PathBuf>,
    /// Pixal3D: seeds of the structure, shape, refine and texture samplers, comma separated (default: --seed for
    /// the first three, --seed + 1 for the texture)
    #[arg(long)]
    pixal3d_seeds: Option<String>,
    /// TripoSplat / Idol: an RGBA PNG whose alpha masks the object, turned into a gaussian splat written to --output
    /// (PLY). --dit, --clip-vision and --vae pick the flow model, DINOv3 ViT-H and Flux 2 VAE
    #[arg(long)]
    triposplat: Option<PathBuf>,
    /// TripoSplat: the octree gaussian decoder
    #[arg(long)]
    splat_decoder: Option<PathBuf>,
    /// TripoSplat: sampler steps
    #[arg(long, default_value_t = 20)]
    splat_steps: usize,
    /// TripoSplat: CFG
    #[arg(long, default_value_t = 3.0)]
    splat_cfg: f64,
    /// TripoSplat: gaussians to decode, a multiple of 32 from 32768 to 1048576
    #[arg(long, default_value_t = 262_144)]
    gaussians: usize,
    /// TripoSplat: seed of the octree's point sampler (default: --seed)
    #[arg(long)]
    decode_seed: Option<u64>,
    /// TripoSplat: compare every stage with a dump of tools/triposplat_ref.py instead of generating
    #[arg(long)]
    triposplat_check: Option<PathBuf>,
    /// Background removal: write the PNG with its background made transparent to --output (needs the BiRefNet file)
    #[arg(long)]
    remove_background: Option<PathBuf>,
    /// BiRefNet checkpoint for --remove-background, and for --pixal3d on an image without transparency
    #[arg(long)]
    birefnet: Option<PathBuf>,
    /// BiRefNet: compare every stage with a dump of tools/birefnet_ref.py instead of running
    #[arg(long)]
    birefnet_check: Option<PathBuf>,
    /// Run Qwen Image 2.1 (--dit, --te and --vae then default to its files, --shift to its 0.69 at 1024x1024)
    #[arg(long)]
    qwen21: bool,
    /// Qwen Image 2.1: compare the VAE with a dump of tools/qwen21_ref.py --vae-only instead of generating
    #[arg(long)]
    qwen21_vae_check: Option<PathBuf>,
    /// MOSS Image: compare the Flux 2 VAE's encoder with a dump of tools/flux2_vae_ref.py instead of generating
    #[arg(long)]
    flux2_vae_check: Option<PathBuf>,
    /// SeedVR2: compare each stage with a dump of tools/seedvr_ref.py (with --dit and --vae) instead of generating
    #[arg(long)]
    seedvr_check: Option<PathBuf>,
    /// RelayVSR: compare the keyframe model and the Flash decoder with a dump of tools/relayvsr_ref.py (with --dit and
    /// --vae) instead of generating
    #[arg(long)]
    relayvsr_check: Option<PathBuf>,
}

/// RGB f32 [height, width, 3], 0 to 1.
fn read_png(path: &Path) -> Result<diffusion::vision::PromptImage> {
    let mut decoder = png::Decoder::new(std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?);
    decoder.set_transformations(png::Transformations::normalize_to_color8() | png::Transformations::ALPHA);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let channels = info.color_type.samples();
    let rgb = buf[..info.buffer_size()].chunks_exact(channels).flat_map(|px| {
        let (c, a) = (if channels >= 3 { [px[0], px[1], px[2]] } else { [px[0]; 3] }, px[channels - 1] as f32 / 255.0);
        // composite on white only where a pixel isn't opaque: `* 1.0 + 1.0 - 1.0` rounds values a hair below their 8-bit
        // level, and the lanczos resize (as ComfyUI's) truncates them back to the level below
        if channels % 2 == 0 && a < 1.0 { c.map(|v| v as f32 / 255.0 * a + 1.0 - a) } else { c.map(|v| v as f32 / 255.0) }
    }).collect();
    Ok(diffusion::vision::PromptImage { width: info.width as usize, height: info.height as usize, rgb })
}

/// `path`, `path:strength` or `path:strength:clip_strength`, as in `--lora`: the LoRAs for the
/// diffusion model and for the text encoder.
fn parse_loras(args: &Args) -> Result<(Vec<diffusion::lora::Lora>, Vec<diffusion::lora::Lora>)> {
    let (mut model, mut clip) = (Vec::new(), Vec::new());
    for l in &args.lora {
        let (mut path, mut values) = (l.as_str(), Vec::new());
        while values.len() < 2 {
            match path.rsplit_once(':').map(|(p, v)| (p, v.parse::<f32>())) {
                Some((p, Ok(v))) => {
                    values.insert(0, v);
                    path = p;
                }
                _ => break,
            }
        }
        if !std::path::Path::new(path).exists() {
            bail!("no LoRA at {path}");
        }
        let strength = values.first().copied().unwrap_or(1.0);
        model.push(diffusion::lora::Lora { path: PathBuf::from(path), strength });
        clip.push(diffusion::lora::Lora { path: PathBuf::from(path), strength: values.get(1).copied().unwrap_or(strength) });
    }
    Ok((model, clip))
}

fn apply_loras(gen: &mut pipeline::Generator, args: &Args) -> Result<()> {
    let (model, clip) = parse_loras(args)?;
    gen.set_loras(&model)?;
    gen.set_text_encoder_loras(&clip)
}

const KREA_DIT: &str = "Stable-diffusion/kroma-v0.3.1-turbo-opd-W8A8.safetensors";
const KREA_TE: &str = "clip/qwen3vl_4b.safetensors";
const QWEN_VAE: &str = "VAE/qwen_image_vae.safetensors";
const ANIMA_DIT: &str = "Stable-diffusion/animaInt8Mxfp8_turboV11Int8.safetensors";
const ANIMA_TE: &str = "clip/qwen_3_600m.safetensors";
const MOSS_DIT: &str = "Stable-diffusion/moss_image_sft_int8_convrot_attn.safetensors";
const MOSS_TE: &str = "clip/qwen3.5_9b_int8_convrot.safetensors";
const SD1_DIT: &str = "Stable-diffusion/v1-5-pruned-emaonly-fp16.safetensors";
const SDXL_DIT: &str = "Stable-diffusion/sd_xl_base_1.0.safetensors";
const MOSS_VAE: &str = "VAE/flux2-vae.safetensors";
const SEEDVR_DIT: &str = "diffusion_models/SeedVR2/seedvr2_3b_int8_convrot.safetensors";
const SEEDVR_VAE: &str = "VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors";
const NEMOTRON: &str = "diffusion_models/Nemotron-Labs-Image-30B-A3B-nvfp4.safetensors";
const RELAYVSR_DIT: &str = "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors";
const RELAYVSR_VAE: &str = "VAE/relayvsr_flash_vae_x2_fp32.safetensors";
const QWEN21_DIT: &str = "Stable-diffusion/rayQwen21_v1Int8Convrot.safetensors";
const QWEN21_TE: &str = "clip/qwen3vl_8b_int8_convrot.safetensors";
const QWEN21_VAE: &str = "VAE/QwenImage/qwen_image_2.1_vae_bf16.safetensors";
const BIREFNET: &str = "background_removal/birefnet.safetensors";
const PIXAL_DIT: &str = "diffusion_models/Pixal3D/pixal3d_int8_convrot.safetensors";
const PIXAL_DINO: &str = "clip_vision/dino_v3_L_naf_fp32.safetensors";
const PIXAL_SHAPE_VAE: &str = "VAE/Pixal3D/trellis_2_shape_vae_bf16.safetensors";
const PIXAL_TEX_VAE: &str = "VAE/Pixal3D/trellis_2_texture_vae_bf16.safetensors";
const SPLAT_DIT: &str = "diffusion_models/idol_fp16.safetensors";
const SPLAT_DINO: &str = "clip_vision/dino_v3_vit_h_bf16.safetensors";
const SPLAT_DECODER: &str = "VAE/idol_decoder_fp16.safetensors";

impl Args {
    /// `--attention`: INT8 unless flash is asked for.
    fn int8_attention(&self) -> Result<bool> {
        match self.attention.as_deref().unwrap_or("int8") {
            "flash" => Ok(false),
            "int8" | "sage" => Ok(true),
            a => bail!("unknown --attention {a} (flash | int8)"),
        }
    }

    fn sampling(&self, default: &str, ms: ModelSampling, seed: u64, latent: (usize, usize)) -> Result<pipeline::Sampling> {
        pipeline::Sampling::new(self.sampler.as_deref().unwrap_or(default), &self.scheduler, ms, self.steps, self.cfg, seed, latent)
    }

    /// The sampling of images `first..first + k` of a prompt, sampled together.
    fn batch_sampling(&self, default: &str, ms: ModelSampling, (first, k): (usize, usize), latent: (usize, usize)) -> Result<pipeline::Sampling> {
        let mut s = self.sampling(default, ms, self.seed + first as u64, latent)?;
        (s.batch_size, s.images) = (k, k);
        Ok(s)
    }

    /// A prompt's images in the groups `--batch` samples together: (first, count).
    fn batches(&self) -> Vec<(usize, usize)> {
        let b = self.batch.max(1);
        (0..self.count).step_by(b).map(|i| (i, b.min(self.count - i))).collect()
    }

    /// The noise of images `first..first + k`, `n` values each: each its seed's, or the reference dump's.
    fn noise(&self, (first, k): (usize, usize), n: usize) -> Result<Vec<f32>> {
        match &self.ref_noise {
            Some(p) if k == 1 => st::SafeTensors::open(p)?.f32_vec("noise", false),
            Some(_) => bail!("--ref-noise takes --batch 1"),
            None => Ok((first..first + k).flat_map(|i| diffusion::sampling::torch_randn(self.seed + i as u64, n)).collect()),
        }
    }

    /// `path`, or `default` inside the models folder.
    fn model(&self, path: &Option<PathBuf>, default: &str) -> PathBuf {
        path.clone().unwrap_or_else(|| {
            let root = self.models.clone().or_else(|| std::env::var_os("MODELS_ROOT").map(PathBuf::from));
            root.unwrap_or_else(|| PathBuf::from("models")).join(default)
        })
    }
}

fn run_anima(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    let (dit, te) = (args.model(&args.dit, ANIMA_DIT), args.model(&args.te, ANIMA_TE));
    let shift = if args.shift == 1.15 { 3.0 } else { args.shift };
    let mut gen = pipeline::Generator::new(&load)?;
    gen.ensure_vae(&args.model(&args.vae, QWEN_VAE))?;
    gen.ensure_anima(&dit)?;
    gen.ensure_qwen3(&te)?;
    apply_loras(&mut gen, args)?;
    let snapref = match &args.snapref {
        Some(adapter) => {
            if args.image.is_empty() {
                bail!("--snapref needs at least one --image");
            }
            let mut references = Vec::new();
            for p in &args.image {
                let img = read_png(p)?;
                let (rgb, w, h) = diffusion::snapref::shrink(&img.rgb, img.width, img.height, args.snapref_megapixels);
                references.push(gen.encode_image(&rgb, w, h)?);
            }
            let strength = args.snapref_strength;
            if strength <= 0.0 {
                bail!("--snapref-strength must be above 0");
            }
            let fidelity = args.snapref_fidelity;
            Some(pipeline::SnapRefRequest {
                adapter: adapter.clone(),
                biases: vec![strength.ln(); references.len()],
                references,
                mode: args.snapref_mode.clone(),
                caption: Prompt::new(diffusion::tokenizer::token_weights("", 1.0)),
                in_uncond: false,
                uncond_bias: (fidelity < 1.0).then(|| (1.0 - fidelity).max(1e-6).ln()),
            })
        }
        None => None,
    };
    if args.steer_record.is_some() {
        gen.record_anima(args.steps)?;
    }
    if args.steer_tokens.is_some() {
        let blocks = args.steer_token_blocks.split(',').map(|b| b.trim().parse::<usize>()).collect::<Result<Vec<_>, _>>()?;
        gen.sample_anima_tokens(blocks, diffusion::steer::parse_range(&args.steer_token_steps)?, args.steer_token_stride)?;
    }
    if let Some(p) = &args.style_image {
        let img = read_png(p)?;
        let (rgb, w, h) = diffusion::snapref::shrink(&img.rgb, img.width, img.height, args.style_megapixels);
        let reference = gen.encode_image(&rgb, w, h)?;
        if args.style_strength <= 0.0 {
            bail!("--style-strength must be above 0");
        }
        let steps = diffusion::steer::parse_range(&args.style_steps)?;
        let blocks = args.style_blocks.as_deref().map(diffusion::steer::parse_range).transpose()?.unwrap_or((0, usize::MAX));
        eprintln!("style reference {} at {}x{}, strength {}, steps {steps:?}, blocks {blocks:?}, frame {}", p.display(), w, h,
            args.style_strength, args.style_frame);
        gen.style_anima(Some(pipeline::StyleRequest {
            reference,
            caption: Prompt::new(diffusion::tokenizer::token_weights(
                args.style_caption.as_deref().unwrap_or(pipeline::StyleRequest::NODE_CAPTION), 1.0)),
            bias: args.style_strength.ln(),
            steps,
            blocks,
            frame: args.style_frame,
            seed: args.seed ^ 0x5eed,
            attention: !args.style_no_attention,
            adain: args.style_adain,
            shuffle: args.style_shuffle / 8,
            adain_blocks: args.style_adain_blocks.as_deref().map(diffusion::steer::parse_range).transpose()?.unwrap_or((0, usize::MAX)),
        }))?;
    }
    if let Some(c) = &args.concept {
        let p = |t: &str| Prompt::new(diffusion::tokenizer::token_weights(t, 1.0));
        let steps = diffusion::steer::parse_range(&args.concept_steps)?;
        let blocks = args.concept_blocks.as_deref().map(diffusion::steer::parse_range).transpose()?;
        gen.concept_anima(&p(c), &p(&args.concept_null), args.concept_alpha, steps, blocks)?;
        eprintln!("concept steering towards {c:?} at alpha {} over steps {steps:?}, blocks {blocks:?}", args.concept_alpha);
    }
    for spec in &args.steer {
        // FILE[:ALPHA[:STEPS[:BLOCKS]]]; a drive letter's colon stays in the file
        let (file, rest) = match spec.find(".safetensors") {
            Some(i) => spec.split_at(i + ".safetensors".len()),
            None => (spec.as_str(), ""),
        };
        let parts: Vec<&str> = rest.split(':').skip(1).collect();
        let alpha = match parts.first() {
            Some(a) => a.parse::<f32>()?,
            None => args.steer_alpha,
        };
        let steps = diffusion::steer::parse_range(parts.get(1).copied().unwrap_or(&args.steer_steps))?;
        let blocks = parts.get(2).copied().or(args.steer_blocks.as_deref()).map(diffusion::steer::parse_range).transpose()?;
        gen.steer_anima(Path::new(file), alpha, blocks, steps, !args.steer_no_preserve)?;
        eprintln!("steering with {file} at alpha {alpha} over blocks {blocks:?}, steps {steps:?}");
    }
    let n = 16 * (args.height / 8) * (args.width / 8);
    let init = match &args.init_image {
        Some(p) => {
            let img = read_png(p)?;
            let rgb = diffusion::pixal3d::lanczos_f32(&img.rgb, img.width, img.height, 3, args.width, args.height);
            Some(gen.encode_image(&rgb, args.width, args.height)?.latent)
        }
        None => None,
    };
    let prompts = args.prompts()?;
    let total = prompts.len() * args.count;
    for (pi, prompt) in prompts.iter().enumerate() {
    for group in args.batches() {
        let r = pipeline::AnimaRequest {
            positive: vec![pipeline::Cond::plain(Prompt::new(diffusion::tokenizer::token_weights(prompt, 1.0)))],
            negative: vec![pipeline::Cond::plain(Prompt::new(diffusion::tokenizer::token_weights(&args.negative, 1.0)))],
            width: args.width,
            height: args.height,
            sampling: {
                let mut s = args.batch_sampling("er_sde", ModelSampling::Flow { shift }, group, (args.width / 8, args.height / 8))?;
                if let Some(l) = &init {
                    let (sampler, sched, lat) = (s.sampler.clone(), args.scheduler.clone(), (args.width / 8, args.height / 8));
                    let schedule = |k| diffusion::samplers::sigmas(ModelSampling::Flow { shift }, &sampler, &sched, k, lat, None);
                    s.sigmas = diffusion::samplers::ksampler_sigmas(args.steps, args.denoise, 0, args.steps, false, schedule)?
                        .context("--denoise leaves no step")?;
                    s.latent = Some(l.repeat(group.1));
                }
                s
            },
            noise: args.noise(group, n)?,
            snapref: snapref.clone(),
            style: None,
        };
        let mut dump = pipeline::Dump::new();
        let t0 = Instant::now();
        let (latents, _) = gen.sample_anima(&r, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
        for (j, latent) in latents.chunks_exact(n).enumerate() {
            let i = group.0 + j;
            let rgb = gen.decode_latent(latent, args.width, args.height)?;
            let path = out_path(&args.out, pi * args.count + i, total);
            std::fs::write(&path, pipeline::encode_png(args.width, args.height, &rgb)?)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("[{}] saved {} (seed {})", pi * args.count + i, path.display(), args.seed + i as u64);
        }
        eprintln!("{} image(s) done in {:.2}s", group.1, t0.elapsed().as_secs_f64());
        if let Some(d) = &args.dump {
            write_safetensors(d, &dump)?;
        }
    }
    }
    if let Some(p) = &args.steer_record {
        gen.save_anima_record(p)?;
        eprintln!("steering statistics written to {}", p.display());
    }
    if let Some(p) = &args.steer_tokens {
        gen.save_anima_tokens(p)?;
        eprintln!("tokens written to {}", p.display());
    }
    Ok(())
}

fn run_moss(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    let (dit, te, vae) = (args.model(&args.dit, MOSS_DIT), args.model(&args.te, MOSS_TE), args.model(&args.vae, MOSS_VAE));
    let shift = if args.shift == 1.15 { 5.0 } else { args.shift };
    let mut gen = pipeline::Generator::new(&load)?;
    gen.ensure_moss(&dit)?;
    gen.ensure_qwen35(&te)?;
    gen.ensure_flux2_vae(&vae)?;
    apply_loras(&mut gen, args)?;
    let n = 128 * (args.height / 16) * (args.width / 16);
    let prompts = args.prompts()?;
    for (p, prompt) in prompts.iter().enumerate() {
        for group in args.batches() {
            let r = pipeline::MossRequest {
                positive: vec![pipeline::Cond::plain(Prompt::text(prompt.trim()))],
                negative: vec![pipeline::Cond::plain(Prompt::text(args.negative.trim()))],
                width: args.width,
                height: args.height,
                sampling: args.batch_sampling("euler", ModelSampling::Flow { shift }, group, (args.width / 16, args.height / 16))?,
                noise: args.noise(group, n)?,
            };
            let mut dump = pipeline::Dump::new();
            let t0 = Instant::now();
            let (latents, _) = gen.sample_moss(&r, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
            for (j, latent) in latents.chunks_exact(n).enumerate() {
                let i = p * args.count + group.0 + j;
                let t1 = Instant::now();
                let rgb = gen.decode_moss(latent, args.width, args.height)?;
                eprintln!("VAE decode {:.0} ms", t1.elapsed().as_secs_f64() * 1e3);
                let path = out_path(&args.out, i, prompts.len() * args.count);
                std::fs::write(&path, pipeline::encode_png(args.width, args.height, &rgb)?)
                    .with_context(|| format!("writing {}", path.display()))?;
                eprintln!("[{i}] saved {} (seed {})", path.display(), args.seed + (group.0 + j) as u64);
            }
            eprintln!("{} image(s) done in {:.2}s", group.1, t0.elapsed().as_secs_f64());
            if let Some(d) = &args.dump {
                write_safetensors(d, &dump)?;
            }
        }
    }
    Ok(())
}

fn run_sd(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    let dit = args.model(&args.dit, if args.sdxl { SDXL_DIT } else { SD1_DIT });
    let mut args = args.clone();
    if args.steps == 8 {
        args.steps = 20;
    }
    if args.cfg == 1.0 {
        args.cfg = 7.0;
    }
    let args = &args;
    let mut gen = pipeline::Generator::new(&load)?;
    gen.ensure_sd(&dit)?;
    if let Some(vae) = &args.vae {
        gen.ensure_sd_vae(vae)?;
    }
    if gen.sd_is_xl() != args.sdxl {
        bail!("{} is {}", dit.display(), if gen.sd_is_xl() { "SDXL: use --sdxl" } else { "not SDXL: use --sd" });
    }
    apply_loras(&mut gen, args)?;
    let (lw, lh) = (args.width / 8, args.height / 8);
    let n = 4 * lh * lw;
    let prompts = args.prompts()?;
    for (p, prompt) in prompts.iter().enumerate() {
        for group in args.batches() {
            let size = pipeline::SizeCond::of(args.width, args.height);
            let r = pipeline::SdRequest {
                positive: vec![pipeline::Cond::plain(Prompt::new(diffusion::tokenizer::token_weights(prompt.trim(), 1.0)))],
                negative: vec![pipeline::Cond::plain(Prompt::new(diffusion::tokenizer::token_weights(args.negative.trim(), 1.0)))],
                width: args.width,
                height: args.height,
                positive_size: vec![size],
                negative_size: vec![size],
                clip_layer: args.clip_layer,
                v_pred: None,
                sampling: args.batch_sampling("euler", ModelSampling::Discrete, group, (lw, lh))?,
                noise: args.noise(group, n)?,
            };
            let mut dump = pipeline::Dump::new();
            let t0 = Instant::now();
            let (latents, _) = gen.sample_sd(&r, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
            for (j, latent) in latents.chunks_exact(n).enumerate() {
                let i = p * args.count + group.0 + j;
                let t1 = Instant::now();
                let rgb = gen.decode_sd(latent, args.width, args.height)?;
                eprintln!("VAE decode {:.0} ms", t1.elapsed().as_secs_f64() * 1e3);
                let path = out_path(&args.out, i, prompts.len() * args.count);
                std::fs::write(&path, pipeline::encode_png(args.width, args.height, &rgb)?)
                    .with_context(|| format!("writing {}", path.display()))?;
                eprintln!("[{i}] saved {} (seed {})", path.display(), args.seed + (group.0 + j) as u64);
            }
            eprintln!("{} image(s) done in {:.2}s", group.1, t0.elapsed().as_secs_f64());
            if let Some(d) = &args.dump {
                write_safetensors(d, &dump)?;
            }
        }
    }
    Ok(())
}

fn run_qwen21(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    let (dit, te, vae) = (args.model(&args.dit, QWEN21_DIT), args.model(&args.te, QWEN21_TE), args.model(&args.vae, QWEN21_VAE));
    // ComfyUI's value for 1024x1024, whatever the size
    let shift = if args.shift == 1.15 { 0.69 } else { args.shift };
    let mut gen = pipeline::Generator::new(&load)?;
    gen.ensure_qwen21(&dit)?;
    gen.ensure_te(&te)?;
    gen.ensure_vae21(&vae)?;
    apply_loras(&mut gen, args)?;
    // each image is shown to the text encoder over white, and goes into the sequence as a reference latent
    let mut images = Vec::new();
    let mut references = Vec::new();
    for p in args.image.iter().chain(&args.ref_image) {
        let (rgba, w, h) = read_rgba(p)?;
        let f: Vec<f32> = rgba.iter().map(|v| *v as f32 / 255.0).collect();
        images.push(diffusion::vision::PromptImage {
            width: w,
            height: h,
            rgb: f.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]].map(|c| c * px[3] + 1.0 - px[3])).collect(),
        });
        references.push(gen.encode_image21(&f, w, h)?);
    }
    let n = 64 * (args.height / 16) * (args.width / 16);
    for group in args.batches() {
        let r = pipeline::Qwen21Request {
            positive: vec![pipeline::Cond::plain(Prompt::text(args.prompt.trim()))],
            negative: vec![pipeline::Cond::plain(Prompt::text(args.negative.trim()))],
            width: args.width,
            height: args.height,
            sampling: args.batch_sampling("euler", ModelSampling::Flux { shift }, group, (args.width / 16, args.height / 16))?,
            noise: args.noise(group, n)?,
            images: [vec![images.clone()], vec![images.clone()]],
            references: [vec![references.clone()], vec![references.clone()]],
            attn_token_weights: None,
        };
        let mut dump = pipeline::Dump::new();
        let t0 = Instant::now();
        let (latents, _) = gen.sample_qwen21(&r, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
        for (j, latent) in latents.chunks_exact(n).enumerate() {
            let i = group.0 + j;
            let t1 = Instant::now();
            let rgba = gen.decode_qwen21(latent, args.width, args.height)?;
            eprintln!("VAE decode {:.0} ms", t1.elapsed().as_secs_f64() * 1e3);
            let path = out_path(&args.out, i, args.count);
            write_rgba_png(&path, &rgba, args.width, args.height)?;
            eprintln!("[{i}] saved {} (seed {})", path.display(), args.seed + i as u64);
        }
        eprintln!("{} image(s) done in {:.2}s", group.1, t0.elapsed().as_secs_f64());
        if let Some(d) = &args.dump {
            write_safetensors(d, &dump)?;
        }
    }
    Ok(())
}

const H3_DIT: &str = "Stable-diffusion/minimax_h3_fl2va_pruned_w6a8.safetensors";
const H3_TE: &str = "clip/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
const H3_VAE: &str = "VAE/MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors";
const H3_AUDIO_VAE: &str = "VAE/MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors";
const H3_IMAGE_DIT: &str = "Stable-diffusion/minimax_h3_image_pruned_int8_convrot.safetensors";
const H3_IMAGE_VAE: &str = "VAE/MiniMaxH3/minimax_h3_image_vae_fp16.safetensors";

/// MiniMax-H3-Image: one image from H3's transformer without its audio, on the sigma grid its heads were distilled on.
fn run_h3_image(args: &Args, load: pipeline::LoadOptions, dit: PathBuf, te: PathBuf) -> Result<()> {
    use diffusion::h3;
    let t_all = Instant::now();
    if args.width % 32 != 0 || args.height % 32 != 0 {
        bail!("MiniMax-H3-Image needs a width and height in multiples of 32");
    }
    let grid: Vec<f32> = h3::image_sigmas(&st::SafeTensors::open(&dit)?)?.iter().map(|g| *g as f32).collect();
    let ms = ModelSampling::Grid(diffusion::samplers::SigmaGrid::new(&grid)?);
    let shape = h3::Shape::image(args.width, args.height);
    let mut gen = pipeline::Generator::new(&load)?;
    let t0 = Instant::now();
    gen.ensure_h3(&dit)?;
    gen.ensure_h3_te(&te)?;
    eprintln!("MiniMax-H3-Image ready in {:.1}s", t0.elapsed().as_secs_f64());
    let (model, _) = parse_loras(args)?;
    gen.set_loras(&model)?;
    let noise = match &args.ref_noise {
        Some(p) => st::SafeTensors::open(p)?.f32_vec("noise", false)?,
        None => diffusion::sampling::torch_randn(args.seed, shape.video_len()),
    };
    // the heads are distilled for their grid's steps: --steps left at its default takes them
    let steps = if args.steps == 8 { grid.len() - 1 } else { args.steps };
    let sampling = pipeline::Sampling::new(args.sampler.as_deref().unwrap_or("euler"), &args.scheduler, ms, steps, args.cfg, args.seed, (args.width / 16, args.height / 16))?;
    // MiniMaxH3ImageConditioning's edit pictures, through the image VAE
    let images = args.image.iter().map(|p| read_png(p)).collect::<Result<Vec<_>>>()?;
    if !args.ref_video.is_empty() {
        bail!("--ref-video on MiniMax-H3-Image");
    }
    let conds = pipeline::H3Conds {
        label: "Picture {}: ".into(),
        pictures: pipeline::image_pictures(&images, true),
        videos: Vec::new(),
        vae: Some(args.model(&args.vae, H3_IMAGE_VAE)),
        keyframes: Vec::new(),
    };
    let prompts = args.prompts()?;
    for (i, prompt) in prompts.iter().enumerate() {
        let r = pipeline::H3Request {
            positive: vec![pipeline::Cond::plain(Prompt::text(prompt))],
            negative: vec![pipeline::Cond::plain(Prompt::text(&args.negative))],
            shape,
            sampling: sampling.clone(),
            shifts: (h3::SHIFT_VIDEO, h3::SHIFT_AUDIO),
            noise: noise.clone(),
            sparse: None,
            windows: None,
            conds: vec![conds.clone()],
            seed: args.seed,
        };
        let mut dump = pipeline::Dump::new();
        let t0 = Instant::now();
        let (latent, _) = gen.sample_h3(&r, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
        eprintln!("sampled in {:.2}s", t0.elapsed().as_secs_f64());
        dump.insert("samples".into(), ("F32".into(), vec![latent.len()], st::f32_bytes(&latent).to_vec()));
        let t1 = Instant::now();
        let img = gen.decode_h3(&args.model(&args.vae, H3_IMAGE_VAE), &latent, shape, 1)?;
        eprintln!("decoded in {:.2}s", t1.elapsed().as_secs_f64());
        let (h, w) = (img.h, img.w);
        let mut rgb = vec![0u8; h * w * 3];
        let mut hwc = vec![0f32; h * w * 3];
        for c in 0..3 {
            for i in 0..h * w {
                let v = img.data[c * h * w + i];
                hwc[i * 3 + c] = v;
                rgb[i * 3 + c] = (v * 255.0).round() as u8;
            }
        }
        dump.insert("image".into(), ("F32".into(), vec![1, h, w, 3], st::f32_bytes(&hwc).to_vec()));
        if let Some(d) = &args.dump {
            write_safetensors(&out_path(d, i, prompts.len()), &dump)?;
        }
        let out = out_path(&args.out, i, prompts.len());
        std::fs::write(&out, pipeline::encode_png(w, h, &rgb)?)?;
        eprintln!("saved {} (seed {}), {:.1}s in all", out.display(), args.seed, t_all.elapsed().as_secs_f64());
    }
    Ok(())
}

fn run_nemotron(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    // what this model has no use for is refused rather than dropped
    let unused = [
        (!args.lora.is_empty(), "--lora"),
        (!args.image.is_empty() || !args.ref_image.is_empty(), "--image / --ref-image"),
        (args.init_image.is_some(), "--init-image"),
        (args.style_image.is_some() || args.snapref.is_some(), "--style-image / --snapref"),
        (args.te.is_some() || args.vae.is_some(), "--te / --vae (the file holds its tokenizer and decoder)"),
        (args.sampler.is_some(), "--sampler"),
        (args.ref_noise.is_some() || args.dump.is_some(), "--ref-noise / --dump"),
        (!args.steer.is_empty() || args.concept.is_some(), "--steer / --concept"),
    ];
    if let Some((_, flag)) = unused.iter().find(|(set, _)| *set) {
        bail!("Nemotron-Labs-Image doesn't take {flag}");
    }
    if args.batch > diffusion::nemotron::MAX_BATCH {
        bail!("Nemotron-Labs-Image samples up to {} images together (--batch)", diffusion::nemotron::MAX_BATCH);
    }
    let t_start = Instant::now();
    let mut gen = pipeline::Generator::new(&load)?;
    gen.ensure_nemotron(&args.model(&args.dit, NEMOTRON))?;
    let prompts = args.prompts()?;
    let n_jobs = prompts.len() * args.count;
    for (p, prompt) in prompts.iter().enumerate() {
        for (first, k) in args.batches() {
            let req = pipeline::NemotronRequest {
                prompt: prompt.clone(),
                width: args.width,
                height: args.height,
                seed: args.seed + first as u64,
                images: k,
                temperature: args.temperature,
            };
            let t0 = Instant::now();
            let mut last = Instant::now();
            let images = gen.generate_nemotron(&req, |done, of| {
                if last.elapsed().as_secs_f64() >= 5.0 || done == of {
                    eprintln!("  {done} / {of} tokens");
                    last = Instant::now();
                }
                true
            })?;
            for (j, rgb) in images.iter().enumerate() {
                let img_i = p * args.count + first + j;
                let path = out_path(&args.out, img_i, n_jobs);
                std::fs::write(&path, pipeline::encode_png(args.width, args.height, rgb)?).with_context(|| format!("writing {}", path.display()))?;
                eprintln!("[{img_i}] saved {} (seed {}) after {:.1}s", path.display(), req.seed + j as u64, t0.elapsed().as_secs_f64());
            }
        }
    }
    eprintln!("total {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}

fn run_h3(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    use diffusion::h3;
    let t_all = Instant::now();
    let (dit, te) = (args.model(&args.dit, H3_DIT), args.model(&args.te, H3_TE));
    if args.h3_image || h3::is_h3_image(|k| st::SafeTensors::open(&dit).is_ok_and(|f| f.has(k))) {
        let dit = if args.h3_image { args.model(&args.dit, H3_IMAGE_DIT) } else { dit };
        return run_h3_image(args, load, dit, te);
    }
    if args.batch > 1 {
        bail!("--batch on MiniMax H3: use --count for several clips, one after another");
    }
    if args.count > 1 && args.ref_noise.is_some() {
        bail!("--ref-noise takes --count 1");
    }
    let frames = h3::align_frames(args.frames);
    if args.width % 32 != 0 || args.height % 32 != 0 {
        bail!("MiniMax H3 needs a width and height in multiples of 32");
    }
    let shape = h3::Shape::new(args.width, args.height, frames);
    let mut gen = pipeline::Generator::new(&load)?;
    let t0 = Instant::now();
    gen.ensure_h3(&dit)?;
    gen.ensure_h3_te(&te)?;
    eprintln!("MiniMax H3 ready in {:.1}s", t0.elapsed().as_secs_f64());
    let (model, _) = parse_loras(args)?;
    gen.set_loras(&model)?;
    let windows = match args.context_windows.as_deref() {
        None => None,
        Some(s) => {
            let (l, o) = s.split_once(',').context("--context-windows takes LENGTH,OVERLAP in frames")?;
            Some(diffusion::h3window::ContextWindows::from_frames(l.trim().parse()?, o.trim().parse()?, diffusion::h3window::Fuse::Pyramid, true)?)
        }
    };
    // HyperFlow H3 Flash was distilled with euler; res_multistep also runs on it
    let default = if h3::hyperflow_config(&dit).is_some() { "euler" } else { "res_multistep" };
    let ms = ModelSampling::Flow { shift: h3::SHIFT_VIDEO };
    let method = match args.sparse.as_deref() {
        None => None,
        Some("sol") => Some(h3::SparseMethod::Sol { tau: args.sparse_tau }),
        Some("sla") => Some(h3::SparseMethod::TopK { ratio: args.sparse_keep / 100.0 }),
        Some(m) => bail!("--sparse takes sol or sla, not {m}"),
    };
    let sparse = method.map(|method| h3::SparseAttn {
        method,
        sigma_start: ms.percent_to_sigma(args.sparse_start),
        sigma_end: ms.percent_to_sigma(1.0),
        min_tokens: args.sparse_min_tokens,
        dense_blocks: Vec::new(),
        sinks: h3::SparseSinks::ExactKvAndRows,
        extra_tokens: args.sparse_extra_tokens,
        verbose: true,
    });
    // reference pictures as MiniMaxH3ReferenceToVideo's "match" sizes them, and keyframes as SwarmUI's node stretches
    // them, all through the video VAE
    let vae = args.model(&args.vae, H3_VAE);
    let images = args.image.iter().map(|p| read_png(p)).collect::<Result<Vec<_>>>()?;
    let mut keyframes = Vec::new();
    for (path, index) in [(&args.first_frame, 0), (&args.last_frame, frames - 1)] {
        if let Some(p) = path {
            let img = pipeline::h3_resize(&read_png(p)?, args.width, args.height);
            keyframes.push((index, gen.encode_h3(&vae, &img)?));
        }
    }
    let mut videos = Vec::new();
    for p in &args.ref_video {
        videos.push(pipeline::ref2va_video(&read_video(p)?, frames, true)?);
    }
    let conds = pipeline::H3Conds {
        label: "<Picture {}>: ".into(),
        pictures: pipeline::ref2va_pictures(&images, (args.width, args.height), false, true),
        videos,
        vae: Some(vae),
        keyframes,
    };
    // clips one after another, each prompt with seed, seed + 1, ..., with the models loaded once
    let prompts = args.prompts()?;
    let total = prompts.len() * args.count;
    for (pi, prompt) in prompts.iter().enumerate() {
        for i in 0..args.count {
            let seed = args.seed + i as u64;
            let out = out_path(&args.out, pi * args.count + i, total);
            let mut noise = match &args.ref_noise {
                Some(p) => {
                    let st = st::SafeTensors::open(p)?;
                    let mut v = st.f32_vec("noise0", false)?;
                    v.extend(st.f32_vec("noise1", false)?);
                    v
                }
                None => diffusion::sampling::torch_randn_streams(seed, &[shape.video_len(), shape.audio_len()]).concat(),
            };
            if let Some(w) = &windows {
                w.freenoise(&mut noise, &shape, seed);
            }
            let sampling = args.sampling(default, ModelSampling::Flow { shift: h3::SHIFT_VIDEO }, seed, (args.width / 16, args.height / 16))?;
            let r = pipeline::H3Request {
                positive: vec![pipeline::Cond::plain(Prompt::text(prompt))],
                negative: vec![pipeline::Cond::plain(Prompt::text(&args.negative))],
                shape,
                sampling,
                shifts: (h3::SHIFT_VIDEO, h3::SHIFT_AUDIO),
                noise,
                sparse: sparse.clone(),
                windows: windows.clone(),
                conds: vec![conds.clone()],
                seed,
            };
            let mut dump = pipeline::Dump::new();
            let t0 = Instant::now();
            let (latent, _) = gen.sample_h3(&r, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
            eprintln!("sampled in {:.2}s", t0.elapsed().as_secs_f64());
            dump.insert("samples".into(), ("F32".into(), vec![latent.len()], st::f32_bytes(&latent).to_vec()));
            if let Some(d) = &args.dump {
                write_safetensors(d, &dump)?;
            }
            let t1 = Instant::now();
            // a DWD adapter's pixel head already decoded the frames, in place of the VAE
            let clip = match gen.take_h3_dwd_pixels() {
                Some(c) => {
                    eprintln!("video from the DWD adapter's pixel head (no VAE decode)");
                    c
                }
                None => {
                    let c = gen.decode_h3(&args.model(&args.vae, H3_VAE), &latent, shape, frames)?;
                    eprintln!("video decoded in {:.2}s", t1.elapsed().as_secs_f64());
                    c
                }
            };
            let t1 = Instant::now();
            let wave = gen.decode_h3_audio(&args.model(&None, H3_AUDIO_VAE), &latent[shape.video_len()..], shape.audio_t)?;
            eprintln!("audio decoded in {:.2}s", t1.elapsed().as_secs_f64());
            let (h, w) = (clip.h, clip.w);
            let mut frames_hwc = vec![0f32; frames * h * w * 3];
            let mut rgb = vec![0u8; frames * h * w * 3];
            for c in 0..3 {
                for f in 0..frames {
                    for i in 0..h * w {
                        let v = clip.data[((c * frames + f) * h) * w + i];
                        frames_hwc[(f * h * w + i) * 3 + c] = v;
                        rgb[(f * h * w + i) * 3 + c] = (v * 255.0).round() as u8;
                    }
                }
            }
            let samples = wave.len() / 2;
            let stereo: Vec<f32> = (0..samples).flat_map(|i| [wave[i], wave[samples + i]]).collect();
            dump.insert("frames".into(), ("F32".into(), vec![1, frames, h, w, 3], st::f32_bytes(&frames_hwc).to_vec()));
            dump.insert("wave".into(), ("F32".into(), vec![1, samples, 2], st::f32_bytes(&stereo).to_vec()));
            if let Some(d) = &args.dump {
                write_safetensors(d, &dump)?;
            }
            write_video(&out, &rgb, w, h, &stereo)?;
            eprintln!("saved {} ({frames} frames at 24 fps, seed {seed}), {:.1}s in all", out.display(), t_all.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

/// The frames of a video at 24 fps through ffmpeg, RGB 0 to 1.
fn read_video(path: &Path) -> Result<Vec<diffusion::vision::PromptImage>> {
    let probe = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .context("running ffprobe")?;
    let dims = String::from_utf8_lossy(&probe.stdout);
    let (w, h) = dims.trim().split_once(',').with_context(|| format!("{}: no video stream", path.display()))?;
    let (w, h): (usize, usize) = (w.trim().parse()?, h.trim().parse()?);
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-r", "24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .context("running ffmpeg")?;
    if !out.status.success() {
        bail!("ffmpeg failed on {}", path.display());
    }
    let frames: Vec<_> = out.stdout.chunks_exact(w * h * 3).map(|f| diffusion::vision::PromptImage { width: w, height: h, rgb: f.iter().map(|&v| v as f32 / 255.0).collect() }).collect();
    eprintln!("{}: {} frames of {w}x{h}", path.display(), frames.len());
    Ok(frames)
}

/// Encode with ffmpeg, which reads the frames from a pipe and the audio from a WAV next to the output.
fn write_video(out: &Path, rgb: &[u8], w: usize, h: usize, stereo: &[f32]) -> Result<()> {
    use std::io::Write;
    let wav = out.with_extension("wav");
    let data = (stereo.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&32000u32.to_le_bytes());
    b.extend_from_slice(&(32000u32 * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    for v in stereo {
        b.extend_from_slice(&((v * 32767.0).round() as i16).to_le_bytes());
    }
    std::fs::write(&wav, b)?;
    let mut child = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", "24", "-i", "-", "-i"])
        .arg(&wav)
        .args(["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest"])
        .arg(out)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("running ffmpeg")?;
    child.stdin.take().unwrap().write_all(rgb)?;
    if !child.wait()?.success() {
        bail!("ffmpeg failed");
    }
    std::fs::remove_file(wav)?;
    Ok(())
}

/// Minimal safetensors writer for debug dumps.
fn write_safetensors(path: &Path, tensors: &BTreeMap<String, (String, Vec<usize>, Vec<u8>)>) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (k, (dt, shape, data)) in tensors {
        header.insert(k.clone(), serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [offset, offset + data.len()]}));
        offset += data.len();
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + offset);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for (_, _, d) in tensors.values() {
        out.extend_from_slice(d);
    }
    std::fs::write(path, out)?;
    Ok(())
}

impl Args {
    /// The prompts to run: `--prompt-file`'s lines, else `--prompt`.
    fn prompts(&self) -> Result<Vec<String>> {
        Ok(match &self.prompt_file {
            Some(p) => std::fs::read_to_string(p)
                .with_context(|| format!("reading {}", p.display()))?
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
            None => vec![self.prompt.clone()],
        })
    }
}

fn out_path(base: &Path, i: usize, count: usize) -> PathBuf {
    if count == 1 {
        return base.to_path_buf();
    }
    let stem = base.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let ext = base.extension().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "png".into());
    base.with_file_name(format!("{stem}_{i}.{ext}"))
}

fn read_rgba(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    let mut decoder = png::Decoder::new(std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?);
    decoder.set_transformations(png::Transformations::normalize_to_color8() | png::Transformations::ALPHA);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let channels = info.color_type.samples();
    let rgba = buf[..info.buffer_size()]
        .chunks_exact(channels)
        .flat_map(|px| match channels {
            4 => [px[0], px[1], px[2], px[3]],
            3 => [px[0], px[1], px[2], 255],
            2 => [px[0], px[0], px[0], px[1]],
            _ => [px[0], px[0], px[0], 255],
        })
        .collect();
    Ok((rgba, info.width as usize, info.height as usize))
}

/// An RGBA image whose alpha is BiRefNet's foreground mask.
fn cut_out(model: &Path, rgba: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    let (rgb, _) = diffusion::pixal3d::split_rgba(rgba);
    let mask = diffusion::birefnet::remove_background(model, &rgb, w, h)?;
    Ok(rgba.chunks_exact(4).zip(&mask).flat_map(|(px, m)| [px[0], px[1], px[2], (m * 255.0).round() as u8]).collect())
}

fn write_rgba_png(path: &Path, rgba: &[u8], w: usize, h: usize) -> Result<()> {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}

fn run_remove_background(args: &Args, image: &Path) -> Result<()> {
    diffusion::cuda::init(0)?;
    let model = args.model(&args.birefnet, BIREFNET);
    if let Some(r) = &args.birefnet_check {
        return diffusion::birefnet::check(&model, r);
    }
    let (rgba, w, h) = read_rgba(image)?;
    let out = if args.out.extension().is_some_and(|e| e == "png") { args.out.clone() } else { args.out.with_extension("png") };
    write_rgba_png(&out, &cut_out(&model, &rgba, w, h)?, w, h)?;
    eprintln!("saved {}", out.display());
    Ok(())
}

fn run_pixal3d(args: &Args, image: &Path) -> Result<()> {
    diffusion::cuda::init(0)?;
    let paths = diffusion::pixal3d::Paths {
        dit: args.model(&args.dit, PIXAL_DIT),
        dino: args.model(&args.clip_vision, PIXAL_DINO),
        shape_vae: args.model(&args.vae, PIXAL_SHAPE_VAE),
        texture_vae: args.model(&args.texture_vae, PIXAL_TEX_VAE),
    };
    let (rgba, w, h) = read_rgba(image)?;
    if let Some(r) = &args.pixal3d_check {
        return diffusion::pixal3d::check(&paths, &rgba, w, h, r, args.threads, args.int8_attention()?);
    }
    let mut p = diffusion::pixal3d::Params::example(args.seed);
    p.fov = args.fov;
    p.resolution = args.resolution;
    p.texture_enabled = !args.no_texture;
    p.max_faces = args.max_faces;
    p.int8_attn = args.int8_attention()?;
    if let Some(seeds) = &args.pixal3d_seeds {
        let v = seeds.split(',').map(|s| s.trim().parse::<u64>()).collect::<Result<Vec<_>, _>>()?;
        let [a, b, c, d] = v[..] else { bail!("--pixal3d-seeds takes four seeds") };
        (p.structure.seed, p.shape.seed, p.refine.seed, p.texture.seed) = (a, b, c, d);
    }
    let mut last = String::new();
    let mut progress = |stage: &str, i: usize, n: usize| -> Result<()> {
        if stage != last {
            eprintln!("[pixal3d] {stage}");
            last = stage.to_string();
        }
        let _ = (i, n);
        Ok(())
    };
    let mut rgba = rgba;
    if rgba.chunks_exact(4).all(|px| px[3] == 255) {
        // nothing to mask with: cut the object out first, as the example workflow's RemoveBackground does
        let model = args.model(&args.birefnet, BIREFNET);
        if !model.exists() {
            bail!("{} has no transparency: give --pixal3d an RGBA PNG, or --birefnet the background removal model ({} not found)", image.display(), model.display());
        }
        eprintln!("[pixal3d] removing the background with {}", model.display());
        rgba = cut_out(&model, &rgba, w, h)?;
    }
    let (rgb, mask) = diffusion::pixal3d::split_rgba(&rgba);
    let mesh = diffusion::pixal3d::run(&paths, &rgb, &mask, w, h, &p, args.threads, &mut progress)?;
    let out = if args.out.extension().is_some_and(|e| e == "glb") { args.out.clone() } else { args.out.with_extension("glb") };
    std::fs::write(&out, diffusion::pixal3d::glb(&mesh))?;
    eprintln!("saved {}", out.display());
    Ok(())
}

fn run_triposplat(args: &Args, image: &Path) -> Result<()> {
    diffusion::cuda::init(0)?;
    let paths = diffusion::triposplat::Paths {
        dit: args.model(&args.dit, SPLAT_DIT),
        dino: args.model(&args.clip_vision, SPLAT_DINO),
        flux_vae: args.model(&args.vae, MOSS_VAE),
        decoder: args.model(&args.splat_decoder, SPLAT_DECODER),
    };
    let (mut rgba, w, h) = read_rgba(image)?;
    let mut p = diffusion::triposplat::Params::example(args.seed);
    (p.steps, p.cfg, p.num_gaussians) = (args.splat_steps, args.splat_cfg, args.gaussians);
    p.decode_seed = args.decode_seed.unwrap_or(args.seed);
    if let Some(r) = &args.triposplat_check {
        return diffusion::triposplat::check(&paths, &rgba, w, h, r, &p, args.threads);
    }
    let mut last = String::new();
    let mut progress = |stage: &str, _i: usize, _n: usize| -> Result<()> {
        if stage != last {
            eprintln!("[triposplat] {stage}");
            last = stage.to_string();
        }
        Ok(())
    };
    if rgba.chunks_exact(4).all(|px| px[3] == 255) {
        let model = args.model(&args.birefnet, BIREFNET);
        if !model.exists() {
            bail!("{} has no transparency: give --triposplat an RGBA PNG, or --birefnet the background removal model ({} not found)", image.display(), model.display());
        }
        eprintln!("[triposplat] removing the background with {}", model.display());
        rgba = cut_out(&model, &rgba, w, h)?;
    }
    let (rgb, mask) = diffusion::pixal3d::split_rgba(&rgba);
    let splat = diffusion::triposplat::run(&paths, &rgb, &mask, w, h, &p, args.threads, &mut progress)?;
    let out = if args.out.extension().is_some_and(|e| e == "ply") { args.out.clone() } else { args.out.with_extension("ply") };
    std::fs::write(&out, diffusion::triposplat::ply(&splat))?;
    eprintln!("saved {}", out.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(image) = &args.triposplat {
        return run_triposplat(&args, image);
    }
    if let Some(image) = &args.pixal3d {
        return run_pixal3d(&args, image);
    }
    if let Some(image) = &args.remove_background {
        return run_remove_background(&args, image);
    }
    if let Some(reference) = &args.qwen21_vae_check {
        return diffusion::vae21::check(&args.model(&args.vae, QWEN21_VAE), reference);
    }
    if let Some(reference) = &args.seedvr_check {
        return diffusion::seedvr::check(&args.model(&args.dit, SEEDVR_DIT), &args.model(&args.vae, SEEDVR_VAE), reference);
    }
    if let Some(reference) = &args.relayvsr_check {
        return diffusion::relayvsr::check(&args.model(&args.dit, RELAYVSR_DIT), &args.model(&args.vae, RELAYVSR_VAE), reference);
    }
    if let Some(reference) = &args.flux2_vae_check {
        return diffusion::flux2vae::check(&args.model(&args.vae, MOSS_VAE), reference);
    }
    if args.birefnet_check.is_some() {
        diffusion::cuda::init(0)?;
        return diffusion::birefnet::check(&args.model(&args.birefnet, BIREFNET), args.birefnet_check.as_ref().unwrap());
    }
    let int8_attn = args.int8_attention()?;
    let load = pipeline::LoadOptions {
        dit: args.model(&args.dit, KREA_DIT),
        te: args.model(&args.te, KREA_TE),
        vae: args.model(&args.vae, QWEN_VAE),
        max_width: args.width,
        max_height: args.height,
        max_pixels: args.width * args.height,
        reserve_vram_mb: args.reserve_vram.unwrap_or(512),
        // comfy_memory_mode is the default, so only --legacy-offload changes anything
        resident_blocks: args.resident_blocks,
        slots: args.slots,
        threads: args.threads,
        attn_variant: args.attn_variant,
        int8_attn,
        int8_weights: !args.no_quant,
        lossy: args.lossy,
        comfy_memory: (!args.legacy_offload).then(|| args.reserve_vram.unwrap_or(diffusion::comfy_memory::DEFAULT_HEADROOM_MB)),
    };
    if args.h3 || args.h3_image {
        return run_h3(&args, load);
    }
    if args.nemotron {
        return run_nemotron(&args, load);
    }
    if args.anima {
        return run_anima(&args, load);
    }
    if args.moss {
        return run_moss(&args, load);
    }
    if args.qwen21 {
        return run_qwen21(&args, load);
    }
    if args.sd || args.sdxl {
        return run_sd(&args, load);
    }
    run_krea(&args, load)
}

fn run_krea(args: &Args, load: pipeline::LoadOptions) -> Result<()> {
    let t_start = Instant::now();
    let mut gen = pipeline::Generator::load_krea(&load)?;
    apply_loras(&mut gen, args)?;
    let prompts = args.prompts()?;
    let jobs: Vec<(usize, &String, (usize, usize))> = prompts.iter().enumerate().flat_map(|(p, prompt)| args.batches().into_iter().map(move |g| (p, prompt, g))).collect();
    let n_jobs = prompts.len() * args.count;
    let n = 16 * (args.height / 8) * (args.width / 8);
    let timestep_zero = match args.ref_method.as_str() {
        "index" => false,
        "index_timestep_zero" => true,
        m => bail!("unknown --ref-method {m} (index | index_timestep_zero)"),
    };
    let images = args.image.iter().map(|p| read_png(p)).collect::<Result<Vec<_>>>()?;
    let references = args.ref_image.iter().map(|p| {
        let i = read_png(p)?;
        gen.encode_image(&i.rgb, i.width, i.height)
    }).collect::<Result<Vec<_>>>()?;
    let style = match &args.style_image {
        Some(p) => {
            if args.style_adain <= 0.0 {
                bail!("--style-image on Krea 2 needs --style-adain above 0 (the node's default is 0.6)");
            }
            let img = read_png(p)?;
            let (rgb, w, h) = diffusion::snapref::shrink(&img.rgb, img.width, img.height, args.style_megapixels);
            let mut s = pipeline::StyleRequest::node(gen.encode_image(&rgb, w, h)?, args.style_adain, false, args.steps);
            s.steps = diffusion::steer::parse_range(&args.style_steps)?;
            s.adain_blocks = args.style_adain_blocks.as_deref().map(diffusion::steer::parse_range).transpose()?.unwrap_or((0, usize::MAX));
            s.caption = Prompt::text(args.style_caption.as_deref().unwrap_or(pipeline::StyleRequest::KREA2_NODE_CAPTION));
            eprintln!("style reference {} at {w}x{h}, AdaIN {}, steps {:?}", p.display(), args.style_adain, s.steps);
            Some(s)
        }
        None => None,
    };
    for (p, prompt, group) in jobs {
        let mut positive = pipeline::KreaEntry::new(Prompt::text(prompt), images.clone());
        (positive.references, positive.timestep_zero) = (references.clone(), timestep_zero);
        let req = pipeline::KreaRequest {
            positive: vec![positive],
            negative: vec![pipeline::KreaEntry::new(Prompt::text(&args.negative), images.clone())],
            seed: args.seed + group.0 as u64,
            width: args.width,
            height: args.height,
            sampling: args.batch_sampling("euler", ModelSampling::Flux { shift: args.shift }, group, (args.width / 8, args.height / 8))?,
            noise: Some(args.noise(group, n)?),
            attn_token_weights: None,
            style: style.clone(),
        };
        let mut dump = pipeline::Dump::new();
        if let Some(r) = references.first() {
            dump.insert("ref0".into(), ("F32".into(), vec![16, r.height, r.width], st::f32_bytes(&r.latent).to_vec()));
        }
        let (rgb, _) = gen.generate_krea(&req, |_| true, args.dump.as_ref().map(|_| &mut dump))?;
        for (j, rgb) in rgb.chunks_exact(args.width * args.height * 3).enumerate() {
            let img_i = p * args.count + group.0 + j;
            let path = out_path(&args.out, img_i, n_jobs);
            std::fs::write(&path, pipeline::encode_png(args.width, args.height, rgb)?)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("[{img_i}] saved {} (seed {})", path.display(), args.seed + (group.0 + j) as u64);
        }
        if let Some(d) = &args.dump {
            write_safetensors(d, &dump)?;
            eprintln!("dump written to {}", d.display());
        }
    }
    eprintln!("total {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}
