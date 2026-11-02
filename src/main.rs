//! krea2: Rust + CUDA reimplementation of ComfyUI's Krea 2 (W8A8 int8) text-to-image pipeline.

use anyhow::{bail, Context, Result};
use clap::Parser;
use krea2::samplers::ModelSampling;
use krea2::{pipeline, serve, st};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;


#[derive(Parser, Debug)]
#[command(version, about = "Krea 2 (int8) text-to-image, ComfyUI-compatible")]
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
    /// Output path; with several images, "_<i>" is inserted before the extension
    #[arg(short, long, default_value = "krea2.png")]
    out: PathBuf,
    /// Folder holding Stable-diffusion/, clip/ and VAE/ (default: $KREA2_MODELS, else ./models)
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
    /// VRAM (MB) to leave untouched for the desktop / other apps
    #[arg(long, default_value_t = 512)]
    reserve_vram: usize,
    /// Force the number of DiT blocks kept resident in VRAM (default: as many as fit)
    #[arg(long)]
    resident_blocks: Option<usize>,
    /// Streaming ring slots for offloaded blocks
    #[arg(long, default_value_t = 2)]
    slots: usize,
    /// Loader threads
    #[arg(long, default_value_t = 6)]
    threads: usize,
    /// FlashAttention tile variant (0: 128x64, 1: 128x32)
    #[arg(long, default_value_t = 0)]
    attn_variant: i32,
    /// Attention for the DiT blocks: "int8" (SageAttention-style INT8 attention from comfy-kitchen,
    /// fastest) or "flash" (bf16 FlashAttention-2, matches ComfyUI's default numerics most closely)
    #[arg(long, default_value = "int8")]
    attention: String,
    /// Use the noise stored in a ComfyUI reference dump (tools/ref_dump.py) instead of the seed
    #[arg(long)]
    ref_noise: Option<PathBuf>,
    /// Run the web UI on this address (e.g. 127.0.0.1:7860) instead of generating once.
    /// --width/--height then set the default and the largest size the VRAM plan allows.
    #[arg(long)]
    serve: Option<String>,
    /// Web UI: largest side length (pixels, at 1:1) a request may use. VRAM residency is re-planned
    /// for each request, so this does not reserve memory up front.
    #[arg(long, default_value_t = 2048)]
    max_side: usize,
    /// Where the web UI saves generated images
    #[arg(long, default_value = "outputs")]
    out_dir: PathBuf,
    /// Write intermediate tensors (safetensors) for parity checks
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Run Anima instead of Krea 2 (--dit and --te then default to the Anima files, --shift to 3)
    #[arg(long)]
    anima: bool,
    /// Run MOSS Image instead of Krea 2 (--dit, --te and --vae then default to the MOSS files, --shift to 5)
    #[arg(long)]
    moss: bool,
    /// MOSS Image: encode with the Krea 2 template, as SwarmUI does
    #[arg(long)]
    swarm_template: bool,
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
    /// LoRA file, `path` or `path:strength`, in order (repeatable)
    #[arg(long)]
    lora: Vec<String>,
    /// Krea 2: PNG for the text encoder, at its own size; SwarmUI scales these to about 384x384 (repeatable)
    #[arg(long)]
    image: Vec<PathBuf>,
    /// Krea 2: PNG for a reference latent, for edit LoRAs (repeatable)
    #[arg(long)]
    ref_image: Vec<PathBuf>,
    /// Krea 2: how reference latents are used, "index" or "index_timestep_zero"
    #[arg(long, default_value = "index")]
    ref_method: String,
}

/// RGB f32 [height, width, 3], 0 to 1.
fn read_png(path: &Path) -> Result<krea2::vision::PromptImage> {
    let mut decoder = png::Decoder::new(std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?);
    decoder.set_transformations(png::Transformations::normalize_to_color8() | png::Transformations::ALPHA);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let channels = info.color_type.samples();
    let rgb = buf[..info.buffer_size()].chunks_exact(channels).flat_map(|px| {
        let (c, a) = (if channels >= 3 { [px[0], px[1], px[2]] } else { [px[0]; 3] }, px[channels - 1] as f32 / 255.0);
        let alpha = if channels % 2 == 0 { a } else { 1.0 };
        c.map(|v| v as f32 / 255.0 * alpha + 1.0 - alpha)
    }).collect();
    Ok(krea2::vision::PromptImage { width: info.width as usize, height: info.height as usize, rgb })
}

/// `path` or `path:strength`, as in `--lora`.
fn parse_loras(args: &Args) -> Result<Vec<krea2::lora::Lora>> {
    args.lora.iter().map(|l| {
        let (path, strength) = match l.rsplit_once(':').map(|(p, v)| (p, v.parse::<f32>())) {
            Some((p, Ok(v))) => (p, v),
            _ => (l.as_str(), 1.0),
        };
        if !std::path::Path::new(path).exists() {
            bail!("no LoRA at {path}");
        }
        Ok(krea2::lora::Lora { path: PathBuf::from(path), strength })
    }).collect()
}

const KREA_DIT: &str = "Stable-diffusion/kroma-v0.3.1-turbo-opd-W8A8.safetensors";
const KREA_TE: &str = "clip/qwen3vl_4b.safetensors";
const QWEN_VAE: &str = "VAE/qwen_image_vae.safetensors";
const ANIMA_DIT: &str = "Stable-diffusion/animaInt8Mxfp8_turboV11Int8.safetensors";
const ANIMA_TE: &str = "clip/qwen_3_600m.safetensors";
const MOSS_DIT: &str = "Stable-diffusion/moss_image_sft_int8_convrot_attn.safetensors";
const MOSS_TE: &str = "clip/qwen3.5_9b_int8_convrot.safetensors";
const MOSS_VAE: &str = "VAE/flux2-vae.safetensors";

impl Args {
    fn sampling(&self, default: &str, ms: ModelSampling, seed: u64, latent: (usize, usize)) -> Result<pipeline::Sampling> {
        pipeline::Sampling::new(self.sampler.as_deref().unwrap_or(default), &self.scheduler, ms, self.steps, self.cfg, seed, latent)
    }

    /// `path`, or `default` inside the models folder.
    fn model(&self, path: &Option<PathBuf>, default: &str) -> PathBuf {
        path.clone().unwrap_or_else(|| {
            let root = self.models.clone().or_else(|| std::env::var_os("KREA2_MODELS").map(PathBuf::from));
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
    gen.set_loras(&parse_loras(args)?)?;
    let n = 16 * (args.height / 8) * (args.width / 8);
    for i in 0..args.count {
        let seed = args.seed + i as u64;
        let noise = match &args.ref_noise {
            Some(p) => st::SafeTensors::open(p)?.f32_vec("noise", false)?,
            None => krea2::sampling::torch_randn(seed, n),
        };
        let r = pipeline::AnimaRequest {
            positive: krea2::tokenizer::token_weights(&args.prompt, 1.0),
            negative: krea2::tokenizer::token_weights(&args.negative, 1.0),
            width: args.width,
            height: args.height,
            sampling: args.sampling("er_sde", ModelSampling::Flow { shift }, seed, (args.width / 8, args.height / 8))?,
            noise,
        };
        let mut dump = pipeline::Dump::new();
        let t0 = Instant::now();
        let (latent, _) = gen.sample_anima(&r, |_, _| true, args.dump.as_ref().map(|_| &mut dump))?;
        let rgb = gen.decode_latent(&latent, args.width, args.height)?;
        eprintln!("image done in {:.2}s", t0.elapsed().as_secs_f64());
        let path = out_path(&args.out, i, args.count);
        std::fs::write(&path, pipeline::encode_png(args.width, args.height, &rgb)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("[{i}] saved {} (seed {seed})", path.display());
        if let Some(d) = &args.dump {
            write_safetensors(d, &dump)?;
        }
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
    gen.set_loras(&parse_loras(args)?)?;
    let n = 128 * (args.height / 16) * (args.width / 16);
    for i in 0..args.count {
        let seed = args.seed + i as u64;
        let noise = match &args.ref_noise {
            Some(p) => st::SafeTensors::open(p)?.f32_vec("noise", false)?,
            None => krea2::sampling::torch_randn(seed, n),
        };
        let r = pipeline::MossRequest {
            positive: args.prompt.clone(),
            negative: args.negative.clone(),
            template: args.swarm_template.then(|| krea2::te::TEMPLATE.to_string()),
            width: args.width,
            height: args.height,
            sampling: args.sampling("euler", ModelSampling::Flow { shift }, seed, (args.width / 16, args.height / 16))?,
            noise,
        };
        let mut dump = pipeline::Dump::new();
        let t0 = Instant::now();
        let (latent, _) = gen.sample_moss(&r, |_, _| true, args.dump.as_ref().map(|_| &mut dump))?;
        let t1 = Instant::now();
        let rgb = gen.decode_moss(&latent, args.width, args.height)?;
        eprintln!("VAE decode {:.0} ms, image done in {:.2}s", t1.elapsed().as_secs_f64() * 1e3, t0.elapsed().as_secs_f64());
        let path = out_path(&args.out, i, args.count);
        std::fs::write(&path, pipeline::encode_png(args.width, args.height, &rgb)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("[{i}] saved {} (seed {seed})", path.display());
        if let Some(d) = &args.dump {
            write_safetensors(d, &dump)?;
        }
    }
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

fn out_path(base: &Path, i: usize, count: usize) -> PathBuf {
    if count == 1 {
        return base.to_path_buf();
    }
    let stem = base.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let ext = base.extension().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "png".into());
    base.with_file_name(format!("{stem}_{i}.{ext}"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let int8_attn = match args.attention.as_str() {
        "flash" => false,
        "int8" | "sage" => true,
        a => bail!("unknown --attention {a} (flash | int8)"),
    };
    let load = pipeline::LoadOptions {
        dit: args.model(&args.dit, KREA_DIT),
        te: args.model(&args.te, KREA_TE),
        vae: args.model(&args.vae, QWEN_VAE),
        max_width: args.width,
        max_height: args.height,
        // the web UI accepts any size up to --max-side squared; residency is re-planned per request
        max_pixels: if args.serve.is_some() {
            (args.max_side * args.max_side).max(args.width * args.height)
        } else {
            args.width * args.height
        },
        reserve_vram_mb: args.reserve_vram,
        resident_blocks: args.resident_blocks,
        slots: args.slots,
        threads: args.threads,
        attn_variant: args.attn_variant,
        int8_attn,
    };
    if args.anima {
        return run_anima(&args, load);
    }
    if args.moss {
        return run_moss(&args, load);
    }
    if let Some(addr) = &args.serve {
        return serve::run(load, serve::ServeOptions {
            addr: addr.clone(),
            out_dir: args.out_dir.clone(),
            default_steps: args.steps,
            default_width: args.width,
            default_height: args.height,
            shift: args.shift,
        });
    }

    let t_start = Instant::now();
    let mut gen = pipeline::Generator::load(&load)?;
    gen.set_loras(&parse_loras(&args)?)?;
    let prompts: Vec<String> = match &args.prompt_file {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("reading {}", p.display()))?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        None => vec![args.prompt.clone()],
    };
    let jobs: Vec<(String, u64)> = prompts
        .iter()
        .flat_map(|p| (0..args.count).map(move |i| (p.clone(), i as u64)))
        .collect();
    let n_jobs = jobs.len();
    let ref_noise = match &args.ref_noise {
        Some(p) => Some(st::SafeTensors::open(p)?.f32_vec("noise", false)?),
        None => None,
    };
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
    for (img_i, (prompt, seed_off)) in jobs.into_iter().enumerate() {
        let req = pipeline::Request {
            prompt,
            negative: args.negative.clone(),
            seed: args.seed + seed_off,
            width: args.width,
            height: args.height,
            sampling: args.sampling("euler", ModelSampling::Flux { shift: args.shift }, args.seed + seed_off, (args.width / 8, args.height / 8))?,
            noise: ref_noise.clone(),
            images: images.clone(),
            negative_images: images.clone(),
            references: references.clone(),
            timestep_zero,
        };
        let mut dump = pipeline::Dump::new();
        if let Some(r) = references.first() {
            dump.insert("ref0".into(), ("F32".into(), vec![16, r.height, r.width], st::f32_bytes(&r.latent).to_vec()));
        }
        let (rgb, _) = gen.generate(&req, |_, _| true, args.dump.as_ref().map(|_| &mut dump))?;
        let path = out_path(&args.out, img_i, n_jobs);
        std::fs::write(&path, pipeline::encode_png(args.width, args.height, &rgb)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("[{img_i}] saved {} (seed {})", path.display(), req.seed);
        if let Some(d) = &args.dump {
            write_safetensors(d, &dump)?;
            eprintln!("dump written to {}", d.display());
        }
    }
    eprintln!("total {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}
