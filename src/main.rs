//! krea2: Rust + CUDA reimplementation of ComfyUI's Krea 2 (W8A8 int8) text-to-image pipeline.

use anyhow::{bail, Context, Result};
use clap::Parser;
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
    #[arg(long, default_value = r"G:\stable-diffusion-webui-forge\models\Stable-diffusion\kroma-v0.3.1-turbo-opd-W8A8.safetensors")]
    dit: PathBuf,
    #[arg(long, default_value = r"G:\stable-diffusion-webui-forge\models\clip\qwen3vl_4b.safetensors")]
    te: PathBuf,
    #[arg(long, default_value = r"G:\stable-diffusion-webui-forge\models\VAE\qwen_image_vae.safetensors")]
    vae: PathBuf,
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
        dit: args.dit.clone(),
        te: args.te.clone(),
        vae: args.vae.clone(),
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
    for (img_i, (prompt, seed_off)) in jobs.into_iter().enumerate() {
        let req = pipeline::Request {
            prompt,
            seed: args.seed + seed_off,
            width: args.width,
            height: args.height,
            steps: args.steps,
            shift: args.shift,
            noise: ref_noise.clone(),
        };
        let mut dump = pipeline::Dump::new();
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
