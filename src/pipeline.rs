//! Loaded models + one text-to-image generation, shared by the CLI and the web server.

use crate::cuda::{self, DevBuf, Event, Stream};
use crate::{dit, ffi, offload, sampling, st, te, vae};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MB: usize = 1 << 20;

pub struct LoadOptions {
    pub dit: PathBuf,
    pub te: PathBuf,
    pub vae: PathBuf,
    /// Size the initial residency plan is made for (it is re-planned per request).
    pub max_width: usize,
    pub max_height: usize,
    /// Largest request accepted (width * height).
    pub max_pixels: usize,
    pub reserve_vram_mb: usize,
    pub resident_blocks: Option<usize>,
    pub slots: usize,
    pub threads: usize,
    pub attn_variant: i32,
    pub int8_attn: bool,
}

pub struct Request {
    pub prompt: String,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub shift: f64,
    /// Use this noise instead of the seed (parity tests).
    pub noise: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub text_ms: f64,
    pub sample_ms: f64,
    pub vae_ms: f64,
    pub total_ms: f64,
    pub streamed_gb: f64,
}

/// Debug tensors: name -> (safetensors dtype, shape, bytes)
pub type Dump = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;

/// Returned (through `anyhow`) when the progress callback asks to stop.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

struct DitModel {
    path: PathBuf,
    dit: dit::Dit,
    st: st::SafeTensors,
}

pub struct Generator {
    st: Stream,
    te: Option<(PathBuf, te::TextEncoder)>,
    dit: Option<DitModel>,
    vae: Option<(PathBuf, vae::Vae)>,
    cached: Option<(String, te::Cond, DevBuf)>,
    pub max_pixels: usize,
    max_width: usize,
    max_height: usize,
    reserve: usize,
    resident_blocks: Option<usize>,
    slots: usize,
    threads: usize,
    attn_variant: i32,
    int8_attn: bool,
    /// Re-plan DiT block residency from live free VRAM (off when --resident-blocks is forced).
    dynamic: bool,
}

/// Re-plan which DiT blocks live in VRAM from the memory free right now (other applications
/// included), keeping `need` bytes for activations plus `reserve`. Blocks are promoted from or
/// demoted to pinned RAM; nothing is reloaded from disk. The stream is synchronized first.
fn rebalance(st: &Stream, dit: &mut dit::Dit, dst: &st::SafeTensors, reserve: usize, slots: usize, need: usize,
    demote_only: bool) -> Result<()> {
    st.sync()?;
    cuda::trim_pool()?;
    let free = cuda::device_free()?;
    let blocks = &dit.blocks;
    let owned = blocks.vram_bytes();
    let avail = (free + owned).saturating_sub(need + reserve);
    let mut p = offload::plan(&blocks.layouts, avail, slots);
    if std::env::var_os("KREA_DEBUG_VRAM").is_some() {
        let (cf, _) = cuda::mem_info()?;
        eprintln!("rebalance: free {} MB (cuda {} MB), owned {} MB, need {} MB, avail {} MB -> {} resident",
            free / MB, cf / MB, owned / MB, need / MB, avail / MB, p.resident);
    }
    let n = blocks.len();
    let block = blocks.layouts.iter().map(|l| l.size).max().unwrap_or(0);
    if p.resident < n && avail < block * p.slots {
        // Not even full streaming fits (other applications hold the memory). Keep going with the
        // smallest footprint like ComfyUI does; the driver pages to system memory and it gets slower.
        p = offload::Plan { resident: 0, slots: 1 };
        eprintln!("warning: VRAM oversubscribed (free {} MB, activations {} MB, reserve {} MB); streaming every                    block through one slot, expect slowdowns", free / MB, need / MB, reserve / MB);
    }
    if demote_only && p.resident >= blocks.resident() {
        return Ok(());
    }
    let want_slots = if p.resident < n { p.slots } else { 0 };
    if p.resident != blocks.resident() || want_slots != blocks.slot_count() {
        dit.blocks.set_resident(p.resident, p.slots, dst)?;
    }
    Ok(())
}

const LT_MAX: usize = 512; // text tokens budgeted for activation planning

/// Peak activation memory of one request (denoising step or VAE decode, whichever is larger).
fn activation_need(width: usize, height: usize) -> usize {
    let (lh, lw) = (height / 8, width / 8);
    dit::step_activation_bytes(LT_MAX + (lh / 2) * (lw / 2)).max(vae::Vae::activation_bytes(lh, lw))
}

impl Generator {
    /// Start the GPU runtime with no models loaded; the `ensure_*` methods load them.
    pub fn new(o: &LoadOptions) -> Result<Self> {
        cuda::init(0)?;
        ffi::ok(unsafe { ffi::krea_blas_init(64 * MB) }, "cublasLt init")?;
        let st0 = Stream::new()?;
        let (_, total) = cuda::mem_info()?;
        eprintln!("GPU memory: {} MB free (device-wide) / {} MB total", cuda::device_free()? / MB, total / MB);
        Ok(Generator { st: st0, te: None, dit: None, vae: None, cached: None, max_pixels: o.max_pixels,
            max_width: o.max_width, max_height: o.max_height, reserve: o.reserve_vram_mb * MB,
            resident_blocks: o.resident_blocks, slots: o.slots, threads: o.threads, attn_variant: o.attn_variant,
            int8_attn: o.int8_attn, dynamic: o.resident_blocks.is_none() })
    }

    pub fn load(o: &LoadOptions) -> Result<Self> {
        let t_start = Instant::now();
        let mut gen = Self::new(o)?;
        gen.ensure_vae(&o.vae)?;
        gen.ensure_te(&o.te)?;
        gen.ensure_dit(&o.dit)?;
        eprintln!("startup total {:.1}s", t_start.elapsed().as_secs_f64());
        Ok(gen)
    }

    /// Load the VAE unless `path` is already loaded (replaces any other VAE).
    pub fn ensure_vae(&mut self, path: &Path) -> Result<()> {
        if self.vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.vae = None;
        let t0 = Instant::now();
        self.vae = Some((path.to_path_buf(), vae::Vae::load(path, &self.st)?));
        eprintln!("VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the text encoder unless `path` is already loaded (streamed from pinned RAM).
    pub fn ensure_te(&mut self, path: &Path) -> Result<()> {
        if self.te.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        self.te = None;
        self.cached = None;
        let t0 = Instant::now();
        self.te = Some((path.to_path_buf(), te::TextEncoder::load(path, 0, self.threads, &self.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the DiT unless `path` is already loaded, planning block residency from what is free.
    pub fn ensure_dit(&mut self, path: &Path) -> Result<()> {
        if self.dit.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        self.dit = None;
        self.cached = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = dit::globals_layout(&dst)?.size;
        let txt_slots = 2 * dit::txt_layouts(&dst)?.iter().map(|l| l.size).max().unwrap_or(0);
        let te_slots = 2 * 110 * MB;
        let act = activation_need(self.max_width, self.max_height) + txt_slots + te_slots;
        let free = cuda::device_free()?;
        let mut budget = free.saturating_sub(self.reserve + globals + act);
        if let Some(r) = self.resident_blocks {
            let one = dit::block_layout(&dst, 0)?.size;
            budget = if r >= dit::n_blocks(&dst) { r * one } else { (r + self.slots) * one };
        }
        let mut dit = dit::Dit::load(&dst, budget, self.slots, self.threads, &self.st)?;
        dit.attn_variant = self.attn_variant;
        dit.int8_attn = self.int8_attn;
        eprintln!("DiT ready in {:.1}s (activation reserve {} MB)", t0.elapsed().as_secs_f64(), act / MB);
        self.dit = Some(DitModel { path: path.to_path_buf(), dit, st: dst });
        Ok(())
    }

    /// Choose the DiT attention: INT8 (fastest) or bf16 FlashAttention. Takes effect on the next step.
    pub fn set_int8_attention(&mut self, on: bool) {
        self.int8_attn = on;
        if let Some(m) = &mut self.dit {
            m.dit.int8_attn = on;
        }
    }

    /// Drop every model and give their VRAM and pinned RAM back. `ensure_*` loads them again.
    pub fn unload(&mut self) -> Result<()> {
        self.st.sync()?;
        self.cached = None;
        self.dit = None;
        self.te = None;
        self.vae = None;
        cuda::trim_pool()
    }

    /// Give back cached VRAM without unloading anything.
    pub fn trim(&mut self) -> Result<()> {
        self.st.sync()?;
        cuda::trim_pool()
    }

    /// Generate one image; returns RGB8 pixels. `progress(step, steps)` is called after each step
    /// and stops the run with [`Cancelled`] by returning false.
    pub fn generate(&mut self, r: &Request, progress: impl FnMut(usize, usize) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<u8>, Stats)> {
        let t_img = Instant::now();
        let (latent, mut stats) = self.denoise(r, progress, dump.as_deref_mut())?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let t0 = Instant::now();
        let rgb = self.decode(&latent, lh, lw)?;
        stats.vae_ms = t0.elapsed().as_secs_f64() * 1e3;
        stats.total_ms = t_img.elapsed().as_secs_f64() * 1e3;
        eprintln!("VAE decode {:.0} ms, image done in {:.2}s", stats.vae_ms, stats.total_ms / 1e3);
        if let Some(d) = dump {
            d.insert("image_rgb8".into(), ("U8".into(), vec![lh * 8, lw * 8, 3], rgb.clone()));
        }
        Ok((rgb, stats))
    }

    /// Text encoding and denoising only; returns the final latent ([16, H/8, W/8] f32) on the host.
    pub fn sample_latent(&mut self, r: &Request, progress: impl FnMut(usize, usize) -> bool) -> Result<(Vec<f32>, Stats)> {
        let (latent, stats) = self.denoise(r, progress, None)?;
        let mut b = vec![0u8; latent.size()];
        latent.download(0, &mut b, &self.st)?;
        self.st.sync()?;
        Ok((b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(), stats))
    }

    /// Decode a host latent from [`Generator::sample_latent`] to RGB8 pixels.
    pub fn decode_latent(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / 8, width / 8);
        if latent.len() != dit::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), dit::CH * lh * lw);
        }
        let d = DevBuf::alloc_async(latent.len() * 4, &self.st)?;
        d.upload(0, &st::f32_bytes(latent), &self.st)?;
        self.decode(&d, lh, lw)
    }

    fn denoise(&mut self, r: &Request, mut progress: impl FnMut(usize, usize) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(DevBuf, Stats)> {
        if r.width % 16 != 0 || r.height % 16 != 0 || r.width == 0 || r.height == 0 {
            bail!("width and height must be positive multiples of 16");
        }
        if r.width * r.height > self.max_pixels {
            bail!("{}x{} exceeds the {} pixel limit (--max-side)", r.width, r.height, self.max_pixels);
        }
        if r.steps == 0 || r.steps > 100 {
            bail!("steps must be in 1..=100");
        }
        let m = self.dit.as_mut().context("no diffusion model loaded")?;
        let te = &mut self.te.as_mut().context("no text encoder loaded")?.1;
        let need = activation_need(r.width, r.height);
        if self.dynamic {
            rebalance(&self.st, &mut m.dit, &m.st, self.reserve, self.slots, need, false)?;
        }
        let st = &self.st;
        let mut stats = Stats::default();
        let (lh, lw) = (r.height / 8, r.width / 8);

        // text encoding (reused while the prompt is unchanged)
        let t0 = Instant::now();
        if self.cached.as_ref().map_or(true, |(p, _, _)| *p != r.prompt) {
            drop(self.cached.take());
            let cond = te.encode(&r.prompt, st)?;
            let ctx = m.dit.encode_context(&cond.ctx, cond.len, st)?;
            st.sync()?;
            eprintln!("text: {} tokens ({} conditioning) in {:.0} ms", cond.token_ids.len(), cond.len,
                t0.elapsed().as_secs_f64() * 1e3);
            self.cached = Some((r.prompt.clone(), cond, ctx));
        }
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (_, cond, ctx) = self.cached.as_ref().unwrap();
        if let Some(d) = dump.as_deref_mut() {
            let mut b = vec![0u8; cond.ctx.size()];
            cond.ctx.download(0, &mut b, st)?;
            d.insert("cond".into(), ("BF16".into(), vec![cond.len, dit::TXT_LAYERS * dit::TXT_D], b));
            let mut b = vec![0u8; ctx.size()];
            ctx.download(0, &mut b, st)?;
            d.insert("txtmlp_out".into(), ("BF16".into(), vec![cond.len, dit::F], b));
            let ids: Vec<u8> = cond.token_ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect();
            d.insert("token_ids".into(), ("I64".into(), vec![cond.token_ids.len()], ids));
        }

        // noise (same as ComfyUI's KSampler for the seed)
        let n = dit::CH * lh * lw;
        let noise = match &r.noise {
            Some(v) => v.clone(),
            None => sampling::torch_randn(r.seed, n),
        };
        if noise.len() != n {
            bail!("noise has {} values, expected {n}", noise.len());
        }
        let latent = DevBuf::alloc_async(n * 4, st)?;
        latent.upload(0, &st::f32_bytes(&noise), st)?;
        let lt = cond.len;
        let rope = dit::Dit::rope_table(lt, lh / 2, lw / 2);
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &st::f32_bytes(&rope), st)?;

        // denoise
        let sigmas = sampling::simple_sigmas(r.shift, r.steps);
        let (e0, e1) = (Event::timing()?, Event::timing()?);
        let copied0 = m.dit.blocks.copied_bytes;
        for i in 0..r.steps {
            e0.record(st)?;
            let mut probe_store: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if i == 0 && (name == "block0" || name == "block27" || name == "dit_out_tokens") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probe_store.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> =
                    if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.step(&latent, lh, lw, ctx, lt, &rope_d, sigmas[i], sigmas[i + 1], st, None, probe_ref)?;
            }
            e1.record(st)?;
            e1.sync()?;
            let ms = e1.elapsed_ms(&e0)? as f64;
            stats.sample_ms += ms;
            eprintln!("step {}/{} sigma {:.4}: {:.0} ms", i + 1, r.steps, sigmas[i], ms);
            if let Some(d) = dump.as_deref_mut() {
                for (name, v) in probe_store {
                    let shape = if name == "dit_out_tokens" { vec![(lh / 2) * (lw / 2), 64] } else { vec![v.len() / 2 / dit::F, dit::F] };
                    d.insert(name, ("BF16".into(), shape, v));
                }
            }
            // another application took VRAM mid-run: demote blocks before the driver starts
            // paging to shared system memory (which would be far slower than streaming)
            if self.dynamic && i + 1 < r.steps && cuda::device_free()? < self.reserve / 2 {
                eprintln!("VRAM pressure: free {} MB < {} MB", cuda::device_free()? / MB, self.reserve / 2 / MB);
                // NVML clamps free at 0 when oversubscribed, hiding the real shortfall: ask for an
                // extra reserve each time so blocks keep moving out until pressure clears
                let step_need = dit::step_activation_bytes(LT_MAX + (lh / 2) * (lw / 2));
                rebalance(st, &mut m.dit, &m.st, self.reserve, self.slots, step_need + self.reserve, true)?;
            }
            if !progress(i + 1, r.steps) {
                st.sync()?;
                return Err(Cancelled.into());
            }
        }
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("sampling {:.2}s ({:.0} ms/step avg, {:.2} GB streamed over PCIe)", stats.sample_ms / 1e3,
            stats.sample_ms / r.steps as f64, stats.streamed_gb);
        drop(rope_d);
        if let Some(d) = dump {
            let mut b = vec![0u8; n * 4];
            latent.download(0, &mut b, st)?;
            d.insert("latent_final".into(), ("F32".into(), vec![dit::CH, lh, lw], b));
        }
        Ok((latent, stats))
    }

    fn decode(&mut self, latent: &DevBuf, lh: usize, lw: usize) -> Result<Vec<u8>> {
        let st = &self.st;
        // the decoder's activations may need memory freed since the request started
        if let Some(m) = self.dit.as_mut().filter(|_| self.dynamic) {
            if cuda::device_free()? < vae::Vae::activation_bytes(lh, lw) + self.reserve {
                rebalance(st, &mut m.dit, &m.st, self.reserve, self.slots, vae::Vae::activation_bytes(lh, lw), true)?;
            }
        }
        self.vae.as_ref().context("no VAE loaded")?.1.decode(latent, lh, lw, st)
    }
}

pub fn encode_png(w: usize, h: usize, rgb: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(rgb)?;
    }
    Ok(out)
}
