//! MOSS Image: its diffusion model, its Qwen3.5 text encoder, the Flux 2 VAE and the prompts encoded through them.

use super::{crop, nothing_for, remember, run_batches, scale_rows, touch, upload};
use super::{Dump, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Stats, Step, TokenKey, CONTEXTS};
use crate::cuda::{self, DevBuf, Stream};
use crate::{ffi, flux2vae, lora, moss, offload, qwen35, st, tokenizer};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// MOSS Image latents are at /16 and the VAE's latent has no text budget; 300 tokens is a long prompt.
const MOSS_TEXT: usize = 300;

pub struct MossRequest {
    pub positive: Prompt,
    /// Only encoded when the sampling uses it.
    pub negative: Prompt,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise, one after another.
    pub noise: Vec<f32>,
}

struct Model {
    path: PathBuf,
    dit: moss::Dit,
    st: st::SafeTensors,
    lora: lora::State,
}

#[derive(Default)]
pub(super) struct Moss {
    model: Option<Model>,
    te: Option<(PathBuf, qwen35::TextEncoder)>,
    tokenizer: Option<tokenizer::Tokenizer>,
    vae: Option<(PathBuf, flux2vae::Vae)>,
    /// Latest encoded tokens, most recent last.
    contexts: Vec<(TokenKey, moss::Context)>,
}

impl FamilyState for Moss {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        if lora::apply(lora::Arch::Moss, &m.st, "", m.dit.stores(), &mut m.lora, loras, st)? {
            m.dit.clear_mods();
            self.clear_prompts();
        }
        Ok(())
    }

    fn set_text_encoder_loras(&mut self, _: Option<&mut SharedTe>, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some((_, te)) = &mut self.te else { return nothing_for(loras, "text encoder") };
        if te.apply_loras(loras, st)? {
            self.clear_prompts();
        }
        Ok(())
    }

    fn clear_prompts(&mut self) {
        self.contexts.clear();
    }

    fn set_int8_attention(&mut self, on: bool) {
        if let Some(m) = &mut self.model {
            m.dit.int8_attn = on;
        }
    }

    fn blocks(&mut self) -> Option<(&mut offload::WeightSet, &st::SafeTensors)> {
        self.model.as_mut().map(|m| (&mut m.dit.blocks, &m.st))
    }

    fn idle(&mut self) -> Result<()> {
        if let Some(m) = &mut self.model {
            m.dit.blocks.set_resident(0, 1, &m.st)?;
        }
        Ok(())
    }
}

/// One image's activations in a denoising step.
fn step_need(width: usize, height: usize) -> usize {
    moss::step_activation_bytes((height / 16) * (width / 16) + MOSS_TEXT)
}

/// Peak activation memory of `images` sampled together (denoising step or VAE decode, whichever is larger).
fn moss_need(width: usize, height: usize, images: usize) -> usize {
    (images * step_need(width, height)).max(flux2vae::Vae::activation_bytes(height / 16, width / 16))
}

/// Token ids and per-token factors of a MOSS Image chunk: SwarmTextEncodeAdvanced tokenizes each
/// segment apart and puts them inside the template.
pub fn moss_tokens(t: &tokenizer::Tokenizer, segments: &[(String, f32)]) -> Result<(Vec<u32>, Vec<f32>)> {
    let (head, tail) = qwen35::TEMPLATE.split_once("{}").context("template")?;
    let mut ids = t.encode(head)?;
    let mut factors = vec![1.0; ids.len()];
    for (text, weight) in segments {
        ids.extend(t.encode(text)?);
        factors.resize(ids.len(), *weight);
    }
    ids.extend(t.encode(tail)?);
    factors.resize(ids.len(), 1.0);
    Ok((ids, factors))
}

impl Moss {
    /// The tokens of a prompt, one pair per chunk.
    fn key(&self, prompt: &Prompt) -> Result<TokenKey> {
        let t = self.tokenizer.as_ref().context("no text encoder loaded")?;
        prompt.chunks.iter().map(|segments| {
            let (ids, mut factors) = moss_tokens(t, segments)?;
            factors.iter_mut().for_each(|f| *f *= prompt.scale);
            Ok((ids, factors))
        }).collect()
    }

    /// Move the context of `key` to the end of the cache, encoding it if needed. `dump` sees the first chunk.
    fn context(&mut self, key: &TokenKey, st: &Stream, mut dump: Option<&mut Dump>) -> Result<()> {
        if touch(&mut self.contexts, |(p, _)| p == key) {
            return Ok(());
        }
        let m = self.model.as_mut().context("no diffusion model loaded")?;
        let te = &mut self.te.as_mut().context("no text encoder loaded")?.1;
        let t0 = Instant::now();
        let total: usize = key.iter().map(|(ids, _)| ids.len()).sum();
        let hidden = DevBuf::alloc_async(total * qwen35::D * 4, st)?;
        let mut at = 0;
        for (chunk, (ids, factors)) in key.iter().enumerate() {
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                let mut v = vec![0u8; bytes];
                b.download(0, &mut v, st).unwrap();
                probes.push((name.to_string(), v));
            };
            let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if chunk == 0 && dump.is_some() { Some(&mut probe) } else { None };
            let h = te.encode(ids, st, probe_ref)?;
            scale_rows(&h, qwen35::D, factors, st)?;
            cuda::memcpy_d2d(ffi::off(hidden.ptr(), at * qwen35::D * 4), h.ptr(), ids.len() * qwen35::D * 4, st)?;
            at += ids.len();
            st.sync()?;
            if let (0, Some(d)) = (chunk, dump.as_deref_mut()) {
                d.insert("ids".into(), ("I64".into(), vec![ids.len()], ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect()));
                for (name, v) in probes {
                    d.insert(name, ("F32".into(), vec![ids.len(), qwen35::D], v));
                }
            }
        }
        let ctx = m.dit.encode_context(&hidden, total, st)?;
        st.sync()?;
        eprintln!("text: {total} tokens in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        if let Some(d) = dump {
            let mut b = vec![0u8; hidden.size()];
            hidden.download(0, &mut b, st)?;
            d.insert("cond".into(), ("F32".into(), vec![total, qwen35::D], b));
            let mut b = vec![0u8; ctx.ctx.size()];
            ctx.ctx.download(0, &mut b, st)?;
            d.insert("text_embedder_out".into(), ("BF16".into(), vec![total, 3072], b));
        }
        remember(&mut self.contexts, (key.clone(), ctx), CONTEXTS);
        Ok(())
    }
}

impl super::Family {
    fn moss(&mut self) -> Option<&mut Moss> {
        match self {
            super::Family::Moss(f) => Some(f),
            _ => None,
        }
    }
}

impl Generator {
    /// Load a MOSS Image diffusion model unless `path` is loaded.
    pub fn ensure_moss(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Moss)?;
        let env = &self.env;
        let f = self.family.moss().context("MOSS Image not entered")?;
        if f.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.clear_prompts();
        f.model = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = moss::globals_layout(&dst)?.size;
        let free = cuda::device_free()?;
        let budget = free.saturating_sub(env.reserve + globals + moss_need(env.max_width, env.max_height, 1));
        let mut dit = moss::Dit::load(&dst, budget, env.slots, env.threads, &env.st)?;
        dit.attn_variant = env.attn_variant;
        dit.int8_attn = env.int8_attn;
        eprintln!("MOSS Image ready in {:.1}s", t0.elapsed().as_secs_f64());
        f.model = Some(Model { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Load MOSS Image's Qwen3.5 text encoder unless `path` is loaded; streamed from pinned RAM,
    /// since it doesn't fit in VRAM with the blocks on most cards.
    pub fn ensure_qwen35(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Moss)?;
        let env = &self.env;
        let f = self.family.moss().context("MOSS Image not entered")?;
        if f.te.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.clear_prompts();
        f.te = None;
        let t0 = Instant::now();
        if f.tokenizer.is_none() {
            f.tokenizer = Some(tokenizer::Tokenizer::qwen35()?);
        }
        let size = qwen35::weight_bytes(&st::SafeTensors::open(path)?)?;
        let keep = env.reserve + moss_need(env.max_width, env.max_height, 1);
        let budget = if cuda::device_free()? >= size + keep { size } else { 0 };
        f.te = Some((path.to_path_buf(), qwen35::TextEncoder::load(path, budget, env.threads, &env.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Load the Flux 2 VAE unless `path` is loaded.
    pub fn ensure_flux2_vae(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Moss)?;
        let st = &self.env.st;
        let f = self.family.moss().context("MOSS Image not entered")?;
        if f.vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        f.vae = None;
        let t0 = Instant::now();
        f.vae = Some((path.to_path_buf(), flux2vae::Vae::load(path, st)?));
        eprintln!("VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// MOSS Image latents, sampled as ComfyUI's KSampler does.
    pub fn sample_moss(&mut self, r: &MossRequest, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        env.check_size(r.width, r.height)?;
        let (lh, lw) = (r.height / 16, r.width / 16);
        let images = r.sampling.images;
        let n = images * moss::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let f = self.family.moss().context("no diffusion model loaded")?;
        let together = {
            let m = f.model.as_mut().context("no diffusion model loaded")?;
            let together = env.batch_fit(&m.dit.blocks, images, 0, step_need(r.width, r.height))?;
            env.plan(&mut m.dit.blocks, &m.st, moss_need(r.width, r.height, together))?;
            together
        };
        let st = &env.st;
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let guided = r.sampling.uses_negative();
        let pos_key = f.key(&r.positive)?;
        let neg_key = f.key(&r.negative)?;
        if guided {
            f.context(&neg_key, st, None)?;
        }
        f.context(&pos_key, st, dump.as_deref_mut())?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        let m = f.model.as_mut().unwrap();
        let contexts = &f.contexts;
        let find = |p: &TokenKey| contexts.iter().find(|(k, _)| k == p).map(|(_, c)| c).unwrap();
        let pos = find(&pos_key);
        let neg = guided.then(|| find(&neg_key));
        let rope = m.dit.rope(lh, lw, pos.len, st)?;
        let neg_rope = neg.map(|c| m.dit.rope(lh, lw, c.len, st)).transpose()?;
        let copied0 = m.dit.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, eval: usize, k: usize| -> Result<()> {
            if negative {
                return m.dit.predict(x, k, lh, lw, neg.unwrap(), neg_rope.as_ref().unwrap(), sigma, out, st, None);
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, k * step_need(r.width, r.height))?;
            }
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && matches!(name, "video_embedder_out" | "block0" | "block1" | "block41" | "final_tokens") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probes.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.predict(x, k, lh, lw, pos, &rope, sigma, out, st, probe_ref)?;
            }
            if let Some(dm) = dump.as_deref_mut() {
                for (name, v) in probes {
                    dm.insert(name, ("BF16".into(), vec![v.len() / 2], v));
                }
            }
            Ok(())
        };
        let latent = run_batches(&r.sampling, &r.noise, together, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(dm) = dump {
            dm.insert("samples".into(), ("F32".into(), vec![moss::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }

    /// Decode a host MOSS Image latent ([128, H/16, W/16] f32) to RGB8 pixels.
    pub fn decode_moss(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / 16, width / 16);
        if latent.len() != moss::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), moss::CH * lh * lw);
        }
        self.make_room(flux2vae::Vae::activation_bytes(lh, lw))?;
        let st = &self.env.st;
        let vae = &self.family.moss().and_then(|f| f.vae.as_ref()).context("no VAE loaded")?.1;
        vae.decode(&upload(latent, st)?, lh, lw, st)
    }

    /// Encode RGB f32 [height, width, 3], 0 to 1, with the Flux 2 VAE into a MOSS Image latent
    /// ([128, H/16, W/16] f32). Pixels past a multiple of 16 are cropped evenly from each side, as ComfyUI's VAE does.
    pub fn encode_moss(&mut self, rgb: &[f32], width: usize, height: usize) -> Result<Reference> {
        let (cropped, cw, ch) = crop(rgb, width, height, 3, 16)?;
        self.make_room(flux2vae::Vae::activation_bytes(ch / 16, cw / 16))?;
        let st = &self.env.st;
        let vae = &self.family.moss().and_then(|f| f.vae.as_ref()).context("no VAE loaded")?.1;
        let t0 = Instant::now();
        let latent = vae.encode(&cropped, ch, cw, st)?;
        eprintln!("VAE encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        Ok(Reference { latent, height: ch / 16, width: cw / 16 })
    }
}
