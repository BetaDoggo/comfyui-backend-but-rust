//! Qwen Image 2.1: its diffusion model, its VAE and encoded prompts. The text encoder is the generator's
//! (the 8B Qwen3-VL), shared with Krea 2.

use super::{crop, nothing_for, remember, run_sampler, shared_te_loras, touch, upload};
use super::{Dump, Env, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Stats, Step, CONTEXTS};
use crate::cuda::{DevBuf, Stream};
use crate::vision::{self, PromptImage};
use crate::{cuda, lora, offload, qwen21, st, te, vae21};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Largest prefix cache kept in VRAM for one prompt; past this the prefix runs again in every step.
const CACHE_LIMIT: usize = 3 << 30;
/// Text rows budgeted when planning before the prompts are known.
const QWEN21_TEXT: usize = 300;

pub struct Qwen21Request {
    pub positive: Prompt,
    /// Only encoded when the sampling uses it.
    pub negative: Prompt,
    pub width: usize,
    pub height: usize,
    pub sampling: Sampling,
    pub noise: Vec<f32>,
    /// Prompt images for the vision tower, which the DiT takes the place of as reference latents.
    pub images: Vec<PromptImage>,
    pub negative_images: Vec<PromptImage>,
    /// Reference latents of both prompts, from [`Generator::encode_image21`].
    pub references: Vec<Reference>,
}

struct Model {
    path: PathBuf,
    dit: qwen21::Dit,
    st: st::SafeTensors,
    /// the file's key prefix
    prefix: &'static str,
    lora: lora::State,
}

/// A prompt as the text encoder saw it.
#[derive(PartialEq)]
struct Key {
    text: String,
    scale: f32,
    images: Vec<PromptImage>,
}

impl Key {
    fn new(prompt: &Prompt, images: &[PromptImage]) -> Result<Self> {
        let text: String = prompt.segments()?.iter().map(|(t, _)| t.as_str()).collect();
        Ok(Key { text: te::qwen21_prompt(&text, images.len()), scale: prompt.scale, images: images.to_vec() })
    }
}

struct Context21 {
    ctx: qwen21::Context,
    slots: Vec<usize>,
}

#[derive(Default)]
pub(super) struct Qwen21 {
    model: Option<Model>,
    vae: Option<(PathBuf, vae21::Vae)>,
    /// Latest encoded prompts, most recent last. They hold `txt_in`'s output.
    contexts: Vec<(Key, Context21)>,
}

impl FamilyState for Qwen21 {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        if lora::apply(lora::Arch::Qwen21, &m.st, m.prefix, m.dit.stores(), &mut m.lora, loras, st)? {
            self.clear_prompts();
        }
        Ok(())
    }

    fn set_text_encoder_loras(&mut self, shared: Option<&mut SharedTe>, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        if shared_te_loras(shared, loras, st)? {
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

/// Peak activation memory of one request: a step over an image of `width` x `height` with `prefix` rows
/// before it, or the VAE.
fn qwen21_need(width: usize, height: usize, prefix: usize) -> usize {
    let (lh, lw) = (height / vae21::SCALE, width / vae21::SCALE);
    let prefix = prefix.max(QWEN21_TEXT);
    qwen21::step_activation_bytes(lh * lw, prefix + lh * lw).max(vae21::Vae::activation_bytes(lh, lw)).max(qwen21::step_activation_bytes(prefix, prefix))
}

impl Qwen21 {
    /// Move the context of `prompt` with `images` to the end of the cache, encoding it if needed.
    fn context(&mut self, te: &mut SharedTe, env: &Env, prompt: &Prompt, images: &[PromptImage], mut dump: Option<&mut Dump>) -> Result<()> {
        let key = Key::new(prompt, images)?;
        if touch(&mut self.contexts, |(k, _)| *k == key) {
            return Ok(());
        }
        let st = &env.st;
        let embeds = te.vision_embeds(images, env, dump.as_deref_mut())?;
        let refs: Vec<&vision::Embeds> = embeds.iter().map(|e| &**e).collect();
        let t0 = Instant::now();
        if te.enc.spec != te::QWEN21 {
            bail!("this text encoder is not Qwen Image 2.1's (the 8B)");
        }
        let m = self.model.as_mut().context("no diffusion model loaded")?;
        let cond = te.enc.encode_qwen21(&key.text, &refs, false, prompt.scale, st)?;
        let ctx = m.dit.encode_context(&cond.ctx, cond.len, st)?;
        st.sync()?;
        eprintln!("text: {} tokens ({} conditioning, {} images) in {:.0} ms", cond.token_ids.len(), cond.len, images.len(), t0.elapsed().as_secs_f64() * 1e3);
        if let Some(d) = dump {
            let ids: Vec<u8> = cond.token_ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect();
            d.insert("ids".into(), ("I64".into(), vec![cond.token_ids.len()], ids));
            let mut b = vec![0u8; cond.ctx.size()];
            cond.ctx.download(0, &mut b, st)?;
            d.insert("cond".into(), ("BF16".into(), vec![cond.len, qwen21::F], b));
            let mut b = vec![0u8; ctx.ctx.size()];
            ctx.ctx.download(0, &mut b, st)?;
            d.insert("txt_in".into(), ("BF16".into(), vec![cond.len, qwen21::F], b));
            let slots: Vec<u8> = cond.slots.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect();
            d.insert("image_slots".into(), ("I64".into(), vec![cond.slots.len()], slots));
        }
        remember(&mut self.contexts, (key, Context21 { ctx, slots: cond.slots }), CONTEXTS);
        Ok(())
    }
}

impl super::Family {
    fn qwen21(&mut self) -> Option<&mut Qwen21> {
        match self {
            super::Family::Qwen21(f) => Some(f),
            _ => None,
        }
    }
}

impl Generator {
    /// Load a Qwen Image 2.1 diffusion model unless `path` is loaded. Its text encoder is [`Generator::ensure_te`]'s, the
    /// 8B this time.
    pub fn ensure_qwen21(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Qwen21)?;
        let env = &self.env;
        let f = self.family.qwen21().context("Qwen Image 2.1 not entered")?;
        if f.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.clear_prompts();
        f.model = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let pre = qwen21::key_prefix(|k| dst.has(k)).context("not a Qwen Image 2.1 checkpoint")?;
        let globals = qwen21::globals_layout(&dst, pre)?.size;
        let free = cuda::device_free()?;
        let budget = free.saturating_sub(env.reserve + globals + qwen21_need(env.max_width, env.max_height, 0));
        let mut dit = qwen21::Dit::load(&dst, pre, budget, env.slots, env.threads, &env.st)?;
        dit.attn_variant = env.attn_variant;
        dit.int8_attn = env.int8_attn;
        eprintln!("Qwen Image 2.1 ready in {:.1}s", t0.elapsed().as_secs_f64());
        f.model = Some(Model { path: path.to_path_buf(), dit, st: dst, prefix: pre, lora: lora::State::default() });
        Ok(())
    }

    /// Load Qwen Image 2.1's VAE unless `path` is loaded.
    pub fn ensure_vae21(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Qwen21)?;
        let st = &self.env.st;
        let f = self.family.qwen21().context("Qwen Image 2.1 not entered")?;
        if f.vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        f.vae = None;
        let t0 = Instant::now();
        f.vae = Some((path.to_path_buf(), vae21::Vae::load(path, st)?));
        eprintln!("VAE ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// One Qwen Image 2.1 latent, sampled as ComfyUI's KSampler does.
    pub fn sample_qwen21(&mut self, r: &Qwen21Request, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        env.check_size(r.width, r.height)?;
        let (lh, lw) = (r.height / vae21::SCALE, r.width / vae21::SCALE);
        let n = qwen21::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let f = self.family.qwen21().context("no diffusion model loaded")?;
        let te = self.te.as_mut().context("no text encoder loaded")?;
        let guided = r.sampling.uses_negative();
        let mut stats = Stats::default();
        let t0 = Instant::now();
        if guided {
            f.context(te, env, &r.negative, &r.negative_images, None)?;
        }
        f.context(te, env, &r.positive, &r.images, dump.as_deref_mut())?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        let refs: Vec<(Vec<f32>, usize, usize)> = r.references.iter().map(|l| (l.latent.clone(), l.height, l.width)).collect();
        let ref_rows: usize = refs.iter().map(|(_, h, w)| h * w).sum();
        let contexts = &f.contexts;
        let find = |k: &Key| contexts.iter().find(|(key, _)| key == k).map(|(_, c)| c).unwrap();
        let (pos_key, neg_key) = (Key::new(&r.positive, &r.images)?, Key::new(&r.negative, &r.negative_images)?);
        let rows = |c: &Context21| c.ctx.len + ref_rows;
        let (pos_rows, neg_rows) = (rows(find(&pos_key)), if guided { rows(find(&neg_key)) } else { 0 });
        let m = f.model.as_mut().context("no diffusion model loaded")?;
        let nb = m.dit.blocks.len();
        // The cache is worth its memory when it fits. It costs the same for every step it spares.
        let caching = dump.is_none() && std::env::var_os("QWEN21_NO_CACHE").is_none();
        let fits = |p: usize| qwen21::Prefix::cache_bytes(nb, p) <= CACHE_LIMIT;
        let (cache_pos, cache_neg) = (caching && fits(pos_rows), caching && guided && fits(neg_rows));
        let cache_total = (cache_pos as usize) * qwen21::Prefix::cache_bytes(nb, pos_rows) + (cache_neg as usize) * qwen21::Prefix::cache_bytes(nb, neg_rows);
        env.plan(&mut m.dit.blocks, &m.st, qwen21_need(r.width, r.height, pos_rows.max(neg_rows)) + cache_total)?;
        let st = &env.st;
        let t1 = Instant::now();
        let mut build = |c: &Context21, cache: bool| -> Result<qwen21::Prefix> {
            let mut pf = m.dit.prefix(&c.ctx, &c.slots, &refs, (lh, lw), st)?;
            if cache {
                m.dit.fill_cache(&mut pf, st, None)?;
            }
            Ok(pf)
        };
        let pos_pf = build(find(&pos_key), cache_pos)?;
        let neg_pf = if guided { Some(build(find(&neg_key), cache_neg)?) } else { None };
        st.sync()?;
        eprintln!("prefix: {pos_rows} rows{}{} in {:.0} ms", if cache_pos { ", cached" } else { "" },
            if guided { format!(" and {neg_rows}{}", if cache_neg { ", cached" } else { "" }) } else { String::new() }, t1.elapsed().as_secs_f64() * 1e3);
        stats.text_ms += t1.elapsed().as_secs_f64() * 1e3;

        let copied0 = m.dit.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, eval: usize| -> Result<()> {
            if negative {
                return m.dit.predict(x, lh, lw, neg_pf.as_ref().unwrap(), sigma, out, st, None);
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, qwen21_need(lw * vae21::SCALE, lh * vae21::SCALE, pos_rows))?;
            }
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && matches!(name, "temb" | "block0" | "block1" | "block31" | "prefix_block0" | "prefix_block1" | "prefix_block31" | "proj_out") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probes.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.predict(x, lh, lw, &pos_pf, sigma, out, st, probe_ref)?;
            }
            if let Some(dm) = dump.as_deref_mut() {
                for (name, v) in probes {
                    dm.insert(name, ("BF16".into(), vec![v.len() / 2], v));
                }
            }
            Ok(())
        };
        let latent = run_sampler(&r.sampling, &r.noise, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(dm) = dump {
            dm.insert("samples".into(), ("F32".into(), vec![qwen21::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
            qwen21::TRACE.with(|t| {
                for (name, v) in t.borrow_mut().drain(..) {
                    dm.entry(name).or_insert(("BF16".into(), vec![v.len() / 2], v));
                }
            });
        }
        Ok((latent, stats))
    }

    /// VAE-encode RGBA f32 [height, width, 4], 0 to 1, into a reference latent. Pixels past a multiple of 16 are
    /// cropped evenly from each side, as ComfyUI's VAE does.
    pub fn encode_image21(&mut self, rgba: &[f32], width: usize, height: usize) -> Result<Reference> {
        let s = vae21::SCALE;
        let (cropped, cw, ch) = crop(rgba, width, height, 4, s)?;
        self.make_room(vae21::Vae::activation_bytes(ch / s, cw / s))?;
        let st = &self.env.st;
        let vae = &self.family.qwen21().and_then(|f| f.vae.as_ref()).context("no VAE loaded")?.1;
        let t0 = Instant::now();
        let latent = vae.encode(&cropped, ch, cw, false, st)?;
        eprintln!("VAE encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        Ok(Reference { latent, height: ch / s, width: cw / s })
    }

    /// Decode a host Qwen Image 2.1 latent ([64, H/16, W/16] f32) to RGBA8 pixels.
    pub fn decode_qwen21(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / vae21::SCALE, width / vae21::SCALE);
        if latent.len() != qwen21::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), qwen21::CH * lh * lw);
        }
        self.make_room(vae21::Vae::activation_bytes(lh, lw))?;
        let st = &self.env.st;
        let vae = &self.family.qwen21().and_then(|f| f.vae.as_ref()).context("no VAE loaded")?.1;
        vae.decode(&upload(latent, st)?, lh, lw, st)
    }
}
