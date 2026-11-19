//! Qwen Image 2.1: its diffusion model, its VAE and encoded prompts. The text encoder is the generator's
//! (the 8B Qwen3-VL), shared with Krea 2.

use super::{crop, nothing_for, regions, remember, run_batches, shared_te_loras, touch, upload, PLAIN};
use super::{Cond, Dump, Env, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Side, Stats, Step, CONTEXTS};
use crate::cuda::{DevBuf, Stream};
use crate::vision::{self, PromptImage};
use crate::{cuda, lora, offload, qwen21, st, te, vae21};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Largest prefix cache kept in VRAM for one prompt; past this the prefix runs again in every step.
const CACHE_LIMIT: usize = 3 << 30;
/// Largest prefix cache kept for all of a request's prompts together.
const CACHE_TOTAL: usize = 2 * CACHE_LIMIT;
/// Text rows budgeted when planning before the prompts are known.
const QWEN21_TEXT: usize = 300;

pub struct Qwen21Request {
    /// The positive conditioning entries, mixed as ComfyUI's `calc_cond_batch` mixes them.
    pub positive: Vec<Cond>,
    /// Only encoded when the sampling uses it.
    pub negative: Vec<Cond>,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise, one after another.
    pub noise: Vec<f32>,
    /// Each entry's prompt images for the vision tower, which the DiT takes the place of as reference latents: one list
    /// per entry of `positive`, then one per entry of `negative`.
    pub images: [Vec<Vec<PromptImage>>; 2],
    /// Each entry's reference latents, from [`Generator::encode_image21`], as `images`.
    pub references: [Vec<Vec<Reference>>; 2],
    /// `SwarmAttnTokenWeights`: the first `<break>` chunk of its conditioning's first entry, and that entry's prompt
    /// image count.
    pub attn_token_weights: Option<(Vec<(String, f32)>, usize)>,
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
    prompt: Prompt,
    images: Vec<PromptImage>,
}

struct Context21 {
    ctx: qwen21::Context,
    slots: Vec<usize>,
}

// Only reached through the Generator, which one thread holds at a time.
unsafe impl Sync for Context21 {}

/// What SwarmTextEncodeAdvanced does with the weights of a `<break>` chunk on a tokenizer that has none, as Qwen Image
/// 2.1's (`use_attn_token_weights`): it encodes at weight 1 and multiplies the rows of weighted tokens, or every row by
/// a weight all the text shares if that left none. With weights that differ it also records the weighted rows for
/// `SwarmAttnTokenWeights`.
#[derive(Debug, Default, PartialEq)]
struct ChunkWeights {
    /// (row of the conditioning, factor)
    rows: Vec<(usize, f32)>,
    /// `attn_token_weights`
    attn: Vec<(usize, f32)>,
}

/// [`ChunkWeights`] of `segments`, whose tokens weigh `weights`, for a conditioning of `rows` rows.
fn chunk_weights(segments: &[(String, f32)], weights: &[f32], rows: usize) -> ChunkWeights {
    if segments.iter().all(|(_, w)| *w == 1.0) {
        return ChunkWeights::default();
    }
    // `uniform_weight`: the weight of every segment with text, if they share one
    let mut texts = segments.iter().filter(|(t, _)| !t.trim().is_empty()).map(|(_, w)| *w);
    let uniform = texts.next().filter(|&w| texts.all(|v| v == w));
    let pairs = te::qwen21_weight_rows(weights, rows);
    let attn = if uniform.is_none() { pairs.clone() } else { Vec::new() };
    let rows = match uniform {
        Some(w) if pairs.is_empty() && w != 1.0 => (0..rows).map(|r| (r, w)).collect(),
        _ => pairs,
    };
    ChunkWeights { rows, attn }
}

fn same_references(a: &[Reference], b: &[Reference]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.height == b.height && a.width == b.width && a.latent == b.latent)
}

#[derive(Default)]
pub(super) struct Qwen21 {
    model: Option<Model>,
    vae: Option<(PathBuf, vae21::Vae)>,
    /// Latest encoded prompts, most recent last. They hold `txt_in`'s output; a request keeps its own until it ends.
    contexts: Vec<(Key, Arc<Context21>)>,
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

/// One image's activations in a step over an image of `width` x `height` with `prefix` rows before it.
fn step_need(width: usize, height: usize, prefix: usize) -> usize {
    let l = (height / vae21::SCALE) * (width / vae21::SCALE);
    qwen21::step_activation_bytes(l, prefix.max(QWEN21_TEXT) + l)
}

/// Peak activation memory of a request of `images` sampled together: their step, the prefix's, or the VAE.
fn qwen21_need(width: usize, height: usize, prefix: usize, images: usize) -> usize {
    let (lh, lw) = (height / vae21::SCALE, width / vae21::SCALE);
    let prefix = prefix.max(QWEN21_TEXT);
    (images * step_need(width, height, prefix)).max(vae21::Vae::activation_bytes(lh, lw)).max(qwen21::step_activation_bytes(prefix, prefix))
}

impl Qwen21 {
    /// The context of `prompt` with `images`, moved to the end of the cache, encoding it if needed. Its `<break>` chunks
    /// are encoded apart and joined along the tokens with the first one's image slots, as SwarmTextEncodeAdvanced keeps
    /// the first chunk's extra keys.
    fn context(&mut self, te: &mut SharedTe, env: &Env, prompt: &Prompt, images: &[PromptImage], mut dump: Option<&mut Dump>)
        -> Result<Arc<Context21>> {
        let key = Key { prompt: prompt.clone(), images: images.to_vec() };
        if touch(&mut self.contexts, |(k, _)| *k == key) {
            return Ok(self.contexts.last().unwrap().1.clone());
        }
        if prompt.chunks.is_empty() {
            bail!("a prompt without chunks");
        }
        let st = &env.st;
        let embeds = te.vision_embeds(images, env, dump.as_deref_mut())?;
        let refs: Vec<&vision::Embeds> = embeds.iter().map(|e| &**e).collect();
        let t0 = Instant::now();
        if te.enc.spec != te::QWEN21 {
            bail!("this text encoder is not Qwen Image 2.1's (the 8B)");
        }
        let m = self.model.as_mut().context("no diffusion model loaded")?;
        let mut conds = Vec::with_capacity(prompt.chunks.len());
        for segments in &prompt.chunks {
            let (ids, weights) = te.enc.qwen21_tokens(segments, images.len())?;
            let rows = chunk_weights(segments, &weights, te::qwen21_rows(&ids, images.len())).rows;
            conds.push(te.enc.encode_qwen21(&ids, &refs, false, &rows, prompt.scale, st)?);
        }
        let slots = conds[0].slots.clone();
        let mut cond = te::Cond::concat(conds, st)?;
        cond.slots = slots;
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
        let c = Arc::new(Context21 { ctx, slots: cond.slots });
        remember(&mut self.contexts, (key, c.clone()), CONTEXTS);
        Ok(c)
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
        let budget = free.saturating_sub(env.reserve + globals + qwen21_need(env.max_width, env.max_height, 0, 1));
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

    /// Qwen Image 2.1 latents, sampled as ComfyUI's KSampler does. Each conditioning entry is its own prefix: its
    /// prompt with its prompt images, and its reference latents.
    pub fn sample_qwen21(&mut self, r: &Qwen21Request, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        env.check_size(r.width, r.height)?;
        let (lh, lw) = (r.height / vae21::SCALE, r.width / vae21::SCALE);
        let images = r.sampling.images;
        let n = images * qwen21::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let sides: [&[Cond]; 2] = [&r.positive, &r.negative];
        for (s, conds) in sides.iter().enumerate() {
            if r.images[s].len() != conds.len() || r.references[s].len() != conds.len() {
                bail!("{} conditioning entries with {} lists of prompt images and {} of reference latents", conds.len(), r.images[s].len(),
                    r.references[s].len());
            }
        }
        let f = self.family.qwen21().context("no diffusion model loaded")?;
        let te = self.te.as_mut().context("no text encoder loaded")?;
        let guided = r.sampling.uses_negative();
        let used = if guided { 2 } else { 1 };
        let mut stats = Stats::default();
        let t0 = Instant::now();
        // every entry's context, held until the run ends whatever the cache drops
        let mut contexts: [Vec<Arc<Context21>>; 2] = Default::default();
        for s in 0..used {
            for (i, c) in sides[s].iter().enumerate() {
                let d = if s == 0 && i == 0 { dump.as_deref_mut() } else { None };
                contexts[s].push(f.context(te, env, &c.prompt, &r.images[s][i], d)?);
            }
        }
        // SwarmAttnTokenWeights' attn1 patch: the values of the rows it names times their weight below 1, in every
        // positive evaluation. Its bias for weights above 1 does nothing: the model doesn't read the patch's attn_mask.
        let v_scale: Vec<(usize, f32)> = match &r.attn_token_weights {
            Some((segments, count)) => {
                let (ids, weights) = te.enc.qwen21_tokens(segments, *count)?;
                chunk_weights(segments, &weights, te::qwen21_rows(&ids, *count)).attn.into_iter().filter(|&(_, w)| w < 1.0).collect()
            }
            None => Vec::new(),
        };
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        // one prefix per context, reference latents and value scaling
        let mut unique: Vec<(&Arc<Context21>, &[Reference], bool)> = Vec::new();
        let mut which: [Vec<usize>; 2] = Default::default();
        for s in 0..used {
            for (i, c) in contexts[s].iter().enumerate() {
                let (refs, scaled) = (&r.references[s][i][..], s == 0 && !v_scale.is_empty());
                let at = unique.iter().position(|&(uc, ur, us)| Arc::ptr_eq(uc, c) && same_references(ur, refs) && us == scaled);
                which[s].push(at.unwrap_or_else(|| {
                    unique.push((c, refs, scaled));
                    unique.len() - 1
                }));
            }
        }
        let rows: Vec<usize> = unique.iter().map(|(c, refs, _)| c.ctx.len + refs.iter().map(|l| l.height * l.width).sum::<usize>()).collect();
        let m = f.model.as_mut().context("no diffusion model loaded")?;
        let nb = m.dit.blocks.len();
        // The cache is worth its memory when it fits. It costs the same for every step it spares.
        let caching = dump.is_none() && std::env::var_os("QWEN21_NO_CACHE").is_none();
        let mut cache_total = 0;
        let cached: Vec<bool> = rows.iter().map(|&p| {
            let bytes = qwen21::Prefix::cache_bytes(nb, p);
            let fits = caching && bytes <= CACHE_LIMIT && cache_total + bytes <= CACHE_TOTAL;
            cache_total += fits as usize * bytes;
            fits
        }).collect();
        let prefix = rows.iter().copied().max().unwrap_or(0);
        let together = env.batch_fit(&m.dit.blocks, images, cache_total, step_need(r.width, r.height, prefix))?;
        env.plan(&mut m.dit.blocks, &m.st, qwen21_need(r.width, r.height, prefix, together) + cache_total)?;
        let st = &env.st;
        let t1 = Instant::now();
        let mut prefixes = Vec::with_capacity(unique.len());
        for (u, &(c, refs, scaled)) in unique.iter().enumerate() {
            let refs: Vec<(Vec<f32>, usize, usize)> = refs.iter().map(|l| (l.latent.clone(), l.height, l.width)).collect();
            let mut pf = m.dit.prefix(&c.ctx, &c.slots, &refs, (lh, lw), st)?;
            if scaled {
                pf.v_scale = v_scale.clone();
            }
            if cached[u] {
                m.dit.fill_cache(&mut pf, st, None)?;
            }
            prefixes.push(pf);
        }
        st.sync()?;
        let list: Vec<String> = rows.iter().zip(&cached).map(|(p, c)| format!("{p} rows{}", if *c { ", cached" } else { "" })).collect();
        eprintln!("prefix: {} in {:.0} ms", list.join("; "), t1.elapsed().as_secs_f64() * 1e3);
        stats.text_ms += t1.elapsed().as_secs_f64() * 1e3;

        let copied0 = m.dit.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, side: Side, eval: usize, k: usize| -> Result<()> {
            let Some(&u) = which[side.negative as usize].get(side.entry) else {
                bail!("no prompt encoded for entry {} of the {} side", side.entry, if side.negative { "negative" } else { "positive" });
            };
            let pf = &prefixes[u];
            if side.negative {
                return m.dit.predict(x, k, lh, lw, pf, sigma, out, st, None);
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, qwen21_need(r.width, r.height, prefix, k))?;
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
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() && side == Side::POSITIVE { Some(&mut probe) } else { None };
                m.dit.predict(x, k, lh, lw, pf, sigma, out, st, probe_ref)?;
            }
            if let Some(dm) = dump.as_deref_mut() {
                for (name, v) in probes {
                    dm.insert(name, ("BF16".into(), vec![v.len() / 2], v));
                }
            }
            Ok(())
        };
        // an unguided run never evaluates the negative side
        let negative = if guided { regions(&r.negative) } else { PLAIN.to_vec() };
        let latent = run_batches(&r.sampling, &r.noise, [&regions(&r.positive)[..], &negative[..]], together, st, &mut stats, &mut model, &mut progress)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(s: &[(&str, f32)]) -> Vec<(String, f32)> {
        s.iter().map(|(t, w)| (t.to_string(), *w)).collect()
    }

    /// SwarmTextEncodeAdvanced on ComfyUI's QwenImage21Tokenizer, from SwarmUI's code: which rows it multiplies and
    /// which it records for SwarmAttnTokenWeights.
    #[test]
    fn weights_follow_swarm() {
        let tok = crate::tokenizer::Tokenizer::new().unwrap();
        let weigh = |s: &[(&str, f32)], images: usize| {
            let segments = seg(s);
            let (ids, weights) = te::qwen21_tokens(&tok, &segments, images).unwrap();
            chunk_weights(&segments, &weights, te::qwen21_rows(&ids, images))
        };
        // no weights
        assert_eq!(weigh(&[("a lake at dawn", 1.0)], 0), ChunkWeights::default());
        // weights that differ: the rows, and the same pairs for the patch
        let w = weigh(&[("a ", 1.0), ("red", 1.5), (" boat", 1.0)], 0);
        assert_eq!(w, ChunkWeights { rows: vec![(5, 1.5)], attn: vec![(5, 1.5)] });
        assert_eq!(weigh(&[("a ", 1.0), ("red", 1.5), (" boat", 1.0)], 2).rows, [(17, 1.5)]);
        // one weight for all the text: the rows only
        assert_eq!(weigh(&[("a lake", 0.5)], 0), ChunkWeights { rows: vec![(3, 0.5), (4, 0.5)], attn: vec![] });
        assert_eq!(weigh(&[("a lake", 0.5)], 2).rows, [(15, 0.5), (16, 0.5)]);
        // a blank segment doesn't count for the uniform weight, but its tokens are weighted
        let w = weigh(&[("mist", 0.5), (" ", 2.0)], 0);
        assert_eq!(w, ChunkWeights { rows: vec![(3, 0.5), (4, 2.0)], attn: vec![] });
        // segments with the same weight around unweighted text
        let w = weigh(&[("mist", 0.5), (" on the ", 1.0), ("lake", 0.5)], 0);
        assert_eq!(w.attn, [(3, 0.5), (7, 0.5)]);
    }
}
