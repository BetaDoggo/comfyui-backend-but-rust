//! Anima: its diffusion model, its Qwen3 text encoder and the prompts encoded through them. The VAE is the generator's.

use super::{regions, Cond, PLAIN, Side, nothing_for, remember, run_batches, scale_rows, touch};
use super::{Dump, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Stats, Step, CONTEXTS};
use crate::cuda::{self, DevBuf, Stream};
use crate::{anima, dit, lora, offload, qwen3, snapref, st, tokenizer, vae};
use std::hash::{Hash, Hasher};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct AnimaRequest {
    /// The positive side's conditioning entries, mixed as ComfyUI's `calc_cond_batch` does. A prompt's chunks are
    /// segments (see [`tokenizer::token_weights`]), joined as [`chunk_tokens`] says.
    pub positive: Vec<Cond>,
    /// Only encoded when the sampling uses it.
    pub negative: Vec<Cond>,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise, one after another.
    pub noise: Vec<f32>,
    /// SnapRef's reference adapter, if any.
    pub snapref: Option<SnapRefRequest>,
    /// A training-free style reference for this request (the `BrokeAnimaStyleReference` node); without one, the one set
    /// with [`super::Generator::style_anima`], if any.
    pub style: Option<StyleRequest>,
}

/// The SnapRef node's settings (see [`crate::snapref`]).
#[derive(Clone)]
pub struct SnapRefRequest {
    /// Its weight file.
    pub adapter: PathBuf,
    /// The pictures' latents at model scale, each encoded on its own.
    pub references: Vec<Reference>,
    /// The log bias on each reference's keys: `strength + ln(weight)`.
    pub biases: Vec<f32>,
    /// `cs` (character and style) or `char`.
    pub mode: String,
    /// `ref_caption`, the reference stream's prompt.
    pub caption: Prompt,
    /// The negative prompt's evaluations read the references too, at the same biases.
    pub in_uncond: bool,
    /// Fidelity below 1: the negative's evaluations read the references, at this extra bias. Ignored with `in_uncond`.
    pub uncond_bias: Option<f32>,
}

/// A training-free style reference: the picture is noised to each step's sigma with fixed noise and run through the
/// plain model, and the target's self-attention also reads its keys and values ("shared attention").
#[derive(Clone)]
pub struct StyleRequest {
    /// The picture's latent at model scale.
    pub reference: Reference,
    /// The reference pass's prompt.
    pub caption: Prompt,
    /// Natural log of the weight on the reference's attention logits.
    pub bias: f32,
    /// Half-open ranges of the steps and blocks that read the reference.
    pub steps: (usize, usize),
    pub blocks: (usize, usize),
    /// The reference's RoPE frame: 0 puts its tokens on the target's positions.
    pub frame: f32,
    /// Seed of the noise the picture is noised with.
    pub seed: u64,
    /// The target's self-attention reads the reference's keys and values.
    pub attention: bool,
    /// Residual AdaIN towards the reference's statistics (see [`anima::AdaIn`]): its strength (0 = off), and blocks.
    pub adain: f32,
    pub adain_blocks: (usize, usize),
    /// Shuffle the reference's latent in square tiles of this many latent pixels (even; 0 = off), so that its local
    /// texture survives but no object does.
    pub shuffle: usize,
}

impl StyleRequest {
    /// The `BrokeAnimaStyleReference` node's prompt for the reference pass.
    pub const NODE_CAPTION: &'static str = "no humans, scenery, detailed background";
    /// The node's prompt for the reference pass on Krea 2 (`KREA2_CAPTION`), whose text encoder reads sentences.
    pub const KREA2_NODE_CAPTION: &'static str = "scenery with a detailed background, no people";
    /// The node's default `reference_megapixels`: it shrinks the picture to this area (never enlarging it).
    pub const NODE_MEGAPIXELS: f64 = 1.0;
    /// The range the node accepts for `reference_megapixels`.
    pub const NODE_MEGAPIXELS_RANGE: std::ops::RangeInclusive<f64> = 0.5..=4.0;
    /// The node's noise for the picture: `torch.randn` from this seed on the CPU, as [`crate::sampling::torch_randn`].
    pub const NODE_SEED: u64 = 0x5EED;

    /// The `BrokeAnimaStyleReference` node (and the SwarmUI extension): residual AdaIN at `stats_match` in every block, no
    /// shared attention. With `restyle`, the first quarter of the `steps` (rounded) runs without the picture.
    pub fn node(reference: Reference, stats_match: f32, restyle: bool, steps: usize) -> Self {
        let first = if restyle { (steps as f64 * 0.25).round() as usize } else { 0 };
        StyleRequest {
            reference,
            caption: Prompt::new(tokenizer::token_weights(Self::NODE_CAPTION, 1.0)),
            bias: 0.0,
            steps: (first, usize::MAX),
            blocks: (0, usize::MAX),
            frame: 0.0,
            seed: Self::NODE_SEED,
            attention: false,
            adain: stats_match,
            adain_blocks: (0, usize::MAX),
            shuffle: 0,
        }
    }
}

/// `latent` (f32 [ch, h, w]) cut into `tile` x `tile` tiles, rearranged by a permutation from `seed`. Edge tiles that
/// don't fill a whole tile stay where they are.
pub fn shuffle_tiles(latent: &[f32], ch: usize, h: usize, w: usize, tile: usize, seed: u64) -> Vec<f32> {
    let (th, tw) = (h / tile, w / tile);
    let n = th * tw;
    let mut perm: Vec<usize> = (0..n).collect();
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for i in (1..n).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        perm.swap(i, (x % (i as u64 + 1)) as usize);
    }
    let mut out = latent.to_vec();
    for (dst, &src) in perm.iter().enumerate() {
        let (dy, dx, sy, sx) = (dst / tw * tile, dst % tw * tile, src / tw * tile, src % tw * tile);
        for c in 0..ch {
            for y in 0..tile {
                let (o, i) = (c * h * w + (dy + y) * w + dx, c * h * w + (sy + y) * w + sx);
                out[o..o + tile].copy_from_slice(&latent[i..i + tile]);
            }
        }
    }
    out
}

/// What the cached reference keys depend on, hashed.
fn ref_key(r: &SnapRefRequest, target: (usize, usize), int8_attn: bool) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    r.adapter.hash(&mut h);
    r.mode.hash(&mut h);
    format!("{:?}", r.caption).hash(&mut h);
    (target, int8_attn).hash(&mut h);
    for x in &r.references {
        (x.height, x.width).hash(&mut h);
        x.latent.iter().for_each(|v| v.to_bits().hash(&mut h));
    }
    h.finish()
}

struct Model {
    path: PathBuf,
    dit: anima::Dit,
    st: st::SafeTensors,
    lora: lora::State,
}

#[derive(Default)]
pub(super) struct Anima {
    model: Option<Model>,
    te: Option<(PathBuf, qwen3::TextEncoder)>,
    tokenizers: Option<(tokenizer::Tokenizer, tokenizer::T5Tokenizer)>,
    /// Latest encoded prompts, most recent last.
    contexts: Vec<(Prompt, anima::Context)>,
    /// SnapRef's weights, loaded by the first request that uses them.
    snapref: Option<snapref::Adapter>,
    /// The latest references' keys and values, by [`ref_key`]. They come out of the diffusion model and the caption.
    ref_kv: Option<(u64, snapref::RefKv)>,
    /// A training-free style reference for every request until cleared.
    style: Option<StyleRequest>,
}

impl FamilyState for Anima {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        let prefix = m.dit.prefix();
        if lora::apply(lora::Arch::Anima, &m.st, &prefix, m.dit.stores(), &mut m.lora, loras, st)? {
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
        if let Some(m) = &mut self.model {
            m.dit.concept = None;
        }
        self.ref_kv = None;
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
        // about 0.9 GB per megapixel of reference, and cheap to make again
        self.ref_kv = None;
        if let Some(m) = &mut self.model {
            m.dit.blocks.set_resident(0, 1, &m.st)?;
        }
        Ok(())
    }
}

/// One image's activations in a denoising step.
fn step_need(width: usize, height: usize) -> usize {
    anima::step_activation_bytes((height / 16) * (width / 16))
}

/// Peak activation memory of `images` sampled together (denoising step or VAE decode, whichever is larger).
fn anima_need(width: usize, height: usize, images: usize) -> usize {
    (images * step_need(width, height)).max(vae::Vae::activation_bytes(height / 8, width / 8))
}

/// The tokens of a prompt's `<break>` chunks, as SwarmTextEncodeAdvanced joins their conditioning: each chunk's Qwen3
/// ids, encoded apart and their hidden states concatenated along tokens, and the first chunk's T5 ids and weights,
/// the only extras the joined conditioning keeps. The LLM adapter then reads every chunk's states (at positions that
/// run on across chunks) with the first chunk's T5 tokens; the later chunks' T5 ids and weights are dropped.
fn chunk_tokens(tq: &tokenizer::Tokenizer, tt: &tokenizer::T5Tokenizer, chunks: &[Vec<(String, f32)>])
    -> Result<(Vec<Vec<u32>>, Vec<u32>, Vec<f32>)> {
    let (first, rest) = chunks.split_first().context("a prompt without chunks")?;
    let (q, tids, weights) = tokenizer::anima_tokens(tq, tt, first)?;
    let mut qids = vec![q];
    for chunk in rest {
        qids.push(tokenizer::anima_tokens(tq, tt, chunk)?.0);
    }
    Ok((qids, tids, weights))
}

impl Anima {
    /// Move `prompt`'s context to the end of the cache, encoding it if needed.
    fn context(&mut self, prompt: &Prompt, st: &Stream, dump: Option<&mut Dump>) -> Result<()> {
        if touch(&mut self.contexts, |(p, _)| p == prompt) {
            return Ok(());
        }
        let m = self.model.as_mut().context("no diffusion model loaded")?;
        let te = &mut self.te.as_mut().context("no text encoder loaded")?.1;
        let (tq, tt) = self.tokenizers.as_ref().context("no text encoder loaded")?;
        let t0 = Instant::now();
        let (chunks, tids, weights) = chunk_tokens(tq, tt, &prompt.chunks)?;
        let qids: Vec<u32> = chunks.concat();
        let row = qwen3::D * 4;
        let hidden = if let [ids] = chunks.as_slice() {
            te.encode(ids, st)?
        } else {
            let joined = DevBuf::alloc_async(qids.len() * row, st)?;
            let mut at = 0;
            for ids in &chunks {
                let h = te.encode(ids, st)?;
                cuda::memcpy_d2d(joined.at(at * row), h.ptr(), ids.len() * row, st)?;
                at += ids.len();
            }
            joined
        };
        scale_rows(&hidden, qwen3::D, &vec![prompt.scale; qids.len()], st)?;
        let ctx = m.dit.encode_context(&m.st, &hidden, qids.len(), &tids, &weights, st)?;
        st.sync()?;
        eprintln!("text: {} Qwen3 / {} T5 tokens in {:.0} ms", qids.len(), tids.len(), t0.elapsed().as_secs_f64() * 1e3);
        if let Some(d) = dump {
            let ids = |v: &[u32]| v.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect::<Vec<u8>>();
            d.insert("qwen_ids".into(), ("I64".into(), vec![qids.len()], ids(&qids)));
            d.insert("t5_ids".into(), ("I64".into(), vec![tids.len()], ids(&tids)));
            let mut b = vec![0u8; hidden.size()];
            hidden.download(0, &mut b, st)?;
            d.insert("cond".into(), ("F32".into(), vec![qids.len(), qwen3::D], b));
            let mut b = vec![0u8; ctx.ctx.size()];
            ctx.ctx.download(0, &mut b, st)?;
            d.insert("dit_in_ctx".into(), ("BF16".into(), vec![ctx.len, anima::CTX], b));
        }
        remember(&mut self.contexts, (prompt.clone(), ctx), CONTEXTS);
        Ok(())
    }

    /// The contexts of a run's prompts, each prompt once, taken out of the cache so that none is dropped while the run
    /// needs it; [`Self::restore`] puts them back. `dump` goes with the first prompt, if it is encoded.
    fn take_contexts(&mut self, prompts: &[&Prompt], st: &Stream, mut dump: Option<&mut Dump>) -> Result<Vec<(Prompt, anima::Context)>> {
        let mut out: Vec<(Prompt, anima::Context)> = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            if out.iter().any(|(k, _)| k == *p) {
                continue;
            }
            self.context(p, st, if i == 0 { dump.take() } else { None })?;
            out.push(self.contexts.pop().context("prompt not encoded")?);
        }
        Ok(out)
    }

    /// Back into the cache after a run, the first most recent.
    fn restore(&mut self, contexts: Vec<(Prompt, anima::Context)>) {
        for c in contexts.into_iter().rev() {
            remember(&mut self.contexts, c, CONTEXTS);
        }
    }
}

/// Each entry's place in the contexts [`Anima::take_contexts`] gave.
fn entry_contexts<C>(conds: &[Cond], contexts: &[(Prompt, C)]) -> Result<Vec<usize>> {
    conds.iter().map(|c| contexts.iter().position(|(p, _)| *p == c.prompt).context("prompt not encoded")).collect()
}

impl Anima {
    /// SnapRef's references' keys and values, through the cache: loads the adapter and encodes the caption first.
    fn prepare_snapref(&mut self, r: &SnapRefRequest, target: (usize, usize), env: &super::Env) -> Result<()> {
        let st = &env.st;
        self.context(&r.caption, st, None)?;
        let m = self.model.as_mut().context("no diffusion model loaded")?;
        if self.snapref.as_ref().is_none_or(|a| a.path != r.adapter) {
            self.snapref = None;
            self.ref_kv = None;
            let t0 = Instant::now();
            self.snapref = Some(snapref::Adapter::load(&r.adapter, m.dit.blocks.len(), anima::WIDTH, st)?);
            eprintln!("SnapRef adapter ready in {:.1}s", t0.elapsed().as_secs_f64());
        }
        let key = ref_key(r, target, m.dit.int8_attn);
        if self.ref_kv.as_ref().is_some_and(|(k, _)| *k == key) {
            return Ok(());
        }
        self.ref_kv = None;
        let t0 = Instant::now();
        let sizes: Vec<usize> = r.references.iter().map(|x| (x.height / 2) * (x.width / 2)).collect();
        let most = sizes.iter().copied().max().unwrap_or(0);
        let nb = m.dit.blocks.len();
        let adapter = self.snapref.as_ref().unwrap();
        adapter.check_mode(&r.mode)?;
        env.plan(&mut m.dit.blocks, &m.st, anima::reference_bytes(most, adapter.max_rank.max(256)) + anima::ref_kv_bytes(nb, sizes.iter().sum()))?;
        let kv = snapref::RefKv::alloc(nb, sizes.clone(), anima::REF_ROW_BYTES)?;
        let caption = &self.contexts.iter().find(|(p, _)| *p == r.caption).context("caption not encoded")?.1;
        let mut row0 = 0;
        for (x, rows) in r.references.iter().zip(&sizes) {
            if x.latent.len() != dit::CH * x.height * x.width {
                bail!("reference latent has {} values for {}x{}", x.latent.len(), x.height, x.width);
            }
            let d = super::upload(&x.latent, st)?;
            m.dit.encode_reference(&d, x.height, x.width, target, caption, adapter, &r.mode, &kv, row0, st)?;
            row0 += rows;
        }
        st.sync()?;
        eprintln!("SnapRef: {} reference(s), {} tokens in {:.0} ms", sizes.len(), kv.rows, t0.elapsed().as_secs_f64() * 1e3);
        self.ref_kv = Some((key, kv));
        Ok(())
    }
}

impl super::Family {
    fn anima(&mut self) -> Option<&mut Anima> {
        match self {
            super::Family::Anima(f) => Some(f),
            _ => None,
        }
    }
}

impl Generator {
    /// Load an Anima checkpoint unless `path` is loaded. One model family is loaded at a time.
    pub fn ensure_anima(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Anima)?;
        let env = &self.env;
        let a = self.family.anima().context("Anima not entered")?;
        if a.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        a.clear_prompts();
        a.model = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = anima::globals_layout(&dst)?.size;
        let free = cuda::device_free()?;
        let budget = free.saturating_sub(env.reserve + globals + anima_need(env.max_width, env.max_height, 1));
        let mut dit = anima::Dit::load(&dst, budget, env.slots, env.threads, &env.st)?;
        dit.attn_variant = env.attn_variant;
        dit.int8_attn = env.int8_attn;
        dit.lossy = env.lossy;
        eprintln!("Anima ready in {:.1}s", t0.elapsed().as_secs_f64());
        a.model = Some(Model { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Load Anima's Qwen3 text encoder unless `path` is loaded; in VRAM if it fits, else streamed.
    pub fn ensure_qwen3(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Anima)?;
        let env = &self.env;
        let a = self.family.anima().context("Anima not entered")?;
        if a.te.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        env.st.sync()?;
        a.clear_prompts();
        a.te = None;
        let t0 = Instant::now();
        if a.tokenizers.is_none() {
            a.tokenizers = Some((tokenizer::Tokenizer::new()?, tokenizer::T5Tokenizer::new()?));
        }
        let size = qwen3::weight_bytes(&st::SafeTensors::open(path)?)?;
        let keep = env.reserve + anima_need(env.max_width, env.max_height, 1);
        let budget = if cuda::device_free()? >= size + keep { size } else { 0 };
        a.te = Some((path.to_path_buf(), qwen3::TextEncoder::load(path, budget, env.threads, &env.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Record the loaded Anima model's block outputs over `steps` steps from now on (see [`crate::steer`]).
    pub fn record_anima(&mut self, steps: usize) -> Result<()> {
        let st = &self.env.st;
        let a = self.family.anima().context("Anima not entered")?;
        let dit = &mut a.model.as_mut().context("no diffusion model loaded")?.dit;
        dit.recorder = Some(crate::steer::Recorder::new(dit.blocks.len(), anima::WIDTH, steps, st)?);
        Ok(())
    }

    /// Keep every `stride`-th token of `blocks` at `steps` from now on (see [`crate::steer::TokenSampler`]).
    pub fn sample_anima_tokens(&mut self, blocks: Vec<usize>, steps: (usize, usize), stride: usize) -> Result<()> {
        let a = self.family.anima().context("Anima not entered")?;
        let dit = &mut a.model.as_mut().context("no diffusion model loaded")?.dit;
        dit.sampler = Some(crate::steer::TokenSampler::new(anima::WIDTH, blocks, steps, stride));
        Ok(())
    }

    /// Write what [`Self::sample_anima_tokens`] kept, and stop.
    pub fn save_anima_tokens(&mut self, path: &Path) -> Result<()> {
        let a = self.family.anima().context("Anima not entered")?;
        let dit = &mut a.model.as_mut().context("no diffusion model loaded")?.dit;
        dit.sampler.take().context("no tokens kept")?.save(path)
    }

    /// Steer the loaded Anima model's cross-attention towards `concept` and away from `null` from now on (see
    /// [`anima::Concept`]); `blocks` defaults to every block.
    pub fn concept_anima(&mut self, concept: &Prompt, null: &Prompt, alpha: f32, steps: (usize, usize), blocks: Option<(usize, usize)>)
        -> Result<()> {
        let st = &self.env.st;
        let a = self.family.anima().context("Anima not entered")?;
        let mut take = |p: &Prompt| -> Result<anima::Context> {
            a.context(p, st, None)?;
            let i = a.contexts.iter().position(|(k, _)| k == p).context("prompt not encoded")?;
            Ok(a.contexts.remove(i).1)
        };
        let (concept, null) = (take(concept)?, take(null)?);
        let dit = &mut a.model.as_mut().context("no diffusion model loaded")?.dit;
        let blocks = blocks.unwrap_or((0, dit.blocks.len()));
        dit.concept = Some(anima::Concept { concept, null, alpha, steps, blocks });
        Ok(())
    }

    /// Write what [`Self::record_anima`] recorded, and stop recording.
    pub fn save_anima_record(&mut self, path: &Path) -> Result<()> {
        let st = &self.env.st;
        let a = self.family.anima().context("Anima not entered")?;
        let dit = &mut a.model.as_mut().context("no diffusion model loaded")?.dit;
        dit.recorder.take().context("nothing recorded")?.save(path, st)
    }

    /// Also steer the loaded Anima model with the vector or map file at `path` from now on; `blocks` defaults to every block.
    pub fn steer_anima(&mut self, path: &Path, alpha: f32, blocks: Option<(usize, usize)>, steps: (usize, usize), preserve_norm: bool)
        -> Result<()> {
        let st = &self.env.st;
        let a = self.family.anima().context("Anima not entered")?;
        let dit = &mut a.model.as_mut().context("no diffusion model loaded")?.dit;
        let nb = dit.blocks.len();
        dit.steer.push(crate::steer::Steer::load(path, alpha, blocks.unwrap_or((0, nb)), steps, preserve_norm, nb, anima::WIDTH, st)?);
        Ok(())
    }

    /// Use (or with `None`, stop using) a training-free style reference for every Anima request from now on.
    pub fn style_anima(&mut self, style: Option<StyleRequest>) -> Result<()> {
        self.family.anima().context("Anima not entered")?.style = style;
        Ok(())
    }

    /// Anima latents, sampled as ComfyUI's KSampler does.
    pub fn sample_anima(&mut self, r: &AnimaRequest, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        env.check_size(r.width, r.height)?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let images = r.sampling.images;
        let n = images * dit::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let a = self.family.anima().context("no diffusion model loaded")?;
        let st = &env.st;
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let guided = r.sampling.uses_negative();
        if r.positive.is_empty() || (guided && r.negative.is_empty()) {
            bail!("a prompt without conditioning entries");
        }
        if r.positive.len() > 1 && a.model.as_ref().is_some_and(|m| m.dit.recorder.is_some() || m.dit.sampler.is_some()) {
            bail!("recording block outputs takes one positive conditioning entry");
        }
        if let Some(sr) = &r.snapref {
            if sr.references.is_empty() || sr.references.len() != sr.biases.len() {
                bail!("SnapRef needs one bias per reference, and at least one reference");
            }
            a.prepare_snapref(sr, (lh, lw), env)?;
            if let (Some(d), Some((_, kv))) = (dump.as_deref_mut(), &a.ref_kv) {
                let x = &sr.references[0];
                d.insert("ref_latent".into(), ("F32".into(), vec![dit::CH, x.height, x.width], st::f32_bytes(&x.latent)));
                let last = a.model.as_ref().map_or(0, |m| m.dit.blocks.len() - 1);
                for (name, b) in [("ref_kv_block0", 0), ("ref_kv_last", last)] {
                    let mut bytes = vec![0u8; kv.rows * kv.row_bytes];
                    kv.kv.download(b * kv.rows * kv.row_bytes, &mut bytes, st)?;
                    d.insert(name.into(), ("BF16".into(), vec![kv.rows, 2, anima::WIDTH], bytes));
                }
            }
        }
        // SnapRef's extra activations per image: its LoRA scratch and the reference keys beside the image's
        let snap_need = match (&r.snapref, &a.ref_kv, &a.snapref) {
            (Some(sr), Some((_, kv)), Some(adapter)) => {
                let groups = snapref::RefRead { biases: sr.biases.clone() }.groups(&kv.sizes).len();
                anima::snapref_step_bytes((lh / 2) * (lw / 2), kv.rows, groups, adapter.max_rank.max(128))
            }
            _ => 0,
        };
        // a style reference: its own pass and keys, and the keys beside each image's
        let mut style = r.style.clone().or_else(|| a.style.clone());
        if let Some(s) = style.as_mut().filter(|s| s.shuffle > 0) {
            let (rh, rw) = (s.reference.height, s.reference.width);
            s.reference.latent = shuffle_tiles(&s.reference.latent, dit::CH, rh, rw, s.shuffle, s.seed);
        }
        let style_rows = style.as_ref().map_or(0, |s| (s.reference.height / 2) * (s.reference.width / 2));
        // shared attention: the reference's keys beside each image's; AdaIN alone keeps no keys
        let style_attention = style.as_ref().is_some_and(|s| s.attention);
        let snap_need = snap_need + if style_attention { anima::snapref_step_bytes((lh / 2) * (lw / 2), style_rows, 1, 128) } else { 0 };
        if style.is_some() && r.snapref.is_some() {
            bail!("a style reference and SnapRef can't be used together");
        }
        // every entry's context, out of the cache for the whole run, before residency is planned around them
        let mut prompts: Vec<&Prompt> = r.positive.iter().map(|c| &c.prompt).collect();
        if guided {
            prompts.extend(r.negative.iter().map(|c| &c.prompt));
        }
        if let Some(s) = &style {
            prompts.push(&s.caption);
        }
        let contexts = a.take_contexts(&prompts, st, dump.as_deref_mut())?;
        let together = {
            let m = a.model.as_mut().context("no diffusion model loaded")?;
            let style_fixed = if style.is_some() {
                anima::reference_bytes(style_rows, 256) + if style_attention { anima::ref_kv_bytes(m.dit.blocks.len(), style_rows) } else { 0 }
            } else {
                0
            };
            let together = env.batch_fit(&m.dit.blocks, images, style_fixed, step_need(r.width, r.height) + snap_need)?;
            env.plan(&mut m.dit.blocks, &m.st, anima_need(r.width, r.height, together) + together * snap_need + style_fixed)?;
            together
        };
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        let m = a.model.as_mut().unwrap();
        // SnapRef: the positive reads the references; the negative too with ref_in_uncond, or weaker with Fidelity
        let reads = r.snapref.as_ref().map(|sr| {
            let neg = if sr.in_uncond {
                Some(sr.biases.clone())
            } else {
                sr.uncond_bias.map(|u| sr.biases.iter().map(|b| b + u).collect())
            };
            (snapref::RefRead { biases: sr.biases.clone() }, neg.map(|biases| snapref::RefRead { biases }))
        });
        let (pos_pass, neg_pass) = match (&reads, a.snapref.as_ref(), a.ref_kv.as_ref()) {
            (Some((pr, nr)), Some(adapter), Some((_, kv))) => {
                let mode = r.snapref.as_ref().map_or("cs", |sr| sr.mode.as_str());
                (
                    anima::Pass::Target { adapter: Some((adapter, mode)), read: Some((kv, pr)) },
                    anima::Pass::Target { adapter: Some((adapter, mode)), read: nr.as_ref().map(|n| (kv, n)) },
                )
            }
            (Some(_), _, _) => bail!("SnapRef's references weren't encoded"),
            _ => (anima::Pass::Plain, anima::Pass::Plain),
        };
        let pos_at = entry_contexts(&r.positive, &contexts)?;
        let neg_at = if guided { entry_contexts(&r.negative, &contexts)? } else { Vec::new() };
        let ctx = |at: &[usize], entry: usize| at.get(entry).map(|&i| &contexts[i].1).context("no such conditioning entry");
        let style_ctx = style.as_ref().map(|s| contexts.iter().find(|(p, _)| *p == s.caption).map(|(_, c)| c).context("caption not encoded"))
            .transpose()?;
        let style_buf = match &style {
            Some(s) => {
                let (rh, rw) = (s.reference.height, s.reference.width);
                if s.reference.latent.len() != dit::CH * rh * rw {
                    bail!("style reference latent has {} values for {rh}x{rw}", s.reference.latent.len());
                }
                let kv = s.attention.then(|| snapref::RefKv::alloc(m.dit.blocks.len(), vec![style_rows], anima::REF_ROW_BYTES)).transpose()?;
                if s.shuffle % 2 != 0 {
                    bail!("the style reference's shuffle tile must be even");
                }
                let noise = crate::sampling::torch_randn(s.seed, s.reference.latent.len());
                let xref = DevBuf::alloc(s.reference.latent.len() * 4)?;
                Some((kv, noise, xref, snapref::RefRead { biases: vec![s.bias] }))
            }
            None => None,
        };
        m.dit.ref_blocks = style.as_ref().map_or((0, usize::MAX), |s| s.blocks);
        m.dit.adain = match &style {
            Some(s) if s.adain != 0.0 => Some(anima::AdaIn {
                stats: DevBuf::alloc(m.dit.blocks.len() * 2 * anima::WIDTH * 4)?,
                rows: style_rows,
                lambda: s.adain,
                blocks: s.adain_blocks,
                capture: true,
                active: false,
            }),
            _ => None,
        };
        let rope = m.dit.rope(lh / 2, lw / 2, st)?;
        let copied0 = m.dit.blocks.copied_bytes;
        let sigmas = r.sampling.sigmas.clone();
        // the sigma the style reference last ran at: it depends on nothing else, so a side's other entries reuse it
        let mut style_at: Option<u32> = None;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, side: Side, eval: usize, k: usize| -> Result<()> {
            let negative = side.negative;
            // the step whose sigma this is, for activation steering; a solver's intermediate sigma takes the step it is in
            m.dit.step = sigmas.windows(2).position(|w| sigma <= w[0] && sigma > w[1]);
            m.dit.negative = negative;
            if negative {
                // the negative's first block, for SnapRef's Fidelity
                let mut first: Option<Vec<u8>> = None;
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && name == "block0" {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        first = Some(v);
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.predict(x, k, lh, lw, ctx(&neg_at, side.entry)?, &rope, sigma, out, st, &neg_pass, probe_ref)?;
                if let (Some(dm), Some(v)) = (dump.as_deref_mut(), first) {
                    dm.insert("neg_block0".into(), ("BF16".into(), vec![v.len() / 2], v));
                }
                return Ok(());
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, k * step_need(r.width, r.height))?;
            }
            // the style reference, noised to this sigma, through the plain model; its keys are read in this evaluation
            if let Some(ad) = m.dit.adain.as_mut() {
                ad.active = false;
            }
            let pos_pass = match (&style, &style_buf, m.dit.step) {
                (Some(s), Some((kv, noise, xref, read)), Some(step)) if (s.steps.0..s.steps.1).contains(&step) => {
                    if style_at != Some(sigma.to_bits()) {
                        let mix: Vec<f32> = s.reference.latent.iter().zip(noise).map(|(x0, e)| (1.0 - sigma) * x0 + sigma * e).collect();
                        xref.upload(0, &st::f32_bytes(&mix), st)?;
                        let (rh, rw) = (s.reference.height, s.reference.width);
                        m.dit.encode_style_reference(xref, rh, rw, (lh, lw), sigma, s.frame, style_ctx.unwrap(), kv.as_ref(), st)?;
                        style_at = Some(sigma.to_bits());
                    }
                    if let Some(ad) = m.dit.adain.as_mut() {
                        ad.active = true;
                    }
                    match kv {
                        Some(kv) => anima::Pass::Target { adapter: None, read: Some((kv, read)) },
                        None => pos_pass,
                    }
                }
                _ => pos_pass,
            };
            let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
            {
                let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                    if eval == 0 && matches!(name, "x_embedder_out" | "block0" | "block27" | "final_tokens" | "block0_self_attn_raw") {
                        let mut v = vec![0u8; bytes];
                        b.download(0, &mut v, st).unwrap();
                        probes.push((name.to_string(), v));
                    }
                };
                let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.predict(x, k, lh, lw, ctx(&pos_at, side.entry)?, &rope, sigma, out, st, &pos_pass, probe_ref)?;
            }
            if let Some(dm) = dump.as_deref_mut() {
                for (name, v) in probes {
                    dm.insert(name, ("BF16".into(), vec![v.len() / 2], v));
                }
            }
            Ok(())
        };
        let neg_regions = if guided { regions(&r.negative) } else { PLAIN.to_vec() };
        let latent = run_batches(&r.sampling, &r.noise, [&regions(&r.positive), &neg_regions], together, st, &mut stats, &mut model, &mut progress);
        m.dit.step = None;
        m.dit.ref_blocks = (0, usize::MAX);
        m.dit.adain = None;
        a.restore(contexts);
        let latent = latent?;
        let m = a.model.as_mut().unwrap();
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(dm) = dump {
            dm.insert("samples".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(chunks: &[Vec<(String, f32)>]) -> (Vec<Vec<u32>>, Vec<u32>, Vec<f32>) {
        chunk_tokens(&tokenizer::Tokenizer::new().unwrap(), &tokenizer::T5Tokenizer::new().unwrap(), chunks).unwrap()
    }

    fn seg(text: &str, weight: f32) -> (String, f32) {
        (text.to_string(), weight)
    }

    // ids from SwarmTextEncodeAdvanced's combine_leaves on ComfyUI's AnimaTokenizer, chunk by chunk
    #[test]
    fn breaks_join_qwen_chunks_and_keep_the_first_t5() {
        // "lake<break>pine forest"
        let (q, t, w) = tokens(&[vec![seg("lake", 1.0)], vec![seg("pine forest", 1.0)]]);
        assert_eq!(q, [vec![62110], vec![38038, 13638]]);
        assert_eq!((t, w), (vec![6957, 1], vec![1.0, 1.0]));
        // "<break>b" and "a<break>": an empty chunk is one Qwen3 pad token, and only T5's end token
        let (q, t, _) = tokens(&[vec![], vec![seg("b", 1.0)]]);
        assert_eq!((q, t), (vec![vec![151643], vec![65]], vec![1]));
        let (q, t, _) = tokens(&[vec![seg("a", 1.0)], vec![]]);
        assert_eq!((q, t), (vec![vec![64], vec![151643]], vec![3, 9, 1]));
    }

    #[test]
    fn weights_stay_on_the_first_chunks_t5_tokens() {
        // "a <weight[1.5]:red> boat<break>sky"
        let (q, t, w) = tokens(&[vec![seg("a ", 1.0), seg("red", 1.5), seg(" boat", 1.0)], vec![seg("sky", 1.0)]]);
        assert_eq!(q[0], [64, 220, 1151, 15328]);
        assert_eq!(t, [3, 9, 1131, 3432, 1]);
        assert_eq!(w, [1.0, 1.0, 1.5, 1.0, 1.0]);
        assert!(chunk_tokens(&tokenizer::Tokenizer::new().unwrap(), &tokenizer::T5Tokenizer::new().unwrap(), &[]).is_err());
    }

    #[test]
    fn entries_find_their_deduplicated_contexts() {
        let p = |t: &str| Prompt::text(t);
        let conds = [Cond::plain(p("a")), Cond::plain(p("b")), Cond::plain(p("a"))];
        // as take_contexts gives them: each prompt once, in first-use order
        let contexts = [(p("a"), ()), (p("b"), ())];
        assert_eq!(entry_contexts(&conds, &contexts).unwrap(), [0, 1, 0]);
        // ConditioningMultiply's scale is part of the prompt's context
        let scaled = Cond::plain(Prompt { scale: 0.5, ..p("a") });
        assert!(entry_contexts(&[scaled], &contexts).is_err());
    }
}
