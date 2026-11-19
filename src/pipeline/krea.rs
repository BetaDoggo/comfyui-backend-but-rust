//! Krea 2: its diffusion model and encoded prompts. The text encoder and the VAE are the generator's.

use super::{remember, run_batches, touch, Cond, Dump, Env, FamilyState, Generator, Kind, Prompt, Reference, Region, Sampling, SharedTe, Side, Stats,
    Step, CONTEXTS, MB, PLAIN};
use super::{nothing_for, shared_te_loras, upload, StyleRequest};
use crate::cuda::{self, DevBuf, Stream};
use crate::vision::{self, PromptImage};
use crate::{dit, lora, offload, sampling, st, te, tokenizer, vae};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Text tokens budgeted for activation planning.
const LT_MAX: usize = 512;

/// One conditioning entry of a Krea 2 prompt: its prompt and where it applies, and what its text encode and
/// `ReferenceLatent` gave it.
#[derive(Clone)]
pub struct KreaEntry {
    pub cond: Cond,
    /// SwarmTextEncodeAdvanced's prompt: each segment is a prompt leaf, tokenized apart and weighted as the node does
    /// (see [`te::krea_tokens`]). Otherwise each chunk's text is tokenized whole, as CLIPTextEncode.
    pub leaves: bool,
    /// Prompt images, through the vision tower into the text encoder.
    pub images: Vec<PromptImage>,
    /// Reference latents, see [`Generator::encode_image`].
    pub references: Vec<Reference>,
    /// ComfyUI's `index_timestep_zero` rather than `index`.
    pub timestep_zero: bool,
}

impl KreaEntry {
    /// A prompt everywhere for the whole run, tokenized whole, without reference latents.
    pub fn new(prompt: Prompt, images: Vec<PromptImage>) -> Self {
        KreaEntry { cond: Cond::plain(prompt), leaves: false, images, references: Vec::new(), timestep_zero: false }
    }

    fn key(&self) -> PromptKey {
        PromptKey { prompt: self.cond.prompt.clone(), leaves: self.leaves, images: self.images.clone() }
    }
}

pub struct KreaRequest {
    /// The positive prompt's entries, mixed as ComfyUI's `calc_cond_batch` mixes them.
    pub positive: Vec<KreaEntry>,
    /// Only encoded when the sampling uses it.
    pub negative: Vec<KreaEntry>,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise one after another, instead of ComfyUI's for the seed and the whole batch.
    pub noise: Option<Vec<f32>>,
    /// `SwarmAttnTokenWeights`: the entry (its prompt, tokenization and images) whose first chunk's `attn_token_weights`
    /// pairs change the attention of every positive evaluation.
    pub attn_token_weights: Option<KreaEntry>,
    /// A training-free style reference (the `BrokeAnimaStyleReference` node): residual AdaIN only, so its `attention`,
    /// `shuffle`, `bias`, `blocks` and `frame` aren't read.
    pub style: Option<StyleRequest>,
}

/// A prompt as the text encoder saw it.
#[derive(Clone, PartialEq)]
struct PromptKey {
    prompt: Prompt,
    leaves: bool,
    images: Vec<PromptImage>,
}

/// An encoded prompt: its conditioning and the DiT's context from it.
struct Encoded {
    key: PromptKey,
    cond: te::Cond,
    ctx: DevBuf,
}

struct Model {
    path: PathBuf,
    dit: dit::Dit,
    st: st::SafeTensors,
    lora: lora::State,
}

#[derive(Default)]
pub(super) struct Krea {
    model: Option<Model>,
    /// Latest encoded prompts, most recent last. A request keeps every one it needs until it ends.
    contexts: Vec<Encoded>,
}

impl FamilyState for Krea {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        if lora::apply(lora::Arch::Krea2, &m.st, "", m.dit.stores(), &mut m.lora, loras, st)? {
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

/// One image's activations in a denoising step. `extra` is the tokens of its prompt images and reference latents.
fn step_need(width: usize, height: usize, extra: usize) -> usize {
    dit::step_activation_bytes(LT_MAX + extra + (height / 16) * (width / 16))
}

/// Peak activation memory of a request of `images` sampled together (denoising step or VAE decode, whichever is larger).
fn activation_need(width: usize, height: usize, extra: usize, images: usize) -> usize {
    (images * step_need(width, height, extra)).max(vae::Vae::activation_bytes(height / 8, width / 8))
}

/// What SwarmTextEncodeAdvanced does with one chunk's weights on a tokenizer without any (`use_attn_token_weights`).
#[derive(Debug, PartialEq)]
struct ChunkWeights {
    /// Conditioning rows multiplied by their token's weight (`multiply_cond_by_token_weights`).
    rows: Vec<(usize, f32)>,
    /// Every row's factor: the one weight of every leaf, when no row took it.
    whole: f32,
    /// `attn_token_weights`, when the leaves' weights differ.
    pairs: Vec<(usize, f32)>,
}

/// `uniform_weight`: the weight every segment with text other than whitespace has, if they all have one.
fn uniform_weight(segments: &[(String, f32)]) -> Option<f32> {
    let mut weights = segments.iter().filter(|(t, _)| !t.trim().is_empty()).map(|(_, w)| *w);
    let first = weights.next()?;
    weights.all(|w| w == first).then_some(first)
}

/// [`ChunkWeights`] of a chunk's `segments`, tokenized to `factors` (one per token), with a conditioning of `cond_len`
/// rows: as `token_weights_from_tokens`, the token at `i` weights row `cond_len - len(tokens) + i`.
fn chunk_weights(segments: &[(String, f32)], factors: &[f32], cond_len: usize) -> ChunkWeights {
    if segments.iter().all(|(_, w)| *w == 1.0) {
        return ChunkWeights { rows: Vec::new(), whole: 1.0, pairs: Vec::new() };
    }
    let offset = cond_len as i64 - factors.len() as i64;
    let rows: Vec<(usize, f32)> = factors.iter().enumerate().filter(|(_, f)| **f != 1.0).map(|(i, f)| (offset + i as i64, *f))
        .filter(|&(p, _)| p >= 0 && (p as usize) < cond_len).map(|(p, f)| (p as usize, f)).collect();
    let uniform = uniform_weight(segments);
    let whole = match uniform {
        Some(w) if rows.is_empty() => w,
        _ => 1.0,
    };
    let pairs = if uniform.is_none() { rows.clone() } else { Vec::new() };
    ChunkWeights { rows, whole, pairs }
}

/// A prompt chunk ready for the text encoder.
struct Chunk {
    ids: Vec<u32>,
    weights: ChunkWeights,
}

/// The chunks of `prompt` with prompt images of `image_tokens` tokens each.
fn krea_chunks(tok: &tokenizer::Tokenizer, prompt: &Prompt, leaves: bool, image_tokens: &[usize]) -> Result<Vec<Chunk>> {
    prompt.chunks.iter().map(|segments| {
        let (ids, factors) = te::krea_tokens(tok, segments, leaves, image_tokens.len())?;
        // an image grows the sequence from its <|image_pad|>
        let pads = ids.iter().filter(|&&t| t == te::IMAGE_PAD).count();
        let grown: usize = image_tokens.iter().take(pads).map(|t| t.saturating_sub(1)).sum();
        let cond_len = ids.len() + grown - te::template_end(&ids);
        Ok(Chunk { weights: chunk_weights(segments, &factors, cond_len), ids })
    }).collect()
}

/// Encode `key` into `contexts` unless it is there, and make it the most recent. `embeds` are its prompt images'.
fn ensure_context(contexts: &mut Vec<Encoded>, dit: &mut dit::Dit, te: &mut te::TextEncoder, key: &PromptKey, embeds: &[&vision::Embeds],
    st: &Stream) -> Result<()> {
    if touch(contexts, |c| c.key == *key) {
        return Ok(());
    }
    let t0 = Instant::now();
    let tokens: Vec<usize> = embeds.iter().map(|e| e.tokens).collect();
    let chunks = krea_chunks(te.tokenizer(), &key.prompt, key.leaves, &tokens)?;
    let conds = chunks.iter().map(|c| te.encode_krea(&c.ids, embeds, &c.weights.rows, c.weights.whole, key.prompt.scale, st))
        .collect::<Result<Vec<_>>>()?;
    let cond = te::Cond::concat(conds, st)?;
    let ctx = dit.encode_context(&cond.ctx, cond.len, st)?;
    st.sync()?;
    eprintln!("text: {} tokens ({} conditioning) in {:.0} ms", cond.token_ids.len(), cond.len, t0.elapsed().as_secs_f64() * 1e3);
    // kept past CONTEXTS until the request ends
    remember(contexts, Encoded { key: key.clone(), cond, ctx }, usize::MAX);
    Ok(())
}

/// Reference latents with the same pictures.
fn same_references(a: &[Reference], b: &[Reference]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.height == b.height && a.width == b.width && a.latent == b.latent)
}

/// What one entry's evaluations read: its context and conditioning length, its reference tokens and its RoPE table,
/// as indexes into what the entries share.
#[derive(Clone, Copy)]
struct Run {
    ctx: usize,
    lt: usize,
    refs: Option<usize>,
    rope: usize,
}

impl super::Family {
    fn krea(&mut self) -> Option<&mut Krea> {
        match self {
            super::Family::Krea(f) => Some(f),
            _ => None,
        }
    }
}

impl Generator {
    /// Start the GPU runtime with the Krea 2 files of `o` loaded, for the CLI and the server.
    pub fn load_krea(o: &super::LoadOptions) -> Result<Self> {
        let t_start = Instant::now();
        let mut gen = Self::new(o)?;
        gen.ensure_vae(&o.vae)?;
        gen.ensure_te(&o.te)?;
        gen.ensure_krea(&o.dit)?;
        eprintln!("startup total {:.1}s", t_start.elapsed().as_secs_f64());
        Ok(gen)
    }

    /// Load a Krea 2 diffusion model unless `path` is already loaded, planning block residency from what is free.
    pub fn ensure_krea(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Krea)?;
        let env = &self.env;
        let k = self.family.krea().context("Krea 2 not entered")?;
        if k.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        k.clear_prompts();
        k.model = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let globals = dit::globals_layout(&dst)?.size;
        let txt_slots = 2 * dit::txt_layouts(&dst)?.iter().map(|l| l.size).max().unwrap_or(0);
        let te_slots = 2 * 110 * MB;
        let act = activation_need(env.max_width, env.max_height, 0, 1) + txt_slots + te_slots;
        let free = cuda::device_free()?;
        let mut budget = free.saturating_sub(env.reserve + globals + act);
        if let Some(r) = env.resident_blocks {
            let one = dit::block_layout(&dst, 0)?.size;
            budget = if r >= dit::n_blocks(&dst) { r * one } else { (r + env.slots) * one };
        }
        let mut dit = dit::Dit::load(&dst, budget, env.slots, env.threads, &env.st)?;
        dit.attn_variant = env.attn_variant;
        dit.int8_attn = env.int8_attn;
        eprintln!("DiT ready in {:.1}s (activation reserve {} MB)", t0.elapsed().as_secs_f64(), act / MB);
        k.model = Some(Model { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Generate Krea 2 images; returns their RGB8 pixels one after another. `progress(step, steps)` is called after each
    /// step and stops the run with [`super::Cancelled`] by returning false.
    pub fn generate_krea(&mut self, r: &KreaRequest, progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<u8>, Stats)> {
        let t_img = Instant::now();
        let (latent, mut stats) = self.sample_krea(r, progress, dump.as_deref_mut())?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let t0 = Instant::now();
        let mut rgb = Vec::new();
        for image in latent.chunks_exact(dit::CH * lh * lw) {
            rgb.extend(self.decode_latent(image, r.width, r.height)?);
        }
        stats.vae_ms = t0.elapsed().as_secs_f64() * 1e3;
        stats.total_ms = t_img.elapsed().as_secs_f64() * 1e3;
        eprintln!("VAE decode {:.0} ms, image done in {:.2}s", stats.vae_ms, stats.total_ms / 1e3);
        if let Some(d) = dump {
            d.insert("image_rgb8".into(), ("U8".into(), vec![r.sampling.images * lh * 8, lw * 8, 3], rgb.clone()));
        }
        Ok((rgb, stats))
    }

    /// Krea 2: text encoding and denoising only; returns the final latents ([images, 16, H/8, W/8] f32) on the host.
    pub fn sample_krea(&mut self, r: &KreaRequest, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        let env: &Env = &self.env;
        env.check_size(r.width, r.height)?;
        let guided = r.sampling.uses_negative();
        let negative: &[KreaEntry] = if guided { &r.negative } else { &[] };
        if r.positive.is_empty() || (guided && negative.is_empty()) {
            bail!("a prompt without conditioning entries");
        }
        let k = self.family.krea().context("no diffusion model loaded")?;
        let Krea { model, contexts } = k;
        let m = model.as_mut().context("no diffusion model loaded")?;
        let te = self.te.as_mut().context("no text encoder loaded")?;
        // what an earlier request that failed left
        while contexts.len() > CONTEXTS {
            contexts.remove(0);
        }
        let entries: Vec<&KreaEntry> = r.positive.iter().chain(negative).collect();
        let t0 = Instant::now();
        let mut vision = Vec::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            vision.push(te.vision_embeds(&e.images, env, if i == 0 { dump.as_deref_mut() } else { None })?);
        }
        // SwarmAttnTokenWeights reads the pairs of its conditioning's first entry, which come from its first chunk
        let pairs = match &r.attn_token_weights {
            Some(e) => {
                let tokens: Vec<usize> = te.vision_embeds(&e.images, env, None)?.iter().map(|v| v.tokens).collect();
                let mut chunks = krea_chunks(te.enc.tokenizer(), &e.cond.prompt, e.leaves, &tokens)?;
                if chunks.is_empty() { Vec::new() } else { chunks.swap_remove(0).weights.pairs }
            }
            None => Vec::new(),
        };
        if !m.dit.int8_attn && dit::bias_values(&pairs).len() > dit::BIAS_GROUPS {
            return Err(lora::Unsupported(format!("{} different prompt weights above 1 with FlashAttention (at most {})",
                dit::bias_values(&pairs).len(), dit::BIAS_GROUPS)).into());
        }
        let vision_ms = t0.elapsed().as_secs_f64() * 1e3;
        let grids = |e: &KreaEntry| -> Vec<(usize, usize)> {
            e.references.iter().map(|l| (l.height.div_ceil(dit::PATCH), l.width.div_ceil(dit::PATCH))).collect()
        };
        let extra = entries.iter().zip(&vision)
            .map(|(e, v)| v.iter().map(|x| x.tokens).sum::<usize>() + grids(e).iter().map(|(h, w)| h * w).sum::<usize>())
            .max().unwrap_or(0);
        let images = r.sampling.images;
        m.dit.adain = None;
        // a style reference: its own pass before each positive evaluation, as one image of its own size
        let style = r.style.as_ref().filter(|s| s.adain != 0.0);
        if let Some(s) = style {
            if s.attention || s.shuffle != 0 {
                bail!("Krea 2's style reference takes residual AdaIN only (no shared attention or shuffle)");
            }
            let (rh, rw) = (s.reference.height, s.reference.width);
            if s.reference.latent.len() != dit::CH * rh * rw {
                bail!("style reference latent has {} values for {rh}x{rw}", s.reference.latent.len());
            }
            if rh % dit::PATCH != 0 || rw % dit::PATCH != 0 {
                bail!("style reference latent size must be even");
            }
        }
        let int8 = m.dit.int8_attn;
        let style_rows = style.map_or(0, |s| (s.reference.height / dit::PATCH) * (s.reference.width / dit::PATCH));
        let style_fixed = style.map_or(0, |s| 2 * s.reference.latent.len() * 4 + m.dit.blocks.len() * 2 * dit::F * 4);
        let style_need = style.map_or(0, |_| {
            dit::step_activation_bytes(LT_MAX + style_rows) + dit::token_weight_bytes(&pairs, int8, LT_MAX + style_rows)
        });
        let weights_need = dit::token_weight_bytes(&pairs, int8, LT_MAX + extra + (r.height / 16) * (r.width / 16));
        let together = env.batch_fit(&m.dit.blocks, images, style_fixed, step_need(r.width, r.height, extra) + weights_need)?;
        env.plan(&mut m.dit.blocks, &m.st, (activation_need(r.width, r.height, extra, together) + together * weights_need).max(style_need)
            + style_fixed)?;
        let st = &env.st;
        let mut stats = Stats::default();
        let (lh, lw) = (r.height / 8, r.width / 8);

        // text encoding: every entry's, and the style reference's caption, reused while they stay in the cache
        let t0 = Instant::now();
        for (e, v) in entries.iter().zip(&vision) {
            let embeds: Vec<&vision::Embeds> = v.iter().map(|e| &**e).collect();
            ensure_context(contexts, &mut m.dit, &mut te.enc, &e.key(), &embeds, st)?;
        }
        let caption = style.map(|s| PromptKey { prompt: s.caption.clone(), leaves: false, images: Vec::new() });
        if let Some(key) = &caption {
            ensure_context(contexts, &mut m.dit, &mut te.enc, key, &[], st)?;
        }
        stats.text_ms = vision_ms + t0.elapsed().as_secs_f64() * 1e3;

        let result = {
            let ctxs: &[Encoded] = contexts;
            let find = |key: &PromptKey| ctxs.iter().position(|c| c.key == *key).context("an encoded prompt left the cache");
            // each entry's reference tokens and RoPE table, shared between entries that have the same
            let mut refs: Vec<(&[Reference], bool, dit::References)> = Vec::new();
            let mut ropes: Vec<((usize, Vec<(usize, usize)>), DevBuf)> = Vec::new();
            let mut runs = Vec::with_capacity(entries.len());
            for e in &entries {
                let ctx = find(&e.key())?;
                let lt = ctxs[ctx].cond.len;
                let at = if e.references.is_empty() {
                    None
                } else if let Some(i) = refs.iter().position(|(r, tz, _)| *tz == e.timestep_zero && same_references(r, &e.references)) {
                    Some(i)
                } else {
                    let latents: Vec<(Vec<f32>, usize, usize)> = e.references.iter().map(|l| (l.latent.clone(), l.height, l.width)).collect();
                    refs.push((&e.references, e.timestep_zero, m.dit.references(&latents, e.timestep_zero, st)?));
                    Some(refs.len() - 1)
                };
                let g = grids(e);
                let rope = match ropes.iter().position(|((l, gr), _)| *l == lt && *gr == g) {
                    Some(i) => i,
                    None => {
                        ropes.push(((lt, g.clone()), upload(&dit::Dit::rope_table(lt, lh / 2, lw / 2, &g), st)?));
                        ropes.len() - 1
                    }
                };
                runs.push(Run { ctx, lt, refs: at, rope });
            }
            let (pos_runs, neg_runs) = runs.split_at(r.positive.len());
            if let Some(d) = dump.as_deref_mut() {
                let Encoded { cond, ctx, .. } = &ctxs[pos_runs[0].ctx];
                let mut b = vec![0u8; cond.ctx.size()];
                cond.ctx.download(0, &mut b, st)?;
                d.insert("cond".into(), ("BF16".into(), vec![cond.len, dit::TXT_LAYERS * dit::TXT_D], b));
                let mut b = vec![0u8; ctx.size()];
                ctx.download(0, &mut b, st)?;
                d.insert("txtmlp_out".into(), ("BF16".into(), vec![cond.len, dit::F], b));
                let ids: Vec<u8> = cond.token_ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect();
                d.insert("token_ids".into(), ("I64".into(), vec![cond.token_ids.len()], ids));
            }

            // the style reference's caption, its RoPE grid (its own size), its fixed noise and buffers
            let style_pass = match (style, &caption) {
                (Some(s), Some(key)) => {
                    let c = &ctxs[find(key)?];
                    let (rh, rw) = (s.reference.height, s.reference.width);
                    let rope = upload(&dit::Dit::rope_table(c.cond.len, rh / dit::PATCH, rw / dit::PATCH, &[]), st)?;
                    let noise = sampling::torch_randn(s.seed, s.reference.latent.len());
                    let xref = DevBuf::alloc(s.reference.latent.len() * 4)?;
                    let xout = DevBuf::alloc(s.reference.latent.len() * 4)?;
                    m.dit.adain = Some(crate::anima::AdaIn {
                        stats: DevBuf::alloc(m.dit.blocks.len() * 2 * dit::F * 4)?,
                        rows: style_rows,
                        lambda: s.adain,
                        blocks: s.adain_blocks,
                        capture: false,
                        active: false,
                    });
                    Some((s, &c.ctx, c.cond.len, rope, noise, xref, xout))
                }
                _ => None,
            };

            // noise (same as ComfyUI's KSampler for the seed)
            let n = images * dit::CH * lh * lw;
            let noise = match &r.noise {
                Some(v) => v.clone(),
                None => sampling::torch_randn(r.seed, n),
            };
            if noise.len() != n {
                bail!("noise has {} values, expected {n}", noise.len());
            }

            // denoise
            let copied0 = m.dit.blocks.copied_bytes;
            let sigmas = r.sampling.sigmas.clone();
            // the sigma whose style statistics the AdaIN buffer holds: every positive entry at a sigma shares them
            let mut captured: Option<f32> = None;
            let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, side: Side, eval: usize, k: usize| -> Result<()> {
                let run = if side.negative { neg_runs[side.entry] } else { pos_runs[side.entry] };
                let (ctx, refs, rope) = (&ctxs[run.ctx].ctx, run.refs.map(|i| &refs[i].2), &ropes[run.rope].1);
                if side.negative {
                    return m.dit.step(x, k, lh, lw, ctx, run.lt, refs, rope, sigma, sigma, st, Some(out), None, &[]);
                }
                if eval > 0 {
                    env.relieve(&mut m.dit.blocks, &m.st, k * step_need(r.width, r.height, extra))?;
                }
                // the style reference, noised to this sigma, through the plain model; its statistics move this evaluation's
                if let Some((s, sctx, slt, srope, noise, xref, xout)) = &style_pass {
                    // a solver's intermediate sigma takes the step it is in
                    let step = sigmas.windows(2).position(|w| sigma <= w[0] && sigma > w[1]);
                    if step.is_some_and(|i| (s.steps.0..s.steps.1).contains(&i)) {
                        if captured != Some(sigma) {
                            let mix: Vec<f32> = s.reference.latent.iter().zip(noise).map(|(x0, e)| (1.0 - sigma) * x0 + sigma * e).collect();
                            xref.upload(0, &st::f32_bytes(&mix), st)?;
                            m.dit.adain.as_mut().unwrap().capture = true;
                            // the patch's model runs the reference pass too, as a positive evaluation
                            m.dit.step(xref, 1, s.reference.height, s.reference.width, sctx, *slt, None, srope, sigma, sigma, st, Some(xout), None,
                                &pairs)?;
                            m.dit.adain.as_mut().unwrap().capture = false;
                            captured = Some(sigma);
                        }
                        m.dit.adain.as_mut().unwrap().active = true;
                    }
                }
                let mut probe_store: Vec<(String, Vec<u8>)> = Vec::new();
                {
                    let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
                        if eval == 0 && (name == "block0" || name == "block27" || name == "dit_out_tokens") {
                            let mut v = vec![0u8; bytes];
                            b.download(0, &mut v, st).unwrap();
                            probe_store.push((name.to_string(), v));
                        }
                    };
                    let probe_ref: Option<&mut dyn FnMut(&str, &DevBuf, usize)> =
                        if dump.is_some() && side.entry == 0 { Some(&mut probe) } else { None };
                    m.dit.step(x, k, lh, lw, ctx, run.lt, refs, rope, sigma, sigma, st, Some(out), probe_ref, &pairs)?;
                }
                if let Some(ad) = m.dit.adain.as_mut() {
                    ad.active = false;
                }
                if let Some(d) = dump.as_deref_mut() {
                    for (name, v) in probe_store {
                        let shape = if name == "dit_out_tokens" { vec![(lh / 2) * (lw / 2), 64] } else { vec![v.len() / 2 / dit::F, dit::F] };
                        d.insert(name, ("BF16".into(), shape, v));
                    }
                }
                Ok(())
            };
            let positive: Vec<Region> = r.positive.iter().map(|e| e.cond.region.clone()).collect();
            let negative: Vec<Region> = negative.iter().map(|e| e.cond.region.clone()).collect();
            let regions: [&[Region]; 2] = [&positive, if guided { &negative } else { PLAIN }];
            let latent = run_batches(&r.sampling, &noise, regions, together, st, &mut stats, &mut model, &mut progress);
            latent.map(|l| (l, copied0))
        };
        m.dit.adain = None;
        while contexts.len() > CONTEXTS {
            contexts.remove(0);
        }
        let (latent, copied0) = result?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(d) = dump {
            d.insert("latent_final".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(s: &[(&str, f32)]) -> Vec<(String, f32)> {
        s.iter().map(|(t, w)| (t.to_string(), *w)).collect()
    }

    /// The first chunk's weights of `s` as leaves, with prompt images of `image_tokens` tokens.
    fn weights(s: &[(&str, f32)], image_tokens: &[usize]) -> ChunkWeights {
        let tok = tokenizer::Tokenizer::new().unwrap();
        let prompt = Prompt { chunks: vec![segments(s)], scale: 1.0 };
        krea_chunks(&tok, &prompt, true, image_tokens).unwrap().swap_remove(0).weights
    }

    #[test]
    fn weights_land_where_swarm_puts_them() {
        // SwarmTextEncodeAdvanced's token_weights_from_tokens on ComfyUI's Krea2Tokenizer, with uniform_weight
        let nested = weights(&[("lake, ", 1.0), ("pine ", 1.3), ("bark", 2.6), (" (sky:1.2)", 1.0)], &[]);
        let expect = vec![(3, 1.3), (4, 1.3), (5, 2.6), (6, 2.6)];
        assert_eq!(nested, ChunkWeights { rows: expect.clone(), whole: 1.0, pairs: expect });
        // one weight everywhere scales the rows, with no attention pairs; whitespace leaves don't count
        assert_eq!(weights(&[("a lake", 0.5)], &[]), ChunkWeights { rows: vec![(0, 0.5), (1, 0.5)], whole: 1.0, pairs: Vec::new() });
        assert_eq!(weights(&[("lake", 0.5), (" ", 1.0)], &[]), ChunkWeights { rows: vec![(0, 0.5)], whole: 1.0, pairs: Vec::new() });
        assert_eq!(weights(&[("a lake at dawn", 1.0)], &[]), ChunkWeights { rows: Vec::new(), whole: 1.0, pairs: Vec::new() });
        // prompt images grow the conditioning but not the token list, so weighted rows move past them
        let w = weights(&[("a ", 1.0), ("red", 1.5), (" lake", 1.0)], &[10, 10]);
        assert_eq!(w.rows, [(20, 1.5)]);
    }

    #[test]
    fn a_uniform_weight_without_rows_scales_everything() {
        // every weighted row before the conditioning: the node scales the whole conditioning instead
        let w = chunk_weights(&segments(&[("lake", 1.5)]), &[1.5, 1.0, 1.0], 2);
        assert_eq!(w, ChunkWeights { rows: Vec::new(), whole: 1.5, pairs: Vec::new() });
        assert_eq!(uniform_weight(&segments(&[(" ", 2.0)])), None);
        assert_eq!(uniform_weight(&segments(&[("a", 2.0), ("b", 1.0)])), None);
    }
}
