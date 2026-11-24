//! SenseNova U1.6: its generation tower, its text tower (the same file) and the prompts that tower has run. There is
//! no VAE: the sampler works on pixels.

use super::{nothing_for, regions, remember, run_batches, touch, PLAIN};
use super::{Cond, Env, FamilyState, Generator, Kind, Sampling, SharedTe, Side, Stats, Step, CONTEXTS};
use crate::cuda::{DevBuf, Stream};
use crate::samplers::ModelSampling;
use crate::sensenova::{self, Gen, Prefix, Reference, TextTower};
use crate::{cuda, lora, offload, st};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Prompt tokens budgeted when planning before the prompts are known: the system turn is about 230.
const PROMPT_TOKENS: usize = 400;

pub struct SenseNovaRequest {
    /// The checkpoint whose text tower encodes the prompts.
    pub text_encoder: PathBuf,
    /// The positive conditioning entries, mixed as ComfyUI's `calc_cond_batch` mixes them.
    pub positive: Vec<Cond>,
    /// Only encoded when the sampling uses it.
    pub negative: Vec<Cond>,
    /// Each positive entry's reference pictures: with any, the entry is an edit of them, its prompt the text and the
    /// pictures (SenseNova Conditioning's `image` task).
    pub positive_refs: Vec<Vec<Reference>>,
    /// Each negative entry's: with any, the entry is the pictures alone, its text unread, as that task encodes it.
    pub negative_refs: Vec<Vec<Reference>>,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise, one after another, as SwarmKSampler makes it: this scales it by the size's noise scale.
    pub noise: Vec<f32>,
}

struct Model {
    path: PathBuf,
    gen: Gen,
    st: st::SafeTensors,
    lora: lora::State,
}

#[derive(Default)]
pub(super) struct SenseNova {
    model: Option<Model>,
    text: Option<(PathBuf, TextTower)>,
    /// Latest prompts' keys and values, by their tokens and their pictures' fingerprints, most recent last.
    prompts: Vec<((Vec<u32>, Vec<u64>), Arc<Prefix>)>,
}

impl FamilyState for SenseNova {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        // the generation tower's weights only: the prompts' keys and values come from the text tower
        lora::apply(lora::Arch::SenseNova, &m.st, "", m.gen.stores(), &mut m.lora, loras, st)?;
        Ok(())
    }

    fn set_text_encoder_loras(&mut self, _shared: Option<&mut SharedTe>, loras: &[lora::Lora], _st: &Stream) -> Result<()> {
        if !loras.is_empty() {
            return Err(lora::Unsupported("LoRAs on SenseNova's text tower".into()).into());
        }
        Ok(())
    }

    fn clear_prompts(&mut self) {
        self.prompts.clear();
    }

    fn set_int8_attention(&mut self, on: bool) {
        if let Some(m) = &mut self.model {
            m.gen.int8_attn = on;
        }
    }

    fn blocks(&mut self) -> Option<(&mut offload::WeightSet, &st::SafeTensors)> {
        self.model.as_mut().map(|m| (&mut m.gen.blocks, &m.st))
    }

    fn idle(&mut self) -> Result<()> {
        if let Some(m) = &mut self.model {
            m.gen.blocks.set_resident(0, 1, &m.st)?;
        }
        Ok(())
    }
}

/// One image's activations in a step at `width` x `height` against a prompt of `prefix` tokens.
fn step_need(width: usize, height: usize, prefix: usize) -> usize {
    let l = (height / sensenova::PATCH) * (width / sensenova::PATCH);
    sensenova::step_activation_bytes(l, prefix.max(PROMPT_TOKENS))
}

/// The text tower's activations for a prompt of `tokens`: its float32 weight, states and the two streamed layers.
fn text_need(tokens: usize) -> usize {
    2 * 12288 * 4096 * 4 + tokens * (12288 * 4 * 3 + 6144 * 6 + 4096 * 12) + 2 * 200 * (1 << 20)
}

impl super::Family {
    fn sensenova(&mut self) -> Option<&mut SenseNova> {
        match self {
            super::Family::SenseNova(f) => Some(f),
            _ => None,
        }
    }
}

impl SenseNova {
    /// The keys and values of `text` with the pictures `refs` (alone with `image_only`), moved to the end of the cache,
    /// running the text tower if needed.
    fn prefix(&mut self, env: &Env, text: &str, refs: &[Reference], image_only: bool) -> Result<Arc<Prefix>> {
        let tower = &mut self.text.as_mut().context("no SenseNova text tower loaded")?.1;
        let mut ids = sensenova::tokens(tower.tokenizer(), text)?;
        if !refs.is_empty() {
            let grids: Vec<_> = refs.iter().map(Reference::grid).collect();
            ids = sensenova::with_images(&ids, &grids, image_only);
        }
        let key = (ids, refs.iter().map(Reference::fingerprint).collect::<Vec<_>>());
        if touch(&mut self.prompts, |(k, _)| *k == key) {
            return Ok(self.prompts.last().unwrap().1.clone());
        }
        let ids = &key.0;
        if let Some(m) = &mut self.model {
            env.make_room(&mut m.gen.blocks, &m.st, text_need(ids.len()))?;
        }
        let t0 = Instant::now();
        let p = Arc::new(tower.encode(ids, refs, &env.st)?);
        env.st.sync()?;
        eprintln!("text: {} tokens ({} pictures) in {:.0} ms", ids.len(), refs.len(), t0.elapsed().as_secs_f64() * 1e3);
        remember(&mut self.prompts, (key, p.clone()), CONTEXTS);
        Ok(p)
    }
}

/// The one plain text of a prompt, or why it can't be one.
fn prompt_text(c: &Cond) -> Result<&str> {
    let p = &c.prompt;
    if p.scale != 1.0 {
        return Err(lora::Unsupported("Conditioning multipliers and zeroed conditioning on SenseNova (its prompts are keys and values, which ComfyUI's multiply leaves alone)".into()).into());
    }
    match p.chunks.as_slice() {
        [segments] => match segments.as_slice() {
            [] => Ok(""),
            [(text, w)] if *w == 1.0 => Ok(text),
            _ => Err(lora::Unsupported("prompt weights on SenseNova".into()).into()),
        },
        _ => Err(lora::Unsupported("<break> on SenseNova".into()).into()),
    }
}

impl Generator {
    /// Load a SenseNova checkpoint's generation tower unless `path` is loaded.
    pub fn ensure_sensenova(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::SenseNova)?;
        let env = &self.env;
        let f = self.family.sensenova().context("SenseNova not entered")?;
        if f.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.model = None;
        let t0 = Instant::now();
        let dst = st::SafeTensors::open(path)?;
        let pre = sensenova::key_prefix(|k| dst.has(k)).context("not a SenseNova checkpoint")?;
        if !pre.is_empty() {
            bail!("a SenseNova checkpoint with ComfyUI's model.diffusion_model. prefix (this backend runs the release's own names)");
        }
        let globals = sensenova::globals_layout(&dst, pre)?.size;
        let free = cuda::device_free()?;
        let budget = free.saturating_sub(env.reserve + globals + step_need(env.max_width, env.max_height, 0) + text_need(PROMPT_TOKENS));
        let mut gen = Gen::load(&dst, pre, budget, env.slots, env.threads, &env.st)?;
        gen.attn_variant = env.attn_variant;
        gen.int8_attn = env.int8_attn;
        eprintln!("SenseNova generation tower ready in {:.1}s", t0.elapsed().as_secs_f64());
        f.model = Some(Model { path: path.to_path_buf(), gen, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Load a SenseNova checkpoint's text tower unless `path`'s is loaded.
    pub fn ensure_sensenova_te(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::SenseNova)?;
        let threads = self.env.threads;
        let f = self.family.sensenova().context("SenseNova not entered")?;
        if f.text.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        f.text = None;
        f.prompts.clear();
        let t0 = Instant::now();
        f.text = Some((path.to_path_buf(), TextTower::load(path, threads)?));
        eprintln!("SenseNova text tower ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// SenseNova images, sampled as ComfyUI's KSampler does: f32 [images, 3, height, width], -1 to 1.
    pub fn sample_sensenova(&mut self, r: &SenseNovaRequest, mut progress: impl FnMut(Step) -> bool) -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        env.check_size_of(r.width, r.height, sensenova::PATCH)?;
        let ModelSampling::SenseNova { shift } = r.sampling.model_sampling else {
            bail!("SenseNova samples with its own model sampling");
        };
        let images = r.sampling.images;
        let n = images * 3 * r.height * r.width;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let f = self.family.sensenova().context("no SenseNova model loaded")?;
        if !f.text.as_ref().is_some_and(|(p, _)| *p == r.text_encoder) {
            bail!("the prompts' SenseNova text tower isn't loaded");
        }
        let guided = r.sampling.uses_negative();
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let mut prefixes: [Vec<Arc<Prefix>>; 2] = Default::default();
        if r.positive_refs.len() != r.positive.len() || guided && r.negative_refs.len() != r.negative.len() {
            bail!("reference pictures for {} and {} entries, of {} and {}", r.positive_refs.len(), r.negative_refs.len(), r.positive.len(), r.negative.len());
        }
        for (s, (side, refs)) in [(&r.positive, &r.positive_refs), (&r.negative, &r.negative_refs)].into_iter().enumerate().take(if guided { 2 } else { 1 }) {
            for (c, refs) in side.iter().zip(refs) {
                prefixes[s].push(f.prefix(env, prompt_text(c)?, refs, s == 1)?);
            }
        }
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;
        let m = f.model.as_mut().context("no SenseNova model loaded")?;
        let longest = prefixes.iter().flatten().map(|p| p.len).max().unwrap_or(0);
        let held: usize = prefixes.iter().flatten().map(|p| Prefix::bytes(p.len)).sum();
        let together = env.batch_fit(&m.gen.blocks, images, held, step_need(r.width, r.height, longest))?;
        env.plan(&mut m.gen.blocks, &m.st, together * step_need(r.width, r.height, longest))?;
        let st = &env.st;
        // ComfyUI's noise_scaling scales the noise by the size's noise scale
        let scale = sensenova::noise_scale(r.height, r.width) as f32;
        let noise: Vec<f32> = r.noise.iter().map(|v| v * scale).collect();
        let copied0 = m.gen.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, side: Side, eval: usize, k: usize| -> Result<()> {
            let Some(pf) = prefixes[side.negative as usize].get(side.entry) else {
                bail!("no prompt encoded for entry {} of the {} side", side.entry, if side.negative { "negative" } else { "positive" });
            };
            if eval > 0 && !side.negative {
                env.relieve(&mut m.gen.blocks, &m.st, k * step_need(r.width, r.height, pf.len))?;
            }
            m.gen.predict(x, k, r.height, r.width, pf, sigma, shift, out, st)
        };
        // an unguided run never evaluates the negative side
        let negative = if guided { regions(&r.negative) } else { PLAIN.to_vec() };
        let latent = run_batches(&r.sampling, &noise, [&regions(&r.positive)[..], &negative[..]], together, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.gen.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        Ok((latent, stats))
    }
}
