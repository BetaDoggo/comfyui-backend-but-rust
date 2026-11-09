//! Anima: its diffusion model, its Qwen3 text encoder and the prompts encoded through them. The VAE is the generator's.

use super::{nothing_for, remember, run_batches, scale_rows, touch};
use super::{Dump, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Stats, Step, CONTEXTS};
use crate::cuda::{self, DevBuf, Stream};
use crate::{anima, dit, lora, offload, qwen3, snapref, st, tokenizer, vae};
use std::hash::{Hash, Hasher};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct AnimaRequest {
    /// One chunk of segments, see [`tokenizer::token_weights`]
    pub positive: Prompt,
    /// Only encoded when the sampling uses it.
    pub negative: Prompt,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise, one after another.
    pub noise: Vec<f32>,
    /// SnapRef's reference adapter, if any.
    pub snapref: Option<SnapRefRequest>,
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
        let (qids, tids, weights) = tokenizer::anima_tokens(tq, tt, prompt.segments()?)?;
        let hidden = te.encode(&qids, st)?;
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
        env.plan(&mut m.dit.blocks, &m.st, anima::reference_bytes(most) + anima::ref_kv_bytes(nb, sizes.iter().sum()))?;
        let kv = snapref::RefKv::alloc(nb, sizes.clone(), anima::REF_ROW_BYTES)?;
        let caption = &self.contexts.iter().find(|(p, _)| *p == r.caption).context("caption not encoded")?.1;
        let adapter = self.snapref.as_ref().unwrap();
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
        let snap_need = match (&r.snapref, &a.ref_kv) {
            (Some(sr), Some((_, kv))) => {
                let groups = snapref::RefRead { biases: sr.biases.clone() }.groups(&kv.sizes).len();
                anima::snapref_step_bytes((lh / 2) * (lw / 2), kv.rows, groups)
            }
            _ => 0,
        };
        let together = {
            let m = a.model.as_mut().context("no diffusion model loaded")?;
            let together = env.batch_fit(&m.dit.blocks, images, 0, step_need(r.width, r.height) + snap_need)?;
            env.plan(&mut m.dit.blocks, &m.st, anima_need(r.width, r.height, together) + together * snap_need)?;
            together
        };
        let guided = r.sampling.uses_negative();
        if guided {
            a.context(&r.negative, st, None)?;
        }
        a.context(&r.positive, st, dump.as_deref_mut())?;
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
            (Some((pr, nr)), Some(adapter), Some((_, kv))) => (
                anima::Pass::Target { adapter, read: Some((kv, pr)) },
                anima::Pass::Target { adapter, read: nr.as_ref().map(|n| (kv, n)) },
            ),
            (Some(_), _, _) => bail!("SnapRef's references weren't encoded"),
            _ => (anima::Pass::Plain, anima::Pass::Plain),
        };
        let contexts = &a.contexts;
        let find = |p: &Prompt| contexts.iter().find(|(k, _)| k == p).map(|(_, c)| c).unwrap();
        let pos = find(&r.positive);
        let neg = guided.then(|| find(&r.negative));
        let rope = m.dit.rope(lh / 2, lw / 2, st)?;
        let copied0 = m.dit.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, eval: usize, k: usize| -> Result<()> {
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
                m.dit.predict(x, k, lh, lw, neg.unwrap(), &rope, sigma, out, st, &neg_pass, probe_ref)?;
                if let (Some(dm), Some(v)) = (dump.as_deref_mut(), first) {
                    dm.insert("neg_block0".into(), ("BF16".into(), vec![v.len() / 2], v));
                }
                return Ok(());
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, k * step_need(r.width, r.height))?;
            }
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
                m.dit.predict(x, k, lh, lw, pos, &rope, sigma, out, st, &pos_pass, probe_ref)?;
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
            dm.insert("samples".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }
}
