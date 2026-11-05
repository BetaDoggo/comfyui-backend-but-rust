//! Anima: its diffusion model, its Qwen3 text encoder and the prompts encoded through them. The VAE is the generator's.

use super::{nothing_for, remember, run_sampler, scale_rows, touch};
use super::{Dump, FamilyState, Generator, Kind, Prompt, Sampling, SharedTe, Stats, Step, CONTEXTS};
use crate::cuda::{self, DevBuf, Stream};
use crate::{anima, dit, lora, offload, qwen3, st, tokenizer, vae};
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
    pub sampling: Sampling,
    pub noise: Vec<f32>,
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

fn anima_need(width: usize, height: usize) -> usize {
    let (lh, lw) = (height / 8, width / 8);
    anima::step_activation_bytes((lh / 2) * (lw / 2)).max(vae::Vae::activation_bytes(lh, lw))
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
        let budget = free.saturating_sub(env.reserve + globals + anima_need(env.max_width, env.max_height));
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
        let keep = env.reserve + anima_need(env.max_width, env.max_height);
        let budget = if cuda::device_free()? >= size + keep { size } else { 0 };
        a.te = Some((path.to_path_buf(), qwen3::TextEncoder::load(path, budget, env.threads, &env.st)?));
        eprintln!("text encoder ready in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// One Anima latent, sampled as ComfyUI's KSampler does.
    pub fn sample_anima(&mut self, r: &AnimaRequest, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>)
        -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        env.check_size(r.width, r.height)?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let n = dit::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let a = self.family.anima().context("no diffusion model loaded")?;
        {
            let m = a.model.as_mut().context("no diffusion model loaded")?;
            env.plan(&mut m.dit.blocks, &m.st, anima_need(r.width, r.height))?;
        }
        let st = &env.st;
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let guided = r.sampling.uses_negative();
        if guided {
            a.context(&r.negative, st, None)?;
        }
        a.context(&r.positive, st, dump.as_deref_mut())?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;

        let m = a.model.as_mut().unwrap();
        let contexts = &a.contexts;
        let find = |p: &Prompt| contexts.iter().find(|(k, _)| k == p).map(|(_, c)| c).unwrap();
        let pos = find(&r.positive);
        let neg = guided.then(|| find(&r.negative));
        let rope = m.dit.rope(lh / 2, lw / 2, st)?;
        let copied0 = m.dit.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, negative: bool, eval: usize| -> Result<()> {
            if negative {
                return m.dit.predict(x, lh, lw, neg.unwrap(), &rope, sigma, out, st, None);
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, anima::step_activation_bytes((lh / 2) * (lw / 2)))?;
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
                m.dit.predict(x, lh, lw, pos, &rope, sigma, out, st, probe_ref)?;
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
            dm.insert("samples".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }
}
