//! MiniMax H3: its diffusion model, text encoder, video and audio decoders, and encoded prompts.

use super::{nothing_for, run_sampler, scale_rows, touch};
use super::{Dump, FamilyState, Generator, Kind, Prompt, Sampling, SharedTe, Stats, Step, TokenKey, CONTEXTS};
use crate::cuda::{self, DevBuf, Stream};
use crate::{h3, h3te, h3vae, lora, offload, st, tokenizer};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub struct H3Request {
    pub prompt: Prompt,
    /// Only encoded when the sampling uses it.
    pub negative: Prompt,
    pub shape: h3::Shape,
    pub sampling: Sampling,
    /// video, audio
    pub shifts: (f64, f64),
    /// video, then audio
    pub noise: Vec<f32>,
}

struct Model {
    path: PathBuf,
    dit: h3::Dit,
    st: Arc<st::SafeTensors>,
    lora: lora::State,
}

#[derive(Default)]
pub(super) struct H3 {
    model: Option<Model>,
    te: Option<(PathBuf, h3te::TextEncoder)>,
    tokenizer: Option<tokenizer::Tokenizer>,
    vae: Option<(PathBuf, h3vae::VideoDecoder)>,
    audio: Option<(PathBuf, h3vae::AudioDecoder)>,
    /// Latest encoded tokens with their length, most recent last.
    contexts: Vec<(TokenKey, usize, DevBuf)>,
}

impl FamilyState for H3 {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        if m.dit.is_hyperflow() && !loras.is_empty() {
            bail!("LoRAs on HyperFlow H3 Flash: its layers are pruned and factorized, so H3 LoRAs don't fit them");
        }
        if lora::apply(lora::Arch::H3, &m.st, "", m.dit.stores(), &mut m.lora, loras, st)? {
            self.clear_prompts();
        }
        Ok(())
    }

    fn set_text_encoder_loras(&mut self, _: Option<&mut SharedTe>, loras: &[lora::Lora], _: &Stream) -> Result<()> {
        if !loras.is_empty() {
            bail!("MiniMax H3's text encoder doesn't take LoRAs");
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

    /// H3 places its own blocks per request ([`h3::Dit::reserve`]) and parks them for the decoders.
    fn blocks(&mut self) -> Option<(&mut offload::WeightSet, &st::SafeTensors)> {
        None
    }

    fn idle(&mut self) -> Result<()> {
        if let Some(m) = &mut self.model {
            m.dit.park()?;
        }
        Ok(())
    }
}

/// Token ids and per-token factors of a MiniMax H3 prompt, which has no template.
pub fn h3_tokens(t: &tokenizer::Tokenizer, segments: &[(String, f32)]) -> Result<(Vec<u32>, Vec<f32>)> {
    let (mut ids, mut factors) = (Vec::new(), Vec::new());
    for (text, weight) in segments {
        ids.extend(t.encode(text)?);
        factors.resize(ids.len(), *weight);
    }
    if ids.is_empty() {
        ids.push(151643);
        factors.push(1.0);
    }
    Ok((ids, factors))
}

impl H3 {
    /// Move the refined context of `prompt` to the end of the cache, encoding it if needed, and
    /// return its key.
    fn context(&mut self, prompt: &Prompt, st: &Stream, dump: Option<&mut Dump>) -> Result<TokenKey> {
        let tok = self.tokenizer.as_ref().context("no H3 tokenizer")?;
        let (ids, mut factors) = h3_tokens(tok, prompt.segments()?)?;
        factors.iter_mut().for_each(|f| *f *= prompt.scale);
        let key = vec![(ids, factors)];
        if touch(&mut self.contexts, |(k, ..)| *k == key) {
            return Ok(key);
        }
        let (ids, factors) = &key[0];
        let t0 = Instant::now();
        let (_, te) = self.te.as_mut().context("no H3 text encoder")?;
        let cond = te.encode(ids, st)?;
        scale_rows(&cond, h3te::D, factors, st)?;
        if let Some(d) = dump {
            let mut b = vec![0u8; ids.len() * h3te::D * 4];
            cond.download(0, &mut b, st)?;
            d.insert("context".into(), ("F32".into(), vec![1, ids.len(), h3te::D], b));
        }
        let m = self.model.as_mut().context("no H3 model")?;
        let ctx = m.dit.encode_context(&cond, ids.len(), st)?;
        st.sync()?;
        eprintln!("text encoded in {:.2}s, {} tokens", t0.elapsed().as_secs_f64(), ids.len());
        if self.contexts.len() >= CONTEXTS {
            self.contexts.remove(0);
        }
        self.contexts.push((key.clone(), ids.len(), ctx));
        Ok(key)
    }
}

impl super::Family {
    fn h3(&mut self) -> Option<&mut H3> {
        match self {
            super::Family::H3(f) => Some(f),
            _ => None,
        }
    }
}

impl Generator {
    /// Load a MiniMax H3 diffusion model unless `path` is loaded. VRAM is taken when it samples.
    pub fn ensure_h3(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::H3)?;
        let env = &self.env;
        let f = self.family.h3().context("MiniMax H3 not entered")?;
        if f.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.clear_prompts();
        f.model = None;
        let t0 = Instant::now();
        let mut file = st::SafeTensors::open(path)?;
        h3::prepare(&mut file)?;
        let dst = Arc::new(file);
        let mut dit = h3::Dit::load(&dst, env.threads, &env.st)?;
        dit.attn_variant = env.attn_variant;
        dit.int8_attn = env.int8_attn;
        let name = if dit.is_hyperflow() { "HyperFlow H3 Flash" } else { "MiniMax H3" };
        eprintln!("{name} loaded in {:.1}s", t0.elapsed().as_secs_f64());
        f.model = Some(Model { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default() });
        Ok(())
    }

    /// Load H3's text encoder unless `path` is loaded. It streams from pinned RAM.
    pub fn ensure_h3_te(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::H3)?;
        let env = &self.env;
        let f = self.family.h3().context("MiniMax H3 not entered")?;
        if f.te.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.clear_prompts();
        f.te = None;
        f.te = Some((path.to_path_buf(), h3te::TextEncoder::load(path, env.threads)?));
        if f.tokenizer.is_none() {
            f.tokenizer = Some(tokenizer::Tokenizer::new()?);
        }
        Ok(())
    }

    /// The diffusion model leaves VRAM for a decoder, keeping its LoRAs and encoded prompts.
    fn park_h3(&mut self) -> Result<()> {
        self.enter(Kind::H3)?;
        self.env.st.sync()?;
        if let Some(m) = self.family.h3().and_then(|f| f.model.as_mut()) {
            m.dit.park()?;
        }
        cuda::trim_pool()
    }

    /// Decode the audio of a clip, [32, 2, t] latents to stereo samples.
    pub fn decode_h3_audio(&mut self, path: &Path, latent: &[f32], t: usize) -> Result<Vec<f32>> {
        self.park_h3()?;
        let f = self.family.h3().context("MiniMax H3 not entered")?;
        if f.audio.as_ref().is_none_or(|(p, _)| p != path) {
            f.audio = None;
            f.audio = Some((path.to_path_buf(), h3vae::AudioDecoder::load(path)?));
        }
        h3vae::decode_audio(&f.audio.as_ref().unwrap().1, latent, t, &self.env.st)
    }

    /// Decode the video of a clip. Without the diffusion model in VRAM, the decoder's layers stay
    /// resident rather than crossing the bus once per tile.
    pub fn decode_h3(&mut self, path: &Path, latent: &[f32], sh: h3::Shape, frames: usize) -> Result<h3vae::Clip> {
        let t0 = Instant::now();
        self.park_h3()?;
        let parked = t0.elapsed().as_secs_f64();
        let env = &self.env;
        let budget = cuda::device_free()?.saturating_sub(env.reserve + (3 << 30));
        let f = self.family.h3().context("MiniMax H3 not entered")?;
        if f.vae.as_ref().is_none_or(|(p, _)| p != path) {
            f.vae = None;
            f.vae = Some((path.to_path_buf(), h3vae::VideoDecoder::load(path, env.threads, &env.st)?));
        }
        let v = &mut f.vae.as_mut().unwrap().1;
        v.place(budget)?;
        let loaded = t0.elapsed().as_secs_f64();
        let out = v.decode(&latent[..sh.video_len()], sh.t, sh.h, sh.w, frames, &env.st)?;
        env.st.sync()?;
        v.park()?;
        if std::env::var_os("PROFILE").is_some() {
            eprintln!("[prof] decode: park {:.0} ms, load {:.0} ms, tiles and blends {:.0} ms", parked * 1e3, (loaded - parked) * 1e3, (t0.elapsed().as_secs_f64() - loaded) * 1e3);
        }
        Ok(out)
    }

    /// Sample a clip; returns the pack of latents, video then audio.
    pub fn sample_h3(&mut self, r: &H3Request, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>) -> Result<(Vec<f32>, Stats)> {
        let t0 = Instant::now();
        let mut stats = Stats::default();
        let env = &self.env;
        let st = &env.st;
        let f = self.family.h3().context("no H3 model")?;
        let shifts = (r.shifts.0 as f32, r.shifts.1 as f32);
        let dit = &mut f.model.as_mut().context("no H3 model")?.dit;
        // its time embedding reads where each step ends, which only euler's steps are sure to
        if dit.is_hyperflow() && r.sampling.sampler != "euler" {
            bail!("the {} sampler on HyperFlow H3 Flash (only euler for now)", r.sampling.sampler);
        }
        dit.shifts = shifts;
        let neg_key = if r.sampling.uses_negative() { Some(f.context(&r.negative, st, None)?) } else { None };
        let pos_key = f.context(&r.prompt, st, dump.as_deref_mut())?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;
        let contexts = &f.contexts;
        let find = |key: &TokenKey| contexts.iter().find(|(k, ..)| k == key).map(|(_, len, ctx)| (ctx, *len)).unwrap();
        let (pos, text) = find(&pos_key);
        let neg = neg_key.as_ref().map(find);
        if let Some(d) = dump.as_deref_mut() {
            let bytes = text * 5376 * 2;
            let mut v = vec![0u8; bytes];
            pos.download(0, &mut v, st)?;
            d.insert("ctx".into(), ("BF16".into(), vec![bytes / 2], v));
        }
        let m = f.model.as_mut().unwrap();
        // what is left after the activations goes to the blocks, re-planned for this shape
        st.sync()?;
        cuda::trim_pool()?;
        let tokens = text.max(neg.map_or(0, |n| n.1)) + 2 * r.shape.audio_t + r.shape.t * (r.shape.h / 2) * (r.shape.w / 2);
        let budget = (cuda::device_free()? + m.dit.blocks_vram()).saturating_sub(env.reserve + (1 << 29) + m.dit.step_activation_bytes(tokens) + m.dit.attn_workspace(tokens));
        m.dit.reserve(budget, env.slots)?;
        let rope = m.dit.rope(text, &r.shape, st)?;
        let rope_neg = neg.map(|(_, l)| m.dit.rope(l, &r.shape, st)).transpose()?;
        let dit = &mut m.dit;
        let n = r.noise.len();
        let mut first = dump;
        let sigmas = &r.sampling.sigmas;
        let mut model = |x: &DevBuf, den: &DevBuf, sigma: f32, negative: bool, e: usize| -> Result<()> {
            // the schedule's next sigma after the nearest one, as HyperFlow's ComfyUI node finds it
            let i = (0..sigmas.len()).min_by(|&a, &b| (sigmas[a] - sigma).abs().total_cmp(&(sigmas[b] - sigma).abs())).unwrap_or(0);
            let next = sigmas.get(i + 1).or(sigmas.last()).copied().unwrap_or(0.0);
            let mut keep = |name: &str, b: &DevBuf, bytes: usize| {
                if let (0, false, Some(d)) = (e, negative, first.as_mut()) {
                    let mut v = vec![0u8; bytes];
                    if b.download(0, &mut v, st).is_ok() {
                        d.insert(name.to_string(), ("BF16".into(), vec![bytes / 2], v));
                    }
                }
            };
            match (negative, neg, &rope_neg) {
                (true, Some((c, l)), Some(rn)) => dit.predict(x, &r.shape, c, l, rn, sigma, next, den, st, None)?,
                _ => dit.predict(x, &r.shape, pos, text, &rope, sigma, next, den, st, Some(&mut keep))?,
            }
            if let (0, false, Some(d)) = (e, negative, first.as_mut()) {
                let mut b = vec![0u8; n * 4];
                den.download(0, &mut b, st)?;
                d.insert("denoised0".into(), ("F32".into(), vec![n], b));
            }
            Ok(())
        };
        let mut out = run_sampler(&r.sampling, &r.noise, st, &mut stats, &mut model, &mut progress)?;
        // the sampler carries the audio scaled onto the video schedule
        let scale = shifts.0 / shifts.1;
        for v in &mut out[r.shape.video_len()..] {
            *v /= scale;
        }
        Ok((out, stats))
    }
}
