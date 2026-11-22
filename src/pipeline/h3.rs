//! MiniMax H3: its diffusion model, text encoder with its vision tower, video and audio decoders, the VAE encoder for
//! reference pictures and keyframes, and encoded prompts.

use super::{PLAIN, Side, nothing_for, regions, remember, run_sampler, scale_rows, touch};
use super::{Cond, Dump, FamilyState, Generator, Kind, Prompt, Sampling, SharedTe, Stats, Step, TokenKey, CONTEXTS, VISION_CACHE};
use crate::cuda::{self, DevBuf, Stream};
use crate::st::f32_bytes;
use crate::vision::{self, PromptImage};
use crate::{h3, h3dwd, h3te, h3vae, h3window, lora, offload, st, tokenizer};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// A reference picture: the copy the vision tower sees, and the one the VAE encodes into a reference latent (sides
/// multiples of 32), if any.
#[derive(Clone, PartialEq)]
pub struct H3Picture {
    pub vision: PromptImage,
    pub latent: Option<PromptImage>,
}

/// What a positive conditioning entry is conditioned on besides its text.
#[derive(Clone, Default, PartialEq)]
pub struct H3Conds {
    /// The label before each picture in the text encoder's presentation, `{}` its number from 1: `<Picture {}>: ` for
    /// the video model's references, `Picture {}: ` for MiniMax-H3-Image's edits.
    pub label: String,
    pub pictures: Vec<H3Picture>,
    /// The VAE whose encoder makes the pictures' latents.
    pub vae: Option<PathBuf>,
    /// Keyframes: the pixel frame each anchors, and its latent `[24, 1, h, w]` at the clip's size.
    pub keyframes: Vec<(usize, Vec<f32>)>,
}

/// Reference latents kept between runs.
const REF_LATENTS: usize = 8;

/// `_size_short_edge`: an aspect-kept size whose short edge is within [low, high], each side rounded to 32.
pub fn size_short_edge(width: usize, height: usize, low: usize, high: usize) -> (usize, usize) {
    let short = width.min(height) as f64;
    let scale = short.clamp(low as f64, high as f64) / short;
    let side = |v: usize| ((v as f64 * scale / 32.0).round_ties_even() as usize * 32).max(32);
    (side(width), side(height))
}

/// `MiniMaxH3ReferenceToVideo`'s size for a reference picture: scaled down to the clip's area (`match`) or to a 2048
/// short edge (`max`), never up, each side rounded to 32.
pub fn ref2va_size(width: usize, height: usize, (cw, ch): (usize, usize), max: bool) -> (usize, usize) {
    let scale = if max { (2048.0 / width.min(height) as f64).min(1.0) } else { ((cw * ch) as f64 / (width * height) as f64).sqrt().min(1.0) };
    let side = |v: usize| ((v as f64 * scale / 32.0).round_ties_even() as usize * 32).max(32);
    (side(width), side(height))
}

/// `common_upscale(..., "lanczos", "disabled")` to `width` x `height`; `_resize_to` leaves a picture already that size
/// untouched.
pub fn resize(img: &PromptImage, width: usize, height: usize) -> PromptImage {
    if (img.width, img.height) == (width, height) {
        return img.clone();
    }
    PromptImage { width, height, rgb: crate::pixal3d::lanczos_f32(&img.rgb, img.width, img.height, 3, width, height) }
}

/// MiniMax-H3-Image's edit pictures from `images` (`MiniMaxH3ImageConditioning`): the vision tower sees each at a 1024
/// short edge, the VAE within 1024 to 2048, when there is one.
pub fn image_pictures(images: &[PromptImage], vae: bool) -> Vec<H3Picture> {
    images.iter().map(|img| {
        let (vw, vh) = size_short_edge(img.width, img.height, 1024, 1024);
        let (lw, lh) = size_short_edge(img.width, img.height, 1024, 2048);
        H3Picture { vision: resize(img, vw, vh), latent: vae.then(|| resize(img, lw, lh)) }
    }).collect()
}

/// The video model's reference pictures (`MiniMaxH3ReferenceToVideo`, SwarmUI's references): one copy for both.
pub fn ref2va_pictures(images: &[PromptImage], clip: (usize, usize), max: bool, vae: bool) -> Vec<H3Picture> {
    images.iter().map(|img| {
        let (w, h) = ref2va_size(img.width, img.height, clip, max);
        let r = resize(img, w, h);
        H3Picture { latent: vae.then(|| r.clone()), vision: r }
    }).collect()
}

pub struct H3Request {
    /// The positive conditioning entries, mixed as ComfyUI's `calc_cond_batch` does.
    pub positive: Vec<Cond>,
    /// Only encoded when the sampling uses it. Takes no pictures or keyframes.
    pub negative: Vec<Cond>,
    pub shape: h3::Shape,
    pub sampling: Sampling,
    /// video, audio
    pub shifts: (f64, f64),
    /// video, then audio
    pub noise: Vec<f32>,
    /// block-sparse self-attention
    pub sparse: Option<h3::SparseAttn>,
    /// Video Context Window: each evaluation runs on overlapping windows of the clip. FreeNoise
    /// is applied to `noise` by the caller.
    pub windows: Option<h3window::ContextWindows>,
    /// Reference pictures and keyframes of each positive entry.
    pub conds: Vec<H3Conds>,
    /// The sampler's seed, which the conditions' noise is drawn from.
    pub seed: u64,
}

/// A prompt's tokens with the pictures shown before them.
type H3Key = (TokenKey, Vec<PromptImage>);

struct Model {
    path: PathBuf,
    dit: h3::Dit,
    st: Arc<st::SafeTensors>,
    lora: lora::State,
    /// the LakonLab adapter among the LoRAs, whose small tensors the model holds
    adapter: Option<(PathBuf, Arc<st::SafeTensors>)>,
}

#[derive(Default)]
pub(super) struct H3 {
    model: Option<Model>,
    te: Option<(PathBuf, h3te::TextEncoder)>,
    tokenizer: Option<tokenizer::Tokenizer>,
    vae: Option<(PathBuf, h3vae::VideoDecoder)>,
    audio: Option<(PathBuf, h3vae::AudioDecoder)>,
    /// The text encoder file's vision tower, loaded by the first picture.
    vision: Option<(PathBuf, vision::Vision)>,
    /// Latest pictures through the tower, most recent last.
    vision_cache: Vec<(PromptImage, Arc<vision::Embeds>)>,
    /// Latest reference latents by VAE and picture, most recent last.
    ref_latents: Vec<((PathBuf, PromptImage), Arc<Vec<f32>>)>,
    /// Latest encoded prompts with their length and per-token tags (1 text, 0 vision), most recent last.
    contexts: Vec<(H3Key, usize, DevBuf, Vec<u8>)>,
    /// The pixel head of the adapter's file, loaded by the first run that decodes with it.
    dwd: Option<(PathBuf, h3dwd::Refiner)>,
    /// The frames that head decoded at the end of the latest run.
    dwd_pixels: Option<h3vae::Clip>,
}

impl FamilyState for H3 {
    fn set_loras(&mut self, loras: &[lora::Lora], st: &Stream) -> Result<()> {
        let Some(m) = &mut self.model else { return nothing_for(loras, "diffusion model") };
        if m.dit.is_hyperflow() && !loras.is_empty() {
            bail!("LoRAs on HyperFlow H3 Flash: its layers are pruned and factorized, so H3 LoRAs don't fit them");
        }
        if m.dit.is_image() && !loras.is_empty() {
            bail!("LoRAs on MiniMax-H3-Image (not supported yet)");
        }
        // a LakonLab adapter (pi-H3) also replaces small tensors and may bring a DWD pixel head
        let mut adapters = Vec::new();
        for l in loras {
            let f = st::SafeTensors::open(&l.path)?;
            if lora::is_lakonlab_adapter(&f) {
                adapters.push((l.path.clone(), f));
            }
        }
        if adapters.len() > 1 {
            return Err(lora::Unsupported("more than one LakonLab adapter on MiniMax H3".into()).into());
        }
        let adapter = adapters.pop();
        if m.adapter.as_ref().map(|a| &a.0) != adapter.as_ref().map(|a| &a.0) {
            let name = adapter.as_ref().map(|(p, _)| p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()));
            m.dit.set_adapter(&m.st, adapter.as_ref().map(|a| &a.1), st).map_err(|e| lora::Unsupported(format!("{}: {e:#}", name.unwrap_or_default())))?;
            m.adapter = adapter.map(|(p, f)| (p, Arc::new(f)));
            self.dwd = None;
            self.dwd_pixels = None;
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

/// Drop the oldest entries of `cache` until one more fits within [`CONTEXTS`], never one `kept` holds on to.
fn evict<T>(cache: &mut Vec<T>, kept: impl Fn(&T) -> bool) {
    while cache.len() >= CONTEXTS {
        match cache.iter().position(|e| !kept(e)) {
            Some(i) => drop(cache.remove(i)),
            None => break,
        }
    }
}

/// The distinct (prompt, pictures and keyframes) of a run's entries, and which one each entry of each side uses.
fn units<K: PartialEq, C: PartialEq>(positive: impl IntoIterator<Item = (K, C)>, negative: impl IntoIterator<Item = (K, C)>) -> (Vec<(K, C)>, [Vec<usize>; 2]) {
    let mut units: Vec<(K, C)> = Vec::new();
    let mut map = [Vec::new(), Vec::new()];
    for (side, entries) in [(0, positive.into_iter().collect::<Vec<_>>()), (1, negative.into_iter().collect())] {
        for u in entries {
            let i = units.iter().position(|v| *v == u).unwrap_or_else(|| {
                units.push(u);
                units.len() - 1
            });
            map[side].push(i);
        }
    }
    (units, map)
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

/// The ids and per-token factors of `pictures` pictures, each `label` with its number then its vision block, before
/// `segments`: `MiniMaxH3Tokenizer`'s presentation, which has no template.
pub fn h3_presentation(t: &tokenizer::Tokenizer, label: &str, pictures: usize, segments: &[(String, f32)]) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut ids = Vec::new();
    for i in 1..=pictures {
        ids.extend(t.encode(&label.replace("{}", &i.to_string()))?);
        ids.extend([h3te::VISION_START, h3te::IMAGE_PAD, h3te::VISION_END]);
    }
    let mut factors = vec![1.0; ids.len()];
    if pictures == 0 {
        return h3_tokens(t, segments);
    }
    for (text, weight) in segments {
        ids.extend(t.encode(text)?);
        factors.resize(ids.len(), *weight);
    }
    Ok((ids, factors))
}

impl H3 {
    /// The vision tower's tokens for `images`, through the cache.
    fn vision_embeds(&mut self, images: &[PromptImage], threads: usize, st: &Stream) -> Result<Vec<Arc<vision::Embeds>>> {
        let mut out = Vec::with_capacity(images.len());
        for img in images {
            if touch(&mut self.vision_cache, |(k, _)| k == img) {
                out.push(self.vision_cache.last().unwrap().1.clone());
                continue;
            }
            let (path, _) = self.te.as_ref().context("no H3 text encoder")?;
            if self.vision.as_ref().is_none_or(|(p, _)| p != path) {
                self.vision = None;
                let t0 = Instant::now();
                self.vision = Some((path.clone(), vision::Vision::load(st::SafeTensors::open(path)?, threads, st)?));
                eprintln!("vision tower ready in {:.1}s", t0.elapsed().as_secs_f64());
            }
            let t0 = Instant::now();
            let e = Arc::new(self.vision.as_mut().unwrap().1.encode(img, st, None)?);
            eprintln!("vision: {}x{}, {} tokens in {:.0} ms", img.width, img.height, e.tokens, t0.elapsed().as_secs_f64() * 1e3);
            out.push(e.clone());
            remember(&mut self.vision_cache, (img.clone(), e), VISION_CACHE);
        }
        Ok(out)
    }

    /// The cache key of `prompt` with `pictures` before it.
    fn key(&self, prompt: &Prompt, label: &str, pictures: &[PromptImage]) -> Result<H3Key> {
        let tok = self.tokenizer.as_ref().context("no H3 tokenizer")?;
        let (ids, mut factors) = h3_presentation(tok, label, pictures.len(), prompt.segments()?)?;
        factors.iter_mut().for_each(|f| *f *= prompt.scale);
        Ok((vec![(ids, factors)], pictures.to_vec()))
    }

    /// Move the refined context of `key` to the end of the cache, encoding it if needed. A new one makes room by
    /// dropping the oldest that isn't in `keep`, the run's others.
    fn context(&mut self, key: &H3Key, keep: &[&H3Key], threads: usize, st: &Stream, dump: Option<&mut Dump>) -> Result<()> {
        if touch(&mut self.contexts, |(k, ..)| k == key) {
            return Ok(());
        }
        let t0 = Instant::now();
        let embeds = self.vision_embeds(&key.1, threads, st)?;
        let refs: Vec<&vision::Embeds> = embeds.iter().map(|e| &**e).collect();
        let (ids, factors) = &key.0[0];
        // each picture's factor and tags cover its tokens
        let expanded = h3te::TextEncoder::expand(ids, &refs);
        let mut grown = Vec::with_capacity(expanded.len());
        let mut next = refs.iter();
        for (&t, &f) in ids.iter().zip(factors) {
            let n = if t == h3te::IMAGE_PAD { next.next().map_or(1, |e| e.tokens) } else { 1 };
            grown.extend(std::iter::repeat_n(f, n));
        }
        let vision_ids = [h3te::VISION_START, h3te::IMAGE_PAD, h3te::VISION_END];
        let tags: Vec<u8> = expanded.iter().map(|t| u8::from(!vision_ids.contains(t))).collect();
        let len = expanded.len();
        let (_, te) = self.te.as_mut().context("no H3 text encoder")?;
        let cond = te.encode(ids, &embeds, st)?;
        scale_rows(&cond, h3te::D, &grown, st)?;
        if let Some(d) = dump {
            let mut b = vec![0u8; len * h3te::D * 4];
            cond.download(0, &mut b, st)?;
            d.insert("context".into(), ("F32".into(), vec![1, len, h3te::D], b));
        }
        let m = self.model.as_mut().context("no H3 model")?;
        let ctx = m.dit.encode_context(&cond, len, st)?;
        st.sync()?;
        eprintln!("text encoded in {:.2}s, {len} tokens", t0.elapsed().as_secs_f64());
        evict(&mut self.contexts, |(k, ..)| keep.contains(&k));
        self.contexts.push((key.clone(), len, ctx, tags));
        Ok(())
    }

    /// The sampler's latents of `pictures` through the encoder of `vae`, through the cache.
    fn ref_latents(&mut self, vae: &Path, pictures: &[&PromptImage], st: &Stream) -> Result<Vec<Arc<Vec<f32>>>> {
        let mut out = Vec::with_capacity(pictures.len());
        let mut enc = None;
        for &img in pictures {
            if touch(&mut self.ref_latents, |((p, k), _)| p == vae && k == img) {
                out.push(self.ref_latents.last().unwrap().1.clone());
                continue;
            }
            if enc.is_none() {
                // the diffusion model's blocks make room; it places them again when it samples
                if let Some(m) = &mut self.model {
                    st.sync()?;
                    m.dit.park()?;
                    cuda::trim_pool()?;
                }
                enc = Some(h3vae::Encoder::load(vae, st)?);
            }
            let t0 = Instant::now();
            let z = Arc::new(enc.as_ref().unwrap().encode(&img.rgb, img.height, img.width, st)?);
            eprintln!("reference picture {}x{} encoded in {:.0} ms", img.width, img.height, t0.elapsed().as_secs_f64() * 1e3);
            out.push(z.clone());
            remember(&mut self.ref_latents, ((vae.to_path_buf(), img.clone()), z), REF_LATENTS);
        }
        Ok(out)
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
        if h3::is_hyperflow_diffusers(path) {
            bail!("{}", h3::HYPERFLOW_DIFFUSERS);
        }
        let dst = Arc::new(st::SafeTensors::open(path)?);
        let mut dit = h3::Dit::load(&dst, env.threads, &env.st)?;
        dit.attn_variant = env.attn_variant;
        dit.int8_attn = env.int8_attn;
        let name = if dit.is_hyperflow() { "HyperFlow H3 Flash" } else { "MiniMax H3" };
        eprintln!("{name} loaded in {:.1}s", t0.elapsed().as_secs_f64());
        f.model = Some(Model { path: path.to_path_buf(), dit, st: dst, lora: lora::State::default(), adapter: None });
        f.dwd_pixels = None;
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

    /// A picture through the encoder of the H3 VAE at `vae` (sides multiples of 16): the sampler's latent
    /// `[24, h / 16, w / 16]`, for keyframes.
    pub fn encode_h3(&mut self, vae: &Path, img: &PromptImage) -> Result<Vec<f32>> {
        self.enter(Kind::H3)?;
        let st = &self.env.st;
        let f = self.family.h3().context("MiniMax H3 not entered")?;
        Ok(f.ref_latents(vae, &[img], st)?.remove(0).as_ref().clone())
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

    /// The frames a DWD adapter's pixel head decoded at the end of the latest [`Self::sample_h3`], RGB f32
    /// `[3, frames, h, w]` in 0..1.
    pub fn take_h3_dwd_pixels(&mut self) -> Option<h3vae::Clip> {
        self.family.h3().and_then(|f| f.dwd_pixels.take())
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
        if dit.is_hyperflow() && !h3::hyperflow_sampler_ok(&r.sampling.sampler) {
            bail!("the {} sampler on HyperFlow H3 Flash (only euler and res_multistep)", r.sampling.sampler);
        }
        dit.shifts = shifts;
        dit.sparse = r.sparse.clone();
        if r.conds.len() != r.positive.len() {
            bail!("{} sets of pictures and keyframes for {} positive entries", r.conds.len(), r.positive.len());
        }
        let windowed = r.windows.as_ref().is_some_and(|w| w.applies(&r.shape));
        let dwd = dit.has_dwd();
        if dwd && windowed {
            return Err(lora::Unsupported("context windows with a DWD adapter (its pixel head decodes the whole clip at the last step)".into()).into());
        }
        if dwd && r.positive.len() > 1 {
            // ComfyUI-piFlow only decodes an evaluation of one positive entry, and finds no pixels otherwise
            return Err(lora::Unsupported("several positive conditioning entries with a DWD adapter".into()).into());
        }
        let conditioned = |c: &H3Conds| !c.keyframes.is_empty() || c.pictures.iter().any(|p| p.latent.is_some());
        if r.conds.iter().any(conditioned) && (windowed || r.sparse.is_some()) {
            bail!("keyframes and reference pictures with context windows or sparse attention");
        }
        let several = r.positive.len() > 1 || r.sampling.uses_negative() && r.negative.len() > 1;
        if windowed && several && r.windows.as_ref().is_some_and(|w| w.split_conds) {
            bail!("split_conds_to_windows with several conditioning entries");
        }
        // each distinct prompt with its pictures and keyframes is encoded and packed once
        let none = H3Conds::default();
        let mut positive = Vec::with_capacity(r.positive.len());
        for (p, c) in r.positive.iter().zip(&r.conds) {
            let vision: Vec<PromptImage> = c.pictures.iter().map(|p| p.vision.clone()).collect();
            positive.push((f.key(&p.prompt, &c.label, &vision)?, c));
        }
        let mut negative = Vec::new();
        if r.sampling.uses_negative() {
            for n in &r.negative {
                negative.push((f.key(&n.prompt, "", &[])?, &none));
            }
        }
        let (units, map) = units(positive, negative);
        let Some(&main) = map[0].first() else { bail!("a positive prompt without conditioning entries") };
        let needed: Vec<&H3Key> = units.iter().map(|(k, _)| k).collect();
        for (u, (key, _)) in units.iter().enumerate() {
            f.context(key, &needed, env.threads, st, if u == main { dump.as_deref_mut() } else { None })?;
        }
        // per unit: keyframes, then the references, as they pack, and their embedded rows
        let mut packs: Vec<(h3::Extra, Option<DevBuf>)> = Vec::with_capacity(units.len());
        for (u, (_, c)) in units.iter().enumerate() {
            let mut extra = h3::Extra::default();
            let mut latents: Vec<(Arc<Vec<f32>>, usize, usize, usize)> = Vec::new();
            for (index, z) in &c.keyframes {
                if z.len() != 24 * r.shape.h * r.shape.w {
                    bail!("a keyframe latent of {} values for a {}x{} clip", z.len(), r.shape.w * 16, r.shape.h * 16);
                }
                extra.keyframes.push((*index, 1));
                latents.push((Arc::new(z.clone()), 1, r.shape.h, r.shape.w));
            }
            let to_encode: Vec<&PromptImage> = c.pictures.iter().filter_map(|p| p.latent.as_ref()).collect();
            if !to_encode.is_empty() {
                let vae = c.vae.as_ref().context("reference pictures to encode without a VAE")?;
                for (z, img) in f.ref_latents(vae, &to_encode, st)?.into_iter().zip(&to_encode) {
                    extra.refs.push((img.height / 16, img.width / 16));
                    latents.push((z, 1, img.height / 16, img.width / 16));
                }
            }
            if let (true, Some(d)) = (u == main, dump.as_deref_mut()) {
                for (i, (z, ..)) in latents.iter().skip(c.keyframes.len()).enumerate() {
                    d.insert(format!("ref{i}"), ("F32".into(), vec![z.len()], f32_bytes(z).to_vec()));
                }
            }
            let m = f.model.as_ref().context("no H3 model")?;
            let rows = if extra.is_empty() {
                None
            } else {
                let views: Vec<(&[f32], usize, usize, usize)> = latents.iter().map(|(z, t, h, w)| (z.as_slice(), *t, *h, *w)).collect();
                Some(m.dit.cond_rows(&views, r.seed, st)?)
            };
            packs.push((extra, rows));
        }
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;
        let contexts = &f.contexts;
        let mut encoded: Vec<(&DevBuf, usize, &[u8])> = Vec::with_capacity(units.len());
        for (key, _) in &units {
            let (_, len, ctx, tags) = contexts.iter().find(|(k, ..)| k == key).context("an encoded prompt left the cache")?;
            encoded.push((ctx, *len, tags.as_slice()));
        }
        if let Some(d) = dump.as_deref_mut() {
            let (pos, text, _) = encoded[main];
            let bytes = text * 5376 * 2;
            let mut v = vec![0u8; bytes];
            pos.download(0, &mut v, st)?;
            d.insert("ctx".into(), ("BF16".into(), vec![bytes / 2], v));
        }
        let m = f.model.as_mut().unwrap();
        // what is left after the activations goes to the blocks, re-planned for this shape
        st.sync()?;
        cuda::trim_pool()?;
        // with context windows, a call only holds one window's tokens
        let windows = r.windows.as_ref().filter(|c| c.applies(&r.shape));
        let shapes: Vec<h3::Shape> = match windows {
            Some(c) => c.windows(&r.shape).iter().map(|w| w.shape(&r.shape)).collect(),
            None => vec![r.shape],
        };
        let largest = shapes.iter().map(|s| 2 * s.audio_t + s.t * (s.h / 2) * (s.w / 2)).max().unwrap_or(0);
        let tokens = encoded.iter().zip(&packs).map(|((_, len, _), (extra, _))| len + extra.rows(&r.shape)).max().unwrap_or(0) + largest;
        let budget = (cuda::device_free()? + m.dit.blocks_vram()).saturating_sub(env.reserve + (1 << 29) + m.dit.step_activation_bytes(tokens) + m.dit.attn_workspace(tokens));
        m.dit.reserve(budget, env.slots)?;
        // a rope table per text length, packed extras and window shape: windows can differ by an audio latent
        let mut ropes: Vec<(usize, &h3::Extra, h3::Shape, DevBuf)> = Vec::new();
        for ((_, len, _), (extra, _)) in encoded.iter().zip(&packs) {
            for s in &shapes {
                if !ropes.iter().any(|(l, e, k, _)| l == len && *e == extra && k == s) {
                    ropes.push((*len, extra, *s, m.dit.rope(*len, s, extra, st)?));
                }
            }
        }
        if let Some(c) = windows {
            eprintln!("context windows: {} of {} latents with {} overlapping, for {} latents", shapes.len(), c.length, c.overlap, r.shape.t);
        }
        let dit = &mut m.dit;
        let n = r.noise.len();
        let mut first = dump;
        let sigmas = &r.sampling.sigmas;
        // the DWD head decodes the positive evaluation at the schedule's last sigma above zero, as `torch.isclose` finds it
        let last = sigmas.iter().rev().copied().find(|s| *s > 0.0).unwrap_or(0.0);
        f.dwd_pixels = None;
        let mut model = |x: &DevBuf, den: &DevBuf, sigma: f32, side: Side, e: usize| -> Result<()> {
            let u = *map[usize::from(side.negative)].get(side.entry).context("no such conditioning entry")?;
            let (ctx, text, tags) = encoded[u];
            let (extra, rows) = &packs[u];
            let at_last = (sigma - last).abs() <= 1e-8 + 1e-5 * last.abs();
            let pack = h3::Pack { tags, cond: rows.as_ref().map(|c| (c, extra)), dwd: dwd && !side.negative && at_last };
            // the first evaluation of the first positive entry is dumped
            let dumped = e == 0 && side == Side::POSITIVE;
            // the schedule's next sigma after the nearest one, as HyperFlow's ComfyUI node finds it
            let i = (0..sigmas.len()).min_by(|&a, &b| (sigmas[a] - sigma).abs().total_cmp(&(sigmas[b] - sigma).abs())).unwrap_or(0);
            let next = sigmas.get(i + 1).or(sigmas.last()).copied().unwrap_or(0.0);
            let mut keep = |name: &str, b: &DevBuf, bytes: usize| {
                if let (true, Some(d)) = (dumped, first.as_mut()) {
                    let mut v = vec![0u8; bytes];
                    if b.download(0, &mut v, st).is_ok() {
                        d.insert(name.to_string(), ("BF16".into(), vec![bytes / 2], v));
                    }
                }
            };
            let mut call = |x: &DevBuf, sh: &h3::Shape, den: &DevBuf, probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>| -> Result<()> {
                let (.., rope) = ropes.iter().find(|(l, e, k, _)| *l == text && *e == extra && k == sh).context("no rope table for this window")?;
                dit.predict(x, sh, ctx, text, pack, rope, sigma, next, den, st, probe)
            };
            match windows {
                None => call(x, &r.shape, den, Some(&mut keep))?,
                Some(c) => {
                    let mut b = vec![0u8; n * 4];
                    x.download(0, &mut b, st)?;
                    let full: Vec<f32> = b.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
                    let fused = c.run(&full, &r.shape, |pack, sh| {
                        let wx = DevBuf::alloc_async(pack.len() * 4, st)?;
                        wx.upload(0, &f32_bytes(pack), st)?;
                        let wd = DevBuf::alloc_async(pack.len() * 4, st)?;
                        call(&wx, sh, &wd, None)?;
                        let mut o = vec![0u8; pack.len() * 4];
                        wd.download(0, &mut o, st)?;
                        Ok(o.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect())
                    })?;
                    den.upload(0, &f32_bytes(&fused), st)?;
                }
            }
            if let (true, Some(d)) = (dumped, first.as_mut()) {
                let mut b = vec![0u8; n * 4];
                den.download(0, &mut b, st)?;
                d.insert("denoised0".into(), ("F32".into(), vec![n], b));
            }
            Ok(())
        };
        // the negative's entries are only read under guidance
        let pos_regions = regions(&r.positive);
        let neg_regions = if r.sampling.uses_negative() { regions(&r.negative) } else { PLAIN.to_vec() };
        let mut out = run_sampler(&r.sampling, &r.noise, [&pos_regions, &neg_regions], st, &mut stats, &mut model, &mut progress)?;
        if let Some((features, grid)) = dit.take_dwd_features() {
            if let Some(d) = first.as_mut() {
                let mut v = vec![0u8; features.size()];
                features.download(0, &mut v, st)?;
                d.insert("dwd_features".into(), ("BF16".into(), vec![v.len() / 2], v));
            }
            // the pixel head runs with the blocks out of VRAM, as a decoder does
            let t1 = Instant::now();
            st.sync()?;
            dit.park()?;
            cuda::trim_pool()?;
            let (path, ad) = m.adapter.as_ref().context("DWD features without an adapter")?;
            if f.dwd.as_ref().is_none_or(|(p, _)| p != path) {
                f.dwd = None;
                f.dwd = Some((path.clone(), h3dwd::Refiner::load(ad, env.threads, st)?));
            }
            let head = &mut f.dwd.as_mut().unwrap().1;
            head.place(cuda::device_free()?.saturating_sub(env.reserve + (3 << 30) + h3dwd::activation_bytes(grid)))?;
            let pixels = head.decode(&features, grid, st)?;
            drop(features);
            head.park()?;
            eprintln!("DWD head: {} frames of {}x{} in {:.2}s", pixels.t, pixels.w, pixels.h, t1.elapsed().as_secs_f64());
            f.dwd_pixels = Some(pixels);
        } else if dwd {
            eprintln!("DWD adapter: no evaluation at the schedule's last sigma kept features, so there are no DWD frames");
        }
        // the sampler carries the audio scaled onto the video schedule
        let scale = shifts.0 / shifts.1;
        for v in &mut out[r.shape.video_len()..] {
            *v /= scale;
        }
        Ok((out, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_share_units_by_prompt_pictures_and_keyframes() {
        let (units, map) = units([("a", 1), ("b", 1), ("a", 1), ("a", 2)], [("a", 0), ("n", 0)]);
        assert_eq!(units, [("a", 1), ("b", 1), ("a", 2), ("a", 0), ("n", 0)]);
        assert_eq!(map, [vec![0, 1, 0, 2], vec![3, 4]]);
    }

    #[test]
    fn eviction_keeps_the_runs_prompts() {
        let mut cache: Vec<usize> = (0..CONTEXTS).collect();
        evict(&mut cache, |e| *e == 0);
        assert_eq!(cache, (0..CONTEXTS).filter(|&e| e != 1).collect::<Vec<_>>());
        // a run that needs more than CONTEXTS keeps them all
        let mut cache: Vec<usize> = (0..CONTEXTS + 2).collect();
        evict(&mut cache, |_| true);
        assert_eq!(cache.len(), CONTEXTS + 2);
        // and the next run trims back
        evict(&mut cache, |_| false);
        assert_eq!(cache.len(), CONTEXTS - 1);
    }
}
