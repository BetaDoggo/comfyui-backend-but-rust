//! SD 1.x and SDXL: a checkpoint's UNet, its CLIP text encoders and its VAE, and the prompts encoded through them.

use super::{regions, Side, crop, nothing_for, remember, run_batches, touch, upload, PLAIN};
use super::{Cond, Dump, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Stats, Step, CONTEXTS};
use crate::clip::{self, Clip};
use crate::cuda::Stream;
use crate::tokenizer::{clip_chunks, ClipTokenizer, CLIP_CHUNK, CLIP_END, CLIP_START};
use crate::unet::{self, Unet};
use crate::{lora, offload, sdvae, st};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// bigG pads its chunks with token 0 (`SDXLClipGTokenizer`'s `pad_with_end=False`).
const G_PAD: u32 = 0;

/// SDXL's size conditioning (`CLIPTextEncodeSDXL`'s inputs); a plain prompt has the latent's size, no crop, and
/// targets that size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeCond {
    pub width: f32,
    pub height: f32,
    pub crop_w: f32,
    pub crop_h: f32,
    pub target_width: f32,
    pub target_height: f32,
}

impl SizeCond {
    /// What `encode_model_conds` gives a prompt with no size: the picture's.
    pub fn of(width: usize, height: usize) -> Self {
        let (w, h) = (width as f32, height as f32);
        SizeCond { width: w, height: h, crop_w: 0.0, crop_h: 0.0, target_width: w, target_height: h }
    }
}

pub struct SdRequest {
    /// The positive's conditioning entries, each its own model evaluation while it is active.
    pub positive: Vec<Cond>,
    /// Only encoded when the sampling uses it.
    pub negative: Vec<Cond>,
    pub width: usize,
    pub height: usize,
    /// SDXL: each entry's size conditioning, one per entry of `positive` and of `negative`; `encode_adm` reads them
    /// per entry, with the entry's pooled output.
    pub positive_size: Vec<SizeCond>,
    pub negative_size: Vec<SizeCond>,
    /// `CLIPSetLastLayer`'s `stop_at_clip_layer` (-1 is the last layer), if the graph has one.
    pub clip_layer: Option<i32>,
    /// `ModelSamplingDiscrete`'s prediction type, in place of the checkpoint's (its `v_pred` key).
    pub v_pred: Option<bool>,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise, one after another.
    pub noise: Vec<f32>,
}

struct Model {
    path: PathBuf,
    unet: Unet,
    /// From the checkpoint: ComfyUI's `v_pred` key.
    v_pred: bool,
}

/// One encoded prompt.
struct Encoded {
    ctx: unet::Context,
    /// SDXL: bigG's pooled output, for the size conditioning
    pooled: Option<Vec<f32>>,
    /// The text encoders' output, float32 [chunks * 77, context_dim], for --dump
    hidden: Vec<f32>,
}

/// What an encoded prompt depends on: the chunks, the CLIP layer, and the scale.
type Key = (Vec<Vec<(u32, f32)>>, Option<i32>, u32);

#[derive(Default)]
pub(super) struct Sd {
    model: Option<Model>,
    /// The checkpoint the text encoders came from, CLIP-L and, on SDXL, bigG.
    te: Option<(PathBuf, Clip, Option<Clip>)>,
    tokenizer: Option<ClipTokenizer>,
    vae: Option<(PathBuf, sdvae::Vae)>,
    /// Latest encoded prompts, most recent last.
    contexts: Vec<(Key, Encoded)>,
}

impl FamilyState for Sd {
    fn set_loras(&mut self, loras: &[lora::Lora], _: &Stream) -> Result<()> {
        if self.model.is_none() {
            return nothing_for(loras, "diffusion model");
        }
        if !loras.is_empty() {
            return Err(lora::Unsupported("LoRAs on SD 1.x and SDXL".into()).into());
        }
        Ok(())
    }

    fn set_text_encoder_loras(&mut self, _: Option<&mut SharedTe>, loras: &[lora::Lora], _: &Stream) -> Result<()> {
        if self.te.is_none() {
            return nothing_for(loras, "text encoder");
        }
        if !loras.is_empty() {
            return Err(lora::Unsupported("LoRAs on SD 1.x and SDXL".into()).into());
        }
        Ok(())
    }

    fn clear_prompts(&mut self) {
        self.contexts.clear();
    }

    fn set_int8_attention(&mut self, on: bool) {
        if let Some(m) = &mut self.model {
            m.unet.int8_attn = on;
        }
    }

    fn blocks(&mut self) -> Option<(&mut offload::WeightSet, &st::SafeTensors)> {
        None
    }

    fn idle(&mut self) -> Result<()> {
        Ok(())
    }
}

impl super::Family {
    fn sd(&mut self) -> Option<&mut Sd> {
        match self {
            super::Family::Sd(f) => Some(f),
            _ => None,
        }
    }
}

/// The layer to read and whether it goes through the final LayerNorm, as `SDClipModel` picks them: SD 1.x reads the
/// last layer, normed, and SDXL the one before it, not normed; `CLIPSetLastLayer` picks another, normed on SD 1.x only.
fn clip_layer(layers: usize, stop: Option<i32>, xl: bool) -> Result<(usize, bool)> {
    match stop {
        None if xl => Ok((layers - 2, false)),
        None => Ok((layers - 1, true)),
        Some(n) if n < 0 && n.unsigned_abs() as usize <= layers => Ok(((layers as i32 + n) as usize, !xl)),
        Some(n) => bail!("CLIP layer {n} of {layers}"),
    }
}

/// `ClipTokenWeightEncoder.encode_token_weights` of one encoder: the chunks' hidden states, those of weighted tokens
/// pushed away from an empty prompt's by their weight, and the first chunk's pooled output.
fn encode_weighted(enc: &Clip, chunks: &[Vec<(u32, f32)>], pad: u32, layer: (usize, bool), pooled: bool, st: &Stream) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let weighted = chunks.iter().flatten().any(|(_, w)| *w != 1.0);
    let mut ids: Vec<Vec<u32>> = chunks.iter().map(|c| c.iter().map(|(t, _)| *t).collect()).collect();
    if weighted {
        // gen_empty_tokens: start, end, then padding
        let mut empty = vec![CLIP_START, CLIP_END];
        empty.resize(CLIP_CHUNK, pad);
        ids.push(empty);
    }
    let out = enc.encode(&ids, layer.0, layer.1, pooled, st)?;
    let d = enc.width;
    let mut hidden = out.hidden;
    if weighted {
        let (body, empty) = hidden.split_at_mut(chunks.len() * CLIP_CHUNK * d);
        for (c, chunk) in chunks.iter().enumerate() {
            for (j, (_, w)) in chunk.iter().enumerate() {
                if *w != 1.0 {
                    let row = &mut body[(c * CLIP_CHUNK + j) * d..][..d];
                    for (v, z) in row.iter_mut().zip(&empty[j * d..(j + 1) * d]) {
                        *v = (*v - z) * w + z;
                    }
                }
            }
        }
        hidden.truncate(chunks.len() * CLIP_CHUNK * d);
    }
    Ok((hidden, out.pooled))
}

impl Sd {
    /// The chunks of a prompt: each `<break>` chunk tokenized apart and their 77-token chunks joined, as ComfyUI
    /// concatenates the conditionings.
    fn chunks(&self, prompt: &Prompt) -> Result<Vec<Vec<(u32, f32)>>> {
        let t = self.tokenizer.as_ref().context("no text encoder loaded")?;
        let mut out = Vec::new();
        for segments in &prompt.chunks {
            out.extend(clip_chunks(t, segments, CLIP_END)?);
        }
        Ok(out)
    }

    fn key(&self, prompt: &Prompt, layer: Option<i32>) -> Result<Key> {
        Ok((self.chunks(prompt)?, layer, prompt.scale.to_bits()))
    }

    /// Move the encoding of `key` to the end of the cache, encoding it if needed, and keep at most `keep` encodings.
    fn encode(&mut self, key: &Key, keep: usize, st: &Stream) -> Result<()> {
        if touch(&mut self.contexts, |(k, _)| k == key) {
            return Ok(());
        }
        let m = self.model.as_ref().context("no diffusion model loaded")?;
        let (_, l, g) = self.te.as_ref().context("no text encoder loaded")?;
        let t0 = Instant::now();
        let (chunks, stop, scale) = key;
        let scale = f32::from_bits(*scale);
        let xl = g.is_some();
        let (mut ctx, pooled) = match g {
            None => encode_weighted(l, chunks, CLIP_END, clip_layer(l.layers(), *stop, false)?, false, st)?,
            Some(g) => {
                let (lh, _) = encode_weighted(l, chunks, CLIP_END, clip_layer(l.layers(), *stop, true)?, false, st)?;
                // bigG's chunks pad with 0
                let g_chunks: Vec<Vec<(u32, f32)>> = chunks.iter().map(|c| {
                    let end = c.iter().position(|(t, _)| *t == CLIP_END).unwrap_or(c.len() - 1);
                    c.iter().enumerate().map(|(i, &(t, w))| if i > end { (G_PAD, w) } else { (t, w) }).collect()
                }).collect();
                let (gh, pooled) = encode_weighted(g, &g_chunks, G_PAD, clip_layer(g.layers(), *stop, true)?, true, st)?;
                let rows = chunks.len() * CLIP_CHUNK;
                let (dl, dg) = (l.width, g.width);
                let mut ctx = Vec::with_capacity(rows * (dl + dg));
                for r in 0..rows {
                    ctx.extend_from_slice(&lh[r * dl..(r + 1) * dl]);
                    ctx.extend_from_slice(&gh[r * dg..(r + 1) * dg]);
                }
                (ctx, pooled)
            }
        };
        let pooled = pooled.map(|p| p.iter().map(|v| v * scale).collect::<Vec<_>>());
        if scale != 1.0 {
            ctx.iter_mut().for_each(|v| *v *= scale);
        }
        if ctx.len() / (chunks.len() * CLIP_CHUNK) != m.unet.context_dim {
            bail!("the text encoders give {} wide prompts, the UNet reads {}", ctx.len() / (chunks.len() * CLIP_CHUNK), m.unet.context_dim);
        }
        let e = Encoded { ctx: m.unet.context(&ctx, st)?, pooled: xl.then(|| pooled).flatten(), hidden: ctx };
        st.sync()?;
        eprintln!("text: {} chunk(s) in {:.0} ms", chunks.len(), t0.elapsed().as_secs_f64() * 1e3);
        remember(&mut self.contexts, (key.clone(), e), keep);
        Ok(())
    }

    /// Encode every key a run needs. They all stay cached for the run: the cache keeps at least as many as there are,
    /// and drops back to [`CONTEXTS`] on the next run that needs fewer.
    fn encode_all(&mut self, keys: &[&Key], st: &Stream) -> Result<()> {
        let keep = CONTEXTS.max((0..keys.len()).filter(|&i| !keys[..i].contains(&keys[i])).count());
        while self.contexts.len() > keep {
            self.contexts.remove(0);
        }
        for key in keys {
            self.encode(key, keep, st)?;
        }
        Ok(())
    }
}

/// True for a checkpoint of either family: its UNet's keys.
pub fn is_sd(has: impl Fn(&str) -> bool) -> bool {
    unet::key_prefix(has).is_some()
}

impl Generator {
    /// Load an SD 1.x or SDXL checkpoint unless `path` is loaded: its UNet, and its text encoders and VAE when it has
    /// them. The UNet's transformer linears are quantized to int8 as they load unless `int8_weights` is off.
    pub fn ensure_sd(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Sd)?;
        let env = &self.env;
        let f = self.family.sd().context("SD not entered")?;
        if f.model.as_ref().is_some_and(|m| m.path == path) {
            return Ok(());
        }
        env.st.sync()?;
        f.clear_prompts();
        f.model = None;
        let t0 = Instant::now();
        let st = st::SafeTensors::open(path)?;
        let mut unet = Unet::load(&st, env.int8_weights, &env.st)?;
        unet.int8_attn = env.int8_attn;
        let xl = unet.adm > 0;
        eprintln!("{} UNet ready in {:.1}s ({})", if xl { "SDXL" } else { "SD 1.x" }, t0.elapsed().as_secs_f64(),
            if env.int8_weights { "int8 linears" } else { "fp16" });
        let v_pred = unet::is_v_pred(&st);
        f.model = Some(Model { path: path.to_path_buf(), unet, v_pred });
        if clip::has_clip_l(&st) && f.te.as_ref().map_or(true, |t| t.0 != path) {
            let t1 = Instant::now();
            f.te = None;
            let l = Clip::load(&st, 768, &env.st)?;
            let g = if xl { Some(Clip::load(&st, 1280, &env.st)?) } else { None };
            f.te = Some((path.to_path_buf(), l, g));
            if f.tokenizer.is_none() {
                f.tokenizer = Some(ClipTokenizer::new()?);
            }
            eprintln!("text encoders ready in {:.1}s", t1.elapsed().as_secs_f64());
        }
        // a VAE already loaded stays, the checkpoint's or one picked apart; ensure_sd_vae changes it
        if sdvae::key_prefix(|k| st.has(k)).is_some() && f.vae.is_none() {
            let scale = if xl { sdvae::SDXL_SCALE } else { sdvae::SD1_SCALE };
            f.vae = Some((path.to_path_buf(), sdvae::Vae::from(&st, scale, &env.st)?));
        }
        Ok(())
    }

    /// Load an SD VAE file in place of the checkpoint's, unless `path` is loaded.
    pub fn ensure_sd_vae(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Sd)?;
        let st = &self.env.st;
        let f = self.family.sd().context("SD not entered")?;
        if f.vae.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        let xl = f.model.as_ref().map(|m| m.unet.adm > 0).or(f.te.as_ref().map(|t| t.2.is_some())).unwrap_or(false);
        f.vae = None;
        let scale = if xl { sdvae::SDXL_SCALE } else { sdvae::SD1_SCALE };
        f.vae = Some((path.to_path_buf(), sdvae::Vae::load(path, scale, st)?));
        Ok(())
    }

    /// The loaded SD VAE's latent scale: [`Generator::decode_sd`] divides by it and [`Generator::encode_sd`] multiplies.
    pub fn sd_vae_scale(&mut self) -> Option<f32> {
        self.family.sd().and_then(|f| f.vae.as_ref()).map(|v| v.1.scale)
    }

    /// True when the loaded SD checkpoint is SDXL.
    pub fn sd_is_xl(&mut self) -> bool {
        self.family.sd().and_then(|f| f.model.as_ref()).is_some_and(|m| m.unet.adm > 0)
    }

    /// SD 1.x or SDXL latents, sampled as ComfyUI's KSampler does.
    /// `dump` gets the positive prompt's `cond` and `pooled`, and `denoised0`, the first evaluation's prediction.
    pub fn sample_sd(&mut self, r: &SdRequest, mut progress: impl FnMut(Step) -> bool, mut dump: Option<&mut Dump>) -> Result<(Vec<f32>, Stats)> {
        let env = &self.env;
        // any multiple of 8, as in ComfyUI: the UNet rounds odd sides up on the way down and crops on the way up
        env.check_size_of(r.width, r.height, 8)?;
        let (lh, lw) = (r.height / 8, r.width / 8);
        let images = r.sampling.images;
        let n = images * unet::CH * lh * lw;
        if r.noise.len() != n {
            bail!("noise has {} values, expected {n}", r.noise.len());
        }
        let st = &env.st;
        let f = self.family.sd().context("no diffusion model loaded")?;
        let mut stats = Stats::default();
        let t0 = Instant::now();
        let guided = r.sampling.uses_negative();
        if r.positive.is_empty() || (guided && r.negative.is_empty()) {
            bail!("a prompt without conditioning entries");
        }
        if r.positive_size.len() != r.positive.len() || (guided && r.negative_size.len() != r.negative.len()) {
            bail!("{} and {} size conditionings for {} and {} prompt entries", r.positive_size.len(), r.negative_size.len(),
                r.positive.len(), r.negative.len());
        }
        let pos_keys = r.positive.iter().map(|c| f.key(&c.prompt, r.clip_layer)).collect::<Result<Vec<_>>>()?;
        let neg_keys = if guided { r.negative.iter().map(|c| f.key(&c.prompt, r.clip_layer)).collect::<Result<Vec<_>>>()? } else { Vec::new() };
        f.encode_all(&pos_keys.iter().chain(&neg_keys).collect::<Vec<_>>(), st)?;
        stats.text_ms = t0.elapsed().as_secs_f64() * 1e3;
        let m = f.model.as_ref().context("no diffusion model loaded")?;
        let find = |k: &Key| f.contexts.iter().find(|(c, _)| c == k).map(|(_, e)| e).unwrap();
        // each entry with its own `adm`, from its pooled output and its size conditioning
        let entries = |keys: &[Key], sizes: &[SizeCond]| -> Vec<(&Encoded, Option<Vec<f32>>)> {
            keys.iter().zip(sizes).map(|(k, s)| {
                let e = find(k);
                let adm = e.pooled.as_ref().map(|p| unet::sdxl_adm(p, s.width, s.height, s.crop_w, s.crop_h, s.target_width, s.target_height));
                (e, adm)
            }).collect()
        };
        let pos = entries(&pos_keys, &r.positive_size);
        let neg = entries(&neg_keys, &r.negative_size);
        if let Some(d) = dump.as_deref_mut() {
            let width = m.unet.context_dim;
            let first = pos[0].0;
            d.insert("cond".into(), ("F32".into(), vec![first.hidden.len() / width, width], st::f32_bytes(&first.hidden).to_vec()));
            if let Some(p) = &first.pooled {
                d.insert("pooled".into(), ("F32".into(), vec![p.len()], st::f32_bytes(p).to_vec()));
            }
        }
        let ms = r.sampling.model_sampling;
        let v_pred = r.v_pred.unwrap_or(m.v_pred);
        let mut model = |x: &crate::cuda::DevBuf, out: &crate::cuda::DevBuf, sigma: f32, side: Side, eval: usize, k: usize| -> Result<()> {
            let (e, adm) = if side.negative { &neg[side.entry] } else { &pos[side.entry] };
            m.unet.predict(x, k, lh, lw, &e.ctx, adm.as_deref(), sigma, ms.timestep(sigma), v_pred, out, st)?;
            if let (Some(d), 0, Side::POSITIVE) = (dump.as_deref_mut(), eval, side) {
                let mut v = vec![0u8; k * unet::CH * lh * lw * 4];
                out.download(0, &mut v, st)?;
                d.insert("denoised0".into(), ("F32".into(), vec![k, unet::CH, lh, lw], v));
            }
            Ok(())
        };
        let together = images;
        let neg_regions = if guided { regions(&r.negative) } else { PLAIN.to_vec() };
        let latent = run_batches(&r.sampling, &r.noise, [&regions(&r.positive), &neg_regions], together, st, &mut stats, &mut model, &mut progress)?;
        Ok((latent, stats))
    }

    /// Decode a host SD latent ([4, H/8, W/8] f32, at the sampler's scale) to RGB8 pixels.
    pub fn decode_sd(&mut self, latent: &[f32], width: usize, height: usize) -> Result<Vec<u8>> {
        let (lh, lw) = (height / 8, width / 8);
        if latent.len() != unet::CH * lh * lw {
            bail!("latent has {} values, expected {}", latent.len(), unet::CH * lh * lw);
        }
        let st = &self.env.st;
        let vae = &self.family.sd().and_then(|f| f.vae.as_ref()).context("no VAE loaded")?.1;
        vae.decode(&upload(latent, st)?, lh, lw, st)
    }

    /// Encode RGB f32 [height, width, 3], 0 to 1, into an SD latent at the sampler's scale. Pixels past a multiple of 8
    /// are cropped evenly from each side, as ComfyUI's VAE does.
    pub fn encode_sd(&mut self, rgb: &[f32], width: usize, height: usize) -> Result<Reference> {
        let (cropped, cw, ch) = crop(rgb, width, height, 3, 8)?;
        let st = &self.env.st;
        let vae = &self.family.sd().and_then(|f| f.vae.as_ref()).context("no VAE loaded")?.1;
        if !vae.has_encoder() {
            bail!("this VAE has no encoder");
        }
        let t0 = Instant::now();
        let latent = vae.encode(&cropped, ch, cw, st)?;
        eprintln!("VAE encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        Ok(Reference { latent, height: ch / 8, width: cw / 8 })
    }
}
