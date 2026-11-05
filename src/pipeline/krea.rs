//! Krea 2: its diffusion model and encoded prompts. The text encoder and the VAE are the generator's.

use super::{run_batches, Dump, Env, FamilyState, Generator, Kind, Prompt, Reference, Sampling, SharedTe, Stats, Step, MB};
use super::{nothing_for, shared_te_loras, upload};
use crate::cuda::{self, DevBuf, Stream};
use crate::vision::{self, PromptImage};
use crate::{dit, lora, offload, sampling, st, te, vae};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Text tokens budgeted for activation planning.
const LT_MAX: usize = 512;

pub struct KreaRequest {
    pub prompt: Prompt,
    /// Only encoded when the sampling uses it.
    pub negative: Prompt,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    /// Its `images` are sampled together.
    pub sampling: Sampling,
    /// The images' noise one after another, instead of ComfyUI's for the seed and the whole batch.
    pub noise: Option<Vec<f32>>,
    /// Prompt images, through the vision tower into the text encoder.
    pub images: Vec<PromptImage>,
    pub negative_images: Vec<PromptImage>,
    /// Reference latents of the positive prompt, see [`Generator::encode_image`].
    pub references: Vec<Reference>,
    /// ComfyUI's `index_timestep_zero` rather than `index`.
    pub timestep_zero: bool,
}

/// A prompt as the text encoder saw it.
#[derive(PartialEq)]
struct PromptKey {
    prompt: Prompt,
    images: Vec<PromptImage>,
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
    /// The latest positive and negative prompts, with the conditioning and the DiT's context of each.
    prompt: Option<(PromptKey, te::Cond, DevBuf)>,
    negative: Option<(PromptKey, te::Cond, DevBuf)>,
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
        self.prompt = None;
        self.negative = None;
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

/// A Krea 2 prompt's conditioning: its chunks encoded apart and joined along the tokens.
fn krea_cond(te: &mut te::TextEncoder, prompt: &Prompt, images: &[&vision::Embeds], st: &Stream) -> Result<te::Cond> {
    let conds = prompt.chunks.iter().map(|segments| {
        let text: String = segments.iter().map(|(t, _)| t.as_str()).collect();
        let text = if images.is_empty() { text } else { te::image_prompt(&text, images.len()) };
        te.encode_with_images(&text, images, prompt.scale, st)
    }).collect::<Result<Vec<_>>>()?;
    te::Cond::concat(conds, st)
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
        let k = self.family.krea().context("no diffusion model loaded")?;
        let m = k.model.as_mut().context("no diffusion model loaded")?;
        let te = self.te.as_mut().context("no text encoder loaded")?;
        let t0 = Instant::now();
        let pos_vision = te.vision_embeds(&r.images, env, dump.as_deref_mut())?;
        let neg_vision = if guided { te.vision_embeds(&r.negative_images, env, None)? } else { Vec::new() };
        let vision_ms = t0.elapsed().as_secs_f64() * 1e3;
        let ref_grids: Vec<(usize, usize)> = r.references.iter().map(|l| (l.height.div_ceil(dit::PATCH), l.width.div_ceil(dit::PATCH))).collect();
        let extra = pos_vision.iter().chain(&neg_vision).map(|e| e.tokens).max().unwrap_or(0) * r.images.len().max(r.negative_images.len())
            + ref_grids.iter().map(|(h, w)| h * w).sum::<usize>();
        let images = r.sampling.images;
        let together = env.batch_fit(&m.dit.blocks, images, 0, step_need(r.width, r.height, extra))?;
        env.plan(&mut m.dit.blocks, &m.st, activation_need(r.width, r.height, extra, together))?;
        let st = &env.st;
        let mut stats = Stats::default();
        let (lh, lw) = (r.height / 8, r.width / 8);

        // text encoding (reused while the prompt is unchanged)
        let t0 = Instant::now();
        let refs: Vec<&vision::Embeds> = pos_vision.iter().map(|e| &**e).collect();
        let key = PromptKey { prompt: r.prompt.clone(), images: r.images.clone() };
        if k.prompt.as_ref().is_none_or(|(p, _, _)| *p != key) {
            drop(k.prompt.take());
            let cond = krea_cond(&mut te.enc, &r.prompt, &refs, st)?;
            let ctx = m.dit.encode_context(&cond.ctx, cond.len, st)?;
            st.sync()?;
            eprintln!("text: {} tokens ({} conditioning) in {:.0} ms", cond.token_ids.len(), cond.len,
                t0.elapsed().as_secs_f64() * 1e3);
            k.prompt = Some((key, cond, ctx));
        }
        let key = PromptKey { prompt: r.negative.clone(), images: r.negative_images.clone() };
        if guided && k.negative.as_ref().is_none_or(|(p, _, _)| *p != key) {
            drop(k.negative.take());
            let refs: Vec<&vision::Embeds> = neg_vision.iter().map(|e| &**e).collect();
            let cond = krea_cond(&mut te.enc, &r.negative, &refs, st)?;
            let ctx = m.dit.encode_context(&cond.ctx, cond.len, st)?;
            st.sync()?;
            k.negative = Some((key, cond, ctx));
        }
        stats.text_ms = vision_ms + t0.elapsed().as_secs_f64() * 1e3;
        let (_, cond, ctx) = k.prompt.as_ref().unwrap();
        if let Some(d) = dump.as_deref_mut() {
            let mut b = vec![0u8; cond.ctx.size()];
            cond.ctx.download(0, &mut b, st)?;
            d.insert("cond".into(), ("BF16".into(), vec![cond.len, dit::TXT_LAYERS * dit::TXT_D], b));
            let mut b = vec![0u8; ctx.size()];
            ctx.download(0, &mut b, st)?;
            d.insert("txtmlp_out".into(), ("BF16".into(), vec![cond.len, dit::F], b));
            let ids: Vec<u8> = cond.token_ids.iter().flat_map(|&t| (t as i64).to_le_bytes()).collect();
            d.insert("token_ids".into(), ("I64".into(), vec![cond.token_ids.len()], ids));
        }

        // noise (same as ComfyUI's KSampler for the seed)
        let n = images * dit::CH * lh * lw;
        let noise = match &r.noise {
            Some(v) => v.clone(),
            None => sampling::torch_randn(r.seed, n),
        };
        if noise.len() != n {
            bail!("noise has {} values, expected {n}", noise.len());
        }
        let upload_rope = |lt: usize, grids: &[(usize, usize)]| upload(&dit::Dit::rope_table(lt, lh / 2, lw / 2, grids), st);
        let lt = cond.len;
        let rope_d = upload_rope(lt, &ref_grids)?;
        let latents: Vec<(Vec<f32>, usize, usize)> = r.references.iter().map(|l| (l.latent.clone(), l.height, l.width)).collect();
        let references = if latents.is_empty() { None } else { Some(m.dit.references(&latents, r.timestep_zero, st)?) };
        let negative = if guided { k.negative.as_ref().map(|(_, c, ctx)| (ctx, c.len)) } else { None };
        let neg_rope = negative.map(|(_, len)| upload_rope(len, &[])).transpose()?;

        // denoise
        let copied0 = m.dit.blocks.copied_bytes;
        let mut model = |x: &DevBuf, out: &DevBuf, sigma: f32, neg: bool, eval: usize, k: usize| -> Result<()> {
            if neg {
                let (c, len) = negative.unwrap();
                return m.dit.step(x, k, lh, lw, c, len, None, neg_rope.as_ref().unwrap(), sigma, sigma, st, Some(out), None);
            }
            if eval > 0 {
                env.relieve(&mut m.dit.blocks, &m.st, k * step_need(r.width, r.height, extra))?;
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
                    if dump.is_some() { Some(&mut probe) } else { None };
                m.dit.step(x, k, lh, lw, ctx, lt, references.as_ref(), &rope_d, sigma, sigma, st, Some(out), probe_ref)?;
            }
            if let Some(d) = dump.as_deref_mut() {
                for (name, v) in probe_store {
                    let shape = if name == "dit_out_tokens" { vec![(lh / 2) * (lw / 2), 64] } else { vec![v.len() / 2 / dit::F, dit::F] };
                    d.insert(name, ("BF16".into(), shape, v));
                }
            }
            Ok(())
        };
        let latent = run_batches(&r.sampling, &noise, together, st, &mut stats, &mut model, &mut progress)?;
        stats.streamed_gb = (m.dit.blocks.copied_bytes - copied0) as f64 / 1e9;
        eprintln!("{:.2} GB streamed over PCIe", stats.streamed_gb);
        if let Some(d) = dump {
            d.insert("latent_final".into(), ("F32".into(), vec![dit::CH, lh, lw], st::f32_bytes(&latent).to_vec()));
        }
        Ok((latent, stats))
    }
}
