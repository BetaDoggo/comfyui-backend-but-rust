//! RelayVSR (ComfyUI's `comfy.ldm.wan.model_vsr`): video super-resolution in two networks.
//!
//! The keyframe model ([`Dit`]) is Wan 2.2 TI2V 5B with the low-quality frames projected in beside the patch
//! embedding. It runs one step over every 15th frame (the keyframes, and the last frame), each keyframe a latent frame
//! of its own. Self-attention is local: a token sees a 22x40 window of its own keyframe's tokens and of the three
//! keyframes before it, the window shifted inward at the border. Keyframes go through in chunks of about 32768 tokens;
//! each block keeps the last three keyframes' keys and values for the next chunk, as int8 rows with a bf16 scale each.
//! The cross-attention reads a fixed prompt stored in the checkpoint, so its keys and values are computed once.
//!
//! The Flash decoder ([`Flash`]), loaded as the VAE, restores every frame: a small transformer over the latent grid of
//! one low-quality frame attends windows of the two keyframe latents around it, of the previous frame and of itself,
//! and its output, pixel-shuffled, is added to a bicubic upscale of the frame. It runs in fp16, as ComfyUI picks for
//! this VAE on this card.

use crate::cuda::{memcpy_2d_d2d, memcpy_d2d, DevBuf, Stream};
use crate::ffi::{
    k, null, off, s, WinArgs, DT_BF16, DT_F16, DT_F32, P, PM, PRO_GELU_TANH, PRO_LN_AFFINE, PRO_LN_MOD, PRO_PLAIN, PRO_SILU,
};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::st::{bf16_bytes, f16_bytes, f32_bytes, Dt, SafeTensors};
use anyhow::{bail, Context, Result};
use half::bf16;
use std::collections::HashMap;

/// Latent channels (Wan 2.2's VAE layout).
pub const CH: usize = 48;
/// Frames between keyframes.
pub const KEYFRAME_INTERVAL: usize = 15;
/// Keyframes each keyframe's tokens attend, its own included.
const CONTEXT: usize = 4;
/// The keyframe model's attention window (rows, columns of tokens).
const WINDOW: (usize, usize) = (22, 40);
const CHUNK_TOKENS: usize = 32768;
/// A clip this many keyframes longer than a chunk is chunked: the per-block cache costs about as much VRAM.
const CACHE_KEYFRAMES: usize = 7;
const HD: usize = 128;
const EPS: f32 = 1e-6;
const FREQ: usize = 256;
/// `ModelSamplingDiscreteFlow`'s multiplier.
const MULTIPLIER: f32 = 1000.0;
/// Rows through the feed-forward at a time.
const FFN_CHUNK: usize = 16384;
/// Per-sigma modulations kept.
const MOD_CACHE: usize = 8;

/// `relay_keyframes`: every 15th frame and the last.
pub fn keyframes(frames: usize) -> Vec<usize> {
    let mut k: Vec<usize> = (0..frames.saturating_sub(1)).step_by(KEYFRAME_INTERVAL).collect();
    k.push(frames.saturating_sub(1));
    k
}

/// `relay_latent_size`: the latent of a `height` x `width` low-quality frame, whole 2x2 patches of `latent_stride`
/// low-quality pixels per latent pixel.
pub fn latent_size(height: usize, width: usize, latent_stride: usize) -> (usize, usize) {
    let s = latent_stride * 2;
    (height.div_ceil(s) * 2, width.div_ceil(s) * 2)
}

/// The keyframe model's key prefix, if `has` names its tensors.
pub fn key_prefix(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    ["model.diffusion_model.", ""].into_iter().find(|p| has(&format!("{p}lq_proj_in.conv1.weight")) && has(&format!("{p}patch_embedding.weight")))
}

/// The Flash decoder, as ComfyUI's VAE loader tells it apart.
pub fn is_flash(has: impl Fn(&str) -> bool) -> bool {
    has("lr_encoder.layers.0.weight") && has("condition_encoder.projection.weight")
}

/// The upscale factor of a keyframe model: its low-quality frames go in pixel-unshuffled by 4 times it.
pub fn factor_of(conv1_in: usize) -> Result<usize> {
    let r = ((conv1_in / 3) as f64).sqrt().round() as usize;
    if r * r * 3 != conv1_in || r % 4 != 0 || !matches!(r / 4, 2 | 4) {
        bail!("RelayVSR's low-quality projection takes {conv1_in} channels, not a x2 or x4 model's");
    }
    Ok(r / 4)
}

/// The Flash decoder's upscale factor, from its `config` metadata (the x2 and x4 files have the same weight shapes).
pub fn flash_factor(metadata: &HashMap<String, String>) -> Result<usize> {
    let cfg: serde_json::Value = serde_json::from_str(metadata.get("config").map(String::as_str).unwrap_or("{}")).context("RelayVSR Flash config")?;
    match cfg["upscale_factor"].as_u64() {
        Some(f @ (2 | 4)) => Ok(f as usize),
        Some(f) => bail!("RelayVSR Flash decoder for x{f}"),
        None => bail!("RelayVSR Flash decoder without its upscale_factor metadata, use the released relayvsr_flash_x2/x4 file"),
    }
}

fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn upload(bytes: &[u8], st: &Stream) -> Result<DevBuf> {
    let d = DevBuf::alloc(bytes.len().max(4))?;
    d.upload(0, bytes, st)?;
    Ok(d)
}

/// `rope` of one axis: (cos, sin) of `pos * 10000^(-2i / dim)` for each of `dim / 2` pairs, in f64 like ComfyUI's.
fn rope_axis(pos: f32, dim: usize) -> Vec<f32> {
    (0..dim / 2)
        .flat_map(|i| {
            let omega = 1.0 / 10000f64.powf(2.0 * i as f64 / dim as f64);
            let a = pos as f64 * omega;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect()
}

/// A table of `rope_axis` for each position.
fn rope_table(positions: impl Iterator<Item = f32>, dim: usize) -> Vec<f32> {
    positions.flat_map(|p| rope_axis(p, dim)).collect()
}

/// A linear layer in a unit: int8 ConvRot (weight, per-row scale) or bf16, with a bias.
#[derive(Clone, Copy, Debug)]
struct Lin {
    int8: bool,
    n: usize,
    k: usize,
}

/// The int8 ConvRot form of `name` (a `.weight` prefix), refusing other quantizations.
fn is_int8(st: &SafeTensors, name: &str) -> Result<bool> {
    let w = st.info(&format!("{name}.weight"))?;
    if w.dtype != Dt::I8 {
        return Ok(false);
    }
    let q = format!("{name}.comfy_quant");
    if st.has(&q) {
        let cfg: serde_json::Value = serde_json::from_slice(st.bytes(&q)?).context("comfy_quant")?;
        if cfg["convrot"] != true || cfg["convrot_groupsize"].as_u64() != Some(256) || cfg["format"].as_str().is_some_and(|f| !f.starts_with("int8")) {
            bail!("{name}: only int8 ConvRot (groups of 256) weights run here, not {cfg}");
        }
    }
    Ok(true)
}

/// `name.weight_scale` as one float per output row.
fn row_scales(st: &SafeTensors, name: &str, rows: usize) -> Result<Vec<f32>> {
    let s = st.f32_vec(&format!("{name}.weight_scale"), false)?;
    match s.len() {
        1 => Ok(vec![s[0]; rows]),
        n if n == rows => Ok(s),
        n => bail!("{name}: {n} weight scales for {rows} rows"),
    }
}

/// Add the linear layers `names` (prefixes of `.weight`, `.weight_scale`, `.bias`), concatenated along their outputs,
/// as `key.w`, `key.s` and `key.b`.
fn add_lin(l: &mut Layout, st: &SafeTensors, key: &str, names: &[String]) -> Result<Lin> {
    let int8 = is_int8(st, &names[0])?;
    let mut n = 0;
    let mut kk = 0;
    for name in names {
        if is_int8(st, name)? != int8 {
            bail!("{name}: mixed int8 and float layers in one GEMM");
        }
        let i = st.info(&format!("{name}.weight"))?;
        if i.shape.len() != 2 {
            bail!("{name}: a {:?} linear weight", i.shape);
        }
        if kk != 0 && i.shape[1] != kk {
            bail!("{name}: {} inputs, expected {kk}", i.shape[1]);
        }
        kk = i.shape[1];
        n += i.shape[0];
    }
    if int8 {
        if kk % 256 != 0 {
            bail!("{}: int8 ConvRot layers need inputs in groups of 256, not {kk}", names[0]);
        }
        l.raw(st, &format!("{key}.w"), &names.iter().map(|n| format!("{n}.weight")).collect::<Vec<_>>())?;
        let mut sc = Vec::with_capacity(n);
        for name in names {
            sc.extend(row_scales(st, name, st.info(&format!("{name}.weight"))?.shape[0])?);
        }
        l.bytes(&format!("{key}.s"), f32_bytes(&sc));
    } else {
        l.bf16(st, &format!("{key}.w"), &names.iter().map(|n| format!("{n}.weight")).collect::<Vec<_>>())?;
    }
    l.bf16(st, &format!("{key}.b"), &names.iter().map(|n| format!("{n}.bias")).collect::<Vec<_>>())?;
    Ok(Lin { int8, n, k: kk })
}

/// `name` as f32 values rounded through bf16, the model dtype ComfyUI holds it in.
fn norm_bytes(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(f32_bytes(&st.f32_vec(name, true)?))
}

/// `y = act(x) w^T + b` for `rows` rows of bf16 `x` (row stride `x_rs` elements), into bf16 `out` (rows of `lin.n`).
/// `pro` is the prologue the int8 quantizer fuses (`PRO_PLAIN`, `PRO_SILU`, `PRO_GELU_TANH`).
#[allow(clippy::too_many_arguments)]
fn run_lin(lin: &Lin, w: P, sc: P, b: P, x: P, x_rs: usize, rows: usize, pro: i32, out: PM, st: &Stream) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    if lin.int8 {
        let a8 = DevBuf::alloc_async(rows * lin.k, st)?;
        let asc = DevBuf::alloc_async(rows * 4, st)?;
        k!(gpu_quant_rows(DT_BF16, pro, x, x_rs as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), rows as i32, lin.k as i32, 1,
            s(st)))?;
        return k!(gpu_int8_gemm(a8.ptr(), w, asc.ptr(), sc, b, out, rows as i64, lin.n as i64, lin.k as i64, DT_BF16, s(st)));
    }
    let (src, src_rs, _tmp) = if pro == PRO_PLAIN {
        (x, x_rs, None)
    } else {
        let t = DevBuf::alloc_async(rows * lin.k * 2, st)?;
        k!(gpu_prologue_rows_b(DT_BF16, pro, x, x_rs as i64, null(), null(), 0.0, null(), null(), t.ptr(), lin.k as i64, rows as i32, lin.k as i32,
            s(st)))?;
        (t.ptr() as P, lin.k, Some(t))
    };
    k!(gpu_gemm(src, src_rs as i64, w, lin.k as i64, out, lin.n as i64, b, rows as i64, lin.n as i64, lin.k as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0,
        s(st)))
}

/// The keyframe model.
pub struct Dit {
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    /// Upscale factor, 2 or 4.
    pub factor: usize,
    dim: usize,
    heads: usize,
    ffn: usize,
    lq_in: usize,
    lins: HashMap<&'static str, Lin>,
    /// bf16 values of each block's `modulation` [6][dim], and the head's [2][dim]
    block_mod: Vec<Vec<f32>>,
    head_mod: Vec<f32>,
    /// Cross-attention keys and values of the stored prompt: bf16 [blocks][k, v][text][dim]
    ctx_kv: DevBuf,
    text_len: usize,
    /// bf16 [blocks][6][dim] then the head's [2][dim], by sigma bits
    mods: HashMap<u32, DevBuf>,
    pub int8_attn: bool,
    /// When set, `predict` keeps the first chunk's `lq_proj_in` output and block outputs here, f32 [rows][dim], for
    /// `--relayvsr-check`.
    pub probes: Option<Vec<(String, Vec<f32>)>>,
}

impl Dit {
    /// Pixels of the low-quality frames per latent pixel.
    pub fn latent_stride(&self) -> usize {
        16 / self.factor
    }

    pub fn load(st: &SafeTensors, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let p = key_prefix(|k| st.has(k)).context("not a RelayVSR model")?;
        let n = |x: &str| format!("{p}{x}");
        let pe = st.info(&n("patch_embedding.weight"))?.shape.clone();
        if pe.len() != 5 || pe[1] != CH || pe[2..] != [1, 2, 2] {
            bail!("RelayVSR patch embedding of shape {pe:?}");
        }
        let dim = pe[0];
        if dim % HD != 0 {
            bail!("RelayVSR width {dim} isn't whole heads");
        }
        let heads = dim / HD;
        let nb = (0..).take_while(|i| st.has(&n(&format!("blocks.{i}.self_attn.q.weight")))).count();
        let ffn = st.info(&n("blocks.0.ffn.0.weight"))?.shape[0];
        let lq_in = st.info(&n("lq_proj_in.conv1.weight"))?.shape[1];
        let factor = factor_of(lq_in)?;

        // everything outside the blocks
        let mut gl = Layout::new();
        let mut lins = HashMap::new();
        for (key, name) in [("tx1", "text_embedding.0"), ("tx2", "text_embedding.2"), ("te1", "time_embedding.0"), ("te2", "time_embedding.2"),
            ("tp", "time_projection.1"), ("hd", "head.head"), ("lql", "lq_proj_in.linear")] {
            lins.insert(key, add_lin(&mut gl, st, key, &[n(name)])?);
        }
        let flat = |name: &str| -> Result<Vec<f32>> {
            if st.info(name)?.dtype == Dt::I8 {
                bail!("{name}: an int8 convolution");
            }
            st.f32_vec(name, false)
        };
        // the patch embedding runs in f32
        gl.bytes("pe.w", f32_bytes(&flat(&n("patch_embedding.weight"))?));
        gl.bytes("pe.b", f32_bytes(&flat(&n("patch_embedding.bias"))?));
        // the 1x1 conv as a GEMM, the 3x3 one with its taps outermost to match the im2col's columns
        let c1 = st.info(&n("lq_proj_in.conv1.weight"))?.shape.clone();
        gl.bytes("c1.w", bf16_bytes(&flat(&n("lq_proj_in.conv1.weight"))?));
        gl.bytes("c1.b", bf16_bytes(&flat(&n("lq_proj_in.conv1.bias"))?));
        gl.bytes("n1", norm_bytes(st, &n("lq_proj_in.norm1.weight"))?);
        let c2s = st.info(&n("lq_proj_in.conv2.weight"))?.shape.clone();
        if c2s[1] != c1[0] || c2s[2..] != [3, 3] {
            bail!("RelayVSR lq_proj_in.conv2 of shape {c2s:?}");
        }
        let (c2o, c2i) = (c2s[0], c2s[1]);
        let w2 = flat(&n("lq_proj_in.conv2.weight"))?;
        let mut w2r = vec![0f32; w2.len()];
        for o in 0..c2o {
            for c in 0..c2i {
                for t in 0..9 {
                    w2r[o * 9 * c2i + t * c2i + c] = w2[(o * c2i + c) * 9 + t];
                }
            }
        }
        gl.bytes("c2.w", bf16_bytes(&w2r));
        gl.bytes("c2.b", bf16_bytes(&flat(&n("lq_proj_in.conv2.bias"))?));
        gl.bytes("n2", norm_bytes(st, &n("lq_proj_in.norm2.weight"))?);
        gl.bf16(st, "pos", &[n("positive_conditioning")])?;
        gl.bytes("ones", bf16_bytes(&vec![1.0f32; dim]));
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        drop(host);

        let mut layouts = Vec::with_capacity(nb);
        let mut block_mod = Vec::with_capacity(nb);
        for i in 0..nb {
            let b = |x: &str| n(&format!("blocks.{i}.{x}"));
            let mut l = Layout::new();
            // (key, layers, outputs, inputs): the blocks run int8 GEMMs only
            for (key, names, rows, cols) in [
                ("qkv", vec![b("self_attn.q"), b("self_attn.k"), b("self_attn.v")], 3 * dim, dim),
                ("o", vec![b("self_attn.o")], dim, dim),
                ("cq", vec![b("cross_attn.q")], dim, dim),
                ("co", vec![b("cross_attn.o")], dim, dim),
                ("f1", vec![b("ffn.0")], ffn, dim),
                ("f2", vec![b("ffn.2")], dim, ffn),
            ] {
                let lin = add_lin(&mut l, st, key, &names)?;
                if !lin.int8 {
                    bail!("{}: this backend runs the int8 ConvRot RelayVSR release", names[0]);
                }
                if (lin.n, lin.k) != (rows, cols) {
                    bail!("{}: {}x{}, expected {rows}x{cols}", names[0], lin.n, lin.k);
                }
            }
            for (key, name) in [("nq", "self_attn.norm_q.weight"), ("nk", "self_attn.norm_k.weight"), ("cnq", "cross_attn.norm_q.weight"),
                ("n3w", "norm3.weight"), ("n3b", "norm3.bias")] {
                l.bytes(key, norm_bytes(st, &b(name))?);
            }
            layouts.push(l);
            block_mod.push(st.f32_vec(&b("modulation"), true)?);
        }
        let head_mod = st.f32_vec(&n("head.modulation"), true)?;
        let text_len = st.info(&n("positive_conditioning"))?.shape[0];
        let txt_dim = st.info(&n("positive_conditioning"))?.shape[1];

        let pl = plan(&layouts, block_budget, slots);
        let mut dit = Dit {
            globals,
            gl,
            blocks: WeightSet::load("relayvsr blocks", st, layouts, pl, threads)?,
            factor,
            dim,
            heads,
            ffn,
            lq_in,
            lins,
            block_mod,
            head_mod,
            ctx_kv: DevBuf::alloc(4)?,
            text_len,
            mods: HashMap::new(),
            int8_attn: true,
            probes: None,
        };
        dit.ctx_kv = dit.prompt_kv(st, p, txt_dim, s0)?;
        s0.sync()?;
        Ok(dit)
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    fn glin(&self, key: &'static str, x: P, x_rs: usize, rows: usize, pro: i32, out: PM, st: &Stream) -> Result<()> {
        let l = self.lins[key];
        let sc = if l.int8 { self.g(&format!("{key}.s")) as P } else { null() };
        run_lin(&l, self.g(&format!("{key}.w")), sc, self.g(&format!("{key}.b")), x, x_rs, rows, pro, out, st)
    }

    /// Everywhere this model keeps weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Buf(&self.globals, &self.gl)]
    }

    /// The stored prompt through `text_embedding`, then each block's cross-attention keys (normalized) and values.
    fn prompt_kv(&self, st: &SafeTensors, p: &str, txt_dim: usize, s0: &Stream) -> Result<DevBuf> {
        let (d, lt) = (self.dim, self.text_len);
        let h = DevBuf::alloc(lt * d * 2)?;
        let ctx = DevBuf::alloc(lt * d * 2)?;
        self.glin("tx1", self.g("pos"), txt_dim, lt, PRO_PLAIN, h.ptr(), s0)?;
        self.glin("tx2", h.ptr(), d, lt, PRO_GELU_TANH, ctx.ptr(), s0)?;
        let nb = self.blocks.len();
        let out = DevBuf::alloc(nb * 2 * lt * d * 2)?;
        for b in 0..nb {
            let name = |x: &str| format!("{p}blocks.{b}.cross_attn.{x}");
            let mut l = Layout::new();
            let lk = add_lin(&mut l, st, "k", &[name("k")])?;
            let lv = add_lin(&mut l, st, "v", &[name("v")])?;
            l.bytes("nk", norm_bytes(st, &name("norm_k.weight"))?);
            let mut host = vec![0u8; l.size];
            l.pack(st, &mut host)?;
            let w = DevBuf::alloc(l.size)?;
            w.upload(0, &host, s0)?;
            let at = |key: &str| w.at(l.off(key));
            let sc = |lin: &Lin, key: &str| if lin.int8 { at(key) as P } else { null() };
            let kp = off(out.ptr(), (b * 2) * lt * d * 2);
            let vp = off(out.ptr(), (b * 2 + 1) * lt * d * 2);
            run_lin(&lk, at("k.w"), sc(&lk, "k.s"), at("k.b"), ctx.ptr(), d, lt, PRO_PLAIN, kp, s0)?;
            k!(gpu_relay_rms(DT_BF16, kp, d as i64, kp, d as i64, at("nk"), null(), EPS, lt as i64, d as i32, 1, s(s0)))?;
            run_lin(&lv, at("v.w"), sc(&lv, "v.s"), at("v.b"), ctx.ptr(), d, lt, PRO_PLAIN, vp, s0)?;
            s0.sync()?;
        }
        Ok(out)
    }

    /// The modulations for timestep `sigma * 1000`: each block's `modulation + time_projection(e)` and the head's
    /// `modulation + e`, in bf16 like ComfyUI.
    fn modulation(&mut self, sigma: f32, st: &Stream) -> Result<()> {
        if self.mods.contains_key(&sigma.to_bits()) {
            return Ok(());
        }
        if self.mods.len() >= MOD_CACHE {
            self.mods.clear();
        }
        let d = self.dim;
        // sinusoidal_embedding_1d: cos then sin, in f32, then the model's dtype
        let t = sigma * MULTIPLIER;
        let half = FREQ / 2;
        let mut sinus = vec![0f32; FREQ];
        for i in 0..half {
            let a = t * 10000f32.powf(-(i as f32) / half as f32);
            sinus[i] = a.cos();
            sinus[half + i] = a.sin();
        }
        let x = upload(&bf16_bytes(&sinus), st)?;
        let h = DevBuf::alloc_async(d * 2, st)?;
        let e = DevBuf::alloc_async(d * 2, st)?;
        let e0 = DevBuf::alloc_async(6 * d * 2, st)?;
        self.glin("te1", x.ptr(), FREQ, 1, PRO_PLAIN, h.ptr(), st)?;
        self.glin("te2", h.ptr(), d, 1, PRO_SILU, e.ptr(), st)?;
        self.glin("tp", e.ptr(), d, 1, PRO_SILU, e0.ptr(), st)?;
        let down = |b: &DevBuf, n: usize| -> Result<Vec<f32>> {
            let mut bytes = vec![0u8; n * 2];
            b.download(0, &mut bytes, st)?;
            Ok(bytes.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect())
        };
        let (ev, e0v) = (down(&e, d)?, down(&e0, 6 * d)?);
        let mut host = Vec::with_capacity((self.blocks.len() * 6 + 2) * d);
        for m in &self.block_mod {
            host.extend(m.iter().zip(&e0v).map(|(a, b)| bf(a + b)));
        }
        for j in 0..2 {
            host.extend((0..d).map(|c| bf(self.head_mod[j * d + c] + ev[c])));
        }
        let m = upload(&bf16_bytes(&host), st)?;
        self.mods.insert(sigma.to_bits(), m);
        Ok(())
    }

    /// Keyframes per chunk for `keyframes` keyframes of `tokens` tokens each, as `forward_orig` picks them.
    pub fn chunk(keyframes: usize, tokens: usize) -> usize {
        let c = (CHUNK_TOKENS / tokens).max(1);
        if keyframes <= c + CACHE_KEYFRAMES { keyframes } else { c }
    }

    /// Activation memory of one evaluation of `keyframes` latent frames of `lh` x `lw` (bytes).
    pub fn activation_bytes(&self, keyframes: usize, lh: usize, lw: usize) -> usize {
        let tokens = (lh / 2) * (lw / 2);
        let chunk = Self::chunk(keyframes, tokens);
        let rows = chunk * tokens;
        let d = self.dim;
        let cache = if chunk < keyframes { self.blocks.len() * 2 * (CONTEXT - 1) * tokens * (d + self.heads * 2) } else { 0 };
        let lq = lh * lw * (self.lq_in * 2 + 768 * 2 * 2) + tokens * (9 * 768 * 2 + 3072 * 2 * 2);
        rows * d * 2 * 3 /*x, attn, cross q*/ + (CONTEXT - 1 + chunk) * tokens * 3 * d * 2 + rows * (d + 4)
            + FFN_CHUNK.min(rows) * self.ffn * 3 + rows * d * 4 /*patch embedding*/ + lq + cache + (128 << 20)
    }

    /// One evaluation of the latent `x` (f32 [48][keyframes][lh][lw], device) at `sigma`, the low-quality keyframes
    /// `lq` (f32 [keyframes][h][w][c] in 0..1, device) beside it: writes the denoised latent to `den` (the same shape).
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, x: &DevBuf, lq: &DevBuf, (h, w, c): (usize, usize, usize), keyframes: usize, lh: usize, lw: usize, sigma: f32,
        den: &DevBuf, st: &Stream, mut progress: impl FnMut(usize, usize) -> Result<()>) -> Result<()> {
        if lh % 2 != 0 || lw % 2 != 0 || keyframes == 0 {
            bail!("RelayVSR latents need an even size, got {keyframes} frames of {lw}x{lh}");
        }
        let ls = self.latent_stride();
        let (hp, wp) = (lh * ls, lw * ls);
        if h > hp || w > wp {
            bail!("a {w}x{h} frame doesn't fit the {lw}x{lh} latent of a x{} model", self.factor);
        }
        self.modulation(sigma, st)?;
        let (d, nh, ffn) = (self.dim, self.heads, self.ffn);
        let (th, tw) = (lh / 2, lw / 2);
        let tokens = th * tw;
        let chunk = Self::chunk(keyframes, tokens);
        let cached = chunk < keyframes;
        let past_max = CONTEXT - 1;
        let qkv_w = 3 * d;

        // RoPE: keyframes along t (positions as bf16, the dtype ComfyUI builds them in), token rows, token columns
        let (pt, ph) = (HD / 2 - 2 * (HD / 2 / 3), HD / 2 / 3);
        let tt = upload(&f32_bytes(&rope_table((0..keyframes).map(|i| bf(i as f32)), 2 * pt)), st)?;
        let rh = upload(&f32_bytes(&rope_table((0..th).map(|i| bf(i as f32)), 2 * ph)), st)?;
        let rw = upload(&f32_bytes(&rope_table((0..tw).map(|i| bf(i as f32)), 2 * ph)), st)?;

        let nb = self.blocks.len();
        let cache_q = if cached { Some(DevBuf::alloc_async(nb * 2 * past_max * tokens * d, st)?) } else { None };
        let cache_s = if cached { Some(DevBuf::alloc_async(nb * 2 * past_max * tokens * nh * 2, st)?) } else { None };
        let rows_max = chunk * tokens;
        let xb = DevBuf::alloc_async(rows_max * d * 2, st)?;
        let qkv = DevBuf::alloc_async((past_max + chunk) * tokens * qkv_w * 2, st)?;
        let attn = DevBuf::alloc_async(rows_max * d * 2, st)?;
        let cq = DevBuf::alloc_async(rows_max * d * 2, st)?;
        let fc = FFN_CHUNK.min(rows_max);
        let a8 = DevBuf::alloc_async((rows_max * d).max(fc * ffn), st)?;
        let asc = DevBuf::alloc_async(rows_max * 4, st)?;
        let hid = DevBuf::alloc_async(fc * ffn * 2, st)?;
        let ws = DevBuf::alloc_async(unsafe { crate::ffi::gpu_int8_attn_workspace(1, nh as i32, nh as i32, rows_max as i32, self.text_len as i32) }
            .max(nh as i64 * rows_max as i64 * 4) as usize, st)?;
        let mods = self.mods[&sigma.to_bits()].ptr();
        let scale = 1.0 / (HD as f32).sqrt();
        let ones = self.g("ones");
        let row = |p: PM, r: usize, width: usize| off(p, r * width * 2);
        let gemm = |a: P, w: P, xs: P, wsc: P, bias: P, out: PM, n: usize, cols: usize, kk: usize| -> Result<()> {
            k!(gpu_int8_gemm(a, w, xs, wsc, bias, out, n as i64, cols as i64, kk as i64, DT_BF16, s(st)))
        };
        let gemm_res = |a: P, w: P, xs: P, wsc: P, bias: P, gate: P, out: PM, n: usize, cols: usize, kk: usize| -> Result<()> {
            k!(gpu_int8_gemm_residual(a, w, xs, wsc, bias, gate, out, out, n as i64, cols as i64, kk as i64, DT_BF16, s(st)))
        };
        let starts: Vec<usize> = (0..keyframes).step_by(chunk).collect();
        let total = starts.len() * nb;
        let mut done = 0;

        for &start in &starts {
            let n = chunk.min(keyframes - start);
            let rows = n * tokens;
            let store = start + n < keyframes;
            let past = start.min(past_max);
            let xp = xb.ptr();

            // the low-quality frames through lq_proj_in, then the patch embedding added
            self.lq_project(lq, (h, w, c), start, n, lh, lw, xp, st)?;
            if start == 0 {
                self.probe("lq_proj", &xb, rows * d, st)?;
            }
            {
                let tok = DevBuf::alloc_async(rows * 4 * CH * 4, st)?;
                k!(gpu_relay_patchify(x.ptr(), CH as i32, keyframes as i32, start as i32, n as i32, lh as i32, lw as i32, tok.ptr(), s(st)))?;
                let pe = DevBuf::alloc_async(rows * d * 4, st)?;
                k!(gpu_gemm(tok.ptr(), (4 * CH) as i64, self.g("pe.w"), (4 * CH) as i64, pe.ptr(), d as i64, self.g("pe.b"), rows as i64, d as i64,
                    (4 * CH) as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
                k!(gpu_relay_embed_add(pe.ptr(), xp, xp, (rows * d) as i64, s(st)))?;
            }

            for b in 0..nb {
                let base = self.blocks.acquire(b, st)?;
                let lay = self.blocks.layouts[b].clone();
                let wp =|key: &str| off(base, lay.off(key));
                let md = |j: usize| off(mods, (b * 6 + j) * d * 2);
                // the chunk's rows of q|k|v after the cached keyframes' rows
                let q0 = row(qkv.ptr(), past_max * tokens, qkv_w);

                // self-attention: x * (1 + scale) + shift after a LayerNorm, q, k and v
                k!(gpu_quant_rows(DT_BF16, PRO_LN_MOD, xp, d as i64, null(), 0, 0, null(), EPS, md(1), md(0), a8.ptr(), asc.ptr(), rows as i32,
                    d as i32, 1, s(st)))?;
                gemm(a8.ptr(), wp("qkv.w"), asc.ptr(), wp("qkv.s"), wp("qkv.b"), q0, rows, qkv_w, d)?;
                for (col, norm) in [(0, "nq"), (d, "nk")] {
                    let p_ = off(q0, col * 2);
                    k!(gpu_relay_rms(DT_BF16, p_, qkv_w as i64, p_, qkv_w as i64, wp(norm), null(), EPS, rows as i64, d as i32, 1, s(st)))?;
                    k!(gpu_relay_rope(DT_BF16, p_, qkv_w as i64, p_, qkv_w as i64, nh as i32, HD as i32, rows as i64, th as i32, tw as i32,
                        off(tt.ptr(), start * pt * 8), rh.ptr(), rw.ptr(), pt as i32, ph as i32, s(st)))?;
                }
                // the cached keyframes' keys and values in front of the chunk's
                let plane_q = tokens * d;
                let plane_s = tokens * nh;
                if past > 0 {
                    let (cq_, cs_) = (cache_q.as_ref().unwrap(), cache_s.as_ref().unwrap());
                    for kv in 0..2 {
                        let src_q = off(cq_.ptr(), ((b * 2 + kv) * past_max) * plane_q);
                        let src_s = off(cs_.ptr(), ((b * 2 + kv) * past_max) * plane_s * 2);
                        let dst = off(row(qkv.ptr(), (past_max - past) * tokens, qkv_w), (1 + kv) * d * 2);
                        k!(gpu_relay_kv_dequant(src_q, src_s, dst, qkv_w as i64, (past * tokens) as i64, nh as i32, s(st)))?;
                    }
                }
                let kb = off(row(qkv.ptr(), (past_max - past) * tokens, qkv_w), d * 2);
                let mut wa = WinArgs::new(q0, qkv_w as i64, attn.ptr(), d as i64, th, tw, nh, nh, true);
                wa.q_ps = (tokens * qkv_w) as i64;
                wa.o_ps = (tokens * d) as i64;
                wa.kb = kb;
                wa.vb = off(kb, d * 2);
                wa.k_ps = (tokens * qkv_w) as i64;
                wa.v_ps = wa.k_ps;
                wa.k_rs = qkv_w as i64;
                wa.v_rs = qkv_w as i64;
                wa.uwh = WINDOW.0 as i32;
                wa.uww = WINDOW.1 as i32;
                wa.tk = (past + n) as i32;
                wa.tq = n as i32;
                wa.tw = CONTEXT as i32;
                k!(gpu_win_attn(DT_BF16, HD as i32, &wa, scale, s(st)))?;
                if store {
                    // the cache moves on to the last keyframes, older cached ones first when this chunk is short
                    let (cq_, cs_) = (cache_q.as_ref().unwrap(), cache_s.as_ref().unwrap());
                    let new = n.min(past_max);
                    let old = past.min(past_max - new);
                    for kv in 0..2 {
                        let q_at = |i: usize| off(cq_.ptr(), ((b * 2 + kv) * past_max + i) * plane_q);
                        let s_at = |i: usize| off(cs_.ptr(), (((b * 2 + kv) * past_max + i) * plane_s) * 2);
                        for i in 0..old {
                            memcpy_d2d(q_at(i), q_at(past - old + i), plane_q, st)?;
                            memcpy_d2d(s_at(i), s_at(past - old + i), plane_s * 2, st)?;
                        }
                        let src = off(row(q0, (n - new) * tokens, qkv_w), (1 + kv) * d * 2);
                        k!(gpu_relay_kv_quant(src, qkv_w as i64, q_at(old), s_at(old), (new * tokens) as i64, nh as i32, s(st)))?;
                    }
                }
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), d as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), rows as i32,
                    d as i32, 1, s(st)))?;
                gemm_res(a8.ptr(), wp("o.w"), asc.ptr(), wp("o.s"), wp("o.b"), md(2), xp, rows, d, d)?;

                // cross-attention over the stored prompt, after norm3
                k!(gpu_quant_rows_b(DT_BF16, PRO_LN_AFFINE, xp, d as i64, wp("n3w"), wp("n3b"), EPS, null(), null(), a8.ptr(), asc.ptr(),
                    rows as i32, d as i32, 1, s(st)))?;
                gemm(a8.ptr(), wp("cq.w"), asc.ptr(), wp("cq.s"), wp("cq.b"), cq.ptr(), rows, d, d)?;
                k!(gpu_relay_rms(DT_BF16, cq.ptr(), d as i64, cq.ptr(), d as i64, wp("cnq"), null(), EPS, rows as i64, d as i32, 1, s(st)))?;
                let lt = self.text_len;
                let kc = off(self.ctx_kv.ptr(), (b * 2) * lt * d * 2);
                let vc = off(self.ctx_kv.ptr(), (b * 2 + 1) * lt * d * 2);
                let (di, hs) = (d as i64, HD as i64);
                if self.int8_attn {
                    k!(gpu_int8_attn(cq.ptr(), kc, vc, attn.ptr(), ws.ptr(), 1, rows as i32, lt as i32, nh as i32, nh as i32, 0, di, hs, 0, di, hs, 0, di,
                        hs, 0, di, hs, scale, s(st)))?;
                } else {
                    k!(gpu_flash_attn(cq.ptr(), kc, vc, attn.ptr(), ws.ptr(), 1, rows as i32, lt as i32, nh as i32, nh as i32, 0, di, hs, 0, di, hs, 0,
                        di, hs, 0, di, hs, scale, 0, s(st)))?;
                }
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), d as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), rows as i32,
                    d as i32, 1, s(st)))?;
                gemm_res(a8.ptr(), wp("co.w"), asc.ptr(), wp("co.s"), wp("co.b"), ones, xp, rows, d, d)?;

                // feed-forward, gelu(tanh) fused into the down projection's quantizer
                for r0 in (0..rows).step_by(fc) {
                    let m = fc.min(rows - r0);
                    let xr = row(xp, r0, d);
                    k!(gpu_quant_rows(DT_BF16, PRO_LN_MOD, xr, d as i64, null(), 0, 0, null(), EPS, md(4), md(3), a8.ptr(), asc.ptr(), m as i32,
                        d as i32, 1, s(st)))?;
                    gemm(a8.ptr(), wp("f1.w"), asc.ptr(), wp("f1.s"), wp("f1.b"), hid.ptr(), m, ffn, d)?;
                    k!(gpu_quant_rows(DT_BF16, PRO_GELU_TANH, hid.ptr(), ffn as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(),
                        m as i32, ffn as i32, 1, s(st)))?;
                    gemm_res(a8.ptr(), wp("f2.w"), asc.ptr(), wp("f2.s"), wp("f2.b"), md(5), xr, m, d, ffn)?;
                }
                self.blocks.release(b, st)?;
                if start == 0 {
                    self.probe(&format!("block{b}"), &xb, rows * d, st)?;
                }
                done += 1;
                progress(done, total)?;
            }

            // head: LayerNorm, modulation, projection; then back to the latent grid and x - sigma * v
            let hm = off(mods, nb * 6 * d * 2);
            let normed = DevBuf::alloc_async(rows * d * 2, st)?;
            k!(gpu_prologue_rows_b(DT_BF16, PRO_LN_MOD, xp, d as i64, null(), null(), EPS, off(hm, d * 2), hm, normed.ptr(), d as i64, rows as i32,
                d as i32, s(st)))?;
            let hl = self.lins["hd"];
            let tok = DevBuf::alloc_async(rows * hl.n * 2, st)?;
            self.glin("hd", normed.ptr(), d, rows, PRO_PLAIN, tok.ptr(), st)?;
            k!(gpu_relay_unpatch_denoise(tok.ptr(), hl.n as i32, x.ptr(), den.ptr(), CH as i32, keyframes as i32, start as i32, n as i32, lh as i32,
                lw as i32, sigma, s(st)))?;
        }
        Ok(())
    }

    /// Keeps the first `count` bf16 values of `buf` as probe `name`, when probing.
    fn probe(&mut self, name: &str, buf: &DevBuf, count: usize, st: &Stream) -> Result<()> {
        let Some(probes) = &mut self.probes else { return Ok(()) };
        let mut bytes = vec![0u8; count * 2];
        buf.download(0, &mut bytes, st)?;
        st.sync()?;
        probes.push((name.to_string(), bytes.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect()));
        Ok(())
    }

    /// `lq_proj_in` of keyframes `start..start + n` into bf16 rows [n tokens][dim] at `out`.
    #[allow(clippy::too_many_arguments)]
    fn lq_project(&self, lq: &DevBuf, (h, w, c): (usize, usize, usize), start: usize, n: usize, lh: usize, lw: usize, out: PM, st: &Stream)
        -> Result<()> {
        let ls = self.latent_stride();
        let (hp, wp) = (lh * ls, lw * ls);
        let up = if self.factor == 4 { 4 } else { 1 };
        let r = 4 * self.factor;
        let cin = self.lq_in;
        let c1 = 768usize;
        let c2 = self.dim;
        let (th, tw) = (lh / 2, lw / 2);
        let px = lh * lw;
        let un = DevBuf::alloc_async(px * cin * 2, st)?;
        let y1 = DevBuf::alloc_async(px * c1 * 2, st)?;
        let cols = DevBuf::alloc_async(th * tw * 9 * c1 * 2, st)?;
        let y2 = DevBuf::alloc_async(th * tw * c2 * 2, st)?;
        for f in 0..n {
            let src = off(lq.ptr(), (start + f) * h * w * c * 4);
            k!(gpu_relay_lq_unshuffle(src, h as i32, w as i32, c as i32, hp as i32, wp as i32, up, r as i32, un.ptr(), s(st)))?;
            k!(gpu_gemm(un.ptr(), cin as i64, self.g("c1.w"), cin as i64, y1.ptr(), c1 as i64, self.g("c1.b"), px as i64, c1 as i64, cin as i64, 1,
                DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
            k!(gpu_relay_rms(DT_BF16, y1.ptr(), c1 as i64, y1.ptr(), c1 as i64, self.g("n1"), null(), EPS, px as i64, c1 as i32, 1, s(st)))?;
            k!(gpu_relay_silu(DT_BF16, y1.ptr(), (px * c1) as i64, s(st)))?;
            k!(gpu_relay_im2col3(DT_BF16, y1.ptr(), lh as i32, lw as i32, c1 as i32, 2, cols.ptr(), (9 * c1) as i32, s(st)))?;
            k!(gpu_gemm(cols.ptr(), (9 * c1) as i64, self.g("c2.w"), (9 * c1) as i64, y2.ptr(), c2 as i64, self.g("c2.b"), (th * tw) as i64,
                c2 as i64, (9 * c1) as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
            k!(gpu_relay_rms(DT_BF16, y2.ptr(), c2 as i64, y2.ptr(), c2 as i64, self.g("n2"), null(), EPS, (th * tw) as i64, c2 as i32, 1, s(st)))?;
            k!(gpu_relay_silu(DT_BF16, y2.ptr(), (th * tw * c2) as i64, s(st)))?;
            self.glin("lql", y2.ptr(), c2, th * tw, PRO_PLAIN, off(out, f * th * tw * self.dim * 2), st)?;
        }
        Ok(())
    }
}

/// The Flash decoder's width, heads, key/value groups and head width.
const FD: usize = 512;
const FH: usize = 8;
const FKV: usize = 2;
const FHD: usize = 64;
const FEPS: f32 = 1e-5;
const FCOND: (usize, usize) = (48, 48);
const FVIDEO: (usize, usize) = (24, 24);

/// The Flash decoder: every weight in VRAM, fp16.
pub struct Flash {
    w: DevBuf,
    l: Layout,
    pub factor: usize,
    depth: usize,
    ffn: usize,
    head: usize,
}

/// Bytes of a float tensor as fp16.
fn f16_of(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(f16_bytes(&st.f32_vec(name, false)?))
}

impl Flash {
    pub fn load(st: &SafeTensors, s0: &Stream) -> Result<Self> {
        if !is_flash(|k| st.has(k)) {
            bail!("not a RelayVSR Flash decoder");
        }
        let factor = flash_factor(&st.metadata)?;
        let depth = (0..).take_while(|i| st.has(&format!("backbone.blocks.{i}.attention.query.weight"))).count();
        let ffn = st.info("backbone.blocks.0.feed_forward.down.weight")?.shape[1];
        let head = st.info("spatial_head.0.weight")?.shape[0];
        if st.info("backbone.blocks.0.attention.query.weight")?.shape != [FD, FD] || st.info("backbone.blocks.0.attention.key.weight")?.shape != [FKV * FHD, FD] {
            bail!("a RelayVSR Flash decoder of another width");
        }
        let mut l = Layout::new();
        // the encoder's 3x3 convs with their taps outermost (the im2col's columns), inputs padded to 8
        for (i, idx) in [0usize, 2, 4].into_iter().enumerate() {
            let name = format!("lr_encoder.layers.{idx}.weight");
            let shape = st.info(&name)?.shape.clone();
            let (o, ci) = (shape[0], shape[1]);
            let k = (9 * ci).next_multiple_of(8);
            let w = st.f32_vec(&name, false)?;
            let mut r = vec![0f32; o * k];
            for oo in 0..o {
                for c in 0..ci {
                    for t in 0..9 {
                        r[oo * k + t * ci + c] = w[(oo * ci + c) * 9 + t];
                    }
                }
            }
            l.bytes(&format!("e{i}.w"), f16_bytes(&r));
            l.bytes(&format!("e{i}.b"), f16_of(st, &format!("lr_encoder.layers.{idx}.bias"))?);
        }
        l.bytes("e3.w", f16_of(st, "lr_encoder.layers.6.weight")?);
        l.bytes("e3.b", f16_of(st, "lr_encoder.layers.6.bias")?);
        l.bytes("en", f32_bytes(&st.f32_vec("lr_encoder.norm.weight", false)?));
        l.bytes("cp.w", f16_of(st, "condition_encoder.projection.weight")?);
        l.bytes("cp.b", f16_of(st, "condition_encoder.projection.bias")?);
        l.bytes("cn", f32_bytes(&st.f32_vec("condition_encoder.norm.weight", false)?));
        l.bytes("vte", f16_of(st, "video_type_embedding")?);
        l.bytes("cte", f16_of(st, "condition_type_embedding")?);
        for b in 0..depth {
            let n = |x: &str| format!("backbone.blocks.{b}.{x}");
            let cat = |names: &[&str]| -> Result<Vec<u8>> {
                let mut v = Vec::new();
                for x in names {
                    v.extend(st.f32_vec(&n(x), false)?);
                }
                Ok(f16_bytes(&v))
            };
            l.bytes(&format!("{b}.an"), f32_bytes(&st.f32_vec(&n("attention_norm.weight"), false)?));
            l.bytes(&format!("{b}.vqkv.w"), cat(&["attention.query.weight", "attention.key.weight", "attention.value.weight"])?);
            l.bytes(&format!("{b}.vqkv.b"), cat(&["attention.query.bias", "attention.key.bias", "attention.value.bias"])?);
            l.bytes(&format!("{b}.cqkv.w"), cat(&["attention.query.weight", "attention.condition_key.weight", "attention.condition_value.weight"])?);
            l.bytes(&format!("{b}.cqkv.b"), cat(&["attention.query.bias", "attention.condition_key.bias", "attention.condition_value.bias"])?);
            l.bytes(&format!("{b}.kn"), f32_bytes(&st.f32_vec(&n("attention.key_norm.weight"), false)?));
            l.bytes(&format!("{b}.vn"), f32_bytes(&st.f32_vec(&n("attention.value_norm.weight"), false)?));
            l.bytes(&format!("{b}.o.w"), cat(&["attention.output.weight"])?);
            l.bytes(&format!("{b}.o.b"), cat(&["attention.output.bias"])?);
            l.bytes(&format!("{b}.fn"), f32_bytes(&st.f32_vec(&n("feed_forward_norm.weight"), false)?));
            l.bytes(&format!("{b}.gu"), cat(&["feed_forward.gate_up_proj.weight"])?);
            l.bytes(&format!("{b}.dn"), cat(&["feed_forward.down.weight"])?);
        }
        l.bytes("sh.w", f16_of(st, "spatial_head.0.weight")?);
        l.bytes("sh.b", f16_of(st, "spatial_head.0.bias")?);
        l.bytes("rp.w", f16_of(st, "residual_projection.weight")?);
        l.bytes("rp.b", f16_of(st, "residual_projection.bias")?);
        let mut host = vec![0u8; l.size];
        l.pack(st, &mut host)?;
        let w = DevBuf::alloc(l.size)?;
        w.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Flash { w, l, factor, depth, ffn, head })
    }

    fn g(&self, key: &str) -> PM {
        self.w.at(self.l.off(key))
    }

    pub fn latent_stride(&self) -> usize {
        16 / self.factor
    }

    /// VRAM the weights take.
    pub fn weight_bytes(&self) -> usize {
        self.w.size()
    }

    /// Activation memory of decoding a latent grid of `h` x `w` (bytes).
    pub fn activation_bytes(&self, h: usize, w: usize) -> usize {
        let n = h * w;
        let ls = self.latent_stride();
        let (hp, wp) = (h * ls, w * ls);
        let enc = (hp / 2) * (wp / 2) * (32 + 64) * 2 + (hp / 4) * (wp / 4) * (9 * 64 + 128) * 2 + n * (9 * 128 + 128) * 2;
        n * 2 * (FD * 3 + 768 * 2 + 2 * self.ffn + self.ffn + self.head + 2 * FKV * FHD) + self.depth * n * 2 * FKV * FHD * 2 * 3 + enc
            + hp * wp * 3 * 6 + (64 << 20)
    }

    fn gemm(&self, a: P, lda: usize, w: &str, out: PM, bias: Option<String>, m: usize, n: usize, kk: usize, st: &Stream) -> Result<()> {
        let b = bias.map_or(null(), |b| self.g(&b) as P);
        k!(gpu_gemm(a, lda as i64, self.g(w), kk as i64, out, n as i64, b, m as i64, n as i64, kk as i64, 1, DT_F16, DT_F16, 1.0, 0.0, s(st)))
    }

    /// Decode keyframe latents `lat` (f32 [48][keyframes][h][w], host) of the clip `frames` (f32 [frames][fh][fw][c] in
    /// 0..1, host): f32 [frames][fh * factor][fw * factor][3] in 0..1, frames calling `progress` as they finish.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(&self, lat: &[f32], keyframe_count: usize, h: usize, w: usize, frames: &[f32], (count, fh, fw, c): (usize, usize, usize, usize),
        detail: f32, st: &Stream, mut progress: impl FnMut(usize, usize) -> Result<()>) -> Result<Vec<f32>> {
        let ls = self.latent_stride();
        let kf = keyframes(count);
        if kf.len() != keyframe_count || (h, w) != latent_size(fh, fw, ls) {
            bail!("the latent ({keyframe_count} keyframes of {w}x{h}) doesn't fit the x{} RelayVSR Flash decoder and these {count} {fw}x{fh} frames: use \
                the RelayVSR model and Flash decoder of the same factor, and the same video in RelayVSR Conditioning and VSR Decode", self.factor);
        }
        if lat.len() != CH * keyframe_count * h * w || frames.len() != count * fh * fw * c {
            bail!("latent or frames of the wrong size");
        }
        let n = h * w;
        let (hp, wp) = (h * ls, w * ls);
        let up = self.factor;
        let (oh, ow) = (fh * up, fw * up);
        let lat_d = upload(&f32_bytes(lat), st)?;
        // RoPE: t = 0 for this frame and the keyframes, t = -1 for the previous frame; rows; columns
        let (pt, ph) = (8usize, 12usize);
        let t0 = upload(&f32_bytes(&rope_axis(0.0, 2 * pt)), st)?;
        let tm1 = upload(&f32_bytes(&rope_axis(-1.0, 2 * pt)), st)?;
        let rh = upload(&f32_bytes(&rope_table((0..h).map(|i| i as f32), 2 * ph)), st)?;
        let rw = upload(&f32_bytes(&rope_table((0..w).map(|i| i as f32), 2 * ph)), st)?;
        let kvw = FKV * FHD;
        let qkv_w = FD + 2 * kvw;
        let x = DevBuf::alloc_async(n * FD * 2, st)?;
        let xn = DevBuf::alloc_async(n * FD * 2, st)?;
        let qkv = DevBuf::alloc_async(n * qkv_w * 2, st)?;
        let attn = DevBuf::alloc_async(n * FD * 2, st)?;
        let gu = DevBuf::alloc_async(n * 2 * self.ffn * 2, st)?;
        let act = DevBuf::alloc_async(n * self.ffn * 2, st)?;
        let kcur = DevBuf::alloc_async(n * kvw * 2, st)?;
        let kpast = DevBuf::alloc_async(n * kvw * 2, st)?;
        // per block: the two keyframes' rotated keys and values, and the previous frame's keys and values
        let plane = n * 2 * kvw * 2;
        let mut cond = [DevBuf::alloc_async(self.depth * plane, st)?, DevBuf::alloc_async(self.depth * plane, st)?];
        let hist = DevBuf::alloc_async(self.depth * plane, st)?;
        let lr_f = DevBuf::alloc_async(hp * wp * 3 * 4, st)?;
        let lr_t = DevBuf::alloc_async(hp * wp * 3 * 2, st)?;
        let out_d = DevBuf::alloc_async(oh * ow * 3 * 4, st)?;
        let frames_d = upload(&f32_bytes(frames), st)?;
        let scale = 1.0 / (FHD as f32).sqrt();
        let mut out = vec![0f32; count * oh * ow * 3];

        // one block's attention input, q|k|v with k and v normalized and q rotated
        let project = |b: usize, which: &str| -> Result<()> {
            k!(gpu_relay_rms(DT_F16, x.ptr(), FD as i64, xn.ptr(), FD as i64, self.g(&format!("{b}.an")), null(), FEPS, n as i64, FD as i32, 1, s(st)))?;
            self.gemm(xn.ptr(), FD, &format!("{b}.{which}.w"), qkv.ptr(), Some(format!("{b}.{which}.b")), n, qkv_w, FD, st)?;
            let k_ = off(qkv.ptr(), FD * 2);
            let v_ = off(qkv.ptr(), (FD + kvw) * 2);
            k!(gpu_relay_rms(DT_F16, k_, qkv_w as i64, k_, qkv_w as i64, self.g(&format!("{b}.kn")), null(), FEPS, n as i64, FHD as i32, FKV as i32,
                s(st)))?;
            k!(gpu_relay_rms(DT_F16, v_, qkv_w as i64, v_, qkv_w as i64, self.g(&format!("{b}.vn")), null(), FEPS, n as i64, FHD as i32, FKV as i32,
                s(st)))?;
            k!(gpu_relay_rope(DT_F16, qkv.ptr(), qkv_w as i64, qkv.ptr(), qkv_w as i64, FH as i32, FHD as i32, n as i64, h as i32, w as i32, t0.ptr(),
                rh.ptr(), rw.ptr(), pt as i32, ph as i32, s(st)))
        };
        // x += output(attn); x += ffn(x)
        let finish = |b: usize| -> Result<()> {
            self.gemm(attn.ptr(), FD, &format!("{b}.o.w"), xn.ptr(), Some(format!("{b}.o.b")), n, FD, FD, st)?;
            k!(gpu_add_f16(x.ptr(), xn.ptr(), (n * FD) as i64, s(st)))?;
            k!(gpu_relay_rms(DT_F16, x.ptr(), FD as i64, xn.ptr(), FD as i64, self.g(&format!("{b}.fn")), null(), FEPS, n as i64, FD as i32, 1, s(st)))?;
            self.gemm(xn.ptr(), FD, &format!("{b}.gu"), gu.ptr(), None, n, 2 * self.ffn, FD, st)?;
            k!(gpu_ts_silu_mul_f16(gu.ptr(), (2 * self.ffn) as i64, act.ptr(), self.ffn as i64, n as i64, self.ffn as i32, s(st)))?;
            self.gemm(act.ptr(), self.ffn, &format!("{b}.dn"), xn.ptr(), None, n, FD, self.ffn, st)?;
            k!(gpu_add_f16(x.ptr(), xn.ptr(), (n * FD) as i64, s(st)))
        };
        // prepare_condition: a keyframe latent through the blocks, keeping each one's rotated keys and values
        let condition = |ki: usize, dst: &DevBuf| -> Result<()> {
            k!(gpu_relay_plane_rows(DT_F16, lat_d.ptr(), CH as i32, keyframe_count as i32, ki as i32, h as i32, w as i32, xn.ptr(), s(st)))?;
            self.gemm(xn.ptr(), CH, "cp.w", x.ptr(), Some("cp.b".into()), n, FD, CH, st)?;
            k!(gpu_relay_rms(DT_F16, x.ptr(), FD as i64, x.ptr(), FD as i64, self.g("cn"), self.g("cte"), FEPS, n as i64, FD as i32, 1, s(st)))?;
            for b in 0..self.depth {
                project(b, "cqkv")?;
                let kv = off(dst.ptr(), b * plane);
                // rotated keys then the values, [n][k | v]
                k!(gpu_relay_rope(DT_F16, off(qkv.ptr(), FD * 2), qkv_w as i64, kv, (2 * kvw) as i64, FKV as i32, FHD as i32, n as i64, h as i32,
                    w as i32, t0.ptr(), rh.ptr(), rw.ptr(), pt as i32, ph as i32, s(st)))?;
                memcpy_2d_d2d(off(kv, kvw * 2), 2 * kvw * 2, off(qkv.ptr(), (FD + kvw) * 2), qkv_w * 2, kvw * 2, n, st)?;
                let a = WinArgs::new(qkv.ptr(), qkv_w as i64, attn.ptr(), FD as i64, h, w, FH, FKV, false).plane(kv, (2 * kvw) as i64,
                    off(kv, kvw * 2), (2 * kvw) as i64, FCOND);
                k!(gpu_win_attn(DT_F16, FHD as i32, &a, scale, s(st)))?;
                finish(b)?;
            }
            Ok(())
        };

        let pairs: Vec<(usize, usize)> = if kf.len() > 1 { kf.windows(2).map(|p| (p[0], p[1])).collect() } else { vec![(kf[0], kf[0])] };
        condition(0, &cond[0])?;
        let mut written = 0;
        for (wi, &(first, last)) in pairs.iter().enumerate() {
            condition((wi + 1).min(kf.len() - 1), &cond[1])?;
            for frame in first..=last {
                let has_past = frame > first;
                // the low-quality frame: f32 for the bicubic upscale, fp16 through the encoder
                let src = off(frames_d.ptr(), frame * fh * fw * c * 4);
                k!(gpu_relay_lr_frame(DT_F16, src, fh as i32, fw as i32, c as i32, hp as i32, wp as i32, lr_f.ptr(), lr_t.ptr(), s(st)))?;
                self.encode_lr(&lr_t, hp, wp, &x, st)?;
                for b in 0..self.depth {
                    project(b, "vqkv")?;
                    let kn = off(qkv.ptr(), FD * 2);
                    let vn = off(qkv.ptr(), (FD + kvw) * 2);
                    k!(gpu_relay_rope(DT_F16, kn, qkv_w as i64, kcur.ptr(), kvw as i64, FKV as i32, FHD as i32, n as i64, h as i32, w as i32, t0.ptr(),
                        rh.ptr(), rw.ptr(), pt as i32, ph as i32, s(st)))?;
                    let hb = off(hist.ptr(), b * plane);
                    let (cs, ce) = (off(cond[0].ptr(), b * plane), off(cond[1].ptr(), b * plane));
                    let kvs = (2 * kvw) as i64;
                    let mut a = WinArgs::new(qkv.ptr(), qkv_w as i64, attn.ptr(), FD as i64, h, w, FH, FKV, false)
                        .plane(cs, kvs, off(cs, kvw * 2), kvs, FCOND)
                        .plane(ce, kvs, off(ce, kvw * 2), kvs, FCOND);
                    if has_past {
                        k!(gpu_relay_rope(DT_F16, hb, kvs, kpast.ptr(), kvw as i64, FKV as i32, FHD as i32, n as i64, h as i32, w as i32, tm1.ptr(),
                            rh.ptr(), rw.ptr(), pt as i32, ph as i32, s(st)))?;
                        a = a.plane(kpast.ptr(), kvw as i64, off(hb, kvw * 2), kvs, FVIDEO);
                    }
                    a = a.plane(kcur.ptr(), kvw as i64, vn, qkv_w as i64, FVIDEO);
                    k!(gpu_win_attn(DT_F16, FHD as i32, &a, scale, s(st)))?;
                    // this frame's normalized keys and values are the next one's past
                    memcpy_2d_d2d(hb, 2 * kvw * 2, kn, qkv_w * 2, 2 * kvw * 2, n, st)?;
                    finish(b)?;
                }
                if wi == 0 || frame > first {
                    // spatial head and residual projection, then the upscale
                    self.gemm(x.ptr(), FD, "sh.w", gu.ptr(), Some("sh.b".into()), n, self.head, FD, st)?;
                    k!(gpu_relay_silu(DT_F16, gu.ptr(), (n * self.head) as i64, s(st)))?;
                    self.gemm(gu.ptr(), self.head, "rp.w", act.ptr(), Some("rp.b".into()), n, 768, self.head, st)?;
                    k!(gpu_relay_flash_out(DT_F16, act.ptr(), lr_f.ptr(), hp as i32, wp as i32, up as i32, detail, out_d.ptr(), oh as i32, ow as i32,
                        w as i32, s(st)))?;
                    let dst = &mut out[frame * oh * ow * 3..(frame + 1) * oh * ow * 3];
                    let mut bytes = vec![0u8; oh * ow * 3 * 4];
                    out_d.download(0, &mut bytes, st)?;
                    for (o, b4) in dst.iter_mut().zip(bytes.chunks_exact(4)) {
                        *o = f32::from_le_bytes([b4[0], b4[1], b4[2], b4[3]]);
                    }
                    written += 1;
                    progress(written, count)?;
                }
            }
            cond.swap(0, 1);
        }
        Ok(out)
    }

    /// `lr_encoder` and the video type embedding: the fp16 frame (NHWC [hp][wp][3]) to rows [h w][512] in `x`.
    fn encode_lr(&self, lr: &DevBuf, hp: usize, wp: usize, x: &DevBuf, st: &Stream) -> Result<()> {
        let strides = [2usize, 2, self.latent_stride() / 4];
        let chans = [3usize, 64, 128];
        let outs = [64usize, 128, 128];
        let (mut hh, mut ww) = (hp, wp);
        let mut src: Option<DevBuf> = None;
        for i in 0..3 {
            let (oh, ow) = ((hh - 1) / strides[i] + 1, (ww - 1) / strides[i] + 1);
            let k = (9 * chans[i]).next_multiple_of(8);
            let cols = DevBuf::alloc_async(oh * ow * k * 2, st)?;
            let inp = src.as_ref().map_or(lr.ptr(), |b| b.ptr());
            k!(gpu_relay_im2col3(DT_F16, inp, hh as i32, ww as i32, chans[i] as i32, strides[i] as i32, cols.ptr(), k as i32, s(st)))?;
            let y = DevBuf::alloc_async(oh * ow * outs[i] * 2, st)?;
            self.gemm(cols.ptr(), k, &format!("e{i}.w"), y.ptr(), Some(format!("e{i}.b")), oh * ow, outs[i], k, st)?;
            k!(gpu_relay_silu(DT_F16, y.ptr(), (oh * ow * outs[i]) as i64, s(st)))?;
            src = Some(y);
            (hh, ww) = (oh, ow);
        }
        let n = hh * ww;
        self.gemm(src.unwrap().ptr(), 128, "e3.w", x.ptr(), Some("e3.b".into()), n, FD, 128, st)?;
        k!(gpu_relay_rms(DT_F16, x.ptr(), FD as i64, x.ptr(), FD as i64, self.g("en"), self.g("vte"), FEPS, n as i64, FD as i32, 1, s(st)))
    }
}

/// `--relayvsr-check`: the keyframe model's denoised latent and the Flash decoder's frames against a dump of
/// `tools/relayvsr_ref.py`, each from ComfyUI's own inputs. `RELAYVSR_FLASH` set runs flash attention instead of int8.
pub fn check(dit: &std::path::Path, vae: &std::path::Path, reference: &std::path::Path) -> Result<()> {
    use crate::pixal3d::report;
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let r = SafeTensors::open(reference)?;
    let st = Stream::new()?;
    let psnr = |a: &[f32], b: &[f32]| 10.0 * (a.len() as f64 / a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>()).log10();
    // [frames, h, w, c]
    let fs = r.info("frames")?.shape.clone();
    let (count, h, w, c) = (fs[0], fs[1], fs[2], fs[3]);
    let frames = r.f32_vec("frames", false)?;
    let kf = keyframes(count);
    let px = h * w * c;
    let kdata: Vec<f32> = kf.iter().flat_map(|&i| frames[i * px..(i + 1) * px].iter().copied()).collect();

    let file = SafeTensors::open(dit)?;
    let free = crate::cuda::device_free()?;
    let mut d = Dit::load(&file, free.saturating_sub(6 << 30), 2, 6, &st)?;
    d.int8_attn = std::env::var_os("RELAYVSR_FLASH").is_none();
    let (lh, lw) = latent_size(h, w, d.latent_stride());
    let n = CH * kf.len() * lh * lw;
    let noise = r.f32_vec("noise", false)?;
    if noise.len() != n {
        bail!("the dump's noise has {} values, a x{} model's latent of {count} frames of {w}x{h} {n}", noise.len(), d.factor);
    }
    eprintln!("x{}: {count} frames of {w}x{h}, {} keyframes of {lw}x{lh} latents", d.factor, kf.len());
    let sigma = r.f32_vec("dit_t", false)?[0] / MULTIPLIER;
    let (x, lq) = (upload(&f32_bytes(&noise), &st)?, upload(&f32_bytes(&kdata), &st)?);
    let den = DevBuf::alloc(n * 4)?;
    for run in 0..2 {
        d.probes = if run == 1 { Some(Vec::new()) } else { None };
        let t0 = std::time::Instant::now();
        d.predict(&x, &lq, (h, w, c), kf.len(), lh, lw, sigma, &den, &st, |_, _| Ok(()))?;
        st.sync()?;
        eprintln!("evaluation {run}: {:.0} ms{}", t0.elapsed().as_secs_f64() * 1e3, if run == 1 { " (probed)" } else { "" });
    }
    for (name, ours) in d.probes.take().unwrap_or_default() {
        if r.has(&name) {
            report(&name, &ours, &r.f32_vec(&name, false)?);
        }
    }
    let mut bytes = vec![0u8; n * 4];
    den.download(0, &mut bytes, &st)?;
    st.sync()?;
    let ours: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let (x_in, out) = (r.f32_vec("dit_in", false)?, r.f32_vec("dit_out", false)?);
    let theirs: Vec<f32> = x_in.iter().zip(&out).map(|(x, o)| x - sigma * o).collect();
    let samples = r.f32_vec("samples", false)?;
    report("denoised", &ours, &theirs);
    report("denoised vs samples", &ours, &samples);
    // per keyframe: later chunks read the int8 key cache
    let plane = lh * lw;
    let per_key: Vec<String> = (0..kf.len()).map(|k| {
        let (mut num, mut den2) = (0f64, 0f64);
        for ch in 0..CH {
            let o = (ch * kf.len() + k) * plane;
            for i in o..o + plane {
                num += ((ours[i] - theirs[i]) as f64).powi(2);
                den2 += (theirs[i] as f64).powi(2);
            }
        }
        format!("{:.2}%", 100.0 * (num / den2).sqrt())
    }).collect();
    eprintln!("denoised rel by keyframe (chunks of {}): {}", Dit::chunk(kf.len(), (lh / 2) * (lw / 2)), per_key.join(" "));
    drop(d);
    if !r.has("decoded") {
        return Ok(());
    }

    let vf = SafeTensors::open(vae)?;
    let flash = Flash::load(&vf, &st)?;
    let detail = r.f32_vec("detail", false)?[0];
    let decoded = r.f32_vec("decoded", false)?;
    for run in 0..2 {
        let t0 = std::time::Instant::now();
        let theirs_lat = flash.decode(&samples, kf.len(), lh, lw, &frames, (count, h, w, c), detail, &st, |_, _| Ok(()))?;
        if run == 1 {
            eprintln!("Flash decode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            report("decoded", &theirs_lat, &decoded);
            let ours_px = flash.decode(&ours, kf.len(), lh, lw, &frames, (count, h, w, c), detail, &st, |_, _| Ok(()))?;
            // our last frame, end to end, beside the dump
            let (oh, ow) = (h * flash.factor, w * flash.factor);
            let last: Vec<u8> = ours_px[(count - 1) * oh * ow * 3..].iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
            let path = reference.with_file_name("ours_last.png");
            let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&path)?), ow as u32, oh as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.write_header()?.write_image_data(&last)?;
            eprintln!("decoded: ComfyUI's latent through our decoder {:.2} dB from ComfyUI's frames; ours end to end {:.2} dB", psnr(&theirs_lat, &decoded),
                psnr(&ours_px, &decoded));
            let per = decoded.len() / count;
            let worst = (0..count).map(|f| (f, psnr(&theirs_lat[f * per..(f + 1) * per], &decoded[f * per..(f + 1) * per])))
                .min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            eprintln!("worst frame through our decoder: {} at {:.2} dB", worst.0, worst.1);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframes_like_comfy() {
        assert_eq!(keyframes(1), vec![0]);
        assert_eq!(keyframes(2), vec![0, 1]);
        assert_eq!(keyframes(16), vec![0, 15]);
        assert_eq!(keyframes(17), vec![0, 15, 16]);
        assert_eq!(keyframes(31), vec![0, 15, 30]);
        assert_eq!(keyframes(32), vec![0, 15, 30, 31]);
    }

    #[test]
    fn latent_sizes() {
        // x2: 8 low-quality pixels per latent pixel, whole 2x2 patches
        assert_eq!(latent_size(1080, 1920, 8), (136, 240));
        assert_eq!(latent_size(480, 854, 4), (120, 214));
        assert_eq!(latent_size(17, 9, 8), (4, 2));
        assert_eq!(factor_of(768).unwrap(), 4);
        assert_eq!(factor_of(192).unwrap(), 2);
        assert!(factor_of(48).is_err());
    }

    /// comfy-kitchen's eager `window_attention` on the CPU, f32: q [tq][h][w][heads][hd], each plane's k and v
    /// [h][w][kv][hd], the planes each query plane attends.
    #[allow(clippy::too_many_arguments)]
    fn window_ref(q: &[f32], planes: &[(Vec<f32>, Vec<f32>, (usize, usize))], per_q: &[Vec<usize>], (tq, h, w): (usize, usize, usize), heads: usize,
        kvh: usize, hd: usize, clamp: bool) -> Vec<f32> {
        let lo = |i: usize, win: usize, size: usize| -> (i64, i64) {
            if clamp {
                let wl = win.min(size);
                let l = (i as i64 - (wl / 2) as i64).clamp(0, (size - wl) as i64);
                (l, l + wl as i64)
            } else {
                let l = i as i64 - (win / 2) as i64;
                (l.max(0), (l + win as i64).min(size as i64))
            }
        };
        let mut out = vec![0f32; q.len()];
        let scale = 1.0 / (hd as f32).sqrt();
        for t in 0..tq {
            for y in 0..h {
                for x in 0..w {
                    for hh in 0..heads {
                        let g = hh / (heads / kvh);
                        let qv = &q[(((t * h + y) * w + x) * heads + hh) * hd..][..hd];
                        let mut scores = Vec::new();
                        let mut vals: Vec<&[f32]> = Vec::new();
                        for &p in &per_q[t] {
                            let (k, v, win) = &planes[p];
                            let (y0, y1) = lo(y, win.0, h);
                            let (x0, x1) = lo(x, win.1, w);
                            for ky in y0..y1 {
                                for kx in x0..x1 {
                                    let at = ((ky as usize * w + kx as usize) * kvh + g) * hd;
                                    scores.push(qv.iter().zip(&k[at..at + hd]).map(|(a, b)| a * b).sum::<f32>() * scale);
                                    vals.push(&v[at..at + hd]);
                                }
                            }
                        }
                        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                        let sum: f32 = e.iter().sum();
                        let o = &mut out[(((t * h + y) * w + x) * heads + hh) * hd..][..hd];
                        for (p, v) in e.iter().zip(&vals) {
                            for (oo, vv) in o.iter_mut().zip(v.iter()) {
                                *oo += p / sum * vv;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The window attention kernel against [`window_ref`]: the keyframe model's uniform planes (bf16, clamped, 128
    /// wide) and the Flash decoder's explicit ones (fp16, cut at the border, grouped heads). Needs a GPU.
    #[test]
    #[ignore]
    fn window_attention_matches_eager() {
        crate::cuda::init(0).unwrap();
        let st = Stream::new().unwrap();
        let mut seed = 1u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 4.0
                })
                .collect()
        };
        let round = |v: Vec<f32>, f16: bool| -> Vec<f32> {
            v.into_iter().map(|x| if f16 { half::f16::from_f32(x).to_f32() } else { bf16::from_f32(x).to_f32() }).collect()
        };
        let bytes = |v: &[f32], f16: bool| if f16 { f16_bytes(v) } else { bf16_bytes(v) };
        let back = |b: &DevBuf, n: usize, f16: bool| -> Vec<f32> {
            let mut raw = vec![0u8; n * 2];
            b.download(0, &mut raw, &st).unwrap();
            raw.chunks_exact(2).map(|c| if f16 { half::f16::from_le_bytes([c[0], c[1]]).to_f32() } else { bf16::from_le_bytes([c[0], c[1]]).to_f32() }).collect()
        };
        let check = |ours: &[f32], theirs: &[f32], what: &str| {
            let err = ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let mag = theirs.iter().map(|x| x.abs()).fold(0f32, f32::max);
            eprintln!("{what}: max error {err} of {mag}");
            assert!(err < 0.03 * mag.max(1.0), "{what}: max error {err}");
        };

        // uniform: 2 cached planes then 3 query planes, each attending up to 4, 13x37, window 5x11 clamped
        {
            let (h, w, heads, hd, tq, tk) = (13usize, 37usize, 2usize, 128usize, 3usize, 5usize);
            let plane = h * w * heads * hd;
            let q = round(rnd(tq * plane), false);
            let k = round(rnd(tk * plane), false);
            let v = round(rnd(tk * plane), false);
            let planes: Vec<_> = (0..tk).map(|p| (k[p * plane..][..plane].to_vec(), v[p * plane..][..plane].to_vec(), (5, 11))).collect();
            let per_q: Vec<Vec<usize>> = (0..tq).map(|t| { let last = t + tk - tq; (last.saturating_sub(3)..=last).collect() }).collect();
            let theirs = window_ref(&q, &planes, &per_q, (tq, h, w), heads, heads, hd, true);
            let (qd, kd, vd) = (upload(&bytes(&q, false), &st).unwrap(), upload(&bytes(&k, false), &st).unwrap(), upload(&bytes(&v, false), &st).unwrap());
            let od = DevBuf::alloc(q.len() * 2).unwrap();
            let rs = (heads * hd) as i64;
            let mut a = WinArgs::new(qd.ptr(), rs, od.ptr(), rs, h, w, heads, heads, true);
            a.q_ps = plane as i64;
            a.o_ps = plane as i64;
            a.kb = kd.ptr();
            a.vb = vd.ptr();
            a.k_ps = plane as i64;
            a.v_ps = plane as i64;
            a.k_rs = rs;
            a.v_rs = rs;
            (a.uwh, a.uww, a.tk, a.tq, a.tw) = (5, 11, tk as i32, tq as i32, 4);
            k!(gpu_win_attn(DT_BF16, hd as i32, &a, 1.0 / (hd as f32).sqrt(), s(&st))).unwrap();
            check(&back(&od, q.len(), false), &theirs, "uniform bf16");
        }
        // explicit: 3 planes (two 7x9 windows, one 3x5), 20x45, 8 heads over 2 groups, cut at the border
        {
            let (h, w, heads, kvh, hd) = (20usize, 45usize, 8usize, 2usize, 64usize);
            let q = round(rnd(h * w * heads * hd), true);
            let kplane = h * w * kvh * hd;
            let wins = [(7usize, 9usize), (7, 9), (3, 5)];
            let planes: Vec<_> = wins.iter().map(|&win| (round(rnd(kplane), true), round(rnd(kplane), true), win)).collect();
            let theirs = window_ref(&q, &planes, &[vec![0, 1, 2]], (1, h, w), heads, kvh, hd, false);
            let qd = upload(&bytes(&q, true), &st).unwrap();
            let od = DevBuf::alloc(q.len() * 2).unwrap();
            let bufs: Vec<_> = planes.iter().map(|(k, v, _)| (upload(&bytes(k, true), &st).unwrap(), upload(&bytes(v, true), &st).unwrap())).collect();
            let rs = (kvh * hd) as i64;
            let mut a = WinArgs::new(qd.ptr(), (heads * hd) as i64, od.ptr(), (heads * hd) as i64, h, w, heads, kvh, false);
            for ((k, v), (_, _, win)) in bufs.iter().zip(&planes) {
                a = a.plane(k.ptr(), rs, v.ptr(), rs, *win);
            }
            k!(gpu_win_attn(DT_F16, hd as i32, &a, 1.0 / (hd as f32).sqrt(), s(&st))).unwrap();
            check(&back(&od, q.len(), true), &theirs, "explicit fp16");
        }
    }

    #[test]
    fn chunks_like_comfy() {
        // 8160 tokens a keyframe: 4 per chunk, but up to 11 keyframes go through whole
        assert_eq!(Dit::chunk(11, 8160), 11);
        assert_eq!(Dit::chunk(12, 8160), 4);
        assert_eq!(Dit::chunk(3, 100_000), 3);
        assert_eq!(Dit::chunk(9, 100_000), 1);
    }
}
