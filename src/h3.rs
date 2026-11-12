//! MiniMax H3: one transformer over packed [text ; audio ; video] tokens. Video latents have 24
//! channels at /16 with a 2x2 patch, audio 32 channels at 40 Hz in two stereo rows. Blocks hold w6a8
//! weights, decoded to int8 just before each GEMM so VRAM holds the packed form, and modulation is
//! per segment from a small basis of the time curve. bf16 activations like ComfyUI.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, DT_F32, PM, PRO_ADALN, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, Plan, Store, WeightSet};
use crate::st::{f32_bytes, Dt, SafeTensors};
use anyhow::{bail, Context, Result};
use half::bf16;
use std::sync::Arc;

const F: usize = 5376;
const H: usize = 56;
const HD: usize = 128;
const INNER: usize = H * HD; // 7168
const QKV: usize = 3 * INNER;
const FFN: usize = 14336;
pub const TXT: usize = 5120;
const VCH: usize = 24;
const PATCH: usize = 4 * VCH;
const ACH: usize = 32;
const G: i32 = 32;
const EPS: f32 = 1e-5;
const ROPE_PAIRS: usize = 48;
const REFINER: usize = 2;
const MODS: usize = 18 * F; // 3 modalities x 6 vectors
pub const SHIFT_VIDEO: f64 = 12.0;
pub const SHIFT_AUDIO: f64 = 3.0;
const FRAME_RESCALE: f64 = 5.0 / 3.0;
/// `VISUAL_COND_TIMESTEP`: keyframe and reference rows run at this time or later, and are mixed with this much of
/// themselves and the rest noise
const COND_T: f32 = 0.999;
pub(crate) const FRAME_PER_TOKEN: [usize; 5] = [1, 4, 4, 4, 4];
const MODULES: [(&str, &str); 4] = [("qkv", "attn.qkv_proj"), ("o", "attn.out_proj"), ("up", "mlp.fc1"), ("d", "mlp.fc2")];

/// True if `has` finds the keys of a MiniMax H3 diffusion model.
pub fn is_h3(has: impl Fn(&str) -> bool) -> bool {
    has("video_patch_proj.weight") && has("audio_patch_proj.weight")
}

/// True for MiniMax-H3-Image: the H3 transformer without its audio stream, denoising one latent frame.
pub fn is_h3_image(has: impl Fn(&str) -> bool) -> bool {
    has("video_patch_proj.weight") && has("condition_proj.weight") && !has("audio_patch_proj.weight")
}

/// Frames of a clip: the model's 17k + 5 grid, at or above `length`.
pub fn align_frames(length: usize) -> usize {
    let mut n = length.max(5);
    while n % 17 != 5 {
        n += 1;
    }
    n
}

pub fn video_latent_t(frames: usize) -> usize {
    if frames <= 5 { 2 } else { (frames - 5) / 17 * 5 + 2 }
}

pub fn audio_latent_t(frames: usize) -> usize {
    (frames as f64 / 24.0 * 40.0).round_ties_even() as usize
}

/// The shape of one generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// latent frames, height and width
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub audio_t: usize,
}

impl Shape {
    pub fn new(width: usize, height: usize, frames: usize) -> Self {
        Shape { t: video_latent_t(frames), h: height / 16, w: width / 16, audio_t: audio_latent_t(frames) }
    }

    /// One image for MiniMax-H3-Image: a single latent frame and no audio.
    pub fn image(width: usize, height: usize) -> Self {
        Shape { t: 1, h: height / 16, w: width / 16, audio_t: 0 }
    }

    pub fn video_len(&self) -> usize {
        VCH * self.t * self.h * self.w
    }

    pub fn audio_len(&self) -> usize {
        ACH * 2 * self.audio_t
    }

    fn video_rows(&self) -> usize {
        self.t * self.h.div_ceil(2) * self.w.div_ceil(2)
    }
}

/// `time_shift_sigma` from the video schedule to the audio one.
pub fn audio_sigma(sigma: f32, (from, to): (f32, f32)) -> f32 {
    let base = sigma / (from + sigma * (1.0 - from));
    to * base / (1.0 + (to - 1.0) * base)
}

/// `_axis_from_sqrt_area`
fn axis(dim: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / 2;
    (0..n).map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0).collect()
}

/// `_frame_grid`: the (h, w) of a latent frame's 2x2 patches, for latent sides padded to even.
fn frame_grid(h: usize, w: usize) -> Vec<[f64; 2]> {
    let (lh, lw) = (h.div_ceil(2) * 2, w.div_ceil(2) * 2);
    let area = ((lh * lw) as f64).sqrt();
    let (ys, xs) = (axis(lh, area), axis(lw, area));
    ys.iter().flat_map(|y| xs.iter().map(move |x| [*y, *x])).collect()
}

/// What a prompt packs beside its text and the targets: keyframes, then reference pictures, as `PackedLayout` lays them
/// out in [text | keyframes | references | audio | video].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extra {
    /// each keyframe's pixel frame in the clip and its latent frames, at the target's size
    pub keyframes: Vec<(usize, usize)>,
    /// each reference picture's latent height and width
    pub refs: Vec<(usize, usize)>,
}

impl Extra {
    /// Its rows in the sequence, for a target of shape `sh`.
    pub fn rows(&self, sh: &Shape) -> usize {
        let frame = sh.h.div_ceil(2) * sh.w.div_ceil(2);
        self.keyframes.iter().map(|k| k.1 * frame).sum::<usize>() + self.refs.iter().map(|(h, w)| h.div_ceil(2) * w.div_ceil(2)).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty() && self.refs.is_empty()
    }
}

/// `_video_t_grid`: the time of each of `n` latent frames from `origin`.
fn frame_times(n: usize, origin: f64) -> impl Iterator<Item = f64> {
    (0..n).scan(origin, |t, f| {
        let at = *t;
        *t += FRAME_RESCALE * FRAME_PER_TOKEN[f % 5] as f64;
        Some(at)
    })
}

/// `PackedLayout.position_ids`: (t, h, w) of each token.
pub fn positions(text: usize, sh: &Shape, extra: &Extra) -> Vec<[f64; 3]> {
    let frame = frame_grid(sh.h, sh.w);
    let mut p = Vec::with_capacity(text + extra.rows(sh) + 2 * sh.audio_t + sh.video_rows());
    p.extend((0..text).map(|i| [i as f64, 0.0, 0.0]));
    // a reference takes one time step, ahead of the targets
    let cursor = (text + extra.refs.len()) as f64;
    for &(index, t) in &extra.keyframes {
        for at in frame_times(t, cursor + FRAME_RESCALE * index as f64) {
            p.extend(frame.iter().map(|[y, x]| [at, *y, *x]));
        }
    }
    for (i, &(h, w)) in extra.refs.iter().enumerate() {
        p.extend(frame_grid(h, w).iter().map(|[y, x]| [(text + i) as f64, *y, *x]));
    }
    // the audio rows sit at the frame's outer columns
    for w in [frame[0][1], frame[frame.len() - 1][1]] {
        p.extend((0..sh.audio_t).map(|i| [cursor + i as f64, 0.0, w]));
    }
    for at in frame_times(sh.t, cursor) {
        p.extend(frame.iter().map(|[y, x]| [at, *y, *x]));
    }
    p
}

/// Rows of a latent `[24, t, h, w]` (h, w even), as `patchify_video` makes them: `[t * h/2 * w/2, (c, 2, 2)]`.
fn patchify(v: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    let (th, tw) = (h / 2, w / 2);
    let mut rows = vec![0f32; t * th * tw * PATCH];
    for f in 0..t {
        for y in 0..th {
            for xx in 0..tw {
                let r = &mut rows[(((f * th) + y) * tw + xx) * PATCH..][..PATCH];
                for c in 0..VCH {
                    for p in 0..2 {
                        for q in 0..2 {
                            r[c * 4 + p * 2 + q] = v[((c * t + f) * h + y * 2 + p) * w + xx * 2 + q];
                        }
                    }
                }
            }
        }
    }
    rows
}

/// How one prompt packs: per token of its text 1, or 0 for the vision tokens, which take the video modality (empty for
/// all text); and the embedded keyframe and reference rows of [`Dit::cond_rows`], with what they are.
#[derive(Clone, Copy, Default)]
pub struct Pack<'a> {
    pub tags: &'a [u8],
    pub cond: Option<(&'a DevBuf, &'a Extra)>,
}

/// RoPE table [rows][48][(cos, sin)], in bf16 steps as ComfyUI's rotation table is.
pub fn rope_table(pos: &[[f64; 3]], inv: &[f32]) -> Vec<f32> {
    let mut t = Vec::with_capacity(pos.len() * ROPE_PAIRS * 2);
    for p in pos {
        for a in p {
            for f in inv {
                let ang = *a as f32 * *f;
                t.push(bf16::from_f32(ang.cos()).to_f32());
                t.push(bf16::from_f32(ang.sin()).to_f32());
            }
        }
    }
    t
}

fn row_bytes(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(f32_bytes(&st.f32_vec(name, false)?))
}

/// True for HyperFlow H3 Flash in ComfyUI's single file, as `ComfyUI-HyperFlow-Flash` reads it: H3 conditioned on the
/// step's endpoint as well as its time.
pub fn is_hyperflow(has: impl Fn(&str) -> bool) -> bool {
    has("endpoint_time_embedder.proj_in.weight")
}

/// True if HyperFlow H3 Flash can sample with `sampler`. Its time embedding reads where each step ends, taken as the
/// schedule's next sigma (as HyperFlow's ComfyUI node takes it), so a sampler must evaluate only at the schedule's
/// sigmas and land on the next one: euler and res_multistep do; ancestral, SDE and midpoint samplers don't.
pub fn hyperflow_sampler_ok(sampler: &str) -> bool {
    matches!(sampler, "euler" | "res_multistep" | "res_multistep_cfg_pp")
}

/// What a HyperFlow H3 Flash file says of itself, if `path` is one: its `hyperflow` metadata
/// (`quantization.method`, `num_layers`, `dropped_blocks`, `variant` `fl2va` or `ref2va`).
pub fn hyperflow_config(path: &std::path::Path) -> Option<serde_json::Value> {
    hyperflow_comfy_config(&comfy_metadata(path)?)
}

/// True if `path` is a shard of HyperFlow's diffusers release, which isn't read here: its `config.json` is beside it.
pub fn is_hyperflow_diffusers(path: &std::path::Path) -> bool {
    let config = path.parent().map(|d| d.join("config.json"));
    let c = config.and_then(|c| std::fs::read(c).ok()).and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
    c.is_some_and(|c| c["_class_name"] == "HyperFlowH3FlashTransformer3DModel")
}

/// The refusal for [`is_hyperflow_diffusers`].
pub const HYPERFLOW_DIFFUSERS: &str = "HyperFlow H3 Flash's diffusers folders aren't read here; use ComfyUI's single file \
    from `fl2va_int8_convrot_comfyui/` or `ref2va_int8_convrot_comfyui/` in its repository";

/// The `__metadata__` of a safetensors file, read from its header alone.
fn comfy_metadata(path: &std::path::Path) -> Option<std::collections::HashMap<String, String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n).ok()?;
    let n = u64::from_le_bytes(n);
    if n > 64 << 20 {
        return None;
    }
    let mut h = vec![0u8; n as usize];
    f.read_exact(&mut h).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&h).ok()?;
    serde_json::from_value(v.get("__metadata__")?.clone()).ok()
}

/// ComfyUI-HyperFlow-Flash's `hyperflow` metadata, in the terms of HyperFlow's diffusers `config.json`.
fn hyperflow_comfy_config(meta: &std::collections::HashMap<String, String>) -> Option<serde_json::Value> {
    let m: serde_json::Value = serde_json::from_str(meta.get("hyperflow")?).ok()?;
    let blocks = m["blocks"].as_u64()?;
    let dropped = m["dropped_blocks"].as_array().cloned().unwrap_or_default();
    Some(serde_json::json!({
        "_class_name": "HyperFlowH3FlashTransformer3DModel",
        "layout": "comfyui",
        "format": m["format"],
        "variant": m["variant"],
        "quantization": { "method": m["weights"], "group": m["convrot_groupsize"] },
        "num_layers": blocks + dropped.len() as u64,
        "dropped_blocks": dropped,
    }))
}

/// The layout version of ComfyUI-HyperFlow-Flash's files that this reads.
const HYPERFLOW_COMFY_FORMAT: u64 = 1;

/// True if `quant` is ComfyUI's int8 ConvRot with 256-wide groups, which these kernels run.
fn is_convrot256(st: &SafeTensors, quant: &str) -> bool {
    let q = st.bytes(quant).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok());
    q.is_some_and(|q| q["format"] == "int8_tensorwise" && q["convrot"] == true && q["convrot_groupsize"] == 256)
}

/// How one linear of a block is stored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lin {
    /// ComfyUI's w6a8, decoded to int8 before its GEMM
    W6,
    /// HyperFlow's int8 ConvRot: W8A8 behind the same 256-wide Hadamard rotation
    Int8,
    /// HyperFlow's factorized `up(down(x))`, both bf16
    LowRank,
}

/// Where each linear's low-rank pairs start among a block's six (q, k, v, o, up, d).
const RANK_SLOT: [usize; 4] = [0, 3, 4, 5];
/// Low-rank ranks are zero-padded to this, so every stride is 16-byte aligned.
const RANK_ALIGN: usize = 8;

#[derive(Clone, Copy, Debug)]
struct Unit {
    lins: [Lin; 4],
    /// padded rank of each low-rank pair, 0 for the others
    ranks: [usize; 6],
}

impl Unit {
    /// Columns of the widest low-rank middle: q, k and v's together, or one other pair's.
    fn mid_width(&self) -> usize {
        (self.ranks[0] + self.ranks[1] + self.ranks[2]).max(self.ranks[3]).max(self.ranks[4]).max(self.ranks[5])
    }
}

/// Linear `m` of a block as low-rank pairs, one per part (`<part>.down.weight`, `<part>.up.weight`).
fn low_rank_layout(st: &SafeTensors, l: &mut Layout, u: &mut Unit, m: usize, parts: &[String]) -> Result<()> {
    u.lins[m] = Lin::LowRank;
    // ranks padded with zeros to multiples of 8 for aligned GEMMs; q, k and v share one down GEMM
    let mut downs = Vec::new();
    for (j, x) in parts.iter().enumerate() {
        let slot = RANK_SLOT[m] + j;
        let (down, up) = (format!("{x}.down.weight"), format!("{x}.up.weight"));
        let (di, ui) = (st.info(&down)?, st.info(&up)?);
        if di.dtype != Dt::BF16 || ui.dtype != Dt::BF16 {
            bail!("{down}: low-rank pairs must be bf16");
        }
        let r = di.shape[0].next_multiple_of(RANK_ALIGN);
        u.ranks[slot] = r;
        downs.push((down, r, di.shape[1]));
        l.padded(st, &format!("up{slot}"), &[(up, ui.shape[0], r)])?;
    }
    l.padded(st, &format!("dn{}", RANK_SLOT[m]), &downs)
}

/// The parts of each linear when ComfyUI's HyperFlow file stores it low-rank: q, k and v stay apart there.
const HFC_LOW_RANK: [&[&str]; 4] = [&["attn.q_proj", "attn.k_proj", "attn.v_proj"], &["attn.out_proj"], &["mlp.fc1"], &["mlp.fc2"]];

/// A block of HyperFlow's ComfyUI file: each linear int8 ConvRot under ComfyUI's names (q, k and v joined, the SwiGLU
/// input already `[gate; value]`), or low-rank pairs.
fn hfc_block_layout(st: &SafeTensors, i: usize) -> Result<(Layout, Unit)> {
    let n = |x: &str| format!("blocks.{i}.{x}");
    let mut l = Layout::new();
    let mut u = Unit { lins: [Lin::Int8; 4], ranks: [0; 6] };
    for (m, ((key, module), parts)) in MODULES.iter().zip(HFC_LOW_RANK).enumerate() {
        let w = n(&format!("{module}.weight"));
        if st.has(&w) {
            if st.info(&w)?.dtype != Dt::I8 || !is_convrot256(st, &n(&format!("{module}.comfy_quant"))) {
                bail!("{w}: HyperFlow H3 Flash's dense linears must be int8 ConvRot with 256-wide groups");
            }
            l.raw(st, &format!("w_{key}"), &[w])?;
            l.raw(st, &format!("c_{key}"), &[n(&format!("{module}.weight_scale"))])?;
        } else if parts.iter().all(|x| st.has(&n(&format!("{x}.down.weight")))) {
            low_rank_layout(st, &mut l, &mut u, m, &parts.iter().map(|x| n(x)).collect::<Vec<_>>())?;
        } else {
            bail!("{}: neither int8 ConvRot nor low-rank", n(module));
        }
    }
    l.bf16(st, "n1", &[n("norm1.weight")])?;
    l.bf16(st, "n2", &[n("norm2.weight")])?;
    l.f32(st, "qn", &n("attn.q_norm.weight"), 0.0)?;
    l.f32(st, "kn", &n("attn.k_norm.weight"), 0.0)?;
    Ok((l, u))
}

/// The small tensors of HyperFlow's ComfyUI file: H3's names, with float32 patch projections and heads.
fn hfc_globals_layout(st: &SafeTensors, blocks: usize) -> Result<Layout> {
    let mut g = Layout::new();
    for (key, name) in [("vid_w", "video_patch_proj.weight"), ("vid_b", "video_patch_proj.bias"), ("aud_w", "audio_patch_proj.weight"), ("aud_b", "audio_patch_proj.bias"),
        ("ao_w", "final_layer.audio_out.weight"), ("ao_b", "final_layer.audio_out.bias"), ("vo_w", "final_layer.video_out.weight"), ("vo_b", "final_layer.video_out.bias")]
    {
        if st.info(name)?.dtype != Dt::F32 {
            bail!("{name}: expected float32");
        }
        g.raw(st, key, &[name.to_string()])?;
    }
    g.bf16(st, "cond_w", &["condition_proj.weight".to_string()])?;
    g.bf16(st, "cond_b", &["condition_proj.bias".to_string()])?;
    g.bf16(st, "fn", &["final_layer.norm.weight".to_string()])?;
    g.f32(st, "rn", "token_refiner.final_norm.weight", 0.0)?;
    g.bytes("fa_w", row_bytes(st, "final_layer.adaln_proj.linear.weight")?);
    g.bytes("fa_b", row_bytes(st, "final_layer.adaln_proj.linear.bias")?);
    for i in 0..blocks {
        g.bytes(&format!("a{i}_w"), row_bytes(st, &format!("blocks.{i}.adaln_proj.linear.weight"))?);
        g.bytes(&format!("a{i}_b"), row_bytes(st, &format!("blocks.{i}.adaln_proj.linear.bias"))?);
    }
    Ok(g)
}

/// Diffusers' `Timesteps(256, flip_sin_to_cos=True)` then `TimestepEmbedding`, in float32 on the host.
struct TimeMlp {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

const FREQ: usize = 256;

impl TimeMlp {
    /// `p`'s two linears, `proj_in` and `proj_out` in ComfyUI's `TimeEmbedder`.
    fn load(st: &SafeTensors, p: &str) -> Result<Self> {
        let v = |x: &str| st.f32_vec(&format!("{p}.{x}"), false);
        let m = TimeMlp { w1: v("proj_in.weight")?, b1: v("proj_in.bias")?, w2: v("proj_out.weight")?, b2: v("proj_out.bias")? };
        if m.w1.len() != m.b1.len() * FREQ || m.w2.len() != m.b2.len() * m.b1.len() {
            bail!("{p}: unexpected shapes");
        }
        Ok(m)
    }

    fn at(&self, t: f32) -> Vec<f32> {
        let half = FREQ / 2;
        let emb: Vec<f32> = (0..FREQ)
            .map(|j| {
                let f = (-(10000f32).ln() * (j % half) as f32 / half as f32).exp();
                if j < half { (t * f).cos() } else { (t * f).sin() }
            })
            .collect();
        let hid: Vec<f32> = self.b1.iter().enumerate().map(|(o, b)| silu(b + dot(&self.w1[o * FREQ..][..FREQ], &emb))).collect();
        let k = hid.len();
        self.b2.iter().enumerate().map(|(o, b)| b + dot(&self.w2[o * k..][..k], &hid)).collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// HyperFlow's two-time conditioning: `temb(t, r) = emb_t(t) + gate * (emb_r(r) - emb_t(t))`.
struct Hyper {
    gate: f32,
    te: TimeMlp,
    ee: TimeMlp,
    /// by the bits of (t, r): a schedule repeats its pairs every run, and CFG asks twice per step
    cache: std::collections::HashMap<(u32, u32), Vec<f32>>,
}

impl Hyper {
    /// What the AdaLN projections read: `silu(temb)`, rounded to their bf16. ComfyUI's model also rounds `temb` to bf16
    /// before the SiLU.
    fn adaln_input(&mut self, t: f32, r: f32) -> Vec<f32> {
        let (te, ee, gate) = (&self.te, &self.ee, self.gate);
        let rnd = |v: f32| bf16::from_f32(v).to_f32();
        self.cache
            .entry((t.to_bits(), r.to_bits()))
            .or_insert_with(|| {
                let (a, b) = (te.at(t), ee.at(r));
                a.iter().zip(&b).map(|(a, b)| a + gate * (b - a)).map(|e| rnd(silu(rnd(e)))).collect()
            })
            .clone()
    }
}

/// One up projection of a low-rank pair: it reads `rank` middle columns from `col`, and writes rows `first..first + rows`
/// of `w` to `out`, `ldo` apart.
struct Up {
    col: usize,
    rank: usize,
    w: PM,
    first: usize,
    rows: usize,
    out: PM,
    ldo: usize,
}

/// `x · downᵀ` into `width` middle columns, then each up projection: diffusers' two bf16 linears.
#[allow(clippy::too_many_arguments)]
fn low_rank(x: PM, ldx: usize, k: usize, down: PM, width: usize, ups: &[Up], mid: PM, l: usize, st: &Stream) -> Result<()> {
    let wd = width as i64;
    k!(gpu_gemm(x, ldx as i64, down, k as i64, mid, wd, null(), l as i64, wd, k as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
    for u in ups {
        k!(gpu_gemm(off(mid, u.col * 2), wd, off(u.w, u.first * u.rank * 2), u.rank as i64, u.out, u.ldo as i64, null(), l as i64, u.rows as i64, u.rank as i64, 1, DT_BF16,
            DT_BF16, 1.0, 0.0, s(st)))?;
    }
    Ok(())
}

pub fn block_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("blocks.{i}.{x}");
    let mut l = Layout::new();
    for (key, m) in MODULES {
        l.raw(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
        l.raw(st, &format!("r_{key}"), &[n(&format!("{m}.weight_s_rel"))])?;
        l.raw(st, &format!("c_{key}"), &[n(&format!("{m}.weight_s_channel"))])?;
    }
    l.bf16(st, "n1", &[n("norm1.weight")])?;
    l.bf16(st, "n2", &[n("norm2.weight")])?;
    l.f32(st, "qn", &n("attn.q_norm.weight"), 0.0)?;
    l.f32(st, "kn", &n("attn.k_norm.weight"), 0.0)?;
    Ok(l)
}

fn refiner_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("token_refiner.blocks.{i}.{x}");
    let mut l = Layout::new();
    for (key, m) in MODULES {
        l.raw(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
        l.raw(st, &format!("s_{key}"), &[n(&format!("{m}.weight_scale"))])?;
    }
    l.f32(st, "n1", &n("norm1.weight"), 0.0)?;
    l.f32(st, "n2", &n("norm2.weight"), 0.0)?;
    l.f32(st, "qn", &n("attn.q_norm.weight"), 0.0)?;
    l.f32(st, "kn", &n("attn.k_norm.weight"), 0.0)?;
    Ok(l)
}

/// A MiniMax-H3-Image block: ComfyUI's int8 ConvRot, run like HyperFlow's int8 linears.
fn int8_block_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("blocks.{i}.{x}");
    let mut l = Layout::new();
    for (key, m) in MODULES {
        l.raw(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
        l.raw(st, &format!("c_{key}"), &[n(&format!("{m}.weight_scale"))])?;
    }
    l.bf16(st, "n1", &[n("norm1.weight")])?;
    l.bf16(st, "n2", &[n("norm2.weight")])?;
    l.f32(st, "qn", &n("attn.q_norm.weight"), 0.0)?;
    l.f32(st, "kn", &n("attn.k_norm.weight"), 0.0)?;
    Ok(l)
}

/// MiniMax-H3-Image's token refiner is plain bf16, under ComfyUI's names.
fn bf16_refiner_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("token_refiner.blocks.{i}.{x}");
    let mut l = Layout::new();
    for (key, m) in MODULES {
        l.bf16(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
    }
    l.f32(st, "n1", &n("norm1.weight"), 0.0)?;
    l.f32(st, "n2", &n("norm2.weight"), 0.0)?;
    l.f32(st, "qn", &n("attn.q_norm.weight"), 0.0)?;
    l.f32(st, "kn", &n("attn.k_norm.weight"), 0.0)?;
    Ok(l)
}

/// MiniMax-H3-Image's small tensors. Its video head is a PDD bank, kept on the host in [`Pdd`].
fn image_globals_layout(st: &SafeTensors, blocks: usize) -> Result<Layout> {
    let mut g = Layout::new();
    for (key, name) in [("vid_w", "video_patch_proj.weight"), ("vid_b", "video_patch_proj.bias")] {
        if st.info(name)?.dtype != Dt::F32 {
            bail!("{name}: expected float32");
        }
        g.raw(st, key, &[name.to_string()])?;
    }
    g.bf16(st, "cond_w", &["condition_proj.weight".to_string()])?;
    g.bf16(st, "cond_b", &["condition_proj.bias".to_string()])?;
    g.bf16(st, "fn", &["final_layer.norm.weight".to_string()])?;
    g.f32(st, "rn", "token_refiner.final_norm.weight", 0.0)?;
    g.bytes("fa_w", row_bytes(st, "final_layer.adaln_proj.linear.weight")?);
    g.bytes("fa_b", row_bytes(st, "final_layer.adaln_proj.linear.bias")?);
    for i in 0..blocks {
        g.bytes(&format!("a{i}_w"), row_bytes(st, &format!("blocks.{i}.adaln_proj.linear.weight"))?);
        g.bytes(&format!("a{i}_b"), row_bytes(st, &format!("blocks.{i}.adaln_proj.linear.bias"))?);
    }
    Ok(g)
}

/// The sigma grid MiniMax-H3-Image's PDD heads were distilled on, from 1 down to 0, from the file's metadata.
pub fn image_sigmas(st: &SafeTensors) -> Result<Vec<f64>> {
    let cfg: serde_json::Value = serde_json::from_str(st.metadata.get("config").context("MiniMax-H3-Image: no config in the file's metadata")?)?;
    let grid: Vec<f64> = cfg["transformer"]["pdd_sigmas"].as_array().context("MiniMax-H3-Image: no pdd_sigmas in its config")?.iter().filter_map(|v| v.as_f64()).collect();
    let steps = cfg["transformer"]["pdd_steps"].as_u64().unwrap_or(1) as usize;
    if grid.len() != steps + 1 || steps < 2 || grid.windows(2).any(|p| p[0] <= p[1]) || grid[0] != 1.0 || grid[steps] != 0.0 {
        bail!("MiniMax-H3-Image: {steps} PDD heads don't fit the grid {grid:?}");
    }
    Ok(grid)
}

/// A bank of PDD output heads, each distilled for one interval of `grid`. Row block 0 is a full head and the later
/// blocks are offsets from it; a step from `grid[start]` to `grid[stop]` takes the dt-weighted mean of the heads it
/// spans (`_pdd_head`).
struct Pdd {
    /// sigmas from 1 down to 0, one more than the heads
    grid: Vec<f64>,
    /// [heads][PATCH][F] and [heads][PATCH], float32
    w: Vec<f32>,
    b: Vec<f32>,
    /// the last combined head on the device, weight then bias, by its (start, stop)
    head: Option<((usize, usize), DevBuf)>,
}

impl Pdd {
    fn heads(&self) -> usize {
        self.grid.len() - 1
    }

    /// The grid intervals a step from `sigma` to `next` spans, as `MiniMaxH3FinalLayer.forward` finds them.
    fn span(&self, sigma: f32, next: f32) -> (usize, usize) {
        let n = self.heads();
        let near = |s: f32| (0..=n).min_by(|&a, &b| (self.grid[a] - s as f64).abs().total_cmp(&(self.grid[b] - s as f64).abs())).unwrap_or(0);
        let start = near(sigma).min(n - 1);
        (start, near(next).max(start + 1))
    }

    /// The head of the step over `start..stop`: [PATCH][F] weight then [PATCH] bias.
    fn combine(&self, (start, stop): (usize, usize)) -> Vec<f32> {
        let dt: Vec<f64> = (start..stop).map(|k| self.grid[k] - self.grid[k + 1]).collect();
        let sum: f64 = dt.iter().sum();
        let wts: Vec<f32> = dt.iter().map(|d| *d as f32 / sum as f32).collect();
        let (wn, bn) = (PATCH * F, PATCH);
        let mut out = Vec::with_capacity(wn + bn);
        out.extend_from_slice(&self.w[..wn]);
        out.extend_from_slice(&self.b[..bn]);
        for k in start.max(1)..stop {
            let c = wts[k - start];
            for (o, v) in out[..wn].iter_mut().zip(&self.w[k * wn..][..wn]) {
                *o += c * v;
            }
            for (o, v) in out[wn..].iter_mut().zip(&self.b[k * bn..][..bn]) {
                *o += c * v;
            }
        }
        out
    }
}

fn globals_layout(st: &SafeTensors, blocks: usize) -> Result<Layout> {
    let mut g = Layout::new();
    for (key, name) in [("vid_w", "video_patch_proj.weight"), ("vid_b", "video_patch_proj.bias"), ("aud_w", "audio_patch_proj.weight"), ("aud_b", "audio_patch_proj.bias"),
        ("ao_w", "final_layer.audio_out.weight"), ("ao_b", "final_layer.audio_out.bias"), ("vo_b", "final_layer.video_out.bias"), ("vo_w", "final_layer.video_out.weight"),
        ("vo_r", "final_layer.video_out.weight_s_rel"), ("vo_c", "final_layer.video_out.weight_s_channel")]
    {
        g.raw(st, key, &[name.to_string()])?;
    }
    g.bf16(st, "cond_w", &["condition_proj.weight".to_string()])?;
    g.bf16(st, "cond_b", &["condition_proj.bias".to_string()])?;
    g.bf16(st, "fn", &["final_layer.norm.weight".to_string()])?;
    g.f32(st, "rn", "token_refiner.final_norm.weight", 0.0)?;
    g.bytes("fa_w", row_bytes(st, "final_layer.adaln_proj.linear.weight")?);
    g.bytes("fa_b", row_bytes(st, "final_layer.adaln_proj.linear.bias")?);
    for i in 0..blocks {
        g.bytes(&format!("a{i}_w"), row_bytes(st, &format!("blocks.{i}.adaln_proj.linear.weight"))?);
        g.bytes(&format!("a{i}_b"), row_bytes(st, &format!("blocks.{i}.adaln_proj.linear.bias"))?);
    }
    Ok(g)
}

pub fn n_blocks(st: &SafeTensors) -> usize {
    (0..).take_while(|i| st.has(&format!("blocks.{i}.attn.qkv_proj.weight"))).count()
}

/// Rough activation footprint of one evaluation (bytes) for `l` tokens.
/// ComfyUI's `BlockSparseAttention` on the blocks' self-attention, through comfy-kitchen's Sol-Attn kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseAttn {
    pub method: SparseMethod,
    /// Sparse while the video sigma is in `sigma_end..=sigma_start`, dense outside.
    pub sigma_start: f32,
    pub sigma_end: f32,
    /// Shorter sequences stay dense.
    pub min_tokens: usize,
    /// Blocks that always run dense, by their index in the checkpoint.
    pub dense_blocks: Vec<usize>,
    pub sinks: SparseSinks,
    /// Unrouted tokens each query block attends exactly: 0, or a multiple of 64 up to 256.
    pub extra_tokens: usize,
    /// Log, once per sequence length, whether the attention ran sparse.
    pub verbose: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SparseMethod {
    /// The training-free adaptive threshold, in sigmas of each query block's score distribution.
    Sol { tau: f32 },
    /// SLA-style: keep this fraction of the key blocks of every query block.
    TopK { ratio: f64 },
}

/// What of the packed text and audio rows stays exact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SparseSinks {
    Off,
    /// Every query attends the text and audio rows exactly.
    ExactKv,
    /// That, and the audio query rows attend everything exactly.
    ExactKvAndRows,
}

impl SparseAttn {
    /// Key blocks `_topk_count` keeps besides the sinks, as comfy-kitchen counts them.
    fn topk_count(ratio: f64, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let k = (ratio * n as f64).round_ties_even() as i64;
        k.max(1).min(n as i64 - 1) as usize
    }

    /// Workspace bytes for a sequence of `l` tokens.
    fn bytes(&self, l: usize) -> usize {
        let ws = unsafe { crate::ffi::gpu_sol_attn_workspace(l as i32, H as i32, self.extra_tokens as i32) }.max(0) as usize;
        ws + match self.method {
            SparseMethod::Sol { .. } => 0,
            SparseMethod::TopK { .. } => (unsafe { crate::ffi::gpu_sol_topk_workspace(l as i32, H as i32) }) as usize + H * l.div_ceil(64) * 4,
        }
    }
}

pub fn step_activation_bytes(l: usize) -> usize {
    l * (F * 2 + INNER * 2 + 2 * FFN * 2 + FFN + 1024) + 4 * 28672 * F / 3 + (128 << 20)
}

/// A sparse call's buffers: the Sol-Attn workspace, and top-k's thresholds and their scratch.
struct SparseWork {
    ws: DevBuf,
    thr: Option<DevBuf>,
    scratch: Option<DevBuf>,
}

pub struct Dit {
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    /// how each block's linears are stored
    units: Vec<Unit>,
    refiner: WeightSet,
    table: Vec<f32>,
    inv_freq: Vec<f32>,
    /// width of what the AdaLN projections read: the curve basis, or HyperFlow's time embedding
    tdim: usize,
    hyper: Option<Hyper>,
    /// columns of the low-rank middle buffer
    mid_width: usize,
    pub int8_attn: bool,
    pub attn_variant: i32,
    /// block-sparse self-attention, when set
    pub sparse: Option<SparseAttn>,
    /// sequence lengths whose sparse or dense choice was logged
    sparse_logged: Vec<(usize, bool)>,
    /// each block's index in the checkpoint
    block_ids: Vec<usize>,
    /// video, audio
    pub shifts: (f32, f32),
    /// MiniMax-H3-Image's PDD head bank: set for the image model, which has no audio stream
    pdd: Option<Pdd>,
    /// the token refiner is plain bf16
    dense_refiner: bool,
    pub prof: crate::prof::Prof,
}

impl Dit {
    /// Reads the small tensors. The blocks take no VRAM until [`Self::reserve`], and are read from the
    /// file as they are used.
    pub fn load(st: &Arc<SafeTensors>, threads: usize, s0: &Stream) -> Result<Self> {
        let (gl, layouts, units, ref_layouts, table, inv_freq, tdim, hyper, block_ids);
        let (mut pdd, mut dense_refiner) = (None, false);
        if is_hyperflow(|k| st.has(k)) {
            dense_refiner = true;
            let cfg = hyperflow_comfy_config(&st.metadata).context("HyperFlow H3 Flash: no `hyperflow` metadata in the file")?;
            if cfg["format"].as_u64().is_none_or(|f| f > HYPERFLOW_COMFY_FORMAT) {
                bail!("HyperFlow H3 Flash: layout {} of ComfyUI-HyperFlow-Flash's files; this reads up to {HYPERFLOW_COMFY_FORMAT}", cfg["format"]);
            }
            let q = &cfg["quantization"];
            if q["method"] != "int8_convrot" || q["group"] != 256 {
                bail!("HyperFlow H3 Flash: only the int8 ConvRot release (group 256) runs here, not {q}");
            }
            // dropped blocks are left out and the rest renumbered, as ComfyUI counts them
            let n = (0..).take_while(|i| st.has(&format!("blocks.{i}.norm1.weight"))).count();
            let blocks = cfg["num_layers"].as_u64().unwrap_or(0) as usize - cfg["dropped_blocks"].as_array().map_or(0, |a| a.len());
            if n != blocks {
                bail!("HyperFlow H3 Flash: {n} blocks in the file, {blocks} in its metadata");
            }
            let gate = *st.f32_vec("hyperflow_gate", false)?.first().context("HyperFlow H3 Flash: an empty hyperflow_gate")?;
            let (te, ee) = (TimeMlp::load(st, "time_embedder")?, TimeMlp::load(st, "endpoint_time_embedder")?);
            tdim = te.b2.len();
            if ee.b2.len() != tdim {
                bail!("HyperFlow H3 Flash: the two time embedders differ in width");
            }
            hyper = Some(Hyper { gate, te, ee, cache: Default::default() });
            gl = hfc_globals_layout(st, n)?;
            block_ids = (0..n).collect();
            (layouts, units) = (0..n).map(|i| hfc_block_layout(st, i)).collect::<Result<Vec<_>>>()?.into_iter().unzip();
            ref_layouts = (0..REFINER).map(|i| bf16_refiner_layout(st, i)).collect::<Result<Vec<_>>>()?;
            table = Vec::new();
            inv_freq = st.f32_vec("rope.inv_freq", false)?;
        } else if is_h3_image(|k| st.has(k)) {
            let quant = st.bytes("blocks.0.attn.qkv_proj.comfy_quant").ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok());
            match &quant {
                Some(q) if q["format"] == "int8_tensorwise" && q["convrot"] == true && q["convrot_groupsize"] == 256 => {}
                _ => bail!("MiniMax-H3-Image: only the int8 ConvRot release runs here (minimax_h3_image_pruned_int8_convrot), not {quant:?}"),
            }
            let grid = image_sigmas(st)?;
            let steps = grid.len() - 1;
            let (w, b) = (st.f32_vec("final_layer.video_out.weight", false)?, st.f32_vec("final_layer.video_out.bias", false)?);
            if w.len() != steps * PATCH * F || b.len() != steps * PATCH {
                bail!("MiniMax-H3-Image: {steps} PDD heads don't fit the head's {} values", w.len());
            }
            pdd = Some(Pdd { grid, w, b, head: None });
            dense_refiner = true;
            let n = n_blocks(st);
            block_ids = (0..n).collect();
            gl = image_globals_layout(st, n)?;
            layouts = (0..n).map(|i| int8_block_layout(st, i)).collect::<Result<Vec<_>>>()?;
            units = vec![Unit { lins: [Lin::Int8; 4], ranks: [0; 6] }; n];
            ref_layouts = (0..REFINER).map(|i| bf16_refiner_layout(st, i)).collect::<Result<Vec<_>>>()?;
            table = st.f32_vec("adaln_t_table", false)?;
            inv_freq = st.f32_vec("rope.inv_freq", false)?;
            tdim = 8;
            hyper = None;
        } else {
            let n = n_blocks(st);
            block_ids = (0..n).collect();
            gl = globals_layout(st, n)?;
            layouts = (0..n).map(|i| block_layout(st, i)).collect::<Result<Vec<_>>>()?;
            units = vec![Unit { lins: [Lin::W6; 4], ranks: [0; 6] }; n];
            ref_layouts = (0..REFINER).map(|i| refiner_layout(st, i)).collect::<Result<Vec<_>>>()?;
            table = st.f32_vec("adaln_t_table", false)?;
            inv_freq = st.f32_vec("rope.inv_freq", false)?;
            tdim = 8;
            hyper = None;
        }
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let mid_width = units.iter().map(Unit::mid_width).max().unwrap_or(0);
        let blocks = WeightSet::lazy("h3 blocks", st.clone(), layouts, Plan { resident: 0, slots: 0 }, threads)?;
        let refiner = WeightSet::lazy("h3 refiner", st.clone(), ref_layouts, Plan { resident: 0, slots: 0 }, threads)?;
        refiner.once_per_prompt();
        Ok(Dit { globals, gl, blocks, units, refiner, table, inv_freq, tdim, hyper, mid_width, int8_attn: true, attn_variant: 0,
            sparse: None, sparse_logged: Vec::new(), block_ids, shifts: (SHIFT_VIDEO as f32, SHIFT_AUDIO as f32), pdd, dense_refiner,
            prof: crate::prof::Prof::new() })
    }

    /// MiniMax-H3-Image, which samples one image without audio.
    pub fn is_image(&self) -> bool {
        self.pdd.is_some()
    }

    /// The sigmas MiniMax-H3-Image's heads were distilled on, from 1 down to 0.
    pub fn pdd_sigmas(&self) -> Option<Vec<f32>> {
        self.pdd.as_ref().map(|p| p.grid.iter().map(|g| *g as f32).collect())
    }

    /// The PDD head of a step from `sigma` to `next` on the device: [PATCH][F] weight then [PATCH] bias.
    fn pdd_head(&mut self, sigma: f32, next: f32, st: &Stream) -> Result<Option<PM>> {
        let Some(p) = &mut self.pdd else { return Ok(None) };
        let span = p.span(sigma, next);
        if p.head.as_ref().is_none_or(|(s, _)| *s != span) {
            let v = p.combine(span);
            let d = DevBuf::alloc(v.len() * 4)?;
            d.upload(0, &f32_bytes(&v), st)?;
            p.head = Some((span, d));
        }
        Ok(p.head.as_ref().map(|(_, d)| d.ptr()))
    }

    /// HyperFlow H3 Flash, which reads the step's endpoint.
    pub fn is_hyperflow(&self) -> bool {
        self.hyper.is_some()
    }

    /// Rough activation footprint of one evaluation (bytes) for `l` tokens.
    pub fn step_activation_bytes(&self, l: usize) -> usize {
        step_activation_bytes(l)
            + if self.hyper.is_some() { l * (FFN + self.mid_width) * 2 } else { 0 }
            + self.sparse.as_ref().filter(|sp| l >= sp.min_tokens).map_or(0, |sp| sp.bytes(l))
    }

    /// Give the blocks `budget` bytes of VRAM (what they hold now counts).
    pub fn reserve(&mut self, budget: usize, slots: usize) -> Result<()> {
        let p = plan(&self.blocks.layouts, budget, slots);
        self.blocks.replan(p)
    }

    pub fn blocks_vram(&self) -> usize {
        self.blocks.vram_bytes()
    }

    /// Take the blocks out of VRAM, for the decoders. Bakes survive; the next pass reads them back.
    pub fn park(&mut self) -> Result<()> {
        self.blocks.replan(Plan { resident: 0, slots: 0 })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Everywhere this model keeps LoRA-patchable weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Set(&mut self.refiner)]
    }

    /// `condition_proj` and the token refiner depend on the prompt alone, so they run once per prompt.
    pub fn encode_context(&mut self, cond: &DevBuf, len: usize, st: &Stream) -> Result<DevBuf> {
        self.refiner.replan(Plan { resident: 0, slots: 1 })?;
        let (l, m, fi) = (len, len as i32, F as i64);
        let c16 = DevBuf::alloc_async(l * TXT * 2, st)?;
        k!(gpu_f32_to_bf16(cond.ptr(), c16.ptr(), (l * TXT) as i64, s(st)))?;
        let x = DevBuf::alloc_async(l * F * 2, st)?;
        k!(gpu_gemm(c16.ptr(), TXT as i64, self.g("cond_w"), TXT as i64, x.ptr(), fi, self.g("cond_b"), l as i64, fi, TXT as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        let a8 = DevBuf::alloc_async(l * FFN, st)?;
        let asc = DevBuf::alloc_async(l * 4, st)?;
        let qkv = DevBuf::alloc_async(l * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(l * INNER * 2, st)?;
        let y = DevBuf::alloc_async(l * F * 2, st)?;
        let hid = DevBuf::alloc_async(l * 2 * FFN * 2, st)?;
        let (xp, q) = (x.ptr(), qkv.ptr());
        let dense = self.dense_refiner;
        let (h16, h16b) = if dense { (Some(DevBuf::alloc_async(l * F * 2, st)?), Some(DevBuf::alloc_async(l * FFN * 2, st)?)) } else { (None, None) };
        for b in 0..REFINER {
            let base = self.refiner.acquire(b, st)?;
            let lay = self.refiner.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            if let (Some(h), Some(hb)) = (&h16, &h16b) {
                let gemm = |a: PM, k: usize, w: &str, out: PM, n: usize| -> Result<()> {
                    k!(gpu_gemm(a, k as i64, wp(w), k as i64, out, n as i64, null(), l as i64, n as i64, k as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))
                };
                k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, wp("n1"), EPS, null(), null(), h.ptr(), fi, m, F as i32, s(st)))?;
                gemm(h.ptr(), F, "w_qkv", q, QKV)?;
                k!(gpu_qk_norm_rope(DT_BF16, HD as i32, q, QKV as i64, 0, INNER as i32, H as i32, H as i32, wp("qn"), wp("kn"), EPS, null(), 0, m, s(st)))?;
                k!(gpu_attn_small(DT_BF16, q, QKV as i64, off(q, INNER * 2), off(q, 2 * INNER * 2), QKV as i64, H as i32, H as i32, HD as i32, m, m, attn.ptr(), INNER as i64,
                    1.0 / (HD as f32).sqrt(), 0, s(st)))?;
                gemm(attn.ptr(), INNER, "w_o", y.ptr(), F)?;
                k!(gpu_add_gated_bf16(xp, fi, y.ptr(), fi, null(), m, F as i32, s(st)))?;
                k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, wp("n2"), EPS, null(), null(), h.ptr(), fi, m, F as i32, s(st)))?;
                gemm(h.ptr(), F, "w_up", hid.ptr(), 2 * FFN)?;
                k!(gpu_prologue_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * FFN) as i64, null(), 0, FFN as i64, null(), 0.0, null(), null(), hb.ptr(), FFN as i64, m,
                    FFN as i32, s(st)))?;
                gemm(hb.ptr(), FFN, "w_d", y.ptr(), F)?;
                k!(gpu_add_gated_bf16(xp, fi, y.ptr(), fi, null(), m, F as i32, s(st)))?;
                self.refiner.release(b, st)?;
                continue;
            }
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, wp("n1"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_qkv"), asc.ptr(), wp("s_qkv"), null(), q, l as i64, QKV as i64, fi, DT_BF16, s(st)))?;
            k!(gpu_qk_norm_rope(DT_BF16, HD as i32, q, QKV as i64, 0, INNER as i32, H as i32, H as i32, wp("qn"), wp("kn"), EPS, null(), 0, m, s(st)))?;
            k!(gpu_attn_small(DT_BF16, q, QKV as i64, off(q, INNER * 2), off(q, 2 * INNER * 2), QKV as i64, H as i32, H as i32, HD as i32, m, m, attn.ptr(), INNER as i64,
                1.0 / (HD as f32).sqrt(), 0, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), INNER as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, INNER as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_o"), asc.ptr(), wp("s_o"), null(), y.ptr(), l as i64, fi, INNER as i64, DT_BF16, s(st)))?;
            k!(gpu_add_gated_bf16(xp, fi, y.ptr(), fi, null(), m, F as i32, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, wp("n2"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_up"), asc.ptr(), wp("s_up"), null(), hid.ptr(), l as i64, (2 * FFN) as i64, fi, DT_BF16, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * FFN) as i64, null(), 0, FFN as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, FFN as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_d"), asc.ptr(), wp("s_d"), null(), y.ptr(), l as i64, fi, FFN as i64, DT_BF16, s(st)))?;
            k!(gpu_add_gated_bf16(xp, fi, y.ptr(), fi, null(), m, F as i32, s(st)))?;
            self.refiner.release(b, st)?;
        }
        let out = DevBuf::alloc_async(l * F * 2, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, self.g("rn"), 1e-5, null(), null(), out.ptr(), fi, m, F as i32, s(st)))?;
        st.sync()?;
        self.refiner.replan(Plan { resident: 0, slots: 0 })?;
        self.refiner.evict();
        Ok(out)
    }

    /// The embedded rows of keyframe and reference latents `[24, t, h, w]`, in the order they pack, bf16 `[rows, F]`:
    /// `_cond_video_rows`, which mixes each latent's rows with noise from the sampler's `seed`, the same for every one.
    pub fn cond_rows(&self, latents: &[(&[f32], usize, usize, usize)], seed: u64, st: &Stream) -> Result<DevBuf> {
        let mut rows = Vec::new();
        for &(v, t, h, w) in latents {
            anyhow::ensure!(h % 2 == 0 && w % 2 == 0 && v.len() == VCH * t * h * w, "a condition latent of {t}x{h}x{w}");
            let r = patchify(v, t, h, w);
            let noise = crate::sampling::torch_randn(seed, r.len());
            let (a, b) = (COND_T, (1.0 - COND_T as f64) as f32);
            rows.extend(r.iter().zip(&noise).map(|(x, n)| a * x + b * n));
        }
        let n = rows.len() / PATCH;
        let up = DevBuf::alloc_async(rows.len() * 4, st)?;
        up.upload(0, &f32_bytes(&rows), st)?;
        let f = DevBuf::alloc_async(n * F * 4, st)?;
        k!(gpu_gemm(up.ptr(), PATCH as i64, self.g("vid_w"), PATCH as i64, f.ptr(), F as i64, self.g("vid_b"), n as i64, F as i64, PATCH as i64, 1, DT_F32, DT_F32, 1.0, 0.0,
            s(st)))?;
        let out = DevBuf::alloc_async(n * F * 2, st)?;
        k!(gpu_f32_to_bf16(f.ptr(), out.ptr(), (n * F) as i64, s(st)))?;
        st.sync()?;
        Ok(out)
    }

    pub fn rope(&self, text: usize, sh: &Shape, extra: &Extra, st: &Stream) -> Result<DevBuf> {
        let t = rope_table(&positions(text, sh, extra), &self.inv_freq);
        let d = DevBuf::alloc_async(t.len() * 4, st)?;
        d.upload(0, &f32_bytes(&t), st)?;
        Ok(d)
    }

    /// The curve table at `t`, as the adaln projections read it.
    fn curve(&self, t: f32) -> [f32; 8] {
        let rows = self.table.len() / 8;
        let pos = t.clamp(0.0, 1.0) * (rows - 1) as f32;
        let i = (pos.floor() as usize).min(rows - 2);
        let w = pos - i as f32;
        std::array::from_fn(|j| self.table[i * 8 + j] + w * (self.table[(i + 1) * 8 + j] - self.table[i * 8 + j]))
    }

    pub fn attn_workspace(&self, l: usize) -> usize {
        if self.int8_attn {
            unsafe { crate::ffi::gpu_int8_attn_workspace(1, H as i32, H as i32, l as i32, l as i32) as usize }
        } else {
            H * l * 4
        }
    }

    /// Block-sparse attention over the packed rows: text and audio (the first `al`, audio from `text`), then video.
    #[allow(clippy::too_many_arguments)]
    fn sparse_attention(&self, sp: &SparseAttn, q: PM, out: PM, sw: &SparseWork, l: usize, (text, al): (usize, usize), st: &Stream) -> Result<()> {
        let sc = 1.0 / (HD as f32).sqrt();
        let (kp, vp) = (off(q, INNER * 2), off(q, 2 * INNER * 2));
        let prefix = al.div_ceil(64) as i32;
        let (sink, sink_q) = match sp.sinks {
            SparseSinks::Off => ((0, 0), (0, 0)),
            SparseSinks::ExactKv => ((0, prefix), (0, 0)),
            SparseSinks::ExactKvAndRows if al > text => ((0, prefix), ((text / 64) as i32, prefix)),
            SparseSinks::ExactKvAndRows => ((0, prefix), (0, prefix)),
        };
        let (tau, thr) = match sp.method {
            SparseMethod::Sol { tau } => (tau, null()),
            SparseMethod::TopK { ratio } => {
                let n = l.div_ceil(64);
                let sinks = (sink.1 - sink.0).max(0) as usize;
                let kk = SparseAttn::topk_count(ratio, n - sinks.min(n));
                let thr = sw.thr.as_ref().context("no top-k threshold buffer")?;
                k!(gpu_sol_topk_threshold(q, kp, QKV as i64, HD as i64, l as i32, H as i32, kk as i32, sc, sink.0, sink.1, thr.ptr(), sw.scratch.as_ref().unwrap().ptr(), s(st)))?;
                (1.3, thr.ptr() as crate::ffi::P)
            }
        };
        k!(gpu_sol_attn(q, kp, vp, out, sw.ws.ptr(), l as i32, H as i32, QKV as i64, HD as i64, tau, sc, thr, sink.0, sink.1, sink_q.0, sink_q.1, sp.extra_tokens as i32, s(st)))
    }

    /// The sparse settings and buffers for a call of `l` tokens at `sigma`, if it runs sparse.
    fn sparse_work(&mut self, l: usize, sigma: f32, st: &Stream) -> Result<Option<(SparseAttn, SparseWork)>> {
        let Some(sp) = self.sparse.clone() else { return Ok(None) };
        let reason = if sigma > sp.sigma_start || sigma < sp.sigma_end {
            Some(format!("sigma {sigma:.3} outside the start/end window"))
        } else if l < sp.min_tokens {
            Some(format!("{l} tokens < min_tokens {}", sp.min_tokens))
        } else {
            None
        };
        if sp.verbose && !self.sparse_logged.contains(&(l, reason.is_none())) {
            self.sparse_logged.push((l, reason.is_none()));
            match &reason {
                Some(r) => eprintln!("sparse attention: dense ({l} tokens): {r}"),
                None => eprintln!("sparse attention: {:?} over {l} tokens", sp.method),
            }
        }
        if reason.is_some() {
            return Ok(None);
        }
        let n = unsafe { crate::ffi::gpu_sol_attn_workspace(l as i32, H as i32, sp.extra_tokens as i32) };
        if n < 0 {
            bail!("sparse attention can't take {l} tokens with {} extra tokens", sp.extra_tokens);
        }
        let ws = DevBuf::alloc_async(n as usize, st)?;
        let (thr, scratch) = match sp.method {
            SparseMethod::Sol { .. } => (None, None),
            SparseMethod::TopK { .. } => {
                if l.div_ceil(64) * 4 > 96 * 1024 {
                    bail!("top-k sparse attention over {l} tokens: more key blocks than its selection holds");
                }
                let scratch = unsafe { crate::ffi::gpu_sol_topk_workspace(l as i32, H as i32) } as usize;
                (Some(DevBuf::alloc_async(H * l.div_ceil(64) * 4, st)?), Some(DevBuf::alloc_async(scratch, st)?))
            }
        };
        Ok(Some((sp, SparseWork { ws, thr, scratch })))
    }

    fn attention(&self, q: PM, out: PM, ws: PM, l: usize, st: &Stream) -> Result<()> {
        let (hi, hs, sc) = (H as i32, HD as i64, 1.0 / (HD as f32).sqrt());
        let (bs, rs, ob, or) = ((l * QKV) as i64, QKV as i64, (l * INNER) as i64, INNER as i64);
        let (kp, vp) = (off(q, INNER * 2), off(q, 2 * INNER * 2));
        if self.int8_attn {
            k!(gpu_int8_attn(q, kp, vp, out, ws, 1, l as i32, l as i32, hi, hi, bs, rs, hs, bs, rs, hs, bs, rs, hs, ob, or, hs, sc, s(st)))
        } else {
            k!(gpu_flash_attn(q, kp, vp, out, ws, 1, l as i32, l as i32, hi, hi, bs, rs, hs, bs, rs, hs, bs, rs, hs, ob, or, hs, sc, self.attn_variant, s(st)))
        }
    }

    /// One model evaluation. `x` is the sampler's pack (video, then audio carried on the video
    /// schedule); writes the denoised pack to `den`. `sigma_next` is where this step ends, which only HyperFlow reads.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, x_in: &DevBuf, sh: &Shape, ctx: &DevBuf, text: usize, pack: Pack, rope: &DevBuf, sigma: f32, sigma_next: f32, den: &DevBuf, st: &Stream,
        mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        anyhow::ensure!(pack.tags.is_empty() || pack.tags.len() == text, "{} token tags for {text} tokens", pack.tags.len());
        self.prof.mark("start", st)?;
        let t0 = std::time::Instant::now();
        let (vn, an) = (sh.video_len(), sh.audio_len());
        let mut raw = vec![0u8; (vn + an) * 4];
        x_in.download(0, &mut raw, st)?;
        let latents: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let (vx, ax) = latents.split_at(vn);

        let sigma_v = sigma.max(1e-6);
        let sigma_a = audio_sigma(sigma_v, self.shifts);
        let carry = sigma_a / sigma_v;
        let (th, tw) = (sh.h / 2, sh.w / 2);
        let vr = sh.t * th * tw;
        let ar = 2 * sh.audio_t;
        // [t, c, ...] -> rows of (c, y2, x2)
        let rows = patchify(vx, sh.t, sh.h, sh.w);
        let nc = pack.cond.map_or(0, |(_, e)| e.rows(sh));
        if nc > 0 && self.sparse.is_some() {
            bail!("keyframes and reference pictures with sparse attention");
        }
        let mut arows = vec![0f32; ar * ACH];
        for ch in 0..2 {
            for t in 0..sh.audio_t {
                for c in 0..ACH {
                    arows[(ch * sh.audio_t + t) * ACH + c] = ax[((c * 2 + ch) * sh.audio_t) + t] * carry;
                }
            }
        }

        self.prof.host("unpack", t0);
        // the audio rows start at a0, the video's at al
        let a0 = text + nc;
        let (al, l) = (a0 + ar, a0 + ar + vr);
        let fi = F as i64;
        let (t_v, t_a) = (1.0 - sigma_v, 1.0 - sigma_a);
        // time rows: video, audio, and the conditions'
        let curve: Vec<f32> = match &mut self.hyper {
            None => [self.curve(t_v), self.curve(t_a), self.curve(t_v.max(COND_T))].concat(),
            // text follows video; audio's endpoint is on its own schedule, as its time is
            // keyframe and reference rows have no endpoint of their own: HyperFlow's node looks their time up among the
            // targets' and, not finding it, gives them r = t
            Some(h) => {
                let next = sigma_next.clamp(0.0, 1.0);
                let (r_v, t_c) = (1.0 - next, t_v.max(COND_T));
                let r_c = if t_c == t_v { r_v } else { t_c };
                [h.adaln_input(t_v, r_v), h.adaln_input(t_a, 1.0 - audio_sigma(next, self.shifts)), h.adaln_input(t_c, r_c)].concat()
            }
        };
        let td = self.tdim;
        let trows = curve.len() / td;
        let cd = DevBuf::alloc_async(curve.len() * 4, st)?;
        cd.upload(0, &f32_bytes(&curve), st)?;

        // tokens
        let x = DevBuf::alloc_async(l * F * 2, st)?;
        // qkv, the gate-up rows and the f32 rows at either end are never live together
        let hid = DevBuf::alloc_async(l * 2 * FFN * 2, st)?;
        let (qkv, fbuf) = (&hid, &hid);
        let up = DevBuf::alloc_async((rows.len() + arows.len()) * 4, st)?;
        up.upload(0, &f32_bytes(&rows), st)?;
        up.upload(rows.len() * 4, &f32_bytes(&arows), st)?;
        k!(gpu_gemm(up.ptr(), PATCH as i64, self.g("vid_w"), PATCH as i64, fbuf.ptr(), fi, self.g("vid_b"), vr as i64, fi, PATCH as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
        k!(gpu_f32_to_bf16(fbuf.ptr(), off(x.ptr(), al * F * 2), (vr * F) as i64, s(st)))?;
        if ar > 0 {
            k!(gpu_gemm(off(up.ptr(), rows.len() * 4), ACH as i64, self.g("aud_w"), ACH as i64, fbuf.ptr(), fi, self.g("aud_b"), ar as i64, fi, ACH as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
            k!(gpu_f32_to_bf16(fbuf.ptr(), off(x.ptr(), a0 * F * 2), (ar * F) as i64, s(st)))?;
        }
        crate::cuda::memcpy_d2d(x.ptr(), ctx.ptr(), text * F * 2, st)?;
        if let Some((c, _)) = pack.cond.filter(|_| nc > 0) {
            crate::cuda::memcpy_d2d(off(x.ptr(), text * F * 2), c.ptr(), nc * F * 2, st)?;
        }
        self.prof.mark("embed", st)?;

        let a8 = DevBuf::alloc_async(l * FFN, st)?;
        let asc = DevBuf::alloc_async(l * 4, st)?;
        let attn = DevBuf::alloc_async(l * INNER * 2, st)?;
        let wdec = DevBuf::alloc_async(2 * FFN * F, st)?;
        let ws = DevBuf::alloc_async(self.attn_workspace(l), st)?;
        let sparse = self.sparse_work(l, sigma, st)?;
        let modf = DevBuf::alloc_async(trows * MODS * 4, st)?;
        let modb = DevBuf::alloc_async(trows * MODS * 2, st)?;
        let folded = DevBuf::alloc_async(6 * F * 2, st)?;
        // HyperFlow's low-rank pairs: the SwiGLU output in bf16, and the rank-wide middle
        let (lrin, mid) = match self.hyper {
            Some(_) => (Some(DevBuf::alloc_async(l * FFN * 2, st)?), Some(DevBuf::alloc_async(l * self.mid_width.max(8) * 2, st)?)),
            None => (None, None),
        };
        if let Some(p) = probe.as_mut() {
            p("x_in", &x, l * F * 2);
        }
        let (xp, q) = (x.ptr(), qkv.ptr());
        // (first token, tokens, time row, modality): the text in runs of its tags, the conditions at their own time, then
        // the targets; the image model has no audio rows
        let mut segs: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut i = 0;
        while i < text {
            let tag = pack.tags.get(i).copied().unwrap_or(1);
            let run = pack.tags.get(i..).map_or(text - i, |t| t.iter().take_while(|&&g| g == tag).count());
            segs.push((i, run, 0, tag as usize));
            i += run;
        }
        segs.extend([(text, nc, 2, 0), (a0, ar, 1, 2), (al, vr, 0, 0)].into_iter().filter(|s| s.1 > 0));
        let pdd_head = self.pdd_head(sigma, sigma_next, st)?;

        for b in 0..self.blocks.len() {
            let u = self.units[b];
            k!(gpu_gemm(cd.ptr(), td as i64, self.g(&format!("a{b}_w")), td as i64, modf.ptr(), MODS as i64, self.g(&format!("a{b}_b")), trows as i64, MODS as i64, td as i64, 1,
                DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
            k!(gpu_f32_to_bf16(modf.ptr(), modb.ptr(), (trows * MODS) as i64, s(st)))?;
            self.prof.mark("mod", st)?;
            let base = self.blocks.acquire(b, st)?;
            self.prof.mark("stream", st)?;
            let lay = self.blocks.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            let vecs = |seg: &(usize, usize, usize, usize), j: usize| off(modb.ptr(), ((seg.2 * 3 + seg.3) * 6 + j) * F * 2);
            let fold = |i: usize, j: usize| off(folded.ptr(), (i * 2 + j) * F * 2);
            let mid = mid.as_ref().map_or(null() as PM, |m| m.ptr());
            // the int8 weight of linear `m`: w6a8 decoded into `wdec`, or HyperFlow's as stored
            let weight = |m: usize, n: usize, kk: usize| -> Result<PM> {
                let key = MODULES[m].0;
                if u.lins[m] == Lin::W6 {
                    k!(gpu_w6_decode(wp(&format!("w_{key}")), wp(&format!("r_{key}")), wdec.ptr(), n as i64, kk as i32, G, s(st)))?;
                    Ok(wdec.ptr())
                } else {
                    Ok(wp(&format!("w_{key}")))
                }
            };
            // the normed and modulated rows of every segment: int8 for an int8 linear, else bf16 in `attn`
            let modulate = |m: usize, j: usize| -> Result<()> {
                for (i, seg) in segs.iter().enumerate() {
                    let (xs, rows, (sc, sh)) = (off(xp, seg.0 * F * 2), seg.1 as i32, (fold(i, j), vecs(seg, 3 * j)));
                    if u.lins[m] == Lin::LowRank {
                        k!(gpu_prologue_rows(DT_BF16, PRO_ADALN, xs, fi, null(), 0, 0, null(), EPS, sc, sh, off(attn.ptr(), seg.0 * F * 2), fi, rows, F as i32, s(st)))?;
                    } else {
                        k!(gpu_quant_rows(DT_BF16, PRO_ADALN, xs, fi, null(), 0, 0, null(), EPS, sc, sh, off(a8.ptr(), seg.0 * F), off(asc.ptr(), seg.0 * 4), rows, F as i32, 1, s(st)))?;
                    }
                }
                Ok(())
            };
            // x += gate * y, segment by segment, for a low-rank output in `y`
            let add_gated = |y: PM, j: usize| -> Result<()> {
                for seg in &segs {
                    k!(gpu_add_gated_bf16(off(xp, seg.0 * F * 2), fi, off(y, seg.0 * F * 2), fi, vecs(seg, j), seg.1 as i32, F as i32, s(st)))?;
                }
                Ok(())
            };
            let (dn, up, r) = (|slot: usize| wp(&format!("dn{slot}")), |slot: usize| wp(&format!("up{slot}")), u.ranks);
            for (i, seg) in segs.iter().enumerate() {
                k!(gpu_adaln_weight(wp("n1"), vecs(seg, 1), fold(i, 0), F as i32, s(st)))?;
                k!(gpu_adaln_weight(wp("n2"), vecs(seg, 4), fold(i, 1), F as i32, s(st)))?;
            }
            modulate(0, 0)?;
            self.prof.mark("quant", st)?;
            if u.lins[0] == Lin::LowRank {
                let ups: Vec<Up> =
                    (0..3).map(|j| Up { col: r[..j].iter().sum(), rank: r[j], w: up(j), first: 0, rows: INNER, out: off(q, j * INNER * 2), ldo: QKV }).collect();
                low_rank(attn.ptr(), F, F, dn(0), r[0] + r[1] + r[2], &ups, mid, l, st)?;
                self.prof.mark("lowrank", st)?;
            } else {
                let w = weight(0, QKV, F)?;
                self.prof.mark("w6", st)?;
                k!(gpu_int8_gemm(a8.ptr(), w, asc.ptr(), wp("c_qkv"), null(), q, l as i64, QKV as i64, fi, DT_BF16, s(st)))?;
                self.prof.mark("gemm", st)?;
            }
            k!(gpu_rope_bf16(q, QKV as i64, HD as i64, wp("qn"), EPS, rope.ptr(), l as i32, H as i32, HD as i32, ROPE_PAIRS as i32, s(st)))?;
            k!(gpu_rope_bf16(off(q, INNER * 2), QKV as i64, HD as i64, wp("kn"), EPS, rope.ptr(), l as i32, H as i32, HD as i32, ROPE_PAIRS as i32, s(st)))?;
            self.prof.mark("rope", st)?;
            match &sparse {
                Some((sp, sw)) if !sp.dense_blocks.contains(&self.block_ids[b]) => {
                    self.sparse_attention(sp, q, attn.ptr(), sw, l, (text, al), st)?;
                    self.prof.mark("sparse attn", st)?;
                }
                _ => {
                    self.attention(q, attn.ptr(), ws.ptr(), l, st)?;
                    self.prof.mark("attn", st)?;
                }
            }
            if b == 0 {
                if let Some(p) = probe.as_mut() {
                    p("b0.out_proj.in", &attn, l * INNER * 2);
                    p("b0.q", qkv, l * QKV * 2);
                }
            }
            if u.lins[1] == Lin::LowRank {
                // q, k and v are spent, so their buffer takes the output
                low_rank(attn.ptr(), INNER, INNER, dn(3), r[3], &[Up { col: 0, rank: r[3], w: up(3), first: 0, rows: F, out: hid.ptr(), ldo: F }], mid, l, st)?;
                add_gated(hid.ptr(), 2)?;
                self.prof.mark("lowrank", st)?;
            } else {
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), INNER as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), l as i32, INNER as i32, 1, s(st)))?;
                self.prof.mark("quant", st)?;
                let w = weight(1, F, INNER)?;
                self.prof.mark("w6", st)?;
                for seg in &segs {
                    let xs = off(xp, seg.0 * F * 2);
                    k!(gpu_int8_gemm_residual(off(a8.ptr(), seg.0 * INNER), w, off(asc.ptr(), seg.0 * 4), wp("c_o"), null(), vecs(seg, 2), xs, xs, seg.1 as i64, fi, INNER as i64,
                        DT_BF16, s(st)))?;
                }
            }
            self.prof.mark("gemm", st)?;
            modulate(2, 1)?;
            self.prof.mark("quant", st)?;
            if u.lins[2] == Lin::LowRank {
                // the up rows are [gate; value], as the SwiGLU kernels read them
                let half = |first: usize, out: PM| Up { col: 0, rank: r[4], w: up(4), first, rows: FFN, out, ldo: 2 * FFN };
                low_rank(attn.ptr(), F, F, dn(4), r[4], &[half(0, hid.ptr()), half(FFN, off(hid.ptr(), FFN * 2))], mid, l, st)?;
                self.prof.mark("lowrank", st)?;
            } else {
                let w = weight(2, 2 * FFN, F)?;
                self.prof.mark("w6", st)?;
                k!(gpu_int8_gemm(a8.ptr(), w, asc.ptr(), wp("c_up"), null(), hid.ptr(), l as i64, (2 * FFN) as i64, fi, DT_BF16, s(st)))?;
                self.prof.mark("gemm", st)?;
            }
            if u.lins[3] == Lin::LowRank {
                let lrin = lrin.as_ref().context("low-rank layers without their buffers")?;
                k!(gpu_prologue_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * FFN) as i64, null(), 0, FFN as i64, null(), 0.0, null(), null(), lrin.ptr(), FFN as i64, l as i32,
                    FFN as i32, s(st)))?;
                low_rank(lrin.ptr(), FFN, FFN, dn(5), r[5], &[Up { col: 0, rank: r[5], w: up(5), first: 0, rows: F, out: hid.ptr(), ldo: F }], mid, l, st)?;
                add_gated(hid.ptr(), 5)?;
                self.prof.mark("lowrank", st)?;
            } else {
                k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * FFN) as i64, null(), 0, FFN as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), l as i32, FFN as i32, 1,
                    s(st)))?;
                self.prof.mark("quant", st)?;
                let w = weight(3, F, FFN)?;
                self.prof.mark("w6", st)?;
                for seg in &segs {
                    let xs = off(xp, seg.0 * F * 2);
                    k!(gpu_int8_gemm_residual(off(a8.ptr(), seg.0 * FFN), w, off(asc.ptr(), seg.0 * 4), wp("c_d"), null(), vecs(seg, 5), xs, xs, seg.1 as i64, fi, FFN as i64,
                        DT_BF16, s(st)))?;
                }
            }
            self.prof.mark("gemm", st)?;
            self.blocks.release(b, st)?;
            if matches!(b, 0 | 1 | 49) {
                if let Some(p) = probe.as_mut() {
                    p(&format!("b{b}.out"), &x, l * F * 2);
                }
            }
        }

        // final layer: [2][shift | scale]
        let fm = DevBuf::alloc_async(2 * 2 * F * 4, st)?;
        k!(gpu_gemm(cd.ptr(), td as i64, self.g("fa_w"), td as i64, fm.ptr(), (2 * F) as i64, self.g("fa_b"), 2, (2 * F) as i64, td as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
        let fb = DevBuf::alloc_async(2 * 2 * F * 2, st)?;
        k!(gpu_f32_to_bf16(fm.ptr(), fb.ptr(), (4 * F) as i64, s(st)))?;
        let h16 = &attn;
        for (row, first, n) in [(0usize, al, vr), (1, a0, ar)].into_iter().filter(|s| s.2 > 0) {
            let (shift, scale) = (off(fb.ptr(), row * 2 * F * 2), off(fb.ptr(), (row * 2 + 1) * F * 2));
            k!(gpu_adaln_weight(self.g("fn"), scale, folded.ptr(), F as i32, s(st)))?;
            k!(gpu_prologue_rows(DT_BF16, PRO_ADALN, off(xp, first * F * 2), fi, null(), 0, 0, null(), EPS, folded.ptr(), shift, off(h16.ptr(), first * F * 2), fi, n as i32, F as i32, s(st)))?;
        }
        k!(gpu_bf16_to_f32(off(h16.ptr(), a0 * F * 2), fbuf.ptr(), ((ar + vr) * F) as i64, s(st)))?;
        let vout = DevBuf::alloc_async((vr * PATCH + ar * ACH) * 4, st)?;
        if let Some(head) = pdd_head {
            k!(gpu_gemm(off(fbuf.ptr(), ar * F * 4), fi, head, fi, vout.ptr(), PATCH as i64, off(head, PATCH * F * 4), vr as i64, PATCH as i64, fi, 1, DT_F32, DT_F32, 1.0, 0.0,
                s(st)))?;
        } else if self.hyper.is_some() {
            // HyperFlow keeps the video head in float32
            k!(gpu_gemm(off(fbuf.ptr(), ar * F * 4), fi, self.g("vo_w"), fi, vout.ptr(), PATCH as i64, self.g("vo_b"), vr as i64, PATCH as i64, fi, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
        } else {
            k!(gpu_w6_decode(self.g("vo_w"), self.g("vo_r"), wdec.ptr(), PATCH as i64, F as i32, G, s(st)))?;
            k!(gpu_quant_rows(DT_F32, PRO_PLAIN, off(fbuf.ptr(), ar * F * 4), fi, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), vr as i32, F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wdec.ptr(), asc.ptr(), self.g("vo_c"), self.g("vo_b"), vout.ptr(), vr as i64, PATCH as i64, fi, DT_F32, s(st)))?;
        }
        if ar > 0 {
            k!(gpu_gemm(fbuf.ptr(), fi, self.g("ao_w"), fi, off(vout.ptr(), vr * PATCH * 4), ACH as i64, self.g("ao_b"), ar as i64, ACH as i64, fi, 1, DT_F32, DT_F32, 1.0, 0.0,
                s(st)))?;
        }

        self.prof.mark("head", st)?;
        let t0 = std::time::Instant::now();
        let mut raw = vec![0u8; (vr * PATCH + ar * ACH) * 4];
        vout.download(0, &mut raw, st)?;
        let out: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // velocity of each stream is the negative head output; x0 = x - sigma * velocity
        let mut d = vec![0f32; vn + an];
        for t in 0..sh.t {
            for y in 0..th {
                for xx in 0..tw {
                    let r = &out[(((t * th) + y) * tw + xx) * PATCH..][..PATCH];
                    for c in 0..VCH {
                        for p in 0..2 {
                            for q in 0..2 {
                                let i = ((c * sh.t + t) * sh.h + y * 2 + p) * sh.w + xx * 2 + q;
                                d[i] = vx[i] + sigma * r[c * 4 + p * 2 + q];
                            }
                        }
                    }
                }
            }
        }
        let scale = self.shifts.0 / self.shifts.1;
        let aout = &out[vr * PATCH..];
        for ch in 0..2 {
            for t in 0..sh.audio_t {
                for c in 0..ACH {
                    let i = (c * 2 + ch) * sh.audio_t + t;
                    let own = ax[i] * carry;
                    let v = (1.0 - scale) * own + (1.0 + (scale - 1.0) * sigma_a) * -aout[(ch * sh.audio_t + t) * ACH + c];
                    d[vn + i] = ax[i] - sigma * v;
                }
            }
        }
        self.prof.host("pack", t0);
        self.prof.report("step", st)?;
        den.upload(0, &f32_bytes(&d), st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_snap_to_the_grid() {
        assert_eq!((align_frames(1), align_frames(5), align_frames(12), align_frames(124)), (5, 5, 22, 124));
        let sh = Shape::new(512, 512, 5);
        assert_eq!((sh.t, sh.h, sh.w, sh.audio_t), (2, 32, 32, 8));
        assert_eq!(video_latent_t(124), 37);
    }

    #[test]
    fn audio_schedule_meets_the_video_one_at_the_ends() {
        let shifts = (SHIFT_VIDEO as f32, SHIFT_AUDIO as f32);
        assert!((audio_sigma(1.0, shifts) - 1.0).abs() < 1e-6);
        assert!(audio_sigma(0.5, shifts) < 0.5);
    }

    #[test]
    fn hyperflow_schedule_is_its_grid_shifted_like_video() {
        let ms = crate::samplers::ModelSampling::Flow { shift: SHIFT_VIDEO };
        let s = crate::samplers::hyperflow_sigmas(ms, 6).unwrap();
        let want = [1.0, 0.98271, 0.91463, 0.67479, 0.38518, 0.15911, 0.0];
        assert!(s.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-4), "{s:?}");
        assert!(crate::samplers::hyperflow_sigmas(ms, 5).is_err());
        assert!(crate::samplers::hyperflow_sigmas(crate::samplers::ModelSampling::Flux { shift: 1.0 }, 6).is_err());
    }

    #[test]
    fn image_steps_take_the_heads_their_interval_spans() {
        let grid = vec![1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0];
        let p = Pdd { grid: grid.clone(), w: Vec::new(), b: Vec::new(), head: None };
        // the stock schedule at 6 steps lands near each grid point
        let ms = crate::samplers::ModelSampling::Grid(crate::samplers::SigmaGrid::new(&grid.iter().map(|g| *g as f32).collect::<Vec<_>>()).unwrap());
        let s = crate::samplers::sigmas(ms, "euler", "simple", 6, (64, 64), None).unwrap();
        assert!(s.iter().zip(&grid).all(|(a, b)| (*a as f64 - b).abs() < 2e-3), "{s:?}");
        assert_eq!((0..6).map(|i| p.span(s[i], s[i + 1])).collect::<Vec<_>>(), (0..6).map(|i| (i, i + 1)).collect::<Vec<_>>());
        // three steps span two heads each; the last step's end clamps to the grid's end
        assert_eq!(p.span(1.0, 0.8875), (0, 2));
        assert_eq!(p.span(0.2320, 0.0), (5, 6));
    }

    #[test]
    fn time_embedding_is_cos_then_sin() {
        // one hidden unit reading emb[0] = cos(0) = 1 and one reading emb[128] = sin(0) = 0
        let mut w1 = vec![0f32; 2 * FREQ];
        w1[0] = 1.0;
        w1[FREQ + 128] = 1.0;
        let m = TimeMlp { w1, b1: vec![0.0; 2], w2: vec![1.0, 0.0, 0.0, 1.0], b2: vec![0.0; 2] };
        let out = m.at(0.0);
        assert!((out[0] - silu(1.0)).abs() < 1e-6 && out[1].abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn positions_follow_the_packed_layout() {
        let sh = Shape::new(512, 512, 5);
        let p = positions(3, &sh, &Extra::default());
        assert_eq!(p.len(), 3 + 16 + 512);
        assert_eq!(p[2], [2.0, 0.0, 0.0]);
        assert_eq!(p[3][0], 3.0);
        assert_eq!(p[11][0], 3.0);
        assert!(p[3][2] < p[11][2]);
        // frame 1 starts one frame's rescale later
        assert!((p[19 + 256][0] - (3.0 + 5.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn conditions_pack_between_text_and_targets() {
        // PackedLayout(3, 2, 4, 4, 0, keyframes=[frame 4, one latent], refs=[an image of 2 x 6 latents])
        let sh = Shape { t: 2, h: 4, w: 4, audio_t: 0 };
        let extra = Extra { keyframes: vec![(4, 1)], refs: vec![(2, 6)] };
        let p = positions(3, &sh, &extra);
        assert_eq!(extra.rows(&sh), 4 + 3);
        assert_eq!(p.len(), 3 + 7 + 8);
        // the keyframe at the target's origin (after the reference's step) plus 4 frames, on the target's grid
        assert!((p[3][0] - (4.0 + 4.0 * 5.0 / 3.0)).abs() < 1e-12);
        assert_eq!(p[3][1..], p[10][1..]);
        // the reference at the text's end, on its own grid: 2 x 6 latents are 1 x 3 patches over sqrt(12)
        let r = 2.0 / 12f64.sqrt();
        assert_eq!(p[7][0], 3.0);
        assert!((p[7][1] - (1.0 - r) / 2.0 * 32.0).abs() < 1e-12);
        assert!((p[9][2] - p[7][2] - 2.0 * (6.0 / 12f64.sqrt() / 3.0) * 32.0).abs() < 1e-9);
        // the targets start one step after the text
        assert_eq!(p[10][0], 4.0);
    }
}
