//! SeedVR2 3B: ByteDance's one-step restoration NaDiT, int8 ConvRot blocks, for pictures and clips. bf16 activations
//! like ComfyUI's `comfy.ldm.seedvr.model`.
//!
//! The image tokens and a fixed text prompt (stored in the checkpoint) run through 32 MM-DiT blocks; the first ten
//! keep separate image and text weights, the rest share them. Attention is windowed: the token grid (frames, rows,
//! columns) is cut into windows of about 400 tokens a frame (fewer, larger ones on small images) and up to a quarter of
//! the clip's frames, every window attends to itself and to the whole text, alternating with half-shifted windows. A
//! text token's output is the mean of its windows'.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, P, PM, PRO_ADALN, PRO_PLAIN, PRO_RMS, PRO_SILU, PRO_SWIGLU};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::st::{bf16_bytes, f32_bytes, SafeTensors};
use anyhow::{bail, Result};
use half::bf16;
use std::collections::HashMap;

/// Latent channels.
pub const CH: usize = 16;
const D: usize = 2560;
const H: usize = 20;
const HD: usize = 128;
const QKV: usize = 3 * D;
const MLP: usize = 6912;
const TXT_IN: usize = 5120;
const EMB: usize = 6 * D;
const SIN: usize = 256;
/// The patch embedding's 2x2x33 inputs, padded for the GEMM.
const TOK_IN: usize = 136;
const TOK_OUT: usize = 4 * CH;
const EPS: f32 = 1e-5;
/// RoPE pairs per axis (of 63 rotated; the head's last two features are left alone).
const ROPE_F: usize = 21;
/// `ModelSamplingDiscreteFlow`'s multiplier: the model's timestep is sigma times this.
const MULTIPLIER: f32 = 1000.0;
/// Per-sigma modulations kept (2 MB each).
const MOD_CACHE: usize = 16;
/// Rows through the MLP at a time: hidden states for 32768 rows are 0.9 GB. ComfyUI's `SEEDVR2_MLP_CHUNK` is 8192, which
/// measured 7% slower here at 2048x2048.
const MLP_CHUNK: usize = 32768;
/// Window geometries kept, one per (latent frames and size, text length).
const GEO_CACHE: usize = 4;

/// The 3B checkpoint, as ComfyUI's `model_detection` recognizes it.
pub fn is_seedvr2(has: impl Fn(&str) -> bool) -> bool {
    has("blocks.31.mlp.all.proj_in_gate.weight") && has("positive_conditioning")
}

/// The 7B checkpoints, which this engine doesn't run.
pub fn is_seedvr2_7b(has: impl Fn(&str) -> bool) -> bool {
    has("blocks.35.mlp.all.proj_in_gate.weight") || has("blocks.35.mlp.vid.proj_out.weight")
}

/// The VAE's keys, as ComfyUI tells it apart.
pub fn is_seedvr2_vae(has: impl Fn(&str) -> bool) -> bool {
    has("decoder.up_blocks.2.upsamplers.0.upscale_conv.weight")
}

/// A branch's AdaLN parameters (bf16 values): shift, scale and gate of the attention, then of the MLP.
#[derive(Clone)]
struct Ada([Vec<f32>; 6]);

pub struct Dit {
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    /// Blocks with separate image and text weights.
    mm: Vec<bool>,
    /// Per block, the image's then the text's AdaLN parameters (the same in shared blocks).
    ada: Vec<[Ada; 2]>,
    /// `vid_out_ada`'s shift and scale.
    out_ada: (Vec<f32>, Vec<f32>),
    freqs: [f32; ROPE_F],
    /// The checkpoint's positive and negative prompts through `txt_in`: bf16 [len, D].
    text: [(DevBuf, usize); 2],
    pub int8_attn: bool,
    /// bf16 modulation vectors by sigma bits, see [`Dit::modulation`].
    mods: HashMap<u32, DevBuf>,
    geos: Vec<((usize, usize, usize, usize), Geometry)>,
}

/// One partition of the token grid into windows, and its way back.
struct WinPlan {
    /// f32 [image tokens][64][(cos, sin)]: positions inside each token's window, the frame axis after the text.
    rope: DevBuf,
    /// i32 rows of the [image | text] buffer, window after window, each window's image tokens then the text.
    gather: DevBuf,
    rows: usize,
    /// (first gathered row, windows, rows per window) of each run of equal windows.
    runs: Vec<(usize, usize, usize)>,
    /// Gathered rows of each token (one for an image token, one per window for a text token): i32 sources and the
    /// [tokens + 1] offsets into them.
    src: DevBuf,
    offs: DevBuf,
}

struct Geometry {
    plans: [WinPlan; 2],
    /// f32 [text tokens][64][(cos, sin)]
    text_rope: DevBuf,
    ws: DevBuf,
}

/// `math.ceil(a / b)` of two positive integers.
fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// `make_720Pwindows_bysize` / `make_shifted_720Pwindows_bysize` for a (t, h, w) token grid and window counts
/// (4, 3, 3): each window's ranges, in the reference's order (width outermost).
pub fn windows(t: usize, h: usize, w: usize, shifted: bool) -> Vec<[(usize, usize); 3]> {
    let (rnt, rnh, rnw) = (4usize, 3usize, 3usize);
    let scale = (3600.0f64 / (h * w) as f64).sqrt();
    let rh = (h as f64 * scale).round_ties_even() as usize;
    let rw = (w as f64 * scale).round_ties_even() as usize;
    let (wh, ww) = (ceil_div(rh, rnh).max(1), ceil_div(rw, rnw).max(1));
    let wt = ceil_div(t.min(30), rnt).max(1);
    let mut out = Vec::new();
    if !shifted {
        let (nt, nh, nw) = (ceil_div(t, wt), ceil_div(h, wh), ceil_div(w, ww));
        for iw in 0..nw {
            for ih in 0..nh {
                for it in 0..nt {
                    let r = [(it * wt, ((it + 1) * wt).min(t)), (ih * wh, ((ih + 1) * wh).min(h)), (iw * ww, ((iw + 1) * ww).min(w))];
                    if r.iter().all(|(a, b)| b > a) {
                        out.push(r);
                    }
                }
            }
        }
        return out;
    }
    let shift = |win: usize, len: usize| if win < len { 0.5f64 } else { 0.0 };
    let (st, sh, sw) = (shift(wt, t), shift(wh, h), shift(ww, w));
    let count = |len: usize, s: f64, win: usize| if s > 0.0 { ((len as f64 - s) / win as f64).ceil() as usize + 1 } else { 1 };
    let (nt, nh, nw) = (count(t, st, wt), count(h, sh, wh), count(w, sw, ww));
    // Python's int() truncates toward zero
    let range = |i: usize, s: f64, win: usize, len: usize| {
        let a = (((i as f64 - s) * win as f64) as i64).max(0) as usize;
        let b = (((i as f64 - s + 1.0) * win as f64) as i64).min(len as i64).max(0) as usize;
        (a, b)
    };
    for iw in 0..nw {
        for ih in 0..nh {
            for it in 0..nt {
                let r = [range(it, st, wt, t), range(ih, sh, wh, h), range(iw, sw, ww, w)];
                if r.iter().all(|(a, b)| b > a) {
                    out.push(r);
                }
            }
        }
    }
    out
}

/// bf16 rounding of an f32.
fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn upload(bytes: &[u8], st: &Stream) -> Result<DevBuf> {
    let d = DevBuf::alloc(bytes.len().max(4))?;
    d.upload(0, bytes, st)?;
    Ok(d)
}

/// Weight keys of a block's branch: `v` and `t` in blocks with separate weights, `a` in shared ones.
fn tag(mm: bool, text: bool) -> &'static str {
    match (mm, text) {
        (false, _) => "a",
        (true, false) => "v",
        (true, true) => "t",
    }
}

pub fn block_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("blocks.{i}.{x}");
    let mm = st.has(&n("attn.proj_qkv.vid.weight"));
    let mut l = Layout::new();
    let branches: &[(&str, &str)] = if mm { &[("v", "vid"), ("t", "txt")] } else { &[("a", "all")] };
    for &(t, b) in branches {
        l.raw(st, &format!("w_qkv.{t}"), &[n(&format!("attn.proj_qkv.{b}.weight"))])?;
        l.raw(st, &format!("s_qkv.{t}"), &[n(&format!("attn.proj_qkv.{b}.weight_scale"))])?;
        l.raw(st, &format!("w_o.{t}"), &[n(&format!("attn.proj_out.{b}.weight"))])?;
        l.raw(st, &format!("s_o.{t}"), &[n(&format!("attn.proj_out.{b}.weight_scale"))])?;
        l.bf16(st, &format!("b_o.{t}"), &[n(&format!("attn.proj_out.{b}.bias"))])?;
        l.raw(st, &format!("w_in.{t}"), &[n(&format!("mlp.{b}.proj_in_gate.weight")), n(&format!("mlp.{b}.proj_in.weight"))])?;
        l.raw(st, &format!("s_in.{t}"), &[n(&format!("mlp.{b}.proj_in_gate.weight_scale")), n(&format!("mlp.{b}.proj_in.weight_scale"))])?;
        l.raw(st, &format!("w_out.{t}"), &[n(&format!("mlp.{b}.proj_out.weight"))])?;
        l.raw(st, &format!("s_out.{t}"), &[n(&format!("mlp.{b}.proj_out.weight_scale"))])?;
        // ComfyUI keeps the norms in bf16
        for (key, name) in [("qn", "norm_q"), ("kn", "norm_k")] {
            l.bytes(&format!("{key}.{t}"), f32_bytes(&st.f32_vec(&n(&format!("attn.{name}.{b}.weight")), true)?));
        }
    }
    Ok(l)
}

/// Everything outside the blocks: small enough to stay in VRAM.
pub fn globals_layout(st: &SafeTensors) -> Result<Layout> {
    let mut g = Layout::new();
    for (key, name) in [("e1", "emb_in.proj_in"), ("e2", "emb_in.proj_hid"), ("e3", "emb_in.proj_out"), ("txt", "txt_in"), ("out", "vid_out.proj")] {
        g.bf16(st, &format!("{key}.w"), &[format!("{name}.weight")])?;
        g.bf16(st, &format!("{key}.b"), &[format!("{name}.bias")])?;
    }
    // the patch embedding's 132 inputs padded to TOK_IN
    let w = st.f32_vec("vid_in.proj.weight", false)?;
    let k = st.info("vid_in.proj.weight")?.shape[1];
    if k != 132 {
        bail!("vid_in.proj takes {k} inputs, expected 132");
    }
    let mut padded = vec![0f32; D * TOK_IN];
    for (o, row) in w.chunks_exact(k).enumerate() {
        padded[o * TOK_IN..o * TOK_IN + k].copy_from_slice(row);
    }
    g.bytes("in.w", bf16_bytes(&padded));
    g.bf16(st, "in.b", &["vid_in.proj.bias".to_string()])?;
    g.bytes("out_norm", f32_bytes(&st.f32_vec("vid_out_norm.weight", true)?));
    g.bf16(st, "pos", &["positive_conditioning".to_string()])?;
    g.bf16(st, "neg", &["negative_conditioning".to_string()])?;
    Ok(g)
}

pub fn n_blocks(st: &SafeTensors) -> usize {
    (0..).take_while(|i| st.has(&format!("blocks.{i}.attn.proj_out.all.weight")) || st.has(&format!("blocks.{i}.attn.proj_out.vid.weight"))).count()
}

/// Activation memory of one evaluation over `l` image tokens (every frame's) and `lt` text tokens in `windows` windows
/// (bytes).
pub fn step_activation_bytes(l: usize, lt: usize, windows: usize) -> usize {
    let rows = l + lt;
    let gathered = l + windows * lt;
    let chunk = rows.min(MLP_CHUNK);
    rows * (D * 2 * 2 /*x, attn*/ + QKV * 2 + D /*int8*/ + 8) + chunk * (2 * MLP * 2 + MLP)
        + gathered * (QKV * 2 + D * 2 + 3 * HD * 2 /*int8 attn*/) + l * (TOK_IN + TOK_OUT) * 2 + (64 << 20)
}

/// Windows `frames` latent frames of `lh` x `lw` are attended in, at most.
pub fn window_count(frames: usize, lh: usize, lw: usize) -> usize {
    let (h, w) = (lh / 2, lw / 2);
    windows(frames, h, w, false).len().max(windows(frames, h, w, true).len())
}

/// `get_timestep_embedding(t, 256, flip_sin_to_cos=False, downscale_freq_shift=0)`, rounded to bf16.
fn timestep_embedding(t: f32) -> Vec<f32> {
    let half = SIN / 2;
    let e = (10000f64.ln() / half as f64) as f32;
    let mut out = vec![0f32; SIN];
    for i in 0..half {
        let a = t * (-(e * i as f32)).exp();
        out[i] = bf(a.sin());
        out[half + i] = bf(a.cos());
    }
    out
}

impl Dit {
    pub fn load(st: &SafeTensors, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        if !is_seedvr2(|k| st.has(k)) {
            bail!("not a SeedVR2 3B model");
        }
        let gl = globals_layout(st)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        let nb = n_blocks(st);
        let layouts = (0..nb).map(|i| block_layout(st, i)).collect::<Result<Vec<_>>>()?;
        let mm: Vec<bool> = (0..nb).map(|i| st.has(&format!("blocks.{i}.attn.proj_qkv.vid.weight"))).collect();
        let mut ada = Vec::with_capacity(nb);
        for (i, &m) in mm.iter().enumerate() {
            let read = |b: &str| -> Result<Ada> {
                let v = |x: &str| st.f32_vec(&format!("blocks.{i}.ada.{b}.{x}"), true);
                Ok(Ada([v("attn_shift")?, v("attn_scale")?, v("attn_gate")?, v("mlp_shift")?, v("mlp_scale")?, v("mlp_gate")?]))
            };
            ada.push(if m { [read("vid")?, read("txt")?] } else { let a = read("all")?; [a.clone(), a] });
        }
        let out_ada = (st.f32_vec("vid_out_ada.out_shift", true)?, st.f32_vec("vid_out_ada.out_scale", true)?);
        let f = st.f32_vec("blocks.0.attn.rope.rope.freqs", false)?;
        if f.len() != ROPE_F {
            bail!("SeedVR2 rope has {} frequencies, expected {ROPE_F}", f.len());
        }
        let mut freqs = [0f32; ROPE_F];
        freqs.copy_from_slice(&f);
        let p = plan(&layouts, block_budget, slots);
        let blocks = WeightSet::load("seedvr2 blocks", st, layouts, p, threads)?;
        let lens = [st.info("positive_conditioning")?.shape[0], st.info("negative_conditioning")?.shape[0]];
        let mut dit = Dit {
            globals,
            gl,
            blocks,
            mm,
            ada,
            out_ada,
            freqs,
            text: [(DevBuf::alloc(4)?, 0), (DevBuf::alloc(4)?, 0)],
            int8_attn: true,
            mods: HashMap::new(),
            geos: Vec::new(),
        };
        for (i, key) in ["pos", "neg"].into_iter().enumerate() {
            let l = lens[i];
            let out = DevBuf::alloc(l * D * 2)?;
            k!(gpu_gemm(dit.g(key), TXT_IN as i64, dit.g("txt.w"), TXT_IN as i64, out.ptr(), D as i64, dit.g("txt.b"), l as i64, D as i64,
                TXT_IN as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(s0)))?;
            dit.text[i] = (out, l);
        }
        s0.sync()?;
        Ok(dit)
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Everywhere this model keeps weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Buf(&self.globals, &self.gl)]
    }

    /// The text tokens of the positive or negative prompt.
    pub fn text_len(&self, negative: bool) -> usize {
        self.text[negative as usize].1
    }

    /// bf16 [blocks][image, text][attn shift, attn scale - 1, attn gate, mlp shift, mlp scale - 1, mlp gate][D], then the
    /// output norm's [shift, scale - 1][D], for timestep `sigma * 1000`: what `AdaSingle` adds to the timestep
    /// embedding's slices, in bf16 like ComfyUI.
    fn modulation(&mut self, sigma: f32, st: &Stream) -> Result<()> {
        if self.mods.contains_key(&sigma.to_bits()) {
            return Ok(());
        }
        if self.mods.len() >= MOD_CACHE {
            self.mods.clear();
        }
        let emb_in = DevBuf::alloc_async(SIN * 2, st)?;
        emb_in.upload(0, &bf16_bytes(&timestep_embedding(sigma * MULTIPLIER)), st)?;
        let (h1, h2, emb) = (DevBuf::alloc_async(D * 2, st)?, DevBuf::alloc_async(D * 2, st)?, DevBuf::alloc_async(EMB * 2, st)?);
        let di = D as i64;
        k!(gpu_gemm(emb_in.ptr(), SIN as i64, self.g("e1.w"), SIN as i64, h1.ptr(), di, self.g("e1.b"), 1, di, SIN as i64, 1, DT_BF16, DT_BF16,
            1.0, 0.0, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, h1.ptr(), di, null(), 0, 0, null(), 0.0, null(), null(), h2.ptr(), di, 1, D as i32, s(st)))?;
        k!(gpu_gemm(h2.ptr(), di, self.g("e2.w"), di, h1.ptr(), di, self.g("e2.b"), 1, di, di, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, h1.ptr(), di, null(), 0, 0, null(), 0.0, null(), null(), h2.ptr(), di, 1, D as i32, s(st)))?;
        k!(gpu_gemm(h2.ptr(), di, self.g("e3.w"), di, emb.ptr(), EMB as i64, self.g("e3.b"), 1, EMB as i64, di, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        let mut bytes = vec![0u8; EMB * 2];
        emb.download(0, &mut bytes, st)?;
        let e: Vec<f32> = bytes.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
        // emb is [D][layer: attn, mlp][shift, scale, gate]
        let a = |layer: usize, j: usize, c: usize| e[c * 6 + layer * 3 + j];
        let nb = self.blocks.len();
        let mut host = Vec::with_capacity((nb * 12 + 2) * D);
        for b in 0..nb {
            for br in 0..2 {
                let p = &self.ada[b][br].0;
                for layer in 0..2 {
                    let (shift, scale, gate) = (&p[layer * 3], &p[layer * 3 + 1], &p[layer * 3 + 2]);
                    host.extend((0..D).map(|c| bf(a(layer, 0, c) + shift[c])));
                    host.extend((0..D).map(|c| bf(bf(a(layer, 1, c) + scale[c]) - 1.0)));
                    host.extend((0..D).map(|c| bf(a(layer, 2, c) + gate[c])));
                }
            }
        }
        // vid_out_ada reads the blocks' cached "attn" slice
        host.extend((0..D).map(|c| bf(a(0, 0, c) + self.out_ada.0[c])));
        host.extend((0..D).map(|c| bf(bf(a(0, 1, c) + self.out_ada.1[c]) - 1.0)));
        let m = DevBuf::alloc(host.len() * 2)?;
        m.upload(0, &bf16_bytes(&host), st)?;
        self.mods.insert(sigma.to_bits(), m);
        Ok(())
    }

    /// The windows of `frames` latent frames of `lh` x `lw` with `lt` text tokens, built once.
    fn geometry(&mut self, frames: usize, lh: usize, lw: usize, lt: usize, st: &Stream) -> Result<usize> {
        let key = (frames, lh, lw, lt);
        if let Some(i) = self.geos.iter().position(|(k, _)| *k == key) {
            return Ok(i);
        }
        let (h, w) = (lh / 2, lw / 2);
        let l = frames * h * w;
        let pair = |a: f32| [a.cos(), a.sin()];
        let mut ws_bytes = 0usize;
        let mut plans = Vec::with_capacity(2);
        for shifted in [false, true] {
            let mut wins: Vec<Vec<usize>> = windows(frames, h, w, shifted)
                .into_iter()
                .map(|[(t0, t1), (h0, h1), (w0, w1)]| {
                    let mut v = Vec::with_capacity((t1 - t0) * (h1 - h0) * (w1 - w0));
                    for t in t0..t1 {
                        for y in h0..h1 {
                            for x in w0..w1 {
                                v.push((t * h + y) * w + x);
                            }
                        }
                    }
                    v
                })
                .collect();
            // each token's place in its window
            let mut rope = vec![0f32; l * 128];
            for [(t0, t1), (h0, h1), (w0, w1)] in windows(frames, h, w, shifted) {
                for t in t0..t1 {
                    for y in h0..h1 {
                        for x in w0..w1 {
                            let r = &mut rope[((t * h + y) * w + x) * 128..][..128];
                            let pos = [(lt + t - t0) as f32, (y - h0) as f32, (x - w0) as f32];
                            for j in 0..63 {
                                r[2 * j..2 * j + 2].copy_from_slice(&pair(pos[j / ROPE_F] * self.freqs[j % ROPE_F]));
                            }
                            r[126..128].copy_from_slice(&[1.0, 0.0]);
                        }
                    }
                }
            }
            // equal windows next to each other, as the reference's stable sort leaves them
            wins.sort_by_key(|v| v.len());
            let mut gather = Vec::new();
            let mut runs: Vec<(usize, usize, usize)> = Vec::new();
            let mut sources: Vec<Vec<i32>> = vec![Vec::new(); l + lt];
            for win in &wins {
                let len = win.len() + lt;
                match runs.last_mut() {
                    Some(r) if r.2 == len => r.1 += 1,
                    _ => runs.push((gather.len(), 1, len)),
                }
                for &tkn in win {
                    sources[tkn].push(gather.len() as i32);
                    gather.push(tkn as i32);
                }
                for j in 0..lt {
                    sources[l + j].push(gather.len() as i32);
                    gather.push((l + j) as i32);
                }
            }
            if sources[..l].iter().any(|s| s.len() != 1) {
                bail!("SeedVR2 windows don't cover the {frames}x{h}x{w} grid once");
            }
            let mut offs = vec![0i32];
            let mut src = Vec::with_capacity(gather.len());
            for s_ in &sources {
                src.extend_from_slice(s_);
                offs.push(src.len() as i32);
            }
            for &(_, n, len) in &runs {
                ws_bytes = ws_bytes.max(unsafe { crate::ffi::gpu_int8_attn_workspace(n as i32, H as i32, H as i32, len as i32, len as i32) } as usize);
            }
            // flash attention's log-sum-exp scratch
            ws_bytes = ws_bytes.max(H * gather.len() * 4);
            plans.push(WinPlan {
                rope: upload(&f32_bytes(&rope), st)?,
                rows: gather.len(),
                gather: upload(&i32_bytes(&gather), st)?,
                runs,
                src: upload(&i32_bytes(&src), st)?,
                offs: upload(&i32_bytes(&offs), st)?,
            });
        }
        let mut text_rope = vec![0f32; lt * 128];
        for (j, r) in text_rope.chunks_exact_mut(128).enumerate() {
            for i in 0..63 {
                r[2 * i..2 * i + 2].copy_from_slice(&pair(j as f32 * self.freqs[i % ROPE_F]));
            }
            r[126..128].copy_from_slice(&[1.0, 0.0]);
        }
        let [a, b]: [WinPlan; 2] = plans.try_into().map_err(|_| anyhow::anyhow!("two window plans"))?;
        let geo = Geometry { plans: [a, b], text_rope: upload(&f32_bytes(&text_rope), st)?, ws: DevBuf::alloc(ws_bytes.max(256))? };
        st.sync()?;
        if self.geos.len() >= GEO_CACHE {
            self.geos.remove(0);
        }
        self.geos.push((key, geo));
        Ok(self.geos.len() - 1)
    }

    /// One model evaluation of a latent (f32 [16, frames, lh, lw]) restoring `cond` (the VAE's latent of the picture
    /// or clip, the same shape): writes the denoised latent to `den`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, latent: &DevBuf, cond: &DevBuf, frames: usize, lh: usize, lw: usize, sigma: f32, negative: bool, den: &DevBuf,
        st: &Stream, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        if lh % 2 != 0 || lw % 2 != 0 || frames == 0 {
            bail!("SeedVR2 latents need an even size, got {frames} frames of {lw}x{lh}");
        }
        self.modulation(sigma, st)?;
        let lt = self.text_len(negative);
        let gi = self.geometry(frames, lh, lw, lt, st)?;
        let l = frames * (lh / 2) * (lw / 2);
        // the frames' rows one after another: patching works on [16, frames * lh, lw] as on a tall picture
        let rows_h = frames * lh;
        let rows = l + lt;
        let di = D as i64;

        let x = DevBuf::alloc_async(rows * D * 2, st)?;
        {
            let tok = DevBuf::alloc_async(l * TOK_IN * 2, st)?;
            k!(gpu_seedvr_patch_in(latent.ptr(), cond.ptr(), tok.ptr(), rows_h as i32, lw as i32, TOK_IN as i32, s(st)))?;
            k!(gpu_gemm(tok.ptr(), TOK_IN as i64, self.g("in.w"), TOK_IN as i64, x.ptr(), di, self.g("in.b"), l as i64, di, TOK_IN as i64, 1,
                DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        }
        crate::cuda::memcpy_d2d(off(x.ptr(), l * D * 2), self.text[negative as usize].0.ptr(), lt * D * 2, st)?;
        if let Some(p) = probe.as_mut() {
            p("vid_in", &x, l * D * 2);
            p("txt_in", &self.text[negative as usize].0, lt * D * 2);
        }

        // int8 rows of D at their row, or a chunk of MLP rows from the start
        let chunk = rows.min(MLP_CHUNK);
        let a8 = DevBuf::alloc_async((rows * D).max(chunk * MLP), st)?;
        let asc = DevBuf::alloc_async(rows * 4, st)?;
        let qkv = DevBuf::alloc_async(rows * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(rows * D * 2, st)?;
        let hid = DevBuf::alloc_async(chunk * 2 * MLP * 2, st)?;
        let max_rows = self.geos[gi].1.plans.iter().map(|p| p.rows).max().unwrap_or(0);
        let gathered = DevBuf::alloc_async(max_rows * QKV * 2, st)?;
        let gout = DevBuf::alloc_async(max_rows * D * 2, st)?;
        let mods = self.mods[&sigma.to_bits()].ptr();
        let nb = self.blocks.len();
        let xp = x.ptr();
        let row = |p: PM, r: usize, width: usize| off(p, r * width * 2);
        let i8row = |r: usize| off(a8.ptr(), r * D);
        let srow = |r: usize| off(asc.ptr(), r * 4);
        let scale = 1.0 / (HD as f32).sqrt();
        let gemm = |a: P, w: P, xs: P, ws: P, bias: P, d: PM, n: usize, cols: usize, kk: usize| -> Result<()> {
            k!(gpu_int8_gemm(a, w, xs, ws, bias, d, n as i64, cols as i64, kk as i64, DT_BF16, s(st)))
        };
        // d += gate * (a w + bias), in place
        let gemm_res = |a: P, w: P, xs: P, ws: P, bias: P, gate: P, d: PM, n: usize, cols: usize, kk: usize| -> Result<()> {
            k!(gpu_int8_gemm_residual(a, w, xs, ws, bias, gate, d, d, n as i64, cols as i64, kk as i64, DT_BF16, s(st)))
        };

        for b in 0..nb {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let mm = self.mm[b];
            let last = b + 1 == nb;
            let wp = |key: &str, text: bool| off(base, lay.off(&format!("{key}.{}", tag(mm, text))));
            let md = |text: bool, j: usize| off(mods, ((b * 2 + text as usize) * 6 + j) * D * 2);
            // the image rows, then the text rows: (first row, rows, is text)
            let parts: Vec<(usize, usize, bool)> = if mm { vec![(0, l, false), (l, lt, true)] } else { vec![(0, rows, false)] };
            let geo = &self.geos[gi].1;
            let wpl = &geo.plans[b % 2];

            // attention input: rmsnorm(x) * (1 + scale) + shift, quantized; the last block's text is only normalized
            if last {
                k!(gpu_quant_rows(DT_BF16, PRO_ADALN, xp, di, null(), 0, 0, null(), EPS, md(false, 1), md(false, 0), i8row(0), srow(0), l as i32,
                    D as i32, 1, s(st)))?;
                k!(gpu_quant_rows(DT_BF16, PRO_RMS, row(xp, l, D), di, null(), 0, 0, null(), EPS, null(), null(), i8row(l), srow(l), lt as i32,
                    D as i32, 1, s(st)))?;
            } else {
                for &(r0, n, text) in &parts {
                    k!(gpu_quant_rows(DT_BF16, PRO_ADALN, row(xp, r0, D), di, null(), 0, 0, null(), EPS, md(text, 1), md(text, 0), i8row(r0),
                        srow(r0), n as i32, D as i32, 1, s(st)))?;
                }
            }
            let gemm_parts: Vec<(usize, usize, bool)> = if mm || last { vec![(0, l, false), (l, lt, mm)] } else { vec![(0, rows, false)] };
            for &(r0, n, text) in &gemm_parts {
                gemm(i8row(r0), wp("w_qkv", text), srow(r0), wp("s_qkv", text), null(), row(qkv.ptr(), r0, QKV), n, QKV, D)?;
            }
            if b == 0 {
                if let Some(p) = probe.as_mut() {
                    p("block0_qkv", &qkv, l * QKV * 2);
                    let t = DevBuf::alloc(lt * QKV * 2)?;
                    crate::cuda::memcpy_d2d(t.ptr(), off(qkv.ptr(), l * QKV * 2), lt * QKV * 2, st)?;
                    p("block0_qkv_txt", &t, lt * QKV * 2);
                }
            }
            // per-head RMSNorm and RoPE, each token at its place in its window
            for &(r0, n, text) in &[(0, l, false), (l, lt, true)] {
                let rope = if text { geo.text_rope.ptr() } else { wpl.rope.ptr() };
                k!(gpu_qk_norm_rope(DT_BF16, HD as i32, row(qkv.ptr(), r0, QKV), QKV as i64, 0, D as i32, H as i32, H as i32,
                    wp("qn", text && mm), wp("kn", text && mm), EPS, rope, 1, n as i32, s(st)))?;
            }
            k!(gpu_gather_rows(qkv.ptr(), (QKV * 2) as i64, wpl.gather.ptr(), wpl.rows as i32, gathered.ptr(), s(st)))?;
            for &(r0, n, len) in &wpl.runs {
                let (q, o) = (row(gathered.ptr(), r0, QKV), row(gout.ptr(), r0, D));
                let (qb, qr, hs, ob, or) = ((len * QKV) as i64, QKV as i64, HD as i64, (len * D) as i64, D as i64);
                let (hi, bi, li) = (H as i32, n as i32, len as i32);
                if self.int8_attn {
                    k!(gpu_int8_attn(q, off(q, D * 2), off(q, 2 * D * 2), o, geo.ws.ptr(), bi, li, li, hi, hi, qb, qr, hs, qb, qr, hs, qb, qr, hs,
                        ob, or, hs, scale, s(st)))?;
                } else {
                    k!(gpu_flash_attn(q, off(q, D * 2), off(q, 2 * D * 2), o, geo.ws.ptr(), bi, li, li, hi, hi, qb, qr, hs, qb, qr, hs,
                        qb, qr, hs, ob, or, hs, scale, 0, s(st)))?;
                }
            }
            k!(gpu_seedvr_unwindow(gout.ptr(), D as i32, wpl.src.ptr(), wpl.offs.ptr(), rows as i32, attn.ptr(), s(st)))?;
            if b == 0 {
                if let Some(p) = probe.as_mut() {
                    p("block0_attn", &attn, l * D * 2);
                    let t = DevBuf::alloc(lt * D * 2)?;
                    crate::cuda::memcpy_d2d(t.ptr(), off(attn.ptr(), l * D * 2), lt * D * 2, st)?;
                    p("block0_attn_txt", &t, lt * D * 2);
                }
            }

            // x += gate * proj_out(attn); after the last block only the image matters
            let out_rows = if last { l } else { rows };
            let out_parts: Vec<(usize, usize, bool)> = if mm { parts.clone() } else { vec![(0, out_rows, false)] };
            for &(r0, n, text) in &out_parts {
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, row(attn.ptr(), r0, D), di, null(), 0, 0, null(), 0.0, null(), null(), i8row(r0), srow(r0),
                    n as i32, D as i32, 1, s(st)))?;
                gemm_res(i8row(r0), wp("w_o", text), srow(r0), wp("s_o", text), wp("b_o", text), md(text, 2), row(xp, r0, D),
                    n, D, D)?;
            }

            // x += gate * proj_out(silu(gate_proj(h)) * proj_in(h)), h = rmsnorm(x) * (1 + scale) + shift
            let mlp_parts: Vec<(usize, usize, bool)> = if mm { parts.clone() } else { vec![(0, out_rows, false)] };
            // in chunks of rows, as ComfyUI's _chunked_mlp: the same arithmetic, a chunk's hidden states at a time
            for &(r0, n, text) in &mlp_parts {
                for c0 in (r0..r0 + n).step_by(chunk) {
                    let m = chunk.min(r0 + n - c0);
                    let (a, sc) = (a8.ptr(), asc.ptr());
                    k!(gpu_quant_rows(DT_BF16, PRO_ADALN, row(xp, c0, D), di, null(), 0, 0, null(), EPS, md(text, 4), md(text, 3), a, sc, m as i32,
                        D as i32, 1, s(st)))?;
                    gemm(a, wp("w_in", text), sc, wp("s_in", text), null(), hid.ptr(), m, 2 * MLP, D)?;
                    k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * MLP) as i64, null(), 0, MLP as i64, null(), 0.0, null(), null(), a, sc,
                        m as i32, MLP as i32, 1, s(st)))?;
                    gemm_res(a, wp("w_out", text), sc, wp("s_out", text), null(), md(text, 5), row(xp, c0, D), m, D, MLP)?;
                }
            }
            self.blocks.release(b, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("block{b}"), &x, l * D * 2);
                let t = DevBuf::alloc(lt * D * 2)?;
                crate::cuda::memcpy_d2d(t.ptr(), off(xp, l * D * 2), lt * D * 2, st)?;
                p(&format!("block{b}_txt"), &t, lt * D * 2);
            }
        }

        // vid_out_norm, then vid_out_ada: rmsnorm(x) * w * scale + shift; the projection; x - v * sigma
        let normed = DevBuf::alloc_async(l * D * 2, st)?;
        let out_mod = off(mods, nb * 12 * D * 2);
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, self.g("out_norm"), EPS, off(out_mod, D * 2), out_mod,
            normed.ptr(), di, l as i32, D as i32, s(st)))?;
        let tok = DevBuf::alloc_async(l * TOK_OUT * 2, st)?;
        k!(gpu_gemm(normed.ptr(), di, self.g("out.w"), di, tok.ptr(), TOK_OUT as i64, self.g("out.b"), l as i64, TOK_OUT as i64, di, 1, DT_BF16,
            DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_unpatch_denoised(tok.ptr(), TOK_OUT as i64, latent.ptr(), den.ptr(), CH as i32, rows_h as i32, lw as i32, 2,
            sigma, s(st)))
    }
}

/// Compare each stage with a dump of `tools/seedvr_ref.py`, every stage from ComfyUI's own inputs.
pub fn check(dit: &std::path::Path, vae: &std::path::Path, reference: &std::path::Path) -> Result<()> {
    use crate::pixal3d::report;
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let r = SafeTensors::open(reference)?;
    let st = Stream::new()?;
    let mut v = crate::seedvr_vae::Vae::load(vae, &st)?;
    // [1, frames, h, w, 3]
    let shape = r.info("pre")?.shape.clone();
    let (pixel_frames, h, w) = (shape[1], shape[2], shape[3]);
    let (lh, lw) = (h / 8, w / 8);
    let pre = r.f32_vec("pre", false)?;
    let psnr = |a: &[f32], b: &[f32]| 10.0 * (a.len() as f64 / a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>()).log10();
    let up = |v: &[f32]| -> Result<DevBuf> {
        let d = DevBuf::alloc(v.len() * 4)?;
        d.upload(0, &f32_bytes(v), &st)?;
        Ok(d)
    };
    let theirs = r.f32_vec("latent", false)?;
    let frames = theirs.len() / (CH * lh * lw);
    // a picture through the image path, a clip through the causal one
    let clip = pixel_frames > 1;
    eprintln!("{pixel_frames} frames of {w}x{h}, {frames} latent frames");
    let decode = |v: &mut crate::seedvr_vae::Vae, lat: &[f32]| -> Result<Vec<f32>> {
        if clip { Ok(v.decode_clip(lat, frames, lh, lw, None, &st)?.0) } else { v.decode(&up(lat)?, lh, lw, &st) }
    };
    for run in 0..2 {
        let t0 = std::time::Instant::now();
        let z = if clip { v.encode_clip(&pre, pixel_frames, h, w, None, &st)?.0 } else { v.encode(&pre, h, w, &st)? };
        st.sync()?;
        if run == 1 {
            eprintln!("VAE encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            report("latent", &z, &theirs);
        }
    }
    let back = decode(&mut v, &theirs)?;
    eprintln!("roundtrip of ComfyUI's latent: {:.2} dB from ComfyUI's, {:.2} dB from the picture", psnr(&back, &r.f32_vec("roundtrip", false)?),
        psnr(&back, &pre));

    let file = SafeTensors::open(dit)?;
    let free = crate::cuda::device_free()?;
    let mut d = Dit::load(&file, free.saturating_sub(3 << 30), 2, 6, &st)?;
    d.int8_attn = std::env::var_os("SEEDVR_FLASH").is_none();
    let (x, cond) = (up(&r.f32_vec("noise", false)?)?, up(&theirs)?);
    let den = DevBuf::alloc(CH * frames * lh * lw * 4)?;
    let mut probes: Vec<(String, Vec<u8>)> = Vec::new();
    for run in 0..3 {
        let t0 = std::time::Instant::now();
        let mut probe = |name: &str, b: &DevBuf, bytes: usize| {
            let mut v = vec![0u8; bytes];
            b.download(0, &mut v, &st).unwrap();
            probes.push((name.to_string(), v));
        };
        let p: Option<&mut dyn FnMut(&str, &DevBuf, usize)> = if run == 0 { Some(&mut probe) } else { None };
        d.predict(&x, &cond, frames, lh, lw, 1.0, false, &den, &st, p)?;
        st.sync()?;
        eprintln!("evaluation {run}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    let qkv_theirs = r.f32_vec("block0_qkv", false)?;
    for (name, bytes) in &probes {
        let ours: Vec<f32> = bytes.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
        if r.has(name) {
            let theirs = if name == "block0_qkv" { qkv_theirs.clone() } else { r.f32_vec(name, false)? };
            report(name, &ours, &theirs);
        }
    }
    let mut bytes = vec![0u8; den.size()];
    den.download(0, &mut bytes, &st)?;
    let ours: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let samples = r.f32_vec("samples", false)?;
    report("denoised", &ours, &samples);
    for run in 0..2 {
        let t0 = std::time::Instant::now();
        let px = decode(&mut v, &ours)?;
        if run == 1 {
            eprintln!("VAE decode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            let theirs = decode(&mut v, &samples)?;
            let decoded = r.f32_vec("decoded", false)?;
            eprintln!("decoded: ours {:.2} dB from ComfyUI's; ComfyUI's latent through our decoder {:.2} dB from ComfyUI's", psnr(&px, &decoded),
                psnr(&theirs, &decoded));
            if clip {
                let n = h * w * 3;
                let decoded = r.f32_vec("decoded", false)?;
                for f in 0..px.len() / n {
                    eprintln!("  frame {f}: {:.2} dB", psnr(&px[f * n..][..n], &decoded[f * n..][..n]));
                }
            }
            write_png(&std::env::temp_dir().join("seedvr_check.png"), &px[..h * w * 3], w, h)?;
        }
    }
    if let Some(t) = std::env::var("SEEDVR_VAE_TILE").ok().and_then(|v| v.parse::<usize>().ok()).filter(|t| *t > 0) {
        if clip {
            let t0 = std::time::Instant::now();
            let tiled = v.decode_clip(&samples, frames, lh, lw, Some(t), &st)?.0;
            eprintln!("tiled clip decode ({t} latents) {:.0} ms: ComfyUI's latent {:.2} dB from ComfyUI's decode", t0.elapsed().as_secs_f64() * 1e3,
                psnr(&tiled, &r.f32_vec("decoded", false)?));
            let z = v.encode_clip(&pre, pixel_frames, h, w, Some((t * 8, t * 2)), &st)?.0;
            report("latent (tiled)", &z, &theirs);
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let tiled = v.decode_tiled(&samples, lh, lw, t, &st)?;
        eprintln!("tiled decode ({t} latents) {:.0} ms: ComfyUI's latent {:.2} dB from ComfyUI's decode", t0.elapsed().as_secs_f64() * 1e3,
            psnr(&tiled, &r.f32_vec("decoded", false)?));
        let t0 = std::time::Instant::now();
        let z = v.encode_tiled(&pre, h, w, t * 8, t * 2, &st)?;
        eprintln!("tiled encode ({} px) {:.0} ms", t * 8, t0.elapsed().as_secs_f64() * 1e3);
        report("latent (tiled)", &z, &r.f32_vec("latent", false)?);
        let back = v.decode(&up(&z)?, lh, lw, &st)?;
        let whole = v.decode(&up(&r.f32_vec("latent", false)?)?, lh, lw, &st)?;
        eprintln!("round trip to the picture: tiled encode {:.2} dB, ComfyUI's whole encode {:.2} dB", psnr(&back, &pre), psnr(&whole, &pre));
    }
    Ok(())
}

fn write_png(path: &std::path::Path, rgb: &[f32], w: usize, h: usize) -> Result<()> {
    let px: Vec<u8> = rgb.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    std::fs::write(path, crate::pipeline::encode_png(w, h, &px)?)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::windows;

    #[test]
    fn windows_tile_the_grid_once() {
        for &(h, w) in &[(64usize, 64usize), (32, 48), (128, 96), (7, 5), (1, 1), (45, 80), (100, 37)] {
            for t in [1usize, 2, 3, 5, 9, 41] {
                for shifted in [false, true] {
                    let mut seen = vec![0u8; t * h * w];
                    for [(t0, t1), (h0, h1), (w0, w1)] in windows(t, h, w, shifted) {
                        for f in t0..t1 {
                            for y in h0..h1 {
                                for x in w0..w1 {
                                    seen[(f * h + y) * w + x] += 1;
                                }
                            }
                        }
                    }
                    assert!(seen.iter().all(|&c| c == 1), "{t}x{h}x{w} shifted {shifted}");
                }
            }
        }
    }

    #[test]
    fn windows_match_the_reference_at_1024() {
        // 64x64 tokens: scale 0.9375, 60 resized, windows of 20
        let w = windows(1, 64, 64, false);
        assert_eq!(w.len(), 16);
        assert_eq!(w[0], [(0, 1), (0, 20), (0, 20)]);
        assert_eq!(w[1], [(0, 1), (20, 40), (0, 20)]);
        let s = windows(1, 64, 64, true);
        assert_eq!(s.len(), 16);
        assert_eq!(s[0], [(0, 1), (0, 10), (0, 10)]);
        assert_eq!(s[1], [(0, 1), (10, 30), (0, 10)]);
    }
}
