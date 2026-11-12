//! Qwen Image 2.1: a single-stream MMDiT (W8A8 int8 ConvRot blocks, bf16 activations like ComfyUI) over one sequence of
//! [text ; reference images ; the image being made]. One modulation, from the timestep, serves every block.
//!
//! The text and the reference images never see the image, and they are modulated as if at t = 0, so what they put into
//! each block's keys and values is the same at every step. A [`Prefix`] computes that once; a step then runs only the
//! image's rows, against [the cached keys and values ; its own]. Without room for the cache, the prefix is run again
//! in every step instead, block by block, which gives the same result.

use crate::cuda::{memcpy_2d_d2d, memcpy_d2d, DevBuf, Stream};
use crate::dit::Dit as KreaDit;
use crate::ffi::{k, null, off, s, DT_BF16, PM, PRO_LN_MOD, PRO_PLAIN, PRO_RMS, PRO_SILU, PRO_SWIGLU};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::st::{bf16_bytes, SafeTensors};
use anyhow::{bail, Result};
use half::bf16;

pub const F: usize = 4096;
const H: usize = 32;
const HD: usize = 128;
const QKV: usize = 3 * F;
const MLP: usize = 12288;
pub const CH: usize = 64;
const TDIM: usize = 256;
const EPS: f32 = 1e-6;
const THETA: f64 = 10_000.0;
const AXES: [usize; 3] = [16, 56, 56];
/// Where the weights sit in the file, with or without ComfyUI's prefix.
const PREFIXES: [&str; 2] = ["model.diffusion_model.", ""];

/// The key prefix of a Qwen Image 2.1 diffusion model in `has`' file.
pub fn key_prefix(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    PREFIXES.into_iter().find(|p| {
        ["txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.attn.norm_q.weight", "img_in.weight", "proj_out.weight"]
            .iter()
            .all(|k| has(&format!("{p}{k}")))
            && (has(&format!("{p}transformer_blocks.0.img_mlp.gate_up.weight")) || has(&format!("{p}transformer_blocks.0.img_mlp.proj.weight")))
    })
}

thread_local! {
    /// Debugging: buffers named by `QWEN21_TRACE` (comma separated, as in `p1,t1`: the prefix and the image of block 1).
    pub static TRACE: std::cell::RefCell<Vec<(String, Vec<u8>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn trace(tag: &str, name: &str, b: &DevBuf, bytes: usize, st: &Stream) -> Result<()> {
    if std::env::var("QWEN21_TRACE").is_ok_and(|v| v.split(',').any(|t| t == tag)) {
        st.sync()?;
        let mut v = vec![0u8; bytes];
        b.download(0, &mut v, st)?;
        st.sync()?;
        TRACE.with(|t| t.borrow_mut().push((format!("{name}_{tag}"), v)));
    }
    Ok(())
}

pub struct Dit {
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    pub attn_variant: i32,
    pub int8_attn: bool,
    pub prof: std::cell::RefCell<crate::prof::Prof>,
}

/// One prompt's text through `txt_in`, ready to be part of a sequence.
pub struct Context {
    /// bf16 [len, F]
    pub ctx: DevBuf,
    pub len: usize,
}

/// A run of the prefix: prompt tokens, or the tokens of one reference image.
#[derive(Clone, Debug, PartialEq)]
pub enum Seg {
    /// `n` rows of the prompt
    Text { from: usize, n: usize },
    /// a `h` x `w` grid
    Image { h: usize, w: usize },
}

impl Seg {
    fn rows(&self) -> usize {
        match self {
            Seg::Text { n, .. } => *n,
            Seg::Image { h, w } => h * w,
        }
    }
}

/// The text with each reference image spliced in at its slot, in the order of the sequence: the prompt's tokens up to
/// slot 0, image 0, the tokens up to slot 1, image 1, and so on. Images without a slot come after the text.
pub fn order(text: usize, slots: &[usize], images: &[(usize, usize)]) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut at = 0;
    for (i, &(h, w)) in images.iter().enumerate() {
        let slot = slots.get(i).copied().unwrap_or(text).clamp(at, text);
        if slot > at {
            segs.push(Seg::Text { from: at, n: slot - at });
        }
        segs.push(Seg::Image { h, w });
        at = slot;
    }
    if text > at {
        segs.push(Seg::Text { from: at, n: text - at });
    }
    segs
}

/// RoPE table [rows][64][(cos, sin)] for the sequence `segs`, then the `target` grid. Text tokens count up on all three
/// axes; a grid sits at one position on the first axis, which then moves on by its longer side, with its rows and
/// columns centred on the target's (a half token off where the parities differ).
pub fn rope_table(segs: &[Seg], target: (usize, usize)) -> Vec<f32> {
    let mut pos: Vec<[f64; 3]> = Vec::new();
    let mut at = 0.0f64;
    let grid = |h: usize, w: usize, at: &mut f64, pos: &mut Vec<[f64; 3]>| {
        let centre = |n: usize, t: usize| -> Vec<f64> {
            (0..n).map(|i| i as f64 - (n - n / 2) as f64 + 0.5 * ((n % 2) as f64 - (t % 2) as f64)).collect()
        };
        let (ys, xs) = (centre(h, target.0), centre(w, target.1));
        for y in &ys {
            for x in &xs {
                pos.push([*at, *y, *x]);
            }
        }
        *at += h.max(w) as f64;
    };
    for seg in segs {
        match seg {
            Seg::Text { n, .. } => {
                pos.extend((0..*n).map(|i| [at + i as f64; 3]));
                at += *n as f64;
            }
            Seg::Image { h, w } => grid(*h, *w, &mut at, &mut pos),
        }
    }
    grid(target.0, target.1, &mut at, &mut pos);
    let mut omegas: Vec<(usize, f64)> = Vec::new();
    for (a, &dim) in AXES.iter().enumerate() {
        for kk in 0..dim / 2 {
            omegas.push((a, 1.0 / THETA.powf(kk as f64 * 2.0 / dim as f64)));
        }
    }
    let mut t = Vec::with_capacity(pos.len() * 128);
    for p in &pos {
        for &(a, om) in &omegas {
            let ang = p[a] * om;
            t.push(ang.cos() as f32);
            t.push(ang.sin() as f32);
        }
    }
    t
}

/// ComfyUI rounds the timestep to bf16 twice on the way in: `(t * 1000).to(bf16) / 1000`.
pub fn flow_time(sigma: f32) -> f32 {
    let r = |v: f32| bf16::from_f32(v).to_f32();
    r(r(sigma * 1000.0) / 1000.0)
}

/// Everything about the prefix of one prompt (with its reference images) and one image size that doesn't change
/// from step to step.
pub struct Prefix {
    pub rows: usize,
    segs: Vec<Seg>,
    /// bf16 [rows, F]: the prefix as it enters the first block
    tokens: DevBuf,
    /// f32 rope table of the prefix, then of the image
    rope: DevBuf,
    /// bf16 [blocks][2][rows][F]: each block's keys, then values
    kv: Option<DevBuf>,
}

impl Prefix {
    /// Bytes of the cache for `rows` prefix rows.
    pub fn cache_bytes(blocks: usize, rows: usize) -> usize {
        blocks * 2 * rows * F * 2
    }

    pub fn cached(&self) -> bool {
        self.kv.is_some()
    }
}

struct Mod {
    /// bf16 [F]: the embedding of the timestep
    temb: DevBuf,
    /// bf16 [4][F]: the first norm's scale, the attention's gate (tanh), the second scale and gate
    vecs: DevBuf,
}

pub fn block_layout(st: &SafeTensors, pre: &str, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("{pre}transformer_blocks.{i}.{x}");
    let mut l = Layout::new();
    l.raw(st, "wqkv", &[n("attn.to_q.weight"), n("attn.to_k.weight"), n("attn.to_v.weight")])?;
    l.raw(st, "sqkv", &[n("attn.to_q.weight_scale"), n("attn.to_k.weight_scale"), n("attn.to_v.weight_scale")])?;
    l.raw(st, "wo", &[n("attn.to_out.0.weight")])?;
    l.raw(st, "so", &[n("attn.to_out.0.weight_scale")])?;
    l.raw(st, "wgu", &[n("img_mlp.gate_up.weight")])?;
    l.raw(st, "sgu", &[n("img_mlp.gate_up.weight_scale")])?;
    l.raw(st, "wd", &[n("img_mlp.out.weight")])?;
    l.raw(st, "sd", &[n("img_mlp.out.weight_scale")])?;
    l.f32(st, "qn", &n("attn.norm_q.weight"), 0.0)?;
    l.f32(st, "kn", &n("attn.norm_k.weight"), 0.0)?;
    Ok(l)
}

pub fn globals_layout(st: &SafeTensors, pre: &str) -> Result<Layout> {
    let mut g = Layout::new();
    for (key, name) in [("img_in", "img_in.weight"), ("t1", "time_text_embed.timestep_embedder.linear_1.weight"),
        ("t2", "time_text_embed.timestep_embedder.linear_2.weight"), ("modulation", "modulation.1.weight"),
        ("norm_out", "norm_out.linear.weight"), ("proj_out", "proj_out.weight"), ("txt_in", "txt_in.in_layer.weight"),
        ("txt_out", "txt_in.out_layer.weight")]
    {
        g.bf16(st, key, &[format!("{pre}{name}")])?;
    }
    // ZeroCenteredRMSNorm stores its scale minus one
    g.f32(st, "text_norm", &format!("{pre}txt_in.text_norm.weight"), 1.0)?;
    g.bytes("zero", vec![0u8; F * 2]);
    Ok(g)
}

pub fn n_blocks(st: &SafeTensors, pre: &str) -> usize {
    (0..).take_while(|i| st.has(&format!("{pre}transformer_blocks.{i}.attn.to_q.weight"))).count()
}

/// Rough activation footprint of one denoising step (bytes) for `l` image rows against `keys` keys.
pub fn step_activation_bytes(l: usize, keys: usize) -> usize {
    l * (F * 2 /*x*/ + QKV * 2 + F * 2 /*attn*/ + 2 * MLP * 2 /*gate|up*/ + MLP /*int8*/ + 16 + CH * 2) + 2 * keys * F * 2 /*k, v*/ + H * l * 4 + (128 << 20)
}

impl Dit {
    pub fn load(st: &SafeTensors, pre: &str, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        if !st.has(&format!("{pre}transformer_blocks.0.img_mlp.gate_up.weight_scale")) {
            bail!("only W8A8 (ConvRot int8) Qwen Image 2.1 checkpoints with a fused gate and up projection");
        }
        let gl = globals_layout(st, pre)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let layouts = (0..n_blocks(st, pre)).map(|i| block_layout(st, pre, i)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, block_budget, slots);
        let blocks = WeightSet::load("qwen image 2.1 blocks", st, layouts, p, threads)?;
        Ok(Dit { globals, gl, blocks, attn_variant: 0, int8_attn: true, prof: std::cell::RefCell::new(crate::prof::Prof::new()) })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Everywhere this model keeps weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Buf(&self.globals, &self.gl)]
    }

    /// `out[rows, n] = a[rows, k] @ w[n, k]^T`, bf16, no bias.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, a: PM, w: PM, out: PM, rows: usize, n: usize, kk: usize, st: &Stream) -> Result<()> {
        k!(gpu_gemm(a, kk as i64, w, kk as i64, out, n as i64, null(), rows as i64, n as i64, kk as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))
    }

    /// Text through `txt_in`: a zero-centred RMSNorm, then two linears around a tanh GELU. It depends on the prompt alone.
    pub fn encode_context(&self, cond: &DevBuf, len: usize, st: &Stream) -> Result<Context> {
        let fi = F as i64;
        let h = DevBuf::alloc_async(len * F * 2, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, cond.ptr(), fi, null(), 0, 0, self.g("text_norm"), EPS, null(), null(), h.ptr(), fi, len as i32, F as i32,
            s(st)))?;
        let a = DevBuf::alloc_async(len * F * 2, st)?;
        self.gemm(h.ptr(), self.g("txt_in"), a.ptr(), len, F, F, st)?;
        k!(gpu_gelu_tanh_bf16(a.ptr(), (len * F) as i64, s(st)))?;
        let out = DevBuf::alloc_async(len * F * 2, st)?;
        self.gemm(a.ptr(), self.g("txt_out"), out.ptr(), len, F, F, st)?;
        Ok(Context { ctx: out, len })
    }

    /// The prefix of `ctx` with reference latents (f32 [64, h, w] at the sampler's scale), for an image of `target` tokens.
    /// `slots` say after how many text tokens each reference goes.
    pub fn prefix(&self, ctx: &Context, slots: &[usize], refs: &[(Vec<f32>, usize, usize)], target: (usize, usize), st: &Stream) -> Result<Prefix> {
        let grids: Vec<(usize, usize)> = refs.iter().map(|(_, h, w)| (*h, *w)).collect();
        let segs = order(ctx.len, slots, &grids);
        let rows: usize = segs.iter().map(Seg::rows).sum();
        let tokens = DevBuf::alloc_async(rows * F * 2, st)?;
        let (mut at, mut image) = (0, 0);
        for seg in &segs {
            match seg {
                Seg::Text { from, n } => memcpy_d2d(off(tokens.ptr(), at * F * 2), off(ctx.ctx.ptr(), from * F * 2), n * F * 2, st)?,
                Seg::Image { h, w } => {
                    let (latent, ..) = &refs[image];
                    image += 1;
                    let d = DevBuf::alloc_async(latent.len() * 4, st)?;
                    d.upload(0, &crate::st::f32_bytes(latent), st)?;
                    let tok = DevBuf::alloc_async(h * w * CH * 2, st)?;
                    k!(gpu_patchify(d.ptr(), tok.ptr(), CH as i32, *h as i32, *w as i32, 1, s(st)))?;
                    self.gemm(tok.ptr(), self.g("img_in"), off(tokens.ptr(), at * F * 2), h * w, F, CH, st)?;
                }
            }
            at += seg.rows();
        }
        let table = rope_table(&segs, target);
        let rope = DevBuf::alloc_async(table.len() * 4, st)?;
        rope.upload(0, &crate::st::f32_bytes(&table), st)?;
        Ok(Prefix { rows, segs, tokens, rope, kv: None })
    }

    /// The modulation of the flow time `t`: the embedding, and the four vectors every block takes from it.
    fn modulation(&self, t: f32, st: &Stream) -> Result<Mod> {
        let fi = F as i64;
        let emb = DevBuf::alloc_async(TDIM * 2, st)?;
        emb.upload(0, &bf16_bytes(&KreaDit::timestep_embedding(t)), st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        self.gemm(emb.ptr(), self.g("t1"), t1.ptr(), 1, F, TDIM, st)?;
        let silu = DevBuf::alloc_async(F * 2, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, t1.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), silu.ptr(), fi, 1, F as i32, s(st)))?;
        let temb = DevBuf::alloc_async(F * 2, st)?;
        self.gemm(silu.ptr(), self.g("t2"), temb.ptr(), 1, F, F, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, temb.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), silu.ptr(), fi, 1, F as i32, s(st)))?;
        let vecs = DevBuf::alloc_async(4 * F * 2, st)?;
        self.gemm(silu.ptr(), self.g("modulation"), vecs.ptr(), 1, 4 * F, F, st)?;
        k!(gpu_tanh_bf16(off(vecs.ptr(), F * 2), F as i64, s(st)))?;
        k!(gpu_tanh_bf16(off(vecs.ptr(), 3 * F * 2), F as i64, s(st)))?;
        Ok(Mod { temb, vecs })
    }

    #[allow(clippy::too_many_arguments)]
    fn flash(&self, q: PM, kp: PM, v: PM, o: PM, lse: PM, batch: usize, sq: usize, sk: usize, qrow: usize, krow: usize, st: &Stream) -> Result<()> {
        let hs = HD as i64;
        k!(gpu_flash_attn(q, kp, v, o, lse, batch as i32, sq as i32, sk as i32, H as i32, H as i32, (sq * qrow) as i64, qrow as i64, hs,
            (sk * krow) as i64, krow as i64, hs, (sk * krow) as i64, krow as i64, hs, (sq * F) as i64, F as i64, hs, 1.0 / (HD as f32).sqrt(),
            self.attn_variant, s(st)))
    }

    /// Attention of `batch` images, each of `lq` rows of `q` (rows `qrow` apart) over its `lk` rows of keys and values
    /// (`krow` apart).
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, q: PM, kp: PM, vp: PM, out: PM, batch: usize, lq: usize, lk: usize, qrow: usize, krow: usize, st: &Stream) -> Result<()> {
        let (hs, b) = (HD as i64, batch as i32);
        if self.int8_attn {
            let ws = DevBuf::alloc_async(unsafe { crate::ffi::gpu_int8_attn_workspace(b, H as i32, H as i32, lq as i32, lk as i32) } as usize, st)?;
            k!(gpu_int8_attn(q, kp, vp, out, ws.ptr(), b, lq as i32, lk as i32, H as i32, H as i32, (lq * qrow) as i64, qrow as i64, hs,
                (lk * krow) as i64, krow as i64, hs, (lk * krow) as i64, krow as i64, hs, (lq * F) as i64, F as i64, hs, 1.0 / (HD as f32).sqrt(), s(st)))
        } else {
            let lse = DevBuf::alloc_async(H * batch * lq * 4, st)?;
            self.flash(q, kp, vp, out, lse.ptr(), batch, lq, lk, qrow, krow, st)
        }
    }

    /// One block over `rows` rows of `x` that `scale`, `gate` etc. modulate alike: `batch` sequences one after another, each
    /// taking `rope`. `kv` is called with the rows of q, k and v (after the norm and RoPE) and returns the attention output
    /// for them.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, tag: &str, base: PM, lay: &Layout, x: &DevBuf, batch: usize, rows: usize, m: &Mod, rope: PM, st: &Stream,
        attend: &mut dyn FnMut(&DevBuf, &DevBuf, &Stream) -> Result<()>) -> Result<()> {
        let wp = |key: &str| off(base, lay.off(key));
        let (fi, mi) = (F as i64, rows as i32);
        let vec = |j: usize| off(m.vecs.ptr(), j * F * 2);
        let zero = self.g("zero");
        let mark = |stage: &'static str| self.prof.borrow_mut().mark(stage, st);
        let a8 = DevBuf::alloc_async(rows * MLP, st)?;
        let asc = DevBuf::alloc_async(rows * 4, st)?;
        let qkv = DevBuf::alloc_async(rows * QKV * 2, st)?;
        mark("start")?;
        k!(gpu_quant_rows(DT_BF16, PRO_LN_MOD, x.ptr(), fi, null(), 0, 0, null(), EPS, vec(0), zero, a8.ptr(), asc.ptr(), mi, F as i32, 1, s(st)))?;
        mark("quant")?;
        k!(gpu_int8_gemm(a8.ptr(), wp("wqkv"), asc.ptr(), wp("sqkv"), null(), qkv.ptr(), rows as i64, QKV as i64, fi, DT_BF16, s(st)))?;
        mark("qkv gemm")?;
        let each = rows / batch;
        for i in 0..batch {
            k!(gpu_qk_norm_rope(DT_BF16, HD as i32, off(qkv.ptr(), i * each * QKV * 2), QKV as i64, 0, F as i32, H as i32, H as i32, wp("qn"), wp("kn"),
                EPS, rope, 1, each as i32, s(st)))?;
        }
        mark("norm rope")?;
        let attn = DevBuf::alloc_async(rows * F * 2, st)?;
        trace(tag, "x_in", x, rows * F * 2, st)?;
        attend(&qkv, &attn, st)?;
        mark("attention")?;
        trace(tag, "attn", &attn, rows * F * 2, st)?;
        drop(qkv);
        k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), mi, F as i32, 1, s(st)))?;
        mark("quant")?;
        drop(attn);
        k!(gpu_int8_gemm_residual(a8.ptr(), wp("wo"), asc.ptr(), wp("so"), null(), vec(1), x.ptr(), x.ptr(), rows as i64, fi, fi, DT_BF16, s(st)))?;
        mark("o gemm")?;
        trace(tag, "x_attn", x, rows * F * 2, st)?;
        k!(gpu_quant_rows(DT_BF16, PRO_LN_MOD, x.ptr(), fi, null(), 0, 0, null(), EPS, vec(2), zero, a8.ptr(), asc.ptr(), mi, F as i32, 1, s(st)))?;
        mark("quant")?;
        let gu = DevBuf::alloc_async(rows * 2 * MLP * 2, st)?;
        k!(gpu_int8_gemm(a8.ptr(), wp("wgu"), asc.ptr(), wp("sgu"), null(), gu.ptr(), rows as i64, (2 * MLP) as i64, fi, DT_BF16, s(st)))?;
        mark("gate up gemm")?;
        k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, gu.ptr(), (2 * MLP) as i64, null(), 0, MLP as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), mi,
            MLP as i32, 1, s(st)))?;
        mark("swiglu quant")?;
        drop(gu);
        k!(gpu_int8_gemm_residual(a8.ptr(), wp("wd"), asc.ptr(), wp("sd"), null(), vec(3), x.ptr(), x.ptr(), rows as i64, fi, MLP as i64, DT_BF16, s(st)))?;
        mark("down gemm")
    }

    /// One block over the prefix `x` (modulated by `m0`). Its keys and values go to `kout` and `vout`, [rows, F] each.
    #[allow(clippy::too_many_arguments)]
    fn prefix_block(&self, tag: &str, base: PM, lay: &Layout, x: &DevBuf, pf: &Prefix, m0: &Mod, kout: PM, vout: PM, st: &Stream) -> Result<()> {
        let p = pf.rows;
        self.block(&format!("p{tag}"), base, lay, x, 1, p, m0, pf.rope.ptr(), st, &mut |qkv, attn, st| {
            let q = qkv.ptr();
            let (kp, vp) = (off(q, F * 2), off(q, 2 * F * 2));
            memcpy_2d_d2d(kout, F * 2, kp, QKV * 2, F * 2, p, st)?;
            memcpy_2d_d2d(vout, F * 2, vp, QKV * 2, F * 2, p, st)?;
            let mut at = 0;
            for seg in &pf.segs {
                let n = seg.rows();
                let (qs, os) = (off(q, at * QKV * 2), off(attn.ptr(), at * F * 2));
                match seg {
                    // the tokens before it, and the ones of its own up to itself
                    Seg::Text { .. } => k!(gpu_attn_prefix(qs, QKV as i64, kp, vp, QKV as i64, H as i32, n as i32, at as i32, os, F as i64,
                        1.0 / (HD as f32).sqrt(), s(st)))?,
                    // everything before it and all of itself
                    Seg::Image { .. } => {
                        let lse = DevBuf::alloc_async(H * n * 4, st)?;
                        self.flash(qs, kp, vp, os, lse.ptr(), 1, n, at + n, QKV, QKV, st)?;
                    }
                }
                at += n;
            }
            Ok(())
        })
    }

    /// Fill the cache of `pf`: run the prefix through every block once. Takes the VRAM of [`Prefix::cache_bytes`].
    pub fn fill_cache(&mut self, pf: &mut Prefix, st: &Stream, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        let (p, nb) = (pf.rows, self.blocks.len());
        let kv = DevBuf::alloc(Prefix::cache_bytes(nb, p))?;
        let x = DevBuf::alloc_async(p * F * 2, st)?;
        memcpy_d2d(x.ptr(), pf.tokens.ptr(), p * F * 2, st)?;
        let m0 = self.modulation(0.0, st)?;
        for b in 0..nb {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let slot = off(kv.ptr(), b * 2 * p * F * 2);
            self.prefix_block(&b.to_string(), base, &lay, &x, pf, &m0, slot, off(slot, p * F * 2), st)?;
            self.blocks.release(b, st)?;
            if let Some(pr) = probe.as_mut() {
                pr(&format!("prefix_block{b}"), &x, p * F * 2);
            }
        }
        pf.kv = Some(kv);
        Ok(())
    }

    /// One model evaluation of `batch` latents (f32 [batch, 64, h, w]) that share the prefix: writes the denoised ones
    /// to `den`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, latent: &DevBuf, batch: usize, h: usize, w: usize, pf: &Prefix, sigma: f32, den: &DevBuf, st: &Stream,
        mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        let (lt, p) = (h * w, pf.rows);
        let (rows, kl) = (batch * lt, p + lt);
        let plane = CH * lt * 4;
        let fi = F as i64;
        let m = self.modulation(flow_time(sigma), st)?;
        if let Some(pr) = probe.as_mut() {
            pr("temb", &m.temb, F * 2);
        }

        let x = DevBuf::alloc_async(rows * F * 2, st)?;
        let tok = DevBuf::alloc_async(rows * CH * 2, st)?;
        // cuBLAS picks its algorithm by the rows, so each image keeps its own GEMMs and its result alone
        for i in 0..batch {
            let ti = off(tok.ptr(), i * lt * CH * 2);
            k!(gpu_patchify(off(latent.ptr(), i * plane), ti, CH as i32, h as i32, w as i32, 1, s(st)))?;
            self.gemm(ti, self.g("img_in"), off(x.ptr(), i * lt * F * 2), lt, F, CH, st)?;
        }

        // each image's keys and values: the prefix's, then its own
        let kcat = DevBuf::alloc_async(batch * kl * F * 2, st)?;
        let vcat = DevBuf::alloc_async(batch * kl * F * 2, st)?;
        let (xp, m0) = if pf.kv.is_none() {
            let xp = DevBuf::alloc_async(p * F * 2, st)?;
            memcpy_d2d(xp.ptr(), pf.tokens.ptr(), p * F * 2, st)?;
            (Some(xp), Some(self.modulation(0.0, st)?))
        } else {
            (None, None)
        };
        let rope_t = off(pf.rope.ptr(), p * 128 * 4);
        for b in 0..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            match (&pf.kv, &xp, &m0) {
                (Some(kv), ..) => {
                    let slot = off(kv.ptr(), b * 2 * p * F * 2);
                    memcpy_d2d(kcat.ptr(), slot, p * F * 2, st)?;
                    memcpy_d2d(vcat.ptr(), off(slot, p * F * 2), p * F * 2, st)?;
                }
                (None, Some(xp), Some(m0)) => {
                    self.prefix_block(&b.to_string(), base, &lay, xp, pf, m0, kcat.ptr(), vcat.ptr(), st)?;
                }
                _ => unreachable!(),
            }
            for i in 1..batch {
                memcpy_d2d(off(kcat.ptr(), i * kl * F * 2), kcat.ptr(), p * F * 2, st)?;
                memcpy_d2d(off(vcat.ptr(), i * kl * F * 2), vcat.ptr(), p * F * 2, st)?;
            }
            let (kc, vc) = (kcat.ptr(), vcat.ptr());
            self.block(&format!("t{b}"), base, &lay, &x, batch, rows, &m, rope_t, st, &mut |qkv, attn, st| {
                for i in 0..batch {
                    let (q, at) = (off(qkv.ptr(), i * lt * QKV * 2), (i * kl + p) * F * 2);
                    memcpy_2d_d2d(off(kc, at), F * 2, off(q, F * 2), QKV * 2, F * 2, lt, st)?;
                    memcpy_2d_d2d(off(vc, at), F * 2, off(q, 2 * F * 2), QKV * 2, F * 2, lt, st)?;
                }
                self.attention(qkv.ptr(), kc, vc, attn.ptr(), batch, lt, kl, QKV, F, st)
            })?;
            self.blocks.release(b, st)?;
            if let Some(pr) = probe.as_mut() {
                if let Some(xp) = &xp {
                    pr(&format!("prefix_block{b}"), xp, p * F * 2);
                }
                pr(&format!("block{b}"), &x, rows * F * 2);
            }
        }

        // the last layer: a scale from the embedding and no shift, then the projection to the latent
        let silu = DevBuf::alloc_async(F * 2, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, m.temb.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), silu.ptr(), fi, 1, F as i32, s(st)))?;
        let scale = DevBuf::alloc_async(F * 2, st)?;
        self.gemm(silu.ptr(), self.g("norm_out"), scale.ptr(), 1, F, F, st)?;
        let normed = DevBuf::alloc_async(rows * F * 2, st)?;
        k!(gpu_prologue_rows_b(DT_BF16, PRO_LN_MOD, x.ptr(), fi, null(), null(), EPS, scale.ptr(), self.g("zero"), normed.ptr(), fi, rows as i32,
            F as i32, s(st)))?;
        for i in 0..batch {
            self.gemm(off(normed.ptr(), i * lt * F * 2), self.g("proj_out"), off(tok.ptr(), i * lt * CH * 2), lt, CH, F, st)?;
        }
        if let Some(pr) = probe.as_mut() {
            pr("proj_out", &tok, rows * CH * 2);
        }
        for i in 0..batch {
            k!(gpu_unpatch_denoised(off(tok.ptr(), i * lt * CH * 2), CH as i64, off(latent.ptr(), i * plane), off(den.ptr(), i * plane), CH as i32,
                h as i32, w as i32, 1, sigma, s(st)))?;
        }
        self.prof.borrow_mut().mark("end", st)?;
        self.prof.borrow_mut().report("qwen21 step", st)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_go_where_the_slots_say() {
        // 5 text tokens, an image after 2 and another after 5
        assert_eq!(order(5, &[2, 5], &[(2, 2), (1, 3)]), [Seg::Text { from: 0, n: 2 }, Seg::Image { h: 2, w: 2 }, Seg::Text { from: 2, n: 3 }, Seg::Image { h: 1, w: 3 }]);
        // an image at the start, and one without a slot
        assert_eq!(order(3, &[0], &[(1, 1), (1, 1)]), [Seg::Image { h: 1, w: 1 }, Seg::Text { from: 0, n: 3 }, Seg::Image { h: 1, w: 1 }]);
        assert_eq!(order(4, &[], &[]), [Seg::Text { from: 0, n: 4 }]);
    }

    #[test]
    fn grids_are_centred_and_numbered_after_the_text() {
        let t = rope_table(&[Seg::Text { from: 0, n: 3 }], (2, 2));
        assert_eq!(t.len(), 7 * 128);
        // the first image token: position 3 on the first axis (pair 0), row -1 on the second (pair 8)
        let row = &t[3 * 128..];
        assert!((row[0] - 3f32.cos()).abs() < 1e-6);
        let omega = 1.0f64 / THETA.powf(0.0);
        assert!((row[16] - (-1.0 * omega).cos() as f32).abs() < 1e-6);
        // text token 2 counts on every axis
        let row = &t[2 * 128..];
        assert!((row[16] - 2f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn the_timestep_rounds_twice() {
        assert_eq!(flow_time(1.0), 1.0);
        assert_eq!(flow_time(0.5), 0.5);
        assert!((flow_time(0.7) - 0.7).abs() < 0.005);
    }
}
