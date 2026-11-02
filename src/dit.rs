//! Krea 2 single-stream MMDiT (W8A8 int8 ConvRot blocks), bf16 activations like ComfyUI.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, PRO_GATE, PRO_PLAIN, PRO_RMS, PRO_SWIGLU, PM};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::st::SafeTensors;
use anyhow::{bail, Result};
use half::bf16;

pub const F: usize = 6144; // features
const HQ: usize = 48;
const HK: usize = 12;
const HD: usize = 128;
const QKVG: usize = (HQ + 2 * HK) * HD + F; // q | k | v | gate = 15360
const MLP: usize = 16384;
pub const TXT_D: usize = 2560;
pub const TXT_LAYERS: usize = 12;
const TXT_H: usize = 20;
const TXT_QKVG: usize = 4 * TXT_D;
const TXT_MLP: usize = 6912;
const TDIM: usize = 256;
pub const PATCH: usize = 2;
pub const CH: usize = 16;
const EPS: f32 = 1e-5;
const THETA: f64 = 1000.0;
const MOD: usize = 6 * F;

pub struct Dit {
    globals: DevBuf,
    gl: Layout,
    txt: WeightSet,
    pub blocks: WeightSet,
    pub attn_variant: i32,
    /// Use INT8 (Sage-style) attention for the main blocks instead of FlashAttention-2 bf16.
    pub int8_attn: bool,
}

/// Reference images as DiT tokens, added after the image's.
pub struct References {
    /// bf16 [tokens, F]
    tokens: DevBuf,
    pub count: usize,
    /// ComfyUI's `index_timestep_zero`: these tokens are modulated as if at timestep 0.
    timestep_zero: bool,
}

/// `pad_to_patch_size` of [16, h, w]: an odd side wraps around.
fn pad_circular(latent: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h + h % PATCH, w + w % PATCH);
    let mut out = Vec::with_capacity(CH * ph * pw);
    for c in 0..CH {
        for y in 0..ph {
            for x in 0..pw {
                out.push(latent[(c * h + y % h) * w + x % w]);
            }
        }
    }
    (out, ph, pw)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect()
}

/// Per-block VRAM image size (bytes) without loading, for planning.
pub fn block_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("blocks.{i}.{x}");
    let mut l = Layout::new();
    l.raw(st, "wqkvg", &[n("attn.wq.weight"), n("attn.wk.weight"), n("attn.wv.weight"), n("attn.gate.weight")])?;
    l.raw(st, "sqkvg", &[n("attn.wq.weight_scale"), n("attn.wk.weight_scale"), n("attn.wv.weight_scale"), n("attn.gate.weight_scale")])?;
    l.raw(st, "wo", &[n("attn.wo.weight")])?;
    l.raw(st, "so", &[n("attn.wo.weight_scale")])?;
    l.raw(st, "wgu", &[n("mlp.gate.weight"), n("mlp.up.weight")])?;
    l.raw(st, "sgu", &[n("mlp.gate.weight_scale"), n("mlp.up.weight_scale")])?;
    l.raw(st, "wd", &[n("mlp.down.weight")])?;
    l.raw(st, "sd", &[n("mlp.down.weight_scale")])?;
    l.f32(st, "pre", &n("prenorm.scale"), 1.0)?;
    l.f32(st, "post", &n("postnorm.scale"), 1.0)?;
    l.f32(st, "qn", &n("attn.qknorm.qnorm.scale"), 1.0)?;
    l.f32(st, "kn", &n("attn.qknorm.knorm.scale"), 1.0)?;
    l.f32(st, "mod", &n("mod.lin"), 0.0)?;
    Ok(l)
}

fn txt_block_layout(st: &SafeTensors, p: &str) -> Result<Layout> {
    let n = |x: &str| format!("{p}.{x}");
    let mut l = Layout::new();
    l.f32(st, "pre", &n("prenorm.scale"), 1.0)?;
    l.f32(st, "post", &n("postnorm.scale"), 1.0)?;
    l.f32(st, "qn", &n("attn.qknorm.qnorm.scale"), 1.0)?;
    l.f32(st, "kn", &n("attn.qknorm.knorm.scale"), 1.0)?;
    l.bf16(st, "wqkvg", &[n("attn.wq.weight"), n("attn.wk.weight"), n("attn.wv.weight"), n("attn.gate.weight")])?;
    l.bf16(st, "wo", &[n("attn.wo.weight")])?;
    l.bf16(st, "wgu", &[n("mlp.gate.weight"), n("mlp.up.weight")])?;
    l.bf16(st, "wd", &[n("mlp.down.weight")])?;
    Ok(l)
}

pub fn globals_layout(st: &SafeTensors) -> Result<Layout> {
    let mut g = Layout::new();
    for (k, n) in [("first_w", "first.weight"), ("first_b", "first.bias"), ("t0_w", "tmlp.0.weight"),
        ("t0_b", "tmlp.0.bias"), ("t2_w", "tmlp.2.weight"), ("t2_b", "tmlp.2.bias"), ("tp_w", "tproj.1.weight"),
        ("tp_b", "tproj.1.bias"), ("last_mod", "last.modulation.lin"), ("last_w", "last.linear.weight"),
        ("last_b", "last.linear.bias"), ("proj", "txtfusion.projector.weight")]
    {
        g.bf16(st, k, &[n.to_string()])?;
    }
    g.f32(st, "last_norm", "last.norm.scale", 1.0)?;
    Ok(g)
}

pub fn txt_layouts(st: &SafeTensors) -> Result<Vec<Layout>> {
    let mut v = Vec::new();
    for i in 0..2 {
        v.push(txt_block_layout(st, &format!("txtfusion.layerwise_blocks.{i}"))?);
    }
    for i in 0..2 {
        v.push(txt_block_layout(st, &format!("txtfusion.refiner_blocks.{i}"))?);
    }
    let mut t = Layout::new();
    t.f32(st, "norm", "txtmlp.0.scale", 1.0)?;
    for (k, n) in [("w1", "txtmlp.1.weight"), ("b1", "txtmlp.1.bias"), ("w3", "txtmlp.3.weight"), ("b3", "txtmlp.3.bias")] {
        t.bf16(st, k, &[n.to_string()])?;
    }
    v.push(t);
    Ok(v)
}

pub fn n_blocks(st: &SafeTensors) -> usize {
    (0..).take_while(|i| st.has(&format!("blocks.{i}.mod.lin"))).count()
}

/// Rough activation footprint of one denoising step (bytes) for `l` tokens.
pub fn step_activation_bytes(l: usize) -> usize {
    l * (F * 2 /*x*/ + QKVG * 2 + F * 2 /*attn*/ + 2 * MLP * 2 /*gate|up*/ + MLP /*int8*/ + 16) + HQ * l * 4 + (64 << 20)
}

impl Dit {
    pub fn load(st: &SafeTensors, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let gl = globals_layout(st)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let tl = txt_layouts(st)?;
        let tp = plan(&tl, 0, 2);
        let txt = WeightSet::load("txtfusion", st, tl, tp, threads)?;
        let nb = n_blocks(st);
        let layouts = (0..nb).map(|i| block_layout(st, i)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, block_budget, slots);
        let blocks = WeightSet::load("dit blocks", st, layouts, p, threads)?;
        Ok(Dit { globals, gl, txt, blocks, attn_variant: 0, int8_attn: false })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Everywhere this model keeps weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Set(&mut self.txt), Store::Buf(&self.globals, &self.gl)]
    }

    fn flash(&self, q: PM, kp: PM, v: PM, o: PM, lse: PM, b: usize, sl: usize, h: usize, hk: usize, row: usize,
        orow: usize, st: &Stream) -> Result<()> {
        let (bs, rs, hs) = ((sl * row) as i64, row as i64, HD as i64);
        let (obs, ors) = ((sl * orow) as i64, orow as i64);
        k!(gpu_flash_attn(q, kp, v, o, lse, b as i32, sl as i32, sl as i32, h as i32, hk as i32, bs, rs, hs, bs, rs,
            hs, bs, rs, hs, obs, ors, hs, 1.0 / (HD as f32).sqrt(), self.attn_variant, s(st)))
    }

    /// One text-fusion block on x [nb*sl, 2560] (bf16), attention within groups of `sl` rows.
    fn txt_block(&mut self, unit: usize, x: PM, nb: usize, sl: usize, st: &Stream) -> Result<()> {
        let m = nb * sl;
        let base = self.txt.acquire(unit, st)?;
        let lay = &self.txt.layouts[unit];
        let w = |key: &str| off(base, lay.off(key));
        let h = DevBuf::alloc_async(m * TXT_MLP * 2, st)?;
        let qkvg = DevBuf::alloc_async(m * TXT_QKVG * 2, st)?;
        let attn = DevBuf::alloc_async(m * TXT_D * 2, st)?;
        let y = DevBuf::alloc_async(m * TXT_D * 2, st)?;
        let gu = DevBuf::alloc_async(m * 2 * TXT_MLP * 2, st)?;
        let lse = DevBuf::alloc_async(m * TXT_H * 4, st)?;
        let (mi, d) = (m as i32, TXT_D as i64);
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, x, d, null(), 0, 0, w("pre"), EPS, null(), null(), h.ptr(), d, mi,
            TXT_D as i32, s(st)))?;
        k!(gpu_gemm(h.ptr(), d, w("wqkvg"), d, qkvg.ptr(), TXT_QKVG as i64, null(), m as i64, TXT_QKVG as i64, d, 1,
            DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_qk_norm_rope(DT_BF16, HD as i32, qkvg.ptr(), TXT_QKVG as i64, 0, TXT_D as i32, TXT_H as i32, TXT_H as i32,
            w("qn"), w("kn"), EPS, null(), 0, mi, s(st)))?;
        let q = qkvg.ptr();
        self.flash(q, off(q, TXT_D * 2), off(q, 2 * TXT_D * 2), attn.ptr(), lse.ptr(), nb, sl, TXT_H, TXT_H,
            TXT_QKVG, TXT_D, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_GATE, attn.ptr(), d, off(q, 3 * TXT_D * 2), TXT_QKVG as i64, 0, null(),
            0.0, null(), null(), h.ptr(), d, mi, TXT_D as i32, s(st)))?;
        k!(gpu_gemm(h.ptr(), d, w("wo"), d, y.ptr(), d, null(), m as i64, d, d, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_add_gated_bf16(x, d, y.ptr(), d, null(), mi, TXT_D as i32, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, x, d, null(), 0, 0, w("post"), EPS, null(), null(), h.ptr(), d, mi,
            TXT_D as i32, s(st)))?;
        k!(gpu_gemm(h.ptr(), d, w("wgu"), d, gu.ptr(), (2 * TXT_MLP) as i64, null(), m as i64, (2 * TXT_MLP) as i64,
            d, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SWIGLU, gu.ptr(), (2 * TXT_MLP) as i64, null(), 0, TXT_MLP as i64, null(),
            0.0, null(), null(), h.ptr(), TXT_MLP as i64, mi, TXT_MLP as i32, s(st)))?;
        k!(gpu_gemm(h.ptr(), TXT_MLP as i64, w("wd"), TXT_MLP as i64, y.ptr(), d, null(), m as i64, d,
            TXT_MLP as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_add_gated_bf16(x, d, y.ptr(), d, null(), mi, TXT_D as i32, s(st)))?;
        self.txt.release(unit, st)
    }

    /// Text-fusion adapter + txtmlp: cond bf16 [lc, 12*2560] -> context bf16 [lc, 6144].
    /// (Timestep independent, so it runs once per prompt instead of once per step.)
    pub fn encode_context(&mut self, cond: &DevBuf, lc: usize, st: &Stream) -> Result<DevBuf> {
        let rows = lc * TXT_LAYERS;
        let x = DevBuf::alloc_async(rows * TXT_D * 2, st)?;
        crate::cuda::memcpy_d2d(x.ptr(), cond.ptr(), rows * TXT_D * 2, st)?;
        for u in 0..2 {
            self.txt_block(u, x.ptr(), lc, TXT_LAYERS, st)?;
        }
        let y = DevBuf::alloc_async(lc * TXT_D * 2, st)?;
        k!(gpu_layer_project(x.ptr(), self.g("proj"), y.ptr(), lc as i32, TXT_LAYERS as i32, TXT_D as i32, s(st)))?;
        for u in 2..4 {
            self.txt_block(u, y.ptr(), 1, lc, st)?;
        }
        let base = self.txt.acquire(4, st)?;
        let lay = self.txt.layouts[4].clone();
        let w = |key: &str| off(base, lay.off(key));
        let h = DevBuf::alloc_async(lc * TXT_D * 2, st)?;
        let a = DevBuf::alloc_async(lc * F * 2, st)?;
        let out = DevBuf::alloc_async(lc * F * 2, st)?;
        let d = TXT_D as i64;
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, y.ptr(), d, null(), 0, 0, w("norm"), EPS, null(), null(), h.ptr(), d,
            lc as i32, TXT_D as i32, s(st)))?;
        k!(gpu_gemm(h.ptr(), d, w("w1"), d, a.ptr(), F as i64, w("b1"), lc as i64, F as i64, d, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        k!(gpu_gelu_tanh_bf16(a.ptr(), (lc * F) as i64, s(st)))?;
        k!(gpu_gemm(a.ptr(), F as i64, w("w3"), F as i64, out.ptr(), F as i64, w("b3"), lc as i64, F as i64,
            F as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        self.txt.release(4, st)?;
        Ok(out)
    }

    /// Run reference latents ([16, h, w] f32, model scale) through the input layer.
    pub fn references(&self, latents: &[(Vec<f32>, usize, usize)], timestep_zero: bool, st: &Stream) -> Result<References> {
        let token_grid = |h: usize, w: usize| h.div_ceil(PATCH) * w.div_ceil(PATCH);
        let count = latents.iter().map(|(_, h, w)| token_grid(*h, *w)).sum();
        let tokens = DevBuf::alloc_async(count * F * 2, st)?;
        let pd = (CH * PATCH * PATCH) as i64;
        let mut at = 0;
        for (latent, h, w) in latents {
            let (padded, ph, pw) = pad_circular(latent, *h, *w);
            let n = token_grid(*h, *w);
            let d = DevBuf::alloc_async(padded.len() * 4, st)?;
            d.upload(0, &crate::st::f32_bytes(&padded), st)?;
            let tok = DevBuf::alloc_async(n * CH * PATCH * PATCH * 2, st)?;
            k!(gpu_patchify(d.ptr(), tok.ptr(), CH as i32, ph as i32, pw as i32, PATCH as i32, s(st)))?;
            k!(gpu_gemm(tok.ptr(), pd, self.g("first_w"), pd, off(tokens.ptr(), at * F * 2), F as i64, self.g("first_b"), n as i64, F as i64,
                pd, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
            at += n;
        }
        Ok(References { tokens, count, timestep_zero })
    }

    /// RoPE table [l][64][(cos, sin)] for lt text tokens (position 0) followed by an ht x wt grid, then
    /// by one grid per reference latent (rows, columns), numbered from 1 on the first axis.
    pub fn rope_table(lt: usize, ht: usize, wt: usize, refs: &[(usize, usize)]) -> Vec<f32> {
        let axes = [HD - 12 * (HD / 16), 6 * (HD / 16), 6 * (HD / 16)]; // 32, 48, 48
        let mut omegas: Vec<(usize, f64)> = Vec::new(); // (axis, omega) per pair
        for (a, &dim) in axes.iter().enumerate() {
            for kk in 0..dim / 2 {
                let scale = kk as f64 * 2.0 / dim as f64;
                omegas.push((a, 1.0 / THETA.powf(scale)));
            }
        }
        let l = lt + ht * wt + refs.iter().map(|(h, w)| h * w).sum::<usize>();
        let mut t = Vec::with_capacity(l * 128);
        let mut push = |pos: [f64; 3]| {
            for &(a, om) in &omegas {
                let ang = pos[a] * om;
                t.push(ang.cos() as f32);
                t.push(ang.sin() as f32);
            }
        };
        (0..lt).for_each(|_| push([0.0; 3]));
        for (index, &(h, w)) in std::iter::once(&(ht, wt)).chain(refs).enumerate() {
            for r in 0..h * w {
                push([index as f64, (r / w) as f64, (r % w) as f64]);
            }
        }
        t
    }

    /// Timestep embedding (flux style, time_factor 1000), rounded to bf16.
    pub fn timestep_embedding(sigma: f32) -> Vec<f32> {
        let half = TDIM / 2;
        let t = 1000.0f32 * sigma;
        let neg_log = -(10000f64.ln()) as f32;
        let mut e = vec![0f32; TDIM];
        for i in 0..half {
            let freq = (neg_log * i as f32 / half as f32).exp();
            let a = t * freq;
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        e
    }

    /// (t, tvec) of the timestep MLPs: t = tmlp(emb), tvec = tproj(gelu(t)).
    fn timestep(&self, sigma: f32, st: &Stream) -> Result<(DevBuf, DevBuf)> {
        let fi = F as i64;
        let emb = DevBuf::alloc_async(TDIM * 2, st)?;
        emb.upload(0, &bf16_bytes(&Self::timestep_embedding(sigma)), st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        let t = DevBuf::alloc_async(F * 2, st)?;
        let tg = DevBuf::alloc_async(F * 2, st)?;
        let tvec = DevBuf::alloc_async(MOD * 2, st)?;
        k!(gpu_gemm(emb.ptr(), TDIM as i64, self.g("t0_w"), TDIM as i64, t1.ptr(), fi, self.g("t0_b"), 1, fi,
            TDIM as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_gelu_tanh_bf16(t1.ptr(), fi, s(st)))?;
        k!(gpu_gemm(t1.ptr(), fi, self.g("t2_w"), fi, t.ptr(), fi, self.g("t2_b"), 1, fi, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        crate::cuda::memcpy_d2d(tg.ptr(), t.ptr(), F * 2, st)?;
        k!(gpu_gelu_tanh_bf16(tg.ptr(), fi, s(st)))?;
        k!(gpu_gemm(tg.ptr(), fi, self.g("tp_w"), fi, tvec.ptr(), MOD as i64, self.g("tp_b"), 1, MOD as i64, fi, 1,
            DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        Ok((t, tvec))
    }

    /// One model evaluation + Euler update of `latent` (f32 [16, H, W]) from sigma to sigma_next.
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, latent: &DevBuf, h: usize, w: usize, ctx: &DevBuf, lt: usize, refs: Option<&References>, rope: &DevBuf, sigma: f32,
        sigma_next: f32, st: &Stream, denoised: Option<&DevBuf>, probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>)
        -> Result<()> {
        if h % PATCH != 0 || w % PATCH != 0 {
            bail!("latent size must be even");
        }
        let (ht, wt) = (h / PATCH, w / PATCH);
        let li = ht * wt;
        let nr = refs.map_or(0, |r| r.count);
        let l = lt + li + nr;
        let fi = F as i64;
        let mut probe = probe;

        let (t, tvec) = self.timestep(sigma, st)?;
        let tvec0 = match refs {
            Some(r) if r.timestep_zero => Some(self.timestep(0.0, st)?.1),
            _ => None,
        };

        // combined sequence x = [context ; first(patchify(latent))]
        let x = DevBuf::alloc_async(l * F * 2, st)?;
        crate::cuda::memcpy_d2d(x.ptr(), ctx.ptr(), lt * F * 2, st)?;
        let tok = DevBuf::alloc_async(li * CH * PATCH * PATCH * 2, st)?;
        k!(gpu_patchify(latent.ptr(), tok.ptr(), CH as i32, h as i32, w as i32, PATCH as i32, s(st)))?;
        let pd = (CH * PATCH * PATCH) as i64;
        k!(gpu_gemm(tok.ptr(), pd, self.g("first_w"), pd, off(x.ptr(), lt * F * 2), fi, self.g("first_b"),
            li as i64, fi, pd, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        if let Some(r) = refs {
            crate::cuda::memcpy_d2d(off(x.ptr(), (lt + li) * F * 2), r.tokens.ptr(), nr * F * 2, st)?;
        }

        let a8 = DevBuf::alloc_async(l * MLP, st)?;
        let asc = DevBuf::alloc_async(l * 4, st)?;
        let qkvg = DevBuf::alloc_async(l * QKVG * 2, st)?;
        let attn = DevBuf::alloc_async(l * F * 2, st)?;
        let gu = DevBuf::alloc_async(l * 2 * MLP * 2, st)?;
        let lse = DevBuf::alloc_async(HQ * l * 4, st)?;
        let int8_ws = if self.int8_attn {
            let n = unsafe { crate::ffi::gpu_int8_attn_workspace(1, HQ as i32, HK as i32, l as i32, l as i32) };
            Some(DevBuf::alloc_async(n as usize, st)?)
        } else {
            None
        };
        let modv = DevBuf::alloc_async(MOD * 2, st)?;
        let modv0 = DevBuf::alloc_async(MOD * 2, st)?;
        let (m, xp) = (l as i32, x.ptr());
        // (first row, rows, modulation): one group, or the references apart at timestep 0
        let split = lt + li;
        let groups: Vec<(usize, usize, &DevBuf)> = if tvec0.is_some() { vec![(0, split, &modv), (split, nr, &modv0)] } else { vec![(0, l, &modv)] };
        let mo = |b: &DevBuf, i: usize| off(b.ptr(), i * F * 2);

        for b in 0..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = &self.blocks.layouts[b];
            let wp = |key: &str| off(base, lay.off(key));
            k!(gpu_add_vec_bf16(modv.ptr(), tvec.ptr(), wp("mod"), 1, MOD as i32, s(st)))?;
            if let Some(t0) = &tvec0 {
                k!(gpu_add_vec_bf16(modv0.ptr(), t0.ptr(), wp("mod"), 1, MOD as i32, s(st)))?;
            }
            // attention
            for &(r0, rows, mv) in &groups {
                k!(gpu_quant_rows(DT_BF16, PRO_RMS, off(xp, r0 * F * 2), fi, null(), 0, 0, wp("pre"), EPS, mo(mv, 0), mo(mv, 1), off(a8.ptr(), r0 * F),
                    off(asc.ptr(), r0 * 4), rows as i32, F as i32, 1, s(st)))?;
            }
            k!(gpu_int8_gemm(a8.ptr(), wp("wqkvg"), asc.ptr(), wp("sqkvg"), null(), qkvg.ptr(), l as i64,
                QKVG as i64, fi, DT_BF16, s(st)))?;
            k!(gpu_qk_norm_rope(DT_BF16, HD as i32, qkvg.ptr(), QKVG as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32,
                wp("qn"), wp("kn"), EPS, rope.ptr(), 1, m, s(st)))?;
            let q = qkvg.ptr();
            let (kp, vp) = (off(q, HQ * HD * 2), off(q, (HQ + HK) * HD * 2));
            if let Some(ws) = &int8_ws {
                let (bs, rs, hs) = ((l * QKVG) as i64, QKVG as i64, HD as i64);
                k!(gpu_int8_attn(q, kp, vp, attn.ptr(), ws.ptr(), 1, l as i32, l as i32, HQ as i32, HK as i32, bs, rs,
                    hs, bs, rs, hs, bs, rs, hs, (l * F) as i64, fi, hs, 1.0 / (HD as f32).sqrt(), s(st)))?;
            } else {
                self.flash(q, kp, vp, attn.ptr(), lse.ptr(), 1, l, HQ, HK, QKVG, F, st)?;
            }
            k!(gpu_quant_rows(DT_BF16, PRO_GATE, attn.ptr(), fi, off(q, (HQ + 2 * HK) * HD * 2), QKVG as i64, 0,
                null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, F as i32, 1, s(st)))?;
            for &(r0, rows, mv) in &groups {
                let xr = off(xp, r0 * F * 2);
                k!(gpu_int8_gemm_residual(off(a8.ptr(), r0 * F), wp("wo"), off(asc.ptr(), r0 * 4), wp("so"), null(), mo(mv, 2), xr, xr, rows as i64, fi,
                    fi, DT_BF16, s(st)))?;
            }
            // MLP
            for &(r0, rows, mv) in &groups {
                k!(gpu_quant_rows(DT_BF16, PRO_RMS, off(xp, r0 * F * 2), fi, null(), 0, 0, wp("post"), EPS, mo(mv, 3), mo(mv, 4), off(a8.ptr(), r0 * F),
                    off(asc.ptr(), r0 * 4), rows as i32, F as i32, 1, s(st)))?;
            }
            k!(gpu_int8_gemm(a8.ptr(), wp("wgu"), asc.ptr(), wp("sgu"), null(), gu.ptr(), l as i64, (2 * MLP) as i64,
                fi, DT_BF16, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, gu.ptr(), (2 * MLP) as i64, null(), 0, MLP as i64, null(), 0.0,
                null(), null(), a8.ptr(), asc.ptr(), m, MLP as i32, 1, s(st)))?;
            for &(r0, rows, mv) in &groups {
                let xr = off(xp, r0 * F * 2);
                k!(gpu_int8_gemm_residual(off(a8.ptr(), r0 * MLP), wp("wd"), off(asc.ptr(), r0 * 4), wp("sd"), null(), mo(mv, 5), xr, xr, rows as i64, fi,
                    MLP as i64, DT_BF16, s(st)))?;
            }
            self.blocks.release(b, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("block{b}"), &x, l * F * 2);
            }
        }

        // final layer on image tokens: (1 + scale) * norm(x) + shift -> linear
        let lm = DevBuf::alloc_async(2 * F * 2, st)?;
        k!(gpu_add_vec_bf16(lm.ptr(), t.ptr(), self.g("last_mod"), 0, F as i32, s(st)))?;
        k!(gpu_add_vec_bf16(off(lm.ptr(), F * 2), t.ptr(), off(self.g("last_mod"), F * 2), 0, F as i32, s(st)))?;
        let xi = off(xp, lt * F * 2);
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xi, fi, null(), 0, 0, self.g("last_norm"), EPS, lm.ptr(),
            off(lm.ptr(), F * 2), attn.ptr(), fi, li as i32, F as i32, s(st)))?;
        k!(gpu_gemm(attn.ptr(), fi, self.g("last_w"), fi, tok.ptr(), pd, self.g("last_b"), li as i64, pd, fi, 1,
            DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        if let Some(p) = probe.as_mut() {
            p("dit_out_tokens", &tok, li * pd as usize * 2);
        }
        k!(gpu_unpatch_euler(tok.ptr(), pd, latent.ptr(), CH as i32, h as i32, w as i32, PATCH as i32, sigma,
            sigma_next, denoised.map_or(std::ptr::null_mut(), |d| d.ptr()), s(st)))?;
        let _ = PRO_PLAIN;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_grids_are_numbered_from_one() {
        let t = Dit::rope_table(1, 1, 1, &[(1, 2)]);
        assert_eq!(t.len(), 4 * 128);
        // pair 1 of the first axis: position 0 for the image, 1 for the reference
        let omega = 1.0f64 / 1000f64.powf(2.0 / 32.0);
        assert_eq!(t[128 + 2], 1.0);
        assert!((t[128 * 2 + 2] - omega.cos() as f32).abs() < 1e-6);
    }

    #[test]
    fn odd_sides_wrap_around() {
        let latent: Vec<f32> = (0..CH * 3 * 2).map(|i| i as f32).collect();
        let (out, h, w) = pad_circular(&latent, 3, 2);
        assert_eq!((h, w), (4, 2));
        assert_eq!(out[6..8], latent[0..2]);
    }
}
