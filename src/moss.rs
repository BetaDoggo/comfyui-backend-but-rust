//! MOSS Image: single-stream DiT (int8 ConvRot blocks) over [image tokens ; text tokens] with one
//! modulation shared by all blocks, 128-channel latents at /16. bf16 activations like ComfyUI.

use crate::cuda::{DevBuf, Stream};
use crate::dit::Dit as KreaDit;
use crate::ffi::{k, null, off, s, DT_BF16, PM, PRO_ADALN, PRO_PLAIN, PRO_RMS, PRO_SITU};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::st::SafeTensors;
use anyhow::Result;
use half::bf16;
use std::collections::HashMap;

const F: usize = 3072;
const H: usize = 24;
const HD: usize = 128;
const QKV: usize = 3 * F;
const MLP: usize = 8192;
const TXT: usize = 4096;
pub const CH: usize = 128;
const TDIM: usize = 256;
const EPS: f32 = 1e-6;
const THETA: f64 = 10000.0;
const TXT_BLOCKS: usize = 2;
/// One axis of the (r, f, h, w) RoPE.
const AXIS: usize = HD / 4;
/// Sigmas whose modulation stays cached.
const MOD_CACHE: usize = 64;

/// True if `has` finds the keys of a MOSS Image diffusion model.
pub fn is_moss(has: impl Fn(&str) -> bool) -> bool {
    has("shared_block_modulation.weight") && has("layers.0.proj.ffn_up_gate.weight")
}

pub struct Dit {
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    pub attn_variant: i32,
    pub int8_attn: bool,
    mods: HashMap<u32, Mods>,
}

/// Everything derived from sigma alone.
struct Mods {
    /// bf16 [6][F]: shift, scale, gate of the attention, then of the MLP
    shared: DevBuf,
    /// bf16 [F] x 2: shift, scale folded with the final norm weight
    head: DevBuf,
    /// bf16 [blocks][2][F]: the two scales folded with each block's norm weights
    scales: DevBuf,
}

/// One prompt, ready for the blocks.
pub struct Context {
    /// bf16 [len, 3072]
    pub ctx: DevBuf,
    pub len: usize,
}

pub fn block_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("layers.{i}.{x}");
    let mut l = Layout::new();
    for (key, name) in [("qkv", "qkv"), ("o", "o"), ("ug", "ffn_up_gate"), ("d", "ffn_down")] {
        l.raw(st, &format!("w_{key}"), &[n(&format!("proj.{name}.weight"))])?;
        l.raw(st, &format!("s_{key}"), &[n(&format!("proj.{name}.weight_scale"))])?;
    }
    l.bf16(st, "n1", &[n("norm1.weight")])?;
    l.bf16(st, "n2", &[n("norm2.weight")])?;
    l.f32(st, "qn", &n("q_norm.weight"), 0.0)?;
    l.f32(st, "kn", &n("k_norm.weight"), 0.0)?;
    Ok(l)
}

pub fn globals_layout(st: &SafeTensors) -> Result<Layout> {
    let mut g = Layout::new();
    for (key, name) in [("vid_w", "video_embedder.weight"), ("vid_b", "video_embedder.bias"), ("t0_w", "time_embedding.0.weight"),
        ("t0_b", "time_embedding.0.bias"), ("t2_w", "time_embedding.2.weight"), ("t2_b", "time_embedding.2.bias"),
        ("sm_w", "shared_block_modulation.weight"), ("sm_b", "shared_block_modulation.bias"), ("hm_w", "video_head_modulation.weight"),
        ("hm_b", "video_head_modulation.bias"), ("fn", "final_norm_video.weight"), ("fl", "final_linear_video.weight"),
        ("tp_w", "text_embedder.proj.weight"), ("tp_b", "text_embedder.proj.bias")]
    {
        g.bf16(st, key, &[name.to_string()])?;
    }
    for i in 0..TXT_BLOCKS {
        let b = |x: &str| format!("text_embedder.blocks.{i}.{x}");
        let key = |x: &str| format!("t{i}_{x}");
        for x in ["qkv", "proj", "ffn_up", "ffn_gate", "ffn_down"] {
            g.bf16(st, &key(&format!("{x}_w")), &[b(&format!("{x}.weight"))])?;
            g.bf16(st, &key(&format!("{x}_b")), &[b(&format!("{x}.bias"))])?;
        }
        for x in ["norm1", "norm2", "q_norm", "k_norm"] {
            g.f32(st, &key(x), &b(&format!("{x}.weight")), 0.0)?;
        }
    }
    Ok(g)
}

pub fn n_blocks(st: &SafeTensors) -> usize {
    (0..).take_while(|i| st.has(&format!("layers.{i}.proj.qkv.weight"))).count()
}

/// Rough activation footprint of one denoising step (bytes) for `l` tokens.
pub fn step_activation_bytes(l: usize) -> usize {
    l * (F * 2 /*x*/ + QKV * 2 + F * 2 /*attn*/ + 2 * MLP * 2 + MLP /*int8*/ + 8) + 2 * l * 2 * F * 2 /*int8 attn*/ + (128 << 20)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect()
}

/// RoPE table [rows][64][(cos, sin)] for an h x w image, then `len` text tokens. The image sits after
/// the text on the f axis, its rows and columns centered.
pub fn rope_table(h: usize, w: usize, len: usize) -> Vec<f32> {
    let omegas: Vec<f64> = (0..AXIS / 2).map(|i| 1.0 / THETA.powf(2.0 * i as f64 / AXIS as f64)).collect();
    let mut t = Vec::with_capacity((h * w + len) * HD);
    let mut row = |pos: [f32; 4]| {
        for p in pos {
            for om in &omegas {
                let a = p as f64 * om;
                t.push(a.cos() as f32);
                t.push(a.sin() as f32);
            }
        }
    };
    for y in 0..h {
        for x in 0..w {
            row([0.0, len as f32, y as f32 - (h as f32 - 1.0) / 2.0, x as f32 - (w as f32 - 1.0) / 2.0]);
        }
    }
    for i in 0..len {
        row([0.0, i as f32, 0.0, 0.0]);
    }
    t
}

impl Dit {
    pub fn load(st: &SafeTensors, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let gl = globals_layout(st)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let layouts = (0..n_blocks(st)).map(|i| block_layout(st, i)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, block_budget, slots);
        let blocks = WeightSet::load("moss blocks", st, layouts, p, threads)?;
        Ok(Dit { globals, gl, blocks, attn_variant: 0, int8_attn: true, mods: HashMap::new() })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Everywhere this model keeps weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Buf(&self.globals, &self.gl)]
    }

    /// Forget what was derived from the weights.
    pub fn clear_mods(&mut self) {
        self.mods.clear();
    }

    /// bf16 linear. With image-sized inputs torch's result is the rounded product plus the bias (exact
    /// here); with short ones it isn't, and the fused bias lands closest (measured on the first step).
    fn gemm(&self, a: PM, w: PM, bias: PM, out: PM, rows: usize, n: usize, kk: usize, ldd: usize, st: &Stream) -> Result<()> {
        if rows < 1024 {
            return k!(krea_gemm(a, kk as i64, w, kk as i64, out, ldd as i64, bias, rows as i64, n as i64, kk as i64, 1, DT_BF16, DT_BF16, 1.0,
                0.0, s(st)));
        }
        k!(krea_gemm(a, kk as i64, w, kk as i64, out, ldd as i64, null(), rows as i64, n as i64, kk as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        if bias.is_null() {
            return Ok(());
        }
        k!(krea_add_gated_bf16(out, ldd as i64, bias, 0, null(), rows as i32, n as i32, s(st)))
    }

    /// The text embedder depends on the prompt alone, so it runs once per prompt.
    pub fn encode_context(&self, cond: &DevBuf, len: usize, st: &Stream) -> Result<Context> {
        let (l, m, fi) = (len, len as i32, F as i64);
        let c16 = DevBuf::alloc_async(l * TXT * 2, st)?;
        k!(krea_f32_to_bf16(cond.ptr(), c16.ptr(), (l * TXT) as i64, s(st)))?;
        let x = DevBuf::alloc_async(l * F * 2, st)?;
        self.gemm(c16.ptr(), self.g("tp_w"), self.g("tp_b"), x.ptr(), l, F, TXT, F, st)?;
        let rope = rope_table(0, 0, l);
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;
        let h = DevBuf::alloc_async(l * F * 2, st)?;
        let qkv = DevBuf::alloc_async(l * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(l * F * 2, st)?;
        let y = DevBuf::alloc_async(l * F * 2, st)?;
        let ug = DevBuf::alloc_async(l * 2 * MLP * 2, st)?;
        let a = DevBuf::alloc_async(l * MLP * 2, st)?;
        let ws = DevBuf::alloc_async(self.attn_workspace(l), st)?;
        let xp = x.ptr();
        for i in 0..TXT_BLOCKS {
            let g = |x: &str| self.g(&format!("t{i}_{x}"));
            k!(krea_prologue_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, g("norm1"), EPS, null(), null(), h.ptr(), fi, m, F as i32, s(st)))?;
            self.gemm(h.ptr(), g("qkv_w"), g("qkv_b"), qkv.ptr(), l, QKV, F, QKV, st)?;
            let q = qkv.ptr();
            k!(krea_qk_norm_rope(DT_BF16, HD as i32, q, QKV as i64, 0, F as i32, H as i32, H as i32, g("q_norm"), g("k_norm"), EPS,
                rope_d.ptr(), 2, m, s(st)))?;
            self.attention(q, off(q, F * 2), off(q, 2 * F * 2), attn.ptr(), ws.ptr(), l, st)?;
            self.gemm(attn.ptr(), g("proj_w"), g("proj_b"), y.ptr(), l, F, F, F, st)?;
            k!(krea_add_gated_bf16(xp, fi, y.ptr(), fi, null(), m, F as i32, s(st)))?;
            k!(krea_prologue_rows(DT_BF16, PRO_RMS, xp, fi, null(), 0, 0, g("norm2"), EPS, null(), null(), h.ptr(), fi, m, F as i32, s(st)))?;
            self.gemm(h.ptr(), g("ffn_up_w"), g("ffn_up_b"), ug.ptr(), l, MLP, F, 2 * MLP, st)?;
            self.gemm(h.ptr(), g("ffn_gate_w"), g("ffn_gate_b"), off(ug.ptr(), MLP * 2), l, MLP, F, 2 * MLP, st)?;
            k!(krea_prologue_rows(DT_BF16, PRO_SITU, ug.ptr(), (2 * MLP) as i64, null(), 0, MLP as i64, null(), 0.0, null(), null(), a.ptr(),
                MLP as i64, m, MLP as i32, s(st)))?;
            self.gemm(a.ptr(), g("ffn_down_w"), g("ffn_down_b"), y.ptr(), l, F, MLP, F, st)?;
            k!(krea_add_gated_bf16(xp, fi, y.ptr(), fi, null(), m, F as i32, s(st)))?;
        }
        Ok(Context { ctx: x, len })
    }

    pub fn rope(&self, h: usize, w: usize, len: usize, st: &Stream) -> Result<DevBuf> {
        let rope = rope_table(h, w, len);
        let d = DevBuf::alloc_async(rope.len() * 4, st)?;
        d.upload(0, &crate::st::f32_bytes(&rope), st)?;
        Ok(d)
    }

    /// vec = time_embedding(sinusoid), then the shared and head modulation.
    fn time_mods(&self, sigma: f32, st: &Stream) -> Result<Mods> {
        let fi = F as i64;
        let emb = DevBuf::alloc_async(TDIM * 2, st)?;
        emb.upload(0, &bf16_bytes(&KreaDit::timestep_embedding(sigma)), st)?;
        let t0 = DevBuf::alloc_async(F * 2, st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        let vec = DevBuf::alloc_async(F * 2, st)?;
        self.gemm(emb.ptr(), self.g("t0_w"), self.g("t0_b"), t0.ptr(), 1, F, TDIM, F, st)?;
        k!(krea_prologue_rows(DT_BF16, crate::ffi::PRO_SILU, t0.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), t1.ptr(), fi, 1, F as i32,
            s(st)))?;
        self.gemm(t1.ptr(), self.g("t2_w"), self.g("t2_b"), vec.ptr(), 1, F, F, F, st)?;
        let shared = DevBuf::alloc(6 * F * 2)?;
        self.gemm(vec.ptr(), self.g("sm_w"), self.g("sm_b"), shared.ptr(), 1, 6 * F, F, 6 * F, st)?;
        let head = DevBuf::alloc(2 * F * 2)?;
        self.gemm(vec.ptr(), self.g("hm_w"), self.g("hm_b"), head.ptr(), 1, 2 * F, F, 2 * F, st)?;
        let scale = off(head.ptr(), F * 2);
        k!(krea_adaln_weight(self.g("fn"), scale, scale, F as i32, s(st)))?;
        let scales = DevBuf::alloc(self.blocks.len() * 2 * F * 2)?;
        Ok(Mods { shared, head, scales })
    }

    /// flash attention only needs a softmax scratch; int8 attention quantizes q, k and v into it
    fn attn_workspace(&self, l: usize) -> usize {
        if self.int8_attn {
            unsafe { crate::ffi::krea_int8_attn_workspace(1, H as i32, H as i32, l as i32, l as i32) as usize }
        } else {
            H * l * 4
        }
    }

    fn attention(&self, q: PM, kp: PM, vp: PM, out: PM, ws: PM, l: usize, st: &Stream) -> Result<()> {
        let (hi, hs, sc) = (H as i32, HD as i64, 1.0 / (HD as f32).sqrt());
        let (bs, rs, ob, or) = ((l * QKV) as i64, QKV as i64, (l * F) as i64, F as i64);
        if self.int8_attn {
            k!(krea_int8_attn(q, kp, vp, out, ws, 1, l as i32, l as i32, hi, hi, bs, rs, hs, bs, rs, hs, bs, rs, hs, ob, or, hs, sc, s(st)))
        } else {
            k!(krea_flash_attn(q, kp, vp, out, ws, 1, l as i32, l as i32, hi, hi, bs, rs, hs, bs, rs, hs, bs, rs, hs, ob, or, hs, sc,
                self.attn_variant, s(st)))
        }
    }

    /// One model evaluation: writes the denoised latent (f32 [128, H, W]) to `den`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, latent: &DevBuf, h: usize, w: usize, c: &Context, rope: &DevBuf, sigma: f32, den: &DevBuf, st: &Stream,
        mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        let li = h * w;
        let l = li + c.len;
        let (fi, m, nil) = (F as i64, l as i32, std::ptr::null_mut());

        let cached = self.mods.remove(&sigma.to_bits());
        let hit = cached.is_some();
        if !hit && self.mods.len() >= MOD_CACHE {
            self.mods.clear();
        }
        let mods = match cached {
            Some(md) => md,
            None => self.time_mods(sigma, st)?,
        };

        // [image tokens ; text tokens]
        let x = DevBuf::alloc_async(l * F * 2, st)?;
        let tok = DevBuf::alloc_async(li * CH * 2, st)?;
        k!(krea_patchify(latent.ptr(), tok.ptr(), CH as i32, h as i32, w as i32, 1, s(st)))?;
        self.gemm(tok.ptr(), self.g("vid_w"), self.g("vid_b"), x.ptr(), li, F, CH, F, st)?;
        crate::cuda::memcpy_d2d(off(x.ptr(), li * F * 2), c.ctx.ptr(), c.len * F * 2, st)?;
        if let Some(p) = probe.as_mut() {
            p("video_embedder_out", &x, li * F * 2);
        }

        let a8 = DevBuf::alloc_async(l * MLP, st)?;
        let asc = DevBuf::alloc_async(l * 4, st)?;
        let qkv = DevBuf::alloc_async(l * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(l * F * 2, st)?;
        let hid = DevBuf::alloc_async(l * 2 * MLP * 2, st)?;
        let ws = DevBuf::alloc_async(self.attn_workspace(l), st)?;
        let (xp, q) = (x.ptr(), qkv.ptr());
        let shared = |j: usize| off(mods.shared.ptr(), j * F * 2);

        for b in 0..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            let scale = |j: usize| off(mods.scales.ptr(), (b * 2 + j) * F * 2);
            if !hit {
                k!(krea_adaln_weight(wp("n1"), shared(1), scale(0), F as i32, s(st)))?;
                k!(krea_adaln_weight(wp("n2"), shared(4), scale(1), F as i32, s(st)))?;
            }
            k!(krea_quant_rows(DT_BF16, PRO_ADALN, xp, fi, null(), 0, 0, null(), EPS, scale(0), shared(0), a8.ptr(), asc.ptr(), m, F as i32, 1,
                s(st)))?;
            k!(krea_int8_gemm(a8.ptr(), wp("w_qkv"), asc.ptr(), wp("s_qkv"), null(), q, l as i64, QKV as i64, fi, DT_BF16, s(st)))?;
            k!(krea_qk_norm_rope(DT_BF16, HD as i32, q, QKV as i64, 0, F as i32, H as i32, H as i32, wp("qn"), wp("kn"), EPS, rope.ptr(), 2, m,
                s(st)))?;
            self.attention(q, off(q, F * 2), off(q, 2 * F * 2), attn.ptr(), ws.ptr(), l, st)?;
            k!(krea_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, F as i32, 1,
                s(st)))?;
            k!(krea_int8_gemm_residual(a8.ptr(), wp("w_o"), asc.ptr(), wp("s_o"), null(), shared(2), xp, xp, l as i64, fi, fi, DT_BF16, s(st)))?;

            k!(krea_quant_rows(DT_BF16, PRO_ADALN, xp, fi, null(), 0, 0, null(), EPS, scale(1), shared(3), a8.ptr(), asc.ptr(), m, F as i32, 1,
                s(st)))?;
            k!(krea_int8_gemm(a8.ptr(), wp("w_ug"), asc.ptr(), wp("s_ug"), null(), hid.ptr(), l as i64, (2 * MLP) as i64, fi, DT_BF16, s(st)))?;
            k!(krea_quant_rows(DT_BF16, PRO_SITU, hid.ptr(), (2 * MLP) as i64, null(), 0, MLP as i64, null(), 0.0, null(), null(), a8.ptr(),
                asc.ptr(), m, MLP as i32, 1, s(st)))?;
            k!(krea_int8_gemm_residual(a8.ptr(), wp("w_d"), asc.ptr(), wp("s_d"), null(), shared(5), xp, xp, l as i64, fi, MLP as i64, DT_BF16,
                s(st)))?;
            self.blocks.release(b, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("block{b}"), &x, l * F * 2);
            }
        }

        let head_scale = off(mods.head.ptr(), F * 2);
        k!(krea_prologue_rows(DT_BF16, PRO_ADALN, xp, fi, null(), 0, 0, null(), EPS, head_scale, mods.head.ptr(), attn.ptr(), fi, li as i32,
            F as i32, s(st)))?;
        self.gemm(attn.ptr(), self.g("fl"), nil, tok.ptr(), li, CH, F, CH, st)?;
        if let Some(p) = probe.as_mut() {
            p("final_tokens", &tok, li * CH * 2);
        }
        k!(krea_unpatch_denoised(tok.ptr(), CH as i64, latent.ptr(), den.ptr(), CH as i32, h as i32, w as i32, 1, sigma, s(st)))?;
        self.mods.insert(sigma.to_bits(), mods);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_puts_the_image_after_the_text() {
        let t = rope_table(2, 2, 3);
        assert_eq!(t.len(), 7 * HD);
        // first image row: f = 3 on the second axis, h = -0.5 on the third
        let row = &t[..HD];
        assert!((row[AXIS] - 3f32.cos()).abs() < 1e-6);
        assert!((row[2 * AXIS] - 0.5f32.cos()).abs() < 1e-6);
        // text token 2: f = 2, centered axes at zero
        let row = &t[6 * HD..];
        assert!((row[AXIS] - 2f32.cos()).abs() < 1e-6);
        assert_eq!(row[2 * AXIS], 1.0);
    }
}
