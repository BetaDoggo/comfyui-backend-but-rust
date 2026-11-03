//! Trellis 2 flow transformers as Pixal3D ships them (int8 ConvRot blocks, bf16 activations like ComfyUI): the
//! sparse-structure model over the dense 16^3 grid and the shape and texture models over occupied voxels. Each
//! block's cross-attention also adds a projection of Pixal3D's image features at the token (`ProjectAttention`).
//!
//! Cond and uncond run as separate passes: every op is per token, and attention is per sequence, so the result is
//! the same as ComfyUI's CFG batch.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, PM, PRO_GELU_TANH, PRO_LAYERNORM, PRO_LN_AFFINE, PRO_LN_MOD, PRO_PLAIN, PRO_SILU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::SafeTensors;
use anyhow::{bail, Result};
use half::bf16;

const F: usize = 1536;
const H: usize = 12;
const HD: usize = 128;
const QKV: usize = 3 * F;
const KV: usize = 2 * F;
const MLP: usize = 8192;
const MOD: usize = 6 * F;
const FREQ: usize = 256;
/// Image tokens of a DINOv3 encode: CLS and four registers.
pub const CTX_LEN: usize = 5;
pub const CTX: usize = 1024;
const EPS: f32 = 1e-6;
/// `F.rms_norm` without eps uses the epsilon of its float opmath type.
const RMS_EPS: f32 = f32::EPSILON;
/// `F.layer_norm` before the output layer, at torch's default eps.
const OUT_EPS: f32 = 1e-5;

/// The four sub-models of a Pixal3D checkpoint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// Sparse structure: 8 channels on the dense 16^3 grid.
    Structure,
    /// First shape pass on the 32^3 coords.
    Shape512,
    /// Second shape pass on the upsampled coords.
    Shape,
    /// Texture, with the shape latent concatenated to its input.
    Texture,
}

impl Stage {
    fn prefix(self) -> &'static str {
        match self {
            Stage::Structure => "model.structure_model.",
            Stage::Shape512 => "model.img2shape_512.",
            Stage::Shape => "model.img2shape.",
            Stage::Texture => "model.shape2txt.",
        }
    }

    pub fn in_ch(self) -> usize {
        match self {
            Stage::Structure => 8,
            Stage::Texture => 64,
            _ => 32,
        }
    }

    pub fn out_ch(self) -> usize {
        if self == Stage::Structure { 8 } else { 32 }
    }

    /// Width of the projected features: the low-res DINO map, plus the NAF map after the structure stage.
    pub fn proj_ch(self) -> usize {
        if self == Stage::Structure { 1024 } else { 2048 }
    }
}

/// True if `st` holds a Pixal3D (Trellis 2 with projection heads) diffusion model.
pub fn is_pixal3d(has: impl Fn(&str) -> bool) -> bool {
    has("model.img2shape.blocks.0.cross_attn.proj_linear.weight") && has("model.structure_model.t_embedder.mlp.0.weight")
}

fn block_layout(st: &SafeTensors, stage: Stage, i: usize) -> Result<Layout> {
    let p = format!("{}blocks.{i}.", stage.prefix());
    let n = |x: &str| vec![format!("{p}{x}")];
    let mut l = Layout::new();
    let lin = |l: &mut Layout, key: &str, name: &str| -> Result<()> {
        l.raw(st, &format!("w_{key}"), &n(&format!("{name}.weight")))?;
        l.raw(st, &format!("s_{key}"), &n(&format!("{name}.weight_scale")))?;
        l.bf16(st, &format!("b_{key}"), &n(&format!("{name}.bias")))
    };
    lin(&mut l, "qkv", "self_attn.to_qkv")?;
    lin(&mut l, "o", "self_attn.to_out")?;
    lin(&mut l, "q", "cross_attn.cross_attn_block.to_q")?;
    lin(&mut l, "kv", "cross_attn.cross_attn_block.to_kv")?;
    lin(&mut l, "co", "cross_attn.cross_attn_block.to_out")?;
    lin(&mut l, "p", "cross_attn.proj_linear")?;
    lin(&mut l, "1", "mlp.mlp.0")?;
    lin(&mut l, "2", "mlp.mlp.2")?;
    for (key, name) in [("qn", "self_attn.q_rms_norm.gamma"), ("kn", "self_attn.k_rms_norm.gamma"),
        ("cqn", "cross_attn.cross_attn_block.q_rms_norm.gamma"), ("ckn", "cross_attn.cross_attn_block.k_rms_norm.gamma"),
        ("n2w", "norm2.weight"), ("n2b", "norm2.bias")]
    {
        l.f32(st, key, &format!("{p}{name}"), 0.0)?;
    }
    l.bf16(st, "mod", &n("modulation"))?;
    Ok(l)
}

fn globals_layout(st: &SafeTensors, stage: Stage) -> Result<Layout> {
    let p = stage.prefix();
    let mut g = Layout::new();
    for (key, name) in [("t1w", "t_embedder.mlp.0.weight"), ("t1b", "t_embedder.mlp.0.bias"), ("t2w", "t_embedder.mlp.2.weight"),
        ("t2b", "t_embedder.mlp.2.bias"), ("adaw", "adaLN_modulation.1.weight"), ("adab", "adaLN_modulation.1.bias"),
        ("inw", "input_layer.weight"), ("inb", "input_layer.bias"), ("outw", "out_layer.weight"), ("outb", "out_layer.bias")]
    {
        g.bf16(st, key, &[format!("{p}{name}")])?;
    }
    Ok(g)
}

/// Bytes of one stage's weights in VRAM.
pub fn stage_bytes(st: &SafeTensors, stage: Stage) -> Result<usize> {
    let mut total = globals_layout(st, stage)?.size;
    for i in 0..n_blocks(st, stage) {
        total += block_layout(st, stage, i)?.size;
    }
    Ok(total)
}

fn n_blocks(st: &SafeTensors, stage: Stage) -> usize {
    (0..).take_while(|i| st.has(&format!("{}blocks.{i}.mlp.mlp.0.weight", stage.prefix()))).count()
}

/// Activation bytes of one evaluation over `n` tokens.
pub fn step_activation_bytes(n: usize) -> usize {
    n * (F * 2 * 3 /*x, attn, h*/ + QKV * 2 + MLP * 3 /*hid + int8*/ + 4 + HD * 4 + H * 4) + (64 << 20)
}

/// `RotaryPositionEmbedder` for voxel coordinates: [n][64][(cos, sin)], 21 frequencies per axis and one zero pair.
pub fn rope_table(coords: &[[i32; 3]]) -> Vec<f32> {
    let freq_dim = HD / 2 / 3;
    let freqs: Vec<f32> = (0..freq_dim).map(|i| 1.0f32 / 10000f32.powf(i as f32 / freq_dim as f32)).collect();
    let mut t = Vec::with_capacity(coords.len() * HD);
    for c in coords {
        for axis in 0..3 {
            for f in &freqs {
                let a = c[axis] as f32 * f;
                t.extend_from_slice(&[a.cos(), a.sin()]);
            }
        }
        for _ in 3 * freq_dim..HD / 2 {
            t.extend_from_slice(&[1.0, 0.0]);
        }
    }
    t
}

/// The dense grid's coordinates in token order (x major).
pub fn grid_coords(r: usize) -> Vec<[i32; 3]> {
    (0..r * r * r).map(|i| [(i / (r * r)) as i32, ((i / r) % r) as i32, (i % r) as i32]).collect()
}

/// `TimestepEmbedder.timestep_embedding` of the bf16-rounded timestep: cos, then sin, rounded to bf16.
fn timestep_embedding(sigma: f32) -> Vec<u8> {
    let t = bf16::from_f32(sigma * 1000.0).to_f32();
    let half = FREQ / 2;
    let scale = -(10000f64.ln() / half as f64) as f32;
    let mut e = vec![0f32; FREQ];
    for i in 0..half {
        let a = t * (i as f32 * scale).exp();
        e[i] = a.cos();
        e[half + i] = a.sin();
    }
    e.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect()
}

pub struct Flow {
    pub stage: Stage,
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    pub attn_variant: i32,
}

/// What one stage's conditioning contributes to every evaluation.
pub struct StageCond {
    /// bf16 [blocks][CTX_LEN][k | v]: cross-attention keys and values of the image tokens, then of zeros (uncond)
    kv: DevBuf,
    kv_uncond: DevBuf,
    /// int8 rows of the projected features and their scales; uncond's are zero, which leaves the bias
    proj8: DevBuf,
    proj_scales: DevBuf,
    pub n: usize,
}

impl Flow {
    pub fn load(st: &SafeTensors, stage: Stage, budget: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let gl = globals_layout(st, stage)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let layouts = (0..n_blocks(st, stage)).map(|i| block_layout(st, stage, i)).collect::<Result<Vec<_>>>()?;
        if layouts.is_empty() {
            bail!("the checkpoint has no {stage:?} blocks");
        }
        let p = plan(&layouts, budget, 2);
        let blocks = WeightSet::load(&format!("pixal3d {stage:?}"), st, layouts, p, threads)?;
        Ok(Flow { stage, globals, gl, blocks, attn_variant: 0 })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Per-block cross-attention keys and values of `ctx` (f32 [CTX_LEN, CTX]), and the projected features
    /// (bf16 [n, proj_ch]) quantized once.
    pub fn stage_cond(&mut self, ctx: &[f32], proj: &DevBuf, n: usize, st: &Stream) -> Result<StageCond> {
        if ctx.len() != CTX_LEN * CTX {
            bail!("expected {CTX_LEN} image tokens");
        }
        let pc = self.stage.proj_ch();
        let proj8 = DevBuf::alloc_async(n * pc, st)?;
        let proj_scales = DevBuf::alloc_async(n * 4, st)?;
        k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, proj.ptr(), pc as i64, null(), 0, 0, null(), 0.0, null(), null(), proj8.ptr(),
            proj_scales.ptr(), n as i32, pc as i32, 1, s(st)))?;
        let kv = self.context_kv(ctx, st)?;
        let kv_uncond = self.context_kv(&vec![0.0; ctx.len()], st)?;
        Ok(StageCond { kv, kv_uncond, proj8, proj_scales, n })
    }

    fn context_kv(&mut self, ctx: &[f32], st: &Stream) -> Result<DevBuf> {
        let host: Vec<u8> = ctx.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect();
        let c = DevBuf::alloc_async(host.len(), st)?;
        c.upload(0, &host, st)?;
        let a8 = DevBuf::alloc_async(CTX_LEN * CTX, st)?;
        let asc = DevBuf::alloc_async(CTX_LEN * 4, st)?;
        k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, c.ptr(), CTX as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(),
            CTX_LEN as i32, CTX as i32, 1, s(st)))?;
        let nb = self.blocks.len();
        let kv = DevBuf::alloc_async(nb * CTX_LEN * KV * 2, st)?;
        for b in 0..nb {
            let base = self.blocks.acquire(b, st)?;
            let lay = &self.blocks.layouts[b];
            let out = off(kv.ptr(), b * CTX_LEN * KV * 2);
            k!(gpu_int8_gemm(a8.ptr(), off(base, lay.off("w_kv")), asc.ptr(), off(base, lay.off("s_kv")), off(base, lay.off("b_kv")),
                out, CTX_LEN as i64, KV as i64, CTX as i64, DT_BF16, s(st)))?;
            k!(gpu_qk_rms_heads(out, KV as i64, 0, 0, 0, H as i32, null(), off(base, lay.off("ckn")), RMS_EPS, null(), CTX_LEN as i32,
                s(st)))?;
            self.blocks.release(b, st)?;
        }
        Ok(kv)
    }

    #[allow(clippy::too_many_arguments)]
    fn flash(&self, q: PM, q_row: usize, kp: PM, vp: PM, kv_row: usize, out: PM, lse: PM, lq: usize, lk: usize, st: &Stream) -> Result<()> {
        let (hs, sc) = (HD as i64, 1.0 / (HD as f32).sqrt());
        k!(gpu_flash_attn(q, kp, vp, out, lse, 1, lq as i32, lk as i32, H as i32, H as i32, (lq * q_row) as i64, q_row as i64, hs,
            (lk * kv_row) as i64, kv_row as i64, hs, (lk * kv_row) as i64, kv_row as i64, hs, (lq * F) as i64, F as i64, hs, sc,
            self.attn_variant, s(st)))
    }

    /// One evaluation: `x` bf16 [n, in_ch] tokens, `rope` the table of their coordinates; writes the velocity
    /// (bf16 [n, out_ch]) to `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, x_in: &DevBuf, rope: &DevBuf, sigma: f32, c: &StageCond, uncond: bool, out: &DevBuf, st: &Stream,
        mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        let n = c.n;
        let (fi, m) = (F as i64, n as i32);
        let in_ch = self.stage.in_ch();

        // t_emb = mlp(sinusoid); shared modulation = adaLN(silu(t_emb))
        let emb = DevBuf::alloc_async(FREQ * 2, st)?;
        emb.upload(0, &timestep_embedding(sigma), st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        let t2 = DevBuf::alloc_async(F * 2, st)?;
        let t6 = DevBuf::alloc_async(MOD * 2, st)?;
        k!(gpu_gemm(emb.ptr(), FREQ as i64, self.g("t1w"), FREQ as i64, t1.ptr(), fi, self.g("t1b"), 1, fi, FREQ as i64, 1, DT_BF16,
            DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, t1.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), t2.ptr(), fi, 1, F as i32,
            s(st)))?;
        k!(gpu_gemm(t2.ptr(), fi, self.g("t2w"), fi, t1.ptr(), fi, self.g("t2b"), 1, fi, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, t1.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), t2.ptr(), fi, 1, F as i32,
            s(st)))?;
        k!(gpu_gemm(t2.ptr(), fi, self.g("adaw"), fi, t6.ptr(), MOD as i64, self.g("adab"), 1, MOD as i64, fi, 1, DT_BF16, DT_BF16,
            1.0, 0.0, s(st)))?;

        let x = DevBuf::alloc_async(n * F * 2, st)?;
        k!(gpu_gemm(x_in.ptr(), in_ch as i64, self.g("inw"), in_ch as i64, x.ptr(), fi, self.g("inb"), n as i64, fi, in_ch as i64, 1,
            DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        if let Some(p) = probe.as_mut() {
            p("input", &x, n * F * 2);
        }

        let a8 = DevBuf::alloc_async(n * MLP, st)?;
        let asc = DevBuf::alloc_async(n * 4, st)?;
        let qkv = DevBuf::alloc_async(n * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(n * F * 2, st)?;
        let h = DevBuf::alloc_async(n * F * 2, st)?;
        let hid = DevBuf::alloc_async(n * MLP * 2, st)?;
        let lse = DevBuf::alloc_async(H * n * 4, st)?;
        let modv = DevBuf::alloc_async(MOD * 2, st)?;
        let xp = x.ptr();
        let kv_all = if uncond { &c.kv_uncond } else { &c.kv };
        for b in 0..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            k!(gpu_add_vec_bf16(modv.ptr(), wp("mod"), t6.ptr(), 0, MOD as i32, s(st)))?;
            let mo = |j: usize| off(modv.ptr(), j * F * 2);
            let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (mo(0), mo(1), mo(2), mo(3), mo(4), mo(5));

            // self attention
            k!(gpu_quant_rows_b(DT_BF16, PRO_LN_MOD, xp, fi, null(), null(), EPS, scale_msa, shift_msa, a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_qkv"), asc.ptr(), wp("s_qkv"), wp("b_qkv"), qkv.ptr(), n as i64, QKV as i64, fi, DT_BF16,
                s(st)))?;
            k!(gpu_qk_rms_heads(qkv.ptr(), QKV as i64, 0, F as i32, H as i32, H as i32, wp("qn"), wp("kn"), RMS_EPS, rope.ptr(), m,
                s(st)))?;
            let q = qkv.ptr();
            self.flash(q, QKV, off(q, F * 2), off(q, KV * 2), QKV, attn.ptr(), lse.ptr(), n, n, st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_o"), asc.ptr(), wp("s_o"), wp("b_o"), gate_msa, xp, xp, n as i64, fi, fi,
                DT_BF16, s(st)))?;

            // cross attention on the image tokens, plus the projected features
            k!(gpu_quant_rows_b(DT_BF16, PRO_LN_AFFINE, xp, fi, wp("n2w"), wp("n2b"), EPS, null(), null(), a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_q"), asc.ptr(), wp("s_q"), wp("b_q"), h.ptr(), n as i64, fi, fi, DT_BF16, s(st)))?;
            k!(gpu_qk_rms_heads(h.ptr(), fi, 0, 0, H as i32, 0, wp("cqn"), null(), RMS_EPS, null(), m, s(st)))?;
            let kv = off(kv_all.ptr(), b * CTX_LEN * KV * 2);
            self.flash(h.ptr(), F, kv, off(kv, F * 2), KV, attn.ptr(), lse.ptr(), n, CTX_LEN, st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_co"), asc.ptr(), wp("s_co"), wp("b_co"), h.ptr(), n as i64, fi, fi, DT_BF16, s(st)))?;
            if uncond {
                k!(gpu_add_gated_bf16(h.ptr(), fi, wp("b_p"), 0, null(), m, F as i32, s(st)))?;
            } else {
                k!(gpu_int8_gemm(c.proj8.ptr(), wp("w_p"), c.proj_scales.ptr(), wp("s_p"), wp("b_p"), attn.ptr(), n as i64, fi,
                    self.stage.proj_ch() as i64, DT_BF16, s(st)))?;
                k!(gpu_add_gated_bf16(h.ptr(), fi, attn.ptr(), fi, null(), m, F as i32, s(st)))?;
            }
            k!(gpu_add_gated_bf16(xp, fi, h.ptr(), fi, null(), m, F as i32, s(st)))?;

            // MLP
            k!(gpu_quant_rows_b(DT_BF16, PRO_LN_MOD, xp, fi, null(), null(), EPS, scale_mlp, shift_mlp, a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_1"), asc.ptr(), wp("s_1"), wp("b_1"), hid.ptr(), n as i64, MLP as i64, fi, DT_BF16, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_GELU_TANH, hid.ptr(), MLP as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(),
                asc.ptr(), m, MLP as i32, 1, s(st)))?;
            k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_2"), asc.ptr(), wp("s_2"), wp("b_2"), gate_mlp, xp, xp, n as i64, fi,
                MLP as i64, DT_BF16, s(st)))?;
            self.blocks.release(b, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("block{b}"), &x, n * F * 2);
            }
        }

        k!(gpu_prologue_rows(DT_BF16, PRO_LAYERNORM, xp, fi, null(), 0, 0, null(), OUT_EPS, null(), null(), attn.ptr(), fi, m,
            F as i32, s(st)))?;
        let oc = self.stage.out_ch() as i64;
        k!(gpu_gemm(attn.ptr(), fi, self.g("outw"), fi, out.ptr(), oc, self.g("outb"), n as i64, oc, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        Ok(())
    }
}
