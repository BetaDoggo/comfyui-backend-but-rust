//! Anima: Cosmos Predict2 MiniTrainDIT (int8 ConvRot blocks) with an LLM adapter that turns Qwen3
//! hidden states and T5 token ids into cross-attention context. bf16 activations like ComfyUI.

use crate::cuda::{self, DevBuf, Stream};
use crate::ffi::{k, null, off, s, MergeGroups, DT_BF16, MERGE_GROUPS, P, PM, PRO_GELU, PRO_LAYERNORM, PRO_PLAIN, PRO_RMS, PRO_SILU};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::snapref::{self, Adapter, LoraStream, Mat, Module, RefKv, RefRead};
use crate::st::SafeTensors;
use anyhow::{bail, Result};
use half::bf16;
use std::collections::HashMap;

const F: usize = 2048;
const H: usize = 16;
const HD: usize = 128;
const QKV: usize = 3 * F;
const KV: usize = 2 * F;
const MLP: usize = 8192;
const ADALN: usize = 256;
const MOD: usize = 3 * F;
pub const CTX: usize = 1024;
/// The blocks' width.
pub const WIDTH: usize = F;
const MIN_CTX: usize = 512;
const PATCH: usize = 2;
const CH: usize = 16;
const TOK: usize = CH * PATCH * PATCH;
const EPS: f32 = 1e-6;
// LLM adapter
const A_H: usize = 16;
const A_HD: usize = 64;
const A_MLP: usize = 4096;
const A_LAYERS: usize = 6;
const VOCAB: usize = 32128;
const A_THETA: f32 = 10000.0;
/// `rope_{h,w}_extrapolation_ratio` for 16-channel Cosmos models.
const ROPE_EXTRAPOLATION: f64 = 4.0;
/// Sigmas whose block modulation stays cached (1 MB each).
const MOD_CACHE: usize = 64;

pub struct Dit {
    prefix: String,
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    pub attn_variant: i32,
    pub int8_attn: bool,
    /// bf16 [blocks][3][shift | scale | gate] by sigma bits: it depends on nothing else, and
    /// repeated schedules skip 250 small kernels per step.
    mods: HashMap<u32, DevBuf>,
}

/// One prompt, ready for the blocks.
pub struct Context {
    /// bf16 [len, 1024], adapter output padded to at least 512 rows
    pub ctx: DevBuf,
    /// bf16 [blocks][len][k | v], normalized: both are timestep independent, so each block's
    /// cross-attention projections run once per prompt instead of once per step
    kv: DevBuf,
    pub len: usize,
}

/// What a model evaluation runs besides the plain model: SnapRef's streams (see [`crate::snapref`]).
#[derive(Clone, Copy)]
pub enum Pass<'a> {
    Plain,
    /// A target evaluation: the adapter's "tgt" LoRA, and the references' keys and values when `read` is set.
    Target { adapter: &'a Adapter, read: Option<(&'a RefKv, &'a RefRead)> },
    /// A reference: its "ref" LoRA, recording each block's self-attention k and v at row `row0` of `out`.
    Reference { adapter: &'a Adapter, out: &'a RefKv, row0: usize },
}

impl Pass<'_> {
    fn lora(&self) -> Option<(&Adapter, LoraStream)> {
        match *self {
            Pass::Plain => None,
            Pass::Target { adapter, .. } => Some((adapter, LoraStream::Tgt)),
            Pass::Reference { adapter, .. } => Some((adapter, LoraStream::Ref)),
        }
    }
}

/// [`Dit::time`]'s embeddings of a timestep.
struct TimeEmb {
    /// bf16 [F], normalized
    tn: DevBuf,
    /// bf16 [3F], the AdaLN LoRA
    lora: DevBuf,
    /// silu(tn), int8 and its scale
    a8t: DevBuf,
    ast: DevBuf,
}

/// File prefix of the diffusion model's keys, if `st` looks like Anima.
pub fn key_prefix(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    ["model.diffusion_model.", "net.", ""].into_iter().find(|p| has(&format!("{p}llm_adapter.blocks.0.cross_attn.q_proj.weight")))
}

fn prefix(st: &SafeTensors) -> Result<&'static str> {
    match key_prefix(|k| st.has(k)) {
        Some(p) => Ok(p),
        None => bail!("not an Anima model"),
    }
}

pub fn block_layout(st: &SafeTensors, i: usize) -> Result<Layout> {
    let p = prefix(st)?;
    let n = |x: &str| format!("{p}blocks.{i}.{x}");
    let mut l = Layout::new();
    for (key, a) in [("self", "self_attn"), ("cross", "cross_attn")] {
        let names = |suffix: &str| [n(&format!("{a}.q_proj.{suffix}")), n(&format!("{a}.k_proj.{suffix}")), n(&format!("{a}.v_proj.{suffix}"))];
        if key == "self" {
            l.raw(st, "w_self", &names("weight"))?;
            l.raw(st, "s_self", &names("weight_scale"))?;
        } else {
            let (w, sc) = (names("weight"), names("weight_scale"));
            l.raw(st, "w_cq", &w[..1])?;
            l.raw(st, "s_cq", &sc[..1])?;
            l.raw(st, "w_ckv", &w[1..])?;
            l.raw(st, "s_ckv", &sc[1..])?;
        }
        l.raw(st, &format!("wo_{key}"), &[n(&format!("{a}.output_proj.weight"))])?;
        l.raw(st, &format!("so_{key}"), &[n(&format!("{a}.output_proj.weight_scale"))])?;
        l.f32(st, &format!("qn_{key}"), &n(&format!("{a}.q_norm.weight")), 0.0)?;
        l.f32(st, &format!("kn_{key}"), &n(&format!("{a}.k_norm.weight")), 0.0)?;
    }
    for (j, a) in ["self_attn", "cross_attn", "mlp"].iter().enumerate() {
        for layer in 1..=2 {
            l.raw(st, &format!("w_mod{j}_{layer}"), &[n(&format!("adaln_modulation_{a}.{layer}.weight"))])?;
            l.raw(st, &format!("s_mod{j}_{layer}"), &[n(&format!("adaln_modulation_{a}.{layer}.weight_scale"))])?;
        }
    }
    l.raw(st, "w_l1", &[n("mlp.layer1.weight")])?;
    l.raw(st, "s_l1", &[n("mlp.layer1.weight_scale")])?;
    l.raw(st, "w_l2", &[n("mlp.layer2.weight")])?;
    l.raw(st, "s_l2", &[n("mlp.layer2.weight_scale")])?;
    Ok(l)
}

/// Everything outside the blocks (the LLM adapter too): small enough to always stay in VRAM.
pub fn globals_layout(st: &SafeTensors) -> Result<Layout> {
    let p = prefix(st)?;
    let n = |x: &str| format!("{p}{x}");
    let mut g = Layout::new();
    for (key, name) in [("x_w", "x_embedder.proj.1.weight"), ("t1", "t_embedder.1.linear_1.weight"), ("t2", "t_embedder.1.linear_2.weight"),
        ("fm1", "final_layer.adaln_modulation.1.weight"), ("fm2", "final_layer.adaln_modulation.2.weight"), ("fl", "final_layer.linear.weight"),
        ("a_out_w", "llm_adapter.out_proj.weight"), ("a_out_b", "llm_adapter.out_proj.bias")]
    {
        g.bf16(st, key, &[n(name)])?;
    }
    g.f32(st, "tn", &n("t_embedding_norm.weight"), 0.0)?;
    g.f32(st, "a_norm", &n("llm_adapter.norm.weight"), 0.0)?;
    for i in 0..A_LAYERS {
        let b = |x: &str| n(&format!("llm_adapter.blocks.{i}.{x}"));
        let key = |x: &str| format!("a{i}_{x}");
        for (norm, name) in [("ns", "norm_self_attn"), ("nc", "norm_cross_attn"), ("nm", "norm_mlp")] {
            g.f32(st, &key(norm), &b(&format!("{name}.weight")), 0.0)?;
        }
        g.bf16(st, &key("sqkv"), &[b("self_attn.q_proj.weight"), b("self_attn.k_proj.weight"), b("self_attn.v_proj.weight")])?;
        g.bf16(st, &key("so"), &[b("self_attn.o_proj.weight")])?;
        g.f32(st, &key("sqn"), &b("self_attn.q_norm.weight"), 0.0)?;
        g.f32(st, &key("skn"), &b("self_attn.k_norm.weight"), 0.0)?;
        g.bf16(st, &key("cq"), &[b("cross_attn.q_proj.weight")])?;
        g.bf16(st, &key("ckv"), &[b("cross_attn.k_proj.weight"), b("cross_attn.v_proj.weight")])?;
        g.bf16(st, &key("co"), &[b("cross_attn.o_proj.weight")])?;
        g.f32(st, &key("cqn"), &b("cross_attn.q_norm.weight"), 0.0)?;
        g.f32(st, &key("ckn"), &b("cross_attn.k_norm.weight"), 0.0)?;
        for (x, name) in [("m0w", "mlp.0.weight"), ("m0b", "mlp.0.bias"), ("m2w", "mlp.2.weight"), ("m2b", "mlp.2.bias")] {
            g.bf16(st, &key(x), &[b(name)])?;
        }
    }
    Ok(g)
}

pub fn n_blocks(st: &SafeTensors) -> Result<usize> {
    let p = prefix(st)?;
    Ok((0..).take_while(|i| st.has(&format!("{p}blocks.{i}.mlp.layer1.weight"))).count())
}

/// Rough activation footprint of one denoising step (bytes) for `l` image tokens.
pub fn step_activation_bytes(l: usize) -> usize {
    l * (F * 2 /*x*/ + QKV * 2 + F * 2 /*q*/ + F * 2 /*attn*/ + MLP * 2 + MLP /*int8*/ + TOK * 2 + 8) + 2 * l * 2 * F * 2 /*int8 attn*/
        + (128 << 20)
}

/// Rope table [L][64][(cos, sin)] for a ht x wt grid: 22 temporal, 21 height and 21 width pairs.
fn rope_table(ht: usize, wt: usize) -> Vec<f32> {
    rope_positions(ht, wt, 0.0, 0, 0)
}

/// [`rope_table`] for frame `t`, the grid's first row and column at `h0` and `w0` (SnapRef's reference).
fn rope_positions(ht: usize, wt: usize, t: f32, h0: i64, w0: i64) -> Vec<f32> {
    let (dim_h, dim_t) = (42usize, 44usize);
    let ntk = ROPE_EXTRAPOLATION.powf(dim_h as f64 / (dim_h - 2) as f64);
    let theta = (10000.0 * ntk) as f32;
    let spatial: Vec<f32> = (0..dim_h / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / dim_h as f32)).collect();
    // the temporal extrapolation ratio is 1
    let temporal: Vec<f32> = (0..dim_t / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / dim_t as f32)).collect();
    let mut table = Vec::with_capacity(ht * wt * HD);
    for y in 0..ht {
        for x in 0..wt {
            for f in &temporal {
                let a = t * f;
                table.extend_from_slice(&[a.cos(), a.sin()]);
            }
            for pos in [y as i64 + h0, x as i64 + w0] {
                for f in &spatial {
                    let a = pos as f32 * f;
                    table.extend_from_slice(&[a.cos(), a.sin()]);
                }
            }
        }
    }
    table
}

/// Adapter rope [rows][32][(cos, sin)].
fn adapter_rope(rows: usize) -> Vec<f32> {
    let mut t = Vec::with_capacity(rows * A_HD);
    for pos in 0..rows {
        for j in 0..A_HD / 2 {
            let inv = 1.0f32 / A_THETA.powf((2 * j) as f32 / A_HD as f32);
            let a = inv * pos as f32;
            // RotaryEmbedding returns these in the activation dtype
            t.extend_from_slice(&[bf16::from_f32(a.cos()).to_f32(), bf16::from_f32(a.sin()).to_f32()]);
        }
    }
    t
}

/// Cosmos `Timesteps`: cos first, then sin, timestep is sigma itself (multiplier 1).
fn timestep_embedding(sigma: f32) -> Vec<u8> {
    let half = F / 2;
    let neg_log = -(10000f64.ln()) as f32;
    let mut e = vec![0f32; F];
    for i in 0..half {
        let a = sigma * (neg_log * i as f32 / half as f32).exp();
        e[i] = a.cos();
        e[half + i] = a.sin();
    }
    e.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect()
}

impl Dit {
    pub fn load(st: &SafeTensors, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let gl = globals_layout(st)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let layouts = (0..n_blocks(st)?).map(|i| block_layout(st, i)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, block_budget, slots);
        let blocks = WeightSet::load("anima blocks", st, layouts, p, threads)?;
        Ok(Dit { prefix: prefix(st)?.to_string(), globals, gl, blocks, attn_variant: 0, int8_attn: true, mods: HashMap::new() })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Everywhere this model keeps weights.
    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Buf(&self.globals, &self.gl)]
    }

    /// The file's key prefix.
    pub fn prefix(&self) -> String {
        self.prefix.clone()
    }

    /// Forget what was derived from the weights.
    pub fn clear_mods(&mut self) {
        self.mods.clear();
    }

    /// LLM adapter: Qwen3 hidden states (f32 [lq, 1024]) + T5 ids -> bf16 [ids, 1024].
    fn adapter(&self, st_file: &SafeTensors, qwen: &DevBuf, lq: usize, ids: &[u32], st: &Stream) -> Result<DevBuf> {
        let l = ids.len();
        let (d, m, lqi) = (CTX as i64, l as i32, lq as i32);
        let table = st_file.bytes(&format!("{}llm_adapter.embed.weight", self.prefix))?;
        let mut emb = Vec::with_capacity(l * CTX * 2);
        for &t in ids {
            if t as usize >= VOCAB {
                bail!("T5 token {t} is out of range");
            }
            emb.extend_from_slice(&table[t as usize * CTX * 2..(t as usize + 1) * CTX * 2]);
        }
        let x = DevBuf::alloc_async(l * CTX * 2, st)?;
        x.upload(0, &emb, st)?;
        let cond = DevBuf::alloc_async(lq * CTX * 2, st)?;
        k!(gpu_f32_to_bf16(qwen.ptr(), cond.ptr(), (lq * CTX) as i64, s(st)))?;
        let rope = adapter_rope(l.max(lq));
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;

        let h = DevBuf::alloc_async(l * A_MLP * 2, st)?;
        let y = DevBuf::alloc_async(l * A_MLP * 2, st)?;
        let attn = DevBuf::alloc_async(l * CTX * 2, st)?;
        let qkv = DevBuf::alloc_async(l * 3 * CTX * 2, st)?;
        let kv = DevBuf::alloc_async(lq * 2 * CTX * 2, st)?;
        let scale = 1.0 / (A_HD as f32).sqrt();
        let (ad, ah, xp) = (A_HD as i32, A_H as i32, x.ptr());
        let gemm = |a: PM, w: PM, bias: PM, out: PM, rows: usize, n: usize, kk: usize| -> Result<()> {
            k!(gpu_gemm(a, kk as i64, w, kk as i64, out, n as i64, bias, rows as i64, n as i64, kk as i64, 1, DT_BF16, DT_BF16,
                1.0, 0.0, s(st)))
        };
        let nil = std::ptr::null_mut();
        for i in 0..A_LAYERS {
            let g = |x: &str| self.g(&format!("a{i}_{x}"));
            let rms = |key: &str| -> Result<()> {
                k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, d, null(), 0, 0, g(key), EPS, null(), null(), h.ptr(), d, m,
                    CTX as i32, s(st)))
            };
            // self attention
            rms("ns")?;
            gemm(h.ptr(), g("sqkv"), nil, qkv.ptr(), l, 3 * CTX, CTX)?;
            let q = qkv.ptr();
            k!(gpu_qk_norm_rope(DT_BF16, ad, q, (3 * CTX) as i64, 0, CTX as i32, ah, ah, g("sqn"), g("skn"), EPS,
                rope_d.ptr(), 2, m, s(st)))?;
            k!(gpu_attn_small(DT_BF16, q, (3 * CTX) as i64, off(q, CTX * 2), off(q, 2 * CTX * 2), (3 * CTX) as i64, ah, ah,
                ad, m, m, attn.ptr(), d, scale, 0, s(st)))?;
            gemm(attn.ptr(), g("so"), nil, y.ptr(), l, CTX, CTX)?;
            k!(gpu_add_gated_bf16(xp, d, y.ptr(), d, null(), m, CTX as i32, s(st)))?;
            // cross attention on the Qwen3 states
            rms("nc")?;
            gemm(h.ptr(), g("cq"), nil, qkv.ptr(), l, CTX, CTX)?;
            k!(gpu_qk_norm_rope(DT_BF16, ad, qkv.ptr(), d, 0, 0, ah, 0, g("cqn"), null(), EPS, rope_d.ptr(), 2, m, s(st)))?;
            gemm(cond.ptr(), g("ckv"), nil, kv.ptr(), lq, 2 * CTX, CTX)?;
            k!(gpu_qk_norm_rope(DT_BF16, ad, kv.ptr(), (2 * CTX) as i64, 0, 0, 0, ah, null(), g("ckn"), EPS, rope_d.ptr(), 2,
                lqi, s(st)))?;
            k!(gpu_attn_small(DT_BF16, qkv.ptr(), d, kv.ptr(), off(kv.ptr(), CTX * 2), (2 * CTX) as i64, ah, ah, ad, m, lqi,
                attn.ptr(), d, scale, 0, s(st)))?;
            gemm(attn.ptr(), g("co"), nil, y.ptr(), l, CTX, CTX)?;
            k!(gpu_add_gated_bf16(xp, d, y.ptr(), d, null(), m, CTX as i32, s(st)))?;
            // MLP
            rms("nm")?;
            gemm(h.ptr(), g("m0w"), g("m0b"), y.ptr(), l, A_MLP, CTX)?;
            k!(gpu_prologue_rows(DT_BF16, PRO_GELU, y.ptr(), A_MLP as i64, null(), 0, 0, null(), 0.0, null(), null(), h.ptr(),
                A_MLP as i64, m, A_MLP as i32, s(st)))?;
            gemm(h.ptr(), g("m2w"), g("m2b"), y.ptr(), l, CTX, A_MLP)?;
            k!(gpu_add_gated_bf16(xp, d, y.ptr(), d, null(), m, CTX as i32, s(st)))?;
        }
        gemm(xp, self.g("a_out_w"), self.g("a_out_b"), h.ptr(), l, CTX, CTX)?;
        let out = DevBuf::alloc_async(l * CTX * 2, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, h.ptr(), d, null(), 0, 0, self.g("a_norm"), EPS, null(), null(), out.ptr(), d,
            m, CTX as i32, s(st)))?;
        Ok(out)
    }

    /// Adapter output weighted by the T5 token weights and padded, plus every block's cross k and v.
    pub fn encode_context(&mut self, st_file: &SafeTensors, qwen: &DevBuf, lq: usize, ids: &[u32], weights: &[f32], st: &Stream)
        -> Result<Context> {
        let adapted = self.adapter(st_file, qwen, lq, ids, st)?;
        let mut host = vec![0u8; ids.len() * CTX * 2];
        adapted.download(0, &mut host, st)?;
        let len = ids.len().max(MIN_CTX);
        let mut padded = vec![0u8; len * CTX * 2];
        for (r, row) in host.chunks_exact(CTX * 2).enumerate() {
            for (o, v) in padded[r * CTX * 2..].chunks_exact_mut(2).zip(row.chunks_exact(2)) {
                let x = bf16::from_le_bytes([v[0], v[1]]).to_f32();
                o.copy_from_slice(&bf16::from_f32(x * weights[r]).to_le_bytes());
            }
        }
        let ctx = DevBuf::alloc_async(padded.len(), st)?;
        ctx.upload(0, &padded, st)?;

        let nb = self.blocks.len();
        let kv = DevBuf::alloc_async(nb * len * KV * 2, st)?;
        let a8 = DevBuf::alloc_async(len * CTX, st)?;
        let asc = DevBuf::alloc_async(len * 4, st)?;
        k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, ctx.ptr(), CTX as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(),
            asc.ptr(), len as i32, CTX as i32, 1, s(st)))?;
        for b in 0..nb {
            let base = self.blocks.acquire(b, st)?;
            let lay = &self.blocks.layouts[b];
            let out = off(kv.ptr(), b * len * KV * 2);
            k!(gpu_int8_gemm(a8.ptr(), off(base, lay.off("w_ckv")), asc.ptr(), off(base, lay.off("s_ckv")), null(), out,
                len as i64, KV as i64, CTX as i64, DT_BF16, s(st)))?;
            k!(gpu_qk_norm_rope(DT_BF16, HD as i32, out, KV as i64, 0, 0, 0, H as i32, null(), off(base, lay.off("kn_cross")),
                EPS, null(), 0, len as i32, s(st)))?;
            self.blocks.release(b, st)?;
        }
        Ok(Context { ctx, kv, len })
    }

    pub fn rope(&self, ht: usize, wt: usize, st: &Stream) -> Result<DevBuf> {
        let rope = rope_table(ht, wt);
        let d = DevBuf::alloc_async(rope.len() * 4, st)?;
        d.upload(0, &crate::st::f32_bytes(&rope), st)?;
        Ok(d)
    }

    /// Attention of `batch` images of `lq` rows each. `shared_kv`: every image reads the same keys and values.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, q: PM, kp: PM, vp: PM, out: PM, ws: PM, batch: usize, lq: usize, lk: usize, q_row: usize, kv_row: usize,
        shared_kv: bool, st: &Stream) -> Result<()> {
        let (hi, hs, sc, b) = (H as i32, HD as i64, 1.0 / (HD as f32).sqrt(), batch as i32);
        let (qb, qr, kr) = ((lq * q_row) as i64, q_row as i64, kv_row as i64);
        let kb = if shared_kv { 0 } else { (lk * kv_row) as i64 };
        let (ob, or) = ((lq * F) as i64, F as i64);
        if self.int8_attn && shared_kv && batch > 1 {
            // its quantizer takes no zero stride; the prompt's keys are few, so each image goes alone
            for i in 0..batch {
                self.attention(off(q, i * lq * q_row * 2), kp, vp, off(out, i * lq * F * 2), ws, 1, lq, lk, q_row, kv_row, true, st)?;
            }
            return Ok(());
        }
        if self.int8_attn {
            k!(gpu_int8_attn(q, kp, vp, out, ws, b, lq as i32, lk as i32, hi, hi, qb, qr, hs, kb, kr, hs, kb, kr, hs, ob, or, hs,
                sc, s(st)))
        } else {
            k!(gpu_flash_attn(q, kp, vp, out, ws, b, lq as i32, lk as i32, hi, hi, qb, qr, hs, kb, kr, hs, kb, kr, hs, ob, or, hs,
                sc, self.attn_variant, s(st)))
        }
    }

    /// The timestep embedding at `sigma`: its normalized form (plus `add`, SnapRef's mode embedding), the AdaLN LoRA,
    /// and silu of the former quantized for the modulation MLPs.
    fn time(&self, sigma: f32, add: Option<PM>, st: &Stream) -> Result<TimeEmb> {
        let fi = F as i64;
        // t_emb = norm(sinusoid); adaln_lora = linear_2(silu(linear_1(sinusoid)))
        let emb = DevBuf::alloc_async(F * 2, st)?;
        emb.upload(0, &timestep_embedding(sigma), st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        let t2 = DevBuf::alloc_async(F * 2, st)?;
        let lora = DevBuf::alloc_async(MOD * 2, st)?;
        let tn = DevBuf::alloc_async(F * 2, st)?;
        k!(gpu_gemm(emb.ptr(), fi, self.g("t1"), fi, t1.ptr(), fi, null(), 1, fi, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, t1.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), t2.ptr(), fi, 1,
            F as i32, s(st)))?;
        k!(gpu_gemm(t2.ptr(), fi, self.g("t2"), fi, lora.ptr(), MOD as i64, null(), 1, MOD as i64, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, emb.ptr(), fi, null(), 0, 0, self.g("tn"), EPS, null(), null(), tn.ptr(), fi, 1,
            F as i32, s(st)))?;
        if let Some(e) = add {
            k!(gpu_add_vec_bf16(tn.ptr(), tn.ptr(), e, 0, F as i32, s(st)))?;
        }
        // silu(t_emb) enters all 84 modulation MLPs, so it is quantized once
        let a8t = DevBuf::alloc_async(F, st)?;
        let ast = DevBuf::alloc_async(4, st)?;
        k!(gpu_quant_rows(DT_BF16, PRO_SILU, tn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), a8t.ptr(), ast.ptr(), 1,
            F as i32, 1, s(st)))?;
        Ok(TimeEmb { tn, lora, a8t, ast })
    }

    /// Patch embedding of `batch` latents (f32 [batch, 16, h, w]): bf16 [batch * l, F] tokens, and the token buffer
    /// the final layer reuses.
    fn embed(&self, latent: &DevBuf, batch: usize, h: usize, w: usize, st: &Stream) -> Result<(DevBuf, DevBuf)> {
        let l = (h / PATCH) * (w / PATCH);
        let (rows, plane, fi) = (batch * l, CH * h * w * 4, F as i64);
        // the 17th channel is an all-zero padding mask, so its columns are skipped
        let x = DevBuf::alloc_async(rows * F * 2, st)?;
        let tok = DevBuf::alloc_async(rows * TOK * 2, st)?;
        // cuBLAS picks its algorithm by the rows, so each image keeps its own GEMMs and its result alone
        for i in 0..batch {
            let ti = off(tok.ptr(), i * l * TOK * 2);
            k!(gpu_patchify(off(latent.ptr(), i * plane), ti, CH as i32, h as i32, w as i32, PATCH as i32, s(st)))?;
            k!(gpu_gemm(ti, TOK as i64, self.g("x_w"), (TOK + 4) as i64, off(x.ptr(), i * l * F * 2), fi, null(), l as i64, fi, TOK as i64, 1,
                DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        }
        Ok((x, tok))
    }

    /// One model evaluation of `batch` latents (f32 [batch, 16, H, W]) that share the prompt: writes the denoised ones
    /// to `den`. `pass` adds SnapRef's target stream.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, latent: &DevBuf, batch: usize, h: usize, w: usize, c: &Context, rope: &DevBuf, sigma: f32, den: &DevBuf,
        st: &Stream, pass: &Pass, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        if h % PATCH != 0 || w % PATCH != 0 {
            bail!("latent size must be even");
        }
        if matches!(pass, Pass::Reference { .. }) {
            bail!("a reference pass goes through encode_reference");
        }
        let l = (h / PATCH) * (w / PATCH);
        let rows = batch * l;
        let plane = CH * h * w * 4;
        let (fi, m, nil) = (F as i64, rows as i32, std::ptr::null_mut());
        let t = self.time(sigma, None, st)?;
        let (x, tok) = self.embed(latent, batch, h, w, st)?;
        if let Some(p) = probe.as_mut() {
            p("x_embedder_out", &x, rows * F * 2);
        }
        self.blocks(&x, batch, l, c, rope.ptr(), &t, Some(sigma), pass, st, &mut probe)?;

        // final layer: (1 + scale) * norm(x) + shift -> linear
        let xp = x.ptr();
        let normed = DevBuf::alloc_async(rows * F * 2, st)?;
        let ts = DevBuf::alloc_async(F * 2, st)?;
        let a256 = DevBuf::alloc_async(ADALN * 2, st)?;
        let fm = DevBuf::alloc_async(2 * F * 2, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, t.tn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), ts.ptr(), fi, 1, F as i32,
            s(st)))?;
        k!(gpu_gemm(ts.ptr(), fi, self.g("fm1"), fi, a256.ptr(), ADALN as i64, nil, 1, ADALN as i64, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0,
            s(st)))?;
        k!(gpu_gemm(a256.ptr(), ADALN as i64, self.g("fm2"), ADALN as i64, fm.ptr(), (2 * F) as i64, nil, 1, (2 * F) as i64,
            ADALN as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        k!(gpu_add_vec_bf16(fm.ptr(), fm.ptr(), t.lora.ptr(), 0, (2 * F) as i32, s(st)))?;
        k!(gpu_prologue_rows(DT_BF16, PRO_LAYERNORM, xp, fi, null(), 0, 0, null(), EPS, off(fm.ptr(), F * 2), fm.ptr(), normed.ptr(), fi,
            m, F as i32, s(st)))?;
        for i in 0..batch {
            k!(gpu_gemm(off(normed.ptr(), i * l * F * 2), fi, self.g("fl"), fi, off(tok.ptr(), i * l * TOK * 2), TOK as i64, nil, l as i64,
                TOK as i64, fi, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        }
        if let Some(p) = probe.as_mut() {
            p("final_tokens", &tok, rows * TOK * 2);
        }
        for i in 0..batch {
            k!(gpu_unpatch_denoised(off(tok.ptr(), i * l * TOK * 2), TOK as i64, off(latent.ptr(), i * plane), off(den.ptr(), i * plane),
                CH as i32, h as i32, w as i32, PATCH as i32, sigma, s(st)))?;
        }
        Ok(())
    }

    /// SnapRef's reference stream for one reference (f32 [16, h, w] latent at model scale): sigma 0, the adapter's
    /// "ref" LoRA and `mode`'s embedding, RoPE frame `ref_t` centred on a target of `target` latent pixels, prompted
    /// by `c` (the caption). Every block's self-attention k and v go to `out` from row `row0`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_reference(&mut self, latent: &DevBuf, h: usize, w: usize, target: (usize, usize), c: &Context, adapter: &Adapter,
        mode: &str, out: &RefKv, row0: usize, st: &Stream) -> Result<()> {
        if h % PATCH != 0 || w % PATCH != 0 {
            bail!("reference latent size must be even");
        }
        let (ht, wt) = (h / PATCH, w / PATCH);
        if row0 + ht * wt > out.rows || out.row_bytes != KV * 2 {
            bail!("reference keys don't fit their cache");
        }
        let (h0, w0) = snapref::offset(target, (ht, wt));
        let table = rope_positions(ht, wt, adapter.ref_t, h0, w0);
        let rope = DevBuf::alloc_async(table.len() * 4, st)?;
        rope.upload(0, &crate::st::f32_bytes(&table), st)?;
        let t = self.time(0.0, adapter.mode_embed(mode, F)?, st)?;
        let (x, _) = self.embed(latent, 1, h, w, st)?;
        self.blocks(&x, 1, ht * wt, c, rope.ptr(), &t, None, &Pass::Reference { adapter, out, row0 }, st, &mut None)
    }

    /// The transformer blocks over `x` (bf16 [batch * l, F] tokens), in place. `sigma` keys the modulation cache; the
    /// reference stream (`None`) computes its own. A reference pass stops once the last block's keys are recorded.
    #[allow(clippy::too_many_arguments)]
    fn blocks(&mut self, x: &DevBuf, batch: usize, l: usize, c: &Context, rope: PM, t: &TimeEmb, sigma: Option<f32>, pass: &Pass,
        st: &Stream, probe: &mut Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<()> {
        let rows = batch * l;
        let (fi, m) = (F as i64, rows as i32);
        let lora = pass.lora();
        let read = match pass {
            Pass::Target { read: Some((kv, r)), .. } => Some((*kv, *r)),
            _ => None,
        };

        let a8 = DevBuf::alloc_async(rows * MLP, st)?;
        let asc = DevBuf::alloc_async(rows * 4, st)?;
        let qkv = DevBuf::alloc_async(rows * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(rows * F * 2, st)?;
        let hid = DevBuf::alloc_async(rows * MLP * 2, st)?;
        let a8y = DevBuf::alloc_async(ADALN, st)?;
        let asy = DevBuf::alloc_async(4, st)?;
        let y256 = DevBuf::alloc_async(ADALN * 2, st)?;
        let cached = sigma.and_then(|sg| self.mods.remove(&sg.to_bits()));
        let hit = cached.is_some();
        if sigma.is_some() && !hit && self.mods.len() >= MOD_CACHE {
            self.mods.clear();
        }
        let modv = match cached {
            Some(m) => m,
            None if sigma.is_some() => DevBuf::alloc(self.blocks.len() * 3 * MOD * 2)?,
            None => DevBuf::alloc_async(self.blocks.len() * 3 * MOD * 2, st)?,
        };

        // SnapRef's LoRAs: their bf16 inputs, rank-wide intermediates and deltas
        let lora_bufs = match lora {
            Some((a, stream)) => {
                let wide = if stream == LoraStream::Ref { MLP } else { QKV };
                Some((DevBuf::alloc_async(rows * wide * 2, st)?, DevBuf::alloc_async(rows * 3 * a.max_rank.max(1) * 2, st)?,
                    DevBuf::alloc_async(rows * wide * 2, st)?, DevBuf::alloc_async(rows * F * 2, st)?))
            }
            None => None,
        };
        // the references' keys: prepended in a copy of each image's (plain), or a group of their own each (biased)
        let ref_rows = read.map_or(0, |(kv, _)| kv.rows);
        let groups = read.filter(|(_, r)| !r.plain()).map(|(kv, r)| r.groups(&kv.sizes));
        let cat = match read {
            Some((kv, r)) if r.plain() => Some(DevBuf::alloc_async(batch * (kv.rows + l) * KV * 2, st)?),
            _ => None,
        };
        let merge = match &groups {
            Some(g) => {
                let n = g.len() + 1;
                if n > MERGE_GROUPS {
                    bail!("SnapRef: {} references of different strengths (at most {})", g.len(), MERGE_GROUPS - 1);
                }
                Some((DevBuf::alloc_async(n * rows * F * 2, st)?, DevBuf::alloc_async(n * batch * H * l * 4, st)?))
            }
            None => None,
        };

        // flash attention only needs a softmax scratch; int8 attention quantizes q, k and v into it
        let lk_self = l + if cat.is_some() { ref_rows } else { 0 };
        let ws_bytes = if self.int8_attn {
            unsafe {
                let b = batch as i32;
                crate::ffi::gpu_int8_attn_workspace(b, H as i32, H as i32, l as i32, lk_self as i32)
                    .max(crate::ffi::gpu_int8_attn_workspace(b, H as i32, H as i32, l as i32, c.len as i32)) as usize
            }
        } else {
            H * rows * 4
        };
        let ws = DevBuf::alloc_async(ws_bytes, st)?;
        let xp = x.ptr();
        let q = qkv.ptr();
        let lc = c.len;
        let nb = self.blocks.len();

        for b in 0..nb {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            let bm = off(modv.ptr(), b * 3 * MOD * 2);
            for j in (0..3).filter(|_| !hit) {
                k!(gpu_int8_gemm(t.a8t.ptr(), wp(&format!("w_mod{j}_1")), t.ast.ptr(), wp(&format!("s_mod{j}_1")), null(), y256.ptr(), 1,
                    ADALN as i64, fi, DT_BF16, s(st)))?;
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, y256.ptr(), ADALN as i64, null(), 0, 0, null(), 0.0, null(), null(),
                    a8y.ptr(), asy.ptr(), 1, ADALN as i32, 1, s(st)))?;
                k!(gpu_int8_gemm(a8y.ptr(), wp(&format!("w_mod{j}_2")), asy.ptr(), wp(&format!("s_mod{j}_2")), null(),
                    off(bm, j * MOD * 2), 1, MOD as i64, ADALN as i64, DT_BF16, s(st)))?;
                k!(gpu_add_vec_bf16(off(bm, j * MOD * 2), off(bm, j * MOD * 2), t.lora.ptr(), 0, MOD as i32, s(st)))?;
            }
            let shift = |j: usize| off(bm, j * MOD * 2);
            let scale = |j: usize| off(bm, (j * MOD + F) * 2);
            let gate = |j: usize| off(bm, (j * MOD + 2 * F) * 2);
            let mat = |module: Module| lora.and_then(|(a, stream)| a.mat(stream, b, module).map(|mm| (a, mm)));
            let lb = lora_bufs.as_ref().map(|(xn, lt, dl, yb)| (xn.ptr(), lt.ptr(), dl.ptr(), yb.ptr()));
            // layer norm j's modulated output, as the projections' bf16 input
            let normed = |j: usize, out: PM| -> Result<()> {
                k!(gpu_prologue_rows(DT_BF16, PRO_LAYERNORM, xp, fi, null(), 0, 0, null(), EPS, scale(j), shift(j), out, fi, m, F as i32,
                    s(st)))
            };
            // an output projection: gate * (W a + lora(a)) added to x
            let out_proj = |module: Module, w: &str, sc: &str, j: usize, input: PM, width: usize| -> Result<()> {
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, input, width as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m,
                    width as i32, 1, s(st)))?;
                match (mat(module), lb) {
                    (Some((a, mm)), Some((_, lt, dl, yb))) => {
                        k!(gpu_int8_gemm(a8.ptr(), wp(w), asc.ptr(), wp(sc), null(), yb, rows as i64, fi, width as i64, DT_BF16, s(st)))?;
                        lora_delta(a, mm, input, width, rows, lt, dl, F, st)?;
                        k!(gpu_add_gated_bf16(yb, fi, dl, fi, null(), m, F as i32, s(st)))?;
                        k!(gpu_add_gated_bf16(xp, fi, yb, fi, gate(j), m, F as i32, s(st)))
                    }
                    _ => k!(gpu_int8_gemm_residual(a8.ptr(), wp(w), asc.ptr(), wp(sc), null(), gate(j), xp, xp, rows as i64, fi,
                        width as i64, DT_BF16, s(st))),
                }
            };

            // self attention
            k!(gpu_quant_rows(DT_BF16, PRO_LAYERNORM, xp, fi, null(), 0, 0, null(), EPS, scale(0), shift(0), a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_self"), asc.ptr(), wp("s_self"), null(), q, rows as i64, QKV as i64, fi, DT_BF16, s(st)))?;
            if let (Some((a, mm)), Some((xn, lt, dl, _))) = (mat(Module::SelfQkv), lb) {
                normed(0, xn)?;
                lora_delta(a, mm, xn, F, rows, lt, dl, QKV, st)?;
                k!(gpu_add_gated_bf16(q, QKV as i64, dl, QKV as i64, null(), m, QKV as i32, s(st)))?;
            }
            // the RoPE table is one image's
            for i in 0..batch {
                k!(gpu_qk_norm_rope(DT_BF16, HD as i32, off(q, i * l * QKV * 2), QKV as i64, 0, F as i32, H as i32, H as i32, wp("qn_self"),
                    wp("kn_self"), EPS, rope, 2, l as i32, s(st)))?;
            }
            if let Pass::Reference { out, row0, .. } = pass {
                cuda::memcpy_2d_d2d(off(out.block(b), row0 * KV * 2), KV * 2, off(q, F * 2), QKV * 2, KV * 2, rows, st)?;
                if b + 1 == nb {
                    // nothing after the last block's keys reaches them
                    self.blocks.release(b, st)?;
                    break;
                }
            }
            match (read, &cat, &merge, &groups) {
                (Some((kv, _)), Some(cat), _, _) => {
                    let lk = kv.rows + l;
                    for i in 0..batch {
                        let dst = off(cat.ptr(), i * lk * KV * 2);
                        cuda::memcpy_d2d(dst, kv.block(b), kv.rows * KV * 2, st)?;
                        cuda::memcpy_2d_d2d(off(dst, kv.rows * KV * 2), KV * 2, off(q, (i * l * QKV + F) * 2), QKV * 2, KV * 2, l, st)?;
                    }
                    self.attention(q, cat.ptr(), off(cat.ptr(), F * 2), attn.ptr(), ws.ptr(), batch, l, lk, QKV, KV, false, st)?;
                }
                (Some((kv, _)), None, Some((outs, lses)), Some(groups)) => {
                    let (hi, hs, sc, bi) = (H as i32, HD as i64, 1.0 / (HD as f32).sqrt(), batch as i32);
                    let (qb, qr) = ((l * QKV) as i64, QKV as i64);
                    let (ob, or) = ((l * F) as i64, F as i64);
                    let o = |g: usize| off(outs.ptr(), g * rows * F * 2);
                    let lse = |g: usize| off(lses.ptr(), g * batch * H * l * 4);
                    let mut mg = MergeGroups { o: [null(); MERGE_GROUPS], lse: [null(); MERGE_GROUPS], bias: [0.0; MERGE_GROUPS] };
                    // the image's own keys
                    k!(gpu_flash_attn(q, off(q, F * 2), off(q, KV * 2), o(0), lse(0), bi, l as i32, l as i32, hi, hi, qb, qr,
                        hs, qb, qr, hs, qb, qr, hs, ob, or, hs, sc, self.attn_variant, s(st)))?;
                    (mg.o[0], mg.lse[0]) = (o(0) as P, lse(0) as P);
                    for (g, &(start, size, bias)) in groups.iter().enumerate() {
                        let kr = off(kv.block(b), start * KV * 2);
                        let kvr = KV as i64;
                        k!(gpu_flash_attn(q, kr, off(kr, F * 2), o(g + 1), lse(g + 1), bi, l as i32, size as i32, hi, hi, qb, qr, hs, 0,
                            kvr, hs, 0, kvr, hs, ob, or, hs, sc, self.attn_variant, s(st)))?;
                        (mg.o[g + 1], mg.lse[g + 1], mg.bias[g + 1]) = (o(g + 1) as P, lse(g + 1) as P, bias);
                    }
                    k!(gpu_attn_merge(&mg, (groups.len() + 1) as i32, attn.ptr(), bi, l as i32, hi, HD as i32, s(st)))?;
                }
                _ => self.attention(q, off(q, F * 2), off(q, KV * 2), attn.ptr(), ws.ptr(), batch, l, l, QKV, QKV, false, st)?,
            }
            if b == 0 {
                if let Some(p) = probe.as_mut() {
                    p("block0_self_attn_raw", &attn, rows * F * 2);
                }
            }
            out_proj(Module::SelfOut, "wo_self", "so_self", 0, attn.ptr(), F)?;

            // cross attention, k and v come from the per-prompt cache
            let kv = off(c.kv.ptr(), b * lc * KV * 2);
            k!(gpu_quant_rows(DT_BF16, PRO_LAYERNORM, xp, fi, null(), 0, 0, null(), EPS, scale(1), shift(1), a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_cq"), asc.ptr(), wp("s_cq"), null(), q, rows as i64, fi, fi, DT_BF16, s(st)))?;
            if let (Some((a, mm)), Some((xn, lt, dl, _))) = (mat(Module::CrossQ), lb) {
                normed(1, xn)?;
                lora_delta(a, mm, xn, F, rows, lt, dl, F, st)?;
                k!(gpu_add_gated_bf16(q, fi, dl, fi, null(), m, F as i32, s(st)))?;
            }
            k!(gpu_qk_norm_rope(DT_BF16, HD as i32, q, fi, 0, 0, H as i32, 0, wp("qn_cross"), null(), EPS, null(), 0, m, s(st)))?;
            self.attention(q, kv, off(kv, F * 2), attn.ptr(), ws.ptr(), batch, l, lc, F, KV, true, st)?;
            out_proj(Module::CrossOut, "wo_cross", "so_cross", 1, attn.ptr(), F)?;

            // MLP
            k!(gpu_quant_rows(DT_BF16, PRO_LAYERNORM, xp, fi, null(), 0, 0, null(), EPS, scale(2), shift(2), a8.ptr(), asc.ptr(), m,
                F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_l1"), asc.ptr(), wp("s_l1"), null(), hid.ptr(), rows as i64, MLP as i64, fi, DT_BF16, s(st)))?;
            if let (Some((a, mm)), Some((xn, lt, dl, _))) = (mat(Module::Mlp1), lb) {
                normed(2, xn)?;
                lora_delta(a, mm, xn, F, rows, lt, dl, MLP, st)?;
                k!(gpu_add_gated_bf16(hid.ptr(), MLP as i64, dl, MLP as i64, null(), m, MLP as i32, s(st)))?;
            }
            match (mat(Module::Mlp2), lb) {
                (Some((a, mm)), Some((g, lt, dl, yb))) => {
                    k!(gpu_quant_rows(DT_BF16, PRO_GELU, hid.ptr(), MLP as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(),
                        asc.ptr(), m, MLP as i32, 1, s(st)))?;
                    k!(gpu_int8_gemm(a8.ptr(), wp("w_l2"), asc.ptr(), wp("s_l2"), null(), yb, rows as i64, fi, MLP as i64, DT_BF16, s(st)))?;
                    k!(gpu_prologue_rows(DT_BF16, PRO_GELU, hid.ptr(), MLP as i64, null(), 0, 0, null(), 0.0, null(), null(), g,
                        MLP as i64, m, MLP as i32, s(st)))?;
                    lora_delta(a, mm, g, MLP, rows, lt, dl, F, st)?;
                    k!(gpu_add_gated_bf16(yb, fi, dl, fi, null(), m, F as i32, s(st)))?;
                    k!(gpu_add_gated_bf16(xp, fi, yb, fi, gate(2), m, F as i32, s(st)))?;
                }
                _ => {
                    k!(gpu_quant_rows(DT_BF16, PRO_GELU, hid.ptr(), MLP as i64, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(),
                        asc.ptr(), m, MLP as i32, 1, s(st)))?;
                    k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_l2"), asc.ptr(), wp("s_l2"), null(), gate(2), xp, xp, rows as i64, fi,
                        MLP as i64, DT_BF16, s(st)))?;
                }
            }
            self.blocks.release(b, st)?;
            if let Some(p) = probe.as_mut() {
                p(&format!("block{b}"), x, rows * F * 2);
            }
        }
        if let Some(sg) = sigma {
            self.mods.insert(sg.to_bits(), modv);
        }
        Ok(())
    }
}

/// A projection's LoRA delta: `up(down(x)) * scale` into `out` (bf16, row stride `ldo`), each part into its own
/// columns. `x` is bf16 [rows, k] with row stride `ldx`; `tmp` takes [rows, parts * rank].
#[allow(clippy::too_many_arguments)]
fn lora_delta(a: &Adapter, mm: &Mat, x: PM, ldx: usize, rows: usize, tmp: PM, out: PM, ldo: usize, st: &Stream) -> Result<()> {
    let (r, pr) = (mm.rank, mm.parts * mm.rank);
    k!(gpu_gemm(x, ldx as i64, a.at(mm.down), mm.k as i64, tmp, pr as i64, null(), rows as i64, pr as i64, mm.k as i64, 1, DT_BF16,
        DT_BF16, 1.0, 0.0, s(st)))?;
    for p in 0..mm.parts {
        k!(gpu_gemm(off(tmp, p * r * 2), pr as i64, a.at(mm.up[p]), r as i64, off(out, p * mm.n * 2), ldo as i64, null(), rows as i64,
            mm.n as i64, r as i64, 1, DT_BF16, DT_BF16, mm.scale, 0.0, s(st)))?;
    }
    Ok(())
}

/// Activation memory SnapRef adds to a target step of `l` tokens per image next to `ref_rows` reference tokens: the
/// LoRA scratch, and the keys prepended (or their groups' outputs, `groups` of them).
pub fn snapref_step_bytes(l: usize, ref_rows: usize, groups: usize) -> usize {
    l * (QKV * 2 * 2 + F * 2 + 3 * 128 * 2) + (ref_rows + l) * KV * 2 * 2 + (groups + 1) * l * (F * 2 + H * 4)
}

/// Activation memory of SnapRef's reference stream over `l` reference tokens, the keys it keeps aside.
pub fn reference_bytes(l: usize) -> usize {
    step_activation_bytes(l) + l * (MLP * 2 * 2 + F * 2 + 3 * 256 * 2)
}

/// Bytes of the keys and values kept for `rows` reference tokens in a model of `blocks` blocks.
pub fn ref_kv_bytes(blocks: usize, rows: usize) -> usize {
    blocks * rows * KV * 2
}

/// Bytes of one reference row of [`RefKv`].
pub const REF_ROW_BYTES: usize = KV * 2;
