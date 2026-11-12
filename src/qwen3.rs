//! Anima text encoder: Qwen3 0.6B (bf16 weights, fp32 activations like ComfyUI's manual_cast path),
//! conditioning is the final-norm output of the last layer.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, PM, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::SafeTensors;
use anyhow::Result;
use half::bf16;
use std::path::Path;

pub const D: usize = 1024;
const HQ: usize = 16;
const HK: usize = 8;
const HD: usize = 128;
const FF: usize = 3072;
const QKV: usize = (HQ + 2 * HK) * HD; // 4096
const LAYERS: usize = 28;
const EPS: f32 = 1e-6;
const THETA: f32 = 1_000_000.0;

pub struct TextEncoder {
    st: SafeTensors,
    ws: WeightSet,
    /// final RMSNorm, f32
    norm: DevBuf,
    lora: crate::lora::State,
}

/// Bytes of the weights, for deciding how much of the encoder can stay in VRAM.
pub fn weight_bytes(st: &SafeTensors) -> Result<usize> {
    Ok(layer_layouts(st)?.iter().map(|l| l.size).sum())
}

fn layer_layouts(st: &SafeTensors) -> Result<Vec<Layout>> {
    let mut layouts = Vec::new();
    for i in 0..LAYERS {
        let n = |x: &str| format!("model.layers.{i}.{x}");
        let mut l = Layout::new();
        l.bf16(st, "wqkv", &[n("self_attn.q_proj.weight"), n("self_attn.k_proj.weight"), n("self_attn.v_proj.weight")])?;
        l.bf16(st, "wo", &[n("self_attn.o_proj.weight")])?;
        l.bf16(st, "wgu", &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
        l.bf16(st, "wd", &[n("mlp.down_proj.weight")])?;
        l.f32(st, "ln1", &n("input_layernorm.weight"), 0.0)?;
        l.f32(st, "ln2", &n("post_attention_layernorm.weight"), 0.0)?;
        l.f32(st, "qn", &n("self_attn.q_norm.weight"), 0.0)?;
        l.f32(st, "kn", &n("self_attn.k_norm.weight"), 0.0)?;
        layouts.push(l);
    }
    Ok(layouts)
}

impl TextEncoder {
    pub fn load(path: &Path, vram_budget: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let layouts = layer_layouts(&st)?;
        let p = plan(&layouts, vram_budget, 2);
        let ws = WeightSet::load("text encoder", &st, layouts, p, threads)?;
        ws.once_per_prompt();
        let v = st.f32_vec("model.norm.weight", true)?;
        let norm = DevBuf::alloc(v.len() * 4)?;
        norm.upload(0, &crate::st::f32_bytes(&v), s0)?;
        s0.sync()?;
        Ok(TextEncoder { st, ws, norm, lora: Default::default() })
    }

    /// Bake `loras` into the encoder. Returns true if any weight changed.
    pub fn apply_loras(&mut self, loras: &[crate::lora::Lora], s0: &Stream) -> Result<bool> {
        crate::lora::apply(crate::lora::Arch::TextEncoder { comfy: "qwen3_06b" }, &self.st, "model.", vec![crate::offload::Store::Set(&mut self.ws)], &mut self.lora, loras, s0)
    }

    /// ids -> hidden states f32 [len, 1024].
    pub fn encode(&mut self, ids: &[u32], st: &Stream) -> Result<DevBuf> {
        let l = ids.len();
        let table = self.st.bytes("model.embed_tokens.weight")?;
        let mut emb = Vec::with_capacity(l * D * 4);
        for &t in ids {
            for c in table[t as usize * D * 2..(t as usize + 1) * D * 2].chunks_exact(2) {
                emb.extend_from_slice(&bf16::from_le_bytes([c[0], c[1]]).to_f32().to_le_bytes());
            }
        }
        let x = DevBuf::alloc_async(l * D * 4, st)?;
        x.upload(0, &emb, st)?;

        let mut rope = Vec::with_capacity(l * HD);
        for pos in 0..l {
            for j in 0..HD / 2 {
                let inv = 1.0f32 / THETA.powf((2 * j) as f32 / HD as f32);
                let a = inv * pos as f32;
                rope.push(a.cos());
                rope.push(a.sin());
            }
        }
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;

        let h = DevBuf::alloc_async(l * FF * 4, st)?;
        let wf = DevBuf::alloc_async(2 * FF * D * 4, st)?; // largest weight as f32
        let qkv = DevBuf::alloc_async(l * QKV * 4, st)?;
        let attn = DevBuf::alloc_async(l * HQ * HD * 4, st)?;
        let gu = DevBuf::alloc_async(l * 2 * FF * 4, st)?;
        let (m, xp, d) = (l as i32, x.ptr(), D as i64);

        for i in 0..LAYERS {
            let base = self.ws.acquire(i, st)?;
            let lay = &self.ws.layouts[i];
            let w = |key: &str| off(base, lay.off(key));
            let lin = |a: PM, wk: &str, n: usize, kk: usize, out: PM, beta: f32| -> Result<()> {
                k!(gpu_bf16_to_f32(w(wk), wf.ptr(), (n * kk) as i64, s(st)))?;
                k!(gpu_gemm(a, kk as i64, wf.ptr(), kk as i64, out, n as i64, null(), l as i64, n as i64, kk as i64, 1,
                    DT_F32, DT_F32, 1.0, beta, s(st)))
            };
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln1"), EPS, null(), null(), h.ptr(), d, m,
                D as i32, s(st)))?;
            lin(h.ptr(), "wqkv", QKV, D, qkv.ptr(), 0.0)?;
            let q = qkv.ptr();
            k!(gpu_qk_norm_rope(DT_F32, HD as i32, q, QKV as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32, w("qn"),
                w("kn"), EPS, rope_d.ptr(), 2, m, s(st)))?;
            k!(gpu_attn_small(DT_F32, q, QKV as i64, off(q, HQ * HD * 4), off(q, (HQ + HK) * HD * 4), QKV as i64,
                HQ as i32, HK as i32, HD as i32, m, m, attn.ptr(), (HQ * HD) as i64, 1.0 / (HD as f32).sqrt(), 1, s(st)))?;
            k!(gpu_prologue_rows(DT_F32, PRO_PLAIN, attn.ptr(), (HQ * HD) as i64, null(), 0, 0, null(), 0.0, null(),
                null(), h.ptr(), (HQ * HD) as i64, m, (HQ * HD) as i32, s(st)))?;
            lin(h.ptr(), "wo", D, HQ * HD, xp, 1.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln2"), EPS, null(), null(), h.ptr(), d, m,
                D as i32, s(st)))?;
            lin(h.ptr(), "wgu", 2 * FF, D, gu.ptr(), 0.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, null(), 0, FF as i64, null(), 0.0, null(),
                null(), h.ptr(), FF as i64, m, FF as i32, s(st)))?;
            lin(h.ptr(), "wd", D, FF, xp, 1.0)?;
            self.ws.release(i, st)?;
        }
        let out = DevBuf::alloc_async(l * D * 4, st)?;
        k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, self.norm.ptr(), EPS, null(), null(), out.ptr(), d, m,
            D as i32, s(st)))?;
        Ok(out)
    }
}
