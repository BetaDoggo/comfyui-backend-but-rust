//! MOSS Image text encoder: Qwen3.5 9B language model (int8 ConvRot weights, fp32 activations like
//! ComfyUI's full_precision_mm path). Three of every four layers are GatedDeltaNet linear attention
//! instead of self attention. Conditioning is the final-norm output of the last layer.

use crate::cuda::{memcpy_2d_d2d, DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, PM, PRO_GATE, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::SafeTensors;
use anyhow::Result;
use half::bf16;
use std::path::Path;

pub const D: usize = 4096;
const FF: usize = 12288;
const LAYERS: usize = 32;
const EPS: f32 = 1e-6;
// linear attention
const HV: usize = 32;
const HK: usize = 16;
const DV: usize = 128;
const CONV: usize = 2 * HK * DV + HV * DV; // q | k | v = 8192
const VAL: usize = HV * DV;
// self attention
const HA: usize = 16;
const HKV: usize = 4;
const HD: usize = 256;
const ROPE_PAIRS: usize = 32;
const THETA: f32 = 10_000_000.0;
const PRE: &str = "model.language_model.";

pub const TEMPLATE: &str = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n";

pub struct TextEncoder {
    st: SafeTensors,
    ws: WeightSet,
    /// final RMSNorm, f32
    norm: DevBuf,
    lora: crate::lora::State,
}

/// RMSNorm weight + 1 as f32, rounded the way ComfyUI's bf16 parameter is.
fn plus_one(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.f32_vec(name, true)?.into_iter().map(|x| bf16::from_f32(x + 1.0).to_f32()).collect())
}

fn is_linear(i: usize) -> bool {
    (i + 1) % 4 != 0
}

/// Bytes of the weights, for deciding how much of the encoder can stay in VRAM.
pub fn weight_bytes(st: &SafeTensors) -> Result<usize> {
    Ok(layer_layouts(st)?.iter().map(|l| l.size).sum())
}

fn layer_layouts(st: &SafeTensors) -> Result<Vec<Layout>> {
    let mut layouts = Vec::new();
    for i in 0..LAYERS {
        let n = |x: &str| format!("{PRE}layers.{i}.{x}");
        let mut l = Layout::new();
        let raw = |l: &mut Layout, w: &str, sc: &str, names: &[&str]| -> Result<()> {
            l.raw(st, w, &names.iter().map(|x| n(&format!("{x}.weight"))).collect::<Vec<_>>())?;
            l.raw(st, sc, &names.iter().map(|x| n(&format!("{x}.weight_scale"))).collect::<Vec<_>>())
        };
        raw(&mut l, "wgu", "sgu", &["mlp.gate_proj", "mlp.up_proj"])?;
        raw(&mut l, "wd", "sd", &["mlp.down_proj"])?;
        l.bytes("ln1", crate::st::f32_bytes(&plus_one(st, &n("input_layernorm.weight"))?));
        l.bytes("ln2", crate::st::f32_bytes(&plus_one(st, &n("post_attention_layernorm.weight"))?));
        if is_linear(i) {
            raw(&mut l, "wqkv", "sqkv", &["linear_attn.in_proj_qkv"])?;
            raw(&mut l, "wz", "sz", &["linear_attn.in_proj_z"])?;
            raw(&mut l, "wo", "so", &["linear_attn.out_proj"])?;
            l.bf16(st, "wa", &[n("linear_attn.in_proj_a.weight")])?;
            l.bf16(st, "wb", &[n("linear_attn.in_proj_b.weight")])?;
            l.bf16(st, "conv", &[n("linear_attn.conv1d.weight")])?;
            l.f32(st, "dt", &n("linear_attn.dt_bias"), 0.0)?;
            l.f32(st, "alog", &n("linear_attn.A_log"), 0.0)?;
            l.f32(st, "dn", &n("linear_attn.norm.weight"), 0.0)?;
        } else {
            raw(&mut l, "wq", "sq", &["self_attn.q_proj"])?;
            raw(&mut l, "wkv", "skv", &["self_attn.k_proj", "self_attn.v_proj"])?;
            raw(&mut l, "wo", "so", &["self_attn.o_proj"])?;
            l.bytes("qn", crate::st::f32_bytes(&plus_one(st, &n("self_attn.q_norm.weight"))?));
            l.bytes("kn", crate::st::f32_bytes(&plus_one(st, &n("self_attn.k_norm.weight"))?));
        }
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
        let v = plus_one(&st, &format!("{PRE}norm.weight"))?;
        let norm = DevBuf::alloc(v.len() * 4)?;
        norm.upload(0, &crate::st::f32_bytes(&v), s0)?;
        s0.sync()?;
        Ok(TextEncoder { st, ws, norm, lora: Default::default() })
    }

    /// Bake `loras` into the encoder. Returns true if any weight changed.
    pub fn apply_loras(&mut self, loras: &[crate::lora::Lora], s0: &Stream) -> Result<bool> {
        crate::lora::apply(crate::lora::Arch::TextEncoder { comfy: "qwen35_9b" }, &self.st, PRE, vec![crate::offload::Store::Set(&mut self.ws)], &mut self.lora, loras, s0)
    }

    /// ids -> hidden states f32 [len, 4096].
    pub fn encode(&mut self, ids: &[u32], st: &Stream, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<DevBuf> {
        let l = ids.len();
        let emb = self.st.bytes(&format!("{PRE}embed_tokens.weight"))?;
        let emb_s = self.st.bytes(&format!("{PRE}embed_tokens.weight_scale"))?;
        let mut q = Vec::with_capacity(l * D);
        let mut sc = Vec::with_capacity(l * 4);
        for &t in ids {
            let t = t as usize;
            q.extend_from_slice(&emb[t * D..(t + 1) * D]);
            sc.extend_from_slice(&emb_s[t * 4..t * 4 + 4]);
        }
        let q_d = DevBuf::alloc_async(q.len(), st)?;
        q_d.upload(0, &q, st)?;
        let s_d = DevBuf::alloc_async(sc.len(), st)?;
        s_d.upload(0, &sc, st)?;
        let x = DevBuf::alloc_async(l * D * 4, st)?;
        k!(gpu_embed_dequant(q_d.ptr(), s_d.ptr(), x.ptr(), l as i32, D as i32, 1, s(st)))?;

        let mut rope = Vec::with_capacity(l * ROPE_PAIRS * 2);
        for pos in 0..l {
            for j in 0..ROPE_PAIRS {
                let inv = 1.0f32 / THETA.powf(j as f32 / ROPE_PAIRS as f32);
                let a = inv * pos as f32;
                rope.push(a.cos());
                rope.push(a.sin());
            }
        }
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;

        let n = DevBuf::alloc_async(l * D * 4, st)?;
        let xr = DevBuf::alloc_async(l * FF * 4, st)?;
        let wf = DevBuf::alloc_async(2 * FF * D * 4, st)?; // largest dequantized weight
        let gu = DevBuf::alloc_async(l * 2 * FF * 4, st)?;
        let proj = DevBuf::alloc_async(l * CONV * 4, st)?;
        let mix = DevBuf::alloc_async(l * CONV * 4, st)?;
        let z = DevBuf::alloc_async(l * VAL * 4, st)?;
        let a = DevBuf::alloc_async(l * HV * 4, st)?;
        let b = DevBuf::alloc_async(l * HV * 4, st)?;
        let delta = DevBuf::alloc_async(l * VAL * 4, st)?;
        let kv = DevBuf::alloc_async(l * 2 * HKV * HD * 4, st)?;
        let aw = DevBuf::alloc_async(HV * D * 4, st)?;
        let (m, xp, d) = (l as i32, x.ptr(), D as i64);
        let attn_row = HA * HD;

        for i in 0..LAYERS {
            let base = self.ws.acquire(i, st)?;
            let lay = &self.ws.layouts[i];
            let w = |key: &str| off(base, lay.off(key));
            // x @ ((q*s) H)^T == (x H) @ (q*s)^T, so the activation is rotated instead of the weight
            let lin = |a: PM, wq: &str, sq: &str, rows: usize, kk: usize, out: PM, beta: f32| -> Result<()> {
                k!(gpu_dequant_i8(w(wq), w(sq), wf.ptr(), rows as i64, kk as i64, s(st)))?;
                k!(gpu_gemm(a, kk as i64, wf.ptr(), kk as i64, out, rows as i64, null(), l as i64, rows as i64, kk as i64, 1, DT_F32, DT_F32,
                    1.0, beta, s(st)))
            };
            let rotate = |a: PM, stride: usize, kk: usize| -> Result<()> {
                k!(gpu_rotate_rows_f32(PRO_PLAIN, a, stride as i64, 0, null(), 0.0, xr.ptr(), m, kk as i32, s(st)))
            };
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln1"), EPS, null(), null(), n.ptr(), d, m, D as i32, s(st)))?;
            if i == 0 {
                if let Some(p) = probe.as_mut() {
                    p("te_norm0", &n, l * D * 4);
                }
            }
            rotate(n.ptr(), D, D)?;
            if is_linear(i) {
                lin(xr.ptr(), "wqkv", "sqkv", CONV, D, proj.ptr(), 0.0)?;
                lin(xr.ptr(), "wz", "sz", VAL, D, z.ptr(), 0.0)?;
                for (key, out) in [("wa", &a), ("wb", &b)] {
                    k!(gpu_bf16_to_f32(w(key), aw.ptr(), (HV * D) as i64, s(st)))?;
                    k!(gpu_gemm(n.ptr(), d, aw.ptr(), d, out.ptr(), HV as i64, null(), l as i64, HV as i64, d, 1, DT_F32, DT_F32, 1.0, 0.0,
                        s(st)))?;
                }
                k!(gpu_delta_conv(proj.ptr(), w("conv"), mix.ptr(), m, CONV as i32, s(st)))?;
                k!(gpu_delta_rule(mix.ptr(), a.ptr(), b.ptr(), w("dt"), w("alog"), delta.ptr(), m, HV as i32, HK as i32, DV as i32, s(st)))?;
                k!(gpu_prologue_rows(DT_F32, PRO_RMS, delta.ptr(), DV as i64, null(), 0, 0, w("dn"), EPS, null(), null(), xr.ptr(), DV as i64,
                    (l * HV) as i32, DV as i32, s(st)))?;
                k!(gpu_mul_silu(xr.ptr(), z.ptr(), (l * VAL) as i64, s(st)))?;
                crate::cuda::memcpy_d2d(delta.ptr(), xr.ptr(), l * VAL * 4, st)?;
                rotate(delta.ptr(), VAL, VAL)?;
                lin(xr.ptr(), "wo", "so", D, VAL, xp, 1.0)?;
            } else {
                lin(xr.ptr(), "wq", "sq", 2 * attn_row, D, proj.ptr(), 0.0)?;
                lin(xr.ptr(), "wkv", "skv", 2 * HKV * HD, D, kv.ptr(), 0.0)?;
                // each head's q row is followed by its gate row
                let (qs, gs) = (mix.ptr(), delta.ptr());
                memcpy_2d_d2d(qs, HD * 4, proj.ptr(), 2 * HD * 4, HD * 4, l * HA, st)?;
                memcpy_2d_d2d(gs, HD * 4, off(proj.ptr(), HD * 4), 2 * HD * 4, HD * 4, l * HA, st)?;
                k!(gpu_partial_rope(qs, attn_row as i64, w("qn"), EPS, rope_d.ptr(), m, HA as i32, HD as i32, ROPE_PAIRS as i32, s(st)))?;
                k!(gpu_partial_rope(kv.ptr(), (2 * HKV * HD) as i64, w("kn"), EPS, rope_d.ptr(), m, HKV as i32, HD as i32, ROPE_PAIRS as i32,
                    s(st)))?;
                k!(gpu_attn_small(DT_F32, qs, attn_row as i64, kv.ptr(), off(kv.ptr(), HKV * HD * 4), (2 * HKV * HD) as i64, HA as i32,
                    HKV as i32, HD as i32, m, m, proj.ptr(), attn_row as i64, 1.0 / (HD as f32).sqrt(), 1, s(st)))?;
                k!(gpu_prologue_rows(DT_F32, PRO_GATE, proj.ptr(), attn_row as i64, gs, attn_row as i64, 0, null(), 0.0, null(), null(),
                    z.ptr(), attn_row as i64, m, attn_row as i32, s(st)))?;
                rotate(z.ptr(), attn_row, attn_row)?;
                lin(xr.ptr(), "wo", "so", D, attn_row, xp, 1.0)?;
            }
            if i == 0 {
                if let Some(p) = probe.as_mut() {
                    p("te_attn0_x", &x, l * D * 4);
                }
            }
            k!(gpu_rotate_rows_f32(PRO_RMS, xp, d, 0, w("ln2"), EPS, xr.ptr(), m, D as i32, s(st)))?;
            lin(xr.ptr(), "wgu", "sgu", 2 * FF, D, gu.ptr(), 0.0)?;
            k!(gpu_rotate_rows_f32(PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, FF as i64, null(), 0.0, xr.ptr(), m, FF as i32, s(st)))?;
            lin(xr.ptr(), "wd", "sd", D, FF, xp, 1.0)?;
            self.ws.release(i, st)?;
            if let Some(p) = probe.as_mut() {
                if matches!(i, 0 | 3 | 31) {
                    p(&format!("te_layer{i}"), &x, l * D * 4);
                }
            }
        }
        let out = DevBuf::alloc_async(l * D * 4, st)?;
        k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, self.norm.ptr(), EPS, null(), null(), out.ptr(), d, m, D as i32, s(st)))?;
        Ok(out)
    }
}
