//! MiniMax H3 text encoder: Qwen3-VL 32B language model cut to 50 layers (nvfp4 AWQ weights), fp32
//! activations like ComfyUI's full_precision_mm path. Conditioning is the hidden state after the
//! last layer, with no final norm. Weights are decoded to f32 for each use, so the encoder only
//! needs VRAM for one layer and streams the rest.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, P, PM, PRO_RMS, PRO_SWIGLU};
use crate::offload::{Layout, Plan, WeightSet};
use crate::st::SafeTensors;
use crate::te::{position_ids, rope_table};
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

pub const D: usize = 5120;
const HQ: usize = 64;
const HK: usize = 8;
const HD: usize = 128;
const FF: usize = 25600;
const QKV: usize = (HQ + 2 * HK) * HD; // 10240
const LAYERS: usize = 50;
const EPS: f32 = 1e-6;
const PRE: &str = "model.";
const MODULES: [&str; 7] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

pub struct TextEncoder {
    st: Arc<SafeTensors>,
    ws: WeightSet,
    scales: Vec<[f32; 7]>,
}

fn layer_layouts(st: &SafeTensors) -> Result<Vec<Layout>> {
    let mut layouts = Vec::new();
    for i in 0..LAYERS {
        let n = |x: &str| format!("{PRE}layers.{i}.{x}");
        let mut l = Layout::new();
        for (j, m) in MODULES.iter().enumerate() {
            l.raw(st, &format!("w{j}"), &[n(&format!("{m}.weight"))])?;
            l.raw(st, &format!("s{j}"), &[n(&format!("{m}.weight_scale"))])?;
        }
        l.bf16(st, "pre_o", &[n("self_attn.o_proj.pre_quant_scale")])?;
        l.bf16(st, "pre_d", &[n("mlp.down_proj.pre_quant_scale")])?;
        l.f32(st, "ln1", &n("input_layernorm.weight"), 0.0)?;
        l.f32(st, "ln2", &n("post_attention_layernorm.weight"), 0.0)?;
        l.f32(st, "qn", &n("self_attn.q_norm.weight"), 0.0)?;
        l.f32(st, "kn", &n("self_attn.k_norm.weight"), 0.0)?;
        layouts.push(l);
    }
    Ok(layouts)
}

impl TextEncoder {
    /// Takes no VRAM until it encodes, then streams through two layers of it.
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        let st = Arc::new(SafeTensors::open(path)?);
        let layouts = layer_layouts(&st)?;
        let ws = WeightSet::lazy("text encoder", st.clone(), layouts, Plan { resident: 0, slots: 0 }, threads)?;
        let mut scales = Vec::new();
        for i in 0..LAYERS {
            let mut v = [0.0; 7];
            for (j, m) in MODULES.iter().enumerate() {
                v[j] = st.f32_vec(&format!("{PRE}layers.{i}.{m}.weight_scale_2"), false)?[0];
            }
            scales.push(v);
        }
        Ok(TextEncoder { st, ws, scales })
    }

    /// ids -> hidden states f32 [len, 5120].
    pub fn encode(&mut self, ids: &[u32], st: &Stream) -> Result<DevBuf> {
        self.ws.replan(Plan { resident: 0, slots: 2 })?;
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
        // the table is plain int8 here: no ConvRot, unlike the other encoders
        k!(gpu_dequant_i8(q_d.ptr(), s_d.ptr(), x.ptr(), l as i64, D as i64, s(st)))?;
        let rope = rope_table(&position_ids(l, &[]));
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;

        let n = DevBuf::alloc_async(l * D * 4, st)?;
        let wf = DevBuf::alloc_async(FF * D * 4, st)?;
        let qkv = DevBuf::alloc_async(l * QKV * 4, st)?;
        let attn = DevBuf::alloc_async(l * HQ * HD * 4, st)?;
        let gu = DevBuf::alloc_async(l * 2 * FF * 4, st)?;
        let act = DevBuf::alloc_async(l * FF * 4, st)?;
        let (m, xp, d) = (l as i32, x.ptr(), D as i64);

        for i in 0..LAYERS {
            let base = self.ws.acquire(i, st)?;
            let lay = &self.ws.layouts[i];
            let w = |key: &str| off(base, lay.off(key));
            let sc2 = self.scales[i];
            // out[.., rows] = a[l, kk] @ W^T, with a row stride of `ldd` in `out`
            let lin = |a: PM, j: usize, pre: P, rows: usize, kk: usize, out: PM, ldd: usize, beta: f32| -> Result<()> {
                k!(gpu_nvfp4_dequant(w(&format!("w{j}")), w(&format!("s{j}")), sc2[j], pre, wf.ptr(), rows as i64, kk as i32, s(st)))?;
                k!(gpu_gemm(a, kk as i64, wf.ptr(), kk as i64, out, ldd as i64, null(), l as i64, rows as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, beta, s(st)))
            };
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln1"), EPS, null(), null(), n.ptr(), d, m, D as i32, s(st)))?;
            lin(n.ptr(), 0, null(), HQ * HD, D, qkv.ptr(), QKV, 0.0)?;
            lin(n.ptr(), 1, null(), HK * HD, D, off(qkv.ptr(), HQ * HD * 4), QKV, 0.0)?;
            lin(n.ptr(), 2, null(), HK * HD, D, off(qkv.ptr(), (HQ + HK) * HD * 4), QKV, 0.0)?;
            k!(gpu_qk_norm_rope(DT_F32, HD as i32, qkv.ptr(), QKV as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32, w("qn"), w("kn"), EPS, rope_d.ptr(), 2, m, s(st)))?;
            k!(gpu_attn_small(DT_F32, qkv.ptr(), QKV as i64, off(qkv.ptr(), HQ * HD * 4), off(qkv.ptr(), (HQ + HK) * HD * 4), QKV as i64, HQ as i32, HK as i32, HD as i32, m, m, attn.ptr(),
                (HQ * HD) as i64, 1.0 / (HD as f32).sqrt(), 1, s(st)))?;
            lin(attn.ptr(), 3, w("pre_o") as P, D, HQ * HD, xp, D, 1.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln2"), EPS, null(), null(), n.ptr(), d, m, D as i32, s(st)))?;
            lin(n.ptr(), 4, null(), FF, D, gu.ptr(), 2 * FF, 0.0)?;
            lin(n.ptr(), 5, null(), FF, D, off(gu.ptr(), FF * 4), 2 * FF, 0.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, null(), 0, FF as i64, null(), 0.0, null(), null(), act.ptr(), FF as i64, m, FF as i32, s(st)))?;
            lin(act.ptr(), 6, w("pre_d") as P, D, FF, xp, D, 1.0)?;
            self.ws.release(i, st)?;
        }
        st.sync()?;
        self.ws.replan(Plan { resident: 0, slots: 0 })?;
        Ok(x)
    }
}
