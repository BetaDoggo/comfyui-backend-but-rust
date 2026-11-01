//! Krea2 text encoder: Qwen3-VL-4B language model (int8 ConvRot weights), conditioning taken
//! from the hidden states entering layers 2, 5, ..., 35 (no final norm), fp32 activations.

use crate::cuda::{memcpy_2d_d2d, DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, PM, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::SafeTensors;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use std::path::Path;

pub const TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
pub const TAPS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];

const D: usize = 2560;
const HQ: usize = 32;
const HK: usize = 8;
const HD: usize = 128;
const FF: usize = 9728;
const QKV: usize = (HQ + 2 * HK) * HD; // 6144
const EPS: f32 = 1e-6;
const THETA: f32 = 5_000_000.0;
const PRE: &str = "model.language_model.";

pub struct TextEncoder {
    st: SafeTensors,
    ws: WeightSet,
    tok: Tokenizer,
}

pub struct Cond {
    /// bf16 [len, 12 * 2560]
    pub ctx: DevBuf,
    pub len: usize,
    pub token_ids: Vec<u32>,
}

impl TextEncoder {
    pub fn load(path: &Path, vram_budget: usize, threads: usize, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let n_layers = *TAPS.last().unwrap(); // layers 0..35; the last tap is layer 35's input
        let mut layouts = Vec::new();
        for i in 0..n_layers {
            let p = format!("{PRE}layers.{i}.");
            let n = |x: &str| format!("{p}{x}");
            let mut l = Layout::new();
            l.raw(&st, "wqkv", &[n("self_attn.q_proj.weight"), n("self_attn.k_proj.weight"), n("self_attn.v_proj.weight")])?;
            l.raw(&st, "sqkv", &[n("self_attn.q_proj.weight_scale"), n("self_attn.k_proj.weight_scale"), n("self_attn.v_proj.weight_scale")])?;
            l.raw(&st, "wo", &[n("self_attn.o_proj.weight")])?;
            l.raw(&st, "so", &[n("self_attn.o_proj.weight_scale")])?;
            l.raw(&st, "wgu", &[n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
            l.raw(&st, "sgu", &[n("mlp.gate_proj.weight_scale"), n("mlp.up_proj.weight_scale")])?;
            l.raw(&st, "wd", &[n("mlp.down_proj.weight")])?;
            l.raw(&st, "sd", &[n("mlp.down_proj.weight_scale")])?;
            l.f32(&st, "ln1", &n("input_layernorm.weight"), 0.0)?;
            l.f32(&st, "ln2", &n("post_attention_layernorm.weight"), 0.0)?;
            l.f32(&st, "qn", &n("self_attn.q_norm.weight"), 0.0)?;
            l.f32(&st, "kn", &n("self_attn.k_norm.weight"), 0.0)?;
            layouts.push(l);
        }
        let p = plan(&layouts, vram_budget, 2);
        let ws = WeightSet::load("text encoder", &st, layouts, p, threads)?;
        let _ = s0;
        Ok(TextEncoder { st, ws, tok: Tokenizer::new()? })
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tok.encode(&TEMPLATE.replace("{}", prompt))
    }

    pub fn encode(&mut self, prompt: &str, st: &Stream) -> Result<Cond> {
        let ids = self.tokenize(prompt)?;
        let l = ids.len();

        // Embedding rows: gather int8 rows + scales on the host, dequantize + un-rotate on device.
        let emb = self.st.bytes(&format!("{PRE}embed_tokens.weight"))?;
        let emb_s = self.st.bytes(&format!("{PRE}embed_tokens.weight_scale"))?;
        let mut q = Vec::with_capacity(l * D);
        let mut sc = Vec::with_capacity(l * 4);
        for &t in &ids {
            let t = t as usize;
            q.extend_from_slice(&emb[t * D..(t + 1) * D]);
            sc.extend_from_slice(&emb_s[t * 4..t * 4 + 4]);
        }
        let q_d = DevBuf::alloc_async(q.len(), st)?;
        q_d.upload(0, &q, st)?;
        let s_d = DevBuf::alloc_async(sc.len(), st)?;
        s_d.upload(0, &sc, st)?;
        let x = DevBuf::alloc_async(l * D * 4, st)?;
        k!(krea_embed_dequant(q_d.ptr(), s_d.ptr(), x.ptr(), l as i32, D as i32, 1, s(st)))?;

        // RoPE (split-half, text-only positions 0..l)
        let mut rope = Vec::with_capacity(l * 128);
        for pos in 0..l {
            for j in 0..64 {
                let inv = 1.0f32 / THETA.powf((2 * j) as f32 / HD as f32);
                let a = inv * pos as f32;
                rope.push(a.cos());
                rope.push(a.sin());
            }
        }
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;

        let xr = DevBuf::alloc_async(l * FF * 4, st)?;
        let wf = DevBuf::alloc_async(2 * FF * D * 4, st)?; // largest dequantized weight
        let qkv = DevBuf::alloc_async(l * QKV * 4, st)?;
        let attn = DevBuf::alloc_async(l * HQ * HD * 4, st)?;
        let gu = DevBuf::alloc_async(l * 2 * FF * 4, st)?;
        let taps = DevBuf::alloc_async(l * TAPS.len() * D * 4, st)?;
        let (m, xp) = (l as i32, x.ptr());

        for i in 0..=*TAPS.last().unwrap() {
            if let Some(t) = TAPS.iter().position(|&t| t == i) {
                memcpy_2d_d2d(off(taps.ptr(), t * D * 4), TAPS.len() * D * 4, xp, D * 4, D * 4, l, st)?;
            }
            if i == self.ws.len() {
                break;
            }
            let base = self.ws.acquire(i, st)?;
            let lay = &self.ws.layouts[i];
            let w = |key: &str| off(base, lay.off(key));

            // ComfyUI runs text encoders with full_precision_mm: weights are dequantized and the
            // matmul is fp32. x @ ((q*s) H)^T == (x H) @ (q*s)^T, so rotate the activation instead.
            let lin = |xr: PM, wq: &str, sc: &str, n: usize, kk: usize, out: PM, beta: f32| -> Result<()> {
                k!(krea_dequant_i8(w(wq), w(sc), wf.ptr(), n as i64, kk as i64, s(st)))?;
                k!(krea_gemm(xr, kk as i64, wf.ptr(), kk as i64, out, n as i64,
                    null(), l as i64, n as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, beta, s(st)))
            };
            // attention
            k!(krea_rotate_rows_f32(PRO_RMS, xp, D as i64, 0, w("ln1"), EPS, xr.ptr(), m, D as i32, s(st)))?;
            lin(xr.ptr(), "wqkv", "sqkv", QKV, D, qkv.ptr(), 0.0)?;
            k!(krea_qk_norm_rope(DT_F32, qkv.ptr(), QKV as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32, w("qn"),
                w("kn"), EPS, rope_d.ptr(), 2, m, s(st)))?;
            k!(krea_attn_small_f32(qkv.ptr(), QKV as i64, 0, (HQ * HD) as i32, ((HQ + HK) * HD) as i32, HQ as i32,
                HK as i32, m, attn.ptr(), (HQ * HD) as i64, 1.0 / (HD as f32).sqrt(), 1, s(st)))?;
            k!(krea_rotate_rows_f32(PRO_PLAIN, attn.ptr(), (HQ * HD) as i64, 0, null(), 0.0, xr.ptr(), m,
                (HQ * HD) as i32, s(st)))?;
            lin(xr.ptr(), "wo", "so", D, HQ * HD, xp, 1.0)?;
            // MLP
            k!(krea_rotate_rows_f32(PRO_RMS, xp, D as i64, 0, w("ln2"), EPS, xr.ptr(), m, D as i32, s(st)))?;
            lin(xr.ptr(), "wgu", "sgu", 2 * FF, D, gu.ptr(), 0.0)?;
            k!(krea_rotate_rows_f32(PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, FF as i64, null(), 0.0, xr.ptr(), m,
                FF as i32, s(st)))?;
            lin(xr.ptr(), "wd", "sd", D, FF, xp, 1.0)?;
            self.ws.release(i, st)?;
        }

        // Drop the system prompt + "<|im_start|>user\n" (Krea2TEModel.encode_token_weights).
        let mut template_end = 0usize;
        let mut count = 0;
        for (i, &t) in ids.iter().enumerate() {
            if t == 151644 && count < 2 {
                template_end = i;
                count += 1;
            }
        }
        if l > template_end + 3 && ids[template_end + 1] == 872 && ids[template_end + 2] == 198 {
            template_end += 3;
        }
        let len = l - template_end;
        let row = TAPS.len() * D;
        let ctx = DevBuf::alloc_async(len * row * 2, st)?;
        k!(krea_f32_to_bf16(off(taps.ptr(), template_end * row * 4), ctx.ptr(), (len * row) as i64, s(st)))?;
        Ok(Cond { ctx, len, token_ids: ids })
    }
}
