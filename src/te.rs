//! Krea2 text encoder: Qwen3-VL-4B language model (int8 ConvRot weights), conditioning taken
//! from the hidden states entering layers 2, 5, ..., 35 (no final norm), fp32 activations.

use crate::cuda::{memcpy_2d_d2d, memcpy_d2d, DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, PM, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::SafeTensors;
use crate::tokenizer::Tokenizer;
use crate::vision::Embeds;
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
const IMAGE_PAD: u32 = 151655;
const VISION_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

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

/// The prompt SwarmTextEncodeAdvanced builds for `count` images. Its `input_image_N` replacement
/// never reaches the tokens, because the node tokenizes each prompt leaf apart from that step, so
/// the text stays and every image follows the prompt.
pub fn image_prompt(prompt: &str, count: usize) -> String {
    let mut text = prompt.to_string();
    for i in 1..=count {
        text.push_str(&format!("Picture {i}: {VISION_BLOCK}"));
    }
    text
}

/// MRoPE positions (T, H, W) of `l` tokens with images at `spans` (start, tokens, merged grid), as
/// `qwen2vl_mrope_position_ids` lays them out: an image advances the text past its longest side.
fn position_ids(l: usize, spans: &[(usize, usize, (usize, usize))]) -> Vec<[f32; 3]> {
    let mut pos: Vec<[f32; 3]> = (0..l).map(|i| [i as f32; 3]).collect();
    let mut offset = 0i64;
    for &(start, size, (gh, gw)) in spans {
        let end = start + size;
        let len_max = gh.max(gw).max(1) as i64;
        let next = len_max + start as i64 + offset;
        for (i, p) in pos[end..].iter_mut().enumerate() {
            *p = [(next + i as i64) as f32; 3];
        }
        let origin = start as i64 + offset;
        for i in 0..size {
            pos[start + i] = [origin as f32, (origin + (i / gw) as i64) as f32, (origin + (i % gw) as i64) as f32];
        }
        offset += len_max - size as i64;
    }
    pos
}

/// RoPE table [l][64][(cos, sin)]. Interleaved MRoPE: T by default, H and W take every third pair of
/// the first 60.
fn rope_table(pos: &[[f32; 3]]) -> Vec<f32> {
    let inv: Vec<f32> = (0..64).map(|j| 1.0f32 / THETA.powf((2 * j) as f32 / HD as f32)).collect();
    let mut rope = Vec::with_capacity(pos.len() * 128);
    for p in pos {
        for (j, f) in inv.iter().enumerate() {
            let axis = if j < 60 { j % 3 } else { 0 };
            let a = f * p[axis];
            rope.push(a.cos());
            rope.push(a.sin());
        }
    }
    rope
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
        self.encode_with_images(prompt, &[], st)
    }

    /// `images` go in place of the prompt's first `<|image_pad|>` tokens; add them with [`image_prompt`].
    pub fn encode_with_images(&mut self, prompt: &str, images: &[&Embeds], st: &Stream) -> Result<Cond> {
        let ids = self.tokenize(prompt)?;
        let pads: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == IMAGE_PAD).map(|(i, _)| i).take(images.len()).collect();
        // the image tokens' places, in the sequence with the images in it
        let mut spans = Vec::with_capacity(pads.len());
        let mut grown = 0;
        for (&pad, img) in pads.iter().zip(images) {
            spans.push((pad + grown, img.tokens, img.grid));
            grown += img.tokens - 1;
        }
        let l0 = ids.len();
        let l = l0 + grown;

        // Embedding rows: gather int8 rows + scales on the host, dequantize + un-rotate on device.
        let emb = self.st.bytes(&format!("{PRE}embed_tokens.weight"))?;
        let emb_s = self.st.bytes(&format!("{PRE}embed_tokens.weight_scale"))?;
        let mut q = Vec::with_capacity(l0 * D);
        let mut sc = Vec::with_capacity(l0 * 4);
        for &t in &ids {
            let t = t as usize;
            q.extend_from_slice(&emb[t * D..(t + 1) * D]);
            sc.extend_from_slice(&emb_s[t * 4..t * 4 + 4]);
        }
        let q_d = DevBuf::alloc_async(q.len(), st)?;
        q_d.upload(0, &q, st)?;
        let s_d = DevBuf::alloc_async(sc.len(), st)?;
        s_d.upload(0, &sc, st)?;
        let text = DevBuf::alloc_async(l0 * D * 4, st)?;
        k!(krea_embed_dequant(q_d.ptr(), s_d.ptr(), text.ptr(), l0 as i32, D as i32, 1, s(st)))?;
        let x = DevBuf::alloc_async(l * D * 4, st)?;
        let row = D * 4;
        let (mut from, mut to) = (0, 0);
        for (&pad, img) in pads.iter().zip(images) {
            memcpy_d2d(off(x.ptr(), to * row), off(text.ptr(), from * row), (pad - from) * row, st)?;
            to += pad - from;
            memcpy_d2d(off(x.ptr(), to * row), img.merged.ptr(), img.tokens * row, st)?;
            to += img.tokens;
            from = pad + 1;
        }
        memcpy_d2d(off(x.ptr(), to * row), off(text.ptr(), from * row), (l0 - from) * row, st)?;
        // DeepStack: features added at the image tokens after each of the first layers
        let mut deepstack = Vec::new();
        if !pads.is_empty() {
            for d in 0..images[0].deepstack.len() {
                let b = DevBuf::alloc_async(l * row, st)?;
                b.upload(0, &vec![0u8; l * row], st)?;
                for (&(start, size, _), img) in spans.iter().zip(images) {
                    memcpy_d2d(off(b.ptr(), start * row), img.deepstack[d].ptr(), size * row, st)?;
                }
                deepstack.push(b);
            }
        }

        // RoPE (split-half)
        let rope = rope_table(&position_ids(l, &spans));
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
            k!(krea_qk_norm_rope(DT_F32, HD as i32, qkv.ptr(), QKV as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32, w("qn"),
                w("kn"), EPS, rope_d.ptr(), 2, m, s(st)))?;
            k!(krea_attn_small(DT_F32, qkv.ptr(), QKV as i64, off(qkv.ptr(), HQ * HD * 4),
                off(qkv.ptr(), (HQ + HK) * HD * 4), QKV as i64, HQ as i32, HK as i32, HD as i32, m, m, attn.ptr(),
                (HQ * HD) as i64, 1.0 / (HD as f32).sqrt(), 1, s(st)))?;
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
            if let Some(d) = deepstack.get(i) {
                let (p, c) = ([xp as crate::ffi::P, d.ptr() as crate::ffi::P], [1.0f32, 1.0]);
                k!(krea_lincomb(xp, (l * D) as i64, 2, p.as_ptr(), c.as_ptr(), s(st)))?;
            }
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
        if l0 > template_end + 3 && ids[template_end + 1] == 872 && ids[template_end + 2] == 198 {
            template_end += 3;
        }
        let len = l - template_end;
        let row = TAPS.len() * D;
        let ctx = DevBuf::alloc_async(len * row * 2, st)?;
        k!(krea_f32_to_bf16(off(taps.ptr(), template_end * row * 4), ctx.ptr(), (len * row) as i64, s(st)))?;
        Ok(Cond { ctx, len, token_ids: ids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_prompt_follows_swarm() {
        let block = VISION_BLOCK;
        assert_eq!(image_prompt("lake", 1), format!("lakePicture 1: {block}"));
        assert_eq!(image_prompt("a input_image_2", 2), format!("a input_image_2Picture 1: {block}Picture 2: {block}"));
    }

    #[test]
    fn positions_match_mrope() {
        // comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids on a 2 x 3 image at token 2 of 10
        let p = position_ids(10, &[(2, 6, (2, 3))]);
        assert_eq!(p.iter().map(|v| v[0]).collect::<Vec<_>>(), [0., 1., 2., 2., 2., 2., 2., 2., 5., 6.]);
        assert_eq!(p.iter().map(|v| v[1]).collect::<Vec<_>>(), [0., 1., 2., 2., 2., 3., 3., 3., 5., 6.]);
        assert_eq!(p.iter().map(|v| v[2]).collect::<Vec<_>>(), [0., 1., 2., 3., 4., 2., 3., 4., 5., 6.]);
    }
}
