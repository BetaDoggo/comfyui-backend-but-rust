//! Qwen3-VL language model as a text encoder (int8 ConvRot weights, fp32 activations), conditioning taken from
//! hidden states with no final norm. Krea 2 uses the 4B and the states entering layers 2, 5, ..., 35; Qwen Image
//! 2.1 uses the 8B and the state after its 36 layers.

use crate::cuda::{memcpy_2d_d2d, memcpy_d2d, DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, PM, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::SafeTensors;
use crate::tokenizer::Tokenizer;
use crate::vision::Embeds;
use anyhow::Result;
use std::path::Path;

pub const TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
/// Qwen Image 2.1's system turn, which is dropped from the conditioning again.
const QWEN21_SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
pub const TAPS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];

/// The size and conditioning of one encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    pub d: usize,
    pub ff: usize,
    pub pre: &'static str,
    /// Numbers of layers run before each tap of the hidden state, in order. The last one ends the run.
    pub taps: &'static [usize],
}

pub const KREA2: Spec = Spec { d: 2560, ff: 9728, pre: "model.language_model.", taps: &TAPS };
pub const QWEN21: Spec = Spec { d: 4096, ff: 12288, pre: "model.", taps: &[36] };

impl Spec {
    /// The encoder in `st`, told apart by where its layers are and how wide.
    pub fn detect(st: &SafeTensors) -> Result<Spec> {
        for spec in [KREA2, QWEN21] {
            if st.info(&format!("{}layers.0.input_layernorm.weight", spec.pre)).is_ok_and(|i| i.shape == [spec.d]) {
                return Ok(spec);
            }
        }
        anyhow::bail!("not a Qwen3-VL 4B or 8B text encoder")
    }

    fn qkv(&self) -> usize {
        (HQ + 2 * HK) * HD
    }
}

const HQ: usize = 32;
const HK: usize = 8;
const HD: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f32 = 5_000_000.0;
const IMAGE_PAD: u32 = 151655;
const IM_START: u32 = 151644;
const VISION_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

pub struct TextEncoder {
    st: SafeTensors,
    ws: WeightSet,
    tok: Tokenizer,
    lora: crate::lora::State,
    pub spec: Spec,
}

pub struct Cond {
    /// bf16 [len, taps * d]
    pub ctx: DevBuf,
    pub len: usize,
    /// Row width of `ctx`.
    pub width: usize,
    pub token_ids: Vec<u32>,
    /// Qwen Image 2.1: where each prompt image goes, as a count of the tokens before it.
    pub slots: Vec<usize>,
}

/// Where the encoder put an image: first token, tokens, merged grid.
type Span = (usize, usize, (usize, usize));

impl Cond {
    /// Conditionings joined along the tokens, as SwarmTextEncodeAdvanced joins `<break>` chunks.
    pub fn concat(mut conds: Vec<Cond>, st: &Stream) -> Result<Cond> {
        if conds.len() == 1 {
            return Ok(conds.pop().unwrap());
        }
        let (width, row) = (conds[0].width, conds[0].width * 2);
        let len: usize = conds.iter().map(|c| c.len).sum();
        let ctx = DevBuf::alloc_async(len * row, st)?;
        let (mut at, mut token_ids) = (0, Vec::new());
        for c in &conds {
            memcpy_d2d(off(ctx.ptr(), at * row), c.ctx.ptr(), c.len * row, st)?;
            at += c.len;
            token_ids.extend_from_slice(&c.token_ids);
        }
        st.sync()?;
        Ok(Cond { ctx, len, width, token_ids, slots: Vec::new() })
    }
}

/// The prompt of Qwen Image 2.1 with `count` images: each is named in front of the text, then the text follows.
pub fn qwen21_prompt(prompt: &str, count: usize) -> String {
    let refs: Vec<String> = (1..=count).map(|i| format!("<image{i}>{VISION_BLOCK}")).collect();
    format!("{QWEN21_SYSTEM}<|im_start|>user\n{}{prompt}<|im_end|>\n<|im_start|>assistant\n", refs.join(" "))
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
pub fn position_ids(l: usize, spans: &[(usize, usize, (usize, usize))]) -> Vec<[f32; 3]> {
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
pub fn rope_table(pos: &[[f32; 3]]) -> Vec<f32> {
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
        let spec = Spec::detect(&st)?;
        let n_layers = *spec.taps.last().unwrap(); // the last tap is the state after that many layers
        let mut layouts = Vec::new();
        for i in 0..n_layers {
            let p = format!("{}layers.{i}.", spec.pre);
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
        ws.once_per_prompt();
        let _ = s0;
        Ok(TextEncoder { st, ws, tok: Tokenizer::new()?, lora: Default::default(), spec })
    }

    /// Bake `loras` into the encoder. Returns true if any weight changed.
    pub fn apply_loras(&mut self, loras: &[crate::lora::Lora], s0: &Stream) -> Result<bool> {
        let comfy = if self.spec == QWEN21 { "qwen3vl_8b" } else { "qwen3vl_4b" };
        crate::lora::apply(crate::lora::Arch::TextEncoder { comfy }, &self.st, self.spec.pre, vec![crate::offload::Store::Set(&mut self.ws)], &mut self.lora, loras, s0)
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tok.encode(&TEMPLATE.replace("{}", prompt))
    }

    pub fn encode(&mut self, prompt: &str, st: &Stream) -> Result<Cond> {
        self.encode_with_images(prompt, &[], 1.0, st)
    }

    /// `images` go in place of the prompt's first `<|image_pad|>` tokens; add them with [`image_prompt`].
    /// `scale` multiplies the conditioning.
    pub fn encode_with_images(&mut self, prompt: &str, images: &[&Embeds], scale: f32, st: &Stream) -> Result<Cond> {
        let ids = self.tokenize(prompt)?;
        let (taps, l, _) = self.forward(&ids, images, st)?;
        let l0 = ids.len();

        // Drop the system prompt + "<|im_start|>user\n" (Krea2TEModel.encode_token_weights).
        let mut template_end = 0usize;
        let mut count = 0;
        for (i, &t) in ids.iter().enumerate() {
            if t == IM_START && count < 2 {
                template_end = i;
                count += 1;
            }
        }
        if l0 > template_end + 3 && ids[template_end + 1] == 872 && ids[template_end + 2] == 198 {
            template_end += 3;
        }
        let len = l - template_end;
        let width = self.spec.taps.len() * self.spec.d;
        if scale != 1.0 {
            let (p, c) = ([taps.ptr() as crate::ffi::P], [scale]);
            k!(gpu_lincomb(taps.ptr(), (l * width) as i64, 1, p.as_ptr(), c.as_ptr(), s(st)))?;
        }
        let ctx = DevBuf::alloc_async(len * width * 2, st)?;
        k!(gpu_f32_to_bf16(off(taps.ptr(), template_end * width * 4), ctx.ptr(), (len * width) as i64, s(st)))?;
        Ok(Cond { ctx, len, width, token_ids: ids, slots: Vec::new() })
    }

    /// Qwen Image 2.1's conditioning of `prompt` (see [`qwen21_prompt`]): the hidden state after the last layer, without
    /// the system turn, and, unless `keep_vision`, without the image tokens, which the DiT replaces by reference latents.
    /// `slots` of the result say where each of them was.
    pub fn encode_qwen21(&mut self, prompt: &str, images: &[&Embeds], keep_vision: bool, scale: f32, st: &Stream) -> Result<Cond> {
        let ids = self.tok.encode(prompt)?;
        let (taps, l, spans) = self.forward(&ids, images, st)?;
        // positions in the sequence with the images in it
        let (mut starts, mut grown, mut next) = (Vec::new(), 0, spans.iter());
        for (i, &t) in ids.iter().enumerate() {
            if t == IM_START {
                starts.push(i + grown);
            } else if t == IMAGE_PAD {
                if let Some(span) = next.next() {
                    grown += span.1 - 1;
                }
            }
        }
        let mut keep = vec![true; l];
        keep[..starts.get(1).copied().unwrap_or(0)].fill(false);
        let mut slots = Vec::new();
        if !keep_vision {
            for &(start, size, _) in &spans {
                keep[start..start + size].fill(false);
                slots.push(keep[..start].iter().filter(|k| **k).count());
            }
        }
        let (width, len) = (self.spec.d, keep.iter().filter(|k| **k).count());
        if scale != 1.0 {
            let (p, c) = ([taps.ptr() as crate::ffi::P], [scale]);
            k!(gpu_lincomb(taps.ptr(), (l * width) as i64, 1, p.as_ptr(), c.as_ptr(), s(st)))?;
        }
        let ctx = DevBuf::alloc_async(len * width * 2, st)?;
        let (mut at, mut i) = (0, 0);
        while i < l {
            if !keep[i] {
                i += 1;
                continue;
            }
            let run = keep[i..].iter().take_while(|k| **k).count();
            k!(gpu_f32_to_bf16(off(taps.ptr(), i * width * 4), off(ctx.ptr(), at * width * 2), (run * width) as i64, s(st)))?;
            at += run;
            i += run;
        }
        Ok(Cond { ctx, len, width, token_ids: ids, slots })
    }

    /// The encoder's hidden states at its taps, f32 [tokens, taps * d], with `images` in place of the first
    /// `<|image_pad|>` tokens, and where they went.
    fn forward(&mut self, ids: &[u32], images: &[&Embeds], st: &Stream) -> Result<(DevBuf, usize, Vec<Span>)> {
        let Spec { d: dm, ff, pre, taps: tap_at } = self.spec;
        let qkv_w = self.spec.qkv();
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
        let emb = self.st.bytes(&format!("{pre}embed_tokens.weight"))?;
        let emb_s = self.st.bytes(&format!("{pre}embed_tokens.weight_scale"))?;
        let mut q = Vec::with_capacity(l0 * dm);
        let mut sc = Vec::with_capacity(l0 * 4);
        for &t in ids {
            let t = t as usize;
            q.extend_from_slice(&emb[t * dm..(t + 1) * dm]);
            sc.extend_from_slice(&emb_s[t * 4..t * 4 + 4]);
        }
        let q_d = DevBuf::alloc_async(q.len(), st)?;
        q_d.upload(0, &q, st)?;
        let s_d = DevBuf::alloc_async(sc.len(), st)?;
        s_d.upload(0, &sc, st)?;
        let text = DevBuf::alloc_async(l0 * dm * 4, st)?;
        k!(gpu_embed_dequant(q_d.ptr(), s_d.ptr(), text.ptr(), l0 as i32, dm as i32, 1, s(st)))?;
        let x = DevBuf::alloc_async(l * dm * 4, st)?;
        let row = dm * 4;
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

        let xr = DevBuf::alloc_async(l * ff * 4, st)?;
        let wf = DevBuf::alloc_async(2 * ff * dm * 4, st)?; // largest dequantized weight
        let qkv = DevBuf::alloc_async(l * qkv_w * 4, st)?;
        let attn = DevBuf::alloc_async(l * HQ * HD * 4, st)?;
        let gu = DevBuf::alloc_async(l * 2 * ff * 4, st)?;
        let taps = DevBuf::alloc_async(l * tap_at.len() * dm * 4, st)?;
        let (m, xp) = (l as i32, x.ptr());

        for i in 0..=*tap_at.last().unwrap() {
            if let Some(t) = tap_at.iter().position(|&t| t == i) {
                memcpy_2d_d2d(off(taps.ptr(), t * dm * 4), tap_at.len() * dm * 4, xp, dm * 4, dm * 4, l, st)?;
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
                k!(gpu_dequant_i8(w(wq), w(sc), wf.ptr(), n as i64, kk as i64, s(st)))?;
                k!(gpu_gemm(xr, kk as i64, wf.ptr(), kk as i64, out, n as i64,
                    null(), l as i64, n as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, beta, s(st)))
            };
            // attention
            k!(gpu_rotate_rows_f32(PRO_RMS, xp, dm as i64, 0, w("ln1"), EPS, xr.ptr(), m, dm as i32, s(st)))?;
            lin(xr.ptr(), "wqkv", "sqkv", qkv_w, dm, qkv.ptr(), 0.0)?;
            k!(gpu_qk_norm_rope(DT_F32, HD as i32, qkv.ptr(), qkv_w as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32, w("qn"),
                w("kn"), EPS, rope_d.ptr(), 2, m, s(st)))?;
            k!(gpu_attn_small(DT_F32, qkv.ptr(), qkv_w as i64, off(qkv.ptr(), HQ * HD * 4),
                off(qkv.ptr(), (HQ + HK) * HD * 4), qkv_w as i64, HQ as i32, HK as i32, HD as i32, m, m, attn.ptr(),
                (HQ * HD) as i64, 1.0 / (HD as f32).sqrt(), 1, s(st)))?;
            k!(gpu_rotate_rows_f32(PRO_PLAIN, attn.ptr(), (HQ * HD) as i64, 0, null(), 0.0, xr.ptr(), m,
                (HQ * HD) as i32, s(st)))?;
            lin(xr.ptr(), "wo", "so", dm, HQ * HD, xp, 1.0)?;
            // MLP
            k!(gpu_rotate_rows_f32(PRO_RMS, xp, dm as i64, 0, w("ln2"), EPS, xr.ptr(), m, dm as i32, s(st)))?;
            lin(xr.ptr(), "wgu", "sgu", 2 * ff, dm, gu.ptr(), 0.0)?;
            k!(gpu_rotate_rows_f32(PRO_SWIGLU, gu.ptr(), (2 * ff) as i64, ff as i64, null(), 0.0, xr.ptr(), m,
                ff as i32, s(st)))?;
            lin(xr.ptr(), "wd", "sd", dm, ff, xp, 1.0)?;
            self.ws.release(i, st)?;
            if let Some(d) = deepstack.get(i) {
                let (p, c) = ([xp as crate::ffi::P, d.ptr() as crate::ffi::P], [1.0f32, 1.0]);
                k!(gpu_lincomb(xp, (l * dm) as i64, 2, p.as_ptr(), c.as_ptr(), s(st)))?;
            }
        }
        Ok((taps, l, spans))
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
