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
pub const IMAGE_PAD: u32 = 151655;
const IM_START: u32 = 151644;
/// `<|endoftext|>`, ComfyUI's pad token for Qwen3-VL, which SwarmUI drops from each prompt piece's tokens.
const PAD: u32 = 151643;
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

/// Rows of Qwen Image 2.1's conditioning of `ids` with `images` prompt images: the tokens from the second
/// `<|im_start|>` on, without the image tokens.
pub fn qwen21_rows(ids: &[u32], images: usize) -> usize {
    let start = ids.iter().enumerate().filter(|(_, &t)| t == IM_START).nth(1).map_or(0, |(i, _)| i);
    let mut pads = 0;
    let mut rows = 0;
    for (i, &t) in ids.iter().enumerate() {
        let image = t == IMAGE_PAD && pads < images;
        pads += (t == IMAGE_PAD) as usize;
        rows += (i >= start && !image) as usize;
    }
    rows
}

/// SwarmUI's `token_weights_from_tokens`: (row, weight) of each token whose weight isn't 1, the tokens lined up with
/// the last of the conditioning's `rows` rows.
pub fn qwen21_weight_rows(weights: &[f32], rows: usize) -> Vec<(usize, f32)> {
    let offset = rows as i64 - weights.len() as i64;
    let at = |i: usize| usize::try_from(offset + i as i64).ok().filter(|&p| p < rows);
    weights.iter().enumerate().filter(|(_, &w)| w != 1.0).filter_map(|(i, &w)| at(i).map(|p| (p, w))).collect()
}

/// Qwen Image 2.1's tokens of one `<break>` chunk of (text, weight) segments with `images` prompt images, and each
/// token's weight, as SwarmTextEncodeAdvanced's `combine_leaves(per_leaf=True)` builds them: each segment tokenized
/// alone, without pad tokens, and put in the template where `wrap_in_shell` finds the text goes. The template's
/// tokens weigh 1.
pub fn qwen21_tokens(tok: &Tokenizer, segments: &[(String, f32)], images: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    let empty = tok.encode(&qwen21_prompt("", images))?;
    let probe = tok.encode(&qwen21_prompt("x", images))?;
    let at = empty.iter().zip(&probe).take_while(|(a, b)| a == b).count();
    let (mut ids, mut weights) = (empty[..at].to_vec(), vec![1.0; at]);
    for (text, weight) in segments {
        if names_embedding(text) {
            return Err(crate::lora::Unsupported("textual inversion embeddings in prompts".into()).into());
        }
        // the tokenizer's escape_important: `\(` and `\)` are parentheses
        let text = text.replace("\\)", ")").replace("\\(", "(");
        for id in tok.encode(&text)?.into_iter().filter(|&t| t != PAD) {
            ids.push(id);
            weights.push(*weight);
        }
    }
    ids.extend_from_slice(&empty[at..]);
    weights.resize(ids.len(), 1.0);
    Ok((ids, weights))
}

/// True when ComfyUI's tokenizer would read `text` as naming a textual inversion embedding.
fn names_embedding(text: &str) -> bool {
    text.starts_with("embedding:") || text.char_indices().any(|(i, c)| c.is_whitespace() && text[i + c.len_utf8()..].starts_with("embedding:"))
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

/// ComfyUI's SDTokenizer takes `\(` and `\)` as escaped parentheses, even with weights disabled.
fn unescape(text: &str) -> String {
    text.replace("\\)", ")").replace("\\(", "(")
}

/// ComfyUI's SDTokenizer reads a word starting `embedding:` at the start or after whitespace as an embedding, and drops
/// it when there is none.
fn check_embedding(text: &str) -> Result<()> {
    let mut prev: Option<char> = None;
    for (i, c) in text.char_indices() {
        if prev.is_none_or(char::is_whitespace) && text[i..].starts_with("embedding:") {
            return Err(crate::lora::Unsupported("embedding: in a Krea 2 prompt (ComfyUI reads it as an embedding)".into()).into());
        }
        prev = Some(c);
    }
    Ok(())
}

/// Token ids of a Krea 2 prompt chunk of (text, weight) segments, and each token's weight. With `leaves`, as
/// SwarmTextEncodeAdvanced builds them (`combine_leaves(per_leaf=True)`): each segment is tokenized apart and put after
/// the template's head, and the tail after them carries the `images` prompt images. Otherwise as CLIPTextEncode: the
/// text joined and tokenized inside the template, unless it starts with `<|im_start|>`, with the images after the text.
pub fn krea_tokens(tok: &Tokenizer, segments: &[(String, f32)], leaves: bool, images: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    if !leaves {
        if segments.iter().any(|(_, w)| *w != 1.0) {
            anyhow::bail!("prompt weights on a prompt tokenized as one text");
        }
        let text: String = segments.iter().map(|(t, _)| t.as_str()).collect();
        check_embedding(&text)?;
        let text = image_prompt(&unescape(&text), images);
        let text = if text.starts_with("<|im_start|>") { text } else { TEMPLATE.replace("{}", &text) };
        let ids = tok.encode(&text)?;
        let weights = vec![1.0; ids.len()];
        return Ok((ids, weights));
    }
    let (head, tail) = TEMPLATE.split_once("{}").unwrap();
    let mut ids = tok.encode(head)?;
    let mut weights = vec![1.0; ids.len()];
    for (text, weight) in segments {
        check_embedding(text)?;
        ids.extend(tok.encode(&unescape(text))?.into_iter().filter(|&t| t != PAD));
        weights.resize(ids.len(), *weight);
    }
    ids.extend(tok.encode(&(image_prompt("", images) + tail))?);
    weights.resize(ids.len(), 1.0);
    Ok((ids, weights))
}

/// Where Krea 2's conditioning starts: after the system turn and `<|im_start|>user\n` (Krea2TEModel.encode_token_weights).
pub fn template_end(ids: &[u32]) -> usize {
    let mut end = 0usize;
    let mut count = 0;
    for (i, &t) in ids.iter().enumerate() {
        if t == IM_START && count < 2 {
            end = i;
            count += 1;
        }
    }
    if ids.len() > end + 3 && ids[end + 1] == 872 && ids[end + 2] == 198 {
        end += 3;
    }
    end
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

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Krea 2's conditioning of `ids` (from [`krea_tokens`]): `images` go in place of the first `<|image_pad|>` tokens,
    /// and the system turn is dropped. Then its `rows` (conditioning row, factor) are multiplied by their factors and
    /// every row by `whole`, as SwarmTextEncodeAdvanced weights it, and every row by `scale` (`ConditioningMultiply`).
    pub fn encode_krea(&mut self, ids: &[u32], images: &[&Embeds], rows: &[(usize, f32)], whole: f32, scale: f32, st: &Stream)
        -> Result<Cond> {
        let (taps, l, _) = self.forward(ids, images, st)?;
        let template_end = template_end(ids);
        let len = l - template_end;
        let width = self.spec.taps.len() * self.spec.d;
        let times = |first: usize, count: usize, c: f32| -> Result<()> {
            let x = off(taps.ptr(), first * width * 4);
            let (p, c) = ([x as crate::ffi::P], [c]);
            k!(gpu_lincomb(x, (count * width) as i64, 1, p.as_ptr(), c.as_ptr(), s(st)))
        };
        // runs of consecutive rows with one factor
        let mut i = 0;
        while i < rows.len() {
            let (pos, f) = rows[i];
            if pos >= len {
                anyhow::bail!("weighted row {pos} past the conditioning's {len}");
            }
            let n = rows[i..].iter().enumerate().take_while(|(j, r)| r.0 == pos + j && r.1 == f && r.0 < len).count();
            times(template_end + pos, n, f)?;
            i += n;
        }
        if whole != 1.0 {
            times(template_end, len, whole)?;
        }
        if scale != 1.0 {
            let (p, c) = ([taps.ptr() as crate::ffi::P], [scale]);
            k!(gpu_lincomb(taps.ptr(), (l * width) as i64, 1, p.as_ptr(), c.as_ptr(), s(st)))?;
        }
        let ctx = DevBuf::alloc_async(len * width * 2, st)?;
        k!(gpu_f32_to_bf16(off(taps.ptr(), template_end * width * 4), ctx.ptr(), (len * width) as i64, s(st)))?;
        Ok(Cond { ctx, len, width, token_ids: ids.to_vec(), slots: Vec::new() })
    }

    /// [`qwen21_tokens`] with this encoder's tokenizer.
    pub fn qwen21_tokens(&self, segments: &[(String, f32)], images: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        qwen21_tokens(&self.tok, segments, images)
    }

    /// Qwen Image 2.1's conditioning of `ids` (see [`TextEncoder::qwen21_tokens`]): the hidden state after the last
    /// layer, without the system turn, and, unless `keep_vision`, without the image tokens, which the DiT replaces by
    /// reference latents. `slots` of the result say where each of them was. Each of `row_weights` (row of the result,
    /// factor) multiplies a row, before `scale` multiplies them all.
    pub fn encode_qwen21(&mut self, ids: &[u32], images: &[&Embeds], keep_vision: bool, row_weights: &[(usize, f32)], scale: f32, st: &Stream)
        -> Result<Cond> {
        let ids = ids.to_vec();
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
        if !row_weights.is_empty() {
            let kept: Vec<usize> = (0..l).filter(|&i| keep[i]).collect();
            for &(row, w) in row_weights {
                let Some(&i) = kept.get(row) else { anyhow::bail!("a weight on row {row} of a {len} row conditioning") };
                let (p, c) = ([off(taps.ptr(), i * width * 4) as crate::ffi::P], [w]);
                k!(gpu_lincomb(off(taps.ptr(), i * width * 4), width as i64, 1, p.as_ptr(), c.as_ptr(), s(st)))?;
            }
        }
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

    /// SwarmTextEncodeAdvanced's tokens on ComfyUI's QwenImage21Tokenizer, from SwarmUI's own code: the ids after the
    /// system turn, the conditioning's rows and the weighted ones.
    #[test]
    fn qwen21_tokens_match_swarm() {
        let tok = Tokenizer::new().unwrap();
        let system = [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198];
        let images = [151644, 872, 198, 27, 1805, 16, 29, 151652, 151655, 151653, 366, 1805, 17, 29, 151652, 151655, 151653];
        let end = [151645, 198, 151644, 77091, 198];
        let seg = |s: &[(&str, f32)]| s.iter().map(|(t, w)| (t.to_string(), *w)).collect::<Vec<_>>();
        type Case<'a> = (Vec<(String, f32)>, &'a [u32], &'a [(usize, f32)]);
        let cases: [Case; 6] = [
            (seg(&[("a ", 1.0), ("red", 1.5), (" boat", 1.0)]), &[64, 220, 1151, 15328], &[(5, 1.5)]),
            (seg(&[("lake, ", 1.0), ("pine ", 1.3), ("bark", 2.6), (" (sky:1.2)", 1.0)]),
                &[62110, 11, 220, 38038, 220, 65, 838, 320, 26684, 25, 16, 13, 17, 8], &[(6, 1.3), (7, 1.3), (8, 2.6), (9, 2.6)]),
            (seg(&[("a \\(b\\) c", 1.0)]), &[64, 320, 65, 8, 272], &[]),
            (seg(&[("a", 1.0), ("b", 2.0), ("c", 1.0)]), &[64, 65, 66], &[(4, 2.0)]),
            // tokenized apart from the template's newline
            (seg(&[("\n\nfog over hills", 1.0)]), &[271, 69, 538, 916, 34131], &[]),
            (seg(&[("fog", 0.5), (" and ", 1.0), ("rain", 1.5)]), &[69, 538, 323, 220, 29093], &[(3, 0.5), (4, 0.5), (7, 1.5)]),
        ];
        for (segments, content, weighted) in cases {
            for n in [0, 2] {
                let (ids, weights) = qwen21_tokens(&tok, &segments, n).unwrap();
                let head: &[u32] = if n == 0 { &images[..3] } else { &images };
                let want: Vec<u32> = [&system[..], head, content, &end].concat();
                assert_eq!(ids, want, "{segments:?} with {n} images");
                let rows = qwen21_rows(&ids, n);
                assert_eq!(rows, want.len() - system.len() - n);
                // the images take no rows, and shift the text by the 12 tokens around them
                let shift = if n == 0 { 0 } else { 12 };
                let expect: Vec<(usize, f32)> = weighted.iter().map(|&(r, w)| (r + shift, w)).collect();
                assert_eq!(qwen21_weight_rows(&weights, rows), expect);
            }
        }
        assert!(qwen21_tokens(&tok, &seg(&[("a embedding:x", 1.0)]), 0).is_err());
    }

    /// Tokens after `<|im_start|>user\n`, and their weights.
    fn content(segments: &[(&str, f32)], leaves: bool, images: usize) -> (Vec<u32>, Vec<f32>) {
        let tok = Tokenizer::new().unwrap();
        let segments: Vec<(String, f32)> = segments.iter().map(|(t, w)| (t.to_string(), *w)).collect();
        let (ids, weights) = krea_tokens(&tok, &segments, leaves, images).unwrap();
        let head = if ids[0] == IM_START && ids[1] == 872 { 3 } else { 34 };
        assert_eq!(ids.len(), weights.len());
        (ids[head..].to_vec(), weights[head..].to_vec())
    }

    const TAIL: [u32; 5] = [151645, 198, 151644, 77091, 198];

    #[test]
    fn leaves_match_swarm_text_encode_advanced() {
        // ids and weights from SwarmTextEncodeAdvanced's combine_leaves(per_leaf=True) on ComfyUI's Krea2Tokenizer
        let (ids, w) = content(&[("a ", 1.0), ("red", 1.5), (" boat", 1.0)], true, 0);
        assert_eq!(ids, [&[64, 220, 1151, 15328][..], &TAIL].concat());
        assert_eq!(w[..4], [1.0, 1.0, 1.5, 1.0]);
        let (ids, _) = content(&[("a misty lake ", 1.0), ("<foo>", 1.0), (" at dawn", 1.0)], true, 0);
        assert_eq!(ids, [&[64, 8600, 88, 21800, 220, 27, 7975, 29, 518, 38393][..], &TAIL].concat());
        let (ids, _) = content(&[("\npine forest", 1.0)], true, 0);
        assert_eq!(ids, [&[198, 38038, 13638][..], &TAIL].concat());
        let (ids, _) = content(&[(r"a \(red\) (b:1.1) boat", 1.0)], true, 0);
        assert_eq!(ids, [&[64, 320, 1151, 8, 320, 65, 25, 16, 13, 16, 8, 15328][..], &TAIL].concat());
        let (ids, _) = content(&[], true, 0);
        assert_eq!(ids, TAIL);
        // prompt images follow every leaf
        let (ids, w) = content(&[("a ", 1.0), ("red", 1.5), (" lake", 1.0)], true, 2);
        let picture = |n| [24669, 220, n, 25, 220, 151652, 151655, 151653];
        assert_eq!(ids, [&[64, 220, 1151, 21800][..], &picture(16), &picture(17), &TAIL].concat());
        assert_eq!(w.iter().filter(|w| **w != 1.0).count(), 1);
    }

    #[test]
    fn joined_text_matches_clip_text_encode() {
        // Krea2Tokenizer.tokenize_with_weights: "\n" joins the template's, and a text starting <|im_start|> has no template
        assert_eq!(content(&[("a lake at dawn", 1.0)], false, 0).0, [&[64, 21800, 518, 38393][..], &TAIL].concat());
        let tok = Tokenizer::new().unwrap();
        let joined = |t: &str| krea_tokens(&tok, &[(t.to_string(), 1.0)], false, 0).unwrap().0;
        assert_eq!(joined("\nlake")[31..], [&[151644, 872, 271, 62110][..], &TAIL].concat());
        assert_eq!(joined("  \n lake")[31..], [&[151644, 872, 13887, 21800][..], &TAIL].concat());
        assert_eq!(joined(r"a \(red\) boat")[31..], [&[151644, 872, 198, 64, 320, 1151, 8, 15328][..], &TAIL].concat());
        assert_eq!(joined("<|im_start|>user\nlake<|im_end|>"), [151644, 872, 198, 62110, 151645]);
        assert_eq!(template_end(&joined("a lake at dawn")), 34);
    }

    #[test]
    fn embedding_words_are_refused() {
        let tok = Tokenizer::new().unwrap();
        for (t, leaves) in [("embedding:x", true), ("a embedding:x", true), ("a\nembedding:x", false), ("embedding:x", false)] {
            assert!(krea_tokens(&tok, &[(t.to_string(), 1.0)], leaves, 0).is_err(), "{t:?}");
        }
        assert!(krea_tokens(&tok, &[("a_embedding:x".to_string(), 1.0)], true, 0).is_ok());
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
