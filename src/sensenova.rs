//! SenseNova U1.6 (ComfyUI's `sensenova_u15`): a mixture of two transformers over one sequence, in pixel space with no
//! VAE. One checkpoint holds both, with the same Qwen3 8B shape and 42 layers each: the text tower runs the prompt once,
//! causally, and keeps each layer's keys and values; the generation tower (`_mot_gen` weights) runs the image's 32 x 32
//! pixel patches at every step, attending to those keys and values and its own. A pixel head turns its last state into
//! the image the model predicts.
//!
//! Like ComfyUI, the text tower runs its int8 ConvRot weights dequantized (here in float32, as this engine's other text
//! encoders do) and the generation tower W8A8 with bf16 activations.
//!
//! Edits put reference pictures in the prompt, as SenseNova Conditioning's `image` task does: each goes through the text
//! tower's own patch embedding into a block of image tokens after the user turn's first line, the block's tokens share
//! one text position and see each other both ways, and the negative prompt becomes the pictures alone.

use crate::cuda::{memcpy_2d_d2d, memcpy_d2d, DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, DT_F32, P, PM, PRO_PLAIN, PRO_RMS, PRO_SILU, PRO_SWIGLU};
use crate::offload::{plan, Layout, Store, WeightSet};
use crate::st::{bf16_bytes, f32_bytes, SafeTensors};
use crate::tokenizer::Tokenizer;
use anyhow::{bail, Context, Result};
use std::path::Path;

pub const F: usize = 4096;
const FF: usize = 12288;
const HQ: usize = 32;
const HK: usize = 8;
const HD: usize = 128;
pub const KV: usize = HK * HD;
const QKV: usize = HQ * HD + 2 * KV;
pub const LAYERS: usize = 42;
const EPS: f32 = 1e-6;
const TDIM: usize = 256;
/// Pixels per image token: a 16-pixel patch conv, then a 2x2 one.
pub const PATCH: usize = 32;
/// Channels of the pixel patch conv.
const PE: usize = 1024;
/// RoPE of the text positions (the first 64 values of a head), and of the rows and columns (32 each).
const TEXT_THETA: f32 = 5_000_000.0;
const HW_THETA: f32 = 10_000.0;
/// ComfyUI's `sampling_settings["shift"]`.
pub const SHIFT: f64 = 3.0;

pub const IM_START: u32 = 151644;
pub const IMAGE_CONTEXT: u32 = 151669;
pub const IMAGE_START: u32 = 151670;
pub const IMAGE_END: u32 = 151671;
const IM_END: u32 = 151645;
const USER: u32 = 872;
const ASSISTANT: u32 = 77091;
const NEWLINE: u32 = 198;
/// "Image-1:" and so on, before each picture when there are several: `Image`, `-`, the digits from `0`, `:`.
const IMAGE_LABEL: u32 = 1906;
const HYPHEN: u32 = 12;
const DIGIT_ZERO: u32 = 15;
const COLON: u32 = 25;
/// ImageNet's mean and standard deviation, which `preprocess_reference` normalizes pictures by.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A reference picture: RGB f32 [height][width][3], 0 to 1.
#[derive(Clone, Debug)]
pub struct Reference {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<f32>,
}

impl Reference {
    /// Its image tokens, rows and columns: one per 32 pixels, rounded up (`reference_grid`).
    pub fn grid(&self) -> (usize, usize) {
        (self.height.div_ceil(PATCH).max(1), self.width.div_ceil(PATCH).max(1))
    }

    /// `preprocess_reference` then `_pad_to_merged_patch_size`: normalized f32 [3, h, w] at whole tokens, the padding
    /// wrapping round to the other edge (`pad_to_patch_size`'s circular mode).
    fn pixels(&self) -> Vec<f32> {
        let (gh, gw) = self.grid();
        let (h, w) = (gh * PATCH, gw * PATCH);
        let mut out = vec![0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                let sy = y % self.height;
                for x in 0..w {
                    let v = self.rgb[(sy * self.width + x % self.width) * 3 + c];
                    out[(c * h + y) * w + x] = (v - MEAN[c]) / STD[c];
                }
            }
        }
        out
    }

    /// A key for the prompt cache.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.width, self.height).hash(&mut h);
        for v in &self.rgb {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }
}

/// `condition_input_ids`: a prompt's ids with a block of image tokens for each picture of `grids` after the user turn's
/// first line, each labelled when there are several; or with `image_only`, the pictures alone in a user turn of their
/// own (the negative prompt of an edit).
pub fn with_images(ids: &[u32], grids: &[(usize, usize)], image_only: bool) -> Vec<u32> {
    let block = |&(h, w): &(usize, usize)| std::iter::once(IMAGE_START).chain(std::iter::repeat_n(IMAGE_CONTEXT, h * w)).chain([IMAGE_END]);
    if image_only {
        let mut v = vec![IM_START, USER, NEWLINE];
        v.extend(grids.iter().flat_map(block));
        v.extend([IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE, IMAGE_START]);
        return v;
    }
    let starts: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == IM_START).map(|(i, _)| i).collect();
    let at = if starts.len() > 1 { starts[1] + 3 } else { ids.len() };
    let mut inserted = Vec::new();
    for (i, g) in grids.iter().enumerate() {
        if grids.len() > 1 {
            inserted.extend([IMAGE_LABEL, HYPHEN]);
            inserted.extend((i + 1).to_string().bytes().map(|d| DIGIT_ZERO + (d - b'0') as u32));
            inserted.push(COLON);
        }
        inserted.extend(block(g));
        inserted.push(NEWLINE);
    }
    let mut v = ids.to_vec();
    v.splice(at..at, inserted);
    v
}

/// The runs of image tokens inside image blocks (`image_token_mask`), as row ranges.
fn image_runs(ids: &[u32]) -> Vec<(usize, usize)> {
    let (mut runs, mut inside) = (Vec::<(usize, usize)>::new(), false);
    for (i, &t) in ids.iter().enumerate() {
        match t {
            IMAGE_START => inside = true,
            IMAGE_END => inside = false,
            IMAGE_CONTEXT if inside => match runs.last_mut() {
                Some(r) if r.1 == i => r.1 = i + 1,
                _ => runs.push((i, i + 1)),
            },
            _ => {}
        }
    }
    runs
}

/// `thw_indexes`: each row's (text, row, column) position. A text token takes the next text position, and an image
/// block's tokens all take the one after its start token, each at its row and column. Returns them and the position
/// the generated image takes, one past the largest.
fn positions(ids: &[u32], runs: &[(usize, usize)], grids: &[(usize, usize)]) -> (Vec<[f32; 3]>, usize) {
    let mut pos = Vec::with_capacity(ids.len());
    let mut t: i64 = -1;
    let mut run = 0;
    for i in 0..ids.len() {
        let after_start = i > 0 && ids[i - 1] == IMAGE_START;
        let image = runs.get(run).is_some_and(|r| (r.0..r.1).contains(&i));
        t += after_start as i64 + !image as i64;
        if image {
            let k = i - runs[run].0;
            let gw = grids[run].1;
            pos.push([t as f32, (k / gw) as f32, (k % gw) as f32]);
            if i + 1 == runs[run].1 {
                run += 1;
            }
        } else {
            pos.push([t as f32, 0.0, 0.0]);
        }
    }
    (pos, (t + 1) as usize)
}

/// Where the weights sit in the file: the release's own names, or ComfyUI's prefix on them.
const PREFIXES: [&str; 2] = ["", "model.diffusion_model."];

/// The key prefix of a SenseNova checkpoint, by the keys ComfyUI's `convert_checkpoint` tests.
pub fn key_prefix(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    PREFIXES.into_iter().find(|p| {
        ["fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight", "fm_modules.fm_head.conv1.weight",
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight", "language_model.model.embed_tokens.weight", "language_model.model.norm.weight"]
            .iter()
            .all(|k| has(&format!("{p}{k}")))
    })
}

include!("sensenova_system.rs");

/// The prompt ComfyUI's `SenseNovaTokenizer` builds for an image without thinking: the system turn and the text in
/// three parts tokenized apart, the text's with no special tokens; an empty prompt is the unconditional one.
pub fn prompt_parts(text: &str) -> Vec<(String, bool)> {
    if text.is_empty() {
        return vec![("<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>".into(), true)];
    }
    vec![
        (format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user"), true),
        (format!("\n{text}"), false),
        ("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<img>".into(), true),
    ]
}

/// Token ids of a prompt, as `SenseNovaTokenizer.tokenize_with_weights` makes them: in each part `\(` and `\)` are
/// parentheses, and the part is tokenized in pieces split before an `embedding:` that follows whitespace, as
/// SDTokenizer splits it.
pub fn tokens(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for (part, specials) in prompt_parts(text) {
        let part = part.replace("\\)", ")").replace("\\(", "(");
        for piece in embedding_pieces(&part) {
            ids.extend(if specials { tok.encode(piece)? } else { tok.encode_text(piece)? });
        }
    }
    Ok(ids)
}

/// `re.split(r"(?<=\s)(?=embedding:)", text)`.
fn embedding_pieces(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut prev: Option<char> = None;
    for (i, c) in text.char_indices() {
        if i > start && prev.is_some_and(char::is_whitespace) && text[i..].starts_with("embedding:") {
            pieces.push(&text[start..i]);
            start = i;
        }
        prev = Some(c);
    }
    pieces.push(&text[start..]);
    pieces
}

/// `theta ** (-arange(0, dim, 2) / dim)` in float32.
fn freqs(theta: f32, dim: usize) -> Vec<f32> {
    (0..dim / 2).map(|i| theta.powf(-((2 * i) as f32) / dim as f32)).collect()
}

/// The RoPE table `gpu_sn_qk` takes, [rows][64][(cos, sin)], of positions (text, row, column).
fn rope_table(pos: impl Iterator<Item = [f32; 3]>) -> Vec<f32> {
    let (ft, fhw) = (freqs(TEXT_THETA, HD / 2), freqs(HW_THETA, HD / 4));
    let mut t = Vec::new();
    for p in pos {
        for (axis, f) in [(0, &ft), (1, &fhw), (2, &fhw)] {
            for &w in f.iter() {
                let a = p[axis] * w;
                t.push(a.cos());
                t.push(a.sin());
            }
        }
    }
    t
}

/// `resolution_noise_scale` of a `height` x `width` image.
pub fn noise_scale(height: usize, width: usize) -> f64 {
    let tokens = height.div_ceil(PATCH) * width.div_ceil(PATCH);
    (tokens as f64 / 64.0).sqrt().min(16.0)
}

fn snr_shift(shift: f32, v: f32) -> f32 {
    if shift == 1.0 { v } else { shift * v / (1.0 + (shift - 1.0) * v) }
}

fn inverse_snr_shift(shift: f32, v: f32) -> f32 {
    if shift == 1.0 { v } else { v / (shift - (shift - 1.0) * v) }
}

/// The time the model reads at `sigma`: `process_timestep(model_sampling.timestep(sigma))`, which is 1 - sigma up to
/// float32 rounding.
pub fn model_time(shift: f64, sigma: f32) -> f32 {
    let shift = shift as f32;
    let timestep = (1.0 - inverse_snr_shift(shift, sigma)) * 1000.0;
    1.0 - snr_shift(shift, 1.0 - timestep / 1000.0)
}

/// mmdit's `timestep_embedding` of `t` (no scaling): cosines, then sines.
fn timestep_embedding(t: f32) -> Vec<f32> {
    let half = TDIM / 2;
    let neg_log = -(10000f64.ln()) as f32;
    let mut e = vec![0f32; TDIM];
    for i in 0..half {
        let a = t * (neg_log * i as f32 / half as f32).exp();
        e[i] = a.cos();
        e[half + i] = a.sin();
    }
    e
}

/// `out[rows, n] = a[rows, k] @ w[n, k]^T + bias`, bf16.
#[allow(clippy::too_many_arguments)]
fn linear(a: PM, w: PM, b: PM, out: PM, rows: usize, n: usize, kk: usize, st: &Stream) -> Result<()> {
    k!(gpu_gemm(a, kk as i64, w, kk as i64, out, n as i64, b, rows as i64, n as i64, kk as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))
}

/// The keys of a `VisionEmbeddings`' weights, as `globals_layout` and `vision_layout` name them.
fn embedding_layout(st: &SafeTensors, g: &mut Layout, name: impl Fn(&str) -> String) -> Result<()> {
    for (key, n) in [("pe_w", "patch_embedding.weight"), ("pe_b", "patch_embedding.bias"), ("de_w", "dense_embedding.weight"),
        ("de_b", "dense_embedding.bias")]
    {
        g.bf16(st, key, &[name(n)])?;
    }
    g.bytes("pe_freqs", f32_bytes(&freqs(10_000.0, PE / 2)).to_vec());
    Ok(())
}

/// `VisionEmbeddings` on f32 [3, height, width] pixels at whole tokens: the 16-pixel patch conv, GELU and 2D RoPE, then
/// the 2x2 conv, into bf16 [tokens, F] at `out`. `g` finds its weights.
fn patch_embed(g: &dyn Fn(&str) -> PM, x: P, height: usize, width: usize, out: PM, st: &Stream) -> Result<()> {
    let (gh, gw) = (height / 16, width / 16);
    let l = (height / PATCH) * (width / PATCH);
    let tok = DevBuf::alloc_async(gh * gw * 768 * 2, st)?;
    let pe = DevBuf::alloc_async(gh * gw * PE * 2, st)?;
    let sd = DevBuf::alloc_async(l * 4 * PE * 2, st)?;
    k!(gpu_patchify(x, tok.ptr(), 3, height as i32, width as i32, 16, s(st)))?;
    linear(tok.ptr(), g("pe_w"), g("pe_b"), pe.ptr(), gh * gw, PE, 768, st)?;
    k!(gpu_sn_gelu_rope2d(pe.ptr(), gh as i32, gw as i32, PE as i32, g("pe_freqs"), s(st)))?;
    k!(gpu_sn_space_to_depth2(pe.ptr(), sd.ptr(), gh as i32, gw as i32, PE as i32, s(st)))?;
    linear(sd.ptr(), g("de_w"), g("de_b"), out, l, F, 4 * PE, st)
}

/// What the generation tower needs of a prompt: each layer's keys and values, and where the image goes after it.
pub struct Prefix {
    pub len: usize,
    /// The text position of the image tokens: one past the prompt's last.
    pub time: usize,
    /// bf16 [layers][2][len][KV]: keys, then values, after the norms and RoPE
    kv: DevBuf,
}

// Only reached through the Generator, which one thread holds at a time.
unsafe impl Sync for Prefix {}

impl Prefix {
    pub fn bytes(len: usize) -> usize {
        LAYERS * 2 * len * KV * 2
    }

    /// The keys and values on the host, bf16 [layers][2][len][KV], for checks against ComfyUI's.
    pub fn to_host(&self, st: &Stream) -> Result<Vec<u8>> {
        let mut v = vec![0u8; Self::bytes(self.len)];
        self.kv.download(0, &mut v, st)?;
        st.sync()?;
        Ok(v)
    }
}

/// The text tower, streamed from pinned RAM a layer at a time, once per prompt.
pub struct TextTower {
    st: SafeTensors,
    ws: WeightSet,
    pre: &'static str,
    tok: Tokenizer,
    /// The patch embedding of reference pictures (`vision_model`), in VRAM.
    vision: DevBuf,
    vl: Layout,
}

fn layer_layout(st: &SafeTensors, pre: &str, i: usize, gen: bool) -> Result<Layout> {
    let g = if gen { "_mot_gen" } else { "" };
    let n = |x: &str| format!("{pre}language_model.model.layers.{i}.{x}");
    let mut l = Layout::new();
    let attn = |p: &str| n(&format!("self_attn.{p}{g}.weight"));
    let attn_s = |p: &str| n(&format!("self_attn.{p}{g}.weight_scale"));
    let mlp = |p: &str| n(&format!("mlp{g}.{p}.weight"));
    let mlp_s = |p: &str| n(&format!("mlp{g}.{p}.weight_scale"));
    l.raw(st, "wqkv", &[attn("q_proj"), attn("k_proj"), attn("v_proj")])?;
    l.raw(st, "sqkv", &[attn_s("q_proj"), attn_s("k_proj"), attn_s("v_proj")])?;
    l.raw(st, "wo", &[attn("o_proj")])?;
    l.raw(st, "so", &[attn_s("o_proj")])?;
    l.raw(st, "wgu", &[mlp("gate_proj"), mlp("up_proj")])?;
    l.raw(st, "sgu", &[mlp_s("gate_proj"), mlp_s("up_proj")])?;
    l.raw(st, "wd", &[mlp("down_proj")])?;
    l.raw(st, "sd", &[mlp_s("down_proj")])?;
    l.f32(st, "ln1", &n(&format!("input_layernorm{g}.weight")), 0.0)?;
    l.f32(st, "ln2", &n(&format!("post_attention_layernorm{g}.weight")), 0.0)?;
    for (key, name) in [("qn", "q_norm"), ("qnh", "q_norm_hw"), ("kn", "k_norm"), ("knh", "k_norm_hw")] {
        l.f32(st, key, &n(&format!("self_attn.{name}{g}.weight")), 0.0)?;
    }
    Ok(l)
}

/// Refuses checkpoints this engine can't run: only int8 ConvRot weights, with 256-wide rotation groups.
pub fn check(st: &SafeTensors, pre: &str) -> Result<()> {
    for name in ["language_model.model.layers.0.self_attn.q_proj.weight", "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "language_model.model.embed_tokens.weight"]
    {
        let full = format!("{pre}{name}");
        if st.info(&full)?.dtype != crate::st::Dt::I8 || !st.has(&format!("{}_scale", full)) {
            return Err(crate::lora::Unsupported("SenseNova checkpoints other than the int8 ConvRot release".into()).into());
        }
    }
    let layers = (0..).take_while(|i| st.has(&format!("{pre}language_model.model.layers.{i}.input_layernorm.weight"))).count();
    if layers != LAYERS || st.info(&format!("{pre}language_model.model.layers.0.input_layernorm.weight"))?.shape != [F] {
        bail!("a SenseNova checkpoint of {layers} layers (expected {LAYERS} of width {F})");
    }
    Ok(())
}

impl TextTower {
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let pre = key_prefix(|k| st.has(k)).context("not a SenseNova checkpoint")?;
        check(&st, pre)?;
        let layouts = (0..LAYERS).map(|i| layer_layout(&st, pre, i, false)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, 0, 2);
        let ws = WeightSet::load("sensenova text tower", &st, layouts, p, threads)?;
        ws.once_per_prompt();
        let mut vl = Layout::new();
        embedding_layout(&st, &mut vl, |n| format!("{pre}vision_model.embeddings.{n}"))?;
        let mut host = vec![0u8; vl.size];
        vl.pack(&st, &mut host)?;
        let vision = DevBuf::alloc(vl.size)?;
        let s0 = Stream::new()?;
        vision.upload(0, &host, &s0)?;
        s0.sync()?;
        Ok(TextTower { st, ws, pre, tok: Tokenizer::sensenova()?, vision, vl })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Run the prompt `ids` through the tower and keep every layer's keys and values for the image. `refs` are the
    /// pictures of its image blocks (see [`with_images`]), in order. Text attends causally; an image block's tokens
    /// also see the rest of their block (`block_causal_mask`).
    pub fn encode(&mut self, ids: &[u32], refs: &[Reference], st: &Stream) -> Result<Prefix> {
        let l = ids.len();
        if l == 0 {
            bail!("an empty SenseNova prompt");
        }
        let runs = image_runs(ids);
        let grids: Vec<(usize, usize)> = refs.iter().map(Reference::grid).collect();
        if runs.len() != refs.len() || runs.iter().zip(&grids).any(|(r, g)| r.1 - r.0 != g.0 * g.1) {
            bail!("the prompt's image blocks don't match its {} reference pictures", refs.len());
        }
        let (pos, time) = positions(ids, &runs, &grids);
        let pre = self.pre;
        // embedding rows: int8 rows and their scales gathered on the host, dequantized and un-rotated on the device
        let emb = self.st.bytes(&format!("{pre}language_model.model.embed_tokens.weight"))?;
        let emb_s = self.st.bytes(&format!("{pre}language_model.model.embed_tokens.weight_scale"))?;
        let vocab = emb_s.len() / 4;
        let (mut q, mut sc) = (Vec::with_capacity(l * F), Vec::with_capacity(l * 4));
        for &t in ids {
            let t = t as usize;
            if t >= vocab {
                bail!("token {t} past the vocabulary of {vocab}");
            }
            q.extend_from_slice(&emb[t * F..(t + 1) * F]);
            sc.extend_from_slice(&emb_s[t * 4..t * 4 + 4]);
        }
        let q_d = DevBuf::alloc_async(q.len(), st)?;
        q_d.upload(0, &q, st)?;
        let s_d = DevBuf::alloc_async(sc.len(), st)?;
        s_d.upload(0, &sc, st)?;
        let x = DevBuf::alloc_async(l * F * 4, st)?;
        k!(gpu_embed_dequant(q_d.ptr(), s_d.ptr(), x.ptr(), l as i32, F as i32, 1, s(st)))?;
        // the pictures' tokens in place of their image tokens' embeddings, from bf16 as ComfyUI's compute dtype makes them
        for (r, &(a, b)) in refs.iter().zip(&runs) {
            let (gh, gw) = r.grid();
            let px = r.pixels();
            let px_d = DevBuf::alloc_async(px.len() * 4, st)?;
            px_d.upload(0, &f32_bytes(&px), st)?;
            let emb = DevBuf::alloc_async((b - a) * F * 2, st)?;
            patch_embed(&|key| self.vision.at(self.vl.off(key)), px_d.ptr(), gh * PATCH, gw * PATCH, emb.ptr(), st)?;
            k!(gpu_bf16_to_f32(emb.ptr(), off(x.ptr(), a * F * 4), ((b - a) * F) as i64, s(st)))?;
        }

        let rope = rope_table(pos.into_iter());
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &f32_bytes(&rope), st)?;

        let kv = DevBuf::alloc(Prefix::bytes(l))?;
        let xr = DevBuf::alloc_async(l * FF * 4, st)?;
        let wf = DevBuf::alloc_async(2 * FF * F * 4, st)?;
        let qkv = DevBuf::alloc_async(l * QKV * 4, st)?;
        let attn = DevBuf::alloc_async(l * F * 4, st)?;
        let gu = DevBuf::alloc_async(l * 2 * FF * 4, st)?;
        let kc = DevBuf::alloc_async(l * KV * 4, st)?;
        let (m, xp) = (l as i32, x.ptr());
        for i in 0..LAYERS {
            let base = self.ws.acquire(i, st)?;
            let lay = &self.ws.layouts[i];
            let w = |key: &str| off(base, lay.off(key));
            // ComfyUI runs this tower with its weights dequantized; x @ ((q * s) H)^T == (x H) @ (q * s)^T, so the
            // activation is rotated instead of the weight
            let lin = |xr: PM, wq: &str, sc: &str, n: usize, kk: usize, out: PM, beta: f32| -> Result<()> {
                k!(gpu_dequant_i8(w(wq), w(sc), wf.ptr(), n as i64, kk as i64, s(st)))?;
                k!(gpu_gemm(xr, kk as i64, wf.ptr(), kk as i64, out, n as i64, null(), l as i64, n as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, beta, s(st)))
            };
            k!(gpu_rotate_rows_f32(PRO_RMS, xp, F as i64, 0, w("ln1"), EPS, xr.ptr(), m, F as i32, s(st)))?;
            lin(xr.ptr(), "wqkv", "sqkv", QKV, F, qkv.ptr(), 0.0)?;
            k!(gpu_sn_qk(DT_F32, qkv.ptr(), QKV as i64, 0, F as i32, HQ as i32, HK as i32, w("qn"), w("qnh"), w("kn"), w("knh"), EPS, rope_d.ptr(),
                m, s(st)))?;
            for (j, col) in [(0, F), (1, F + KV)] {
                memcpy_2d_d2d(kc.ptr(), KV * 4, off(qkv.ptr(), col * 4), QKV * 4, KV * 4, l, st)?;
                k!(gpu_f32_to_bf16(kc.ptr(), off(kv.ptr(), (i * 2 + j) * l * KV * 2), (l * KV) as i64, s(st)))?;
            }
            if i + 1 == LAYERS {
                // the last layer's output feeds nothing the image reads
                self.ws.release(i, st)?;
                break;
            }
            self.attend(&qkv, &attn, l, &runs, st)?;
            k!(gpu_rotate_rows_f32(PRO_PLAIN, attn.ptr(), F as i64, 0, null(), 0.0, xr.ptr(), m, F as i32, s(st)))?;
            lin(xr.ptr(), "wo", "so", F, F, xp, 1.0)?;
            k!(gpu_rotate_rows_f32(PRO_RMS, xp, F as i64, 0, w("ln2"), EPS, xr.ptr(), m, F as i32, s(st)))?;
            lin(xr.ptr(), "wgu", "sgu", 2 * FF, F, gu.ptr(), 0.0)?;
            k!(gpu_rotate_rows_f32(PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, FF as i64, null(), 0.0, xr.ptr(), m, FF as i32, s(st)))?;
            lin(xr.ptr(), "wd", "sd", F, FF, xp, 1.0)?;
            self.ws.release(i, st)?;
        }
        Ok(Prefix { len: l, time, kv })
    }

    /// The prompt's attention, f32 [l, QKV] to [l, F]: text rows causally, each image block's rows over every key up
    /// to its end. Image blocks run as bf16 flash attention, the precision ComfyUI runs the whole tower in.
    fn attend(&self, qkv: &DevBuf, attn: &DevBuf, l: usize, runs: &[(usize, usize)], st: &Stream) -> Result<()> {
        let scale = 1.0 / (HD as f32).sqrt();
        let text = |a: usize, b: usize| -> Result<()> {
            if a == b {
                return Ok(());
            }
            // row a + i sees keys up to a + i
            k!(gpu_attn_small(DT_F32, off(qkv.ptr(), a * QKV * 4), QKV as i64, off(qkv.ptr(), F * 4), off(qkv.ptr(), (F + KV) * 4), QKV as i64,
                HQ as i32, HK as i32, HD as i32, (b - a) as i32, b as i32, off(attn.ptr(), a * F * 4), F as i64, scale, (a + 1) as i32, s(st)))
        };
        let Some(&(_, last)) = runs.last() else { return text(0, l) };
        let qb = DevBuf::alloc_async(last * QKV * 2, st)?;
        k!(gpu_f32_to_bf16(qkv.ptr(), qb.ptr(), (last * QKV) as i64, s(st)))?;
        let mut done = 0;
        for &(a, b) in runs {
            text(done, a)?;
            let rows = b - a;
            let o = DevBuf::alloc_async(rows * F * 2, st)?;
            let lse = DevBuf::alloc_async(rows * HQ * 4, st)?;
            let (q, kk, v) = (off(qb.ptr(), a * QKV * 2), off(qb.ptr(), F * 2), off(qb.ptr(), (F + KV) * 2));
            let (qr, hd) = (QKV as i64, HD as i64);
            k!(gpu_flash_attn(q, kk, v, o.ptr(), lse.ptr(), 1, rows as i32, b as i32, HQ as i32, HK as i32, 0, qr, hd, 0, qr, hd, 0, qr, hd,
                0, F as i64, hd, scale, 0, s(st)))?;
            k!(gpu_bf16_to_f32(o.ptr(), off(attn.ptr(), a * F * 4), (rows * F) as i64, s(st)))?;
            done = b;
        }
        text(done, l)
    }
}

/// The generation tower and the layers around it: the pixel patch embedding, the time and noise-scale embeddings, the
/// last norm and the pixel head.
pub struct Gen {
    globals: DevBuf,
    gl: Layout,
    pub blocks: WeightSet,
    pub int8_attn: bool,
    pub attn_variant: i32,
    /// The RoPE table of the latest (prefix time, token rows, token columns).
    rope: std::cell::RefCell<Option<((usize, usize, usize), DevBuf)>>,
}

pub fn globals_layout(st: &SafeTensors, pre: &str) -> Result<Layout> {
    let fm = |x: &str| format!("{pre}fm_modules.{x}");
    let mut g = Layout::new();
    embedding_layout(st, &mut g, |n| fm(&format!("vision_model_mot_gen.embeddings.{n}")))?;
    for (key, name) in [("t1_w", "timestep_embedder.mlp.0.weight"), ("t1_b", "timestep_embedder.mlp.0.bias"), ("t2_w", "timestep_embedder.mlp.2.weight"),
        ("t2_b", "timestep_embedder.mlp.2.bias"), ("n1_w", "noise_scale_embedder.mlp.0.weight"), ("n1_b", "noise_scale_embedder.mlp.0.bias"),
        ("n2_w", "noise_scale_embedder.mlp.2.weight"), ("n2_b", "noise_scale_embedder.mlp.2.bias"), ("c1_b", "fm_head.conv1.bias"),
        ("c2_b", "fm_head.conv2.bias")]
    {
        g.bf16(st, key, &[fm(name)])?;
    }
    g.f32(st, "norm", &format!("{pre}language_model.model.norm_mot_gen.weight"), 0.0)?;
    g.bytes("c1_w", crate::vae::conv_krsc(st, &fm("fm_head.conv1.weight"), 0)?.0);
    g.bytes("c2_w", crate::vae::conv_krsc(st, &fm("fm_head.conv2.weight"), 0)?.0);
    // the residual GEMMs' row scale: the block adds its outputs as they are
    g.bytes("ones", bf16_bytes(&[1.0; F]).to_vec());
    Ok(g)
}

/// Rough activation bytes of one image's evaluation at `l` tokens against a prefix of `prefix` rows.
pub fn step_activation_bytes(l: usize, prefix: usize) -> usize {
    l * (F * 2 + QKV * 2 + F * 2 + 2 * FF * 2 + FF + 16) + 2 * (prefix + l) * KV * 2 + 4 * l * (PE * 2 * 3 + 768 * 2) + (256 << 20)
}

impl Gen {
    pub fn load(st: &SafeTensors, pre: &str, block_budget: usize, slots: usize, threads: usize, s0: &Stream) -> Result<Self> {
        check(st, pre)?;
        let gl = globals_layout(st, pre)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let layouts = (0..LAYERS).map(|i| layer_layout(st, pre, i, true)).collect::<Result<Vec<_>>>()?;
        let p = plan(&layouts, block_budget, slots);
        let blocks = WeightSet::load("sensenova generation tower", st, layouts, p, threads)?;
        Ok(Gen { globals, gl, blocks, int8_attn: true, attn_variant: 0, rope: Default::default() })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    pub fn stores(&mut self) -> Vec<Store<'_>> {
        vec![Store::Set(&mut self.blocks), Store::Buf(&self.globals, &self.gl)]
    }

    /// `out[rows, n] = a[rows, k] @ w[n, k]^T + bias`, bf16.
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, a: PM, w: &str, b: &str, out: PM, rows: usize, n: usize, kk: usize, st: &Stream) -> Result<()> {
        linear(a, self.g(w), self.g(b), out, rows, n, kk, st)
    }

    /// One of the `TimestepEmbedder`s at `v`: bf16 [F].
    fn embed_time(&self, v: f32, which: &str, st: &Stream) -> Result<DevBuf> {
        let e = DevBuf::alloc_async(TDIM * 2, st)?;
        e.upload(0, &bf16_bytes(&timestep_embedding(v)), st)?;
        let h = DevBuf::alloc_async(F * 2, st)?;
        self.linear(e.ptr(), &format!("{which}1_w"), &format!("{which}1_b"), h.ptr(), 1, F, TDIM, st)?;
        k!(gpu_prologue_rows(DT_BF16, PRO_SILU, h.ptr(), F as i64, null(), 0, 0, null(), 0.0, null(), null(), h.ptr(), F as i64, 1, F as i32, s(st)))?;
        let out = DevBuf::alloc_async(F * 2, st)?;
        self.linear(h.ptr(), &format!("{which}2_w"), &format!("{which}2_b"), out.ptr(), 1, F, F, st)?;
        Ok(out)
    }

    /// The image tokens' RoPE table: at the prefix's time, by row and column.
    fn rope(&self, time: usize, th: usize, tw: usize, st: &Stream) -> Result<P> {
        let key = (time, th, tw);
        let mut cache = self.rope.borrow_mut();
        if cache.as_ref().is_none_or(|(k, _)| *k != key) {
            let table = rope_table((0..th * tw).map(|i| [time as f32, (i / tw) as f32, (i % tw) as f32]));
            let d = DevBuf::alloc(table.len() * 4)?;
            d.upload(0, &f32_bytes(&table), st)?;
            *cache = Some((key, d));
        }
        Ok(cache.as_ref().unwrap().1.ptr() as P)
    }

    /// One evaluation of `batch` images (f32 [batch, 3, height, width], -1 to 1) that share `pf`, at `sigma` of a run
    /// with flow shift `shift`: writes what the sampler takes as denoised to `den`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, x: &DevBuf, batch: usize, height: usize, width: usize, pf: &Prefix, sigma: f32, shift: f64, den: &DevBuf,
        st: &Stream) -> Result<()> {
        if height % PATCH != 0 || width % PATCH != 0 {
            bail!("SenseNova images are multiples of {PATCH} pixels");
        }
        let (th, tw) = (height / PATCH, width / PATCH);
        let l = th * tw;
        let (rows, lk, lp) = (batch * l, pf.len + l, pf.len);
        let plane = 3 * height * width;

        // the time, as the model reads it, and the noise scale of this size, embedded and summed
        let t = model_time(shift, sigma);
        let temb = self.embed_time(t, "t", st)?;
        let nemb = self.embed_time((noise_scale(height, width) / 16.0) as f32, "n", st)?;
        k!(gpu_add_vec_bf16(temb.ptr(), temb.ptr(), nemb.ptr(), 0, F as i32, s(st)))?;

        // pixel patches -> GELU and 2D RoPE -> 2x2 conv: one token per 32 x 32 pixels, plus the embedding
        let h = DevBuf::alloc_async(rows * F * 2, st)?;
        for i in 0..batch {
            patch_embed(&|key| self.g(key), off(x.ptr(), i * plane * 4), height, width, off(h.ptr(), i * l * F * 2), st)?;
        }
        k!(gpu_add_vec_rows_bf16(h.ptr(), F as i64, h.ptr(), F as i64, temb.ptr(), rows as i32, F as i32, s(st)))?;
        let rope = self.rope(pf.time, th, tw, st)?;

        let a8 = DevBuf::alloc_async(rows * FF, st)?;
        let asc = DevBuf::alloc_async(rows * 4, st)?;
        let qkv = DevBuf::alloc_async(rows * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(rows * F * 2, st)?;
        // per image: the prefix's keys (values), then its own
        let kc = DevBuf::alloc_async(batch * lk * KV * 2, st)?;
        let vc = DevBuf::alloc_async(batch * lk * KV * 2, st)?;
        let ws = if self.int8_attn {
            Some(DevBuf::alloc_async(unsafe { crate::ffi::gpu_int8_attn_workspace(batch as i32, HQ as i32, HK as i32, l as i32, lk as i32) } as usize, st)?)
        } else {
            None
        };
        let (fi, mi) = (F as i64, rows as i32);
        for b in 0..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let w = |key: &str| off(base, lay.off(key));
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, h.ptr(), fi, null(), 0, 0, w("ln1"), EPS, null(), null(), a8.ptr(), asc.ptr(), mi, F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm(a8.ptr(), w("wqkv"), asc.ptr(), w("sqkv"), null(), qkv.ptr(), rows as i64, QKV as i64, fi, DT_BF16, s(st)))?;
            for i in 0..batch {
                k!(gpu_sn_qk(DT_BF16, off(qkv.ptr(), i * l * QKV * 2), QKV as i64, 0, F as i32, HQ as i32, HK as i32, w("qn"), w("qnh"), w("kn"), w("knh"),
                    EPS, rope, l as i32, s(st)))?;
                let slot = off(pf.kv.ptr(), b * 2 * lp * KV * 2);
                for (j, (cat, col)) in [(&kc, F), (&vc, F + KV)].into_iter().enumerate() {
                    let dst = off(cat.ptr(), i * lk * KV * 2);
                    memcpy_d2d(dst, off(slot, j * lp * KV * 2), lp * KV * 2, st)?;
                    memcpy_2d_d2d(off(dst, lp * KV * 2), KV * 2, off(qkv.ptr(), (i * l * QKV + col) * 2), QKV * 2, KV * 2, l, st)?;
                }
            }
            let scale = 1.0 / (HD as f32).sqrt();
            let (q_bs, k_bs, o_bs) = ((l * QKV) as i64, (lk * KV) as i64, (l * F) as i64);
            let hs = HD as i64;
            match &ws {
                Some(ws) => k!(gpu_int8_attn(qkv.ptr(), kc.ptr(), vc.ptr(), attn.ptr(), ws.ptr(), batch as i32, l as i32, lk as i32, HQ as i32, HK as i32,
                    q_bs, QKV as i64, hs, k_bs, KV as i64, hs, k_bs, KV as i64, hs, o_bs, fi, hs, scale, s(st)))?,
                None => {
                    let lse = DevBuf::alloc_async(HQ * rows * 4, st)?;
                    k!(gpu_flash_attn(qkv.ptr(), kc.ptr(), vc.ptr(), attn.ptr(), lse.ptr(), batch as i32, l as i32, lk as i32, HQ as i32, HK as i32,
                        q_bs, QKV as i64, hs, k_bs, KV as i64, hs, k_bs, KV as i64, hs, o_bs, fi, hs, scale, self.attn_variant, s(st)))?
                }
            }
            k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), fi, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), mi, F as i32, 1, s(st)))?;
            k!(gpu_int8_gemm_residual(a8.ptr(), w("wo"), asc.ptr(), w("so"), null(), self.g("ones"), h.ptr(), h.ptr(), rows as i64, fi, fi, DT_BF16, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, h.ptr(), fi, null(), 0, 0, w("ln2"), EPS, null(), null(), a8.ptr(), asc.ptr(), mi, F as i32, 1, s(st)))?;
            let gu = DevBuf::alloc_async(rows * 2 * FF * 2, st)?;
            k!(gpu_int8_gemm(a8.ptr(), w("wgu"), asc.ptr(), w("sgu"), null(), gu.ptr(), rows as i64, (2 * FF) as i64, fi, DT_BF16, s(st)))?;
            k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, null(), 0, FF as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), mi,
                FF as i32, 1, s(st)))?;
            drop(gu);
            k!(gpu_int8_gemm_residual(a8.ptr(), w("wd"), asc.ptr(), w("sd"), null(), self.g("ones"), h.ptr(), h.ptr(), rows as i64, fi, FF as i64, DT_BF16, s(st)))?;
            self.blocks.release(b, st)?;
        }
        drop((a8, asc, qkv, attn, kc, vc, ws));

        // the last norm, then the pixel head: PixelShuffle(2), conv, GELU, PixelShuffle(2), conv, PixelShuffle(8)
        k!(gpu_prologue_rows(DT_BF16, PRO_RMS, h.ptr(), fi, null(), 0, 0, self.g("norm"), EPS, null(), null(), h.ptr(), fi, mi, F as i32, s(st)))?;
        let a = DevBuf::alloc_async(4 * l * PE * 2, st)?;
        let c = DevBuf::alloc_async(4 * l * PE * 2, st)?;
        let denom = (1.0 - t).max(0.02);
        for i in 0..batch {
            let (h2, w2) = (2 * th, 2 * tw);
            k!(gpu_sn_pixel_shuffle(off(h.ptr(), i * l * F * 2), a.ptr(), th as i32, tw as i32, PE as i32, 2, s(st)))?;
            k!(gpu_conv2d(a.ptr(), self.g("c1_w"), self.g("c1_b"), c.ptr(), 1, h2 as i32, w2 as i32, PE as i32, PE as i32, 3, s(st)))?;
            k!(gpu_sn_gelu(c.ptr(), (4 * l * PE) as i64, s(st)))?;
            k!(gpu_sn_pixel_shuffle(c.ptr(), a.ptr(), h2 as i32, w2 as i32, (PE / 4) as i32, 2, s(st)))?;
            k!(gpu_conv2d(a.ptr(), self.g("c2_w"), self.g("c2_b"), c.ptr(), 1, (2 * h2) as i32, (2 * w2) as i32, (PE / 4) as i32, 192, 3, s(st)))?;
            k!(gpu_sn_denoise(c.ptr(), off(x.ptr(), i * plane * 4), off(den.ptr(), i * plane * 4), height as i32, width as i32, sigma, denom, s(st)))?;
        }
        Ok(())
    }
}

/// Pixels as the sampler leaves them (-1 to 1, [3, height, width]) to RGB8, as ComfyUI's pixel-space "VAE" decodes
/// them: `(x + 1) / 2`, clamped.
pub fn to_rgb8(latent: &[f32], width: usize, height: usize) -> Vec<u8> {
    let plane = width * height;
    let mut out = vec![0u8; plane * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        for (c, v) in px.iter_mut().enumerate() {
            let f = ((latent[c * plane + i] + 1.0) / 2.0).clamp(0.0, 1.0);
            *v = (f * 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_follow_comfy() {
        let tok = Tokenizer::sensenova().unwrap();
        // the unconditional prompt: its <img> is SenseNova's added token
        assert_eq!(tokens(&tok, "").unwrap(), [151644, 872, 198, 151645, 198, 151644, 77091, 198, IMAGE_START]);
        // ComfyUI's ids for this prompt, from tools/sensenova_ref.py --tokens-only
        #[rustfmt::skip]
        let comfy: &[u32] = &[
            151644, 8948, 198, 2610, 525, 458, 2168, 9471, 323, 15664, 17847, 429, 29257, 30769, 323, 51435, 1196, 7385, 382, 2610, 1824, 1378, 19777, 1447,
            16, 13, 21149, 14562, 510, 2679, 279, 3383, 7460, 32711, 11, 498, 27732, 1191, 448, 264, 220, 151667, 151668, 2504, 13, 10224, 678, 32711,
            4766, 279, 2504, 1667, 14396, 1467, 13, 9319, 4183, 2924, 894, 2168, 9492, 13, 13655, 432, 13276, 323, 5961, 5390, 369, 17387, 279, 1590,
            2168, 382, 17, 13, 11581, 12, 38687, 14562, 510, 2679, 902, 32711, 374, 4362, 11, 5961, 8193, 279, 1590, 2168, 382, 6262, 20768, 1447,
            32, 13, 2918, 4686, 12, 1906, 23470, 510, 12, 19813, 264, 1550, 22092, 2168, 3118, 389, 279, 1196, 594, 4008, 624, 12, 29279, 9124,
            31273, 11, 41733, 28137, 11, 323, 79314, 624, 12, 9319, 4183, 19131, 5424, 429, 42461, 476, 2812, 279, 1196, 594, 7385, 382, 33, 13,
            4654, 61053, 510, 12, 5443, 279, 3897, 2168, 1141, 8, 438, 1946, 476, 5785, 369, 17030, 476, 17991, 624, 12, 576, 1102, 646, 387,
            458, 19138, 2168, 476, 264, 501, 2168, 3118, 389, 279, 5785, 1141, 4292, 12, 81306, 678, 54311, 8201, 7241, 20975, 5497, 382, 15415, 22847,
            510, 12, 1752, 894, 9434, 1467, 304, 279, 2168, 11, 1795, 279, 4128, 5189, 369, 279, 22383, 1467, 304, 279, 1196, 594, 4008, 11,
            537, 279, 4128, 315, 279, 9934, 13, 1416, 902, 4128, 374, 5189, 11, 990, 279, 1196, 594, 1946, 4128, 13, 151645, 198, 151644, 872,
            198, 2152, 12677, 11, 50231, 11, 264, 9798, 326, 57909, 389, 264, 55517, 13648, 518, 42984, 151645, 198, 151644, 77091, 198, 151667, 271, 151668,
            271, 151670,
        ];
        assert_eq!(tokens(&tok, "no humans, scenery, a stone lighthouse on a rocky coast at sunset").unwrap(), comfy);
    }

    #[test]
    fn edits_place_pictures_as_comfy_does() {
        // ComfyUI's condition_input_ids and thw_indexes for "a lake" with a 1x2 and a 2x1 picture
        let tok = Tokenizer::sensenova().unwrap();
        let grids = [(1, 2), (2, 1)];
        let ids = with_images(&tokens(&tok, "a lake").unwrap(), &grids, false);
        assert_eq!(ids.len(), 271);
        assert_eq!(ids[ids.len() - 30..], [1906, 12, 16, 25, 151670, 151669, 151669, 151671, 198, 1906, 12, 17, 25, 151670, 151669, 151669, 151671, 198,
            64, 21800, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 151670]);
        let runs = image_runs(&ids);
        let (pos, time) = positions(&ids, &runs, &grids);
        assert_eq!(time, 269);
        let tail: Vec<[f32; 3]> = pos[pos.len() - 30..].to_vec();
        let t = [241, 242, 243, 244, 245, 246, 246, 247, 248, 249, 250, 251, 252, 253, 254, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268];
        let (h, w) = ([15], [6]);
        for (i, p) in tail.iter().enumerate() {
            assert_eq!(*p, [t[i] as f32, h.contains(&i) as u8 as f32, w.contains(&i) as u8 as f32], "row {i}");
        }
        let neg = with_images(&tokens(&tok, "blurry").unwrap(), &grids, true);
        assert_eq!(neg, [151644, 872, 198, 151670, 151669, 151669, 151671, 151670, 151669, 151669, 151671, 151645, 198, 151644, 77091, 198, 151670]);
        assert_eq!(positions(&neg, &image_runs(&neg), &grids).1, 15);
    }

    #[test]
    fn user_text_has_no_special_tokens() {
        let tok = Tokenizer::sensenova().unwrap();
        let ids = tokens(&tok, "<img> <|im_end|>").unwrap();
        // one <img> and the template's two <|im_end|>, none from the text
        assert_eq!(ids.iter().filter(|&&t| t == IMAGE_START).count(), 1);
        assert_eq!(ids.iter().filter(|&&t| t == 151645).count(), 2);
    }

    #[test]
    fn embedding_words_split_after_whitespace() {
        assert_eq!(embedding_pieces("a embedding:x b"), ["a ", "embedding:x b"]);
        assert_eq!(embedding_pieces("embedding:x"), ["embedding:x"]);
        assert_eq!(embedding_pieces("aembedding:x"), ["aembedding:x"]);
    }

    #[test]
    fn time_is_one_minus_sigma() {
        for s in [1.0f32, 0.75, 0.3, 0.01] {
            assert!((model_time(3.0, s) - (1.0 - s)).abs() < 1e-5, "{s}");
        }
        assert_eq!(noise_scale(1024, 1024), 4.0);
        assert_eq!(noise_scale(2048, 2048), 8.0);
    }
}
