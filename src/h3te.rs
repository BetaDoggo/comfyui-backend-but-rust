//! MiniMax H3 text encoder: Qwen3-VL 32B language model cut to 50 layers (nvfp4 AWQ weights), fp32
//! activations like ComfyUI's full_precision_mm path. Conditioning is the hidden state after the
//! last layer, with no final norm. Weights are decoded to f32 for each use, so the encoder only
//! needs VRAM for one layer and streams the rest. Pictures come from the vision tower in the same file: their tokens
//! take the place of `<|image_pad|>` ids, with MRoPE positions and DeepStack features, as in Qwen3-VL.

use crate::cuda::{memcpy_2d_d2d, memcpy_2d_d2h, memcpy_d2d, memcpy_h2d, DevBuf, PinnedBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, P, PM, PRO_RMS, PRO_SWIGLU};
use crate::offload::{Layout, Plan, WeightSet};
use crate::st::SafeTensors;
use crate::te::{position_ids, rope_table};
use crate::vision::Embeds;
use anyhow::Result;
use std::path::Path;
use std::sync::{Arc, Weak};

pub const D: usize = 5120;
const HQ: usize = 64;
const HK: usize = 8;
const HD: usize = 128;
const FF: usize = 25600;
const QKV: usize = (HQ + 2 * HK) * HD; // 10240
const LAYERS: usize = 50;
const EPS: f32 = 1e-6;
const PRE: &str = "model.";
pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
/// stands for one picture's tokens in the ids
pub const IMAGE_PAD: u32 = 151655;
const MODULES: [&str; 7] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

/// Bytes of one token's keys and values in a layer.
const KV: usize = 2 * HK * HD * 4;
/// A prompt of this many tokens keeps its keys and values for the next one. Shorter ones cost little more than
/// reading the weights, however many of their tokens run.
const PREFIX_MIN: usize = 256;
/// The most RAM a kept prompt may take, about 2500 tokens.
const PREFIX_MAX_BYTES: usize = 1 << 30;

/// The last long prompt, as the tokens of a prompt that starts the same way read it.
struct Prefix {
    /// its ids, pictures grown to their tokens
    ids: Vec<u32>,
    /// its pictures: the same picture is the same embeddings, not only the same pixels
    images: Vec<Weak<Embeds>>,
    /// every token's keys after their norm and rope, then its values, f32 `[LAYERS][ids.len()][KV]`
    kv: PinnedBuf,
    /// the final hidden states, f32 `[ids.len()][D]`
    hidden: Vec<u8>,
}

pub struct TextEncoder {
    st: Arc<SafeTensors>,
    ws: WeightSet,
    scales: Vec<[f32; 7]>,
    prefix: Option<Prefix>,
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
        ws.once_per_prompt();
        let mut scales = Vec::new();
        for i in 0..LAYERS {
            let mut v = [0.0; 7];
            for (j, m) in MODULES.iter().enumerate() {
                v[j] = st.f32_vec(&format!("{PRE}layers.{i}.{m}.weight_scale_2"), false)?[0];
            }
            scales.push(v);
        }
        Ok(TextEncoder { st, ws, scales, prefix: None })
    }

    /// The ids with each `IMAGE_PAD` grown to its picture's tokens.
    pub fn expand(ids: &[u32], images: &[&Embeds]) -> Vec<u32> {
        let mut out = Vec::with_capacity(ids.len());
        let mut next = images.iter();
        for &t in ids {
            let n = if t == IMAGE_PAD { next.next().map_or(1, |e| e.tokens) } else { 1 };
            out.extend(std::iter::repeat_n(t, n));
        }
        out
    }

    /// How many of `all`'s first tokens the kept prefix answers for: they are its tokens, with the same pictures, and
    /// only text follows them. 0 when that isn't worth it.
    fn kept(&self, all: &[u32], images: &[Arc<Embeds>], spans: &[(usize, usize, (usize, usize))]) -> usize {
        let Some(p) = &self.prefix else { return 0 };
        // the last token always runs, so there is a row to compute
        let mut c = all.iter().zip(&p.ids).take_while(|(a, b)| a == b).count().min(all.len() - 1);
        for (k, &(start, ..)) in spans.iter().enumerate() {
            if !p.images.get(k).is_some_and(|e| std::ptr::eq(e.as_ptr(), Arc::as_ptr(&images[k]))) {
                c = c.min(start);
            }
        }
        if c < PREFIX_MIN || spans.iter().any(|&(start, size, _)| start + size > c) {
            return 0;
        }
        c
    }

    /// Text tokens' embeddings, f32 [ids.len(), 5120].
    fn embed(&self, ids: &[u32], st: &Stream) -> Result<DevBuf> {
        let emb = self.st.bytes(&format!("{PRE}embed_tokens.weight"))?;
        let emb_s = self.st.bytes(&format!("{PRE}embed_tokens.weight_scale"))?;
        let mut q = Vec::with_capacity(ids.len() * D);
        let mut sc = Vec::with_capacity(ids.len() * 4);
        for &t in ids {
            let t = t as usize;
            q.extend_from_slice(&emb[t * D..(t + 1) * D]);
            sc.extend_from_slice(&emb_s[t * 4..t * 4 + 4]);
        }
        let q_d = DevBuf::alloc_async(q.len(), st)?;
        q_d.upload(0, &q, st)?;
        let s_d = DevBuf::alloc_async(sc.len(), st)?;
        s_d.upload(0, &sc, st)?;
        let text = DevBuf::alloc_async(ids.len() * D * 4, st)?;
        // the table is plain int8 here: no ConvRot, unlike the other encoders
        k!(gpu_dequant_i8(q_d.ptr(), s_d.ptr(), text.ptr(), ids.len() as i64, D as i64, s(st)))?;
        Ok(text)
    }

    /// ids -> hidden states f32 [len, 5120], `images` in place of the `IMAGE_PAD` ids in order, where len counts each
    /// picture's tokens ([`Self::expand`]).
    ///
    /// The model is causal, so a token's states don't depend on what follows it. A long prompt, which is one with
    /// pictures, leaves its keys, values and final states behind ([`Prefix`]), and the next prompt that starts with
    /// the same tokens and pictures only runs its own text against them.
    pub fn encode(&mut self, ids: &[u32], images: &[Arc<Embeds>], st: &Stream) -> Result<DevBuf> {
        let pads: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == IMAGE_PAD).map(|(i, _)| i).collect();
        anyhow::ensure!(pads.len() == images.len(), "{} pictures for {} image pads", images.len(), pads.len());
        // the pictures' tokens in place of their pads, and where they went
        let mut spans = Vec::with_capacity(images.len());
        let mut grown = 0;
        for (&pad, img) in pads.iter().zip(images) {
            spans.push((pad + grown, img.tokens, img.grid));
            grown += img.tokens - 1;
        }
        let (l0, l) = (ids.len(), ids.len() + grown);
        let all = Self::expand(ids, &images.iter().map(|e| &**e).collect::<Vec<_>>());
        let kept = self.kept(&all, images, &spans);
        // taken out while the layers run, and put back when they are through
        let prefix = if kept > 0 { self.prefix.take() } else { None };
        // rows `kept..l` are computed
        let rows = l - kept;
        self.ws.replan(Plan { resident: 0, slots: 2 })?;
        let row = D * 4;
        let mut deepstack = Vec::new();
        let x = if kept > 0 {
            self.embed(&all[kept..], st)?
        } else {
            let text = self.embed(ids, st)?;
            let x = DevBuf::alloc_async(l * row, st)?;
            let (mut from, mut to) = (0, 0);
            for (&pad, img) in pads.iter().zip(images) {
                memcpy_d2d(off(x.ptr(), to * row), off(text.ptr(), from * row), (pad - from) * row, st)?;
                to += pad - from;
                memcpy_d2d(off(x.ptr(), to * row), img.merged.ptr(), img.tokens * row, st)?;
                to += img.tokens;
                from = pad + 1;
            }
            memcpy_d2d(off(x.ptr(), to * row), off(text.ptr(), from * row), (l0 - from) * row, st)?;
            // DeepStack: features added at the picture tokens after each of the first layers
            if !images.is_empty() {
                for d in 0..images[0].deepstack.len() {
                    let b = DevBuf::alloc_async(l * row, st)?;
                    b.upload(0, &vec![0u8; l * row], st)?;
                    for (&(start, size, _), img) in spans.iter().zip(images) {
                        memcpy_d2d(off(b.ptr(), start * row), img.deepstack[d].ptr(), size * row, st)?;
                    }
                    deepstack.push(b);
                }
            }
            x
        };
        let rope = rope_table(&position_ids(l, &spans));
        let rope_d = DevBuf::alloc_async(rope.len() * 4, st)?;
        rope_d.upload(0, &crate::st::f32_bytes(&rope), st)?;
        let rope_rows = off(rope_d.ptr(), kept * HD * 4);

        let n = DevBuf::alloc_async(rows * D * 4, st)?;
        let wf = DevBuf::alloc_async(FF * D * 4, st)?;
        let qkv = DevBuf::alloc_async(rows * QKV * 4, st)?;
        let attn = DevBuf::alloc_async(rows * HQ * HD * 4, st)?;
        let gu = DevBuf::alloc_async(rows * 2 * FF * 4, st)?;
        let act = DevBuf::alloc_async(rows * FF * 4, st)?;
        // every token's keys and values, the kept ones first
        let kv = if kept > 0 { Some(DevBuf::alloc_async(l * KV, st)?) } else { None };
        // a long prompt run whole keeps its own
        let keep = if kept == 0 && l >= PREFIX_MIN && LAYERS * l * KV <= PREFIX_MAX_BYTES { PinnedBuf::alloc(LAYERS * l * KV).ok() } else { None };
        let (m, xp, d) = (rows as i32, x.ptr(), D as i64);
        let (keys, values) = (off(qkv.ptr(), HQ * HD * 4), off(qkv.ptr(), (HQ + HK) * HD * 4));
        let mut host = 0.0;

        let prof = std::cell::RefCell::new(crate::prof::Prof::new());
        prof.borrow_mut().mark("setup", st)?;
        for i in 0..LAYERS {
            let t = std::time::Instant::now();
            let base = self.ws.acquire(i, st)?;
            host += t.elapsed().as_secs_f64();
            prof.borrow_mut().mark("stream", st)?;
            let lay = &self.ws.layouts[i];
            let w = |key: &str| off(base, lay.off(key));
            let sc2 = self.scales[i];
            // out[.., cols] = a[rows, kk] @ W^T, with a row stride of `ldd` in `out`
            let lin = |a: PM, j: usize, pre: P, cols: usize, kk: usize, out: PM, ldd: usize, beta: f32| -> Result<()> {
                k!(gpu_nvfp4_dequant(w(&format!("w{j}")), w(&format!("s{j}")), sc2[j], pre, wf.ptr(), cols as i64, kk as i32, s(st)))?;
                prof.borrow_mut().mark("dequant", st)?;
                k!(gpu_gemm(a, kk as i64, wf.ptr(), kk as i64, out, ldd as i64, null(), rows as i64, cols as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, beta, s(st)))?;
                prof.borrow_mut().mark("gemm", st)
            };
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln1"), EPS, null(), null(), n.ptr(), d, m, D as i32, s(st)))?;
            lin(n.ptr(), 0, null(), HQ * HD, D, qkv.ptr(), QKV, 0.0)?;
            lin(n.ptr(), 1, null(), HK * HD, D, keys, QKV, 0.0)?;
            lin(n.ptr(), 2, null(), HK * HD, D, values, QKV, 0.0)?;
            k!(gpu_qk_norm_rope(DT_F32, HD as i32, qkv.ptr(), QKV as i64, 0, (HQ * HD) as i32, HQ as i32, HK as i32, w("qn"), w("kn"), EPS, rope_rows, 2, m, s(st)))?;
            if let Some(b) = &keep {
                memcpy_2d_d2h(unsafe { b.ptr().add(i * l * KV) }, KV, keys, QKV * 4, KV, l, st)?;
            }
            let scale = 1.0 / (HD as f32).sqrt();
            match (&kv, &prefix) {
                (Some(kv), Some(p)) => {
                    memcpy_h2d(kv.ptr(), unsafe { p.kv.ptr().add(i * p.ids.len() * KV) }, kept * KV, st)?;
                    memcpy_2d_d2d(off(kv.ptr(), kept * KV), KV, keys, QKV * 4, KV, rows, st)?;
                    k!(gpu_attn_small(DT_F32, qkv.ptr(), QKV as i64, kv.ptr(), off(kv.ptr(), HK * HD * 4), (KV / 4) as i64, HQ as i32, HK as i32, HD as i32, m, l as i32, attn.ptr(),
                        (HQ * HD) as i64, scale, kept as i32 + 1, s(st)))?;
                }
                _ => k!(gpu_attn_small(DT_F32, qkv.ptr(), QKV as i64, keys, values, QKV as i64, HQ as i32, HK as i32, HD as i32, m, m, attn.ptr(), (HQ * HD) as i64, scale, 1, s(st)))?,
            }
            prof.borrow_mut().mark("attn", st)?;
            lin(attn.ptr(), 3, w("pre_o") as P, D, HQ * HD, xp, D, 1.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_RMS, xp, d, null(), 0, 0, w("ln2"), EPS, null(), null(), n.ptr(), d, m, D as i32, s(st)))?;
            lin(n.ptr(), 4, null(), FF, D, gu.ptr(), 2 * FF, 0.0)?;
            lin(n.ptr(), 5, null(), FF, D, off(gu.ptr(), FF * 4), 2 * FF, 0.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_SWIGLU, gu.ptr(), (2 * FF) as i64, null(), 0, FF as i64, null(), 0.0, null(), null(), act.ptr(), FF as i64, m, FF as i32, s(st)))?;
            lin(act.ptr(), 6, w("pre_d") as P, D, FF, xp, D, 1.0)?;
            prof.borrow_mut().mark("other", st)?;
            self.ws.release(i, st)?;
            if let Some(d) = deepstack.get(i) {
                let (p, c) = ([xp as P, d.ptr() as P], [1.0f32, 1.0]);
                k!(gpu_lincomb(xp, (l * D) as i64, 2, p.as_ptr(), c.as_ptr(), s(st)))?;
            }
        }
        let mut prof = prof.into_inner();
        prof.host("acquire", std::time::Instant::now() - std::time::Duration::from_secs_f64(host));
        prof.report("text encoder", st)?;
        st.sync()?;
        self.ws.replan(Plan { resident: 0, slots: 0 })?;
        self.ws.evict();
        if let Some(kv) = keep {
            let mut hidden = vec![0u8; l * row];
            x.download(0, &mut hidden, st)?;
            self.prefix = Some(Prefix { ids: all, images: images.iter().map(Arc::downgrade).collect(), kv, hidden });
        }
        let Some(p) = prefix else { return Ok(x) };
        let out = DevBuf::alloc_async(l * row, st)?;
        out.upload(0, &p.hidden[..kept * row], st)?;
        memcpy_d2d(off(out.ptr(), kept * row), x.ptr(), rows * row, st)?;
        eprintln!("text encoder: {kept} of {l} tokens kept from the last prompt");
        self.prefix = Some(p);
        Ok(out)
    }
}
