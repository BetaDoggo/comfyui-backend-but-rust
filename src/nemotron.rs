//! Nemotron-Labs-Image 30B-A3B: text to image by autoregression over visual tokens, from the file
//! tools/nemotron_image_convert.py writes.
//!
//! The generator is a Nemotron-H hybrid (Nemotron 3 Nano): 52 layers of Mamba-2, attention without positions and MoE
//! (128 routed experts, 6 a token, plus a shared one), with 196 560 visual tokens past the text vocabulary. It writes
//! `<|img_start|>`, a size header in text, then the image as rows: `<|vis_row|>` and a visual token for each 16x16
//! patch. The L24-SQ ViT decoder turns the tokens back into pixels: each token is a codebook vector.
//!
//! Every token reads the dense weights (bf16, resident in VRAM) and six experts per MoE layer (nvfp4). The experts live
//! in pinned RAM and a VRAM cache of recently used ones holds as many as fit. Decoding runs a few rows at a time:
//! `nseq` sequences (images of one prompt sampled together) times `t` new tokens of each.
//!
//! The model writes the header and the first row marker itself. After them the grid's structure is fixed, so the rows
//! are constrained to it: a row's tokens are drawn from the visual tokens alone and the markers are put in, where the
//! reference (vLLM's free sampling, then a parser that rejects a malformed grid) would fail.

use crate::cuda::{memcpy_2d_d2d, memcpy_d2d, memcpy_h2d, DevBuf, PinnedBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, P, PM, PRO_RMS, PRO_SWIGLU};
use crate::st::{Dt, SafeTensors};
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::path::Path;
use std::time::Instant;

/// The `nemotron_image` metadata of the files this reads.
pub const FORMAT: &str = "nvfp4-v1";
const D: usize = 2688;
const VOCAB: usize = 327680;
const EPS: f32 = 1e-5;
const PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
// Mamba-2
const MH: usize = 64;
const MP: usize = 64;
const MN: usize = 128;
const MG: usize = 8;
const INNER: usize = MH * MP;
const CONV: usize = INNER + 2 * MG * MN;
const PROJ: usize = INNER + CONV + MH;
const DT_COL: usize = INNER + CONV;
// attention
const HQ: usize = 32;
const HK: usize = 2;
const HD: usize = 128;
const QKV: usize = (HQ + 2 * HK) * HD;
const KVW: usize = HK * HD;
// MoE
const E: usize = 128;
const TOPK: usize = 6;
const FF: usize = 1856;
const SFF: usize = 3712;
const ROUTED_SCALE: f32 = 2.5;
// tokens
pub const IMG_START: u32 = 131072;
pub const IMG_END: u32 = 131073;
pub const VIS_ROW: u32 = 131074;
pub const VIS0: u32 = 131075;
pub const NVIS: usize = 196560;
/// The model card's system prompt.
pub const SYSTEM: &str = "Generate an image matching the requested resolution by detailing the color, quantity, text, shape, size, texture, spatial \
                          relationships of the objects and background.";
/// Images of a prompt decoded together, at most.
pub const MAX_BATCH: usize = 8;
/// Tokens the model may write before `<|img_start|>`, and in the header after it.
const MAX_PRE: usize = 16;
const MAX_HEADER: usize = 48;
// an expert, in the pinned store and in a VRAM slot: up's codes and scales, then down's
const UP_Q: usize = FF * D / 2;
const UP_S: usize = FF * D / 16;
const DN_Q: usize = D * FF / 2;
const DN_S: usize = D * FF / 16;
const EXPERT: usize = UP_Q + UP_S + DN_Q + DN_S;
/// `NEMOTRON_DUMP=<file>` writes the prompt's ids, the first image's tokens and its logits for this many steps, for
/// tools/nemotron_image_ref.py.
const DUMP_STEPS: u64 = 40;
/// VRAM kept for activations, states and the decoder's work when sizing the expert cache.
const WORK: usize = 1200 << 20;
const MB: usize = 1 << 20;

/// The visual tokens' grid of a width x height image: (rows, columns) of 16x16 patches.
pub fn grid(width: usize, height: usize) -> Result<(usize, usize)> {
    if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
        bail!("Nemotron-Labs-Image needs width and height positive multiples of 16, not {width}x{height}");
    }
    Ok((height / 16, width / 16))
}

/// The prompt's ids as the chat template renders it, with thinking off, as the model card runs it.
pub fn prompt_ids(tok: &tokenizers::Tokenizer, prompt: &str, width: usize, height: usize) -> Result<Vec<u32>> {
    let text = format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nGenerate a {width}x{height} image.\n\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"
    );
    Ok(tok.encode(text, false).map_err(|e| anyhow!("Nemotron tokenizer: {e}"))?.get_ids().to_vec())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Fp4Job {
    q: P,
    sc: P,
    scale2: f32,
    xrow: i32,
    orow: i32,
    pad: i32,
}

extern "C" {
    fn gpu_nm_gemv_bf16(w: P, ldw: i64, x: P, ldx: i64, out: PM, ldo: i64, rows: i32, n: i32, k: i32, accumulate: i32, act: i32, s: PM) -> i32;
    fn gpu_nm_gemv_fp4(jobs: P, njobs: i32, x: P, ldx: i64, out: PM, ldo: i64, n: i32, k: i32, act: i32, s: PM) -> i32;
    fn gpu_nm_mamba_conv(inp: P, ld_in: i64, state: PM, w: P, b: P, out: PM, nseq: i32, t: i32, c: i32, s: PM) -> i32;
    fn gpu_nm_mamba_scan(xbc: P, ld_xbc: i64, zx: P, ldzx: i64, dt_col: i32, ssm: PM, a_log: P, d: P, dt_bias: P, y: PM, ldy: i64, nseq: i32, t: i32,
        h: i32, p: i32, n: i32, g: i32, s: PM) -> i32;
    fn gpu_nm_gated_norm(y: P, ldy: i64, z: P, ldz: i64, w: P, eps: f32, out: PM, ldo: i64, rows: i32, c: i32, gs: i32, s: PM) -> i32;
    fn gpu_nm_router(x: P, ldx: i64, w: P, bias: P, rows: i32, e: i32, k: i32, topk: i32, scale: f32, idx: PM, wts: PM, s: PM) -> i32;
    fn gpu_nm_moe_combine(x: PM, ldx: i64, shared: P, y: P, wts: P, rows: i32, topk: i32, d: i32, s: PM) -> i32;
    fn gpu_nm_sample(logits: P, ld: i64, rows: i32, v: i32, temperature: f32, seeds: P, step: u64, out: PM, s: PM) -> i32;
    fn gpu_nm_attn_rows(q: P, q_stride: i64, k: P, v: P, kv_stride: i64, hq: i32, hk: i32, d: i32, lq: i32, lk: i32, out: PM, out_stride: i64, scale: f32,
        offset: i32, s: PM) -> i32;
    fn gpu_nm_codebook_lookup(codebook: P, idx: P, n: i32, d: i32, out: PM, s: PM) -> i32;
    fn gpu_nm_rope2d(x: PM, ld: i64, gh: i32, gw: i32, heads: i32, theta: f32, s: PM) -> i32;
    fn gpu_nm_tanh(x: PM, n: i64, s: PM) -> i32;
    fn gpu_nm_unpatch_rgb8(tok: P, gh: i32, gw: i32, out: PM, s: PM) -> i32;
}

macro_rules! nk {
    ($f:ident ( $($a:expr),* $(,)? )) => {
        crate::ffi::ok(unsafe { $f($($a),*) }, stringify!($f))
    };
}

/// out[r, ..n] (+)= W[n, k] . x[r, ..k] for `rows` rows, bf16 weights, eight rows a launch.
#[allow(clippy::too_many_arguments)]
fn gemv(w: P, k_: usize, x: PM, ldx: usize, out: PM, ldo: usize, rows: usize, n: usize, accumulate: bool, relu2: bool, st: &Stream) -> Result<()> {
    let mut r = 0;
    while r < rows {
        let m = (rows - r).min(8);
        nk!(gpu_nm_gemv_bf16(w, k_ as i64, off(x, r * ldx * 4), ldx as i64, off(out, r * ldo * 4), ldo as i64, m as i32, n as i32, k_ as i32,
            accumulate as i32, relu2 as i32, s(st)))?;
        r += m;
    }
    Ok(())
}

fn rms(x: PM, w: P, out: PM, rows: usize, width: usize, eps: f32, st: &Stream) -> Result<()> {
    let ld = width as i64;
    k!(gpu_prologue_rows(DT_F32, PRO_RMS, x, ld, null(), 0, 0, w as *const f32 as P, eps, null(), null(), out, ld, rows as i32, width as i32, s(st)))
}

/// A tensor of the file in VRAM, as stored; `dt` checks its type.
fn dev(stf: &SafeTensors, name: &str, dt: Dt, st: &Stream) -> Result<DevBuf> {
    let i = stf.info(name)?;
    ensure!(i.dtype == dt, "{name} is {:?}, expected {dt:?}", i.dtype);
    let b = DevBuf::alloc(i.len)?;
    b.upload(0, stf.bytes(name)?, st)?;
    stf.evict(name);
    Ok(b)
}

/// Several tensors of the file side by side in one VRAM buffer.
fn dev_cat(stf: &SafeTensors, names: &[String], dt: Dt, st: &Stream) -> Result<DevBuf> {
    let mut total = 0;
    for n in names {
        let i = stf.info(n)?;
        ensure!(i.dtype == dt, "{n} is {:?}, expected {dt:?}", i.dtype);
        total += i.len;
    }
    let b = DevBuf::alloc(total)?;
    let mut at = 0;
    for n in names {
        let bytes = stf.bytes(n)?;
        b.upload(at, bytes, st)?;
        at += bytes.len();
    }
    Ok(b)
}

enum Mixer {
    Mamba { in_proj: DevBuf, out_proj: DevBuf, conv_w: DevBuf, conv_b: DevBuf, a_log: DevBuf, d: DevBuf, dt_bias: DevBuf, norm: DevBuf },
    /// q, k and v stacked: [QKV, D]
    Attn { qkv: DevBuf, o: DevBuf },
    /// `moe`: its place among the MoE layers, in the expert store
    Moe { moe: usize, router: DevBuf, bias: DevBuf, up: DevBuf, down: DevBuf },
}

struct Layer {
    norm: DevBuf,
    mixer: Mixer,
}

/// Every routed expert, packed as a VRAM slot holds it, in pinned RAM: one buffer a MoE layer.
struct Store {
    layers: Vec<PinnedBuf>,
    /// (up, down) scale_2, [moe layer][expert]
    scale2: Vec<[[f32; 2]; E]>,
}

impl Store {
    fn load(stf: &SafeTensors, moe_layers: &[usize], threads: usize) -> Result<Self> {
        let t0 = Instant::now();
        let mut layers = Vec::with_capacity(moe_layers.len());
        let mut scale2 = Vec::with_capacity(moe_layers.len());
        for &l in moe_layers {
            let mut buf = PinnedBuf::alloc(E * EXPERT).context("pinning RAM for the experts (16.5 GB)")?;
            let mut s2 = [[0f32; 2]; E];
            let name = |e: usize, p: &str, t: &str| format!("backbone.layers.{l}.mixer.experts.{e}.{p}_proj.{t}");
            for (e, s) in s2.iter_mut().enumerate() {
                for (j, p) in ["up", "down"].iter().enumerate() {
                    s[j] = stf.f32_vec(&name(e, p, "weight_scale_2"), false)?[0];
                }
            }
            let parts: Vec<(usize, [&[u8]; 4])> = (0..E)
                .map(|e| {
                    let b = |p: &str, t: &str| stf.bytes(&name(e, p, t));
                    Ok((e, [b("up", "weight")?, b("up", "weight_scale")?, b("down", "weight")?, b("down", "weight_scale")?]))
                })
                .collect::<Result<_>>()?;
            for (e, p) in &parts {
                ensure!([p[0].len(), p[1].len(), p[2].len(), p[3].len()] == [UP_Q, UP_S, DN_Q, DN_S], "layer {l} expert {e}: unexpected sizes");
            }
            let dst = buf.as_mut_slice();
            let chunks: Vec<&mut [u8]> = dst.chunks_mut(EXPERT).collect();
            let per = E.div_ceil(threads.max(1));
            std::thread::scope(|sc| {
                let mut it = chunks.into_iter().zip(parts.iter());
                loop {
                    let group: Vec<_> = it.by_ref().take(per).collect();
                    if group.is_empty() {
                        break;
                    }
                    sc.spawn(move || {
                        for (dst, (_, p)) in group {
                            let mut at = 0;
                            for part in p {
                                dst[at..at + part.len()].copy_from_slice(part);
                                at += part.len();
                            }
                        }
                    });
                }
            });
            for e in 0..E {
                for p in ["up", "down"] {
                    stf.evict(&name(e, p, "weight"));
                    stf.evict(&name(e, p, "weight_scale"));
                }
            }
            layers.push(buf);
            scale2.push(s2);
        }
        eprintln!("Nemotron experts pinned in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(Store { layers, scale2 })
    }
}

/// The experts in VRAM: slots of [`EXPERT`] bytes, the least recently used given up first.
struct Cache {
    pool: DevBuf,
    /// The (moe layer, expert) in each slot.
    owner: Vec<Option<(usize, usize)>>,
    /// The MoE call that last used each slot.
    last: Vec<u64>,
    /// Each (moe layer * E + expert)'s slot.
    slot: Vec<Option<usize>>,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl Cache {
    fn new(slots: usize, moe_layers: usize) -> Result<Self> {
        Ok(Cache {
            pool: DevBuf::alloc(slots * EXPERT)?,
            owner: vec![None; slots],
            last: vec![0; slots],
            slot: vec![None; moe_layers * E],
            tick: 0,
            hits: 0,
            misses: 0,
        })
    }

    /// The device address of an expert, copied in from `store` unless it is there; never one used by this call.
    fn get(&mut self, store: &Store, moe: usize, e: usize, st: &Stream) -> Result<PM> {
        let key = moe * E + e;
        let slot = match self.slot[key] {
            Some(sl) => {
                self.hits += 1;
                sl
            }
            None => {
                self.misses += 1;
                let sl = (0..self.owner.len())
                    .filter(|&i| self.last[i] < self.tick)
                    .min_by_key(|&i| (self.owner[i].is_some(), self.last[i]))
                    .context("the expert cache is too small for one call's experts")?;
                if let Some((m, x)) = self.owner[sl] {
                    self.slot[m * E + x] = None;
                }
                self.owner[sl] = Some((moe, e));
                self.slot[key] = Some(sl);
                // the copy waits on the stream for earlier kernels that read the slot
                memcpy_h2d(self.pool.at(sl * EXPERT), unsafe { store.layers[moe].ptr().add(e * EXPERT) }, EXPERT, st)?;
                sl
            }
        };
        self.last[slot] = self.tick;
        Ok(self.pool.at(slot * EXPERT))
    }
}

/// The L24-SQ ViT decoder.
struct Decoder {
    codebook: DevBuf,
    pq_w: DevBuf,
    pq_b: DevBuf,
    blocks: Vec<DecBlock>,
    norm: DevBuf,
    ffn_w: DevBuf,
    ffn_b: DevBuf,
    out_w: DevBuf,
    out_b: DevBuf,
}

struct DecBlock {
    n1: DevBuf,
    qkv: DevBuf,
    o: DevBuf,
    n2: DevBuf,
    gate_up: DevBuf,
    down: DevBuf,
}

const DW: usize = 1024;
const DHEADS: usize = 16;
const DFF: usize = 2752;
const DLAYERS: usize = 24;
const DCODE: usize = 24;
const DOUT: usize = 3072;

impl Decoder {
    fn load(stf: &SafeTensors, st: &Stream) -> Result<Self> {
        let p = "image_decoder.";
        let f = |n: &str| dev(stf, &format!("{p}{n}"), Dt::F32, st);
        let mut blocks = Vec::new();
        for i in 0..DLAYERS {
            let b = |n: &str| format!("{p}decoder.blocks.{i}.{n}");
            blocks.push(DecBlock {
                n1: dev(stf, &b("norm1.weight"), Dt::F32, st)?,
                qkv: dev_cat(stf, &[b("attn.q_proj.weight"), b("attn.k_proj.weight"), b("attn.v_proj.weight")], Dt::F32, st)?,
                o: dev(stf, &b("attn.out_proj.weight"), Dt::F32, st)?,
                n2: dev(stf, &b("norm2.weight"), Dt::F32, st)?,
                gate_up: dev_cat(stf, &[b("mlp.gate_proj.weight"), b("mlp.up_proj.weight")], Dt::F32, st)?,
                down: dev(stf, &b("mlp.down_proj.weight"), Dt::F32, st)?,
            });
        }
        ensure!(stf.info(&format!("{p}decoder.blocks.0.mlp.gate_proj.weight"))?.shape == [DFF, DW], "unexpected decoder MLP width");
        Ok(Decoder {
            codebook: f("quantize.embedding.weight")?,
            pq_w: f("post_quant_embed.weight")?,
            pq_b: f("post_quant_embed.bias")?,
            blocks,
            norm: f("decoder.norm.weight")?,
            ffn_w: f("decoder.ffn.0.weight")?,
            ffn_b: f("decoder.ffn.0.bias")?,
            out_w: f("decoder.conv_out.weight")?,
            out_b: f("decoder.conv_out.bias")?,
        })
    }

    /// Codebook indices of a gh x gw grid, row major, to RGB8 pixels [gh * 16][gw * 16][3].
    fn decode(&self, codes: &[u32], gh: usize, gw: usize, st: &Stream) -> Result<Vec<u8>> {
        let l = gh * gw;
        ensure!(codes.len() == l, "{} codes for a {gh}x{gw} grid", codes.len());
        let ids: Vec<u8> = codes.iter().flat_map(|&c| (c as i32).to_le_bytes()).collect();
        let idx = DevBuf::alloc_async(ids.len(), st)?;
        idx.upload(0, &ids, st)?;
        let zq = DevBuf::alloc_async(l * DCODE * 4, st)?;
        nk!(gpu_nm_codebook_lookup(self.codebook.ptr(), idx.ptr(), l as i32, DCODE as i32, zq.ptr(), s(st)))?;
        let gemm = |a: PM, lda: usize, w: &DevBuf, out: PM, ldd: usize, bias: P, n: usize, kk: usize, beta: f32| -> Result<()> {
            k!(gpu_gemm(a, lda as i64, w.ptr(), kk as i64, out, ldd as i64, bias, l as i64, n as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, beta, s(st)))
        };
        let h = DevBuf::alloc_async(l * DW * 4, st)?;
        gemm(zq.ptr(), DCODE, &self.pq_w, h.ptr(), DW, self.pq_b.ptr(), DW, DCODE, 0.0)?;
        let n = DevBuf::alloc_async(l * DW * 4, st)?;
        let qkv = DevBuf::alloc_async(l * 3 * DW * 4, st)?;
        let att = DevBuf::alloc_async(l * DW * 4, st)?;
        let gu = DevBuf::alloc_async(l * 2 * DFF * 4, st)?;
        let act = DevBuf::alloc_async(l * DFF * 4, st)?;
        let hd = DW / DHEADS;
        for b in &self.blocks {
            rms(h.ptr(), b.n1.ptr(), n.ptr(), l, DW, 1e-6, st)?;
            gemm(n.ptr(), DW, &b.qkv, qkv.ptr(), 3 * DW, null(), 3 * DW, DW, 0.0)?;
            nk!(gpu_nm_rope2d(qkv.ptr(), (3 * DW) as i64, gh as i32, gw as i32, (2 * DHEADS) as i32, 10000.0, s(st)))?;
            k!(gpu_attn_small(DT_F32, qkv.ptr(), (3 * DW) as i64, off(qkv.ptr(), DW * 4), off(qkv.ptr(), 2 * DW * 4), (3 * DW) as i64, DHEADS as i32,
                DHEADS as i32, hd as i32, l as i32, l as i32, att.ptr(), DW as i64, 1.0 / (hd as f32).sqrt(), 0, s(st)))?;
            gemm(att.ptr(), DW, &b.o, h.ptr(), DW, null(), DW, DW, 1.0)?;
            rms(h.ptr(), b.n2.ptr(), n.ptr(), l, DW, 1e-6, st)?;
            gemm(n.ptr(), DW, &b.gate_up, gu.ptr(), 2 * DFF, null(), 2 * DFF, DW, 0.0)?;
            k!(gpu_prologue_rows(DT_F32, PRO_SWIGLU, gu.ptr(), (2 * DFF) as i64, null(), 0, DFF as i64, null(), 0.0, null(), null(), act.ptr(), DFF as i64,
                l as i32, DFF as i32, s(st)))?;
            gemm(act.ptr(), DFF, &b.down, h.ptr(), DW, null(), DW, DFF, 1.0)?;
        }
        drop((qkv, att, gu, act));
        rms(h.ptr(), self.norm.ptr(), n.ptr(), l, DW, 1e-6, st)?;
        let f = DevBuf::alloc_async(l * DOUT * 4, st)?;
        gemm(n.ptr(), DW, &self.ffn_w, f.ptr(), DOUT, self.ffn_b.ptr(), DOUT, DW, 0.0)?;
        nk!(gpu_nm_tanh(f.ptr(), (l * DOUT) as i64, s(st)))?;
        let o = DevBuf::alloc_async(l * 768 * 4, st)?;
        gemm(f.ptr(), DOUT, &self.out_w, o.ptr(), 768, self.out_b.ptr(), 768, DOUT, 0.0)?;
        let px = DevBuf::alloc_async(l * 768, st)?;
        nk!(gpu_nm_unpatch_rgb8(o.ptr(), gh as i32, gw as i32, px.ptr(), s(st)))?;
        let mut out = vec![0u8; l * 768];
        px.download(0, &mut out, st)?;
        Ok(out)
    }
}

/// What one image's next token is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// Before `<|img_start|>`: sampled from the whole vocabulary.
    Pre(usize),
    /// The size header, up to the first `<|vis_row|>`: sampled from the whole vocabulary.
    Header,
    /// In grid row `row`, `col` tokens of it written.
    Grid { row: usize, col: usize },
    Done,
}

/// One generation: a prompt, its images' seeds and the sampling.
pub struct Request<'a> {
    pub prompt: &'a str,
    pub width: usize,
    pub height: usize,
    /// One image for each, sampled together; at most [`MAX_BATCH`].
    pub seeds: &'a [u64],
    /// Softmax temperature of the tokens; 0 takes the most likely one. The model card uses 0.9.
    pub temperature: f32,
}

pub struct Model {
    stf: SafeTensors,
    tok: tokenizers::Tokenizer,
    layers: Vec<Layer>,
    norm_f: DevBuf,
    lm_head: DevBuf,
    store: Store,
    cache: Cache,
    decoder: Decoder,
    moe_layers: usize,
}

/// The decoding state of `nseq` sequences.
struct State {
    /// Per Mamba layer: conv inputs [nseq, CONV, 3] and the SSM state [nseq, MH, MP, MN], f32.
    conv: Vec<DevBuf>,
    ssm: Vec<DevBuf>,
    /// Per attention layer: keys and values [nseq, cap, KVW], f32.
    keys: Vec<DevBuf>,
    values: Vec<DevBuf>,
    cap: usize,
    /// Tokens in the caches.
    pos: usize,
}

/// Activations of `rows` rows.
struct Work {
    x: DevBuf,
    n: DevBuf,
    zx: DevBuf,
    xbc: DevBuf,
    y: DevBuf,
    g: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    hs: DevBuf,
    sh: DevBuf,
    idx: DevBuf,
    wts: DevBuf,
    h: DevBuf,
    ye: DevBuf,
    jobs: DevBuf,
}

impl Work {
    fn new(rows: usize, st: &Stream) -> Result<Self> {
        let f = |n: usize| DevBuf::alloc_async(n * 4, st);
        let r6 = rows * TOPK;
        Ok(Work {
            x: f(rows * D)?,
            n: f(rows * D)?,
            zx: f(rows * PROJ)?,
            xbc: f(rows * CONV)?,
            y: f(rows * INNER)?,
            g: f(rows * INNER)?,
            qkv: f(rows * QKV)?,
            att: f(rows * HQ * HD)?,
            hs: f(rows * SFF)?,
            sh: f(rows * D)?,
            idx: f(r6)?,
            wts: f(r6)?,
            h: f(r6 * FF)?,
            ye: f(r6 * D)?,
            jobs: DevBuf::alloc_async(2 * r6 * std::mem::size_of::<Fp4Job>(), st)?,
        })
    }
}

fn config_check(stf: &SafeTensors) -> Result<()> {
    let format = stf.metadata.get("nemotron_image").map(String::as_str);
    if format != Some(FORMAT) {
        bail!("not a Nemotron-Labs-Image file of format {FORMAT} (found {format:?}): convert it with tools/nemotron_image_convert.py");
    }
    let cfg: serde_json::Value = serde_json::from_str(stf.metadata.get("config").context("no config in the file")?)?;
    let want: [(&str, serde_json::Value); 12] = [
        ("hidden_size", D.into()),
        ("hybrid_override_pattern", PATTERN.into()),
        ("vocab_size", VOCAB.into()),
        ("mamba_num_heads", MH.into()),
        ("mamba_head_dim", MP.into()),
        ("ssm_state_size", MN.into()),
        ("n_groups", MG.into()),
        ("num_attention_heads", HQ.into()),
        ("num_key_value_heads", HK.into()),
        ("n_routed_experts", E.into()),
        ("num_experts_per_tok", TOPK.into()),
        ("moe_intermediate_size", FF.into()),
    ];
    for (k, v) in want {
        ensure!(cfg[k] == v, "config {k} is {}, this engine is written for {v}", cfg[k]);
    }
    ensure!(cfg["moe_shared_expert_intermediate_size"] == SFF && cfg["conv_kernel"] == 4 && cfg["n_group"] == 1 && cfg["head_dim"] == HD,
        "the config differs from the model this engine is written for");
    ensure!((cfg["routed_scaling_factor"].as_f64() == Some(ROUTED_SCALE as f64)) && cfg["norm_topk_prob"] == true, "unexpected router settings");
    Ok(())
}

impl Model {
    /// Loads the dense weights and the decoder into VRAM, pins the experts in RAM and gives the VRAM left, but for
    /// `reserve` bytes, to the expert cache.
    pub fn load(path: &Path, threads: usize, reserve: usize, st: &Stream) -> Result<Self> {
        let t0 = Instant::now();
        let stf = SafeTensors::open(path)?;
        config_check(&stf)?;
        let tok = tokenizers::Tokenizer::from_bytes(stf.bytes("tokenizer.json")?).map_err(|e| anyhow!("Nemotron tokenizer: {e}"))?;
        let mut layers = Vec::new();
        let mut moe_ids = Vec::new();
        for (i, c) in PATTERN.chars().enumerate() {
            let n = |x: &str| format!("backbone.layers.{i}.mixer.{x}");
            let mixer = match c {
                'M' => Mixer::Mamba {
                    in_proj: dev(&stf, &n("in_proj.weight"), Dt::BF16, st)?,
                    out_proj: dev(&stf, &n("out_proj.weight"), Dt::BF16, st)?,
                    conv_w: dev(&stf, &n("conv1d.weight"), Dt::F32, st)?,
                    conv_b: dev(&stf, &n("conv1d.bias"), Dt::F32, st)?,
                    a_log: dev(&stf, &n("A_log"), Dt::F32, st)?,
                    d: dev(&stf, &n("D"), Dt::F32, st)?,
                    dt_bias: dev(&stf, &n("dt_bias"), Dt::F32, st)?,
                    norm: dev(&stf, &n("norm.weight"), Dt::F32, st)?,
                },
                '*' => Mixer::Attn {
                    qkv: dev_cat(&stf, &[n("q_proj.weight"), n("k_proj.weight"), n("v_proj.weight")], Dt::BF16, st)?,
                    o: dev(&stf, &n("o_proj.weight"), Dt::BF16, st)?,
                },
                _ => {
                    moe_ids.push(i);
                    Mixer::Moe {
                        moe: moe_ids.len() - 1,
                        router: dev(&stf, &n("gate.weight"), Dt::F32, st)?,
                        bias: dev(&stf, &n("gate.e_score_correction_bias"), Dt::F32, st)?,
                        up: dev(&stf, &n("shared_experts.up_proj.weight"), Dt::BF16, st)?,
                        down: dev(&stf, &n("shared_experts.down_proj.weight"), Dt::BF16, st)?,
                    }
                }
            };
            layers.push(Layer { norm: dev(&stf, &format!("backbone.layers.{i}.norm.weight"), Dt::F32, st)?, mixer });
        }
        let norm_f = dev(&stf, "backbone.norm_f.weight", Dt::F32, st)?;
        let lm_head = dev(&stf, "lm_head.weight", Dt::BF16, st)?;
        let decoder = Decoder::load(&stf, st)?;
        st.sync()?;
        eprintln!("Nemotron dense weights in VRAM in {:.1}s", t0.elapsed().as_secs_f64());
        let store = Store::load(&stf, &moe_ids, threads)?;
        crate::cuda::trim_pool()?;
        let free = crate::cuda::device_free()?;
        let total = moe_ids.len() * E;
        let slots = (free.saturating_sub(reserve + WORK) / EXPERT).min(total);
        ensure!(slots >= 2 * E, "only {} MB of VRAM left for Nemotron's experts: it needs {} MB at least", free / MB, 2 * E * EXPERT / MB);
        let cache = Cache::new(slots, moe_ids.len())?;
        eprintln!("Nemotron ready in {:.1}s: {} of {} experts fit in VRAM ({} MB)", t0.elapsed().as_secs_f64(), slots, total, slots * EXPERT / MB);
        Ok(Model { stf, tok, layers, norm_f, lm_head, store, cache, decoder, moe_layers: moe_ids.len() })
    }

    fn state(&self, nseq: usize, cap: usize, st: &Stream) -> Result<State> {
        let zero = |bytes: usize| -> Result<DevBuf> {
            let b = DevBuf::alloc_async(bytes, st)?;
            k!(gpu_zero_f32(b.ptr(), (bytes / 4) as i64, s(st)))?;
            Ok(b)
        };
        let (mut conv, mut ssm, mut keys, mut values) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for l in &self.layers {
            match l.mixer {
                Mixer::Mamba { .. } => {
                    conv.push(zero(nseq * CONV * 3 * 4)?);
                    ssm.push(zero(nseq * MH * MP * MN * 4)?);
                }
                Mixer::Attn { .. } => {
                    keys.push(DevBuf::alloc_async(nseq * cap * KVW * 4, st)?);
                    values.push(DevBuf::alloc_async(nseq * cap * KVW * 4, st)?);
                }
                Mixer::Moe { .. } => {}
            }
        }
        Ok(State { conv, ssm, keys, values, cap, pos: 0 })
    }

    /// Sequence 0's state copied to the others, after a prompt read once for all of them.
    fn spread(&self, s0: &State, nseq: usize, st: &Stream) -> Result<State> {
        let mut out = self.state(nseq, s0.cap, st)?;
        let rep = |src: &DevBuf, dst: &DevBuf, bytes: usize| -> Result<()> {
            for i in 0..nseq {
                memcpy_d2d(dst.at(i * bytes), src.ptr(), bytes, st)?;
            }
            Ok(())
        };
        for (a, b) in s0.conv.iter().zip(&out.conv) {
            rep(a, b, CONV * 3 * 4)?;
        }
        for (a, b) in s0.ssm.iter().zip(&out.ssm) {
            rep(a, b, MH * MP * MN * 4)?;
        }
        for (a, b) in s0.keys.iter().zip(&out.keys).chain(s0.values.iter().zip(&out.values)) {
            rep(a, b, s0.cap * KVW * 4)?;
        }
        out.pos = s0.pos;
        Ok(out)
    }

    /// Embeddings of `ids` into the first rows of `x`.
    fn embed(&self, ids: &[u32], x: &DevBuf, st: &Stream) -> Result<()> {
        let table = self.stf.bytes("backbone.embeddings.weight")?;
        let mut rows = Vec::with_capacity(ids.len() * D * 4);
        for &t in ids {
            ensure!((t as usize) < VOCAB, "token {t} past the vocabulary");
            let row = &table[t as usize * D * 2..(t as usize + 1) * D * 2];
            rows.extend(row.chunks_exact(2).flat_map(|c| (half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes()));
        }
        x.upload(0, &rows, st)
    }

    /// The layers over `t` new tokens of each of `nseq` sequences, whose embeddings are in `w.x` (rows s * t + i).
    fn forward(&mut self, state: &mut State, w: &Work, nseq: usize, t: usize, st: &Stream) -> Result<()> {
        let rows = nseq * t;
        ensure!(state.pos + t <= state.cap, "past the {} token context", state.cap);
        let (x, n) = (w.x.ptr(), w.n.ptr());
        let (mut mi, mut ai) = (0, 0);
        let debug = std::env::var_os("NEMOTRON_DEBUG").is_some();
        for (li, l) in self.layers.iter().enumerate() {
            rms(x, l.norm.ptr(), n, rows, D, EPS, st)?;
            match &l.mixer {
                Mixer::Mamba { in_proj, out_proj, conv_w, conv_b, a_log, d, dt_bias, norm } => {
                    gemv(in_proj.ptr(), D, n, D, w.zx.ptr(), PROJ, rows, PROJ, false, false, st)?;
                    nk!(gpu_nm_mamba_conv(off(w.zx.ptr(), INNER * 4), PROJ as i64, state.conv[mi].ptr(), conv_w.ptr(), conv_b.ptr(), w.xbc.ptr(),
                        nseq as i32, t as i32, CONV as i32, s(st)))?;
                    nk!(gpu_nm_mamba_scan(w.xbc.ptr(), CONV as i64, w.zx.ptr(), PROJ as i64, DT_COL as i32, state.ssm[mi].ptr(), a_log.ptr(), d.ptr(),
                        dt_bias.ptr(), w.y.ptr(), INNER as i64, nseq as i32, t as i32, MH as i32, MP as i32, MN as i32, MG as i32, s(st)))?;
                    nk!(gpu_nm_gated_norm(w.y.ptr(), INNER as i64, w.zx.ptr(), PROJ as i64, norm.ptr(), EPS, w.g.ptr(), INNER as i64, rows as i32,
                        INNER as i32, (INNER / MG) as i32, s(st)))?;
                    gemv(out_proj.ptr(), INNER, w.g.ptr(), INNER, x, D, rows, D, true, false, st)?;
                    mi += 1;
                }
                Mixer::Attn { qkv, o } => {
                    gemv(qkv.ptr(), D, n, D, w.qkv.ptr(), QKV, rows, QKV, false, false, st)?;
                    let (kc, vc) = (&state.keys[ai], &state.values[ai]);
                    for sq in 0..nseq {
                        let src = off(w.qkv.ptr(), sq * t * QKV * 4);
                        let at = (sq * state.cap + state.pos) * KVW * 4;
                        memcpy_2d_d2d(kc.at(at), KVW * 4, off(src, HQ * HD * 4), QKV * 4, KVW * 4, t, st)?;
                        memcpy_2d_d2d(vc.at(at), KVW * 4, off(src, (HQ * HD + KVW) * 4), QKV * 4, KVW * 4, t, st)?;
                        let base = sq * state.cap * KVW * 4;
                        let (o, scale) = (off(w.att.ptr(), sq * t * HQ * HD * 4), 1.0 / (HD as f32).sqrt());
                        if t >= 64 {
                            // the tiled kernel: an online softmax, which has no such trouble
                            k!(gpu_attn_small(DT_F32, src, QKV as i64, kc.at(base), vc.at(base), KVW as i64, HQ as i32, HK as i32, HD as i32, t as i32,
                                (state.pos + t) as i32, o, (HQ * HD) as i64, scale, (state.pos + 1) as i32, s(st)))?;
                        } else {
                            nk!(gpu_nm_attn_rows(src, QKV as i64, kc.at(base), vc.at(base), KVW as i64, HQ as i32, HK as i32, HD as i32, t as i32,
                                (state.pos + t) as i32, o, (HQ * HD) as i64, scale, (state.pos + 1) as i32, s(st)))?;
                        }
                    }
                    if debug {
                        let bad = |b: &DevBuf, n: usize| -> Result<Option<(usize, f32)>> {
                            let mut v = vec![0u8; n * 4];
                            b.download(0, &mut v, st)?;
                            Ok(v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).enumerate().find(|(_, x)| !x.is_finite()))
                        };
                        let q = bad(&w.qkv, rows * QKV)?;
                        let kc = bad(&state.keys[ai], (state.pos + t) * KVW)?;
                        let vc = bad(&state.values[ai], (state.pos + t) * KVW)?;
                        let a = bad(&w.att, rows * HQ * HD)?;
                        if q.is_some() || kc.is_some() || vc.is_some() || a.is_some() {
                            let mut v = vec![0u8; rows * QKV * 4];
                            w.qkv.download(0, &mut v, st)?;
                            let f: Vec<f32> = v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let mx = f.iter().fold(0f32, |m, x| m.max(x.abs()));
                            let mut dump = std::collections::BTreeMap::new();
                            let mut get = |name: &str, b: &DevBuf, n: usize| -> Result<()> {
                                let mut v = vec![0u8; n * 4];
                                b.download(0, &mut v, st)?;
                                dump.insert(name.to_string(), ("F32".to_string(), vec![n], v));
                                Ok(())
                            };
                            get("qkv", &w.qkv, rows * QKV)?;
                            get("keys", &state.keys[ai], (state.pos + t) * KVW)?;
                            get("values", &state.values[ai], (state.pos + t) * KVW)?;
                            get("att", &w.att, rows * HQ * HD)?;
                            if let Some(p) = std::env::var_os("NEMOTRON_DEBUG_DUMP") {
                                crate::steer::write_safetensors(std::path::Path::new(&p), &dump)?;
                            }
                            bail!("layer {li} attention at position {}: qkv {q:?} keys {kc:?} values {vc:?} out {a:?}, max |qkv| {mx}", state.pos);
                        }
                    }
                    gemv(o.ptr(), HQ * HD, w.att.ptr(), HQ * HD, x, D, rows, D, true, false, st)?;
                    ai += 1;
                }
                Mixer::Moe { moe, router, bias, up, down } => {
                    nk!(gpu_nm_router(n, D as i64, router.ptr(), bias.ptr(), rows as i32, E as i32, D as i32, TOPK as i32, ROUTED_SCALE, w.idx.ptr(),
                        w.wts.ptr(), s(st)))?;
                    // the shared expert while the routing comes back
                    gemv(up.ptr(), D, n, D, w.hs.ptr(), SFF, rows, SFF, false, true, st)?;
                    gemv(down.ptr(), SFF, w.hs.ptr(), SFF, w.sh.ptr(), D, rows, D, false, false, st)?;
                    let mut idx = vec![0u8; rows * TOPK * 4];
                    w.idx.download(0, &mut idx, st)?;
                    self.cache.tick += 1;
                    let mut jobs = Vec::with_capacity(2 * rows * TOPK);
                    let mut downs = Vec::with_capacity(rows * TOPK);
                    for (j, c) in idx.chunks_exact(4).enumerate() {
                        let e = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize;
                        ensure!(e < E, "layer {li}: the router chose no expert for row {} at position {} (non-finite scores)", j / TOPK, state.pos);
                        let base = self.cache.get(&self.store, *moe, e, st)?;
                        let s2 = self.store.scale2[*moe][e];
                        let (r, j) = (j as i32 / TOPK as i32, j as i32);
                        jobs.push(Fp4Job { q: base, sc: off(base, UP_Q), scale2: s2[0], xrow: r, orow: j, pad: 0 });
                        downs.push(Fp4Job { q: off(base, UP_Q + UP_S), sc: off(base, UP_Q + UP_S + DN_Q), scale2: s2[1], xrow: j, orow: j, pad: 0 });
                    }
                    let nj = jobs.len();
                    jobs.extend(downs);
                    let bytes = unsafe { std::slice::from_raw_parts(jobs.as_ptr() as *const u8, jobs.len() * std::mem::size_of::<Fp4Job>()) };
                    w.jobs.upload(0, bytes, st)?;
                    nk!(gpu_nm_gemv_fp4(w.jobs.ptr(), nj as i32, n, D as i64, w.h.ptr(), FF as i64, FF as i32, D as i32, 1, s(st)))?;
                    nk!(gpu_nm_gemv_fp4(w.jobs.at(nj * std::mem::size_of::<Fp4Job>()), nj as i32, w.h.ptr(), FF as i64, w.ye.ptr(), D as i64, D as i32,
                        FF as i32, 0, s(st)))?;
                    nk!(gpu_nm_moe_combine(x, D as i64, w.sh.ptr(), w.ye.ptr(), w.wts.ptr(), rows as i32, TOPK as i32, D as i32, s(st)))?;
                    // the jobs table is read by the kernels above before the next upload: uploads are on this stream
                }
            }
            if debug {
                let mut b = vec![0u8; rows * D * 4];
                w.x.download(0, &mut b, st)?;
                let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                if let Some(i) = v.iter().position(|x| !x.is_finite()) {
                    bail!("layer {li} ({}) wrote {} at row {} column {} (position {})", &PATTERN[li..li + 1], v[i], i / D, i % D, state.pos);
                }
            }
        }
        state.pos += t;
        Ok(())
    }

    /// Logits [rows, cols] of the given rows of `w.x` (each the last token of a sequence), over the vocabulary from
    /// token `first`.
    fn logits(&self, w: &Work, rows: &[usize], first: usize, cols: usize, out: &DevBuf, st: &Stream) -> Result<()> {
        for (i, &r) in rows.iter().enumerate() {
            rms(off(w.x.ptr(), r * D * 4), self.norm_f.ptr(), off(w.n.ptr(), i * D * 4), 1, D, EPS, st)?;
        }
        gemv(self.lm_head.at(first * D * 2), D, w.n.ptr(), D, out.ptr(), cols, rows.len(), cols, false, false, st)
    }

    /// Generate the request's images, RGB8 [height][width][3] each. `progress(done, total)` hears each token and stops
    /// the run when it returns false.
    pub fn generate(&mut self, r: &Request, mut progress: impl FnMut(usize, usize) -> bool, st: &Stream) -> Result<Vec<Vec<u8>>> {
        let nseq = r.seeds.len();
        ensure!((1..=MAX_BATCH).contains(&nseq), "Nemotron-Labs-Image samples 1 to {MAX_BATCH} images together, not {nseq}");
        ensure!(r.temperature >= 0.0 && r.temperature.is_finite(), "temperature {} must be 0 or more", r.temperature);
        let (gh, gw) = grid(r.width, r.height)?;
        let ids = prompt_ids(&self.tok, r.prompt, r.width, r.height)?;
        // the image's tokens, its markers and room for the header
        let total = gh * (gw + 1) + 1;
        let cap = ids.len() + MAX_PRE + MAX_HEADER + total + 1;
        let t0 = Instant::now();
        let (h0, m0) = (self.cache.hits, self.cache.misses);

        // the prompt, once for every image
        let w = Work::new(ids.len().max(nseq), st)?;
        let mut s1 = self.state(1, cap, st)?;
        self.embed(&ids, &w.x, st)?;
        self.forward(&mut s1, &w, 1, ids.len(), st)?;
        let logits = DevBuf::alloc_async(nseq * VOCAB * 4, st)?;
        self.logits(&w, &[ids.len() - 1], 0, VOCAB, &logits, st)?;
        // every image starts from the prompt's last logits
        for i in 1..nseq {
            memcpy_d2d(logits.at(i * VOCAB * 4), logits.ptr(), VOCAB * 4, st)?;
        }
        let mut state = if nseq == 1 { s1 } else { self.spread(&s1, nseq, st)? };
        let dump_path = std::env::var_os("NEMOTRON_DUMP").map(std::path::PathBuf::from);
        let mut dump = std::collections::BTreeMap::new();
        let mut dumped: Vec<u32> = Vec::new();
        let keep = |dump: &mut std::collections::BTreeMap<_, _>, step: u64, first: usize, cols: usize| -> Result<()> {
            let mut b = vec![0u8; cols * 4];
            logits.download(0, &mut b, st)?;
            dump.insert(format!("logits.{step}"), ("F32".to_string(), vec![cols], b));
            dump.insert(format!("first.{step}"), ("I32".to_string(), vec![1], (first as i32).to_le_bytes().to_vec()));
            Ok(())
        };
        if dump_path.is_some() {
            dump.insert("ids".to_string(), ("I32".to_string(), vec![ids.len()], ids.iter().flat_map(|&t| (t as i32).to_le_bytes()).collect()));
            keep(&mut dump, 0, 0, VOCAB)?;
        }
        let t_prompt = t0.elapsed().as_secs_f64();

        let seeds_b: Vec<u8> = r.seeds.iter().flat_map(|s| s.to_le_bytes()).collect();
        let seeds = DevBuf::alloc_async(seeds_b.len(), st)?;
        seeds.upload(0, &seeds_b, st)?;
        let picked = DevBuf::alloc_async(4, st)?;
        let mut phase = vec![Phase::Pre(0); nseq];
        let mut header: Vec<Vec<u32>> = vec![Vec::new(); nseq];
        let mut codes: Vec<Vec<u32>> = vec![Vec::with_capacity(gh * gw); nseq];
        // whether `logits` holds the whole vocabulary (else the visual tokens from column 0)
        let mut full = true;
        let mut step = 0u64;
        while phase.iter().any(|p| *p != Phase::Done) {
            let mut next = vec![IMG_END; nseq];
            for i in 0..nseq {
                // the token this image's logits choose, from `range` of the vocabulary
                let sample = |first: u32, count: usize| -> Result<u32> {
                    let col = if full { first as usize } else { first as usize - VIS0 as usize };
                    let ld = if full { VOCAB } else { NVIS };
                    nk!(gpu_nm_sample(logits.at((i * ld + col) * 4), ld as i64, 1, count as i32, r.temperature, off(seeds.ptr(), i * 8), step, picked.ptr(),
                        s(st)))?;
                    let mut b = [0u8; 4];
                    picked.download(0, &mut b, st)?;
                    Ok(first + i32::from_le_bytes(b) as u32)
                };
                next[i] = match phase[i] {
                    Phase::Pre(k) => {
                        ensure!(full, "internal: whole-vocabulary logits expected");
                        let t = sample(0, VOCAB)?;
                        phase[i] = if t == IMG_START {
                            Phase::Header
                        } else if k + 1 >= MAX_PRE {
                            bail!("Nemotron-Labs-Image wrote no <|img_start|> in {MAX_PRE} tokens: {}", self.decode_text(&header[i]))
                        } else {
                            header[i].push(t);
                            Phase::Pre(k + 1)
                        };
                        if t == IMG_START {
                            header[i].clear();
                        }
                        t
                    }
                    Phase::Header => {
                        let t = sample(0, VOCAB)?;
                        if t == VIS_ROW {
                            self.check_header(&header[i], r.width, r.height)?;
                            phase[i] = Phase::Grid { row: 0, col: 0 };
                        } else {
                            ensure!(!(IMG_START..VIS0 + NVIS as u32).contains(&t), "Nemotron-Labs-Image wrote token {t} in the size header: {}",
                                self.decode_text(&header[i]));
                            header[i].push(t);
                            ensure!(header[i].len() <= MAX_HEADER, "Nemotron-Labs-Image's size header ran past {MAX_HEADER} tokens: {}",
                                self.decode_text(&header[i]));
                        }
                        t
                    }
                    Phase::Grid { row, col } if col < gw => {
                        let t = sample(VIS0, NVIS)?;
                        codes[i].push(t - VIS0);
                        phase[i] = Phase::Grid { row, col: col + 1 };
                        t
                    }
                    Phase::Grid { row, .. } if row + 1 < gh => {
                        phase[i] = Phase::Grid { row: row + 1, col: 0 };
                        VIS_ROW
                    }
                    Phase::Grid { .. } | Phase::Done => {
                        phase[i] = Phase::Done;
                        IMG_END
                    }
                };
            }
            if phase.iter().all(|p| *p == Phase::Done) {
                break;
            }
            step += 1;
            if !progress(codes.iter().map(Vec::len).min().unwrap_or(0), gh * gw) {
                bail!(crate::pipeline::Cancelled);
            }
            if dump_path.is_some() && step <= DUMP_STEPS {
                dumped.push(next[0]);
            }
            self.embed(&next, &w.x, st)?;
            self.forward(&mut state, &w, nseq, 1, st)?;
            // the visual tokens' logits are all a grid row needs
            full = phase.iter().any(|p| matches!(p, Phase::Pre(_) | Phase::Header));
            let rows: Vec<usize> = (0..nseq).collect();
            if full {
                self.logits(&w, &rows, 0, VOCAB, &logits, st)?;
            } else {
                self.logits(&w, &rows, VIS0 as usize, NVIS, &logits, st)?;
            }
            if dump_path.is_some() && step <= DUMP_STEPS {
                if full { keep(&mut dump, step, 0, VOCAB)? } else { keep(&mut dump, step, VIS0 as usize, NVIS)? }
            }
        }
        if let Some(p) = &dump_path {
            dump.insert("tokens".to_string(), ("I32".to_string(), vec![dumped.len()], dumped.iter().flat_map(|&t| (t as i32).to_le_bytes()).collect()));
            crate::steer::write_safetensors(p, &dump)?;
            eprintln!("Nemotron: dump written to {}", p.display());
        }
        let t_tokens = t0.elapsed().as_secs_f64() - t_prompt;
        let (hits, misses) = (self.cache.hits - h0, self.cache.misses - m0);
        eprintln!("Nemotron: prompt of {} tokens {:.1}s, {} tokens x {nseq} in {:.1}s ({:.1} ms a step), experts {:.1}% from VRAM ({:.1} GB copied)",
            ids.len(), t_prompt, step, t_tokens, t_tokens * 1e3 / step.max(1) as f64, 100.0 * hits as f64 / (hits + misses).max(1) as f64,
            (misses * EXPERT as u64) as f64 / 1e9);
        drop((w, state, logits));
        let t1 = Instant::now();
        let images = codes.iter().map(|c| self.decoder.decode(c, gh, gw, st)).collect::<Result<Vec<_>>>()?;
        eprintln!("Nemotron: decoded {} image(s) in {:.0} ms", images.len(), t1.elapsed().as_secs_f64() * 1e3);
        Ok(images)
    }

    fn decode_text(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, false).unwrap_or_else(|_| format!("{ids:?}"))
    }

    /// The model's size header must be the size asked for: the grid's shape comes from it.
    fn check_header(&self, ids: &[u32], width: usize, height: usize) -> Result<()> {
        let text = self.decode_text(ids);
        let num = |key: &str| -> Option<usize> {
            let at = text.find(key)? + key.len();
            text[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
        };
        match (num("height="), num("width=")) {
            (Some(h), Some(w)) if (w, h) == (width, height) => Ok(()),
            (Some(h), Some(w)) => bail!("Nemotron-Labs-Image wrote a {w}x{h} header for a {width}x{height} request: {text:?}"),
            _ => bail!("Nemotron-Labs-Image wrote no size header: {text:?}"),
        }
    }

    /// The share of the experts the VRAM cache holds.
    pub fn cached_experts(&self) -> (usize, usize) {
        (self.cache.owner.len(), self.moe_layers * E)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn grid_needs_multiples_of_16() {
        assert_eq!(super::grid(1024, 880).unwrap(), (55, 64));
        assert!(super::grid(1000, 1024).is_err());
    }
}
