//! MiniMax H3's decoders. Video: a 36-layer ViT (int8 ConvRot weights) over 256px tiles of overlapping
//! 7-latent chunks, blended on the host as ComfyUI does. Audio: BigVGAN at 32 kHz, f32. Sizes come from
//! the latents; no encoder yet, so first and last frame conditioning isn't available.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, PM, PRO_LAYERNORM, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, Plan, WeightSet};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::Result;
use half::{bf16, f16};
use std::sync::Arc;

const C: usize = 24;
const D: usize = 2048;
const HEADS: usize = 32;
const DH: usize = 64;
const QKV: usize = 3 * D;
const FF: usize = 8192;
const LAYERS: usize = 36;
const EPS: f32 = 1e-5;
const PAIRS: usize = 24;
const REGISTERS: usize = 4;
const SUFFIX: usize = REGISTERS + 1;
const TILE: usize = 256;
const TILE_OVERLAP: usize = 64;
const RATIO: usize = 16;
const RATIO_T: usize = 4;
const OUT: usize = 3 * RATIO_T * RATIO * RATIO;
const CHUNK_TOKENS: usize = 5;
const TOKEN_OVERLAP: usize = 2;
const PRE_PADDING: usize = 3;
const FRAME_OVERLAP: usize = 5;
const TOKEN_DROP: usize = 3;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Pixels or latents, [channels, frames, height, width].
#[derive(Clone)]
pub struct Clip {
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Clip {
    fn new(c: usize, t: usize, h: usize, w: usize) -> Self {
        Clip { c, t, h, w, data: vec![0.0; c * t * h * w] }
    }

    fn at(&self, c: usize, t: usize, h: usize, w: usize) -> usize {
        ((c * self.t + t) * self.h + h) * self.w + w
    }

    /// The part at `(t0, h0, w0)` of size `(t, h, w)`.
    fn crop(&self, (t0, t): (usize, usize), (h0, h): (usize, usize), (w0, w): (usize, usize)) -> Clip {
        let mut o = Clip::new(self.c, t, h, w);
        for c in 0..self.c {
            for f in 0..t {
                for y in 0..h {
                    let (d, a) = (o.at(c, f, y, 0), self.at(c, t0 + f, h0 + y, w0));
                    o.data[d..d + w].copy_from_slice(&self.data[a..a + w]);
                }
            }
        }
        o
    }
}

/// `blend`: the first `extent` of `b` along axis `dim` (1 frames, 2 height, 3 width) fades in from the end of `a`.
fn blend(a: &Clip, b: &Clip, extent: usize, dim: usize) -> Clip {
    let len = |x: &Clip| [x.c, x.t, x.h, x.w][dim];
    let extent = extent.min(len(a)).min(len(b));
    let mut o = b.clone();
    let a_from = len(a) - extent;
    for c in 0..b.c {
        for t in 0..b.t {
            for y in 0..b.h {
                for x in 0..b.w {
                    let i = [c, t, y, x][dim];
                    if i >= extent {
                        continue;
                    }
                    let mut p = [c, t, y, x];
                    p[dim] += a_from;
                    let wb = i as f32 / extent as f32;
                    let j = b.at(c, t, y, x);
                    o.data[j] = a.data[a.at(p[0], p[1], p[2], p[3])] * (1.0 - wb) + b.data[j] * wb;
                }
            }
        }
    }
    o
}

/// `split_tiles`: starts of the tiles along an axis of `len` pixels, and the overlaps between them.
pub fn split_tiles(len: usize) -> (Vec<usize>, usize, Vec<usize>) {
    if TILE >= len {
        return (vec![0], len, vec![]);
    }
    let mut n = len.div_ceil(TILE);
    let remaining = loop {
        let r = (TILE * n) as i64 - (TILE_OVERLAP * (n - 1)) as i64 - len as i64;
        if r >= 0 {
            break r as usize;
        }
        n += 1;
    };
    let mut overlaps = vec![TILE_OVERLAP; n - 1];
    for i in 0..remaining / RATIO {
        overlaps[i % (n - 1)] += RATIO;
    }
    let mut starts = vec![0];
    for o in &overlaps {
        starts.push(starts[starts.len() - 1] + TILE - o);
    }
    (starts, TILE, overlaps)
}

/// `_decode_temporal_chunks`: latents to pad with and chunks to decode.
pub fn temporal_chunks(z_len: usize) -> (usize, usize) {
    let pseudo = z_len + TOKEN_DROP;
    let mut pad = (CHUNK_TOKENS - pseudo % CHUNK_TOKENS) % CHUNK_TOKENS;
    let total = pseudo + pad;
    let mut chunks = (total / CHUNK_TOKENS) as i64 - 1;
    if chunks < 1 {
        pad += CHUNK_TOKENS;
        chunks += 1;
    }
    (pad, chunks as usize)
}

fn layouts(st: &SafeTensors) -> Result<Vec<Layout>> {
    let mut v = Vec::new();
    for i in 0..LAYERS {
        let n = |x: &str| format!("decoder.transformer_blocks.{i}.{x}");
        let mut l = Layout::new();
        for (key, m) in [("qkv", "attn.to_qkv"), ("o", "attn.to_out"), ("w1", "ff.w1"), ("w2", "ff.w2")] {
            l.raw(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
            l.raw(st, &format!("s_{key}"), &[n(&format!("{m}.weight_scale"))])?;
            l.bf16(st, &format!("b_{key}"), &[n(&format!("{m}.bias"))])?;
        }
        l.f32(st, "n1", &n("norm1.weight"), 0.0)?;
        l.f32(st, "n2", &n("norm2.weight"), 0.0)?;
        l.bf16(st, "s1", &[n("scale1")])?;
        l.bf16(st, "s2", &[n("scale2")])?;
        v.push(l);
    }
    Ok(v)
}

fn rope(t: usize, h: usize, w: usize) -> Vec<f32> {
    let inv: Vec<f32> = (0..8).map(|i| 1.0 / 100f32.powf(i as f32 * 0.125)).collect();
    let coord = |n: usize| -> Vec<f32> { (0..n).map(|i| f16::from_f32(2.0 * ((i as f32 + 0.5) / n as f32) - 1.0).to_f32()).collect() };
    let (ct, ch, cw) = (coord(t), coord(h), coord(w));
    let l = t * h * w + SUFFIX;
    let mut out = Vec::with_capacity(l * PAIRS * 2);
    let mut row = |p: [f32; 3]| {
        for a in p {
            for f in &inv {
                let ang = 2.0 * std::f32::consts::PI * a * f;
                out.push(bf16::from_f32(ang.cos()).to_f32());
                out.push(bf16::from_f32(ang.sin()).to_f32());
            }
        }
    };
    for a in &ct {
        for b in &ch {
            for c in &cw {
                row([*a, *b, *c]);
            }
        }
    }
    for _ in 0..SUFFIX {
        row([0.0; 3]);
    }
    out
}

pub struct VideoDecoder {
    ws: WeightSet,
    globals: DevBuf,
    gl: Layout,
    mean: Vec<f32>,
    std: Vec<f32>,
    post: Vec<f32>,
    post_bias: Vec<f32>,
    threads: usize,
    prof: crate::prof::Prof,
}

impl VideoDecoder {
    /// The layers take no VRAM until [`Self::place`]. They are read from the file once, then held in RAM.
    pub fn load(path: &std::path::Path, threads: usize, st0: &Stream) -> Result<Self> {
        let st = Arc::new(SafeTensors::open(path)?);
        let ws = WeightSet::lazy("h3 vae", st.clone(), layouts(&st)?, Plan { resident: 0, slots: 0 }, threads)?;
        ws.keep_images();
        let mut gl = Layout::new();
        for (key, n) in [("emb_w", "decoder.x_embedder.weight"), ("emb_b", "decoder.x_embedder.bias"), ("proj_w", "decoder.proj_out.weight"), ("proj_b", "decoder.proj_out.bias"), ("regs", "decoder.register_tokens"), ("no_b", "decoder.norm_out.bias")] {
            gl.bf16(&st, key, &[n.to_string()])?;
        }
        gl.f32(&st, "no_w", "decoder.norm_out.weight", 0.0)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(&st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, st0)?;
        st0.sync()?;
        Ok(VideoDecoder { ws, globals, gl, mean: st.f32_vec("latents_mean", false)?, std: st.f32_vec("latents_std", false)?, post: st.f32_vec("post_quant_conv.weight", false)?, post_bias: st.f32_vec("post_quant_conv.bias", false)?, threads, prof: crate::prof::Prof::new() })
    }

    /// Give the layers `budget` bytes of VRAM.
    pub fn place(&mut self, budget: usize) -> Result<()> {
        self.ws.replan(plan(&self.ws.layouts, budget, 2))
    }

    /// Take the layers out of VRAM, for the diffusion model.
    pub fn park(&mut self) -> Result<()> {
        self.ws.replan(Plan { resident: 0, slots: 0 })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// Raw decoder output of latents [24, t, h, w].
    fn pixels(&mut self, z: &Clip, st: &Stream) -> Result<Clip> {
        self.prof.mark("start", st)?;
        let t0 = std::time::Instant::now();
        let (t, h, w) = (z.t, z.h, z.w);
        let n = t * h * w;
        let l = n + SUFFIX;
        let mut rows = vec![0f32; n * C];
        for c in 0..C {
            for f in 0..t {
                for y in 0..h {
                    for x in 0..w {
                        rows[((f * h + y) * w + x) * C + c] = z.data[z.at(c, f, y, x)];
                    }
                }
            }
        }
        // post_quant_conv, 1 x 1 x 1
        let mut conv = vec![0f32; n * C];
        for (o, r) in conv.chunks_exact_mut(C).zip(rows.chunks_exact(C)) {
            for (c, v) in o.iter_mut().enumerate() {
                *v = self.post_bias[c] + (0..C).map(|j| self.post[c * C + j] * r[j]).sum::<f32>();
            }
        }
        let bytes: Vec<u8> = conv.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect();
        let up = DevBuf::alloc_async(bytes.len(), st)?;
        up.upload(0, &bytes, st)?;
        let x = DevBuf::alloc_async(l * D * 2, st)?;
        let di = D as i64;
        k!(gpu_gemm(up.ptr(), C as i64, self.g("emb_w"), C as i64, x.ptr(), di, self.g("emb_b"), n as i64, di, C as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        crate::cuda::memcpy_d2d(off(x.ptr(), n * D * 2), self.g("regs"), REGISTERS * D * 2, st)?;
        x.upload((n + REGISTERS) * D * 2, &vec![0u8; D * 2], st)?;
        let table = rope(t, h, w);
        let rd = DevBuf::alloc_async(table.len() * 4, st)?;
        rd.upload(0, &f32_bytes(&table), st)?;
        self.prof.host("prep", t0);
        self.prof.mark("embed", st)?;

        let m = l as i32;
        let a8 = DevBuf::alloc_async(l * FF, st)?;
        let asc = DevBuf::alloc_async(l * 4, st)?;
        let qkv = DevBuf::alloc_async(l * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(l * D * 2, st)?;
        let hid = DevBuf::alloc_async(l * 2 * FF * 2, st)?;
        let wn = unsafe { crate::ffi::gpu_int8_attn_workspace_d(1, HEADS as i32, HEADS as i32, m, m, DH as i32) as usize };
        let ws = DevBuf::alloc_async(wn, st)?;
        let (xp, q) = (x.ptr(), qkv.ptr());
        let (rs, hs) = (QKV as i64, (3 * DH) as i64);
        for b in 0..LAYERS {
            let base = self.ws.acquire(b, st)?;
            self.prof.mark("stream", st)?;
            let lay = self.ws.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, wp("n1"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, D as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_qkv"), asc.ptr(), wp("s_qkv"), wp("b_qkv"), q, l as i64, QKV as i64, di, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            k!(gpu_rope_bf16(q, rs, hs, null(), EPS, rd.ptr(), m, HEADS as i32, DH as i32, PAIRS as i32, s(st)))?;
            k!(gpu_rope_bf16(off(q, DH * 2), rs, hs, null(), EPS, rd.ptr(), m, HEADS as i32, DH as i32, PAIRS as i32, s(st)))?;
            self.prof.mark("rope", st)?;
            let (kp, vp) = (off(q, DH * 2), off(q, 2 * DH * 2));
            k!(gpu_int8_attn_d(q, kp, vp, attn.ptr(), ws.ptr(), 1, m, m, HEADS as i32, HEADS as i32, DH as i32, l as i64 * rs, rs, hs, l as i64 * rs, rs, hs, l as i64 * rs, rs, hs, (l * D) as i64, di,
                DH as i64, 1.0 / (DH as f32).sqrt(), s(st)))?;
            self.prof.mark("attn", st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), di, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, D as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_o"), asc.ptr(), wp("s_o"), wp("b_o"), wp("s1"), xp, xp, l as i64, di, di, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, wp("n2"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, D as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_w1"), asc.ptr(), wp("s_w1"), wp("b_w1"), hid.ptr(), l as i64, (2 * FF) as i64, di, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * FF) as i64, null(), 0, FF as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, FF as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_w2"), asc.ptr(), wp("s_w2"), wp("b_w2"), wp("s2"), xp, xp, l as i64, di, FF as i64, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            self.ws.release(b, st)?;
        }
        k!(gpu_prologue_rows(DT_BF16, PRO_LAYERNORM, xp, di, null(), 0, 0, self.g("no_w"), EPS, null(), null(), attn.ptr(), di, n as i32, D as i32, s(st)))?;
        k!(gpu_add_gated_bf16(attn.ptr(), di, self.g("no_b"), 0, null(), n as i32, D as i32, s(st)))?;
        let outb = DevBuf::alloc_async(n * OUT * 2, st)?;
        k!(gpu_gemm(attn.ptr(), di, self.g("proj_w"), di, outb.ptr(), OUT as i64, self.g("proj_b"), n as i64, OUT as i64, di, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        self.prof.mark("head", st)?;
        let t0 = std::time::Instant::now();
        let mut raw = vec![0u8; n * OUT * 2];
        outb.download(0, &mut raw, st)?;
        let mut o = Clip::new(3, t * RATIO_T, h * RATIO, w * RATIO);
        let val = |i: usize| bf16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]).to_f32();
        // one plane per channel and frame, a share of them per thread
        let (pw, frames) = (o.h * o.w, o.t);
        let mut planes: Vec<(usize, &mut [f32])> = o.data.chunks_mut(pw).enumerate().collect();
        let share = planes.len().div_ceil(self.threads.max(1));
        std::thread::scope(|sc| {
            for part in planes.chunks_mut(share) {
                let val = &val;
                sc.spawn(move || {
                    for (i, plane) in part.iter_mut() {
                        let (c, ft) = (*i / frames, *i % frames);
                        let (f, pt) = (ft / RATIO_T, ft % RATIO_T);
                        for y in 0..h {
                            for py in 0..RATIO {
                                for x in 0..w {
                                    let tok = ((f * h + y) * w + x) * OUT + (((c * RATIO_T + pt) * RATIO + py) * RATIO);
                                    let row = (y * RATIO + py) * RATIO * w + x * RATIO;
                                    for px in 0..RATIO {
                                        plane[row + px] = val(tok + px);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
        self.prof.host("unpatch", t0);
        self.prof.report("tile", st)?;
        Ok(o)
    }

    /// `tiled_decode`
    fn tiled(&mut self, z: &Clip, st: &Stream) -> Result<Clip> {
        let (height, width) = (z.h * RATIO, z.w * RATIO);
        let (ys, ylen, yo) = split_tiles(height);
        let (xs, xlen, xo) = split_tiles(width);
        let mut canvas: Option<Clip> = None;
        let mut strip: Option<Clip> = None;
        let mut out_y = 0;
        for (i, ypos) in ys.iter().enumerate() {
            let mut new_strip: Option<Clip> = None;
            let mut left: Option<Clip> = None;
            let (mut out_x, mut tile_h) = (0, 0);
            for (j, xpos) in xs.iter().enumerate() {
                let part = z.crop((0, z.t), (ypos / RATIO, ylen / RATIO), (xpos / RATIO, xlen / RATIO));
                let mut tile = self.pixels(&part, st)?;
                if i > 0 {
                    let above = strip.as_ref().unwrap().crop((0, tile.t), (0, yo[i - 1]), (*xpos, xlen));
                    tile = blend(&above, &tile, yo[i - 1], 2);
                }
                if j > 0 {
                    tile = blend(left.as_ref().unwrap(), &tile, xo[j - 1], 3);
                }
                left = (j + 1 < xs.len()).then(|| tile.crop((0, tile.t), (0, tile.h), (tile.w - xo[j], xo[j])));
                if j + 1 < xs.len() {
                    tile = tile.crop((0, tile.t), (0, tile.h), (0, tile.w - xo[j]));
                }
                let canvas = canvas.get_or_insert_with(|| Clip::new(3, tile.t, height, width));
                if i + 1 < ys.len() {
                    let ns = new_strip.get_or_insert_with(|| Clip::new(3, tile.t, yo[i], width));
                    let tail = tile.crop((0, tile.t), (tile.h - yo[i], yo[i]), (0, tile.w));
                    paste(ns, &tail, 0, out_x);
                    tile = tile.crop((0, tile.t), (0, tile.h - yo[i]), (0, tile.w));
                }
                paste(canvas, &tile, out_y, out_x);
                tile_h = tile.h;
                out_x += tile.w;
            }
            strip = new_strip;
            out_y += tile_h;
        }
        Ok(canvas.unwrap())
    }

    /// Latents [24, t, h, w] -> RGB f32 [frames, 3, h, w] in 0..1, with `frames` from the clip's length.
    pub fn decode(&mut self, latent: &[f32], t: usize, h: usize, w: usize, frames: usize, st: &Stream) -> Result<Clip> {
        let mut z = Clip { c: C, t, h, w, data: latent.to_vec() };
        for c in 0..C {
            let a = z.at(c, 0, 0, 0);
            for v in &mut z.data[a..a + t * h * w] {
                *v = *v * self.std[c] + self.mean[c];
            }
        }
        if t == 1 {
            // A still decodes badly alone: like ComfyUI, decode a chunk of copies and keep its first real frame.
            let mut copies = Clip::new(C, CHUNK_TOKENS, h, w);
            for c in 0..C {
                let src = z.at(c, 0, 0, 0);
                for f in 0..CHUNK_TOKENS {
                    let dst = copies.at(c, f, 0, 0);
                    copies.data[dst..dst + h * w].copy_from_slice(&z.data[src..src + h * w]);
                }
            }
            let dec = self.tiled(&copies, st)?;
            let mut out = dec.crop((PRE_PADDING, 1), (0, dec.h), (0, dec.w));
            for c in 0..3 {
                let a = out.at(c, 0, 0, 0);
                for v in &mut out.data[a..a + out.h * out.w] {
                    *v = (*v * STD[c] + MEAN[c]).clamp(0.0, 1.0);
                }
            }
            return Ok(out);
        }
        let (pad, chunks) = temporal_chunks(t);
        let total = t + pad;
        let mut padded = Clip::new(C, total, h, w);
        for c in 0..C {
            for f in 0..total {
                let src = z.at(c, f.min(t - 1), 0, 0);
                let dst = padded.at(c, f, 0, 0);
                padded.data[dst..dst + h * w].copy_from_slice(&z.data[src..src + h * w]);
            }
        }
        let chunk_dec = CHUNK_TOKENS * RATIO_T;
        let mut out = Clip::new(3, frames, h * RATIO, w * RATIO);
        let mut pos = 0;
        let mut overlap: Option<Clip> = None;
        let mut write = |part: Clip, pos: &mut usize| {
            let n = part.t.min(frames.saturating_sub(*pos));
            for c in 0..3 {
                for f in 0..n {
                    for y in 0..part.h {
                        for x in 0..part.w {
                            let v = part.data[part.at(c, f, y, x)] * STD[c] + MEAN[c];
                            let at = out.at(c, *pos + f, y, x);
                            out.data[at] = v.clamp(0.0, 1.0);
                        }
                    }
                }
            }
            *pos += n;
        };
        for i in 0..chunks {
            let start = i * CHUNK_TOKENS;
            let end = (start + CHUNK_TOKENS + TOKEN_OVERLAP).min(total);
            let clip = padded.crop((start, end - start), (0, h), (0, w));
            let dec = self.tiled(&clip, st)?;
            for j in 0..2 {
                let from = j * chunk_dec;
                let to = (from + chunk_dec).min(dec.t);
                let first = from + PRE_PADDING;
                if to <= first {
                    continue;
                }
                let mut part = dec.crop((first, to - first), (0, dec.h), (0, dec.w));
                if j == 0 {
                    if let Some(o) = overlap.take() {
                        part = blend(&o, &part, FRAME_OVERLAP, 1);
                    }
                    write(part, &mut pos);
                } else {
                    overlap = Some(part);
                }
            }
            if i + 1 == chunks {
                if let Some(o) = overlap.take() {
                    write(o, &mut pos);
                }
            }
        }
        Ok(out)
    }
}

fn paste(to: &mut Clip, p: &Clip, y0: usize, x0: usize) {
    for c in 0..p.c {
        for t in 0..p.t {
            for y in 0..p.h {
                let (a, b) = (to.at(c, t, y0 + y, x0), p.at(c, t, y, 0));
                to.data[a..a + p.w].copy_from_slice(&p.data[b..b + p.w]);
            }
        }
    }
}

const UPS: [(usize, usize); 7] = [(5, 9), (5, 9), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)];
const KERNELS: [usize; 3] = [3, 7, 11];
const DILATIONS: [usize; 3] = [1, 3, 5];

/// The audio decoder's weights, held in RAM between clips.
pub struct AudioDecoder {
    gl: Layout,
    host: Vec<u8>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl AudioDecoder {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let f = SafeTensors::open(path)?;
        let mut gl = Layout::new();
        for name in f.tensors.keys() {
            if name.starts_with("decoder.") || name.starts_with("dec_in_proj.") {
                gl.raw(&f, name, std::slice::from_ref(name))?;
            }
        }
        let mut host = vec![0u8; gl.size];
        gl.pack(&f, &mut host)?;
        Ok(AudioDecoder { gl, host, mean: f.f32_vec("latents_mean", false)?, std: f.f32_vec("latents_std", false)? })
    }
}

/// Audio latents [32, 2, t] -> stereo samples [2, 800 t], in -1..1.
pub fn decode_audio(dec: &AudioDecoder, latent: &[f32], t: usize, st: &Stream) -> Result<Vec<f32>> {
    let (gl, mean, std) = (&dec.gl, &dec.mean, &dec.std);
    let w = DevBuf::alloc_async(gl.size, st)?;
    w.upload(0, &dec.host, st)?;
    let g = |n: &str| w.at(gl.off(n));
    let mut z = vec![0f32; 2 * 32 * t];
    for b in 0..2 {
        for c in 0..32 {
            for i in 0..t {
                z[(b * 32 + c) * t + i] = latent[(c * 2 + b) * t + i] * std[c] + mean[c];
            }
        }
    }
    let x0 = DevBuf::alloc_async(z.len() * 4, st)?;
    x0.upload(0, &f32_bytes(&z), st)?;
    let conv = |x: PM, w_: &str, b_: Option<&str>, ci: usize, co: usize, t: usize, kk: usize, dil: usize| -> Result<DevBuf> {
        let o = DevBuf::alloc_async(2 * co * t * 4, st)?;
        let bias: crate::ffi::P = b_.map_or(null(), |n| g(n) as crate::ffi::P);
        k!(gpu_conv1d(x, g(w_), bias, o.ptr(), 2, ci as i32, co as i32, t as i32, kk as i32, dil as i32, ((kk - 1) * dil / 2) as i32, s(st)))?;
        Ok(o)
    };
    let act = |x: &DevBuf, p: &str, c: usize, t: usize| -> Result<()> {
        let tmp = DevBuf::alloc_async(2 * c * 2 * t * 4, st)?;
        k!(gpu_aa_snake(x.ptr(), g(&format!("{p}.act.alpha")), g(&format!("{p}.act.beta")), g(&format!("{p}.upsample.filter")), g(&format!("{p}.downsample.lowpass.filter")), tmp.ptr(), 2, c as i32, t as i32, s(st)))
    };
    let mut x = conv(x0.ptr(), "dec_in_proj.weight", Some("dec_in_proj.bias"), 32, 2048, t, 1, 1)?;
    x = conv(x.ptr(), "decoder.conv_pre.weight", Some("decoder.conv_pre.bias"), 2048, 1024, t, 7, 1)?;
    let (mut ch, mut len) = (1024, t);
    for (i, (u, kk)) in UPS.iter().enumerate() {
        let co = ch / 2;
        let tout = (len - 1) * u + kk - 2 * ((kk - u) / 2);
        let o = DevBuf::alloc_async(2 * co * tout * 4, st)?;
        k!(gpu_conv_t1d(x.ptr(), g(&format!("decoder.ups.{i}.0.weight")), g(&format!("decoder.ups.{i}.0.bias")), o.ptr(), 2, ch as i32, co as i32, len as i32, *kk as i32, *u as i32, ((kk - u) / 2) as i32, s(st)))?;
        (x, ch, len) = (o, co, tout);
        let n = (2 * ch * len) as i64;
        let sum = DevBuf::alloc_async(n as usize * 4, st)?;
        for (j, ks) in KERNELS.iter().enumerate() {
            let r = format!("decoder.resblocks.{}", i * 3 + j);
            let cur = DevBuf::alloc_async(n as usize * 4, st)?;
            crate::cuda::memcpy_d2d(cur.ptr(), x.ptr(), n as usize * 4, st)?;
            for (l, d) in DILATIONS.iter().enumerate() {
                let xt = DevBuf::alloc_async(n as usize * 4, st)?;
                crate::cuda::memcpy_d2d(xt.ptr(), cur.ptr(), n as usize * 4, st)?;
                act(&xt, &format!("{r}.activations.{}", l * 2), ch, len)?;
                let c1 = conv(xt.ptr(), &format!("{r}.convs1.{l}.weight"), Some(&format!("{r}.convs1.{l}.bias")), ch, ch, len, *ks, *d)?;
                act(&c1, &format!("{r}.activations.{}", l * 2 + 1), ch, len)?;
                let c2 = conv(c1.ptr(), &format!("{r}.convs2.{l}.weight"), Some(&format!("{r}.convs2.{l}.bias")), ch, ch, len, *ks, 1)?;
                let (p, co) = ([c2.ptr() as crate::ffi::P, cur.ptr() as crate::ffi::P], [1.0f32, 1.0]);
                k!(gpu_lincomb(cur.ptr(), n, 2, p.as_ptr(), co.as_ptr(), s(st)))?;
            }
            if j == 0 {
                crate::cuda::memcpy_d2d(sum.ptr(), cur.ptr(), n as usize * 4, st)?;
            } else {
                let (p, co) = ([sum.ptr() as crate::ffi::P, cur.ptr() as crate::ffi::P], [1.0f32, 1.0]);
                k!(gpu_lincomb(sum.ptr(), n, 2, p.as_ptr(), co.as_ptr(), s(st)))?;
            }
        }
        let (p, co) = ([sum.ptr() as crate::ffi::P], [1.0f32 / 3.0]);
        k!(gpu_lincomb(sum.ptr(), n, 1, p.as_ptr(), co.as_ptr(), s(st)))?;
        x = sum;
    }
    act(&x, "decoder.activation_post", ch, len)?;
    let o = conv(x.ptr(), "decoder.conv_post.weight", None, ch, 1, len, 7, 1)?;
    let mut raw = vec![0u8; 2 * len * 4];
    o.download(0, &mut raw, st)?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).clamp(-1.0, 1.0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_follow_comfyui() {
        assert_eq!(split_tiles(256), (vec![0], 256, vec![]));
        let (starts, size, overlaps) = split_tiles(512);
        assert_eq!((size, overlaps.len()), (256, 2));
        assert_eq!(*starts.last().unwrap() + size, 512);
    }

    #[test]
    fn short_clips_pad_to_one_chunk() {
        assert_eq!(temporal_chunks(2), (5, 1));
    }

    #[test]
    fn blend_fades_across_the_overlap() {
        let a = Clip { c: 1, t: 1, h: 1, w: 4, data: vec![9.0, 9.0, 1.0, 1.0] };
        let b = Clip { c: 1, t: 1, h: 1, w: 3, data: vec![3.0, 3.0, 3.0] };
        assert_eq!(blend(&a, &b, 2, 3).data, vec![1.0, 2.0, 3.0]);
    }
}
