//! MiniMax H3's decoders. Video: a 36-layer ViT (int8 ConvRot weights) over 256px tiles of overlapping
//! 7-latent chunks, blended on the host as ComfyUI does. MiniMax-H3-Image's: the same ViT at half the width and 24
//! layers, fp16 weights, one frame per latent and rope over height and width only. Audio: BigVGAN at 32 kHz, f32.
//! Sizes come from the latents. The encoder, which both VAE files share, encodes single frames: reference pictures
//! and keyframes.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, DT_F16, PM, PRO_LAYERNORM, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::offload::{plan, Layout, Plan, WeightSet};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::Result;
use half::{bf16, f16};
use std::sync::Arc;

const C: usize = 24;
const DH: usize = 64;
const EPS: f32 = 1e-5;
const REGISTERS: usize = 4;
const SUFFIX: usize = REGISTERS + 1;
const TILE: usize = 256;
const TILE_OVERLAP: usize = 64;
/// Tiles the image decoder runs through its layers at once: about 10 MB of activations each.
const TILE_BATCH: usize = 32;
const RATIO: usize = 16;

/// One of the ViT decoders.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Arch {
    d: usize,
    heads: usize,
    layers: usize,
    /// frames per latent
    ratio_t: usize,
    /// latent axes the rope covers, the trailing ones of (t, h, w)
    axes: usize,
    /// int8 ConvRot weights, else fp16 run as bf16
    int8: bool,
}

const VIDEO: Arch = Arch { d: 2048, heads: 32, layers: 36, ratio_t: 4, axes: 3, int8: true };
const IMAGE: Arch = Arch { d: 1024, heads: 16, layers: 24, ratio_t: 1, axes: 2, int8: false };

impl Arch {
    fn qkv(&self) -> usize {
        3 * self.d
    }

    fn ff(&self) -> usize {
        4 * self.d
    }

    /// rope pairs per head: 8 frequencies per axis
    fn pairs(&self) -> usize {
        8 * self.axes
    }

    fn out(&self) -> usize {
        3 * self.ratio_t * RATIO * RATIO
    }
}

/// `LATENTS_MEAN` and `LATENTS_STD`, which MiniMax-H3-Image's VAE file doesn't carry.
const LATENTS_MEAN: [f32; C] = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468,
    -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765,
    -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
];
const LATENTS_STD: [f32; C] = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.96531379222869875,
    1.05698859691619875, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049925,
    0.7197399735450745, 0.69362932443618775, 2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265, 3.276226282119751,
    3.1627357006073, 2.28168129920959475, 2.6127843856811525,
];
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

fn layouts(st: &mut SafeTensors, a: Arch) -> Result<Vec<Layout>> {
    let mut v = Vec::new();
    for i in 0..a.layers {
        let n = |x: &str| format!("decoder.transformer_blocks.{i}.{x}");
        let mut l = Layout::new();
        if a.int8 {
            for (key, m) in [("qkv", "attn.to_qkv"), ("o", "attn.to_out"), ("w1", "ff.w1"), ("w2", "ff.w2")] {
                l.raw(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
                l.raw(st, &format!("s_{key}"), &[n(&format!("{m}.weight_scale"))])?;
                l.bf16(st, &format!("b_{key}"), &[n(&format!("{m}.bias"))])?;
            }
        } else {
            // to_qkv holds each head's q, k and v together; the exact attention kernel reads [q; k; v] across heads
            for (key, src) in [("w_qkv", n("attn.to_qkv.weight")), ("b_qkv", n("attn.to_qkv.bias"))] {
                let mut parts = Vec::new();
                for part in 0..3 {
                    for h in 0..a.heads {
                        let name = format!("{src}#{part}.{h}");
                        st.alias_rows(&src, &name, (h * 3 + part) * DH, DH)?;
                        parts.push(name);
                    }
                }
                l.bf16(st, key, &parts)?;
            }
            for (key, m) in [("o", "attn.to_out"), ("w1", "ff.w1"), ("w2", "ff.w2")] {
                l.bf16(st, &format!("w_{key}"), &[n(&format!("{m}.weight"))])?;
                l.bf16(st, &format!("b_{key}"), &[n(&format!("{m}.bias"))])?;
            }
        }
        l.f32(st, "n1", &n("norm1.weight"), 0.0)?;
        l.f32(st, "n2", &n("norm2.weight"), 0.0)?;
        l.bf16(st, "s1", &[n("scale1")])?;
        l.bf16(st, "s2", &[n("scale2")])?;
        v.push(l);
    }
    Ok(v)
}

/// The rope table of `t * h * w` tokens and the suffix, over the trailing `axes` of (t, h, w). The video decoder's
/// table is in bf16 steps, the image decoder's in fp16 like its activations.
fn rope(t: usize, h: usize, w: usize, axes: usize) -> Vec<f32> {
    let inv: Vec<f32> = (0..8).map(|i| 1.0 / 100f32.powf(i as f32 * 0.125)).collect();
    let coord = |n: usize| -> Vec<f32> { (0..n).map(|i| f16::from_f32(2.0 * ((i as f32 + 0.5) / n as f32) - 1.0).to_f32()).collect() };
    let (ct, ch, cw) = (coord(t), coord(h), coord(w));
    let round = |v: f32| if axes == 3 { bf16::from_f32(v).to_f32() } else { f16::from_f32(v).to_f32() };
    let l = t * h * w + SUFFIX;
    let mut out = Vec::with_capacity(l * 8 * axes * 2);
    let mut row = |p: [f32; 3]| {
        for a in &p[3 - axes..] {
            for f in &inv {
                let ang = 2.0 * std::f32::consts::PI * a * f;
                out.push(round(ang.cos()));
                out.push(round(ang.sin()));
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
    arch: Arch,
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
        let mut file = SafeTensors::open(path)?;
        let arch = if file.has("decoder.x_embedder.weight") && file.info("decoder.x_embedder.weight")?.shape[0] == IMAGE.d { IMAGE } else { VIDEO };
        if arch == IMAGE && file.info("decoder.proj_out.weight")?.shape[0] != IMAGE.out() {
            anyhow::bail!("{}: a 1024-wide H3 decoder that isn't MiniMax-H3-Image's", path.display());
        }
        let lay = layouts(&mut file, arch)?;
        let st = Arc::new(file);
        let ws = WeightSet::lazy("h3 vae", st.clone(), lay, Plan { resident: 0, slots: 0 }, threads)?;
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
        let (mean, std) = if st.has("latents_mean") {
            (st.f32_vec("latents_mean", false)?, st.f32_vec("latents_std", false)?)
        } else if arch == IMAGE {
            (LATENTS_MEAN.to_vec(), LATENTS_STD.to_vec())
        } else {
            anyhow::bail!("{}: no latents_mean", path.display());
        };
        Ok(VideoDecoder { arch, ws, globals, gl, mean, std, post: st.f32_vec("post_quant_conv.weight", false)?, post_bias: st.f32_vec("post_quant_conv.bias", false)?, threads, prof: crate::prof::Prof::new() })
    }

    /// MiniMax-H3-Image's decoder: one frame per latent.
    pub fn is_image(&self) -> bool {
        self.arch == IMAGE
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

    /// Raw decoder output of latents [24, t, h, w]. Clips of one size go through the layers together, each attending
    /// within itself; the int8 decoder takes them one at a time.
    fn pixels(&mut self, zs: &[Clip], st: &Stream) -> Result<Vec<Clip>> {
        let a = self.arch;
        if a.int8 && zs.len() > 1 {
            return zs.iter().map(|z| Ok(self.pixels(std::slice::from_ref(z), st)?.remove(0))).collect();
        }
        self.prof.mark("start", st)?;
        let t0 = std::time::Instant::now();
        let nb = zs.len();
        let (t, h, w) = (zs[0].t, zs[0].h, zs[0].w);
        anyhow::ensure!(zs.iter().all(|z| (z.t, z.h, z.w) == (t, h, w)), "tiles of different sizes in one batch");
        let n = t * h * w;
        let l = n + SUFFIX;
        let rows_all = nb * l;
        let mut bytes = Vec::with_capacity(nb * n * C * 2);
        for z in zs {
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
            for r in rows.chunks_exact(C) {
                for c in 0..C {
                    let v = self.post_bias[c] + (0..C).map(|j| self.post[c * C + j] * r[j]).sum::<f32>();
                    bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
                }
            }
        }
        let up = DevBuf::alloc_async(bytes.len(), st)?;
        up.upload(0, &bytes, st)?;
        let (d, heads, qkvn, ff, out_n) = (a.d, a.heads, a.qkv(), a.ff(), a.out());
        let x = DevBuf::alloc_async(rows_all * d * 2, st)?;
        let di = d as i64;
        // each clip's embedding is its own GEMM, as it is when it decodes alone: cuBLAS picks its algorithm by the rows
        for i in 0..nb {
            let xi = off(x.ptr(), i * l * d * 2);
            k!(gpu_gemm(off(up.ptr(), i * n * C * 2), C as i64, self.g("emb_w"), C as i64, xi, di, self.g("emb_b"), n as i64, di, C as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0,
                s(st)))?;
            crate::cuda::memcpy_d2d(off(xi, n * d * 2), self.g("regs"), REGISTERS * d * 2, st)?;
            x.upload((i * l + n + REGISTERS) * d * 2, &vec![0u8; d * 2], st)?;
        }
        // the rope kernel reads a row of the table per token, so every clip gets its copy
        let table = rope(t, h, w, a.axes).repeat(nb);
        let rd = DevBuf::alloc_async(table.len() * 4, st)?;
        rd.upload(0, &f32_bytes(&table), st)?;
        self.prof.host("prep", t0);
        self.prof.mark("embed", st)?;

        let m = rows_all as i32;
        let qkv = DevBuf::alloc_async(rows_all * qkvn * 2, st)?;
        let attn = DevBuf::alloc_async(rows_all * d * 2, st)?;
        let hid = DevBuf::alloc_async(rows_all * 2 * ff * 2, st)?;
        let (xp, q) = (x.ptr(), qkv.ptr());
        if !a.int8 {
            // bf16 GEMMs and exact attention (FlashAttention-2), q, k and v each across all heads
            let hb = DevBuf::alloc_async(rows_all * ff * 2, st)?;
            let y = DevBuf::alloc_async(rows_all * d * 2, st)?;
            let lse = DevBuf::alloc_async(nb * heads * l * 4, st)?;
            let gemm = |x: PM, k: usize, w: PM, bias: PM, out: PM, nn: usize| -> Result<()> {
                k!(gpu_gemm(x, k as i64, w, k as i64, out, nn as i64, bias, rows_all as i64, nn as i64, k as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))
            };
            for b in 0..a.layers {
                let base = self.ws.acquire(b, st)?;
                self.prof.mark("stream", st)?;
                let lay = self.ws.layouts[b].clone();
                let wp = |key: &str| off(base, lay.off(key));
                k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, wp("n1"), EPS, null(), null(), hb.ptr(), di, m, d as i32, s(st)))?;
                gemm(hb.ptr(), d, wp("w_qkv"), wp("b_qkv"), q, qkvn)?;
                k!(gpu_rope_bf16(q, qkvn as i64, DH as i64, null(), EPS, rd.ptr(), m, heads as i32, DH as i32, a.pairs() as i32, s(st)))?;
                k!(gpu_rope_bf16(off(q, d * 2), qkvn as i64, DH as i64, null(), EPS, rd.ptr(), m, heads as i32, DH as i32, a.pairs() as i32, s(st)))?;
                let (bs, rs, hs) = ((l * qkvn) as i64, qkvn as i64, DH as i64);
                k!(gpu_flash_attn_d(q, off(q, d * 2), off(q, 2 * d * 2), attn.ptr(), lse.ptr(), nb as i32, l as i32, l as i32, heads as i32, heads as i32, DH as i32, bs, rs, hs,
                    bs, rs, hs, bs, rs, hs, (l * d) as i64, di, hs, 1.0 / (DH as f32).sqrt(), 0, s(st)))?;
                self.prof.mark("attn", st)?;
                gemm(attn.ptr(), d, wp("w_o"), wp("b_o"), y.ptr(), d)?;
                k!(gpu_add_gated_bf16(xp, di, y.ptr(), di, wp("s1"), m, d as i32, s(st)))?;
                k!(gpu_prologue_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, wp("n2"), EPS, null(), null(), hb.ptr(), di, m, d as i32, s(st)))?;
                gemm(hb.ptr(), d, wp("w_w1"), wp("b_w1"), hid.ptr(), 2 * ff)?;
                k!(gpu_prologue_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * ff) as i64, null(), 0, ff as i64, null(), 0.0, null(), null(), hb.ptr(), ff as i64, m, ff as i32,
                    s(st)))?;
                gemm(hb.ptr(), ff, wp("w_w2"), wp("b_w2"), y.ptr(), d)?;
                k!(gpu_add_gated_bf16(xp, di, y.ptr(), di, wp("s2"), m, d as i32, s(st)))?;
                self.prof.mark("gemm", st)?;
                self.ws.release(b, st)?;
            }
        }
        let a8 = DevBuf::alloc_async(if a.int8 { l * ff } else { 0 }, st)?;
        let asc = DevBuf::alloc_async(l * 4, st)?;
        let wn = if a.int8 { unsafe { crate::ffi::gpu_int8_attn_workspace_d(1, heads as i32, heads as i32, m, m, DH as i32) as usize } } else { 0 };
        let ws = DevBuf::alloc_async(wn, st)?;
        let (rs, hs) = (qkvn as i64, (3 * DH) as i64);
        for b in 0..if a.int8 { a.layers } else { 0 } {
            let base = self.ws.acquire(b, st)?;
            self.prof.mark("stream", st)?;
            let lay = self.ws.layouts[b].clone();
            let wp = |key: &str| off(base, lay.off(key));
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, wp("n1"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, d as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_qkv"), asc.ptr(), wp("s_qkv"), wp("b_qkv"), q, l as i64, qkvn as i64, di, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            k!(gpu_rope_bf16(q, rs, hs, null(), EPS, rd.ptr(), m, heads as i32, DH as i32, a.pairs() as i32, s(st)))?;
            k!(gpu_rope_bf16(off(q, DH * 2), rs, hs, null(), EPS, rd.ptr(), m, heads as i32, DH as i32, a.pairs() as i32, s(st)))?;
            self.prof.mark("rope", st)?;
            let (kp, vp) = (off(q, DH * 2), off(q, 2 * DH * 2));
            k!(gpu_int8_attn_d(q, kp, vp, attn.ptr(), ws.ptr(), 1, m, m, heads as i32, heads as i32, DH as i32, l as i64 * rs, rs, hs, l as i64 * rs, rs, hs, l as i64 * rs, rs, hs, (l * d) as i64, di,
                DH as i64, 1.0 / (DH as f32).sqrt(), s(st)))?;
            self.prof.mark("attn", st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, attn.ptr(), di, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, d as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_o"), asc.ptr(), wp("s_o"), wp("b_o"), wp("s1"), xp, xp, l as i64, di, di, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_RMS, xp, di, null(), 0, 0, wp("n2"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, d as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm(a8.ptr(), wp("w_w1"), asc.ptr(), wp("s_w1"), wp("b_w1"), hid.ptr(), l as i64, (2 * ff) as i64, di, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * ff) as i64, null(), 0, ff as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, ff as i32, 1, s(st)))?;
            self.prof.mark("quant", st)?;
            k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_w2"), asc.ptr(), wp("s_w2"), wp("b_w2"), wp("s2"), xp, xp, l as i64, di, ff as i64, DT_BF16, s(st)))?;
            self.prof.mark("gemm", st)?;
            self.ws.release(b, st)?;
        }
        // the suffix rows are normed too and never read
        k!(gpu_prologue_rows(DT_BF16, PRO_LAYERNORM, xp, di, null(), 0, 0, self.g("no_w"), EPS, null(), null(), attn.ptr(), di, m, d as i32, s(st)))?;
        k!(gpu_add_gated_bf16(attn.ptr(), di, self.g("no_b"), 0, null(), m, d as i32, s(st)))?;
        let outb = DevBuf::alloc_async(nb * n * out_n * 2, st)?;
        for i in 0..nb {
            k!(gpu_gemm(off(attn.ptr(), i * l * d * 2), di, self.g("proj_w"), di, off(outb.ptr(), i * n * out_n * 2), out_n as i64, self.g("proj_b"), n as i64, out_n as i64, di, 1,
                DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        }
        self.prof.mark("head", st)?;
        let t0 = std::time::Instant::now();
        let mut all = vec![0u8; nb * n * out_n * 2];
        outb.download(0, &mut all, st)?;
        let ratio_t = a.ratio_t;
        let mut clips = Vec::with_capacity(nb);
        for raw in all.chunks_exact(n * out_n * 2) {
        let mut o = Clip::new(3, t * ratio_t, h * RATIO, w * RATIO);
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
                        let (f, pt) = (ft / ratio_t, ft % ratio_t);
                        for y in 0..h {
                            for py in 0..RATIO {
                                for x in 0..w {
                                    let tok = ((f * h + y) * w + x) * out_n + (((c * ratio_t + pt) * RATIO + py) * RATIO);
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
        clips.push(o);
        }
        self.prof.host("unpatch", t0);
        self.prof.report("tiles", st)?;
        Ok(clips)
    }

    /// `tiled_decode`
    fn tiled(&mut self, z: &Clip, st: &Stream) -> Result<Clip> {
        let (height, width) = (z.h * RATIO, z.w * RATIO);
        let (ys, ylen, yo) = split_tiles(height);
        let (xs, xlen, xo) = split_tiles(width);
        let mut canvas: Option<Clip> = None;
        let mut strip: Option<Clip> = None;
        let mut out_y = 0;
        // the tiles in the order they are blended, decoded a batch ahead
        let places: Vec<(usize, usize)> = ys.iter().flat_map(|y| xs.iter().map(move |x| (*y, *x))).collect();
        let batch = if self.arch.int8 { 1 } else { TILE_BATCH };
        let mut decoded = std::collections::VecDeque::new();
        for (i, _) in ys.iter().enumerate() {
            let mut new_strip: Option<Clip> = None;
            let mut left: Option<Clip> = None;
            let (mut out_x, mut tile_h) = (0, 0);
            for (j, xpos) in xs.iter().enumerate() {
                if decoded.is_empty() {
                    let first = i * xs.len() + j;
                    let parts: Vec<Clip> = places[first..(first + batch).min(places.len())].iter()
                        .map(|(y, x)| z.crop((0, z.t), (y / RATIO, ylen / RATIO), (x / RATIO, xlen / RATIO))).collect();
                    decoded.extend(self.pixels(&parts, st)?);
                }
                let mut tile: Clip = decoded.pop_front().unwrap();
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
        if self.arch == IMAGE {
            if t != 1 {
                anyhow::bail!("MiniMax-H3-Image's VAE decodes single-frame latents only, got {t}");
            }
            let mut out = self.tiled(&z, st)?;
            for c in 0..3 {
                let a = out.at(c, 0, 0, 0);
                for v in &mut out.data[a..a + out.h * out.w] {
                    *v = (*v * STD[c] + MEAN[c]).clamp(0.0, 1.0);
                }
            }
            return Ok(out);
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
        let chunk_dec = CHUNK_TOKENS * VIDEO.ratio_t;
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

// ------------------------------------------------------------------------------------------------
// The encoder, for one frame. Both H3 VAE files carry the same 3D causal CNN. On one frame each causal conv is a 2D conv
// with its kernel's last temporal tap, its spatial padding reflects, and GroupNorm takes its statistics per frame. It
// runs in fp16 like ComfyUI's, on 256px tiles whose moments blend on the host.

/// Channels of the six levels; the first four halve the size.
const ENC_CH: [usize; 6] = [128, 256, 256, 512, 512, 1024];
const ENC_DOWN: usize = 4;
const ENC_GROUPS: i32 = 32;
const ENC_EPS: f32 = 1e-6;
/// RGB padded to 8 channels for the GEMM's alignment
const ENC_RGB: usize = 8;
/// `IMAGE_ENCODE_SEED`: the image VAE conditions on a posterior sample drawn with this seed
const IMAGE_ENCODE_SEED: u64 = 42;

/// Conv weight [Cout, Cin, T, R, R] -> [Cout][R][R][Cin] f16 bytes of the last temporal tap, Cin zero-padded to `cin`.
fn enc_conv(st: &SafeTensors, name: &str, cin: usize) -> Result<Vec<u8>> {
    let info = st.info(name)?.clone();
    let [co, ci, t, r, r2] = info.shape[..] else { anyhow::bail!("{name}: a conv weight of shape {:?}", info.shape) };
    anyhow::ensure!(r == r2 && ci <= cin, "{name}: a conv weight of shape {:?}", info.shape);
    let v = st.f32_vec(name, false)?;
    let mut out = vec![f16::ZERO; co * r * r * cin];
    for o in 0..co {
        for c in 0..ci {
            for y in 0..r {
                for x in 0..r {
                    out[((o * r + y) * r + x) * cin + c] = f16::from_f32(v[(((o * ci + c) * t + t - 1) * r + y) * r + x]);
                }
            }
        }
    }
    Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn f16_bytes(st: &SafeTensors, name: &str) -> Result<Vec<u8>> {
    Ok(st.f32_vec(name, false)?.iter().flat_map(|v| f16::from_f32(*v).to_le_bytes()).collect())
}

/// MiniMax H3's VAE encoder, for images: reference pictures and keyframes.
pub struct Encoder {
    buf: DevBuf,
    lay: Layout,
    /// MiniMax-H3-Image's VAE samples the posterior; the video VAE takes its mean
    sample: bool,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Encoder {
    /// The encoder of an H3 VAE file, held in VRAM until dropped.
    pub fn load(path: &std::path::Path, st0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        if !st.has("encoder.conv_in.weight") {
            anyhow::bail!("{}: this VAE file has no encoder", path.display());
        }
        for l in 0..ENC_CH.len() {
            for i in 0..2 {
                if !st.has(&format!("encoder.down.{l}.block.{i}.conv1.weight")) {
                    anyhow::bail!("{}: a pruned H3 encoder (LightVAE Lite), which isn't supported", path.display());
                }
            }
        }
        let mut lay = Layout::new();
        let mut conv = |key: &str, name: &str, cin: usize| -> Result<()> {
            lay.bytes(&format!("{key}.w"), enc_conv(&st, &format!("{name}.weight"), cin)?);
            lay.bytes(&format!("{key}.b"), f16_bytes(&st, &format!("{name}.bias"))?);
            Ok(())
        };
        conv("in", "encoder.conv_in", ENC_RGB)?;
        let mut cin = ENC_CH[0];
        for (l, &c) in ENC_CH.iter().enumerate() {
            for i in 0..2 {
                let p = format!("encoder.down.{l}.block.{i}");
                conv(&format!("{l}.{i}.c1"), &format!("{p}.conv1"), cin)?;
                conv(&format!("{l}.{i}.c2"), &format!("{p}.conv2"), c)?;
                if cin != c {
                    conv(&format!("{l}.{i}.sc"), &format!("{p}.nin_shortcut"), cin)?;
                }
                cin = c;
            }
            if l < ENC_DOWN {
                conv(&format!("{l}.down"), &format!("encoder.down.{l}.downsample.conv"), c)?;
            }
        }
        conv("out", "encoder.conv_out", ENC_CH[5])?;
        conv("quant", "quant_conv", 2 * C)?;
        for l in 0..ENC_CH.len() {
            for i in 0..2 {
                for n in ["norm1", "norm2"] {
                    for (s, wb) in [("w", "weight"), ("b", "bias")] {
                        lay.bytes(&format!("{l}.{i}.{n}.{s}"), f16_bytes(&st, &format!("encoder.down.{l}.block.{i}.{n}.{wb}"))?);
                    }
                }
            }
        }
        lay.bytes("norm_out.w", f16_bytes(&st, "encoder.norm_out.weight")?);
        lay.bytes("norm_out.b", f16_bytes(&st, "encoder.norm_out.bias")?);
        let mut host = vec![0u8; lay.size];
        lay.pack(&st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, st0)?;
        st0.sync()?;
        // the single-frame image decoders (proj_out of 3 x 16 x 16) come with a posterior-sampling encoder
        let sample = st.has("decoder.proj_out.weight") && st.info("decoder.proj_out.weight")?.shape[0] == 3 * RATIO * RATIO;
        let (mean, std) = if st.has("latents_mean") {
            (st.f32_vec("latents_mean", false)?, st.f32_vec("latents_std", false)?)
        } else {
            (LATENTS_MEAN.to_vec(), LATENTS_STD.to_vec())
        };
        Ok(Encoder { buf, lay, sample, mean, std })
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    /// out (+= with `beta` 1) = conv of `x` [h, w, cin] by `key`: 1x1, or 3x3 at `stride` 1 or 2.
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, key: &str, x: PM, (h, w): (usize, usize), cin: usize, cout: usize, k: usize, stride: usize, out: PM, beta: f32, st: &Stream) -> Result<()> {
        let (wt, b) = (self.w(&format!("{key}.w")), self.w(&format!("{key}.b")));
        if k == 1 {
            return k!(gpu_gemm(x, cin as i64, wt, cin as i64, out, cout as i64, b, (h * w) as i64, cout as i64, cin as i64, 1, DT_F16, DT_F16, 1.0, beta, s(st)));
        }
        let (ho, wo, kk) = (h / stride, w / stride, 9 * cin);
        let rows = ((128usize << 20) / (wo * kk * 2)).clamp(1, ho);
        let col = DevBuf::alloc_async(rows * wo * kk * 2, st)?;
        for y0 in (0..ho).step_by(rows) {
            let r = rows.min(ho - y0);
            k!(gpu_h3_im2col3(x, h as i32, w as i32, cin as i32, stride as i32, y0 as i32, r as i32, col.ptr(), s(st)))?;
            k!(gpu_gemm(col.ptr(), kk as i64, wt, kk as i64, off(out, y0 * wo * cout * 2), cout as i64, b, (r * wo) as i64, cout as i64, kk as i64, 1, DT_F16, DT_F16,
                1.0, beta, s(st)))?;
        }
        Ok(())
    }

    fn norm_silu(&self, key: &str, x: PM, p: usize, c: usize, scratch: &DevBuf, st: &Stream) -> Result<DevBuf> {
        let out = DevBuf::alloc_async(p * c * 2, st)?;
        k!(gpu_h3_group_norm_f16(x, out.ptr(), self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), p as i64, c as i32, ENC_GROUPS, ENC_EPS, 1, scratch.ptr(),
            s(st)))?;
        Ok(out)
    }

    /// The moments of one tile, f16 NHWC pixels [h, w, 8] -> [48, 1, h / 16, w / 16] after `quant_conv`.
    fn tile(&self, px: &DevBuf, h: usize, w: usize, st: &Stream) -> Result<Clip> {
        let alloc = |p: usize, c: usize| DevBuf::alloc_async(p * c * 2, st);
        let scratch = DevBuf::alloc_async(2 * ENC_GROUPS as usize * 8, st)?;
        let (mut hh, mut ww) = (h, w);
        let mut x = alloc(h * w, ENC_CH[0])?;
        self.conv("in", px.ptr(), (h, w), ENC_RGB, ENC_CH[0], 3, 1, x.ptr(), 0.0, st)?;
        let mut cin = ENC_CH[0];
        for (l, &c) in ENC_CH.iter().enumerate() {
            let p = hh * ww;
            for i in 0..2 {
                let t = self.norm_silu(&format!("{l}.{i}.norm1"), x.ptr(), p, cin, &scratch, st)?;
                let h1 = alloc(p, c)?;
                self.conv(&format!("{l}.{i}.c1"), t.ptr(), (hh, ww), cin, c, 3, 1, h1.ptr(), 0.0, st)?;
                drop(t);
                let t = self.norm_silu(&format!("{l}.{i}.norm2"), h1.ptr(), p, c, &scratch, st)?;
                drop(h1);
                // conv2 adds onto the shortcut
                if cin != c {
                    let sc = alloc(p, c)?;
                    self.conv(&format!("{l}.{i}.sc"), x.ptr(), (hh, ww), cin, c, 1, 1, sc.ptr(), 0.0, st)?;
                    x = sc;
                }
                self.conv(&format!("{l}.{i}.c2"), t.ptr(), (hh, ww), c, c, 3, 1, x.ptr(), 1.0, st)?;
                cin = c;
            }
            if l < ENC_DOWN {
                let y = alloc(p / 4, c)?;
                self.conv(&format!("{l}.down"), x.ptr(), (hh, ww), c, c, 3, 2, y.ptr(), 0.0, st)?;
                x = y;
                hh /= 2;
                ww /= 2;
            }
        }
        let p = hh * ww;
        let t = self.norm_silu("norm_out", x.ptr(), p, cin, &scratch, st)?;
        let m = alloc(p, 2 * C)?;
        self.conv("out", t.ptr(), (hh, ww), cin, 2 * C, 3, 1, m.ptr(), 0.0, st)?;
        let q = alloc(p, 2 * C)?;
        self.conv("quant", m.ptr(), (hh, ww), 2 * C, 2 * C, 1, 1, q.ptr(), 0.0, st)?;
        let mut b = vec![0u8; p * 2 * C * 2];
        q.download(0, &mut b, st)?;
        let mut o = Clip::new(2 * C, 1, hh, ww);
        for (i, v) in b.chunks_exact(2).enumerate() {
            o.data[(i % (2 * C)) * p + i / (2 * C)] = f16::from_le_bytes([v[0], v[1]]).to_f32();
        }
        Ok(o)
    }

    /// RGB f32 [h, w, 3] in 0..1 (sides multiples of 16) -> the sampler's latent [24, h / 16, w / 16]: ComfyUI's
    /// `encode` through its 256px tiles.
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize, st: &Stream) -> Result<Vec<f32>> {
        anyhow::ensure!(h.is_multiple_of(RATIO) && w.is_multiple_of(RATIO) && rgb.len() == h * w * 3, "an H3 VAE input of {w}x{h}");
        // process_input to -1..1, then _normalize_pixels, as fp16 ops
        let r = |v: f32| f16::from_f32(v).to_f32();
        let mut px = vec![0f32; h * w * ENC_RGB];
        for (o, v) in px.chunks_exact_mut(ENC_RGB).zip(rgb.chunks_exact(3)) {
            for c in 0..3 {
                let x = r(r(v[c] * 2.0 - 1.0) + 1.0);
                o[c] = r(r(r(x * 0.5) - r(MEAN[c])) / r(STD[c]));
            }
        }
        let (ys, th, yo) = split_tiles(h);
        let (xs, tw, xo) = split_tiles(w);
        let mut rows: Vec<Vec<Clip>> = Vec::new();
        for &y0 in &ys {
            let mut row = Vec::new();
            for &x0 in &xs {
                let mut tile = Vec::with_capacity(th * tw * ENC_RGB);
                for y in 0..th {
                    tile.extend(px[((y0 + y) * w + x0) * ENC_RGB..][..tw * ENC_RGB].iter().map(|v| f16::from_f32(*v)));
                }
                let d = DevBuf::alloc_async(tile.len() * 2, st)?;
                d.upload(0, &tile.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), st)?;
                row.push(self.tile(&d, th, tw, st)?);
            }
            rows.push(row);
        }
        let (ly, lx): (Vec<usize>, Vec<usize>) = (yo.iter().map(|o| o / RATIO).collect(), xo.iter().map(|o| o / RATIO).collect());
        let mut m = Clip::new(2 * C, 1, h / RATIO, w / RATIO);
        let mut y = 0;
        for (i, row) in rows.iter().enumerate() {
            let (mut x, mut height) = (0, 0);
            for (j, raw) in row.iter().enumerate() {
                // each tile blends with its neighbours as they came out of the encoder
                let mut tile = raw.clone();
                if i > 0 {
                    tile = blend(&rows[i - 1][j], &tile, ly[i - 1], 2);
                }
                if j > 0 {
                    tile = blend(&row[j - 1], &tile, lx[j - 1], 3);
                }
                let keep_h = if i + 1 < rows.len() { tile.h - ly[i] } else { tile.h };
                let keep_w = if j + 1 < row.len() { tile.w - lx[j] } else { tile.w };
                paste(&mut m, &tile.crop((0, 1), (0, keep_h), (0, keep_w)), y, x);
                x += keep_w;
                height = keep_h;
            }
            y += height;
        }
        let p = (h / RATIO) * (w / RATIO);
        let noise = if self.sample { crate::sampling::torch_randn(IMAGE_ENCODE_SEED, C * p) } else { Vec::new() };
        let mut out = vec![0f32; C * p];
        for c in 0..C {
            for i in 0..p {
                let mut z = m.data[c * p + i];
                if self.sample {
                    let logvar = m.data[(C + c) * p + i].clamp(-30.0, 20.0);
                    z += (0.5 * logvar).exp() * noise[c * p + i];
                }
                out[c * p + i] = (z - self.mean[c]) / self.std[c];
            }
        }
        Ok(out)
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
