//! The pixel head of a pi-H3 DWD adapter (LakonLab's "Decoding Without a Decoder"): at the last sampling step, the
//! final hidden states of H3's video rows decode straight to pixels, in place of the VAE decoder.
//!
//! The head is ComfyUI-piFlow's `SolH3DWDHead`: the last layers of the Sol-H3-VAE decoder (ComfyUI's
//! `NonCausalViT3DDecoder`), its final norm and pixel projections. A linear expansion maps every 2x2-latent H3 token
//! onto four 16-pixel Sol tokens. Attention is local: 16x16-token windows over 5-latent temporal windows, odd layers
//! shifted by half a window, plus 5 global rows per native 5-latent block (4 register tokens and a zero row) that
//! attend to their whole block and that every window of the block sees. The first latent of a block decodes to one
//! frame through its own projection, the others to four.
//!
//! The layers are int8 ConvRot like the video VAE's, and run with its kernels: int8 attention in the windows, exact
//! attention for the global rows, as ComfyUI does.

use crate::cuda::{DevBuf, Event, PinnedBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, PM, PRO_LN_AFFINE, PRO_PLAIN, PRO_RMS, PRO_SWIGLU};
use crate::h3vae::Clip;
use crate::offload::{plan, Layout, Plan, WeightSet};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, ensure, Result};
use half::slice::HalfFloatSliceExt;
use half::{bf16, f16};
use std::sync::Arc;

const PREFIX: &str = "dwd_pixel_refiner.";
const D: usize = 2048;
const HEADS: usize = 32;
const DH: usize = 64;
const QKV: usize = 3 * D;
const FF: usize = 4 * D;
const EPS: f32 = 1e-5;
const PATCH: usize = 16;
const PATCH_T: usize = 4;
/// Frames dropped from the first latent of every native block: it keeps one of its four.
const FIRST_T: usize = 1;
const REGISTERS: usize = 4;
/// Global rows of a block: the register tokens, then a zero row.
const S: usize = REGISTERS + 1;
const WINDOW: usize = 16;
const BLOCK: usize = 5;
const TIME_SHIFT: usize = 2;
/// Rope pairs per head: 8 frequencies on each of (t, y, x).
const PAIRS: usize = 24;
/// Windows gathered for one attention call.
const MAX_WINDOWS: usize = 32;
/// Rows each norm, quantization and GEMM runs at once, for the size of their buffers.
const CHUNK_ROWS: usize = 16384;
/// Bytes of each of the two pinned buffers the frames download through.
const STAGE_BYTES: usize = 32 << 20;
/// `RGBColorEncoder`'s affine codec of the head's ImageNet-normalized RGB, to [-1, 1].
const COLOR_MEAN: [f32; 3] = [-0.03, -0.088, -0.188];
const COLOR_STD: [f32; 3] = [0.458, 0.448, 0.45];

/// The DWD pixel head in `st` is one this module runs, on features `width` wide: checked from the header alone, so an
/// adapter is refused when it applies rather than after sampling. Returns the head's layer count.
pub fn check(st: &SafeTensors, width: usize) -> Result<usize> {
    let n = |x: &str| format!("{PREFIX}{x}");
    ensure!(st.has(&n("register_tokens")), "a DWD adapter without the pixel head this engine runs (no {PREFIX}register_tokens)");
    let layers = (0..).take_while(|i| st.has(&n(&format!("blocks.{i}.attn.to_qkv.weight")))).count();
    ensure!(layers > 0, "a DWD pixel head without layers");
    let shape = |x: &str| st.info(&n(x)).map(|i| i.shape.clone());
    if shape("feature_expand.weight")? != [4 * D, width] || shape("register_tokens")? != [1, REGISTERS, D]
        || shape("first_frame_proj_out.weight")? != [3 * FIRST_T * PATCH * PATCH, D] || shape("proj_out.weight")? != [3 * PATCH_T * PATCH * PATCH, D]
    {
        bail!("a DWD pixel head of another shape than pi-H3's (Sol tokens {D} wide, 2x2 per H3 token, features {width} wide)");
    }
    for i in 0..layers {
        let b = |x: &str| n(&format!("blocks.{i}.{x}"));
        for (m, want) in [("attn.to_qkv", [QKV, D]), ("attn.to_out", [D, D]), ("ff.w1", [2 * FF, D]), ("ff.w2", [D, FF])] {
            let q = st.bytes(&b(&format!("{m}.comfy_quant"))).ok().and_then(|v| serde_json::from_slice::<serde_json::Value>(v).ok());
            if !q.is_some_and(|q| q["format"] == "int8_tensorwise" && q["convrot"] == true && q["convrot_groupsize"] == 256) {
                bail!("{}: the DWD head runs int8 ConvRot weights with 256-wide groups, as the int8_convrot adapters have them", b(m));
            }
            ensure!(shape(&format!("blocks.{i}.{m}.weight"))? == want, "{}: a DWD head layer of another shape than pi-H3's", b(m));
        }
    }
    Ok(layers)
}

/// Rough VRAM [`Refiner::decode`] takes for a `(t, th, tw)` H3 token grid, besides the features and the layers.
pub fn activation_bytes((t, th, tw): (usize, usize, usize)) -> usize {
    let hw = 4 * th * tw;
    let n = t * hw + t.div_ceil(BLOCK) * S;
    let block = BLOCK * hw + S;
    let chunk = n.clamp(1, CHUNK_ROWS);
    let window = (BLOCK + 1) * WINDOW * WINDOW + 2 * S;
    // q, k and v, the residual stream and the attention output; a chunk's MLP; a block's pixels; one batch of gathered
    // windows
    n * (QKV + 2 * D) * 2 + chunk * (FF + 4 + 4 * FF) + block * 3 * PATCH_T * PATCH * PATCH * 2 + MAX_WINDOWS * window * (QKV + D) * 2 + (256 << 20)
}

/// `[lo, hi)` of the windows along one axis of `n`, shifted by `shift`, clipped to `[0, n)`.
fn spans(n: usize, shift: usize, size: usize) -> Vec<(usize, usize)> {
    (0..(n + shift).div_ceil(size))
        .map(|i| ((i * size).saturating_sub(shift), ((i + 1) * size - shift).min(n)))
        .filter(|(lo, hi)| hi > lo)
        .collect()
}

/// Windows of one call: `n` windows of `lq` queries, each attending its own rows then the global rows of its blocks.
struct Batch {
    lq: usize,
    lk: usize,
    n: usize,
    q: DevBuf,
    k: DevBuf,
}

/// The index tables of one latent shape, as `_build_geometry` lays them out: rows are block-major, a block's tokens in
/// (t, y, x) order and then its global rows.
struct Geometry {
    rows: Vec<(usize, usize)>,
    /// runs of consecutive blocks with the same row count: (first row, rows per block, blocks)
    global_runs: Vec<(usize, usize, usize)>,
    /// rope tables of the plain and the shifted layers, a row of `PAIRS` (cos, sin) per row
    rope: [DevBuf; 2],
    windows: [Vec<Batch>; 2],
}

fn geometry((t, h, w): (usize, usize, usize), st: &Stream) -> Result<Geometry> {
    let hw = h * w;
    let nb = t.div_ceil(BLOCK);
    let starts: Vec<usize> = (0..=nb).map(|b| (b * BLOCK).min(t) * hw + b * S).collect();
    let rows: Vec<(usize, usize)> = starts.windows(2).map(|p| (p[0], p[1])).collect();
    let n = starts[nb];
    let row = |f: usize, y: usize, x: usize| starts[f / BLOCK] + (f % BLOCK) * hw + y * w + x;
    let globals = |b: usize| (starts[b + 1] - S..starts[b + 1]).map(|r| r as i32);
    let inv: Vec<f32> = (0..8).map(|i| 1.0 / 100f32.powf(i as f32 * 0.125)).collect();
    let mut rope = Vec::new();
    let mut windows = Vec::new();
    for (sp, stt) in [(0, 0), (WINDOW / 2, TIME_SHIFT)] {
        // window-local ids in [-1, 1], in the activations' bf16; the global rows keep 0, no rotation
        let id = |i: usize, o: usize, m: usize| bf16::from_f32((((i + o) % m) as f32 + 0.5) / m as f32 * 2.0 - 1.0).to_f32();
        // an axis's 8 (cos, sin) pairs depend on its window-local position alone: `m` of them per axis
        let axis = |o: usize, m: usize| -> Vec<[f32; 16]> {
            (0..m)
                .map(|i| {
                    let v = id(i, o, m);
                    let mut p = [0f32; 16];
                    for (j, fr) in inv.iter().enumerate() {
                        let ang = 2.0 * std::f32::consts::PI * v * fr;
                        p[j * 2] = bf16::from_f32(ang.cos()).to_f32();
                        p[j * 2 + 1] = bf16::from_f32(ang.sin()).to_f32();
                    }
                    p
                })
                .collect()
        };
        let (at, ay, ax) = (axis(stt, BLOCK), axis(sp, WINDOW), axis(sp, WINDOW));
        let mut table = vec![0f32; n * PAIRS * 2];
        for r in table.chunks_exact_mut(2) {
            r[0] = 1.0;
        }
        for f in 0..t {
            for y in 0..h {
                for x in 0..w {
                    let dst = &mut table[row(f, y, x) * PAIRS * 2..][..PAIRS * 2];
                    dst[..16].copy_from_slice(&at[f % BLOCK]);
                    dst[16..32].copy_from_slice(&ay[y % WINDOW]);
                    dst[32..].copy_from_slice(&ax[x % WINDOW]);
                }
            }
        }
        rope.push(table);
        // every window, keyed by its query and key counts
        let mut groups: Vec<((usize, usize), Vec<i32>, Vec<i32>)> = Vec::new();
        for (t0, t1) in spans(t, stt, BLOCK) {
            let mut blocks = vec![t0 / BLOCK, (t1 - 1) / BLOCK];
            blocks.dedup();
            let gk: Vec<i32> = blocks.iter().flat_map(|&b| globals(b)).collect();
            for (y0, y1) in spans(h, sp, WINDOW) {
                for (x0, x1) in spans(w, sp, WINDOW) {
                    let q: Vec<i32> = (t0..t1).flat_map(|f| (y0..y1).flat_map(move |y| (x0..x1).map(move |x| (f, y, x)))).map(|(f, y, x)| row(f, y, x) as i32).collect();
                    let key = (q.len(), q.len() + gk.len());
                    let g = match groups.iter_mut().find(|g| g.0 == key && g.1.len() < MAX_WINDOWS * key.0) {
                        Some(g) => g,
                        None => {
                            groups.push((key, Vec::new(), Vec::new()));
                            groups.last_mut().unwrap()
                        }
                    };
                    g.2.extend(&q);
                    g.2.extend(&gk);
                    g.1.extend(q);
                }
            }
        }
        windows.push(groups);
    }
    // the uploads wait for the stream, so they come after all the host work, which runs behind the GPU's
    let rope = rope
        .iter()
        .map(|table| -> Result<DevBuf> {
            let d = DevBuf::alloc_async(table.len() * 4, st)?;
            d.upload(0, &f32_bytes(table), st)?;
            Ok(d)
        })
        .collect::<Result<Vec<_>>>()?;
    let up = |v: &[i32]| -> Result<DevBuf> {
        let d = DevBuf::alloc_async(v.len() * 4, st)?;
        d.upload(0, &v.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>(), st)?;
        Ok(d)
    };
    let windows = windows
        .into_iter()
        .map(|groups| groups.into_iter().map(|((lq, lk), q, k)| Ok(Batch { lq, lk, n: q.len() / lq, q: up(&q)?, k: up(&k)? })).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;
    let [r0, r1]: [DevBuf; 2] = rope.try_into().map_err(|_| anyhow::anyhow!("two rope tables"))?;
    let [w0, w1]: [Vec<Batch>; 2] = windows.try_into().map_err(|_| anyhow::anyhow!("two window sets"))?;
    let mut global_runs: Vec<(usize, usize, usize)> = Vec::new();
    for &(a, b) in &rows {
        match global_runs.last_mut() {
            Some(r) if r.1 == b - a => r.2 += 1,
            _ => global_runs.push((a, b - a, 1)),
        }
    }
    Ok(Geometry { rows, global_runs, rope: [r0, r1], windows: [w0, w1] })
}

pub struct Refiner {
    ws: WeightSet,
    globals: DevBuf,
    gl: Layout,
    /// width of the features: H3's hidden size
    width: usize,
    /// threads that convert the downloaded frames
    threads: usize,
    /// the pinned buffers of [`Self::download`], made on the first decode
    stage: Option<[PinnedBuf; 2]>,
    prof: crate::prof::Prof,
}

impl Refiner {
    /// The head's tensors in a pi-H3 DWD adapter. The layers take no VRAM until [`Self::place`].
    pub fn load(st: &Arc<SafeTensors>, threads: usize, st0: &Stream) -> Result<Self> {
        let n = |x: &str| format!("{PREFIX}{x}");
        let fe = st.info(&n("feature_expand.weight"))?.shape.clone();
        ensure!(fe.len() == 2, "a DWD feature expansion of shape {fe:?}");
        let layers = check(st, fe[1])?;
        let mut lay = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = |x: &str| n(&format!("blocks.{i}.{x}"));
            let mut l = Layout::new();
            for (key, m) in [("qkv", "attn.to_qkv"), ("o", "attn.to_out"), ("w1", "ff.w1"), ("w2", "ff.w2")] {
                let weight = b(&format!("{m}.weight"));
                l.raw(st, &format!("w_{key}"), &[weight])?;
                l.raw(st, &format!("s_{key}"), &[b(&format!("{m}.weight_scale"))])?;
                l.bf16(st, &format!("b_{key}"), &[b(&format!("{m}.bias"))])?;
            }
            l.f32(st, "n1", &b("norm1.weight"), 0.0)?;
            l.f32(st, "n2", &b("norm2.weight"), 0.0)?;
            l.bf16(st, "s1", &[b("scale1")])?;
            l.bf16(st, "s2", &[b("scale2")])?;
            lay.push(l);
        }
        let ws = WeightSet::lazy("dwd head", st.clone(), lay, Plan { resident: 0, slots: 0 }, threads)?;
        // it runs once per clip: streamed through the cast buffers, it leaves the diffusion model's blocks in VRAM
        ws.once_per_prompt();
        ws.keep_images();
        let mut gl = Layout::new();
        for key in ["feature_expand.weight", "feature_expand.bias", "first_frame_proj_out.weight", "first_frame_proj_out.bias", "proj_out.weight", "proj_out.bias", "register_tokens"] {
            gl.bf16(st, key, &[n(key)])?;
        }
        gl.f32(st, "norm_out.weight", &n("norm_out.weight"), 0.0)?;
        gl.f32(st, "norm_out.bias", &n("norm_out.bias"), 0.0)?;
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, st0)?;
        st0.sync()?;
        Ok(Refiner { ws, globals, gl, width: fe[1], threads, stage: None, prof: crate::prof::Prof::new() })
    }

    /// Width of the features it reads.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Give the layers `budget` bytes of VRAM.
    pub fn place(&mut self, budget: usize) -> Result<()> {
        self.ws.replan(plan(&self.ws.layouts, budget, 2))
    }

    pub fn park(&mut self) -> Result<()> {
        self.ws.replan(Plan { resident: 0, slots: 0 })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// bf16 features `[t * th * tw, width]` of a `(t, th, tw)` H3 token grid, in its (t, y, x) order, to RGB f32
    /// `[3, frames, 32th, 32tw]` in 0..1: `refine` then `MiniMaxH3DWD.decode`, fp16 values as it returns them.
    pub fn decode(&mut self, features: &DevBuf, (t, th, tw): (usize, usize, usize), st: &Stream) -> Result<Clip> {
        self.prof.mark("start", st)?;
        let (h, w) = (th * 2, tw * 2);
        let hw = h * w;
        let tokens = t * th * tw;
        ensure!(features.size() >= tokens * self.width * 2, "DWD features of {} bytes for {tokens} tokens", features.size());
        // the frames' host memory, written once on another thread while the GPU works: a fresh allocation's page
        // faults would otherwise land on the download
        let frames = t * PATCH_T - (PATCH_T - FIRST_T) * t.div_ceil(BLOCK);
        let len = 3 * frames * h * PATCH * w * PATCH;
        let host = std::thread::spawn(move || {
            let mut v = Vec::with_capacity(len);
            v.resize(len, 0f32);
            v
        });

        // feature_expand, then the rows of x gathered from its outputs: each H3 token's four Sol tokens, and the
        // register and zero rows after them. The GEMM runs while the host builds the geometry.
        let src_rows = tokens * 4 + S;
        let e = DevBuf::alloc_async(src_rows * D * 2, st)?;
        k!(gpu_gemm(features.ptr(), self.width as i64, self.g("feature_expand.weight"), self.width as i64, e.ptr(), (4 * D) as i64, self.g("feature_expand.bias"),
            tokens as i64, (4 * D) as i64, self.width as i64, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
        crate::cuda::memcpy_d2d(off(e.ptr(), tokens * 4 * D * 2), self.g("register_tokens"), REGISTERS * D * 2, st)?;
        e.upload((tokens * 4 + REGISTERS) * D * 2, &[0u8; D * 2], st)?;
        let tg = std::time::Instant::now();
        let geo = geometry((t, h, w), st)?;
        self.prof.host("geometry", tg);
        let n = geo.rows.last().map_or(0, |r| r.1);
        let di = D as i64;
        let mut idx = vec![0i32; n];
        for (b, &(r0, r1)) in geo.rows.iter().enumerate() {
            let frames = (r1 - r0 - S) / hw;
            for f in 0..frames {
                for y in 0..h {
                    for x in 0..w {
                        let tok = ((b * BLOCK + f) * th + y / 2) * tw + x / 2;
                        idx[r0 + f * hw + y * w + x] = (tok * 4 + (y % 2) * 2 + x % 2) as i32;
                    }
                }
            }
            for j in 0..S {
                idx[r1 - S + j] = (tokens * 4 + j) as i32;
            }
        }
        let idx_d = DevBuf::alloc_async(n * 4, st)?;
        idx_d.upload(0, &idx.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>(), st)?;
        let x = DevBuf::alloc_async(n * D * 2, st)?;
        k!(gpu_gather_rows(e.ptr(), (D * 2) as i64, idx_d.ptr() as _, n as i32, x.ptr(), s(st)))?;
        drop(e);
        self.prof.mark("expand", st)?;

        let block_rows = geo.rows.iter().map(|r| r.1 - r.0).max().unwrap_or(0);
        // the row-wise work runs over every row at once, in chunks: only attention sees the native blocks
        let chunk = n.clamp(1, CHUNK_ROWS);
        let chunks: Vec<(usize, usize)> = (0..n).step_by(chunk).map(|r0| (r0, (r0 + chunk).min(n))).collect();
        let qkv = DevBuf::alloc_async(n * QKV * 2, st)?;
        let attn = DevBuf::alloc_async(n * D * 2, st)?;
        let a8 = DevBuf::alloc_async(chunk * FF, st)?;
        let asc = DevBuf::alloc_async(chunk * 4, st)?;
        let hid = DevBuf::alloc_async(chunk * 2 * FF * 2, st)?;
        let (mut gather, mut wout, mut wsz) = (0usize, 0usize, 0usize);
        for b in geo.windows.iter().flatten() {
            gather = gather.max(b.n * b.lk);
            wout = wout.max(b.n * b.lq);
            wsz = wsz.max(unsafe { crate::ffi::gpu_int8_attn_workspace_d(b.n as i32, HEADS as i32, HEADS as i32, b.lq as i32, b.lk as i32, DH as i32) } as usize);
        }
        let gbuf = DevBuf::alloc_async(gather * QKV * 2, st)?;
        let obuf = DevBuf::alloc_async(wout * D * 2, st)?;
        let ws = DevBuf::alloc_async(wsz, st)?;
        let lse = DevBuf::alloc_async(geo.rows.len() * HEADS * S * 4, st)?;
        let (xp, q) = (x.ptr(), qkv.ptr());
        let (rs, hs) = (QKV as i64, (3 * DH) as i64);
        let sc = 1.0 / (DH as f32).sqrt();
        for li in 0..self.ws.len() {
            let base = self.ws.acquire(li, st)?;
            self.prof.mark("stream", st)?;
            let lay = self.ws.layouts[li].clone();
            let wp = |key: &str| off(base, lay.off(key));
            let rope = geo.rope[li % 2].ptr();
            for &(r0, r1) in &chunks {
                let m = (r1 - r0) as i32;
                k!(gpu_quant_rows(DT_BF16, PRO_RMS, off(xp, r0 * D * 2), di, null(), 0, 0, wp("n1"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, D as i32, 1, s(st)))?;
                let qb = off(q, r0 * QKV * 2);
                k!(gpu_int8_gemm(a8.ptr(), wp("w_qkv"), asc.ptr(), wp("s_qkv"), wp("b_qkv"), qb, m as i64, QKV as i64, di, DT_BF16, s(st)))?;
                let table = off(rope, r0 * PAIRS * 2 * 4);
                k!(gpu_rope_bf16(qb, rs, hs, null(), EPS, table, m, HEADS as i32, DH as i32, PAIRS as i32, s(st)))?;
                k!(gpu_rope_bf16(off(qb, DH * 2), rs, hs, null(), EPS, table, m, HEADS as i32, DH as i32, PAIRS as i32, s(st)))?;
            }
            self.prof.mark("qkv", st)?;
            // the windows, gathered by their shape: own rows then the blocks' global rows as keys
            for bt in &geo.windows[li % 2] {
                k!(gpu_gather_rows(q, (QKV * 2) as i64, bt.k.ptr() as _, (bt.n * bt.lk) as i32, gbuf.ptr(), s(st)))?;
                let (g, kbs, obs) = (gbuf.ptr(), (bt.lk * QKV) as i64, (bt.lq * D) as i64);
                k!(gpu_int8_attn_d(g, off(g, DH * 2), off(g, 2 * DH * 2), obuf.ptr(), ws.ptr(), bt.n as i32, bt.lq as i32, bt.lk as i32, HEADS as i32, HEADS as i32, DH as i32,
                    kbs, rs, hs, kbs, rs, hs, kbs, rs, hs, obs, di, DH as i64, sc, s(st)))?;
                k!(gpu_scatter_rows(obuf.ptr(), (D * 2) as i64, bt.q.ptr() as _, (bt.n * bt.lq) as i32, attn.ptr(), s(st)))?;
            }
            // the global rows attend to their whole block, exactly: one call per run of same-sized blocks, which lie
            // evenly spaced, so the calls fill the GPU rather than one block's 32 heads at a time
            for &(r0, lk, nblk) in &geo.global_runs {
                let (gq, kb) = (off(q, (r0 + lk - S) * QKV * 2), off(q, r0 * QKV * 2));
                let (qbs, obs) = ((lk * QKV) as i64, (lk * D) as i64);
                k!(gpu_flash_attn_d(gq, off(kb, DH * 2), off(kb, 2 * DH * 2), off(attn.ptr(), (r0 + lk - S) * D * 2), lse.ptr(), nblk as i32, S as i32, lk as i32,
                    HEADS as i32, HEADS as i32, DH as i32, qbs, rs, hs, qbs, rs, hs, qbs, rs, hs, obs, di, DH as i64, sc, 0, s(st)))?;
            }
            self.prof.mark("attn", st)?;
            for &(r0, r1) in &chunks {
                let (m, xs, at) = ((r1 - r0) as i32, off(xp, r0 * D * 2), off(attn.ptr(), r0 * D * 2));
                k!(gpu_quant_rows(DT_BF16, PRO_PLAIN, at, di, null(), 0, 0, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, D as i32, 1, s(st)))?;
                k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_o"), asc.ptr(), wp("s_o"), wp("b_o"), wp("s1"), xs, xs, m as i64, di, di, DT_BF16, s(st)))?;
                k!(gpu_quant_rows(DT_BF16, PRO_RMS, xs, di, null(), 0, 0, wp("n2"), EPS, null(), null(), a8.ptr(), asc.ptr(), m, D as i32, 1, s(st)))?;
                k!(gpu_int8_gemm(a8.ptr(), wp("w_w1"), asc.ptr(), wp("s_w1"), wp("b_w1"), hid.ptr(), m as i64, (2 * FF) as i64, di, DT_BF16, s(st)))?;
                k!(gpu_quant_rows(DT_BF16, PRO_SWIGLU, hid.ptr(), (2 * FF) as i64, null(), 0, FF as i64, null(), 0.0, null(), null(), a8.ptr(), asc.ptr(), m, FF as i32, 1, s(st)))?;
                k!(gpu_int8_gemm_residual(a8.ptr(), wp("w_w2"), asc.ptr(), wp("s_w2"), wp("b_w2"), wp("s2"), xs, xs, m as i64, di, FF as i64, DT_BF16, s(st)))?;
            }
            self.prof.mark("mlp", st)?;
            self.ws.release(li, st)?;
        }
        drop((qkv, a8, asc, hid, gbuf, obuf, ws));

        // norm_out and the pixel projections, block by block
        debug_assert_eq!(frames, geo.rows.iter().map(|&(r0, r1)| FIRST_T + ((r1 - r0 - S) / hw - 1) * PATCH_T).sum::<usize>());
        let mut clip = Clip { c: 3, t: frames, h: h * PATCH, w: w * PATCH, data: Vec::new() };
        let (first_n, rest_n) = (3 * FIRST_T * PATCH * PATCH, 3 * PATCH_T * PATCH * PATCH);
        let out = DevBuf::alloc_async(block_rows * rest_n * 2, st)?;
        let pix = DevBuf::alloc_async(len * 2, st)?;
        let mut f0 = 0;
        for &(r0, r1) in &geo.rows {
            let lat = (r1 - r0 - S) / hw;
            let y = off(attn.ptr(), r0 * D * 2);
            k!(gpu_prologue_rows_b(DT_BF16, PRO_LN_AFFINE, off(xp, r0 * D * 2), di, self.g("norm_out.weight") as _, self.g("norm_out.bias") as _, EPS, null(), null(), y, di,
                (lat * hw) as i32, D as i32, s(st)))?;
            k!(gpu_gemm(y, di, self.g("first_frame_proj_out.weight"), di, out.ptr(), first_n as i64, self.g("first_frame_proj_out.bias"), hw as i64, first_n as i64, di, 1, DT_BF16,
                DT_BF16, 1.0, 0.0, s(st)))?;
            let rest = off(out.ptr(), hw * first_n * 2);
            if lat > 1 {
                k!(gpu_gemm(off(y, hw * D * 2), di, self.g("proj_out.weight"), di, rest, rest_n as i64, self.g("proj_out.bias"), ((lat - 1) * hw) as i64, rest_n as i64, di, 1,
                    DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
            }
            // the patches to their pixels, through the color codec, rounded to fp16
            let bf = FIRST_T + (lat - 1) * PATCH_T;
            k!(gpu_dwd_unpatch(out.ptr(), h as i32, w as i32, bf as i32, pix.ptr(), frames as i32, f0 as i32, COLOR_MEAN.as_ptr(), COLOR_STD.as_ptr(), s(st)))?;
            f0 += bf;
        }
        self.prof.mark("head", st)?;
        let t0 = std::time::Instant::now();
        clip.data = host.join().map_err(|_| anyhow::anyhow!("the frames' host buffer"))?;
        self.download(&pix, &mut clip.data, st)?;
        self.prof.host("download", t0);
        self.prof.report("dwd", st)?;
        Ok(clip)
    }

    /// fp16 `pix` to `out` in f32, through two pinned staging buffers: one chunk converts on every thread while the
    /// next copies. A pageable copy of a whole clip runs at a few GB/s, and its conversion on one thread as long again.
    fn download(&mut self, pix: &DevBuf, out: &mut [f32], st: &Stream) -> Result<()> {
        if self.stage.is_none() {
            self.stage = Some([PinnedBuf::alloc(STAGE_BYTES)?, PinnedBuf::alloc(STAGE_BYTES)?]);
        }
        let stage = self.stage.as_ref().unwrap();
        let per = STAGE_BYTES / 2;
        let chunks: Vec<(usize, usize)> = (0..out.len()).step_by(per).map(|a| (a, (a + per).min(out.len()))).collect();
        let done = [Event::new()?, Event::new()?];
        let copy = |i: usize| -> Result<()> {
            let (a, b) = chunks[i];
            // SAFETY: a stage buffer holds STAGE_BYTES of pinned memory, and is read only after `done` for its copy
            unsafe { crate::cuda::memcpy_2d_d2h(stage[i % 2].ptr(), (b - a) * 2, off(pix.ptr(), a * 2), (b - a) * 2, (b - a) * 2, 1, st) }?;
            done[i % 2].record(st)
        };
        if !chunks.is_empty() {
            copy(0)?;
        }
        let threads = self.threads.max(1);
        for (i, &(a, b)) in chunks.iter().enumerate() {
            if i + 1 < chunks.len() {
                copy(i + 1)?;
            }
            done[i % 2].sync()?;
            let src = unsafe { std::slice::from_raw_parts(stage[i % 2].ptr().cast::<f16>(), b - a) };
            let part = (b - a).div_ceil(threads);
            std::thread::scope(|sc| {
                for (s, d) in src.chunks(part).zip(out[a..b].chunks_mut(part)) {
                    sc.spawn(move || s.convert_to_f32_slice(d));
                }
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_shift_and_clip_like_comfyui() {
        assert_eq!(spans(32, 0, 16), vec![(0, 16), (16, 32)]);
        assert_eq!(spans(32, 8, 16), vec![(0, 8), (8, 24), (24, 32)]);
        assert_eq!(spans(2, 2, 5), vec![(0, 2)]);
        assert_eq!(spans(12, 2, 5), vec![(0, 3), (3, 8), (8, 12)]);
    }

    /// Reads only the headers of pi-H3's int8 ConvRot DWD adapters in `$MODELS_ROOT/Lora`.
    #[test]
    #[ignore]
    fn check_takes_the_pi_h3_adapters() {
        let root = std::path::PathBuf::from(std::env::var("MODELS_ROOT").expect("MODELS_ROOT")).join("Lora");
        for steps in [2, 4] {
            let st = SafeTensors::open(&root.join(format!("pi_h3_{steps}step_dwd_adapter_int8_convrot.safetensors"))).unwrap();
            assert_eq!(check(&st, 5376).unwrap(), 8);
            assert!(check(&st, 4096).is_err());
        }
    }
}
