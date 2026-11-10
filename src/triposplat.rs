//! TripoSplat, and Comfy-Org's fine-tune Idol: image to 3D gaussian splat, following ComfyUI's nodes as the Idol
//! example workflow wires them: `TripoSplatPreprocessImage`, `TripoSplatConditioning` (DINOv3 ViT-H/16+ and the
//! Flux 2 VAE), a `KSampler` on the LatentSeqMMFlowModel over its nested latent (8192 shape tokens and a camera token),
//! and `VAEDecodeTripoSplat`'s octree gaussian decoder. The output is a 3DGS `.ply`, as `SaveGaussianSplat` writes it.
//!
//! Weights and activations are fp16, as ComfyUI runs both models; attention is int8 (comfy-kitchen's SageAttention).
//! Cond and uncond run as separate passes: every op is per token, and attention is per sequence, so the result is the
//! same as ComfyUI's CFG batch.

use crate::cuda::{DevBuf, Stream};
use crate::dino3::Dino3;
use crate::ffi::{k, null, off, s, DT_F16, P, PM, PRO_GELU_TANH, PRO_LN_AFFINE, PRO_LN_MOD, PRO_SILU};
use crate::flux2vae;
use crate::offload::{plan, Layout, WeightSet};
use crate::pixal3d::{pil_lanczos, Progress};
use crate::samplers::{self, CfgTweaks, Eval, ModelSampling, Uncond};
use crate::sampling::{torch_randn_streams, Mt19937};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};
use half::f16;
use std::path::PathBuf;
use std::time::Instant;

/// Shape tokens of the latent, and their channels.
pub const Q_TOKENS: usize = 8192;
pub const LATENT_CH: usize = 16;
/// Channels of the camera token.
pub const CAM_CH: usize = 5;
/// The square image size the model is trained at.
pub const IMAGE_SIZE: usize = 1024;
pub const MIN_GAUSSIANS: usize = 32768;
pub const MAX_GAUSSIANS: usize = 1_048_576;

const F: usize = 1024;
const H: usize = 16;
const HD: usize = 64;
const QKV: usize = 3 * F;
const KV: usize = 2 * F;
const MLP: usize = 4096;
const MOD: usize = 6 * F;
const FREQ: usize = 256;
const COND_CH: usize = 1280;
const COND2_CH: usize = 128;
const REPO: usize = 128;
const REPO_OUT: usize = 3 * H;
/// LayerNorm of the blocks
const EPS: f32 = 1e-6;
/// LayerNorm at its default eps: the RePo layers', and the plain `F.layer_norm` before the output layers
const LN_EPS: f32 = 1e-5;
/// `comfy.rmsnorm.rms_norm`'s default
const RMS_EPS: f32 = 1e-6;
/// K of the tiny input projections, padded so the GEMM's rows stay aligned
const KPAD: usize = 8;

/// The four files of a TripoSplat or Idol install.
#[derive(Clone, Debug)]
pub struct Paths {
    pub dit: PathBuf,
    pub dino: PathBuf,
    pub flux_vae: PathBuf,
    pub decoder: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub seed: u64,
    pub steps: usize,
    pub cfg: f64,
    pub sampler: String,
    pub scheduler: String,
    pub shift: f64,
    /// gaussians of the decode, a multiple of 32 (262144 matches the octree's point density)
    pub num_gaussians: usize,
    /// seeds the octree's point sampler
    pub decode_seed: u64,
    /// `TripoSplatPreprocessImage`'s erosion of the mask, in pixels
    pub erode_radius: usize,
}

impl Params {
    /// The Idol example workflow's settings (`3d_idol_image_to_gaussian_splat.json`), seeded by one seed.
    pub fn example(seed: u64) -> Self {
        Params {
            seed,
            steps: 20,
            cfg: 3.0,
            sampler: "dpmpp_2m".into(),
            scheduler: "simple".into(),
            shift: 3.0,
            num_gaussians: 262_144,
            decode_seed: seed,
            erode_radius: 1,
        }
    }
}

/// Render-ready gaussians, in the viewer frame (Y up): positions, linear scales, unit rotations (w, x, y, z),
/// opacities in [0, 1] and the SH DC colour coefficients.
pub struct Splat {
    pub positions: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh: Vec<[f32; 3]>,
}

/// True if `st` holds a TripoSplat (or Idol) flow model.
pub fn is_triposplat(has: impl Fn(&str) -> bool) -> bool {
    has("cam_out_layer.weight") && has("repo_layers.0.final_map.weight")
}

/// True if `st` holds TripoSplat's octree gaussian decoder.
pub fn is_decoder(has: impl Fn(&str) -> bool) -> bool {
    has("gs.base_offset_scale") && has("octree.out_proj.weight")
}

fn r16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect()
}

fn f16_values(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

fn download_f16(buf: &DevBuf, n: usize, st: &Stream) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    buf.download(0, &mut b, st)?;
    st.sync()?;
    Ok(f16_values(&b))
}

/// A 2D fp16 weight zero-padded to `cols` columns.
fn pad_cols(st: &SafeTensors, name: &str, cols: usize) -> Result<Vec<u8>> {
    let info = st.info(name)?;
    let (r, c) = (info.shape[0], info.shape[1]);
    let v = st.f32_vec(name, false)?;
    let mut out = vec![0f32; r * cols];
    for i in 0..r {
        out[i * cols..i * cols + c].copy_from_slice(&v[i * c..(i + 1) * c]);
    }
    Ok(f16_bytes(&out))
}

/// A Linear's fp16 weight and bias zero-padded to `rows` outputs.
fn pad_rows(st: &SafeTensors, name: &str, rows: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut w = st.f32_vec(&format!("{name}.weight"), false)?;
    let mut b = st.f32_vec(&format!("{name}.bias"), false)?;
    let cols = w.len() / b.len();
    w.resize(rows * cols, 0.0);
    b.resize(rows, 0.0);
    Ok((f16_bytes(&w), f16_bytes(&b)))
}

// ---------------------------------------------------------------------------------------------
// TripoSplatPreprocessImage
// ---------------------------------------------------------------------------------------------

/// `comfy.utils.lanczos` of an RGBA image: Pillow resizes RGBA premultiplied ("RGBa"), then divides the alpha back out.
fn lanczos_rgba(src: &[f32], w: usize, h: usize, nw: usize, nh: usize) -> Vec<f32> {
    let mut px: Vec<u8> = src.iter().map(|v| (255.0f32 * v).clamp(0.0, 255.0) as u8).collect();
    if (w, h) == (nw, nh) {
        // Image.resize returns a copy at the same size, before any premultiplying
        return px.into_iter().map(|v| v as f32 / 255.0).collect();
    }
    let mul = |a: u32, b: u32| {
        let t = a * b + 128;
        ((t >> 8) + t) >> 8
    };
    for p in px.chunks_exact_mut(4) {
        let a = p[3] as u32;
        for c in 0..3 {
            p[c] = mul(p[c] as u32, a) as u8;
        }
    }
    let mut out = pil_lanczos(&px, w, h, 4, nw, nh);
    // Pillow's rgbA2rgba: truncating division, and no colour where there is no alpha
    for p in out.chunks_exact_mut(4) {
        let a = p[3] as u32;
        if a == 0 {
            p[..3].fill(0);
        } else if a != 255 {
            for c in 0..3 {
                p[c] = (255 * p[c] as u32 / a).min(255) as u8;
            }
        }
    }
    out.into_iter().map(|v| v as f32 / 255.0).collect()
}

/// `TripoSplatPreprocessImage`: `rgb` f32 [h, w, 3] and the object's mask (1 on it) to the square model input, the
/// object centred at 1.2x its extent over black, f32 [size, size, 3].
pub fn preprocess(rgb: &[f32], mask: &[f32], w: usize, h: usize, erode: usize, size: usize) -> Result<Vec<f32>> {
    if rgb.len() != w * h * 3 || mask.len() != w * h {
        bail!("the mask must have the image's size");
    }
    let size = (size / 16).max(1) * 16;
    let rgba: Vec<f32> = (0..w * h)
        .flat_map(|i| [rgb[i * 3].clamp(0.0, 1.0), rgb[i * 3 + 1].clamp(0.0, 1.0), rgb[i * 3 + 2].clamp(0.0, 1.0), mask[i].clamp(0.0, 1.0)])
        .collect();
    let sc = size as f64 / w.min(h) as f64;
    // python's round() rounds half to even
    let (rw, rh) = (((w as f64 * sc).round_ties_even() as usize).max(1), ((h as f64 * sc).round_ties_even() as usize).max(1));
    let mut img = lanczos_rgba(&rgba, w, h, rw, rh);
    if erode > 0 {
        // a min filter over the (2r+1)^2 window, inside the image
        let r = erode as i64;
        let a: Vec<f32> = img.chunks_exact(4).map(|p| p[3]).collect();
        let mut rows = vec![0f32; rw * rh];
        for y in 0..rh {
            for x in 0..rw {
                let (x0, x1) = ((x as i64 - r).max(0) as usize, ((x as i64 + r) as usize).min(rw - 1));
                rows[y * rw + x] = (x0..=x1).map(|xx| a[y * rw + xx]).fold(f32::INFINITY, f32::min);
            }
        }
        for y in 0..rh {
            let (y0, y1) = ((y as i64 - r).max(0) as usize, ((y as i64 + r) as usize).min(rh - 1));
            for x in 0..rw {
                img[(y * rw + x) * 4 + 3] = (y0..=y1).map(|yy| rows[yy * rw + x]).fold(f32::INFINITY, f32::min);
            }
        }
    }
    let fg: Vec<(usize, usize)> = (0..rh).flat_map(|y| (0..rw).map(move |x| (y, x))).filter(|&(y, x)| img[(y * rw + x) * 4 + 3] > 0.0).collect();
    if fg.is_empty() {
        bail!("the mask is empty: TripoSplat needs the object cut out, on a transparent background or with a mask");
    }
    let (y0, y1) = (fg.iter().map(|p| p.0).min().unwrap(), fg.iter().map(|p| p.0).max().unwrap());
    let (x0, x1) = (fg.iter().map(|p| p.1).min().unwrap(), fg.iter().map(|p| p.1).max().unwrap());
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let half = (x1 - x0).max(y1 - y0) as f64 / 2.0 * 1.2;
    // python's int() truncates toward zero
    let (left, upper, right, lower) = ((cx - half) as i64, (cy - half) as i64, (cx + half) as i64, (cy + half) as i64);
    let (cw, ch) = ((right - left).max(1) as usize, (lower - upper).max(1) as usize);
    let mut crop = vec![0f32; cw * ch * 4];
    for yy in 0..ch {
        let sy = upper + yy as i64;
        if sy < 0 || sy >= rh as i64 {
            continue;
        }
        for xx in 0..cw {
            let sx = left + xx as i64;
            if sx < 0 || sx >= rw as i64 {
                continue;
            }
            let si = (sy as usize * rw + sx as usize) * 4;
            crop[(yy * cw + xx) * 4..][..4].copy_from_slice(&img[si..si + 4]);
        }
    }
    let sq = lanczos_rgba(&crop, cw, ch, size, size);
    Ok(sq.chunks_exact(4).flat_map(|p| [p[0] * p[3], p[1] * p[3], p[2] * p[3]]).collect())
}

// ---------------------------------------------------------------------------------------------
// torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123): the latent tokens' fixed anchors
// ---------------------------------------------------------------------------------------------

const MAXBIT: usize = 30;

/// `SobolEngine(3, scramble=True, seed).draw(n)`, as torch computes it on the CPU.
pub fn sobol3(n: usize, seed: u64) -> Vec<[f32; 3]> {
    const DIM: usize = 3;
    // _sobol_engine_initialize_state_: direction numbers of the first three dimensions
    let mut ss = [[0i64; MAXBIT]; DIM];
    ss[0] = [1; MAXBIT];
    let polys = [0i64, 3, 7];
    let inits: [&[i64]; 3] = [&[], &[1], &[1, 3]];
    for d in 1..DIM {
        let p = polys[d];
        let m = (64 - p.leading_zeros()) as usize - 1;
        ss[d][..m].copy_from_slice(&inits[d][..m]);
        for j in m..MAXBIT {
            let mut newv = ss[d][j - m];
            let mut pow2 = 1i64;
            for kk in 0..m {
                pow2 <<= 1;
                if (p >> (m - 1 - kk)) & 1 != 0 {
                    newv ^= pow2 * ss[d][j - kk - 1];
                }
            }
            ss[d][j] = newv;
        }
    }
    for row in ss.iter_mut() {
        for (i, v) in row.iter_mut().enumerate() {
            *v <<= MAXBIT - 1 - i;
        }
    }
    // _scramble: torch.randint(2, ...) draws one 32-bit word each; the shift first, then the matrices
    let mut g = Mt19937::new(seed);
    let mut shift = [0i64; DIM];
    for sh in shift.iter_mut() {
        for b in 0..MAXBIT {
            *sh += (g.next() % 2) as i64 * (1i64 << b);
        }
    }
    let mut ltm = vec![[[0i64; MAXBIT]; MAXBIT]; DIM];
    for m in ltm.iter_mut() {
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                let bit = (g.next() % 2) as i64;
                *v = if c <= r { bit } else { 0 };
            }
        }
    }
    for d in 0..DIM {
        // the diagonals become 1, and each row dotted with powers of two
        let dots: Vec<i64> = (0..MAXBIT).map(|r| (0..MAXBIT).map(|c| if c == r { 1 } else { ltm[d][r][c] } << (MAXBIT - 1 - c)).sum()).collect();
        for j in 0..MAXBIT {
            let vdj = ss[d][j];
            let (mut l, mut t2) = (1i64, 0i64);
            for p in (0..MAXBIT).rev() {
                let t1 = ((dots[p] & vdj).count_ones() % 2) as i64;
                t2 += t1 * l;
                l <<= 1;
            }
            ss[d][j] = t2;
        }
    }
    let recip = 1.0f64 / (1u64 << MAXBIT) as f64;
    let mut quasi = shift;
    let mut out = Vec::with_capacity(n);
    out.push(quasi.map(|q| q as f32 / (1u64 << MAXBIT) as f32));
    for i in 0..n.saturating_sub(1) {
        let l = (i as u64).trailing_ones() as usize;
        for d in 0..DIM {
            quasi[d] ^= ss[d][l];
        }
        out.push(quasi.map(|q| (q as f64 * recip) as f32));
    }
    out
}

/// `PcdAbsolutePositionEmbedder(channels, max_res)` of points in f32: per axis 170 sines then 170 cosines, zero padded.
/// `pow2` is the flow model's schedule (computed on the CPU), else the decoders' `log2`.
fn position_embedding(points: &[[f32; 3]], channels: usize, pow2: bool) -> Vec<f32> {
    let fd = channels / 3 / 2;
    let freqs: Vec<f32> = if pow2 {
        let max_res = 16usize;
        let res_dim = fd.saturating_sub(max_res);
        (0..fd)
            .map(|i| if i < max_res { i as f32 } else { (i - max_res) as f32 / res_dim.max(1) as f32 * max_res as f32 })
            .map(|e| 2f32.powf(e) * 2.0)
            .collect()
    } else {
        // torch.linspace(0, 10, fd) fills from both ends
        let (start, end) = (0f32, 10f32);
        let step = (end - start) / (fd - 1) as f32;
        (0..fd).map(|i| if i < fd / 2 { start + step * i as f32 } else { end - step * (fd - i - 1) as f32 }).map(|e| 2f32.powf(e)).collect()
    };
    let pi = std::f32::consts::PI;
    let mut out = vec![0f32; points.len() * channels];
    for (p, row) in points.iter().zip(out.chunks_exact_mut(channels)) {
        for a in 0..3 {
            let seg = &mut row[a * 2 * fd..(a + 1) * 2 * fd];
            for (i, f) in freqs.iter().enumerate() {
                let ang = p[a] * f * pi;
                seg[i] = ang.sin();
                seg[fd + i] = ang.cos();
            }
        }
    }
    out
}

/// The sinusoid of `TimestepEmbedder` (cos then sin over `freq` channels, max period 10000) or of `LevelEmbedder`
/// (`two_pi`, max period 1024), rounded to fp16 for the first Linear.
fn sinusoid(t: f32, freq: usize, max_period: f64, two_pi: bool) -> Vec<f32> {
    let half = freq / 2;
    let lp = -(max_period.ln()) as f32;
    let mut e = vec![0f32; freq];
    for i in 0..half {
        let fr = (lp * i as f32 / half as f32).exp();
        let a = if two_pi { t * fr * 2.0 * std::f32::consts::PI } else { t * fr };
        e[i] = a.cos();
        e[half + i] = a.sin();
    }
    e
}

// ---------------------------------------------------------------------------------------------
// Shared fp16 ops
// ---------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn gemm16(a: P, lda: usize, w: P, ldb: usize, bias: P, d: PM, ldd: usize, m: usize, n: usize, kd: usize, st: &Stream) -> Result<()> {
    k!(gpu_gemm(a, lda as i64, w, ldb as i64, d, ldd as i64, bias, m as i64, n as i64, kd as i64, 1, DT_F16, DT_F16, 1.0, 0.0, s(st)))
}

/// Multi-head attention of fp16 rows (16 heads of 64), int8 like the rest of the engine.
#[allow(clippy::too_many_arguments)]
fn attention(q: P, q_rs: usize, kp: P, vp: P, kv_rs: usize, out: PM, lq: usize, lk: usize, ws: &DevBuf, st: &Stream) -> Result<()> {
    let need = unsafe { crate::ffi::gpu_int8_attn_workspace_d(1, H as i32, H as i32, lq as i32, lk as i32, HD as i32) } as usize;
    if need > ws.size() {
        bail!("attention workspace too small: {need} > {}", ws.size());
    }
    k!(gpu_int8_attn_f16(q, kp, vp, out, ws.ptr(), 1, lq as i32, lk as i32, H as i32, H as i32, HD as i32, (lq * q_rs) as i64, q_rs as i64,
        HD as i64, (lk * kv_rs) as i64, kv_rs as i64, HD as i64, (lk * kv_rs) as i64, kv_rs as i64, HD as i64, (lq * F) as i64, F as i64,
        HD as i64, 1.0 / (HD as f32).sqrt(), s(st)))
}

fn attn_ws(lq: usize, lk: usize, st: &Stream) -> Result<DevBuf> {
    DevBuf::alloc_async(unsafe { crate::ffi::gpu_int8_attn_workspace_d(1, H as i32, H as i32, lq as i32, lk as i32, HD as i32) } as usize, st)
}

/// x += y * gate (or x += y).
fn add(x: PM, x_stride: usize, y: P, y_stride: usize, gate: P, rows: usize, cols: usize, st: &Stream) -> Result<()> {
    k!(gpu_ts_add_f16(x, x_stride as i64, y, y_stride as i64, gate, rows as i64, cols as i32, s(st)))
}

/// An MLP of two Linears with tanh GELU between: `x` [rows, F] -> `out` [rows, F], through `hid` [rows, MLP].
#[allow(clippy::too_many_arguments)]
fn mlp(x: P, w1: P, b1: P, w2: P, b2: P, hid: PM, out: PM, rows: usize, st: &Stream) -> Result<()> {
    gemm16(x, F, w1, F, b1, hid, MLP, rows, MLP, F, st)?;
    k!(gpu_prologue_rows_b(DT_F16, PRO_GELU_TANH, hid, MLP as i64, std::ptr::null(), std::ptr::null(), 0.0, null(), null(), hid,
        MLP as i64, rows as i32, MLP as i32, s(st)))?;
    gemm16(hid, MLP, w2, MLP, b2, out, F, rows, F, MLP, st)
}

/// Scratch for one forward over at most `l` tokens.
pub struct Scratch {
    ln: DevBuf,
    qkv: DevBuf,
    attn: DevBuf,
    hid: DevBuf,
    gc: DevBuf,
    gcm: DevBuf,
    delta: DevBuf,
    modv: DevBuf,
    ws: DevBuf,
    l: usize,
}

impl Scratch {
    pub fn new(l: usize, lk: usize, st: &Stream) -> Result<Self> {
        let a = |n: usize| DevBuf::alloc_async(n * 2, st);
        Ok(Scratch {
            ln: a(l * F)?,
            qkv: a(l * QKV)?,
            attn: a(l * F)?,
            hid: a(l * MLP)?,
            gc: a(l * 2 * REPO)?,
            gcm: a(l * REPO)?,
            delta: a(l * REPO_OUT)?,
            modv: a(MOD)?,
            ws: attn_ws(l, lk.max(l), st)?,
            l,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// LatentSeqMMFlowModel
// ---------------------------------------------------------------------------------------------

/// One `UnifiedTransformerBlock` and the `RePo3DRotaryEmbedding` that feeds it.
#[derive(Clone, Copy, PartialEq)]
enum BlockKind {
    /// modulated by the timestep: the noise refiner and the main blocks
    Modulated,
    /// LayerNorms with weights, no modulation: the context refiner
    Plain,
}

fn block_layout(st: &SafeTensors, block: &str, repo: &str, kind: BlockKind) -> Result<Layout> {
    let n = |x: &str| vec![format!("{block}{x}")];
    let mut l = Layout::new();
    for (key, name) in [("qkv", "attn.qkv.weight"), ("qkvb", "attn.qkv.bias"), ("o", "attn.out.weight"), ("ob", "attn.out.bias"),
        ("qn", "attn.q_norm.gamma"), ("kn", "attn.k_norm.gamma"), ("m1", "mlp.mlp.0.weight"), ("m1b", "mlp.mlp.0.bias"),
        ("m2", "mlp.mlp.2.weight"), ("m2b", "mlp.mlp.2.bias")]
    {
        l.raw(st, key, &n(name))?;
    }
    match kind {
        BlockKind::Modulated => l.raw(st, "shift", &n("shift_table"))?,
        BlockKind::Plain => {
            for (key, name) in [("n1w", "norm1.weight"), ("n1b", "norm1.bias"), ("n2w", "norm2.weight"), ("n2b", "norm2.bias")] {
                l.f32(st, key, &format!("{block}{name}"), 0.0)?;
            }
        }
    }
    l.f32(st, "rnw", &format!("{repo}norm.weight"), 0.0)?;
    l.f32(st, "rnb", &format!("{repo}norm.bias"), 0.0)?;
    l.raw(st, "gc", &[format!("{repo}gate_map.weight"), format!("{repo}content_map.weight")])?;
    l.raw(st, "fm", &[format!("{repo}final_map.weight")])?;
    let mut freqs = Vec::new();
    for i in 0..3 {
        freqs.extend(st.f32_vec(&format!("{repo}freqs_{i}"), false)?);
    }
    if freqs.len() != HD / 2 {
        bail!("{repo}: {} rotary frequencies, expected {}", freqs.len(), HD / 2);
    }
    l.bytes("freqs", f32_bytes(&freqs));
    Ok(l)
}

/// One block over `l` rows of `x` (fp16 [l, F]); `t_mod` is the shared modulation for the modulated ones.
fn run_block(lay: &Layout, base: PM, kind: BlockKind, x: PM, l: usize, t_mod: P, sc: &Scratch, st: &Stream) -> Result<()> {
    if l > sc.l {
        bail!("{l} tokens for scratch of {}", sc.l);
    }
    let w = |key: &str| off(base, lay.off(key));
    let fi = F as i64;
    // RePo: per-head 3D offsets from the block's input
    k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE, x, fi, w("rnw"), w("rnb"), LN_EPS, null(), null(),
        sc.ln.ptr(), fi, l as i32, F as i32, s(st)))?;
    gemm16(sc.ln.ptr(), F, w("gc"), F, null(), sc.gc.ptr(), 2 * REPO, l, 2 * REPO, F, st)?;
    k!(gpu_ts_silu_mul_f16(sc.gc.ptr(), (2 * REPO) as i64, sc.gcm.ptr(), REPO as i64, l as i64, REPO as i32, s(st)))?;
    gemm16(sc.gcm.ptr(), REPO, w("fm"), REPO, null(), sc.delta.ptr(), REPO_OUT, l, REPO_OUT, REPO, st)?;

    let modulated = kind == BlockKind::Modulated;
    if modulated {
        crate::cuda::memcpy_d2d(sc.modv.ptr(), t_mod, MOD * 2, st)?;
        add(sc.modv.ptr(), 0, w("shift"), 0, null(), 1, MOD, st)?;
    }
    let mo = |j: usize| off(sc.modv.ptr(), j * F * 2) as P;
    let norm = |idx: usize, out: PM| -> Result<()> {
        if modulated {
            let (shift, scale) = if idx == 1 { (mo(0), mo(1)) } else { (mo(3), mo(4)) };
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_MOD, x, fi, std::ptr::null(), std::ptr::null(), EPS, scale, shift, out, fi, l as i32,
                F as i32, s(st)))
        } else {
            let (nw, nb) = if idx == 1 { (w("n1w"), w("n1b")) } else { (w("n2w"), w("n2b")) };
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE, x, fi, nw, nb, EPS, null(), null(), out, fi,
                l as i32, F as i32, s(st)))
        }
    };

    // attention: rotary on q and k, then their RMS norms
    norm(1, sc.ln.ptr())?;
    gemm16(sc.ln.ptr(), F, w("qkv"), F, w("qkvb"), sc.qkv.ptr(), QKV, l, QKV, F, st)?;
    k!(gpu_ts_repo_qk(sc.qkv.ptr(), QKV as i64, F as i32, sc.delta.ptr(), REPO_OUT as i64, w("freqs"), w("qn"), w("kn"),
        RMS_EPS, l as i32, H as i32, 1, s(st)))?;
    let q = sc.qkv.ptr();
    attention(q, QKV, off(q, F * 2), off(q, KV * 2), QKV, sc.attn.ptr(), l, l, &sc.ws, st)?;
    gemm16(sc.attn.ptr(), F, w("o"), F, w("ob"), sc.ln.ptr(), F, l, F, F, st)?;
    add(x, F, sc.ln.ptr(), F, if modulated { mo(2) } else { null() }, l, F, st)?;

    // MLP
    norm(2, sc.ln.ptr())?;
    mlp(sc.ln.ptr(), w("m1"), w("m1b"), w("m2"), w("m2b"), sc.hid.ptr(), sc.attn.ptr(), l, st)?;
    add(x, F, sc.attn.ptr(), F, if modulated { mo(5) } else { null() }, l, F, st)
}

pub struct Flow {
    globals: DevBuf,
    gl: Layout,
    blocks: WeightSet,
    /// units: the noise refiner, the context refiner, then the main blocks
    n_noise: usize,
    n_context: usize,
    /// fp16 [Q_TOKENS, F]: the Sobol anchors' embedding
    pos: DevBuf,
}

/// The image conditioning after the context refiner, for every evaluation: fp16 [n, F].
pub struct Context {
    h: DevBuf,
    n: usize,
}

impl Flow {
    pub fn load(st: &SafeTensors, threads: usize, s0: &Stream) -> Result<Self> {
        if !is_triposplat(|k| st.has(k)) {
            bail!("not a TripoSplat diffusion model");
        }
        let count = |p: &str| (0..).take_while(|i| st.has(&format!("{p}.{i}.attn.qkv.weight"))).count();
        let (n_noise, n_context, n_main) = (count("noise_refiner"), count("context_refiner"), count("blocks"));
        let mut gl = Layout::new();
        for (key, name) in [("t1w", "t_embedder.mlp.0.weight"), ("t1b", "t_embedder.mlp.0.bias"), ("t2w", "t_embedder.mlp.2.weight"),
            ("t2b", "t_embedder.mlp.2.bias"), ("adaw", "adaLN_modulation.1.weight"), ("adab", "adaLN_modulation.1.bias"),
            ("inw", "input_layer.weight"), ("inb", "input_layer.bias"), ("c1w", "cond_embedder.weight"), ("c1b", "cond_embedder.bias"),
            ("c2w", "cond_embedder2.weight"), ("c2b", "cond_embedder2.bias"), ("cam1b", "cam_refiner.mlp.0.bias"),
            ("cam2w", "cam_refiner.mlp.2.weight"), ("cam2b", "cam_refiner.mlp.2.bias"), ("shift", "shift_table"),
            ("outw", "out_layer.weight"), ("outb", "out_layer.bias")]
        {
            gl.raw(st, key, &[name.to_string()])?;
        }
        gl.bytes("cam1w", pad_cols(st, "cam_refiner.mlp.0.weight", KPAD)?);
        let (cw, cb) = pad_rows(st, "cam_out_layer", KPAD)?;
        gl.bytes("camow", cw);
        gl.bytes("camob", cb);
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;

        let mut layouts = Vec::new();
        for i in 0..n_noise {
            layouts.push(block_layout(st, &format!("noise_refiner.{i}."), &format!("noise_repo_layers.{i}."), BlockKind::Modulated)?);
        }
        for i in 0..n_context {
            layouts.push(block_layout(st, &format!("context_refiner.{i}."), &format!("context_repo_layers.{i}."), BlockKind::Plain)?);
        }
        for i in 0..n_main {
            layouts.push(block_layout(st, &format!("blocks.{i}."), &format!("repo_layers.{i}."), BlockKind::Modulated)?);
        }
        let p = plan(&layouts, usize::MAX, 0);
        let blocks = WeightSet::load("triposplat", st, layouts, p, threads)?;

        let anchors = sobol3(Q_TOKENS, 123);
        let pos = DevBuf::alloc(Q_TOKENS * F * 2)?;
        pos.upload(0, &f16_bytes(&position_embedding(&anchors, F, true)), s0)?;
        s0.sync()?;
        Ok(Flow { globals, gl, blocks, n_noise, n_context, pos })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// `cond_embedder(feature1) + cond_embedder2(front-padded Flux 2 latent)`, through the context refiner. `feat1`:
    /// f32 [n, 1280] DINOv3 tokens; `reference`: f32 [128, hw] latent, channel major.
    pub fn context(&mut self, feat1: &[f32], reference: &[f32], st: &Stream) -> Result<Context> {
        let n = feat1.len() / COND_CH;
        let hw = reference.len() / COND2_CH;
        if n * COND_CH != feat1.len() || hw * COND2_CH != reference.len() || hw > n {
            bail!("conditioning of {n} image tokens and {hw} latent positions");
        }
        let f1 = DevBuf::alloc_async(n * COND_CH * 2, st)?;
        f1.upload(0, &f16_bytes(feat1), st)?;
        let h = DevBuf::alloc_async(n * F * 2, st)?;
        gemm16(f1.ptr(), COND_CH, self.g("c1w"), COND_CH, self.g("c1b"), h.ptr(), F, n, F, COND_CH, st)?;
        drop(f1);
        let mut f2 = vec![0f32; n * COND2_CH];
        let pad = n - hw;
        for t in 0..hw {
            for c in 0..COND2_CH {
                f2[(pad + t) * COND2_CH + c] = reference[c * hw + t];
            }
        }
        let f2d = DevBuf::alloc_async(f2.len() * 2, st)?;
        f2d.upload(0, &f16_bytes(&f2), st)?;
        let t = DevBuf::alloc_async(n * F * 2, st)?;
        gemm16(f2d.ptr(), COND2_CH, self.g("c2w"), COND2_CH, self.g("c2b"), t.ptr(), F, n, F, COND2_CH, st)?;
        add(h.ptr(), F, t.ptr(), F, null(), n, F, st)?;
        drop((f2d, t));
        let sc = Scratch::new(n, n, st)?;
        for b in self.n_noise..self.n_noise + self.n_context {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            run_block(&lay, base, BlockKind::Plain, h.ptr(), n, null(), &sc, st)?;
            self.blocks.release(b, st)?;
        }
        Ok(Context { h, n })
    }

    /// One evaluation: `z` fp16 [Q_TOKENS, 16] and `cam` fp16 [KPAD] (5 used) at `sigma`; writes the velocities to
    /// `out_z` (fp16 [Q_TOKENS, 16]) and `out_cam` (fp16 [KPAD], 5 used).
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, z: &DevBuf, cam: &DevBuf, sigma: f32, ctx: &Context, sc: &Scratch, out_z: &DevBuf, out_cam: &DevBuf,
        st: &Stream) -> Result<()> {
        let fi = F as i64;
        let total = Q_TOKENS + ctx.n + 1;
        // timestep embedding, and the shared modulation
        let emb = DevBuf::alloc_async(FREQ * 2, st)?;
        emb.upload(0, &f16_bytes(&sinusoid(sigma * 1000.0, FREQ, 10000.0, false)), st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        let t_emb = DevBuf::alloc_async(F * 2, st)?;
        let t_mod = DevBuf::alloc_async(MOD * 2, st)?;
        gemm16(emb.ptr(), FREQ, self.g("t1w"), FREQ, self.g("t1b"), t1.ptr(), F, 1, F, FREQ, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_SILU, t1.ptr(), fi, std::ptr::null(), std::ptr::null(), 0.0, null(), null(), t1.ptr(), fi, 1,
            F as i32, s(st)))?;
        gemm16(t1.ptr(), F, self.g("t2w"), F, self.g("t2b"), t_emb.ptr(), F, 1, F, F, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_SILU, t_emb.ptr(), fi, std::ptr::null(), std::ptr::null(), 0.0, null(), null(), t1.ptr(), fi,
            1, F as i32, s(st)))?;
        gemm16(t1.ptr(), F, self.g("adaw"), F, self.g("adab"), t_mod.ptr(), MOD, 1, MOD, F, st)?;

        // tokens: shape latent (+ anchors), image context, camera
        let h = DevBuf::alloc_async(total * F * 2, st)?;
        gemm16(z.ptr(), LATENT_CH, self.g("inw"), LATENT_CH, self.g("inb"), h.ptr(), F, Q_TOKENS, F, LATENT_CH, st)?;
        add(h.ptr(), F, self.pos.ptr(), F, null(), Q_TOKENS, F, st)?;
        for b in 0..self.n_noise {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            run_block(&lay, base, BlockKind::Modulated, h.ptr(), Q_TOKENS, t_mod.ptr(), sc, st)?;
            self.blocks.release(b, st)?;
        }
        crate::cuda::memcpy_d2d(h.at(Q_TOKENS * F * 2), ctx.h.ptr(), ctx.n * F * 2, st)?;
        let cam_row = h.at((total - 1) * F * 2);
        gemm16(cam.ptr(), KPAD, self.g("cam1w"), KPAD, self.g("cam1b"), t1.ptr(), F, 1, F, KPAD, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_GELU_TANH, t1.ptr(), fi, std::ptr::null(), std::ptr::null(), 0.0, null(), null(), t1.ptr(),
            fi, 1, F as i32, s(st)))?;
        gemm16(t1.ptr(), F, self.g("cam2w"), F, self.g("cam2b"), cam_row, F, 1, F, F, st)?;

        for b in self.n_noise + self.n_context..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            run_block(&lay, base, BlockKind::Modulated, h.ptr(), total, t_mod.ptr(), sc, st)?;
            self.blocks.release(b, st)?;
        }

        // shift and scale from the table and the timestep embedding (not the modulation)
        let fin = DevBuf::alloc_async(2 * F * 2, st)?;
        crate::cuda::memcpy_d2d(fin.ptr(), self.g("shift"), 2 * F * 2, st)?;
        add(fin.ptr(), F, t_emb.ptr(), 0, null(), 2, F, st)?;
        let (shift, scale) = (fin.ptr() as P, fin.at(F * 2) as P);
        k!(gpu_prologue_rows_b(DT_F16, PRO_LN_MOD, h.ptr(), fi, std::ptr::null(), std::ptr::null(), LN_EPS, scale, shift, sc.ln.ptr(), fi,
            Q_TOKENS as i32, F as i32, s(st)))?;
        gemm16(sc.ln.ptr(), F, self.g("outw"), F, self.g("outb"), out_z.ptr(), LATENT_CH, Q_TOKENS, LATENT_CH, F, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_LN_MOD, cam_row, fi, std::ptr::null(), std::ptr::null(), LN_EPS, scale, shift, sc.ln.ptr(), fi,
            1, F as i32, s(st)))?;
        gemm16(sc.ln.ptr(), F, self.g("camow"), F, self.g("camob"), out_cam.ptr(), KPAD, 1, KPAD, F, st)
    }
}

// ---------------------------------------------------------------------------------------------
// OctreeGaussianDecoder
// ---------------------------------------------------------------------------------------------

const OCTREE_LEVELS: usize = 8;
const GAUSSIANS_PER_POINT: usize = 32;
/// `_xyz`, `_features_dc`, `_scaling`, `_rotation`, `_opacity`, `_offset_scale`
const GS_OUT: usize = GAUSSIANS_PER_POINT * (3 + 3 + 3 + 4 + 1 + 1);

pub struct Decoder {
    globals: DevBuf,
    gl: Layout,
    /// the octree's cross-only blocks, then the gaussian decoder's
    blocks: WeightSet,
    n_octree: usize,
    perturbation: Vec<f32>,
    base_offset_scale: f32,
}

fn cross_layout(l: &mut Layout, st: &SafeTensors, p: &str, prefix: &str) -> Result<()> {
    for (key, name) in [("q", "to_q.weight"), ("qb", "to_q.bias"), ("kv", "to_kv.weight"), ("kvb", "to_kv.bias"), ("o", "to_out.weight"),
        ("ob", "to_out.bias"), ("qn", "q_rms_norm.gamma"), ("kn", "k_rms_norm.gamma")]
    {
        l.raw(st, &format!("{prefix}{key}"), &[format!("{p}{name}")])?;
    }
    Ok(())
}

impl Decoder {
    pub fn load(st: &SafeTensors, threads: usize, s0: &Stream) -> Result<Self> {
        if !is_decoder(|k| st.has(k)) {
            bail!("not TripoSplat's octree gaussian decoder");
        }
        let count = |p: &str| (0..).take_while(|i| st.has(&format!("{p}.blocks.{i}.mlp.mlp.0.weight"))).count();
        let (n_octree, n_gs) = (count("octree"), count("gs"));
        let mut gl = Layout::new();
        for m in ["octree", "gs"] {
            gl.bytes(&format!("{m}.inw"), pad_cols(st, &format!("{m}.in_proj.weight"), KPAD)?);
            for (key, name) in [("inb", "in_proj.bias"), ("ilw", "input_layer.weight"), ("ilb", "input_layer.bias"), ("outw", "out_proj.weight"),
                ("outb", "out_proj.bias")]
            {
                gl.raw(st, &format!("{m}.{key}"), &[format!("{m}.{name}")])?;
            }
        }
        for (key, name) in [("l1w", "l_embedder.mlp.0.weight"), ("l1b", "l_embedder.mlp.0.bias"), ("l2w", "l_embedder.mlp.2.weight"),
            ("l2b", "l_embedder.mlp.2.bias"), ("adaw", "adaLN_modulation.1.weight"), ("adab", "adaLN_modulation.1.bias")]
        {
            gl.raw(st, &format!("octree.{key}"), &[format!("octree.{name}")])?;
        }
        let mut host = vec![0u8; gl.size];
        gl.pack(st, &mut host)?;
        let globals = DevBuf::alloc(gl.size)?;
        globals.upload(0, &host, s0)?;
        s0.sync()?;
        let mut layouts = Vec::new();
        for i in 0..n_octree {
            let p = format!("octree.blocks.{i}.");
            let mut l = Layout::new();
            cross_layout(&mut l, st, &format!("{p}cross_attn."), "")?;
            for (key, name) in [("m1", "mlp.mlp.0.weight"), ("m1b", "mlp.mlp.0.bias"), ("m2", "mlp.mlp.2.weight"), ("m2b", "mlp.mlp.2.bias")] {
                l.raw(st, key, &[format!("{p}{name}")])?;
            }
            layouts.push(l);
        }
        for i in 0..n_gs {
            let p = format!("gs.blocks.{i}.");
            let mut l = Layout::new();
            cross_layout(&mut l, st, &format!("{p}cross_attn."), "")?;
            for (key, name) in [("sqkv", "self_attn.to_qkv.weight"), ("sqkvb", "self_attn.to_qkv.bias"), ("so", "self_attn.to_out.weight"),
                ("sob", "self_attn.to_out.bias"), ("sqn", "self_attn.q_rms_norm.gamma"), ("skn", "self_attn.k_rms_norm.gamma"),
                ("m1", "mlp.mlp.0.weight"), ("m1b", "mlp.mlp.0.bias"), ("m2", "mlp.mlp.2.weight"), ("m2b", "mlp.mlp.2.bias")]
            {
                l.raw(st, key, &[format!("{p}{name}")])?;
            }
            l.f32(st, "n2w", &format!("{p}norm2.weight"), 0.0)?;
            l.f32(st, "n2b", &format!("{p}norm2.bias"), 0.0)?;
            layouts.push(l);
        }
        let p = plan(&layouts, usize::MAX, 0);
        let blocks = WeightSet::load("triposplat decoder", st, layouts, p, threads)?;
        let perturbation = st.f32_vec("gs.points_offset_perturbation", false)?;
        if perturbation.len() != GAUSSIANS_PER_POINT * 3 {
            bail!("expected {GAUSSIANS_PER_POINT} gaussians per point");
        }
        let base_offset_scale = st.f32_vec("gs.base_offset_scale", false)?[0];
        Ok(Decoder { globals, gl, blocks, n_octree, perturbation, base_offset_scale })
    }

    fn g(&self, key: &str) -> PM {
        self.globals.at(self.gl.off(key))
    }

    /// `in_proj(x) + pos_embedder(x)`, then `input_layer`: points f32 [n, 3] to fp16 [n, F].
    fn embed_points(&self, m: &str, pts: &[[f32; 3]], out: &DevBuf, tmp: &DevBuf, st: &Stream) -> Result<()> {
        let n = pts.len();
        let mut x = vec![0f32; n * KPAD];
        for (i, p) in pts.iter().enumerate() {
            x[i * KPAD..i * KPAD + 3].copy_from_slice(p);
        }
        let xd = DevBuf::alloc_async(x.len() * 2, st)?;
        xd.upload(0, &f16_bytes(&x), st)?;
        gemm16(xd.ptr(), KPAD, self.g(&format!("{m}.inw")), KPAD, self.g(&format!("{m}.inb")), tmp.ptr(), F, n, F, KPAD, st)?;
        let pe = DevBuf::alloc_async(n * F * 2, st)?;
        pe.upload(0, &f16_bytes(&position_embedding(pts, F, false)), st)?;
        add(tmp.ptr(), F, pe.ptr(), F, null(), n, F, st)?;
        gemm16(tmp.ptr(), F, self.g(&format!("{m}.ilw")), F, self.g(&format!("{m}.ilb")), out.ptr(), F, n, F, F, st)
    }

    /// Per-block cross-attention keys (RMS-normed) and values of the latent: fp16 [blocks][n][k | v].
    fn latent_kv(&mut self, latent: &DevBuf, n: usize, range: std::ops::Range<usize>, st: &Stream) -> Result<DevBuf> {
        let kv = DevBuf::alloc_async(range.len() * n * KV * 2, st)?;
        for (j, b) in range.enumerate() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let out = kv.at(j * n * KV * 2);
            gemm16(latent.ptr(), LATENT_CH, off(base, lay.off("kv")), LATENT_CH, off(base, lay.off("kvb")), out, KV, n, KV, LATENT_CH, st)?;
            k!(gpu_ts_repo_qk(out, KV as i64, 0, null(), 0, std::ptr::null(), off(base, lay.off("kn")), null(), RMS_EPS, n as i32, H as i32, 0,
                s(st)))?;
            self.blocks.release(b, st)?;
        }
        Ok(kv)
    }

    /// Cross attention of `x` (normalized rows in `sc.ln`) on one block's cached latent keys and values; the output
    /// projection lands in `sc.ln`.
    fn cross(&self, lay: &Layout, base: PM, kv: PM, l: usize, lk: usize, sc: &Scratch, st: &Stream) -> Result<()> {
        let w = |key: &str| off(base, lay.off(key));
        gemm16(sc.ln.ptr(), F, w("q"), F, w("qb"), sc.qkv.ptr(), F, l, F, F, st)?;
        k!(gpu_ts_repo_qk(sc.qkv.ptr(), F as i64, 0, null(), 0, std::ptr::null(), w("qn"), null(), RMS_EPS, l as i32, H as i32, 0, s(st)))?;
        attention(sc.qkv.ptr(), F, kv, off(kv, F * 2), KV, sc.attn.ptr(), l, lk, &sc.ws, st)?;
        gemm16(sc.attn.ptr(), F, w("o"), F, w("ob"), sc.ln.ptr(), F, l, F, F, st)
    }

    /// `OctreeProbabilityFixedlenDecoder` at one level: child-occupancy probabilities (softmax of the fp16 logits) of
    /// each parent, [n][8].
    fn octree_probs(&mut self, parents: &[[f32; 3]], res: usize, kv: &DevBuf, lk: usize, sc: &Scratch, st: &Stream) -> Result<Vec<[f32; 8]>> {
        let n = parents.len();
        let fi = F as i64;
        let x = DevBuf::alloc_async(n * F * 2, st)?;
        self.embed_points("octree", parents, &x, &sc.attn, st)?;
        // level embedding, then the shared modulation
        let e = DevBuf::alloc_async(FREQ * 2, st)?;
        e.upload(0, &f16_bytes(&sinusoid(res as f32, FREQ, 1024.0, true)), st)?;
        let t1 = DevBuf::alloc_async(F * 2, st)?;
        let t2 = DevBuf::alloc_async(F * 2, st)?;
        gemm16(e.ptr(), FREQ, self.g("octree.l1w"), FREQ, self.g("octree.l1b"), t1.ptr(), F, 1, F, FREQ, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_SILU, t1.ptr(), fi, std::ptr::null(), std::ptr::null(), 0.0, null(), null(), t1.ptr(), fi, 1,
            F as i32, s(st)))?;
        gemm16(t1.ptr(), F, self.g("octree.l2w"), F, self.g("octree.l2b"), t2.ptr(), F, 1, F, F, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_SILU, t2.ptr(), fi, std::ptr::null(), std::ptr::null(), 0.0, null(), null(), t1.ptr(), fi, 1,
            F as i32, s(st)))?;
        gemm16(t1.ptr(), F, self.g("octree.adaw"), F, self.g("octree.adab"), sc.modv.ptr(), MOD, 1, MOD, F, st)?;
        let mo = |j: usize| off(sc.modv.ptr(), j * F * 2) as P;
        for b in 0..self.n_octree {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let w = |key: &str| off(base, lay.off(key));
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_MOD, x.ptr(), fi, std::ptr::null(), std::ptr::null(), EPS, mo(1), mo(0), sc.ln.ptr(), fi,
                n as i32, F as i32, s(st)))?;
            self.cross(&lay, base, kv.at(b * lk * KV * 2), n, lk, sc, st)?;
            add(x.ptr(), F, sc.ln.ptr(), F, mo(2), n, F, st)?;
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_MOD, x.ptr(), fi, std::ptr::null(), std::ptr::null(), EPS, mo(4), mo(3), sc.ln.ptr(), fi,
                n as i32, F as i32, s(st)))?;
            mlp(sc.ln.ptr(), w("m1"), w("m1b"), w("m2"), w("m2b"), sc.hid.ptr(), sc.attn.ptr(), n, st)?;
            add(x.ptr(), F, sc.attn.ptr(), F, mo(5), n, F, st)?;
            self.blocks.release(b, st)?;
        }
        k!(gpu_prologue_rows_b(DT_F16, crate::ffi::PRO_LAYERNORM, x.ptr(), fi, std::ptr::null(), std::ptr::null(), LN_EPS, null(), null(),
            sc.ln.ptr(), fi, n as i32, F as i32, s(st)))?;
        let logits = DevBuf::alloc_async(n * 8 * 2, st)?;
        gemm16(sc.ln.ptr(), F, self.g("octree.outw"), F, self.g("octree.outb"), logits.ptr(), 8, n, 8, F, st)?;
        let v = download_f16(&logits, n * 8, st)?;
        Ok(v.chunks_exact(8)
            .map(|l| {
                let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let e: Vec<f32> = l.iter().map(|x| (x - m).exp()).collect();
                let sum: f32 = e.iter().sum();
                std::array::from_fn(|i| r16(e[i] / sum))
            })
            .collect())
    }

    /// `ElasticGaussianFixedlenDecoder`: per-point gaussian features, fp16 values [n, GS_OUT].
    fn gaussians(&mut self, points: &[[f32; 3]], latent: &DevBuf, lk: usize, sc: &Scratch, st: &Stream) -> Result<Vec<f32>> {
        let n = points.len();
        let fi = F as i64;
        let x = DevBuf::alloc_async(n * F * 2, st)?;
        self.embed_points("gs", points, &x, &sc.attn, st)?;
        let kvb = DevBuf::alloc_async(lk * KV * 2, st)?;
        for b in self.n_octree..self.blocks.len() {
            let base = self.blocks.acquire(b, st)?;
            let lay = self.blocks.layouts[b].clone();
            let w = |key: &str| off(base, lay.off(key));
            // self attention
            k!(gpu_prologue_rows_b(DT_F16, crate::ffi::PRO_LAYERNORM, x.ptr(), fi, std::ptr::null(), std::ptr::null(), EPS, null(), null(),
                sc.ln.ptr(), fi, n as i32, F as i32, s(st)))?;
            gemm16(sc.ln.ptr(), F, w("sqkv"), F, w("sqkvb"), sc.qkv.ptr(), QKV, n, QKV, F, st)?;
            k!(gpu_ts_repo_qk(sc.qkv.ptr(), QKV as i64, F as i32, null(), 0, std::ptr::null(), w("sqn"), w("skn"), RMS_EPS, n as i32, H as i32,
                1, s(st)))?;
            let q = sc.qkv.ptr();
            attention(q, QKV, off(q, F * 2), off(q, KV * 2), QKV, sc.attn.ptr(), n, n, &sc.ws, st)?;
            gemm16(sc.attn.ptr(), F, w("so"), F, w("sob"), sc.ln.ptr(), F, n, F, F, st)?;
            add(x.ptr(), F, sc.ln.ptr(), F, null(), n, F, st)?;
            // cross attention on the latent
            gemm16(latent.ptr(), LATENT_CH, w("kv"), LATENT_CH, w("kvb"), kvb.ptr(), KV, lk, KV, LATENT_CH, st)?;
            k!(gpu_ts_repo_qk(kvb.ptr(), KV as i64, 0, null(), 0, std::ptr::null(), w("kn"), null(), RMS_EPS, lk as i32, H as i32, 0, s(st)))?;
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE, x.ptr(), fi, w("n2w"), w("n2b"), EPS, null(), null(),
                sc.ln.ptr(), fi, n as i32, F as i32, s(st)))?;
            self.cross(&lay, base, kvb.ptr(), n, lk, sc, st)?;
            add(x.ptr(), F, sc.ln.ptr(), F, null(), n, F, st)?;
            // MLP
            k!(gpu_prologue_rows_b(DT_F16, crate::ffi::PRO_LAYERNORM, x.ptr(), fi, std::ptr::null(), std::ptr::null(), EPS, null(), null(),
                sc.ln.ptr(), fi, n as i32, F as i32, s(st)))?;
            mlp(sc.ln.ptr(), w("m1"), w("m1b"), w("m2"), w("m2b"), sc.hid.ptr(), sc.attn.ptr(), n, st)?;
            add(x.ptr(), F, sc.attn.ptr(), F, null(), n, F, st)?;
            self.blocks.release(b, st)?;
        }
        k!(gpu_prologue_rows_b(DT_F16, crate::ffi::PRO_LAYERNORM, x.ptr(), fi, std::ptr::null(), std::ptr::null(), LN_EPS, null(), null(),
            sc.ln.ptr(), fi, n as i32, F as i32, s(st)))?;
        let feats = DevBuf::alloc_async(n * GS_OUT * 2, st)?;
        gemm16(sc.ln.ptr(), F, self.g("gs.outw"), F, self.g("gs.outb"), feats.ptr(), GS_OUT, n, GS_OUT, F, st)?;
        download_f16(&feats, n * GS_OUT, st)
    }

    /// `OctreeGaussianDecoder.decode` then `render_tensors`: `latent` f32 [Q_TOKENS, 16] token major.
    pub fn decode(&mut self, latent: &[f32], num_gaussians: usize, seed: u64, st: &Stream, progress: &mut Progress) -> Result<Splat> {
        let lk = latent.len() / LATENT_CH;
        let points_n = (num_gaussians / GAUSSIANS_PER_POINT).max(1);
        let lat = DevBuf::alloc_async(latent.len() * 2, st)?;
        lat.upload(0, &f16_bytes(latent), st)?;
        let kv = self.latent_kv(&lat, lk, 0..self.n_octree, st)?;
        let sc = Scratch::new(points_n, lk, st)?;
        let mut g = Mt19937::new(seed);

        // octree sampling: each level splits every parent's count over its 8 children
        let mut coords: Vec<[i64; 3]> = vec![[0, 0, 0]];
        let mut counts: Vec<usize> = vec![points_n];
        for lv in 1..=OCTREE_LEVELS {
            progress("decode", lv - 1, OCTREE_LEVELS + 1)?;
            let res_p = 1i64 << (lv - 1);
            let parents: Vec<[f32; 3]> = coords.iter().map(|c| c.map(|v| (v as f32 + 0.5) / res_p as f32)).collect();
            let probs = self.octree_probs(&parents, 1 << lv, &kv, lk, &sc, st)?;
            let sampled = sample_probs(&probs, &counts, &mut g);
            let (mut nc, mut nn) = (Vec::new(), Vec::new());
            for (p, c) in coords.iter().enumerate() {
                for (child, &cnt) in sampled[p].iter().enumerate() {
                    if cnt > 0 {
                        let o = [(child & 1) as i64, ((child >> 1) & 1) as i64, ((child >> 2) & 1) as i64];
                        nc.push([c[0] * 2 + o[0], c[1] * 2 + o[1], c[2] * 2 + o[2]]);
                        nn.push(cnt);
                    }
                }
            }
            coords = nc;
            counts = nn;
        }
        drop(kv);
        let res = (1u64 << OCTREE_LEVELS) as f32;
        let mut points = Vec::with_capacity(points_n);
        for (c, &n) in coords.iter().zip(&counts) {
            for _ in 0..n {
                points.push(*c);
            }
        }
        if points.len() != points_n {
            bail!("the octree sampled {} points, not {points_n}", points.len());
        }
        let points: Vec<[f32; 3]> = points.iter().map(|c| std::array::from_fn(|a| (c[a] as f32 + g.uniform_f32()) / res)).collect();
        progress("decode", OCTREE_LEVELS, OCTREE_LEVELS + 1)?;
        let feats = self.gaussians(&points, &lat, lk, &sc, st)?;
        progress("decode", OCTREE_LEVELS + 1, OCTREE_LEVELS + 1)?;
        Ok(self.assemble(&points, &feats))
    }

    /// `build_gaussian_models` and `GaussianModel.render_tensors`, with ComfyUI's fp16 rounding.
    fn assemble(&self, points: &[[f32; 3]], feats: &[f32]) -> Splat {
        let ng = GAUSSIANS_PER_POINT;
        let (xyz, dc, scl, rot, opa, osc) = (0, ng * 3, ng * 6, ng * 9, ng * 13, ng * 14);
        let softplus = |x: f32| if x > 20.0 { x } else { x.exp().ln_1p() };
        // the representation's constants, in f32 like the tensors torch makes of them
        let scale_bias = {
            let x = 0.004f32;
            x + (-(-x).exp_m1()).ln()
        };
        let opacity_bias = (0.1f32 / (1.0 - 0.1f32)).ln();
        let min_kernel2 = 0.0009f64.powi(2) as f32;
        let n = points.len() * ng;
        let mut sp = Splat { positions: Vec::with_capacity(n), scales: Vec::with_capacity(n), rotations: Vec::with_capacity(n),
            opacities: Vec::with_capacity(n), sh: Vec::with_capacity(n) };
        for (p, h) in points.iter().zip(feats.chunks_exact(GS_OUT)) {
            for gi in 0..ng {
                let os = r16(softplus(r16(h[osc + gi] + self.base_offset_scale)));
                let mut pos = [0f32; 3];
                for c in 0..3 {
                    let mut o = r16(h[xyz + gi * 3 + c] + self.perturbation[gi * 3 + c]);
                    o = r16(o.tanh());
                    o = r16(o * 0.5);
                    o = r16(o * 1.5);
                    o = r16(o * os);
                    pos[c] = (o + p[c]) * 1.0 + -0.5;
                }
                let s: [f32; 3] = std::array::from_fn(|c| {
                    let v = r16(softplus(r16(h[scl + gi * 3 + c] + scale_bias)));
                    r16(r16(r16(v * v) + min_kernel2).sqrt())
                });
                let op = r16(1.0 / (1.0 + (-r16(h[opa + gi] + opacity_bias)).exp()));
                // the rotation's learning-rate scale (0.1) in fp16, then the identity bias in f32
                let r: [f32; 4] = std::array::from_fn(|c| r16(h[rot + gi * 4 + c] * 0.1));
                let q = [r[0] + 1.0, r[1], r[2], r[3]];
                // the object frame to the viewer's (Y up): T = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                sp.positions.push([pos[0], -pos[2], pos[1]]);
                sp.rotations.push(rotate_quat(q));
                sp.scales.push(s);
                sp.opacities.push(op);
                sp.sh.push([h[dc + gi * 3], h[dc + gi * 3 + 1], h[dc + gi * 3 + 2]]);
            }
        }
        sp
    }
}

/// `sample_probs`: systematic resampling of each parent's count over its 8 children.
fn sample_probs(probs: &[[f32; 8]], counts: &[usize], g: &mut Mt19937) -> Vec<[usize; 8]> {
    if counts.iter().all(|&c| c == 0) {
        return vec![[0; 8]; counts.len()];
    }
    let rands: Vec<f32> = counts.iter().map(|_| g.uniform_f32()).collect();
    probs
        .iter()
        .zip(counts)
        .zip(rands)
        .map(|((p, &cnt), u0)| {
            let p = p.map(|v| v.max(0.0));
            let sum: f32 = p.iter().sum();
            let p = if sum == 0.0 { [1.0 / 8.0; 8] } else { p.map(|v| v / sum.max(1.0)) };
            // cumsum clamped to 1 - 1e-12, which is 1 in f32
            let mut cdf = [0f32; 8];
            let mut acc = 0f32;
            for i in 0..8 {
                acc += p[i];
                cdf[i] = acc.min(1.0);
            }
            let c = cnt.max(1) as f32;
            let mut out = [0usize; 8];
            for j in 0..cnt {
                let u = ((u0 + j as f32) / c).min(1.0);
                let idx = cdf.iter().position(|&v| v >= u).unwrap_or(7).min(7);
                out[idx] += 1;
            }
            out
        })
        .collect()
}

/// `_matrix_to_quat(T @ _quat_to_matrix(q))`, normalized, for the axis transform T.
fn rotate_quat(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let [w, x, y, z] = q.map(|v| v / n);
    let r = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ];
    // T's rows: (1, 0, 0), (0, 0, -1), (0, 1, 0)
    let m = [r[0], r[2].map(|v| -v), r[1]];
    let trace = m[0][0] + m[1][1] + m[2][2];
    let s = (trace + 1.0).max(0.0).sqrt() * 2.0;
    let mut o = if s != 0.0 {
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] >= m[1][1] && m[0][0] >= m[2][2] {
        let s1 = (1.0 + m[0][0] - m[1][1] - m[2][2]).max(0.0).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s1, 0.25 * s1, (m[0][1] + m[1][0]) / s1, (m[0][2] + m[2][0]) / s1]
    } else if m[1][1] > m[0][0] && m[1][1] >= m[2][2] {
        let s2 = (1.0 + m[1][1] - m[0][0] - m[2][2]).max(0.0).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s2, (m[0][1] + m[1][0]) / s2, 0.25 * s2, (m[1][2] + m[2][1]) / s2]
    } else {
        let s3 = (1.0 + m[2][2] - m[0][0] - m[1][1]).max(0.0).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s3, (m[0][2] + m[2][0]) / s3, (m[1][2] + m[2][1]) / s3, 0.25 * s3]
    };
    let n = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]).sqrt();
    o.iter_mut().for_each(|v| *v /= n);
    o
}

// ---------------------------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------------------------

/// `TripoSplatConditioning`: DINOv3 tokens f32 [n, 1280] and the Flux 2 latent f32 [128, hw], of the preprocessed
/// image f32 [size, size, 3].
pub fn condition(paths: &Paths, image: &[f32], size: usize, threads: usize, st: &Stream) -> Result<(Vec<f32>, Vec<f32>)> {
    let dino_st = SafeTensors::open(&paths.dino)?;
    let mut dino = Dino3::load_vith(&dino_st, threads, st)?;
    let tokens = dino.encode(image, size, st)?;
    let n = tokens.buf.size() / 4;
    let mut b = vec![0u8; n * 4];
    tokens.buf.download(0, &mut b, st)?;
    st.sync()?;
    drop((tokens, dino));
    let feat1: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let vae = flux2vae::Vae::load(&paths.flux_vae, st)?;
    let reference = vae.encode(image, size, size, st)?;
    st.sync()?;
    Ok((feat1, reference))
}

/// The `KSampler` over the nested latent: the shape tokens f32 [Q_TOKENS, 16] token major (the camera token is
/// sampled with them and dropped, as `VAEDecodeTripoSplat` does).
pub fn sample(flow: &mut Flow, feat1: &[f32], reference: &[f32], p: &Params, st: &Stream, progress: &mut Progress) -> Result<Vec<f32>> {
    let cond = flow.context(feat1, reference, st)?;
    let uncond = flow.context(&vec![0.0; feat1.len()], &vec![0.0; reference.len()], st)?;
    let total = Q_TOKENS + cond.n + 1;
    let sc = Scratch::new(total, total, st)?;
    let ms = ModelSampling::Flow { shift: p.shift };
    let sigmas = samplers::sigmas(ms, &p.sampler, &p.scheduler, p.steps, (0, 0), None)?;
    let nz = Q_TOKENS * LATENT_CH;
    let noise: Vec<f32> = torch_randn_streams(p.seed, &[nz, CAM_CH]).concat();
    let z = DevBuf::alloc_async(nz * 2, st)?;
    let cam = DevBuf::alloc_async(KPAD * 2, st)?;
    let out_z = DevBuf::alloc_async(nz * 2, st)?;
    let out_cam = DevBuf::alloc_async(KPAD * 2, st)?;
    let tweaks = CfgTweaks::default();
    let t0 = Instant::now();
    let mut evals = 0;
    let run = |flow: &mut Flow, x: &[f32], sigma: f32, ctx: &Context| -> Result<Vec<f32>> {
        z.upload(0, &f16_bytes(&x[..nz]), st)?;
        let mut c = [0f32; KPAD];
        c[..CAM_CH].copy_from_slice(&x[nz..]);
        cam.upload(0, &f16_bytes(&c), st)?;
        flow.predict(&z, &cam, sigma, ctx, &sc, &out_z, &out_cam, st)?;
        let vz = download_f16(&out_z, nz, st)?;
        let vc = download_f16(&out_cam, CAM_CH, st)?;
        Ok(x.iter().zip(vz.iter().chain(&vc)).map(|(x, v)| x - v * sigma).collect())
    };
    let mut eval = |x: &[f32], sigma: f32, mode: Uncond| -> Result<Eval> {
        let cond_den = run(flow, x, sigma, &cond)?;
        let guided = (p.cfg - 1.0).abs() > 1e-9;
        let un = if guided || mode == Uncond::Force { Some(run(flow, x, sigma, &uncond)?) } else { None };
        let (denoised, u) = tweaks.guide(x, &cond_den, un.clone(), p.cfg as f32, sigma);
        evals += 1;
        Ok(Eval { denoised, uncond: un.map(|_| u) })
    };
    let mut step = |i: usize, t: usize| progress("sample", i, t);
    let batch = samplers::Batch::single(unsafe { crate::ffi::gpu_randn_threads() }.max(1) as usize);
    let out = samplers::sample(&p.sampler, ms, &sigmas, &noise, None, p.seed, batch, &mut eval, &mut step)?;
    eprintln!("[triposplat] {} steps, {evals} evaluations in {:.1}s", sigmas.len() - 1, t0.elapsed().as_secs_f64());
    Ok(out[..nz].to_vec())
}

/// Runs the whole pipeline on an image (f32 RGB [h, w, 3]) and its object mask (1 on the object).
#[allow(clippy::too_many_arguments)]
pub fn run(paths: &Paths, rgb: &[f32], mask: &[f32], w: usize, h: usize, p: &Params, threads: usize, progress: &mut Progress) -> Result<Splat> {
    let n = p.num_gaussians;
    if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&n) || n % GAUSSIANS_PER_POINT != 0 {
        bail!("the number of gaussians must be a multiple of {GAUSSIANS_PER_POINT} from {MIN_GAUSSIANS} to {MAX_GAUSSIANS}");
    }
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let st = Stream::new()?;
    let t_all = Instant::now();
    let image = preprocess(rgb, mask, w, h, p.erode_radius, IMAGE_SIZE)?;
    progress("encode", 0, 1)?;
    let (feat1, reference) = condition(paths, &image, IMAGE_SIZE, threads, &st)?;
    progress("encode", 1, 1)?;
    st.sync()?;
    crate::cuda::trim_pool()?;
    let dit_st = SafeTensors::open(&paths.dit)?;
    let mut flow = Flow::load(&dit_st, threads, &st)?;
    let latent = sample(&mut flow, &feat1, &reference, p, &st, progress)?;
    drop(flow);
    st.sync()?;
    crate::cuda::trim_pool()?;
    let dec_st = SafeTensors::open(&paths.decoder)?;
    let mut dec = Decoder::load(&dec_st, threads, &st)?;
    let splat = dec.decode(&latent, n, p.decode_seed, &st, progress)?;
    eprintln!("[triposplat] done in {:.1}s: {} gaussians", t_all.elapsed().as_secs_f64(), splat.positions.len());
    Ok(splat)
}

/// `SaveGaussianSplat` as `.ply`: the standard 3DGS layout (log scales, logit opacities), SH degree 0.
pub fn ply(s: &Splat) -> Vec<u8> {
    let attrs = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
        "rot_2", "rot_3"];
    let n = s.positions.len();
    let mut out = format!("ply\nformat binary_little_endian 1.0\nelement vertex {n}\n").into_bytes();
    for a in attrs {
        out.extend_from_slice(format!("property float {a}\n").as_bytes());
    }
    out.extend_from_slice(b"end_header\n");
    out.reserve(n * attrs.len() * 4);
    for i in 0..n {
        let op = s.opacities[i].clamp(1e-6, 1.0 - 1e-6);
        let mut row = Vec::with_capacity(attrs.len());
        row.extend_from_slice(&s.positions[i]);
        row.extend_from_slice(&[0.0; 3]);
        row.extend_from_slice(&s.sh[i]);
        row.push((op / (1.0 - op)).ln());
        row.extend(s.scales[i].iter().map(|v| v.max(1e-8).ln()));
        row.extend_from_slice(&s.rotations[i]);
        for v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Comparison with ComfyUI
// ---------------------------------------------------------------------------------------------

/// Compares every stage with `ref_path`, a dump of `tools/triposplat_ref.py` on the same image; each stage starts
/// from the reference's input to it.
pub fn check(paths: &Paths, rgba: &[u8], w: usize, h: usize, ref_path: &std::path::Path, p: &Params, threads: usize) -> Result<()> {
    use crate::pixal3d::report;
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let st = Stream::new()?;
    let r = SafeTensors::open(ref_path)?;
    let rf = |name: &str| r.f32_vec(name, false);
    let (rgb, mask) = crate::pixal3d::split_rgba(rgba);
    let image = preprocess(&rgb, &mask, w, h, p.erode_radius, IMAGE_SIZE)?;
    let ri = rf("image")?;
    report("preprocess", &image, &ri);
    let differing = image.iter().zip(&ri).filter(|(a, b)| a != b).count();
    eprintln!("{:24} {differing} of {} values differ", "", ri.len());

    let (feat1, reference) = condition(paths, &ri, IMAGE_SIZE, threads, &st)?;
    let (rfeat, rref) = (rf("feature1")?, rf("reference")?);
    report("feature1", &feat1, &rfeat);
    report("feature1 (cls+regs)", &feat1[..5 * COND_CH], &rfeat[..5 * COND_CH]);
    report("reference latent", &reference, &rref);
    st.sync()?;
    crate::cuda::trim_pool()?;

    let dit = SafeTensors::open(&paths.dit)?;
    let mut flow = Flow::load(&dit, threads, &st)?;
    let n = rfeat.len() / COND_CH;
    let total = Q_TOKENS + n + 1;
    let sc = Scratch::new(total, total, &st)?;
    let nz = Q_TOKENS * LATENT_CH;
    let (z, cam) = (DevBuf::alloc_async(nz * 2, &st)?, DevBuf::alloc_async(KPAD * 2, &st)?);
    let (oz, oc) = (DevBuf::alloc_async(nz * 2, &st)?, DevBuf::alloc_async(KPAD * 2, &st)?);
    for (name, f1, rl) in [("cond", rfeat.clone(), rref.clone()), ("uncond", vec![0.0; rfeat.len()], vec![0.0; rref.len()])] {
        if !r.has(&format!("eval_{name}_z")) {
            eprintln!("{:24} not in the dump", format!("eval {name}"));
            continue;
        }
        let ctx = flow.context(&f1, &rl, &st)?;
        z.upload(0, &f16_bytes(&rf(&format!("eval_{name}_z"))?), &st)?;
        let mut c = rf(&format!("eval_{name}_cam"))?;
        c.resize(KPAD, 0.0);
        cam.upload(0, &f16_bytes(&c), &st)?;
        let t = rf(&format!("eval_{name}_t"))?[0];
        flow.predict(&z, &cam, t / 1000.0, &ctx, &sc, &oz, &oc, &st)?;
        report(&format!("eval {name} z"), &download_f16(&oz, nz, &st)?, &rf(&format!("eval_{name}_out_z"))?);
        report(&format!("eval {name} cam"), &download_f16(&oc, CAM_CH, &st)?, &rf(&format!("eval_{name}_out_cam"))?);
    }
    drop((z, cam, oz, oc, sc));
    let mut quiet = |_: &str, _: usize, _: usize| Ok(());
    let latent = sample(&mut flow, &rfeat, &rref, p, &st, &mut quiet)?;
    let rl = rf("latent")?;
    report("sampled latent", &latent, &rl);
    drop(flow);
    st.sync()?;
    crate::cuda::trim_pool()?;

    // the decode, from the reference's latent
    let dec_st = SafeTensors::open(&paths.decoder)?;
    let mut dec = Decoder::load(&dec_st, threads, &st)?;
    let splat = dec.decode(&rl, p.num_gaussians, p.decode_seed, &st, &mut quiet)?;
    let rpos = rf("splat_positions")?;
    let ours: Vec<f32> = splat.positions.iter().flatten().copied().collect();
    report("splat positions", &ours, &rpos);
    // the gaussian decoder on the reference's points
    let rp = rf("points")?;
    let pts: Vec<[f32; 3]> = rp.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let lat = DevBuf::alloc_async(rl.len() * 2, &st)?;
    lat.upload(0, &f16_bytes(&rl), &st)?;
    let sc = Scratch::new(pts.len(), rl.len() / LATENT_CH, &st)?;
    let feats = dec.gaussians(&pts, &lat, rl.len() / LATENT_CH, &sc, &st)?;
    report("gaussian features", &feats, &rf("gs_features")?);
    let s = dec.assemble(&pts, &rf("gs_features")?);
    report("positions (ref feats)", &s.positions.iter().flatten().copied().collect::<Vec<_>>(), &rpos);
    report("scales (ref feats)", &s.scales.iter().flatten().copied().collect::<Vec<_>>(), &rf("splat_scales")?);
    report("rotations (ref feats)", &s.rotations.iter().flatten().copied().collect::<Vec<_>>(), &rf("splat_rotations")?);
    report("opacities (ref feats)", &s.opacities, &rf("splat_opacities")?);
    report("sh (ref feats)", &s.sh.iter().flatten().copied().collect::<Vec<_>>(), &rf("splat_sh")?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torch.quasirandom.SobolEngine(3, scramble=True, seed=123).draw(8192)
    #[test]
    fn sobol_matches_torch() {
        let s = sobol3(Q_TOKENS, 123);
        assert_eq!(s[0], [0.09025132656097412, 0.5811105370521545, 0.4077436923980713]);
        assert_eq!(s[1], [0.7700406312942505, 0.24629266560077667, 0.6401962041854858]);
        assert_eq!(s[8191], [0.09019839018583298, 0.20313361287117004, 0.6394063234329224]);
    }

    /// torch.manual_seed(5); torch.randn(48); torch.randn(5)
    #[test]
    fn small_randn_matches_torch() {
        let v = torch_randn_streams(5, &[48, 5]);
        assert_eq!(v[1], [0.447927862405777, -0.6088235378265381, 0.23894619941711426, 0.4698884189128876, -1.9540048837661743]);
    }

    /// torch.rand(3, 1, generator=torch.Generator().manual_seed(7))
    #[test]
    fn rand_matches_torch() {
        let mut g = Mt19937::new(7);
        let v: Vec<f32> = (0..3).map(|_| g.uniform_f32()).collect();
        assert_eq!(v, [0.5349225401878357, 0.19880318641662598, 0.6592116951942444]);
    }
}
