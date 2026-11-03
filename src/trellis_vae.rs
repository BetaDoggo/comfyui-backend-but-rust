//! Trellis 2's decoders (f16, as ComfyUI picks for them): the dense sparse-structure decoder, and the sparse UNet
//! decoders of the shape (FlexiDualGrid mesh) and texture (PBR voxels) latents.
//!
//! A submanifold convolution gathers each voxel's 27 neighbours into rows and runs one GEMM against the kernel,
//! which the checkpoint already stores as [out, 3, 3, 3, in]. The dense decoder is the same convolution over every
//! voxel of the grid, so missing neighbours give its zero padding.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F16, PM, PRO_LAYERNORM, PRO_LN_AFFINE, PRO_LN_AFFINE_SILU, PRO_SILU};
use crate::offload::Layout;
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};
use half::f16;

const CHANNELS: [usize; 5] = [1024, 512, 256, 128, 64];
const BLOCKS: [usize; 5] = [4, 16, 8, 4, 0];
const EPS: f32 = 1e-6;
const OUT_EPS: f32 = 1e-5;
const GATHER_BYTES: usize = 256 << 20;

/// `Trellis2ShapeSLAT` and `Trellis2TexSLAT` statistics: decoders take `latent * std + mean`.
pub const SHAPE_MEAN: [f32; 32] = [
    0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
    0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665,
];
pub const SHAPE_STD: [f32; 32] = [
    5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469, 5.652043,
    5.367606, 5.525084, 4.730578, 4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194, 5.634698, 5.235274,
    6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
];
pub const TEX_MEAN: [f32; 32] = [
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479,
    -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
    1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862,
];
pub const TEX_STD: [f32; 32] = [
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343, 2.621215,
    2.381943, 3.186697, 3.021588, 2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999, 2.680878, 2.372054,
    2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190,
];

/// Voxel coordinates (batch, x, y, z) on device, with their neighbour table.
pub struct Voxels {
    pub coords: Vec<[i32; 4]>,
    pub res: usize,
    nbr: DevBuf,
}

impl Voxels {
    pub fn new(coords: Vec<[i32; 4]>, res: usize, st: &Stream) -> Result<Self> {
        let n = coords.len();
        if n == 0 {
            bail!("no voxels");
        }
        let flat: Vec<u8> = coords.iter().flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes())).collect();
        let cd = DevBuf::alloc_async(flat.len(), st)?;
        cd.upload(0, &flat, st)?;
        let cap = (2 * n).next_power_of_two().max(1024);
        let keys = DevBuf::alloc_async(cap * 8, st)?;
        let vals = DevBuf::alloc_async(cap * 4, st)?;
        let r = res as i32;
        k!(gpu_voxel_hash(cd.ptr(), n as i32, r, r, r, keys.ptr(), vals.ptr(), cap as i32, s(st)))?;
        let nbr = DevBuf::alloc_async(n * 27 * 4, st)?;
        k!(gpu_voxel_neighbors(cd.ptr(), n as i32, r, r, r, keys.ptr(), vals.ptr(), cap as i32, nbr.ptr(), s(st)))?;
        Ok(Voxels { coords, res, nbr })
    }

    /// Every voxel of an r^3 grid, x major.
    pub fn dense(r: usize, st: &Stream) -> Result<Self> {
        let coords = (0..r * r * r).map(|i| [0, (i / (r * r)) as i32, ((i / r) % r) as i32, (i % r) as i32]).collect();
        Self::new(coords, r, st)
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }
}

/// One C2S subdivision: which of each voxel's eight children exist (`flat` = voxel * 8 + child, ascending).
pub struct Subdivision {
    pub flat: Vec<i32>,
    pub coords: Vec<[i32; 4]>,
}

impl Subdivision {
    /// `SparseChannel2Spatial._subdivision_plan` from the children's logits (f16 [n, 8]).
    fn from_logits(parent: &[[i32; 4]], logits: &[f16]) -> Self {
        let mut flat = Vec::new();
        let mut coords = Vec::new();
        for (i, c) in parent.iter().enumerate() {
            for sub in 0..8 {
                if logits[i * 8 + sub].to_f32() > 0.0 {
                    flat.push((i * 8 + sub) as i32);
                    coords.push([c[0], 2 * c[1] + (sub & 1) as i32, 2 * c[2] + ((sub >> 1) & 1) as i32, 2 * c[3] + ((sub >> 2) & 1) as i32]);
                }
            }
        }
        Subdivision { flat, coords }
    }
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect()
}

/// A tensor converted to f16 (ComfyUI casts the bf16 weights to the f16 compute dtype), zero-padded to `rows` rows.
fn f16_tensor(st: &SafeTensors, name: &str, rows: Option<usize>) -> Result<Vec<u8>> {
    let mut v = st.f32_vec(name, false)?;
    if let Some(r) = rows {
        let per = v.len() / st.info(name)?.shape[0];
        v.resize(r * per, 0.0);
    }
    Ok(f16_bytes(&v))
}

/// A dense Conv3d kernel [out, in, 3, 3, 3] as the sparse layout [out, 27, in], in f16.
fn dense_kernel(st: &SafeTensors, name: &str, rows: Option<usize>) -> Result<Vec<u8>> {
    let info = st.info(name)?;
    let (co, ci) = (info.shape[0], info.shape[1]);
    let v = st.f32_vec(name, false)?;
    let mut r = vec![0f32; rows.unwrap_or(co) * 27 * ci];
    for o in 0..co {
        for c in 0..ci {
            for kk in 0..27 {
                r[(o * 27 + kk) * ci + c] = v[(o * ci + c) * 27 + kk];
            }
        }
    }
    Ok(f16_bytes(&r))
}

fn ln(st: &SafeTensors, l: &mut Layout, key: &str, name: &str) -> Result<()> {
    l.bytes(&format!("{key}.w"), f32_bytes(&st.f32_vec(&format!("{name}.weight"), false)?.into_iter().map(|x| f16::from_f32(x).to_f32()).collect::<Vec<_>>()));
    l.bytes(&format!("{key}.b"), f32_bytes(&st.f32_vec(&format!("{name}.bias"), false)?.into_iter().map(|x| f16::from_f32(x).to_f32()).collect::<Vec<_>>()));
    Ok(())
}

fn lin(st: &SafeTensors, l: &mut Layout, key: &str, name: &str, rows: Option<usize>) -> Result<()> {
    l.bytes(&format!("{key}.w"), f16_tensor(st, &format!("{name}.weight"), rows)?);
    l.bytes(&format!("{key}.b"), f16_tensor(st, &format!("{name}.bias"), rows)?);
    Ok(())
}

/// Device weights of one decoder.
struct Weights {
    buf: DevBuf,
    l: Layout,
}

impl Weights {
    fn upload(st: &SafeTensors, l: Layout, s0: &Stream) -> Result<Self> {
        let mut host = vec![0u8; l.size];
        l.pack(st, &mut host)?;
        let buf = DevBuf::alloc(l.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Weights { buf, l })
    }

    fn p(&self, key: &str) -> PM {
        self.buf.at(self.l.off(key))
    }
}

/// x (f16 [n, ci]) -> out (f16 [n, co]) through a 3x3x3 submanifold convolution.
fn conv(w: &Weights, key: &str, vox: &Voxels, x: PM, ci: usize, co: usize, out: PM, st: &Stream) -> Result<()> {
    let n = vox.len();
    let kk = 27 * ci;
    let rows = (GATHER_BYTES / (kk * 2)).clamp(1, n);
    let col = DevBuf::alloc_async(rows * kk * 2, st)?;
    let (wp, bp) = (w.p(&format!("{key}.w")), w.p(&format!("{key}.b")));
    for r0 in (0..n).step_by(rows) {
        let r = rows.min(n - r0);
        k!(gpu_sparse_gather(x, ci as i32, vox.nbr.ptr(), r0 as i32, r as i32, col.ptr(), s(st)))?;
        k!(gpu_gemm(col.ptr(), kk as i64, wp, kk as i64, off(out, r0 * co * 2), co as i64, bp, r as i64, co as i64, kk as i64, 1, DT_F16,
            DT_F16, 1.0, 0.0, s(st)))?;
    }
    Ok(())
}

fn linear(w: &Weights, key: &str, x: PM, n: usize, ci: usize, co: usize, out: PM, st: &Stream) -> Result<()> {
    k!(gpu_gemm(x, ci as i64, w.p(&format!("{key}.w")), ci as i64, out, co as i64, w.p(&format!("{key}.b")), n as i64, co as i64, ci as i64,
        1, DT_F16, DT_F16, 1.0, 0.0, s(st)))
}

fn norm(prologue: i32, w: Option<(&Weights, &str)>, x: PM, n: usize, c: usize, eps: f32, out: PM, st: &Stream) -> Result<()> {
    let (nw, nb) = match w {
        Some((w, key)) => (w.p(&format!("{key}.w")) as *const _, w.p(&format!("{key}.b")) as *const _),
        None => (null(), null()),
    };
    k!(gpu_prologue_rows_b(DT_F16, prologue, x, c as i64, nw, nb, eps, null(), null(), out, c as i64, n as i32, c as i32, s(st)))
}

/// What the shape decode leaves for the texture decode and the mesh.
pub struct ShapeOut {
    /// final voxels and their seven outputs (f16 [n, 8], the eighth padding)
    pub voxels: Vec<[i32; 4]>,
    pub feats: Vec<f16>,
    pub subdivisions: Vec<Subdivision>,
}

/// The sparse UNet decoder shared by shape and texture.
pub struct SparseDecoder {
    w: Weights,
    out_ch: usize,
    pred_subdiv: bool,
    ones: DevBuf,
    zeros: DevBuf,
}

impl SparseDecoder {
    fn load(st: &SafeTensors, prefix: &'static str, out_ch: usize, pred_subdiv: bool, s0: &Stream) -> Result<Self> {
        let mut l = Layout::new();
        let p = |x: String| format!("{prefix}.{x}");
        lin(st, &mut l, "from_latent", &p("from_latent".into()), None)?;
        lin(st, &mut l, "out", &p("output_layer".into()), Some(8))?;
        for (i, &nb) in BLOCKS.iter().enumerate() {
            for j in 0..nb {
                let b = format!("{i}.{j}");
                let n = p(format!("blocks.{b}"));
                l.bytes(&format!("{b}.conv.w"), f16_tensor(st, &format!("{n}.conv.weight"), None)?);
                l.bytes(&format!("{b}.conv.b"), f16_tensor(st, &format!("{n}.conv.bias"), None)?);
                ln(st, &mut l, &format!("{b}.norm"), &format!("{n}.norm"))?;
                lin(st, &mut l, &format!("{b}.mlp0"), &format!("{n}.mlp.0"), None)?;
                lin(st, &mut l, &format!("{b}.mlp2"), &format!("{n}.mlp.2"), None)?;
            }
            if i < 4 {
                let b = format!("{i}.up");
                let n = p(format!("blocks.{i}.{nb}"));
                ln(st, &mut l, &format!("{b}.norm1"), &format!("{n}.norm1"))?;
                for c in ["conv1", "conv2"] {
                    l.bytes(&format!("{b}.{c}.w"), f16_tensor(st, &format!("{n}.{c}.weight"), None)?);
                    l.bytes(&format!("{b}.{c}.b"), f16_tensor(st, &format!("{n}.{c}.bias"), None)?);
                }
                if pred_subdiv {
                    lin(st, &mut l, &format!("{b}.subdiv"), &format!("{n}.to_subdiv"), None)?;
                }
            }
        }
        let w = Weights::upload(st, l, s0)?;
        let ones = DevBuf::alloc(1024 * 4)?;
        ones.upload(0, &f32_bytes(&[1.0; 1024]), s0)?;
        let zeros = DevBuf::alloc(1024 * 4)?;
        zeros.upload(0, &f32_bytes(&[0.0; 1024]), s0)?;
        s0.sync()?;
        Ok(SparseDecoder { w, out_ch, pred_subdiv, ones, zeros })
    }

    fn convnext(&self, key: &str, vox: &Voxels, h: &DevBuf, c: usize, st: &Stream) -> Result<()> {
        let n = vox.len();
        let t = DevBuf::alloc_async(n * c * 2, st)?;
        let u = DevBuf::alloc_async(n * c * 2, st)?;
        conv(&self.w, &format!("{key}.conv"), vox, h.ptr(), c, c, t.ptr(), st)?;
        norm(PRO_LN_AFFINE, Some((&self.w, &format!("{key}.norm"))), t.ptr(), n, c, EPS, u.ptr(), st)?;
        drop(t);
        let hid = DevBuf::alloc_async(n * 4 * c * 2, st)?;
        linear(&self.w, &format!("{key}.mlp0"), u.ptr(), n, c, 4 * c, hid.ptr(), st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_SILU, hid.ptr(), (4 * c) as i64, null(), null(), 0.0, null(), null(), hid.ptr(), (4 * c) as i64,
            n as i32, (4 * c) as i32, s(st)))?;
        linear(&self.w, &format!("{key}.mlp2"), hid.ptr(), n, 4 * c, c, u.ptr(), st)?;
        k!(gpu_add_f16(h.ptr(), u.ptr(), (n * c) as i64, s(st)))?;
        Ok(())
    }

    /// `SparseResBlockC2S3d`: returns the children's features and voxels; with `plan` the subdivision is given
    /// (texture), else predicted (shape) and returned.
    fn c2s(&self, i: usize, vox: &Voxels, x: &DevBuf, plan: Option<&Subdivision>, st: &Stream) -> Result<(DevBuf, Voxels, Option<Subdivision>)> {
        let (c, co) = (CHANNELS[i], CHANNELS[i + 1]);
        let key = format!("{i}.up");
        let n = vox.len();
        let mut predicted = None;
        if self.pred_subdiv {
            let lg = DevBuf::alloc_async(n * 8 * 2, st)?;
            linear(&self.w, &format!("{key}.subdiv"), x.ptr(), n, c, 8, lg.ptr(), st)?;
            let mut host = vec![0u8; n * 16];
            lg.download(0, &mut host, st)?;
            st.sync()?;
            let logits: Vec<f16> = host.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]])).collect();
            predicted = Some(Subdivision::from_logits(&vox.coords, &logits));
        }
        let plan = match (plan, &predicted) {
            (Some(p), _) => p,
            (None, Some(p)) => p,
            _ => bail!("the texture decoder needs the shape decoder's subdivisions"),
        };
        let m = plan.flat.len();
        if m == 0 {
            bail!("the decoder subdivided nothing at stage {i}");
        }
        let idx = DevBuf::alloc_async(m * 4, st)?;
        idx.upload(0, &plan.flat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), st)?;
        let t = DevBuf::alloc_async(n * c * 2, st)?;
        norm(PRO_LN_AFFINE_SILU, Some((&self.w, &format!("{key}.norm1"))), x.ptr(), n, c, EPS, t.ptr(), st)?;
        let up = DevBuf::alloc_async(n * co * 8 * 2, st)?;
        conv(&self.w, &format!("{key}.conv1"), vox, t.ptr(), c, co * 8, up.ptr(), st)?;
        drop(t);
        let h = DevBuf::alloc_async(m * co * 2, st)?;
        k!(gpu_gather_rows(up.ptr(), (co * 2) as i64, idx.ptr(), m as i32, h.ptr(), s(st)))?;
        drop(up);
        let skip = DevBuf::alloc_async(m * (c / 8) * 2, st)?;
        k!(gpu_gather_rows(x.ptr(), (c / 8 * 2) as i64, idx.ptr(), m as i32, skip.ptr(), s(st)))?;
        let child = Voxels::new(plan.coords.clone(), vox.res * 2, st)?;
        let u = DevBuf::alloc_async(m * co * 2, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE_SILU, h.ptr(), co as i64, self.ones.ptr(), self.zeros.ptr(), EPS, null(), null(),
            u.ptr(), co as i64, m as i32, co as i32, s(st)))?;
        conv(&self.w, &format!("{key}.conv2"), &child, u.ptr(), co, co, h.ptr(), st)?;
        k!(gpu_skip_add_f16(h.ptr(), skip.ptr(), m as i64, (c / 8) as i32, (co / (c / 8)) as i32, s(st)))?;
        Ok((h, child, predicted))
    }

    fn from_latent(&self, feats: &[f32], n: usize, st: &Stream) -> Result<DevBuf> {
        let x = DevBuf::alloc_async(n * 32 * 2, st)?;
        x.upload(0, &f16_bytes(feats), st)?;
        let h = DevBuf::alloc_async(n * CHANNELS[0] * 2, st)?;
        linear(&self.w, "from_latent", x.ptr(), n, 32, CHANNELS[0], h.ptr(), st)?;
        Ok(h)
    }

    /// Runs the decoder on denormalized latent feats (f32 [n, 32]) at `vox`. `upsample_only` stops before the
    /// last stage and returns its voxels (`SparseUnetVaeDecoder.upsample`).
    fn run(&self, feats: &[f32], vox: Voxels, guide: Option<&[Subdivision]>, upsample_only: bool, st: &Stream,
        progress: &mut dyn FnMut(usize, usize)) -> Result<(DevBuf, Voxels, Vec<Subdivision>)> {
        let mut h = self.from_latent(feats, vox.len(), st)?;
        let mut vox = vox;
        let mut subs = Vec::new();
        let total: usize = BLOCKS.iter().sum::<usize>() + 4;
        let mut done = 0;
        for i in 0..5 {
            if upsample_only && i == 4 {
                break;
            }
            for j in 0..BLOCKS[i] {
                self.convnext(&format!("{i}.{j}"), &vox, &h, CHANNELS[i], st)?;
                done += 1;
                progress(done, total);
            }
            if i < 4 {
                let (nh, nv, sub) = self.c2s(i, &vox, &h, guide.map(|g| &g[i]), st)?;
                h = nh;
                vox = nv;
                if let Some(sub) = sub {
                    subs.push(sub);
                }
                done += 1;
                progress(done, total);
            }
        }
        Ok((h, vox, subs))
    }

    /// Final norm and output layer: f16 [n, 8] on the host, `out_ch` of them real.
    fn output(&self, h: &DevBuf, n: usize, st: &Stream) -> Result<Vec<f16>> {
        let c = CHANNELS[4];
        let t = DevBuf::alloc_async(n * c * 2, st)?;
        norm(PRO_LAYERNORM, None, h.ptr(), n, c, OUT_EPS, t.ptr(), st)?;
        let o = DevBuf::alloc_async(n * 8 * 2, st)?;
        linear(&self.w, "out", t.ptr(), n, c, 8, o.ptr(), st)?;
        let mut host = vec![0u8; n * 16];
        o.download(0, &mut host, st)?;
        st.sync()?;
        let _ = self.out_ch;
        Ok(host.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]])).collect())
    }
}

/// The shape VAE file: the dense structure decoder and the shape decoder.
pub struct ShapeVae {
    structure: Weights,
    shape: SparseDecoder,
}

/// The texture VAE file.
pub struct TextureVae {
    tex: SparseDecoder,
}

pub fn is_shape_vae(has: impl Fn(&str) -> bool) -> bool {
    has("shape_dec.blocks.1.16.to_subdiv.weight") && has("struct_dec.input_layer.weight")
}

pub fn is_texture_vae(has: impl Fn(&str) -> bool) -> bool {
    has("txt_dec.blocks.3.4.conv2.weight")
}

impl ShapeVae {
    pub fn load(st: &SafeTensors, s0: &Stream) -> Result<Self> {
        if !is_shape_vae(|k| st.has(k)) {
            bail!("not a Trellis 2 shape VAE");
        }
        let mut l = Layout::new();
        let p = |x: &str| format!("struct_dec.{x}");
        l.bytes("in.w", dense_kernel(st, &p("input_layer.weight"), None)?);
        l.bytes("in.b", f16_tensor(st, &p("input_layer.bias"), None)?);
        let res_block = |l: &mut Layout, key: &str, name: &str| -> Result<()> {
            ln(st, l, &format!("{key}.norm1"), &format!("{name}.norm1"))?;
            ln(st, l, &format!("{key}.norm2"), &format!("{name}.norm2"))?;
            for c in ["conv1", "conv2"] {
                l.bytes(&format!("{key}.{c}.w"), dense_kernel(st, &format!("{name}.{c}.weight"), None)?);
                l.bytes(&format!("{key}.{c}.b"), f16_tensor(st, &format!("{name}.{c}.bias"), None)?);
            }
            Ok(())
        };
        for i in 0..2 {
            res_block(&mut l, &format!("mid{i}"), &p(&format!("middle_block.{i}")))?;
        }
        for i in 0..8 {
            if i == 2 || i == 5 {
                l.bytes(&format!("b{i}.conv.w"), dense_kernel(st, &p(&format!("blocks.{i}.conv.weight")), None)?);
                l.bytes(&format!("b{i}.conv.b"), f16_tensor(st, &p(&format!("blocks.{i}.conv.bias")), None)?);
            } else {
                res_block(&mut l, &format!("b{i}"), &p(&format!("blocks.{i}")))?;
            }
        }
        ln(st, &mut l, "out.norm", &p("out_layer.0"))?;
        l.bytes("out.conv.w", dense_kernel(st, &p("out_layer.2.weight"), Some(8))?);
        l.bytes("out.conv.b", f16_tensor(st, &p("out_layer.2.bias"), Some(8))?);
        let structure = Weights::upload(st, l, s0)?;
        let shape = SparseDecoder::load(st, "shape_dec", 7, true, s0)?;
        Ok(ShapeVae { structure, shape })
    }

    /// `VaeDecodeStructureTrellis2` at resolution 32: the occupied voxels of the 8-channel structure latent
    /// (f32 [8, 16, 16, 16]), sorted like `argwhere`.
    pub fn decode_structure(&self, latent: &[f32], st: &Stream) -> Result<Vec<[i32; 4]>> {
        let w = &self.structure;
        let mut r = 16;
        let mut vox = Voxels::dense(r, st)?;
        // the latent is channel major; the decoder's rows are voxels
        let mut rows = vec![0f32; r * r * r * 8];
        for c in 0..8 {
            for v in 0..r * r * r {
                rows[v * 8 + c] = latent[c * r * r * r + v];
            }
        }
        let x = DevBuf::alloc_async(rows.len() * 2, st)?;
        x.upload(0, &f16_bytes(&rows), st)?;
        let mut c = 512;
        let mut h = DevBuf::alloc_async(r * r * r * c * 2, st)?;
        conv(w, "in", &vox, x.ptr(), 8, c, h.ptr(), st)?;
        let res = |key: &str, vox: &Voxels, h: &DevBuf, c: usize| -> Result<()> {
            let n = vox.len();
            let t = DevBuf::alloc_async(n * c * 2, st)?;
            let u = DevBuf::alloc_async(n * c * 2, st)?;
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE_SILU, h.ptr(), c as i64, w.p(&format!("{key}.norm1.w")), w.p(&format!("{key}.norm1.b")),
                OUT_EPS, null(), null(), t.ptr(), c as i64, n as i32, c as i32, s(st)))?;
            conv(w, &format!("{key}.conv1"), vox, t.ptr(), c, c, u.ptr(), st)?;
            k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE_SILU, u.ptr(), c as i64, w.p(&format!("{key}.norm2.w")), w.p(&format!("{key}.norm2.b")),
                OUT_EPS, null(), null(), t.ptr(), c as i64, n as i32, c as i32, s(st)))?;
            conv(w, &format!("{key}.conv2"), vox, t.ptr(), c, c, u.ptr(), st)?;
            k!(gpu_add_f16(h.ptr(), u.ptr(), (n * c) as i64, s(st)))?;
            Ok(())
        };
        res("mid0", &vox, &h, c)?;
        res("mid1", &vox, &h, c)?;
        for i in 0..8 {
            if i == 2 || i == 5 {
                let co = c / 4;
                let t = DevBuf::alloc_async(r * r * r * co * 8 * 2, st)?;
                conv(w, &format!("b{i}.conv"), &vox, h.ptr(), c, co * 8, t.ptr(), st)?;
                h = DevBuf::alloc_async(8 * r * r * r * co * 2, st)?;
                k!(gpu_pixel_shuffle3d_f16(t.ptr(), h.ptr(), r as i32, co as i32, s(st)))?;
                r *= 2;
                c = co;
                vox = Voxels::dense(r, st)?;
            } else {
                res(&format!("b{i}"), &vox, &h, c)?;
            }
        }
        let n = vox.len();
        let t = DevBuf::alloc_async(n * c * 2, st)?;
        k!(gpu_prologue_rows_b(DT_F16, PRO_LN_AFFINE_SILU, h.ptr(), c as i64, w.p("out.norm.w"), w.p("out.norm.b"), OUT_EPS, null(), null(),
            t.ptr(), c as i64, n as i32, c as i32, s(st)))?;
        let o = DevBuf::alloc_async(n * 8 * 2, st)?;
        conv(w, "out.conv", &vox, t.ptr(), c, 8, o.ptr(), st)?;
        let mut host = vec![0u8; n * 16];
        o.download(0, &mut host, st)?;
        st.sync()?;
        // occupancy at 64^3, max-pooled to 32^3
        let occ = |i: usize| f16::from_le_bytes([host[i * 16], host[i * 16 + 1]]).to_f32() > 0.0;
        let half_r = r / 2;
        let mut out = Vec::new();
        for x in 0..half_r {
            for y in 0..half_r {
                for z in 0..half_r {
                    let mut any = false;
                    for d in 0..8 {
                        let (xx, yy, zz) = (2 * x + (d >> 2), 2 * y + ((d >> 1) & 1), 2 * z + (d & 1));
                        any |= occ((xx * r + yy) * r + zz);
                    }
                    if any {
                        out.push([0, x as i32, y as i32, z as i32]);
                    }
                }
            }
        }
        Ok(out)
    }

    /// `ShapeVae.upsample_shape(..., 4)`: the voxels the shape decoder reaches at 16x the latent's resolution.
    pub fn upsample(&self, feats: &[f32], coords: Vec<[i32; 4]>, res: usize, st: &Stream) -> Result<Vec<[i32; 4]>> {
        let vox = Voxels::new(coords, res, st)?;
        let (_, vox, _) = self.shape.run(feats, vox, None, true, st, &mut |_, _| {})?;
        Ok(vox.coords)
    }

    /// `decode_shape_slat`: the final voxels with their outputs, and the subdivisions for the texture decode.
    pub fn decode_shape(&self, feats: &[f32], coords: Vec<[i32; 4]>, res: usize, st: &Stream, progress: &mut dyn FnMut(usize, usize))
        -> Result<ShapeOut> {
        let vox = Voxels::new(coords, res, st)?;
        let (h, vox, subdivisions) = self.shape.run(feats, vox, None, false, st, progress)?;
        let feats = self.shape.output(&h, vox.len(), st)?;
        Ok(ShapeOut { voxels: vox.coords, feats, subdivisions })
    }
}

impl TextureVae {
    pub fn load(st: &SafeTensors, s0: &Stream) -> Result<Self> {
        if !is_texture_vae(|k| st.has(k)) {
            bail!("not a Trellis 2 texture VAE");
        }
        Ok(TextureVae { tex: SparseDecoder::load(st, "txt_dec", 6, false, s0)? })
    }

    /// `decode_tex_slat`: per final voxel (in the shape decode's order) base color, metallic, roughness and alpha
    /// in [0, 1], as f32 [n, 6].
    pub fn decode(&self, feats: &[f32], coords: Vec<[i32; 4]>, res: usize, subs: &[Subdivision], st: &Stream,
        progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<f32>> {
        let vox = Voxels::new(coords, res, st)?;
        let (h, vox, _) = self.tex.run(feats, vox, Some(subs), false, st, progress)?;
        let out = self.tex.output(&h, vox.len(), st)?;
        let mut colors = Vec::with_capacity(vox.len() * 6);
        for row in out.chunks_exact(8) {
            for v in &row[..6] {
                let half = f16::from_f32(v.to_f32() * 0.5).to_f32();
                colors.push(f16::from_f32(half + 0.5).to_f32());
            }
        }
        Ok(colors)
    }
}

/// `flexible_dual_grid_to_mesh` with the split weights, on the host: vertices (f32 [n][3], one per voxel, in
/// [-0.5, 0.5]^3) and triangles.
pub fn dual_grid_mesh(voxels: &[[i32; 4]], feats: &[f16], grid: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let n = voxels.len();
    let key = |x: i64, y: i64, z: i64| ((x * grid as i64) + y) * grid as i64 + z;
    let mut index: std::collections::HashMap<i64, u32> = std::collections::HashMap::with_capacity(n);
    for (i, c) in voxels.iter().enumerate() {
        index.insert(key(c[1] as i64, c[2] as i64, c[3] as i64), i as u32);
    }
    let vs = 1.0f32 / grid as f32;
    let h = |v: f32| f16::from_f32(v).to_f32();
    let mut verts = Vec::with_capacity(n);
    let mut split = Vec::with_capacity(n);
    for (i, c) in voxels.iter().enumerate() {
        let f = &feats[i * 8..i * 8 + 8];
        let mut v = [0f32; 3];
        for a in 0..3 {
            let sig = h(1.0 / (1.0 + (-f[a].to_f32()).exp()));
            let dv = h(h(2.0 * sig) - 0.5);
            v[a] = (c[a + 1] as f32 + dv) * vs + -0.5;
        }
        verts.push(v);
        let q = f[6].to_f32();
        split.push(h(if q > 20.0 { q } else { q.exp().ln_1p() }));
    }
    const OFFSETS: [[[i64; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let mut tris = Vec::new();
    for (i, c) in voxels.iter().enumerate() {
        for axis in 0..3 {
            if feats[i * 8 + 3 + axis].to_f32() <= 0.0 {
                continue;
            }
            let mut q = [0u32; 4];
            let mut ok = true;
            for (j, o) in OFFSETS[axis].iter().enumerate() {
                match index.get(&key(c[1] as i64 + o[0], c[2] as i64 + o[1], c[3] as i64 + o[2])) {
                    Some(&v) => q[j] = v,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let w02 = h(split[q[0] as usize] * split[q[2] as usize]);
            let w13 = h(split[q[1] as usize] * split[q[3] as usize]);
            if w02 > w13 {
                tris.push([q[0], q[1], q[2]]);
                tris.push([q[0], q[2], q[3]]);
            } else {
                tris.push([q[0], q[1], q[3]]);
                tris.push([q[3], q[1], q[2]]);
            }
        }
    }
    (verts, tris)
}
