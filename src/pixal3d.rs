//! Pixal3D image to 3D, following Comfy-Org's example workflow node for node: `ImageCropToMask` (pad 1.1),
//! `Pixal3DConditioning`, the sparse-structure, 512 and upsampled shape, and texture samplers with their CFG
//! patches, the shape and texture decodes, and `PaintMesh`. The output is a vertex-coloured mesh.
//!
//! Background removal and MoGe's field-of-view estimate are not part of it: the input's alpha is the mask, and
//! the field of view is a parameter.

use crate::cuda::{DevBuf, Stream};
use crate::dino3::{Dino3, Tokens, PREFIX};
use crate::ffi::{k, s, DT_F16, DT_F32};
use crate::samplers::{self, CfgFunction, CfgTweaks, Eval, ModelSampling, Uncond};
use crate::sampling::torch_randn;
use crate::st::{f32_bytes, SafeTensors};
use crate::trellis::{grid_coords, rope_table, Flow, Stage, StageCond, CTX, CTX_LEN};
use crate::trellis_vae::{dual_grid_mesh, ShapeVae, TextureVae, SHAPE_MEAN, SHAPE_STD, TEX_MEAN, TEX_STD};
use anyhow::{bail, Result};
use half::bf16;
use std::path::PathBuf;
use std::time::Instant;

const LATENT_CH: usize = 32;
const SS_RES: usize = 16;
const SHAPE512_RES: usize = 32;
const COMPOSITE: usize = 1024;
const PAD: f64 = 1.1;

/// The four files of a Pixal3D install.
#[derive(Clone, Debug)]
pub struct Paths {
    pub dit: PathBuf,
    pub dino: PathBuf,
    pub shape_vae: PathBuf,
    pub texture_vae: PathBuf,
}

/// One sampler of the workflow: `KSampler` behind `CFGOverride` (cfg 1 below `override_sigma`) and `RescaleCFG`.
#[derive(Clone, Debug)]
pub struct StageSampling {
    pub seed: u64,
    pub steps: usize,
    pub cfg: f64,
    pub sampler: String,
    pub scheduler: String,
    pub shift: f64,
    /// `CFGOverride(cfg 1, start_percent)`: the percent where guidance turns off (on the unshifted model, shift 3)
    pub cfg_off_percent: Option<f64>,
    pub rescale: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct Params {
    /// horizontal field of view of the photo, degrees
    pub fov: f64,
    /// voxel resolution of the upsampled shape: 1024 to 2048 in steps of 128
    pub resolution: usize,
    pub structure: StageSampling,
    pub shape: StageSampling,
    pub refine: StageSampling,
    pub texture: StageSampling,
    /// skip the texture stage and give a grey mesh
    pub texture_enabled: bool,
    /// simplify the mesh to at most this many triangles (ComfyUI's example decimates to 700k); 0 keeps them all
    pub max_faces: usize,
}

impl Params {
    /// The example workflow's settings (`pixal3d_test_jester.json`), with every seed derived from one.
    pub fn example(seed: u64) -> Self {
        let st = |seed: u64, steps: usize, cfg: f64, scheduler: &str, shift: f64, off: Option<f64>, rescale: Option<f32>| StageSampling {
            seed, steps, cfg, sampler: "euler".into(), scheduler: scheduler.into(), shift, cfg_off_percent: off, rescale,
        };
        Params {
            fov: 49.13,
            resolution: 1536,
            structure: st(seed, 12, 7.5, "normal", 5.0, Some(0.667), Some(0.7)),
            shape: st(seed, 20, 7.5, "normal", 3.0, Some(0.769), Some(0.5)),
            refine: st(seed, 12, 7.5, "simple", 3.0, Some(0.769), Some(0.5)),
            texture: st(seed.wrapping_add(1), 12, 1.0, "normal", 3.0, None, None),
            texture_enabled: true,
            max_faces: 700_000,
        }
    }
}

/// A triangle mesh in glTF's frame (Y up), with linear vertex colours.
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub colors: Option<Vec<[f32; 3]>>,
}

/// Progress: stage name, step, steps.
pub type Progress<'a> = dyn FnMut(&str, usize, usize) -> Result<()> + 'a;

// ---------------------------------------------------------------------------------------------
// Pillow's LANCZOS resize (8 bits per channel), as ComfyUI's `common_upscale(..., "lanczos")` calls it.
// ---------------------------------------------------------------------------------------------

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn lanczos_filter(x: f64) -> f64 {
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { (x * std::f64::consts::PI).sin() / (x * std::f64::consts::PI) };
    if (-3.0..3.0).contains(&x) { sinc(x) * sinc(x / 3.0) } else { 0.0 }
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel, the first input pixel and fixed-point weights.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 3.0 * filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
            let mut k: Vec<f64> = (0..xmax).map(|x| lanczos_filter((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            let fixed = k.iter().map(|&v| if v < 0.0 { (-0.5 + v * (1u64 << PRECISION_BITS) as f64) as i32 } else { (0.5 + v * (1u64 << PRECISION_BITS) as f64) as i32 }).collect();
            (xmin, fixed)
        })
        .collect()
}

fn clip8(v: i64) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resize an 8-bit image with `ch` interleaved channels: horizontal pass, then vertical, like Pillow.
pub fn pil_lanczos(src: &[u8], w: usize, h: usize, ch: usize, nw: usize, nh: usize) -> Vec<u8> {
    if (w, h) == (nw, nh) {
        return src.to_vec();
    }
    let mut cur = src.to_vec();
    let mut cw = w;
    if nw != w {
        let cs = coeffs(w, nw);
        let mut out = vec![0u8; nw * h * ch];
        for y in 0..h {
            for (xx, (xmin, k)) in cs.iter().enumerate() {
                for c in 0..ch {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (i, kv) in k.iter().enumerate() {
                        ss += cur[(y * w + xmin + i) * ch + c] as i64 * *kv as i64;
                    }
                    out[(y * nw + xx) * ch + c] = clip8(ss);
                }
            }
        }
        cur = out;
        cw = nw;
    }
    if nh != h {
        let cs = coeffs(h, nh);
        let mut out = vec![0u8; cw * nh * ch];
        for (yy, (ymin, k)) in cs.iter().enumerate() {
            for x in 0..cw {
                for c in 0..ch {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (i, kv) in k.iter().enumerate() {
                        ss += cur[((ymin + i) * cw + x) * ch + c] as i64 * *kv as i64;
                    }
                    out[(yy * cw + x) * ch + c] = clip8(ss);
                }
            }
        }
        cur = out;
    }
    cur
}

/// `comfy.utils.lanczos` on f32 values: truncated to u8 the way numpy's `astype` does, resized, then / 255.
fn lanczos_f32(src: &[f32], w: usize, h: usize, ch: usize, nw: usize, nh: usize) -> Vec<f32> {
    let u8s: Vec<u8> = src.iter().map(|v| (255.0f32 * v).clamp(0.0, 255.0) as u8).collect();
    pil_lanczos(&u8s, w, h, ch, nw, nh).into_iter().map(|v| v as f32 / 255.0).collect()
}

/// An RGBA image as ComfyUI's IMAGE (f32 RGB) and, from its alpha, a mask that is 1 on the object.
pub fn split_rgba(rgba: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let rgb = rgba.chunks_exact(4).flat_map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();
    (rgb, rgba.chunks_exact(4).map(|p| p[3] as f32 / 255.0).collect())
}

/// `ImageCropToMask(width=1024, height=1024, pad_factor=1.1, grow_mask=0, background=#000000)`: the centred square
/// composite, f32 [1024, 1024, 3]. `mask` may have either polarity, as there.
pub fn crop_to_mask(rgb: &[f32], mask: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
    if rgb.len() != w * h * 3 || mask.len() != w * h {
        bail!("the mask must have the image's size");
    }
    let mut img = rgb.to_vec();
    let mut mask = mask.to_vec();
    // an inverted mask: border and centre of opposite polarity
    let border: Vec<f32> = (0..w).flat_map(|x| [mask[x], mask[(h - 1) * w + x]]).chain((0..h).flat_map(|y| [mask[y * w], mask[y * w + w - 1]])).collect();
    let center: Vec<f32> = (h / 4..h - h / 4).flat_map(|y| (w / 4..w - w / 4).map(move |x| (y, x))).map(|(y, x)| mask[y * w + x]).collect();
    let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
    if mean(&border) > 0.5 && mean(&center) < 0.5 {
        mask.iter_mut().for_each(|m| *m = 1.0 - *m);
    }
    mask.iter_mut().for_each(|m| if *m < 0.05 { *m = 0.0 });
    let (mut w, mut h) = (w, h);
    if w.max(h) > COMPOSITE {
        let scale = COMPOSITE as f64 / w.max(h) as f64;
        let (nw, nh) = ((w as f64 * scale) as usize, (h as f64 * scale) as usize);
        img = lanczos_f32(&img, w, h, 3, nw, nh);
        mask = lanczos_f32(&mask, w, h, 1, nw, nh);
        (w, h) = (nw, nh);
    }
    let fg: Vec<(usize, usize)> = (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).filter(|&(y, x)| (mask[y * w + x].clamp(0.0, 1.0) * 255.0) as u8 > 204).collect();
    if fg.is_empty() {
        bail!("the mask is empty: Pixal3D needs the object cut out, on a transparent background or with a mask");
    }
    let (y_min, y_max) = (fg.iter().map(|p| p.0).min().unwrap(), fg.iter().map(|p| p.0).max().unwrap());
    let (x_min, x_max) = (fg.iter().map(|p| p.1).min().unwrap(), fg.iter().map(|p| p.1).max().unwrap());
    let (cy, cx) = ((y_min + y_max) as f64 / 2.0, (x_min + x_max) as f64 / 2.0);
    let (bw, bh) = ((x_max - x_min + 1) as f64, (y_max - y_min + 1) as f64);
    let (crop_w, crop_h) = if bw / bh.max(1.0) >= 1.0 { ((bw * PAD) as i64, (bw * PAD) as i64) } else { ((bh * PAD) as i64, (bh * PAD) as i64) };
    let (half_w, half_h) = ((crop_w as f64 / 2.0).ceil() as i64, (crop_h as f64 / 2.0).ceil() as i64);
    let (x1, y1) = ((cx - half_w as f64) as i64, (cy - half_h as f64) as i64);
    let (cw, chh) = ((2 * half_w) as usize, (2 * half_h) as usize);
    let mut comp = vec![0f32; cw * chh * 3];
    for y in 0..chh {
        for x in 0..cw {
            let (sy, sx) = (y1 + y as i64, x1 + x as i64);
            if sy < 0 || sx < 0 || sy >= h as i64 || sx >= w as i64 {
                continue;
            }
            let (sy, sx) = (sy as usize, sx as usize);
            let m = mask[sy * w + sx];
            for c in 0..3 {
                comp[(y * cw + x) * 3 + c] = (img[(sy * w + sx) * 3 + c] * m + 0.0 * (1.0 - m)).clamp(0.0, 1.0);
            }
        }
    }
    Ok(lanczos_f32(&comp, cw, chh, 3, COMPOSITE, COMPOSITE))
}

// ---------------------------------------------------------------------------------------------
// Pixal3D's back-projection of image features onto voxels
// ---------------------------------------------------------------------------------------------

/// Camera of `Pixal3DConditioning`: the front view at the distance that frames the unit cube for `fov`.
struct Camera {
    fov: f32,
    distance: f32,
}

impl Camera {
    fn new(fov_deg: f64) -> Self {
        let rad = fov_deg.to_radians();
        Camera { fov: rad as f32, distance: (0.5 / (rad / 2.0).tan()) as f32 }
    }

    /// `_project_points_to_image` then the NDC of `_back_project_to_tokens`, for world points.
    fn ndc(&self, world: &[[f32; 3]], res: usize, as_f16: bool) -> Vec<f32> {
        let focal = 16.0f32 / (self.fov / 2.0).tan();
        let fpx = focal * res as f32 / 32.0;
        let half = res as f32 / 2.0;
        let mut out = Vec::with_capacity(world.len() * 2);
        for p in world {
            // world to camera is the inverse of the front-view transform
            let (xc, yc, zc) = (p[0], p[2], -p[1] + -self.distance);
            let denom = -zc + 1e-8;
            let xp = fpx * xc / denom + half;
            let yp = -fpx * yc / denom + half;
            for v in [xp, yp] {
                let n = (v + 0.5) / res as f32 * 2.0 - 1.0;
                out.push(if as_f16 { half::f16::from_f32(n).to_f32() } else { n });
            }
        }
        out
    }
}

/// `_coords_to_proj_world`: voxel coordinates of a `res` grid to Pixal3D's world frame (mesh scale 1).
fn coords_world(coords: &[[i32; 4]], res: usize) -> Vec<[f32; 3]> {
    coords
        .iter()
        .map(|c| {
            let n = |v: i32| v as f32 / (res - 1) as f32 * 2.0 - 1.0;
            let (x, y, z) = (n(c[1]), n(c[2]), n(c[3]));
            [x / 1.0 / 2.0, -z / 1.0 / 2.0, y / 1.0 / 2.0]
        })
        .collect()
}

/// `_dense_grid_proj_world` for the structure stage's 16^3 grid.
fn dense_world(res: usize) -> Vec<[f32; 3]> {
    let step = 2.0f32 / (res - 1) as f32;
    let lin: Vec<f32> = (0..res).map(|i| if i < res / 2 { -1.0 + step * i as f32 } else { 1.0 - step * (res - i - 1) as f32 }).collect();
    grid_coords(res).iter().map(|c| {
        let (x, y, z) = (lin[c[0] as usize], lin[c[1] as usize], lin[c[2] as usize]);
        [x / 1.0 / 2.0, -z / 1.0 / 2.0, y / 1.0 / 2.0]
    }).collect()
}

/// One stage's feature maps: the DINO patch grid (f32 NHWC), and the NAF map (f16 NHWC) after the structure stage.
struct StageMaps<'a> {
    lr: &'a Tokens,
    hr: Option<(&'a DevBuf, usize)>,
    image_res: usize,
}

/// `compute_stage_proj_feats`, as the bf16 rows the model gets: [n, 1024 (+ 1024)].
fn project(cam: &Camera, world: &[[f32; 3]], maps: &StageMaps, st: &Stream) -> Result<DevBuf> {
    let n = world.len();
    let cols = if maps.hr.is_some() { 2048 } else { 1024 };
    let out = DevBuf::alloc_async(n * cols * 2, st)?;
    let uv = cam.ndc(world, maps.image_res, false);
    let uvd = DevBuf::alloc_async(uv.len() * 4, st)?;
    uvd.upload(0, &f32_bytes(&uv), st)?;
    let side = maps.lr.side as i32;
    k!(gpu_grid_sample_bf16(DT_F32, maps.lr.patches(), side, side, 1024, uvd.ptr(), n as i32, out.ptr(), cols as i64, 0, s(st)))?;
    if let Some((hr, size)) = maps.hr {
        let uv = cam.ndc(world, maps.image_res, true);
        uvd.upload(0, &f32_bytes(&uv), st)?;
        k!(gpu_grid_sample_bf16(DT_F16, hr.ptr(), size as i32, size as i32, 1024, uvd.ptr(), n as i32, out.ptr(), cols as i64, 1024, s(st)))?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

/// One `KSampler` on a sparse (or dense) token set. `x` is ComfyUI's [32, n] channel-major latent; `concat` the
/// texture stage's shape latent, token major [n, 32].
#[allow(clippy::too_many_arguments)]
fn sample(flow: &mut Flow, cond: &StageCond, rope: &DevBuf, n: usize, concat: Option<&[f32]>, sp: &StageSampling, st: &Stream,
    progress: &mut Progress, label: &str) -> Result<Vec<f32>> {
    let ms = ModelSampling::Flow { shift: sp.shift };
    let sigmas = samplers::sigmas(ms, &sp.sampler, &sp.scheduler, sp.steps, (0, 0), None)?;
    // CFGOverride reads the sigma range off the model before ModelSamplingSD3 replaces its shift
    let off_below = sp.cfg_off_percent.map(|p| ModelSampling::Flow { shift: 3.0 }.percent_to_sigma(p) as f64);
    let tweaks = CfgTweaks { function: sp.rescale.map(|multiplier| CfgFunction::Rescale { multiplier }), ..Default::default() };
    let (in_ch, out_ch) = (flow.stage.in_ch(), flow.stage.out_ch());
    let noise = torch_randn(sp.seed, LATENT_CH * n);
    let tokens = DevBuf::alloc_async(n * in_ch * 2, st)?;
    let vel = DevBuf::alloc_async(n * out_ch * 2, st)?;
    let mut host = vec![0u8; n * out_ch * 2];
    let mut run = |flow: &mut Flow, x: &[f32], sigma: f32, uncond: bool| -> Result<Vec<f32>> {
        let mut t = vec![0u8; n * in_ch * 2];
        for i in 0..n {
            for c in 0..in_ch {
                let v = if c < LATENT_CH.min(in_ch) { x[c * n + i] } else { concat.map_or(0.0, |s| s[i * LATENT_CH + c - LATENT_CH]) };
                t[(i * in_ch + c) * 2..][..2].copy_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }
        tokens.upload(0, &t, st)?;
        flow.predict(&tokens, rope, sigma, cond, uncond, &vel, st, None)?;
        vel.download(0, &mut host, st)?;
        st.sync()?;
        let mut den = x.to_vec();
        for i in 0..n {
            for c in 0..out_ch {
                let v = bf16::from_le_bytes([host[(i * out_ch + c) * 2], host[(i * out_ch + c) * 2 + 1]]).to_f32();
                den[c * n + i] = x[c * n + i] - v * sigma;
            }
        }
        Ok(den)
    };
    let mut evals = 0;
    let t0 = Instant::now();
    let mut eval = |x: &[f32], sigma: f32, mode: Uncond| -> Result<Eval> {
        let cfg = if off_below.is_some_and(|hi| (sigma as f64) <= hi) { 1.0 } else { sp.cfg };
        let cond_den = run(flow, x, sigma, false)?;
        let guided = (cfg - 1.0).abs() > 1e-9;
        let uncond = if guided || mode == Uncond::Force { Some(run(flow, x, sigma, true)?) } else { None };
        let (denoised, u) = tweaks.guide(x, &cond_den, uncond.clone(), cfg as f32, sigma);
        evals += 1;
        Ok(Eval { denoised, uncond: uncond.map(|_| u) })
    };
    let total = sigmas.len() - 1;
    let mut step = |i: usize, t: usize| progress(label, i, t);
    let batch = samplers::Batch { index: 0, size: 1, threads: unsafe { crate::ffi::gpu_randn_threads() }.max(1) as usize };
    let out = samplers::sample(&sp.sampler, ms, &sigmas, &noise, sp.seed, batch, &mut eval, &mut step)?;
    eprintln!("[pixal3d] {label}: {total} steps, {evals} evaluations over {n} tokens in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Channel-major [32, n] latent to token-major denormalized feats [n, 32] (`Trellis2SLAT.process_out`).
fn denorm(latent: &[f32], n: usize, mean: &[f32; 32], std: &[f32; 32]) -> Vec<f32> {
    let mut out = vec![0f32; n * LATENT_CH];
    for i in 0..n {
        for c in 0..LATENT_CH {
            out[i * LATENT_CH + c] = latent[c * n + i] * std[c] + mean[c];
        }
    }
    out
}

fn tokens_major(latent: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * LATENT_CH];
    for i in 0..n {
        for c in 0..LATENT_CH {
            out[i * LATENT_CH + c] = latent[c * n + i];
        }
    }
    out
}

fn upload_rope(coords: &[[i32; 3]], st: &Stream) -> Result<DevBuf> {
    let t = rope_table(coords);
    let d = DevBuf::alloc_async(t.len() * 4, st)?;
    d.upload(0, &f32_bytes(&t), st)?;
    Ok(d)
}

fn xyz(coords: &[[i32; 4]]) -> Vec<[i32; 3]> {
    coords.iter().map(|c| [c[1], c[2], c[3]]).collect()
}

/// `Trellis2UpsampleStage._quantize_unique` in Pixal3D mode, sorted like `unique(dim=0)`.
fn quantize_unique(hr: &[[i32; 4]], grid_res: usize) -> Vec<[i32; 4]> {
    let f = (grid_res - 1) as f64 / 512.0;
    let f = f as f32;
    let mut q: Vec<[i32; 4]> = hr
        .iter()
        .map(|c| {
            let r = |v: i32| ((v as f32 + 0.5) * f).round_ties_even() as i32;
            [c[0], r(c[1]), r(c[2]), r(c[3])]
        })
        .collect();
    q.sort_unstable();
    q.dedup();
    q
}

/// Runs the whole pipeline on an image (f32 RGB [h, w, 3]) and its object mask (either polarity).
pub fn run(paths: &Paths, rgb: &[f32], mask: &[f32], w: usize, h: usize, p: &Params, threads: usize, progress: &mut Progress) -> Result<Mesh> {
    if !(1024..=2048).contains(&p.resolution) || p.resolution % 128 != 0 {
        bail!("Pixal3D's resolution must be 1024 to 2048 in steps of 128");
    }
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let st = Stream::new()?;
    let t_all = Instant::now();
    let composite = crop_to_mask(rgb, mask, w, h)?;
    let cam = Camera::new(p.fov);

    // Pixal3DConditioning: DINOv3 at 512 and 1024, then the NAF maps each stage needs
    progress("encode", 0, 2)?;
    let dino_st = SafeTensors::open(&paths.dino)?;
    let mut dino = Dino3::load(&dino_st, threads, &st)?;
    let img512 = lanczos_f32(&composite, COMPOSITE, COMPOSITE, 3, 512, 512);
    let img1024 = lanczos_f32(&composite, COMPOSITE, COMPOSITE, 3, 1024, 1024);
    let t512 = dino.encode(&img512, 512, &st)?;
    progress("encode", 1, 2)?;
    let t1024 = dino.encode(&img1024, 1024, &st)?;
    progress("encode", 2, 2)?;
    let global = |t: &Tokens| -> Result<Vec<f32>> {
        let mut b = vec![0u8; PREFIX * CTX * 4];
        t.buf.download(0, &mut b, &st)?;
        st.sync()?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let (g512, g1024) = (global(&t512)?, global(&t1024)?);
    assert_eq!(CTX_LEN, PREFIX);
    dino.free_vit();

    let dit_st = SafeTensors::open(&paths.dit)?;
    if !crate::trellis::is_pixal3d(|k| dit_st.has(k)) {
        bail!("{} is not a Pixal3D diffusion model", paths.dit.display());
    }
    // stage weights take what the stage's activations leave (freed buffers go back to the driver first)
    let budget = |st: &Stream| -> Result<usize> {
        st.sync()?;
        crate::cuda::trim_pool()?;
        Ok(crate::cuda::device_free()?.saturating_sub(3 << 30))
    };

    // sparse structure
    let ss_coords = grid_coords(SS_RES);
    let ss_proj = project(&cam, &dense_world(SS_RES), &StageMaps { lr: &t512, hr: None, image_res: 512 }, &st)?;
    let mut flow = Flow::load(&dit_st, Stage::Structure, budget(&st)?, threads, &st)?;
    let cond = flow.stage_cond(&g512, &ss_proj, ss_coords.len(), &st)?;
    drop(ss_proj);
    let rope = upload_rope(&ss_coords, &st)?;
    let ss = sample(&mut flow, &cond, &rope, ss_coords.len(), None, &p.structure, &st, progress, "structure")?;
    drop((flow, cond, rope));
    let shape_st = SafeTensors::open(&paths.shape_vae)?;
    let shape_vae = ShapeVae::load(&shape_st, &st)?;
    let coords512 = shape_vae.decode_structure(&ss[..8 * SS_RES * SS_RES * SS_RES], &st)?;
    if coords512.is_empty() {
        bail!("the structure stage found no object");
    }

    // shape at 512
    let n512 = coords512.len();
    let hr512 = dino.naf(&img512, 512, t512.patches(), t512.side, 512, &st)?;
    let proj = project(&cam, &coords_world(&coords512, SHAPE512_RES), &StageMaps { lr: &t512, hr: Some((&hr512, 512)), image_res: 512 }, &st)?;
    drop(hr512);
    let mut flow = Flow::load(&dit_st, Stage::Shape512, budget(&st)?, threads, &st)?;
    let cond = flow.stage_cond(&g512, &proj, n512, &st)?;
    drop(proj);
    let rope = upload_rope(&xyz(&coords512), &st)?;
    let shape512 = sample(&mut flow, &cond, &rope, n512, None, &p.shape, &st, progress, "shape")?;
    drop((flow, cond, rope));

    // upsampled shape
    let hr_up = shape_vae.upsample(&denorm(&shape512, n512, &SHAPE_MEAN, &SHAPE_STD), coords512, SHAPE512_RES, &st)?;
    let grid_res = p.resolution / 16;
    let coords = quantize_unique(&hr_up, grid_res);
    let n = coords.len();
    let hr1024 = dino.naf(&img1024, 1024, t1024.patches(), t1024.side, 512, &st)?;
    let proj = project(&cam, &coords_world(&coords, grid_res), &StageMaps { lr: &t1024, hr: Some((&hr1024, 512)), image_res: 1024 }, &st)?;
    drop(hr1024);
    let mut flow = Flow::load(&dit_st, Stage::Shape, budget(&st)?, threads, &st)?;
    let cond = flow.stage_cond(&g1024, &proj, n, &st)?;
    drop(proj);
    let rope = upload_rope(&xyz(&coords), &st)?;
    let shape = sample(&mut flow, &cond, &rope, n, None, &p.refine, &st, progress, "refine")?;
    drop((flow, cond, rope));

    // mesh
    let mut decode_progress = |i: usize, t: usize| {
        let _ = progress("decode shape", i, t);
    };
    let out = shape_vae.decode_shape(&denorm(&shape, n, &SHAPE_MEAN, &SHAPE_STD), coords.clone(), grid_res, &st, &mut decode_progress)?;
    drop(shape_vae);
    let (vertices, faces) = dual_grid_mesh(&out.voxels, &out.feats, p.resolution);
    eprintln!("[pixal3d] mesh: {} voxels, {} triangles", out.voxels.len(), faces.len());

    let colors = if p.texture_enabled {
        let hr_tex = dino.naf(&img1024, 1024, t1024.patches(), t1024.side, 1024, &st)?;
        let proj = project(&cam, &coords_world(&coords, grid_res), &StageMaps { lr: &t1024, hr: Some((&hr_tex, 1024)), image_res: 1024 }, &st)?;
        drop(hr_tex);
        drop((t512, t1024, dino));
        let mut flow = Flow::load(&dit_st, Stage::Texture, budget(&st)?, threads, &st)?;
        let cond = flow.stage_cond(&g1024, &proj, n, &st)?;
        drop(proj);
        let rope = upload_rope(&xyz(&coords), &st)?;
        let slat = tokens_major(&shape, n);
        let tex = sample(&mut flow, &cond, &rope, n, Some(&slat), &p.texture, &st, progress, "texture")?;
        drop((flow, cond, rope));
        let tex_st = SafeTensors::open(&paths.texture_vae)?;
        let tex_vae = TextureVae::load(&tex_st, &st)?;
        let mut decode_progress = |i: usize, t: usize| {
            let _ = progress("decode texture", i, t);
        };
        let pbr = tex_vae.decode(&denorm(&tex, n, &TEX_MEAN, &TEX_STD), coords, grid_res, &out.subdivisions, &st, &mut decode_progress)?;
        Some(paint(&vertices, &out.voxels, &pbr, p.resolution))
    } else {
        None
    };
    let mut mesh = compact(vertices, faces, colors);
    orient(&mut mesh);
    if p.max_faces > 0 && mesh.faces.len() > p.max_faces {
        let full = mesh.faces.len();
        mesh = simplify(&mesh, p.max_faces, p.resolution);
        eprintln!("[pixal3d] simplified {full} triangles to {}", mesh.faces.len());
    }
    eprintln!("[pixal3d] done in {:.1}s: {} vertices, {} triangles", t_all.elapsed().as_secs_f64(), mesh.vertices.len(), mesh.faces.len());
    Ok(mesh)
}

// ---------------------------------------------------------------------------------------------
// PaintMesh and export
// ---------------------------------------------------------------------------------------------

/// Open-addressing map from voxel coordinates to their index.
struct VoxelIndex {
    keys: Vec<u64>,
    vals: Vec<u32>,
    mask: usize,
}

impl VoxelIndex {
    fn new(voxels: &[[i32; 4]]) -> Self {
        let cap = (2 * voxels.len()).next_power_of_two().max(16);
        let mut m = VoxelIndex { keys: vec![u64::MAX; cap], vals: vec![0; cap], mask: cap - 1 };
        for (i, c) in voxels.iter().enumerate() {
            let k = Self::key(c[1], c[2], c[3]);
            let mut slot = Self::hash(k) & m.mask;
            while m.keys[slot] != u64::MAX && m.keys[slot] != k {
                slot = (slot + 1) & m.mask;
            }
            m.keys[slot] = k;
            m.vals[slot] = i as u32;
        }
        m
    }

    fn key(x: i32, y: i32, z: i32) -> u64 {
        ((x as u64 & 0x1fffff) << 42) | ((y as u64 & 0x1fffff) << 21) | (z as u64 & 0x1fffff)
    }

    fn hash(k: u64) -> usize {
        let mut z = k.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (z ^ (z >> 31)) as usize
    }

    fn get(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        if x < 0 || y < 0 || z < 0 {
            return None;
        }
        let k = Self::key(x, y, z);
        let mut slot = Self::hash(k) & self.mask;
        loop {
            match self.keys[slot] {
                u64::MAX => return None,
                kk if kk == k => return Some(self.vals[slot]),
                _ => slot = (slot + 1) & self.mask,
            }
        }
    }
}

/// `paint_mesh_with_voxels`: each vertex takes the base colour of its nearest voxel (voxel `c` sits at
/// c / res - 0.5), converted to linear RGB with gamma 2.2.
fn paint(vertices: &[[f32; 3]], voxels: &[[i32; 4]], pbr: &[f32], res: usize) -> Vec<[f32; 3]> {
    let index = VoxelIndex::new(voxels);
    vertices
        .iter()
        .map(|v| {
            let g = [(v[0] + 0.5) * res as f32, (v[1] + 0.5) * res as f32, (v[2] + 0.5) * res as f32];
            let c0 = [g[0].round() as i32, g[1].round() as i32, g[2].round() as i32];
            let mut best = (f32::MAX, None);
            for r in 0..64i32 {
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                                continue;
                            }
                            let (x, y, z) = (c0[0] + dx, c0[1] + dy, c0[2] + dz);
                            if let Some(i) = index.get(x, y, z) {
                                let d = (x as f32 - g[0]).powi(2) + (y as f32 - g[1]).powi(2) + (z as f32 - g[2]).powi(2);
                                if d < best.0 {
                                    best = (d, Some(i));
                                }
                            }
                        }
                    }
                }
                if best.1.is_some() && best.0 <= (r as f32 + 0.5).powi(2) {
                    break;
                }
            }
            match best.1 {
                Some(i) => {
                    let c = &pbr[i as usize * 6..i as usize * 6 + 3];
                    [c[0].clamp(0.0, 1.0).powf(2.2), c[1].clamp(0.0, 1.0).powf(2.2), c[2].clamp(0.0, 1.0).powf(2.2)]
                }
                None => [0.0; 3],
            }
        })
        .collect()
}

/// Drops the vertices no triangle uses.
fn compact(vertices: Vec<[f32; 3]>, faces: Vec<[u32; 3]>, colors: Option<Vec<[f32; 3]>>) -> Mesh {
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut nv = Vec::new();
    let mut nc = Vec::new();
    let faces = faces
        .into_iter()
        .map(|f| {
            f.map(|i| {
                if remap[i as usize] == u32::MAX {
                    remap[i as usize] = nv.len() as u32;
                    nv.push(vertices[i as usize]);
                    if let Some(c) = &colors {
                        nc.push(c[i as usize]);
                    }
                }
                remap[i as usize]
            })
        })
        .collect();
    Mesh { vertices: nv, faces, colors: colors.map(|_| nc) }
}

/// Vertex clustering on a grid coarser than the voxels', with the finest cell size that keeps at most `max_faces`
/// triangles: each cell's vertices merge at their mean (colours too), and triangles that collapse go.
pub fn simplify(m: &Mesh, max_faces: usize, res: usize) -> Mesh {
    let mut cells = (res as f64 * (max_faces as f64 / m.faces.len() as f64).sqrt()) as usize;
    loop {
        let out = cluster(m, cells.max(8));
        if out.faces.len() <= max_faces || cells <= 8 {
            return out;
        }
        cells = cells * 9 / 10;
    }
}

fn cluster(m: &Mesh, cells: usize) -> Mesh {
    let cell = |v: f32| (((v + 0.5) * cells as f32).floor() as i32).clamp(0, cells as i32 - 1);
    let keys: Vec<[i32; 4]> = m.vertices.iter().map(|v| [0, cell(v[0]), cell(v[1]), cell(v[2])]).collect();
    let mut uniq = keys.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let index = VoxelIndex::new(&uniq);
    let id: Vec<u32> = keys.iter().map(|k| index.get(k[1], k[2], k[3]).unwrap()).collect();
    let n = uniq.len();
    let (mut pos, mut col, mut count) = (vec![[0f64; 3]; n], vec![[0f64; 3]; n], vec![0u32; n]);
    for (i, v) in m.vertices.iter().enumerate() {
        let c = id[i] as usize;
        count[c] += 1;
        for k in 0..3 {
            pos[c][k] += v[k] as f64;
            if let Some(cs) = &m.colors {
                col[c][k] += cs[i][k] as f64;
            }
        }
    }
    let mean = |a: &[f64; 3], n: u32| [(a[0] / n as f64) as f32, (a[1] / n as f64) as f32, (a[2] / n as f64) as f32];
    let mut seen = std::collections::HashSet::new();
    let faces: Vec<[u32; 3]> = m
        .faces
        .iter()
        .map(|f| f.map(|i| id[i as usize]))
        .filter(|f| f[0] != f[1] && f[1] != f[2] && f[0] != f[2])
        .filter(|f| {
            let mut k = *f;
            k.sort_unstable();
            seen.insert(k)
        })
        .collect();
    let vertices = (0..n).map(|c| mean(&pos[c], count[c])).collect();
    let colors = m.colors.as_ref().map(|_| (0..n).map(|c| mean(&col[c], count[c])).collect());
    compact(vertices, faces, colors)
}

/// The dual grid gives each quad a winding that ignores which side is inside, so normals would cancel: make the
/// winding agree across manifold edges, then turn each connected piece outward (positive signed volume). Runs on
/// the full mesh, whose pieces are closed; simplifying afterwards keeps the winding.
pub fn orient(m: &mut Mesh) {
    let nf = m.faces.len();
    // directed edges as (undirected key, face << 1 | a < b)
    let mut edges: Vec<(u64, u32)> = Vec::with_capacity(nf * 3);
    for (f, t) in m.faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.push((((a.min(b) as u64) << 32) | a.max(b) as u64, (f as u32) << 1 | (a < b) as u32));
        }
    }
    edges.sort_unstable_by_key(|e| e.0);
    // up to one neighbour per edge: (face << 1 | winds the same way), or MAX
    let mut adj = vec![[u32::MAX; 3]; nf];
    let mut push = |f: u32, v: u32| {
        if let Some(slot) = adj[f as usize].iter_mut().find(|s| **s == u32::MAX) {
            *slot = v;
        }
    };
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 {
            j += 1;
        }
        if j - i == 2 {
            let (f, g) = (edges[i].1, edges[i + 1].1);
            // consistent faces cross a shared edge in opposite directions
            let same = ((f & 1) != (g & 1)) as u32;
            push(f >> 1, (g >> 1) << 1 | same);
            push(g >> 1, (f >> 1) << 1 | same);
        }
        i = j;
    }
    drop(edges);
    let mut flip = vec![false; nf];
    let mut seen = vec![false; nf];
    let mut stack = Vec::new();
    let mut component = Vec::new();
    let signed = |m: &Mesh, f: u32| {
        let [a, b, c] = m.faces[f as usize].map(|i| m.vertices[i as usize].map(|x| x as f64));
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    for start in 0..nf {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start as u32);
        component.clear();
        while let Some(f) = stack.pop() {
            component.push(f);
            for &v in &adj[f as usize] {
                if v == u32::MAX {
                    continue;
                }
                let (g, same) = ((v >> 1) as usize, v & 1 == 1);
                if !seen[g] {
                    seen[g] = true;
                    flip[g] = flip[f as usize] != !same;
                    stack.push(g as u32);
                }
            }
        }
        let volume: f64 = component.iter().map(|&f| if flip[f as usize] { -signed(m, f) } else { signed(m, f) }).sum();
        if volume < 0.0 {
            component.iter().for_each(|&f| flip[f as usize] = !flip[f as usize]);
        }
    }
    for (t, f) in m.faces.iter_mut().zip(flip) {
        if f {
            t.swap(1, 2);
        }
    }
}

/// A binary glTF 2.0 file: positions, vertex colours and 32-bit indices, with a double-sided material. Like
/// ComfyUI's SaveGLB it has no normals: viewers shade each face towards the camera, whatever its winding.
pub fn glb(m: &Mesh) -> Vec<u8> {
    let nv = m.vertices.len();
    let mut bin = Vec::new();
    let mut views = Vec::new();
    let mut push = |bin: &mut Vec<u8>, data: Vec<u8>, target: u32| {
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        views.push(format!(r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":{}}}"#, bin.len(), data.len(), target));
        bin.extend(data);
        views.len() - 1
    };
    let flat3 = |v: &[[f32; 3]]| -> Vec<u8> { v.iter().flat_map(|p| p.iter().flat_map(|x| x.to_le_bytes())).collect() };
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &m.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let pos = push(&mut bin, flat3(&m.vertices), 34962);
    let col = m.colors.as_ref().map(|c| push(&mut bin, flat3(c), 34962));
    let idx = push(&mut bin, m.faces.iter().flat_map(|f| f.iter().flat_map(|i| i.to_le_bytes())).collect(), 34963);
    let mut accessors = vec![
        format!(r#"{{"bufferView":{pos},"componentType":5126,"count":{nv},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]),
    ];
    let mut attrs = r#""POSITION":0"#.to_string();
    if let Some(c) = col {
        attrs += &format!(r#","COLOR_0":{}"#, accessors.len());
        accessors.push(format!(r#"{{"bufferView":{c},"componentType":5126,"count":{nv},"type":"VEC3"}}"#));
    }
    let ia = accessors.len();
    accessors.push(format!(r#"{{"bufferView":{idx},"componentType":5125,"count":{},"type":"SCALAR"}}"#, m.faces.len() * 3));
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"diffusion pixal3d"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{{attrs}}},"indices":{ia},"material":0}}]}}],"materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":1}},"doubleSided":true}}],"buffers":[{{"byteLength":{}}}],"bufferViews":[{}],"accessors":[{}]}}"#,
        bin.len(),
        views.join(","),
        accessors.join(",")
    );
    let mut jb = json.into_bytes();
    while jb.len() % 4 != 0 {
        jb.push(b' ');
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + jb.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(jb.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend(jb);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend(bin);
    out
}

// ---------------------------------------------------------------------------------------------
// Parity with ComfyUI: each stage from the reference's own inputs (tools/pixal3d_ref.py)
// ---------------------------------------------------------------------------------------------

pub(crate) fn report(name: &str, ours: &[f32], theirs: &[f32]) {
    if ours.len() != theirs.len() {
        eprintln!("{name:24} length {} vs {}", ours.len(), theirs.len());
        return;
    }
    let (mut max, mut sum, mut dot, mut na, mut nb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (a, b) in ours.iter().zip(theirs) {
        let (a, b) = (*a as f64, *b as f64);
        let d = (a - b).abs();
        max = max.max(d);
        sum += d;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let nans = ours.iter().filter(|v| !v.is_finite()).count();
    if nans > 0 {
        let first = ours.iter().position(|v| !v.is_finite()).unwrap();
        eprintln!("{name:24} {nans} non-finite values, the first at {first}");
    }
    let rel = ours.iter().zip(theirs).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>().sqrt() / nb.sqrt().max(1e-30);
    eprintln!("{name:24} max {max:.3e} mean {:.3e} rel {rel:.3e} cos {:.6}", sum / ours.len() as f64, dot / (na.sqrt() * nb.sqrt()).max(1e-30));
}

fn ref_f32(r: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    r.f32_vec(name, false)
}

fn ref_coords(r: &SafeTensors, name: &str) -> Result<Vec<[i32; 4]>> {
    Ok(r.bytes(name)?.chunks_exact(16).map(|c| std::array::from_fn(|i| i32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()))).collect())
}

/// NCHW [c, h, w] to NHWC rows.
fn nhwc(v: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for ch in 0..c {
        for p in 0..hw {
            o[p * c + ch] = v[ch * hw + p];
        }
    }
    o
}

fn download_f32(b: &DevBuf, n: usize, st: &Stream) -> Result<Vec<f32>> {
    let mut h = vec![0u8; n * 4];
    b.download(0, &mut h, st)?;
    st.sync()?;
    Ok(h.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn download_half(b: &DevBuf, n: usize, bf: bool, st: &Stream) -> Result<Vec<f32>> {
    let mut h = vec![0u8; n * 2];
    b.download(0, &mut h, st)?;
    st.sync()?;
    Ok(h.chunks_exact(2).map(|c| if bf { bf16::from_le_bytes([c[0], c[1]]).to_f32() } else { half::f16::from_le_bytes([c[0], c[1]]).to_f32() }).collect())
}

fn upload_bf16(v: &[f32], st: &Stream) -> Result<DevBuf> {
    let d = DevBuf::alloc_async(v.len() * 2, st)?;
    d.upload(0, &v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect::<Vec<_>>(), st)?;
    Ok(d)
}

/// One evaluation of `stage` on the reference's first-call inputs, cond (batch row 0) and, if present, uncond.
#[allow(clippy::too_many_arguments)]
fn check_eval(r: &SafeTensors, dit: &SafeTensors, stage: Stage, mode: &str, coords: &[[i32; 3]], slat: Option<&[f32]>, threads: usize,
    st: &Stream) -> Result<()> {
    let x = ref_f32(r, &format!("{mode}_x"))?;
    let t = ref_f32(r, &format!("{mode}_t"))?;
    // the sparse passes after the first attend to the 1024 encode (`embeds`), the others to the context
    let ctx_name = if matches!(stage, Stage::Shape | Stage::Texture) { "embeds" } else { "context" };
    let ctx_all = ref_f32(r, &format!("{mode}_{ctx_name}"))?;
    // ComfyUI's CFG batch can hold the uncond first: its context is all zeros
    let is_uncond: Vec<bool> = ctx_all.chunks(CTX_LEN * CTX).map(|c| c.iter().all(|v| *v == 0.0)).collect();
    let ctx = ctx_all.chunks(CTX_LEN * CTX).zip(&is_uncond).find(|(_, u)| !**u).map(|(c, _)| c.to_vec()).unwrap_or_else(|| vec![0.0; CTX_LEN * CTX]);
    let proj = ref_f32(r, &format!("{mode}_proj"))?;
    let out = ref_f32(r, &format!("{mode}_out"))?;
    let batch = t.len();
    let n = coords.len();
    let per = x.len() / batch;
    let (in_ch, out_ch) = (stage.in_ch(), stage.out_ch());
    st.sync()?;
    crate::cuda::trim_pool()?;
    let free = crate::cuda::device_free()?.saturating_sub(3 << 30);
    let mut flow = Flow::load(dit, stage, free, threads, st)?;
    let projd = upload_bf16(&proj[..n * stage.proj_ch()], st)?;
    let cond = flow.stage_cond(&ctx[..CTX_LEN * CTX], &projd, n, st)?;
    let rope = upload_rope(coords, st)?;
    let vel = DevBuf::alloc_async(n * out_ch * 2, st)?;
    for b in 0..batch {
        let xb = &x[b * per..(b + 1) * per];
        let mut tok = vec![0f32; n * in_ch];
        for i in 0..n {
            for c in 0..in_ch {
                tok[i * in_ch + c] = if c < LATENT_CH { xb[c * n + i] } else { slat.unwrap()[i * LATENT_CH + c - LATENT_CH] };
            }
        }
        let td = upload_bf16(&tok, st)?;
        // t is sigma * 1000 before ComfyUI rounds it to bf16, which predict does again
        flow.predict(&td, &rope, t[b] / 1000.0, &cond, is_uncond[b], &vel, st, None)?;
        let v = download_half(&vel, n * out_ch, true, st)?;
        let ob = &out[b * per..(b + 1) * per];
        let theirs: Vec<f32> = (0..n).flat_map(|i| (0..out_ch).map(move |c| ob[c * n + i])).collect();
        report(&format!("{mode} {}", if is_uncond[b] { "uncond" } else { "cond" }), &v, &theirs);
    }
    Ok(())
}

/// Compares every stage with `ref_path`, a dump of `tools/pixal3d_ref.py` on the same image.
pub fn check(paths: &Paths, rgba: &[u8], w: usize, h: usize, ref_path: &std::path::Path, threads: usize) -> Result<()> {
    crate::cuda::init(0)?;
    crate::ffi::ok(unsafe { crate::ffi::gpu_blas_init(64 << 20) }, "cublasLt init")?;
    let st = Stream::new()?;
    let r = SafeTensors::open(ref_path)?;
    let (rgb, mask) = split_rgba(rgba);
    let composite = crop_to_mask(&rgb, &mask, w, h)?;
    let rc = ref_f32(&r, "composite")?;
    report("composite", &composite, &rc);
    let differing = composite.iter().zip(&rc).filter(|(a, b)| a != b).count();
    eprintln!("{:24} {differing} of {} values differ", "", rc.len());

    // conditioning, from the reference's composite
    let dino_st = SafeTensors::open(&paths.dino)?;
    let mut dino = Dino3::load(&dino_st, threads, &st)?;
    let img512 = lanczos_f32(&rc, COMPOSITE, COMPOSITE, 3, 512, 512);
    let img1024 = lanczos_f32(&rc, COMPOSITE, COMPOSITE, 3, 1024, 1024);
    let mut probe = |name: &str, b: &DevBuf, nn: usize| {
        let theirs = match name {
            "embed" => "dino_embed",
            "layer0" => "dino_layer0",
            "layer23" => "dino_layer23",
            n if n.starts_with("dbg") => {
                if let Ok(v) = download_f32(b, nn, &st) {
                    let bad = v.iter().position(|x| !x.is_finite());
                    eprintln!("{name:24} first non-finite {bad:?}, max abs {:.3e}", v.iter().filter(|x| x.is_finite()).fold(0f32, |a, x| a.max(x.abs())));
                }
                return;
            }
            _ => return,
        };
        if let (Ok(v), Ok(t)) = (download_f32(b, nn, &st), ref_f32(&r, theirs)) {
            report(&format!("dino {name}"), &v, &t);
        }
    };
    let t512 = dino.encode_probe(&img512, 512, &st, Some(&mut probe))?;
    let t1024 = dino.encode(&img1024, 1024, &st)?;
    let all512 = download_f32(&t512.buf, (PREFIX + 32 * 32) * 1024, &st)?;
    let all1024 = download_f32(&t1024.buf, (PREFIX + 64 * 64) * 1024, &st)?;
    report("global_512", &all512[..PREFIX * 1024], &ref_f32(&r, "global_512")?);
    report("global_1024", &all1024[..PREFIX * 1024], &ref_f32(&r, "global_1024")?);
    report("patches_512", &all512[PREFIX * 1024..], &nhwc(&ref_f32(&r, "ss_lr")?, 1024, 32 * 32));
    report("patches_1024", &all1024[PREFIX * 1024..], &nhwc(&ref_f32(&r, "shape_1024_lr")?, 1024, 64 * 64));
    dino.free_vit();
    for (name, img, size, tok, out) in [("shape_512", &img512, 512, &t512, 512), ("shape_1024", &img1024, 1024, &t1024, 512),
        ("tex_1024", &img1024, 1024, &t1024, 1024)]
    {
        let hr = dino.naf(img, size, tok.patches(), tok.side, out, &st)?;
        let v = download_half(&hr, out * out * 1024, false, &st)?;
        let sub: Vec<f32> = (0..1024).flat_map(|c| (0..out / 8).flat_map(move |y| (0..out / 8).map(move |x| (c, y, x))))
            .map(|(c, y, x)| v[((y * 8) * out + x * 8) * 1024 + c]).collect();
        report(&format!("{name}_hr (1/8)"), &sub, &ref_f32(&r, &format!("{name}_hr_sub"))?);
        let ch0: Vec<f32> = (0..4).flat_map(|c| (0..out * out).map(move |p| (c, p))).map(|(c, p)| v[p * 1024 + c]).collect();
        report(&format!("{name}_hr (4 ch)"), &ch0, &ref_f32(&r, &format!("{name}_hr_ch0"))?);
    }
    let cam = Camera::new(49.13);
    let ss_proj = project(&cam, &dense_world(SS_RES), &StageMaps { lr: &t512, hr: None, image_res: 512 }, &st)?;
    report("ss_proj", &download_half(&ss_proj, 4096 * 1024, true, &st)?, &ref_f32(&r, "ss_proj")?);
    let c512 = ref_coords(&r, "shape512_coords")?;
    let hr512 = dino.naf(&img512, 512, t512.patches(), t512.side, 512, &st)?;
    let p512 = project(&cam, &coords_world(&c512, SHAPE512_RES), &StageMaps { lr: &t512, hr: Some((&hr512, 512)), image_res: 512 }, &st)?;
    report("shape512_proj", &download_half(&p512, c512.len() * 2048, true, &st)?, &ref_f32(&r, "shape512_proj")?);
    drop((hr512, p512, ss_proj));
    drop((t512, t1024, dino));

    // first evaluation of each sub-model
    let dit = SafeTensors::open(&paths.dit)?;
    let chr = ref_coords(&r, "hr_coords")?;
    check_eval(&r, &dit, Stage::Structure, "structure_generation", &grid_coords(SS_RES), None, threads, &st)?;
    check_eval(&r, &dit, Stage::Shape512, "shape_generation_512", &xyz(&c512), None, threads, &st)?;
    check_eval(&r, &dit, Stage::Shape, "shape_generation", &xyz(&chr), None, threads, &st)?;
    let hr_latent = ref_f32(&r, "hr_latent")?;
    check_eval(&r, &dit, Stage::Texture, "texture_generation", &xyz(&chr), Some(&tokens_major(&hr_latent, chr.len())), threads, &st)?;

    // decoders
    let shape_st = SafeTensors::open(&paths.shape_vae)?;
    let vae = ShapeVae::load(&shape_st, &st)?;
    let ss = ref_f32(&r, "ss_latent")?;
    let coords = vae.decode_structure(&ss[..8 * 4096], &st)?;
    eprintln!("{:24} {} voxels vs {}, same set: {}", "structure decode", coords.len(), c512.len(), coords == c512);
    let s512 = ref_f32(&r, "shape512_latent")?;
    let up = vae.upsample(&denorm(&s512, c512.len(), &SHAPE_MEAN, &SHAPE_STD), c512.clone(), SHAPE512_RES, &st)?;
    let q = quantize_unique(&up, 1536 / 16);
    eprintln!("{:24} {} voxels vs {}, same set: {}", "upsample", q.len(), chr.len(), q == chr);
    let out = vae.decode_shape(&denorm(&hr_latent, chr.len(), &SHAPE_MEAN, &SHAPE_STD), chr.clone(), 1536 / 16, &st, &mut |_, _| {})?;
    for (i, sub) in out.subdivisions.iter().enumerate() {
        let theirs = ref_coords(&r, &format!("sub{}_coords", i + 1)).unwrap_or_default();
        eprintln!("{:24} {} children vs {}, same set: {}", format!("subdivision {i}"), sub.coords.len(), theirs.len(), sub.coords == theirs);
    }
    let (verts, faces) = dual_grid_mesh(&out.voxels, &out.feats, 1536);
    let rv = ref_f32(&r, "mesh_vertices")?;
    eprintln!("{:24} {} vertices vs {}, {} triangles vs {}", "mesh", verts.len(), rv.len() / 3, faces.len(), r.info("mesh_faces")?.shape[0]);
    if verts.len() * 3 == rv.len() {
        report("mesh vertices", &verts.iter().flatten().copied().collect::<Vec<_>>(), &rv);
    }
    drop(vae);
    let tex_st = SafeTensors::open(&paths.texture_vae)?;
    let tex = TextureVae::load(&tex_st, &st)?;
    let tl = ref_f32(&r, "tex_latent")?;
    let colors = tex.decode(&denorm(&tl, chr.len(), &TEX_MEAN, &TEX_STD), chr.clone(), 1536 / 16, &out.subdivisions, &st, &mut |_, _| {})?;
    let rcol = ref_f32(&r, "voxel_colors")?;
    let painted = paint(&verts, &out.voxels, &colors, 1536);
    let rpaint = ref_f32(&r, "vertex_colors")?;
    // the decodes flip a few borderline subdivisions, so compare the voxels both have
    let rvox = ref_coords(&r, "voxel_coords")?;
    let index = VoxelIndex::new(&rvox);
    let (mut a, mut b, mut va, mut vb, mut pa, mut pb) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, c) in out.voxels.iter().enumerate() {
        if let Some(j) = index.get(c[1], c[2], c[3]) {
            let j = j as usize;
            a.extend_from_slice(&colors[i * 6..i * 6 + 6]);
            b.extend_from_slice(&rcol[j * 6..j * 6 + 6]);
            va.extend_from_slice(&verts[i]);
            vb.extend_from_slice(&rv[j * 3..j * 3 + 3]);
            pa.extend_from_slice(&painted[i]);
            pb.extend_from_slice(&rpaint[j * 3..j * 3 + 3]);
        }
    }
    eprintln!("{:24} {} of {} voxels in both", "", a.len() / 6, out.voxels.len());
    report("mesh vertices (shared)", &va, &vb);
    report("voxel colours (shared)", &a, &b);
    report("vertex colours (shared)", &pa, &pb);
    Ok(())
}
