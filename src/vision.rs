//! Qwen3-VL vision tower for prompt images: patch embedding, attention blocks with 2D RoPE, a merger to the
//! text encoder's width, and the three DeepStack mergers. Krea 2's 4B has 24 blocks of 1024 features, Qwen Image
//! 2.1's 8B has 27 of 1152, and MiniMax H3's 32B the 8B's tower with a wider merger (under `visual.`). Each image runs on its own, in f32 like ComfyUI, with its bf16 weights streamed
//! from pinned RAM.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F32, PM, PRO_LAYERNORM};
use crate::offload::{plan, Layout, WeightSet};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{bail, Result};
use half::bf16;

const HEADS: usize = 16;
const PATCH: usize = 16;
const MERGE: usize = 2;
const FRAME: usize = 3 * PATCH * PATCH;
const POS_SIDE: usize = 48;
const EPS: f32 = 1e-6;
const THETA: f32 = 10000.0;
/// where the tower sits: the 4B and 8B files, then MiniMax H3's 32B
const PREFIXES: [&str; 2] = ["model.visual.", "visual."];
// bounds of process_qwen2vl_images
const MIN_PIXELS: usize = 3136;
const MAX_PIXELS: usize = 12845056;

/// The size of a tower: the 4B text encoder's, or the 8B's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    pub d: usize,
    pub ff: usize,
    pub depth: usize,
    pub deepstack: [usize; 3],
    /// Width of the merged tokens, the text encoder's.
    pub out: usize,
}

pub const QWEN3VL_4B: Spec = Spec { d: 1024, ff: 4096, depth: 24, deepstack: [5, 11, 17], out: 2560 };
pub const QWEN3VL_8B: Spec = Spec { d: 1152, ff: 4304, depth: 27, deepstack: [8, 16, 24], out: 4096 };
pub const QWEN3VL_32B: Spec = Spec { out: 5120, ..QWEN3VL_8B };

impl Spec {
    /// The tower in `st` and its prefix, read from its patch embedding and merger.
    pub fn detect(st: &SafeTensors) -> Result<(Spec, &'static str)> {
        let Some(pre) = PREFIXES.into_iter().find(|p| st.has(&format!("{p}patch_embed.proj.weight"))) else {
            bail!("this text encoder file has no vision tower");
        };
        let out = st.info(&format!("{pre}merger.linear_fc2.weight"))?.shape[0];
        let spec = match st.info(&format!("{pre}patch_embed.proj.weight"))?.shape[0] {
            d if d == QWEN3VL_4B.d => QWEN3VL_4B,
            d if d == QWEN3VL_8B.d && out == QWEN3VL_32B.out => QWEN3VL_32B,
            d if d == QWEN3VL_8B.d => QWEN3VL_8B,
            d => bail!("a vision tower of {d} features, which is neither Qwen3-VL 4B's, 8B's nor 32B's"),
        };
        if spec.out != out {
            bail!("a Qwen3-VL vision tower merging to {out} features");
        }
        Ok((spec, pre))
    }

    fn hd(&self) -> usize {
        self.d / HEADS
    }

    fn merged(&self) -> usize {
        self.d * MERGE * MERGE
    }
}

/// An RGB image, `[height, width, 3]`, 0 to 1.
#[derive(Clone, PartialEq)]
pub struct PromptImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<f32>,
}

/// One image's tokens for the text encoder, f32 `[tokens, spec.out]`.
pub struct Embeds {
    pub merged: DevBuf,
    pub deepstack: [DevBuf; 3],
    pub tokens: usize,
    /// Merged (height, width) of the token grid.
    pub grid: (usize, usize),
}

// Only reached through the Generator, which one thread holds at a time.
unsafe impl Sync for Embeds {}

/// Patch grid (height, width), in patches, that ComfyUI resizes an image to.
pub fn patch_grid(height: usize, width: usize) -> (usize, usize) {
    let factor = (PATCH * MERGE) as f64;
    let (h, w) = (height as f64, width as f64);
    let mut hb = (h / factor).round_ties_even() * factor;
    let mut wb = (w / factor).round_ties_even() * factor;
    if hb * wb > MAX_PIXELS as f64 {
        let beta = (h * w / MAX_PIXELS as f64).sqrt();
        hb = factor.max((h / beta / factor).floor() * factor);
        wb = factor.max((w / beta / factor).floor() * factor);
    } else if hb * wb < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        hb = (h * beta / factor).ceil() * factor;
        wb = (w * beta / factor).ceil() * factor;
    }
    (hb as usize / PATCH, wb as usize / PATCH)
}

/// `F.interpolate(mode="bilinear", align_corners=False)` of `[h, w, 3]`.
fn bilinear(img: &PromptImage, oh: usize, ow: usize) -> Vec<f32> {
    if (oh, ow) == (img.height, img.width) {
        return img.rgb.clone();
    }
    let weights = |out: usize, inp: usize| -> Vec<(usize, usize, f32)> {
        let scale = inp as f32 / out as f32;
        (0..out).map(|i| {
            let p = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
            let i0 = (p.floor() as usize).min(inp - 1);
            (i0, (i0 + 1).min(inp - 1), p - i0 as f32)
        }).collect()
    };
    let (ys, xs) = (weights(oh, img.height), weights(ow, img.width));
    let mut out = Vec::with_capacity(oh * ow * 3);
    let at = |y: usize, x: usize, c: usize| img.rgb[(y * img.width + x) * 3 + c];
    for &(y0, y1, fy) in &ys {
        for &(x0, x1, fx) in &xs {
            for c in 0..3 {
                let top = at(y0, x0, c) * (1.0 - fx) + at(y0, x1, c) * fx;
                let bottom = at(y1, x0, c) * (1.0 - fx) + at(y1, x1, c) * fx;
                out.push(top * (1.0 - fy) + bottom * fy);
            }
        }
    }
    out
}

/// Patches `[n, 3 * 16 * 16]`, normalized to -1 to 1, in merge-block order.
fn patches(rgb: &[f32], gh: usize, gw: usize) -> Vec<f32> {
    let stride = gw * PATCH;
    let mut out = Vec::with_capacity(gh * gw * FRAME);
    for (row, col) in merge_order(gh, gw) {
        for c in 0..3 {
            for y in 0..PATCH {
                for x in 0..PATCH {
                    out.push((rgb[((row * PATCH + y) * stride + col * PATCH + x) * 3 + c] - 0.5) / 0.5);
                }
            }
        }
    }
    out
}

/// (row, column) of every patch: each 2 x 2 block stays together, for the mergers.
fn merge_order(gh: usize, gw: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(gh * gw);
    for r in 0..gh / MERGE {
        for c in 0..gw / MERGE {
            for i in 0..MERGE {
                for j in 0..MERGE {
                    v.push((r * MERGE + i, c * MERGE + j));
                }
            }
        }
    }
    v
}

/// The learned 48 x 48 position table, bilinear at this grid. Weights and products stay bf16, as
/// `fast_pos_embed_interpolate` leaves them.
fn position_embed(table: &[f32], gh: usize, gw: usize, d: usize) -> Vec<f32> {
    let line = |n: usize| -> Vec<f32> {
        let last = (POS_SIDE - 1) as f32;
        (0..n).map(|i| if n > 1 { if i == n - 1 { last } else { last * i as f32 / (n - 1) as f32 } } else { 0.0 }).collect()
    };
    let (ys, xs) = (line(gh), line(gw));
    let r = |v: f32| bf16::from_f32(v).to_f32();
    let mut grid = vec![0f32; gh * gw * d];
    for (i, &y) in ys.iter().enumerate() {
        for (j, &x) in xs.iter().enumerate() {
            let (y0, x0) = (y as usize, x as usize);
            let (y1, x1) = ((y0 + 1).min(POS_SIDE - 1), (x0 + 1).min(POS_SIDE - 1));
            let (dy, dx) = (y - y0 as f32, x - x0 as f32);
            let corners = [(y0, x0, (1.0 - dy) * (1.0 - dx)), (y0, x1, (1.0 - dy) * dx), (y1, x0, dy * (1.0 - dx)), (y1, x1, dy * dx)];
            let out = &mut grid[(i * gw + j) * d..][..d];
            for (cy, cx, w) in corners {
                let (w, row) = (r(w), &table[(cy * POS_SIDE + cx) * d..][..d]);
                for (o, v) in out.iter_mut().zip(row) {
                    *o = r(*o + r(v * w));
                }
            }
        }
    }
    let mut out = Vec::with_capacity(gh * gw * d);
    for (row, col) in merge_order(gh, gw) {
        out.extend_from_slice(&grid[(row * gw + col) * d..][..d]);
    }
    out
}

/// RoPE table `[n][HD / 2][(cos, sin)]`: the row's frequencies, then the column's.
fn rope_table(gh: usize, gw: usize, hd: usize) -> Vec<f32> {
    let dim = hd / 2;
    let inv: Vec<f32> = (0..dim / 2).map(|i| 1.0 / THETA.powf((2 * i) as f32 / dim as f32)).collect();
    let mut t = Vec::with_capacity(gh * gw * hd);
    for (row, col) in merge_order(gh, gw) {
        for pos in [row, col] {
            for f in &inv {
                let a = pos as f32 * f;
                t.push(a.cos());
                t.push(a.sin());
            }
        }
    }
    t
}

pub struct Vision {
    spec: Spec,
    pre: &'static str,
    st: SafeTensors,
    ws: WeightSet,
    pos: Vec<f32>,
    zeros: DevBuf,
}

/// fc1 + fc2 of a merger, and its norm.
fn merger_layout(st: &SafeTensors, pre: &str, p: &str) -> Result<Layout> {
    let n = |x: &str| format!("{pre}{p}.{x}");
    let mut l = Layout::new();
    l.f32(st, "nw", &n("norm.weight"), 0.0)?;
    l.f32(st, "nb", &n("norm.bias"), 0.0)?;
    l.bf16(st, "w1", &[n("linear_fc1.weight")])?;
    l.f32(st, "b1", &n("linear_fc1.bias"), 0.0)?;
    l.bf16(st, "w2", &[n("linear_fc2.weight")])?;
    l.f32(st, "b2", &n("linear_fc2.bias"), 0.0)?;
    Ok(l)
}

fn block_layout(st: &SafeTensors, pre: &str, i: usize) -> Result<Layout> {
    let n = |x: &str| format!("{pre}blocks.{i}.{x}");
    let mut l = Layout::new();
    l.f32(st, "n1w", &n("norm1.weight"), 0.0)?;
    l.f32(st, "n1b", &n("norm1.bias"), 0.0)?;
    l.f32(st, "n2w", &n("norm2.weight"), 0.0)?;
    l.f32(st, "n2b", &n("norm2.bias"), 0.0)?;
    l.bf16(st, "qkv", &[n("attn.qkv.weight")])?;
    l.f32(st, "qkvb", &n("attn.qkv.bias"), 0.0)?;
    l.bf16(st, "proj", &[n("attn.proj.weight")])?;
    l.f32(st, "projb", &n("attn.proj.bias"), 0.0)?;
    l.bf16(st, "fc1", &[n("mlp.linear_fc1.weight")])?;
    l.f32(st, "fc1b", &n("mlp.linear_fc1.bias"), 0.0)?;
    l.bf16(st, "fc2", &[n("mlp.linear_fc2.weight")])?;
    l.f32(st, "fc2b", &n("mlp.linear_fc2.bias"), 0.0)?;
    Ok(l)
}

/// The Conv3d weight of two identical frames, as one matrix `[d, 3 * 16 * 16]`: its slices add up.
fn patch_matrix(st: &SafeTensors, pre: &str, d: usize) -> Result<Vec<f32>> {
    let name = format!("{pre}patch_embed.proj.weight");
    let v = st.f32_vec(&name, false)?;
    let mut out = vec![0f32; d * FRAME];
    for o in 0..d {
        for c in 0..3 {
            for t in 0..2 {
                let src = &v[((o * 3 + c) * 2 + t) * PATCH * PATCH..][..PATCH * PATCH];
                for (d, w) in out[(o * 3 + c) * PATCH * PATCH..][..PATCH * PATCH].iter_mut().zip(src) {
                    *d += w;
                }
            }
        }
    }
    Ok(out)
}

impl Vision {
    pub fn load(st: SafeTensors, threads: usize, s0: &Stream) -> Result<Self> {
        let (spec, pre) = Spec::detect(&st)?;
        // streamed units have to be in the order they run
        let mut patch = Layout::new();
        patch.bytes("w", f32_bytes(&patch_matrix(&st, pre, spec.d)?));
        patch.f32(&st, "b", &format!("{pre}patch_embed.proj.bias"), 0.0)?;
        let mut layouts = vec![patch];
        for b in 0..spec.depth {
            layouts.push(block_layout(&st, pre, b)?);
            if let Some(i) = spec.deepstack.iter().position(|&l| l == b) {
                layouts.push(merger_layout(&st, pre, &format!("deepstack_merger_list.{i}"))?);
            }
        }
        layouts.push(merger_layout(&st, pre, "merger")?);
        let p = plan(&layouts, 0, 2);
        let ws = WeightSet::load("vision tower", &st, layouts, p, threads)?;
        let pos = st.f32_vec(&format!("{pre}pos_embed.weight"), false)?;
        let zeros = DevBuf::alloc(spec.merged() * 4)?;
        zeros.upload(0, &vec![0u8; spec.merged() * 4], s0)?;
        s0.sync()?;
        Ok(Vision { spec, pre, st, ws, pos, zeros })
    }

    /// The tokens of one image.
    pub fn encode(&mut self, img: &PromptImage, st: &Stream, mut probe: Option<&mut dyn FnMut(&str, &DevBuf, usize)>) -> Result<Embeds> {
        let _ = (&self.st, self.pre);
        let Spec { d: dm, ff: ff_w, depth, deepstack, out: out_w } = self.spec;
        let (hd, merged_w) = (self.spec.hd(), self.spec.merged());
        let (gh, gw) = patch_grid(img.height, img.width);
        let n = gh * gw;
        let tokens = n / (MERGE * MERGE);
        let resized = bilinear(img, gh * PATCH, gw * PATCH);
        let px = DevBuf::alloc_async(n * FRAME * 4, st)?;
        px.upload(0, &f32_bytes(&patches(&resized, gh, gw)), st)?;
        let rope = DevBuf::alloc_async(n * hd * 4, st)?;
        rope.upload(0, &f32_bytes(&rope_table(gh, gw, hd)), st)?;

        let nf = n as i64;
        let xb = DevBuf::alloc_async(n * dm * 4, st)?;
        let wf = DevBuf::alloc_async((merged_w * merged_w).max(ff_w * dm) * 4, st)?; // largest weight
        let normed = DevBuf::alloc_async(n * dm * 4, st)?;
        let qkv = DevBuf::alloc_async(n * 3 * dm * 4, st)?;
        let attn = DevBuf::alloc_async(n * dm * 4, st)?;
        let y = DevBuf::alloc_async(n * dm * 4, st)?;
        // query rows per chunk of the attention: the f32 scores stay under ~256 MB
        let chunk = ((256 << 20) / (n * 4)).clamp(64, n);
        let scores = DevBuf::alloc_async(chunk * n * 4, st)?;
        let ff = DevBuf::alloc_async(n * ff_w * 4, st)?;
        let x = xb.ptr();
        // y = x @ w^T + b, from a bf16 weight
        let lin = |a: PM, wq: PM, b: PM, rows: i64, out_w: usize, in_w: usize, out: PM| -> Result<()> {
            k!(gpu_bf16_to_f32(wq, wf.ptr(), (out_w * in_w) as i64, s(st)))?;
            k!(gpu_gemm(a, in_w as i64, wf.ptr(), in_w as i64, out, out_w as i64, b, rows, out_w as i64, in_w as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))
        };
        let add = |dst: PM, v: PM, count: usize| -> Result<()> {
            let (p, c) = ([dst as crate::ffi::P, v as crate::ffi::P], [1.0f32, 1.0]);
            k!(gpu_lincomb(dst, count as i64, 2, p.as_ptr(), c.as_ptr(), s(st)))
        };
        let zeros = self.zeros.ptr();
        let norm = |a: PM, w: PM, b: PM, width: usize, rows: usize, out: PM| -> Result<()> {
            k!(gpu_prologue_rows(DT_F32, PRO_LAYERNORM, a, width as i64, null(), 0, 0, w, EPS, zeros, b, out, width as i64, rows as i32, width as i32, s(st)))
        };

        let mut unit = 0;
        let base = self.ws.acquire(unit, st)?;
        let lay = &self.ws.layouts[unit];
        k!(gpu_gemm(px.ptr(), FRAME as i64, off(base, lay.off("w")), FRAME as i64, x, dm as i64, off(base, lay.off("b")), nf, dm as i64,
            FRAME as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
        self.ws.release(unit, st)?;
        drop(px);
        if let Some(p) = probe.as_mut() {
            p("vision_patch", &xb, n * dm * 4);
        }
        let pos = DevBuf::alloc_async(n * dm * 4, st)?;
        pos.upload(0, &f32_bytes(&position_embed(&self.pos, gh, gw, dm)), st)?;
        add(x, pos.ptr(), n * dm)?;
        drop(pos);

        let merged_rows = |x: PM, base: PM, lay: &Layout, out: &DevBuf, h: &DevBuf, g: &DevBuf| -> Result<()> {
            let w = |key: &str| off(base, lay.off(key));
            lin(x, w("w1"), w("b1"), tokens as i64, merged_w, merged_w, h.ptr())?;
            crate::cuda::memcpy_d2d(g.ptr(), h.ptr(), tokens * merged_w * 4, st)?;
            k!(gpu_gelu_f32(g.ptr(), (tokens * merged_w) as i64, 0, s(st)))?;
            lin(g.ptr(), w("w2"), w("b2"), tokens as i64, out_w, merged_w, out.ptr())
        };
        let h1 = DevBuf::alloc_async(tokens * merged_w * 4, st)?;
        let h2 = DevBuf::alloc_async(tokens * merged_w * 4, st)?;
        let mut deep: Vec<DevBuf> = Vec::with_capacity(deepstack.len());
        for b in 0..depth {
            unit += 1;
            let base = self.ws.acquire(unit, st)?;
            let lay = &self.ws.layouts[unit];
            let w = |key: &str| off(base, lay.off(key));
            norm(x, w("n1w"), w("n1b"), dm, n, normed.ptr())?;
            lin(normed.ptr(), w("qkv"), w("qkvb"), nf, 3 * dm, dm, qkv.ptr())?;
            let q = qkv.ptr();
            let (kp, vp) = (off(q, dm * 4), off(q, 2 * dm * 4));
            k!(gpu_rope_rows(q, (3 * dm) as i64, rope.ptr(), n as i32, HEADS as i32, hd as i32, s(st)))?;
            k!(gpu_rope_rows(kp, (3 * dm) as i64, rope.ptr(), n as i32, HEADS as i32, hd as i32, s(st)))?;
            // per head, a chunk of query rows at a time: scores, softmax, then the weighted values
            let row = (3 * dm) as i64;
            for h in 0..HEADS {
                let col = h * hd * 4;
                for r0 in (0..n).step_by(chunk) {
                    let rows = chunk.min(n - r0);
                    k!(gpu_gemm(off(q, r0 * 3 * dm * 4 + col), row, off(kp, col), row, scores.ptr(), n as i64, null(), rows as i64, n as i64, hd as i64, 1,
                        DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
                    k!(gpu_softmax_f32(scores.ptr(), rows as i32, n as i32, 1.0 / (hd as f32).sqrt(), s(st)))?;
                    k!(gpu_gemm(scores.ptr(), n as i64, off(vp, col), row, off(attn.ptr(), r0 * dm * 4 + col), dm as i64, null(), rows as i64, hd as i64,
                        n as i64, 0, DT_F32, DT_F32, 1.0, 0.0, s(st)))?;
                }
            }
            lin(attn.ptr(), w("proj"), w("projb"), nf, dm, dm, y.ptr())?;
            add(x, y.ptr(), n * dm)?;
            norm(x, w("n2w"), w("n2b"), dm, n, normed.ptr())?;
            lin(normed.ptr(), w("fc1"), w("fc1b"), nf, ff_w, dm, ff.ptr())?;
            k!(gpu_gelu_f32(ff.ptr(), (n * ff_w) as i64, 1, s(st)))?;
            lin(ff.ptr(), w("fc2"), w("fc2b"), nf, dm, ff_w, y.ptr())?;
            add(x, y.ptr(), n * dm)?;
            self.ws.release(unit, st)?;
            if let Some(p) = probe.as_mut() {
                if b == 0 || b == depth - 1 {
                    p(&format!("vision_block{b}"), &xb, n * dm * 4);
                }
            }
            if deepstack.contains(&b) {
                unit += 1;
                let base = self.ws.acquire(unit, st)?;
                let lay = &self.ws.layouts[unit];
                norm(x, off(base, lay.off("nw")), off(base, lay.off("nb")), merged_w, tokens, h1.ptr())?;
                let out = DevBuf::alloc_async(tokens * out_w * 4, st)?;
                merged_rows(h1.ptr(), base, lay, &out, &h2, &normed)?;
                self.ws.release(unit, st)?;
                deep.push(out);
            }
        }
        unit += 1;
        let base = self.ws.acquire(unit, st)?;
        let lay = &self.ws.layouts[unit];
        norm(x, off(base, lay.off("nw")), off(base, lay.off("nb")), dm, n, normed.ptr())?;
        let merged = DevBuf::alloc_async(tokens * out_w * 4, st)?;
        merged_rows(normed.ptr(), base, lay, &merged, &h2, &h1)?;
        self.ws.release(unit, st)?;
        st.sync()?;
        if let Some(p) = probe.as_mut() {
            p("vision_merged", &merged, tokens * out_w * 4);
            for (i, d) in deep.iter().enumerate() {
                p(&format!("vision_deepstack{i}"), d, tokens * out_w * 4);
            }
        }
        let [d0, d1, d2]: [DevBuf; 3] = deep.try_into().map_err(|_| anyhow::anyhow!("deepstack"))?;
        Ok(Embeds { merged, deepstack: [d0, d1, d2], tokens, grid: (gh / MERGE, gw / MERGE) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_match_process_qwen2vl_images() {
        // Python's round: 10.5 goes to 10 and 13.5 to 14
        assert_eq!(patch_grid(336, 432), (20, 28));
        assert_eq!(patch_grid(384, 384), (24, 24));
        // below min_pixels it scales up
        assert_eq!(patch_grid(16, 16), (4, 4));
    }

    #[test]
    fn merge_order_keeps_blocks_together() {
        assert_eq!(merge_order(2, 4), [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
    }

    #[test]
    fn bilinear_keeps_a_constant_image() {
        let img = PromptImage { width: 3, height: 2, rgb: vec![0.25; 18] };
        assert!(bilinear(&img, 5, 7).iter().all(|v| (v - 0.25).abs() < 1e-6));
    }
}
