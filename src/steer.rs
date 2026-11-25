//! Activation steering of a DiT's image tokens: per-block statistics of the residual stream while sampling
//! ([`Recorder`]), and a direction added to it after chosen blocks at chosen steps ([`Steer`]).
//!
//! Only Anima uses it. The vector file is built outside the engine (see `tools/steer_build.py`) from two recordings
//! whose prompts differ only by the concept: `dir` f32 [blocks, width] is each block's unit direction (a zero row skips
//! that block), `norm` f32 [blocks] the length `alpha` is measured in (the recorded difference's own, so alpha 1 adds
//! the average shift between the recordings).

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, s, DT_BF16, PM};
use crate::st::SafeTensors;
use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// Sums of each block's output tokens, per denoising step.
pub struct Recorder {
    blocks: usize,
    width: usize,
    /// f32 [steps][blocks][width] column sums, then f32 [steps][blocks] sums of the tokens' L2 norms
    sums: DevBuf,
    norms: DevBuf,
    /// Tokens recorded per step (the same for every block).
    rows: Vec<u64>,
}

impl Recorder {
    pub fn new(blocks: usize, width: usize, steps: usize, st: &Stream) -> Result<Self> {
        let sums = DevBuf::alloc(steps * blocks * width * 4)?;
        sums.upload(0, &vec![0u8; sums.size()], st)?;
        let norms = DevBuf::alloc(steps * blocks * 4)?;
        norms.upload(0, &vec![0u8; norms.size()], st)?;
        Ok(Recorder { blocks, width, sums, norms, rows: vec![0; steps] })
    }

    pub fn steps(&self) -> usize {
        self.rows.len()
    }

    /// Add block `b`'s output, bf16 [rows, width] at `x`, to step `step`'s sums. Block 0 counts the rows.
    pub fn record(&mut self, step: usize, b: usize, x: PM, rows: usize, st: &Stream) -> Result<()> {
        if step >= self.steps() || b >= self.blocks {
            return Ok(());
        }
        if b == 0 {
            self.rows[step] += rows as u64;
        }
        let i = step * self.blocks + b;
        k!(gpu_steer_record(x, rows as i32, self.width as i32, self.sums.at(i * self.width * 4), self.norms.at(i * 4), s(st)))
    }

    /// Write `mean` f32 [steps, blocks, width], `norm` f32 [steps, blocks] (mean token norm) and `tokens` i64 [steps].
    pub fn save(&self, path: &Path, st: &Stream) -> Result<()> {
        let (steps, nb, w) = (self.steps(), self.blocks, self.width);
        let mut sums = vec![0u8; self.sums.size()];
        self.sums.download(0, &mut sums, st)?;
        let mut norms = vec![0u8; self.norms.size()];
        self.norms.download(0, &mut norms, st)?;
        st.sync()?;
        let f = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let (mut mean, mut norm) = (f(&sums), f(&norms));
        for step in 0..steps {
            let n = self.rows[step].max(1) as f32;
            mean[step * nb * w..(step + 1) * nb * w].iter_mut().for_each(|v| *v /= n);
            norm[step * nb..(step + 1) * nb].iter_mut().for_each(|v| *v /= n);
        }
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut t = BTreeMap::new();
        t.insert("mean".to_string(), ("F32".to_string(), vec![steps, nb, w], bytes(&mean)));
        t.insert("norm".to_string(), ("F32".to_string(), vec![steps, nb], bytes(&norm)));
        t.insert("tokens".to_string(), ("I64".to_string(), vec![steps], self.rows.iter().flat_map(|r| (*r as i64).to_le_bytes()).collect()));
        write_safetensors(path, &t)
    }
}

/// Every `stride`-th image token of chosen blocks at chosen steps, kept whole for fitting a steering map: two runs of
/// the same seeds then hold the same tokens of the same noise, row for row.
pub struct TokenSampler {
    width: usize,
    blocks: Vec<usize>,
    steps: (usize, usize),
    stride: usize,
    /// bf16 rows by (block, step)
    rows: BTreeMap<(usize, usize), Vec<u8>>,
}

impl TokenSampler {
    pub fn new(width: usize, blocks: Vec<usize>, steps: (usize, usize), stride: usize) -> Self {
        TokenSampler { width, blocks, steps, stride: stride.max(1), rows: BTreeMap::new() }
    }

    /// Keep tokens of block `b`'s output, bf16 [batch * l, width] at `x`, `l` tokens per image.
    pub fn record(&mut self, step: usize, b: usize, x: PM, batch: usize, l: usize, st: &Stream) -> Result<()> {
        if !self.blocks.contains(&b) || !(self.steps.0..self.steps.1).contains(&step) {
            return Ok(());
        }
        let per = l.div_ceil(self.stride);
        let row = self.width * 2;
        let tmp = DevBuf::alloc_async(batch * per * row, st)?;
        for i in 0..batch {
            let src = crate::ffi::off(x, i * l * row);
            let n = (l - 1) / self.stride + 1;
            crate::cuda::memcpy_2d_d2d(tmp.at(i * per * row), row, src, self.stride * row, row, n, st)?;
        }
        let mut host = vec![0u8; batch * per * row];
        tmp.download(0, &mut host, st)?;
        st.sync()?;
        self.rows.entry((b, step)).or_default().extend_from_slice(&host);
        Ok(())
    }

    /// Write `tok.b<block>.s<step>` bf16 [rows, width] for everything kept.
    pub fn save(&self, path: &Path) -> Result<()> {
        let t = self
            .rows
            .iter()
            .map(|((b, s), v)| (format!("tok.b{b}.s{s}"), ("BF16".to_string(), vec![v.len() / (self.width * 2), self.width], v.clone())))
            .collect();
        write_safetensors(path, &t)
    }
}

/// A low-rank map of each token added to it after one block: `x += alpha * ((x U^T) V^T + c)`.
struct Map {
    block: usize,
    rank: usize,
    /// bf16 [rank, width] and [width, rank]
    u: DevBuf,
    v: DevBuf,
    /// f32 [width] unit direction of `c`, and its length
    c_dir: DevBuf,
    c_len: f32,
}

/// A direction added to the image tokens after some blocks, during some steps, or a low-rank map of them.
pub struct Steer {
    width: usize,
    alpha: f32,
    /// f32 [blocks][width] unit directions
    dirs: DevBuf,
    /// Added per block: `alpha` times the file's `norm`; 0 skips the block.
    amount: Vec<f32>,
    maps: Vec<Map>,
    /// Steps `first..end` are steered.
    steps: (usize, usize),
    /// Rescale each token to its norm before steering.
    pub preserve_norm: bool,
}

impl Steer {
    /// `blocks` and `steps` are half-open ranges; `alpha` multiplies each block's `norm` from the file.
    pub fn load(path: &Path, alpha: f32, blocks: (usize, usize), steps: (usize, usize), preserve_norm: bool, n_blocks: usize,
        width: usize, st: &Stream) -> Result<Self> {
        let f = SafeTensors::open(path)?;
        let dirs = DevBuf::alloc(n_blocks * width * 4)?;
        let mut amount = vec![0.0; n_blocks];
        let mut maps = Vec::new();
        if f.has("map_blocks") {
            // a low-rank map per block (tools/steer_fit.py)
            let bf = |v: &[f32]| v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect::<Vec<u8>>();
            for b in f.f32_vec("map_blocks", false)?.iter().map(|b| *b as usize).filter(|b| (blocks.0..blocks.1).contains(b)) {
                let u = f.f32_vec(&format!("U.b{b}"), false)?;
                let v = f.f32_vec(&format!("V.b{b}"), false)?;
                let c = f.f32_vec(&format!("c.b{b}"), false)?;
                let rank = u.len() / width;
                if u.len() != rank * width || v.len() != rank * width || c.len() != width {
                    bail!("{}: block {b}'s map isn't [rank, {width}]", path.display());
                }
                let (ud, vd) = (DevBuf::alloc(u.len() * 2)?, DevBuf::alloc(v.len() * 2)?);
                ud.upload(0, &bf(&u), st)?;
                vd.upload(0, &bf(&v), st)?;
                let c_len = c.iter().map(|x| x * x).sum::<f32>().sqrt();
                let c_dir = DevBuf::alloc(width * 4)?;
                let unit: Vec<f32> = c.iter().map(|x| x / c_len.max(1e-12)).collect();
                c_dir.upload(0, &crate::st::f32_bytes(&unit), st)?;
                maps.push(Map { block: b, rank, u: ud, v: vd, c_dir, c_len });
            }
        } else {
            let dir = f.f32_vec("dir", false)?;
            let norm = f.f32_vec("norm", false)?;
            if dir.len() != n_blocks * width || norm.len() != n_blocks {
                bail!("{}: expected dir [{n_blocks}, {width}] and norm [{n_blocks}]", path.display());
            }
            for (b, a) in amount.iter_mut().enumerate() {
                let row = &dir[b * width..(b + 1) * width];
                if (blocks.0..blocks.1).contains(&b) && row.iter().any(|v| *v != 0.0) {
                    *a = alpha * norm[b];
                }
            }
            dirs.upload(0, &crate::st::f32_bytes(&dir), st)?;
        }
        st.sync()?;
        Ok(Steer { width, alpha, dirs, amount, maps, steps, preserve_norm })
    }

    /// Steer block `b`'s output, bf16 [rows, width] at `x`, if `step` and `b` are chosen.
    pub fn apply(&self, step: usize, b: usize, x: PM, rows: usize, st: &Stream) -> Result<()> {
        if !(self.steps.0..self.steps.1).contains(&step) {
            return Ok(());
        }
        let w = self.width as i64;
        for m in self.maps.iter().filter(|m| m.block == b) {
            // x += alpha * (x U^T) V^T, then alpha * c
            let tmp = DevBuf::alloc_async(rows * m.rank * 2, st)?;
            let r = m.rank as i64;
            k!(gpu_gemm(x, w, m.u.ptr(), w, tmp.ptr(), r, null(), rows as i64, r, w, 1, DT_BF16, DT_BF16, 1.0, 0.0, s(st)))?;
            k!(gpu_gemm(tmp.ptr(), r, m.v.ptr(), r, x, w, null(), rows as i64, w, r, 1, DT_BF16, DT_BF16, self.alpha, 1.0, s(st)))?;
            if m.c_len > 0.0 {
                k!(gpu_steer_add(x, rows as i32, self.width as i32, m.c_dir.ptr(), self.alpha * m.c_len, 0, s(st)))?;
            }
        }
        let a = self.amount.get(b).copied().unwrap_or(0.0);
        if a == 0.0 {
            return Ok(());
        }
        k!(gpu_steer_add(x, rows as i32, self.width as i32, self.dirs.at(b * self.width * 4), a, self.preserve_norm as i32, s(st)))
    }
}

/// Parse `a-b` (half-open, `b` exclusive) or `a` (just `a`).
pub fn parse_range(s: &str) -> Result<(usize, usize)> {
    match s.split_once('-') {
        Some((a, b)) => {
            let (a, b) = (a.trim().parse::<usize>()?, b.trim().parse::<usize>()?);
            if b <= a {
                bail!("range {s} is empty");
            }
            Ok((a, b))
        }
        None => {
            let a = s.trim().parse::<usize>()?;
            Ok((a, a + 1))
        }
    }
}

pub(crate) fn write_safetensors(path: &Path, tensors: &BTreeMap<String, (String, Vec<usize>, Vec<u8>)>) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (k, (dt, shape, data)) in tensors {
        header.insert(k.clone(), serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [offset, offset + data.len()]}));
        offset += data.len();
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + offset);
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    for (_, _, d) in tensors.values() {
        out.extend_from_slice(d);
    }
    std::fs::write(path, out)?;
    Ok(())
}
