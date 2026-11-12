//! Read-only safetensors access over a memory map (files are never copied or modified).

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[link(name = "kernel32")]
extern "system" {
    fn VirtualUnlock(addr: *const std::ffi::c_void, size: usize) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dt {
    F32,
    F16,
    BF16,
    I8,
    F8,
    U8,
    I32,
    I64,
}

impl Dt {
    pub fn size(self) -> usize {
        match self {
            Dt::F32 => 4,
            Dt::F16 | Dt::BF16 => 2,
            Dt::I8 | Dt::U8 | Dt::F8 => 1,
            Dt::I32 => 4,
            Dt::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dt,
    pub shape: Vec<usize>,
    /// Which of the model's files holds it.
    pub file: usize,
    pub start: usize,
    pub len: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    mmaps: Vec<memmap2::Mmap>,
    pub tensors: HashMap<String, TensorInfo>,
    /// The diffusers `config.json` next to a sharded model.
    pub config: Option<serde_json::Value>,
    /// The `__metadata__` strings of the files, merged.
    pub metadata: HashMap<String, String>,
}

/// The index and config of a diffusers model stored in shards, if `path` is one of its shards.
pub fn diffusers_shards(path: &Path) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".safetensors")?;
    // <base>-00001-of-00003
    let base = stem.rsplit_once("-of-").and_then(|(a, _)| a.rsplit_once('-')).map(|(a, _)| a)?;
    let dir = path.parent()?;
    let index = dir.join(format!("{base}.safetensors.index.json"));
    let config = dir.join("config.json");
    (index.is_file() && config.is_file()).then_some((index, config))
}

impl SafeTensors {
    /// One safetensors file, or every shard of a diffusers model when `path` is one of them.
    pub fn open(path: &Path) -> Result<Self> {
        let Some((index, config)) = diffusers_shards(path) else {
            let mut st = SafeTensors { mmaps: Vec::new(), tensors: HashMap::new(), config: None, metadata: HashMap::new() };
            st.add_file(path)?;
            return Ok(st);
        };
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?).with_context(|| format!("reading {}", index.display()))?;
        let mut files: Vec<&str> = idx["weight_map"].as_object().ok_or_else(|| anyhow!("{}: no weight_map", index.display()))?
            .values().filter_map(|v| v.as_str()).collect();
        files.sort_unstable();
        files.dedup();
        let config = serde_json::from_slice(&std::fs::read(&config)?).with_context(|| format!("reading {}", config.display()))?;
        let mut st = SafeTensors { mmaps: Vec::new(), tensors: HashMap::new(), config: Some(config), metadata: HashMap::new() };
        let dir = index.parent().unwrap();
        for f in files {
            st.add_file(&dir.join(f))?;
        }
        Ok(st)
    }

    fn add_file(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let fi = self.mmaps.len();
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n])?;
        let base = 8 + n;
        for (k, v) in header.as_object().ok_or_else(|| anyhow!("bad header"))? {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().into_iter().flatten() {
                    if let Some(s) = mv.as_str() {
                        self.metadata.insert(mk.clone(), s.to_string());
                    }
                }
                continue;
            }
            let dtype = match v["dtype"].as_str().unwrap_or("") {
                "F32" => Dt::F32,
                "F16" => Dt::F16,
                "BF16" => Dt::BF16,
                "I8" => Dt::I8,
                "U8" => Dt::U8,
                "F8_E4M3" => Dt::F8,
                "I32" => Dt::I32,
                "I64" => Dt::I64,
                d => bail!("unsupported dtype {d} for {k}"),
            };
            let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let off = v["data_offsets"].as_array().unwrap();
            let (a, b) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            self.tensors.insert(k.clone(), TensorInfo { dtype, shape, file: fi, start: base + a, len: b - a });
        }
        self.mmaps.push(mmap);
        Ok(())
    }

    /// Name rows `first..first + rows` of `name` (along its first axis) as the tensor `as_name`, without copying.
    pub fn alias_rows(&mut self, name: &str, as_name: &str, first: usize, rows: usize) -> Result<()> {
        let i = self.info(name)?.clone();
        let row = i.len / i.shape[0];
        if first + rows > i.shape[0] {
            bail!("{name}: rows {first}..{} of {}", first + rows, i.shape[0]);
        }
        let mut shape = i.shape.clone();
        shape[0] = rows;
        self.tensors.insert(as_name.to_string(), TensorInfo { shape, start: i.start + first * row, len: rows * row, ..i });
        Ok(())
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let i = self.info(name)?;
        Ok(&self.mmaps[i.file][i.start..i.start + i.len])
    }

    /// Take a tensor's pages out of this process. They stay in the file cache, so a later read costs
    /// a soft fault per page rather than a disk read: not for tensors read again on every step.
    pub fn evict(&self, name: &str) {
        if let Ok(i) = self.info(name) {
            let m = &self.mmaps[i.file];
            let (start, end) = (i.start & !4095, (i.start + i.len).next_multiple_of(4096).min(m.len()));
            // on pages that aren't locked this reports failure and does what we want, as documented
            unsafe { VirtualUnlock(m[start..end].as_ptr().cast(), end - start) };
        }
    }

    /// Tensor as f32 values, first rounded through bf16 when `via_bf16` (ComfyUI keeps the
    /// non-quantized parameters of these models in bf16).
    pub fn f32_vec(&self, name: &str, via_bf16: bool) -> Result<Vec<f32>> {
        let i = self.info(name)?;
        let b = self.bytes(name)?;
        let v: Vec<f32> = match i.dtype {
            Dt::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Dt::F16 => b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            Dt::BF16 => b.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            d => bail!("{name}: not a float tensor ({d:?})"),
        };
        Ok(if via_bf16 { v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect() } else { v })
    }

    /// Tensor converted to bf16 little-endian bytes.
    pub fn bf16_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.packed_len(name, Dt::BF16)?];
        self.bf16_into(name, &mut out)?;
        Ok(out)
    }

    /// [`Self::bf16_bytes`] written straight to `dst`, with no vector of floats in between: whole models load
    /// through here.
    pub fn bf16_into(&self, name: &str, dst: &mut [u8]) -> Result<()> {
        let i = self.info(name)?;
        let b = self.bytes(name)?;
        if dst.len() != i.numel() * 2 {
            bail!("{name}: {} bytes for {} bf16 values", dst.len(), i.numel());
        }
        match i.dtype {
            Dt::BF16 => dst.copy_from_slice(b),
            Dt::F32 => {
                for (o, c) in dst.chunks_exact_mut(2).zip(b.chunks_exact(4)) {
                    o.copy_from_slice(&bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes());
                }
            }
            Dt::F16 => {
                for (o, c) in dst.chunks_exact_mut(2).zip(b.chunks_exact(2)) {
                    o.copy_from_slice(&bf16::from_f32(f16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes());
                }
            }
            d => bail!("{name}: not a float tensor ({d:?})"),
        }
        Ok(())
    }

    /// Byte length of a tensor once packed in `dt`.
    pub fn packed_len(&self, name: &str, dt: Dt) -> Result<usize> {
        Ok(self.info(name)?.numel() * dt.size())
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 4];
    for (o, x) in out.chunks_exact_mut(4).zip(v) {
        o.copy_from_slice(&x.to_le_bytes());
    }
    out
}

/// `v` rounded to bf16, as little-endian bytes for an upload.
pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 2];
    for (o, x) in out.chunks_exact_mut(2).zip(v) {
        o.copy_from_slice(&bf16::from_f32(*x).to_le_bytes());
    }
    out
}

/// `v` rounded to f16, as little-endian bytes for an upload.
pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 2];
    for (o, x) in out.chunks_exact_mut(2).zip(v) {
        o.copy_from_slice(&f16::from_f32(*x).to_le_bytes());
    }
    out
}
