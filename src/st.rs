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
    pub start: usize,
    pub len: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    pub mmap: memmap2::Mmap,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n])?;
        let base = 8 + n;
        let mut tensors = HashMap::new();
        for (k, v) in header.as_object().ok_or_else(|| anyhow!("bad header"))? {
            if k == "__metadata__" {
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
            tensors.insert(k.clone(), TensorInfo { dtype, shape, start: base + a, len: b - a });
        }
        Ok(SafeTensors { mmap, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let i = self.info(name)?;
        Ok(&self.mmap[i.start..i.start + i.len])
    }

    /// Take a tensor's pages out of this process. They stay in the file cache, so a later read costs
    /// a soft fault per page rather than a disk read: not for tensors read again on every step.
    pub fn evict(&self, name: &str) {
        if let Ok(i) = self.info(name) {
            let (start, end) = (i.start & !4095, (i.start + i.len).next_multiple_of(4096).min(self.mmap.len()));
            // on pages that aren't locked this reports failure and does what we want, as documented
            unsafe { VirtualUnlock(self.mmap[start..end].as_ptr().cast(), end - start) };
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
        let i = self.info(name)?;
        if i.dtype == Dt::BF16 {
            return Ok(self.bytes(name)?.to_vec());
        }
        let v = self.f32_vec(name, false)?;
        Ok(v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect())
    }

    /// Byte length of a tensor once packed in `dt`.
    pub fn packed_len(&self, name: &str, dt: Dt) -> Result<usize> {
        Ok(self.info(name)?.numel() * dt.size())
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
