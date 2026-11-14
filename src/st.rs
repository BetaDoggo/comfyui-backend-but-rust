//! Read-only safetensors access over a memory map (files are never copied or modified).

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

#[link(name = "kernel32")]
extern "system" {
    fn VirtualUnlock(addr: *const std::ffi::c_void, size: usize) -> i32;
}

#[link(name = "ntdll")]
extern "system" {
    fn NtSetInformationThread(thread: isize, class: u32, info: *const std::ffi::c_void, len: u32) -> i32;
    #[cfg(test)]
    fn NtQueryInformationThread(thread: isize, class: u32, info: *mut std::ffi::c_void, len: u32, returned: *mut u32) -> i32;
}

/// The calling thread's reads from disk, page faults on a mapping included, wait for everyone else's from now on. A
/// model is tens of gigabytes, and read at normal priority from the drive that holds the pagefile it freezes the
/// desktop until it is in. With nothing else reading, it loads as fast.
pub fn read_in_background() {
    // ThreadIoPriority, IoPriorityVeryLow: what Windows gives its own background work
    const THREAD_IO_PRIORITY: u32 = 22;
    let very_low: u32 = 0;
    // -2 is the calling thread
    unsafe { NtSetInformationThread(-2, THREAD_IO_PRIORITY, (&very_low as *const u32).cast(), 4) };
}

/// A mapping is page-locked in pieces of this size ([`SafeTensors::pin`]).
const PIN_CHUNK: usize = 64 << 20;

/// Bring `len` mapped bytes at `ptr` into this process from the file, on the calling thread and at its I/O priority.
pub fn read_in(ptr: *const u8, len: usize) {
    let mut sum = 0u8;
    for o in (0..len).step_by(4096) {
        sum = sum.wrapping_add(unsafe { ptr.add(o).read_volatile() });
    }
    std::hint::black_box(sum);
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
    /// The `__metadata__` strings of the files, merged.
    pub metadata: HashMap<String, String>,
    /// The pieces of each mapping page-locked for the device, as (file, piece).
    pinned: Mutex<HashSet<(usize, usize)>>,
}

impl Drop for SafeTensors {
    fn drop(&mut self) {
        let held = self.pinned.get_mut().unwrap_or_else(|e| e.into_inner());
        if held.is_empty() {
            return;
        }
        // copies from the mapping must be over before its pages are let go
        let _ = crate::cuda::device_sync();
        for (file, piece) in held.drain() {
            crate::cuda::host_unregister(unsafe { self.mmaps[file].as_ptr().add(piece * PIN_CHUNK) });
        }
    }
}

impl SafeTensors {
    /// One safetensors file.
    pub fn open(path: &Path) -> Result<Self> {
        let mut st = SafeTensors { mmaps: Vec::new(), tensors: HashMap::new(), metadata: HashMap::new(), pinned: Default::default() };
        st.add_file(path)?;
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

    /// Add tensors held in RAM rather than in a file, each `(name, dtype, shape)`, replacing any of the same name once
    /// all are filled. `fill` writes each one's bytes in order and still sees the tensors it replaces.
    pub fn add_memory(&mut self, tensors: Vec<(String, Dt, Vec<usize>)>, mut fill: impl FnMut(&Self, &str, &mut [u8]) -> Result<()>)
        -> Result<()> {
        let mut infos = Vec::with_capacity(tensors.len());
        let mut end = 0usize;
        for (name, dtype, shape) in tensors {
            let start = end.next_multiple_of(256);
            let len = shape.iter().product::<usize>() * dtype.size();
            end = start + len;
            infos.push((name, TensorInfo { dtype, shape, file: self.mmaps.len(), start, len }));
        }
        let mut m = memmap2::MmapMut::map_anon(end.max(1))?;
        for (name, i) in &infos {
            fill(self, name, &mut m[i.start..i.start + i.len])?;
        }
        self.mmaps.push(m.make_read_only()?);
        self.tensors.extend(infos);
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
            // page-locked pieces are held for the device
            let held = self.pinned.lock().unwrap();
            if i.len == 0 || (i.start / PIN_CHUNK..=(i.start + i.len - 1) / PIN_CHUNK).any(|c| held.contains(&(i.file, c))) {
                return;
            }
            drop(held);
            let m = &self.mmaps[i.file];
            let (start, end) = (i.start & !4095, (i.start + i.len).next_multiple_of(4096).min(m.len()));
            // on pages that aren't locked this reports failure and does what we want, as documented
            unsafe { VirtualUnlock(m[start..end].as_ptr().cast(), end - start) };
        }
    }

    /// Page-lock a tensor's bytes where they are in the mapping, so that copies of them to the device need no page-locked
    /// copy in RAM: the file cache's pages are the only ones. They stay locked until the file is dropped. Returns the
    /// tensor as the runs of bytes to copy, each (address, bytes): a copy can't cross from one locked piece into the
    /// next. None when the driver refuses, which leaves what was locked before as it was.
    pub fn pin(&self, name: &str) -> Option<Vec<(usize, usize)>> {
        let i = self.info(name).ok()?;
        if i.len == 0 {
            return Some(Vec::new());
        }
        let m = &self.mmaps[i.file];
        // the last page is mapped whole
        let end = m.len().next_multiple_of(4096);
        let mut held = self.pinned.lock().unwrap();
        for c in i.start / PIN_CHUNK..=(i.start + i.len - 1) / PIN_CHUNK {
            if held.contains(&(i.file, c)) {
                continue;
            }
            let (a, b) = (c * PIN_CHUNK, ((c + 1) * PIN_CHUNK).min(end));
            if crate::cuda::host_register(unsafe { m.as_ptr().add(a) }, b - a, true).is_err() {
                return None;
            }
            held.insert((i.file, c));
        }
        let (mut runs, mut at, end) = (Vec::new(), i.start, i.start + i.len);
        while at < end {
            let stop = ((at / PIN_CHUNK + 1) * PIN_CHUNK).min(end);
            runs.push((m.as_ptr() as usize + at, stop - at));
            at = stop;
        }
        Some(runs)
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

#[cfg(test)]
mod tests {
    #[test]
    fn a_reader_thread_gets_very_low_io_priority() {
        std::thread::spawn(|| {
            let io_priority = || {
                let (mut v, mut n) = (u32::MAX, 0u32);
                let status = unsafe { super::NtQueryInformationThread(-2, 22, (&mut v as *mut u32).cast(), 4, &mut n) };
                assert_eq!(status, 0);
                v
            };
            assert_eq!(io_priority(), 2, "a new thread reads at normal priority");
            super::read_in_background();
            assert_eq!(io_priority(), 0);
        })
        .join()
        .unwrap();
    }
}
