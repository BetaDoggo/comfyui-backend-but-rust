//! ComfyUI's memory management, the default (`--legacy-offload` turns it off for the older planner).
//!
//! SwarmUI's ComfyUI runs DynamicVRAM: `ModelPatcherDynamic` over comfy-aimdo 0.6.0. This module follows aimdo's
//! `model-vbar.c`, its budget in `plat.h` and `shmem-detect.c`, and its allocator hooks, with one of our streamed units
//! (a block) in place of aimdo's 32 MB page. The policy, as ComfyUI has it:
//!
//! * No activation estimates. Every streamed model ([`crate::offload::WeightSet`]) has a list of units in run order,
//!   and units are faulted into VRAM on first use while the budget allows. A model is "prioritized" when it is loaded
//!   for a node (`load_models_gpu`): it moves to the top of the priority list and its watermark goes back to the top.
//! * Units at or above a model's watermark are not faulted: they go through one of the cast buffers (one per offload
//!   stream, two by default), copied before each use. The watermark only falls until the model is prioritized again.
//! * Every device allocation first frees the budget deficit, and a failed one frees its own size and retries. Freeing
//!   lowers watermarks from the top, the lowest-priority model first and the running model last, so activations take
//!   VRAM from weights, and a model that ran earlier (a text encoder) gives up its weights before the running one.
//! * The budget: aimdo's recorded usage (live device buffers) plus the request plus the headroom (256 MB, or ComfyUI's
//!   `--reserve-vram`) against the card's capacity, and, polled every two seconds, the DXCore process budget less
//!   512 MB and NVML's free memory less 512 MB, whichever is shortest.
//!
//! What differs from ComfyUI: residency moves by unit, not by 32 MB page; the small weights outside the units, the
//! VAEs and encoded prompts are plain allocations (ComfyUI force-loads tensors under 16 KB and streams the rest of
//! those models too); one copy stream carries both cast buffers' copies; and CFG's two halves still run one after the
//! other, where ComfyUI batches them when they fit.

use crate::cuda::{self, DevBuf, Event, Stream, OOM};
use anyhow::{bail, Context, Result};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

const MB: usize = 1 << 20;
/// aimdo's `VRAM_HEADROOM`, kept free when ComfyUI isn't given `--reserve-vram`.
pub const DEFAULT_HEADROOM_MB: usize = 256;
/// aimdo's headrooms on Windows: under the DXCore budget, under NVML's free memory, under CUDA's.
const WDDM_BUDGET_HEADROOM: isize = 512 << 20;
const NVML_BUDGET_HEADROOM: isize = 512 << 20;
const CUDA_BUDGET_HEADROOM: isize = 96 << 20;
/// Extra room asked for around a plain `cudaMalloc` (aimdo's `CUDA_MALLOC_HEADROOM`).
const CUDA_MALLOC_HEADROOM: usize = 128 << 20;
/// How often the driver is asked about memory; between polls the recorded usage carries the deficit.
const POLL_MS: u128 = 2000;

static ENABLED: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);

/// True once [`enable`] has run: every device allocation and streamed model follows ComfyUI's policy.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Switch to ComfyUI's memory management for the rest of the process. `headroom` is aimdo's simple headroom, and
/// `cast_buffers` ComfyUI's offload stream count (`--async-offload`, 2 on NVIDIA). Call before any model loads.
pub fn enable(headroom: usize, cast_buffers: usize) -> Result<()> {
    if cast_buffers == 0 {
        bail!("comfy memory mode needs at least one cast buffer");
    }
    if let Some(st) = lock().as_mut() {
        // a second generator in this process keeps the models already registered
        st.headroom = headroom;
        return Ok(());
    }
    let (_, capacity) = cuda::mem_info()?;
    let casts = (0..cast_buffers).map(|_| Ok(Cast { buf: None, free: Event::new()?, busy: false })).collect::<Result<Vec<_>>>()?;
    *lock() = Some(State {
        capacity,
        headroom,
        last_poll: None,
        usage_at_check: 0,
        deficit_sync: 0,
        vbars: Vec::new(),
        active: None,
        next_id: 1,
        casts,
        next_cast: 0,
        debug: std::env::var_os("DEBUG_VRAM").is_some(),
    });
    ENABLED.store(true, Ordering::Relaxed);
    eprintln!("comfy memory mode: ComfyUI DynamicVRAM policy, {} MB capacity, {} MB headroom, {cast_buffers} cast buffers",
        capacity / MB, headroom / MB);
    Ok(())
}

fn lock() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn with<T>(f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
    let mut g = lock();
    f(g.as_mut().context("comfy memory mode is not enabled")?)
}

struct Unit {
    size: usize,
    buf: Option<DevBuf>,
    /// Uses in flight: a pinned unit is never freed, only marked to go when unpinned.
    pins: u32,
    /// The buffer holds the unit's data (aimdo's matching signature).
    valid: bool,
}

/// One model's units, as aimdo's ModelVBAR holds its pages.
struct Vbar {
    id: u64,
    name: String,
    units: Vec<Unit>,
    /// Units at or above this index are streamed until the model is prioritized again.
    watermark: usize,
    /// Faulted or streamed since it was last prioritized, for the residency report.
    used: bool,
}

impl Vbar {
    fn resident_bytes(&self) -> usize {
        self.units.iter().filter(|u| u.buf.is_some()).map(|u| u.size).sum()
    }

    /// The first unit from `c` that isn't in VRAM (aimdo's `move_cursor_to_absent`).
    fn absent(&self, mut c: usize) -> usize {
        while c < self.watermark && self.units[c].buf.is_some() {
            c += 1;
        }
        c
    }

    /// Pay for the absent units below `target` out of `surplus` (aimdo's `spend_surplus_on_cursor`).
    fn spend(&self, target: usize, mut c: usize, surplus: &mut isize) -> usize {
        while c < target && c < self.watermark && *surplus >= self.units[c].size as isize {
            *surplus -= self.units[c].size as isize;
            c = self.absent(c + 1);
        }
        c
    }
}

/// A cast buffer: streamed units are copied into it before each use.
struct Cast {
    buf: Option<DevBuf>,
    /// Recorded on the compute stream when its last unit is released.
    free: Event,
    busy: bool,
}

struct State {
    capacity: usize,
    headroom: usize,
    last_poll: Option<Instant>,
    usage_at_check: usize,
    deficit_sync: isize,
    /// Lowest priority first.
    vbars: Vec<Vbar>,
    /// The model prioritized last.
    active: Option<u64>,
    next_id: u64,
    casts: Vec<Cast>,
    next_cast: usize,
    debug: bool,
}

impl State {
    fn index(&self, id: u64) -> Result<usize> {
        self.vbars.iter().position(|v| v.id == id).context("streamed model is not registered")
    }

    /// Ask the driver, at most every two seconds (aimdo's `poll_budget_deficit` on Windows).
    fn poll(&mut self) {
        if self.last_poll.is_some_and(|t| t.elapsed().as_millis() < POLL_MS) {
            return;
        }
        self.last_poll = Some(Instant::now());
        let usage = cuda::usage();
        self.usage_at_check = usage;
        let budget = cuda::wddm_budget().unwrap_or(self.capacity);
        let mut d = usage as isize + WDDM_BUDGET_HEADROOM - budget as isize;
        let device = match cuda::nvml_free() {
            Some(free) => Some(NVML_BUDGET_HEADROOM - free as isize),
            None => cuda::mem_info().ok().map(|(free, _)| CUDA_BUDGET_HEADROOM - free as isize),
        };
        if let Some(dc) = device {
            d = d.max(dc);
        }
        self.deficit_sync = d;
    }

    /// Bytes to free before `size` more can be allocated (aimdo's `budget_deficit`).
    fn deficit(&mut self, size: usize) -> isize {
        self.poll();
        let usage = cuda::usage() as isize;
        let simple = usage + size as isize + self.headroom as isize - self.capacity as isize;
        let delta = self.deficit_sync + usage - self.usage_at_check as isize + size as isize;
        simple.max(delta)
    }

    /// Free `vbars[i].units[u]` unless it is pinned; returns the bytes freed. The device must be idle.
    fn evict(&mut self, i: usize, u: usize) -> usize {
        let v = &mut self.vbars[i];
        let unit = &mut v.units[u];
        if unit.buf.is_none() || unit.pins > 0 {
            return 0;
        }
        unit.buf = None;
        unit.valid = false;
        if self.debug {
            eprintln!("comfy memory: [{}] unit {u} out of VRAM ({} MB)", v.name, unit.size / MB);
        }
        unit.size
    }

    /// Free at least `size` bytes, lowering watermarks from the lowest-priority model up (aimdo's `vbars_free`).
    fn free(&mut self, size: isize) -> Result<()> {
        if size <= 0 {
            return Ok(());
        }
        let mut needed = size as usize;
        let mut synced = false;
        for i in 0..self.vbars.len() {
            while needed > 0 && self.vbars[i].watermark > 0 {
                if !synced {
                    cuda::device_sync()?;
                    synced = true;
                }
                let w = self.vbars[i].watermark - 1;
                needed = needed.saturating_sub(self.evict(i, w));
                self.vbars[i].watermark = w;
            }
            if needed == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Make room for `vbars[mv]`'s absent units below `target` (aimdo's `vbars_free_for_vbar`).
    fn free_for_vbar(&mut self, mv: usize, target: usize, mut surplus: isize) -> Result<()> {
        let mut cursor = self.vbars[mv].absent(0);
        cursor = self.vbars[mv].spend(target, cursor, &mut surplus);
        let mut synced = false;
        for i in 0..self.vbars.len() {
            while ((cursor < target && cursor < self.vbars[mv].watermark) || surplus < 0) && self.vbars[i].watermark > 0 {
                let w = self.vbars[i].watermark - 1;
                let unit = &self.vbars[i].units[w];
                if !synced && unit.buf.is_some() && unit.pins == 0 {
                    cuda::device_sync()?;
                    synced = true;
                }
                let freed = self.evict(i, w);
                self.vbars[i].watermark = w;
                if freed > 0 {
                    surplus += freed as isize;
                    cursor = self.vbars[mv].spend(target, cursor, &mut surplus);
                }
            }
        }
        Ok(())
    }

    /// Bring `id`'s unit `u` into VRAM if the budget allows, pinned (aimdo's `vbar_fault`). None means stream it.
    fn fault(&mut self, id: u64, u: usize) -> Result<Option<(*mut c_void, bool)>> {
        let d = self.deficit(0);
        self.free(d)?;
        let mv = self.index(id)?;
        self.vbars[mv].used = true;
        if u >= self.vbars[mv].watermark {
            return Ok(None);
        }
        if self.vbars[mv].units[u].buf.is_none() {
            let size = self.vbars[mv].units[u].size;
            let surplus = self.capacity as isize - (cuda::usage() + size) as isize - self.headroom as isize;
            self.free_for_vbar(mv, u + 1, surplus)?;
            if u >= self.vbars[mv].watermark {
                return Ok(None);
            }
            let mut buf = None;
            if self.deficit(size) <= 0 {
                match DevBuf::try_alloc_raw(size, None) {
                    Ok(b) => buf = Some(b),
                    Err(OOM) => {}
                    Err(e) => bail!("VRAM for {} unit {u}: {}", self.vbars[mv].name, cuda::err_str(e)),
                }
            }
            let buf = match buf {
                Some(b) => b,
                None => {
                    self.free(size as isize)?;
                    if u >= self.vbars[mv].watermark {
                        return Ok(None);
                    }
                    DevBuf::try_alloc_raw(size, None).map_err(|e| anyhow::anyhow!("VRAM for {} unit {u} failed: {}",
                        self.vbars[mv].name, cuda::err_str(e)))?
                }
            };
            let unit = &mut self.vbars[mv].units[u];
            unit.buf = Some(buf);
            unit.valid = false;
        }
        let unit = &mut self.vbars[mv].units[u];
        unit.pins += 1;
        Ok(Some((unit.buf.as_ref().unwrap().ptr(), unit.valid)))
    }

    /// A device allocation through aimdo's hook: free the deficit, and on failure free the size and retry.
    fn alloc(&mut self, size: usize, extra: usize, s: Option<&Stream>) -> Result<DevBuf> {
        let d = self.deficit(size + extra);
        self.free(d)?;
        match DevBuf::try_alloc_raw(size, s) {
            Ok(b) => Ok(b),
            Err(OOM) => {
                self.free((size + extra) as isize)?;
                DevBuf::alloc_raw(size, s)
            }
            Err(e) => bail!("allocating {} MB: {}", size / MB, cuda::err_str(e)),
        }
    }

    /// Release the cast buffers between nodes (ComfyUI's `reset_cast_buffers`), unless one is in use.
    fn reset_casts(&mut self) -> Result<()> {
        if self.casts.iter().any(|c| c.busy) || self.casts.iter().all(|c| c.buf.is_none()) {
            return Ok(());
        }
        cuda::device_sync()?;
        for c in &mut self.casts {
            c.buf = None;
        }
        self.next_cast = 0;
        cuda::trim_pool()
    }

    /// Move `id` to the top and reset its watermark (aimdo's `vbar_prioritize`, from `load_models_gpu`).
    fn prioritize(&mut self, id: u64) -> Result<()> {
        let i = self.index(id)?;
        let mut v = self.vbars.remove(i);
        if v.used {
            report(&v);
        }
        v.watermark = v.units.len();
        v.used = false;
        let staged: usize = v.units.iter().map(|u| u.size).sum();
        eprintln!("[{}] prepared for dynamic VRAM loading: {} MB staged, {} MB in VRAM", v.name, staged / MB, v.resident_bytes() / MB);
        self.vbars.push(v);
        if self.active != Some(id) {
            self.active = Some(id);
            self.reset_casts()?;
        }
        Ok(())
    }
}

fn report(v: &Vbar) {
    let resident = v.units.iter().filter(|u| u.buf.is_some()).count();
    eprintln!("[{}] {resident} of {} units in VRAM ({} MB), {} streamed", v.name, v.units.len(), v.resident_bytes() / MB,
        v.units.len() - v.watermark.min(v.units.len()));
}

/// A device allocation under ComfyUI's policy: `cudaMallocAsync` on `s`, else `cudaMalloc`.
pub fn alloc(size: usize, s: Option<&Stream>) -> Result<DevBuf> {
    with(|st| st.alloc(size, if s.is_none() { CUDA_MALLOC_HEADROOM } else { 0 }, s))
}

/// Add a model with units of `sizes` bytes, in run order, at the top of the priority list (aimdo's `vbar_allocate`).
pub fn register(name: &str, sizes: Vec<usize>) -> Result<u64> {
    with(|st| {
        let id = st.next_id;
        st.next_id += 1;
        let units: Vec<Unit> = sizes.into_iter().map(|size| Unit { size, buf: None, pins: 0, valid: false }).collect();
        st.vbars.push(Vbar { id, name: name.to_string(), watermark: units.len(), units, used: false });
        Ok(id)
    })
}

/// Drop a model and the VRAM it holds.
pub fn unregister(id: u64) {
    let mut g = lock();
    let Some(st) = g.as_mut() else { return };
    let _ = cuda::device_sync();
    st.vbars.retain(|v| v.id != id);
    if st.active == Some(id) {
        st.active = None;
    }
}

/// The model is loaded for a node: it gets the top priority and may fault every unit in again.
pub fn prioritize(id: u64) -> Result<()> {
    with(|st| st.prioritize(id))
}

/// Prioritize `id` unless it was the last model prioritized. A model that starts running after another is being
/// loaded for its node.
pub fn ensure_active(id: u64) -> Result<()> {
    with(|st| if st.active == Some(id) { Ok(()) } else { st.prioritize(id) })
}

/// Unit `u` of `id`, pinned in VRAM with whether it holds its data, or None to stream it.
pub fn fault(id: u64, u: usize) -> Result<Option<(*mut c_void, bool)>> {
    with(|st| st.fault(id, u))
}

/// The unit's buffer now holds its data (the copy is on its way, ahead of any use on the compute stream).
pub fn mark_valid(id: u64, u: usize) -> Result<()> {
    with(|st| {
        let i = st.index(id)?;
        st.vbars[i].units[u].valid = true;
        Ok(())
    })
}

/// Unpin a faulted unit; one above the watermark leaves VRAM (aimdo's `vbar_unpin`).
pub fn unpin(id: u64, u: usize) -> Result<()> {
    with(|st| {
        let i = st.index(id)?;
        let above = u >= st.vbars[i].watermark;
        let unit = &mut st.vbars[i].units[u];
        unit.pins = unit.pins.saturating_sub(1);
        if above && unit.pins == 0 && unit.buf.is_some() {
            cuda::device_sync()?;
            st.evict(i, u);
        }
        Ok(())
    })
}

/// True when unit `u` of `id` is in VRAM with its data and below the watermark.
pub fn holds(id: u64, u: usize) -> bool {
    with(|st| {
        let v = &st.vbars[st.index(id)?];
        Ok(u < v.watermark && v.units[u].buf.is_some() && v.units[u].valid)
    })
    .unwrap_or(false)
}

/// Copy `len` bytes of `src` over unit `u` at `off`, if it is in VRAM with its data. Issued on `s`.
pub fn write(id: u64, u: usize, off: usize, src: &DevBuf, len: usize, s: &Stream) -> Result<()> {
    with(|st| {
        let unit = &st.vbars[st.index(id)?].units[u];
        if let (Some(b), true) = (&unit.buf, unit.valid) {
            cuda::memcpy_d2d(b.at(off), src.ptr(), len, s)?;
        }
        Ok(())
    })
}

/// A cast buffer of at least `size` bytes for a streamed unit, with `copy` made to wait for its last reader. Returns
/// its index, for [`cast_release`], and its address.
pub fn cast_take(size: usize, copy: &Stream) -> Result<(usize, *mut c_void)> {
    with(|st| {
        let n = st.casts.len();
        let idx = (0..n).map(|k| (st.next_cast + k) % n).find(|&i| !st.casts[i].busy)
            .context("every cast buffer is in use: too many streamed units held at once")?;
        st.next_cast = (idx + 1) % n;
        if st.casts[idx].buf.as_ref().is_none_or(|b| b.size() < size) {
            // grows like aimdo's VRAMBuffer, through the budget
            st.casts[idx].free.sync()?;
            st.casts[idx].buf = None;
            let b = st.alloc(size, 0, None)?;
            st.casts[idx].buf = Some(b);
        } else {
            copy.wait(&st.casts[idx].free)?;
        }
        st.casts[idx].busy = true;
        Ok((idx, st.casts[idx].buf.as_ref().unwrap().ptr()))
    })
}

/// The unit in cast buffer `idx` is done with once `compute` reaches this point.
pub fn cast_release(idx: usize, compute: &Stream) -> Result<()> {
    with(|st| {
        st.casts[idx].free.record(compute)?;
        st.casts[idx].busy = false;
        Ok(())
    })
}

/// Free cast buffer `idx` without a reader to wait for, after a failed run. The device is synchronized first.
pub fn cast_abandon(idx: usize) -> Result<()> {
    with(|st| {
        cuda::device_sync()?;
        st.casts[idx].busy = false;
        Ok(())
    })
}

/// VRAM held by every model's resident units: what ComfyUI adds to free memory as evictable (`vbars_analyze`).
pub fn resident_bytes() -> usize {
    with(|st| Ok(st.vbars.iter().map(Vbar::resident_bytes).sum())).unwrap_or(0)
}

/// VRAM held by `id`'s resident units, and their count.
pub fn resident(id: u64) -> (usize, usize) {
    with(|st| {
        let v = &st.vbars[st.index(id)?];
        Ok((v.units.iter().filter(|u| u.buf.is_some()).count(), v.resident_bytes()))
    })
    .unwrap_or((0, 0))
}

/// Between nodes: release the cast buffers and give cached pool memory back, as ComfyUI does after each node.
pub fn node_done() -> Result<()> {
    with(|st| st.reset_casts())
}

/// Print each model's residency since it was last prioritized.
pub fn log_residency() {
    if let Some(st) = lock().as_ref() {
        st.vbars.iter().filter(|v| v.used).for_each(report);
    }
}
