//! Thin CUDA runtime bindings (statically linked cudart) and RAII wrappers.
#![allow(non_camel_case_types)]

use anyhow::{bail, Result};
use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int};

pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;

const H2D: c_int = 1;
const D2H: c_int = 2;
const D2D: c_int = 3;

extern "C" {
    fn cudaSetDevice(dev: c_int) -> c_int;
    fn cudaFree(p: *mut c_void) -> c_int;
    fn cudaMalloc(p: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMallocAsync(p: *mut *mut c_void, size: usize, s: cudaStream_t) -> c_int;
    fn cudaFreeAsync(p: *mut c_void, s: cudaStream_t) -> c_int;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int, s: cudaStream_t) -> c_int;
    fn cudaMemcpy2DAsync(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize,
        height: usize, kind: c_int, s: cudaStream_t) -> c_int;
    fn cudaHostAlloc(p: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    fn cudaFreeHost(p: *mut c_void) -> c_int;
    fn cudaStreamCreateWithFlags(s: *mut cudaStream_t, flags: u32) -> c_int;
    fn cudaStreamDestroy(s: cudaStream_t) -> c_int;
    fn cudaStreamSynchronize(s: cudaStream_t) -> c_int;
    fn cudaStreamWaitEvent(s: cudaStream_t, e: cudaEvent_t, flags: u32) -> c_int;
    fn cudaEventCreateWithFlags(e: *mut cudaEvent_t, flags: u32) -> c_int;
    fn cudaEventDestroy(e: cudaEvent_t) -> c_int;
    fn cudaEventRecord(e: cudaEvent_t, s: cudaStream_t) -> c_int;
    fn cudaEventSynchronize(e: cudaEvent_t) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, a: cudaEvent_t, b: cudaEvent_t) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaGetErrorString(e: c_int) -> *const c_char;
    fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, dev: c_int) -> c_int;
    fn cudaMemPoolSetAttribute(pool: *mut c_void, attr: c_int, value: *mut c_void) -> c_int;
    fn cudaMemPoolTrimTo(pool: *mut c_void, keep: usize) -> c_int;
    fn cudaDeviceGetPCIBusId(id: *mut c_char, len: c_int, dev: c_int) -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
}

/// cudaErrorMemoryAllocation
pub const OOM: c_int = 2;

/// Device memory held by live [`DevBuf`]s: what comfy-aimdo calls its recorded usage.
static USAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Bytes of device memory held by live [`DevBuf`]s.
pub fn usage() -> usize {
    USAGE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Wait for all work on the device, every stream included.
pub fn device_sync() -> Result<()> {
    check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

pub fn err_str(e: i32) -> String {
    unsafe {
        let p = cudaGetErrorString(e);
        if p.is_null() {
            format!("cuda error {e}")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[inline]
pub fn check(e: i32, what: &str) -> Result<()> {
    if e != 0 {
        bail!("{what}: {} ({e})", if e > 0 { err_str(e) } else { "kernel launcher error".to_string() });
    }
    Ok(())
}

macro_rules! ck {
    ($e:expr) => {
        $crate::cuda::check(unsafe { $e }, stringify!($e))
    };
}

pub fn init(device: i32) -> Result<()> {
    ck!(cudaSetDevice(device))?;
    // Keep freed stream-ordered allocations cached in the pool (fast reuse between steps).
    let mut pool = std::ptr::null_mut();
    ck!(cudaDeviceGetDefaultMemPool(&mut pool, device))?;
    let mut thresh: u64 = u64::MAX;
    const RELEASE_THRESHOLD: c_int = 4;
    ck!(cudaMemPoolSetAttribute(pool, RELEASE_THRESHOLD, &mut thresh as *mut u64 as *mut c_void))?;
    Ok(())
}

/// Return cached-but-unused stream-ordered allocations to the driver so free memory is accurate.
pub fn trim_pool() -> Result<()> {
    let mut pool = std::ptr::null_mut();
    ck!(cudaDeviceGetDefaultMemPool(&mut pool, 0))?;
    ck!(cudaMemPoolTrimTo(pool, 0))
}

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

#[link(name = "nvml")]
extern "C" {
    fn nvmlInit_v2() -> c_int;
    fn nvmlDeviceGetHandleByPciBusId_v2(id: *const c_char, dev: *mut *mut c_void) -> c_int;
    fn nvmlDeviceGetMemoryInfo(dev: *mut c_void, mem: *mut NvmlMemory) -> c_int;
}

/// NVML handle of CUDA device 0 (matched by PCI bus id), if NVML is usable.
fn nvml_device() -> Option<usize> {
    static DEV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *DEV.get_or_init(|| unsafe {
        if nvmlInit_v2() != 0 {
            return None;
        }
        let mut id = [0 as c_char; 64];
        if cudaDeviceGetPCIBusId(id.as_mut_ptr(), id.len() as c_int, 0) != 0 {
            return None;
        }
        let mut dev = std::ptr::null_mut();
        (nvmlDeviceGetHandleByPciBusId_v2(id.as_ptr(), &mut dev) == 0).then_some(dev as usize)
    })
}

extern "C" {
    fn gpu_wddm_budget(budget: *mut u64, usage: *mut u64) -> c_int;
}

/// VRAM this process can still use without the OS paging it out.
///
/// On Windows (WDDM) cudaMemGetInfo ignores other processes and NVML's device-wide figure counts
/// memory WDDM has already paged out, so the DXCore per-process budget is preferred (as ComfyUI's
/// aimdo does): budget - current usage. Falls back to NVML, then CUDA; never above CUDA's figure.
pub fn device_free() -> Result<usize> {
    let (cuda_free, _) = mem_info()?;
    let (mut budget, mut usage) = (0u64, 0u64);
    static LOGGED: std::sync::Once = std::sync::Once::new();
    if unsafe { gpu_wddm_budget(&mut budget, &mut usage) } == 0 && budget > 0 {
        LOGGED.call_once(|| eprintln!("VRAM accounting: DXCore process budget ({} MB budget, {} MB in use)",
            budget >> 20, usage >> 20));
        return Ok(cuda_free.min(budget.saturating_sub(usage) as usize));
    }
    LOGGED.call_once(|| eprintln!("VRAM accounting: {}", if nvml_device().is_some() { "NVML" } else { "CUDA" }));
    if let Some(free) = nvml_free() {
        return Ok(cuda_free.min(free));
    }
    Ok(cuda_free)
}

/// Device-wide free memory from NVML, if NVML is usable.
pub fn nvml_free() -> Option<usize> {
    let dev = nvml_device()?;
    let mut m = NvmlMemory { total: 0, free: 0, used: 0 };
    (unsafe { nvmlDeviceGetMemoryInfo(dev as *mut c_void, &mut m) } == 0).then_some(m.free as usize)
}

/// This process's DXCore local memory budget, if DXCore answers.
pub fn wddm_budget() -> Option<usize> {
    let (mut budget, mut usage) = (0u64, 0u64);
    (unsafe { gpu_wddm_budget(&mut budget, &mut usage) } == 0 && budget > 0).then_some(budget as usize)
}

pub fn mem_info() -> Result<(usize, usize)> {
    let (mut f, mut t) = (0usize, 0usize);
    ck!(cudaMemGetInfo(&mut f, &mut t))?;
    Ok((f, t))
}

// ---------------------------------------------------------------------------------------------

pub struct Stream(pub cudaStream_t);
unsafe impl Send for Stream {}
impl Stream {
    pub fn new() -> Result<Self> {
        let mut s = std::ptr::null_mut();
        ck!(cudaStreamCreateWithFlags(&mut s, 1))?; // non-blocking
        Ok(Stream(s))
    }
    pub fn sync(&self) -> Result<()> {
        ck!(cudaStreamSynchronize(self.0))
    }
    pub fn wait(&self, e: &Event) -> Result<()> {
        ck!(cudaStreamWaitEvent(self.0, e.0, 0))
    }
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.0) };
    }
}

pub struct Event(pub cudaEvent_t);
unsafe impl Send for Event {}
unsafe impl Sync for Event {}
impl Event {
    /// Synchronization-only event (no timing).
    pub fn new() -> Result<Self> {
        let mut e = std::ptr::null_mut();
        ck!(cudaEventCreateWithFlags(&mut e, 2))?; // disable timing
        Ok(Event(e))
    }
    /// A second event recorded right after `first`, for another thread to wait on.
    pub fn new_like(_first: &Event, s: &Stream) -> Result<Self> {
        let e = Self::new()?;
        e.record(s)?;
        Ok(e)
    }

    pub fn timing() -> Result<Self> {
        let mut e = std::ptr::null_mut();
        ck!(cudaEventCreateWithFlags(&mut e, 0))?;
        Ok(Event(e))
    }
    pub fn record(&self, s: &Stream) -> Result<()> {
        ck!(cudaEventRecord(self.0, s.0))
    }
    pub fn sync(&self) -> Result<()> {
        ck!(cudaEventSynchronize(self.0))
    }
    pub fn elapsed_ms(&self, since: &Event) -> Result<f32> {
        let mut ms = 0f32;
        ck!(cudaEventElapsedTime(&mut ms, since.0, self.0))?;
        Ok(ms)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0) };
    }
}

// ---------------------------------------------------------------------------------------------

/// Device memory. `stream` is null for cudaMalloc'd (long-lived) buffers, otherwise the stream the
/// stream-ordered allocation is freed on.
pub struct DevBuf {
    ptr: *mut c_void,
    size: usize,
    stream: cudaStream_t,
}
unsafe impl Send for DevBuf {}

impl DevBuf {
    pub fn alloc(size: usize) -> Result<Self> {
        if crate::comfy_memory::enabled() {
            return crate::comfy_memory::alloc(size, None);
        }
        Self::alloc_raw(size, None)
    }
    pub fn alloc_async(size: usize, s: &Stream) -> Result<Self> {
        if crate::comfy_memory::enabled() {
            return crate::comfy_memory::alloc(size, Some(s));
        }
        Self::alloc_raw(size, Some(s))
    }
    /// cudaMalloc, or cudaMallocAsync on `s`, with no memory policy in between.
    pub(crate) fn alloc_raw(size: usize, s: Option<&Stream>) -> Result<Self> {
        Self::try_alloc_raw(size, s).map_err(|e| anyhow::anyhow!("{} of {} MB: {} ({e})",
            if s.is_some() { "cudaMallocAsync" } else { "cudaMalloc" }, size >> 20, err_str(e)))
    }
    /// As [`Self::alloc_raw`], returning the CUDA error so an out-of-memory can be retried.
    pub(crate) fn try_alloc_raw(size: usize, s: Option<&Stream>) -> std::result::Result<Self, c_int> {
        let mut p = std::ptr::null_mut();
        let stream = s.map_or(std::ptr::null_mut(), |s| s.0);
        let e = unsafe { if s.is_some() { cudaMallocAsync(&mut p, size.max(1), stream) } else { cudaMalloc(&mut p, size.max(1)) } };
        if e != 0 {
            unsafe { cudaGetLastError() };
            return Err(e);
        }
        USAGE.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        Ok(DevBuf { ptr: p, size, stream })
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn at(&self, byte_off: usize) -> *mut c_void {
        debug_assert!(byte_off <= self.size);
        unsafe { (self.ptr as *mut u8).add(byte_off) as *mut c_void }
    }
    pub fn size(&self) -> usize {
        self.size
    }
    /// `src` may be freed on return. CUDA buffers small copies itself, but a larger copy from
    /// pageable memory can still be pending behind earlier work, so that case waits for it.
    pub fn upload(&self, off: usize, src: &[u8], s: &Stream) -> Result<()> {
        assert!(off + src.len() <= self.size);
        ck!(cudaMemcpyAsync(self.at(off), src.as_ptr() as *const c_void, src.len(), H2D, s.0))?;
        if src.len() > 64 << 10 {
            s.sync()?;
        }
        Ok(())
    }
    pub fn download(&self, off: usize, dst: &mut [u8], s: &Stream) -> Result<()> {
        assert!(off + dst.len() <= self.size);
        ck!(cudaMemcpyAsync(dst.as_mut_ptr() as *mut c_void, self.at(off), dst.len(), D2H, s.0))?;
        s.sync()
    }
}
impl Drop for DevBuf {
    fn drop(&mut self) {
        USAGE.fetch_sub(self.size, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            if self.stream.is_null() {
                cudaFree(self.ptr);
            } else {
                cudaFreeAsync(self.ptr, self.stream);
            }
        }
    }
}

pub fn memcpy_h2d(dst: *mut c_void, src: *const u8, n: usize, s: &Stream) -> Result<()> {
    ck!(cudaMemcpyAsync(dst, src as *const c_void, n, H2D, s.0))
}
pub fn memcpy_d2d(dst: *mut c_void, src: *const c_void, n: usize, s: &Stream) -> Result<()> {
    ck!(cudaMemcpyAsync(dst, src, n, D2D, s.0))
}
pub fn memcpy_2d_d2d(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize,
    height: usize, s: &Stream) -> Result<()> {
    ck!(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, D2D, s.0))
}

/// Page-locked host memory (DMA source for weight streaming).
pub struct PinnedBuf {
    ptr: *mut u8,
    size: usize,
}
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}
impl PinnedBuf {
    pub fn alloc(size: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        ck!(cudaHostAlloc(&mut p, size.max(1), 0))?;
        Ok(PinnedBuf { ptr: p as *mut u8, size })
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn size(&self) -> usize {
        self.size
    }
}
impl Drop for PinnedBuf {
    fn drop(&mut self) {
        unsafe { cudaFreeHost(self.ptr as *mut c_void) };
    }
}
