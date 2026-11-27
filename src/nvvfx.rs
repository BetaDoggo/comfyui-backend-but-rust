//! NVIDIA RTX Video Super Resolution, the `VideoSuperRes` effect of NVIDIA's Video Effects SDK, as the `nvidia-vfx`
//! Python package (`nvvfx`) and ComfyUI's `RTXVideoSuperResolution` node run it.
//!
//! The network ships compiled inside `nvngx_vsr.dll`, so there are no weights to port: this drives the SDK's C API
//! (`NVVideoEffects.dll` and `NVCVImage.dll`), loaded at run time from the folder that holds `nvngx_vsr.dll`. That
//! folder needs the SDK's runtime beside the model, all from the same `nvidia-vfx` release (its `nvvfx/libs`):
//! see [`LIBS`]. Nothing here links against them, and the libraries are NVIDIA's, under NVIDIA's license.
//!
//! The effect takes RGBA bytes on the GPU. Frames go in and out through `NvCVImage_Transfer` with the same scale
//! factors `nvvfx` uses, so the output matches the package's byte for byte. The effect keeps no state between
//! frames: a video is upscaled frame by frame.

use anyhow::{bail, Context, Result};
use std::ffi::{c_char, c_void, CStr};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The model: the file a user picks, as an upscale model.
pub const MODEL: &str = "nvngx_vsr.dll";

/// The SDK files that must sit beside [`MODEL`], loaded in this order. `nvvfx`'s other libraries (NPP, TrueHDR,
/// frame generation) aren't needed for super resolution.
pub const LIBS: [&str; 6] = ["cudart64_12.dll", "NVCVImage.dll", "nvngxruntime.dll", "NVVideoEffects.dll", "nvVFXVideoSuperRes.dll", MODEL];

/// True if `path` is the RTX Video Super Resolution model, by its file name.
pub fn is_model(path: &Path) -> bool {
    path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.eq_ignore_ascii_case(MODEL))
}

/// The effect's `QualityLevel`, with `nvvfx.QualityLevel`'s names and values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Mode {
    Bicubic = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Ultra = 4,
    DenoiseLow = 8,
    DenoiseMedium = 9,
    DenoiseHigh = 10,
    DenoiseUltra = 11,
    DeblurLow = 12,
    DeblurMedium = 13,
    DeblurHigh = 14,
    DeblurUltra = 15,
    HighbitrateLow = 16,
    HighbitrateMedium = 17,
    HighbitrateHigh = 18,
    HighbitrateUltra = 19,
    StreamingMedium = 21,
    StreamingUltra = 23,
}

impl Mode {
    pub const ALL: [(&'static str, Mode); 19] = [
        ("BICUBIC", Mode::Bicubic),
        ("LOW", Mode::Low),
        ("MEDIUM", Mode::Medium),
        ("HIGH", Mode::High),
        ("ULTRA", Mode::Ultra),
        ("DENOISE_LOW", Mode::DenoiseLow),
        ("DENOISE_MEDIUM", Mode::DenoiseMedium),
        ("DENOISE_HIGH", Mode::DenoiseHigh),
        ("DENOISE_ULTRA", Mode::DenoiseUltra),
        ("DEBLUR_LOW", Mode::DeblurLow),
        ("DEBLUR_MEDIUM", Mode::DeblurMedium),
        ("DEBLUR_HIGH", Mode::DeblurHigh),
        ("DEBLUR_ULTRA", Mode::DeblurUltra),
        ("HIGHBITRATE_LOW", Mode::HighbitrateLow),
        ("HIGHBITRATE_MEDIUM", Mode::HighbitrateMedium),
        ("HIGHBITRATE_HIGH", Mode::HighbitrateHigh),
        ("HIGHBITRATE_ULTRA", Mode::HighbitrateUltra),
        ("STREAMING_MEDIUM", Mode::StreamingMedium),
        ("STREAMING_ULTRA", Mode::StreamingUltra),
    ];

    /// By `nvvfx`'s name, ignoring case, with `-` for `_` (`ultra`, `deblur-high`).
    pub fn from_name(name: &str) -> Option<Mode> {
        let n = name.trim().to_ascii_uppercase().replace('-', "_");
        Self::ALL.iter().find(|(k, _)| *k == n).map(|(_, m)| *m)
    }

    pub fn name(self) -> &'static str {
        Self::ALL.iter().find(|(_, m)| *m == self).map(|(k, _)| *k).unwrap_or("?")
    }

    /// Denoise and deblur clean an image at its own size; the other modes upscale.
    pub fn same_size(self) -> bool {
        matches!(self as u32, 8..=15)
    }
}

/// `NvCVImage`, from the SDK's `nvCVImage.h`.
#[repr(C)]
struct NvCVImage {
    width: u32,
    height: u32,
    pitch: i32,
    pixel_format: i32,
    component_type: i32,
    pixel_bytes: u8,
    component_bytes: u8,
    num_components: u8,
    planar: u8,
    gpu_mem: u8,
    colorspace: u8,
    reserved: [u8; 2],
    pixels: *mut c_void,
    delete_ptr: *mut c_void,
    delete_proc: *mut c_void,
    buffer_bytes: u64,
}
const _: () = assert!(std::mem::size_of::<NvCVImage>() == 64);

impl NvCVImage {
    fn zeroed() -> Box<Self> {
        // SAFETY: plain data, all zero is the SDK's empty image
        Box::new(unsafe { std::mem::zeroed() })
    }
}

const NVCV_RGB: i32 = 4;
const NVCV_RGBA: i32 = 6;
const NVCV_U8: i32 = 1;
const NVCV_F32: i32 = 7;
const NVCV_CHUNKY: u32 = 0;
const NVCV_CPU: u32 = 0;
const NVCV_GPU: u32 = 1;

type Handle = *mut c_void;
type Stream = *mut c_void;
type Status = i32;

/// The SDK's entry points.
struct Api {
    create_effect: unsafe extern "C" fn(*const c_char, *mut Handle) -> Status,
    destroy_effect: unsafe extern "C" fn(Handle),
    set_image: unsafe extern "C" fn(Handle, *const c_char, *mut NvCVImage) -> Status,
    set_u32: unsafe extern "C" fn(Handle, *const c_char, u32) -> Status,
    set_f32: unsafe extern "C" fn(Handle, *const c_char, f32) -> Status,
    set_cuda_stream: unsafe extern "C" fn(Handle, *const c_char, Stream) -> Status,
    load: unsafe extern "C" fn(Handle) -> Status,
    run: unsafe extern "C" fn(Handle, i32) -> Status,
    stream_create: unsafe extern "C" fn(*mut Stream) -> Status,
    stream_destroy: unsafe extern "C" fn(Stream),
    stream_sync: unsafe extern "C" fn(Stream) -> Status,
    image_alloc: unsafe extern "C" fn(*mut NvCVImage, u32, u32, i32, i32, u32, u32, u32) -> Status,
    image_dealloc: unsafe extern "C" fn(*mut NvCVImage),
    image_init: unsafe extern "C" fn(*mut NvCVImage, u32, u32, i32, *mut c_void, i32, i32, u32, u32) -> Status,
    image_transfer: unsafe extern "C" fn(*const NvCVImage, *mut NvCVImage, f32, Stream, *mut NvCVImage) -> Status,
    error_string: unsafe extern "C" fn(Status) -> *const c_char,
}

impl Api {
    fn check(&self, status: Status, what: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: the SDK returns a static string for every code
        let msg = unsafe {
            let p = (self.error_string)(status);
            if p.is_null() { "unknown error".into() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        bail!("NVIDIA VFX {what} failed: {msg} ({status})")
    }
}

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn GetLastError() -> u32;
}
/// Resolve the library's own imports from its folder, then the system's.
const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x100;
const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000;

fn load_library(path: &Path) -> Result<*mut c_void> {
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: a null-terminated path
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) };
    if module.is_null() {
        // SAFETY: no arguments
        let code = unsafe { GetLastError() };
        bail!("loading {} failed (Windows error {code})", path.display());
    }
    Ok(module)
}

fn symbol<T: Copy>(module: *mut c_void, lib: &str, name: &str) -> Result<T> {
    let cname = std::ffi::CString::new(name)?;
    // SAFETY: a valid module and a null-terminated name
    let p = unsafe { GetProcAddress(module, cname.as_ptr()) };
    if p.is_null() {
        bail!("{lib} has no {name}: is it from a different VFX SDK release than the other files?");
    }
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: T is the function pointer type of `name`, from the SDK's headers
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

/// The SDK, loaded once per process (its libraries are never unloaded), and the folder it came from.
static API: Mutex<Option<(PathBuf, &'static Api)>> = Mutex::new(None);

/// The SDK from `dir`, loading it on first use.
fn api(dir: &Path) -> Result<&'static Api> {
    let mut slot = API.lock().unwrap_or_else(|e| e.into_inner());
    // Not `canonicalize`: its `\\?\` paths stop NVVideoEffects.dll finding the feature libraries beside it.
    let dir = std::path::absolute(dir).with_context(|| format!("NVIDIA VFX folder {}", dir.display()))?;
    if let Some((loaded, api)) = slot.as_ref() {
        if *loaded == dir {
            return Ok(api);
        }
        bail!(
            "NVIDIA's VFX SDK is already loaded from {}, and a process can only hold one copy: restart to use the one in {}",
            loaded.display(),
            dir.display()
        );
    }
    let missing: Vec<&str> = LIBS.iter().copied().filter(|l| !dir.join(l).is_file()).collect();
    if !missing.is_empty() {
        bail!(
            "{} needs NVIDIA's VFX runtime beside it, from the nvvfx/libs folder of the same nvidia-vfx release; missing from {}: {}",
            MODEL,
            dir.display(),
            missing.join(", ")
        );
    }
    let mut modules = Vec::new();
    for lib in LIBS {
        modules.push(load_library(&dir.join(lib))?);
    }
    let (vfx, cv) = (modules[3], modules[1]);
    let (v, c) = ("NVVideoEffects.dll", "NVCVImage.dll");
    let api = Api {
        create_effect: symbol(vfx, v, "NvVFX_CreateEffect")?,
        destroy_effect: symbol(vfx, v, "NvVFX_DestroyEffect")?,
        set_image: symbol(vfx, v, "NvVFX_SetImage")?,
        set_u32: symbol(vfx, v, "NvVFX_SetU32")?,
        set_f32: symbol(vfx, v, "NvVFX_SetF32")?,
        set_cuda_stream: symbol(vfx, v, "NvVFX_SetCudaStream")?,
        load: symbol(vfx, v, "NvVFX_Load")?,
        run: symbol(vfx, v, "NvVFX_Run")?,
        stream_create: symbol(vfx, v, "NvVFX_CudaStreamCreate")?,
        stream_destroy: symbol(vfx, v, "NvVFX_CudaStreamDestroy")?,
        stream_sync: symbol(vfx, v, "NvVFX_CudaStreamSynchronize")?,
        image_alloc: symbol(cv, c, "NvCVImage_Alloc")?,
        image_dealloc: symbol(cv, c, "NvCVImage_Dealloc")?,
        image_init: symbol(cv, c, "NvCVImage_Init")?,
        image_transfer: symbol(cv, c, "NvCVImage_Transfer")?,
        error_string: symbol(cv, c, "NvCV_GetErrorStringFromCode")?,
    };
    let api: &'static Api = Box::leak(Box::new(api));
    *slot = Some((dir, api));
    Ok(api)
}

/// One frame of interleaved RGB, at the effect's input size: floats in 0..=1, or bytes.
#[derive(Clone, Copy)]
pub enum Frame<'a> {
    F32(&'a [f32]),
    U8(&'a [u8]),
}

/// A loaded `VideoSuperRes` effect for one mode and one input and output size. Loading takes most of a second, so
/// keep it while the sizes stay the same.
pub struct SuperRes {
    api: &'static Api,
    fx: Handle,
    stream: Stream,
    // boxed: the effect keeps pointers to the first two
    src: Box<NvCVImage>,
    dst: Box<NvCVImage>,
    tmp: Box<NvCVImage>,
    model: PathBuf,
    mode: Mode,
    input: (usize, usize),
    output: (usize, usize),
}

// SAFETY: the handles are only used through &mut self
unsafe impl Send for SuperRes {}

impl SuperRes {
    /// Loads `mode` from `model` (an `nvngx_vsr.dll` with the SDK beside it), for `input` (width, height) frames
    /// upscaled to `output`. Denoise and deblur need `output == input`, the other modes at least `input`.
    pub fn new(model: &Path, mode: Mode, input: (usize, usize), output: (usize, usize)) -> Result<Self> {
        if !is_model(model) {
            bail!("{} isn't NVIDIA's RTX Video Super Resolution model ({MODEL})", model.display());
        }
        let ((w, h), (ow, oh)) = (input, output);
        if w == 0 || h == 0 || ow == 0 || oh == 0 {
            bail!("RTX Video Super Resolution of an empty image ({w}x{h} to {ow}x{oh})");
        }
        if mode.same_size() && output != input {
            bail!("RTX Video Super Resolution {} keeps the size: {w}x{h} can't become {ow}x{oh}", mode.name());
        }
        if ow < w || oh < h {
            bail!("RTX Video Super Resolution doesn't downscale: {w}x{h} to {ow}x{oh}");
        }
        let dir = model.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let api = api(dir)?;
        let mut sr = SuperRes {
            api,
            fx: std::ptr::null_mut(),
            stream: std::ptr::null_mut(),
            src: NvCVImage::zeroed(),
            dst: NvCVImage::zeroed(),
            tmp: NvCVImage::zeroed(),
            model: model.to_path_buf(),
            mode,
            input,
            output,
        };
        // SAFETY: the SDK's documented sequence; `sr` frees whatever was made if a step fails
        unsafe {
            api.check((api.create_effect)(c"VideoSuperRes".as_ptr(), &mut sr.fx), "CreateEffect(VideoSuperRes)")?;
            api.check((api.stream_create)(&mut sr.stream), "CudaStreamCreate")?;
            api.check((api.set_cuda_stream)(sr.fx, c"CudaStream".as_ptr(), sr.stream), "SetCudaStream")?;
            api.check((api.image_alloc)(&mut *sr.src, w as u32, h as u32, NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 32), "allocating the input")?;
            api.check((api.image_alloc)(&mut *sr.dst, ow as u32, oh as u32, NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 32), "allocating the output")?;
            api.check((api.set_image)(sr.fx, c"SrcImage0".as_ptr(), &mut *sr.src), "SetImage(SrcImage0)")?;
            api.check((api.set_image)(sr.fx, c"DstImage0".as_ptr(), &mut *sr.dst), "SetImage(DstImage0)")?;
            api.check((api.set_u32)(sr.fx, c"QualityLevel".as_ptr(), mode as u32), "QualityLevel")?;
            api.check((api.set_f32)(sr.fx, c"Strength".as_ptr(), 1.0), "Strength")?;
            // RGB8, nvvfx's default
            api.check((api.set_u32)(sr.fx, c"ImageEncodingMode".as_ptr(), 0), "ImageEncodingMode")?;
            api.check((api.load)(sr.fx), "Load")?;
        }
        Ok(sr)
    }

    /// True if this effect is `mode` from `model` for these sizes, so it can be reused.
    pub fn is(&self, model: &Path, mode: Mode, input: (usize, usize), output: (usize, usize)) -> bool {
        self.model == model && self.mode == mode && self.input == input && self.output == output
    }

    pub fn output(&self) -> (usize, usize) {
        self.output
    }

    /// Upscales one frame: interleaved RGB bytes at the output size.
    pub fn run(&mut self, frame: Frame<'_>) -> Result<Vec<u8>> {
        let ((w, h), (ow, oh)) = (self.input, self.output);
        let n = w * h * 3;
        let (ptr, ty, size, scale) = match frame {
            Frame::F32(p) => (p.as_ptr() as *mut c_void, NVCV_F32, 4, 255.0),
            Frame::U8(p) => (p.as_ptr() as *mut c_void, NVCV_U8, 1, 1.0),
        };
        let len = match frame {
            Frame::F32(p) => p.len(),
            Frame::U8(p) => p.len(),
        };
        if len != n {
            bail!("RTX Video Super Resolution: a {w}x{h} RGB frame has {n} values, got {len}");
        }
        let mut out = vec![0u8; ow * oh * 3];
        let api = self.api;
        let (mut view, mut out_view) = (NvCVImage::zeroed(), NvCVImage::zeroed());
        // SAFETY: the views wrap `frame` and `out`, which outlive the synchronized stream; the SDK only reads `frame`
        unsafe {
            api.check((api.image_init)(&mut *view, w as u32, h as u32, (w * 3 * size) as i32, ptr, NVCV_RGB, ty, NVCV_CHUNKY, NVCV_CPU), "wrapping the input")?;
            api.check((api.image_transfer)(&*view, &mut *self.src, scale, self.stream, &mut *self.tmp), "uploading the input")?;
            api.check((api.run)(self.fx, 0), "Run")?;
            api.check(
                (api.image_init)(&mut *out_view, ow as u32, oh as u32, (ow * 3) as i32, out.as_mut_ptr() as *mut c_void, NVCV_RGB, NVCV_U8, NVCV_CHUNKY, NVCV_CPU),
                "wrapping the output",
            )?;
            api.check((api.image_transfer)(&*self.dst, &mut *out_view, 1.0, self.stream, &mut *self.tmp), "downloading the output")?;
            api.check((api.stream_sync)(self.stream), "CudaStreamSynchronize")?;
        }
        Ok(out)
    }
}

impl Drop for SuperRes {
    fn drop(&mut self) {
        let api = self.api;
        // SAFETY: each was made by the SDK, or is still empty
        unsafe {
            if !self.stream.is_null() {
                (api.stream_sync)(self.stream);
            }
            if !self.fx.is_null() {
                (api.destroy_effect)(self.fx);
            }
            for im in [&mut self.src, &mut self.dst, &mut self.tmp] {
                (api.image_dealloc)(&mut **im);
            }
            if !self.stream.is_null() {
                (api.stream_destroy)(self.stream);
            }
        }
    }
}

/// Upscales `count` interleaved RGB frames of `input` (width, height) to `output`, loading the effect for this call.
pub fn upscale(model: &Path, mode: Mode, frames: &[Frame<'_>], input: (usize, usize), output: (usize, usize)) -> Result<Vec<Vec<u8>>> {
    let mut sr = SuperRes::new(model, mode, input, output)?;
    frames.iter().map(|f| sr.run(*f)).collect()
}

/// `RTXVideoSuperResolution`'s output size for `scale`: `int(side * scale)` rounded to a multiple of 8 (half to
/// even, as Python's `round`), at least 8.
pub fn node_output_size(width: usize, height: usize, scale: f64) -> (usize, usize) {
    let side = |s: usize| (((s as f64 * scale).trunc() / 8.0).round_ties_even() as usize * 8).max(8);
    (side(width), side(height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_by_name() {
        assert_eq!(Mode::from_name("ultra"), Some(Mode::Ultra));
        assert_eq!(Mode::from_name("deblur-high"), Some(Mode::DeblurHigh));
        assert_eq!(Mode::from_name("STREAMING_ULTRA").map(|m| m as u32), Some(23));
        assert_eq!(Mode::from_name("off"), None);
        assert!(Mode::DenoiseLow.same_size() && Mode::DeblurUltra.same_size() && !Mode::HighbitrateLow.same_size());
        for (name, mode) in Mode::ALL {
            assert_eq!(mode.name(), name);
        }
    }

    #[test]
    fn node_sizes() {
        assert_eq!(node_output_size(512, 768, 2.0), (1024, 1536));
        // 20 / 8 = 2.5 rounds to 2, as in Python
        assert_eq!(node_output_size(20, 28, 1.0), (16, 32));
        assert_eq!(node_output_size(3, 3, 1.0), (8, 8));
        assert_eq!(node_output_size(1000, 600, 1.37), (1368, 824));
    }

    #[test]
    fn model_name() {
        assert!(is_model(Path::new("upscale_models/nvidia/NVNGX_VSR.dll")));
        assert!(!is_model(Path::new("upscale_models/4x_esrgan.pth")));
    }
}
