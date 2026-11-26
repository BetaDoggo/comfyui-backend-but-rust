//! Scratch benchmark of the pi-H3 DWD pixel head alone, without H3's DiT. Not part of the build.
//! Usage: dwd_bench <width> <height> <frames> [iters] [out.bin]: decodes fixed smooth features of that clip's token grid
//! through the 2-step adapter's head and writes the frames' bytes to out.bin if given, so builds compare byte for byte.
use diffusion::cuda::{self, DevBuf, Stream};
use diffusion::{h3, h3dwd, st};
use half::bf16;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).map_or(d, |v| v.parse().unwrap());
    let (width, height, frames, iters) = (arg(1, 512), arg(2, 384), arg(3, 22), arg(4, 5));
    cuda::init(0).unwrap();
    unsafe { diffusion::ffi::gpu_blas_init(64 << 20) };
    let s = Stream::new().unwrap();
    let root = std::path::PathBuf::from(std::env::var("MODELS_ROOT").unwrap());
    let ad = Arc::new(st::SafeTensors::open(&root.join("Lora/pi_h3_2step_dwd_adapter_int8_convrot.safetensors")).unwrap());
    let sh = h3::Shape::new(width, height, frames);
    let grid = (sh.t, sh.h / 2, sh.w / 2);
    let t0 = Instant::now();
    let mut head = h3dwd::Refiner::load(&ad, 8, &s).unwrap();
    let fw = head.width();
    let tokens = grid.0 * grid.1 * grid.2;
    // smooth, roughly unit-variance features, like a LayerNorm's output
    let v: Vec<f32> = (0..tokens * fw).map(|i| ((i % fw) as f32 * 0.37 + (i / fw) as f32 * 0.011).sin() * 1.3).collect();
    let feat = DevBuf::alloc(tokens * fw * 2).unwrap();
    feat.upload(0, &v.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect::<Vec<u8>>(), &s).unwrap();
    head.place(cuda::device_free().unwrap().saturating_sub((3 << 30) + h3dwd::activation_bytes(grid))).unwrap();
    s.sync().unwrap();
    eprintln!("grid {grid:?}, {tokens} tokens; loaded and placed in {:.2}s", t0.elapsed().as_secs_f64());
    let mut out = None;
    for i in 0..iters {
        let t1 = Instant::now();
        let clip = head.decode(&feat, grid, &s).unwrap();
        s.sync().unwrap();
        eprintln!("iter {i}: {} frames of {}x{} in {:.1} ms", clip.t, clip.w, clip.h, t1.elapsed().as_secs_f64() * 1e3);
        out = Some(clip);
    }
    if let (Some(p), Some(c)) = (a.get(5), out) {
        std::fs::write(p, st::f32_bytes(&c.data)).unwrap();
    }
}
