//! Scratch benchmark of the image VAE decoders. Not part of the build.
//! Usage: vae_bench <qwen|qwen21|flux2|sd|seedvr> [latent h] [latent w] [iters] [out.bin]: decodes a fixed smooth latent and
//! writes the output bytes to out.bin if given, so builds can be compared byte for byte.
use diffusion::cuda::{self, DevBuf, Stream};
use diffusion::{flux2vae, sdvae, seedvr_vae, vae, vae21};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let fam = a.get(1).map_or("qwen", |s| s.as_str()).to_string();
    let h: usize = a.get(2).map_or(128, |v| v.parse().unwrap());
    let w: usize = a.get(3).map_or(128, |v| v.parse().unwrap());
    let iters: usize = a.get(4).map_or(5, |v| v.parse().unwrap());
    cuda::init(0).unwrap();
    unsafe { diffusion::ffi::gpu_blas_init(64 << 20) };
    let st = Stream::new().unwrap();
    let root = PathBuf::from(std::env::var("MODELS_ROOT").unwrap());
    let ch = match fam.as_str() {
        "qwen" => 16,
        "qwen21" => vae21::CH,
        "flux2" => flux2vae::CH,
        "sd" => sdvae::Z,
        "seedvr" => seedvr_vae::Z,
        f => panic!("unknown family {f}"),
    };
    // a smooth latent with some structure, so the decoder sees image-like statistics
    let mut lat = vec![0f32; ch * h * w];
    for c in 0..ch {
        for y in 0..h {
            for x in 0..w {
                lat[(c * h + y) * w + x] = ((x as f32 * 0.07 + c as f32).sin() + (y as f32 * 0.05 - c as f32 * 0.3).cos()) * 0.8;
            }
        }
    }
    let d = DevBuf::alloc(lat.len() * 4).unwrap();
    d.upload(0, &diffusion::st::f32_bytes(&lat), &st).unwrap();
    let decode: Box<dyn Fn() -> Vec<u8>> = match fam.as_str() {
        "qwen" => {
            let v = vae::Vae::load(&root.join("VAE/qwen_image_vae.safetensors"), &st).unwrap();
            Box::new(move || v.decode(&d, h, w, &st).unwrap())
        }
        "qwen21" => {
            let v = vae21::Vae::load(&root.join("VAE/QwenImage/qwen_image_2.1_vae_bf16.safetensors"), &st).unwrap();
            Box::new(move || v.decode(&d, h, w, &st).unwrap())
        }
        "flux2" => {
            let v = flux2vae::Vae::load(&root.join("VAE/Flux/flux2-vae.safetensors"), &st).unwrap();
            Box::new(move || v.decode(&d, h, w, &st).unwrap())
        }
        "sd" => {
            let v = sdvae::Vae::load(&root.join("Stable-diffusion/SoushikiV1.0.safetensors"), 0.13025, &st).unwrap();
            Box::new(move || v.decode(&d, h, w, &st).unwrap())
        }
        _ => {
            let v = seedvr_vae::Vae::load(&root.join("VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors"), &st).unwrap();
            Box::new(move || v.decode(&d, h, w, &st).unwrap().iter().flat_map(|f| f.to_le_bytes()).collect())
        }
    };
    let mut out = Vec::new();
    let mut times = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        out = decode();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let warm = &times[1.min(times.len() - 1)..];
    let mut sorted = warm.to_vec();
    sorted.sort_by(f64::total_cmp);
    println!("{fam} {h}x{w}: first {:.1} ms, warm median {:.1} ms", times[0], sorted[sorted.len() / 2]);
    if let Some(p) = a.get(5) {
        std::fs::write(p, &out).unwrap();
    }
}
