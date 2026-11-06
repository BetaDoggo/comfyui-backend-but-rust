//! Block-sparse attention against comfy-kitchen's `sol_attn` and top-k threshold, recorded by `tools/sparse_attn_ref.py`.
//! Needs a CUDA GPU.

use diffusion::cuda::{self, DevBuf, Stream};
use diffusion::ffi;
use serde_json::Value;

const HD: usize = 128;

fn hash_unit(idx: u64, salt: u64) -> f64 {
    let mut u = (idx.wrapping_mul(2654435761).wrapping_add(salt) & 0xFFFF_FFFF) as u32;
    u ^= u >> 15;
    u = u.wrapping_mul(2246822519);
    u ^= u >> 13;
    u = u.wrapping_mul(3266489917);
    u ^= u >> 16;
    u as f64 / 4294967296.0 - 0.5
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The script's `tensor(salt)`, interleaved as rows of q, k and v: [T, 3, H, 128] bf16.
fn qkv(t: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; t * 3 * h * HD * 2];
    for (j, salt) in [1u64, 2, 3].into_iter().enumerate() {
        for r in 0..t {
            for hh in 0..h {
                for d in 0..HD {
                    let group = ((r / 64) % 5) as u64;
                    let shared = hash_unit((group * h as u64 + hh as u64) * HD as u64 + d as u64, salt);
                    let own = hash_unit(((r * h + hh) * HD + d) as u64, salt + 7919);
                    let v = bf16_bits((4.0 * shared + own) as f32);
                    let at = ((r * 3 + j) * h * HD + hh * HD + d) * 2;
                    out[at..at + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    out
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn topk_count(ratio: f64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    ((ratio * n as f64).round_ties_even() as i64).max(1).min(n as i64 - 1) as usize
}

#[test]
fn sparse_attention_matches_comfy_kitchen() {
    let g: Value = serde_json::from_str(include_str!("sparse_golden.json")).unwrap();
    let (t, h, stride) = (g["t"].as_u64().unwrap() as usize, g["h"].as_u64().unwrap() as usize, g["stride"].as_u64().unwrap() as usize);
    cuda::init(0).unwrap();
    let st = Stream::new().unwrap();
    let input = DevBuf::alloc(t * 3 * h * HD * 2).unwrap();
    input.upload(0, &qkv(t, h), &st).unwrap();
    let (q, rs, hs) = (input.ptr(), (3 * h * HD) as i64, HD as i64);
    let (k, v) = (input.at(h * HD * 2), input.at(2 * h * HD * 2));
    let out = DevBuf::alloc(t * h * HD * 2).unwrap();
    let nb = t.div_ceil(64);
    let scale = 1.0 / (HD as f32).sqrt();
    for c in g["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let pair = |key: &str| {
            let a = c[key].as_array().unwrap();
            (a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32)
        };
        let (sinks, sink_q) = (pair("sinks"), pair("sink_q"));
        let (tau, ratio, aug) = (c["tau"].as_f64().unwrap() as f32, c["ratio"].as_f64().unwrap(), c["extra_tokens"].as_i64().unwrap() as i32);
        let ws = DevBuf::alloc(unsafe { ffi::gpu_sol_attn_workspace(t as i32, h as i32, aug) } as usize).unwrap();
        let thr = DevBuf::alloc(h * nb * 4).unwrap();
        let mut thr_ptr = std::ptr::null();
        if ratio > 0.0 {
            let scratch = DevBuf::alloc(unsafe { ffi::gpu_sol_topk_workspace(t as i32, h as i32) } as usize).unwrap();
            let kk = topk_count(ratio, nb - (sinks.1 - sinks.0) as usize);
            let code = unsafe { ffi::gpu_sol_topk_threshold(q, k, rs, hs, t as i32, h as i32, kk as i32, scale, sinks.0, sinks.1, thr.ptr(), scratch.ptr(), st.raw()) };
            assert_eq!(code, 0);
            st.sync().unwrap();
            let mut b = vec![0u8; h * nb * 4];
            thr.download(0, &mut b, &st).unwrap();
            let want: Vec<f32> = c["threshold"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let worst = f32s(&b).iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1e-3)).fold(0f32, f32::max);
            eprintln!("{name}: threshold worst relative error {worst:.2e}");
            assert!(worst < 1e-4, "{name}: thresholds differ by {worst}");
            thr_ptr = thr.ptr() as ffi::P;
        }
        let code = unsafe { ffi::gpu_sol_attn(q, k, v, out.ptr(), ws.ptr(), t as i32, h as i32, rs, hs, tau, scale, thr_ptr as *const f32 as ffi::P,
            sinks.0, sinks.1, sink_q.0, sink_q.1, aug, st.raw()) };
        assert_eq!(code, 0);
        st.sync().unwrap();
        let mut b = vec![0u8; t * h * HD * 2];
        out.download(0, &mut b, &st).unwrap();
        let got: Vec<f32> = b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
        let want: Vec<f32> = c["out"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let (mut worst, mut differ) = (0f32, 0);
        for (i, w) in want.iter().enumerate() {
            let d = (got[i * stride] - w).abs();
            worst = worst.max(d);
            differ += (d > 0.0) as usize;
        }
        eprintln!("{name}: {differ} of {} sampled values differ, worst {worst:.2e}", want.len());
        assert!(worst < 2e-2, "{name}: outputs differ by {worst}");
    }
}
