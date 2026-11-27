//! Sol³ attention (csrc/sol3_attn.cu) against sol3_attn's own SM120 kernels, recorded by `tools/sol3_ref.py`: the bf16
//! exact branch closely, and the int8 one by its rounding. The route is the same in both.
//! Needs a CUDA GPU, and skips when `tests/sol3_golden.json` hasn't been recorded.

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

#[test]
fn sol3_matches_sol3_attn() {
    let Ok(text) = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/sol3_golden.json")) else {
        eprintln!("skipped: record tests/sol3_golden.json with tools/sol3_ref.py");
        return;
    };
    let g: Value = serde_json::from_str(&text).unwrap();
    let (h, stride) = (g["h"].as_u64().unwrap() as usize, g["stride"].as_u64().unwrap() as usize);
    cuda::init(0).unwrap();
    let st = Stream::new().unwrap();
    for c in g["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let (t, prefix) = (c["t"].as_u64().unwrap() as usize, c["prefix"].as_u64().unwrap() as usize);
        let n = t.div_ceil(64);
        let input = DevBuf::alloc(t * 3 * h * HD * 2).unwrap();
        input.upload(0, &qkv(t, h), &st).unwrap();
        let (q, rs, hs) = (input.ptr(), (3 * h * HD) as i64, HD as i64);
        let (k, v) = (input.at(h * HD * 2), input.at(2 * h * HD * 2));
        for int8 in [false, true] {
            let name = &format!("{name} {}", if int8 { "int8" } else { "bf16" });
            let out = DevBuf::alloc(t * h * HD * 2).unwrap();
            let ws = DevBuf::alloc(unsafe { ffi::gpu_sol3_workspace(t as i32, h as i32, h as i32, int8 as i32) } as usize).unwrap();
            let code = unsafe {
                ffi::gpu_sol3_attn(q, k, v, out.ptr(), ws.ptr(), t as i32, h as i32, h as i32, rs, hs, (h * HD) as i64, prefix as i32, int8 as i32,
                    st.raw())
            };
            assert_eq!(code, 0, "{name}: gpu_sol3_attn returned {code}");
            // the prefix's queries attend everything densely, as sol3_attn's default SDPA
            if prefix > 0 {
                let lse = DevBuf::alloc(h * t.div_ceil(128) * 128 * 4).unwrap();
                let bs = t as i64 * rs;
                let code = unsafe {
                    ffi::gpu_flash_attn_d(q, k, v, out.ptr(), lse.ptr(), 1, prefix as i32, t as i32, h as i32, h as i32, HD as i32, bs, rs, hs, bs, rs, hs,
                        bs, rs, hs, (t * h * HD) as i64, (h * HD) as i64, HD as i64, 1.0 / (HD as f32).sqrt(), 0, st.raw())
                };
                assert_eq!(code, 0);
            }
            st.sync().unwrap();

            // the route: each query block's exact, pure and mixed key blocks, and its selected tokens
            let mut offs = [0i64; 3];
            assert_eq!(unsafe { ffi::gpu_sol3_route_offsets(t as i32, h as i32, h as i32, int8 as i32, offs.as_mut_ptr()) }, 0);
            let mut flags = vec![0u8; h * n * n];
            ws.download(offs[0] as usize, &mut flags, &st).unwrap();
            let mut cutb = vec![0u8; h * n * 4];
            ws.download(offs[2] as usize, &mut cutb, &st).unwrap();
            let want_rows = c["rows"].as_array().unwrap();
            let (mut row_diff, mut tok_diff, mut tok_total) = (0, 0i64, 0i64);
            for (i, row) in flags.chunks_exact(n).enumerate() {
                let f: Vec<i8> = row.iter().map(|&b| b as i8).collect();
                let got = [f.iter().filter(|&&x| x == -1).count() as i64, f.iter().filter(|&&x| x == -2).count() as i64,
                    f.iter().filter(|&&x| x >= 0).count() as i64, f.iter().map(|&x| x.max(0) as i64).sum::<i64>()];
                let want: Vec<i64> = want_rows[i].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                row_diff += (got[..3] != want[..3]) as usize;
                tok_diff += (got[3] - want[3]).abs();
                tok_total += want[3];
            }
            let cut: Vec<f32> = cutb.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let want_cut: Vec<f32> = c["cut"].as_array().unwrap().iter().map(|x| x.as_f64().map_or(f32::INFINITY, |v| v as f32)).collect();
            let cut_worst = cut.iter().zip(&want_cut).filter(|(_, w)| w.is_finite()).map(|(a, w)| (a - w).abs() / w.abs().max(1.0)).fold(0f32, f32::max);
            eprintln!("{name}: {row_diff} of {} query blocks route blocks differently, selected tokens off by {tok_diff} of {tok_total}, cuts within {cut_worst:.2e}",
                h * n);

            let mut b = vec![0u8; t * h * HD * 2];
            out.download(0, &mut b, &st).unwrap();
            let got: Vec<f32> = b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
            let want: Vec<f32> = c["out"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let (mut worst, mut sum, mut sq, mut ref_sq) = (0f32, 0f64, 0f64, 0f64);
            for (i, w) in want.iter().enumerate() {
                let d = (got[i * stride] - w).abs();
                worst = worst.max(d);
                sum += d as f64;
                sq += (d as f64).powi(2);
                ref_sq += (*w as f64).powi(2);
            }
            let rel = (sq / ref_sq).sqrt();
            eprintln!("{name}: sampled outputs differ by {:.2e} on average, {worst:.2e} at worst, {:.2}% relative RMS", sum / want.len() as f64, 100.0 * rel);
            assert!(row_diff * 100 <= h * n, "{name}: {row_diff} query blocks route differently");
            assert!(tok_diff * 100 <= tok_total.max(100), "{name}: selected tokens off by {tok_diff} of {tok_total}");
            assert!(cut_worst < 1e-3, "{name}: cuts differ by {cut_worst}");
            if int8 {
                // the exact rows' int8 rounding, against upstream's bf16
                assert!(rel < 0.01, "{name}: outputs differ by {:.2}% RMS", 100.0 * rel);
            } else {
                assert!(worst < 3e-2, "{name}: outputs differ by {worst}");
            }
        }
    }
}
