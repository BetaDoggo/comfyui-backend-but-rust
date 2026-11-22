//! Scratch: every int8 GEMM tile config on a model's shapes, against the heuristic's pick. Not part of the build.
//! `gemm_sweep <krea|moss|qwen21> [m]`
use diffusion::cuda::{self, DevBuf, Event, Stream};
use diffusion::ffi::*;
use std::ffi::c_void;

extern "C" {
    fn launch_cutlass_int8_dequant_config(a: *const c_void, b: *const c_void, xs: *const c_void, ws: *const c_void, d: *mut c_void,
        m: i64, n: i64, k: i64, out_dt: i32, config: i32, st: *mut c_void) -> bool;
    fn cutlass_int8_selected_config(m: i64, n: i64, k: i64) -> i32;
}

fn time(st: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let (a, b) = (Event::timing().unwrap(), Event::timing().unwrap());
    a.record(st).unwrap();
    for _ in 0..iters {
        f();
    }
    b.record(st).unwrap();
    st.sync().unwrap();
    b.elapsed_ms(&a).unwrap() as f64 * 1e3 / iters as f64
}

fn rand_buf(bytes: usize, st: &Stream, seed: u32) -> DevBuf {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    let v: Vec<u8> = (0..bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect();
    let d = DevBuf::alloc(bytes).unwrap();
    d.upload(0, &v, st).unwrap();
    d
}

fn main() {
    cuda::init(0).unwrap();
    let st = Stream::new().unwrap();
    let s = st.raw();
    let model = std::env::args().nth(1).unwrap_or("krea".into());
    let (m0, shapes): (usize, Vec<(&str, usize, usize)>) = match model.as_str() {
        "ab" => (4120, vec![("x", 32768, 16384)]),
        "moss" => (4125, vec![("qkv", 9216, 3072), ("o", 3072, 3072), ("ug", 16384, 3072), ("down", 3072, 8192)]),
        "qwen21" => (4096 + 64, vec![("qkv", 12288, 4096), ("o", 4096, 4096), ("gu", 24576, 4096), ("down", 4096, 12288)]),
        _ => (4120, vec![("qkvg", 15360, 6144), ("o", 6144, 6144), ("gu", 32768, 6144), ("down", 6144, 16384)]),
    };
    let m: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(m0);
    let maxn = shapes.iter().map(|s| s.1).max().unwrap();
    let maxk = shapes.iter().map(|s| s.2).max().unwrap();
    let a8 = rand_buf(m * maxk, &st, 1);
    let w8 = rand_buf(maxn * maxk, &st, 2);
    let xs = DevBuf::alloc(m * 4).unwrap();
    let ws = DevBuf::alloc(maxn * 4).unwrap();
    let ones: Vec<u8> = (0..maxn.max(m)).flat_map(|_| 1e-4f32.to_le_bytes()).collect();
    xs.upload(0, &ones[..m * 4], &st).unwrap();
    ws.upload(0, &ones[..maxn * 4], &st).unwrap();
    let d = DevBuf::alloc(m * maxn * 2).unwrap();
    // `ab N K cfgA cfgB`: the two configs interleaved, medians of 15 rounds
    let args: Vec<String> = std::env::args().collect();
    if model == "ab" {
        let (n, k, ca, cb): (usize, usize, i32, i32) = (args[3].parse().unwrap(), args[4].parse().unwrap(), args[5].parse().unwrap(), args[6].parse().unwrap());
        let a8 = rand_buf(m * k, &st, 1);
        let w8 = rand_buf(n * k, &st, 2);
        let (mut ta, mut tb) = (vec![], vec![]);
        for _ in 0..15 {
            for (cfg, out) in [(ca, &mut ta), (cb, &mut tb)] {
                out.push(time(&st, 10, || unsafe {
                    launch_cutlass_int8_dequant_config(a8.ptr(), w8.ptr(), xs.ptr(), ws.ptr(), d.ptr(), m as i64, n as i64, k as i64, 2, cfg, s);
                }));
            }
        }
        ta.sort_by(f64::total_cmp);
        tb.sort_by(f64::total_cmp);
        println!("M={m} N={n} K={k}: cfg {ca} {:.0} us, cfg {cb} {:.0} us (medians; heuristic picks {})", ta[7], tb[7],
            unsafe { cutlass_int8_selected_config(m as i64, n as i64, k as i64) });
        return;
    }
    for (name, n, k) in shapes {
        let sel = unsafe { cutlass_int8_selected_config(m as i64, n as i64, k as i64) };
        let flops = 2.0 * (m * n * k) as f64;
        let iters = 30;
        let mut best = (f64::MAX, -1);
        let mut line = String::new();
        for cfg in 0..14 {
            let mut ok = true;
            let t = time(&st, iters, || unsafe {
                ok &= launch_cutlass_int8_dequant_config(a8.ptr(), w8.ptr(), xs.ptr(), ws.ptr(), d.ptr(), m as i64, n as i64, k as i64, 2, cfg, s);
            });
            if ok {
                line += &format!(" {cfg}:{t:.0}");
                if t < best.0 {
                    best = (t, cfg);
                }
            }
        }
        // the heuristic again, last, so it's timed as warm as the rest
        let th = time(&st, iters, || unsafe {
            gpu_int8_gemm(a8.ptr(), w8.ptr(), xs.ptr(), ws.ptr(), null(), d.ptr(), m as i64, n as i64, k as i64, DT_BF16, s);
        });
        println!("{model} {name:5} M={m} N={n} K={k}: heuristic cfg {sel} {th:.0} us ({:.0} TOPS); best cfg {} {:.0} us ({:.0} TOPS)",
            flops / th / 1e6, best.1, best.0, flops / best.0 / 1e6);
        println!("   {line}");
    }
}
