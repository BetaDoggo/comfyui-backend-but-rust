//! ComfyUI-compatible noise: torch's CPU and CUDA randn.

/// std::mt19937 / at::mt19937 (torch.manual_seed uses the low 32 bits of the seed).
struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    fn new(seed: u64) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = (seed & 0xffff_ffff) as u32;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[i] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn uniform_f32(&mut self) -> f32 {
        ((self.next() & ((1 << 24) - 1)) as f64 * (1.0 / (1u64 << 24) as f64)) as f32
    }
}

fn normal_fill_16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        d[j] = radius * theta.cos();
        d[j + 8] = radius * theta.sin();
    }
}

/// torch.randn(n, generator=torch.manual_seed(seed)) on CPU, float32 (n >= 16).
pub fn torch_randn(seed: u64, n: usize) -> Vec<f32> {
    assert!(n >= 16);
    let mut g = Mt19937::new(seed);
    let mut d: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
    let mut i = 0;
    while i + 16 <= n {
        normal_fill_16(&mut d[i..i + 16]);
        i += 16;
    }
    if n % 16 != 0 {
        let s = n - 16;
        for v in &mut d[s..] {
            *v = g.uniform_f32();
        }
        normal_fill_16(&mut d[s..]);
    }
    d
}

/// Consecutive `torch.randn` calls on one generator, as ComfyUI draws noise for a nested latent.
pub fn torch_randn_streams(seed: u64, sizes: &[usize]) -> Vec<Vec<f32>> {
    let mut g = Mt19937::new(seed);
    sizes.iter().map(|&n| {
        assert!(n >= 16);
        let mut d: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
        let mut i = 0;
        while i + 16 <= n {
            normal_fill_16(&mut d[i..i + 16]);
            i += 16;
        }
        if n % 16 != 0 {
            let s = n - 16;
            for v in &mut d[s..] {
                *v = g.uniform_f32();
            }
            normal_fill_16(&mut d[s..]);
        }
        d
    }).collect()
}

fn philox4x32(mut c: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const M0: u64 = 0xD251_1F53;
    const M1: u64 = 0xCD9E_8D57;
    for round in 0..10 {
        if round > 0 {
            key[0] = key[0].wrapping_add(0x9E37_79B9);
            key[1] = key[1].wrapping_add(0xBB67_AE85);
        }
        let (a, b) = (M0 * c[0] as u64, M1 * c[2] as u64);
        c = [(b >> 32) as u32 ^ c[1] ^ key[0], b as u32, (a >> 32) as u32 ^ c[3] ^ key[1], a as u32];
    }
    c
}

/// torch.randn on a CUDA generator, for one tensor size. Each of `threads` GPU threads owns a Philox
/// subsequence and draws 4 normals per iteration, laid out `threads` elements apart, so the result
/// depends on the card's core count ([`crate::ffi::gpu_randn_threads`]).
pub struct CudaRandn {
    key: [u32; 2],
    numel: usize,
    threads: usize,
    offset: u32,
}

impl CudaRandn {
    pub fn new(seed: u64, numel: usize, max_threads: usize) -> Self {
        let threads = max_threads.min(numel.div_ceil(256) * 256);
        CudaRandn { key: [seed as u32, (seed >> 32) as u32], numel, threads, offset: 0 }
    }

    /// Elements `start..start + n` of the next call.
    pub fn next(&mut self, start: usize, n: usize) -> Vec<f32> {
        const TWO32: f64 = 2.328_306_4e-10;
        let (threads, offset, key) = (self.threads, self.offset, self.key);
        let normal = |e: usize| {
            let (q, idx) = (e / threads, e % threads);
            let c = philox4x32([offset + (q / 4) as u32, 0, idx as u32, 0], key);
            let (x, y) = if q % 4 < 2 { (c[0], c[1]) } else { (c[2], c[3]) };
            let u = x as f64 * TWO32 + TWO32 / 2.0;
            let v = (y as f64 * TWO32 + TWO32 / 2.0) * std::f64::consts::TAU;
            let r = (-2.0 * u.ln()).sqrt();
            (r * if q % 2 == 0 { v.sin() } else { v.cos() }) as f32
        };
        let mut out = vec![0f32; n];
        let chunk = n.div_ceil(std::thread::available_parallelism().map_or(1, |p| p.get())).max(4096);
        std::thread::scope(|scope| {
            for (i, part) in out.chunks_mut(chunk).enumerate() {
                let normal = &normal;
                scope.spawn(move || {
                    for (j, v) in part.iter_mut().enumerate() {
                        *v = normal(start + i * chunk + j);
                    }
                });
            }
        });
        self.offset += ((self.numel - 1) / (self.threads * 4) + 1) as u32;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_randn_matches_torch() {
        // torch.randn(1000) then torch.randn(70000) on a CUDA generator seeded with 42
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-5);
        let a = CudaRandn::new(42, 1000, 73728).next(0, 3);
        assert!(close(&a, &[0.194_018_8, 2.161_373_6, -0.172_050_22]), "{a:?}");
        let mut g = CudaRandn::new(42, 70000, 73728);
        g.next(0, 1);
        let b = g.next(0, 2);
        assert!(close(&b, &[0.139_137_55, -0.108_216_64]), "{b:?}");
    }

    #[test]
    fn cuda_randn_wraps_at_the_thread_count() {
        // past `threads` elements the same subsequence gives its next normal
        let mut g = CudaRandn::new(42, 200_000, 73728);
        let (first, second) = (g.next(0, 1)[0], { g.offset = 0; g.next(73728, 1)[0] });
        let c = philox4x32([0, 0, 0, 0], [42, 0]);
        let u = c[0] as f64 * 2.328_306_4e-10 + 1.164_153_2e-10;
        let v = (c[1] as f64 * 2.328_306_4e-10 + 1.164_153_2e-10) * std::f64::consts::TAU;
        assert!((first as f64 - (-2.0 * u.ln()).sqrt() * v.sin()).abs() < 1e-6);
        assert!((second as f64 - (-2.0 * u.ln()).sqrt() * v.cos()).abs() < 1e-6);
    }
}
