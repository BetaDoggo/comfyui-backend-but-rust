//! ComfyUI-compatible noise (torch CPU randn) and Flux-style sigma schedule.

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

/// ModelSamplingFlux(shift) + "simple" scheduler.
pub fn simple_sigmas(shift: f64, steps: usize) -> Vec<f32> {
    let n = 10000usize;
    let e = shift.exp();
    let table: Vec<f32> = (1..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let v = (1.0f32 / t - 1.0) as f64;
            (e / (e + v)) as f32
        })
        .collect();
    let ss = n as f64 / steps as f64;
    let mut sig: Vec<f32> = (0..steps).map(|x| table[n - 1 - (x as f64 * ss) as usize]).collect();
    sig.push(0.0);
    sig
}
