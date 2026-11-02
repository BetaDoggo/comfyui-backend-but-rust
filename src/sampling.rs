//! ComfyUI-compatible noise (torch CPU and CUDA randn) and flow sigma schedules.

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

/// `time_snr_shift`.
fn snr_shift(alpha: f64, t: f32) -> f32 {
    if alpha == 1.0 {
        return t;
    }
    let a = alpha as f32;
    a * t / (1.0 + (a - 1.0) * t)
}

/// ModelSamplingDiscreteFlow(shift) + "simple" scheduler.
pub fn flow_sigmas(shift: f64, steps: usize) -> Vec<f32> {
    let n = 1000usize;
    let table: Vec<f32> = (1..=n).map(|i| snr_shift(shift, i as f32 / n as f32)).collect();
    let ss = n as f64 / steps as f64;
    let mut sig: Vec<f32> = (0..steps).map(|x| table[n - 1 - (x as f64 * ss) as usize]).collect();
    sig.push(0.0);
    sig
}

/// `percent_to_sigma` of ModelSamplingDiscreteFlow.
pub fn flow_percent_to_sigma(shift: f64, percent: f64) -> f32 {
    snr_shift(shift, (1.0 - percent) as f32)
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
/// depends on the card's core count ([`crate::ffi::krea_randn_threads`]).
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
        let out = (start..start + n)
            .map(|e| {
                let (q, idx) = (e / self.threads, e % self.threads);
                let c = philox4x32([self.offset + (q / 4) as u32, 0, idx as u32, 0], self.key);
                let (x, y) = if q % 4 < 2 { (c[0], c[1]) } else { (c[2], c[3]) };
                let u = x as f64 * TWO32 + TWO32 / 2.0;
                let v = (y as f64 * TWO32 + TWO32 / 2.0) * std::f64::consts::TAU;
                let r = (-2.0 * u.ln()).sqrt();
                (r * if q % 2 == 0 { v.sin() } else { v.cos() }) as f32
            })
            .collect();
        self.offset += ((self.numel - 1) / (self.threads * 4) + 1) as u32;
        out
    }
}

/// Per-step coefficients of ComfyUI's `sample_er_sde` (VP ER-SDE-Solver-3, s_noise 1).
pub struct ErSdeStep {
    /// x = keep * x + denoised * x_denoised, when not the last step
    pub keep: f32,
    pub denoised: f32,
    /// denoised_d = (denoised - old) / d_div, added with weight d_weight; 0 before stage 2
    pub d_div: f32,
    pub d_weight: f32,
    /// denoised_u = (denoised_d - old_d) / u_div, added with weight u_weight; 0 before stage 3
    pub u_div: f32,
    pub u_weight: f32,
    pub noise: f32,
}

fn er_scale(x: f32) -> f32 {
    x * (x.powf(0.3).exp() + 10.0)
}

/// `sigmas` must already have the first sigma offset below 1. Empty entry for the final step to 0.
pub fn er_sde_steps(sigmas: &[f32]) -> Vec<Option<ErSdeStep>> {
    const POINTS: usize = 200;
    let lam: Vec<f32> = sigmas.iter().map(|s| s / (1.0 - s)).collect();
    (0..sigmas.len() - 1)
        .map(|i| {
            if sigmas[i + 1] == 0.0 {
                return None;
            }
            let (ls, lt) = (lam[i], lam[i + 1]);
            let (alpha_s, alpha_t) = (sigmas[i] / ls, sigmas[i + 1] / lt);
            let r = er_scale(lt) / er_scale(ls);
            let mut st = ErSdeStep { keep: alpha_t / alpha_s * r, denoised: alpha_t * (1.0 - r), d_div: 0.0, d_weight: 0.0,
                u_div: 0.0, u_weight: 0.0, noise: alpha_t * (lt * lt - ls * ls * r * r).max(0.0).sqrt() };
            if i >= 1 {
                let dt = lt - ls;
                let h = -dt / POINTS as f32;
                let (mut sum, mut sum_u) = (0f32, 0f32);
                for p in 0..POINTS {
                    let pos = lt + p as f32 * h;
                    let sp = er_scale(pos);
                    sum += 1.0 / sp;
                    sum_u += (pos - ls) / sp;
                }
                st.d_div = ls - lam[i - 1];
                st.d_weight = alpha_t * (dt + sum * h * er_scale(lt));
                if i >= 2 {
                    st.u_div = (ls - lam[i - 2]) / 2.0;
                    st.u_weight = alpha_t * (dt * dt / 2.0 + sum_u * h * er_scale(lt));
                }
            }
            Some(st)
        })
        .collect()
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

    #[test]
    fn er_sde_first_step_matches_comfy() {
        let mut s = flow_sigmas(3.0, 8);
        assert!((s[1] - 0.954_545_4).abs() < 1e-6, "{s:?}");
        s[0] = flow_percent_to_sigma(3.0, 1e-4);
        assert!((s[0] - 0.999_966_7).abs() < 1e-6, "{}", s[0]);
        let e = er_sde_steps(&s);
        let c = e[0].as_ref().unwrap();
        assert!((c.denoised - 0.045_454_56).abs() < 1e-5, "{}", c.denoised);
        assert!((c.noise - 0.954_545_4).abs() < 1e-5, "{}", c.noise);
        assert!(e[7].is_none());
    }
}
