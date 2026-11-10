//! ComfyUI's samplers and schedulers for flow (`CONST`) models, on the host.
//!
//! Each one follows `comfy/k_diffusion/sampling.py`, `comfy/samplers.py` and `extra_samplers/uni_pc.py`
//! step for step, in f32 like the original, so a seed gives the same noise and the same model inputs.
//! The latent crosses the bus once per model evaluation, which costs far less than the evaluation.

use crate::sampling::{torch_randn, CudaRandn};
use anyhow::{anyhow, bail, Result};

type V = Vec<f32>;

pub const SAMPLERS: &[&str] = &[
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2",
    "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2",
    "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
];

pub const SCHEDULERS: &[&str] = &[
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal", "turbo",
    "align_your_steps", "ltxv", "flux2", "ideogram4", "ideogram4turbo", "hyperflow",
];

/// HyperFlow H3 Flash's raw sigma grids by step count, before the model's shift.
const HYPERFLOW_GRIDS: [&[f32]; 3] = [
    &[1.0, 0.681872, 0.205019, 0.036496, 0.0],
    &[1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0],
    &[1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0],
];

/// The grid HyperFlow H3 Flash was distilled on, shifted as its pipeline shifts it (`shift_sigmas`, in float32).
pub fn hyperflow_sigmas(ms: ModelSampling, steps: usize) -> Result<V> {
    let ModelSampling::Flow { shift } = ms else { bail!("the hyperflow scheduler needs a flow model") };
    let Some(grid) = HYPERFLOW_GRIDS.iter().find(|g| g.len() == steps + 1) else { bail!("the hyperflow scheduler has 4, 6 or 8 steps, not {steps}") };
    let s = shift as f32;
    Ok(grid.iter().map(|&g| s * g / (1.0 + (s - 1.0) * g)).collect())
}

/// True for samplers that read the uncond prediction even at CFG 1 (`disable_cfg1_optimization`).
pub fn forces_uncond(sampler: &str) -> bool {
    sampler.ends_with("_cfg_pp")
}

/// `ModelSamplingDiscreteFlow` with multiplier 1, `ModelSamplingFlux`, or a flow model whose timestep walks an explicit
/// sigma grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelSampling {
    Flow { shift: f64 },
    Flux { shift: f64 },
    Grid(SigmaGrid),
}

/// `MiniMaxH3ImageModelSampling`: sigma is linear in the grid index, so the stock schedulers at the grid's step count
/// land on the grid points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmaGrid {
    /// ascending, from sigma 0 to 1
    points: [f32; 17],
    /// intervals: `points[..=n]` are used
    n: usize,
}

impl SigmaGrid {
    /// A grid from its sigmas in either order, at most 16 intervals.
    pub fn new(sigmas: &[f32]) -> Result<Self> {
        let n = sigmas.len().saturating_sub(1);
        if !(1..=16).contains(&n) {
            bail!("a sigma grid of {} points (2 to 17 are supported)", sigmas.len());
        }
        let mut points = [0f32; 17];
        points[..=n].copy_from_slice(sigmas);
        if points[0] > points[n] {
            points[..=n].reverse();
        }
        Ok(SigmaGrid { points, n })
    }

    /// The grid's sigmas from 1 down to 0.
    pub fn descending(&self) -> V {
        self.points[..=self.n].iter().rev().copied().collect()
    }

    fn sigma(&self, t: f32) -> f32 {
        let pos = t.clamp(0.0, 1.0) * self.n as f32;
        let i = (pos.floor() as usize).min(self.n - 1);
        let (a, b) = (self.points[i], self.points[i + 1]);
        a + (pos - i as f32) * (b - a)
    }
}

fn flux_shift(mu: f64, t: f32) -> f32 {
    let e = mu.exp() as f32;
    e / (e + (1.0 / t - 1.0))
}

impl ModelSampling {
    pub fn sigma(self, t: f32) -> f32 {
        match self {
            ModelSampling::Flow { shift } if shift == 1.0 => t,
            ModelSampling::Flow { shift } => shift as f32 * t / (1.0 + (shift - 1.0) as f32 * t),
            ModelSampling::Flux { shift } => flux_shift(shift, t),
            ModelSampling::Grid(g) => g.sigma(t),
        }
    }

    fn table(self) -> V {
        let n = self.table_len();
        (1..=n).map(|i| self.sigma(i as f32 / n as f32)).collect()
    }

    fn table_len(self) -> usize {
        if matches!(self, ModelSampling::Flux { .. }) { 10000 } else { 1000 }
    }

    /// `model_sampling.sigma_min`: the table's first entry.
    pub fn sigma_min(self) -> f32 {
        self.sigma(1.0 / self.table_len() as f32)
    }

    pub fn percent_to_sigma(self, percent: f64) -> f32 {
        if percent <= 0.0 {
            return 1.0;
        }
        if percent >= 1.0 {
            return 0.0;
        }
        let t = 1.0 - percent;
        if let ModelSampling::Grid(g) = self {
            return g.sigma(t as f32);
        }
        (match self {
            ModelSampling::Flow { shift } if shift == 1.0 => t,
            ModelSampling::Flow { shift } => shift * t / (1.0 + (shift - 1.0) * t),
            ModelSampling::Flux { shift } => shift.exp() / (shift.exp() + (1.0 / t - 1.0)),
            ModelSampling::Grid(_) => unreachable!(),
        }) as f32
    }
}

/// `SetLatentNoiseMask`, as `KSamplerX0Inpaint` applies it: outside the mask each model input is the latent noised to
/// the step's sigma, and each prediction is the latent.
#[derive(Clone, Debug)]
pub struct Inpaint {
    /// One value per latent value (`comfy.utils.reshape_mask`): 1 denoises, 0 keeps the latent.
    pub mask: V,
    /// `DifferentialDiffusion`'s strength: each step denoises where the mask reaches the step's place in the schedule.
    pub differential: Option<f32>,
}

impl Inpaint {
    /// The mask at `sigma` of a run over `sigmas`: `DifferentialDiffusion.forward` when it applies.
    pub fn mask_at(&self, ms: ModelSampling, sigmas: &[f32], sigma: f32) -> std::borrow::Cow<'_, [f32]> {
        let Some(strength) = self.differential else { return std::borrow::Cow::Borrowed(&self.mask) };
        // a flow model's timestep is its sigma (times a multiplier of 1)
        let to = ms.sigma_min().max(sigmas[sigmas.len() - 1]);
        let threshold = (sigma - to) / (sigmas[0] - to);
        let binary = |m: f32| if m >= threshold { 1.0 } else { 0.0 };
        std::borrow::Cow::Owned(if strength != 0.0 && strength < 1.0 {
            self.mask.iter().map(|&m| strength * binary(m) + (1.0 - strength) * m).collect()
        } else {
            self.mask.iter().map(|&m| binary(m)).collect()
        })
    }

    /// The model's input at `sigma`: `x` where `mask` denoises, else the latent noised to `sigma` (`noise_scaling`).
    pub fn input(mask: &[f32], x: &[f32], sigma: f32, noise: &[f32], latent: &[f32]) -> V {
        (0..x.len()).map(|j| x[j] * mask[j] + (sigma * noise[j] + (1.0 - sigma) * latent[j]) * (1.0 - mask[j])).collect()
    }

    /// The prediction the sampler sees: the model's where `mask` denoises, else the latent.
    pub fn prediction(mask: &[f32], denoised: &[f32], latent: &[f32]) -> V {
        (0..denoised.len()).map(|j| denoised[j] * mask[j] + latent[j] * (1.0 - mask[j])).collect()
    }
}

/// The sigmas of a run from `steps` scheduled ones: `KSampler.set_steps` for `denoise`, then `KSampler.sample`'s step
/// range, with the last sigma set to 0 unless the run returns with leftover noise. `schedule(n)` gives `n` steps' sigmas.
/// `None` when no step is left, and the latent comes back as it is.
pub fn ksampler_sigmas(steps: usize, denoise: f64, start: usize, end: usize, leftover_noise: bool, schedule: impl Fn(usize) -> Result<V>)
    -> Result<Option<V>> {
    let mut s = if denoise > 0.9999 {
        schedule(steps)?
    } else if denoise <= 0.0 {
        V::new()
    } else {
        let all = schedule((steps as f64 / denoise) as usize)?;
        all[all.len().saturating_sub(steps + 1)..].to_vec()
    };
    if end + 1 < s.len() {
        s.truncate(end + 1);
        if !leftover_noise {
            *s.last_mut().unwrap() = 0.0;
        }
    }
    if start + 1 >= s.len() {
        return Ok(None);
    }
    Ok(Some(s.split_off(start)))
}

/// `torch.linspace` in f32.
fn linspace(a: f32, b: f32, n: usize) -> V {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f32;
    (0..n).map(|i| if i < n / 2 { a + step * i as f32 } else { b - step * (n - i - 1) as f32 }).collect()
}

fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 6] = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let ser = G.iter().enumerate().fold(1.000000000190015, |s, (j, c)| s + c / (x + 1.0 + j as f64));
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Continued fraction of the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    d = 1.0 / if d.abs() < FPMIN { FPMIN } else { d };
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        for aa in [m * (b - m) * x / ((qam + 2.0 * m) * (a + 2.0 * m)), -(a + m) * (qab + m) * x / ((a + 2.0 * m) * (qap + 2.0 * m))] {
            d = 1.0 + aa * d;
            d = 1.0 / if d.abs() < FPMIN { FPMIN } else { d };
            c = 1.0 + aa / c;
            c = if c.abs() < FPMIN { FPMIN } else { c };
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) { front * beta_cf(a, b, x) / a } else { 1.0 - front * beta_cf(b, a, 1.0 - x) / b }
}

/// `scipy.stats.beta.ppf`.
fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1.0);
    if p <= 0.0 || p >= 1.0 {
        return p.clamp(0.0, 1.0);
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if beta_cdf(mid, a, b) < p { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// Inverse normal CDF: Newton on the Taylor series of the CDF, enough for |x| < 7.
fn ndtri(u: f64) -> f64 {
    if u <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if u >= 1.0 {
        return f64::INFINITY;
    }
    let cdf = |x: f64| {
        let (mut term, mut sum) = (x, x);
        for k in 1..400 {
            term *= x * x / (2 * k + 1) as f64;
            sum += term;
        }
        0.5 + sum * (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    };
    let mut x = 0.0;
    for _ in 0..200 {
        let dx = (cdf(x) - u) * (2.0 * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
        x -= dx.clamp(-1.0, 1.0);
        if dx.abs() < 1e-14 {
            break;
        }
    }
    x
}

fn normal_scheduler(ms: ModelSampling, table: &[f32], steps: usize, sgm: bool) -> V {
    let (start, end) = (table[table.len() - 1], table[0]);
    let mut steps = steps;
    let mut append_zero = true;
    let ts = if sgm {
        let mut t = linspace(start, end, steps + 1);
        t.pop();
        t
    } else {
        if (ms.sigma(end) as f64).abs() <= 1e-5 {
            steps += 1;
            append_zero = false;
        }
        linspace(start, end, steps)
    };
    let mut s: V = ts.iter().map(|&t| ms.sigma(t)).collect();
    if append_zero {
        s.push(0.0);
    }
    s
}

fn karras(n: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> V {
    let (min_inv, max_inv) = (sigma_min.powf(1.0 / rho) as f32, sigma_max.powf(1.0 / rho) as f32);
    let mut s: V = linspace(0.0, 1.0, n).iter().map(|r| (max_inv + r * (min_inv - max_inv)).powf(rho as f32)).collect();
    s.push(0.0);
    s
}

fn exponential(n: usize, sigma_min: f64, sigma_max: f64) -> V {
    let mut s: V = linspace(sigma_max.ln() as f32, sigma_min.ln() as f32, n).iter().map(|v| v.exp()).collect();
    s.push(0.0);
    s
}

fn comfy_scheduler(ms: ModelSampling, name: &str, steps: usize) -> Result<V> {
    let table = ms.table();
    let n = table.len();
    let (sigma_min, sigma_max) = (table[0], table[n - 1]);
    Ok(match name {
        "simple" => {
            let ss = n as f64 / steps as f64;
            let mut s: V = (0..steps).map(|x| table[n - 1 - (x as f64 * ss) as usize]).collect();
            s.push(0.0);
            s
        }
        "sgm_uniform" => normal_scheduler(ms, &table, steps, true),
        "normal" => normal_scheduler(ms, &table, steps, false),
        "karras" => karras(steps, sigma_min as f64, sigma_max as f64, 7.0),
        "exponential" => exponential(steps, sigma_min as f64, sigma_max as f64),
        "ddim_uniform" => {
            let close = (table[1] as f64).abs() <= 1e-5;
            let mut s = if close { Vec::new() } else { vec![0.0] };
            let ss = (n / if close { steps + 1 } else { steps }).max(1);
            s.extend((1..n).step_by(ss).map(|x| table[x]));
            s.reverse();
            s
        }
        "beta" => {
            let total = (n - 1) as f64;
            let mut s = Vec::new();
            let mut last = -1.0;
            for i in 0..steps {
                let t = (beta_ppf(1.0 - i as f64 * (1.0 / steps as f64), 0.6, 0.6) * total).round_ties_even();
                if t != last {
                    s.push(table[t as usize]);
                }
                last = t;
            }
            s.push(0.0);
            s
        }
        "linear_quadratic" => {
            let threshold = 0.025f64;
            let sched: Vec<f64> = if steps == 1 {
                vec![1.0, 0.0]
            } else {
                let linear = steps / 2;
                let lf = linear as f64;
                let mut v: Vec<f64> = (0..linear).map(|i| i as f64 * threshold / lf).collect();
                let diff = lf - threshold * steps as f64;
                let quad = (steps - linear) as f64;
                let qc = diff / (lf * quad * quad);
                let lc = threshold / lf - 2.0 * diff / (quad * quad);
                let c = qc * lf * lf;
                v.extend((linear..steps).map(|i| qc * (i * i) as f64 + lc * i as f64 + c));
                v.push(1.0);
                v.iter().map(|x| 1.0 - x).collect()
            };
            sched.iter().map(|&x| x as f32 * sigma_max).collect()
        }
        "kl_optimal" => {
            let (lo, hi) = ((sigma_min as f64).atan() as f32, (sigma_max as f64).atan() as f32);
            let mut s: V = (0..steps).map(|i| {
                let a = i as f32 / (steps - 1) as f32;
                (a * lo + (1.0 - a) * hi).tan()
            }).collect();
            s.push(0.0);
            s
        }
        s => bail!("unknown scheduler {s}"),
    })
}

/// `get_schedule` of nodes_flux.py.
fn flux2_sigmas(steps: usize, seq_len: f64) -> V {
    let (a1, b1) = (8.73809524e-05, 1.89833333);
    let (a2, b2) = (0.00016927, 0.45666666);
    let seq = seq_len.round_ties_even();
    let mu = if seq > 4300.0 {
        a2 * seq + b2
    } else {
        let (m200, m10) = (a2 * seq + b2, a1 * seq + b1);
        let a = (m200 - m10) / 190.0;
        a * steps as f64 + m200 - 200.0 * a
    };
    let e = mu.exp() as f32;
    linspace(1.0, 0.0, steps + 1).iter().map(|&t| e / (e + (1.0 / t - 1.0))).collect()
}

/// `ideogram4_sigmas`.
fn ideogram4_sigmas(steps: usize, width: f64, height: f64, mu: f64, std: f64) -> V {
    const LOGSNR_MIN: f64 = -18.0;
    const LOGSNR_MAX: f64 = 18.0;
    let mean = mu + 0.5 * ((width * height) / (512.0 * 512.0)).ln();
    let (t_min, t_max) = (1.0 / (1.0 + (0.5 * LOGSNR_MAX).exp()), 1.0 / (1.0 + (0.5 * LOGSNR_MIN).exp()));
    let n = steps + 1;
    let mut s: V = (0..n).rev().map(|i| {
        let u = i as f64 / steps as f64;
        let z = mean + std * ndtri(u);
        let expit = if z.is_infinite() { if z > 0.0 { 1.0 } else { 0.0 } } else { 1.0 / (1.0 + (-z).exp()) };
        (1.0 - (1.0 - expit).clamp(t_min, t_max)) as f32
    }).collect();
    s[steps] = 0.0;
    s
}

/// `loglinear_interp` over the SD1 list, which `SwarmKSampler` falls back to for these models.
fn align_your_steps(steps: usize) -> V {
    const SD1: [f64; 11] = [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904, 0.6523686016,
        0.3977456272, 0.1515232662, 0.0291671582];
    let n = steps + 1;
    let mut s: Vec<f64> = if n == SD1.len() {
        SD1.to_vec()
    } else {
        let k = SD1.len() - 1;
        let ys: Vec<f64> = SD1.iter().rev().map(|v| v.ln()).collect();
        let mut v: Vec<f64> = (0..n).map(|i| {
            let x = if n == 1 { 0.0 } else { i as f64 * k as f64 / (n - 1) as f64 };
            let j = (x.floor() as usize).min(k - 1);
            (ys[j] + (ys[j + 1] - ys[j]) * (x - j as f64)).exp()
        }).collect();
        v.reverse();
        v
    };
    s[steps] = 0.0;
    s.iter().map(|&v| v as f32).collect()
}

/// The sigma overrides `SwarmKSampler` applies to `karras` and `exponential`, when both are set.
#[derive(Clone, Copy, Debug)]
pub struct SigmaRange {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rho: f64,
}

/// `KSampler.calculate_sigmas` and `SwarmKSampler`'s extra schedulers. `(lw, lh)` is the latent's size.
pub fn sigmas(ms: ModelSampling, sampler: &str, scheduler: &str, steps: usize, (lw, lh): (usize, usize), range: Option<SigmaRange>) -> Result<V> {
    let (w, h) = (lw as f64 * 16.0, lh as f64 * 16.0);
    match scheduler {
        "turbo" => {
            let mut s: V = (0..steps.min(10)).map(|i| ms.sigma(999.0 - 100.0 * i as f32)).collect();
            s.push(0.0);
            return Ok(s);
        }
        "flux2" => return Ok(flux2_sigmas(steps, w * h / (16.0 * 16.0))),
        "align_your_steps" => return Ok(align_your_steps(steps)),
        "ideogram4" => return Ok(ideogram4_sigmas(steps, w, h, 0.0, 1.75)),
        "ideogram4turbo" => return Ok(ideogram4_sigmas(steps, w, h, 0.5, 1.75)),
        "hyperflow" => return hyperflow_sigmas(ms, steps),
        _ => {}
    }
    let scheduler = if scheduler == "ltxv" { "simple" } else { scheduler };
    if let (Some(r), "karras" | "exponential") = (range, scheduler) {
        let extra = matches!(sampler, "dpm_2" | "dpm_2_ancestral");
        let n = steps + extra as usize;
        let mut s = if scheduler == "karras" { karras(n, r.sigma_min, r.sigma_max, r.rho) } else { exponential(n, r.sigma_min, r.sigma_max) };
        if extra {
            s.remove(s.len() - 2);
        }
        return Ok(s);
    }
    let discard = matches!(sampler, "dpm_2" | "dpm_2_ancestral" | "uni_pc" | "uni_pc_bh2");
    let mut s = comfy_scheduler(ms, scheduler, steps + discard as usize)?;
    if discard {
        s.remove(s.len() - 2);
    }
    Ok(s)
}

/// `numpy.random.SeedSequence(entropy, spawn_key=(key, depth) or (), pool_size=24).generate_state(n)`.
fn seed_state(entropy: u64, spawn: Option<(u64, u32)>, n: usize) -> Vec<u32> {
    const INIT_A: u32 = 0x43b0d7e5;
    const MULT_A: u32 = 0x931e8875;
    const MULT_B: u32 = 0x58f38ded;
    const INIT_B: u32 = 0x8b51f9dd;
    const MIX_L: u32 = 0xca01f9dd;
    const MIX_R: u32 = 0x4973f715;
    const POOL: usize = 24;
    fn limbs(mut v: u64, out: &mut Vec<u32>) {
        if v == 0 {
            out.push(0);
        }
        while v > 0 {
            out.push(v as u32);
            v >>= 32;
        }
    }
    let mut data = Vec::new();
    limbs(entropy, &mut data);
    if let Some((key, depth)) = spawn {
        data.resize(POOL.max(data.len()), 0);
        limbs(key, &mut data);
        limbs(depth as u64, &mut data);
    }
    let mut hc = INIT_A;
    let mut hashmix = |v: u32| {
        let mut v = v ^ hc;
        hc = hc.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hc);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| {
        let r = x.wrapping_mul(MIX_L).wrapping_sub(y.wrapping_mul(MIX_R));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; POOL];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(data.get(i).copied().unwrap_or(0));
    }
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src]));
            }
        }
    }
    for &d in data.iter().skip(POOL) {
        for dst in 0..POOL {
            pool[dst] = mix(pool[dst], hashmix(d));
        }
    }
    let mut hc = INIT_B;
    (0..n).map(|i| {
        let mut v = pool[i % POOL] ^ hc;
        hc = hc.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hc);
        v ^ (v >> 16)
    }).collect()
}

/// Where a run's images sit in their batch: ComfyUI draws the sampler's noise as one tensor for the whole batch.
#[derive(Clone, Copy)]
pub struct Batch {
    /// The run's first image.
    pub index: usize,
    /// Images in the whole batch.
    pub size: usize,
    /// Images in this run, from `index`; its latent holds them one after another.
    pub images: usize,
    /// [`crate::ffi::gpu_randn_threads`]
    pub threads: usize,
}

impl Batch {
    /// One image.
    pub fn single(threads: usize) -> Self {
        Batch { index: 0, size: 1, images: 1, threads }
    }

    /// For a run of `n` values: where they start in the whole batch's noise, and its length.
    fn span(self, n: usize) -> (usize, usize) {
        let per = n / self.images;
        (self.index * per, self.size * per)
    }
}

struct Interval {
    start: f64,
    end: f64,
    parent: usize,
    is_left: bool,
    midway: Option<f64>,
    children: (usize, usize),
    spawn_key: u64,
    depth: u32,
    w_seed: u32,
}

/// `torchsde.BrownianTree` (dyadic, tol 1e-6, pool 24), without the Levy areas nothing here reads.
struct BrownianTree {
    entropy: u64,
    cpu: bool,
    n: usize,
    batch: Batch,
    nodes: Vec<Interval>,
    cache: Vec<(usize, V)>,
}

fn round6(x: f64) -> f64 {
    format!("{x:.6}").parse().unwrap()
}

impl BrownianTree {
    const CACHE: usize = 45;

    fn new(t0: f64, t1: f64, entropy: u64, cpu: bool, n: usize, batch: Batch) -> Self {
        let root = Interval { start: round6(t0), end: round6(t1), parent: 0, is_left: false, midway: None, children: (0, 0), spawn_key: 0, depth: 0, w_seed: 0 };
        let mut tree = BrownianTree { entropy, cpu, n, batch, nodes: vec![root], cache: Vec::new() };
        let z = tree.randn(seed_state(entropy, None, 1)[0]);
        let s = (t1 - t0).sqrt() as f32;
        tree.cache.push((0, z.iter().map(|v| v * s).collect()));
        tree
    }

    fn randn(&self, seed: u32) -> V {
        let (n, b) = (self.n, self.batch);
        let (start, numel) = b.span(n);
        if self.cpu {
            torch_randn(seed as u64, numel)[start..start + n].to_vec()
        } else {
            CudaRandn::new(seed as u64, numel, b.threads).next(start, n)
        }
    }

    fn split_exact(&mut self, i: usize, midway: f64) {
        let (key, depth) = if i == 0 {
            (0, 0)
        } else {
            let p = &self.nodes[self.nodes[i].parent];
            (2 * p.spawn_key + !self.nodes[i].is_left as u64, p.depth + 1)
        };
        let m = round6(midway);
        let (start, end) = (self.nodes[i].start, self.nodes[i].end);
        let (l, r) = (self.nodes.len(), self.nodes.len() + 1);
        let child = |start, end, is_left| Interval { start, end, parent: i, is_left, midway: None, children: (0, 0), spawn_key: 0, depth: 0, w_seed: 0 };
        self.nodes.push(child(start, m, true));
        self.nodes.push(child(m, end, false));
        let w_seed = seed_state(self.entropy, Some((key, depth)), 1)[0];
        let node = &mut self.nodes[i];
        (node.midway, node.children, node.spawn_key, node.depth, node.w_seed) = (Some(m), (l, r), key, depth, w_seed);
    }

    fn split(&mut self, mut i: usize, midway: f64) {
        loop {
            let half = 0.5 * (self.nodes[i].end + self.nodes[i].start);
            self.split_exact(i, half);
            let (m, (l, r)) = (self.nodes[i].midway.unwrap(), self.nodes[i].children);
            if midway > m {
                i = r;
            } else if midway < m {
                i = l;
            } else {
                return;
            }
        }
    }

    fn locate(&mut self, i: usize, ta: f64, tb: f64, out: &mut Vec<usize>) {
        let (start, end) = (self.nodes[i].start, self.nodes[i].end);
        if ta < start || tb > end {
            return self.locate(self.nodes[i].parent, ta, tb, out);
        }
        if ta == start && tb == end {
            out.push(i);
            return;
        }
        let Some(m) = self.nodes[i].midway else {
            if ta == start {
                self.split(i, tb);
                return self.locate(self.nodes[i].children.0, ta, tb, out);
            }
            self.split(i, ta);
            return self.locate(self.nodes[i].children.1, ta, tb, out);
        };
        let (l, r) = self.nodes[i].children;
        if tb <= m {
            self.locate(l, ta, tb, out)
        } else if ta >= m {
            self.locate(r, ta, tb, out)
        } else {
            self.locate(l, ta, m, out);
            self.locate(r, m, tb, out)
        }
    }

    fn increment(&mut self, i: usize) -> V {
        if let Some(c) = self.cache.iter().position(|(k, _)| *k == i) {
            let hit = self.cache.remove(c);
            self.cache.push(hit);
            return self.cache.last().unwrap().1.clone();
        }
        let p = self.nodes[i].parent;
        let w = self.increment(p);
        let (ps, pe, seed) = (self.nodes[p].start, self.nodes[p].end, self.nodes[p].w_seed);
        let mid = self.nodes[p].midway.unwrap();
        let hr = 1.0 / (pe - ps);
        let (left, right) = (mid - ps, pe - mid);
        let (l, h, root) = (left as f32, hr as f32, (left * right * hr).sqrt() as f32);
        let noise = self.randn(seed);
        let left_w = w.iter().zip(&noise).map(|(w, z)| l * w * h + root * z);
        let out: V = if self.nodes[i].is_left { left_w.collect() } else { w.iter().zip(left_w).map(|(w, lw)| w - lw).collect() };
        if self.cache.len() >= Self::CACHE {
            // the root's increment can't be recomputed
            let oldest = self.cache.iter().position(|(k, _)| *k != 0).unwrap();
            self.cache.remove(oldest);
        }
        self.cache.push((i, out.clone()));
        out
    }

    fn query(&mut self, ta: f64, tb: f64) -> V {
        let (start, end) = (self.nodes[0].start, self.nodes[0].end);
        let (ta, tb) = (round6(ta.clamp(start, end)), round6(tb.clamp(start, end)));
        if ta == tb {
            return vec![0.0; self.n];
        }
        let mut out = Vec::new();
        self.locate(0, ta, tb, &mut out);
        let mut w = self.increment(out[0]);
        for &i in &out[1..] {
            let wi = self.increment(i);
            w.iter_mut().zip(&wi).for_each(|(a, b)| *a += b);
        }
        w
    }
}

enum Noise {
    Default(CudaRandn, usize, usize),
    Tree(Box<BrownianTree>),
}

impl Noise {
    fn default(seed: u64, n: usize, b: Batch) -> Self {
        let (start, numel) = b.span(n);
        Noise::Default(CudaRandn::new(seed, numel, b.threads), start, n)
    }

    fn tree(sigmas: &[f32], seed: u64, cpu: bool, n: usize, b: Batch) -> Self {
        let min = sigmas.iter().copied().filter(|s| *s > 0.0).fold(f32::INFINITY, f32::min);
        let max = sigmas.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Noise::Tree(Box::new(BrownianTree::new(min as f64, max as f64, seed, cpu, n, b)))
    }

    fn next(&mut self, sigma: f32, sigma_next: f32) -> V {
        match self {
            Noise::Default(g, start, n) => g.next(*start, *n),
            Noise::Tree(t) => {
                let (a, b, sign) = if sigma < sigma_next { (sigma, sigma_next, 1.0) } else { (sigma_next, sigma, -1.0) };
                let scale = (sigma_next - sigma).abs().sqrt();
                t.query(a as f64, b as f64).iter().map(|v| v * sign / scale).collect()
            }
        }
    }
}

/// Linear combination, left to right.
fn lc(terms: &[(&[f32], f32)]) -> V {
    let mut out = vec![0f32; terms[0].0.len()];
    for (t, c) in terms {
        out.iter_mut().zip(*t).for_each(|(o, v)| *o += v * c);
    }
    out
}

fn to_d(x: &[f32], den: &[f32], sigma: f32) -> V {
    x.iter().zip(den).map(|(x, d)| (x - d) / sigma).collect()
}

/// `torch.lerp`.
fn lerp(a: f32, b: f32, w: f32) -> f32 {
    if w.abs() < 0.5 { a + w * (b - a) } else { b - (b - a) * (1.0 - w) }
}

fn lerp_v(a: &[f32], b: &[f32], w: f32) -> V {
    a.iter().zip(b).map(|(a, b)| lerp(*a, *b, w)).collect()
}

fn ancestral_step(from: f32, to: f32, eta: f32) -> (f32, f32) {
    if eta == 0.0 {
        return (to, 0.0);
    }
    let v = eta * (to * to * (from * from - to * to) / (from * from)).powf(0.5);
    let up = if v < to { v } else { to };
    ((to * to - up * up).powf(0.5), up)
}

fn lambda(sigma: f32) -> f32 {
    -(sigma / (1.0 - sigma)).ln()
}

fn sigma_of(lambda: f32) -> f32 {
    1.0 / (1.0 + lambda.exp())
}

fn offset_first(sigmas: &[f32], ms: ModelSampling) -> V {
    let mut s = sigmas.to_vec();
    if s.len() > 1 && s[0] >= 1.0 {
        s[0] = ms.percent_to_sigma(1e-4);
    }
    s
}

fn phi1(h: f32) -> f32 {
    h.exp_m1()
}

fn phi2(h: f32) -> f32 {
    (h.exp_m1() - h) / h
}

fn nan_to_num(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(f32::MIN, f32::MAX) }
}

/// Lagrange basis integral in `linear_multistep_coeff`: 3-point Gauss-Legendre is exact up to order 4.
fn lms_coeff(order: usize, t: &[f32], i: usize, j: usize) -> f32 {
    let t: Vec<f64> = t.iter().map(|&v| v as f64).collect();
    let f = |tau: f64| {
        let mut p = 1.0;
        for k in 0..order {
            if k != j {
                p *= (tau - t[i - k]) / (t[i - j] - t[i - k]);
            }
        }
        p
    };
    let (a, b) = (t[i], t[i + 1]);
    let (c, m) = (0.5 * (a + b), 0.5 * (b - a));
    let r = (0.6f64).sqrt();
    (m * (5.0 * f(c - m * r) + 8.0 * f(c) + 5.0 * f(c + m * r)) / 9.0) as f32
}

/// Solve A x = b with partial pivoting, in f32 like `torch.linalg.solve` on these tensors.
fn solve(mut a: Vec<V>, mut b: V) -> V {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..n {
                let v = a[c][k];
                a[r][k] -= f * v;
            }
            let v = b[c];
            b[r] -= f * v;
        }
    }
    for c in (0..n).rev() {
        for k in c + 1..n {
            let v = a[c][k] * b[k];
            b[c] -= v;
        }
        b[c] /= a[c][c];
    }
    b
}

/// `compute_stochastic_adams_b_coeffs` without the simple order 2 scheme.
fn sa_b_coeffs(sigma_next: f32, lambdas: &[f32], lambda_s: f32, lambda_t: f32, tau: f32) -> V {
    let n = lambdas.len();
    let tau_mul = 1.0 + tau * tau;
    let h = lambda_t - lambda_s;
    let terms: V = (0..n).map(|p| lambda_t.powi(p as i32) - lambda_s.powi(p as i32) * (-tau_mul * h).exp()).collect();
    let mut coeffs = vec![0f32; n];
    for (i, c) in coeffs.iter_mut().enumerate() {
        let mut ratio = 1f32;
        for j in (0..=i).rev() {
            *c += ratio * terms[j];
            if j > 0 {
                ratio *= -(j as f32) / tau_mul;
            }
        }
    }
    let vander: Vec<V> = (0..n).map(|p| lambdas.iter().map(|l| l.powi(p as i32)).collect()).collect();
    let alpha_t = sigma_next * lambda_t.exp();
    solve(vander, coeffs).iter().map(|v| alpha_t * v).collect()
}

/// `get_deis_coeff_list` in `tab` mode. `t_steps` still ends at sigma 0, whose entry is left empty.
fn deis_coeffs(sigmas: &[f32], max_order: usize) -> Vec<V> {
    const N: usize = 10000;
    let (eps, smin, smax) = (1e-3f32, 0.002f32, 80f32);
    let beta_d = 2.0 * ((smin * smin + 1.0).ln() / eps - (smax * smax + 1.0).ln()) / (eps - 1.0);
    let beta_min = (smax * smax + 1.0).ln() - 0.5 * beta_d;
    let t: V = sigmas.iter().map(|s| ((beta_min * beta_min + 2.0 * beta_d * (s * s + 1.0).ln()).sqrt() - beta_min) / beta_d).collect();
    let (b0, b1) = (beta_min, beta_d + beta_min);
    (0..t.len() - 1).map(|i| {
        let order = (i + 1).min(max_order);
        if order == 1 || sigmas[i + 1] <= 0.0 {
            return Vec::new();
        }
        let taus = linspace(t[i], t[i + 1], N);
        let dtau = (t[i + 1] - t[i]) / N as f32;
        let prev: V = (0..order).map(|k| t[i - k]).collect();
        let integrand: Vec<f32> = taus.iter().map(|&tau| {
            let alpha = (-0.5 * tau * tau * (b1 - b0) - tau * b0).exp();
            -0.5 * (1.0 / alpha * alpha * (-tau * (b1 - b0) - b0)) / (alpha * (1.0 - alpha)).sqrt()
        }).collect();
        (0..order).map(|j| {
            let sum: f64 = taus.iter().zip(&integrand).map(|(&tau, &g)| {
                let mut poly = 1f32;
                for k in 0..order {
                    if k != j {
                        poly *= (tau - prev[k]) / (prev[j] - prev[k]);
                    }
                }
                (g * poly) as f64
            }).sum();
            sum as f32 * dtau
        }).collect()
    }).collect()
}

/// `sampler_cfg_function`: replaces the CFG mix. A model has one, the last set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CfgFunction {
    /// `RescaleCFG`, in its flow branch.
    Rescale { multiplier: f32 },
    /// `RenormCFG`.
    Renorm { cfg_trunc: f32, renorm_cfg: f32 },
}

/// ComfyUI's guidance patches, each in the place `sampling_function` gives it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CfgTweaks {
    /// `TCFG`: before the mix.
    pub tangential: bool,
    pub function: Option<CfgFunction>,
    /// `CFGZeroStar`: after the mix.
    pub zero_star: bool,
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn std(v: &[f32]) -> f32 {
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt() as f32
}

/// `score_tangential_damping`: `uncond` along the first right singular vector of `[uncond, cond]`.
fn tangential_damping(cond: &[f32], uncond: &[f32]) -> V {
    let (uu, uc, cc) = (dot(uncond, uncond), dot(uncond, cond), dot(cond, cond));
    let largest = (uu + cc) / 2.0 + (((uu - cc) / 2.0).powi(2) + uc * uc).sqrt();
    // eigenvector of [[uu, uc], [uc, cc]]; the singular vector is the rows mixed by it
    let (a, b) = if uc != 0.0 { (largest - cc, uc) } else if uu >= cc { (1.0, 0.0) } else { (0.0, 1.0) };
    let v1: Vec<f64> = uncond.iter().zip(cond).map(|(u, c)| a * *u as f64 + b * *c as f64).collect();
    let norm = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return vec![0.0; uncond.len()];
    }
    let along: f64 = uncond.iter().zip(&v1).map(|(u, v)| *u as f64 * v).sum::<f64>() / (norm * norm);
    v1.iter().map(|v| (along * v) as f32).collect()
}

impl CfgTweaks {
    pub fn any(&self) -> bool {
        *self != Self::default()
    }

    /// `sampling_function` after the model: the guided prediction of one image and the uncond the
    /// sampler sees. Without an uncond evaluation ComfyUI's is zero, and TCFG skips itself.
    pub fn guide(&self, x: &[f32], cond: &[f32], uncond: Option<V>, scale: f32, sigma: f32) -> (V, V) {
        let evaluated = uncond.is_some();
        let mut uncond = uncond.unwrap_or_else(|| vec![0.0; cond.len()]);
        if self.tangential && evaluated {
            let score = |p: &[f32]| lc(&[(x, 1.0), (p, -1.0)]);
            uncond = lc(&[(x, 1.0), (&tangential_damping(&score(cond), &score(&uncond)), -1.0)]);
        }
        let mixed = || lc(&[(&uncond, 1.0 - scale), (cond, scale)]);
        let mut out = match self.function {
            None => mixed(),
            Some(CfgFunction::Rescale { multiplier }) => {
                let cfg = mixed();
                let ratio = std(cond) / std(&cfg).max(1e-8);
                lc(&[(&cfg, multiplier * ratio), (&cfg, 1.0 - multiplier)])
            }
            Some(CfgFunction::Renorm { cfg_trunc, renorm_cfg }) if sigma < cfg_trunc => {
                let mut half = mixed();
                if renorm_cfg > 0.0 {
                    let limit = dot(cond, cond).sqrt() as f32 * renorm_cfg;
                    let norm = dot(&half, &half).sqrt() as f32;
                    if norm >= limit {
                        half = lc(&[(&half, limit / norm)]);
                    }
                }
                half
            }
            Some(CfgFunction::Renorm { .. }) => cond.to_vec(),
        };
        if self.zero_star {
            let (c, u) = (lc(&[(x, 1.0), (cond, -1.0)]), lc(&[(x, 1.0), (&uncond, -1.0)]));
            let alpha = (dot(&c, &u) / (dot(&u, &u) + 1e-8)) as f32;
            out = lc(&[(&out, 1.0), (&uncond, alpha - 1.0 + scale * (1.0 - alpha))]);
        }
        (out, uncond)
    }
}

/// What a sampler wants from the uncond prediction of an evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uncond {
    /// nothing
    Skip,
    /// the prediction, if CFG evaluated one
    IfGuided,
    /// the prediction, evaluating it even at CFG 1
    Force,
}

pub struct Eval {
    pub denoised: V,
    pub uncond: Option<V>,
}

struct Run<'a> {
    ms: ModelSampling,
    eval: &'a mut dyn FnMut(&[f32], f32, Uncond) -> Result<Eval>,
    step: &'a mut dyn FnMut(usize, usize) -> Result<()>,
    total: usize,
    seed: u64,
    batch: Batch,
    n: usize,
}

impl Run<'_> {
    fn den(&mut self, x: &[f32], sigma: f32) -> Result<V> {
        Ok((self.eval)(x, sigma, Uncond::Skip)?.denoised)
    }

    fn pair(&mut self, x: &[f32], sigma: f32, mode: Uncond) -> Result<(V, V)> {
        let e = (self.eval)(x, sigma, mode)?;
        let uncond = match e.uncond {
            Some(u) => u,
            None if mode == Uncond::IfGuided => e.denoised.clone(),
            None => return Err(anyhow!("the model did not return an uncond prediction")),
        };
        Ok((e.denoised, uncond))
    }

    fn done(&mut self, i: usize) -> Result<()> {
        (self.step)((i + 1).min(self.total), self.total)
    }

    fn noise(&self) -> Noise {
        Noise::default(self.seed, self.n, self.batch)
    }

    fn euler(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let d = to_d(&x, &den, s[i]);
            x = lc(&[(&x, 1.0), (&d, s[i + 1] - s[i])]);
            self.done(i)?;
        }
        Ok(x)
    }

    fn euler_ancestral(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let mut noise = self.noise();
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den;
            } else {
                let (sd, a1, ad, renoise) = rf_step(s[i], s[i + 1]);
                let r = sd / s[i];
                x = lc(&[(&x, r), (&den, 1.0 - r)]);
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, a1 / ad), (&z, renoise)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn euler_ancestral_cfg_pp(&mut self, mut x: V, s: &[f32], eta: f32) -> Result<V> {
        let mut noise = self.noise();
        for i in 0..s.len() - 1 {
            let (den, und) = self.pair(&x, s[i], Uncond::Force)?;
            if s[i + 1] == 0.0 {
                x = den;
            } else {
                let alpha_s = s[i] * lambda(s[i]).exp();
                let alpha_t = s[i + 1] * lambda(s[i + 1]).exp();
                let scaled = lc(&[(&und, alpha_s)]);
                let d = to_d(&x, &scaled, s[i]);
                let (sd, su) = ancestral_step(s[i] / alpha_s, s[i + 1] / alpha_t, eta);
                x = lc(&[(&den, alpha_t), (&d, alpha_t * sd)]);
                if eta > 0.0 {
                    let z = noise.next(s[i], s[i + 1]);
                    x = lc(&[(&x, 1.0), (&z, alpha_t * su)]);
                }
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn heun(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let d = to_d(&x, &den, s[i]);
            let dt = s[i + 1] - s[i];
            if s[i + 1] == 0.0 {
                x = lc(&[(&x, 1.0), (&d, dt)]);
            } else {
                let x2 = lc(&[(&x, 1.0), (&d, dt)]);
                let den2 = self.den(&x2, s[i + 1])?;
                let d2 = to_d(&x2, &den2, s[i + 1]);
                let prime = lc(&[(&d, 0.5), (&d2, 0.5)]);
                x = lc(&[(&x, 1.0), (&prime, dt)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn heunpp2(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let end = s[s.len() - 1];
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let d = to_d(&x, &den, s[i]);
            let dt = s[i + 1] - s[i];
            if s[i + 1] == end {
                x = lc(&[(&x, 1.0), (&d, dt)]);
            } else if s[i + 2] == end {
                let x2 = lc(&[(&x, 1.0), (&d, dt)]);
                let den2 = self.den(&x2, s[i + 1])?;
                let d2 = to_d(&x2, &den2, s[i + 1]);
                let w2 = s[i + 1] / (2.0 * s[0]);
                let prime = lc(&[(&d, 1.0 - w2), (&d2, w2)]);
                x = lc(&[(&x, 1.0), (&prime, dt)]);
            } else {
                let x2 = lc(&[(&x, 1.0), (&d, dt)]);
                let den2 = self.den(&x2, s[i + 1])?;
                let d2 = to_d(&x2, &den2, s[i + 1]);
                let x3 = lc(&[(&x2, 1.0), (&d2, s[i + 2] - s[i + 1])]);
                let den3 = self.den(&x3, s[i + 2])?;
                let d3 = to_d(&x3, &den3, s[i + 2]);
                let w = 3.0 * s[0];
                let (w2, w3) = (s[i + 1] / w, s[i + 2] / w);
                let prime = lc(&[(&d, 1.0 - w2 - w3), (&d2, w2), (&d3, w3)]);
                x = lc(&[(&x, 1.0), (&prime, dt)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpm_2(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let d = to_d(&x, &den, s[i]);
            if s[i + 1] == 0.0 {
                x = lc(&[(&x, 1.0), (&d, s[i + 1] - s[i])]);
            } else {
                let mid = lerp(s[i].ln(), s[i + 1].ln(), 0.5).exp();
                let x2 = lc(&[(&x, 1.0), (&d, mid - s[i])]);
                let den2 = self.den(&x2, mid)?;
                let d2 = to_d(&x2, &den2, mid);
                x = lc(&[(&x, 1.0), (&d2, s[i + 1] - s[i])]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpm_2_ancestral(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let mut noise = self.noise();
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let (sd, a1, ad, renoise) = rf_step(s[i], s[i + 1]);
            let d = to_d(&x, &den, s[i]);
            if sd == 0.0 {
                x = lc(&[(&x, 1.0), (&d, sd - s[i])]);
            } else {
                let mid = lerp(s[i].ln(), sd.ln(), 0.5).exp();
                let x2 = lc(&[(&x, 1.0), (&d, mid - s[i])]);
                let den2 = self.den(&x2, mid)?;
                let d2 = to_d(&x2, &den2, mid);
                x = lc(&[(&x, 1.0), (&d2, sd - s[i])]);
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, a1 / ad), (&z, renoise)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn lms(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        const ORDER: usize = 4;
        let mut ds: Vec<V> = Vec::new();
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            ds.push(to_d(&x, &den, s[i]));
            if ds.len() > ORDER {
                ds.remove(0);
            }
            if s[i + 1] == 0.0 {
                x = den;
            } else {
                let order = (i + 1).min(ORDER);
                for (j, d) in ds.iter().rev().enumerate() {
                    let c = lms_coeff(order, s, i, j);
                    x = lc(&[(&x, 1.0), (d, c)]);
                }
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn cfgpp_ud10_ab(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        const HISTORY: f32 = 0.25;
        const UNCOND_HISTORY: f32 = 0.1;
        let mut old_d: Option<V> = None;
        let mut old_und_d = V::new();
        let mut old_den: Option<V> = None;
        for i in 0..s.len() - 1 {
            let (den, und) = self.pair(&x, s[i], Uncond::IfGuided)?;
            let alpha_s = s[i] * lambda(s[i]).exp();
            let alpha_t = if s[i + 1] != 0.0 { s[i + 1] * lambda(s[i + 1]).exp() } else { 1.0 };
            let scaled = lc(&[(&und, alpha_s)]);
            let cur_und_d = to_d(&x, &scaled, s[i]);
            let dt = s[i + 1] - s[i];
            let mut und_d = cur_und_d.clone();
            if i > 0 {
                let c = UNCOND_HISTORY * (dt / (s[i] - s[i - 1]));
                und_d = lc(&[(&und_d, 1.0), (&cur_und_d, c), (&old_und_d, -c)]);
            }
            let euler = lc(&[(&den, alpha_t), (&und_d, s[i + 1]), (&x, -1.0)]);
            let d = lc(&[(&euler, 1.0 / dt)]);
            let step = match &old_d {
                None => euler.clone(),
                Some(old) => {
                    let (c0, c1) = (lms_coeff(2, s, i, 0), lms_coeff(2, s, i, 1));
                    let hist = lc(&[(&d, c0), (old, c1)]);
                    lerp_v(&euler, &hist, HISTORY)
                }
            };
            x = lc(&[(&x, 1.0), (&step, 1.0)]);
            if let (0.0, Some(prev)) = (s[i + 1], &old_den) {
                let slope = lc(&[(&den, 1.0 / (s[i] - s[i - 1])), (prev, -1.0 / (s[i] - s[i - 1]))]);
                x = lc(&[(&den, 1.0), (&slope, -s[i])]);
            }
            old_d = Some(d);
            old_den = Some(den);
            old_und_d = cur_und_d;
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpm_eps(&mut self, x: &[f32], t: f32) -> Result<V> {
        let sigma = (-t).exp();
        let den = self.den(x, sigma)?;
        Ok(to_d(x, &den, sigma))
    }

    /// Returns the result and the eps at `t + r1 * h`.
    fn dpm_solver_2(&mut self, x: &[f32], t: f32, next: f32, r1: f32, eps: &[f32]) -> Result<(V, V)> {
        let sig = |t: f32| (-t).exp();
        let h = next - t;
        let s1 = t + r1 * h;
        let u1 = lc(&[(x, 1.0), (eps, -(sig(s1) * (r1 * h).exp_m1()))]);
        let eps_r1 = self.dpm_eps(&u1, s1)?;
        let (a, b) = (sig(next) * h.exp_m1(), sig(next) / (2.0 * r1) * h.exp_m1());
        Ok((lc(&[(x, 1.0), (eps, -a), (&eps_r1, -b), (eps, b)]), eps_r1))
    }

    fn dpm_solver_3(&mut self, x: &[f32], t: f32, next: f32, eps: &[f32], eps_r1: Option<V>) -> Result<V> {
        let sig = |t: f32| (-t).exp();
        let (r1, r2) = (1.0f32 / 3.0, 2.0f32 / 3.0);
        let h = next - t;
        let (s1, s2) = (t + r1 * h, t + r2 * h);
        let eps_r1 = match eps_r1 {
            Some(e) => e,
            None => {
                let u1 = lc(&[(x, 1.0), (eps, -(sig(s1) * (r1 * h).exp_m1()))]);
                self.dpm_eps(&u1, s1)?
            }
        };
        let c = sig(s2) * (r2 / r1) * ((r2 * h).exp_m1() / (r2 * h) - 1.0);
        let u2 = lc(&[(x, 1.0), (eps, -(sig(s2) * (r2 * h).exp_m1())), (&eps_r1, -c), (eps, c)]);
        let eps_r2 = self.dpm_eps(&u2, s2)?;
        let (a, b) = (sig(next) * h.exp_m1(), sig(next) / r2 * (h.exp_m1() / h - 1.0));
        Ok(lc(&[(x, 1.0), (eps, -a), (&eps_r2, -b), (eps, b)]))
    }

    fn dpm_fast(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let nfe = s.len() - 1;
        let sigma_min = if s[nfe] == 0.0 { s[nfe - 1] } else { s[nfe] };
        let (t_start, t_end) = (-s[0].ln(), -sigma_min.ln());
        let m = nfe / 3 + 1;
        let ts = linspace(t_start, t_end, m + 1);
        let mut orders = if nfe % 3 == 0 { vec![3; m - 2] } else { vec![3; m - 1] };
        if nfe % 3 == 0 {
            orders.extend([2, 1]);
        } else {
            orders.push(nfe % 3);
        }
        for (i, &order) in orders.iter().enumerate() {
            let (t, next) = (ts[i], ts[i + 1]);
            let eps = self.dpm_eps(&x, t)?;
            x = match order {
                1 => lc(&[(&x, 1.0), (&eps, -((-next).exp() * (next - t).exp_m1()))]),
                2 => self.dpm_solver_2(&x, t, next, 0.5, &eps)?.0,
                _ => self.dpm_solver_3(&x, t, next, &eps, None)?,
            };
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpm_adaptive(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        const ORDER: f64 = 3.0;
        let last = s.len() - 1;
        let sigma_min = if s[last] == 0.0 { s[last - 1] } else { s[last] };
        let (t_start, t_end) = (-s[0].ln(), -sigma_min.ln());
        let (rtol, atol) = (0.05f32, 0.0078f32);
        let (b1, b2) = (1.0 / ORDER, 0.0);
        let mut h = 0.05f64;
        let mut errs: Option<[f64; 3]> = None;
        let mut t = t_start;
        let mut x_prev = x.clone();
        let mut steps = 0;
        while t < t_end - 1e-5 {
            let next = t_end.min(t + h as f32);
            let eps = self.dpm_eps(&x, t)?;
            let (low, eps_r1) = self.dpm_solver_2(&x, t, next, 1.0 / 3.0, &eps)?;
            let high = self.dpm_solver_3(&x, t, next, &eps, Some(eps_r1))?;
            let sum: f64 = low.iter().zip(&high).zip(&x_prev).map(|((l, h), p)| {
                let delta = atol.max(rtol * l.abs().max(p.abs()));
                let e = (l - h) / delta;
                (e * e) as f64
            }).sum();
            let error = (sum.sqrt() as f32 / (x.len() as f64).sqrt() as f32) as f64;
            let inv = 1.0 / (error + 1e-8);
            let e = errs.get_or_insert([inv; 3]);
            e[0] = inv;
            let mut factor = e[0].powf(b1) * e[1].powf(b2) * e[2].powf(0.0);
            factor = 1.0 + (factor - 1.0).atan();
            if factor >= 0.81 {
                e[2] = e[1];
                e[1] = e[0];
                x_prev = low;
                x = high;
                t = next;
            }
            h *= factor;
            self.done(steps)?;
            steps += 1;
        }
        Ok(x)
    }

    fn dpmpp_2s_ancestral(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let mut noise = self.noise();
        let lambda_fn = |s: f32| ((1.0 - s) / s).ln();
        let sigma_fn = |l: f32| 1.0 / (l.exp() + 1.0);
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let (sd, a1, ad, renoise) = rf_step(s[i], s[i + 1]);
            if s[i + 1] == 0.0 {
                let d = to_d(&x, &den, s[i]);
                x = lc(&[(&x, 1.0), (&d, sd - s[i])]);
            } else {
                let sigma_s = if s[i] == 1.0 {
                    0.9999
                } else {
                    let (ti, td) = (lambda_fn(s[i]), lambda_fn(sd));
                    sigma_fn(ti + 0.5 * (td - ti))
                };
                let r = sigma_s / s[i];
                let u = lc(&[(&x, r), (&den, 1.0 - r)]);
                let d_i = self.den(&u, sigma_s)?;
                let r = sd / s[i];
                x = lc(&[(&x, r), (&d_i, 1.0 - r)]);
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, a1 / ad), (&z, renoise)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpmpp_2s_ancestral_cfg_pp(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let mut noise = self.noise();
        let sig = |t: f32| (-t).exp();
        let t_fn = |s: f32| -s.ln();
        for i in 0..s.len() - 1 {
            let (den, und) = self.pair(&x, s[i], Uncond::Force)?;
            let (sd, su) = ancestral_step(s[i], s[i + 1], 1.0);
            if sd == 0.0 {
                let d = to_d(&x, &und, s[i]);
                x = lc(&[(&den, 1.0), (&d, sd)]);
            } else {
                let (t, next) = (t_fn(s[i]), t_fn(sd));
                let h = next - t;
                let ss = t + 0.5 * h;
                let shifted = lc(&[(&x, 1.0), (&den, 1.0), (&und, -1.0)]);
                let x2 = lc(&[(&shifted, sig(ss) / sig(t)), (&den, -(-h * 0.5).exp_m1())]);
                // ComfyUI's hook has already replaced the uncond with this evaluation's
                let (den2, und2) = self.pair(&x2, sig(ss), Uncond::Force)?;
                let shifted = lc(&[(&x, 1.0), (&den, 1.0), (&und2, -1.0)]);
                x = lc(&[(&shifted, sig(next) / sig(t)), (&den2, -(-h).exp_m1())]);
            }
            if s[i + 1] > 0.0 {
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, 1.0), (&z, su)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpmpp_sde(&mut self, mut x: V, sigmas: &[f32], cpu: bool) -> Result<V> {
        const R: f32 = 0.5;
        let mut noise = Noise::tree(sigmas, self.seed, cpu, self.n, self.batch);
        let s = offset_first(sigmas, self.ms);
        let fac = 1.0 / (2.0 * R);
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den;
            } else {
                let (ls, lt) = (lambda(s[i]), lambda(s[i + 1]));
                let h = lt - ls;
                let ls1 = ls + R * h;
                let s1 = sigma_of(ls1);
                let (alpha_s, alpha_s1, alpha_t) = (s[i] * ls.exp(), s1 * ls1.exp(), s[i + 1] * lt.exp());

                let (sd, su) = ancestral_step((-ls).exp(), (-ls1).exp(), 1.0);
                let h_ = -sd.ln() - ls;
                let x2 = lc(&[(&x, (alpha_s1 / alpha_s) * (-h_).exp()), (&den, -(alpha_s1 * (-h_).exp_m1()))]);
                let z = noise.next(s[i], s1);
                let x2 = lc(&[(&x2, 1.0), (&z, alpha_s1 * su)]);
                let den2 = self.den(&x2, s1)?;

                let (sd, su) = ancestral_step((-ls).exp(), (-lt).exp(), 1.0);
                let h_ = -sd.ln() - ls;
                let dd = lc(&[(&den, 1.0 - fac), (&den2, fac)]);
                x = lc(&[(&x, (alpha_t / alpha_s) * (-h_).exp()), (&dd, -(alpha_t * (-h_).exp_m1()))]);
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, 1.0), (&z, alpha_t * su)]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpmpp_2m(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let t_fn = |s: f32| -s.ln();
        let sig = |t: f32| (-t).exp();
        let mut old: Option<V> = None;
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let (t, next) = (t_fn(s[i]), t_fn(s[i + 1]));
            let h = next - t;
            let ratio = sig(next) / sig(t);
            x = match &old {
                Some(o) if s[i + 1] != 0.0 => {
                    let r = (t - t_fn(s[i - 1])) / h;
                    let dd = lc(&[(&den, 1.0 + 1.0 / (2.0 * r)), (o, -(1.0 / (2.0 * r)))]);
                    lc(&[(&x, ratio), (&dd, -(-h).exp_m1())])
                }
                _ => lc(&[(&x, ratio), (&den, -(-h).exp_m1())]),
            };
            old = Some(den);
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpmpp_2m_cfg_pp(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let t_fn = |s: f32| -s.ln();
        let mut old: Option<V> = None;
        for i in 0..s.len() - 1 {
            let (den, und) = self.pair(&x, s[i], Uncond::Force)?;
            let (t, next) = (t_fn(s[i]), t_fn(s[i + 1]));
            let h = next - t;
            let e = (-h).exp();
            let mix = match &old {
                Some(o) if s[i + 1] != 0.0 => {
                    let r = (t - t_fn(s[i - 1])) / h;
                    let c = (-h).exp_m1() * (1.0 / (2.0 * r));
                    lc(&[(&und, -e), (&den, -c), (o, c)])
                }
                _ => lc(&[(&und, -e)]),
            };
            x = lc(&[(&den, 1.0), (&mix, 1.0), (&x, e)]);
            old = Some(und);
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpmpp_2m_sde(&mut self, mut x: V, sigmas: &[f32], cpu: bool, heun: bool) -> Result<V> {
        let mut noise = Noise::tree(sigmas, self.seed, cpu, self.n, self.batch);
        let s = offset_first(sigmas, self.ms);
        let mut old: Option<V> = None;
        let (mut h, mut h_last) = (0f32, 0f32);
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den.clone();
            } else {
                let (ls, lt) = (lambda(s[i]), lambda(s[i + 1]));
                h = lt - ls;
                let alpha_t = s[i + 1] * lt.exp();
                let e = -(-2.0 * h).exp_m1();
                x = lc(&[(&x, s[i + 1] / s[i] * (-h).exp()), (&den, alpha_t * e)]);
                if let Some(o) = &old {
                    let r = h_last / h;
                    let c = if heun { alpha_t * (e / (-2.0 * h) + 1.0) * (1.0 / r) } else { 0.5 * alpha_t * e * (1.0 / r) };
                    x = lc(&[(&x, 1.0), (&den, c), (o, -c)]);
                }
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, 1.0), (&z, s[i + 1] * e.sqrt())]);
            }
            old = Some(den);
            h_last = h;
            self.done(i)?;
        }
        Ok(x)
    }

    fn dpmpp_3m_sde(&mut self, mut x: V, sigmas: &[f32], cpu: bool) -> Result<V> {
        let mut noise = Noise::tree(sigmas, self.seed, cpu, self.n, self.batch);
        let s = offset_first(sigmas, self.ms);
        let (mut d1, mut d2): (Option<V>, Option<V>) = (None, None);
        let (mut h, mut h1, mut h2) = (None, None, None);
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den.clone();
            } else {
                let (ls, lt) = (lambda(s[i]), lambda(s[i + 1]));
                let hv: f32 = lt - ls;
                h = Some(hv);
                let h_eta = hv * 2.0;
                let alpha_t = s[i + 1] * lt.exp();
                x = lc(&[(&x, s[i + 1] / s[i] * (-hv).exp()), (&den, alpha_t * -(-h_eta).exp_m1())]);
                let phi_2 = (-h_eta).exp_m1() / h_eta + 1.0;
                if let (Some(a), Some(b), Some(ha), Some(hb)) = (&d1, &d2, h1, h2) {
                    let (r0, r1) = (ha / hv, hb / hv);
                    let d1_0 = lc(&[(&den, 1.0 / r0), (a, -1.0 / r0)]);
                    let d1_1 = lc(&[(a, 1.0 / r1), (b, -1.0 / r1)]);
                    let diff = lc(&[(&d1_0, 1.0), (&d1_1, -1.0)]);
                    let dd = lc(&[(&d1_0, 1.0), (&diff, r0 / (r0 + r1))]);
                    let dd2 = lc(&[(&diff, 1.0 / (r0 + r1))]);
                    let phi_3 = phi_2 / h_eta - 0.5;
                    x = lc(&[(&x, 1.0), (&dd, alpha_t * phi_2), (&dd2, -(alpha_t * phi_3))]);
                } else if let (Some(a), Some(ha)) = (&d1, h1) {
                    let r = ha / hv;
                    let d = lc(&[(&den, 1.0 / r), (a, -1.0 / r)]);
                    x = lc(&[(&x, 1.0), (&d, alpha_t * phi_2)]);
                }
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, 1.0), (&z, s[i + 1] * (-(-2.0 * hv).exp_m1()).sqrt())]);
            }
            d2 = d1.take();
            d1 = Some(den);
            h2 = h1;
            h1 = h;
            self.done(i)?;
        }
        Ok(x)
    }

    fn ddpm(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let mut noise = self.noise();
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            let (sg, prev) = (s[i], s[i + 1]);
            let eps = to_d(&x, &den, sg);
            let scaled = lc(&[(&x, 1.0 / (1.0 + sg * sg).sqrt())]);
            let (ac, ac_prev) = (1.0 / (sg * sg + 1.0), 1.0 / (prev * prev + 1.0));
            let alpha = ac / ac_prev;
            let root = (1.0 / alpha).sqrt();
            let mut mu = lc(&[(&scaled, root), (&eps, -(root * ((1.0 - alpha) / (1.0 - ac).sqrt())))]);
            if prev > 0.0 {
                let z = noise.next(sg, prev);
                mu = lc(&[(&mu, 1.0), (&z, ((1.0 - alpha) * (1.0 - ac_prev) / (1.0 - ac)).sqrt())]);
            }
            x = if prev != 0.0 { lc(&[(&mu, (1.0 + prev * prev).sqrt())]) } else { mu };
            self.done(i)?;
        }
        Ok(x)
    }

    fn lcm(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        let mut noise = self.noise();
        for i in 0..s.len() - 1 {
            x = self.den(&x, s[i])?;
            if s[i + 1] > 0.0 {
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&z, s[i + 1]), (&x, 1.0 - s[i + 1])]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn ipndm(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        const MAX: usize = 4;
        let mut buf: Vec<V> = Vec::new();
        for i in 0..s.len() - 1 {
            let (t, next) = (s[i], s[i + 1]);
            let den = self.den(&x, t)?;
            let d = to_d(&x, &den, t);
            let order = (i + 1).min(MAX);
            let b = buf.len();
            let dt = next - t;
            x = if next == 0.0 {
                den
            } else {
                match order {
                    1 => lc(&[(&x, 1.0), (&d, dt)]),
                    2 => lc(&[(&x, 1.0), (&d, dt * 3.0 / 2.0), (&buf[b - 1], -dt / 2.0)]),
                    3 => lc(&[(&x, 1.0), (&d, dt * 23.0 / 12.0), (&buf[b - 1], -dt * 16.0 / 12.0), (&buf[b - 2], dt * 5.0 / 12.0)]),
                    _ => lc(&[(&x, 1.0), (&d, dt * 55.0 / 24.0), (&buf[b - 1], -dt * 59.0 / 24.0), (&buf[b - 2], dt * 37.0 / 24.0), (&buf[b - 3], -dt * 9.0 / 24.0)]),
                }
            };
            push_history(&mut buf, d, MAX - 1);
            self.done(i)?;
        }
        Ok(x)
    }

    fn ipndm_v(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        const MAX: usize = 4;
        let mut buf: Vec<V> = Vec::new();
        for i in 0..s.len() - 1 {
            let (t, next) = (s[i], s[i + 1]);
            let den = self.den(&x, t)?;
            let d = to_d(&x, &den, t);
            let order = (i + 1).min(MAX);
            let b = buf.len();
            let dt = next - t;
            x = if next == 0.0 {
                den
            } else if order == 1 {
                lc(&[(&x, 1.0), (&d, dt)])
            } else {
                let (hn, hn1) = (dt, t - s[i - 1]);
                let (c1, c2) = ((2.0 + hn / hn1) / 2.0, -(hn / hn1) / 2.0);
                if order == 2 {
                    lc(&[(&x, 1.0), (&d, dt * c1), (&buf[b - 1], dt * c2)])
                } else {
                    let hn2 = s[i - 1] - s[i - 2];
                    let temp1 = (1.0 - hn / (3.0 * (hn + hn1)) * (hn * (hn + hn1)) / (hn1 * (hn1 + hn2))) / 2.0;
                    if order == 3 {
                        let (c1, c2, c3) = (c1 + temp1, c2 - (1.0 + hn1 / hn2) * temp1, temp1 * hn1 / hn2);
                        lc(&[(&x, 1.0), (&d, dt * c1), (&buf[b - 1], dt * c2), (&buf[b - 2], dt * c3)])
                    } else {
                        let hn3 = s[i - 2] - s[i - 3];
                        let temp2 = ((1.0 - hn / (3.0 * (hn + hn1))) / 2.0 + (1.0 - hn / (2.0 * (hn + hn1))) * hn / (6.0 * (hn + hn1 + hn2)))
                            * (hn * (hn + hn1) * (hn + hn1 + hn2)) / (hn1 * (hn1 + hn2) * (hn1 + hn2 + hn3));
                        let k = hn1 * (hn1 + hn2) / (hn2 * (hn2 + hn3));
                        let c1 = c1 + temp1 + temp2;
                        let c2 = c2 - (1.0 + hn1 / hn2) * temp1 - (1.0 + (hn1 / hn2) + k) * temp2;
                        let c3 = temp1 * hn1 / hn2 + ((hn1 / hn2) + k * (1.0 + hn2 / hn3)) * temp2;
                        let c4 = -temp2 * k * hn1 / hn2;
                        lc(&[(&x, 1.0), (&d, dt * c1), (&buf[b - 1], dt * c2), (&buf[b - 2], dt * c3), (&buf[b - 3], dt * c4)])
                    }
                }
            };
            push_history(&mut buf, d, MAX - 1);
            self.done(i)?;
        }
        Ok(x)
    }

    fn deis(&mut self, mut x: V, s: &[f32]) -> Result<V> {
        const MAX: usize = 3;
        let coeffs = deis_coeffs(s, MAX);
        let mut buf: Vec<V> = Vec::new();
        for i in 0..s.len() - 1 {
            let (t, next) = (s[i], s[i + 1]);
            let den = self.den(&x, t)?;
            let d = to_d(&x, &den, t);
            let b = buf.len();
            let c = &coeffs[i];
            x = if next <= 0.0 || c.is_empty() {
                lc(&[(&x, 1.0), (&d, next - t)])
            } else if c.len() == 2 {
                lc(&[(&x, 1.0), (&d, c[0]), (&buf[b - 1], c[1])])
            } else {
                lc(&[(&x, 1.0), (&d, c[0]), (&buf[b - 1], c[1]), (&buf[b - 2], c[2])])
            };
            push_history(&mut buf, d, MAX - 1);
            self.done(i)?;
        }
        Ok(x)
    }

    fn res_multistep(&mut self, mut x: V, s: &[f32], eta: f32, cfg_pp: bool) -> Result<V> {
        let mut noise = self.noise();
        let t_fn = |s: f32| -s.ln();
        let phi1 = |t: f32| t.exp_m1() / t;
        let phi2 = |t: f32| (phi1(t) - 1.0) / t;
        let mode = if cfg_pp { Uncond::Force } else { Uncond::Skip };
        let mut old: Option<V> = None;
        let mut old_down = 0f32;
        for i in 0..s.len() - 1 {
            let e = (self.eval)(&x, s[i], mode)?;
            let (den, und) = (e.denoised, e.uncond);
            let (sd, su) = ancestral_step(s[i], s[i + 1], eta);
            match &old {
                Some(o) if sd != 0.0 => {
                    let (t, t_old, next, prev) = (t_fn(s[i]), t_fn(old_down), t_fn(sd), t_fn(s[i - 1]));
                    let h = next - t;
                    let c2 = (prev - t_old) / h;
                    let (p1, p2) = (phi1(-h), phi2(-h));
                    let (b1, b2) = (nan_to_num(p1 - p2 / c2), nan_to_num(p2 / c2));
                    let exp = (-h).exp();
                    if let Some(u) = &und {
                        let y = lc(&[(&x, 1.0), (&den, 1.0), (u, -1.0)]);
                        x = lc(&[(&y, exp), (u, h * b1), (o, h * b2)]);
                    } else {
                        x = lc(&[(&x, exp), (&den, h * b1), (o, h * b2)]);
                    }
                }
                _ => {
                    if let Some(u) = &und {
                        let d = to_d(&x, u, s[i]);
                        x = lc(&[(&den, 1.0), (&d, sd)]);
                    } else {
                        let d = to_d(&x, &den, s[i]);
                        x = lc(&[(&x, 1.0), (&d, sd - s[i])]);
                    }
                }
            }
            if su > 0.0 {
                let z = noise.next(s[i], s[i + 1]);
                x = lc(&[(&x, 1.0), (&z, su)]);
            }
            old = Some(und.unwrap_or(den));
            old_down = sd;
            self.done(i)?;
        }
        Ok(x)
    }

    fn gradient_estimation(&mut self, mut x: V, s: &[f32], cfg_pp: bool) -> Result<V> {
        const GAMMA: f32 = 2.0;
        let mode = if cfg_pp { Uncond::Force } else { Uncond::Skip };
        let mut old_d = V::new();
        for i in 0..s.len() - 1 {
            let e = (self.eval)(&x, s[i], mode)?;
            let den = e.denoised;
            let d = to_d(&x, e.uncond.as_ref().unwrap_or(&den), s[i]);
            let dt = s[i + 1] - s[i];
            if s[i + 1] == 0.0 {
                x = den;
            } else {
                x = if cfg_pp { lc(&[(&den, 1.0), (&d, s[i + 1])]) } else { lc(&[(&x, 1.0), (&d, dt)]) };
                if i >= 1 {
                    let c = (GAMMA - 1.0) * dt;
                    x = lc(&[(&x, 1.0), (&d, c), (&old_d, -c)]);
                }
            }
            old_d = d;
            self.done(i)?;
        }
        Ok(x)
    }

    fn er_sde(&mut self, mut x: V, sigmas: &[f32]) -> Result<V> {
        const POINTS: usize = 200;
        let scaler = |x: f32| x * (x.powf(0.3).exp() + 10.0);
        let mut noise = self.noise();
        let s = offset_first(sigmas, self.ms);
        let lam: V = s.iter().map(|&v| (-lambda(v)).exp()).collect();
        let (mut old, mut old_d) = (V::new(), V::new());
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den.clone();
            } else {
                let (ls, lt) = (lam[i], lam[i + 1]);
                let (alpha_s, alpha_t) = (s[i] / ls, s[i + 1] / lt);
                let r = scaler(lt) / scaler(ls);
                x = lc(&[(&x, alpha_t / alpha_s * r), (&den, alpha_t * (1.0 - r))]);
                if i >= 1 {
                    let dt = lt - ls;
                    let step = -dt / POINTS as f32;
                    let (mut sum, mut sum_u) = (0f64, 0f64);
                    for p in 0..POINTS {
                        let pos = lt + p as f32 * step;
                        let sp = scaler(pos);
                        sum += (1.0 / sp) as f64;
                        sum_u += ((pos - ls) / sp) as f64;
                    }
                    let c = 1.0 / (ls - lam[i - 1]);
                    let d = lc(&[(&den, c), (&old, -c)]);
                    x = lc(&[(&x, 1.0), (&d, alpha_t * (dt + sum as f32 * step * scaler(lt)))]);
                    if i >= 2 {
                        let c = 1.0 / ((ls - lam[i - 2]) / 2.0);
                        let u = lc(&[(&d, c), (&old_d, -c)]);
                        x = lc(&[(&x, 1.0), (&u, alpha_t * (dt * dt / 2.0 + sum_u as f32 * step * scaler(lt)))]);
                    }
                    old_d = d;
                }
                let z = noise.next(s[i], s[i + 1]);
                let c = alpha_t * nan_to_num((lt * lt - ls * ls * r * r).sqrt());
                x = lc(&[(&x, 1.0), (&z, c)]);
            }
            old = den;
            self.done(i)?;
        }
        Ok(x)
    }

    fn seeds_2(&mut self, mut x: V, sigmas: &[f32], eta: f32, r: f32, phi_2: bool) -> Result<V> {
        let mut noise = self.noise();
        let inject = eta > 0.0;
        let s = offset_first(sigmas, self.ms);
        let fac = 1.0 / (2.0 * r);
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den;
                self.done(i)?;
                continue;
            }
            let (ls, lt) = (lambda(s[i]), lambda(s[i + 1]));
            let h = lt - ls;
            let h_eta = h * (eta + 1.0);
            let ls1 = lerp(ls, lt, r);
            let s1 = sigma_of(ls1);
            let (alpha_s1, alpha_t) = (s1 * ls1.exp(), s[i + 1] * lt.exp());
            let mut x2 = lc(&[(&x, s1 / s[i] * (-r * h * eta).exp()), (&den, -(alpha_s1 * phi1(-r * h_eta)))]);
            let mut sde = V::new();
            if inject {
                let c = (-(-2.0 * r * h * eta).exp_m1()).sqrt();
                let z = noise.next(s[i], s1);
                sde = lc(&[(&z, c)]);
                x2 = lc(&[(&x2, 1.0), (&sde, s1)]);
            }
            let den2 = self.den(&x2, s1)?;
            let keep = s[i + 1] / s[i] * (-h * eta).exp();
            x = if phi_2 {
                let b2 = self::phi2(-h_eta) / r;
                let b1 = phi1(-h_eta) - b2;
                lc(&[(&x, keep), (&den, -(alpha_t * b1)), (&den2, -(alpha_t * b2))])
            } else {
                let dd = lerp_v(&den, &den2, fac);
                lc(&[(&x, keep), (&dd, -(alpha_t * phi1(-h_eta)))])
            };
            if inject {
                let seg = (r - 1.0) * h * eta;
                let z = noise.next(s1, s[i + 1]);
                sde = lc(&[(&sde, seg.exp()), (&z, (-(seg * 2.0).exp_m1()).sqrt())]);
                x = lc(&[(&x, 1.0), (&sde, s[i + 1])]);
            }
            self.done(i)?;
        }
        Ok(x)
    }

    fn seeds_3(&mut self, mut x: V, sigmas: &[f32]) -> Result<V> {
        let (r1, r2, eta) = (1.0f32 / 3.0, 2.0f32 / 3.0, 1.0f32);
        let mut noise = self.noise();
        let s = offset_first(sigmas, self.ms);
        for i in 0..s.len() - 1 {
            let den = self.den(&x, s[i])?;
            if s[i + 1] == 0.0 {
                x = den;
                self.done(i)?;
                continue;
            }
            let (ls, lt) = (lambda(s[i]), lambda(s[i + 1]));
            let h = lt - ls;
            let h_eta = h * (eta + 1.0);
            let (l1, l2) = (lerp(ls, lt, r1), lerp(ls, lt, r2));
            let (s1, s2) = (sigma_of(l1), sigma_of(l2));
            let (alpha_1, alpha_2, alpha_t) = (s1 * l1.exp(), s2 * l2.exp(), s[i + 1] * lt.exp());

            let x2 = lc(&[(&x, s1 / s[i] * (-r1 * h * eta).exp()), (&den, -(alpha_1 * phi1(-r1 * h_eta)))]);
            let c = (-(-2.0 * r1 * h * eta).exp_m1()).sqrt();
            let z = noise.next(s[i], s1);
            let mut sde = lc(&[(&z, c)]);
            let x2 = lc(&[(&x2, 1.0), (&sde, s1)]);
            let den2 = self.den(&x2, s1)?;

            let a2 = r2 / r1 * phi2(-r2 * h_eta);
            let a1 = phi1(-r2 * h_eta) - a2;
            let x3 = lc(&[(&x, s2 / s[i] * (-r2 * h * eta).exp()), (&den, -(alpha_2 * a1)), (&den2, -(alpha_2 * a2))]);
            let seg = (r1 - r2) * h * eta;
            let z = noise.next(s1, s2);
            sde = lc(&[(&sde, seg.exp()), (&z, (-(seg * 2.0).exp_m1()).sqrt())]);
            let x3 = lc(&[(&x3, 1.0), (&sde, s2)]);
            let den3 = self.den(&x3, s2)?;

            let b3 = phi2(-h_eta) / r2;
            let b1 = phi1(-h_eta) - b3;
            x = lc(&[(&x, s[i + 1] / s[i] * (-h * eta).exp()), (&den, -(alpha_t * b1)), (&den3, -(alpha_t * b3))]);
            let seg = (r2 - 1.0) * h * eta;
            let z = noise.next(s2, s[i + 1]);
            sde = lc(&[(&sde, seg.exp()), (&z, (-(seg * 2.0).exp_m1()).sqrt())]);
            x = lc(&[(&x, 1.0), (&sde, s[i + 1])]);
            self.done(i)?;
        }
        Ok(x)
    }

    fn sa_solver(&mut self, x0: V, sigmas: &[f32], pece: bool) -> Result<V> {
        const PRED: usize = 3;
        const CORR: usize = 4;
        let mut gen = self.noise();
        let s = offset_first(sigmas, self.ms);
        let lam: V = s.iter().map(|&v| lambda(v)).collect();
        let (start, end) = (self.ms.percent_to_sigma(0.2), self.ms.percent_to_sigma(0.8));
        let tau_fn = |sigma: f32| if start >= sigma && sigma >= end { 1.0f32 } else { 0.0 };
        let last = s.len() - 1;
        let lower = s[last] == 0.0;
        let mut x = x0.clone();
        let mut x_pred = x0;
        let (mut h, mut tau) = (0f32, 0f32);
        let mut noise = V::new();
        let mut preds: Vec<V> = Vec::new();
        for i in 0..last {
            let mut den = self.den(&x_pred, s[i])?;
            preds.push(den.clone());
            if preds.len() > CORR {
                preds.remove(0);
            }
            let mut pred_order = PRED.min(preds.len());
            let mut corr_order = if i == 0 || (s[i + 1] == 0.0 && !pece) { 0 } else { CORR.min(preds.len()) };
            if lower {
                pred_order = pred_order.min(s.len() - 2 - i);
                corr_order = corr_order.min(s.len() - 1 - i);
            }
            if corr_order == 0 {
                x = x_pred.clone();
            } else {
                let b = sa_b_coeffs(s[i], &lam[i + 1 - corr_order..=i], lam[i - 1], lam[i], tau);
                let tail = &preds[preds.len() - corr_order..];
                let mut y = lc(&[(&x, s[i] / s[i - 1] * (-(tau * tau) * h).exp())]);
                for (p, c) in tail.iter().zip(&b) {
                    y = lc(&[(&y, 1.0), (p, *c)]);
                }
                if tau > 0.0 {
                    y = lc(&[(&y, 1.0), (&noise, 1.0)]);
                }
                x = y;
                if pece {
                    den = self.den(&x, s[i])?;
                    *preds.last_mut().unwrap() = den.clone();
                }
            }
            if s[i + 1] == 0.0 {
                x_pred = den;
            } else {
                tau = tau_fn(s[i + 1]);
                let b = sa_b_coeffs(s[i + 1], &lam[i + 1 - pred_order..=i], lam[i], lam[i + 1], tau);
                let tail = &preds[preds.len() - pred_order..];
                h = lam[i + 1] - lam[i];
                x_pred = lc(&[(&x, s[i + 1] / s[i] * (-(tau * tau) * h).exp())]);
                for (p, c) in tail.iter().zip(&b) {
                    x_pred = lc(&[(&x_pred, 1.0), (p, *c)]);
                }
                if tau > 0.0 {
                    let z = gen.next(s[i], s[i + 1]);
                    noise = lc(&[(&z, s[i + 1] * (-(-2.0 * tau * tau * h).exp_m1()).sqrt())]);
                    x_pred = lc(&[(&x_pred, 1.0), (&noise, 1.0)]);
                }
            }
            self.done(i)?;
        }
        Ok(x_pred)
    }

    /// `data_prediction_fn` through `predict_eps_sigma`.
    fn unipc_model(&mut self, x: &[f32], t: f32) -> Result<V> {
        let input = lc(&[(x, (t * t + 1.0).powf(0.5))]);
        let den = self.den(&input, t)?;
        let eps = to_d(&input, &den, t);
        let (alpha, std) = (vp_alpha(t), vp_std(t));
        Ok(lc(&[(x, 1.0), (&eps, -std)]).iter().map(|v| v / alpha).collect())
    }

    /// `multistep_uni_pc_bh_update`.
    #[allow(clippy::too_many_arguments)]
    fn unipc_update(&mut self, x: &[f32], models: &[V], ts: &[f32], t: f32, order: usize, corrector: bool, bh2: bool) -> Result<(V, Option<V>)> {
        let n = ts.len();
        let (t0, m0) = (ts[n - 1], &models[n - 1]);
        let (l0, lt) = (vp_lambda(t0), vp_lambda(t));
        let (s0, st, alpha_t) = (vp_std(t0), vp_std(t), vp_alpha(t));
        let h = lt - l0;
        let mut rks = V::new();
        let mut d1s: Vec<V> = Vec::new();
        for i in 1..order {
            let rk = (vp_lambda(ts[n - 1 - i]) - l0) / h;
            rks.push(rk);
            d1s.push(lc(&[(&models[n - 1 - i], 1.0 / rk), (m0, -1.0 / rk)]));
        }
        rks.push(1.0);
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let mut h_phi_k = h_phi_1 / hh - 1.0;
        let b_h = if bh2 { hh.exp_m1() } else { hh };
        let mut factorial = 1i64;
        let mut r: Vec<V> = Vec::new();
        let mut b = V::new();
        for i in 1..=order {
            r.push(rks.iter().map(|v| v.powi(i as i32 - 1)).collect());
            b.push(h_phi_k * factorial as f32 / b_h);
            factorial *= i as i64 + 1;
            h_phi_k = h_phi_k / hh - 1.0 / factorial as f32;
        }
        let base = lc(&[(x, st / s0), (m0, -(alpha_t * h_phi_1))]);
        let sum = |ds: &[V], rho: &[f32]| {
            let mut out = vec![0f32; x.len()];
            for (d, c) in ds.iter().zip(rho) {
                out = lc(&[(&out, 1.0), (d, *c)]);
            }
            out
        };
        let pred = if d1s.is_empty() {
            vec![0f32; x.len()]
        } else {
            let rhos = if order == 2 {
                vec![0.5]
            } else {
                solve(r[..order - 1].iter().map(|row| row[..order - 1].to_vec()).collect(), b[..order - 1].to_vec())
            };
            sum(&d1s, &rhos)
        };
        let x_t = lc(&[(&base, 1.0), (&pred, -(alpha_t * b_h))]);
        if !corrector {
            return Ok((x_t, None));
        }
        let rhos = if order == 1 { vec![0.5] } else { solve(r, b) };
        let model_t = self.unipc_model(&x_t, t)?;
        let corr = sum(&d1s, &rhos[..order - 1]);
        let last = rhos[order - 1];
        let total = lc(&[(&corr, 1.0), (&model_t, last), (m0, -last)]);
        Ok((lc(&[(&base, 1.0), (&total, -(alpha_t * b_h))]), Some(model_t)))
    }

    fn unipc(&mut self, noise: V, sigmas: &[f32], bh2: bool) -> Result<V> {
        let mut ts = sigmas.to_vec();
        let last = ts.len() - 1;
        if ts[last] == 0.0 {
            ts[last] = 0.001;
        }
        let mut x = lc(&[(&noise, 1.0 / (1.0 + ts[0] * ts[0]).sqrt())]);
        let order = 3.min(ts.len().saturating_sub(2));
        let steps = last;
        let mut models: Vec<V> = Vec::new();
        let mut t_prev: V = Vec::new();
        for index in 0..steps {
            if index == 0 {
                models.push(self.unipc_model(&x, ts[0])?);
                t_prev.push(ts[0]);
            } else if index < order {
                let (next, m) = self.unipc_update(&x, &models, &t_prev, ts[index], index, true, bh2)?;
                x = next;
                models.push(m.unwrap());
                t_prev.push(ts[index]);
            } else {
                let extra = (index == steps - 1) as usize;
                for step in index..=index + extra {
                    let step_order = order.min(steps + 1 - step);
                    let (next, m) = self.unipc_update(&x, &models, &t_prev, ts[step], step_order, step != steps, bh2)?;
                    x = next;
                    models.remove(0);
                    t_prev.remove(0);
                    t_prev.push(ts[step]);
                    match m {
                        Some(m) if step < steps => models.push(m),
                        _ => models.push(Vec::new()),
                    }
                }
            }
            self.done(index)?;
        }
        let alpha = vp_alpha(ts[last]);
        // ComfyUI sets a last sigma of 0 to 0.001 in place, so `inverse_noise_scaling` divides by 1 - 0.001 after
        // this; any other last sigma is divided out by `sample`
        let quirk = if sigmas[last] == 0.0 { 1.0 - ts[last] } else { 1.0 };
        Ok(x.iter().map(|v| v / alpha / quirk).collect())
    }
}

fn push_history(buf: &mut Vec<V>, d: V, max: usize) {
    buf.push(d);
    if buf.len() > max {
        buf.remove(0);
    }
}

/// `sigma_down`, `alpha_ip1`, `alpha_down` and `renoise_coeff` of the `_RF` samplers at eta 1.
fn rf_step(sigma: f32, next: f32) -> (f32, f32, f32, f32) {
    let sd = next * (1.0 + (next / sigma - 1.0));
    let (a1, ad) = (1.0 - next, 1.0 - sd);
    (sd, a1, ad, (next * next - sd * sd * a1 * a1 / (ad * ad)).powf(0.5))
}

fn vp_log_alpha(s: f32) -> f32 {
    0.5 * (1.0 / (s * s + 1.0)).ln()
}

fn vp_alpha(s: f32) -> f32 {
    vp_log_alpha(s).exp()
}

fn vp_std(s: f32) -> f32 {
    (1.0 - (2.0 * vp_log_alpha(s)).exp()).sqrt()
}

fn vp_lambda(s: f32) -> f32 {
    let la = vp_log_alpha(s);
    la - 0.5 * (1.0 - (2.0 * la).exp()).ln()
}

/// One sampling run: `KSAMPLER.sample`. `latent` is where it starts, at the sampler's scale (`None` for an empty
/// latent), and `sigmas` may start below the model's highest (img2img, refiners). `eval` is the guided model and `step`
/// reports progress; an error from either ends the run.
#[allow(clippy::too_many_arguments)]
pub fn sample(sampler: &str, ms: ModelSampling, sigmas: &[f32], noise: &[f32], latent: Option<&[f32]>, seed: u64, batch: Batch,
    eval: &mut dyn FnMut(&[f32], f32, Uncond) -> Result<Eval>, step: &mut dyn FnMut(usize, usize) -> Result<()>) -> Result<V> {
    if latent.is_some_and(|l| l.len() != noise.len()) {
        bail!("latent of {} values for noise of {}", latent.map_or(0, <[f32]>::len), noise.len());
    }
    // `CFGGuider.sample` returns the latent untouched without sigmas
    let Some(&s0) = sigmas.first() else {
        return Ok(latent.map_or_else(|| vec![0.0; noise.len()], <[f32]>::to_vec));
    };
    // CONST.noise_scaling
    let x = match latent {
        Some(l) => lc(&[(noise, s0), (l, 1.0 - s0)]),
        None => lc(&[(noise, s0)]),
    };
    if sigmas.len() == 1 {
        return Ok(if s0 != 0.0 { x.iter().map(|v| v / (1.0 - s0)).collect() } else { x });
    }
    let s = sigmas;
    let mut r = Run { ms, eval, step, total: s.len() - 1, seed, batch, n: noise.len() };
    let out = match sampler {
        "euler" | "ddim" => r.euler(x, s),
        "euler_cfg_pp" => r.euler_ancestral_cfg_pp(x, s, 0.0),
        "euler_ancestral" => r.euler_ancestral(x, s),
        "euler_ancestral_cfg_pp" => r.euler_ancestral_cfg_pp(x, s, 1.0),
        "heun" => r.heun(x, s),
        "heunpp2" => r.heunpp2(x, s),
        "exp_heun_2_x0" => r.seeds_2(x, s, 0.0, 1.0, true),
        "exp_heun_2_x0_sde" => r.seeds_2(x, s, 1.0, 1.0, true),
        "dpm_2" => r.dpm_2(x, s),
        "dpm_2_ancestral" => r.dpm_2_ancestral(x, s),
        "lms" => r.lms(x, s),
        "dpm_fast" => r.dpm_fast(x, s),
        "dpm_adaptive" => r.dpm_adaptive(x, s),
        "dpmpp_2s_ancestral" => r.dpmpp_2s_ancestral(x, s),
        "dpmpp_2s_ancestral_cfg_pp" => r.dpmpp_2s_ancestral_cfg_pp(x, s),
        "dpmpp_sde" => r.dpmpp_sde(x, s, true),
        "dpmpp_sde_gpu" => r.dpmpp_sde(x, s, false),
        "dpmpp_2m" => r.dpmpp_2m(x, s),
        "dpmpp_2m_cfg_pp" => r.dpmpp_2m_cfg_pp(x, s),
        "dpmpp_2m_sde" => r.dpmpp_2m_sde(x, s, true, false),
        "dpmpp_2m_sde_gpu" => r.dpmpp_2m_sde(x, s, false, false),
        "dpmpp_2m_sde_heun" => r.dpmpp_2m_sde(x, s, true, true),
        "dpmpp_2m_sde_heun_gpu" => r.dpmpp_2m_sde(x, s, false, true),
        "dpmpp_3m_sde" => r.dpmpp_3m_sde(x, s, true),
        "dpmpp_3m_sde_gpu" => r.dpmpp_3m_sde(x, s, false),
        "ddpm" => r.ddpm(x, s),
        "lcm" => r.lcm(x, s),
        "ipndm" => r.ipndm(x, s),
        "ipndm_v" => r.ipndm_v(x, s),
        "deis" => r.deis(x, s),
        "cfgpp_ud10_ab" => r.cfgpp_ud10_ab(x, s),
        "res_multistep" => r.res_multistep(x, s, 0.0, false),
        "res_multistep_cfg_pp" => r.res_multistep(x, s, 0.0, true),
        "res_multistep_ancestral" => r.res_multistep(x, s, 1.0, false),
        "res_multistep_ancestral_cfg_pp" => r.res_multistep(x, s, 1.0, true),
        "gradient_estimation" => r.gradient_estimation(x, s, false),
        "gradient_estimation_cfg_pp" => r.gradient_estimation(x, s, true),
        "er_sde" => r.er_sde(x, s),
        "seeds_2" => r.seeds_2(x, s, 1.0, 0.5, false),
        "seeds_3" => r.seeds_3(x, s),
        "sa_solver" => r.sa_solver(x, s, false),
        "sa_solver_pece" => r.sa_solver(x, s, true),
        "uni_pc" => r.unipc(x, s, false),
        "uni_pc_bh2" => r.unipc(x, s, true),
        other => bail!("unknown sampler {other}"),
    }?;
    let last = s[s.len() - 1];
    Ok(if last != 0.0 { out.iter().map(|v| v / (1.0 - last)).collect() } else { out })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence_matches_numpy() {
        // numpy.random.SeedSequence(entropy=42, spawn_key=(1, 2), pool_size=24).generate_state(4)
        assert_eq!(seed_state(42, Some((1, 2)), 4), [2101547719, 3776121365, 348876581, 899550832]);
        assert_eq!(seed_state(42, None, 3), [387750627, 2536650517, 4078839648]);
    }

    #[test]
    fn brownian_tree_matches_torchsde() {
        // torchsde.BrownianTree(0.05, zeros(1, 4, 8, 8), 1.0, entropy=42)(0.5, 0.75) then (0.3, 0.4)
        let batch = Batch::single(73728);
        let mut t = BrownianTree::new(0.05f32 as f64, 1.0, 42, true, 256, batch);
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-4);
        let w = t.query(0.5f32 as f64, 0.75f32 as f64);
        assert!(close(&w, &[-0.4269, 0.1212, 0.0346]), "{:?}", &w[..3]);
        let w = t.query(0.3f32 as f64, 0.4f32 as f64);
        assert!(close(&w, &[0.1700, 0.2437, 0.0439]), "{:?}", &w[..3]);
    }

    #[test]
    fn brownian_tree_matches_torchsde_at_full_size() {
        // BrownianTree(0.0291, zeros(1, 16, 128, 128), 1.0, entropy=42)(0.7, 0.95), on the CPU and on this GPU
        let batch = Batch::single(73728);
        for (cpu, want) in [(true, [0.779_023_8, -0.386_063_5, -0.527_965_8]), (false, [0.170_391_2, -0.264_144_2, 0.081_959_6])] {
            let mut t = BrownianTree::new(0.0291f32 as f64, 1.0, 42, cpu, 16 * 128 * 128, batch);
            let w = t.query(0.7f32 as f64, 0.95f32 as f64);
            assert!(w.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5), "{:?}", &w[..3]);
        }
    }

    #[test]
    fn a_batch_samples_like_its_images_apart() {
        // the model is per-value, so only the noise ties the images together
        let ms = ModelSampling::Flow { shift: 1.0 };
        let sigmas = [1.0, 0.7, 0.4, 0.1, 0.0];
        let n = 300;
        let noise = torch_randn(5, 2 * n);
        let run = |sampler: &str, noise: &[f32], batch: Batch| {
            let mut eval = |x: &[f32], s: f32, _: Uncond| Ok(Eval { denoised: x.iter().map(|v| v * 0.5 * (1.0 - s)).collect(), uncond: None });
            sample(sampler, ms, &sigmas, noise, None, 9, batch, &mut eval, &mut |_, _| Ok(())).unwrap()
        };
        for sampler in ["euler_ancestral", "dpmpp_sde", "dpmpp_2m_sde_gpu"] {
            let together = run(sampler, &noise, Batch { index: 0, size: 2, images: 2, threads: 73728 });
            for i in 0..2 {
                let alone = run(sampler, &noise[i * n..(i + 1) * n], Batch { index: i, size: 2, images: 1, threads: 73728 });
                assert_eq!(&together[i * n..(i + 1) * n], &alone[..], "{sampler}, image {i}");
            }
        }
    }

    #[test]
    fn beta_ppf_is_symmetric() {
        assert!((beta_ppf(0.5, 0.6, 0.6) - 0.5).abs() < 1e-12);
        assert!((beta_ppf(0.25, 0.6, 0.6) + beta_ppf(0.75, 0.6, 0.6) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ndtri_inverts_known_quantiles() {
        assert!((ndtri(0.975) - 1.959_963_984_540_054).abs() < 1e-9);
        assert_eq!(ndtri(0.5), 0.0);
    }
}
