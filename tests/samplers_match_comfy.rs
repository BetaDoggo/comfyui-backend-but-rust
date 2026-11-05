//! Every sampler and scheduler against ComfyUI's own code, recorded by `tools/sampler_ref.py`.

use diffusion::samplers::{self, Batch, Eval, ModelSampling, Uncond};
use diffusion::sampling::torch_randn;
use serde_json::Value;

/// Worst relative error allowed in a final latent: f32 rounding, not a different algorithm.
const LATENT_TOLERANCE: f64 = 2e-5;

fn golden() -> Value {
    serde_json::from_str(include_str!("sampler_golden.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN) as f32).collect()
}

fn model_sampling(c: &Value) -> ModelSampling {
    let shift = c["shift"].as_f64().unwrap();
    if c["kind"] == "flow" { ModelSampling::Flow { shift } } else { ModelSampling::Flux { shift } }
}

#[test]
fn schedulers_match_comfy() {
    let g = golden();
    let mut worst = (0f64, String::new());
    let cases = g["schedules"].as_array().unwrap();
    for c in cases {
        let want = floats(&c["sigmas"]);
        let (sampler, scheduler, steps) = (c["sampler"].as_str().unwrap(), c["scheduler"].as_str().unwrap(), c["steps"].as_u64().unwrap() as usize);
        let got = samplers::sigmas(model_sampling(c), sampler, scheduler, steps, (128, 128), None).unwrap();
        let name = format!("{} {} {sampler} {scheduler} {steps}", c["kind"], c["shift"]);
        assert_eq!(got.len(), want.len(), "{name}: {got:?} vs {want:?}");
        for (a, b) in got.iter().zip(&want) {
            let e = ((a - b).abs() as f64) / (b.abs() as f64).max(1e-3);
            assert!(!a.is_nan(), "{name}: {got:?}");
            if e > worst.0 {
                worst = (e, name.clone());
            }
            assert!(e < 3e-5, "{name}: {got:?} vs {want:?}");
        }
    }
    eprintln!("{} schedules, worst relative error {:.2e} in {}", cases.len(), worst.0, worst.1);
}

/// The mock model of `tools/sampler_ref.py`.
struct Mock {
    target: Vec<f32>,
    cfg: f32,
    evals: Vec<f32>,
}

impl Mock {
    fn eval(&mut self, x: &[f32], s: f32, mode: Uncond) -> Eval {
        self.evals.push(s);
        let cond: Vec<f32> = x.iter().zip(&self.target).map(|(x, t)| t * (0.8 + 0.2 * s) + 0.3 * x.tanh() * (1.0 - s)).collect();
        let guided = (self.cfg - 1.0).abs() > 1e-9;
        if !guided && mode != Uncond::Force {
            return Eval { denoised: cond, uncond: None };
        }
        let uncond: Vec<f32> = x.iter().zip(&self.target).map(|(x, t)| 0.5 * t + 0.1 * x).collect();
        let denoised = cond.iter().zip(&uncond).map(|(c, u)| u + (c - u) * self.cfg).collect();
        Eval { denoised, uncond: (mode != Uncond::Skip).then_some(uncond) }
    }
}

#[test]
fn samplers_match_comfy() {
    let g = golden();
    let seed = g["seed"].as_u64().unwrap();
    let noise = torch_randn(seed, 256);
    let target: Vec<f32> = (0..256).map(|i| (0.37 * i as f64 + 1.0).sin() as f32).collect();
    let batch = Batch::single(73728);
    let mut failed = Vec::new();
    let mut worst: std::collections::BTreeMap<String, f64> = Default::default();
    for c in g["runs"].as_array().unwrap() {
        let sampler = c["sampler"].as_str().unwrap();
        let ms = model_sampling(c);
        let sigmas = samplers::sigmas(ms, sampler, c["scheduler"].as_str().unwrap(), c["steps"].as_u64().unwrap() as usize, (128, 128), None).unwrap();
        let mut mock = Mock { target: target.clone(), cfg: c["cfg"].as_f64().unwrap() as f32, evals: Vec::new() };
        let mut eval = |x: &[f32], s: f32, mode: Uncond| Ok(mock.eval(x, s, mode));
        let mut step = |_: usize, _: usize| Ok(());
        let x = samplers::sample(sampler, ms, &sigmas, &noise, None, seed, batch, &mut eval, &mut step).unwrap();

        let want = floats(&c["x"]);
        let name = format!("{sampler} ({} {} {} {} cfg {})", c["kind"], c["shift"], c["scheduler"], c["steps"], c["cfg"]);
        let want_evals = floats(&c["evals"]);
        let n = mock.evals.len();
        let close = mock.evals.iter().zip(&want_evals).all(|(a, b)| (a - b).abs() <= 2e-5 * b.abs().max(1e-3));
        if n != want_evals.len() || !close {
            failed.push(format!("{name}: {n} evals vs {}\n  ours {:?}\n  ref  {:?}", want_evals.len(), mock.evals, want_evals));
            continue;
        }
        let norm: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let err: f64 = x[..want.len()].iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt() / norm;
        let w = worst.entry(sampler.to_string()).or_default();
        *w = w.max(err);
        if err > LATENT_TOLERANCE || err.is_nan() {
            failed.push(format!("{name}: relative error {err:.2e}"));
        }
    }
    for (name, e) in &worst {
        eprintln!("{name:32} worst relative error {e:.2e}");
    }
    assert!(failed.is_empty(), "{} failed:\n{}", failed.len(), failed.join("\n"));
}
