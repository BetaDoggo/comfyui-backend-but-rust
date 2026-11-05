//! Partial sampling against ComfyUI's own code, recorded by `tools/partial_ref.py`: KSampler's step ranges, and runs
//! from a latent with SetLatentNoiseMask and DifferentialDiffusion on the mock model of `tools/sampler_ref.py`.

use diffusion::samplers::{self, Batch, Eval, Inpaint, ModelSampling, Uncond};
use diffusion::sampling::torch_randn;
use serde_json::Value;

/// Worst relative error allowed in a final latent: f32 rounding, not a different algorithm.
const LATENT_TOLERANCE: f64 = 2e-5;

fn golden() -> Value {
    serde_json::from_str(include_str!("partial_golden.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN) as f32).collect()
}

fn model_sampling(c: &Value) -> ModelSampling {
    let shift = c["shift"].as_f64().unwrap();
    if c["kind"] == "flow" { ModelSampling::Flow { shift } } else { ModelSampling::Flux { shift } }
}

fn uint(c: &Value, key: &str) -> usize {
    c[key].as_u64().unwrap() as usize
}

#[test]
fn step_ranges_match_comfy() {
    let g = golden();
    let cases = g["schedules"].as_array().unwrap();
    let mut returned = 0;
    for c in cases {
        let ms = model_sampling(c);
        let (sampler, scheduler) = (c["sampler"].as_str().unwrap(), c["scheduler"].as_str().unwrap());
        let schedule = |n| samplers::sigmas(ms, sampler, scheduler, n, (128, 128), None);
        let denoise = c["denoise"].as_f64().unwrap();
        let got = samplers::ksampler_sigmas(uint(c, "steps"), denoise, uint(c, "start"), uint(c, "end"), c["leftover"].as_bool().unwrap(), schedule).unwrap();
        let name = format!("{} {sampler} {scheduler} {} steps, denoise {denoise}, steps {}-{}", c["kind"], c["steps"], c["start"], c["end"]);
        match (got, c["sigmas"].is_null()) {
            (None, true) => returned += 1,
            (Some(got), false) => {
                let want = floats(&c["sigmas"]);
                assert_eq!(got.len(), want.len(), "{name}: {got:?} vs {want:?}");
                assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 3e-5 * b.abs().max(1e-3)), "{name}: {got:?} vs {want:?}");
            }
            (got, _) => panic!("{name}: {got:?} vs {}", c["sigmas"]),
        }
    }
    eprintln!("{} step ranges, {returned} returning the latent", cases.len());
    assert!(returned > 0 && returned < cases.len());
}

/// The mock model of `tools/sampler_ref.py`.
fn mock(x: &[f32], s: f32, mode: Uncond, target: &[f32], cfg: f32) -> Eval {
    let cond: Vec<f32> = x.iter().zip(target).map(|(x, t)| t * (0.8 + 0.2 * s) + 0.3 * x.tanh() * (1.0 - s)).collect();
    if (cfg - 1.0).abs() <= 1e-9 && mode != Uncond::Force {
        return Eval { denoised: cond, uncond: None };
    }
    let uncond: Vec<f32> = x.iter().zip(target).map(|(x, t)| 0.5 * t + 0.1 * x).collect();
    let denoised = cond.iter().zip(&uncond).map(|(c, u)| u + (c - u) * cfg).collect();
    Eval { denoised, uncond: (mode != Uncond::Skip).then_some(uncond) }
}

#[test]
fn partial_runs_match_comfy() {
    let g = golden();
    let seed = g["seed"].as_u64().unwrap();
    let latent = floats(&g["latent"]);
    let mask = floats(&g["mask"]);
    let n = latent.len();
    let target: Vec<f32> = (0..n).map(|i| (0.37 * i as f64 + 1.0).sin() as f32).collect();
    let batch = Batch::single(73728);
    let mut failed = Vec::new();
    let mut worst = 0f64;
    let runs = g["runs"].as_array().unwrap();
    for c in runs {
        let (sampler, ms, cfg) = (c["sampler"].as_str().unwrap(), model_sampling(c), c["cfg"].as_f64().unwrap() as f32);
        let sigmas = floats(&c["sigmas"]);
        let noise = if c["add_noise"].as_bool().unwrap() { torch_randn(seed, n) } else { vec![0.0; n] };
        let inpaint = c["inpaint"].as_str().map(|kind| Inpaint {
            mask: mask.clone(),
            differential: kind.strip_prefix("differential").map(|s| s.trim().parse().unwrap_or(1.0)),
        });
        let mut evals = Vec::new();
        // as pipeline::run_sampler wraps the model
        let mut eval = |x: &[f32], s: f32, mode: Uncond| {
            evals.push(s);
            let Some(i) = &inpaint else { return Ok(mock(x, s, mode, &target, cfg)) };
            let m = i.mask_at(ms, &sigmas, s);
            let e = mock(&Inpaint::input(&m, x, s, &noise, &latent), s, mode, &target, cfg);
            Ok(Eval { denoised: Inpaint::prediction(&m, &e.denoised, &latent), uncond: e.uncond })
        };
        let mut step = |_: usize, _: usize| Ok(());
        let x = samplers::sample(sampler, ms, &sigmas, &noise, Some(&latent), seed, batch, &mut eval, &mut step).unwrap();

        let name = format!("{sampler} ({} steps {}-{} {} noise {})", c["kind"], c["start"], c["end"], c["inpaint"], c["add_noise"]);
        let want_evals = floats(&c["evals"]);
        if evals.len() != want_evals.len() || evals.iter().zip(&want_evals).any(|(a, b)| (a - b).abs() > 2e-5 * b.abs().max(1e-3)) {
            failed.push(format!("{name}: evaluations\n  ours {evals:?}\n  ref  {want_evals:?}"));
            continue;
        }
        let want = floats(&c["x"]);
        let norm: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let err = x.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt() / norm;
        worst = worst.max(err);
        if err > LATENT_TOLERANCE || err.is_nan() {
            failed.push(format!("{name}: relative error {err:.2e}"));
        }
    }
    eprintln!("{} runs, worst relative error {worst:.2e}", runs.len());
    assert!(failed.is_empty(), "{} of {} failed:\n{}", failed.len(), runs.len(), failed.join("\n"));
}
