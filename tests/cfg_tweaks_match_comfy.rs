//! ComfyUI's guidance patches, recorded by `tools/cfg_ref.py`.

use diffusion::samplers::{CfgFunction, CfgTweaks};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn guidance_patches_match_comfy() {
    let cases: Value = serde_json::from_str(include_str!("cfg_golden.json")).unwrap();
    let mut worst = 0f32;
    for c in cases.as_array().unwrap() {
        let t = &c["tweak"];
        let mut tweaks = CfgTweaks { tangential: t["tangential"] == true, zero_star: t["zero_star"] == true, function: None };
        if let Some(m) = t["rescale"].as_f64() {
            tweaks.function = Some(CfgFunction::Rescale { multiplier: m as f32 });
        }
        if let Some(r) = t["renorm"].as_array() {
            tweaks.function = Some(CfgFunction::Renorm { cfg_trunc: r[0].as_f64().unwrap() as f32, renorm_cfg: r[1].as_f64().unwrap() as f32 });
        }
        let (scale, sigma) = (c["scale"].as_f64().unwrap() as f32, c["sigma"].as_f64().unwrap() as f32);
        let (got, _) = tweaks.guide(&floats(&c["x"]), &floats(&c["cond"]), Some(floats(&c["uncond"])), scale, sigma);
        for (a, b) in got.iter().zip(floats(&c["out"])) {
            worst = worst.max((a - b).abs());
            assert!((a - b).abs() < 2e-5, "{t}: {a} vs {b}");
        }
    }
    eprintln!("worst absolute error {worst:.2e}");
}

#[test]
fn only_renorm_acts_without_an_uncond() {
    let (x, cond) = (vec![0.5, -0.25, 1.0, 2.0], vec![1.0, 2.0, -3.0, 0.5]);
    let tweaks = CfgTweaks { tangential: true, zero_star: true, function: Some(CfgFunction::Rescale { multiplier: 0.7 }) };
    assert_eq!(tweaks.guide(&x, &cond, None, 1.0, 0.5).0, cond);
    let tweaks = CfgTweaks { function: Some(CfgFunction::Renorm { cfg_trunc: 100.0, renorm_cfg: 0.5 }), ..Default::default() };
    let halved: Vec<f32> = cond.iter().map(|v| v / 2.0).collect();
    for (a, b) in tweaks.guide(&x, &cond, None, 1.0, 0.5).0.iter().zip(&halved) {
        assert!((a - b).abs() < 1e-6);
    }
}
