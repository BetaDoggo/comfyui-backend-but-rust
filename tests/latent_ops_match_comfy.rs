//! ComfyUI's latent nodes, recorded by `tools/latent_ops_ref.py`.

use diffusion::latent_ops::{self, Shape};
use serde_json::Value;

fn tensor(v: &Value) -> (Vec<f32>, Vec<usize>) {
    let data = v["data"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let shape = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    (data, shape)
}

fn latent(v: &Value) -> (Vec<f32>, Shape) {
    let (data, shape) = tensor(v);
    (data, shape.try_into().unwrap())
}

#[test]
fn latent_nodes_match_comfy() {
    let cases: Value = serde_json::from_str(include_str!("latent_ops_golden.json")).unwrap();
    let mut worst = 0f32;
    for c in cases.as_array().unwrap() {
        let op = c["op"].as_str().unwrap();
        let (want, want_shape) = latent(&c["out"]);
        let (got, shape) = match op {
            "upscale_by" => {
                let (x, s) = latent(&c["x"]);
                latent_ops::upscale_by(&x, s, c["method"].as_str().unwrap(), c["scale"].as_f64().unwrap()).unwrap()
            }
            "add" => {
                let ((a, sa), (b, sb)) = (latent(&c["a"]), latent(&c["b"]));
                (latent_ops::add(&a, sa, &b, sb).unwrap(), sa)
            }
            "multiply" => {
                let (x, s) = latent(&c["x"]);
                (latent_ops::multiply(&x, c["factor"].as_f64().unwrap()), s)
            }
            "batch" => {
                let ((a, sa), (b, sb)) = (latent(&c["a"]), latent(&c["b"]));
                latent_ops::batch(&a, sa, &b, sb).unwrap()
            }
            "blend_masked" => {
                let ((a, sa), (b, sb)) = (latent(&c["a"]), latent(&c["b"]));
                let (mask, ms) = tensor(&c["mask"]);
                (latent_ops::blend_masked(&a, sa, &b, sb, &mask, (ms[1], ms[2]), c["factor"].as_f64().unwrap()).unwrap(), sa)
            }
            _ => panic!("unknown op {op}"),
        };
        let what = format!("{op} {} {}", c["method"], c["scale"]);
        assert_eq!(shape, want_shape, "{what}");
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max((a - b).abs());
            assert!((a - b).abs() < 1e-5, "{what}: {a} vs {b}");
        }
    }
    eprintln!("worst absolute error {worst:.2e}");
}
