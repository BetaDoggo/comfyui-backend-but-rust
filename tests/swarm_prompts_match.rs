//! Token ids and weights against SwarmTextEncodeAdvanced's own code, recorded by `tools/swarm_prompt_ref.py`.

use diffusion::pipeline::{h3_tokens, moss_tokens};
use diffusion::tokenizer::Tokenizer;
use serde_json::Value;

/// What `prompt_chunks` makes of the recorded prompts' weights: leaf text and weight, by chunk.
fn segments(prompt: &str) -> Vec<Vec<(String, f32)>> {
    let s = |t: &str, w: f32| (t.to_string(), w);
    match prompt {
        "a <weight[1.5]:red> boat" => vec![vec![s("a ", 1.0), s("red", 1.5), s(" boat", 1.0)]],
        "<weight[0.5]:a lake>" => vec![vec![s("a lake", 0.5)]],
        "lake<break>pine forest" => vec![vec![s("lake", 1.0)], vec![s("pine forest", 1.0)]],
        "lake, <weight[1.3]:pine <weight[2]:bark>> (sky:1.2)" => vec![vec![s("lake, ", 1.0), s("pine ", 1.3), s("bark", 2.6), s(" (sky:1.2)", 1.0)]],
        "" => vec![vec![]],
        p => vec![vec![s(p, 1.0)]],
    }
}

fn check(cases: &Value, tokens: impl Fn(&[(String, f32)]) -> (Vec<u32>, Vec<f32>)) {
    for c in cases.as_array().unwrap() {
        let prompt = c["prompt"].as_str().unwrap();
        let chunks = c["chunks"].as_array().unwrap();
        let got = segments(prompt);
        assert_eq!(got.len(), chunks.len(), "{prompt}");
        for (segs, want) in got.iter().zip(chunks) {
            let (ids, factors) = tokens(segs);
            let want_ids: Vec<u32> = want["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(ids, want_ids, "{prompt}");
            let want_weights = want["weights"].as_array().unwrap();
            assert_eq!(factors.len(), want_weights.len(), "{prompt}");
            for (a, b) in factors.iter().zip(want_weights) {
                assert!((*a as f64 - b.as_f64().unwrap()).abs() < 1e-5, "{prompt}: {factors:?}");
            }
        }
    }
}

#[test]
fn moss_and_h3_tokens_match_swarm() {
    let g: Value = serde_json::from_str(include_str!("swarm_tokens_golden.json")).unwrap();
    let t = Tokenizer::qwen35().unwrap();
    check(&g["moss"], |s| moss_tokens(&t, s).unwrap());
    let t = Tokenizer::new().unwrap();
    check(&g["h3"], |s| h3_tokens(&t, s).unwrap());
}
