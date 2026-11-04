//! Minimal web UI + JSON API. Models stay loaded on one GPU worker thread; requests are queued.
//!
//!   GET  /                    the UI (web/index.html)
//!   GET  /api/status          {ready, error, queue, max_pixels, defaults}
//!   POST /api/generate        {prompt, seed?, width?, height?, steps?} -> {id}
//!   GET  /api/jobs            recent jobs (newest first)
//!   GET  /api/job/<id>        one job
//!   GET  /api/image/<id>.png  the result

use crate::samplers::ModelSampling;
use crate::pipeline::{encode_png, Generator, KreaRequest, LoadOptions, Prompt, Sampling, Stats};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Response, Server};

const INDEX: &str = include_str!("../web/index.html");
const KEEP_JOBS: usize = 64;

#[derive(Clone)]
struct Job {
    id: u64,
    prompt: String,
    seed: u64,
    width: usize,
    height: usize,
    steps: usize,
    state: &'static str, // queued | running | done | error
    step: usize,
    error: Option<String>,
    stats: Option<Stats>,
    png: Option<Arc<Vec<u8>>>,
    file: Option<String>,
}

impl Job {
    fn json(&self) -> Value {
        json!({
            "id": self.id, "prompt": self.prompt, "seed": self.seed, "width": self.width,
            "height": self.height, "steps": self.steps, "state": self.state, "step": self.step,
            "error": self.error, "file": self.file,
            "stats": self.stats.as_ref().map(|s| json!({
                "text_ms": s.text_ms, "sample_ms": s.sample_ms, "vae_ms": s.vae_ms,
                "total_ms": s.total_ms, "streamed_gb": s.streamed_gb})),
        })
    }
}

#[derive(Default)]
struct State {
    ready: bool,
    error: Option<String>,
    max_pixels: usize,
    next_id: u64,
    jobs: VecDeque<Job>,
}

pub struct ServeOptions {
    pub addr: String,
    pub out_dir: PathBuf,
    pub default_steps: usize,
    pub default_width: usize,
    pub default_height: usize,
    pub shift: f64,
}

fn header(k: &str, v: &str) -> Header {
    Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()
}

fn json_resp(v: Value, code: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(v.to_string()).with_status_code(code).with_header(header("Content-Type", "application/json"))
}

fn random_seed() -> u64 {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    // splitmix64 so consecutive clicks give unrelated seeds
    let mut z = t.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    (z ^ (z >> 31)) % 1_000_000_000_000
}

pub fn run(load: LoadOptions, opts: ServeOptions) -> Result<()> {
    std::fs::create_dir_all(&opts.out_dir)?;
    // ids start at the launch time (ms) so image URLs, which browsers cache forever, never repeat across restarts
    let first_id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    let state = Arc::new(Mutex::new(State { next_id: first_id, ..Default::default() }));
    let (tx, rx) = mpsc::channel::<u64>();

    // GPU worker: owns all CUDA state.
    {
        let state = state.clone();
        let out_dir = opts.out_dir.clone();
        let shift = opts.shift;
        std::thread::spawn(move || {
            let mut gen = match Generator::load_krea(&load) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("model load failed: {e:#}");
                    state.lock().unwrap().error = Some(format!("{e:#}"));
                    return;
                }
            };
            {
                let mut s = state.lock().unwrap();
                s.ready = true;
                s.max_pixels = gen.max_pixels;
            }
            eprintln!("ready");
            for id in rx {
                let req = {
                    let mut s = state.lock().unwrap();
                    let Some(j) = s.jobs.iter_mut().find(|j| j.id == id) else { continue };
                    j.state = "running";
                    let ms = ModelSampling::Flux { shift };
                    let sampling = Sampling::new("euler", "simple", ms, j.steps, 1.0, j.seed, (j.width / 8, j.height / 8)).unwrap();
                    KreaRequest { prompt: Prompt::text(&j.prompt), negative: Prompt::text(""), seed: j.seed, width: j.width, height: j.height, sampling, noise: None, images: Vec::new(), negative_images: Vec::new(), references: Vec::new(), timestep_zero: false }
                };
                let st2 = state.clone();
                let result = gen
                    .generate_krea(&req, |step| {
                        if let Some(j) = st2.lock().unwrap().jobs.iter_mut().find(|j| j.id == id) {
                            j.step = step.index;
                        }
                        true
                    }, None)
                    .and_then(|(rgb, stats)| Ok((encode_png(req.width, req.height, &rgb)?, stats)));
                let mut s = state.lock().unwrap();
                let Some(j) = s.jobs.iter_mut().find(|j| j.id == id) else { continue };
                match result {
                    Ok((png, stats)) => {
                        let name = format!("krea2_{:05}_{}.png", id, req.seed);
                        if let Err(e) = std::fs::write(out_dir.join(&name), &png) {
                            eprintln!("could not save {name}: {e}");
                        } else {
                            j.file = Some(out_dir.join(&name).display().to_string());
                        }
                        j.png = Some(Arc::new(png));
                        j.stats = Some(stats);
                        j.state = "done";
                    }
                    Err(e) => {
                        eprintln!("job {id} failed: {e:#}");
                        j.error = Some(format!("{e:#}"));
                        j.state = "error";
                    }
                }
            }
        });
    }

    let server = Server::http(&opts.addr).map_err(|e| anyhow::anyhow!("binding {}: {e}", opts.addr))?;
    eprintln!("web UI on http://{}", opts.addr);
    for mut req in server.incoming_requests() {
        let url = req.url().split('?').next().unwrap_or("").to_string();
        let resp = match (req.method(), url.as_str()) {
            (Method::Get, "/") | (Method::Get, "/index.html") => {
                Response::from_string(INDEX).with_header(header("Content-Type", "text/html; charset=utf-8"))
            }
            (Method::Get, "/api/status") => {
                let s = state.lock().unwrap();
                let queue = s.jobs.iter().filter(|j| j.state == "queued" || j.state == "running").count();
                json_resp(json!({"ready": s.ready, "error": s.error, "queue": queue, "max_pixels": s.max_pixels,
                    "defaults": {"steps": opts.default_steps, "width": opts.default_width,
                                 "height": opts.default_height}}), 200)
            }
            (Method::Get, "/api/jobs") => {
                let s = state.lock().unwrap();
                json_resp(Value::Array(s.jobs.iter().rev().map(|j| j.json()).collect()), 200)
            }
            (Method::Post, "/api/generate") => {
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                match serde_json::from_str::<Value>(&body) {
                    Ok(v) if v["prompt"].as_str().map_or(false, |p| !p.trim().is_empty()) => {
                        let mut s = state.lock().unwrap();
                        let id = s.next_id;
                        s.next_id += 1;
                        let seed = v["seed"].as_i64().filter(|&x| x >= 0).map(|x| x as u64).unwrap_or_else(random_seed);
                        let u = |k: &str, d: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(d);
                        s.jobs.push_back(Job {
                            id, prompt: v["prompt"].as_str().unwrap().trim().to_string(), seed,
                            width: u("width", opts.default_width), height: u("height", opts.default_height),
                            steps: u("steps", opts.default_steps), state: "queued", step: 0, error: None,
                            stats: None, png: None, file: None,
                        });
                        while s.jobs.len() > KEEP_JOBS && s.jobs.front().map_or(false, |j| j.state == "done" || j.state == "error") {
                            s.jobs.pop_front();
                        }
                        let _ = tx.send(id);
                        json_resp(json!({"id": id, "seed": seed}), 200)
                    }
                    _ => json_resp(json!({"error": "expected JSON with a non-empty \"prompt\""}), 400),
                }
            }
            (Method::Get, u) if u.starts_with("/api/job/") => {
                let id: u64 = u["/api/job/".len()..].parse().unwrap_or(0);
                let s = state.lock().unwrap();
                match s.jobs.iter().find(|j| j.id == id) {
                    Some(j) => json_resp(j.json(), 200),
                    None => json_resp(json!({"error": "no such job"}), 404),
                }
            }
            (Method::Get, u) if u.starts_with("/api/image/") => {
                let id: u64 = u["/api/image/".len()..].trim_end_matches(".png").parse().unwrap_or(0);
                let png = state.lock().unwrap().jobs.iter().find(|j| j.id == id).and_then(|j| j.png.clone());
                match png {
                    Some(p) => Response::from_data(p.as_ref().clone())
                        .with_header(header("Content-Type", "image/png"))
                        .with_header(header("Cache-Control", "max-age=31536000, immutable")),
                    None => json_resp(json!({"error": "not ready"}), 404),
                }
            }
            _ => json_resp(json!({"error": "not found"}), 404),
        };
        let _ = req.respond(resp);
    }
    Ok(())
}
