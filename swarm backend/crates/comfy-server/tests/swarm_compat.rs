//! End-to-end tests that drive the server exactly the way SwarmUI's
//! `ComfyUIAPIAbstractBackend` does (see `AwaitJobLive` / `LoadValueSet`), using
//! workflows shaped like SwarmUI's `WorkflowGenerator` output for Krea 2, Anima,
//! MiniMax H3, MOSS Image, and Qwen Image 2.1. (The MOSS and Qwen 2.1 fixtures were
//! captured from a live SwarmUI via `ComfyGetGeneratedWorkflow`.)

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use comfy_server::config::Config;
use comfy_server::encode;
use comfy_server::mock_engine::MockEngine;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use swarm_engine_api::*;
use tokio_tungstenite::tungstenite::Message;

const MODEL_FILES: &[&str] = &[
    "diffusion_models/krea2/krea2_fp8_scaled.safetensors",
    "diffusion_models/anima/anima-preview.safetensors",
    "diffusion_models/minimax_h3_fp8.safetensors",
    "text_encoders/qwen3vl_4b.safetensors",
    "text_encoders/qwen_3_600m.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "VAE/qwen_image_vae.safetensors",
    "VAE/minimax_h3_video_vae_int8.safetensors",
    "VAE/minimax_h3_audio_vae.safetensors",
    "diffusion_models/moss_image_sft_int8.safetensors",
    "diffusion_models/qwen_image_2.1_fp8.safetensors",
    "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "VAE/Flux/flux2-vae.safetensors",
    "VAE/QwenImage/qwen_image_2.1_vae_bf16.safetensors",
    "Lora/anima/flat_color_style.safetensors",
    "Lora/qwen21_style.safetensors",
    "Embeddings/bad_hands.safetensors",
    "gligen/gligen_test.safetensors",
    "upscale_models/4x_test.safetensors",
    "latent_upscale_models/latent_up_test.safetensors",
    "clip_vision/clip_vision_test.safetensors",
    "clip_vision/sigclip_vision_patch14_384.safetensors",
    "controlnet/controlnet_test.safetensors",
    "Stable-Diffusion/svd_test.safetensors",
    "controlnet/fake_union_controlnet.safetensors",
    "controlnet/fake_alimama_inpaint.safetensors",
    "controlnet/fake_qwen_diffpatch.safetensors",
    "controlnet/fake_anima_lllite.safetensors",
    "style_models/redux_test.safetensors",
    "model_patches/patch_test.safetensors",
    "yolov8/face_yolov8m.pt",
];

struct TestServer {
    addr: SocketAddr,
    http: reqwest::Client,
}

impl TestServer {
    async fn start(step_ms: u64) -> Self {
        Self::start_with(step_ms, false).await
    }

    async fn start_with(step_ms: u64, advertise_todo_nodes: bool) -> Self {
        Self::start_engine(Arc::new(MockEngine::new(step_ms)), step_ms, advertise_todo_nodes).await
    }

    /// A mock that doesn't claim any backlog node through `custom_node`.
    async fn start_native(advertise_todo_nodes: bool) -> Self {
        Self::start_engine(Arc::new(MockEngine::native_only(0)), 0, advertise_todo_nodes).await
    }

    async fn start_engine(engine: Arc<dyn Engine>, step_ms: u64, advertise_todo_nodes: bool) -> Self {
        let root: PathBuf = std::env::temp_dir().join(format!("comfy-server-test-{}", uuid::Uuid::new_v4().simple()));
        for f in MODEL_FILES {
            let p = root.join("models").join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"not really weights").unwrap();
        }
        let mut cfg = Config::for_tests(&root);
        cfg.mock_step_ms = step_ms;
        cfg.advertise_todo_nodes = advertise_todo_nodes;
        let (addr, _) = comfy_server::spawn(&cfg, engine).await.unwrap();
        Self { addr, http: reqwest::Client::new() }
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}/{path}", self.addr)
    }

    async fn get(&self, path: &str) -> Value {
        self.http.get(self.url(path)).send().await.unwrap().json().await.unwrap()
    }

    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        self.http.post(self.url(path)).body(body.to_string()).send().await.unwrap()
    }
}

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    with_test_media(std::fs::read_to_string(path).unwrap())
}

/// Replaces the `$IMAGE_B64` / `$AUDIO_B64` (...) placeholders with generated test media.
fn with_test_media(mut text: String) -> Value {
    if text.contains("$IMAGE_B64") {
        let img = ImageBatch::from_u8(1, 200, 300, 3, (0..200 * 300 * 3).map(|i| (i % 251) as u8).collect());
        text = text.replace("$IMAGE_B64", &base64::engine::general_purpose::STANDARD.encode(encode::png8(&img, 0).unwrap()));
    }
    if text.contains("$IMAGE2_B64") {
        let img = ImageBatch::from_u8(1, 96, 128, 3, (0..96 * 128 * 3).map(|i| (i % 199) as u8).collect());
        text = text.replace("$IMAGE2_B64", &base64::engine::general_purpose::STANDARD.encode(encode::png8(&img, 0).unwrap()));
    }
    if text.contains("$AUDIO_B64") {
        let a = AudioData { sample_rate: 16000, channels: vec![Arc::new((0..16000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect())] };
        text = text.replace("$AUDIO_B64", &base64::engine::general_purpose::STANDARD.encode(encode::wav(&a)));
    }
    serde_json::from_str(&text).unwrap()
}

#[derive(Debug)]
struct Output {
    format: u32,
    node: String,
    bytes: Vec<u8>,
}

#[derive(Debug, Default)]
struct RunResult {
    prompt_id: String,
    outputs: Vec<Output>,
    previews: usize,
    message_types: Vec<String>,
    cached_nodes: Vec<String>,
    history: Value,
}

impl RunResult {
    fn history_entry(&self) -> &Value {
        &self.history[&self.prompt_id]
    }

    fn status(&self) -> &str {
        self.history_entry()["status"]["status_str"].as_str().unwrap_or("")
    }
}

/// A port of SwarmUI's `AwaitJobLive` loop (the parts that matter for compatibility).
async fn run_like_swarm(server: &TestServer, workflow: Value, mut on_progress: impl FnMut(&str)) -> RunResult {
    let client_id = uuid::Uuid::new_v4().to_string();
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/ws?clientId={client_id}", server.addr)).await.unwrap();
    ws.send(Message::Text(json!({ "type": "feature_flags", "data": { "supports_preview_metadata": true } }).to_string().into())).await.unwrap();

    let resp = server.post("prompt", json!({ "prompt": workflow, "client_id": client_id })).await;
    let body: Value = resp.json().await.unwrap();
    assert!(body.get("error").is_none(), "prompt rejected: {body:#}");
    let prompt_id = body["prompt_id"].as_str().unwrap().to_string();

    let mut r = RunResult { prompt_id: prompt_id.clone(), ..Default::default() };
    let (mut is_me, mut receiving_outputs, mut current_node) = (false, false, String::new());
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(120), ws.next()).await.expect("timed out waiting for websocket").unwrap().unwrap();
        match msg {
            Message::Text(text) => {
                // SwarmUI's JSON detection: first 8 bytes must be exactly `{"type":`.
                assert!(text.as_str().starts_with("{\"type\":"), "bad JSON framing: {text}");
                let v: Value = serde_json::from_str(&text).unwrap();
                let ty = v["type"].as_str().unwrap().to_string();
                if !is_me {
                    if ty == "execution_start" && v["data"]["prompt_id"] == prompt_id.as_str() {
                        is_me = true;
                    } else {
                        continue;
                    }
                }
                r.message_types.push(ty.clone());
                match ty.as_str() {
                    "executing" => {
                        if v["data"]["node"].is_null() {
                            break;
                        }
                        current_node = v["data"]["node"].as_str().unwrap().to_string();
                    }
                    "execution_cached" => r.cached_nodes = v["data"]["nodes"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_string()).collect(),
                    "progress" => {
                        let max = v["data"]["max"].as_u64().unwrap();
                        receiving_outputs = matches!(max, 12345 | 12346 | 12347);
                        on_progress(&prompt_id);
                    }
                    _ => {}
                }
            }
            Message::Binary(data) => {
                let event = u32::from_be_bytes(data[0..4].try_into().unwrap());
                let format = u32::from_be_bytes(data[4..8].try_into().unwrap());
                if receiving_outputs {
                    assert!(event == 1 || event == 10, "unexpected output event {event}");
                    r.outputs.push(Output { format, node: current_node.clone(), bytes: data[8..].to_vec() });
                } else {
                    assert_eq!(event, 1, "previews should be PREVIEW_IMAGE events");
                    r.previews += 1;
                }
            }
            _ => {}
        }
    }
    r.history = server.get(&format!("history/{prompt_id}")).await;
    r
}

fn png_dims(bytes: &[u8]) -> (u32, u32) {
    let img = image::load_from_memory(bytes).unwrap();
    (img.width(), img.height())
}

/// `(min, max)` of the PNG's alpha channel, or `None` if it has no alpha.
fn png_alpha_range(bytes: &[u8]) -> Option<(u8, u8)> {
    let img = image::load_from_memory(bytes).unwrap();
    if !img.color().has_alpha() {
        return None;
    }
    let a: Vec<u8> = img.to_rgba8().pixels().map(|p| p[3]).collect();
    Some((*a.iter().min().unwrap(), *a.iter().max().unwrap()))
}

fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
}

#[tokio::test(flavor = "multi_thread")]
async fn object_info_has_what_swarm_reads() {
    let s = TestServer::start(0).await;
    let info = s.get("object_info").await;
    let obj = info.as_object().unwrap();
    // SwarmUI warns "missing the Swarm core nodes" without SwarmKSampler, and maps these nodes to features.
    for node in ["SwarmKSampler", "SwarmSaveImageWS", "SwarmLoadImageB64", "SwarmJustLoadTheModelPlease", "SwarmSaveAudioWS", "SwarmEmbedLoaderListProvider", "EmptyFlux2LatentImage", "SplitImageWithAlpha", "JoinImageWithAlpha"] {
        assert!(obj.contains_key(node), "missing {node}");
    }
    // AssignValuesFromRaw dereferences `category` on every node.
    for (k, v) in obj {
        assert!(v["category"].is_string(), "{k} has no category");
    }
    let unets: Vec<&str> = info["UNETLoader"]["input"]["required"]["unet_name"][0].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(unets.contains(&"krea2/krea2_fp8_scaled.safetensors") && unets.contains(&"minimax_h3_fp8.safetensors"), "{unets:?}");
    let loras = &info["LoraLoader"]["input"]["required"]["lora_name"][0];
    assert_eq!(loras, &json!(["anima/flat_color_style.safetensors", "qwen21_style.safetensors"]));
    let samplers = &info["SwarmKSampler"]["input"]["required"]["sampler_name"][0];
    for s in ["euler", "er_sde", "res_multistep"] {
        assert!(samplers.as_array().unwrap().contains(&json!(s)));
    }
    let clip_types = &info["CLIPLoader"]["input"]["required"]["type"][0];
    for t in ["krea2", "stable_diffusion", "minimax"] {
        assert!(clip_types.as_array().unwrap().contains(&json!(t)));
    }
    assert_eq!(s.get("embeddings").await, json!(["bad_hands"]));
    assert!(s.get("system_stats").await["devices"].is_array());
    assert!(s.get("features").await.is_object());
    assert_eq!(s.get("prompt").await["exec_info"]["queue_remaining"], 0);
    assert_eq!(s.get("api/object_info/SwarmKSampler").await["SwarmKSampler"]["name"], "SwarmKSampler");
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_text_to_image() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("krea2_t2i.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(r.outputs.len(), 1);
    assert_eq!(r.outputs[0].format, 2, "PNG");
    assert_eq!(r.outputs[0].node, "9", "SwarmUI treats node 9 as the real output");
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
    assert!(r.previews >= 1);
    assert_eq!(r.message_types.first().map(String::as_str), Some("execution_start"));
    assert!(r.message_types.contains(&"execution_success".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_with_reference_image() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("krea2_reference_image.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (768, 768));
}

#[tokio::test(flavor = "multi_thread")]
async fn anima_batch_with_lora() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("anima_t2i_lora.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(r.outputs.len(), 2, "batch of 2");
    for o in &r.outputs {
        assert_eq!(png_dims(&o.bytes), (832, 1216));
    }
}

async fn check_h3_video(fixture_name: &str) {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture(fixture_name), |_| {}).await;
    if !ffmpeg_available() {
        assert_eq!(r.status(), "error", "without ffmpeg, mp4 output should fail cleanly");
        return;
    }
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(r.outputs.len(), 1);
    let o = &r.outputs[0];
    assert_eq!(o.format, 5, "mp4");
    assert_eq!(&o.bytes[4..8], b"ftyp");
    // Check the container actually has 22 frames of 448x256 video plus an audio track.
    let path = std::env::temp_dir().join(format!("h3-{}.mp4", uuid::Uuid::new_v4().simple()));
    std::fs::write(&path, &o.bytes).unwrap();
    let probe = std::process::Command::new("ffmpeg").args(["-i", path.to_str().unwrap(), "-map", "0:v", "-f", "null", "-"]).output().unwrap();
    let info = String::from_utf8_lossy(&probe.stderr);
    let _ = std::fs::remove_file(&path);
    assert!(info.contains("448x256"), "{info}");
    assert!(info.contains("Audio:"), "missing audio track: {info}");
    assert!(info.contains("frame=   22") || info.contains("frame=22"), "expected 22 frames: {info}");
}

#[tokio::test(flavor = "multi_thread")]
async fn minimax_h3_text_to_video_with_audio() {
    check_h3_video("minimax_h3_t2v.json").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn minimax_h3_image_to_video_with_references() {
    check_h3_video("minimax_h3_i2v_refs.json").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn moss_image_text_to_image() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("moss_t2i.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(r.outputs.len(), 1);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
    assert_eq!(png_alpha_range(&r.outputs[0].bytes), None, "Flux 2 VAE is RGB");
}

#[tokio::test(flavor = "multi_thread")]
async fn moss_image_with_prompt_image() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("moss_prompt_image.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
}

#[tokio::test(flavor = "multi_thread")]
async fn qwen21_half_size_placeholder_latent_decodes_full_size_rgba() {
    let s = TestServer::start(0).await;
    let wf = fixture("qwen21_t2i_alpha.json");
    assert_eq!((wf["5"]["inputs"]["width"].as_u64(), wf["5"]["inputs"]["height"].as_u64()), (Some(512), Some(384)), "Swarm sends half size");
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 768));
    let (lo, hi) = png_alpha_range(&r.outputs[0].bytes).expect("RGBA PNG");
    assert!(lo < 16 && hi > 240, "mock alpha gradient should survive ({lo}..{hi})");
}

#[tokio::test(flavor = "multi_thread")]
async fn qwen21_edit_with_two_reference_images() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("qwen21_edit_two_images.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
}

#[tokio::test(flavor = "multi_thread")]
async fn qwen21_refiner_upscale_keeps_alpha() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("qwen21_refiner_upscale.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
    assert!(png_alpha_range(&r.outputs[0].bytes).is_some());
}

/// H3, MOSS and Qwen 2.1 have `LorasTargetTextEnc = false`, so SwarmUI sends `LoraLoaderModelOnly`
/// (inserted between the UNET loader and the sampler, exactly as captured from a live SwarmUI).
#[tokio::test(flavor = "multi_thread")]
async fn qwen21_model_only_lora() {
    let s = TestServer::start(0).await;
    let mut wf = fixture("qwen21_t2i_alpha.json");
    wf["2000"] = json!({"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": "qwen21_style.safetensors", "strength_model": 0.8}});
    wf["10"]["inputs"]["model"] = json!(["2000", 0]);
    let r = run_like_swarm(&s, wf.clone(), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 768));

    wf["2000"]["inputs"]["lora_name"] = json!("missing.safetensors");
    let body: Value = s.post("prompt", json!({ "prompt": wf, "client_id": "x" })).await.json().await.unwrap();
    assert_eq!(body["node_errors"]["2000"]["errors"][0]["type"], "value_not_in_list", "{body:#}");
}

#[tokio::test(flavor = "multi_thread")]
async fn split_and_join_alpha_roundtrip() {
    let s = TestServer::start(0).await;
    let mut wf = fixture("qwen21_t2i_alpha.json");
    wf["20"] = json!({"class_type": "SplitImageWithAlpha", "inputs": {"image": ["8", 0]}});
    wf["21"] = json!({"class_type": "ImageScale", "inputs": {"image": ["20", 0], "width": 512, "height": 384, "upscale_method": "lanczos", "crop": "disabled"}});
    wf["22"] = json!({"class_type": "JoinImageWithAlpha", "inputs": {"image": ["21", 0], "alpha": ["20", 1]}});
    wf["9"]["inputs"]["images"] = json!(["22", 0]);
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (512, 384));
    let (lo, hi) = png_alpha_range(&r.outputs[0].bytes).expect("RGBA PNG");
    assert!(lo < 16 && hi > 240, "alpha lost in split/join ({lo}..{hi})");
}

#[tokio::test(flavor = "multi_thread")]
async fn model_preload_and_loader_cache() {
    let s = TestServer::start(0).await;
    let first = run_like_swarm(&s, fixture("model_preload.json"), |_| {}).await;
    assert_eq!(first.status(), "success");
    assert!(first.outputs.is_empty());
    assert!(first.cached_nodes.is_empty());
    let second = run_like_swarm(&s, fixture("model_preload.json"), |_| {}).await;
    let mut cached = second.cached_nodes.clone();
    cached.sort();
    assert_eq!(cached, vec!["100", "101", "4"]);
    s.post("free", json!({ "unload_models": true, "free_memory": true })).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    let third = run_like_swarm(&s, fixture("model_preload.json"), |_| {}).await;
    assert!(third.cached_nodes.is_empty(), "free should drop cached loaders");
}

#[tokio::test(flavor = "multi_thread")]
async fn validation_errors_match_comfy_shape() {
    let s = TestServer::start(0).await;
    let mut wf = fixture("krea2_t2i.json");
    wf["4"]["inputs"]["unet_name"] = json!("does_not_exist.safetensors");
    wf["10"]["inputs"]["steps"] = json!(0);
    let resp = s.post("prompt", json!({ "prompt": wf, "client_id": "x" })).await;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "prompt_outputs_failed_validation");
    assert_eq!(body["node_errors"]["4"]["errors"][0]["type"], "value_not_in_list");
    assert_eq!(body["node_errors"]["10"]["errors"][0]["type"], "value_smaller_than_min");

    let mut wf = fixture("krea2_t2i.json");
    wf["8"]["class_type"] = json!("SomeCustomNode");
    let body: Value = s.post("prompt", json!({ "prompt": wf })).await.json().await.unwrap();
    assert_eq!(body["error"]["type"], "missing_node_type");

    let body: Value = s.post("prompt", json!({ "prompt": { "1": { "class_type": "VAELoader", "inputs": { "vae_name": "qwen_image_vae.safetensors" } } } })).await.json().await.unwrap();
    assert_eq!(body["error"]["type"], "prompt_no_outputs");
}

/// Wraps the mock engine and records the patches each heavy call receives.
#[derive(Default)]
struct Recorded {
    model_patches: Vec<String>,
    text_encoder_patches: Vec<String>,
    vae_patches: Vec<String>,
    /// `(noise_mask, audio_noise_mask)` of every latent handed to `sample`, in order.
    sample_masks: Vec<(Option<Mask>, Option<Mask>)>,
    /// Durations (seconds) of every clip handed to `vae_encode_audio`.
    encoded_audio: Vec<f64>,
    /// Metadata of every positive conditioning entry handed to the last `sample`.
    positive: Vec<CondMeta>,
    negative: Vec<CondMeta>,
    /// `(sampler, seed, sigmas, cfg_conds, cfg_cond2_negative, nested)` of every `sample_custom`.
    custom_samples: Vec<(String, u64, Vec<f64>, f64, f64, bool)>,
    /// `(class, file inputs by name)` of every `custom_node` call.
    custom_nodes: Vec<(String, Vec<(String, String)>)>,
    /// Every `latent_op` (variant name) and `unsample` call, in order.
    latent_ops: Vec<String>,
}

struct RecordingEngine {
    inner: MockEngine,
    seen: std::sync::Mutex<Recorded>,
    /// Makes the next `encode_text` panic (once).
    panic_next_encode: std::sync::atomic::AtomicBool,
}

impl RecordingEngine {
    fn new() -> Arc<Self> {
        Arc::new(Self { inner: MockEngine::new(0), seen: Default::default(), panic_next_encode: Default::default() })
    }
}

impl Engine for RecordingEngine {
    fn info(&self) -> EngineInfo {
        self.inner.info()
    }
    fn load_checkpoint(&self, r: &LoadCheckpoint) -> EngineResult<(ModelHandle, TextEncoderHandle, VaeHandle)> {
        self.inner.load_checkpoint(r)
    }
    fn load_diffusion_model(&self, r: &LoadDiffusionModel) -> EngineResult<ModelHandle> {
        self.inner.load_diffusion_model(r)
    }
    fn load_text_encoder(&self, r: &LoadTextEncoder) -> EngineResult<TextEncoderHandle> {
        self.inner.load_text_encoder(r)
    }
    fn load_vae(&self, r: &LoadVae) -> EngineResult<VaeHandle> {
        self.inner.load_vae(r)
    }
    fn encode_text(&self, r: &TextEncodeRequest) -> EngineResult<Conditioning> {
        if self.panic_next_encode.swap(false, std::sync::atomic::Ordering::SeqCst) {
            panic!("engine bug");
        }
        self.seen.lock().unwrap().text_encoder_patches = r.text_encoder.patches.iter().map(|p| format!("{p:?}")).collect();
        self.inner.encode_text(r)
    }
    fn empty_latent(&self, r: &EmptyLatentRequest) -> EngineResult<Latent> {
        self.inner.empty_latent(r)
    }
    fn sample(&self, r: &SampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        self.seen.lock().unwrap().sample_masks.push((r.latent.noise_mask.clone(), r.latent.audio_noise_mask.clone()));
        self.seen.lock().unwrap().positive = r.positive.entries.iter().map(|e| e.meta.clone()).collect();
        self.seen.lock().unwrap().negative = r.negative.entries.iter().map(|e| e.meta.clone()).collect();
        self.seen.lock().unwrap().model_patches = r
            .model
            .patches
            .iter()
            .map(|p| match p {
                ModelPatch::AttnTokenWeights { .. } => "AttnTokenWeights".to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        self.inner.sample(r, progress)
    }
    fn vae_decode(&self, r: &VaeDecodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<ImageBatch> {
        self.seen.lock().unwrap().vae_patches = r.vae.patches.iter().map(|p| format!("{p:?}")).collect();
        self.inner.vae_decode(r, progress)
    }
    fn vae_encode(&self, r: &VaeEncodeRequest) -> EngineResult<Latent> {
        self.inner.vae_encode(r)
    }
    fn vae_decode_audio(&self, r: &VaeDecodeAudioRequest) -> EngineResult<AudioData> {
        self.inner.vae_decode_audio(r)
    }
    fn vae_encode_audio(&self, r: &VaeEncodeAudioRequest) -> EngineResult<Latent> {
        self.seen.lock().unwrap().encoded_audio.push(r.audio.duration_seconds());
        self.inner.vae_encode_audio(r)
    }
    fn audio_silence(&self, r: &AudioSilenceRequest) -> EngineResult<Latent> {
        self.inner.audio_silence(r)
    }
    fn latent_op(&self, op: &LatentOp) -> EngineResult<Latent> {
        let name = format!("{op:?}");
        self.seen.lock().unwrap().latent_ops.push(name.split([' ', '{', '(']).next().unwrap().to_string());
        self.inner.latent_op(op)
    }
    fn supports_custom_node(&self, class: &str) -> bool {
        self.inner.supports_custom_node(class)
    }
    fn custom_node(&self, r: &CustomNodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<Vec<NodeValue>> {
        let files = r.inputs.iter().filter_map(|(k, v)| if let NodeValue::File(f) = v { Some((k.clone(), format!("{:?}/{}", f.folder, f.name))) } else { None }).collect();
        self.seen.lock().unwrap().custom_nodes.push((r.class_type.clone(), files));
        self.inner.custom_node(r, progress)
    }
    fn model_sigmas(&self, r: &ScheduleRequest) -> EngineResult<Sigmas> {
        self.inner.model_sigmas(r)
    }
    fn sample_custom(&self, r: &CustomSampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<CustomSampleOutput> {
        let Guider::DualCfg { cfg_conds, cfg_cond2_negative, nested, .. } = &r.guider;
        self.seen.lock().unwrap().custom_samples.push((r.sampler.name.clone(), r.noise.seed, r.sigmas.0.clone(), *cfg_conds, *cfg_cond2_negative, *nested));
        self.inner.sample_custom(r, progress)
    }
    fn clip_vision_encode(&self, r: &ClipVisionEncodeRequest) -> EngineResult<ClipVisionOutput> {
        self.inner.clip_vision_encode(r)
    }
    fn unsample(&self, r: &UnsampleRequest, progress: &mut dyn ProgressSink) -> EngineResult<Latent> {
        self.seen.lock().unwrap().latent_ops.push(format!("Unsample({}..{})", r.start_at_step, r.steps));
        self.inner.unsample(r, progress)
    }
}

fn gif_frames(bytes: &[u8]) -> usize {
    use image::AnimationDecoder;
    image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes)).unwrap().into_frames().count()
}

// The video fixtures below were captured from SwarmUI for MiniMax H3 (22 frames at 448x256),
// with the save format switched from h264-mp4 to gif so they don't need ffmpeg.

#[tokio::test(flavor = "multi_thread")]
async fn h3_trim_then_boomerang() {
    let s = TestServer::start(0).await;
    // Trim Video Start Frames 2, End Frames 3, Video Boomerang. SwarmUI also leaves an unused
    // boomerang node on the untrimmed frames, which must not run or break anything.
    let r = run_like_swarm(&s, fixture("h3_boomerang_trim.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(r.outputs.len(), 1);
    assert_eq!(gif_frames(&r.outputs[0].bytes), (22 - 2 - 3) * 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn h3_video_audio_input_is_encoded_and_kept() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let r = run_like_swarm(&s, fixture("h3_audio_input.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    // The 1 s test clip is already longer than SwarmEnsureAudio's 0.1 s minimum.
    assert_eq!(seen.encoded_audio, [1.0]);
    // SolidMask(0) on the audio half: sampling keeps the encoded audio, the video is free.
    let (video_mask, audio_mask) = &seen.sample_masks[0];
    assert!(video_mask.is_none());
    let audio_mask = audio_mask.as_ref().expect("audio noise mask");
    assert!(audio_mask.data.iter().all(|v| *v == 0.0));
}

#[tokio::test(flavor = "multi_thread")]
async fn h3_silent_prefix_and_suffix_mask_the_audio_ends() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Audio Silent Prefix Duration 0.5 s, Suffix 0.25 s.
    let r = run_like_swarm(&s, fixture("h3_silent_prefix_suffix.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    let (video_mask, audio_mask) = &seen.sample_masks[0];
    assert!(video_mask.is_none());
    // 22 frames at 24 fps -> 37 audio latent frames at 40/s: 20 kept at the start, 10 at the end.
    let m = audio_mask.as_ref().expect("audio noise mask");
    assert_eq!(m.width, 37);
    assert!(m.data[..20].iter().all(|v| *v == 0.0) && m.data[27..].iter().all(|v| *v == 0.0));
    assert!(m.data[20..27].iter().all(|v| *v == 1.0));
}

#[tokio::test(flavor = "multi_thread")]
async fn h3_extend_chains_a_second_clip() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // `a cat walks <extend:20> the cat runs`: frame-count math with SwarmCountFrames/IntAdd/
    // ImageWidth/ImageHeight feeding INT inputs, a second keyframed H3 run, then the clips
    // (and their audio, via AudioConcat) joined.
    let r = run_like_swarm(&s, fixture("h3_extend.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(engine.seen.lock().unwrap().sample_masks.len(), 2);
    assert_eq!(r.outputs.len(), 2);
    let frames: Vec<usize> = r.outputs.iter().map(|o| gif_frames(&o.bytes)).collect();
    // Intermediate first clip, then first clip + the 11 new frames of the extension.
    assert!(frames.contains(&22) && frames.contains(&33), "{frames:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_to_h3_with_video_end_image() {
    let s = TestServer::start(0).await;
    // Krea 2 image, then MiniMax H3 image-to-video with Video End Image: BatchImagesNode joins
    // the start and end images (different sizes) as text-encoder references.
    let r = run_like_swarm(&s, fixture("krea2_i2v_end_image.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(r.outputs.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn guidance_tweaks_reach_the_engine_in_graph_order() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Captured from SwarmUI: Krea 2 with every Alternate Guidance option, FreeU v2, Seamless Tileable
    // X-Only, CLIP Stop At Layer -2, EasyCache, attention backend, Sol-Attn sparse attention, Torch Compile.
    let r = run_like_swarm(&s, fixture("krea2_guidance_tweaks.json"), |_| {}).await;
    assert_eq!(r.status(), "success");
    assert_eq!(r.outputs.len(), 1);
    let seen = engine.seen.lock().unwrap();
    assert_eq!(
        seen.model_patches,
        [
            "FreeU { version: 2, b1: 1.1, b2: 1.2, s1: 0.9, s2: 0.2 }",
            "SelfAttentionGuidance { scale: 0.6, blur_sigma: 1.5 }",
            "PerturbedAttentionGuidance { scale: 3.0 }",
            "RescaleCfg { multiplier: 0.6 }",
            "RenormCfg { cfg_trunc: 100.0, renorm_cfg: 1.2 }",
            "CfgZeroStar",
            "Tcfg",
            "NormalizedAttentionGuidance { scale: 5.0, alpha: 0.4, tau: 2.0 }",
            "Tiling { axis: X }",
            "EasyCache { reuse_threshold: 0.3, start_percent: 0.0, end_percent: 1.0, verbose: false }",
            // SwarmUI's "pytorch attention" default becomes the engine's first backend.
            "AttentionBackend { backend: \"default\" }",
            "BlockSparseAttention(SparseAttention { method: SolAttn { tau: 1.3 }, start_percent: 0.2, end_percent: 1.0, dense_blocks: \"\", min_tokens: 12288, extra_tokens: 256, sink_conditioning: \"exact_kv_and_rows\", verbose: false })",
            "Compile { backend: \"inductor\" }",
            "AttnTokenWeights",
        ]
    );
    assert_eq!(seen.text_encoder_patches, ["StopAtLayer { layer: -2 }"]);
    assert_eq!(seen.vae_patches, ["Tiling { axis: X }"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn guidance_node_schemas_cover_swarm_inputs() {
    let s = TestServer::start(0).await;
    let info = s.get("object_info").await;
    // Swarm reads this list for its "Model Attention Backend" dropdown; its own default must validate.
    assert_eq!(info["ModelAttentionBackend"]["input"]["required"]["attention"][0], json!(["pytorch attention", "default", "flash"]));
    // V3 DynamicCombo rendering, as in stock ComfyUI.
    let sel = &info["BlockSparseAttention"]["input"]["required"]["selection"];
    assert_eq!(sel[0], "COMFY_DYNAMICCOMBO_V3");
    assert_eq!(sel[1]["options"][0]["key"], "sol-attn");
    assert_eq!(sel[1]["options"][0]["inputs"]["required"]["tau"][1]["default"], 1.3);

    // DynamicCombo sub-inputs are validated against the chosen option's spec.
    let mut wf = fixture("krea2_guidance_tweaks.json");
    let sparse = wf.as_object().unwrap().iter().find(|(_, n)| n["class_type"] == "BlockSparseAttention").unwrap().0.clone();
    wf[&sparse]["inputs"]["selection.tau"] = json!(9.0);
    let body: Value = s.post("prompt", json!({ "prompt": wf })).await.json().await.unwrap();
    assert_eq!(body["node_errors"][&sparse]["errors"][0]["type"], "value_bigger_than_max");
    wf[&sparse]["inputs"]["selection"] = json!("nope");
    let body: Value = s.post("prompt", json!({ "prompt": wf })).await.json().await.unwrap();
    assert_eq!(body["node_errors"][&sparse]["errors"][0]["type"], "value_not_in_list");
}

fn mask_at(m: &Mask, x: f64, y: f64) -> f32 {
    m.data[(y * m.height as f64) as usize * m.width as usize + (x * m.width as f64) as usize]
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_regional_prompting_masks_and_strengths() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Captured from SwarmUI: `a park <region:0,0,0.6,0.6> a red balloon <region:0.4,0.4,0.6,0.6,-0.5> a blue kite`,
    // Global Region Factor 0.3, Debug Regional Prompting on.
    let r = run_like_swarm(&s, fixture("krea2_regions.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    // The image plus a debug mask image for the background and each region.
    assert_eq!(r.outputs.len(), 4);
    let seen = engine.seen.lock().unwrap();
    // background, region 1, region 2, then the global prompt over everything.
    let strengths: Vec<f64> = seen.positive.iter().map(|m| m.mask.as_ref().expect("every entry is masked").strength).collect();
    assert_eq!(strengths.len(), 4);
    assert!((strengths[0] - 0.7).abs() < 1e-9 && (strengths[1] - 0.7).abs() < 1e-9 && (strengths[2] - 0.7).abs() < 1e-9 && (strengths[3] - 0.3).abs() < 1e-9, "{strengths:?}");
    let m: Vec<&Mask> = seen.positive.iter().map(|e| &e.mask.as_ref().unwrap().mask).collect();
    assert_eq!((m[0].width, m[0].height), (256, 256));
    // Region 2 has strength -0.5: its mask is inverted, 1.0 outside its box and 0.5 inside.
    // Top-left is inside region 1 and outside region 2: both cover it fully, so each gets half.
    assert_eq!((mask_at(m[1], 0.1, 0.1), mask_at(m[2], 0.1, 0.1), mask_at(m[0], 0.1, 0.1)), (0.5, 0.5, 0.0));
    // Bottom-right is inside region 2 only (at 0.5) and nothing else: region 2 keeps 0.5, background the rest.
    assert_eq!((mask_at(m[1], 0.9, 0.9), mask_at(m[2], 0.9, 0.9), mask_at(m[0], 0.9, 0.9)), (0.0, 0.5, 0.5));
    // The global prompt covers the whole image.
    assert!(m[3].data.iter().all(|v| *v == 1.0));
    // Regions turn on Differential Diffusion.
    assert!(seen.model_patches.iter().any(|p| p.starts_with("DifferentialDiffusion")), "{:?}", seen.model_patches);
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_gligen_boxes() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Captured from SwarmUI: GLIGEN Model set, `a park <region:0,0,0.5,0.5> a red balloon` at 1024x1024.
    let r = run_like_swarm(&s, fixture("krea2_gligen.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    assert_eq!(seen.positive.len(), 1);
    let boxes = &seen.positive[0].gligen;
    assert_eq!(boxes.len(), 1);
    let b = &boxes[0];
    assert_eq!((b.x, b.y, b.width, b.height), (0, 0, 512, 512));
    assert_eq!(b.model.kind, AddonKind::Gligen);
    assert_eq!(b.model.source.name, "gligen_test.safetensors");
    assert_eq!(b.text.entries.len(), 1);
}

/// Every implemented node's `/object_info` entry matches stock ComfyUI's (input names, order,
/// types, combo options, numeric specs, outputs, category). `tests/fixtures/stock_schemas.json`
/// is regenerated by `tools/snapshot_stock_schemas.py`.
#[tokio::test(flavor = "multi_thread")]
async fn stock_schema_parity() {
    fn norm(spec: &Value) -> Value {
        let ty = &spec[0];
        let o = spec.get(1).cloned().unwrap_or(json!({}));
        let mut keep = serde_json::Map::new();
        for k in ["default", "min", "max", "step"] {
            if let Some(v) = o.get(k) {
                // 100 and 100.0 are the same value to ComfyUI and SwarmUI.
                keep.insert(k.into(), v.as_f64().map(|f| json!(f)).unwrap_or_else(|| v.clone()));
            }
        }
        let ty = if ty == "COMBO" { o["options"].clone() } else { ty.clone() };
        if let Some(t) = o.get("template").filter(|t| t.get("prefix").is_none()) {
            // MatchType: `{template_id, allowed_types}`.
            keep.insert("template".into(), t.clone());
        } else if let Some(t) = o.get("template") {
            keep.insert("template".into(), json!([t["prefix"], t["min"], t["max"], t["input"]["required"].as_object().unwrap().iter().map(|(k, v)| json!([k, norm(v)])).collect::<Vec<_>>()]));
        }
        if spec[0] == "COMFY_DYNAMICCOMBO_V3" {
            let opts: Vec<Value> = o["options"].as_array().unwrap().iter().map(|x| json!([x["key"], x["inputs"]["required"].as_object().map(|m| m.iter().map(|(k, v)| json!([k, norm(v)])).collect::<Vec<_>>())])).collect();
            keep.insert("options".into(), json!(opts));
        }
        // Model-file entries depend on the folders, not the schema; compare the fixed entries
        // (e.g. VAELoader's "pixel_space") of lists that hold files.
        let ty = match ty.as_array() {
            Some(a) => json!(a.iter().filter(|x| !x.as_str().is_some_and(|s| s.contains('.'))).collect::<Vec<_>>()),
            None => ty,
        };
        json!([ty, keep])
    }
    // Documented, intentional deviations.
    let allowed = [("VAEDecodeTiled", "temporal_size"), ("ModelAttentionBackend", "attention")];
    let stock: Value = serde_json::from_str(&std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stock_schemas.json")).unwrap()).unwrap();
    let ours = TestServer::start(0).await.get("object_info").await;
    let mut problems = Vec::new();
    for (name, s) in stock.as_object().unwrap() {
        let Some(o) = ours.get(name) else { continue };
        for sec in ["required", "optional"] {
            let (si, oi) = (&s["input"][sec], &o["input"][sec]);
            let keys = |v: &Value| v.as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
            if keys(si) != keys(oi) {
                problems.push(format!("{name}.{sec}: stock {:?} vs ours {:?}", keys(si), keys(oi)));
                continue;
            }
            for k in keys(si) {
                if allowed.contains(&(name.as_str(), k.as_str())) {
                    continue;
                }
                if norm(&si[&k]) != norm(&oi[&k]) {
                    problems.push(format!("{name}.{k}: stock {} vs ours {}", norm(&si[&k]), norm(&oi[&k])));
                }
            }
        }
        for f in ["output", "output_name", "category", "output_node"] {
            if s[f] != o[f] {
                problems.push(format!("{name}.{f}: stock {} vs ours {}", s[f], o[f]));
            }
        }
    }
    assert!(problems.is_empty(), "schema drift from stock ComfyUI:\n{}", problems.join("\n"));
}

// Inpainting fixtures were captured from SwarmUI with an init image and a Mask Image; the test
// suite substitutes its 300x200 pattern image for both, so the mask is that image's red channel.

#[tokio::test(flavor = "multi_thread")]
async fn qwen21_inpaint_with_blur_grow_and_color_correction() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Mask Blur 8, Mask Grow 16, Color Correction Behavior Linear2.
    let r = run_like_swarm(&s, fixture("qwen21_inpaint_blur_grow.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    // Recomposited into the init image.
    assert_eq!(png_dims(&r.outputs[0].bytes), (300, 200));
    let seen = engine.seen.lock().unwrap();
    assert!(seen.sample_masks[0].0.is_some(), "the sampled latent carries the noise mask");
    assert!(seen.model_patches.iter().any(|p| p.starts_with("DifferentialDiffusion")));
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_inpaint_crops_to_the_mask_and_recomposites() {
    let s = TestServer::start(0).await;
    // Mask Shrink Grow 16: threshold -> bounds -> crop image and mask -> scale to the model's
    // megapixels -> sample -> scale back -> composite with Uniform color correction.
    let r = run_like_swarm(&s, fixture("krea2_inpaint_crop.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(png_dims(&r.outputs[0].bytes), (300, 200));
}

#[tokio::test(flavor = "multi_thread")]
async fn moss_init_noise_and_masked_reset_to_norm() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Init Image Noise 0.3, Init Image Reset To Norm 0.4 with a mask, Mask Behavior Simple Latent.
    let r = run_like_swarm(&s, fixture("moss_inpaint_noise_reset.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    assert_eq!(seen.latent_ops, ["BlendMasked"]);
    assert!(!seen.model_patches.iter().any(|p| p.starts_with("DifferentialDiffusion")), "Simple Latent mask behavior");
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_unsampler_runs_before_sampling() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Unsampler Prompt "a dog", Init Image Creativity 0.6 of 6 steps -> unsample stops 2 steps short.
    let r = run_like_swarm(&s, fixture("krea2_unsampler.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(engine.seen.lock().unwrap().latent_ops, ["Unsample(2..6)"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_inpainting_encode_runs() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Use Inpainting Encode: VAEEncodeForInpaint (grow_mask_by 6) instead of VAEEncode.
    let r = run_like_swarm(&s, fixture("krea2_inpainting_encode.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    // SwarmUI follows VAEEncodeForInpaint with SetLatentNoiseMask on the raw mask, which replaces
    // the grown, rounded one (as it does in ComfyUI).
    let mask = seen.sample_masks[0].0.as_ref().expect("noise mask");
    assert_eq!((mask.width, mask.height), (300, 200));
    assert!(mask.data.iter().any(|v| *v != 0.0 && *v != 1.0), "the raw (unrounded) mask");
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_segment_refines_the_detected_region() {
    let s = TestServer::start(0).await;
    // `<segment:face,0.6,0.5>` with Save Segment Mask, Segment Mask Blur 4 / Grow 8: CLIPSeg mask,
    // threshold, bounds, crop, rescale, inpaint, recomposite. The mock segments the middle third.
    let r = run_like_swarm(&s, fixture("krea2_segment_text.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    // The segment mask image, then the final image.
    assert_eq!(r.outputs.len(), 2);
    for o in &r.outputs {
        assert_eq!(png_dims(&o.bytes), (1024, 1024));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_refiner_with_model_upscale() {
    let s = TestServer::start(0).await;
    // Refiner Upscale 2 with Refiner Upscale Method = model-4x...: upscale x4 with the model, then
    // scale to the 2x target, re-encode and refine (with Refiner Do Tiling).
    let r = run_like_swarm(&s, fixture("krea2_refine_model_upscale.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    // (Captured at 1024², shrunk to 256² to keep the test fast.)
    assert_eq!(png_dims(&r.outputs.last().unwrap().bytes), (512, 512));
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_refiner_with_latent_upscale_and_hypertile() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let r = run_like_swarm(&s, fixture("krea2_refine_latent_upscale.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    assert_eq!(seen.latent_ops, ["UpscaleBy"]);
    assert!(seen.model_patches.iter().any(|p| p.starts_with("HyperTile")), "{:?}", seen.model_patches);
    assert_eq!(png_dims(&r.outputs.last().unwrap().bytes), (1536, 1536));
}

#[tokio::test(flavor = "multi_thread")]
async fn qwen21_remove_background_outputs_rgba() {
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("qwen21_remove_background.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let (lo, hi) = png_alpha_range(&r.outputs[0].bytes).expect("RGBA output");
    assert!(lo < 50 && hi > 200, "alpha {lo}..{hi}");
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_style_model_with_merge_strength_and_apply_start() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // Use Style Model + Multiply Strength 0.8, Merge Strength 0.7, Apply Start 0.2: the plain
    // prompt for 0..0.2, then the styled prompt averaged with the plain one.
    let r = run_like_swarm(&s, fixture("krea2_style_model.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    let ranges: Vec<(f64, f64)> = seen.positive.iter().map(|m| (m.start_percent, m.end_percent)).collect();
    assert_eq!(ranges, [(0.0, 0.2), (0.2, 1.0)]);
    assert!(seen.positive[0].embedding_ops.is_empty());
    match &seen.positive[1].embedding_ops[..] {
        [EmbeddingOp::Style { model, vision, strength, strength_type }, EmbeddingOp::Average { from, to_strength }] => {
            assert_eq!(model.source.name, "redux_test.safetensors");
            assert_eq!(vision.model.source.name, "sigclip_vision_patch14_384.safetensors");
            assert_eq!((*strength, *strength_type, *to_strength), (0.8, StyleStrength::Multiply, 0.7));
            assert!(from.meta.embedding_ops.is_empty());
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_reference_only_prepends_the_reference() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let r = run_like_swarm(&s, fixture("krea2_reference_only.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    assert!(seen.model_patches.iter().any(|p| p == "ReferenceOnly { batch: 2 }"), "{:?}", seen.model_patches);
    assert_eq!(seen.latent_ops, ["PrependReference", "FromBatch"]);
    // The reference frame is kept (mask 0), the generation is free (64x64 ones, as SwarmUI's node).
    let mask = seen.sample_masks[0].0.as_ref().expect("noise mask");
    assert_eq!((mask.count, mask.width, mask.height), (2, 64, 64));
    assert!(mask.data[..64 * 64].iter().all(|v| *v == 0.0) && mask.data[64 * 64..].iter().all(|v| *v == 1.0));
    // Only the generated image comes out.
    assert_eq!(r.outputs.len(), 1);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
}

/// `(controlnet file, union type, strength, start, end, has vae, has inpaint mask)` per hint.
fn hints(m: &CondMeta) -> Vec<(String, Option<String>, f64, f64, f64, bool, bool)> {
    m.controls
        .iter()
        .map(|h| (h.controlnet.source.name.clone(), h.controlnet.control_type.clone(), h.strength, h.start_percent, h.end_percent, h.vae.is_some(), h.inpaint_mask.is_some()))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_two_controlnets_chain_with_union_type() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    // ControlNet One (strength 0.8, 0.1..0.9) then ControlNet Two (union, type depth, 0.5).
    let r = run_like_swarm(&s, fixture("krea2_controlnet_two_union.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    let want = vec![
        ("controlnet_test.safetensors".to_string(), None, 0.8, 0.1, 0.9, false, false),
        ("fake_union_controlnet.safetensors".to_string(), Some("depth".to_string()), 0.5, 0.0, 1.0, false, false),
    ];
    assert_eq!(hints(&seen.positive[0]), want);
    assert_eq!(hints(&seen.negative[0]), want);
}

#[tokio::test(flavor = "multi_thread")]
async fn krea2_alimama_inpainting_controlnet_gets_the_mask() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let r = run_like_swarm(&s, fixture("krea2_controlnet_alimama.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    assert_eq!(hints(&seen.positive[0]), [("fake_alimama_inpaint.safetensors".to_string(), None, 0.9, 0.0, 1.0, true, true)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn diffsynth_and_lllite_controlnets_patch_the_model() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let r = run_like_swarm(&s, fixture("qwen21_controlnet_diffpatch.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    {
        let seen = engine.seen.lock().unwrap();
        let p = seen.model_patches.iter().find(|p| p.starts_with("DiffSynthControl")).expect("DiffSynth patch");
        assert!(p.contains("fake_qwen_diffpatch.safetensors") && p.contains("strength: 0.7") && p.contains("mask: None"), "{p}");
    }
    let r = run_like_swarm(&s, fixture("anima_controlnet_lllite.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    let p = seen.model_patches.iter().find(|p| p.starts_with("AnimaLLLite")).expect("LLLite patch");
    assert!(p.contains("fake_anima_lllite.safetensors") && p.contains("strength: 0.6") && p.contains("mask: Some") && p.contains("start_percent: 0.2"), "{p}");
}

#[tokio::test(flavor = "multi_thread")]
async fn resize_image_mask_node_modes() {
    let s = TestServer::start(0).await;
    // SwarmUI emits `scale to multiple` (8) after a ControlNet preprocessor; check a few modes on
    // a 1000x600 mask (MaskToImage so the result can be saved).
    for (mode, extra, want) in [
        ("scale to multiple", json!({ "resize_type.multiple": 64 }), (960, 576)),
        ("scale longer dimension", json!({ "resize_type.longer_size": 500 }), (500, 300)),
        ("scale width", json!({ "resize_type.width": 250 }), (250, 150)),
        ("scale total pixels", json!({ "resize_type.megapixels": 0.25 }), (661, 397)),
    ] {
        let mut inputs = json!({ "input": ["1", 0], "resize_type": mode, "scale_method": "lanczos" });
        inputs.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let wf = json!({
            "1": { "class_type": "SolidMask", "inputs": { "value": 1.0, "width": 1000, "height": 600 } },
            "2": { "class_type": "ResizeImageMaskNode", "inputs": inputs },
            "3": { "class_type": "MaskToImage", "inputs": { "mask": ["2", 0] } },
            "4": { "class_type": "SwarmSaveImageWS", "inputs": { "images": ["3", 0] } }
        });
        let r = run_like_swarm(&s, wf, |_| {}).await;
        assert_eq!(r.status(), "success", "{mode}: {:#}", r.history);
        assert_eq!(png_dims(&r.outputs[0].bytes), want, "{mode}");
    }
}

/// SwarmUI's `emitAsCustomAdvanced` (IP2P CFG 2), grafted onto the Krea 2 graph in place of
/// `SwarmKSampler` "10": RandomNoise + KSamplerSelect + a scheduler + SplitSigmas for the start
/// and end steps + DualCFGGuider -> SamplerCustomAdvanced.
fn custom_sampler_workflow(scheduler: Value, start: Option<u32>, end: Option<u32>) -> Value {
    let mut wf = fixture("krea2_t2i.json");
    let k = wf["10"]["inputs"].clone();
    wf["900"] = json!({ "class_type": "RandomNoise", "inputs": { "noise_seed": 7 } });
    wf["901"] = json!({ "class_type": "KSamplerSelect", "inputs": { "sampler_name": "dpmpp_2m" } });
    wf["902"] = scheduler;
    let mut sigmas = json!(["902", 0]);
    if let Some(s) = start {
        wf["903"] = json!({ "class_type": "SplitSigmas", "inputs": { "sigmas": sigmas, "step": s } });
        sigmas = json!(["903", 1]);
    }
    if let Some(e) = end {
        wf["904"] = json!({ "class_type": "SplitSigmas", "inputs": { "sigmas": sigmas, "step": e } });
        sigmas = json!(["904", 0]);
    }
    wf["905"] = json!({ "class_type": "DualCFGGuider", "inputs": {
        "model": k["model"], "cond1": k["positive"], "cond2": k["negative"], "negative": k["negative"],
        "cfg_conds": 4.0, "cfg_cond2_negative": 1.5, "style": "regular" } });
    wf["10"] = json!({ "class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["900", 0], "guider": ["905", 0], "sampler": ["901", 0], "sigmas": sigmas, "latent_image": k["latent_image"] } });
    wf
}

#[tokio::test(flavor = "multi_thread")]
async fn custom_sampler_graph_reaches_the_engine() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let basic = json!({ "class_type": "BasicScheduler", "inputs": { "model": ["4", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0 } });
    let r = run_like_swarm(&s, custom_sampler_workflow(basic, Some(1), Some(3)), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    // The mock's 4-step schedule is [1, .75, .5, .25, 0]. SplitSigmas(1) keeps [.75, .5, .25, 0];
    // SwarmUI then splits *that* at the absolute end step 3, which keeps all 4 (as in ComfyUI).
    assert_eq!(engine.seen.lock().unwrap().custom_samples.pop().unwrap(), ("dpmpp_2m".into(), 7, vec![0.75, 0.5, 0.25, 0.0], 4.0, 1.5, false));

    // Partial denoise keeps the tail of a longer schedule (ComfyUI: int(4 / 0.5) = 8 steps).
    let basic = json!({ "class_type": "BasicScheduler", "inputs": { "model": ["4", 0], "scheduler": "simple", "steps": 4, "denoise": 0.5 } });
    let r = run_like_swarm(&s, custom_sampler_workflow(basic, None, None), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(engine.seen.lock().unwrap().custom_samples.pop().unwrap().2, [0.5, 0.375, 0.25, 0.125, 0.0]);

    let karras = json!({ "class_type": "KarrasScheduler", "inputs": { "steps": 4, "sigma_max": 14.614642, "sigma_min": 0.0291675, "rho": 7.0 } });
    let r = run_like_swarm(&s, custom_sampler_workflow(karras, None, None), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let k = engine.seen.lock().unwrap().custom_samples.pop().unwrap().2;
    assert_eq!(k.len(), 5);
    assert!((k[1] - 3.1686).abs() < 1e-3 && k[4] == 0.0, "{k:?}");

    let turbo = json!({ "class_type": "SDTurboScheduler", "inputs": { "model": ["4", 0], "steps": 2, "denoise": 1.0 } });
    let r = run_like_swarm(&s, custom_sampler_workflow(turbo, None, None), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(engine.seen.lock().unwrap().custom_samples.pop().unwrap().2.len(), 3);
    assert_eq!(png_dims(&r.outputs[0].bytes), (1024, 1024));
}

#[tokio::test(flavor = "multi_thread")]
async fn save_audio_advanced_accepts_swarms_nested_format() {
    // SwarmUI's audio save without its custom nodes: `format` arrives as a nested dict.
    let s = TestServer::start(0).await;
    let r = run_like_swarm(&s, fixture("save_audio_advanced.json"), |_| {}).await;
    let has_ffmpeg = std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success());
    if !has_ffmpeg {
        eprintln!("ffmpeg not found; only checked that the nested format validates");
        assert!(r.history.to_string().contains("ffmpeg"), "{:#}", r.history);
        return;
    }
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let h = r.history.to_string();
    assert!(h.contains("SwarmUI_1A2B__00001_.flac") && h.contains(r#""type":"output""#), "{h}");
}

#[tokio::test(flavor = "multi_thread")]
async fn kandinsky_style_video_latent_frame_swap() {
    // SwarmUI's Kandinsky 5 image-to-video tail: ReplaceVideoLatentFrames(index 0) then
    // NormalizeVideoLatentStart(4, 5), grafted onto the H3 video latent (7 latent frames).
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let mut wf = fixture("h3_boomerang_trim.json");
    wf["900"] = json!({ "class_type": "ReplaceVideoLatentFrames", "inputs": { "destination": ["104", 0], "source": ["104", 0], "index": 0 } });
    wf["901"] = json!({ "class_type": "NormalizeVideoLatentStart", "inputs": { "latent": ["900", 0], "start_frame_count": 4, "reference_frame_count": 5 } });
    // A source that doesn't fit at index -3 is skipped (ComfyUI warns and passes the destination on).
    wf["902"] = json!({ "class_type": "ReplaceVideoLatentFrames", "inputs": { "destination": ["901", 0], "source": ["104", 0], "index": -3 } });
    wf["8"]["inputs"]["samples"] = json!(["902", 0]);
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(engine.seen.lock().unwrap().latent_ops, ["ReplaceFrames", "NormalizeStart"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn image_only_checkpoint_loader_outputs_model_vision_and_vae() {
    let s = TestServer::start(0).await;
    let mut wf = fixture("krea2_t2i.json");
    wf["950"] = json!({ "class_type": "ImageOnlyCheckpointLoader", "inputs": { "ckpt_name": "svd_test.safetensors" } });
    wf["8"]["inputs"]["vae"] = json!(["950", 2]);
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
}

#[tokio::test(flavor = "multi_thread")]
async fn other_architecture_nodes_go_through_custom_node() {
    // An engine that claims WanImageToVideo gets it routed through `custom_node` with the stock
    // schema (widget defaults filled in); the mock returns typed placeholders.
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let info = s.get("object_info").await;
    assert_eq!(info["WanImageToVideo"]["input"]["required"]["length"], json!(["INT", { "default": 81, "min": 1, "max": 16384, "step": 4 }]));
    let mut wf = fixture("krea2_t2i.json");
    wf["900"] = json!({ "class_type": "WanImageToVideo", "inputs": { "positive": ["6", 0], "negative": ["7", 0], "vae": ["101", 0], "width": 256, "height": 256, "length": 9, "batch_size": 1 } });
    wf["10"]["inputs"]["positive"] = json!(["900", 0]);
    wf["10"]["inputs"]["negative"] = json!(["900", 1]);
    wf["10"]["inputs"]["latent_image"] = json!(["900", 2]);
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    assert_eq!(engine.seen.lock().unwrap().custom_nodes, [("WanImageToVideo".to_string(), vec![])]);
    // A 9-frame 256x256 video latent decodes to 9 frames.
    assert_eq!(r.outputs.len(), 9);
    assert_eq!(png_dims(&r.outputs[0].bytes), (256, 256));
}

#[tokio::test(flavor = "multi_thread")]
async fn third_party_nodes_get_model_files_resolved() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    let info = s.get("object_info").await;
    // File lists come from this server's model roots.
    assert!(info["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0].as_array().unwrap().contains(&json!("krea2/krea2_fp8_scaled.safetensors")));
    // SwarmUI's GGUF loader + IP-Adapter (unified loader, advanced apply) chain on the Krea 2 graph.
    let mut wf = fixture("krea2_t2i.json");
    wf["4"] = json!({ "class_type": "UnetLoaderGGUF", "inputs": { "unet_name": "krea2/krea2_fp8_scaled.safetensors" } });
    wf["900"] = json!({ "class_type": "SwarmLoadImageB64", "inputs": { "image_base64": "$IMAGE_B64" } });
    wf["901"] = json!({ "class_type": "IPAdapterUnifiedLoader", "inputs": { "model": ["4", 0], "preset": "PLUS (high strength)" } });
    wf["902"] = json!({ "class_type": "IPAdapterAdvanced", "inputs": { "model": ["901", 0], "ipadapter": ["901", 1], "image": ["900", 0], "weight": 0.8,
        "start_at": 0.0, "end_at": 1.0, "weight_type": "linear", "combine_embeds": "concat", "embeds_scaling": "V only" } });
    wf["102"]["inputs"]["model"] = json!(["902", 0]);
    let r = run_like_swarm(&s, with_test_media(wf.to_string()), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let seen = engine.seen.lock().unwrap();
    let classes: Vec<&str> = seen.custom_nodes.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(classes, ["UnetLoaderGGUF", "IPAdapterUnifiedLoader", "IPAdapterAdvanced"]);
    assert_eq!(seen.custom_nodes[0].1, [("unet_name".to_string(), "DiffusionModels/krea2/krea2_fp8_scaled.safetensors".to_string())]);
}

#[tokio::test(flavor = "multi_thread")]
async fn backlog_nodes_fail_validation_with_a_todo_message() {
    let s = TestServer::start_native(false).await;
    // A workflow using two backlog nodes (stock ComfyUI classes nothing here implements yet).
    let mut wf = fixture("krea2_t2i.json");
    wf["900"] = json!({ "class_type": "WanImageToVideo", "inputs": {} });
    wf["901"] = json!({ "class_type": "EmptyMochiLatentVideo", "inputs": {} });
    let resp = s.post("prompt", json!({ "prompt": wf, "client_id": "x" })).await;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "missing_node_type");
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.starts_with("This backend doesn't implement 2 node type(s)"), "{msg}");
    for node in ["EmptyMochiLatentVideo", "WanImageToVideo"] {
        assert!(msg.contains(&format!("- {node} (Other model architectures)")), "{node} missing from: {msg}");
    }

    // Unknown nodes outside the backlog keep ComfyUI's wording.
    let mut wf = fixture("krea2_t2i.json");
    wf["8"]["class_type"] = json!("SomeCustomNode");
    let body: Value = s.post("prompt", json!({ "prompt": wf })).await.json().await.unwrap();
    assert_eq!(body["error"]["message"], "Node 'SomeCustomNode' not found. The custom node may not be installed.");
}

#[tokio::test(flavor = "multi_thread")]
async fn backlog_nodes_are_only_advertised_on_request() {
    let plain = TestServer::start_native(false).await.get("object_info").await;
    assert!(plain.get("WanImageToVideo").is_none());
    assert!(plain.get("EmptyMochiLatentVideo").is_none());

    let s = TestServer::start_native(true).await;
    let info = s.get("object_info").await;
    // Stock ComfyUI schemas.
    assert_eq!(info["WanImageToVideo"]["input"]["required"]["width"][0], "INT");
    assert_eq!(info["EmptyMochiLatentVideo"]["input"]["required"]["length"][0], "INT");
    // Hand-written third-party schemas too.
    assert_eq!(info["IPAdapterUnifiedLoader"]["input"]["required"]["preset"][0][1], "STANDARD (medium strength)");
    // Implemented nodes keep this server's schema (model lists from our model roots).
    assert_eq!(info["UNETLoader"], plain["UNETLoader"]);
    let plain_count = plain.as_object().unwrap().len();
    assert!(info.as_object().unwrap().len() > plain_count + 40);
    // Still never executable.
    let mut wf = fixture("krea2_t2i.json");
    wf["900"] = json!({ "class_type": "LatentUpscaleBy", "inputs": {} });
    wf["900"] = json!({ "class_type": "WanImageToVideo", "inputs": {} });
    let resp = s.post("prompt", json!({ "prompt": wf })).await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test(flavor = "multi_thread")]
async fn backlog_does_not_shadow_implemented_nodes() {
    let s = TestServer::start_native(false).await;
    let info = s.get("object_info").await;
    let todo = comfy_server::todo::TodoCatalog::get();
    let overlap: Vec<&String> = todo.nodes.keys().filter(|k| info.get(k.as_str()).is_some()).collect();
    assert!(overlap.is_empty(), "implemented nodes still listed in todo_nodes.json (remove them from tools/todo_catalog.py): {overlap:?}");
    for (k, n) in &todo.nodes {
        assert!(todo.areas.contains_key(&n.area), "{k} has unknown area {}", n.area);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn execution_error_reaches_history() {
    let s = TestServer::start(0).await;
    let wf = json!({
        "1": { "class_type": "SwarmLoadImageB64", "inputs": { "image_base64": "%%%not base64%%%" } },
        "9": { "class_type": "SwarmSaveImageWS", "inputs": { "images": ["1", 0] } }
    });
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "error");
    // SwarmUI scans status.messages for execution_error and shows exception_message.
    let msgs = r.history_entry()["status"]["messages"].as_array().unwrap();
    let err = msgs.iter().find(|m| m[0] == "execution_error").expect("execution_error recorded");
    assert_eq!(err[1]["node_id"], "1");
    assert!(err[1]["exception_message"].as_str().unwrap().contains("base64"));
}

#[tokio::test(flavor = "multi_thread")]
async fn interrupt_stops_sampling() {
    let s = TestServer::start(20).await;
    let mut wf = fixture("krea2_t2i.json");
    wf["10"]["inputs"]["steps"] = json!(500);
    let http = s.http.clone();
    let interrupt_url = s.url("interrupt");
    let mut sent = false;
    let r = run_like_swarm(&s, wf, |pid| {
        if !sent {
            sent = true;
            let (http, url, pid) = (http.clone(), interrupt_url.clone(), pid.to_string());
            tokio::spawn(async move { http.post(url).body(json!({ "prompt_id": pid }).to_string()).send().await.unwrap() });
        }
    })
    .await;
    assert!(r.message_types.contains(&"execution_interrupted".to_string()), "{:?}", r.message_types);
    assert!(r.outputs.is_empty());
    assert_eq!(r.status(), "error");
}

#[tokio::test(flavor = "multi_thread")]
async fn upload_load_save_view_roundtrip() {
    let s = TestServer::start(0).await;
    let img = ImageBatch::from_u8(1, 40, 60, 3, vec![200; 40 * 60 * 3]);
    let part = reqwest::multipart::Part::bytes(encode::png8(&img, 0).unwrap()).file_name("init_image_sui_backend_0_1.png");
    let form = reqwest::multipart::Form::new().part("image", part).text("overwrite", "true");
    let up: Value = s.http.post(s.url("upload/image")).multipart(form).send().await.unwrap().json().await.unwrap();
    assert_eq!(up["name"], "init_image_sui_backend_0_1.png");
    assert_eq!(up["type"], "input");

    let wf = json!({
        "1": { "class_type": "LoadImage", "inputs": { "image": "init_image_sui_backend_0_1.png" } },
        "2": { "class_type": "SaveImage", "inputs": { "images": ["1", 0], "filename_prefix": "tests/SwarmUI_0001_" } }
    });
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    // SwarmUI's fallback path: read history outputs, then GET /view for each file.
    let file = &r.history_entry()["outputs"]["2"]["images"][0];
    assert_eq!(file["type"], "output");
    let url = format!(
        "view?filename={}&type=output&subfolder={}",
        file["filename"].as_str().unwrap(),
        file["subfolder"].as_str().unwrap()
    );
    let bytes = s.http.get(s.url(&url)).send().await.unwrap().bytes().await.unwrap();
    assert_eq!(png_dims(&bytes), (60, 40));
    let traversal = s.http.get(s.url("view?filename=../../Cargo.toml&type=output")).send().await.unwrap();
    assert_ne!(traversal.status(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_panic_fails_the_prompt_and_the_worker_keeps_going() {
    let engine = RecordingEngine::new();
    let s = TestServer::start_engine(engine.clone(), 0, false).await;
    engine.panic_next_encode.store(true, std::sync::atomic::Ordering::SeqCst);
    let r = run_like_swarm(&s, fixture("krea2_t2i.json"), |_| {}).await;
    assert_eq!(r.status(), "error");
    assert!(r.history.to_string().contains("internal error in SwarmTextEncodeAdvanced: engine bug"), "{:#}", r.history);
    let r = run_like_swarm(&s, fixture("krea2_t2i.json"), |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
}

#[tokio::test(flavor = "multi_thread")]
async fn mask_threshold_with_min_above_max_matches_torch() {
    // Segment Threshold Max below the minimum: torch clamps everything to max, so the mask is all 1s.
    let s = TestServer::start(0).await;
    let wf = json!({
        "1": { "class_type": "SolidMask", "inputs": { "value": 0.5, "width": 8, "height": 8 } },
        "2": { "class_type": "SwarmMaskThreshold", "inputs": { "mask": ["1", 0], "min": 0.01, "max": 0.0 } },
        "3": { "class_type": "MaskToImage", "inputs": { "mask": ["2", 0] } },
        "4": { "class_type": "SwarmSaveImageWS", "inputs": { "images": ["3", 0] } },
    });
    let r = run_like_swarm(&s, wf, |_| {}).await;
    assert_eq!(r.status(), "success", "{:#}", r.history);
    let img = image::load_from_memory(&r.outputs[0].bytes).unwrap().to_luma8();
    assert!(img.pixels().all(|p| p[0] == 255));
}
