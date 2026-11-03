//! HTTP routes, matching the ComfyUI server endpoints SwarmUI (and Comfy tooling) call.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Multipart, Path, Query, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json as AxJson, Router};
use serde_json::{json, Value as Json};
use swarm_engine_api::{Engine, ModelFolder};

use crate::env::ServerEnv;
use crate::graph;
use crate::nodes::NodeRegistry;
use crate::queue::{FreeRequest, PromptQueue, QueueItem};
use crate::ws::{server_feature_flags, Hub};

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<NodeRegistry>,
    pub env: Arc<ServerEnv>,
    pub queue: Arc<PromptQueue>,
    pub hub: Arc<Hub>,
    pub engine: Arc<dyn Engine>,
}

pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/ws", get(ws_handler))
        .route("/object_info", get(object_info))
        .route("/object_info/{class}", get(object_info_one))
        .route("/embeddings", get(embeddings))
        .route("/models", get(model_types))
        .route("/models/{folder}", get(models_in_folder))
        .route("/extensions", get(|| async { AxJson(json!([])) }))
        .route("/system_stats", get(system_stats))
        .route("/features", get(|| async { AxJson(server_feature_flags()) }))
        .route("/prompt", get(get_prompt).post(post_prompt))
        .route("/queue", get(get_queue).post(post_queue))
        .route("/interrupt", post(post_interrupt))
        .route("/free", post(post_free))
        .route("/history", get(get_history).post(post_history))
        .route("/history/{prompt_id}", get(get_history_one))
        .route("/view", get(view))
        .route("/upload/image", post(upload_image))
        .with_state(state);
    Router::new().route("/", get(index)).merge(api.clone()).nest("/api", api).layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024))
}

/// Parses a JSON body leniently (SwarmUI posts some bodies without a JSON content type).
fn body_json(body: &Bytes) -> Json {
    serde_json::from_slice(body).unwrap_or(Json::Null)
}

async fn index() -> Html<&'static str> {
    Html("<!doctype html><title>comfy-server</title><h1>comfy-server</h1><p>ComfyUI-compatible API for SwarmUI. There is no node editor UI here; use SwarmUI.</p>")
}

async fn ws_handler(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>, ws: WebSocketUpgrade) -> Response {
    let sid = q.get("clientId").filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let remaining = s.queue.remaining();
    ws.max_message_size(512 * 1024 * 1024).on_upgrade(move |socket| async move { s.hub.serve(socket, sid, remaining).await })
}

async fn object_info(State(s): State<AppState>) -> AxJson<Json> {
    let env = s.env.clone();
    let reg = s.registry.clone();
    AxJson(tokio::task::spawn_blocking(move || reg.object_info(&env)).await.unwrap_or(Json::Null))
}

async fn object_info_one(State(s): State<AppState>, Path(class): Path<String>) -> AxJson<Json> {
    let all = object_info(State(s)).await.0;
    AxJson(match all.get(&class) {
        Some(v) => json!({ class: v }),
        None => json!({}),
    })
}

async fn embeddings(State(s): State<AppState>) -> AxJson<Json> {
    let names: Vec<String> = s.env.models.list(ModelFolder::Embeddings).into_iter().map(|n| n.rsplit_once('.').map(|(a, _)| a.to_string()).unwrap_or(n)).collect();
    AxJson(json!(names))
}

async fn model_types() -> AxJson<Json> {
    AxJson(json!(ModelFolder::ALL.iter().map(|f| f.comfy_key()).collect::<Vec<_>>()))
}

async fn models_in_folder(State(s): State<AppState>, Path(folder): Path<String>) -> Response {
    match ModelFolder::from_comfy_key(&folder) {
        Some(f) => AxJson(json!(s.env.models.list(f))).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn system_stats(State(s): State<AppState>) -> AxJson<Json> {
    let info = s.engine.info();
    AxJson(json!({
        "system": {
            "os": std::env::consts::OS,
            "ram_total": info.ram_total,
            "ram_free": info.ram_free,
            "comfyui_version": concat!("comfy-server-", env!("CARGO_PKG_VERSION")),
            "python_version": "n/a (native rust)",
            "pytorch_version": format!("{} {}", info.name, info.version),
            "embedded_python": false,
            "argv": std::env::args().collect::<Vec<_>>(),
        },
        "devices": info.devices.iter().map(|d| json!({
            "name": d.name, "type": d.kind, "index": d.index,
            "vram_total": d.vram_total, "vram_free": d.vram_free,
            "torch_vram_total": d.vram_total, "torch_vram_free": d.vram_free,
        })).collect::<Vec<_>>(),
    }))
}

async fn get_prompt(State(s): State<AppState>) -> AxJson<Json> {
    AxJson(json!({ "exec_info": { "queue_remaining": s.queue.remaining() } }))
}

async fn post_prompt(State(s): State<AppState>, body: Bytes) -> Response {
    let body = body_json(&body);
    let Some(prompt) = body.get("prompt").cloned() else {
        let e = json!({ "error": { "type": "no_prompt", "message": "No prompt provided", "details": "No prompt provided", "extra_info": {} }, "node_errors": {} });
        return (StatusCode::BAD_REQUEST, AxJson(e)).into_response();
    };
    let prompt_id = match body.get("prompt_id").and_then(Json::as_str) {
        Some(id) => match uuid::Uuid::parse_str(id) {
            Ok(u) => u.hyphenated().to_string(),
            Err(_) => {
                let e = json!({ "error": { "type": "invalid_prompt_id", "message": "prompt_id must be a valid UUID", "details": "", "extra_info": {} }, "node_errors": {} });
                return (StatusCode::BAD_REQUEST, AxJson(e)).into_response();
            }
        },
        None => uuid::Uuid::new_v4().hyphenated().to_string(),
    };
    let reg = s.registry.clone();
    let env = s.env.clone();
    let p = prompt.clone();
    let validated = tokio::task::spawn_blocking(move || {
        let g = graph::parse(&p)?;
        graph::validate(&g, &reg.schemas(&env))
    })
    .await
    .expect("validation task panicked");
    let outputs = match validated {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!("invalid prompt: {}", e.error);
            return (StatusCode::BAD_REQUEST, AxJson(e.to_json())).into_response();
        }
    };
    let number = match body.get("number").and_then(Json::as_f64) {
        Some(n) => n as i64,
        None => s.queue.next_number(body.get("front").and_then(Json::as_bool).unwrap_or(false)),
    };
    let client_id = body.get("client_id").and_then(Json::as_str).map(str::to_string);
    let mut extra = body.get("extra_data").cloned().unwrap_or_else(|| json!({}));
    if let (Some(cid), Some(obj)) = (&client_id, extra.as_object_mut()) {
        obj.insert("client_id".into(), json!(cid));
    }
    s.queue.push(QueueItem { number, prompt_id: prompt_id.clone(), prompt, extra_data: extra, outputs, client_id });
    s.hub.send_json("status", json!({ "status": { "exec_info": { "queue_remaining": s.queue.remaining() } } }), None);
    AxJson(json!({ "prompt_id": prompt_id, "number": number, "node_errors": {} })).into_response()
}

async fn get_queue(State(s): State<AppState>) -> AxJson<Json> {
    AxJson(s.queue.queue_json())
}

fn id_list(v: &Json) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default()
}

async fn post_queue(State(s): State<AppState>, body: Bytes) -> StatusCode {
    let body = body_json(&body);
    if body.get("clear").and_then(Json::as_bool) == Some(true) {
        s.queue.clear_pending();
    }
    if let Some(d) = body.get("delete") {
        s.queue.delete_pending(&id_list(d));
    }
    StatusCode::OK
}

async fn post_interrupt(State(s): State<AppState>, body: Bytes) -> StatusCode {
    let body = body_json(&body);
    s.queue.interrupt(body.get("prompt_id").and_then(Json::as_str));
    StatusCode::OK
}

async fn post_free(State(s): State<AppState>, body: Bytes) -> StatusCode {
    let body = body_json(&body);
    s.queue.request_free(FreeRequest {
        unload_models: body.get("unload_models").and_then(Json::as_bool).unwrap_or(false),
        free_memory: body.get("free_memory").and_then(Json::as_bool).unwrap_or(false),
    });
    StatusCode::OK
}

async fn get_history(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> AxJson<Json> {
    let max = q.get("max_items").and_then(|v| v.parse().ok());
    let offset = q.get("offset").and_then(|v| v.parse::<i64>().ok()).filter(|o| *o >= 0).map(|o| o as usize);
    AxJson(s.queue.history_json(None, max, offset))
}

async fn get_history_one(State(s): State<AppState>, Path(id): Path<String>) -> AxJson<Json> {
    AxJson(s.queue.history_json(Some(&id), None, None))
}

async fn post_history(State(s): State<AppState>, body: Bytes) -> StatusCode {
    let body = body_json(&body);
    if body.get("clear").and_then(Json::as_bool) == Some(true) {
        s.queue.clear_history();
    }
    if let Some(d) = body.get("delete") {
        s.queue.delete_history(&id_list(d));
    }
    StatusCode::OK
}

fn mime_for(name: &str) -> &'static str {
    match name.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "flac" => "audio/flac",
        "ogg" | "opus" => "audio/ogg",
        "json" => "application/json",
        "glb" => "model/gltf-binary",
        _ => "application/octet-stream",
    }
}

async fn view(State(s): State<AppState>, Query(q): Query<HashMap<String, String>>) -> Response {
    let Some(filename) = q.get("filename") else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let ty = q.get("type").map(String::as_str).unwrap_or("output");
    let Some(dir) = s.env.dir_for_type(ty) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Some(path) = ServerEnv::safe_join(dir, q.get("subfolder").map(String::as_str).unwrap_or(""), filename) else {
        return StatusCode::FORBIDDEN.into_response();
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, mime_for(filename))], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn upload_image(State(s): State<AppState>, mut form: Multipart) -> Response {
    let mut file: Option<(String, Bytes)> = None;
    let mut overwrite = false;
    let mut subfolder = String::new();
    let mut ty = "input".to_string();
    while let Ok(Some(field)) = form.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" => {
                let fname = field.file_name().unwrap_or("upload.png").to_string();
                match field.bytes().await {
                    Ok(b) => file = Some((fname, b)),
                    Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                }
            }
            "overwrite" => overwrite = matches!(field.text().await.as_deref(), Ok("true") | Ok("1")),
            "subfolder" => subfolder = field.text().await.unwrap_or_default(),
            "type" => ty = field.text().await.unwrap_or_else(|_| "input".into()),
            _ => {}
        }
    }
    let Some((mut fname, bytes)) = file else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Some(dir) = s.env.dir_for_type(&ty) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let Some(mut path) = ServerEnv::safe_join(dir, &subfolder, &fname) else {
        return StatusCode::FORBIDDEN.into_response();
    };
    if !overwrite {
        // ComfyUI renames to "name (1).ext" on collision.
        let (stem, ext) = fname.rsplit_once('.').map(|(a, b)| (a.to_string(), format!(".{b}"))).unwrap_or((fname.clone(), String::new()));
        let mut i = 1;
        while path.exists() {
            fname = format!("{stem} ({i}){ext}");
            path = ServerEnv::safe_join(dir, &subfolder, &fname).unwrap();
            i += 1;
        }
    }
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    if tokio::fs::write(&path, &bytes).await.is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    AxJson(json!({ "name": fname, "subfolder": subfolder, "type": ty })).into_response()
}
