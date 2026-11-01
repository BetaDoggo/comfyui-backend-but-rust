//! Websocket client registry and message framing.
//!
//! Framing matters: SwarmUI decides whether a frame is JSON by checking that its
//! first 8 bytes are exactly `{"type":`, so `type` must always be serialized
//! first. Binary frames are a 4-byte big-endian event id followed by the payload.

use std::collections::HashMap;
use std::sync::Mutex;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Map, Value as Json};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

/// Encodes `{"type": event, "data": data}` with `type` first.
pub fn encode_json(event: &str, data: &Json) -> String {
    let mut m = Map::new();
    m.insert("type".into(), json!(event));
    m.insert("data".into(), data.clone());
    Json::Object(m).to_string()
}

pub fn encode_binary(event: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 4);
    out.extend_from_slice(&event.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

struct Client {
    conn_id: u64,
    tx: UnboundedSender<Message>,
}

#[derive(Default)]
pub struct Hub {
    clients: Mutex<HashMap<String, Client>>,
    next_conn: Mutex<u64>,
}

impl Hub {
    fn send_to(&self, sid: Option<&str>, msg: Message) {
        let clients = self.clients.lock().unwrap();
        match sid {
            Some(sid) => {
                if let Some(c) = clients.get(sid) {
                    let _ = c.tx.send(msg);
                }
            }
            None => {
                for c in clients.values() {
                    let _ = c.tx.send(msg.clone());
                }
            }
        }
    }

    /// Sends to one client (`sid`) or broadcasts (`None`).
    pub fn send_json(&self, event: &str, data: Json, sid: Option<&str>) {
        self.send_to(sid, Message::Text(encode_json(event, &data).into()));
    }

    pub fn send_binary(&self, event: u32, payload: &[u8], sid: Option<&str>) {
        self.send_to(sid, Message::Binary(encode_binary(event, payload).into()));
    }

    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }

    /// Runs one websocket connection until it closes.
    pub async fn serve(&self, socket: WebSocket, sid: String, queue_remaining: usize) {
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = unbounded_channel::<Message>();
        let conn_id = {
            let mut n = self.next_conn.lock().unwrap();
            *n += 1;
            *n
        };
        // Reusing a clientId replaces the old connection, like ComfyUI.
        self.clients.lock().unwrap().insert(sid.clone(), Client { conn_id, tx: tx.clone() });
        let _ = tx.send(Message::Text(encode_json("status", &json!({ "status": { "exec_info": { "queue_remaining": queue_remaining } }, "sid": sid })).into()));

        let writer = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if sink.send(msg).await.is_err() {
                    break;
                }
            }
        });

        let mut first = true;
        while let Some(Ok(msg)) = stream.next().await {
            match msg {
                Message::Text(text) => {
                    if first {
                        if let Ok(v) = serde_json::from_str::<Json>(&text) {
                            if v["type"] == "feature_flags" {
                                tracing::debug!("client {sid} feature flags: {}", v["data"]);
                                let _ = tx.send(Message::Text(encode_json("feature_flags", &server_feature_flags()).into()));
                            }
                        }
                    }
                    first = false;
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
        {
            let mut clients = self.clients.lock().unwrap();
            if clients.get(&sid).is_some_and(|c| c.conn_id == conn_id) {
                clients.remove(&sid);
            }
        }
        writer.abort();
    }
}

/// What `GET /features` and the websocket `feature_flags` reply report.
pub fn server_feature_flags() -> Json {
    json!({
        "supports_preview_metadata": false,
        "max_upload_size": 100 * 1024 * 1024,
    })
}
