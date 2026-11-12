//! Runs queued prompts on the worker thread, emitting the same websocket event
//! sequence ComfyUI does:
//!
//! `execution_start` → `execution_cached` → (`executing` → `progress`* → binary* → `executed`?)*
//! → `execution_success` | `execution_error` | `execution_interrupted` → `executing {node: null}`
//!
//! SwarmUI keys off `executing` with a null node to stop listening and fetch `/history`,
//! where it looks for `execution_error` in `status.messages`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value as Json};
use swarm_engine_api::Engine;

use crate::env::ServerEnv;
use crate::graph::{self, classify, InputRef, PromptGraph};
use crate::nodes::{EventSink, ExecContext, NodeError, NodeInputs, NodeRegistry};
use crate::queue::{FreeRequest, HistoryEntry, PromptQueue, QueueItem, Work};
use crate::schema::{InputKind, NodeSchema};
use crate::value::Value;
use crate::ws::Hub;

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

/// How long the engine waits for work before it gives VRAM back; IDLE_RELEASE_SECS, 0 to never.
fn idle_after() -> Option<Duration> {
    let secs = std::env::var("IDLE_RELEASE_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    (secs > 0).then_some(Duration::from_secs(secs))
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub struct Executor {
    pub registry: Arc<NodeRegistry>,
    pub engine: Arc<dyn Engine>,
    pub env: Arc<ServerEnv>,
    pub queue: Arc<PromptQueue>,
    pub hub: Arc<Hub>,
    /// Loader outputs reused across prompts, keyed by class + literal inputs.
    cache: Mutex<HashMap<String, Vec<Value>>>,
}

struct PromptEvents<'a> {
    hub: &'a Hub,
    queue: &'a PromptQueue,
    sid: Option<String>,
    prompt_id: String,
}

impl EventSink for PromptEvents<'_> {
    fn progress(&mut self, node: &str, value: u32, max: u32) {
        self.hub.send_json("progress", json!({ "value": value, "max": max, "prompt_id": self.prompt_id, "node": node }), self.sid.as_deref());
    }

    fn binary(&mut self, event: u32, payload: Vec<u8>) {
        self.hub.send_binary(event, &payload, self.sid.as_deref());
    }

    fn interrupted(&self) -> bool {
        self.queue.is_interrupted()
    }
}

impl Executor {
    pub fn new(registry: Arc<NodeRegistry>, engine: Arc<dyn Engine>, env: Arc<ServerEnv>, queue: Arc<PromptQueue>, hub: Arc<Hub>) -> Self {
        Self { registry, engine, env, queue, hub, cache: Mutex::new(HashMap::new()) }
    }

    /// The worker loop. Runs forever on a dedicated thread.
    pub fn run_forever(&self) {
        let mut wait = None;
        loop {
            match self.queue.next_work(wait) {
                None => {
                    self.engine.idle();
                    wait = None;
                }
                Some(Work::Free(FreeRequest { unload_models, free_memory })) => {
                    if unload_models {
                        self.cache.lock().unwrap().clear();
                    }
                    self.engine.free_memory(unload_models, free_memory);
                }
                Some(Work::Prompt(item)) => {
                    let entry = self.run_prompt(&item);
                    // Like ComfyUI's `task_done` then `executing None`: history must exist before
                    // the client sees the end marker, since SwarmUI fetches `/history` right away.
                    let sid = item.client_id.clone();
                    let pid = item.prompt_id.clone();
                    self.queue.finish(entry);
                    self.hub.send_json("executing", json!({ "node": null, "prompt_id": pid }), sid.as_deref());
                    self.broadcast_status();
                    wait = idle_after();
                }
            }
        }
    }

    pub fn broadcast_status(&self) {
        self.hub.send_json("status", json!({ "status": { "exec_info": { "queue_remaining": self.queue.remaining() } } }), None);
    }

    fn cache_key(node: &graph::GraphNode) -> Option<String> {
        if node.inputs.values().any(|v| matches!(classify(v), Some(InputRef::Link { .. }))) {
            return None;
        }
        Some(format!("{}|{}", node.class_type, Json::Object(node.inputs.clone())))
    }

    fn resolve_inputs(
        &self,
        id: &str,
        node: &graph::GraphNode,
        schema: &NodeSchema,
        results: &HashMap<String, Vec<Value>>,
    ) -> Result<NodeInputs, NodeError> {
        let mut values = HashMap::new();
        for (name, raw) in &node.inputs {
            let Some(spec) = schema.resolve_input(name) else { continue };
            match classify(raw) {
                None => {}
                Some(InputRef::Literal(v)) => {
                    let val = graph::convert_literal(&spec, v).map_err(|e| NodeError::Input(e["details"].as_str().unwrap_or("bad input").to_string()))?;
                    values.insert(name.clone(), val);
                }
                Some(InputRef::Link { node: src, slot }) => {
                    let out = results.get(&src).and_then(|o| o.get(slot)).ok_or_else(|| NodeError::Input(format!("input '{name}' links to {src}[{slot}] which produced no value")))?;
                    values.insert(name.clone(), out.clone());
                }
            }
        }
        for spec in &schema.inputs {
            if !values.contains_key(&spec.name) {
                if let Some(d) = spec.default_json() {
                    if let Ok(v) = graph::convert_literal(spec, &d) {
                        values.insert(spec.name.clone(), v);
                    }
                }
            }
            // Defaults for the chosen DynamicCombo option's sub-inputs (`"name.sub"`).
            if let (InputKind::DynamicCombo { options }, Some(Value::Str(chosen))) = (&spec.kind, values.get(&spec.name).cloned()) {
                for sub in options.iter().filter(|(k, _)| *k == chosen).flat_map(|(_, inputs)| inputs) {
                    let key = format!("{}.{}", spec.name, sub.name);
                    if let (false, Some(d)) = (values.contains_key(&key), sub.default_json()) {
                        if let Ok(v) = graph::convert_literal(sub, &d) {
                            values.insert(key, v);
                        }
                    }
                }
            }
        }
        Ok(NodeInputs { node_id: id.to_string(), class_type: node.class_type.clone(), values })
    }

    /// Executes one prompt and returns its history entry. The caller stores it and then
    /// sends the closing `executing {node: null}`.
    pub fn run_prompt(&self, item: &QueueItem) -> HistoryEntry {
        let sid = item.client_id.clone();
        let pid = item.prompt_id.clone();
        let mut messages: Vec<(String, Json)> = Vec::new();
        let mut outputs_ui: Map<String, Json> = Map::new();
        let mut meta: Map<String, Json> = Map::new();
        let send = |messages: &mut Vec<(String, Json)>, event: &str, data: Json, record: bool| {
            self.hub.send_json(event, data.clone(), sid.as_deref());
            if record {
                messages.push((event.to_string(), data));
            }
        };

        send(&mut messages, "execution_start", json!({ "prompt_id": pid, "timestamp": now_ms() }), true);

        let finish = |status: &str, completed: bool, messages: Vec<(String, Json)>, outputs: Map<String, Json>, meta: Map<String, Json>| {
            HistoryEntry { item: item.clone(), outputs, status_str: status.to_string(), completed, messages, meta }
        };

        let graph: PromptGraph = match graph::parse(&item.prompt) {
            Ok(g) => g,
            Err(e) => {
                let data = json!({ "prompt_id": pid, "node_id": "", "node_type": "", "executed": [], "exception_message": e.error["message"], "exception_type": "ValueError", "traceback": [], "current_inputs": {}, "current_outputs": {}, "timestamp": now_ms() });
                send(&mut messages, "execution_error", data, true);
                return finish("error", false, messages, outputs_ui, meta);
            }
        };
        let schemas = self.registry.schemas(&self.env);
        let order = graph::execution_order(&graph, &item.outputs);

        // Loader nodes with unchanged literal inputs are reused, and reported as cached.
        let mut results: HashMap<String, Vec<Value>> = HashMap::new();
        let mut cached_ids = Vec::new();
        {
            let cache = self.cache.lock().unwrap();
            for id in &order {
                let node = &graph.nodes[id];
                let cacheable = self.registry.get(&node.class_type).is_some_and(|d| d.cacheable);
                if let (true, Some(key)) = (cacheable, Self::cache_key(node)) {
                    if let Some(v) = cache.get(&key) {
                        results.insert(id.clone(), v.clone());
                        cached_ids.push(id.clone());
                    }
                }
            }
        }
        send(&mut messages, "execution_cached", json!({ "nodes": cached_ids, "prompt_id": pid, "timestamp": now_ms() }), true);

        let cached: HashSet<String> = cached_ids.into_iter().collect();
        let mut executed: Vec<String> = cached.iter().cloned().collect();
        let mut events = PromptEvents { hub: &self.hub, queue: &self.queue, sid: sid.clone(), prompt_id: pid.clone() };

        for id in &order {
            if cached.contains(id) {
                continue;
            }
            let node = &graph.nodes[id];
            let def = self.registry.get(&node.class_type).expect("validated");
            let schema = &schemas[&node.class_type];

            if self.queue.is_interrupted() {
                let data = json!({ "prompt_id": pid, "node_id": id, "node_type": node.class_type, "executed": executed, "timestamp": now_ms() });
                send(&mut messages, "execution_interrupted", data, true);
                return finish("error", false, messages, outputs_ui, meta);
            }
            self.hub.send_json("executing", json!({ "node": id, "display_node": id, "prompt_id": pid }), sid.as_deref());
            meta.insert(id.clone(), json!({ "node_id": id, "display_node": id, "parent_node": null, "real_node_id": id }));

            let result = self.resolve_inputs(id, node, schema, &results).and_then(|inputs| {
                let mut ctx = ExecContext { engine: self.engine.as_ref(), env: &self.env, prompt_id: &pid, node_id: id.clone(), events: &mut events };
                tracing::debug!("executing {id} ({})", node.class_type);
                // A panicking node (or engine) fails this prompt like any other error; letting it
                // unwind would kill the worker thread and leave every later prompt queued forever.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (def.exec)(&inputs, &mut ctx)))
                    .unwrap_or_else(|p| Err(NodeError::Other(anyhow::anyhow!("internal error in {}: {}", node.class_type, panic_message(&p)))))
            });

            match result {
                Ok(out) => {
                    if def.cacheable {
                        if let Some(key) = Self::cache_key(node) {
                            self.cache.lock().unwrap().insert(key, out.values.clone());
                        }
                    }
                    if let Some(ui) = out.ui {
                        self.hub.send_json("executed", json!({ "node": id, "display_node": id, "output": ui, "prompt_id": pid }), sid.as_deref());
                        outputs_ui.insert(id.clone(), ui);
                    }
                    results.insert(id.clone(), out.values);
                    executed.push(id.clone());
                }
                Err(e) if e.is_interrupt() => {
                    tracing::info!("prompt {pid} interrupted at node {id}");
                    let data = json!({ "prompt_id": pid, "node_id": id, "node_type": node.class_type, "executed": executed, "timestamp": now_ms() });
                    send(&mut messages, "execution_interrupted", data, true);
                    return finish("error", false, messages, outputs_ui, meta);
                }
                Err(e) => {
                    tracing::warn!("prompt {pid} failed at node {id} ({}): {e:#}", node.class_type);
                    let data = json!({
                        "prompt_id": pid,
                        "node_id": id,
                        "node_type": node.class_type,
                        "executed": executed,
                        "exception_message": format!("{e:#}"),
                        "exception_type": e.exception_type(),
                        "traceback": [],
                        "current_inputs": {},
                        "current_outputs": {},
                        "timestamp": now_ms(),
                    });
                    send(&mut messages, "execution_error", data, true);
                    return finish("error", false, messages, outputs_ui, meta);
                }
            }
        }
        send(&mut messages, "execution_success", json!({ "prompt_id": pid, "timestamp": now_ms() }), true);
        finish("success", true, messages, outputs_ui, meta)
    }
}
