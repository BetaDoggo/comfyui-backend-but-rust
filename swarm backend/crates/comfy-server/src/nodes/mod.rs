//! Node registry and the plumbing nodes use to read inputs and talk to the engine.
//!
//! Each ComfyUI node class is registered with a schema function (for
//! `/object_info` and validation) and an execute function that turns the
//! node's resolved inputs into either an engine call or pure bookkeeping.

mod conditioning;
mod controlnet;
mod custom_sampling;
mod generic;
mod guidance;
mod imageprompt;
mod inpaint;
mod loaders;
mod masks;
mod media;
mod output;
mod sampling;
mod seedvr;
mod segment;
mod snapref;
mod three_d;
mod video;
mod vae;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::{Map, Value as Json};
use swarm_engine_api::{
    AddonHandle, AddonKind, AudioData, ClipVisionOutput, Conditioning, Engine, EngineError, ImageBatch, Latent, LoraHook, Mask, ModelHandle, PreviewImage, ProgressSink, TextEncoderHandle,
    VaeHandle, VideoClip,
};

use crate::encode;
use crate::env::ServerEnv;
use crate::schema::NodeSchema;
use crate::todo::TodoCatalog;
use crate::value::Value;

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("{0}")]
    Input(String),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl NodeError {
    pub fn is_interrupt(&self) -> bool {
        matches!(self, NodeError::Engine(EngineError::Interrupted))
    }

    /// The Python-ish exception type name ComfyUI would report.
    pub fn exception_type(&self) -> &'static str {
        match self {
            NodeError::Input(_) => "ValueError",
            NodeError::Engine(EngineError::Interrupted) => "comfy.model_management.InterruptProcessingException",
            NodeError::Engine(EngineError::Unsupported(_)) => "NotImplementedError",
            NodeError::Engine(EngineError::InvalidInput(_)) => "ValueError",
            NodeError::Engine(EngineError::Internal(_)) | NodeError::Other(_) => "RuntimeError",
        }
    }
}

pub type NodeResult = Result<NodeOutput, NodeError>;

pub fn input_err(msg: impl Into<String>) -> NodeError {
    NodeError::Input(msg.into())
}

#[derive(Default)]
pub struct NodeOutput {
    pub values: Vec<Value>,
    /// The `ui` dict ComfyUI reports in the `executed` message and `/history` outputs
    /// (e.g. `{"images": [{"filename": ..., "subfolder": ..., "type": "output"}]}`).
    pub ui: Option<Json>,
    /// The items of `output_is_list` slots, by slot; `values` holds the first item there.
    pub lists: HashMap<usize, Vec<Value>>,
}

impl NodeOutput {
    pub fn values(values: Vec<Value>) -> Self {
        Self { values, ..Self::default() }
    }

    pub fn one(value: Value) -> Self {
        Self { values: vec![value], ..Self::default() }
    }

    /// Slot `slot` as a list of `items` (an `output_is_list` output).
    pub fn with_list(mut self, slot: usize, items: Vec<Value>) -> Self {
        if let Some(first) = items.first() {
            if self.values.len() <= slot {
                self.values.resize(slot + 1, first.clone());
            }
            self.values[slot] = first.clone();
        }
        self.lists.insert(slot, items);
        self
    }

    pub fn empty() -> Self {
        Self::default()
    }
}

type SchemaFn = Box<dyn Fn(&ServerEnv) -> NodeSchema + Send + Sync>;
type ExecFn = Box<dyn Fn(&NodeInputs, &mut ExecContext) -> NodeResult + Send + Sync>;

pub struct NodeDef {
    pub schema: SchemaFn,
    pub exec: ExecFn,
    /// Loader-style nodes whose outputs depend only on their literal inputs. The executor
    /// reuses their outputs across prompts (like ComfyUI's cache), reporting them as cached.
    pub cacheable: bool,
}

#[derive(Default)]
pub struct NodeRegistry {
    nodes: BTreeMap<String, Arc<NodeDef>>,
}

impl NodeRegistry {
    /// All nodes this server supports: the native ones, plus the backlog nodes `engine` claims
    /// (see `generic.rs`).
    pub fn standard(env: &ServerEnv, engine: &dyn Engine) -> Self {
        let mut r = Self::native(env);
        let added = generic::register_supported(&mut r, env, engine);
        if !added.is_empty() {
            tracing::info!("engine handles {} more node class(es) through custom_node", added.len());
        }
        r
    }

    /// Only the natively implemented nodes.
    pub fn native(env: &ServerEnv) -> Self {
        let mut r = Self::default();
        loaders::register(&mut r, env);
        masks::register(&mut r, env);
        conditioning::register(&mut r, env);
        guidance::register(&mut r, env);
        inpaint::register(&mut r, env);
        sampling::register(&mut r, env);
        vae::register(&mut r, env);
        media::register(&mut r, env);
        output::register(&mut r, env);
        video::register(&mut r, env);
        segment::register(&mut r, env);
        imageprompt::register(&mut r, env);
        controlnet::register(&mut r, env);
        custom_sampling::register(&mut r, env);
        three_d::register(&mut r, env);
        snapref::register(&mut r, env);
        seedvr::register(&mut r, env);
        r
    }

    fn insert(&mut self, env: &ServerEnv, schema: SchemaFn, exec: ExecFn, cacheable: bool) {
        let class = schema(env).class;
        self.nodes.insert(class, Arc::new(NodeDef { schema, exec, cacheable }));
    }

    pub fn add(
        &mut self,
        env: &ServerEnv,
        schema: impl Fn(&ServerEnv) -> NodeSchema + Send + Sync + 'static,
        exec: impl Fn(&NodeInputs, &mut ExecContext) -> NodeResult + Send + Sync + 'static,
    ) {
        self.insert(env, Box::new(schema), Box::new(exec), false);
    }

    pub fn add_cacheable(
        &mut self,
        env: &ServerEnv,
        schema: impl Fn(&ServerEnv) -> NodeSchema + Send + Sync + 'static,
        exec: impl Fn(&NodeInputs, &mut ExecContext) -> NodeResult + Send + Sync + 'static,
    ) {
        self.insert(env, Box::new(schema), Box::new(exec), true);
    }

    pub fn get(&self, class: &str) -> Option<Arc<NodeDef>> {
        self.nodes.get(class).cloned()
    }

    pub fn classes(&self) -> impl Iterator<Item = &String> {
        self.nodes.keys()
    }

    /// Schemas for every node, rendered fresh (model lists are rescanned each call).
    pub fn schemas(&self, env: &ServerEnv) -> HashMap<String, NodeSchema> {
        self.nodes.iter().map(|(k, d)| (k.clone(), (d.schema)(env))).collect()
    }

    pub fn object_info(&self, env: &ServerEnv) -> Json {
        let mut m = Map::new();
        for (k, d) in &self.nodes {
            m.insert(k.clone(), (d.schema)(env).to_v1());
        }
        if env.advertise_todo_nodes {
            for (k, n) in &TodoCatalog::get().nodes {
                if let (false, Some(info)) = (m.contains_key(k), &n.object_info) {
                    m.insert(k.clone(), info.clone());
                }
            }
        }
        Json::Object(m)
    }
}

/// A node's resolved inputs: literals converted to [`Value`]s per the schema, links
/// replaced by upstream outputs, and missing widgets filled with their defaults.
pub struct NodeInputs {
    pub node_id: String,
    pub class_type: String,
    pub values: HashMap<String, Value>,
    /// For an `INPUT_IS_LIST` node, every input's whole list; `values` holds the first items.
    pub lists: HashMap<String, Vec<Value>>,
}

macro_rules! typed_getters {
    ($req:ident, $opt:ident, $variant:ident, $ty:ty) => {
        pub fn $req(&self, name: &str) -> Result<&$ty, NodeError> {
            match self.values.get(name) {
                Some(Value::$variant(v)) => Ok(v),
                Some(other) => Err(input_err(format!("input '{name}' expected {} but got {}", stringify!($variant), other.type_name()))),
                None => Err(input_err(format!("required input '{name}' is missing"))),
            }
        }

        pub fn $opt(&self, name: &str) -> Result<Option<&$ty>, NodeError> {
            match self.values.get(name) {
                None => Ok(None),
                Some(_) => self.$req(name).map(Some),
            }
        }
    };
}

impl NodeInputs {
    /// An input's whole list, on an `INPUT_IS_LIST` node (one item elsewhere).
    pub fn list(&self, name: &str) -> Result<Vec<Value>, NodeError> {
        match (self.lists.get(name), self.values.get(name)) {
            (Some(l), _) => Ok(l.clone()),
            (None, Some(v)) => Ok(vec![v.clone()]),
            (None, None) => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    typed_getters!(model, opt_model, Model, ModelHandle);
    typed_getters!(clip, opt_clip, Clip, TextEncoderHandle);
    typed_getters!(vae, opt_vae, Vae, VaeHandle);
    typed_getters!(conditioning, opt_conditioning, Conditioning, Conditioning);
    typed_getters!(latent, opt_latent, Latent, Latent);
    typed_getters!(image, opt_image, Image, ImageBatch);
    typed_getters!(mask, opt_mask, Mask, Mask);
    typed_getters!(audio, opt_audio, Audio, AudioData);
    typed_getters!(video, opt_video, Video, VideoClip);
    typed_getters!(hooks, opt_hooks, Hooks, Vec<LoraHook>);
    typed_getters!(clip_vision_output, opt_clip_vision_output, ClipVision, ClipVisionOutput);

    /// An add-on model input of the given kind (GLIGEN, CONTROL_NET, ...).
    pub fn addon(&self, name: &str, kind: AddonKind) -> Result<&AddonHandle, NodeError> {
        match self.values.get(name) {
            Some(Value::Addon(a)) if a.kind == kind => Ok(a),
            Some(other) => Err(input_err(format!("input '{name}' should be {}, got {}", kind.comfy_type(), other.type_name()))),
            None => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    /// A [`Value::Other`] input of ComfyUI type `ty` carrying a `T` (`SIGMAS`, `NOISE`, ...).
    pub fn other<T: std::any::Any>(&self, name: &str, ty: &str) -> Result<&T, NodeError> {
        match self.values.get(name) {
            Some(Value::Other { ty: t, data }) if t == ty => data.downcast_ref::<T>().ok_or_else(|| input_err(format!("input '{name}' holds an unexpected {ty} payload"))),
            Some(other) => Err(input_err(format!("input '{name}' expected {ty} but got {}", other.type_name()))),
            None => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    pub fn int(&self, name: &str) -> Result<i64, NodeError> {
        match self.values.get(name) {
            Some(Value::Int(v)) => Ok(*v),
            Some(Value::Float(v)) => Ok(*v as i64),
            Some(other) => Err(input_err(format!("input '{name}' expected INT but got {}", other.type_name()))),
            None => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    pub fn uint(&self, name: &str) -> Result<u32, NodeError> {
        Ok(self.int(name)?.clamp(0, u32::MAX as i64) as u32)
    }

    pub fn float(&self, name: &str) -> Result<f64, NodeError> {
        match self.values.get(name) {
            Some(Value::Float(v)) => Ok(*v),
            Some(Value::Int(v)) => Ok(*v as f64),
            Some(other) => Err(input_err(format!("input '{name}' expected FLOAT but got {}", other.type_name()))),
            None => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    pub fn string(&self, name: &str) -> Result<&str, NodeError> {
        match self.values.get(name) {
            Some(Value::Str(v)) => Ok(v),
            Some(other) => Err(input_err(format!("input '{name}' expected STRING but got {}", other.type_name()))),
            None => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    pub fn opt_string(&self, name: &str) -> Result<Option<&str>, NodeError> {
        match self.values.get(name) {
            None => Ok(None),
            Some(_) => self.string(name).map(Some),
        }
    }

    pub fn boolean(&self, name: &str) -> Result<bool, NodeError> {
        match self.values.get(name) {
            Some(Value::Bool(v)) => Ok(*v),
            Some(Value::Str(s)) => Ok(matches!(s.as_str(), "true" | "True" | "enable")),
            Some(other) => Err(input_err(format!("input '{name}' expected BOOLEAN but got {}", other.type_name()))),
            None => Err(input_err(format!("required input '{name}' is missing"))),
        }
    }

    /// Members of an Autogrow group (`"{group}.{prefix}{i}"`), ordered by index.
    pub fn group(&self, group: &str) -> Vec<(u32, &Value)> {
        let prefix = format!("{group}.");
        let mut out: Vec<(u32, &Value)> = self
            .values
            .iter()
            .filter_map(|(k, v)| {
                let member = k.strip_prefix(&prefix)?;
                let digits: String = member.chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
                Some((digits.parse().ok()?, v))
            })
            .collect();
        out.sort_by_key(|(i, _)| *i);
        out
    }
}

/// What the executor provides to nodes for sending live data to the SwarmUI client.
pub trait EventSink: Send {
    /// `{"type": "progress", "data": {"value", "max", "prompt_id", "node"}}`.
    fn progress(&mut self, node: &str, value: u32, max: u32);
    /// A binary websocket frame: 4-byte big-endian event id, then `payload`.
    fn binary(&mut self, event: u32, payload: Vec<u8>);
    fn interrupted(&self) -> bool;
}

/// ComfyUI `BinaryEventTypes.PREVIEW_IMAGE`.
pub const EVENT_PREVIEW_IMAGE: u32 = 1;
/// Fake progress `max` values that tell SwarmUI the next binary frame is a final output.
pub const SWARM_OUTPUT_IMAGE: u32 = 12345;
pub const SWARM_OUTPUT_VIDEO: u32 = 12346;
pub const SWARM_OUTPUT_TEXT: u32 = 12347;

/// Per-node execution context.
pub struct ExecContext<'a> {
    pub engine: &'a dyn Engine,
    pub env: &'a ServerEnv,
    pub prompt_id: &'a str,
    pub node_id: String,
    pub events: &'a mut dyn EventSink,
}

impl ExecContext<'_> {
    /// Sends a final output to SwarmUI the way the `SwarmSave*WS` nodes do:
    /// a `progress` with `max = special_id`, then a `PREVIEW_IMAGE` binary frame whose
    /// payload is a 4-byte big-endian format code followed by the file bytes.
    ///
    /// Format codes SwarmUI understands: 1 jpg, 2 png, 3 webp, 4 gif, 5 mp4, 6 webm, 7 mov,
    /// 8 mp3, 9 wav, 10 flac, 11 ogg. For batches, `code | (index << 4)`.
    pub fn send_output(&mut self, special_id: u32, format_code: u32, bytes: &[u8]) {
        let node = self.node_id.clone();
        self.events.progress(&node, special_id, special_id);
        let mut payload = Vec::with_capacity(bytes.len() + 4);
        payload.extend_from_slice(&format_code.to_be_bytes());
        payload.extend_from_slice(bytes);
        self.events.binary(EVENT_PREVIEW_IMAGE, payload);
    }

    pub fn check_interrupt(&self) -> Result<(), NodeError> {
        if self.events.interrupted() {
            Err(EngineError::Interrupted.into())
        } else {
            Ok(())
        }
    }
}

impl ProgressSink for ExecContext<'_> {
    fn progress(&mut self, value: u32, max: u32) {
        let node = self.node_id.clone();
        self.events.progress(&node, value, max);
    }

    fn preview(&mut self, image: PreviewImage) {
        let (code, bytes) = match image {
            PreviewImage::Jpeg(b) => (1u32, b),
            PreviewImage::Png(b) => (2u32, b),
            PreviewImage::Rgb(batch) => match encode::jpeg(&batch, 0, 85) {
                Ok(b) => (1u32, b),
                Err(e) => {
                    tracing::warn!("failed to encode preview: {e:#}");
                    return;
                }
            },
        };
        let mut payload = Vec::with_capacity(bytes.len() + 4);
        payload.extend_from_slice(&code.to_be_bytes());
        payload.extend_from_slice(&bytes);
        self.events.binary(EVENT_PREVIEW_IMAGE, payload);
    }

    fn interrupted(&self) -> bool {
        self.events.interrupted()
    }
}
