//! The generic route: backlog nodes (other model families' stock nodes and third-party packs)
//! that the engine claims via `Engine::supports_custom_node`. Their schemas come from
//! `todo_nodes.json` (stock ComfyUI dumps, or `tools/third_party_schemas.py`); execution hands the
//! resolved inputs to `Engine::custom_node`.

use std::collections::BTreeMap;

use serde_json::Value as Json;
use swarm_engine_api::{CustomNodeRequest, Engine, LoraHook, MiniMaxReferences, ModelFile, ModelFolder, NodeValue, Opaque};

use super::loaders::resolve;
use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::{InputKind, NodeSchema};
use crate::todo::TodoCatalog;
use crate::value::Value;

/// File-list combos and the folder each lists: the schema's `x_model_folders`, else ComfyUI's
/// usual input names for combos that are empty in the stock dump (it had no models).
fn file_inputs(info: &Json) -> BTreeMap<String, ModelFolder> {
    let mut out = BTreeMap::new();
    if let Some(m) = info["x_model_folders"].as_object() {
        for (k, v) in m {
            if let Some(f) = v.as_str().and_then(ModelFolder::from_comfy_key) {
                out.insert(k.clone(), f);
            }
        }
    }
    for sec in ["required", "optional"] {
        let Some(m) = info["input"].get(sec).and_then(Json::as_object) else { continue };
        for (k, spec) in m {
            let empty_list = spec[0].as_array().is_some_and(|a| a.is_empty());
            if out.contains_key(k) || !empty_list {
                continue;
            }
            let folder = match k.as_str() {
                "ckpt_name" => ModelFolder::Checkpoints,
                "unet_name" => ModelFolder::DiffusionModels,
                "vae_name" => ModelFolder::Vae,
                "lora_name" => ModelFolder::Loras,
                n if n.starts_with("clip_name") || n.starts_with("text_encoder") => ModelFolder::TextEncoders,
                _ => continue,
            };
            out.insert(k.clone(), folder);
        }
    }
    out
}

pub fn to_node_value(v: &Value) -> NodeValue {
    match v.clone() {
        Value::Model(x) => NodeValue::Model(x),
        Value::Clip(x) => NodeValue::Clip(x),
        Value::Vae(x) => NodeValue::Vae(x),
        Value::Conditioning(x) => NodeValue::Conditioning(x),
        Value::Latent(x) => NodeValue::Latent(x),
        Value::Image(x) => NodeValue::Image(x),
        Value::Mask(x) => NodeValue::Mask(x),
        Value::Audio(x) => NodeValue::Audio(x),
        Value::Video(x) => NodeValue::Video(x),
        Value::Addon(x) => NodeValue::Addon(x),
        Value::ClipVision(x) => NodeValue::ClipVisionOutput(x),
        Value::Hooks(x) => NodeValue::Other { ty: "HOOKS".into(), data: Opaque::new(x) },
        Value::MiniMaxRefs(x) => NodeValue::Other { ty: "MiniMaxReferences".into(), data: Opaque::new(x) },
        Value::Embedding(x) => NodeValue::Other { ty: "EMBEDDING".into(), data: Opaque::new(x) },
        Value::Other { ty, data } => NodeValue::Other { ty, data },
        Value::Int(x) => NodeValue::Int(x),
        Value::Float(x) => NodeValue::Float(x),
        Value::Str(x) => NodeValue::Str(x),
        Value::Bool(x) => NodeValue::Bool(x),
    }
}

pub fn from_node_value(v: NodeValue) -> Value {
    match v {
        NodeValue::Model(x) => Value::Model(x),
        NodeValue::Clip(x) => Value::Clip(x),
        NodeValue::Vae(x) => Value::Vae(x),
        NodeValue::Conditioning(x) => Value::Conditioning(x),
        NodeValue::Latent(x) => Value::Latent(x),
        NodeValue::Image(x) => Value::Image(x),
        NodeValue::Mask(x) => Value::Mask(x),
        NodeValue::Audio(x) => Value::Audio(x),
        NodeValue::Video(x) => Value::Video(x),
        NodeValue::Addon(x) => Value::Addon(x),
        NodeValue::ClipVisionOutput(x) => Value::ClipVision(x),
        NodeValue::File(f) => Value::Str(f.name),
        NodeValue::Other { ty, data } => match ty.as_str() {
            "HOOKS" if data.downcast_ref::<Vec<LoraHook>>().is_some() => Value::Hooks(data.downcast_ref::<Vec<LoraHook>>().unwrap().clone()),
            "MiniMaxReferences" if data.downcast_ref::<MiniMaxReferences>().is_some() => Value::MiniMaxRefs(data.downcast_ref::<MiniMaxReferences>().unwrap().clone()),
            "EMBEDDING" if data.downcast_ref::<ModelFile>().is_some() => Value::Embedding(data.downcast_ref::<ModelFile>().unwrap().clone()),
            _ => Value::Other { ty, data },
        },
        NodeValue::Int(x) => Value::Int(x),
        NodeValue::Float(x) => Value::Float(x),
        NodeValue::Str(x) => Value::Str(x),
        NodeValue::Bool(x) => Value::Bool(x),
    }
}

/// Registers every backlog node with a schema that the engine claims and the server doesn't
/// implement natively. Returns the registered classes.
pub fn register_supported(r: &mut NodeRegistry, env: &ServerEnv, engine: &dyn Engine) -> Vec<String> {
    let mut added = Vec::new();
    for (class, node) in &TodoCatalog::get().nodes {
        let Some(info) = &node.object_info else { continue };
        if r.get(class).is_some() || !engine.supports_custom_node(class) {
            continue;
        }
        let files = file_inputs(info);
        let (class_s, info_s, files_s) = (class.clone(), info.clone(), files.clone());
        r.add(
            env,
            move |env| {
                let mut s = NodeSchema::from_v1(&class_s, &info_s);
                for i in &mut s.inputs {
                    if let (Some(folder), InputKind::Combo { options, .. }) = (files_s.get(&i.name), &mut i.kind) {
                        *options = env.models.list(*folder);
                        i.raw_v1 = None;
                    }
                }
                s
            },
            {
                let class = class.clone();
                let outputs: Vec<String> = info["output"].as_array().map(|a| a.iter().map(|t| t.as_str().unwrap_or("*").to_string()).collect()).unwrap_or_default();
                move |i, ctx| {
                    let mut inputs = BTreeMap::new();
                    for (name, v) in &i.values {
                        let nv = match (files.get(name), v) {
                            (Some(folder), Value::Str(s)) => NodeValue::File(resolve(ctx.env, &[*folder], s)?),
                            _ => to_node_value(v),
                        };
                        inputs.insert(name.clone(), nv);
                    }
                    let req = CustomNodeRequest { class_type: class.clone(), inputs, output_types: outputs.clone() };
                    let engine = ctx.engine;
                    let out = engine.custom_node(&req, ctx)?;
                    if out.len() != outputs.len() {
                        return Err(input_err(format!("engine returned {} output(s) for {class}, expected {}", out.len(), outputs.len())));
                    }
                    Ok(NodeOutput::values(out.into_iter().map(from_node_value).collect()))
                }
            },
        );
        added.push(class.clone());
    }
    added
}
