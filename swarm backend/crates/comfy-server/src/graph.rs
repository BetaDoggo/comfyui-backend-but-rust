//! Parsing and validating ComfyUI "API format" prompts:
//! `{ "<node id>": { "class_type": "...", "inputs": { "<name>": <literal> | ["<node id>", <slot>] } } }`.
//!
//! Validation mirrors `execution.validate_prompt` closely enough that SwarmUI's
//! error display (it pretty-prints the whole error JSON) reads the same.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{json, Map, Value as Json};

use crate::schema::{types_compatible, InputKind, InputSpec, NodeSchema};
use crate::todo::TodoCatalog;
use crate::value::Value;

#[derive(Clone, Debug)]
pub struct GraphNode {
    pub class_type: String,
    pub inputs: Map<String, Json>,
}

#[derive(Clone, Debug)]
pub struct PromptGraph {
    pub nodes: BTreeMap<String, GraphNode>,
}

pub enum InputRef<'a> {
    Link { node: String, slot: usize },
    Literal(&'a Json),
}

/// ComfyUI treats `[ "<id>", <int> ]` as a link; `null` as "not provided".
pub fn classify(v: &Json) -> Option<InputRef<'_>> {
    match v {
        Json::Null => None,
        Json::Array(a) if a.len() == 2 && (a[0].is_string() || a[0].is_number()) && a[1].is_u64() => {
            let node = match &a[0] {
                Json::String(s) => s.clone(),
                other => other.to_string(),
            };
            Some(InputRef::Link { node, slot: a[1].as_u64().unwrap() as usize })
        }
        other => Some(InputRef::Literal(other)),
    }
}

/// `{"error": {...}, "node_errors": {...}}`, returned with HTTP 400.
#[derive(Debug)]
pub struct PromptError {
    pub error: Json,
    pub node_errors: Json,
}

impl PromptError {
    fn simple(ty: &str, message: &str, details: &str) -> Self {
        Self { error: err_item(ty, message, details, json!({})), node_errors: json!({}) }
    }

    pub fn to_json(&self) -> Json {
        json!({ "error": self.error, "node_errors": self.node_errors })
    }
}

fn err_item(ty: &str, message: &str, details: &str, extra: Json) -> Json {
    json!({ "type": ty, "message": message, "details": details, "extra_info": extra })
}

pub fn parse(prompt: &Json) -> Result<PromptGraph, PromptError> {
    let obj = prompt.as_object().ok_or_else(|| PromptError::simple("invalid_prompt", "Invalid prompt", "prompt must be an object of nodes"))?;
    let mut nodes = BTreeMap::new();
    for (id, node) in obj {
        let class_type = node.get("class_type").and_then(Json::as_str);
        let Some(class_type) = class_type else {
            return Err(PromptError::simple(
                "invalid_prompt",
                &format!("Cannot execute because a node is missing the class_type property.: Node ID '#{id}'"),
                "",
            ));
        };
        let mut inputs = node.get("inputs").and_then(Json::as_object).cloned().unwrap_or_default();
        // V3 DynamicCombo values may come nested (`"format": {"format": "mp3", "quality": "V0"}`,
        // as SwarmUI sends SaveAudioAdvanced); flatten them to `"format"` / `"format.quality"`.
        let nested: Vec<(String, Map<String, Json>)> = inputs.iter().filter_map(|(k, v)| Some((k.clone(), v.as_object()?.clone()))).collect();
        for (name, obj) in nested {
            inputs.remove(&name);
            for (k, v) in obj {
                inputs.insert(if k == name { name.clone() } else { format!("{name}.{k}") }, v);
            }
        }
        nodes.insert(id.clone(), GraphNode { class_type: class_type.to_string(), inputs });
    }
    Ok(PromptGraph { nodes })
}

/// Converts a literal JSON value per the input spec (the same coercions ComfyUI applies).
pub fn convert_literal(spec: &InputSpec, v: &Json) -> Result<Value, Json> {
    let name = spec.name.as_str();
    let bad_type = |want: &str| {
        err_item("invalid_input_type", &format!("Failed to convert an input value to a {want} value"), &format!("{name}, {v}"), json!({ "input_name": name, "received_value": v }))
    };
    let as_f64 = |v: &Json| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()));
    match &spec.kind {
        InputKind::Int { min, max, .. } => {
            let n = v.as_i64().or_else(|| as_f64(v).map(|f| f as i64)).ok_or_else(|| bad_type("INT"))?;
            range_check(name, n as f64, *min as f64, *max as f64, v)?;
            Ok(Value::Int(n))
        }
        InputKind::Float { min, max, .. } => {
            let f = as_f64(v).ok_or_else(|| bad_type("FLOAT"))?;
            range_check(name, f, *min, *max, v)?;
            Ok(Value::Float(f))
        }
        InputKind::String { .. } => Ok(Value::Str(match v {
            Json::String(s) => s.clone(),
            other => other.to_string(),
        })),
        InputKind::Boolean { .. } => match v {
            Json::Bool(b) => Ok(Value::Bool(*b)),
            Json::String(s) if s.eq_ignore_ascii_case("true") => Ok(Value::Bool(true)),
            Json::String(s) if s.eq_ignore_ascii_case("false") => Ok(Value::Bool(false)),
            _ => Err(bad_type("BOOLEAN")),
        },
        InputKind::DynamicCombo { options } => {
            let keys: Vec<String> = options.iter().map(|(k, _)| k.clone()).collect();
            convert_literal(&InputSpec { kind: InputKind::Combo { options: keys, default: None }, ..spec.clone() }, v)
        }
        InputKind::Combo { options, .. } => {
            let s = match v {
                Json::String(s) => s.clone(),
                Json::Number(n) => n.to_string(),
                Json::Bool(b) => b.to_string(),
                _ => return Err(bad_type("COMBO")),
            };
            let norm = s.replace('\\', "/");
            match options.iter().find(|o| **o == s || o.replace('\\', "/") == norm) {
                Some(o) => Ok(Value::Str(o.clone())),
                None => {
                    let shown: Vec<&String> = options.iter().take(50).collect();
                    Err(err_item(
                        "value_not_in_list",
                        "Value not in list",
                        &format!("{name}: '{s}' not in {shown:?}{}", if options.len() > 50 { " ..." } else { "" }),
                        json!({ "input_name": name, "input_config": spec.to_v1(), "received_value": v }),
                    ))
                }
            }
        }
        InputKind::Link(t) | InputKind::Autogrow { item: _, prefix: t, .. } | InputKind::MatchType { allowed: t, .. } => Err(err_item(
            "bad_linked_input",
            "Bad linked input, must be a length-2 list of [node_id, slot_index]",
            &format!("{name} (expects a {t} link)"),
            json!({ "input_name": name, "received_value": v }),
        )),
    }
}

fn range_check(name: &str, val: f64, min: f64, max: f64, raw: &Json) -> Result<(), Json> {
    if val < min {
        return Err(err_item("value_smaller_than_min", &format!("Value {val} smaller than min of {min}"), &format!("{name}"), json!({ "input_name": name, "received_value": raw })));
    }
    if val > max {
        return Err(err_item("value_bigger_than_max", &format!("Value {val} bigger than max of {max}"), &format!("{name}"), json!({ "input_name": name, "received_value": raw })));
    }
    Ok(())
}

/// The ComfyUI type a widget input accepts when something is linked into it.
fn widget_link_type(spec: &InputSpec) -> &str {
    match &spec.kind {
        InputKind::Int { .. } => "INT",
        InputKind::Float { .. } => "FLOAT,INT",
        InputKind::String { .. } => "STRING",
        InputKind::Boolean { .. } => "BOOLEAN",
        InputKind::Combo { .. } | InputKind::DynamicCombo { .. } => "COMBO,STRING",
        InputKind::Link(t) => t,
        InputKind::Autogrow { .. } => "*",
        InputKind::MatchType { .. } => crate::schema::MATCH_TYPE,
    }
}

/// Validates the graph; returns the output node ids to execute.
pub fn validate(graph: &PromptGraph, schemas: &HashMap<String, NodeSchema>) -> Result<Vec<String>, PromptError> {
    let missing: Vec<(&String, &GraphNode)> = graph.nodes.iter().filter(|(_, n)| !schemas.contains_key(&n.class_type)).collect();
    if let Some(&(id, node)) = missing.first() {
        let message = TodoCatalog::get()
            .message(missing.iter().map(|(_, n)| n.class_type.as_str()))
            .unwrap_or_else(|| format!("Node '{}' not found. The custom node may not be installed.", node.class_type));
        let mut e = PromptError::simple("missing_node_type", &message, &format!("Node ID '#{id}'"));
        e.error["extra_info"] = json!({ "node_id": id, "class_type": node.class_type });
        return Err(e);
    }
    let outputs: Vec<String> = graph.nodes.iter().filter(|(_, n)| schemas[&n.class_type].output_node).map(|(id, _)| id.clone()).collect();
    if outputs.is_empty() {
        return Err(PromptError::simple("prompt_no_outputs", "Prompt has no outputs", ""));
    }

    let mut node_errors: Map<String, Json> = Map::new();
    let mut checked: HashMap<String, bool> = HashMap::new();
    let mut good_outputs = Vec::new();
    for out in &outputs {
        let mut visiting = HashSet::new();
        if validate_node(graph, schemas, out, &mut checked, &mut node_errors, &mut visiting) {
            good_outputs.push(out.clone());
        } else {
            // Record which outputs depend on each failing node, as ComfyUI does.
            for (_, entry) in node_errors.iter_mut() {
                if let Some(list) = entry.get_mut("dependent_outputs").and_then(Json::as_array_mut) {
                    if !list.iter().any(|v| v == out) {
                        list.push(json!(out));
                    }
                }
            }
        }
    }
    if good_outputs.len() != outputs.len() || !node_errors.is_empty() {
        let details: Vec<String> = node_errors
            .iter()
            .flat_map(|(id, e)| {
                e["errors"].as_array().cloned().unwrap_or_default().into_iter().map(move |x| format!("{}: {} ({})", id, x["message"].as_str().unwrap_or(""), x["details"].as_str().unwrap_or("")))
            })
            .collect();
        return Err(PromptError {
            error: err_item("prompt_outputs_failed_validation", "Prompt outputs failed validation", &details.join("\n"), json!({})),
            node_errors: Json::Object(node_errors),
        });
    }
    Ok(outputs)
}

fn validate_node(
    graph: &PromptGraph,
    schemas: &HashMap<String, NodeSchema>,
    id: &str,
    checked: &mut HashMap<String, bool>,
    node_errors: &mut Map<String, Json>,
    visiting: &mut HashSet<String>,
) -> bool {
    if let Some(ok) = checked.get(id) {
        return *ok;
    }
    if !visiting.insert(id.to_string()) {
        node_errors.insert(id.to_string(), json!({ "errors": [err_item("dependency_cycle_detected", "Dependency cycle detected", id, json!({}))], "dependent_outputs": [], "class_type": "" }));
        return false;
    }
    let node = &graph.nodes[id];
    let schema = &schemas[&node.class_type];
    let mut errors = Vec::new();
    let mut ok = true;

    for spec in &schema.inputs {
        // Autogrow groups arrive as `"{name}.{prefix}{i}"` members; required means at least `min` of them.
        if let InputKind::Autogrow { min, .. } = &spec.kind {
            let prefix = format!("{}.", spec.name);
            let count = node.inputs.iter().filter(|(k, v)| k.starts_with(&prefix) && classify(v).is_some()).count();
            if !spec.optional && count < (*min).max(1) as usize {
                errors.push(err_item("required_input_missing", "Required input is missing", &spec.name, json!({ "input_name": spec.name })));
            }
            continue;
        }
        let provided = node.inputs.get(&spec.name).and_then(classify);
        if provided.is_none() && !spec.optional && !spec.is_widget() {
            errors.push(err_item("required_input_missing", "Required input is missing", &spec.name, json!({ "input_name": spec.name })));
        }
    }
    for (name, raw) in &node.inputs {
        let Some(spec) = schema.resolve_input(name) else { continue }; // ComfyUI ignores unknown inputs
        match classify(raw) {
            None => {}
            Some(InputRef::Literal(v)) => {
                if let Err(e) = convert_literal(&spec, v) {
                    errors.push(e);
                }
            }
            Some(InputRef::Link { node: src, slot }) => {
                let Some(src_node) = graph.nodes.get(&src) else {
                    errors.push(err_item("bad_linked_input", "Linked node does not exist", &format!("{name}: node '{src}'"), json!({ "input_name": name })));
                    continue;
                };
                let src_schema = &schemas[&src_node.class_type];
                let Some(out) = src_schema.outputs.get(slot) else {
                    errors.push(err_item("bad_linked_input", "Linked output slot does not exist", &format!("{name}: {}[{slot}]", src_node.class_type), json!({ "input_name": name })));
                    continue;
                };
                let want = widget_link_type(&spec);
                if !types_compatible(&out.ty, want) {
                    errors.push(err_item(
                        "return_type_mismatch",
                        "Return type mismatch between linked nodes",
                        &format!("{name}, received_type({}) mismatch input_type({want})", out.ty),
                        json!({ "input_name": name, "received_type": out.ty, "linked_node": [src, slot] }),
                    ));
                    continue;
                }
                if !validate_node(graph, schemas, &src, checked, node_errors, visiting) {
                    ok = false;
                }
            }
        }
    }
    visiting.remove(id);
    if !errors.is_empty() {
        ok = false;
        node_errors.insert(id.to_string(), json!({ "errors": errors, "dependent_outputs": [], "class_type": node.class_type }));
    }
    checked.insert(id.to_string(), ok);
    ok
}

/// Topological execution order covering everything the given outputs depend on.
pub fn execution_order(graph: &PromptGraph, outputs: &[String]) -> Vec<String> {
    fn visit(graph: &PromptGraph, id: &str, seen: &mut HashSet<String>, order: &mut Vec<String>) {
        if !seen.insert(id.to_string()) {
            return;
        }
        if let Some(node) = graph.nodes.get(id) {
            for v in node.inputs.values() {
                if let Some(InputRef::Link { node: src, .. }) = classify(v) {
                    visit(graph, &src, seen, order);
                }
            }
        }
        order.push(id.to_string());
    }
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for out in outputs {
        visit(graph, out, &mut seen, &mut order);
    }
    order
}
