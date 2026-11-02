//! Node schemas: rendered into `/object_info` (ComfyUI "V1" format) and used to
//! validate incoming prompts the way `execution.validate_prompt` does.

use serde_json::{json, Map, Value as Json};

/// ComfyUI's `nodes.MAX_RESOLUTION`.
pub const MAX_RESOLUTION: i64 = 16384;
/// Max for seeds (`0xffffffffffffffff`). JSON can't hold u64::MAX exactly in every client,
/// but ComfyUI sends it this way too.
pub const MAX_SEED: i64 = i64::MAX;

#[derive(Clone, Debug)]
pub enum InputKind {
    Int { default: i64, min: i64, max: i64, step: Option<i64> },
    Float { default: f64, min: f64, max: f64, step: Option<f64> },
    String { default: String, multiline: bool },
    Boolean { default: bool },
    Combo { options: Vec<String>, default: Option<String> },
    /// A link-only input of the given ComfyUI type (`"MODEL"`, `"CLIP,GEMMA"`, `"*"`).
    Link(String),
    /// ComfyUI V3 `Autogrow` (dynamic `prefix0..prefixN` inputs, sent as `"{name}.{prefix}{i}"`).
    /// With `names` (V3 `TemplateNames`), the members are those names instead (`prefix` unused).
    Autogrow { prefix: String, min: u32, max: u32, item: Box<InputSpec>, names: Vec<String> },
    /// ComfyUI V3 `DynamicCombo`: a combo whose chosen option brings its own sub-inputs,
    /// sent as `"{name}.{sub}"` (e.g. `selection` = `"sol-attn"`, `selection.tau` = `1.3`).
    DynamicCombo { options: Vec<(String, Vec<InputSpec>)> },
    /// ComfyUI V3 `MatchType`: a link of any of `allowed` (comma-separated); outputs declared as
    /// [`MATCH_TYPE`] have the type of whatever was linked here.
    MatchType { template_id: String, allowed: String },
}

/// The V3 `MatchType` link type name. Compatible with everything, as in ComfyUI's validation.
pub const MATCH_TYPE: &str = "COMFY_MATCHTYPE_V3";

#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub kind: InputKind,
    pub optional: bool,
    pub tooltip: Option<String>,
    pub advanced: bool,
    /// Widget declares no default (ComfyUI leaves `default` out, e.g. prompt text, base64 payloads).
    pub no_default: bool,
    /// Optional link that declares `"default": null` (some Swarm nodes do).
    pub null_default: bool,
    /// Widget rendered as an input socket (`"forceInput": true`); literals are still accepted.
    pub force_input: bool,
    /// The original V1 entry for schemas parsed with [`input_from_v1`]; `/object_info` shows it
    /// verbatim (validation uses the parsed form).
    pub raw_v1: Option<Json>,
}

impl InputSpec {
    fn new(name: &str, kind: InputKind) -> Self {
        Self { name: name.to_string(), kind, optional: false, tooltip: None, advanced: false, no_default: false, null_default: false, force_input: false, raw_v1: None }
    }

    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    pub fn tooltip(mut self, t: &str) -> Self {
        self.tooltip = Some(t.to_string());
        self
    }

    pub fn advanced(mut self) -> Self {
        self.advanced = true;
        self
    }

    pub fn no_default(mut self) -> Self {
        self.no_default = true;
        self
    }

    pub fn null_default(mut self) -> Self {
        self.null_default = true;
        self
    }

    pub fn force_input(mut self) -> Self {
        self.force_input = true;
        self
    }

    pub fn is_widget(&self) -> bool {
        !matches!(self.kind, InputKind::Link(_) | InputKind::Autogrow { .. } | InputKind::MatchType { .. })
    }

    /// The widget's default value as JSON (`None` for links).
    pub fn default_json(&self) -> Option<Json> {
        if self.no_default {
            return None;
        }
        Some(match &self.kind {
            InputKind::Int { default, .. } => json!(default),
            InputKind::Float { default, .. } => json!(default),
            InputKind::String { default, .. } => json!(default),
            InputKind::Boolean { default } => json!(default),
            InputKind::Combo { options, default } => json!(default.clone().or_else(|| options.first().cloned())?),
            InputKind::DynamicCombo { options } => json!(options.first()?.0),
            InputKind::Link(_) | InputKind::Autogrow { .. } | InputKind::MatchType { .. } => return None,
        })
    }

    /// V1 `object_info` entry: `["INT", {...}]`, `[[options], {...}]`, `["MODEL"]`, ...
    pub fn to_v1(&self) -> Json {
        if let Some(raw) = &self.raw_v1 {
            return raw.clone();
        }
        let mut opts = Map::new();
        let ty: Json = match &self.kind {
            InputKind::Int { default, min, max, step } => {
                opts.insert("default".into(), json!(default));
                opts.insert("min".into(), json!(min));
                // ComfyUI's seed max is 0xffffffffffffffff; values are i64 here (SwarmUI's seeds fit),
                // but the advertised schema matches.
                opts.insert("max".into(), if *max == MAX_SEED { json!(u64::MAX) } else { json!(max) });
                if let Some(s) = step {
                    opts.insert("step".into(), json!(s));
                }
                json!("INT")
            }
            InputKind::Float { default, min, max, step } => {
                opts.insert("default".into(), json!(default));
                opts.insert("min".into(), json!(min));
                opts.insert("max".into(), json!(max));
                if let Some(s) = step {
                    opts.insert("step".into(), json!(s));
                }
                json!("FLOAT")
            }
            InputKind::String { default, multiline } => {
                opts.insert("default".into(), json!(default));
                if *multiline {
                    opts.insert("multiline".into(), json!(true));
                }
                json!("STRING")
            }
            InputKind::Boolean { default } => {
                opts.insert("default".into(), json!(default));
                json!("BOOLEAN")
            }
            InputKind::Combo { options, default } => {
                if let Some(d) = default {
                    opts.insert("default".into(), json!(d));
                }
                json!(options)
            }
            InputKind::Link(t) => json!(t),
            InputKind::MatchType { template_id, allowed } => {
                opts.insert("template".into(), json!({ "template_id": template_id, "allowed_types": allowed }));
                json!(MATCH_TYPE)
            }
            InputKind::Autogrow { prefix, min, max, item, names } => {
                let mut tmpl_input = Map::new();
                let section = if item.optional { "optional" } else { "required" };
                tmpl_input.insert(section.into(), json!({ item.name.clone(): item.to_v1() }));
                let tmpl = if names.is_empty() {
                    json!({ "input": tmpl_input, "prefix": prefix, "min": min, "max": max })
                } else {
                    json!({ "input": tmpl_input, "names": names, "min": min })
                };
                opts.insert("template".into(), tmpl);
                json!("COMFY_AUTOGROW_V3")
            }
            InputKind::DynamicCombo { options } => {
                let rendered: Vec<Json> = options
                    .iter()
                    .map(|(key, inputs)| {
                        let mut required = Map::new();
                        let mut optional = Map::new();
                        for i in inputs {
                            if i.optional { &mut optional } else { &mut required }.insert(i.name.clone(), i.to_v1());
                        }
                        let mut sections = Map::new();
                        sections.insert("required".into(), Json::Object(required));
                        if !optional.is_empty() {
                            sections.insert("optional".into(), Json::Object(optional));
                        }
                        json!({ "key": key, "inputs": sections })
                    })
                    .collect();
                opts.insert("options".into(), json!(rendered));
                json!("COMFY_DYNAMICCOMBO_V3")
            }
        };
        if self.no_default {
            opts.remove("default");
        }
        if self.null_default {
            opts.insert("default".into(), Json::Null);
        }
        if self.force_input {
            opts.insert("forceInput".into(), json!(true));
        }
        if let Some(t) = &self.tooltip {
            opts.insert("tooltip".into(), json!(t));
        }
        if self.advanced {
            opts.insert("advanced".into(), json!(true));
        }
        if opts.is_empty() {
            json!([ty])
        } else {
            json!([ty, opts])
        }
    }
}

pub fn match_type(name: &str, template_id: &str, allowed: &str) -> InputSpec {
    InputSpec::new(name, InputKind::MatchType { template_id: template_id.to_string(), allowed: allowed.to_string() })
}

pub fn int(name: &str, default: i64, min: i64, max: i64) -> InputSpec {
    InputSpec::new(name, InputKind::Int { default, min, max, step: None })
}

pub fn int_step(name: &str, default: i64, min: i64, max: i64, step: i64) -> InputSpec {
    InputSpec::new(name, InputKind::Int { default, min, max, step: Some(step) })
}

pub fn float(name: &str, default: f64, min: f64, max: f64, step: f64) -> InputSpec {
    InputSpec::new(name, InputKind::Float { default, min, max, step: Some(step) })
}

/// A FLOAT widget that declares no `step` (some Swarm nodes).
pub fn float_nostep(name: &str, default: f64, min: f64, max: f64) -> InputSpec {
    InputSpec::new(name, InputKind::Float { default, min, max, step: None })
}

pub fn string(name: &str, default: &str, multiline: bool) -> InputSpec {
    InputSpec::new(name, InputKind::String { default: default.to_string(), multiline })
}

pub fn boolean(name: &str, default: bool) -> InputSpec {
    InputSpec::new(name, InputKind::Boolean { default })
}

pub fn combo<S: AsRef<str>>(name: &str, options: &[S]) -> InputSpec {
    InputSpec::new(name, InputKind::Combo { options: options.iter().map(|s| s.as_ref().to_string()).collect(), default: None })
}

pub fn combo_default<S: AsRef<str>>(name: &str, options: &[S], default: &str) -> InputSpec {
    InputSpec::new(
        name,
        InputKind::Combo { options: options.iter().map(|s| s.as_ref().to_string()).collect(), default: Some(default.to_string()) },
    )
}

pub fn link(name: &str, ty: &str) -> InputSpec {
    InputSpec::new(name, InputKind::Link(ty.to_string()))
}

pub fn autogrow(name: &str, prefix: &str, min: u32, max: u32, item: InputSpec) -> InputSpec {
    InputSpec::new(name, InputKind::Autogrow { prefix: prefix.to_string(), min, max, item: Box::new(item), names: Vec::new() })
}

pub fn dynamic_combo(name: &str, options: Vec<(&str, Vec<InputSpec>)>) -> InputSpec {
    InputSpec::new(name, InputKind::DynamicCombo { options: options.into_iter().map(|(k, i)| (k.to_string(), i)).collect() })
}

/// Parses one V1 `object_info` input entry (`["INT", {...}]`, `[[options]]`, `["MODEL"]`, V3
/// Autogrow / DynamicCombo / MatchType). Keeps what validation and SwarmUI use (defaults, ranges,
/// steps, options, tooltips, `advanced`, `forceInput`); display-only keys are dropped.
pub fn input_from_v1(name: &str, spec: &Json, optional: bool) -> InputSpec {
    let empty = Map::new();
    let o = spec.get(1).and_then(Json::as_object).unwrap_or(&empty);
    let def = o.get("default");
    let int_of = |v: Option<&Json>, or: i64| v.and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|_| MAX_SEED)).or_else(|| v.as_f64().map(|f| f as i64))).unwrap_or(or);
    let float_of = |v: Option<&Json>, or: f64| v.and_then(Json::as_f64).unwrap_or(or);
    let strings = |v: Option<&Json>| -> Vec<String> { v.and_then(Json::as_array).map(|a| a.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect()).unwrap_or_default() };
    let kind = match &spec[0] {
        Json::Array(_) => InputKind::Combo { options: strings(Some(&spec[0])), default: def.and_then(Json::as_str).map(str::to_string) },
        // V3 MultiType widgets (`"FLOAT,INT"`) validate as their first scalar type.
        Json::String(t) => match t.split(',').map(str::trim).find(|p| ["INT", "FLOAT", "STRING", "BOOLEAN"].contains(p)).filter(|_| t.contains(',')).unwrap_or(t.as_str()) {
            "COMBO" => InputKind::Combo { options: strings(o.get("options")), default: def.and_then(Json::as_str).map(str::to_string) },
            "INT" => InputKind::Int { default: int_of(def, 0), min: int_of(o.get("min"), i64::MIN), max: int_of(o.get("max"), i64::MAX), step: o.get("step").and_then(Json::as_i64) },
            "FLOAT" => InputKind::Float { default: float_of(def, 0.0), min: float_of(o.get("min"), f64::MIN), max: float_of(o.get("max"), f64::MAX), step: o.get("step").and_then(Json::as_f64) },
            "STRING" => InputKind::String { default: def.and_then(Json::as_str).unwrap_or("").to_string(), multiline: o.get("multiline").and_then(Json::as_bool).unwrap_or(false) },
            "BOOLEAN" => InputKind::Boolean { default: def.and_then(Json::as_bool).unwrap_or(false) },
            "COMFY_DYNAMICCOMBO_V3" => {
                let options = o.get("options").and_then(Json::as_array).cloned().unwrap_or_default();
                InputKind::DynamicCombo {
                    options: options
                        .iter()
                        .map(|opt| {
                            let mut subs = Vec::new();
                            for (sec, opt_flag) in [("required", false), ("optional", true)] {
                                if let Some(m) = opt["inputs"].get(sec).and_then(Json::as_object) {
                                    subs.extend(m.iter().map(|(k, v)| input_from_v1(k, v, opt_flag)));
                                }
                            }
                            (opt["key"].as_str().unwrap_or_default().to_string(), subs)
                        })
                        .collect(),
                }
            }
            "COMFY_AUTOGROW_V3" => {
                let tmpl = o.get("template").cloned().unwrap_or_default();
                let (item_name, item_spec, item_opt) = ["required", "optional"]
                    .iter()
                    .find_map(|sec| tmpl["input"].get(*sec).and_then(Json::as_object).and_then(|m| m.iter().next()).map(|(k, v)| (k.clone(), v.clone(), *sec == "optional")))
                    .unwrap_or_else(|| ("item".into(), json!(["*"]), true));
                let names: Vec<String> = strings(tmpl.get("names"));
                InputKind::Autogrow {
                    prefix: tmpl["prefix"].as_str().unwrap_or("").to_string(),
                    min: tmpl["min"].as_u64().unwrap_or(1) as u32,
                    max: tmpl["max"].as_u64().unwrap_or(names.len() as u64) as u32,
                    item: Box::new(input_from_v1(&item_name, &item_spec, item_opt)),
                    names,
                }
            }
            MATCH_TYPE => InputKind::MatchType {
                template_id: o.get("template").and_then(|t| t["template_id"].as_str()).unwrap_or("T").to_string(),
                allowed: o.get("template").and_then(|t| t["allowed_types"].as_str()).unwrap_or("*").to_string(),
            },
            other => InputKind::Link(other.to_string()),
        },
        other => InputKind::Link(other.to_string()),
    };
    let mut s = InputSpec::new(name, kind);
    s.optional = optional;
    s.tooltip = o.get("tooltip").and_then(Json::as_str).map(str::to_string);
    s.advanced = o.get("advanced").and_then(Json::as_bool).unwrap_or(false);
    s.force_input = o.get("forceInput").and_then(Json::as_bool).unwrap_or(false);
    s.null_default = def.is_some_and(Json::is_null);
    s.no_default = def.is_none() && s.is_widget() && !matches!(s.kind, InputKind::Combo { .. } | InputKind::DynamicCombo { .. } | InputKind::Boolean { .. });
    s.raw_v1 = Some(spec.clone());
    s
}

#[derive(Clone, Debug)]
pub struct OutputSpec {
    pub ty: String,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct NodeSchema {
    pub class: String,
    pub display_name: String,
    pub category: String,
    pub description: String,
    pub python_module: String,
    pub output_node: bool,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<OutputSpec>,
}

impl NodeSchema {
    pub fn new(class: &str, display_name: &str, category: &str) -> Self {
        let python_module = if class.starts_with("Swarm") { "custom_nodes.SwarmComfyCommon" } else { "nodes" };
        Self {
            class: class.to_string(),
            display_name: display_name.to_string(),
            category: category.to_string(),
            description: String::new(),
            python_module: python_module.to_string(),
            output_node: false,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// A node schema from its V1 `object_info` entry (stock ComfyUI dumps, third-party packs).
    pub fn from_v1(class: &str, info: &Json) -> Self {
        let display = info["display_name"].as_str().filter(|s| !s.is_empty()).unwrap_or(class);
        let mut s = Self::new(class, display, info["category"].as_str().unwrap_or(""));
        s.description = info["description"].as_str().unwrap_or("").to_string();
        if let Some(m) = info["python_module"].as_str() {
            s.python_module = m.to_string();
        }
        s.output_node = info["output_node"].as_bool().unwrap_or(false);
        // Keep ComfyUI's declared order when present.
        for (sec, optional) in [("required", false), ("optional", true)] {
            let Some(m) = info["input"].get(sec).and_then(Json::as_object) else { continue };
            let order: Vec<String> = info["input_order"][sec].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_else(|| m.keys().cloned().collect());
            for k in order {
                if let Some(v) = m.get(&k) {
                    s.inputs.push(input_from_v1(&k, v, optional));
                }
            }
        }
        let names = info["output_name"].as_array().cloned().unwrap_or_default();
        for (i, ty) in info["output"].as_array().cloned().unwrap_or_default().iter().enumerate() {
            let ty = ty.as_str().map(str::to_string).unwrap_or_else(|| "COMBO".to_string());
            let name = names.get(i).and_then(Json::as_str).unwrap_or(&ty).to_string();
            s.outputs.push(OutputSpec { ty, name });
        }
        s
    }

    pub fn describe(mut self, d: &str) -> Self {
        self.description = d.to_string();
        self
    }

    pub fn input(mut self, i: InputSpec) -> Self {
        self.inputs.push(i);
        self
    }

    pub fn output(mut self, ty: &str, name: &str) -> Self {
        self.outputs.push(OutputSpec { ty: ty.to_string(), name: name.to_string() });
        self
    }

    pub fn output_node(mut self) -> Self {
        self.output_node = true;
        self
    }

    pub fn find_input(&self, name: &str) -> Option<&InputSpec> {
        self.inputs.iter().find(|i| i.name == name)
    }

    /// Finds the spec for a (possibly Autogrow-expanded `"group.prefixN"`, or DynamicCombo
    /// `"combo.sub"`) input name.
    pub fn resolve_input(&self, name: &str) -> Option<InputSpec> {
        if let Some(i) = self.find_input(name) {
            return Some(i.clone());
        }
        let (group, member) = name.split_once('.')?;
        let spec = self.find_input(group)?;
        let mut s = match &spec.kind {
            InputKind::Autogrow { names, item, .. } if !names.is_empty() => {
                names.iter().any(|n| n == member).then(|| (**item).clone())?
            }
            InputKind::Autogrow { prefix, max, item, .. } => {
                let idx: u32 = member.strip_prefix(prefix.as_str())?.parse().ok()?;
                if idx >= *max {
                    return None;
                }
                (**item).clone()
            }
            InputKind::DynamicCombo { options } => options.iter().flat_map(|(_, inputs)| inputs).find(|i| i.name == member)?.clone(),
            _ => return None,
        };
        s.name = name.to_string();
        s.optional = true;
        Some(s)
    }

    pub fn to_v1(&self) -> Json {
        let mut required = Map::new();
        let mut optional = Map::new();
        for i in &self.inputs {
            if i.optional {
                optional.insert(i.name.clone(), i.to_v1());
            } else {
                required.insert(i.name.clone(), i.to_v1());
            }
        }
        let mut input = Map::new();
        let mut order = Map::new();
        order.insert("required".into(), json!(required.keys().collect::<Vec<_>>()));
        input.insert("required".into(), Json::Object(required.clone()));
        if !optional.is_empty() {
            order.insert("optional".into(), json!(optional.keys().collect::<Vec<_>>()));
            input.insert("optional".into(), Json::Object(optional));
        }
        json!({
            "input": input,
            "input_order": order,
            "output": self.outputs.iter().map(|o| o.ty.clone()).collect::<Vec<_>>(),
            "output_is_list": self.outputs.iter().map(|_| false).collect::<Vec<_>>(),
            "output_name": self.outputs.iter().map(|o| o.name.clone()).collect::<Vec<_>>(),
            "name": self.class,
            "display_name": self.display_name,
            "description": self.description,
            "python_module": self.python_module,
            "category": self.category,
            "output_node": self.output_node,
        })
    }
}

/// ComfyUI link type compatibility: exact match, `*` wildcard, or comma-separated unions.
pub fn types_compatible(output_ty: &str, input_ty: &str) -> bool {
    if output_ty == "*" || input_ty == "*" || output_ty == MATCH_TYPE || input_ty == MATCH_TYPE {
        return true;
    }
    let outs: Vec<&str> = output_ty.split(',').map(str::trim).collect();
    input_ty.split(',').map(str::trim).any(|i| outs.contains(&i))
}
