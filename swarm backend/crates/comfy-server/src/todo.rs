//! The node catalog beyond native support: every ComfyUI node class SwarmUI's workflow generator
//! can emit that this server doesn't implement natively, with its schema (stock ComfyUI, or
//! `tools/third_party_schemas.py`), generated into `todo_nodes.json` by `tools/gen_todo_nodes.py`
//! (see `docs/COMPATIBILITY.md`).
//!
//! Classes the engine claims (`Engine::supports_custom_node`) are registered from these schemas
//! and run through `Engine::custom_node` (`nodes/generic.rs`). A prompt that uses an unclaimed
//! one fails validation up front with a message naming the node and the SwarmUI parameters that
//! produce it, instead of ComfyUI's generic "custom node may not be installed". With
//! `--advertise-todo-nodes` unclaimed schemas are listed in `/object_info` anyway, which makes
//! SwarmUI turn on the matching features so their workflows can be inspected.
//!
//! Implementing a node = register it in `nodes/` and delete its entry from the catalog
//! (`tools/todo_catalog.py`), then regenerate. Registered nodes always win over this list.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::Value as Json;

#[derive(Debug, Deserialize)]
pub struct TodoArea {
    pub title: String,
    pub swarm_params: String,
}

/// What implementing a node takes (see `tools/todo_catalog.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoNeeds {
    /// Pure data work the server can do itself.
    Server,
    /// Server-side bookkeeping, but the engine must honor a new patch/metadata field.
    Patch,
    /// A new `Engine` operation.
    Engine,
    /// Part of supporting another model architecture.
    Arch,
}

#[derive(Debug, Deserialize)]
pub struct TodoNode {
    pub area: String,
    pub needs: TodoNeeds,
    pub note: String,
    /// SwarmUI parameters observed (live) to add this node to the workflow.
    pub triggers: Vec<String>,
    /// Where SwarmUI's generator creates it (`file:line`).
    pub swarm_source: Vec<String>,
    /// The node's entry in a stock ComfyUI `/object_info`; `None` for third-party node packs.
    pub object_info: Option<Json>,
}

#[derive(Debug, Deserialize)]
pub struct TodoCatalog {
    pub areas: BTreeMap<String, TodoArea>,
    pub nodes: BTreeMap<String, TodoNode>,
}

impl TodoCatalog {
    pub fn get() -> &'static TodoCatalog {
        static CATALOG: OnceLock<TodoCatalog> = OnceLock::new();
        CATALOG.get_or_init(|| serde_json::from_str(include_str!("todo_nodes.json")).expect("todo_nodes.json is invalid"))
    }

    pub fn node(&self, class: &str) -> Option<&TodoNode> {
        self.nodes.get(class)
    }

    /// One line of the user-facing error for a prompt that uses `class`.
    pub fn describe(&self, class: &str) -> Option<String> {
        let n = self.node(class)?;
        let area = self.areas.get(&n.area).map(|a| a.title.as_str()).unwrap_or(&n.area);
        let used_by = if n.triggers.is_empty() { String::new() } else { format!(" Used for: {}.", n.triggers.join(", ")) };
        Some(format!("- {class} ({area}): {}{used_by}", n.note))
    }

    /// The validation error message for a prompt that uses backlog nodes (`None` if none of
    /// `classes` are in the backlog).
    pub fn message<'a>(&self, classes: impl IntoIterator<Item = &'a str>) -> Option<String> {
        let mut lines: Vec<String> = classes.into_iter().filter_map(|c| self.describe(c)).collect();
        lines.sort();
        lines.dedup();
        if lines.is_empty() {
            return None;
        }
        Some(format!(
            "This backend doesn't implement {} node type(s) this workflow needs yet (see docs/COMPATIBILITY.md):\n{}",
            lines.len(),
            lines.join("\n")
        ))
    }
}
