//! Parser for the SwarmUI prompt tags that survive into the ComfyUI workflow.
//!
//! SwarmUI resolves most of its prompt syntax (wildcards, `<random>`, `<lora:>`
//! loading, regions, ...) before it builds the workflow. What reaches
//! `SwarmTextEncodeAdvanced` is this subset, which this module parses exactly
//! the way `SwarmText.py::parse_prompt` does:
//!
//! - `<break>`: split into separately encoded chunks
//! - `<weight[1.5]:text>`: weight a span
//! - `<alternate:a, b>` / `<alt:a|b>`: swap per sampling step
//! - `<fromto[0.5]:a, b>`: `a` before the step (or fraction), `b` after
//! - `<embed:name>` / `<embedding:name>`: textual inversion embedding
//! - `<lora:name//hook=N//>`: reference to the Nth entry of `lora_hooks`
//!
//! Classic `(text:1.2)` emphasis is NOT handled here; ComfyUI's tokenizer applies
//! it, so the engine's tokenizer is expected to handle it inside `Text` leaves.

#[derive(Clone, Debug, PartialEq)]
pub enum PromptNode {
    Text(String),
    Embed(String),
    Break,
    LoraHook(u32),
    Group { weight: f64, parts: Vec<PromptNode> },
    Alternate(Vec<PromptNode>),
    FromTo { when: f64, from: Box<PromptNode>, to: Box<PromptNode> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum PromptLeafKind {
    Text(String),
    Embed(String),
    Break,
    LoraHook(u32),
}

/// A flattened, weighted prompt piece for a specific sampling step.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptLeaf {
    pub kind: PromptLeafKind,
    pub weight: f64,
}

/// A parsed prompt, plus the raw text it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedPrompt {
    pub raw: String,
    pub root: PromptNode,
}

impl ParsedPrompt {
    pub fn parse(raw: &str) -> Self {
        Self { raw: raw.to_string(), root: parse_prompt(raw) }
    }

    /// True if the prompt changes across steps (`<alternate>` / `<fromto>`), in which case
    /// the engine must encode per-step (or per distinct step range) conditioning.
    pub fn has_steps(&self) -> bool {
        self.root.has_steps()
    }

    /// The leaves active at `step` out of `steps`.
    pub fn flatten(&self, step: u32, steps: u32) -> Vec<PromptLeaf> {
        self.root.flatten(step, steps, 1.0)
    }

    /// The leaves active at `step`, split into `<break>`-separated chunks.
    pub fn chunks(&self, step: u32, steps: u32) -> Vec<Vec<PromptLeaf>> {
        let mut out = vec![Vec::new()];
        for leaf in self.flatten(step, steps) {
            if leaf.kind == PromptLeafKind::Break {
                out.push(Vec::new());
            } else {
                out.last_mut().unwrap().push(leaf);
            }
        }
        out
    }
}

impl PromptNode {
    pub fn has_steps(&self) -> bool {
        match self {
            PromptNode::Alternate(_) | PromptNode::FromTo { .. } => true,
            PromptNode::Group { parts, .. } => parts.iter().any(|p| p.has_steps()),
            _ => false,
        }
    }

    pub fn flatten(&self, step: u32, steps: u32, weight: f64) -> Vec<PromptLeaf> {
        match self {
            PromptNode::Alternate(options) => {
                if options.is_empty() {
                    return Vec::new();
                }
                options[step as usize % options.len()].flatten(step, steps, weight)
            }
            PromptNode::FromTo { when, from, to } => {
                let when = if *when < 1.0 { when * steps as f64 } else { *when };
                let chosen = if (step as f64) < when { from } else { to };
                chosen.flatten(step, steps, weight)
            }
            PromptNode::Group { weight: w, parts } => parts.iter().flat_map(|p| p.flatten(step, steps, weight * w)).collect(),
            PromptNode::Text(t) if t.is_empty() => Vec::new(),
            PromptNode::Text(t) => vec![PromptLeaf { kind: PromptLeafKind::Text(t.clone()), weight }],
            PromptNode::Embed(e) => vec![PromptLeaf { kind: PromptLeafKind::Embed(e.clone()), weight }],
            PromptNode::Break => vec![PromptLeaf { kind: PromptLeafKind::Break, weight }],
            PromptNode::LoraHook(h) => vec![PromptLeaf { kind: PromptLeafKind::LoraHook(*h), weight }],
        }
    }
}

fn group(parts: Vec<PromptNode>) -> PromptNode {
    PromptNode::Group { weight: 1.0, parts }
}

/// Port of `quick_simple_tag_filler(text, '<', '>', reader, do_subtags=False)`.
fn tag_filler(text: &str, reader: &dyn Fn(&str) -> PromptNode) -> PromptNode {
    let Some(mut start) = text.find('<') else {
        return PromptNode::Text(text.to_string());
    };
    match text.rfind('>') {
        Some(end) if end >= start => {}
        _ => return PromptNode::Text(text.to_string()),
    }
    let bytes = text.as_bytes();
    let mut result = Vec::new();
    if start > 0 {
        result.push(PromptNode::Text(text[..start].to_string()));
    }
    let mut depth = 0i32;
    let mut i = start + 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'<' {
            depth += 1;
        }
        if c == b'>' {
            if depth == 0 {
                i += 1;
                let content = &text[start + 1..i - 1];
                result.push(reader(content));
                match text[i..].find('<') {
                    None => {
                        if i < text.len() {
                            result.push(PromptNode::Text(text[i..].to_string()));
                        }
                        return group(result);
                    }
                    Some(rel) => {
                        let next = i + rel;
                        if next > i {
                            result.push(PromptNode::Text(text[i..next].to_string()));
                        }
                        start = next;
                        i = start + 1;
                        continue;
                    }
                }
            }
            depth -= 1;
        }
        i += 1;
    }
    result.push(PromptNode::Text(text[start..].to_string()));
    group(result)
}

/// Index of `c` in `val` that is not nested inside `<...>`.
fn index_of_noncontained(val: &str, c: u8) -> Option<usize> {
    let mut depth = 0i32;
    for (i, &ch) in val.as_bytes().iter().enumerate() {
        if ch == b'<' {
            depth += 1;
        } else if ch == b'>' {
            depth -= 1;
        } else if ch == c && depth == 0 {
            return Some(i);
        }
    }
    None
}

/// Splits `prefix[predata]:data` into `(prefix lowercased, predata, data)`.
fn split_tag(tag: &str) -> (String, Option<String>, String) {
    let (mut prefix, data) = match index_of_noncontained(tag, b':') {
        Some(ci) => (tag[..ci].to_string(), tag[ci + 1..].to_string()),
        None => (tag.to_string(), String::new()),
    };
    let mut predata = None;
    if prefix.ends_with(']') {
        if let Some(b) = index_of_noncontained(&prefix, b'[') {
            predata = Some(prefix[b + 1..prefix.len() - 1].to_string());
            prefix = prefix[..b].to_string();
        }
    }
    (prefix.to_lowercase(), predata, data)
}

/// Splits on `,`, or `|` / `||` if present at the top level, ignoring separators inside tags.
fn split_smart(inp: &str) -> Vec<String> {
    let b = inp.as_bytes();
    let mut sep: &[u8] = b",";
    let mut depth = 0i32;
    for i in 0..b.len() {
        match b[i] {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            b'|' if depth == 0 && i > 0 && b[i - 1] == b'|' => {
                sep = b"||";
                break;
            }
            b'|' if depth == 0 => sep = b"|",
            _ => {}
        }
    }
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'<' {
            depth += 1;
        } else if b[i] == b'>' {
            depth -= 1;
        } else if depth == 0 && b[i..].starts_with(sep) {
            out.push(inp[start..i].trim().to_string());
            start = i + sep.len();
            i += sep.len() - 1;
        }
        i += 1;
    }
    if start <= b.len() {
        out.push(inp[start..].trim().to_string());
    }
    out
}

fn read_tag(content: &str) -> PromptNode {
    let (prefix, predata, data) = split_tag(content);
    let literal = || PromptNode::Text(format!("<{content}>"));
    match prefix.as_str() {
        "lora" => {
            const MARKER: &str = "//hook=";
            if let Some(pos) = data.find(MARKER) {
                let rest = &data[pos + MARKER.len()..];
                let id = rest.split("//").next().unwrap_or("");
                return id.parse::<u32>().map(PromptNode::LoraHook).unwrap_or_else(|_| literal());
            }
            literal()
        }
        "break" => PromptNode::Break,
        "embed" | "embedding" => PromptNode::Embed(data),
        "weight" => match predata.as_deref().and_then(|p| p.trim().parse::<f64>().ok()) {
            Some(w) => {
                let parts = match parse_prompt(&data) {
                    PromptNode::Group { parts, .. } if !parts.is_empty() => parts,
                    other => vec![other],
                };
                PromptNode::Group { weight: w, parts }
            }
            None => literal(),
        },
        "alternate" | "alt" => PromptNode::Alternate(split_smart(&data).iter().map(|v| parse_prompt(v)).collect()),
        "fromto" => {
            let Some(when) = predata.as_deref().and_then(|p| p.trim().parse::<f64>().ok()) else {
                return literal();
            };
            let vals = split_smart(&data);
            if vals.len() != 2 {
                return literal();
            }
            PromptNode::FromTo { when, from: Box::new(parse_prompt(&vals[0])), to: Box::new(parse_prompt(&vals[1])) }
        }
        _ => literal(),
    }
}

pub fn parse_prompt(text: &str) -> PromptNode {
    tag_filler(text, &read_tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(leaves: &[PromptLeaf]) -> Vec<(String, f64)> {
        leaves
            .iter()
            .map(|l| match &l.kind {
                PromptLeafKind::Text(t) => (t.clone(), l.weight),
                PromptLeafKind::Embed(e) => (format!("embed:{e}"), l.weight),
                PromptLeafKind::Break => ("BREAK".into(), l.weight),
                PromptLeafKind::LoraHook(h) => (format!("hook:{h}"), l.weight),
            })
            .collect()
    }

    #[test]
    fn plain_text() {
        let p = ParsedPrompt::parse("a cat, (sitting:1.2)");
        assert_eq!(p.root, PromptNode::Text("a cat, (sitting:1.2)".into()));
        assert!(!p.has_steps());
    }

    #[test]
    fn break_and_weight() {
        let p = ParsedPrompt::parse("a cat<break>a <weight[1.5]:red> hat");
        let chunks = p.chunks(0, 20);
        assert_eq!(chunks.len(), 2);
        assert_eq!(texts(&chunks[1]), vec![("a ".into(), 1.0), ("red".into(), 1.5), (" hat".into(), 1.0)]);
    }

    #[test]
    fn fromto_and_alternate() {
        let p = ParsedPrompt::parse("<fromto[0.5]:day, night> sky <alt:red|blue>");
        assert!(p.has_steps());
        assert_eq!(texts(&p.flatten(0, 10))[0].0, "day");
        assert_eq!(texts(&p.flatten(6, 10))[0].0, "night");
        assert_eq!(texts(&p.flatten(1, 10))[2].0, "blue");
    }

    #[test]
    fn embed_hook_and_unknown() {
        let p = ParsedPrompt::parse("<embed:foo> <lora:bar//hook=2//> <unknown:x>");
        let t = texts(&p.flatten(0, 1));
        assert_eq!(t[0].0, "embed:foo");
        assert_eq!(t[2].0, "hook:2");
        assert_eq!(t[4].0, "<unknown:x>");
    }

    #[test]
    fn nested_tags() {
        let p = ParsedPrompt::parse("<weight[2]:a <alt:b, c>>");
        assert_eq!(texts(&p.flatten(1, 2)), vec![("a ".into(), 2.0), ("c".into(), 2.0)]);
    }
}
