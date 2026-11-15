//! Qwen2 byte-level BPE tokenizer (matches transformers' Qwen2Tokenizer as used by ComfyUI), the CLIP tokenizer of SD 1.x
//! and SDXL, and ComfyUI's prompt weights.

use anyhow::{anyhow, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

const VOCAB: &str = include_str!("../assets/qwen25_tokenizer/vocab.json");
const MERGES: &str = include_str!("../assets/qwen25_tokenizer/merges.txt");
const CONFIG: &str = include_str!("../assets/qwen25_tokenizer/tokenizer_config.json");
const VOCAB35: &str = include_str!("../assets/qwen35_tokenizer/vocab.json");
const MERGES35: &str = include_str!("../assets/qwen35_tokenizer/merges.txt");
const CONFIG35: &str = include_str!("../assets/qwen35_tokenizer/tokenizer_config.json");
const PRETOKENIZE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    special: Vec<(String, u32)>,
    re: Regex,
    byte_enc: [char; 256],
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

impl Tokenizer {
    pub fn new() -> Result<Self> {
        Self::build(VOCAB, MERGES, CONFIG)
    }

    /// The tokenizer of Qwen3.5 text encoders.
    pub fn qwen35() -> Result<Self> {
        Self::build(VOCAB35, MERGES35, CONFIG35)
    }

    fn build(vocab: &str, merges: &str, config: &str) -> Result<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_str(vocab)?;
        let mut ranks = HashMap::new();
        for (i, line) in merges.lines().skip(1).enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i as u32);
            }
        }
        let cfg: serde_json::Value = serde_json::from_str(config)?;
        let mut special = Vec::new();
        for (id, t) in cfg["added_tokens_decoder"].as_object().ok_or_else(|| anyhow!("tokenizer config"))? {
            special.push((t["content"].as_str().unwrap().to_string(), id.parse::<u32>()?));
        }
        // longest first so that overlapping specials split greedily
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Tokenizer { vocab, ranks, special, re: Regex::new(PRETOKENIZE)?, byte_enc: bytes_to_unicode() })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // earliest special token occurrence
            let mut best: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.special {
                if let Some(p) = rest.find(s.as_str()) {
                    if best.map_or(true, |b| p < b.0) {
                        best = Some((p, s, *id));
                    }
                }
            }
            match best {
                Some((p, s, id)) => {
                    self.encode_plain(&rest[..p], &mut out)?;
                    out.push(id);
                    rest = &rest[p + s.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut out)?;
                    break;
                }
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let norm: String = text.nfc().collect();
        for m in self.re.find_iter(&norm) {
            let piece: String = m?.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            for tok in self.bpe(&piece) {
                out.push(*self.vocab.get(&tok).ok_or_else(|| anyhow!("token {tok:?} not in vocab"))?);
            }
        }
        Ok(())
    }

    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |b| r < b.0) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let (a, b) = (parts[i].clone(), parts[i + 1].clone());
            // merge every occurrence of the pair, left to right
            let mut merged = Vec::with_capacity(parts.len());
            let mut j = 0;
            while j < parts.len() {
                if j + 1 < parts.len() && parts[j] == a && parts[j + 1] == b {
                    merged.push(format!("{a}{b}"));
                    j += 2;
                } else {
                    merged.push(parts[j].clone());
                    j += 1;
                }
            }
            parts = merged;
        }
        parts
    }
}

const CLIP_VOCAB: &str = include_str!("../assets/sd1_tokenizer/vocab.json");
const CLIP_MERGES: &str = include_str!("../assets/sd1_tokenizer/merges.txt");
const CLIP_PATTERN: &str = r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";
pub const CLIP_START: u32 = 49406;
pub const CLIP_END: u32 = 49407;
/// Tokens in one CLIP chunk, start and end included.
pub const CLIP_CHUNK: usize = 77;
/// `SDTokenizer.max_word_length`: a segment this long may be split between chunks.
const MAX_WORD: usize = 8;

/// transformers' `CLIPTokenizer` as ComfyUI runs it, without ftfy: its `BasicTokenizer` cleans and lowercases the text,
/// then byte-level BPE with `</w>` ending each word.
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    re: Regex,
    byte_enc: [char; 256],
}

/// `BasicTokenizer._is_chinese_char`.
fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF
        | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// Format characters (category Cf), which `BasicTokenizer._is_control` drops with the control characters.
fn is_format(c: char) -> bool {
    matches!(c as u32, 0xAD | 0x600..=0x605 | 0x61C | 0x6DD | 0x70F | 0x180E | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064
        | 0x2066..=0x206F | 0xFEFF | 0xFFF9..=0xFFFB | 0xE0001 | 0xE0020..=0xE007F)
}

impl ClipTokenizer {
    pub fn new() -> Result<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_str(CLIP_VOCAB)?;
        let mut ranks = HashMap::new();
        // CLIPTokenizer reads merges[1 : 49152 - 256 - 2 + 1]
        for (i, line) in CLIP_MERGES.lines().skip(1).take(49152 - 256 - 2).enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i as u32);
            }
        }
        Ok(ClipTokenizer { vocab, ranks, re: Regex::new(CLIP_PATTERN)?, byte_enc: bytes_to_unicode() })
    }

    /// `BasicTokenizer(strip_accents=False, do_split_on_punc=False).tokenize`, joined with spaces.
    fn clean(text: &str) -> String {
        let mut s = String::with_capacity(text.len());
        for c in text.chars() {
            if c == '\0' || c == '\u{fffd}' || (c.is_control() && !matches!(c, '\t' | '\n' | '\r')) || is_format(c) {
                continue;
            }
            if c.is_whitespace() {
                s.push(' ');
            } else if is_cjk(c) {
                s.push(' ');
                s.push(c);
                s.push(' ');
            } else {
                s.push(c);
            }
        }
        let norm: String = s.nfc().collect();
        norm.split_whitespace().map(|w| w.to_lowercase()).collect::<Vec<_>>().join(" ")
    }

    /// The ids of `text`, without the start and end tokens the tokenizer adds.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let clean = Self::clean(text);
        let mut out = Vec::new();
        for m in self.re.find_iter(&clean) {
            match m?.as_str() {
                "<|startoftext|>" => out.push(CLIP_START),
                "<|endoftext|>" => out.push(CLIP_END),
                m => {
                    let piece: Vec<String> = m.bytes().map(|b| self.byte_enc[b as usize].to_string()).collect();
                    for tok in self.bpe(piece) {
                        out.push(self.vocab.get(&tok).copied().unwrap_or(CLIP_END));
                    }
                }
            }
        }
        Ok(out)
    }

    fn bpe(&self, mut parts: Vec<String>) -> Vec<String> {
        if let Some(last) = parts.last_mut() {
            last.push_str("</w>");
        }
        while parts.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |b| r < b.0) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let (a, b) = (parts[i].clone(), parts[i + 1].clone());
            let mut merged = Vec::with_capacity(parts.len());
            let mut j = 0;
            while j < parts.len() {
                if j + 1 < parts.len() && parts[j] == a && parts[j + 1] == b {
                    merged.push(format!("{a}{b}"));
                    j += 2;
                } else {
                    merged.push(parts[j].clone());
                    j += 1;
                }
            }
            parts = merged;
        }
        parts
    }
}

/// True when ComfyUI would read `text` as naming a textual inversion embedding: `embedding:` at its start or after
/// whitespace.
fn names_embedding(text: &str) -> bool {
    text.starts_with("embedding:") || text.char_indices().any(|(i, c)| c.is_whitespace() && text[i + c.len_utf8()..].starts_with("embedding:"))
}

/// `SDTokenizer.tokenize_with_weights` for weighted segments (from [`token_weights`]): chunks of [`CLIP_CHUNK`] (token,
/// weight) pairs, each starting and ending with the start and end tokens and padded with `pad`. A segment shorter than
/// eight tokens that doesn't fit moves whole to the next chunk. Embedding names are refused: none load here.
pub fn clip_chunks(t: &ClipTokenizer, segments: &[(String, f32)], pad: u32) -> Result<Vec<Vec<(u32, f32)>>> {
    let mut groups: Vec<Vec<(u32, f32)>> = Vec::new();
    for (text, weight) in segments {
        if text.is_empty() {
            continue;
        }
        if names_embedding(text) {
            return Err(crate::lora::Unsupported("textual inversion embeddings in prompts".into()).into());
        }
        groups.push(t.encode(text)?.into_iter().map(|id| (id, *weight)).collect());
    }
    let mut chunks = Vec::new();
    let mut batch = vec![(CLIP_START, 1.0)];
    for mut g in groups {
        let large = g.len() >= MAX_WORD;
        while !g.is_empty() {
            if g.len() + batch.len() > CLIP_CHUNK - 1 {
                let remaining = CLIP_CHUNK - batch.len() - 1;
                if large {
                    batch.extend(g.drain(..remaining));
                    batch.push((CLIP_END, 1.0));
                } else {
                    batch.push((CLIP_END, 1.0));
                    batch.resize(batch.len() + remaining, (pad, 1.0));
                }
                chunks.push(std::mem::replace(&mut batch, vec![(CLIP_START, 1.0)]));
            } else {
                batch.append(&mut g);
            }
        }
    }
    batch.push((CLIP_END, 1.0));
    batch.resize(CLIP_CHUNK.max(batch.len()), (pad, 1.0));
    chunks.push(batch);
    Ok(chunks)
}

const T5_JSON: &[u8] = include_bytes!("../assets/t5_tokenizer/tokenizer.json");
const QWEN_PAD: u32 = 151643;
const T5_END: u32 = 1;

/// T5 tokenizer (ids only: Anima's adapter embeds them itself).
pub struct T5Tokenizer(tokenizers::Tokenizer);

impl T5Tokenizer {
    pub fn new() -> Result<Self> {
        Ok(T5Tokenizer(tokenizers::Tokenizer::from_bytes(T5_JSON).map_err(|e| anyhow!("T5 tokenizer: {e}"))?))
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.0.encode(text, false).map_err(|e| anyhow!("T5 tokenizer: {e}"))?.get_ids().to_vec())
    }
}

/// `sd1_clip.parse_parentheses`.
fn parse_parentheses(text: &str) -> Vec<String> {
    let (mut out, mut cur, mut nesting) = (Vec::new(), String::new(), 0i32);
    for c in text.chars() {
        match c {
            '(' => {
                if nesting == 0 {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    cur.push('(');
                } else {
                    cur.push(c);
                }
                nesting += 1;
            }
            ')' => {
                nesting -= 1;
                if nesting == 0 {
                    cur.push(')');
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(c);
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn weights_inner(text: &str, weight: f64, out: &mut Vec<(String, f32)>) {
    for x in parse_parentheses(text) {
        if x.len() >= 2 && x.starts_with('(') && x.ends_with(')') {
            let inner = &x[1..x.len() - 1];
            let mut w = weight * 1.1;
            let mut body = inner;
            if let Some(i) = inner.rfind(':').filter(|i| *i > 0) {
                if let Ok(v) = inner[i + 1..].trim().parse::<f64>() {
                    w = v;
                    body = &inner[..i];
                }
            }
            weights_inner(body, w, out);
        } else {
            out.push((x, weight as f32));
        }
    }
}

/// ComfyUI's emphasis syntax: `(text:1.2)` and `(text)`, `\(` and `\)` being literal.
pub fn token_weights(text: &str, weight: f32) -> Vec<(String, f32)> {
    let escaped = text.replace("\\)", "\0\u{1}").replace("\\(", "\0\u{2}");
    let mut out = Vec::new();
    weights_inner(&escaped, weight as f64, &mut out);
    for (t, _) in &mut out {
        *t = t.replace("\0\u{1}", ")").replace("\0\u{2}", "(");
    }
    out
}

/// `AnimaTokenizer`: Qwen3 ids (weights dropped) and T5 ids with weights, for weighted segments.
pub fn anima_tokens(qwen: &Tokenizer, t5: &T5Tokenizer, segments: &[(String, f32)]) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let (mut q, mut t, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for (text, weight) in segments.iter().filter(|(text, _)| !text.is_empty()) {
        q.extend(qwen.encode(text)?);
        for id in t5.encode(text)? {
            t.push(id);
            w.push(*weight);
        }
    }
    if q.is_empty() {
        q.push(QWEN_PAD);
    }
    t.push(T5_END);
    w.push(1.0);
    Ok((q, t, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emphasis_matches_comfy() {
        let w = token_weights("a (b:1.2) c \\(d\\) (e)", 1.0);
        assert_eq!(w[0], ("a ".to_string(), 1.0));
        assert_eq!(w[1], ("b".to_string(), 1.2));
        assert_eq!(w[2], (" c (d) ".to_string(), 1.0));
        assert_eq!(w[3].0, "e");
        assert!((w[3].1 - 1.1).abs() < 1e-6);
    }
}
