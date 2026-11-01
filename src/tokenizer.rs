//! Qwen2 byte-level BPE tokenizer (matches transformers' Qwen2Tokenizer as used by ComfyUI).

use anyhow::{anyhow, Result};
use fancy_regex::Regex;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

const VOCAB: &str = include_str!("../assets/qwen25_tokenizer/vocab.json");
const MERGES: &str = include_str!("../assets/qwen25_tokenizer/merges.txt");
const CONFIG: &str = include_str!("../assets/qwen25_tokenizer/tokenizer_config.json");
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
        let vocab: HashMap<String, u32> = serde_json::from_str(VOCAB)?;
        let mut ranks = HashMap::new();
        for (i, line) in MERGES.lines().skip(1).enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i as u32);
            }
        }
        let cfg: serde_json::Value = serde_json::from_str(CONFIG)?;
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
