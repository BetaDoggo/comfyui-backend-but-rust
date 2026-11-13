//! LoRAs, baked into the weights the way ComfyUI patches a quantized model: each patched int8 weight
//! is dequantized to bf16, the patches are added in bf16, and it is requantized with fresh row scales
//! and stochastic rounding seeded by the module's name. Sampling then runs the normal int8 kernels.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, DT_F32};
use crate::lora_adapters::{self as adapters, Tensor, Work};
use crate::offload::{Placed, Store, Stored};
use crate::st::{f32_bytes, SafeTensors};
use anyhow::{anyhow, Context, Result};
use half::bf16;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct Lora {
    pub path: PathBuf,
    pub strength: f32,
}

/// A LoRA this engine can't apply as written. The message is for the user.
#[derive(Debug)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Krea2,
    Anima,
    Moss,
    H3,
    Qwen21,
    /// Only for [`matched_modules`]: the engine can't patch SeedVR2.
    SeedVr2,
    /// `comfy` names the encoder inside ComfyUI's text encoder model, as in `qwen3vl_4b`.
    TextEncoder { comfy: &'static str },
}

impl Arch {
    /// ComfyUI's name for a module, which seeds its requantization.
    fn module(self, stem: &str) -> String {
        match self {
            Arch::TextEncoder { comfy } => format!("{comfy}.transformer.model.{stem}"),
            _ => format!("diffusion_model.{stem}"),
        }
    }

    /// The dtype ComfyUI patches in: its compute dtype, float32 for text encoders (`full_precision_mm`).
    fn dtype(self) -> i32 {
        if matches!(self, Arch::TextEncoder { .. }) { DT_F32 } else { DT_BF16 }
    }
}

fn elem(dt: i32) -> usize {
    if dt == DT_F32 { 4 } else { 2 }
}

fn encode(dt: i32, v: &[f32]) -> Vec<u8> {
    if dt == DT_F32 {
        return f32_bytes(v).to_vec();
    }
    let mut out = vec![0u8; v.len() * 2];
    adapters::par_items(&mut out, 2, |i, o| o.copy_from_slice(&bf16::from_f32(v[i]).to_le_bytes()));
    out
}

fn decode(dt: i32, b: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0; b.len() / elem(dt)];
    adapters::par_items(&mut out, 1, |i, o| {
        o[0] = if dt == DT_F32 { f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) } else { bf16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32() };
    });
    out
}

/// A tensor of a LoRA file as device bytes in `dt`.
fn bytes_of(f: &SafeTensors, name: &str, dt: i32) -> Result<Vec<u8>> {
    if dt == DT_F32 { Ok(f32_bytes(&f.f32_vec(name, false)?).to_vec()) } else { f.bf16_bytes(name) }
}

/// What is currently baked into a model.
#[derive(Default)]
pub struct State {
    stack: Vec<Lora>,
    /// model tensor -> what its patches were
    baked: HashMap<String, String>,
}

/// A model tensor, optionally only rows `.0 .. .0 + .1` of it.
#[derive(Clone, Debug, PartialEq)]
struct Target {
    tensor: String,
    rows: Option<(usize, usize)>,
}

/// ComfyUI's `string_to_seed` (a CRC-32).
pub fn string_to_seed(data: &str) -> u64 {
    let mut crc = 0xFFFF_FFFFu32;
    for c in data.chars() {
        crc ^= c as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    (crc ^ 0xFFFF_FFFF) as u64
}

/// `comfy.utils.krea2_to_diffusers`, without its output prefix: (diffusers name, model module).
fn krea2_to_diffusers(layers: usize) -> Vec<(String, String)> {
    const BLOCK: [(&str, &str); 9] = [("attn.to_q", "attn.wq"), ("attn.to_k", "attn.wk"), ("attn.to_v", "attn.wv"), ("attn.to_gate", "attn.gate"),
        ("attn.to_out.0", "attn.wo"), ("attn.to_out", "attn.wo"), ("ff.gate", "mlp.gate"), ("ff.up", "mlp.up"), ("ff.down", "mlp.down")];
    let mut v = Vec::new();
    let mut block = |to: String, from: String| {
        for (d, c) in BLOCK {
            v.push((format!("{to}.{d}"), format!("{from}.{c}")));
        }
    };
    for i in 0..layers {
        block(format!("transformer_blocks.{i}"), format!("blocks.{i}"));
    }
    for i in 0..2 {
        block(format!("text_fusion.layerwise_blocks.{i}"), format!("txtfusion.layerwise_blocks.{i}"));
        block(format!("text_fusion.refiner_blocks.{i}"), format!("txtfusion.refiner_blocks.{i}"));
    }
    for (d, c) in [("img_in", "first"), ("time_embed.linear_1", "tmlp.0"), ("time_embed.linear_2", "tmlp.2"), ("time_mod_proj", "tproj.1"),
        ("txt_in.linear_1", "txtmlp.1"), ("txt_in.linear_2", "txtmlp.3"), ("text_fusion.projector", "txtfusion.projector"),
        ("final_layer.linear", "last.linear")]
    {
        v.push((d.to_string(), c.to_string()));
    }
    v
}

/// `comfy.lora.model_lora_keys_clip` for an encoder of one Llama-like model: `lora_te_` is kohya's.
fn text_encoder_keys(comfy: &str, keys: &[(String, usize)]) -> HashMap<String, Target> {
    let mut map = HashMap::new();
    for (k, _) in keys {
        if let Some(base) = k.strip_suffix(".weight") {
            let to = Target { tensor: k.clone(), rows: None };
            map.insert(format!("text_encoders.{comfy}.transformer.model.{base}"), to.clone());
            map.insert(format!("text_encoders.transformer.model.{base}"), to.clone());
            map.insert(format!("lora_te_{}", base.replace('.', "_")), to);
        }
    }
    map
}

/// `comfy.lora.model_lora_keys_unet` for these models, or the text encoder's. `keys` are the model's
/// tensors with `.0` the row count of each.
fn key_map(arch: Arch, keys: &[(String, usize)]) -> HashMap<String, Target> {
    if let Arch::TextEncoder { comfy } = arch {
        return text_encoder_keys(comfy, keys);
    }
    let mut map = HashMap::new();
    let whole = |t: &str| Target { tensor: t.to_string(), rows: None };
    for (k, rows) in keys {
        match k.strip_suffix(".weight") {
            Some(base) => {
                map.insert(format!("lora_unet_{}", base.replace('.', "_")), whole(k));
                map.insert(format!("diffusion_model.{base}"), whole(k));
                if arch == Arch::Moss {
                    map.insert(base.to_string(), whole(k));
                    if let Some(proj) = base.strip_suffix(".qkv").filter(|b| b.ends_with(".proj")) {
                        let third = rows / 3;
                        for (i, p) in ["q", "k", "v"].iter().enumerate() {
                            let key = format!("{proj}.{p}");
                            let to = Target { tensor: k.clone(), rows: Some((i * third, third)) };
                            map.insert(format!("diffusion_model.{key}"), to.clone());
                            map.insert(format!("lora_unet_{}", key.replace('.', "_")), to.clone());
                            map.insert(key, to);
                        }
                    }
                }
            }
            None => {
                map.insert(format!("diffusion_model.{k}"), whole(k));
            }
        }
    }
    if arch == Arch::Krea2 {
        let layers = keys.iter().filter(|(k, _)| k.ends_with(".attn.gate.weight") && k.starts_with("blocks.")).count();
        for (d, c) in krea2_to_diffusers(layers) {
            let to = whole(&format!("{c}.weight"));
            map.insert(format!("diffusion_model.{d}"), to.clone());
            map.insert(format!("transformer.{d}"), to.clone());
            map.insert(format!("lycoris_{}", d.replace('.', "_")), to.clone());
            map.insert(d, to);
        }
    }
    map
}

/// What follows a module's name in the tensors of a LoRA file, for each thing that can patch it.
const SUFFIXES: [&str; 36] = [".lora_up.weight", "_lora.up.weight", ".lora_B.weight", ".lora.up.weight", ".lora_B", ".lora_linear_layer.up.weight",
    ".lora_B.default.weight", ".lora_down.weight", "_lora.down.weight", ".lora_A.weight", ".lora.down.weight", ".lora_A", ".lora_linear_layer.down.weight",
    ".lora_A.default.weight", ".hada_w1_a", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w2_a", ".a1.weight", ".oft_blocks", ".w_norm", ".b_norm", ".diff",
    ".diff_b", ".set_weight", ".alpha", ".dora_scale", ".reshape_weight", ".lora_mid.weight", ".lokr_w1_b", ".lokr_w2_b", ".hada_w1_b", ".a2.weight",
    ".b1.weight", ".b2.weight"];

fn module_names(st: &SafeTensors) -> Vec<&str> {
    let mut names: Vec<&str> = st.tensors.keys().filter_map(|k| SUFFIXES.iter().find_map(|sfx| k.strip_suffix(sfx))).collect();
    names.sort();
    names.dedup();
    names
}

/// One patch, with tensors named as the LoRA file names them. Everything but `Diff` and `Set` is a
/// `comfy.weight_adapter`; `dora` is its `dora_scale`.
#[derive(Clone, Debug)]
enum Op {
    /// `strength * alpha / rank * up @ down`
    Lora { up: String, down: String, mid: Option<String>, alpha: Option<f32>, dora: Option<String> },
    Loha { w1: (String, String), w2: (String, String), tucker: Option<(String, String)>, alpha: Option<f32>, dora: Option<String> },
    Lokr { w1: Factor, w2: Factor, alpha: Option<f32>, dora: Option<String> },
    Glora { a: (String, String), b: (String, String), alpha: Option<f32>, dora: Option<String> },
    Oft { blocks: String, alpha: Option<f32>, dora: Option<String> },
    Boft { blocks: String, rescale: Option<String>, alpha: Option<f32>, dora: Option<String> },
    Diff(String),
    Set(String),
    /// A MOSS Image control adapter's wider `video_embedder`: with no control image its extra inputs
    /// are zeros, so only the columns the model has count.
    SetLeft(String),
}

/// One side of a Kronecker product: a whole tensor, or two with an optional Tucker core.
#[derive(Clone, Debug)]
enum Factor {
    Full(String),
    Pair(String, String, Option<String>),
}

#[derive(Clone, Debug)]
struct Patch {
    file: usize,
    op: Op,
    strength: f32,
    rows: Option<(usize, usize)>,
}

fn bias_of(t: &Target) -> Target {
    Target { tensor: format!("{}.bias", t.tensor.strip_suffix(".weight").unwrap_or(&t.tensor)), rows: t.rows }
}

/// The adapter of module `x`, like the loop in `comfy.lora.load_lora`: the last that matches wins.
fn adapter_of(st: &SafeTensors, x: &str, alpha: Option<f32>, dora: Option<&String>, name: &str) -> Result<Option<Op>> {
    let n = |sfx: &str| format!("{x}{sfx}");
    let has = |sfx: &str| st.has(&n(sfx));
    let dora = dora.cloned();
    let mut op = None;
    const PAIRS: [(&str, &str); 7] = [(".lora_up.weight", ".lora_down.weight"), ("_lora.up.weight", "_lora.down.weight"), (".lora_B.weight", ".lora_A.weight"),
        (".lora.up.weight", ".lora.down.weight"), (".lora_B", ".lora_A"), (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
        (".lora_B.default.weight", ".lora_A.default.weight")];
    if let Some((up, down)) = PAIRS.iter().find(|(up, _)| has(up)) {
        if has(".reshape_weight") {
            return Err(Unsupported(format!("{name}: reshaped patches, which widen a model's inputs")).into());
        }
        let mid = (*up == ".lora_up.weight" && has(".lora_mid.weight")).then(|| n(".lora_mid.weight"));
        op = Some(Op::Lora { up: n(up), down: n(down), mid, alpha, dora: dora.clone() });
    }
    if has(".hada_w1_a") {
        let tucker = has(".hada_t1").then(|| (n(".hada_t1"), n(".hada_t2")));
        op = Some(Op::Loha { w1: (n(".hada_w1_a"), n(".hada_w1_b")), w2: (n(".hada_w2_a"), n(".hada_w2_b")), tucker, alpha, dora: dora.clone() });
    }
    if [".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w2_a"].iter().any(|s| has(s)) {
        let w1 = if has(".lokr_w1") { Factor::Full(n(".lokr_w1")) } else { Factor::Pair(n(".lokr_w1_a"), n(".lokr_w1_b"), None) };
        let w2 = if has(".lokr_w2") { Factor::Full(n(".lokr_w2")) } else { Factor::Pair(n(".lokr_w2_a"), n(".lokr_w2_b"), has(".lokr_t2").then(|| n(".lokr_t2"))) };
        op = Some(Op::Lokr { w1, w2, alpha, dora: dora.clone() });
    }
    if has(".a1.weight") {
        op = Some(Op::Glora { a: (n(".a1.weight"), n(".a2.weight")), b: (n(".b1.weight"), n(".b2.weight")), alpha, dora: dora.clone() });
    }
    if has(".oft_blocks") {
        let blocks = n(".oft_blocks");
        match st.info(&blocks)?.shape.len() {
            3 => op = Some(Op::Oft { blocks, alpha, dora }),
            4 => op = Some(Op::Boft { blocks, rescale: has(".rescale").then(|| n(".rescale")), alpha, dora }),
            _ => {}
        }
    }
    Ok(op)
}

/// Every patch of `file`, as (model tensor, patch), like `comfy.lora.load_lora` + `add_patches`: a
/// file has one patch per target, the last one found. Also returns how many modules matched nothing.
fn load_patches(st: &SafeTensors, file: usize, name: &str, strength: f32, arch: Arch, map: &HashMap<String, Target>, tensors: &HashMap<String, usize>)
    -> Result<(Vec<(String, Patch)>, usize)> {
    let text_encoder = matches!(arch, Arch::TextEncoder { .. });
    let mut found: BTreeMap<(String, Option<(usize, usize)>), Op> = BTreeMap::new();
    let mut put = |to: &Target, op: Op| {
        if tensors.contains_key(&to.tensor) {
            found.insert((to.tensor.clone(), to.rows), op);
        }
    };
    if arch == Arch::Moss && st.has("video_embedder.weight") && st.has("layers.0.proj.q.lora_A.weight") {
        for t in ["video_embedder.weight", "video_embedder.bias"] {
            if st.has(t) {
                put(&Target { tensor: t.to_string(), rows: None }, Op::SetLeft(t.to_string()));
            }
        }
    }
    let mut ignored = 0;
    for x in module_names(st) {
        if text_encoder && x.contains(".transformer.visual.") {
            return Err(Unsupported(format!("{name}: LoRAs that patch the vision tower")).into());
        }
        let Some(to) = map.get(x) else {
            ignored += ["lora_te", "text_encoder"].iter().all(|p| !x.starts_with(p)) as usize;
            continue;
        };
        let scalar = |k: String| st.has(&k).then(|| st.f32_vec(&k, false).map(|v| v[0])).transpose();
        let alpha = scalar(format!("{x}.alpha"))?;
        let dora = format!("{x}.dora_scale");
        let dora = st.has(&dora).then_some(dora);
        if let Some(op) = adapter_of(st, x, alpha, dora.as_ref(), name)? {
            put(to, op);
        }
        let n = |sfx: &str| format!("{x}{sfx}");
        if st.has(&n(".w_norm")) {
            put(to, Op::Diff(n(".w_norm")));
            if st.has(&n(".b_norm")) {
                put(&bias_of(to), Op::Diff(n(".b_norm")));
            }
        }
        if st.has(&n(".diff")) {
            put(to, Op::Diff(n(".diff")));
        }
        if st.has(&n(".diff_b")) {
            put(&bias_of(to), Op::Diff(n(".diff_b")));
        }
        if st.has(&n(".set_weight")) {
            put(to, Op::Set(n(".set_weight")));
        }
    }
    Ok((found.into_iter().map(|((t, rows), op)| (t, Patch { file, op, strength, rows })).collect(), ignored))
}

/// A tensor of a LoRA file, cast to `dt` as `calculate_weight` casts it.
fn tensor_of(f: &SafeTensors, name: &str, dt: i32) -> Result<Tensor> {
    Ok(Tensor::new(f.info(name)?.shape.clone(), f.f32_vec(name, dt == DT_BF16)?))
}

/// A matrix product on the device, in `dt`.
fn device_mm(a: &Tensor, b: &Tensor, dt: i32, st: &Stream) -> Result<Tensor> {
    let (rows, k, cols) = (a.shape[0], a.shape[1], b.shape[1]);
    let (x, y) = (upload(&encode(dt, &a.data), st)?, upload(&encode(dt, &b.data), st)?);
    let d = DevBuf::alloc_async(rows * cols * elem(dt), st)?;
    k!(gpu_gemm(x.ptr(), k as i64, y.ptr(), cols as i64, d.ptr(), cols as i64, null(), rows as i64, cols as i64, k as i64, 0, dt, dt, 1.0, 0.0, s(st)))?;
    let mut bytes = vec![0u8; rows * cols * elem(dt)];
    d.download(0, &mut bytes, st)?;
    Ok(Tensor::new(vec![rows, cols], decode(dt, &bytes)))
}

/// What an adapter does to a weight.
enum Change {
    /// values to add
    Add(Vec<f32>),
    /// the new weight
    Replace(Vec<f32>),
}

/// `calculate_weight` of an adapter. `weight` reads the weight as it stands, which only some need.
fn change_of(op: &Op, f: &SafeTensors, strength: f32, dt: i32, weight: &mut dyn FnMut() -> Result<Tensor>, st: &Stream) -> Result<Change> {
    let t = |n: &String| tensor_of(f, n, dt);
    let work = if dt == DT_F32 { Work::F32 } else { Work::BF16 };
    let mut mm = |a: &Tensor, b: &Tensor| device_mm(a, b, dt, st);
    let per_rank = |alpha: &Option<f32>, rank: usize| alpha.map_or(1.0, |a| a / rank as f32);
    let (diff, scale, dora) = match op {
        Op::Lora { up, down, mid, alpha, dora } => {
            let (u, d) = (t(up)?, t(down)?);
            let diff = match mid {
                Some(m) => adapters::lora_mid(&u, &d, &t(m)?, &mut mm)?,
                None => mm(&u.flat(), &d.flat())?,
            };
            (diff, per_rank(alpha, d.rows()), dora)
        }
        Op::Loha { w1, w2, tucker, alpha, dora } => {
            let (a, b) = ((t(&w1.0)?, t(&w1.1)?), (t(&w2.0)?, t(&w2.1)?));
            let cores = tucker.as_ref().map(|(x, y)| Ok::<_, anyhow::Error>((t(x)?, t(y)?))).transpose()?;
            (adapters::loha((&a.0, &a.1), (&b.0, &b.1), cores.as_ref().map(|(x, y)| (x, y)), work, &mut mm)?, per_rank(alpha, a.1.rows()), dora)
        }
        Op::Lokr { w1, w2, alpha, dora } => {
            let mut dim = None;
            let mut side = |w: &Factor, mm: &mut dyn FnMut(&Tensor, &Tensor) -> Result<Tensor>| -> Result<Tensor> {
                Ok(match w {
                    Factor::Full(n) => t(n)?,
                    Factor::Pair(a, b, core) => {
                        let (a, b) = (t(a)?, t(b)?);
                        dim = Some(b.rows());
                        match core {
                            Some(c) => adapters::tucker(&t(c)?, &b, &a, mm)?,
                            None => mm(&a.flat(), &b.flat())?,
                        }
                    }
                })
            };
            let (a, b) = (side(w1, &mut mm)?, side(w2, &mut mm)?);
            (adapters::kron(&a, &b, work), dim.map_or(1.0, |d| per_rank(alpha, d)), dora)
        }
        Op::Glora { a, b, alpha, dora } => {
            let w = weight()?;
            let (diff, rank) = adapters::glora(&t(&a.0)?, &t(&a.1)?, &t(&b.0)?, &t(&b.1)?, &w, work, &mut mm)?;
            (diff, per_rank(alpha, rank), dora)
        }
        Op::Oft { blocks, alpha, dora } => {
            let a = alpha.unwrap_or(0.0);
            let diff = adapters::oft(&t(blocks)?, a, strength, &weight()?, &mut mm)?;
            (diff, if dora.is_some() { a } else { 1.0 }, dora)
        }
        Op::Boft { blocks, rescale, alpha, dora } => {
            let a = alpha.ok_or_else(|| anyhow!("BOFT without an alpha"))?;
            let rescale = rescale.as_ref().map(t).transpose()?;
            let diff = adapters::boft(&t(blocks)?, rescale.as_ref(), a, strength, &weight()?)?;
            (diff, if dora.is_some() { a } else { 1.0 }, dora)
        }
        Op::Diff(_) | Op::Set(_) | Op::SetLeft(_) => unreachable!("handled by add_patches"),
    };
    Ok(match dora {
        Some(d) => Change::Replace(adapters::dora(&t(d)?, &weight()?, &diff.data, scale, strength, work)?),
        None => {
            let mut v = diff.data;
            adapters::par_items(&mut v, 1, |_, o| o[0] *= strength * scale);
            Change::Add(v)
        }
    })
}

/// Apply `patches` to the weight `w` of `per_row` x rows values, in `dt`.
fn add_patches(w: &DevBuf, dt: i32, rows: usize, per_row: usize, patches: &[Patch], files: &[SafeTensors], tensor: &str, st: &Stream) -> Result<()> {
    let size = elem(dt);
    for p in patches {
        let f = &files[p.file];
        let (start, n) = p.rows.unwrap_or((0, rows));
        let first = start * per_row * size;
        let dst = off(w.ptr(), first);
        let want = n * per_row;
        match &p.op {
            Op::Lora { up, down, alpha, mid: None, dora: None } if f.info(up)?.shape.len() == 2 && f.info(down)?.shape.len() == 2 => {
                let (ui, di) = (f.info(up)?, f.info(down)?);
                let ([nu, ru], [rd, kd]) = (ui.shape[..2].try_into()?, di.shape[..2].try_into()?);
                if ru != rd || nu * kd != want {
                    eprintln!("[lora] ERROR {tensor}: shape {nu}x{ru} @ {rd}x{kd} doesn't fit {want} values, not merged");
                    continue;
                }
                let a = upload(&bytes_of(f, up, dt)?, st)?;
                let b = upload(&bytes_of(f, down, dt)?, st)?;
                let d = DevBuf::alloc_async(want * size, st)?;
                k!(gpu_gemm(a.ptr(), ru as i64, b.ptr(), kd as i64, d.ptr(), kd as i64, null(), nu as i64, kd as i64, ru as i64, 0, dt, dt, 1.0, 0.0,
                    s(st)))?;
                let scale = p.strength as f64 * alpha.map_or(1.0, |v| v as f64 / rd as f64);
                k!(gpu_lora_add(dt, dst, d.ptr(), scale as f32, want as i64, s(st)))?;
            }
            Op::Diff(name) | Op::Set(name) => {
                if f.info(name)?.numel() != want {
                    eprintln!("[lora] WARNING SHAPE MISMATCH {tensor} WEIGHT NOT MERGED");
                    continue;
                }
                let d = upload(&bytes_of(f, name, dt)?, st)?;
                if matches!(p.op, Op::Set(_)) {
                    crate::cuda::memcpy_d2d(dst, d.ptr(), want * size, st)?;
                } else if p.strength != 0.0 {
                    k!(gpu_lora_add(dt, dst, d.ptr(), p.strength, want as i64, s(st)))?;
                }
            }
            Op::SetLeft(name) => {
                let t = tensor_of(f, name, dt)?;
                let cols = t.data.len() / t.rows().max(1);
                if t.rows() != n || cols < per_row {
                    eprintln!("[lora] WARNING SHAPE MISMATCH {tensor} WEIGHT NOT MERGED");
                    continue;
                }
                let left: Vec<f32> = t.data.chunks_exact(cols).flat_map(|r| r[..per_row].iter().copied()).collect();
                w.upload(first, &encode(dt, &left), st)?;
            }
            op => {
                let mut weight = || {
                    st.sync()?;
                    let mut b = vec![0u8; want * size];
                    w.download(first, &mut b, st)?;
                    Ok(Tensor::new(vec![n, per_row], decode(dt, &b)))
                };
                // like ComfyUI, a patch that doesn't fit is reported and left out
                match change_of(op, f, p.strength, dt, &mut weight, st) {
                    Ok(Change::Add(v)) | Ok(Change::Replace(v)) if v.len() != want => eprintln!("[lora] ERROR {tensor}: {} values don't fit {want}, not merged", v.len()),
                    Ok(Change::Add(v)) => {
                        let d = upload(&encode(dt, &v), st)?;
                        k!(gpu_lora_add(dt, dst, d.ptr(), 1.0, want as i64, s(st)))?;
                    }
                    Ok(Change::Replace(v)) => w.upload(first, &encode(dt, &v), st)?,
                    Err(e) => eprintln!("[lora] ERROR {tensor}: {e:#}, not merged"),
                }
            }
        }
    }
    Ok(())
}

fn upload(bytes: &[u8], st: &Stream) -> Result<DevBuf> {
    let d = DevBuf::alloc_async(bytes.len(), st)?;
    d.upload(0, bytes, st)?;
    Ok(d)
}

/// Parity tests (tools/lora_cmp.py): write a weight out if `$LORA_DUMP/<stem>.want` exists.
fn dump_baked(stem: &str, w: &DevBuf, q: &DevBuf, sc: &DevBuf, st: &Stream) -> Result<()> {
    let Ok(dir) = std::env::var("LORA_DUMP") else {
        return Ok(());
    };
    if Path::new(&dir).join(format!("{stem}.want")).exists() {
        for (name, b) in [("float", w), ("q", q), ("scale", sc)] {
            let mut v = vec![0u8; b.size()];
            b.download(0, &mut v, st)?;
            std::fs::write(format!("{dir}/{stem}.{name}.bin"), &v)?;
        }
    }
    Ok(())
}

struct Model<'a> {
    arch: Arch,
    st: &'a SafeTensors,
    prefix: &'a str,
    stores: Vec<Store<'a>>,
    /// model tensor -> (store, unit, place)
    places: HashMap<String, (usize, usize, Placed)>,
}

impl Model<'_> {
    fn write(&mut self, place: (usize, usize, Placed), src: &DevBuf, st: &Stream) -> Result<()> {
        let (store, unit, p) = place;
        self.stores[store].write(unit, p.off, src, p.len, st)
    }

    fn place(&self, tensor: &str) -> Result<(usize, usize, Placed)> {
        self.places.get(tensor).copied().ok_or_else(|| Unsupported(format!("a LoRA patching {tensor}, which isn't a stored weight here")).into())
    }

    fn file_name(&self, tensor: &str) -> String {
        format!("{}{tensor}", self.prefix)
    }

    /// Put a tensor back as the file holds it.
    fn restore(&mut self, tensor: &str, st: &Stream) -> Result<()> {
        let place = self.place(tensor)?;
        let name = self.file_name(tensor);
        let bytes = match place.2.kind {
            Stored::Raw => {
                if let Some(stem) = tensor.strip_suffix(".weight") {
                    let names = if self.st.has(&self.file_name(&format!("{stem}.weight_s_rel"))) { vec![format!("{stem}.weight_s_rel"), format!("{stem}.weight_s_channel")] } else { vec![format!("{stem}.weight_scale")] };
                    for scale in names {
                        let d = upload(self.st.bytes(&self.file_name(&scale))?, st)?;
                        let sp = self.place(&scale)?;
                        self.write(sp, &d, st)?;
                    }
                }
                self.st.bytes(&name)?.to_vec()
            }
            Stored::Bf16 => self.st.bf16_bytes(&name)?,
            Stored::F32(add) => f32_bytes(&self.st.f32_vec(&name, true)?.into_iter().map(|x| x + add).collect::<Vec<_>>()),
        };
        let d = upload(&bytes, st)?;
        self.write(place, &d, st)
    }

    fn bake(&mut self, tensor: &str, patches: &[Patch], files: &[SafeTensors], st: &Stream) -> Result<()> {
        let place = self.place(tensor)?;
        let name = self.file_name(tensor);
        let info = self.st.info(&name)?;
        let rows = info.shape.first().copied().unwrap_or(1);
        let per_row = info.numel() / rows.max(1);
        let dt = self.arch.dtype();
        if let Some(stem) = tensor.strip_suffix(".weight").filter(|s| self.st.has(&self.file_name(&format!("{s}.weight_s_rel")))) {
            return self.bake_w6(tensor, stem, patches, files, st);
        }
        let w = DevBuf::alloc_async(info.numel() * elem(dt), st)?;
        match place.2.kind {
            Stored::Raw => {
                let stem = tensor.strip_suffix(".weight").ok_or_else(|| anyhow!("{tensor}: not a weight"))?;
                let scale_name = format!("{stem}.weight_scale");
                let sp = self.place(&scale_name)?;
                let q = upload(self.st.bytes(&name)?, st)?;
                let sc = upload(self.st.bytes(&self.file_name(&scale_name))?, st)?;
                k!(gpu_convrot_dequant(dt, q.ptr(), sc.ptr(), w.ptr(), rows as i64, per_row as i32, s(st)))?;
                add_patches(&w, dt, rows, per_row, patches, files, tensor, st)?;
                let seed = string_to_seed(&self.arch.module(stem));
                k!(gpu_convrot_requant(dt, w.ptr(), q.ptr(), sc.ptr(), rows as i64, per_row as i32, seed, s(st)))?;
                dump_baked(stem, &w, &q, &sc, st)?;
                self.write(place, &q, st)?;
                self.write(sp, &sc, st)
            }
            Stored::Bf16 | Stored::F32(_) => {
                w.upload(0, &bytes_of(self.st, &name, dt)?, st)?;
                add_patches(&w, dt, rows, per_row, patches, files, tensor, st)?;
                let mut b = vec![0u8; info.numel() * elem(dt)];
                w.download(0, &mut b, st)?;
                let v = decode(dt, &b);
                let d = match place.2.kind {
                    Stored::F32(add) => upload(&f32_bytes(&v.iter().map(|x| x + add).collect::<Vec<_>>()), st)?,
                    _ => upload(&encode(DT_BF16, &v), st)?,
                };
                self.write(place, &d, st)
            }
        }
    }
}

impl Model<'_> {
    /// As [`Model::bake`] for w6a8 weights: 6-bit codes -> int8 -> bf16 -> patches -> 6-bit codes.
    fn bake_w6(&mut self, tensor: &str, stem: &str, patches: &[Patch], files: &[SafeTensors], st: &Stream) -> Result<()> {
        const G: usize = 32;
        let (rel, chan) = (format!("{stem}.weight_s_rel"), format!("{stem}.weight_s_channel"));
        let (place, rp, cp) = (self.place(tensor)?, self.place(&rel)?, self.place(&chan)?);
        let rows = self.st.info(&self.file_name(tensor))?.shape[0];
        let per_row = self.st.info(&self.file_name(&rel))?.shape[1] * G;
        let q = upload(self.st.bytes(&self.file_name(tensor))?, st)?;
        let r = upload(self.st.bytes(&self.file_name(&rel))?, st)?;
        let c = upload(self.st.bytes(&self.file_name(&chan))?, st)?;
        let i8s = DevBuf::alloc_async(rows * per_row, st)?;
        let w = DevBuf::alloc_async(rows * per_row * 2, st)?;
        k!(gpu_w6_decode(q.ptr(), r.ptr(), i8s.ptr(), rows as i64, per_row as i32, G as i32, s(st)))?;
        k!(gpu_convrot_dequant(DT_BF16, i8s.ptr(), c.ptr(), w.ptr(), rows as i64, per_row as i32, s(st)))?;
        add_patches(&w, DT_BF16, rows, per_row, patches, files, tensor, st)?;
        let seed = string_to_seed(&self.arch.module(stem));
        k!(gpu_w6_requant(w.ptr(), q.ptr(), r.ptr(), c.ptr(), rows as i64, per_row as i32, seed, s(st)))?;
        self.write(place, &q, st)?;
        self.write(rp, &r, st)?;
        self.write(cp, &c, st)
    }
}

/// The model's weight tensors under `prefix`, each with its row count, and ComfyUI's LoRA key names for them.
fn model_keys(arch: Arch, st: &SafeTensors, prefix: &str) -> (HashMap<String, usize>, HashMap<String, Target>) {
    let tensors: HashMap<String, usize> = st.tensors.iter()
        .filter_map(|(k, i)| Some((k.strip_prefix(prefix)?.to_string(), i.shape.first().copied().unwrap_or(1))))
        .filter(|(k, _)| !k.ends_with(".weight_scale") && !k.ends_with(".weight_s_rel") && !k.ends_with(".weight_s_channel") && !k.ends_with(".comfy_quant"))
        .collect();
    let keys: Vec<(String, usize)> = tensors.iter().map(|(k, r)| (k.clone(), *r)).collect();
    let map = key_map(arch, &keys);
    (tensors, map)
}

/// How many modules of `lora` name a tensor of `model` by ComfyUI's keys for it. ComfyUI skips a module
/// that names none, logging `lora key not loaded`, so a LoRA that matches nothing changes nothing there.
pub fn matched_modules(arch: Arch, model: &SafeTensors, prefix: &str, lora: &Path) -> Result<usize> {
    let (tensors, map) = model_keys(arch, model, prefix);
    let file = SafeTensors::open(lora)?;
    Ok(module_names(&file).into_iter().filter(|x| map.get(*x).is_some_and(|t| tensors.contains_key(&t.tensor))).count())
}

/// Make the model's weights hold `loras`, in order. Only tensors whose patches changed are touched.
/// Returns true if any weight changed, so whatever was derived from the weights must be dropped.
pub fn apply(arch: Arch, st: &SafeTensors, prefix: &str, mut stores: Vec<Store>, state: &mut State, loras: &[Lora], s0: &Stream) -> Result<bool> {
    if state.stack == loras {
        return Ok(false);
    }
    let t0 = Instant::now();
    let (tensors, map) = model_keys(arch, st, prefix);

    let files = loras.iter().map(|l| SafeTensors::open(&l.path)).collect::<Result<Vec<_>>>()?;
    let mut want: BTreeMap<String, Vec<Patch>> = BTreeMap::new();
    for (i, l) in loras.iter().enumerate() {
        let name = file_label(&l.path);
        let (patches, ignored) = load_patches(&files[i], i, &name, l.strength, arch, &map, &tensors).with_context(|| format!("reading {name}"))?;
        if ignored > 0 && !matches!(arch, Arch::TextEncoder { .. }) {
            eprintln!("[lora] {name}: {ignored} modules not loaded (no such key in this model)");
        }
        for (t, p) in patches {
            want.entry(t).or_default().push(p);
        }
    }
    let signature = |patches: &[Patch]| {
        patches.iter().map(|p| format!("{}|{}|{:?}|{:?}", loras[p.file].path.display(), p.strength, p.rows, p.op)).collect::<Vec<_>>().join(";")
    };

    let mut places = HashMap::new();
    for (i, store) in stores.iter().enumerate() {
        for (u, l) in store.layouts().iter().enumerate() {
            for (name, p) in l.tensors(st)? {
                if let Some(n) = name.strip_prefix(prefix) {
                    places.insert(n.to_string(), (i, u, p));
                }
            }
        }
    }

    // an encoder's last layers can go unused and unloaded, and then nothing can read their patches
    if matches!(arch, Arch::TextEncoder { .. }) {
        let layer = |t: &str| t.strip_prefix("layers.")?.split('.').next()?.parse::<usize>().ok();
        let last = places.keys().filter_map(|t| layer(t)).max().unwrap_or(0);
        want.retain(|t, _| layer(t).is_none_or(|i| i <= last));
    }

    s0.sync()?;
    for store in &stores {
        if let Store::Set(w) = store {
            w.quiesce()?;
        }
    }
    let restore: Vec<String> = state.baked.keys().filter(|k| !want.contains_key(*k)).cloned().collect();
    let mut m = Model { arch, st, prefix, stores: std::mem::take(&mut stores), places };
    for tensor in &restore {
        m.restore(tensor, s0)?;
        state.baked.remove(tensor);
    }
    let mut baked = 0;
    for (tensor, patches) in &want {
        let sig = signature(patches);
        if state.baked.get(tensor) == Some(&sig) {
            continue;
        }
        // a failure part way leaves the tensor unknown: forget the stack, keep what is baked for a retry
        state.stack = vec![Lora { path: PathBuf::new(), strength: f32::NAN }];
        m.bake(tensor, patches, &files, s0)?;
        s0.sync()?;
        state.baked.insert(tensor.clone(), sig);
        baked += 1;
    }
    s0.sync()?;
    for store in &mut m.stores {
        if let Store::Set(w) = store {
            w.restart_ring()?;
            w.evict_rewritten();
        }
    }
    state.stack = loras.to_vec();
    eprintln!("[lora] {} LoRAs: {baked} tensors baked, {} restored in {:.2}s", loras.len(), restore.len(), t0.elapsed().as_secs_f64());
    Ok(baked + restore.len() > 0)
}

fn file_label(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file of one-element bf16 tensors with these names.
    fn tiny_safetensors(name: &str, tensors: &[&str]) -> PathBuf {
        let header: Vec<String> = tensors.iter().enumerate()
            .map(|(i, t)| format!("\"{t}\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[{},{}]}}", i * 2, i * 2 + 2))
            .collect();
        let json = format!("{{{}}}", header.join(","));
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.resize(bytes.len() + tensors.len() * 2, 0);
        let path = std::env::temp_dir().join(format!("diffusion-lora-test-{}-{name}.safetensors", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn matched_modules_skips_another_models_lora() {
        let model = tiny_safetensors("seedvr", &["blocks.0.attn.proj_qkv.vid.weight", "vid_in.proj.weight"]);
        let other = tiny_safetensors("moss", &["diffusion_model.layers.0.proj.q.lora_A.weight", "diffusion_model.layers.0.proj.q.lora_B.weight"]);
        let own = tiny_safetensors("own", &["lora_unet_blocks_0_attn_proj_qkv_vid.lora_down.weight", "lora_unet_blocks_0_attn_proj_qkv_vid.lora_up.weight",
            "diffusion_model.vid_in.proj.lora_A.weight", "diffusion_model.vid_in.proj.lora_B.weight"]);
        let st = SafeTensors::open(&model).unwrap();
        assert_eq!(matched_modules(Arch::SeedVr2, &st, "", &other).unwrap(), 0);
        assert_eq!(matched_modules(Arch::SeedVr2, &st, "", &own).unwrap(), 2);
        drop(st);
        for p in [model, other, own] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn seed_is_comfys_crc() {
        assert_eq!(string_to_seed("123456789"), 0xCBF4_3926);
    }

    #[test]
    fn moss_splits_fused_qkv() {
        let m = key_map(Arch::Moss, &[("layers.0.proj.qkv.weight".to_string(), 9216)]);
        let to = Target { tensor: "layers.0.proj.qkv.weight".to_string(), rows: Some((3072, 3072)) };
        assert_eq!(m["diffusion_model.layers.0.proj.k"], to);
        assert_eq!(m["layers.0.proj.k"], to);
        assert_eq!(m["lora_unet_layers_0_proj_k"], to);
        assert_eq!(m["layers.0.proj.qkv"].rows, None);
    }

    #[test]
    fn krea_maps_diffusers_names() {
        let m = key_map(Arch::Krea2, &[("blocks.0.attn.gate.weight".to_string(), 6144), ("txtfusion.projector.weight".to_string(), 1)]);
        assert_eq!(m["transformer.transformer_blocks.0.attn.to_gate"].tensor, "blocks.0.attn.gate.weight");
        assert_eq!(m["diffusion_model.blocks.0.attn.gate"].tensor, "blocks.0.attn.gate.weight");
        assert_eq!(m["lycoris_text_fusion_projector"].tensor, "txtfusion.projector.weight");
    }

    #[test]
    fn qwen21_takes_diffusers_names_as_stored() {
        let m = key_map(Arch::Qwen21, &[("transformer_blocks.3.attn.to_out.0.weight".to_string(), 4096), ("transformer_blocks.3.img_mlp.gate_up.weight".to_string(), 24576)]);
        assert_eq!(m["diffusion_model.transformer_blocks.3.attn.to_out.0"].tensor, "transformer_blocks.3.attn.to_out.0.weight");
        assert_eq!(m["lora_unet_transformer_blocks_3_img_mlp_gate_up"].rows, None);
        assert_eq!(Arch::Qwen21.module("transformer_blocks.3.attn.to_out.0"), "diffusion_model.transformer_blocks.3.attn.to_out.0");
    }

    #[test]
    fn anima_uses_module_paths() {
        let m = key_map(Arch::Anima, &[("blocks.0.cross_attn.k_proj.weight".to_string(), 2048)]);
        assert_eq!(m["lora_unet_blocks_0_cross_attn_k_proj"].tensor, "blocks.0.cross_attn.k_proj.weight");
        assert_eq!(m["diffusion_model.blocks.0.cross_attn.k_proj"].tensor, "blocks.0.cross_attn.k_proj.weight");
        assert!(!m.contains_key("blocks.0.cross_attn.k_proj"));
    }

    #[test]
    fn text_encoders_use_comfys_three_names() {
        let m = key_map(Arch::TextEncoder { comfy: "qwen3_06b" }, &[("layers.0.self_attn.q_proj.weight".to_string(), 4096)]);
        for k in ["lora_te_layers_0_self_attn_q_proj", "text_encoders.transformer.model.layers.0.self_attn.q_proj",
            "text_encoders.qwen3_06b.transformer.model.layers.0.self_attn.q_proj"] {
            assert_eq!(m[k].tensor, "layers.0.self_attn.q_proj.weight");
        }
        assert_eq!(Arch::TextEncoder { comfy: "qwen3_06b" }.module("layers.0"), "qwen3_06b.transformer.model.layers.0");
    }
}
