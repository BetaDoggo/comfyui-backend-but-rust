//! LoRAs, baked into the weights the way ComfyUI patches a quantized model: each patched int8 weight
//! is dequantized to bf16, the patches are added in bf16, and it is requantized with fresh row scales
//! and stochastic rounding seeded by the module's name. Sampling then runs the normal int8 kernels.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_BF16, PM};
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

/// `comfy.lora.model_lora_keys_unet` for these models. `keys` are the model's tensors with
/// `.0` the row count of each.
fn key_map(arch: Arch, keys: &[(String, usize)]) -> HashMap<String, Target> {
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

/// The parts of one patched module in a LoRA file.
#[derive(Default, Debug)]
struct Parts {
    up: Option<String>,
    down: Option<String>,
    mid: bool,
    alpha: Option<String>,
    dora: bool,
    reshape: bool,
    diff: Option<String>,
    diff_b: Option<String>,
    set: Option<String>,
    w_norm: Option<String>,
    b_norm: Option<String>,
}

/// Formats that have no adapter here yet (LoHa, LoKr, GLoRA, OFT/BOFT).
const OTHER_ADAPTERS: [&str; 7] = [".hada_w1_a", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".a1.weight", ".oft_blocks", ".oft_diag"];

fn parts_of(st: &SafeTensors, name: &str) -> Result<HashMap<String, Parts>> {
    let mut parts: HashMap<String, Parts> = HashMap::new();
    for key in st.tensors.keys() {
        if OTHER_ADAPTERS.iter().any(|a| key.ends_with(a)) {
            return Err(Unsupported(format!("{name}: LoHa, LoKr, GLoRA and OFT patches")).into());
        }
        const ROLES: [(&str, usize); 22] = [(".lora_up.weight", 0), ("_lora.up.weight", 0), (".lora_B.weight", 0), (".lora.up.weight", 0), (".lora_B", 0),
            (".lora_linear_layer.up.weight", 0), (".lora_B.default.weight", 0), (".lora_down.weight", 1), ("_lora.down.weight", 1), (".lora_A.weight", 1),
            (".lora.down.weight", 1), (".lora_A", 1), (".lora_linear_layer.down.weight", 1), (".lora_A.default.weight", 1), (".lora_mid.weight", 2),
            (".alpha", 3), (".dora_scale", 4), (".reshape_weight", 5), (".diff", 6), (".diff_b", 7), (".set_weight", 8), (".w_norm", 9)];
        let found = ROLES.iter().find_map(|(sfx, r)| key.strip_suffix(sfx).map(|x| (x, *r)));
        let (x, role) = match (found, key.strip_suffix(".b_norm")) {
            (Some(f), _) => f,
            (None, Some(x)) => (x, 10),
            _ => continue,
        };
        let p = parts.entry(x.to_string()).or_default();
        let k = Some(key.clone());
        match role {
            0 => p.up = k,
            1 => p.down = k,
            2 => p.mid = true,
            3 => p.alpha = k,
            4 => p.dora = true,
            5 => p.reshape = true,
            6 => p.diff = k,
            7 => p.diff_b = k,
            8 => p.set = k,
            9 => p.w_norm = k,
            _ => p.b_norm = k,
        }
    }
    Ok(parts)
}

#[derive(Clone, Debug)]
enum Op {
    /// `strength * alpha / rank * up @ down`
    Lora { up: String, down: String, alpha: Option<f32> },
    Diff(String),
    Set(String),
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

/// Every patch of `file`, as (model tensor, patch), like `comfy.lora.load_lora` + `add_patches`.
fn load_patches(st: &SafeTensors, file: usize, name: &str, strength: f32, map: &HashMap<String, Target>, tensors: &HashMap<String, usize>)
    -> Result<(Vec<(String, Patch)>, usize)> {
    if st.has("video_embedder.weight") && st.has("layers.0.proj.q.lora_A.weight") {
        return Err(Unsupported(format!("{name}: MOSS Image control LoRAs")).into());
    }
    let parts = parts_of(st, name)?;
    let mut out = Vec::new();
    let mut ignored = 0;
    let mut names: Vec<_> = parts.keys().collect();
    names.sort();
    for x in names {
        let (p, Some(to)) = (&parts[x], map.get(x)) else {
            ignored += 1;
            continue;
        };
        if p.mid || p.dora || p.reshape {
            return Err(Unsupported(format!("{name}: LoCon mid weights, DoRA and reshaped patches")).into());
        }
        let mut push = |to: &Target, op: Op| {
            if tensors.contains_key(&to.tensor) {
                out.push((to.tensor.clone(), Patch { file, op, strength, rows: to.rows }));
            }
        };
        if let (Some(up), Some(down)) = (&p.up, &p.down) {
            let alpha = p.alpha.as_ref().map(|a| st.f32_vec(a, false).map(|v| v[0])).transpose()?;
            push(to, Op::Lora { up: up.clone(), down: down.clone(), alpha });
        }
        if let Some(d) = &p.w_norm {
            push(to, Op::Diff(d.clone()));
        }
        if let Some(d) = &p.b_norm {
            push(&bias_of(to), Op::Diff(d.clone()));
        }
        if let Some(d) = &p.diff {
            push(to, Op::Diff(d.clone()));
        }
        if let Some(d) = &p.diff_b {
            push(&bias_of(to), Op::Diff(d.clone()));
        }
        if let Some(d) = &p.set {
            push(to, Op::Set(d.clone()));
        }
    }
    Ok((out, ignored))
}

/// Apply `patches` to the bf16 weight `w` of `elems_per_row` x rows values.
fn add_patches(w: PM, rows: usize, per_row: usize, patches: &[Patch], files: &[SafeTensors], tensor: &str, st: &Stream) -> Result<()> {
    let (bf, dt) = (DT_BF16, DT_BF16);
    for p in patches {
        let f = &files[p.file];
        let (start, n) = p.rows.unwrap_or((0, rows));
        let dst = off(w, start * per_row * 2);
        let want = n * per_row;
        match &p.op {
            Op::Lora { up, down, alpha } => {
                let (ui, di) = (f.info(up)?, f.info(down)?);
                let [nu, ru] = ui.shape[..] else {
                    eprintln!("[lora] {tensor}: skipping non-linear patch");
                    continue;
                };
                let [rd, kd] = di.shape[..] else {
                    eprintln!("[lora] {tensor}: skipping non-linear patch");
                    continue;
                };
                if ru != rd || nu * kd != want {
                    eprintln!("[lora] ERROR {tensor}: shape {nu}x{ru} @ {rd}x{kd} doesn't fit {want} values, not merged");
                    continue;
                }
                let a = DevBuf::alloc_async(nu * ru * 2, st)?;
                let b = DevBuf::alloc_async(rd * kd * 2, st)?;
                let d = DevBuf::alloc_async(want * 2, st)?;
                a.upload(0, &f.bf16_bytes(up)?, st)?;
                b.upload(0, &f.bf16_bytes(down)?, st)?;
                k!(gpu_gemm(a.ptr(), ru as i64, b.ptr(), kd as i64, d.ptr(), kd as i64, null(), nu as i64, kd as i64, ru as i64, 0, bf, dt, 1.0, 0.0,
                    s(st)))?;
                let scale = p.strength as f64 * alpha.map_or(1.0, |v| v as f64 / rd as f64);
                k!(gpu_lora_add_bf16(dst, d.ptr(), scale as f32, want as i64, s(st)))?;
            }
            Op::Diff(name) | Op::Set(name) => {
                if f.info(name)?.numel() != want {
                    eprintln!("[lora] WARNING SHAPE MISMATCH {tensor} WEIGHT NOT MERGED");
                    continue;
                }
                let d = DevBuf::alloc_async(want * 2, st)?;
                d.upload(0, &f.bf16_bytes(name)?, st)?;
                if matches!(p.op, Op::Set(_)) {
                    crate::cuda::memcpy_d2d(dst, d.ptr(), want * 2, st)?;
                } else if p.strength != 0.0 {
                    k!(gpu_lora_add_bf16(dst, d.ptr(), p.strength, want as i64, s(st)))?;
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
                    let scale = format!("{stem}.weight_scale");
                    let d = upload(self.st.bytes(&self.file_name(&scale))?, st)?;
                    let sp = self.place(&scale)?;
                    self.write(sp, &d, st)?;
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
        let w = DevBuf::alloc_async(info.numel() * 2, st)?;
        match place.2.kind {
            Stored::Raw => {
                let stem = tensor.strip_suffix(".weight").ok_or_else(|| anyhow!("{tensor}: not a weight"))?;
                let scale_name = format!("{stem}.weight_scale");
                let sp = self.place(&scale_name)?;
                let q = upload(self.st.bytes(&name)?, st)?;
                let sc = upload(self.st.bytes(&self.file_name(&scale_name))?, st)?;
                k!(gpu_convrot_dequant_bf16(q.ptr(), sc.ptr(), w.ptr(), rows as i64, per_row as i32, s(st)))?;
                add_patches(w.ptr(), rows, per_row, patches, files, tensor, st)?;
                let seed = string_to_seed(&format!("diffusion_model.{stem}"));
                k!(gpu_convrot_requant_bf16(w.ptr(), q.ptr(), sc.ptr(), rows as i64, per_row as i32, seed, s(st)))?;
                dump_baked(stem, &w, &q, &sc, st)?;
                self.write(place, &q, st)?;
                self.write(sp, &sc, st)
            }
            Stored::Bf16 => {
                w.upload(0, &self.st.bf16_bytes(&name)?, st)?;
                add_patches(w.ptr(), rows, per_row, patches, files, tensor, st)?;
                self.write(place, &w, st)
            }
            Stored::F32(add) => {
                w.upload(0, &self.st.bf16_bytes(&name)?, st)?;
                add_patches(w.ptr(), rows, per_row, patches, files, tensor, st)?;
                let mut b = vec![0u8; info.numel() * 2];
                w.download(0, &mut b, st)?;
                let v: Vec<f32> = b.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32() + add).collect();
                let d = upload(&f32_bytes(&v), st)?;
                self.write(place, &d, st)
            }
        }
    }
}

/// Make the model's weights hold `loras`, in order. Only tensors whose patches changed are touched.
/// Returns true if any weight changed, so whatever was derived from the weights must be dropped.
pub fn apply(arch: Arch, st: &SafeTensors, prefix: &str, mut stores: Vec<Store>, state: &mut State, loras: &[Lora], s0: &Stream) -> Result<bool> {
    if state.stack == loras {
        return Ok(false);
    }
    let t0 = Instant::now();
    let tensors: HashMap<String, usize> = st.tensors.iter()
        .filter_map(|(k, i)| Some((k.strip_prefix(prefix)?.to_string(), i.shape.first().copied().unwrap_or(1))))
        .filter(|(k, _)| !k.ends_with(".weight_scale") && !k.ends_with(".comfy_quant"))
        .collect();
    let keys: Vec<(String, usize)> = tensors.iter().map(|(k, r)| (k.clone(), *r)).collect();
    let map = key_map(arch, &keys);

    let files = loras.iter().map(|l| SafeTensors::open(&l.path)).collect::<Result<Vec<_>>>()?;
    let mut want: BTreeMap<String, Vec<Patch>> = BTreeMap::new();
    for (i, l) in loras.iter().enumerate() {
        let name = file_label(&l.path);
        let (patches, ignored) = load_patches(&files[i], i, &name, l.strength, &map, &tensors).with_context(|| format!("reading {name}"))?;
        if ignored > 0 {
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

    s0.sync()?;
    for store in &stores {
        if let Store::Set(w) = store {
            w.quiesce()?;
        }
    }
    let restore: Vec<String> = state.baked.keys().filter(|k| !want.contains_key(*k)).cloned().collect();
    let mut m = Model { st, prefix, stores: std::mem::take(&mut stores), places };
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
        }
    }
    state.stack = loras.to_vec();
    eprintln!("[lora] {} LoRAs: {baked} tensors baked, {} restored in {:.2}s", loras.len(), restore.len(), t0.elapsed().as_secs_f64());
    Ok(baked + restore.len() > 0)
}

fn file_label(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

/// True if a LoRA file also patches a text encoder (`comfy.lora.model_lora_keys_clip`'s key names).
pub fn has_text_encoder_keys(keys: impl IntoIterator<Item = impl AsRef<str>>) -> bool {
    keys.into_iter().any(|k| ["lora_te", "lora_prior_te", "text_encoders.", "text_encoder."].iter().any(|p| k.as_ref().starts_with(p)))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn anima_uses_module_paths() {
        let m = key_map(Arch::Anima, &[("blocks.0.cross_attn.k_proj.weight".to_string(), 2048)]);
        assert_eq!(m["lora_unet_blocks_0_cross_attn_k_proj"].tensor, "blocks.0.cross_attn.k_proj.weight");
        assert_eq!(m["diffusion_model.blocks.0.cross_attn.k_proj"].tensor, "blocks.0.cross_attn.k_proj.weight");
        assert!(!m.contains_key("blocks.0.cross_attn.k_proj"));
    }

    #[test]
    fn text_encoder_keys_are_recognised() {
        assert!(has_text_encoder_keys(["lora_te_layers_0_self_attn_q_proj.lora_up.weight"]));
        assert!(!has_text_encoder_keys(["lora_unet_blocks_0.alpha"]));
    }
}
