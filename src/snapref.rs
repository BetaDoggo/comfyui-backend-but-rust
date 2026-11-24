//! SnapRef (https://github.com/ArcAnima/SnapRef), a reference adapter for Anima, as its ComfyUI node runs it.
//!
//! A reference picture runs once through the diffusion model as its own stream: sigma 0, the adapter's "ref" LoRA, a
//! mode embedding added to the timestep embedding, and RoPE frame `ref_t` centred on the target. Every block's
//! self-attention keys and values are kept ([`RefKv`]). Each target evaluation then runs with the "tgt" LoRA, and the
//! evaluations that read the reference prepend those keys and values in self-attention, optionally with a bias on
//! their logits. The model code that does this is in [`crate::anima`].
//!
//! The LoRAs stay apart from the model's weights, which are int8: each targeted projection adds
//! `up(down(x)) * alpha / rank` to its output, from its bf16 input, as the node's forward hooks do.
//!
//! Adapters from v2s on may carry mode-gated branches (`anima_ref_kv.branch.<branch>.<module>.down / up / alpha`):
//! extra LoRAs of one stream that run only in the modes they list, added after that stream's LoRA. A reference-stream
//! branch runs while the reference is encoded, a target-stream one only in target evaluations that read the reference.
//! Style mode exists only through such a branch.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::PM;
use crate::st::{Dt, SafeTensors};
use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The weight file format the node reads (`anima_ref_kv/1`).
const FORMAT: &str = "anima_ref_kv/1";
const PREFIX: &str = "anima_ref_kv.";
/// Anima-2.9B's 12 blocks inserted into the 28-block base, shifting the original ones.
const INSERTED_29B: [usize; 12] = [2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36];

/// Which stream a LoRA belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoraStream {
    /// The reference's own pass.
    Ref,
    /// Every target evaluation.
    Tgt,
}

/// The projections the adapter can target, as the model's code groups them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Module {
    /// `self_attn.{q,k,v}_proj` together, as the model's fused projection.
    SelfQkv,
    SelfOut,
    CrossQ,
    CrossOut,
    Mlp1,
    Mlp2,
}

impl Module {
    fn parse(name: &str) -> Option<(Self, usize)> {
        Some(match name {
            "self_attn.q_proj" => (Module::SelfQkv, 0),
            "self_attn.k_proj" => (Module::SelfQkv, 1),
            "self_attn.v_proj" => (Module::SelfQkv, 2),
            "self_attn.output_proj" => (Module::SelfOut, 0),
            "cross_attn.q_proj" => (Module::CrossQ, 0),
            "cross_attn.output_proj" => (Module::CrossOut, 0),
            "mlp.layer1" => (Module::Mlp1, 0),
            "mlp.layer2" => (Module::Mlp2, 0),
            _ => return None,
        })
    }

    fn parts(self) -> usize {
        if self == Module::SelfQkv { 3 } else { 1 }
    }
}

/// One projection's LoRA in [`Adapter`]'s buffer: `parts` down matrices stacked ([parts * rank, k]) and one up matrix
/// per part ([n, rank]).
#[derive(Clone, Debug)]
pub struct Mat {
    pub down: usize,
    pub up: [usize; 3],
    pub parts: usize,
    pub rank: usize,
    /// Inputs and outputs (per part).
    pub k: usize,
    pub n: usize,
    /// alpha / rank
    pub scale: f32,
}

/// A loaded SnapRef weight file, in VRAM.
pub struct Adapter {
    pub path: PathBuf,
    /// RoPE frame of the reference.
    pub ref_t: f32,
    pub modes: Vec<String>,
    /// bf16 [modes][2048]
    mode_embed: Option<DevBuf>,
    buf: DevBuf,
    mats: HashMap<(LoraStream, usize, Module), Mat>,
    /// The projections a mode's branches touch, by mode: the stream's LoRA (if any) and each branch's, merged into one
    /// LoRA of their summed rank (see [`merge`]). Projections no branch touches keep their entry in `mats`.
    branched: HashMap<(String, LoraStream, usize, Module), Mat>,
    /// Modes whose branches target something the engine can't apply, and why: refused when used.
    broken: Vec<(String, String)>,
    /// Largest rank, merged ones included, for scratch sizes.
    pub max_rank: usize,
}

/// One LoRA on a projection: the key root of each part (none for a part it leaves alone) and its matrices' suffixes.
#[derive(Clone)]
struct Lora {
    roots: Vec<Option<String>>,
    down: &'static str,
    up: &'static str,
}

/// A branch's metadata entry (`branches.<name>`), as the node reads it.
struct Branch {
    name: String,
    stream: LoraStream,
    modes: Vec<String>,
}

/// The modes that exist only through a branch: `cs` and `char` work with every adapter.
const BRANCH_MODES: [&str; 1] = ["style"];

/// The JSON metadata of a safetensors file.
fn metadata(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n) as usize;
    if n > 100 << 20 {
        bail!("{}: not a safetensors file", path.display());
    }
    let mut header = vec![0u8; n];
    f.read_exact(&mut header)?;
    let h: serde_json::Value = serde_json::from_slice(&header)?;
    Ok(h.get("__metadata__").and_then(|m| m.as_object()).cloned().unwrap_or_default())
}

/// `float8_e4m3fn` to f32.
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xf, b & 0x7);
    if e == 0xf && m == 0x7 {
        return f32::NAN;
    }
    sign * if e == 0 { m as f32 / 8.0 * 2f32.powi(-6) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) }
}

/// A LoRA matrix, rounded to bf16 as the node holds it, and its rows and columns. FP8 files carry a float32 scale per
/// row (`<key>_scale`), as the node dequantizes them.
fn matrix(st: &SafeTensors, key: &str) -> Result<(Vec<f32>, usize, usize)> {
    let i = st.info(key)?;
    let &[rows, cols] = i.shape.as_slice() else { bail!("{key}: expected a matrix, got shape {:?}", i.shape) };
    let scale_key = format!("{key}_scale");
    let values: Vec<f32> = match i.dtype {
        Dt::F8 => st.bytes(key)?.iter().map(|&b| e4m3(b)).collect(),
        Dt::BF16 => st.bytes(key)?.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dt::F16 => st.bytes(key)?.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dt::F32 => st.f32_vec(key, false)?,
        d => bail!("{key}: unsupported dtype {d:?}"),
    };
    let values = if st.has(&scale_key) {
        let s = st.f32_vec(&scale_key, false)?;
        if s.len() != rows {
            bail!("{scale_key}: {} values for {rows} rows", s.len());
        }
        values.chunks_exact(cols).zip(&s).flat_map(|(row, f)| row.iter().map(move |v| v * f)).collect()
    } else if i.dtype == Dt::F8 {
        bail!("{key}: FP8 without {scale_key}");
    } else {
        values
    };
    Ok((values.iter().map(|v| bf16::from_f32(*v).to_f32()).collect(), rows, cols))
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect()
}

/// Several LoRAs on one projection as one: their down matrices stacked (rank r1 + r2 + ...) and their up matrices side
/// by side, each later one's up scaled by its scale over the first's, which stays the GEMM's. The sum of their deltas,
/// as the node adds them one after another, in one bf16 rounding instead of one per LoRA. A part a LoRA leaves alone
/// gets zeros. Returns each part's down [rank, k] and up [n, rank], the summed rank, k, n and the scale.
#[allow(clippy::type_complexity)]
fn merge(st: &SafeTensors, loras: &[Lora], parts: usize) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, usize, usize, usize, f32)> {
    // per LoRA: per part (down, up) or none, its rank and scale
    let mut loaded = Vec::new();
    let (mut k, mut n) = (0, 0);
    for l in loras {
        let (mut rank, mut scale) = (0, 0.0);
        let mut mats = Vec::new();
        for root in &l.roots {
            let Some(root) = root else {
                mats.push(None);
                continue;
            };
            let (down, r, kk) = matrix(st, &format!("{root}.{}", l.down))?;
            let (up, nn, r2) = matrix(st, &format!("{root}.{}", l.up))?;
            let alpha = st.f32_vec(&format!("{root}.alpha"), false)?.first().copied().unwrap_or(r as f32);
            if r2 != r || (rank > 0 && r != rank) || (k > 0 && (kk, nn) != (k, n)) {
                bail!("{root}: mismatched LoRA shapes");
            }
            (rank, k, n, scale) = (r, kk, nn, alpha / r as f32);
            mats.push(Some((down, up)));
        }
        loaded.push((mats, rank, scale));
    }
    let total: usize = loaded.iter().map(|l| l.1).sum();
    let first = loaded[0].2;
    let (mut downs, mut ups) = (Vec::new(), Vec::new());
    for p in 0..parts {
        let mut down = Vec::with_capacity(total * k);
        let mut up = vec![0.0; n * total];
        let mut at = 0;
        for (mats, rank, scale) in &loaded {
            match &mats[p] {
                Some((d, u)) => {
                    down.extend_from_slice(d);
                    let f = scale / first;
                    for row in 0..n {
                        for c in 0..*rank {
                            up[row * total + at + c] = u[row * rank + c] * f;
                        }
                    }
                }
                None => down.resize(down.len() + rank * k, 0.0),
            }
            at += rank;
        }
        downs.push(down);
        ups.push(up);
    }
    Ok((downs, ups, total, k, n, first))
}

/// The node's `_read_branches`: the branches its metadata lists, in their order. A stream other than ref or tgt, or a
/// combine other than add, is refused as the node refuses it.
fn branches(meta: &serde_json::Value, path: &Path) -> Result<Vec<Branch>> {
    let Some(specs) = meta["branches"].as_object() else { return Ok(Vec::new()) };
    specs.iter().map(|(name, spec)| {
        let combine = spec["combine"].as_str().unwrap_or("add");
        let stream = match spec["stream"].as_str() {
            Some("ref") if combine == "add" => LoraStream::Ref,
            Some("tgt") if combine == "add" => LoraStream::Tgt,
            s => bail!("{}: branch {name} (stream {s:?}, combine {combine}) is not supported (stream ref or tgt, combine add)", path.display()),
        };
        let modes = spec["modes"].as_array().map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect()).unwrap_or_default();
        Ok(Branch { name: name.clone(), stream, modes })
    }).collect()
}

/// `<prefix>blocks.<i>.<module>` of a key root: the block and the projection and part.
fn target(key: &str, name: &str) -> Result<(usize, Module, usize)> {
    let Some((index, module)) = name.strip_prefix("blocks.").and_then(|n| n.split_once('.')) else {
        bail!("{key}: SnapRef LoRAs outside the blocks are not supported");
    };
    let index: usize = index.parse().map_err(|_| anyhow!("{key}: bad block index"))?;
    let (module, part) = Module::parse(module).ok_or_else(|| anyhow!("{key}: unsupported target {module}"))?;
    Ok((index, module, part))
}

/// The node's `_adapter_block_map`: model block of each adapter block.
fn block_map(indices: &[usize], model_blocks: usize) -> Result<Vec<usize>> {
    let max = *indices.iter().max().ok_or_else(|| anyhow!("the adapter has no diffusion block weights"))?;
    if max + 1 == model_blocks {
        return Ok((0..model_blocks).collect());
    }
    if model_blocks == 40 && indices.len() == 28 && max == 27 {
        return Ok((0..40).filter(|i| !INSERTED_29B.contains(i)).collect());
    }
    bail!("the adapter covers {} blocks but the model has {model_blocks}; only native layouts and the 28-to-40-block Anima-2.9B expansion are supported", max + 1)
}

impl Adapter {
    /// Load the weights at `path` for a model of `model_blocks` blocks of width `width`.
    pub fn load(path: &Path, model_blocks: usize, width: usize, st0: &Stream) -> Result<Self> {
        let md = metadata(path)?;
        let meta = md.get("snapref").or_else(|| md.get("anima_ref_kv")).and_then(|v| v.as_str()).unwrap_or("{}");
        let meta: serde_json::Value = serde_json::from_str(meta).with_context(|| format!("{}: bad SnapRef metadata", path.display()))?;
        if meta["format"].as_str() != Some(FORMAT) {
            bail!("{} is not a SnapRef ({FORMAT}) adapter", path.display());
        }
        let ref_t = meta["ref_t"].as_f64().unwrap_or(10.0) as f32;
        let modes: Vec<String> = match meta["modes"].as_array() {
            Some(a) => a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect(),
            None => vec!["cs".into()],
        };
        let st = SafeTensors::open(path)?;
        let branch_specs = branches(&meta, path)?;

        // anima_ref_kv.<stream>.blocks.<i>.<module>.lora_down, and the branches' anima_ref_kv.branch.<branch>.<module>.down
        type Found = HashMap<(LoraStream, usize, Module), Vec<Option<String>>>;
        let mut found: Found = HashMap::new();
        let mut branch_found: HashMap<String, Found> = HashMap::new();
        // branches with a target the engine can't apply: refused when their mode is used, so the others still load
        let mut unsupported: HashMap<String, String> = HashMap::new();
        let mut indices = Vec::new();
        let branch_prefix = format!("{PREFIX}branch.");
        for key in st.tensors.keys() {
            if let Some(rest) = key.strip_prefix(&branch_prefix) {
                let Some(root) = key.strip_suffix(".down") else { continue };
                let (branch, name) = rest[..root.len() - branch_prefix.len()].split_once('.').ok_or_else(|| anyhow!("{key}: unexpected key"))?;
                let Some(spec) = branch_specs.iter().find(|b| b.name == branch) else {
                    bail!("{}: branch {branch} has weights but no metadata (branches.{branch})", path.display());
                };
                match target(key, name) {
                    Ok((index, module, part)) => {
                        indices.push(index);
                        branch_found.entry(branch.to_string()).or_default()
                            .entry((spec.stream, index, module)).or_insert_with(|| vec![None; module.parts()])[part] = Some(root.to_string());
                    }
                    Err(e) => {
                        unsupported.entry(branch.to_string()).or_insert(e.to_string());
                    }
                }
                continue;
            }
            let Some(root) = key.strip_suffix(".lora_down") else { continue };
            let rest = root.strip_prefix(PREFIX).ok_or_else(|| anyhow!("{key}: unexpected key"))?;
            let (stream, name) = rest.split_once('.').ok_or_else(|| anyhow!("{key}: unexpected key"))?;
            let stream = match stream {
                "ref" => LoraStream::Ref,
                "tgt" => LoraStream::Tgt,
                s => bail!("{key}: unknown stream {s}"),
            };
            let (index, module, part) = target(key, name)?;
            indices.push(index);
            found.entry((stream, index, module)).or_insert_with(|| vec![None; module.parts()])[part] = Some(root.to_string());
        }
        indices.sort_unstable();
        indices.dedup();
        let map = block_map(&indices, model_blocks)?;

        let mut host: Vec<u8> = Vec::new();
        let mut push = |values: &[f32]| {
            let at = host.len().next_multiple_of(256);
            host.resize(at, 0);
            host.extend_from_slice(&bf16_bytes(values));
            at
        };
        let mut max_rank = 0;
        let mut upload = |loras: &[Lora], index: usize, module: Module| -> Result<Mat> {
            let (downs, ups, rank, k, n, scale) = merge(&st, loras, module.parts())?;
            if k != width && !matches!(module, Module::Mlp2) {
                bail!("block {index}: a LoRA of {k} inputs on a model {width} wide");
            }
            let mut up = [0usize; 3];
            for (p, u) in ups.iter().enumerate() {
                up[p] = push(u);
            }
            max_rank = max_rank.max(rank);
            Ok(Mat { down: push(&downs.concat()), up, parts: module.parts(), rank, k, n, scale })
        };
        let stream_lora = |roots: &Vec<Option<String>>| Lora { roots: roots.clone(), down: "lora_down", up: "lora_up" };
        let mut mats = HashMap::new();
        for (&(stream, index, module), roots) in &found {
            if roots.iter().any(Option::is_none) {
                bail!("block {index}: the adapter has some but not all of self_attn.q/k/v_proj");
            }
            mats.insert((stream, map[index], module), upload(&[stream_lora(roots)], index, module)?);
        }
        // each branch mode's projections: the stream's LoRA, then its branches in metadata order (`style_branch._forward`)
        let mut branched = HashMap::new();
        let mut broken = Vec::new();
        for mode in modes.iter().filter(|m| BRANCH_MODES.contains(&m.as_str())) {
            let active: Vec<&Branch> = branch_specs.iter().filter(|b| b.modes.contains(mode) && branch_found.contains_key(&b.name)).collect();
            if let Some(e) = active.iter().find_map(|b| unsupported.get(&b.name)) {
                broken.push((mode.clone(), e.clone()));
                continue;
            }
            let mut keys: Vec<(LoraStream, usize, Module)> = active.iter().flat_map(|b| branch_found[&b.name].keys().copied()).collect();
            keys.sort_by_key(|&(s, i, m)| (s as u8, i, m as u8));
            keys.dedup();
            for (stream, index, module) in keys {
                let mut loras: Vec<Lora> = found.get(&(stream, index, module)).map(stream_lora).into_iter().collect();
                for b in active.iter().filter(|b| b.stream == stream) {
                    if let Some(roots) = branch_found[&b.name].get(&(stream, index, module)) {
                        loras.push(Lora { roots: roots.clone(), down: "down", up: "up" });
                    }
                }
                branched.insert((mode.clone(), stream, map[index], module), upload(&loras, index, module)?);
            }
        }
        let buf = DevBuf::alloc(host.len().max(1))?;
        buf.upload(0, &host, st0)?;

        let mode_embed = match st.has(&format!("{PREFIX}mode_embed")) {
            true => {
                let e = st.f32_vec(&format!("{PREFIX}mode_embed"), false)?;
                if e.len() != modes.len() * width {
                    bail!("mode_embed has {} values for {} modes", e.len(), modes.len());
                }
                // `.to(t_emb)`: rounded to the activations' bf16 before it is added
                let bytes: Vec<u8> = e.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect();
                let d = DevBuf::alloc(bytes.len())?;
                d.upload(0, &bytes, st0)?;
                Some(d)
            }
            false => None,
        };
        st0.sync()?;
        Ok(Adapter { path: path.to_path_buf(), ref_t, modes, mode_embed, buf, mats, branched, broken, max_rank })
    }

    /// The LoRA on a projection in `stream`; with `branches`, the one of that mode's branches merged in, where they
    /// touch it.
    pub fn mat(&self, stream: LoraStream, block: usize, module: Module, branches: Option<&str>) -> Option<&Mat> {
        branches.and_then(|mode| self.branched.get(&(mode.to_string(), stream, block, module)))
            .or_else(|| self.mats.get(&(stream, block, module)))
    }

    /// The node's `check_mode`: `cs` and `char` always pass; a branch mode needs the adapter to list it, to carry its
    /// mode embedding and to have a branch for it, or the render would silently be a `cs`-like one.
    pub fn check_mode(&self, mode: &str) -> Result<()> {
        if !BRANCH_MODES.contains(&mode) {
            return Ok(());
        }
        let name = self.path.file_name().map_or_else(|| self.path.display().to_string(), |n| n.to_string_lossy().into_owned());
        if !self.modes.iter().any(|m| m == mode) {
            bail!("SnapRef: {name} has no '{mode}' mode (its modes: {}). Style mode needs an adapter with a style branch, such as \
                snapref_v2s_pre1; use mode cs or char with this adapter.", self.modes.join(", "));
        }
        if let Some((_, e)) = self.broken.iter().find(|(m, _)| m == mode) {
            bail!("SnapRef: {name}'s '{mode}' branch can't run on this engine: {e}");
        }
        if self.mode_embed.is_none() || !self.branched.keys().any(|k| k.0 == mode) {
            bail!("SnapRef: {name} lists the '{mode}' mode but has no mode_embed row or branch weights for it. Style mode needs an \
                adapter such as snapref_v2s_pre1.");
        }
        Ok(())
    }

    pub fn at(&self, off: usize) -> PM {
        self.buf.at(off)
    }

    /// The mode embedding to add for `mode`, or none when the file has no embedding. A mode the file doesn't list is
    /// refused (the node would silently skip the embedding).
    pub fn mode_embed(&self, mode: &str, width: usize) -> Result<Option<PM>> {
        let Some(e) = &self.mode_embed else { return Ok(None) };
        let i = self.modes.iter().position(|m| m == mode)
            .ok_or_else(|| anyhow!("SnapRef mode '{mode}' is not one of this adapter's ({})", self.modes.join(", ")))?;
        Ok(Some(e.at(i * width * 2)))
    }
}

/// Every block's self-attention keys and values of the references: bf16 [blocks][rows][k | v], rows being the
/// references' tokens one after another.
pub struct RefKv {
    pub kv: DevBuf,
    pub rows: usize,
    /// Tokens of each reference.
    pub sizes: Vec<usize>,
    /// Bytes of one row (k and v).
    pub row_bytes: usize,
}

impl RefKv {
    pub fn alloc(blocks: usize, sizes: Vec<usize>, row_bytes: usize) -> Result<Self> {
        let rows = sizes.iter().sum();
        Ok(RefKv { kv: DevBuf::alloc(blocks * rows * row_bytes)?, rows, sizes, row_bytes })
    }

    /// Block `b`'s rows.
    pub fn block(&self, b: usize) -> PM {
        self.kv.at(b * self.rows * self.row_bytes)
    }

    /// First row of reference `i`.
    pub fn start(&self, i: usize) -> usize {
        self.sizes[..i].iter().sum()
    }
}

/// Which of the references' keys a target evaluation reads, and the bias on each reference's logits.
#[derive(Clone, Debug)]
pub struct RefRead {
    /// One per reference, natural log.
    pub biases: Vec<f32>,
}

impl RefRead {
    /// Consecutive references of equal bias, merged into one group of keys: (first row, rows, bias).
    pub fn groups(&self, sizes: &[usize]) -> Vec<(usize, usize, f32)> {
        let mut out: Vec<(usize, usize, f32)> = Vec::new();
        let mut start = 0;
        for (&size, &bias) in sizes.iter().zip(&self.biases) {
            match out.last_mut() {
                Some(g) if g.2 == bias => g.1 += size,
                _ => out.push((start, size, bias)),
            }
            start += size;
        }
        out
    }

    /// No bias: the keys can simply be prepended.
    pub fn plain(&self) -> bool {
        self.biases.iter().all(|b| *b == 0.0)
    }
}

/// The node's `_encode_image` resize of RGB f32 [h, w, 3]: shrunk (never enlarged) to `megapixels` with the aspect
/// kept, each side cut down to a multiple of 16, through ComfyUI's lanczos (which runs even at the same size).
pub fn shrink(rgb: &[f32], w: usize, h: usize, megapixels: f64) -> (Vec<f32>, usize, usize) {
    let s = (megapixels * 1024.0 * 1024.0 / (w * h) as f64).sqrt().min(1.0);
    let side = |n: usize| ((n as f64 * s) as usize / 16 * 16).max(16);
    let (nw, nh) = (side(w), side(h));
    (crate::pixal3d::lanczos_f32(rgb, w, h, 3, nw, nh), nw, nh)
}

/// The node's `_stitch` (multi_layout "stitch"): RGB f32 pictures side by side at the first one's height, each
/// through ComfyUI's lanczos.
pub fn stitch(images: &[(Vec<f32>, usize, usize)]) -> (Vec<f32>, usize, usize) {
    let h = images[0].2;
    let parts: Vec<(Vec<f32>, usize)> = images.iter().map(|(rgb, w, ih)| {
        let nw = round_half_even(*w as f64 * h as f64 / *ih as f64).max(1) as usize;
        (crate::pixal3d::lanczos_f32(rgb, *w, *ih, 3, nw, h), nw)
    }).collect();
    let total: usize = parts.iter().map(|p| p.1).sum();
    let mut out = Vec::with_capacity(total * h * 3);
    for y in 0..h {
        for (rgb, w) in &parts {
            out.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
        }
    }
    (out, total, h)
}

/// Python's `round`: halves go to the even neighbour.
pub fn round_half_even(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The reference's first token row and column: centred on the target's token grid of `target` latent pixels, as the
/// node places it.
pub fn offset(target: (usize, usize), tokens: (usize, usize)) -> (i64, i64) {
    let c = |t: usize, n: usize| round_half_even((t / 2) as f64 / 2.0 - n as f64 / 2.0);
    (c(target.0, tokens.0), c(target.1, tokens.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_decodes() {
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0xb8), -1.0);
        assert_eq!(e4m3(0x7e), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert!(e4m3(0x7f).is_nan());
    }

    #[test]
    fn block_maps_like_the_node() {
        let all: Vec<usize> = (0..28).collect();
        assert_eq!(block_map(&all, 28).unwrap(), all);
        let m = block_map(&all, 40).unwrap();
        assert_eq!(m.len(), 28);
        assert_eq!(&m[..4], &[0, 1, 3, 4]);
        assert!(block_map(&all, 30).is_err());
    }

    #[test]
    fn reference_is_centred_like_the_node() {
        // 1024x1024 target, 1 MP reference: same grid
        assert_eq!(offset((128, 128), (64, 64)), (0, 0));
        // round((128 // 2) / 2 - 52 / 2) = 6, round(32 - 38) = -6
        assert_eq!(offset((128, 128), (52, 76)), (6, -6));
        // 32 - 12.5 = 19.5 rounds to 20; 32 - 13.5 = 18.5 rounds to 18
        assert_eq!(offset((128, 128), (25, 27)), (20, 18));
    }

    #[test]
    fn shrinks_like_the_node() {
        let (_, w, h) = shrink(&vec![0.5; 2048 * 1024 * 3], 2048, 1024, 1.0);
        // s = sqrt(0.5): int(1448.15) // 16 * 16, int(724.08) // 16 * 16
        assert_eq!((w, h), (1440, 720));
        let (_, w, h) = shrink(&vec![0.5; 500 * 300 * 3], 500, 300, 1.0);
        assert_eq!((w, h), (496, 288));
    }

    #[test]
    fn stitches_at_the_first_height() {
        let a = (vec![1.0; 4 * 2 * 3], 4, 2);
        let b = (vec![0.0; 2 * 4 * 3], 2, 4);
        let (rgb, w, h) = stitch(&[a, b]);
        assert_eq!((w, h), (5, 2));
        assert_eq!(&rgb[..12], &[1.0; 12]);
        assert_eq!(&rgb[12..15], &[0.0; 3]);
    }

    /// A safetensors file of f32 tensors.
    fn write_st(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            data.extend_from_slice(&crate::st::f32_bytes(values));
            header.insert(name.to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn merged_loras_sum_their_deltas() {
        let path = std::env::temp_dir().join(format!("snapref_merge_{}.safetensors", std::process::id()));
        // stream LoRA: rank 2, alpha 1 (scale 0.5); branch: rank 1, alpha 2 (scale 2); k = 3, n = 2
        let (d1, u1) = (vec![1.0, 0.0, 2.0, 0.0, 1.0, -1.0], vec![1.0, 2.0, -1.0, 0.5]);
        let (d2, u2) = (vec![0.5, 0.5, 0.5], vec![1.0, -2.0]);
        write_st(&path, &[
            ("s.lora_down", vec![2, 3], d1.clone()), ("s.lora_up", vec![2, 2], u1.clone()), ("s.alpha", vec![], vec![1.0]),
            ("b.down", vec![1, 3], d2.clone()), ("b.up", vec![2, 1], u2.clone()), ("b.alpha", vec![], vec![2.0]),
        ]);
        let st = SafeTensors::open(&path).unwrap();
        let loras = [
            Lora { roots: vec![Some("s".into())], down: "lora_down", up: "lora_up" },
            Lora { roots: vec![Some("b".into())], down: "down", up: "up" },
        ];
        let (downs, ups, rank, k, n, scale) = merge(&st, &loras, 1).unwrap();
        drop(st);
        let _ = std::fs::remove_file(&path);
        assert_eq!((rank, k, n, scale), (3, 3, 2, 0.5));
        let x = [0.3f32, -1.0, 2.0];
        // up(down(x)) * scale, for one LoRA of rank r
        let delta = |d: &[f32], u: &[f32], r: usize, s: f32| -> Vec<f32> {
            let t: Vec<f32> = (0..r).map(|i| (0..3).map(|j| d[i * 3 + j] * x[j]).sum()).collect();
            (0..2).map(|o| (0..r).map(|i| u[o * r + i] * t[i]).sum::<f32>() * s).collect()
        };
        let want: Vec<f32> = delta(&d1, &u1, 2, 0.5).iter().zip(delta(&d2, &u2, 1, 2.0)).map(|(a, b)| a + b).collect();
        let got = delta(&downs[0], &ups[0], 3, scale);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn a_missing_part_is_zero() {
        let path = std::env::temp_dir().join(format!("snapref_parts_{}.safetensors", std::process::id()));
        write_st(&path, &[("q.down", vec![1, 2], vec![1.0, 1.0]), ("q.up", vec![2, 1], vec![1.0, 1.0]), ("q.alpha", vec![], vec![1.0])]);
        let st = SafeTensors::open(&path).unwrap();
        let (downs, ups, rank, ..) = merge(&st, &[Lora { roots: vec![Some("q".into()), None, None], down: "down", up: "up" }], 3).unwrap();
        drop(st);
        let _ = std::fs::remove_file(&path);
        assert_eq!(rank, 1);
        assert_eq!(downs[0], vec![1.0, 1.0]);
        assert_eq!((downs[1].clone(), ups[2].clone()), (vec![0.0, 0.0], vec![0.0, 0.0]));
    }

    #[test]
    fn equal_neighbours_share_a_group() {
        let r = RefRead { biases: vec![0.5, 0.5, -1.0, 0.5] };
        assert_eq!(r.groups(&[10, 20, 30, 40]), vec![(0, 30, 0.5), (30, 30, -1.0), (60, 40, 0.5)]);
        assert!(!r.plain());
        assert!(RefRead { biases: vec![0.0, 0.0] }.plain());
    }
}
