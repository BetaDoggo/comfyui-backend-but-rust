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
    /// Largest rank, for scratch sizes.
    pub max_rank: usize,
}

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

/// A LoRA matrix as bf16 bytes. FP8 files carry a float32 scale per row (`<key>_scale`), as the node dequantizes them.
fn matrix(st: &SafeTensors, key: &str) -> Result<(Vec<u8>, usize, usize)> {
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
    Ok((values.iter().flat_map(|v| bf16::from_f32(*v).to_le_bytes()).collect(), rows, cols))
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

        // anima_ref_kv.<stream>.blocks.<i>.<module>.lora_down
        let mut found: HashMap<(LoraStream, usize, Module), Vec<Option<String>>> = HashMap::new();
        let mut indices = Vec::new();
        for key in st.tensors.keys() {
            let Some(root) = key.strip_suffix(".lora_down") else { continue };
            let rest = root.strip_prefix(PREFIX).ok_or_else(|| anyhow!("{key}: unexpected key"))?;
            let (stream, name) = rest.split_once('.').ok_or_else(|| anyhow!("{key}: unexpected key"))?;
            let stream = match stream {
                "ref" => LoraStream::Ref,
                "tgt" => LoraStream::Tgt,
                s => bail!("{key}: unknown stream {s}"),
            };
            let Some((index, module)) = name.strip_prefix("blocks.").and_then(|n| n.split_once('.')) else {
                bail!("{key}: SnapRef LoRAs outside the blocks are not supported");
            };
            let index: usize = index.parse().map_err(|_| anyhow!("{key}: bad block index"))?;
            let (module, part) = Module::parse(module).ok_or_else(|| anyhow!("{key}: unsupported target {module}"))?;
            indices.push(index);
            found.entry((stream, index, module)).or_insert_with(|| vec![None; module.parts()])[part] = Some(root.to_string());
        }
        indices.sort_unstable();
        indices.dedup();
        let map = block_map(&indices, model_blocks)?;

        let mut host: Vec<u8> = Vec::new();
        let mut push = |bytes: &[u8]| {
            let at = host.len().next_multiple_of(256);
            host.resize(at, 0);
            host.extend_from_slice(bytes);
            at
        };
        let mut mats = HashMap::new();
        let mut max_rank = 0;
        for ((stream, index, module), roots) in found {
            let roots: Vec<String> = roots.into_iter().collect::<Option<_>>()
                .ok_or_else(|| anyhow!("block {index}: the adapter has some but not all of self_attn.q/k/v_proj"))?;
            let mut downs = Vec::new();
            let mut ups = [0usize; 3];
            let (mut rank, mut k, mut n, mut scale) = (0, 0, 0, 0.0);
            for (p, root) in roots.iter().enumerate() {
                let (down, r, kk) = matrix(&st, &format!("{root}.lora_down"))?;
                let (up, nn, r2) = matrix(&st, &format!("{root}.lora_up"))?;
                let alpha = st.f32_vec(&format!("{root}.alpha"), false)?.first().copied().unwrap_or(r as f32);
                if r2 != r || (p > 0 && (r, kk, nn) != (rank, k, n)) {
                    bail!("{root}: mismatched LoRA shapes");
                }
                (rank, k, n, scale) = (r, kk, nn, alpha / r as f32);
                downs.extend_from_slice(&down);
                ups[p] = push(&up);
            }
            if k != width && !matches!(module, Module::Mlp2) {
                bail!("block {index}: a LoRA of {k} inputs on a model {width} wide");
            }
            let down = push(&downs);
            max_rank = max_rank.max(rank);
            mats.insert((stream, map[index], module), Mat { down, up: ups, parts: module.parts(), rank, k, n, scale });
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
        Ok(Adapter { path: path.to_path_buf(), ref_t, modes, mode_embed, buf, mats, max_rank })
    }

    pub fn mat(&self, stream: LoraStream, block: usize, module: Module) -> Option<&Mat> {
        self.mats.get(&(stream, block, module))
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

    #[test]
    fn equal_neighbours_share_a_group() {
        let r = RefRead { biases: vec![0.5, 0.5, -1.0, 0.5] };
        assert_eq!(r.groups(&[10, 20, 30, 40]), vec![(0, 30, 0.5), (30, 30, -1.0), (60, 40, 0.5)]);
        assert!(!r.plain());
        assert!(RefRead { biases: vec![0.0, 0.0] }.plain());
    }
}
