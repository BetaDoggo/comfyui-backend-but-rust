//! CLIP text encoders of SD 1.x and SDXL: CLIP ViT-L/14 and OpenCLIP ViT-bigG/14, as ComfyUI's `sd1_clip` runs them. The
//! weights are held in fp16, as ComfyUI loads them, and everything is computed in float32 (`manual_cast` with float32
//! activations).

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F16, DT_F32, PM};
use crate::offload::Layout;
use crate::st::{f16_bytes, Dt, SafeTensors};
use crate::tokenizer::{CLIP_CHUNK, CLIP_END};
use anyhow::{bail, Context, Result};

const EPS: f32 = 1e-5;
const HEAD: usize = 64;

/// Where the weights of an encoder are in a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Naming {
    /// transformers' `CLIPTextModel`: `text_model.encoder.layers.N.self_attn.q_proj`, ...
    Hf,
    /// OpenCLIP: `transformer.resblocks.N.attn.in_proj_weight`, ..., and `text_projection` as a matrix to multiply by
    OpenClip,
}

pub struct Clip {
    buf: DevBuf,
    lay: Layout,
    pub width: usize,
    layers: usize,
    heads: usize,
    ff: usize,
    /// quick_gelu (CLIP-L) or gelu (bigG)
    gelu: bool,
    /// `text_projection`, for the pooled output; transposed when false (OpenCLIP's matrix)
    projection: Option<bool>,
    /// fp16 weights cast to float32 for one layer at a time
    scratch: DevBuf,
}

/// What [`Clip::encode`] returns for each chunk, float32 on the host.
pub struct Encoded {
    /// [chunks * 77, width]: the chosen layer's output
    pub hidden: Vec<f32>,
    /// The first chunk's pooled output, projected: bigG's, for SDXL's conditioning.
    pub pooled: Option<Vec<f32>>,
}

/// The encoder's prefix and naming in `st`: the key before `text_model.embeddings.token_embedding.weight` or
/// `token_embedding.weight`, with this many `width`.
fn find(st: &SafeTensors, width: usize) -> Option<(String, Naming)> {
    for (name, info) in &st.tensors {
        if info.shape.len() != 2 || info.shape[1] != width || info.shape[0] != 49408 {
            continue;
        }
        if let Some(p) = name.strip_suffix("text_model.embeddings.token_embedding.weight") {
            return Some((p.to_string(), Naming::Hf));
        }
        if let Some(p) = name.strip_suffix("token_embedding.weight") {
            if st.has(&format!("{p}transformer.resblocks.0.attn.in_proj_weight")) {
                return Some((p.to_string(), Naming::OpenClip));
            }
        }
    }
    None
}

/// True if `st` holds a CLIP-L (768 wide) text encoder.
pub fn has_clip_l(st: &SafeTensors) -> bool {
    find(st, 768).is_some()
}

/// True if `st` holds an OpenCLIP bigG (1280 wide) text encoder.
pub fn has_clip_g(st: &SafeTensors) -> bool {
    find(st, 1280).is_some()
}

/// fp16 bytes of the tensors `names`, one after another.
fn f16_of(st: &SafeTensors, names: &[String]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for n in names {
        if st.info(n)?.dtype == Dt::F16 {
            out.extend_from_slice(st.bytes(n)?);
        } else {
            out.extend(f16_bytes(&st.f32_vec(n, false)?));
        }
    }
    Ok(out)
}

impl Clip {
    /// CLIP-L (`width` 768) or bigG (1280) from `st`, wherever in the file it is.
    pub fn load(st: &SafeTensors, width: usize, s0: &Stream) -> Result<Self> {
        let (p, naming) = find(st, width).with_context(|| format!("no CLIP text encoder {width} wide"))?;
        let mut lay = Layout::new();
        let f16 = |lay: &mut Layout, key: &str, names: &[String]| -> Result<()> {
            lay.bytes(key, f16_of(st, names)?);
            Ok(())
        };
        // norms and biases as float32, of the fp16 values ComfyUI holds
        let f32 = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            lay.bytes(key, crate::st::f32_bytes(&f16_round(&st.f32_vec(name, false)?)));
            Ok(())
        };
        let (layers, ff, projection);
        match naming {
            Naming::Hf => {
                let t = format!("{p}text_model.");
                layers = (0..).take_while(|i| st.has(&format!("{t}encoder.layers.{i}.layer_norm1.weight"))).count();
                ff = st.info(&format!("{t}encoder.layers.0.mlp.fc1.weight"))?.shape[0];
                f16(&mut lay, "tok", &[format!("{t}embeddings.token_embedding.weight")])?;
                f16(&mut lay, "pos", &[format!("{t}embeddings.position_embedding.weight")])?;
                for i in 0..layers {
                    let l = |x: &str| format!("{t}encoder.layers.{i}.{x}");
                    f32(&mut lay, &format!("{i}.ln1.w"), &l("layer_norm1.weight"))?;
                    f32(&mut lay, &format!("{i}.ln1.b"), &l("layer_norm1.bias"))?;
                    f32(&mut lay, &format!("{i}.ln2.w"), &l("layer_norm2.weight"))?;
                    f32(&mut lay, &format!("{i}.ln2.b"), &l("layer_norm2.bias"))?;
                    let qkv = ["q", "k", "v"];
                    f16(&mut lay, &format!("{i}.qkv.w"), &qkv.map(|x| l(&format!("self_attn.{x}_proj.weight"))))?;
                    let biases: Vec<f32> = qkv.iter().map(|x| st.f32_vec(&l(&format!("self_attn.{x}_proj.bias")), false)).collect::<Result<Vec<_>>>()?
                        .concat();
                    lay.bytes(&format!("{i}.qkv.b"), crate::st::f32_bytes(&f16_round(&biases)));
                    f16(&mut lay, &format!("{i}.o.w"), &[l("self_attn.out_proj.weight")])?;
                    f32(&mut lay, &format!("{i}.o.b"), &l("self_attn.out_proj.bias"))?;
                    f16(&mut lay, &format!("{i}.fc1.w"), &[l("mlp.fc1.weight")])?;
                    f32(&mut lay, &format!("{i}.fc1.b"), &l("mlp.fc1.bias"))?;
                    f16(&mut lay, &format!("{i}.fc2.w"), &[l("mlp.fc2.weight")])?;
                    f32(&mut lay, &format!("{i}.fc2.b"), &l("mlp.fc2.bias"))?;
                }
                f32(&mut lay, "lnf.w", &format!("{t}final_layer_norm.weight"))?;
                f32(&mut lay, "lnf.b", &format!("{t}final_layer_norm.bias"))?;
                let proj = format!("{p}text_projection.weight");
                projection = st.has(&proj).then_some(true);
                if projection.is_some() {
                    f16(&mut lay, "proj", &[proj])?;
                }
            }
            Naming::OpenClip => {
                layers = (0..).take_while(|i| st.has(&format!("{p}transformer.resblocks.{i}.ln_1.weight"))).count();
                ff = st.info(&format!("{p}transformer.resblocks.0.mlp.c_fc.weight"))?.shape[0];
                f16(&mut lay, "tok", &[format!("{p}token_embedding.weight")])?;
                f16(&mut lay, "pos", &[format!("{p}positional_embedding")])?;
                for i in 0..layers {
                    let l = |x: &str| format!("{p}transformer.resblocks.{i}.{x}");
                    f32(&mut lay, &format!("{i}.ln1.w"), &l("ln_1.weight"))?;
                    f32(&mut lay, &format!("{i}.ln1.b"), &l("ln_1.bias"))?;
                    f32(&mut lay, &format!("{i}.ln2.w"), &l("ln_2.weight"))?;
                    f32(&mut lay, &format!("{i}.ln2.b"), &l("ln_2.bias"))?;
                    f16(&mut lay, &format!("{i}.qkv.w"), &[l("attn.in_proj_weight")])?;
                    f32(&mut lay, &format!("{i}.qkv.b"), &l("attn.in_proj_bias"))?;
                    f16(&mut lay, &format!("{i}.o.w"), &[l("attn.out_proj.weight")])?;
                    f32(&mut lay, &format!("{i}.o.b"), &l("attn.out_proj.bias"))?;
                    f16(&mut lay, &format!("{i}.fc1.w"), &[l("mlp.c_fc.weight")])?;
                    f32(&mut lay, &format!("{i}.fc1.b"), &l("mlp.c_fc.bias"))?;
                    f16(&mut lay, &format!("{i}.fc2.w"), &[l("mlp.c_proj.weight")])?;
                    f32(&mut lay, &format!("{i}.fc2.b"), &l("mlp.c_proj.bias"))?;
                }
                f32(&mut lay, "lnf.w", &format!("{p}ln_final.weight"))?;
                f32(&mut lay, "lnf.b", &format!("{p}ln_final.bias"))?;
                let proj = format!("{p}text_projection");
                projection = st.has(&proj).then_some(false);
                if projection.is_some() {
                    f16(&mut lay, "proj", &[proj])?;
                }
            }
        }
        if layers == 0 || width % HEAD != 0 {
            bail!("CLIP text encoder of {layers} layers, {width} wide");
        }
        let mut host = vec![0u8; lay.size];
        lay.pack(st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        // the largest layer's matrices as float32
        let scratch = DevBuf::alloc((3 * width * width + width * width + 2 * ff * width) * 4)?;
        s0.sync()?;
        Ok(Clip { buf, lay, width, layers, heads: width / HEAD, ff, gelu: width != 768, projection, scratch })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    /// `x @ w.T + b`, float32 rows [m, k] -> [m, n]; `w` is an fp16 [n, k] weight, cast into `wf` first.
    fn linear(&self, x: PM, w: &str, b: &str, wf: PM, out: PM, m: usize, n: usize, kk: usize, st: &Stream) -> Result<()> {
        k!(gpu_cast(DT_F16, DT_F32, self.w(w), wf, (n * kk) as i64, s(st)))?;
        k!(gpu_gemm(x, kk as i64, wf, kk as i64, out, n as i64, self.w(b), m as i64, n as i64, kk as i64, 1, DT_F32, DT_F32, 1.0, 0.0, s(st)))
    }

    /// Run 77-token chunks through the encoder together. `layer` is the layer whose output is returned (counted from
    /// 0; the last for "last"), through the final LayerNorm when `final_norm`. `pooled` also returns the first chunk's
    /// projected pooled output: the final layer's, normed, at its first end token.
    pub fn encode(&self, chunks: &[Vec<u32>], layer: usize, final_norm: bool, pooled: bool, st: &Stream) -> Result<Encoded> {
        let (d, b) = (self.width, chunks.len());
        if layer >= self.layers || b == 0 || chunks.iter().any(|c| c.len() != CLIP_CHUNK) {
            bail!("CLIP layer {layer} of {} for {b} chunks", self.layers);
        }
        let rows = b * CLIP_CHUNK;
        let ids: Vec<u8> = chunks.iter().flatten().flat_map(|&t| (t as i32).to_le_bytes()).collect();
        let idb = DevBuf::alloc_async(ids.len(), st)?;
        idb.upload(0, &ids, st)?;
        let alloc = |n: usize| DevBuf::alloc_async(n * 4, st);
        let x = alloc(rows * d)?;
        for c in 0..b {
            // each chunk's positions start again at 0
            k!(gpu_sd_clip_embed(self.w("tok"), self.w("pos"), off(idb.ptr(), c * CLIP_CHUNK * 4), off(x.ptr(), c * CLIP_CHUNK * d * 4),
                CLIP_CHUNK as i32, d as i32, s(st)))?;
        }
        let (h, qkv, att, mid) = (alloc(rows * d)?, alloc(rows * 3 * d)?, alloc(rows * d)?, alloc(rows * self.ff)?);
        let wf = self.scratch.ptr();
        let mut picked = None;
        for i in 0..self.layers {
            let key = |x: &str| format!("{i}.{x}");
            k!(gpu_sd_layer_norm(DT_F32, x.ptr(), d as i64, h.ptr(), d as i64, self.w(&key("ln1.w")), self.w(&key("ln1.b")), rows as i32, d as i32,
                EPS, s(st)))?;
            self.linear(h.ptr(), &key("qkv.w"), &key("qkv.b"), wf, qkv.ptr(), rows, 3 * d, d, st)?;
            for c in 0..b {
                let q = off(qkv.ptr(), c * CLIP_CHUNK * 3 * d * 4);
                k!(gpu_attn_small(DT_F32, q, (3 * d) as i64, off(q, d * 4), off(q, 2 * d * 4), (3 * d) as i64, self.heads as i32, self.heads as i32,
                    HEAD as i32, CLIP_CHUNK as i32, CLIP_CHUNK as i32, off(att.ptr(), c * CLIP_CHUNK * d * 4), d as i64,
                    1.0 / (HEAD as f32).sqrt(), 1, s(st)))?;
            }
            self.linear(att.ptr(), &key("o.w"), &key("o.b"), wf, h.ptr(), rows, d, d, st)?;
            k!(gpu_add_f32(x.ptr(), h.ptr(), (rows * d) as i64, s(st)))?;
            k!(gpu_sd_layer_norm(DT_F32, x.ptr(), d as i64, h.ptr(), d as i64, self.w(&key("ln2.w")), self.w(&key("ln2.b")), rows as i32, d as i32,
                EPS, s(st)))?;
            self.linear(h.ptr(), &key("fc1.w"), &key("fc1.b"), wf, mid.ptr(), rows, self.ff, d, st)?;
            k!(gpu_sd_clip_act(mid.ptr(), (rows * self.ff) as i64, self.gelu as i32, s(st)))?;
            self.linear(mid.ptr(), &key("fc2.w"), &key("fc2.b"), wf, h.ptr(), rows, d, self.ff, st)?;
            k!(gpu_add_f32(x.ptr(), h.ptr(), (rows * d) as i64, s(st)))?;
            if i == layer {
                let out = alloc(rows * d)?;
                if final_norm {
                    k!(gpu_sd_layer_norm(DT_F32, x.ptr(), d as i64, out.ptr(), d as i64, self.w("lnf.w"), self.w("lnf.b"), rows as i32, d as i32, EPS,
                        s(st)))?;
                } else {
                    crate::cuda::memcpy_d2d(out.ptr(), x.ptr(), rows * d * 4, st)?;
                }
                picked = Some(out);
                if !pooled {
                    break;
                }
            }
        }
        let read = |b: &DevBuf, n: usize| -> Result<Vec<f32>> {
            let mut bytes = vec![0u8; n * 4];
            b.download(0, &mut bytes, st)?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let hidden = read(picked.as_ref().unwrap(), rows * d)?;
        let pooled = if pooled {
            let Some(transposed) = self.projection else { bail!("this CLIP has no text projection") };
            // the final layer, normed, at the first chunk's first end token
            let eos = chunks[0].iter().position(|&t| t == CLIP_END).unwrap_or(0);
            k!(gpu_sd_layer_norm(DT_F32, off(x.ptr(), eos * d * 4), d as i64, h.ptr(), d as i64, self.w("lnf.w"), self.w("lnf.b"), 1, d as i32, EPS,
                s(st)))?;
            k!(gpu_cast(DT_F16, DT_F32, self.w("proj"), wf, (d * d) as i64, s(st)))?;
            let out = alloc(d)?;
            k!(gpu_gemm(h.ptr(), d as i64, wf, d as i64, out.ptr(), d as i64, null(), 1, d as i64, d as i64, transposed as i32, DT_F32, DT_F32, 1.0, 0.0,
                s(st)))?;
            Some(read(&out, d)?)
        } else {
            None
        };
        Ok(Encoded { hidden, pooled })
    }
}

/// Values rounded to fp16, as ComfyUI holds the encoder's biases.
fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect()
}
