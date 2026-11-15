//! The UNet of SD 1.x and SDXL (ComfyUI's `openaimodel.UNetModel`), NHWC fp16 as ComfyUI runs it. Its layout is read
//! from the checkpoint's keys, so both families share this code: SD 1.x has eight heads per transformer and three
//! downsamplings, SDXL heads 64 wide, two downsamplings and the size conditioning (`label_emb`).
//!
//! The transformers' linears run fp16 as stored, or int8 (W8A8, quantized per row as they load, ConvRot-rotated in
//! Hadamard groups of 256, or of 64 where their input width isn't a multiple of 256). Convolutions and everything else stay fp16. The cross-attention keys and
//! values depend only on the prompt, so they are computed once per prompt ([`Context`]).

use crate::cuda::{memcpy_2d_d2d, DevBuf, Stream};
use crate::ffi::{k, null, off, s, DT_F16, DT_F32, P, PM};
use crate::offload::Layout;
use crate::st::{f16_bytes, f32_bytes, Dt, SafeTensors};
use anyhow::{bail, Context as _, Result};
use half::f16;
use std::collections::HashMap;

/// Latent channels, and as many padded for the convolution kernels.
pub const CH: usize = 4;
const CH_PAD: usize = 8;
const GROUPS: usize = 32;
const RES_EPS: f32 = 1e-5;
const ST_EPS: f32 = 1e-6;
const LN_EPS: f32 = 1e-5;
/// SDXL's size conditioning: each of six numbers embedded this wide.
const ADM_EMBED: usize = 256;

/// The prefix of the UNet's keys in `st`, if it has one.
pub fn key_prefix(has: impl Fn(&str) -> bool) -> Option<&'static str> {
    ["model.diffusion_model.", ""].into_iter().find(|p| has(&format!("{p}input_blocks.0.0.weight")) && has(&format!("{p}time_embed.0.weight")))
}

#[derive(Clone, Debug)]
enum Op {
    /// `input_blocks.0`: the first convolution
    ConvIn,
    Res { key: String, cin: usize, cout: usize, skip: bool },
    Transformer { key: String, c: usize, depth: usize, heads: usize, hd: usize },
    Down { key: String, c: usize },
    Up { key: String, c: usize },
}

/// One input, middle or output block's weights: fp16 and float32 in `main`, the transformers' linears in `lin`.
struct Part {
    main: DevBuf,
    lay: Layout,
    lin: Lin,
}

enum Lin {
    None,
    F16(DevBuf, Layout),
    /// int8 rows, each linear's at (weights, scales) in the buffer, with its rotation's group (0 for none)
    I8(DevBuf, HashMap<String, (usize, usize, i32)>),
}

/// A linear's shape and where its bias is.
#[derive(Clone)]
struct LinInfo {
    n: usize,
    k: usize,
    part: usize,
}

pub struct Unet {
    parts: Vec<Part>,
    /// part index of every weight key
    linears: HashMap<String, LinInfo>,
    where_: HashMap<String, usize>,
    input: Vec<Vec<Op>>,
    middle: Vec<Op>,
    output: Vec<Vec<Op>>,
    model_channels: usize,
    /// SDXL's `adm_in_channels`, 0 on SD 1.x
    pub adm: usize,
    pub context_dim: usize,
    /// The number of 2x downsamplings (odd sides round up, and the upsamples crop back to the skip connections)
    pub downsamplings: usize,
    pub int8_attn: bool,
    pub int8: bool,
}

/// The prompt as every cross-attention reads it.
pub struct Context {
    /// fp16 [len, 2 * inner] keys then values, by transformer block
    kv: HashMap<String, DevBuf>,
    pub len: usize,
}

/// Conv weight [Cout, Cin, R, R] -> fp16 [cout_pad, R, R, cin_pad], zero padded.
fn conv_krsc_f16(st: &SafeTensors, name: &str, cin_pad: usize, cout_pad: usize) -> Result<(Vec<u8>, usize, usize, usize)> {
    let info = st.info(name)?.clone();
    if info.shape.len() != 4 || info.shape[2] != info.shape[3] {
        bail!("{name}: conv weight of shape {:?}", info.shape);
    }
    let v = st.f32_vec(name, false)?;
    let (co, ci, r) = (info.shape[0], info.shape[1], info.shape[2]);
    let (cop, cip) = (cout_pad.max(co), cin_pad.max(ci));
    let mut out = vec![0f32; cop * r * r * cip];
    for o in 0..co {
        for c in 0..ci {
            for y in 0..r {
                for x in 0..r {
                    out[((o * r + y) * r + x) * cip + c] = v[((o * ci + c) * r + y) * r + x];
                }
            }
        }
    }
    Ok((f16_bytes(&out), cop, cip, r))
}

/// float32 bytes of an fp16-held vector, zero padded to `pad`.
fn vec_f32(st: &SafeTensors, name: &str, pad: usize) -> Result<Vec<u8>> {
    let mut v: Vec<f32> = st.f32_vec(name, false)?.iter().map(|x| f16::from_f32(*x).to_f32()).collect();
    v.resize(pad.max(v.len()), 0.0);
    Ok(f32_bytes(&v))
}

/// fp16 bytes of a vector, zero padded to `pad`.
fn vec_f16(st: &SafeTensors, name: &str, pad: usize) -> Result<Vec<u8>> {
    let mut v = st.f32_vec(name, false)?;
    v.resize(pad.max(v.len()), 0.0);
    Ok(f16_bytes(&v))
}

/// fp16 bytes of matrices stacked along their rows (a 1x1 conv's weight reads as its matrix).
fn mat_f16(st: &SafeTensors, names: &[String]) -> Result<Vec<u8>> {
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

/// `timestep_embedding(t, dim)` in float32: cosines then sines.
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let l = -(10000f64.ln() as f32);
    let freqs: Vec<f32> = (0..half).map(|i| (l * i as f32 / half as f32).exp()).collect();
    let args: Vec<f32> = freqs.iter().map(|f| t * f).collect();
    args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect()
}

/// SDXL's `adm` vector: the pooled prompt, then the original size, the crop and the target size, each embedded.
pub fn sdxl_adm(pooled: &[f32], width: f32, height: f32, crop_w: f32, crop_h: f32, target_width: f32, target_height: f32) -> Vec<f32> {
    let mut v = pooled.to_vec();
    for x in [height, width, crop_h, crop_w, target_height, target_width] {
        v.extend(timestep_embedding(x, ADM_EMBED));
    }
    v
}

struct Builder<'a> {
    st: &'a SafeTensors,
    p: &'a str,
    main: Layout,
    lin: Layout,
    lins: Vec<(String, usize, usize)>,
}

impl Builder<'_> {
    fn name(&self, x: &str) -> String {
        format!("{}{x}", self.p)
    }
    fn conv(&mut self, key: &str, name: &str, cin_pad: usize, cout_pad: usize) -> Result<(usize, usize)> {
        let (w, cop, cip, _) = conv_krsc_f16(self.st, &self.name(&format!("{name}.weight")), cin_pad, cout_pad)?;
        self.main.bytes(&format!("{key}.w"), w);
        self.main.bytes(&format!("{key}.b"), vec_f16(self.st, &self.name(&format!("{name}.bias")), cop)?);
        Ok((cip, cop))
    }
    fn norm(&mut self, key: &str, name: &str) -> Result<()> {
        self.main.bytes(&format!("{key}.w"), vec_f32(self.st, &self.name(&format!("{name}.weight")), 0)?);
        self.main.bytes(&format!("{key}.b"), vec_f32(self.st, &self.name(&format!("{name}.bias")), 0)?);
        Ok(())
    }
    /// A linear kept fp16 in `main` (small ones, and those run once per prompt).
    fn dense(&mut self, key: &str, names: &[String], bias: Option<&str>) -> Result<()> {
        let names: Vec<String> = names.iter().map(|n| self.name(n)).collect();
        self.main.bytes(&format!("{key}.w"), mat_f16(self.st, &names)?);
        if let Some(b) = bias {
            self.main.bytes(&format!("{key}.b"), vec_f16(self.st, &self.name(b), 0)?);
        }
        Ok(())
    }
    /// A transformer linear: `lin` holds its weight, `main` its bias.
    fn linear(&mut self, key: &str, names: &[String], bias: Option<&str>) -> Result<()> {
        let names: Vec<String> = names.iter().map(|n| self.name(n)).collect();
        let mut n = 0;
        let mut kk = 0;
        for name in &names {
            let i = self.st.info(name)?;
            n += i.shape[0];
            kk = i.shape[1];
        }
        self.lin.bytes(&format!("{key}.w"), mat_f16(self.st, &names)?);
        if let Some(b) = bias {
            self.main.bytes(&format!("{key}.b"), vec_f16(self.st, &self.name(b), 0)?);
        }
        self.lins.push((key.to_string(), n, kk));
        Ok(())
    }
}

impl Unet {
    pub fn load(st: &SafeTensors, int8: bool, s0: &Stream) -> Result<Self> {
        let p = key_prefix(|k| st.has(k)).context("no UNet in this file")?;
        let has = |x: &str| st.has(&format!("{p}{x}"));
        let shape = |x: &str| st.info(&format!("{p}{x}")).map(|i| i.shape.clone());
        let model_channels = shape("time_embed.0.weight")?[1];
        let adm = if has("label_emb.0.0.weight") { shape("label_emb.0.0.weight")?[1] } else { 0 };
        let mut context_dim = 0;
        // SD 1.x has 8 heads per transformer, SDXL heads 64 wide
        let heads_of = |c: usize| if adm > 0 { (c / 64, 64) } else { (8, c / 8) };
        let ops_of = |block: &str, cin: &mut usize, context_dim: &mut usize| -> Result<Vec<Op>> {
            let mut ops = Vec::new();
            for j in 0.. {
                let key = format!("{block}.{j}");
                if has(&format!("{key}.in_layers.0.weight")) {
                    let cout = shape(&format!("{key}.in_layers.2.weight"))?[0];
                    let c_in = shape(&format!("{key}.in_layers.2.weight"))?[1];
                    if c_in != *cin {
                        bail!("{key}: {c_in} input channels, {cin} expected");
                    }
                    ops.push(Op::Res { key, cin: c_in, cout, skip: has(&format!("{block}.{j}.skip_connection.weight")) });
                    *cin = cout;
                } else if has(&format!("{key}.transformer_blocks.0.attn1.to_q.weight")) {
                    let depth = (0..).take_while(|t| has(&format!("{key}.transformer_blocks.{t}.attn1.to_q.weight"))).count();
                    let (heads, hd) = heads_of(*cin);
                    *context_dim = shape(&format!("{key}.transformer_blocks.0.attn2.to_k.weight"))?[1];
                    ops.push(Op::Transformer { key, c: *cin, depth, heads, hd });
                } else if has(&format!("{key}.op.weight")) {
                    ops.push(Op::Down { key, c: *cin });
                } else if has(&format!("{key}.conv.weight")) {
                    ops.push(Op::Up { key, c: *cin });
                } else {
                    break;
                }
            }
            Ok(ops)
        };
        let mut c = model_channels;
        let mut input = vec![vec![Op::ConvIn]];
        let mut skips = vec![c];
        for i in 1.. {
            let block = format!("input_blocks.{i}");
            if !has(&format!("{block}.0.in_layers.0.weight")) && !has(&format!("{block}.0.op.weight")) {
                break;
            }
            input.push(ops_of(&block, &mut c, &mut context_dim)?);
            skips.push(c);
        }
        let middle = ops_of("middle_block", &mut c, &mut context_dim)?;
        let mut output = Vec::new();
        for i in 0.. {
            let block = format!("output_blocks.{i}");
            if !has(&format!("{block}.0.in_layers.0.weight")) {
                break;
            }
            c += skips.pop().context("more output blocks than skips")?;
            output.push(ops_of(&block, &mut c, &mut context_dim)?);
        }
        if !skips.is_empty() || c != model_channels {
            bail!("the UNet's blocks don't line up ({} skips left, {c} channels at the end)", skips.len());
        }
        let downsamplings = input.iter().flatten().filter(|o| matches!(o, Op::Down { .. })).count();

        // weights, one part per block: 0 the globals, then the input blocks, the middle and the output blocks
        let mut parts = Vec::new();
        let mut linears = HashMap::new();
        let mut where_ = HashMap::new();
        let all: Vec<(String, Vec<Op>)> = std::iter::once(("g".to_string(), Vec::new()))
            .chain(input.iter().enumerate().map(|(i, o)| (format!("input_blocks.{i}"), o.clone())))
            .chain(std::iter::once(("middle_block".to_string(), middle.clone())))
            .chain(output.iter().enumerate().map(|(i, o)| (format!("output_blocks.{i}"), o.clone())))
            .collect();
        for (index, (_, ops)) in all.iter().enumerate() {
            let mut b = Builder { st, p, main: Layout::new(), lin: Layout::new(), lins: Vec::new() };
            if index == 0 {
                b.dense("t0", &["time_embed.0.weight".into()], Some("time_embed.0.bias"))?;
                b.dense("t2", &["time_embed.2.weight".into()], Some("time_embed.2.bias"))?;
                if adm > 0 {
                    b.dense("l0", &["label_emb.0.0.weight".into()], Some("label_emb.0.0.bias"))?;
                    b.dense("l2", &["label_emb.0.2.weight".into()], Some("label_emb.0.2.bias"))?;
                }
                b.norm("out.n", "out.0")?;
                b.conv("out.c", "out.2", 0, CH_PAD)?;
            }
            for op in ops {
                match op {
                    Op::ConvIn => {
                        b.conv("in", "input_blocks.0.0", CH_PAD, 0)?;
                    }
                    Op::Res { key, skip, .. } => {
                        b.norm(&format!("{key}.n1"), &format!("{key}.in_layers.0"))?;
                        b.conv(&format!("{key}.c1"), &format!("{key}.in_layers.2"), 0, 0)?;
                        b.dense(&format!("{key}.emb"), &[format!("{key}.emb_layers.1.weight")], Some(&format!("{key}.emb_layers.1.bias")))?;
                        b.norm(&format!("{key}.n2"), &format!("{key}.out_layers.0"))?;
                        b.conv(&format!("{key}.c2"), &format!("{key}.out_layers.3"), 0, 0)?;
                        if *skip {
                            b.conv(&format!("{key}.skip"), &format!("{key}.skip_connection"), 0, 0)?;
                        }
                    }
                    Op::Transformer { key, depth, .. } => {
                        b.norm(&format!("{key}.norm"), &format!("{key}.norm"))?;
                        b.linear(&format!("{key}.pin"), &[format!("{key}.proj_in.weight")], Some(&format!("{key}.proj_in.bias")))?;
                        b.linear(&format!("{key}.pout"), &[format!("{key}.proj_out.weight")], Some(&format!("{key}.proj_out.bias")))?;
                        for t in 0..*depth {
                            let tk = format!("{key}.transformer_blocks.{t}");
                            for n in ["norm1", "norm2", "norm3"] {
                                b.norm(&format!("{tk}.{n}"), &format!("{tk}.{n}"))?;
                            }
                            b.linear(&format!("{tk}.qkv"), &["q", "k", "v"].map(|x| format!("{tk}.attn1.to_{x}.weight")), None)?;
                            b.linear(&format!("{tk}.o1"), &[format!("{tk}.attn1.to_out.0.weight")], Some(&format!("{tk}.attn1.to_out.0.bias")))?;
                            b.linear(&format!("{tk}.q2"), &[format!("{tk}.attn2.to_q.weight")], None)?;
                            b.dense(&format!("{tk}.kv2"), &[format!("{tk}.attn2.to_k.weight"), format!("{tk}.attn2.to_v.weight")], None)?;
                            b.linear(&format!("{tk}.o2"), &[format!("{tk}.attn2.to_out.0.weight")], Some(&format!("{tk}.attn2.to_out.0.bias")))?;
                            b.linear(&format!("{tk}.ff1"), &[format!("{tk}.ff.net.0.proj.weight")], Some(&format!("{tk}.ff.net.0.proj.bias")))?;
                            b.linear(&format!("{tk}.ff2"), &[format!("{tk}.ff.net.2.weight")], Some(&format!("{tk}.ff.net.2.bias")))?;
                        }
                    }
                    Op::Down { key, .. } => {
                        b.conv(&format!("{key}.down"), &format!("{key}.op"), 0, 0)?;
                    }
                    Op::Up { key, .. } => {
                        b.conv(&format!("{key}.up"), &format!("{key}.conv"), 0, 0)?;
                    }
                }
            }
            let mut host = vec![0u8; b.main.size];
            b.main.pack(st, &mut host)?;
            let main = DevBuf::alloc(b.main.size.max(1))?;
            main.upload(0, &host, s0)?;
            drop(host);
            let lin = if b.lins.is_empty() {
                Lin::None
            } else {
                let mut host = vec![0u8; b.lin.size];
                b.lin.pack(st, &mut host)?;
                let f = DevBuf::alloc(b.lin.size)?;
                f.upload(0, &host, s0)?;
                drop(host);
                if int8 {
                    let total: usize = b.lins.iter().map(|(_, n, kk)| (n * kk).next_multiple_of(256) + (n * 4).next_multiple_of(256)).sum();
                    let q = DevBuf::alloc(total)?;
                    let mut offs = HashMap::new();
                    let mut at = 0;
                    for (key, n, kk) in &b.lins {
                        let (w, sc) = (at, at + (n * kk).next_multiple_of(256));
                        // the widest Hadamard group that divides the row: SD 1.5's 320 and 640 wide linears take 64
                        let rotate = [256, 64].into_iter().find(|g| *kk % *g as usize == 0).unwrap_or(0i32);
                        k!(gpu_sd_quant_rows(f.at(b.lin.off(&format!("{key}.w"))), *kk as i64, q.at(w), q.at(sc), *n as i32, *kk as i32, rotate,
                            s(s0)))?;
                        offs.insert(key.clone(), (w, sc, rotate));
                        at = sc + (n * 4).next_multiple_of(256);
                    }
                    s0.sync()?;
                    Lin::I8(q, offs)
                } else {
                    Lin::F16(f, b.lin)
                }
            };
            for (key, n, kk) in &b.lins {
                linears.insert(key.clone(), LinInfo { n: *n, k: *kk, part: index });
            }
            for e in b.main.keys() {
                where_.insert(e, index);
            }
            parts.push(Part { main, lay: b.main, lin });
        }
        s0.sync()?;
        Ok(Unet { parts, linears, where_, input, middle, output, model_channels, adm, context_dim, downsamplings, int8_attn: true, int8 })
    }

    fn w(&self, key: &str) -> PM {
        let part = &self.parts[*self.where_.get(key).unwrap_or_else(|| panic!("UNet weight {key}"))];
        part.main.at(part.lay.off(key))
    }

    /// `x @ W.T + b` for a transformer linear: fp16 rows [m, k] -> [m, n].
    fn linear(&self, key: &str, x: PM, m: usize, out: PM, st: &Stream) -> Result<()> {
        let info = &self.linears[key];
        let (n, kk) = (info.n, info.k);
        let bias_key = format!("{key}.b");
        let bias: P = if self.where_.contains_key(&bias_key) { self.w(&bias_key) } else { null() };
        match &self.parts[info.part].lin {
            Lin::F16(buf, lay) => k!(gpu_gemm(x, kk as i64, buf.at(lay.off(&format!("{key}.w"))), kk as i64, out, n as i64, bias, m as i64, n as i64,
                kk as i64, 1, DT_F16, DT_F16, 1.0, 0.0, s(st))),
            Lin::I8(buf, offs) => {
                let (w, sc, rotate) = offs[key];
                let q = DevBuf::alloc_async(m * kk, st)?;
                let xs = DevBuf::alloc_async(m * 4, st)?;
                k!(gpu_sd_quant_rows(x, kk as i64, q.ptr(), xs.ptr(), m as i32, kk as i32, rotate, s(st)))?;
                k!(gpu_int8_gemm(q.ptr(), buf.at(w), xs.ptr(), buf.at(sc), bias, out, m as i64, n as i64, kk as i64, DT_F16, s(st)))
            }
            Lin::None => bail!("no linears in this part"),
        }
    }

    /// An fp16 linear held in `main`: rows [m, k] -> [m, n].
    fn dense(&self, key: &str, x: PM, m: usize, n: usize, kk: usize, bias: bool, out: PM, st: &Stream) -> Result<()> {
        let b: P = if bias { self.w(&format!("{key}.b")) } else { null() };
        k!(gpu_gemm(x, kk as i64, self.w(&format!("{key}.w")), kk as i64, out, n as i64, b, m as i64, n as i64, kk as i64, 1, DT_F16, DT_F16, 1.0,
            0.0, s(st)))
    }

    fn conv(&self, key: &str, x: PM, out: PM, n: usize, h: usize, w: usize, cin: usize, cout: usize, r: usize, stride: usize, st: &Stream)
        -> Result<()> {
        k!(gpu_conv2d_f16(x, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), out, n as i32, h as i32, w as i32, cin as i32, cout as i32,
            r as i32, stride as i32, s(st)))
    }

    fn group_norm(&self, key: &str, x: PM, out: PM, n: usize, p: usize, c: usize, eps: f32, silu: bool, st: &Stream) -> Result<()> {
        let sums = DevBuf::alloc_async(n * GROUPS * 16, st)?;
        k!(gpu_sd_group_norm(x, out, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), sums.ptr(), n as i32, p as i64, c as i32,
            GROUPS as i32, eps, silu as i32, s(st)))
    }

    /// Keys and values of every cross-attention for a prompt: `ctx` is float32 [len, context_dim].
    pub fn context(&self, ctx: &[f32], st: &Stream) -> Result<Context> {
        let len = ctx.len() / self.context_dim;
        if len == 0 || ctx.len() != len * self.context_dim {
            bail!("a context of {} values isn't rows of {}", ctx.len(), self.context_dim);
        }
        let c16 = DevBuf::alloc_async(ctx.len() * 2, st)?;
        c16.upload(0, &f16_bytes(ctx), st)?;
        let mut kv = HashMap::new();
        for op in self.input.iter().flatten().chain(&self.middle).chain(self.output.iter().flatten()) {
            if let Op::Transformer { key, c, depth, .. } = op {
                for t in 0..*depth {
                    let tk = format!("{key}.transformer_blocks.{t}");
                    let buf = DevBuf::alloc_async(len * 2 * c * 2, st)?;
                    self.dense(&format!("{tk}.kv2"), c16.ptr(), len, 2 * c, self.context_dim, false, buf.ptr(), st)?;
                    kv.insert(tk, buf);
                }
            }
        }
        Ok(Context { kv, len })
    }

    /// Attention over heads `hd` wide: `q` rows of `n * lq` queries (`n` images), `k` and `v` rows of `n * lk` keys, or of
    /// `lk` keys shared by every image when `shared`. Writes [n * lq, heads * hd] to `out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, q: PM, q_rs: usize, k_: PM, v: PM, kv_rs: usize, n: usize, lq: usize, lk: usize, shared: bool, heads: usize, hd: usize,
        out: PM, st: &Stream) -> Result<()> {
        let inner = heads * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        if self.int8_attn {
            let dp = [64, 128, 256].into_iter().find(|&d| d >= hd).context("head too wide for the int8 attention")?;
            let (b, lq_b) = if shared { (1, n * lq) } else { (n, lq) };
            let kv_rows = if shared { lk } else { n * lk };
            let ws = DevBuf::alloc_async(unsafe { crate::ffi::gpu_int8_attn_workspace_d(b as i32, heads as i32, heads as i32, lq_b as i32, lk as i32,
                dp as i32) } as usize, st)?;
            if dp == hd {
                let (q_bs, k_bs) = ((lq_b * q_rs) as i64, if shared { 0 } else { (lk * kv_rs) as i64 });
                return k!(gpu_int8_attn_f16(q, k_, v, out, ws.ptr(), b as i32, lq_b as i32, lk as i32, heads as i32, heads as i32, dp as i32, q_bs,
                    q_rs as i64, hd as i64, k_bs, kv_rs as i64, hd as i64, k_bs, kv_rs as i64, hd as i64, (lq_b * inner) as i64, inner as i64,
                    hd as i64, scale, s(st)));
            }
            let pad = |src: PM, rs: usize, rows: usize| -> Result<DevBuf> {
                let d = DevBuf::alloc_async(rows * heads * dp * 2, st)?;
                k!(gpu_sd_pad_heads(src, rs as i64, d.ptr(), rows as i64, heads as i32, hd as i32, dp as i32, s(st)))?;
                Ok(d)
            };
            let (qp, kp, vp) = (pad(q, q_rs, n * lq)?, pad(k_, kv_rs, kv_rows)?, pad(v, kv_rs, kv_rows)?);
            let op = DevBuf::alloc_async(n * lq * heads * dp * 2, st)?;
            let rs = (heads * dp) as i64;
            let k_bs = if shared { 0 } else { lk as i64 * rs };
            k!(gpu_int8_attn_f16(qp.ptr(), kp.ptr(), vp.ptr(), op.ptr(), ws.ptr(), b as i32, lq_b as i32, lk as i32, heads as i32, heads as i32,
                dp as i32, lq_b as i64 * rs, rs, dp as i64, k_bs, rs, dp as i64, k_bs, rs, dp as i64, lq_b as i64 * rs, rs, dp as i64, scale,
                s(st)))?;
            return k!(gpu_sd_unpad_heads(op.ptr(), out, (n * lq) as i64, heads as i32, hd as i32, dp as i32, s(st)));
        }
        // exact fp16 attention: scores in float32, probabilities in fp16, a block of queries at a time
        let chunk = ((256usize << 20) / (lk * 6)).clamp(64, 4096).min(lq);
        let sc = DevBuf::alloc_async(chunk * lk * 4, st)?;
        let pr = DevBuf::alloc_async(chunk * lk * 2, st)?;
        for i in 0..n {
            let (kb, vb) = if shared { (k_, v) } else { (off(k_, i * lk * kv_rs * 2), off(v, i * lk * kv_rs * 2)) };
            for h in 0..heads {
                for r0 in (0..lq).step_by(chunk) {
                    let rows = chunk.min(lq - r0);
                    let qr = off(q, ((i * lq + r0) * q_rs + h * hd) * 2);
                    k!(gpu_gemm(qr, q_rs as i64, off(kb, h * hd * 2), kv_rs as i64, sc.ptr(), lk as i64, null(), rows as i64, lk as i64, hd as i64, 1,
                        DT_F16, DT_F32, 1.0, 0.0, s(st)))?;
                    k!(gpu_sd_softmax(sc.ptr(), pr.ptr(), rows as i32, lk as i32, scale, s(st)))?;
                    k!(gpu_gemm(pr.ptr(), lk as i64, off(vb, h * hd * 2), kv_rs as i64, off(out, ((i * lq + r0) * inner + h * hd) * 2), inner as i64, null(),
                        rows as i64, hd as i64, lk as i64, 0, DT_F16, DT_F16, 1.0, 0.0, s(st)))?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn res(&self, key: &str, x: DevBuf, emb: &DevBuf, n: usize, h: usize, w: usize, cin: usize, cout: usize, skip: bool, st: &Stream) -> Result<DevBuf> {
        let p = h * w;
        let alloc = |c: usize| DevBuf::alloc_async(n * p * c * 2, st);
        let t = alloc(cin)?;
        self.group_norm(&format!("{key}.n1"), x.ptr(), t.ptr(), n, p, cin, RES_EPS, true, st)?;
        let u = alloc(cout)?;
        self.conv(&format!("{key}.c1"), t.ptr(), u.ptr(), n, h, w, cin, cout, 3, 1, st)?;
        drop(t);
        let e = DevBuf::alloc_async(n * cout * 2, st)?;
        self.dense(&format!("{key}.emb"), emb.ptr(), n, cout, self.time_dim(), true, e.ptr(), st)?;
        k!(gpu_sd_add_emb(u.ptr(), e.ptr(), n as i32, p as i64, cout as i32, s(st)))?;
        let t2 = alloc(cout)?;
        self.group_norm(&format!("{key}.n2"), u.ptr(), t2.ptr(), n, p, cout, RES_EPS, true, st)?;
        self.conv(&format!("{key}.c2"), t2.ptr(), u.ptr(), n, h, w, cout, cout, 3, 1, st)?;
        drop(t2);
        let residual = if skip {
            let r = alloc(cout)?;
            self.conv(&format!("{key}.skip"), x.ptr(), r.ptr(), n, h, w, cin, cout, 1, 1, st)?;
            r
        } else {
            x
        };
        // skip_connection(x) + h
        k!(gpu_sd_add(u.ptr(), residual.ptr(), (n * p * cout) as i64, s(st)))?;
        Ok(u)
    }

    fn time_dim(&self) -> usize {
        self.model_channels * 4
    }

    #[allow(clippy::too_many_arguments)]
    fn transformer(&self, key: &str, x: &DevBuf, ctx: &Context, n: usize, p: usize, c: usize, depth: usize, heads: usize, hd: usize, st: &Stream)
        -> Result<()> {
        let m = n * p;
        let alloc = |cols: usize| DevBuf::alloc_async(m * cols * 2, st);
        let nrm = alloc(c)?;
        self.group_norm(&format!("{key}.norm"), x.ptr(), nrm.ptr(), n, p, c, ST_EPS, false, st)?;
        let t = alloc(c)?;
        self.linear(&format!("{key}.pin"), nrm.ptr(), m, t.ptr(), st)?;
        let ln = |name: &str, out: &DevBuf| -> Result<()> {
            k!(gpu_sd_layer_norm(DT_F16, t.ptr(), c as i64, out.ptr(), c as i64, self.w(&format!("{name}.w")), self.w(&format!("{name}.b")), m as i32,
                c as i32, LN_EPS, s(st)))
        };
        for b in 0..depth {
            let tk = format!("{key}.transformer_blocks.{b}");
            // self-attention
            ln(&format!("{tk}.norm1"), &nrm)?;
            let qkv = alloc(3 * c)?;
            self.linear(&format!("{tk}.qkv"), nrm.ptr(), m, qkv.ptr(), st)?;
            let att = alloc(c)?;
            self.attention(qkv.ptr(), 3 * c, off(qkv.ptr(), c * 2), off(qkv.ptr(), 2 * c * 2), 3 * c, n, p, p, false, heads, hd, att.ptr(), st)?;
            drop(qkv);
            self.linear(&format!("{tk}.o1"), att.ptr(), m, nrm.ptr(), st)?;
            k!(gpu_sd_add(t.ptr(), nrm.ptr(), (m * c) as i64, s(st)))?;
            // cross-attention
            ln(&format!("{tk}.norm2"), &nrm)?;
            let q = alloc(c)?;
            self.linear(&format!("{tk}.q2"), nrm.ptr(), m, q.ptr(), st)?;
            let kv = ctx.kv.get(&tk).context("context of another model")?;
            self.attention(q.ptr(), c, kv.ptr(), off(kv.ptr(), c * 2), 2 * c, n, p, ctx.len, true, heads, hd, att.ptr(), st)?;
            drop(q);
            self.linear(&format!("{tk}.o2"), att.ptr(), m, nrm.ptr(), st)?;
            k!(gpu_sd_add(t.ptr(), nrm.ptr(), (m * c) as i64, s(st)))?;
            drop(att);
            // GEGLU feed-forward
            ln(&format!("{tk}.norm3"), &nrm)?;
            let inner2 = self.linears[&format!("{tk}.ff1")].n;
            let f = alloc(inner2)?;
            self.linear(&format!("{tk}.ff1"), nrm.ptr(), m, f.ptr(), st)?;
            let g = alloc(inner2 / 2)?;
            k!(gpu_sd_geglu(f.ptr(), g.ptr(), m as i64, (inner2 / 2) as i32, s(st)))?;
            drop(f);
            self.linear(&format!("{tk}.ff2"), g.ptr(), m, nrm.ptr(), st)?;
            // ff(norm3(x)) + x
            k!(gpu_sd_add(t.ptr(), nrm.ptr(), (m * c) as i64, s(st)))?;
        }
        self.linear(&format!("{key}.pout"), t.ptr(), m, nrm.ptr(), st)?;
        k!(gpu_sd_add(x.ptr(), nrm.ptr(), (m * c) as i64, s(st)))
    }

    /// `up_to` is the size an upsample in these ops makes: the next skip connection's, as ComfyUI's `output_shape`.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, ops: &[Op], mut x: DevBuf, emb: &DevBuf, ctx: &Context, n: usize, hw: &mut (usize, usize), c: &mut usize,
        up_to: Option<(usize, usize)>, st: &Stream) -> Result<DevBuf> {
        for op in ops {
            let (h, w) = *hw;
            match op {
                Op::ConvIn => bail!("the first convolution in another block"),
                Op::Res { key, cin, cout, skip } => {
                    debug_assert_eq!(*cin, *c);
                    x = self.res(key, x, emb, n, h, w, *cin, *cout, *skip, st)?;
                    *c = *cout;
                }
                Op::Transformer { key, c: ct, depth, heads, hd } => self.transformer(key, &x, ctx, n, h * w, *ct, *depth, *heads, *hd, st)?,
                Op::Down { key, c: cd } => {
                    // stride 2 with padding 1 rounds odd sides up
                    let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
                    let y = DevBuf::alloc_async(n * oh * ow * cd * 2, st)?;
                    self.conv(&format!("{key}.down"), x.ptr(), y.ptr(), n, h, w, *cd, *cd, 3, 2, st)?;
                    x = y;
                    *hw = (oh, ow);
                }
                Op::Up { key, c: cu } => {
                    let (oh, ow) = up_to.unwrap_or((2 * h, 2 * w));
                    let up = DevBuf::alloc_async(n * oh * ow * cu * 2, st)?;
                    k!(gpu_sd_upsample(x.ptr(), up.ptr(), n as i32, h as i32, w as i32, oh as i32, ow as i32, *cu as i32, s(st)))?;
                    let y = DevBuf::alloc_async(n * oh * ow * cu * 2, st)?;
                    self.conv(&format!("{key}.up"), up.ptr(), y.ptr(), n, oh, ow, *cu, *cu, 3, 1, st)?;
                    x = y;
                    *hw = (oh, ow);
                }
            }
        }
        Ok(x)
    }

    /// The denoised prediction of `n` latents `x` (float32 [n, 4, h, w]) at `sigma` and its `timestep`, into `out`.
    /// `adm` is SDXL's size conditioning, float32 [adm] for every image. `v_pred` reads the output as v-prediction.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&self, x: &DevBuf, n: usize, h: usize, w: usize, ctx: &Context, adm: Option<&[f32]>, sigma: f32, timestep: f32, v_pred: bool,
        out: &DevBuf, st: &Stream) -> Result<()> {
        if h == 0 || w == 0 {
            bail!("an empty latent");
        }
        let mc = self.model_channels;
        let td = self.time_dim();
        // the timestep's embedding, through time_embed
        let te = DevBuf::alloc_async(n * mc * 2, st)?;
        te.upload(0, &f16_bytes(&timestep_embedding(timestep, mc).repeat(n)), st)?;
        let e1 = DevBuf::alloc_async(n * td * 2, st)?;
        self.dense("t0", te.ptr(), n, td, mc, true, e1.ptr(), st)?;
        k!(gpu_sd_silu(e1.ptr(), (n * td) as i64, s(st)))?;
        let emb = DevBuf::alloc_async(n * td * 2, st)?;
        self.dense("t2", e1.ptr(), n, td, td, true, emb.ptr(), st)?;
        if self.adm > 0 {
            let y = adm.context("SDXL needs its size conditioning")?;
            if y.len() != self.adm {
                bail!("size conditioning of {} values, {} expected", y.len(), self.adm);
            }
            let yb = DevBuf::alloc_async(n * self.adm * 2, st)?;
            yb.upload(0, &f16_bytes(&y.repeat(n)), st)?;
            self.dense("l0", yb.ptr(), n, td, self.adm, true, e1.ptr(), st)?;
            k!(gpu_sd_silu(e1.ptr(), (n * td) as i64, s(st)))?;
            let l = DevBuf::alloc_async(n * td * 2, st)?;
            self.dense("l2", e1.ptr(), n, td, td, true, l.ptr(), st)?;
            k!(gpu_sd_add(emb.ptr(), l.ptr(), (n * td) as i64, s(st)))?;
        }
        // every ResBlock reads SiLU(emb)
        k!(gpu_sd_silu(emb.ptr(), (n * td) as i64, s(st)))?;

        let sig = DevBuf::alloc_async(n * 4, st)?;
        sig.upload(0, &f32_bytes(&vec![sigma; n]), st)?;
        let p = h * w;
        let xin = DevBuf::alloc_async(n * p * CH_PAD * 2, st)?;
        k!(gpu_sd_unet_in(x.ptr(), xin.ptr(), n as i32, CH as i32, p as i64, CH_PAD as i32, sig.ptr(), s(st)))?;
        let mut hx = DevBuf::alloc_async(n * p * mc * 2, st)?;
        self.conv("in", xin.ptr(), hx.ptr(), n, h, w, CH_PAD, mc, 3, 1, st)?;
        drop(xin);
        let mut hw = (h, w);
        let mut c = mc;
        // each skip connection with its channels and size
        let mut skips: Vec<(DevBuf, usize, (usize, usize))> = Vec::new();
        let copy = |b: &DevBuf, bytes: usize| -> Result<DevBuf> {
            let d = DevBuf::alloc_async(bytes, st)?;
            crate::cuda::memcpy_d2d(d.ptr(), b.ptr(), bytes, st)?;
            Ok(d)
        };
        skips.push((copy(&hx, n * p * mc * 2)?, mc, hw));
        for ops in &self.input[1..] {
            hx = self.run(ops, hx, &emb, ctx, n, &mut hw, &mut c, None, st)?;
            skips.push((copy(&hx, n * hw.0 * hw.1 * c * 2)?, c, hw));
        }
        hx = self.run(&self.middle, hx, &emb, ctx, n, &mut hw, &mut c, None, st)?;
        for ops in &self.output {
            let (skip, cs, shw) = skips.pop().context("skips")?;
            if shw != hw {
                bail!("a skip connection of {}x{} against {}x{}", shw.1, shw.0, hw.1, hw.0);
            }
            let rows = n * hw.0 * hw.1;
            let cat = DevBuf::alloc_async(rows * (c + cs) * 2, st)?;
            memcpy_2d_d2d(cat.ptr(), (c + cs) * 2, hx.ptr(), c * 2, c * 2, rows, st)?;
            memcpy_2d_d2d(off(cat.ptr(), c * 2), (c + cs) * 2, skip.ptr(), cs * 2, cs * 2, rows, st)?;
            c += cs;
            let up_to = skips.last().map(|s| s.2);
            hx = self.run(ops, cat, &emb, ctx, n, &mut hw, &mut c, up_to, st)?;
        }
        let t = DevBuf::alloc_async(n * p * c * 2, st)?;
        self.group_norm("out.n", hx.ptr(), t.ptr(), n, p, c, RES_EPS, true, st)?;
        let y = DevBuf::alloc_async(n * p * CH_PAD * 2, st)?;
        self.conv("out.c", t.ptr(), y.ptr(), n, h, w, c, CH_PAD, 3, 1, st)?;
        k!(gpu_sd_unet_out(y.ptr(), x.ptr(), out.ptr(), n as i32, CH as i32, p as i64, CH_PAD as i32, sig.ptr(), v_pred as i32, s(st)))
    }

    /// Rough activation memory of one image through the model (bytes), at an h x w latent.
    pub fn activation_bytes(&self, h: usize, w: usize) -> usize {
        let p = h * w;
        // skips and the largest block's temporaries at full size, and the attention at the first transformer's level
        p * self.model_channels * 2 * 24 + (256 << 20)
    }
}

/// True for a ComfyUI `v_pred` marker key in a checkpoint (v-prediction SDXL and SD 2 models carry it).
pub fn is_v_pred(st: &SafeTensors) -> bool {
    st.has("v_pred")
}
