//! Flux 2 VAE decoder for MOSS Image's 128-channel latents, NHWC bf16.

use crate::cuda::{DevBuf, Stream};
use crate::ffi::{k, null, s, PM};
use crate::offload::Layout;
use crate::st::{f32_bytes, SafeTensors};
use crate::vae::{bias_pad, conv_krsc, single_head_attention};
use anyhow::Result;
use std::path::Path;

const OUT_PAD: usize = 8;
const GROUPS: usize = 32;
const EPS: f32 = 1e-6;
const BN_EPS: f32 = 1e-4;
const LEVELS: usize = 4;
const BLOCKS: usize = 3;
const Z: usize = 32;
pub const CH: usize = 128;

pub struct Vae {
    buf: DevBuf,
    lay: Layout,
    /// (out channels, in channels of the first block) of each level
    levels: Vec<(usize, usize)>,
    block_in: usize,
}

impl Vae {
    pub fn load(path: &Path, s0: &Stream) -> Result<Self> {
        let st = SafeTensors::open(path)?;
        let mut lay = Layout::new();
        let conv = |lay: &mut Layout, key: &str, name: &str, pad: usize| -> Result<()> {
            let (w, cop, _, _) = conv_krsc(&st, &format!("{name}.weight"), pad)?;
            lay.bytes(&format!("{key}.w"), w);
            lay.bytes(&format!("{key}.b"), bias_pad(&st, &format!("{name}.bias"), cop)?);
            Ok(())
        };
        let norm = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            lay.bf16(&st, &format!("{key}.w"), &[format!("{name}.weight")])?;
            lay.bf16(&st, &format!("{key}.b"), &[format!("{name}.bias")])
        };
        let res = |lay: &mut Layout, key: &str, name: &str| -> Result<()> {
            norm(lay, &format!("{key}.n1"), &format!("{name}.norm1"))?;
            norm(lay, &format!("{key}.n2"), &format!("{name}.norm2"))?;
            conv(lay, &format!("{key}.c1"), &format!("{name}.conv1"), 0)?;
            conv(lay, &format!("{key}.c2"), &format!("{name}.conv2"), 0)?;
            for sc in ["nin_shortcut", "conv_shortcut"] {
                if st.has(&format!("{name}.{sc}.weight")) {
                    conv(lay, &format!("{key}.sc"), &format!("{name}.{sc}"), 0)?;
                }
            }
            Ok(())
        };
        // Published copies of this VAE use either ComfyUI's names or diffusers'.
        let diffusers = st.has("decoder.mid_block.resnets.0.conv1.weight");
        let mid = |i: usize| if diffusers { format!("decoder.mid_block.resnets.{i}") } else { format!("decoder.mid.block_{}", i + 1) };
        let attn = |comfy: &str, diff: &str| {
            if diffusers { format!("decoder.mid_block.attentions.0.{diff}") } else { format!("decoder.mid.attn_1.{comfy}") }
        };
        let up = |i: usize| if diffusers { format!("decoder.up_blocks.{}", LEVELS - 1 - i) } else { format!("decoder.up.{i}") };
        let up_block = |i: usize, b: usize| if diffusers { format!("{}.resnets.{b}", up(i)) } else { format!("{}.block.{b}", up(i)) };
        let pq = if st.has("decoder.post_quant_conv.weight") { "decoder.post_quant_conv" } else { "post_quant_conv" };
        conv(&mut lay, "pq", pq, 0)?;
        conv(&mut lay, "in", "decoder.conv_in", 0)?;
        res(&mut lay, "m1", &mid(0))?;
        res(&mut lay, "m2", &mid(1))?;
        norm(&mut lay, "attn.n", &attn("norm", "group_norm"))?;
        let (q, k_, v, o) = (attn("q", "to_q"), attn("k", "to_k"), attn("v", "to_v"), attn("proj_out", "to_out.0"));
        lay.bf16(&st, "attn.qkv.w", &[format!("{q}.weight"), format!("{k_}.weight"), format!("{v}.weight")])?;
        lay.bf16(&st, "attn.qkv.b", &[format!("{q}.bias"), format!("{k_}.bias"), format!("{v}.bias")])?;
        lay.bf16(&st, "attn.proj.w", &[format!("{o}.weight")])?;
        lay.bf16(&st, "attn.proj.b", &[format!("{o}.bias")])?;
        let mut levels = vec![(0, 0); LEVELS];
        let block_in = st.info("decoder.conv_in.weight")?.shape[0];
        for i in (0..LEVELS).rev() {
            let first = st.info(&format!("{}.conv1.weight", up_block(i, 0)))?;
            levels[i] = (first.shape[0], first.shape[1]);
            for b in 0..BLOCKS {
                res(&mut lay, &format!("u{i}.{b}"), &up_block(i, b))?;
            }
            if i != 0 {
                let name = if diffusers { format!("{}.upsamplers.0.conv", up(i)) } else { format!("{}.upsample.conv", up(i)) };
                conv(&mut lay, &format!("u{i}.up"), &name, 0)?;
            }
        }
        norm(&mut lay, "out.n", if diffusers { "decoder.conv_norm_out" } else { "decoder.norm_out" })?;
        conv(&mut lay, "out", "decoder.conv_out", OUT_PAD)?;
        let m = st.f32_vec("bn.running_mean", false)?;
        let sd: Vec<f32> = st.f32_vec("bn.running_var", false)?.iter().map(|v| (v + BN_EPS).sqrt()).collect();
        lay.bytes("mean", f32_bytes(&m));
        lay.bytes("std", f32_bytes(&sd));
        let mut host = vec![0u8; lay.size];
        lay.pack(&st, &mut host)?;
        let buf = DevBuf::alloc(lay.size)?;
        buf.upload(0, &host, s0)?;
        s0.sync()?;
        Ok(Vae { buf, lay, levels, block_in })
    }

    fn w(&self, key: &str) -> PM {
        self.buf.at(self.lay.off(key))
    }

    fn conv(&self, key: &str, x: PM, out: PM, h: usize, w: usize, cin: usize, cout: usize, r: usize, st: &Stream) -> Result<()> {
        k!(krea_conv2d(x, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), out, 1, h as i32, w as i32, cin as i32, cout as i32,
            r as i32, s(st)))
    }

    fn norm(&self, key: &str, x: PM, out: PM, stats: PM, p: usize, c: usize, silu: bool, st: &Stream) -> Result<()> {
        k!(krea_group_norm(x, out, self.w(&format!("{key}.w")), self.w(&format!("{key}.b")), stats, p as i64, c as i32, GROUPS as i32, EPS,
            silu as i32, s(st)))
    }

    /// Peak activation memory (bytes) for decoding an h x w latent.
    pub fn activation_bytes(h: usize, w: usize) -> usize {
        let full = h * 16 * w * 16;
        full * CH * 2 * 3 + full * 512 * 2 / 2 + (512 << 20)
    }

    fn res(&self, key: &str, x: DevBuf, h: usize, w: usize, cin: usize, cout: usize, stats: PM, st: &Stream) -> Result<DevBuf> {
        let p = h * w;
        let alloc = |c: usize| DevBuf::alloc_async(p * c * 2, st);
        let t = alloc(cin)?;
        self.norm(&format!("{key}.n1"), x.ptr(), t.ptr(), stats, p, cin, true, st)?;
        let u = alloc(cout)?;
        self.conv(&format!("{key}.c1"), t.ptr(), u.ptr(), h, w, cin, cout, 3, st)?;
        drop(t);
        let t2 = alloc(cout)?;
        self.norm(&format!("{key}.n2"), u.ptr(), t2.ptr(), stats, p, cout, true, st)?;
        self.conv(&format!("{key}.c2"), t2.ptr(), u.ptr(), h, w, cout, cout, 3, st)?;
        drop(t2);
        let residual = if cin != cout {
            let sc = alloc(cout)?;
            self.conv(&format!("{key}.sc"), x.ptr(), sc.ptr(), h, w, cin, cout, 1, st)?;
            sc
        } else {
            x
        };
        k!(krea_add_gated_bf16(u.ptr(), cout as i64, residual.ptr(), cout as i64, null(), p as i32, cout as i32, s(st)))?;
        Ok(u)
    }

    /// latent f32 [128, h, w] -> RGB8 [16h, 16w, 3]
    pub fn decode(&self, latent: &DevBuf, h: usize, w: usize, st: &Stream) -> Result<Vec<u8>> {
        let (mut hh, mut ww) = (h * 2, w * 2);
        let alloc = |c: usize, a: usize, b: usize| DevBuf::alloc_async(a * b * c * 2, st);
        let stats = DevBuf::alloc_async(GROUPS * 8, st)?;
        let sp = stats.ptr();
        let z = alloc(Z, hh, ww)?;
        k!(krea_flux2_latent_in(latent.ptr(), self.w("mean"), self.w("std"), z.ptr(), CH as i32, h as i32, w as i32, s(st)))?;
        let z2 = alloc(Z, hh, ww)?;
        self.conv("pq", z.ptr(), z2.ptr(), hh, ww, Z, Z, 1, st)?;
        let mut c = self.block_in;
        let mut x = alloc(c, hh, ww)?;
        self.conv("in", z2.ptr(), x.ptr(), hh, ww, Z, c, 3, st)?;
        drop((z, z2));

        x = self.res("m1", x, hh, ww, c, c, sp, st)?;
        let p = hh * ww;
        let n = alloc(c, hh, ww)?;
        self.norm("attn.n", x.ptr(), n.ptr(), sp, p, c, false, st)?;
        single_head_attention(&x, &n, p, c, [self.w("attn.qkv.w"), self.w("attn.qkv.b"), self.w("attn.proj.w"), self.w("attn.proj.b")], st)?;
        drop(n);
        x = self.res("m2", x, hh, ww, c, c, sp, st)?;

        for i in (0..LEVELS).rev() {
            let (cout, cin) = self.levels[i];
            for b in 0..BLOCKS {
                let from = if b == 0 { cin } else { cout };
                debug_assert_eq!(from, c);
                x = self.res(&format!("u{i}.{b}"), x, hh, ww, from, cout, sp, st)?;
                c = cout;
            }
            if i != 0 {
                let up = alloc(c, 2 * hh, 2 * ww)?;
                k!(krea_upsample2x(x.ptr(), up.ptr(), hh as i32, ww as i32, c as i32, s(st)))?;
                hh *= 2;
                ww *= 2;
                let y = alloc(c, hh, ww)?;
                self.conv(&format!("u{i}.up"), up.ptr(), y.ptr(), hh, ww, c, c, 3, st)?;
                x = y;
            }
        }
        let p = hh * ww;
        let t = alloc(c, hh, ww)?;
        self.norm("out.n", x.ptr(), t.ptr(), sp, p, c, true, st)?;
        let o = alloc(OUT_PAD, hh, ww)?;
        self.conv("out", t.ptr(), o.ptr(), hh, ww, c, OUT_PAD, 3, st)?;
        let rgb = DevBuf::alloc_async(p * 3, st)?;
        k!(krea_to_rgb8(o.ptr(), OUT_PAD as i32, rgb.ptr(), p as i64, s(st)))?;
        let mut out = vec![0u8; p * 3];
        rgb.download(0, &mut out, st)?;
        Ok(out)
    }
}
