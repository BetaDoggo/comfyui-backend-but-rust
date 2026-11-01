//! ControlNet One/Two/Three: classic ControlNets (with union types and AliMama inpainting),
//! DiffSynth / Anima LLLite model patches, and the control-image resize SwarmUI adds after a
//! preprocessor.

use swarm_engine_api::{AddonKind, Conditioning, ControlHint, ImageBatch, Mask, ModelPatch};

use super::{input_err, NodeError, NodeInputs, NodeOutput, NodeRegistry, NodeResult};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

const UNION_TYPES: [&str; 9] = ["auto", "openpose", "depth", "hed/pidi/scribble/ted", "canny/lineart/anime_lineart/mlsd", "normal", "segment", "tile", "repaint"];
const SCALE_METHODS: [&str; 5] = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];

/// ComfyUI's `ControlNetApplyAdvanced`: the same hint on every positive and negative entry
/// (strength 0 is a no-op).
fn apply_control(i: &NodeInputs, inpaint_mask: Option<Mask>) -> NodeResult {
    let (positive, negative) = (i.conditioning("positive")?, i.conditioning("negative")?);
    let strength = i.float("strength")?;
    if strength == 0.0 {
        return Ok(NodeOutput::values(vec![Value::Conditioning(positive.clone()), Value::Conditioning(negative.clone())]));
    }
    let hint = ControlHint {
        controlnet: i.addon("control_net", AddonKind::ControlNet)?.clone(),
        image: i.image("image")?.to_rgb(),
        strength,
        start_percent: i.float("start_percent")?,
        end_percent: i.float("end_percent")?,
        vae: i.opt_vae("vae")?.cloned(),
        inpaint_mask,
    };
    let add = |c: &Conditioning| Value::Conditioning(c.map_meta(|m| m.controls.push(hint.clone())));
    Ok(NodeOutput::values(vec![add(positive), add(negative)]))
}

fn control_inputs(s: NodeSchema, advanced_range: bool) -> NodeSchema {
    let (start, end) = (float("start_percent", 0.0, 0.0, 1.0, 0.001), float("end_percent", 1.0, 0.0, 1.0, 0.001));
    let (start, end) = if advanced_range { (start.advanced(), end.advanced()) } else { (start, end) };
    s.input(float("strength", 1.0, 0.0, 10.0, 0.01)).input(start).input(end).output("CONDITIONING", "positive").output("CONDITIONING", "negative")
}

/// An IMAGE or MASK batch as `[B][H][W][C]` floats (masks have one channel).
fn as_frames(v: &Value) -> Result<ImageBatch, NodeError> {
    match v {
        Value::Image(b) => Ok(b.clone()),
        Value::Mask(m) => Ok(ImageBatch::from_f32(m.count, m.height, m.width, 1, m.data.to_vec())),
        other => Err(input_err(format!("expected IMAGE or MASK, got {}", other.type_name()))),
    }
}

fn like(input: &Value, b: ImageBatch) -> Value {
    match input {
        Value::Mask(_) => {
            let data: Vec<f32> = (0..b.count).flat_map(|i| b.frame_f32(i).into_owned()).collect();
            Value::Mask(Mask { count: b.count, height: b.height, width: b.width, data: std::sync::Arc::new(data) })
        }
        _ => Value::Image(b),
    }
}

/// `x0..x0+w`, `y0..y0+h` of every frame.
fn crop(b: &ImageBatch, x0: u32, y0: u32, w: u32, h: u32) -> ImageBatch {
    let c = b.channels as usize;
    let mut out = Vec::with_capacity((b.count * w * h) as usize * c);
    for i in 0..b.count {
        let f = b.frame_f32(i);
        for y in y0..y0 + h {
            let row = (y * b.width + x0) as usize * c;
            out.extend_from_slice(&f[row..row + w as usize * c]);
        }
    }
    ImageBatch::from_f32(b.count, h, w, b.channels, out)
}

/// ComfyUI's `ResizeImageMaskNode` (`comfy_extras/nodes_post_processing.py`).
fn resize_image_mask(i: &NodeInputs) -> Result<Value, NodeError> {
    let input = i.get("input").ok_or_else(|| input_err("required input 'input' is missing"))?;
    let b = as_frames(input)?;
    let method = i.string("scale_method")?;
    let (w, h) = (b.width as f64, b.height as f64);
    let scaled = |tw: f64, th: f64, crop_mode: &str| -> Result<ImageBatch, NodeError> {
        Ok(encode::resize(&b, (tw.round() as u32).max(1), (th.round() as u32).max(1), method, crop_mode)?)
    };
    let sub = |k: &str| format!("resize_type.{k}");
    let out = match i.string("resize_type")? {
        "scale by multiplier" => {
            let m = i.float(&sub("multiplier"))?;
            scaled(w * m, h * m, "disabled")?
        }
        mode @ ("scale dimensions" | "scale width" | "scale height") => {
            let tw = if mode == "scale height" { 0.0 } else { i.float(&sub("width"))? };
            let th = if mode == "scale width" { 0.0 } else { i.float(&sub("height"))? };
            let crop_mode = if mode == "scale dimensions" { i.string(&sub("crop"))?.to_string() } else { "disabled".into() };
            match (tw == 0.0, th == 0.0) {
                (true, true) => b,
                (true, false) => scaled((w * th / h).round().max(1.0), th, &crop_mode)?,
                (false, true) => scaled(tw, (h * tw / w).round().max(1.0), &crop_mode)?,
                (false, false) => scaled(tw, th, &crop_mode)?,
            }
        }
        mode @ ("scale longer dimension" | "scale shorter dimension") => {
            let longer = mode == "scale longer dimension";
            let size = i.float(&sub(if longer { "longer_size" } else { "shorter_size" }))?;
            let (tw, th) = if (h > w) == longer && h != w {
                ((w / h * size).round(), size)
            } else if h != w {
                (size, (h / w * size).round())
            } else {
                (size, size)
            };
            scaled(tw, th, "disabled")?
        }
        "scale total pixels" => {
            let total = (i.float(&sub("megapixels"))? * 1024.0 * 1024.0).trunc();
            let s = (total / (w * h)).sqrt();
            scaled(w * s, h * s, "disabled")?
        }
        "match size" => {
            let m = as_frames(i.get(&sub("match")).ok_or_else(|| input_err("resize_type.match is missing"))?)?;
            scaled(m.width as f64, m.height as f64, i.string(&sub("crop"))?)?
        }
        "scale to multiple" => {
            // Scale to cover the largest multiple-of-N size, then center-crop to it.
            let n = i.int(&sub("multiple"))?.max(1) as u32;
            let (tw, th) = (b.width / n * n, b.height / n * n);
            if n <= 1 || tw == 0 || th == 0 || (tw == b.width && th == b.height) {
                b
            } else {
                let (sw, sh) = (tw as f64 / w, th as f64 / h);
                let (cw, ch) = if sw >= sh { (tw, ((h * sw).ceil() as u32).max(th)) } else { (((w * sh).ceil() as u32).max(tw), th) };
                let big = encode::resize(&b, cw, ch, method, "disabled")?;
                crop(&big, (cw - tw) / 2, (ch - th) / 2, tw, th)
            }
        }
        other => return Err(input_err(format!("Unsupported resize type: {other}"))),
    };
    Ok(like(input, out))
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            let s = NodeSchema::new("ControlNetApplyAdvanced", "Apply ControlNet", "model/conditioning/controlnet")
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("control_net", "CONTROL_NET"))
                .input(link("image", "IMAGE"));
            control_inputs(s, false).input(link("vae", "VAE").optional())
        },
        |i, _| apply_control(i, None),
    );
    r.add(
        env,
        |_| {
            let s = NodeSchema::new("ControlNetInpaintingAliMamaApply", "Apply ControlNet Inpainting (AliMama)", "model/conditioning/controlnet")
                .input(link("positive", "CONDITIONING"))
                .input(link("negative", "CONDITIONING"))
                .input(link("control_net", "CONTROL_NET"))
                .input(link("vae", "VAE"))
                .input(link("image", "IMAGE"))
                .input(link("mask", "MASK"));
            control_inputs(s, true)
        },
        |i, _| apply_control(i, Some(i.mask("mask")?.clone())),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SetUnionControlNetType", "Set Union ControlNet Type", "model/conditioning/controlnet")
                .input(link("control_net", "CONTROL_NET"))
                .input(combo("type", &UNION_TYPES))
                .output("CONTROL_NET", "CONTROL_NET")
        },
        |i, _| {
            let mut cn = i.addon("control_net", AddonKind::ControlNet)?.clone();
            let ty = i.string("type")?;
            cn.control_type = (ty != "auto" && UNION_TYPES.contains(&ty)).then(|| ty.to_string());
            Ok(NodeOutput::one(Value::Addon(cn)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("QwenImageDiffsynthControlnet", "Apply Qwen Image DiffSynth ControlNet", "model/patch/qwen")
                .input(link("model", "MODEL"))
                .input(link("model_patch", "MODEL_PATCH"))
                .input(link("vae", "VAE"))
                .input(link("image", "IMAGE"))
                .input(float("strength", 1.0, -10.0, 10.0, 0.01))
                .input(link("mask", "MASK").optional())
                .input(float("start_percent", 0.0, 0.0, 1.0, 0.001).optional().advanced())
                .input(float("end_percent", 1.0, 0.0, 1.0, 0.001).optional().advanced())
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::DiffSynthControl {
                patch: i.addon("model_patch", AddonKind::ModelPatch)?.clone(),
                vae: i.vae("vae")?.clone(),
                image: i.image("image")?.to_rgb(),
                strength: i.float("strength")?,
                mask: i.opt_mask("mask")?.cloned(),
                start_percent: i.float("start_percent").unwrap_or(0.0),
                end_percent: i.float("end_percent").unwrap_or(1.0),
            };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(p))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("AnimaLLLiteApply", "Apply Anima LLLite", "model/patch/anima")
                .input(link("model", "MODEL"))
                .input(link("model_patch", "MODEL_PATCH"))
                .input(link("image", "IMAGE"))
                .input(float("strength", 1.0, -10.0, 10.0, 0.01))
                .input(float("start_percent", 0.0, 0.0, 1.0, 0.001))
                .input(float("end_percent", 1.0, 0.0, 1.0, 0.001))
                .input(link("mask", "MASK").optional())
                .output("MODEL", "MODEL")
        },
        |i, _| {
            let p = ModelPatch::AnimaLLLite {
                patch: i.addon("model_patch", AddonKind::ModelPatch)?.clone(),
                image: i.image("image")?.to_rgb(),
                strength: i.float("strength")?,
                mask: i.opt_mask("mask")?.cloned(),
                start_percent: i.float("start_percent")?,
                end_percent: i.float("end_percent")?,
            };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(p))))
        },
    );
    r.add(
        env,
        |_| {
            let tip_crop = "How to handle aspect ratio mismatch: 'disabled' stretches to fit, 'center' crops to maintain aspect ratio.";
            let crop = || combo_default("crop", &["disabled", "center"], "center").tooltip(tip_crop);
            let px = |n: &str, tip: &str| int_step(n, 512, 0, 16384, 1).tooltip(tip);
            NodeSchema::new("ResizeImageMaskNode", "Resize Image/Mask", "image/transform")
                .input(match_type("input", "input_type", "IMAGE,MASK"))
                .input(
                    dynamic_combo(
                        "resize_type",
                        vec![
                            (
                                "scale dimensions",
                                vec![
                                    px("width", "Target width in pixels. Set to 0 to auto-calculate from height while preserving aspect ratio."),
                                    px("height", "Target height in pixels. Set to 0 to auto-calculate from width while preserving aspect ratio."),
                                    crop(),
                                ],
                            ),
                            ("scale by multiplier", vec![float("multiplier", 1.0, 0.01, 8.0, 0.01).tooltip("Scale factor (e.g., 2.0 doubles size, 0.5 halves size).")]),
                            ("scale longer dimension", vec![px("longer_size", "The longer edge will be resized to this value. Aspect ratio is preserved.")]),
                            ("scale shorter dimension", vec![px("shorter_size", "The shorter edge will be resized to this value. Aspect ratio is preserved.")]),
                            ("scale width", vec![px("width", "Target width in pixels. Height auto-adjusts to preserve aspect ratio.")]),
                            ("scale height", vec![px("height", "Target height in pixels. Width auto-adjusts to preserve aspect ratio.")]),
                            ("scale total pixels", vec![float("megapixels", 1.0, 0.01, 16.0, 0.01).tooltip("Target total megapixels (e.g., 1.0 \u{2248} 1024\u{d7}1024). Aspect ratio is preserved.")]),
                            ("match size", vec![link("match", "IMAGE,MASK").tooltip("Resize input to match the dimensions of this reference image or mask."), crop()]),
                            (
                                "scale to multiple",
                                vec![int_step("multiple", 8, 1, 16384, 1).tooltip("Resize so width and height are divisible by this number. Useful for latent alignment (e.g., 8 or 64).")],
                            ),
                        ],
                    )
                    .tooltip("Select how to resize: by exact dimensions, scale factor, matching another image, etc."),
                )
                .input(
                    combo_default("scale_method", &SCALE_METHODS, "area")
                        .tooltip("Interpolation algorithm. 'area' is best for downscaling, 'lanczos' for upscaling, 'nearest-exact' for pixel art."),
                )
                .output(MATCH_TYPE, "resized")
        },
        |i, _| Ok(NodeOutput::one(resize_image_mask(i)?)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_takes_the_window() {
        let b = ImageBatch::from_f32(1, 2, 3, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let c = crop(&b, 1, 1, 2, 1);
        assert_eq!(c.frame_f32(0).as_ref(), [4.0, 5.0]);
    }
}
