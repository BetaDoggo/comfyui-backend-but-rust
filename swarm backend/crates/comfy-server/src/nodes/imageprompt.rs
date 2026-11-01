//! Image prompting: CLIP-Vision encoding, style models (Use Style Model with its merge strength
//! and apply start), ReVision / unCLIP, Use Reference Only, and the conditioning transforms
//! they're built from.

use std::sync::Arc;

use swarm_engine_api::{AddonKind, ClipVisionEncodeRequest, Conditioning, EmbeddingOp, LatentOp, Mask, ModelPatch, StyleStrength, UnclipInput};

use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// ComfyUI's `ConditioningAverage` / `ConditioningConcat`: one op per `to` entry, using only the
/// first `from` entry.
fn with_from(to: &Conditioning, from: &Conditioning, op: impl Fn(Box<swarm_engine_api::CondEntry>) -> EmbeddingOp) -> Result<Conditioning, super::NodeError> {
    let first = from.entries.first().ok_or_else(|| input_err("conditioning_from is empty"))?;
    Ok(to.map_meta(|m| m.embedding_ops.push(op(Box::new(first.clone())))))
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("CLIPVisionEncode", "CLIP Vision Encode", "model/conditioning")
                .input(link("clip_vision", "CLIP_VISION"))
                .input(link("image", "IMAGE"))
                .input(combo("crop", &["center", "none"]))
                .output("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT")
        },
        |i, ctx| {
            let req = ClipVisionEncodeRequest {
                model: i.addon("clip_vision", AddonKind::ClipVision)?.clone(),
                image: i.image("image")?.clone(),
                crop: i.string("crop")?.to_string(),
            };
            Ok(NodeOutput::one(Value::ClipVision(ctx.engine.clip_vision_encode(&req)?)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("StyleModelApply", "Apply Style Model", "model/conditioning")
                .input(link("conditioning", "CONDITIONING"))
                .input(link("style_model", "STYLE_MODEL"))
                .input(link("clip_vision_output", "CLIP_VISION_OUTPUT"))
                .input(float("strength", 1.0, 0.0, 10.0, 0.001))
                .input(combo("strength_type", &["multiply", "attn_bias"]))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let op = EmbeddingOp::Style {
                model: i.addon("style_model", AddonKind::StyleModel)?.clone(),
                vision: i.clip_vision_output("clip_vision_output")?.clone(),
                strength: i.float("strength")?,
                strength_type: if i.string("strength_type")? == "attn_bias" { StyleStrength::AttnBias } else { StyleStrength::Multiply },
            };
            Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning")?.map_meta(|m| m.embedding_ops.push(op.clone())))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("unCLIPConditioning", "unCLIPConditioning", "model/conditioning")
                .input(link("conditioning", "CONDITIONING"))
                .input(link("clip_vision_output", "CLIP_VISION_OUTPUT"))
                .input(float("strength", 1.0, -10.0, 10.0, 0.01))
                .input(float("noise_augmentation", 0.0, 0.0, 1.0, 0.01))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let cond = i.conditioning("conditioning")?;
            let strength = i.float("strength")?;
            // ComfyUI returns the conditioning untouched at strength 0.
            if strength == 0.0 {
                return Ok(NodeOutput::one(Value::Conditioning(cond.clone())));
            }
            let u = UnclipInput { vision: i.clip_vision_output("clip_vision_output")?.clone(), strength, noise_augmentation: i.float("noise_augmentation")? };
            Ok(NodeOutput::one(Value::Conditioning(cond.map_meta(|m| m.unclip.push(u.clone())))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningAverage", "Conditioning (Average)", "model/conditioning/transform")
                .input(link("conditioning_to", "CONDITIONING"))
                .input(link("conditioning_from", "CONDITIONING"))
                .input(float("conditioning_to_strength", 1.0, 0.0, 1.0, 0.01))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let to_strength = i.float("conditioning_to_strength")?;
            let out = with_from(i.conditioning("conditioning_to")?, i.conditioning("conditioning_from")?, |from| EmbeddingOp::Average { from, to_strength })?;
            Ok(NodeOutput::one(Value::Conditioning(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningConcat", "Conditioning (Concat)", "model/conditioning/transform")
                .input(link("conditioning_to", "CONDITIONING"))
                .input(link("conditioning_from", "CONDITIONING"))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let out = with_from(i.conditioning("conditioning_to")?, i.conditioning("conditioning_from")?, |from| EmbeddingOp::Concat { from })?;
            Ok(NodeOutput::one(Value::Conditioning(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningSetTimestepRange", "ConditioningSetTimestepRange", "model/conditioning/transform")
                .input(link("conditioning", "CONDITIONING"))
                .input(float("start", 0.0, 0.0, 1.0, 0.001))
                .input(float("end", 1.0, 0.0, 1.0, 0.001))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let (start, end) = (i.float("start")?, i.float("end")?);
            Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning")?.map_meta(|m| {
                m.start_percent = start;
                m.end_percent = end;
            }))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmReferenceOnly", "SwarmReferenceOnly", "SwarmUI/sampling")
                .describe("Applies 'reference only' image-prompting to the generation. Must forward the new model, and the new latent, to work properly.")
                .input(link("model", "MODEL"))
                .input(link("reference", "LATENT"))
                .input(link("latent", "LATENT"))
                .output("MODEL", "MODEL")
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let (reference, latent) = (i.latent("reference")?, i.latent("latent")?);
            let batch = reference.pixel.batch_size.max(1) + latent.pixel.batch_size.max(1);
            let model = i.model("model")?.with_patch(ModelPatch::ReferenceOnly { batch });
            let mut out = ctx.engine.latent_op(&LatentOp::PrependReference { reference: reference.clone(), latent: latent.clone() })?;
            // As SwarmUI's node: the latent's mask (or 64x64 ones), repeated to its batch if short,
            // after a single all-zero frame (keep the reference).
            let mask = latent.noise_mask.clone().unwrap_or_else(|| Mask { count: 1, height: 64, width: 64, data: Arc::new(vec![1.0; 64 * 64]) });
            let n = latent.pixel.batch_size.max(1);
            let reps = if mask.count < n { n } else { 1 };
            let plane = (mask.width * mask.height) as usize;
            let mut data = vec![0.0f32; plane];
            for _ in 0..reps {
                data.extend_from_slice(&mask.data);
            }
            out.noise_mask = Some(Mask { count: 1 + mask.count * reps, height: mask.height, width: mask.width, data: Arc::new(data) });
            Ok(NodeOutput::values(vec![Value::Model(model), Value::Latent(out)]))
        },
    );
}
