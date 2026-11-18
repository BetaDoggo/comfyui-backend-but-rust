//! `BrokeAnimaStyleReference` (Broke Anima Style Reference): a training-free art style from a picture, for Anima and Krea 2, as the SwarmUI-BrokeAnimaStyleReference extension
//! emits it (its ComfyUI node is in that extension's `comfy_node` folder). The node's inputs become a
//! [`ModelPatch::StyleReference`]; the engine encodes the picture and runs it beside every step.

use swarm_engine_api::{ModelPatch, StyleRefPatch, TextEncodeMode, TextEncodeRequest};

use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// The node's reference-pass prompt (`CAPTION` in its Python).
pub const CAPTION: &str = "no humans, scenery, detailed background";
/// Its prompt on Krea 2 (`KREA2_CAPTION`), whose text encoder reads sentences.
pub const KREA2_CAPTION: &str = "scenery with a detailed background, no people";

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("BrokeAnimaStyleReference", "Broke Anima Style Reference", "advanced/model")
                .input(link("model", "MODEL").tooltip("an Anima or Krea 2 model"))
                .input(link("clip", "CLIP").tooltip("the model's text encoder: encodes the reference pass's neutral caption"))
                .input(link("vae", "VAE"))
                .input(link("image", "IMAGE").tooltip("the style picture (first image of the batch)"))
                .input(float("stats_match", 0.6, 0.0, 1.0, 0.05).tooltip(
                    "how far each channel moves towards the picture's statistics after every block. 0 = off; 0.6 is a good default; near 1 the image falls apart on Anima",
                ))
                .input(boolean("restyle", true).tooltip(
                    "leave the first quarter of the steps alone, keeping the composition the prompt would have had without the picture",
                ))
                .input(float("reference_megapixels", 1.0, 0.5, 4.0, 0.05).tooltip(
                    "the picture is shrunk (never enlarged) to this area, aspect kept. About the generated image's size works best (1 for 1024x1024): smaller loses coarse texture such as pixel-art blocks, larger drifts off the style and falls apart near 4. Bigger is slower",
                ))
                .output("MODEL", "MODEL")
        },
        |i, ctx| {
            let model = i.model("model")?;
            let stats_match = i.float("stats_match")?;
            if !(0.0..=1.0).contains(&stats_match) {
                return Err(input_err("BrokeAnimaStyleReference: stats_match must be 0 to 1"));
            }
            // older workflows don't send it
            let reference_megapixels = if i.get("reference_megapixels").is_none() { 1.0 } else { i.float("reference_megapixels")? };
            if !(0.5..=4.0).contains(&reference_megapixels) {
                return Err(input_err("BrokeAnimaStyleReference: reference_megapixels must be 0.5 to 4"));
            }
            if stats_match == 0.0 {
                return Ok(NodeOutput::one(Value::Model(model.clone())));
            }
            let caption = ctx.engine.encode_text(&TextEncodeRequest {
                text_encoder: i.clip("clip")?.clone(),
                // the Python picks it by the model's class; the engine names the architecture it loaded
                mode: TextEncodeMode::Plain { text: if model.arch.as_deref() == Some("krea-2") { KREA2_CAPTION } else { CAPTION }.to_string() },
            })?;
            let image = i.image("image")?;
            let patch = StyleRefPatch {
                vae: i.vae("vae")?.clone(),
                caption,
                image: image.slice(0, 1),
                stats_match,
                restyle: i.boolean("restyle")?,
                reference_megapixels,
            };
            Ok(NodeOutput::one(Value::Model(model.with_patch(ModelPatch::StyleReference(Box::new(patch))))))
        },
    );
}
