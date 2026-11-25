//! `NemotronImageGenerate`: Nemotron-Labs-Image's text to image in one node. The model is autoregressive: it writes
//! the image as visual tokens and decodes them with its own decoder, so there is no sampler, CFG, VAE or text encoder to
//! wire. ComfyUI has no node for it; this one is the engine's, for a SwarmUI extension to call.

use swarm_engine_api::{ModelFolder, NemotronImageRequest};

use super::loaders::resolve;
use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |env| {
            NodeSchema::new("NemotronImageGenerate", "Nemotron-Labs-Image Generate", "image")
                .describe("Nemotron-Labs-Image 30B-A3B: text to image, the image written as visual tokens one at a time and decoded by the model's own decoder. Needs the file tools/nemotron_image_convert.py writes.")
                .input(combo("unet_name", &env.models.list(ModelFolder::DiffusionModels)))
                .input(string("prompt", "", true))
                .input(int_step("width", 1024, 16, 4096, 16).tooltip("A multiple of 16. The model card's sizes are 1024x1024, 1184x880, 1248x832 and 880x1184."))
                .input(int_step("height", 1024, 16, 4096, 16))
                .input(int("seed", 0, 0, MAX_SEED))
                .input(int("batch_size", 1, 1, 8).tooltip("Images sampled together, each with the next seed. More cost little more time than one."))
                .input(float("temperature", 0.9, 0.0, 2.0, 0.01).tooltip("Softmax temperature of the image tokens. The model card uses 0.9; 0 takes the likeliest token."))
                .output("IMAGE", "IMAGE")
        },
        |i, ctx| {
            let dim = |name: &str| -> Result<u32, super::NodeError> {
                let v = i.int(name)?;
                if v <= 0 || v % 16 != 0 || v > u32::MAX as i64 {
                    return Err(input_err(format!("NemotronImageGenerate: {name} must be a positive multiple of 16, not {v}")));
                }
                Ok(v as u32)
            };
            let batch = i.int("batch_size")?;
            if !(1..=8).contains(&batch) {
                return Err(input_err(format!("NemotronImageGenerate: batch_size must be 1 to 8, not {batch}")));
            }
            let req = NemotronImageRequest {
                model: resolve(ctx.env, &[ModelFolder::DiffusionModels], i.string("unet_name")?)?,
                prompt: i.string("prompt")?.to_string(),
                width: dim("width")?,
                height: dim("height")?,
                seed: i.int("seed")? as u64,
                batch_size: batch as u32,
                temperature: i.float("temperature")?,
            };
            let engine = ctx.engine;
            Ok(NodeOutput::one(Value::Image(engine.nemotron_image(&req, ctx)?)))
        },
    );
}
