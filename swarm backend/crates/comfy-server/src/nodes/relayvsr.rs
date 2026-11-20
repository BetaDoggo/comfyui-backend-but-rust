//! RelayVSR's nodes (`comfy_extras/nodes_wan.py`): the conditioning that picks the keyframes and sizes the latent, and
//! the decode that reads the whole clip beside it. The keyframe model and the Flash decoder are the engine's.

use swarm_engine_api::requests::relayvsr;
use swarm_engine_api::{CondMeta, Conditioning, EmptyLatentKind, EmptyLatentRequest, LatentFormat, LatentSamples, Opaque, RelayVsrCondition, VsrDecodeRequest};

use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("RelayVSRConditioning", "RelayVSRConditioning", "model/conditioning/wan")
                .describe("Conditioning and empty latent for RelayVSR video upscaling, sample with 1 step at cfg 1. Only every 15th frame goes through the model, VSR Decode restores all of them.")
                .input(link("model", "MODEL"))
                .input(link("images", "IMAGE").tooltip("The low-quality video at its native resolution, the model decides the upscale factor."))
                .output("CONDITIONING", "positive")
                .output("LATENT", "latent")
        },
        |i, ctx| {
            let model = i.model("model")?;
            let images = i.image("images")?;
            if images.count == 0 {
                return Err(input_err("RelayVSRConditioning expected at least one frame."));
            }
            let stride = ctx.engine.relayvsr_latent_stride(model)?;
            let keys = relayvsr::keyframes(images.count);
            let (lh, lw) = relayvsr::latent_size(images.height, images.width, stride);
            let meta = CondMeta { relayvsr: Some(RelayVsrCondition { keyframes: images.frames(&keys), frames: images.count }), ..CondMeta::default() };
            let kind = EmptyLatentKind::RelayVsr {
                keyframes: keys.len() as u32,
                latent_height: lh,
                latent_width: lw,
                width: images.width * 16 / stride,
                height: images.height * 16 / stride,
                frames: images.count,
            };
            // Not marked empty: the zeros are the model's input as they are, never widened or replaced.
            let latent = ctx.engine.empty_latent(&EmptyLatentRequest { kind, batch_size: 1 })?;
            Ok(NodeOutput::values(vec![Value::Conditioning(Conditioning::single(Opaque::none(), meta)), Value::Latent(latent)]))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("VSRDecode", "VSR Decode", "latent")
                .describe("Decode RelayVSR latents with the RelayVSR Flash decoder, which also reads the low-quality frames.")
                .input(link("vae", "VAE"))
                .input(link("samples", "LATENT"))
                .input(link("images", "IMAGE").tooltip("The same low-quality video given to the conditioning node."))
                .input(float("detail", 1.0, 0.0, 1.0, 0.01).tooltip("How much of the detail the decoder adds to the bicubic upscale of each frame is kept, 1 is the full model output."))
                .output("IMAGE", "IMAGE")
        },
        |i, ctx| {
            let latent = i.latent("samples")?.clone();
            let LatentSamples::Single(t) = &latent.samples else {
                return Err(input_err("VSRDecode expects a RelayVSR latent, not an audio/video one"));
            };
            if !matches!(t.format, LatentFormat::RelayVsr | LatentFormat::EngineDefined) || t.shape.get(1) != Some(&48) {
                return Err(input_err(format!("VSRDecode expects a RelayVSR latent with 48 channels; got a {:?} latent of shape {:?}.", t.format, t.shape)));
            }
            let images = i.image("images")?.clone();
            let want = relayvsr::keyframes(images.count).len();
            if t.shape.get(2) != Some(&want) {
                return Err(input_err(format!("VSRDecode: the latent has {:?} keyframes but a clip of {} frames has {want}; give it the same images as RelayVSRConditioning.", t.shape.get(2), images.count)));
            }
            let engine = ctx.engine;
            let req = VsrDecodeRequest { vae: i.vae("vae")?.clone(), latent, images, detail: i.float("detail")? };
            Ok(NodeOutput::one(Value::Image(engine.vsr_decode(&req, ctx)?)))
        },
    );
}
