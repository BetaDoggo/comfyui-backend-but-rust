//! VAE encode/decode nodes.

use swarm_engine_api::{VaeDecodeAudioRequest, VaeDecodeRequest, VaeEncodeRequest, VaeTiling};

use super::{NodeError, NodeInputs, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

fn tiling(i: &NodeInputs) -> Result<VaeTiling, NodeError> {
    Ok(VaeTiling {
        tile_size: i.uint("tile_size")?,
        overlap: i.uint("overlap")?,
        temporal_size: i.uint("temporal_size")?,
        temporal_overlap: i.uint("temporal_overlap")?,
    })
}

/// `decode`: VAEDecodeTiled (tile step 32). ComfyUI declares temporal_size max 4096 there too, but
/// SwarmUI sends 9999 ("no temporal tiling") for MiniMax H3, Wan, and HunyuanVideo 1.5, so the
/// decode limit is relaxed to accept what SwarmUI actually sends. Encode (tile step 64) is stock.
fn tiled_inputs(s: NodeSchema, decode: bool) -> NodeSchema {
    let verb = if decode { "decode" } else { "encode" };
    s.input(int_step("tile_size", 512, 64, 4096, if decode { 32 } else { 64 }).advanced())
        .input(int_step("overlap", 64, 0, 4096, 32).advanced())
        .input(int_step("temporal_size", 64, 8, if decode { MAX_RESOLUTION } else { 4096 }, 4).tooltip(&format!("Only used for video VAEs: Amount of frames to {verb} at a time.")).advanced())
        .input(int_step("temporal_overlap", 8, 4, 4096, 4).tooltip("Only used for video VAEs: Amount of frames to overlap.").advanced())
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| NodeSchema::new("VAEDecode", "VAE Decode", "model/latent").input(link("samples", "LATENT")).input(link("vae", "VAE")).output("IMAGE", "IMAGE"),
        |i, ctx| {
            let engine = ctx.engine;
            let req = VaeDecodeRequest { vae: i.vae("vae")?.clone(), latent: i.latent("samples")?.clone(), tiling: None };
            Ok(NodeOutput::one(Value::Image(engine.vae_decode(&req, ctx)?)))
        },
    );

    r.add(
        env,
        |_| tiled_inputs(NodeSchema::new("VAEDecodeTiled", "VAE Decode (Tiled)", "model/latent").input(link("samples", "LATENT")).input(link("vae", "VAE")), true).output("IMAGE", "IMAGE"),
        |i, ctx| {
            let engine = ctx.engine;
            let req = VaeDecodeRequest { vae: i.vae("vae")?.clone(), latent: i.latent("samples")?.clone(), tiling: Some(tiling(i)?) };
            Ok(NodeOutput::one(Value::Image(engine.vae_decode(&req, ctx)?)))
        },
    );

    r.add(
        env,
        |_| NodeSchema::new("VAEEncode", "VAE Encode", "model/latent").input(link("pixels", "IMAGE")).input(link("vae", "VAE")).output("LATENT", "LATENT"),
        |i, ctx| {
            let req = VaeEncodeRequest { vae: i.vae("vae")?.clone(), pixels: i.image("pixels")?.clone(), tiling: None };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.vae_encode(&req)?)))
        },
    );

    r.add(
        env,
        |_| tiled_inputs(NodeSchema::new("VAEEncodeTiled", "VAE Encode (Tiled)", "model/latent").input(link("pixels", "IMAGE")).input(link("vae", "VAE")), false).output("LATENT", "LATENT"),
        |i, ctx| {
            let req = VaeEncodeRequest { vae: i.vae("vae")?.clone(), pixels: i.image("pixels")?.clone(), tiling: Some(tiling(i)?) };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.vae_encode(&req)?)))
        },
    );

    r.add(
        env,
        |_| NodeSchema::new("VAEDecodeAudio", "VAE Decode Audio", "model/latent").input(link("samples", "LATENT")).input(link("vae", "VAE")).output("AUDIO", "AUDIO"),
        |i, ctx| {
            let req = VaeDecodeAudioRequest { vae: i.vae("vae")?.clone(), latent: i.latent("samples")?.clone(), tiling: None };
            Ok(NodeOutput::one(Value::Audio(ctx.engine.vae_decode_audio(&req)?)))
        },
    );
}
