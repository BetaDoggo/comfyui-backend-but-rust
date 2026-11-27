//! `RTXVideoSuperResolution`: NVIDIA's RTX Video Super Resolution, as the `nvidia-vfx` ComfyUI node runs it (an optional
//! deblur pass at the input's size, then the upscale), on the engine. The model is the `nvngx_vsr.dll` in
//! `upscale_models/`, which the node finds itself: ComfyUI's node takes no model input either.

use swarm_engine_api::{ModelFile, ModelFolder, NvidiaVsrLevel, NvidiaVsrMode, NvidiaVsrRequest, NVIDIA_VSR_MODEL};

use super::{input_err, NodeError, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

const LEVELS: [&str; 4] = ["LOW", "MEDIUM", "HIGH", "ULTRA"];
const DEBLUR: [&str; 5] = ["OFF", "LOW", "MEDIUM", "HIGH", "ULTRA"];

/// The node's output size for `scale`: `int(side * scale)` rounded to a multiple of 8 (half to even, as Python's
/// `round`), at least 8.
pub fn output_size(width: u32, height: u32, scale: f64) -> (u32, u32) {
    let side = |s: u32| (((s as f64 * scale).trunc() / 8.0).round_ties_even() as u32 * 8).max(8);
    (side(width), side(height))
}

/// The first `nvngx_vsr.dll` among the upscale models.
fn model(env: &ServerEnv) -> Result<ModelFile, NodeError> {
    env.models
        .list(ModelFolder::UpscaleModels)
        .iter()
        .filter(|n| n.rsplit('/').next().is_some_and(|f| f.eq_ignore_ascii_case(NVIDIA_VSR_MODEL)))
        .find_map(|n| env.models.resolve(ModelFolder::UpscaleModels, n))
        .ok_or_else(|| {
            input_err(format!(
                "RTX Video Super Resolution needs NVIDIA's {NVIDIA_VSR_MODEL} in an upscale_models folder, with the VFX runtime from the nvidia-vfx package beside it"
            ))
        })
}

fn level(name: &str) -> Result<NvidiaVsrLevel, NodeError> {
    NvidiaVsrLevel::from_name(name).ok_or_else(|| input_err(format!("RTXVideoSuperResolution: unknown level '{name}'")))
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("RTXVideoSuperResolution", "RTX Video Super Resolution", "image/upscaling")
                .describe("NVIDIA RTX Video Super Resolution, from the nvngx_vsr.dll in upscale_models.")
                .input(link("images", "IMAGE"))
                .input(float("scale", 2.0, 1.0, 4.0, 0.01).tooltip("Scale factor (e.g., 2.0 doubles the size)."))
                .input(combo_default("quality", &LEVELS, "ULTRA").tooltip("Upscaling quality level."))
                .input(
                    combo_default("deblur", &DEBLUR, "OFF")
                        .tooltip("Deblur/sharpen strength applied to the source before upscaling. OFF disables it."),
                )
                .output("IMAGE", "upscaled_images")
        },
        |i, ctx| {
            let images = i.image("images")?.clone();
            if images.count == 0 {
                return Err(input_err("RTXVideoSuperResolution expected at least one image."));
            }
            let upscale = level(i.string("quality")?)?;
            let deblur = match i.string("deblur")? {
                "OFF" => None,
                d => Some(level(d)?),
            };
            let (w, h) = (images.width, images.height);
            let (width, height) = output_size(w, h, i.float("scale")?);
            let model = model(ctx.env)?;
            let engine = ctx.engine;
            let mut out = images;
            if let Some(d) = deblur {
                let req = NvidiaVsrRequest { model: model.clone(), images: out, width: w, height: h, mode: NvidiaVsrMode::Deblur(d) };
                out = engine.nvidia_vsr(&req, ctx)?;
            }
            let req = NvidiaVsrRequest { model, images: out, width, height, mode: NvidiaVsrMode::Upscale(upscale) };
            Ok(NodeOutput::one(Value::Image(engine.nvidia_vsr(&req, ctx)?)))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_round_like_python() {
        assert_eq!(output_size(512, 768, 2.0), (1024, 1536));
        // 20 / 8 = 2.5 rounds to 2, as Python's round does
        assert_eq!(output_size(20, 28, 1.0), (16, 32));
        assert_eq!(output_size(3, 3, 1.0), (8, 8));
        assert_eq!(output_size(1000, 600, 1.37), (1368, 824));
    }
}
