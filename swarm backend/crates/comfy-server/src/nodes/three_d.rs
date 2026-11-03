//! Image to 3D: the Pixal3D extension's node, which runs ComfyUI's Pixal3D example workflow as one engine call
//! and saves the mesh like `SaveGLB`.

use serde_json::json;
use swarm_engine_api::{ImageTo3DRequest, ModelFolder};

use super::loaders::resolve;
use super::output::output_slot;
use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |env| {
            let vaes = env.models.list(ModelFolder::Vae);
            NodeSchema::new("Pixal3DImageTo3D", "Pixal3D Image to 3D", "3d")
                .describe("Pixal3D's example workflow in one node: crop to the mask, DINOv3 + NAF conditioning, the structure, shape, refine and texture samplers, the shape and texture decodes and PaintMesh. Saves a vertex-coloured GLB.")
                .input(link("image", "IMAGE"))
                .input(combo("unet_name", &env.models.list(ModelFolder::DiffusionModels)))
                .input(combo("clip_vision_name", &env.models.list(ModelFolder::ClipVision)))
                .input(combo("shape_vae_name", &vaes))
                .input(combo("texture_vae_name", &vaes))
                .input(int("seed", 0, 0, MAX_SEED))
                .input(float("fov", 49.13, 1.0, 170.0, 0.01).tooltip("Horizontal field of view of the photo, in degrees."))
                .input(int_step("resolution", 1536, 1024, 2048, 128).tooltip("Voxel resolution of the refined shape. Higher is finer and slower."))
                .input(int("structure_steps", 12, 1, 200))
                .input(int("shape_steps", 20, 1, 200))
                .input(int("refine_steps", 12, 1, 200))
                .input(int("texture_steps", 12, 1, 200))
                .input(float("cfg", 7.5, 1.0, 30.0, 0.1).tooltip("CFG of the geometry samplers; the texture sampler runs at 1."))
                .input(boolean("texture", true))
                .input(int("max_faces", 700_000, 0, 100_000_000).tooltip("Simplify the mesh to at most this many triangles; 0 keeps every one."))
                .input(string("filename_prefix", "3d/pixal3d", false))
                .input(link("mask", "MASK").optional().tooltip("The object's mask. Without one, the image's alpha."))
                .output_node()
        },
        |i, ctx| {
            let file = |name: &str, folder: ModelFolder| resolve(ctx.env, &[folder], i.string(name)?);
            let positive = |name: &str| -> Result<u32, super::NodeError> { Ok(i.int(name)?.clamp(0, u32::MAX as i64) as u32) };
            let req = ImageTo3DRequest {
                diffusion_model: file("unet_name", ModelFolder::DiffusionModels)?,
                clip_vision: file("clip_vision_name", ModelFolder::ClipVision)?,
                shape_vae: file("shape_vae_name", ModelFolder::Vae)?,
                texture_vae: file("texture_vae_name", ModelFolder::Vae)?,
                image: i.image("image")?.clone(),
                mask: i.opt_mask("mask")?.cloned(),
                seed: i.int("seed")? as u64,
                fov: i.float("fov")?,
                resolution: positive("resolution")?,
                structure_steps: positive("structure_steps")?,
                shape_steps: positive("shape_steps")?,
                refine_steps: positive("refine_steps")?,
                texture_steps: positive("texture_steps")?,
                cfg: i.float("cfg")?,
                texture: i.boolean("texture")?,
                max_faces: positive("max_faces")?,
            };
            if req.image.count == 0 {
                return Err(input_err("Pixal3DImageTo3D needs an image"));
            }
            let engine = ctx.engine;
            let glb = engine.image_to_3d(&req, ctx)?;
            let (target, subfolder, stem, counter) = output_slot(&ctx.env.output_dir, i.string("filename_prefix")?)?;
            let filename = format!("{stem}_{counter:05}_.glb");
            std::fs::write(target.join(&filename), glb).map_err(anyhow::Error::from)?;
            Ok(NodeOutput { values: Vec::new(), ui: Some(json!({ "3d": [{ "filename": filename, "subfolder": subfolder, "type": "output" }] })) })
        },
    );
}
