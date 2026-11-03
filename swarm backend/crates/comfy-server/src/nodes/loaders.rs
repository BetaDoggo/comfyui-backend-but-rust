//! Model loaders and LoRA / hook nodes.

use swarm_engine_api::{
    AddonKind, LoadAddon, Opaque, VaeHandle, LoadCheckpoint, LoadDiffusionModel, LoadTextEncoder, LoadVae, LoraHook, ModelFile, ModelFolder, ModelPatch,
    TextEncoderPatch, WeightDtype,
};

use super::{input_err, NodeError, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// `CLIPLoader.type` options, from ComfyUI `nodes.py`.
pub const CLIP_TYPES: &[&str] = &[
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma",
    "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu",
    "krea2", "joyimage", "mage", "minimax", "yue2", "flux3", "moss_image", "fibo2",
];

pub const DUAL_CLIP_TYPES: &[&str] = &["sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image", "hunyuan_video_15", "kandinsky5", "kandinsky5_image", "ltxv", "newbie", "ace"];

pub fn resolve(env: &ServerEnv, folders: &[ModelFolder], name: &str) -> Result<ModelFile, NodeError> {
    folders
        .iter()
        .find_map(|f| env.models.resolve(*f, name))
        .ok_or_else(|| input_err(format!("model file '{name}' not found in {}", folders.iter().map(|f| f.comfy_key()).collect::<Vec<_>>().join("/"))))
}

const PIXEL_SPACE_VAE: &str = "pixel_space";

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add_cacheable(
        env,
        |env| {
            NodeSchema::new("CheckpointLoaderSimple", "Load Checkpoint", "model/loaders")
                .input(combo("ckpt_name", &env.models.list(ModelFolder::Checkpoints)))
                .output("MODEL", "MODEL")
                .output("CLIP", "CLIP")
                .output("VAE", "VAE")
        },
        |i, ctx| {
            let file = resolve(ctx.env, &[ModelFolder::Checkpoints], i.string("ckpt_name")?)?;
            let (m, c, v) = ctx.engine.load_checkpoint(&LoadCheckpoint { file })?;
            Ok(NodeOutput::values(vec![Value::Model(m), Value::Clip(c), Value::Vae(v)]))
        },
    );

    r.add_cacheable(
        env,
        |env| {
            NodeSchema::new("ImageOnlyCheckpointLoader", "Load Checkpoint Image Only (img2vid model)", "model/loaders")
                .input(combo("ckpt_name", &env.models.list(ModelFolder::Checkpoints)))
                .output("MODEL", "MODEL")
                .output("CLIP_VISION", "CLIP_VISION")
                .output("VAE", "VAE")
        },
        |i, ctx| {
            let file = resolve(ctx.env, &[ModelFolder::Checkpoints], i.string("ckpt_name")?)?;
            let (m, cv, v) = ctx.engine.load_image_only_checkpoint(&LoadCheckpoint { file })?;
            Ok(NodeOutput::values(vec![Value::Model(m), Value::Addon(cv), Value::Vae(v)]))
        },
    );

    r.add_cacheable(
        env,
        |env| {
            NodeSchema::new("UNETLoader", "Load Diffusion Model", "model/loaders")
                .input(combo("unet_name", &env.models.list(ModelFolder::DiffusionModels)))
                .input(combo("weight_dtype", &WeightDtype::OPTIONS).advanced())
                .output("MODEL", "MODEL")
        },
        |i, ctx| {
            let file = resolve(ctx.env, &[ModelFolder::DiffusionModels, ModelFolder::Checkpoints], i.string("unet_name")?)?;
            let weight_dtype = WeightDtype::parse(i.string("weight_dtype")?).unwrap_or(WeightDtype::Default);
            let m = ctx.engine.load_diffusion_model(&LoadDiffusionModel { file, weight_dtype })?;
            Ok(NodeOutput::one(Value::Model(m)))
        },
    );

    r.add_cacheable(
        env,
        |env| {
            NodeSchema::new("CLIPLoader", "Load CLIP", "model/loaders")
                .input(combo("clip_name", &env.models.list(ModelFolder::TextEncoders)))
                .input(combo("type", CLIP_TYPES))
                .input(combo("device", &["default", "cpu"]).optional().advanced())
                .output("CLIP", "CLIP")
        },
        |i, ctx| {
            let file = resolve(ctx.env, &[ModelFolder::TextEncoders], i.string("clip_name")?)?;
            let req = LoadTextEncoder {
                files: vec![file],
                encoder_type: i.string("type")?.to_string(),
                device: i.opt_string("device")?.unwrap_or("default").to_string(),
            };
            Ok(NodeOutput::one(Value::Clip(ctx.engine.load_text_encoder(&req)?)))
        },
    );

    r.add_cacheable(
        env,
        |env| {
            let list = env.models.list(ModelFolder::TextEncoders);
            NodeSchema::new("DualCLIPLoader", "DualCLIPLoader", "model/loaders")
                .input(combo("clip_name1", &list))
                .input(combo("clip_name2", &list))
                .input(combo("type", DUAL_CLIP_TYPES))
                .input(combo("device", &["default", "cpu"]).optional().advanced())
                .output("CLIP", "CLIP")
        },
        |i, ctx| {
            let files = vec![
                resolve(ctx.env, &[ModelFolder::TextEncoders], i.string("clip_name1")?)?,
                resolve(ctx.env, &[ModelFolder::TextEncoders], i.string("clip_name2")?)?,
            ];
            let req = LoadTextEncoder { files, encoder_type: i.string("type")?.to_string(), device: i.opt_string("device")?.unwrap_or("default").to_string() };
            Ok(NodeOutput::one(Value::Clip(ctx.engine.load_text_encoder(&req)?)))
        },
    );

    r.add_cacheable(
        env,
        |env| {
            // ComfyUI appends "pixel_space": the identity VAE for pixel-space models (Chroma Radiance, HiDream O1).
            let mut names = env.models.list(ModelFolder::Vae);
            names.push(PIXEL_SPACE_VAE.to_string());
            NodeSchema::new("VAELoader", "Load VAE", "model/loaders").input(combo("vae_name", &names)).output("VAE", "VAE")
        },
        |i, ctx| {
            if i.string("vae_name")? == PIXEL_SPACE_VAE {
                return Ok(NodeOutput::one(Value::Vae(VaeHandle::new(None, Opaque::none()))));
            }
            let file = resolve(ctx.env, &[ModelFolder::Vae], i.string("vae_name")?)?;
            Ok(NodeOutput::one(Value::Vae(ctx.engine.load_vae(&LoadVae { file })?)))
        },
    );

    // LoRAs are pure bookkeeping here: they become patches the engine applies at sample/encode time.
    r.add(
        env,
        |env| {
            NodeSchema::new("LoraLoader", "Load LoRA", "model/loaders")
                .input(link("model", "MODEL"))
                .input(link("clip", "CLIP"))
                .input(combo("lora_name", &env.models.list(ModelFolder::Loras)))
                .input(float("strength_model", 1.0, -100.0, 100.0, 0.01))
                .input(float("strength_clip", 1.0, -100.0, 100.0, 0.01))
                .output("MODEL", "MODEL")
                .output("CLIP", "CLIP")
        },
        |i, ctx| {
            let file = resolve(ctx.env, &[ModelFolder::Loras], i.string("lora_name")?)?;
            let sm = i.float("strength_model")? as f32;
            let sc = i.float("strength_clip")? as f32;
            let model = i.model("model")?;
            let clip = i.clip("clip")?;
            let model = if sm != 0.0 { model.with_patch(ModelPatch::Lora { file: file.clone(), strength: sm }) } else { model.clone() };
            let clip = if sc != 0.0 { clip.with_patch(TextEncoderPatch::Lora { file, strength: sc }) } else { clip.clone() };
            Ok(NodeOutput::values(vec![Value::Model(model), Value::Clip(clip)]))
        },
    );

    r.add(
        env,
        |env| {
            NodeSchema::new("LoraLoaderModelOnly", "LoraLoaderModelOnly", "model/loaders")
                .input(link("model", "MODEL"))
                .input(combo("lora_name", &env.models.list(ModelFolder::Loras)))
                .input(float("strength_model", 1.0, -100.0, 100.0, 0.01))
                .output("MODEL", "MODEL")
        },
        |i, ctx| {
            let file = resolve(ctx.env, &[ModelFolder::Loras], i.string("lora_name")?)?;
            let s = i.float("strength_model")? as f32;
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(ModelPatch::Lora { file, strength: s }))))
        },
    );

    r.add(
        env,
        |env| {
            NodeSchema::new("CreateHookLora", "Create Hook LoRA", "advanced/hooks/create")
                .input(combo("lora_name", &env.models.list(ModelFolder::Loras)))
                .input(float("strength_model", 1.0, -20.0, 20.0, 0.01))
                .input(float("strength_clip", 1.0, -20.0, 20.0, 0.01))
                .input(link("prev_hooks", "HOOKS").optional())
                .output("HOOKS", "HOOKS")
        },
        |i, ctx| {
            let mut hooks = i.opt_hooks("prev_hooks")?.cloned().unwrap_or_default();
            hooks.push(LoraHook {
                file: resolve(ctx.env, &[ModelFolder::Loras], i.string("lora_name")?)?,
                strength_model: i.float("strength_model")? as f32,
                strength_clip: i.float("strength_clip")? as f32,
            });
            Ok(NodeOutput::one(Value::Hooks(hooks)))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SetClipHooks", "Set CLIP Hooks", "advanced/hooks/clip")
                .input(link("clip", "CLIP"))
                .input(boolean("apply_to_conds", true).advanced())
                .input(boolean("schedule_clip", false).advanced())
                .input(link("hooks", "HOOKS").optional())
                .output("CLIP", "CLIP")
        },
        |i, _| {
            let clip = i.clip("clip")?;
            let out = match i.opt_hooks("hooks")? {
                Some(h) => clip.with_patch(TextEncoderPatch::Hooks {
                    hooks: h.clone(),
                    apply_to_conds: i.boolean("apply_to_conds")?,
                    schedule_clip: i.boolean("schedule_clip")?,
                }),
                None => clip.clone(),
            };
            Ok(NodeOutput::one(Value::Clip(out)))
        },
    );

    // Exists so SwarmUI can read the embeddings list out of object_info.
    r.add(
        env,
        |env| {
            NodeSchema::new("SwarmEmbedLoaderListProvider", "Swarm Embed Loader List Provider", "SwarmUI/internal")
                .describe("Internal node just intended to provide a list of currently known embeddings to Swarm.")
                .input(combo("embed_name", &env.models.list(ModelFolder::Embeddings)))
                .output("EMBEDDING", "EMBEDDING")
        },
        |i, ctx| Ok(NodeOutput::one(Value::Embedding(resolve(ctx.env, &[ModelFolder::Embeddings], i.string("embed_name")?)?))),
    );

    // Add-on models: one loader per ComfyUI type, each yielding an `AddonHandle` for the nodes that use it.
    let addons: [(&'static str, &'static str, &'static str, ModelFolder, AddonKind); 8] = [
        ("GLIGENLoader", "Load GLIGEN Model", "gligen_name", ModelFolder::Gligen, AddonKind::Gligen),
        ("UpscaleModelLoader", "Load Upscale Model", "model_name", ModelFolder::UpscaleModels, AddonKind::UpscaleModel),
        ("LatentUpscaleModelLoader", "Load Latent Upscale Model", "model_name", ModelFolder::LatentUpscaleModels, AddonKind::LatentUpscaleModel),
        ("CLIPVisionLoader", "Load CLIP Vision", "clip_name", ModelFolder::ClipVision, AddonKind::ClipVision),
        ("ControlNetLoader", "Load ControlNet Model", "control_net_name", ModelFolder::ControlNet, AddonKind::ControlNet),
        ("StyleModelLoader", "Load Style Model", "style_model_name", ModelFolder::StyleModels, AddonKind::StyleModel),
        ("ModelPatchLoader", "Load Model Patch", "name", ModelFolder::ModelPatches, AddonKind::ModelPatch),
        ("LoadBackgroundRemovalModel", "Load Background Removal Model", "bg_removal_name", ModelFolder::BackgroundRemoval, AddonKind::BackgroundRemoval),
    ];
    for (class, display, input, folder, kind) in addons {
        r.add_cacheable(
            env,
            move |env| NodeSchema::new(class, display, "model/loaders").input(combo(input, &env.models.list(folder))).output(kind.comfy_type(), kind.comfy_type()),
            move |i, ctx| {
                let file = resolve(ctx.env, &[folder], i.string(input)?)?;
                Ok(NodeOutput::one(Value::Addon(ctx.engine.load_addon(&LoadAddon { kind, file })?)))
            },
        );
    }
}
