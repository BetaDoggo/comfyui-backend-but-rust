//! `SnapRef`: ArcAnima's reference adapter for Anima (https://github.com/ArcAnima/SnapRef), as the
//! SwarmUI-SnapRef extension emits it. The node's inputs become a [`ModelPatch::SnapRef`]; the engine encodes the
//! pictures and runs the reference stream.

use swarm_engine_api::{ModelFolder, ModelPatch, SnapRefPatch, TextEncodeMode, TextEncodeRequest};

use super::loaders::resolve;
use super::{input_err, NodeOutput, NodeRegistry};
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// The node's `_adapter_files`: the LoRA folder's SnapRef files, or every file when none is named like one.
fn adapter_files(env: &ServerEnv) -> Vec<String> {
    let all = env.models.list(ModelFolder::Loras);
    let named: Vec<String> = all.iter().filter(|n| n.to_lowercase().contains("snapref") || n.contains("anima_ref_kv")).cloned().collect();
    if named.is_empty() { all } else { named }
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |env| {
            let weight = |n: &str, tip: &str| float(n, 1.0, 0.0, 2.0, 0.05).optional().tooltip(tip);
            NodeSchema::new("SnapRef", "SnapRef (Advanced)", "SnapRef")
                .input(link("model", "MODEL"))
                .input(link("clip", "CLIP").tooltip("encodes ref_caption for the reference stream"))
                .input(link("vae", "VAE"))
                .input(link("image", "IMAGE").tooltip("the reference (first image of the batch)"))
                .input(combo("adapter", &adapter_files(env)))
                .input(combo("mode", &["cs", "char"]).tooltip("cs = character and style from the reference; char = character only (put artist tags in the prompt)"))
                .input(boolean("ref_in_uncond", false).tooltip("off (trained default): only the positive branch reads the reference, so CFG pushes toward it. on: weaker, closer to the prompt"))
                .input(float("ref_megapixels", 1.0, 0.25, 2.0, 0.05).tooltip("the reference is shrunk (never enlarged) to this area, aspect kept; trained at <= 1.0"))
                .input(float("strength", 0.0, -3.0, 3.0, 0.1).optional().tooltip("attention to the reference, log scale. 0 = as trained; +1 = about 2.7x more, -1 = about 2.7x less. Non-zero values make sampling about 1.3x slower"))
                .input(string("ref_caption", "", true).optional().tooltip("optional appearance tags of the reference (gender, solo, hair, eyes, outfit); no name or artist. Empty works."))
                .input(boolean("ref_guidance_on", false).optional().tooltip("separate reference guidance (one extra model call per step until ref_end). Puts the reference in both CFG branches and pushes from 'no reference' toward it by ref_guidance; ignores ref_in_uncond"))
                .input(float("ref_guidance", 4.5, 0.0, 15.0, 0.1).optional().tooltip("reference strength, separate from cfg. Equal to cfg = the same strength as with ref_guidance off (plus APG); above cfg = closer to the reference, below = closer to the prompt; 1 = no push"))
                .input(combo("correction", &["apg", "rescale", "none"]).optional().tooltip("keeps colors in check when ref_guidance is on"))
                .input(float("ref_end", 1.0, 0.0, 1.0, 0.01).optional().tooltip("last step with the extra call, as a fraction of the denoising; ~0.6 is faster"))
                .input(link("image_2", "IMAGE").optional().tooltip("another reference of the SAME character (e.g. full body when image is an upper-body shot). For two different characters, stitch them into image instead"))
                .input(link("image_3", "IMAGE").optional().tooltip("a third reference of the same character"))
                .input(boolean("use_batch", false).optional().tooltip("use every image of each IMAGE batch as a reference of the same character (off: only the first image of each input)"))
                .input(combo("multi_layout", &["separate", "stitch"]).optional().tooltip("several references: separate = each at ref_megapixels, encoded on its own (sharper, slower with more references); stitch = side by side in one ref_megapixels image (as Image Stitch, same speed as one reference)"))
                .input(weight("weight_1", "how much image (the first reference) counts; 1 = as trained. Only with several references in multi_layout separate. No speed cost"))
                .input(weight("weight_2", "how much image_2 counts (e.g. 0.4: mostly the first image, a little of this one); 0 = ignored"))
                .input(weight("weight_3", "how much image_3 counts"))
                .input(float("fidelity", 1.0, 0.0, 1.0, 0.05).optional().tooltip("1 = keep every detail of the picture; lower = follow the prompt more freely. Ignored when ref_in_uncond or ref_guidance_on is on"))
                .output("MODEL", "MODEL")
        },
        |i, ctx| {
            let adapter = resolve(ctx.env, &[ModelFolder::Loras], i.string("adapter")?)?;
            let opt_float = |n: &str, d: f64| if i.get(n).is_none() { Ok(d) } else { i.float(n) };
            let opt_bool = |n: &str| if i.get(n).is_none() { Ok(false) } else { i.boolean(n) };
            let use_batch = opt_bool("use_batch")?;
            // `_reference_images`: the first image of each input (every one with use_batch), weight-0 inputs dropped
            let mut references = Vec::new();
            for (name, weight) in [("image", "weight_1"), ("image_2", "weight_2"), ("image_3", "weight_3")] {
                let w = opt_float(weight, 1.0)?;
                let Some(img) = i.opt_image(name)? else { continue };
                if w <= 0.0 {
                    continue;
                }
                for f in 0..if use_batch { img.count } else { 1 } {
                    references.push((img.slice(f, 1), w));
                }
            }
            if references.is_empty() {
                return Err(input_err("SnapRef: every reference has weight 0"));
            }
            let caption = i.opt_string("ref_caption")?.unwrap_or("").trim().to_string();
            let caption = ctx.engine.encode_text(&TextEncodeRequest { text_encoder: i.clip("clip")?.clone(), mode: TextEncodeMode::Plain { text: caption } })?;
            let ref_guidance = match opt_bool("ref_guidance_on")? {
                true => Some((opt_float("ref_guidance", 4.5)?, i.opt_string("correction")?.unwrap_or("apg").to_string(), opt_float("ref_end", 1.0)?)),
                false => None,
            };
            let patch = SnapRefPatch {
                adapter,
                vae: i.vae("vae")?.clone(),
                caption,
                references,
                mode: i.string("mode")?.to_string(),
                ref_in_uncond: i.boolean("ref_in_uncond")?,
                ref_megapixels: i.float("ref_megapixels")?,
                strength: opt_float("strength", 0.0)?,
                fidelity: opt_float("fidelity", 1.0)?,
                stitch: i.opt_string("multi_layout")? == Some("stitch"),
                ref_guidance,
            };
            Ok(NodeOutput::one(Value::Model(i.model("model")?.with_patch(ModelPatch::SnapRef(Box::new(patch))))))
        },
    );
}
