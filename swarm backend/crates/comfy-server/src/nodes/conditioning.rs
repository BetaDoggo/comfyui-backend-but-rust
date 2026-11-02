//! Text encoding and conditioning metadata nodes.

use swarm_engine_api::requests::minimax_h3;
use swarm_engine_api::{
    AddonKind, CondMask, GligenBox, MiniMaxKeyframe, MiniMaxReferences, ParsedPrompt, SwarmTextEncode, TextEncodeMode, TextEncodeRequest, VaeEncodeRequest,
};

use super::{NodeOutput, NodeRegistry};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("CLIPTextEncode", "CLIP Text Encode (Prompt)", "model/conditioning")
                .input(string("text", "", true).no_default())
                .input(link("clip", "CLIP"))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, ctx| {
            let req = TextEncodeRequest { text_encoder: i.clip("clip")?.clone(), mode: TextEncodeMode::Plain { text: i.string("text")?.to_string() } };
            Ok(NodeOutput::one(Value::Conditioning(ctx.engine.encode_text(&req)?)))
        },
    );

    // SwarmClipTextEncodeAdvanced is the older name (no LoRA hooks); SwarmUI no longer emits it.
    for (class, display, with_hooks) in [("SwarmTextEncodeAdvanced", "Swarm Text Encode Advanced", true), ("SwarmClipTextEncodeAdvanced", "SwarmClipTextEncodeAdvanced", false)] {
    r.add(
        env,
        move |_| {
            let s = NodeSchema::new(class, display, "SwarmUI/clip")
                .describe("Acts like the regular CLIPTextEncode, but supports Swarm prompt tags such as '<break>', '<fromto[0.5]:a, b>', '<alternate:a, b>', '<weight[2]:text>', '<embed:name>'.")
                .input(link("clip", "CLIP"))
                .input(int("steps", 20, 1, 10000))
                .input(string("prompt", "", true).no_default())
                .input(int("width", 1024, 0, MAX_RESOLUTION))
                .input(int("height", 1024, 0, MAX_RESOLUTION))
                .input(int("target_width", 1024, 0, MAX_RESOLUTION))
                .input(int("target_height", 1024, 0, MAX_RESOLUTION))
                .input(float("guidance", -1.0, -1.0, 100.0, 0.1).optional())
                .input(string("llama_template", "", true).optional())
                .input(link("clip_vision_output", "CLIP_VISION_OUTPUT").optional().null_default())
                .input(link("images", "IMAGE").optional().null_default())
                .input(link("minimax_refs", "MiniMaxReferences").optional().null_default());
            let s = if with_hooks { s.input(link("lora_hooks", "HOOKS").optional().null_default()) } else { s };
            s.output("CONDITIONING", "CONDITIONING")
        },
        |i, ctx| {
            let guidance = i.float("guidance")?;
            let template = i.opt_string("llama_template")?.filter(|t| !t.trim().is_empty());
            let enc = SwarmTextEncode {
                prompt: ParsedPrompt::parse(i.string("prompt")?),
                steps: i.uint("steps")?,
                width: i.uint("width")?,
                height: i.uint("height")?,
                target_width: i.uint("target_width")?,
                target_height: i.uint("target_height")?,
                guidance: (guidance >= 0.0).then_some(guidance),
                llama_template: template.map(str::to_string),
                images: i.opt_image("images")?.cloned(),
                minimax_refs: match i.get("minimax_refs") {
                    Some(Value::MiniMaxRefs(r)) => Some(r.clone()),
                    _ => None,
                },
                lora_hooks: i.opt_hooks("lora_hooks")?.cloned(),
                clip_vision: i.opt_clip_vision_output("clip_vision_output")?.cloned(),
            };
            let req = TextEncodeRequest { text_encoder: i.clip("clip")?.clone(), mode: TextEncodeMode::Swarm(enc) };
            Ok(NodeOutput::one(Value::Conditioning(ctx.engine.encode_text(&req)?)))
        },
    );
    }

    // ---------------------------------------------------------- regional prompting ---
    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningSetMask", "Conditioning (Set Mask)", "model/conditioning/transform")
                .input(link("conditioning", "CONDITIONING"))
                .input(link("mask", "MASK"))
                .input(float("strength", 1.0, 0.0, 10.0, 0.01))
                .input(combo("set_cond_area", &["default", "mask bounds"]))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let area = CondMask { mask: i.mask("mask")?.clone(), strength: i.float("strength")?, set_area_to_bounds: i.string("set_cond_area")? != "default" };
            Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning")?.map_meta(|m| m.mask = Some(area.clone())))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningCombine", "Conditioning (Combine)", "model/conditioning/transform")
                .input(link("conditioning_1", "CONDITIONING"))
                .input(link("conditioning_2", "CONDITIONING"))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            // ComfyUI: conditioning_1 + conditioning_2 (list concatenation; the sampler runs every entry).
            let mut out = i.conditioning("conditioning_1")?.clone();
            out.entries.extend(i.conditioning("conditioning_2")?.entries.iter().cloned());
            Ok(NodeOutput::one(Value::Conditioning(out)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("GLIGENTextBoxApply", "Apply GLIGEN Text Box", "model/conditioning/gligen")
                .input(link("conditioning_to", "CONDITIONING"))
                .input(link("clip", "CLIP"))
                .input(link("gligen_textbox_model", "GLIGEN"))
                .input(string("text", "", true).no_default())
                .input(int_step("width", 64, 8, MAX_RESOLUTION, 8))
                .input(int_step("height", 64, 8, MAX_RESOLUTION, 8))
                .input(int_step("x", 0, 0, MAX_RESOLUTION, 8))
                .input(int_step("y", 0, 0, MAX_RESOLUTION, 8))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, ctx| {
            // ComfyUI encodes the box text with the same CLIP (it uses the pooled output).
            let req = TextEncodeRequest { text_encoder: i.clip("clip")?.clone(), mode: TextEncodeMode::Plain { text: i.string("text")?.to_string() } };
            let text = Box::new(ctx.engine.encode_text(&req)?);
            let gligen = GligenBox {
                model: i.addon("gligen_textbox_model", AddonKind::Gligen)?.clone(),
                text,
                x: i.uint("x")?,
                y: i.uint("y")?,
                width: i.uint("width")?,
                height: i.uint("height")?,
            };
            Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning_to")?.map_meta(|m| m.gligen.push(gligen.clone())))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningZeroOut", "ConditioningZeroOut", "model/conditioning/transform")
                .input(link("conditioning", "CONDITIONING"))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning")?.map_meta(|m| m.zeroed = true)))),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ConditioningMultiply", "Conditioning (Multiply)", "model/conditioning/transform")
                .input(link("conditioning", "CONDITIONING"))
                .input(float("multiplier", 1.0, -100.0, 100.0, 0.01))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let f = i.float("multiplier")?;
            Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning")?.map_meta(|m| m.multiplier *= f))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ReferenceLatent", "ReferenceLatent", "model/conditioning")
                .input(link("conditioning", "CONDITIONING"))
                .input(link("latent", "LATENT").optional())
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let cond = i.conditioning("conditioning")?;
            let out = match i.opt_latent("latent")? {
                Some(l) => cond.map_meta(|m| m.reference_latents.push(l.clone())),
                None => cond.clone(),
            };
            Ok(NodeOutput::one(Value::Conditioning(out)))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("FluxKontextMultiReferenceLatentMethod", "Edit Model Reference Method", "model/conditioning/flux")
                .input(link("conditioning", "CONDITIONING"))
                .input(combo(
                    "reference_latents_method",
                    &["offset", "index", "uxo/uno", "index_timestep_zero", "index_timestep_zero_isolated", "index_timestep_zero_text_start"],
                ))
                .output("CONDITIONING", "CONDITIONING")
        },
        |i, _| {
            let method = i.string("reference_latents_method")?.to_string();
            Ok(NodeOutput::one(Value::Conditioning(i.conditioning("conditioning")?.map_meta(|m| m.reference_latents_method = Some(method.clone())))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmMiniMaxH3CollectReferences", "Swarm MiniMax H3 Collect References", "SwarmUI/video")
                .describe("<Picture i> / <Video k> / <Audio j> reference collection for MiniMax H3. Feed the output into SwarmTextEncodeAdvanced.")
                .input(link("vae", "VAE"))
                .input(link("audio_vae", "VAE"))
                .input(int_step("width", 1344, 32, MAX_RESOLUTION, 32))
                .input(int_step("height", 768, 32, MAX_RESOLUTION, 32))
                .input(int_step("length", 124, 1, 3600, 1))
                .input(combo_default("ref_image_size", &["match", "max"], "match"))
                .input(autogrow("ref_images", "ref_image_", 0, 9, link("ref_image", "IMAGE")).optional())
                .input(autogrow("ref_videos", "ref_video_", 0, 3, link("ref_video", "IMAGE")).optional())
                .input(autogrow("ref_video_audios", "ref_video_audio_", 0, 3, link("ref_video_audio", "AUDIO")).optional())
                .input(autogrow("ref_audios", "ref_audio_", 0, 3, link("ref_audio", "AUDIO")).optional())
                .output("MiniMaxReferences", "MiniMax References")
        },
        |i, _| {
            let images = i.group("ref_images").into_iter().filter_map(|(_, v)| if let Value::Image(b) = v { Some(b.clone()) } else { None }).collect();
            let audios = i.group("ref_audios").into_iter().filter_map(|(_, v)| if let Value::Audio(a) = v { Some(a.clone()) } else { None }).collect();
            let soundtracks = i.group("ref_video_audios");
            let videos = i
                .group("ref_videos")
                .into_iter()
                .filter_map(|(idx, v)| {
                    let Value::Image(frames) = v else { return None };
                    let track = soundtracks.iter().find(|(j, _)| *j == idx).and_then(|(_, a)| if let Value::Audio(a) = a { Some((*a).clone()) } else { None });
                    Some((frames.clone(), track))
                })
                .collect();
            Ok(NodeOutput::one(Value::MiniMaxRefs(MiniMaxReferences {
                vae: i.vae("vae")?.clone(),
                audio_vae: i.vae("audio_vae")?.clone(),
                width: i.uint("width")?,
                height: i.uint("height")?,
                length: i.uint("length")?,
                ref_image_size: i.string("ref_image_size")?.to_string(),
                images,
                videos,
                audios,
            })))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmMiniMaxH3AddKeyframes", "Swarm MiniMax H3 Add Keyframes", "SwarmUI/video")
                .input(link("vae", "VAE"))
                .input(link("latent", "LATENT"))
                .input(link("conditioning", "CONDITIONING"))
                .input(link("first_frame", "IMAGE").optional())
                .input(link("last_frame", "IMAGE").optional())
                .output("CONDITIONING", "conditioning")
        },
        |i, ctx| {
            let latent = i.latent("latent")?;
            let (w, h) = (latent.pixel.width, latent.pixel.height);
            let (frame_count, _, _) = minimax_h3::temporal_shape(latent.pixel.frames);
            let mut keyframes = Vec::new();
            for (name, index) in [("first_frame", 0), ("last_frame", frame_count.saturating_sub(1))] {
                if let Some(img) = i.opt_image(name)? {
                    let pixels = encode::resize(&img.slice(0, 1), w, h, "lanczos", "disabled")?;
                    let encoded = ctx.engine.vae_encode(&VaeEncodeRequest { vae: i.vae("vae")?.clone(), pixels, tiling: None })?;
                    keyframes.push(MiniMaxKeyframe { frame_index: index, latent: encoded });
                }
            }
            let cond = i.conditioning("conditioning")?;
            if keyframes.is_empty() {
                return Ok(NodeOutput::one(Value::Conditioning(cond.clone())));
            }
            Ok(NodeOutput::one(Value::Conditioning(cond.map_meta(|m| {
                m.minimax_keyframes = keyframes.clone();
                m.minimax_frame_count = Some(frame_count);
            }))))
        },
    );
}
