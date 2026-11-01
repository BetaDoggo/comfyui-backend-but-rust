//! Output nodes: the websocket-streaming `SwarmSave*WS` nodes SwarmUI uses, the
//! file-saving ComfyUI fallbacks, and the model warmup node.

use serde_json::json;
use swarm_engine_api::WarmupRequest;

use super::{input_err, NodeError, NodeOutput, NodeRegistry, NodeResult, SWARM_OUTPUT_IMAGE, SWARM_OUTPUT_VIDEO};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// ComfyUI `BinaryEventTypes` value SwarmUI treats as "raw BMP" (used by `bit_depth = raw`).
const EVENT_RAW_BMP: u32 = 10;

/// Saves `images` as `{prefix}_{counter:05}_.png` in `dir`, ComfyUI style. Returns the `ui` entries.
/// ComfyUI's `get_save_image_path`: the target folder, subfolder, file stem and next free counter
/// for `prefix` under `dir`.
fn output_slot(dir: &std::path::Path, prefix: &str) -> Result<(std::path::PathBuf, String, String, u32), NodeError> {
    let prefix = prefix.replace('\\', "/");
    let (subfolder, stem) = match prefix.rsplit_once('/') {
        Some((s, f)) => (s.to_string(), f.to_string()),
        None => (String::new(), prefix.clone()),
    };
    let target = ServerEnv::safe_join(dir, &subfolder, "x").and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_else(|| dir.to_path_buf());
    std::fs::create_dir_all(&target).map_err(anyhow::Error::from)?;
    let counter = std::fs::read_dir(&target)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    name.strip_prefix(&format!("{stem}_"))?.get(..5)?.parse::<u32>().ok()
                })
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
        + 1;
    Ok((target, subfolder, stem, counter))
}

fn save_pngs(dir: &std::path::Path, type_name: &str, prefix: &str, images: &swarm_engine_api::ImageBatch) -> NodeResult {
    let (target, subfolder, stem, mut counter) = output_slot(dir, prefix)?;
    let mut results = Vec::new();
    for idx in 0..images.count {
        let filename = format!("{stem}_{counter:05}_.png");
        std::fs::write(target.join(&filename), encode::png8(images, idx)?).map_err(anyhow::Error::from)?;
        results.push(json!({ "filename": filename, "subfolder": subfolder, "type": type_name }));
        counter += 1;
    }
    Ok(NodeOutput { values: vec![Value::Image(images.clone())], ui: Some(json!({ "images": results })) })
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmSaveImageWS", "Swarm Save Image WS", "SwarmUI/images")
                .describe("Acts like a special version of 'SaveImage' that doesn't actual save to disk, instead it sends directly over websocket.")
                .input(link("images", "IMAGE"))
                .input(combo_default("bit_depth", &["8bit", "16bit", "raw"], "8bit").optional())
                .output_node()
        },
        |i, ctx| {
            let images = i.image("images")?;
            let depth = i.opt_string("bit_depth")?.unwrap_or("8bit");
            for idx in 0..images.count {
                ctx.check_interrupt()?;
                match depth {
                    "raw" => {
                        let bytes = encode::bmp(images, idx)?;
                        let node = ctx.node_id.clone();
                        ctx.events.progress(&node, SWARM_OUTPUT_IMAGE, SWARM_OUTPUT_IMAGE);
                        let mut payload = 1u32.to_be_bytes().to_vec();
                        payload.extend_from_slice(&bytes);
                        ctx.events.binary(EVENT_RAW_BMP, payload);
                    }
                    "16bit" => ctx.send_output(SWARM_OUTPUT_IMAGE, 2, &encode::png16(images, idx)?),
                    _ => ctx.send_output(SWARM_OUTPUT_IMAGE, 2, &encode::png8(images, idx)?),
                }
            }
            Ok(NodeOutput::empty())
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmSaveAnimationWS", "Swarm Save Animation WS", "SwarmUI/video")
                .input(link("images", "IMAGE"))
                .input(float("fps", 6.0, 0.01, 1000.0, 0.01))
                .input(boolean("lossless", true))
                .input(int("quality", 80, 0, 100))
                .input(combo("method", &["default", "fastest", "slowest"]))
                .input(combo("format", &["webp", "gif", "gif-hd", "h264-mp4", "h265-mp4", "webm", "prores"]))
                .input(link("audio", "AUDIO").optional())
                .output_node()
        },
        |i, ctx| {
            let images = i.image("images")?;
            if images.count == 0 {
                return Ok(NodeOutput::empty());
            }
            if images.count == 1 {
                ctx.send_output(SWARM_OUTPUT_IMAGE, 2, &encode::png8(images, 0)?);
                return Ok(NodeOutput::empty());
            }
            let fps = i.float("fps")?;
            let format = i.string("format")?;
            let audio = i.opt_audio("audio")?;
            let (code, bytes) = match encode::encode_video(&ctx.env.ffmpeg, &ctx.env.temp_dir, images, fps, format, audio) {
                Ok(v) => v,
                // Animated webp needs an ffmpeg built with libwebp; GIF is the closest native fallback.
                Err(e) if format == "webp" => {
                    tracing::warn!("webp encode failed ({e:#}), falling back to gif");
                    (4, encode::gif_animated(images, fps)?)
                }
                Err(e) => return Err(NodeError::Other(e)),
            };
            ctx.send_output(SWARM_OUTPUT_VIDEO, code, &bytes);
            Ok(NodeOutput::empty())
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmSaveAudioWS", "Swarm Save Audio WS", "SwarmUI/audio")
                .input(link("audio", "AUDIO"))
                .input(combo_default("format", &["mp3", "wav", "flac", "ogg"], "mp3"))
                .output_node()
        },
        |i, ctx| {
            let audio = i.audio("audio")?;
            let format = i.string("format")?;
            let (code, bytes) = match encode::encode_audio(&ctx.env.ffmpeg, &ctx.env.temp_dir, audio, format) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("{format} encode failed ({e:#}), sending wav instead");
                    (9, encode::wav(audio))
                }
            };
            ctx.send_output(SWARM_OUTPUT_IMAGE, code, &bytes);
            Ok(NodeOutput::empty())
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SaveAudioAdvanced", "Save Audio (Advanced)", "audio")
                .input(link("audio", "AUDIO").tooltip("The audio to save."))
                .input(string("filename_prefix", "audio/ComfyUI", false).tooltip("The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd%."))
                .input(
                    dynamic_combo(
                        "format",
                        vec![
                            ("flac", vec![]),
                            ("mp3", vec![combo_default("quality", &["V0", "128k", "320k"], "V0")]),
                            ("opus", vec![combo_default("quality", &["64k", "96k", "128k", "192k", "320k"], "128k")]),
                        ],
                    )
                    .tooltip("The file format in which to save the audio."),
                )
                .output("AUDIO", "audio")
                .output_node()
        },
        |i, ctx| {
            // SwarmUI's fallback when its own audio saver is missing (it always sends flac or mp3 V0).
            let audio = i.audio("audio")?;
            let format = i.string("format")?;
            let quality = i.opt_string("format.quality")?.unwrap_or("");
            let q: Vec<&str> = match (format, quality) {
                ("flac", _) => vec!["-c:a", "flac"],
                ("mp3", "V0") | ("mp3", "") => vec!["-c:a", "libmp3lame", "-q:a", "0"],
                ("mp3", b) => vec!["-c:a", "libmp3lame", "-b:a", b],
                ("opus", b) => vec!["-c:a", "libopus", "-b:a", if b.is_empty() { "128k" } else { b }, "-ar", "48000"],
                (other, _) => return Err(input_err(format!("unknown audio format '{other}'"))),
            };
            let ext = if format == "opus" { "opus" } else { format };
            let bytes = encode::ffmpeg_audio(&ctx.env.ffmpeg, &ctx.env.temp_dir, audio, ext, &q)?;
            let (target, subfolder, stem, counter) = output_slot(&ctx.env.output_dir, i.string("filename_prefix")?)?;
            let filename = format!("{stem}_{counter:05}_.{ext}");
            std::fs::write(target.join(&filename), bytes).map_err(anyhow::Error::from)?;
            let ui = json!({ "audio": [{ "filename": filename, "subfolder": subfolder, "type": "output" }] });
            Ok(NodeOutput { values: vec![Value::Audio(audio.clone())], ui: Some(ui) })
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SaveImage", "Save Image", "image")
                .input(link("images", "IMAGE"))
                .input(string("filename_prefix", "ComfyUI", false))
                .output("IMAGE", "images")
                .output_node()
        },
        |i, ctx| save_pngs(&ctx.env.output_dir, "output", i.string("filename_prefix")?, i.image("images")?),
    );

    r.add(
        env,
        |_| NodeSchema::new("PreviewImage", "Preview Image", "image").input(link("images", "IMAGE")).output("IMAGE", "images").output_node(),
        |i, ctx| save_pngs(&ctx.env.temp_dir, "temp", &format!("ComfyUI_temp_{}", &uuid::Uuid::new_v4().simple().to_string()[..5]), i.image("images")?),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmJustLoadTheModelPlease", "Swarm Just Load The Model Please", "SwarmUI/internal")
                .describe("Internal node that acts as a final output for a model/clip/vae. This allows swarm to load models when needed without generating anything.")
                .input(link("model", "MODEL"))
                .input(link("clip", "CLIP,GEMMA"))
                .input(link("vae", "VAE"))
                .output_node()
        },
        |i, ctx| {
            ctx.engine.warmup(&WarmupRequest { model: i.model("model")?.clone(), text_encoder: i.clip("clip")?.clone(), vae: i.vae("vae")?.clone() })?;
            Ok(NodeOutput::empty())
        },
    );
}
