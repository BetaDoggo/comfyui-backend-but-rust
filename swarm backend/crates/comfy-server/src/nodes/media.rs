//! Image / audio / video input and manipulation nodes. These run on the server
//! (no engine involvement) since they only touch concrete pixels and samples.

use std::sync::Arc;

use base64::Engine as _;
use swarm_engine_api::{AudioData, ImageBatch, Mask};

use super::{input_err, NodeError, NodeOutput, NodeRegistry};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

fn b64(data: &str) -> Result<Vec<u8>, NodeError> {
    let data = data.trim();
    let data = match data.find(";base64,") {
        Some(i) if data.starts_with("data:") => &data[i + 8..],
        _ => data,
    };
    base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| input_err(format!("invalid base64: {e}")))
}

fn list_input_files(env: &ServerEnv) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(&env.input_dir)
        .map(|rd| rd.flatten().filter(|e| e.path().is_file()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    out.sort();
    out
}

/// `SwarmVideoResampleFPS`: keeps duration, changes frame count.
fn resample_fps(frames: &ImageBatch, fps_in: f64, fps_out: f64, linear: bool) -> ImageBatch {
    let n = frames.count as usize;
    let out_n = ((n as f64 / fps_in * fps_out).round() as usize).max(1);
    let len = frames.frame_len();
    let mut data = Vec::with_capacity(out_n * len);
    for k in 0..out_n {
        let pos = (k as f64 * fps_in / fps_out).min((n - 1) as f64);
        let a = pos.floor() as u32;
        let b = (a + 1).min(frames.count - 1);
        let t = (pos - a as f64) as f32;
        if !linear || t == 0.0 || a == b {
            let idx = if linear { a } else { pos.round() as u32 };
            data.extend_from_slice(&frames.frame_f32(idx.min(frames.count - 1)));
        } else {
            let (fa, fb) = (frames.frame_f32(a), frames.frame_f32(b));
            data.extend(fa.iter().zip(fb.iter()).map(|(x, y)| x * (1.0 - t) + y * t));
        }
    }
    ImageBatch::from_f32(out_n as u32, frames.height, frames.width, frames.channels, data)
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmLoadImageB64", "Swarm Load Image B64", "SwarmUI/images")
                .describe("Loads an image from a base64 string.")
                .input(string("image_base64", "", true).no_default())
                .output("IMAGE", "IMAGE")
                .output("MASK", "MASK")
        },
        |i, _| {
            let (img, mask) = encode::decode_image(&b64(i.string("image_base64")?)?)?;
            Ok(NodeOutput::values(vec![Value::Image(img), Value::Mask(mask)]))
        },
    );

    r.add(
        env,
        |env| NodeSchema::new("LoadImage", "Load Image", "image").input(combo("image", &list_input_files(env))).output("IMAGE", "IMAGE").output("MASK", "MASK"),
        |i, ctx| {
            let path = ServerEnv::safe_join(&ctx.env.input_dir, "", i.string("image")?).ok_or_else(|| input_err("invalid image path"))?;
            let bytes = std::fs::read(&path).map_err(|e| input_err(format!("can't read input image: {e}")))?;
            let (img, mask) = encode::decode_image(&bytes)?;
            Ok(NodeOutput::values(vec![Value::Image(img), Value::Mask(mask)]))
        },
    );

    r.add(
        env,
        |_| NodeSchema::new("ImageBatch", "Batch Images", "image/batch").input(link("image1", "IMAGE")).input(link("image2", "IMAGE")).output("IMAGE", "IMAGE"),
        |i, _| {
            let (mut a, mut b) = (i.image("image1")?.clone(), i.image("image2")?.clone());
            // ComfyUI pads the RGB side with opaque alpha when batching RGB with RGBA.
            if a.channels != b.channels {
                (a, b) = (a.to_rgba(), b.to_rgba());
            }
            let b = encode::resize(&b, a.width, a.height, "bilinear", "center")?;
            Ok(NodeOutput::one(Value::Image(a.concat(&b))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SplitImageWithAlpha", "Split Image with Alpha", "image/compositing")
                .input(link("image", "IMAGE"))
                .output("IMAGE", "IMAGE")
                .output("MASK", "MASK")
        },
        |i, _| {
            let img = i.image("image")?;
            // ComfyUI masks are inverted alpha (1.0 = transparent).
            let mask: Vec<f32> = img.alpha().into_iter().map(|a| 1.0 - a).collect();
            let mask = Mask { count: img.count, height: img.height, width: img.width, data: Arc::new(mask) };
            Ok(NodeOutput::values(vec![Value::Image(img.to_rgb()), Value::Mask(mask)]))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("JoinImageWithAlpha", "Join Image with Alpha", "image/compositing")
                .input(link("image", "IMAGE"))
                .input(link("alpha", "MASK"))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("image")?;
            let mask = encode::resize_mask(i.mask("alpha")?, img.width, img.height)?;
            let n = img.count.max(mask.count);
            let plane = (img.width * img.height) as usize;
            let mut out = Vec::with_capacity(n as usize * plane * 4);
            for b in 0..n {
                let rgb = img.frame_f32(b.min(img.count - 1));
                let c = img.channels as usize;
                let m = &mask.data[(b.min(mask.count - 1) as usize) * plane..][..plane];
                for (px, mv) in rgb.chunks_exact(c).zip(m) {
                    out.extend_from_slice(&px[..3]);
                    out.push(1.0 - mv);
                }
            }
            Ok(NodeOutput::one(Value::Image(ImageBatch::from_f32(n, img.height, img.width, 4, out))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ImageFromBatch", "Get Image from Batch", "image/batch")
                .input(link("image", "IMAGE"))
                .input(int("batch_index", 0, -MAX_RESOLUTION, MAX_RESOLUTION))
                .input(int("length", 1, 1, 4096))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("image")?;
            let mut idx = i.int("batch_index")?;
            if idx < 0 {
                idx += img.count as i64;
            }
            Ok(NodeOutput::one(Value::Image(img.slice(idx.max(0) as u32, i.uint("length")?))))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("ImageScale", "Upscale Image", "image/upscaling")
                .input(link("image", "IMAGE"))
                .input(combo("upscale_method", &["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]))
                .input(int_step("width", 512, 0, MAX_RESOLUTION, 1))
                .input(int_step("height", 512, 0, MAX_RESOLUTION, 1))
                .input(combo("crop", &["disabled", "center"]))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("image")?;
            let (mut w, mut h) = (i.uint("width")?, i.uint("height")?);
            if w == 0 && h == 0 {
                return Ok(NodeOutput::one(Value::Image(img.clone())));
            }
            if w == 0 {
                w = ((img.width as f64) * h as f64 / img.height as f64).round().max(1.0) as u32;
            } else if h == 0 {
                h = ((img.height as f64) * w as f64 / img.width as f64).round().max(1.0) as u32;
            }
            Ok(NodeOutput::one(Value::Image(encode::resize(img, w, h, i.string("upscale_method")?, i.string("crop")?)?)))
        },
    );

    r.add(
        env,
        |_| NodeSchema::new("SwarmLoadAudioB64", "Swarm Load Audio B64", "SwarmUI/images").input(string("audio_base64", "", true).no_default()).output("AUDIO", "AUDIO"),
        |i, ctx| Ok(NodeOutput::one(Value::Audio(encode::decode_audio(&ctx.env.ffmpeg, &ctx.env.temp_dir, &b64(i.string("audio_base64")?)?)?))),
    );

    r.add(
        env,
        |_| NodeSchema::new("SwarmLoadVideoB64", "Swarm Load Video B64", "SwarmUI/images").input(string("video_base64", "", true).no_default()).output("VIDEO", "VIDEO"),
        |i, ctx| Ok(NodeOutput::one(Value::Video(encode::decode_video(&ctx.env.ffmpeg, &ctx.env.temp_dir, &b64(i.string("video_base64")?)?)?))),
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("GetVideoComponents", "Get Video Components", "video")
                .input(link("video", "VIDEO"))
                .output("IMAGE", "images")
                .output("AUDIO", "audio")
                .output("FLOAT", "fps")
                .output("COMBO", "bit_depth")
                .output("COMBO", "color_space")
        },
        |i, _| {
            let v = i.video("video")?;
            // Videos without a soundtrack get matching-length silence so downstream audio inputs still work.
            let audio = v.audio.clone().unwrap_or_else(|| AudioData {
                sample_rate: 44100,
                channels: vec![Arc::new(vec![0.0; (v.frames.count as f64 / v.fps * 44100.0) as usize])],
            });
            // Decoded frames are 8-bit sRGB (ComfyUI reports the stream's depth / "sRGB" by default).
            Ok(NodeOutput::values(vec![Value::Image(v.frames.clone()), Value::Audio(audio), Value::Float(v.fps), Value::Int(8), Value::Str("sRGB".into())]))
        },
    );

    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmVideoResampleFPS", "Swarm Video Resample FPS", "SwarmUI/video")
                .describe("Resample a video from fps_in to fps_out while preserving total duration.")
                .input(link("images", "IMAGE"))
                .input(float("fps_in", 24.0, 0.01, 99999.9, 1.0).no_default())
                .input(float("fps_out", 24.0, 0.01, 99999.9, 1.0))
                .input(combo_default("method", &["linear", "nearest"], "linear"))
                .output("IMAGE", "images")
        },
        |i, _| {
            let (fin, fout) = (i.float("fps_in")?, i.float("fps_out")?);
            let img = i.image("images")?;
            if (fin - fout).abs() < 1e-6 || img.count == 0 {
                return Ok(NodeOutput::one(Value::Image(img.clone())));
            }
            Ok(NodeOutput::one(Value::Image(resample_fps(img, fin, fout, i.string("method")? == "linear"))))
        },
    );
}
