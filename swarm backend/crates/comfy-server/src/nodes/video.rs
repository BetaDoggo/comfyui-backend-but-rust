//! Video and audio plumbing SwarmUI wraps around generations: Video Boomerang, Trim Video
//! Start/End Frames, Video End Image, Video Audio Input, Audio Silent Prefix/Suffix, and the
//! frame math behind `<extend:N>`. Everything here is pure data except the audio VAE calls.

use std::sync::Arc;

use swarm_engine_api::{AudioData, AudioSilenceRequest, AudioTiling, ImageBatch, Latent, Mask, VaeDecodeAudioRequest, VaeEncodeAudioRequest};

use super::{input_err, NodeOutput, NodeRegistry};
use crate::encode;
use crate::env::ServerEnv;
use crate::schema::*;
use crate::value::Value;

/// Width / height / frame count of an IMAGE as an INT (SwarmUI's `<extend:N>` frame math).
fn image_int(r: &mut NodeRegistry, env: &ServerEnv, class: &'static str, display: &'static str, desc: &'static str, f: fn(&ImageBatch) -> u32) {
    r.add(
        env,
        move |_| NodeSchema::new(class, display, "SwarmUI/images").describe(desc).input(link("image", "IMAGE")).output("INT", "INT"),
        move |i, _| Ok(NodeOutput::one(Value::Int(f(i.image("image")?) as i64))),
    );
}

pub fn register(r: &mut NodeRegistry, env: &ServerEnv) {
    // ------------------------------------------------------------- frames
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmVideoBoomerang", "Swarm Video Boomerang", "SwarmUI/video")
                .describe("Creates a boomerang effect by having the video play in reverse after the end, as a simple trick to make it appear to loop smoothly.")
                .input(link("images", "IMAGE"))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("images")?;
            // torch.cat((images, images.flip(0))): the last and first frames both appear twice.
            let order: Vec<u32> = (0..img.count).chain((0..img.count).rev()).collect();
            Ok(NodeOutput::one(Value::Image(img.frames(&order))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmTrimFrames", "Swarm Trim Frames", "SwarmUI/images")
                .describe("Trims frames from the start and end of a video.")
                .input(link("image", "IMAGE"))
                .input(int("trim_start", 0, 0, 4096))
                .input(int("trim_end", 0, 0, 4096))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let img = i.image("image")?;
            if img.count <= 1 {
                return Ok(NodeOutput::one(Value::Image(img.clone())));
            }
            let start = i.uint("trim_start")?.min(img.count);
            let end = img.count - i.uint("trim_end")?.min(img.count);
            if start >= end {
                return Err(input_err(format!("trimming {start} start and {} end frame(s) leaves none of the {}", img.count - end, img.count)));
            }
            Ok(NodeOutput::one(Value::Image(img.frames(&(start..end).collect::<Vec<_>>()))))
        },
    );
    image_int(r, env, "SwarmCountFrames", "Swarm Count Frames", "Counts the number of frames in an image.", |b| b.count);
    image_int(r, env, "SwarmImageWidth", "Swarm Image Width", "Gets the width of an image.", |b| b.width);
    image_int(r, env, "SwarmImageHeight", "Swarm Image Height", "Gets the height of an image.", |b| b.height);
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmIntAdd", "Swarm Int Add", "SwarmUI/math")
                .describe("Adds two integers. Use a negative number to subtract.")
                .input(int("a", 0, -2147483647, 2147483647))
                .input(int("b", 0, -2147483647, 2147483647))
                .output("INT", "INT")
        },
        |i, _| Ok(NodeOutput::one(Value::Int(i.int("a")? + i.int("b")?))),
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("BatchImagesNode", "Batch Images", "image/batch")
                .input(autogrow("images", "image", 1, 50, link("image", "IMAGE")))
                .output("IMAGE", "IMAGE")
        },
        |i, _| {
            let mut images: Vec<ImageBatch> = i.group("images").into_iter().filter_map(|(_, v)| if let Value::Image(b) = v { Some(b.clone()) } else { None }).collect();
            if images.is_empty() {
                return Err(input_err("BatchImagesNode needs at least one image"));
            }
            // ComfyUI's batch_images: pad RGB with opaque alpha if any input has it, then resize
            // everything to the first image (bilinear, center crop).
            if images.iter().any(|b| b.channels == 4) {
                images = images.iter().map(ImageBatch::to_rgba).collect();
            }
            let (w, h) = (images[0].width, images[0].height);
            let mut out = images[0].clone();
            for b in &images[1..] {
                out = out.concat(&encode::resize(b, w, h, "bilinear", "center")?);
            }
            Ok(NodeOutput::one(Value::Image(out)))
        },
    );

    // ------------------------------------------------------------- audio
    r.add(
        env,
        |_| {
            NodeSchema::new("AudioConcat", "Concatenate Audio", "audio")
                .describe("Concatenates the audio1 to audio2 in the specified direction.")
                .input(link("audio1", "AUDIO"))
                .input(link("audio2", "AUDIO"))
                .input(combo_default("direction", &["after", "before"], "after").tooltip("Whether to append audio2 after or before audio1."))
                .output("AUDIO", "AUDIO")
        },
        |i, _| {
            let (a, b) = (i.audio("audio1")?, i.audio("audio2")?);
            let joined = if i.string("direction")? == "before" { b.concat(a) } else { a.concat(b) };
            Ok(NodeOutput::one(Value::Audio(joined)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmEnsureAudio", "Swarm Ensure Audio", "SwarmUI/audio")
                .describe("Ensures audio exists and is at least the given duration")
                .input(link("audio", "AUDIO"))
                .input(float("target_duration", 0.0, 0.0, 9999999.0, 0.1))
                .output("AUDIO", "AUDIO")
        },
        |i, _| {
            // Swarm's fallback for "no audio" is an empty 48 kHz stereo clip.
            let audio = i.opt_audio("audio")?.cloned().unwrap_or_else(|| AudioData { sample_rate: 48000, channels: vec![Arc::new(vec![]), Arc::new(vec![])] });
            Ok(NodeOutput::one(Value::Audio(audio.pad_start_to(i.float("target_duration")?))))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("VAEEncodeAudio", "VAE Encode Audio", "model/latent").input(link("audio", "AUDIO")).input(link("vae", "VAE")).output("LATENT", "LATENT")
        },
        |i, ctx| {
            let req = VaeEncodeAudioRequest { vae: i.vae("vae")?.clone(), audio: i.audio("audio")?.clone() };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.vae_encode_audio(&req)?)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("VAEDecodeAudioTiled", "VAE Decode Audio (Tiled)", "model/latent")
                .input(link("samples", "LATENT"))
                .input(link("vae", "VAE"))
                .input(int_step("tile_size", 512, 32, 8192, 8))
                .input(int_step("overlap", 64, 0, 1024, 8))
                .output("AUDIO", "AUDIO")
        },
        |i, ctx| {
            let tiling = AudioTiling { tile_size: i.uint("tile_size")?, overlap: i.uint("overlap")? };
            let req = VaeDecodeAudioRequest { vae: i.vae("vae")?.clone(), latent: i.latent("samples")?.clone(), tiling: Some(tiling) };
            Ok(NodeOutput::one(Value::Audio(ctx.engine.vae_decode_audio(&req)?)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SwarmAudioSilentMaskPrefixSuffix", "Swarm Audio Silent Mask Prefix Suffix", "SwarmUI/latents")
                .describe("Encodes silent audio and writes it into the start and/or end of a latent audio clip, masking that prefix/suffix so sampling cannot update it.")
                .input(link("latent", "LATENT"))
                .input(link("vae", "VAE"))
                .input(float("prefix_duration", 0.1, 0.0, 9999.0, 0.001).tooltip("Seconds of silence to encode and mask off at the start of the audio latent."))
                .input(float("suffix_duration", 0.0, 0.0, 9999.0, 0.001).tooltip("Seconds of silence to encode and mask off at the end of the audio latent."))
                .output("LATENT", "LATENT")
        },
        |i, ctx| {
            let latent = i.latent("latent")?;
            let (prefix, suffix) = (i.float("prefix_duration")?, i.float("suffix_duration")?);
            if prefix <= 0.0 && suffix <= 0.0 {
                return Ok(NodeOutput::one(Value::Latent(latent.clone())));
            }
            let req = AudioSilenceRequest { vae: i.vae("vae")?.clone(), latent: latent.clone(), prefix_seconds: prefix, suffix_seconds: suffix };
            Ok(NodeOutput::one(Value::Latent(ctx.engine.audio_silence(&req)?)))
        },
    );

    // ------------------------------------------------------------- masks on latents
    r.add(
        env,
        |_| {
            NodeSchema::new("SolidMask", "SolidMask", "image/mask")
                .input(float("value", 1.0, 0.0, 1.0, 0.01))
                .input(int_step("width", 512, 1, MAX_RESOLUTION, 1))
                .input(int_step("height", 512, 1, MAX_RESOLUTION, 1))
                .output("MASK", "MASK")
        },
        |i, _| {
            let (w, h) = (i.uint("width")?, i.uint("height")?);
            let mask = Mask { count: 1, height: h, width: w, data: Arc::new(vec![i.float("value")? as f32; (w * h) as usize]) };
            Ok(NodeOutput::one(Value::Mask(mask)))
        },
    );
    r.add(
        env,
        |_| {
            NodeSchema::new("SetLatentNoiseMask", "Set Latent Noise Mask", "model/latent")
                .input(link("samples", "LATENT"))
                .input(link("mask", "MASK"))
                .output("LATENT", "LATENT")
        },
        |i, _| {
            // Stored as given; the engine resizes it to the latent (ComfyUI's reshape_mask).
            let latent = Latent { noise_mask: Some(i.mask("mask")?.clone()), ..i.latent("samples")?.clone() };
            Ok(NodeOutput::one(Value::Latent(latent)))
        },
    );
}
