//! Media encoding/decoding at the SwarmUI boundary: decoding base64 inputs and
//! encoding outputs in the formats `SwarmSave*WS` produce. Images and WAV are
//! handled natively; video and compressed audio go through ffmpeg exactly the
//! way SwarmUI's own Python nodes call it.

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use image::codecs::gif::{GifEncoder, Repeat};
use image::codecs::jpeg::JpegEncoder;
use image::{Delay, DynamicImage, Frame, ImageBuffer, ImageFormat, Rgb, Rgba};
use swarm_engine_api::{AudioData, ImageBatch, Mask, VideoClip};

// ------------------------------------------------------------------ images ---

/// One frame as an 8-bit `DynamicImage`, RGBA when the batch has alpha.
fn image8(batch: &ImageBatch, index: u32) -> Result<DynamicImage> {
    let (w, h, raw) = (batch.width, batch.height, batch.frame_u8(index).into_owned());
    Ok(match batch.channels {
        3 => DynamicImage::ImageRgb8(ImageBuffer::from_raw(w, h, raw).ok_or_else(|| anyhow!("bad frame size"))?),
        4 => DynamicImage::ImageRgba8(ImageBuffer::from_raw(w, h, raw).ok_or_else(|| anyhow!("bad frame size"))?),
        c => bail!("unsupported image channel count {c}"),
    })
}

/// JPEG has no alpha; it's dropped (SwarmUI previews never need it).
pub fn jpeg(batch: &ImageBatch, index: u32, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    DynamicImage::ImageRgb8(image8(batch, index)?.to_rgb8()).write_with_encoder(JpegEncoder::new_with_quality(&mut out, quality))?;
    Ok(out)
}

/// RGB or RGBA PNG, matching the batch (SwarmUI keeps the alpha of alpha-capable models).
pub fn png8(batch: &ImageBatch, index: u32) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    image8(batch, index)?.write_to(&mut out, ImageFormat::Png)?;
    Ok(out.into_inner())
}

pub fn png16(batch: &ImageBatch, index: u32) -> Result<Vec<u8>> {
    let (w, h, raw) = (batch.width, batch.height, batch.frame_u16(index));
    let img = match batch.channels {
        3 => DynamicImage::ImageRgb16(ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, raw).ok_or_else(|| anyhow!("bad frame size"))?),
        4 => DynamicImage::ImageRgba16(ImageBuffer::<Rgba<u16>, _>::from_raw(w, h, raw).ok_or_else(|| anyhow!("bad frame size"))?),
        c => bail!("unsupported image channel count {c}"),
    };
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, ImageFormat::Png)?;
    Ok(out.into_inner())
}

pub fn bmp(batch: &ImageBatch, index: u32) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    image8(batch, index)?.write_to(&mut out, ImageFormat::Bmp)?;
    Ok(out.into_inner())
}

pub fn gif_animated(batch: &ImageBatch, fps: f64) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = GifEncoder::new_with_speed(&mut out, 10);
        enc.set_repeat(Repeat::Infinite)?;
        let delay = Delay::from_numer_denom_ms((1000.0 / fps.max(0.01)).round() as u32, 1);
        for i in 0..batch.count {
            let rgba: ImageBuffer<Rgba<u8>, Vec<u8>> = image8(batch, i)?.to_rgba8();
            enc.encode_frame(Frame::from_parts(rgba, 0, 0, delay))?;
        }
    }
    Ok(out)
}

/// Decodes any common image format into an `IMAGE` (1 frame) and `MASK` (`1 - alpha`,
/// or all zeros when the image has no alpha), like ComfyUI's `LoadImage`.
pub fn decode_image(bytes: &[u8]) -> Result<(ImageBatch, Mask)> {
    let img = image::load_from_memory(bytes).context("decoding image")?;
    let (w, h) = (img.width(), img.height());
    let has_alpha = img.color().has_alpha();
    let rgba = img.to_rgba32f();
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    let mut mask = Vec::with_capacity((w * h) as usize);
    for p in rgba.pixels() {
        rgb.extend_from_slice(&[p[0], p[1], p[2]]);
        mask.push(if has_alpha { 1.0 - p[3] } else { 0.0 });
    }
    Ok((ImageBatch::from_f32(1, h, w, 3, rgb), Mask { count: 1, height: h, width: w, data: Arc::new(mask) }))
}

/// Resizes every frame (ComfyUI `ImageScale` / `common_upscale`). `crop = "center"` scales
/// to cover then center-crops; otherwise the image is stretched.
pub fn resize(batch: &ImageBatch, width: u32, height: u32, method: &str, crop: &str) -> Result<ImageBatch> {
    use image::imageops::{self, FilterType};
    if width == batch.width && height == batch.height {
        return Ok(batch.clone());
    }
    let filter = match method {
        "nearest-exact" | "nearest" => FilterType::Nearest,
        "bilinear" | "area" => FilterType::Triangle,
        "bicubic" => FilterType::CatmullRom,
        _ => FilterType::Lanczos3,
    };
    let crop_rect = |ow: u32, oh: u32| {
        let (ow, oh, tw, th) = (ow as f64, oh as f64, width as f64, height as f64);
        let (cw, ch) = if ow / oh > tw / th { ((oh * tw / th).round(), oh) } else { (ow, (ow * th / tw).round()) };
        (((ow - cw) / 2.0).round() as u32, ((oh - ch) / 2.0).round() as u32, cw as u32, ch as u32)
    };
    fn scale<P: image::Pixel<Subpixel = f32> + 'static>(
        frame: ImageBuffer<P, Vec<f32>>,
        crop: Option<(u32, u32, u32, u32)>,
        width: u32,
        height: u32,
        filter: FilterType,
    ) -> Vec<f32> {
        match crop {
            Some((x, y, w, h)) => imageops::resize(&imageops::crop_imm(&frame, x, y, w, h).to_image(), width, height, filter).into_raw(),
            None => imageops::resize(&frame, width, height, filter).into_raw(),
        }
    }
    let crop = (crop == "center").then(|| crop_rect(batch.width, batch.height));
    let mut out = Vec::with_capacity((batch.count * width * height * batch.channels) as usize);
    for i in 0..batch.count {
        let raw = batch.frame_f32(i).into_owned();
        let bad = || anyhow!("bad frame size");
        out.extend(match batch.channels {
            3 => scale(ImageBuffer::<Rgb<f32>, _>::from_raw(batch.width, batch.height, raw).ok_or_else(bad)?, crop, width, height, filter),
            4 => scale(ImageBuffer::<Rgba<f32>, _>::from_raw(batch.width, batch.height, raw).ok_or_else(bad)?, crop, width, height, filter),
            1 => scale(ImageBuffer::<image::Luma<f32>, _>::from_raw(batch.width, batch.height, raw).ok_or_else(bad)?, crop, width, height, filter),
            c => bail!("unsupported image channel count {c}"),
        });
    }
    Ok(ImageBatch::from_f32(batch.count, height, width, batch.channels, out))
}

/// Bilinear mask resize (ComfyUI `resize_mask`).
pub fn resize_mask(mask: &Mask, width: u32, height: u32) -> Result<Mask> {
    use image::imageops::{self, FilterType};
    if mask.width == width && mask.height == height {
        return Ok(mask.clone());
    }
    let plane = (mask.width * mask.height) as usize;
    let mut out = Vec::with_capacity((mask.count * width * height) as usize);
    for b in 0..mask.count as usize {
        let frame: ImageBuffer<image::Luma<f32>, Vec<f32>> =
            ImageBuffer::from_raw(mask.width, mask.height, mask.data[b * plane..(b + 1) * plane].to_vec()).ok_or_else(|| anyhow!("bad mask size"))?;
        out.extend(imageops::resize(&frame, width, height, FilterType::Triangle).into_raw());
    }
    Ok(Mask { count: mask.count, height, width, data: Arc::new(out) })
}

// ------------------------------------------------------------------- audio ---

/// 16-bit PCM WAV.
pub fn wav(audio: &AudioData) -> Vec<u8> {
    let channels = audio.channels.len().max(1) as u16;
    let n = audio.num_samples();
    let data_len = (n * channels as usize * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&audio.sample_rate.to_le_bytes());
    out.extend_from_slice(&(audio.sample_rate * channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..n {
        for c in &audio.channels {
            let s = (c[i].clamp(-1.0, 1.0) * 32767.0) as i16;
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

/// Parses PCM (8/16/24/32-bit int) or IEEE float WAV.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioData> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a WAV file");
    }
    let mut pos = 12;
    let (mut fmt, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            fmt = u16::from_le_bytes(body[0..2].try_into()?);
            channels = u16::from_le_bytes(body[2..4].try_into()?);
            rate = u32::from_le_bytes(body[4..8].try_into()?);
            bits = u16::from_le_bytes(body[14..16].try_into()?);
            if fmt == 0xFFFE && body.len() >= 26 {
                fmt = u16::from_le_bytes(body[24..26].try_into()?);
            }
        } else if id == b"data" {
            if channels == 0 {
                bail!("WAV data before fmt chunk");
            }
            let bps = (bits / 8) as usize;
            let frame = bps * channels as usize;
            let n = body.len() / frame;
            let mut chans = vec![Vec::with_capacity(n); channels as usize];
            for i in 0..n {
                for (c, chan) in chans.iter_mut().enumerate() {
                    let s = &body[i * frame + c * bps..i * frame + (c + 1) * bps];
                    let v = match (fmt, bits) {
                        (3, 32) => f32::from_le_bytes(s.try_into()?),
                        (1, 8) => (s[0] as f32 - 128.0) / 128.0,
                        (1, 16) => i16::from_le_bytes(s.try_into()?) as f32 / 32768.0,
                        (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8388608.0,
                        (1, 32) => i32::from_le_bytes(s.try_into()?) as f32 / 2147483648.0,
                        _ => bail!("unsupported WAV format {fmt} / {bits} bits"),
                    };
                    chan.push(v);
                }
            }
            return Ok(AudioData { sample_rate: rate, channels: chans.into_iter().map(Arc::new).collect() });
        }
        pos += 8 + len + (len & 1);
    }
    bail!("WAV has no data chunk")
}

/// Decodes any audio container: WAV natively, everything else through ffmpeg.
pub fn decode_audio(ffmpeg: &str, temp_dir: &Path, bytes: &[u8]) -> Result<AudioData> {
    if let Ok(a) = decode_wav(bytes) {
        return Ok(a);
    }
    let tmp = TempFiles::new(temp_dir);
    let input = tmp.path("in.bin");
    std::fs::write(&input, bytes)?;
    let out = tmp.path("out.wav");
    run_ffmpeg(ffmpeg, &["-v", "error", "-y", "-i", path_str(&input)?, "-vn", "-acodec", "pcm_f32le", path_str(&out)?], None)?;
    decode_wav(&std::fs::read(&out)?)
}

// ----------------------------------------------------------------- ffmpeg ---

/// Deletes its files on drop.
struct TempFiles {
    dir: PathBuf,
    stem: String,
    made: std::cell::RefCell<Vec<PathBuf>>,
}

impl TempFiles {
    fn new(dir: &Path) -> Self {
        Self { dir: dir.to_path_buf(), stem: format!("swarm_tmp_{}", uuid::Uuid::new_v4().simple()), made: Default::default() }
    }

    fn path(&self, suffix: &str) -> PathBuf {
        let p = self.dir.join(format!("{}_{suffix}", self.stem));
        self.made.borrow_mut().push(p.clone());
        p
    }
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        for p in self.made.borrow().iter() {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str().ok_or_else(|| anyhow!("non-UTF8 temp path"))
}

/// Runs ffmpeg, optionally feeding `stdin`, returning `(stdout, stderr)`.
fn run_ffmpeg(ffmpeg: &str, args: &[&str], stdin: Option<&[u8]>) -> Result<(Vec<u8>, String)> {
    let mut child = Command::new(ffmpeg)
        .args(args)
        .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start ffmpeg ('{ffmpeg}'); install it or pass --ffmpeg"))?;
    let out = std::thread::scope(|s| {
        if let Some(data) = stdin {
            let mut pipe = child.stdin.take().unwrap();
            s.spawn(move || {
                let _ = pipe.write_all(data);
            });
        }
        child.wait_with_output()
    })?;
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        bail!("ffmpeg failed ({}): {}", out.status, stderr.trim());
    }
    Ok((out.stdout, stderr))
}

/// Encodes frames (+ optional audio) like `SwarmSaveAnimationWS`. Returns `(format code, bytes)`.
/// Formats: `webp`, `gif`, `gif-hd`, `h264-mp4`, `h265-mp4`, `webm`, `prores`.
pub fn encode_video(ffmpeg: &str, temp_dir: &Path, frames: &ImageBatch, fps: f64, format: &str, audio: Option<&AudioData>) -> Result<(u32, Vec<u8>)> {
    if format == "gif" {
        return Ok((4, gif_animated(frames, fps)?));
    }
    let (ext, code, mut video_args, audio_args): (&str, u32, Vec<&str>, Vec<&str>) = match format {
        "h264-mp4" => ("mp4", 5, vec!["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19"], vec!["-c:a", "aac"]),
        "h265-mp4" => ("mp4", 5, vec!["-c:v", "libx265", "-pix_fmt", "yuv420p"], vec!["-c:a", "aac"]),
        "webm" => ("webm", 6, vec!["-pix_fmt", "yuv420p", "-crf", "23"], vec!["-c:a", "libvorbis"]),
        "prores" => ("mov", 7, vec!["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], vec!["-c:a", "pcm_s16le"]),
        "gif-hd" => ("gif", 4, vec!["-filter_complex", "split=2 [a][b]; [a] palettegen [pal]; [b] [pal] paletteuse"], vec![]),
        "webp" => ("webp", 3, vec!["-c:v", "libwebp_anim", "-loop", "0", "-quality", "80"], vec![]),
        other => bail!("unknown video format '{other}'"),
    };
    let tmp = TempFiles::new(temp_dir);
    let out = tmp.path(&format!("out.{ext}"));
    let size = format!("{}x{}", frames.width, frames.height);
    let fps_s = format!("{fps}");
    let mut args: Vec<&str> = vec!["-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-r", &fps_s, "-i", "-", "-y"];
    let audio_path;
    if let (Some(a), false) = (audio, audio_args.is_empty()) {
        // Trim/pad to the video's duration, as the Swarm node does.
        let target = (frames.count as f64 / fps * a.sample_rate as f64) as usize;
        let fitted = AudioData {
            sample_rate: a.sample_rate,
            channels: a
                .channels
                .iter()
                .map(|c| {
                    let mut v: Vec<f32> = c.iter().copied().take(target).collect();
                    v.resize(target, 0.0);
                    Arc::new(v)
                })
                .collect(),
        };
        audio_path = tmp.path("audio.wav");
        std::fs::write(&audio_path, wav(&fitted))?;
        args.extend_from_slice(&["-i", path_str(&audio_path)?]);
        video_args.extend_from_slice(&audio_args);
    }
    args.extend_from_slice(&video_args);
    args.push(path_str(&out)?);
    run_ffmpeg(ffmpeg, &args, Some(&frames.to_rgb().all_u8()))?;
    Ok((code, std::fs::read(&out)?))
}

/// Encodes audio like `SwarmSaveAudioWS`. Returns `(format code, bytes)`.
pub fn encode_audio(ffmpeg: &str, temp_dir: &Path, audio: &AudioData, format: &str) -> Result<(u32, Vec<u8>)> {
    let (ext, code, codec_args): (&str, u32, &[&str]) = match format {
        "wav" => return Ok((9, wav(audio))),
        "mp3" => ("mp3", 8, &["-c:a", "libmp3lame", "-q:a", "1"]),
        "flac" => ("flac", 10, &["-c:a", "flac"]),
        "ogg" => ("ogg", 11, &["-c:a", "libopus", "-b:a", "128k", "-ar", "48000"]),
        other => bail!("unknown audio format '{other}'"),
    };
    Ok((code, ffmpeg_audio(ffmpeg, temp_dir, audio, ext, codec_args)?))
}

/// Encodes audio to a file of extension `ext` with the given ffmpeg codec arguments.
pub fn ffmpeg_audio(ffmpeg: &str, temp_dir: &Path, audio: &AudioData, ext: &str, codec_args: &[&str]) -> Result<Vec<u8>> {
    let tmp = TempFiles::new(temp_dir);
    let input = tmp.path("in.wav");
    let out = tmp.path(&format!("out.{ext}"));
    std::fs::write(&input, wav(audio))?;
    let mut args = vec!["-v", "error", "-y", "-i", path_str(&input)?];
    args.extend_from_slice(codec_args);
    args.push(path_str(&out)?);
    run_ffmpeg(ffmpeg, &args, None)?;
    Ok(std::fs::read(&out)?)
}

/// Decodes a video file (any container ffmpeg reads) into RGB frames, fps and audio.
pub fn decode_video(ffmpeg: &str, temp_dir: &Path, bytes: &[u8]) -> Result<VideoClip> {
    let tmp = TempFiles::new(temp_dir);
    let input = tmp.path("in.bin");
    std::fs::write(&input, bytes)?;
    let (raw, stderr) = run_ffmpeg(ffmpeg, &["-i", path_str(&input)?, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], None)?;
    let (width, height, fps) = parse_video_stream_info(&stderr).ok_or_else(|| anyhow!("couldn't read video stream info from ffmpeg"))?;
    let frame_len = (width * height * 3) as usize;
    if frame_len == 0 || raw.len() < frame_len {
        bail!("video has no frames");
    }
    let count = (raw.len() / frame_len) as u32;
    let frames = ImageBatch::from_u8(count, height, width, 3, raw[..count as usize * frame_len].to_vec());
    let wav_path = tmp.path("audio.wav");
    let audio = run_ffmpeg(ffmpeg, &["-v", "error", "-y", "-i", path_str(&input)?, "-vn", "-acodec", "pcm_f32le", path_str(&wav_path)?], None)
        .ok()
        .and_then(|_| decode_wav(&std::fs::read(&wav_path).ok()?).ok());
    Ok(VideoClip { frames, fps, audio })
}

/// Pulls `WxH` and `N fps` from ffmpeg's `Stream #0:0...: Video: ...` banner line.
fn parse_video_stream_info(stderr: &str) -> Option<(u32, u32, f64)> {
    let line = stderr.lines().find(|l| l.contains("Stream #") && l.contains("Video:"))?;
    let mut dims = None;
    for tok in line.split([' ', ',']) {
        if let Some((w, h)) = tok.split_once('x') {
            if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                dims = Some((w, h));
                break;
            }
        }
    }
    let fps = line.split(',').map(str::trim).find_map(|p| p.strip_suffix(" fps")?.parse::<f64>().ok()).unwrap_or(24.0);
    let (w, h) = dims?;
    Some((w, h, fps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip() {
        let a = AudioData { sample_rate: 16000, channels: vec![Arc::new(vec![0.0, 0.5, -0.5, 1.0])] };
        let back = decode_wav(&wav(&a)).unwrap();
        assert_eq!(back.sample_rate, 16000);
        assert_eq!(back.num_samples(), 4);
        assert!((back.channels[0][1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn image_roundtrip_and_resize() {
        let b = ImageBatch::from_u8(1, 2, 3, 3, vec![255; 18]);
        let (img, mask) = decode_image(&png8(&b, 0).unwrap()).unwrap();
        assert_eq!((img.width, img.height, mask.data[0]), (3, 2, 0.0));
        let r = resize(&img, 8, 4, "lanczos", "center").unwrap();
        assert_eq!((r.width, r.height, r.count), (8, 4, 1));
    }

    #[test]
    fn ffmpeg_banner_parse() {
        let s = "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 640x360 [SAR 1:1 DAR 16:9], 1203 kb/s, 24 fps, 24 tbr";
        assert_eq!(parse_video_stream_info(s), Some((640, 360, 24.0)));
    }
}
