//! Concrete pixel and audio buffers. These are the only tensors the server
//! itself reads or writes (to decode SwarmUI's base64 inputs and to encode
//! outputs), so they use ComfyUI's layouts.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

/// Pixel storage. ComfyUI always uses `f32`, but a 5 second 960x960 MiniMax H3
/// clip is ~1.4 GB as `f32`, so engines may hand back `U8` frames instead.
#[derive(Clone)]
pub enum PixelData {
    /// `0.0..=1.0`
    F32(Arc<Vec<f32>>),
    /// `0..=255`
    U8(Arc<Vec<u8>>),
}

impl PixelData {
    pub fn len(&self) -> usize {
        match self {
            PixelData::F32(v) => v.len(),
            PixelData::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A batch of images (or video frames): ComfyUI `IMAGE`, layout `[B, H, W, C]`.
/// `C == 3` (RGB) for almost everything; `C == 4` (RGBA, straight alpha, 1.0 = opaque)
/// for alpha-capable VAEs such as Qwen Image 2.1's.
#[derive(Clone)]
pub struct ImageBatch {
    pub count: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub data: PixelData,
}

impl ImageBatch {
    pub fn from_f32(count: u32, height: u32, width: u32, channels: u32, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), (count * height * width * channels) as usize, "ImageBatch data length mismatch");
        Self { count, height, width, channels, data: PixelData::F32(Arc::new(data)) }
    }

    pub fn from_u8(count: u32, height: u32, width: u32, channels: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), (count * height * width * channels) as usize, "ImageBatch data length mismatch");
        Self { count, height, width, channels, data: PixelData::U8(Arc::new(data)) }
    }

    pub fn frame_len(&self) -> usize {
        (self.height * self.width * self.channels) as usize
    }

    fn range(&self, index: u32) -> std::ops::Range<usize> {
        let len = self.frame_len();
        let start = index as usize * len;
        start..start + len
    }

    /// One frame as `f32` (borrowed when stored as `f32`).
    pub fn frame_f32(&self, index: u32) -> Cow<'_, [f32]> {
        match &self.data {
            PixelData::F32(v) => Cow::Borrowed(&v[self.range(index)]),
            PixelData::U8(v) => Cow::Owned(v[self.range(index)].iter().map(|&b| b as f32 / 255.0).collect()),
        }
    }

    pub fn has_alpha(&self) -> bool {
        self.channels == 4
    }

    /// One frame as 8-bit, all channels (borrowed when stored as `u8`).
    pub fn frame_u8(&self, index: u32) -> Cow<'_, [u8]> {
        match &self.data {
            PixelData::U8(v) => Cow::Borrowed(&v[self.range(index)]),
            PixelData::F32(v) => Cow::Owned(v[self.range(index)].iter().map(|&f| (f.clamp(0.0, 1.0) * 255.0).round() as u8).collect()),
        }
    }

    /// One frame as 16-bit, all channels (for `bit_depth = "16bit"`).
    pub fn frame_u16(&self, index: u32) -> Vec<u16> {
        self.frame_f32(index).iter().map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect()
    }

    /// All frames as one contiguous 8-bit buffer, all channels.
    pub fn all_u8(&self) -> Cow<'_, [u8]> {
        match &self.data {
            PixelData::U8(v) => Cow::Borrowed(&v[..]),
            PixelData::F32(v) => Cow::Owned(v.iter().map(|&f| (f.clamp(0.0, 1.0) * 255.0).round() as u8).collect()),
        }
    }

    /// Selects `length` frames starting at `start` (ComfyUI `ImageFromBatch`, clamped the same way).
    pub fn slice(&self, start: u32, length: u32) -> Self {
        let start = start.min(self.count.saturating_sub(1));
        let length = length.min(self.count - start).max(1);
        let len = self.frame_len();
        let r = start as usize * len..(start + length) as usize * len;
        let data = match &self.data {
            PixelData::F32(v) => PixelData::F32(Arc::new(v[r].to_vec())),
            PixelData::U8(v) => PixelData::U8(Arc::new(v[r].to_vec())),
        };
        Self { count: length, height: self.height, width: self.width, channels: self.channels, data }
    }

    /// Drops the alpha channel (ComfyUI `SplitImageWithAlpha`'s image output). Clones if already RGB.
    pub fn to_rgb(&self) -> Self {
        self.map_channels(3, |px, out| out.extend_from_slice(&px[..3]))
    }

    /// Adds an opaque alpha channel (what ComfyUI's `ImageBatch` pads with). Clones if already RGBA.
    pub fn to_rgba(&self) -> Self {
        self.map_channels(4, |px, out| {
            out.extend_from_slice(px);
            out.push(1.0);
        })
    }

    /// The alpha channel per pixel, `[B, H, W]` (all 1.0 for RGB).
    pub fn alpha(&self) -> Vec<f32> {
        let c = self.channels as usize;
        let mut out = Vec::with_capacity((self.count * self.height * self.width) as usize);
        for i in 0..self.count {
            let f = self.frame_f32(i);
            if c == 4 {
                out.extend(f.chunks_exact(4).map(|p| p[3]));
            } else {
                out.extend(std::iter::repeat(1.0).take(f.len() / c));
            }
        }
        out
    }

    fn map_channels(&self, to: u32, f: impl Fn(&[f32], &mut Vec<f32>)) -> Self {
        if self.channels == to {
            return self.clone();
        }
        let c = self.channels as usize;
        let mut out = Vec::with_capacity((self.count * self.height * self.width * to) as usize);
        for i in 0..self.count {
            for px in self.frame_f32(i).chunks_exact(c) {
                f(px, &mut out);
            }
        }
        Self::from_f32(self.count, self.height, self.width, to, out)
    }

    /// A new batch made of the given frames, in order (repeats allowed). Used for boomerang /
    /// trim. Panics on an out-of-range index or an empty list.
    pub fn frames(&self, indices: &[u32]) -> Self {
        assert!(!indices.is_empty(), "an image batch needs at least one frame");
        let len = self.frame_len();
        let data = match &self.data {
            PixelData::U8(v) => PixelData::U8(Arc::new(indices.iter().flat_map(|&i| &v[i as usize * len..(i as usize + 1) * len]).copied().collect())),
            PixelData::F32(v) => PixelData::F32(Arc::new(indices.iter().flat_map(|&i| &v[i as usize * len..(i as usize + 1) * len]).copied().collect())),
        };
        Self { count: indices.len() as u32, height: self.height, width: self.width, channels: self.channels, data }
    }

    /// Concatenates two batches of identical frame size. (ComfyUI's `ImageBatch` node resizes
    /// the second input first; the server does that before calling this.)
    pub fn concat(&self, other: &ImageBatch) -> Self {
        assert_eq!((self.width, self.height, self.channels), (other.width, other.height, other.channels));
        let data = match (&self.data, &other.data) {
            (PixelData::U8(a), PixelData::U8(b)) => PixelData::U8(Arc::new([a.as_slice(), b.as_slice()].concat())),
            _ => {
                let mut v = Vec::with_capacity(self.data.len() + other.data.len());
                for i in 0..self.count {
                    v.extend_from_slice(&self.frame_f32(i));
                }
                for i in 0..other.count {
                    v.extend_from_slice(&other.frame_f32(i));
                }
                PixelData::F32(Arc::new(v))
            }
        };
        Self { count: self.count + other.count, height: self.height, width: self.width, channels: self.channels, data }
    }
}

/// ComfyUI `MASK`, layout `[B, H, W]`, `f32` in `0.0..=1.0`.
#[derive(Clone)]
pub struct Mask {
    pub count: u32,
    pub height: u32,
    pub width: u32,
    pub data: Arc<Vec<f32>>,
}

/// ComfyUI `AUDIO`: `{"waveform": [B, C, N], "sample_rate": int}`. Stored planar,
/// one `Vec<f32>` per channel (`-1.0..=1.0`), batch size 1.
#[derive(Clone)]
pub struct AudioData {
    pub sample_rate: u32,
    pub channels: Vec<Arc<Vec<f32>>>,
}

impl AudioData {
    pub fn num_samples(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.num_samples() as f64 / self.sample_rate as f64
    }

    /// Mono is duplicated into two channels (as ComfyUI's `AudioConcat` does); anything else is unchanged.
    pub fn to_stereo(&self) -> Self {
        match self.channels.as_slice() {
            [mono] => Self { sample_rate: self.sample_rate, channels: vec![mono.clone(), mono.clone()] },
            _ => self.clone(),
        }
    }

    /// Linear-interpolation resample. (ComfyUI uses a windowed-sinc resampler; this is the
    /// server-side approximation for joining clips, not an inference input.)
    pub fn resampled(&self, rate: u32) -> Self {
        if rate == self.sample_rate || self.sample_rate == 0 {
            return self.clone();
        }
        let n_out = (self.num_samples() as f64 * rate as f64 / self.sample_rate as f64).round() as usize;
        let step = self.sample_rate as f64 / rate as f64;
        let channels = self
            .channels
            .iter()
            .map(|c| {
                let out: Vec<f32> = (0..n_out)
                    .map(|i| {
                        let pos = i as f64 * step;
                        let j = pos.floor() as usize;
                        let t = (pos - j as f64) as f32;
                        let a = c.get(j).copied().unwrap_or(0.0);
                        let b = c.get(j + 1).copied().unwrap_or(a);
                        a + (b - a) * t
                    })
                    .collect();
                Arc::new(out)
            })
            .collect();
        Self { sample_rate: rate, channels }
    }

    /// ComfyUI `AudioConcat`: `self` then `other`. Mono inputs become stereo and the
    /// lower sample rate is resampled up to the higher one.
    pub fn concat(&self, other: &AudioData) -> Self {
        let rate = self.sample_rate.max(other.sample_rate);
        let (a, b) = (self.to_stereo().resampled(rate), other.to_stereo().resampled(rate));
        assert_eq!(a.channels.len(), b.channels.len(), "can't join audio with different channel counts");
        let channels = a.channels.iter().zip(&b.channels).map(|(x, y)| Arc::new([x.as_slice(), y.as_slice()].concat())).collect();
        Self { sample_rate: rate, channels }
    }

    /// `SwarmEnsureAudio`: prepends silence so the clip lasts at least `seconds`.
    pub fn pad_start_to(&self, seconds: f64) -> Self {
        let missing = (seconds * self.sample_rate as f64) as i64 - self.num_samples() as i64;
        if missing <= 0 {
            return self.clone();
        }
        let pad = vec![0.0f32; missing as usize];
        let channels = self.channels.iter().map(|c| Arc::new([pad.as_slice(), c.as_slice()].concat())).collect();
        Self { sample_rate: self.sample_rate, channels }
    }
}

/// ComfyUI `VIDEO` (as produced by `SwarmLoadVideoB64` / consumed by `GetVideoComponents`).
#[derive(Clone, Debug)]
pub struct VideoClip {
    pub frames: ImageBatch,
    pub fps: f64,
    pub audio: Option<AudioData>,
}

// Summaries only: these hold megabytes of samples, and requests get logged with `{:?}`.

impl fmt::Debug for PixelData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelData::F32(v) => write!(f, "F32[{}]", v.len()),
            PixelData::U8(v) => write!(f, "U8[{}]", v.len()),
        }
    }
}

impl fmt::Debug for ImageBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageBatch({}x{}x{}, {} frame(s), {:?})", self.width, self.height, self.channels, self.count, self.data)
    }
}

impl fmt::Debug for Mask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mask({}x{}, {} frame(s))", self.width, self.height, self.count)
    }
}

impl fmt::Debug for AudioData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AudioData({} Hz, {} ch, {:.2}s)", self.sample_rate, self.channels.len(), self.duration_seconds())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_helpers() {
        let rgb = ImageBatch::from_u8(1, 1, 2, 3, vec![255, 0, 0, 0, 255, 0]);
        let rgba = rgb.to_rgba();
        assert!(rgba.has_alpha());
        assert_eq!(rgba.alpha(), vec![1.0, 1.0]);
        assert_eq!(rgba.to_rgb().frame_u8(0).as_ref(), rgb.frame_u8(0).as_ref());
        let half = ImageBatch::from_f32(1, 1, 1, 4, vec![0.2, 0.4, 0.6, 0.5]);
        assert_eq!(half.alpha(), vec![0.5]);
        assert_eq!(half.to_rgb().frame_f32(0).as_ref(), &[0.2, 0.4, 0.6]);
    }

    #[test]
    fn frame_selection() {
        let b = ImageBatch::from_u8(3, 1, 1, 1, vec![10, 20, 30]);
        let boomerang = b.frames(&[0, 1, 2, 2, 1, 0]);
        assert_eq!(boomerang.count, 6);
        assert_eq!(boomerang.all_u8().as_ref(), &[10, 20, 30, 30, 20, 10]);
    }

    #[test]
    fn audio_helpers() {
        let mono = AudioData { sample_rate: 8000, channels: vec![Arc::new(vec![0.5; 8000])] };
        let stereo = AudioData { sample_rate: 16000, channels: vec![Arc::new(vec![0.1; 4000]), Arc::new(vec![0.2; 4000])] };
        let joined = mono.concat(&stereo);
        assert_eq!(joined.sample_rate, 16000);
        assert_eq!(joined.channels.len(), 2);
        assert!((joined.duration_seconds() - 1.25).abs() < 1e-9);
        assert_eq!(joined.channels[1][16000 + 10], 0.2);
        let padded = stereo.pad_start_to(0.5);
        assert_eq!(padded.num_samples(), 8000);
        assert_eq!((padded.channels[0][0], padded.channels[0][7999]), (0.0, 0.1));
        assert_eq!(stereo.pad_start_to(0.1).num_samples(), 4000);
    }
}
