//! Latent tensors (ComfyUI `LATENT`). The data is engine-owned; the shape and
//! format are tracked so the server can do bookkeeping (splitting joint AV
//! latents, reporting sizes) and so engines can sanity-check what they receive.

use crate::handles::Opaque;
use crate::media::Mask;

/// What a latent tensor's channels/strides mean. Informational: the engine decides
/// how to interpret data, but the server fills this in from the node that created it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentFormat {
    /// `EmptyLatentImage`: `[B, 4, H/8, W/8]` (SD1/SDXL family).
    Image4Ch,
    /// `EmptySD3LatentImage`: `[B, 16, H/8, W/8]`. Used by Krea 2 and Anima (Qwen-Image VAE family).
    Image16Ch,
    /// `EmptyFlux2LatentImage`: `[B, 128, H/16, W/16]`. Flux 2 VAE family (MOSS Image).
    Flux2,
    /// Qwen Image 2.1 RGBA VAE: `[B, 64, H/16, W/16]`. SwarmUI never creates this directly; see
    /// [`Latent::conform_empty`].
    QwenImage21,
    /// MiniMax H3 video half of the joint latent: `[B, 24, T, H/16, W/16]`,
    /// `T = 2` for <= 5 frames else `((frames - 5) / 17) * 5 + 2`.
    MiniMaxH3Video,
    /// MiniMax H3 audio half of the joint latent: `[B, 32, 2, round(frames / 24 * 40)]`.
    MiniMaxH3Audio,
    /// Produced by an engine operation whose format the server doesn't track (e.g. `VAEEncode`).
    EngineDefined,
}

/// One engine-owned latent tensor.
#[derive(Clone, Debug)]
pub struct LatentTensor {
    pub format: LatentFormat,
    pub shape: Vec<usize>,
    pub data: Opaque,
}

#[derive(Clone, Debug)]
pub enum LatentSamples {
    /// A plain image or video latent.
    Single(LatentTensor),
    /// A joint audio+video latent (ComfyUI wraps this in a `NestedTensor((video, audio))`).
    /// Created by `SwarmEmptyMiniMaxH3LatentAV`, split by `LTXVSeparateAVLatent`.
    AudioVideo { video: LatentTensor, audio: LatentTensor },
}

/// The media a latent decodes to. Filled in by the node that created the latent
/// and carried through sampling, so decoders (and the mock engine) know the
/// target size without re-deriving it from the tensor shape.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelInfo {
    pub width: u32,
    pub height: u32,
    /// 1 for still images.
    pub frames: u32,
    pub batch_size: u32,
    pub fps: Option<f64>,
    pub audio_seconds: Option<f64>,
}

impl PixelInfo {
    pub fn image(width: u32, height: u32, batch_size: u32) -> Self {
        Self { width, height, frames: 1, batch_size, fps: None, audio_seconds: None }
    }
}

/// ComfyUI `LATENT` dict: `{"samples": ..., "noise_mask": ..., "batch_index": ...}`.
#[derive(Clone, Debug)]
pub struct Latent {
    pub samples: LatentSamples,
    pub pixel: PixelInfo,
    /// `SetLatentNoiseMask`: where sampling may change the latent (1 = denoise, 0 = keep).
    /// As in ComfyUI the mask can be any resolution; the engine resizes it to the latent
    /// (`comfy.utils.reshape_mask`). `None` means "denoise everywhere". For
    /// [`LatentSamples::AudioVideo`] this is the video stream's mask; see `audio_noise_mask`.
    pub noise_mask: Option<Mask>,
    /// The audio stream's noise mask of an [`LatentSamples::AudioVideo`] latent (ComfyUI keeps a
    /// `NestedTensor` of masks). Set by `LTXVConcatAVLatent`, e.g. for Video Audio Input (all 0:
    /// keep the encoded audio) or Audio Silent Prefix/Suffix Duration.
    pub audio_noise_mask: Option<Mask>,
    pub batch_index: Option<Vec<i64>>,
    /// All zeros, straight from an `Empty*Latent*` node. The server sets this (engines don't
    /// need to) and clears it on sampler output. Samplers use it to decide whether
    /// [`Latent::conform_empty`] applies.
    pub empty: bool,
}

impl Latent {
    pub fn single(tensor: LatentTensor, pixel: PixelInfo) -> Self {
        Self { samples: LatentSamples::Single(tensor), pixel, noise_mask: None, audio_noise_mask: None, batch_index: None, empty: false }
    }

    /// Returns a copy with new samples but the same bookkeeping (what samplers return).
    pub fn with_samples(&self, samples: LatentSamples) -> Self {
        Self { samples, ..self.clone() }
    }

    /// ComfyUI's `fix_empty_latent_channels` as `SwarmKSampler` calls it: an empty latent whose
    /// channel count doesn't match the model is widened to `channels`, keeping its spatial size.
    /// SwarmUI relies on this for Qwen Image 2.1, sending `EmptyLatentImage` at half the
    /// resolution (`[B, 4, H/16, W/16]`) and expecting `[B, 64, H/16, W/16]` to be sampled and
    /// decoded at the full `W x H`. `downscale` is the model's VAE ratio (16 there), used to
    /// recompute [`PixelInfo`]. Returns `None` if the latent isn't empty or already matches.
    pub fn conform_empty(&self, format: LatentFormat, channels: usize, downscale: u32) -> Option<Latent> {
        let LatentSamples::Single(t) = &self.samples else { return None };
        if !self.empty || t.shape.len() != 4 || t.shape[1] == channels {
            return None;
        }
        let shape = vec![t.shape[0], channels, t.shape[2], t.shape[3]];
        let pixel = PixelInfo { width: t.shape[3] as u32 * downscale, height: t.shape[2] as u32 * downscale, ..self.pixel.clone() };
        Some(Latent { samples: LatentSamples::Single(LatentTensor { format, shape, data: t.data.clone() }), pixel, ..self.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latent(shape: Vec<usize>, empty: bool) -> Latent {
        let t = LatentTensor { format: LatentFormat::Image4Ch, shape, data: Opaque::none() };
        Latent { empty, ..Latent::single(t, PixelInfo::image(512, 384, 1)) }
    }

    #[test]
    fn conform_empty_matches_swarm_qwen21_trick() {
        // EmptyLatentImage(512x384) as sent for a 1024x768 Qwen Image 2.1 generation.
        let l = latent(vec![1, 4, 48, 64], true).conform_empty(LatentFormat::QwenImage21, 64, 16).unwrap();
        let LatentSamples::Single(t) = &l.samples else { panic!() };
        assert_eq!(t.shape, vec![1, 64, 48, 64]);
        assert_eq!(t.format, LatentFormat::QwenImage21);
        assert_eq!((l.pixel.width, l.pixel.height), (1024, 768));
    }

    #[test]
    fn conform_empty_leaves_real_latents_alone() {
        assert!(latent(vec![1, 4, 48, 64], false).conform_empty(LatentFormat::QwenImage21, 64, 16).is_none());
        assert!(latent(vec![1, 64, 48, 64], true).conform_empty(LatentFormat::QwenImage21, 64, 16).is_none());
    }
}
