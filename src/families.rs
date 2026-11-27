//! Each model family's built-in defaults, in one place: the files it loads from a models folder in SwarmUI's layout,
//! how it samples, and the sampler, steps and CFG the CLI uses when they aren't given. The CLI and the server's native
//! engine both read these, so a correction here reaches both. Paths are relative to the models folder.
//!
//! Adding a family starts with an entry here; `swarm backend/docs/ADDING_MODELS.md` has the rest.

use crate::samplers::ModelSampling;

/// A family's defaults.
#[derive(Clone, Copy, Debug)]
pub struct Defaults {
    /// What the CLI's messages call it
    pub name: &'static str,
    pub dit: &'static str,
    /// The text encoder, unless the checkpoint holds it
    pub te: Option<&'static str>,
    /// The VAE, unless the checkpoint holds it
    pub vae: Option<&'static str>,
    /// ComfyUI's model sampling for the family, or `None` for the sigma grid stored in the checkpoint
    pub sampling: Option<ModelSampling>,
    /// Whether `--shift` may replace the shift. Off where the model fixes its sigmas, or has no shift
    pub takes_shift: bool,
    pub sampler: &'static str,
    /// `None`: one step per interval of the checkpoint's sigma grid, which the model was distilled for
    pub steps: Option<usize>,
    pub cfg: f64,
}

impl Defaults {
    /// The family's model sampling with `shift` in place of its own, if given. `None` for a checkpoint's grid.
    pub fn sampling_with(&self, shift: Option<f64>) -> Option<ModelSampling> {
        let s = self.sampling?;
        Some(match (s, shift) {
            (ModelSampling::Flow { .. }, Some(shift)) => ModelSampling::Flow { shift },
            (ModelSampling::Flux { .. }, Some(shift)) => ModelSampling::Flux { shift },
            (ModelSampling::SenseNova { .. }, Some(shift)) => ModelSampling::SenseNova { shift },
            _ => s,
        })
    }
}

/// Krea 2 Turbo: ComfyUI's `ModelSamplingFlux` at its fixed shift.
pub const KREA2: Defaults = Defaults {
    name: "Krea 2",
    dit: "Stable-diffusion/kroma-v0.3.1-turbo-opd-W8A8.safetensors",
    te: Some("clip/qwen3vl_4b.safetensors"),
    vae: Some(QWEN_IMAGE_VAE),
    sampling: Some(ModelSampling::Flux { shift: 1.15 }),
    takes_shift: true,
    sampler: "euler",
    steps: Some(8),
    cfg: 1.0,
};

pub const ANIMA: Defaults = Defaults {
    name: "Anima",
    dit: "Stable-diffusion/animaInt8Mxfp8_turboV11Int8.safetensors",
    te: Some("clip/qwen_3_600m.safetensors"),
    vae: Some(QWEN_IMAGE_VAE),
    sampling: Some(ModelSampling::Flow { shift: 3.0 }),
    takes_shift: true,
    sampler: "er_sde",
    steps: Some(8),
    cfg: 1.0,
};

pub const MOSS: Defaults = Defaults {
    name: "MOSS Image",
    dit: "Stable-diffusion/moss_image_sft_int8_convrot_attn.safetensors",
    te: Some("clip/qwen3.5_9b_int8_convrot.safetensors"),
    vae: Some(FLUX2_VAE),
    sampling: Some(ModelSampling::Flow { shift: 5.0 }),
    takes_shift: true,
    sampler: "euler",
    steps: Some(8),
    cfg: 1.0,
};

/// Qwen Image 2.1: ComfyUI's `shift`, its mu at 1024x1024.
pub const QWEN21: Defaults = Defaults {
    name: "Qwen Image 2.1",
    dit: "Stable-diffusion/rayQwen21_v1Int8Convrot.safetensors",
    te: Some("clip/qwen3vl_8b_int8_convrot.safetensors"),
    vae: Some("VAE/QwenImage/qwen_image_2.1_vae_bf16.safetensors"),
    sampling: Some(ModelSampling::Flux { shift: 0.69 }),
    takes_shift: true,
    sampler: "euler",
    steps: Some(8),
    cfg: 1.0,
};

/// SD 1.x: CLIP and VAE come in the checkpoint.
pub const SD1: Defaults = Defaults {
    name: "SD 1.x",
    dit: "Stable-diffusion/v1-5-pruned-emaonly-fp16.safetensors",
    te: None,
    vae: None,
    sampling: Some(ModelSampling::Discrete),
    takes_shift: false,
    sampler: "euler",
    steps: Some(20),
    cfg: 7.0,
};

pub const SDXL: Defaults = Defaults { name: "SDXL", dit: "Stable-diffusion/sd_xl_base_1.0.safetensors", ..SD1 };

/// MiniMax H3, text to video with audio. Its video shift is fixed; the audio's is [`crate::h3::SHIFT_AUDIO`]. HyperFlow
/// H3 Flash checkpoints sample with `euler` instead (see [`crate::h3::hyperflow_config`]).
pub const H3: Defaults = Defaults {
    name: "MiniMax H3",
    dit: "Stable-diffusion/minimax_h3_fl2va_pruned_w6a8.safetensors",
    te: Some("clip/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    vae: Some("VAE/MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors"),
    sampling: Some(ModelSampling::Flow { shift: crate::h3::SHIFT_VIDEO }),
    takes_shift: false,
    sampler: "res_multistep",
    steps: Some(8),
    cfg: 1.0,
};

/// MiniMax H3's audio VAE, beside [`H3`]'s video VAE.
pub const H3_AUDIO_VAE: &str = "VAE/MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors";

/// MiniMax-H3-Image: H3's text encoder, and the sigma grid its heads were distilled on, from the checkpoint.
pub const H3_IMAGE: Defaults = Defaults {
    name: "MiniMax-H3-Image",
    dit: "Stable-diffusion/minimax_h3_image_pruned_int8_convrot.safetensors",
    vae: Some("VAE/MiniMaxH3/minimax_h3_image_vae_fp16.safetensors"),
    sampling: None,
    sampler: "euler",
    steps: None,
    ..H3
};

/// Nemotron-Labs-Image isn't diffusion: one file holds its generator, tokenizer and image decoder.
pub const NEMOTRON: &str = "diffusion_models/Nemotron-Labs-Image-30B-A3B-nvfp4.safetensors";

pub const QWEN_IMAGE_VAE: &str = "VAE/qwen_image_vae.safetensors";
pub const FLUX2_VAE: &str = "VAE/flux2-vae.safetensors";

// Restoration, upscaling, background removal and 3D, which the CLI runs as tools.
pub const SEEDVR2_DIT: &str = "diffusion_models/SeedVR2/seedvr2_3b_int8_convrot.safetensors";
pub const SEEDVR2_VAE: &str = "VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors";
pub const RELAYVSR_DIT: &str = "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors";
pub const RELAYVSR_VAE: &str = "VAE/relayvsr_flash_vae_x2_fp32.safetensors";
pub const BIREFNET: &str = "background_removal/birefnet.safetensors";
pub const NVIDIA_VSR: &str = "upscale_models/nvidia-vfx/nvngx_vsr.dll";
pub const PIXAL3D_DIT: &str = "diffusion_models/Pixal3D/pixal3d_int8_convrot.safetensors";
pub const PIXAL3D_DINO: &str = "clip_vision/dino_v3_L_naf_fp32.safetensors";
pub const PIXAL3D_SHAPE_VAE: &str = "VAE/Pixal3D/trellis_2_shape_vae_bf16.safetensors";
pub const PIXAL3D_TEXTURE_VAE: &str = "VAE/Pixal3D/trellis_2_texture_vae_bf16.safetensors";
/// Idol (TripoSplat), which decodes with [`FLUX2_VAE`] too.
pub const IDOL_DIT: &str = "diffusion_models/idol_fp16.safetensors";
pub const IDOL_DINO: &str = "clip_vision/dino_v3_vit_h_bf16.safetensors";
pub const IDOL_DECODER: &str = "VAE/idol_decoder_fp16.safetensors";

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Defaults; 8] = [KREA2, ANIMA, MOSS, QWEN21, SD1, SDXL, H3, H3_IMAGE];

    #[test]
    fn shift_replaces_only_a_shift() {
        assert_eq!(ANIMA.sampling_with(Some(2.0)), Some(ModelSampling::Flow { shift: 2.0 }));
        assert_eq!(KREA2.sampling_with(Some(2.0)), Some(ModelSampling::Flux { shift: 2.0 }));
        assert_eq!(KREA2.sampling_with(None), KREA2.sampling);
        assert_eq!(SD1.sampling_with(Some(2.0)), Some(ModelSampling::Discrete));
        assert_eq!(H3_IMAGE.sampling_with(Some(2.0)), None);
    }

    #[test]
    fn families_that_take_a_shift_have_one() {
        for f in ALL {
            let shifted = matches!(f.sampling, Some(ModelSampling::Flow { .. } | ModelSampling::Flux { .. }));
            assert!(!f.takes_shift || shifted, "{} takes --shift but has none", f.name);
            assert_eq!(f.sampling.is_none(), f.steps.is_none(), "{}: a grid sets the steps, and only a grid", f.name);
        }
    }
}
