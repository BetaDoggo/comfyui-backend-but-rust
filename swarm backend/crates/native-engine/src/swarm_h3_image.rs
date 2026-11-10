//! TEMPORARY, to be removed once SwarmUI supports MiniMax-H3-Image: lets SwarmUI run it before it knows the model.
//!
//! SwarmUI tells MiniMax H3 by its audio projection, which the image checkpoint lacks, so it treats the file as an
//! unknown model and sends its default graph: `CheckpointLoaderSimple`, with the CLIP and VAE from the loader's own
//! outputs, `SwarmTextEncodeAdvanced`, `EmptyLatentImage` (`[B, 4, H/8, W/8]`), `SwarmKSampler` and `VAEDecode`.
//! Here the loader hands back H3's text encoder and the image VAE, found next to the checkpoint in its models folder,
//! and the sampler widens the 4-channel placeholder to the model's one-frame latent. Set Steps to 6 and CFG Scale to 1
//! in SwarmUI: they are honoured as sent.
//!
//! With the model's class set to MiniMax H3 in SwarmUI instead, it sends H3's graph: `MiniMaxH3SigmaShift` (ignored,
//! with a warning), and the joint audio/video latent, whose video half must be one frame (Text2Video Frames 1). The
//! picture comes back with silent audio. Both H3 VAEs share the encoder, so either decodes it; pick
//! `minimax_h3_image_vae_fp16` as the VAE for the image decoder.

use std::path::{Path, PathBuf};

use swarm_engine_api::{EngineError, EngineResult, LatentFormat, LatentTensor, ModelFile, ModelFolder, Opaque, PixelInfo, TextEncoderHandle, VaeHandle};

const TEXT_ENCODER: &str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
const VAE: &str = "minimax_h3_image_vae_fp16.safetensors";

/// The models folder `checkpoint` sits in: the parent of its `Stable-diffusion` (or `checkpoints`, `diffusion_models`)
/// folder.
fn models_root(checkpoint: &Path) -> Option<PathBuf> {
    checkpoint.ancestors().skip(1).find_map(|d| {
        let name = d.file_name()?.to_str()?.to_ascii_lowercase();
        ["stable-diffusion", "checkpoints", "diffusion_models", "unet"].contains(&name.as_str()).then(|| d.parent().map(Path::to_path_buf)).flatten()
    })
}

/// `name` in any of `folders` under `root`, or one subfolder down.
fn find(root: &Path, folders: &[&str], name: &str) -> Option<PathBuf> {
    for f in folders {
        let dir = root.join(f);
        let direct = dir.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = e.path().join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// The text encoder and VAE `CheckpointLoaderSimple` returns for a MiniMax-H3-Image checkpoint.
pub fn companions(checkpoint: &ModelFile) -> EngineResult<(ModelFile, ModelFile)> {
    let missing = || EngineError::Unsupported(format!("MiniMax-H3-Image through SwarmUI needs {TEXT_ENCODER} in clip/ and {VAE} in VAE/ of the checkpoint's models folder"));
    let root = models_root(&checkpoint.path).ok_or_else(missing)?;
    let te = find(&root, &["clip", "text_encoders"], TEXT_ENCODER).ok_or_else(missing)?;
    let vae = find(&root, &["VAE", "vae"], VAE).ok_or_else(missing)?;
    let file = |folder, path: PathBuf| ModelFile { folder, name: path.file_name().unwrap().to_string_lossy().into_owned(), path };
    Ok((file(ModelFolder::TextEncoders, te), file(ModelFolder::Vae, vae)))
}

pub fn text_encoder(file: ModelFile, encoder_type: &str) -> TextEncoderHandle {
    TextEncoderHandle { sources: vec![file], encoder_type: encoder_type.into(), device: "default".into(), engine: Opaque::none(), patches: vec![] }
}

pub fn vae(file: ModelFile) -> VaeHandle {
    VaeHandle::new(Some(file), Opaque::none())
}

/// With the model set to SwarmUI's MiniMax H3 class: the video half of its joint latent, which must be one frame.
pub fn still(video: &LatentTensor) -> EngineResult<()> {
    match video.shape.as_slice() {
        [1, 24, 1, _, _] => Ok(()),
        _ => Err(EngineError::Unsupported("MiniMax-H3-Image makes one image: set Text2Video Frames to 1".into())),
    }
}

/// SwarmUI's empty `[1, 4, H/8, W/8]` placeholder as the model's `[1, 24, 1, H/16, W/16]`, with its pixel size.
pub fn widen(t: &LatentTensor, empty: bool) -> EngineResult<Option<(LatentTensor, PixelInfo)>> {
    let &[b, 4, h8, w8] = t.shape.as_slice() else { return Ok(None) };
    if !empty {
        return Err(EngineError::Unsupported("init images on MiniMax-H3-Image".into()));
    }
    if b != 1 {
        return Err(EngineError::Unsupported("batches on MiniMax-H3-Image through SwarmUI (use Images instead of Batch Size)".into()));
    }
    let (lh, lw) = (h8 / 2, w8 / 2);
    let tensor = LatentTensor { format: LatentFormat::MiniMaxH3Video, shape: vec![1, 24, 1, lh, lw], data: Opaque::none() };
    Ok(Some((tensor, PixelInfo::image(lw as u32 * 16, lh as u32 * 16, 1))))
}
