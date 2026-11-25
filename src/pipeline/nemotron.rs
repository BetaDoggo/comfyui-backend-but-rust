//! Nemotron-Labs-Image: one file holds the generator, its tokenizer and its image decoder. There is no sampler, VAE
//! or text encoder: the image's tokens are drawn one at a time (see [`crate::nemotron`]).

use super::{FamilyState, Generator, Kind, SharedTe};
use crate::cuda::Stream;
use crate::{lora, nemotron, offload, st};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct NemotronRequest {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    /// The first image's seed; image `i` takes `seed + i`.
    pub seed: u64,
    /// Images sampled together, 1 to [`nemotron::MAX_BATCH`].
    pub images: usize,
    /// Softmax temperature of the tokens: 0.9 on the model card, 0 for the most likely token.
    pub temperature: f32,
}

#[derive(Default)]
pub(super) struct Nemotron {
    model: Option<(PathBuf, nemotron::Model)>,
}

fn no_loras(loras: &[lora::Lora]) -> Result<()> {
    if !loras.is_empty() {
        return Err(lora::Unsupported("LoRAs on Nemotron-Labs-Image".into()).into());
    }
    Ok(())
}

impl FamilyState for Nemotron {
    fn set_loras(&mut self, loras: &[lora::Lora], _st: &Stream) -> Result<()> {
        no_loras(loras)
    }

    fn set_text_encoder_loras(&mut self, _shared: Option<&mut SharedTe>, loras: &[lora::Lora], _st: &Stream) -> Result<()> {
        no_loras(loras)
    }

    fn clear_prompts(&mut self) {}

    fn set_int8_attention(&mut self, _on: bool) {}

    fn blocks(&mut self) -> Option<(&mut offload::WeightSet, &st::SafeTensors)> {
        None
    }

    /// The expert cache keeps its VRAM: refilling it from RAM costs more than the idle VRAM is worth.
    fn idle(&mut self) -> Result<()> {
        Ok(())
    }
}

impl Generator {
    /// Load Nemotron-Labs-Image unless `path` is loaded. Other families go first: its expert cache takes the VRAM that
    /// is free.
    pub fn ensure_nemotron(&mut self, path: &Path) -> Result<()> {
        self.enter(Kind::Nemotron)?;
        let super::Family::Nemotron(f) = &mut self.family else { unreachable!() };
        if f.model.as_ref().is_some_and(|(p, _)| p == path) {
            return Ok(());
        }
        f.model = None;
        crate::cuda::trim_pool()?;
        let t0 = Instant::now();
        let m = nemotron::Model::load(path, self.env.threads, self.env.reserve, &self.env.st)?;
        eprintln!("Nemotron-Labs-Image ready in {:.1}s", t0.elapsed().as_secs_f64());
        f.model = Some((path.to_path_buf(), m));
        Ok(())
    }

    /// The request's images, RGB8 [height][width][3] each. `progress(tokens, of)` hears each step of the slowest
    /// image and stops the run with [`super::Cancelled`] by returning false.
    pub fn generate_nemotron(&mut self, r: &NemotronRequest, progress: impl FnMut(usize, usize) -> bool) -> Result<Vec<Vec<u8>>> {
        self.env.check_size_of(r.width, r.height, 16)?;
        let super::Family::Nemotron(f) = &mut self.family else { anyhow::bail!("no Nemotron-Labs-Image model loaded") };
        let (_, m) = f.model.as_mut().context("no Nemotron-Labs-Image model loaded")?;
        let seeds: Vec<u64> = (0..r.images as u64).map(|i| r.seed.wrapping_add(i)).collect();
        let req = nemotron::Request { prompt: &r.prompt, width: r.width, height: r.height, seeds: &seeds, temperature: r.temperature };
        m.generate(&req, progress, &self.env.st)
    }
}
