//! Model folder scanning and name resolution (the equivalent of ComfyUI's `folder_paths`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use swarm_engine_api::{ModelFile, ModelFolder};

const MODEL_EXTENSIONS: &[&str] = &["safetensors", "sft", "ckpt", "pt", "pth", "bin", "gguf", "engine", "onnx"];

/// Sub-folders searched under each model root, per ComfyUI folder key. Includes both
/// ComfyUI's names and SwarmUI's (`Stable-Diffusion`, `Lora`, `VAE`, `clip`, ...), matching
/// the `extra_model_paths` mapping SwarmUI writes for its self-started ComfyUI.
fn subfolders(folder: ModelFolder) -> &'static [&'static str] {
    match folder {
        ModelFolder::Checkpoints => &["checkpoints", "Stable-Diffusion"],
        // SwarmUI lists `Stable-Diffusion;tensorrt;diffusion_models;unet` as one model set and
        // sends diffusion-format models from any of them to `UNETLoader`.
        ModelFolder::DiffusionModels => &["diffusion_models", "unet", "Stable-Diffusion"],
        ModelFolder::TextEncoders => &["text_encoders", "clip", "CLIP"],
        ModelFolder::Vae => &["vae", "VAE"],
        ModelFolder::Loras => &["loras", "Lora", "LyCORIS"],
        ModelFolder::Embeddings => &["embeddings", "Embeddings"],
        ModelFolder::UpscaleModels => &["upscale_models", "ESRGAN", "RealESRGAN", "SwinIR", "upscale-models"],
        ModelFolder::LatentUpscaleModels => &["latent_upscale_models"],
        ModelFolder::ClipVision => &["clip_vision"],
        ModelFolder::ControlNet => &["controlnet", "ControlNet", "model_patches"],
        ModelFolder::Gligen => &["gligen"],
        ModelFolder::StyleModels => &["style_models"],
        // As SwarmUI configures its self-started ComfyUI: ControlNet folders hold model patches too.
        ModelFolder::ModelPatches => &["model_patches", "controlnet", "ControlNet"],
        ModelFolder::Yolov8 => &["yolov8"],
        ModelFolder::IpAdapter => &["ipadapter"],
        ModelFolder::TensorRt => &["tensorrt"],
    }
}

#[derive(Clone, Debug)]
pub struct ModelPaths {
    roots: Vec<PathBuf>,
}

impl ModelPaths {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    pub fn dirs(&self, folder: ModelFolder) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for root in &self.roots {
            for sub in subfolders(folder) {
                let p = root.join(sub);
                // Windows is case-insensitive, so "clip" and "CLIP" are the same folder.
                if p.is_dir() && !out.iter().any(|o: &PathBuf| same_dir(o, &p)) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Sorted, de-duplicated model names (relative paths with `/`), like `folder_paths.get_filename_list`.
    pub fn list(&self, folder: ModelFolder) -> Vec<String> {
        let mut names = BTreeSet::new();
        for dir in self.dirs(folder) {
            walk(&dir, &dir, &mut names);
        }
        names.into_iter().collect()
    }

    /// Resolves a workflow model name to a file. Accepts `/` or `\` separators.
    pub fn resolve(&self, folder: ModelFolder, name: &str) -> Option<ModelFile> {
        let normalized = name.replace('\\', "/");
        if normalized.split('/').any(|part| part == "..") {
            return None;
        }
        for dir in self.dirs(folder) {
            let path = dir.join(&normalized);
            if path.is_file() {
                return Some(ModelFile { folder, name: normalized, path });
            }
        }
        None
    }
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn walk(base: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(base, &path, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| MODEL_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        {
            if let Ok(rel) = path.strip_prefix(base) {
                out.insert(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}
