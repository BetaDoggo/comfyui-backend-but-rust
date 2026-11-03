//! Model folder scanning and name resolution (the equivalent of ComfyUI's `folder_paths`).

use std::collections::{BTreeMap, BTreeSet};
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

/// The folders a key of a ComfyUI `extra_model_paths` file fills. Keys the server has no use for
/// (`hypernetworks`, `custom_nodes`, ...) fill none.
fn config_folders(key: &str) -> &'static [ModelFolder] {
    match key {
        // As in `subfolders`: SwarmUI sends diffusion-format models from its model folders to `UNETLoader`.
        "checkpoints" => &[ModelFolder::Checkpoints, ModelFolder::DiffusionModels],
        "diffusion_models" | "unet" => &[ModelFolder::DiffusionModels],
        "text_encoders" | "clip" => &[ModelFolder::TextEncoders],
        "vae" => &[ModelFolder::Vae],
        "loras" => &[ModelFolder::Loras],
        "embeddings" => &[ModelFolder::Embeddings],
        "upscale_models" => &[ModelFolder::UpscaleModels],
        "latent_upscale_models" => &[ModelFolder::LatentUpscaleModels],
        "clip_vision" => &[ModelFolder::ClipVision],
        "controlnet" => &[ModelFolder::ControlNet],
        "gligen" => &[ModelFolder::Gligen],
        "style_models" => &[ModelFolder::StyleModels],
        "model_patches" => &[ModelFolder::ModelPatches],
        "yolov8" => &[ModelFolder::Yolov8],
        "ipadapter" => &[ModelFolder::IpAdapter],
        "tensorrt" => &[ModelFolder::TensorRt],
        _ => &[],
    }
}

/// The folders of one `extra_model_paths` file, read the way ComfyUI's `load_extra_path_config` does.
fn read_config(path: &Path) -> std::io::Result<Vec<(ModelFolder, PathBuf)>> {
    let bad = |e: serde_yaml::Error| std::io::Error::other(format!("{}: {e}", path.display()));
    let sections: BTreeMap<String, BTreeMap<String, serde_yaml::Value>> = serde_yaml::from_str(&std::fs::read_to_string(path)?).map_err(bad)?;
    let here = path.parent().unwrap_or(Path::new("")).to_path_buf();
    let mut out = Vec::new();
    for section in sections.values() {
        let base = section.get("base_path").and_then(|v| v.as_str()).map_or(here.clone(), |b| here.join(b));
        for (key, value) in section {
            for line in value.as_str().unwrap_or("").lines().map(str::trim).filter(|l| !l.is_empty()) {
                out.extend(config_folders(key).iter().map(|f| (*f, base.join(line))));
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct ModelPaths {
    roots: Vec<PathBuf>,
    /// From `extra_model_paths` files, searched first.
    configured: Vec<(ModelFolder, PathBuf)>,
}

impl ModelPaths {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots, configured: Vec::new() }
    }

    pub fn with_configs(roots: Vec<PathBuf>, configs: &[PathBuf]) -> std::io::Result<Self> {
        let mut configured = Vec::new();
        for path in configs {
            configured.extend(read_config(path)?);
        }
        Ok(Self { roots, configured })
    }

    pub fn dirs(&self, folder: ModelFolder) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for (_, p) in self.configured.iter().filter(|(f, _)| *f == folder) {
            if p.is_dir() && !out.iter().any(|o: &PathBuf| same_dir(o, p)) {
                out.push(p.clone());
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_swarm_model_paths_file() {
        let root = std::env::temp_dir().join(format!("model-paths-{}", uuid::Uuid::new_v4()));
        for (dir, file) in [("one/Custom SD", "a.safetensors"), ("one/clip", "te.safetensors"), ("two/Custom SD", "b.safetensors")] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            std::fs::write(root.join(dir).join(file), b"").unwrap();
        }
        let (one, two) = (root.join("one"), root.join("two"));
        // Laid out as ComfyUISelfStartBackend.EnsureComfyFile writes it, Windows paths included.
        let yaml = format!(
            "swarmui:\n    base_path: {}\n    is_default: true\n    checkpoints: |\n       Custom SD\n       missing\n    clip: |\n       clip\n    hypernetworks: |\n       hypernetworks\n\nswarmui2:\n    base_path: {}\n    \n    checkpoints: |\n       Custom SD\n    vae: \n\nswarmui_nodes:\n    custom_nodes: |\n       G:/nodes\n",
            one.display(),
            two.display()
        );
        let file = root.join("paths.yaml");
        std::fs::write(&file, yaml).unwrap();

        let paths = ModelPaths::with_configs(Vec::new(), &[file]).unwrap();
        assert_eq!(paths.list(ModelFolder::Checkpoints), ["a.safetensors", "b.safetensors"]);
        assert_eq!(paths.list(ModelFolder::DiffusionModels), ["a.safetensors", "b.safetensors"]);
        assert_eq!(paths.list(ModelFolder::TextEncoders), ["te.safetensors"]);
        assert!(paths.list(ModelFolder::Vae).is_empty());
        assert_eq!(paths.resolve(ModelFolder::Checkpoints, "b.safetensors").unwrap().path, two.join("Custom SD/b.safetensors"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
