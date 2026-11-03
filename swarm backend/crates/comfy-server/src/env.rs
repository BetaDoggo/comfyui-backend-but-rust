use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::models::ModelPaths;

/// Filesystem / tool configuration shared by nodes and HTTP routes.
#[derive(Clone, Debug)]
pub struct ServerEnv {
    pub models: ModelPaths,
    pub output_dir: PathBuf,
    pub input_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub ffmpeg: String,
    /// See `Config::advertise_todo_nodes`.
    pub advertise_todo_nodes: bool,
    /// From `EngineInfo::attention_backends` (filled in by `build`).
    pub attention_backends: Vec<String>,
}

impl ServerEnv {
    pub fn from_config(cfg: &Config) -> std::io::Result<Self> {
        for d in [&cfg.output_dir, &cfg.input_dir, &cfg.temp_dir] {
            std::fs::create_dir_all(d)?;
        }
        Ok(Self {
            models: ModelPaths::with_configs(cfg.models_roots.clone(), &cfg.extra_model_paths)?,
            output_dir: cfg.output_dir.clone(),
            input_dir: cfg.input_dir.clone(),
            temp_dir: cfg.temp_dir.clone(),
            ffmpeg: cfg.ffmpeg.clone(),
            advertise_todo_nodes: cfg.advertise_todo_nodes,
            attention_backends: Vec::new(),
        })
    }

    /// The directory for a `/view` / upload `type` (`output`, `input`, `temp`).
    pub fn dir_for_type(&self, ty: &str) -> Option<&Path> {
        match ty {
            "output" => Some(&self.output_dir),
            "input" => Some(&self.input_dir),
            "temp" => Some(&self.temp_dir),
            _ => None,
        }
    }

    /// Joins `dir/subfolder/filename`, refusing anything that escapes `dir`.
    pub fn safe_join(dir: &Path, subfolder: &str, filename: &str) -> Option<PathBuf> {
        let mut p = dir.to_path_buf();
        for part in subfolder.split(['/', '\\']).chain(filename.split(['/', '\\'])) {
            if part.is_empty() || part == "." {
                continue;
            }
            if part == ".." || part.contains(':') {
                return None;
            }
            p.push(part);
        }
        if p == dir {
            return None;
        }
        Some(p)
    }
}
