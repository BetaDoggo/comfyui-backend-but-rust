use std::path::PathBuf;

use clap::Parser;

/// Command-line configuration. Defaults mirror ComfyUI (`127.0.0.1:8188`).
#[derive(Parser, Debug, Clone)]
#[command(name = "comfy-server", about = "ComfyUI-compatible API server for SwarmUI, backed by a native Rust engine")]
pub struct Config {
    #[arg(long, default_value = "127.0.0.1")]
    pub listen: String,

    #[arg(long, default_value_t = 8188)]
    pub port: u16,

    /// Model root(s), laid out like SwarmUI's `Models/` folder (or ComfyUI's `models/`).
    /// Point this at the same folder SwarmUI uses so model names line up. Repeatable.
    #[arg(long = "models-root", default_value = "data/models")]
    pub models_roots: Vec<PathBuf>,

    /// ComfyUI `extra_model_paths` YAML, such as the one SwarmUI writes for its self-started
    /// ComfyUI. Searched before `--models-root`. Repeatable.
    #[arg(long = "extra-model-paths-config")]
    pub extra_model_paths: Vec<PathBuf>,

    #[arg(long, default_value = "data/output")]
    pub output_dir: PathBuf,

    #[arg(long, default_value = "data/input")]
    pub input_dir: PathBuf,

    #[arg(long, default_value = "data/temp")]
    pub temp_dir: PathBuf,

    /// ffmpeg binary used for mp4/webm/mov/mp3/flac/ogg encoding and video input decoding.
    #[arg(long, default_value = "ffmpeg")]
    pub ffmpeg: String,

    /// Which engine runs generations: `mock`, or `native` (needs a build with `--features native`).
    #[arg(long, default_value = "mock")]
    pub engine: String,

    /// native engine: VRAM (MB) to leave untouched for the desktop and other applications.
    #[arg(long, default_value_t = 512)]
    pub reserve_vram: usize,

    /// native engine: largest side (pixels, at 1:1) of a generation.
    #[arg(long, default_value_t = 2048)]
    pub max_side: usize,

    /// native engine: send latent previews while sampling. Off by default.
    #[arg(long, default_value_t = false)]
    pub live_previews: bool,

    /// Mock engine only: artificial delay per sampling step, so progress/previews are visible.
    #[arg(long, default_value_t = 40)]
    pub mock_step_ms: u64,

    /// Mock engine only: don't claim the backlog nodes through `custom_node` (other model
    /// families, third-party packs), so `/object_info` lists only natively implemented nodes.
    #[arg(long, default_value_t = false)]
    pub mock_native_only: bool,

    /// How many finished prompts to keep in `/history`.
    #[arg(long, default_value_t = 1000)]
    pub max_history: usize,

    /// Also list the `todo_nodes.json` classes the engine doesn't claim in `/object_info`
    /// (with their schemas). SwarmUI then enables the matching features, so their workflows
    /// can be generated and inspected; running them still fails validation.
    #[arg(long, default_value_t = false)]
    pub advertise_todo_nodes: bool,
}

impl Config {
    /// Config for tests: everything under one temp directory, no mock delay.
    pub fn for_tests(root: &std::path::Path) -> Self {
        Self {
            listen: "127.0.0.1".into(),
            port: 0,
            models_roots: vec![root.join("models")],
            extra_model_paths: Vec::new(),
            output_dir: root.join("output"),
            input_dir: root.join("input"),
            temp_dir: root.join("temp"),
            ffmpeg: "ffmpeg".into(),
            engine: "mock".into(),
            reserve_vram: 512,
            max_side: 2048,
            live_previews: false,
            mock_step_ms: 0,
            mock_native_only: false,
            max_history: 100,
            advertise_todo_nodes: false,
        }
    }
}
