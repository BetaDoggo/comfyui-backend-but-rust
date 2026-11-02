//! diffusion: Rust + CUDA reimplementation of ComfyUI's image generation, for quantized checkpoints of Krea 2, Anima, MOSS Image and MiniMax H3.

pub mod anima;
pub mod cuda;
pub mod dit;
pub mod ffi;
pub mod flux2vae;
pub mod h3;
pub mod h3te;
pub mod h3vae;
pub mod lora;
mod lora_adapters;
pub mod moss;
pub mod offload;
pub mod pipeline;
pub mod qwen3;
pub mod qwen35;
pub mod samplers;
pub mod sampling;
pub mod serve;
pub mod st;
pub mod te;
pub mod tokenizer;
pub mod vae;
pub mod vision;
