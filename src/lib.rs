//! diffusion: Rust + CUDA reimplementation of ComfyUI's image generation, for quantized checkpoints of Krea 2, Anima, MOSS Image and MiniMax H3,
//! and Pixal3D image to 3D.

pub mod anima;
pub mod birefnet;
pub mod comfy_memory;
pub mod cuda;
pub mod dino3;
pub mod dit;
pub mod ffi;
pub mod flux2vae;
pub mod h3;
pub mod h3te;
pub mod h3vae;
pub mod latent_ops;
pub mod lora;
mod lora_adapters;
pub mod moss;
pub mod offload;
pub mod pipeline;
pub mod pixal3d;
pub mod preview;
pub mod prof;
pub mod qwen21;
pub mod qwen3;
pub mod qwen35;
pub mod samplers;
pub mod sampling;
pub mod st;
pub mod te;
pub mod tokenizer;
pub mod trellis;
pub mod trellis_vae;
pub mod vae;
pub mod vae21;
pub mod vision;
