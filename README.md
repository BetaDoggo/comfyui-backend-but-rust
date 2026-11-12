# ComfyUI backend, but Rust

> [!CAUTION]
> # VIBECODED WIP. DO NOT USE THIS.
>
> **Unless you hate yourself, use ComfyUI.** This entire project was vibecoded by an AI. It is a work in progress that
> runs on one person's machine, has no stability guarantees, and hasn't been reviewed by anyone who could tell you
> whether it's a good idea. It drives your GPU through hand-written CUDA kernels. Expect crashes, refused features,
> and images that differ from ComfyUI's. Nobody will support you.

A Rust and CUDA backend that speaks ComfyUI's API and replaces the ComfyUI features SwarmUI uses most, so SwarmUI can
generate without Python or ComfyUI. The goal is speed first.

It really generates with these image, video and image-to-3D models, from specific checkpoints only:

| Family | Generates | 1024x1024 sampling here | ComfyUI |
|---|---|---|---|
| Krea 2 Turbo | Text to image, with Prompt Images and reference latents | 0.58 to 0.61 s/step | about 1.00 s/step |
| Anima | Text to image | 115 ms/step | 199 ms/step |
| MOSS Image | Text to image | 550 to 573 ms/step at CFG 4 | 601 ms/step |
| Qwen Image 2.1 | Text to image with alpha, and edits with reference images | 0.36 s/step, 0.22 to 0.44 s/step with references | 0.46 s/step |
| MiniMax H3 | Text to video with audio | 0.24 s/step, 5 frames at 512x512 | 1.59 s/step |
| HyperFlow H3 Flash | Text to video with audio, 6 steps | 0.49 s/step, 22 frames at 512x512 | needs custom nodes, not timed |
| MiniMax-H3-Image | Text to image, 6 steps | 0.35 s/step, 1.07 s/step at 2048x2048 | 0.68 s/step, 1.34 s/step at 2048x2048 |
| Pixal3D | Image to a vertex-coloured 3D mesh (GLB), cutting the object out first with BiRefNet | 40 s per mesh at resolution 1024, 110 s at 1536 | not timed |
| Idol (TripoSplat) | Image to a 3D gaussian splat (PLY), cutting the object out first with BiRefNet | 14.4 s per splat, 0.57 s/step at CFG 3 | 111 s/step headless here, not tuned |
| SeedVR2 3B | Restores a picture, as SwarmUI's SeedVR stage | 0.79 s per 1024 restore, DiT 145 ms | DiT 235 ms |

The image and video models take every ComfyUI sampler and scheduler and LoRAs, with exceptions listed in
[Models](docs/MODELS.md): HyperFlow H3 Flash samples with `euler` and `res_multistep` only, and it and MiniMax-H3-Image refuse LoRAs. Init
images, inpainting and refiners work on the image models. ControlNet and every other model family don't work yet. The
server accepts far more of SwarmUI's features than that, but they only reach a mock engine, so run this beside a normal
ComfyUI backend. [Status](docs/STATUS.md) lists what works and what's left, in priority order.

## Quick start

You need Windows x64, an NVIDIA GPU (the default build targets RTX 50-series, `sm_120`; set `CUDA_ARCH` for others),
[Rust](https://rustup.rs), the CUDA 13 toolkit (`CUDA_PATH`) and MSVC Build Tools 2022.

The build takes CUTLASS and FlashAttention-2 from a comfy-kitchen checkout. Run this inside this repository:

```bash
git clone --recursive https://github.com/Comfy-Org/comfy-kitchen ref/comfy-kitchen
```

Then build, which takes several minutes the first time:

```bash
cargo build --release
```

The toolkit's `bin` folder must be on `PATH` at run time, because the programs load `cublasLt64_13.dll` from it.

### Models

Put your models in one folder, keeping these subfolders, and point `MODELS_ROOT` at it (or pass `--models`, or `--dit`,
`--te` and `--vae` for any other file). Files are read in place, and nothing is copied or converted.

| Model | `Stable-diffusion/` | `clip/` | `VAE/` |
|---|---|---|---|
| Krea 2 Turbo | `kroma-v0.3.1-turbo-opd-W8A8.safetensors` | `qwen3vl_4b.safetensors` | `qwen_image_vae.safetensors` |
| Anima | `animaInt8Mxfp8_turboV11Int8.safetensors` | `qwen_3_600m.safetensors` | `qwen_image_vae.safetensors` |
| MOSS Image | `moss_image_sft_int8_convrot_attn.safetensors` | `qwen3.5_9b_int8_convrot.safetensors` | `flux2-vae.safetensors` |
| Qwen Image 2.1 | `rayQwen21_v1Int8Convrot.safetensors` | `qwen3vl_8b_int8_convrot.safetensors` | `QwenImage/qwen_image_2.1_vae_bf16.safetensors` |
| MiniMax H3 | `minimax_h3_fl2va_pruned_w6a8.safetensors` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors`, `MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors` |
| HyperFlow H3 Flash | `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors` from `ref2va_int8_convrot_comfyui/` in [its repository](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash), or a folder holding `fl2va_int8_convrot/` (pick any shard) | as MiniMax H3 | as MiniMax H3 |
| MiniMax-H3-Image | `minimax_h3_image_pruned_int8_convrot.safetensors` | as MiniMax H3 | `MiniMaxH3/minimax_h3_image_vae_fp16.safetensors` |

Pixal3D uses other folders: `diffusion_models/Pixal3D/pixal3d_int8_convrot.safetensors`, `clip_vision/dino_v3_L_naf_fp32.safetensors`,
and `VAE/Pixal3D/trellis_2_shape_vae_bf16.safetensors` with `trellis_2_texture_vae_bf16.safetensors`, all from
[Comfy-Org/Pixal3D](https://huggingface.co/Comfy-Org/Pixal3D). An image without a transparent background is cut out
first by BiRefNet, `background_removal/birefnet.safetensors` from [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet).
In SwarmUI it runs from its own tab, the [Pixal3D extension](swarm%20backend/pixal3d-extension/README.md).

SeedVR2, SwarmUI's restoration stage (SeedVR Model), uses `diffusion_models/SeedVR2/seedvr2_3b_int8_convrot.safetensors` and
`VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors`, from [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2).

Idol uses `diffusion_models/idol_fp16.safetensors`, `clip_vision/dino_v3_vit_h_bf16.safetensors`, and
`VAE/idol_decoder_fp16.safetensors` with the Flux 2 VAE (`flux2-vae.safetensors`, anywhere in `VAE/`), all from
[Comfy-Org/Idol](https://huggingface.co/Comfy-Org/Idol). It runs from the same tab, as its second model.

### Generate from the command line

```bash
target/release/diffusion.exe -p "a red fox in a snowy forest" --seed 42 -o fox.png
```

```bash
target/release/diffusion.exe --anima -p "no humans, scenery, mountain lake at sunrise" -o lake.png
```

```bash
target/release/diffusion.exe --moss -p "a stone lighthouse on a sea cliff at dawn" --steps 20 --cfg 4 -o lighthouse.png
```

Add `--lora path/to/lora.safetensors:0.8` to any of them. H3 has its own [example](docs/MODELS.md#minimax-h3).

### Generate from SwarmUI

Set `MODELS_ROOT`, run [`swarm backend/start-backend.bat`](swarm%20backend/start-backend.bat), and add
`http://127.0.0.1:8188` as a "ComfyUI API By URL" backend. Keep a real ComfyUI backend next to it, because anything else
is refused or reaches a mock engine. See the [SwarmUI backend guide](swarm%20backend/README.md).

To have SwarmUI start and stop the server itself, use the comfy-rust backend extension instead. It uses SwarmUI's own model
folders, so `MODELS_ROOT` isn't needed.

1. Build the server from `swarm backend/`, with the CUDA toolkit's `bin` folder on `PATH`. SwarmUI doesn't build it, so
   repeat this after pulling changes:

   ```bash
   cargo build --release -p comfy-server --features native
   ```

2. Stop SwarmUI, then link the extension into it. Run this in `cmd` from the SwarmUI folder, with your own repository path:

   ```bat
   mklink /J src\Extensions\ComfyRustBackend "C:\path\to\repo\swarm backend\swarm-extension"
   ```

3. Start SwarmUI, which builds the extension.
4. In Server, Backends, add **comfy-rust Self-Starting** and set Start Script to the full path of
   [`swarm backend/selfstart.bat`](swarm%20backend/selfstart.bat). For MiniMax H3, add `--ffmpeg <path>` to ExtraArgs if
   ffmpeg isn't on `PATH`.

## Documentation

For users:

* [Usage](docs/USAGE.md): command line flags, LoRAs and samplers
* [Models](docs/MODELS.md): each family's timings, closeness to ComfyUI and limits
* [SwarmUI backend](swarm%20backend/README.md): running the server and what SwarmUI can use
* [Status](docs/STATUS.md): everything that works and everything left to do

For contributors:

* [AGENTS.md](AGENTS.md): working rules for people and coding agents
* [Technical overview](docs/TECHNICAL_OVERVIEW.md): how the pieces fit and what they're built on
* [Adding models](swarm%20backend/docs/ADDING_MODELS.md): how to add a model family
* [Input mapping](swarm%20backend/docs/INPUT_MAPPING.md): what SwarmUI sends and what the engine receives
* [Compatibility](swarm%20backend/docs/COMPATIBILITY.md): what the server accepts for every SwarmUI parameter

## License

Licensed under the [GNU GPL v3](LICENSE), like ComfyUI, which this code was written from. Files under `csrc/vendor/` stay
Apache-2.0, see `csrc/vendor/LICENSE.comfy-kitchen` and its attributions in `csrc/vendor/NOTICE.comfy-kitchen`. The
tokenizer files under `assets/` keep their upstream licenses, listed in `assets/README.md`.
