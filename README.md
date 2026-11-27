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
| SD 1.x and SDXL | Text to image from fp16 checkpoints, linears quantized to int8 at load | SDXL 0.19 s/step at CFG 7; SD 1.5 45 ms/step at 512x512 | 0.29 s/step; 50 ms/step |
| MiniMax H3 | Text to video with audio | 0.24 s/step, 5 frames at 512x512 | 1.59 s/step |
| HyperFlow H3 Flash | Text to video with audio, 6 steps | 0.49 s/step, 22 frames at 512x512 | needs custom nodes, not timed |
| MiniMax-H3-Image | Text to image, 6 steps | 0.35 s/step, 1.07 s/step at 2048x2048 | 0.68 s/step, 1.34 s/step at 2048x2048 |
| SenseNova U1.6 | Text to image and edits with reference pictures, in pixel space, with its 8-step LoRA | 0.14 s/step at CFG 1 | about 1.04 s/step |
| Pixal3D | Image to a vertex-coloured 3D mesh (GLB), cutting the object out first with BiRefNet | 40 s per mesh at resolution 1024, 110 s at 1536 | not timed |
| Idol (TripoSplat) | Image to a 3D gaussian splat (PLY), cutting the object out first with BiRefNet | 14.4 s per splat, 0.57 s/step at CFG 3 | 111 s/step headless here, not tuned |
| SeedVR2 3B | Restores a picture or clip, as SwarmUI's SeedVR stage | 0.79 s per 1024 restore, DiT 145 ms | DiT 235 ms |
| RTX Video Super Resolution | NVIDIA's VFX SDK upscaler, up to x4, as ComfyUI's `RTXVideoSuperResolution` node and as an upscale model; works in SwarmUI, byte-identical to the node | 7 to 10 ms per 512x768 frame to 2x, after a 0.6 to 0.9 s load | NVIDIA's own code |
| Nemotron-Labs-Image | Text to image, written as visual tokens by a 30B-A3B language model; CLI, and SwarmUI through its extension | 99 s per image, 21.8 ms/token; 2 together in 139 s | not in ComfyUI |

The image and video models take every ComfyUI sampler and scheduler and LoRAs, with exceptions listed in
[Models](docs/MODELS.md): HyperFlow H3 Flash samples with `euler` and `res_multistep` only, and it, MiniMax-H3-Image, SD 1.x and SDXL refuse LoRAs. Nemotron-Labs-Image isn't diffusion, so it has no sampler and
refuses LoRAs. Init
images, inpainting and refiners work on the image models. ControlNet and every other model family don't work yet. The
server accepts far more of SwarmUI's features than that, but they only reach a mock engine, so run this beside a normal
ComfyUI backend. [Status](docs/STATUS.md) lists what works and what's left, in priority order.

## Quick start

You need Windows x64, an NVIDIA GPU (the default build targets RTX 50-series, `sm_120`; set `CUDA_ARCH` for others),
[Rust](https://rustup.rs), the CUDA 13 toolkit (`CUDA_PATH`) and MSVC Build Tools 2022.

The build takes CUTLASS and FlashAttention-2 from a comfy-kitchen checkout, at the commit the kernels are tested
against. Run these inside this repository:

```bash
git clone https://github.com/Comfy-Org/comfy-kitchen ref/comfy-kitchen
```

```bash
git -C ref/comfy-kitchen checkout f8b7e97a1418d2e529bd5405fbf86312be198ae6
```

```bash
git -C ref/comfy-kitchen submodule update --init --recursive
```

The build warns if CUTLASS or FlashAttention is at another commit. To move to a newer comfy-kitchen, update the commits
at the top of `build.rs` and this step together.

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
| HyperFlow H3 Flash | `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors`, or the `fl2va` file, from the `_comfyui` folders of [its repository](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) | as MiniMax H3 | as MiniMax H3 |
| MiniMax-H3-Image | `minimax_h3_image_pruned_int8_convrot.safetensors` | as MiniMax H3 | `MiniMaxH3/minimax_h3_image_vae_fp16.safetensors` |
| SenseNova U1.6 | `sensenova_u1.6_8b_mot_int8_convrot.safetensors` from [Comfy-Org/SenseNova-U1](https://huggingface.co/Comfy-Org/SenseNova-U1); its 8-step LoRA goes in `Lora/` | in the checkpoint | none: it samples pixels |
| SD 1.x | `v1-5-pruned-emaonly-fp16.safetensors`, or another SD 1.x checkpoint | in the checkpoint | in the checkpoint, or any SD VAE |
| SDXL | `sd_xl_base_1.0.safetensors`, or another SDXL base checkpoint | in the checkpoint | in the checkpoint, or any SD VAE |

Pixal3D uses other folders: `diffusion_models/Pixal3D/pixal3d_int8_convrot.safetensors`, `clip_vision/dino_v3_L_naf_fp32.safetensors`,
and `VAE/Pixal3D/trellis_2_shape_vae_bf16.safetensors` with `trellis_2_texture_vae_bf16.safetensors`, all from
[Comfy-Org/Pixal3D](https://huggingface.co/Comfy-Org/Pixal3D). An image without a transparent background is cut out
first by BiRefNet, `background_removal/birefnet.safetensors` from [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet).
In SwarmUI it runs from its own tab, the [Pixal3D extension](swarm%20backend/pixal3d-extension/README.md).

SeedVR2, SwarmUI's restoration stage (SeedVR Model), uses `diffusion_models/SeedVR2/seedvr2_3b_int8_convrot.safetensors` and
`VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors`, from [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2).

RelayVSR, video upscaling x2 or x4, uses `diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors` or
`relayvsr_x4_int8_convrot.safetensors`. Its decoder is `VAE/relayvsr_flash_vae_x2_fp32.safetensors` or
`relayvsr_flash_vae_x4_fp32.safetensors`. All four are from [Comfy-Org/RelayVSR](https://huggingface.co/Comfy-Org/RelayVSR).
In SwarmUI it is a stage added by the [RelayVSR extension](swarm%20backend/relayvsr-extension/README.md). It works in
SwarmUI, and it has been compared with ComfyUI through the CLI.

RTX Video Super Resolution uses NVIDIA's `nvngx_vsr.dll` in an `upscale_models` folder, such as
`upscale_models/nvidia-vfx/`, with `cudart64_12.dll`, `NVCVImage.dll`, `nvngxruntime.dll`, `NVVideoEffects.dll` and
`nvVFXVideoSuperRes.dll` beside it. All six are in the `nvvfx/libs` folder of NVIDIA's `nvidia-vfx` wheel
(`pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com`), under NVIDIA's license. In SwarmUI it shows as an
upscale model, and runs the `RTXVideoSuperResolution` node the RTXVSRExtension adds; see
[Models](docs/MODELS.md#rtx-video-super-resolution).

Idol uses `diffusion_models/idol_fp16.safetensors`, `clip_vision/dino_v3_vit_h_bf16.safetensors`, and
`VAE/idol_decoder_fp16.safetensors` with the Flux 2 VAE (`flux2-vae.safetensors`, anywhere in `VAE/`), all from
[Comfy-Org/Idol](https://huggingface.co/Comfy-Org/Idol). It runs from the same tab, as its second model.

Nemotron-Labs-Image uses `diffusion_models/Nemotron-Labs-Image-30B-A3B-nvfp4.safetensors`, one 24 GB file holding the
generator with nvfp4 experts, its tokenizer and its image decoder. `tools/nemotron_image_convert.py` writes it from
[nvidia/Nemotron-Labs-Image-30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B), one shard at a time,
and a converted copy is at [HDiffusion/Nemotron-Labs-Image-30B-A3B-NVFP4](https://huggingface.co/HDiffusion/Nemotron-Labs-Image-30B-A3B-NVFP4).
Both are under NVIDIA's non-commercial license. It needs about 17 GB of free RAM for the pinned experts. In SwarmUI it
runs through the [Nemotron extension](swarm%20backend/nemotron-extension/README.md), and works there.

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

```bash
target/release/diffusion.exe --sdxl --scheduler normal -p "no humans, scenery, a lighthouse on a rocky coast at sunset" -o lighthouse_xl.png
```

Add `--lora path/to/lora.safetensors:0.8` to any of them but SD 1.x and SDXL. H3 has its own [example](docs/MODELS.md#minimax-h3).

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
* [Activation steering](docs/STEERING.md): Anima's training-free style and concept steering, experimental
* [Models](docs/MODELS.md): each family's timings, closeness to ComfyUI and limits
* [SwarmUI backend](swarm%20backend/README.md): running the server and what SwarmUI can use
* [Status](docs/STATUS.md): everything that works and everything left to do

For contributors:

* [AGENTS.md](AGENTS.md): working rules for people and coding agents
* [Technical overview](docs/TECHNICAL_OVERVIEW.md): how the pieces fit and what they're built on
* [Testing](docs/TESTING.md): setting up ComfyUI and SwarmUI to check the engine against them
* [Adding models](swarm%20backend/docs/ADDING_MODELS.md): how to add a model family
* [Input mapping](swarm%20backend/docs/INPUT_MAPPING.md): what SwarmUI sends and what the engine receives
* [Compatibility](swarm%20backend/docs/COMPATIBILITY.md): what the server accepts for every SwarmUI parameter

## License

Licensed under the [GNU GPL v3](LICENSE), like ComfyUI, which this code was written from. Files under `csrc/vendor/` stay
Apache-2.0, see `csrc/vendor/LICENSE.comfy-kitchen` and its attributions in `csrc/vendor/NOTICE.comfy-kitchen`. The
tokenizer files under `assets/` keep their upstream licenses, listed in `assets/README.md`.
