# ComfyUI backend, but Rust

> [!CAUTION]
> # VIBECODED WIP. DO NOT USE THIS.
>
> **Unless you hate yourself, use ComfyUI.** This entire project was vibecoded by an AI. It is a work in progress that
> runs on one person's machine, has no stability guarantees, and hasn't been reviewed by anyone who could tell you
> whether it's a good idea. It drives your GPU through hand-written CUDA kernels. Expect crashes, refused features,
> and images that differ from ComfyUI's. Nobody will support you.

A Rust and CUDA backend that speaks ComfyUI's API, so SwarmUI can generate without Python or ComfyUI. It only runs three
int8 model families, text-to-image only.

## Quick start

You need Windows x64, an NVIDIA GPU (the default build targets RTX 50-series, `sm_120`; set `KREA_CUDA_ARCH` for others),
[Rust](https://rustup.rs), the CUDA 13 toolkit (`CUDA_PATH`) and MSVC Build Tools 2022.

```bash
git clone --recursive https://github.com/Comfy-Org/comfy-kitchen ref/comfy-kitchen
```

Run that inside this repository: the build takes CUTLASS and FlashAttention-2 from that checkout. Then build, which takes
about 6 minutes the first time:

```bash
cargo build --release
```

Put your models in one folder, keeping these subfolders, and point `KREA2_MODELS` at it (or pass `--models`, or `--dit`,
`--te` and `--vae` for any other file):

| Model | `Stable-diffusion/` | `clip/` | `VAE/` |
|---|---|---|---|
| Krea 2 Turbo | `kroma-v0.3.1-turbo-opd-W8A8.safetensors` | `qwen3vl_4b.safetensors` | `qwen_image_vae.safetensors` |
| Anima | `animaInt8Mxfp8_turboV11Int8.safetensors` | `qwen_3_600m.safetensors` | `qwen_image_vae.safetensors` |
| MOSS Image | `moss_image_sft_int8_convrot_attn.safetensors` | `qwen3.5_9b_int8_convrot.safetensors` | `flux2-vae.safetensors` |

Generate an image from the command line:

```bash
target/release/krea2.exe -p "a red fox in a snowy forest" --seed 42 -o fox.png
```

```bash
target/release/krea2.exe --anima -p "no humans, scenery, mountain lake at sunrise" -o lake.png
```

```bash
target/release/krea2.exe --moss -p "a stone lighthouse on a sea cliff at dawn" --steps 20 --cfg 4 -o lighthouse.png
```

Add `--lora path/to/lora.safetensors:0.8` to any of them. To use it from SwarmUI, set `KREA2_MODELS`, run
[`swarm backend/start-krea2.bat`](swarm%20backend/start-krea2.bat), and add `http://127.0.0.1:8188` as a "ComfyUI API By URL"
backend. Keep a real ComfyUI backend next to it, because anything else is refused or reaches a mock engine.

More: [technical overview](docs/TECHNICAL_OVERVIEW.md), [status](docs/STATUS.md), [SwarmUI backend](swarm%20backend/README.md).

This repository has no license yet. Files under `csrc/vendor/` are Apache-2.0, see `csrc/vendor/LICENSE.comfy-kitchen`.

## What this is

A Rust and CUDA backend that speaks ComfyUI's API and replaces the ComfyUI features SwarmUI uses most, so
SwarmUI can generate without Python or ComfyUI. The goal is speed first: Krea 2 Turbo takes about 4.65 s per
1024x1024 image here, against about 9 s in ComfyUI on the same machine, and Anima Turbo samples 8 steps in
0.92 s against 1.59 s. MOSS Image gains least, about 5 to 8% per step, because ComfyUI already runs it on the same kernels.

**Status:** three model families really generate, text-to-image only, with LoRAs: Krea 2 Turbo, Anima and MOSS Image. The server accepts far more of
SwarmUI's features than that, but they only reach a mock engine so far. [`docs/STATUS.md`](docs/STATUS.md) lists everything that works and everything
still to do, in priority order. Run it today beside a normal ComfyUI backend.

* `swarm backend/` is the server: the ComfyUI protocol, the node graph, and the `Engine` interface. Start it with
  [`start-krea2.bat`](swarm%20backend/start-krea2.bat) and see [its README](swarm%20backend/README.md).
* The repository root is the real engine: the Krea 2, Anima and MOSS Image pipelines described below. They also run alone
  from the command line, and Krea 2 has a small web UI.
* [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md) explains what the project is built on and how the pieces fit.

## LoRAs

All three models take LoRAs, from SwarmUI or with `--lora path[:strength]` (repeatable, applied in order):

```bash
target/release/krea2.exe --moss --lora moss-dmdv1.safetensors --steps 8 --cfg 1 -p "a stone lighthouse on a sea cliff at dawn, no people" -o lighthouse.png
```

They are baked, as ComfyUI does for quantized weights: each patched int8 weight is dequantized to bf16, the patches are
added in bf16, and it is requantized with new row scales and stochastic rounding seeded by the module's name. Sampling
then runs the same int8 kernels, so a LoRA costs no time per step. Only tensors whose patches changed are rewritten, and
removing a LoRA puts the file's bytes back. Prompts are encoded again after a change, because their contexts come out of
the diffusion model.

Compared with ComfyUI under DynamicVRAM, which SwarmUI's `main.py` uses, on the same LoRAs at 1024x1024:

| | Krea 2 (2 LoRAs, 12 GB model) | Anima (Turbo LoRA) | MOSS Image (DMD LoRA) |
|---|---|---|---|
| Bake, files cached | 2.7 to 3.5 s | 0.4 to 0.7 s | 1.2 to 1.7 s |
| ComfyUI's first run minus a later run (includes loading the model) | about 27 s | about 2.4 s | about 12 s |
| Sampling with the LoRA | 615 ms/step | 128 ms/step | 289 ms/step |
| ComfyUI, later runs | 1102 ms/step | 223 ms/step | 309 ms/step |
| Baked weights | identical | identical | identical |
| Image PSNR with LoRA / same settings without | 31.6 dB / not measured | 21.2 / 25.6 dB | 22.3 / 40.3 dB |

"Identical" means every value of the bf16 weight, the row scales and the int8 weight. That was checked on three Krea 2
modules, four Anima modules, and all 168 MOSS Image weights by fingerprint. What still differs is what already differed
without a LoRA. MOSS Image's error in one model step is the same either way (3.2% and 2.8%), and the DMD LoRA's short
path magnifies it, as Turbo trajectories do on Krea 2. Check this with `tools/lora_ref.py` and `tools/lora_cmp.py`.

Like ComfyUI, keys that match nothing in the model are logged and ignored, so a LoRA for another model changes nothing.
Plain LoRA, `diff` and `set` patches work. These are refused with an error: LoHa, LoKr, GLoRA, OFT, DoRA, LoCon mid
weights, reshaped patches, MOSS Image control LoRAs, LoRA hooks, and LoRAs with text encoder keys.

## MOSS Image engine

Runs int8 ConvRot MOSS Image checkpoints with fused q/k/v, the Qwen3.5 9B text encoder and the Flux 2 VAE. Both
published key layouts of that VAE load. `euler` (SwarmUI's default) and `er_sde` work with the `simple` scheduler, at any CFG.
Prompts are literal: like ComfyUI, this encoder has no `(word:1.2)` emphasis.

```bash
target/release/krea2.exe --moss -p "a stone lighthouse on a sea cliff at dawn, no people" --steps 20 --cfg 4 --negative "blurry" -o lighthouse.png
```

Measured at 1024x1024, 20 steps, CFG 4 (two model evaluations per step), after warm-up:

| | ComfyUI | krea2 (`--attention flash`) | krea2 (default, `int8`) |
|---|---|---|---|
| Sampling | 12.0 s (601 ms/step) | 15.2 s (760 ms/step) | 11.0 to 11.5 s (550 to 573 ms/step) |
| Text, one prompt | 437 ms | 248 ms | 248 ms |
| VAE decode | 530 to 780 ms | 268 ms | 268 ms |

Where the time goes in one evaluation (about 275 ms): the int8 GEMMs take 173 ms and attention 58 ms. Those are the
same kernels ComfyUI uses, so sampling gains little. The text encoder is streamed from pinned RAM here and timed with
ComfyUI's held in VRAM.

The prompt's text embedder runs once, and the modulation is cached per sigma across images.

The output is not pixel-identical to ComfyUI. Token ids match, Qwen3.5 hidden states match to 3e-6 (the GatedDeltaNet
layers run as the plain delta-rule recursion on the GPU), and the image embedder matches exactly. The text embedder
differs by 0.5%, because torch's bf16 GEMM isn't exact at those sizes. The same seed gives the same picture at about
34 dB PSNR with int8 attention and 33 dB with flash.

`--swarm-template` encodes with the Krea 2 template, which SwarmUI sends for this model. Test prompts here must not
generate people.

## Anima engine

Runs int8 ConvRot Anima checkpoints with the Qwen3 0.6B text encoder and the Qwen-Image VAE. Files are read in
place, as with Krea 2. `er_sde` (SwarmUI's default) and `euler` work with the `simple` scheduler, at any CFG, with
negative prompts and `(word:1.2)` emphasis.

```bash
target/release/krea2.exe --anima -p "no humans, scenery, mountain lake at sunrise" --seed 42 -o lake.png
```

Measured at 1024x1024 with 8 steps, CFG 1 and `er_sde`, after warm-up:

| | ComfyUI | krea2 (`--attention flash`) | krea2 (default, `int8`) |
|---|---|---|---|
| Sampling | 1.59 s (199 ms/step) | 1.31 s (164 ms/step) | 0.92 s (115 ms/step) |

A whole image takes about 1.04 s from the command line and 1.14 s through SwarmUI, decode included.

Where it is faster than ComfyUI:

* Each block's cross-attention k and v depend only on the prompt, so they're computed once per prompt.
* Block modulation depends only on sigma, so it's cached per sigma across images.
* `silu(t_emb)` is quantized once for all 84 modulation networks, and the all-zero padding-mask channel is
  dropped from the patch embedding.

The output is not pixel-identical to ComfyUI. Tokens, Qwen3 hidden states (error 1e-6), the sigma schedule and
the sampler's noise match, including torch's CUDA Philox layout, which depends on the GPU's core count. The LLM
adapter output differs by about 1.7%, cause not yet found, so the same seed gives the same picture at
26 dB PSNR with flash attention and 25 dB with int8.

Test prompts here should be scenery with no people: this model drifts towards explicit images even on neutral prompts.

## Krea 2 engine

A standalone reimplementation of ComfyUI's **Krea-2 Turbo** text-to-image workflow
(`blueprints/Text to Image (Krea-2 Turbo).json`: 1024², 8 steps, euler / simple, CFG 1, shift 1.15)
for the int8 checkpoint `kroma-v0.3.1-turbo-opd-W8A8`, the Qwen3-VL-4B text encoder and the
Qwen-Image (Wan 2.1) VAE. It loads the original `.safetensors` files read-only via mmap; nothing
is copied or converted on disk.

```bash
cargo build --release
```

```bash
target/release/krea2.exe -p "a red fox sitting in a snowy forest, golden hour, photograph" --seed 42 -o fox.png
```

### Web UI

```bash
target/release/krea2.exe --serve 127.0.0.1:7860
```

Then open http://127.0.0.1:7860. Models load once and stay in memory; requests queue on the GPU.
The page has prompt, a side-length slider with aspect-ratio buttons (pixel count ≈ side², multiples of 16), steps, image count, seed (random or fixed), live step
progress and a history strip. Images are also saved to `--out-dir` (default `outputs/`).
The side length goes up to 2048 (`--max-side`, default 2048); `--width/--height` set the default
size. Planning VRAM for 2048² keeps fewer DiT blocks resident (9 instead of 16 on a 12 GB card),
which costs ~3% at 1024²; a 2048² image takes ~26 s. JSON API: `POST /api/generate`
`{prompt, seed?, width?, height?, steps?}` → `{id}`, `GET /api/job/<id>`, `GET /api/image/<id>.png`.

Useful flags:

| flag | meaning |
| --- | --- |
| `--width/--height` | multiples of 16 (default 1024×1024) |
| `--steps`, `--shift`, `--seed` | sampler settings; the seed gives the **same noise as ComfyUI's KSampler** |
| `--prompt-file f.txt`, `-n N` | many prompts / seeds with the models loaded once |
| `--attention flash` | bf16 FlashAttention-2 like ComfyUI's default (closest numerics, ~15% slower per step); the default is `int8`, comfy-kitchen's SageAttention-style INT8 attention |
| `--reserve-vram MB` | VRAM left untouched for the desktop (default 512) |
| `--resident-blocks N`, `--slots N` | override the automatic VRAM residency plan |
| `--dump f.safetensors`, `--ref-noise ref.safetensors` | parity debugging against `tools/ref_dump.py` |

## Results (RTX 5070 12 GB, PCIe 5.0, 64 GB RAM)

1024×1024, 8 steps, warm page cache:

| | ComfyUI (main.py, DynamicVRAM) | krea2 (`--attention flash`) | krea2 (default, `int8`) |
| --- | --- | --- | --- |
| per step | ~1.00 s | ~0.69–0.72 s | ~0.58–0.61 s |
| per image (text + sample + VAE) | ~9.0 s | ~6.2 s | ~5.3 s |
| process start → models ready | (server) | ~3.8 s | ~3.8 s |

Parity with ComfyUI (`tools/ref_dump.py` + `tools/compare.py`, same prompt, seed and noise):
identical tokens, bit-identical seed noise, text-encoder hidden states within ~1e-6 relative,
one DiT evaluation within 0.7% (bf16 rounding level), final image 36–38 dB PSNR (mean ≈1.4/255 per
channel). Turbo trajectories amplify rounding differently per prompt: on some prompts fine details
move, but changing only the attention tile order in this implementation moves them as much.

## How it maps to ComfyUI

* **Text encoder** (`src/te.rs`): Qwen3-VL-4B language model, Krea2 template, hidden states
  entering layers 2, 5, …, 35 (layer 35 itself never runs), system prefix stripped exactly like
  `Krea2TEModel.encode_token_weights`. ComfyUI builds text encoders with `full_precision_mm`, so the
  int8 weights are used weight-only with fp32 activations; this does the same, computing
  `x·((q·s)H)ᵀ` as `(xH)·(q·s)ᵀ` (Hadamard-rotate the activation instead of the weight). The int8
  embedding rows are rounded to bf16 like ComfyUI's quantized embedding.
* **DiT** (`src/dit.rs`): 28 single-stream blocks with W8A8 int8 ConvRot linears. Activations are
  Hadamard-rotated (256-wide groups) and quantized per row in one fused kernel that also applies the
  RMSNorm + AdaLN modulation / sigmoid gate / SwiGLU prologue, rounding to bf16 at the same points
  where ComfyUI materializes bf16 tensors. GEMMs use comfy-kitchen's CUTLASS int8 kernel with the
  dequant (and gated residual) fused in the epilogue. q|k|v|gate and gate|up are fused into single
  GEMMs. Attention is FlashAttention-2 (bf16) reading q/k/v in place from the fused projection.
  The text-fusion adapter is timestep-independent, so it runs once per prompt instead of every step.
* **VAE** (`src/vae.rs`): for one frame every causal 3D conv equals a 2D conv with the kernel's last
  temporal slice, so the decoder runs as NHWC CUTLASS implicit-GEMM 2D convolutions.
* **Sampling** (`src/sampling.rs`): torch-compatible `mt19937` + Box-Muller noise, ModelSamplingFlux
  sigmas and the "simple" scheduler, Euler steps fused into the final un-patchify kernel.

## System-memory offloading (`src/offload.rs`)

Comparable to ComfyUI's DynamicVRAM, but planned statically:

1. Every block is packed once into a contiguous image in the exact device layout the kernels read.
2. From free VRAM minus the activation peak (largest of DiT step / VAE decode) and `--reserve-vram`,
   the planner keeps blocks `0..R` resident and leaves the rest in **page-locked RAM**.
3. Streamed blocks go through a ring of VRAM slots on a dedicated copy stream. A slot is refilled
   as soon as the compute stream releases it (CUDA events), so the PCIe copy of a block overlaps the
   compute of earlier ones; the compute stream only waits if a copy is genuinely late.

On this machine 16 of 28 blocks stay resident at 1024² and 12 stream (5.2 GB/step at ~26 GB/s).
Streaming *all* 28 blocks still costs only ~5% per step, because a block's copy (~17 ms) is shorter
than its compute (~24 ms). The text encoder and text-fusion adapter stream the same way. Planning
stays conservative on purpose: oversubscribing VRAM makes Windows fall back to shared system memory,
which doubles step time.

## Layout

```
build.rs            nvcc build of csrc/ (works around vcvars by passing a clean MSVC env)
csrc/kernels.cu     fused prologue+ConvRot+int8-quantize, QK-norm+RoPE, small attention, VAE ops
csrc/gemm.cu        cuBLASLt bf16/fp32 GEMM, CUTLASS conv2d, int8 GEMM wrappers
csrc/attn.cu        FlashAttention-2 forward wrapper (only the needed variants instantiated)
csrc/attn_int8.cu   optional INT8 attention wrapper
csrc/vendor/        comfy-kitchen CUTLASS int8 GEMM + Sage INT8 attention (Apache-2.0)
src/                Rust: loader, offload engine, tokenizer, text encoder, DiT, VAE, LoRA baking, CLI
tools/              ComfyUI reference dump / comparison / benchmark scripts
```

The CUTLASS and flash-attention checkouts default to `ref/comfy-kitchen/third_party/*` (override with
`KREA_CUTLASS_DIR` / `KREA_FLASH_ATTN_DIR`). Object files are cached by content hash.

Not implemented: CFG > 1 / negative prompts (the turbo workflow uses CFG 1), image inputs to the
text encoder and reference latents (edit mode).
