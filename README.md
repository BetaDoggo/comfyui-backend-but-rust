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

You need Windows x64, an NVIDIA GPU (the default build targets RTX 50-series, `sm_120`; set `CUDA_ARCH` for others),
[Rust](https://rustup.rs), the CUDA 13 toolkit (`CUDA_PATH`) and MSVC Build Tools 2022.

```bash
git clone --recursive https://github.com/Comfy-Org/comfy-kitchen ref/comfy-kitchen
```

Run that inside this repository: the build takes CUTLASS and FlashAttention-2 from that checkout. Then build, which takes
about 6 minutes the first time:

```bash
cargo build --release
```

Put your models in one folder, keeping these subfolders, and point `MODELS_ROOT` at it (or pass `--models`, or `--dit`,
`--te` and `--vae` for any other file):

| Model | `Stable-diffusion/` | `clip/` | `VAE/` |
|---|---|---|---|
| Krea 2 Turbo | `kroma-v0.3.1-turbo-opd-W8A8.safetensors` | `qwen3vl_4b.safetensors` | `qwen_image_vae.safetensors` |
| Anima | `animaInt8Mxfp8_turboV11Int8.safetensors` | `qwen_3_600m.safetensors` | `qwen_image_vae.safetensors` |
| MOSS Image | `moss_image_sft_int8_convrot_attn.safetensors` | `qwen3.5_9b_int8_convrot.safetensors` | `flux2-vae.safetensors` |

Generate an image from the command line:

```bash
target/release/diffusion.exe -p "a red fox in a snowy forest" --seed 42 -o fox.png
```

```bash
target/release/diffusion.exe --anima -p "no humans, scenery, mountain lake at sunrise" -o lake.png
```

```bash
target/release/diffusion.exe --moss -p "a stone lighthouse on a sea cliff at dawn" --steps 20 --cfg 4 -o lighthouse.png
```

Add `--lora path/to/lora.safetensors:0.8` to any of them. To use it from SwarmUI, set `MODELS_ROOT`, run
[`swarm backend/start-backend.bat`](swarm%20backend/start-backend.bat), and add `http://127.0.0.1:8188` as a "ComfyUI API By URL"
backend. Keep a real ComfyUI backend next to it, because anything else is refused or reaches a mock engine.

More: [technical overview](docs/TECHNICAL_OVERVIEW.md), [status](docs/STATUS.md), [SwarmUI backend](swarm%20backend/README.md).

Licensed under the [GNU GPL v3](LICENSE), like ComfyUI, which this code was written from. Files under `csrc/vendor/` stay Apache-2.0, see `csrc/vendor/LICENSE.comfy-kitchen`.

## What this is

A Rust and CUDA backend that speaks ComfyUI's API and replaces the ComfyUI features SwarmUI uses most, so
SwarmUI can generate without Python or ComfyUI. The goal is speed first: Krea 2 Turbo takes about 4.65 s per
1024x1024 image here, against about 9 s in ComfyUI on the same machine, and Anima Turbo samples 8 steps in
0.92 s against 1.59 s. MOSS Image gains least, about 5 to 8% per step, because ComfyUI already runs it on the same kernels.

**Status:** three model families really generate, text-to-image only, with LoRAs: Krea 2 Turbo, Anima and MOSS Image. The server accepts far more of
SwarmUI's features than that, but they only reach a mock engine so far. [`docs/STATUS.md`](docs/STATUS.md) lists everything that works and everything
still to do, in priority order. Run it today beside a normal ComfyUI backend.

* `swarm backend/` is the server: the ComfyUI protocol, the node graph, and the `Engine` interface. Start it with
  [`start-backend.bat`](swarm%20backend/start-backend.bat) and see [its README](swarm%20backend/README.md).
* The repository root is the real engine: the Krea 2, Anima and MOSS Image pipelines described below. They also run alone
  from the command line, and Krea 2 has a small web UI.
* [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md) explains what the project is built on and how the pieces fit.

## LoRAs

All three models take LoRAs, from SwarmUI or with `--lora path[:strength[:clip_strength]]` (repeatable, applied in order; the clip strength defaults to the strength):

```bash
target/release/diffusion.exe --moss --lora moss-dmdv1.safetensors --steps 8 --cfg 1 -p "a stone lighthouse on a sea cliff at dawn, no people" -o lighthouse.png
```

They are baked, as ComfyUI does for quantized weights: each patched int8 weight is dequantized to bf16 (float32 in a text
encoder), the patches are added in that dtype, and it is requantized with new row scales and stochastic rounding seeded by the module's name. Sampling
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

### LoRA formats

Every format in ComfyUI's `comfy.weight_adapter` and `load_lora` reads on all three models, with all its key styles:

* Plain LoRA in each naming (`lora_up`/`lora_down`, `lora_A`/`lora_B`, the diffusers, Mochi, transformers and `.default` forms), `alpha`, and LoCon mid weights
* DoRA on any adapter, LoHa, LoKr, GLoRA, OFT, BOFT
* `diff`, `diff_b`, `w_norm`, `b_norm` and `set_weight`; when a file patches a weight twice the last wins, as in ComfyUI
* Text encoder patches, with `lora_te_`, `text_encoders.<encoder>.transformer.model.` and `text_encoders.transformer.model.` keys, at their own clip strength
* MOSS Image control adapters, whose extra inputs are zeros because control images aren't supported yet

`tools/lora_formats.py` writes one file with every format on real modules, to compare through `tools/lora_ref.py` and
`tools/lora_cmp.py`. The results against ComfyUI, on all three models, with the files' values random:

| Format | Patched weights compared with ComfyUI |
|---|---|
| Plain, `alpha`, LoCon mid, LoHa, LoKr, GLoRA, `diff`, `set`, `w_norm` | identical |
| Tucker LoHa | under 1e-8 |
| DoRA | within 2e-4, from the norm's rounding |
| Text encoder patches on Krea 2 and MOSS Image | float weights identical; about 0.0001% of int8 values off by 1 |

Differences to know about:

* **ComfyUI fails on OFT, BOFT and Tucker LoKr here.** On these int8 models its adapters log an error and skip the module, the first two
  because they mix float32 and bf16, the last in `torch.kron`. These apply them instead, so a file using them gives a different image from ComfyUI. OFT and BOFT were
  compared with ComfyUI's adapters run by hand in float32, and agree to 1.6% of the change. Tucker LoKr has no reference result.
* LoHa and LoKr multiply out to a full weight on the host, so a large file bakes slowly: 13 full-size patches on Krea 2 took 73 s, against 11 s for 228 plain ones.
* Anima's text encoder keeps bf16 weights where ComfyUI computes in float32, so patched weights there are rounded to bf16.
* Refused with an error: reshaped patches, which widen a model's inputs; text encoder patches to the vision tower, `embed_tokens` or
  norm weights this engine doesn't store as patchable; control images; different LoRAs on the positive and negative text encoders;
  and LoRA hooks. Files for ComfyUI custom nodes, such as `anima_ref_kv`, match no key and change nothing, as in ComfyUI.

## Samplers

All 45 ComfyUI samplers and all 9 ComfyUI schedulers run on all three models, plus SwarmUI's `turbo`, `align_your_steps`,
`flux2`, `ideogram4`, `ideogram4turbo` and `ltxv` (`ltxv-image` is refused), its Sigma Min, Sigma Max and Rho overrides,
and any CFG with a negative prompt, now including Krea 2:

```bash
target/release/diffusion.exe --anima --sampler dpmpp_2m_sde_gpu --scheduler karras --steps 12 --cfg 4 --negative "lowres" -p "no humans, scenery, mountain lake"
```

They are ports of ComfyUI's code in f32 ([`src/samplers.rs`](src/samplers.rs)), running on the host between model evaluations.
Evaluations cost what they did: Anima at 122 ms and Krea 2 at 574 ms at 1024x1024, with the latent's transfers included.
The Brownian samplers reimplement `torchsde.BrownianTree` and `numpy.random.SeedSequence`, for the CPU and GPU variants.

`tools/sampler_ref.py` runs ComfyUI's own samplers on an analytic mock model, and `tests/samplers_match_comfy.rs` replays that:

| Check | Result |
|---|---|
| 1105 schedules (5 model samplings, 7 step counts) | worst relative error 1.7e-6 |
| 45 samplers in 5 setups (225 runs), same sigmas passed to the model in the same order | worst relative error 8.1e-6, most about 1e-7 |
| Brownian increments at full latent size, CPU and GPU | equal to rounding |

So the gap to ComfyUI on a real model is the gap the model already has. Stochastic and high CFG paths magnify it: on
Anima, `dpmpp_2m_sde_gpu` with `beta` at CFG 4 is 11 dB from ComfyUI, and also 12 dB from this engine under the other
attention backend. `dpmpp_2m_sde` with `karras` is 24 dB.

Differences to know about:

* `dpm_adaptive` picks its steps from the whole batch in ComfyUI, and from each image here, so its batches can differ.
* `dpm_fast` and `dpm_adaptive` stop at the last non-zero sigma, as in ComfyUI, so they can leave visible noise.
* Combinations ComfyUI runs badly run badly here too: `turbo` and `align_your_steps` use sigmas above 1 on these models.
* The Brownian samplers (`dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde` and their variants) generate their noise on the CPU, even
  the `_gpu` ones. An 8-step Anima image at 1024x1024 took 2.4 s with `dpmpp_2m_sde` and 2.0 s with its `_gpu` variant, against
  1.3 s with `euler`.

## Krea 2 reference images

Krea 2 takes images two ways, both as in ComfyUI, and SwarmUI's Prompt Images use them:

* **Through the text encoder**, the default. Each image goes through Qwen3-VL's vision tower ([`src/vision.rs`](src/vision.rs)) and
  its tokens follow the prompt as `Picture N:`, on the negative prompt too.
* **As reference latents**, with SwarmUI's Enable Reference Latents. The image is also VAE encoded and joins the DiT's tokens,
  with `index` or `index_timestep_zero`. These are for edit LoRAs; the base model mostly copies the reference.

```bash
target/release/diffusion.exe -p "no people, a stone lighthouse at sunset" --image ref_small.png --ref-image ref_full.png --ref-method index
```

The CLI reads PNGs at their own size. SwarmUI scales them first, to about 384x384 for the text encoder and to the generation's
size for the latent.

Compared with ComfyUI's own code and SwarmUI's own text node (`tools/krea_image_ref.py`, `tools/krea_image_cmp.py`), at 768x768 with 8 steps:

| Case | ComfyUI | This engine | Image against ComfyUI |
|---|---|---|---|
| Two prompt images, flash attention | 694 ms/evaluation | 430 ms | 40.8 dB |
| One prompt image and an `index` reference | 1040 ms/step | 787 ms flash, 651 ms int8 | 24.7 dB |
| The same with `index_timestep_zero`, CFG 2 | 1660 ms/step | 1183 ms flash, 1011 ms int8 | 24.9 dB |

Reference latents are sensitive to rounding: the same picture appears, but int8 and flash attention here differ from each other
by 27 dB and 19 dB on those two cases, so a gap of that size is expected from rounding. Passing ComfyUI's exact latent to this
engine changed nothing. The model's first output is within about 1% in every case, the vision tokens within 0.25%, and the VAE encoder's latent within 0.6%.

A prompt image at about 384x384 costs 130 ms in the vision tower and a VAE encode about 50 ms. The vision tower's attention
grows with the square of its tokens: a 1024x1024 image, SwarmUI's Large Text Encoded Image, took 3.9 s.

Differences to know about:

* `input_image_N` in a prompt stays plain text. SwarmUI's code means to replace it, but ComfyUI's tokens keep the text, so these do as well.
* No edit LoRA for Krea 2 is installed here, so reference latents have only run on the base model.
* The other reference methods (`offset`, `uxo/uno` and so on) are refused, and so are references on the negative prompt, which SwarmUI doesn't send.

## MiniMax H3 engine

H3 runs text to video with audio, from SwarmUI or with `--h3` on the CLI: the `w6a8` checkpoint, the nvfp4 text encoder, and the turbo LoRA in its ComfyUI form.
The CLI writes an mp4 through `ffmpeg`, which must be on `PATH`.

```bash
diffusion --h3 --width 512 --height 512 --frames 5 --steps 4 --lora "$MODELS_ROOT/Lora/minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16_comfyui.safetensors" -p "a stone lighthouse on a sea cliff at dawn, no people" -o clip.mp4
```

`--frames` snaps up to the model's 17k + 5 grid, so 12 becomes 22. The defaults are ComfyUI's template: `res_multistep`, `simple`, CFG 1, 4 steps with the LoRA.

Warm files, 512x512, 4 steps, turbo LoRA, on this machine:

| Stage | ComfyUI, 5 frames | Here, 5 frames | Here, 22 frames |
|---|---|---|---|
| Text encoding | 13.1 s | 2.2 s | 2.3 s |
| Per step, warm | 1.59 s | 0.24 s | 0.53 s |

Through real SwarmUI with everything warm, a repeat generation took 2.6 s at 5 frames and 3.8 s at 22, because the bake, the encoded prompt and both decoders are kept.
That covers sampling, video decode and audio decode. ComfyUI took 28.3 s for video decode and 3.6 s for audio decode alone.
The first video decode reads 2.4 GB from the file, about 23 s on this drive.

Baking the LoRA adds about 11 s when it changes, and text encoding from a cold disk takes about 140 s on both engines, as the drive reads at about 100 MB/s.
ComfyUI's text encoding took 151 s cold. Runs longer than 22 frames and sizes above 512x512 haven't been tested.

Offloading is the lazy kind in [`src/offload.rs`](src/offload.rs): nothing is read at load, reader threads fill up to four pinned staging buffers ahead of compute, taken only when a unit needs one and returned when the set parks. Pages read once leave the process right after, which took a H3 run's peak working set from 52 GB to 32 GB.
Weights stay in packed 6-bit form in VRAM and decode to int8 before each GEMM, which leaves room for 30 of the 50 blocks at 512x512. Streamed blocks are spread evenly among the resident ones, so each copy overlaps compute instead of queueing behind the last; at 5 frames that took a step from 0.33 s to 0.24 s, and copies no longer hold it up.
Splitting weights to keep them in L2 gained 2 to 3%, so it isn't used: the int8 GEMM already runs close to what this GPU can do.
LoRA-baked blocks keep a pinned host copy, about 15 GB, and go straight to VRAM: this machine copies 18 GB/s in RAM but 28 GB/s from pinned memory to VRAM, so a pass through staging cost about 200 ms a step.
If pinning fails they fall back to pageable memory and staging. The blocks leave VRAM before the video decode and come back for the next clip, re-planned for its size.
The text encoder and refiner only hold VRAM while they run. The video decoder holds 2.4 GB of pinned RAM between clips, and takes no VRAM while sampling.

`PROFILE=1` prints the time of each stage per step and per decoded tile. Stages are timed with events on the compute stream, so the run keeps its overlap and its output.

Compared with ComfyUI on the same noise, the context is within 8.6e-4, and the first step within 3.4% on video and 13% on audio. Frames come to 21 dB with the same composition.
That gap is rounding: both engines are about 1.2% from an exact float32 run of the token refiner (`tools/h3_refiner_check.py`), and differ by 1.5% from each other because their int8 activations round differently.
The audio hasn't been listened to or judged by ear. `tools/h3_ref.py` and `tools/h3_cmp.py` repeat the comparison.

The LoRA's diffusers file, `minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors`, matches no key here or in ComfyUI. Use the `_comfyui` file.

SwarmUI's Sigma Shift, CFG, every sampler and LoRAs work. These are refused with an error: Variation Seed, first and last frame or reference conditioning, single-frame generation, and VAE tile sizes other than 256 / 64.

Not done: the video and audio encoders, sparse attention for long clips, and the web UI.

## MOSS Image engine

Runs int8 ConvRot MOSS Image checkpoints with fused q/k/v, the Qwen3.5 9B text encoder and the Flux 2 VAE. Both
published key layouts of that VAE load. Every sampler and scheduler works, at any CFG (see [Samplers](#samplers)); SwarmUI's default is `euler`.
Prompts are literal: like ComfyUI, this encoder has no `(word:1.2)` emphasis.

```bash
target/release/diffusion.exe --moss -p "a stone lighthouse on a sea cliff at dawn, no people" --steps 20 --cfg 4 --negative "blurry" -o lighthouse.png
```

Measured at 1024x1024, 20 steps, CFG 4 (two model evaluations per step), after warm-up:

| | ComfyUI | This engine (`--attention flash`) | This engine (default, `int8`) |
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
place, as with Krea 2. Every sampler and scheduler works (see [Samplers](#samplers); SwarmUI's default is `er_sde`), at any CFG, with
negative prompts and `(word:1.2)` emphasis.

```bash
target/release/diffusion.exe --anima -p "no humans, scenery, mountain lake at sunrise" --seed 42 -o lake.png
```

Measured at 1024x1024 with 8 steps, CFG 1 and `er_sde`, after warm-up:

| | ComfyUI | This engine (`--attention flash`) | This engine (default, `int8`) |
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
target/release/diffusion.exe -p "a red fox sitting in a snowy forest, golden hour, photograph" --seed 42 -o fox.png
```

### Web UI

```bash
target/release/diffusion.exe --serve 127.0.0.1:7860
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
| `--sampler`, `--scheduler`, `--cfg`, `--negative` | any ComfyUI sampler and scheduler, on any model |
| `--image`, `--ref-image`, `--ref-method` | Krea 2 reference images, see [Krea 2 reference images](#krea-2-reference-images) |
| `--prompt-file f.txt`, `-n N` | many prompts / seeds with the models loaded once |
| `--attention flash` | bf16 FlashAttention-2 like ComfyUI's default (closest numerics, ~15% slower per step); the default is `int8`, comfy-kitchen's SageAttention-style INT8 attention |
| `--reserve-vram MB` | VRAM left untouched for the desktop (default 512) |
| `--resident-blocks N`, `--slots N` | override the automatic VRAM residency plan |
| `--dump f.safetensors`, `--ref-noise ref.safetensors` | parity debugging against `tools/ref_dump.py` |

## Results (RTX 5070 12 GB, PCIe 5.0, 64 GB RAM)

1024×1024, 8 steps, warm page cache:

| | ComfyUI (main.py, DynamicVRAM) | This engine (`--attention flash`) | This engine (default, `int8`) |
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
* **Sampling** (`src/sampling.rs`, `src/samplers.rs`): torch-compatible `mt19937` + Box-Muller noise, and
  ComfyUI's samplers and schedulers on the host.

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
`CUTLASS_DIR` / `FLASH_ATTN_DIR`). Object files are cached by content hash.

Not implemented: init images and inpainting, which need partial sampling.
