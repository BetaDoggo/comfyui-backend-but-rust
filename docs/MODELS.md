# Models

What each model family needs, how fast it is and how close it comes to ComfyUI. Timings are from an RTX 5070 12 GB with
PCIe 5.0 and 64 GB of RAM. The files each family reads are listed in the [README](../README.md#models).

| Family | Generates | Sampler default | Checkpoint this engine accepts |
|---|---|---|---|
| Krea 2 Turbo | Images | `euler`, `simple` | W8A8 int8 only |
| Anima | Images | `er_sde`, `simple` | int8 ConvRot only |
| MOSS Image | Images | `euler`, `simple` | int8 ConvRot with fused q/k/v only |
| Qwen Image 2.1 | Images with alpha, and edits | `euler`, `simple` | int8 ConvRot only |
| MiniMax H3 | Video with audio | `res_multistep`, `simple` | `w6a8` only |
| HyperFlow H3 Flash | Video with audio, with reference pictures on `ref2va` | `euler` or `res_multistep`, `hyperflow`, 6 steps | int8 ConvRot, ComfyUI's single file of either task |
| MiniMax-H3-Image | Images | `euler`, `simple`, 6 steps | int8 ConvRot (`minimax_h3_image_pruned_int8_convrot`) only |
| Pixal3D | A 3D mesh from one image | the example workflow's four `euler` samplers | int8 ConvRot (`pixal3d_int8_convrot`) only |
| Idol (TripoSplat) | A 3D gaussian splat from one image | `dpmpp_2m`, `simple`, 20 steps, CFG 3 | Comfy-Org's fp16 files |
| SeedVR2 3B | A restored copy of a picture (SwarmUI's SeedVR stage) | `euler`, `simple`, 1 step, CFG 1 | int8 ConvRot (`seedvr2_3b_int8_convrot`) only |

## Support matrix

What each family does through this backend. **Yes** means it produced real output; the sections below and
[STATUS](STATUS.md) say how it was tested and against what. **Refused** means the engine returns an error that SwarmUI
shows, and never drops the setting. **Not tested** means the code accepts it but nothing has run it on that family.

| | Krea 2 Turbo | Anima | MOSS Image | Qwen Image 2.1 | MiniMax H3 | Pixal3D |
|---|---|---|---|---|---|---|
| Makes | Images | Images | Images | Images with alpha | Video with audio | A mesh from one image |
| Checkpoint | W8A8 int8 | int8 ConvRot | int8 ConvRot, fused q/k/v | int8 ConvRot | `w6a8` | int8 ConvRot |
| Other file formats (bf16, fp8, ...) | Refused | Refused | Refused | Refused | Refused | Refused |
| Run through SwarmUI itself | Yes | Yes | Yes | Yes | Yes | Not yet (CLI and `comfy-server` only) |
| Every sampler and scheduler | Yes | Yes | Yes | Yes | Yes | Fixed to the example's four `euler` |
| CFG and negative prompts | Yes | Yes | Yes | Yes | Yes | - |
| Rescale, Renorm, Zero Star, TCFG | Yes | Yes | Yes | Not tested | Refused | - |
| Conditioning multipliers, Zero Negative | Yes | Yes | Yes | Not tested | Not tested | - |
| Batch size (sampled together) | Yes | Yes | Yes | Yes (CLI) | Refused above 1 | First image only |
| Variation seed | Yes | Not tested | Not tested | Not tested | Yes | - |
| Live previews | Yes | Yes | Yes | Not tested | First frame only | - |
| LoRAs | Yes | Yes | Yes | Yes | Yes (the turbo LoRA's ComfyUI file) | - |
| Text encoder LoRAs | Yes | Yes | Yes | Not tested (the code is shared with Krea 2) | Not tested | - |
| Prompt Images and reference images | Yes | Refused | Refused | Yes (one and two tested) | Yes (pictures and keyframes, not compared with ComfyUI) | The input itself |
| Prompt weights | Refused | Yes | Yes | Refused | Yes | - |
| `<break>` | Yes | Refused | Yes | Refused | Refused | - |
| `<alternate>`, `<fromto>` | Refused | Refused | Refused | Refused | Refused | - |
| Init images, refiners, inpainting (partial sampling) | Yes | Yes (refiner not tested) | Yes | Yes (refiner not tested) | Refused | - |
| VAE tiling, Seamless Tileable | Refused | Refused | Refused | Refused | Only the 256 / 64 tile | - |
| Background removal | - | - | - | - | - | Yes (BiRefNet) |

HyperFlow H3 Flash, MiniMax-H3-Image, Idol and SeedVR2 aren't in this matrix; their own sections below say what they
take. ControlNet, regions, IP-Adapter, style models, GLIGEN, LoRA hooks and embeddings are refused on every family. Anima
takes both `(word:1.2)` and `<weight>`; Qwen Image 2.1's encoder reads `(word:1.2)` as plain text.

### Batches

The images of a batch go through the diffusion model together, as one batch in every block, and the sampler steps them
together as ComfyUI does. When they don't all fit in VRAM, the engine samples as many at a time as fit, after moving
blocks out to make room. With `euler`, each image of a batch of 2 was pixel-identical to its own render at batch 1 on
Krea 2, Anima, MOSS Image and Qwen Image 2.1, at CFG 1 and with a negative prompt (CFG 2 or 3). That held through the
CLI and, for Anima, through `comfy-server` over HTTP. Krea 2 matched too at batch 8, which ran with 7 of its 28 blocks
left in VRAM, and when forced to sample a batch of 3 one image at a time.

Sampling time, the same images one at a time against all together:

| Family, size | One at a time | Together | PCIe streamed |
| --- | --- | --- | --- |
| Krea 2, 4 at 512² | 7.83 s | 4.26 s | 139 GB → 38 GB |
| Krea 2, 2 at 1024² | 9.1 s | 9.05 s | 83 GB → 42 GB |
| Krea 2, 8 at 1024² | not run (4.56 s per image alone) | 37.0 s | 21 of 28 blocks streamed |
| Anima, 4 at 512² | 0.98 s | 0.89 s | none (resident) |
| MOSS Image, 4 at 512² | 2.16 s | 1.98 s | |
| MOSS Image, 2 at 1024², CFG 3 | 9.0 s | 8.72 s | |
| Qwen Image 2.1, 2 at 1024², CFG 3 | 11.47 s | 11.63 s | |

Batching pays when the model streams blocks and each image is small: every streamed block is then copied once per step
for the whole batch. At 1024² these models are compute-bound, so a batch takes as long as its images one at a time.
Not measured: other sizes and Qwen Image 2.1 below 1024².

## Krea 2

Runs the W8A8 int8 Krea 2 Turbo checkpoint `kroma-v0.3.1-turbo-opd-W8A8` with the Qwen3-VL-4B text encoder and the
Qwen-Image (Wan 2.1) VAE. Files are read in place. Its defaults follow ComfyUI's Krea-2 Turbo blueprint
(`blueprints/Text to Image (Krea-2 Turbo).json`): 1024², 8 steps, `euler` / `simple`, CFG 1, shift 1.15. Every sampler
and scheduler works, at any CFG, with negative prompts, as on the other families.

```bash
target/release/diffusion.exe -p "a red fox sitting in a snowy forest, golden hour, photograph" --seed 42 -o fox.png
```

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

### Reference images

Krea 2 takes images two ways, both as in ComfyUI, and SwarmUI's Prompt Images use them:

* **Through the text encoder**, the default. Each image goes through Qwen3-VL's vision tower ([`src/vision.rs`](../src/vision.rs)) and
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

## Anima

Runs int8 ConvRot Anima checkpoints with the Qwen3 0.6B text encoder and the Qwen-Image VAE. Files are read in
place. Every sampler and scheduler works (see [Samplers](USAGE.md#samplers); SwarmUI's default is `er_sde`), at any CFG, with
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

Anima drifts towards explicit images even on neutral prompts, so choose prompts with care.

### SnapRef

[SnapRef](https://github.com/ArcAnima/SnapRef) is a reference adapter for Anima: pictures of a character, and a prompt
that describes only the scene. SwarmUI's [SwarmUI-SnapRef](https://github.com/BetaDoggo/SwarmUI-SnapRef) extension sends
its `SnapRef` node, which this backend runs natively ([`src/snapref.rs`](../src/snapref.rs)). Each picture is shrunk
to `ref_megapixels`, VAE-encoded, and run once through the model at sigma 0 with the adapter's reference LoRA and mode
embedding; every block's self-attention keys and values are kept (about 0.9 GB per 1 MP picture) and prepended in every
target evaluation, which runs with the adapter's target LoRA. The keys are cached until the pictures, caption, mode,
size, LoRAs or model change, and dropped when the backend goes idle.

```bash
target/release/diffusion.exe --anima --snapref Lora/snapref_v2a.safetensors --image robot.png -p "no humans, robot, sitting on a park bench, autumn leaves"
```

`--snapref-mode`, `--snapref-strength`, `--snapref-fidelity` and `--snapref-megapixels` are the extension's parameters.

Compared with ComfyUI's own node (`tools/snapref_ref.py`, `tools/snapref_cmp.py`) on v2a, a 1024x1024 picture of a
robot and an 8-step turbo run at 1024x1024, both engines start from this engine's reference latent:

| | Relative error |
|---|---|
| Reference keys / values, first block | 0.17% / 0.04% |
| Reference keys / values, last block | 0.55% / 1.3% |
| Target first block, first evaluation (strength 1, strength 1.5, the negative at Fidelity 0.5 and CFG 3) | 1.2%, 1.2%, 1.3% |

The VAE encode of that picture differs from ComfyUI's by 2.2%, which moves the first block's reference keys by 10%.
Images are 17 to 18 dB from ComfyUI's, the same as this engine's int8 and flash attention are from each other (17.3 dB),
and show the same character in the same pose. Only v2a was compared; v1a and v1b were not run.

Speed at 1024x1024 with one 1 MP picture: encoding the picture's keys takes about 225 ms once. Each evaluation takes
183 ms instead of 129 ms. A strength other than 1, or Fidelity below 1 on the negative, uses bf16 flash attention with a
log-sum-exp merge, which takes 264 ms per evaluation.

Refused: `ref_guidance_on`, which needs an extra model call per step (SwarmUI never sends it), a CLIP other than the
prompt's, and more than one SnapRef node. `multi_layout` stitch, `use_batch` and the per-picture weights are implemented
but were only run through the mock server, not on real output. The extension itself was not run in SwarmUI; the node
was tested through `comfy-server` over HTTP and through the CLI.

## MOSS Image

Runs int8 ConvRot MOSS Image checkpoints with fused q/k/v, the Qwen3.5 9B text encoder and the Flux 2 VAE. Both
published key layouts of that VAE load, with its encoder, whose latents have a cosine similarity of 0.99984 with ComfyUI's on a 768x512 picture. Every sampler and scheduler works, at any CFG (see [Samplers](USAGE.md#samplers)); SwarmUI's default is `euler`.
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

## Qwen Image 2.1

Runs int8 ConvRot Qwen Image 2.1 checkpoints with the Qwen3-VL 8B text encoder and its RGBA VAE (64 channels at 1/16, so
the image can have alpha). Every sampler and scheduler works. The model is a CFG 1 model: at CFG 1.5 the picture turns
oversharp, and at 3 it breaks into stripes, exactly as in ComfyUI. Init images and inpainting work, 47.6 and 51.7 dB from ComfyUI through SwarmUI.

```bash
target/release/diffusion.exe --qwen21 -p "no people, a stone lighthouse on a rocky coast at sunset, calm sea" --steps 8 -o lighthouse.png
target/release/diffusion.exe --qwen21 --image lighthouse.png -p "turn it into a night scene with a starry sky" --steps 8 -o night.png
```

**Edits.** Each `--image` is shown to the text encoder (through the 8B vision tower, which is 334 ms for a 768x768
image) and VAE-encoded as a reference latent. The sequence is the text with each image spliced in at its slot, then the
image being made. The text and the references are modulated as if at t = 0 and attend only to what comes before them, so
their keys and values are the same at every step: they are computed once and kept (a cache of 2.2 GB per 1024x1024
reference, per prompt). A step then runs only the new image's rows. With no room, up to 3 GiB per prompt, the prefix
runs again in every step instead, which gives the same picture byte for byte (`QWEN21_NO_CACHE=1` forces it).

Measured here (RTX 5070, 12 GB), 8 steps, after warm-up:

| | ComfyUI | This engine |
|---|---|---|
| 1024x1024 text to image, sampling | 0.46 s/step (3.68 s) | 0.36 s/step (2.90 s) |
| 768x768 edit, one reference, sampling | not timed | 0.22 s/step cached, 0.48 s/step without the cache |
| 1024x1024 edit, one 1024x1024 reference | not timed | 0.44 s/step, part of the blocks streaming to make room for the cache |
| VAE decode, 1024x1024 | not timed | 0.42 s |
| Text encoder, a short prompt | not timed | 0.3 s |

In one step at 1024x1024 the int8 GEMMs take 241 ms (gate and up 110, down 57, qkv 54, o 20) and attention 68 ms.

**Against ComfyUI** (same seed and noise, its own code on the same files):

| | Difference |
|---|---|
| Text encoder output, with and without an image | 1.7e-3 and 6.8e-3 relative |
| First model evaluation, each block's image rows | 6e-3 to 2e-2 relative; the prefix rows 2e-3 to 9e-3 |
| VAE decode of a random latent | 3.3e-3 relative |
| VAE round trip of a picture | 39.01 dB from the original, ComfyUI's 38.98 dB; ours against theirs 54.5 dB |
| 512x512 text to image, 4 steps | latent 3.8e-2 relative, picture 36.6 dB |
| 768x768 edit, 4 steps | latent 1.2e-2 relative, picture 46.1 dB |

**LoRAs.** They are baked into the int8 weights as ComfyUI does (dequantize, patch in bf16, requantize with stochastic
rounding seeded by the module name), for the DiT and for the 8B text encoder. Tested with `qwen-image-2.1-fix-1.0`, a
rank 32 DoRA on 22 blocks' attention and MLP weights (132 weights): block 5's four modules match ComfyUI's baked weights
to 0.05% of values differing (the requantized int8 codes differ in 0.003% by one step), and the picture at 768x768 and 8
steps is 45.1 dB from ComfyUI's. The first generation with it takes 23 s to bake, then it costs nothing, and switching the
LoRA off restores the weights in 1.3 s with a picture identical to one made before it was loaded (checked through
`comfy-server`). That bake was 63 s before the DoRA arithmetic ran on every core. The maintainer reports that it and the other LoRA formats work through SwarmUI; I only ran the DoRA above, through
`comfy-server` and the CLI. A LoRA with text encoder keys wasn't tried.

**Transparent references.** SwarmUI's image loader (`SwarmLoadImageB64`) drops the alpha channel, here and in ComfyUI, so a
reference with a transparent background reaches the model as an opaque picture of whatever colour is stored under the
transparent pixels. Qwen Image 2.1's own transparent areas are stored as magenta, so an alpha picture it made and then
edited through SwarmUI comes back with a pink opaque background. From the CLI (`--image`) the alpha is kept: the VAE encoder
matches ComfyUI's with alpha in the picture (round trip 44.6 dB in alpha and 48.7 dB in colour where opaque), and the
edit keeps its transparency.

Not tested: SwarmUI itself (the two graphs SwarmUI makes for alpha text to image and a two-picture edit ran through
`comfy-server` over HTTP, 8 s and 11 s from a cold start), more than two reference images, and the fp8 and bf16 files,
which are refused.

## MiniMax H3

H3 runs text to video with audio, from SwarmUI or with `--h3` on the CLI: the `w6a8` checkpoint, the nvfp4 text encoder, and the turbo LoRA in its ComfyUI form.
The CLI writes an mp4 through `ffmpeg`, which must be on `PATH`.

```bash
target/release/diffusion.exe --h3 --width 512 --height 512 --frames 5 --steps 4 --lora "$MODELS_ROOT/Lora/minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16_comfyui.safetensors" -p "a stone lighthouse on a sea cliff at dawn, no people" -o clip.mp4
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
ComfyUI's text encoding took 151 s cold.

Longer clips, per step once warm, 4 turbo LoRA steps, against ComfyUI run with SwarmUI's flags (`--fast --use-ck-attention`,
comfy-kitchen's int8 attention):

| Clip | Tokens | ComfyUI | Here, int8 (default) | Here, `--attention flash` |
|---|---|---|---|---|
| 512x512, 90 frames | about 7k | 3.46 s | 1.99 s | 2.75 s |
| 768x512, 124 frames | about 12k | 6.2 s | 5.0 s | 8.37 s |

H3 used to default to flash attention. Attention grows with the square of the length, so past about 100 frames flash
fell behind ComfyUI: at 768x512 and 124 frames it took 5.4 s of the 8.4 s step, where int8 takes 1.9 s. Offloading didn't
matter: 25 of the 50 blocks were streamed, and the waits for their copies came to 0 to 11 ms per step. The GEMMs, 2.5 s
of the step, run at about 230 TOPS, near this card's int8 peak. These int8 runs sampled, but their clips weren't
watched, and int8 and flash weren't compared on H3. Nothing above 768x512 or 124 frames was tried.

Compared with ComfyUI on the same noise, the context is within 8.6e-4, and the first step within 3.4% on video and 13% on audio. Frames come to 21 dB with the same composition.
That gap is rounding: both engines are about 1.2% from an exact float32 run of the token refiner (`tools/h3_refiner_check.py`), and differ by 1.5% from each other because their int8 activations round differently.
The audio hasn't been listened to or judged by ear. `tools/h3_ref.py` and `tools/h3_cmp.py` repeat the comparison.

The LoRA's diffusers file, `minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors`, matches no key here or in ComfyUI. Use the `_comfyui` file.

SwarmUI's Sigma Shift, CFG, every sampler and LoRAs work. These are refused with an error: Rescale CFG, Renorm CFG, CFG Zero Star and TCFG, `<break>`, reference videos and audio, and VAE tile sizes other than 256 / 64.

Single image mode (1 frame) samples one latent frame and decodes it as ComfyUI does: a chunk of five copies through the
video decoder, keeping the first real frame. A 768x512 still at 20 steps and CFG 5 took about 1 s per step once warm, through
`comfy-server` over HTTP; not compared with ComfyUI's picture, and not run through SwarmUI itself.

### Reference pictures and keyframes

These were tried through SwarmUI, where the maintainer found them working. Which modes were tried wasn't recorded, nothing
was measured, and nothing was compared with ComfyUI. They follow ComfyUI's code:

* **Pictures** go through the 32B text encoder's vision tower (the 8B's tower with a 5120-wide merger, under `visual.`),
  shown as `<Picture i>: ` then the vision block, before the prompt, with Qwen3-VL's MRoPE positions and DeepStack. The
  vision tokens, with their start and end tokens, take the video modality's modulation in the DiT. With a VAE, each
  picture is also encoded and packed as reference rows between the text and the targets, as `PackedLayout` does.
  SwarmUI's references (`SwarmMiniMaxH3CollectReferences`) are scaled to the clip's area (`match`) or a 2048 short edge
  (`max`), with lanczos, to multiples of 32. Its Prompt Images on `SwarmTextEncodeAdvanced` only reach the text encoder.
* **Keyframes** (`SwarmMiniMaxH3AddKeyframes`: Image To Video's first frame, Video End Frame) are encoded by the video
  VAE and packed on the target's grid at their frame's time.
* Both kinds of rows run at time `max(t, 0.999)` and are mixed with 0.1% of noise from the sampler's seed, as ComfyUI's
  `_cond_video_rows` does.
* The VAE encoder ([`h3vae::Encoder`](../src/h3vae.rs)) runs one frame: each causal conv becomes a 2D conv with its last
  temporal tap, with reflect padding and per-frame GroupNorm, in fp16 on 256px tiles blended on the host, as ComfyUI's
  `tiled_encode` does. The video VAE takes the posterior's mean. MiniMax-H3-Image's VAE samples it with seed 42, as
  ComfyUI does.

On the CLI, `--image` adds a reference picture and `--first-frame` and `--last-frame` add keyframes (stretched to the
clip). Refused with an error: reference videos and audio (they need the video-block vision path and the audio encoder),
references or keyframes with context windows or sparse attention, pictures on the negative prompt,
and SwarmUI Prompt Images together with references.

Not done: the audio encoder, and encoding clips longer than one frame.

### Sparse attention

SwarmUI's Use Sparse Attention, ComfyUI's Model Sparse Attention node, runs on H3 and HyperFlow H3 Flash with comfy-kitchen's
own Sol-Attn kernels, copied into `csrc/vendor/sage/`. `sol` is the training-free adaptive threshold. `topk`, ComfyUI's `sla`,
keeps a fixed share of key blocks, with the threshold ported from comfy-kitchen to `csrc/sparse_attn.cu`. The node's window,
`min_tokens`, `extra_tokens`, `dense_blocks` and `sink_conditioning` work as in ComfyUI: the text and audio rows stay exact,
and the audio queries run dense. On HyperFlow, `dense_blocks` counts blocks by their index in the checkpoint. VSA is refused
with an error that suggests the other two methods: it needs FastH3's VSA-trained weights. Other model families refuse the
node.

ComfyUI feeds H3's kernel through a chunked producer that quantizes with the previous step's statistics. This engine already
has q, k and v after RoPE, so it uses comfy-kitchen's direct entry, which takes the statistics from the current step.
`tests/sparse_attn_match_comfy.rs` runs both methods, with and without sinks and extra tokens, on inputs from
`tools/sparse_attn_ref.py`. Its output matches comfy-kitchen's `sol_attn` bit for bit, and the top-k thresholds match within
3e-7. Those inputs come out 12 to 16% from dense attention.

On the CLI, 512x512, 22 frames (1,882 tokens), 4 turbo LoRA steps, sparse was forced with `--sparse-min-tokens 0`. Attention
took 53 to 60 ms per step with `sol` and 66 to 74 ms with `sla`, against 90 ms dense. The first step stayed dense, as the
window sets. `sla` kept dense's composition (14.9 dB). `sol` changed the composition (11.2 dB) and showed some blocky texture.
Sequences that short are below the default 12,288 tokens, where both stay dense. The maintainer reports that it works
through SwarmUI, though it hasn't been tested extensively. Its speedup on longer clips, above that threshold, hasn't been
measured, and its output hasn't been compared with ComfyUI's.

### Context windows

SwarmUI's Video Context Window (Video Context Length and Overlap) sends ComfyUI's `MiniMaxH3ContextWindows` node, which
`src/h3window.rs` follows. Each model evaluation runs on overlapping stretches of the clip, and their denoised outputs are
blended. Lengths in frames become latents as the node converts them, so with an overlap of at most 5 latents below the
length, every window starts on one of the model's 5-latent chunks. Each window takes the audio latents of the same pixel
frames, a window's rope positions start at its own first latent, and FreeNoise shuffles the initial noise from a CPU
generator seeded with the sampler's seed, video and audio apart. All four fuse methods are there. SwarmUI uses `pyramid`. A
clip no longer than one window samples as it would without the node. Other families refuse it.

The windows, the audio split and FreeNoise were checked against ComfyUI's own functions (`cargo test --lib h3window`). On
the CLI, 384x384, 260 frames (77 latents), the turbo LoRA's 4 steps and windows of 124 and 39 frames, against
`tools/h3_ref.py --context-windows 124,39` on the same noise, the first fused evaluation is within 4.6% on video and 11.3%
on audio. That is close to the 3.4% and 13% without windows, and the error is flat across the clip, with no rise where
windows meet. Decoded with ComfyUI's VAE, the two clips have the same composition at 16.6 dB. Both break up into blocks and
streaks in the later windows, so that comes from context windows at 4 steps, not from this port. Through SwarmUI with the
same settings, sampling took 18.4 s for the 4 steps, against ComfyUI's 41.4 s on its warm second run. Other sizes,
lengths, step counts and fuse methods weren't tried, and the audio wasn't judged by ear.

### HyperFlow H3 Flash

[HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) is a community few-step H3, which
ComfyUI doesn't run without its custom nodes. It runs from ComfyUI-HyperFlow-Flash's single files, put in
`Stable-diffusion/`: `ref2va_int8_convrot_comfyui/` (reference to video) or `fl2va_int8_convrot_comfyui/` (text, first
and last frames to video) in that repository. Each is picked by the `hyperflow` entry in its metadata, and loads as H3
with H3's own text encoder and decoders. The diffusers folders (`fl2va_int8_convrot/`, `ref2va_int8_convrot/`) are
refused. Reference pictures go through H3's own path ([Reference pictures and keyframes](#reference-pictures-and-keyframes)):
`--image` on the CLI, `<Picture n>` in the prompt, or SwarmUI's references.

```bash
target/release/diffusion.exe --h3 --dit "$MODELS_ROOT/Stable-diffusion/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors" --width 512 --height 512 --frames 22 --steps 6 --scheduler hyperflow --image fox.png -p "<Picture 1> trots slowly through the snowy forest at sunset, no people" -o clip.mp4
```

What differs from H3, all read from the file as it is:

* Some blocks are dropped (9 in `ref2va`; 6, 21 and 22 in `fl2va`) and the rest renumbered, so block indices, such as
  sparse attention's `dense_blocks`, count as ComfyUI counts them.
* Each linear is either ComfyUI's `int8_tensorwise` ConvRot (W8A8 behind the same 256-wide Hadamard, q, k and v
  joined) or a low-rank bf16 pair run as two GEMMs, with q, k and v apart and already padded to multiples of 64. q, k
  and v share one down GEMM. Every SwiGLU input is already in ComfyUI's `[gate; value]` order.
* The token refiner and the video head are bf16 and float32, not int8.
* Each AdaLN reads `temb(t, r) = emb_t(t) + gate (emb_r(r) - emb_t(t))`, where `r` is where the step ends: the
  schedule's next sigma, found as HyperFlow's ComfyUI node finds it. Audio's `r` follows its own shift, as its `t`
  does. As in ComfyUI's model, `temb` is rounded to bf16 before the SiLU. The gate (0.25) and the 6-step schedule are
  tensors (`hyperflow_gate`, `hyperflow_sigmas`); the stored schedule matches the `hyperflow` scheduler's within 1e-4.
* Keyframe and reference rows run at `max(t, 0.999)`, as on H3, with `r = t`. HyperFlow's node looks the endpoint up
  by time among the targets' and finds none for them.

The `hyperflow` scheduler is its 4, 6 or 8-step grid, shifted by the model's video shift (12, or SwarmUI's Sigma Shift).
The server lists it on `SwarmKSampler`, so SwarmUI offers it; other models refuse it. Other schedulers run too, with `r`
taken from their own next sigma, as in the ComfyUI node; only the grid is what the model was distilled on. Samplers:
`euler`, which it was distilled with, and `res_multistep` (SwarmUI's default) with its `cfg_pp` form. Each evaluates
once per step at the schedule's sigma and lands on the next, so the endpoint `r` is where the step really ends.
On the `ref2va` fox clip timed below, `res_multistep` kept the same fox, scene and motion as `euler`, but left bright white
blown-out patches around the legs and tail that `euler` doesn't, so `euler` stays the better choice. Refused: other
samplers, since ancestral, SDE and midpoint steps evaluate or land elsewhere, and LoRAs, since its layers no longer
match H3's.

The comfy-rust SwarmUI extension registers the files as MiniMax H3 models. SwarmUI's own check needs block 49 and
misses them.

`ref2va`, on this machine, 512x512, 22 frames, 6 steps, one 512x512 reference picture of a fox in a snowy forest: 0.53 s
per evaluation warm, and 5.4 s for the first, which reads the blocks. The clip kept the fox and the scene, and the fox
stood and walked. Reading the files cold, the vision tower took 29 s to pack and the text encoder 32 s. The maintainer
reports it works through SwarmUI. Not compared with ComfyUI's HyperFlow node, nothing larger tried, audio not listened to.
Reference videos and audio are refused, as on H3. Keyframes on HyperFlow are no longer refused, but weren't tried.

`fl2va`'s single file has the same tensor names and format as `ref2va`'s, and differs only in its dropped blocks and
per-block ranks, but it hasn't been run. Its diffusers folder used to run here, before the single files: at 512x512,
22 frames, 6 steps, a clean, coherent clip at 0.49 s per evaluation warm, with the int8 GEMMs about 215 ms, the
low-rank pairs about 100 ms and attention about 90 ms.

## MiniMax-H3-Image

[MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) is text to image on H3's transformer without its
audio stream, step-distilled to 6 steps. It runs from `--h3-image` on the CLI, or from `comfy-server` with ComfyUI's
`MiniMaxH3ImageConditioning` node. It uses the int8 ConvRot release, H3's nvfp4 text encoder, and its own VAE.

```bash
target/release/diffusion.exe --h3-image --width 2048 --height 2048 -p "a stone lighthouse on a sea cliff at dawn, waves crashing on the rocks below, soft golden light, no people" -o image.png
```

`--steps` defaults to the model's 6 there, with `euler`, `simple` and CFG 1, as the release asks. It reuses H3's code:

* The blocks are ComfyUI's int8 ConvRot, run like HyperFlow's int8 linears. The token refiner is plain bf16 and goes
  through HyperFlow's bf16 refiner path. The packing, rope and modulation are H3's, with no audio rows.
* The output head is a bank of 6 PDD heads, each distilled for one interval of the sigma grid in the file's metadata. A
  step takes the dt-weighted mean of the heads its interval spans, as ComfyUI's `_pdd_head` does. That head is combined
  on the host and cached while the interval stays the same.
* The model's sampling walks that grid (`ModelSampling::Grid`, ComfyUI's `MiniMaxH3ImageModelSampling`), so the stock
  schedulers at 6 steps land on it.
* The VAE decoder is H3's ViT decoder at half the width (1024, 16 heads, 24 layers), with fp16 weights run as bf16, one
  frame per latent, and rope over height and width. It decodes in H3's 256px tiles, with exact attention, as ComfyUI does
  for this unquantized decoder. The file has no latent statistics, so ComfyUI's constants are used.

Compared with ComfyUI on the same noise at 1024x1024 (`tools/h3_image_ref.py`), the context is within 8.6e-4, the first
evaluation within 2.5%, the final latent within 4.6%, and the images are 30.6 dB apart with the same composition.

Warm, on this machine's 12 GB card, which holds 23 of the 50 blocks and streams the rest:

| Size | Here | ComfyUI |
|---|---|---|
| 1024x1024, per step | 0.35 s | 0.68 s |
| 2048x2048, per step | 1.07 s | 1.34 s |
| 2048x2048, VAE decode | 0.55 s of tiles, 3.2 s with the first read of its weights | 0.93 s |

Streamed blocks are held in pinned RAM after their first read, and lie between resident ones, so their copies overlap compute; at 1024 that took the step from 0.86 s to 0.35 s, with the same image byte for byte. The VAE decodes 32 tiles at a time with FlashAttention-2 (54.5 dB from the earlier one-tile decode). The first two evaluations still wait for the blocks' first read. Through `comfy-server` over HTTP, a
warm 2048x2048 graph took 10.6 s in all and a 1024x1024 one 6.1 s.

**Text encoding.** The encoder's layers stay page-locked in the file's mapping after their first read and are streamed
from there for each prompt, so the diffusion model keeps its VRAM
([details](TECHNICAL_OVERVIEW.md#comfyui-memory-management)). From the CLI with `--prompt-file`, a prompt change took
about 1.0 s to encode (0.8 s of it the layers' own work, for about 20 tokens), where it took 53 to 355 s when each
prompt read the layers from the file again. The first prompt of a new process took 2.2 s with the file in Windows' cache,
against 50 to 240 s, and 2.5 to 4 minutes from this machine's drive when it wasn't. The images were the same byte for byte. These
were measured with three readers at normal I/O priority; the single low-priority reader used now wasn't timed.

**Edits.** `MiniMaxH3ImageConditioning`'s pictures, and `--image` on the CLI, are shown to the text encoder as
`Picture i: ` at a 1024 short edge and, with the VAE, packed as reference rows at a short edge between 1024 and 2048, as
described under [reference pictures](#reference-pictures-and-keyframes). With pictures, the node's width and height of
0 take the first picture's VAE size. Edits were tried through SwarmUI's graph (below) and found working, but not compared with
ComfyUI. `tools/h3_image_ref.py --image` dumps ComfyUI's reference latents (`ref0`, ...) to compare with `--dump`.

With one 1024x1024 picture the prompt is 1048 tokens and took 7.2 to 7.5 s to encode. The encoder is causal, so a
prompt of 256 tokens or more leaves its keys, values and final states in RAM (about 410 KB a token, up to 1 GB), and the
next prompt that starts with the same tokens and pictures, followed only by text, runs just its own tokens against
them ([`h3te::Prefix`](../src/h3te.rs)). That path is written but untested: it has not run to completion, so its
speed and its output are unknown.

Refused with an error: LoRAs, init images, noise masks, `<break>`, context windows and sparse attention. The bf16 and
w4a8 releases are refused too.

In SwarmUI the model has its own class, MiniMax H3 Image. SwarmUI loads H3's text encoder and the image VAE with their
own loaders, makes the latent with `SwarmEmptyMiniMaxH3ImageLatent`, and passes Prompt Images through
`SwarmMiniMaxH3ImageCollectReferences` to `SwarmTextEncodeAdvanced`, labelled `Picture i: ` and sized as
`MiniMaxH3ImageConditioning` sizes them. Set Steps to 6 and CFG Scale to 1. Batch Size above 1 is refused; use Images
instead. Through `comfy-server` over HTTP, that graph made a 1024x1024 picture in 6.0 s warm, and an edit of it from one
Prompt Image in 45 s, 37 s of it encoding the prompt with the picture on its first run. Both were checked by eye, not compared with ComfyUI,
and not yet run from SwarmUI itself.

Only `euler` and `simple`, and only 1024x1024 and 2048x2048, were compared with ComfyUI.

## Pixal3D

Pixal3D turns one cut-out photo into a vertex-coloured mesh: `--pixal3d object.png` on the CLI, or the
[Pixal3D extension's](../swarm%20backend/pixal3d-extension/README.md) tab in SwarmUI, through the `Pixal3DImageTo3D` node.
It follows Comfy-Org's example workflow node for node: `ImageCropToMask` (pad 1.1), `Pixal3DConditioning` (DINOv3 at 512
and 1024, three NAF maps), the structure, shape and refine samplers with their CFG Override, Rescale CFG and shift
patches, the texture sampler, the shape and texture decodes and `PaintMesh`. The example's background removal and MoGe
field-of-view estimate aren't part of it: the image's alpha is the mask, and the field of view is a parameter (49.13 by
default).

```bash
target/release/diffusion.exe --pixal3d teapot.png --resolution 1536 -o teapot.glb
```

Warm files, on this machine, with the example's steps:

| Resolution | Tokens in the refine stage | Refine | Texture | Whole mesh |
|---|---|---|---|---|
| 1024 | 10,851 | 13.1 s | 7.2 s | 40 s |
| 1536 | 26,000 to 28,500 | 56 to 67 s | 30 to 36 s | 110 to 128 s |

ComfyUI wasn't timed. On 12 GB it needs `comfy_aimdo`: without it, the same workflow ran out of VRAM.

Compared stage by stage with ComfyUI on its own inputs (`tools/pixal3d_ref.py` dumps them, `--pixal3d-check` compares):

| Stage | Difference |
|---|---|
| `ImageCropToMask` composite | identical |
| DINOv3 tokens (float32) | 1.6e-4 relative |
| NAF maps (float16) | 3e-4 relative |
| Projected features | 1.6e-3 relative |
| First evaluation of each sub-model, cond and uncond | 0.6 to 1.2% relative with int8 attention, 0.6 to 1.0% with `--attention flash`; cosine 0.9999 or better |
| Structure decode | the same voxels |
| Shape decode | 99.99% of the final voxels shared; on those, vertices within 6e-6 and colours within 7e-4 relative |

The float16 decoders flip a few subdivision logits that sit at zero (2 voxels in 25,706 at the upsample), so the final
voxel sets differ slightly. ComfyUI varies more than that between its own runs: its structure stage gave 2206 and then
2200 voxels from the same seed, depending on how it batched CFG.

The full mesh at 1536 has about 16 million triangles. Vertex clustering simplifies it to `--max-faces` (700,000 by
default, the number the example decimates to; 0 keeps everything). Like ComfyUI's GLB it has no normals and a
double-sided material: the dual grid doesn't say which side of a face is inside, and windings are made consistent only
where the mesh is manifold. Self-attention is int8 by default, as on the other families: an evaluation over 25,706 tokens took 1.20 s against 2.59 s with bf16 FlashAttention-2 (`--attention flash`). The cross-attention to the five image tokens stays bf16. The mesh timings above are from before this change, and weren't measured again (the DINOv3 file wasn't installed). The engine refuses the
multi-view checkpoint, and uses only the first image of a batch.

Not done: multi-view, the field-of-view estimate, UV unwrapping and texture baking, and the bf16 checkpoint.

### Background removal (BiRefNet)

An opaque image is cut out first, as the example workflow's `RemoveBackground` does: `--pixal3d` runs BiRefNet when
the PNG has no transparency, and comfy-server has the stock `LoadBackgroundRemovalModel` and `RemoveBackground` nodes.
The model is Comfy-Org's `birefnet.safetensors` in `background_removal/` (float16 weights; the only architecture
ComfyUI's loader knows, and the only one this engine accepts). Its Swin-L backbone runs on the image at 1024 and at 512,
and a decoder of deformable convolutions makes the mask, which is resized back to the image's size.

```bash
target/release/diffusion.exe --remove-background teapot.png -o teapot_cut.png
```

Everything runs in float32 with the weights widened and every BatchNorm folded into the convolution before it. On the
teapot (1024x1024) the model takes 0.86 s to load and 0.46 s to run, and a workflow of the two nodes took 1.5 to 2.0 s
through the server, image decoding and saving included. Against ComfyUI, which ran the float16 weights, on that image and on a 1280x960
copy (so the resizes before and after are checked too):

| Stage | Difference |
|---|---|
| Input after the resize and rounding to 8 bits | 6e-8 relative at 1024, 7e-6 at 1280x960 |
| Backbone outputs, each of the 8 | 9e-4 to 5e-3 relative, cosine 0.99998 or better |
| Squeeze block | 7e-4 relative at 1024, 1.3e-3 at 1280x960 |
| Mask logits | 4.7e-3 to 4.9e-3 relative |
| Final mask | at most 1.7e-3 (1024) and 1.8e-3 (1280x960) from ComfyUI's, 1e-6 to 2e-6 on average |

The differences are the size of float16 rounding in ComfyUI. The image is always resized to 1024x1024, so a mask of an
image with another aspect ratio comes from a squashed copy, as in ComfyUI. Not tested: other BiRefNet variants, and
SwarmUI's own Remove Background parameter, which sends `SwarmRemBg` (rembg's U2-Net) and is still refused.

## TripoSplat and Idol

Idol, Comfy-Org's fine-tune of TripoSplat, turns one cut-out photo into a 3D gaussian splat: `--triposplat object.png`
on the CLI, or the [Pixal3D extension's](../swarm%20backend/pixal3d-extension/README.md) tab in SwarmUI (Model: Idol),
through the `TripoSplatImageToSplat` node. It follows Comfy-Org's Idol workflow: `TripoSplatPreprocessImage` (erode 1,
1024), `TripoSplatConditioning` (DINOv3 ViT-H/16+ and the Flux 2 VAE), a `KSampler` over the nested latent (8192 shape
tokens and a camera token) with `dpmpp_2m`, `simple`, 20 steps and CFG 3, and `VAEDecodeTripoSplat` with 262144
gaussians. The splat is saved as `SaveGaussianSplat` writes a `.ply`. The workflow's live preview, its splat-to-mesh
and video branches, and the separate decode seed aren't part of it: the decode takes the sampler's seed.

```bash
target/release/diffusion.exe --triposplat teapot.png --vae VAE/flux2-vae.safetensors -o teapot.ply
```

Warm files, on this machine, the teapot at the example's settings: the whole splat in 14.3 to 14.6 s, of which sampling
is 11.3 to 11.6 s (40 model evaluations over 12,294 tokens). Through comfy-server's node over HTTP it took 15.6 s, and
16.6 s for an opaque image cut out by BiRefNet first. ComfyUI ran the same workflow at 111 s/step in the headless
reference script (`tools/triposplat_ref.py`, with comfy_aimdo on 12 GB); that isn't a tuned ComfyUI timing.

Compared stage by stage with ComfyUI on its own inputs (`--triposplat-check` against the script's dump):

| Stage | Difference |
|---|---|
| `TripoSplatPreprocessImage` | identical |
| DINOv3 ViT-H tokens (float32) | 1.8e-6 relative |
| Flux 2 VAE latent | 1.1e-2 relative, cosine 0.99994 |
| First evaluation, uncond | 0.4% relative, cosine 0.99999 (the dump holds no cond evaluation) |
| Sampled latent after 20 steps at CFG 3 | 9% relative, cosine 0.996 |
| Gaussian decoder features, on ComfyUI's points | 2.6% relative, cosine 0.9997 |
| Gaussians assembled from ComfyUI's features | identical positions and colours; scales and opacities within 1e-4; rotations the same up to sign |

The model and decoder run in float16 like ComfyUI's, with int8 attention (there is no float16 or head-dim-64 FlashAttention
build here to compare it with). Small differences grow over the 20 CFG steps, but the splats from the same seed look
the same as ComfyUI's in the tab's viewer: shape, lid, spout, handle and glaze highlights. The octree's point sampling
follows torch's CPU generator, and the token anchors torch's scrambled Sobol sequence, bit for bit.

Not tested: TripoSplat's own checkpoint (only Idol was run), other image sizes than 1024, and running through SwarmUI.

## SeedVR2

ByteDance's one-step restoration model, run as SwarmUI's SeedVR stage does: SwarmUI sends `SeedVR2Preprocess` (pad to
multiples of 16), `VAEEncode` with SeedVR2's VAE, `SeedVR2Conditioning`, one `euler` step at CFG 1, `VAEDecode` and
`SeedVR2PostProcessing` (colour matched to the picture, CIELAB by default). The DiT is [`src/seedvr.rs`](../src/seedvr.rs):
32 MM-DiT blocks with a fixed text prompt stored in the checkpoint, attending in windows of about 400 tokens that
alternate with half-shifted ones. On one frame the causal video VAE is a 2D VAE with each conv's temporal taps summed
([`src/seedvr_vae.rs`](../src/seedvr_vae.rs)); a clip runs through it a frame at a time, see [Clips](#clips). The pre-
and post-processing nodes run in the server. SeedVR2 is held
beside the loaded family, so the model that made the picture stays loaded (its weights only move to RAM for room).

Files: `diffusion_models/SeedVR2/seedvr2_3b_int8_convrot.safetensors` and `VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors`
(the path SwarmUI downloads its SeedVR2 VAE to), from [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2).

Warm, on this machine, a 512 picture upscaled to the size shown, against ComfyUI running the same nodes headless
(`tools/seedvr_ref.py`):

| Size | VAE encode | DiT (one evaluation) | VAE decode | ComfyUI sampling | ComfyUI decode |
|---|---|---|---|---|---|
| 1024x1024 | 134 ms | 145 ms | 234 ms | 235 ms | 276 ms |
| 2048x2048 | 705 ms | 568 ms | 1119 ms | 855 ms | 2631 ms (tiled) |

The whole 1024 job through `comfy-server` over HTTP, image loading and two PNG saves included, took 0.79 s warm; the
colour correction is 136 ms of it, on the CPU. ComfyUI's encode wasn't timed.

Compared with ComfyUI on its own inputs (`--seedvr-check` against the script's dump), at 1024:

| Stage | Difference |
|---|---|
| VAE latent | 1.4% relative, cosine 0.99990 |
| ComfyUI's latent through our decoder | 63 dB from ComfyUI's decode |
| Block 0 / block 9 image tokens | 0.5% / 1.0% relative |
| Denoised latent after 32 blocks | 7.7% relative, cosine 0.997 |
| Decoded picture | 44.5 dB from ComfyUI's (int8 and flash attention here are 44.1 dB apart) |
| Final picture through `comfy-server`, colour correction included | 44.3 dB from ComfyUI's (the input changes by 27 dB) |

At 2048 the denoised latent is again at cosine 0.997, and the decoded picture 40.2 dB from ComfyUI's, which tiles its
decode at this size on 12 GB (its own latent through our untiled decoder: 41.4 dB).

The VAE tiles itself when a picture doesn't fit, as ComfyUI's does: square tiles overlapping by a quarter, blended with
cosine ramps, sized by ComfyUI's `preferred_decode_tile` (256 to 768 pixels) from the free VRAM plus the weights the
memory management can move to RAM. When to tile is decided by this engine's own estimate, about half of ComfyUI's, so
pictures it tiles can stay whole here (2048x2048 does on 12 GB; ComfyUI tiles it). Decoding ComfyUI's 2048 latent in
ComfyUI's 768 pixel tiles comes out 62.6 dB from ComfyUI's tiled decode. ComfyUI tiles its encode only after running out
of memory, in 256 pixel tiles; here the encode tiles like the decode. A tiled encode decodes back 38.0 dB from the
picture at 2048, against 39.3 dB for a whole one: the encoder's attention spans the picture, so any tiling changes it.
`SEEDVR_VAE_TILE=<latent pixels>` forces a tile size, 0 none.

`--max-side` doesn't bound SeedVR2: its VAE tiles and its DiT attends in windows, so memory grows linearly with the
picture. Through `comfy-server`, a 1024 picture restored at 4096x4096 took 15.7 s warm: encode 3.5 s and decode 6.3 s in
768 pixel tiles, the DiT 2.3 s. 3328x4864 took 17.8 s cold, the same stages at 3.5, 2.4 and 6.4 s. A 2560x1440
restore, untiled, took 3.0 s. ComfyUI wasn't timed at these sizes. Above 2900 squared the DiT runs its MLP in chunks of
32768 rows. Nothing larger than 4096x4096 was tried.

The maintainer has run it through SwarmUI, up to 3328x4864.

### Clips

A batch of frames is one clip, as in ComfyUI: `SeedVR2Preprocess` pads it to 4n + 1 frames, the VAE makes n + 1 latent
frames (`[1, 16, n + 1, H/8, W/8]`), the DiT samples them together with windows over time as well as space, and the
decoder makes the 4n + 1 frames back. `SeedVR2PostProcessing` drops the padding frames. The VAE runs a frame at a time:
each causal conv keeps its last two input frames and sees them in front of the new one, which gives the same sums as
ComfyUI's pass over the whole clip. Those kept frames cost about 8 KB per output pixel in the decoder and 4 KB in the
encoder, so a clip tiles at a smaller size than a picture; tiles are square, the largest that fit.

Against ComfyUI on its own inputs (`tools/seedvr_ref.py --frames N`, then `--seedvr-check`), a scenery pan, warm:

| Clip | VAE encode | DiT (one evaluation) | VAE decode | ComfyUI sampling | Decoded frames from ComfyUI's |
|---|---|---|---|---|---|
| 9 frames, 384x256 | 186 ms | 47 ms | 452 ms | not warm | 48.3 to 53.1 dB |
| 17 frames, 768x432 | 1227 ms | 252 ms | 2850 ms | 372 ms | 50.8 to 52.1 dB |

At 17 frames the VAE latent is at cosine 0.99996 and ComfyUI's latent through this decoder 60.5 dB from ComfyUI's
decode; the denoised latent is at cosine 0.9994, and flash attention lands as far from ComfyUI as int8. ComfyUI's decode
of the 17 frames took 18.8 s in the script's single run, which wasn't a warm comparison. The 9-frame clip through
`comfy-server` over HTTP with SwarmUI's node chain (frames joined by `ImageBatch`, CIELAB colour correction) came out
45.7 to 49.7 dB from ComfyUI's final frames. Tiled clip decodes of ComfyUI's latent: 50.97 dB from its untiled decode in
256 pixel tiles, 38.7 dB in 128 pixel ones. Clips weren't run through SwarmUI itself, nor longer than 17 frames or larger
than 768x432.

SeedVR Split Latent samples a clip in chunks of latent frames, as ComfyUI's `SeedVR2TemporalChunk` and
`SeedVR2TemporalMerge` do: the chunks overlap by Temporal Video Overlap, each is conditioned and sampled with the same
seed, and the overlaps are crossfaded back with a Hann window over their middle third. The server runs the conditioning
and sampler once per chunk, as ComfyUI does for list outputs. SwarmUI always sends `auto`, which ComfyUI sizes with a
law fitted to its own memory use (8.5 GB held back, so on 12 GB it picks one latent frame a chunk); here `auto` takes the
most latent frames one evaluation fits by this engine's estimate, so the chunks, and the frames, can differ from
ComfyUI's on the same card. `manual` chunks are ComfyUI's. The 17-frame clip at 768x432 in `manual` chunks of 13 frames
overlapping by 3 latent frames (two chunks of 4) came out 46.1 to 49.8 dB from ComfyUI's chunked run through
`comfy-server` over HTTP. `auto` was only run where the whole clip fits, which it then samples in one chunk; the estimate
wasn't tested at the edge of VRAM.

### What SeedVR2 is missing here

Each of these is refused with an error SwarmUI shows, unless marked untested:

| SwarmUI setting | Through ComfyUI | Here |
|---|---|---|
| Variation Seed, or a noise mask, on a clip | apply | **Refused** on clips; pictures take them as before |
| SeedVR Model: 7B, 7B sharp, the 3B's fp16, fp8, mxfp8 and nvfp4 files | all run | **Refused**: only the 3B int8 ConvRot file runs |
| SeedVR Upscale Method `model-*` (ESRGAN and the like) | `ImageUpscaleWithModel` | **Refused**: the engine runs no upscale models yet, for any family |
| SeedVR Upscale Method `pidmodel-*` | a pixel decoder | **Refused** |
| SeedVR Upscale Method `latent-*` | `LatentUpscaleBy` | Should work (latent upscale works on the image models); **untested** with SeedVR |
| VAE Tile Size (`VAEDecodeTiled`, `VAEEncodeTiled`) | tiles at that size | **Refused**; the VAE tiles by itself instead |
| LoRAs on the SeedVR model | apply | **Refused** |
| SeedVR2 as the Refiner Model | runs the same stage | **Untested** (same code path) |
| SeedVR CFG above 1 | the model's stored negative prompt | **Untested** |
| Colour correction wavelet, AdaIN, none | GPU | Implemented on the CPU; only CIELAB compared with ComfyUI, the others have unit tests |
| Live previews | none for SeedVR2's latents | none either |

The `pixel-*` upscale methods resize with the Rust `image` crate's filters, not PIL's, so SeedVR's input differs a
little from ComfyUI's when SwarmUI upscales before it. The encoder's stride 2 convolutions run as stride 1 convolutions
whose odd samples are kept, which wastes about a tenth of the encode.
