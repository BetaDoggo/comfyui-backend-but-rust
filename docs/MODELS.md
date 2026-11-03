# Models

What each model family needs, how fast it is and how close it comes to ComfyUI. Timings are from an RTX 5070 12 GB with
PCIe 5.0 and 64 GB of RAM. The files each family reads are listed in the [README](../README.md#models).

| Family | Generates | Sampler default | Checkpoint this engine accepts |
|---|---|---|---|
| Krea 2 Turbo | Images | `euler`, `simple` | W8A8 int8 only |
| Anima | Images | `er_sde`, `simple` | int8 ConvRot only |
| MOSS Image | Images | `euler`, `simple` | int8 ConvRot with fused q/k/v only |
| MiniMax H3 | Video with audio | `res_multistep`, `simple` | `w6a8` only |
| Pixal3D | A 3D mesh from one image | the example workflow's four `euler` samplers | int8 ConvRot (`pixal3d_int8_convrot`) only |

## Krea 2

A standalone reimplementation of ComfyUI's **Krea-2 Turbo** text-to-image workflow
(`blueprints/Text to Image (Krea-2 Turbo).json`: 1024², 8 steps, euler / simple, CFG 1, shift 1.15)
for the int8 checkpoint `kroma-v0.3.1-turbo-opd-W8A8`, the Qwen3-VL-4B text encoder and the
Qwen-Image (Wan 2.1) VAE. It loads the original `.safetensors` files read-only via mmap; nothing
is copied or converted on disk.

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
place, as with Krea 2. Every sampler and scheduler works (see [Samplers](USAGE.md#samplers); SwarmUI's default is `er_sde`), at any CFG, with
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

## MOSS Image

Runs int8 ConvRot MOSS Image checkpoints with fused q/k/v, the Qwen3.5 9B text encoder and the Flux 2 VAE. Both
published key layouts of that VAE load. Every sampler and scheduler works, at any CFG (see [Samplers](USAGE.md#samplers)); SwarmUI's default is `euler`.
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
oversharp, and at 3 it breaks into stripes, exactly as in ComfyUI. LoRAs, init images and partial sampling are refused.

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
ComfyUI's text encoding took 151 s cold. Runs longer than 22 frames and sizes above 512x512 haven't been tested.

Compared with ComfyUI on the same noise, the context is within 8.6e-4, and the first step within 3.4% on video and 13% on audio. Frames come to 21 dB with the same composition.
That gap is rounding: both engines are about 1.2% from an exact float32 run of the token refiner (`tools/h3_refiner_check.py`), and differ by 1.5% from each other because their int8 activations round differently.
The audio hasn't been listened to or judged by ear. `tools/h3_ref.py` and `tools/h3_cmp.py` repeat the comparison.

The LoRA's diffusers file, `minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors`, matches no key here or in ComfyUI. Use the `_comfyui` file.

SwarmUI's Sigma Shift, CFG, every sampler and LoRAs work. These are refused with an error: Rescale CFG, Renorm CFG, CFG Zero Star and TCFG, `<break>`, first and last frame or reference conditioning, single-frame generation, and VAE tile sizes other than 256 / 64.

Not done: the video and audio encoders, sparse attention for long clips, and the web UI.

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
| First evaluation of each sub-model, cond and uncond | 0.6 to 1.0% relative, cosine 0.99995 or better |
| Structure decode | the same voxels |
| Shape decode | 99.99% of the final voxels shared; on those, vertices within 6e-6 and colours within 7e-4 relative |

The float16 decoders flip a few subdivision logits that sit at zero (2 voxels in 25,706 at the upsample), so the final
voxel sets differ slightly. ComfyUI varies more than that between its own runs: its structure stage gave 2206 and then
2200 voxels from the same seed, depending on how it batched CFG.

The full mesh at 1536 has about 16 million triangles. Vertex clustering simplifies it to `--max-faces` (700,000 by
default, the number the example decimates to; 0 keeps everything). Like ComfyUI's GLB it has no normals and a
double-sided material: the dual grid doesn't say which side of a face is inside, and windings are made consistent only
where the mesh is manifold. Attention is bf16 FlashAttention-2, which dominates the refine stage. The engine refuses the
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
