# Models

What each model family needs, how fast it is and how close it comes to ComfyUI. Timings are from an RTX 5070 12 GB with
PCIe 5.0 and 64 GB of RAM. The files each family reads are listed in the [README](../README.md#models).

| Family | Generates | Sampler default | Checkpoint this engine accepts |
|---|---|---|---|
| Krea 2 Turbo | Images | `euler`, `simple` | W8A8 int8 only |
| Anima | Images | `er_sde`, `simple` | int8 ConvRot only |
| MOSS Image | Images | `euler`, `simple` | int8 ConvRot with fused q/k/v only |
| MiniMax H3 | Video with audio | `res_multistep`, `simple` | `w6a8` only |

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
