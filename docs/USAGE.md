# Usage

How to run the command line, and how LoRAs and samplers behave. Timings are from one machine: an
RTX 5070 12 GB with PCIe 5.0 and 64 GB of RAM. To set up, see the [README](../README.md). For each model, see
[Models](MODELS.md).

## Command line flags

`target/release/diffusion.exe` runs Krea 2 unless you pick another family.

| flag | meaning |
| --- | --- |
| `--width/--height` | multiples of 16 (default 1024×1024) |
| `--steps`, `--shift`, `--seed` | sampler settings; the seed gives the **same noise as ComfyUI's KSampler** |
| `--sampler`, `--scheduler`, `--cfg`, `--negative` | any ComfyUI sampler and scheduler, on any model |
| `--image`, `--ref-image`, `--ref-method` | Krea 2 reference images, see [Krea 2 reference images](MODELS.md#reference-images) |
| `--prompt-file f.txt`, `-n N` | many prompts / seeds with the models loaded once. `--prompt-file` is read by Krea 2, MOSS Image and MiniMax-H3-Image |
| `--batch N` | sample up to N of a prompt's `-n` images together, through the model at once. Each image keeps its own seed's noise, and the sampler's own noise is one tensor for the batch, as in ComfyUI, so stochastic samplers give other images than with `--batch 1` |
| `--attention flash` | bf16 FlashAttention-2 like ComfyUI's default (closest numerics, ~15% slower per step); the default is `int8`, comfy-kitchen's SageAttention-style INT8 attention |
| `--reserve-vram MB` | VRAM left untouched for the desktop (default 256, as ComfyUI's `--reserve-vram`; 512 with `--legacy-offload`) |
| `--slots N` | cast buffers for streamed blocks (default 2, ComfyUI's offload streams); the streaming ring's slots with `--legacy-offload` |
| `--legacy-offload` | plan VRAM residency from estimates, the older offloading, instead of the default [ComfyUI memory management](TECHNICAL_OVERVIEW.md#comfyui-memory-management) |
| `--resident-blocks N` | with `--legacy-offload`: force the number of diffusion blocks kept in VRAM |
| `--anima`, `--moss`, `--qwen21`, `--h3`, `--h3-image` | pick the model family; the default is Krea 2. `--h3` with a MiniMax-H3-Image `--dit` runs it too; see [MiniMax-H3-Image](MODELS.md#minimax-h3-image) |
| `--snapref file`, `--image a.png`, `--snapref-mode`, `--snapref-strength`, `--snapref-fidelity`, `--snapref-megapixels` | Anima: the SnapRef reference adapter with pictures of a character (repeatable `--image`); see [SnapRef](MODELS.md#snapref) |
| `--image a.png` | Qwen Image 2.1 edits: each image goes to the text encoder and into the sequence as a reference latent (repeatable); the prompt then says what to change, as in `put the boat from picture 2 on the sea in picture 1`. Output is RGBA |
| `--image a.png`, `--first-frame a.png`, `--last-frame b.png` | MiniMax H3: reference pictures, `<Picture n>` in the prompt (repeatable), and keyframes stretched to the clip. MiniMax-H3-Image: pictures to edit or refer to. See [reference pictures](MODELS.md#reference-pictures-and-keyframes) |
| `--qwen21-vae-check ref.safetensors` | Qwen Image 2.1: compare the VAE with `tools/qwen21_ref.py --vae-only` |
| `--flux2-vae-check ref.safetensors` | MOSS Image: compare the Flux 2 VAE encoder with `tools/flux2_vae_ref.py` |
| `--frames N` | H3 video length, snapped up to the 17k + 5 grid (default 5) |
| `--sparse sol` or `--sparse sla` | H3 block-sparse attention, as ComfyUI's Model Sparse Attention node: Sol-Attn's adaptive threshold (`--sparse-tau`, default 1.3) or SLA-style top-k (`--sparse-keep` percent, default 10). `--sparse-start` (default 0.2), `--sparse-min-tokens` (default 12288) and `--sparse-extra-tokens` (default 256) are the node's start_percent, min_tokens and extra_tokens; see [sparse attention](MODELS.md#sparse-attention) |
| `--context-windows 124,39` | H3: sample in context windows of LENGTH,OVERLAP frames, as SwarmUI's Video Context Window: pyramid blending and FreeNoise; see [context windows](MODELS.md#context-windows) |
| `--lora path[:strength[:clip_strength]]` | repeatable, see [LoRAs](#loras) |
| `--models`, `--dit`, `--te`, `--vae` | model folder, or individual files |
| `--dump f.safetensors`, `--ref-noise ref.safetensors` | parity debugging against `tools/ref_dump.py` |
| `--pixal3d image.png` | Pixal3D: a GLB of the object in the PNG (its alpha is the mask; an opaque image is cut out with BiRefNet first), written to `-o`; see [Pixal3D](MODELS.md#pixal3d) |
| `--remove-background image.png`, `--birefnet model.safetensors` | BiRefNet: the PNG with its background made transparent, written to `-o`; `--birefnet` is also the model `--pixal3d` cuts out with (default `background_removal/birefnet.safetensors`) |
| `--birefnet-check ref.safetensors` | BiRefNet: the stage-by-stage comparison with `tools/birefnet_ref.py` |
| `--fov`, `--resolution`, `--max-faces`, `--no-texture`, `--pixal3d-seeds a,b,c,d` | Pixal3D: the photo's field of view, the refine stage's voxel resolution, the triangle limit, a grey mesh, and the four samplers' seeds |
| `--triposplat image.png` | TripoSplat / Idol: a gaussian splat (PLY) of the object in the PNG (its alpha is the mask; an opaque image is cut out with BiRefNet first), written to `-o`; `--dit`, `--clip-vision` and `--vae` pick the flow model, DINOv3 ViT-H and Flux 2 VAE, `--splat-decoder` the octree gaussian decoder; see [TripoSplat and Idol](MODELS.md#triposplat-and-idol) |
| `--splat-steps`, `--splat-cfg`, `--gaussians`, `--decode-seed`, `--triposplat-check ref.safetensors` | TripoSplat: the sampler's steps (20) and CFG (3), the gaussians to decode (262144), the decode's point-sampling seed (default `--seed`), and the stage-by-stage comparison with `tools/triposplat_ref.py` |
| `--seedvr-check ref.safetensors` | SeedVR2: the stage-by-stage comparison with `tools/seedvr_ref.py`; `--dit` and `--vae` pick the files (default `diffusion_models/SeedVR2/seedvr2_3b_int8_convrot.safetensors` and `VAE/SeedVR2/seedvr2_ema_vae_fp16.safetensors`); `SEEDVR_FLASH=1` runs it with flash attention |
| `--clip-vision`, `--texture-vae`, `--pixal3d-check ref.safetensors` | Pixal3D's DINOv3 and texture VAE files (`--dit` and `--vae` are its diffusion model and shape VAE), and the stage-by-stage comparison with `tools/pixal3d_ref.py` |

Set `PROFILE=1` to print the time of each stage per step and per decoded tile. Stages are timed with events on the compute
stream, so the run keeps its overlap and its output.

## LoRAs

Every model takes LoRAs, from SwarmUI or with `--lora path[:strength[:clip_strength]]` (repeatable, applied in order; the clip strength defaults to the strength):

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

Every format in ComfyUI's `comfy.weight_adapter` and `load_lora` reads on Krea 2, Anima and MOSS Image, with all its key styles:

* Plain LoRA in each naming (`lora_up`/`lora_down`, `lora_A`/`lora_B`, the diffusers, Mochi, transformers and `.default` forms), `alpha`, and LoCon mid weights
* DoRA on any adapter, LoHa, LoKr, GLoRA, OFT, BOFT
* `diff`, `diff_b`, `w_norm`, `b_norm` and `set_weight`; when a file patches a weight twice the last wins, as in ComfyUI
* Text encoder patches, with `lora_te_`, `text_encoders.<encoder>.transformer.model.` and `text_encoders.transformer.model.` keys, at their own clip strength
* MOSS Image control adapters, whose extra inputs are zeros because control images aren't supported yet

`tools/lora_formats.py` writes one file with every format on real modules, to compare through `tools/lora_ref.py` and
`tools/lora_cmp.py`. The results against ComfyUI, on Krea 2, Anima and MOSS Image, with the files' values random:

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
  and LoRA hooks. Files for ComfyUI custom nodes, such as SnapRef's `anima_ref_kv`, match no key and change nothing, as in ComfyUI.
  SnapRef files go to `--snapref` instead.

## Samplers

Every ComfyUI sampler and scheduler runs on the image and video models, plus SwarmUI's `turbo`, `align_your_steps`,
`flux2`, `ideogram4`, `ideogram4turbo` and `ltxv` (`ltxv-image` is refused), its Sigma Min, Sigma Max and Rho overrides,
and any CFG with a negative prompt, now including Krea 2. One more scheduler, `hyperflow`, is HyperFlow H3 Flash's
4, 6 or 8-step grid and runs on that model only, with `euler` ([details](MODELS.md#hyperflow-h3-flash)):

```bash
target/release/diffusion.exe --anima --sampler dpmpp_2m_sde_gpu --scheduler karras --steps 12 --cfg 4 --negative "lowres" -p "no humans, scenery, mountain lake"
```

They are ports of ComfyUI's code in f32 ([`src/samplers.rs`](../src/samplers.rs)), running on the host between model evaluations.
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

* `dpm_adaptive` picks its steps from the whole batch, as in ComfyUI, unless the batch doesn't fit in VRAM at once: then it
  picks them per part, so such batches can differ.
* `dpm_fast` and `dpm_adaptive` stop at the last non-zero sigma, as in ComfyUI, so they can leave visible noise.
* Combinations ComfyUI runs badly run badly here too: `turbo` and `align_your_steps` use sigmas above 1 on these models.
* The Brownian samplers (`dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde` and their variants) generate their noise on the CPU, even
  the `_gpu` ones. An 8-step Anima image at 1024x1024 took 2.4 s with `dpmpp_2m_sde` and 2.0 s with its `_gpu` variant, against
  1.3 s with `euler`.
