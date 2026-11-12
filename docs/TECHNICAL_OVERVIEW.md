# Technical overview

This project is a native image and video generation backend for SwarmUI. SwarmUI thinks it is talking to ComfyUI, but the
requests are served by a Rust server, and the model math runs in Rust and CUDA with no Python or PyTorch.

ComfyUI is the reference for behaviour, not a runtime dependency: its API, model definitions and numerics are what this
project reproduces. The int8 GEMM and attention kernels come from comfy-kitchen, the library ComfyUI uses for them.

Only Windows x64 and NVIDIA GPUs are supported. `build.rs` targets `x86_64-pc-windows-msvc`, and kernels compile for
`sm_120` by default (`CUDA_ARCH` changes that).

## How the pieces fit

```
SwarmUI (unchanged, "ComfyUI API By URL" backend)
   |  HTTP and websocket, ComfyUI's protocol
   v
comfy-server            axum + tokio: routes, node graph validation, execution, queue
   |  swarm_engine_api::Engine trait
   +--> mock engine     placeholder outputs for model families with no real engine yet
   +--> native-engine    picks the model family, refuses what it cannot honour
           |  Rust library calls
           v
diffusion library         loader, offloading, tokenizers, models, samplers (Rust)
           |  C ABI (src/ffi.rs)
           v
gpu_kernels.lib           kernels built by nvcc from csrc/
           |
           v
CUDA runtime, cuBLASLt, GPU
```

The real engine sits behind a Cargo feature (`--features native`). A default build of the server needs no CUDA toolkit and
serves only the mock engine.

## What it is built on

### Rust

| Crate | Used for |
|---|---|
| `axum`, `tokio`, `futures-util` | The ComfyUI-compatible HTTP and websocket server |
| `serde`, `serde_json` | ComfyUI graphs and API payloads, safetensors headers, tokenizer configs |
| `memmap2` | Reading `.safetensors` files in place, read-only |
| `half` | bf16 and fp16 values on the CPU side |
| `fancy-regex`, `unicode-normalization` | The hand-written Qwen BPE tokenizer |
| `tokenizers` | The T5 tokenizer Anima uses |
| `image`, `png`, `base64` | Image encoding and decoding |
| `clap`, `anyhow` | Command line, errors |
| `cc` (build only) | Finds MSVC for the nvcc build and the static library |

There is no Rust binding to a tensor library. CUDA, NVML and the kernels are called through `extern "C"` declarations in
[`src/cuda.rs`](../src/cuda.rs) and [`src/ffi.rs`](../src/ffi.rs).

### GPU libraries

| Library | How it arrives | Used for |
|---|---|---|
| CUDA 13 toolkit | Installed, `CUDA_PATH` | `nvcc`, the static CUDA runtime, streams, events, pinned memory |
| cuBLASLt | Linked from the toolkit | bf16 and fp32 GEMMs, such as text encoders and non-quantized layers |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | Build-time checkout, BSD-3-Clause | Int8 GEMMs with fused dequantization, and VAE 2D convolutions |
| [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | Copied into `csrc/vendor/`, Apache-2.0 | The int8 GEMM kernel, INT8 attention, and H3's Sol-Attn block-sparse attention |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Reaches us through comfy-kitchen, Apache-2.0 | Origin of the INT8 attention, the default for the diffusion models |
| [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) | Build-time checkout, BSD-3-Clause | bf16 attention, selected with `--attention flash` |
| NVML, DXCore | NVML linked, DXCore loaded at run time | How much VRAM is really free under Windows |

`csrc/compat/` holds small stand-ins for the PyTorch headers that FlashAttention-2 includes, so it builds without
libtorch. CUTLASS and FlashAttention-2 are not in this repository. The build finds them in
`ref/comfy-kitchen/third_party/` (gitignored) or through `CUTLASS_DIR` and `FLASH_ATTN_DIR`.

### Our own kernels

`csrc/kernels.cu` is the code ComfyUI would express as many separate tensor operations. One kernel does several of those
steps in a single pass: normalization, modulation, the ConvRot Hadamard rotation and int8 quantization. It also covers RoPE, activations,
the delta-rule recursion for Qwen3.5, GroupNorm, the VAE helpers and the small attention kernel the text encoders use. `csrc/gemm.cu` and `csrc/attn*.cu` wrap the libraries above
behind a C ABI, and instantiate only the variants the pipelines need.

LoRAs ([`src/lora.rs`](../src/lora.rs)) follow ComfyUI's own path for quantized weights: dequantize, patch in bf16, then
requantize with stochastic rounding. Three kernels at the end of `kernels.cu` do this with comfy-kitchen's arithmetic, so
the baked weights match ComfyUI's bit for bit. Sampling then runs the normal int8 kernels.

## Models

| Family | Diffusion model | Text encoder | VAE |
|---|---|---|---|
| Krea 2 Turbo | `src/dit.rs`, int8 W8A8 | Qwen3-VL-4B, `src/te.rs` | Qwen-Image (Wan 2.1), `src/vae.rs` |
| Anima | `src/anima.rs`, int8 ConvRot | Qwen3 0.6B, `src/qwen3.rs` | Qwen-Image (Wan 2.1), `src/vae.rs` |
| MOSS Image | `src/moss.rs`, int8 ConvRot | Qwen3.5 9B, `src/qwen35.rs` | Flux 2, `src/flux2vae.rs` |
| Qwen Image 2.1 | `src/qwen21.rs`, int8 ConvRot | Qwen3-VL 8B, `src/te.rs` and `src/vision.rs` | RGBA Wan 2.2 layout, `src/vae21.rs` |
| MiniMax H3 | `src/h3.rs`, w6a8 | Qwen3-VL 32B, `src/h3te.rs` | Video and audio, `src/h3vae.rs` |
| Pixal3D | `src/trellis.rs`, int8 ConvRot | DINOv3 ViT-L and NAF, `src/dino3.rs` | Sparse convolution shape and texture decoders, `src/trellis_vae.rs` |
| SeedVR2 3B | `src/seedvr.rs`, int8 ConvRot | none: a fixed prompt stored in the checkpoint | Causal video VAE on one frame, `src/seedvr_vae.rs` |
| TripoSplat / Idol | `src/triposplat.rs`, fp16 | DINOv3 ViT-H, `src/dino3.rs`, and the Flux 2 VAE's encoder | Octree gaussian decoder, `src/triposplat.rs` |

Each model is written directly against its checkpoint, with no general layer or graph abstraction. That is what allows
operations to be fused and work to be cached. For example, Anima's cross-attention k and v are computed once per prompt,
and block modulation is cached per sigma.

The generator ([`src/pipeline/`](../src/pipeline/mod.rs)) holds one family at a time. Each family has a file with a state
struct for its diffusion model, the encoders and decoders only it uses, and every prompt cache it keeps. The same file
drops those caches when a LoRA or a text encoder changes. Loading any part of another family drops the current one. Two
parts outlive a switch: the Qwen3-VL text encoder, shared by Krea 2 and Qwen Image 2.1, and the Wan 2.1 VAE, shared by
Krea 2 and Anima. SeedVR2 isn't a family: it restores any family's picture in the same job, so it is held beside the
loaded one ([`src/pipeline/seedvr.rs`](../src/pipeline/seedvr.rs)), and the memory management moves weights between them.

## Why it is faster than ComfyUI

The gains come from running the same kernels with less around them, not from better kernels:

* **Fused operations.** Several tensor operations become one kernel, which avoids intermediate tensors and memory traffic.
* **Less repeated work.** Anything that depends only on the prompt or only on sigma is computed once.
* **LoRAs are baked once.** Weights are rewritten in bulk before sampling, and nothing is added per step.
* **Packed blocks.** [`src/offload.rs`](../src/offload.rs) packs each block once in the layout the kernels read. As many
  as fit stay in VRAM, and the rest stream from page-locked RAM on a separate copy stream, overlapping with compute.
* **No interpreter.** There is no Python or tensor object overhead between kernels.

This also sets the limit. Where GPU time is already almost all GEMM and attention, as with MOSS Image, the gain is small.

## How parity is checked

The scripts in [`tools/`](../tools) run the real ComfyUI pipeline and compare tokens, hidden states, single model
evaluations, baked LoRA weights and final images with this engine. They need Python and a ComfyUI checkout, but only for that testing.

Samplers and schedulers ([`src/samplers.rs`](../src/samplers.rs)) are checked against ComfyUI's own code on a mock model by
`tools/sampler_ref.py` and `tests/samplers_match_comfy.rs`, which needs no checkpoint.

Krea 2's image paths are checked against ComfyUI by `tools/krea_image_ref.py`, run with SwarmUI's own Python, and
`tools/krea_image_cmp.py`. Unlike the older tools it uses SwarmUI's ComfyUI copy and its text node, since the vision tower
isn't in `ref/ComfyUI`'s version and the node decides the tokens.

Noise is reimplemented in [`src/sampling.rs`](../src/sampling.rs): `mt19937` for torch's CPU generator and Philox for its
CUDA generator, so seeds give the same noise as ComfyUI. Images are close but not identical, because rounding differs
at some steps. [Models](MODELS.md) gives the measured PSNR for each one.

## System-memory offloading

[`src/offload.rs`](../src/offload.rs) packs every block once into a contiguous image in the exact device layout the
kernels read, and keeps a copy in page-locked RAM. Which blocks stay in VRAM is decided by ComfyUI's own policy by
default ([ComfyUI memory management](#comfyui-memory-management)), or by the older planner with `--legacy-offload`.

### Legacy offload

`--legacy-offload` (CLI and `comfy-server`) plans residency from estimates, statically:

1. From free VRAM minus the activation peak (largest of DiT step / VAE decode) and `--reserve-vram` (512 MB by default),
   the planner keeps blocks `0..R` resident and leaves the rest in **page-locked RAM**.
2. Streamed blocks go through a ring of VRAM slots on a dedicated copy stream. A slot is refilled
   as soon as the compute stream releases it (CUDA events), so the PCIe copy of a block overlaps the
   compute of earlier ones; the compute stream only waits if a copy is genuinely late.

For Krea 2 on this machine, 16 of 28 blocks stay resident at 1024² and 12
stream (5.2 GB/step at ~26 GB/s).
Streaming *all* 28 blocks still costs only ~5% per step, because a block's copy (~17 ms) is shorter
than its compute (~24 ms). The text encoder and text-fusion adapter stream the same way. Planning
stays conservative on purpose: oversubscribing VRAM makes Windows fall back to shared system memory,
which doubles step time. That caution is what costs it under tight memory: when the estimates leave no room for two
slots, it streams every block through one, and Krea 2 slows from about 630 to 994 ms/evaluation.

### ComfyUI memory management

The default on the CLI and `comfy-server`, because models do better under it as memory gets tight. SwarmUI's ComfyUI
runs DynamicVRAM: `ModelPatcherDynamic` over comfy-aimdo 0.6.0.
[`src/comfy_memory.rs`](../src/comfy_memory.rs) follows aimdo's `model-vbar.c`, its budget and its allocator hooks, with
one of our streamed units (a block) in place of aimdo's 32 MB page:

* Nothing is planned from estimates. Each streamed model registers its units in run order. A unit is faulted into VRAM
  on first use while the budget allows, and otherwise copied into one of two shared cast buffers before each use
  (ComfyUI's two offload streams; `--slots` sets the count).
* A model is prioritized when it starts running for a node, as `load_models_gpu` does: it moves to the top of the
  priority list and its watermark returns to the top. Units at or above the watermark stream, and the watermark only
  falls until the model is prioritized again, so a run settles instead of thrashing.
* Every device allocation (`DevBuf`) frees the budget deficit first, and a failed one frees its size and retries.
  Freeing lowers watermarks from the top, the lowest-priority model first and the running model last, so activations
  push weights out, and a text encoder that ran earlier goes before the diffusion model.
* The budget is aimdo's: live device buffers plus the request plus the headroom (256 MB, or `--reserve-vram`) against
  capacity, and, polled every two seconds, the DXCore process budget less 512 MB and NVML's free memory less 512 MB.
* Weights stay in VRAM between prompts, and the cast buffers are released after each sampler run, as ComfyUI does
  after each node. A latent batch is never split.

What differs from ComfyUI: residency moves by block, not by 32 MB page, so a block is either in VRAM or streamed;
the watermark runs over the blocks in bit-reversed order, so the streamed ones lie between resident ones and each copy
overlaps compute (Krea 2 at 512x512: 190 to 150 ms per evaluation, the same images); the
small weights outside the blocks, the image VAEs and encoded prompts are plain allocations, where ComfyUI streams those
models too; one copy stream carries both cast buffers; host copies are packed and pinned for every block at load,
without ComfyUI's RAM-pressure pin eviction; and CFG's two halves still run one after the other, where ComfyUI batches
them when they fit.

Text encoders, and H3's token refiner, run once per prompt, and are treated apart
([`WeightSet::once_per_prompt`](../src/offload.rs)):

* Unless every loaded model's weights fit in three quarters of the card together, such a model is only streamed through
  the cast buffers and takes no VRAM. Faulted in, it pushed the diffusion model's blocks out and was pushed out itself
  before the next prompt. On MOSS Image at 1024x1024 a prompt change went from 0.36 to 0.65 s of text encoding and 3.9
  to 10 s of sampling to 0.25 s and 2.2 s, and on Krea 2 from 200 to 166 ms of text encoding, with the same images byte
  for byte.
* A model read on use (H3's) keeps its streamed units in RAM from their first read, while that leaves 8 GB available.
  The units aren't copied: the file's mapping is page-locked for the device in 64 MB pieces
  (`cudaHostRegister`, read-only) and copies go from it at the speed of pinned memory, 28 GB/s here. Only what the file
  doesn't hold as the unit reads it, such as norm weights, is packed into a small pinned buffer. A unit that is mostly
  such data, or one a LoRA rewrote, keeps a pinned copy as before. With a second copy in RAM, Windows dropped the file
  from its cache and every new process read it from the disk again.
* The threads that read weights from disk ask for very low I/O priority, and a model read on use has one reader. Read at
  normal priority from the drive that also held the pagefile, a model froze the desktop until it was in. How much
  slower a load is with one low-priority reader wasn't measured.

Measured on the RTX 5070 (12 GB), Krea 2 at 1024x1024, 8 steps, warm second image, pictures pixel-identical in both
modes:

| `--reserve-vram` | legacy offload | ComfyUI memory management |
|---|---|---|
| default (512 / 256) | 16 blocks resident, 622 ms/evaluation | 18 resident, 614 ms |
| 6000 | 3 resident, 637 ms | 10 resident, 620 ms |
| 9000 | one-slot fallback, 994 ms | 2 resident and two cast buffers, 631 ms |

Anima at 1024x1024: 125 ms/evaluation against 120, all blocks resident in both, pixel-identical. MOSS Image, Qwen Image
2.1, MiniMax H3 (the lazy sets and decoders), Pixal3D and LoRAs haven't been measured in this mode; the maintainer
reports it working well through SwarmUI.

## Batches

A batch is sampled together. `run_batches` in [`src/pipeline/mod.rs`](../src/pipeline/mod.rs) gives the sampler the
images' latents one after another, and each model's step takes a batch: the int8 GEMMs and the norms run over every
image's rows at once, and attention runs with a batch dimension. RoPE runs once per image, because the table is one
image's. The cuBLAS GEMMs of the embedders and the last layer also run once per image, because cuBLAS picks its
algorithm by the row count, and a batched image would otherwise round differently from the same image alone. The
sampler's noise is drawn for the whole batch, as ComfyUI draws it, so sampling a batch in parts gives the same images.
Before sampling, `Env::batch_fit` works out how many images fit next to the fewest streamed blocks, and the batch runs in
parts of that size.

## Family notes

### Krea 2

How each stage maps to ComfyUI:

* **Text encoder** (`src/te.rs`, shared with Qwen Image 2.1): Qwen3-VL-4B language model, Krea2 template, hidden states
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
  GEMMs. Attention is comfy-kitchen's INT8 attention by default, or FlashAttention-2 (bf16) with `--attention flash`,
  reading q/k/v in place from the fused projection.
  The text-fusion adapter is timestep-independent, so it runs once per prompt instead of every step.
* **VAE** (`src/vae.rs`, shared with Anima): for one frame every causal 3D conv equals a 2D conv with the kernel's last
  temporal slice, so the decoder runs as NHWC CUTLASS implicit-GEMM 2D convolutions.

### Pixal3D

[`src/pixal3d.rs`](../src/pixal3d.rs) runs the example workflow's stages one after another and keeps only one in VRAM at
a time: DINOv3, then each of the four flow sub-models (about 1.4 GB each), the NAF map a stage needs, and the decoders.
The flow models are transformers over the occupied voxels, so the sparse part is only the token set and the 3D RoPE.
The decoders' submanifold convolutions gather each voxel's 27 neighbours, found through a GPU hash of the coordinates,
into rows for one f16 GEMM against the kernel, in chunks. The dense structure decoder is the same convolution over a full
grid. Mesh extraction, painting and simplification run on the CPU.

[`src/birefnet.rs`](../src/birefnet.rs) cuts the object out of an opaque image first. It is written for ComfyUI's BiRefNet
checkpoint: a Swin-L backbone whose window attention is one kernel per (window, head), run at 1024 and at 512, and a
decoder whose deformable convolutions gather bilinear samples into columns for one GEMM. Convolutions are im2col plus
a float32 GEMM in chunks, with BatchNorm folded into the weights when they load.

### TripoSplat and Idol

[`src/triposplat.rs`](../src/triposplat.rs) follows Comfy-Org's Idol workflow: `TripoSplatPreprocessImage` (Pillow's
premultiplied RGBA Lanczos, as `comfy.utils.lanczos` gets it), DINOv3 ViT-H in f32 with its bf16 weights rounded to f16
as ComfyUI holds them, the Flux 2 VAE's encoder, a `KSampler` over the nested latent (8192 shape tokens and a camera
token, sampled as one vector), and the octree gaussian decoder. The flow model and the decoder run in fp16 like ComfyUI,
with int8 attention in fp16 (`gpu_int8_attn_f16`), so neither needs another dtype's kernels; the fp16 element-wise ops
are in `csrc/triposplat.cu`. RePo's rotary angles come from the hidden states at every block, so the rotation and the
q/k RMS norms that follow it are one kernel. The context refiner doesn't depend on the timestep, so it runs once per cond
and uncond rather than every evaluation. The token anchors are torch's scrambled Sobol sequence, reproduced bit for bit,
and the octree's point sampling uses torch's CPU generator the same way. Assembling the gaussians from the decoder's
features, and the PLY, run on the CPU.

### MiniMax H3

Offloading is the lazy kind in [`src/offload.rs`](../src/offload.rs): nothing is read at load, and reader threads fill
up to four pinned staging buffers ahead of compute, taken only when a unit needs one and returned when the set parks.
Pages read once leave the process right after, which took a H3 run's peak working set from 52 GB to 32 GB. A unit that
turns out to be streamed is kept in pinned RAM after its next read, while 8 GB of RAM stays available, so it goes
straight to VRAM afterwards.

Weights stay in packed 6-bit form in VRAM and decode to int8 before each GEMM, which leaves room for 30 of the 50 blocks
at 512x512. Streamed blocks are spread evenly among the resident ones, so each copy overlaps compute instead of queueing
behind the last; at 5 frames that took a step from 0.33 s to 0.24 s, and copies no longer hold it up. Splitting weights
to keep them in L2 gained 2 to 3%, so it isn't used: the int8 GEMM already runs close to what this GPU can do.

LoRA-baked blocks keep a pinned host copy, about 15 GB, and go straight to VRAM: this machine copies 18 GB/s in RAM but
28 GB/s from pinned memory to VRAM, so a pass through staging cost about 200 ms a step. If pinning fails they fall back
to pageable memory and staging. The blocks leave VRAM before the video decode and come back for the next clip,
re-planned for its size. The text encoder and refiner only hold VRAM while they run. The video decoder holds 2.4 GB of
pinned RAM between clips, and takes no VRAM while sampling.

## Source layout

```
build.rs            nvcc build of csrc/ (works around vcvars by passing a clean MSVC env)
csrc/common.cuh     shared device helpers
csrc/kernels.cu     fused prologue+ConvRot+int8-quantize, QK-norm+RoPE, small attention, VAE ops
csrc/gemm.cu        cuBLASLt bf16/fp32 GEMM, CUTLASS conv2d, int8 GEMM wrappers
csrc/attn.cu        FlashAttention-2 forward wrapper (only the needed variants instantiated), for --attention flash
csrc/attn_int8.cu   INT8 attention wrapper, the default
csrc/h3.cu          MiniMax H3 kernels
csrc/qwen21.cu      Qwen Image 2.1: VAE shortcuts, prefix attention for text and reference tokens
csrc/seedvr.cu      SeedVR2: patch embedding input, window attention reordering, VAE pixel shuffle
csrc/trellis.cu     Pixal3D: per-head QK norm, voxel hashing and sparse-conv gathers, NAF attention, feature sampling
csrc/triposplat.cu  TripoSplat: fp16 gated adds, RePo rotary + q/k RMS norm, SiLU gating
csrc/birefnet.cu    BiRefNet: im2col, deformable-conv columns, Swin window gather and attention, bilinear resize
csrc/wddm.cu        VRAM queries under Windows
csrc/sparse_attn.cu block-sparse attention over the vendored Sol-Attn kernels, and the SLA top-k threshold
csrc/vendor/        comfy-kitchen CUTLASS int8 GEMM, Sage INT8 attention and Sol-Attn (Apache-2.0)
src/                Rust: loader, offloading, tokenizers, each family's models, samplers, LoRA baking, CLI
tools/              ComfyUI reference dump / comparison / benchmark scripts
```

The CUTLASS and flash-attention checkouts default to `ref/comfy-kitchen/third_party/*` (override with
`CUTLASS_DIR` / `FLASH_ATTN_DIR`). Object files are cached by content hash.

## Further reading

* [Models](MODELS.md): timings and parity results for each model
* [Usage](USAGE.md): the command line, LoRAs and samplers
* [Swarm backend README](../swarm%20backend/README.md): running the server and what SwarmUI can use today
* [Input mapping](../swarm%20backend/docs/INPUT_MAPPING.md): what SwarmUI sends and what the engine receives
* [Adding models](../swarm%20backend/docs/ADDING_MODELS.md): how to add a model family
