# Technical overview

This project is a native image generation backend for SwarmUI. SwarmUI thinks it is talking to ComfyUI, but the
requests are served by a Rust server, and the model math runs in Rust and CUDA with no Python or PyTorch.

ComfyUI is the reference for behaviour, not a runtime dependency: its API, model definitions and numerics are what this
project reproduces. The int8 GEMM and attention kernels come from comfy-kitchen, the library ComfyUI uses for them.

Only Windows x64 and NVIDIA GPUs are supported. `build.rs` targets `x86_64-pc-windows-msvc`, and kernels compile for
`sm_120` by default (`KREA_CUDA_ARCH` changes that).

## How the pieces fit

```
SwarmUI (unchanged, "ComfyUI API By URL" backend)
   |  HTTP and websocket, ComfyUI's protocol
   v
comfy-server            axum + tokio: routes, node graph validation, execution, queue
   |  swarm_engine_api::Engine trait
   +--> mock engine     placeholder outputs for model families with no real engine yet
   +--> krea2-engine    picks the model family, refuses what it cannot honour
           |  Rust library calls
           v
krea2 library           loader, offloading, tokenizers, models, samplers (Rust)
           |  C ABI (src/ffi.rs)
           v
krea_cuda.lib           kernels built by nvcc from csrc/
           |
           v
CUDA runtime, cuBLASLt, GPU
```

The real engine sits behind a Cargo feature (`--features krea2`). A default build of the server needs no CUDA toolkit and
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
| `clap`, `tiny_http`, `anyhow` | Command line, the small Krea 2 web UI, errors |
| `cc` (build only) | Finds MSVC for the nvcc build and the static library |

There is no Rust binding to a tensor library. CUDA, NVML and the kernels are called through `extern "C"` declarations in
[`src/cuda.rs`](../src/cuda.rs) and [`src/ffi.rs`](../src/ffi.rs).

### GPU libraries

| Library | How it arrives | Used for |
|---|---|---|
| CUDA 13 toolkit | Installed, `CUDA_PATH` | `nvcc`, the static CUDA runtime, streams, events, pinned memory |
| cuBLASLt | Linked from the toolkit | bf16 and fp32 GEMMs, such as text encoders and non-quantized layers |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | Build-time checkout, BSD-3-Clause | Int8 GEMMs with fused dequantization, and VAE 2D convolutions |
| [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | Copied into `csrc/vendor/`, Apache-2.0 | The int8 GEMM kernel and INT8 attention |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Reaches us through comfy-kitchen, Apache-2.0 | Origin of the INT8 attention, the default for the diffusion models |
| [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) | Build-time checkout, BSD-3-Clause | bf16 attention, selected with `--attention flash` |
| NVML, DXCore | NVML linked, DXCore loaded at run time | How much VRAM is really free under Windows |

`csrc/compat/` holds small stand-ins for the PyTorch headers that FlashAttention-2 includes, so it builds without
libtorch. CUTLASS and FlashAttention-2 are not in this repository. The build finds them in
`ref/comfy-kitchen/third_party/` (gitignored) or through `KREA_CUTLASS_DIR` and `KREA_FLASH_ATTN_DIR`.

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

Each model is written directly against its checkpoint, with no general layer or graph abstraction. That is what allows
operations to be fused and work to be cached. For example, Anima's cross-attention k and v are computed once per prompt,
and block modulation is cached per sigma.

## Why it is faster than ComfyUI

The gains come from running the same kernels with less around them, not from better kernels:

* **Fused operations.** Several tensor operations become one kernel, which avoids intermediate tensors and memory traffic.
* **Less repeated work.** Anything that depends only on the prompt or only on sigma is computed once.
* **LoRAs are baked once.** Weights are rewritten in bulk before sampling, and nothing is added per step.
* **Planned memory.** [`src/offload.rs`](../src/offload.rs) packs each block once in the layout the kernels read. It keeps
  as many as fit in VRAM and streams the rest from page-locked RAM on a separate copy stream, overlapping with compute.
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
at some steps. The README gives the measured PSNR for each model.

## Further reading

* [README](../README.md): usage, timings and parity results for each model
* [Swarm backend README](../swarm%20backend/README.md): running the server and what SwarmUI can use today
* [Input mapping](../swarm%20backend/docs/INPUT_MAPPING.md): what SwarmUI sends and what the engine receives
* [Adding models](../swarm%20backend/docs/ADDING_MODELS.md): how to add a model family
