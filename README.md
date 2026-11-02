# Native ComfyUI replacement for SwarmUI

A Rust and CUDA backend that speaks ComfyUI's API and replaces the ComfyUI features SwarmUI uses most, so
SwarmUI can generate without Python or ComfyUI. The goal is speed first: Krea 2 Turbo takes about 4.65 s per
1024x1024 image here, against about 9 s in ComfyUI on the same machine, and Anima Turbo samples 8 steps in
0.92 s against 1.59 s.

**Status:** two model families really generate, text-to-image only: Krea 2 Turbo and Anima. The server accepts far more of
SwarmUI's features than that, but they only reach a mock engine so far. The
[project board](https://claude.ai/artifact/ARcm3yAXPwmZ5ZUHHhdNX9) lists everything that works and everything
still to do, in priority order. Run it today beside a normal ComfyUI backend.

* `swarm backend/` is the server: the ComfyUI protocol, the node graph, and the `Engine` interface. Start it with
  [`start-krea2.bat`](swarm%20backend/start-krea2.bat) and see [its README](swarm%20backend/README.md).
* The repository root is the real engine: the Krea 2 and Anima pipelines described below. They also run alone
  from the command line, and Krea 2 has a small web UI.

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

Model paths default to the files under `G:\stable-diffusion-webui-forge\models`; override them with
`--dit`, `--te` and `--vae`. Useful flags:

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
src/                Rust: loader, offload engine, tokenizer, text encoder, DiT, VAE, CLI
tools/              ComfyUI reference dump / comparison / benchmark scripts
```

Build requirements: CUDA 13 (`CUDA_PATH`), MSVC Build Tools 2022, and the CUTLASS and
flash-attention checkouts (default `ref/comfy-kitchen/third_party/*`, override with
`KREA_CUTLASS_DIR` / `KREA_FLASH_ATTN_DIR`). The default target is `sm_120` (`KREA_CUDA_ARCH`).
The first build takes ~6 minutes (CUTLASS int8 GEMM); object files are cached by content hash.

Not implemented: CFG > 1 / negative prompts (the turbo workflow uses CFG 1), image inputs to the
text encoder, reference latents (edit mode), and LoRAs.
