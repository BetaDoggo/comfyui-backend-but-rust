# swarm backend (Rust)

A ComfyUI-API-compatible server that SwarmUI can use as a normal **"ComfyUI API By URL"** backend,
fronting a native Rust generation engine. It ships with two engines: **krea2**, which really generates Krea 2
Turbo, Anima and MOSS Image text-to-image with the Rust + CUDA pipelines in the parent repository, and the **mock engine**,
which returns placeholder images/video/audio of the correct shapes for every other model family.

```
crates/
  engine-api/     the contract: Engine trait + typed request structs (no web deps)
  comfy-server/   HTTP + websocket server that speaks ComfyUI's protocol, validates and runs graphs
  krea2-engine/   the real engine: Krea 2, Anima and MOSS Image int8 text-to-image (optional, `--features krea2`)
docs/
  INPUT_MAPPING.md   what SwarmUI sends for Krea 2 / Anima / MiniMax H3 / MOSS Image / Qwen Image 2.1, and what the engine receives
  COMPATIBILITY.md   every Swarm parameter probed live, and the node classes served through the generic custom_node route
tools/            scripts that (re)generate the node catalog, third-party schemas and COMPATIBILITY.md from live SwarmUI / ComfyUI
reference/        (gitignored) SwarmUI + ComfyUI clones used as the protocol reference
```

## Run

```bash
cargo run --release -p comfy-server -- --models-root "C:/path/to/SwarmUI/Models"
```

Flags: `--listen 127.0.0.1 --port 8188 --models-root <dir> (repeatable) --output-dir --input-dir --temp-dir --ffmpeg ffmpeg --mock-step-ms 40 --mock-native-only --advertise-todo-nodes`.
Set `RUST_LOG=debug` for per-node logs. The mock engine logs every typed request it receives at `info`.

Point `--models-root` at the **same** folder SwarmUI uses, so model names match. The server searches
both ComfyUI and SwarmUI folder names (`diffusion_models`/`unet`/`Stable-Diffusion`, `text_encoders`/`clip`,
`vae`/`VAE`, `loras`/`Lora`/`LyCORIS`, `embeddings`/`Embeddings`, …).

ffmpeg (on PATH, or `--ffmpeg`) is needed for mp4/webm/mov video output (MiniMax H3), mp3/flac/ogg audio,
and video inputs. PNG/JPEG/GIF/WAV are handled natively.

## Run Krea 2 for real

The krea2 engine needs the parent repository's build requirements (CUDA 13, MSVC Build Tools, CUTLASS and
flash-attention checkouts). The first build takes about 10 minutes, so it's behind a feature and the default
build stays mock-only.

```bash
cargo run --release -p comfy-server --features krea2 -- --engine krea2 --models-root "C:/path/to/SwarmUI/Models"
```

On Windows, [`start-krea2.bat`](start-krea2.bat) does this for the folder in `KREA2_MODELS`, and prints what to enter
in SwarmUI. Extra arguments go to the server, so `--models-root` can add a second folder.

Extra flags: `--krea2-reserve-vram 512` (MB left for the desktop and other apps) and `--krea2-max-side 2048`.

In SwarmUI, select the W8A8 checkpoint and set **CFG Scale 1** and **Steps 8**. These work: size up to
`--krea2-max-side` squared in pixels (multiples of 16), seed, Variation Seed, Batch Size, Sigma Shift, cancelling,
Model Attention Backend (`int8`, or `flash` for ComfyUI's numerics), Free VRAM. The server accepts the model from
`Stable-Diffusion/` as well as `diffusion_models/`, and a generation is pixel-identical to the `krea2` CLI for the
same seed. There are no live previews.

Every sampler and scheduler works on all three families, at any CFG with a negative prompt, along with Sampler Sigma Min,
Sigma Max and Rho. Only the `ltxv-image` scheduler is refused. See the [README](../README.md#samplers).

Krea 2 takes Prompt Images, and Enable Reference Latents with `index` and `index_timestep_zero`; see the
[README](../README.md#krea-2-reference-images).

Everything else is refused with an error in SwarmUI rather than silently ignored: `<weight>` and other per-step or per-chunk prompt syntax,
init images and refiners, ControlNet, regions, VAE tiling, Seamless Tileable, and
non-W8A8 Krea 2 files.

**Anima** runs from the same `--engine krea2` server. The checkpoint's keys pick the family, and one family is
loaded at a time, so switching took 4 s to Anima and 9 s to Krea 2 with the files cached. Select an int8 ConvRot Anima checkpoint with
Steps 8 and CFG Scale 1 for turbo models. These work: any CFG with a negative
prompt, `(word:1.2)` and `<weight>`, seed, Variation Seed, Batch Size, Sigma Shift, Model Attention Backend, cancelling and
Free VRAM. LLLite ControlNets, `<break>`, `<alternate>`, `<fromto>`, init images and
VAE tiling are refused with an error.

**MOSS Image** runs from the same server. Select an int8 ConvRot MOSS Image checkpoint; SwarmUI adds
`qwen3.5_9b_int8_convrot.safetensors` and the Flux 2 VAE itself. Tested through SwarmUI at 1024x1024 and 832x1216: `euler` and `er_sde` with
`simple`, CFG 1 and CFG 4 with a negative prompt, seed, Batch Size and Sigma Shift. Prompt Images, `<weight>`,
`<break>`, `<alternate>`, `<fromto>` and init images are refused with an error. A 20-step CFG 4 image took 11.5 s through SwarmUI,
and switching from Krea 2 or Anima took about 3 s extra.

**LoRAs** work on all three, from the LoRA parameter or `<lora:name:weight>`, stacked, at any strength. The first image
after a change bakes them into the weights (about 3 s on Krea 2, 1 s on the others, with the files cached), and sampling
stays as fast as without. Tested through SwarmUI: repeats, strength changes, stacks, removal (the base image comes back
pixel-identical), and baked blocks moving to RAM and back. LoHa, LoKr, GLoRA, OFT, DoRA, LoRA hooks and LoRAs with text
encoder keys are refused with an error. See the [README](../README.md#loras).

**Alongside ComfyUI.** Add this server next to SwarmUI's normal ComfyUI backend. In a live test with both running,
SwarmUI sent every Krea 2 job here and Anima to ComfyUI, keeping each model on the backend that had it loaded.
Nothing guarantees where the first Krea 2 job goes when both are idle. Set **Exact Backend ID** in a preset to pin
Krea 2 to this backend. The two share the GPU: krea2 re-plans DiT residency from live free VRAM for each request.

## Connect SwarmUI

Server → Backends → **ComfyUI API By URL** → Address `http://127.0.0.1:8188`.
SwarmUI reads `/object_info`, sees the Swarm core nodes (`SwarmKSampler`, `SwarmSaveImageWS`, …), and from then
on generates exactly as it would against ComfyUI with Swarm's custom nodes installed.

Nodes beyond the five target models (other model families' stock nodes, third-party packs such as IP-Adapter,
SAM2 and RIFE) are served through the engine's generic `custom_node` hook, and only for classes the engine
claims. The mock claims all of them, so SwarmUI enables every feature. `--mock-native-only` turns that off.
`--advertise-todo-nodes` lists unclaimed classes in `/object_info` anyway, for inspecting their workflows;
running them fails validation. See [COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Plugging in the real engine

[`crates/krea2-engine`](crates/krea2-engine/src/lib.rs) is a worked example. Implement
[`swarm_engine_api::Engine`](crates/engine-api/src/engine.rs). It has 11 required methods. The
optional feature operations (audio, latent ops, segmentation, upscaling, CLIP Vision, custom sampling, the generic
`custom_node` hook, warmup, free-memory) default to "unsupported" or a no-op. Then add it to `make_engine` in
[`main.rs`](crates/comfy-server/src/main.rs). Engines receive:

* **Handles** (`ModelHandle`, `TextEncoderHandle`, `VaeHandle`) wrapping your own `Opaque` data, plus the
  `patches` list the graph applied (sigma shift, LoRA, attention token weights).
* **Typed requests** such as `SampleRequest`, `TextEncodeRequest` (with the Swarm prompt syntax already parsed),
  and `EmptyLatentRequest` (with exact latent shapes per model).
* **A `ProgressSink`** for step progress, live previews, and cancellation.

Two things to handle in `sample` / `vae_decode` (details in the mapping doc):

* Qwen Image 2.1 arrives with a half-size 4-channel placeholder latent. Call `Latent::conform_empty`.
* Alpha-capable VAEs (Qwen Image 2.1) return 4-channel `ImageBatch`es. The server saves them as RGBA PNGs.

Everything ComfyUI does with pure data (patch nodes, conditioning metadata, image resizing/batching,
base64 decoding, output encoding, queue/history/websocket protocol) is already handled by the server.

[docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) walks through adding a new model family: capturing what SwarmUI
sends, the generic route versus native nodes, the engine-side work, and tests.

[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) shows, for every SwarmUI parameter, whether it is handled natively or
through `custom_node`. Each generic-route node class is tagged with what native support would take: server-only
work, a new patch/metadata field, a new `Engine` operation, or a new model architecture.

## Test

```bash
cargo test
```

`crates/comfy-server/tests/swarm_compat.rs` drives the server the way SwarmUI's `ComfyUIAPIAbstractBackend`
does (websocket handshake, `/prompt`, frame classification, `/history`), using workflows shaped like
SwarmUI's generator output for Krea 2, Anima, MiniMax H3, MOSS Image, and Qwen Image 2.1 (`tests/fixtures/`).
