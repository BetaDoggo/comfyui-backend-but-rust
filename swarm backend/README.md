# swarm backend (Rust)

A ComfyUI-API-compatible server that SwarmUI can use as a normal **"ComfyUI API By URL"** backend,
fronting a native Rust generation engine. It ships with two engines: **native**, which really generates Krea 2
Turbo, Anima and MOSS Image images and MiniMax H3 video with the Rust + CUDA pipelines in the parent repository, and the
**mock engine**, which returns placeholder images/video/audio of the correct shapes for every other model family.

```
crates/
  engine-api/     the contract: Engine trait + typed request structs (no web deps)
  comfy-server/   HTTP + websocket server that speaks ComfyUI's protocol, validates and runs graphs
  native-engine/   the real engine (optional, `--features native`)
docs/
  INPUT_MAPPING.md   what SwarmUI sends for Krea 2 / Anima / MiniMax H3 / MOSS Image / Qwen Image 2.1, and what the engine receives
  COMPATIBILITY.md   every Swarm parameter probed live, and the node classes served through the generic custom_node route
tools/            scripts that (re)generate the node catalog, third-party schemas and COMPATIBILITY.md from live SwarmUI / ComfyUI
reference/        (gitignored, local) SwarmUI + ComfyUI clones used as the protocol reference
```

## Run

```bash
cargo run --release -p comfy-server -- --models-root "C:/path/to/SwarmUI/Models"
```

Flags: `--engine mock --listen 127.0.0.1 --port 8188 --models-root <dir> (repeatable) --output-dir --input-dir --temp-dir --ffmpeg ffmpeg --mock-step-ms 40 --mock-native-only --advertise-todo-nodes`.
Set `RUST_LOG=debug` for per-node logs. The mock engine logs every typed request it receives at `info`.

Point `--models-root` at the **same** folder SwarmUI uses, so model names match. The server searches
both ComfyUI and SwarmUI folder names (`diffusion_models`/`unet`/`Stable-Diffusion`, `text_encoders`/`clip`,
`vae`/`VAE`, `loras`/`Lora`/`LyCORIS`, `embeddings`/`Embeddings`, …).

ffmpeg (on PATH, or `--ffmpeg`) is needed for mp4/webm/mov video output (MiniMax H3), mp3/flac/ogg audio,
and video inputs. PNG/JPEG/GIF/WAV are handled natively.

## Run the native engine

The native engine needs the parent repository's build requirements (CUDA 13, MSVC Build Tools, CUTLASS and
flash-attention checkouts). The first build takes about 10 minutes, so it's behind a feature and the default
build stays mock-only.

```bash
cargo run --release -p comfy-server --features native -- --engine native --models-root "C:/path/to/SwarmUI/Models"
```

On Windows, [`start-backend.bat`](start-backend.bat) does this for the folder in `MODELS_ROOT`, and prints what to enter
in SwarmUI. Extra arguments go to the server, so `--models-root` can add a second folder.

Extra flags: `--reserve-vram 512` (MB left for the desktop and other apps), `--max-side 2048` and `--live-previews` (off by default).

The checkpoint's keys pick the family, and one family is loaded at a time. Switching took 4 s to Anima, 9 s to Krea 2 and
about 3 s to MOSS Image with the files cached.

| | Krea 2 | Anima | MOSS Image | MiniMax H3 |
|---|---|---|---|---|
| Select | The W8A8 checkpoint | An int8 ConvRot checkpoint | A fused int8 ConvRot checkpoint | The `w6a8` checkpoint and the `_comfyui` turbo LoRA |
| Turbo settings | CFG 1, Steps 8 | CFG 1, Steps 8 | Steps 8, CFG 1 with a DMD LoRA; otherwise Steps 20, CFG 4 | CFG 1, Steps 4 with the LoRA |
| Seed, Sigma Shift, cancelling, Free VRAM | Yes | Yes | Yes | Yes |
| Batch Size | Yes | Yes | Yes | Not tested |
| Variation Seed | Yes | Yes | Not tested | Yes |
| Model Attention Backend | `int8`, `flash` | `int8`, `flash` | `int8`, `flash` | Not tested |
| `<weight>` | Refused | Yes | Yes | Yes |
| `(word:1.2)` | Plain text | Yes | Plain text | Plain text |
| `<break>` | Yes, not with Prompt Images | Refused | Yes | Refused |
| Rescale CFG, Renorm CFG, CFG Zero Star, TCFG | Yes | Yes | Yes | Refused |
| Prompt Images | Yes, with `index` and `index_timestep_zero` reference latents | Not supported | Refused | Refused |

These work on every family: all samplers and schedulers except `ltxv-image`, any CFG with a negative prompt, Sampler Sigma
Min, Sigma Max and Rho, LoRAs, live previews with `--live-previews`, Conditioning Multiplier, Negative Conditioning Multiplier and Zero Negative. See [Usage](../docs/USAGE.md) and [Models](../docs/MODELS.md).

Anything else is refused with an error in SwarmUI rather than silently ignored, including `<alternate>`,
`<fromto>`, LoRA hooks, init images, refiners, ControlNet, regions, Seamless Tileable, and VAE tiling. H3 only
accepts VAE tile sizes of 256 / 64.

**LoRAs** come from the LoRA parameter or `<lora:name:weight>`, stacked, at any strength. The first image after a change
bakes them into the weights (about 3 s on Krea 2, 1 s on Anima and MOSS Image, 11 s on H3, with the files cached), and sampling
stays as fast as without. Tested through SwarmUI on the image models: repeats, strength changes, stacks, removal (the base
image comes back pixel-identical), and baked blocks moving to RAM and back.

**Alongside ComfyUI.** Add this server next to SwarmUI's normal ComfyUI backend. In a live test with both running,
SwarmUI sent every Krea 2 job here and Anima to ComfyUI, keeping each model on the backend that had it loaded.
Nothing guarantees where the first job goes when both are idle. Set **Exact Backend ID** in a preset to pin a model to
this backend. The two share the GPU: the native engine re-plans DiT residency from live free VRAM for each request.

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

[`crates/native-engine`](crates/native-engine/src/lib.rs) is a worked example. Implement
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

[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) shows, for every SwarmUI parameter, whether the server handles it natively or
through `custom_node`. That doesn't mean the native engine can generate it; [STATUS](../docs/STATUS.md) covers that. Each generic-route node class is tagged with what native support would take: server-only
work, a new patch/metadata field, a new `Engine` operation, or a new model architecture.

## Idle memory

After 5 minutes with no work the engine moves diffusion blocks to RAM, so other applications get the VRAM.
The next generation takes them back. `IDLE_RELEASE_SECS` changes the wait, and 0 turns this off.

## Test

```bash
cargo test
```

`crates/comfy-server/tests/swarm_compat.rs` drives the server the way SwarmUI's `ComfyUIAPIAbstractBackend`
does (websocket handshake, `/prompt`, frame classification, `/history`), using workflows shaped like
SwarmUI's generator output for Krea 2, Anima, MiniMax H3, MOSS Image, and Qwen Image 2.1 (`tests/fixtures/`).
