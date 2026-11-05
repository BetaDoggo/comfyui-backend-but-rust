# Working on this repository

Rules and context for people and coding agents. The [README](README.md) covers setup, and the
[technical overview](docs/TECHNICAL_OVERVIEW.md) covers the design.

## Rules

* **Never move, copy, convert or redownload model files.** They're read in place, and disk space is tight.
* **Test images must be safe.** Use scenery or objects with no people, such as `no humans, scenery, ...`. If a test really
  needs a person, make them clearly adult and describe modest, appropriate clothing. Never use school-themed or youthful
  characters. This covers reference dumps, benchmarks, SwarmUI tests and tool defaults. Anima can drift explicit on
  neutral prompts. Delete anything inappropriate at once and tell the maintainer.
* **Commit or push only when asked.** Don't add AI co-author or attribution lines to commits.
* **Refuse, don't ignore.** A feature the engine can't honour returns `EngineError::Unsupported`, which SwarmUI shows to
  the user. Never drop a parameter silently.
* **Don't call something working until real output has been tested.** See [Keeping the docs true](#keeping-the-docs-true).
* New code is GPL-3.0. Files in `csrc/vendor/` stay Apache-2.0.
* Match the surrounding code. Model code is written directly against each checkpoint, with no general layer abstraction,
  because that is what allows fused kernels and caching.

## Layout

| Path | What it is |
|---|---|
| `src/`, `csrc/`, `build.rs` | The `diffusion` library and CLI, and the CUDA kernels |
| `swarm backend/crates/engine-api` | The `Engine` trait and typed requests |
| `swarm backend/crates/comfy-server` | ComfyUI protocol, node graph, mock engine |
| `swarm backend/crates/native-engine` | `Engine` on top of the `diffusion` library |
| `swarm backend/swarm-extension` | comfy-rust backend: SwarmUI extension that starts the server |
| `swarm backend/pixal3d-extension` | Pixal3D tab: SwarmUI extension, meant to become its own repository |
| `tools/` | Comparisons with real ComfyUI |
| `swarm backend/tools/` | Node catalog and COMPATIBILITY.md generators |
| `ref/`, `swarm backend/reference/`, `swarm backend/scratch/` | Local and gitignored; may not exist |

`swarm backend/` is a separate Cargo workspace, so run its commands from that folder.

## Commands

Build and test the server without CUDA, on the mock engine, from `swarm backend/`:

```bash
cargo test
```

Build the CLI, and check the samplers against ComfyUI's recorded output, from the repository root. Both need the CUDA
toolkit, and neither test needs a checkpoint:

```bash
cargo build --release
```

```bash
cargo test --release
```

Build the server with the real engine, from `swarm backend/`:

```bash
cargo build --release -p comfy-server --features native
```

The kernels build for `sm_120` unless `CUDA_ARCH` is set. Windows x64 is the only supported platform. `PROFILE=1` prints
stage timings, and `RUST_LOG=debug` prints per-node server logs.

## How generation works

Patches are plain data, not closures. The server lowers each node to a typed request or a `ModelPatch`, and the engine
applies them. There are no `nn.Module`s to hook, so ComfyUI custom nodes can't run against the engine. Add a native port
instead, or a claimed class through `Engine::custom_node`.

Several results are cached, including modulation by sigma, per-prompt cross-attention k and v on Anima, and baked LoRAs.
Anything that changes what a cached value depends on must invalidate it, or the output is silently wrong.

## Comparing with ComfyUI

The scripts in `tools/` run ComfyUI's real code and compare the results. Most need SwarmUI's copy of ComfyUI and its
`python_embeded` Python, because `ref/ComfyUI` is too old for the newer models. Each script's docstring gives its usage.

* Images are close to ComfyUI's, not identical. Judge a change by tokens, hidden states and the first model evaluation
  before PSNR.
* Stochastic samplers, turbo paths and reference latents magnify rounding. Compare `int8` with `flash` attention in this
  engine first: they often differ by as much as this engine differs from ComfyUI.
* Scripts that run ComfyUI's LoRA path headless must set up `comfy_aimdo` and `ModelPatcherDynamic`, or they silently take
  the legacy path with a different seed key. `tools/lora_ref.py` shows how.
* ComfyUI's `samples` are after `process_latent_out`, so compare images or step latents instead.
* Compare warm with warm. A cold read can dominate a run.

## Gotchas

* `build.rs` runs `nvcc` with a cleared environment and a minimal `PATH`, because nvcc's own `vcvars64.bat` setup fails on
  some machines. Reuse that pattern for any other CUDA build.
* C++ exceptions don't cross the `extern "C"` kernels. They abort the process.
* SwarmUI sends `CheckpointLoaderSimple`, not `UNETLoader`, for anything in `Stable-diffusion/`, and picks it by folder
  alone. Fixtures don't cover everything, so check SwarmUI-facing behaviour against a real install.
* JSON websocket frames must begin with the exact bytes `{"type":`.
* On Windows, pages read through a mapping stay in the working set. Don't evict tensors that are re-read every step,
  because each page then costs a soft fault.
* H3 needs flash attention by default.

## Testing against SwarmUI

Only run live tests when asked. To avoid touching a real install's configuration, start a separate instance from the
SwarmUI folder:

```bash
dotnet src/bin/live_release/SwarmUI.dll --data_dir <tmp> --settings_file <tmp> --backends_file <tmp> --port 7872 --launch_mode none
```

* Start it after `comfy-server` is listening, or call `RestartBackends`.
* While another SwarmUI is running, set `ModelMetadataPerFolder: false` in its `Settings.fds`, or models lose their class.
* Write `.fds` files without a BOM: a BOM makes SwarmUI reset them. In them, `\s` is a backslash.
* Keep `AutoUpdate` false on any self-start ComfyUI backend used for testing.

## Keeping the docs true

| Document | Holds | Update it when |
|---|---|---|
| [README.md](README.md) | Setup and a short summary | Requirements or the supported families change |
| [docs/USAGE.md](docs/USAGE.md) | CLI, LoRAs, samplers | A flag or behaviour changes |
| [docs/MODELS.md](docs/MODELS.md) | Timings, parity, limits per family | Anything is measured again |
| [docs/STATUS.md](docs/STATUS.md) | What works, by priority | A feature lands |
| [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) | Design and dependencies | The design changes |
| [swarm backend/README.md](swarm%20backend/README.md) | Running the server | Flags or SwarmUI behaviour change |
| [INPUT_MAPPING.md](swarm%20backend/docs/INPUT_MAPPING.md) | What SwarmUI sends | SwarmUI or the engine API changes |
| [COMPATIBILITY.md](swarm%20backend/docs/COMPATIBILITY.md) | Generated, don't edit | Edit `tools/gen_compat_doc.py` and run it |

* In STATUS, **Working** needs real output, tested end to end. Server plus mock is **Plumbing**.
* In COMPATIBILITY, **Works** only means the server and mock handle it. Don't quote it as a feature claim.
* Write only numbers you measured, and say what wasn't tested.
* Keep maintainers' machine paths out of the repository.
