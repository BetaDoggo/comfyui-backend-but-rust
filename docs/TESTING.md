# Testing against ComfyUI and SwarmUI

How to set up a machine to check this engine against real ComfyUI and SwarmUI, from nothing. The
[README](../README.md#quick-start) covers building and the models folder. This page covers everything else.

The checks, cheapest first:

| Check | Needs | Command |
|---|---|---|
| Server on the mock engine | Rust | `cargo test -p comfy-server -p swarm-engine-api` |
| Everything above, plus samplers, latent ops, CFG tweaks, prompts and sparse attention against ComfyUI's recorded output, and the native engine's request mapping | The CUDA build and a GPU | `cargo test --release` |
| One model against ComfyUI, tensor by tensor | Checkpoints and a ComfyUI to run | A `tools/*_ref.py` script, then `diffusion --dump`, then a `*_cmp.py` script |
| SwarmUI end to end | A SwarmUI install | A separate SwarmUI instance pointed at `comfy-server` |

Every test image must be safe: scenery or objects with no people. See [AGENTS.md](../AGENTS.md#rules).

## What to install

| Piece | Where | Used for |
|---|---|---|
| This repository, built | See the [README](../README.md#quick-start) | Everything |
| `ref/comfy-kitchen` | At the pinned commit, see the [README](../README.md#quick-start) | The CUDA build (CUTLASS and FlashAttention-2) |
| A models folder in SwarmUI's layout | Anywhere, named by `MODELS_ROOT` | The CLI, the server and the reference scripts |
| SwarmUI with its bundled ComfyUI | Its own folder, named by `SWARM_DIR` | Most reference scripts, and the live tests |
| `ref/ComfyUI` with `ref/venv` | Inside this repository | The older reference scripts |

`ref/`, `refout/` and `prof/` are gitignored, so nothing there is committed.

### SwarmUI

Install [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) normally, with its ComfyUI backend, and launch it once
so it builds. Its installer puts the pieces this page uses at:

| Path, inside the SwarmUI folder | What it is |
|---|---|
| `dlbackend/comfy/ComfyUI` | SwarmUI's copy of ComfyUI, with SwarmUI's own nodes |
| `dlbackend/comfy/python_embeded/python.exe` | The Python that runs it, with torch, `comfy_kitchen` and `comfy_aimdo` installed |
| `src/bin/live_release/SwarmUI.dll` | The built server, for starting a separate instance |
| `Data/Settings.fds`, `Data/Backends.fds` | The install's own configuration. Don't edit these for tests |

Point its model root at the same folder as `MODELS_ROOT`, so model names match between SwarmUI, ComfyUI and this
engine.

### `ref/ComfyUI`

Some older scripts load a plain ComfyUI checkout from `ref/ComfyUI`, run with a virtual environment in `ref/venv`:

```bash
git clone https://github.com/comfyanonymous/ComfyUI ref/ComfyUI
```

```bash
python -m venv ref/venv
```

Install a CUDA build of PyTorch into `ref/venv`, then `ref/ComfyUI/requirements.txt`. This checkout falls behind: it
lacks the newer models and SwarmUI's nodes, which is why newer scripts use SwarmUI's copy instead.

## Environment variables

| Variable | Meaning |
|---|---|
| `MODELS_ROOT` | The models folder. The CLI, `start-backend.bat` and most reference scripts read it |
| `SWARM_DIR` | The SwarmUI folder. Scripts that use SwarmUI's ComfyUI read it, and stop with a message when it isn't set |
| `COMFY_DIR` | A different ComfyUI, for a model that only an unmerged ComfyUI branch has. `tools/h3_image_ref.py` reads it |
| `CUDA_ARCH` | The GPU architecture to build for, `120` by default |
| `PROFILE=1`, `RUST_LOG=debug` | Stage timings, and per-node server logs |

Set them once for your Windows user, so every new terminal has them:

```bash
setx MODELS_ROOT "<models folder>"
```

```bash
setx SWARM_DIR "<SwarmUI folder>"
```

Some scripts take the SwarmUI folder as their first argument instead of `SWARM_DIR`.

## Which ComfyUI each script uses

Each script's docstring gives its usage. Run the ones on SwarmUI's copy with SwarmUI's `python_embeded/python.exe`, and
the ones on `ref/ComfyUI` with `ref/venv/Scripts/python.exe`.

| ComfyUI | Scripts |
|---|---|
| SwarmUI's, through `SWARM_DIR` | `birefnet_ref`, `flux2_vae_ref`, `h3_ref`, `h3_image_ref`, `krea_image_ref`, `lora_ref`, `pih3_ref`, `pixal3d_ref`, `qwen21_ref`, `relayvsr_ref`, `seedvr_ref`, `sensenova_ref`, `snapref_ref`, `triposplat_ref` |
| SwarmUI's, with the SwarmUI folder as the first argument | `cfg_ref`, `latent_ops_ref`, `partial_ref`, `swarm_prompt_ref` |
| `ref/ComfyUI` | `anima_ref`, `moss_ref`, `ref_dump` (Krea 2), `sampler_ref`, `sd_ref` (or `COMFY_DIR`), `te_hidden`, `te_l0`, `chk_dtypes`, `chk_te` |
| None: the model's own code from its Hugging Face repository | `nemotron_image_ref` (Nemotron-Labs-Image; run with SwarmUI's Python, which has torch and transformers) |
| None: NVIDIA's `nvvfx` package | `nvidia_vsr_ref` (RTX Video Super Resolution; run with a Python that has torch and `nvidia-vfx`, such as SwarmUI's after `pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com`) |
| None: they compare, inspect or write files | Every `*_cmp.py` except `qwen21_cmp`, `compare.py`, `psnr.py`, `glb_preview.py`, `st_header.py`, `h3_refiner_check`, `lora_formats` |

`pih3_ref` also needs [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow), cloned to `ref/ComfyUI-piFlow` or
wherever `PIFLOW_DIR` points; it imports the nodes itself, so they needn't be installed in SwarmUI's ComfyUI. Compare its
output with `tools/h3_cmp.py`. `h3_ref --ref-image` and `--ref-video` add a reference through `MiniMaxH3ReferenceToVideo`, and need `ffmpeg` and `ffprobe` on `PATH` for a video. `qwen21_cmp` imports SwarmUI's ComfyUI. `sparse_attn_ref` and `te_q` need `comfy_kitchen`, which SwarmUI's Python has. `sol3_ref` needs a CUDA PyTorch with Triton (`triton-windows` on Windows) and [sol3_attn](https://github.com/FofGofx/sol3) at the commit its docstring names, which SwarmUI's Python doesn't have.
`comfy_bench.py` benchmarks a running ComfyUI over HTTP.

## Golden files

The tests under `tests/` compare against JSON files recorded from ComfyUI's own code, on mock models, so they need no
checkpoint. Regenerate one when ComfyUI changes the code it records:

| Golden file | Recorded by |
|---|---|
| `tests/sampler_golden.json` | `tools/sampler_ref.py` |
| `tests/partial_golden.json` | `tools/partial_ref.py` |
| `tests/cfg_golden.json` | `tools/cfg_ref.py` |
| `tests/latent_ops_golden.json` | `tools/latent_ops_ref.py` |
| `tests/sparse_golden.json` | `tools/sparse_attn_ref.py` |
| `tests/sol3_golden.json` | `tools/sol3_ref.py`; `tests/sol3_attn_match.rs` skips until it is recorded |
| `tests/swarm_tokens_golden.json` | `tools/swarm_prompt_ref.py` |

Each script writes a `source` key into its file through `tools/golden_source.py`: the script, and the commit of the
ComfyUI (and SwarmUI) checkout or the version of the package it ran, with `+dirty` for uncommitted changes. Compare it
with the previous file's to see what moved the numbers. Files recorded before that have no `source` key, so what they
came from wasn't recorded. Write the JSON from Git Bash or with the script's own output path: PowerShell 5.1's `>`
writes UTF-16 or a byte-order mark, and the tests can't read either.

## Comparing one model with ComfyUI

Each model family has a script that runs ComfyUI's pipeline headless and dumps its tensors, and the CLI's `--dump` writes
the same tensors from this engine. A compare script then reports where they part. For MOSS Image:

```bash
ref/venv/Scripts/python.exe tools/moss_ref.py refout/moss "a stone lighthouse on a sea cliff at dawn, no people" --seed 42 --steps 20 --cfg 4
```

```bash
target/release/diffusion.exe --moss -p "a stone lighthouse on a sea cliff at dawn, no people" --seed 42 --steps 20 --cfg 4 --dump refout/moss/ours.safetensors
```

For a bf16 checkpoint, pass it as `--dit` to both, and add `--no-quant` to the engine's run to compare the bf16 path.

```bash
python tools/moss_cmp.py refout/moss/ref.safetensors refout/moss/ours.safetensors
```

The other families follow the same pattern with their own `*_ref.py` and `*_cmp.py`; check each docstring for the
arguments. SD 1.x and SDXL compare in the reference script itself: run the engine first with `--dump`, then pass its dump
and picture to `tools/sd_ref.py` with `--compare`, using the same prompt, seed, size and sampler:

```bash
target/release/diffusion.exe --sdxl --no-quant --attention flash --seed 1 --scheduler normal -p "no humans, scenery, a lighthouse on a rocky coast at sunset" --negative "blurry, low quality" -o refout/sd/ours.png --dump refout/sd/ours.safetensors
```

```bash
ref/venv/Scripts/python.exe tools/sd_ref.py refout/sd/comfy "$MODELS_ROOT/Stable-diffusion/sd_xl_base_1.0.safetensors" "no humans, scenery, a lighthouse on a rocky coast at sunset" --seed 1 --compare refout/sd/ours.safetensors refout/sd/ours.png
``` [AGENTS.md](../AGENTS.md#comparing-with-comfyui) has the rules for judging the result: tokens and hidden
states before PSNR, warm runs against warm runs, and `int8` against `flash` attention first.

## Testing in SwarmUI

Test against a separate SwarmUI instance, so the real install's settings and backends are never touched. Only run live
tests when asked.

1. **Start `comfy-server`** with the native engine and the same models folder as SwarmUI:

   ```bash
   cargo run --release -p comfy-server --features native -- --engine native --port 8188 --models-root "<models folder>"
   ```

2. **Write the instance's configuration** to a temporary folder. Write the `.fds` files without a BOM: a BOM makes
   SwarmUI reset them to defaults. In `.fds` values, `\s` stands for a backslash, so `C:\models` is written `C:\smodels`.
   `Settings.fds`:

   ```yaml
   Paths:
   	ModelRoot: <models folder, with \s for each backslash>
   Metadata:
   	ModelMetadataPerFolder: false
   IsInstalled: true
   ```

   `ModelMetadataPerFolder: false` is needed while the main SwarmUI is running, because the second instance can't open
   the shared model metadata database, and models then lose their class. `Backends.fds`, with tabs for indentation:

   ```yaml
   0:
   	type: comfyui_api
   	title: comfy-rust
   	enabled: true
   	settings:
   		Address: http://127.0.0.1:8188
   ```

3. **Start the instance** from the SwarmUI folder, after `comfy-server` is listening (or call `RestartBackends` later):

   ```bash
   dotnet src/bin/live_release/SwarmUI.dll --data_dir <tmp> --settings_file <tmp>/Settings.fds --backends_file <tmp>/Backends.fds --port 7872 --launch_mode none
   ```

4. **Generate** in the browser at `http://127.0.0.1:7872`, or through SwarmUI's API: `POST /API/GetNewSession` returns a
   `session_id`, `POST /API/GenerateText2Image` generates, and `POST /API/ComfyGetGeneratedWorkflow` returns the ComfyUI
   graph SwarmUI would send, without running it. Each takes JSON with the `session_id` and SwarmUI's parameter names.

Things the mock tests can't show, which only a live SwarmUI does:

* SwarmUI sends `CheckpointLoaderSimple`, not `UNETLoader`, for anything in `Stable-diffusion/`, and picks it by folder
  alone.
* Which nodes and inputs each SwarmUI parameter produces. `swarm backend/tools/probe_swarm_params.py` records that for
  [COMPATIBILITY.md](../swarm%20backend/docs/COMPATIBILITY.md#regenerating). It expects SwarmUI on port 7801.
* Keep `AutoUpdate` false on any self-starting ComfyUI backend used for testing, so a test doesn't update the install.

To capture what SwarmUI sends for a new model, see [Adding a new model](../swarm%20backend/docs/ADDING_MODELS.md).
