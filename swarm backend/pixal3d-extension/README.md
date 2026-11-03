# Pixal3D for SwarmUI

A SwarmUI extension that adds a **Pixal3D** tab: drop in a photo of an object, get a vertex-coloured 3D mesh (GLB) to
orbit in the page and download. A photo without a transparent background is cut out first with BiRefNet.

It runs Comfy-Org's Pixal3D example workflow as one ComfyUI node, `Pixal3DImageTo3D`, on one of SwarmUI's ComfyUI-type
backends:

* **comfy-rust** has the node natively (this repository's Rust engine).
* A **ComfyUI self-start backend** gets it from this extension's `comfy_nodes` folder; restart the backend after
  installing. It uses ComfyUI's own Pixal3D nodes, so ComfyUI must be new enough to have them.

## Install

1. Copy (or clone) this folder into `SwarmUI/src/Extensions/`, then restart SwarmUI. It rebuilds with the extension.
   SwarmUI names each compiled extension after its folder's git commit and reuses it, so after changing the C# of an
   extension that isn't its own git repository, delete `SwarmUI/src/bin/extensions/SwarmExtensionpixal3d-extension`
   before restarting. (The tab's JavaScript and HTML are read from the folder and need no rebuild.)
2. Download the Pixal3D files from [Comfy-Org/Pixal3D](https://huggingface.co/Comfy-Org/Pixal3D) into your model folders:

| Folder | File |
|---|---|
| `diffusion_models/` | `pixal3d_int8_convrot.safetensors` (comfy-rust runs only this one), or `pixal3d_bf16.safetensors` on ComfyUI |
| `clip_vision/` | `dino_v3_L_naf_fp32.safetensors` |
| `VAE/` | `trellis_2_shape_vae_bf16.safetensors`, `trellis_2_texture_vae_bf16.safetensors` |
| `background_removal/` | `birefnet.safetensors` from [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet), only for images without transparency |

Subfolders are fine; the tab picks the files by name and lists them all under Advanced.

## Use

* Drop an image on the tab, click it to choose a file, or paste one with Ctrl+V while the tab is showing.
* A **transparent background** is the object's mask. An image without one has its background removed first, by
  ComfyUI's own `RemoveBackground` node with BiRefNet; **Remove background** is ticked for it on its own, and can be
  ticked for an image that has transparency too (its alpha is then ignored). The tab says so when no BiRefNet model is
  installed.
* **Field of view** is the photo's horizontal field of view. 49 degrees suits most photos; ComfyUI's example estimates it
  per image with MoGe, which this tab doesn't do.
* **Resolution** is the voxel grid of the refined shape. 1024 takes about 40 s on an RTX 5070 with comfy-rust, and 1536
  about 110 s.
* **Max triangles** simplifies the result (the full mesh at 1536 is about 16 million triangles); 0 keeps it all.
* Models are saved to your output folder under `pixal3d/<date>/`.

## Files

| Path | What it is |
|---|---|
| `Pixal3DExtension.cs` | Registers the tab's assets and API calls, and the ComfyUI node folder |
| `Tabs/Text2Image/Pixal3D.html` | The tab |
| `Assets/pixal3d.js`, `Assets/pixal3d.css` | Its logic, including a small WebGL2 GLB viewer, and styles |
| `comfy_nodes/Pixal3D/` | The node for ComfyUI |

API calls: `Pixal3DListModels` reads the node's file lists, and the background removal models, from the first backend
that has the node, and the websocket call `Pixal3DGenerateWS` claims a ComfyUI-type backend, runs the node (after
`LoadBackgroundRemovalModel` and `RemoveBackground` when removing the background), streams progress and returns the
saved GLB's path.

## License

GPL-3.0, like the repository it comes from.
