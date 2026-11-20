# RelayVSR for SwarmUI

A SwarmUI extension that adds a **RelayVSR** parameter group. When it's on, the finished video (or picture) is upscaled
by 2 or 4 with Comfy-Org's [RelayVSR](https://huggingface.co/Comfy-Org/RelayVSR), as in its example workflow:

1. Every 15th frame and the last go through the keyframe model, a Wan 2.2 TI2V 5B, in one euler step at CFG 1.
2. The Flash decoder rebuilds every frame from those keyframes and the low-quality video.

The stage emits stock ComfyUI nodes:

* `UNETLoader`
* `RelayVSRConditioning`
* `KSampler`
* `VAELoader`
* `VSRDecode`

So it runs on any ComfyUI-type backend that has those nodes:

* **comfy-rust** (this repository's engine), with the int8 ConvRot files only.
* A **ComfyUI** new enough to have RelayVSR.

## Install

1. Copy this folder into `SwarmUI/src/Extensions/`, then restart SwarmUI. It rebuilds with the extension.

   SwarmUI names each compiled extension after its folder's git commit and reuses it. So after changing the C# of an
   extension that isn't its own git repository, delete `SwarmUI/src/bin/extensions/SwarmExtensionrelayvsr-extension`
   before restarting.
2. Download the files of the factor you want from [Comfy-Org/RelayVSR](https://huggingface.co/Comfy-Org/RelayVSR):

| Folder | x2 | x4 |
|---|---|---|
| `diffusion_models/` | `relayvsr_x2_lightx2v_int8_convrot.safetensors` | `relayvsr_x4_int8_convrot.safetensors` |
| `VAE/` | `relayvsr_flash_vae_x2_fp32.safetensors` | `relayvsr_flash_vae_x4_fp32.safetensors` |

   Subfolders are fine.

SwarmUI's own Wan 2.2 TI2V 5B class would claim the keyframe model, so the extension narrows that class to leave out
RelayVSR files. A file SwarmUI read before the extension was installed keeps the class it was given then. Fix that by
setting its architecture to **RelayVSR** in the model's metadata editor, or by resetting its metadata.

## Use

* **RelayVSR Model** turns the stage on. The file sets the factor.
* **RelayVSR Decoder** is the Flash decoder of the same factor. **Automatic** picks the one whose file name has the
  model's `x2` or `x4`.
* **RelayVSR Detail** scales what the decoder adds to a plain bicubic upscale of each frame. 1 is the model's full
  output, 0 is bicubic.

The stage runs after SeedVR and before the video is trimmed and saved. The upscale seed is the generation's seed plus
10. With Save Intermediate Images on, the frames before the upscale are saved too.

## Status

* The extension builds against SwarmUI.
* comfy-rust's nodes are tested with its mock engine, and they ran over HTTP with the native engine.
* The native engine has been compared with ComfyUI through its CLI, on the x2 and x4 files. Its final frames are 48 to
  50 dB from ComfyUI's, about as far as ComfyUI's own fp16 and bf16 runs are from each other.
* The stage works in SwarmUI on comfy-rust, as tested by the maintainer. It has not been tested on a ComfyUI backend.
