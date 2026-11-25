# Nemotron-Labs-Image for SwarmUI

A SwarmUI extension that generates with NVIDIA's [Nemotron-Labs-Image 30B-A3B](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B)
on the **comfy-rust** backend. The model isn't diffusion: a language model writes the image as visual tokens, and its
own decoder turns them into pixels. comfy-rust runs all of that in one node, `NemotronImageGenerate`.

When the selected model is a Nemotron-Labs-Image file, the extension replaces SwarmUI's whole workflow (model loader,
prompt encoding, sampler, VAE decode) with that node and the save. It runs only on comfy-rust: ComfyUI has no node for
this model.

## Install

1. Copy this folder into `SwarmUI/src/Extensions/`, then restart SwarmUI. It rebuilds with the extension.

   SwarmUI names each compiled extension after its folder's git commit and reuses it. So after changing the C# of an
   extension that isn't its own git repository, delete `SwarmUI/src/bin/extensions/SwarmExtensionnemotron-extension`
   before restarting.
2. Put `Nemotron-Labs-Image-30B-A3B-nvfp4.safetensors` in `diffusion_models/` (or `Stable-Diffusion/`). It is the
   single file `tools/nemotron_image_convert.py` writes; a converted copy is at
   [HDiffusion/Nemotron-Labs-Image-30B-A3B-NVFP4](https://huggingface.co/HDiffusion/Nemotron-Labs-Image-30B-A3B-NVFP4).
   Both are under NVIDIA's non-commercial license.
3. Use comfy-rust built with the native engine. It needs about 17 GB of free RAM for the model's experts.

The extension recognises the file by the `nemotron_image` entry in its metadata, and lists it as
**Nemotron-Labs-Image**. A file SwarmUI read before the extension was installed keeps the class it was given then. Fix
that by setting its architecture to **Nemotron-Labs-Image** in the model's metadata editor, or by resetting its
metadata.

## Use

Pick the model and generate. These settings reach the model:

* **Prompt**, sent as the model card's chat template builds it.
* **Width** and **Height**, multiples of 16. The model card's sizes are 1024x1024, 1184x880, 1248x832 and 880x1184.
* **Seed**, and **Images**, each with the next seed.
* **Batch Size**, 1 to 8 images sampled together. Two take about 1.4 times as long as one.
* **Nemotron Temperature**, in the Nemotron-Labs-Image group: 0.9 by default, as on the model card. 0 always takes the
  likeliest token.

Steps, CFG, sampler and scheduler don't apply, and are left out. Anything else the model can't honour is refused with
an error, not dropped:

* LoRAs and embeddings
* A negative prompt
* Prompt syntax SwarmUI leaves to the backend, such as `<break>`, `<segment>`, `<region>`, `<object>`, `<alternate>`,
  `<fromto>` and `<base>`/`<refiner>` sections
* Init images, masks and prompt images
* The refiner and its upscale, ControlNet and variation seeds
* Image to video, Remove Background, SeedVR and RelayVSR, because the workflow ends at the save

The model preload (SwarmUI's load model action) sends `NemotronImageLoad`, which loads the file without generating.

The model writes a size header before the image. If it doesn't match the size you asked for, comfy-rust refuses the
image: at 512x512 the model wrote a 1024x1024 header. On an RTX 5070 one 1024x1024 image takes about 99 s.

## Status

* The extension builds against SwarmUI.
* comfy-rust's `NemotronImageGenerate` and `NemotronImageLoad` nodes pass with the mock engine, and
  `NemotronImageGenerate` made real images over HTTP with the native engine, the same pixels as the CLI.
* It works in SwarmUI on comfy-rust, as tested by the maintainer.
