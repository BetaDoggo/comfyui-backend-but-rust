"""ComfyUI's Flux 2 VAE (MOSS Image's) on one picture: the encoded latent, and that latent decoded again.

Run with SwarmUI's ComfyUI and its python_embeded:

usage: python flux2_vae_ref.py <out.safetensors> --image a.png [--vae path] [--size W H]

The picture is cropped as ComfyUI's VAEEncode crops it, to a multiple of 16. Check it with
`diffusion --flux2-vae-check <out.safetensors>`.
"""
import argparse
import os
import sys

SWARM = os.environ.get("SWARM_DIR", r"G:\StableSwarmUI")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--image", required=True)
ap.add_argument("--vae", default=os.path.join(os.environ.get("MODELS_ROOT", ""), "VAE", "flux2-vae.safetensors"))
ap.add_argument("--size", type=int, nargs=2, default=None, help="resize the picture to W H first")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.sd
import comfy.utils

torch.set_grad_enabled(False)
im = Image.open(args.image).convert("RGB")
if args.size:
    im = im.resize(tuple(args.size), Image.LANCZOS)
pixels = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
print("vae dtype", vae.vae_dtype, file=sys.stderr)
cropped = vae.vae_encode_crop_pixels(pixels)
latent = vae.encode(pixels)  # as VAEEncode: it crops the same way
back = vae.decode(latent)
print("image", tuple(cropped.shape), "latent", tuple(latent.shape), file=sys.stderr)
save_file({
    "image0": cropped.float().contiguous(),
    "latent0": latent.float().contiguous(),
    "back0": back.float().reshape(cropped.shape).contiguous(),
}, args.out)
