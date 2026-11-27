"""NVIDIA RTX Video Super Resolution as the ComfyUI `RTXVideoSuperResolution` node runs it, through NVIDIA's `nvvfx`
package, to compare with this engine's output (`diffusion --super-res`, or comfy-server's node).

The node's steps: an optional deblur pass at the image's size, then the upscale to `int(side * scale)` rounded to a
multiple of 8. `--model-upscale` is instead what this engine does for `ImageUpscaleWithModel` on `nvngx_vsr.dll`: 4x
at ULTRA. Needs a Python with torch and `nvidia-vfx` (`pip install nvidia-vfx --extra-index-url
https://pypi.nvidia.com`), such as SwarmUI's `python_embeded`. The engine's output should match byte for byte.

usage: python nvidia_vsr_ref.py image.png out.png [--scale 2] [--quality ULTRA] [--deblur OFF] [--model-upscale]
                                [--compare engine.png]
"""
import argparse

import numpy as np
import nvvfx
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--quality", default="ULTRA", choices=["LOW", "MEDIUM", "HIGH", "ULTRA"])
ap.add_argument("--deblur", default="OFF", choices=["OFF", "LOW", "MEDIUM", "HIGH", "ULTRA"])
ap.add_argument("--model-upscale", action="store_true", help="4x ULTRA, as ImageUpscaleWithModel on nvngx_vsr.dll")
ap.add_argument("--compare", help="an engine PNG to compare with the reference")
args = ap.parse_args()


def effect(x, quality, width, height):
    """One pass over a (3, H, W) float CUDA image, as the node's _run_effect does per frame."""
    with nvvfx.VideoSuperRes(quality) as sr:
        sr.output_width, sr.output_height = width, height
        sr.load()
        return torch.from_dlpack(sr.run(x).image).clone()


src = np.asarray(Image.open(args.image).convert("RGB"))
h, w, _ = src.shape
x = torch.from_numpy(src.astype(np.float32) / 255.0).cuda().permute(2, 0, 1).contiguous()
Q = nvvfx.QualityLevel
if args.model_upscale:
    out = effect(x, Q.ULTRA, w * 4, h * 4)
else:
    ow, oh = (max(8, round(int(s * args.scale) / 8) * 8) for s in (w, h))
    if args.deblur != "OFF":
        x = effect(x, Q[f"DEBLUR_{args.deblur}"], w, h)
    out = effect(x, Q[args.quality], ow, oh)
ref = np.round(out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(ref).save(args.out)
print(f"{w}x{h} -> {ref.shape[1]}x{ref.shape[0]}, saved {args.out}")
if args.compare:
    got = np.asarray(Image.open(args.compare).convert("RGB"))
    if got.shape != ref.shape:
        raise SystemExit(f"engine image is {got.shape[1]}x{got.shape[0]}, the reference {ref.shape[1]}x{ref.shape[0]}")
    d = np.abs(got.astype(int) - ref.astype(int))
    print(f"engine vs nvvfx: max difference {d.max()} levels, {np.count_nonzero(d)} of {d.size} values differ")
