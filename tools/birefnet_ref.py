"""ComfyUI's BiRefNet background removal (LoadBackgroundRemovalModel + RemoveBackground) with its intermediates dumped.

Hooks the backbone (called at 1024 and at 512), the squeeze block and the model output, and runs RemoveBackground
for the final mask, so `diffusion --birefnet-check` can compare each stage. Run with SwarmUI's ComfyUI (ref/ComfyUI
predates the background removal nodes):

usage: python_embeded/python.exe birefnet_ref.py <out.safetensors> --image image.png --model birefnet.safetensors
"""
import argparse
import os
import sys

SWARM = os.environ.get("SWARM_DIR") or sys.exit("set SWARM_DIR to the SwarmUI folder")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--image", required=True)
ap.add_argument("--model", required=True)
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.bg_removal_model as bgm
import comfy.clip_model

bg = bgm.load(args.model)
assert bg is not None, "not a BiRefNet checkpoint"
net = bg.model
print("model dtype", bg.dtype, "device", bg.load_device)

img = np.asarray(Image.open(args.image).convert("RGB")).astype(np.float32) / 255.0
image = torch.from_numpy(img)[None]
dump = {}
calls = {"bb": 0}


def nhwc(t):
    return t.detach().float().permute(0, 2, 3, 1).contiguous().cpu()


def bb_hook(_m, _i, out):
    tag = "big" if calls["bb"] == 0 else "small"
    calls["bb"] += 1
    for i, o in enumerate(out):
        dump[f"bb_{tag}_{i}"] = nhwc(o)


net.bb.register_forward_hook(bb_hook)
net.squeeze_module.register_forward_hook(lambda _m, _i, out: dump.__setitem__("squeeze", nhwc(out)))
net.register_forward_hook(lambda _m, _i, out: dump.__setitem__("logits", nhwc(out)))

mask = bg.encode_image(image)
dump["mask"] = mask.float().contiguous().cpu()
# the model's input exactly as ComfyUI builds it, to feed this engine the same pixels
pv = comfy.clip_model.clip_preprocess(image.to(bg.load_device), size=bg.image_size, mean=bg.image_mean, std=bg.image_std, crop=False)
dump["pixel_values"] = nhwc(pv)
dump["image"] = image.contiguous()
save_file({k: v.contiguous() for k, v in dump.items()}, args.out)
print({k: tuple(v.shape) for k, v in dump.items()})
print("mask min/max/mean", float(mask.min()), float(mask.max()), float(mask.mean()))
