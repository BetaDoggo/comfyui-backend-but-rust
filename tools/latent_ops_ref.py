"""Record ComfyUI's latent nodes (LatentAdd, LatentMultiply, LatentBatch, LatentUpscaleBy and
SwarmUI's SwarmLatentBlendMasked) on small random latents, through their real code, for
tests/latent_ops_match_comfy.rs. No checkpoint is needed.

Run with SwarmUI's python_embeded, from anywhere:
    python tools/latent_ops_ref.py <SwarmUI root> > tests/latent_ops_golden.json
"""
import json
import os
import sys

root = sys.argv[1]
comfy_dir = os.path.join(root, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, comfy_dir)
sys.path.insert(0, os.path.join(root, "src", "BuiltinExtensions", "ComfyUIBackend", "ExtraNodes", "SwarmComfyCommon"))
os.chdir(comfy_dir)

import torch

import nodes
import comfy_extras.nodes_post_processing  # imports nodes_latent; importing that first is circular
from comfy_extras.nodes_latent import LatentAdd, LatentBatch, LatentMultiply
from SwarmBlending import SwarmLatentBlendMasked

torch.manual_seed(0)


def rand(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def t(x):
    return {"shape": list(x.shape), "data": [round(v, 7) for v in x.flatten().tolist()]}


def result(out):
    return (out.result if hasattr(out, "result") else out)[0]["samples"]


cases = []

for method in ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]:
    for scale in [1.5, 2.0, 0.6, 1.37]:
        x = rand(2, 4, 5, 7)
        out = result(nodes.LatentUpscaleBy().upscale({"samples": x}, method, scale))
        cases.append({"op": "upscale_by", "method": method, "scale": scale, "x": t(x), "out": t(out)})

# a zero vector and a pair of opposite vectors take bislerp's edge cases
x = rand(1, 4, 3, 3)
x[0, :, 1, 1] = 0
x[0, :, 0, 1] = -x[0, :, 0, 0]
out = result(nodes.LatentUpscaleBy().upscale({"samples": x}, "bislerp", 2.0))
cases.append({"op": "upscale_by", "method": "bislerp", "scale": 2.0, "x": t(x), "out": t(out)})

for sb in [(2, 4, 6, 8), (1, 4, 6, 8), (3, 4, 6, 8), (1, 4, 5, 9), (2, 4, 9, 6)]:
    a, b = rand(2, 4, 6, 8), rand(*sb)
    cases.append({"op": "add", "a": t(a), "b": t(b), "out": t(result(LatentAdd.execute({"samples": a}, {"samples": b})))})

x = rand(2, 4, 3, 5)
cases.append({"op": "multiply", "factor": 0.3, "x": t(x), "out": t(result(LatentMultiply.execute({"samples": x}, 0.3)))})

for sb in [(1, 4, 6, 8), (3, 4, 5, 9)]:
    a, b = rand(2, 4, 6, 8), rand(*sb)
    cases.append({"op": "batch", "a": t(a), "b": t(b), "out": t(result(LatentBatch.execute({"samples": a}, {"samples": b})))})

for sb, mask_shape, factor in [((2, 4, 6, 8), (6, 8), 0.5), ((2, 4, 5, 7), (48, 64), 0.8), ((1, 4, 6, 8), (13, 9), 1.0)]:
    a, b = rand(2, 4, 6, 8), rand(*sb)
    mask = torch.rand(1, *mask_shape)
    out = result(SwarmLatentBlendMasked().blend({"samples": a}, {"samples": b}, factor, mask))
    cases.append({"op": "blend_masked", "factor": factor, "a": t(a), "b": t(b), "mask": t(mask), "out": t(out)})

json.dump(cases, sys.stdout)
