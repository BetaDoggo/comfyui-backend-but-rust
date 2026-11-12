"""Compare a diffusion --qwen21 --dump file with tools/qwen21_ref.py output.

usage: python_embeded/python.exe qwen21_cmp.py <ref dump.safetensors> <ours.safetensors>

Run the reference with --no-cache so its block outputs cover the whole sequence; ours then splits them into the
prefix's rows (prefix_blockN) and the image's (blockN). The latent is compared at 512x512.
"""
import os
import sys

import torch
from safetensors.torch import load_file

ref = load_file(sys.argv[1])
ours = load_file(sys.argv[2])


def stats(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        print(f"{name:26} SIZE {a.numel()} vs {b.numel()}")
        return
    err = (a - b).norm() / b.norm().clamp_min(1e-12)
    cos = torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)
    print(f"{name:26} rel err {err:.3e}  cos {cos:.6f}  max {(a - b).abs().max():.3e}  ref rms {b.pow(2).mean().sqrt():.3e}")


same = ref["ids"].equal(ours["ids"])
print(f"ids {'same' if same else 'DIFFERENT'} ({ref['ids'].numel()} / {ours['ids'].numel()})")
if not same:
    print("  ref ", ref["ids"].tolist())
    print("  ours", ours["ids"].tolist())
if "image_slots" in ref:
    print("image_slots", ref["image_slots"].tolist(), ours.get("image_slots", torch.zeros(0)).tolist())
stats("cond", ours["cond"], ref["cond"])
stats("txt_in", ours["txt_in"], ref["txt_in"])
if "temb" in ours:
    stats("temb", ours["temb"], ref["temb"][0])
p = ours["cond"].shape[0]
for b in (0, 1, 31):
    name = f"block{b}"
    if name in ref and name in ours:
        r = ref[name].float().reshape(-1, 4096)
        p = r.shape[0] - ours[name].numel() // 4096
        if f"prefix_{name}" in ours:
            stats(f"prefix_{name} ({p} rows)", ours[f"prefix_{name}"], r[:p])
        stats(name, ours[name], r[p:])
if "proj_out" in ours:
    stats("dit_out", ours["proj_out"].float().reshape(-1, 64), ref["dit_out"].float()[0].flatten(1).T)
# ComfyUI's samples come out after process_latent_out; ours stay at the sampler's scale
sys.path.insert(0, os.path.join(os.environ.get("SWARM_DIR") or sys.exit("set SWARM_DIR to the SwarmUI folder"), "dlbackend", "comfy", "ComfyUI"))
from comfy.latent_formats import QwenImage21
stats("samples", QwenImage21().process_out(ours["samples"].float().reshape(ref["samples"].shape)), ref["samples"])
