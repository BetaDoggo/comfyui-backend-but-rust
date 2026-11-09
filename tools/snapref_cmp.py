"""Compare `diffusion --anima --snapref ... --dump` with tools/snapref_ref.py's output.

usage: python snapref_cmp.py <ref_dir> <ours.safetensors> <ours.png>

Prints the relative error of the reference stream's keys and values (first and last block), the target stream's first
and last block outputs on the first evaluation, and the image PSNR. (ComfyUI's samples are after process_latent_out.)
"""
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

ref_dir, ours_path, ours_png = sys.argv[1:4]
ref = load_file(f"{ref_dir}/ref.safetensors")
ours = load_file(ours_path)


def stats(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        print(f"{name:16} SIZE {a.numel()} vs {b.numel()}")
        return
    err = (a - b).norm() / a.norm().clamp_min(1e-12)
    print(f"{name:16} rel err {err:.3e}  max {(a - b).abs().max():.3e}  ref rms {a.pow(2).mean().sqrt():.3e}")


stats("ref latent", ref["ref_latent"], ours["ref_latent"])
# ours: [rows, k | v, 2048]; ComfyUI's: [1, rows, heads, 128] each
for block in ("block0", "last"):
    kv = ours[f"ref_kv_{block}"]
    stats(f"ref k {block}", ref[f"ref_k_{block}"], kv[:, 0])
    stats(f"ref v {block}", ref[f"ref_v_{block}"], kv[:, 1])
stats("block0", ref["block0_out"], ours["block0"])
last = max(int(k[5:]) for k in ours if k.startswith("block") and k[5:].isdigit())
stats(f"block{last}", ref["blocklast_out"], ours[f"block{last}"])
if "neg_block0" in ours and "block0_out_call1" in ref:
    stats("neg block0", ref["block0_out_call1"], ours["neg_block0"])
a = np.asarray(Image.open(f"{ref_dir}/ref.png").convert("RGB")).astype(np.float64)
b = np.asarray(Image.open(ours_png).convert("RGB")).astype(np.float64)
mse = ((a - b) ** 2).mean()
print(f"image PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB")
