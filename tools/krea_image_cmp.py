"""Compare tools/krea_image_ref.py with a krea2 --dump. usage: python krea_image_cmp.py <ref_dir> <dump> <image.png>"""
import os
import sys

import numpy as np
from PIL import Image
from safetensors.torch import load_file

ref = load_file(os.path.join(sys.argv[1], "dump.safetensors"))
ours = load_file(sys.argv[2])


def rel(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        print(f"{name}: {a.numel()} values against {b.numel()}")
        return
    print(f"{name}: relative error {float((a - b).norm() / a.norm()):.3e}, worst {float((a - b).abs().max()):.3e}")


for name in sorted(ref):
    if name.startswith("vision") and name in ours:
        rel(name, ref[name], ours[name])
ids = ref["ids"]
print("ids", ids.tolist())
print("ours", ours["token_ids"].tolist())
rel("cond", ref["cond"], ours["cond"])
if "ref0_in" in ref:
    rel("ref0", ref["ref0_in"], ours["ref0"])
tokens = ours["dit_out_tokens"]
_, _, _, h, w = ref["dit_out"].shape
rel("dit_out", ref["dit_out"], tokens.reshape(h // 2, w // 2, 16, 2, 2).permute(2, 0, 3, 1, 4).reshape(16, h, w))
a, b = (np.asarray(Image.open(p).convert("RGB")).astype(np.float64) for p in (os.path.join(sys.argv[1], "image.png"), sys.argv[3]))
print(f"image PSNR {10 * np.log10(255 ** 2 / ((a - b) ** 2).mean()):.2f} dB")
