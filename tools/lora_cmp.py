"""Compare baked weights and the image with tools/lora_ref.py output.

usage: python lora_cmp.py <ref_dir> <ours_dir> [ours.png]

ours_dir is $KREA_LORA_DUMP of a krea2 run: create an empty `<stem>.want` there for each module without `diffusion_model.`.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

ref = load_file(os.path.join(sys.argv[1], "ref.safetensors"))
for key in sorted(k for k in ref if k.endswith(".q")):
    m = key[:-2]
    stem = m[len("diffusion_model."):]
    q = ref[key]
    n, k = q.shape
    ours_q = torch.from_numpy(np.fromfile(os.path.join(sys.argv[2], f"{stem}.q.bin"), dtype=np.int8)).reshape(n, k)
    ours_s = torch.from_numpy(np.fromfile(os.path.join(sys.argv[2], f"{stem}.scale.bin"), dtype=np.float32))
    raw = np.fromfile(os.path.join(sys.argv[2], f"{stem}.float.bin"), dtype=np.uint8)
    ours_f = torch.frombuffer(bytearray(raw.tobytes()), dtype=torch.bfloat16).reshape(n, k).float()
    rf = ref[f"{m}.float"]
    rs = ref[f"{m}.scale"].flatten().float()
    df = (rf - ours_f).abs()
    print(f"{stem}: {n}x{k}")
    print(f"  float: {(df != 0).float().mean() * 100:.3f}% differ, rel err {(rf - ours_f).norm() / rf.norm():.3e}")
    print(f"  scale: {(rs != ours_s).float().mean() * 100:.3f}% differ, max rel {((rs - ours_s).abs() / rs).max():.3e}")
    dq = (q.int() - ours_q.int()).abs()
    print(f"  q: {(dq != 0).float().mean() * 100:.4f}% differ, max {dq.max()}")

if len(sys.argv) > 3:
    a = np.asarray(Image.open(os.path.join(sys.argv[1], "ref.png")).convert("RGB")).astype(np.float64)
    b = np.asarray(Image.open(sys.argv[3]).convert("RGB")).astype(np.float64)
    mse = ((a - b) ** 2).mean()
    print(f"image: PSNR {10 * np.log10(255 ** 2 / mse) if mse else float('inf'):.2f} dB, mean abs {np.abs(a - b).mean():.3f}")
