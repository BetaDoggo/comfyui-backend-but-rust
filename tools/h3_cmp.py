"""Compare a `diffusion --h3 --dump` file with tools/h3_ref.py's ref.safetensors.

usage: python h3_cmp.py <ref.safetensors> <ours.safetensors>
"""
import sys

import torch
from safetensors.torch import load_file

r = load_file(sys.argv[1])
o = load_file(sys.argv[2])


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


if "context" in o:
    print(f"context rel {rel(o['context'], r['context']):.3e}")
pack = torch.cat([r["samples0"].flatten(), r["samples1"].flatten()])
if "denoised0" in o and "step0_out" in r:
    d, ref = o["denoised0"].flatten(), r["step0_out"].flatten()
    n = r["samples0"].numel()
    if "step0_in" in r:  # absent with context windows, whose step 0 is the fused pack
        print("step 0 sigma", r["step0_sigma"].flatten()[:2].tolist(), "shapes", r["step0_in"].shape, r["step0_out"].shape)
    print(f"step 0 denoised video rel {rel(d[:n], ref[:n]):.3e}, audio rel {rel(d[n:], ref[n:]):.3e}")
    print(f"step 0 norms ours {d[:n].norm():.1f} / {d[n:].norm():.1f}, ref {ref[:n].norm():.1f} / {ref[n:].norm():.1f}")
for name, rk in (("ctx", "ctx"), ("b0.out_proj.in", "b0.out_proj.in"), ("b0.out", "b0.out"), ("b1.out", "b1.out"), ("b49.out", "b49.out")):
    if name in o and rk in r:
        a, b = o[name].float().flatten(), r[rk].float().flatten()
        F = 5376 if rk != "b0.out_proj.in" else 7168
        a, b = a.view(-1, F), b.view(-1, F)
        t = 24
        print(f"{name}: all {rel(a, b):.3e}, text {rel(a[:t], b[:t]):.3e}, audio {rel(a[t:t + 16], b[t:t + 16]):.3e}, video {rel(a[t + 16:], b[t + 16:]):.3e}; rows {b[:t].norm(dim=1).mean():.0f} {b[t:t + 16].norm(dim=1).mean():.0f} {b[t + 16:].norm(dim=1).mean():.0f}")
s = o["samples"].flatten()
n = r["samples0"].numel()
print(f"video latent rel {rel(s[:n], pack[:n]):.3e}, audio latent rel {rel(s[n:], pack[n:]):.3e}")
if "frames" in r and "frames" in o:
    a, b = o["frames"].float(), r["frames"].float()
    print(f"frames psnr {(-10 * torch.log10(((a - b) ** 2).mean())).item():.2f} dB")
if "wave" in r and "wave" in o:
    print(f"wave rel {rel(o['wave'], r['wave']):.3e}")
