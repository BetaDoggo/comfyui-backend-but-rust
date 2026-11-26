"""Compare a `diffusion --h3 --dump` file with tools/h3_ref.py's or tools/pih3_ref.py's ref.safetensors.

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
    if o["context"].shape[-2] != r["context"].shape[-2]:
        print(f"context lengths differ: ours {tuple(o['context'].shape)}, ref {tuple(r['context'].shape)}")
    else:
        print(f"context rel {rel(o['context'], r['context']):.3e}")
if "ref0" in o and "ref0" in r:  # the first reference latent
    print(f"ref0 latent rel {rel(o['ref0'].flatten(), r['ref0'].flatten()):.3e}, ref shape {tuple(r['ref0'].shape)}")
pack = torch.cat([r["samples0"].flatten(), r["samples1"].flatten()])
if "denoised0" in o and "step0_out" in r:
    d, ref = o["denoised0"].flatten(), r["step0_out"].flatten()
    n = r["samples0"].numel()
    if "step0_in" in r:  # absent with context windows, whose step 0 is the fused pack
        print("step 0 sigma", r["step0_sigma"].flatten()[:2].tolist(), "shapes", r["step0_in"].shape, r["step0_out"].shape)
    print(f"step 0 denoised video rel {rel(d[:n], ref[:n]):.3e}, audio rel {rel(d[n:], ref[n:]):.3e}")
    print(f"step 0 norms ours {d[:n].norm():.1f} / {d[n:].norm():.1f}, ref {ref[:n].norm():.1f} / {ref[n:].norm():.1f}")
    if "step0_in" in r:
        # Near sigma 1 the audio's denoised pack is the noise minus a velocity of almost the same size, so its relative
        # error is several times the network's own. Undo denoised = x - sigma * v, and on audio ModelSamplingAV's carry
        # (H3's shifts 12 and 3), to compare the network's outputs. Both runs start from the reference's input.
        x, sv = r["step0_in"].flatten(), r["step0_sigma"].flatten()[0].item()
        base = sv / (12.0 + sv * (1.0 - 12.0))
        sa, s = 3.0 * base / (1.0 + 2.0 * base), 12.0 / 3.0

        def net(den):
            v = (x - den) / sv
            return v[:n], (v[n:] - (1.0 - s) * (sa / sv) * x[n:]) / (1.0 + (s - 1.0) * sa)

        (ov, oa), (rv, ra) = net(d), net(ref)
        print(f"step 0 network output video rel {rel(ov, rv):.3e}, audio rel {rel(oa, ra):.3e}")
for name, rk in [("ctx", "ctx"), ("b0.out_proj.in", "b0.out_proj.in")] + [(f"b{i}.out", f"b{i}.out") for i in range(50)]:
    if name in o and rk in r:
        a, b = o[name].float().flatten(), r[rk].float().flatten()
        F = 5376 if rk != "b0.out_proj.in" else 7168
        a, b = a.view(-1, F), b.view(-1, F)
        t = 24
        print(f"{name}: all {rel(a, b):.3e}, text {rel(a[:t], b[:t]):.3e}, audio {rel(a[t:t + 16], b[t:t + 16]):.3e}, video {rel(a[t + 16:], b[t + 16:]):.3e}; rows {b[:t].norm(dim=1).mean():.0f} {b[t:t + 16].norm(dim=1).mean():.0f} {b[t + 16:].norm(dim=1).mean():.0f}")
if "dwd_features" in r and "dwd_features" in o:  # pih3_ref.py: the DWD head's normed features at the last step
    print(f"dwd features rel {rel(o['dwd_features'].flatten(), r['dwd_features'].flatten()):.3e}")
s = o["samples"].flatten()
n = r["samples0"].numel()
print(f"video latent rel {rel(s[:n], pack[:n]):.3e}, audio latent rel {rel(s[n:], pack[n:]):.3e}")
if "frames" in r and "frames" in o:
    a, b = o["frames"].float(), r["frames"].float()
    print(f"frames psnr {(-10 * torch.log10(((a - b) ** 2).mean())).item():.2f} dB")
if "wave" in r and "wave" in o:
    print(f"wave rel {rel(o['wave'], r['wave']):.3e}")
