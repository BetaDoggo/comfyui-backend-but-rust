"""Compare a diffusion --anima --dump file with tools/anima_ref.py output.

usage: python anima_cmp.py <ref.safetensors> <ours.safetensors>
"""
import sys

import torch
from safetensors.torch import load_file

ref = load_file(sys.argv[1])
ours = load_file(sys.argv[2]) if len(sys.argv) > 2 else {}


def stats(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        print(f"{name:22} SIZE {a.numel()} vs {b.numel()}")
        return
    err = (a - b).norm() / a.norm().clamp_min(1e-12)
    print(f"{name:22} rel err {err:.3e}  max {(a - b).abs().max():.3e}  ref rms {a.pow(2).mean().sqrt():.3e}")


# torch CUDA randn against the Philox port
sys.path.insert(0, ".")
if ours:
    for key in ("qwen_ids", "t5_ids"):
        same = ref[key].equal(ours[key])
        print(f"{key:22} {'same' if same else 'DIFFERENT'} ({ref[key].numel()} / {ours[key].numel()})")
        if not same:
            print("  ref ", ref[key].tolist()[:40])
            print("  ours", ours[key].tolist()[:40])
    pairs = [("cond", "cond"), ("dit_in_ctx", "dit_in_ctx"), ("x_embedder_out", "x_embedder_out"),
             ("block0_out", "block0"), ("blocklast_out", "block27"), ("samples", "samples")]
    for r, o in pairs:
        if r in ref and o in ours:
            stats(r, ref[r], ours[o])
    if "final_tokens" in ours:
        v = ref["dit_out"].float()
        print("dit_out shape", v.shape)
print({k: list(v.shape) for k, v in ref.items() if k.startswith("step")}, ref["sigmas"].tolist())
