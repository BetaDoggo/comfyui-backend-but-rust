"""Compare a diffusion --moss --dump file with tools/moss_ref.py output.

usage: python moss_cmp.py <ref.safetensors> <ours.safetensors>
"""
import sys

import torch
from safetensors.torch import load_file

ref = load_file(sys.argv[1])
ours = load_file(sys.argv[2])


def stats(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        print(f"{name:22} SIZE {a.numel()} vs {b.numel()}")
        return
    err = (a - b).norm() / a.norm().clamp_min(1e-12)
    print(f"{name:22} rel err {err:.3e}  max {(a - b).abs().max():.3e}  ref rms {a.pow(2).mean().sqrt():.3e}")


same = ref["ids"].equal(ours["ids"])
print(f"ids {'same' if same else 'DIFFERENT'} ({ref['ids'].numel()} / {ours['ids'].numel()})")
if not same:
    print("  ref ", ref["ids"].tolist())
    print("  ours", ours["ids"].tolist())
for r, o in [("te_norm0", "te_norm0"), ("te_layer0", "te_layer0"), ("te_layer3", "te_layer3"), ("te_layer31", "te_layer31"), ("cond", "cond"),
             ("text_embedder_out", "text_embedder_out"), ("video_embedder_out", "video_embedder_out"), ("block0_out", "block0"),
             ("block1_out", "block1"), ("blocklast_out", "block41"), ("samples", "samples")]:
    if r in ref and o in ours:
        stats(r, ref[r], ours[o])
        if r.startswith("block"):
            a, b = ref[r].float().reshape(-1, 3072), ours[o].float().reshape(-1, 3072)
            stats("  image tokens", a[:-ref["ids"].numel()], b[:-ref["ids"].numel()])
            stats("  text tokens", a[-ref["ids"].numel():], b[-ref["ids"].numel():])
if "final_tokens" in ours:
    v = ref["dit_out"].float()[0].flatten(1).T
    stats("dit_out", v, ours["final_tokens"])
