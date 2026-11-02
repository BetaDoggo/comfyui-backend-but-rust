"""Exact float32 run of H3's token refiner, to judge how close ComfyUI's and this engine's int8 paths are to it.

usage: python h3_refiner_check.py <dit.safetensors> <ref.safetensors> <ours.safetensors>
"""
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

dit, ref, ours = sys.argv[1:4]
r, o = load_file(ref), load_file(ours)
dev = "cuda"
f = safe_open(dit, "pt", device=dev)


def h256(w):
    # the ConvRot basis, self-inverse: H4^(x4)/16 per 256 columns
    h4 = 0.5 * torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=w.dtype, device=w.device)
    n, k = w.shape
    x = w.view(n, k // 256, 4, 4, 4, 4)
    for dim in (5, 4, 3, 2):
        x = torch.tensordot(x, h4, dims=([dim], [1])).movedim(-1, dim)
    return x.reshape(n, k)


def weight(name):
    q = f.get_tensor(name + ".weight").float()
    s = f.get_tensor(name + ".weight_scale").float().view(-1, 1)
    return h256(q * s)


def rms(x, w, eps=1e-5):
    return F.rms_norm(x, (x.shape[-1],), weight=w, eps=eps)


x = r["context"].to(dev).float()[0]
x = F.linear(x, f.get_tensor("condition_proj.weight").float(), f.get_tensor("condition_proj.bias").float())
for i in range(2):
    p = f"token_refiner.blocks.{i}."
    h = rms(x, f.get_tensor(p + "norm1.weight").float())
    qkv = F.linear(h, weight(p + "attn.qkv_proj")).view(x.shape[0], 3, 56, 128)
    q = rms(qkv[:, 0], f.get_tensor(p + "attn.q_norm.weight").float())
    k = rms(qkv[:, 1], f.get_tensor(p + "attn.k_norm.weight").float())
    v = qkv[:, 2]
    a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1).reshape(x.shape[0], -1)
    x = x + F.linear(a, weight(p + "attn.out_proj"))
    h = rms(x, f.get_tensor(p + "norm2.weight").float())
    gate, up = F.linear(h, weight(p + "mlp.fc1")).chunk(2, -1)
    x = x + F.linear(F.silu(gate) * up, weight(p + "mlp.fc2"))
x = rms(x, f.get_tensor("token_refiner.final_norm.weight").float()).cpu()
c, n = r["ctx"].view(-1, 5376).float(), o["ctx"].view(-1, 5376).float()
for name, a, b in (("ComfyUI vs exact", c, x), ("ours vs exact", n, x), ("ours vs ComfyUI", n, c)):
    print(f"{name}: {((a - b).norm() / b.norm()).item():.3e}")
