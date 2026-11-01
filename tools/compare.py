"""Compare a krea2 --dump file against tools/ref_dump.py output.

usage: python compare.py ref.safetensors rust.safetensors [--noise-check SEED]
"""
import sys
import numpy as np
import torch
from safetensors.torch import load_file

ref = load_file(sys.argv[1])
rs = load_file(sys.argv[2])


def stat(name, a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() != b.numel():
        print(f"{name:18s} size mismatch {a.numel()} vs {b.numel()}")
        return
    d = (a - b).abs()
    rel = d.norm() / a.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
    print(f"{name:18s} max|d|={d.max():.4g} mean|d|={d.mean():.4g} rel_l2={rel:.4g} cos={cos:.6f}")


print("tokens equal:", ref["token_ids"].tolist() == rs["token_ids"].tolist())
stat("cond", ref["cond"][0], rs["cond"])
stat("txtmlp_out", ref["txtmlp_out"][0], rs["txtmlp_out"])
lt = rs["txtmlp_out"].shape[0]
if "block0" in rs:
    stat("block0_out", ref["block0_out"][0], rs["block0"])
    stat("blocklast_out", ref["blocklast_out"][0], rs["block27"])
if "dit_out_tokens" in rs:
    from einops import rearrange
    o = ref["dit_out"].reshape(1, 16, ref["dit_out"].shape[-2], ref["dit_out"].shape[-1])
    o = rearrange(o, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)[0]
    stat("dit_out(step0)", o, rs["dit_out_tokens"])
# reference samples are after process_latent_out
mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(16, 1, 1)
std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(16, 1, 1)
stat("final latent", ref["samples"][0, :, 0], rs["latent_final"] * std + mean)
ri = (np.clip(ref["image"][0, 0].numpy() * 255.0, 0, 255)).astype(np.uint8)
stat("image (0-255)", torch.from_numpy(ri.astype(np.float32)), rs["image_rgb8"].float())
diff = np.abs(ri.astype(np.int32) - rs["image_rgb8"].numpy().astype(np.int32))
print("image: mean abs diff %.3f / 255, PSNR %.2f dB" % (diff.mean(), 10 * np.log10(255 ** 2 / max((diff ** 2).mean(), 1e-9))))

if "--noise-check" in sys.argv:
    seed = int(sys.argv[sys.argv.index("--noise-check") + 1])
    g = torch.manual_seed(seed)
    n = torch.randn(ref["noise"].shape, dtype=torch.float32, generator=g)
    stat("noise(torch,seed)", ref["noise"], n)
