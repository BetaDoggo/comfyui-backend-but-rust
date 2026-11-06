"""Record comfy-kitchen's Sol-Attn and top-k sparse attention on generated q/k/v, for tests/sparse_attn_match_comfy.rs.

The inputs come from an integer hash that the test repeats, so only outputs are stored: every 61st output value and
the top-k thresholds. Run with SwarmUI's ComfyUI Python, which has comfy_kitchen:

    python_embeded/python.exe -I tools/sparse_attn_ref.py tests/sparse_golden.json
"""

import json
import sys

import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.cuda import _topk_threshold

T, H, D = 1600, 2, 128
STRIDE = 61
CASES = [
    # name, tau, topk_ratio, sink_blocks, sink_q, token_aug
    ("sol", 1.3, 0.0, [0, 2], [1, 2], 256),
    ("sol_no_extra", 1.0, 0.0, [0, 0], [0, 0], 0),
    ("topk", 1.3, 0.15, [0, 2], [1, 2], 256),
    ("topk_no_extra", 1.3, 0.10, [0, 0], [0, 0], 0),
]


def hash_unit(idx, salt):
    """(idx * 2654435761 + salt) through two xorshift-multiply rounds, as uniform [-0.5, 0.5)."""
    m = 0xFFFFFFFF
    u = (idx * 2654435761 + salt) & m
    u ^= u >> 15
    u = (u * 2246822519) & m
    u ^= u >> 13
    u = (u * 3266489917) & m
    u ^= u >> 16
    return u.double() / 2**32 - 0.5


def tensor(salt):
    """[1, T, H, D] bf16: a pattern shared by blocks of the same residue mod 5, plus per-token noise."""
    t = torch.arange(T, dtype=torch.int64).view(T, 1, 1)
    h = torch.arange(H, dtype=torch.int64).view(1, H, 1)
    d = torch.arange(D, dtype=torch.int64).view(1, 1, D)
    group = (t // 64) % 5
    shared = hash_unit((group * H + h) * D + d, salt)
    own = hash_unit((t * H + h) * D + d, salt + 7919)
    return (4.0 * shared + own).float().to(torch.bfloat16).unsqueeze(0).cuda()


def main():
    q, k, v = tensor(1), tensor(2), tensor(3)
    scale = D ** -0.5
    out = []
    for name, tau, ratio, sinks, sink_q, aug in CASES:
        o = ck.sol_attn(q, k, v, tau=tau, scale=None, sink_blocks=sinks, sink_q=sink_q, topk_ratio=ratio, token_aug=aug)
        thr = None
        if ratio:
            thr = _topk_threshold(q, k, ratio, scale, None, None, sinks).flatten().tolist()
        flat = o.float().flatten()
        out.append({
            "name": name, "tau": tau, "ratio": ratio, "sinks": sinks, "sink_q": sink_q, "extra_tokens": aug,
            "out": flat[::STRIDE].tolist(), "threshold": thr,
        })
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump({"t": T, "h": H, "stride": STRIDE, "cases": out}, f)
    print(f"wrote {len(out)} cases")


if __name__ == "__main__":
    main()
