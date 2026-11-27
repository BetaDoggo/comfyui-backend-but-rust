"""Record sol3_attn's Sol³ attention on generated q/k/v, for tests/sol3_attn_match.rs.

The inputs come from the integer hash tools/sparse_attn_ref.py uses, which the test repeats, so only results are
stored: every 251st output value, each query block's route (exact, pure and mixed key blocks, and the tokens selected in
the mixed ones) and its token cut. Needs a Python with CUDA PyTorch, Triton and sol3_attn on an SM120 GPU:

    python -I tools/sol3_ref.py tests/sol3_golden.json

sol3_attn is github.com/FofGofx/sol3 at 21a509a1cf232168fefddcc06983ee1456588bca, the commit MagoNodes pins.
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source  # noqa: E402

import torch

from sol3_attn.interface import _forward
from sol3_attn.backends.sm120._providers import h3_mu_route

H, D = 2, 128
STRIDE = 251
# name, tokens, protected prefix rows. 16420 takes upstream's long-sequence launch (64-row chunks, as this engine's);
# both end on a partial block.
CASES = [("long_prefix", 16420, 100), ("short_no_prefix", 4000, 0)]


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


def tensor(t_len, salt):
    """[1, T, H, D] bf16: a pattern shared by blocks of the same residue mod 5, plus per-token noise."""
    t = torch.arange(t_len, dtype=torch.int64).view(t_len, 1, 1)
    h = torch.arange(H, dtype=torch.int64).view(1, H, 1)
    d = torch.arange(D, dtype=torch.int64).view(1, 1, D)
    group = (t // 64) % 5
    shared = hash_unit((group * H + h) * D + d, salt)
    own = hash_unit((t * H + h) * D + d, salt + 7919)
    return (4.0 * shared + own).float().to(torch.bfloat16).unsqueeze(0).cuda().contiguous()


def main():
    cases = []
    for name, t_len, prefix in CASES:
        q, k, v = tensor(t_len, 1), tensor(t_len, 2), tensor(t_len, 3)
        out, route = _forward(q, k, v, prefix_tokens=prefix, dense=None)
        torch.cuda.synchronize()
        # rows with no mixed tokens have an infinite cut, stored as null
        cut = [c if c != float("inf") else None for c in h3_mu_route.LAST["cut"].float().flatten().tolist()]
        flags = route["flags"].view(H, -1, route["flags"].shape[-1])
        rows = []
        for f in flags.reshape(-1, flags.shape[-1]):
            rows.append([int((f == -1).sum()), int((f == -2).sum()), int((f >= 0).sum()), int(f.clamp(min=0).sum())])
        cases.append({
            "name": name, "t": t_len, "prefix": prefix,
            "out": out.float().flatten()[::STRIDE].tolist(), "rows": rows, "cut": cut,
        })
        print(f"{name}: {sum(r[0] for r in rows)} exact, {sum(r[1] for r in rows)} pure, {sum(r[2] for r in rows)} mixed "
              f"pairs, {sum(r[3] for r in rows)} selected tokens")
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump({"source": source(packages=["sol3_attn"]), "h": H, "stride": STRIDE, "cases": cases}, f)
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
