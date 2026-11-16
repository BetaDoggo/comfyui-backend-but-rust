"""Fit a low-rank steering map for Anima from two `--steer-tokens` files of the same prompts' seeds, one run with the
concept and one without (row i of each is the same token of the same noise):

    python tools/steer_fit.py with.safetensors without.safetensors -o snow_map.safetensors --steps 0-3 --rank 32

For each block it fits `with - without ~= (without - mu) B + d` by ridge regression, keeps B's top `--rank` directions
of the fitted change (reduced-rank regression), and writes `U.b<b>` [rank, width], `V.b<b>` [width, rank] and `c.b<b>`
[width] so that the engine adds `alpha * (x U^T V^T + c)` to every token. Massive-activation channels (mean magnitude
over `--outlier` times the median) are left out on both sides. The last `--holdout` images score the fit: R^2 of the
map, and of a constant shift alone (what a plain steering vector can reach).

Needs only numpy.
"""
import argparse
import json
import re
import struct

import numpy as np


def read_bf16(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        raw = np.asarray(mm[(base + a) // 2:(base + b) // 2], dtype=np.uint32) << 16
        out[k] = raw.view(np.float32).reshape(v["shape"])
    return out


def write_st(path, tensors):
    header, blobs, off = {}, [], 0
    for k, v in tensors.items():
        b = np.ascontiguousarray(v, dtype=np.float32).tobytes()
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def r2(pred, target):
    return 1 - ((target - pred) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("with_concept")
    p.add_argument("without_concept")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--steps", default="0-3", help="steps fitted, a-b with b exclusive")
    p.add_argument("--blocks", default=None, help="comma separated; default every block in the files")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1e-2, help="relative to the mean input variance")
    p.add_argument("--outlier", type=float, default=8.0)
    p.add_argument("--images", type=int, default=64, help="images in each run")
    p.add_argument("--holdout", type=int, default=8)
    a = p.parse_args()
    lo, hi = (int(x) for x in a.steps.split("-"))
    pos, neg = read_bf16(a.with_concept), read_bf16(a.without_concept)
    found = sorted({int(m.group(1)) for k in neg for m in [re.match(r"tok\.b(\d+)\.s\d+", k)] if m})
    blocks = [int(b) for b in a.blocks.split(",")] if a.blocks else found
    out = {"map_blocks": np.array(blocks, dtype=np.float32)}
    print("block  R2 map  R2 shift  |c|      |change| per token")
    for b in blocks:
        xs, ds, tr = [], [], []
        for s in range(lo, hi):
            x, y = neg[f"tok.b{b}.s{s}"], pos[f"tok.b{b}.s{s}"]
            per = len(x) // a.images
            xs.append(x)
            ds.append(y - x)
            tr.append(np.arange(len(x)) < per * (a.images - a.holdout))
        X, D, train = np.concatenate(xs).astype(np.float64), np.concatenate(ds).astype(np.float64), np.concatenate(tr)
        mag = np.abs(X).mean(0)
        keep = mag <= a.outlier * np.median(mag)
        X[:, ~keep] = 0
        D[:, ~keep] = 0
        Xt, Dt = X[train], D[train]
        mu, dm = Xt.mean(0), Dt.mean(0)
        Xc, Dc = Xt - mu, Dt - dm
        G = Xc.T @ Xc
        lam = a.ridge * np.trace(G) / keep.sum()
        B = np.linalg.solve(G + lam * np.eye(len(G)), Xc.T @ Dc)
        _, _, vt = np.linalg.svd(Xc @ B, full_matrices=False)
        Vr = vt[: a.rank].T  # [width, rank]
        Ut = B @ Vr  # [width, rank]: x -> rank coordinates
        c = dm - mu @ Ut @ Vr.T
        Xh, Dh = X[~train], D[~train]
        pred = Xh @ Ut @ Vr.T + c
        print(f"{b:5d}  {r2(pred, Dh):6.3f}  {r2(np.broadcast_to(dm, Dh.shape), Dh):8.3f}  {np.linalg.norm(c):7.1f}  "
              f"{np.linalg.norm(Dh, axis=1).mean():7.1f}")
        out[f"U.b{b}"] = Ut.T
        out[f"V.b{b}"] = Vr
        out[f"c.b{b}"] = c
    write_st(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
