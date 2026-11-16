"""Build an activation-steering vector for Anima from two `--steer-record` files.

Record the same prompts and seeds twice, once with the concept and once without, then:

    python tools/steer_build.py with.safetensors without.safetensors -o snow.safetensors --steps 0-3

Each block's direction is the mean difference of its image tokens over `--steps` (b exclusive), with the model's
massive-activation channels zeroed (channels whose mean magnitude in the reference recording is over `--outlier` times
the block's median), normalised to unit length. `norm` is the difference's length, which `--steer-alpha` scales: alpha 1
adds the average shift between the recordings. Prints how strong the difference is per block, relative to the mean
token norm.

Needs only numpy.
"""
import argparse
import json
import struct

import numpy as np


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        dt = {"F32": np.float32, "I64": np.int64}[v["dtype"]]
        out[k] = np.frombuffer(data[a:b], dtype=dt).reshape(v["shape"])
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("with_concept")
    p.add_argument("without_concept")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--steps", default="0-3", help="steps averaged, a-b with b exclusive")
    p.add_argument("--outlier", type=float, default=8.0)
    a = p.parse_args()
    lo, hi = (int(x) for x in a.steps.split("-"))
    pos, neg = read_st(a.with_concept), read_st(a.without_concept)
    mp, mn = pos["mean"][lo:hi].mean(0), neg["mean"][lo:hi].mean(0)  # [blocks, width]
    norm = neg["norm"][lo:hi].mean(0)  # [blocks]
    diff = mp - mn
    mag = np.abs(mn)
    med = np.median(mag, axis=1, keepdims=True)
    outliers = mag > a.outlier * np.maximum(med, 1e-6)
    diff[outliers] = 0.0
    length = np.linalg.norm(diff, axis=1)
    d = diff / np.maximum(length[:, None], 1e-12)
    cos_next = (d[:-1] * d[1:]).sum(1)
    print("block  |diff|/norm  outliers  cos(next block)")
    for b in range(len(norm)):
        c = f"{cos_next[b]:.2f}" if b < len(cos_next) else ""
        print(f"{b:5d}  {length[b] / norm[b]:10.4f}  {int(outliers[b].sum()):8d}  {c}")
    # `norm` is the unit --steer-alpha multiplies: the difference's own length, so alpha 1 adds the average shift
    # between the two recordings. Anima's token norms are dominated by a few massive channels, so they make a poor unit.
    write_st(a.out, {"dir": d, "norm": length, "token_norm": norm})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
