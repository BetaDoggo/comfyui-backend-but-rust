"""Convert nvidia/Nemotron-Labs-Image-30B-A3B (130 GB of F32 shards) into the single nvfp4 file the engine loads.

Each shard is downloaded into RAM, converted and dropped, so the F32 original never touches the disk and the run needs
only the output's space (about 24 GB). The next shard downloads while the current one converts.

    python tools/nemotron_image_convert.py <out.safetensors> [--work <dir>]

What the file holds:
  * routed experts (`backbone.layers.N.mixer.experts.E.{up,down}_proj`): nvfp4. `weight` is U8 [N, K/2], two E2M1
    values a byte, element 2i in the low nibble; `weight_scale` is F8_E4M3 [N, K/16], one scale per 16 columns, row
    major; `weight_scale_2` is an F32 scalar. A value is e2m1 * f32(e4m3) * scale_2. Each block's scale is chosen from a
    few candidates below its absolute maximum for the least squared error.
  * other matrices (Mamba projections, attention, shared experts, embeddings, lm_head): BF16.
  * vectors, the conv1d kernels and the router: F32, as shipped.
  * `image_decoder.*`: the decoder half of Nemotron-Labs-Image-Tokenizer (post_quant_embed, decoder blocks, ffn,
    conv_out and the codebook), F32 as shipped. The encoder and the distill head aren't needed to make images.
  * `tokenizer.json`: the text tokenizer's file, as U8 bytes.
  * metadata: `nemotron_image` (the format version), `config` (config.json), `manifest`
    (unitron_extension_manifest.json) and `decoder_config` (the tokenizer model's config.json).

`--work` keeps progress there, so an interrupted run resumes at the next shard. Run with a Python that has torch and
CUDA, such as SwarmUI's python_embeded.
"""

import argparse
import json
import os
import pickle
import queue
import sys
import threading
import time
import urllib.request

import torch
from safetensors.torch import load as st_load

REPO = "https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B/resolve/main/"
FORMAT = "nvfp4-v1"
HEADER_RESERVE = 8 << 20
ALIGN = 64
# multiples of each block's ideal scale (absmax / 6) tried for the least error
FACTORS = [1.06, 1.03, 1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73]
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DT_NAME = {torch.float32: "F32", torch.bfloat16: "BF16", torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3"}


def _range(url, buf, start, end, retries):
    """buf[start:end] from the URL, resuming after dropped connections."""
    at = start
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={at}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                while at < end:
                    chunk = r.read(min(4 << 20, end - at))
                    if not chunk:
                        break
                    buf[at:at + len(chunk)] = chunk
                    at += len(chunk)
            if at == end:
                return
            raise IOError(f"short read at {at} of {end}")
        except Exception as e:  # noqa: BLE001 - any network error retries
            print(f"  range {start}-{end}: {e}, retrying ({attempt + 1})", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"giving up on {url} bytes {start}-{end}")


def fetch(name, retries=20, streams=8):
    """The file's bytes, read into RAM over several connections."""
    url = REPO + name
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
                total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            break
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: {e}, retrying ({attempt + 1})", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    else:
        raise RuntimeError(f"giving up on {name}")
    buf = bytearray(total)
    n = streams if total > (64 << 20) else 1
    step = -(-total // n)
    failed = []

    def part(i):
        try:
            _range(url, buf, i * step, min(total, (i + 1) * step), retries)
        except Exception as e:  # noqa: BLE001 - reported below
            failed.append(e)

    threads = [threading.Thread(target=part, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        raise RuntimeError(f"{name}: {failed[0]}")
    return bytes(buf)


def nvfp4(w):
    """w [N, K] f32 on the GPU -> (q U8 [N, K/2], scales F8 [N, K/16], scale_2 F32 scalar, relative rms error)."""
    n, k = w.shape
    assert k % 32 == 0, w.shape
    grid = torch.tensor(E2M1, device=w.device)
    bounds = (grid[1:] + grid[:-1]) / 2
    amax = w.abs().max().clamp(min=1e-30)
    s2 = (amax / (6.0 * 448.0)).float()
    b = w.view(n, k // 16, 16)
    bmax = b.abs().amax(-1)
    best_err = best_sc = best_idx = None
    for f in FACTORS:
        sc = (bmax / 6.0 / s2 * f).clamp(max=448.0).to(torch.float8_e4m3fn)
        s = sc.float() * s2
        x = torch.where(s[..., None] > 0, b / s[..., None].clamp(min=1e-38), torch.zeros_like(b))
        idx = torch.bucketize(x.abs().clamp(max=6.0), bounds)
        deq = grid[idx] * torch.sign(x) * s[..., None]
        err = (deq - b).square().sum(-1)
        code = idx + torch.where((x < 0) & (idx > 0), 8, 0)
        if best_err is None:
            best_err, best_sc, best_idx = err, sc, code
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_sc = torch.where(better, sc.view(torch.uint8), best_sc.view(torch.uint8)).view(torch.float8_e4m3fn)
            best_idx = torch.where(better[..., None], code, best_idx)
    codes = best_idx.reshape(n, k).to(torch.uint8)
    q = codes[:, 0::2] | (codes[:, 1::2] << 4)
    rel = (best_err.sum() / b.square().sum().clamp(min=1e-30)).sqrt().item()
    return q.contiguous(), best_sc.contiguous(), s2.reshape(()), rel


def is_expert(name):
    return ".mixer.experts." in name and name.endswith("_proj.weight")


def convert(name, t, dev):
    """[(name, tensor on the CPU)] for one source tensor, and the expert's error or None."""
    if is_expert(name):
        q, sc, s2, rel = nvfp4(t.to(dev, torch.float32))
        base = name[: -len(".weight")]
        return [(name, q.cpu()), (base + ".weight_scale", sc.cpu()), (base + ".weight_scale_2", s2.cpu())], rel
    if t.ndim == 2 and not name.endswith(".mixer.gate.weight"):
        return [(name, t.to(torch.bfloat16))], None
    return [(name, t.to(torch.float32))], None


class Writer:
    """Big tensors go to the file as they come; small ones wait for the end, f32 first, so every offset stays aligned to
    its element."""

    def __init__(self, path, state):
        self.path = path
        self.entries = state.get("entries", {})
        self.tail = state.get("tail", {})
        self.offset = state.get("offset", 0)
        mode = "r+b" if os.path.exists(path) and self.offset else "wb"
        self.f = open(path, mode)
        if mode == "wb":
            self.f.write(b" " * (8 + HEADER_RESERVE))
        self.f.truncate(8 + HEADER_RESERVE + self.offset)
        self.f.seek(8 + HEADER_RESERVE + self.offset)

    def add(self, name, t):
        data = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if len(data) >= (1 << 20) and len(data) % ALIGN == 0:
            self.f.write(data)
            self.entries[name] = {"dtype": DT_NAME[t.dtype], "shape": list(t.shape), "data_offsets": [self.offset, self.offset + len(data)]}
            self.offset += len(data)
        else:
            self.tail[name] = (DT_NAME[t.dtype], list(t.shape), data)

    def state(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        return {"entries": self.entries, "tail": self.tail, "offset": self.offset}

    def finish(self, metadata):
        order = {"F32": 0, "BF16": 1, "F8_E4M3": 2, "U8": 2}
        for name, (dt, shape, data) in sorted(self.tail.items(), key=lambda kv: (order[kv[1][0]], kv[0])):
            self.f.write(data)
            self.entries[name] = {"dtype": dt, "shape": shape, "data_offsets": [self.offset, self.offset + len(data)]}
            self.offset += len(data)
        header = dict(sorted(self.entries.items(), key=lambda kv: kv[1]["data_offsets"][0]))
        header = {"__metadata__": metadata, **header}
        raw = json.dumps(header, separators=(",", ":")).encode()
        assert len(raw) <= HEADER_RESERVE, f"header {len(raw)} bytes over the reserve"
        self.f.seek(0)
        self.f.write(HEADER_RESERVE.to_bytes(8, "little"))
        self.f.write(raw + b" " * (HEADER_RESERVE - len(raw)))
        self.f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--work", default=None, help="folder for progress, default next to the output")
    args = ap.parse_args()
    work = args.work or os.path.splitext(args.out)[0] + ".convert"
    os.makedirs(work, exist_ok=True)
    state_path = os.path.join(work, "state.pkl")
    state = pickle.load(open(state_path, "rb")) if os.path.exists(state_path) else {}
    done = set(state.get("done", []))
    errs = state.get("errs", [])
    dev = torch.device("cuda")

    index = json.loads(fetch("model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(index.values()))
    writer = Writer(args.out, state)
    todo = [s for s in shards if s not in done] + (["decoder"] if "decoder" not in done else [])
    print(f"{len(done)} parts done, {len(todo)} to go", flush=True)

    # one download ahead of the conversion
    q = queue.Queue(maxsize=1)

    def downloader():
        for s in todo:
            t0 = time.time()
            data = fetch("Nemotron-Labs-Image-Tokenizer/model.safetensors" if s == "decoder" else s)
            print(f"{s}: {len(data) / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
            q.put((s, data))
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    while (item := q.get()) is not None:
        s, data = item
        t0 = time.time()
        tensors = st_load(data)
        del data
        for name in sorted(tensors):
            t = tensors[name]
            if s == "decoder":
                keep = name.startswith(("decoder.", "post_quant_embed.", "quantize.")) and not name.startswith("decoder.distill_mlp.")
                if keep:
                    writer.add("image_decoder." + name, t.to(torch.float32))
                continue
            outs, rel = convert(name, t, dev)
            for out_name, out in outs:
                writer.add(out_name, out)
            if rel is not None:
                errs.append(rel)
        del tensors
        done.add(s)
        st = writer.state()
        st.update(done=sorted(done), errs=errs)
        with open(state_path + ".tmp", "wb") as f:
            pickle.dump(st, f)
        os.replace(state_path + ".tmp", state_path)
        print(f"{s}: converted in {time.time() - t0:.0f} s, {writer.offset / 1e9:.2f} GB written", flush=True)

    meta = {"format": "pt", "nemotron_image": FORMAT}
    for key, name in [("config", "config.json"), ("manifest", "unitron_extension_manifest.json"),
                      ("decoder_config", "Nemotron-Labs-Image-Tokenizer/config.json")]:
        meta[key] = fetch(name).decode()
    tok = fetch("tokenizer.json")
    writer.tail["tokenizer.json"] = ("U8", [len(tok)], tok)
    writer.finish(meta)
    if errs:
        e = torch.tensor(errs)
        print(f"expert nvfp4 relative rms error: mean {e.mean():.4f}, max {e.max():.4f} over {len(errs)}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
