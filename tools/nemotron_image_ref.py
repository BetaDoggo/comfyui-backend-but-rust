"""Check the engine's Nemotron-Labs-Image logits against the model's own code (modeling_nemotron_h.py from the
model's repository), run a layer at a time so it fits in 12 GB.

    diffusion.exe --nemotron ... with NEMOTRON_DUMP=<dump.safetensors> set, then
    python tools/nemotron_image_ref.py <model.safetensors> <dump.safetensors> <folder with modeling_nemotron_h.py and
        configuration_nemotron_h.py> [--prompt TEXT --width W --height H]

With --prompt, the dump's prompt ids are first checked against the `tokenizers` library's encoding of the chat
template the model card renders (the model's tokenizer wrapper encodes it with the base tokenizer, no special tokens).

The reference reads the same converted file: nvfp4 experts are decoded to float32, the rest widened to float32, so
only the engine's arithmetic differs. The whole sequence (prompt and the dumped tokens) runs at once through the
blocks' full-sequence path, which uses Mamba-2's chunked scan where the engine steps one token at a time.

mamba_ssm and causal_conv1d aren't installed on Windows; stand-ins with their names let the file import, and
torch_forward, its pure PyTorch path, runs. Its gated RMSNorm (`rmsnorm_fn`) is written here from mamba_ssm's
definition: x * silu(z), normalized over groups.
"""

import importlib.util
import json
import sys
import types

import torch
from safetensors import safe_open

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, **_):
    dtype = x.dtype
    x = x.float()
    if z is not None and not norm_before_gate:
        x = x * torch.nn.functional.silu(z.float())
    g = x.view(*x.shape[:-1], -1, group_size)
    g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + eps)
    out = g.view_as(x) * weight.float()
    if z is not None and norm_before_gate:
        out = out * torch.nn.functional.silu(z.float())
    return out.to(dtype)


def stub_mamba():
    for name in ["mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.selective_state_update",
                 "mamba_ssm.ops.triton.ssd_combined", "mamba_ssm.ops.triton.layernorm_gated", "causal_conv1d"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mamba_ssm.ops.triton.selective_state_update"].selective_state_update = None
    sys.modules["mamba_ssm.ops.triton.ssd_combined"].mamba_chunk_scan_combined = None
    sys.modules["mamba_ssm.ops.triton.ssd_combined"].mamba_split_conv1d_scan_combined = None
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = rmsnorm_fn
    sys.modules["causal_conv1d"].causal_conv1d_fn = None
    sys.modules["causal_conv1d"].causal_conv1d_update = None


def load_code(folder):
    stub_mamba()
    # the stand-ins have no spec for transformers' checks: report the packages missing, so the fast path stays off
    import transformers.utils.import_utils as iu
    iu.is_mamba_2_ssm_available = lambda: False
    iu.is_causal_conv1d_available = lambda: False
    pkg = types.ModuleType("nemo")
    pkg.__path__ = [folder]
    sys.modules["nemo"] = pkg
    mods = {}
    for name in ["configuration_nemotron_h", "modeling_nemotron_h"]:
        spec = importlib.util.spec_from_file_location(f"nemo.{name}", f"{folder}/{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"nemo.{name}"] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods["configuration_nemotron_h"], mods["modeling_nemotron_h"]


def dequant(f, base, dev):
    q = f.get_tensor(base + ".weight").to(dev)
    sc = f.get_tensor(base + ".weight_scale").to(dev).float()
    s2 = f.get_tensor(base + ".weight_scale_2").item()
    lut = torch.tensor(E2M1, device=dev)
    vals = torch.stack([lut[(q & 15).long()], lut[(q >> 4).long()]], -1).reshape(q.shape[0], -1)
    return (vals.view(q.shape[0], -1, 16) * sc[..., None] * s2).reshape(q.shape[0], -1)


SYSTEM = ("Generate an image matching the requested resolution by detailing the color, quantity, text, shape, size, texture, "
          "spatial relationships of the objects and background.")


def check_ids(f, ids, prompt, width, height):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(bytes(f.get_tensor("tokenizer.json").tolist()).decode())
    text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nGenerate a {width}x{height} image.\n\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>")
    want = tok.encode(text, add_special_tokens=False).ids
    print(f"prompt ids {'match' if want == ids else 'DIFFER'} ({len(ids)} engine, {len(want)} tokenizers)")
    if want != ids:
        print("engine    ", ids)
        print("tokenizers", want)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    for a in ["model", "dump", "code"]:
        ap.add_argument(a)
    ap.add_argument("--prompt")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()
    model_path, dump_path, code = args.model, args.dump, args.code
    dev = torch.device("cuda")
    cfg_mod, modeling = load_code(code)
    f = safe_open(model_path, "pt")
    cfg_dict = json.loads(f.metadata()["config"])
    cfg_dict.pop("auto_map", None)
    tl = cfg_dict.get("time_step_limit")
    if isinstance(tl, list) and isinstance(tl[1], dict):
        cfg_dict["time_step_limit"] = (tl[0], float("inf"))
    config = cfg_mod.NemotronHConfig(**cfg_dict)
    # the sdpa class still reshapes to hidden_size (2688, not 32 x 128); the eager one is causal sdpa with the fixed reshape
    config._attn_implementation = "eager"
    d = safe_open(dump_path, "pt")
    ids = d.get_tensor("ids").long().tolist()
    tokens = d.get_tensor("tokens").long().tolist()
    seq = ids + tokens
    if args.prompt is not None:
        check_ids(f, ids, args.prompt, args.width, args.height)
    steps = len(tokens)
    print(f"{len(ids)} prompt tokens, {steps} generated", flush=True)

    with torch.no_grad():
        emb = f.get_tensor("backbone.embeddings.weight")
        x = emb[torch.tensor(seq)].float().to(dev)[None]
        del emb
        for i in range(config.num_hidden_layers):
            block = modeling.NemotronHBlock(config, layer_idx=i).float()
            prefix = f"backbone.layers.{i}."
            sd = {}
            for name, p in block.state_dict().items():
                key = prefix + name
                if ".experts." in key and key.endswith("_proj.weight"):
                    sd[name] = dequant(f, key[: -len(".weight")], dev)
                else:
                    sd[name] = f.get_tensor(key).float()
            block.load_state_dict(sd)
            block.to(dev)
            x = block(x)
            del block, sd
            torch.cuda.empty_cache()
            print(f"layer {i} ({config.hybrid_override_pattern[i]}) rms {x.float().square().mean().sqrt().item():.4f}", flush=True)
        norm_w = f.get_tensor("backbone.norm_f.weight").float().to(dev)
        h = x[0].float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + config.layer_norm_epsilon) * norm_w
        head = f.get_tensor("lm_head.weight").to(dev)
        rows = h[len(ids) - 1:]
        for s in range(steps + 1):
            key = f"logits.{s}"
            if key not in d.keys():
                break
            ours = d.get_tensor(key).float().to(dev)
            first = int(d.get_tensor(f"first.{s}")[0])
            ref = (head[first:first + ours.numel()].float() @ rows[s]).float()
            diff = (ours - ref)
            rel = (diff.norm() / ref.norm()).item()
            top_o, top_r = ours.argmax().item() + first, ref.argmax().item() + first
            # agreement of the top 10 tokens
            o10, r10 = set(ours.topk(10).indices.tolist()), set(ref.topk(10).indices.tolist())
            print(f"step {s:3d}: logits rel err {rel:.2e}, max abs {diff.abs().max().item():.2e} (of {ref.abs().max().item():.1f}), argmax ours {top_o} ref {top_r}, "
                  f"top-10 overlap {len(o10 & r10)}/10")


if __name__ == "__main__":
    main()
