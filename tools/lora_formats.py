"""Write a LoRA with every format ComfyUI reads, one module each, for checking ours with lora_ref.py and lora_cmp.py.

usage: python lora_formats.py <krea|anima|moss> <out.safetensors> [--te] [--computable]

--te also patches the text encoder, with the three key styles ComfyUI accepts. --computable leaves out
OFT, BOFT and Tucker LoKr, which ComfyUI's adapters fail on for these models, so images can be compared.
Prints the --capture arguments for lora_ref.py. The values are random: this tests arithmetic, not pictures.
"""
import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("family", choices=["krea", "anima", "moss"])
ap.add_argument("out")
ap.add_argument("--te", action="store_true")
ap.add_argument("--computable", action="store_true")
args = ap.parse_args()

M = os.environ["MODELS_ROOT"]
FILES = {
    "krea": (r"\Stable-diffusion\kroma-v0.3.1-turbo-opd-W8A8.safetensors", r"\clip\qwen3vl_4b.safetensors", "model.language_model.", "qwen3vl_4b", "blocks.{}.", ["attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"]),
    "anima": (r"\Stable-diffusion\animaInt8Mxfp8_turboV11Int8.safetensors", r"\clip\qwen_3_600m.safetensors", "model.", "qwen3_06b", "blocks.{}.", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj", "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj"]),
    "moss": (r"\Stable-diffusion\moss_image_sft_int8_convrot_attn.safetensors", r"\clip\qwen3.5_9b_int8_convrot.safetensors", "model.language_model.", "qwen35_9b", "layers.{}.", ["proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"]),
}
dit, te, te_pre, comfy, block, leaves = FILES[args.family]
torch.manual_seed(1)
sd = {}


def r(*shape, scale=0.02):
    return (torch.randn(*shape) * scale).float()


def shapes(path):
    with safe_open(M + path, "pt") as f:
        return {k: f.get_slice(k).get_shape() for k in f.keys()}


def lokr_factors(n):
    a = next(x for x in (8, 4, 2, 1) if n % x == 0)
    return a, n // a


def write(x, n, k, form, rank=8):
    """One module of n rows and k columns, named `x` in the file."""
    a1, a2 = lokr_factors(n)
    b1, b2 = lokr_factors(k)
    if form == "lora_alpha":
        sd.update({f"{x}.lora_up.weight": r(n, rank), f"{x}.lora_down.weight": r(rank, k), f"{x}.alpha": torch.tensor(4.0)})
    elif form == "lora_dora_row":
        sd.update({f"{x}.lora_B.weight": r(n, rank), f"{x}.lora_A.weight": r(rank, k), f"{x}.dora_scale": r(n, 1, scale=1.0).abs() + 0.5})
    elif form == "lora_dora_col":
        sd.update({f"{x}.lora_B.weight": r(n, rank), f"{x}.lora_A.weight": r(rank, k), f"{x}.dora_scale": r(1, k, scale=1.0).abs() + 0.5, f"{x}.alpha": torch.tensor(2.0)})
    elif form == "locon_mid":
        sd.update({f"{x}.lora_up.weight": r(n, rank).reshape(n, rank, 1, 1), f"{x}.lora_mid.weight": r(rank, rank, 1, 1), f"{x}.lora_down.weight": r(rank, k).reshape(rank, k, 1, 1)})
    elif form == "loha":
        sd.update({f"{x}.hada_w1_a": r(n, rank), f"{x}.hada_w1_b": r(rank, k), f"{x}.hada_w2_a": r(n, rank), f"{x}.hada_w2_b": r(rank, k), f"{x}.alpha": torch.tensor(4.0)})
    elif form == "loha_tucker":
        sd.update({f"{x}.hada_w1_a": r(rank, n), f"{x}.hada_w1_b": r(rank, k), f"{x}.hada_w2_a": r(rank, n), f"{x}.hada_w2_b": r(rank, k), f"{x}.hada_t1": r(rank, rank, 1, 1), f"{x}.hada_t2": r(rank, rank, 1, 1)})
    elif form == "lokr_full":
        sd.update({f"{x}.lokr_w1": r(a1, b1), f"{x}.lokr_w2": r(a2, b2)})
    elif form == "lokr_pair":
        sd.update({f"{x}.lokr_w1": r(a1, b1), f"{x}.lokr_w2_a": r(a2, 4), f"{x}.lokr_w2_b": r(4, b2), f"{x}.alpha": torch.tensor(2.0)})
    elif form == "lokr_tucker":
        sd.update({f"{x}.lokr_w1_a": r(a1, 2), f"{x}.lokr_w1_b": r(2, b1), f"{x}.lokr_w2_a": r(4, a2), f"{x}.lokr_w2_b": r(4, b2), f"{x}.lokr_t2": r(4, 4, 1, 1), f"{x}.alpha": torch.tensor(2.0)})
    elif form == "glora":
        sd.update({f"{x}.a1.weight": r(k, rank), f"{x}.a2.weight": r(rank, k), f"{x}.b1.weight": r(n, rank), f"{x}.b2.weight": r(rank, k), f"{x}.alpha": torch.tensor(4.0)})
    elif form == "oft":
        size = 8
        sd.update({f"{x}.oft_blocks": r(n // size, size, size), f"{x}.alpha": torch.tensor(1.0)})
    elif form == "boft":
        size = 8
        sd.update({f"{x}.oft_blocks": r(2, n // size, size, size), f"{x}.alpha": torch.tensor(1.0)})
    elif form == "diff":
        sd[f"{x}.diff"] = r(n, k)
    elif form == "set":
        sd[f"{x}.set_weight"] = r(n, k, scale=1.0)
    elif form == "w_norm":
        sd[f"{x}.w_norm"] = r(n, k)
    elif form == "diff_over_lora":
        sd.update({f"{x}.lora_up.weight": r(n, rank), f"{x}.lora_down.weight": r(rank, k), f"{x}.diff": r(n, k)})
    else:
        raise ValueError(form)


FORMS = ["lora_alpha", "lora_dora_row", "lora_dora_col", "locon_mid", "loha", "loha_tucker", "lokr_full", "lokr_pair", "lokr_tucker", "glora", "oft", "boft", "diff", "set", "w_norm", "diff_over_lora"]
if args.computable:
    FORMS = [f for f in FORMS if f not in ("oft", "boft", "lokr_tucker")]
s = shapes(dit)
prefix = next((p for p in ("", "net.", "model.diffusion_model.", "diffusion_model.") if any(k.startswith(p + block.format(0)) for k in s)), "")
capture = []
slots = [block.format(b) + l for b in range(4) for l in leaves]
if args.family == "moss":
    # MOSS Image fuses q, k and v: address them apart, which also tests the row ranges
    slots = [m for m in slots if m.split(".")[-1] not in ("q", "k", "v")]
used = 0
for form, m in zip(FORMS, slots):
    key = f"{prefix}{m}.weight"
    if key not in s:
        raise SystemExit(f"{key} isn't in the model")
    n, k = s[key]
    write(f"diffusion_model.{m}", n, k, form)
    capture.append(f"diffusion_model.{m}")
    print(f"{form}: {m} {n}x{k}")
    used += 1

if args.family == "moss":
    m = block.format(5) + "proj"
    n, k = s[f"{m}.qkv.weight"]
    for p in "qk":
        write(f"{m}.{p}", n // 3, k, "lora_alpha")
    capture.append(f"diffusion_model.{m}.qkv")

if args.te:
    t = shapes(te)
    layer = te_pre + "layers.{}."
    leaf = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    names = [lambda m: "lora_te_" + m.replace(".", "_"), lambda m: f"text_encoders.{comfy}.transformer.model.{m}", lambda m: f"text_encoders.transformer.model.{m}"]
    for i, (name, form) in enumerate(zip(names, ["lora_alpha", "loha", "lokr_full"])):
        m = f"layers.{i}.{leaf[i]}"
        n, k = t[f"{te_pre}{m}.weight"]
        write(name(m), n, k, form)
        capture.append(f"{comfy}.transformer.model.{m}")
        print(f"te {form}: {name(m)} {n}x{k}")

save_file(sd, args.out)
print("--capture " + " --capture ".join(capture))
