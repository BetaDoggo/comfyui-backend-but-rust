"""ComfyUI's SenseNova U1.6 as SwarmUI's text-to-image graph runs it, with what it computes dumped.

Run with SwarmUI's ComfyUI and its Python (ref/ComfyUI predates SenseNova). The graph is SwarmUI's: CheckpointLoaderSimple,
CLIPTextEncode for both prompts, SenseNovaSamplingOptions when --shift is given, EmptyHiDreamO1LatentImage and KSampler,
then the checkpoint's pixel-space VAE. Each --ref picture makes it an edit: HiDreamO1ReferenceImages, then SenseNova
Conditioning's `image` task, which ComfyUI needs to read them.

usage: python_embeded/python.exe sensenova_ref.py <out_dir> [--prompt "..."] [--negative "..."] [--size W H] [--steps N]
           [--cfg X] [--seed N] [--sampler euler] [--scheduler normal] [--shift S] [--lora path[:strength]] [--ref picture]
           [--tokens-only]

Writes dump.safetensors (token ids, with the pictures' image tokens for an edit, the first and last layers' prefix keys and values, the first model evaluation's
input, time and output, the sampled pixels) and image.png.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_env  # noqa: E402
COMFY = comfy_env.use_comfy()

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--prompt", default="no humans, scenery, a stone lighthouse on a rocky coast at sunset, calm sea, photograph")
ap.add_argument("--negative", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--sampler", default="euler")
ap.add_argument("--scheduler", default="normal")
ap.add_argument("--shift", type=float, default=None)
ap.add_argument("--lora", action="append", default=[], help="path[:strength], LoraLoaderModelOnly through ComfyUI's DynamicVRAM path")
ap.add_argument("--ref", action="append", default=[], help="a reference picture, for an edit")
ap.add_argument("--tokens-only", action="store_true")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

comfy_env.aimdo_start()

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils

MODELS = comfy_env.models_root()
CKPT = os.path.join(MODELS, r"Stable-diffusion\sensenova_u1.6_8b_mot_int8_convrot.safetensors")

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu", copy=True).contiguous()
    print(f"{name}: {tuple(t.shape)} {t.dtype}")


if args.tokens_only:
    from comfy.text_encoders.sensenova import SenseNovaTokenizer
    tok = SenseNovaTokenizer()
    for name, text in (("ids", args.prompt), ("neg_ids", args.negative)):
        put(name, torch.tensor([t[0] for t in tok.tokenize_with_weights(text)["sensenova_u15"][0]], dtype=torch.int64))
    save_file(dump, os.path.join(args.out, "dump.safetensors"))
    raise SystemExit

if True:
    # DynamicVRAM, as main.py runs it: the legacy path seeds LoRA rounding differently, and runs out of VRAM on an edit's
    # long prompt
    comfy_env.aimdo_enable()

t0 = time.perf_counter()
model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(CKPT, output_vae=True, output_clip=True)
print(f"loaded in {time.perf_counter() - t0:.1f}s")
for entry in args.lora:
    path, _, s = entry.rpartition(":") if entry.rsplit(":", 1)[-1].replace(".", "").replace("-", "").isdigit() else (entry, "", "1")
    model, _ = comfy.sd.load_lora_for_models(model, None, comfy.utils.load_torch_file(path, safe_load=True), float(s), 0.0)
    print("lora", path, s, len(model.patches), "patched weights")
if args.shift is not None:
    from comfy_extras.nodes_sensenova import SenseNovaSamplingOptions
    model = SenseNovaSamplingOptions.execute(model=model, shift=args.shift).args[0]


def encode(text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens), torch.tensor([t[0] for t in tokens["sensenova_u15"][0]], dtype=torch.int64)


positive, ids = encode(args.prompt)
negative, neg_ids = encode(args.negative)
if args.ref:
    from comfy.ldm.sensenova.conditioning import condition_input_ids, reference_grid
    from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
    from comfy_extras.nodes_sensenova import SenseNovaConditioning
    pics = [torch.from_numpy(np.asarray(Image.open(r).convert("RGB")).astype(np.float32) / 255.0)[None] for r in args.ref]
    positive, negative = HiDreamO1ReferenceImages.execute(positive=positive, negative=negative, images={f"image_{i + 1}": p for i, p in enumerate(pics)}).args
    out = SenseNovaConditioning.execute(conditioning=positive, task={"task": "image"}, max_tokens=1024, clip=clip, negative=negative).args
    positive, negative = out[0], out[2]
    grids = [reference_grid(p.shape[1], p.shape[2], 32) for p in pics]
    ids = condition_input_ids(ids[None], grids)[0]
    neg_ids = condition_input_ids(neg_ids[None], grids, image_only=True)[0]
put("ids", ids)
put("neg_ids", neg_ids)
extra = positive[0][1]
# SN_ALL_LAYERS=1 dumps every layer's, to see where a difference grows
for layer in range(len(extra["prefix_keys"])) if os.environ.get("SN_ALL_LAYERS") else (0, 1, len(extra["prefix_keys"]) - 1):
    put(f"prefix_k{layer}", extra["prefix_keys"][layer])
    put(f"prefix_v{layer}", extra["prefix_values"][layer])
put("prefix_time", extra["prefix_time"])

dm = model.model.diffusion_model
evals = {"n": 0}
orig_forward = dm._forward


def forward(x, timesteps, *a, **k):
    out = orig_forward(x, timesteps, *a, **k)
    if evals["n"] == 0:
        put("dit_in", x)
        put("dit_t", timesteps)
        put("dit_out", out)
    evals["n"] += 1
    return out


dm._forward = forward

w, h = args.size
latent = torch.zeros([1, 3, h, w])
noise = comfy.sample.prepare_noise(latent, args.seed)
t0 = time.perf_counter()
samples = comfy.sample.sample(model, noise, args.steps, args.cfg, args.sampler, args.scheduler, positive, negative, latent, seed=args.seed)
print(f"sampled in {time.perf_counter() - t0:.1f}s, {evals['n']} evaluations")
put("samples", samples)
image = vae.decode(samples)
a = image.detach().float().cpu().reshape(-1, *image.shape[-3:])[0]
Image.fromarray((a.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8), "RGB").save(os.path.join(args.out, "image.png"))
save_file(dump, os.path.join(args.out, "dump.safetensors"))
