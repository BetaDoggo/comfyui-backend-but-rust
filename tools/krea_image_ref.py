"""ComfyUI's Krea 2 with images: vision tower, text encoder, VAE encoder and reference latents.

Run with SwarmUI's ComfyUI, whose Python and SwarmTextEncodeAdvanced node are what real workflows use:

usage: python_embeded/python.exe krea_image_ref.py <out_dir> --image a.png [--image b.png] [--ref c.png]
           [--method none|index|index_timestep_zero] [--prompt "..."] [--size W H] [--steps N] [--cfg X]
"""
import argparse
import os
import sys

SWARM = os.environ.get("SWARM_DIR", r"G:\StableSwarmUI")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(SWARM, "src", "BuiltinExtensions", "ComfyUIBackend", "ExtraNodes"))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--ref", action="append", default=[])
ap.add_argument("--method", default="none")
ap.add_argument("--prompt", default="no people, a stone lighthouse on a rocky coast at sunset")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[768, 768])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=1.0)
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.model_management
import comfy.sample
import comfy.sd
import comfy.utils
import nodes
import node_helpers
from SwarmComfyCommon.SwarmText import SwarmTextEncodeAdvanced

MODELS = os.environ["KREA2_MODELS"]
DIT = os.path.join(MODELS, r"Stable-diffusion\kroma-v0.3.1-turbo-opd-W8A8.safetensors")
TE = os.path.join(MODELS, r"clip\qwen3vl_4b.safetensors")
VAE = os.path.join(MODELS, r"VAE\qwen_image_vae.safetensors")

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu", copy=True).contiguous()
    print(f"{name}: {tuple(t.shape)} {t.dtype}")


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0)[None]


clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.KREA2)
visual = clip.cond_stage_model.qwen3vl_4b.transformer.visual


def hook(module, name):
    module.register_forward_hook(lambda _m, _in, out: put(name, out) if name not in dump else None)


hook(visual.patch_embed, "vision_patch")
hook(visual.blocks[0], "vision_block0")
hook(visual.blocks[23], "vision_block23")
hook(visual.merger, "vision_merged")
for i, m in enumerate(visual.deepstack_merger_list):
    hook(m, f"vision_deepstack{i}")

orig_encode = clip.encode_from_tokens_scheduled
last_tokens = []


def encode_from_tokens_scheduled(tokens, *a, **k):
    last_tokens.append(tokens)
    return orig_encode(tokens, *a, **k)


clip.encode_from_tokens_scheduled = encode_from_tokens_scheduled


def text_encode(prompt):
    images = None
    if args.image:
        loaded = [load(p) for p in args.image]
        if len({i.shape for i in loaded}) > 1:
            raise SystemExit("SwarmUI's ImageBatch would rescale these; pass images of one size")
        images = torch.cat(loaded)
    w, h = args.size
    return SwarmTextEncodeAdvanced().encode(clip, args.steps, prompt, w, h, w, h, llama_template="krea2", images=images)[0]


def token_ids(tokens):
    row = tokens["qwen3vl_4b"][0]
    return torch.tensor([t[0] if isinstance(t[0], (int, float)) else -1 for t in row], dtype=torch.int64)


positive = text_encode(args.prompt)
put("ids", token_ids(last_tokens[0]))
put("cond", positive[0][0])
negative = text_encode("")
put("neg_ids", token_ids(last_tokens[-1]))
put("neg_cond", negative[0][0])

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
refs = []
for i, p in enumerate(args.ref):
    raw = vae.encode(load(p))
    put(f"ref_raw{i}", raw)
    refs.append(raw)
if refs and args.method != "none":
    for r in refs:
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [r]}, append=True)
    positive = node_helpers.conditioning_set_values(positive, {"reference_latents_method": args.method})

model = comfy.sd.load_diffusion_model(DIT)
dm = model.model.diffusion_model
evals = {"n": 0}
orig_forward = dm._forward


def forward(x, timesteps, context, *a, **k):
    out = orig_forward(x, timesteps, context, *a, **k)
    if evals["n"] == 0:
        put("dit_in", x)
        put("dit_out", out)
        print("sigma", float(timesteps[0]))
    evals["n"] += 1
    return out


dm._forward = forward

w, h = args.size
latent = torch.zeros([1, 16, h // 8, w // 8], device=comfy.model_management.intermediate_device())
latent = comfy.sample.fix_empty_latent_channels(model, latent)
noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)
samples = comfy.sample.sample(model, noise, args.steps, args.cfg, "euler", "simple", positive, negative, latent, seed=args.seed)
put("samples", samples)
image = vae.decode(samples)
Image.fromarray((image.reshape(-1, *image.shape[-3:])[0, ..., :3].clamp(0, 1).detach().cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(os.path.join(args.out, "image.png"))
if "ref_raw0" in dump:
    dump["ref0_in"] = model.model.process_latent_in(dump["ref_raw0"]).contiguous()
save_file(dump, os.path.join(args.out, "dump.safetensors"))
print("evaluations", evals["n"])
