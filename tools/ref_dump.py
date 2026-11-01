"""Run ComfyUI's Krea2 pipeline headlessly and dump intermediate tensors for parity tests.

usage: python ref_dump.py <out_dir> [prompt] [--seed N] [--size W H] [--steps N]
"""
import argparse
import os
import sys
import time

COMFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI")
sys.path.insert(0, os.path.abspath(COMFY))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("prompt", nargs="?", default="a red fox sitting in a snowy forest, golden hour, photograph")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--no-dump-blocks", action="store_true")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

import torch
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.model_management
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
import nodes

DIT = r"G:\stable-diffusion-webui-forge\models\Stable-diffusion\kroma-v0.3.1-turbo-opd-W8A8.safetensors"
TE = r"G:\stable-diffusion-webui-forge\models\clip\qwen3vl_4b.safetensors"
VAE = r"G:\stable-diffusion-webui-forge\models\VAE\qwen_image_vae.safetensors"

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu").contiguous().clone()


t0 = time.perf_counter()
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.KREA2)
tokens = clip.tokenize(args.prompt)
ids = [t[0] for t in tokens["qwen3vl_4b"][0]]
put("token_ids", torch.tensor(ids, dtype=torch.int64))
cond = clip.encode_from_tokens_scheduled(tokens)
put("cond", cond[0][0])
print("cond", cond[0][0].shape, cond[0][0].dtype, "te time", time.perf_counter() - t0)

model = comfy.sd.load_diffusion_model(DIT)
dm = model.model.diffusion_model

calls = {"n": 0}
orig_forward = dm._forward


def hooked(x, timesteps, context, *a, **kw):
    out = orig_forward(x, timesteps, context, *a, **kw)
    i = calls["n"]
    if i == 0:
        put("dit_in_x", x)
        put("dit_in_t", timesteps)
        put("dit_in_ctx", context)
        put("dit_out", out)
    put(f"step{i}_t", timesteps)
    calls["n"] += 1
    return out


dm._forward = hooked

if not args.no_dump_blocks:
    # capture intermediate activations from the first call only
    def mk(name):
        def h(mod, inp, out):
            if calls["n"] == 0:
                put(name, out if torch.is_tensor(out) else out[0])
        return h
    dm.txtfusion.register_forward_hook(mk("txtfusion_out"))
    dm.txtmlp.register_forward_hook(mk("txtmlp_out"))
    dm.first.register_forward_hook(mk("first_out"))
    dm.tmlp.register_forward_hook(mk("tmlp_out"))
    dm.tproj.register_forward_hook(mk("tproj_out"))
    dm.blocks[0].register_forward_hook(mk("block0_out"))
    dm.blocks[0].attn.register_forward_hook(mk("block0_attn_out"))
    dm.blocks[0].mlp.register_forward_hook(mk("block0_mlp_out"))
    dm.blocks[len(dm.blocks) - 1].register_forward_hook(mk("blocklast_out"))

w, h = args.size
latent = torch.zeros([1, 4, h // 8, w // 8])
latent = comfy.sample.fix_empty_latent_channels(model, latent)
noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)
neg = [[torch.zeros_like(cond[0][0]), cond[0][1]]]

t1 = time.perf_counter()
samples = comfy.sample.sample(model, noise, args.steps, 1.0, "euler", "simple", cond, neg, latent,
                              denoise=1.0, seed=args.seed)
torch.cuda.synchronize()
print("sample time", time.perf_counter() - t1)
put("samples", samples)

ms = model.get_model_object("model_sampling")
sigmas = comfy.samplers.calculate_sigmas(ms, "simple", args.steps)
put("sigmas", sigmas)

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
t2 = time.perf_counter()
img = vae.decode(samples)
torch.cuda.synchronize()
print("vae time", time.perf_counter() - t2, img.shape)
put("image", img)
print("total", time.perf_counter() - t0)

save_file(dump, os.path.join(args.out, "ref.safetensors"))
from PIL import Image
import numpy as np
arr = (img.detach()[0].reshape(-1, *img.shape[-3:])[0].clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(os.path.join(args.out, "ref.png"))
print("saved")
