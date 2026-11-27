"""Run ComfyUI's SD 1.x or SDXL pipeline headlessly, dump what `diffusion --sd/--sdxl --dump` dumps, and compare.

usage: python sd_ref.py <out_dir> <checkpoint> [prompt] [--seed N] [--size W H] [--steps N] [--cfg X]
                        [--sampler NAME] [--scheduler NAME] [--negative TEXT] [--clip-layer N]
                        [--repeat N] [--compare rust.safetensors rust.png]

Writes <out_dir>/ref.safetensors (tokens, cond, pooled, noise, denoised0: the first evaluation's prediction of the
positive prompt) and <out_dir>/ref.png. With --compare it prints how far this engine's dump and image are from them.
--repeat samples that many times and prints each run's time; the later runs are the warm baseline.
Any ComfyUI runs SD 1.x and SDXL, so ref/ComfyUI and its venv do.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_env  # noqa: E402
COMFY = comfy_env.use_comfy("ref")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("checkpoint")
ap.add_argument("prompt", nargs="?", default="no humans, scenery, a lighthouse on a rocky coast at sunset, dramatic clouds, detailed")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--size", type=int, nargs=2, default=None)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=7.0)
ap.add_argument("--sampler", default="euler")
ap.add_argument("--scheduler", default="normal")
ap.add_argument("--negative", default="blurry, low quality")
ap.add_argument("--clip-layer", type=int)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--compare", nargs=2)
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.model_management
import comfy.sd
import comfy.sample
import comfy.samplers

torch.set_grad_enabled(False)

model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(args.checkpoint, output_vae=True, output_clip=True)
xl = model.model.adm_channels > 0
w, h = args.size or ((1024, 1024) if xl else (512, 512))
if args.clip_layer is not None:
    clip = clip.clone()
    clip.clip_layer(args.clip_layer)
os.makedirs(args.out, exist_ok=True)
dump = {}


def encode(text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return tokens, [[cond, {"pooled_output": pooled}]]


tokens, positive = encode(args.prompt)
_, negative = encode(args.negative)
dump["tokens"] = torch.tensor([t[0] for t in tokens["l"][0]], dtype=torch.int64)
dump["cond"] = positive[0][0][0].float()
if xl:
    dump["pooled"] = positive[0][1]["pooled_output"][0].float()

latent = torch.zeros([1, 4, h // 8, w // 8])
noise = comfy.sample.prepare_noise(latent, args.seed)
dump["noise"] = noise


def wrapper(apply_model, a):
    out = apply_model(a["input"], a["timestep"], **a["c"])
    # ComfyUI evaluates the prompts in separate calls when memory is short; take the positive one's
    if "denoised0" not in dump and 0 in a["cond_or_uncond"]:
        i = a["cond_or_uncond"].index(0)
        dump["denoised0"] = out[i:i + 1].float().cpu()
    return out


model = model.clone()
model.set_model_unet_function_wrapper(wrapper)
for run in range(args.repeat):
    t0 = time.perf_counter()
    samples = comfy.sample.sample(model, noise, args.steps, args.cfg, args.sampler, args.scheduler, positive, negative, latent, seed=args.seed)
    torch.cuda.synchronize()
    print(f"run {run}: sampling {time.perf_counter() - t0:.2f}s")
img = vae.decode(samples)
rgb = (img[0].clamp(0, 1).cpu().float().numpy() * 255).round().astype(np.uint8)
Image.fromarray(rgb).save(os.path.join(args.out, "ref.png"))
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
print(f"{'SDXL' if xl else 'SD 1.x'} {w}x{h}, unet {model.model.get_dtype()}, vae {vae.vae_dtype}: wrote {args.out}")

if args.compare:
    rs = load_file(args.compare[0])

    def stat(name, a, b):
        a, b = a.float().flatten(), b.float().flatten()
        if a.numel() != b.numel():
            print(f"{name:10s} size mismatch {a.numel()} vs {b.numel()}")
            return
        d = (a - b).abs()
        rel = d.norm() / a.norm().clamp_min(1e-12)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
        print(f"{name:10s} max|d|={d.max():.4g} mean|d|={d.mean():.4g} rel_l2={rel:.4g} cos={cos:.6f}")

    stat("cond", dump["cond"], rs["cond"])
    if "pooled" in dump:
        stat("pooled", dump["pooled"], rs["pooled"])
    stat("denoised0", dump["denoised0"], rs["denoised0"])
    a = np.asarray(Image.open(args.compare[1]).convert("RGB"), dtype=np.float64)
    mse = ((a - rgb.astype(np.float64)) ** 2).mean()
    print(f"image PSNR {10 * np.log10(255 ** 2 / max(mse, 1e-12)):.2f} dB")
