"""Run ComfyUI's MiniMax-H3-Image path (DynamicVRAM, as main.py sets it up), dump the result, and compare it with a
`diffusion --h3-image --dump` file.

Run with SwarmUI's python_embeded and a ComfyUI that has MiniMax-H3-Image (Comfy-Org/ComfyUI#17655), set by COMFY_DIR:

    python_embeded/python.exe h3_image_ref.py <out_dir> [options]
    python h3_image_ref.py --cmp <out_dir>/ref.safetensors <ours.safetensors>

Prompts must be scenery with no people. Dumps ref.safetensors: noise, context, the reference latents of --image
pictures (ref0, ref1, ...), the first evaluation's input and denoised output, the hidden rows of blocks 0, 1 and 49 with
--trace (which stops after that evaluation), samples, and the decoded image. With --image the size defaults, as the node
does, to the first picture's.
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?")
ap.add_argument("--cmp", nargs=2, metavar=("REF", "OURS"))
ap.add_argument("--prompt", default="a stone lighthouse on a sea cliff at dawn, waves crashing on the rocks below, soft golden light, no people")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=None, help="width height; 1024 1024, or the first --image's size")
ap.add_argument("--image", action="append", default=[], help="a PNG to edit or refer to, <Picture n> in the prompt (repeatable)")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--sampler", default="euler")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--no-vae", action="store_true")
ap.add_argument("--trace", action="store_true", help="keep tensors from inside the first evaluation and stop")
args, _ = ap.parse_known_args()

if args.cmp:
    import torch
    from safetensors.torch import load_file

    r, o = load_file(args.cmp[0]), load_file(args.cmp[1])

    def rel(a, b):
        return ((a.float().flatten() - b.float().flatten()).norm() / b.float().flatten().norm()).item()

    for k in sorted(k for k in r if k.startswith("ref")):
        if k in o:
            print(f"{k}: latent rel {rel(o[k], r[k]):.3e}")
    if "context" in o:
        print(f"context rel {rel(o['context'], r['context']):.3e}, tokens {r['context'].shape[1]}")
    t = r["context"].shape[1]
    for name in ("ctx", "b0.out", "b1.out", "b49.out"):
        if name in o and name in r:
            a, b = o[name].float().view(-1, 5376), r[name].float().view(-1, 5376)
            print(f"{name}: all {rel(a, b):.3e}, text {rel(a[:t], b[:t]):.3e}, image {rel(a[t:], b[t:]):.3e}")
    if "denoised0" in o and "step0_out" in r:
        print(f"step 0 at sigma {r['step0_sigma'].flatten()[0].item():.4f}: denoised rel {rel(o['denoised0'], r['step0_out']):.3e}")
    print(f"latent rel {rel(o['samples'], r['samples']):.3e}")
    if "image" in r and "image" in o:
        a, b = o["image"].float().flatten(), r["image"].float().flatten()
        print(f"image psnr {(-10 * torch.log10(((a - b) ** 2).mean())).item():.2f} dB")
    sys.exit(0)

COMFY = os.environ.get("COMFY_DIR") or os.path.join(os.environ.get("SWARM_DIR") or sys.exit("set COMFY_DIR, or SWARM_DIR to the SwarmUI folder"), "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)
sys.argv = [sys.argv[0]]

M = os.environ["MODELS_ROOT"]
DIT = M + r"\Stable-diffusion\minimax_h3_image_pruned_int8_convrot.safetensors"
TE = M + r"\clip\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = M + r"\VAE\MiniMaxH3\minimax_h3_image_vae_fp16.safetensors"

import comfy_aimdo.control
comfy_aimdo.control.init()

import torch
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.memory_management
import comfy.model_management
assert comfy_aimdo.control.init_devices((d.index, int(comfy.cli_args.args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
comfy.memory_management.aimdo_enabled = True
import comfy.model_patcher
import comfy.sample
import comfy.sd
import comfy.utils
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
import nodes  # noqa
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

os.makedirs(args.out, exist_ok=True)
dump = {}
t0 = time.perf_counter()
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.MINIMAX)
model = comfy.sd.load_diffusion_model(DIT)
print(f"load {time.perf_counter() - t0:.1f}s, {type(model.model).__name__}", flush=True)

t0 = time.perf_counter()
w, h = args.size or ((0, 0) if args.image else (1024, 1024))
pictures = {}
if args.image:
    from PIL import Image
    import numpy as np
    for i, p in enumerate(args.image):
        pictures[f"image_{i + 1}"] = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE)) if pictures else None
res = MiniMaxH3ImageConditioning.execute(clip, args.prompt, w, h, vae=vae, images=pictures or None)
cond, latent = res.args[0], res.args[1]
for i, ref in enumerate(cond[0][1].get("minimax_refs", [])):
    dump[f"ref{i}"] = ref["latent"].detach().cpu().float()
neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
torch.cuda.synchronize()
print(f"encode {time.perf_counter() - t0:.1f}s, tokens {cond[0][0].shape}", flush=True)
dump["context"] = cond[0][0].detach().cpu().float()
samples = latent["samples"]
noise = comfy.sample.prepare_noise(samples, args.seed)
dump["noise"] = noise.detach().cpu()


def capture(apply_model, a):
    out = apply_model(a["input"], a["timestep"], **a["c"])
    if "step0_in" not in dump:
        dump["step0_in"] = a["input"].detach().cpu().float()
        dump["step0_sigma"] = a["timestep"].detach().cpu().float()
        dump["step0_out"] = out.detach().cpu().float()
    return out


model.set_model_unet_function_wrapper(capture)
if args.trace:
    dm = model.model.diffusion_model

    def keep(name):
        def hook(mod, inp, out):
            if name not in dump:
                o = out[0] if isinstance(out, (list, tuple)) else out
                dump[name] = o.detach().cpu().float()
        return hook

    for i in (0, 1, 49):
        dm.blocks[i].register_forward_hook(keep(f"b{i}.out"))

for run in ("first", "again"):
    t1 = time.perf_counter()
    out = comfy.sample.sample(model, noise, args.steps, args.cfg, args.sampler, args.scheduler, cond, neg, samples, denoise=1.0, seed=args.seed)
    torch.cuda.synchronize()
    print(f"sample {run} {time.perf_counter() - t1:.2f}s ({(time.perf_counter() - t1) / args.steps:.2f} s/step)", flush=True)
    if args.trace:
        break
dump["samples"] = out.detach().cpu()
print("sigmas", model.get_model_object("model_sampling").sigma(torch.tensor([1000.0, 500.0])).tolist(), flush=True)
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
if args.no_vae:
    sys.exit(0)

vae = vae or comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
with torch.inference_mode():
    for run in ("first", "again"):
        t1 = time.perf_counter()
        img = vae.decode(out)
        torch.cuda.synchronize()
        print(f"vae decode {run} {time.perf_counter() - t1:.2f}s {tuple(img.shape)}", flush=True)
img = img.reshape(-1, *img.shape[-3:])[0]
dump["image"] = img.detach().cpu().float()
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
from PIL import Image
import numpy as np
Image.fromarray((img.clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)).save(os.path.join(args.out, "ref.png"))
print("saved", flush=True)
