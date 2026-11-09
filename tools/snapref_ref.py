"""Run ComfyUI's Anima pipeline with the SnapRef node headlessly, dumping tensors to compare with
`diffusion --anima --snapref ... --dump`.

usage: python snapref_ref.py <out_dir> <snapref node dir> <adapter.safetensors> <reference.png> [prompt]
                             [--seed N] [--size W H] [--steps N] [--cfg X] [--negative TEXT] [--mode cs|char]
                             [--strength X] [--fidelity X] [--latent ours.safetensors]

Run it with SwarmUI's python_embeded and ComfyUI (ref/ComfyUI is too old), with SWARM_DIR set to the SwarmUI folder
and MODELS_ROOT to the models folder. <snapref node dir> is the node package, e.g. the
`comfy_node/snapref` folder of the SwarmUI-SnapRef extension. `--strength` is the SwarmUI extension's: every
reference's weight. Dumps the reference stream's keys and values of the first and last blocks (from the node's
cache), the target stream's first and last block outputs on the first evaluation, the samples and the image.
"""
import argparse
import importlib.util
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("node")
ap.add_argument("adapter")
ap.add_argument("reference")
ap.add_argument("prompt", nargs="?", default="no humans, robot, sitting on a wooden park bench, autumn leaves, outdoors, sunny")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--sampler", default="er_sde")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--negative", default="")
ap.add_argument("--mode", default="cs")
ap.add_argument("--strength", type=float, default=1.0)
ap.add_argument("--fidelity", type=float, default=1.0)
ap.add_argument("--latent", help="a --dump of this engine: use its reference latent instead of ComfyUI's VAE encode")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.join(os.environ["SWARM_DIR"], "dlbackend", "comfy", "ComfyUI"))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

root = os.environ["MODELS_ROOT"]
DIT = os.path.join(root, "Stable-diffusion", "animaInt8Mxfp8_turboV11Int8.safetensors")
TE = os.path.join(root, "clip", "qwen_3_600m.safetensors")
VAE = os.path.join(root, "VAE", "qwen_image_vae.safetensors")
folder_paths.add_model_folder_path("loras", os.path.dirname(os.path.abspath(args.adapter)))

spec = importlib.util.spec_from_file_location("snapref", os.path.join(args.node, "__init__.py"),
                                              submodule_search_locations=[args.node])
snapref = importlib.util.module_from_spec(spec)
sys.modules["snapref"] = snapref
spec.loader.exec_module(snapref)

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu").contiguous().clone()


clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
cond = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
neg = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
model = comfy.sd.load_diffusion_model(DIT)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
dm = model.model.diffusion_model

encode_image = snapref._encode_image


def recorded_encode(*a, **kw):
    lat, size = encode_image(*a, **kw)
    if args.latent:
        from safetensors.torch import load_file
        lat = load_file(args.latent)["ref_latent"].reshape(lat.shape).to(lat)
    put("ref_latent", lat)
    return lat, size


snapref._encode_image = recorded_encode
img = torch.from_numpy(np.asarray(Image.open(args.reference).convert("RGB")).astype(np.float32) / 255.0)[None]
(patched,) = snapref.SnapRef().apply(model, clip, vae, img, os.path.basename(args.adapter), args.mode, False, 1.0,
                                     strength=0.0, weight_1=args.strength, fidelity=args.fidelity)

# the target stream's blocks on the first evaluation (the reference stream runs them too, before it)
evals = {"n": 0}


def block_hook(name):
    def h(mod, inp, out):
        if snapref._STATE["stream"] == "tgt" and evals["n"] == 0 and name not in dump:
            put(name, out)
        elif snapref._STATE["stream"] == "tgt" and evals["n"] == 1 and name + "_call1" not in dump:
            # with CFG, the other prompt's first evaluation
            put(name + "_call1", out)
    return h


dm.blocks[0].register_forward_hook(block_hook("block0_out"))
dm.blocks[len(dm.blocks) - 1].register_forward_hook(block_hook("blocklast_out"))
orig = dm._forward


def counted(*a, **kw):
    out = orig(*a, **kw)
    if evals["n"] == 0:
        put("dit_out", out)
    evals["n"] += 1
    return out


dm._forward = counted

w, h = args.size
latent = torch.zeros([1, 16, h // 8, w // 8])
latent = comfy.sample.fix_empty_latent_channels(patched, latent)
noise = comfy.sample.prepare_noise(latent, args.seed)
t0 = time.perf_counter()
samples = comfy.sample.sample(patched, noise, args.steps, args.cfg, args.sampler, args.scheduler, cond, neg, latent,
                              denoise=1.0, seed=args.seed)
torch.cuda.synchronize()
print(f"sample time {time.perf_counter() - t0:.3f}s")
put("samples", samples)

(ad,) = snapref._adapters.values()
(records, sizes), = ad.cache.values()
for name, b in (("ref_k_block0", 0), ("ref_k_last", len(records) - 1)):
    k, v = records[b]
    put(name, k)
    put(name.replace("_k_", "_v_"), v)

image = vae.decode(samples)
put("image", image)
save_file(dump, os.path.join(args.out, "ref.safetensors"))
arr = (image.detach()[0].reshape(-1, *image.shape[-3:])[0].clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(os.path.join(args.out, "ref.png"))
print("saved", {k: list(v.shape) for k, v in dump.items()})
