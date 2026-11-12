"""ComfyUI's TripoSplat / Idol image-to-splat pipeline, as in Comfy-Org's Idol example workflow, with every stage dumped.

Runs the stock nodes (TripoSplatPreprocessImage, TripoSplatConditioning, KSampler, VAEDecodeTripoSplat, and
SaveGaussianSplat's .ply writer) on an RGBA image whose alpha is the object mask. The example's background removal is
left out: the alpha is the mask. The first model evaluation is dumped with its inputs, so a port can be compared on
the same tensors, and the decode is dumped from the sampled latent.

Run with SwarmUI's ComfyUI (ref/ComfyUI predates TripoSplat), MODELS_ROOT set to the models folder:

usage: python_embeded/python.exe triposplat_ref.py <out_dir> --image object.png [--seed 46] [--steps 20] [--cfg 3]
           [--gaussians 262144] [--decode-seed 0] [--dit diffusion_models/idol_fp16.safetensors]
"""
import argparse
import os
import sys

SWARM = os.environ.get("SWARM_DIR") or sys.exit("set SWARM_DIR to the SwarmUI folder")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--image", required=True)
ap.add_argument("--seed", type=int, default=46)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--gaussians", type=int, default=262144)
ap.add_argument("--decode-seed", type=int, default=0)
ap.add_argument("--dit", default=r"diffusion_models\idol_fp16.safetensors")
ap.add_argument("--dino", default=r"clip_vision\dino_v3_vit_h_bf16.safetensors")
ap.add_argument("--flux-vae", default=r"VAE\flux2-vae.safetensors")
ap.add_argument("--decoder", default=r"VAE\idol_decoder_fp16.safetensors")
ap.add_argument("--stop-after", choices=["cond", "sample"], default=None)
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

import comfy_aimdo.control
comfy_aimdo.control.init()

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.memory_management
import comfy.model_management
assert comfy_aimdo.control.init_devices((d.index, int(comfy.cli_args.args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
comfy.memory_management.aimdo_enabled = True
import comfy.model_patcher
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
import comfy.clip_vision
import comfy.sd
import comfy.utils
import nodes
from comfy.ldm.triposplat import vae as ts_vae
from comfy_extras import nodes_triposplat as ts
from comfy_extras.nodes_gaussian_splat import _gaussian_ply_bytes

MODELS = os.environ["MODELS_ROOT"]
path = lambda p: os.path.join(MODELS, p)

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu", copy=True).contiguous()
    print(f"{name}: {tuple(t.shape)} {t.dtype}")


def save(stage):
    save_file(dump, os.path.join(args.out, "ref.safetensors"))
    if args.stop_after == stage:
        print(f"stopped after {stage}")
        sys.exit(0)


def result(out):
    return out.result if hasattr(out, "result") else out


rgba = np.asarray(Image.open(args.image).convert("RGBA")).astype(np.float32) / 255.0
image = torch.from_numpy(rgba[..., :3])[None]
mask = torch.from_numpy(rgba[..., 3])[None]
(prepared,) = result(ts.TripoSplatPreprocessImage.execute(image, mask, 1, 1024))
put("image", prepared)

clip_vision = comfy.clip_vision.load(path(args.dino))
dino = clip_vision.model
dino.layer[0].register_forward_hook(lambda _m, _in, out: put("dino_layer0", out[0] if isinstance(out, tuple) else out) if "dino_layer0" not in dump else None)
flux_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path(args.flux_vae)))
positive, negative, latent = result(ts.TripoSplatConditioning.execute(clip_vision, flux_vae, prepared))
put("feature1", positive[0][0])
put("reference", positive[0][1]["reference_latents"][0])
save("cond")

model = comfy.sd.load_diffusion_model(path(args.dit))
inner = model.model.diffusion_model
orig_forward = inner._forward
calls = []


def forward(x, t, context=None, ref_latents=None, transformer_options={}, **kw):
    out = orig_forward(x, t, context, ref_latents, transformer_options, **kw)
    # the first evaluation of cond and of uncond, batched together or not: tell them apart by the context
    for i in range(context.shape[0]):
        name = "cond" if context[i].abs().sum() > 0 else "uncond"
        if name not in calls:
            calls.append(name)
            put(f"eval_{name}_z", x[0][i])
            put(f"eval_{name}_cam", x[1][i])
            put(f"eval_{name}_t", t[i:i + 1])
            put(f"eval_{name}_out_z", out[0][i])
            put(f"eval_{name}_out_cam", out[1][i])
    return out


inner._forward = forward
samples = nodes.common_ksampler(model, args.seed, args.steps, args.cfg, "dpmpp_2m", "simple", positive, negative, latent)[0]
lat, cam = samples["samples"].unbind()
put("latent", lat)
put("camera", cam)
save("sample")

decoder = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path(args.decoder)))
orig_sample = ts_vae.OctreeProbabilityFixedlenDecoder.sample


def sample(*a, **kw):
    r = orig_sample(*a, **kw)
    put("points", r["points"])
    return r


ts_vae.OctreeProbabilityFixedlenDecoder.sample = staticmethod(sample)
decoder.first_stage_model.gs.register_forward_hook(lambda _m, _in, out: put("gs_features", out["features"]))
(splat,) = result(ts.VAEDecodeTripoSplat.execute(samples, decoder, args.gaussians, args.decode_seed))
for name in ["positions", "scales", "rotations", "opacities", "sh"]:
    put(f"splat_{name}", getattr(splat, name)[0])
with open(os.path.join(args.out, "ref.ply"), "wb") as f:
    f.write(_gaussian_ply_bytes(splat.positions[0], splat.scales[0], splat.rotations[0], splat.opacities[0], splat.sh[0]))
save("decode")
print("done")
