"""ComfyUI's RelayVSR on a short clip, as the RelayVSR extension's stage runs it, with the stages dumped.

RelayVSRConditioning, a 1-step euler/simple KSampler at CFG 1, and VSRDecode. Run with SwarmUI's ComfyUI and its
python_embeded (ref/ComfyUI predates RelayVSR):

usage: python_embeded/python.exe relayvsr_ref.py <out_dir> --image a.png [--size W H] [--frames N] [--pan PX] [--seed N]
           [--detail X] [--dit path] [--vae path] [--repeat N]

The clip is --frames frames (17 by default: keyframes 0, 15 and 16) panning across the picture by --pan pixels a
frame, after resizing it to --size with PIL's Lanczos. Use a scenery picture with no people. The first and last frames
of the input and the output are saved as PNGs. Check the dump with
`diffusion --relayvsr-check <out_dir>/dump.safetensors --dit <dit> --vae <vae>`.
"""
import argparse
import os
import sys
import time

SWARM = os.environ.get("SWARM_DIR") or sys.exit("set SWARM_DIR to the SwarmUI folder")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)
MODELS = os.environ.get("MODELS_ROOT") or sys.exit("set MODELS_ROOT to the models folder")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--image", required=True)
ap.add_argument("--size", type=int, nargs=2, default=None)
ap.add_argument("--frames", type=int, default=17)
ap.add_argument("--pan", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--detail", type=float, default=1.0)
ap.add_argument("--dit", default=os.path.join(MODELS, "diffusion_models", "relayvsr_x2_lightx2v_int8_convrot.safetensors"))
ap.add_argument("--vae", default=os.path.join(MODELS, "VAE", "relayvsr_flash_vae_x2_fp32.safetensors"))
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--probe-blocks", default="all", help="comma-separated blocks whose first chunk to dump, or all")
ap.add_argument("--no-decode", action="store_true", help="stop after sampling, for long clips")
ap.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto", help="the keyframe model's compute dtype; ComfyUI picks fp16 on its own")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]] + {"bf16": ["--bf16-unet"], "fp16": ["--fp16-unet"]}.get(args.dtype, [])

import comfy_aimdo.control
comfy_aimdo.control.init()

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
from comfy_extras.nodes_wan import RelayVSRConditioning, VSRDecode

torch.set_grad_enabled(False)
os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu", copy=True).float().contiguous()
    print(f"{name}: {tuple(t.shape)} {t.dtype}")


def save_image(t, path):
    a = t.detach().float().cpu()
    Image.fromarray((a.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8), "RGB").save(path)


im = Image.open(args.image).convert("RGB")
w, h = args.size or im.size
# a pan: the picture a little wider, a window sliding across it
wide = im.resize((w + args.pan * (args.frames - 1), h), Image.LANCZOS)
frames = [wide.crop((f * args.pan, 0, f * args.pan + w, h)) for f in range(args.frames)]
image = torch.from_numpy(np.stack([np.asarray(f).astype(np.float32) / 255.0 for f in frames]))
put("frames", image)
save_image(image[0], os.path.join(args.out, "input_first.png"))
save_image(image[-1], os.path.join(args.out, "input_last.png"))

if args.dtype != "auto":
    # with a quantized file ComfyUI computes in the first of the model's supported dtypes, whatever the flags say
    import comfy.supported_models
    first = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    comfy.supported_models.WAN22_RelayVSR.supported_inference_dtypes = [first, torch.float16, torch.bfloat16, torch.float32]
model = comfy.sd.load_diffusion_model(args.dit)
dm = model.model.diffusion_model
print("dit dtype", model.model.get_dtype(), "manual cast", model.model.manual_cast_dtype, "latent stride", dm.latent_stride)
positive, latent = RelayVSRConditioning.execute(model, image).result
latent = latent["samples"]
put("latent", latent)

evals = {"n": 0}
orig_forward = dm.forward_orig


def forward(x, t, context, *a, **k):
    out = orig_forward(x, t, context, *a, **k)
    if evals["n"] == 0:
        put("dit_in", x)
        put("dit_t", t)
        put("dit_out", out)
        put("lq_pixels", k["lq_pixels"])
    evals["n"] += 1
    return out


dm.forward_orig = forward


def once(name):
    def hook(_m, _in, out):
        if evals["n"] == 0 and name not in dump:
            put(name, out)
    return hook


dm.lq_proj_in.register_forward_hook(once("lq_proj"))
probed = range(len(dm.blocks)) if args.probe_blocks == "all" else [int(b) for b in args.probe_blocks.split(",")]
for i in probed:
    dm.blocks[i].register_forward_hook(once(f"block{i}"))

noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)
for run in range(args.repeat):
    t1 = time.perf_counter()
    samples = comfy.sample.sample(model, noise, 1, 1.0, "euler", "simple", positive, positive, latent, seed=args.seed)
    torch.cuda.synchronize()
    print(f"run {run}: sampling {time.perf_counter() - t1:.3f}s")
put("samples", samples)
if args.no_decode:
    save_file(dump, os.path.join(args.out, "dump.safetensors"))
    sys.exit(0)

sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
vae = comfy.sd.VAE(sd=sd, metadata=metadata)
print("vae dtype", vae.vae_dtype)
t1 = time.perf_counter()
decoded = VSRDecode.execute(vae, {"samples": samples}, image, args.detail).result[0]
torch.cuda.synchronize()
print(f"decode {time.perf_counter() - t1:.3f}s")
put("decoded", decoded)
put("detail", torch.tensor([args.detail]))
save_image(decoded[0], os.path.join(args.out, "comfy_first.png"))
save_image(decoded[-1], os.path.join(args.out, "comfy_last.png"))
save_file(dump, os.path.join(args.out, "dump.safetensors"))
