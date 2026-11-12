"""Run ComfyUI's real LoRA path (DynamicVRAM, as main.py sets it up) and dump the result for parity tests.

Run with SwarmUI's ComfyUI: python_embeded/python.exe lora_ref.py ...

usage: python lora_ref.py <krea|anima|moss> <out_dir> <lora[:strength]>... [--prompt TEXT] [--negative TEXT] [--seed N] [--size W H]
                          [--steps N] [--cfg X] [--shift X] [--sampler NAME] [--capture MODULE]... [--clip-strength X]

--capture saves a module's weight as ComfyUI requantizes it: `<module>.float` (bf16 after the patches),
`<module>.q` and `<module>.scale`. Prompts must be scenery with no people.
"""
import argparse
import os
import sys
import time

SWARM = os.environ.get("SWARM_DIR") or sys.exit("set SWARM_DIR to the SwarmUI folder")
sys.path.insert(0, os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI"))

ap = argparse.ArgumentParser()
ap.add_argument("family", choices=["krea", "anima", "moss"])
ap.add_argument("out")
ap.add_argument("loras", nargs="*")
ap.add_argument("--prompt")
ap.add_argument("--negative", default="blurry, low quality")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int)
ap.add_argument("--cfg", type=float)
ap.add_argument("--shift", type=float)
ap.add_argument("--sampler")
ap.add_argument("--capture", action="append", default=[])
ap.add_argument("--clip-strength", type=float, default=0.0)
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

M = os.environ["MODELS_ROOT"]
FAMILY = {
    "krea": dict(dit=M + r"\Stable-diffusion\kroma-v0.3.1-turbo-opd-W8A8.safetensors", te=M + r"\clip\qwen3vl_4b.safetensors",
                 vae=M + r"\VAE\qwen_image_vae.safetensors", clip="KREA2", ch=16, down=8, steps=8, cfg=1.0, shift=1.15, sampler="euler",
                 prompt="a red fox sitting in a snowy forest, golden hour, photograph"),
    "anima": dict(dit=M + r"\Stable-diffusion\animaInt8Mxfp8_turboV11Int8.safetensors", te=M + r"\clip\qwen_3_600m.safetensors",
                  vae=M + r"\VAE\qwen_image_vae.safetensors", clip="STABLE_DIFFUSION", ch=16, down=8, steps=8, cfg=1.0, shift=3.0, sampler="er_sde",
                  prompt="no humans, scenery, mountain lake at sunrise, pine forest, clouds, reflection, masterpiece, best quality"),
    "moss": dict(dit=M + r"\Stable-diffusion\moss_image_sft_int8_convrot_attn.safetensors", te=M + r"\clip\qwen3.5_9b_int8_convrot.safetensors",
                 vae=M + r"\VAE\flux2-vae.safetensors", clip="MOSS_IMAGE", ch=128, down=16, steps=20, cfg=4.0, shift=5.0, sampler="euler",
                 prompt="a stone lighthouse on a sea cliff at dawn, crashing waves, no people"),
}
f = FAMILY[args.family]
steps = args.steps or f["steps"]
cfg = f["cfg"] if args.cfg is None else args.cfg
shift = args.shift or f["shift"]
sampler = args.sampler or f["sampler"]
prompt = args.prompt or f["prompt"]

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
import comfy.quant_ops
import comfy.sample
import comfy.sd
import comfy.utils
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic

os.makedirs(args.out, exist_ok=True)
dump = {}

seeds = {comfy.utils.string_to_seed(m): m for m in args.capture}
requantize = comfy.quant_ops.QuantizedTensor.requantize_from_float


def capture(self, tensor, **kw):
    y = requantize(self, tensor, **kw)
    m = seeds.get(kw.get("stochastic_rounding"))
    if m is not None and f"{m}.q" not in dump:
        dump[f"{m}.float"] = tensor.detach().cpu().float()
        dump[f"{m}.q"] = y._qdata.detach().cpu().clone()
        dump[f"{m}.scale"] = y._params.scale.detach().cpu().clone()
    return y


comfy.quant_ops.QuantizedTensor.requantize_from_float = capture

t0 = time.perf_counter()
clip = comfy.sd.load_clip(ckpt_paths=[f["te"]], clip_type=getattr(comfy.sd.CLIPType, f["clip"]))
model = comfy.sd.load_diffusion_model(f["dit"])
print("patcher", type(model).__name__, "dynamic", model.is_dynamic())
model.get_model_object("model_sampling").set_parameters(shift=shift, **({} if args.family == "krea" else {"multiplier": 1.0}))
for l in args.loras:
    path, _, s = l.rpartition(":") if l.rsplit(":", 1)[-1].replace(".", "").replace("-", "").isdigit() else (l, "", "1")
    model, clip = comfy.sd.load_lora_for_models(model, clip, comfy.utils.load_torch_file(path, safe_load=True), float(s), args.clip_strength)
cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
neg = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
print(f"load {time.perf_counter() - t0:.2f}s, {len(model.patches)} patched weights")

w, h = args.size
latent = torch.zeros([1, f["ch"], h // f["down"], w // f["down"]])
latent = comfy.sample.fix_empty_latent_channels(model, latent)
noise = comfy.sample.prepare_noise(latent, args.seed)
dump["noise"] = noise
t1 = time.perf_counter()
samples = comfy.sample.sample(model, noise, steps, cfg, sampler, "simple", cond, neg, latent, denoise=1.0, seed=args.seed)
torch.cuda.synchronize()
print(f"sample {time.perf_counter() - t1:.2f}s ({(time.perf_counter() - t1) / steps * 1e3:.0f} ms/step)")
first = samples.detach().cpu()
t1 = time.perf_counter()
samples = comfy.sample.sample(model, noise, steps, cfg, sampler, "simple", cond, neg, latent, denoise=1.0, seed=args.seed)
torch.cuda.synchronize()
print(f"sample again {time.perf_counter() - t1:.2f}s ({(time.perf_counter() - t1) / steps * 1e3:.0f} ms/step)")
dump["samples"] = samples.detach().cpu()
print("second run vs first: rel", ((dump["samples"] - first).norm() / first.norm()).item())

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(f["vae"]))
img = vae.decode(samples)
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
from PIL import Image
import numpy as np
arr = (img.detach()[0].reshape(-1, *img.shape[-3:])[0].clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(os.path.join(args.out, "ref.png"))
print("saved")
