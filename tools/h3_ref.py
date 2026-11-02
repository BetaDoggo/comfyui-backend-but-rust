"""Run ComfyUI's MiniMax H3 path (DynamicVRAM, as main.py sets it up) and dump the result for parity tests.

Run with SwarmUI's ComfyUI: python_embeded/python.exe h3_ref.py <out_dir> [options]

Prompts must be scenery with no people. Dumps ref.safetensors: noise (video, audio), context, tags, velocity of
the first step when --trace is given, samples, and the decoded frames.
"""
import argparse
import os
import sys
import time

SWARM = os.environ.get("SWARM_DIR", r"G:\StableSwarmUI")
sys.path.insert(0, os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI"))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--prompt", default="a stone lighthouse on a sea cliff at dawn, waves crashing on the rocks, seagulls calling, no people")
ap.add_argument("--negative", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[512, 512])
ap.add_argument("--length", type=int, default=5)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--sampler", default="res_multistep")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--lora", default="minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16_comfyui.safetensors")
ap.add_argument("--strength", type=float, default=1.0)
ap.add_argument("--no-vae", action="store_true")
ap.add_argument("--load-only", action="store_true")
ap.add_argument("--trace", action="store_true", help="keep tensors from inside the first evaluation and stop")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

M = os.environ["MODELS_ROOT"]
DIT = M + r"\Stable-diffusion\minimax_h3_fl2va_pruned_w6a8.safetensors"
TE = M + r"\clip\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = M + r"\VAE\MiniMaxH3\minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = M + r"\VAE\MiniMaxH3\minimax_h3_audio_vae_fp32.safetensors"

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
import comfy.nested_tensor
import comfy.sample
import comfy.sd
import comfy.utils
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
import nodes
sys.path.insert(0, os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI", "comfy_extras"))
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

os.makedirs(args.out, exist_ok=True)
dump = {}
t0 = time.perf_counter()
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.MINIMAX)
print(f"te load {time.perf_counter() - t0:.1f}s", flush=True)
t0 = time.perf_counter()
model = comfy.sd.load_diffusion_model(DIT)
print(f"dit load {time.perf_counter() - t0:.1f}s, patcher {type(model).__name__}, shift {model.get_model_object('model_sampling').shift}", flush=True)
if args.lora:
    t0 = time.perf_counter()
    sd = comfy.utils.load_torch_file(os.path.join(M, "Lora", args.lora), safe_load=True)
    model, clip = comfy.sd.load_lora_for_models(model, clip, sd, args.strength, 0.0)
    print(f"lora {time.perf_counter() - t0:.1f}s, {len(model.patches)} patched weights", flush=True)
if args.load_only:
    sys.exit(0)

t0 = time.perf_counter()
cond = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
neg = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
torch.cuda.synchronize()
print(f"encode {time.perf_counter() - t0:.1f}s, tokens {cond[0][0].shape}", flush=True)
dump["context"] = cond[0][0].detach().cpu().float()

w, h = args.size
latent, frames = _empty_av_latent(w, h, args.length)
samples = latent["samples"]
print("frames", frames, "shapes", [tuple(t.shape) for t in samples.unbind()], flush=True)
noise = comfy.sample.prepare_noise(samples, args.seed)
for i, t in enumerate(noise.unbind()):
    dump[f"noise{i}"] = t.detach().cpu()


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

    def keep(name, inputs=False):
        def hook(mod, inp, out):
            if name + ".out" in dump:
                return
            if inputs:
                dump[name + ".in"] = inp[0].detach().cpu().float()
            o = out[0] if isinstance(out, (list, tuple)) else out
            dump[name + ".out"] = o.detach().cpu().float()
        return hook

    dm.blocks[0].attn.register_forward_hook(keep("b0.attn", True))
    dm.blocks[0].mlp.register_forward_hook(keep("b0.mlp", True))
    dm.blocks[0].attn.out_proj.register_forward_hook(keep("b0.out_proj", True))
    dm.blocks[0].register_forward_hook(keep("b0"))
    dm.blocks[1].register_forward_hook(keep("b1"))
    dm.blocks[49].register_forward_hook(keep("b49"))
    dm.final_layer.register_forward_hook(keep("final", True))
    dump["ctx"] = dm.preprocess_text_embeds(cond[0][0].to(comfy.model_management.get_torch_device(), torch.bfloat16)).detach().cpu().float()
for run in ("first", "again"):
    t1 = time.perf_counter()
    out = comfy.sample.sample(model, noise, args.steps, args.cfg, args.sampler, args.scheduler, cond, neg, samples, denoise=1.0, seed=args.seed)
    torch.cuda.synchronize()
    print(f"sample {run} {time.perf_counter() - t1:.2f}s ({(time.perf_counter() - t1) / args.steps:.2f} s/step)", flush=True)
    if args.trace:
        break
for i, t in enumerate(out.unbind()):
    dump[f"samples{i}"] = t.detach().cpu()
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
if args.no_vae:
    sys.exit(0)

v, a = out.unbind()
t1 = time.perf_counter()
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
print(f"vae load {time.perf_counter() - t1:.1f}s", flush=True)
t1 = time.perf_counter()
frames = vae.decode(v)
print(f"vae decode {time.perf_counter() - t1:.2f}s {tuple(frames.shape)}", flush=True)
dump["frames"] = frames.detach().cpu().float()
t1 = time.perf_counter()
avae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(AUDIO_VAE))
print(f"audio vae load {time.perf_counter() - t1:.1f}s", flush=True)
t1 = time.perf_counter()
wave = avae.decode(a)
print(f"audio decode {time.perf_counter() - t1:.2f}s {tuple(wave.shape)}", flush=True)
dump["wave"] = wave.detach().cpu().float()
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
from PIL import Image
import numpy as np
arr = (frames.detach().reshape(-1, *frames.shape[-3:]).clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
for i, f in enumerate(arr):
    Image.fromarray(f).save(os.path.join(args.out, f"ref_{i:03d}.png"))
print("saved", flush=True)
