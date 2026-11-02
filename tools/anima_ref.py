"""Run ComfyUI's Anima pipeline headlessly and dump intermediate tensors for parity tests.

usage: python anima_ref.py <out_dir> [prompt] [--seed N] [--size W H] [--steps N] [--cfg X]
                           [--sampler er_sde] [--scheduler simple] [--negative TEXT] [--repeat N]

--repeat runs the sampler that many times; the timings of the later runs are the warm baseline.
"""
import argparse
import os
import sys
import time

COMFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI")
sys.path.insert(0, os.path.abspath(COMFY))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("prompt", nargs="?", default="no humans, scenery, mountain lake at sunrise, (pine forest:1.2), clouds, reflection, masterpiece, best quality")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--sampler", default="er_sde")
ap.add_argument("--scheduler", default="simple")
ap.add_argument("--negative", default="")
ap.add_argument("--repeat", type=int, default=1)
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

DIT = os.path.join(os.environ["MODELS_ROOT"], r"Stable-diffusion\animaInt8Mxfp8_turboV11Int8.safetensors")
TE = os.path.join(os.environ["MODELS_ROOT"], r"clip\qwen_3_600m.safetensors")
VAE = os.path.join(os.environ["MODELS_ROOT"], r"VAE\qwen_image_vae.safetensors")

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu").contiguous().clone()


# torch's CUDA generator, for the Philox port
g = torch.Generator(device="cuda")
g.manual_seed(args.seed)
put("randn_a", torch.randn(1000, generator=g, device="cuda", dtype=torch.float32))
put("randn_b", torch.randn(70000, generator=g, device="cuda", dtype=torch.float32))

t0 = time.perf_counter()
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
tokens = clip.tokenize(args.prompt)
put("qwen_ids", torch.tensor([t[0] for t in tokens["qwen3_06b"][0]], dtype=torch.int64))
put("t5_ids", torch.tensor([t[0] for t in tokens["t5xxl"][0]], dtype=torch.int64))
put("t5_weights", torch.tensor([t[1] for t in tokens["t5xxl"][0]], dtype=torch.float32))
cond = clip.encode_from_tokens_scheduled(tokens)
put("cond", cond[0][0])
print("cond", cond[0][0].shape, cond[0][0].dtype, "te time", time.perf_counter() - t0)
neg = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
put("neg_cond", neg[0][0])

model = comfy.sd.load_diffusion_model(DIT)
dm = model.model.diffusion_model
print("dit dtype", next(dm.parameters()).dtype, "manual cast", model.model.manual_cast_dtype)

calls = {"n": 0}
orig_forward = dm._forward


def hooked(x, timesteps, context, *a, **kw):
    out = orig_forward(x, timesteps, context, *a, **kw)
    i = calls["n"]
    if i == 0:
        put("dit_in_x", x)
        put("dit_in_ctx", context)
        put("dit_out", out)
    put(f"step{i}_t", timesteps)
    if i > 0:
        put(f"dit_in_x_step{i}", x)
    calls["n"] += 1
    return out


dm._forward = hooked


def mk(name):
    def h(mod, inp, out):
        if calls["n"] == 0 and name not in dump:
            put(name, out if torch.is_tensor(out) else out[0])
    return h


dm.llm_adapter.register_forward_hook(mk("adapter_out"))
dm.llm_adapter.blocks[0].register_forward_hook(mk("adapter_block0"))
dm.x_embedder.register_forward_hook(mk("x_embedder_out"))
dm.t_embedding_norm.register_forward_hook(mk("t_norm_out"))
dm.blocks[0].self_attn.register_forward_hook(mk("block0_self_attn"))
dm.blocks[0].cross_attn.register_forward_hook(mk("block0_cross_attn"))
dm.blocks[0].mlp.register_forward_hook(mk("block0_mlp"))
dm.blocks[0].register_forward_hook(mk("block0_out"))
dm.blocks[len(dm.blocks) - 1].register_forward_hook(mk("blocklast_out"))
dm.final_layer.register_forward_hook(mk("final_out"))

w, h = args.size
latent = torch.zeros([1, 16, h // 8, w // 8])
latent = comfy.sample.fix_empty_latent_channels(model, latent)
noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)

for run in range(args.repeat):
    t1 = time.perf_counter()
    samples = comfy.sample.sample(model, noise, args.steps, args.cfg, args.sampler, args.scheduler, cond, neg, latent,
                                  denoise=1.0, seed=args.seed)
    torch.cuda.synchronize()
    print(f"run {run}: sample time {time.perf_counter() - t1:.3f}s ({(time.perf_counter() - t1) / args.steps:.3f} s/step)")
put("samples", samples)

ms = model.get_model_object("model_sampling")
put("sigmas", comfy.samplers.calculate_sigmas(ms, "simple", args.steps))

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
