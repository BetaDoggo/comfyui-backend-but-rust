"""Run ComfyUI's MOSS Image pipeline headlessly and dump intermediate tensors for parity tests.

usage: python moss_ref.py <out_dir> [prompt] [--seed N] [--size W H] [--steps N] [--cfg X] [--shift X]
                          [--negative TEXT] [--repeat N] [--swarm-template]

--repeat runs the sampler that many times; the timings of the later runs are the warm baseline.
--swarm-template encodes with the Krea 2 template, which SwarmUI sends for MOSS Image.
"""
import argparse
import os
import sys
import time

COMFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI")
sys.path.insert(0, os.path.abspath(COMFY))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("prompt", nargs="?", default="a mountain lake at sunrise, pine forest on the far shore, low clouds, still water with reflections, landscape photograph, no people")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--shift", type=float, default=5.0)
ap.add_argument("--negative", default="blurry, low quality")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--swarm-template", action="store_true")
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

DIT = r"G:\stable-diffusion-webui-forge\models\Stable-diffusion\moss_image_sft_int8_convrot_attn.safetensors"
TE = r"G:\stable-diffusion-webui-forge\models\clip\qwen3.5_9b_int8_convrot.safetensors"
VAE = r"G:\stable-diffusion-webui-forge\models\VAE\flux2-vae.safetensors"
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu").contiguous().clone()


t0 = time.perf_counter()
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.MOSS_IMAGE)
extra = {"llama_template": KREA2_TEMPLATE} if args.swarm_template else {}
te = clip.cond_stage_model.qwen35_9b.transformer
te_hooks = []


def te_hook(name):
    def h(mod, inp, out):
        if name not in dump:
            put(name, out[0] if isinstance(out, tuple) else out)
    return h


te_hooks.append(te.model.layers[0].input_layernorm.register_forward_hook(te_hook("te_norm0")))
te_hooks.append(te.model.layers[0].linear_attn.register_forward_hook(te_hook("te_attn0")))
te_hooks.append(te.model.layers[0].register_forward_hook(te_hook("te_layer0")))
te_hooks.append(te.model.layers[3].register_forward_hook(te_hook("te_layer3")))
te_hooks.append(te.model.layers[31].register_forward_hook(te_hook("te_layer31")))

tokens = clip.tokenize(args.prompt, **extra)
key = next(iter(tokens))
put("ids", torch.tensor([t[0] for t in tokens[key][0]], dtype=torch.int64))
t1 = time.perf_counter()
cond = clip.encode_from_tokens_scheduled(tokens)
put("cond", cond[0][0])
print("cond", cond[0][0].shape, cond[0][0].dtype, f"encode {time.perf_counter() - t1:.3f}s, load+encode {time.perf_counter() - t0:.3f}s")
for h in te_hooks:
    h.remove()
t1 = time.perf_counter()
cond = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt, **extra))
print(f"encode again {time.perf_counter() - t1:.3f}s")
neg_tokens = clip.tokenize(args.negative, **extra)
put("neg_ids", torch.tensor([t[0] for t in neg_tokens[key][0]], dtype=torch.int64))
neg = clip.encode_from_tokens_scheduled(neg_tokens)
put("neg_cond", neg[0][0])

model = comfy.sd.load_diffusion_model(DIT)
dm = model.model.diffusion_model
print("dit dtype", next(dm.parameters()).dtype, "manual cast", model.model.manual_cast_dtype)
ms = model.get_model_object("model_sampling")
ms.set_parameters(shift=args.shift, multiplier=1.0)

calls = {"n": 0}
orig_forward = dm._forward


def hooked(x, timesteps, context, *a, **kw):
    out = orig_forward(x, timesteps, context, *a, **kw)
    i = calls["n"]
    if i == 0:
        put("dit_in_x", x)
        put("dit_in_ctx", context)
        put("dit_out", out)
    put(f"call{i}_t", timesteps)
    if i > 0 and i < 4:
        put(f"dit_in_x_call{i}", x)
        put(f"dit_out_call{i}", out)
    calls["n"] += 1
    return out


dm._forward = hooked


def mk(name):
    def h(mod, inp, out):
        if calls["n"] == 0 and name not in dump:
            put(name, out if torch.is_tensor(out) else out[0])
    return h


dm.video_embedder.register_forward_hook(mk("video_embedder_out"))
dm.text_embedder.register_forward_hook(mk("text_embedder_out"))
dm.time_embedding.register_forward_hook(mk("vec"))
dm.shared_block_modulation.register_forward_hook(mk("shared_mod"))
dm.video_head_modulation.register_forward_hook(mk("head_mod"))
dm.layers[0].register_forward_hook(mk("block0_out"))
dm.layers[1].register_forward_hook(mk("block1_out"))
dm.layers[len(dm.layers) - 1].register_forward_hook(mk("blocklast_out"))

w, h = args.size
latent = torch.zeros([1, 128, h // 16, w // 16])
latent = comfy.sample.fix_empty_latent_channels(model, latent)
noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)

for run in range(args.repeat):
    t1 = time.perf_counter()
    samples = comfy.sample.sample(model, noise, args.steps, args.cfg, "euler", "simple", cond, neg, latent,
                                  denoise=1.0, seed=args.seed)
    torch.cuda.synchronize()
    print(f"run {run}: sample time {time.perf_counter() - t1:.3f}s ({(time.perf_counter() - t1) / args.steps:.3f} s/step)")
put("samples", samples)
put("sigmas", comfy.samplers.calculate_sigmas(ms, "simple", args.steps))

vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
print("vae dtype", vae.vae_dtype)
t2 = time.perf_counter()
img = vae.decode(samples)
torch.cuda.synchronize()
print("vae time", time.perf_counter() - t2, img.shape)
t2 = time.perf_counter()
img = vae.decode(samples)
torch.cuda.synchronize()
print("vae time again", time.perf_counter() - t2)
put("image", img)
print("total", time.perf_counter() - t0)

save_file(dump, os.path.join(args.out, "ref.safetensors"))
from PIL import Image
import numpy as np
arr = (img.detach()[0].reshape(-1, *img.shape[-3:])[0].clamp(0, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr).save(os.path.join(args.out, "ref.png"))
print("saved")
