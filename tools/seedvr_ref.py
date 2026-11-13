"""ComfyUI's SeedVR2 on one picture or a short clip, as SwarmUI's graph runs it, with every stage dumped.

SeedVR2Preprocess, VAEEncode, SeedVR2Conditioning, a 1-step euler/simple KSampler at CFG 1, VAEDecode and
SeedVR2PostProcessing. Run with SwarmUI's ComfyUI and its python_embeded (ref/ComfyUI predates SeedVR2):

usage: python_embeded/python.exe seedvr_ref.py <out_dir> --image a.png [--size W H] [--frames N] [--pan PX] [--seed N]
           [--steps N] [--cfg X] [--color lab|wavelet|adain|none] [--dit path] [--vae path] [--repeat N]
           [--chunk FRAMES --overlap N]

--size resizes the picture first with PIL's Lanczos (SwarmUI's SeedVR Upscale does it with ImageScale lanczos); the
resized picture is saved as input.png, so the engine can start from the same pixels. --frames N makes a clip of N
frames panning across the picture by --pan pixels a frame (SeedVR2Preprocess pads it to 4n + 1 frames). Check the dump
with `diffusion --seedvr-check <out_dir>/dump.safetensors`.

--chunk samples the clip as SeedVR Split Latent does: SeedVR2TemporalChunk in manual mode with that many pixel frames a
chunk and --overlap latent frames of overlap, each chunk conditioned and sampled with the same seed, then
SeedVR2TemporalMerge. Its dump's DiT stages are the first chunk's, so compare its final frames (`post`) instead.
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
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--pan", type=int, default=4)
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--overlap", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=1)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--color", default="lab")
ap.add_argument("--dit", default=os.path.join(MODELS, "diffusion_models", "SeedVR2", "seedvr2_3b_int8_convrot.safetensors"))
ap.add_argument("--vae", default=os.path.join(MODELS, "VAE", "SeedVR2", "seedvr2_ema_vae_fp16.safetensors"))
ap.add_argument("--repeat", type=int, default=1)
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
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess

torch.set_grad_enabled(False)
os.makedirs(args.out, exist_ok=True)
dump = {}


def put(name, t):
    dump[name] = t.detach().to("cpu", copy=True).contiguous()
    print(f"{name}: {tuple(t.shape)} {t.dtype}")


def save_image(t, path):
    a = t.detach().float().cpu().reshape(-1, *t.shape[-3:])[0]
    Image.fromarray((a.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8), "RGB").save(path)


im = Image.open(args.image).convert("RGB")
w, h = args.size or im.size
if args.frames > 1:
    # a pan: the picture a little wider, a window sliding across it
    wide = im.resize((w + args.pan * (args.frames - 1), h), Image.LANCZOS)
    frames = [wide.crop((f * args.pan, 0, f * args.pan + w, h)) for f in range(args.frames)]
else:
    frames = [im.resize((w, h), Image.LANCZOS) if args.size else im]
frames[0].save(os.path.join(args.out, "input.png"))
image = torch.from_numpy(np.stack([np.asarray(f).astype(np.float32) / 255.0 for f in frames]))
put("image", image)

t0 = time.perf_counter()
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(args.vae))
print("vae dtype", vae.vae_dtype)
pre = SeedVR2Preprocess.execute(image).result[0]
put("pre", pre)
latent = vae.encode(pre)
put("latent", latent)
back = vae.decode(latent)
put("roundtrip", back)

model = comfy.sd.load_diffusion_model(args.dit)
dm = model.model.diffusion_model
print("dit dtype", model.model.get_dtype(), "manual cast", model.model.manual_cast_dtype)
positive, negative = SeedVR2Conditioning.execute(model, {"samples": latent}).result

evals = {"n": 0}
orig_forward = dm.forward


def forward(x, timestep, context, *a, **k):
    out = orig_forward(x, timestep, context, *a, **k)
    if evals["n"] == 0:
        put("dit_in", x)
        put("dit_t", timestep)
        put("dit_out", out)
        put("dit_cond", k["condition"])
    evals["n"] += 1
    return out


dm.forward = forward


def once(name):
    def h(_m, _in, out):
        if evals["n"] == 0 and name not in dump:
            put(name, out[0] if isinstance(out, tuple) else out)
    return h


dm.txt_in.register_forward_hook(once("txt_in"))
dm.emb_in.register_forward_hook(once("emb"))
dm.vid_in.register_forward_hook(once("vid_in"))
def both(name):
    def h(_m, _in, out):
        if evals["n"] == 0 and name not in dump:
            put(name, out[0])
            put(name + "_txt", out[1])
    return h


for i in range(len(dm.blocks)):
    dm.blocks[i].register_forward_hook(both(f"block{i}"))
def pre2(name):
    def h(_m, inp):
        if evals["n"] == 0 and name not in dump:
            put(name, inp[0])
            put(name + "_txt", inp[1])
    return h


def out2(name):
    def h(_m, _in, out):
        if evals["n"] == 0 and name not in dump:
            put(name, out[0])
            put(name + "_txt", out[1])
    return h


dm.blocks[0].attn.proj_out.register_forward_pre_hook(pre2("block0_attn"))
dm.blocks[0].attn.proj_qkv.register_forward_hook(out2("block0_qkv"))

samples_in = {"samples": latent}
noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)



def sample_chunked():
    from comfy_extras.nodes_seedvr import SeedVR2TemporalChunk, SeedVR2TemporalMerge
    mode = {"chunking_mode": "manual", "frames_per_chunk": args.chunk}
    chunks, overlap = SeedVR2TemporalChunk.execute({"samples": latent}, args.overlap, mode).result
    print(f"{len(chunks)} chunks of {[c['samples'].shape[2] for c in chunks]} latent frames, overlap {overlap}")
    out = []
    for c in chunks:
        pos, neg = SeedVR2Conditioning.execute(model, c).result
        nz = comfy.sample.prepare_noise(c["samples"], args.seed)
        s = comfy.sample.sample(model, nz, args.steps, args.cfg, "euler", "simple", pos, neg, c["samples"], seed=args.seed)
        out.append({"samples": s})
    return SeedVR2TemporalMerge.execute(out, [overlap]).result[0]["samples"]


for run in range(args.repeat):
    t1 = time.perf_counter()
    if args.chunk:
        samples = sample_chunked()
    else:
        samples = comfy.sample.sample(model, noise, args.steps, args.cfg, "euler", "simple", positive, negative, latent, seed=args.seed)
    torch.cuda.synchronize()
    print(f"run {run}: sampling {time.perf_counter() - t1:.3f}s")
put("samples", samples)
t1 = time.perf_counter()
decoded = vae.decode(samples)
torch.cuda.synchronize()
print(f"decode {time.perf_counter() - t1:.3f}s")
put("decoded", decoded)
post = SeedVR2PostProcessing.execute(decoded, image, args.color).result[0]
put("post", post)
save_image(post, os.path.join(args.out, "comfy.png"))
save_file(dump, os.path.join(args.out, "dump.safetensors"))
print("evaluations", evals["n"], f"total {time.perf_counter() - t0:.1f}s")
