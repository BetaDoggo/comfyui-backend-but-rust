"""ComfyUI's Qwen Image 2.1 with every stage dumped: VAE, vision tower, text encoder, DiT and sampler.

Run with SwarmUI's ComfyUI and its Python (ref/ComfyUI predates Qwen Image 2.1). The prompt is encoded the way
SwarmUI's graph does it: SwarmTextEncodeAdvanced with the images, then a VAEEncode and ReferenceLatent per image.

usage: python_embeded/python.exe qwen21_ref.py <out_dir> [--image a.png [--image b.png]] [--prompt "..."]
           [--size W H] [--steps N] [--cfg X] [--seed N] [--negative TEXT] [--vae-only] [--no-cache]

--image    prompt image (goes through the vision tower, then the VAE as a reference latent). Same size for all.
--no-cache runs the DiT over the whole sequence every step (QwenImage21Cache device=off), so block outputs can be
           hooked. The result is the same: the cache is lossless.
"""
import argparse
import os
import sys
import time

SWARM = os.environ.get("SWARM_DIR", r"G:\StableSwarmUI")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(SWARM, "src", "BuiltinExtensions", "ComfyUIBackend", "ExtraNodes"))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--prompt", default="no people, a stone lighthouse on a rocky coast at sunset, calm sea, photograph")
ap.add_argument("--negative", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--shift", type=float, default=None, help="mu of the flux time shift; default is the model's 0.69")
ap.add_argument("--vae-only", action="store_true")
ap.add_argument("--keep-alpha", action="store_true", help="VAE-encode the images with their alpha, as QwenImage21 does, not just RGB")
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--lora", action="append", default=[], help="path[:strength], applied to the DiT through ComfyUI's DynamicVRAM path, as main.py does")
ap.add_argument("--capture", action="append", default=[], help="a patched module such as diffusion_model.transformer_blocks.3.attn.to_q.weight: dump its weight as ComfyUI requantizes it")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--trace-block", type=int, action="append", default=[], help="dump what goes in and out of the parts of this block (needs --no-cache)")
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
import comfy.model_patcher
import comfy.quant_ops
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import node_helpers
from SwarmComfyCommon.SwarmText import SwarmTextEncodeAdvanced

MODELS = os.environ.get("MODELS_ROOT", r"G:\stable-diffusion-webui-forge\models")
DIT = os.path.join(MODELS, r"Stable-diffusion\rayQwen21_v1Int8Convrot.safetensors")
TE = os.path.join(MODELS, r"clip\qwen3vl_8b_int8_convrot.safetensors")
VAE = os.path.join(MODELS, r"VAE\QwenImage\qwen_image_2.1_vae_bf16.safetensors")

os.makedirs(args.out, exist_ok=True)
dump = {}

if args.lora:
    # LoRAs go through DynamicVRAM as in main.py; without it ComfyUI takes the legacy path, which seeds the rounding differently
    assert comfy_aimdo.control.init_devices((d.index, int(comfy.cli_args.args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
    comfy.memory_management.aimdo_enabled = True
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
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


def put(name, t):
    dump[name] = t.detach().to("cpu", copy=True).contiguous()
    print(f"{name}: {tuple(t.shape)} {t.dtype}")


def load(path):
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    return torch.from_numpy(a)[None]


def save_image(t, path):
    a = t.detach().float().cpu().reshape(-1, *t.shape[-3:])[0]
    mode = "RGBA" if a.shape[-1] == 4 else "RGB"
    Image.fromarray((a.clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8), mode).save(path)


t0 = time.perf_counter()
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(VAE))
print("vae dtype", vae.vae_dtype, "channels", vae.latent_channels, "out", vae.output_channels, "pad", vae.pad_channel_value)

images = [load(p) for p in args.image]
refs = []
for i, img in enumerate(images):
    put(f"image{i}", img)
    raw = vae.encode(img if args.keep_alpha else img[..., :3])
    put(f"ref_raw{i}", raw)
    refs.append(raw)
    # the engine's decode of the same latent, to compare against a round trip
    back = vae.decode(raw)
    put(f"ref_back{i}", back)
    save_image(back, os.path.join(args.out, f"roundtrip{i}.png"))
if args.vae_only:
    # a decode of a fixed random latent exercises every layer without a sampler
    g = torch.Generator().manual_seed(1)
    z = torch.randn([1, 64, 24, 32], generator=g) * 0.7
    put("vae_in", z)
    out = vae.decode(z)
    put("vae_out", out)
    save_image(out, os.path.join(args.out, "vae_out.png"))
    save_file(dump, os.path.join(args.out, "dump.safetensors"))
    raise SystemExit

clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
cond_model = clip.cond_stage_model
lm = getattr(cond_model, cond_model.clip).transformer
visual = lm.visual


def hook(module, name, first=True):
    def h(_m, _in, out):
        if not first or name not in dump:
            put(name, out[0] if isinstance(out, tuple) else out)
    module.register_forward_hook(h)


hook(visual.patch_embed, "vision_patch")
hook(visual.blocks[0], "vision_block0")
hook(visual.blocks[len(visual.blocks) - 1], f"vision_block{len(visual.blocks) - 1}")
hook(visual.merger, "vision_merged")
for i, m in enumerate(visual.deepstack_merger_list):
    hook(m, f"vision_deepstack{i}")
hook(lm.model.layers[0], "te_layer0")
hook(lm.model.layers[1], "te_layer1")
hook(lm.model.layers[34], "te_layer34")
hook(lm.model.layers[35], "te_layer35")

last = []
orig_encode = clip.encode_from_tokens_scheduled


def encode_from_tokens_scheduled(tokens, *a, **k):
    last.append(tokens)
    return orig_encode(tokens, *a, **k)


clip.encode_from_tokens_scheduled = encode_from_tokens_scheduled


def text_encode(prompt):
    w, h = args.size
    batch = torch.cat([i[..., :3] * i[..., 3:] + (1 - i[..., 3:]) for i in images]) if images else None
    return SwarmTextEncodeAdvanced().encode(clip, args.steps, prompt, w, h, w, h, images=batch)[0]


def token_ids(tokens):
    row = tokens["qwen3vl_8b"][0]
    return torch.tensor([t[0] if isinstance(t[0], (int, float)) else -1 for t in row], dtype=torch.int64)


positive = text_encode(args.prompt)
put("ids", token_ids(last[0]))
put("cond", positive[0][0])
print("positive extras", {k: (v if not torch.is_tensor(v) else tuple(v.shape)) for k, v in positive[0][1].items()})
if "image_slots" in positive[0][1]:
    put("image_slots", torch.tensor(positive[0][1]["image_slots"], dtype=torch.int64))
negative = text_encode(args.negative)
put("neg_ids", token_ids(last[-1]))
put("neg_cond", negative[0][0])
for r in refs:
    positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [r]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [r]}, append=True)

model = comfy.sd.load_diffusion_model(DIT)
for entry in args.lora:
    path, _, s = entry.rpartition(":") if entry.rsplit(":", 1)[-1].replace(".", "").replace("-", "").isdigit() else (entry, "", "1")
    model, _ = comfy.sd.load_lora_for_models(model, clip, comfy.utils.load_torch_file(path, safe_load=True), float(s), 0.0)
    print("lora", path, s, len(model.patches), "patched weights")
if args.no_cache:
    model = model.clone()
    model.model_options["transformer_options"]["qwen_image21_cache"] = {"device": "off", "dtype": "default"}
dm = model.model.diffusion_model
ms = model.get_model_object("model_sampling")
if args.shift is not None:
    ms.set_parameters(shift=args.shift, multiplier=1.0)
print("model sampling", type(ms).__name__, "shift", getattr(ms, "shift", None))

evals = {"n": 0}
orig_forward = dm._forward


def forward(x, timesteps, context, ref_latents=None, image_slots=None, *a, **k):
    out = orig_forward(x, timesteps, context, ref_latents, image_slots, *a, **k)
    if evals["n"] == 0:
        put("dit_in", x)
        put("dit_t", timesteps)
        put("dit_ctx", context)
        put("dit_out", out)
        for i, r in enumerate(ref_latents or []):
            put(f"dit_ref{i}", r)
        print("sigma", float(timesteps[0]), "image_slots", image_slots)
    evals["n"] += 1
    return out


dm._forward = forward
if args.trace_block:
    import comfy.ldm.qwen_image21.model as q21
    gated = {"n": 0}
    orig_gated = q21._gated_residual

    def gated_residual(x, y, gate, prefix_len):
        out = orig_gated(x, y, gate, prefix_len)
        b, part = divmod(gated["n"], 2)
        gated["n"] += 1
        if evals["n"] == 0 and b in args.trace_block:
            put(f"{'x_attn' if part == 0 else 'x_out'}_b{b}", out)
            put(f"{'y_attn' if part == 0 else 'y_mlp'}_b{b}", y)
        return out

    q21._gated_residual = gated_residual
    for b in args.trace_block:
        blk = dm.transformer_blocks[b]
        blk.register_forward_pre_hook(lambda _m, inp, b=b: put(f"x_in_b{b}", inp[0]) if evals["n"] == 0 else None)
        blk.attn.to_out[0].register_forward_pre_hook(lambda _m, inp, b=b: put(f"attn_b{b}", inp[0]) if evals["n"] == 0 else None)
if args.no_cache:
    for i in (0, 1, len(dm.transformer_blocks) - 1):
        def mk(idx):
            def h(_m, _in, out):
                if evals["n"] == 0 and f"block{idx}" not in dump:
                    put(f"block{idx}", out)
            return h
        dm.transformer_blocks[i].register_forward_hook(mk(i))
    dm.txt_in.register_forward_hook(lambda _m, _in, out: put("txt_in", out) if "txt_in" not in dump else None)
    dm.time_text_embed.register_forward_hook(lambda _m, _in, out: put("temb", out) if "temb" not in dump else None)
    dm.norm_out.register_forward_hook(lambda _m, _in, out: put("norm_out", out) if "norm_out" not in dump else None)

w, h = args.size
latent = torch.zeros([1, 4, h // 16, w // 16], device=comfy.model_management.intermediate_device())
latent = comfy.sample.fix_empty_latent_channels(model, latent)
print("latent", tuple(latent.shape))
noise = comfy.sample.prepare_noise(latent, args.seed)
put("noise", noise)
for run in range(args.repeat):
    t1 = time.perf_counter()
    samples = comfy.sample.sample(model, noise, args.steps, args.cfg, "euler", "simple", positive, negative, latent, seed=args.seed)
    torch.cuda.synchronize()
    print(f"run {run}: {time.perf_counter() - t1:.2f}s ({(time.perf_counter() - t1) / args.steps:.3f} s/step)")
put("samples", samples)
put("sigmas", comfy.samplers.calculate_sigmas(ms, "simple", args.steps))
image = vae.decode(samples)
put("image", image)
save_image(image, os.path.join(args.out, "image.png"))
save_file(dump, os.path.join(args.out, "dump.safetensors"))
print("evaluations", evals["n"], f"total {time.perf_counter() - t0:.1f}s")
