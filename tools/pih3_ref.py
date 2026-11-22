"""Run ComfyUI's MiniMax H3 with a pi-H3 DWD adapter through ComfyUI-piFlow's own nodes, and dump the result.

Run with SwarmUI's ComfyUI: python_embeded/python.exe pih3_ref.py <out_dir> [options]

ComfyUI-piFlow (https://github.com/Lakonik/ComfyUI-piFlow) is read from PIFLOW_DIR, by default ref/ComfyUI-piFlow next
to this folder. The adapter loads as `Load LakonLab Adapter` loads it, the sigmas come from `pi-Flow Scheduler`
(max_raw_t 0.93 for 2 steps, 0.96 otherwise, final_step_size_scale 0.6), the sampler is plain euler at CFG 1, and the
frames are `DWD Decode`'s.

Prompts must be scenery with no people. Dumps ref.safetensors: noise (video, audio), sigmas, the first evaluation's input
and output (step0_in, step0_out), the DWD head's normed features (dwd_features) and Sol tokens before its layers
(dwd_tokens), samples, and the DWD frames [F, H, W, 3] in 0..1 (frames).
"""
import argparse
import importlib.util
import os
import sys
import time

SWARM = os.environ.get("SWARM_DIR") or sys.exit("set SWARM_DIR to the SwarmUI folder")
COMFY = os.path.join(SWARM, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)
PIFLOW = os.environ.get("PIFLOW_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI-piFlow")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--prompt", default="a stone lighthouse on a sea cliff at dawn, waves crashing on the rocks, seagulls calling, no people")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--size", type=int, nargs=2, default=[512, 512])
ap.add_argument("--length", type=int, default=5)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--adapter", default="pi_h3_2step_dwd_adapter_int8_convrot.safetensors")
ap.add_argument("--no-audio", action="store_true")
ap.add_argument("--weights", nargs="+", help="save these modules' weights after the adapter's patches, before requantization "
                "(<module>.float, as `diffusion` writes them for a `$LORA_DUMP/<module>.want`), and stop")
args, _ = ap.parse_known_args()
sys.argv = [sys.argv[0]]

M = os.environ["MODELS_ROOT"]
DIT = M + r"\Stable-diffusion\minimax_h3_fl2va_pruned_w6a8.safetensors"
TE = M + r"\clip\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
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
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
import nodes  # noqa: F401, sets up folder_paths and the node registry
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

# ComfyUI-piFlow as the package ComfyUI would import from custom_nodes
spec = importlib.util.spec_from_file_location("comfyui_piflow", os.path.join(PIFLOW, "__init__.py"), submodule_search_locations=[PIFLOW])
sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules[spec.name])
from comfyui_piflow.lakonlab_loaders import ADAPTER_KEY_RULES, DWD_HEAD_PATCHES  # noqa: E402
import comfyui_piflow.modules.loader_utils as lu  # noqa: E402
from comfyui_piflow.modules.loader_utils import load_adapter_module  # noqa: E402
from comfyui_piflow.modules.architectures.dwd_head import get_dwd_pixels  # noqa: E402
from comfyui_piflow.piflow_scheduler import get_piflow_schedule  # noqa: E402



def patch_model_with_adapter(model, adapter_sd, key_mapping, adapter_strength=1.0):
    """ComfyUI-piFlow's, with the shape check on logical shapes: it compares a w6a8 weight's packed shape, and refuses
    the w6a8 base. ComfyUI's set patch then requantizes each replaced weight to w6, as on any quantized weight."""
    full_sd, lora_sd = lu.split_adapter_state_dict(adapter_sd, model.model.model_config.unet_config["image_model"])
    diffusion_model = model.model.diffusion_model
    sd = {}
    for key, value in full_sd.items():
        if key in key_mapping:
            lu.convert_diffusers_to_comfyui(sd, value, key_mapping[key])
        else:
            layer, postfix = lu._quantization_layer_from_key(key)
            if postfix is not None:
                key = lu._mapped_weight_key(key_mapping, layer + ".weight")[:-len("weight")] + postfix
            sd[key] = value
    model_sd = diffusion_model.state_dict()

    def in_model(key):
        layer, postfix = lu._quantization_layer_from_key(key)
        return (key if postfix is None else layer + ".weight") in model_sd
    ignored = [k for k in sd if not in_model(k)]
    assert not ignored, ignored
    for layer in {k[:-len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant")}:
        old = comfy.utils.get_attr(diffusion_model, layer)
        linear = comfy.ops.mixed_precision_ops().Linear(old.in_features, old.out_features, bias=False, device="cpu")
        load_adapter_module(linear, {k[len(layer) + 1:]: sd.pop(k) for k in list(sd) if k.rpartition(".")[0] == layer and k.rpartition(".")[2] != "bias"})
        sd[layer + ".weight"] = linear.weight.dequantize()
    for key, value in sd.items():
        mod = comfy.utils.get_attr(diffusion_model, key.rpartition(".")[0])
        shape = (mod.out_features, mod.in_features) if key.endswith(".weight") and hasattr(mod, "in_features") else tuple(model_sd[key].shape)
        assert tuple(value.shape) == shape, (key, tuple(value.shape), shape)
    model = model.clone()
    for key in sd:
        model.patches.pop("diffusion_model." + key, None)
    model.add_patches({"diffusion_model." + key: ("set", (value,)) for key, value in sd.items()}, 1.0)
    if len(lora_sd) > 0:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, adapter_strength, 0)
    return model


import comfy.ops  # noqa: E402
os.makedirs(args.out, exist_ok=True)
dump = {}
t0 = time.perf_counter()
model = comfy.sd.load_diffusion_model(DIT)
print(f"loaded in {time.perf_counter() - t0:.1f}s, shift {model.get_model_object('model_sampling').shift}", flush=True)

# LakonLabAdapterLoader.load_adapter, from a path
t0 = time.perf_counter()
image_model = model.model.model_config.unet_config.get("image_model")
adapter_sd = comfy.utils.load_torch_file(os.path.join(M, "Lora", args.adapter))
dwd_sd = {k: adapter_sd.pop(k) for k in list(adapter_sd) if k.startswith("dwd_")}
model = patch_model_with_adapter(model, adapter_sd, ADAPTER_KEY_RULES[image_model](model.model.model_config.unet_config), 1.0)
model = DWD_HEAD_PATCHES[image_model](model, dwd_sd, load_adapter_module)
print(f"adapter {time.perf_counter() - t0:.1f}s, {len(model.patches)} patched weights", flush=True)

if args.weights:
    import comfy.lora
    import comfy.model_patcher as mp
    for m in args.weights:
        key = f"diffusion_model.{m}.weight"
        weight, _, convert = mp.get_key_weight(model.model, key)
        base = weight.dequantize().float() if hasattr(weight, "dequantize") else weight.float()
        dump[f"{m}.float"] = comfy.lora.calculate_weight(model.patches[key], base.cuda(), key).to(torch.bfloat16).cpu()
        print(m, tuple(dump[f"{m}.float"].shape), flush=True)
    save_file(dump, os.path.join(args.out, "weights.safetensors"))
    sys.exit(0)
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=comfy.sd.CLIPType.MINIMAX)
head = model.additional_models["lakonlab_dwd"][0].model
def keep_features(m, i, o):
    dump.setdefault("dwd_features", o.detach().cpu().float())


head.dwd_norm_out.register_forward_hook(keep_features)
expand = head.dwd_pixel_refiner.expand


def keep_expand(*a, **k):
    x, geo, shape = expand(*a, **k)
    dump.setdefault("dwd_tokens", x.detach().cpu().float())
    return x, geo, shape


head.dwd_pixel_refiner.expand = keep_expand

cond = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
dump["context"] = cond[0][0].detach().cpu().float()
w, h = args.size
latent, frames = _empty_av_latent(w, h, args.length)
samples = latent["samples"]
print("frames", frames, "shapes", [tuple(t.shape) for t in samples.unbind()], flush=True)
noise = comfy.sample.prepare_noise(samples, args.seed)
for i, t in enumerate(noise.unbind()):
    dump[f"noise{i}"] = t.detach().cpu()

max_raw_t = 0.96 if args.steps > 2 else 0.93
sigmas = get_piflow_schedule(model.get_model_object("model_sampling"), args.steps, 0.0, max_raw_t, 0.6)
print("sigmas", sigmas.tolist(), flush=True)
dump["sigmas"] = sigmas.float()


def capture(apply_model, a):
    out = apply_model(a["input"], a["timestep"], **a["c"])
    if "step0_in" not in dump:
        dump["step0_in"] = a["input"].detach().cpu().float()
        dump["step0_sigma"] = a["timestep"].detach().cpu().float()
        dump["step0_out"] = out.detach().cpu().float()
    return out


model.set_model_unet_function_wrapper(capture)
dm = model.model.diffusion_model


def keep(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, (list, tuple)) else out
        dump.setdefault(name + ".out", o.detach().cpu().float())
    return hook


for i in (0, 1, 49):  # the first evaluation's, as `diffusion --dump` probes them
    dm.blocks[i].register_forward_hook(keep(f"b{i}"))
for run in ("first", "again"):
    t1 = time.perf_counter()
    out = comfy.sample.sample_custom(model, noise, 1.0, comfy.samplers.sampler_object("euler"), sigmas, cond, [], samples, seed=args.seed)
    torch.cuda.synchronize()
    print(f"sample {run} {time.perf_counter() - t1:.2f}s", flush=True)
for i, t in enumerate(out.unbind()):
    dump[f"samples{i}"] = t.detach().cpu()
pixels = get_dwd_pixels({"samples": out})
print("dwd frames", tuple(pixels.shape), flush=True)
dump["frames"] = pixels.detach().cpu().float()
if not args.no_audio:
    avae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(AUDIO_VAE))
    dump["wave"] = avae.decode(out.unbind()[1]).detach().cpu().float()
save_file({k: v.contiguous() for k, v in dump.items()}, os.path.join(args.out, "ref.safetensors"))
from PIL import Image
import numpy as np
arr = (pixels.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
for i, f in enumerate(arr):
    Image.fromarray(f).save(os.path.join(args.out, f"ref_{i:03d}.png"))
print("saved", flush=True)
