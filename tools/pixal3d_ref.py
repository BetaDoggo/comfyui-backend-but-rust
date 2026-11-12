"""ComfyUI's Pixal3D image-to-3D pipeline, as in Comfy-Org's example workflow, with every stage dumped.

Runs the stock nodes (ImageCropToMask, Pixal3DConditioning, the Trellis2 stage nodes, four KSamplers, the shape and
texture VAE decodes, PaintMesh, SaveGLB) on an RGBA image whose alpha is the object mask. The background-removal
and MoGe nodes of the example are left out: the alpha is the mask, and the field of view is a parameter.

Run with SwarmUI's ComfyUI (ref/ComfyUI predates Trellis2):

usage: python_embeded/python.exe pixal3d_ref.py <out_dir> --image object.png [--fov 49.13] [--resolution 1536]
           [--ss-steps 12] [--shape-steps 20] [--hr-steps 12] [--tex-steps 12]
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
ap.add_argument("--fov", type=float, default=49.13)
ap.add_argument("--resolution", type=int, default=1536)
ap.add_argument("--ss-steps", type=int, default=12)
ap.add_argument("--shape-steps", type=int, default=20)
ap.add_argument("--hr-steps", type=int, default=12)
ap.add_argument("--tex-steps", type=int, default=12)
ap.add_argument("--stop-after", choices=["cond", "ss", "shape", "hr", "mesh", "tex"], default=None)
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
# ComfyUI's own memory management (DynamicVRAM), as the app runs it; without it this pipeline runs out of VRAM
assert comfy_aimdo.control.init_devices((d.index, int(comfy.cli_args.args.vram_headroom * 1024 ** 3)) for d in comfy.model_management.get_all_torch_devices())
comfy.memory_management.aimdo_enabled = True
import comfy.model_patcher
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
import comfy.clip_vision
import comfy.sd
import comfy.utils
import nodes
from comfy_extras.nodes_custom_sampler import CFGOverride
from comfy_extras.nodes_images import ImageCropToMask
from comfy_extras.nodes_mesh_postprocess import PaintMesh
from comfy_extras.nodes_model_advanced import ModelSamplingSD3, RescaleCFG
from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
from comfy_extras import nodes_trellis2 as t2

MODELS = os.environ["MODELS_ROOT"]
DIT = os.path.join(MODELS, r"diffusion_models\Pixal3D\pixal3d_int8_convrot.safetensors")
DINO = os.path.join(MODELS, r"clip_vision\dino_v3_L_naf_fp32.safetensors")
SHAPE_VAE = os.path.join(MODELS, r"VAE\Pixal3D\trellis_2_shape_vae_bf16.safetensors")
TEX_VAE = os.path.join(MODELS, r"VAE\Pixal3D\trellis_2_texture_vae_bf16.safetensors")

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
(composite,) = result(ImageCropToMask.execute(image, mask, 1024, 1024, 1.1, 0, "#000000"))
put("composite", composite)

clip_vision = comfy.clip_vision.load(DINO)
dino = clip_vision.model


def hook(module, name):
    module.register_forward_hook(lambda _m, _in, out: put(name, out[0] if isinstance(out, tuple) else out) if name not in dump else None)


hook(dino.embeddings, "dino_embed")
hook(dino.layer[0], "dino_layer0")
hook(dino.layer[23], "dino_layer23")
positive, negative = result(t2.Pixal3DConditioning.execute(clip_vision, composite, args.fov))
pack = positive[0][1]["proj_feat_pack"]
put("global_512", positive[0][0])
put("global_1024", positive[0][1]["embeds"])
put("ss_proj", positive[0][1]["trellis2_proj_feats"])
put("transform_matrix", pack["transform_matrix"])
for stage, entry in pack["stages"].items():
    put(f"{stage}_lr", entry["feature_map"])
    if entry["feature_map_hr"] is not None:
        hr = entry["feature_map_hr"]
        put(f"{stage}_hr_sub", hr[:, :, ::8, ::8])
        put(f"{stage}_hr_ch0", hr[:, :4])
save("cond")

model = comfy.sd.load_diffusion_model(DIT)
first_calls = {}
inner = model.model.diffusion_model
orig_forward = inner.forward


def forward(x, timestep, context, **kw):
    out = orig_forward(x, timestep, context, **kw)
    mode = kw.get("trellis2_generation_mode", "structure_generation")
    if mode not in first_calls:
        first_calls[mode] = True
        put(f"{mode}_x", x)
        put(f"{mode}_t", timestep)
        put(f"{mode}_context", context)
        put(f"{mode}_embeds", kw["embeds"])
        if kw.get("trellis2_proj_feats") is not None:
            put(f"{mode}_proj", kw["trellis2_proj_feats"])
        put(f"{mode}_out", out)
    return out


inner.forward = forward
ss_model = ModelSamplingSD3().patch(result(RescaleCFG().patch(result(CFGOverride.execute(model, 1.0, 0.667, 1.0))[0], 0.7))[0], 5)[0]
shape_model = result(RescaleCFG().patch(result(CFGOverride.execute(model, 1.0, 0.769, 1.0))[0], 0.5))[0]
shape_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(SHAPE_VAE))
tex_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(TEX_VAE))


def ksample(m, seed, steps, cfg, scheduler, pos, neg, latent):
    return nodes.common_ksampler(m, seed, steps, cfg, "euler", scheduler, pos, neg, latent)[0]


(empty,) = result(t2.EmptyTrellis2LatentStructure.execute(1))
ss = ksample(ss_model, 56, args.ss_steps, 7.5, "normal", positive, negative, empty)
put("ss_latent", ss["samples"])
(voxel,) = result(t2.VaeDecodeStructureTrellis2.execute(ss, shape_vae, "32"))
put("ss_voxel", voxel.data)
save("ss")

p1, n1, lat1 = result(t2.Trellis2ShapeStage.execute(positive, negative, voxel))
put("shape512_coords", lat1["coords"])
put("shape512_proj", p1[0][1]["trellis2_proj_feats"])
shape512 = ksample(shape_model, 42, args.shape_steps, 7.5, "normal", p1, n1, lat1)
put("shape512_latent", shape512["samples"])
save("shape")

p2, n2, lat2 = result(t2.Trellis2UpsampleStage.execute(p1, n1, shape512, shape_vae, args.resolution))
put("hr_coords", lat2["coords"])
put("hr_proj", p2[0][1]["trellis2_proj_feats"])
hr = ksample(shape_model, 42, args.hr_steps, 7.5, "simple", p2, n2, lat2)
put("hr_latent", hr["samples"])
save("hr")

mesh, subs = result(t2.VaeDecodeShapeTrellis.execute(hr, shape_vae))
put("mesh_vertices", mesh.vertices[0])
put("mesh_faces", mesh.faces[0])
for i, sub in enumerate(subs):
    put(f"sub{i}_coords", sub.coords)
    put(f"sub{i}_feats", sub.feats)
save("mesh")

p3, n3, lat3 = result(t2.Trellis2TextureStage.execute(p2, n2, hr))
put("tex_proj", p3[0][1]["trellis2_proj_feats"])
tex = ksample(model, 43, args.tex_steps, 1.0, "normal", p3, n3, lat3)
put("tex_latent", tex["samples"])
(colors,) = result(t2.VaeDecodeTextureTrellis.execute(tex, tex_vae, subs))
put("voxel_coords", colors.data)
put("voxel_colors", colors.voxel_colors)
(painted,) = result(PaintMesh.execute(mesh, colors))
put("vertex_colors", painted.vertex_colors[0])
with open(os.path.join(args.out, "ref.glb"), "wb") as f:
    f.write(mesh_item_to_glb_bytes(painted, 0))
save("tex")
print("done")
