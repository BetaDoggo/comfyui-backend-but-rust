"""The Pixal3D tab's nodes for ComfyUI. comfy-rust implements the same nodes natively, so the SwarmUI tab works on either
backend.

Pixal3DImageTo3D: Comfy-Org's Pixal3D example workflow as one node, from the cut-out photo to a saved vertex-coloured
GLB. The stock nodes do the work: ImageCropToMask, Pixal3DConditioning, the Trellis2 stage nodes, four KSamplers with the
example's CFGOverride / RescaleCFG / ModelSamplingSD3 patches, the VAE decodes, DecimateMesh's simplification and PaintMesh.

TripoSplatImageToSplat: Comfy-Org's Idol example workflow as one node, from the cut-out photo to a saved 3D gaussian
splat (.ply), for TripoSplat or its fine-tune Idol: TripoSplatPreprocessImage, TripoSplatConditioning, a KSampler and
VAEDecodeTripoSplat, saved like SaveGaussianSplat.
"""
import os

import comfy.clip_vision
import comfy.sd
import comfy.utils
import folder_paths
import nodes
from comfy_extras import nodes_trellis2 as t2
from comfy_extras.nodes_custom_sampler import CFGOverride
from comfy_extras.nodes_images import ImageCropToMask
import comfy.model_management
from comfy_extras.mesh3d.postprocess.qem_decimate import QEMConfig, qem_simplify
from comfy_extras.nodes_mesh_postprocess import PaintMesh
from comfy_api.latest import Types
from comfy_extras.nodes_model_advanced import ModelSamplingSD3, RescaleCFG
from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes

_cache = {}


def _result(out):
    return out.result if hasattr(out, "result") else out


def _load(kind, name):
    key = (kind, name)
    if key not in _cache:
        if kind == "unet":
            _cache[key] = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", name))
        elif kind == "clip_vision":
            _cache[key] = comfy.clip_vision.load(folder_paths.get_full_path_or_raise("clip_vision", name))
        else:
            _cache[key] = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", name)))
    return _cache[key]


class Pixal3DImageTo3D:
    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "image": ("IMAGE",),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "clip_vision_name": (folder_paths.get_filename_list("clip_vision"),),
                "shape_vae_name": (vaes,),
                "texture_vae_name": (vaes,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "fov": ("FLOAT", {"default": 49.13, "min": 1.0, "max": 170.0, "step": 0.01, "tooltip": "Horizontal field of view of the photo, in degrees."}),
                "resolution": ("INT", {"default": 1536, "min": 1024, "max": 2048, "step": 128, "tooltip": "Voxel resolution of the refined shape. Higher is finer and slower."}),
                "structure_steps": ("INT", {"default": 12, "min": 1, "max": 200}),
                "shape_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "refine_steps": ("INT", {"default": 12, "min": 1, "max": 200}),
                "texture_steps": ("INT", {"default": 12, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "CFG of the geometry samplers; the texture sampler runs at 1."}),
                "texture": ("BOOLEAN", {"default": True}),
                "max_faces": ("INT", {"default": 700000, "min": 0, "max": 100000000, "tooltip": "Simplify the mesh to at most this many triangles; 0 keeps every one."}),
                "filename_prefix": ("STRING", {"default": "3d/pixal3d"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The object's mask. Without one, the image's alpha."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "3d"
    DESCRIPTION = "Pixal3D's example workflow in one node: crop to the mask, DINOv3 + NAF conditioning, the structure, shape, refine and texture samplers, the shape and texture decodes and PaintMesh. Saves a vertex-coloured GLB."

    def run(self, image, unet_name, clip_vision_name, shape_vae_name, texture_vae_name, seed, fov, resolution, structure_steps,
            shape_steps, refine_steps, texture_steps, cfg, texture, max_faces, filename_prefix, mask=None):
        image = image[:1]
        if mask is None:
            if image.shape[-1] != 4:
                raise ValueError("Pixal3D needs the object cut out: an image with transparency, or a mask")
            mask = image[..., 3]
        (composite,) = _result(ImageCropToMask.execute(image[..., :3], mask[:1], 1024, 1024, 1.1, 0, "#000000"))
        positive, negative = _result(t2.Pixal3DConditioning.execute(_load("clip_vision", clip_vision_name), composite, fov))
        model = _load("unet", unet_name)
        shape_vae = _load("vae", shape_vae_name)
        # CFGOverride reads its sigma range off the model before ModelSamplingSD3 changes the shift, as in the example
        ss_model = ModelSamplingSD3().patch(_result(RescaleCFG().patch(_result(CFGOverride.execute(model, 1.0, 0.667, 1.0))[0], 0.7))[0], 5)[0]
        shape_model = _result(RescaleCFG().patch(_result(CFGOverride.execute(model, 1.0, 0.769, 1.0))[0], 0.5))[0]

        def ksample(m, s, steps, c, scheduler, pos, neg, latent):
            return nodes.common_ksampler(m, s, steps, c, "euler", scheduler, pos, neg, latent)[0]

        (empty,) = _result(t2.EmptyTrellis2LatentStructure.execute(1))
        ss = ksample(ss_model, seed, structure_steps, cfg, "normal", positive, negative, empty)
        (voxel,) = _result(t2.VaeDecodeStructureTrellis2.execute(ss, shape_vae, "32"))
        p1, n1, lat1 = _result(t2.Trellis2ShapeStage.execute(positive, negative, voxel))
        shape512 = ksample(shape_model, seed, shape_steps, cfg, "normal", p1, n1, lat1)
        p2, n2, lat2 = _result(t2.Trellis2UpsampleStage.execute(p1, n1, shape512, shape_vae, resolution))
        hr = ksample(shape_model, seed, refine_steps, cfg, "simple", p2, n2, lat2)
        mesh, subs = _result(t2.VaeDecodeShapeTrellis.execute(hr, shape_vae))
        if max_faces > 0 and mesh.faces.shape[1] > max_faces:
            # DecimateMesh(placement_mode=midpoint), on the GPU like the node
            device = comfy.model_management.get_torch_device()
            v, f, _ = qem_simplify(mesh.vertices[0].to(device), mesh.faces[0].to(device), int(max_faces), colors=None, config=QEMConfig())
            mesh = Types.MESH(vertices=v.cpu()[None], faces=f.cpu()[None])
        if texture:
            p3, n3, lat3 = _result(t2.Trellis2TextureStage.execute(p2, n2, hr))
            tex = ksample(model, seed + 1, texture_steps, 1.0, "normal", p3, n3, lat3)
            (colors,) = _result(t2.VaeDecodeTextureTrellis.execute(tex, _load("vae", texture_vae_name), subs))
            (mesh,) = _result(PaintMesh.execute(mesh, colors))
        out_dir, stem, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        filename = f"{stem}_{counter:05}_.glb"
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(mesh_item_to_glb_bytes(mesh, 0))
        return {"ui": {"3d": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


class TripoSplatImageToSplat:
    @classmethod
    def INPUT_TYPES(cls):
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "image": ("IMAGE",),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "clip_vision_name": (folder_paths.get_filename_list("clip_vision"),),
                "vae_name": (vaes, {"tooltip": "The Flux 2 VAE, which encodes the image."}),
                "decoder_name": (vaes, {"tooltip": "TripoSplat's octree gaussian decoder."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "num_gaussians": ("INT", {"default": 262144, "min": 32768, "max": 1048576, "step": 32, "tooltip": "Gaussians to decode. 262144 matches the octree's point density; more oversamples the same points."}),
                "filename_prefix": ("STRING", {"default": "3d/idol"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The object's mask, 1 on the object. Without one, the image's alpha."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "3d"
    DESCRIPTION = "Comfy-Org's Idol example workflow in one node: TripoSplatPreprocessImage, TripoSplatConditioning (DINOv3 ViT-H and the Flux 2 VAE), the sampler and VAEDecodeTripoSplat. Saves a 3D gaussian splat (.ply). Works for TripoSplat and Idol."

    def run(self, image, unet_name, clip_vision_name, vae_name, decoder_name, seed, steps, cfg, num_gaussians, filename_prefix, mask=None):
        from comfy_extras import nodes_triposplat as ts
        from comfy_extras.nodes_gaussian_splat import _gaussian_ply_bytes
        if num_gaussians % 32 != 0:
            raise ValueError("num_gaussians must be a multiple of 32")
        image = image[:1]
        if mask is None:
            if image.shape[-1] != 4:
                raise ValueError("TripoSplat needs the object cut out: an image with transparency, or a mask")
            mask = image[..., 3]
        (prepared,) = _result(ts.TripoSplatPreprocessImage.execute(image[..., :3], mask[:1], 1, 1024))
        positive, negative, latent = _result(ts.TripoSplatConditioning.execute(_load("clip_vision", clip_vision_name), _load("vae", vae_name), prepared))
        samples = nodes.common_ksampler(_load("unet", unet_name), seed, steps, cfg, "dpmpp_2m", "simple", positive, negative, latent)[0]
        (splat,) = _result(ts.VAEDecodeTripoSplat.execute(samples, _load("vae", decoder_name), num_gaussians, seed))
        out_dir, stem, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        filename = f"{stem}_{counter:05}_.ply"
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(_gaussian_ply_bytes(splat.positions[0], splat.scales[0], splat.rotations[0], splat.opacities[0], splat.sh[0]))
        return {"ui": {"3d": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {"Pixal3DImageTo3D": Pixal3DImageTo3D, "TripoSplatImageToSplat": TripoSplatImageToSplat}
NODE_DISPLAY_NAME_MAPPINGS = {"Pixal3DImageTo3D": "Pixal3D Image to 3D", "TripoSplatImageToSplat": "TripoSplat / Idol Image to Splat"}
