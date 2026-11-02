"""Record ComfyUI's guidance patches on small random predictions, through its real cfg_function and
pre-CFG hooks, for tests/cfg_tweaks_match_comfy.rs. No checkpoint is needed.

Run with SwarmUI's python_embeded, from anywhere:
    python tools/cfg_ref.py <SwarmUI root> > tests/cfg_golden.json
"""
import json
import os
import sys

root = sys.argv[1]
comfy_dir = os.path.join(root, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, comfy_dir)
os.chdir(comfy_dir)

import torch

import comfy.model_sampling
import comfy.samplers
from comfy_extras.nodes_cfg import CFGZeroStar
from comfy_extras.nodes_lumina2 import RenormCFG
from comfy_extras.nodes_model_advanced import RescaleCFG
from comfy_extras.nodes_tcfg import TCFG


class Model:
    """The parts of a ModelPatcher the patches touch: a flow model of 4 channels."""

    def __init__(self):
        self.model_options = {}
        self.model = self
        self.diffusion_model = self
        self.in_channels = 4

    def clone(self):
        return self

    def get_model_object(self, name):
        return comfy.model_sampling.CONST()

    def set_model_sampler_cfg_function(self, fn):
        self.model_options["sampler_cfg_function"] = fn

    def set_model_sampler_post_cfg_function(self, fn):
        self.model_options.setdefault("sampler_post_cfg_function", []).append(fn)

    def set_model_sampler_pre_cfg_function(self, fn):
        self.model_options.setdefault("sampler_pre_cfg_function", []).append(fn)


def apply(tweak):
    m = Model()
    if "rescale" in tweak:
        RescaleCFG().patch(m, tweak["rescale"])
    if "renorm" in tweak:
        RenormCFG.execute(m, *tweak["renorm"])
    if tweak.get("zero_star"):
        CFGZeroStar.execute(m)
    if tweak.get("tangential"):
        TCFG.execute(m)
    return m


def guide(m, x, cond, uncond, scale, sigma):
    out = [cond, uncond]
    for fn in m.model_options.get("sampler_pre_cfg_function", []):
        out = fn({"conds": [{}, {}], "conds_out": out, "cond_scale": scale, "timestep": sigma, "input": x, "sigma": sigma, "model": m, "model_options": m.model_options})
    return comfy.samplers.cfg_function(m, out[0], out[1], scale, x, sigma, model_options=m.model_options)


TWEAKS = [
    {"rescale": 0.7},
    {"rescale": 1.0},
    {"renorm": (100.0, 1.0)},
    {"renorm": (100.0, 0.5)},
    {"renorm": (0.1, 1.0)},
    {"zero_star": True},
    {"tangential": True},
    {"tangential": True, "rescale": 0.5, "zero_star": True},
]

torch.manual_seed(1)
cases = []
for tweak in TWEAKS:
    for scale, sigma in ((4.0, 0.8), (1.5, 0.3)):
        x, cond, uncond = (torch.randn(1, 4, 3, 5) for _ in range(3))
        sigma_t = torch.tensor([sigma])
        out = guide(apply(tweak), x, cond, uncond, scale, sigma_t)
        cases.append({"tweak": tweak, "scale": scale, "sigma": sigma, "x": x.flatten().tolist(), "cond": cond.flatten().tolist(),
                      "uncond": uncond.flatten().tolist(), "out": out.flatten().tolist()})
json.dump(cases, sys.stdout)
