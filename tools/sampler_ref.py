"""Run ComfyUI's own samplers and schedulers on a mock model and write what they produce.

usage: python sampler_ref.py <out.json>

The JSON is tests/sampler_golden.json, which tests/samplers_match_comfy.rs checks the Rust ports against.
Everything from sampler_object() down is ComfyUI's code. Only calc_cond_batch is replaced, by the same
analytic model the Rust test uses, so no checkpoint is needed. It runs on CUDA like a real generation:
the default noise sampler changes its seed on the CPU.
"""
import json
import math
import os
import sys
from types import SimpleNamespace

COMFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI")
sys.path.insert(0, os.path.abspath(COMFY))
out_path = sys.argv[1]
sys.argv = [sys.argv[0]]

import numpy as np
import torch
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.model_sampling
import comfy.samplers
from comfy_extras.nodes_flux import get_schedule
from comfy_extras.nodes_ideogram4 import ideogram4_sigmas

N = (4, 8, 8)
NUMEL = 4 * 8 * 8
SEED = 42
KEPT = 32  # values of each final latent written out
DEVICE = "cuda"
TARGET = torch.tensor([math.sin(0.37 * i + 1.0) for i in range(NUMEL)], dtype=torch.float32, device=DEVICE).reshape((1,) + N)
AYS_SD1 = [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904, 0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582]


def short(values):
    return [float(f"{v:.9g}") for v in values]


def model_sampling(kind, shift):
    if kind == "flow":
        class M(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass
    else:
        class M(comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST):
            pass
    return M(SimpleNamespace(sampling_settings={"multiplier": 1.0, "shift": shift}))


evals = []


def calc_cond_batch(model, conds, x, timestep, model_options):
    s = timestep[0]
    evals.append(float(s))
    cond = TARGET * (0.8 + 0.2 * s) + 0.3 * torch.tanh(x) * (1.0 - s)
    uncond = 0.5 * TARGET + 0.1 * x
    return [cond, uncond if conds[1] is not None else torch.zeros_like(x)]


comfy.samplers.calc_cond_batch = calc_cond_batch


class Wrap:
    def __init__(self, ms, cfg):
        self.inner_model = SimpleNamespace(model_sampling=ms)
        self.model_patcher = SimpleNamespace(get_model_object=lambda name: ms, model=self)
        self.cfg = cfg

    def __call__(self, x, sigma, model_options={}, seed=None):
        return comfy.samplers.sampling_function(self.inner_model, x, sigma, "uncond", "cond", self.cfg, model_options, seed)


def ksampler_sigmas(ms, sampler, scheduler, steps):
    discard = sampler in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS
    s = comfy.samplers.calculate_sigmas(ms, scheduler, steps + 1 if discard else steps)
    return torch.cat([s[:-2], s[-1:]]) if discard else s


def swarm_sigmas(ms, scheduler, steps, lw, lh):
    if scheduler == "turbo":
        t = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[:steps]
        s = ms.sigma(t)
        return torch.cat([s, s.new_zeros([1])])
    if scheduler == "flux2":
        return get_schedule(steps, round((lw * 16) * (lh * 16) / (16 * 16)))
    if scheduler == "align_your_steps":
        s = AYS_SD1[:]
        if steps + 1 != len(s):
            xs = np.linspace(0, 1, len(s))
            ys = np.log(s[::-1])
            s = np.exp(np.interp(np.linspace(0, 1, steps + 1), xs, ys))[::-1].copy()
        s[-1] = 0
        return torch.FloatTensor(s)
    if scheduler == "ideogram4":
        return ideogram4_sigmas(steps, lw * 16, lh * 16, 0, 1.75)
    return ideogram4_sigmas(steps, lw * 16, lh * 16, 0.5, 1.75)


MS = [("flow", 3.0), ("flow", 5.0), ("flow", 1.0), ("flux", 1.15), ("flux", 0.5)]
schedules = []
for kind, shift in MS:
    ms = model_sampling(kind, shift)
    for steps in (1, 2, 3, 7, 8, 20, 33):
        for scheduler in comfy.samplers.SCHEDULER_NAMES:
            if scheduler == "kl_optimal" and steps == 1:
                continue
            for sampler in ("euler", "dpm_2", "uni_pc"):
                s = ksampler_sigmas(ms, sampler, scheduler, steps)
                schedules.append(dict(kind=kind, shift=shift, sampler=sampler, scheduler=scheduler, steps=steps, sigmas=short(s.tolist())))
        for scheduler in ("turbo", "flux2", "align_your_steps", "ideogram4", "ideogram4turbo"):
            s = swarm_sigmas(ms, scheduler, steps, 128, 128)
            schedules.append(dict(kind=kind, shift=shift, sampler="euler", scheduler=scheduler, steps=steps, sigmas=short(s.tolist())))

CASES = [("flow", 3.0, "simple", 8, 1.0), ("flow", 5.0, "karras", 9, 4.0), ("flux", 1.15, "beta", 7, 3.0), ("flux", 1.15, "simple", 3, 1.0),
         ("flow", 3.0, "normal", 20, 5.0)]
runs = []
for kind, shift, scheduler, steps, cfg in CASES:
    ms = model_sampling(kind, shift)
    for name in comfy.samplers.SAMPLER_NAMES:
        evals.clear()
        sigmas = ksampler_sigmas(ms, name, scheduler, steps).to(DEVICE)
        wrap = Wrap(ms, cfg)
        n = torch.randn((1,) + N, generator=torch.manual_seed(SEED)).to(DEVICE)
        latent = torch.zeros_like(n)
        sampler = comfy.samplers.sampler_object(name)
        with torch.no_grad():
            x = sampler.sample(wrap, sigmas, {"model_options": {}, "seed": SEED}, None, n, latent, None, True)
        v = x.flatten().cpu()
        runs.append(dict(kind=kind, shift=shift, scheduler=scheduler, steps=steps, cfg=cfg, sampler=name, evals=short(evals), x=short(v[:KEPT].tolist())))
        print(f"{name:32} {kind} {scheduler:8} {len(evals):4} evals  rms {v.pow(2).mean().sqrt():.5f}", flush=True)

json.dump(dict(seed=SEED, schedules=schedules, runs=runs), open(out_path, "w"))
