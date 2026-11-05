"""Run ComfyUI's own partial sampling on a mock model and write what it produces: the step ranges of KSampler (denoise,
start and end steps, leftover noise), and runs from a latent with SetLatentNoiseMask and DifferentialDiffusion.

usage: python partial_ref.py <SwarmUI root> > tests/partial_golden.json

The JSON is tests/partial_golden.json, which tests/partial_sampling_match_comfy.rs checks the Rust ports against.
Everything from KSampler and sampler_object() down is ComfyUI's code, from SwarmUI's copy. Only calc_cond_batch is
replaced, by the analytic model of tools/sampler_ref.py, so no checkpoint is needed.
"""
import json
import math
import os
import sys
from types import SimpleNamespace

COMFY = os.path.join(sys.argv[1], "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, COMFY)
sys.argv = [sys.argv[0]]

import torch
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa
import comfy.model_sampling
import comfy.sampler_helpers
import comfy.samplers
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

torch.set_grad_enabled(False)
N = (4, 8, 8)
NUMEL = 4 * 8 * 8
SEED = 42
DEVICE = "cuda"
TARGET = torch.tensor([math.sin(0.37 * i + 1.0) for i in range(NUMEL)], dtype=torch.float32, device=DEVICE).reshape((1,) + N)
# the image sampling starts from, at the sampler's scale
LATENT = torch.tensor([math.cos(0.23 * i) * 0.8 for i in range(NUMEL)], dtype=torch.float32, device=DEVICE).reshape((1,) + N)
# a soft mask at another resolution: fully kept on the left, a ramp, fully denoised on the right
MASK = torch.tensor([[min(max((x - 3) / 7, 0.0), 1.0) for x in range(13)] for _ in range(11)], dtype=torch.float32).reshape(1, 11, 13)


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
        def scale_latent_inpaint(sigma, noise, latent_image, **kwargs):  # BaseModel's
            return ms.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)), noise, latent_image)
        self.inner_model = SimpleNamespace(model_sampling=ms, scale_latent_inpaint=scale_latent_inpaint)
        self.model_patcher = SimpleNamespace(get_model_object=lambda name: ms, model=self)
        self.cfg = cfg

    def __call__(self, x, sigma, model_options={}, seed=None):
        return comfy.samplers.sampling_function(self.inner_model, x, sigma, "uncond", "cond", self.cfg, model_options, seed)


captured = []
comfy.samplers.sample = lambda *args, **kwargs: captured.append(args[7]) or "sampled"


def ksampler_sigmas(ms, sampler, scheduler, steps, denoise, start, end, leftover):
    """KSampler(denoise).sample(start_step, last_step, force_full_denoise) as KSamplerAdvanced and SwarmKSampler call it:
    the sigmas it samples, or None when it returns the latent."""
    model = SimpleNamespace(get_model_object=lambda name: ms)
    k = comfy.samplers.KSampler(model, steps, "cpu", sampler=sampler, scheduler=scheduler, denoise=denoise)
    captured.clear()
    out = k.sample(torch.zeros((1,) + N), None, None, 1.0, latent_image=LATENT.cpu(), start_step=start, last_step=end, force_full_denoise=not leftover)
    return captured[0] if isinstance(out, str) else None


schedules = []
for kind, shift in [("flow", 3.0), ("flux", 1.15)]:
    ms = model_sampling(kind, shift)
    for sampler, scheduler in [("euler", "simple"), ("dpm_2", "karras"), ("uni_pc", "beta"), ("euler", "normal")]:
        for steps in (1, 4, 20):
            for denoise in (1.0, 0.99995, 0.6, 0.35, 0.0):
                for start, end, leftover in [(0, 10000, False), (2, 10000, False), (0, 3, True), (0, 3, False), (3, 3, False), (5, 12, True), (25, 10000, False)]:
                    s = ksampler_sigmas(ms, sampler, scheduler, steps, denoise, start, end, leftover)
                    schedules.append(dict(kind=kind, shift=shift, sampler=sampler, scheduler=scheduler, steps=steps, denoise=denoise, start=start, end=end,
                                          leftover=leftover, sigmas=None if s is None else short(s.tolist())))

mask = comfy.sampler_helpers.prepare_mask(MASK, (1,) + N, DEVICE).float()
runs = []
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "uni_pc", "res_multistep", "euler_cfg_pp", "er_sde"]
for kind, shift, scheduler, steps, cfg in [("flow", 3.0, "simple", 12, 1.0), ("flux", 1.15, "beta", 10, 3.0)]:
    ms = model_sampling(kind, shift)
    for name in SAMPLERS:
        for start, end, leftover, add_noise, inpaint in [(5, 10000, False, True, None), (4, 10000, False, True, "mask"), (4, 10000, False, True, "differential"),
                                                          (3, 10000, False, True, "differential 0.4"), (0, 6, True, True, None), (6, 10000, False, False, None)]:
            sigmas = ksampler_sigmas(ms, name, scheduler, steps, 1.0, start, end, leftover).to(DEVICE)
            noise = torch.randn((1,) + N, generator=torch.manual_seed(SEED)).to(DEVICE) if add_noise else torch.zeros((1,) + N, device=DEVICE)
            options = {}
            if inpaint is not None and inpaint.startswith("differential"):
                strength = float(inpaint.split()[1]) if " " in inpaint else 1.0
                options["denoise_mask_function"] = lambda sigma, m, extra_options, strength=strength: DifferentialDiffusion.forward(sigma, m, extra_options, strength)
            evals.clear()
            recorded = short(sigmas.tolist())  # uni_pc sets the last sigma to 0.001 in place
            x = comfy.samplers.sampler_object(name).sample(Wrap(ms, cfg), sigmas, {"model_options": options, "seed": SEED}, None, noise, LATENT,
                                                          None if inpaint is None else mask, True)
            v = x.flatten().cpu()
            runs.append(dict(kind=kind, shift=shift, scheduler=scheduler, steps=steps, cfg=cfg, sampler=name, start=start, end=end, leftover=leftover,
                             add_noise=add_noise, inpaint=inpaint, sigmas=recorded, evals=short(evals), x=short(v.tolist())))
            print(f"{name:20} {kind} {start}-{end} {inpaint or '':18} {len(evals):3} evals  rms {v.pow(2).mean().sqrt():.5f}", file=sys.stderr, flush=True)

json.dump(dict(seed=SEED, latent=short(LATENT.flatten().tolist()), mask=short(mask.flatten().tolist()), schedules=schedules, runs=runs), sys.stdout)
