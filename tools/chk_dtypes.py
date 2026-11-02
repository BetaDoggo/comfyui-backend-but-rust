import os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI"))
sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd, torch
m = comfy.sd.load_diffusion_model(os.path.join(os.environ["KREA2_MODELS"], r"Stable-diffusion\kroma-v0.3.1-turbo-opd-W8A8.safetensors"))
dm = m.model.diffusion_model
print("model dtype", m.model.get_dtype(), "manual_cast", m.model.manual_cast_dtype, flush=True)
for n, p in dm.named_parameters(recurse=True):
    if n.startswith("blocks.") and not n.startswith("blocks.0."): continue
    if "layerwise_blocks.1" in n or "refiner_blocks" in n: continue
    print(n, p.dtype, type(p.data).__name__, flush=True)
