import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI"))
out = sys.argv[1]; sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd, torch
clip = comfy.sd.load_clip(ckpt_paths=[r"G:\stable-diffusion-webui-forge\models\clip\qwen3vl_4b.safetensors"], clip_type=comfy.sd.CLIPType.KREA2)
m = clip.cond_stage_model.qwen3vl_4b
m.layer = list(range(36)); m.layer_idx = None
tokens = clip.tokenize("a red fox sitting in a snowy forest, golden hour, photograph")
comfy.model_management.load_model_gpu(clip.patcher)
o = m.forward([[t[0] for t in b] for b in tokens["qwen3vl_4b"]])
h = o[0] if isinstance(o, tuple) else o
print(h.shape)
for i in range(h.shape[1]):
    np.save(os.path.join(out, f"ref_h{i}.npy"), h[0, i].float().cpu().numpy())
