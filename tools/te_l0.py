import os
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI"))
out = sys.argv[1]; sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd, torch
clip = comfy.sd.load_clip(ckpt_paths=[os.path.join(os.environ["KREA2_MODELS"], r"clip\qwen3vl_4b.safetensors")], clip_type=comfy.sd.CLIPType.KREA2)
m = clip.cond_stage_model.qwen3vl_4b
lay = m.transformer.model.layers[0]
saved = {}
def hk(name, inp=False):
    def f(mod, i, o):
        t = i[0] if inp else (o[0] if isinstance(o, tuple) else o)
        if name not in saved: saved[name] = t.detach().float().cpu().numpy()
    return f
lay.self_attn.q_proj.register_forward_hook(hk("q"))
lay.self_attn.k_proj.register_forward_hook(hk("k"))
lay.self_attn.v_proj.register_forward_hook(hk("v"))
lay.self_attn.o_proj.register_forward_hook(hk("attn", inp=True))
lay.self_attn.q_norm.register_forward_hook(hk("qn"))
lay.post_attention_layernorm.register_forward_hook(hk("x_attn", inp=True))
tokens = clip.tokenize("a red fox sitting in a snowy forest, golden hour, photograph")
comfy.model_management.load_model_gpu(clip.patcher)
m.forward([[t[0] for t in b] for b in tokens["qwen3vl_4b"]])
for k, v in saved.items():
    print(k, v.shape); np.save(os.path.join(out, f"ref_l0_{k}.npy"), v)
