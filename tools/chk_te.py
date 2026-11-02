import os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "ComfyUI"))
sys.argv = [sys.argv[0]]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd, torch
clip = comfy.sd.load_clip(ckpt_paths=[os.path.join(os.environ["MODELS_ROOT"], r"clip\qwen3vl_4b.safetensors")], clip_type=comfy.sd.CLIPType.KREA2)
tr = clip.cond_stage_model.qwen3vl_4b.transformer
lay = tr.model.layers[0]
def h(name):
    def f(m, i, o):
        ii = i[0] if len(i) else None
        oo = o[0] if isinstance(o, tuple) else o
        print(name, "in", getattr(ii, "dtype", None), "out", getattr(oo, "dtype", None), flush=True)
    return f
for n in ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj", "input_layernorm", "self_attn"]:
    m = lay
    for p in n.split("."): m = getattr(m, p)
    m.register_forward_hook(h(n))
print("embed weight", type(tr.model.embed_tokens.weight).__name__, tr.model.embed_tokens.weight.dtype)
tokens = clip.tokenize("a red fox sitting in a snowy forest, golden hour, photograph")
c = clip.encode_from_tokens_scheduled(tokens)
print(type(lay.self_attn.q_proj).__mro__[:4])
