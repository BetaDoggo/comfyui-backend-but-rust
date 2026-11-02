import os
import numpy as np, torch, comfy_kitchen as ck
from safetensors import safe_open
f = safe_open(os.path.join(os.environ["MODELS_ROOT"], r"clip\qwen3vl_4b.safetensors"), "pt")
P = "model.language_model.layers.0."
w = f.get_tensor(P+"self_attn.q_proj.weight").cuda(); ws = f.get_tensor(P+"self_attn.q_proj.weight_scale").cuda()
ln = f.get_tensor(P+"input_layernorm.weight").cuda()
x = torch.from_numpy(np.load("refout/te/ref_h0.npy")).cuda()
h = torch.nn.functional.rms_norm(x, ln.shape, weight=ln.float(), eps=1e-6)
q_ck = ck.int8_linear(h, w, ws, None, torch.float32, convrot=True, convrot_groupsize=256)
ref = torch.from_numpy(np.load("refout/te/ref_l0_q.npy")[0]).cuda()
mine = torch.from_numpy(np.fromfile("refout/te/l0_qkv.bin", dtype=np.float32).reshape(-1, 6144)[:, :4096]).cuda()
rel = lambda a, b: float((a-b).norm()/a.norm())
print("ck vs comfy", rel(ref, q_ck), " mine vs ck", rel(q_ck, mine))
# what if ComfyUI rounds input to bf16?
q_bf = ck.int8_linear(h.bfloat16(), w, ws, None, torch.float32, convrot=True, convrot_groupsize=256)
print("bf16-in vs comfy", rel(ref, q_bf))
deq = (w.float() * ws.float())
H = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=torch.float32, device="cuda")
Hm = H
for _ in range(3): Hm = torch.kron(Hm, H)
Hm = Hm / 16
wd = (deq.view(-1, 10, 256) @ Hm.T).view(deq.shape)
q_exact = h @ wd.T
print("exact vs comfy", rel(q_exact, ref), " exact vs mine", rel(q_exact, mine), " exact vs ck", rel(q_exact, q_ck))
