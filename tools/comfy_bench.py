"""Benchmark real ComfyUI (main.py, default memory management) on the Krea-2 Turbo workflow.

Start ComfyUI first, e.g.:  ref/venv/Scripts/python ref/ComfyUI/main.py --port 8199
usage: python comfy_bench.py [--port 8199] [--runs 3] [--size 1024 1024]
Each run changes the prompt so the text encoder, sampler and VAE all execute.
"""
import argparse, json, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8199)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024])
ap.add_argument("--steps", type=int, default=8)
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}"


def workflow(prompt, seed):
    w, h = a.size
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "kroma-v0.3.1-turbo-opd-W8A8.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b.safetensors", "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": a.steps, "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0], "negative": ["5", 0],
              "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "krea2_bench"}},
    }


def post(path, data):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(URL + path).read())


cid = str(uuid.uuid4())
for i in range(a.runs):
    prompt = f"a red fox sitting in a snowy forest, golden hour, photograph, variation {i} {time.time()}"
    t0 = time.perf_counter()
    pid = post("/prompt", {"prompt": workflow(prompt, 42 + i), "client_id": cid})["prompt_id"]
    while True:
        h = get(f"/history/{pid}")
        if pid in h and h[pid].get("status", {}).get("completed"):
            break
        time.sleep(0.05)
    print(f"run {i}: {time.perf_counter() - t0:.2f}s wall", flush=True)
