"""Token ids and per-token weights SwarmTextEncodeAdvanced builds, chunk by chunk, using SwarmUI's own
node code and ComfyUI's tokenizers. No checkpoint is needed.

Run with SwarmUI's python_embeded, from anywhere:
    python tools/swarm_prompt_ref.py <SwarmUI root> > out.json
"""
import importlib
import json
import os
import sys
import types

root = sys.argv[1]
comfy_dir = os.path.join(root, "dlbackend", "comfy", "ComfyUI")
sys.path.insert(0, comfy_dir)
os.chdir(comfy_dir)

import comfy.text_encoders.minimax
import comfy.text_encoders.moss_image

nodes = os.path.join(root, "src", "BuiltinExtensions", "ComfyUIBackend", "ExtraNodes", "SwarmComfyCommon")
package = types.ModuleType("swarm_common")
package.__path__ = [nodes]
sys.modules["swarm_common"] = package
swarm = importlib.import_module("swarm_common.SwarmText")


class Clip:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, False, **kwargs)


PROMPTS = [
    "a lake at dawn",
    " a lake ",
    "",
    "a <weight[1.5]:red> boat",
    "<weight[0.5]:a lake>",
    "lake<break>pine forest",
    "lake, <weight[1.3]:pine <weight[2]:bark>> (sky:1.2)",
]

out = {}
for name, clip in (("moss", Clip(comfy.text_encoders.moss_image.MOSSImageTokenizer())), ("h3", Clip(comfy.text_encoders.minimax.MiniMaxH3Tokenizer()))):
    cases = []
    for prompt in PROMPTS:
        chunks = []
        for leaves in swarm.chunks_from_leaves(swarm.parse_prompt(prompt).flatten(0, 8)):
            tokens, _ = swarm.combine_leaves(clip, leaves, clip.tokenize, per_leaf=True)
            batch = tokens[next(iter(tokens))][0]
            chunks.append({"ids": [t[0] for t in batch], "weights": [t[1] for t in batch]})
        cases.append({"prompt": prompt, "chunks": chunks})
    out[name] = cases
json.dump(out, sys.stdout)
