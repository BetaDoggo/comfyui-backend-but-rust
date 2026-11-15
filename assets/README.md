# Tokenizer files

These are data files, not code, and they keep their upstream licenses rather than this repository's GPL-3.0. Each folder
is a byte-identical copy of the folder with the same name in ComfyUI's `comfy/text_encoders/`, except `sd1_tokenizer/`,
which is ComfyUI's `comfy/sd1_tokenizer/`.

| Folder | Upstream | License |
|---|---|---|
| `sd1_tokenizer/` | CLIP tokenizer (OpenAI), which SD 1.x's CLIP-L and SDXL's CLIP-L and bigG share | MIT |
| `qwen25_tokenizer/` | Qwen2.5 tokenizer (Qwen team, Alibaba Cloud) | Apache-2.0 |
| `qwen35_tokenizer/` | Qwen3.5 tokenizer (Qwen team, Alibaba Cloud) | Apache-2.0 |
| `t5_tokenizer/` | T5 tokenizer (Google) | Apache-2.0 |

The CLIP tokenizer's MIT license is at <https://github.com/openai/CLIP/blob/main/LICENSE>. The Apache-2.0 text is in [`csrc/vendor/LICENSE.comfy-kitchen`](../csrc/vendor/LICENSE.comfy-kitchen) and at
<https://www.apache.org/licenses/LICENSE-2.0>.
