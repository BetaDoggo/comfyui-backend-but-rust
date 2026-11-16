# Activation steering (Anima, experimental)

Three ways to push Anima towards a concept from inside the diffusion model, without changing the prompt. They are
command-line only; SwarmUI can't send them. Code: [`src/steer.rs`](../src/steer.rs), the hooks in
[`src/anima.rs`](../src/anima.rs), and two fitting scripts in `tools/`.

Measured on one concept, "snowy environment", with Anima Turbo (8 steps, CFG 1, `er_sde`, 1024x1024). The test set was 8
scenery prompts with no weather in them, 4 seeds each, none of them used for fitting. "Snow-prompt control" is the same
prompts with the snow words appended. The scores are crude pixel statistics from the experiment's own scripts, not human
judgement:

| Method | Best setting | Snow score | Blue cast | Saturation | Layout kept | Result, by eye |
|---|---|---|---|---|---|---|
| None (baseline) | | 0.121 | +0.015 | 0.392 | 1.00 | |
| Snow words in the prompt (control) | | 0.197 | +0.075 | 0.317 | 0.37 | snow |
| Residual vector, `--steer` from `steer_build.py` | alpha 0.05 to 0.25, blocks 4-22 | 0.133 to 0.047 | +0.07 to +0.36 | up to 0.58 | 0.90 to 0.59 | a blue tint, no snow; blocky colour at higher alpha |
| Low-rank map, `--steer` from `steer_fit.py` | early map alpha 1, late 0.5, blocks 2-11 | 0.165 | +0.023 | 0.338 | 0.53 | snow only where the scene leans wintry (mountains), none on a beach or street |
| Cross-attention concept, `--concept` | alpha 1, every block and step | 0.187 | +0.083 | 0.309 | 0.40 | snow in every scene |
| Cross-attention concept, `--concept` | alpha 2, steps 2-8 | 0.168 | +0.025 | 0.384 | 0.78 | snow added to the unsteered picture's layout |

Snow score is the share of bright, unsaturated pixels. Blue cast is mean(B - R). Layout kept is the correlation of a
32x32 grey thumbnail with the baseline image of the same seed.

## Why the residual vector fails

A steering vector is one offset added to every image token. Anima's per-token LayerNorm leaves an offset that is
shared across tokens in place, so it reaches the output as a shift of the whole picture's colour. Snow depends on where
a token is: on ground and roofs, not on sky or walls. In held-out pairs (same seed, with and without the snow words),
a constant shift explains none of the per-token change (R^2 -0.01). Anima's token norms are also dominated by a few
massive channels (about 1e4 mid-network), so the builder zeroes those channels and measures alpha in multiples of the
recorded difference instead.

## Low-rank map

`tools/steer_fit.py` fits `with - without ~= (without - mu) B + c` per block from token samples of paired runs, keeping
B's top directions (reduced-rank ridge regression). It explains 45 to 66% of the held-out change at blocks 2 to 8 over
steps 0 to 2, and less deeper in. Applied at a few early blocks it moves the image towards snow, but only partway.

## Cross-attention concept

Anima reads the text only through cross-attention. `--concept` encodes a concept prompt and a null prompt once. At
each steered block and step it adds `alpha * (attn(q, concept) - attn(q, null))` to the cross-attention output, so
each image token reads what the concept would have told it. Two extra cross-attention reads per block cost little. Put
the same filler in both prompts, such as `no humans, scenery`, so only the concept is left in the difference.

```bash
target/release/diffusion.exe --anima -p "no humans, scenery, a tropical beach with palm trees, a wooden pier, detailed background, wide shot" \
  --concept "no humans, scenery, snowy environment, deep snow covering the ground and rooftops, falling snowflakes, winter" \
  --concept-null "no humans, scenery, detailed background, wide shot" --concept-alpha 1 -o beach_snow.png
```

`--concept-steps 2-8` with alpha about 2 leaves the first steps alone, so the picture keeps the layout it would have had,
with snow added.

## Recording and fitting

```bash
# the same seeds with and without the concept: --steer-record for a vector, --steer-tokens for a map
target/release/diffusion.exe --anima --prompt-file with.txt -n 2 --seed 1000 --steer-record with.safetensors -o rec/a.png
target/release/diffusion.exe --anima --prompt-file without.txt -n 2 --seed 1000 --steer-record without.safetensors -o rec/b.png
python tools/steer_build.py with.safetensors without.safetensors -o vec.safetensors --steps 0-3
python tools/steer_fit.py with_tokens.safetensors without_tokens.safetensors -o map.safetensors --steps 0-3 --rank 128
```

`--steer FILE[:ALPHA[:STEPS[:BLOCKS]]]` is repeatable, so an early-step and a late-step map can run together. Ranges are
`a-b` with `b` exclusive, and steps count from 0.

Not tested: other concepts, other samplers, CFG above 1, batches above 1, LoRAs together with steering, and SwarmUI.
A LoRA change drops `--concept`'s encoded prompts.

# Style from a picture (Anima, experimental)

Training-free ways to take a picture's art style. Each step, the picture is noised to that step's sigma, with fixed noise,
and run through the unmodified model, prompted by `--style-caption`. Then:

* `--style-adain L` moves every channel of the target's tokens, after each block in `--style-adain-blocks`, a share `L`
  of the way to the picture's mean and standard deviation at that block. Only per-channel statistics cross over.
* Unless `--style-no-attention` is set, the target's self-attention also reads the picture's keys and values, in
  `--style-blocks`, with logits weighted by `--style-strength` (shared attention).
* `--style-steps` limits both to some steps. `--style-frame` places the picture at another RoPE frame.
  `--style-shuffle N` rearranges it in N-pixel tiles first. `--style-megapixels` sets the size it's shrunk to (1 by default; never enlarged).

Measured with `--style-megapixels 0.5` and six Anima-made references (watercolour, pixel art, ukiyo-e, ink sketch, oil, flat vector) and six unrelated
prompts at seeds 21 and 22. Qwen3-VL-4B (Krea 2's text encoder file) judged style match 0 to 10, and whether the
reference's subject leaked in:

| Settings | Style | Reference subject leaked |
|---|---|---|
| none | 0.7 / 1.1 | 0 |
| hand-written style words in the prompt | 4.7 / 5.3 | 0 |
| shared attention, `--style-strength 1` | 3.7 / 4.1 | 69% / 72% |
| `--style-no-attention --style-adain 0.6` | 4.5 / 4.7 | 3% / 0% |
| the same, plus Qwen3-VL's style tags as `--concept` at alpha 1.5 | 5.7 / 6.1 | 0% / 3% |

Shared attention copies the reference's subject whatever the blocks, steps or RoPE frame. Shuffling the reference stops
that, but it also stops most of the style. AdaIN at 1.0 breaks the image. Adding `--style-steps 2-8` keeps more of the
unstyled composition (thumbnail correlation 0.62 against 0.46). Not tested: real artwork or photos as references,
CFG above 1, batches above 1, and SwarmUI. SnapRef and a style reference can't be combined.

## Broke Anima Style Reference (ComfyUI node, SwarmUI extension, comfy-server)

The AdaIN-only setting is packaged as the `BrokeAnimaStyleReference` node, with three controls: `stats_match` (0.6),
`restyle` (on, which skips the first quarter of the steps, rounded) and `reference_megapixels` (1, from 0.5 to 4). The reference pass is prompted by
`no humans, scenery, detailed background`. The picture is shrunk (never enlarged) to `reference_megapixels` and noised with `torch.randn` from
seed `0x5EED`. There is no shared attention and there are no tags. Its ComfyUI node and the SwarmUI extension that emits
it (the style picture is the first prompt-box image, as for SnapRef) are a separate project, SwarmUI-BrokeAnimaStyleReference. The comfy-server
handles the node as `ModelPatch::StyleReference` (see `swarm backend/docs/INPUT_MAPPING.md`), and the engine sets
`AnimaRequest::style` from `StyleRequest::node`. Not combinable with SnapRef. Only the positive prompt's evaluations are
matched. With AdaIN alone, the engine keeps none of the reference's keys and values.

Reference size, with the node's settings at 1024x1024 (the six references and prompts, seeds 21 / 22; Qwen3-VL style
score, and the time the reference pass adds to a step with restyle):

| `reference_megapixels` | Style | Step time added | By eye |
|---|---|---|---|
| 0.25 | 2.6 / 2.8 | 29% | textured pictures break into speckle noise |
| 0.5 | 4.4 / 4.7 | 52% | pixel-art blocks smaller than the picture's |
| 1 | 4.0 / 4.5 | 108% | pixel-art blocks closest to the picture's |
| 2 | | 240% | blocks soften |
| 4 | | 620% | falls apart |

The 1 megapixel score is lower mostly through the ink sketch (1.7 / 2.0 against 3.0 / 3.2). The 0.25 row predates the last-block fix below. The 2 and 4 megapixel rows
are one pixel-art image each. ComfyUI took 70 s at 2 megapixels and 451 s at 4 on a 12 GB GPU (against 5 s at 1); this
engine took 2.6 and 5.2 s. The reference pass now runs every block, so the last block's statistics are matched too.

`--init-image a.png --denoise 0.6` is plain img2img for Anima: the picture is resized to `--width` x `--height`.
