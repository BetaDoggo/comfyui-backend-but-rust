# Engine input mapping

What SwarmUI actually sends for each model, and what the Rust engine has to accept as a result. This is the contract, not
a status report: the native engine does only part of it, and [STATUS](../../docs/STATUS.md) says which part.

Everything here was traced from SwarmUI's `WorkflowGenerator*.cs` / `WGNodeData.cs` and the Swarm
Python nodes (`ExtraNodes/SwarmComfyCommon`), cloned 2026-10-29 (SwarmUI `e7f38c3`, ComfyUI `fd7639f`).
The fixtures in `crates/comfy-server/tests/fixtures/` are concrete examples of each graph.
To add another model, see [ADDING_MODELS.md](ADDING_MODELS.md). HyperFlow H3 Flash, MiniMax-H3-Image, SnapRef and
Idol don't have sections here yet; their fixtures, and the [Models](../../docs/MODELS.md) sections, show what they take.

**Verified against a live SwarmUI** (built from the clone above, "ComfyUI API By URL" → this server,
mock engine, fake model files tagged via `modelspec.architecture`): backend loads with all Swarm-node
features detected; Krea 2, Anima, and MiniMax H3 (22-frame mp4 with audio) generate and save in SwarmUI;
the model-preload workflow and SwarmUI's Interrupt both work. SwarmUI always preloads a model the first
time with a `SwarmJustLoadTheModelPlease` workflow (`Engine::warmup`) before the first real generation.
MOSS Image and Qwen Image 2.1 were checked the same way (text-to-image, prompt images, Qwen 2.1 edit with two
images, and a Qwen 2.1 refiner upscale that saves an RGBA PNG). Their fixtures are the exact workflows
SwarmUI generated, captured with its `ComfyGetGeneratedWorkflow` API.

---

## 1. How the layers fit

```
SwarmUI ──HTTP/WS (ComfyUI API)──▶ comfy-server ──typed requests──▶ Engine (your Rust backend)
          node graph JSON           validates, walks graph,          load_*, encode_text, empty_latent,
                                    resolves pure-data nodes          sample, vae_decode/encode, ...
```

The server turns every node into one of two things:

* **Engine call.** Loaders, text encode, empty latent, sample, and VAE encode/decode. These are the
  [`Engine`](../crates/engine-api/src/engine.rs) trait methods, and each takes one request struct from
  [`requests.rs`](../crates/engine-api/src/requests.rs).
* **Pure bookkeeping.** Sigma shift, LoRA, attention token weights, reference latents, keyframes,
  conditioning multiply/zero, AV latent split/join, image batch/scale/crop, and base64 decoding. These are
  resolved server-side into data that rides on the handles (`ModelHandle.patches`, `CondEntry.meta`,
  `Latent.pixel`). The engine sees the finished recipe when it samples.

So the engine only ever has to implement the **11 required `Engine` methods**, and they're the same for every model. The
rest are optional. Their default implementations return "unsupported", which SwarmUI shows as an error only when
someone uses that feature:

* `vae_encode_audio`, `audio_silence`: Video Audio Input, Audio Silent Prefix/Suffix.
* `latent_op`, `unsample`: init-image and video latent arithmetic, Unsampler Prompt.
* `segment`, `remove_background`, `upscale_image`: `<segment:>` / `<clear:>`, Remove Background, model upscalers.
* `clip_vision_encode`: style models, unCLIP, image-to-video vision inputs.
* `model_sigmas`, `sample_custom`: the custom-sampler graph.
* `load_addon`: has a lazy default that just keeps the file.
* `load_image_only_checkpoint`: SVD checkpoints.
* `custom_node` + `supports_custom_node`: the generic route for every other node class; see "Shared features" below.

| Op | Krea 2 | Anima | MiniMax H3 | MOSS Image | Qwen Image 2.1 |
|---|---|---|---|---|---|
| `load_diffusion_model` | ✔ | ✔ | ✔ | ✔ | ✔ |
| `load_text_encoder` | Qwen3-VL 4B, `type=krea2` | Qwen3 0.6B, `type=stable_diffusion` | Qwen3-VL 32B, `type=minimax` | Qwen3.5 9B, `type=moss_image` | Qwen3-VL 8B, `type=qwen_image` |
| `load_vae` | Qwen-Image VAE | Qwen-Image VAE | ×2: video VAE + audio VAE | Flux 2 VAE | Qwen 2.1 **RGBA** VAE |
| `encode_text` | always `Swarm` mode | usually `Plain` | `Plain` or `Swarm` (refs) | always `Swarm` | always `Swarm` |
| `empty_latent` | `Image16Ch` | `Image16Ch` | `MiniMaxH3AudioVideo` | `Flux2` | `Image4Ch` at **half size** (placeholder) |
| `sample` | ✔ | ✔ | ✔ (joint AV latent) | ✔ | ✔ + `Latent::conform_empty` |
| `vae_decode` | ✔ | ✔ | ✔ tiled, video frames | ✔ | ✔ returns RGBA |
| `vae_decode_audio` | | | ✔ | | |
| `vae_encode` | init image / ref images | init image | keyframes | init image | ref images, init image, RGBA re-encode |
| `warmup` | "load model" button | same | same | same | same |

**LoRAs** work the same way for every model. Verified live for each model in this table, both from the LoRA parameter and
from `<lora:name:weight>` in the prompt:

| Model | `LorasTargetTextEnc` | Node SwarmUI sends | Engine sees |
|---|---|---|---|
| Krea 2, Anima | true | `LoraLoader{lora_name, strength_model, strength_clip}` | `ModelPatch::Lora` + `TextEncoderPatch::Lora` |
| MiniMax H3, MOSS Image, Qwen Image 2.1 | false | `LoraLoaderModelOnly{lora_name, strength_model}` | `ModelPatch::Lora` only |
| any (TE weight set to 0) | | `LoraLoaderModelOnly` | `ModelPatch::Lora` only |
| any, `<lora:…//hook=N//>` or region-confined | | `CreateHookLora` chain → `SetClipHooks` / `lora_hooks` | `TextEncoderPatch::Hooks` / `SwarmTextEncode.lora_hooks` |

LoRAs are listed from `loras/`, `Lora/` and `LyCORIS/`. Patches stack in graph order. The engine applies
them, at the given strengths, when it samples or encodes; the server never touches weights.

**Guidance tweaks and runtime options** work the same way for every model. SwarmUI inserts one patch node
per option between the loader and the sampler, in a fixed order, and each one appends to the handle's
`patches`. Verified live for all of them (Krea 2), and end to end in the `guidance_tweaks_reach_the_engine_in_graph_order` test:

| SwarmUI parameter(s) | Node | Engine sees |
|---|---|---|
| [FreeU] Apply To / Version / Block One, Two / Skip One, Two | `FreeU` / `FreeU_V2` | `ModelPatch::FreeU { version, b1, b2, s1, s2 }` |
| Self-Attention Guidance Scale / Sigma Blur | `SelfAttentionGuidance` | `ModelPatch::SelfAttentionGuidance` |
| Perturbed-Attention Guidance Scale | `PerturbedAttentionGuidance` | `ModelPatch::PerturbedAttentionGuidance` |
| Rescale CFG Multiplier | `RescaleCFG` | `ModelPatch::RescaleCfg` |
| Renorm CFG | `RenormCFG` (`cfg_trunc` always 100) | `ModelPatch::RenormCfg` |
| Use CFG Zero Star / Use TCFG | `CFGZeroStar` / `TCFG` | `ModelPatch::CfgZeroStar` / `ModelPatch::Tcfg` |
| Normalized Attention Guidance Scale / Alpha / Tau (scale > 0) | `NAGuidance` | `ModelPatch::NormalizedAttentionGuidance` |
| Seamless Tileable (`true` / `X-Only` / `Y-Only`) | `SwarmModelTiling` + `SwarmTileableVAE` | `ModelPatch::Tiling { axis }` + `VaePatch::Tiling { axis }` on `VaeHandle.patches` |
| CLIP Stop At Layer | `CLIPSetLastLayer` | `TextEncoderPatch::StopAtLayer { layer }` |
| EasyCache Mode / Threshold / Start / End | `EasyCache` | `ModelPatch::EasyCache` |
| Model Attention Backend | `ModelAttentionBackend` | `ModelPatch::AttentionBackend { backend }`: one of `EngineInfo::attention_backends`. SwarmUI always offers its own default, `pytorch attention`, so the server lists that first and turns it into the engine's first backend. |
| Use Sparse Attention (`sol` / `topk` / `vsa`) | `BlockSparseAttention` (V3 DynamicCombo `selection` + `selection.tau` / `selection.keep_percent`) | `ModelPatch::BlockSparseAttention(SparseAttention { method: SolAttn{tau} / Sla{keep_percent} / Vsa{keep_percent}, .. })` |
| Video Context Window: Video Context Length / Overlap (MiniMax H3; SwarmUI sends Wan and LTX-V models other nodes) | `MiniMaxH3ContextWindows` (`fuse_method: pyramid`, `freenoise: true`, `split_conds_to_windows: false`) | `ModelPatch::MiniMaxH3ContextWindows`, lengths in frames. SwarmUI refuses an overlap not below the length; the node's minimums (22 and 5 frames) are validated like ComfyUI's |
| Torch Compile | `TorchCompileModel` | `ModelPatch::Compile { backend }` (map to the engine's own compilation, or ignore) |
| Override Prediction Type (SD-family models only) | `ModelSamplingDiscrete` | `ModelPatch::SamplingDiscrete` |
| automatic for some model classes (Lens, HiDream O1) | `CFGNorm` / `ModelNoiseScale` | `ModelPatch::CfgNorm` / `ModelPatch::NoiseScale` |

### Shared features

These work the same way for every model. Each row was verified live through SwarmUI and is covered by a
test in [`swarm_compat.rs`](../crates/comfy-server/tests/swarm_compat.rs).

| SwarmUI feature | Nodes | Engine sees |
|---|---|---|
| `<region:>`, `<object:>`, Global Region Factor | `SwarmSquareMaskFromPercent`, the overlap-fix mask nodes, `ConditioningSetMask`, `ConditioningCombine` | One `CondEntry` per region with `CondMeta.mask = CondMask { mask, strength }`. SwarmUI's mask math runs server-side. |
| GLIGEN Model + regions | `GLIGENLoader`, `GLIGENTextBoxApply` | `CondMeta.gligen: Vec<GligenBox>`; the box text is encoded with the same text encoder. |
| Mask Image, Mask Blur / Grow / Shrink, Init Image Recomposite Mask, color correction, Init Image Noise | `SwarmMask*`, `SwarmImageCrop`, `SwarmImageScaleForMP`, `SwarmImageCompositeMaskedColorCorrecting`, `SwarmImageNoise`, `SetLatentNoiseMask` | Server-side pixels and masks; `Latent.noise_mask` for the sampler. |
| Use Inpainting Encode | `VAEEncodeForInpaint`, `InpaintModelConditioning` | `VaeEncodeRequest` on the masked image; `CondMeta.inpaint = InpaintConcat { latent, mask }`. |
| Init Image Reset To Norm, Batch Size with init images, Unsampler Prompt | `LatentAdd` / `LatentMultiply` / `SwarmLatentBlendMasked` / `RepeatLatentBatch` / `LatentFromBatch`, `SwarmUnsampler` | `Engine::latent_op(LatentOp::..)`, `Engine::unsample`. |
| `<segment:text>`, `<segment:yolo-...>`, `<clear:>`, Remove Background | `SwarmClipSeg`, `SwarmYoloDetection`, `SwarmRemBg` | `Engine::segment(SegmentRequest::{ClipSeg, Yolo})`, `Engine::remove_background`. The node's mask is `1 - alpha`. |
| Refiner Upscale Method `model-*` / `latent-*`, Refiner HyperTile | `UpscaleModelLoader` + `ImageUpscaleWithModel`, `LatentUpscaleBy`, `HyperTile` | `Engine::upscale_image`, `LatentOp::UpscaleBy`, `ModelPatch::HyperTile`. |
| Use Style Model + Multiply / Merge Strength / Apply Start | `CLIPVisionEncode`, `StyleModelApply`, `ConditioningAverage`, `ConditioningSetTimestepRange`, `ConditioningCombine` | `Engine::clip_vision_encode` → `ClipVisionOutput`. Then `CondMeta.embedding_ops = [Style {..}, Average {from, to_strength}]` (applied in order), with `start_percent` / `end_percent` splitting the plain and styled prompts. |
| Use Reference Only | `SwarmReferenceOnly`, then `LatentFromBatch(1)` | `ModelPatch::ReferenceOnly { batch }` + `LatentOp::PrependReference`. The noise mask keeps the reference frame (0). |
| ReVision (unCLIP) | `unCLIPConditioning` | `CondMeta.unclip`. SwarmUI only allows it for SDXL. |
| ControlNet One/Two/Three (+ Union Type) | `ControlNetLoader` → [`SetUnionControlNetType`] → `ControlNetApplyAdvanced` | `CondMeta.controls: Vec<ControlHint>` on positive and negative, oldest first. The union type is in `AddonHandle.control_type`, and the VAE is set for SD3 / Flux / Qwen Image classes. |
| AliMama inpainting ControlNet | `ControlNetInpaintingAliMamaApply` | As above with `ControlHint.inpaint_mask`. |
| DiffSynth / Anima LLLite ControlNets (chosen by the ControlNet model's class) | `ModelPatchLoader` + `QwenImageDiffsynthControlnet` / `AnimaLLLiteApply` | `ModelPatch::DiffSynthControl` / `ModelPatch::AnimaLLLite`. SwarmUI passes the inpaint mask to LLLite. |
| ControlNet preprocessors (third-party), Preview Only | preprocessor node → `ResizeImageMaskNode` (`scale to multiple`, 8) | The resize runs server-side. The preprocessor goes through the generic route. |
| IP2P CFG 2 (SDXL edit, OmniGen) | `RandomNoise`, `KSamplerSelect`, `BasicScheduler` / `KarrasScheduler` / `SDTurboScheduler`, `SplitSigmas`, `DualCFGGuider`, `SamplerCustomAdvanced` | `Engine::model_sigmas` for model-dependent schedules, then `Engine::sample_custom(CustomSampleRequest { noise, guider, sampler, sigmas, latent })`. Karras schedules and splits are computed server-side. |
| Audio output without SwarmUI's nodes | `SaveAudioAdvanced` | Encoded with ffmpeg and saved to `output/`. `format` arrives as a nested dict and is flattened to `format` / `format.quality`. |
| Kandinsky 5 image-to-video | `ReplaceVideoLatentFrames`, `NormalizeVideoLatentStart` | `LatentOp::ReplaceFrames` / `LatentOp::NormalizeStart`. |
| SVD | `ImageOnlyCheckpointLoader` | `Engine::load_image_only_checkpoint` → model, CLIP Vision, VAE. |
| **Everything else**: other model families (Wan, LTX-V, Hunyuan, SD3, Cosmos, ...) and third-party packs (IP-Adapter, SAM2, RIFE/FILM/GIMM, GGUF, NF4, Nunchaku, TensorRT, TeaCache, AITemplate, ExtraModels, Dynamic Thresholding) | the 92 classes in [COMPATIBILITY.md](COMPATIBILITY.md) | Registered only if `Engine::supports_custom_node(class)`, with the stock or hand-written schema. Runs as `Engine::custom_node(CustomNodeRequest { class_type, inputs, output_types })`. Inputs are `NodeValue`s; model-file combos are resolved to `NodeValue::File`. The engine returns one `NodeValue` per output. Verified live: Wan 2.1 text-to-video with TeaCache, the Krea 2 → Wan 2.1 image-to-video chain, IP-Adapter, the MiDaS preprocessor, RIFE / FILM, SAM2 points, Dynamic Thresholding, CLIP device and AITemplate. |

---

## 2. Krea 2

**Detected by SwarmUI as:** compat class `krea-2` (VAE family `qwenimage`), standard res 1024².
**Files:**

| Role | Node | File SwarmUI asks for | Notes |
|---|---|---|---|
| Diffusion model | `UNETLoader` | user's model (e.g. `krea2/…safetensors`) | `weight_dtype = fp8_e4m3fn` by default (`fp8_e4m3fn_fast` on 30xx+) |
| Text encoder | `CLIPLoader` | `qwen3vl_4b.safetensors` (or the user's `Qwen Model` param) | `type = "krea2"` |
| VAE | `VAELoader` | Qwen-Image VAE (`QwenImage/qwen_image_vae.safetensors`, or user default/override) | |

**Graph SwarmUI emits** (`tests/fixtures/krea2_t2i.json`):

```
UNETLoader(4) ─┬─▶ SwarmAttnTokenWeights(model, conditioning=+prompt) ─▶ SwarmKSampler(10) ─▶ VAEDecode(8) ─▶ SwarmSaveImageWS(9)
CLIPLoader ────┼─▶ SwarmTextEncodeAdvanced(6, +)  ──────────────────────▲  ▲
               └─▶ SwarmTextEncodeAdvanced(7, −)  ─────────────────────────┘
EmptySD3LatentImage(5) ─────────────────────────────────────────────────▶ latent_image
VAELoader ──────────────────────────────────────────────────────────────▶ VAEDecode.vae
```

**Engine inputs, with SwarmUI parameter → request field:**

| SwarmUI param | Arrives as | Engine sees |
|---|---|---|
| Prompt / Negative Prompt | `SwarmTextEncodeAdvanced.prompt` | `SwarmTextEncode.prompt: ParsedPrompt` (`<break>`, `<weight>`, `<fromto>`, `<alternate>`, `<embed>` parsed; `(x:1.2)` left in text) |
| (fixed) | `llama_template = "krea2"` | `llama_template: Some("krea2")`, so wrap in the Krea 2 chat template (see below) |
| Steps | `steps` (also on encoder, for `<fromto>`) | `SampleRequest.steps`, `SwarmTextEncode.steps` |
| Width / Height | `EmptySD3LatentImage`, encoder `width/height/target_*` | `EmptyLatentKind::Image16Ch{w,h}` → `[B,16,H/8,W/8]` |
| Batch Size | `batch_size` | `EmptyLatentRequest.batch_size` |
| Seed / Variation Seed + Strength | `noise_seed`, `var_seed`, `var_seed_strength` | `noise_seed`, `variation: Option<VariationSeed>` |
| CFG Scale | `cfg` | `cfg` |
| Sampler / Scheduler | `sampler_name` / `scheduler` | defaults **`euler` / `simple`** |
| Sigma Shift (only if set) | `ModelSamplingFlux{max_shift=base_shift=shift, width, height}` | `ModelPatch::SamplingFlux` |
| "Model specific enhancements" (on by default) | `SwarmAttnTokenWeights` | `ModelPatch::AttnTokenWeights{conditioning}`: apply per-token prompt weights inside attention (`attn1_patch`) rather than scaling embeddings |
| Flux Guidance Scale (if set) | `guidance` | `guidance: Some(f)` (otherwise -1 → `None`) |
| Prompt images | `SwarmLoadImageB64` → `ImageScale` (to ≈ gen res) → `ImageBatch` chain → `images` | `SwarmTextEncode.images: Option<ImageBatch>` (VL input; template appends image tokens) |
| Enable Reference Latents ≠ none | + `VAEEncode` → `ReferenceLatent` → `FluxKontextMultiReferenceLatentMethod` | `CondMeta.reference_latents` + `reference_latents_method` (`index`, `index_timestep_zero`, …). Default `none` = images go to the text encoder only, which is correct for the base model |
| LoRAs | `LoraLoader` (model+clip) | `ModelPatch::Lora`, `TextEncoderPatch::Lora` |
| `<lora:…//hook=N//>` in prompt | `CreateHookLora` chain → `lora_hooks` | `SwarmTextEncode.lora_hooks` |
| Init image + Creativity | `SwarmLoadImageB64` → `VAEEncode` → sampler `start_at_step = round(steps·(1−creativity))` | `vae_encode`, `SampleRequest.start_at_step` |
| Tile sample / sigma min/max / rho / previews | `SwarmKSampler` extras | `tile_size`, `sigma_min/max`, `rho`, `previews` |

Krea 2 chat template (from `SwarmTextHandling.py`), with `{}` replaced by the prompt:

```
<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text,
spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n
```

**Engine-side work specific to Krea 2:** Qwen3-VL 4B encoding with the template, plus optional image tokens.
Flow-matching sampling with the Flux-style resolution-dependent shift. Attention-level token weighting. A
16-channel Qwen-Image latent at /8.

---

## 3. Anima

**Detected as:** compat class `anima` (VAE family `qwenimage`), 1024². Older `anima-preview` IDs are remapped.

| Role | Node | File | Notes |
|---|---|---|---|
| Diffusion model | `UNETLoader` | user's model | `weight_dtype = "default"` (small dense model, so SwarmUI trusts the file's dtype) |
| Text encoder | `CLIPLoader` | `qwen_3_600m.safetensors` | `type = "stable_diffusion"`. ComfyUI identifies Anima's Qwen3 0.6B by its weights |
| VAE | `VAELoader` | Qwen-Image VAE | |

**Graph** (`tests/fixtures/anima_t2i_lora.json`): `UNETLoader → [ModelSamplingAuraFlow] → [LoraLoader…] → SwarmKSampler`,
`CLIPLoader → [LoraLoader] → CLIPTextEncode ×2`, `EmptySD3LatentImage`, `VAEDecode`, `SwarmSaveImageWS`.

| SwarmUI param | Arrives as | Engine sees |
|---|---|---|
| Prompt / Negative | `CLIPTextEncode.text`, or `SwarmTextEncodeAdvanced` when the prompt contains Swarm tags **or any `(`** | `TextEncodeMode::Plain{text}` or `Swarm(..)`, so support both. Verified live: SwarmUI rewrites `(silver hair:1.2)` to `<weight[1.2]:silver hair>` before sending, and the Swarm-mode encode gets `width/height` × 1.5 (e.g. 1280×1792 for an 832×1216 image) as size conditioning |
| Sampler / Scheduler | | defaults **`er_sde` / `simple`** |
| Sigma Shift (only if set) | `ModelSamplingAuraFlow{shift}` | `ModelPatch::SamplingAuraFlow` |
| Size / batch / seed / cfg / steps | as Krea 2 | `Image16Ch`, etc. |
| LoRAs | `LoraLoader` (`anima/lora` class) | model + text-encoder LoRA patches |
| ControlNet (Anima LLLite) | `ModelPatchLoader` + `AnimaLLLiteApply` | `ModelPatch::AnimaLLLite` (see "Shared features") |
| SnapRef group (SwarmUI-SnapRef extension) | `SnapRef` after the model loader and LoRAs: the prompt images as `image`, `image_2`, `image_3` (at most 3, each `weight_N` = SnapRef Strength), `adapter` = the chosen version's file in the LoRA folder (SwarmUI downloads it first), `mode`, `ref_megapixels`, `ref_caption`, `fidelity`, `ref_in_uncond` false. The prompt images then no longer reach the text encoder | `ModelPatch::SnapRef` |

Note: SwarmUI's "Images: N" becomes N separate prompts with consecutive seeds (`batch_size` stays 1);
only the "Batch Size" param raises `batch_size`.

**Engine-side work:** Qwen3 0.6B → Anima's LLM adapter conditioning, Anima DiT, and the Qwen-Image VAE.
Accept `(word:1.2)` emphasis in plain text, since SwarmUI marks Anima `SupportLegacyPromptParser`.

---

## 4. MiniMax H3 (text/image → video **with audio**)

**Detected as:** compat class `minimax-h3`: `IsText2Video`, `IsImage2Video`, `HasJointAVLatents`,
`ResolutionPrecision = 32`, `LorasTargetTextEnc = false`. Standard res 960². Frames snap to the
**17k + 5** grid (5, 22, 39, …, 124 ≈ 5 s @ 24 fps). Default length is 124 frames.

| Role | Node | File | Notes |
|---|---|---|---|
| Diffusion model | `UNETLoader` | user's model | `fp8_e4m3fn` default |
| Text encoder | `CLIPLoader` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `type = "minimax"` |
| Video VAE | `VAELoader` | `MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors` (or user default) | |
| Audio VAE | `VAELoader` (plain loader) | `MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors` | a *second* VAE handle |

**Text-to-video graph** (`tests/fixtures/minimax_h3_t2v.json`):

```
UNETLoader ─▶ MiniMaxH3SigmaShift(shift_video, shift_audio=3) ─▶ SwarmKSampler(previews=animate)
SwarmEmptyMiniMaxH3LatentAV(w,h,length) ─▶ latent_image
SwarmKSampler ─▶ LTXVSeparateAVLatent ─┬─[0]▶ VAEDecodeTiled(video VAE, 256/64, temporal 9999) ─▶ images ┐
                                       └─[1]▶ VAEDecodeAudio(audio VAE) ─────────────────────▶ audio ──┴▶ SwarmSaveAnimationWS(fps 24, h264-mp4)
```

| SwarmUI param | Arrives as | Engine sees |
|---|---|---|
| Width/Height | `SwarmEmptyMiniMaxH3LatentAV` | `EmptyLatentKind::MiniMaxH3AudioVideo{w,h,length}` → joint latent: video `[1,24,T,H/16,W/16]` with `T = 2` if ≤5 frames else `((f−5)/17)·5+2`; audio `[1,32,2,round(f/24·40)]`. Helpers: `requests::minimax_h3::temporal_shape` |
| Text2Video Frames | `length` (pre-aligned by SwarmUI) | `Latent.pixel.frames` |
| Video FPS | `SwarmSaveAnimationWS.fps` (24) | server-side encode only |
| Sigma Shift | `MiniMaxH3SigmaShift.shift_video` (default **12**), `shift_audio = 3` | `ModelPatch::MiniMaxH3Shift`, **always** present |
| Sampler / Scheduler | | defaults **`res_multistep` / `simple`** |
| Video Preview Type | `previews` (default `animate`) | `PreviewMode::Animate` |
| Prompt images / audio / videos (`<Picture i>`, `<Audio j>`, `<Video k>`) | `SwarmMiniMaxH3CollectReferences` (≤9 images, ≤3 audio, ≤3 videos resampled to 24 fps) → `SwarmTextEncodeAdvanced.minimax_refs` | `SwarmTextEncode.minimax_refs: MiniMaxReferences` (`style: Video`) holding **raw** media + both VAE handles. The engine VAE-encodes refs and feeds Qwen3-VL. The Python reference scales images (`match` = gen pixel area) and crops videos to `17k+5` frames. ComfyUI's `MiniMaxH3ImageConditioning` sends its pictures the same way, as `TextEncodeMode::MiniMax` with `style: Image` and an optional VAE. SwarmUI's MiniMax H3 Image class sends `SwarmEmptyMiniMaxH3ImageLatent(w,h,batch)` → `EmptyLatentKind::MiniMaxH3Image` (`[B,24,1,H/16,W/16]`; the engine refuses B > 1) and `SwarmMiniMaxH3ImageCollectReferences(vae, ref_images)` → `minimax_refs` with `style: Image` |
| Image-to-video (init/first frame), Video End Frame | `SwarmMiniMaxH3AddKeyframes(first_frame, last_frame)` | server resizes and calls `vae_encode`; `CondMeta.minimax_keyframes = [{frame_index: 0 / frames−1, latent}]`, `minimax_frame_count`. I2V default CFG is 1 |
| Video Audio Input | `SwarmLoadAudioB64` → `SwarmEnsureAudio(0.1 s)` → `VAEEncodeAudio` → `SetLatentNoiseMask(SolidMask 0)` → `LTXVConcatAVLatent` | `Engine::vae_encode_audio` (clip padded with leading silence to ≥ 0.1 s), then a joint latent whose **`Latent.audio_noise_mask`** is all 0: keep the encoded audio, generate only the video |
| Audio Silent Prefix / Suffix Duration | `LTXVSeparateAVLatent[1]` → `SwarmAudioSilentMaskPrefixSuffix(audio VAE)` → `LTXVConcatAVLatent` | `Engine::audio_silence(AudioSilenceRequest)`: splice encoded silence into the first/last seconds of the audio latent and return it with a time-axis noise mask that is 0 there (ends up as `audio_noise_mask`) |
| Video Boomerang, Trim Video Start/End Frames | `SwarmTrimFrames` → `SwarmVideoBoomerang` after decode | server-side only (frames reversed / dropped) |
| `<extend:N>` (Video Extend Model, Frame Overlap, …) | `SwarmCountFrames` / `SwarmImageWidth` / `SwarmImageHeight` / `SwarmIntAdd` feeding INT inputs, `ImageFromBatch` + `ImageScale` for the overlap frames, a second keyframed H3 run, `ImageBatch` + `AudioConcat` to join | the second run is a normal image-to-video request; joining frames and audio is server-side (`AudioData::concat` duplicates mono to stereo and resamples to the higher rate) |
| Video End Image (from another model's image) | `BatchImagesNode(start, end)` → `SwarmTextEncodeAdvanced.images`, plus `last_frame` keyframe | server batches the images (resized to the first), the engine sees them as text-encoder images and a last-frame keyframe |
| VAE tiling | `VAEDecodeTiled{256, 64, temporal_size 9999, 8}` | `VaeDecodeRequest.tiling` (9999 = don't tile temporally) |
| 2-frame request | extra `ImageFromBatch(0,1)` after decode | handled server-side |

**Engine-side work:** a joint audio+video DiT over a nested (video, audio) latent; separate video and
audio flow shifts; Qwen3-VL 32B encoding with multimodal refs; keyframe conditioning; video VAE decode to
24 fps frames and audio VAE decode to a waveform. For memory, return video frames as `PixelData::U8`
(124 × 960² × 3 as f32 is ~1.4 GB).

---

## 5. MOSS Image

**Detected as:** compat class `moss-image` (VAE family `flux-2`), standard res 1024², `LorasTargetTextEnc = false`.

| Role | Node | File SwarmUI asks for | Notes |
|---|---|---|---|
| Diffusion model | `UNETLoader` | user's model (`Comfy-Org/MOSS-Image`, bf16 or int8) | `fp8_e4m3fn` default |
| Text encoder | `CLIPLoader` | `qwen3.5_9b_int8_convrot.safetensors` (or the `Qwen Model` param) | `type = "moss_image"` |
| VAE | `VAELoader` | `Flux/flux2-vae.safetensors` (or the user's `DefaultFlux2VAE`) | |

**Graph** (`tests/fixtures/moss_t2i.json`): the Krea 2 shape without the attention-weights node:
`UNETLoader → [ModelSamplingAuraFlow] → SwarmKSampler`, `SwarmTextEncodeAdvanced ×2`, `EmptyFlux2LatentImage`,
`VAEDecode`, `SwarmSaveImageWS`.

| SwarmUI param | Arrives as | Engine sees |
|---|---|---|
| Prompt / Negative | `SwarmTextEncodeAdvanced` (same branch as Krea 2) | `Swarm` mode with **`llama_template: Some("krea2")`**, but `SwarmTextEncodeAdvanced` only uses a template with images. Without them the model's own `<\|im_start\|>user\n{}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n` wraps the prompt, untrimmed. `tools/swarm_prompt_ref.py` records the tokens |
| Width / Height | `EmptyFlux2LatentImage` | `EmptyLatentKind::Flux2{w,h}` → `[B,128,H/16,W/16]` |
| Sampler / Scheduler | | defaults **`euler` / `simple`** |
| Sigma Shift (only if set) | `ModelSamplingAuraFlow{shift}` | `ModelPatch::SamplingAuraFlow` |
| Prompt images | `SwarmLoadImageB64 → ImageScale` to ≈384² pixel area (e.g. 640×480 → 443×333) → `ImageBatch` chain (≤10) | `SwarmTextEncode.images` (vision tokens only; MOSS gets no reference latents) |
| Flux Guidance Scale (if set) | `guidance` | `Some(f)`, else `None` |
| CFG, steps, seed, LoRAs, init image | as Krea 2 | as Krea 2 |

**Engine-side work:** Qwen3.5 9B encoding, the MOSS DiT, and the Flux 2 VAE (128-ch latent at /16).
The upstream SkipLayerGuidance tip relies on a Swarm extension node that isn't in the default graph.

---

## 6. Qwen Image 2.1 (with alpha)

**Detected as:** compat class `qwen-image-2.1` (VAE family `qwenimage21`), standard res 1024², `ResolutionPrecision = 32`,
**`SupportsAlpha = true`**, `LorasTargetTextEnc = false`. The model is built for 2048², because its VAE downscales by 16.

| Role | Node | File SwarmUI asks for | Notes |
|---|---|---|---|
| Diffusion model | `UNETLoader` | user's model (`Comfy-Org/Qwen-Image-2.1`, regular or Turbo) | `fp8_e4m3fn` default |
| Text encoder | `CLIPLoader` | `qwen3vl_8b_int8_convrot.safetensors` | `type = "qwen_image"`, the same type string as Qwen Image 1. Tell them apart by the file |
| VAE | `VAELoader` | `QwenImage/qwen_image_2.1_vae_bf16.safetensors` (or `DefaultQwen21VAE`, e.g. the texture-fix VAE) | 64-ch latent, /16, decodes **RGBA** |

**Text-to-image graph** (`tests/fixtures/qwen21_t2i_alpha.json`):

```
UNETLoader(4) ─────────────────────────────────▶ SwarmKSampler(10) ─▶ VAEDecode(8) ─▶ SwarmSaveImageWS(9)  (RGBA PNG)
CLIPLoader ─▶ SwarmTextEncodeAdvanced(6, +) ───▲  ▲
          └─▶ SwarmTextEncodeAdvanced(7, −) ──────┘
EmptyLatentImage(5, width/2, height/2) ────────▶ latent_image
```

**The latent placeholder.** ComfyUI has no "empty Qwen 2.1 latent" node, so SwarmUI sends
`EmptyLatentImage` at **half** the requested size, which is `[B,4,H/16,W/16]`. `SwarmKSampler` then calls ComfyUI's
`fix_empty_latent_channels`, which widens an all-zero latent to the model's 64 channels and keeps its
spatial size (`[B,64,H/16,W/16]`). The VAE then decodes that to the real `W×H`. The engine does the same:

* The server sets `Latent.empty = true` on latents from `Empty*Latent*` nodes, and clears it on sampler output.
* In `sample`, call `req.latent.conform_empty(LatentFormat::QwenImage21, 64, 16)`. It returns the widened
  latent with `pixel` corrected to `W×H`, or `None` if nothing needs changing. The mock engine does this.

| SwarmUI param | Arrives as | Engine sees |
|---|---|---|
| Prompt / Negative | `SwarmTextEncodeAdvanced` with **no** `guidance` or `llama_template` | `Swarm` mode, `llama_template: None`, `guidance: None`. The upstream transparency preset is plain prompt text: `This is an RGBA image with transparency. {prompt}. The image has alpha channel and the background is transparent.` |
| Width / Height | `EmptyLatentImage(w/2, h/2)` + encoder `width/height` = real size | `EmptyLatentKind::Image4Ch{w/2,h/2}`, then `conform_empty` (see above) |
| Sampler / Scheduler | | defaults **`euler` / `simple`**. Turbo: Steps 8 and Scheduler `align_your_steps`, which `SwarmKSampler` maps to a fixed Qwen 2.1 sigma list (`1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.001`) |
| CFG | `cfg` (upstream recommends 1) | `cfg` |
| Prompt images (edit/reference, ≤16) | `SwarmLoadImageB64` (not resized) → `ImageBatch` chain → `images` on **both** prompts; **and** per image `VAEEncode → ReferenceLatent` on **both** positive and negative | `SwarmTextEncode.images` (batched; `ImageBatch` resizes later images to the first one's size), plus `CondMeta.reference_latents` on both conds, one per image, in order. No `reference_latents_method` |
| Init image (no prompt images) | the init image is also added as a single reference latent | same, one entry |
| Refiner upscale | `VAEDecode → ImageScale (RGBA) → VAEEncode → SwarmKSampler` | `vae_encode` with **4-channel** pixels; `start_at_step` for the refine pass |
| Model upscaler / background removal | `SplitImageWithAlpha` / `JoinImageWithAlpha` around the RGB-only step | server-side (both nodes implemented) |

**Alpha end to end.** `vae_decode` returns `ImageBatch { channels: 4 }` with straight alpha (1.0 = opaque),
and the server keeps it:

* `SwarmSaveImageWS` writes an RGBA PNG (8- or 16-bit), or an RGBA BMP for `raw`.
* `ImageScale` and `ImageBatch` handle 4 channels. Batching RGB with RGBA pads alpha with 1.0, like ComfyUI.
* JPEG previews and video encodes drop alpha.

Verified live: SwarmUI saved a 1024×768 RGBA PNG with the mock's alpha gradient intact.

**Engine-side work (done in `native-engine`):**

* Qwen3-VL 8B encoding with vision inputs. `CLIPLoader type=qwen_image` is told from Qwen Image 1 by the file (the 8B has a
  vision tower with DeepStack mergers). The prompt is the template with `<imageN>` and the vision block of each image in
  front of the text; the system turn and the image tokens are dropped, and `slots` say where each image goes.
* The MMDiT with reference latents, with the text/reference prefix cached across steps as ComfyUI does.
* The RGBA VAE: RGB input is padded with alpha 1.0 when encoding, and decoding gives 4 channels.
* `EmptyLatentImage` returns a 4-channel latent and `sample` widens it with `Latent::conform_empty`.
* Refused: `<break>`, prompt weights, differing reference latents on the
  two prompts.

---

## 7. What the server sends back (return types)

SwarmUI never reads files for these models. It gets outputs inline over the websocket:

1. JSON `{"type":"progress","data":{"value":N,"max":M,…}}`. **JSON frames must begin with the exact bytes
   `{"type":`**, because that is how SwarmUI tells JSON from binary.
2. Previews: binary `[u32 BE event=1][u32 BE format=1 (jpeg)][jpeg bytes]` (from `ProgressSink::preview`).
3. Final output: a `progress` with `max = 12345` (image/audio) or `12346` (video), then
   `[u32 event=1][u32 format][bytes]`. Format codes: 1 jpg, 2 png, 3 webp, 4 gif, 5 mp4, 6 webm, 7 mov,
   8 mp3, 9 wav, 10 flac, 11 ogg. `bit_depth=raw` uses event 10 + BMP. PNG/BMP keep alpha when the image has it.
4. Output node id: SwarmUI marks outputs from node ids `< 100` (except `9`) and `≥ 50000` as intermediate.
   The real output is node `9`.
5. End of job: `{"type":"executing","data":{"node":null,"prompt_id":…}}`, sent only **after** the history
   entry is stored (ComfyUI's order). SwarmUI then `GET /history/{id}` and raises any
   `["execution_error", {exception_message…}]` in `status.messages` to the user.

`ProgressSink::progress(value, max)` drives SwarmUI's per-image bar. Return `EngineError::Interrupted`
when `progress.interrupted()` turns true (SwarmUI's cancel button → `POST /interrupt`).

---

## 8. Server endpoints SwarmUI uses

| Call | When | Implemented |
|---|---|---|
| `GET /object_info` | backend init (model lists, samplers, feature detection) | ✔ the native nodes, plus the generic-route nodes the engine claims (all of them with the mock); model lists rescanned per call |
| `GET /features` | idle health check | ✔ |
| `GET /ws?clientId=` + `feature_flags` | per generation (socket reused) | ✔ |
| `POST /prompt` `{prompt, client_id}` | generate | ✔ Comfy-shaped validation errors (HTTP 400) |
| `GET /history/{id}`, `POST /history {delete}` | after each job | ✔ |
| `POST /queue {delete}`, `POST /interrupt {prompt_id}` | cancel | ✔ |
| `POST /free {unload_models, free_memory}` | "Free VRAM" | ✔ → `Engine::free_memory`. `unload_models` drops the models and the loader cache; `free_memory` alone moves the weights from VRAM to RAM and keeps the models loaded |
| `POST /upload/image`, `GET /view` | fallback when Swarm nodes are missing / `SaveImage` | ✔ |
| `/api/*` aliases | "EnableFrontendDev" setting | ✔ |

## 9. Known gaps

* No node-editor web UI at `/`. SwarmUI's "Comfy Workflow" tab won't render, but generation works.
* [COMPATIBILITY.md](COMPATIBILITY.md) lists every SwarmUI parameter and prompt-syntax feature, probed live
  against Krea 2, Anima, MiniMax H3, MOSS Image and Qwen Image 2.1. Every node class SwarmUI can emit is native or on the generic route. An engine that
  doesn't claim a generic-route class gets a `missing_node_type` validation error for it, naming the node and
  the SwarmUI parameters behind it.
* The third-party schemas in `tools/third_party_schemas.py` were written from the packs' public node
  definitions and checked against everything SwarmUI sends and reads back. They weren't diffed against
  installed packs. One deliberate deviation: RIFE's `scale_factor` is a FLOAT rather than a float combo, so
  SwarmUI's integer `1` validates.
* **SwarmUI quirks seen while testing:** Use Reference Only with several prompt images keeps batch index 1,
  which is the second reference rather than the generation. The custom-sampler path applies the end step
  to a schedule it has already split at the start step.
* The upstream MOSS SkipLayerGuidance tip relies on an extension node that SwarmUI never emits, so it is not
  in the backlog.
* **Possible upstream SwarmUI bug:** for MiniMax H3 (and Wan / HunyuanVideo 1.5) SwarmUI sends
  `VAEDecodeTiled.temporal_size = 9999`, but ComfyUI declares `max: 4096` for that input, so stock ComfyUI
  validation should reject it. This server relaxes that one limit so H3 works.
* **Upstream SwarmUI bug:** the EasyCache step checks the *TeaCache* mode for its "base gen only" and
  "video only" restrictions (`WorkflowGeneratorSteps.cs`, EasyCache block). With TeaCache off, EasyCache
  is applied to every stage, including image models, whatever EasyCache Mode says.

## 10. Pixal3D (the extension's tab)

SwarmUI's own generator never sends this: the [Pixal3D extension](../pixal3d-extension/README.md) builds a two-node graph
(more with background removal, or an `InvertMask` for Idol) from its tab and runs it on a ComfyUI-type backend it claims, outside SwarmUI's image
pipeline.

| Node | Inputs | Engine |
|---|---|---|
| `SwarmLoadImageB64` | the tab's image | IMAGE (RGB) and MASK (1 - alpha) |
| `LoadBackgroundRemovalModel`, `RemoveBackground` | only for an image without transparency: `bg_removal_name` from `background_removal/`, then the image | `AddonHandle` (`BACKGROUND_REMOVAL`), then `Engine::background_mask`: the foreground mask, which becomes `Pixal3DImageTo3D`'s `mask` |
| `InvertMask` | only for Idol without background removal: `SwarmLoadImageB64`'s mask, to make it 1 on the object | the mask's complement |
| `TripoSplatImageToSplat` | for Idol, instead of `Pixal3DImageTo3D`: `image`, `mask`, `unet_name`, `clip_vision_name`, `vae_name` (Flux 2), `decoder_name`, `seed`, `steps`, `cfg`, `num_gaussians`, `filename_prefix` | `ImageToSplatRequest` to `Engine::image_to_splat`, which returns PLY bytes |
| `Pixal3DImageTo3D` | `image`, `mask`, the four model files, `seed`, `fov`, `resolution`, the four stages' steps, `cfg`, `texture`, `max_faces`, `filename_prefix` | `ImageTo3DRequest` to `Engine::image_to_3d`, which returns GLB bytes |

`TripoSplatImageToSplat` takes the mask as `TripoSplatPreprocessImage` does, 1 on the object, and refuses files that
aren't the four it needs, a mask of another size, and gaussian counts that aren't a multiple of 32 from 32768 to 1048576.
For `Pixal3DImageTo3D`, the engine detects the mask's polarity like `ImageCropToMask`, and refuses images without a usable mask, the multi-view
checkpoint, and resolutions off the 1024 to 2048 grid of 128. `RemoveBackground` refuses anything but a BiRefNet
checkpoint. The extension then reads `history/<id>` for the saved file,
fetches it through `/view` and saves it in the SwarmUI user's output folder.

## 11. SeedVR2 (SeedVR Model)

SwarmUI's SeedVR stage (and a SeedVR2 refiner model) restores the current picture or video. Its `CreateSeedVR2Restore`
sends:

| Node | Inputs | Engine |
|---|---|---|
| `ImageScale` | only with SeedVR Upscale / Pre-Downscale, `pixel-*` methods | server |
| `SeedVR2Preprocess` | `resized_images` | server: RGB, clamped, zero-padded right and bottom to multiples of 16; a batch is one clip padded to 4n + 1 frames |
| `UNETLoader` / `CheckpointLoaderSimple`, `VAELoader` | the SeedVR model by folder; `SeedVR2/seedvr2_ema_vae_fp16.safetensors` | `ensure_seedvr`, `ensure_seedvr_vae` |
| `VAEEncode` | the padded picture or clip, SeedVR2's VAE | a `LatentFormat::SeedVr2` latent `[1, 16, H/8, W/8]`, or `[1, 16, n + 1, H/8, W/8]` for a clip of 4n + 1 frames |
| `SeedVR2TemporalChunk` | only with SeedVR Split Latent: `latent`, `temporal_overlap` (Temporal Video Overlap), `chunking_mode` `auto` | server: a list of latent chunks (`LatentOp::SliceFrames`), `auto` sized by `Engine::sample_frames_that_fit`; the nodes after it run once per chunk |
| `SeedVR2Conditioning` | `model`, `vae_conditioning` | server: positive and negative entries with `CondMeta::seedvr2` (the latent, and which stored prompt) and no embedding |
| `SwarmKSampler` | steps 1, CFG 1, `euler`, `simple` unless the SeedVR section sets them; seed = Seed + 9 | `Engine::sample`: the model's own stored prompts, the latent as `condition` |
| `SeedVR2TemporalMerge` | only with SeedVR Split Latent: the sampled chunks, the chunk node's overlap | server: one latent, overlaps crossfaded (`LatentOp::MergeChunks`) |
| `VAEDecode` | SeedVR2's VAE | f32 RGB, 4n + 1 frames for a clip |
| `SeedVR2PostProcessing` | `images`, `original_resized_images`, `color_correction_method` (`lab` default, `wavelet`, `adain`, `none`) | server: cropped to the picture, colour matched, alpha put back, sides cut to even |

A batch of pictures or a video's frames is one clip, as in ComfyUI. The engine tiles SeedVR2's VAE by itself when a
picture or clip doesn't fit, as ComfyUI does. It refuses `VAEDecodeTiled` / `VAEEncodeTiled` (VAE Tile Size), LoRAs,
other formats of the model, and Variation Seed or a noise mask on a clip.
