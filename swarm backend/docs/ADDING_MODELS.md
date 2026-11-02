# Adding a new model

How to get a new model family generating through SwarmUI → comfy-server → your engine. Read
[INPUT_MAPPING.md](INPUT_MAPPING.md) §1 first for how the layers fit.

## Who decides what

Three layers each own part of a model's behavior. Knowing which one a change belongs to is most of the work:

| Layer | Decides | Where |
|---|---|---|
| **SwarmUI** | Which *compat class* a file is, then which ComfyUI nodes to emit for it: loaders, text-encoder `type`, latent node, sampling-shift node, default sampler, prompt template | `src/Text2Image/T2IModelClassSorter.cs` (class detection: the safetensors `modelspec.architecture` header, else tensor-key sniffing), `src/BuiltinExtensions/ComfyUIBackend/WorkflowGenerator*.cs` (the `IsXxx()` branches) |
| **comfy-server** | How each node becomes typed engine requests, patches and metadata | `crates/comfy-server/src/nodes/` |
| **Engine** | The actual weights, architecture and math | your `impl Engine` |

The server never needs to know *which model* is running. It translates nodes. A new model therefore needs
server changes only when SwarmUI emits a node the server doesn't implement yet.

So start by finding out which nodes SwarmUI emits for the model.

## Step 1: capture what SwarmUI sends

1. **Make a fake model file** with the right architecture ID, so SwarmUI picks the right class without real
   weights. Add a line to `scratch/make_fake_models.py`, using the `ID` from `T2IModelClassSorter.cs`:

   ```python
   "diffusion_models/newmodel_test.safetensors": "new-model-id",
   ```

   Also add the text encoder and VAE files SwarmUI asks for by name. These are in the model's branch of
   `WorkflowGeneratorModelSupport.cs`, e.g. `qwen3vl_4b.safetensors` for Krea 2. Then run
   `python scratch/make_fake_models.py reference/SwarmUI/Models`.
2. **Start comfy-server with `--mock-native-only`.** That way `/object_info` shows exactly what's native,
   and anything else fails validation with a `missing_node_type` error naming the node. Then start SwarmUI
   pointed at it (see the README).
3. **Capture the workflow.** Add a case to a `scratch/cases_*.py` module, e.g.
   `CASES = {'newmodel_t2i': dict(K2, model='newmodel_test.safetensors')}`, and run
   `python scratch/capture.py scratch/cases_newmodel.py`. It saves the graph SwarmUI generates (via its
   `ComfyGetGeneratedWorkflow` API) and then runs it live. Capture each mode the model supports: text-to-image,
   prompt images, init image, video length, and so on.

## Step 2: triage the nodes

Sort every `class_type` in the captured graph into one of three buckets:

| Bucket | How to tell | What to do |
|---|---|---|
| **Native already** | Listed in `/object_info` under `--mock-native-only` | Nothing on the server. Check the arguments against [INPUT_MAPPING.md](INPUT_MAPPING.md). |
| **In the catalog** | Listed in [`todo_nodes.json`](../crates/comfy-server/src/todo_nodes.json) / [COMPATIBILITY.md](COMPATIBILITY.md) (most stock nodes for Wan, LTX-V, Hunyuan, SD3, Cosmos, ...) | Pick a route: the generic route (Step 3a) or native support (Step 3b). |
| **Unknown** | Neither (a node ComfyUI added after our stock dump) | Regenerate the catalog from a newer ComfyUI `/object_info` (see the end of COMPATIBILITY.md), or write it natively. |

Most new image models reuse the shared nodes (`UNETLoader`, `CLIPLoader`, `VAELoader`,
`SwarmTextEncodeAdvanced`, `SwarmKSampler`, `VAEDecode`). They differ only in their empty-latent node and
their sampling-shift node, which are often already native too. For example, MOSS Image needed nothing new
on the server beyond `EmptyFlux2LatentImage`.

**Text-encoder `type` strings.** `CLIPLoader.type` is a combo, so validation rejects a value it doesn't
list (`value_not_in_list`). If SwarmUI sends a new one, add it to `CLIP_TYPES` / `DUAL_CLIP_TYPES` in
[`loaders.rs`](../crates/comfy-server/src/nodes/loaders.rs), keeping ComfyUI's order.

## Step 3a: the generic route (fastest, no server code)

Every catalog node with a schema can run through two engine methods:

```rust
fn supports_custom_node(&self, class: &str) -> bool {
    matches!(class, "WanImageToVideo" | "ModelSamplingSD3" /* ... */)
}

fn custom_node(&self, req: &CustomNodeRequest, progress: &mut dyn ProgressSink) -> EngineResult<Vec<NodeValue>> {
    match req.class_type.as_str() {
        "WanImageToVideo" => { /* req.inputs["positive"], ["vae"], ["start_image"], ["width"], ... */ }
        _ => Err(EngineError::Unsupported(req.class_type.clone())),
    }
}
```

* Claimed classes are registered at startup using their stock schema, and listed in `/object_info`. SwarmUI
  enables a feature only when it sees the node, so **claim only what `custom_node` really handles**.
* `req.inputs` holds the resolved inputs as `NodeValue`s, after links, widgets and V3 nested dicts.
  Model-file combos (`ckpt_name`, `unet_name`, `x_model_folders`, ...) arrive as `NodeValue::File` with the
  path on disk.
* Return one `NodeValue` per entry in `req.output_types`. Engine-private types such as `WANVIDIMAGE_EMBEDS`
  go in `NodeValue::Other { ty, data: Opaque::new(..) }`, and they come back unchanged to whichever node
  consumes them.
* Natively implemented nodes in the same graph (loaders, `SwarmKSampler`, VAE decode, saving) still use the
  typed path. Only the unfamiliar nodes go through `custom_node`.

This is how Wan 2.1 text-to-video and the Krea 2 → Wan 2.1 image-to-video chain were verified live
(`other_architecture_nodes_go_through_custom_node` in `swarm_compat.rs`). It's the right choice for a model you
want working soon, or for nodes whose inputs are only meaningful to the engine.

## Step 3b: native support (typed requests)

Make a node native when the server can do real work for it, or when a typed request makes the engine's job
clearer. Typical cases: a new latent shape (so `Latent::conform_empty`, video frame math and the mock all
know it), pixel and mask work, or a patch that should stack with the others in graph order. Here is what
each kind of node needs, with the existing model it can be copied from:

| Node kind | engine-api change | server change | Copy from |
|---|---|---|---|
| Empty latent (`EmptyXxxLatent`) | `EmptyLatentKind::Xxx` in [`requests.rs`](../crates/engine-api/src/requests.rs), `LatentFormat::Xxx` in [`latent.rs`](../crates/engine-api/src/latent.rs) | register in [`sampling.rs`](../crates/comfy-server/src/nodes/sampling.rs) → `empty_latent(ctx, kind, batch)` | `EmptyFlux2LatentImage`, `SwarmEmptyMiniMaxH3LatentAV` |
| Sampling shift / model patch | `ModelPatch::Xxx` in [`patches.rs`](../crates/engine-api/src/patches.rs) | `model.with_patch(..)` in `sampling.rs` or [`guidance.rs`](../crates/comfy-server/src/nodes/guidance.rs) | `MiniMaxH3SigmaShift`, `ModelSamplingAuraFlow` |
| Model-specific conditioning (keyframes, reference latents, image embeds) | a field on `CondMeta` in [`conditioning.rs`](../crates/engine-api/src/conditioning.rs) | set it on each `CondEntry` in [`conditioning.rs`](../crates/comfy-server/src/nodes/conditioning.rs) / [`video.rs`](../crates/comfy-server/src/nodes/video.rs) | `SwarmMiniMaxH3AddKeyframes`, `ReferenceLatent` |
| Latent arithmetic | `LatentOp::Xxx` | call `ctx.engine.latent_op` | `ReplaceVideoLatentFrames` |
| Something only the engine can compute | a new `Engine` method with an "unsupported" default | call it from the node | `clip_vision_encode`, `upscale_image` |
| Pure pixel / mask / int work | none | implement it in the node | `ResizeImageMaskNode`, `SwarmTrimFrames` |

Keep the schema identical to stock ComfyUI: names, types, defaults, ranges, combo options, and output order.
`stock_schema_parity` in `swarm_compat.rs` checks every native node against
`tests/fixtures/stock_schemas.json`, and `tools/check_schemas.py` diffs a live dump. A schema mismatch
means SwarmUI sends something the server rejects, or feature detection quietly turns off.

After making a node native:

1. Remove it from `tools/todo_catalog.py`. A native registration always wins over the catalog, so leaving
   it in does no harm, but the docs would still list it as generic.
2. Teach the mock engine: give the new `EmptyLatentKind` a shape in
   [`mock_engine.rs`](../crates/comfy-server/src/mock_engine.rs), plus any new `LatentOp` or `Engine` method.
   The integration tests run on the mock.
3. Regenerate the catalog and the compatibility doc (`sh scratch/regen.sh`, or the steps at the end of
   COMPATIBILITY.md).

## Step 4: the engine side

Whichever route the nodes take, this part is always engine work:

* **Identify the architecture on load.** `load_diffusion_model` / `load_checkpoint` get a `ModelFile` with the
  path on disk. Read the safetensors header (`modelspec.architecture`, or the tensor keys) the way SwarmUI
  does. Nothing in the request names the model family.
* **Text encoder.** Dispatch on `LoadTextEncoder.encoder_type` plus the file. Some type strings are shared:
  `qwen_image` covers both Qwen Image 1 and 2.1, and `stable_diffusion` is also Anima's Qwen3 0.6B. In
  `encode_text`, honor `TextEncodeMode::Plain` and `Swarm`, `SwarmTextEncode.llama_template`, and any
  `images` / `clip_vision` the model accepts.
* **Latent and sampling.** Produce the right tensor for the `EmptyLatentKind`. In `sample`, apply
  `ModelHandle.patches` in order, including the new shift patch, and call `Latent::conform_empty` if
  SwarmUI sends a placeholder latent (as for Qwen Image 2.1).
* **VAE.** Return RGBA from `vae_decode` if the VAE has alpha. For video, return frames as
  `PixelData::U8` to keep memory down.
* **Warmup.** `warmup` is what SwarmUI's "load model" button and first-use preload call.

## Step 5: tests and docs

1. **Fixtures.** Copy each captured workflow into `crates/comfy-server/tests/fixtures/<model>_<mode>.json`.
   `python scratch/make_fixture.py <capture>:<fixture>` swaps the media for placeholders and maps the model names to the test suite's files.
2. **Tests** in [`swarm_compat.rs`](../crates/comfy-server/tests/swarm_compat.rs):
   * A smoke test runs the fixture end to end, like `moss_image_text_to_image`:
     `run_like_swarm(&s, fixture(..))`, then assert success and the output size and channels.
   * A recording test checks what the engine saw: use `RecordingEngine` with `TestServer::start_engine`,
     then assert on the latent kind, the patches in order, and the conditioning metadata.
   * Add the fake model files to `MODEL_FILES` so the loaders' combos list them.
3. **Parameter probe.** Add the model to `MODELS` in `tools/probe_swarm_params.py`, and to `MODELS` /
   `MODEL_NAMES` in `tools/gen_compat_doc.py`. Re-run the probe so COMPATIBILITY.md covers every SwarmUI
   parameter for the new model. This is how model-specific quirks surface, such as Qwen Image 2.1's
   half-size placeholder latent and MOSS reusing the Krea 2 template.
4. **INPUT_MAPPING.md.** Add a column to the op table and a section in the same shape as the existing
   ones: detected class, files, graph, the param → request table, and the engine-side work.

## A model SwarmUI doesn't know yet

If `T2IModelClassSorter.cs` has no class for the model, SwarmUI can't build a workflow for it. The fix
belongs upstream in SwarmUI: register the class, then add `IsXxx()` branches in
`WorkflowGeneratorModelSupport.cs` / `WorkflowGenerator.cs`. Until that lands, users can still run the
model from SwarmUI's Comfy Workflow tab with a hand-built graph. Any node in that graph goes through the
routes above.
