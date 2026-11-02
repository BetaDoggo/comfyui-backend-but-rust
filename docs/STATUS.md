# Status

What works and what doesn't, by status and priority. P0 comes first in each group.

* **Working:** really generates, or really works end to end, tested through SwarmUI.
* **Doing:** partly done.
* **Plumbing:** the server accepts it and passes it on, but only the mock engine receives it.
* **Todo:** not started.

[COMPATIBILITY.md](../swarm%20backend/docs/COMPATIBILITY.md) answers a different question: whether the server accepts a parameter. Its "Works" can mean only the mock engine has handled it, and this page is what counts.

## Working

| Priority | Area | Item | Notes |
|---|---|---|---|
| P0 | Server | ComfyUI HTTP and websocket protocol | /prompt, /history, /queue, /interrupt, /free, /object_info and the binary image frames. Tested against real SwarmUI. |
| P0 | Server | Swarm backend connection | Connects as "ComfyUI API By URL". Swarm reads the model list and features and generates. |
| P0 | Server | Runs beside a real ComfyUI backend | Live test: Krea 2 stayed here and Anima went to ComfyUI. First job isn't pinned without Exact Backend ID. |
| P0 | Server | Model folder scanning | Uses Swarm's and ComfyUI's folder names, including Stable-diffusion, clip and VAE. |
| P0 | Server | Start script | start-backend.bat builds, starts the server and prints the connection steps. |
| P0 | Engine core | Low-memory block streaming | Packed blocks in pinned RAM, streamed through VRAM slots on a copy stream. |
| P0 | Engine core | Cancelling and Free VRAM | Interrupt stops at the next step. Free VRAM unloads models, and they reload on next use. |
| P0 | Engine core | Sampler: er_sde | Anima and MOSS Image, generated through SwarmUI. Noise follows torch's CUDA Philox layout on this GPU. Other GPUs untested. |
| P0 | Krea 2 | Turbo text-to-image | W8A8 int8 DiT, 4.65 s at 1024. Pixel-identical to the CLI through SwarmUI. |
| P0 | Krea 2 | Seed parity with SwarmKSampler | Per-image seeds match Swarm's noise code. Batches confirmed against single renders. |
| P0 | Krea 2 | Batch Size | Direct workflow test. Not yet tried from Swarm's UI. |
| P0 | Krea 2 | LoRAs | Baked into the int8 ConvRot weights like ComfyUI. Weights bit-identical on the modules checked, 31.6 dB image, about 3 s to bake 12 GB. Tested through SwarmUI. |
| P0 | Anima | Anima engine | Int8 turbo checkpoint through real SwarmUI: 115 ms/step, 1.14 s per image at 1024, against 199 ms/step in ComfyUI. Same picture as ComfyUI at about 25 dB, not pixel-identical. |
| P0 | Anima | Anima LoRAs | Baked into the int8 ConvRot weights like ComfyUI, and cached contexts and modulation are dropped. Weights bit-identical on the modules checked. Tested through SwarmUI. |
| P0 | LoRA and control | LoRA formats | Every ComfyUI format on all three models: LoCon, DoRA, LoHa, LoKr, GLoRA, OFT, BOFT, diff, set, MOSS control adapters. Compared with ComfyUI on real modules, identical or within 2e-4 where ComfyUI can compute them. |
| P0 | LoRA and control | Text encoder LoRAs | Baked in float32 at the clip strength. Weights match ComfyUI on Krea 2 and MOSS Image. Run through the server on Anima, not through SwarmUI. |
| P0 | MOSS Image | MOSS Image LoRAs | Baked like ComfyUI, including fused q/k/v slices. All 168 weights identical. Tested through SwarmUI with the DMD LoRA; the edit LoRA applies but has no image input yet. |
| P0 | Engine core | Sampler: euler for all flow models | Every flow model now samples on the host, Krea 2 included. Real images on all three. |
| P0 | Engine core | Sampler: res_multistep | Ported with the other samplers. Tested on Anima against ComfyUI; 25 dB with flash attention. |
| P0 | Engine core | CFG and negative prompts | All three models, tested through SwarmUI. |
| P0 | Krea 2 | CFG above 1 and negative prompts | Tested through SwarmUI at CFG 2 with prompt images and references. Compared with ComfyUI only with a reference latent, at 24.9 dB. |
| P0 | Image workflows | VAE encoder | Qwen Image VAE encoder, 0.6% from ComfyUI, about 50 ms at 768. Used by Krea 2 reference latents. The Flux 2 VAE has no encoder yet; init images still need partial sampling. |
| P1 | Krea 2 | Variation Seed | Direct workflow test; output changes as expected. Not compared with ComfyUI. |
| P1 | Krea 2 | Sigma Shift | ModelSamplingFlux. Direct workflow test. |
| P1 | MiniMax H3 | H3 text to video with audio | Through real SwarmUI and the CLI: w6a8 DiT, nvfp4 text encoder, turbo LoRA, video and audio decoders, mp4 through ffmpeg. 512x512, 5 and 22 frames, 0.24 to 0.53 s/step against 1.59 in ComfyUI; a warm repeat takes 2.6 to 3.8 s. Same picture at 21 dB; both are 1.2% from an exact refiner run. Audio not judged by ear. |
| P1 | Engine core | Idle VRAM release | Diffusion blocks move to RAM after 5 minutes with no work (IDLE_RELEASE_SECS). Direct workflow test. |
| P1 | Engine core | Lazy offloading | H3 reads weights on use through a few pinned staging buffers, ahead of compute; LoRA-baked blocks stay pinned and skip staging. Other models still pin everything at load. |
| P1 | MOSS Image | MOSS engine | Int8 ConvRot checkpoint through real SwarmUI: 550 to 573 ms/step at CFG 4, against 601 in ComfyUI, VAE 268 ms against 530 or more. Same picture at about 34 dB, not pixel-identical. |
| P1 | Anima | Anima prompt weights | (word:1.2) and <weight> scale the T5 tokens, as ComfyUI does. Token ids match ComfyUI. |
| P1 | Anima | Anima Batch Size, Sigma Shift and CFG | Generated through real SwarmUI. Output not compared with ComfyUI at CFG above 1. |
| P1 | MOSS Image | MOSS Image Batch Size, Sigma Shift and CFG | Generated through real SwarmUI at 1024x1024 and 832x1216. Euler at CFG 4 compared with ComfyUI; er_sde and CFG 1 not compared. |
| P1 | Engine core | Samplers: dpmpp_2m, dpmpp_2m_sde, uni_pc, ddim | All 45 ComfyUI samplers: within 8e-6 of ComfyUI's own code on a mock model. Real images on all three models, through the server API and CLI, not the SwarmUI page. dpm_adaptive picks steps per image, not per batch. |
| P1 | Engine core | Schedulers: normal, karras, sgm_uniform, beta | All 9 ComfyUI schedulers plus turbo, align_your_steps, flux2 and ideogram4. Within 2e-6 of ComfyUI on 1105 schedules. ltxv-image is refused. |
| P2 | Engine core | Sigma overrides | Sigma Min, Sigma Max and Rho for karras and exponential, as SwarmKSampler does. Run once through the server API. |
| P2 | Krea 2 | Prompt Images | Vision tower and DeepStack, up to several images, on both prompts. Tokens match SwarmUI's text node, 40.8 dB from ComfyUI, tested through SwarmUI. A 1024 image costs 3.9 s against 130 ms at 384. |
| P1 | Engine core | Live previews | Off unless the server gets `--live-previews`. All four families, through the ComfyUI latent-to-RGB factors, honouring Swarm's preview modes. Direct workflow test: previews arrive and the final image is pixel-identical without them. Not looked at in SwarmUI's page. H3 shows the first frame only, where ComfyUI can animate. |
| P1 | Engine core | Conditioning Multiplier, Negative Conditioning Multiplier, Zero Negative | All four families in the engine. Direct workflow tests on Krea 2, MOSS Image and Anima; H3 not run. Krea 2 and MOSS Image match ComfyUI's picture at 23 to 26 dB, as their plain images do. Zero Negative's mottled texture at CFG above 1 is in ComfyUI too. |
| P1 | Krea 2 | <break> | Chunks encoded apart and joined. 27.9 dB from ComfyUI at 768, direct workflow test. Refused with Prompt Images. |
| P1 | MOSS Image | MOSS Image <weight> and <break> | Tokens and weights match SwarmTextEncodeAdvanced's own code. 25.9 and 31.6 dB from ComfyUI at 768, direct workflow test. Prompts now use the model's template, as SwarmUI does without images. |
| P1 | MiniMax H3 | H3 <weight> | Tokens and weights match SwarmTextEncodeAdvanced's own code. Direct workflow test, output changes as expected. Not compared with ComfyUI on a real run. |
| P1 | MiniMax H3 | H3 Variation Seed | Noise matches swarm_fixed_noise on a small latent. Direct workflow test. Not compared with ComfyUI on a real run. |
| P2 | Guidance and speed | CFG tweaks | Rescale, Renorm, CFG Zero Star and TCFG on Krea 2, Anima and MOSS Image, against ComfyUI's own code on random tensors, within 2e-5. Sample of real output by direct workflow test, not compared with ComfyUI. Refused on H3. |
| P2 | Krea 2 | Reference latents | index and index_timestep_zero, tested through SwarmUI. 1% from ComfyUI on the first evaluation; images 25 dB, about what int8 and flash differ by here (27 and 19 dB). No Krea 2 edit LoRA installed, so base model only. |

## Doing

| Priority | Area | Item | Notes |
|---|---|---|---|
| P0 | Engine core | Route each model to its engine | One backend now serves Krea 2, Anima, MOSS Image and MiniMax H3, picked from the checkpoint's keys. Still inside native-engine; other families need a real router. |
| P0 | Engine core | Dynamic VRAM offloading | Live residency works and adapted to ComfyUI's memory. Open: 16 to 11 drop at 1024, one 867 ms/step run after ComfyUI. |
| P1 | Krea 2 | Model Attention Backend | int8 and flash, per request. flash not yet run through the server. |

## Plumbing

| Priority | Area | Item | Notes |
|---|---|---|---|
| P0 | Server | Graph validation and execution | 163 native nodes with stock schemas. Exercised with real workflows only for Krea 2 text-to-image. |
| P0 | Server | Swarm prompt syntax parsing | <break>, <weight>, <alternate>, <fromto>, <embed> and LoRA hooks parse. See the rows for each family and the Todo rows for what is refused. |
| P0 | Image workflows | Init image and Creativity | Graph runs on the mock. Needs the VAE encoder and partial sampling. |
| P0 | Image workflows | Refiner and upscale | Both latent and model upscale paths reach the engine API. |
| P0 | LoRA and control | LoRAs on non-quantised models | Loader, patches and hooks reach the engine. No non-quantised model runs yet; int8 LoRAs work. The LoRA formats are done in src/lora.rs. |
| P1 | Server | Mask and image nodes | Crop, scale, blur, grow, threshold and composite run in the server. Not yet used with real generations. |
| P1 | Server | Video and audio output | mp4, webm, mov, gif and audio encoding through ffmpeg. H3 mp4 with audio works through SwarmUI. The other formats have only had placeholder frames. |
| P1 | Image workflows | Masks and inpainting | Noise masks, Differential Diffusion and Use Inpainting Encode. |
| P1 | Image workflows | Model upscalers | ESRGAN-family networks through upscale_image. |
| P1 | Image workflows | VAE tiling | VAE Tile Size, for large images on small cards. |
| P1 | LoRA and control | ControlNet One, Two and Three | Hints arrive in conditioning metadata, with union types. |
| P1 | Masks and detection | <segment:text> with CLIPSeg | Face and object refining. Needs a CLIPSeg model. |
| P1 | Masks and detection | <segment:yolo-...> | Needs YOLOv8 inference. |
| P1 | Guidance and speed | EasyCache | Often speeds up long runs. Needs step-output reuse in the sampler. |
| P2 | Server | Generic custom_node route | 92 extra node classes can be routed to an engine that claims them. Only the mock does. |
| P2 | Image workflows | Regions: <region> and <object> | Mask maths is done. The sampler has to apply per-region conditioning. |
| P2 | Image workflows | Unsampler Prompt | Sampling in reverse. |
| P2 | Image workflows | Seamless Tileable | Circular padding in the model and VAE. |
| P2 | LoRA and control | Style models and CLIP Vision | Needs a CLIP Vision encoder. |
| P2 | LoRA and control | Use Reference Only | Self-attention key and value extension. |
| P2 | LoRA and control | GLIGEN | Boxes reach the engine. |
| P2 | LoRA and control | IP-Adapter | Generic route only. |
| P2 | Masks and detection | Remove Background | Needs a background removal model. |
| P2 | Guidance and speed | Attention guidance | SAG, PAG and NAG. |
| P2 | Guidance and speed | FreeU | UNet models only. |
| P2 | Video and audio | Video generation workflow | Frames, swap, extend, boomerang, trim and end image, on the mock. |
| P3 | LoRA and control | ControlNet preprocessors | Third-party nodes. Generic route only. |
| P3 | Masks and detection | SAM2 points and box | Generic route only. |
| P3 | Guidance and speed | Sparse attention and HyperTile | Patches reach the engine. |
| P3 | Guidance and speed | Dynamic Thresholding and TeaCache | Third-party. Generic route only. |
| P3 | Video and audio | Video frame interpolation | RIFE and FILM. Generic route only. |

## Todo

| Priority | Area | Item | Notes |
|---|---|---|---|
| P0 | Engine core | Move Krea 2 behind a shared engine crate | Share the CUDA runtime, kernels, offloading and sampler between model families instead of hard-coding Krea 2. |
| P0 | Engine core | Latent operations | Add, multiply, blend, batch, repeat and upscale. Needed by init images, refiners and regions. |
| P1 | Engine core | Hardware beyond this card | Everything targets sm_120 on Windows. Other GPUs and Linux are untested. |
| P1 | Krea 2 | Prompt weights | <weight> through SwarmAttnTokenWeights, inside attention. |
| P1 | Krea 2 | <alternate> and <fromto> | Per-step conditioning. |
| P1 | Krea 2 | Non-W8A8 Krea 2 files | fp8 and other formats are refused. |
| P1 | Qwen Image 2.1 | Qwen Image 2.1 engine | Installed. Qwen3-VL 8B text encoder, RGBA VAE, 2x placeholder latent. |
| P1 | SD family | SD1 and SDXL engine | SD1 checkpoint installed. CLIP, UNet and classic VAE, and all-in-one checkpoints. Would widen coverage most. |
| P1 | LoRA and control | LoRA hooks | Refused with an error. |
| P1 | Anima | Other Anima checkpoint formats | bf16 and fp8 files are refused. Only int8 ConvRot runs. |
| P1 | Anima | Close the ComfyUI output gap | LLM adapter output differs by 1.7%, cause unknown. Same picture at about 25 dB PSNR. |
| P1 | Anima | Anima <break>, <alternate> and <fromto> | Per-chunk and per-step conditioning. |
| P1 | MiniMax H3 | H3 first and last frame, references | Needs the video and audio encoders and the 32B vision tower. The turbo LoRA is trained for these modes. |
| P1 | MiniMax H3 | H3 long clips | Longer than 22 frames and above 512x512 untested. Sparse attention isn't ported; its source is in comfy-kitchen. |
| P1 | MOSS Image | MOSS Image reference images | Prompt Images and edit LoRAs need Qwen3.5's vision encoder, the VAE encoder and reference tokens. |
| P1 | MOSS Image | Other MOSS Image checkpoint formats | bf16 and unfused q/k/v files are refused. Only fused int8 ConvRot runs. |
| P2 | Server | ComfyUI node editor page | SwarmUI's Comfy Workflow tab won't render. Generation doesn't need it. |
| P2 | Engine core | Tiled sampling | Swarm's Tile Sample and refiner tiling. |
| P2 | Other models | Flux and Flux 2 | Very common in Swarm. T5 and CLIP encoders, GGUF and NF4 formats. |
| P2 | Anima | Anima LLLite ControlNet | Needs the patch hooks inside the blocks. |
| P2 | MOSS Image | Close the MOSS Image output gap | torch's bf16 GEMM isn't exact at text-embedder sizes, so that stage differs by 0.5%. Same picture at about 34 dB PSNR. |
| P3 | Other models | Wan, LTX-V, Hunyuan and other video | 63 stock nodes across model families. |
| P3 | Video and audio | Text-to-audio models | ACE-Step, YuE and MiniMax Music. |
