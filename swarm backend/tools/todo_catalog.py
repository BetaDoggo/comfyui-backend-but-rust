"""Hand-curated classification of every ComfyUI node class SwarmUI's workflow generator can emit that
comfy-server does not implement yet. Consumed by gen_todo_nodes.py.

`needs` says what implementing the node takes:
  server  - pure data work the server can do itself (images, masks, ints, metadata on existing values)
  patch   - bookkeeping only on the server (a new ModelPatch / CondMeta / Latent field), but the engine
            must honor the new field while sampling/encoding
  engine  - needs a new Engine operation (weights, tensor math on engine-owned latents, a model run)
  arch    - part of supporting a model architecture this engine doesn't target yet
"""

AREAS = {
    'other_arch': ('Other model architectures', 'Stock ComfyUI nodes for model families beyond the five targets (SD1/SDXL, SD3, Flux, Wan, LTX-V, Hunyuan, Cosmos, ...). Routed to Engine::custom_node when the engine claims them'),
    'third_party': ('Third-party custom nodes (optional features)', 'SwarmUI only uses these when the backend advertises them; real ComfyUI needs extra node packs for them. Routed to Engine::custom_node when the engine claims them'),
}

# class -> (area, needs, note)
NODES = {
    # ---------------------------------------------------------------- inpainting / masks
    # ---------------------------------------------------------------- regional / segmentation
    # ---------------------------------------------------------------- refiner / upscale / latent ops
    # ---------------------------------------------------------------- sampling / guidance
    # ---------------------------------------------------------------- image prompting
    # ---------------------------------------------------------------- controlnet
    # ---------------------------------------------------------------- video / audio plumbing
    # ---------------------------------------------------------------- other architectures
    'CLIPTextEncodeSDXL': ('other_arch', 'arch', 'SDXL'),
    'InstructPixToPixConditioning': ('other_arch', 'arch', 'SD1.5 InstructPix2Pix'),
    'SVD_img2vid_Conditioning': ('other_arch', 'arch', 'Stable Video Diffusion'),
    'SwarmOffsetEmptyLatentImage': ('other_arch', 'arch', 'SD-family offset-noise latents'),
    'TripleCLIPLoader': ('other_arch', 'arch', 'SD3'),
    'T5TokenizerOptions': ('other_arch', 'arch', 'T5-based models'),
    'QuadrupleCLIPLoader': ('other_arch', 'arch', 'HiDream'),
    'EmptyHiDreamO1LatentImage': ('other_arch', 'arch', 'HiDream O1'),
    'HiDreamO1PatchSeamSmoothing': ('other_arch', 'arch', 'HiDream O1'),
    'HiDreamO1ReferenceImages': ('other_arch', 'arch', 'HiDream O1'),
    'FluxDisableGuidance': ('other_arch', 'arch', 'Flux dev (Flux Disable Guidance)'),
    'FluxKVCache': ('other_arch', 'arch', 'Flux'),
    'StableCascade_EmptyLatentImage': ('other_arch', 'arch', 'Stable Cascade'),
    'StableCascade_StageB_Conditioning': ('other_arch', 'arch', 'Stable Cascade'),
    'StableCascade_StageC_VAEEncode': ('other_arch', 'arch', 'Stable Cascade'),
    'EmptyCosmosLatentVideo': ('other_arch', 'arch', 'Cosmos'),
    'CosmosImageToVideoLatent': ('other_arch', 'arch', 'Cosmos'),
    'EmptyHunyuanImageLatent': ('other_arch', 'arch', 'Hunyuan Image'),
    'EmptyHunyuanLatentVideo': ('other_arch', 'arch', 'Hunyuan Video'),
    'EmptyHunyuanVideo15Latent': ('other_arch', 'arch', 'Hunyuan Video 1.5'),
    'HunyuanImageToVideo': ('other_arch', 'arch', 'Hunyuan Video'),
    'HunyuanRefinerLatent': ('other_arch', 'arch', 'Hunyuan Image refiner'),
    'HunyuanVideo15ImageToVideo': ('other_arch', 'arch', 'Hunyuan Video 1.5'),
    'HunyuanVideo15LatentUpscaleWithModel': ('other_arch', 'arch', 'Hunyuan Video 1.5'),
    'HunyuanVideo15SuperResolution': ('other_arch', 'arch', 'Hunyuan Video 1.5'),
    'TextEncodeHunyuanVideo_ImageToVideo': ('other_arch', 'arch', 'Hunyuan Video'),
    'EmptyLTXVLatentVideo': ('other_arch', 'arch', 'LTX-V'),
    'LTXAVTextEncoderLoader': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'LTXVAddGuide': ('other_arch', 'arch', 'LTX-V'),
    'LTXVAudioVAEDecode': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'LTXVAudioVAEEncode': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'LTXVAudioVAELoader': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'SwarmLTXVAudioVAELoader': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'LTXVConditioning': ('other_arch', 'arch', 'LTX-V'),
    'LTXVCropGuides': ('other_arch', 'arch', 'LTX-V'),
    'LTXVEmptyLatentAudio': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'LTXVImgToVideo': ('other_arch', 'arch', 'LTX-V'),
    'LTXVImgToVideoInplace': ('other_arch', 'arch', 'LTX-V'),
    'LTXVLatentUpsampler': ('other_arch', 'arch', 'LTX-V'),
    'LTXVPreprocess': ('other_arch', 'arch', 'LTX-V'),
    'LTXVReferenceAudio': ('other_arch', 'arch', 'LTX-V 2 audio-video'),
    'EmptyMochiLatentVideo': ('other_arch', 'arch', 'Genmo Mochi'),
    'WanImageToVideo': ('other_arch', 'arch', 'Wan 2.1'),
    'WanFirstLastFrameToVideo': ('other_arch', 'arch', 'Wan 2.1 FLF2V'),
    'Wan22ImageToVideoLatent': ('other_arch', 'arch', 'Wan 2.2'),
    'WanPhantomSubjectToVideo': ('other_arch', 'arch', 'Wan Phantom'),
    'WanVaceToVideo': ('other_arch', 'arch', 'Wan VACE'),
    'Kandinsky5ImageToVideo': ('other_arch', 'arch', 'Kandinsky 5'),
    'EmptyChromaRadianceLatentImage': ('other_arch', 'arch', 'Chroma Radiance (pixel space)'),
    'PiDConditioning': ('other_arch', 'arch', 'Pixel decoders (Pixel Decoder Model / <pixeldecoder>)'),
    'EmptyMiniMaxMusic3LatentAudio': ('other_arch', 'arch', 'MiniMax Music 3'),
    'MiniMaxMusic3TextEncode': ('other_arch', 'arch', 'MiniMax Music 3'),
    'EmptyYuE2LatentAudio': ('other_arch', 'arch', 'YuE 2'),
    'YuE2GenerateABC': ('other_arch', 'arch', 'YuE 2'),
    'YuE2GenerateMusic': ('other_arch', 'arch', 'YuE 2'),
    'SenseNovaSamplingOptions': ('other_arch', 'arch', 'SenseNova'),
    'TextEncodeQwenImageEdit': ('other_arch', 'arch', 'Qwen Image Edit (2509 and earlier)'),
    'TextEncodeQwenImageEditPlus': ('other_arch', 'arch', 'Qwen Image Edit Plus'),
    'SeedVR2Conditioning': ('other_arch', 'arch', 'SeedVR2 restoration (SeedVR Model param)'),
    'SeedVR2PostProcessing': ('other_arch', 'arch', 'SeedVR2 restoration (SeedVR Model param)'),
    'SeedVR2Preprocess': ('other_arch', 'arch', 'SeedVR2 restoration (SeedVR Model param)'),
    'SeedVR2TemporalChunk': ('other_arch', 'arch', 'SeedVR2 restoration (SeedVR Model param)'),
    'SeedVR2TemporalMerge': ('other_arch', 'arch', 'SeedVR2 restoration (SeedVR Model param)'),
    # ---------------------------------------------------------------- third-party (not in core ComfyUI)
    'IPAdapter': ('third_party', 'engine', 'Use IP-Adapter (ComfyUI_IPAdapter_plus).'),
    'IPAdapterAdvanced': ('third_party', 'engine', 'Use IP-Adapter (ComfyUI_IPAdapter_plus).'),
    'IPAdapterApply': ('third_party', 'engine', 'Use IP-Adapter (legacy node name).'),
    'IPAdapterModelLoader': ('third_party', 'engine', 'Use IP-Adapter (ComfyUI_IPAdapter_plus).'),
    'IPAdapterUnifiedLoader': ('third_party', 'engine', 'Use IP-Adapter (ComfyUI_IPAdapter_plus).'),
    'MiDaS-DepthMapPreprocessor': ('third_party', 'engine', 'ControlNet preprocessors (comfyui_controlnet_aux); any node in category image/preprocessors is offered.'),
    'RIFE VFI': ('third_party', 'engine', 'Video Frame Interpolation Method = RIFE (ComfyUI-Frame-Interpolation).'),
    'FILM VFI': ('third_party', 'engine', 'Video Frame Interpolation Method = FILM (ComfyUI-Frame-Interpolation).'),
    'GIMMVFI_interpolate': ('third_party', 'engine', 'Video Frame Interpolation Method = GIMM-VFI.'),
    'DownloadAndLoadGIMMVFIModel': ('third_party', 'engine', 'Video Frame Interpolation Method = GIMM-VFI.'),
    'Sam2Segmentation': ('third_party', 'engine', 'SAM2 points / bbox masking (ComfyUI-segment-anything-2).'),
    'DownloadAndLoadSAM2Model': ('third_party', 'engine', 'SAM2 (ComfyUI-segment-anything-2).'),
    'Sam2AutoSegmentation': ('third_party', 'engine', 'SAM2 auto-segment preprocessor.'),
    'UnetLoaderGGUF': ('third_party', 'engine', '.gguf diffusion models (ComfyUI-GGUF).'),
    'CheckpointLoaderNF4': ('third_party', 'engine', 'bitsandbytes NF4 checkpoints.'),
    'NunchakuFluxDiTLoader': ('third_party', 'engine', 'Nunchaku SVDQuant Flux models.'),
    'NunchakuQwenImageDiTLoader': ('third_party', 'engine', 'Nunchaku SVDQuant Qwen Image models.'),
    'NunchakuFluxLoraLoader': ('third_party', 'engine', 'LoRAs on Nunchaku Flux models.'),
    'TensorRTLoader': ('third_party', 'engine', 'TensorRT engines.'),
    'TeaCache': ('third_party', 'patch', 'TeaCache Mode/Threshold/Start (ComfyUI-TeaCache).'),
    'TeaCacheForImgGen': ('third_party', 'patch', 'TeaCache (old node name).'),
    'TeaCacheForVidGen': ('third_party', 'patch', 'TeaCache (old node name).'),
    'AITemplateLoader': ('third_party', 'engine', 'Enable AITemplate.'),
    'OverrideCLIPDevice': ('third_party', 'patch', 'Set CLIP Device.'),
    'PixArtCheckpointLoader': ('third_party', 'arch', 'PixArt (ExtraModels).'),
    'SanaCheckpointLoader': ('third_party', 'arch', 'Sana (ExtraModels).'),
    'SanaTextEncode': ('third_party', 'arch', 'Sana (ExtraModels).'),
    'GemmaLoader': ('third_party', 'arch', 'Sana (ExtraModels).'),
    'DynamicThresholdingFull': ('third_party', 'patch', '[DT] Dynamic Thresholding params (sd-dynamic-thresholding).'),
}
