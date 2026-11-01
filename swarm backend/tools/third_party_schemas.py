"""Writes tools/data/third_party_schemas.json: V1 /object_info entries for the third-party node
packs SwarmUI can use (IP-Adapter, ControlNet preprocessors, frame interpolation, SAM2, GGUF, NF4,
Nunchaku, TensorRT, TeaCache, AITemplate, ExtraModels, Dynamic Thresholding).

Stock ComfyUI can't produce these, so they're written by hand from the packs' public node
definitions, cross-checked against every input SwarmUI's generator sends (and every combo it reads
back from /object_info). gen_todo_nodes.py merges them into todo_nodes.json; the server advertises
them only for classes the engine claims via `Engine::supports_custom_node`.

`x_model_folders` (not part of ComfyUI's format) maps file-list inputs to the model folder they
list; the server fills those combos from its own model roots.

    python tools/third_party_schemas.py
"""
import json, math, pathlib

MAX_SEED = 0xffffffffffffffff


def INT(d, mn=0, mx=MAX_SEED, step=1, **kw):
    return ['INT', dict(default=d, min=mn, max=mx, step=step, **kw)]


def FLOAT(d, mn, mx, step, **kw):
    return ['FLOAT', dict(default=d, min=mn, max=mx, step=step, **kw)]


def BOOL(d, **kw):
    return ['BOOLEAN', dict(default=d, **kw)]


def COMBO(options, default=None, **kw):
    o = dict(kw)
    if default is not None:
        o['default'] = default
    return [list(options), o] if o else [list(options)]


def STRING(d='', multiline=False, **kw):
    return ['STRING', dict(default=d, multiline=multiline, **kw)]


def FILES():
    return [[]]


def node(cls, category, required, outputs, optional=None, display=None, folders=None, module='custom_nodes', description=''):
    names = [n for _, n in outputs]
    info = {
        'input': {'required': required, **({'optional': optional} if optional else {})},
        'input_order': {'required': list(required), **({'optional': list(optional)} if optional else {})},
        'output': [t for t, _ in outputs],
        'output_is_list': [False] * len(outputs),
        'output_name': names,
        'name': cls,
        'display_name': display or cls,
        'description': description,
        'python_module': module,
        'category': category,
        'output_node': False,
    }
    if folders:
        info['x_model_folders'] = folders
    return cls, info


IPA_WEIGHT_TYPES = ['linear', 'ease in', 'ease out', 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle',
                    'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']
IPA_PRESETS = ['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)',
               'FULL FACE - SD1.5 only (portraits stronger)']
SCHED_MODES = ['Constant', 'Linear Down', 'Half Cosine Down', 'Cosine Down', 'Linear Up', 'Half Cosine Up', 'Cosine Up', 'Power Up', 'Power Down', 'Linear Repeating', 'Cosine Repeating']
TEACACHE_TYPES = ['flux', 'flux-kontext', 'ltxv', 'lumina_2', 'hunyuan_video', 'hidream_i1_full', 'hidream_i1_dev', 'hidream_i1_fast', 'wan2.1_t2v_1.3B', 'wan2.1_t2v_14B',
                  'wan2.1_i2v_480p_14B', 'wan2.1_i2v_720p_14B', 'wan2.1_t2v_1.3B_ret_mode', 'wan2.1_t2v_14B_ret_mode', 'wan2.1_i2v_480p_14B_ret_mode', 'wan2.1_i2v_720p_14B_ret_mode']
SAM2_MODELS = ['sam2_hiera_base_plus.safetensors', 'sam2_hiera_large.safetensors', 'sam2_hiera_small.safetensors', 'sam2_hiera_tiny.safetensors',
               'sam2.1_hiera_base_plus.safetensors', 'sam2.1_hiera_large.safetensors', 'sam2.1_hiera_small.safetensors', 'sam2.1_hiera_tiny.safetensors']
RIFE_CKPTS = ['rife40.pth', 'rife41.pth', 'rife42.pth', 'rife43.pth', 'rife44.pth', 'rife45.pth', 'rife46.pth', 'rife47.pth', 'rife48.pth', 'rife49.pth',
              'sudo_rife4_269.662_testV1_scale1.pth']
TEACACHE = dict(model=['MODEL'], model_type=COMBO(TEACACHE_TYPES, 'flux'), rel_l1_thresh=FLOAT(0.4, 0.0, 10.0, 0.01), max_skip_steps=INT(3, 1, 100),
                start_percent=FLOAT(0.0, 0.0, 1.0, 0.01), end_percent=FLOAT(1.0, 0.0, 1.0, 0.01), cache_device=COMBO(['cuda', 'cpu'], 'cuda'))

NODES = dict([
    # ------------------------------------------------ ComfyUI_IPAdapter_plus (cubiq)
    node('IPAdapterUnifiedLoader', 'ipadapter', dict(model=['MODEL'], preset=COMBO(IPA_PRESETS)), [('MODEL', 'model'), ('IPADAPTER', 'ipadapter')],
         optional=dict(ipadapter=['IPADAPTER']), display='IPAdapter Unified Loader'),
    node('IPAdapterModelLoader', 'ipadapter/loaders', dict(ipadapter_file=FILES()), [('IPADAPTER', 'IPADAPTER')], display='IPAdapter Model Loader',
         folders=dict(ipadapter_file='ipadapter')),
    node('IPAdapter', 'ipadapter', dict(model=['MODEL'], ipadapter=['IPADAPTER'], image=['IMAGE'], weight=FLOAT(1.0, -1.0, 3.0, 0.05),
                                         start_at=FLOAT(0.0, 0.0, 1.0, 0.001), end_at=FLOAT(1.0, 0.0, 1.0, 0.001),
                                         weight_type=COMBO(['standard', 'prompt is more important', 'style transfer'])),
         [('MODEL', 'MODEL')], optional=dict(attn_mask=['MASK']), display='IPAdapter'),
    node('IPAdapterAdvanced', 'ipadapter', dict(model=['MODEL'], ipadapter=['IPADAPTER'], image=['IMAGE'], weight=FLOAT(1.0, -1.0, 5.0, 0.05),
                                                 weight_type=COMBO(IPA_WEIGHT_TYPES), combine_embeds=COMBO(['concat', 'add', 'subtract', 'average', 'norm average']),
                                                 start_at=FLOAT(0.0, 0.0, 1.0, 0.001), end_at=FLOAT(1.0, 0.0, 1.0, 0.001),
                                                 embeds_scaling=COMBO(['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'])),
         [('MODEL', 'MODEL')], optional=dict(image_negative=['IMAGE'], attn_mask=['MASK'], clip_vision=['CLIP_VISION']), display='IPAdapter Advanced'),
    # The pre-2024 cubiq API (SwarmUI still emits it when only this is available).
    node('IPAdapterApply', 'ipadapter', dict(ipadapter=['IPADAPTER'], clip_vision=['CLIP_VISION'], image=['IMAGE'], model=['MODEL'],
                                              weight=FLOAT(1.0, -1.0, 3.0, 0.05), noise=FLOAT(0.0, 0.0, 1.0, 0.01),
                                              weight_type=COMBO(['original', 'linear', 'channel penalty'])),
         [('MODEL', 'MODEL')], optional=dict(start_at=FLOAT(0.0, 0.0, 1.0, 0.001), end_at=FLOAT(1.0, 0.0, 1.0, 0.001), unfold_batch=BOOL(False), attn_mask=['MASK']),
         display='Apply IPAdapter'),
    # ------------------------------------------------ comfyui_controlnet_aux
    node('MiDaS-DepthMapPreprocessor', 'ControlNet Preprocessors/Normal and Depth Estimators', dict(image=['IMAGE']), [('IMAGE', 'IMAGE')],
         optional=dict(a=FLOAT(round(math.pi * 2, 6), 0.0, round(math.pi * 5, 6), 0.05), bg_threshold=FLOAT(0.1, 0.0, 1.0, 0.1),
                       resolution=INT(512, 64, 16384, 64)), display='MiDaS Depth Map'),
    # ------------------------------------------------ ComfyUI-Frame-Interpolation / ComfyUI-GIMM-VFI
    # RIFE's scale_factor is a float combo upstream ([0.25, 0.5, 1.0, 2.0, 4.0]); declared as FLOAT so integer literals validate.
    node('RIFE VFI', 'ComfyUI-Frame-Interpolation/VFI', dict(ckpt_name=COMBO(RIFE_CKPTS, 'rife47.pth'), frames=['IMAGE'], clear_cache_after_n_frames=INT(10, 1, 1000),
                                                         multiplier=INT(2, 1), fast_mode=BOOL(True), ensemble=BOOL(True), scale_factor=FLOAT(1.0, 0.25, 4.0, 0.25)),
         [('IMAGE', 'IMAGE')], optional=dict(optional_interpolation_states=['INTERPOLATION_STATES']), display='RIFE VFI (recommend rife47 and rife49)'),
    node('FILM VFI', 'ComfyUI-Frame-Interpolation/VFI', dict(ckpt_name=COMBO(['film_net_fp32.pt']), frames=['IMAGE'], clear_cache_after_n_frames=INT(10, 1, 1000),
                                                         multiplier=INT(2, 1)),
         [('IMAGE', 'IMAGE')], optional=dict(optional_interpolation_states=['INTERPOLATION_STATES']), display='FILM VFI'),
    node('DownloadAndLoadGIMMVFIModel', 'GIMM-VFI', dict(model=COMBO(['gimmvfi_r_arb_lpips_fp32.safetensors', 'gimmvfi_f_arb_lpips_fp32.safetensors']),
                                                         precision=COMBO(['fp32', 'bf16', 'fp16'], 'fp32'), torch_compile=BOOL(False)),
         [('GIMMVIF_MODEL', 'gimmvfi_model')], display='(Down)Load GIMMVFI Model'),
    node('GIMMVFI_interpolate', 'GIMM-VFI', dict(gimmvfi_model=['GIMMVIF_MODEL'], images=['IMAGE'], ds_factor=FLOAT(1.0, 0.01, 1.0, 0.01),
                                                 interpolation_factor=INT(8, 1, 100), seed=INT(0), output_flows=BOOL(False)),
         [('IMAGE', 'images'), ('IMAGE', 'flow_tensors')], display='GIMM-VFI Interpolate'),
    # ------------------------------------------------ ComfyUI-segment-anything-2 (kijai)
    node('DownloadAndLoadSAM2Model', 'SAM2', dict(model=COMBO(SAM2_MODELS), segmentor=COMBO(['single_image', 'video', 'automaskgenerator']),
                                                  device=COMBO(['cuda', 'cpu', 'mps']), precision=COMBO(['fp16', 'bf16', 'fp32'], 'fp16')),
         [('SAM2MODEL', 'sam2_model')], display='(Down)Load SAM2Model'),
    node('Sam2Segmentation', 'SAM2', dict(sam2_model=['SAM2MODEL'], image=['IMAGE'], keep_model_loaded=BOOL(True)), [('MASK', 'mask')],
         optional=dict(coordinates_positive=['STRING', dict(forceInput=True)], coordinates_negative=['STRING', dict(forceInput=True)], bboxes=['BBOX'],
                       individual_objects=BOOL(False), mask=['MASK']), display='Sam2Segmentation'),
    node('Sam2AutoSegmentation', 'SAM2', dict(sam2_model=['SAM2MODEL'], image=['IMAGE'], points_per_side=INT(32, 1, 100), points_per_batch=INT(64, 1, 100),
                                              pred_iou_thresh=FLOAT(0.8, 0.0, 1.0, 0.01), stability_score_thresh=FLOAT(0.95, 0.0, 1.0, 0.01),
                                              stability_score_offset=FLOAT(1.0, 0.0, 1.0, 0.01), mask_threshold=FLOAT(0.0, 0.0, 1.0, 0.01),
                                              crop_n_layers=INT(0, 0, 100), box_nms_thresh=FLOAT(0.7, 0.0, 1.0, 0.01), crop_nms_thresh=FLOAT(0.7, 0.0, 1.0, 0.01),
                                              crop_overlap_ratio=FLOAT(0.34133, 0.0, 1.0, 0.01), crop_n_points_downscale_factor=INT(1, 1, 100),
                                              min_mask_region_area=FLOAT(0.0, 0.0, 1.0, 0.01), use_m2m=BOOL(False), keep_model_loaded=BOOL(True)),
         [('MASK', 'mask'), ('IMAGE', 'segmented_image'), ('BBOX', 'bbox')], display='Sam2AutoSegmentation'),
    # ------------------------------------------------ quantized / accelerated loaders
    node('UnetLoaderGGUF', 'bootleg', dict(unet_name=FILES()), [('MODEL', 'MODEL')], display='Unet Loader (GGUF)', folders=dict(unet_name='diffusion_models')),
    node('CheckpointLoaderNF4', 'loaders', dict(ckpt_name=FILES()), [('MODEL', 'MODEL'), ('CLIP', 'CLIP'), ('VAE', 'VAE')],
         optional=dict(bnb_dtype=COMBO(['nf4', 'fp4'], 'nf4')), display='CheckpointLoaderNF4', folders=dict(ckpt_name='checkpoints')),
    node('NunchakuFluxDiTLoader', 'Nunchaku', dict(model_path=FILES(), cache_threshold=FLOAT(0.0, 0.0, 1.0, 0.001), attention=COMBO(['nunchaku-fp16', 'flash-attention2']),
                                                   cpu_offload=COMBO(['auto', 'enable', 'disable']), device_id=INT(0, 0, 64),
                                                   data_type=COMBO(['bfloat16', 'float16'])),
         [('MODEL', 'MODEL')], optional=dict(i2f_mode=COMBO(['enabled', 'always'])), display='Nunchaku FLUX DiT Loader', folders=dict(model_path='diffusion_models')),
    node('NunchakuQwenImageDiTLoader', 'Nunchaku', dict(model_name=FILES(), cpu_offload=COMBO(['auto', 'enable', 'disable']), num_blocks_on_gpu=INT(1, 1, 60),
                                                        use_pin_memory=COMBO(['enable', 'disable'])),
         [('MODEL', 'MODEL')], display='Nunchaku Qwen-Image DiT Loader', folders=dict(model_name='diffusion_models')),
    node('NunchakuFluxLoraLoader', 'Nunchaku', dict(model=['MODEL'], lora_name=FILES(), lora_strength=FLOAT(1.0, -100.0, 100.0, 0.01)), [('MODEL', 'MODEL')],
         display='Nunchaku FLUX LoRA Loader', folders=dict(lora_name='loras')),
    node('TensorRTLoader', 'TensorRT', dict(unet_name=FILES(), model_type=COMBO(['sdxl_base', 'sdxl_refiner', 'sd1.x', 'sd2.x-768v', 'svd', 'sd3', 'auraflow', 'flux_dev', 'flux_schnell'])),
         [('MODEL', 'MODEL')], display='TensorRT Loader', folders=dict(unet_name='tensorrt')),
    node('AITemplateLoader', 'loaders', dict(model=['MODEL'], keep_loaded=COMBO(['enable', 'disable'])), [('MODEL', 'MODEL')], display='Load AITemplate'),
    # ------------------------------------------------ ComfyUI-TeaCache (the old names take the same inputs)
    node('TeaCache', 'TeaCache', TEACACHE, [('MODEL', 'model')], display='TeaCache'),
    node('TeaCacheForImgGen', 'TeaCache', TEACACHE, [('MODEL', 'model')], display='TeaCache For Img Gen'),
    node('TeaCacheForVidGen', 'TeaCache', TEACACHE, [('MODEL', 'model')], display='TeaCache For Vid Gen'),
    # ------------------------------------------------ ComfyUI_ExtraModels (city96)
    node('OverrideCLIPDevice', 'other', dict(clip=['CLIP'], device=COMBO(['cpu', 'cuda:0'])), [('CLIP', 'CLIP')], display='Force/Set CLIP Device'),
    node('PixArtCheckpointLoader', 'ExtraModels/PixArt', dict(ckpt_name=FILES(), model=COMBO(['PixArtMS_XL_2', 'PixArt_XL_2', 'PixArtMS_Sigma_XL_2', 'PixArtMS_Sigma_XL_2_900M',
                                                                                         'PixArtMS_Sigma_XL_2_2K', 'PixArt_Sigma_XL_2_512', 'ControlPixArtHalf', 'ControlPixArtMSHalf'])),
         [('MODEL', 'model')], display='PixArt Checkpoint Loader', folders=dict(ckpt_name='checkpoints')),
    node('SanaCheckpointLoader', 'ExtraModels/Sana', dict(ckpt_name=FILES(), model=COMBO(['SanaMS_600M_P1_D28', 'SanaMS_1600M_P1_D20', 'SanaMS_1600M_P1_D20_2K', 'SanaMS_1600M_P1_D20_4K'])),
         [('MODEL', 'model')], display='Sana Checkpoint Loader', folders=dict(ckpt_name='checkpoints')),
    node('GemmaLoader', 'ExtraModels/Sana', dict(model_name=COMBO(['Efficient-Large-Model/gemma-2-2b-it', 'unsloth/gemma-2-2b-it-bnb-4bit']),
                                                 device=COMBO(['cpu', 'cuda']), dtype=COMBO(['default', 'auto (comfy)', 'FP32', 'FP16', 'BF16'])),
         [('GEMMA', 'GEMMA')], display='Gemma Loader'),
    node('SanaTextEncode', 'ExtraModels/Sana', dict(text=STRING('', True), GEMMA=['GEMMA']), [('CONDITIONING', 'CONDITIONING')], display='Sana Text Encode'),
    # ------------------------------------------------ sd-dynamic-thresholding
    node('DynamicThresholdingFull', 'advanced/mcmonkey', dict(model=['MODEL'], mimic_scale=FLOAT(7.0, 0.0, 100.0, 0.5), threshold_percentile=FLOAT(1.0, 0.0, 1.0, 0.001),
                                                              mimic_mode=COMBO(SCHED_MODES), mimic_scale_min=FLOAT(0.0, 0.0, 100.0, 0.5), cfg_mode=COMBO(SCHED_MODES),
                                                              cfg_scale_min=FLOAT(0.0, 0.0, 100.0, 0.5), sched_val=FLOAT(1.0, 0.0, 100.0, 0.01),
                                                              separate_feature_channels=COMBO(['enable', 'disable']), scaling_startpoint=COMBO(['MEAN', 'ZERO']),
                                                              variability_measure=COMBO(['AD', 'STD']), interpolate_phi=FLOAT(1.0, 0.0, 1.0, 0.01)),
         [('MODEL', 'MODEL')], display='DynamicThresholdingFull'),
])

if __name__ == '__main__':
    out = pathlib.Path(__file__).parent / 'data' / 'third_party_schemas.json'
    out.write_text(json.dumps(NODES, indent=1) + '\n', encoding='utf-8')
    print(f'{len(NODES)} third-party schemas -> {out}')
