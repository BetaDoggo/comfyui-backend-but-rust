"""Generates docs/COMPATIBILITY.md from tools/data/param_probe.json and crates/comfy-server/src/todo_nodes.json.

Usage (from the repo root): python tools/gen_compat_doc.py
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from todo_catalog import AREAS

probe = json.load(open('tools/data/param_probe.json', encoding='utf-8'))
todo = json.load(open('crates/comfy-server/src/todo_nodes.json', encoding='utf-8'))['nodes']
MODELS = ['krea2', 'anima', 'minimax_h3', 'moss', 'qwen21']
MODEL_NAMES = {'krea2': 'Krea 2', 'anima': 'Anima', 'minimax_h3': 'MiniMax H3', 'moss': 'MOSS', 'qwen21': 'Qwen 2.1'}
AREA_ORDER = ['other_arch', 'third_party']
NEEDS = {'server': 'server', 'patch': 'server + engine flag', 'engine': 'engine op', 'arch': 'new arch'}

# Parameters that produced no workflow change for any supported model, and why.
SWARM_ONLY = 'SwarmUI-side only: never reaches the backend'
OTHER_ARCH = 'Only applies to other model architectures'
THIRD = 'Needs a third-party node pack (advertised only when the engine claims it)'
NOT_PROBED = 'Not probed (needs a matching model file); see note'
NO_EFFECT = {
    'revisionstrength': (OTHER_ARCH, 'SDXL ReVision; Swarm refuses it for these models.'),
    'revisionzeroprompt': (OTHER_ARCH, 'SDXL ReVision.'),
    'textaudioduration': (OTHER_ARCH, 'Text-to-audio models (ACE-Step, YuE, MiniMax Music).'),
    'textaudiostyle': (OTHER_ARCH, 'Text-to-audio models.'), 'textaudiobpm': (OTHER_ARCH, 'ACE-Step.'),
    'textaudiotimesignature': (OTHER_ARCH, 'ACE-Step.'), 'textaudiolanguage': (OTHER_ARCH, 'ACE-Step.'),
    'textaudiokeyscale': (OTHER_ARCH, 'ACE-Step.'), 'audioformat': (OTHER_ARCH, 'Audio-only outputs (text-to-audio models).'),
    'cascadelatentcompression': (OTHER_ARCH, 'Stable Cascade.'), 'sdtextencs': (OTHER_ARCH, 'SD3.'),
    'videomincfg': (OTHER_ARCH, 'Stable Video Diffusion (VideoLinearCFGGuidance).'),
    'videomotionbucket': (OTHER_ARCH, 'Stable Video Diffusion.'), 'videoaugmentationlevel': (OTHER_ARCH, 'Stable Video Diffusion.'),
    'ippcfg': (OTHER_ARCH, 'InstructPix2Pix (DualCFGGuider).'), 'overridepredictiontype': (OTHER_ARCH, 'SD-family checkpoints (ModelSamplingDiscrete).'),
    'shiftedlatentaverageinit': (OTHER_ARCH, 'SD-family offset latents (SwarmOffsetEmptyLatentImage).'),
    'pixeldecodermodel': (OTHER_ARCH, 'Needs a PiD pixel-decoder model (PiDConditioning).'),
    'cliplmodel': (NOT_PROBED, 'Text-encoder overrides are only used by models whose loader takes that encoder; Krea 2 / Anima / MOSS / Qwen use the Qwen Model slot.'),
    'clipgmodel': (NOT_PROBED, 'SDXL / SD3 encoders.'), 'txxlmodel': (NOT_PROBED, 'Flux / SD3 / video models.'),
    'llavamodel': (NOT_PROBED, 'Hunyuan Video.'), 'llamamodel': (NOT_PROBED, 'HiDream.'),
    'qwenmodel': (NOT_PROBED, 'Overrides the CLIPLoader file for Qwen-encoder models; rides the existing CLIPLoader node.'),
    'mistralmodel': (NOT_PROBED, 'Mistral-encoder models.'), 'gemmamodel': (NOT_PROBED, 'Gemma-encoder models.'),
    'gptossmodel': (NOT_PROBED, 'GPT-OSS-encoder models.'),
    'clipvisionmodel': ('Needs other params', 'Only used together with Use Style Model / unCLIP / IP-Adapter.'),
    'negativemodel': ('Needs other params', 'Only takes effect with a compatible second model; no change observed with the test models.'),
    'negativemodelincludeloras': ('Needs other params', 'Only with Negative Model.'),
    'automaticvae': (SWARM_ONLY, 'Swarm picks the VAE before building the workflow (result shows up as VAELoader.vae_name).'),
    'outputintermediateimages': ('Needs other params', 'Adds extra SwarmSaveImageWS outputs only when a refiner/segment stage runs.'),
    'donotsave': (SWARM_ONLY, 'Swarm discards the result itself.'), 'donotsaveintermediates': (SWARM_ONLY, ''),
    'noloadmodels': (SWARM_ONLY, ''), 'nointernalspecialhandling': (SWARM_ONLY, ''), 'continueaftererrors': (SWARM_ONLY, ''),
    'forwardrawbackenddata': (SWARM_ONLY, ''), 'forwardswarmdata': (SWARM_ONLY, ''), 'webhooks': (SWARM_ONLY, ''),
    'wildcardseed': (SWARM_ONLY, 'Resolved into the prompt text.'), 'wildcardseedbehavior': (SWARM_ONLY, ''), 'noseedincrement': (SWARM_ONLY, ''),
    'personalnote': (SWARM_ONLY, ''), 'imageformat': (SWARM_ONLY, 'Swarm re-encodes the PNG it receives.'), 'overrideoutpathformat': (SWARM_ONLY, ''),
    'placeholderparamgroupstarred': (SWARM_ONLY, 'UI placeholder.'), 'placeholderparamgroupuserone': (SWARM_ONLY, 'UI placeholder.'),
    'placeholderparamgroupusertwo': (SWARM_ONLY, 'UI placeholder.'), 'placeholderparamgroupuserthree': (SWARM_ONLY, 'UI placeholder.'),
    'refinercontrolpercentage': ('Needs other params', 'Enables the refiner together with Refiner Method; see Refiner Method / Refiner Upscale Method rows.'),
    'refinerupscale': ('Needs other params', 'Takes effect with Refiner Method; see Refiner Upscale Method.'),
    'controlnetimageinput': ('Needs other params', 'ControlNet image, model and strength only act together; see ControlNet Strength.'),
    'controlnetmodel': ('Needs other params', 'See ControlNet Strength.'), 'controlnettwoimageinput': ('Needs other params', 'See ControlNet Two Strength.'),
    'controlnettwomodel': ('Needs other params', 'See ControlNet Two Strength.'), 'controlnetthreeimageinput': ('Needs other params', 'See ControlNet Three Strength.'),
    'controlnetthreemodel': ('Needs other params', 'See ControlNet Three Strength.'),
    'controlnetpreviewonly': ('Needs other params', 'Only with a ControlNet image and a preprocessor (comfyui_controlnet_aux, generic route). Verified live with the MiDaS preprocessor.'),
    'controlnetpreprocessor': (THIRD, 'Preprocessors come from comfyui_controlnet_aux (category image/preprocessors).'),
    'controlnettwopreprocessor': (THIRD, ''), 'controlnetthreepreprocessor': (THIRD, ''),
    'videoextendmodel': ('Needs other params', 'Used with the <extend:N> prompt syntax; see that row.'),
    'videoextendswappercent': ('Needs other params', 'With Video Extend Swap Model.'), 'videoextendformat': ('Needs other params', 'With <extend:N>; no change observed.'),
    'regionalobjectcleanupfactor': ('Needs other params', 'Only with <object:> plus an inpainting pass.'),
    'regionalobjectinpaintingmodel': ('Needs other params', 'Only with <object:>.'),
    'maskcompositeunthresholded': ('Needs other params', 'Only when region masks overlap.'),
    'segmentsortorder': ('Needs other params', 'Only with multiple matches for one <segment:>.'),
    'segmentapplyafter': ('Needs other params', 'Moves segment stages relative to the refiner.'),
    'vaetileoverlap': ('Needs other params', 'With VAE Tile Size (VAEDecodeTiled.overlap).'),
    'vaetemporaltileoverlap': ('Needs other params', 'With VAE Temporal Tile Size.'),
    'easycacheend': ('Needs other params', 'With EasyCache Mode.'),
    'modelattentionbackend': ('Needs other params', 'Options come from the engine\'s attention backends (ModelAttentionBackend schema).'),
    'seedvrupscalemethod': ('Needs other params', 'With SeedVR Model.'), 'seedvrtemporalvideooverlap': ('Needs other params', 'SeedVR on video.'),
    'useipadapter': (THIRD, 'ComfyUI_IPAdapter_plus.'), 'ipadapterweight': ('Needs other params', 'Only with Use IP-Adapter. Verified live together (generic route).'),
    'ipadapterstart': ('Needs other params', 'Only with Use IP-Adapter.'), 'ipadapterend': ('Needs other params', 'Only with Use IP-Adapter.'),
    'ipadapterweighttype': ('Needs other params', 'Only with Use IP-Adapter.'), 'enableaitemplate': (THIRD, 'AITemplate.'), 'teacachemode': (OTHER_ARCH, 'ComfyUI-TeaCache; SwarmUI only adds it for model families TeaCache supports. Verified live on Wan 2.1 (generic route).'),
    'teacachethreshold': ('Needs other params', 'Only with TeaCache Mode. Verified live on Wan 2.1 (generic route).'), 'teacachestart': ('Needs other params', 'Only with TeaCache Mode.'),
    'nunchakucachethreshold': ('Needs other params', 'Only with a Nunchaku (SVDQuant) model file loaded through the Nunchaku nodes.'),
    'setclipdevice': (THIRD, 'OverrideCLIPDevice.'), 'videoframeinterpolationmultiplier': ('Needs other params', 'Only with Video Frame Interpolation Method on a video model. Verified live with RIFE and FILM (generic route).'),
    'yolomodelinternal': ('Needs other params', 'Internal; used through <segment:yolo-...>, which works natively (Engine::segment).'),
}
for k in ['dtmimicscale', 'dtthresholdpercentile', 'dtcfgscalemode', 'dtcfgscaleminimum', 'dtmimicscalemode', 'dtmimicscaleminimum',
          'dtschedulervalue', 'dtseparatefeaturechannels', 'dtscalingstartpoint', 'dtvariabilitymeasure', 'dtinterpolatephi']:
    NO_EFFECT[k] = (THIRD, 'sd-dynamic-thresholding (DynamicThresholdingFull).')
SYNTAX_NOTES = {
    '<refiner>': ('Needs other params', 'Swaps the prompt for the refiner stage only.'),
    '<pixeldecoder>': (OTHER_ARCH, 'Pixel decoder models.'),
    '<trigger>': (SWARM_ONLY, 'Expanded to the model\'s trigger phrase by Swarm.'),
    '<comment:text>': (SWARM_ONLY, 'Stripped by Swarm.'),
}
# Syntax that changes the workflow but couldn't be run live, and why.
RAN_NOTES = {
    '<embed:name>': 'Not run live: the probe has no embedding file, so SwarmUI refuses the request itself. Parsing is covered by the engine-api prompt tests.',
}
NOTES = {
    'justloadmodel': 'Swarm-internal flag that drops every output node, so ComfyUI answers "Prompt has no outputs" too. Swarm\'s real model-preload path (SwarmJustLoadTheModelPlease) works.',
    'model': 'Base model selection.',
    'promptembed': '',
}


def esc(s):
    return s.replace('|', '\\|').replace('<', '&lt;').replace('>', '&gt;')


def code_list(xs):
    return ', '.join(f'`{x}`' for x in xs)


def summarize(case):
    ms = case['models']
    affected = [m for m in MODELS if ms.get(m) and (ms[m].get('new_nodes') or ms[m].get('removed_nodes') or ms[m].get('changed_inputs'))]
    # Everything the parameter's workflow needs that is still in the backlog (including nodes its companions add).
    blocked = sorted({n for m in affected for n in ms[m].get('missing', []) if n in todo})
    introduced = sorted({n for m in affected for n in ms[m].get('new_nodes', []) if n in todo})
    reaches = sorted({x for m in affected for x in ms[m].get('changed_inputs', [])} |
                     {n for m in affected for n in ms[m].get('new_nodes', []) if n not in todo})
    ran = [m for m in MODELS if ms.get(m, {}).get('run') == 'ok']
    return affected, blocked, introduced, reaches, ran


def status_of(blocked):
    if not blocked:
        return 'Works'
    return 'Generic' if all(todo[n].get('object_info') for n in blocked) else 'Blocked'


rows_by_group = {}
node_params = {}  # backlog node -> params whose workflow newly adds it
syntax_rows = []
counts = {'Works': 0, 'Generic': 0, 'Blocked': 0}
for case in probe['cases']:
    pid = case['id']
    affected, blocked, introduced, reaches, ran = summarize(case)
    if pid is None:
        label = case['param'].replace('prompt syntax ', '')
        if affected:
            status = status_of(blocked)
            note = RAN_NOTES.get(label, '')
        else:
            status, note = SYNTAX_NOTES.get(label, ('No change observed', ''))
        syntax_rows.append((label, status, affected, blocked, reaches, ran, note))
        for n in introduced:
            node_params.setdefault(n, []).append(label)
        continue
    meta = probe['params'][pid]
    if affected:
        status = status_of(blocked)
        note = NOTES.get(pid, '')
    else:
        status, note = NO_EFFECT.get(pid, ('No change observed', NOTES.get(pid, '')))
        if pid in NOTES:
            note = NOTES[pid]
    counts[status] = counts.get(status, 0) + 1
    group = meta['group'] or 'Core'
    rows_by_group.setdefault(group, []).append((meta['name'], pid, status, affected, blocked, reaches, ran, note))
    for n in introduced:
        node_params.setdefault(n, []).append(meta['name'])


out = []
w = out.append
total_params = sum(len(rows) for rows in rows_by_group.values())
w('# SwarmUI compatibility')
w('')
w('<!-- Generated by tools/gen_compat_doc.py from tools/data/param_probe.json and crates/comfy-server/src/todo_nodes.json. Do not edit by hand. -->')
w('')
w('This covers every parameter SwarmUI exposes, every prompt-syntax feature, and every ComfyUI node class SwarmUI\'s workflow generator can emit, all checked against this backend.')
w('')
w('It was measured, not guessed. `tools/probe_swarm_params.py` drove a live SwarmUI that was pointed at comfy-server. For each parameter it generated the workflow with and without that parameter, for each of the five supported models, and diffed the two. Each case that changed the workflow was then run end to end through SwarmUI on the mock engine (the **Ran** column).')
w('')
param_counts = {k: sum(1 for rows in rows_by_group.values() for r in rows if r[2] == k) for k in ('Works', 'Generic', 'Blocked')}
w(f'**{total_params} parameters probed:** {param_counts["Works"]} work natively, {param_counts["Generic"]} work through the generic `custom_node` route, {param_counts["Blocked"]} are blocked, and the rest never reach the backend for these models (handled inside SwarmUI, or they only act together with another parameter).')
w('')
w('## Native and generic support')
w('')
w('Every node class SwarmUI can emit is either implemented natively or has a schema here, so every workflow SwarmUI builds validates and runs:')
w('')
w('- **Native**: the server implements the node, lowering it to typed engine requests, patches and metadata (see [INPUT_MAPPING.md](INPUT_MAPPING.md)). This covers everything SwarmUI does with the five target models.')
w(f'- **Generic** ({len(todo)} node classes, listed below): other model families\' stock ComfyUI nodes and third-party node packs. The schemas live in [`crates/comfy-server/src/todo_nodes.json`](../crates/comfy-server/src/todo_nodes.json). Stock schemas come from a real ComfyUI dump; third-party ones are written in `tools/third_party_schemas.py`. At startup the server asks `Engine::supports_custom_node(class)` for each. Claimed classes are registered with that schema and listed in `/object_info`, which makes SwarmUI enable the matching features. They execute through `Engine::custom_node`, with their resolved inputs and with model-file combos resolved to files. The mock engine claims all of them and returns typed placeholder outputs; run it with `--mock-native-only` to turn that off.')
w('- An unclaimed class fails validation before anything runs. The `missing_node_type` error names the node, its area and the parameters behind it, and SwarmUI shows it to the user. `--advertise-todo-nodes` lists unclaimed schemas in `/object_info` anyway, for inspecting workflows; running them still fails validation.')
w('- To make a node native: register it under `crates/comfy-server/src/nodes/` (a native schema always wins), remove it from `tools/todo_catalog.py`, then regenerate (see the end of this file).')
w('')
w('**Needs** column (what native support would take):')
w('- *server*: pure data work the server can do itself (images, masks, ints).')
w('- *server + engine flag*: the server records a new patch or metadata field (for example on `ModelPatch` or `CondMeta`), and the engine has to honor it while sampling.')
w('- *engine op*: needs a new `Engine` method.')
w('- *new arch*: belongs to supporting another model family.')
w('')
w('## Generic-route nodes by area')
w('')
for area in AREA_ORDER:
    title, params = AREAS[area]
    nodes = sorted((k, v) for k, v in todo.items() if v['area'] == area)
    w(f'### {title} ({len(nodes)})')
    w('')
    w(f'SwarmUI features: {esc(params)}.')
    w('')
    w('| Node | Needs | What it does | Emitted for (observed live) |')
    w('|---|---|---|---|')
    for k, v in nodes:
        trig = node_params.get(k) or v['triggers']
        trig = sorted(set(trig))
        w(f"| `{k}` | {NEEDS[v['needs']]} | {esc(v['note'])} | {esc(', '.join(trig)) if trig else '(not observed with these models; ' + esc(', '.join(v['swarm_source'][:2])) + ')'} |")
    w('')

w('## Parameter matrix')
w('')
w('Status values:')
w('- **Works**: the workflow only uses natively implemented nodes.')
w('- **Generic**: the workflow also needs generic-route nodes (listed in **Via custom_node**). It works when the engine claims them; it ran on the mock.')
w('- **Blocked**: the workflow needs a node with no schema at all.')
w('- Anything else explains why the parameter did not change the workflow for these models.')
w('')
w('**Reaches** lists the node inputs that changed and the implemented nodes that were added. These are the fields the engine has to honor; see [INPUT_MAPPING.md](INPUT_MAPPING.md) for what each one becomes in the engine API.')
w('')
w('Model keys: ' + ', '.join(f'`{k}` = {v}' for k, v in MODEL_NAMES.items()) + '.')
w('')
GROUPS = probe.get('groups', {})
for group, rows in sorted(rows_by_group.items(), key=lambda kv: GROUPS.get(kv[0], {}).get('priority', -1000)):
    w(f"### {GROUPS.get(group, {}).get('name', 'Core (ungrouped)')}")
    w('')
    w('| Parameter | Status | Models | Via custom_node | Reaches | Ran | Notes |')
    w('|---|---|---|---|---|---|---|')
    for name, pid, status, affected, blocked, reaches, ran, note in rows:
        models = 'all' if len(affected) == len(MODELS) else ', '.join(affected)
        w(f"| {esc(name)} (`{pid}`) | {status} | {models} | {code_list(blocked)} | {code_list(reaches[:8])}{' ...' if len(reaches) > 8 else ''} | {'yes' if ran else ''} | {esc(note)} |")
    w('')

w('## Prompt syntax')
w('')
w('| Syntax | Status | Models | Via custom_node | Reaches | Ran | Notes |')
w('|---|---|---|---|---|---|---|')
for label, status, affected, blocked, reaches, ran, note in syntax_rows:
    models = 'all' if len(affected) == len(MODELS) else ', '.join(affected)
    w(f"| `{label}` | {status} | {models} | {code_list(blocked)} | {code_list(reaches[:8])}{' ...' if len(reaches) > 8 else ''} | {'yes' if ran else ''} | {esc(note)} |")
w('')
w('Wildcards, `<random>`, `<repeat>`, `<preset>`, `<var>`/`<setvar>`, `<macro>`/`<setmacro>` and `<param>` are resolved inside SwarmUI before the workflow is built. They only ever change prompt text or parameter values.')
w('')
w('## Regenerating')
w('')
w('Run these from the repo root, with `reference/SwarmUI` and `reference/ComfyUI` cloned.')
w('')
w('1. Start a stock ComfyUI on CPU with SwarmUI\'s extra nodes, and save its `/object_info`:')
w('   `python main.py --cpu --port 8199 --extra-model-paths-config <yaml with custom_nodes: .../ComfyUIBackend/ExtraNodes>`')
w('2. Save comfy-server\'s own `/object_info` twice: started with `--mock-native-only` (ours.json), and without it (full.json, which adds the generic-route nodes).')
w('3. Run `python tools/swarm_inventory.py real.json ours.json inventory.json`, which scans SwarmUI\'s generator source.')
w('4. Run `python tools/third_party_schemas.py`, then `python tools/gen_todo_nodes.py real.json inventory.json` to write `todo_nodes.json`. It fails if SwarmUI can emit a node that is neither implemented nor classified in `tools/todo_catalog.py`.')
w('5. Restart comfy-server (full mock), start SwarmUI with the fake models from `scratch/make_fake_models.py`, then run `python tools/probe_swarm_params.py ours.json tools/data/param_probe.json --run --generic full.json`.')
w('6. Run `python tools/gen_todo_nodes.py real.json inventory.json tools/data/param_probe.json`, then `python tools/gen_compat_doc.py`.')
pathlib.Path('docs/COMPATIBILITY.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('wrote docs/COMPATIBILITY.md', counts)
