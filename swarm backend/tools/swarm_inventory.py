"""Inventories every SwarmUI T2I parameter and every ComfyUI node class SwarmUI's workflow generator can emit,
then maps parameters to the nodes created in the same generator step/method (plus one level of helper calls).

Usage: python tools/swarm_inventory.py <real ComfyUI object_info.json> <our object_info.json> <out.json>
Run from the repo root with SwarmUI cloned at reference/SwarmUI.
"""
import json, re, pathlib, sys

ROOT = pathlib.Path('reference/SwarmUI/src')
PARAM_FILES = ['Text2Image/T2IParamTypes.cs', 'BuiltinExtensions/ComfyUIBackend/ComfyUIBackendExtension.cs',
               'BuiltinExtensions/DynamicThresholding/DynamicThresholdingExtension.cs']
GEN_FILES = ['BuiltinExtensions/ComfyUIBackend/WorkflowGenerator.cs', 'BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorSteps.cs',
             'BuiltinExtensions/ComfyUIBackend/WorkflowGeneratorModelSupport.cs', 'BuiltinExtensions/ComfyUIBackend/WGNodeData.cs',
             'BuiltinExtensions/DynamicThresholding/DynamicThresholdingExtension.cs', 'BuiltinExtensions/ComfyUIBackend/ComfyUIBackendExtension.cs']

real = json.load(open(sys.argv[1], encoding='utf-8'))
ours = json.load(open(sys.argv[2], encoding='utf-8'))
THIRD_PARTY = ['AITemplateLoader', 'IPAdapter', 'IPAdapterApply', 'IPAdapterModelLoader', 'IPAdapterUnifiedLoader', 'IPAdapterAdvanced',
               'IPAdapterEncoder', 'IPAdapterCombineEmbeds', 'IPAdapterEmbeds', 'IPAdapterFaceID', 'IPAdapterInsightFaceLoader',
               'MiDaS-DepthMapPreprocessor', 'RIFE VFI', 'FILM VFI', 'GIMMVFI_interpolate', 'DownloadAndLoadGIMMVFIModel', 'Sam2Segmentation',
               'DownloadAndLoadSAM2Model', 'Sam2AutoSegmentation', 'PixArtCheckpointLoader', 'PixArtResolutionSelect', 'PixArtT5TextEncode',
               'SanaCheckpointLoader', 'SanaResolutionSelect', 'SanaTextEncode', 'GemmaLoader', 'CheckpointLoaderNF4', 'UnetLoaderGGUF',
               'NunchakuFluxDiTLoader', 'NunchakuTextEncoderLoader', 'NunchakuFluxLoraLoader', 'NunchakuQwenImageDiTLoader', 'TensorRTLoader',
               'TeaCache', 'TeaCacheForVidGen', 'TeaCacheForImgGen', 'OverrideCLIPDevice', 'SwarmYoloDetection', 'SwarmRemBg',
               'SwarmSaveAnimationWS', 'DynamicThresholdingFull']

# ---------------------------------------------------------------- params
params = []
reg = re.compile(r'(\w+)\s*=\s*(?:T2IParamTypes\.)?Register<([^>]+(?:>[^>(]*)?)>\(new\(\s*\$?"([^"]+)"')
for f in PARAM_FILES:
    text = (ROOT / f).read_text(encoding='utf-8')
    for m in reg.finditer(text):
        # registration args run until the matching '));' — grab a generous window
        window = text[m.end():m.end() + 4000]
        end = window.find('\n            ));')
        if end < 0:
            end = window.find('));')
        args = window[:end]
        def opt(key):
            mm = re.search(key + r':\s*("([^"]*)"|[\w.]+)', args)
            return (mm.group(2) if mm and mm.group(2) is not None else (mm.group(1) if mm else None))
        desc = re.match(r',\s*\$?"((?:[^"\\]|\\.)*)"', args)
        params.append({
            'var': m.group(1), 'type': m.group(2), 'name': m.group(3), 'file': f,
            'line': text[:m.start()].count('\n') + 1,
            'group': opt('Group'), 'feature': opt('FeatureFlag'),
            'advanced': opt('IsAdvanced') == 'true', 'hidden': opt('VisibleNormally') == 'false',
            'unused': opt('IntentionalUnused') == 'true',
            'desc': (desc.group(1).split('\\n')[0] if desc else ''),
        })


# ---------------------------------------------------------------- generator blocks
known_nodes = set(real) | set(THIRD_PARTY) | set(ours)
lit = re.compile(r'"([A-Za-z0-9_+\- ]{3,})"')
method_sig = re.compile(r'^    (?:public|private|internal|protected)\s+(?:static\s+)?[\w<>\[\],. ?()]+?\s+(\w+)\s*\(', re.M)
step_sig = re.compile(r'^\s*(?:AddStep|AddModelGenStep)\(', re.M)

blocks = []  # (file, startline, name, text)
for f in GEN_FILES:
    text = (ROOT / f).read_text(encoding='utf-8')
    cuts = sorted([(m.start(), m.group(1)) for m in method_sig.finditer(text)] +
                  [(m.start(), None) for m in step_sig.finditer(text)])
    cuts.append((len(text), None))
    for (a, name), (b, _) in zip(cuts, cuts[1:]):
        line = text[:a].count('\n') + 1
        blocks.append({'file': f, 'line': line, 'name': name or f'step@{line}', 'text': text[a:b]})

param_vars = {p['var'] for p in params}
methods = {b['name']: b for b in blocks if not b['name'].startswith('step@')}
for b in blocks:
    t = b['text']
    b['nodes'] = sorted({s for s in lit.findall(t) if s in known_nodes})
    b['params'] = sorted({v for v in param_vars if re.search(r'(?:T2IParamTypes|ComfyUIBackendExtension|DynamicThresholdingExtension|\bg)\.' + v + r'\b|\b' + v + r'\b', t)})
    b['calls'] = sorted({c for c in re.findall(r'\b(\w+)\(', t) if c in methods and c != b['name']})

def reach(b, depth=2, seen=None):
    seen = seen if seen is not None else set()
    out = set(b['nodes'])
    if depth == 0:
        return out
    for c in b['calls']:
        if c not in seen:
            seen.add(c)
            out |= reach(methods[c], depth - 1, seen)
    return out

param_nodes = {}
for b in blocks:
    rn = reach(b)
    for v in b['params']:
        param_nodes.setdefault(v, {}).setdefault(f"{b['file'].split('/')[-1]}:{b['line']} {b['name']}", sorted(rn))

all_emitted = set()
for b in blocks:
    all_emitted |= set(b['nodes'])

json.dump({'params': params, 'param_nodes': param_nodes, 'emitted_nodes': sorted(all_emitted),
           'not_implemented': sorted(n for n in all_emitted if n not in ours),
           'blocks': [{k: v for k, v in b.items() if k != 'text'} for b in blocks]},
          open(sys.argv[3], 'w', encoding='utf-8'), indent=1)
print(len(params), 'params;', len(all_emitted), 'emitted node types;',
      len([n for n in all_emitted if n not in ours]), 'not implemented here')
