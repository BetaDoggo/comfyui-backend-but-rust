"""Compares comfy-server's /object_info against a stock ComfyUI's, for every node both define.

Usage: python tools/check_schemas.py <real object_info.json> <ours object_info.json>
Checks input names (required/optional, in order), types, combo options for non-model lists,
default/min/max/step, outputs and output names, category. Exit code 1 if anything differs.
Model-file combos and a few documented, intentional differences are skipped.
"""
import json, sys

real = json.load(open(sys.argv[1], encoding='utf-8'))
ours = json.load(open(sys.argv[2], encoding='utf-8'))

# Combos filled from model folders / engine capabilities differ by design.
DYNAMIC = {('ModelAttentionBackend', 'attention')}
# Documented intentional deviations: (node, input) -> reason.
ALLOWED = {
    ('VAEDecodeTiled', 'temporal_size'): 'max relaxed so SwarmUI\'s 9999 for MiniMax H3 validates',
}


def spec(s):
    t = s[0]
    o = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
    if t == 'COMBO':
        t = o.get('options')
    keep = {k: o[k] for k in ('default', 'min', 'max', 'step') if k in o}
    if s[0] == 'COMFY_AUTOGROW_V3':
        tp = o['template']
        keep['template'] = (tp.get('prefix'), tp.get('names'), tp.get('min'), tp.get('max'), {k: spec(v) for k, v in tp['input'].get('required', {}).items()})
    if s[0] == 'COMFY_DYNAMICCOMBO_V3':
        keep['options'] = [(x['key'], {k: spec(v) for k, v in x['inputs'].get('required', {}).items()}) for x in o['options']]
    return t, keep


def is_model_list(t):
    return isinstance(t, list) and any(str(x).endswith(('.safetensors', '.pth', '.pt', '.ckpt', '.gguf', '.bin', '.sft')) for x in t)


bad = 0
for n in sorted(ours):
    if n not in real:
        continue
    probs = []
    for sec in ('required', 'optional'):
        ri, oi = real[n]['input'].get(sec, {}), ours[n]['input'].get(sec, {})
        if list(ri) != list(oi):
            probs.append(f'{sec} inputs {list(ri)} vs {list(oi)}')
        for k in ri:
            if k not in oi or (n, k) in ALLOWED or (n, k) in DYNAMIC:
                continue
            (rt, rk), (ot, ok) = spec(ri[k]), spec(oi[k])
            if isinstance(rt, list) and isinstance(ot, list) and (is_model_list(ot) or rt == [] or ot == []):
                rt = ot = 'MODEL_LIST'
            if (rt, rk) != (ot, ok):
                probs.append(f'{k}: stock {(rt, rk)} vs ours {(ot, ok)}')
    if real[n]['output'] != ours[n]['output'] or real[n]['output_name'] != ours[n]['output_name']:
        probs.append(f"outputs stock {real[n]['output']}{real[n]['output_name']} vs ours {ours[n]['output']}{ours[n]['output_name']}")
    if real[n].get('output_node') != ours[n].get('output_node'):
        probs.append('output_node')
    if real[n]['category'] != ours[n]['category']:
        probs.append(f"category stock {real[n]['category']!r} vs ours {ours[n]['category']!r}")
    if probs:
        bad += 1
        print(n, probs)
print(f'{len([n for n in ours if n in real])} shared nodes checked, {bad} differ; {len([n for n in ours if n not in real])} not in stock: {[n for n in ours if n not in real]}')
sys.exit(1 if bad else 0)
