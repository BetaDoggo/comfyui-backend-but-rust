"""Probes a live SwarmUI (pointed at comfy-server) to learn what every T2I parameter does to the
generated ComfyUI workflow, for each supported model.

For each parameter: generate a baseline workflow (base params + the parameter's companions, e.g.
an init image for Mask Blur), then the same plus the parameter, via /API/ComfyGetGeneratedWorkflow,
and diff them: node classes added/removed and node inputs whose values changed.

Usage (from the repo root, SwarmUI on :7801 with a session id in scratch/session, comfy-server
started with --advertise-todo-nodes so feature-gated paths are reachable):
    python tools/probe_swarm_params.py <our object_info.json (without todo nodes)> tools/data/param_probe.json [--run]

--run additionally executes each case whose workflow only uses implemented nodes through
/API/GenerateText2Image (once, on the first model it changes) to confirm it completes. With
--generic <full object_info.json> (from a mock that claims the backlog through custom_node), cases
that need backlog nodes listed there are run too.
"""
import base64, io, json, struct, sys, urllib.request, wave
from PIL import Image, ImageDraw

SWARM = 'http://127.0.0.1:7801/API/'
SESSION = open('scratch/session').read().strip()
implemented = set(json.load(open(sys.argv[1], encoding='utf-8')))
OUT = sys.argv[2]
RUN = '--run' in sys.argv
# --generic <object_info.json>: a full dump (mock claiming the backlog through custom_node); cases
# whose missing nodes all appear there are run too.
generic = set(json.load(open(sys.argv[sys.argv.index('--generic') + 1], encoding='utf-8'))) if '--generic' in sys.argv else set()


def call(ep, p, timeout=120):
    req = urllib.request.Request(SWARM + ep, json.dumps(dict(p, session_id=SESSION)).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def png(img):
    b = io.BytesIO()
    img.save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def wav(seconds=1.0, rate=16000):
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b'\0\0' * int(seconds * rate))
    return 'data:audio/wav;base64,' + base64.b64encode(b.getvalue()).decode()


IMG = png(Image.new('RGB', (1024, 1024), (90, 140, 200)))
_m = Image.new('RGB', (1024, 1024), (0, 0, 0))
ImageDraw.Draw(_m).rectangle((300, 300, 700, 700), fill=(255, 255, 255))
MASK = png(_m)
AUDIO = wav()

MODELS = {
    'krea2': ('krea2_test.safetensors', {}),
    'anima': ('anima_test.safetensors', {}),
    'minimax_h3': ('minimax_h3_test.safetensors', {'width': 448, 'height': 256, 'text2videoframes': 22}),
    'moss': ('moss_image_test.safetensors', {}),
    'qwen21': ('qwen_image_2.1_test.safetensors', {}),
}
BASE = dict(images=1, width=1024, height=1024, steps=6, cfgscale=4, seed=1, prompt='a cat', negativeprompt='')

INIT = {'initimage': IMG, 'initimagecreativity': 0.6}
MASKED = dict(INIT, maskimage=MASK)
REFINE = {'refinercontrolpercentage': 0.3, 'refinermethod': 'PostApply'}
# ControlNet groups only apply once their Strength is set (it is the group toggle).
CN = {'controlnetimageinput': IMG, 'controlnetmodel': 'fake_controlnet.safetensors', 'controlnetstrength': 1}
CN2 = {'controlnettwoimageinput': IMG, 'controlnettwomodel': 'fake_controlnet.safetensors', 'controlnettwostrength': 1}
CN3 = {'controlnetthreeimageinput': IMG, 'controlnetthreemodel': 'fake_controlnet.safetensors', 'controlnetthreestrength': 1}
VIDEO = {'videomodel': 'minimax_h3_test.safetensors', 'videoframes': 22, 'videoresolution': 'Image Aspect, Model Res'}
SEGMENT = {'prompt': 'a cat <segment:face> a happy face'}
REGION = {'prompt': 'a cat <region:0,0,0.5,0.5> a dog'}
OBJECT = {'prompt': 'a cat <object:0,0,0.5,0.5> a dog'}
PROMPT_IMG = {'promptimages': [IMG]}
EXTEND = {'prompt': 'a cat <extend:20> walking', 'videoextendmodel': 'minimax_h3_test.safetensors'}

# Companion params a parameter needs before it can have any effect (included in its baseline too).
COMPANIONS = {
    'initimagecreativity': {'initimage': IMG}, 'initimageresettonorm': INIT, 'initimagenoise': INIT, 'maskimage': INIT,
    'unsamplerprompt': INIT, 'batchsize': INIT,
    'maskshrinkgrow': MASKED, 'maskblur': MASKED, 'maskgrow': MASKED, 'initimagerecompositemask': MASKED,
    'useinpaintingencode': MASKED, 'maskbehavior': MASKED, 'colorcorrectionbehavior': MASKED,
    'refinercontrolpercentage': {'refinermethod': 'PostApply'}, 'refinermethod': {'refinercontrolpercentage': 0.3}, 'refinerupscale': REFINE,
    'refinerdotiling': REFINE, 'refinermodel': REFINE, 'refinervae': REFINE, 'refinersteps': REFINE,
    'refinercfgscale': REFINE, 'refinersampler': REFINE, 'refinerscheduler': REFINE, 'refinerhypertile': REFINE,
    'refinerupscalemethod': dict(REFINE, refinerupscale=1.5),
    'controlnetstrength': {k: v for k, v in CN.items() if k != 'controlnetstrength'},
    'controlnetimageinput': {'controlnetmodel': 'fake_controlnet.safetensors', 'controlnetstrength': 1},
    'controlnetmodel': {'controlnetimageinput': IMG, 'controlnetstrength': 1}, 'controlnetstart': CN, 'controlnetend': CN, 'controlnetpreprocessor': CN, 'controlnetuniontype': CN,
    'controlnetpreviewonly': CN, 'controlnettwostrength': {k: v for k, v in CN2.items() if k != 'controlnettwostrength'}, 'controlnettwostart': CN2, 'controlnettwoend': CN2,
    'controlnettwopreprocessor': CN2, 'controlnettwouniontype': CN2, 'controlnetthreestrength': {k: v for k, v in CN3.items() if k != 'controlnetthreestrength'}, 'controlnetthreestart': CN3,
    'controlnetthreeend': CN3, 'controlnetthreepreprocessor': CN3, 'controlnetthreeuniontype': CN3,
    'videoswapmodel': VIDEO, 'videoswappercent': dict(VIDEO, videoswapmodel='minimax_h3_test.safetensors'), 'videoframes': VIDEO,
    'videosteps': VIDEO, 'videocfg': VIDEO, 'videoresolution': VIDEO, 'videovideocreativity': VIDEO, 'videoendimage': VIDEO,
    'videomincfg': VIDEO, 'videomotionbucket': VIDEO, 'videoaugmentationlevel': VIDEO,
    'videoextendframeoverlap': EXTEND, 'videoextendswapmodel': EXTEND, 'videoextendswappercent': dict(EXTEND, videoextendswapmodel='minimax_h3_test.safetensors'),
    'videoextendformat': EXTEND,
    'savesegmentmask': SEGMENT, 'segmentmaskblur': SEGMENT, 'segmentmaskgrow': SEGMENT, 'segmentmaskoversize': SEGMENT,
    'segmentthresholdmax': SEGMENT, 'segmentsortorder': SEGMENT, 'segmentapplyafter': SEGMENT, 'segmenttargetresolution': SEGMENT,
    'segmentmodel': SEGMENT, 'segmentsteps': SEGMENT, 'segmentcfgscale': SEGMENT,
    'globalregionfactor': REGION, 'maskcompositeunthresholded': REGION, 'gligenmodel': REGION, 'debugregionalprompting': REGION,
    'regionalobjectcleanupfactor': OBJECT, 'regionalobjectinpaintingmodel': OBJECT,
    'revisionstrength': PROMPT_IMG, 'revisionzeroprompt': PROMPT_IMG, 'usereferenceonly': PROMPT_IMG, 'smartimagepromptresizing': PROMPT_IMG,
    'clipvisionmodel': PROMPT_IMG, 'useipadapter': PROMPT_IMG, 'enablereferencelatents': PROMPT_IMG, 'textencodedimage': PROMPT_IMG,
    'usestylemodel': PROMPT_IMG, 'ipadapterweight': PROMPT_IMG, 'ipadapterstart': PROMPT_IMG, 'ipadapterend': PROMPT_IMG,
    'ipadapterweighttype': PROMPT_IMG,
    'stylemodelmergestrength': dict(PROMPT_IMG, usestylemodel='fake_redux.safetensors'),
    'stylemodelmultiplystrength': dict(PROMPT_IMG, usestylemodel='fake_redux.safetensors'),
    'stylemodelapplystart': dict(PROMPT_IMG, usestylemodel='fake_redux.safetensors'),
    'freeuversion': {'freeuapplyto': 'Both'}, 'freeublockone': {'freeuapplyto': 'Both'}, 'freeublocktwo': {'freeuapplyto': 'Both'},
    'freeuskipone': {'freeuapplyto': 'Both'}, 'freeuskiptwo': {'freeuapplyto': 'Both'},
    'easycachethreshold': {'easycachemode': 'all'}, 'easycachestart': {'easycachemode': 'all'}, 'easycacheend': {'easycachemode': 'all'},
    'teacachethreshold': {'teacachemode': 'all'}, 'teacachestart': {'teacachemode': 'all'},
    'selfattentionguidancesigmablur': {'selfattentionguidancescale': 0.5},
    'normalizedattentionguidancealpha': {'normalizedattentionguidancescale': 5}, 'normalizedattentionguidancetau': {'normalizedattentionguidancescale': 5},
    'loraweights': {'loras': ['__LORA__']}, 'loratencweights': {'loras': ['__LORA__']}, 'lorasectionconfinement': {'loras': ['__LORA__']},
    'negativemodelincludeloras': {'negativemodel': '__MODEL__', 'loras': ['__LORA__']},
    'variationseedstrength': {'variationseed': 5},
    'vaetileoverlap': {'vaetilesize': 512}, 'vaetemporaltileoverlap': {'vaetemporaltilesize': 16},
    'samnegativepoints': {'sampositivepoints': '[[500,500]]', 'initimage': IMG},
    'sampositivepoints': {'initimage': IMG}, 'sambbox': {'initimage': IMG},
    'seedvrpredownscale': {'seedvrmodel': '__MODEL__'}, 'seedvrupscale': {'seedvrmodel': '__MODEL__'}, 'seedvrupscalemethod': {'seedvrmodel': '__MODEL__'},
    'seedvrcolorcorrectionbehavior': {'seedvrmodel': '__MODEL__'}, 'seedvrsplitlatent': {'seedvrmodel': '__MODEL__'},
    'seedvrtemporalvideooverlap': {'seedvrmodel': '__MODEL__'},
    'dtthresholdpercentile': {'dtmimicscale': 7}, 'dtcfgscalemode': {'dtmimicscale': 7}, 'dtcfgscaleminimum': {'dtmimicscale': 7},
    'dtmimicscalemode': {'dtmimicscale': 7}, 'dtmimicscaleminimum': {'dtmimicscale': 7}, 'dtschedulervalue': {'dtmimicscale': 7},
    'dtseparatefeaturechannels': {'dtmimicscale': 7}, 'dtscalingstartpoint': {'dtmimicscale': 7}, 'dtvariabilitymeasure': {'dtmimicscale': 7},
    'dtinterpolatephi': {'dtmimicscale': 7},
    'videoframeinterpolationmethod': {'videoframeinterpolationmultiplier': 2},
    'audiosilentsuffixduration': {}, 'audiosilentprefixduration': {},
    'trimvideoendframes': {}, 'trimvideostartframes': {},
    'yolomodelinternal': {'prompt': 'a cat <segment:yolo-face_yolov8m.pt> a happy face'},
}

# Explicit values (otherwise picked automatically from ListT2IParams metadata).
VALUES = {
    'initimage': IMG, 'maskimage': MASK, 'controlnetimageinput': IMG, 'controlnettwoimageinput': IMG, 'controlnetthreeimageinput': IMG,
    'videoendimage': IMG, 'promptimages': [IMG], 'promptaudios': [AUDIO], 'videoaudioinput': AUDIO,
    'controlnetmodel': 'fake_controlnet.safetensors', 'controlnettwomodel': 'fake_controlnet.safetensors', 'controlnetthreemodel': 'fake_controlnet.safetensors',
    'clipvisionmodel': 'fake_clip_vision.safetensors', 'videomodel': 'minimax_h3_test.safetensors', 'videoswapmodel': 'minimax_h3_test.safetensors',
    'videoextendmodel': 'minimax_h3_test.safetensors', 'videoextendswapmodel': 'minimax_h3_test.safetensors',
    'refinermodel': '__OTHER_MODEL__', 'segmentmodel': '__OTHER_MODEL__', 'negativemodel': '__OTHER_MODEL__', 'regionalobjectinpaintingmodel': '__OTHER_MODEL__',
    'pixeldecodermodel': '__OTHER_MODEL__', 'seedvrmodel': '__MODEL__', 'refinervae': '__VAE__', 'vae': '__VAE__',
    'loras': ['__LORA__'], 'loraweights': ['0.5'], 'loratencweights': ['0.3'], 'lorasectionconfinement': ['1'],
    'unsamplerprompt': 'a dog', 'textaudiostyle': 'rock', 'refinerupscale': 1.5, 'refinercontrolpercentage': 0.3, 'batchsize': 2,
    'globalregionfactor': 0.3, 'segmenttargetresolution': '512x768', 'maskshrinkgrow': 16, 'maskblur': 8, 'maskgrow': 16, 'refinerupscalemethod': 'model-4x_fake.safetensors',
    'usestylemodel': 'fake_redux.safetensors', 'gligenmodel': 'fake_gligen.safetensors', 'freeuapplyto': 'Both', 'easycachemode': 'all',
    'teacachemode': 'all', 'videoframeinterpolationmultiplier': 2, 'sampositivepoints': '[[500,500]]', 'samnegativepoints': '[[100,100]]',
    'sambbox': '[100,100,600,600]', 'seamlesstileable': 'true', 'clipstopatlayer': -2, 'removebackground': True, 'variationseed': 5,
    'variationseedstrength': 0.5, 'endstepsearly': 0.5, 'trimvideostartframes': 2, 'trimvideoendframes': 2, 'videoboomerang': True,
    'audiosilentprefixduration': 0.5, 'audiosilentsuffixduration': 0.5, 'sigmashift': 2.5, 'samplersigmamin': 0.05, 'samplersigmamax': 10,
    'samplerrho': 5, 'conditioningmultiplier': 1.5, 'negativeconditioningmultiplier': 0.5, 'dtmimicscale': 7, 'seedvrupscale': 2,
    'zeronegative': True, 'ippcfg': 1.5, 'vaetilesize': 512, 'vaetileoverlap': 64, 'vaetemporaltilesize': 16, 'vaetemporaltileoverlap': 4,
    'videoformat': 'webm', 'imageformat': 'JPG', 'colordepth': '16bit', 'webhooks': 'Manual', 'personalnote': 'probe', 'audioformat': 'wav',
    'refinerhypertile': 256, 'selfattentionguidancescale': 0.5, 'perturbedattentionguidancescale': 3, 'rescalecfgmultiplier': 0.7,
    'renormcfg': 1, 'normalizedattentionguidancescale': 5, 'wildcardseed': 3,
}
SKIP = {'prompt', 'model', 'images', 'seed', 'width', 'height', 'gridgenpromptreplace', 'gridgenpromptadd', 'gridgenpresets',
        'autowebuisampler', 'comfyuicustomworkflow', 'promptvideos', 'exactbackendid', 'internalbackendtype', 'aspectratio', 'sidelength',
        'rawresolution', 'altresolutionheightmultiplier'}
LORAS = {'krea2': 'krea2_style', 'anima': 'anima_style', 'minimax_h3': 'h3_motion', 'moss': 'moss_style', 'qwen21': 'qwen21_style'}
VAES = {'krea2': 'QwenImage/qwen_image_vae.safetensors', 'anima': 'QwenImage/qwen_image_vae.safetensors', 'minimax_h3': 'MiniMaxH3/minimax_h3_video_vae_int8_convrot.safetensors',
        'moss': 'Flux/flux2-vae.safetensors', 'qwen21': 'QwenImage/qwen_image_2.1_vae_bf16.safetensors'}
OTHER = {'krea2': 'anima_test.safetensors', 'anima': 'krea2_test.safetensors', 'minimax_h3': 'minimax_h3_test.safetensors',
         'moss': 'qwen_image_2.1_test.safetensors', 'qwen21': 'moss_image_test.safetensors'}

# Prompt syntax (not parameters) that changes the workflow.
SYNTAX = {
    '<region:x,y,w,h>': REGION, '<object:x,y,w,h>': OBJECT, '<segment:text>': SEGMENT,
    '<segment:yolo-model>': {'prompt': 'a cat <segment:yolo-face_yolov8m.pt> a happy face'},
    '<clear:text>': {'prompt': 'a cat <clear:background>'}, '<extend:frames>': EXTEND,
    '<refiner>': {'prompt': 'a cat <refiner> a detailed cat', 'refinercontrolpercentage': 0.3},
    '<base>': {'prompt': 'a cat <base> a simple cat'}, '<video>': {'prompt': 'a cat <video> the cat walks', 'videomodel': 'minimax_h3_test.safetensors'},
    '<videoswap>': {'prompt': 'a cat <videoswap> the cat runs', **VIDEO, 'videoswapmodel': 'minimax_h3_test.safetensors'},
    '<pixeldecoder>': {'prompt': 'a cat <pixeldecoder> sharp'}, '<break>': {'prompt': 'a cat <break> a hat'},
    '<lora:name:weight>': {'prompt': 'a cat <lora:__LORA__:0.6>'}, '<embed:name>': {'prompt': 'a cat <embed:none>'},
    '<alternate:a,b>': {'prompt': 'a <alternate:cat, dog>'}, '<fromto[0.5]:a,b>': {'prompt': 'a <fromto[0.5]:cat, dog>'},
    '<weight[1.5]:text>': {'prompt': 'a <weight[1.5]:cat>'}, '<trigger>': {'prompt': 'a cat <trigger>'},
    '<comment:text>': {'prompt': 'a cat <comment:note>'},
}


def auto_value(p):
    d = str(p['default'])
    t = p['type']
    if t == 'boolean':
        return d.lower() != 'true'
    if t == 'dropdown':
        for v in p['values'] or []:
            v = v.split('///')[0]
            if v != d and v.lower() not in ('none', 'disabled', '(none)'):
                return v
        return None
    if t in ('integer', 'decimal'):
        for e in p.get('examples') or []:
            if str(e) != d:
                try:
                    return float(e) if t == 'decimal' else int(float(e))
                except ValueError:
                    pass
        lo, hi = p['min'], p['max']
        cand = (float(d) if d else 0) + (p['step'] or 1)
        if hi > lo and cand > hi:
            cand = lo
        return cand if t == 'decimal' else int(cand)
    if t == 'text':
        return 'probe text'
    return None


def subst(v, short):
    if isinstance(v, list):
        return [subst(x, short) for x in v]
    if v == '__LORA__':
        return LORAS[short]
    if v == '__MODEL__':
        return MODELS[short][0]
    if v == '__OTHER_MODEL__':
        return OTHER[short]
    if v == '__VAE__':
        return VAES[short]
    if isinstance(v, str) and '__LORA__' in v:
        return v.replace('__LORA__', LORAS[short])
    return v


def params_for(short, extra):
    model, mextra = MODELS[short]
    p = dict(BASE, model=model, **mextra)
    for k, v in extra.items():
        p[k] = subst(v, short)
    return p


_cache = {}


def workflow(params):
    key = json.dumps(params, sort_keys=True)
    if key not in _cache:
        try:
            d = call('ComfyGetGeneratedWorkflow', params)
        except Exception as ex:  # noqa: BLE001
            d = {'error': f'{type(ex).__name__}: {ex}'}
        if 'workflow' in d:
            wf = d['workflow'] if isinstance(d['workflow'], dict) else json.loads(d['workflow'])
            _cache[key] = (wf, None)
        else:
            _cache[key] = (None, d.get('error') or json.dumps(d)[:300])
    return _cache[key]


def diff(a, b):
    ca = {}
    for n in a.values():
        ca[n['class_type']] = ca.get(n['class_type'], 0) + 1
    cb = {}
    for n in b.values():
        cb[n['class_type']] = cb.get(n['class_type'], 0) + 1
    new = sorted(k for k in cb if k not in ca)
    removed = sorted(k for k in ca if k not in cb)
    changed = set()
    for nid, n in b.items():
        o = a.get(nid)
        if o is None or o['class_type'] != n['class_type']:
            continue
        for k, v in n['inputs'].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                continue
            if o['inputs'].get(k) != v:
                changed.add(f"{n['class_type']}.{k}")
    return new, removed, sorted(changed)


def probe(label, param_id, extra, companions):
    case = {'param': label, 'id': param_id, 'models': {}}
    ran = False
    for short in MODELS:
        base_p = params_for(short, companions)
        test_p = params_for(short, dict(companions, **extra))
        bwf, berr = workflow(base_p)
        twf, terr = workflow(test_p)
        r = {}
        if terr:
            r['error'] = terr
        elif berr:
            r['error'] = f'baseline failed: {berr}'
        else:
            new, removed, changed = diff(bwf, twf)
            r = {'new_nodes': new, 'removed_nodes': removed, 'changed_inputs': changed,
                 'missing': sorted({n['class_type'] for n in twf.values() if n['class_type'] not in implemented})}
            if RUN and not ran and not (set(r['missing']) - generic) and (new or removed or changed):
                try:
                    g = call('GenerateText2Image', test_p, timeout=300)
                    r['run'] = 'ok' if g.get('images') else (g.get('error') or json.dumps(g))[:300]
                except Exception as ex:  # noqa: BLE001
                    r['run'] = f'{type(ex).__name__}: {ex}'
                ran = True
        case['models'][short] = r
    case['new_nodes'] = sorted({n for r in case['models'].values() for n in r.get('new_nodes', [])})
    return case


def main():
    if RUN:
        # SwarmUI's first generation after comfy-server restarts can fail on a stale websocket; absorb it.
        try:
            call('GenerateText2Image', params_for('krea2', {}), timeout=300)
        except Exception:  # noqa: BLE001
            pass
    listing = call('ListT2IParams', {})
    params = listing['list']
    cases = []
    for p in params:
        pid = p['id']
        if pid in SKIP:
            continue
        v = VALUES.get(pid, None)
        if v is None:
            v = auto_value(p)
        if v is None:
            cases.append({'param': p['name'], 'id': pid, 'models': {}, 'new_nodes': [], 'skipped': 'no usable non-default value'})
            continue
        companions = COMPANIONS.get(pid, {})
        cases.append(probe(p['name'], pid, {pid: v}, companions))
        print(f"{p['name']:45} new={cases[-1]['new_nodes']}", flush=True)
    for label, extra in SYNTAX.items():
        prompt_extra = {k: v for k, v in extra.items()}
        cases.append(probe(f'prompt syntax {label}', None, prompt_extra, {}))
        print(f"{label:45} new={cases[-1]['new_nodes']}", flush=True)
    meta = {p['id']: {k: p[k] for k in ('name', 'type', 'default', 'feature_flag', 'group', 'advanced', 'visible')} for p in params}
    groups = {g['id']: {'name': g['name'], 'priority': g['priority']} for g in listing['groups']}
    json.dump({'params': meta, 'groups': groups, 'cases': cases}, open(OUT, 'w', encoding='utf-8'), indent=1)


main()
