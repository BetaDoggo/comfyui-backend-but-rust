import json, struct, sys, collections
p = sys.argv[1]
with open(p, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
meta = h.pop('__metadata__', None)
print("META:", json.dumps(meta)[:3000] if meta else None)
full = len(sys.argv) > 2
cnt = collections.Counter()
for k, v in h.items():
    cnt[v['dtype']] += 1
    if full or ('.0.' in k or 'blocks.' not in k and 'layers.' not in k):
        print(k, v['dtype'], v['shape'])
print(cnt, len(h))
