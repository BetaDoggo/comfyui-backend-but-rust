"""Writes crates/comfy-server/tests/fixtures/stock_schemas.json: stock ComfyUI /object_info entries for
every node comfy-server implements (the `stock_schema_parity` test compares against it).

Usage: python tools/snapshot_stock_schemas.py <real object_info.json> <ours object_info.json>
"""
import json, sys

real = json.load(open(sys.argv[1], encoding='utf-8'))
ours = json.load(open(sys.argv[2], encoding='utf-8'))
snap = {n: real[n] for n in sorted(ours) if n in real}
path = 'crates/comfy-server/tests/fixtures/stock_schemas.json'
json.dump(snap, open(path, 'w', encoding='utf-8'), indent=1)
print(f'{len(snap)} stock schemas -> {path}')
