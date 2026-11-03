"""Renders front, side and top views of a vertex-coloured GLB (as written by Pixal3D here or ComfyUI's SaveGLB) to
a PNG, by splatting its vertices and face centres with a z-buffer and two-sided face shading. Needs only numpy and Pillow.

usage: python glb_preview.py mesh.glb out.png [--size 512] [--flat]
"""
import argparse
import json
import struct

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("glb")
ap.add_argument("out")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--flat", action="store_true", help="ignore normals: colour only")
args = ap.parse_args()

data = open(args.glb, "rb").read()
jlen = struct.unpack_from("<I", data, 12)[0]
gltf = json.loads(data[20:20 + jlen])
bin_off = 20 + jlen + 8
prim = gltf["meshes"][0]["primitives"][0]


def accessor(i):
    a = gltf["accessors"][i]
    v = gltf["bufferViews"][a["bufferView"]]
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    start = bin_off + v.get("byteOffset", 0) + a.get("byteOffset", 0)
    arr = np.frombuffer(data, dtype=dt, count=a["count"] * n, offset=start).reshape(a["count"], n)
    if a.get("normalized"):
        arr = arr.astype(np.float32) / np.iinfo(dt).max
    return arr


pos = accessor(prim["attributes"]["POSITION"]).astype(np.float32)
col = accessor(prim["attributes"]["COLOR_0"])[:, :3] if "COLOR_0" in prim["attributes"] else np.full_like(pos, 0.6)
col = np.clip(col, 0, 1) ** (1 / 2.2)
# splat the triangles' centroids too, so a simplified mesh has no holes, shaded two-sided by the face normal
faces = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
vn = np.zeros_like(pos)
for k in range(3):
    np.add.at(vn, faces[:, k], np.abs(fn))  # only used through |n . view| below
nrm = None if args.flat else np.concatenate([vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20), fn])
pos = np.concatenate([pos, pos[faces].mean(1)])
col = np.concatenate([col, col[faces].mean(1)])
S = args.size
center = (pos.max(0) + pos.min(0)) / 2
scale = 0.9 * S / (pos.max(0) - pos.min(0)).max()


def view(right, up, fwd):
    p = (pos - center) @ np.stack([right, up, fwd]).T
    x = (p[:, 0] * scale + S / 2).astype(np.int32)
    y = (S / 2 - p[:, 1] * scale).astype(np.int32)
    z = p[:, 2]
    ok = (x >= 0) & (x < S) & (y >= 0) & (y < S)
    x, y, z = x[ok], y[ok], z[ok]
    c = col[ok]
    if nrm is not None:
        shade = 0.35 + 0.65 * np.abs(nrm[ok] @ fwd)
        c = c * shade[:, None]
    order = np.argsort(-z)  # far to near, nearest written last
    img = np.full((S, S, 3), 0.15, np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            img[np.minimum(y[order] + dy, S - 1), np.minimum(x[order] + dx, S - 1)] = c[order]
    return img


views = [view(np.array([1, 0, 0.]), np.array([0, 1, 0.]), np.array([0, 0, -1.])),   # front, looking along -z
         view(np.array([0, 0, -1.]), np.array([0, 1, 0.]), np.array([-1, 0, 0.])),  # side
         view(np.array([1, 0, 0.]), np.array([0, 0, -1.]), np.array([0, -1, 0.]))]  # top
Image.fromarray((np.concatenate(views, 1) * 255).astype(np.uint8)).save(args.out)
print(f"{len(pos)} vertices -> {args.out}")
