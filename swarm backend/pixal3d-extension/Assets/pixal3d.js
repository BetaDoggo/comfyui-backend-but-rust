// Pixal3D tab: image in, a GLB mesh (Pixal3D) or a PLY gaussian splat (Idol) out, shown in a small WebGL2 viewer
// (no third-party code).

/** Reads a binary glTF's first primitive: positions, linear vertex colours (if any) and indices. */
function pixal3dParseGlb(buffer) {
    let view = new DataView(buffer);
    if (view.getUint32(0, true) != 0x46546c67) {
        throw new Error('not a GLB file');
    }
    let jsonLen = view.getUint32(12, true);
    let gltf = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen)));
    let binStart = 20 + jsonLen + 8;
    let prim = gltf.meshes[0].primitives[0];
    let types = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
    let sizes = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
    function accessor(i) {
        let a = gltf.accessors[i];
        let bv = gltf.bufferViews[a.bufferView];
        let T = types[a.componentType];
        let n = sizes[a.type];
        let offset = binStart + (bv.byteOffset || 0) + (a.byteOffset || 0);
        let stride = bv.byteStride || 0;
        let out = new Float32Array(a.count * n);
        if (stride && stride != n * T.BYTES_PER_ELEMENT) {
            for (let k = 0; k < a.count; k++) {
                let row = new T(buffer, offset + k * stride, n);
                for (let c = 0; c < n; c++) {
                    out[k * n + c] = row[c];
                }
            }
        }
        else {
            out.set(new T(buffer.slice(offset, offset + a.count * n * T.BYTES_PER_ELEMENT)));
        }
        if (a.normalized && T != Float32Array) {
            let max = { 5121: 255, 5123: 65535, 5120: 127, 5122: 32767 }[a.componentType];
            out.forEach((v, k) => out[k] = v / max);
        }
        return { data: out, size: n, count: a.count, min: a.min, max: a.max };
    }
    let pos = accessor(prim.attributes.POSITION);
    let colors = prim.attributes.COLOR_0 !== undefined ? accessor(prim.attributes.COLOR_0) : null;
    let indices;
    if (prim.indices !== undefined) {
        let a = gltf.accessors[prim.indices];
        let bv = gltf.bufferViews[a.bufferView];
        let T = types[a.componentType];
        let offset = binStart + (bv.byteOffset || 0) + (a.byteOffset || 0);
        indices = new Uint32Array(new T(buffer.slice(offset, offset + a.count * T.BYTES_PER_ELEMENT)));
    }
    else {
        indices = new Uint32Array(pos.count).map((_, i) => i);
    }
    return { positions: pos.data, colors: colors, indices: indices, vertexCount: pos.count };
}

/** Reads a binary little-endian 3DGS PLY: centres, colours (SH DC), opacities, and each gaussian's 3D covariance.
 * Splat files are Y down, as ComfyUI writes them and most splat viewers expect, so they turn 180 degrees about X
 * into this viewer's Y up frame. */
function pixal3dParsePly(buffer) {
    let bytes = new Uint8Array(buffer);
    let head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
    let end = head.indexOf('end_header\n');
    if (!head.startsWith('ply') || end < 0) {
        throw new Error('not a PLY file');
    }
    if (!head.includes('format binary_little_endian')) {
        throw new Error('only binary little-endian PLY files are supported');
    }
    let count = 0, props = [], inVertex = false;
    for (let line of head.substring(0, end).split('\n')) {
        let w = line.trim().split(/\s+/);
        if (w[0] == 'element') {
            inVertex = w[1] == 'vertex';
            if (inVertex) {
                count = parseInt(w[2]);
            }
        }
        else if (w[0] == 'property' && inVertex) {
            if (w[1] != 'float' && w[1] != 'float32') {
                throw new Error(`unsupported PLY property type ${w[1]}`);
            }
            props.push(w[2]);
        }
    }
    let at = name => {
        let i = props.indexOf(name);
        if (i < 0) {
            throw new Error(`the PLY has no ${name}: not a gaussian splat`);
        }
        return i;
    };
    let [ix, iy, iz, ir, ig, ib, io] = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'].map(at);
    let isc = [0, 1, 2].map(k => at(`scale_${k}`)), iro = [0, 1, 2, 3].map(k => at(`rot_${k}`));
    let stride = props.length;
    let data = new Float32Array(buffer.slice(end + 'end_header\n'.length, end + 'end_header\n'.length + count * stride * 4));
    let centers = new Float32Array(count * 3), colors = new Float32Array(count * 4), cov = new Float32Array(count * 6);
    const C0 = 0.28209479177387814;
    for (let i = 0; i < count; i++) {
        let r = data.subarray(i * stride, (i + 1) * stride);
        centers.set([r[ix], -r[iy], -r[iz]], i * 3);
        colors.set([0.5 + C0 * r[ir], 0.5 + C0 * r[ig], 0.5 + C0 * r[ib], 1 / (1 + Math.exp(-r[io]))].map(v => Math.max(0, Math.min(1, v))), i * 4);
        let [w, x, y, z] = iro.map(k => r[k]);
        let n = Math.hypot(w, x, y, z) || 1;
        [w, x, y, z] = [w / n, x / n, y / n, z / n];
        let R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
        let sc = isc.map(k => Math.exp(r[k]));
        // M = R S, covariance M M^T
        let M = [0, 1, 2].flatMap(row => [0, 1, 2].map(c => R[row * 3 + c] * sc[c]));
        let dot = (a, b) => M[a * 3] * M[b * 3] + M[a * 3 + 1] * M[b * 3 + 1] + M[a * 3 + 2] * M[b * 3 + 2];
        // diag(1, -1, -1) on both sides
        cov.set([dot(0, 0), -dot(0, 1), -dot(0, 2), dot(1, 1), dot(1, 2), dot(2, 2)], i * 6);
    }
    return { count: count, centers: centers, colors: colors, cov: cov };
}

/** Orbit viewer of one mesh or one gaussian splat. Faces are flat-shaded from screen derivatives, so their winding
 * doesn't matter; splats are sorted back to front each frame and blended as screen-space gaussians. */
class Pixal3DViewer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2', { antialias: true });
        this.yaw = 0.6;
        this.pitch = 0.25;
        this.distance = 2.2;
        this.target = [0, 0, 0];
        this.count = 0;
        if (!this.gl) {
            return;
        }
        let gl = this.gl;
        let vs = `#version 300 es
            in vec3 aPos; in vec3 aColor;
            uniform mat4 uMvp; uniform mat4 uView;
            out vec3 vView; out vec3 vColor;
            void main() { vView = (uView * vec4(aPos, 1.0)).xyz; vColor = aColor; gl_Position = uMvp * vec4(aPos, 1.0); }`;
        let fs = `#version 300 es
            precision highp float;
            in vec3 vView; in vec3 vColor; out vec4 outColor;
            void main() {
                vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
                float light = 0.3 + 0.7 * abs(n.z);
                outColor = vec4(pow(clamp(vColor, 0.0, 1.0), vec3(1.0 / 2.2)) * light, 1.0);
            }`;
        let compile = (type, src) => {
            let s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                throw new Error(gl.getShaderInfoLog(s));
            }
            return s;
        };
        let link = (v, f) => {
            let prog = gl.createProgram();
            gl.attachShader(prog, compile(gl.VERTEX_SHADER, v));
            gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, f));
            gl.linkProgram(prog);
            if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
                throw new Error(gl.getProgramInfoLog(prog));
            }
            return prog;
        };
        this.program = link(vs, fs);
        this.vao = gl.createVertexArray();
        this.buffers = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
        // splats: per-gaussian data in float textures, read by the sorted index of each instance
        let svs = `#version 300 es
            precision highp float; precision highp int;
            in vec2 aCorner; in uint aIndex;
            uniform highp sampler2D uCenter, uCovA, uCovB, uColor;
            uniform mat4 uView, uProj; uniform vec2 uViewport; uniform int uWidth;
            out vec4 vColor; out vec2 vPos;
            void main() {
                ivec2 t = ivec2(int(aIndex) % uWidth, int(aIndex) / uWidth);
                vec4 c = texelFetch(uCenter, t, 0), ca = texelFetch(uCovA, t, 0), cb = texelFetch(uCovB, t, 0);
                vColor = texelFetch(uColor, t, 0);
                vec4 cam = uView * vec4(c.xyz, 1.0);
                vec4 clip = uProj * cam;
                float lim = 1.2 * clip.w;
                if (cam.z > -1e-4 || clip.x < -lim || clip.x > lim || clip.y < -lim || clip.y > lim) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }
                mat3 cov3 = mat3(ca.x, ca.y, ca.z, ca.y, ca.w, cb.x, ca.z, cb.x, cb.y);
                vec2 focal = vec2(uProj[0][0], uProj[1][1]) * uViewport * 0.5;
                float z = -cam.z;
                mat3 J = mat3(focal.x / z, 0.0, 0.0, 0.0, focal.y / z, 0.0, focal.x * cam.x / (z * z), focal.y * cam.y / (z * z), 0.0);
                mat3 W = mat3(uView);
                mat3 T = J * W;
                mat3 c2 = T * cov3 * transpose(T);
                float a = c2[0][0] + 0.3, b = c2[0][1], d = c2[1][1] + 0.3;
                float mid = 0.5 * (a + d), rad = length(vec2(0.5 * (a - d), b));
                float l1 = mid + rad, l2 = mid - rad;
                if (l2 <= 0.0) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }
                vec2 v1 = normalize(vec2(b, l1 - a) + vec2(1e-9, 0.0));
                vec2 major = min(sqrt(2.0 * l1), 1024.0) * v1, minor = min(sqrt(2.0 * l2), 1024.0) * vec2(v1.y, -v1.x);
                vPos = aCorner;
                vec2 ndc = clip.xy / clip.w + (aCorner.x * major + aCorner.y * minor) * 2.0 / uViewport;
                gl_Position = vec4(ndc, 0.0, 1.0);
            }`;
        let sfs = `#version 300 es
            precision highp float;
            in vec4 vColor; in vec2 vPos; out vec4 outColor;
            void main() {
                float A = -dot(vPos, vPos);
                if (A < -4.0) discard;
                float B = exp(A) * vColor.a;
                outColor = vec4(B * vColor.rgb, B);
            }`;
        this.splatProgram = link(svs, sfs);
        this.splatVao = gl.createVertexArray();
        this.splatCorners = gl.createBuffer();
        this.splatIndex = gl.createBuffer();
        this.splatTextures = [gl.createTexture(), gl.createTexture(), gl.createTexture(), gl.createTexture()];
        gl.bindVertexArray(this.splatVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.splatCorners);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
        let lc = gl.getAttribLocation(this.splatProgram, 'aCorner');
        gl.enableVertexAttribArray(lc);
        gl.vertexAttribPointer(lc, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.splatIndex);
        let li = gl.getAttribLocation(this.splatProgram, 'aIndex');
        gl.enableVertexAttribArray(li);
        gl.vertexAttribIPointer(li, 1, gl.UNSIGNED_INT, 0, 0);
        gl.vertexAttribDivisor(li, 1);
        gl.bindVertexArray(null);
        this.splat = null;
        this.bindControls();
        new ResizeObserver(() => this.draw()).observe(canvas);
    }

    bindControls() {
        let canvas = this.canvas;
        let dragging = null, last = null;
        canvas.addEventListener('contextmenu', e => e.preventDefault());
        canvas.addEventListener('pointerdown', e => {
            dragging = e.button == 2 || e.shiftKey ? 'pan' : 'orbit';
            last = [e.clientX, e.clientY];
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = 'grabbing';
        });
        canvas.addEventListener('pointerup', e => {
            dragging = null;
            canvas.style.cursor = 'grab';
        });
        canvas.addEventListener('pointermove', e => {
            if (!dragging) {
                return;
            }
            let dx = e.clientX - last[0], dy = e.clientY - last[1];
            last = [e.clientX, e.clientY];
            if (dragging == 'orbit') {
                this.yaw -= dx * 0.008;
                this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.008));
            }
            else {
                let s = this.distance * 0.0015;
                let [right, up] = this.basis();
                for (let k = 0; k < 3; k++) {
                    this.target[k] += (-dx * right[k] + dy * up[k]) * s;
                }
            }
            this.draw();
        });
        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            this.distance *= Math.exp(e.deltaY * 0.001);
            this.draw();
        }, { passive: false });
    }

    basis() {
        let cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
        let back = [sy * cp, sp, cy * cp];
        let right = [cy, 0, -sy];
        let up = [back[1] * right[2] - back[2] * right[1], back[2] * right[0] - back[0] * right[2], back[0] * right[1] - back[1] * right[0]];
        return [right, up, back];
    }

    /** Points the camera at the box around `p` (xyz triples). */
    frame(p) {
        let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < p.length; i += 3) {
            for (let k = 0; k < 3; k++) {
                lo[k] = Math.min(lo[k], p[i + k]);
                hi[k] = Math.max(hi[k], p[i + k]);
            }
        }
        this.target = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
        let extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
        this.distance = extent * 1.9;
        this.near = extent * 0.01;
    }

    setMesh(mesh) {
        let gl = this.gl;
        if (!gl) {
            return;
        }
        let p = mesh.positions;
        this.frame(p);
        this.splat = null;
        let colors = new Float32Array(mesh.vertexCount * 3).fill(0.55);
        if (mesh.colors) {
            for (let i = 0; i < mesh.vertexCount; i++) {
                for (let k = 0; k < 3; k++) {
                    colors[i * 3 + k] = mesh.colors.data[i * mesh.colors.size + k];
                }
            }
        }
        gl.bindVertexArray(this.vao);
        let attrib = (buf, name, data) => {
            gl.bindBuffer(gl.ARRAY_BUFFER, buf);
            gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
            let loc = gl.getAttribLocation(this.program, name);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
        };
        attrib(this.buffers[0], 'aPos', p);
        attrib(this.buffers[1], 'aColor', colors);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers[2]);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
        gl.bindVertexArray(null);
        this.count = mesh.indices.length;
        this.draw();
    }

    setSplat(splat) {
        let gl = this.gl;
        if (!gl) {
            return;
        }
        this.frame(splat.centers);
        let n = splat.count;
        let width = Math.min(n, 2048), height = Math.ceil(n / width);
        let pack = (fill) => {
            let a = new Float32Array(width * height * 4);
            for (let i = 0; i < n; i++) {
                fill(a, i);
            }
            return a;
        };
        let c = splat.centers, v = splat.cov, col = splat.colors;
        let data = [
            pack((a, i) => a.set([c[i * 3], c[i * 3 + 1], c[i * 3 + 2], 1], i * 4)),
            pack((a, i) => a.set([v[i * 6], v[i * 6 + 1], v[i * 6 + 2], v[i * 6 + 3]], i * 4)),
            pack((a, i) => a.set([v[i * 6 + 4], v[i * 6 + 5], 0, 0], i * 4)),
            pack((a, i) => a.set([col[i * 4], col[i * 4 + 1], col[i * 4 + 2], col[i * 4 + 3]], i * 4))
        ];
        data.forEach((d, k) => {
            gl.bindTexture(gl.TEXTURE_2D, this.splatTextures[k]);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, d);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        });
        this.splat = { count: n, width: width, centers: c, order: new Uint32Array(n), depth: new Int32Array(n), sortedFor: null };
        this.count = 0;
        this.draw();
    }

    /** Orders the splats back to front along the view direction `back` (a counting sort over 16-bit depths). */
    sortSplats(back) {
        let sp = this.splat;
        let key = back.map(v => v.toFixed(3)).join(',');
        if (sp.sortedFor == key) {
            return;
        }
        sp.sortedFor = key;
        let c = sp.centers, n = sp.count, depth = sp.depth;
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < n; i++) {
            // larger is nearer the camera
            let d = c[i * 3] * back[0] + c[i * 3 + 1] * back[1] + c[i * 3 + 2] * back[2];
            depth[i] = d * 4096 | 0;
            lo = Math.min(lo, depth[i]);
            hi = Math.max(hi, depth[i]);
        }
        let scale = 65535 / Math.max(1, hi - lo);
        let counts = new Uint32Array(65536);
        for (let i = 0; i < n; i++) {
            depth[i] = ((depth[i] - lo) * scale) | 0;
            counts[depth[i]]++;
        }
        for (let i = 1; i < 65536; i++) {
            counts[i] += counts[i - 1];
        }
        // ascending depth key: the farthest first
        for (let i = n - 1; i >= 0; i--) {
            sp.order[--counts[depth[i]]] = i;
        }
        let gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.splatIndex);
        gl.bufferData(gl.ARRAY_BUFFER, sp.order, gl.DYNAMIC_DRAW);
    }

    draw() {
        let gl = this.gl;
        if (!gl) {
            return;
        }
        let dpr = window.devicePixelRatio || 1;
        let w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
        if (this.canvas.width != w || this.canvas.height != h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
        gl.viewport(0, 0, w, h);
        gl.clearColor(0.13, 0.13, 0.14, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        if (!this.count && !this.splat) {
            return;
        }
        let [right, up, back] = this.basis();
        let eye = this.target.map((t, k) => t + back[k] * this.distance);
        // view = rotation from (right, up, back), translation -eye
        let dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let view = [right[0], up[0], back[0], 0, right[1], up[1], back[1], 0, right[2], up[2], back[2], 0, -dot(right, eye), -dot(up, eye), -dot(back, eye), 1];
        let near = Math.max(this.near || 0.001, this.distance * 0.01), far = this.distance * 10;
        let f = 1 / Math.tan(0.35);
        let proj = [f * h / w, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
        let mvp = new Float32Array(16);
        for (let c = 0; c < 4; c++) {
            for (let r = 0; r < 4; r++) {
                let s = 0;
                for (let k = 0; k < 4; k++) {
                    s += proj[k * 4 + r] * view[c * 4 + k];
                }
                mvp[c * 4 + r] = s;
            }
        }
        if (this.splat) {
            this.sortSplats(back);
            let sp = this.splat, prog = this.splatProgram;
            gl.disable(gl.DEPTH_TEST);
            gl.disable(gl.CULL_FACE);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.useProgram(prog);
            gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uView'), false, new Float32Array(view));
            gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uProj'), false, new Float32Array(proj));
            gl.uniform2f(gl.getUniformLocation(prog, 'uViewport'), w, h);
            gl.uniform1i(gl.getUniformLocation(prog, 'uWidth'), sp.width);
            ['uCenter', 'uCovA', 'uCovB', 'uColor'].forEach((name, k) => {
                gl.activeTexture(gl.TEXTURE0 + k);
                gl.bindTexture(gl.TEXTURE_2D, this.splatTextures[k]);
                gl.uniform1i(gl.getUniformLocation(prog, name), k);
            });
            gl.bindVertexArray(this.splatVao);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, sp.count);
            gl.bindVertexArray(null);
            gl.disable(gl.BLEND);
            return;
        }
        gl.enable(gl.DEPTH_TEST);
        gl.disable(gl.CULL_FACE);
        gl.useProgram(this.program);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'uMvp'), false, mvp);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'uView'), false, new Float32Array(view));
        gl.bindVertexArray(this.vao);
        gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
        gl.bindVertexArray(null);
    }
}

class Pixal3DTab {
    constructor() {
        this.ready = false;
        this.image = null;
        this.size = '';
        this.cutOut = false;
        this.socket = null;
        this.history = [];
    }

    el(id) {
        return document.getElementById(`pixal3d_${id}`);
    }

    init() {
        if (this.ready || !this.el('canvas')) {
            return;
        }
        this.ready = true;
        this.viewer = new Pixal3DViewer(this.el('canvas'));
        if (!this.viewer.gl) {
            this.el('model_info').innerText = 'This browser has no WebGL2: download the model to view it.';
        }
        let drop = this.el('drop');
        drop.addEventListener('click', () => this.el('file').click());
        this.el('file').addEventListener('change', e => e.target.files[0] && this.loadFile(e.target.files[0]));
        drop.addEventListener('dragover', e => {
            e.preventDefault();
            drop.classList.add('pixal3d-dragover');
        });
        drop.addEventListener('dragleave', () => drop.classList.remove('pixal3d-dragover'));
        drop.addEventListener('drop', e => {
            e.preventDefault();
            drop.classList.remove('pixal3d-dragover');
            if (e.dataTransfer.files[0]) {
                this.loadFile(e.dataTransfer.files[0]);
            }
        });
        // Ctrl+V of an image, while this tab is showing
        document.addEventListener('paste', e => {
            let file = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith('image/'));
            if (file && drop.offsetParent !== null) {
                e.preventDefault();
                this.loadFile(file);
            }
        });
        this.el('generate').addEventListener('click', () => this.generate());
        this.el('cancel').addEventListener('click', () => this.cancel());
        this.el('refresh_models').addEventListener('click', () => this.loadModels());
        this.el('remove_background').addEventListener('change', () => this.updateNote());
        this.el('model_type').addEventListener('change', () => this.setKind());
        this.el('bg_removal_name').addEventListener('change', () => this.updateNote());
        this.loadModels();
    }

    setStatus(text, error = false) {
        let s = this.el('status');
        s.innerText = text;
        s.classList.toggle('pixal3d-error', error);
    }

    setProgress(frac) {
        this.el('progress_bar').style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
    }

    /** Shows the fields of the chosen model: Pixal3D's mesh, or Idol's splat. */
    setKind() {
        let kind = this.el('model_type').value;
        this.el('root').dataset.kind = kind;
        this.el('generate').innerText = kind == 'splat' ? 'Generate splat' : 'Generate 3D';
    }

    loadModels() {
        genericRequest('Pixal3DListModels', {}, data => {
            let prefer = { unet_name: /pixal3d(?!.*multiview).*convrot/i, clip_vision_name: /dino_v3_L/i, shape_vae_name: /shape_vae/i, texture_vae_name: /texture_vae/i,
                splat_unet_name: /idol|triposplat/i, splat_clip_vision_name: /dino_v3_vit_h/i, splat_vae_name: /flux2/i, splat_decoder_name: /idol_decoder|triposplat.*decoder|decoder/i,
                bg_removal_name: /birefnet/i };
            this.available = { mesh: data.has_pixal3d !== false, splat: !!data.has_splat };
            for (let opt of this.el('model_type').options) {
                opt.disabled = !this.available[opt.value];
            }
            if (!this.available[this.el('model_type').value]) {
                let other = this.el('model_type').value == 'mesh' ? 'splat' : 'mesh';
                if (this.available[other]) {
                    this.el('model_type').value = other;
                }
            }
            this.setKind();
            for (let key of Object.keys(prefer)) {
                let select = this.el(key);
                let previous = select.value;
                select.innerHTML = '';
                for (let name of data[key] || []) {
                    let opt = document.createElement('option');
                    opt.value = name;
                    opt.innerText = name;
                    select.appendChild(opt);
                }
                let fallback = { bg_removal_name: list => list[0], unet_name: list => list.find(n => /pixal3d(?!.*multiview)/i.test(n)) }[key];
                let pick = (data[key] || []).find(n => n == previous) || (data[key] || []).find(n => prefer[key].test(n)) || (fallback ? fallback(data[key] || []) : null);
                if (pick) {
                    select.value = pick;
                }
            }
            this.setStatus('');
            this.updateNote();
        }, 0, e => this.setStatus(`${e}`, true));
    }

    loadFile(file) {
        let reader = new FileReader();
        reader.onload = () => {
            this.image = reader.result;
            let img = this.el('image_preview');
            img.src = this.image;
            img.style.display = '';
            this.el('drop_hint').style.display = 'none';
            img.onload = () => this.checkAlpha(img);
        };
        reader.readAsDataURL(file);
    }

    /** Looks for transparent pixels, since the alpha is the mask; an image without any gets its background removed. */
    checkAlpha(img) {
        let c = document.createElement('canvas');
        let scale = Math.min(1, 256 / Math.max(img.naturalWidth, img.naturalHeight));
        c.width = Math.max(1, Math.round(img.naturalWidth * scale));
        c.height = Math.max(1, Math.round(img.naturalHeight * scale));
        let ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, c.width, c.height);
        let data = ctx.getImageData(0, 0, c.width, c.height).data;
        let transparent = 0;
        for (let i = 3; i < data.length; i += 4) {
            if (data[i] < 128) {
                transparent++;
            }
        }
        this.size = `${img.naturalWidth}x${img.naturalHeight}`;
        this.cutOut = transparent > 0 && transparent < data.length / 4;
        this.el('remove_background').checked = !this.cutOut;
        this.updateNote();
    }

    /** The note under the image: its size, and what will make the mask. */
    updateNote() {
        let note = this.el('image_note');
        if (!this.size) {
            note.innerText = '';
            return;
        }
        let remove = this.el('remove_background').checked;
        let haveModel = !!this.el('bg_removal_name').value;
        let warn = false;
        let text = this.size;
        if (remove && !haveModel) {
            text += ': no background removal model found. Put birefnet.safetensors (Comfy-Org/BiRefNet) in the background_removal model folder and refresh the models, or use an image with a transparent background.';
            warn = true;
        } else if (remove) {
            text += this.cutOut ? ": the background will be removed, ignoring this image's transparency." : ': no transparent background, so the object is cut out with BiRefNet first.';
        } else if (!this.cutOut) {
            text += ': this image has no transparent background, and the object has to be cut out.';
            warn = true;
        }
        note.innerText = text;
        note.classList.toggle('pixal3d-warn', warn);
    }

    params() {
        let num = id => parseFloat(this.el(id).value);
        if (this.el('model_type').value == 'splat') {
            return {
                model_type: 'splat',
                image: this.image,
                seed: Math.round(num('seed')),
                splat_num_gaussians: Math.round(num('splat_num_gaussians')),
                splat_steps: Math.round(num('splat_steps')),
                splat_cfg: num('splat_cfg'),
                splat_unet_name: this.el('splat_unet_name').value,
                splat_clip_vision_name: this.el('splat_clip_vision_name').value,
                splat_vae_name: this.el('splat_vae_name').value,
                splat_decoder_name: this.el('splat_decoder_name').value,
                remove_background: this.el('remove_background').checked,
                bg_removal_name: this.el('bg_removal_name').value
            };
        }
        return {
            model_type: 'mesh',
            image: this.image,
            seed: Math.round(num('seed')),
            fov: num('fov'),
            resolution: Math.round(num('resolution')),
            max_faces: Math.round(num('max_faces')),
            texture: this.el('texture').checked,
            cfg: num('cfg'),
            structure_steps: Math.round(num('structure_steps')),
            shape_steps: Math.round(num('shape_steps')),
            refine_steps: Math.round(num('refine_steps')),
            texture_steps: Math.round(num('texture_steps')),
            unet_name: this.el('unet_name').value,
            clip_vision_name: this.el('clip_vision_name').value,
            shape_vae_name: this.el('shape_vae_name').value,
            texture_vae_name: this.el('texture_vae_name').value,
            remove_background: this.el('remove_background').checked,
            bg_removal_name: this.el('bg_removal_name').value
        };
    }

    setBusy(busy) {
        this.el('generate').disabled = busy;
        this.el('cancel').style.display = busy ? '' : 'none';
    }

    generate() {
        if (!this.image) {
            this.setStatus('Choose an image first.', true);
            return;
        }
        let p = this.params();
        if (p.model_type == 'splat' && !(p.splat_unet_name && p.splat_clip_vision_name && p.splat_vae_name && p.splat_decoder_name)) {
            this.setStatus('The Idol models are missing on the backends: idol_fp16 (diffusion_models), dino_v3_vit_h_bf16 (clip_vision), flux2-vae and idol_decoder_fp16 (VAE). Check the model files, then Refresh models.', true);
            return;
        }
        if (p.model_type == 'mesh' && !p.unet_name) {
            this.setStatus('No Pixal3D models found on the backends. Check the model files, then Refresh models.', true);
            return;
        }
        if (p.remove_background && !p.bg_removal_name) {
            this.updateNote();
            this.setStatus('Removing the background needs a BiRefNet model. See the note under the image.', true);
            return;
        }
        this.setBusy(true);
        this.setProgress(0);
        this.setStatus('Starting...');
        let started = Date.now();
        this.socket = makeWSRequest('Pixal3DGenerateWS', p, data => {
            if (data.status) {
                this.setStatus(data.status);
            }
            if (data.progress !== undefined) {
                this.setProgress(data.progress);
            }
            if (data.model) {
                this.setBusy(false);
                this.setProgress(1);
                this.setStatus(`Done in ${((Date.now() - started) / 1000).toFixed(0)} s (seed ${data.seed}).`);
                this.history.unshift({ path: data.model, seed: data.seed });
                this.showModel(data.model);
                this.renderHistory();
            }
        }, 0, e => {
            this.setBusy(false);
            this.setStatus(`${e}`, true);
        });
    }

    cancel() {
        if (this.socket && this.socket.readyState == WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ interrupt: true }));
        }
        this.setStatus('Cancelling...');
    }

    showModel(path) {
        this.current = path;
        let link = this.el('download');
        link.href = path;
        link.style.display = '';
        link.setAttribute('download', path.split('/').pop());
        let splat = path.toLowerCase().endsWith('.ply');
        link.innerText = splat ? 'Download PLY' : 'Download GLB';
        this.el('model_info').innerText = 'Loading model...';
        fetch(path).then(r => {
            if (!r.ok) {
                throw new Error(`HTTP ${r.status}`);
            }
            return r.arrayBuffer();
        }).then(buf => {
            let size = `${(buf.byteLength / 1048576).toFixed(1)} MB`;
            if (splat) {
                let s = pixal3dParsePly(buf);
                this.viewer.setSplat(s);
                this.el('model_info').innerText = `${s.count.toLocaleString()} gaussians, ${size}`;
            }
            else {
                let mesh = pixal3dParseGlb(buf);
                this.viewer.setMesh(mesh);
                this.el('model_info').innerText = `${mesh.vertexCount.toLocaleString()} vertices, ${(mesh.indices.length / 3).toLocaleString()} triangles, ${size}`;
            }
            this.renderHistory();
        }).catch(e => this.el('model_info').innerText = `Could not load the model: ${e}`);
    }

    renderHistory() {
        let box = this.el('history');
        box.innerHTML = '';
        for (let h of this.history) {
            let b = document.createElement('button');
            b.className = 'basic-button' + (h.path == this.current ? ' pixal3d-current' : '');
            b.innerText = h.path.split('/').pop();
            b.title = h.path;
            b.addEventListener('click', () => this.showModel(h.path));
            box.appendChild(b);
        }
    }
}

let pixal3dTab = new Pixal3DTab();
document.addEventListener('DOMContentLoaded', () => {
    let tab = document.getElementById('maintab_pixal3d');
    if (tab) {
        tab.addEventListener('shown.bs.tab', () => {
            pixal3dTab.init();
            pixal3dTab.viewer?.draw();
        });
    }
    if (document.getElementById('pixal3d_canvas')) {
        pixal3dTab.init();
    }
});
