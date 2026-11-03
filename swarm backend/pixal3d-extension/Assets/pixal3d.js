// Pixal3D tab: image in, GLB out, shown in a small WebGL2 viewer (no third-party code).

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

/** Orbit viewer of one mesh. Faces are flat-shaded from screen derivatives, so their winding doesn't matter. */
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
        this.program = gl.createProgram();
        gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vs));
        gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(this.program);
        this.vao = gl.createVertexArray();
        this.buffers = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
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

    setMesh(mesh) {
        let gl = this.gl;
        if (!gl) {
            return;
        }
        let p = mesh.positions;
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
        if (!this.count) {
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
            this.el('model_info').innerText = 'This browser has no WebGL2: download the GLB to view it.';
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

    loadModels() {
        genericRequest('Pixal3DListModels', {}, data => {
            let prefer = { unet_name: /pixal3d(?!.*multiview).*convrot/i, clip_vision_name: /dino_v3/i, shape_vae_name: /shape_vae/i, texture_vae_name: /texture_vae/i, bg_removal_name: /birefnet/i };
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
                let pick = (data[key] || []).find(n => n == previous) || (data[key] || []).find(n => prefer[key].test(n)) || (key == 'bg_removal_name' ? (data[key] || [])[0] : (data[key] || []).find(n => /pixal3d(?!.*multiview)/i.test(n)));
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
            text += ': this image has no transparent background and Pixal3D needs the object cut out.';
            warn = true;
        }
        note.innerText = text;
        note.classList.toggle('pixal3d-warn', warn);
    }

    params() {
        let num = id => parseFloat(this.el(id).value);
        return {
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
        if (!p.unet_name) {
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
        this.el('model_info').innerText = 'Loading model...';
        fetch(path).then(r => {
            if (!r.ok) {
                throw new Error(`HTTP ${r.status}`);
            }
            return r.arrayBuffer();
        }).then(buf => {
            let mesh = pixal3dParseGlb(buf);
            this.viewer.setMesh(mesh);
            this.el('model_info').innerText = `${mesh.vertexCount.toLocaleString()} vertices, ${(mesh.indices.length / 3).toLocaleString()} triangles, ${(buf.byteLength / 1048576).toFixed(1)} MB`;
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
