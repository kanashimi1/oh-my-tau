// 3D preview for the artifacts panel: Blockbench .bbmodel (textured cubes,
// bone hierarchy, keyframe animations) and glTF / GLB (three.js GLTFLoader,
// skinning, animation clips). ES module, loaded on demand by artifacts.js.
//
//   const v = await mount(host, { url, format: "bbmodel" | "gltf" });
//   v.animations → names · v.play(name|null) · v.reload(url) · v.snapshot()
//   v.resetView() · v.dispose()
//
// bbmodel conventions (Blockbench): coordinates in pixels (16 = 1 block),
// rotations in degrees applied in ZYX order around the cube/bone `origin`,
// face UVs in texture pixels; animation rotations negate x/y, positions
// negate x (Bedrock/GeckoLib channel convention).
import * as THREE from "/assets/vendor/three.min.js";

const DEG = Math.PI / 180;

// Face → corners (x, y, z: 0 = from, 1 = to) for top-left, top-right,
// bottom-left as seen from outside; UV (u0,v0) → tl, (u1,v0) → tr, (u0,v1) → bl.
const FACE_CORNERS = {
  north: [[1, 1, 0], [0, 1, 0], [1, 0, 0]],
  south: [[0, 1, 1], [1, 1, 1], [0, 0, 1]],
  east: [[1, 1, 1], [1, 1, 0], [1, 0, 1]],
  west: [[0, 1, 0], [0, 1, 1], [0, 0, 0]],
  up: [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
  down: [[0, 0, 1], [1, 0, 1], [0, 0, 0]],
};

const num = (v, d = 0) => {
  if (typeof v === "number") return Number.isFinite(v) ? v : d;
  if (v == null || String(v).trim() === "") return d;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN; // NaN → molang expression
};

// ── molang (the subset keyframes use) ─────────────────────────────────────
// Translated to JS only when every identifier left is whitelisted.
const MOLANG_LIB = {
  sind: (d) => Math.sin(d * DEG), cosd: (d) => Math.cos(d * DEG),
  m_abs: Math.abs, m_floor: Math.floor, m_ceil: Math.ceil, m_round: Math.round, m_sqrt: Math.sqrt,
  m_min: Math.min, m_max: Math.max, m_pow: Math.pow, m_exp: Math.exp, m_ln: Math.log, m_trunc: Math.trunc,
  m_clamp: (v, a, b) => Math.min(Math.max(v, a), b), m_lerp: (a, b, k) => a + (b - a) * k,
  m_mod: (a, b) => a % b, m_pi: Math.PI, m_random: (a = 0, b = 1) => a + Math.random() * (b - a),
  m_asin: (v) => Math.asin(v) / DEG, m_acos: (v) => Math.acos(v) / DEG, m_atan: (v) => Math.atan(v) / DEG,
  m_atan2: (y, x) => Math.atan2(y, x) / DEG,
};
const MOLANG_IDS = new Set(["t", ...Object.keys(MOLANG_LIB)]);
const molangCache = new Map();
function molang(src) {
  if (molangCache.has(src)) return molangCache.get(src);
  let fn = () => 0;
  const js = String(src).toLowerCase().replace(/;\s*$/, "")
    .replace(/\b(query|q)\.(anim_time|life_time)\b/g, "t")
    .replace(/\b(query|q|variable|v|temp|context|c)\.[a-z_][a-z0-9_]*/g, "0")
    .replace(/\bmath\.(sin|cos)\b/g, "$1d")
    .replace(/\bmath\.([a-z0-9_]+)\b/g, "m_$1");
  const ids = js.match(/[a-z_][a-z0-9_]*/g) || [];
  if (/^[\s\w.+\-*/%()<>=!?:&|,]*$/.test(js) && ids.every((id) => MOLANG_IDS.has(id))) {
    try {
      const names = Object.keys(MOLANG_LIB);
      const f = new Function("t", ...names, `"use strict"; return (${js});`);
      const lib = names.map((k) => MOLANG_LIB[k]);
      fn = (t) => { const r = Number(f(t, ...lib)); return Number.isFinite(r) ? r : 0; };
      fn(0);
    } catch { fn = () => 0; }
  }
  molangCache.set(src, fn);
  return fn;
}
const value = (v, t, d = 0) => {
  const n = num(v, d);
  return Number.isNaN(n) ? molang(v)(t) : n;
};

// ── bbmodel → three.js ────────────────────────────────────────────────────
function textureOf(model, t, manager) {
  const w = t.uv_width || model.resolution?.width || 16;
  const h = t.uv_height || model.resolution?.height || 16;
  let map = null;
  if (t.source && t.source.startsWith("data:")) {
    map = new THREE.TextureLoader(manager).load(t.source);
    map.magFilter = THREE.NearestFilter;
    map.minFilter = THREE.NearestFilter;
    map.generateMipmaps = false;
    map.colorSpace = THREE.SRGBColorSpace;
  }
  const material = new THREE.MeshLambertMaterial({
    map, color: map ? 0xffffff : 0xb8b8b8, alphaTest: 0.05, side: THREE.DoubleSide,
  });
  return { material, w, h };
}

function cubeMesh(el, mats, fallback) {
  const inf = num(el.inflate, 0);
  const lo = el.from.map((v) => num(v) - inf), hi = el.to.map((v) => num(v) + inf);
  const origin = (el.origin || [0, 0, 0]).map((v) => num(v));
  const byMat = new Map();
  for (const [face, corners] of Object.entries(FACE_CORNERS)) {
    const f = el.faces && el.faces[face];
    if (!f || f.texture === null || f.texture === undefined || f.enabled === false) continue;
    const tex = mats[f.texture] || fallback;
    const [tl, tr, bl] = corners;
    const br = tl.map((_, i) => tr[i] + bl[i] - tl[i]);
    const pos = (c) => [c[0] ? hi[0] : lo[0], c[1] ? hi[1] : lo[1], c[2] ? hi[2] : lo[2]].map((v, i) => v - origin[i]);
    const [u0, v0, u1, v1] = (f.uv || [0, 0, 0, 0]).map((v) => num(v));
    const uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].map(([u, v]) => [u / tex.w, 1 - v / tex.h]);
    const steps = ((Math.round(num(f.rotation, 0) / 90) % 4) + 4) % 4;
    const at = (i) => uvs[(i - steps + 4) % 4];
    // clockwise corners tl, tr, br, bl → triangles (tl, bl, tr), (tr, bl, br)
    const quad = [[tl, at(0)], [tr, at(1)], [br, at(2)], [bl, at(3)]];
    let g = byMat.get(tex);
    if (!g) byMat.set(tex, (g = { pos: [], uv: [] }));
    for (const i of [0, 3, 1, 1, 3, 2]) { g.pos.push(...pos(quad[i][0])); g.uv.push(...quad[i][1]); }
  }
  const obj = new THREE.Group();
  for (const [tex, g] of byMat) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(g.pos, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(g.uv, 2));
    geo.computeVertexNormals();
    obj.add(new THREE.Mesh(geo, tex.material));
  }
  const r = el.rotation || [0, 0, 0];
  const rot = Array.isArray(r) ? r : (() => { // legacy {angle, axis}
    const a = [0, 0, 0]; a["xyz".indexOf(r.axis)] = num(r.angle); return a;
  })();
  obj.rotation.set(num(rot[0]) * DEG, num(rot[1]) * DEG, num(rot[2]) * DEG, "ZYX");
  obj.userData.origin = origin;
  return obj;
}

function buildBbmodel(model, manager) {
  const groups = new Map((model.groups || []).map((g) => [g.uuid, g]));
  const elements = new Map((model.elements || []).map((e) => [e.uuid, e]));
  const mats = (model.textures || []).map((t) => textureOf(model, t, manager));
  const fallback = textureOf(model, {}, manager);
  const bones = new Map();
  const root = new THREE.Group();

  function visit(node, parent, parentOrigin) {
    if (typeof node === "string") {
      const el = elements.get(node);
      if (!el || el.visibility === false || (el.type && el.type !== "cube") || !el.from || !el.to) return;
      const mesh = cubeMesh(el, mats, fallback);
      const o = mesh.userData.origin;
      mesh.position.set(o[0] - parentOrigin[0], o[1] - parentOrigin[1], o[2] - parentOrigin[2]);
      parent.add(mesh);
      return;
    }
    const g = { ...(groups.get(node.uuid) || {}), ...node };
    if (g.visibility === false) return;
    const origin = (g.origin || [0, 0, 0]).map((v) => num(v));
    const rot = (g.rotation || [0, 0, 0]).map((v) => num(v) * DEG);
    const bone = new THREE.Group();
    bone.name = g.name || "bone";
    bone.position.set(origin[0] - parentOrigin[0], origin[1] - parentOrigin[1], origin[2] - parentOrigin[2]);
    bone.rotation.set(rot[0], rot[1], rot[2], "ZYX");
    bone.userData.base = { pos: bone.position.clone(), rot: rot.slice() };
    bones.set(g.uuid, bone);
    if (g.name) bones.set("name:" + g.name, bone);
    parent.add(bone);
    for (const c of g.children || []) visit(c, bone, origin);
  }
  const outliner = model.outliner && model.outliner.length ? model.outliner : [...elements.keys()];
  for (const n of outliner) visit(n, root, [0, 0, 0]);
  root.scale.setScalar(1 / 16);
  return { root, bones, animations: model.animations || [] };
}

// Keyframe sampling (Blockbench linear / catmullrom / step; bezier → linear).
function sampler(keys) {
  const sorted = keys.slice().sort((a, b) => num(a.time) - num(b.time));
  const vec = (k, t, post) => {
    const dp = (post && k.data_points[1]) || k.data_points[0] || {};
    return ["x", "y", "z"].map((c) => value(dp[c], t, 0));
  };
  return (t) => {
    if (t <= num(sorted[0].time)) return vec(sorted[0], t, false);
    for (let i = 0; i < sorted.length - 1; i++) {
      const a = sorted[i], b = sorted[i + 1];
      const t0 = num(a.time), t1 = num(b.time);
      if (t < t0 || t > t1) continue;
      const va = vec(a, t, true), vb = vec(b, t, false);
      if (a.interpolation === "step" || t1 <= t0) return va;
      const u = (t - t0) / (t1 - t0);
      if (a.interpolation === "catmullrom" || b.interpolation === "catmullrom") {
        const p0 = vec(sorted[Math.max(0, i - 1)], t, true), p3 = vec(sorted[Math.min(sorted.length - 1, i + 2)], t, false);
        return va.map((v, j) => 0.5 * (2 * v + (-p0[j] + vb[j]) * u + (2 * p0[j] - 5 * v + 4 * vb[j] - p3[j]) * u * u
          + (-p0[j] + 3 * v - 3 * vb[j] + p3[j]) * u * u * u));
      }
      return va.map((v, j) => v + (vb[j] - v) * u);
    }
    return vec(sorted[sorted.length - 1], t, true);
  };
}

function bbAnimation(anim, bones) {
  const tracks = [];
  for (const [key, an] of Object.entries(anim.animators || {})) {
    const bone = bones.get(key) || (an.name && bones.get("name:" + an.name));
    if (!bone) continue;
    const by = (ch) => (an.keyframes || []).filter((k) => k.channel === ch && k.data_points && k.data_points.length);
    const mk = (ch) => (by(ch).length ? sampler(by(ch)) : null);
    tracks.push({ bone, rot: mk("rotation"), pos: mk("position"), scale: mk("scale") });
  }
  const times = Object.values(anim.animators || {}).flatMap((a) => (a.keyframes || []).map((k) => num(k.time)));
  const length = num(anim.length, 0) || Math.max(0, ...times) || 1;
  return {
    name: anim.name, length,
    apply(t) {
      for (const tr of tracks) {
        const { pos, rot } = tr.bone.userData.base;
        const r = tr.rot && tr.rot(t), p = tr.pos && tr.pos(t), s = tr.scale && tr.scale(t);
        tr.bone.rotation.set(rot[0] - (r ? r[0] * DEG : 0), rot[1] - (r ? r[1] * DEG : 0), rot[2] + (r ? r[2] * DEG : 0), "ZYX");
        tr.bone.position.set(pos.x - (p ? p[0] : 0), pos.y + (p ? p[1] : 0), pos.z + (p ? p[2] : 0));
        if (s) tr.bone.scale.set(s[0], s[1], s[2]); else tr.bone.scale.set(1, 1, 1);
      }
    },
  };
}

function resetBones(bones) {
  for (const bone of new Set(bones.values())) {
    const { pos, rot } = bone.userData.base;
    bone.position.copy(pos);
    bone.rotation.set(rot[0], rot[1], rot[2], "ZYX");
    bone.scale.set(1, 1, 1);
  }
}

// ── loading ───────────────────────────────────────────────────────────────
async function loadModel(url, format) {
  const manager = new THREE.LoadingManager();
  const settled = new Promise((resolve) => { manager.onLoad = resolve; manager.onError = resolve; });
  if (format === "bbmodel") {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const model = await r.json();
    const textured = (model.textures || []).some((t) => t.source && t.source.startsWith("data:"));
    const built = buildBbmodel(model, manager);
    if (textured) await Promise.race([settled, new Promise((ok) => setTimeout(ok, 3000))]);
    return {
      scene: built.root,
      clips: built.animations.map((a) => bbAnimation(a, built.bones)),
      reset: () => resetBones(built.bones),
    };
  }
  const gltf = await new THREE.GLTFLoader(manager).loadAsync(url);
  const scene = gltf.scene || gltf.scenes[0];
  const mixer = new THREE.AnimationMixer(scene);
  let action = null;
  return {
    scene,
    clips: (gltf.animations || []).map((clip) => ({
      name: clip.name || "animation", length: clip.duration || 1,
      start() { if (action) action.stop(); action = mixer.clipAction(clip); action.reset().play(); },
      apply(t) { mixer.setTime(t); },
    })),
    reset: () => { if (action) action.stop(); action = null; mixer.setTime(0); },
  };
}

// ── viewer ────────────────────────────────────────────────────────────────
export async function mount(host, { url, format }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x1b1d24);
  const wrap = document.createElement("div");
  wrap.className = "v3d";
  const bar = document.createElement("div");
  bar.className = "v3d-bar";
  wrap.append(renderer.domElement, bar);
  host.replaceChildren(wrap);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x5a5560, 1.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(-3, 6, 4);
  scene.add(sun);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  let grid = null;

  const state = { model: null, clip: null, playing: true, t0: 0, pausedAt: 0, raf: 0, disposed: false };
  const select = document.createElement("select");
  select.title = "Animation";
  const playBtn = document.createElement("button");
  playBtn.className = "icon-btn";
  const resetBtn = document.createElement("button");
  resetBtn.className = "icon-btn";
  resetBtn.textContent = "⟲";
  resetBtn.title = "Reset view";
  const info = document.createElement("span");
  info.className = "v3d-info";
  bar.append(select, playBtn, resetBtn, info);

  // Front three-quarter view: Blockbench entities face north (-z), glTF
  // models face +z. Fits the bounding sphere in the narrower field of view.
  const VIEW_DIR = format === "bbmodel" ? new THREE.Vector3(-0.6, 0.35, -1) : new THREE.Vector3(-0.6, 0.35, 1);
  let userMoved = false;
  controls.addEventListener("start", () => { userMoved = true; });
  const frameModel = () => {
    userMoved = false;
    const box = new THREE.Box3().setFromObject(state.model.scene);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    const radius = size.length() / 2 || 1;
    const vHalf = (camera.fov * DEG) / 2;
    const hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
    const dist = (radius / Math.sin(Math.min(vHalf, hHalf))) * 0.92;
    camera.near = dist / 100; camera.far = dist * 100; camera.updateProjectionMatrix();
    camera.position.copy(center).add(VIEW_DIR.clone().normalize().multiplyScalar(dist));
    controls.target.copy(center);
    controls.update();
    if (grid) scene.remove(grid);
    const span = Math.max(size.x, size.z) * 2.4;
    const cells = Math.max(4, Math.min(64, Math.round(span)));
    grid = new THREE.GridHelper(span, cells, 0x3b3f4c, 0x2a2d36);
    grid.position.set(center.x, box.min.y, center.z);
    scene.add(grid);
  };

  const now = () => performance.now() / 1000;
  const setPlaying = (on) => {
    if (!on) state.pausedAt = now() - state.t0;
    else state.t0 = now() - state.pausedAt;
    state.playing = on;
    playBtn.textContent = on ? "⏸" : "▶";
    playBtn.title = on ? "Pause" : "Play";
  };
  const pick = (name) => {
    state.model.reset();
    state.clip = state.model.clips.find((c) => c.name === name) || null;
    if (state.clip && state.clip.start) state.clip.start();
    state.pausedAt = 0;
    state.t0 = now();
    select.value = state.clip ? state.clip.name : "";
    setPlaying(true);
  };
  select.onchange = () => pick(select.value);
  playBtn.onclick = () => setPlaying(!state.playing);
  resetBtn.onclick = () => frameModel();

  const resize = () => {
    const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (state.model && !userMoved) frameModel();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(host);

  const render = () => {
    if (state.clip && state.model) {
      const t = ((state.playing ? now() - state.t0 : state.pausedAt) % state.clip.length + state.clip.length) % state.clip.length;
      state.clip.apply(t);
      info.textContent = `${t.toFixed(2)} / ${state.clip.length.toFixed(2)}s`;
    } else info.textContent = "";
    controls.update();
    renderer.render(scene, camera);
  };
  const loop = () => { if (state.disposed) return; render(); state.raf = requestAnimationFrame(loop); };

  async function load(src, keepView) {
    const next = await loadModel(src, format);
    const prevClip = state.clip && state.clip.name;
    if (state.model) scene.remove(state.model.scene);
    state.model = next;
    scene.add(next.scene);
    const label = (n) => n.replace(/^animation\.[^.]+\./, "");
    select.replaceChildren(new Option("— pose —", ""), ...next.clips.map((c) => new Option(label(c.name), c.name)));
    select.hidden = playBtn.hidden = !next.clips.length;
    if (!keepView) frameModel();
    pick(keepView && prevClip && next.clips.some((c) => c.name === prevClip) ? prevClip : "");
  }

  resize();
  await load(url, false);
  loop();

  return {
    get animations() { return state.model ? state.model.clips.map((c) => c.name) : []; },
    get time() { return state.clip ? ((state.playing ? now() - state.t0 : state.pausedAt) % state.clip.length) : 0; },
    get clip() { return state.clip ? state.clip.name : null; },
    play: (name) => pick(name || ""),
    pause: () => setPlaying(false),
    reload: (src) => load(src || url, true),
    resetView: frameModel,
    snapshot() { render(); return renderer.domElement.toDataURL("image/png"); },
    dispose() {
      state.disposed = true;
      cancelAnimationFrame(state.raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
      renderer.dispose();
    },
  };
}
