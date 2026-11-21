"""Blender scene-diff eval (HCU-style feedback) through the AI CAD bridge, headless Blender 4.2.3.

Arms: base = bridge replies without diff (diff:false); diff = replies carry diff/rev/external_changes; strict = diff + strict_external (bridge refuses calls while teammate edits are unacked; the agent must resend with --expect-rev).
The agent drives Blender only through ./bl.py (template xblend_bl.py) in its workspace (bash tool).
"""
import json, os, secrets, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.expanduser("~/fleet-split/blender-ai-toolkit")
BLENDER = os.path.expanduser("~/opt/blender-4.2.3-linux-x64/blender")
BL_CLI = open(os.path.join(HERE, "xblend_bl.py")).read()

STARTER = """
import os, sys, time, bpy
sys.path.insert(0, os.environ["TOOLKIT"]); sys.path.insert(0, os.path.join(os.environ["TOOLKIT"], "tests"))
bpy.ops.wm.read_factory_settings(use_empty=True)
import ai_cad_bridge
ai_cad_bridge._server = ai_cad_bridge._Bridge(port=int(os.environ["BL_PORT"]))
print("AI_CAD_BRIDGE_READY", flush=True)
while not bpy.context.scene.get("bridge_test_stop", False):
    ai_cad_bridge._server._drain(); time.sleep(.005)
ai_cad_bridge.unregister()
"""

TABLE = """
import bpy
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
def box(name, loc, dims):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc); o = bpy.context.active_object; o.name = name; o.dimensions = dims; return o
box("Top", (0, 0, 0.725), (1.2, 0.8, 0.05))
for i, (x, y) in enumerate([(-0.55, -0.35), (0.55, -0.35), (-0.55, 0.35), (0.55, 0.35)], 1):
    box(f"Leg{i}", (x, y, 0.35), (0.05, 0.05, 0.7))
box("Lamp", (0.3, 0.1, 0.85), (0.1, 0.1, 0.2))
bpy.context.view_layer.update()
"""

MECHA_CHECK = """
import bpy, json
bpy.context.view_layer.update()
from inspect_scene import inspect_scene
import mecha_fixture
r = inspect_scene(expected_contacts=mecha_fixture.CONTACTS, collisions=True)
c = {f"{x['a']}-{x['b']}": x["status"] for x in r["expected_contacts"]}
horn = bpy.data.objects.get("horn")
hx = (sum((horn.matrix_world @ v.co).x for v in horn.data.vertices) / len(horn.data.vertices)) if horn else 9
t, n = bpy.data.objects["torso"], bpy.data.objects["neck"]
json.dumps({"head_neck": c.get("neck-head") in ("touching", "intersecting"), "horn_centred": abs(hx) < 0.01,
            "jaw_normals": not any(i["type"] == "inverted_normals" and i["object"] == "jaw" for i in r["issues"]),
            "torso_neck_fixed": max(abs(t.location.x), abs(t.location.y), abs(t.location.z - 1.4), abs(n.location.x), abs(n.location.y), abs(n.location.z - 2.55)) < 1e-3})
"""

TABLE_CHECK = """
import bpy, json
bpy.context.view_layer.update()
from mathutils import Vector
def bb(n):
    o = bpy.data.objects[n]; ps = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return [min(p[i] for p in ps) for i in range(3)], [max(p[i] for p in ps) for i in range(3)]
ok = {}
lo, hi = bb("Top"); ok["top_size"] = abs((hi[0]-lo[0]) - 1.6) < 2e-3 and abs((hi[1]-lo[1]) - 0.8) < 2e-3 and abs(hi[2]-0.75) < 2e-3 and abs(lo[2]-0.7) < 2e-3
cx = (lo[0]+hi[0])/2; cy = (lo[1]+hi[1])/2
want = {(-1, -1), (1, -1), (-1, 1), (1, 1)}; got = set(); legs_ok = True
for i in range(1, 5):
    l, h = bb(f"Leg{i}"); mx, my = (l[0]+h[0])/2, (l[1]+h[1])/2
    sx = 1 if mx > cx else -1; sy = 1 if my > cy else -1; got.add((sx, sy))
    ex = cx + sx*(0.8-0.05-0.025); ey = cy + sy*(0.4-0.05-0.025)
    legs_ok = legs_ok and abs(mx-ex) < 3e-3 and abs(my-ey) < 3e-3 and abs(l[2]) < 2e-3 and abs(h[2]-lo[2]) < 2e-3
ok["legs"] = legs_ok and got == want
l, h = bb("Lamp"); ok["lamp"] = abs(l[2]-hi[2]) < 2e-3 and abs(hi[0] - (l[0]+h[0])/2 - 0.3) < 5e-3
json.dumps(ok)
"""

POSTS_SETUP = """
import bpy
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1)); bpy.context.active_object.name = "Crate"
"""

POSTS_CHECK = """
import bpy, json, math
bpy.context.view_layer.update()
from mathutils import Vector
ok = {}
c = bpy.data.objects.get("Crate"); ok["crate"] = c is not None and (c.location - Vector((0, 0, 1))).length < 1e-3
pts, okp = [], True
for i in range(8):
    o = bpy.data.objects.get(f"Post{i}")
    if not o:
        okp = False; pts.append(None); continue
    ps = [o.matrix_world @ Vector(v) for v in o.bound_box]; lo = [min(p[k] for p in ps) for k in range(3)]; hi = [max(p[k] for p in ps) for k in range(3)]
    m = Vector(((lo[0]+hi[0])/2, (lo[1]+hi[1])/2)); a = 2*math.pi*i/8; pts.append(m)
    okp = okp and (m - Vector((3*math.cos(a), 3*math.sin(a)))).length < 5e-3 and abs(lo[2]) < 2e-3 and abs(hi[2]-1.0) < 2e-3 and abs((hi[0]-lo[0]) - 0.2) < 0.02
ok["posts"] = okp
okr = okp
for i in range(8):
    o = bpy.data.objects.get(f"Rail{i}")
    if not o or not okp:
        okr = False; continue
    ps = [o.matrix_world @ Vector(v) for v in o.bound_box]; cen = sum(ps, Vector()) / 8
    a, b = pts[i], pts[(i+1) % 8]; mid = (a + b) / 2
    length = max((ps[j]-ps[k]).length for j in range(8) for k in range(8))
    okr = okr and (Vector((cen.x, cen.y)) - mid).length < 0.01 and abs(cen.z - 0.8) < 0.01 and abs(length - math.sqrt((a-b).length**2 + 0.05**2 + 0.05**2)) < 0.03
ok["rails"] = okr
json.dumps(ok)
"""

BENCH = """
import bpy
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
def box(name, loc, dims):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc); o = bpy.context.active_object; o.name = name; o.dimensions = dims; return o
box("Seat", (0, 0, 0.475), (1.2, 0.3, 0.05))
box("LegL", (-0.525, 0, 0.225), (0.05, 0.05, 0.45))
box("LegR", (0.525, 0, 0.225), (0.05, 0.05, 0.45))
box("Sign", (0.4, 0, 0.55), (0.15, 0.1, 0.05))
bpy.context.view_layer.update()
"""

SIGN_CHECK = """
import bpy, json
bpy.context.view_layer.update()
from mathutils import Vector
def bb(n):
    o = bpy.data.objects[n]; ps = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return [min(p[i] for p in ps) for i in range(3)], [max(p[i] for p in ps) for i in range(3)]
ok = {}
lo, hi = bb("Seat"); cx = (lo[0]+hi[0])/2
ok["seat"] = abs((hi[0]-lo[0]) - 1.8) < 2e-3 and abs((hi[1]-lo[1]) - 0.3) < 2e-3 and abs(hi[2]-0.5) < 2e-3 and abs(lo[2]-0.45) < 2e-3 and abs(cx) < 2e-3
legs = True
for n, s in (("LegL", -1), ("LegR", 1)):
    l, h = bb(n)
    legs = legs and abs((l[0]+h[0])/2 - (cx + s*(0.9-0.075))) < 3e-3 and abs(l[2]) < 2e-3 and abs(h[2]-0.45) < 2e-3 and abs((l[1]+h[1])/2) < 2e-3 and abs((h[1]-l[1])-0.05) < 2e-3
ok["legs"] = legs
if bpy.data.objects.get("Sign"):
    l, h = bb("Sign"); ok["sign"] = abs((l[0]+h[0])/2) < 2e-3 and abs((l[1]+h[1])/2) < 2e-3 and abs(l[2]-0.5) < 2e-3
else:
    ok["sign"] = False
json.dumps(ok)
"""

STRICT_ADDENDUM = ("\n\nBridge note (strict mode): your bl.py calls run with strict_external checks. If a teammate "
                   "changed the scene since your last accepted call, the bridge refuses the call with error_code "
                   "external_changes_pending and no side effect; the refusal lists external_changes (what changed), "
                   "the current rev and a hint. Review those changes, take them into account in your plan, then "
                   "resend the same call with --expect-rev <rev from the refusal> to acknowledge and proceed. "
                   "Every successful reply also carries the new rev. A refused call has no effect on the scene.\n")

TASKS = {
    "b_mecha": dict(
        setup="import mecha_fixture; mecha_fixture.build(defective=True)",
        prompt=("A Blender scene with a small mecha (objects torso, neck, head, horn, jaw, shoulder plates, ...) is open. "
                "Repair it without moving torso or neck: the head must rest on the neck (touching, no gap), the horn must be centred "
                "on x=0 above the head, and the jaw mesh must have outward-facing normals. Use only `python3 bl.py` to inspect and edit "
                "the scene (run it without arguments for usage; `import ai_cad as cad` works inside Blender). Verify before finishing."),
        check=MECHA_CHECK),
    "b_table": dict(
        setup=TABLE,
        prompt=("A Blender scene with a table (Top, Leg1..Leg4) and a Lamp on it is open. Make the table top 1.6 m long in X (keep "
                "its width, thickness and height), move the legs so each stays 0.05 m inset from its corner and still touches the "
                "floor (z=0) and the underside of the top, and keep the Lamp standing on the top surface with its centre 0.3 m from the +X edge. "
                "A teammate may be editing the same scene in the Blender GUI while you work; the final scene must meet every "
                "requirement. Use only `python3 bl.py` (run it without arguments for usage). Verify before finishing."),
        external=dict(after_calls=2, code="import bpy\nbpy.data.objects['Leg3'].location.x += 0.4\nbpy.data.objects['Lamp'].location.z += 0.15\n"),
        check=TABLE_CHECK),
    "b_posts": dict(
        setup=POSTS_SETUP,
        prompt=("A Blender scene is open. Around the existing object Crate, create a fence of 8 cylinders named Post0..Post7 "
                "(radius 0.1, height 1.0) evenly spaced on a circle of radius 3 centred on the Crate's XY centre, Post0 on +X, "
                "counter-clockwise, all standing on z=0; then join them with 8 horizontal boxes Rail0..Rail7 (0.05 x 0.05 cross-section) "
                "running between the two neighbouring posts' centres at height 0.8 (Rail_i between Post_i and Post_(i+1) mod 8). "
                "Do not move the Crate. Use only `python3 bl.py` (run it without arguments for usage). Verify before finishing."),
        check=POSTS_CHECK),
    "b_sign": dict(
        setup=BENCH,
        prompt=("A Blender scene contains a bench: Seat (a 1.2 x 0.3 x 0.05 board, top at z=0.5), legs LegL and "
                "LegR standing on the floor, and a small Sign plaque standing on top of the Seat. Make the Seat "
                "1.8 m long in X (keep its width, thickness, height and centre), keep both legs 0.05 m inset from "
                "the seat ends, standing on the floor (z=0) and touching the underside of the seat, and move the "
                "Sign so it stands on the Seat top centred at x=0, y=0. A teammate may be editing the same scene "
                "in the Blender GUI while you work; the final scene must meet every requirement. Use only "
                "`python3 bl.py` (run it without arguments for usage). Verify before finishing."),
        external=dict(after_calls=2, code="import bpy\nbpy.data.objects.remove(bpy.data.objects['Sign'], do_unlink=True)\n"),
        check=SIGN_CHECK),
}


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Blender:
    def __init__(self):
        self.port, self.token = _free_port(), secrets.token_urlsafe(16)
        self.dir = tempfile.mkdtemp(prefix="xb-bl-"); st = os.path.join(self.dir, "start.py"); open(st, "w").write(STARTER)
        env = dict(os.environ, TOOLKIT=TOOLKIT, BL_PORT=str(self.port), AI_CAD_BRIDGE_TOKEN=self.token)
        self.p = subprocess.Popen([BLENDER, "-b", "--factory-startup", "--python", st], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ready = threading.Event()
        def pump():
            for line in self.p.stdout:
                if "AI_CAD_BRIDGE_READY" in line: ready.set()
        threading.Thread(target=pump, daemon=True).start()
        if not ready.wait(90):
            self.p.kill(); raise RuntimeError("blender did not start")

    def run(self, code, **kw):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/execute", data=json.dumps({"code": code, "diff": False, **kw}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=300).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read() or b"{}")

    def close(self):
        try: self.run("import bpy\nbpy.context.scene['bridge_test_stop'] = True")
        except Exception: pass
        try: self.p.wait(15)
        except subprocess.TimeoutExpired: self.p.kill()
        shutil.rmtree(self.dir, ignore_errors=True)


def blend_one(arm, tid, rep, a):
    import xeval as X
    spec = TASKS[tid]; bl = Blender(); ws = tempfile.mkdtemp(prefix=f"xb-{tid}-")
    try:
        s0 = bl.run(spec["setup"])
        if not s0.get("ok"): raise RuntimeError("setup failed: " + str(s0)[:300])
        open(os.path.join(ws, "bl.py"), "w").write(BL_CLI)
        json.dump({"port": bl.port, "token": bl.token, "arm": arm, "external": spec.get("external")}, open(os.path.join(ws, ".bl.json"), "w"))
        s = X.Sess(ws, a.model, ["--tools", "bash,read,write,edit", "--max-time", "10m"])
        try:
            r = s.prompt(spec["prompt"] + (STRICT_ADDENDUM if arm == "strict" else ""), timeout=660)
        finally:
            s.close()
        chk = bl.run(spec["check"])
        checks = json.loads(chk["result"]) if chk.get("ok") else {"check_error": False}
        cp = os.path.join(ws, ".bl_count"); n_bl = int(open(cp).read()) if os.path.exists(cp) else 0
        tdir = os.path.join(HERE, "results", "x-blend-traces"); os.makedirs(tdir, exist_ok=True)
        tp = os.path.join(ws, ".bl_trace.jsonl")
        bl_trace = [json.loads(l) for l in open(tp) if l.strip()] if os.path.exists(tp) else []
        json.dump({"steps": r["steps"], "final": r["final"], "bl": bl_trace, "checks": checks},
                  open(os.path.join(tdir, f"{a.label}-{arm}-{tid}-{rep}.json"), "w"), ensure_ascii=False, indent=0)
    finally:
        bl.close(); shutil.rmtree(ws, ignore_errors=True)
    return {"arm": arm, "task": tid, "rep": rep, "model": a.model, "status": r["status"], "score": round(sum(checks.values()) / len(checks), 3),
            "success": all(checks.values()), "checks": checks, "calls": len(r["steps"]), "bl_calls": n_bl,
            "errors": sum(x["error"] for x in r["steps"]), "tokens": X.tok(r["usage"])}