#!/usr/bin/env python3
"""Run Python code inside the running Blender (bpy available, Z up, units = metres).
usage: python3 bl.py FILE.py | python3 bl.py -c "CODE"   [--inspect]  [--expect-rev REV]
The value of the last expression is returned as "result"; print() goes to "stdout".
--inspect adds full scene telemetry (bounds, contacts, collisions, topology issues).
--expect-rev REV asserts the scene is still at REV: refused with scene_stale otherwise.
With strict_external on (server-side), a call is also refused (error_code
external_changes_pending, no side effect) while teammate edits are unacked; the refusal
prints external_changes, the current rev and a hint: review, then resend the SAME call
with --expect-rev <rev> to acknowledge and proceed. A refused call changes nothing."""
import json, os, sys, urllib.request, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, ".bl.json")))
args = sys.argv[1:]; inspect = "--inspect" in args; args = [a for a in args if a != "--inspect"]
rev = None
if "--expect-rev" in args:
    i = args.index("--expect-rev"); rev = args[i + 1]; del args[i:i + 2]
if not args: sys.exit(__doc__)
code = args[1] if args[0] == "-c" else open(args[0]).read()
def post(payload):
    req = urllib.request.Request("http://127.0.0.1:%d/execute" % cfg["port"], data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["token"]})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=300).read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")
def clip(v, n=1500):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s[:n]
def trace(rec):
    with open(os.path.join(HERE, ".bl_trace.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
cnt_p = os.path.join(HERE, ".bl_count")
n = int(open(cnt_p).read()) if os.path.exists(cnt_p) else 0
open(cnt_p, "w").write(str(n + 1))
ext = cfg.get("external")
if ext and n == ext["after_calls"]:
    ext_r = post({"code": ext["code"], "diff": False})   # a teammate edits the scene in the Blender GUI
    trace({"call": n, "event": "teammate", "ok": ext_r.get("ok")})
payload = {"code": code, "inspect": inspect}
if cfg["arm"] == "base": payload["diff"] = False
if cfg["arm"] == "strict": payload["strict_external"] = True
if rev: payload["expect_rev"] = rev
r = post(payload)
trace({"call": n, "event": "agent", "expect_rev": rev, "ok": r.get("ok"), "error_code": r.get("error_code"),
       "rev": r.get("rev"), "external": r.get("external_changes"), "result": clip(r.get("result")),
       "stdout": clip(r.get("stdout", ""))})
keep = ["ok", "error", "error_code", "result", "stdout", "stderr", "traceback", "inspection"]
if cfg["arm"] in ("diff", "strict"): keep += ["rev", "external_changes", "diff"]
if cfg["arm"] == "strict": keep += ["hint"]
out = {k: r[k] for k in keep if k in r and r[k] not in ("", None)}
print(json.dumps(out, indent=1, ensure_ascii=False)[:20000])
