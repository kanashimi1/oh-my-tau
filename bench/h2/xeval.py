#!/usr/bin/env python3
"""Follow-up evals for papers 2610.18784 (CarryOn), 2610.01415 (PoS), 2610.41432 (HCU) on the h2 task suite.

  xeval.py compact --arms none,default,cgs --tasks ... --reps N   two-phase task, forced compaction between phases
  xeval.py loop    --arms none,generic,pos --tasks ... --reps N   stagnation/loop detector extension (ext/pos-guard.ts)
  xeval.py gui     --arms base,hcu --tasks ... --reps N           simulated Minecraft GUI through the real mcp_mc.py server

Rows go to results/x-<cmd>-<label>.log (one JSON per line).
"""
import argparse, concurrent.futures as cf, datetime as dt, json, os, shutil, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "context")); sys.path.insert(0, HERE)
import run as R  # noqa: E402
import h2tasks as T  # noqa: E402
import h2  # noqa: E402
import xtasks as XT  # noqa: E402


def task_mod(tid):
    return XT if tid.startswith("t_") else T

RESULTS = os.path.join(HERE, "results")
EXT = os.path.join(HERE, "ext")


class Sess:
    """One omt rpc session; several prompts / commands; per-phase steps + usage."""

    def __init__(self, ws, model, extra=(), config_yml="", env_extra=None, no_ext=True):
        self.root = tempfile.mkdtemp(prefix="xe-h-"); agent, env = R.make_home(self.root)
        if config_yml:
            with open(os.path.join(agent, "config.yml"), "a") as f: f.write(config_yml)
        env.update(env_extra or {})
        cmd = h2.base_cmd(ws, *extra, model=model)
        if not no_ext: cmd.remove("--no-extensions")
        self.c = R.Rpc(cmd, env, ws); self.rid = 0

    def _send(self, obj):
        self.rid += 1; obj = dict(obj, id=self.rid)
        self.c.p.stdin.write(json.dumps(obj) + "\n"); self.c.p.stdin.flush(); return self.rid

    def command(self, obj, timeout=300):
        rid = self._send(obj); deadline = time.monotonic() + timeout
        while True:
            f = self.c.frame(max(1, deadline - time.monotonic()))
            if f.get("type") == "response" and f.get("id") == rid:
                return f

    def prompt(self, message, timeout=600):
        rid = self._send({"type": "prompt", "message": message}); deadline = time.monotonic() + timeout
        steps, assistants, starts, status = [], [], {}, "timeout"
        try:
            while True:
                f = self.c.frame(max(1, deadline - time.monotonic())); t = f.get("type")
                if t == "tool_execution_start":
                    starts[f.get("toolCallId")] = f.get("args") or f.get("input") or {}
                elif t == "tool_execution_end":
                    res = f.get("result") or {}
                    txt = "\n".join(b.get("text", "") for b in (res.get("content") or []) if isinstance(b, dict))
                    steps.append({"tool": f.get("toolName"), "args": h2._clip(starts.get(f.get("toolCallId"), {}), 600),
                                  "error": bool(f.get("isError")), "out": h2._clip(txt, 700)})
                elif t == "message_end" and (f.get("message") or {}).get("role") == "assistant":
                    assistants.append(f["message"])
                elif t == "response" and f.get("id") == rid and f.get("success") is False:
                    status = "rejected: " + json.dumps(f)[:200]; break
                elif t == "prompt_result" and f.get("id") == rid:
                    status = f.get("status"); break
        except (TimeoutError, RuntimeError) as e:
            status = f"error: {e}"[:200]
        return {"status": status, "steps": steps, "usage": R.usage_of(assistants),
                "final": R.text_of(assistants[-1]) if assistants else ""}

    def close(self):
        self.c.close(); shutil.rmtree(self.root, ignore_errors=True)


def tok(u):
    return u["uncached"] + u["cached"] + u["completion"]


def emit(path, row):
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(path, "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")


def grid(args, fn, cmd):
    arms = args.arms.split(","); tids = args.tasks.split(",")
    jobs = [(arm, tid, rep) for rep in range(args.reps) for tid in tids for arm in arms]
    out = os.path.join(RESULTS, f"x-{cmd}-{args.label}.log")
    with cf.ThreadPoolExecutor(args.par) as ex:
        futs = [ex.submit(fn, arm, tid, rep, args) for arm, tid, rep in jobs]
        for fu in cf.as_completed(futs):
            try:
                emit(out, fu.result())
            except Exception as e:  # noqa: BLE001  one broken run must not kill the grid
                emit(out, {"status": f"harness-error: {e}"[:300]})


# ---------------------------------------------------------------- 1. compaction (CarryOn CGS)
PHASE1 = ("\n\nPHASE 1 ONLY: investigate the inputs and the existing code/data thoroughly and note every quirk, "
          "format detail and edge case that matters for a correct result. Do NOT create or modify any files yet; "
          "end with a short findings list.")
PHASE2 = "Now complete the original task in full (create/modify the required files) and verify the result."
# CarryOn-Guided Summary principles (paper p.18-23, Tables 12/14), as compaction focus instructions.
CGS = """Organize the summary around reusable task objects plus an execution context, not chronology.
OBJECTS: for every file/dataset/result the work depends on, give its exact path and what it contains or what is known about it (formats, encodings, separators, units, quirks, counts, exact values), with the narrowest scope the evidence supports. Retain exact values, names and excerpts the next step needs to act without re-reading; a bare path without a useful account is not enough.
EXECUTION CONTEXT: (1) the goal and every requirement of the original task, verbatim where precise; (2) open obligations: each deliverable or instruction step not yet completed, one imperative line each; (3) active constraints; (4) the next atomic action.
RULES: every retained sentence must change a future action or prevent rediscovery; an action whose outcome was not observed is an attempt with unknown result, never a success; never turn a tentative finding into a fact; do not keep raw chronology or generic advice."""


def compact_one(arm, tid, rep, a):
    ws = tempfile.mkdtemp(prefix=f"xc-{tid}-"); T.setup(tid, ws)
    s = Sess(ws, a.model, ["--tools", h2.EXEC_TOOLS, "--max-time", "10m"],
             config_yml="compaction:\n  keepRecentTokens: 0\n", env_extra={"OMP_COMPACTION_MODEL": a.summarizer})
    row = {"arm": arm, "task": tid, "rep": rep, "model": a.model}
    try:
        p1 = s.prompt(T.TASKS[tid][1] + PHASE1, timeout=420)
        wrote = [x for x in p1["steps"] if x["tool"] in ("write", "edit")]
        summ = ""
        if arm != "none":
            r = s.command({"type": "compact", **({"customInstructions": CGS} if arm == "cgs" else {})})
            if r.get("success") is False:
                row["compact_error"] = json.dumps(r)[:300]
            summ = ((r.get("data") or {}).get("summary") or "")
        p2 = s.prompt(PHASE2, timeout=480)
    finally:
        s.close()
    score, miss = T.check(tid, ws); shutil.rmtree(ws, ignore_errors=True)
    rereads = sum(1 for x in p2["steps"] if x["tool"] in ("read",) or (x["tool"] == "bash" and any(k in x["args"] for k in ("cat ", "head", "ls", "file ", "xxd", "od "))))
    row.update(status=f"{p1['status']}|{p2['status']}", score=round(score, 3), miss=miss, p1_writes=len(wrote),
               summary_chars=len(summ), p1_calls=len(p1["steps"]), p2_calls=len(p2["steps"]), p2_rereads=rereads,
               p1_tok=tok(p1["usage"]), p2_tok=tok(p2["usage"]), errors=sum(x["error"] for x in p1["steps"] + p2["steps"]))
    return row


# ---------------------------------------------------------------- 3. loop / stagnation detector (PoS)
def loop_one(arm, tid, rep, a):
    TM = task_mod(tid); ws = tempfile.mkdtemp(prefix=f"xl-{tid}-"); TM.setup(tid, ws)
    extra = ["--tools", h2.EXEC_TOOLS, "--max-time", "8m"]
    env = {"POS_GUARD_MODE": arm, "POS_GUARD_LOG": os.path.join(ws, ".posguard.jsonl")}
    env["POS_GUARD_MODE"] = "shadow" if arm == "none" else arm  # none = detector logs only, never injects
    extra += ["--extension", os.path.join(EXT, "pos-guard.ts")]
    s = Sess(ws, a.model, extra, env_extra=env, no_ext=False)
    try:
        r = s.prompt(TM.TASKS[tid][1], timeout=540)
    finally:
        s.close()
    fires = []
    if os.path.exists(env["POS_GUARD_LOG"]):
        fires = [json.loads(l) for l in open(env["POS_GUARD_LOG"]) if l.strip()]
        os.remove(env["POS_GUARD_LOG"])
    score, miss = TM.check(tid, ws); shutil.rmtree(ws, ignore_errors=True)
    errs = [x["error"] or ("Traceback" in x["out"]) for x in r["steps"]]
    streak = mx = 0
    for e in errs: streak = streak + 1 if e else 0; mx = max(mx, streak)
    return {"arm": arm, "task": tid, "rep": rep, "model": a.model, "status": r["status"], "score": round(score, 3), "miss": miss,
            "calls": len(r["steps"]), "errors": sum(errs), "max_err_streak": mx, "tokens": tok(r["usage"]),
            "fires": sum(1 for f in fires if f.get("fired")), "would_fire": sum(1 for f in fires if f.get("would")),
            "patterns": [f"{f.get('pattern')}/{f.get('gap')}" for f in fires if f.get("would")], "guard_rows": len(fires)}


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compact", "loop", "gui", "blend"):
        p = sp.add_parser(name); p.add_argument("--arms", required=True); p.add_argument("--tasks", required=True)
        p.add_argument("--reps", type=int, default=2); p.add_argument("--par", type=int, default=4)
        p.add_argument("--model", default=h2.EXEC_MODEL); p.add_argument("--summarizer", default="zai/glm-5.3-flash:low")
        p.add_argument("--mcp-prefix", default="mcp__sim_")
        p.add_argument("--label", default=dt.datetime.now().strftime("%Y%m%d-%H%M"))
    a = ap.parse_args(); os.makedirs(RESULTS, exist_ok=True)
    if a.cmd == "compact": grid(a, compact_one, "compact")
    elif a.cmd == "loop": grid(a, loop_one, "loop")
    elif a.cmd == "blend":
        import xblend
        grid(a, xblend.blend_one, "blend")
    else:
        import xgui
        grid(a, xgui.gui_one, "gui")


if __name__ == "__main__":
    main()
