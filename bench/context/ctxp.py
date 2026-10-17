#!/usr/bin/env python3
"""Context-policy experiment (Laya as controller for noop | clm_edit | rlm_offload).

Every rollout = one scripted rpc session of STEPS steps (run.py's message script) with the host
policy `schedule`/`rule`/`laya` (OMP_CONTEXT_POLICY*) deciding per user turn. Per-step records carry
billed usage (cost-weighted), latency, edits, RLM sub-call usage and the controller state the
policy saw (OMP_CONTEXT_POLICY_LOG).

  ctxp.py cf     --tasks t1,t3 --seeds 0-6          # counterfactual rollouts (base + 3 points x 2 actions)
  ctxp.py run    --policy append|clm|clm-host|rule|laya|oracle --sources ...   # policy comparison
  analysis (labels, tuning, dataset, report) lives in ctxp_analysis.py
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R  # noqa: E402
from score import score  # noqa: E402
from tasks import envelopes_for, task_spec  # noqa: E402

OUT = os.path.join(HERE, "results", "ctxp")
STEPS = 12
DECISION_POINTS = (3, 6, 9)
WINDOW = 3
MODEL = "zai/glm-5.3-flash"
ACTIONS = {"noop": 0, "clm_edit": 1, "rlm_offload": 2}
NAMES = {v: k for k, v in ACTIONS.items()}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def source_id(task, seed, length="mid"):
    return f"{task}-{length}-seed{seed}" if seed else f"{task}-{length}"


def heldout(src):
    """SPEC.md rule: split by SOURCE, sha1(src) % 5 == 0 is held out."""
    return int(hashlib.sha1(src.encode()).hexdigest(), 16) % 5 == 0


def iso_epoch(ts):
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def cost_of(msgs):
    return sum(((m.get("usage") or {}).get("cost") or {}).get("total") or 0 for m in msgs)


def run_rollout(name, task, seed, policy, script=None, rule=None, strategy="clm", laya=None,
                steps=STEPS, model=MODEL, force=False):
    """policy: 'append' (plain), 'legacy-clm'/'legacy-clm-host' (harness maint every step),
    'schedule' (script of 0/1/2 per user turn), 'rule', 'laya'."""
    out_path = os.path.join(OUT, name + ".json")
    if os.path.exists(out_path) and not force:
        return json.load(open(out_path))
    spec = task_spec(task, "mid", seed)
    envs = {s: (t, k) for s, t, k in envelopes_for(task, steps)}
    home = tempfile.mkdtemp(prefix="omp-ctxp-")
    try:
        agent_dir, env = R.make_home(home)
        work = os.path.join(home, "work")
        os.makedirs(work)
        plog = os.path.join(home, "policy.jsonl")
        env["OMP_CONTEXT_OFFLOAD_DIR"] = os.path.join(work, ".offload")
        env["OMP_CONTEXT_OFFLOAD_MODEL"] = model
        env["OMP_RLM_MAX_CALLS"] = "12"
        legacy = policy.startswith("legacy-")
        if policy == "append":
            strat = "append"
        elif legacy:
            strat = policy[len("legacy-"):]
        else:
            strat = strategy
            env["OMP_CONTEXT_POLICY"] = policy
            env["OMP_CONTEXT_POLICY_LOG"] = plog
            if policy == "schedule":
                env["OMP_CONTEXT_POLICY_SCRIPT"] = ",".join(str(x) for x in (script or []))
            if policy == "rule":
                env["OMP_CONTEXT_POLICY_RULE_TOKENS"] = str(rule["tokens"])
                env["OMP_CONTEXT_POLICY_RULE_EVERY"] = str(rule["every"])
                env["OMP_CONTEXT_POLICY_RULE_OFFLOAD"] = str(rule.get("offload", 0))
            if policy == "laya":
                env["OMP_CONTEXT_LAYA_CKPT"] = laya["ckpt"]
                env["OMP_CONTEXT_LAYA_ONNX"] = laya["onnx"]
        cmd = [R.DEFAULT_BIN, "--mode", "rpc", "--model", model, "--thinking", "minimal", "--tools", "eval",
               "--no-extensions", "--no-skills", "--no-rules", "--no-title", "--no-lsp", "--no-pty",
               "--context-strategy", strat]
        rpc = R.Rpc(cmd, env, work)
        recs, replies, windows = [], [], []
        t_cell = time.time()
        try:
            for step in range(0, steps + 1):
                final = step == steps
                if step == 0:
                    msg = R.BRIEF
                else:
                    body = ""
                    if step == 1:
                        body = f"WORKING CONTEXT (the only source of truth):\n<<<CTXT\n{spec['seed']}\nCTXT>>>\n\n"
                    if final:
                        body += f"CURRENT INSTRUCTION (final step {step}/{steps}): " + spec["task"]
                    else:
                        body += (f"CURRENT INSTRUCTION (step {step}/{steps}): continue the task, append a "
                                 f"one-line progress note [step {step}] to your log.")
                    if step in envs:
                        body += f"\n\nINCOMING MESSAGE:\n{envs[step][0]}"
                    if legacy and not final:
                        body += R.CLM_MAINT
                    msg = body
                t0 = time.time()
                status, msgs, tools = rpc.prompt(msg, 600)
                t1 = time.time()
                windows.append((step, t0, t1))
                if status != "completed" or not msgs:
                    raise RuntimeError(f"step {step}: status={status} err={(msgs[-1].get('errorMessage') if msgs else None)}")
                reply = R.text_of(msgs[-1])
                if step > 0:
                    replies.append(reply)
                u = R.usage_of(msgs)
                recs.append({"step": step, "latency": round(t1 - t0, 2), **u, "cost": cost_of(msgs),
                             "edits_ok": sum(1 for t in tools if t["name"] == "context_edit" and not t["error"]),
                             "edits_rej": sum(1 for t in tools if t["name"] == "context_edit" and t["error"]),
                             "eval_calls": sum(1 for t in tools if t["name"] == "eval"),
                             "reply": reply[:4000 if final else 200]})
        finally:
            rpc.close()
        # sub-call (RLM) usage from the session file, attributed to steps by timestamp
        for root, _, files in os.walk(home):
            for fn in files:
                if fn.endswith(".jsonl") and "sessions" in root and fn != "eval-journal.jsonl":
                    for line in open(os.path.join(root, fn)):
                        d = json.loads(line)
                        if d.get("type") == "model_usage" and d.get("purpose") == "rlm":
                            ts = iso_epoch(d["timestamp"])
                            u = d.get("usage") or {}
                            for rec, (st, a, b) in zip(recs, windows):
                                if a - 1 <= ts <= b + 1:
                                    rec["rlm_calls"] = rec.get("rlm_calls", 0) + 1
                                    rec["uncached"] += u.get("input") or 0
                                    rec["cached"] += u.get("cacheRead") or 0
                                    rec["cache_write"] += u.get("cacheWrite") or 0
                                    rec["completion"] += u.get("output") or 0
                                    rec["cost"] += (u.get("cost") or {}).get("total") or 0
                                    break
        states = {}
        if os.path.exists(plog):
            for line in open(plog):
                d = json.loads(line)
                states[d["step"]] = d
        for rec in recs:
            d = states.get(rec["step"])
            rec["policy"] = {"action": d["action"], "state": d["state"]} if d else None
        verdict = score(f"{task}-mid", recs[-1]["reply"], spec["answer"], replies)
        tot = {k: sum(r.get(k, 0) for r in recs)
               for k in ("uncached", "cached", "cache_write", "completion", "calls", "cost", "edits_ok",
                         "edits_rej", "eval_calls", "rlm_calls")}
        denom = tot["uncached"] + tot["cached"] + tot["cache_write"]
        res = {"name": name, "source": source_id(task, seed), "task": task, "seed": seed, "policy": policy,
               "strategy": strat, "script": script, "rule": rule, "steps": steps, "model": model,
               "totals": tot, "cache_hit_pct": round(100 * tot["cached"] / denom, 1) if denom else 0.0,
               "latency_s": round(time.time() - t_cell, 1), "score": bool(verdict["verdict"]),
               "score_detail": verdict["detail"], "records": recs}
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
        return res
    finally:
        shutil.rmtree(home, ignore_errors=True)


def safe(fn, *a, **k):
    for attempt in range(2):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
            print("ERR", a[0] if a else "", err, flush=True)
    return {"name": a[0], "error": err}


def cf_jobs(tasks, seeds):
    jobs = []
    for t in tasks:
        for s in seeds:
            src = source_id(t, s)
            jobs.append((f"cf-{src}-base", t, s, "schedule", [0] * (STEPS + 1)))
            for d in DECISION_POINTS:
                for a in (1, 2):
                    sc = [0] * (STEPS + 1)
                    sc[d] = a
                    jobs.append((f"cf-{src}-d{d}-{NAMES[a]}", t, s, "schedule", sc))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cf")
    c.add_argument("--tasks", default="t1,t3")
    c.add_argument("--seeds", default="0-6")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--policy", required=True)
    r.add_argument("--sources", required=True, help="comma list like t1:0,t3:2")
    r.add_argument("--rule", default="")
    r.add_argument("--laya", default="")
    r.add_argument("--script-from", default="", help="json {source: script} (oracle)")
    r.add_argument("--tag", default="")
    r.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "cf":
        jobs = cf_jobs(a.tasks.split(","), parse_seeds(a.seeds))
        if a.limit:
            jobs = jobs[: a.limit]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = [ex.submit(safe, run_rollout, n, t, s, p, sc) for n, t, s, p, sc in jobs]
            for f in cf.as_completed(futs):
                x = f.result()
                print(x["name"], "ERR" if "error" in x else
                      f"score={x['score']} cost={x['totals']['cost']:.5f} lat={x['latency_s']} edits={x['totals']['edits_ok']} rlm={x['totals']['rlm_calls']}",
                      flush=True)
    else:
        srcs = [(x.split(":")[0], int(x.split(":")[1])) for x in a.sources.split(",")]
        scripts = json.load(open(a.script_from)) if a.script_from else {}
        rule = dict(zip(("tokens", "every", "offload"), map(int, a.rule.split(",")))) if a.rule else None
        laya = dict(zip(("ckpt", "onnx"), a.laya.split(","))) if a.laya else None
        pol = {"oracle": "schedule", "clm": "legacy-clm", "clm-host": "legacy-clm-host"}.get(a.policy, a.policy)
        tag = a.tag or a.policy  # reps: --tag append.r2 (report groups by the part before the dot)
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = []
            for t, s in srcs:
                src = source_id(t, s)
                futs.append(ex.submit(safe, run_rollout, f"ev-{tag}-{src}", t, s, pol,
                                      scripts.get(src), rule, "clm", laya))
            for f in cf.as_completed(futs):
                x = f.result()
                print(x["name"], "ERR" if "error" in x else
                      f"score={x['score']} cost={x['totals']['cost']:.5f} lat={x['latency_s']} edits={x['totals']['edits_ok']}",
                      flush=True)


if __name__ == "__main__":
    main()
