#!/usr/bin/env python3
"""Phase B: true counterfactual labels by branching from session snapshots.

For a source (family L1/L2, seed) the *prefix* run executes the episode with an all-noop host policy and copies the
session file before each decision step d. Every action branch then resumes that snapshot in a fresh process
(`switch_session`), sends one warm-up turn (re-warms the provider cache identically for all actions; its cost is NOT
counted), applies the action at the next turn (host policy `schedule`), runs H more episode steps and a quality probe
(questions with known ground truth). Window cost = Σ cost of those H turns + probe; label = cheapest action whose probe
score is not worse than noop (non-noop needs >= MARGIN window saving).

  branch.py prefix --family L1 --seeds 10-17 [--steps 36]
  branch.py branches --family L1 --seeds 10-17 --workers 2
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longbench as L  # noqa: E402
import run as R  # noqa: E402

OUT = os.path.join(HERE, "results", "branch")
SNAP = os.path.join(OUT, "snap")
H = 4
MARGIN = 0.03
POINTS = {"L1": [2, 6, 10, 14, 18, 22, 26, 30], "L2": [4, 9, 14, 19, 24, 29, 34]}
STEPS = {"L1": 36, "L2": 40}
ACTION_ID = {"noop": 0, "clm_edit": 1, "rlm_offload": 2, "rlm_recall": 3}


def _env(home, schedule):
    agent_dir, env = R.make_home(home)
    work = os.path.join(home, "work")
    os.makedirs(work, exist_ok=True)
    env["OMP_CONTEXT_OFFLOAD_DIR"] = os.path.join(work, ".offload")
    env["OMP_CONTEXT_OFFLOAD_MODEL"] = "smol"
    env["OMP_RLM_MAX_CALLS"] = "80"
    env["OMP_CONTEXT_POLICY"] = "schedule"
    env["OMP_CONTEXT_POLICY_SCRIPT"] = ",".join(str(x) for x in schedule)
    plog = os.path.join(home, "policy.jsonl")
    env["OMP_CONTEXT_POLICY_LOG"] = plog
    with open(os.path.join(agent_dir, "config.yml"), "a") as f:
        f.write(f"modelRoles:\n  smol: {L.RELAY}\n")
    return env, work, plog


def _cmd():
    return [R.DEFAULT_BIN, "--mode", "rpc", "--model", L.MODEL, "--thinking", "minimal", "--tools", "bash,read,eval",
            "--no-extensions", "--no-skills", "--no-rules", "--no-title", "--no-lsp", "--no-pty",
            "--context-strategy", "clm"]


def _session_file(home):
    fs = [p for p in glob.glob(home + "/**/*.jsonl", recursive=True)
          if "sessions" in p and not p.endswith("eval-journal.jsonl")]
    return max(fs, key=os.path.getmtime) if fs else None


def _write_files(work, ep):
    for rel, content in ep["files"].items():
        p = os.path.join(work, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)


def run_prefix(family, seed):
    steps = STEPS[family]
    name = f"{family}-s{seed}"
    done = os.path.join(SNAP, name, "done.json")
    if os.path.exists(done):
        return json.load(open(done))
    ep = L.get_episode(family, seed, steps)
    pts = POINTS[family]
    os.makedirs(os.path.join(SNAP, name), exist_ok=True)
    home = tempfile.mkdtemp(prefix="omp-br-")
    try:
        env, work, plog = _env(home, [0] * (steps + 4))
        _write_files(work, ep)
        rpc = R.Rpc(_cmd(), env, work)
        recs = []
        try:
            for step in range(0, max(pts)):
                if step in pts:
                    sf = _session_file(home)
                    shutil.copy(sf, os.path.join(SNAP, name, f"d{step}.jsonl"))
                msg = ep["brief"] if step == 0 else ep["msgs"][step]
                t0 = time.time()
                status, msgs, tools = rpc.prompt(msg, ep["timeout"])
                if status != "completed" or not msgs:
                    raise RuntimeError(f"prefix step {step}: {status}")
                recs.append({"step": step, "cost": L.cost_of(msgs), "latency": round(time.time() - t0, 1)})
            if max(pts) not in [r["step"] for r in recs]:
                shutil.copy(_session_file(home), os.path.join(SNAP, name, f"d{max(pts)}.jsonl"))
        finally:
            rpc.close()
        res = {"name": name, "points": pts, "prefix_cost": sum(r["cost"] for r in recs)}
        json.dump(res, open(done, "w"))
        with L.LOCK:
            with open(L.LEDGER, "a") as f:
                f.write(json.dumps({"name": "prefix-" + name, "cost": res["prefix_cost"], "t": time.time()}) + "\n")
        return res
    finally:
        shutil.rmtree(home, ignore_errors=True)


def run_branch(family, seed, d, action):
    name = f"{family}-s{seed}-d{d}-{action}"
    out = os.path.join(OUT, name + ".json")
    if os.path.exists(out):
        return json.load(open(out))
    steps = STEPS[family]
    ep = L.get_episode(family, seed, steps)
    snap = os.path.join(SNAP, f"{family}-s{seed}", f"d{d}.jsonl")
    home = tempfile.mkdtemp(prefix="omp-brb-")
    try:
        # warm turn has st.step = d, decision turn st.step = d + 1
        sched = [0] * (steps + 6)
        sched[d + 1] = ACTION_ID[action]
        env, work, plog = _env(home, sched)
        _write_files(work, ep)
        rpc = R.Rpc(_cmd(), env, work)
        rid = rpc.next_id
        rpc.next_id += 1
        rpc.p.stdin.write(json.dumps({"id": rid, "type": "switch_session", "sessionPath": snap}) + "\n")
        rpc.p.stdin.flush()
        while True:
            f = rpc.frame(300)
            if f.get("type") == "response" and f.get("id") == rid:
                if not f.get("success"):
                    raise RuntimeError(f"switch_session failed: {f}")
                break
        recs = []
        try:
            status, msgs, tools = rpc.prompt("WARM-UP: reply with exactly: ok", 600)
            ptext, pscore = ep["probe"](d + H)
            turns = [ep["msgs"][k] for k in range(d, d + H)] + [ptext]
            reply = ""
            for i, msg in enumerate(turns):
                t0 = time.time()
                status, msgs, tools = rpc.prompt(msg, ep["timeout"])
                if status != "completed" or not msgs:
                    raise RuntimeError(f"branch turn {i}: {status}")
                reply = L.R.text_of(msgs[-1])
                tn = {}
                for t in tools:
                    tn[t["name"]] = tn.get(t["name"], 0) + 1
                recs.append({"i": i, "cost": L.cost_of(msgs), "latency": round(time.time() - t0, 1),
                             "calls": len(msgs), "tools": tn})
        finally:
            rpc.close()
        sub = 0
        for root, _, files in os.walk(home):
            for fn in files:
                if fn.endswith(".jsonl") and "sessions" in root and fn != "eval-journal.jsonl":
                    sub += sum(1 for l in open(os.path.join(root, fn)) if '"purpose":"rlm"' in l)
        state = None
        if os.path.exists(plog):
            for line in open(plog):
                dd = json.loads(line)
                if dd["step"] == d + 1:
                    state = dd["state"]
        res = {"name": name, "family": family, "seed": seed, "d": d, "action": action, "cost": sum(r["cost"] for r in recs),
               "latency": sum(r["latency"] for r in recs), "score": pscore(reply), "sub_calls": sub, "state": state,
               "tool_calls": sum(sum(r["tools"].values()) for r in recs), "turns": recs}
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        with L.LOCK:
            with open(L.LEDGER, "a") as f:
                f.write(json.dumps({"name": name, "cost": res["cost"], "t": time.time()}) + "\n")
        return res
    finally:
        shutil.rmtree(home, ignore_errors=True)


def block_chars(state):
    import re
    m = re.search(r"largest_old_block_chars: (\d+)", state or "")
    return int(m.group(1)) if m else 0


def safe(fn, *a):
    for _ in range(2):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            print("ERR", a, str(e)[:200], flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefix", "branches"])
    ap.add_argument("--family", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--actions", default="noop,rlm_offload,clm_edit")
    ap.add_argument("--points", default="")
    ap.add_argument("--max-spend", type=float, default=15.0)
    a = ap.parse_args()
    seeds = L.parse_seeds(a.seeds)
    os.makedirs(SNAP, exist_ok=True)
    pts = [int(x) for x in a.points.split(",")] if a.points else POINTS[a.family]
    if a.cmd == "prefix":
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(lambda s: safe(run_prefix, a.family, s), seeds):
                print(r, flush=True)
        return
    acts = a.actions.split(",")
    jobs = []
    for s in seeds:
        for d in pts:
            # noop first: its state decides whether offload is possible
            jobs.append((s, d))

    def point(sd):
        s, d = sd
        if L.spent() > a.max_spend:
            return None
        base = safe(run_branch, a.family, s, d, "noop")
        out = [base]
        if base and block_chars(base["state"]) >= 2000 and "rlm_offload" in acts:
            out.append(safe(run_branch, a.family, s, d, "rlm_offload"))
        if "clm_edit" in acts:
            out.append(safe(run_branch, a.family, s, d, "clm_edit"))
        return out

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for out in ex.map(point, jobs):
            if out:
                print(" ".join(f"{r['action']}:{r['cost']*1e3:.2f}m$/{r['score']:.2f}" for r in out if r), flush=True)


if __name__ == "__main__":
    main()
