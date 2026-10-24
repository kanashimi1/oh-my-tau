#!/usr/bin/env python3
"""Context-strategy bench: scripted multi-step runs through `omp-rs --mode rpc`.

Port of the clm-blackbox harness to the real runtime. One *cell* = (task, length, strategy,
model, steps). Every cell is one rpc session (temp HOME, real provider credentials copied from
~/.omp/agent/agent.db, no tools except the strategy's own `context_edit`):

  step 0   handshake (the harness' system text; reply ACK)             -> same for every strategy
  step 1   WORKING CONTEXT (seed) + instruction            \\
  step k   "continue, append a one-line progress note"      |  envelopes injected at the
  step N   final instruction = the task                    /   clm-blackbox schedule

The history is whatever the strategy puts on the wire; the harness only talks rpc. Per-step usage
is summed over every assistant message of the prompt (a `context_edit` call adds an LLM call).

Metrics per cell: prompt uncached (usage.input), prompt cached (cacheRead), cache write, completion
tokens, cache-hit % = cached / (uncached + cached + cacheWrite), wall latency, task score.

  python3 bench/context/run.py --tasks t1 --strategies append,clm --model zai/glm-5.3-flash
  python3 bench/context/run.py --horizon --strategies append,clm --model kimi-code/kimi-for-coding
"""
import argparse
import json
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import score  # noqa: E402
from tasks import envelopes_for, task_spec  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_BIN = os.environ.get("OMP_RS_BIN", os.path.join(REPO, "target/shared/release/omt"))
REAL_AGENT_DIR = os.environ.get("OMP_REAL_AGENT_DIR", os.path.expanduser("~/.omp/agent"))

BRIEF = ("You are a careful agent working over a long WORKING CONTEXT that you must keep usable "
         "within limited space. Follow the CURRENT INSTRUCTION exactly. Never invent data that is "
         "not in the context. The WORKING CONTEXT arrives in the next message. Reply with exactly: ACK")

CLM_MAINT = ("\n\nCONTEXT MAINTENANCE: your context is managed. Raw conversation before your "
             "`context_edit` call is dropped from your context on the next turn, so first copy every "
             "task-critical fact (all `ANCHOR-<n> code=<c> value=<v>` lines, counts, PINs, incoming "
             "notices and instructions) into slots, then keep the slots compact. Then give your "
             "one-line log.")


# ------------------------------------------------------------------ rpc client

class Rpc:
    def __init__(self, cmd, env, cwd):
        self.p = subprocess.Popen(cmd, env=env, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        self.stderr = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=lambda: [self.stderr.append(l) for l in self.p.stderr], daemon=True).start()
        self.next_id = 1
        while True:
            f = self.frame(60)
            if f.get("type") == "ready":
                break

    def _pump(self):
        for line in self.p.stdout:
            line = line.strip()
            if line:
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.q.put({"type": "__eof__"})

    def frame(self, timeout):
        try:
            f = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("rpc frame timeout") from None
        if f.get("type") == "__eof__":
            raise RuntimeError("omp-rs exited: " + "".join(self.stderr[-10:]))
        return f

    def prompt(self, message, timeout=600):
        """Send one prompt; returns (frames, assistant_messages, tool_results)."""
        rid = self.next_id
        self.next_id += 1
        self.p.stdin.write(json.dumps({"id": rid, "type": "prompt", "message": message}) + "\n")
        self.p.stdin.flush()
        deadline = time.monotonic() + timeout
        assistants, tools, status = [], [], None
        while True:
            f = self.frame(max(1, deadline - time.monotonic()))
            t = f.get("type")
            if t == "message_end" and f["message"].get("role") == "assistant":
                assistants.append(f["message"])
            elif t == "tool_execution_end":
                tools.append({"name": f.get("toolName"), "error": bool(f.get("isError"))})
            elif t == "response" and f.get("id") == rid and f.get("success") is False:
                raise RuntimeError(f"prompt rejected: {f}")
            elif t == "prompt_result" and f.get("id") == rid:
                status = f.get("status")
                break
        return status, assistants, tools

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()


def make_home(root):
    agent = os.path.join(root, ".omp", "agent")
    os.makedirs(os.path.join(agent, "sessions"))
    src = sqlite3.connect(f"file:{os.path.join(REAL_AGENT_DIR, 'agent.db')}?mode=ro", uri=True)
    dst = sqlite3.connect(os.path.join(agent, "agent.db"))
    src.backup(dst)
    dst.close()
    src.close()
    for name in ("models.db",):
        p = os.path.join(REAL_AGENT_DIR, name)
        if os.path.exists(p):
            m = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            d = sqlite3.connect(os.path.join(agent, name))
            m.backup(d)
            d.close()
            m.close()
    if os.path.exists(os.path.join(REAL_AGENT_DIR, "models.yml")):
        shutil.copy(os.path.join(REAL_AGENT_DIR, "models.yml"), agent)
    with open(os.path.join(agent, "config.yml"), "w") as f:
        f.write("skills:\n  enabled: false\n")
    env = dict(os.environ, HOME=root, PI_CODING_AGENT_DIR=agent, OMP_AGENT_DIR=agent, NO_COLOR="1",
               TERM="dumb")
    return agent, env


def text_of(msg):
    return "\n".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")


def usage_of(msgs):
    u = {"uncached": 0, "cached": 0, "cache_write": 0, "completion": 0, "calls": len(msgs)}
    for m in msgs:
        mu = m.get("usage") or {}
        u["uncached"] += mu.get("input") or 0
        u["cached"] += mu.get("cacheRead") or 0
        u["cache_write"] += mu.get("cacheWrite") or 0
        u["completion"] += mu.get("output") or 0
    return u


# ------------------------------------------------------------------------ cell

def run_cell(a, task_id, length, strategy, steps, rep):
    spec = task_spec(task_id, length)
    envs = {s: (t, k) for s, t, k in envelopes_for(task_id, steps)}
    clm = strategy.startswith("clm")
    tag = f"{task_id}-{length}-{strategy}-s{steps}-r{rep}-{a.model.replace('/', '_')}"
    out_dir = os.path.join(a.out, a.run_id)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, tag + ".jsonl")
    home = tempfile.mkdtemp(prefix="omp-ctx-bench-")
    try:
        agent_dir, env = make_home(home)
        cmd = [a.bin, "--mode", "rpc", "--model", a.model, "--thinking", a.thinking, "--no-tools",
               "--no-extensions", "--no-skills", "--no-rules", "--no-title", "--no-lsp", "--no-pty",
               "--context-strategy", strategy]
        work = os.path.join(home, "work")
        os.makedirs(work, exist_ok=True)
        rpc = Rpc(cmd, env, work)
        records = []
        replies = []
        t_cell = time.monotonic()
        try:
            for step in range(0, steps + 1):
                final = step == steps
                if step == 0:
                    msg = BRIEF
                else:
                    if step == 1:
                        body = f"WORKING CONTEXT (the only source of truth):\n<<<CTXT\n{spec['seed']}\nCTXT>>>\n\n"
                    else:
                        body = ""
                    if final:
                        body += f"CURRENT INSTRUCTION (final step {step}/{steps}): " + spec["task"]
                    else:
                        body += (f"CURRENT INSTRUCTION (step {step}/{steps}): continue the task, append a "
                                 f"one-line progress note [step {step}] to your log.")
                    if step in envs:
                        body += f"\n\nINCOMING MESSAGE:\n{envs[step][0]}"
                    if clm and not final:
                        body += CLM_MAINT
                    msg = body
                t0 = time.monotonic()
                status, msgs, tools = rpc.prompt(msg, a.step_timeout)
                lat = time.monotonic() - t0
                reply = text_of(msgs[-1]) if msgs else ""
                if status != "completed" or not msgs:
                    raise RuntimeError(f"step {step}: status={status} calls={len(msgs)} "
                                       f"err={(msgs[-1].get('errorMessage') if msgs else None)}")
                if step > 0:
                    replies.append(reply)
                rec = {"step": step, "final": final, "latency": round(lat, 2), **usage_of(msgs),
                       "edits_ok": sum(1 for t in tools if t["name"] == "context_edit" and not t["error"]),
                       "edits_rejected": sum(1 for t in tools if t["name"] == "context_edit" and t["error"]),
                       "reply": reply[:4000 if final else 300]}
                records.append(rec)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            session_files = []
            for root, _, files in os.walk(home):
                session_files += [os.path.join(root, x) for x in files
                                  if x.endswith(".jsonl") and "sessions" in root]
            recorded = None
            for sf in session_files:
                with open(sf) as f:
                    for line in f:
                        if "context-strategy" in line:
                            recorded = json.loads(line)
            rpc.close()
        verdict = score(f"{task_id}-{length}", records[-1]["reply"], spec["answer"], replies)
        tot = {k: sum(r[k] for r in records) for k in
               ("uncached", "cached", "cache_write", "completion", "calls", "edits_ok", "edits_rejected")}
        denom = tot["uncached"] + tot["cached"] + tot["cache_write"]
        summary = {
            "tag": tag, "task": f"{task_id}-{length}", "strategy": strategy, "model": a.model,
            "steps": steps, "rep": rep, **tot,
            "cache_hit_pct": round(100 * tot["cached"] / denom, 1) if denom else 0.0,
            "latency_s": round(time.monotonic() - t_cell, 1),
            "score": bool(verdict["verdict"]), "score_detail": verdict["detail"],
            "session_records_strategy": (recorded or {}).get("data", {}).get("strategy")
            if recorded else None,
        }
        with open(os.path.join(out_dir, tag + ".summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        return summary
    finally:
        shutil.rmtree(home, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="t1,t2,t3")
    ap.add_argument("--length", default="mid")
    ap.add_argument("--strategies", default="append,clm")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--horizon", action="store_true", help="30-step t1-mid horizon")
    ap.add_argument("--model", default="zai/glm-5.3-flash")
    ap.add_argument("--thinking", default="minimal")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%S"))
    ap.add_argument("--step-timeout", type=int, default=600)
    a = ap.parse_args()
    if a.horizon:
        a.tasks, a.length, a.steps = "t1", "mid", 30
    rows = []
    for rep in range(1, a.repeats + 1):
        for task in a.tasks.split(","):
            for strat in a.strategies.split(","):
                try:
                    s = run_cell(a, task, a.length, strat, a.steps, rep)
                except Exception as e:  # noqa: BLE001
                    s = {"task": f"{task}-{a.length}", "strategy": strat, "model": a.model,
                         "steps": a.steps, "rep": rep, "error": str(e)[:300]}
                rows.append(s)
                print(json.dumps(s, ensure_ascii=False), flush=True)
    with open(os.path.join(a.out, a.run_id, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
