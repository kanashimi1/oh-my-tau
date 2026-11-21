#!/usr/bin/env python3
"""Harness² pilot for omt (paper 2610.14224): Propose -> Probe -> Compose over an EXPERIENCE.md library.

Executor and editor are both omt --mode rpc on zai/glm-5.3-flash, each in a throw-away HOME + workspace.
Open-world: the editor never sees hidden checkers; it sees traces only. Hidden scores are logged for the report.

  h2.py run  TASK [--lib FILE] [--tag T]          one executor episode -> results/episodes.jsonl
  h2.py evolve [--k 2] [--out lib/EXPERIENCE.md]  stream tasks: propose K lessons, probe each by replay, compose supported
  h2.py eval --libs none=,naive=F,h2=F --reps 2   evaluate arms on stream + held-out tasks
"""
import argparse, concurrent.futures as cf, datetime as dt, json, os, re, shutil, sys, tempfile, time, uuid
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "context")); sys.path.insert(0, HERE)
import run as R  # noqa: E402
import h2tasks as T  # noqa: E402

MODEL = os.environ.get("H2_MODEL", "zai/glm-5.3-flash")  # editor (and default executor)
EXEC_MODEL = os.environ.get("H2_EXEC_MODEL", MODEL)
RESULTS = os.path.join(HERE, "results"); LIBDIR = os.path.join(HERE, "lib")
EXEC_TOOLS = "bash,read,edit,write"
FENCE = chr(96) * 3
LIB_HEADER = "EXPERIENCE (lessons from earlier tasks; each was kept only because replaying a task with it changed behavior for the better):\n"


def base_cmd(ws, *extra, model=None):
    return [R.DEFAULT_BIN, "--mode", "rpc", "--model", model or MODEL, "--thinking", "low", "--no-session", "--no-extensions",
            "--approval-mode", "yolo", "--cwd", ws, *extra]


def _clip(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[: n // 2] + f" …[{len(s) - n} cut]… " + s[-n // 2:]


def _add_provider_header(path, provider, key, value):
    import yaml
    cfg = (yaml.safe_load(open(path)) if os.path.exists(path) else None) or {}
    cfg.setdefault("providers", {}).setdefault(provider, {}).setdefault("headers", {})[key] = value
    with open(path, "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)


def session(cmd_extra, message, ws, timeout=600, model=None, env_extra=None):
    """Run one rpc prompt; return dict(status, final, steps[], usage, secs)."""
    root = tempfile.mkdtemp(prefix="h2h-"); agent, env = R.make_home(root)
    env.update(env_extra or {})
    if (model or MODEL).startswith("opencode-go/"):
        # opencode-go routes only requests that carry x-opencode-session; omt adds it for compaction only.
        _add_provider_header(os.path.join(agent, "models.yml"), "opencode-go", "x-opencode-session", f"h2-{uuid.uuid4().hex}")
    t0 = time.monotonic(); c = R.Rpc(base_cmd(ws, *cmd_extra, model=model), env, ws); rid = 1
    c.p.stdin.write(json.dumps({"id": rid, "type": "prompt", "message": message}) + "\n"); c.p.stdin.flush()
    steps, assistants, status, starts = [], [], "timeout", {}
    deadline = time.monotonic() + timeout
    try:
        while True:
            f = c.frame(max(1, deadline - time.monotonic())); t = f.get("type")
            if t == "tool_execution_start":
                starts[f.get("toolCallId")] = f.get("args") or f.get("input") or {}
            elif t == "tool_execution_end":
                res = f.get("result") or {}
                txt = "\n".join(b.get("text", "") for b in (res.get("content") or []) if isinstance(b, dict))
                steps.append({"tool": f.get("toolName"), "args": _clip(starts.get(f.get("toolCallId"), {}), 600),
                              "error": bool(f.get("isError")), "out": _clip(txt, 700)})
            elif t == "message_end" and (f.get("message") or {}).get("role") == "assistant":
                assistants.append(f["message"])
            elif t == "prompt_result" and f.get("id") == rid:
                status = f.get("status"); break
    except (TimeoutError, RuntimeError) as e:
        status = f"error: {e}"[:200]
    finally:
        c.close(); shutil.rmtree(root, ignore_errors=True)
    final = R.text_of(assistants[-1]) if assistants else ""
    return {"status": status, "final": final, "steps": steps, "usage": R.usage_of(assistants), "secs": round(time.monotonic() - t0, 1)}


def scope_lib(lib_text, tid):
    """Keep only invariant lessons + the task's domain (d_* -> domain:data, c_* -> domain:code).

    Understands both library layouts: '## <scope>' sections (composed) and '- [<scope>] ...' lines (naive)."""
    want = {"invariant", "domain:data" if tid.startswith("d_") else "domain:code"}
    out, sec = [], "invariant"
    for line in lib_text.splitlines():
        h = re.match(r"^#+\s*(\S+)", line)
        if h:
            sec = h.group(1).strip().lower(); out.append(line) if sec in want else None; continue
        m = re.match(r"^\s*-\s*\[([^\]]+)\]", line)
        scope = m.group(1).strip().lower() if m else sec
        if scope in want: out.append(line)
    return "\n".join(out)


def execute(tid, lib_text="", tag="", keep=False, model=None, scoped=False):
    ws = tempfile.mkdtemp(prefix=f"h2w-{tid}-"); T.setup(tid, ws)
    extra = ["--tools", EXEC_TOOLS, "--max-time", "8m"]
    if scoped: lib_text = scope_lib(lib_text, tid)
    if lib_text.strip():
        extra += ["--append-system-prompt", LIB_HEADER + lib_text.strip()]
    r = session(extra, T.TASKS[tid][1], ws, model=model or EXEC_MODEL)
    score, miss = T.check(tid, ws)
    r.update(task=tid, tag=tag, score=round(score, 3), miss=miss, lib_chars=len(lib_text), model=model or EXEC_MODEL,
             ts=dt.datetime.now().isoformat(timespec="seconds"))
    if not keep: shutil.rmtree(ws, ignore_errors=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "episodes.jsonl"), "a") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return r


def trace_text(r, n=14000):
    lines = [f"[{i}] {s['tool']}{' ERROR' if s['error'] else ''} args={s['args']}\n    -> {s['out']}" for i, s in enumerate(r["steps"])]
    body = "\n".join(lines) + f"\n[final answer] {_clip(r['final'], 1500)}\n[status] {r['status']}  tool_calls={len(r['steps'])} errors={sum(s['error'] for s in r['steps'])}"
    return _clip(body, n)


def editor(prompt, timeout=300):
    ws = tempfile.mkdtemp(prefix="h2e-")
    try: r = session(["--no-tools", "--max-time", "5m"], prompt, ws, timeout)
    finally: shutil.rmtree(ws, ignore_errors=True)
    return r["final"], r["usage"]


def parse_json(text):
    m = re.search(r"\[.*\]|\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return None


def strip_fence(s):
    s = s.strip()
    if s.startswith(FENCE): s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith(FENCE): s = s[: -3]
    return s.strip()


PROPOSE = """You improve the operating notes ("EXPERIENCE library") of a coding/data agent. Below are the task the agent got,
the current library, and the agent's full tool trace. You do NOT know whether the result was correct; judge from the trace.
Find behaviors that risk wrong results or wasted calls (unverified assumptions, ignored format details, no self-check, retries).
Propose {k} DISTINCT candidate lessons. Each lesson must be a short imperative rule (<= 2 sentences) that would change behavior
on similar future tasks, plus a scope: "invariant" (any task) or "domain:data" / "domain:code", plus a falsifiable prediction
of what will look different in a replayed trace if the lesson is followed.
Answer ONLY a JSON list: [{{"lesson": "...", "scope": "...", "prediction": "..."}}]

TASK:
{task}

CURRENT LIBRARY:
{lib}

TRACE:
{trace}"""

PROBE_JUDGE = """Two agents solved the same task. Agent B was given an extra lesson; agent A was not.
Neither result was graded; you only see traces. Decide whether the lesson's predicted behavior change actually happened in B
and whether B's process is more trustworthy than A's (more verification of the exact required output format/values,
fewer unfounded assumptions, fewer errors/wasted calls). Do not reward length.
Answer ONLY JSON: {{"followed": true|false, "better": "A"|"B"|"tie", "why": "<= 2 sentences"}}

TASK:
{task}

LESSON: {lesson}
PREDICTION: {prediction}

=== TRACE A (without lesson) ===
{a}

=== TRACE B (with lesson) ===
{b}"""

COMPOSE = """Merge the accepted lessons into the EXPERIENCE library below. Keep it under {cap} characters, keep two sections
"## invariant" and "## domain:data" / "## domain:code" as needed, deduplicate, keep rules short and imperative, keep
every accepted lesson's meaning, drop nothing else. Output ONLY the new library markdown.

CURRENT LIBRARY:
{lib}

ACCEPTED LESSONS:
{new}"""


def lib_read(p): return open(p).read() if p and os.path.exists(p) else ""


def evolve(args):
    os.makedirs(LIBDIR, exist_ok=True)
    lib = lib_read(args.out); naive = lib_read(args.naive_out)
    log = open(os.path.join(RESULTS, "evolve.jsonl"), "a")
    for tid in T.STREAM:
        task = T.TASKS[tid][1]
        a = execute(tid, lib, tag="evolve-base")
        txt, u1 = editor(PROPOSE.format(k=args.k, task=task, lib=lib or "(empty)", trace=trace_text(a)))
        cands = [c for c in (parse_json(txt) or []) if isinstance(c, dict) and c.get("lesson")][: args.k]
        with cf.ThreadPoolExecutor(len(cands) or 1) as ex:
            probes = list(ex.map(lambda c: execute(tid, (lib + "\n" if lib else "") + f"- {c['lesson']}", tag="evolve-probe"), cands))
        accepted = []
        for c, b in zip(cands, probes):
            jt, _ = editor(PROBE_JUDGE.format(task=task, lesson=c["lesson"], prediction=c.get("prediction", ""), a=trace_text(a, 9000), b=trace_text(b, 9000)))
            j = parse_json(jt) or {}
            ok = bool(j.get("followed")) and j.get("better") == "B"
            c.update(judge=j, supported=ok, hidden_base=a["score"], hidden_probe=b["score"])
            if ok: accepted.append(c)
        if accepted:
            new = "\n".join(f"- [{c.get('scope', 'invariant')}] {c['lesson']}" for c in accepted)
            out, _ = editor(COMPOSE.format(cap=args.cap, lib=lib or "(empty)", new=new))
            lib = strip_fence(out) or lib
        naive += "".join(f"- [{c.get('scope', 'invariant')}] {c['lesson']}\n" for c in cands)  # BTM-style: accept everything proposed
        open(args.out, "w").write(lib); open(args.naive_out, "w").write(naive)
        rec = {"task": tid, "base_score": a["score"], "cands": cands, "lib_chars": len(lib), "editor_usage": u1}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
        print(f"{tid}: base={a['score']} cands={len(cands)} supported={sum(c['supported'] for c in cands)} "
              f"probe_scores={[c['hidden_probe'] for c in cands]} lib={len(lib)}ch", flush=True)


def evaluate(args):
    arms = dict(x.split("=", 1) for x in args.libs.split(","))  # name=path[@scoped]
    tids = args.tasks.split(",") if args.tasks else T.STREAM + T.HELDOUT
    jobs = [(arm, tid, rep) for rep in range(args.reps) for tid in tids for arm in arms]
    def one(j):
        arm, tid, rep = j; path, _, mode = arms[arm].partition("@")
        return arm, tid, execute(tid, lib_read(path), tag=f"eval-{arm}-{args.label}", model=args.model, scoped=mode == "scoped")
    rows = []
    with cf.ThreadPoolExecutor(args.par) as ex:
        for arm, tid, r in ex.map(one, jobs):
            rows.append({"arm": arm, "task": tid, "model": r["model"], "rep_lib_chars": r["lib_chars"], "status": r["status"], "score": r["score"], "calls": len(r["steps"]), "errors": sum(s["error"] for s in r["steps"]),
                         "tokens": r["usage"]["uncached"] + r["usage"]["cached"] + r["usage"]["completion"], "secs": r["secs"], "miss": r["miss"]})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    with open(os.path.join(RESULTS, f"eval-{args.label}.json"), "w") as f: json.dump(rows, f, indent=1)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run"); p.add_argument("task"); p.add_argument("--lib"); p.add_argument("--tag", default="manual")
    p = sp.add_parser("evolve"); p.add_argument("--k", type=int, default=2); p.add_argument("--cap", type=int, default=2500)
    p.add_argument("--out", default=os.path.join(LIBDIR, "EXPERIENCE.md")); p.add_argument("--naive-out", default=os.path.join(LIBDIR, "NAIVE.md"))
    p = sp.add_parser("eval"); p.add_argument("--libs", required=True); p.add_argument("--reps", type=int, default=2)
    p.add_argument("--tasks", default=""); p.add_argument("--model", default=None)
    p.add_argument("--par", type=int, default=4); p.add_argument("--label", default=dt.datetime.now().strftime("%Y%m%d-%H%M"))
    a = ap.parse_args(); os.makedirs(RESULTS, exist_ok=True)
    if a.cmd == "run":
        r = execute(a.task, lib_read(a.lib), tag=a.tag)
        print(json.dumps({k: r[k] for k in ("task", "score", "miss", "status", "secs", "usage")}, ensure_ascii=False), len(r["steps"]), "steps")
    elif a.cmd == "evolve": evolve(a)
    else: evaluate(a)


if __name__ == "__main__":
    main()
