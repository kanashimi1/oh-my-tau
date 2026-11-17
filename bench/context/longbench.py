#!/usr/bin/env python3
"""Long-session bench: where do RLM (offload + llm_query) and CLM (context_edit) pay off?

Families (episodes of 24-60 steps through `omp-rs --mode rpc`, temp HOME, root model zai/glm-5.3-flash):
  L1  big handbook (~70k chars) early, sparse lookup questions later        -> offload / llm_query
  L2  36 modules tested one per step, facts live only in tool outputs        -> clm_edit
  L3  400k-char log that never enters the prompt, classify every record     -> llm_query REQUIRED
  L4  real long session prefix (read-only copy), recall questions after it   -> either

Arms: append | clm | clm-host (legacy per-step maintenance) | offload (rule: offload every >=2000-char old block)
      | rule (offload >=8000-char blocks aged >=3 turns, edit when history >=24k tokens) | schedule/laya via ctxp env.

  longbench.py run  --family L1 --arms append,clm --seeds 0-2 --reps 2 --workers 6
  longbench_analysis.py report
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R  # noqa: E402

OUT = os.path.join(HERE, "results", "long")
MODEL = "zai/glm-5.3-flash"
RELAY = os.environ.get("LONGBENCH_RELAY", "zen-cli/space-bunny-free")
LEDGER = os.path.join(OUT, "ledger.jsonl")
LOCK = threading.Lock()

ARMS = {
    "append": dict(strategy="append"),
    "clm": dict(strategy="clm", legacy=True),
    "clm-host": dict(strategy="clm-host", legacy=True),
    "offload": dict(strategy="clm", policy="rule", rule=dict(offload=2000, age=0, restated=0, tokens=0, every=0)),
    "rule": dict(strategy="clm", policy="rule", rule=dict(offload=8000, age=3, restated=0, tokens=24000, every=0)),
    # Learned controller over the same real RPC/context pipeline; ckpt/onnx come from OMP_CONTEXT_LAYA_*.
    "laya": dict(strategy="clm", policy="laya", rule=dict(offload=8000, age=3, restated=0, tokens=24000, every=0)),
    # LCM-lite (arXiv 2605.04050): engine-managed, no model decision. Below `hist` tokens it does nothing (wire == append
    # modulo the clm slot table); above it the oldest large blocks become extractive summary + lossless file pointer.
    # L3 additionally gets the engine-run `llm_map` primitive instead of a model-written chunk loop.
    "lcm": dict(strategy="clm", policy="rule",
                rule=dict(offload=4000, age=2, restated=0, tokens=10**9, every=0, hist=16000, summary=400)),
    # lcm with the block threshold lowered to the size of an L2 tool result (lcm's 4000 never fires there)
    "lcm2": dict(strategy="clm", policy="rule",
                 rule=dict(offload=2000, age=2, restated=0, tokens=10**9, every=0, hist=16000, summary=400)),
    # TierMem T1 (runs/hybrid-ctx/DESIGN.md): cache-aligned lossless eviction — nothing happens below `high`
    # tokens; at `high` the oldest eligible blocks go in ONE turn (file + extractive summary) down to `low`,
    # so the prompt cache breaks once per cycle instead of every turn (lcm's 2.2x on L4).
    "tier": dict(strategy="clm", policy="rule",
                 rule=dict(offload=0, age=1, restated=0, tokens=10**9, every=0, summary=400, high=32000, low=16000)),
    # TierMem T2: tier + recall-on-miss — an offloaded block that the new user turn points at (quoted line,
    # file name, id) is re-inlined into that turn (appended, sticky: the cached prefix is untouched).
    "tier2": dict(strategy="clm", policy="rule",
                  rule=dict(offload=0, age=1, restated=0, tokens=10**9, every=0, summary=400, high=32000, low=16000,
                            recall=24000)),
    # L3 only: no sub-model recursion; the root reads the log through the read tool and counts itself (does RLM recursion pay?).
    "direct": dict(strategy="append"),
    # Actual ultra modes, activated through the same RPC command as the desktop selector.
    "deep": dict(strategy="append", ultra="ultradeep"),
    "long": dict(strategy="append", ultra="ultralong"),
    "hyper": dict(strategy="append", ultra="hypercode"),
    # Hypercode with its existing rule/offload knobs enabled; no new memory architecture.
    "hyper-offload": dict(strategy="append", ultra="hypercode", policy="rule",
                          rule=dict(offload=8000, age=3, restated=0, tokens=8000, every=0)),
}


# ----------------------------------------------------------------------------- generators

def _filler(rng, n):
    bits = ["Operators should follow the standard checklist before any intervention.",
            "Historical notes indicate that this unit has been stable across audits.",
            "The vendor recommends periodic inspection according to the maintenance calendar.",
            "Cross-reference the safety bulletin when changing any setpoint of this unit.",
            "Spare units are stored in the central warehouse and released on request.",
            "Documentation of earlier revisions is available in the archive on request.",
            "This section is informational and does not alter the commissioning procedure.",
            "Environmental conditions in the bay are logged hourly by the building system."]
    return " ".join(rng.choice(bits) for _ in range(n))


def gen_l1(seed, steps=48):
    rng = random.Random(f"l1-{seed}")
    kinds = ["Valve controller", "Pump station", "Relay board", "Sensor array", "Cooling loop", "Power inverter",
             "Gateway node", "Buffer tank"]
    pre = {"Valve controller": "VC", "Pump station": "PS", "Relay board": "RB", "Sensor array": "SA",
           "Cooling loop": "CL", "Power inverter": "PI", "Gateway node": "GN", "Buffer tank": "BT"}
    first = ["Ada", "Boris", "Chen", "Dara", "Emil", "Farah", "Goran", "Hana", "Ivo", "Jun", "Kira", "Liam"]
    last = ["Novak", "Ortiz", "Petrov", "Quinn", "Rossi", "Sato", "Tran", "Ueda", "Voss", "Wren", "Xu", "Yilmaz"]
    secs = []
    for i in range(1, 71):
        k = rng.choice(kinds)
        title = f"{k} {pre[k]}-{i:02d}"
        facts = {
            "maximum pressure (kPa)": str(rng.randint(100, 999)),
            "calibration code": f"{rng.choice('KXQZ')}{rng.choice('KXQZ')}-{rng.randint(1000, 9999)}",
            "installation date": f"20{rng.randint(15, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "responsible owner": f"{rng.choice(first)} {rng.choice(last)}",
            "bay": f"Bay-{rng.randint(1, 40)}",
            "spare part number": f"SP-{rng.randint(10000, 99999)}",
        }
        body = [f"SECTION {i}: {title}", _filler(rng, 3)]
        for f, v in facts.items():
            body.append(f"- {f}: {v}")
            body.append(_filler(rng, 1))
        body.append(_filler(rng, 2))
        secs.append((title, facts, "\n".join(body)))
    doc = "\n\n".join(s[2] for s in secs)
    qsteps = [round(6 + j * (steps - 12) / 4) for j in range(5)]  # 5 sparse questions
    picks = rng.sample(range(70), 8)
    qs = []
    for j, idx in enumerate(picks):
        title, facts, _ = secs[idx]
        f = rng.choice(list(facts))
        qs.append((title, f, facts[f]))

    def qtext(n):
        t, f, _ = qs[n - 1]
        return f"QUESTION Q{n}: in the handbook section titled '{t}', what is the {f}? Reply on one line as `ANSWER Q{n}: <value>`."

    msgs = {}
    for k in range(1, steps):
        body = f"STEP {k}/{steps}: continue; reply with a one-line progress note [step {k}]."
        if k == 1:
            body = ("HANDBOOK (the only source of truth, ~%d chars):\n<<<DOC\n%s\nDOC>>>\n\n" % (len(doc), doc)) + body
        msgs[k] = body
    for j, k in enumerate(qsteps):
        msgs[k] = f"STEP {k}/{steps}: " + qtext(j + 1)
    msgs[steps] = (f"FINAL STEP {steps}/{steps}: answer all of these, one per line as `ANSWER Qn: <value>`: "
                   + " ".join(qtext(n) for n in (6, 7, 8))
                   + " Also re-answer Q1 and Q2 exactly as `ANSWER Q1: <value>` and `ANSWER Q2: <value>`.")
    final_qs = [6, 7, 8, 1, 2]
    asked = {qsteps[j]: [j + 1] for j in range(5)}
    asked[steps] = final_qs

    def scorer(replies):
        ok, detail = 0, []
        for step, ns in asked.items():
            r = replies.get(step, "")
            for n in ns:
                m = re.findall(rf"ANSWER Q{n}:\s*(.+)", r)
                want = qs[n - 1][2]
                hit = bool(m) and want.lower() in m[-1].lower()
                ok += hit
                detail.append(f"Q{n}{'+' if hit else '-'}")
        total = sum(len(v) for v in asked.values())
        return {"score": ok / total, "pass": ok == total, "detail": " ".join(detail), "valid": True}

    def probe(d):
        ns = (6, 7, 8)
        text = ("PROBE: answer all of these, one per line as `ANSWER Qn: <value>`: " + " ".join(qtext(n) for n in ns))

        def sc(reply):
            ok = 0
            for n in ns:
                m = re.findall(rf"ANSWER Q{n}:\s*(.+)", reply)
                ok += bool(m) and qs[n - 1][2].lower() in m[-1].lower()
            return ok / len(ns)
        return text, sc

    return dict(files={}, msgs=msgs, steps=steps, scorer=scorer, sub="relay", timeout=600, probe=probe,
                brief=("You are a careful agent working over a long HANDBOOK that arrives in the next message, with tools "
                       "available (bash, read, eval). Follow each STEP exactly. Never invent data that is not in the "
                       "handbook. Reply with exactly: ACK"), meta=dict(doc_chars=len(doc)))


def gen_l2(seed, steps=40):
    rng = random.Random(f"l2-{seed}")
    m = steps - 4
    bad = sorted(rng.sample(range(1, m + 1), 4))
    files, truth = {}, {}
    names = ["converter", "scaler", "mapper", "encoder", "packer", "router", "filter", "mixer"]
    for k in range(1, m + 1):
        spec = rng.randint(11, 97)
        off = rng.randint(1, 9)
        rate = spec
        if k in bad:
            rate = spec + rng.choice([-3, -2, 2, 3, 7])
            truth[f"mod_{k:02d}"] = spec
        helpers = "\n".join(
            f"def helper_{j}(x):\n    \"\"\"Helper {j} of the {rng.choice(names)} pipeline.\"\"\"\n    return (x * {rng.randint(2, 9)} + {rng.randint(1, 99)}) % {rng.randint(101, 997)}\n"
            for j in range(7))
        files[f"proj/mod_{k:02d}.py"] = (
            f'"""Module {k:02d}: {rng.choice(names)} stage."""\n# SPEC: RATE must equal {spec}\nRATE = {rate}\nOFFSET = {off}\n\n'
            f"def compute(x):\n    return x * RATE + OFFSET\n\n{helpers}")
        files[f"proj/expected_{k:02d}.json"] = json.dumps({"input": 10, "expected": 10 * spec + off})
    files["run_tests.py"] = '''import importlib, json, random, sys, time
k = int(sys.argv[1])
sys.path.insert(0, "proj")
mod = importlib.import_module(f"mod_{k:02d}")
exp = json.load(open(f"proj/expected_{k:02d}.json"))
rnd = random.Random(k)
for i in range(34):
    print(f"[debug] fixture {i:02d} loaded in {rnd.randint(2, 90)}ms; deprecation: legacy path /opt/lib{rnd.randint(1, 9)}/compat.py:{rnd.randint(10, 400)}")
got = mod.compute(exp["input"])
print(f"test_compute (mod_{k:02d}.ModuleTest) ... " + ("ok" if got == exp["expected"] else "FAIL"))
if got != exp["expected"]:
    print(f"AssertionError: {got} != {exp['expected']}")
    print("Ran 1 test in 0.00" + str(rnd.randint(1, 9)) + "s\\n\\nFAILED (failures=1)")
else:
    print("Ran 1 test in 0.00" + str(rnd.randint(1, 9)) + "s\\n\\nOK")
'''
    msgs = {}
    for k in range(1, steps):
        if k <= m:
            msgs[k] = (f"STEP {k}/{steps}: run `python3 run_tests.py {k}` in the work directory with the bash tool and read "
                       f"`proj/mod_{k:02d}.py` with the read tool. Do not write findings in your reply; reply with exactly: noted")
        else:
            msgs[k] = f"STEP {k}/{steps}: nothing to run this step; reply with exactly: noted"
    msgs[steps] = (f"FINAL STEP {steps}/{steps}: list every module whose test FAILED together with the RATE its SPEC comment "
                   "requires. Output one line per failing module exactly as `FAIL mod_XX RATE=<spec value>` and nothing else.")

    def scorer(replies):
        r = replies.get(steps, "")
        got = {mm.group(1): int(mm.group(2)) for mm in re.finditer(r"FAIL\s+(mod_\d+)\s+RATE\s*=\s*(\d+)", r)}
        tp = sum(1 for k, v in got.items() if truth.get(k) == v)
        fp = len(got) - tp
        return {"score": tp / (len(truth) + fp), "pass": tp == len(truth) and fp == 0,
                "detail": f"tp={tp}/{len(truth)} fp={fp}", "valid": True}

    def probe(d):
        tested = min(d, m)
        want = {k: v for k, v in truth.items() if int(k.split("_")[1]) <= tested}
        text = ("PROBE: list every module among those you tested in steps 1..%d whose test FAILED together with the RATE its SPEC "
                "comment requires. One line per failing module exactly as `FAIL mod_XX RATE=<spec value>` and nothing else "
                "(write NONE if there is none)." % tested)

        def sc(reply):
            got = {mm.group(1): int(mm.group(2)) for mm in re.finditer(r"FAIL\s+(mod_\d+)\s+RATE\s*=\s*(\d+)", reply)}
            tp = sum(1 for k, v in got.items() if want.get(k) == v)
            fp = len(got) - tp
            return 1.0 if not want and not got else tp / max(1, len(want) + fp)
        return text, sc

    return dict(files=files, msgs=msgs, steps=steps, scorer=scorer, sub="relay", timeout=600, probe=probe,
                brief=("You are an agent in a sandbox work directory with tools (bash, read, eval). Follow each STEP "
                       "exactly. A final question about all the steps will come at the end. Reply with exactly: ACK"),
                meta=dict(bad=list(truth)))


CATS = ["auth_failure", "disk_full", "timeout", "config_change", "benign"]


def gen_l3(seed, steps=24):
    rng = random.Random(f"l3-{seed}")
    users = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
    svcs = ["billing-api", "search", "ledger", "auth-proxy", "mailer", "scheduler", "exports"]
    keys = ["cache.ttl", "retry.max", "pool.size", "log.level", "flag.beta_ui", "batch.size"]
    T = {
        "auth_failure": ["login rejected for user {u}: bad credentials", "authentication token expired for {u}",
                         "too many failed sign-in attempts from 10.{a}.{b}.{c}", "SSO assertion invalid for account {u}"],
        "disk_full": ["no space left on device /var/data{d}", "volume data{d} reached 100% capacity, writes are failing",
                      "ENOSPC while flushing journal on data{d}", "quota exceeded on storage pool data{d}"],
        "timeout": ["upstream {s} did not answer within {t}s", "request to {s} timed out after {t} seconds",
                    "deadline exceeded calling {s}", "gateway gave up waiting on {s} ({t}s)"],
        "config_change": ["setting {k} changed from {x} to {y} by {u}", "operator {u} updated {k}={y}",
                          "reloaded configuration, {k} is now {y}", "feature switch {k} flipped to {y}"],
        "benign": ["health check passed for {s}", "heartbeat from {s} ok, latency {t}ms", "cache warm complete, {n} entries",
                   "scheduled backup finished without errors", "login ok for user {u}",
                   "timeout setting verified at {t}s, no action needed", "disk usage {p}% on data{d}, within limits",
                   "config checksum unchanged for {k}"],
    }
    weights = [rng.uniform(0.6, 1.6) for _ in CATS[:-1]] + [3.0]
    n = 4500
    lines, truth = [], {c: 0 for c in CATS}
    for i in range(n):
        c = rng.choices(CATS, weights)[0]
        truth[c] += 1
        msg = rng.choice(T[c]).format(u=rng.choice(users), s=rng.choice(svcs), k=rng.choice(keys), x=rng.randint(1, 50),
                                      y=rng.randint(51, 99), t=rng.randint(2, 90), d=rng.randint(1, 4),
                                      n=rng.randint(100, 9999), p=rng.randint(20, 79), a=rng.randint(0, 255),
                                      b=rng.randint(0, 255), c=rng.randint(0, 255))
        lines.append(f"[{i:05d}] 2026-10-{rng.randint(1, 9):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z "
                     f"host-{rng.randint(1, 40):02d} {rng.choice(svcs)}: {msg}")
    corpus = "\n".join(lines)
    task = ("The file `corpus/events.log` (about %d characters, %d records, one per line) is far too large to read into your "
            "context. Count how many records fall into each category: auth_failure, disk_full, timeout, config_change, "
            "benign (benign = routine messages, including ones that merely mention logins, timeouts, disk or config without a "
            "problem or a change). You MUST classify the records with `llm_query(prompt, context=chunk)` inside an eval cell "
            "(python), over chunks of at most 150 records, asking for per-category counts; do not write keyword rules and do not "
            "print the file. Keep the aggregated counts in a kernel variable `counts`. After you have the totals reply with one line "
            "`TOTALS auth_failure=<n> disk_full=<n> timeout=<n> config_change=<n> benign=<n>`." % (len(corpus), n))
    msgs = {1: "STEP 1/%d: " % steps + task}
    for k in range(2, steps):
        qc = CATS[k % 5]
        msgs[k] = (f"STEP {k}/{steps}: from the counts you already computed (do not re-run the classification), reply with one "
                   f"line: `COUNT {qc}=<n>`.")
    msgs[steps] = (f"FINAL STEP {steps}/{steps}: reply with exactly one line "
                   "`TOTALS auth_failure=<n> disk_full=<n> timeout=<n> config_change=<n> benign=<n>` for the whole log.")

    def scorer(replies):
        r = replies.get(steps, "")
        got = {m.group(1): int(m.group(2)) for m in re.finditer(r"(auth_failure|disk_full|timeout|config_change|benign)\s*=\s*(\d+)", r)}
        ok = [c for c in CATS if c in got and abs(got[c] - truth[c]) <= max(5, 0.08 * truth[c])]
        return {"score": len(ok) / len(CATS), "pass": len(ok) == len(CATS),
                "detail": "got=%s truth=%s" % (json.dumps(got), json.dumps(truth)), "valid": True}

    return dict(files={"corpus/events.log": corpus}, msgs=msgs, steps=steps, scorer=scorer, sub="zai", timeout=3600,
                brief=("You are an agent in a sandbox work directory with tools (bash, read, eval). The python eval kernel "
                       "provides `llm_query(prompt, context=None)` for delegating work to a sub-model. Follow each STEP "
                       "exactly. Reply with exactly: ACK"), meta=dict(truth=truth, chars=len(corpus)))


def gen_l5(seed, steps=22):
    """Mixed long-context episode: L1 sparse recall interleaved with L2 tool evidence."""
    if steps != 22:
        raise ValueError("L5 has a fixed 22-step interleaved schedule")
    handbook = gen_l1(seed, 22)
    modules = gen_l2(seed, 12)  # eight real modules, four failing
    msgs = dict(handbook["msgs"])
    # Keep all five scheduled handbook questions; replace eight progress-only turns.
    module_steps = [2, 3, 4, 5, 7, 9, 10, 12]
    for k, step in enumerate(module_steps, 1):
        msgs[step] = (f"STEP {step}/22: run `python3 run_tests.py {k}` in the work directory with the bash tool and read "
                      f"`proj/mod_{k:02d}.py` with the read tool. Do not write findings in your reply; reply with exactly: noted")
    msgs[22] = handbook["msgs"][22] + "\n\n" + modules["msgs"][12]

    def scorer(replies):
        h = handbook["scorer"](replies)
        m = modules["scorer"]({12: replies.get(22, "")})
        return {"score": (h["score"] + m["score"]) / 2,
                "pass": h["pass"] and m["pass"],
                "detail": f"handbook[{h['detail']}] modules[{m['detail']}]", "valid": True}

    return dict(files=modules["files"], msgs=msgs, steps=22, scorer=scorer, sub="relay", timeout=600,
                brief=("You are an agent in a sandbox work directory with tools (bash, read, eval). A long HANDBOOK arrives "
                       "in the next message; later steps interleave sparse handbook questions with module tests. Preserve both "
                       "handbook facts and test findings in the provided context/RLM mechanisms. Do not create or modify files, "
                       "and never repair modules or tests. Never invent missing data. Follow each STEP exactly. Reply with exactly: ACK"),
                meta=dict(doc_chars=handbook["meta"]["doc_chars"], bad=modules["meta"]["bad"], mixed=True))


GENS = {"L1": gen_l1, "L2": gen_l2, "L3": gen_l3, "L5": gen_l5}
DEFAULT_STEPS = {"L1": 48, "L2": 40, "L3": 24, "L4": 14, "L5": 22}


def get_episode(family, seed, steps=None):
    if family == "L4":
        import l4
        return l4.gen_l4(seed, steps or DEFAULT_STEPS["L4"])
    return GENS[family](seed, steps or DEFAULT_STEPS[family])


# ----------------------------------------------------------------------------- runner

def iso_epoch(ts):
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def cost_of(msgs):
    return sum(((m.get("usage") or {}).get("cost") or {}).get("total") or 0 for m in msgs)


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l)["cost"] for l in open(LEDGER) if l.strip())


def run_rollout(name, family, seed, arm, rep, steps=None, force=False, max_spend=15.0, deadline=None):
    out_path = os.path.join(OUT, name + ".json")
    if os.path.exists(out_path) and not force:
        return json.load(open(out_path))
    if deadline and time.time() > deadline:
        raise RuntimeError("deadline reached")
    if spent() > max_spend:
        raise RuntimeError("spend cap reached")
    ep = get_episode(family, seed, steps)
    steps = ep["steps"]
    cfg = ARMS[arm]
    if arm == "lcm" and family == "L3":
        ep["msgs"][1] = ep["msgs"][1].replace(
            "You MUST classify the records with `llm_query(prompt, context=chunk)` inside an eval cell "
            "(python), over chunks of at most 150 records, asking for per-category counts;",
            "You MUST classify the records with the engine primitive `llm_map(prompt, text, chunk_lines=150)` inside an eval "
            "cell (python; it runs `llm_query(prompt, context=chunk)` over every chunk and returns the list of replies; "
            "ask each chunk for per-category counts) and sum the replies;")
        assert "llm_map" in ep["msgs"][1]
    if arm == "direct":
        assert family == "L3"
        a = ep["msgs"][1].index("You MUST classify")
        b = ep["msgs"][1].index("Keep the aggregated counts")
        ep["msgs"][1] = (ep["msgs"][1][:a] + "Read the whole file with the read tool (several reads, no code) and count the records of each "
                         "category yourself; do not write keyword rules or scripts. " + ep["msgs"][1][b:])
    home = tempfile.mkdtemp(prefix="omp-long-")
    try:
        agent_dir, env = R.make_home(home)
        sub_model = RELAY if ep["sub"] == "relay" else MODEL
        with open(os.path.join(agent_dir, "config.yml"), "a") as f:
            f.write(f"modelRoles:\n  smol: {sub_model}\n")
        work = os.path.join(home, "work")
        os.makedirs(work)
        for rel, content in ep["files"].items():
            p = os.path.join(work, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(content)
        source_hashes = ({rel: hashlib.sha256(open(os.path.join(work, rel), "rb").read()).hexdigest() for rel in ep["files"]}
                         if family == "L5" else {})
        plog = os.path.join(home, "policy.jsonl")
        env["OMP_CONTEXT_OFFLOAD_DIR"] = os.path.join(work, ".offload")
        env["OMP_CONTEXT_OFFLOAD_MODEL"] = "smol"
        env["OMP_RLM_MAX_CALLS"] = "80"
        if cfg.get("policy"):
            env["OMP_CONTEXT_POLICY"] = cfg["policy"]
            env["OMP_CONTEXT_POLICY_LOG"] = plog
            r = cfg["rule"]
            env["OMP_CONTEXT_POLICY_RULE_TOKENS"] = str(r["tokens"])
            env["OMP_CONTEXT_POLICY_RULE_EVERY"] = str(r["every"])
            env["OMP_CONTEXT_POLICY_RULE_OFFLOAD"] = str(r["offload"])
            env["OMP_CONTEXT_POLICY_RULE_AGE"] = str(r["age"])
            env["OMP_CONTEXT_POLICY_RULE_RESTATED"] = str(r["restated"])
            if r.get("hist"):
                env["OMP_CONTEXT_POLICY_RULE_OFFLOAD_HIST"] = str(r["hist"])
            if r.get("summary"):
                env["OMP_CONTEXT_OFFLOAD_SUMMARY"] = str(r["summary"])
            if r.get("recall"):
                env["OMP_CONTEXT_POLICY_RECALL"] = str(r["recall"])
            if r.get("high"):
                env["OMP_CONTEXT_POLICY_RULE_HIGH"] = str(r["high"])
                env["OMP_CONTEXT_POLICY_RULE_LOW"] = str(r["low"])
        cmd = [R.DEFAULT_BIN, "--mode", "rpc", "--model", MODEL, "--thinking", os.environ.get("LONGBENCH_THINKING", "minimal"), "--tools", "bash,read,eval",
               "--no-extensions", "--no-skills", "--no-rules", "--no-title", "--no-lsp", "--no-pty",
               "--context-strategy", cfg["strategy"]]
        first_step = 0
        if ep.get("resume"):
            first_step = 1
        rpc = R.Rpc(cmd, env, work)
        if cfg.get("ultra"):
            rid = rpc.next_id
            rpc.next_id += 1
            rpc.p.stdin.write(json.dumps({"id": rid, "type": "set_thinking_level", "level": cfg["ultra"]}) + "\n")
            rpc.p.stdin.flush()
            while True:
                f = rpc.frame(60)
                if f.get("type") == "response" and f.get("id") == rid:
                    if not f.get("success"):
                        raise RuntimeError(f"set ultra failed: {f}")
                    break
            rid = rpc.next_id
            rpc.next_id += 1
            rpc.p.stdin.write(json.dumps({"id": rid, "type": "get_state"}) + "\n")
            rpc.p.stdin.flush()
            while True:
                f = rpc.frame(60)
                if f.get("type") == "response" and f.get("id") == rid:
                    state = f.get("data") or {}
                    if state.get("ultra") != cfg["ultra"]:
                        raise RuntimeError(f"ultra activation mismatch: {f}")
                    break
        if ep.get("resume"):
            rid = rpc.next_id
            rpc.next_id += 1
            rpc.p.stdin.write(json.dumps({"id": rid, "type": "switch_session", "sessionPath": ep["resume"]}) + "\n")
            rpc.p.stdin.flush()
            while True:
                f = rpc.frame(300)
                if f.get("type") == "response" and f.get("id") == rid:
                    if not f.get("success"):
                        raise RuntimeError(f"switch_session failed: {f}")
                    break
        recs, replies, windows = [], {}, []
        t_cell = time.time()
        try:
            for step in range(first_step, steps + 1):
                final = step == steps
                if step == 0:
                    msg = ep["brief"]
                else:
                    msg = ep["msgs"][step]
                    if cfg.get("legacy") and not final:
                        msg += R.CLM_MAINT
                t0 = time.time()
                status, msgs, tools = rpc.prompt(msg, ep["timeout"])
                t1 = time.time()
                windows.append((step, t0, t1))
                if status != "completed" or not msgs:
                    raise RuntimeError(f"step {step}: status={status} err={(msgs[-1].get('errorMessage') if msgs else None)}")
                reply = R.text_of(msgs[-1])
                replies[step] = reply
                u = R.usage_of(msgs)
                tn = {}
                for t in tools:
                    tn[t["name"]] = tn.get(t["name"], 0) + 1
                recs.append({"step": step, "latency": round(t1 - t0, 2), **u, "cost": cost_of(msgs),
                             "edits_ok": sum(1 for t in tools if t["name"] == "context_edit" and not t["error"]),
                             "edits_rej": sum(1 for t in tools if t["name"] == "context_edit" and t["error"]),
                             "tools": tn, "reply": reply[:3000 if final else 160]})
        finally:
            rpc.close()
        for root, _, files in os.walk(home):
            for fn in files:
                if fn.endswith(".jsonl") and "sessions" in root and fn != "eval-journal.jsonl":
                    for line in open(os.path.join(root, fn)):
                        if '"model_usage"' not in line:
                            continue
                        d = json.loads(line)
                        if d.get("type") == "model_usage" and d.get("purpose") == "rlm":
                            ts = iso_epoch(d["timestamp"])
                            u = d.get("usage") or {}
                            for rec, (st, a, b) in zip(recs, windows):
                                if a - 1 <= ts <= b + 1:
                                    rec["sub_calls"] = rec.get("sub_calls", 0) + 1
                                    rec["sub_tokens"] = rec.get("sub_tokens", 0) + (u.get("input") or 0) + (u.get("output") or 0)
                                    rec["sub_cost"] = rec.get("sub_cost", 0.0) + ((u.get("cost") or {}).get("total") or 0)
                                    rec["sub_model"] = d.get("model")
                                    break
        states = {}
        if os.path.exists(plog):
            for line in open(plog):
                d = json.loads(line)
                states[d["step"]] = d
        for rec in recs:
            d = states.get(rec["step"])
            rec["policy"] = {"action": d["action"], "state": d["state"]} if d else None
        verdict = ep["scorer"](replies)
        source_mutations = [rel for rel, digest in source_hashes.items()
                            if not os.path.isfile(os.path.join(work, rel))
                            or hashlib.sha256(open(os.path.join(work, rel), "rb").read()).hexdigest() != digest]
        if source_mutations:
            verdict["valid"] = False
            verdict["detail"] += f" source_mutations={source_mutations}"
        tot = {k: sum(r.get(k, 0) for r in recs) for k in
               ("uncached", "cached", "cache_write", "completion", "calls", "cost", "edits_ok", "edits_rej",
                "sub_calls", "sub_tokens", "sub_cost")}
        tot["tool_calls"] = sum(sum(r["tools"].values()) for r in recs)
        denom = tot["uncached"] + tot["cached"] + tot["cache_write"]
        res = {"name": name, "family": family, "seed": seed, "arm": arm, "rep": rep, "steps": steps, "model": MODEL,
               "sub_model": sub_model, "totals": tot, "cache_hit_pct": round(100 * tot["cached"] / denom, 1) if denom else 0.0,
               "latency_s": round(time.time() - t_cell, 1), "score": verdict["score"], "pass": verdict["pass"],
               "valid": verdict["valid"] and (family != "L3" or arm == "direct" or tot["sub_calls"] > 0), "score_detail": verdict["detail"],
               "meta": ep.get("meta"), "source_mutations": source_mutations, "records": recs}
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
        with LOCK:
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"name": name, "cost": tot["cost"], "t": time.time()}) + "\n")
        return res
    finally:
        shutil.rmtree(home, ignore_errors=True)


def safe(*a, **k):
    err = ""
    for attempt in range(2):
        try:
            return run_rollout(*a, **k)
        except Exception as e:  # noqa: BLE001
            err = str(e)[:300]
            print("ERR", a[0], err, flush=True)
            if "cap reached" in err or "deadline" in err:
                break
    return {"name": a[0], "error": err}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--arms", default="append,clm,clm-host,offload,rule")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-spend", type=float, default=15.0)
    ap.add_argument("--deadline-epoch", type=float, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    jobs = []
    for rep in range(1, a.reps + 1):
        for s in parse_seeds(a.seeds):
            for arm in a.arms.split(","):
                jobs.append((f"{a.tag}{a.family}-s{s}-{arm}-r{rep}", a.family, s, arm, rep))
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(safe, n, f, s, arm, rep, a.steps or None, False, a.max_spend, a.deadline_epoch or None)
                for n, f, s, arm, rep in jobs]
        for f in cf.as_completed(futs):
            x = f.result()
            if "error" in x:
                print(x["name"], "ERR", x["error"], flush=True)
            else:
                t = x["totals"]
                print(f"{x['name']} score={x['score']:.2f} pass={x['pass']} valid={x['valid']} cost={t['cost']:.4f} "
                      f"hit={x['cache_hit_pct']} lat={x['latency_s']} edits={t['edits_ok']} sub={t['sub_calls']} "
                      f"tools={t['tool_calls']} | {x['score_detail'][:90]}", flush=True)


if __name__ == "__main__":
    main()
