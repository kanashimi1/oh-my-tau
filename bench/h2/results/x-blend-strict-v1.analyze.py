#!/usr/bin/env python3
"""Analyse results/x-blend-strict-v1.log: diff vs strict arms, bootstrap CIs, repairs.
Bootstrap = 95% CI of the per-task-mean difference (10k resamples, stratified by task),
same method as REPORT-xeval.md. Writes x-blend-strict-v1.analysis.{json,md}.
Repairs: b_table teammate edit affects checks {legs, lamp}; b_sign affects {sign}.
Refusal/ack accounting reads results/x-blend-traces/strict-v1-*.json bl traces."""
import glob, json, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [json.loads(l) for l in open(os.path.join(HERE, "x-blend-strict-v1.log")) if l.strip()]
rows = [r for r in ROWS if r.get("status") == "completed" and "checks" in r]
AFFECTED = {"b_table": ["legs", "lamp"], "b_sign": ["sign"]}

def metric(r, m):
    if m == "score": return r["score"]
    if m == "success": return 1.0 if r["success"] else 0.0
    if m == "calls": return r["calls"]
    if m == "bl_calls": return r.get("bl_calls", 0)
    if m == "tokens": return r["tokens"]
    if m == "repair":
        aff = AFFECTED[r["task"]]
        return sum(1 for k in aff if r["checks"].get(k)) / len(aff)
    raise KeyError(m)

def per_task(rs, m):
    tasks = sorted(set(r["task"] for r in rs))
    return tasks, [statistics.fmean([metric(r, m) for r in rs if r["task"] == t]) for t in tasks]

def boot_ci(rs_a, rs_b, m, n=10000, seed=7):
    """95% CI of mean-of-per-task-means difference (b - a), resampling runs within task."""
    rng = random.Random(seed)
    tasks = sorted(set(r["task"] for r in rs_a) | set(r["task"] for r in rs_b))
    by_a = {t: [metric(r, m) for r in rs_a if r["task"] == t] for t in tasks}
    by_b = {t: [metric(r, m) for r in rs_b if r["task"] == t] for t in tasks}
    diffs = []
    for _ in range(n):
        ds = []
        for t in tasks:
            if by_a[t] and by_b[t]:
                ds.append(statistics.fmean(rng.choices(by_b[t], k=len(by_b[t])))
                          - statistics.fmean(rng.choices(by_a[t], k=len(by_a[t]))))
        if ds:
            diffs.append(statistics.fmean(ds))
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]

arms = {}
for arm in ("diff", "strict"):
    rs = [r for r in rows if r["arm"] == arm]
    arms[arm] = rs
summary = {}
for arm, rs in arms.items():
    tasks, pt_score = per_task(rs, "score")
    summary[arm] = {
        "runs": len(rs),
        "success": sum(1 for r in rs if r["success"]),
        "score": round(statistics.fmean([r["score"] for r in rs]), 3),
        "per_task_score": dict(zip(tasks, [round(v, 3) for v in pt_score])),
        "calls_mean": round(statistics.fmean([metric(r, "calls") for r in rs]), 2) if rs else None,
        "bl_calls_mean": round(statistics.fmean([metric(r, "bl_calls") for r in rs]), 2) if rs else None,
        "tokens_mean_k": round(statistics.fmean([metric(r, "tokens") for r in rs]) / 1000, 1) if rs else None,
        "repair_mean": round(statistics.fmean([metric(r, "repair") for r in rs]), 3) if rs else None,
        "repair_per_task": {t: round(statistics.fmean([metric(r, "repair") for r in rs if r["task"] == t]), 3)
                            for t in sorted(set(r["task"] for r in rs))},
        "errors_mean": round(statistics.fmean([r["errors"] for r in rs]), 2) if rs else None,
    }
ci = {}
for m in ("score", "success", "calls", "bl_calls", "tokens", "repair"):
    lo, hi = boot_ci(arms["diff"], arms["strict"], m)
    pt = statistics.fmean([v for v in per_task(arms["strict"], m)[1]]) - statistics.fmean([v for v in per_task(arms["diff"], m)[1]])
    ci[m] = {"diff": round(pt, 3), "lo": round(lo, 3), "hi": round(hi, 3)}

# traces: refusals seen and acks issued by strict-arm agents
tr = {"refusals": 0, "runs_with_refusal": 0, "acks": 0, "teammate_edits": 0, "runs": 0}
for f in glob.glob(os.path.join(HERE, "x-blend-traces", "strict-v1-strict-*.json")):
    d = json.load(open(f))
    bl = d.get("bl") or []
    tr["runs"] += 1
    n_ref = sum(1 for x in bl if x.get("event") == "agent" and x.get("error_code") == "external_changes_pending")
    tr["refusals"] += n_ref
    tr["runs_with_refusal"] += 1 if n_ref else 0
    tr["acks"] += sum(1 for x in bl if x.get("event") == "agent" and x.get("expect_rev"))
    tr["teammate_edits"] += sum(1 for x in bl if x.get("event") == "teammate")
tr_refusal = {}
for f in glob.glob(os.path.join(HERE, "x-blend-traces", "strict-v1-diff-*.json")):
    d = json.load(open(f))
    tr_refusal["teammate_edits"] = tr_refusal.get("teammate_edits", 0) + sum(1 for x in (d.get("bl") or []) if x.get("event") == "teammate")

# repair conditioned on the teammate edit actually firing (fair: no-edit runs trivially "repair")
fired = {}
for arm in ("diff", "strict"):
    for f in glob.glob(os.path.join(HERE, "x-blend-traces", f"strict-v1-{arm}-*.json")):
        d = json.load(open(f))
        if any(x.get("event") == "teammate" for x in (d.get("bl") or [])):
            base = os.path.basename(f)[:-5].split("-")
            fired[(arm, base[3], int(base[4]))] = True
rep_fired = {}
for arm in ("diff", "strict"):
    rs = [r for r in rows if r["arm"] == arm and fired.get((arm, r["task"], r["rep"]))]
    rep_fired[arm] = {
        "runs_edited": len(rs),
        "repair_mean": round(statistics.fmean([metric(r, "repair") for r in rs]), 3) if rs else None,
        "fully_repaired": sum(1 for r in rs if metric(r, "repair") == 1.0),
        "success": sum(1 for r in rs if r["success"]),
    }

out = {"summary": summary, "ci_strict_minus_diff": ci, "repair_given_edit": rep_fired,
       "strict_traces": tr, "diff_traces": tr_refusal,
       "excluded_rows": len(ROWS) - len(rows)}
json.dump(out, open(os.path.join(HERE, "x-blend-strict-v1.analysis.json"), "w"), indent=1)
md = ["# x-blend strict-v1 analysis", "",
      f"rows completed: {len(rows)} (excluded {len(ROWS)-len(rows)}), arms diff/strict, tasks b_table,b_sign", "",
      "| metric | diff | strict | strict-diff [95% CI] |", "|---|---|---|---|"]
def fmt(v): return v
for key, label, cik in [("success", "success n", "success"), ("score", "score", "score"),
                   ("repair_mean", "affected checks repaired", "repair"),
                   ("calls_mean", "agent calls", "calls"), ("bl_calls_mean", "bridge calls", "bl_calls"),
                   ("tokens_mean_k", "tokens (k)", "tokens"), ("errors_mean", "tool errors", None)]:
    a, b = summary["diff"].get(key), summary["strict"].get(key)
    c = ci.get(cik, {}) if cik else {}
    if cik == "tokens":
        c = {k: round(v / 1000, 1) for k, v in c.items()}
    md.append(f"| {label} | {a} | {b} | {c.get('diff')} [{c.get('lo')}, {c.get('hi')}] |")
md += ["", f"strict traces: {tr}"]
md += ["", "per-task repair:", json.dumps({t: {a: summary[a]["repair_per_task"].get(t) for a in ("diff", "strict")}
                                          for t in sorted(set(summary["diff"]["repair_per_task"]) | set(summary["strict"]["repair_per_task"]))})]
md += ["", "repair GIVEN teammate edit fired (fair comparison):", json.dumps(rep_fired)]
open(os.path.join(HERE, "x-blend-strict-v1.analysis.md"), "w").write("\n".join(md) + "\n")
print(json.dumps(out, indent=1)[:3000])
