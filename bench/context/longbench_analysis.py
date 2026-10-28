#!/usr/bin/env python3
"""Tables for the long-session bench.

  longbench_analysis.py report [--prefix ""]      per family x arm table, gate, cumulative-cost curves
  longbench_analysis.py spend                      ledger total
"""
import glob
import json
import os
import re
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "long")
P_IN = 0.15e-6
ARM_ORDER = ["append", "clm", "clm-host", "offload", "rule", "lcm", "lcm2", "tier", "direct", "laya", "oracle"]


def teq(c):
    return c / P_IN


def load(prefix=""):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, prefix + "L*-s*-r*.json"))):
        d = json.load(open(p))
        if "records" in d:
            rows.append(d)
    return rows


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def offloads(r):
    return sum(1 for x in r["records"] if (x.get("policy") or {}).get("action") == "rlm_offload")


def curves(rs, step_cap):
    """mean cumulative cost (teq) per step index over rollouts"""
    steps = max(len(r["records"]) for r in rs)
    cum = []
    for r in rs:
        c, run = [], 0.0
        for x in r["records"]:
            run += x["cost"]
            c.append(teq(run))
        c += [c[-1]] * (steps - len(c))
        cum.append(c)
    return [mean([c[i] for c in cum]) for i in range(steps)]


def crossing(arm_curve, base_curve):
    """first step from which the arm's cumulative cost stays <= append's (None if never)"""
    n = min(len(arm_curve), len(base_curve))
    for i in range(n):
        if all(arm_curve[j] <= base_curve[j] for j in range(i, n)):
            return i
    return None


def report(prefix=""):
    rows = [r for r in load(prefix)]
    fams = sorted({r["family"] for r in rows})
    for fam in fams:
        fr = [r for r in rows if r["family"] == fam]
        by = {}
        for r in fr:
            by.setdefault(r["arm"], []).append(r)
        base = by.get("append", [])
        print(f"\n### {fam}  (steps {fr[0]['steps']}, sub-call model {fr[0]['sub_model']}, rollouts {len(fr)})\n")
        print("| arm | n | valid | billed-eq tokens | vs append | cache-hit % | latency s | vs append | quality (mean) | Δ quality pp | pass | "
              "edits | offloads | llm_query | sub tokens | tool calls | root $ |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        bt = mean([teq(r["totals"]["cost"]) for r in base]) if base else None
        bl = mean([r["latency_s"] for r in base]) if base else None
        bq = mean([r["score"] for r in base]) if base else None
        for arm in ARM_ORDER:
            rs = by.get(arm)
            if not rs:
                continue
            t = mean([teq(r["totals"]["cost"]) for r in rs])
            l = mean([r["latency_s"] for r in rs])
            q = mean([r["score"] for r in rs])
            print(f"| {arm} | {len(rs)} | {sum(r['valid'] for r in rs)}/{len(rs)} | {t:.0f} | "
                  f"{(t / bt) if bt else float('nan'):.2f}x | {mean([r['cache_hit_pct'] for r in rs]):.1f} | {l:.0f} | "
                  f"{(l / bl) if bl else float('nan'):.2f}x | {q:.3f} | {100 * (q - bq) if bq is not None else float('nan'):+.1f} | "
                  f"{sum(r['pass'] for r in rs)}/{len(rs)} | {sum(r['totals']['edits_ok'] for r in rs)} | "
                  f"{sum(offloads(r) for r in rs)} | {sum(r['totals']['sub_calls'] for r in rs)} | "
                  f"{sum(r['totals']['sub_tokens'] for r in rs)} | {sum(r['totals']['tool_calls'] for r in rs)} | "
                  f"{sum(r['totals']['cost'] for r in rs):.3f} |")
        if base:
            print("\ngate (tokens <= 0.80x, quality drop <= 1 pp, latency <= 1.3x):")
            for arm in ARM_ORDER[1:]:
                rs = by.get(arm)
                if not rs:
                    continue
                t = mean([teq(r["totals"]["cost"]) for r in rs]) / bt
                q = 100 * (mean([r["score"] for r in rs]) - bq)
                l = mean([r["latency_s"] for r in rs]) / bl
                ok = t <= 0.80 and q >= -1 and l <= 1.3
                print(f"- {arm}: tokens {t:.2f}x, quality {q:+.1f} pp, latency {l:.2f}x -> {'PASS' if ok else 'fail'}")
            print("\ncumulative billed-eq tokens (mean over rollouts) at steps, and crossing step vs append:")
            bc = curves(base, None)
            n = len(bc)
            marks = sorted({max(0, n * k // 8 - 1) for k in range(1, 9)})
            print("| arm | " + " | ".join(f"step {m}" for m in marks) + " | crosses append at |")
            print("|---|" + "---|" * (len(marks) + 1))
            for arm in ARM_ORDER:
                rs = by.get(arm)
                if not rs:
                    continue
                c = curves(rs, None)
                cr = crossing(c, bc) if arm != "append" else "-"
                print(f"| {arm} | " + " | ".join(f"{c[min(m, len(c) - 1)]:.0f}" for m in marks) + f" | {cr} |")


def spend():
    p = os.path.join(OUT, "ledger.jsonl")
    tot = sum(json.loads(l)["cost"] for l in open(p) if l.strip())
    n = sum(1 for l in open(p) if l.strip())
    print(f"rollouts {n} root spend ${tot:.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        pre = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else ""
        report(pre)
    elif cmd == "spend":
        spend()
