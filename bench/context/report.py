#!/usr/bin/env python3
"""Markdown table over bench/context/results/*/**.summary.json.

  python3 bench/context/report.py bench/context/results/<run-id> [more dirs...]

One row per cell: prompt uncached / cached / cache-write tokens, completion tokens, cache-hit %,
latency, LLM calls, accepted/rejected context_edit calls, task score. A trailing block compares
each strategy against `append` on the same (task, model, steps): cache-hit delta in percentage
points (acceptance: CLM within 5 pp of append) and the token ratio.
"""
import glob
import json
import os
import sys
from collections import defaultdict


def load(dirs):
    rows = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
            rows.append(json.load(open(p)))
    return rows


def main():
    rows = load(sys.argv[1:])
    if not rows:
        sys.exit("no summaries")
    print("| task | steps | strategy | model | rep | prompt uncached | prompt cached | cache write | completion | cache-hit % | latency s | calls | edits ok/rej | score |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["task"], r["steps"], r["model"], r["strategy"], r["rep"])):
        print(f"| {r['task']} | {r['steps']} | {r['strategy']} | {r['model']} | {r['rep']} | {r['uncached']} | "
              f"{r['cached']} | {r['cache_write']} | {r['completion']} | {r['cache_hit_pct']} | "
              f"{r['latency_s']} | {r['calls']} | {r['edits_ok']}/{r['edits_rejected']} | "
              f"{'PASS' if r['score'] else 'FAIL'} ({r['score_detail']}) |")
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["task"], r["steps"], r["model"])].setdefault(r["strategy"], []).append(r)
    print("\n| task | steps | model | strategy | cache-hit % | Δ vs append (pp) | billed-ish tokens ratio | latency ratio | within 5 pp |")
    print("|---|---|---|---|---|---|---|---|---|")
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    for (task, steps, model), by in sorted(groups.items()):
        base = by.get("append")
        for strat, rs in sorted(by.items()):
            hit = mean([r["cache_hit_pct"] for r in rs])
            if base and strat != "append":
                bh = mean([r["cache_hit_pct"] for r in base])
                tok = lambda r: r["uncached"] + r["cached"] + r["cache_write"] + r["completion"]  # noqa: E731
                print(f"| {task} | {steps} | {model} | {strat} | {hit:.1f} | {hit - bh:+.1f} | "
                      f"{mean([tok(r) for r in rs]) / mean([tok(r) for r in base]):.2f} | "
                      f"{mean([r['latency_s'] for r in rs]) / mean([r['latency_s'] for r in base]):.2f} | "
                      f"{'yes' if abs(hit - bh) <= 5 or hit >= bh else 'NO'} |")
            else:
                print(f"| {task} | {steps} | {model} | {strat} | {hit:.1f} | - | - | - | - |")


if __name__ == "__main__":
    main()
