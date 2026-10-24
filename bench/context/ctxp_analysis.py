#!/usr/bin/env python3
"""Analysis for the context-policy experiment (see ctxp.py).

  ctxp_analysis.py labels            # counterfactual table -> results/ctxp/labels.json (+ summary)
  ctxp_analysis.py tune              # rule X/N on train sources (lookup over the counterfactual table)
  ctxp_analysis.py data              # ctx_policy family + trainer files in $CTXP_LAYA_DIR (default results/ctxp/laya-ctx-policy)
  ctxp_analysis.py oracle-scripts    # oracle schedules for the held-out sources
  ctxp_analysis.py report            # policy comparison table over ev-* rollouts (held-out sources)
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ctxp import DECISION_POINTS, NAMES, OUT, STEPS, WINDOW, heldout, source_id  # noqa: E402

P_IN = 0.15e-6  # USD per uncached prompt token of zai/glm-5.3-flash (cost.input / usage.input)
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LAYA_DIR = os.environ.get("CTXP_LAYA_DIR", os.path.join(OUT, "laya-ctx-policy"))
MARGIN = 0.03  # a non-noop action must be >= 3% cheaper in the window than noop to win the label


def load(prefix):
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT, prefix + "*.json"))):
        if os.path.basename(p) in ("labels.json", "rule.json"):
            continue
        d = json.load(open(p))
        if "records" in d:
            out[d["name"]] = d
    return out


def pol_of(name):
    return re.match(r"ev-(.+?)-t\d-mid", name).group(1).split(".")[0]


def window(res, d, key="cost"):
    return sum(r[key] for r in res["records"] if d <= r["step"] <= min(d + WINDOW, STEPS))


def teq(cost):
    return cost / P_IN


def cf_table():
    cfs = load("cf-")
    sources = sorted({r["source"] for r in cfs.values()})
    rows = []
    for src in sources:
        base = cfs.get(f"cf-{src}-base")
        if not base:
            continue
        for d in DECISION_POINTS:
            acts = {"noop": base}
            for a in ("clm_edit", "rlm_offload"):
                r = cfs.get(f"cf-{src}-d{d}-{a}")
                if r:
                    acts[a] = r
            if len(acts) < 3:
                continue
            st = base["records"][d].get("policy")
            if not st:
                continue
            w = {a: window(r, d) for a, r in acts.items()}
            lat = {a: window(r, d, "latency") for a, r in acts.items()}
            tot = {a: r["totals"]["cost"] for a, r in acts.items()}
            sc = {a: r["score"] for a, r in acts.items()}
            eligible = [a for a in acts if a == "noop" or sc[a] >= sc["noop"]]
            best = min(eligible, key=lambda a: w[a])
            label = best if best == "noop" or w[best] <= (1 - MARGIN) * w["noop"] else "noop"
            rows.append({"src": src, "task": base["task"], "dp": d, "state": st["state"], "label": label,
                         "window_cost": w, "window_latency": lat, "total_cost": tot, "score": sc})
    return rows


def parse_state(text):
    g = lambda k: re.search(rf"{k}: (\S+)", text).group(1)  # noqa: E731
    since = g("steps_since_last_edit")
    return {"step": int(g("step")), "hist": int(g("history_tokens")), "block": int(g("largest_old_block_chars")),
            "since": None if since == "none" else int(since)}


def cmd_labels():
    rows = cf_table()
    json.dump(rows, open(os.path.join(OUT, "labels.json"), "w"), indent=1)
    print(f"{len(rows)} decision points from {len({r['src'] for r in rows})} sources")
    from collections import Counter
    for split, sel in (("train", lambda s: not heldout(s)), ("heldout", heldout)):
        rs = [r for r in rows if sel(r["src"])]
        print(split, len(rs), dict(Counter(r["label"] for r in rs)))
    for a in ("noop", "clm_edit", "rlm_offload"):
        c = sum(teq(r["window_cost"][a]) for r in rows)
        l = sum(r["window_latency"][a] for r in rows)
        s = sum(r["score"][a] for r in rows)
        print(f"{a:12s} window billed-eq tokens {c:9.0f}  window latency {l:7.1f}s  final-pass {s}/{len(rows)}")


def rule_decision(st, tokens, every, offload=0):
    if offload and st["block"] >= max(offload, 2000):
        return "rlm_offload"
    big = tokens > 0 and st["hist"] >= tokens
    due = every > 0 and (st["since"] if st["since"] is not None else st["step"] + 1) >= every
    return "clm_edit" if big or due else "noop"


def cmd_tune():
    rows = [r for r in cf_table() if not heldout(r["src"])]
    grid = []
    for offload in (0, 2000, 6000):
        for tokens in (0, 2500, 4500, 6500, 8000, 15000):
            for every in (0, 2, 4, 6):
                cost = loss = agree = 0
                acts = {"noop": 0, "clm_edit": 0, "rlm_offload": 0}
                for r in rows:
                    a = rule_decision(parse_state(r["state"]), tokens, every, offload)
                    cost += r["window_cost"][a]
                    acts[a] += 1
                    loss += int(r["score"][a] < r["score"]["noop"])
                    agree += a == r["label"]
                grid.append({"offload": offload, "tokens": tokens, "every": every, "window_teq": round(teq(cost)),
                             "score_losses": loss, "label_agree": agree, **acts})
    # same target Laya is trained on: agreement with the oracle label, ties -> cheaper window
    pick = lambda pool: min(pool, key=lambda g: (-g["label_agree"], g["window_teq"]))  # noqa: E731
    best = pick(grid)
    best_edit = pick([g for g in grid if g["offload"] == 0 and (g["tokens"] or g["every"])])
    print(*grid[0].keys())
    for g in grid:
        print(*g.values())
    print("BEST train rule (edit+offload):", best)
    print("BEST train rule (edit only, fires at least once):", best_edit)
    json.dump({"best": best, "best_edit_only": best_edit, "n_points": len(rows), "grid": grid},
              open(os.path.join(OUT, "rule.json"), "w"), indent=1)


def cmd_data():
    """ctx_policy family (SPEC.md record format) + trainer files (train.py format), split by SOURCE.

    Test sources (sha1(src)%5==0) never enter the trainer files; 1/4 of the remaining sources
    (stable hash) become `val` for temperature fit / epoch selection.
    """
    import hashlib
    rows = cf_table()
    qs = json.load(open(os.path.join(REPO, "crates/omp-judge/src/policy_questions.json")))
    os.makedirs(os.path.join(LAYA_DIR, "families"), exist_ok=True)
    fam = open(os.path.join(LAYA_DIR, "families", "ctx_policy.jsonl"), "w")
    st = open(os.path.join(LAYA_DIR, "states.jsonl"), "w")
    lb = open(os.path.join(LAYA_DIR, "labels.jsonl"), "w")
    split = {"train": [], "val": []}
    n = 0
    for i, r in enumerate(rows):
        tags = ["numeric"] + (["long_ctx"] if parse_state(r["state"])["hist"] > 4000 else [])
        ev = {"window_cost_teq": {a: round(teq(c)) for a, c in r["window_cost"].items()},
              "window_latency_s": {a: round(c, 1) for a, c in r["window_latency"].items()},
              "final_pass": r["score"]}
        fam.write(json.dumps({"id": f"ctx_policy-{i}", "family": "ctx_policy", "difficulty": tags, "src": r["src"],
                              "state": r["state"], "questions": qs, "labels": {"action": r["label"]},
                              "label_source": "exact", "evidence": json.dumps(ev)}) + "\n")
        if heldout(r["src"]):
            continue
        st.write(json.dumps({"family": "ctx_policy", "src": r["src"], "state": r["state"], "questions": qs}) + "\n")
        lb.write(json.dumps({"id": n, "family": "ctx_policy", "labels": {"action": r["label"]}, "difficulty": tags}) + "\n")
        dev = int(hashlib.sha1(("dev" + r["src"]).encode()).hexdigest(), 16) % 4 == 0
        split["val" if dev else "train"].append(n)
        n += 1
    json.dump(split, open(os.path.join(LAYA_DIR, "split.json"), "w"))
    print(f"family: {len(rows)} records; trainer: train {len(split['train'])} val {len(split['val'])} "
          f"(held-out test sources excluded)")


def cmd_oracle_scripts():
    rows = cf_table()
    out = {}
    for src in sorted({r["src"] for r in rows}):
        sc = [0] * (STEPS + 1)
        for r in rows:
            if r["src"] == src:
                sc[r["dp"]] = {"noop": 0, "clm_edit": 1, "rlm_offload": 2}[r["label"]]
        out[src] = sc
    json.dump(out, open(os.path.join(OUT, "oracle_scripts.json"), "w"))
    print({s: "".join(map(str, v)) for s, v in out.items()})


def cmd_report():
    runs = load("ev-")
    pols = {}
    for r in runs.values():
        pols.setdefault(pol_of(r["name"]), []).append(r)
    srcs = sorted({r["source"] for r in runs.values()})
    print("held-out sources:", srcs, "| rollouts:", len(runs))
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731

    def per_src(rs, f):
        by = {}
        for r in rs:
            by.setdefault(r["source"], []).append(f(r))
        return {k: mean(v) for k, v in by.items()}

    base_c = per_src(pols.get("append", []), lambda r: teq(r["totals"]["cost"]))
    base_l = per_src(pols.get("append", []), lambda r: r["latency_s"])
    print("| policy | n | billed-eq tokens (mean/rollout) | vs append | cache-hit % | latency s (mean) | vs append | "
          "t1 anchors pass | t3 resist pass | edits | offloads | rlm calls | spend $ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pol in ("append", "clm", "clm-host", "rule", "laya", "oracle"):
        rs = pols.get(pol, [])
        if not rs:
            continue
        pc, pl = per_src(rs, lambda r: teq(r["totals"]["cost"])), per_src(rs, lambda r: r["latency_s"])
        common = [k for k in pc if k in base_c]
        rt = f"{sum(pc[k] for k in common) / sum(base_c[k] for k in common):.2f}x" if common else "-"
        rl = f"{sum(pl[k] for k in common) / sum(base_l[k] for k in common):.2f}x" if common else "-"
        p = lambda t: (lambda x: f"{sum(r['score'] for r in x)}/{len(x)}")([r for r in rs if r["task"] == t])  # noqa: E731
        offl = sum(1 for r in rs for x in r["records"] if (x.get("policy") or {}).get("action") == "rlm_offload")
        print(f"| {pol} | {len(rs)} | {mean([teq(r['totals']['cost']) for r in rs]):.0f} | {rt} | "
              f"{mean([r['cache_hit_pct'] for r in rs]):.1f} | {mean([r['latency_s'] for r in rs]):.1f} | {rl} | "
              f"{p('t1')} | {p('t3')} | {sum(r['totals']['edits_ok'] for r in rs)} | {offl} | "
              f"{sum(r['totals']['rlm_calls'] for r in rs)} | {sum(r['totals']['cost'] for r in rs):.4f} |")


if __name__ == "__main__":
    {"labels": cmd_labels, "tune": cmd_tune, "data": cmd_data, "oracle-scripts": cmd_oracle_scripts,
     "report": cmd_report}[sys.argv[1]]()
