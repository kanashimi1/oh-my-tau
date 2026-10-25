#!/usr/bin/env python3
"""Post-hoc signal analysis on recorded long-session trajectories (no LLM calls).

Question: which cheap per-step signals predict that compacting (offload / context_edit) pays at that step?
Signals: SoT-Embed proxies (bge-m3 CPU: delta = distance to the running mean, v = step displacement, c = cosine of consecutive
displacements, H = mean cosine distance to the 3 nearest earlier steps; same definitions as crates/omp-sot), SDO events
(new entity first seen, tool error, offloaded file accessed), 'block already restated' (recorded by the host policy hook),
plus baselines step number and history size.
Label (counterfactual from paired arms, same family/seed/rep): saving_t = 1 - cost_arm_t / cost_append_t averaged over the arm
group; y = saving_t >= THR. Groups: offload = offload,rule,lcm ; edit = clm,clm-host.

  sot_signals.py            -> markdown on stdout
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longbench as LB  # noqa: E402

OUT = os.path.join(HERE, "results", "long")
EMB = os.path.join(OUT, "sot_emb.npz")
BGE = "/home/kaneki/security-kg-rag/models/bge-m3-local"
THR = 0.15
GROUPS = {"offload": ["offload", "rule", "lcm"], "edit": ["clm", "clm-host"]}


def load():
    rows = {}
    for p in glob.glob(os.path.join(OUT, "L*-s*-r*.json")):
        d = json.load(open(p))
        if "records" in d:
            rows[(d["family"], d["seed"], d["rep"], d["arm"])] = d
    return rows


def embed(texts):
    cache = dict(np.load(EMB, allow_pickle=True)) if os.path.exists(EMB) else {}
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        import torch
        torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(BGE, device="cpu")
        m.max_seq_length = 256
        v = m.encode(todo, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        for t, e in zip(todo, v):
            cache[t] = e
        np.savez(EMB, **cache)
    return {t: cache[t] for t in texts}


def state_of(rec):
    pol = rec.get("policy")
    if not pol:
        return {}
    out = {}
    for ln in pol["state"].splitlines():
        if ": " in ln:
            k, v = ln.split(": ", 1)
            m = re.match(r"\d+", v)
            if m:
                out[k] = int(m.group(0))
    return out


def build():
    R = load()
    eps = sorted({(f, s, r) for f, s, r, a in R})
    items = []  # (family, seed, rep, step, text)
    cache_ep = {}
    for f, s, r in eps:
        base = R.get((f, s, r, "append"))
        if not base:
            continue
        if (f, s) not in cache_ep:
            cache_ep[(f, s)] = LB.get_episode(f, s, base["steps"])
        ep = cache_ep[(f, s)]
        for rec in base["records"]:
            st = rec["step"]
            m = ep["brief"] if st == 0 else ep["msgs"].get(st, "")
            items.append((f, s, r, st, (m[:600] + " || " + rec["reply"][:160])))
    E = embed([i[4] for i in items])
    rows = []
    seen = {}
    for f, s, r in eps:
        base = R.get((f, s, r, "append"))
        if not base:
            continue
        ep = cache_ep[(f, s)]
        U = [E[i[4]] for i in items if i[:3] == (f, s, r)]
        recs = base["records"]
        bad = set(ep.get("meta", {}).get("bad", [])) if f == "L2" else set()
        ents = set()
        pols = [R[(f, s, rr, a)] for rr in (1, 2) for a in GROUPS["offload"] if (f, s, rr, a) in R]
        for t, rec in enumerate(recs):
            u = U[t]
            prev = U[:t]
            v = float(np.linalg.norm(u - U[t - 1])) if t >= 1 else 0.0
            delta = float(np.linalg.norm(u - np.mean(prev, 0))) if t >= 1 else 0.0
            if t >= 2:
                d1, d2 = u - U[t - 1], U[t - 1] - U[t - 2]
                c = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))
            else:
                c = 0.0
            if t >= 1:
                dist = sorted(1 - np.array(prev) @ u)[:3]
                H = float(np.mean(dist))
            else:
                H = 0.0
            msg = ep["brief"] if rec["step"] == 0 else ep["msgs"].get(rec["step"], "")
            found = set(re.findall(r"mod_\d+|Q\d+|SECTION \d+", msg)) | set(re.findall(r"`python3 run_tests.py (\d+)`", msg))
            new_ent = int(bool(found - ents))
            ents |= found
            tool_err = int(rec["edits_rej"] > 0 or ("mod_%02d" % rec["step"] in bad))
            ps = [state_of(p["records"][t]) for p in pols if t < len(p["records"])]
            ps = [x for x in ps if x]
            restated = float(np.mean([x.get("block_restated_pct", 0) for x in ps])) if ps else 0.0
            age = float(np.mean([x.get("block_age_turns", 0) for x in ps])) if ps else 0.0
            since = float(np.mean([x.get("block_since_ref_turns", 0) for x in ps])) if ps else 0.0
            blk = float(np.mean([x.get("largest_old_block_chars", 0) for x in ps])) if ps else 0.0
            acc = float(np.mean([x.get("offload_reads", 0) for x in ps])) if ps else 0.0
            hist = rec["uncached"] + rec["cached"] + rec["cache_write"]
            lab = {}
            for g, arms in GROUPS.items():
                sv = []
                for a in arms:
                    x = R.get((f, s, r, a))
                    if x and t < len(x["records"]) and rec["cost"] > 0:
                        sv.append(1 - x["records"][t]["cost"] / rec["cost"])
                lab[g] = float(np.mean(sv)) if sv else None
            rows.append(dict(fam=f, seed=s, rep=r, step=rec["step"], hist=hist, v=v, delta=delta, c=c, H=H, new_ent=new_ent,
                             tool_err=tool_err, restated=restated, age=age, since_ref=since, block=blk, offload_access=acc,
                             sav_offload=lab["offload"], sav_edit=lab["edit"]))
    return rows


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    ranks = np.concatenate([pos, neg])
    # average ranks for ties
    from scipy.stats import rankdata
    rk = rankdata(ranks)
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def lift(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int)
    if len(set(x)) < 2 or y.sum() == 0:
        return float("nan")
    k = max(1, len(x) // 4)
    top = np.argsort(-x, kind="stable")[:k]
    return float(y[top].mean() / y.mean())


SIGS = ["step", "hist", "delta", "v", "c", "H", "new_ent", "tool_err", "restated", "age", "since_ref", "block", "offload_access"]


def main():
    rows = build()
    for r in rows:
        r["step_"] = r["step"]
    out = [f"label threshold: group-mean saving_t >= {THR}; rows with a label per group shown as n; AUC>0.5 = higher signal -> compacting pays\n"]
    for g in GROUPS:
        out.append(f"\n#### group `{g}` ({', '.join(GROUPS[g])})\n")
        out.append("| signal | all AUC | all lift@25% | " + " | ".join(f"{f} AUC" for f in ["L1", "L2", "L3", "L4"]) + " | within-family mean AUC |")
        out.append("|---|---|---|---|---|---|---|---|")
        R = [r for r in rows if r["sav_" + g] is not None and r["step"] >= 1]
        y = [int(r["sav_" + g] >= THR) for r in R]
        out.append(f"| (n rows / base rate) | {len(R)} | {np.mean(y):.2f} | " + " | ".join(
            f"{sum(1 for r in R if r['fam'] == f)}/{np.mean([int(r['sav_' + g] >= THR) for r in R if r['fam'] == f] or [float('nan')]):.2f}"
            for f in ["L1", "L2", "L3", "L4"]) + " | |")
        for sg in SIGS:
            key = "step" if sg == "step" else sg
            a_all = auc([r[key] for r in R], y)
            l_all = lift([r[key] for r in R], y)
            per = []
            for f in ["L1", "L2", "L3", "L4"]:
                Rf = [r for r in R if r["fam"] == f]
                per.append(auc([r[key] for r in Rf], [int(r["sav_" + g] >= THR) for r in Rf]))
            fm = np.nanmean(per) if not all(np.isnan(per)) else float("nan")
            out.append(f"| {sg} | {a_all:.2f} | {l_all:.2f} | " + " | ".join(f"{p:.2f}" for p in per) + f" | {fm:.2f} |")
    # rule: logistic on the best non-baseline signals, leave-one-(family,seed)-out
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out.append("\n#### learned rule (logistic, leave-one-episode-out AUC, group `offload`)\n")
    R = [r for r in rows if r["sav_offload"] is not None and r["step"] >= 1]
    y = np.array([int(r["sav_offload"] >= THR) for r in R])
    keys_sets = {"step only": ["step"], "hist only": ["hist"], "step+hist": ["step", "hist"],
                 "SoT (delta,v,c,H)": ["delta", "v", "c", "H"], "SDO (new_ent,tool_err,offload_access)": ["new_ent", "tool_err", "offload_access"],
                 "restated+age+block": ["restated", "age", "block"], "hist+restated+age+block": ["hist", "restated", "age", "block"],
                 "all": [s for s in SIGS]}
    eps = np.array([f"{r['fam']}-{r['seed']}" for r in R])
    out.append("| feature set | LOEO AUC | note |\n|---|---|---|")
    for name, ks in keys_sets.items():
        X = np.array([[r[k] for k in ks] for r in R], float)
        pred = np.zeros(len(R))
        ok = True
        for e in sorted(set(eps)):
            te = eps == e
            if len(set(y[~te])) < 2:
                ok = False
                break
            sc = StandardScaler().fit(X[~te])
            m = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(X[~te]), y[~te])
            pred[te] = m.decision_function(sc.transform(X[te]))
        out.append(f"| {name} | {auc(pred, y):.2f} | {'' if ok else 'degenerate'} |" if ok else f"| {name} | n/a | degenerate |")
    json.dump(rows, open(os.path.join(OUT, "sot_rows.json"), "w"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
