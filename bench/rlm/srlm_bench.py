#!/usr/bin/env python3
"""SRLM bench: llm_query k=1 vs k=3 (consistency / uncertainty) on a small seeded QA set.

Drives a real `omp-rs --mode rpc` session (temp HOME with copied provider credentials) whose
agent runs the eval cells below verbatim; every sub-call goes through the real `__llm_query__`
host op (budgets, memo, OMP_RLM_LOG). Prints accuracy and sub-call tokens per arm.

    python3 bench/rlm/srlm_bench.py --n 24 --model zai/glm-5.3-flash:minimal
"""
import argparse, json, os, random, re, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "context"))
import run as ctxrun  # noqa: E402  (Rpc, make_home)

SUFFIX = "\nAnswer with only the final answer (one word or one number), no explanation."
NAMES = ["Alma", "Boris", "Chen", "Dara", "Emil", "Fiona", "Goran", "Hana", "Ivo", "Jun", "Kira", "Leo"]
CITIES = {"Oslo": "Norway", "Lyon": "France", "Porto": "Portugal", "Graz": "Austria", "Turku": "Finland",
          "Bern": "Switzerland", "Cork": "Ireland", "Brno": "Czechia", "Gent": "Belgium", "Riga": "Latvia"}
ITEMS = ["bolts", "nuts", "gears", "springs", "valves"]


def make_items(n, seed=7):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 2 == 0:  # multi-hop with moves: current country of a person
            who = rng.sample(NAMES, 9)
            cities = rng.sample(list(CITIES), 10)
            lines, truth = [], {}
            for p in who:
                a, b, c = rng.sample(cities, 3)
                lines.append(f"{p} lived in {a} until 2015, then moved to {b}.")
                lines.append(f"In 2021 {p} relocated to {c}." if rng.random() < .6 else f"{p} still lives in {b}.")
                truth[p] = c if lines[-1].startswith("In 2021") else b
            lines += [f"{c} is a city in {CITIES[c]}." for c in cities]
            rng.shuffle(lines)
            p = rng.choice(who)
            out.append({"ctx": "\n".join(lines), "q": f"In which country does {p} live now?", "gold": CITIES[truth[p]].lower()})
        else:  # aggregation: total of one item over an order log
            item = rng.choice(ITEMS)
            lines, total = [], 0
            for j in range(30):
                it, qty, price = rng.choice(ITEMS), rng.randint(3, 40), rng.randint(2, 19)
                lines.append(f"Order {100 + j}: {qty} {it} at {price} each.")
                if it == item:
                    total += qty * price
            out.append({"ctx": "\n".join(lines), "q": f"What is the total cost of all {item} ordered (sum of quantity*price)?", "gold": str(total)})
    return out


CELL = '''
import json
M = %(model)r
items = json.load(open("items.json"))
res = globals().setdefault("RES", [])
SUF = %(suffix)r
def tok():
    return llm_stats()["total_tokens"]
def cand(r):
    return [c["text"] for c in r.candidates]
def run(a, b):
    for i in range(a, b):
        it = items[i]
        q = it["q"] + SUF
        t0 = tok(); r1 = llm_query(q, context=it["ctx"] + "\\n", model=M)
        t1 = tok(); r3 = llm_query(q, context=it["ctx"], model=M, k=3, select="consistency")
        t2 = tok(); ru = llm_query(q, context=it["ctx"] + " ", model=M, k=3, select="uncertainty")
        t3 = tok()
        res.append({"i": i, "gold": it["gold"], "k1": str(r1), "k1_tok": t1 - t0,
                    "c": str(r3), "c_cands": cand(r3), "c_agree": r3.agreement, "c_tok": t2 - t1,
                    "u": str(ru), "u_cands": [(c["text"], c["confidence"], c["score"]) for c in ru.candidates],
                    "u_agree": ru.agreement, "u_tok": t3 - t2})
    json.dump(res, open("res.json", "w"))
    print(len(res), "done")
'''


def norm(s):
    return re.sub(r"[^\w ]+", "", str(s).lower().replace(",", "")).split()


def ok(ans, gold):
    t = norm(ans)
    return bool(t) and (t[-1] == gold or t == [gold] or gold in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="zai/glm-5.3-flash:minimal")
    ap.add_argument("--driver", default="zai/glm-5.3-flash")
    ap.add_argument("--bin", default=ctxrun.DEFAULT_BIN)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    home = tempfile.mkdtemp(prefix="omp-srlm-bench-")
    try:
        agent, env = ctxrun.make_home(home)
        work = os.path.join(home, "work")
        os.makedirs(work)
        env["OMP_RLM_LOG"] = os.path.join(home, "rlm.jsonl")
        env.update(OMP_RLM_MAX_CALLS="5000", OMP_RLM_MAX_TOKENS="20000000")
        items = make_items(a.n)
        json.dump(items, open(os.path.join(work, "items.json"), "w"))
        cmd = [a.bin, "--mode", "rpc", "--model", a.driver, "--thinking", "off", "--tools", "eval",
               "--no-extensions", "--no-skills", "--no-rules", "--no-title", "--no-lsp", "--no-pty"]
        rpc = ctxrun.Rpc(cmd, env, work)
        orig_frame = rpc.frame

        def frame(t):
            f = orig_frame(t)
            if f.get("type") == "tool_execution_end" and f.get("isError"):
                print("TOOL ERROR:", json.dumps(f.get("result"))[:2000], file=sys.stderr)
            return f
        rpc.frame = frame
        try:
            def cell(code, timeout=900):
                msg = ("Call the eval tool exactly once with language=python, timeout=%d and this code "
                       "VERBATIM (do not change anything), then reply DONE:\n```python\n%s\n```" % (timeout, code))
                status, asst, tools = rpc.prompt(msg, timeout=timeout + 120)
                if not tools or any(t["error"] for t in tools):
                    raise RuntimeError(f"cell failed: {status} {tools} {json.dumps(asst)[:3000]}")
            cell(CELL % {"model": a.model, "suffix": SUFFIX})
            for s in range(0, a.n, a.chunk):
                cell(f"run({s}, {min(s + a.chunk, a.n)})")
                print("chunk", s, file=sys.stderr)
        finally:
            rpc.close()
        res = json.load(open(os.path.join(work, "res.json")))
        log = [json.loads(l) for l in open(env["OMP_RLM_LOG"])]
    finally:
        shutil.rmtree(home, ignore_errors=True)
    n = len(res)
    acc = lambda key: sum(ok(r[key], r["gold"]) for r in res) / n
    sample0 = sum(ok((r["c_cands"] or [""])[0], r["gold"]) for r in res) / n
    summ = {
        "n": n, "model": a.model,
        "acc_k1": acc("k1"), "acc_k3_consistency": acc("c"), "acc_k3_uncertainty": acc("u"),
        "acc_k3_sample0": sample0,
        "tok_k1": sum(r["k1_tok"] for r in res), "tok_k3_consistency": sum(r["c_tok"] for r in res),
        "tok_k3_uncertainty": sum(r["u_tok"] for r in res),
        "mean_agreement_c": sum(r["c_agree"] for r in res) / n,
        "mean_agreement_u": sum(r["u_agree"] for r in res) / n,
        "agreement_vs_correct": {
            "agree=1.0": [sum(ok(r["c"], r["gold"]) for r in res if r["c_agree"] == 1.0), sum(r["c_agree"] == 1.0 for r in res)],
            "agree<1.0": [sum(ok(r["c"], r["gold"]) for r in res if r["c_agree"] < 1.0), sum(r["c_agree"] < 1.0 for r in res)],
        },
        "log_select_lines": sum(1 for l in log if l.get("kind") == "select"),
        "log_call_lines": sum(1 for l in log if l.get("kind") != "select"),
    }
    stamp = tempfile.mktemp(prefix="srlm-", dir=a.out).rsplit("/", 1)[1]
    json.dump({"summary": summ, "rows": res, "log": log}, open(os.path.join(a.out, stamp + ".json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("saved", os.path.join(a.out, stamp + ".json"))


if __name__ == "__main__":
    main()
