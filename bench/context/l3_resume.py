"""Resume L3: run only rollouts whose result json is missing (survives reboots: just rerun)."""
import concurrent.futures as cf, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import longbench as lb
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "long")
arms = "append,lcm,offload,rule,clm,clm-host".split(",")
jobs = [(f"L3-s{s}-{a}-r1", s, a) for s in (0, 2, 3) for a in arms
        if not os.path.exists(os.path.join(R, f"L3-s{s}-{a}-r1.json"))]
print("todo", len(jobs), [j[0] for j in jobs], flush=True)
with cf.ThreadPoolExecutor(2) as ex:
    fs = {ex.submit(lb.safe, n, "L3", s, a, 1, None, False, 15.0, None): n for n, s, a in jobs}
    for f in cf.as_completed(fs):
        x = f.result()
        if "error" in x: print(x["name"], "ERR", x["error"], flush=True)
        else:
            t = x["totals"]; print(f"{x['name']} score={x['score']:.2f} cost={t['cost']:.4f} lat={x['latency_s']} sub={t['sub_calls']}", flush=True)
print("done", flush=True)
