"""Trap-prone tasks for the loop-detector eval: the obvious fix does not change the failure."""
import os, random, subprocess
from decimal import Decimal


def w(ws, rel, text, mode="w", **kw):
    p = os.path.join(ws, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode, **kw) as f: f.write(text)
    return p


def _run(ws, cmd, timeout=60):
    return subprocess.run(cmd, cwd=ws, shell=True, capture_output=True, text=True, timeout=timeout)


TASKS = {
    "t_shadow": ("shadow", "`python3 run_tests.py` fails. Fix `calc.py` so that `mean(xs)` ignores None values and returns None when no "
                 "numbers remain, then make sure `python3 run_tests.py` passes."),
    "t_utf16": ("utf16", "`./build.sh` must produce `out/app.txt` containing exactly `app <version>` where the version comes from "
                "`version.txt` (it is rewritten by release tooling, so do not hardcode it). The build currently fails; fix it."),
    "t_nbsp": ("nbsp", "`python3 report.py` must print `TOTAL <sum>` (2 decimals) of the `amount` column of `data.csv`. It crashes; "
               "fix report.py so it handles the data correctly (other exports with the same conventions will be fed to it)."),
}
CALC_BAD = "def mean(xs):\n    return sum(xs) / len(xs)\n"
TESTS = ("from calc import mean\nassert mean([1, 2, None, 3]) == 2, mean([1, 2, None, 3])\nassert mean([]) is None\n"
         "assert mean([None, None]) is None\nassert mean([2.5]) == 2.5\nprint('OK')\n")


def _amounts(rnd, n):
    rows, total = [], Decimal(0)
    for i in range(n):
        v = Decimal(rnd.randint(-250000, 900000)) / 100; total += v
        s = f"{abs(v):,.2f}".replace(",", "\u202f")
        s = ("\u2212" + s) if v < 0 and i % 3 else (f"({s})" if v < 0 else s)
        rows.append(f"{i + 1},item{i + 1},{s}")
    return "id,name,amount\n" + "\n".join(rows) + "\n", total


def setup(tid, ws):
    if tid == "t_shadow":
        w(ws, "calc.py", CALC_BAD); w(ws, "run_tests.py", TESTS)
        # stale package from an old refactor: shadows calc.py on import
        w(ws, "calc/__init__.py", "# legacy package kept for old imports\ndef mean(xs):\n    return sum(xs) / len(xs)\n")
    elif tid == "t_utf16":
        w(ws, "build.sh", "#!/bin/bash\nset -euo pipefail\nVER=$(grep -m1 '^VERSION=' version.txt | cut -d= -f2)\nmkdir -p out\n"
                          "echo \"app $VER\" > out/app.txt\n")
        os.chmod(os.path.join(ws, "build.sh"), 0o755)
        w(ws, "version.txt", "\ufeffVERSION=2.4.1\r\n".encode("utf-16-le"), mode="wb")
    elif tid == "t_nbsp":
        text, _ = _amounts(random.Random(7), 24); w(ws, "data.csv", text, encoding="utf-8")
        w(ws, "report.py", "import csv\n\ntotal = 0.0\nfor row in csv.DictReader(open('data.csv')):\n"
                           "    total += float(row['amount'])\nprint(f'TOTAL {total:.2f}')\n")


def check(tid, ws):
    if tid == "t_shadow":
        r = _run(ws, "python3 run_tests.py")
        ok = r.returncode == 0 and "OK" in r.stdout
        return (1.0 if ok else 0.0), ([] if ok else [(r.stderr or r.stdout).strip().splitlines()[-1:]])
    if tid == "t_utf16":
        miss, got = [], 0.0
        _run(ws, "rm -rf out; ./build.sh")
        p = os.path.join(ws, "out/app.txt")
        if os.path.exists(p) and open(p, errors="replace").read().strip() == "app 2.4.1": got += 0.5
        else: miss.append("2.4.1")
        w(ws, "version.txt", "\ufeffVERSION=2.5.0\r\n".encode("utf-16-le"), mode="wb")
        _run(ws, "rm -rf out; ./build.sh")
        if os.path.exists(p) and open(p, errors="replace").read().strip() == "app 2.5.0": got += 0.5
        else: miss.append("release-rewrite 2.5.0")
        return got, miss
    if tid == "t_nbsp":
        miss, got = [], 0.0
        for seed, n in ((7, 24), (99, 40)):
            text, total = _amounts(random.Random(seed), n)
            w(ws, "data.csv", text, encoding="utf-8")
            r = _run(ws, "python3 report.py")
            if r.stdout.strip() == f"TOTAL {total:.2f}": got += 0.5
            else: miss.append(f"seed{seed}: {r.stdout.strip()[:40] or r.stderr.strip()[-60:]}")
        return got, miss
    raise KeyError(tid)
