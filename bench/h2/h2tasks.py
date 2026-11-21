"""Sandbox tasks for the Harness² pilot (bench/h2). Each task: setup(ws) writes inputs, PROMPT is what the
executor sees, check(ws) -> (score 0..1, notes). Checkers are hidden from executor AND from the
Propose/Probe/Compose/Reflect sessions (open-world setting); they are used only for reporting."""
import csv, gzip, io, json, os, random, re, subprocess, sys, textwrap
from decimal import Decimal, ROUND_HALF_UP

def w(ws, rel, text, newline="\n", mode="w"):
    p = os.path.join(ws, rel); os.makedirs(os.path.dirname(p) or ws, exist_ok=True)
    if mode == "wb":
        open(p, "wb").write(text); return
    with open(p, "w", newline=newline) as f: f.write(text)

def frac(checks):
    ok = sum(1 for c in checks if c[1]); return ok / len(checks), [c[0] for c in checks if not c[1]]

# ---------------------------------------------------------------- data: reconcile
def d_reconcile_setup(ws):
    rnd = random.Random(7); inv, pay = [], []
    for i in range(1, 41):
        amt = Decimal(rnd.randint(1000, 900000)) / 100
        eu = f"{amt:,.2f}".replace(",", " ").replace(".", ",")
        inv.append({"invoice": f"INV-{i:03d}", "amount": eu, "currency": "EUR"})
        if i % 9 == 0: continue                       # unpaid
        p = amt + (Decimal("12.40") if i % 7 == 0 else Decimal(0))
        pid = f"INV-{i}" if i % 2 else f"inv-{i:03d}"     # id formats differ
        if i % 5 == 0: pid = f" {pid} "                     # stray whitespace
        if i % 4 == 0:                                      # split transfer
            h = (p / 2).quantize(Decimal("0.01")); pay.append({"ref": pid, "paid": f"{h:.2f}"}); p = p - h
        pay.append({"ref": pid, "paid": str(p.normalize()) if i % 3 else f"{p:.2f}"})
    pay.append({"ref": "INV-099", "paid": "10.00"})     # payment without invoice
    buf = io.StringIO(); wr = csv.DictWriter(buf, ["invoice", "amount", "currency"], delimiter=";"); wr.writeheader(); wr.writerows(inv)
    w(ws, "invoices.csv", buf.getvalue()); w(ws, "payments.json", json.dumps(pay, indent=1))
    exp = {"mismatches": [], "unpaid": [], "unknown_payments": ["INV-099"]}
    for i in range(1, 41):
        amt = Decimal(inv[i - 1]["amount"].replace(" ", "").replace(",", "."))
        if i % 9 == 0: exp["unpaid"].append(f"INV-{i:03d}"); continue
        if i % 7 == 0: exp["mismatches"].append({"invoice": f"INV-{i:03d}", "invoiced": f"{amt:.2f}", "paid": f"{amt + Decimal('12.40'):.2f}", "diff": "12.40"})
    w(ws, ".expected.json", json.dumps(exp)) if False else None
    return exp
D_RECONCILE = """Reconcile invoices.csv against payments.json (payment "ref" = invoice id).
Write report.json with exactly these keys:
  "mismatches": list of {"invoice","invoiced","paid","diff"} for invoices whose paid amount differs from the invoiced amount;
                amounts as strings with exactly 2 decimals, diff = paid - invoiced; sorted by invoice id; ids in the INV-NNN form used by invoices.csv.
  "unpaid": sorted list of canonical invoice ids with no payment.
  "unknown_payments": sorted list of payment refs (INV-NNN form) that match no invoice."""
def d_reconcile_check(ws):
    exp = d_reconcile_setup(os.path.join(ws, ".chk")) if False else _exp("d_reconcile")
    try: got = json.load(open(os.path.join(ws, "report.json")))
    except Exception as e: return 0.0, [f"report.json unreadable: {e}"]
    return frac([("mismatches", got.get("mismatches") == exp["mismatches"]), ("unpaid", got.get("unpaid") == exp["unpaid"]),
                 ("unknown_payments", got.get("unknown_payments") == exp["unknown_payments"]), ("keys", sorted(got) == sorted(exp))])

# ---------------------------------------------------------------- data: logs
def d_logs_setup(ws):
    rnd = random.Random(11); svcs = ["auth", "billing", "search", "mailer", "gateway"]; count = {s: 0 for s in svcs}
    for n in range(6):
        lines = []
        for k in range(400):
            s = rnd.choice(svcs); lvl = rnd.choice(["INFO", "INFO", "WARN", "ERROR", "error", "DEBUG"])
            lines.append(f"2026-10-0{1 + n}T{k // 60:02d}:{k % 60:02d}:00Z svc={s} level={lvl} msg=op {k}")
            if lvl.upper() == "ERROR":
                count[s] += 1
                if rnd.random() < 0.4: lines += ["  Traceback (most recent call last):", "    level=ERROR svc=" + rnd.choice(svcs) + " in nested frame", "  ValueError: bad"]
        body = "\n".join(lines) + "\n"
        if n % 2: w(ws, f"logs/app-{n}.log.gz", gzip.compress(body.encode()), mode="wb")
        else: w(ws, f"logs/app-{n}.log", body)
    w(ws, "logs/README.txt", "rotated logs; .gz files are older rotations\n")
    return sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))
D_LOGS = """logs/ contains our application logs. Count ERROR-level log records per service across all of them and write
errors_by_service.csv with header "service,errors", rows sorted by errors descending then service ascending."""
def d_logs_check(ws):
    exp = [tuple(x) for x in _exp("d_logs")]
    try: rows = list(csv.reader(open(os.path.join(ws, "errors_by_service.csv"))))
    except Exception as e: return 0.0, [f"csv unreadable: {e}"]
    got = [(r[0], int(r[1])) for r in rows[1:] if len(r) == 2 and r[1].strip().isdigit()]
    return frac([("header", rows[:1] == [["service", "errors"]]), ("counts", sorted(got) == sorted(exp)), ("order", got == exp)])

# ---------------------------------------------------------------- data: budget (held-out)
def d_budget_setup(ws):
    rnd = random.Random(5); tot = {}
    fmts = {"venue": lambda v: f"€{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), "labor": lambda v: f"{v:,.2f} EUR",
            "catering": lambda v: f"{v:.2f}".replace(".", ","), "av": lambda v: f"EUR {v:,.0f}" if v == v.to_integral() else f"EUR {v:,.2f}"}
    for cat, f in fmts.items():
        rows = ["item,cost,status"]; t = Decimal(0)
        for i in range(12):
            v = Decimal(rnd.randint(100, 500000)) / (1 if cat == "av" else 100)
            st = rnd.choice(["confirmed", "Confirmed", "confirmed ", "cancelled", "quote", "unconfirmed"])
            if st.strip().lower() == "confirmed": t += v
            rows.append(f'{cat}-{i},"{f(v)}",{st}')
        tot[cat] = f"{t:.2f}"; w(ws, f"budget/{cat}.csv", "\n".join(rows) + "\n")
    tot["grand"] = f"{sum(Decimal(v) for v in tot.values()):.2f}"
    return tot
D_BUDGET = """budget/ holds one CSV per cost category. Only confirmed rows count. Write memo.md containing one line per category, in the form
"TOTAL <category> = <amount>" (category = file name without .csv, amount with exactly 2 decimals, '.' decimal separator,
no thousands separator), plus a final line "TOTAL grand = <amount>"."""
def d_budget_check(ws):
    exp = _exp("d_budget")
    try: txt = open(os.path.join(ws, "memo.md")).read()
    except Exception as e: return 0.0, [f"memo.md unreadable: {e}"]
    got = dict(re.findall(r"TOTAL (\w+) = (-?\d+\.\d{2})\b", txt))
    return frac([(k, got.get(k) == v) for k, v in exp.items()])

# ---------------------------------------------------------------- code: pricing bugfix
PRICING = '''"""Pricing helpers."""


def apply_discount(price, percent):
    """Return price after percent discount, rounded half-up to cents."""
    return round(price * (1 - percent / 100), 2)


def total(lines, percent=0):
    """lines: iterable of (unit_price, qty). Each line total (unit_price * qty) is discounted and rounded; result is their sum."""
    return round(sum(apply_discount(p * q, percent) for p, q in lines), 2)
'''
PRICING_TESTS = '''import unittest
from shop.pricing import apply_discount, total


class T(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(str(apply_discount(100, 10)), "90.00")

    def test_half_up(self):
        self.assertEqual(str(apply_discount(2.675, 0)), "2.68")
        self.assertEqual(str(apply_discount(0.125, 0)), "0.13")

    def test_total(self):
        self.assertEqual(str(total([(19.99, 3), (5.005, 2)], 0)), "69.98")


if __name__ == "__main__":
    unittest.main()
'''
def c_pricing_setup(ws):
    w(ws, "shop/__init__.py", ""); w(ws, "shop/pricing.py", PRICING); w(ws, "tests/__init__.py", ""); w(ws, "tests/test_pricing.py", PRICING_TESTS)
    return {}
C_PRICING = """The unit tests in tests/ fail (run: python3 -m unittest discover -s tests -t .). Fix shop/pricing.py so they pass.
Money must be exact and rounded half-up to cents; callers pass int, float or str amounts. Do not modify the tests."""
def c_pricing_check(ws):
    hidden = textwrap.dedent('''
        import sys; sys.path.insert(0, ".")
        from decimal import Decimal
        from shop.pricing import apply_discount, total
        r = [str(apply_discount(100, 10)) == "90.00", str(apply_discount("2.675", 0)) == "2.68", str(apply_discount(1.005, 0)) == "1.01",
             str(apply_discount(33.335, 50)) == "16.67",
             str(total([(19.99, 3), (5.005, 2)], 0)) == "69.98", str(total([("10.00", 1), (0.105, 1)], 0)) == "10.11"]
        print(sum(r), len(r))''')
    ok_tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ws, capture_output=True, timeout=60).returncode == 0
    same_tests = open(os.path.join(ws, "tests/test_pricing.py")).read() == PRICING_TESTS
    try:
        o = subprocess.run([sys.executable, "-c", hidden], cwd=ws, capture_output=True, text=True, timeout=60).stdout.split()
        hid = int(o[0]) / int(o[1])
    except Exception: hid = 0.0
    s, miss = frac([("unit tests", ok_tests), ("tests untouched", same_tests)])
    return (s * 2 + hid * 5) / 7, miss + ([f"hidden {hid:.2f}"] if hid < 1 else [])

# ---------------------------------------------------------------- code: CRLF properties
PROPS = ["# shader options", "screen=<profile> [LIGHTING] [WATER]", "screen.LIGHTING=SUN_I MOON_I <empty> AO_I",
         "screen.WATER=WATER_FOG WAVES", "sliders=SUN_I MOON_I AO_I", "profile.LOW=!WAVES", "", "# end"]
LANG = ["option.SUN_I=Sun Intensity", "option.MOON_I=Moon Intensity", "option.AO_I=Ambient Occlusion"]
def c_crlf_setup(ws):
    w(ws, "pack/shaders.properties", "\r\n".join(PROPS) + "\r\n", newline=""); w(ws, "pack/lang/en_US.lang", "\r\n".join(LANG) + "\r\n", newline="")
    return {}
C_CRLF = """Add three options to the shader pack in pack/: GI_DAY, GI_NIGHT and AO_I.
They belong on the screen.LIGHTING line, appended after the existing entries in that order, and all three are sliders
(add to the sliders= line, after existing entries). Add English labels "GI Day", "GI Night", "Ambient Occlusion" as option.<NAME>=<label>
lines in pack/lang/en_US.lang. No duplicate tokens or keys; every other line must stay byte-identical."""
def c_crlf_check(ws):
    try:
        p = open(os.path.join(ws, "pack/shaders.properties"), "rb").read(); l = open(os.path.join(ws, "pack/lang/en_US.lang"), "rb").read()
    except Exception as e: return 0.0, [str(e)]
    pl = p.split(b"\r\n"); crlf = b"\n" not in p.replace(b"\r\n", b"") and b"\n" not in l.replace(b"\r\n", b"")
    lines = {x.split(b"=", 1)[0]: x for x in pl if b"=" in x}
    exp_light = b"screen.LIGHTING=SUN_I MOON_I <empty> AO_I GI_DAY GI_NIGHT"; exp_sl = b"sliders=SUN_I MOON_I AO_I GI_DAY GI_NIGHT"
    others = [x.encode() for x in PROPS if not x.startswith(("screen.LIGHTING", "sliders"))]
    ll = l.split(b"\r\n")
    return frac([("crlf", crlf), ("screen.LIGHTING", lines.get(b"screen.LIGHTING") == exp_light), ("sliders", lines.get(b"sliders") == exp_sl),
                 ("others identical", all(o in pl for o in others) and len([x for x in pl if x]) == len([x for x in PROPS if x])),
                 ("lang", b"option.GI_DAY=GI Day" in ll and b"option.GI_NIGHT=GI Night" in ll and ll.count(b"option.AO_I=Ambient Occlusion") == 1)])

# ---------------------------------------------------------------- code: rename (held-out)
def c_rename_setup(ws):
    w(ws, "app/__init__.py", ""); w(ws, "app/utils.py", 'def fetch_data(source, limit=10):\n    """Return up to limit records from source."""\n    return [f"{source}:{i}" for i in range(limit)]\n')
    w(ws, "app/api.py", "from app.utils import fetch_data\n\n\ndef handle(src):\n    return fetch_data(src, limit=3)\n")
    w(ws, "app/jobs.py", "from app import utils\n\n\ndef nightly():\n    return len(utils.fetch_data('db', 5))\n")
    w(ws, "app/registry.py", "from app.utils import fetch_data as _fd\n\nHANDLERS = {\"fetch_data\": _fd}\n")
    w(ws, "app/cli.py", "import app.utils\n\n\ndef main():\n    print(app.utils.fetch_data('cli')[0])\n")
    w(ws, "app/plugins.py", "from app import utils\n\nLOADER = 'fetch_data'\n\n\ndef run(name):\n    return getattr(utils, LOADER)(name, 2)\n")
    w(ws, "tests/__init__.py", ""); w(ws, "tests/test_app.py", "import unittest\nfrom app import api, jobs, registry\n\n\nclass T(unittest.TestCase):\n    def test_all(self):\n        self.assertEqual(api.handle('x'), ['x:0', 'x:1', 'x:2'])\n        self.assertEqual(jobs.nightly(), 5)\n        self.assertEqual(registry.HANDLERS['fetch_data']('r', 1), ['r:0'])\n")
    return {}
C_RENAME = """Rename the function fetch_data in app/utils.py to load_records and update every use in app/.
The registry key "fetch_data" in app/registry.py is a public wire name and must stay unchanged (its value must be the renamed function).
No function named fetch_data may remain. Tests: python3 -m unittest discover -s tests -t ."""
def c_rename_check(ws):
    src = "".join(open(os.path.join(ws, "app", f)).read() for f in os.listdir(os.path.join(ws, "app")) if f.endswith(".py"))
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ws, capture_output=True, timeout=60).returncode == 0
    probe = subprocess.run([sys.executable, "-c", "from app.utils import load_records as f; from app.registry import HANDLERS as H; import app.utils as u; "
                            "assert H['fetch_data'] is f and not hasattr(u, 'fetch_data'); from app import plugins; assert plugins.run('p') == ['p:0', 'p:1']; from app import cli; cli.main()"], cwd=ws, capture_output=True, text=True, timeout=60)
    return frac([("tests", tests), ("probe", probe.returncode == 0 and probe.stdout.strip() == "cli:0"), ("no def", "def fetch_data" not in src)])

# ---------------------------------------------------------------- data: sessions (held-out)
def d_sessions_setup(ws):
    import datetime as _dt
    rnd = random.Random(23); users = [f"u{n:03d}" for n in range(400)]; per = {}
    for n in range(4):
        lines = ["# access log v2: ts user path status"]
        for k in range(120):
            base = _dt.datetime(2026, 10, 10, tzinfo=_dt.timezone.utc) + _dt.timedelta(minutes=rnd.choice([rnd.randint(0, 3 * 1440), rnd.randint(1, 1440) * 1 + 1440 * rnd.randint(1, 2) - rnd.randint(0, 300)]))
            off = rnd.choice([0, 2, -5]); tz = _dt.timezone(_dt.timedelta(hours=off))
            ts = base.astimezone(tz).isoformat().replace("+00:00", "Z")
            u = rnd.choice(users); st = rnd.choice([200, 200, 200, 304, 401])
            lines.append(f"{ts} {u} /p/{k % 17} {st}")
            if st != 401: per.setdefault(base.date().isoformat(), set()).add(u)
        body = "\n".join(lines) + "\n"
        if n >= 2: w(ws, f"access/access-{n}.log.gz", gzip.compress(body.encode()), mode="wb")
        else: w(ws, f"access/access-{n}.log", body)
    return {d: len(v) for d, v in sorted(per.items())}
D_SESSIONS = """access/ contains web access logs. Count distinct authenticated users per UTC calendar day
(a request with status 401 is not authenticated). Write users_per_day.json as an object {"YYYY-MM-DD": count} covering every day present."""
def d_sessions_check(ws):
    exp = _exp("d_sessions")
    try: got = json.load(open(os.path.join(ws, "users_per_day.json")))
    except Exception as e: return 0.0, [f"json unreadable: {e}"]
    return frac([(d, got.get(d) == v) for d, v in exp.items()] + [("no extra days", set(got) == set(exp))])

# ---------------------------------------------------------------- code: durations (held-out)
DUR = r'''"""Duration parsing."""
import re

UNITS = {"s": 1, "m": 60, "h": 3600}


def parse_duration(text):
    """Parse '90s', '1h30m', '2d', '1w2d', '1.5h' or a bare number of seconds ('45') into integer seconds.
    Units: w (week), d (day), h, m, s; case-insensitive; surrounding whitespace allowed. Raise ValueError on anything else."""
    total = 0
    for num, unit in re.findall(r"(\d+)([smh])", text):
        total += int(num) * UNITS[unit]
    return total
'''
DUR_TESTS = '''import unittest
from timeutil.duration import parse_duration


class T(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("2d"), 172800)

    def test_bad(self):
        with self.assertRaises(ValueError):
            parse_duration("abc")
'''
def c_duration_setup(ws):
    w(ws, "timeutil/__init__.py", ""); w(ws, "timeutil/duration.py", DUR); w(ws, "tests/__init__.py", ""); w(ws, "tests/test_duration.py", DUR_TESTS)
    return {}
C_DURATION = """tests/ fail (python3 -m unittest discover -s tests -t .). Fix timeutil/duration.py so that parse_duration
fully implements its docstring. Do not modify the tests."""
DUR_HIDDEN = """
import sys; sys.path.insert(0, ".")
from timeutil.duration import parse_duration as p
def bad(x):
    try: p(x); return False
    except ValueError: return True
r = [p("1h30m") == 5400, p("2d") == 172800, p("1w2d") == 777600, p("1.5h") == 5400, p("45") == 45, p(" 2H ") == 7200,
     p("90s") == 90, bad("abc"), bad("5x"), bad(""), bad("1h junk"), bad("h")]
print(sum(r), len(r))
"""
def c_duration_check(ws):
    ok = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ws, capture_output=True, timeout=60).returncode == 0
    same = open(os.path.join(ws, "tests/test_duration.py")).read() == DUR_TESTS
    try:
        o = subprocess.run([sys.executable, "-c", DUR_HIDDEN], cwd=ws, capture_output=True, text=True, timeout=60).stdout.split(); hid = int(o[0]) / int(o[1])
    except Exception: hid = 0.0
    s, miss = frac([("unit tests", ok), ("tests untouched", same)])
    return (s * 2 + hid * 6) / 8, miss + ([f"hidden {hid:.2f}"] if hid < 1 else [])

# ================================================================ held-out v2 (harder; added after pilot: none-arm ceiling was 0.925)
# ---------------------------------------------------------------- data: latency p95 (held-out2)
def d_metrics_setup(ws):
    import math
    rnd = random.Random(31); eps = ["/api/cart", "/api/login", "/api/search", "/api/user"]; seen = {}
    recs = []
    for k in range(900):
        ms = int(rnd.lognormvariate(4.6, 0.7)) + 1
        st = rnd.choice([200, 200, 200, 201, 204, 404, 500, 503])
        recs.append({"id": f"r{k:05d}", "endpoint": rnd.choice(eps), "ms": ms, "status": st})
    files = [[], [], []]
    for r in recs:
        f = rnd.randrange(3)
        lat = rnd.choice([f"{r['ms']}ms", r["ms"], f"{r['ms'] / 1000:g}s" if r["ms"] % 10 == 0 else f"{r['ms']}ms"])
        st = r["status"] if rnd.random() < 0.7 else str(r["status"])
        line = json.dumps({"ts": f"2026-10-0{f + 1}T00:00:{r['ms'] % 60:02d}Z", "id": r["id"], "endpoint": r["endpoint"], "latency": lat, "status": st})
        files[f].append(line)
        if rnd.random() < 0.12: files[(f + 1) % 3].append(line)   # re-shipped by the log forwarder
        seen[r["id"]] = r
    for n, lines in enumerate(files):
        rnd.shuffle(lines); lines.insert(0, "# forwarder v3"); lines.insert(rnd.randrange(1, len(lines)), '{"id": "r9", "endpo')
        body = "\n".join(lines) + "\n"
        if n == 1: w(ws, f"metrics/req-{n}.jsonl.gz", gzip.compress(body.encode()), mode="wb")
        else: w(ws, f"metrics/req-{n}.jsonl", body)
    exp = {}
    for e in eps:
        v = sorted(r["ms"] for r in seen.values() if r["endpoint"] == e and r["status"] < 500)
        exp[e] = v[math.ceil(0.95 * len(v)) - 1]
    return exp
D_METRICS = """metrics/ holds request logs from our log forwarder. For each endpoint compute the 95th percentile latency in
milliseconds (nearest-rank method) over successful requests (status below 500), counting each request once.
Write p95.csv with header "endpoint,p95_ms", one row per endpoint sorted by endpoint, integer milliseconds."""
def d_metrics_check(ws):
    exp = _exp("d_metrics")
    try: rows = list(csv.reader(open(os.path.join(ws, "p95.csv"))))
    except Exception as e: return 0.0, [f"csv unreadable: {e}"]
    got = {r[0]: r[1].strip() for r in rows[1:] if len(r) == 2}
    return frac([("header", rows[:1] == [["endpoint", "p95_ms"]])] + [(e, got.get(e) == str(v)) for e, v in exp.items()]
                + [("order", [r[0] for r in rows[1:]] == sorted(exp))])

# ---------------------------------------------------------------- data: multi-currency ledger (held-out2)
def d_ledger_setup(ws):
    rnd = random.Random(37); accts = ["ACME", "Globex", "Initech", "Umbrella"]
    rates = {"USD": Decimal("1"), "EUR": Decimal("1.0835"), "GBP": Decimal("1.2710")}
    bal = {a: Decimal("0") for a in accts}
    for month in ("09", "10"):
        out = io.StringIO(); wr = csv.writer(out, lineterminator="\r\n")
        wr.writerow(["date", "account", "description", "amount", "currency"]); total = Decimal("0")
        for k in range(70):
            a = rnd.choice(accts); cur = rnd.choice(["USD", "USD", "EUR", "GBP"])
            amt = Decimal(rnd.randint(-90000, 250000)) / 100
            if k % 9 == 0: amt = Decimal(rnd.randint(1, 400)) * 2 + Decimal("0.5")    # x.50 * rate -> half-cent cases
            s = f"{abs(amt):,.2f}" if abs(amt) >= 1000 else f"{abs(amt):.2f}"
            s = (f"({s})" if rnd.random() < 0.5 else f"-{s}") if amt < 0 else s
            wr.writerow([f"2026-{month}-{k % 28 + 1:02d}", a, f"item {k}", s, cur])
            bal[a] += (amt * rates[cur]).quantize(Decimal("0.01"), ROUND_HALF_UP); total += amt
        wr.writerow(["TOTAL", "", "", f"{total:.2f}", "mixed"])
        w(ws, f"ledger/2026-{month}.csv", "\ufeff" + out.getvalue(), newline="")
    # the EU entity's export: ';'-separated, decimal comma, '.' thousands, trailing minus for negatives
    out = io.StringIO(); wr = csv.writer(out, delimiter=";", lineterminator="\r\n")
    wr.writerow(["Datum", "Konto", "Text", "Betrag", "Währung"])
    for k in range(40):
        a = rnd.choice(accts); cur = rnd.choice(["EUR", "EUR", "GBP", "USD"])
        amt = Decimal(rnd.randint(-300000, 400000)) / 100
        if k % 7 == 0: amt = Decimal(rnd.randint(1, 400)) * 2 + Decimal("0.5")
        s = f"{abs(amt):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + ("-" if amt < 0 else "")
        wr.writerow([f"{k % 28 + 1:02d}.10.2026", a, f"Posten {k}", s, cur])
        bal[a] += (amt * rates[cur]).quantize(Decimal("0.01"), ROUND_HALF_UP)
    w(ws, "ledger/2026-10-eu.csv", out.getvalue(), newline="")
    w(ws, "ledger/rates.json", json.dumps({"EUR": "1.0835", "GBP": "1.2710"}, indent=1) + "\n")
    return {a: f"{v:.2f}" for a, v in bal.items()}
D_LEDGER = """ledger/ holds monthly transaction exports (exported from our accounting tool) and rates.json (USD per 1 unit of
currency). Convert every transaction to USD, rounding each converted transaction to cents (half up), then sum per
account. Write balances.csv with header "account,balance_usd", one row per account sorted by account, 2 decimals."""
def d_ledger_check(ws):
    exp = _exp("d_ledger")
    try: rows = list(csv.reader(open(os.path.join(ws, "balances.csv"), encoding="utf-8-sig")))
    except Exception as e: return 0.0, [f"csv unreadable: {e}"]
    got = {r[0].strip(): r[1].strip() for r in rows[1:] if len(r) == 2}
    return frac([("header", rows[:1] == [["account", "balance_usd"]])] + [(a, got.get(a) == v) for a, v in exp.items()]
                + [("only accounts", set(got) == set(exp))])

# ---------------------------------------------------------------- code: LRU memo (held-out2)
LRU = '''"""Small LRU cache.

LRU(maxsize): get(key, default=None) returns the value and marks the key most recently used; put(key, value)
inserts or updates (an update also marks it most recently used and never evicts anything); when an insert makes
the cache exceed maxsize the least recently used key is evicted. maxsize=0 means nothing is ever stored.

@memoize(maxsize) caches a function on its positional AND keyword arguments (keyword order does not matter).
"""
 LRU = class:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.keys = []
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key, value):
        self.keys.append(key)
        self.data[key] = value
        if len(self.keys) > self.maxsize:
            old = self.keys.pop(0)
            self.data.pop(old, None)

    def __len__(self):
        return len(self.data)


def memoize(maxsize=128):
    def deco(fn):
        cache = LRU(maxsize)
        def wrapper(*args, **kwargs):
            hit = cache.get(args)
            if hit is not None:
                return hit
            val = fn(*args, **kwargs)
            cache.put(args, val)
            return val
        return wrapper
    return deco
'''
LRU_TESTS = '''import unittest
from memo.lru import LRU


class T(unittest.TestCase):
    def test_eviction(self):
        c = LRU(2); c.put("a", 1); c.put("b", 2); c.put("c", 3)
        self.assertIsNone(c.get("a")); self.assertEqual(c.get("c"), 3)

    def test_get_refreshes(self):
        c = LRU(2); c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
        self.assertEqual(c.get("a"), 1); self.assertIsNone(c.get("b"))


if __name__ == "__main__":
    unittest.main()
'''
LRU_HIDDEN = """
import sys; sys.path.insert(0, '.')
from memo.lru import LRU, memoize
R = {}
c = LRU(2); c.put('a', 1); c.put('b', 2); c.put('a', 10); c.put('c', 3)
R['update_no_evict'] = (c.get('a'), c.get('b'), len(c)) == (10, None, 2)
c = LRU(0); c.put('x', 1); R['zero'] = (c.get('x'), len(c)) == (None, 0)
n = []
@memoize(8)
def f(x, scale=1): n.append(1); return x * scale
R['kwargs'] = (f(2, scale=3), f(2, scale=5)) == (6, 10)
@memoize(8)
def g(a=0, b=0): n.append(1); return a - b
g(a=1, b=2); k = len(n); g(b=2, a=1); R['kw_order'] = len(n) == k
@memoize(8)
def z(x): n.append(1); return 0 if x else None
z(0); z(0); k = len(n); z(0); R['falsy_cached'] = len(n) == k
import json; print(json.dumps(R))
"""
def c_cache_setup(ws):
    w(ws, "memo/__init__.py", ""); w(ws, "memo/lru.py", LRU); w(ws, "tests/__init__.py", ""); w(ws, "tests/test_lru.py", LRU_TESTS)
C_CACHE = """tests/ fail (python3 -m unittest discover -s tests -t .). Fix memo/lru.py so the tests pass and the module
behaves as its docstring specifies."""
def c_cache_check(ws):
    u = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ws, capture_output=True, text=True, timeout=60)
    h = subprocess.run([sys.executable, "-c", LRU_HIDDEN], cwd=ws, capture_output=True, text=True, timeout=60)
    try: R = json.loads(h.stdout.strip().splitlines()[-1])
    except Exception: R = {}
    keys = ["update_no_evict", "zero", "kwargs", "kw_order", "falsy_cached"]
    return frac([("unit tests", u.returncode == 0)] + [(k, R.get(k) is True) for k in keys])

# ---------------------------------------------------------------- code: slugify (held-out2)
SLUG = '''"""URL slugs.

slugify(text, max_len=None): lowercase; accented letters lose their accents (e.g. "Crème" -> "creme"); every run of
characters that are not a-z/0-9 becomes a single "-"; no leading/trailing "-". With max_len the slug is cut to at
most max_len characters at a "-" boundary when possible (never ending in "-"). An empty result becomes "n-a".
"""
import re


def slugify(text, max_len=None):
    s = text.lower()
    s = re.sub(r"[^a-z0-9]", "-", s)
    if max_len:
        s = s[:max_len]
    return s
'''
SLUG_TESTS = '''import unittest
from text.slug import slugify


class T(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_runs(self):
        self.assertEqual(slugify("a -- b"), "a-b")


if __name__ == "__main__":
    unittest.main()
'''
SLUG_HIDDEN = """
import sys, json, re; sys.path.insert(0, '.')
from text.slug import slugify
cases = {"accents": (("Crème Brûlée à la carte",), "creme-brulee-a-la-carte"), "edges": (("  --Hi!--  ",), "hi"),
         "empty": (("!!!",), "n-a"), "cut": (("alpha beta gamma", 12), "alpha-beta"), "cut_nohyphen": (("abcdefghij", 4), "abcd"),
         "unicode_word": (("Øresund straße",), None)}
R = {}
for k, (a, want) in cases.items():
    try: got = slugify(*a)
    except Exception as e: got = repr(e)
    R[k] = (got == want) if want is not None else bool(re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', got or ''))
print(json.dumps(R))
"""
def c_slug_setup(ws):
    w(ws, "text/__init__.py", ""); w(ws, "text/slug.py", SLUG); w(ws, "tests/__init__.py", ""); w(ws, "tests/test_slug.py", SLUG_TESTS)
C_SLUG = """tests/ fail (python3 -m unittest discover -s tests -t .). Fix text/slug.py so the tests pass and slugify
behaves as its docstring specifies."""
def c_slug_check(ws):
    u = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ws, capture_output=True, text=True, timeout=60)
    h = subprocess.run([sys.executable, "-c", SLUG_HIDDEN], cwd=ws, capture_output=True, text=True, timeout=60)
    try: R = json.loads(h.stdout.strip().splitlines()[-1])
    except Exception: R = {}
    keys = ["accents", "edges", "empty", "cut", "cut_nohyphen", "unicode_word"]
    return frac([("unit tests", u.returncode == 0)] + [(k, R.get(k) is True) for k in keys])

TASKS = {
    "d_reconcile": ("data", D_RECONCILE, d_reconcile_setup, d_reconcile_check),
    "d_logs": ("data", D_LOGS, d_logs_setup, d_logs_check),
    "c_pricing": ("code", C_PRICING, c_pricing_setup, c_pricing_check),
    "c_crlf": ("code", C_CRLF, c_crlf_setup, c_crlf_check),
    "d_budget": ("data", D_BUDGET, d_budget_setup, d_budget_check),     # held-out
    "c_rename": ("code", C_RENAME, c_rename_setup, c_rename_check),     # held-out
    "d_sessions": ("data", D_SESSIONS, d_sessions_setup, d_sessions_check),  # held-out
    "c_duration": ("code", C_DURATION, c_duration_setup, c_duration_check),  # held-out
    "d_metrics": ("data", D_METRICS, d_metrics_setup, d_metrics_check),  # held-out2
    "d_ledger": ("data", D_LEDGER, d_ledger_setup, d_ledger_check),      # held-out2
    "c_cache": ("code", C_CACHE, c_cache_setup, c_cache_check),          # held-out2
    "c_slug": ("code", C_SLUG, c_slug_setup, c_slug_check),              # held-out2
}
STREAM = ["d_reconcile", "c_pricing", "d_logs", "c_crlf"]
HELDOUT = ["d_budget", "c_rename", "d_sessions", "c_duration"]
HELDOUT2 = ["d_metrics", "d_ledger", "c_cache", "c_slug"]

_EXP = {}
def _exp(tid):
    if tid not in _EXP:
        import tempfile, shutil
        d = tempfile.mkdtemp(prefix="h2exp-"); _EXP[tid] = TASKS[tid][2](d); shutil.rmtree(d)
    return _EXP[tid]

def setup(tid, ws):
    os.makedirs(ws, exist_ok=True); TASKS[tid][2](ws)

def check(tid, ws):
    try: return TASKS[tid][3](ws)
    except Exception as e: return 0.0, [f"checker error: {e}"]
