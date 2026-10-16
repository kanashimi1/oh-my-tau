#!/usr/bin/env python3
"""End-to-end TUI parity: the same scripted scenarios against TS `omp` and
`omp-rs`, each in a real PTY with an isolated HOME and a local mock provider.

Every scenario states what the user must see. A check counts against omp-rs
when it fails there; when it also fails on omp the harness itself is suspect
and the run reports it as `baseline` instead of a parity failure. Screens of
both clients are written side by side for a visual comparison.

usage: run.py [--rs BIN] [--ts BIN] [--only NAME[,NAME]] [--out DIR] [--client rs|ts|both]
exit 0 = omp-rs passes every check omp passes.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from driver import Term  # noqa: E402
import mock_provider  # noqa: E402

REPO = HERE.parent.parent
DEFAULT_RS = REPO / "target/shared/release/omp-rs"


# ── isolated client home ────────────────────────────────────────────────────
def make_home(root: Path, port: int) -> dict[str, str]:
    agent = root / ".omp" / "agent"
    (agent / "sessions").mkdir(parents=True)
    (agent / "models.yml").write_text(
        "providers:\n"
        "  mock:\n"
        f"    baseUrl: http://127.0.0.1:{port}/v1\n"
        "    apiKey: e2e-not-a-secret\n"
        "    api: openai-completions\n"
        "    models:\n"
        + "".join(f"      - id: {m}\n        name: e2e {m}\n        contextWindow: 128000\n        maxTokens: 4096\n"
                  for m in ("text", "tool", "fail"))
    )
    # No onboarding wizard / update probe / changelog: they would cover the
    # chat screen on a fresh HOME (TS omp; omp-rs ignores unknown keys).
    (agent / "config.yml").write_text(
        "startup:\n  setupWizard: false\n  checkUpdate: false\n  changelogMode: hidden\n"
    )
    work = root / "work"
    work.mkdir()
    (work / "e2e-read-me.txt").write_text("E2E-FILE-CONTENT\n")
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "USER", "SHELL", "TMPDIR")}
    return {
        **keep,
        "HOME": str(root),
        "PI_CODING_AGENT_DIR": str(agent),
        "OMP_AGENT_DIR": str(agent),
        "TERM": "xterm-256color",
        "COLORTERM": "truecolor",
        "PI_NO_DESKTOP_NOTIFY": "1",
        "OMP_NO_UPDATE_CHECK": "1",
        "PI_SKIP_VERSION_CHECK": "1",
        "E2E_WORK": str(work),
    }


# ── scenarios ───────────────────────────────────────────────────────────────
@dataclass
class Result:
    checks: dict[str, bool] = field(default_factory=dict)
    screens: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool) -> bool:
        self.checks[name] = bool(ok)
        return ok


def submit(t: Term, text: str) -> None:
    t.type(text, per_char=0.005)
    t.pump(0.2)
    t.key("enter")


USAGE_RE = r"⏱|tok/s|\d+(\.\d+)?[kKM]? tok"
ERROR_RE = r"(?i)\berror\b|prompt failed"
# Status line both clients draw once the chat is interactive: `π > ⬢ <model> > …`.
READY_RE = r"π > ⬢"


def ready(t: Term, timeout: float = 60) -> bool:
    """Wait until the chat is interactive (TS shows a splash first)."""
    ok = t.wait_for(READY_RE, timeout, in_transcript=False)
    t.settle(1.0, 15)
    return ok


def error_headers(text: str, needle: str) -> int:
    """Lines that present `needle` as an error (✗/✘ glyph or `Error:`),
    ignoring detail lines that merely repeat the provider text."""
    return sum(1 for line in text.splitlines()
               if needle in line and re.search(r"[✗✘]|(?i:error:)", line))


def sc_startup(t: Term, r: Result) -> None:
    r.check("startup: chat ready", ready(t))
    r.screens["startup"] = t.text()
    r.check("startup: model shown", "mock/text" in t.text() or "e2e text" in t.text())
    r.check("startup: no error", re.search(ERROR_RE, t.text()) is None)


def sc_two_prompts(t: Term, r: Result) -> None:
    ready(t)
    submit(t, "first e2e prompt")
    r.check("prompt1: reply shown", t.wait_for(r"E2E-REPLY-1", 90))
    t.wait_for(READY_RE, 30, in_transcript=False)
    t.settle(1.0, 30)
    r.screens["after-prompt-1"] = t.transcript()
    r.check("prompt1: no error", re.search(ERROR_RE, t.transcript()) is None)
    submit(t, "second e2e prompt")
    r.check("prompt2: reply shown", t.wait_for(r"E2E-REPLY-2", 60))
    t.settle(1.0, 30)
    tr = t.transcript()
    r.screens["after-prompt-2"] = tr
    r.check("prompt2: no error", re.search(ERROR_RE, tr) is None)
    r.check("prompts: both user messages kept", "first e2e prompt" in tr and "second e2e prompt" in tr)
    r.notes.append(f"usage rows: {len(re.findall(USAGE_RE, tr))}")


def sc_tool(t: Term, r: Result) -> None:
    ready(t)
    submit(t, "read the file")
    r.check("tool: final reply", t.wait_for(r"E2E-TOOL-DONE", 60))
    t.settle(1.0, 30)
    tr = t.transcript()
    r.screens["tool-turn"] = tr
    r.check("tool: read card names the file", "e2e-read-me.txt" in tr)
    r.check("tool: no error", re.search(ERROR_RE, tr) is None)


def sc_fail(t: Term, r: Result) -> None:
    ready(t)
    submit(t, "this will fail")
    r.check("fail: provider message shown", t.wait_for(r"E2E-PROVIDER-FAILURE", 60))
    t.settle(1.5, 30)
    tr = t.transcript()
    r.screens["provider-error"] = tr
    r.check("fail: shown once", error_headers(tr, "E2E-PROVIDER-FAILURE") == 1)
    submit(t, "after failure")
    r.check("fail: next prompt accepted", t.wait_for(r"after failure", 20))


def sc_slash(t: Term, r: Result) -> None:
    ready(t)
    before = t.transcript()
    submit(t, "/session")
    t.settle(1.0, 15)
    tr = t.transcript()
    r.screens["slash-session"] = tr
    added = tr[len(before):] if tr.startswith(before) else tr
    r.check("slash: /session prints session info", re.search(r"(?i)session", added) is not None)
    r.check("slash: no error", re.search(r"(?i)unknown command|not ported|error", added) is None)


def sc_exit(t: Term, r: Result) -> None:
    ready(t)
    t.key("ctrl-c")
    t.pump(0.2)
    t.key("ctrl-c")
    deadline = time.time() + 15
    while time.time() < deadline and t.proc.poll() is None:
        t.pump(0.1)
    r.screens["exit"] = t.text()
    r.check("exit: double Ctrl+C quits", t.proc.poll() is not None)


SCENARIOS = {
    "startup": ("text", sc_startup),
    "two_prompts": ("text", sc_two_prompts),
    "tool": ("tool", sc_tool),
    "fail": ("fail", sc_fail),
    "slash": ("text", sc_slash),
    "exit": ("text", sc_exit),
}


def reap_home_processes(root: Path) -> list[str]:
    """Stop daemons the client detached under this throwaway HOME (TS omp
    spawns e.g. text-predict). Only processes whose environment has exactly
    this HOME are touched."""
    killed = []
    needle = f"HOME={root}".encode() + b"\0"
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
            continue
        try:
            environ = (pid_dir / "environ").read_bytes() + b"\0"
        except OSError:
            continue
        if needle in environ:
            try:
                os.kill(int(pid_dir.name), signal.SIGTERM)
                killed.append(pid_dir.name)
            except OSError:
                pass
    return [f"reaped daemons {','.join(killed)}"] if killed else []


def run_client(binary: str, model: str, fn, port: int, keep: Path) -> Result:
    root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
    env = make_home(root, port)
    r = Result()
    t = Term([binary, "--model", f"mock/{model}"], env=env, cwd=env["E2E_WORK"])
    try:
        fn(t, r)
    except Exception as e:  # a crashed scenario is a failed check, not a crashed run
        r.check(f"scenario raised {type(e).__name__}", False)
        r.notes.append(repr(e))
    finally:
        code = t.close()
        r.notes.append(f"exit code {code}")
        sessions = root / ".omp" / "agent" / "sessions"
        if sessions.is_dir():
            shutil.copytree(sessions, keep / "sessions", dirs_exist_ok=True)
        r.notes.extend(reap_home_processes(root))
        shutil.rmtree(root, ignore_errors=True)
    return r


def normalize(text: str) -> list[str]:
    """Screen lines with run-specific noise masked: temp HOME names, session
    ids, numbers, the rotating startup tip, trailing blanks. Leading
    indentation is kept (layout is part of the look)."""
    out = []
    skip_tip = False
    for line in text.splitlines():
        if re.match(r"\s*Tip:", line):
            skip_tip = True
            continue
        if skip_tip and line.strip() and line.startswith(" " * 20):
            continue  # wrapped continuation of the tip
        skip_tip = False
        line = re.sub(r"e2e-home-[A-Za-z0-9_]+", "<HOME>", line)
        line = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{14,}", "<ID>", line)
        line = re.sub(r"\d+", "#", line).rstrip()
        if line.strip():
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", default=str(DEFAULT_RS))
    ap.add_argument("--ts", default=shutil.which("omp") or "omp")
    ap.add_argument("--only", default="")
    ap.add_argument("--client", default="both", choices=["rs", "ts", "both"])
    ap.add_argument("--out", default=str(REPO.parent / "runs/omp-rust/e2e-tui" / time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mock_provider.LOG = str(out / "mock.jsonl")
    srv = mock_provider.serve(0)
    port = srv.server_address[1]

    names = [n for n in SCENARIOS if not args.only or n in args.only.split(",")]
    clients = {"ts": args.ts, "rs": args.rs}
    if args.client != "both":
        clients = {args.client: clients[args.client]}
    results: dict[str, dict[str, Result]] = {}
    for name in names:
        model, fn = SCENARIOS[name]
        for tag, binary in clients.items():
            print(f"[{tag}] {name} …", flush=True)
            keep = out / name / tag
            keep.mkdir(parents=True, exist_ok=True)
            res = run_client(binary, model, fn, port, keep)
            results.setdefault(name, {})[tag] = res
            for screen, text in res.screens.items():
                (keep / f"{screen}.txt").write_text(text)

    parity_fail = 0
    lines = ["# TUI e2e: omp vs omp-rs", ""]
    for name in names:
        lines.append(f"## {name}")
        per = results[name]
        checks = sorted({c for r in per.values() for c in r.checks})
        lines.append("| check | omp | omp-rs | verdict |")
        lines.append("|---|---|---|---|")
        for c in checks:
            ts_ok = per.get("ts", Result()).checks.get(c)
            rs_ok = per.get("rs", Result()).checks.get(c)
            if rs_ok is None:
                verdict = "baseline only" if ts_ok else "baseline (omp fails)"
            elif rs_ok:
                verdict = "ok"
            elif ts_ok is False:
                verdict = "baseline (omp fails too)"
            else:
                verdict = "**FAIL**"
                parity_fail += 1
            fmt = lambda v: "-" if v is None else ("✓" if v else "✗")
            lines.append(f"| {c} | {fmt(ts_ok)} | {fmt(rs_ok)} | {verdict} |")
        for tag, r in per.items():
            if r.notes:
                lines.append(f"- {tag}: " + "; ".join(r.notes))
        if "ts" in per and "rs" in per:
            for screen in per["rs"].screens:
                a = normalize(per["ts"].screens.get(screen, ""))
                b = normalize(per["rs"].screens[screen])
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                lines.append(f"- screen `{screen}` similarity {ratio:.2f}")
                diff = difflib.unified_diff(a, b, "omp", "omp-rs", lineterm="")
                (out / name / f"{screen}.diff").write_text("\n".join(diff))
        lines.append("")
    lines.append(f"PARITY FAILURES: {parity_fail}")
    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report)
    print(report)
    print(f"\nscreens + diffs: {out}")
    srv.shutdown()
    return 1 if parity_fail else 0


if __name__ == "__main__":
    sys.exit(main())
