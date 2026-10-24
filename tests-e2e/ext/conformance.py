#!/usr/bin/env python3
"""Extension conformance: load every real user extension (copied into a temp HOME) into omp (TS, oracle) and
omp-rs, drive the same scripted session against the same mock provider, and diff the observable side effects:

  - probe.jsonl      every extension event delivered (type, payload keys, ctx facts), order preserved
  - hook_trace       per-handler trace written by the extensions' own lib/hook-trace.ts (action: none/block/replace/inject)
  - file tree        files created under the temp HOME (excluding the copied extensions / agent dir noise)
  - shim logs        every call to the PATH-shimmed herdr (never the real binary)

Modes: print (`-p`), rpc (`--mode rpc`: prompt, /new, compact).  Usage:
  conformance.py [--rs BIN] [--ts BIN] [--mode print|rpc|all] [--out DIR]
Exit 1 when an unexplained diff remains (EXPLAINED below documents the allowed ones)."""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tui"))
import mock_provider  # noqa: E402
import run as tui_run  # noqa: E402

REPO = HERE.parent.parent
EXT_SRC = Path.home() / ".omp/agent/extensions"
SHIM = """#!/bin/sh
echo "$(basename "$0") $@" >> "$HOME/shim.log"
case "$1 $2" in
  "pane get") echo '{"result":{"pane":{"pane_id":"w9:fake"}}}' ;;
esac
exit 0
"""

# Documented, accepted differences (substring of the diff line -> reason).
EXPLAINED: dict[str, str] = {
    "pane process-info": "omp-rs herdr reporter probes once whether the pane is owned by TS omp (herdr:omp) and stays silent there (fixes the rejected pane.report_agent storm, e01e205); TS omp has no such probe.",
    # TS passes the live AbortSignal on session_stop; it has no JSON form on the host wire.
    "session_before_compact keys=": "AbortSignal `signal` key (not serializable across the host boundary)",
    "session_stop keys=": "AbortSignal `signal` key (not serializable across the host boundary)",
    # TS-internal state (sqlite caches, logs), not extension side effects.
    ".db": "TS-internal sqlite state",
    ".omp/logs/": "TS-internal log file",
    # omp-rs reports pane state natively through the herdr CLI (herdr.rs); TS does it only via the
    # herdr-omp-agent-state extension (socket API, compared under `herdr_api`).
    "--source omp-rs": "omp-rs native herdr reporter (rs-only, different source id)",
    # TS TUI-internal state, not extension side effects (rs keeps its n-gram predictor under agent/predict).
    ".omp/agent/last-changelog-version": "TS TUI state",
    ".omp/agent/terminal-sessions/": "TS TUI state",
    ".omp/run/daemons/": "TS text-predict daemon files",
    ".omp/agent/predict/": "omp-rs n-gram predictor state",
    # TS passes the live AbortSignal on session_before_tree (same as session_before_compact/session_stop).
    "session_before_tree keys=": "AbortSignal `signal` key (not serializable across the host boundary)",
}
# `events` are compared as an ordered sequence (the vendored pa-agent loop follows TS agent-loop.ts order).


class FakeHerdr:
    """Unix-socket stand-in for the herdr API (herdr-omp-agent-state talks to it directly)."""

    def __init__(self, path: str):
        import socket, threading
        self.path, self.requests = path, []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path); self.srv.listen(16)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import threading
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        buf = b""
        try:
            while b"\n" not in buf:
                d = c.recv(65536)
                if not d:
                    return
                buf += d
            self.requests.append(buf.split(b"\n")[0].decode())
            c.sendall(b'{"id":"x","result":{}}\n')
        finally:
            c.close()

    def lines(self) -> list[str]:
        out = []
        for r in self.requests:
            try:
                d = json.loads(r)
            except ValueError:
                continue
            p = d.get("params", d)
            out.append(f"{d.get('method')} state={p.get('state')} msg={p.get('message')} source={p.get('source')} agent={p.get('agent')}")
        return out

    def close(self):
        self.srv.close()


def copy_extensions(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in EXT_SRC.iterdir():
        if p.name.startswith(".") or ".bak" in p.name:
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, ignore=shutil.ignore_patterns("*.bak*", ".wsckpt", ".*"))
        else:
            shutil.copy2(p, dst / p.name)
    shutil.copy2(HERE / "probe-all-events.ts", dst / "zz-conformance-probe.ts")


def prepare(client: str, port: int) -> tuple[Path, dict[str, str]]:
    root = Path(tempfile.mkdtemp(prefix=f"extconf-{client}-"))
    env = tui_run.make_home(root, port, "compaction:\n  keepRecentTokens: 150\n  reserveTokens: 2000\nretry:\n  baseDelayMs: 200\n")
    agent = root / ".omp/agent"
    copy_extensions(agent / "extensions")
    bindir = root / "shimbin"
    bindir.mkdir()
    for n in ("herdr", "broker", "ckpt"):
        (bindir / n).write_text(SHIM)
        (bindir / n).chmod(0o755)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env.update({"HERDR_SOCKET_PATH": str(root / "herdr.sock"), "HERDR_ENV": "1", "HERDR_BIN_PATH": str(bindir / "herdr"), "HERDR_PANE_ID": "w9:fake",
                "OMP_HOME": str(root / ".omp"), "OMP_PKG_ROOT": str(Path.home() / ".bun/install/global/node_modules/@oh-my-pi/pi-coding-agent"), "HOOK_TRACE": "1", "OMP_RS_HOOK_TRACE": "1", "OMP_NO_TITLE": "1"})
    return root, env


def run_print(binary: str, env: dict[str, str], extra: list[str], model: str = "tools") -> str:
    p = subprocess.run([binary, "-p", "--model", f"mock/{model}", *extra, "do the scripted tool run"], env=env,
                       cwd=env["E2E_WORK"], capture_output=True, text=True, timeout=240)
    return p.stdout + p.stderr


def run_approval_rpc(binary: str, env: dict[str, str]) -> str:
    """rpc with `--approval-mode always-ask`: approve the first prompt, deny the second, then cancel the rest."""
    import select
    proc = subprocess.Popen([binary, "--mode", "rpc", "--model", "mock/tools", "--approval-mode", "always-ask"], env=env, cwd=env["E2E_WORK"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: list[str] = []
    answers = ["Approve", "Deny"]
    buf = ""
    deadline = time.time() + 120
    sent = False
    while time.time() < deadline:
        r, _, _ = select.select([proc.stdout], [], [], 0.2)
        if not r:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        out.append(line)
        if '"type":"ready"' in line and not sent:
            proc.stdin.write(json.dumps({"type": "prompt", "message": "do the scripted tool run", "id": "p1"}) + "\n"); proc.stdin.flush(); sent = True
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "extension_ui_request" and d.get("method") == "select":
            ans = answers.pop(0) if answers else "Deny"
            proc.stdin.write(json.dumps({"type": "extension_ui_response", "id": d["id"], "value": ans}) + "\n"); proc.stdin.flush()
        if d.get("type") == "prompt_result":
            break
    time.sleep(1)
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
    return "".join(out)


def ui_prompts(log: str) -> list[str]:
    rows = []
    for line in log.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "extension_ui_request" and d.get("method") == "select":
            rows.append(f"select title={d.get('title')!r} options={d.get('options')}")
    return rows


def run_retry(binary: str, env: dict[str, str], port: int) -> str:
    import urllib.request
    urllib.request.urlopen(f"http://127.0.0.1:{port}/reset", timeout=5).read()
    return run_print(binary, env, [], "retry")


def run_rpc(binary: str, env: dict[str, str]) -> str:
    cmds = [{"type": "prompt", "message": "do the scripted tool run " + "filler words to grow the context. " * 3000, "id": "p1"}]
    proc = subprocess.Popen([binary, "--mode", "rpc", "--model", "mock/tools"], env=env, cwd=env["E2E_WORK"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: list[str] = []

    def send(c):
        proc.stdin.write(json.dumps(c) + "\n"); proc.stdin.flush()

    import select
    def pump(until: str, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([proc.stdout], [], [], 0.2)
            if r:
                line = proc.stdout.readline()
                if not line:
                    return False
                out.append(line)
                if until in line:
                    return True
        return False

    pump('"type":"ready"', 30)
    send(cmds[0]); pump('"type":"prompt_result"', 120); time.sleep(1)
    send({"type": "prompt", "message": "second prompt " + "more filler. " * 3000, "id": "p2"}); pump('"id":"p2","status"', 60); time.sleep(2)
    send({"type": "compact", "id": "c1"}); pump('"id":"c1"', 60); time.sleep(1)
    # fork/branch at the first user message of the transcript, a bash command and a tree navigation
    send({"type": "get_branch_messages", "id": "g1"}); pump('"id":"g1"', 30)
    entry = next((m.get("entryId") for l in out if '"id":"g1"' in l for m in (json.loads(l).get("data", {}).get("messages") or [])), None)
    if entry:
        send({"type": "fork", "entryId": entry, "id": "f1"}); pump('"id":"f1"', 30); time.sleep(1)
        send({"type": "get_branch_messages", "id": "g2"}); pump('"id":"g2"', 30)
        entry2 = next((m.get("entryId") for l in out if '"id":"g2"' in l for m in (json.loads(l).get("data", {}).get("messages") or [])), None)
        if entry2:
            send({"type": "branch", "entryId": entry2, "id": "b1"}); pump('"id":"b1"', 30); time.sleep(1)
    send({"type": "bash", "command": "echo probe", "id": "sh1"}); pump('"id":"sh1"', 30); time.sleep(1)
    send({"type": "prompt", "message": "third prompt", "id": "p3"}); pump('"id":"p3","status"', 60); time.sleep(2)
    send({"type": "prompt", "message": "/probe-tree", "id": "t1"}); pump('"id":"t1"', 30); time.sleep(1)
    send({"type": "new_session", "id": "n1"}); pump('"id":"n1"', 30); time.sleep(1)
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
    return "".join(out)


def run_tui(binary: str, env: dict[str, str]) -> str:
    from driver import Term
    t = Term([binary, "--model", "mock/tools"], env=env, cwd=env["E2E_WORK"])
    try:
        tui_run.ready(t, 90)
        tui_run.submit(t, "do the scripted tool run")
        t.wait_for(r"E2E-TOOLS-DONE", 120)
        t.settle(1.5, 30)
        tui_run.submit(t, "/new")
        t.settle(2.0, 30)
        time.sleep(2.0)
        t.key("ctrl-c"); t.pump(0.3); t.key("ctrl-c")
        end = time.time() + 20
        while time.time() < end and t.proc.poll() is None:
            t.pump(0.1)
        return t.transcript()
    finally:
        t.close()


def norm(s: str, root: Path) -> str:
    s = s.replace(str(root), "<HOME>")
    s = re.sub(r"extconf-(ts|rs)-[A-Za-z0-9_]+", "<HOME>", s)
    s = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{14,}", "<ID>", s)
    s = re.sub(r"--seq \d+", "--seq N", s)
    return s


def probe_rows(root: Path) -> list[dict]:
    f = root / "probe.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def probe_events(root: Path) -> list[str]:
    return [f"{d['ev']} role={d.get('role')} tool={d.get('tool')} reason={d.get('reason')} source={d.get('source')}" for d in probe_rows(root)]


def probe_keys(root: Path) -> list[str]:
    return sorted({f"{d['ev']} keys={','.join(d['keys'])}" for d in probe_rows(root)})


FIXED_CTX = ("hasUI", "kind", "sid", "sfile", "cwd", "mode", "model", "jobs")


def probe_ctx(root: Path) -> list[str]:
    rows = set()
    for d in probe_rows(root):
        c = d["ctx"]
        rows.add(f"{d['ev']} " + " ".join(f"{k}={c.get(k)}" for k in FIXED_CTX) + f" usage={c.get('usage')} branch={c.get('branch')}")
    return sorted(rows)


def hook_trace(root: Path) -> list[str]:
    f = root / ".omp/state/hook_trace.jsonl"
    rows = []
    if f.exists():
        for line in f.read_text().splitlines():
            d = json.loads(line)
            rows.append(f"{d['ext']} {d['event']} tool={d.get('tool')} action={d['action']} type={d.get('type')}")
    return rows


def tree(root: Path) -> list[str]:
    skip = (".bun", ".cache", ".omp-rs", ".local", "extensions", "sessions", "shimbin", "work", "models.yml", "config.yml", "probe.jsonl", "shim.log")
    rows = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() or rel.parts[0] in skip or "sessions" in rel.parts or rel.parts[:2] == (".omp", "agent") and rel.name in ("models.yml", "config.yml"):
            continue
        if rel.parts[0] == "extensions" or "extensions" in rel.parts[:3]:
            continue
        rows.append(str(rel))
    return rows


def shim_log(root: Path) -> list[str]:
    f = root / "shim.log"
    return norm(f.read_text(), root).splitlines() if f.exists() else []


def collect(root: Path) -> dict[str, list[str]]:
    return {"events": probe_events(root), "keys": probe_keys(root), "ctx": probe_ctx(root),
            "hook_trace": hook_trace(root), "files": tree(root), "shim": shim_log(root)}


def dedupe_runs(rows: list[str]) -> list[str]:
    out = []
    for r in rows:
        if not out or out[-1] != r:
            out.append(r)
    return out


def diff(a: list[str], b: list[str]) -> list[str]:
    import difflib
    return [l for l in difflib.unified_diff(a, b, "omp", "omp-rs", lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", default=str(REPO / "target/shared/release/omt"))
    ap.add_argument("--ts", default=shutil.which("omp") or "omp")
    ap.add_argument("--mode", default="all", choices=["print", "rpc", "tui", "guards", "approval", "approval-rpc", "retry", "autocompact", "all"])
    ap.add_argument("--out", default=str(REPO.parent / "runs/omp-rust/ext-conformance" / time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    srv = mock_provider.serve(0); port = srv.server_address[1]
    modes = ["print", "rpc", "tui", "guards", "approval", "approval-rpc", "retry", "autocompact"] if args.mode == "all" else [args.mode]
    bad = 0
    for mode in modes:
        res: dict[str, dict[str, list[str]]] = {}
        for client, binary in (("ts", args.ts), ("rs", args.rs)):
            root, env = prepare(client, port)
            herdr = FakeHerdr(env["HERDR_SOCKET_PATH"])
            try:
                log = {"print": lambda: run_print(binary, env, []), "guards": lambda: run_print(binary, env, [], "guards"), "rpc": lambda: run_rpc(binary, env), "tui": lambda: run_tui(binary, env),
                       "retry": lambda: run_retry(binary, env, port), "autocompact": lambda: run_print(binary, env, [], "full"),
                       "approval": lambda: run_print(binary, env, ["--approval-mode", "always-ask"]), "approval-rpc": lambda: run_approval_rpc(binary, env)}[mode]()
                (out / f"{mode}-{client}.log").write_text(norm(log, root))
                time.sleep(1.0)
                res[client] = {k: [norm(x, root) for x in v] for k, v in collect(root).items()}
                res[client]["herdr_api"] = herdr.lines()
                res[client]["ui_prompts"] = ui_prompts(log)
            finally:
                herdr.close()
                shutil.rmtree(root, ignore_errors=True)
        for client in res:
            for kind, rows in res[client].items():
                (out / f"{mode}-{client}.{kind}.txt").write_text("\n".join(rows) + "\n")
        for kind in ("events", "keys", "ctx", "hook_trace", "files", "shim", "herdr_api", "ui_prompts"):
            a, b = res["ts"][kind], res["rs"][kind]
            if kind in ("hook_trace", "keys", "ctx", "herdr_api"):  # concurrent handlers / sets / order-insensitive
                a, b = sorted(a), sorted(b)
            else:
                a, b = dedupe_runs(a), dedupe_runs(b)
            d = [l for l in diff(a, b) if not any(k in l for k in EXPLAINED)]
            if mode in ("retry", "autocompact"):
                # Forwarding checks (not a TS diff): TS retries 5xx inside the provider transport (no
                # `auto_retry_*` is visible) and runs a skipped pre-prompt compaction pass + other
                # compaction-flow details omp-rs does not mirror, so only delivery to extensions is asserted.
                d = []
                if kind == "events":
                    want = ("auto_retry_start", "auto_retry_end") if mode == "retry" else ("auto_compaction_start", "auto_compaction_end")
                    for ev in want:
                        got = [x for x in b if x.startswith(ev)]
                        print(f"[{mode}] {ev}: rs={len(got)}")
                        if not got:
                            d.append(f"{mode}: omp-rs delivered no {ev} to the extension")
                if kind == "keys":
                    for ev, keys in (("auto_retry_start", "attempt,delayMs,errorMessage,maxAttempts,type"), ("auto_compaction_start", "action,reason,type")):
                        if mode == ("retry" if ev.startswith("auto_retry") else "autocompact") and f"{ev} keys={keys}" not in b:
                            d.append(f"{mode}: {ev} payload keys differ from TS ({keys})")
            if mode.startswith("approval") and kind == "events":
                n_req = len([x for x in a if x.startswith("tool_approval_requested")])
                print(f"[{mode}] tool_approval_requested: ts={n_req} rs={len([x for x in b if x.startswith('tool_approval_requested')])}")
                if not n_req:
                    d.append("approval scenario produced no approval request under omp (oracle) — scenario is vacuous")
            if mode == "guards" and kind == "hook_trace":
                blocks = [x for x in a if "action=block" in x]
                print(f"[guards] blocked tool calls: ts={len(blocks)} rs={len([x for x in b if 'action=block' in x])}")
                if not blocks:
                    d.append("guards scenario produced no blocked call under omp (oracle) — scenario is vacuous")
            (out / f"{mode}-{kind}.diff").write_text("\n".join(d) + "\n")
            print(f"[{mode}] {kind}: ts={len(a)} rs={len(b)} diff={len(d)}")
            for l in d[:25]:
                print("   ", l[:200])
            bad += len(d)
    print(f"UNEXPLAINED DIFFS: {bad}\nartifacts: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
