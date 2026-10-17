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
    # Core TUI feature (turn-complete notification through the herdr CLI), not an extension: open gap in omp-rs.
    "notification show omp": "TS completion notification via herdr CLI; not implemented in omp-rs TUI (core feature, outside the extension surface)",
}
# `events` ordering inside a turn follows the vendored pa-agent loop (context/tool_call placement differ from
# TS agent-loop.ts); compared as an informational `events_order` row, hard-compared as a multiset.


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
    env = tui_run.make_home(root, port, "compaction:\n  keepRecentTokens: 150\n  reserveTokens: 2000\n")
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


FIXED_CTX = ("hasUI", "kind", "sid", "sfile", "cwd", "mode", "model")


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
    ap.add_argument("--rs", default=str(REPO / "target/shared/release/omp-rs"))
    ap.add_argument("--ts", default=shutil.which("omp") or "omp")
    ap.add_argument("--mode", default="all", choices=["print", "rpc", "tui", "guards", "all"])
    ap.add_argument("--out", default=str(REPO.parent / "runs/omp-rust/ext-conformance" / time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    srv = mock_provider.serve(0); port = srv.server_address[1]
    modes = ["print", "rpc", "tui", "guards"] if args.mode == "all" else [args.mode]
    bad = 0
    for mode in modes:
        res: dict[str, dict[str, list[str]]] = {}
        for client, binary in (("ts", args.ts), ("rs", args.rs)):
            root, env = prepare(client, port)
            herdr = FakeHerdr(env["HERDR_SOCKET_PATH"])
            try:
                log = {"print": lambda: run_print(binary, env, []), "guards": lambda: run_print(binary, env, [], "guards"), "rpc": lambda: run_rpc(binary, env), "tui": lambda: run_tui(binary, env)}[mode]()
                (out / f"{mode}-{client}.log").write_text(norm(log, root))
                time.sleep(1.0)
                res[client] = {k: [norm(x, root) for x in v] for k, v in collect(root).items()}
                res[client]["herdr_api"] = herdr.lines()
            finally:
                herdr.close()
                shutil.rmtree(root, ignore_errors=True)
        for client in res:
            for kind, rows in res[client].items():
                (out / f"{mode}-{client}.{kind}.txt").write_text("\n".join(rows) + "\n")
        for kind in ("events", "keys", "ctx", "hook_trace", "files", "shim", "herdr_api"):
            a, b = res["ts"][kind], res["rs"][kind]
            if kind in ("hook_trace", "keys", "ctx", "herdr_api", "events"):  # concurrent handlers / sets / order-insensitive
                a, b = sorted(a), sorted(b)
            else:
                a, b = dedupe_runs(a), dedupe_runs(b)
            d = [l for l in diff(a, b) if not any(k in l for k in EXPLAINED)]
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
