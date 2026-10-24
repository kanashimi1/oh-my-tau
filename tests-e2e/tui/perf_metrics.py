#!/usr/bin/env python3
"""Pi-Bolt-style client metrics for omp vs omp-rs, on the e2e harness:
ready-to-type (spawn → first interactive frame in a PTY), own memory
(Private_Dirty from smaps_rollup, Pi-Bolt's metric) and CPU of all threads
(schedstat) while streaming Markdown answers of 5k/20k/60k characters from
Pi-Bolt's fake_model_stress.py (`SCENARIO md:<n>`, 24 chars per event).

usage: perf_metrics.py [--pibolt /tmp/pi-bolt] [--pace-ms 20] [--sizes 5000,20000,60000] name=binary ...
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from driver import Term  # noqa: E402
import run  # noqa: E402


def cpu_ms(pid: int) -> float:
    total = 0
    for tid in os.listdir(f"/proc/{pid}/task"):
        with contextlib.suppress(OSError):
            total += int(open(f"/proc/{pid}/task/{tid}/schedstat").read().split()[0])
    return total / 1e6


def own_mb(pid: int) -> dict[str, float]:
    out = {}
    with contextlib.suppress(OSError):
        for line in open(f"/proc/{pid}/smaps_rollup"):
            key, value = line.split(":", 1)
            if key in ("Rss", "Private_Dirty"):
                out[{"Rss": "rss", "Private_Dirty": "own"}[key]] = round(int(value.split()[0]) / 1024, 1)
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def home_for(port: int) -> dict[str, str]:
    root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
    env = run.make_home(root, port)
    agent = Path(env["PI_CODING_AGENT_DIR"])
    (agent / "models.yml").write_text(
        "providers:\n  fake:\n"
        f"    baseUrl: http://127.0.0.1:{port}/v1\n    apiKey: fake\n    api: openai-completions\n"
        "    models:\n      - id: fake-model\n        name: fake-model\n        contextWindow: 200000\n        maxTokens: 8192\n"
    )
    return env


def measure(name: str, binary: str, port: int, sizes: list[int]) -> None:
    env = home_for(port)
    root = Path(env["HOME"])
    started = time.perf_counter()
    t = Term([binary, "--model", "fake/fake-model", "--no-session"], env=env, cwd=env["E2E_WORK"], rows=48, cols=160)
    try:
        if not t.wait_for(run.READY_RE, 60, in_transcript=False):
            print(f"{name}: TUI never became interactive")
            return
        ready_ms = (time.perf_counter() - started) * 1000
        t.settle(1.0, 15)
        mem0 = own_mb(t.proc.pid)
        print(f"{name:8} ready-to-type {ready_ms:7.0f} ms   idle own {mem0.get('own', 0):6.1f} MB  rss {mem0.get('rss', 0):6.1f} MB", flush=True)
        idle_cpu0, idle_t0 = cpu_ms(t.proc.pid), time.perf_counter()
        t.pump(5.0)
        idle = (cpu_ms(t.proc.pid) - idle_cpu0) / (time.perf_counter() - idle_t0)
        print(f"{name:8} idle CPU {idle:6.2f} ms/s", flush=True)
        for n, size in enumerate(sizes, 1):
            cpu0, t0 = cpu_ms(t.proc.pid), time.perf_counter()
            # Trailing word: a prompt ending in a path opens path completion
            # and Enter would accept it instead of submitting.
            run.submit(t, f"SCENARIO md:{size} CWD {env['E2E_WORK']} go")
            if not t.wait_for(rf"Done: md, answer {n}\.", float(os.environ.get("PERF_ANSWER_TIMEOUT", "900"))):
                print(f"{name}: no answer of {size}; screen tail:")
                print("\n".join(l for l in t.transcript().splitlines() if l.strip())[-1500:])
                return
            wall = time.perf_counter() - t0
            t.settle(0.5, 10)
            used = (cpu_ms(t.proc.pid) - cpu0) / 1000
            mem = own_mb(t.proc.pid)
            print(f"{name:8} stream {size:6} chars  wall {wall:6.1f} s  cpu {used:6.2f} s  ({100 * used / wall:4.0f}% core)  own {mem.get('own', 0):6.1f} MB", flush=True)
    finally:
        t.close()
        run.reap_home_processes(root)
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pibolt", default="/tmp/pi-bolt")
    ap.add_argument("--pace-ms", default="20")
    ap.add_argument("--sizes", default="5000,20000,60000")
    ap.add_argument("builds", nargs="+")
    a = ap.parse_args()
    port = free_port()
    model = subprocess.Popen([sys.executable, f"{a.pibolt}/bench/fake_model_stress.py", str(port), "--pace-ms", a.pace_ms],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        sizes = [int(s) for s in a.sizes.split(",")]
        for spec in a.builds:
            name, binary = spec.split("=", 1)
            measure(name, binary, port, sizes)
    finally:
        model.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
