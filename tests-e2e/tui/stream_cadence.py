#!/usr/bin/env python3
"""Streaming smoothness probe: while the mock `slow` model streams small chunks
(E2E_SLOW_MS apart), record when the PTY screen content changes and report the
inter-update interval distribution for TS omp vs omp-rs.

usage: stream_cadence.py name=binary ...   (e.g. ts=omp rs=target/shared/release/omt)
"""
from __future__ import annotations

import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from driver import Term  # noqa: E402
from perf_metrics import cpu_ms  # noqa: E402
import mock_provider  # noqa: E402
import run  # noqa: E402


def measure(name: str, binary: str, port: int) -> None:
    root = Path(tempfile.mkdtemp(prefix="e2e-home-"))
    env = run.make_home(root, port)
    agent = Path(env["PI_CODING_AGENT_DIR"])
    (agent / "models.yml").write_text(
        "providers:\n  mock:\n"
        f"    baseUrl: http://127.0.0.1:{port}/v1\n    apiKey: e2e-not-a-secret\n    api: openai-completions\n"
        "    models:\n      - id: slow\n        name: e2e slow\n        contextWindow: 128000\n        maxTokens: 4096\n"
    )
    t = Term([binary, "--model", "mock/slow"], env=env, cwd=env["E2E_WORK"], rows=40, cols=120)
    try:
        if not run.ready(t):
            print(f"{name}: never ready")
            return
        i0, w0 = cpu_ms(t.proc.pid), time.perf_counter()
        t.pump(5.0)
        print(f"{name:3} idle CPU {(cpu_ms(t.proc.pid) - i0) / (time.perf_counter() - w0):6.2f} ms/s", flush=True)
        c0, w0 = cpu_ms(t.proc.pid), time.perf_counter()
        run.submit(t, "stream slowly please")
        last, stamps, raw_stamps, raw_len = None, [], [], len(t.raw)
        text_stamps, text_len = [], -1
        start = time.perf_counter()
        while time.perf_counter() - start < 60:
            t.pump(0.002)
            now = time.perf_counter()
            if len(t.raw) != raw_len:
                raw_len = len(t.raw)
                raw_stamps.append(now)
                n = sum(1 for c in t.transcript() if c.isalnum())  # revealed text only (spinner excluded)
                if n != text_len:
                    text_len = n
                    text_stamps.append(now)
            snap = "\n".join(t.screen())
            if snap != last:
                last = snap
                stamps.append(now)
            if "SLOW-DONE" in snap and now - stamps[-1] > 0.5:
                break
        # Streaming window: first update showing a word → the update showing SLOW-DONE.
        t.settle(0.3, 5)
        print(f"{name:3} stream CPU {(cpu_ms(t.proc.pid) - c0):6.0f} ms over {time.perf_counter() - w0:4.1f} s", flush=True)
        for label, ts in (("text", text_stamps), ("screen", stamps), ("bytes", raw_stamps)):
            gaps = [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
            # keep the streaming section only (drop the idle tail)
            gaps = [g for g in gaps if g < 2000]
            if len(gaps) < 5:
                print(f"{name:3} {label}: too few updates ({len(gaps)})")
                continue
            q = statistics.quantiles(gaps, n=20)
            print(f"{name:3} {label:6} updates {len(gaps) + 1:4}  median {statistics.median(gaps):6.1f} ms  "
                  f"p95 {q[18]:6.1f} ms  max {max(gaps):6.1f} ms", flush=True)
    finally:
        t.close()
        run.reap_home_processes(root)
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    server = mock_provider.serve(0)
    port = server.server_address[1]
    for spec in sys.argv[1:]:
        name, binary = spec.split("=", 1)
        measure(name, binary, port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
