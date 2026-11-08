#!/usr/bin/env python3
"""Run real judge-sessions inference and measure only the daemon started for this model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from common import DEFAULT_CKPT


def rss(pid: int) -> dict:
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "RssAnon:", "RssFile:")):
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0])
    return fields


def daemon_pids(onnx: Path) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            args = [x.decode() for x in argv if x]
            if "laya-serve" not in args:
                continue
            if str(onnx) in args and "--onnx" in args:
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            pass
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omt", type=Path, default=Path("/home/kaneki/fleet-split/omt/target/shared/release/omt"))
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.onnx = args.onnx.resolve()
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    before = set(daemon_pids(args.onnx))
    if before:
        raise RuntimeError(f"refusing to reuse/stop pre-existing daemon(s) for this exact model: {before}")
    command = [str(args.omt), "judge-sessions", str(args.sessions), "--out", str(args.scores),
               "--ckpt", str(args.ckpt), "--onnx", str(args.onnx), "--threads", str(args.threads)]
    if args.limit:
        command += ["--limit", str(args.limit)]
    started = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    own_pid, ready_rss, samples = None, None, []
    while proc.poll() is None:
        matches = [p for p in daemon_pids(args.onnx) if p not in before]
        if len(matches) > 1:
            proc.kill()
            raise RuntimeError(f"ambiguous daemon pids: {matches}")
        if matches:
            own_pid = matches[0]
            try:
                sample = {"t": time.time() - started, **rss(own_pid)}
                samples.append(sample)
                cmdline = Path(f"/proc/{own_pid}/cmdline").read_bytes().split(b"\0")
                if "--sock" in cmdline:
                    sock = Path(cmdline[cmdline.index(b"--sock") + 1].decode())
                    log = sock.with_suffix(".log")
                    if ready_rss is None and log.exists() and " ready in " in log.read_text():
                        ready_rss = sample
            except FileNotFoundError:
                pass
        time.sleep(.2)
    stdout, stderr = proc.communicate()
    if own_pid is not None:
        try:
            samples.append({"t": time.time() - started, **rss(own_pid)})
            # This PID was first observed after our unique-model CLI invocation and was not pre-existing.
            os.kill(own_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if proc.returncode:
        raise RuntimeError(f"judge-sessions failed ({proc.returncode}):\n{stdout}\n{stderr}")
    rows = [json.loads(x) for x in args.scores.read_text().splitlines() if x.strip()]
    latencies = [r["latency_ms"] for r in rows]
    result = {
        "onnx": str(args.onnx), "onnx_bytes": args.onnx.stat().st_size, "daemon_pid": own_pid,
        "episodes": len(rows), "ready_rss": ready_rss,
        "peak_sampled_rss_kb": max((s["VmRSS"] for s in samples), default=None),
        "peak_hwm_rss_kb": max((s["VmHWM"] for s in samples), default=None),
        "final_rss": samples[-1] if samples else None,
        "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else None,
                       "p50": sorted(latencies)[len(latencies) // 2] if latencies else None,
                       "max": max(latencies) if latencies else None},
        "stderr_tail": stderr.splitlines()[-20:],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
