#!/usr/bin/env python3
"""Run one child process and terminate it before its RSS exceeds the task budget."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rss-kb", type=int, default=3_800_000)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("missing command")
    proc = subprocess.Popen(command)
    peak = 0
    while proc.poll() is None:
        try:
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    peak = max(peak, int(line.split()[1]))
        except FileNotFoundError:
            pass
        if peak > args.rss_kb:
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait()
            raise RuntimeError(f"child {proc.pid} exceeded {args.rss_kb} KB RSS (observed {peak} KB)")
        time.sleep(.1)
    print(f"BOUNDED_RSS_KB {peak}", flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
