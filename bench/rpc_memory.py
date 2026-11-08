#!/usr/bin/env python3
"""Measure RPC core RSS while a mock provider streams many small deltas.

No network, credentials, user sessions, or extensions are used. The workload is
intentionally worst-case for full-partial ``message_update`` frames: a final
text of N bytes is emitted in small chunks, so serializing every partial value
allocates O(N^2 / chunk_size) bytes before the turn settles.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "tests-e2e" / "tui"))
import mock_provider  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", type=Path, default=REPO / "target/shared/release/omt")
    p.add_argument("--turns", type=int, default=50)
    p.add_argument("--payload-kb", type=int, default=4)
    p.add_argument("--spike-kb", type=int, help="use a larger first-turn payload to test retained spikes")
    p.add_argument(
        "--tool-turns",
        type=int,
        default=0,
        help="model turns per prompt that stream text, then call eval; 0 = plain streaming",
    )
    p.add_argument("--stream-chunk-bytes", type=int, default=16)
    p.add_argument("--vary-payload", action="store_true", help="vary each turn's size deterministically")
    p.add_argument("--out", type=Path, help="write all samples and the summary as JSON")
    p.add_argument(
        "--max-settled-growth-kb",
        type=int,
        help="fail if turn 50 anonymous RSS minus turn 1 exceeds this bound",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.turns < 2 or args.payload_kb < 1 or args.stream_chunk_bytes < 1:
        raise SystemExit("turns >= 2, payload-kb >= 1 and stream-chunk-bytes >= 1 required")

    def stream_script(model: str, messages: list[dict]):
        if model.removeprefix("mock/") == "stream":
            turn = next((
                int(m.get("content", "").removeprefix("turn "))
                for m in reversed(messages)
                if m.get("role") == "user" and str(m.get("content", "")).startswith("turn ")
            ), 1)
            # Model turns already taken inside THIS prompt: the mock sees the
            # full wire history, so count assistant tool_calls after the last
            # real user message (openai wire: tool results carry role "tool").
            last_user = max(
                (i for i, m in enumerate(messages) if m.get("role") == "user"),
                default=-1,
            )
            loop_turns = sum(
                1
                for m in messages[last_user + 1 :]
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            if args.tool_turns and loop_turns < args.tool_turns - 1:
                # One tool-loop model turn: streamed thinking-sized text plus a
                # cheap eval tool call; the result feeds the next turn.
                size = 8 * 1024
                step = args.stream_chunk_bytes
                parts = ["x" * min(step, size - i) for i in range(0, size, step)]
                call = {
                    "index": 0,
                    "id": f"call_{turn}_{loop_turns}",
                    "type": "function",
                    "function": {"name": "eval", "arguments": json.dumps({"language": "py", "code": "1"})},
                }
                return (
                    [{"role": "assistant", "content": parts[0], "tool_calls": [call]}]
                    + [{"content": part} for part in parts[1:]]
                    + [{"content": ""}],
                    "tool_calls",
                )
            size_kb = args.spike_kb if turn == 1 and args.spike_kb else args.payload_kb
            if args.vary_payload and not (turn == 1 and args.spike_kb):
                size_kb = max(1, args.payload_kb + ((turn * 37) % 17 - 8) * args.payload_kb // 8)
            size = size_kb * 1024
            step = args.stream_chunk_bytes
            parts = ["x" * min(step, size - i) for i in range(0, size, step)]
            return [{"role": "assistant", "content": parts[0]}] + [
                {"content": part} for part in parts[1:]
            ], "stop"
        return [{"role": "assistant", "content": "ok"}], "stop"

    mock_provider.script = stream_script
    server = mock_provider.serve(0)
    server.daemon_threads = True
    proc = None
    stderr = None
    try:
        with tempfile.TemporaryDirectory(prefix="omt-rpc-memory-") as tmp:
            root = Path(tmp)
            agent = root / ".omp" / "agent"
            agent.mkdir(parents=True)
            (agent / "models.yml").write_text(
                "providers:\n"
                "  mock:\n"
                f"    baseUrl: http://127.0.0.1:{server.server_address[1]}/v1\n"
                "    apiKey: benchmark-not-a-secret\n"
                "    api: openai-completions\n"
                "    models:\n"
                "      - id: stream\n"
                "        name: streaming memory benchmark\n"
                "        contextWindow: 1000000\n"
                "        maxTokens: 4096\n"
            )
            (agent / "config.yml").write_text(
                "startup:\n  setupWizard: false\n  checkUpdate: false\n"
            )
            work = root / "work"
            work.mkdir()
            keep = {
                key: value
                for key, value in os.environ.items()
                if key
                in (
                    "PATH",
                    "LANG",
                    "LC_ALL",
                    "USER",
                    "SHELL",
                    "TMPDIR",
                    "MIMALLOC_PURGE_DELAY",
                    "MIMALLOC_GENERIC_COLLECT",
                )
            }
            env = {
                **keep,
                "HOME": str(root),
                "PI_CODING_AGENT_DIR": str(agent),
                "OMP_AGENT_DIR": str(agent),
                "OMP_CONTEXT_POLICY": "always",
                "OMP_NO_TITLE": "1",
                "OMP_HUB": "0",
                "OMP_NO_UPDATE_CHECK": "1",
                "PI_SKIP_VERSION_CHECK": "1",
                "TERM": "dumb",
            }
            output_path = args.out or root / "rpc-memory.json"
            stderr = output_path.with_suffix(output_path.suffix + ".stderr").open("w")
            proc = subprocess.Popen(
                [str(args.bin), "--mode", "rpc", "--model", "mock/stream", "--no-extensions"],
                cwd=work,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
            )
            lines = queue.Queue()
            threading.Thread(
                target=lambda: [lines.put(line) for line in proc.stdout], daemon=True
            ).start()

            def rss() -> dict[str, int]:
                status = Path(f"/proc/{proc.pid}/status").read_text()
                roll = Path(f"/proc/{proc.pid}/smaps_rollup").read_text()

                def field(text: str, name: str) -> int:
                    return int(next(
                        line.split()[1]
                        for line in text.splitlines()
                        if line.startswith(name + ":")
                    ))

                return {
                    "rssKb": field(status, "VmRSS"),
                    "rssAnonKb": field(status, "RssAnon"),
                    "swapKb": field(roll, "Swap"),
                }

            peak = {"rssKb": 0, "rssAnonKb": 0}
            frame_count = 0

            def read_until(predicate, timeout: float = 180) -> dict:
                nonlocal frame_count
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        line = lines.get(timeout=max(0.001, deadline - time.monotonic()))
                    except queue.Empty:
                        raise TimeoutError("RPC frame timeout") from None
                    frame_count += 1
                    if frame_count % 10 == 0:
                        now = rss()
                        for key in peak:
                            peak[key] = max(peak[key], now[key])
                    frame = json.loads(line)
                    if predicate(frame):
                        return frame

            read_until(lambda f: f.get("type") == "ready")
            points = [{"turn": 0, **rss(), "peakRssAnonKb": 0, "peakRssKb": 0}]
            started = time.monotonic()
            for turn in range(1, args.turns + 1):
                peak = {"rssKb": 0, "rssAnonKb": 0}
                frame_count = 0
                request = {"id": f"p{turn}", "type": "prompt", "message": f"turn {turn}"}
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
                result = read_until(
                    lambda f: f.get("type") == "prompt_result" and f.get("id") == f"p{turn}"
                )
                if result.get("status") != "completed":
                    raise RuntimeError(f"prompt failed: {result}")
                time.sleep(0.05)
                points.append({"turn": turn, **rss(), **{f"peak{k[0].upper() + k[1:]}": v for k, v in peak.items()}})
                if turn == 1 or turn % 10 == 0:
                    print(json.dumps(points[-1]), flush=True)

            first, last = points[1], points[-1]
            settled_growth = last["rssAnonKb"] - first["rssAnonKb"]
            growth_from_start = last["rssAnonKb"] - points[0]["rssAnonKb"]
            summary = {
                "turns": args.turns,
                "payloadKbPerTurn": args.payload_kb,
                "spikeKbFirstTurn": args.spike_kb,
                "toolTurnsPerPrompt": args.tool_turns,
                "varyPayload": args.vary_payload,
                "streamChunkBytes": args.stream_chunk_bytes,
                "deltasPerTurn": (args.payload_kb * 1024 + args.stream_chunk_bytes - 1) // args.stream_chunk_bytes,
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "turn1RssAnonKb": first["rssAnonKb"],
                "settledRssAnonKb": last["rssAnonKb"],
                "settledGrowthKb": settled_growth,
                "settledGrowthKbPerTurn": round(settled_growth / (args.turns - 1), 2),
                "startRssAnonKb": points[0]["rssAnonKb"],
                "settledGrowthFromStartKb": growth_from_start,
                "settledGrowthFromStartKbPerTurn": round(growth_from_start / args.turns, 2),
                "maxPeakRssAnonKb": max(point["peakRssAnonKb"] for point in points[1:]),
            }
            result = {"summary": summary, "points": points}
            output_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"summary": summary, "out": str(output_path)}))
            if args.max_settled_growth_kb is not None and settled_growth > args.max_settled_growth_kb:
                print(f"FAIL: settled growth {settled_growth} KiB exceeds {args.max_settled_growth_kb} KiB")
                return 1
            return 0
    finally:
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                proc.terminate()
                proc.wait(timeout=10)
        if stderr is not None:
            stderr.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
