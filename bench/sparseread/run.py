#!/usr/bin/env python3
"""A/B baseline read vs official SparseRead MCP tools on real omt agents.

Synthetic tasks have deterministic answers and file evidence. No network judge,
no GPU, and no shared repository files are touched by the agents.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, shutil, subprocess, sys, tempfile, time
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "context"))
from run import Rpc, make_home  # noqa: E402
BINARY = HERE.parent.parent / "target/release/omt"
OFFICIAL_REPO = Path(os.environ.get("SPARSEREAD_REPO", "/tmp/SparseReading-official")).resolve()
MODELS = ["zai/glm-5.3-flash", "kimi-code/kimi-for-coding"]
ARMS = ["baseline", "sparse"]
TASKS = ["config", "diagnose", "logs", "whole"]

def padded(lines: list[str], target: int, prefix: str = "filler") -> str:
    while len(lines) < target:
        lines.append(f"{prefix} {len(lines) + 1}: unrelated operational detail " + "x" * 45)
    return "\n".join(lines) + "\n"

def make_fixture(work: Path) -> None:
    (work / "deploy").mkdir(parents=True)
    for name, count in [("database", 520), ("service", 480), ("features", 460)]:
        (work / "deploy" / f"{name}.toml").write_text(padded(["[production]"], count, f"{name} documentation"))
    for name in ["staging", "development", "example"]:
        (work / "deploy" / f"{name}.toml").write_text(padded([], 180, f"decoy {name}"))
    for path, line, value in [("database.toml", 267, "max_connections = 640"),("service.toml", 239, "read_timeout_ms = 2750"),("features.toml", 223, 'feature_flag = "blue-green-v2"')]:
        target = work / "deploy" / path
        rows = target.read_text().splitlines(); rows[line - 1] = value
        target.write_text("\n".join(rows) + "\n")
    package = work / "src"; package.mkdir()
    retry = """\"\"\"Retry execution policy.\"\"\"
from dataclasses import dataclass
@dataclass
class RetryPolicy:
    max_attempts: int = 3
def execute(operation, policy):
    attempts = 0
    last_error = None
    while attempts < policy.max_attempts - 1:
        attempts += 1
        try:
            return operation()
        except TransientError as error:
            last_error = error
    raise RuntimeError("retry exhausted") from last_error
class TransientError(Exception):
    pass
""".splitlines()
    (package / "retry.py").write_text(padded(retry, 360, "retry implementation note"))
    tests = work / "tests"; tests.mkdir()
    test_retry = """from src.retry import execute, RetryPolicy, TransientError
def test_three_attempts_are_allowed():
    calls = []
    def always_transient():
        calls.append(1)
        raise TransientError("try again")
    try:
        execute(always_transient, RetryPolicy(max_attempts=3))
    except RuntimeError:
        pass
    assert len(calls) == 3, "regression: only two attempts executed"
""".splitlines()
    (tests / "test_retry.py").write_text(padded(test_retry, 220, "test documentation"))
    for name in ["client.py", "server.py", "scheduler.py", "metrics.py", "auth.py", "storage.py"]:
        (package / name).write_text(padded([f'"""Unrelated {name} module."""'], 240, name))
    log = []
    for number in range(1, 760):
        log.append(f"2026-04-01T10:{number // 60:02d}:{number % 60:02d} INFO request_id=req-{number:04d} module=api status=ok")
        if number in (118, 644):
            log.append(f"2026-04-01T11:{number // 60:02d}:{number % 60:02d} ERROR request_id=req-old-{number} code=E_TIMEOUT module=network")
    log[-12:] = ["2026-04-01T12:00:00 INFO request_id=req-7f3a9 module=api status=started","2026-04-01T12:00:01 WARN request_id=req-7f3a9 module=cache.refresh stampede_detected=true","2026-04-01T12:00:02 ERROR request_id=req-7f3a9 code=E_CACHE_STAMPEDE module=cache.refresh fatal=true"] + log[-9:]
    (work / "service.log").write_text("\n".join(log) + "\n")
    protocol = padded(["PROTOCOL COMPLETE INSPECTION DOCUMENT"], 540, "protocol paragraph").splitlines()
    for section, start, code in [("alpha", 80, "ALPHA-7319"), ("beta", 260, "BETA-2846"), ("gamma", 440, "GAMMA-9052")]:
        protocol[start - 1] = f"SECTION {section}: ceremonial instructions"
        protocol[start] = f"The authoritative seal code for section {section} is {code}."
    (work / "protocol.txt").write_text(padded(protocol, 540, "protocol paragraph"))

PROMPTS = {
 "config": "Using the files under deploy/, find the production max_connections, read_timeout_ms, and feature_flag. Ignore staging/development/example decoys. Reply with exactly one JSON object: {\"max_connections\": integer, \"read_timeout_ms\": integer, \"feature_flag\": string}.",
 "diagnose": "Diagnose the retry regression using src/retry.py and tests/test_retry.py. Reply with exactly one JSON object: {\"root_cause\": string, \"minimal_fix\": string, \"evidence\": [two path:line strings, one for each file]}. The root cause must identify the precise off-by-one expression and the fix must state the corrected loop condition.",
 "logs": "Inspect service.log. For the latest fatal error, reply with exactly one JSON object: {\"request_id\": string, \"code\": string, \"module\": string}. Do not confuse older E_TIMEOUT failures with the latest fatal error.",
 "whole": "Read the entire protocol.txt, including beginning, middle and end. Return the authoritative seal code for each section as exactly one JSON object: {\"alpha\": string, \"beta\": string, \"gamma\": string}.",
}
SPARSE_PROTOCOL = """You are testing the official SparseRead tools. The ordinary read tool is unavailable. Use sro_preview for a bounded preview and FileCard, then sro_read with a target, mode and evidence hint. Use sro_raw only for explicit raw fallback when complete original text is required. Preserve source anchors and unresolved slots; empty evidence is not proof of absence. Do not modify files. """
BASELINE_PROTOCOL = "Use read, grep and glob as appropriate to inspect the files. Do not modify files. "

def result_text(result: dict) -> str:
    return "".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")

def json_answer(text: str):
    decoder = json.JSONDecoder(); candidates = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict): candidates.append(value)
        except json.JSONDecodeError: pass
    return candidates[-1] if candidates else None

def score(task: str, answer) -> tuple[float, str]:
    if not isinstance(answer, dict): return 0.0, "no JSON object"
    if task == "config":
        expected = {"max_connections": 640, "read_timeout_ms": 2750, "feature_flag": "blue-green-v2"}
        got = sum(answer.get(key) == value for key, value in expected.items()); return got / 3, f"{got}/3 exact fields"
    if task == "logs":
        expected = {"request_id": "req-7f3a9", "code": "E_CACHE_STAMPEDE", "module": "cache.refresh"}
        got = sum(answer.get(key) == value for key, value in expected.items()); return got / 3, f"{got}/3 exact fields"
    if task == "whole":
        expected = {"alpha": "ALPHA-7319", "beta": "BETA-2846", "gamma": "GAMMA-9052"}
        got = sum(answer.get(key) == value for key, value in expected.items()); return got / 3, f"{got}/3 exact fields"
    root = str(answer.get("root_cause", "")).lower().replace(" ", "")
    fix = str(answer.get("minimal_fix", "")).lower().replace(" ", "")
    evidence = " ".join(map(str, answer.get("evidence", [])))
    root_ok = "off-by-one" in root or "offbyone" in root or "max_attempts-1" in root
    fix_ok = ("attempts<policy.max_attempts" in fix or "attempts<max_attempts" in fix or "attempts<=policy.max_attempts-1" in fix or "attempts<=max_attempts-1" in fix)
    evidence_ok = "src/retry.py:" in evidence and "tests/test_retry.py:" in evidence
    got = sum([root_ok, fix_ok, evidence_ok]); return got / 3, f"root={root_ok}, fix={fix_ok}, evidence={evidence_ok}"

def one_cell(model: str, arm: str, task: str, repeat: int, timeout: int) -> dict:
    home = Path(tempfile.mkdtemp(prefix="omt-sparse-home-")); work = Path(tempfile.mkdtemp(prefix="omt-sparse-work-"))
    session_id = f"{model.split('/')[0]}-{arm}-{task}-r{repeat}-{int(time.time() * 1000)}"
    row = {"model": model, "arm": arm, "task": task, "repeat": repeat, "session_id": session_id,
           "sparseread_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL_REPO, text=True).strip() if arm == "sparse" else None}
    try:
        make_fixture(work)
        _, env = make_home(str(home))
        if arm == "sparse":
            assert OFFICIAL_REPO.is_dir(), f"set SPARSEREAD_REPO to the official checkout; missing {OFFICIAL_REPO}"
            mcp = {"mcpServers": {"sro": {"type": "stdio", "command": sys.executable,
                "args": [str(HERE / "official_mcp.py")],
                "env": {"SPARSEREAD_REPO": str(OFFICIAL_REPO), "SPARSEREAD_MODE": "auto"}, "enabled": True}}}
            (home / ".omp/agent/mcp.json").write_text(json.dumps(mcp))
        tools = "read,grep,glob" if arm == "baseline" else "mcp__sro_preview,mcp__sro_read,mcp__sro_raw,mcp__sro_card,grep,glob"
        protocol = BASELINE_PROTOCOL if arm == "baseline" else SPARSE_PROTOCOL
        command = [str(BINARY), "--mode", "rpc", "--model", model, "--thinking", "minimal","--tools", tools, "--no-extensions", "--no-skills", "--no-rules", "--no-title","--no-lsp", "--no-pty", "--allow-home"]
        rpc = Rpc(command, env=env, cwd=work)
        calls = []; call_by_id = {}; call_names = Counter(); call_counts = Counter()
        errors = 0; output_chars = 0; full_fallbacks = 0; replies = []
        usage = {"uncached": 0, "cached": 0, "cache_write": 0, "completion": 0, "calls": 0}
        started = time.monotonic()
        def event(frame):
            nonlocal errors, output_chars, full_fallbacks
            kind = frame.get("type")
            if kind == "tool_execution_start":
                signature = json.dumps([frame.get("toolName"), frame.get("args")], sort_keys=True)
                call = {"name": frame.get("toolName"), "args": frame.get("args"), "output_chars": 0}
                calls.append(call); call_by_id[frame.get("toolCallId")] = call
                call_names[frame.get("toolName")] += 1; call_counts[signature] += 1
            elif kind == "tool_execution_end":
                text = result_text(frame.get("result", {})); output_chars += len(text)
                call = call_by_id.get(frame.get("toolCallId"))
                if call is not None:
                    call["output_chars"] = len(text)
                    try:
                        payload = json.loads(text); call["kind"] = payload.get("kind") or payload.get("type")
                        if (call["name"] == "mcp__sro_raw" and not frame.get("isError") and not call["args"].get("range") and not call["args"].get("selector")): full_fallbacks += 1
                    except (json.JSONDecodeError, AttributeError): pass
                errors += int(bool(frame.get("isError")))
            elif kind == "message_end":
                message = frame.get("message", {})
                if message.get("role") == "assistant":
                    replies.append("".join(part.get("text", "") for part in message.get("content", []) if part.get("type") == "text"))
                    value = message.get("usage", {})
                    if "input" in value:
                        usage["uncached"] += value.get("input", 0); usage["cached"] += value.get("cacheRead", 0)
                        usage["cache_write"] += value.get("cacheWrite", 0); usage["completion"] += value.get("output", 0); usage["calls"] += 1
        try:
            request_id = rpc.next_id; rpc.next_id += 1
            rpc.p.stdin.write(json.dumps({"id": request_id, "type": "prompt", "message": protocol + PROMPTS[task]}) + "\n")
            rpc.p.stdin.flush(); deadline = time.monotonic() + timeout
            while True:
                frame = rpc.frame(max(1, deadline - time.monotonic())); event(frame)
                if frame.get("type") == "response" and frame.get("id") == request_id and frame.get("success") is False:
                    raise RuntimeError(f"prompt rejected: {frame}")
                if frame.get("type") == "prompt_result" and frame.get("id") == request_id: break
        finally: rpc.close()
        row["runtime_stderr"] = "".join(rpc.stderr)[-8000:]
        reply = replies[-1] if replies else ""; answer = json_answer(reply); value, reason = score(task, answer)
        row.update({"ok": True, "score": value, "score_reason": reason, "answer": answer, "reply": reply,"latency_s": round(time.monotonic() - started, 3), **usage,"total_tokens": usage["uncached"] + usage["cached"] + usage["cache_write"] + usage["completion"],"tool_calls": sum(call_names.values()), "tool_calls_by_name": dict(call_names),"tool_errors": errors, "tool_output_chars": output_chars,"repeated_identical_calls": sum(count - 1 for count in call_counts.values() if count > 1),"raw_calls": call_names["mcp__sro_raw"], "full_view_fallbacks": full_fallbacks, "tool_trace": calls})
    except Exception as error: row.update({"ok": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        shutil.rmtree(home, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    return row

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--models", default=",".join(MODELS)); parser.add_argument("--arms", default=",".join(ARMS)); parser.add_argument("--tasks", default=",".join(TASKS)); parser.add_argument("--repeat", type=int, default=1); parser.add_argument("--start-repeat", type=int, default=1); parser.add_argument("--timeout", type=int, default=300); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        for task in args.tasks.split(","):
            for arm in args.arms.split(","):
                for repeat in range(args.start_repeat, args.repeat + 1):
                    print(f"START {model} {task} {arm} r{repeat}", flush=True)
                    row = one_cell(model, arm, task, repeat, args.timeout)
                    with args.output.open("a") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("END", json.dumps({key: row.get(key) for key in ["model", "task", "arm", "ok", "score", "total_tokens", "latency_s", "tool_calls", "error"]}), flush=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
