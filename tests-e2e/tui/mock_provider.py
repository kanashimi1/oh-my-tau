"""Scripted OpenAI-compatible provider for the TUI e2e (no network, no keys).

Models (request `model`, `mock/` prefix optional):
  text      -> "E2E-REPLY-<n>" where n counts user messages (so a 2nd prompt
               gets a different, checkable reply)
  tool      -> first call: one `read` tool call on E2E_READ_FILE; after the
               tool result: "E2E-TOOL-DONE"
  tools     -> bash, write, grep, glob calls (one per request), then "E2E-TOOLS-DONE"
  slow      -> 150 words in chunks of E2E_SLOW_WORDS (default 1), paced E2E_SLOW_MS apart (default
               40 ms), then "SLOW-DONE"; for the streaming-cadence probe
  retry     -> HTTP 503 on the first request after GET /reset, else like `text`
  full      -> like `text` but reports a context-filling usage (auto-compaction trigger)
  fail      -> HTTP 400 with an OpenAI-style error body "E2E-PROVIDER-FAILURE"
               when the last user message contains FAIL_MARKER, else like `text`
Every request is appended to E2E_MOCK_LOG (JSON lines) when set.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Relative: the read tool resolves it against the client cwd (the e2e work dir).
READ_FILE = os.environ.get("E2E_READ_FILE", "e2e-read-me.txt")
LOG = os.environ.get("E2E_MOCK_LOG", "")
SLOW_S = float(os.environ.get("E2E_SLOW_MS", "40")) / 1000
_lock = threading.Lock()
RETRY = {"n": 0}
REQS: list[dict] = []  # every /chat/completions request body, in arrival order (in-process scenarios)


def log(event: dict) -> None:
    if not LOG:
        return
    with _lock, open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), **event}) + "\n")


# `tools`: one call per request through the core tool families, then a reply.
TOOLS_SCRIPT = [
    ("bash", {"command": "echo E2E-BASH-OUT"}),
    ("write", {"path": "e2e-new.txt", "content": "alpha\nbeta\n"}),
    ("grep", {"pattern": "beta", "path": "."}),
    ("glob", {"path": "*.txt"}),
]

# `guards`: calls the user's tool_call gates must refuse (each is harmless if a gate fails to block:
# pgrep only lists, `find ~` walks the throwaway HOME, the herdr binary is a PATH shim), then a reply.
GUARDS_SCRIPT = [
    ("bash", {"command": "pgrep -f bun"}),
    ("bash", {"command": "find ~ -name zzz-nothing"}),
    ("bash", {"command": "herdr server stop"}),
]

# The `fail` model rejects only prompts carrying this marker, so a follow-up
# prompt succeeds and the pinned error banner's dismissal is observable.
FAIL_MARKER = "will fail"


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def script(model: str, messages: list[dict]) -> tuple[list[dict], str]:
    model = model.removeprefix("mock/")
    users = sum(1 for m in messages if m.get("role") == "user")
    has_tool = any(m.get("role") == "tool" for m in messages)
    if model == "tool":
        if not has_tool:
            call = {
                "index": 0,
                "id": "call_read_0",
                "type": "function",
                "function": {"name": "read", "arguments": json.dumps({"path": READ_FILE})},
            }
            return [{"role": "assistant", "content": "", "tool_calls": [call]}], "tool_calls"
        return [{"role": "assistant", "content": "E2E-TOOL-DONE"}], "stop"
    if model == "tools":
        done = sum(1 for m in messages if m.get("role") == "tool")
        if done < len(TOOLS_SCRIPT):
            name, args = TOOLS_SCRIPT[done]
            call = {"index": 0, "id": f"call_{name}_{done}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}
            return [{"role": "assistant", "content": "", "tool_calls": [call]}], "tool_calls"
        return [{"role": "assistant", "content": "E2E-TOOLS-DONE"}], "stop"
    if model == "guards":
        done = sum(1 for m in messages if m.get("role") == "tool")
        if done < len(GUARDS_SCRIPT):
            name, args = GUARDS_SCRIPT[done]
            call = {"index": 0, "id": f"call_{name}_{done}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}
            return [{"role": "assistant", "content": "", "tool_calls": [call]}], "tool_calls"
        return [{"role": "assistant", "content": "E2E-GUARDS-DONE"}], "stop"
    if model == "slow":
        n = int(os.environ.get("E2E_SLOW_WORDS", "1"))  # words per chunk (burstiness)
        words = [f"w{i:03d} " for i in range(150)]
        parts = ["".join(words[i:i + n]) for i in range(0, len(words), n)]
        deltas = [{"role": "assistant", "content": parts[0]}] + [{"content": x} for x in parts[1:]]
        return deltas + [{"content": "SLOW-DONE"}], "stop"
    reply = f"E2E-REPLY-{users}"
    return [{"role": "assistant", "content": reply[:6]}, {"content": reply[6:]}], "stop"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/reset"):
            RETRY["n"] = 0
            self._json(200, {"ok": True})
            return
        if self.path.rstrip("/").endswith("/models"):
            ids = ["text", "tool", "tools", "guards", "fail", "retry", "full", "devrole"]
            self._json(200, {"object": "list", "data": [
                {"id": f"mock/{i}", "object": "model", "created": 0, "owned_by": "mock"} for i in ids]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        model = str(body.get("model", "text"))
        log({"kind": "req", "path": self.path, "model": model, "n_messages": len(body.get("messages", [])),
             "tools": {t.get("function", {}).get("name"): sorted((t.get("function", {}).get("parameters") or {}).get("properties", {}))
                       for t in body.get("tools", []) or []}})
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        REQS.append(body)
        if model.removeprefix("mock/") == "fail" and FAIL_MARKER in last_user_text(body.get("messages", [])):
            self._json(400, {"error": {"message": "E2E-PROVIDER-FAILURE", "type": "invalid_request_error",
                                       "code": "e2e_failure"}})
            return
        if model.removeprefix("mock/") == "retry" and body.get("tools") and RETRY["n"] < 1:  # the tool-less title request must not eat the failure
            RETRY["n"] += 1
            self._json(503, {"error": {"message": "E2E-OVERLOADED", "type": "server_error", "code": "overloaded"}})
            return
        deltas, finish = script(model, body.get("messages", []))
        cid = f"chatcmpl-{time.time_ns()}"

        def chunk(delta: dict, finish_reason: str | None = None) -> bytes:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": 0, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if finish_reason:
                prompt = 7500 if model.removeprefix("mock/") == "full" else 100
                payload["usage"] = {"prompt_tokens": prompt, "completion_tokens": 7, "total_tokens": prompt + 7}
            return f"data: {json.dumps(payload)}\n\n".encode()

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

        def send(data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        for d in deltas:
            send(chunk(d))
            time.sleep(SLOW_S if model.removeprefix("mock/") == "slow" else 0.02)
        send(chunk({}, finish))
        send(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def serve(port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    srv = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(f"mock provider on 127.0.0.1:{srv.server_address[1]}", flush=True)
    threading.Event().wait()
