"""Scripted OpenAI-compatible provider for the TUI e2e (no network, no keys).

Models (request `model`, `mock/` prefix optional):
  text      -> "E2E-REPLY-<n>" where n counts user messages (so a 2nd prompt
               gets a different, checkable reply)
  tool      -> first call: one `read` tool call on E2E_READ_FILE; after the
               tool result: "E2E-TOOL-DONE"
  fail      -> HTTP 400 with an OpenAI-style error body "E2E-PROVIDER-FAILURE"
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
_lock = threading.Lock()


def log(event: dict) -> None:
    if not LOG:
        return
    with _lock, open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), **event}) + "\n")


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
        if self.path.rstrip("/").endswith("/models"):
            ids = ["text", "tool", "fail"]
            self._json(200, {"object": "list", "data": [
                {"id": f"mock/{i}", "object": "model", "created": 0, "owned_by": "mock"} for i in ids]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        model = str(body.get("model", "text"))
        log({"kind": "req", "path": self.path, "model": model, "n_messages": len(body.get("messages", []))})
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        if model.removeprefix("mock/") == "fail":
            self._json(400, {"error": {"message": "E2E-PROVIDER-FAILURE", "type": "invalid_request_error",
                                       "code": "e2e_failure"}})
            return
        deltas, finish = script(model, body.get("messages", []))
        cid = f"chatcmpl-{time.time_ns()}"

        def chunk(delta: dict, finish_reason: str | None = None) -> bytes:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": 0, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if finish_reason:
                payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}
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
            time.sleep(0.02)
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
