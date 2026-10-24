#!/usr/bin/env python3
"""omp-rs OAuth refresh e2e (temp HOME, mock Kimi token endpoint + mock Anthropic-format provider).

A: expired kimi-code token in agent.db -> `omp-rs -p` refreshes it (form grant), writes it back, the
   request carries the new bearer.
B: token valid on paper but rejected (401) -> exactly one forced refresh + retry, answer arrives.
C: auth broker configured (mock): snapshot per request, near-expiry row refreshed via POST
   /v1/credential/:id/refresh, request carries the broker's bearer; agent.db is untouched.
Usage: refresh_e2e.py [path/to/omt]
"""
import json, os, shutil, sqlite3, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../target/shared/debug/omt")
LOG = []  # (method, path, authorization/x-api-key, body)
STATE = {"revoked": set(), "tokens": 0}


def sse(text):
    ev = [
        ("message_start", {"type": "message_start", "message": {"id": "m1", "type": "message", "role": "assistant", "model": "kimi-for-coding", "content": [], "usage": {"input_tokens": 3, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in ev).encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _do(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        auth = self.headers.get("authorization") or self.headers.get("x-api-key") or ""
        LOG.append((self.command, self.path, auth, body))
        def send(code, payload, ctype="application/json"):
            self.send_response(code); self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
        if self.path == "/api/oauth/token":
            STATE["tokens"] += 1
            return send(200, json.dumps({"access_token": f"minted-{STATE['tokens']}", "refresh_token": f"rt-{STATE['tokens']}", "expires_in": 900}).encode())
        if self.path.startswith("/v1/snapshot"):
            return send(200, json.dumps({"generation": 1, "credentials": [{"id": 7, "provider": "kimi-code", "credential": {
                "type": "oauth", "access": "broker-old", "refresh": "__remote__", "expires": int(time.time() * 1000) + 5000}}]}).encode())
        if self.path.startswith("/v1/credential/7/refresh"):
            return send(200, json.dumps({"entry": {"id": 7, "provider": "kimi-code", "credential": {
                "type": "oauth", "access": "broker-new", "refresh": "__remote__", "expires": int(time.time() * 1000) + 900000}}}).encode())
        if "messages" in self.path or "chat/completions" in self.path:
            token = auth.replace("Bearer ", "")
            if token in STATE["revoked"]:
                return send(401, json.dumps({"error": {"type": "invalid_authentication_error", "message": "token revoked"}}).encode())
            return send(200, sse("pong"), "text/event-stream")
        send(404, b"{}")
    do_GET = do_POST = _do


def make_home(row, broker=None):
    home = tempfile.mkdtemp(prefix="omp-auth-e2e-")
    agent = os.path.join(home, ".omp", "agent"); os.makedirs(agent)
    db = sqlite3.connect(os.path.join(agent, "agent.db"))
    db.executescript("""CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL,
        credential_type TEXT NOT NULL, data TEXT NOT NULL, disabled_cause TEXT DEFAULT NULL, identity_key TEXT DEFAULT NULL,
        created_at INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0);""")
    if row:
        db.execute("INSERT INTO auth_credentials(provider, credential_type, data) VALUES ('kimi-code','oauth',?)", (json.dumps(row),))
    db.commit(); db.close()
    return home, agent


def run(home, port, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_AUTH_BROKER", "KIMI_", "OMP_AGENT_DIR", "PI_CODING"))}
    env.update(HOME=home, KIMI_OAUTH_HOST=f"http://127.0.0.1:{port}", **(extra_env or {}))
    with open(os.path.join(home, ".omp", "agent", "models.yml"), "w") as f:
        f.write(f"providers:\n  kimi-code:\n    baseUrl: http://127.0.0.1:{port}/coding/v1\n")
    p = subprocess.run([BIN, "-p", "say pong", "--model", "kimi-code/kimi-for-coding", "--no-session"], env=env, capture_output=True, text=True, timeout=120, cwd=home)
    return p


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}")); ok &= bool(cond)

    # A: expired token refreshed before the request, written back.
    LOG.clear(); STATE["tokens"] = 0
    home, agent = make_home({"access": "expired-old", "refresh": "rt-0", "expires": int(time.time() * 1000) - 60_000, "accountId": "acct"})
    p = run(home, port)
    msgs = [e for e in LOG if "messages" in e[1] or "chat/completions" in e[1]]
    check("A exit 0 + answer", p.returncode == 0 and "pong" in p.stdout, p.stdout + p.stderr[-500:])
    check("A one token grant", STATE["tokens"] == 1 and any("grant_type=refresh_token" in e[3] and "refresh_token=rt-0" in e[3] for e in LOG), LOG)
    check("A request used minted-1", len(msgs) >= 1 and all("minted-1" in m[2] for m in msgs), [(m[1], m[2]) for m in msgs])
    row = json.loads(sqlite3.connect(os.path.join(agent, "agent.db")).execute("select data from auth_credentials").fetchone()[0])
    check("A written back", row["access"] == "minted-1" and row["refresh"] == "rt-1" and row["accountId"] == "acct", row)
    shutil.rmtree(home)

    # B: valid-on-paper but revoked token: 401 -> one forced refresh -> retry succeeds.
    LOG.clear(); STATE["tokens"] = 0; STATE["revoked"] = {"revoked-tok"}
    home, agent = make_home({"access": "revoked-tok", "refresh": "rt-0", "expires": int(time.time() * 1000) + 600_000})
    p = run(home, port)
    msgs = [e for e in LOG if "messages" in e[1] or "chat/completions" in e[1]]
    check("B exit 0 + answer", p.returncode == 0 and "pong" in p.stdout, p.stdout + p.stderr[-500:])
    # The agent turn is the large request; the small one is the session-title side call.
    turn = [m for m in msgs if len(m[3]) > 5000]
    check("B 401 then one retry with minted-1", [m[2] for m in turn] == ["revoked-tok", "minted-1"] and STATE["tokens"] == 1, ([(m[1], m[2], len(m[3])) for m in msgs], STATE))
    shutil.rmtree(home)

    # C: broker mode; agent.db has no credential at all.
    LOG.clear(); STATE["revoked"] = set()
    home, agent = make_home(None)
    p = run(home, port, {"OMP_AUTH_BROKER_URL": f"http://127.0.0.1:{port}", "OMP_AUTH_BROKER_TOKEN": "btok"})
    msgs = [e for e in LOG if "messages" in e[1] or "chat/completions" in e[1]]
    check("C exit 0 + answer", p.returncode == 0 and "pong" in p.stdout, p.stdout + p.stderr[-500:])
    check("C broker refresh + bearer", any(e[1].startswith("/v1/credential/7/refresh") for e in LOG) and msgs and "broker-new" in msgs[-1][2], LOG)
    check("C no local token grant", not any(e[1] == "/api/oauth/token" for e in LOG))
    shutil.rmtree(home)
    print("OK" if ok else "FAILED"); sys.exit(0 if ok else 1)


main()
