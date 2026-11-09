#![allow(clippy::disallowed_methods)] // tests/entry points: throwaway runtimes drive async code
//! Call-lifecycle tests for the three upstream 18.12.4 MCP fixes:
//! `mcp-cancel-notification` (aborted/timed-out stdio calls send exactly one
//! `notifications/cancelled`), `mcp-retry-double-run` (a request already
//! written to a dropped connection fails non-retryable instead of being
//! replayed), and `mcp-tool-progress` (`notifications/progress` updates
//! keyed by the `_meta.progressToken` reach the caller while the call runs).
//!
//! All scenarios run against a fake stdio MCP server (python3) that logs
//! every message it receives to `MCP_TEST_LOG` so the tests can assert on
//! the server's side of the wire.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use omp_mcp::errors::McpError;
use omp_mcp::transport::stdio::StdioTransport;
use omp_mcp::transport::McpTransport;
use omp_mcp::{McpManager, McpServerConfig, ServerKind};
use serde_json::{json, Value};

/// Fake MCP stdio server:
/// - logs every received message (method, id, params) as one JSON line to
///   the file named by `MCP_TEST_LOG`;
/// - `tools/list` advertises `slow` (runs 30s, never responds), `progressy`
///   (two `notifications/progress` updates then a normal result) and `die`
///   (exits the process immediately, mid-request);
/// - `slow` runs its sleep on a daemon thread so the main loop keeps
///   reading stdin — an incoming `notifications/cancelled` must be logged.
const FAKE_SERVER_PY: &str = r#"
import json, os, sys, threading, time

LOG = os.environ.get("MCP_TEST_LOG")

def log(obj):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(obj) + "\n")

def send(obj):
    log({"send": obj.get("method"), "id": obj.get("id")})
    print(json.dumps(obj), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    log({"m": msg.get("method"), "id": msg.get("id"), "params": msg.get("params")})
    method = msg.get("method")
    if method == "initialize":
        if os.environ.get("MCP_TEST_SLOW_INIT"):
            time.sleep(30)
        else:
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0.1"}}})
    elif method == "tools/list":
        tools = [{"name": n, "description": n, "inputSchema": {"type": "object", "properties": {}}}
                 for n in ("slow", "progressy", "die")]
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": tools}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        if name == "slow":
            threading.Thread(target=time.sleep, args=(30,), daemon=True).start()
        elif name == "progressy":
            token = msg["params"].get("_meta", {}).get("progressToken")
            for progress, message in ((0.5, "halfway"), (1.0, None)):
                note = {"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": token, "progress": progress, "total": 2.0}}
                if message:
                    note["params"]["message"] = message
                send(note)
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": "done"}]}})
        elif name == "die":
            sys.exit(0)
"#;

fn python_available() -> bool {
    std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_ok()
}

fn write_fake_server(dir: &Path) -> PathBuf {
    let path = dir.join("lifecycle_mcp_server.py");
    std::fs::write(&path, FAKE_SERVER_PY).expect("write fake server script");
    path
}

fn stdio_config(script: &Path, log: &Path, timeout_ms: u64) -> McpServerConfig {
    let mut env = BTreeMap::new();
    env.insert(
        "MCP_TEST_LOG".to_string(),
        log.to_string_lossy().into_owned(),
    );
    McpServerConfig {
        enabled: Some(true),
        timeout: Some(timeout_ms),
        kind: ServerKind::Stdio,
        command: Some("python3".to_string()),
        args: Some(vec![script.to_string_lossy().into_owned()]),
        env: Some(env),
        cwd: None,
        url: None,
        headers: None,
        extra: BTreeMap::new(),
    }
}

/// The server's side of the wire: every message it received, oldest first.
fn read_log(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("log line is JSON"))
        .collect()
}

fn calls(log: &[Value]) -> Vec<&Value> {
    log.iter().filter(|e| e["m"] == "tools/call").collect()
}

fn cancelled_notes(log: &[Value]) -> Vec<&Value> {
    log.iter()
        .filter(|e| e["m"] == "notifications/cancelled")
        .collect()
}

async fn handshake(transport: &mut StdioTransport) {
    transport
        .request(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {"roots": {"listChanged": false}},
                "clientInfo": {"name": "omt-test", "version": "0.0.0"},
            }),
        )
        .await
        .expect("initialize");
    transport
        .notify("notifications/initialized", json!({}))
        .await
        .expect("notifications/initialized");
}

/// Aborted stdio calls (future dropped) must make the transport send exactly
/// one `notifications/cancelled` with the original request id — never for
/// `initialize`, without blocking on the pipe (mcp-cancel-notification).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_sends_cancelled_once() {
    if !python_available() {
        eprintln!("python3 not available; skipping");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let script = write_fake_server(dir.path());
    let log_path = dir.path().join("wire.log");
    let mut transport = StdioTransport::new(
        "abort",
        stdio_config(&script, &log_path, 0),
        Arc::new(PathBuf::from(".")),
    );
    transport.connect().await.expect("connect");
    handshake(&mut transport).await;

    let call = transport.request("tools/call", json!({"name": "slow", "arguments": {}}));
    let handle = tokio::spawn(call);
    tokio::time::sleep(Duration::from_millis(400)).await;
    handle.abort();
    // The Drop guard spawns a bounded best-effort write; give it a moment.
    tokio::time::sleep(Duration::from_millis(1_000)).await;

    let wire = read_log(&log_path);
    let call_id = calls(&wire)[0]["id"].clone();
    let notes = cancelled_notes(&wire);
    assert_eq!(
        notes.len(),
        1,
        "exactly one cancelled notification, wire: {wire:?}"
    );
    assert_eq!(notes[0]["params"]["requestId"], call_id);
    assert!(
        wire.iter()
            .filter(|e| e["m"] == "initialize" && e["params"].get("requestId").is_some())
            .count()
            == 0,
        "initialize must never be cancelled"
    );
    transport.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborted_initialize_never_sends_cancelled() {
    if !python_available() {
        eprintln!("python3 not available; skipping");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let script = write_fake_server(dir.path());
    let log_path = dir.path().join("wire.log");
    let mut config = stdio_config(&script, &log_path, 0);
    config
        .env
        .as_mut()
        .unwrap()
        .insert("MCP_TEST_SLOW_INIT".to_string(), "1".to_string());
    let mut transport = StdioTransport::new("abort-init", config, Arc::new(PathBuf::from(".")));
    transport.connect().await.expect("connect");
    let call = transport.request("initialize", json!({}));
    let handle = tokio::spawn(call);
    tokio::time::sleep(Duration::from_millis(400)).await;
    handle.abort();
    tokio::time::sleep(Duration::from_millis(700)).await;
    let wire = read_log(&log_path);
    assert_eq!(wire.iter().filter(|e| e["m"] == "initialize").count(), 1);
    assert!(cancelled_notes(&wire).is_empty(), "wire: {wire:?}");
    transport.close().await;
}

/// A timed-out stdio call whose frame reached the server must fail
/// non-retryable with a "may have already run" diagnostic (the manager must
/// not replay it) and still owe the server one `notifications/cancelled`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeout_is_non_retryable_and_sends_cancelled() {
    if !python_available() {
        eprintln!("python3 not available; skipping");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let script = write_fake_server(dir.path());
    let log_path = dir.path().join("wire.log");
    let mut transport = StdioTransport::new(
        "timeout",
        stdio_config(&script, &log_path, 600),
        Arc::new(PathBuf::from(".")),
    );
    transport.connect().await.expect("connect");
    handshake(&mut transport).await;

    let error = transport
        .request("tools/call", json!({"name": "slow", "arguments": {}}))
        .await
        .expect_err("slow call must time out");
    let McpError::Transport(transport_error) = error else {
        panic!("expected a transport error, got {error:?}");
    };
    assert!(
        !transport_error.retryable,
        "a written-then-failed request must not be replayed: {transport_error}"
    );
    assert!(
        transport_error.message.contains("may have already run"),
        "diagnostic must say the call may have run: {}",
        transport_error.message
    );

    tokio::time::sleep(Duration::from_millis(1_000)).await;
    let wire = read_log(&log_path);
    let call_id = calls(&wire)[0]["id"].clone();
    let notes = cancelled_notes(&wire);
    assert_eq!(notes.len(), 1, "wire: {wire:?}");
    assert_eq!(notes[0]["params"]["requestId"], call_id);
    transport.close().await;
}

/// A request already written to a server whose connection then dies must not
/// be replayed after reconnect: the manager surfaces the "may have already
/// run" diagnostic and the server sees exactly one `tools/call`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mid_request_drop_is_not_retried() {
    if !python_available() {
        eprintln!("python3 not available; skipping");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let script = write_fake_server(dir.path());
    let log_path = dir.path().join("wire.log");
    let mut configs = BTreeMap::new();
    configs.insert("s".to_string(), stdio_config(&script, &log_path, 15_000));
    let manager = McpManager::new(Arc::new(dir.path().to_path_buf()), configs);

    let error = manager
        .call("s", "die", json!({}))
        .await
        .expect_err("die call must fail");
    assert!(
        error.to_string().contains("may have already run"),
        "expected the may-have-run diagnostic, got: {error}"
    );

    // The retry path must have been skipped: the server saw the call once.
    // (A replay would show up as a second tools/call, even across respawns,
    // because every incarnation appends to the same log file.)
    tokio::time::sleep(Duration::from_millis(300)).await;
    let wire = read_log(&log_path);
    assert_eq!(
        calls(&wire).len(),
        1,
        "the call must not run twice, wire: {wire:?}"
    );
}

/// `tools/call` requests advertise `_meta.progressToken`, and the server's
/// `notifications/progress` updates reach the caller while the call runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn progress_updates_reach_caller() {
    if !python_available() {
        eprintln!("python3 not available; skipping");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let script = write_fake_server(dir.path());
    let log_path = dir.path().join("wire.log");
    let mut configs = BTreeMap::new();
    configs.insert("s".to_string(), stdio_config(&script, &log_path, 15_000));
    let manager = McpManager::new(Arc::new(dir.path().to_path_buf()), configs);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let result = manager
        .call_with_progress("s", "progressy", json!({}), move |update| {
            let _ = tx.send(update);
        })
        .await
        .expect("progressy call");

    let first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for the first progress update")
        .expect("first progress update");
    let second = rx.recv().await.expect("second progress update");
    assert_eq!(first.progress, Some(0.5));
    assert_eq!(first.total, Some(2.0));
    assert_eq!(first.message.as_deref(), Some("halfway"));
    assert_eq!(second.progress, Some(1.0));

    // The progress token we asked for is the request id, and the server
    // echoed it back in its notifications.
    let wire = read_log(&log_path);
    let call = calls(&wire)[0];
    assert_eq!(
        call["params"]["_meta"]["progressToken"], call["id"],
        "tools/call must carry _meta.progressToken = request id"
    );
    assert_eq!(first.token, call["id"]);
    assert_eq!(second.token, call["id"]);

    let text = omp_mcp::format::format_tool_result(&result);
    assert!(text.contains("done"), "formatted: {text}");
}

/// Streamable HTTP: an SSE EOF after the request was accepted is ambiguous,
/// not a stale-connection error. It must carry the no-replay diagnostic and
/// the manager must not reconnect and issue the tool call a second time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_sse_eof_is_not_retried_and_carries_progress_token() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn request(stream: &mut tokio::net::TcpStream) -> Value {
        let mut bytes = Vec::new();
        let (header_end, content_len) = loop {
            let mut chunk = [0_u8; 4096];
            let n = stream.read(&mut chunk).await.expect("read HTTP request");
            assert!(n > 0, "request ended before headers");
            bytes.extend_from_slice(&chunk[..n]);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..end]);
                let len = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().expect("content length"))
                    })
                    .unwrap_or(0);
                break (end + 4, len);
            }
        };
        while bytes.len() < header_end + content_len {
            let mut chunk = [0_u8; 4096];
            let n = stream.read(&mut chunk).await.expect("read HTTP body");
            assert!(n > 0, "request ended before body");
            bytes.extend_from_slice(&chunk[..n]);
        }
        serde_json::from_slice(&bytes[header_end..header_end + content_len]).expect("JSON body")
    }

    async fn respond(stream: &mut tokio::net::TcpStream, content_type: &str, body: String) {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        stream.shutdown().await.expect("shutdown response");
    }

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake HTTP MCP");
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server =
        tokio::spawn(async move {
            let mut calls = Vec::new();
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.expect("accept HTTP MCP");
                let body = request(&mut stream).await;
                let method = body["method"].as_str().unwrap().to_string();
                let id = body["id"].clone();
                match method.as_str() {
                "notifications/initialized" => {
                    respond(&mut stream, "application/json", "{}".to_string()).await
                }
                "initialize" => respond(&mut stream, "application/json", json!({
                    "jsonrpc":"2.0", "id":id, "result": {
                        "protocolVersion":"2025-11-25", "capabilities":{"tools":{}},
                        "serverInfo":{"name":"fake-http","version":"0.1"}}
                }).to_string()).await,
                "tools/list" => respond(&mut stream, "application/json", json!({
                    "jsonrpc":"2.0", "id":id, "result":{"tools":[{
                        "name":"eof", "description":"eof", "inputSchema":{"type":"object"}}]}
                }).to_string()).await,
                "tools/call" => {
                    assert_eq!(body["params"]["_meta"]["progressToken"], id);
                    calls.push(body);
                    let progress = json!({"jsonrpc":"2.0", "method":"notifications/progress",
                        "params":{"progressToken":id,"progress":1.0}});
                    respond(&mut stream, "text/event-stream",
                        format!("event: message\ndata: {progress}\n\n")).await;
                }
                other => panic!("unexpected HTTP MCP method {other}"),
            }
            }
            calls
        });

    let mut configs = BTreeMap::new();
    configs.insert(
        "http".to_string(),
        McpServerConfig {
            enabled: Some(true),
            timeout: Some(2_000),
            kind: ServerKind::Http,
            command: None,
            args: None,
            env: None,
            cwd: None,
            url: Some(url),
            headers: None,
            extra: BTreeMap::new(),
        },
    );
    let manager = McpManager::new(Arc::new(PathBuf::from(".")), configs);
    let error = manager
        .call("http", "eof", json!({}))
        .await
        .expect_err("SSE EOF must fail");
    assert!(
        error.to_string().contains("may have already run"),
        "{error}"
    );
    let calls = server.await.expect("fake HTTP server");
    assert_eq!(calls.len(), 1, "written request must never be replayed");

    // An HTTP error is also a server response: the request reached the
    // server, so even a normally-stale 503 must not authorize a replay.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind HTTP 503 MCP");
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let mut calls = 0;
        for _ in 0..4 {
            let (mut stream, _) = listener.accept().await.expect("accept HTTP 503 MCP");
            let body = request(&mut stream).await;
            let method = body["method"].as_str().unwrap();
            let id = body["id"].clone();
            match method {
                "notifications/initialized" => {
                    respond(&mut stream, "application/json", "{}".to_string()).await
                }
                "initialize" => respond(&mut stream, "application/json", json!({
                    "jsonrpc":"2.0", "id":id, "result": {
                        "protocolVersion":"2025-11-25", "capabilities":{"tools":{}},
                        "serverInfo":{"name":"fake-http","version":"0.1"}}
                }).to_string()).await,
                "tools/list" => respond(&mut stream, "application/json", json!({
                    "jsonrpc":"2.0", "id":id, "result":{"tools":[{
                        "name":"status", "description":"status", "inputSchema":{"type":"object"}}]}
                }).to_string()).await,
                "tools/call" => {
                    calls += 1;
                    let body = "unavailable";
                    let response = format!(
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    stream.write_all(response.as_bytes()).await.expect("write 503");
                    stream.shutdown().await.expect("shutdown 503");
                }
                other => panic!("unexpected HTTP MCP method {other}"),
            }
        }
        calls
    });
    let mut configs = BTreeMap::new();
    configs.insert(
        "http".to_string(),
        McpServerConfig {
            enabled: Some(true),
            timeout: Some(2_000),
            kind: ServerKind::Http,
            command: None,
            args: None,
            env: None,
            cwd: None,
            url: Some(url),
            headers: None,
            extra: BTreeMap::new(),
        },
    );
    let manager = McpManager::new(Arc::new(PathBuf::from(".")), configs);
    let error = manager
        .call("http", "status", json!({}))
        .await
        .expect_err("503 must fail");
    assert!(
        error.to_string().contains("may have already run"),
        "{error}"
    );
    assert_eq!(server.await.expect("fake HTTP 503 server"), 1);

    // Timeout drops the in-flight POST and emits exactly one cancellation
    // with the original request id, not a replay of the tool call.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind HTTP timeout MCP");
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let server =
        tokio::spawn(async move {
            let mut calls = 0;
            let mut cancellations = 0;
            let mut call_id = Value::Null;
            let mut pending_stream = None;
            for _ in 0..5 {
                let (mut stream, _) = listener.accept().await.expect("accept HTTP timeout MCP");
                let body = request(&mut stream).await;
                let method = body["method"].as_str().unwrap();
                let id = body["id"].clone();
                match method {
                "notifications/initialized" => {
                    respond(&mut stream, "application/json", "{}".to_string()).await
                }
                "initialize" => respond(&mut stream, "application/json", json!({
                    "jsonrpc":"2.0", "id":id, "result": {
                        "protocolVersion":"2025-11-25", "capabilities":{"tools":{}},
                        "serverInfo":{"name":"fake-http","version":"0.1"}}
                }).to_string()).await,
                "tools/list" => respond(&mut stream, "application/json", json!({
                    "jsonrpc":"2.0", "id":id, "result":{"tools":[{
                        "name":"slow", "description":"slow", "inputSchema":{"type":"object"}}]}
                }).to_string()).await,
                "tools/call" => {
                    calls += 1;
                    call_id = id;
                    pending_stream = Some(stream);
                }
                "notifications/cancelled" => {
                    cancellations += 1;
                    assert_eq!(body["params"]["requestId"], call_id);
                    respond(&mut stream, "application/json", "{}".to_string()).await;
                }
                other => panic!("unexpected HTTP MCP method {other}"),
            }
            }
            drop(pending_stream);
            (calls, cancellations)
        });
    let mut configs = BTreeMap::new();
    configs.insert(
        "http".to_string(),
        McpServerConfig {
            enabled: Some(true),
            timeout: Some(100),
            kind: ServerKind::Http,
            command: None,
            args: None,
            env: None,
            cwd: None,
            url: Some(url),
            headers: None,
            extra: BTreeMap::new(),
        },
    );
    let manager = McpManager::new(Arc::new(PathBuf::from(".")), configs);
    let error = manager
        .call("http", "slow", json!({}))
        .await
        .expect_err("timeout must fail");
    assert!(error.to_string().contains("timeout"), "{error}");
    assert_eq!(server.await.expect("fake HTTP timeout server"), (1, 1));
}
