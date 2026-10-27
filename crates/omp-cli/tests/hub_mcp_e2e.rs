#![allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
//! `omt hub mcp` end-to-end: a real stdio MCP client (raw JSON-RPC 2.0, and
//! omt's own `omp-mcp` client in `omt_mcp_client_interop`) talks to the
//! spawned `omt hub mcp` process against a temp `OMP_HUB_DIR`; `omt hub` CLI
//! is the other endpoint. The real `~/.omp/hub` is never touched (temp HOME
//! + explicit hub dir everywhere).
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One throwaway hub + HOME + spawned processes.
struct World {
    home: tempfile::TempDir,
    hub: PathBuf,
    serve: Child,
}

impl World {
    fn new() -> Self {
        let home = tempfile::tempdir().unwrap();
        let hub = home.path().join("hub-dir");
        std::fs::create_dir_all(&hub).unwrap();
        let mut serve = omt(home.path(), &hub)
            .args(["hub", "serve", "--dir"])
            .arg(&hub)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let sock = hub.join("hub.sock");
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            if sock.exists() {
                break;
            }
            if Instant::now() > deadline {
                let _ = serve.kill();
                panic!("hub serve did not create {} in 20s", sock.display());
            }
            if let Some(status) = serve.try_wait().unwrap() {
                panic!("hub serve exited early: {status}");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Self { home, hub, serve }
    }
}

impl Drop for World {
    fn drop(&mut self) {
        let _ = self.serve.kill();
        let _ = self.serve.wait();
    }
}

fn omt(home: &Path, hub: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_omt"));
    cmd.env("HOME", home)
        .env("OMP_HUB_DIR", hub)
        .env("OMP_HUB_NO_AUTOSTART", "1")
        .env("NO_COLOR", "1")
        .env_remove("OMP_HUB_SOCK");
    cmd
}

/// Minimal newline-JSON-RPC MCP stdio client.
struct RawClient {
    stdin: ChildStdin,
    out: BufReader<std::process::ChildStdout>,
    next_id: AtomicU64,
}

fn spawn_mcp(world: &World, extra_args: &[&str]) -> (Child, RawClient) {
    let mut child = omt(world.home.path(), &world.hub)
        .args(["hub", "mcp"])
        .args(extra_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    (
        child,
        RawClient {
            stdin,
            out: BufReader::new(stdout),
            next_id: AtomicU64::new(1),
        },
    )
}

impl RawClient {
    fn request(&mut self, method: &str, params: &serde_json::Value) -> serde_json::Value {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req =
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut line = serde_json::to_string(&req).unwrap();
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let mut buf = String::new();
            assert!(
                Instant::now() <= deadline,
                "timeout waiting for response to {method}"
            );
            match self.out.read_line(&mut buf) {
                Ok(0) => panic!("MCP server closed stdout waiting for {method}"),
                Ok(_) => {}
                Err(e) => panic!("read response: {e}"),
            }
            let msg: serde_json::Value = serde_json::from_str(&buf).unwrap_or_else(|e| {
                panic!("non-JSON line {buf:?}: {e}");
            });
            if msg.get("id").and_then(serde_json::Value::as_u64) == Some(id) {
                return msg;
            }
            // A notification or foreign frame: keep reading.
        }
    }

    fn notify(&mut self, method: &str, params: &serde_json::Value) {
        let msg = serde_json::json!({"jsonrpc": "2.0", "method": method, "params": params});
        let mut line = serde_json::to_string(&msg).unwrap();
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn result(&mut self, method: &str, params: &serde_json::Value) -> serde_json::Value {
        let msg = self.request(method, params);
        assert!(msg.get("error").is_none(), "{method} failed: {msg}");
        msg["result"].clone()
    }
}

fn call(client: &mut RawClient, name: &str, args: &serde_json::Value) -> serde_json::Value {
    let result = client.result(
        "tools/call",
        &serde_json::json!({"name": name, "arguments": args}),
    );
    assert!(
        result.get("isError").is_none(),
        "tool {name} returned error: {}",
        serde_json::to_string_pretty(&result).unwrap()
    );
    result["structuredContent"].clone()
}

fn cli(world: &World, args: &[&str]) -> std::process::Output {
    omt(world.home.path(), &world.hub)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn omt {args:?}: {e}"))
}

fn cli_json(world: &World, args: &[&str]) -> serde_json::Value {
    let mut a = args.to_vec();
    a.push("--json");
    let out = cli(world, &a);
    assert!(
        out.status.success(),
        "omt {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "bad JSON from omt {args:?}: {e}; stdout={}",
            String::from_utf8_lossy(&out.stdout)
        )
    })
}

#[test]
#[allow(clippy::too_many_lines)] // one scripted MCP session; claims and messages build on each other
fn mcp_round_trip_against_temp_hub() {
    let world = World::new();
    let (mut child, mut c) = spawn_mcp(&world, &["--agent", "itest"]);

    // initialize: negotiate 2025-06-18, remember nothing yet.
    let init = c.result(
        "initialize",
        &serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "raw-test", "version": "0"},
        }),
    );
    assert_eq!(init["protocolVersion"], "2025-06-18");
    assert_eq!(init["serverInfo"]["name"], "omt-hub");
    c.notify("notifications/initialized", &serde_json::json!({}));

    // tools/list: exactly the nine hub tools.
    let tools = c.result("tools/list", &serde_json::json!({}));
    let names: Vec<&str> = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["send", "inbox", "ack", "answer", "status", "claim", "release", "check", "claims"]
    );

    // send -> CLI endpoint reads, acks, answers -> MCP inbox sees the answer.
    let schema = serde_json::json!({"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]});
    let receipt = call(
        &mut c,
        "send",
        &serde_json::json!({
            "to": "cli-bob",
            "payload": "what is 6*7?",
            "expected": "json answer",
            "reply_schema": schema,
        }),
    );
    let question_id = receipt["message_id"].as_str().unwrap().to_string();
    assert_eq!(receipt["state"], "sent");
    assert!(!question_id.is_empty());

    let inbox = cli_json(&world, &["hub", "inbox", "--recipient", "cli-bob"]);
    let envs = inbox["envelopes"].as_array().unwrap();
    assert_eq!(envs.len(), 1);
    assert_eq!(envs[0]["sender"], "itest");
    assert_eq!(envs[0]["payload"], "what is 6*7?");

    cli_json(
        &world,
        &[
            "hub",
            "ack",
            "--message-id",
            &question_id,
            "--state",
            "read",
        ],
    );
    let ans = cli_json(
        &world,
        &[
            "hub",
            "answer",
            "--causation-id",
            &question_id,
            "--from",
            "cli-bob",
            "--payload",
            "{\"answer\":42}",
        ],
    );
    assert_eq!(ans["question"]["state"], "answered");

    let mine = call(&mut c, "inbox", &serde_json::json!({"mark_read": true}));
    let mine_envs = mine["envelopes"].as_array().unwrap();
    assert_eq!(mine_envs.len(), 1);
    assert_eq!(mine_envs[0]["kind"], "agent.answer");
    assert_eq!(mine_envs[0]["causation_id"], question_id);
    assert_eq!(mine_envs[0]["sender"], "cli-bob");
    assert!(mine["mark_read_errors"].as_array().unwrap().is_empty());

    // Status: no more unread for itest; question answered with reply id.
    let st = call(&mut c, "status", &serde_json::json!({}));
    assert!(st["unread"].as_array().unwrap().is_empty());
    assert!(st["pending_replies"].as_array().unwrap().is_empty());
    let q = call(
        &mut c,
        "status",
        &serde_json::json!({"message_id": question_id}),
    );
    assert_eq!(q["state"], "answered");
    assert_eq!(q["reply_message_id"], ans["answer"]["message_id"]);

    // Large payloads go by payload_ref with server-computed sha256/bytes.
    let dir = tempfile::tempdir().unwrap();
    let big = dir.path().join("big.txt");
    let body = "Lorem ipsum dolor sit amet. ".repeat(400); // > 8 KiB
    std::fs::write(&big, &body).unwrap();
    let sent = call(
        &mut c,
        "send",
        &serde_json::json!({"to": "cli-bob", "payload_path": big.to_str().unwrap()}),
    );
    assert!(sent["state"].is_string());
    let bob_inbox = cli_json(&world, &["hub", "inbox", "--recipient", "cli-bob"]);
    let refr = bob_inbox["envelopes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["message_id"] == sent["message_id"])
        .and_then(|e| e["payload_ref"].as_object())
        .expect("payload_ref present");
    assert_eq!(
        usize::try_from(refr["bytes"].as_u64().unwrap()).unwrap(),
        body.len()
    );
    assert_eq!(refr["sha256"].as_str().unwrap().len(), 64);
    assert!(refr["path"].as_str().unwrap().ends_with("big.txt"));

    // Claims: MCP identity claims, CLI owner conflicts, fencing on release.
    let claim = call(
        &mut c,
        "claim",
        &serde_json::json!({"resource": "itest/res", "ttl_s": 60, "intent": "testing"}),
    );
    assert_eq!(claim["granted"], true);
    let token = claim["fencing_token"].as_u64().unwrap();
    // A conflicting claim exits 1 by design; parse the JSON from stdout.
    let conflict_out = cli(
        &world,
        &[
            "hub",
            "claim",
            "--resource",
            "itest/res",
            "--owner",
            "other",
            "--ttl-s",
            "60",
            "--json",
        ],
    );
    assert!(!conflict_out.status.success());
    let conflict: serde_json::Value = serde_json::from_slice(&conflict_out.stdout).unwrap();
    assert_eq!(conflict["granted"], false);
    assert_eq!(conflict["holder"], "itest");
    let held = call(
        &mut c,
        "check",
        &serde_json::json!({"resource": "itest/res"}),
    );
    assert_eq!(held["held"], true);
    assert_eq!(held["holder"]["owner"], "itest");
    let wrong = c.result(
        "tools/call",
        &serde_json::json!({"name": "release", "arguments": {"resource": "itest/res", "fencing_token": token + 1}}),
    );
    assert_eq!(wrong["isError"], true);
    let ok = call(
        &mut c,
        "release",
        &serde_json::json!({"resource": "itest/res", "fencing_token": token}),
    );
    assert_eq!(ok["released"], true);
    let free = call(
        &mut c,
        "check",
        &serde_json::json!({"resource": "itest/res"}),
    );
    assert_eq!(free["held"], false);
    let _ = cli(
        &world,
        &[
            "hub",
            "claim",
            "--resource",
            "itest/keep",
            "--owner",
            "keeper",
            "--ttl-s",
            "120",
        ],
    );
    let claims = call(&mut c, "claims", &serde_json::json!({"prefix": "itest/"}));
    let listed: Vec<&str> = claims["claims"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["resource"].as_str().unwrap())
        .collect();
    assert_eq!(listed, ["itest/keep"]);

    // Schema validation: an answer violating reply_schema is an isError result.
    let q2 = call(
        &mut c,
        "send",
        &serde_json::json!({"to": "cli-bob", "reply_schema": schema, "payload": "again"}),
    );
    let bad = c.result(
        "tools/call",
        &serde_json::json!({
            "name": "answer",
            "arguments": {"causation_id": q2["message_id"], "payload": "{\"answer\":\"nope\"}"},
        }),
    );
    assert_eq!(bad["isError"], true);
    let detail = bad["structuredContent"]["error"].as_str().unwrap();
    assert!(detail.contains("reply_schema"), "{detail}");

    // Unknown tool and unknown method.
    let no_tool = c.result(
        "tools/call",
        &serde_json::json!({"name": "nope", "arguments": {}}),
    );
    assert_eq!(no_tool["isError"], true);
    let no_method = c.request("resources/list", &serde_json::json!({}));
    assert_eq!(no_method["error"]["code"], -32601);

    // Clean shutdown: close stdin, server exits 0.
    drop(c.stdin);
    let status = child.wait().unwrap();
    assert!(status.success(), "mcp server exit: {status}");
    let mut err = String::new();
    child.stderr.unwrap().read_to_string(&mut err).unwrap();
    assert!(err.contains("omt hub mcp"), "stderr: {err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn omt_mcp_client_interop() {
    let world = World::new();
    // omt's own MCP client (crates/omp-mcp) against `omt hub mcp`.
    let config = omp_mcp::config::McpServerConfig {
        enabled: Some(true),
        timeout: Some(30_000),
        kind: omp_mcp::config::ServerKind::Stdio,
        command: Some(env!("CARGO_BIN_EXE_omt").to_string()),
        args: Some(
            ["hub", "mcp", "--agent", "omcp-client"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        ),
        env: Some(std::collections::BTreeMap::from([
            ("HOME".to_string(), world.home.path().display().to_string()),
            ("OMP_HUB_DIR".to_string(), world.hub.display().to_string()),
            ("OMP_HUB_NO_AUTOSTART".to_string(), "1".to_string()),
        ])),
        cwd: None,
        url: None,
        headers: None,
        extra: std::collections::BTreeMap::default(),
    };
    let cwd = Arc::new(world.home.path().to_path_buf());
    let mut conn = omp_mcp::client::connect_to_server("omt-hub", &config, cwd)
        .await
        .unwrap_or_else(|e| panic!("connect_to_server: {e}"));

    let tools = omp_mcp::client::list_tools(&mut conn)
        .await
        .unwrap_or_else(|e| panic!("list_tools: {e}"));
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"send"), "{names:?}");
    assert!(names.contains(&"inbox"), "{names:?}");

    let sent = omp_mcp::client::call_tool(
        &mut conn,
        "send",
        serde_json::json!({"to": "rust-endpoint", "payload": "hello from omp-mcp"}),
    )
    .await
    .unwrap_or_else(|e| panic!("call_tool send: {e}"));
    assert!(!sent.is_error, "{sent:?}");
    let receipt = sent.structured_content.expect("structuredContent");
    assert!(receipt["message_id"].is_string());
    assert_eq!(receipt["state"], "sent");

    // The CLI endpoint sees exactly this envelope.
    let inbox = cli_json(&world, &["hub", "inbox", "--recipient", "rust-endpoint"]);
    let envs = inbox["envelopes"].as_array().unwrap();
    assert_eq!(envs.len(), 1);
    assert_eq!(envs[0]["message_id"], receipt["message_id"]);
    assert_eq!(envs[0]["payload"], "hello from omp-mcp");
}
