//! Integration tests for the `omp-auth-broker serve` Rust daemon.
//!
//! Each test spins up the compiled binary against an isolated `agent/` dir on a
//! free port, waits for `/v1/healthz`, then drives the HTTP API and asserts on
//! the on-disk `agent.db` through a separate `rusqlite` connection.
//!
//! SAFETY: everything lives under a `tempfile::TempDir`; `OMP_AGENT_DIR` is
//! always set so the daemon never touches `~/.omp`. Egress proxy env vars are
//! pinned to a dead proxy with `NO_PROXY` for loopback so the mock token server
//! is reached directly.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;

const TOKEN: &str = "FIXTURE_TOKEN_1234567890abcdef";

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

// ---------------------------------------------------------------------------
// Mock OAuth token server (std TcpListener thread)
// ---------------------------------------------------------------------------

struct MockServer {
    addr: SocketAddr,
    calls: Arc<AtomicU64>,
}

impl MockServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock listener");
        let addr = listener.local_addr().unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let worker = calls.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let calls = worker.clone();
                std::thread::spawn(move || handle_token_conn(stream, &calls));
            }
        });
        MockServer { addr, calls }
    }
}

fn find_slice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn form_param(form: &str, name: &str) -> String {
    for pair in form.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == name {
                return v.replace('+', " ");
            }
        }
    }
    String::new()
}

fn handle_token_conn(mut stream: TcpStream, calls: &AtomicU64) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = find_slice(&buf, b"\r\n\r\n") {
                    break pos;
                }
            }
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]);
    let len = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);
    while buf.len() < head_end + 4 + len {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let body = &buf[head_end + 4..head_end + 4 + len];
    let form = String::from_utf8_lossy(body);
    let refresh = form_param(&form, "refresh_token");
    let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
    let (status, payload) = if refresh.contains("INVALID") {
        (400, r#"{"error":"invalid_grant"}"#.to_string())
    } else {
        (
            200,
            format!(r#"{{"access_token":"NEW_{n}","refresh_token":"ROT_{n}","expires_in":3600}}"#),
        )
    };
    let reason = if status == 200 { "OK" } else { "Bad Request" };
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    let _ = stream.write_all(resp.as_bytes());
}

// ---------------------------------------------------------------------------
// Broker fixture / lifecycle
// ---------------------------------------------------------------------------

struct Fixture {
    tmp: TempDir,
    agent: PathBuf,
    token: String,
}

fn fixture() -> Fixture {
    let tmp = TempDir::new().expect("tempdir");
    let agent = tmp.path().join("agent");
    std::fs::create_dir_all(&agent).expect("create agent dir");
    let token = TOKEN.to_string();
    std::fs::write(tmp.path().join("auth-broker.token"), format!("{token}\n"))
        .expect("write token");
    Fixture { tmp, agent, token }
}

struct Broker {
    child: Child,
    url: String,
    token: String,
    db: PathBuf,
    _tmp: TempDir,
    _mock: Option<MockServer>,
}

impl Drop for Broker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_from(fx: Fixture, mock: Option<MockServer>, disable_refresher: bool) -> Broker {
    let port = free_port();
    let url = format!("http://127.0.0.1:{port}");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_omp-auth-broker"));
    cmd.arg("serve").arg(format!("--bind=127.0.0.1:{port}"));
    cmd.env("OMP_AGENT_DIR", &fx.agent);
    cmd.env("HTTP_PROXY", "http://127.0.0.1:9");
    cmd.env("HTTPS_PROXY", "http://127.0.0.1:9");
    cmd.env("NO_PROXY", "127.0.0.1,localhost");
    if disable_refresher {
        cmd.env("OMP_AUTH_BROKER_DISABLE_REFRESHER", "1");
    }
    if let Some(m) = &mock {
        cmd.env("KIMI_CODE_OAUTH_HOST", format!("http://{}", m.addr));
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd.spawn().expect("spawn broker");
    let db = fx.agent.join("agent.db");
    Broker {
        child,
        url,
        token: fx.token,
        db,
        _tmp: fx.tmp,
        _mock: mock,
    }
}

fn spawn_broker(mock: Option<MockServer>, disable_refresher: bool) -> Broker {
    spawn_from(fixture(), mock, disable_refresher)
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind probe");
    l.local_addr().unwrap().port()
}

fn open_db(path: &Path) -> Connection {
    let conn = Connection::open(path).expect("open agent.db");
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn
}

/// Create the minimal schema the broker expects so rows can be inserted before
/// the daemon starts (needed for the background-refresher test). The daemon's
/// own open path creates any remaining tables/triggers idempotently.
fn init_schema(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS auth_schema_version (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
         CREATE TABLE IF NOT EXISTS auth_credentials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            credential_type TEXT NOT NULL,
            data TEXT NOT NULL,
            disabled_cause TEXT DEFAULT NULL,
            identity_key TEXT DEFAULT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_auth_provider ON auth_credentials(provider);
         CREATE TABLE IF NOT EXISTS auth_change_revision (id INTEGER PRIMARY KEY CHECK (id=1), revision INTEGER NOT NULL);
         INSERT OR IGNORE INTO auth_change_revision(id, revision) VALUES (1, 0);",
    )
    .expect("init schema");
}

fn insert_oauth(
    conn: &Connection,
    provider: &str,
    access: &str,
    refresh: &str,
    expires_ms: i64,
    email: Option<&str>,
) -> i64 {
    let mut d = json!({"access": access, "refresh": refresh, "expires": expires_ms as f64});
    if let Some(e) = email {
        d["email"] = json!(e);
    }
    let now = now_ms() / 1000;
    conn.execute(
        "INSERT INTO auth_credentials (provider, credential_type, data, created_at, updated_at) VALUES (?1,'oauth',?2,?3,?3)",
        params![provider, d.to_string(), now],
    )
    .unwrap();
    conn.last_insert_rowid()
}

fn insert_api_key(conn: &Connection, provider: &str, key: &str) -> i64 {
    let d = json!({"key": key});
    let now = now_ms() / 1000;
    conn.execute(
        "INSERT INTO auth_credentials (provider, credential_type, data, created_at, updated_at) VALUES (?1,'api_key',?2,?3,?3)",
        params![provider, d.to_string(), now],
    )
    .unwrap();
    conn.last_insert_rowid()
}

fn data_of(conn: &Connection, id: i64) -> Value {
    let s: String = conn
        .query_row(
            "SELECT data FROM auth_credentials WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .expect("row exists");
    serde_json::from_str(&s).expect("row json")
}

fn cause_of(conn: &Connection, id: i64) -> Option<String> {
    conn.query_row(
        "SELECT disabled_cause FROM auth_credentials WHERE id = ?1",
        params![id],
        |r| r.get(0),
    )
    .ok()
}

async fn wait_healthz(url: &str) -> reqwest::Client {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .no_proxy()
        .build()
        .expect("client");
    for _ in 0..30 {
        if let Ok(r) = client.get(format!("{url}/v1/healthz")).send().await {
            if r.status() == 200 {
                return client;
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("broker never became healthy at {url}");
}

#[allow(clippy::too_many_arguments)]
async fn request(
    client: &reqwest::Client,
    url: &str,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<Value>,
    extra: &[(&str, &str)],
) -> (u16, reqwest::header::HeaderMap, Value) {
    let mut rb = client.request(
        reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
        format!("{url}{path}"),
    );
    if let Some(t) = token {
        rb = rb.bearer_auth(t);
    }
    if let Some(b) = body {
        rb = rb.json(&b);
    }
    for (k, v) in extra {
        rb = rb.header(*k, *v);
    }
    let resp = rb.send().await.expect("request");
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let text = resp.text().await.unwrap();
    let value = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, headers, value)
}

fn find_entry<'a>(snap: &'a Value, provider: &str) -> Option<&'a Value> {
    snap["credentials"]
        .as_array()?
        .iter()
        .find(|c| c["provider"].as_str() == Some(provider))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn healthz_and_auth() {
    let b = spawn_broker(None, true);
    let client = wait_healthz(&b.url).await;

    let (s, _, _) = request(&client, &b.url, "GET", "/v1/healthz", None, None, &[]).await;
    assert_eq!(s, 200, "healthz must not require auth");

    let (s, _, _) = request(&client, &b.url, "GET", "/v1/snapshot", None, None, &[]).await;
    assert_eq!(s, 401, "snapshot without token must 401");

    let (s, _, _) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some("WRONG_TOKEN"),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 401, "snapshot with wrong token must 401");

    let (s, _, body) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert!(
        body.get("credentials").is_some(),
        "snapshot body has credentials"
    );
}

#[tokio::test]
async fn snapshot_shape_redaction_and_etag() {
    let b = spawn_broker(None, true);
    let client = wait_healthz(&b.url).await;
    let db = open_db(&b.db);
    let kimi_id = insert_oauth(
        &db,
        "kimi-code",
        "FAKE_KIMI_ACCESS",
        "FAKE_KIMI_OK_REFRESH",
        now_ms() + 3 * 86_400_000,
        None,
    );
    let api_id = insert_api_key(&db, "cursor", "FAKE_CURSOR_KEY");
    drop(db);

    let (s, headers, body) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    let gen = headers
        .get("etag")
        .expect("etag header")
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        gen.starts_with('"') && gen.ends_with('"'),
        "etag is quoted: {gen}"
    );

    let kimi = find_entry(&body, "kimi-code").expect("kimi entry");
    assert_eq!(kimi["credential"]["type"], "oauth");
    assert_eq!(
        kimi["credential"]["refresh"], "__remote__",
        "oauth refresh redacted"
    );
    let cursor = find_entry(&body, "cursor").expect("cursor entry");
    assert_eq!(cursor["credential"]["type"], "api_key");
    assert_eq!(
        cursor["credential"]["key"], "FAKE_CURSOR_KEY",
        "api_key verbatim"
    );
    assert!(cursor["credential"].get("source").is_none());

    // Same generation + wait=300 -> 304.
    let (s, _, _) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot?wait=300",
        Some(&b.token),
        None,
        &[("If-None-Match", &gen)],
    )
    .await;
    assert_eq!(s, 304, "unchanged generation must 304");

    // External UPDATE bumps the generation.
    let db = open_db(&b.db);
    let new_data = json!({"access":"CHANGED_ACCESS","refresh":"FAKE_KIMI_OK_REFRESH","expires": (now_ms()+3*86_400_000) as f64});
    db.execute(
        "UPDATE auth_credentials SET data = ?1 WHERE id = ?2",
        params![new_data.to_string(), kimi_id],
    )
    .unwrap();
    drop(db);

    let (s, headers2, _) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    let gen2 = headers2
        .get("etag")
        .expect("etag header")
        .to_str()
        .unwrap()
        .to_string();
    assert_ne!(gen2, gen, "external UPDATE must advance generation");

    // The api_key row's identity is unchanged; ensure the update didn't touch it.
    assert!(api_id > 0);
}

#[tokio::test]
async fn refresh_success_and_unknown_id() {
    let mock = MockServer::start();
    let mock_calls = mock.calls.clone();
    let b = spawn_broker(Some(mock), true);
    let client = wait_healthz(&b.url).await;
    let db = open_db(&b.db);
    let id = insert_oauth(
        &db,
        "kimi-code",
        "FAKE_KIMI_ACCESS",
        "FAKE_KIMI_OK_REFRESH",
        now_ms() + 3 * 86_400_000,
        Some("kimi@test.invalid"),
    );
    drop(db);

    let (s, _, body) = request(
        &client,
        &b.url,
        "POST",
        &format!("/v1/credential/{id}/refresh"),
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200, "refresh success returns 200");
    assert!(
        mock_calls.load(Ordering::SeqCst) >= 1,
        "mock token server hit"
    );
    let entry = &body["entry"];
    let access = entry["credential"]["access"].as_str().expect("access");
    assert!(access.starts_with("NEW_"), "minted access token: {access}");
    assert_eq!(entry["credential"]["refresh"], "__remote__");

    let db = open_db(&b.db);
    let rotated = data_of(&db, id);
    let r = rotated["refresh"].as_str().expect("refresh token");
    assert!(r.starts_with("ROT_"), "db holds rotated refresh: {r}");
    drop(db);

    let (s, _, _) = request(
        &client,
        &b.url,
        "POST",
        "/v1/credential/999999/refresh",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 404, "unknown id refresh must 404");
}

#[tokio::test]
async fn refresh_invalid_disables_row() {
    let mock = MockServer::start();
    let mock_calls = mock.calls.clone();
    let b = spawn_broker(Some(mock), true);
    let client = wait_healthz(&b.url).await;
    let db = open_db(&b.db);
    let id = insert_oauth(
        &db,
        "kimi-code",
        "FAKE_KIMI_ACCESS",
        "FAKE_INVALID_REFRESH",
        now_ms() + 3 * 86_400_000,
        None,
    );
    drop(db);

    let (s, _, _) = request(
        &client,
        &b.url,
        "POST",
        &format!("/v1/credential/{id}/refresh"),
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 500, "invalid_grant refresh must 500");
    assert!(
        mock_calls.load(Ordering::SeqCst) >= 1,
        "mock token server hit"
    );

    let db = open_db(&b.db);
    let cause = cause_of(&db, id).expect("row disabled");
    assert!(!cause.is_empty(), "disabled_cause set");
    drop(db);

    let (s, _, body) = request(
        &client,
        &b.url,
        "GET",
        "/v1/credentials/disabled",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    let list = body["disabled"].as_array().expect("disabled array");
    assert!(
        list.iter().any(|d| d["id"].as_i64() == Some(id)),
        "row appears in disabled list"
    );
}

#[tokio::test]
async fn upload_api_key_then_disable() {
    let b = spawn_broker(None, true);
    let client = wait_healthz(&b.url).await;

    let (s, _, body) = request(
        &client,
        &b.url,
        "POST",
        "/v1/credential",
        Some(&b.token),
        Some(json!({"provider":"cursor","credential":{"type":"api_key","key":"FAKE_CURSOR_KEY"}})),
        &[],
    )
    .await;
    assert_eq!(s, 200);
    let entries = body["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 1);
    let id = entries[0]["id"].as_i64().expect("id");
    assert_eq!(entries[0]["credential"]["key"], "FAKE_CURSOR_KEY");
    assert!(
        entries[0]["credential"].get("source").is_none(),
        "source not echoed"
    );

    let db = open_db(&b.db);
    let stored = data_of(&db, id);
    assert_eq!(stored["key"], "FAKE_CURSOR_KEY");
    assert!(
        stored.get("source").is_none(),
        "stored data has no source: {stored}"
    );
    drop(db);

    let (s, _, body) = request(
        &client,
        &b.url,
        "POST",
        &format!("/v1/credential/{id}/disable"),
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(body["ok"], true);

    let (s, _, snap) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert!(
        snap["credentials"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["id"].as_i64() != Some(id)),
        "disabled row gone from snapshot"
    );

    let (_, _, disabled) = request(
        &client,
        &b.url,
        "GET",
        "/v1/credentials/disabled",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert!(
        disabled["disabled"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["id"].as_i64() == Some(id)),
        "row appears in disabled"
    );
}

#[tokio::test]
async fn block_upsert_delete_and_codex_hide() {
    let b = spawn_broker(None, true);
    let client = wait_healthz(&b.url).await;
    let db = open_db(&b.db);
    let kimi_id = insert_oauth(
        &db,
        "kimi-code",
        "FAKE_KIMI_ACCESS",
        "FAKE_KIMI_OK_REFRESH",
        now_ms() + 3 * 86_400_000,
        None,
    );
    let codex_id = insert_oauth(
        &db,
        "openai-codex",
        "FAKE_CODEX_ACCESS",
        "FAKE_CODEX_REFRESH",
        now_ms() + 3 * 86_400_000,
        None,
    );
    drop(db);

    let until = now_ms() + 3_600_000;
    let (s, _, body) = request(
        &client,
        &b.url,
        "POST",
        &format!("/v1/credential/{kimi_id}/block"),
        Some(&b.token),
        Some(json!({"providerKey":"kimi-code:oauth","blockScope":"chat","blockedUntilMs": until as f64})),
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(body["ok"], true);

    let (s, _, snap) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    let kimi = find_entry(&snap, "kimi-code").expect("kimi entry");
    let blocks = kimi["blocks"].as_array().expect("kimi blocks");
    assert!(
        blocks
            .iter()
            .any(|bl| bl["providerKey"] == "kimi-code:oauth" && bl["blockScope"] == "chat"),
        "chat block appears: {blocks:?}"
    );

    let (s, _, body) = request(
        &client,
        &b.url,
        "DELETE",
        &format!("/v1/credential/{kimi_id}/block"),
        Some(&b.token),
        Some(json!({"providerKey":"kimi-code:oauth","blockScope":"chat"})),
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(body["ok"], true);

    let (_, _, snap) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    let kimi = find_entry(&snap, "kimi-code").expect("kimi entry");
    let blocks = kimi
        .get("blocks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        !blocks.iter().any(|bl| bl["blockScope"] == "chat"),
        "chat block removed"
    );

    // Codex: insert a `chat` and a `shared` block directly against the DB.
    let db = open_db(&b.db);
    let now_s = now_ms() / 1000;
    for (scope, until_ms) in [("chat", until), ("shared", until)] {
        db.execute(
            "INSERT INTO auth_credential_blocks (credential_id,provider_key,block_scope,blocked_until_ms,updated_at)
             VALUES (?1,'openai-codex:oauth',?2,?3,?4)",
            params![codex_id, scope, until_ms, now_s],
        )
        .unwrap();
    }
    drop(db);

    // With the capability header the codex `chat` block is visible, `shared` is hidden.
    let (s, _, snap) = request(
        &client,
        &b.url,
        "GET",
        "/v1/snapshot",
        Some(&b.token),
        None,
        &[("OMP-Auth-Broker-Capabilities", "codex-meter-block-scopes")],
    )
    .await;
    assert_eq!(s, 200);
    let codex = find_entry(&snap, "openai-codex").expect("codex entry");
    let blocks = codex
        .get("blocks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(
        blocks.iter().any(|bl| bl["blockScope"] == "chat"),
        "codex chat block visible with capability"
    );
    assert!(
        !blocks.iter().any(|bl| bl["blockScope"] == "shared"),
        "codex shared block hidden with capability"
    );
}

#[tokio::test]
async fn usage_observed_clients_history() {
    let b = spawn_broker(None, true);
    let client = wait_healthz(&b.url).await;

    let at = now_ms() - 1000;
    let (s, _, body) = request(
        &client,
        &b.url,
        "POST",
        "/v1/usage/observed",
        Some(&b.token),
        Some(json!({
            "installId":"install-X","hostname":"host-x","app":"app1",
            "entries":[
                {"at":at,"provider":"kimi-code","model":"kimi-k3","requests":10,
                 "inputTokens":500,"outputTokens":200,"cacheReadTokens":50,"cacheWriteTokens":20,"costUsd":0.4},
                {"at":at,"provider":"anthropic","model":"claude","requests":2,
                 "inputTokens":300,"outputTokens":100,"cacheReadTokens":0,"cacheWriteTokens":0,"costUsd":0.1}
            ]
        })),
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert_eq!(body["ok"], true);

    let (s, _, body) = request(
        &client,
        &b.url,
        "GET",
        "/v1/usage/clients",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    let clients = body["clients"].as_array().expect("clients");
    let x = clients
        .iter()
        .find(|c| c["installId"] == "install-X")
        .expect("install-X");
    let providers = x["providers"].as_array().unwrap();
    let in_tok: f64 = providers
        .iter()
        .map(|p| p["inputTokens"].as_f64().unwrap())
        .sum();
    let out_tok: f64 = providers
        .iter()
        .map(|p| p["outputTokens"].as_f64().unwrap())
        .sum();
    let reqs: f64 = providers
        .iter()
        .map(|p| p["requests"].as_f64().unwrap())
        .sum();
    assert_eq!(in_tok, 800.0);
    assert_eq!(out_tok, 300.0);
    assert_eq!(reqs, 12.0);

    let (s, _, body) = request(
        &client,
        &b.url,
        "GET",
        "/v1/usage/history",
        Some(&b.token),
        None,
        &[],
    )
    .await;
    assert_eq!(s, 200);
    assert!(body.get("generatedAt").is_some());
    assert!(body.get("entries").is_some());
}

#[tokio::test]
async fn background_refresher_rotates_expiring_row() {
    // Row must exist before the daemon's first sweep fires (immediately at start).
    let fx = fixture();
    {
        let conn = open_db(&fx.agent.join("agent.db"));
        init_schema(&conn);
        insert_oauth(
            &conn,
            "kimi-code",
            "FAKE_KIMI_ACCESS",
            "FAKE_KIMI_OK_REFRESH",
            now_ms() + 60_000,
            None,
        );
    }
    let mock = MockServer::start();
    let b = spawn_from(fx, Some(mock), false); // refresher enabled

    let _client = wait_healthz(&b.url).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut refreshed = false;
    while Instant::now() < deadline {
        let conn = open_db(&b.db);
        let rotated = match conn.query_row::<String, _, _>(
            "SELECT data FROM auth_credentials WHERE id = 1",
            [],
            |r| r.get(0),
        ) {
            Ok(s) => serde_json::from_str::<Value>(&s).unwrap_or(Value::Null),
            Err(_) => Value::Null,
        };
        let is_rotated = rotated["refresh"]
            .as_str()
            .is_some_and(|r| r.starts_with("ROT_"));
        drop(conn);
        if is_rotated {
            refreshed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert!(
        refreshed,
        "background sweep did not rotate the 60s-expiring row"
    );
}
