#![allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
//! Protocol-level guard for web-UI view-switch latency: a real `omp-rs web`
//! hub runs against a temp HOME with a seeded 400-message session, and
//! `ATTACH_CYCLES` times a view switch is replayed at the protocol level —
//! fresh WebSocket to the bridge plus the exact RPC set the UI issues on
//! attach (`get_state` + `get_messages` + `get_available_commands`; same
//! measurement as `bench/results/viewswitch-2026-10-24/bench-proto.mjs`).
//!
//! What is measured: the server-side path only (hub bridge + `--mode rpc`
//! child loading the session), not UI rendering — the browser adds ~200 ms
//! on top of the protocol attach. The guard exists for the degradation class
//! of bench/results 2026-10-14, where resuming a 24 MB session took 3.9 s.
//!
//! Ceilings, per attach, measured from connect start to the last of the
//! three responses: `p95 < VIEW_SWITCH_CEILING_P95_MS` and
//! `max < VIEW_SWITCH_CEILING_MAX_MS`; a violation panics with the full
//! histogram.
//!
//! Self-contained: loopback only, no LLM provider contact. The hub's rpc
//! children resolve their model from `OMP_RS_BASE_URL`/`OMP_RS_MODEL`
//! (see `model::select` in `src/model.rs`), which synthesizes a fixture
//! model without touching provider catalogs; nothing dials the endpoint
//! because no agent turn is ever streamed.
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use futures::{SinkExt as _, StreamExt as _};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::handshake::client::generate_key;
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

/// Message entries seeded after the file header (bench-proto's transcript
/// shape: alternating user question / 1 KiB assistant answer chain).
const SEEDED_MESSAGES: usize = 400;
/// Measured attach cycles, after one warmup attach (like the bench script).
const ATTACH_CYCLES: usize = 10;
/// Cool-down between cycles, so runs stay independent like human-paced
/// switches in the UI.
const CYCLE_PAUSE: Duration = Duration::from_millis(100);
/// Protocol attach of a 400-message session must stay far below a human
/// noticeable switch: the UI render adds ~200 ms on top of this number.
/// Guard precedent: bench/results 2026-10-14, where a 24 MB resume took
/// 3.9 s. The 2026-10-24 protocol bench measured max 34.8 ms.
const VIEW_SWITCH_CEILING_P95_MS: f64 = 250.0;
/// Protocol attach of a 400-message session; UI render adds ~200 ms on top;
/// degradation precedent bench/results 2026-10-14 (24 MB resume, 3.9 s).
const VIEW_SWITCH_CEILING_MAX_MS: f64 = 500.0;
/// One attach must resolve within this budget or the run fails outright.
const ATTACH_TIMEOUT: Duration = Duration::from_secs(15);
/// `omp-rs web` startup banner budget.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
/// Hub API calls must answer within this budget.
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
/// Forwarded bridge frames kept per attach for failure diagnostics.
const FRAME_TRAIL: usize = 32;
/// stderr lines included in failure panics.
const STDERR_TAIL: usize = 40;
/// Stand-in provider endpoint for the fixture model: loopback, port 9
/// (discard, never bound). Model resolution only reads the variable; the
/// endpoint is never dialed because no turn is streamed.
const FIXTURE_BASE_URL: &str = "http://127.0.0.1:9/";
/// Any `provider/id` resolves against [`FIXTURE_BASE_URL`]; the value is inert.
const FIXTURE_MODEL: &str = "custom/view-switch-fixture";

/// The running `omp-rs web` child: killed on drop, its stderr kept on disk
/// for failure diagnostics.
struct WebServer {
    child: tokio::process::Child,
    port: u16,
    err_path: PathBuf,
    /// Keeps the stderr file alive until the temp HOME goes away.
    _err_file: tempfile::NamedTempFile,
}

impl Drop for WebServer {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Timings of one attach: from connect start (`total_ms`) and from socket
/// open (`rpc_ms`) to the last of the three responses, plus per-RPC splits
/// and the resumed transcript size.
struct Attach {
    total_ms: f64,
    rpc_ms: f64,
    ws_open_ms: f64,
    get_state_ms: f64,
    get_messages_ms: f64,
    get_available_commands_ms: f64,
    messages: Option<usize>,
}

/// Seed one stored session in the temp store (`seed.mjs` shape: title +
/// session header + parent-linked message chain) and return its path.
fn seed_session(home: &Path, proj: &Path) -> PathBuf {
    let id = "01aaaaaa-0000-7000-8000-00000000000a";
    let stamp = "2026-10-24T00:00:00.000Z";
    // `persist::sessions_dir` slug: the cwd with `/` replaced.
    let dir = home
        .join(".omp-rs/sessions")
        .join(proj.display().to_string().replace('/', "-"));
    std::fs::create_dir_all(&dir).expect("seed session dir");
    let mut lines = vec![
        json!({"type": "title", "v": 1, "title": "view-switch fixture", "updatedAt": stamp, "pad": ""})
            .to_string(),
        json!({"type": "session", "version": 3, "id": id, "timestamp": stamp, "cwd": proj.display().to_string()})
            .to_string(),
    ];
    let mut parent = None;
    for i in 0..SEEDED_MESSAGES {
        let assistant = i % 2 == 1;
        let mid = format!("{id}-m{i}");
        let text = if assistant {
            "a".repeat(1024)
        } else {
            format!("question {i}")
        };
        lines.push(
            json!({
                "type": "message",
                "id": mid,
                "parentId": parent,
                "timestamp": stamp,
                "message": {
                    "role": if assistant { "assistant" } else { "user" },
                    "content": [{"type": "text", "text": text}],
                    "attribution": if assistant { "agent" } else { "user" },
                    "timestamp": 1_791_303_310_547u64 + i as u64,
                },
            })
            .to_string(),
        );
        parent = Some(mid);
    }
    let file = dir.join(format!("2026-10-24T00-00-00-000Z_{id}.jsonl"));
    std::fs::write(&file, lines.join("\n") + "\n").expect("seed session file");
    file
}

/// Port from the startup banner: `omt web UI: http://127.0.0.1:<port>/`.
fn banner_port(line: &str) -> Option<u16> {
    line.strip_prefix("omt web UI: http://127.0.0.1:")?
        .split('/')
        .next()?
        .parse()
        .ok()
}

/// Start the hub bound to an ephemeral port (`--port 0`, real port parsed
/// from the banner). `HOME` is the temp dir, so the seeded session file
/// passes `is_session_path`.
async fn spawn_web_server(home: &Path, proj: &Path) -> WebServer {
    let err_file = tempfile::NamedTempFile::new_in(home).expect("web stderr tempfile");
    let err_path = err_file.path().to_path_buf();
    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_omt"));
    cmd.args(["web", "--port", "0", "--cwd"])
        .arg(proj)
        .env("HOME", home)
        .env("OMP_HUB", "0") // no auto-started hub outliving the temp HOME
        .env("OMP_RS_BASE_URL", FIXTURE_BASE_URL)
        .env("OMP_RS_MODEL", FIXTURE_MODEL)
        .env("NO_COLOR", "1")
        .env_remove("OMP_AGENT_DIR")
        .env_remove("OMP_AUTH_BROKER_URL")
        .env_remove("OMP_AUTH_BROKER_TOKEN")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(err_file.reopen().expect("web stderr handle")))
        .kill_on_drop(true);
    let mut child = cmd.spawn().expect("spawn `omp-rs web`");
    let mut lines = BufReader::new(child.stdout.take().expect("web stdout")).lines();
    let mut boot_out = String::new();
    let port = tokio::time::timeout(STARTUP_TIMEOUT, async {
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    boot_out.push_str(&line);
                    boot_out.push('\n');
                    if let Some(port) = banner_port(&line) {
                        return Some(port);
                    }
                }
                _ => return None,
            }
        }
    })
    .await
    .ok()
    .flatten();
    let Some(port) = port else {
        let status = child.try_wait().ok().flatten();
        let _ = child.start_kill();
        let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
        panic!(
            "`omp-rs web` printed no banner within {STARTUP_TIMEOUT:?} (status {status:?})\n\
             stdout:\n{boot_out}stderr ({}):\n{stderr}",
            err_path.display(),
        );
    };
    WebServer {
        child,
        port,
        err_path,
        _err_file: err_file,
    }
}

/// Minimal loopback HTTP/1.1 client for the hub's JSON API (`Connection:
/// close`, body read to EOF — the workspace has no HTTP client crate wired
/// for plain requests, and this keeps the test dependency-free).
async fn http_json(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let body = body.unwrap_or("");
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap_or_else(|e| panic!("connect 127.0.0.1:{port}: {e}"));
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: http://127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    let raw = String::from_utf8_lossy(&raw);
    let status = raw
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    let body = raw
        .split_once("\r\n\r\n")
        .map_or_else(String::new, |(_, body)| body.to_string());
    (status, body)
}

/// One view switch at the protocol level: fresh WebSocket to the hub bridge
/// (same-origin header, the hub rejects foreign origins), then `get_state` +
/// `get_messages` + `get_available_commands` in parallel, resolved when the
/// last of the three responses lands.
#[allow(clippy::too_many_lines)] // one attach: three parallel RPC requests, then the frame checks
async fn attach_once(port: u16, agent_id: &str, seq: &mut usize) -> Result<Attach, String> {
    let request = Request::builder()
        .method("GET")
        .uri(format!("ws://127.0.0.1:{port}/ws/{agent_id}"))
        .header("Host", format!("127.0.0.1:{port}"))
        .header("Origin", format!("http://127.0.0.1:{port}"))
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", generate_key())
        .body(())
        .map_err(|e| format!("build request: {e}"))?;

    let t0 = Instant::now();
    let (ws, _) = connect_async(request)
        .await
        .map_err(|e| format!("ws connect: {e}"))?;
    let ws_open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (mut sink, mut stream) = ws.split();

    let rpcs = ["get_state", "get_messages", "get_available_commands"];
    let mut pending: Vec<(String, &'static str, Instant)> = Vec::new();
    for rpc in rpcs {
        *seq += 1;
        let id = format!("vs{seq}");
        sink.send(Message::text(json!({"id": id, "type": rpc}).to_string()))
            .await
            .map_err(|e| format!("send {rpc}: {e}"))?;
        pending.push((id, rpc, Instant::now()));
    }

    let mut marks: Vec<(&'static str, f64)> = Vec::new();
    let mut frames: Vec<String> = Vec::new();
    let mut messages = None;
    loop {
        let frame = match tokio::time::timeout(ATTACH_TIMEOUT, stream.next()).await {
            Ok(Some(Ok(frame))) => frame,
            Ok(Some(Err(e))) => return Err(format!("ws read: {e}; frames: {}", trail(&frames))),
            Ok(_) => return Err(format!("ws closed by the hub; frames: {}", trail(&frames))),
            Err(_) => {
                return Err(format!(
                    "no completion within {ATTACH_TIMEOUT:?}; frames: {}",
                    trail(&frames)
                ));
            }
        };
        let Message::Text(text) = frame else { continue };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let ftype = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if ftype == "web_agent_exit" {
            return Err(format!("rpc child exited: {value}"));
        }
        let fid = value.get("id").and_then(Value::as_str).unwrap_or_default();
        let hit = pending
            .iter()
            .find(|(id, _, _)| id.as_str() == fid)
            .map(|(_, rpc, sent)| (*rpc, *sent));
        let Some((rpc, sent)) = hit else {
            if frames.len() < FRAME_TRAIL {
                frames.push(ftype.to_string());
            }
            continue;
        };
        if value.get("success").and_then(Value::as_bool) == Some(false) {
            let error = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(format!("{rpc} failed: {error}"));
        }
        marks.push((rpc, sent.elapsed().as_secs_f64() * 1000.0));
        if rpc == "get_messages" {
            messages = value
                .pointer("/data/messages")
                .and_then(Value::as_array)
                .map(Vec::len);
        }
        pending.retain(|(id, _, _)| id.as_str() != fid);
        if pending.is_empty() {
            break;
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mark = |name: &str| {
        marks
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(0.0, |(_, ms)| *ms)
    };
    Ok(Attach {
        total_ms,
        rpc_ms: total_ms - ws_open_ms,
        ws_open_ms,
        get_state_ms: mark("get_state"),
        get_messages_ms: mark("get_messages"),
        get_available_commands_ms: mark("get_available_commands"),
        messages,
    })
}

/// Compact frame trail for error messages.
fn trail(frames: &[String]) -> String {
    if frames.is_empty() {
        "none".to_string()
    } else {
        frames.join(",")
    }
}

/// Nearest-rank percentile of a slice sorted ascending (`num`/`den`, e.g.
/// 95/100 for p95): rank = ceil(num * n / den).
fn nearest_rank(sorted: &[f64], num: usize, den: usize) -> Option<f64> {
    let rank = (num * sorted.len()).div_ceil(den);
    sorted.get(rank.max(1) - 1).copied()
}

/// Full per-run table plus the percentile summary, for the failure panic.
fn histogram(runs: &[Attach]) -> String {
    let mut lines = vec![
        "  run |  total ms |   rpc ms | ws open | get_state | get_messages | commands | messages"
            .to_string(),
    ];
    for (i, run) in runs.iter().enumerate() {
        lines.push(format!(
            "{:>5} | {:>9.1} | {:>8.1} | {:>7.1} | {:>9.1} | {:>12.1} | {:>8.1} | {}",
            i + 1,
            run.total_ms,
            run.rpc_ms,
            run.ws_open_ms,
            run.get_state_ms,
            run.get_messages_ms,
            run.get_available_commands_ms,
            run.messages
                .map_or_else(|| "n/a".to_string(), |m| m.to_string()),
        ));
    }
    let mut totals: Vec<f64> = runs.iter().map(|run| run.total_ms).collect();
    totals.sort_by(f64::total_cmp);
    if let (Some(p50), Some(p95), Some(max)) = (
        nearest_rank(&totals, 1, 2),
        nearest_rank(&totals, 95, 100),
        nearest_rank(&totals, 1, 1),
    ) {
        lines.push(format!(
            "totals: n={} p50={p50:.1}ms p95={p95:.1}ms max={max:.1}ms \
             (ceilings: p95<{VIEW_SWITCH_CEILING_P95_MS}ms, max<{VIEW_SWITCH_CEILING_MAX_MS}ms)",
            totals.len(),
        ));
    }
    lines.join("\n")
}

/// Last lines of the hub's stderr (its own boot errors; rpc-child crashes
/// surface as `web_agent_exit` frames instead).
fn stderr_tail(path: &Path, lines: usize) -> String {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let all: Vec<&str> = text.lines().collect();
            let start = all.len().saturating_sub(lines);
            all[start..].join("\n")
        }
        Err(e) => format!("<unreadable {}: {e}>", path.display()),
    }
}

#[tokio::test]
async fn web_view_switch_protocol_attach_of_400_messages_stays_under_ceiling() {
    let home = tempfile::tempdir().expect("temp HOME");
    let proj = home.path().join("proj");
    std::fs::create_dir_all(&proj).expect("proj dir");
    let session = seed_session(home.path(), &proj);

    let mut server = spawn_web_server(home.path(), &proj).await;
    let err_path = server.err_path.clone();
    let stderr = || stderr_tail(&err_path, STDERR_TAIL);

    // Resume the fixture through the same API the sidebar uses.
    let payload =
        json!({"cwd": proj.display().to_string(), "resume": session.display().to_string()})
            .to_string();
    let (status, body) = tokio::time::timeout(
        HTTP_TIMEOUT,
        http_json(server.port, "POST", "/api/agents", Some(payload.as_str())),
    )
    .await
    .expect("POST /api/agents timed out");
    assert_eq!(
        status, 200,
        "POST /api/agents failed: HTTP {status}: {body}"
    );
    let agent_id = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("id").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| panic!("no agent id in POST /api/agents response: {body}"));

    // Warmup: the first attach also absorbs the rpc child's startup frames
    // (ready / available_commands_update), like the bench script's warmup.
    let mut seq = 0;
    let warm = attach_once(server.port, &agent_id, &mut seq)
        .await
        .unwrap_or_else(|e| panic!("warmup attach failed: {e}\nweb stderr:\n{}", stderr()));
    assert_eq!(
        warm.messages,
        Some(SEEDED_MESSAGES),
        "warmup attach resumed {:?} of {SEEDED_MESSAGES} messages — fixture or resume path broken",
        warm.messages
    );

    let mut runs = Vec::with_capacity(ATTACH_CYCLES);
    for cycle in 0..ATTACH_CYCLES {
        let outcome = tokio::time::timeout(
            ATTACH_TIMEOUT,
            attach_once(server.port, &agent_id, &mut seq),
        )
        .await;
        match outcome {
            Ok(Ok(run)) => {
                assert_eq!(
                    run.messages,
                    Some(SEEDED_MESSAGES),
                    "attach {cycle} resumed {:?} of {SEEDED_MESSAGES} messages\n{}",
                    run.messages,
                    histogram(&runs),
                );
                runs.push(run);
            }
            Ok(Err(e)) => panic!(
                "attach {cycle} failed: {e}\n{}\nweb stderr:\n{}",
                histogram(&runs),
                stderr()
            ),
            Err(elapsed) => panic!(
                "attach {cycle} did not complete within {ATTACH_TIMEOUT:?} ({elapsed})\n{}\nweb stderr:\n{}",
                histogram(&runs),
                stderr(),
            ),
        }
        tokio::time::sleep(CYCLE_PAUSE).await;
    }

    let mut totals: Vec<f64> = runs.iter().map(|run| run.total_ms).collect();
    totals.sort_by(f64::total_cmp);
    let p95 = nearest_rank(&totals, 95, 100).expect("p95 of the runs");
    let max = nearest_rank(&totals, 1, 1).expect("max of the runs");
    assert!(
        p95 < VIEW_SWITCH_CEILING_P95_MS && max < VIEW_SWITCH_CEILING_MAX_MS,
        "view-switch attach exceeded the protocol ceiling\n{}\nweb stderr:\n{}",
        histogram(&runs),
        stderr(),
    );

    // Shutdown: kill the agent through the API, then the hub; the temp HOME
    // (seeded store, stderr file) goes with the TempDir.
    let (status, body) = tokio::time::timeout(
        HTTP_TIMEOUT,
        http_json(
            server.port,
            "DELETE",
            &format!("/api/agents/{agent_id}"),
            None,
        ),
    )
    .await
    .expect("DELETE /api/agents timed out");
    assert_eq!(
        status, 200,
        "DELETE /api/agents failed: HTTP {status}: {body}"
    );
    server.child.kill().await.expect("kill web server");
}
