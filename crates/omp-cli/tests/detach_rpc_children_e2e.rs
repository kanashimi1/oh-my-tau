#![allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
//! End-to-end guard for detached rpc children (`web/detach.rs` scenario):
//! an `--mode rpc` child must survive its hub terminating, a restarting hub
//! must adopt it back into the roster, and `DELETE /api/agents/<id>` must
//! still SIGTERM the group and clear the registry.
//!
//! Linux-only: liveness is probed via `/proc/<pid>` and shutdown uses
//! `kill -TERM` process groups, matching the implementation.
//!
//! Flow (black box, real hub binary, no provider traffic — the fixture model
//! points at loopback:9 and no turn is streamed):
//!   1. hub A on an ephemeral port, temp `HOME`;
//!   2. `POST /api/agents` spawns a detached child;
//!   3. `registry.json` records it and `/proc/<pid>` shows it alive;
//!   4. SIGTERM hub A, wait for exit — the child stays alive;
//!   5. hub B boots and adopts the child (`GET /api/agents` lists it);
//!   6. `DELETE /api/agents/<id>` — the registry entry is removed and the
//!      process group is gone (dead or reaped zombie).
#![allow(clippy::unwrap_used, clippy::expect_used)] // test code: a failed step should fail the test loudly

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader};

/// Hub startup banner budget.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
/// Wait budget for SIGTERM'd processes to disappear and for adoption side
/// effects (registry writes) to land.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
/// Polling cadence while waiting on process/registry state transitions.
const POLL_CADENCE: Duration = Duration::from_millis(100);
/// Stand-in provider endpoint for the fixture model: loopback, port 9
/// (discard, never bound). Model resolution only reads the variable; the
/// endpoint is never dialed because no turn is streamed.
const FIXTURE_BASE_URL: &str = "http://127.0.0.1:9/";
/// Any `provider/id` resolves against [`FIXTURE_BASE_URL`]; the value is inert.
const FIXTURE_MODEL: &str = "custom/detach-fixture";

/// The running `omt web` child: `SIGTERMmed` (not killed) by the test body —
/// the shutdown path is part of what is being verified.
struct WebServer {
    child: tokio::process::Child,
    port: u16,
    #[allow(dead_code)]
    err_file: tempfile::NamedTempFile,
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
/// from the banner). `HOME` is the temp dir, so the registry, sockets and
/// logs all land inside it.
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
        .stderr(Stdio::from(err_file.reopen().expect("web stderr handle")));
    let mut child = cmd.spawn().expect("spawn `omt web`");
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
            "`omt web` printed no banner within {STARTUP_TIMEOUT:?} (status {status:?})\n\
             stdout:\n{boot_out}stderr ({}):\n{stderr}",
            err_path.display(),
        );
    };
    WebServer {
        child,
        port,
        err_file,
    }
}

/// Minimal loopback HTTP/1.1 client for the hub's JSON API (`Connection:
/// close`, body read to EOF — same dependency-free approach as the
/// view-switch guard).
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

/// `kill -TERM <pid>` through the external binary (no libc dependency in
/// the test; the implementation itself shells out the same way for groups).
fn sigterm(pid: u32) {
    let status = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .expect("run kill -TERM");
    assert!(status.success(), "kill -TERM {pid} failed: {status}");
}

/// Process state from `/proc/<pid>/stat` (`None` = no such process). A
/// reaped-but-listed zombie reads as `Some('Z')`.
fn proc_state(pid: u32) -> Option<char> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The comm field can contain spaces and parens; the state follows the
    // last ')'.
    let after_comm = stat.rsplit_once(')')?.1.trim_start();
    after_comm.chars().next()
}

/// Alive in the sense this test needs: present and not a zombie.
fn alive(pid: u32) -> bool {
    matches!(proc_state(pid), Some(state) if state != 'Z')
}

/// SIGTERM the hub and wait for it to exit within [`SHUTDOWN_TIMEOUT`].
async fn terminate_hub(server: &mut WebServer, label: &str) {
    let pid = server.child.id().expect("hub pid");
    sigterm(pid);
    tokio::time::timeout(SHUTDOWN_TIMEOUT, server.child.wait())
        .await
        .unwrap_or_else(|_| panic!("hub {label} ignored SIGTERM for {SHUTDOWN_TIMEOUT:?}"))
        .expect("wait hub {label}");
}

/// Read the temp registry and return the entry for `id`.
fn registry_entry(home: &Path, id: &str) -> Option<Value> {
    let text = std::fs::read_to_string(home.join(".omp-rs/agents/registry.json")).ok()?;
    let entries = serde_json::from_str::<Value>(&text)
        .ok()?
        .as_array()?
        .clone();
    entries.into_iter().find(|e| e["id"] == json!(id))
}

#[tokio::test]
async fn detached_rpc_child_survives_hub_restart_and_is_adopted() {
    let home = TempDir::new().expect("temp HOME");
    let proj = home.path().join("proj");
    std::fs::create_dir_all(&proj).expect("proj dir");
    let home_path = home.path().to_path_buf();

    // 1-2. Hub A + spawn a detached child.
    let mut hub_a = spawn_web_server(&home_path, &proj).await;
    let (status, body) = http_json(
        hub_a.port,
        "POST",
        "/api/agents",
        Some(&json!({"cwd": proj}).to_string()),
    )
    .await;
    assert_eq!(status, 200, "POST /api/agents body:\n{body}");
    let id = serde_json::from_str::<Value>(&body).expect("agent info json")["id"]
        .as_str()
        .expect("agent id")
        .to_string();

    // 3. The registry records the child, and it is alive.
    let entry = tokio::time::timeout(SHUTDOWN_TIMEOUT, async {
        loop {
            if let Some(entry) = registry_entry(&home_path, &id) {
                return entry;
            }
            tokio::time::sleep(POLL_CADENCE).await;
        }
    })
    .await
    .expect("registry entry appears");
    let child_pid = u32::try_from(entry["pid"].as_u64().expect("child pid")).expect("pid fits u32");
    assert!(
        alive(child_pid),
        "rpc child {child_pid} not alive right after spawn"
    );

    // 4. SIGTERM hub A: the child survives.
    terminate_hub(&mut hub_a, "A").await;
    assert!(
        alive(child_pid),
        "rpc child {child_pid} died with hub A (registry entry: {entry})"
    );

    // 5. Hub B adopts the child back into the roster.
    let mut hub_b = spawn_web_server(&home_path, &proj).await;
    let adopted = tokio::time::timeout(SHUTDOWN_TIMEOUT, async {
        loop {
            let (status, body) = http_json(hub_b.port, "GET", "/api/agents", None).await;
            if status == 200 && body.contains(&id) {
                return body;
            }
            tokio::time::sleep(POLL_CADENCE).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("hub B never listed adopted agent {id}"));
    let listed = serde_json::from_str::<Value>(&adopted).expect("roster json");
    let mine = listed["agents"]
        .as_array()
        .expect("agents array")
        .iter()
        .find(|a| a["id"] == json!(id))
        .unwrap_or_else(|| panic!("adopted agent {id} missing from roster: {adopted}"))
        .clone();
    assert_eq!(mine["adopted"], json!(true), "adopted flag on {id}: {mine}");

    // 6. DELETE: group SIGTERMmed, registry cleared.
    let (status, body) = http_json(hub_b.port, "DELETE", &format!("/api/agents/{id}"), None).await;
    assert_eq!(status, 200, "DELETE /api/agents/{id} body:\n{body}");
    tokio::time::timeout(SHUTDOWN_TIMEOUT, async {
        loop {
            let registry_cleared = registry_entry(&home_path, &id).is_none();
            let child_gone = !alive(child_pid);
            if registry_cleared && child_gone {
                return;
            }
            tokio::time::sleep(POLL_CADENCE).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "DELETE left residue: registry entry {entry:?}, child alive {}",
            alive(child_pid)
        )
    });

    // Leave a clean field: hub B down before the temp HOME disappears.
    terminate_hub(&mut hub_b, "B").await;
}
