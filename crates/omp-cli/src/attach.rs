//! `omt attach` — the terminal UI as a client of a live `omt web` hub agent.
//!
//! The hub runs every open session as a detached `omt --mode rpc` child
//! (`subcmd/web/detach.rs`): commands go in on its control socket, every
//! frame lands in its log. This module drives the regular `omp_tui` app over
//! that pair — commands through the socket, frames tailed from the log — so
//! the same live session is usable from the desktop app, the browser and a
//! terminal at once, with one process writing the session file.
//!
//! - `omt attach [AGENT|SESSION]` attaches explicitly;
//! - `omt --resume X` attaches when X is already open in the hub;
//! - `/desktop` in a local TUI hands the session to the hub, raises the
//!   desktop app and re-execs into `omt attach`.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context as _, Result};
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt as _, AsyncSeekExt as _, AsyncWriteExt as _};
use tokio::sync::{mpsc, oneshot};

use crate::subcmd::web::detach;
use crate::tui::{desktop_command, prompt_completed, prompt_failed};

const DEFAULT_HUB: &str = "http://127.0.0.1:7467";

const USAGE: &str = "usage: omt attach [AGENT_ID|SESSION_FILE|SESSION_ID]

Open a live session of the omt web/desktop hub in this terminal. Desktop,
browser and terminal share the running agent; leaving the terminal (Ctrl+D,
/exit) keeps it running. Without an argument the newest agent in the
current directory is used. Hub address: $OMT_HUB_URL (default
http://127.0.0.1:7467).";

/// Responses to these commands mean the agent now shows another transcript.
const SESSION_CHANGING: [&str; 5] = [
    "switch_session",
    "new_session",
    "open_session",
    "branch",
    "handoff",
];

fn hub_url() -> String {
    std::env::var("OMT_HUB_URL")
        .ok()
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| DEFAULT_HUB.to_owned())
}

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap_or_default()
}

/// `GET /api/agents` of the hub.
///
/// # Errors
/// The hub is not reachable or answered garbage.
pub async fn hub_agents() -> Result<Vec<Value>> {
    let url = hub_url();
    let v: Value = http()
        .get(format!("{url}/api/agents"))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .with_context(|| format!("omt web hub is not reachable at {url}"))?
        .json()
        .await
        .context("hub /api/agents")?;
    Ok(v["agents"].as_array().cloned().unwrap_or_default())
}

async fn hub_post(path: &str, body: &Value) -> Result<Value> {
    let url = hub_url();
    let res = http()
        .post(format!("{url}{path}"))
        .json(body)
        .send()
        .await
        .with_context(|| format!("omt web hub is not reachable at {url}"))?;
    let status = res.status();
    let v: Value = res.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        bail!(
            "hub {path}: {}",
            v["error"]
                .as_str()
                .map_or_else(|| status.to_string(), str::to_owned)
        );
    }
    Ok(v)
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Live hub agent that has `session` open.
pub async fn agent_for_session(session: &Path) -> Option<String> {
    hub_agents().await.ok()?.into_iter().find_map(|a| {
        let open = a["alive"].as_bool() == Some(true)
            && a["sessionFile"]
                .as_str()
                .is_some_and(|f| same_file(Path::new(f), session));
        open.then(|| a["id"].as_str().map(str::to_owned))?
    })
}

/// `/desktop` handoff, after the local TUI released `session`: open it in
/// the hub, show it in the desktop app, then replace this process with
/// `omt attach <id>` (only returns on failure).
///
/// # Errors
/// The hub refused the session or the exec failed.
pub async fn handoff(session: &Path, cwd: &Path) -> Result<()> {
    let agent = hub_post(
        "/api/agents",
        &json!({"cwd": cwd.display().to_string(), "resume": session.display().to_string()}),
    )
    .await?;
    let id = agent["id"]
        .as_str()
        .context("hub returned no agent id")?
        .to_owned();
    if let Err(e) = hub_post(&format!("/api/agents/{id}/focus"), &json!({})).await {
        eprintln!("omt: desktop app: {e:#}");
    }
    // The attached TUI replays the whole transcript: clear the local copy
    // (screen + scrollback) so it is not shown twice.
    print!("\x1b[H\x1b[2J\x1b[3J");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    exec_attach(&id)
}

fn exec_attach(id: &str) -> Result<()> {
    use std::os::unix::process::CommandExt as _;
    let exe = std::env::current_exe().context("current exe")?;
    let err = std::process::Command::new(exe).args(["attach", id]).exec();
    Err(anyhow!("exec omt attach: {err}"))
}

/// `omt attach` subcommand entry (runs on its own runtime thread, like `web`).
///
/// # Errors
/// Bad arguments, no matching agent, or a failed TUI.
#[allow(clippy::disallowed_methods)] // entry point: dedicated runtime on its own thread
pub fn run(args: &[String]) -> Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(0);
    }
    if args.len() > 1 {
        bail!("{USAGE}");
    }
    let target = args.first().cloned();
    std::thread::spawn(move || -> Result<i32> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("create tokio runtime")?;
        rt.block_on(async move {
            let id = resolve(target.as_deref()).await?;
            attach(&id).await?;
            Ok(0)
        })
    })
    .join()
    .map_err(|_| anyhow!("attach thread panicked"))?
}

/// Pick the hub agent for `target`: agent id prefix, session file, session
/// id prefix — or, without a target, the newest agent in the current dir.
async fn resolve(target: Option<&str>) -> Result<String> {
    let agents: Vec<Value> = hub_agents()
        .await?
        .into_iter()
        .filter(|a| a["alive"].as_bool() == Some(true))
        .collect();
    let id = |a: &Value| a["id"].as_str().unwrap_or_default().to_owned();
    let found = if let Some(t) = target {
        agents.iter().find(|a| {
            a["id"].as_str().is_some_and(|i| i.starts_with(t))
                || a["sessionFile"]
                    .as_str()
                    .is_some_and(|f| same_file(Path::new(f), Path::new(t)))
                || a["sessionId"].as_str().is_some_and(|s| s.starts_with(t))
        })
    } else {
        let cwd = std::env::current_dir().unwrap_or_default();
        agents
            .iter()
            .filter(|a| {
                a["cwd"]
                    .as_str()
                    .is_some_and(|c| same_file(Path::new(c), &cwd))
            })
            .max_by_key(|a| a["startedMs"].as_i64().unwrap_or(0))
    };
    if let Some(a) = found {
        return Ok(id(a));
    }
    let mut msg = match target {
        Some(t) => format!("no live hub agent matches {t:?}"),
        None => "no live hub agent in this directory".to_owned(),
    };
    for a in &agents {
        let _ = write!(
            msg,
            "\n  {}  {}  {}",
            id(a),
            a["cwd"].as_str().unwrap_or_default(),
            a["sessionName"].as_str().unwrap_or_default()
        );
    }
    bail!(msg)
}

type Reply = oneshot::Sender<std::result::Result<Value, String>>;

/// One attached client: socket writer, pending replies, frames to the TUI.
struct Conn {
    out: mpsc::UnboundedSender<String>,
    pending: Mutex<HashMap<String, Reply>>,
    seq: AtomicU64,
    ui: mpsc::UnboundedSender<Value>,
    levels: Mutex<Vec<String>>,
    agent_id: String,
    cwd: PathBuf,
}

impl Conn {
    fn next_id(&self) -> String {
        format!(
            "tui-{}-{}",
            std::process::id(),
            self.seq.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// Send `cmd` (an `id` is added when missing) and await its response.
    async fn request(&self, mut cmd: Value) -> std::result::Result<Value, String> {
        let id = if let Some(id) = cmd.get("id").and_then(Value::as_str) {
            id.to_owned()
        } else {
            let id = self.next_id();
            cmd["id"] = json!(id);
            id
        };
        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id.clone(), tx);
        if self.out.send(cmd.to_string()).is_err() {
            self.pending.lock().remove(&id);
            return Err("agent connection closed".into());
        }
        rx.await
            .unwrap_or_else(|_| Err("agent connection closed".into()))
    }

    /// Fire a command; a failure surfaces as a `prompt_result` error.
    fn command(self: &Arc<Self>, cmd: Value) {
        let me = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = me.request(cmd).await {
                let _ = me.ui.send(prompt_failed(e));
            }
        });
    }

    /// Re-render the agent's transcript (`session_loaded`).
    async fn reload_transcript(&self) {
        let messages = self.request(json!({"type": "get_messages"})).await;
        let state = self
            .request(json!({"type": "get_state"}))
            .await
            .unwrap_or_default();
        if let Ok(data) = messages {
            let _ = self.ui.send(json!({
                "type": "session_loaded",
                "messages": data["messages"],
                "name": state["sessionName"],
            }));
        }
    }

    async fn refresh_levels(&self) {
        if let Ok(data) = self
            .request(json!({"type": "get_available_thinking_levels"}))
            .await
        {
            let levels = data["levels"]
                .as_array()
                .map(|l| {
                    l.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            *self.levels.lock() = levels;
        }
    }

    /// Route one agent frame: replies to their waiters, everything else to
    /// the TUI (plus the follow-ups a local runtime would have produced).
    fn on_frame(self: &Arc<Self>, mut frame: Value) {
        let kind = frame
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        match kind.as_str() {
            "response" => {
                let id = frame.get("id").and_then(Value::as_str).unwrap_or("");
                let waiter = self.pending.lock().remove(id);
                let ok = frame["success"].as_bool() == Some(true);
                let command = frame["command"].as_str().unwrap_or("");
                if ok && SESSION_CHANGING.contains(&command) {
                    // Any client switched the agent's session: follow it.
                    let me = Arc::clone(self);
                    tokio::spawn(async move { me.reload_transcript().await });
                }
                if let Some(w) = waiter {
                    let _ = w.send(if ok {
                        Ok(frame.get("data").cloned().unwrap_or(Value::Null))
                    } else {
                        Err(frame["error"]
                            .as_str()
                            .unwrap_or("command failed")
                            .to_owned())
                    });
                }
            }
            "ready" => {}
            _ => {
                if kind == "available_commands_update" {
                    add_desktop(&mut frame["commands"]);
                }
                if matches!(kind.as_str(), "model_changed" | "model_change") {
                    let me = Arc::clone(self);
                    tokio::spawn(async move { me.refresh_levels().await });
                }
                let _ = self.ui.send(frame);
            }
        }
    }
}

fn add_desktop(commands: &mut Value) {
    if let Some(list) = commands.as_array_mut() {
        if !list.iter().any(|c| c["name"] == "desktop") {
            list.push(desktop_command());
        }
    }
}

/// Tail the agent log from its current end, one frame per line
/// (`rpc_chunk` sequences reassembled). Non-JSON lines are the agent's stderr.
fn start_tail(conn: Arc<Conn>, mut file: tokio::fs::File, socket: PathBuf) {
    tokio::spawn(async move {
        let mut chunks = crate::rpc::ChunkAssembler::default();
        let mut updates = crate::rpc::UpdateExpander::default();
        // The log holds `message_update` deltas; hand the TUI full frames.
        let frames = Arc::clone(&conn);
        let mut deliver = move |text: &str, frame: Value| match updates.expand(text, &frame) {
            crate::rpc::Expansion::Unchanged => frames.on_frame(frame),
            crate::rpc::Expansion::Expanded(full) => {
                if let Ok(frame) = serde_json::from_str::<Value>(&full) {
                    frames.on_frame(frame);
                }
            }
            crate::rpc::Expansion::Unsynced => {}
        };
        let mut carry: Vec<u8> = Vec::new();
        let mut buf = vec![0u8; 64 * 1024];
        let mut idle_ticks = 0u32;
        loop {
            match file.read(&mut buf).await {
                Ok(0) | Err(_) => {
                    tokio::time::sleep(Duration::from_millis(40)).await;
                    idle_ticks += 1;
                    // The hub removes the socket when the agent is stopped.
                    if idle_ticks.is_multiple_of(50) && !socket.exists() {
                        let _ = conn.ui.send(json!({"type": "notice", "level": "error",
                            "message": "the hub agent stopped; leaving attach mode"}));
                        tokio::time::sleep(Duration::from_millis(1500)).await;
                        let _ = conn.ui.send(json!({"type": "ui_exit"}));
                        return;
                    }
                }
                Ok(n) => {
                    idle_ticks = 0;
                    for &b in &buf[..n] {
                        if b != b'\n' {
                            carry.push(b);
                            continue;
                        }
                        let line = std::mem::take(&mut carry);
                        let Ok(line) = std::str::from_utf8(&line) else {
                            continue;
                        };
                        let Ok(frame) = serde_json::from_str::<Value>(line) else {
                            continue;
                        };
                        if frame["type"] == "rpc_chunk" {
                            if let Some(text) = chunks.push(&frame) {
                                if let Ok(frame) = serde_json::from_str::<Value>(&text) {
                                    deliver(&text, frame);
                                }
                            }
                        } else if frame.is_object() {
                            deliver(line, frame);
                        }
                    }
                }
            }
        }
    });
}

/// Attach the TUI to hub agent `id` until the user leaves.
///
/// # Errors
/// The agent's socket/log is unavailable or the TUI loop failed.
pub async fn attach(id: &str) -> Result<()> {
    let socket = detach::socket_path(id);
    let log = detach::log_path(id);
    let stream = tokio::net::UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connect {}", socket.display()))?;
    let mut file = tokio::fs::File::open(&log)
        .await
        .with_context(|| format!("open {}", log.display()))?;
    file.seek(std::io::SeekFrom::End(0)).await?;
    let info = hub_agents()
        .await?
        .into_iter()
        .find(|a| a["id"] == id)
        .ok_or_else(|| anyhow!("hub agent {id} is gone"))?;
    let cwd = info["cwd"].as_str().map_or_else(
        || std::env::current_dir().unwrap_or_default(),
        PathBuf::from,
    );

    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<String>();
    let (_, mut writer) = stream.into_split();
    tokio::spawn(async move {
        while let Some(mut line) = out_rx.recv().await {
            line.push('\n');
            if writer.write_all(line.as_bytes()).await.is_err() {
                break;
            }
        }
    });
    let (ui_tx, ui_rx) = mpsc::unbounded_channel::<Value>();
    let conn = Arc::new(Conn {
        out: out_tx,
        pending: Mutex::new(HashMap::new()),
        seq: AtomicU64::new(1),
        ui: ui_tx,
        levels: Mutex::new(Vec::new()),
        agent_id: id.to_owned(),
        cwd: cwd.clone(),
    });
    start_tail(Arc::clone(&conn), file, socket);

    let state = conn
        .request(json!({"type": "get_state"}))
        .await
        .map_err(|e| anyhow!("get_state: {e}"))?;
    conn.reload_transcript().await;
    conn.refresh_levels().await;
    if let Ok(data) = conn
        .request(json!({"type": "get_available_commands"}))
        .await
    {
        let mut commands = data["commands"].clone();
        add_desktop(&mut commands);
        let _ = conn
            .ui
            .send(json!({"type": "available_commands_update", "commands": commands}));
    }
    let _ = conn.request(json!({"type": "enable_context_usage"})).await;
    let _ = conn.ui.send(json!({"type": "notice", "level": "info",
        "message": format!("attached to hub agent {id} — the desktop app shows the same session; /desktop raises it")}));

    let model = &state["model"];
    let osc11_bg = omp_tui::chrome::probe_osc11_background(Duration::from_millis(200));
    let session_sources = crate::persist::sessions_dir(&cwd)
        .map(|d| vec![omp_tui::overlays::sessions::SessionSource::new(d)])
        .unwrap_or_default();
    omp_tui::run(
        omp_tui::RunConfig {
            frames: ui_rx,
            model: format!(
                "{}/{}",
                model["provider"].as_str().unwrap_or_default(),
                model["id"].as_str().unwrap_or_default()
            ),
            model_name: model["name"]
                .as_str()
                .filter(|n| !n.is_empty())
                .map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            cwd,
            models: crate::tui::picker_models(),
            session_sources,
            osc11_bg,
        },
        RemotePort(conn),
    )
    .await
}

/// [`omp_tui::RuntimePort`] over a hub agent's control socket.
struct RemotePort(Arc<Conn>);

impl RemotePort {
    fn prompt(&self, id: Option<String>, text: String, behavior: Option<&'static str>) {
        let conn = Arc::clone(&self.0);
        tokio::spawn(async move {
            let mut cmd = json!({"type": "prompt", "message": text});
            if let Some(b) = behavior {
                cmd["streamingBehavior"] = json!(b);
            }
            if let Some(id) = &id {
                // The agent tags its `prompt_result` with the command id.
                cmd["id"] = json!(id);
            }
            let result = conn.request(cmd).await;
            let mut frame = match result {
                // A slash command handled without a run answers with
                // `agentInvoked`; close the TUI ticket like the local port.
                Ok(data) if data.get("agentInvoked").is_some() => {
                    prompt_completed(data["agentInvoked"].as_bool() == Some(true))
                }
                Ok(_) => return, // the agent emits its own prompt_result
                Err(e) => prompt_failed(e),
            };
            if let Some(id) = id {
                frame["id"] = json!(id);
            }
            let _ = conn.ui.send(frame);
        });
    }

    fn panel(&self, kind: &'static str) {
        let conn = Arc::clone(&self.0);
        tokio::spawn(async move {
            match conn.request(json!({"type": kind})).await {
                Ok(frame) => {
                    let _ = conn.ui.send(frame);
                }
                Err(e) => {
                    let _ = conn.ui.send(prompt_failed(e));
                }
            }
        });
    }
}

impl omp_tui::RuntimePort for RemotePort {
    fn submit(&self, id: Option<String>, text: String, behavior: Option<omp_tui::SubmitBehavior>) {
        let behavior = match behavior {
            Some(omp_tui::SubmitBehavior::Steer) => Some("steer"),
            Some(omp_tui::SubmitBehavior::FollowUp) => Some("followUp"),
            None => None,
        };
        self.prompt(id, text, behavior);
    }

    fn steer(&self, text: String) {
        self.0.command(json!({"type": "steer", "message": text}));
    }

    fn follow_up(&self, text: String) {
        self.0
            .command(json!({"type": "follow_up", "message": text}));
    }

    fn remove_queued(&self, text: String, queue: omp_tui::action::QueueKind) {
        let queue = match queue {
            omp_tui::action::QueueKind::Steering => "steering",
            omp_tui::action::QueueKind::FollowUp => "followUp",
        };
        self.0
            .command(json!({"type": "remove_queued_message", "message": text, "queue": queue}));
    }

    fn promote_queued(&self, text: String) {
        self.0
            .command(json!({"type": "promote_queued_message", "message": text}));
    }

    fn abort(&self) {
        self.0.command(json!({"type": "abort"}));
    }

    fn set_model(&self, provider: String, id: String) {
        self.0
            .command(json!({"type": "set_model", "provider": provider, "modelId": id}));
    }

    fn set_thinking_level(&self, level: String) {
        self.0
            .command(json!({"type": "set_thinking_level", "level": level}));
    }

    fn available_thinking_levels(&self) -> Vec<String> {
        self.0.levels.lock().clone()
    }

    fn run_slash(&self, name: String, args: String) {
        if name == "desktop" {
            // Already a hub session: just show it in the desktop app.
            let conn = Arc::clone(&self.0);
            tokio::spawn(async move {
                let path = format!("/api/agents/{}/focus", conn.agent_id);
                let frame = match hub_post(&path, &json!({})).await {
                    Ok(_) => prompt_completed(false),
                    Err(e) => prompt_failed(format!("/desktop: {e:#}")),
                };
                let _ = conn.ui.send(frame);
            });
            return;
        }
        let text = if args.is_empty() {
            format!("/{name}")
        } else {
            format!("/{name} {args}")
        };
        self.prompt(None, text, None);
    }

    fn extension_ui_response(&self, id: String, value: Value) {
        let mut envelope = json!({"type": "extension_ui_response", "id": id});
        if let (Value::Object(env), Value::Object(members)) = (&mut envelope, value) {
            env.extend(members);
        }
        let _ = self.0.out.send(envelope.to_string());
        // The agent does not echo answers; close the dialog locally.
        let _ = self.0.ui.send(envelope);
    }

    fn resume_session(&self, path: PathBuf, fork: bool) {
        let cmd = if fork {
            json!({"type": "fork"})
        } else {
            json!({"type": "switch_session", "sessionPath": path.display().to_string()})
        };
        self.0.command(cmd);
    }

    fn continue_run(&self) {
        self.prompt(None, omp_tui::MANUAL_CONTINUE_PROMPT.to_owned(), None);
    }

    fn abort_retry(&self) {
        self.0.command(json!({"type": "abort_retry"}));
    }

    fn rewind_entry(&self, entry_id: String) {
        self.0
            .command(json!({"type": "navigate_tree", "entryId": entry_id}));
    }

    fn abort_compaction(&self) {}

    fn refresh_models(&self) {
        let ui = self.0.ui.clone();
        crate::tui::refresh_models_into(move |frame| {
            let _ = ui.send(frame);
        });
    }

    fn settings_get(&self, ids: &[String]) -> Vec<(String, Value)> {
        let settings = omp_settings::Settings::load(&self.0.cwd);
        ids.iter()
            .map(|id| (id.clone(), settings.get(id)))
            .collect()
    }

    fn settings_set(&self, id: String, value: Value) -> Result<(), String> {
        omp_settings::write_global(&id, &value)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn request_session_info(&self) {
        self.panel("get_session_info_panel");
    }

    fn request_usage(&self) {
        self.panel("get_usage_panel");
    }
}
