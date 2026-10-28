//! One browser-driven agent: an `omp-rs --mode rpc` child process, started
//! detached (own process group, stdout/stderr logged to
//! `~/.omp-rs/agents/<id>/log`, control over the per-agent socket
//! `~/.omp-rs/agents/<id>.sock`) so it survives hub restarts.
//!
//! The log is tailed for frames (fanned out to attached browser sockets);
//! browser frames are written to the child's control socket verbatim. The
//! bridge keeps just enough state to make reattaching cheap: unanswered
//! dialogs (replayed on attach) and a session summary refreshed through the
//! child's own `get_state`.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context as _, Result};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::broadcast;

use super::detach;

/// Response id of the bridge's own `get_state` probes (never forwarded).
const STATE_PROBE_ID: &str = "__web_state";
/// Response id of the startup `switch_session` that resumes a stored session
/// (`--resume` is ignored in rpc mode: the rpc runtime owns the session
/// lifecycle, so the bridge resumes through the protocol).
const RESUME_ID: &str = "__web_resume";
/// Response id of the bridge's `negotiate_protocol` (never forwarded). Protocol
/// v2 splits frames over the 1 MiB line limit into `rpc_chunk` sequences, so
/// large replies (`get_messages` of an image-heavy session) arrive whole.
const PROTOCOL_ID: &str = "__web_protocol";
/// Dialog methods a browser must answer; replayed to late attachers.
const DIALOG_METHODS: &[&str] = &["select", "confirm", "input", "editor", "ask"];
const STDERR_TAIL: usize = 40;

fn lock<T>(m: &Mutex<T>) -> parking_lot::MutexGuard<'_, T> {
    m.lock()
}

/// Push (`frame`) or resolve (`None`) a pending dialog and keep
/// `Summary::needs_input` in sync with the queue, so the roster status is
/// live between `get_state` probes. Free function over the two locks so the
/// bookkeeping stays unit-testable without spawning a child.
fn set_dialog_pending(
    pending: &Mutex<Vec<(String, Arc<str>)>>,
    summary: &Mutex<Summary>,
    id: &str,
    frame: Option<Arc<str>>,
) {
    let mut queue = lock(pending);
    match frame {
        Some(frame) => queue.push((id.to_string(), frame)),
        None => queue.retain(|(p, _)| p != id),
    }
    lock(summary).needs_input = !queue.is_empty();
}

/// What the sidebar shows for a running agent.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub session_file: Option<String>,
    pub session_id: Option<String>,
    pub session_name: Option<String>,
    pub model: Option<String>,
    pub streaming: bool,
    /// A dialog is waiting for an answer (`pending_ui` non-empty); recomputed
    /// where the queue mutates so the roster status stays live.
    pub needs_input: bool,
    /// A tool call is in flight. The child's `get_state` payload carries no
    /// tool-level flag (`Runtime::state` only exposes `isStreaming`), so this
    /// honestly mirrors `streaming` until the state grows such a signal.
    pub running_tools: bool,
    /// The last run ended parked on background `task` children
    /// (`agent_end` with `awaitingAsyncWork`) and nothing ran since: when.
    /// Lets the UI flag a session silently waiting on a stuck subagent.
    pub awaiting_tasks_since_ms: Option<i64>,
    /// While an adopted agent's log is replayed: the log's mtime, used as the
    /// clock for replayed frames (the replay itself happens "now").
    replay_clock_ms: Option<i64>,
}

/// How to start an agent.
#[derive(Debug, Clone, Default)]
pub struct Spawn {
    pub cwd: PathBuf,
    /// Session file (or id prefix) to resume.
    pub resume: Option<String>,
    /// `provider/id[:thinking]`.
    pub model: Option<String>,
    /// `always-ask` | `write` | `yolo`, forwarded as `--approval-mode`.
    pub approval_mode: Option<String>,
}

pub struct Agent {
    pub id: String,
    pub cwd: PathBuf,
    pub started_ms: i64,
    /// Child pid (spawned or adopted).
    pub pid: u32,
    /// Process group id: the child leads its own group, so a whole subtree can
    /// be SIGTERM'd without killing the hub.
    pub pgid: u32,
    /// Rebuilt from the registry after a hub restart (not spawned by us).
    pub adopted: bool,
    /// `--approval-mode` the child runs with (kept from spawn/adoption for `info_json`).
    approval_mode: Option<String>,
    /// Client connection to the child's control socket (replaces the stdin pipe).
    control: tokio::sync::Mutex<Option<UnixStream>>,
    /// `Some` only for children we spawned ourselves; adopted children have no
    /// handle (their process is not our child).
    child: tokio::sync::Mutex<Option<Child>>,
    tx: broadcast::Sender<Arc<str>>,
    /// Unanswered dialog requests in arrival order (id, raw frame).
    pending_ui: Mutex<Vec<(String, Arc<str>)>>,
    summary: Mutex<Summary>,
    stderr: Mutex<VecDeque<String>>,
    /// Bridge notices (e.g. a failed startup resume) replayed to new sockets.
    notices: Mutex<Vec<Arc<str>>>,
    /// Partially received protocol-v2 `rpc_chunk` sequences by chunk id.
    chunks: Mutex<crate::rpc::ChunkAssembler>,
    alive: AtomicBool,
}

impl Agent {
    /// Start a detached child (own process group; stdout/stderr logged; control
    /// over a per-agent Unix socket) and drive it: tail the log for frames and
    /// watch the child for exit. `on_exit` runs once the child is gone (after
    /// the exit frame was broadcast).
    pub async fn spawn(
        id: String,
        exe: &Path,
        spec: &Spawn,
        on_exit: impl FnOnce(&str) + Send + 'static,
    ) -> Result<Arc<Self>> {
        let dir = detach::agents_dir().join(&id);
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let log_path = dir.join("log");
        let sock_path = detach::socket_path(&id);
        let _ = std::fs::remove_file(&sock_path); // stale socket from an earlier incarnation

        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("open {}", log_path.display()))?;

        let mut cmd = Command::new(exe);
        cmd.arg("--mode").arg("rpc").arg("--cwd").arg(&spec.cwd);
        if let Some(model) = spec.model.as_deref().filter(|m| !m.is_empty()) {
            cmd.arg("--model").arg(model);
        }
        if let Some(mode) = spec.approval_mode.as_deref().filter(|m| !m.is_empty()) {
            cmd.arg("--approval-mode").arg(mode);
        }
        #[cfg(unix)]
        {
            // The child leads its own process group, so it survives hub restarts
            // and can be SIGTERM'd as a whole on DELETE.
            cmd.process_group(0);
        }
        cmd.current_dir(&spec.cwd)
            .env("OMP_RPC_SOCKET", &sock_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        let child = cmd
            .spawn()
            .with_context(|| format!("spawn {} --mode rpc", exe.display()))?;
        let pid = child.id().unwrap_or(0);
        // process_group(0): the group id is the child's own pid
        let group_id = pid;

        let control = match Self::connect_control(&sock_path).await {
            Ok(stream) => stream,
            Err(e) => {
                detach::kill_group(pid, group_id);
                let _ = std::fs::remove_file(&sock_path); // the child may have bound it before dying
                return Err(e);
            }
        };

        let (tx, _) = broadcast::channel(8192);
        let agent = Arc::new(Self {
            id: id.clone(),
            cwd: spec.cwd.clone(),
            started_ms: pa_agent::now_ms(),
            pid,
            pgid: group_id,
            adopted: false,
            approval_mode: spec.approval_mode.clone(),
            control: tokio::sync::Mutex::new(Some(control)),
            child: tokio::sync::Mutex::new(Some(child)),
            tx,
            pending_ui: Mutex::new(Vec::new()),
            summary: Mutex::new(Summary {
                session_file: spec.resume.clone(),
                ..Summary::default()
            }),
            stderr: Mutex::new(VecDeque::new()),
            notices: Mutex::new(Vec::new()),
            chunks: Mutex::new(crate::rpc::ChunkAssembler::default()),
            alive: AtomicBool::new(true),
        });

        Self::start_tail(Arc::clone(&agent), log_path, false);
        Self::start_exit_watcher(Arc::clone(&agent), on_exit);

        agent.negotiate_protocol().await?;
        if let Some(resume) = &spec.resume {
            let frame = json!({"id": RESUME_ID, "type": "switch_session", "sessionPath": resume});
            agent.send_line(&frame.to_string()).await?;
        }
        agent.probe_state().await;
        Ok(agent)
    }

    /// Adopt a live detached child from the registry: replay its log history to
    /// rebuild summary/dialog state, connect its control socket, and watch it
    /// for exit. No `Child` handle is held — the process is not our child.
    pub async fn adopt(
        entry: &detach::Entry,
        on_exit: impl FnOnce(&str) + Send + 'static,
    ) -> Result<Arc<Self>> {
        let cwd = detach::pid_cwd(entry.pid)
            .ok_or_else(|| anyhow!("agent {} cwd unreadable", entry.id))?;
        let control = Self::connect_control(&entry.socket)
            .await
            .with_context(|| format!("connect {}", entry.socket.display()))?;
        let agent = Arc::new(Self {
            id: entry.id.clone(),
            cwd,
            started_ms: entry.started_ms,
            pid: entry.pid,
            pgid: entry.pgid,
            adopted: true,
            approval_mode: entry.approval_mode.clone(),
            control: tokio::sync::Mutex::new(Some(control)),
            child: tokio::sync::Mutex::new(None),
            tx: broadcast::channel(8192).0,
            pending_ui: Mutex::new(Vec::new()),
            summary: Mutex::new(Summary {
                session_file: entry.session.clone(),
                ..Summary::default()
            }),
            stderr: Mutex::new(VecDeque::new()),
            notices: Mutex::new(Vec::new()),
            chunks: Mutex::new(crate::rpc::ChunkAssembler::default()),
            alive: AtomicBool::new(true),
        });
        // Replay from the start to rebuild state, then tail live.
        Self::start_tail(Arc::clone(&agent), detach::log_path(&entry.id), true);
        Self::start_adopted_watcher(Arc::clone(&agent), entry.pid, on_exit);
        agent.negotiate_protocol().await?;
        agent.probe_state().await;
        Ok(agent)
    }

    /// Poll the child's control socket until it accepts us (the child binds
    /// asynchronously during its own startup).
    async fn connect_control(sock: &Path) -> Result<UnixStream> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let last = loop {
            match UnixStream::connect(sock).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if Instant::now() >= deadline {
                        break e;
                    }
                    tokio::time::sleep(Duration::from_millis(30)).await;
                }
            }
        };
        Err(anyhow!(last.to_string()))
    }

    /// Tail the agent log. JSON object lines go to `on_child_line`; other lines
    /// (the child's stderr, mixed into the same file) feed the stderr tail used
    /// by the exit frame. When `from_start` the whole log is replayed first
    /// (adoption), otherwise tailing starts at the current end.
    fn start_tail(agent: Arc<Self>, log_path: PathBuf, from_start: bool) {
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncSeekExt};
            let Ok(mut file) = tokio::fs::File::open(&log_path).await else {
                agent.alive.store(false, Ordering::SeqCst);
                return;
            };
            if from_start {
                let mtime = file
                    .metadata()
                    .await
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .and_then(|d| i64::try_from(d.as_millis()).ok());
                lock(&agent.summary).replay_clock_ms = mtime;
            } else {
                let _ = file.seek(std::io::SeekFrom::End(0)).await;
            }
            let mut carry: Vec<u8> = Vec::new();
            let mut buf = [0u8; 8192];
            loop {
                match file.read(&mut buf).await {
                    Ok(0) => {
                        lock(&agent.summary).replay_clock_ms = None; // caught up: live from here
                        if !agent.alive.load(Ordering::SeqCst) {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    Ok(n) => {
                        for &b in &buf[..n] {
                            if b == b'\n' {
                                let line = String::from_utf8_lossy(&carry).into_owned();
                                carry.clear();
                                if !line.trim().is_empty() {
                                    agent.feed_log_line(&line).await;
                                }
                            } else {
                                carry.push(b);
                            }
                        }
                    }
                    Err(_) => {
                        if !agent.alive.load(Ordering::SeqCst) {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        });
    }

    /// One complete log line: a JSON object is a protocol frame; anything else
    /// is stderr diagnostics kept for the exit frame.
    async fn feed_log_line(&self, line: &str) {
        match serde_json::from_str::<Value>(line) {
            Ok(v) if v.get("type").and_then(Value::as_str) == Some("rpc_chunk") => {
                let frame = lock(&self.chunks).push(&v);
                if let Some(frame) = frame {
                    self.on_child_line(&frame).await;
                }
            }
            Ok(v) if v.is_object() => self.on_child_line(line).await,
            _ => {
                let mut tail = lock(&self.stderr);
                if tail.len() == STDERR_TAIL {
                    tail.pop_front();
                }
                tail.push_back(line.to_string());
            }
        }
    }

    /// Watch a *spawned* child (we hold the handle and can `wait` on it).
    fn start_exit_watcher(agent: Arc<Self>, on_exit: impl FnOnce(&str) + Send + 'static) {
        tokio::spawn(async move {
            let code = match agent.child.lock().await.as_mut() {
                Some(child) => child.wait().await.ok().and_then(|s| s.code()),
                None => None,
            };
            tokio::time::sleep(Duration::from_millis(300)).await;
            agent.finish(code, on_exit).await;
        });
    }

    /// Watch an *adopted* child (no handle; poll liveness via `/proc`).
    fn start_adopted_watcher(
        agent: Arc<Self>,
        pid: u32,
        on_exit: impl FnOnce(&str) + Send + 'static,
    ) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                if !detach::pid_alive(pid) {
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    agent.finish(None, on_exit).await;
                    break;
                }
            }
        });
    }

    /// Mark the agent dead, broadcast the exit frame, then notify the hub.
    async fn finish(&self, code: Option<i32>, on_exit: impl FnOnce(&str)) {
        self.alive.store(false, Ordering::SeqCst);
        // Give the tail task a moment to drain the final log lines.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let stderr: Vec<String> = lock(&self.stderr).iter().cloned().collect();
        let frame = json!({"type": "web_agent_exit", "code": code, "stderr": stderr});
        let _ = self.tx.send(Arc::from(frame.to_string()));
        on_exit(&self.id);
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    pub fn summary(&self) -> Summary {
        lock(&self.summary).clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<str>> {
        self.tx.subscribe()
    }

    /// Frames a new socket must see first: bridge notices, then dialogs
    /// still waiting for an answer.
    pub fn replay_frames(&self) -> Vec<Arc<str>> {
        let mut out = lock(&self.notices).clone();
        out.extend(lock(&self.pending_ui).iter().map(|(_, f)| Arc::clone(f)));
        out
    }

    pub fn info_json(&self) -> Value {
        let s = self.summary();
        json!({
            "id": self.id,
            "cwd": self.cwd.display().to_string(),
            "startedMs": self.started_ms,
            "adopted": self.adopted,
            "alive": self.is_alive(),
            "approvalMode": self.approval_mode,
            "sessionFile": s.session_file,
            "sessionId": s.session_id,
            "sessionName": s.session_name,
            "model": s.model,
            "streaming": s.streaming,
            "needsInput": s.needs_input,
            "runningTools": s.running_tools,
            "awaitingTasksSinceMs": s.awaiting_tasks_since_ms,
        })
    }

    /// Write one browser frame to the child.
    pub async fn send_line(&self, line: &str) -> Result<()> {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v.get("type").and_then(Value::as_str) == Some("extension_ui_response") {
                if let Some(id) = v.get("id").and_then(Value::as_str) {
                    set_dialog_pending(&self.pending_ui, &self.summary, id, None);
                    // Other tabs showing the same dialog close it.
                    let close = json!({"type": "extension_ui_request", "method": "cancelDialog",
                                       "params": {"id": id}, "web": "answered"});
                    let _ = self.tx.send(Arc::from(close.to_string()));
                }
            }
        }
        let mut control = self.control.lock().await;
        if let Some(stream) = control.as_mut() {
            stream.write_all(line.as_bytes()).await?;
            stream.write_all(b"\n").await?;
            stream.flush().await?;
        }
        Ok(())
    }

    pub async fn kill(&self) {
        if !detach::kill_group(self.pid, self.pgid) {
            if let Some(child) = self.child.lock().await.as_mut() {
                let _ = child.start_kill();
            }
        }
    }

    async fn negotiate_protocol(&self) -> Result<()> {
        let frame = json!({"id": PROTOCOL_ID, "type": "negotiate_protocol", "protocolVersion": 2});
        self.send_line(&frame.to_string()).await
    }

    async fn probe_state(&self) {
        let probe = json!({"id": STATE_PROBE_ID, "type": "get_state"}).to_string();
        let _ = self.send_line(&probe).await;
    }

    async fn on_child_line(&self, text: &str) {
        let Ok(frame) = serde_json::from_str::<Value>(text) else {
            let _ = self.tx.send(Arc::from(text));
            return;
        };
        let kind = frame.get("type").and_then(Value::as_str).unwrap_or("");
        let mut refresh = false;
        match kind {
            "response" if frame.get("id").and_then(Value::as_str) == Some(STATE_PROBE_ID) => {
                if let Some(data) = frame.get("data") {
                    self.apply_state(data);
                }
                return; // the bridge's own probe
            }
            "response" if frame.get("id").and_then(Value::as_str) == Some(PROTOCOL_ID) => {
                return; // the bridge's own protocol negotiation
            }
            "response" if frame.get("id").and_then(Value::as_str) == Some(RESUME_ID) => {
                if frame.get("success").and_then(Value::as_bool) != Some(true) {
                    let error = frame
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error");
                    let note: Arc<str> = Arc::from(
                        json!({"type": "web_error", "message": format!("Could not resume the session: {error}")})
                            .to_string(),
                    );
                    lock(&self.notices).push(Arc::clone(&note));
                    let _ = self.tx.send(note);
                }
                self.probe_state().await;
                return;
            }
            "response" => {
                let command = frame.get("command").and_then(Value::as_str).unwrap_or("");
                refresh = matches!(
                    command,
                    "new_session"
                        | "switch_session"
                        | "open_session"
                        | "fork"
                        | "branch"
                        | "set_session_name"
                        | "set_model"
                        | "cycle_model"
                );
            }
            "extension_ui_request" => {
                let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
                if DIALOG_METHODS.contains(&method) {
                    if let Some(id) = frame.get("id").and_then(Value::as_str) {
                        set_dialog_pending(
                            &self.pending_ui,
                            &self.summary,
                            id,
                            Some(Arc::from(text)),
                        );
                    }
                } else if method == "cancelDialog" {
                    if let Some(id) = frame.pointer("/params/id").and_then(Value::as_str) {
                        set_dialog_pending(&self.pending_ui, &self.summary, id, None);
                    }
                }
            }
            "agent_start" => note_run_frame(&mut lock(&self.summary), &frame, pa_agent::now_ms()),
            "agent_end" | "session_settled" => {
                note_run_frame(&mut lock(&self.summary), &frame, pa_agent::now_ms());
                refresh = true;
            }
            "session_info" | "session_loaded" | "session_reset" | "model_changed" => refresh = true,
            _ => {}
        }
        let _ = self.tx.send(Arc::from(text));
        if refresh {
            self.probe_state().await;
        }
    }

    fn apply_state(&self, data: &Value) {
        let mut s = lock(&self.summary);
        let text = |k: &str| data.get(k).and_then(Value::as_str).map(str::to_owned);
        if let Some(f) = text("sessionFile") {
            s.session_file = Some(f);
        }
        s.session_id = text("sessionId").or(s.session_id.take());
        s.session_name = text("sessionName");
        if let Some(m) = data.get("model") {
            let provider = m.get("provider").and_then(Value::as_str).unwrap_or("");
            let id = m.get("id").and_then(Value::as_str).unwrap_or("");
            s.model = Some(format!("{provider}/{id}"));
        }
        if let Some(b) = data.get("isStreaming").and_then(Value::as_bool) {
            s.streaming = b;
            s.running_tools = b;
        }
    }
}

/// Run-state bookkeeping for `agent_start` / `agent_end` / `session_settled`:
/// streaming flag plus "parked on background subagents since" — an
/// `agent_end` with `awaitingAsyncWork` starts the clock (replayed frames use
/// the log's mtime), any new run or a normal settle clears it.
fn note_run_frame(s: &mut Summary, frame: &Value, now_ms: i64) {
    match frame.get("type").and_then(Value::as_str) {
        Some("agent_start") => {
            s.streaming = true;
            s.awaiting_tasks_since_ms = None;
        }
        Some("agent_end") => {
            s.streaming = false;
            if frame.get("awaitingAsyncWork").and_then(Value::as_bool) == Some(true) {
                let at = s.replay_clock_ms.unwrap_or(now_ms);
                s.awaiting_tasks_since_ms.get_or_insert(at);
            } else {
                s.awaiting_tasks_since_ms = None;
            }
        }
        Some("session_settled") => {
            s.streaming = false;
            s.awaiting_tasks_since_ms = None;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `needs_input` follows the pending-dialog queue: any unanswered dialog
    /// sets it, resolving the last one clears it.
    #[test]
    fn parked_on_subagents_is_tracked_until_the_next_run() {
        let mut s = Summary::default();
        note_run_frame(&mut s, &json!({"type": "agent_start"}), 1);
        assert!(s.streaming);
        note_run_frame(
            &mut s,
            &json!({"type": "agent_end", "awaitingAsyncWork": true}),
            100,
        );
        assert_eq!(s.awaiting_tasks_since_ms, Some(100));
        assert!(!s.streaming);
        // A second parked end keeps the original start of the wait.
        note_run_frame(
            &mut s,
            &json!({"type": "agent_end", "awaitingAsyncWork": true}),
            500,
        );
        assert_eq!(s.awaiting_tasks_since_ms, Some(100));
        // The delivery turn starts: no longer waiting.
        note_run_frame(&mut s, &json!({"type": "agent_start"}), 600);
        assert_eq!(s.awaiting_tasks_since_ms, None);
        note_run_frame(
            &mut s,
            &json!({"type": "agent_end", "awaitingAsyncWork": true}),
            700,
        );
        note_run_frame(&mut s, &json!({"type": "session_settled"}), 800);
        assert_eq!(s.awaiting_tasks_since_ms, None, "settled: nothing pending");
        // Replay of an old log dates the wait by the log, not by now.
        s.replay_clock_ms = Some(42);
        note_run_frame(
            &mut s,
            &json!({"type": "agent_end", "awaitingAsyncWork": true}),
            9_999,
        );
        assert_eq!(s.awaiting_tasks_since_ms, Some(42));
    }

    #[test]
    fn needs_input_tracks_pending_ui() {
        let pending = Mutex::new(Vec::new());
        let summary = Mutex::new(Summary::default());
        set_dialog_pending(&pending, &summary, "d1", Some(Arc::from("{}")));
        assert!(lock(&summary).needs_input);
        set_dialog_pending(&pending, &summary, "d2", Some(Arc::from("{}")));
        set_dialog_pending(&pending, &summary, "d1", None);
        assert!(lock(&summary).needs_input, "d2 is still unanswered");
        set_dialog_pending(&pending, &summary, "d2", None);
        assert!(!lock(&summary).needs_input);
    }
}
