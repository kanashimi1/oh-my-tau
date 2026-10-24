//! One browser-driven agent: an `omp-rs --mode rpc` child process.
//!
//! Every stdout line (one rpc frame) is fanned out to the attached browser
//! sockets; browser frames are written to the child's stdin verbatim. The
//! bridge keeps just enough state to make reattaching cheap: unanswered
//! dialogs (replayed on attach) and a session summary refreshed through the
//! child's own `get_state`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::broadcast;

/// Response id of the bridge's own `get_state` probes (never forwarded).
const STATE_PROBE_ID: &str = "__web_state";
/// Response id of the startup `switch_session` that resumes a stored session
/// (`--resume` is ignored in rpc mode: the rpc runtime owns the session
/// lifecycle, so the bridge resumes through the protocol).
const RESUME_ID: &str = "__web_resume";
/// Dialog methods a browser must answer; replayed to late attachers.
const DIALOG_METHODS: &[&str] = &["select", "confirm", "input", "editor", "ask"];
const STDERR_TAIL: usize = 40;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What the sidebar shows for a running agent.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub session_file: Option<String>,
    pub session_id: Option<String>,
    pub session_name: Option<String>,
    pub model: Option<String>,
    pub streaming: bool,
}

/// How to start an agent.
#[derive(Debug, Clone, Default)]
pub struct Spawn {
    pub cwd: PathBuf,
    /// Session file (or id prefix) to resume.
    pub resume: Option<String>,
    /// `provider/id[:thinking]`.
    pub model: Option<String>,
}

pub struct Agent {
    pub id: String,
    pub cwd: PathBuf,
    pub started_ms: i64,
    stdin: tokio::sync::Mutex<ChildStdin>,
    child: tokio::sync::Mutex<Option<Child>>,
    tx: broadcast::Sender<Arc<str>>,
    /// Unanswered dialog requests in arrival order (id, raw frame).
    pending_ui: Mutex<Vec<(String, Arc<str>)>>,
    summary: Mutex<Summary>,
    stderr: Mutex<VecDeque<String>>,
    /// Bridge notices (e.g. a failed startup resume) replayed to new sockets.
    notices: Mutex<Vec<Arc<str>>>,
    alive: AtomicBool,
}

impl Agent {
    /// Start the child and its reader tasks. `on_exit` runs once the child's
    /// stdout closes (after the exit frame was broadcast).
    pub async fn spawn(
        id: String,
        exe: &Path,
        spec: &Spawn,
        on_exit: impl FnOnce(&str) + Send + 'static,
    ) -> Result<Arc<Self>> {
        let mut cmd = Command::new(exe);
        cmd.arg("--mode").arg("rpc").arg("--cwd").arg(&spec.cwd);
        if let Some(model) = spec.model.as_deref().filter(|m| !m.is_empty()) {
            cmd.arg("--model").arg(model);
        }
        cmd.current_dir(&spec.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {} --mode rpc", exe.display()))?;
        let stdin = child.stdin.take().context("child stdin")?;
        let stdout = child.stdout.take().context("child stdout")?;
        let stderr = child.stderr.take().context("child stderr")?;
        let (tx, _) = broadcast::channel(8192);
        let agent = Arc::new(Self {
            id,
            cwd: spec.cwd.clone(),
            started_ms: pa_agent::now_ms(),
            stdin: tokio::sync::Mutex::new(stdin),
            child: tokio::sync::Mutex::new(Some(child)),
            tx,
            pending_ui: Mutex::new(Vec::new()),
            summary: Mutex::new(Summary {
                session_file: spec.resume.clone(),
                ..Summary::default()
            }),
            stderr: Mutex::new(VecDeque::new()),
            notices: Mutex::new(Vec::new()),
            alive: AtomicBool::new(true),
        });

        let err_agent = Arc::clone(&agent);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut tail = lock(&err_agent.stderr);
                if tail.len() == STDERR_TAIL {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        });

        let out_agent = Arc::clone(&agent);
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let text = line.trim_end_matches(['\n', '\r']);
                if !text.is_empty() {
                    out_agent.on_child_line(text).await;
                }
            }
            out_agent.alive.store(false, Ordering::SeqCst);
            let code = match out_agent.child.lock().await.as_mut() {
                Some(child) => child.wait().await.ok().and_then(|s| s.code()),
                None => None,
            };
            let stderr: Vec<String> = lock(&out_agent.stderr).iter().cloned().collect();
            let frame = json!({"type": "web_agent_exit", "code": code, "stderr": stderr});
            let _ = out_agent.tx.send(Arc::from(frame.to_string()));
            on_exit(&out_agent.id);
        });

        if let Some(resume) = &spec.resume {
            let frame = json!({"id": RESUME_ID, "type": "switch_session", "sessionPath": resume});
            agent.send_line(&frame.to_string()).await?;
        }
        agent.probe_state().await;
        Ok(agent)
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
            "alive": self.is_alive(),
            "sessionFile": s.session_file,
            "sessionId": s.session_id,
            "sessionName": s.session_name,
            "model": s.model,
            "streaming": s.streaming,
        })
    }

    /// Write one browser frame to the child.
    pub async fn send_line(&self, line: &str) -> Result<()> {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v.get("type").and_then(Value::as_str) == Some("extension_ui_response") {
                if let Some(id) = v.get("id").and_then(Value::as_str) {
                    lock(&self.pending_ui).retain(|(p, _)| p != id);
                    // Other tabs showing the same dialog close it.
                    let close = json!({"type": "extension_ui_request", "method": "cancelDialog",
                                       "params": {"id": id}, "web": "answered"});
                    let _ = self.tx.send(Arc::from(close.to_string()));
                }
            }
        }
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;
        Ok(())
    }

    pub async fn kill(&self) {
        if let Some(child) = self.child.lock().await.as_mut() {
            let _ = child.start_kill();
        }
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
            "response" if frame.get("id").and_then(Value::as_str) == Some(RESUME_ID) => {
                if frame.get("success").and_then(Value::as_bool) != Some(true) {
                    let error = frame.get("error").and_then(Value::as_str).unwrap_or("unknown error");
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
                    "new_session" | "switch_session" | "open_session" | "fork" | "branch"
                        | "set_session_name" | "set_model" | "cycle_model"
                );
            }
            "extension_ui_request" => {
                let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
                if DIALOG_METHODS.contains(&method) {
                    if let Some(id) = frame.get("id").and_then(Value::as_str) {
                        lock(&self.pending_ui).push((id.to_string(), Arc::from(text)));
                    }
                } else if method == "cancelDialog" {
                    if let Some(id) = frame.pointer("/params/id").and_then(Value::as_str) {
                        lock(&self.pending_ui).retain(|(p, _)| p != id);
                    }
                }
            }
            "agent_start" => lock(&self.summary).streaming = true,
            "agent_end" | "session_settled" => {
                lock(&self.summary).streaming = false;
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
        }
    }
}
