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
/// Bound retained frames: large history replies must not accumulate for slow clients.
/// Lagging clients receive `web_lagged` and resynchronize their transcript.
const FRAME_QUEUE: usize = 64;
const STDERR_TAIL: usize = 40;
#[cfg(target_os = "linux")]
struct LogEvents(nix::sys::inotify::Inotify);
#[cfg(target_os = "linux")]
impl std::os::fd::AsRawFd for LogEvents {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        use std::os::fd::AsFd;
        self.0.as_fd().as_raw_fd()
    }
}
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

/// Bridge state persisted out-of-band (`state.json`) so adoption restores
/// dialogs / run markers without scanning a multi-GB log. `pending` are
/// unanswered dialog frames `{id, raw payload}`; `awaiting_tasks_since_ms`
/// cannot be re-probed via RPC, so it survives a restart here.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct PersistedState {
    pending: Vec<(String, String)>,
    awaiting_tasks_since_ms: Option<i64>,
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
    /// Requested agent role (`POST /api/agents {role}`): the child runs as the
    /// matching registered agent definition, not as a generic session.
    pub role: Option<String>,
    /// `POST /api/agents {worktree:true}`: run the child in a fresh git
    /// worktree (branch `omt/<slug>` of the role or id) instead of `cwd`.
    /// A git failure falls back to plain `cwd` — the spawn still succeeds.
    pub worktree: bool,
}

/// Roles the web UI offers for a new agent (`OMP.agentPanel`). `universal`
/// keeps the unrestricted default session; every other role must resolve to a
/// registered agent definition (project `.omp/agents` shadows
/// `~/.omp/agent/agents`) so the role picks real instructions, tools and
/// model — never a cosmetic label.
pub const AGENT_ROLES: &[&str] = &["universal", "coder", "blockbench", "blender", "reviewer"];

/// Resolve a requested role to a registered agent definition: a definition
/// named exactly like the role wins, then a stable preferred general-purpose
/// definition, then any name containing the role token as a fallback.
/// `home` is a parameter (not `HOME`) so tests can point at a fixture dir.
fn resolve_role_in(
    cwd: &Path,
    home: &Path,
    role: &str,
) -> Option<omp_task::agent::AgentDefinition> {
    let mut all = omp_task::discovery::discover_project_agents(cwd).agents;
    all.extend(omp_task::discovery::discover_user_agents(home).agents);
    let token = role.trim().to_lowercase();
    if let Some(def) = all.iter().find(|a| a.fields.name.to_lowercase() == token) {
        return Some(def.clone());
    }
    let preferred = match token.as_str() {
        "coder" => &["kimi-coder", "haiku-coder"][..],
        "blockbench" => &["blockbench-modeler"][..],
        "blender" => &["blender-modeler"][..],
        "reviewer" => &["asymmetric-reviewer"][..],
        _ => &[][..],
    };
    for name in preferred {
        if let Some(def) = all.iter().find(|a| a.fields.name == *name) {
            return Some(def.clone());
        }
    }
    all.iter()
        .filter(|a| a.fields.name.to_lowercase().contains(&token))
        .min_by(|a, b| a.fields.name.cmp(&b.fields.name))
        .cloned()
}

/// The argv a role contributes to the child: the definition's tool
/// allow-list (`--tools`), body as the system-prompt addition
/// (`--append-system-prompt`), `thinkingLevel` (`--thinking`) and model
/// chain (`--model`, only when the caller did not pick one). Mirrors the
/// task tool's `stamp_agent_definition`, so a web agent with a role runs on
/// exactly the configuration a `task` spawn of that agent would get.
fn role_args(model: Option<&str>, def: &omp_task::agent::AgentDefinition) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(tools) = def.fields.tools.as_ref().filter(|t| !t.is_empty()) {
        args.extend(["--tools".to_string(), tools.join(",")]);
    }
    let body = def.system_prompt.trim();
    if !body.is_empty() {
        args.extend(["--append-system-prompt".to_string(), body.to_string()]);
    }
    let level = match def.fields.thinking_level {
        Some(omp_task::agent::ThinkingLevel::Low) => Some("low"),
        Some(omp_task::agent::ThinkingLevel::Medium) => Some("medium"),
        Some(omp_task::agent::ThinkingLevel::High) => Some("high"),
        Some(omp_task::agent::ThinkingLevel::Max) => Some("max"),
        Some(omp_task::agent::ThinkingLevel::Other(_)) | None => None,
    };
    // A `provider/model:level` selector already pins the level.
    let pinned = model.is_some_and(|m| m.split(',').next().unwrap_or("").contains(':'));
    if let (Some(level), false) = (level, pinned) {
        args.extend(["--thinking".to_string(), level.to_string()]);
    }
    let chain = def
        .fields
        .model
        .as_ref()
        .filter(|chain| !chain.is_empty())
        .map(|chain| chain.join(","));
    if let Some(chain) = chain.filter(|_| model.is_none_or(|m| m.trim().is_empty())) {
        args.extend(["--model".to_string(), chain]);
    }
    args
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
    /// Requested role: the resolved agent definition name (`spawn`); adopted
    /// agents report none (the registry entry carries no role yet).
    role: Option<String>,
    /// Git worktree the child runs in (`worktree:true` spawn); adopted
    /// agents report none (the registry entry carries no worktree).
    worktree: Option<PathBuf>,
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

        // `worktree:true`: the child runs in a fresh git worktree (branch
        // `omt/<slug>` of the requested role, else the id) via the `/wt new`
        // helpers. A git failure downgrades to plain `cwd` so the spawn
        // still succeeds.
let (cwd, worktree) = if spec.worktree {
            let slug = spec
                .role
                .as_deref()
                .map(str::trim)
                .filter(|r| !r.is_empty())
                .unwrap_or(&id);
            let created = crate::features::worktree::create_worktree(&spec.cwd, slug)
                .or_else(|_| crate::features::worktree::create_worktree(&spec.cwd, &id));
            match created {
                Ok(path) => (path.clone(), Some(path)),
                Err(e) => {
                    eprintln!("omt web: worktree spawn fell back to {}: {e}", spec.cwd.display());
                    (spec.cwd.clone(), None)
                }
            }
        } else {
            (spec.cwd.clone(), None)
        };

        let mut cmd = Command::new(exe);
        cmd.arg("--mode").arg("rpc").arg("--cwd").arg(&cwd);
        if let Some(model) = spec.model.as_deref().filter(|m| !m.is_empty()) {
            cmd.arg("--model").arg(model);
        }
        if let Some(mode) = spec.approval_mode.as_deref().filter(|m| !m.is_empty()) {
            cmd.args(["--approval-mode", mode]);
        }
        // A requested role re-configures the child beyond a label: the
        // resolved definition's instructions, tool allow-list, thinking level
        // and model chain become real child flags. `universal` is the
        // explicit default (no stamping). An unknown role fails the spawn so
        // the UI reports it instead of silently running a generic session.
        let mut role = spec.role.clone();
        if let Some(requested) = role.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
            if requested != "universal" {
                let home = std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .ok_or_else(|| anyhow!("HOME is not set; cannot resolve role {requested:?}"))?;
                let Some(def) = resolve_role_in(&cwd, &home, requested) else {
                    let discovered: Vec<String> = {
                        let mut names = omp_task::discovery::discover_project_agents(&cwd)
                            .agents
                            .iter()
                            .chain(
                                omp_task::discovery::discover_user_agents(&home)
                                    .agents
                                    .iter(),
                            )
                            .map(|a| a.fields.name.clone())
                            .collect::<Vec<_>>();
                        names.sort();
                        names.dedup();
                        names
                    };
                    return Err(anyhow!(
                        "unknown agent role {requested:?} (expected one of {:?}); discovered agent definitions: {}",
                        AGENT_ROLES,
                        if discovered.is_empty() {
                            "none".to_string()
                        } else {
                            discovered.join(", ")
                        }
                    ));
                };
                for chunk in role_args(spec.model.as_deref().filter(|m| !m.trim().is_empty()), &def)
                {
                    cmd.arg(chunk);
                }
                role = Some(def.fields.name.clone());
            } else {
                role = Some("universal".to_string());
            }
        } else {
            role = None;
        }
        #[cfg(unix)]
        {
            // The child leads its own process group, so it survives hub restarts
            // and can be SIGTERM'd as a whole on DELETE.
            cmd.process_group(0);
        }
        cmd.current_dir(&cwd)
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

        let (tx, _) = broadcast::channel(FRAME_QUEUE);
        let agent = Arc::new(Self {
            id: id.clone(),
            cwd,
            started_ms: pa_agent::now_ms(),
            pid,
            pgid: group_id,
            adopted: false,
            approval_mode: spec.approval_mode.clone(),
            role,
            worktree,
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
        // A freshly spawned agent starts with empty persisted state (any
        // state.json here belongs to a previous incarnation of the id).
        detach::save_state(&id, &PersistedState::default());

        Self::start_tail(Arc::clone(&agent), log_path);
        Self::start_exit_watcher(Arc::clone(&agent), on_exit);

        agent.negotiate_protocol().await?;
        if let Some(resume) = &spec.resume {
            let frame = json!({"id": RESUME_ID, "type": "switch_session", "sessionPath": resume});
            agent.send_line(&frame.to_string()).await?;
        }
        agent.probe_state().await;
        Ok(agent)
    }

    /// Adopt a live detached child from the registry: restore dialogs / run
    /// markers from the persisted bridge state, connect its control socket,
    /// and watch it for exit. No `Child` handle is held — the process is not
    /// our child.
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
            role: None,
            worktree: None,
            control: tokio::sync::Mutex::new(Some(control)),
            child: tokio::sync::Mutex::new(None),
            tx: broadcast::channel(FRAME_QUEUE).0,
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
        // Adoption restores dialogs / run markers from persisted state — not by
        // scanning the (possibly multi-GB) log — then tails live from the end.
        agent.restore_state();
        Self::start_tail(Arc::clone(&agent), detach::log_path(&entry.id));
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

    /// Tail the agent log from its current end. JSON object lines go to
    /// `on_child_line`; other lines (the child's stderr, mixed into the same
    /// file) feed the stderr tail used by the exit frame. The log is never
    /// replayed from the front — adoption restores state from `state.json`.
    fn start_tail(agent: Arc<Self>, log_path: PathBuf) {
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncSeekExt};
            let Ok(mut file) = tokio::fs::File::open(&log_path).await else {
                agent.alive.store(false, Ordering::SeqCst);
                return;
            };
            let _ = file.seek(std::io::SeekFrom::End(0)).await;
            #[cfg(target_os = "linux")]
            let log_events = (|| {
                use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
                let inotify = Inotify::init(InitFlags::IN_NONBLOCK).ok()?;
                inotify
                    .add_watch(
                        &log_path,
                        AddWatchFlags::IN_MODIFY
                            | AddWatchFlags::IN_CLOSE_WRITE
                            | AddWatchFlags::IN_MOVE_SELF
                            | AddWatchFlags::IN_DELETE_SELF,
                    )
                    .ok()?;
                tokio::io::unix::AsyncFd::new(LogEvents(inotify)).ok()
            })();
            let mut carry: Vec<u8> = Vec::new();
            let mut buf = [0u8; 8192];
            loop {
                match file.read(&mut buf).await {
                    Ok(0) => {
                        if !agent.alive.load(Ordering::SeqCst) {
                            break;
                        }
                        #[cfg(target_os = "linux")]
                        if let Some(events) = &log_events {
                            match tokio::time::timeout(Duration::from_millis(50), events.readable())
                                .await
                            {
                                Ok(Ok(mut ready)) => {
                                    let _ = events.get_ref().0.read_events();
                                    ready.clear_ready();
                                }
                                Ok(Err(_)) => {
                                    tokio::time::sleep(Duration::from_millis(50)).await;
                                }
                                Err(_) => {}
                            }
                            continue;
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
        // The child is dead: the log is complete, so rotate it safely (the
        // live writer/tail are gone).
        detach::rotate_log(&self.id);
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
            "role": self.role,
            "worktree": self.worktree.as_ref().map(|p| p.display().to_string()),
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
                    self.set_dialog(id, None);
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
        drop(control);
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

    /// Snapshot the bridge state that must survive a hub restart (pending
    /// dialogs, parked-task marker) to `state.json`. Adoption reads this
    /// instead of scanning the log, so it stays O(1) in log size.
    fn persist_state(&self) {
        let pending = lock(&self.pending_ui)
            .iter()
            .map(|(id, f)| (id.clone(), f.to_string()))
            .collect();
        let awaiting_tasks_since_ms = lock(&self.summary).awaiting_tasks_since_ms;
        detach::save_state(
            &self.id,
            &PersistedState {
                pending,
                awaiting_tasks_since_ms,
            },
        );
    }

    /// Restore bridge state saved by a previous hub incarnation (adoption).
    fn restore_state(&self) {
        if let Some(st) = detach::load_state::<PersistedState>(&self.id) {
            let mut pending = lock(&self.pending_ui);
            *pending = st
                .pending
                .into_iter()
                .map(|(id, f)| (id, Arc::from(f)))
                .collect();
            lock(&self.summary).awaiting_tasks_since_ms = st.awaiting_tasks_since_ms;
            lock(&self.summary).needs_input = !pending.is_empty();
        }
    }

    /// Push (`frame`) or resolve (`None`) a dialog, keeping `needs_input`
    /// live and persisting the queue so adoption can restore it.
    fn set_dialog(&self, id: &str, frame: Option<Arc<str>>) {
        set_dialog_pending(&self.pending_ui, &self.summary, id, frame);
        self.persist_state();
    }

    /// Apply a run-state frame and persist any parked-task marker.
    fn note_run(&self, frame: &Value) {
        note_run_frame(&mut lock(&self.summary), frame, pa_agent::now_ms());
        self.persist_state();
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
            // An answered dialog: close the stale pending copy when the answer
            // catches up; never rebroadcast the answer or forward it again.
            "extension_ui_response" => {
                if let Some(id) = frame.get("id").and_then(Value::as_str) {
                    self.set_dialog(id, None);
                    let close = json!({"type": "extension_ui_request", "method": "cancelDialog",
                                       "params": {"id": id}, "web": "answered"});
                    let _ = self.tx.send(Arc::from(close.to_string()));
                }
                return;
            }
            "extension_ui_request" => {
                let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
                if DIALOG_METHODS.contains(&method) {
                    if let Some(id) = frame.get("id").and_then(Value::as_str) {
                        self.set_dialog(id, Some(Arc::from(text)));
                    }
                } else if method == "cancelDialog" {
                    if let Some(id) = frame.pointer("/params/id").and_then(Value::as_str) {
                        self.set_dialog(id, None);
                    }
                }
            }
            "agent_start" => self.note_run(&frame),
            "agent_end" | "session_settled" => {
                self.note_run(&frame);
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
/// `agent_end` with `awaitingAsyncWork` starts the clock, any new run or a
/// normal settle clears it.
fn note_run_frame(s: &mut Summary, frame: &Value, now_ms: i64) {
    match frame.get("type").and_then(Value::as_str) {
        Some("agent_start") => {
            s.streaming = true;
            s.awaiting_tasks_since_ms = None;
        }
        Some("agent_end") => {
            s.streaming = false;
            if frame.get("awaitingAsyncWork").and_then(Value::as_bool) == Some(true) {
                s.awaiting_tasks_since_ms.get_or_insert(now_ms);
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
    /// `HOME` is process-global; serialize the tests that redirect it to a
    /// tempdir so they cannot clobber each other (they run in parallel).
    static HOME_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// A bridge with no child: just enough to drive `feed_log_line`.
    fn bare_agent() -> Agent {
        Agent {
            id: "t".into(),
            cwd: PathBuf::new(),
            started_ms: 0,
            pid: 0,
            pgid: 0,
            adopted: false,
            approval_mode: None,
            role: None,
            worktree: None,
            control: tokio::sync::Mutex::new(None),
            child: tokio::sync::Mutex::new(None),
            tx: broadcast::channel(FRAME_QUEUE).0,
            pending_ui: Mutex::new(Vec::new()),
            summary: Mutex::new(Summary::default()),
            stderr: Mutex::new(VecDeque::new()),
            notices: Mutex::new(Vec::new()),
            chunks: Mutex::new(crate::rpc::ChunkAssembler::default()),
            alive: AtomicBool::new(true),
        }
    }

    /// One streamed assistant reply, as the runtime frames it.
    fn stream(reply: &str) -> Vec<Value> {
        let msg = |text: &str| {
            json!({"role": "assistant", "content": [{"type": "text", "text": text}],
                   "provider": "p", "model": "m", "usage": {"input": 1, "output": 2},
                   "stopReason": "stop", "timestamp": 1})
        };
        let mut frames = vec![
            json!({"type": "agent_start"}),
            json!({"type": "message_start", "message": msg("")}),
        ];
        let mut text = String::new();
        for chunk in reply.split_inclusive(' ') {
            text.push_str(chunk);
            frames.push(json!({"type": "message_update", "message": msg(&text),
                "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": chunk}}));
        }
        frames.push(json!({"type": "message_end", "message": msg(&text)}));
        frames.push(json!({"type": "agent_end", "messages": [msg(&text)]}));
        frames
    }

    /// Feed `lines` to a fresh bridge; collect what its websocket clients get.
    async fn broadcast_of(lines: &[String]) -> Vec<String> {
        let agent = bare_agent();
        let mut rx = agent.subscribe();
        let mut out = Vec::new();
        for line in lines {
            agent.feed_log_line(line).await;
            while let Ok(frame) = rx.try_recv() {
                out.push(frame.to_string());
            }
        }
        out
    }

    /// The child logs `message_update` as deltas; websocket clients get those
    /// compact frames verbatim (the browser reconstructs them), so the web
    /// path never expands/serializes O(reply²) full frames. Attachment/TUI
    /// still expand via `rpc::UpdateExpander` (see attach.rs) and are
    /// unaffected.
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn web_receives_compact_delta_frames_verbatim() {
        let _guard = HOME_LOCK.lock().await;
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", home.path());
        let reply: String = "lorem ipsum dolor sit amet ü ".repeat(300);
        let frames = stream(&reply);
        let plain = crate::rpc::encode_log(&frames, false);
        let compact = crate::rpc::encode_log(&frames, true);
        let bytes = |l: &[String]| l.iter().map(|l| l.len() + 1).sum::<usize>();
        assert!(bytes(&compact) * 20 < bytes(&plain));
        assert!(compact.iter().any(|l| l.contains("\"messageDelta\"")));
        assert_eq!(
            broadcast_of(&compact).await,
            compact,
            "delta frames must pass through verbatim"
        );
        // Joining mid-stream: deltas (with their keyframe base) forward as-is.
        let mid = compact.len() / 3;
        let got = broadcast_of(&compact[mid..]).await;
        assert_eq!(got, &compact[mid..]);
        assert!(got.iter().any(|l| l.contains("\"messageDelta\"")));
    }

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

    /// An answered dialog is dropped from the persisted bridge state, so after
    /// a hub restart adoption restores no stale ask (no resurrection).
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn answered_dialog_is_cleared_from_persisted_state() {
        let _guard = HOME_LOCK.lock().await;
        let request = json!({"type":"extension_ui_request","method":"ask","id":"d1",
                             "params":{"message":"allow write?"}})
        .to_string();
        let answer = json!({"type":"extension_ui_response","id":"d1",
                            "params":{"confirm":true}})
        .to_string();
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", home.path());

        let agent = bare_agent();
        agent.feed_log_line(&request).await;
        assert_eq!(lock(&agent.pending_ui).len(), 1);
        assert!(lock(&agent.summary).needs_input);
        let st: PersistedState = detach::load_state("t").expect("request persisted");
        assert_eq!(st.pending.len(), 1);
        assert_eq!(st.pending[0].0, "d1");

        agent.send_line(&answer).await.unwrap();
        assert!(lock(&agent.pending_ui).is_empty());
        assert!(!lock(&agent.summary).needs_input);
        let st: PersistedState = detach::load_state("t").expect("state present");
        assert!(st.pending.is_empty(), "answered dialog dropped from state");

        // Adoption recovery reads the persisted state: no stale ask remains.
        let adopted = bare_agent();
        adopted.restore_state();
        assert!(lock(&adopted.pending_ui).is_empty());
        assert!(!lock(&adopted.summary).needs_input);
    }

    /// Several dialogs: only the answered id is dropped, an unanswered one
    /// stays pending (and answered-then-new reuse of the id stays open).
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn replay_clears_only_answered_dialogs() {
        let _guard = HOME_LOCK.lock().await;
        let ask = |id: &str| {
            json!({"type": "extension_ui_request", "method": "ask", "id": id,
                   "params": {"message": "m"}})
            .to_string()
        };
        let answer = |id: &str| {
            json!({"type": "extension_ui_response", "id": id, "params": {"confirm": false}})
                .to_string()
        };
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", home.path());
        let agent = bare_agent();
        agent.feed_log_line(&ask("d1")).await;
        agent.feed_log_line(&ask("d2")).await;
        agent.feed_log_line(&answer("d1")).await;
        {
            let pending = lock(&agent.pending_ui);
            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].0, "d2");
            assert!(lock(&agent.summary).needs_input);
        }
        agent.feed_log_line(&ask("d3")).await;
        agent.feed_log_line(&answer("d3")).await;
        agent.feed_log_line(&ask("d3")).await; // id reuse: pending again
        assert_eq!(lock(&agent.pending_ui).len(), 2);
    }

    /// Adoption restores unanswered dialogs and the parked-task marker from the
    /// persisted state file (not from the log).
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn restore_state_seeds_pending_dialogs_and_wait_marker() {
        let _guard = HOME_LOCK.lock().await;
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", home.path());
        detach::save_state(
            "t",
            &PersistedState {
                pending: vec![(
                    "d9".into(),
                    r#"{"type":"extension_ui_request","method":"ask","id":"d9"}"#.into(),
                )],
                awaiting_tasks_since_ms: Some(12345),
            },
        );
        let agent = bare_agent();
        agent.restore_state();
        assert_eq!(lock(&agent.pending_ui).len(), 1);
        assert_eq!(lock(&agent.pending_ui)[0].0, "d9");
        assert!(lock(&agent.summary).needs_input);
        assert_eq!(lock(&agent.summary).awaiting_tasks_since_ms, Some(12345));
    }

    /// Adoption never replays the log from the front: the live tail opens at
    /// the current end, so preexisting history neither sets streaming nor is
    /// broadcast. (Dialogs come from `state.json`, see the tests above.)
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn adopted_tail_skips_preexisting_log_history() {
        let _guard = HOME_LOCK.lock().await;
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", home.path());
        let log_path = detach::agents_dir().join("t").join("log");
        std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
        let mut log = String::new();
        for _ in 0..5_000 {
            log.push_str(r#"{"type":"command_output","text":"noise"}"#);
            log.push('\n');
        }
        log.push_str(r#"{"type":"agent_start"}"#);
        log.push('\n');
        std::fs::write(&log_path, log).unwrap();

        let agent = Arc::new(bare_agent());
        let mut rx = agent.subscribe();
        Agent::start_tail(Arc::clone(&agent), log_path);
        tokio::time::sleep(Duration::from_millis(350)).await;
        assert!(
            !lock(&agent.summary).streaming,
            "preexisting history was replayed"
        );
        assert!(rx.try_recv().is_err(), "preexisting frames were broadcast");
        agent.alive.store(false, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn tail_wakes_on_new_frames_without_poll_delay() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("log");
        std::fs::write(&log_path, "").unwrap();
        let agent = Arc::new(bare_agent());
        let mut rx = agent.subscribe();
        Agent::start_tail(Arc::clone(&agent), log_path.clone());
        tokio::time::sleep(Duration::from_millis(120)).await;
        let mut log = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .unwrap();
        writeln!(log, r#"{{"type":"command_output","text":"live"}}"#).unwrap();
        log.flush().unwrap();
        let frame = tokio::time::timeout(Duration::from_millis(25), rx.recv())
            .await
            .expect("new frame waited for the old 50 ms polling interval")
            .unwrap();
        assert!(frame.contains("live"));
        agent.alive.store(false, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(60)).await;
    }

    #[tokio::test]
    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    async fn slow_receivers_do_not_retain_unbounded_frames() {
        let agent = bare_agent();
        let mut rx = agent.subscribe();
        for _ in 0..FRAME_QUEUE + 8 {
            agent
                .feed_log_line(r#"{"type":"command_output","text":"x"}"#)
                .await;
        }
        assert!(matches!(
            rx.recv().await,
            Err(tokio::sync::broadcast::error::RecvError::Lagged(8))
        ));
    }
}

#[cfg(test)]
mod role_tests {
    use super::{resolve_role_in, role_args, AGENT_ROLES};
    use omp_task::agent::{parse_agent, AgentSource};
    use std::fs;
    use std::path::Path;

    /// Fixture dirs: `cwd/.omp/agents/<name>.md` (project, shadows) and
    /// `<home>/.omp/agent/agents/<name>.md` (user level).
    fn fixture(cwd: &Path, home: &Path, project: &[(&str, &str)], user: &[(&str, &str)]) {
        for (dir, defs) in [
            (cwd.join(".omp/agents"), project),
            (home.join(".omp/agent/agents"), user),
        ] {
            fs::create_dir_all(&dir).unwrap();
            for (name, body) in defs {
                fs::write(
                    dir.join(format!("{name}.md")),
                    format!("---\nname: {name}\ndescription: d\n{body}\n---\nDo the {name} job."),
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn exact_name_beats_substring() {
        let tmp = std::env::temp_dir().join(format!("omp-role-{}", std::process::id()));
        let cwd = tmp.join("proj");
        let home = tmp.join("home");
        let _ = fs::remove_dir_all(&tmp);
        fixture(
            &cwd,
            &home,
            &[("coder", "tools: [read]")],
            &[
                ("haiku-coder", "tools: [read, grep]"),
                ("reviewer", "tools: [read]"),
            ],
        );
        let hit = resolve_role_in(&cwd, &home, "coder").unwrap();
        assert_eq!(hit.fields.name, "coder");
        // The parser appends `yield` to every explicit tool list.
        assert_eq!(hit.fields.tools.as_deref().unwrap(), ["read", "yield"]);
        // Substring fallback, deterministic (lexicographically first).
        assert_eq!(
            resolve_role_in(&cwd, &home, "coder").unwrap().fields.name,
            "coder"
        );
        assert_eq!(
            resolve_role_in(&cwd, &home, "review").unwrap().fields.name,
            "reviewer"
        );
        // Project definitions shadow user-level ones of the same name.
        fixture(
            &cwd,
            &home,
            &[("blockbench", "tools: [read]")],
            &[("blockbench", "tools: [bash]")],
        );
        let bb = resolve_role_in(&cwd, &home, "blockbench").unwrap();
        assert_eq!(bb.fields.tools.as_deref().unwrap(), ["read", "yield"]);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn preferred_role_does_not_depend_on_lexicographic_order() {
        let tmp = std::env::temp_dir().join(format!("omp-role-preferred-{}", std::process::id()));
        let cwd = tmp.join("proj");
        let home = tmp.join("home");
        let _ = fs::remove_dir_all(&tmp);
        fixture(
            &cwd,
            &home,
            &[],
            &[("cat-jam-coder", ""), ("kimi-coder", "")],
        );
        assert_eq!(
            resolve_role_in(&cwd, &home, "coder").unwrap().fields.name,
            "kimi-coder"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn stamp_turns_a_definition_into_real_child_flags() {
        let def = parse_agent(
            "test-coder.md",
            "---\nname: coder\ndescription: d\ntools: [read, edit, bash]\nmodel: zai/glm-5.3-flash, kimi/k2\nthinkingLevel: high\n---\nCheck every edit with the build.\n",
            AgentSource::User,
        )
        .unwrap();
        // Caller picked no model: definition's model chain applies too.
        let args = role_args(None, &def);
        let expected = [
            "--tools",
            "read,edit,bash,yield",
            "--append-system-prompt",
            "Check every edit with the build.",
            "--thinking",
            "high",
            "--model",
            "zai/glm-5.3-flash,kimi/k2",
        ];
        assert_eq!(
            args,
            expected.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        // Caller's model wins over the chain; a pinned `:level` keeps --thinking off.
        let args = role_args(Some("openai/gpt-5:high"), &def);
        assert!(!args.iter().any(|a| a == "--thinking"));
        assert!(!args.iter().any(|a| a == "--model"));
    }

    #[test]
    fn known_roles_and_unknown_resolution() {
        assert_eq!(
            AGENT_ROLES,
            ["universal", "coder", "blockbench", "blender", "reviewer"]
        );
        let tmp = std::env::temp_dir().join(format!("omp-role-none-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // No agents dir at all: only `universal` resolves (the default);
        // everything else must report a clear miss.
        assert!(resolve_role_in(&tmp, &tmp, "universal").is_none());
        assert!(resolve_role_in(&tmp, &tmp, "coder").is_none());
        let _ = fs::remove_dir_all(&tmp);
    }
}
