use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>;

/// How to start the host process.
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// `bun` executable.
    pub bun: String,
    /// Path to `ext-host/host.ts`.
    pub host_script: PathBuf,
    /// Append one JSON line per emitted event (per-handler action/ms) to this file.
    pub trace: Option<PathBuf>,
    /// Handler budget in ms (host default 30000).
    pub handler_timeout_ms: Option<u64>,
    /// Extra environment for the host. `HOOK_TRACE=0` is set by default so the user's own
    /// `lib/hook-trace.ts` wrapper does not write into the real `~/.omp/state/hook_trace.jsonl`.
    pub env: Vec<(String, String)>,
    /// Opt-in shared host (`OMP_RS_EXT_SHARED=1`): instead of spawning one bun
    /// process per session, connect to this unix socket, starting the shared
    /// daemon (`ext-host/shard.ts`) on demand. The daemon runs each session in
    /// its own bun Worker (fresh module graph == fresh process-level state).
    /// Per-session settings (trace, handler timeout) travel with the `load` RPC.
    pub shared_socket: Option<PathBuf>,
}

impl HostConfig {
    #[must_use]
    pub fn new(host_script: impl Into<PathBuf>) -> Self {
        Self {
            bun: "bun".into(),
            host_script: host_script.into(),
            trace: None,
            handler_timeout_ms: None,
            env: Vec::new(),
            shared_socket: None,
        }
    }
}

/// Message initiated by the host (not a response).
#[derive(Debug, Clone)]
pub enum HostMessage {
    /// Fire-and-forget: `append_entry`, `send_message`, `send_user_message`, `log`, `ui_notify`, ...
    Notification { method: String, params: Value },
    /// Needs an answer via [`ExtHost::respond`]: `compact`, `navigate_tree`, `ui_confirm`, ...
    Request {
        id: Value,
        method: String,
        params: Value,
    },
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadedExtension {
    pub path: String,
    pub name: String,
    #[serde(default)]
    pub handlers: HashMap<String, u64>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadResult {
    pub loaded: Vec<LoadedExtension>,
    pub errors: Vec<LoadError>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadError {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct HandlerRecord {
    pub ext: String,
    pub event: String,
    #[serde(default)]
    pub tool: Option<String>,
    pub ms: f64,
    pub action: String,
    #[serde(default)]
    pub error: Option<String>,
}

/// Aggregated outcome of one `emit`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EmitResult {
    /// Event-specific aggregate (see `ext-host/host.ts`); `Null` when no handler produced anything.
    #[serde(default)]
    pub result: Value,
    #[serde(default)]
    pub handlers: Vec<HandlerRecord>,
    /// Handlers that threw or timed out; the host keeps running (omp reports these per handler).
    #[serde(default)]
    pub errors: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    pub extension: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CommandInfo {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub extension: String,
}

/// Delivery state of host-initiated messages (`Incoming`).
enum Delivery {
    /// No consumer yet: messages buffer here until `take_incoming` attaches one.
    Pending(VecDeque<HostMessage>),
    /// A consumer is attached; messages stream to its channel.
    Live,
}

/// Channel halves plus delivery state; one lock makes "check-then-deliver" atomic.
struct Incoming {
    delivery: Delivery,
    tx: mpsc::UnboundedSender<HostMessage>,
    rx: Option<mpsc::UnboundedReceiver<HostMessage>>,
}

/// Connection to a running extension host.
pub struct ExtHost {
    /// Event types with at least one handler across all loaded extensions,
    /// learned from `load` (mirrors the TS runner's `hasHandlers` gate).
    /// `None` = not loaded yet / unknown: send everything.
    subscribed: Mutex<Option<HashSet<String>>>,
    /// Newline-delimited JSON-RPC writer (child stdin or shared-daemon socket).
    writer: tokio::sync::Mutex<BoxedWriter>,
    /// The spawned child; `None` in shared mode (the daemon outlives this
    /// session and is shared with other sessions, so we never kill it).
    child: tokio::sync::Mutex<Option<Child>>,
    pending: Pending,
    next_id: AtomicU64,
    incoming: Mutex<Incoming>,
    /// Per-session trace file forwarded with the `load` RPC (shared mode).
    trace: Option<String>,
    /// Per-session handler budget forwarded with the `load` RPC (shared mode).
    handler_timeout_ms: Option<u64>,
}

/// Object-safe writer half of `AsyncWriteExt` (child stdin or daemon socket).
type BoxedWriter = Box<dyn tokio::io::AsyncWrite + Send + Unpin>;

impl ExtHost {
    /// Spawn `bun <host_script>` (or connect to the shared daemon when
    /// [`HostConfig::shared_socket`] is set) and start the reader tasks.
    ///
    /// # Errors
    /// Fails when the process cannot be spawned or the shared daemon does not
    /// accept a connection within ~10 s.
    pub async fn spawn(cfg: &HostConfig) -> Result<Arc<Self>> {
        if let Some(sock) = &cfg.shared_socket {
            return Self::spawn_shared(cfg, sock).await;
        }
        let mut cmd = Command::new(&cfg.bun);
        // `--smol` trades more frequent GC for a smaller heap footprint
        // (~3-4 MB per session; measured 92.8 → 87.0 MB peak on the 36-extension
        // suite). Per-event cost impact measured at parity by bench s6.
        // Opt out with OMP_EXT_NO_SMOL=1 (e.g. older bun without the flag).
        // Only passed to a real `bun` binary — tests substitute fake runners.
        if std::env::var_os("OMP_EXT_NO_SMOL").is_none()
            && cfg.bun.rsplit('/').next().is_some_and(|b| b == "bun" || b.starts_with("bun-"))
        {
            cmd.arg("--smol");
        }
        cmd.arg(&cfg.host_script)
            .env("HOOK_TRACE", "0")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(t) = &cfg.trace {
            cmd.env("OMP_EXT_TRACE", t);
        }
        if let Some(ms) = cfg.handler_timeout_ms {
            cmd.env("OMP_EXT_TIMEOUT_MS", ms.to_string());
        }
        cmd.envs(cfg.env.iter().map(|(k, v)| (k, v)));
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {} {}", cfg.bun, cfg.host_script.display()))?;
        let stdin = child.stdin.take().context("host stdin")?;
        let stdout = child.stdout.take().context("host stdout")?;
        let stderr = child.stderr.take().context("host stderr")?;

        let pending: Pending = Arc::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let host = Arc::new(Self {
            subscribed: Mutex::new(None),
            writer: tokio::sync::Mutex::new(Box::new(stdin)),
            child: tokio::sync::Mutex::new(Some(child)),
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            incoming: Mutex::new(Incoming {
                delivery: Delivery::Pending(VecDeque::new()),
                tx,
                rx: Some(rx),
            }),
            trace: cfg.trace.as_ref().map(|t| t.to_string_lossy().into_owned()),
            handler_timeout_ms: cfg.handler_timeout_ms,
        });

        let weak = Arc::downgrade(&host);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                route(&msg, &pending, weak.upgrade());
            }
            // Host gone: fail everything still waiting.
            for (_, w) in pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain()
            {
                let _ = w.send(Err("extension host exited".into()));
            }
        });
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[ext-host] {line}");
            }
        });
        Ok(host)
    }

    /// Receiver of host-initiated messages; available once. Messages that
    /// arrived before the consumer attached are delivered first, so an
    /// extension's load-time requests are not lost to the "not supported"
    /// race (`ReviewWave3`). Without a taker, messages keep buffering.
    pub fn take_incoming(&self) -> Option<mpsc::UnboundedReceiver<HostMessage>> {
        let mut inc = self
            .incoming
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rx = inc.rx.take()?;
        if let Delivery::Pending(buf) = &mut inc.delivery {
            let buffered: Vec<_> = buf.drain(..).collect();
            for msg in buffered {
                let _ = inc.tx.send(msg);
            }
        }
        inc.delivery = Delivery::Live;
        Some(rx)
    }

    /// Answer a [`HostMessage::Request`].
    ///
    /// # Errors
    /// Fails when the host's stdin is closed.
    pub async fn respond(&self, id: Value, result: Result<Value, String>) -> Result<()> {
        let msg = match result {
            Ok(v) => json!({"jsonrpc":"2.0","id":id,"result":v}),
            Err(m) => json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":m}}),
        };
        self.write(&msg).await
    }

    async fn write(&self, msg: &Value) -> Result<()> {
        let mut line = serde_json::to_vec(msg)?;
        line.push(b'\n');
        let mut writer = self.writer.lock().await;
        writer.write_all(&line).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Raw JSON-RPC request.
    ///
    /// # Errors
    /// Transport failure, host exit, or a JSON-RPC error response.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, tx);
        if let Err(e) = self
            .write(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .await
        {
            self.pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            return Err(e);
        }
        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(m)) => Err(anyhow!("{method}: {m}")),
            Err(_) => bail!("{method}: extension host dropped the request"),
        }
    }

    /// Load extensions: explicit `paths`, or every extension in `dir` (default
    /// `~/.omp/agent/extensions`) when `paths` is `None`.
    ///
    /// # Errors
    /// Transport failure. Per-extension failures are returned in [`LoadResult::errors`].
    pub async fn load(
        &self,
        paths: Option<&[PathBuf]>,
        dir: Option<&str>,
        cwd: &str,
        ctx: Option<Value>,
    ) -> Result<LoadResult> {
        let mut p = json!({ "cwd": cwd });
        if let Some(paths) = paths {
            p["paths"] = json!(paths);
        }
        if let Some(dir) = dir {
            p["dir"] = json!(dir);
        }
        if let Some(ctx) = ctx {
            p["ctx"] = ctx;
        }
        // Per-session overrides for the shared daemon (its env is process-wide);
        // the stdio host reads the same values from its env / OMP_EXT_TRACE.
        if let Some(ms) = self.handler_timeout_ms {
            p["timeoutMs"] = json!(ms);
        }
        if let Some(trace) = &self.trace {
            p["trace"] = json!(trace);
        }
        let result: LoadResult = serde_json::from_value(self.request("load", p).await?)?;
        let subscribed: HashSet<String> = result
            .loaded
            .iter()
            .flat_map(|e| e.handlers.iter())
            .filter(|(_, n)| **n > 0)
            .map(|(ty, _)| ty.clone())
            .collect();
        *self
            .subscribed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(subscribed);
        Ok(result)
    }

    /// Emit an omp event (`{"type": ..., ...}`); `ctx` merges into the host's session state
    /// (`sessionId`, `sessionFile`, `branch`, `leafId`, `model`, `contextUsage`, `hasUI`, ...).
    ///
    /// # Errors
    /// Transport failure. Handler failures are reported in [`EmitResult::handlers`]/`errors`.
    pub async fn emit(&self, event: Value, ctx: Option<Value>) -> Result<EmitResult> {
        // `hasHandlers` gate (TS agent-session): when no loaded extension
        // subscribes to this event type, the host would run zero handlers and
        // return `{result: null, handlers: [], errors: []}` — the default
        // below. Skip the round-trip. `session_shutdown` always goes through:
        // the host clears extension timers even with no handlers, and a `ctx`
        // merge is a side effect we must not drop.
        if ctx.is_none() {
            let subscribed = self
                .subscribed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(subscribed) = subscribed.as_ref() {
                let ty = event.get("type").and_then(Value::as_str).unwrap_or("");
                if ty != "session_shutdown" && !subscribed.contains(ty) {
                    return Ok(EmitResult::default());
                }
            }
        }
        let mut p = json!({ "event": event });
        if let Some(ctx) = ctx {
            p["ctx"] = ctx;
        }
        Ok(serde_json::from_value(self.request("emit", p).await?)?)
    }

    /// # Errors
    /// Transport failure.
    pub async fn list_tools(&self) -> Result<Vec<ToolInfo>> {
        Ok(serde_json::from_value(
            self.request("list_tools", json!({})).await?,
        )?)
    }

    /// Execute an extension tool; returns its `{content, details, isError?}` result.
    ///
    /// # Errors
    /// Unknown tool, tool failure, or transport failure.
    pub async fn call_tool(
        &self,
        name: &str,
        tool_call_id: &str,
        params: Value,
        ctx: Option<Value>,
    ) -> Result<Value> {
        self.request(
            "call_tool",
            json!({"name": name, "toolCallId": tool_call_id, "params": params, "ctx": ctx}),
        )
        .await
    }

    /// Cancel a running extension tool call in the host (`AbortController`
    /// abort). Best-effort by contract: an unknown/expired id is not an error.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn cancel_tool(&self, tool_call_id: &str) -> Result<()> {
        self.request(
            "cancel_tool",
            json!({"toolCallId": tool_call_id}),
        )
        .await?;
        Ok(())
    }

    /// # Errors
    /// Transport failure.
    pub async fn list_commands(&self) -> Result<Vec<CommandInfo>> {
        Ok(serde_json::from_value(
            self.request("list_commands", json!({})).await?,
        )?)
    }

    /// # Errors
    /// Unknown command, command failure, or transport failure.
    pub async fn call_command(&self, name: &str, args: &str) -> Result<Value> {
        self.request("call_command", json!({"name": name, "args": args}))
            .await
    }

    /// Merge fields into the host's session state without emitting an event.
    ///
    /// # Errors
    /// Transport failure.
    pub async fn set_ctx(&self, ctx: Value) -> Result<()> {
        self.request("set_ctx", json!({ "ctx": ctx })).await?;
        Ok(())
    }

    /// Graceful exit: ask the host to exit, wait up to 2 s, then kill.
    ///
    /// # Errors
    /// Only on kill failure.
    pub async fn shutdown(&self) -> Result<()> {
        let _ =
            tokio::time::timeout(Duration::from_secs(2), self.request("shutdown", json!({}))).await;
        let mut child = self.child.lock().await;
        if let Some(child) = child.as_mut() {
            if tokio::time::timeout(Duration::from_secs(2), child.wait())
                .await
                .is_err()
            {
                child.kill().await?;
            }
        }
        // Shared mode: no child to reap — the daemon ends this session when the
        // socket closes (which happens when the writer drops); give the reader
        // a moment to observe the close so `pending` waiters drain cleanly.
        Ok(())
    }

    /// Connect to (starting on demand) the shared extension-host daemon on
    /// `sock`. One bun process serves every session; each connection is a bun
    /// Worker with a fresh module graph, so session state is as isolated as in
    /// a dedicated process. Dropping the last `Arc<ExtHost>` closes the
    /// socket; the daemon then terminates that session's Worker, and exits
    /// entirely once no session remains.
    ///
    /// # Errors
    /// No daemon accepted a connection within ~10 s.
    async fn spawn_shared(cfg: &HostConfig, sock: &Path) -> Result<Arc<Self>> {
        if cfg.bun.rsplit('/').next().is_some_and(|b| b != "bun" && !b.starts_with("bun-")) {
            bail!("shared ext host requires a real `bun` binary, got {}", cfg.bun);
        }
        let stream = Self::connect_daemon(cfg, sock).await?;

        let (read_half, write_half) = stream.into_split();
        let pending: Pending = Arc::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let host = Arc::new(Self {
            subscribed: Mutex::new(None),
            writer: tokio::sync::Mutex::new(Box::new(write_half)),
            child: tokio::sync::Mutex::new(None),
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            incoming: Mutex::new(Incoming {
                delivery: Delivery::Pending(VecDeque::new()),
                tx,
                rx: Some(rx),
            }),
            trace: cfg.trace.as_ref().map(|t| t.to_string_lossy().into_owned()),
            handler_timeout_ms: cfg.handler_timeout_ms,
        });

        let weak = Arc::downgrade(&host);
        tokio::spawn(async move {
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                route(&msg, &pending, weak.upgrade());
            }
            for (_, w) in pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain()
            {
                let _ = w.send(Err("extension host exited".into()));
            }
        });
        Ok(host)
    }

    /// Connect to the daemon's socket, spawning it on the first attempt.
    /// Another racing spawner may win the bind; our daemon then exits and we
    /// connect to the winner. A leftover socket file with no listener is
    /// removed once before retrying.
    async fn connect_daemon(cfg: &HostConfig, sock: &Path) -> Result<UnixStream> {
        let shard = cfg
            .host_script
            .parent()
            .context("shared ext host: host script has no parent dir")?
            .join("shard.ts");
        let mut spawned = false;
        let mut cleaned_stale = false;
        for attempt in 0..100 {
            match UnixStream::connect(sock).await {
                Ok(stream) => return Ok(stream),
                Err(_) => {
                    if !spawned {
                        Self::start_daemon(cfg, &shard, sock)?;
                        spawned = true;
                    } else if attempt >= 3 && !cleaned_stale && sock.exists() {
                        // Daemon is up but nothing listens: stale socket file
                        // (e.g. daemon crashed). Remove it and respawn once.
                        let _ = std::fs::remove_file(sock);
                        cleaned_stale = true;
                        spawned = false;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        bail!(
            "shared ext host: no daemon on {} after 10 s (spawned: {spawned})",
            sock.display()
        )
    }

    fn start_daemon(cfg: &HostConfig, shard: &std::path::Path, sock: &Path) -> Result<()> {
        let mut cmd = Command::new(&cfg.bun);
        if std::env::var_os("OMP_EXT_NO_SMOL").is_none() {
            cmd.arg("--smol");
        }
        let mut child = cmd
            .arg(shard)
            .env("OMP_RS_EXT_SOCKET", sock)
            .env("HOOK_TRACE", "0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(false)
            .spawn()
            .with_context(|| format!("spawn {} {}", cfg.bun, shard.display()))?;
        // The daemon logs to stderr; forward it like the per-session host's.
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    eprintln!("[ext-shard] {line}");
                }
            });
        }
        Ok(())
    }
}

fn route(msg: &Value, pending: &Pending, host: Option<Arc<ExtHost>>) {
    let method = msg.get("method").and_then(Value::as_str);
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    match (method, msg.get("id")) {
        (None, Some(id)) => {
            let Some(id) = id.as_u64() else { return };
            let Some(w) = pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id)
            else {
                return;
            };
            let _ = w.send(match msg.get("error") {
                Some(e) => Err(e
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
                    .to_owned()),
                None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
            });
        }
        (Some(m), None) => {
            let note = HostMessage::Notification {
                method: m.to_owned(),
                params,
            };
            let Some(host) = host else { return };
            let mut inc = host
                .incoming
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match &mut inc.delivery {
                Delivery::Pending(buf) => buf.push_back(note),
                Delivery::Live => {
                    let _ = inc.tx.send(note);
                }
            }
        }
        (Some(m), Some(id)) => {
            let req = HostMessage::Request {
                id: id.clone(),
                method: m.to_owned(),
                params,
            };
            let Some(host) = host else { return };
            // One lock covers check-and-deliver: a request can no longer slip
            // between "no consumer yet" and the channel send.
            let delivered = {
                let mut inc = host
                    .incoming
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match &mut inc.delivery {
                    Delivery::Pending(buf) => {
                        buf.push_back(req);
                        true
                    }
                    Delivery::Live => inc.tx.send(req).is_ok(),
                }
            };
            if !delivered {
                // The consumer went away: answer on its own task -- the reply
                // needs the stdin lock, and awaiting it here would stall the
                // stdout reader (a large write holding stdin while the host's
                // stdout pipe is full wedges both).
                let m = m.to_owned();
                let id = id.clone();
                tokio::spawn(async move {
                    let _ = host
                        .respond(id, Err(format!("{m} is not supported by the core")))
                        .await;
                });
            }
        }
        (None, None) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{serve, HeadlessResponder};

    fn pid_alive(pid: u32) -> bool {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }

    fn fake_host(dir: &std::path::Path, body: &str) -> HostConfig {
        let script = dir.join("fake-host.sh");
        std::fs::write(&script, body).unwrap();
        HostConfig {
            bun: "/bin/sh".into(),
            host_script: script,
            trace: None,
            handler_timeout_ms: None,
            env: Vec::new(),
            shared_socket: None,
        }
    }

    /// Once `serve()` is running, the loop must not keep the host alive:
    /// dropping the last `Arc<ExtHost>` handle kills the bun child
    /// (`kill_on_drop`). Previously `serve` held a strong reference and the
    /// process outlived every handle until the explicit close call.
    #[tokio::test]
    async fn host_process_exits_when_last_handle_drops() {
        let dir = tempfile::tempdir().unwrap();
        // `sleep 60` keeps the child alive unless it is killed on drop.
        let cfg = fake_host(dir.path(), "sleep 60\n");
        let host = ExtHost::spawn(&cfg).await.unwrap();
        let pid = host
            .child
            .lock()
            .await
            .as_ref()
            .and_then(|c| c.id())
            .expect("child pid");
        assert!(pid_alive(pid));
        let incoming = host.take_incoming().expect("incoming is available once");
        let _notes = serve(&host, incoming, Arc::new(HeadlessResponder));
        drop(host);
        // Tokio reaps the killed child asynchronously; allow generous time,
        // but far less than the 60 s the fake host would otherwise sleep.
        let mut exited = false;
        for _ in 0..200 {
            if !pid_alive(pid) {
                exited = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(exited, "bun host (pid {pid}) outlived its last handle");
    }

    /// A host request that arrives before any consumer attached is buffered,
    /// not rejected: once `take_incoming` + `serve` attach, the buffered
    /// request is answered (`ReviewWave3` -- previously the check-then-send
    /// race dropped it as "not supported").
    #[tokio::test]
    async fn early_request_is_buffered_until_consumer_attaches() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("replied");
        let script = dir.path().join("fake-host.sh");
        std::fs::write(
            &script,
            concat!(
                "printf '%s\n' '{\"jsonrpc\":\"2.0\",\"id\":101,\"method\":\"compact\",\"params\":{}}'\n",
                "while IFS= read -r line; do\n",
                "  case \"$line\" in\n",
                "    *'\"result\":null'*) touch \"$MARKER\"; break;;\n",
                "  esac\n",
                "done\n",
                "sleep 60\n",
            ),
        )
        .unwrap();
        let cfg = HostConfig {
            bun: "/bin/sh".into(),
            host_script: script,
            trace: None,
            handler_timeout_ms: None,
            env: vec![("MARKER".into(), marker.display().to_string())],
            shared_socket: None,
        };
        let host = ExtHost::spawn(&cfg).await.unwrap();
        // Give the host's request a moment to arrive while still unclaimed.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let incoming = host.take_incoming().expect("incoming is available once");
        let _notes = serve(&host, incoming, Arc::new(HeadlessResponder));
        let answered = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if marker.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await;
        drop(host);
        assert!(
            answered.is_ok(),
            "buffered request was never answered after attach"
        );
    }
}
