use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader};
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, watch};

/// Response waiters of one connection. `closed` is set when the connection
/// failed (reader EOF or writer error): every waiter is failed at that moment
/// and later registrations are refused, so no request can wait on a dead
/// connection (`ReviewExtDelta`: a request registered while the failing writer
/// drained used to hang forever).
#[derive(Default)]
struct PendingTable {
    waiters: HashMap<u64, oneshot::Sender<Result<Value, String>>>,
    closed: Option<&'static str>,
}

impl PendingTable {
    fn fail_all(&mut self, why: &'static str) {
        self.closed.get_or_insert(why);
        for (_, w) in self.waiters.drain() {
            let _ = w.send(Err(why.into()));
        }
    }
}

type Pending = Arc<Mutex<PendingTable>>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One transport generation: the writer queue plus the waiters of requests
/// sent on it. A reconnect replaces it wholesale, so a failing old
/// connection can only fail its own requests, never ones sent to the new host.
struct Conn {
    out: mpsc::UnboundedSender<Vec<u8>>,
    pending: Pending,
}

impl Conn {
    fn new(w: BoxedWriter) -> Self {
        let pending = Pending::default();
        Self {
            out: spawn_writer(w, pending.clone()),
            pending,
        }
    }
}

/// Transparent reconnects to the shared daemon per host before falling back
/// to a private stdio host (guards against a crash-looping daemon).
const MAX_SHARED_RECONNECTS: usize = 5;

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
/// Supplies the live ctx for an emit of the given event type (see [`ExtHost::set_ctx_provider`]).
pub type CtxProvider = std::sync::Arc<dyn Fn(&str) -> Option<Value> + Send + Sync>;

pub struct ExtHost {
    /// Event types with at least one handler across all loaded extensions,
    /// learned from `load` (mirrors the TS runner's `hasHandlers` gate).
    /// `None` = not loaded yet / unknown: send everything.
    subscribed: Mutex<Option<HashSet<String>>>,
    /// Live session state merged into the host ctx of every emit/tool call that carries no
    /// explicit `ctx` (branch, leafId, contextUsage, idle, pending, ...).
    ctx_provider: Mutex<Option<CtxProvider>>,
    /// Current connection: queue feeding the writer task (child stdin or
    /// shared-daemon socket) and the waiters of requests sent on it.
    conn: Mutex<Conn>,
    /// The spawned child; `None` in shared mode (the daemon outlives this
    /// session and is shared with other sessions, so we never kill it).
    child: tokio::sync::Mutex<Option<Child>>,
    next_id: AtomicU64,
    incoming: Mutex<Incoming>,
    /// Per-session trace file forwarded with the `load` RPC (shared mode).
    trace: Option<String>,
    /// Per-session handler budget forwarded with the `load` RPC (shared mode).
    handler_timeout_ms: Option<u64>,
    /// Config that created this host, kept for the shared-daemon-death
    /// fallback: respawns a private stdio host on the same `Arc<ExtHost>`.
    config: HostConfig,
    /// Bumped (with the new generation number) every time the underlying
    /// connection was transparently replaced (shared daemon death → private
    /// fallback). `watch` so late subscribers still see the latest generation.
    generation: watch::Sender<usize>,
    /// Set by [`Self::shutdown`]: the daemon closing our socket is then the
    /// expected end of the session, not a death to recover from.
    closing: std::sync::atomic::AtomicBool,
}

/// Writer task: coalesces every queued line into one buffered write and
/// flushes only when the queue drains. A write failure closes the queue
/// (later sends error), then fails every waiter of this connection and
/// refuses new ones.
fn spawn_writer(mut w: BoxedWriter, pending: Pending) -> mpsc::UnboundedSender<Vec<u8>> {
    const MAX_BATCH: usize = 256 * 1024;
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    tokio::spawn(async move {
        while let Some(mut buf) = rx.recv().await {
            while buf.len() < MAX_BATCH {
                match rx.try_recv() {
                    Ok(next) => buf.extend_from_slice(&next),
                    Err(_) => break,
                }
            }
            if w.write_all(&buf).await.is_err() || w.flush().await.is_err() {
                rx.close();
                lock(&pending).fail_all("extension host stdin closed");
                // Lines still queued are never written; their waiters were
                // failed above (registration happens before the enqueue).
                while rx.try_recv().is_ok() {}
                return;
            }
        }
    });
    tx
}

/// Object-safe writer half of `AsyncWriteExt` (child stdin or daemon socket).
type BoxedWriter = Box<dyn tokio::io::AsyncWrite + Send + Unpin>;

/// Build the per-session stdio host command (`bun ext-host/host.ts`). Shared
/// by [`ExtHost::spawn`] and the shared-daemon-death fallback, which respawns
/// the same private host on the existing `Arc<ExtHost>`.
/// `HOOK_TRACE` value for the host: `OMP_RS_HOOK_TRACE`, default off.
fn hook_trace_env() -> String {
    std::env::var("OMP_RS_HOOK_TRACE").unwrap_or_else(|_| "0".into())
}

fn private_command(cfg: &HostConfig) -> Command {
    let mut cmd = Command::new(&cfg.bun);
    // `--smol` trades more frequent GC for a smaller heap footprint
    // (~3-4 MB per session; measured 92.8 → 87.0 MB peak on the 36-extension
    // suite). Per-event cost impact measured at parity by bench s6.
    // Opt out with OMP_EXT_NO_SMOL=1 (e.g. older bun without the flag).
    // Only passed to a real `bun` binary — tests substitute fake runners.
    if std::env::var_os("OMP_EXT_NO_SMOL").is_none()
        && cfg
            .bun
            .rsplit('/')
            .next()
            .is_some_and(|b| b == "bun" || b.starts_with("bun-"))
    {
        cmd.arg("--smol");
    }
    // `OMP_RS_HOOK_TRACE=1` opts back in (extension conformance harness diffs the trace against omp).
    let hook_trace = hook_trace_env();
    cmd.arg(&cfg.host_script)
        .env("HOOK_TRACE", hook_trace)
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
    cmd
}

/// Reader task of one connection: routes every line, then fails the
/// connection's remaining waiters at EOF.
fn spawn_reader(
    reader: impl AsyncRead + Send + Unpin + 'static,
    pending: Pending,
    host: Weak<ExtHost>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            route(&msg, &pending, host.upgrade());
        }
        lock(&pending).fail_all("extension host exited");
    })
}

fn spawn_stderr(stderr: tokio::process::ChildStderr) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            eprintln!("[ext-host] {line}");
        }
    });
}

impl ExtHost {
    /// Spawn `bun <host_script>` (or connect to the shared daemon when
    /// [`HostConfig::shared_socket`] is set) and start the reader tasks.
    ///
    /// # Errors
    /// Fails when the process cannot be spawned or the shared daemon does not
    /// accept a connection within ~10 s.
    pub async fn spawn(cfg: &HostConfig) -> Result<Arc<Self>> {
        if let Some(sock) = &cfg.shared_socket {
            #[cfg(unix)]
            return Self::spawn_shared(cfg, sock).await;
            #[cfg(not(unix))]
            {
                let _ = sock;
                bail!("shared ext host requires unix sockets; unset OMP_RS_EXT_SHARED on this platform");
            }
        }
        let mut child = private_command(cfg)
            .spawn()
            .with_context(|| format!("spawn {} {}", cfg.bun, cfg.host_script.display()))?;
        let stdin = child.stdin.take().context("host stdin")?;
        let stdout = child.stdout.take().context("host stdout")?;
        let stderr = child.stderr.take().context("host stderr")?;

        let conn = Conn::new(Box::new(stdin));
        let pending = conn.pending.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        let (generation, _) = watch::channel(0usize);
        let host = Arc::new(Self {
            subscribed: Mutex::new(None),
            ctx_provider: Mutex::new(None),
            conn: Mutex::new(conn),
            child: tokio::sync::Mutex::new(Some(child)),
            next_id: AtomicU64::new(1),
            incoming: Mutex::new(Incoming {
                delivery: Delivery::Pending(VecDeque::new()),
                tx,
                rx: Some(rx),
            }),
            trace: cfg.trace.as_ref().map(|t| t.to_string_lossy().into_owned()),
            handler_timeout_ms: cfg.handler_timeout_ms,
            config: cfg.clone(),
            generation,
            closing: std::sync::atomic::AtomicBool::new(false),
        });

        // Host gone: the reader fails everything still waiting.
        spawn_reader(stdout, pending, Arc::downgrade(&host));
        spawn_stderr(stderr);
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
        self.write(&msg)
    }

    /// Enqueue one line for the writer task (synchronous: preserves call order).
    fn write(&self, msg: &Value) -> Result<()> {
        let mut line = serde_json::to_vec(msg)?;
        line.push(b'\n');
        lock(&self.conn)
            .out
            .send(line)
            .map_err(|_| anyhow!("extension host stdin closed"))
    }

    /// Register a pending response and enqueue the request; the receiver
    /// resolves with the response.
    fn send_request(
        &self,
        method: &str,
        params: Value,
    ) -> Result<oneshot::Receiver<Result<Value, String>>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut line =
            serde_json::to_vec(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        line.push(b'\n');
        let (tx, rx) = oneshot::channel();
        // Register and enqueue on the same connection (held across both, so a
        // reconnect cannot split them): the waiter belongs to the connection
        // that carries the request, and only that connection can fail it.
        let conn = lock(&self.conn);
        {
            let mut table = lock(&conn.pending);
            if let Some(why) = table.closed {
                bail!("{why}");
            }
            table.waiters.insert(id, tx);
        }
        if conn.out.send(line).is_err() {
            lock(&conn.pending).waiters.remove(&id);
            bail!("extension host stdin closed");
        }
        Ok(rx)
    }

    /// Raw JSON-RPC request.
    ///
    /// # Errors
    /// Transport failure, host exit, or a JSON-RPC error response.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let rx = self.send_request(method, params)?;
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

    /// Install the live-ctx provider: its value rides every emit / tool call that has no explicit
    /// `ctx`, so handlers see the TS-live `ctx` (branch, context usage, idle state, ...). It never
    /// forces an emit that `has_handlers` would skip.
    pub fn set_ctx_provider(&self, f: Option<CtxProvider>) {
        *self
            .ctx_provider
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = f;
    }

    fn provided_ctx(&self, ctx: Option<Value>, event_type: &str) -> Option<Value> {
        if ctx.is_some() {
            return ctx;
        }
        let f = self
            .ctx_provider
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        f.and_then(|f| f(event_type))
    }

    /// Emit an omp event (`{"type": ..., ...}`); `ctx` merges into the host's session state
    /// (`sessionId`, `sessionFile`, `branch`, `leafId`, `model`, `contextUsage`, `hasUI`, ...).
    ///
    /// # Errors
    /// Transport failure. Handler failures are reported in [`EmitResult::handlers`]/`errors`.
    pub async fn emit(&self, event: Value, ctx: Option<Value>) -> Result<EmitResult> {
        self.emit_skipping(event, ctx, &[]).await
    }

    /// [`emit`](Self::emit) whose handlers of the extensions named in `skip`
    /// (host `ext.name`: file stem, or the directory of an `index.ts`) do not
    /// run for this one event (`skipExtensions`; omitted when empty).
    ///
    /// # Errors
    /// Transport failure.
    pub async fn emit_skipping(
        &self,
        event: Value,
        ctx: Option<Value>,
        skip: &[String],
    ) -> Result<EmitResult> {
        if self.skip_emit(&event, ctx.is_some()) {
            return Ok(EmitResult::default());
        }
        let ctx = self.provided_ctx(ctx, event.get("type").and_then(Value::as_str).unwrap_or(""));
        let mut p = emit_payload(&event, skip);
        if let Some(ctx) = ctx {
            p["ctx"] = ctx;
        }
        Ok(serde_json::from_value(self.request("emit", p).await?)?)
    }

    /// `true` unless the host is loaded and no extension subscribes to `event_type`
    /// (the TS runner's `hasHandlers` gate). Callers use it to skip building
    /// expensive payloads that `emit` would discard anyway.
    #[must_use]
    pub fn has_handlers(&self, event_type: &str) -> bool {
        let subscribed = self
            .subscribed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match subscribed.as_ref() {
            Some(s) => event_type == "session_shutdown" || s.contains(event_type),
            None => true,
        }
    }

    /// `hasHandlers` gate (TS agent-session): when no loaded extension
    /// subscribes to this event type, the host would run zero handlers and
    /// return `{result: null, handlers: [], errors: []}` — the default. Skip
    /// the round-trip. `session_shutdown` always goes through: the host clears
    /// extension timers even with no handlers, and a `ctx` merge is a side
    /// effect we must not drop.
    fn skip_emit(&self, event: &Value, has_ctx: bool) -> bool {
        !has_ctx && !self.has_handlers(event.get("type").and_then(Value::as_str).unwrap_or(""))
    }

    /// Fire-and-forget [`emit`](Self::emit) for notification-only events whose
    /// handler results cannot change agent behavior. The request is enqueued
    /// synchronously (wire order == call order) and the host runs these emits
    /// strictly in order, ahead of any later awaited `emit`/`call_tool`.
    /// `on_done` receives the outcome (handler errors, transport failure).
    ///
    /// # Errors
    /// Fails only when the request cannot be enqueued (host stdin closed).
    pub fn emit_nowait(
        &self,
        event: &Value,
        ctx: Option<Value>,
        on_done: impl FnOnce(Result<EmitResult>) + Send + 'static,
    ) -> Result<()> {
        if self.skip_emit(event, ctx.is_some()) {
            return Ok(());
        }
        let mut p = json!({ "event": event, "nowait": true });
        let ctx = self.provided_ctx(ctx, event.get("type").and_then(Value::as_str).unwrap_or(""));
        if let Some(ctx) = ctx {
            p["ctx"] = ctx;
        }
        let rx = self.send_request("emit", p)?;
        tokio::spawn(async move {
            let r = match rx.await {
                Ok(Ok(v)) => serde_json::from_value(v).map_err(Into::into),
                Ok(Err(m)) => Err(anyhow!("emit: {m}")),
                Err(_) => Err(anyhow!("emit: extension host dropped the request")),
            };
            on_done(r);
        });
        Ok(())
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
        let ctx = self.provided_ctx(ctx, "call_tool");
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
        self.request("cancel_tool", json!({"toolCallId": tool_call_id}))
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
        self.closing.store(true, Ordering::SeqCst);
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
    #[cfg(unix)]
    async fn spawn_shared(cfg: &HostConfig, sock: &Path) -> Result<Arc<Self>> {
        if cfg
            .bun
            .rsplit('/')
            .next()
            .is_some_and(|b| b != "bun" && !b.starts_with("bun-"))
        {
            bail!(
                "shared ext host requires a real `bun` binary, got {}",
                cfg.bun
            );
        }
        let stream = Self::connect_daemon(cfg, sock).await?;

        let (read_half, write_half) = stream.into_split();
        let conn = Conn::new(Box::new(write_half));
        let pending = conn.pending.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        let (generation, _) = watch::channel(0usize);
        let host = Arc::new(Self {
            subscribed: Mutex::new(None),
            ctx_provider: Mutex::new(None),
            conn: Mutex::new(conn),
            child: tokio::sync::Mutex::new(None),
            next_id: AtomicU64::new(1),
            incoming: Mutex::new(Incoming {
                delivery: Delivery::Pending(VecDeque::new()),
                tx,
                rx: Some(rx),
            }),
            trace: cfg.trace.as_ref().map(|t| t.to_string_lossy().into_owned()),
            handler_timeout_ms: cfg.handler_timeout_ms,
            config: cfg.clone(),
            generation,
            closing: std::sync::atomic::AtomicBool::new(false),
        });

        let reader = spawn_reader(read_half, pending, Arc::downgrade(&host));
        Self::watch_daemon(&host, reader);
        Ok(host)
    }

    /// Subscribe to connection replacements (shared-daemon death → private
    /// fallback). The receiver yields the new generation each time the
    /// underlying connection was transparently re-established; `changed()`
    /// fires immediately if a reconnect already happened.
    #[must_use]
    pub fn reconnections(&self) -> watch::Receiver<usize> {
        self.generation.subscribe()
    }

    /// Current connection generation (0 until the first transparent
    /// reconnect). Host-side session state does not survive a generation
    /// change.
    #[must_use]
    pub fn generation(&self) -> usize {
        *self.generation.borrow()
    }

    /// `true` when this host may transparently reconnect (shared mode).
    /// Private hosts die with their session and never reconnect.
    #[must_use]
    pub fn may_reconnect(&self) -> bool {
        self.config.shared_socket.is_some()
    }

    /// Daemon died (e.g. kill -9): the reader fails everything still waiting
    /// so in-flight emits/tool calls get a clean error, not a hang. Then
    /// recover transparently on the same `Arc<ExtHost>` so every consumer
    /// (hooks, tools, lifecycle forwarding) keeps working; `reconnections()`
    /// subscribers (the omp-cli supervisor) re-load extensions and re-emit
    /// session ctx.
    fn watch_daemon(host: &Arc<Self>, reader: tokio::task::JoinHandle<()>) {
        let weak = Arc::downgrade(host);
        tokio::spawn(async move {
            let _ = reader.await;
            if let Some(host) = weak.upgrade().filter(|h| !h.closing.load(Ordering::SeqCst)) {
                match host.reconnect_after_daemon_death().await {
                    Ok(true) => eprintln!(
                        "omt: shared extension-host daemon lost; reconnected to a fresh daemon"
                    ),
                    Ok(false) => eprintln!(
                        "omt: shared extension-host daemon lost; fell back to a private host"
                    ),
                    Err(e) => eprintln!("omt: shared extension-host daemon lost: {e:#}"),
                }
            }
        });
    }

    /// Re-establish the connection after the shared daemon's connection
    /// ended. First tries the shared daemon again (restarting it under the
    /// start lock if nobody else did; at most [`MAX_SHARED_RECONNECTS`] times
    /// per host so a crash-looping daemon cannot spin), else spawns a private
    /// stdio host. Swaps the writer/child, resets the `hasHandlers` gate and
    /// the incoming channel (the old consumer's receiver ends when its sender
    /// drops), restarts the reader task, and bumps the generation so
    /// supervisors can re-load extensions and re-send the session ctx.
    ///
    /// Returns `true` when reconnected to the shared daemon, `false` for the
    /// private fallback.
    ///
    /// # Errors
    /// Spawn failure; the host stays unusable (writes fail cleanly).
    async fn reconnect_after_daemon_death(self: &Arc<Self>) -> Result<bool> {
        if let Some(sock) = &self.config.shared_socket {
            #[cfg(unix)]
            if self.generation() < MAX_SHARED_RECONNECTS {
                match Self::connect_daemon(&self.config, sock).await {
                    Ok(stream) => {
                        let (r, w) = stream.into_split();
                        let reader = self.install(Box::new(w), r, None).await;
                        Self::watch_daemon(self, reader);
                        self.generation.send_modify(|g| *g += 1);
                        return Ok(true);
                    }
                    Err(e) => eprintln!("omt: shared extension-host daemon unavailable: {e:#}"),
                }
            }
            #[cfg(not(unix))]
            {
                let _ = sock;
            }
        }
        let mut child = private_command(&self.config).spawn().with_context(|| {
            format!(
                "reconnect: spawn {} {}",
                self.config.bun,
                self.config.host_script.display()
            )
        })?;
        let stdin = child.stdin.take().context("reconnect: host stdin")?;
        let stdout = child.stdout.take().context("reconnect: host stdout")?;
        let stderr = child.stderr.take().context("reconnect: host stderr")?;
        self.install(Box::new(stdin), stdout, Some(child)).await;
        spawn_stderr(stderr);
        // Publish last, after the new connection is fully usable.
        self.generation.send_modify(|g| *g += 1);
        Ok(false)
    }

    /// Swap in a fresh connection: new writer and waiter table (the old
    /// connection keeps its own, so its late failure cannot touch requests
    /// sent here), child, reset `hasHandlers` gate and incoming channel (the
    /// old consumer's receiver ends when its sender drops); start its reader.
    async fn install(
        self: &Arc<Self>,
        writer: BoxedWriter,
        reader: impl AsyncRead + Send + Unpin + 'static,
        child: Option<Child>,
    ) -> tokio::task::JoinHandle<()> {
        let conn = Conn::new(writer);
        let pending = conn.pending.clone();
        *lock(&self.conn) = conn;
        *self.child.lock().await = child;
        *lock(&self.subscribed) = None;
        let (tx, rx) = mpsc::unbounded_channel();
        *lock(&self.incoming) = Incoming {
            delivery: Delivery::Pending(VecDeque::new()),
            tx,
            rx: Some(rx),
        };
        spawn_reader(reader, pending, Arc::downgrade(self))
    }

    /// Connect to the daemon's socket, starting it when nothing listens.
    /// Starting is serialised across processes by an exclusive `flock` on
    /// `<sock>.lock`, held until the new daemon accepts connections, so
    /// concurrent first clients never race to bind. Under the lock a leftover
    /// socket file with no listener (crashed daemon) is removed before
    /// spawning.
    #[cfg(unix)]
    async fn connect_daemon(cfg: &HostConfig, sock: &Path) -> Result<UnixStream> {
        let mut stream = Self::connect_daemon_raw(cfg, sock).await?;
        // Handshake: the daemon's own env belongs to whichever session started
        // it, so the session's real environment is sent as the first line and
        // becomes the `process.env` of this session's Worker.
        let mut env: serde_json::Map<String, Value> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, json!(v.into_string().ok()?))))
            .collect();
        env.insert("HOOK_TRACE".into(), json!(hook_trace_env()));
        if let Some(t) = &cfg.trace {
            env.insert("OMP_EXT_TRACE".into(), json!(t));
        }
        if let Some(ms) = cfg.handler_timeout_ms {
            env.insert("OMP_EXT_TIMEOUT_MS".into(), json!(ms.to_string()));
        }
        for (k, v) in &cfg.env {
            env.insert(k.clone(), json!(v));
        }
        let line = format!(
            "{}\n",
            json!({"jsonrpc": "2.0", "method": "omp-rs-env", "params": {"env": env}})
        );
        stream
            .write_all(line.as_bytes())
            .await
            .context("shared ext host: env handshake")?;
        Ok(stream)
    }

    #[cfg(unix)]
    async fn connect_daemon_raw(cfg: &HostConfig, sock: &Path) -> Result<UnixStream> {
        if let Ok(stream) = UnixStream::connect(sock).await {
            return Ok(stream);
        }
        let shard = cfg
            .host_script
            .parent()
            .context("shared ext host: host script has no parent dir")?
            .join("shard.ts");
        if let Some(dir) = sock.parent() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                std::fs::DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(dir)
                    .with_context(|| format!("create {}", dir.display()))?;
            }
            #[cfg(not(unix))]
            std::fs::DirBuilder::new()
                .recursive(true)
                .create(dir)
                .with_context(|| format!("create {}", dir.display()))?;
        }
        let mut lock_path = sock.as_os_str().to_owned();
        lock_path.push(".lock");
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("open {}", Path::new(&lock_path).display()))?;
        let mut waited = 0;
        loop {
            match lock_file.try_lock() {
                Ok(()) => break,
                Err(std::fs::TryLockError::WouldBlock) if waited < 300 => {
                    waited += 1;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => bail!(
                    "shared ext host: start lock {}: {e}",
                    Path::new(&lock_path).display()
                ),
            }
        }
        // Another client may have started it while we waited for the lock.
        if let Ok(stream) = UnixStream::connect(sock).await {
            return Ok(stream);
        }
        let _ = std::fs::remove_file(sock);
        Self::start_daemon(cfg, &shard, sock)?;
        for _ in 0..100 {
            if let Ok(stream) = UnixStream::connect(sock).await {
                return Ok(stream); // lock released on drop
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        bail!(
            "shared ext host: no daemon on {} after 10 s",
            sock.display()
        )
    }

    fn start_daemon(cfg: &HostConfig, shard: &std::path::Path, sock: &Path) -> Result<()> {
        // The cleared env below has no PATH, so resolve `bun` here.
        let bun = if cfg.bun.contains('/') {
            PathBuf::from(&cfg.bun)
        } else {
            std::env::var_os("PATH")
                .and_then(|p| {
                    std::env::split_paths(&p)
                        .map(|d| d.join(&cfg.bun))
                        .find(|c| c.is_file())
                })
                .unwrap_or_else(|| PathBuf::from(&cfg.bun))
        };
        let mut cmd = Command::new(bun);
        if std::env::var_os("OMP_EXT_NO_SMOL").is_none() {
            cmd.arg("--smol");
        }
        // Clean daemon env: a bun Worker cannot override env keys that exist
        // in the daemon process, so the daemon must hold none of the starting
        // session's variables (each session's env arrives in its handshake).
        // Only bun's own `BUN_*` knobs and the idle timeout carry over.
        cmd.env_clear();
        for (k, v) in std::env::vars_os() {
            let keep = k
                .to_str()
                .is_some_and(|k| k.starts_with("BUN_") || k == "OMP_RS_EXT_SHARED_IDLE_MS");
            if keep {
                cmd.env(k, v);
            }
        }
        let mut child = cmd
            .arg(shard)
            .env("OMP_RS_EXT_SOCKET", sock)
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
            let Some(w) = lock(pending).waiters.remove(&id) else {
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

/// `emit` request params: `{event, skipExtensions?}`.
fn emit_payload(event: &Value, skip: &[String]) -> Value {
    let mut p = json!({ "event": event });
    if !skip.is_empty() {
        p["skipExtensions"] = json!(skip);
    }
    p
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
            .and_then(tokio::process::Child::id)
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

    /// The host closes its stdin but stays alive (stdout open), so only the
    /// writer can notice the dead connection. Every request -- including ones
    /// registered while the failing writer drains -- must resolve, none hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn writer_failure_resolves_every_pending_request() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = fake_host(dir.path(), "exec 0<&-\nsleep 60\n");
        for _ in 0..10 {
            let host = ExtHost::spawn(&cfg).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            let mut tasks = tokio::task::JoinSet::new();
            for i in 0..200 {
                let host = host.clone();
                tasks.spawn(async move {
                    tokio::time::timeout(
                        Duration::from_secs(5),
                        host.request("emit", json!({ "i": i })),
                    )
                    .await
                });
            }
            while let Some(r) = tasks.join_next().await {
                let r = r.unwrap().expect("request hung after the writer failed");
                assert!(r.is_err(), "no host answered, yet: {r:?}");
            }
        }
    }

    type SentLines = tokio::io::Lines<BufReader<tokio::io::DuplexStream>>;

    /// Line-oriented fake host on in-memory pipes: `(writer, reader)` for
    /// [`ExtHost::install`] and the peer ends the test drives.
    fn pipe_conn() -> (
        (BoxedWriter, tokio::io::DuplexStream),
        (SentLines, tokio::io::DuplexStream),
    ) {
        let (to_host, from_core) = tokio::io::duplex(1 << 16);
        let (from_host, to_core) = tokio::io::duplex(1 << 16);
        (
            (Box::new(to_host), to_core),
            (BufReader::new(from_core).lines(), from_host),
        )
    }

    /// A connection that failed after it was replaced fails only its own
    /// requests: a request already sent to the new connection still gets
    /// the new host's answer (`ReviewExtDelta`: the old writer drained the
    /// shared waiter table, failing it with "stdin closed").
    #[tokio::test]
    async fn late_failure_of_a_replaced_connection_spares_the_new_one() {
        let dir = tempfile::tempdir().unwrap();
        let host = ExtHost::spawn(&fake_host(dir.path(), "sleep 60\n"))
            .await
            .unwrap();

        // Old connection: its writer blocks on a full 16-byte pipe until the
        // peer goes away, then fails (a dead daemon socket / stdin).
        let (old_w, old_peer) = tokio::io::duplex(16);
        let (old_r, _old_r_peer) = tokio::io::duplex(16);
        host.install(Box::new(old_w), old_r, None).await;
        let old = host
            .send_request("emit", json!({"queued": "on the old connection"}))
            .unwrap();

        let ((new_w, new_r), (mut sent, mut answer)) = pipe_conn();
        host.install(new_w, new_r, None).await;
        let new = host.send_request("emit", json!({})).unwrap();
        let line = tokio::time::timeout(Duration::from_secs(5), sent.next_line())
            .await
            .unwrap()
            .unwrap()
            .expect("request reached the new host");
        let id = serde_json::from_str::<Value>(&line).unwrap()["id"].clone();

        drop(old_peer);
        let old = tokio::time::timeout(Duration::from_secs(5), old)
            .await
            .unwrap()
            .unwrap();
        assert!(
            old.is_err(),
            "old request fails with its connection: {old:?}"
        );

        let mut reply =
            serde_json::to_vec(&json!({"jsonrpc":"2.0","id":id,"result":"ok"})).unwrap();
        reply.push(b'\n');
        answer.write_all(&reply).await.unwrap();
        let new = tokio::time::timeout(Duration::from_secs(5), new)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            new,
            Ok(json!("ok")),
            "new request must get the new host's answer"
        );
    }

    /// Writer failure: every queued request resolves, and a request issued
    /// once the connection failed is refused at once instead of waiting on a
    /// connection that will never answer.
    #[tokio::test]
    async fn failed_writer_resolves_queued_and_refuses_new_requests() {
        let dir = tempfile::tempdir().unwrap();
        let host = ExtHost::spawn(&fake_host(dir.path(), "sleep 60\n"))
            .await
            .unwrap();
        let (w, peer) = tokio::io::duplex(16);
        let (r, _r_peer) = tokio::io::duplex(16);
        host.install(Box::new(w), r, None).await;
        let queued: Vec<_> = (0..8)
            .map(|i| host.send_request("emit", json!({ "i": i })).unwrap())
            .collect();
        drop(peer);
        for rx in queued {
            let r = tokio::time::timeout(Duration::from_secs(5), rx)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(r, Err("extension host stdin closed".into()));
        }
        // The table is closed with the failure: no window where a new waiter
        // can be registered on the dead connection.
        assert!(lock(&lock(&host.conn).pending).closed.is_some());
        assert!(host.send_request("emit", json!({})).is_err());
    }
}
