//! JSON-RPC stdio LSP client and process registry (omp `lsp/client.ts`).
//!
//! One client per (command, cwd, config-identity) key. Initialization runs
//! under a per-key lock so concurrent callers share one in-flight handshake;
//! deterministic init failures are negative-cached for three minutes. Clients
//! are reaped by a background idle checker when a positive `idleTimeoutMs`
//! is configured globally or in the workspace config.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, watch, Mutex as AsyncMutex};
use tracing::warn;

use crate::types::{LspConfig, PublishDiagnosticsParams, ServerConfig};
use crate::utils::{file_to_uri, uri_key};
use crate::Cancel;

/// Default warmup timeout (omp `WARMUP_TIMEOUT_MS`).
pub const WARMUP_TIMEOUT_MS: u64 = 5000;
const PROJECT_LOAD_TIMEOUT_MS: u64 = 15_000;
const SHUTDOWN_TIMEOUT_MS: u64 = 5000;
const EXIT_TIMEOUT_MS: u64 = 1000;
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 30_000;
const INIT_FAILURE_BACKOFF_MS: u64 = 3 * 60 * 1000;
const IDLE_CHECK_INTERVAL_MS: u64 = 60_000;
const RUST_ANALYZER_WORKSPACE_READY_TIMEOUT_MS: u64 = 5000;
const RUST_ANALYZER_WORKSPACE_READY_POLL_MS: u64 = 100;
const RUST_ANALYZER_WORKSPACE_READY_SETTLE_MS: u64 = 2000;
const RUST_ANALYZER_STATUS_REQUEST_TIMEOUT_MS: u64 = 1000;

/// Client lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientStatus {
    Connecting,
    Ready,
    Error,
}

/// An open document tracked on the server.
#[derive(Debug, Clone)]
pub struct OpenFile {
    pub version: i64,
    pub language_id: String,
    pub synced_text: String,
}

/// Error surfaced to requesters (timeouts, JSON-RPC errors, dead servers).
#[derive(Debug, Clone)]
pub struct LspError {
    pub message: String,
    pub cancelled: bool,
}

impl std::fmt::Display for LspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LspError {}

struct PendingEntry {
    tx: oneshot::Sender<Result<Value, LspError>>,
}

/// Live language-server client. Cheap to clone (all state is shared).
#[derive(Clone)]
pub struct LspClient {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    name: String,
    cwd: PathBuf,
    config: ServerConfig,
    request_id: AtomicI64,
    status: Mutex<ClientStatus>,
    open_files: Mutex<HashMap<PathBuf, OpenFile>>,
    /// Keyed by canonical path (`EquivalentUriMap` in TS).
    diagnostics: Mutex<HashMap<PathBuf, crate::types::PublishedDiagnostics>>,
    diagnostics_version: AtomicU64,
    pending: Mutex<HashMap<i64, PendingEntry>>,
    server_capabilities: Mutex<Value>,
    last_activity: AtomicI64,
    started_at: i64,
    /// Server stdin: the spawned child's pipe, or a broker-shared mux socket
    /// driven exactly like one (omp `LspTransport` union).
    stdin: Arc<AsyncMutex<Option<Box<dyn tokio::io::AsyncWrite + Unpin + Send>>>>,
    child: Mutex<Option<Child>>,
    /// This link is broker-shared (`sharedMux` in TS): there is no local
    /// child, teardown detaches the link and the mux keeps the server warm.
    shared_mux: bool,
    mux_pid: Option<u32>,
    /// Signalled when the reader task observes the transport going away.
    transport_dead: Arc<tokio::sync::Notify>,
    stderr: Arc<Mutex<Vec<u8>>>,
    dynamic_registrations: Mutex<HashMap<String, String>>,
    /// Resolves to `true` when the initial project load completes (or times out).
    project_loaded_tx: watch::Sender<bool>,
    /// Keeps the watch channel alive so project-load signals are not lost.
    _project_loaded_rx: watch::Receiver<bool>,
    rust_analyzer_ready: Mutex<bool>,
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(0))
}

impl LspClient {
    /// Registry key: command + cwd + config identity (omp `clientKey`).
    fn name_for(config: &ServerConfig, cwd: &Path) -> String {
        let spawn = config
            .resolved_command
            .as_deref()
            .unwrap_or(&config.command);
        let identity = json!([
            config.args.clone().unwrap_or_default(),
            config.init_options.clone().unwrap_or(Value::Null),
            config.settings.clone().unwrap_or(Value::Null),
            config.language_id.clone(),
        ]);
        format!(
            "{spawn}:{}:{identity}",
            crate::utils::normalize_path(cwd).display()
        )
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }
    #[must_use]
    pub fn cwd(&self) -> &Path {
        &self.inner.cwd
    }
    #[must_use]
    pub fn config(&self) -> &ServerConfig {
        &self.inner.config
    }
    #[must_use]
    pub fn status(&self) -> ClientStatus {
        *self
            .inner
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    #[must_use]
    pub fn server_capabilities(&self) -> Value {
        self.inner
            .server_capabilities
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    #[must_use]
    pub fn diagnostics_version(&self) -> u64 {
        self.inner.diagnostics_version.load(Ordering::SeqCst)
    }
    #[must_use]
    pub fn open_file_version(&self, uri: &str) -> Option<i64> {
        self.inner
            .open_files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&uri_key(uri))
            .map(|f| f.version)
    }
    /// Latest published diagnostics for `uri`, keyed canonically.
    #[must_use]
    pub fn published_diagnostics(&self, uri: &str) -> Option<crate::types::PublishedDiagnostics> {
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&uri_key(uri))
            .cloned()
    }
    /// Whether `uri` is currently open on the server.
    #[must_use]
    pub fn is_open(&self, uri: &str) -> bool {
        self.inner
            .open_files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&uri_key(uri))
    }
    #[must_use]
    pub fn is_rust_analyzer(&self) -> bool {
        fn basename(cmd: &str) -> &str {
            cmd.rsplit(['/', '\\']).next().unwrap_or(cmd)
        }
        basename(&self.inner.config.command) == "rust-analyzer"
            || self
                .inner
                .config
                .resolved_command
                .as_deref()
                .is_some_and(|c| basename(c) == "rust-analyzer")
    }
    #[must_use]
    pub fn started_at(&self) -> i64 {
        self.inner.started_at
    }
    /// Whether this link is broker-shared (`sharedMux` in TS): teardown only
    /// detaches, and a wedged reload must ask the mux to kill the server
    /// (`omp/muxRestartServer`).
    #[must_use]
    pub fn is_shared_mux(&self) -> bool {
        self.inner.shared_mux
    }
    /// Pid of the shared server process behind a mux link, when known.
    #[must_use]
    pub fn mux_server_pid(&self) -> Option<u32> {
        self.inner.mux_pid
    }
    /// Whether the server advertised LSP 3.17 pull diagnostics.
    #[must_use]
    pub fn supports_document_diagnostics(&self) -> bool {
        let caps = self.server_capabilities();
        if caps.get("diagnosticProvider").is_some() {
            return true;
        }
        self.inner
            .dynamic_registrations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .any(|m| m == "textDocument/diagnostic")
    }
}

impl LspClient {
    /// Write one framed JSON-RPC message to the server's stdin.
    async fn write_message(&self, message: &Value) -> Result<(), LspError> {
        let body = serde_json::to_string(message).map_err(|e| LspError {
            message: e.to_string(),
            cancelled: false,
        })?;
        let frame = format!("Content-Length: {}\r\n\r\n{body}", body.len());
        let mut guard = self.inner.stdin.lock().await;
        let Some(stdin) = guard.as_mut() else {
            return Err(LspError {
                message: "LSP server stdin closed".to_string(),
                cancelled: false,
            });
        };
        stdin
            .write_all(frame.as_bytes())
            .await
            .map_err(|e| LspError {
                message: format!("LSP write failed: {e}"),
                cancelled: false,
            })?;
        stdin.flush().await.map_err(|e| LspError {
            message: format!("LSP flush failed: {e}"),
            cancelled: false,
        })?;
        Ok(())
    }

    /// Send a request and await the response (omp `sendRequest`).
    ///
    /// Cancellation policy mirrors TS: with an explicit timeout or a caller
    /// cancel handle, no internal timer is installed; otherwise a 30s default
    /// bounds the wait.
    ///
    /// # Errors
    ///
    /// `LspError` on JSON-RPC errors, timeouts, connection loss, or cancellation.
    pub async fn send_request(
        &self,
        method: &str,
        params: Value,
        cancel: Option<&dyn Cancel>,
        timeout: Option<Duration>,
    ) -> Result<Value, LspError> {
        let id = self.inner.request_id.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(c) = cancel {
            if c.is_cancelled() {
                return Err(LspError {
                    message: "operation cancelled".to_string(),
                    cancelled: true,
                });
            }
        }
        let (tx, rx) = oneshot::channel::<Result<Value, LspError>>();
        self.inner
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, PendingEntry { tx });
        self.inner
            .last_activity
            .store(now_millis(), Ordering::SeqCst);

        let result = self
            .write_message(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await;
        if let Err(e) = result {
            self.inner
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            return Err(e);
        }

        let default_timeout = if cancel.is_none() && timeout.is_none() {
            Some(Duration::from_millis(DEFAULT_REQUEST_TIMEOUT_MS))
        } else {
            timeout
        };

        let wait = async {
            tokio::select! {
                res = rx => res.unwrap_or_else(|_| {
                    Err(LspError { message: "LSP request dropped".to_string(), cancelled: false })
                }),
                () = async { match cancel {
                    Some(c) => c.cancelled().await,
                    None => std::future::pending().await,
                } } => {
                    self.inner.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&id);
                    let _ = self
                        .write_message(&json!({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": id}}))
                        .await;
                    Err(LspError { message: "operation cancelled".to_string(), cancelled: true })
                }
            }
        };
        let outcome = match default_timeout {
            Some(t) => match tokio::time::timeout(t, wait).await {
                Ok(r) => r,
                Err(_elapsed) => {
                    self.inner
                        .pending
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .remove(&id);
                    let _ = self
                        .write_message(&json!({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": id}}))
                        .await;
                    Err(LspError {
                        message: format!(
                            "LSP request {method} timed out after {}ms",
                            t.as_millis()
                        ),
                        cancelled: false,
                    })
                }
            },
            None => wait.await,
        };
        self.inner
            .last_activity
            .store(now_millis(), Ordering::SeqCst);
        outcome
    }

    /// Send a notification (fire-and-forget; only write errors surface).
    ///
    /// # Errors
    ///
    /// Returns `Err` when the frame cannot be written to the server stdin.
    pub async fn send_notification(&self, method: &str, params: Value) -> Result<(), LspError> {
        self.inner
            .last_activity
            .store(now_millis(), Ordering::SeqCst);
        self.write_message(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .await
    }

    fn resolve_pending(&self, id: i64, result: Result<Value, LspError>) {
        if let Some(entry) = self
            .inner
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id)
        {
            let _ = entry.tx.send(result);
        }
    }
}

/// Frame parser: `Content-Length` headers + JSON body, with resync on junk.
pub(crate) struct Framer {
    buf: Vec<u8>,
}

impl Framer {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
    /// Bytes read past the last complete frame (mux handshake leftover).
    pub(crate) fn take_remainder(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buf)
    }
    /// Pop the next complete message body; `None` when more bytes are needed.
    pub(crate) fn next(&mut self) -> Option<String> {
        loop {
            let header_end = find_subslice(&self.buf, b"\r\n\r\n")?;
            let header_bytes = &self.buf[..header_end];
            let mut content_length: Option<usize> = None;
            let mut ok = true;
            for line in header_bytes.split(|&b| b == b'\n') {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                if let Some((name, value)) = split_header(line) {
                    if name.eq_ignore_ascii_case(b"content-length") {
                        content_length = std::str::from_utf8(value)
                            .ok()
                            .and_then(|v| v.trim().parse().ok());
                    }
                } else if !line.is_empty() {
                    ok = false;
                }
            }
            let Some(len) = content_length else {
                if !ok {
                    warn!("LSP framing resync: header block without Content-Length");
                }
                // Drop the bogus header block and resync past it.
                self.buf.drain(..header_end + 4);
                continue;
            };
            let total = header_end + 4 + len;
            if self.buf.len() < total {
                return None;
            }
            let body = self.buf[header_end + 4..total].to_vec();
            self.buf.drain(..total);
            return String::from_utf8_lossy(&body).into_owned().into();
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn split_header(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let idx = line.iter().position(|&b| b == b':')?;
    Some((&line[..idx], &line[idx + 1..]))
}

impl LspClient {
    /// Spawn the stdout reader task; routes responses, notifications and
    /// server-initiated requests (omp `startMessageReader`).
    fn start_reader(&self, mut stdout: Box<dyn tokio::io::AsyncRead + Unpin + Send>) {
        let client = self.clone();
        tokio::spawn(async move {
            let mut framer = Framer::new();
            let mut chunk = vec![0u8; 64 * 1024];
            loop {
                match stdout.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => framer.push(&chunk[..n]),
                }
                while let Some(text) = framer.next() {
                    let Ok(message) = serde_json::from_str::<Value>(&text) else {
                        continue;
                    };
                    client.route_message(message).await;
                }
            }
            client.on_reader_exit();
        });
    }

    async fn route_message(&self, message: Value) {
        // `method` first: server request ids collide with our pending ids.
        if let Some(method) = message.get("method").and_then(Value::as_str) {
            let id = message.get("id").cloned();
            match id {
                Some(id) => {
                    self.handle_server_request(method, id, message.get("params").cloned())
                        .await
                }
                None => self.handle_server_notification(method, message.get("params").cloned()),
            }
            return;
        }
        if let Some(id) = message.get("id").and_then(Value::as_i64) {
            if let Some(error) = message.get("error") {
                let code = error.get("code").and_then(Value::as_i64);
                let msg = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                let text = match code {
                    Some(c) => format!("LSP error {c}: {msg}"),
                    None => format!("LSP error: {msg}"),
                };
                self.resolve_pending(
                    id,
                    Err(LspError {
                        message: text,
                        cancelled: false,
                    }),
                );
            } else {
                self.resolve_pending(
                    id,
                    Ok(message.get("result").cloned().unwrap_or(Value::Null)),
                );
            }
        }
    }

    fn handle_server_notification(&self, method: &str, params: Option<Value>) {
        match (method, params) {
            ("textDocument/publishDiagnostics", Some(params)) => {
                if let Ok(published) = serde_json::from_value::<PublishDiagnosticsParams>(params) {
                    let key = uri_key(&published.uri);
                    self.inner
                        .diagnostics
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(
                            key,
                            crate::types::PublishedDiagnostics {
                                diagnostics: published.diagnostics,
                                version: published.version,
                            },
                        );
                    self.inner
                        .diagnostics_version
                        .fetch_add(1, Ordering::SeqCst);
                }
            }
            ("$/progress", Some(params)) => {
                let kind = params
                    .get("value")
                    .and_then(|v| v.get("kind"))
                    .and_then(Value::as_str);
                if kind == Some("end") {
                    let _ = self.inner.project_loaded_tx.send(true);
                }
            }
            _ => {}
        }
    }

    async fn handle_server_request(&self, method: &str, id: Value, params: Option<Value>) {
        let respond = |result: Value| {
            let client = self.clone();
            let id = id.clone();
            async move {
                let _ = client
                    .write_message(&json!({"jsonrpc": "2.0", "id": id, "result": result}))
                    .await;
            }
        };
        match method {
            "workspace/configuration" => {
                respond(self.workspace_configuration_result(params.as_ref())).await;
            }
            "workspace/workspaceFolders" => {
                let uri = file_to_uri(&self.inner.cwd);
                let name = self
                    .inner
                    .cwd
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| "workspace".to_string());
                respond(json!([{ "uri": uri, "name": name }])).await;
            }
            "workspace/applyEdit" => {
                let applied = match params.as_ref().and_then(|p| p.get("edit")) {
                    Some(edit) => {
                        match crate::edits::apply_workspace_edit_value(edit, &self.inner.cwd).await
                        {
                            Ok(_) => json!({ "applied": true }),
                            Err(e) => json!({ "applied": false, "failureReason": e.to_string() }),
                        }
                    }
                    None => json!({ "applied": false, "failureReason": "No edit provided" }),
                };
                respond(applied).await;
            }
            "client/registerCapability" => {
                if let Some(regs) = params
                    .as_ref()
                    .and_then(|p| p.get("registrations"))
                    .and_then(Value::as_array)
                {
                    let mut map = self
                        .inner
                        .dynamic_registrations
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    for reg in regs {
                        if let (Some(id), Some(m)) = (
                            reg.get("id").and_then(Value::as_str),
                            reg.get("method").and_then(Value::as_str),
                        ) {
                            map.insert(id.to_string(), m.to_string());
                        }
                    }
                }
                respond(Value::Null).await;
            }
            "client/unregisterCapability" => {
                if let Some(regs) = params
                    .as_ref()
                    .and_then(|p| {
                        p.get("unregistrations")
                            .or_else(|| p.get("unregisterations"))
                    })
                    .and_then(Value::as_array)
                {
                    let mut map = self
                        .inner
                        .dynamic_registrations
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    for reg in regs {
                        if let Some(id) = reg.get("id").and_then(Value::as_str) {
                            map.remove(id);
                        }
                    }
                }
                respond(Value::Null).await;
            }
            "window/workDoneProgress/create" | "window/showMessageRequest" => {
                respond(Value::Null).await;
            }
            "window/showDocument" => {
                respond(json!({ "success": false })).await;
            }
            m if m.starts_with("workspace/") && m.ends_with("/refresh") => {
                respond(Value::Null).await;
            }
            other => {
                let _ = self
                    .write_message(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("Method not found: {other}") }
                    }))
                    .await;
            }
        }
    }

    /// Answer `workspace/configuration` items from the server config settings.
    fn workspace_configuration_result(&self, params: Option<&Value>) -> Value {
        let items = params
            .and_then(|p| p.get("items"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let settings = self.inner.config.settings.clone();
        let result: Vec<Value> = items
            .iter()
            .map(|item| {
                let section = item.get("section").and_then(Value::as_str).unwrap_or("");
                settings
                    .as_ref()
                    .and_then(|s| s.get(section))
                    .cloned()
                    .unwrap_or(Value::Null)
            })
            .collect();
        Value::Array(result)
    }

    /// Reader exited: tear the client down so the next call respawns.
    fn on_reader_exit(&self) {
        let _ = self.inner.project_loaded_tx.send(true);
        self.inner.transport_dead.notify_waiters();
        *self
            .inner
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = ClientStatus::Error;
        let stderr = String::from_utf8_lossy(
            &self
                .inner
                .stderr
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
        .trim()
        .to_string();
        let code = self
            .inner
            .child
            .lock()
            .map(|mut c| {
                c.as_mut()
                    .and_then(|ch| ch.try_wait().ok().flatten())
                    .map(|s| s.code())
            })
            .ok()
            .flatten()
            .flatten();
        let message = if stderr.is_empty() {
            format!(
                "LSP server exited unexpectedly (code {})",
                code.map_or("?".to_string(), |c| c.to_string())
            )
        } else {
            format!(
                "LSP server exited (code {}): {stderr}",
                code.map_or("?".to_string(), |c| c.to_string())
            )
        };
        let err = LspError {
            message,
            cancelled: false,
        };
        let pending = std::mem::take(
            &mut *self
                .inner
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for (_, entry) in pending {
            let _ = entry.tx.send(Err(err.clone()));
        }
        registry_remove(&self.inner.name);
        // Kill the process if it is still alive.
        if let Ok(mut guard) = self.inner.child.lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
        }
    }
}

// =============================================================================
// Registry
// =============================================================================

type SharedInit = Arc<tokio::sync::OnceCell<Result<LspClient, Arc<LspError>>>>;

struct Registry {
    clients: Mutex<HashMap<String, LspClient>>,
    pending: Mutex<HashMap<String, SharedInit>>,
    init_failures: Mutex<HashMap<String, (Instant, String)>>,
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| Registry {
        clients: Mutex::new(HashMap::new()),
        pending: Mutex::new(HashMap::new()),
        init_failures: Mutex::new(HashMap::new()),
    })
}

fn registry_remove(name: &str) {
    let reg = registry();
    let _ = reg.clients.lock().map(|mut g| g.remove(name));
    maybe_stop_idle_checker();
}

fn registry_insert(client: &LspClient) {
    let _ = registry()
        .clients
        .lock()
        .map(|mut g| g.insert(client.name().to_string(), client.clone()));
}

/// All currently live clients (omp `getActiveClients` input).
#[must_use]
pub fn active_clients() -> Vec<LspClient> {
    registry()
        .clients
        .lock()
        .map(|g| g.values().cloned().collect())
        .unwrap_or_default()
}

/// Public status row for one active server.
#[derive(Debug, Clone)]
pub struct LspServerStatus {
    pub name: String,
    pub status: ClientStatus,
    pub file_types: Vec<String>,
    pub error: Option<String>,
}

/// Status of currently active LSP servers (omp `getActiveClients`).
#[must_use]
pub fn get_active_clients() -> Vec<LspServerStatus> {
    active_clients()
        .into_iter()
        .map(|c| LspServerStatus {
            name: c.config().command.clone(),
            status: c.status(),
            file_types: c.config().file_types.clone(),
            error: None,
        })
        .collect()
}

/// Allow an explicit user reload to retry a matching initialization failure.
pub fn clear_initialization_failure(config: &ServerConfig, cwd: &Path) {
    let key = LspClient::name_for(config, cwd);
    let _ = registry().init_failures.lock().map(|mut g| g.remove(&key));
}

/// Get or create a client for `config` in `cwd` (omp `getOrCreateClient`).
///
/// Concurrent callers share one in-flight initialization. Deterministic init
/// failures are negative-cached for `INIT_FAILURE_BACKOFF_MS`.
///
/// # Errors
///
/// Spawn/initialize failures, negative-cache hits, and cancellation.
pub async fn get_or_create_client(
    config: &ServerConfig,
    cwd: &Path,
    init_timeout: Option<Duration>,
    cancel: Option<&dyn Cancel>,
) -> Result<LspClient, LspError> {
    let key = LspClient::name_for(config, cwd);
    let reg = registry();

    if let Ok(guard) = reg.clients.lock() {
        if let Some(existing) = guard.get(&key) {
            existing
                .inner
                .last_activity
                .store(now_millis(), Ordering::SeqCst);
            maybe_start_idle_checker();
            return Ok(existing.clone());
        }
    }

    let cell: SharedInit = {
        let mut guard = reg
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cell) = guard.get(&key) {
            let cell = Arc::clone(cell);
            drop(guard);
            return match cell.get() {
                Some(Ok(client)) => Ok(client.clone()),
                Some(Err(e)) => Err((**e).clone()),
                None => Err(LspError {
                    message: "LSP initialization aborted".to_string(),
                    cancelled: false,
                }),
            };
        }
        let cell: SharedInit = Arc::new(tokio::sync::OnceCell::new());
        guard.insert(key.clone(), Arc::clone(&cell));
        cell
    };

    let result = init_client(config, cwd, &key, init_timeout, cancel).await;
    let result = match result {
        Ok(client) => {
            registry_insert(&client);
            reg.init_failures
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&key);
            Ok(client)
        }
        Err(e) => {
            // Negative-cache deterministic failures (not caller aborts/short deadlines).
            let transient = e.cancelled
                || init_timeout.is_some_and(|t| {
                    e.message.contains("timed out")
                        && t < Duration::from_millis(DEFAULT_REQUEST_TIMEOUT_MS)
                });
            if !transient {
                reg.init_failures
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(key.clone(), (Instant::now(), e.message.clone()));
            }
            Err(e)
        }
    };
    reg.pending
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&key);
    let _ = cell.set(result.map_err(Arc::new));
    match cell.get() {
        Some(Ok(client)) => Ok(client.clone()),
        Some(Err(e)) => Err((**e).clone()),
        None => Err(LspError {
            message: "LSP initialization aborted".to_string(),
            cancelled: false,
        }),
    }
}

/// A live client if one exists, without starting a server (omp `getActiveOrPendingClient`).
pub async fn get_active_or_pending_client(
    config: &ServerConfig,
    cwd: &Path,
    cancel: Option<&dyn Cancel>,
) -> Option<LspClient> {
    let key = LspClient::name_for(config, cwd);
    let reg = registry();
    if let Ok(guard) = reg.clients.lock() {
        if let Some(client) = guard.get(&key) {
            client
                .inner
                .last_activity
                .store(now_millis(), Ordering::SeqCst);
            return Some(client.clone());
        }
    }
    let cell = reg
        .pending
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .cloned()?;
    // Poll the shared init cell; tokio's OnceCell has no wait-on-set, so poll
    // with a short sleep until the in-flight initialization publishes.
    loop {
        if let Some(result) = cell.get() {
            return match result {
                Ok(client) => Some(client.clone()),
                Err(_) => None,
            };
        }
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(25)) => {}
            () = async { if let Some(c) = cancel { c.cancelled().await } } => return None,
        }
    }
}

/// Client capabilities advertised in `initialize` (omp `CLIENT_CAPABILITIES`).
#[must_use]
pub fn client_capabilities() -> Value {
    json!({
        "textDocument": {
            "synchronization": { "didSave": true, "dynamicRegistration": false, "willSave": false, "willSaveWaitUntil": false },
            "hover": { "contentFormat": ["markdown", "plaintext"], "dynamicRegistration": false },
            "definition": { "dynamicRegistration": false, "linkSupport": true },
            "typeDefinition": { "dynamicRegistration": false, "linkSupport": true },
            "implementation": { "dynamicRegistration": false, "linkSupport": true },
            "references": { "dynamicRegistration": false },
            "documentSymbol": {
                "dynamicRegistration": false,
                "hierarchicalDocumentSymbolSupport": true,
                "symbolKind": { "valueSet": (1..=26).collect::<Vec<i32>>() }
            },
            "rename": { "dynamicRegistration": false, "prepareSupport": true },
            "codeAction": {
                "dynamicRegistration": false,
                "codeActionLiteralSupport": { "codeActionKind": { "valueSet": [
                    "quickfix", "refactor", "refactor.extract", "refactor.inline",
                    "refactor.rewrite", "source", "source.organizeImports", "source.fixAll"
                ] } },
                "resolveSupport": { "properties": ["edit"] }
            },
            "formatting": { "dynamicRegistration": false },
            "rangeFormatting": { "dynamicRegistration": false },
            "publishDiagnostics": {
                "relatedInformation": true, "versionSupport": true,
                "tagSupport": { "valueSet": [1, 2] },
                "codeDescriptionSupport": true, "dataSupport": true
            },
            "diagnostic": { "dynamicRegistration": true }
        },
        "window": { "workDoneProgress": true },
        "workspace": {
            "applyEdit": true,
            "workspaceEdit": {
                "documentChanges": true,
                "resourceOperations": ["create", "rename", "delete"],
                "failureHandling": "abort"
            },
            "configuration": true,
            "workspaceFolders": true,
            "symbol": { "dynamicRegistration": false, "symbolKind": { "valueSet": (1..=26).collect::<Vec<i32>>() } },
            "fileOperations": {
                "dynamicRegistration": false,
                "willCreate": false, "didCreate": false,
                "willRename": true, "didRename": true,
                "willDelete": false, "didDelete": false
            }
        }
    })
}

type ServerWriter = Box<dyn tokio::io::AsyncWrite + Unpin + Send>;
type ServerReader = Box<dyn tokio::io::AsyncRead + Unpin + Send>;

/// Open the server transport for one client: the broker-shared mux link when
/// sharing is enabled, otherwise a process-local spawn (omp
/// `connectSharedLspTransport` + the `ptree.spawn` fallback). Any
/// shared-path failure degrades to the local spawn so LSP never regresses
/// on mux trouble.
async fn open_server_transport(
    command: &str,
    args: &[String],
    cwd: &Path,
) -> Result<(ServerWriter, ServerReader, Option<Child>, Option<u32>), LspError> {
    if crate::mux::shared_lsp_enabled() {
        if let Some(transport) =
            crate::mux::connect_shared_lsp_transport(command, args, cwd, None).await
        {
            return Ok((
                Box::new(transport.writer),
                Box::new(transport.reader),
                None,
                transport.pid,
            ));
        }
    }
    let mut child = Command::new(command)
        .args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| LspError {
            message: format!("failed to spawn LSP server {command}: {e}"),
            cancelled: false,
        })?;
    let stdin = child.stdin.take().ok_or_else(|| LspError {
        message: "LSP server stdin unavailable".to_string(),
        cancelled: false,
    })?;
    let stdout = child.stdout.take().ok_or_else(|| LspError {
        message: "LSP server stdout unavailable".to_string(),
        cancelled: false,
    })?;
    Ok((Box::new(stdin), Box::new(stdout), Some(child), None))
}

async fn init_client(
    config: &ServerConfig,
    cwd: &Path,
    key: &str,
    init_timeout: Option<Duration>,
    cancel: Option<&dyn Cancel>,
) -> Result<LspClient, LspError> {
    // Fail fast on a recent deterministic init failure.
    if let Ok(failures) = registry().init_failures.lock() {
        if let Some((at, message)) = failures.get(key) {
            if at.elapsed() < Duration::from_millis(INIT_FAILURE_BACKOFF_MS) {
                return Err(LspError {
                    message: format!(
                        "LSP server {} failed to initialize recently: {message}",
                        config.command
                    ),
                    cancelled: false,
                });
            }
        }
    }
    if let Some(c) = cancel {
        if c.is_cancelled() {
            return Err(LspError {
                message: "operation cancelled".to_string(),
                cancelled: true,
            });
        }
    }

    let command = config
        .resolved_command
        .clone()
        .unwrap_or_else(|| config.command.clone());
    let args: Vec<String> = config.args.clone().unwrap_or_default();

    let (writer, reader, mut child, mux_pid) = open_server_transport(&command, &args, cwd).await?;
    let mut stderr_pipe = match child.as_mut() {
        Some(child) => child.stderr.take(),
        None => None,
    };

    let (project_tx, project_rx) = watch::channel(false);
    // Auto-resolve after the project-load timeout (servers may never publish progress).
    {
        let project_tx = project_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(PROJECT_LOAD_TIMEOUT_MS)).await;
            let _ = project_tx.send(true);
        });
    }

    let inner = Arc::new(ClientInner {
        name: key.to_string(),
        cwd: cwd.to_path_buf(),
        config: config.clone(),
        request_id: AtomicI64::new(0),
        status: Mutex::new(ClientStatus::Connecting),
        open_files: Mutex::new(HashMap::new()),
        diagnostics: Mutex::new(HashMap::new()),
        diagnostics_version: AtomicU64::new(0),
        pending: Mutex::new(HashMap::new()),
        server_capabilities: Mutex::new(Value::Null),
        last_activity: AtomicI64::new(now_millis()),
        started_at: now_millis(),
        stdin: Arc::new(AsyncMutex::new(Some(writer))),
        child: Mutex::new(child),
        shared_mux: mux_pid.is_some(),
        mux_pid,
        transport_dead: Arc::new(tokio::sync::Notify::new()),
        stderr: Arc::new(Mutex::new(Vec::new())),
        dynamic_registrations: Mutex::new(HashMap::new()),
        project_loaded_tx: project_tx,
        _project_loaded_rx: project_rx,
        rust_analyzer_ready: Mutex::new(false),
    });
    let client = LspClient { inner };

    // Drain stderr into a buffer for post-mortems (mux links have none).
    if let Some(pipe) = stderr_pipe.take() {
        drain_stderr(pipe, Arc::clone(&client.inner.stderr));
    }

    client.start_reader(reader);

    run_initialize_handshake(&client, config, cwd, cancel, init_timeout).await?;

    *client
        .inner
        .status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = ClientStatus::Ready;
    maybe_start_idle_checker();
    Ok(client)
}

/// Spawn the task copying the server stderr pipe into the shared post-mortem buffer.
fn drain_stderr(mut pipe: tokio::process::ChildStderr, sink: Arc<Mutex<Vec<u8>>>) {
    tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        loop {
            match pipe.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => sink
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend_from_slice(&buf[..n]),
            }
        }
    });
}

/// Run the `initialize` handshake through `initialized` +
/// `workspace/didChangeConfiguration`, tearing the client down on any failure.
///
/// # Errors
///
/// Initialize/notification failures (client is failed before returning).
async fn run_initialize_handshake(
    client: &LspClient,
    config: &ServerConfig,
    cwd: &Path,
    cancel: Option<&dyn Cancel>,
    init_timeout: Option<Duration>,
) -> Result<(), LspError> {
    let init_result = client
        .send_request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": file_to_uri(cwd),
                "rootPath": cwd.to_string_lossy(),
                "capabilities": client_capabilities(),
                "initializationOptions": config.init_options.clone().unwrap_or(json!({})),
                "workspaceFolders": [{
                    "uri": file_to_uri(cwd),
                    "name": cwd.file_name().map(|n| n.to_string_lossy().into_owned()).filter(|n| !n.is_empty()).unwrap_or_else(|| "workspace".to_string())
                }],
            }),
            cancel,
            init_timeout,
        )
        .await;

    let init_result = match init_result {
        Ok(r) if !r.is_null() => r,
        Ok(_) => {
            client.fail_init();
            return Err(LspError {
                message: "Failed to initialize LSP: no response".to_string(),
                cancelled: false,
            });
        }
        Err(e) => {
            client.fail_init();
            return Err(e);
        }
    };

    *client
        .inner
        .server_capabilities
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = init_result
        .get("capabilities")
        .cloned()
        .unwrap_or(Value::Null);

    if let Err(e) = client.send_notification("initialized", json!({})).await {
        client.fail_init();
        return Err(e);
    }
    if let Err(e) = client
        .send_notification(
            "workspace/didChangeConfiguration",
            json!({ "settings": config.settings.clone().unwrap_or(json!({})) }),
        )
        .await
    {
        client.fail_init();
        return Err(e);
    }
    Ok(())
}

impl LspClient {
    fn fail_init(&self) {
        *self
            .inner
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = ClientStatus::Error;
        if let Ok(mut guard) = self.inner.stdin.try_lock() {
            *guard = None;
        }
        if let Ok(mut guard) = self.inner.child.lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
        }
        let pending = std::mem::take(
            &mut *self
                .inner
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for (_, entry) in pending {
            let _ = entry.tx.send(Err(LspError {
                message: "LSP initialization failed".to_string(),
                cancelled: false,
            }));
        }
    }
}

impl LspClient {
    /// Record a pulled diagnostic report in the publish map (omp stores the
    /// pull result into `client.diagnostics` before returning it).
    pub fn store_pulled_diagnostics(
        &self,
        uri: &str,
        diagnostics: Vec<crate::types::Diagnostic>,
        expected_document_version: Option<i64>,
    ) {
        let key = crate::utils::uri_key(uri);
        let version = expected_document_version.or_else(|| {
            self.inner
                .open_files
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&key)
                .map(|f| f.version)
        });
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                key,
                crate::types::PublishedDiagnostics {
                    diagnostics,
                    version,
                },
            );
        self.inner
            .diagnostics_version
            .fetch_add(1, Ordering::SeqCst);
    }
}

// =============================================================================
// Project-Loaded Waiting / Shutdown / Idle Reaping
// =============================================================================

impl LspClient {
    /// Wait for the server's initial project load (omp `waitForProjectLoaded`).
    pub async fn wait_for_project_loaded(&self, cancel: Option<&dyn Cancel>) {
        let mut rx = self.inner.project_loaded_tx.subscribe();
        loop {
            if *rx.borrow() {
                break;
            }
            tokio::select! {
                changed = rx.changed() => {
                    if changed.is_err() || *rx.borrow() {
                        break;
                    }
                }
                () = async { if let Some(c) = cancel { c.cancelled().await } } => return,
            }
        }
        if self.is_rust_analyzer() {
            self.wait_for_rust_analyzer_workspace(cancel).await;
        }
    }

    async fn wait_for_rust_analyzer_workspace(&self, cancel: Option<&dyn Cancel>) {
        {
            let ready = self
                .inner
                .rust_analyzer_ready
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if *ready {
                return;
            }
        }
        let timings = self.inner.config.workspace_ready_timings.clone();
        let timeout_ms = timings
            .as_ref()
            .and_then(|t| t.timeout_ms)
            .unwrap_or(RUST_ANALYZER_WORKSPACE_READY_TIMEOUT_MS);
        let poll_ms = timings
            .as_ref()
            .and_then(|t| t.poll_ms)
            .unwrap_or(RUST_ANALYZER_WORKSPACE_READY_POLL_MS);
        let settle_ms = timings
            .as_ref()
            .and_then(|t| t.settle_ms)
            .unwrap_or(RUST_ANALYZER_WORKSPACE_READY_SETTLE_MS);
        let status_timeout = Duration::from_millis(
            timings
                .as_ref()
                .and_then(|t| t.status_request_timeout_ms)
                .unwrap_or(RUST_ANALYZER_STATUS_REQUEST_TIMEOUT_MS),
        );
        let started = Instant::now();
        loop {
            if cancel.is_some_and(super::Cancel::is_cancelled) {
                return;
            }
            match self
                .send_request(
                    "rust-analyzer/analyzerStatus",
                    json!({}),
                    cancel,
                    Some(status_timeout),
                )
                .await
            {
                Ok(Value::String(status)) => {
                    let ready = !status.starts_with("No workspaces");
                    if ready && started.elapsed() >= Duration::from_millis(settle_ms) {
                        *self
                            .inner
                            .rust_analyzer_ready
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
                        return;
                    }
                    if started.elapsed() >= Duration::from_millis(timeout_ms) {
                        return;
                    }
                }
                Err(e) => {
                    let timed_out = e
                        .message
                        .starts_with("LSP request rust-analyzer/analyzerStatus timed out after ");
                    if !timed_out || started.elapsed() >= Duration::from_millis(timeout_ms) {
                        return;
                    }
                }
                _ => return,
            }
            tokio::select! {
                () = tokio::time::sleep(Duration::from_millis(poll_ms)) => {}
                () = async { if let Some(c) = cancel { c.cancelled().await } } => return,
            }
        }
    }

    /// Graceful `shutdown` + `exit`, then kill (omp `shutdownClientInstance`).
    /// Returns `true` once the process is confirmed exited.
    pub async fn shutdown_instance(&self) -> bool {
        registry_remove(&self.inner.name);
        let pending = std::mem::take(
            &mut *self
                .inner
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for (_, entry) in pending {
            let _ = entry.tx.send(Err(LspError {
                message: "LSP client shutdown".to_string(),
                cancelled: false,
            }));
        }
        let shutdown_completed = self
            .send_request(
                "shutdown",
                Value::Null,
                None,
                Some(Duration::from_millis(SHUTDOWN_TIMEOUT_MS)),
            )
            .await
            .is_ok();
        if shutdown_completed {
            let _ = self.send_notification("exit", Value::Null).await;
            if self
                .wait_for_exit(Duration::from_millis(EXIT_TIMEOUT_MS))
                .await
            {
                return true;
            }
        }
        if let Ok(mut guard) = self.inner.child.lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
        }
        self.wait_for_exit(Duration::from_millis(EXIT_TIMEOUT_MS))
            .await
    }

    async fn wait_for_exit(&self, timeout: Duration) -> bool {
        if self.inner.shared_mux {
            // No local child: closing the link makes the mux sever the
            // session (best-effort didClose/cancel), and the reader task
            // observes the socket teardown.
            *self.inner.stdin.lock().await = None;
            let notified = self.inner.transport_dead.notified();
            tokio::select! {
                () = notified => return true,
                () = tokio::time::sleep(timeout) => return true,
            }
        }
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let mut guard = self
                    .inner
                    .child
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(child) = guard.as_mut() {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            *guard = None;
                            return true;
                        }
                        Ok(None) => {}
                        Err(_) => return true,
                    }
                } else {
                    return true;
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// Whether a client may be reaped by the idle checker (omp `isIdleClient`).
#[must_use]
pub fn is_idle_client(client: &LspClient, timeout_ms: u64) -> bool {
    let pending = client.inner.pending.lock().map_or(0, |g| g.len());
    if pending > 0 {
        return false;
    }
    let elapsed = now_millis().saturating_sub(client.inner.last_activity.load(Ordering::SeqCst));
    elapsed > timeout_ms.cast_signed()
}

static IDLE_OVERRIDE_MS: OnceLock<Mutex<Option<u64>>> = OnceLock::new();
static IDLE_HANDLE: OnceLock<Mutex<Option<tokio::task::JoinHandle<()>>>> = OnceLock::new();

fn idle_override() -> &'static Mutex<Option<u64>> {
    IDLE_OVERRIDE_MS.get_or_init(|| Mutex::new(None))
}

fn idle_handle() -> &'static Mutex<Option<tokio::task::JoinHandle<()>>> {
    IDLE_HANDLE.get_or_init(|| Mutex::new(None))
}

/// Configure a global fallback idle timeout (used by tests/overrides).
pub fn set_idle_timeout(ms: Option<u64>) {
    *idle_override()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = ms;
    reconcile_idle_checker();
}

fn effective_idle_timeout(client: Option<&LspClient>, config: Option<&LspConfig>) -> Option<u64> {
    if let Some(ms) = *idle_override()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        if ms > 0 {
            return Some(ms);
        }
    }
    if let Some(client) = client {
        let cwd = client.cwd().to_path_buf();
        if let Some(ms) = crate::config::get_config(&cwd).idle_timeout_ms {
            if ms > 0 {
                return Some(ms);
            }
        }
    }
    if let Some(config) = config {
        if let Some(ms) = config.idle_timeout_ms {
            if ms > 0 {
                return Some(ms);
            }
        }
    }
    None
}

fn has_configured_idle_timeout() -> bool {
    if effective_idle_timeout(None, None).is_some() {
        return true;
    }
    registry().clients.lock().is_ok_and(|g| {
        g.values()
            .any(|c| effective_idle_timeout(Some(c), None).is_some())
    })
}

/// Reconcile the background idle checker against configured timeouts.
pub fn reconcile_idle_checker() {
    if has_configured_idle_timeout() {
        let mut guard = idle_handle()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_none() {
            *guard = Some(tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(IDLE_CHECK_INTERVAL_MS));
                loop {
                    interval.tick().await;
                    check_idle_clients().await;
                }
            }));
        }
    } else {
        let mut guard = idle_handle()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(handle) = guard.take() {
            handle.abort();
        }
    }
}

fn maybe_start_idle_checker() {
    reconcile_idle_checker();
}

fn maybe_stop_idle_checker() {
    reconcile_idle_checker();
}

/// Whether the background idle checker interval is currently active.
#[must_use]
pub fn is_idle_checker_running() -> bool {
    idle_handle()
        .lock()
        .is_ok_and(|g| g.as_ref().is_some_and(|h| !h.is_finished()))
}

/// Sweep all registered clients against their idle timeout (omp `checkIdleClients`).
pub async fn check_idle_clients() {
    let clients = active_clients();
    for client in clients {
        if let Some(timeout_ms) = effective_idle_timeout(Some(&client), None) {
            if is_idle_client(&client, timeout_ms) {
                let _ = client.shutdown_instance().await;
            }
        }
    }
}

/// Shut down all clients (omp `shutdownAll`).
pub async fn shutdown_all() {
    if let Ok(mut guard) = idle_handle().lock() {
        if let Some(handle) = guard.take() {
            handle.abort();
        }
    }
    let clients: Vec<LspClient> = registry()
        .clients
        .lock()
        .map(|mut g| {
            let v = g.values().cloned().collect();
            g.clear();
            v
        })
        .unwrap_or_default();
    for client in clients {
        let _ = client.shutdown_instance().await;
    }
}

impl LspClient {
    /// Canonical disk paths of currently open documents.
    #[must_use]
    pub fn open_file_paths(&self) -> Vec<PathBuf> {
        self.inner
            .open_files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .cloned()
            .collect()
    }
}

impl LspClient {
    /// Close a document: send `didClose` and drop its overlay + diagnostics
    /// (omp `notifyRename` closes old URIs the same way).
    ///
    /// # Errors
    ///
    /// Write failure on the server pipe.
    pub async fn close_file(&self, uri: &str) -> Result<(), LspError> {
        self.send_notification(
            "textDocument/didClose",
            json!({ "textDocument": { "uri": uri } }),
        )
        .await?;
        let key = uri_key(uri);
        self.inner
            .open_files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&key);
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&key);
        Ok(())
    }

    /// Re-sync an open document from disk: bump the version, send full-text
    /// `didChange` + `didSave` (omp `refreshFile`). Opens the document first
    /// when the server has not seen it yet. Missing files are a no-op.
    pub async fn refresh_file(&self, path: &Path) {
        let uri = file_to_uri(path);
        // Drop cached diagnostics so waiters don't accept stale unversioned rows.
        self.inner
            .diagnostics
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&uri_key(&uri));
        let open = self
            .inner
            .open_files
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(&uri_key(&uri));
        if !open {
            let Ok(text) = std::fs::read_to_string(path) else {
                return;
            };
            let language_id = self
                .inner
                .config
                .language_id
                .clone()
                .unwrap_or_else(|| crate::utils::detect_language_id(path));
            if self
                .send_notification(
                    "textDocument/didOpen",
                    json!({
                        "textDocument": { "uri": uri, "languageId": language_id, "version": 1, "text": text }
                    }),
                )
                .await
                .is_err()
            {
                return;
            }
            self.inner
                .open_files
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(
                    uri_key(&uri),
                    OpenFile {
                        version: 1,
                        language_id,
                        synced_text: text,
                    },
                );
            self.inner
                .last_activity
                .store(now_millis(), Ordering::SeqCst);
            return;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            return;
        };
        let version = {
            let mut files = self
                .inner
                .open_files
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(info) = files.get_mut(&uri_key(&uri)) else {
                return;
            };
            info.version += 1;
            info.synced_text.clone_from(&content);
            info.version
        };
        if self
            .send_notification(
                "textDocument/didChange",
                json!({
                    "textDocument": { "uri": uri, "version": version },
                    "contentChanges": [{ "text": content }]
                }),
            )
            .await
            .is_err()
        {
            return;
        }
        let _ = self
            .send_notification(
                "textDocument/didSave",
                json!({ "textDocument": { "uri": uri }, "text": content }),
            )
            .await;
        self.inner
            .last_activity
            .store(now_millis(), Ordering::SeqCst);
    }
}
