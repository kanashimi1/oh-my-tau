//! Stdio transport: JSON-RPC 2.0 over a subprocess's stdin/stdout with
//! newline-delimited JSON, a port of `transports/stdio.ts`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{broadcast, oneshot, Mutex};
use tokio_util::sync::CancellationToken;

use crate::config::McpServerConfig;
use crate::errors::{may_have_run, Failure, McpError, McpTransportError, TransportKind};
use crate::timeout::is_timeout_enabled;
use crate::transport::{
    await_with_timeout, default_handler, dispatch_message, McpProgress, McpTransport,
    NotificationCallback, PendingRequests, RequestHandler, RequestIdAllocator, RespondCallback,
    TransportFut,
};

/// Stdio transport state shared with the read loop.
struct Shared {
    pending: Arc<parking_lot::Mutex<PendingRequests>>,
    handler: RequestHandler,
    respond: RespondCallback,
    on_notification: NotificationCallback,
}

/// MCP stdio transport: spawns `command [args...]` and speaks NDJSON on
/// stdin/stdout. Stderr is drained and logged at debug level (omp forwards it
/// to the log).
pub struct StdioTransport {
    config: McpServerConfig,
    server_name: String,
    cwd: Arc<PathBuf>,
    timeout_ms: u64,
    child: Option<Child>,
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    ids: RequestIdAllocator,
    shared: Option<Arc<Shared>>,
    shutdown: CancellationToken,
    /// Server→client progress updates (`notifications/progress`).
    progress_tx: broadcast::Sender<McpProgress>,
}

/// Per-request guard: when a request frame that reached the server is
/// abandoned (aborted = future dropped, or timed out) we owe the server one
/// `notifications/cancelled` with the original request id — exactly once,
/// never for `initialize`, and without waiting on the pipe (upstream
/// `mcp-cancel-notification`). Failing to send must not block the caller.
struct CancelGuard {
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    id: Option<Value>,
    method: String,
    /// The frame was fully written; the server may be running the tool.
    written: bool,
    /// The call settled normally (response or an error we report ourselves):
    /// no cancellation is owed.
    settled: bool,
}

impl CancelGuard {
    fn settle(&mut self) {
        self.settled = true;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.written || self.settled || self.method == "initialize" {
            return;
        }
        let (Some(stdin), Some(id)) = (self.stdin.take(), self.id.take()) else {
            return;
        };
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": { "requestId": id },
        });
        tokio::spawn(async move {
            let line = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string());
            // Best effort: bounded lock + write, give up rather than park.
            let _ = tokio::time::timeout(std::time::Duration::from_millis(500), async {
                let mut guard = stdin.lock().await;
                let _ = guard.write_all(line.as_bytes()).await;
                let _ = guard.write_all(b"\n").await;
                let _ = guard.flush().await;
            })
            .await;
        });
    }
}

/// Write one NDJSON frame under the stdin lock.
async fn write_frame(stdin: &Mutex<ChildStdin>, line: &str) -> std::io::Result<()> {
    let mut guard = stdin.lock().await;
    guard.write_all(line.as_bytes()).await?;
    guard.write_all(b"\n").await?;
    guard.flush().await
}

/// Run `fut` under the request timeout (disabled when `timeout_ms == 0`);
/// `Err(())` means the deadline elapsed.
async fn bounded<T>(timeout_ms: u64, fut: impl std::future::Future<Output = T>) -> Result<T, ()> {
    if !is_timeout_enabled(timeout_ms) {
        return Ok(fut.await);
    }
    tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), fut)
        .await
        .map_err(|_| ())
}

fn timeout_error(timeout_ms: u64) -> McpError {
    McpError::Transport(McpTransportError {
        transport: TransportKind::Stdio,
        stage: Failure::Timeout,
        failure: Failure::Timeout,
        retryable: false,
        message: format!("Request timeout after {timeout_ms}ms (stdin write blocked)"),
        code: None,
        data: None,
    })
}

impl StdioTransport {
    /// Build a transport for `server_name`; `cwd` is the project root used
    /// for `roots/list` answers and as the child process cwd default.
    pub fn new(server_name: &str, config: McpServerConfig, cwd: Arc<PathBuf>) -> Self {
        let timeout_ms = config.timeout_ms();
        Self {
            config,
            server_name: server_name.to_string(),
            cwd,
            timeout_ms,
            child: None,
            stdin: None,
            ids: RequestIdAllocator::default(),
            shared: None,
            shutdown: CancellationToken::new(),
            progress_tx: broadcast::channel(16).0,
        }
    }

    fn not_connected() -> McpError {
        McpError::Transport(McpTransportError {
            transport: TransportKind::Stdio,
            stage: Failure::Connect,
            failure: Failure::Closed,
            retryable: true,
            message: "Transport not connected".to_string(),
            code: None,
            data: None,
        })
    }
}

impl McpTransport for StdioTransport {
    fn request(&mut self, method: &str, params: Value) -> TransportFut<Result<Value, McpError>> {
        let Some(stdin) = self.stdin.clone() else {
            return Box::pin(async { Err(Self::not_connected()) });
        };
        let Some(shared) = self.shared.clone() else {
            return Box::pin(async { Err(Self::not_connected()) });
        };
        let string_ids = self
            .config
            .extra
            .get("requestIdFormat")
            .and_then(Value::as_str)
            .is_some_and(|f| f == "string");
        let id = self.ids.next(string_ids);
        let timeout_ms = self.timeout_ms;
        let method = method.to_string();
        // Long-running calls ask the server for `notifications/progress`
        // keyed by the request id (MCP `_meta.progressToken`).
        let mut params = params;
        if method == "tools/call" {
            if let Some(meta) = params
                .as_object_mut()
                .map(|m| m.entry("_meta").or_insert_with(|| json!({})))
                .and_then(Value::as_object_mut)
            {
                meta.entry("progressToken").or_insert_with(|| id.clone());
            }
        }
        let frame = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        Box::pin(async move {
            let (tx, rx) = oneshot::channel();
            shared
                .pending
                .lock()
                .map
                .insert(PendingRequests::key(&id), tx);
            let line = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string());
            let started = tokio::time::Instant::now();
            // Aborts surface as this future being dropped; timeouts settle
            // it below. Either way the guard owes the server at most one
            // `notifications/cancelled` once the frame was written.
            let mut guard = CancelGuard {
                stdin: Some(stdin.clone()),
                id: Some(id.clone()),
                method,
                written: false,
                settled: false,
            };
            // The stdin lock and the write both count against the request
            // timeout: a server that never drains stdin must not hang us.
            let write_result = bounded(timeout_ms, write_frame(&stdin, &line)).await;
            let write_err = match write_result {
                Ok(Ok(())) => None,
                Ok(Err(err)) => Some(McpError::Transport(McpTransportError {
                    transport: TransportKind::Stdio,
                    stage: Failure::Send,
                    failure: Failure::Send,
                    retryable: true,
                    message: format!("failed to write to MCP subprocess stdin: {err}"),
                    code: None,
                    data: None,
                })),
                Err(()) => Some(timeout_error(timeout_ms)),
            };
            if let Some(err) = write_err {
                guard.settle();
                shared.pending.lock().map.remove(&PendingRequests::key(&id));
                return Err(err);
            }
            guard.written = true;
            let remaining_ms = if timeout_ms == 0 {
                0
            } else {
                let spent = started.elapsed().as_millis() as u64;
                timeout_ms.saturating_sub(spent).max(1)
            };
            let mut result = await_with_timeout(rx, remaining_ms, TransportKind::Stdio).await;
            match result {
                Ok(_) => guard.settle(),
                Err(_) => {
                    shared.pending.lock().map.remove(&PendingRequests::key(&id));
                    if guard.written {
                        // The frame reached the server and then the call
                        // failed (timeout or the read loop's EOF): replaying
                        // would run the tool a second time
                        // (mcp-retry-double-run). Not settling the guard also
                        // owes the server a `notifications/cancelled`.
                        result = result.map_err(may_have_run);
                    }
                }
            }
            result
        })
    }

    fn progress_sink(&self) -> Option<broadcast::Sender<McpProgress>> {
        Some(self.progress_tx.clone())
    }

    fn notify(&mut self, method: &str, params: Value) -> TransportFut<Result<(), McpError>> {
        let Some(stdin) = self.stdin.clone() else {
            return Box::pin(async { Err(Self::not_connected()) });
        };
        let timeout_ms = self.timeout_ms;
        let frame = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        Box::pin(async move {
            let line = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string());
            match bounded(timeout_ms, write_frame(&stdin, &line)).await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(err)) => Err(McpError::Transport(McpTransportError {
                    transport: TransportKind::Stdio,
                    stage: Failure::Send,
                    failure: Failure::Send,
                    retryable: true,
                    message: format!("failed to write notification: {err}"),
                    code: None,
                    data: None,
                })),
                Err(()) => Err(timeout_error(timeout_ms)),
            }
        })
    }

    fn close(&mut self) -> TransportFut<()> {
        self.shutdown.cancel();
        let mut child = self.child.take();
        self.stdin.take();
        Box::pin(async move {
            if let Some(mut child) = child.take() {
                // SIGTERM, brief grace, then SIGKILL — omp terminates the
                // process group; we terminate the direct child.
                let _ = child.start_kill();
                let _ = tokio::time::timeout(std::time::Duration::from_millis(1000), child.wait())
                    .await;
            }
        })
    }
}

impl StdioTransport {
    /// Spawn the subprocess and start the read loop. Mirrors
    /// `StdioTransport.connect()`.
    pub async fn connect(&mut self) -> Result<(), McpError> {
        if self.child.is_some() {
            return Ok(());
        }
        let command = self.config.command.clone().ok_or_else(|| {
            McpError::Config(format!(
                "stdio MCP server \"{}\" has no command",
                self.server_name
            ))
        })?;
        // TS `detached: true` on Linux (setsid): the server tree has no
        // controlling terminal, so job-control signals cannot stop it and it
        // cannot touch the TUI's tty. util-linux `setsid` execs in place.
        let setsid = ["/usr/bin/setsid", "/bin/setsid"]
            .into_iter()
            .find(|p| cfg!(target_os = "linux") && std::path::Path::new(p).is_file());
        let mut cmd = match setsid {
            Some(setsid) => {
                let mut c = Command::new(setsid);
                c.arg("--wait").arg(&command);
                c
            }
            None => Command::new(&command),
        };
        cmd.args(self.config.args.clone().unwrap_or_default())
            .envs(self.config.env.clone().unwrap_or_default())
            .current_dir(
                self.config
                    .cwd
                    .clone()
                    .map_or_else(|| self.cwd.as_ref().clone(), PathBuf::from),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|err| {
            McpError::Transport(McpTransportError {
                transport: TransportKind::Stdio,
                stage: Failure::Connect,
                failure: Failure::Connect,
                retryable: true,
                message: format!("failed to spawn MCP server \"{}\": {err}", self.server_name),
                code: None,
                data: None,
            })
        })?;
        let stdout = child.stdout.take().ok_or_else(Self::not_connected)?;
        let stderr = child.stderr.take();
        let stdin = child.stdin.take().ok_or_else(Self::not_connected)?;
        let stdin = Arc::new(Mutex::new(stdin));

        let pending = Arc::new(parking_lot::Mutex::new(PendingRequests {
            map: HashMap::new(),
        }));
        let respond: RespondCallback = {
            let stdin = stdin.clone();
            Arc::new(move |msg: Value| {
                let stdin = stdin.clone();
                tokio::spawn(async move {
                    let line = serde_json::to_string(&msg).unwrap_or_else(|_| "{}".to_string());
                    let mut guard = stdin.lock().await;
                    let _ = guard.write_all(line.as_bytes()).await;
                    let _ = guard.write_all(b"\n").await;
                    let _ = guard.flush().await;
                });
            })
        };
        let progress_tx = self.progress_tx.clone();
        let on_notification: NotificationCallback = Arc::new(move |method, params| {
            tracing::debug!("MCP server notification: {method} {params}");
            if method == "notifications/progress" {
                if let Some(update) = McpProgress::from_params(&params) {
                    let _ = progress_tx.send(update);
                }
            }
        });
        let shared = Arc::new(Shared {
            pending: pending.clone(),
            handler: default_handler(self.cwd.clone()),
            respond,
            on_notification,
        });
        self.shared = Some(shared.clone());
        self.stdin = Some(stdin);
        self.child = Some(child);

        // Read loop: NDJSON on stdout.
        let mut reader = BufReader::new(stdout).lines();
        let shutdown = self.shutdown.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown.cancelled() => break,
                    line = reader.next_line() => {
                        let Ok(Some(line)) = line else {
                            // EOF: reject all pending requests, like
                            // `#handleClose` with an eof failure.
                            let mut guard = pending.lock();
                            for (_, tx) in guard.map.drain() {
                                let _ = tx.send(Err(McpError::Transport(McpTransportError {
                                    transport: TransportKind::Stdio,
                                    stage: Failure::Receive,
                                    failure: Failure::Eof,
                                    retryable: true,
                                    message: "MCP subprocess closed stdout before responding".to_string(),
                                    code: None,
                                    data: None,
                                })));
                            }
                            break;
                        };
                        if line.trim().is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Value>(&line) {
                            Ok(msg) => {
                                // Inline (not spawned): JSON-RPC over a stream
                                // is ordered — a `notifications/progress`
                                // must reach subscribers before the response
                                // that ends the call, and dispatch only parks
                                // briefly on the pending-request lock.
                                dispatch_message(
                                    msg,
                                    TransportKind::Stdio,
                                    &pending,
                                    &shared.handler,
                                    &shared.respond,
                                    &shared.on_notification,
                                )
                                .await;
                            }
                            Err(err) => tracing::debug!("ignoring malformed MCP frame: {err}"),
                        }
                    }
                }
            }
        });

        // Stderr drain (omp logs it; kept at debug).
        if let Some(stderr) = stderr {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) if !line.trim().is_empty() => {
                            tracing::debug!("MCP server stderr: {line}");
                        }
                        _ => break,
                    }
                }
            });
        }
        Ok(())
    }
}
