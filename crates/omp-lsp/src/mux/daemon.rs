//! Client half of the broker-shared LSP mux (omp `src/lsp/mux/daemon.ts`,
//! v18.6.1): ensure the per-project mux daemon is running, dial its socket,
//! perform the `omp/muxConnect` handshake, and hand back a socket transport
//! the ordinary LSP client drives exactly like a stdio pipe.
//!
//! omp-rs has no standalone broker process (omp's broker is folded into the
//! session), so "ensure the daemon" spawns the `omp-lsp-mux` binary detached
//! instead of asking a broker. Everything else follows TS: any failure
//! degrades to `None` so the caller falls back to a process-local spawn.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use super::protocol::{
    lsp_mux_endpoint, MuxConnectParams, MuxConnectResult, CONNECT_TIMEOUT_MS, ENSURE_ATTEMPTS,
    HANDSHAKE_TIMEOUT_MS, LSP_MUX_PROJECT_DIR_ENV, LSP_MUX_SOCKET_ENV, MUX_CONNECT_METHOD,
    MUX_PING_METHOD, MUX_PING_RESULT, PROBE_TIMEOUT_MS, READY_TIMEOUT_MS,
};
use crate::client::Framer;

static NEXT_HANDSHAKE_ID: AtomicU64 = AtomicU64::new(1);

/// A connected, handshaken mux link: the read half (with any bytes the
/// handshake read past the reply), the write half, and the shared server's
/// pid from the connect result.
pub struct MuxTransport {
    pub reader: MuxReader,
    pub writer: tokio::net::unix::OwnedWriteHalf,
    pub pid: Option<u32>,
}

/// Read half replaying bytes the handshake consumed past the connect reply
/// before reading from the socket.
pub struct MuxReader {
    half: tokio::net::unix::OwnedReadHalf,
    pending: Vec<u8>,
}

impl MuxReader {
    fn new(half: tokio::net::unix::OwnedReadHalf, pending: Vec<u8>) -> Self {
        Self { half, pending }
    }
}

impl AsyncRead for MuxReader {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if !this.pending.is_empty() {
            let n = std::cmp::min(buf.remaining(), this.pending.len());
            buf.put_slice(&this.pending[..n]);
            this.pending.drain(..n);
            return std::task::Poll::Ready(Ok(()));
        }
        std::pin::Pin::new(&mut this.half).poll_read(cx, buf)
    }
}

fn frame(message: &Value) -> Vec<u8> {
    let body = serde_json::to_string(message).expect("serialize mux frame");
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

/// Resolve the `omp-lsp-mux` binary: explicit override, sibling of the
/// current executable, then a PATH search.
fn mux_binary() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("OMP_LSP_MUX_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.with_file_name("omp-lsp-mux");
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    let name = "omp-lsp-mux";
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

async fn connect_endpoint(endpoint: &Path, timeout: Duration) -> std::io::Result<UnixStream> {
    tokio::time::timeout(timeout, UnixStream::connect(endpoint))
        .await
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "timed out connecting to LSP mux",
            )
        })?
}

/// Send one request on a fresh socket and await its response, returning the
/// response plus any bytes read past it (omp `requestOnSocket`).
async fn request_on_socket(
    reader: &mut tokio::net::unix::OwnedReadHalf,
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    request: &Value,
    timeout: Duration,
) -> Result<(Value, Vec<u8>), String> {
    writer
        .write_all(&frame(request))
        .await
        .map_err(|e| format!("LSP mux handshake write failed: {e}"))?;
    let mut framer = Framer::new();
    let mut chunk = vec![0u8; 16 * 1024];
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        while let Some(text) = framer.next() {
            let Ok(message) = serde_json::from_str::<Value>(&text) else {
                continue;
            };
            if message.get("id") != request.get("id") {
                continue;
            }
            if message.get("error").is_some() {
                let msg = message
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                return Err(format!("LSP mux request failed: {msg}"));
            }
            let leftover = framer.take_remainder();
            return Ok((message, leftover));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("LSP mux handshake timed out".to_string());
        }
        match tokio::time::timeout(
            deadline - tokio::time::Instant::now(),
            reader.read(&mut chunk),
        )
        .await
        {
            Ok(Ok(0) | Err(_)) => {
                return Err("LSP mux connection closed during handshake".to_string())
            }
            Ok(Ok(n)) => framer.push(&chunk[..n]),
            Err(_) => return Err("LSP mux handshake timed out".to_string()),
        }
    }
}

/// Dial and handshake one server link; fails on any error (omp `dialMuxServer`).
async fn dial_mux_server(
    endpoint: &Path,
    params: &MuxConnectParams,
) -> Result<MuxTransport, String> {
    let stream = connect_endpoint(endpoint, Duration::from_millis(CONNECT_TIMEOUT_MS))
        .await
        .map_err(|e| format!("failed to connect to LSP mux: {e}"))?;
    let (mut reader, mut writer) = stream.into_split();
    let request = json!({
        "jsonrpc": "2.0",
        "id": format!("connect:{}", NEXT_HANDSHAKE_ID.fetch_add(1, Ordering::SeqCst) + 1),
        "method": MUX_CONNECT_METHOD,
        "params": serde_json::to_value(params).expect("serialize connect params"),
    });
    let (response, leftover) = request_on_socket(
        &mut reader,
        &mut writer,
        &request,
        Duration::from_millis(HANDSHAKE_TIMEOUT_MS),
    )
    .await?;
    let result: MuxConnectResult =
        serde_json::from_value(response.get("result").cloned().unwrap_or(Value::Null))
            .map_err(|e| format!("invalid mux connect result: {e}"))?;
    Ok(MuxTransport {
        reader: MuxReader::new(reader, leftover),
        writer,
        pid: result.pid,
    })
}

/// True when a mux answers the ping handshake at `endpoint` (omp `probeMux`).
async fn probe_mux(endpoint: &Path) -> bool {
    let Ok(stream) = connect_endpoint(endpoint, Duration::from_millis(PROBE_TIMEOUT_MS)).await
    else {
        return false;
    };
    let (mut reader, mut writer) = stream.into_split();
    let request = json!({
        "jsonrpc": "2.0",
        "id": "probe",
        "method": MUX_PING_METHOD,
        "params": Value::Null,
    });
    matches!(
        request_on_socket(&mut reader, &mut writer, &request, Duration::from_millis(PROBE_TIMEOUT_MS)).await,
        Ok((response, _)) if response.get("result").and_then(Value::as_str) == Some(MUX_PING_RESULT)
    )
}

/// Ensure the project's mux daemon is running and reachable, spawning the
/// `omp-lsp-mux` binary detached when the socket does not answer. Returns the
/// endpoint, or `None` when the shared path is unavailable
/// (omp `ensureLspMuxDaemon`, broker start replaced by a detached spawn).
async fn ensure_lsp_mux_daemon(project_dir: &Path) -> Option<std::path::PathBuf> {
    let endpoint = lsp_mux_endpoint(project_dir)?;
    if probe_mux(&endpoint).await {
        return Some(endpoint);
    }
    let binary = mux_binary()?;
    for _attempt in 0..ENSURE_ATTEMPTS {
        // A concurrent start may have won since the last round; adopt it.
        if probe_mux(&endpoint).await {
            return Some(endpoint);
        }
        spawn_detached_mux(&binary, &endpoint, project_dir);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(READY_TIMEOUT_MS);
        loop {
            if probe_mux(&endpoint).await {
                return Some(endpoint);
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    probe_mux(&endpoint).await.then_some(endpoint)
}

/// Spawn the mux daemon detached from this process's lifetime.
fn spawn_detached_mux(binary: &Path, endpoint: &Path, project_dir: &Path) {
    let mut command = std::process::Command::new(binary);
    command
        .env(LSP_MUX_SOCKET_ENV, endpoint)
        .env(LSP_MUX_PROJECT_DIR_ENV, project_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Own process group so the daemon is not signalled with the session.
        let _ = command.process_group(0);
    }
    let _ = command.spawn();
}

/// Open a broker-shared transport for one language server, ensuring the mux
/// daemon first. Returns `None` (after logging) when the shared path is
/// unavailable so the caller falls back to a process-local spawn
/// (omp `connectSharedLspTransport`).
pub async fn connect_shared_lsp_transport(
    command: &str,
    args: &[String],
    cwd: &Path,
    env: Option<&std::collections::BTreeMap<String, String>>,
) -> Option<MuxTransport> {
    let project_dir = crate::utils::normalize_path(cwd);
    let endpoint = ensure_lsp_mux_daemon(&project_dir).await?;
    let params = MuxConnectParams {
        command: command.to_string(),
        args: args.to_vec(),
        cwd: project_dir.to_string_lossy().into_owned(),
        env: env.cloned(),
    };
    match dial_mux_server(&endpoint, &params).await {
        Ok(transport) => Some(transport),
        Err(error) => {
            tracing::debug!(
                "shared LSP transport unavailable; falling back to local spawn: {error}"
            );
            None
        }
    }
}
