//! Cross-process contract for the broker-shared LSP mux daemon
//! (omp `src/lsp/mux/protocol.ts`, v18.6.1). The wire protocol is kept
//! byte-compatible with the TS daemon: both use Content-Length framed
//! JSON-RPC 2.0 over a unix socket and the same private method names, so a
//! Rust session and a TS session can share one daemon per project.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::json;

/// Environment key carrying the socket endpoint the mux must listen on.
pub const LSP_MUX_SOCKET_ENV: &str = "OMP_LSP_MUX_SOCKET";

/// Environment key carrying the canonical project directory the mux serves.
pub const LSP_MUX_PROJECT_DIR_ENV: &str = "OMP_LSP_MUX_PROJECT_DIR";

/// Stable broker daemon name for the shared LSP mux.
pub const LSP_MUX_DAEMON_NAME: &str = "omp.lsp.mux";

/// Broker readiness regex matched against the banner printed by the worker.
pub const LSP_MUX_READY_PATTERN: &str = r"omp lsp mux listening on \S+";

/// Banner printed on stdout once the mux socket accepts connections.
#[must_use]
pub fn lsp_mux_ready_banner(endpoint: &str) -> String {
    format!("omp lsp mux listening on {endpoint}")
}

/// First (and only pre-LSP) request on a fresh connection: binds the link to
/// an idle server instance or spawns one.
pub const MUX_CONNECT_METHOD: &str = "omp/muxConnect";

/// Liveness probe answered with [`MUX_PING_RESULT`] without binding the link.
pub const MUX_PING_METHOD: &str = "omp/muxPing";
pub const MUX_PING_RESULT: &str = "pong";

/// Notification on a bound link: kill that link's server process (sent by a
/// tearing-down session so the next one cold-starts a fresh server).
pub const MUX_RESTART_METHOD: &str = "omp/muxRestartServer";

/// Client-side connect timeout (omp `CONNECT_TIMEOUT_MS`).
pub const CONNECT_TIMEOUT_MS: u64 = 3_000;
/// muxConnect handshake timeout (omp `HANDSHAKE_TIMEOUT_MS`).
pub const HANDSHAKE_TIMEOUT_MS: u64 = 10_000;
/// Ping probe timeout (omp `PROBE_TIMEOUT_MS`).
pub const PROBE_TIMEOUT_MS: u64 = 1_500;
/// Daemon ready-banner timeout (omp `READY_TIMEOUT_MS`).
pub const READY_TIMEOUT_MS: u64 = 15_000;
/// probe→spawn→probe rounds bounding cross-process start races.
pub const ENSURE_ATTEMPTS: usize = 3;

/// Handshake parameters identifying a reusable server process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuxConnectParams {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    pub cwd: String,
    /// Extra environment for the spawned server. `BTreeMap` keeps keys sorted
    /// so insertion order never splits one identity in two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<BTreeMap<String, String>>,
}

/// Handshake result (`MuxConnectResult` in TS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuxConnectResult {
    pub key: String,
    pub spawned: bool,
    pub pid: Option<u32>,
}

/// Stable hash helper (first `n` hex chars of SHA-256 over `data`).
fn hash_hex(data: &str, n: usize) -> String {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(data.as_bytes());
    let mut out = String::with_capacity(n);
    for byte in digest {
        if out.len() >= n {
            break;
        }
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Resolve the runtime directory holding per-project mux state.
///
/// TS keys the endpoint off the project dir inside a per-user runtime dir;
/// hashed because the path may be long and can contain characters unsafe for
/// socket paths.
#[must_use]
pub fn lsp_mux_runtime_dir(project_dir: &Path) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("TMPDIR")
                .map(PathBuf::from)
                .or_else(|| Some(std::env::temp_dir()))
        })?;
    let canonical = crate::utils::normalize_path(project_dir);
    let scope = hash_hex(&canonical.to_string_lossy(), 16);
    Some(base.join("omp-rs").join("lsp-mux").join(scope))
}

/// Resolve the Unix socket for one project scope.
#[must_use]
pub fn lsp_mux_endpoint(project_dir: &Path) -> Option<PathBuf> {
    lsp_mux_runtime_dir(project_dir).map(|dir| dir.join("lsp-mux.sock"))
}

/// Server identity key used by the mux registry.
///
/// Every input that changes what the spawned process actually *is* belongs in
/// this key, like TS: command + args + cwd + sorted env entries, JSON-encoded
/// and SHA-256 hashed (env values may carry credentials and must not leak
/// through the handshake result or logs).
#[must_use]
pub fn mux_server_key(params: &MuxConnectParams) -> String {
    let env_entries: Vec<serde_json::Value> = params
        .env
        .iter()
        .flat_map(|env| env.iter())
        .map(|(k, v)| json!([k, v]))
        .collect();
    let identity = json!([params.command, params.args, params.cwd, env_entries]).to_string();
    format!("sha256:{}", hash_hex(&identity, 64))
}
