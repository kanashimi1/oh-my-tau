//! Runtime local Collab host registry.
//!
//! Port of `vendor-src/oh-my-pi/packages/coding-agent/src/collab/registry.ts`
//! (762 lines). Every connected Collab host publishes a private, per-process
//! IPC endpoint (Unix domain socket) under `~/.omp/run/collab-hosts` so a
//! separate local process can discover live hosts and, on explicit request,
//! retrieve a shareable URL. Two contracts are served over that endpoint:
//!
//! - `snapshot`: non-capability metadata (identity, session, cwd, model,
//!   participants, relay/attention/activity state) suitable for listing and
//!   polling;
//! - `link`: one browser URL for one exact host generation and access level,
//!   rejected when the generation moved or the access level is not published.
//!
//! Full-control and view-only URLs, room keys, and write tokens stay in host
//! memory; disk carries only ephemeral discovery metadata (protocol version,
//! instance ID, PID, endpoint, creation time, random bearer token) with
//! owner-only permissions. Endpoints die with the host process, so a crash
//! leaves at most stale metadata that the next list operation prunes
//! best-effort.
//!
//! Transport follows the launch daemon broker conventions: newline-delimited
//! JSON envelopes (byte-identical to the TS `JSON.stringify` stream — the
//! workspace `serde_json` `preserve_order` feature keeps map insertion
//! order), per-request bearer authentication, and bounded buffers. Guests
//! never publish; this registry is host-only.
//!
//! Ownership note (collab port contract): [`CollabHostSnapshot`],
//! [`CollabHostRecord`], and [`collab_hosts_runtime_dir`] are the fixed
//! cross-task interfaces and live HERE; `host.rs`/`guest.rs`/`controller.rs`
//! consume them rather than defining their own copies.
//!
//! Deviations from the TS source (each is safe-direction or noted):
//! - POSIX only. TS branches to Windows named pipes (registry.ts:455-456);
//!   the Rust port serves Unix domain sockets exclusively.
//! - No `process.once("exit")` hook (registry.ts:513): teardown runs on
//!   [`CollabHostPublication::close`] / `Drop`. Crash survivors are pruned by
//!   the next list, exactly as the TS comment at registry.ts:473-476 plans.
//! - `boundField` truncates by UTF-16 code unit like TS but never splits a
//!   surrogate pair (a Rust `&str` cannot hold a lone surrogate), so output
//!   can differ from TS only when the 1024-unit boundary falls inside an
//!   astral character.
//! - Integer JSON fields are parsed with serde integer types: a peer emitting
//!   `2.0`-style floats for integer fields (never produced by
//!   `JSON.stringify` of integral numbers) would be rejected where TS would
//!   accept it. Records/snapshots that fail to parse are pruned/skipped
//!   either way.
//! - `localeCompare` in the list sort (registry.ts:682) is byte-wise
//!   comparison here; instance ids are `[a-z0-9-]` so ordering is identical.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rand::RngCore;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    net::UnixListener,
    task::{JoinHandle, JoinSet},
    time::timeout,
};

/// Discovery metadata / IPC protocol version. Mixed omp versions fail safely
/// (registry.ts:32).
pub const COLLAB_REGISTRY_VERSION: u64 = 1;

/// Reject request lines beyond this size; a valid request is <300 bytes
/// (registry.ts:35).
const MAX_REQUEST_BYTES: usize = 4 * 1024;

/// Reject responses beyond this size (registry.ts:37).
const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Longest string any snapshot field is sent with. Session names, imported
/// session ids, and working directories have no length limit of their own;
/// bounding them here keeps every valid snapshot response (five such fields,
/// worst case fully escaped) well inside [`MAX_RESPONSE_BYTES`], so an
/// unusual title never makes an otherwise healthy host invisible
/// (registry.ts:45).
const MAX_SNAPSHOT_FIELD_CHARS: usize = 1024;

/// Per-entry connect+response deadline during listing (registry.ts:47).
const DEFAULT_QUERY_TIMEOUT_MS: u64 = 1_500;

/// Concurrency bound for querying discovery entries (registry.ts:49).
const LIST_CONCURRENCY: usize = 8;

/// `sun_path` capacity: 104 bytes on macOS, 108 elsewhere; the kernel rejects
/// paths at or past it (registry.ts:397).
const SUN_PATH_LIMIT: usize = if cfg!(target_os = "macos") { 104 } else { 108 };

/// Base for the short socket directory used when the canonical socket path
/// would overflow `sun_path` (registry.ts:398).
const DEFAULT_SOCKET_FALLBACK_BASE: &str = "/tmp";

/// Access a link grants: `view` (bare room key) or `control` (room key + write
/// token) (registry.ts:52).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CollabAccess {
    /// Bare room key: read-only guest.
    View,
    /// Room key + write token: full-control guest.
    Control,
}

impl std::fmt::Display for CollabAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::View => f.write_str("view"),
            Self::Control => f.write_str("control"),
        }
    }
}

/// Model the host session is currently using, when one is selected
/// (registry.ts:87-89). Field order is wire order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollabHostModel {
    pub provider: String,
    pub id: String,
}

/// Non-capability host state, computed by the host process at query time
/// (registry.ts:59-97). Field declaration order is the wire order, matching
/// TS `JSON.stringify` of the interface-shaped object byte for byte.
/// Free-form strings (session id and name, cwd, model) are bounded to
/// [`MAX_SNAPSHOT_FIELD_CHARS`] UTF-16 units on the wire by
/// [`bound_snapshot`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollabHostSnapshot {
    /// Random per-process identity; stable across the host's room generations.
    pub instance_id: String,
    /// Increments every time this process starts a new room (session switch,
    /// restart).
    pub generation: u64,
    /// Host process ID.
    pub pid: i64,
    /// Session ID of the hosted conversation.
    pub session_id: String,
    /// Human-readable session name, when one is set.
    pub session_name: Option<String>,
    /// Host working directory.
    pub cwd: String,
    /// Model the host session is currently using, when one is selected.
    pub model: Option<CollabHostModel>,
    /// Epoch milliseconds when the host first connected to the relay.
    pub started_at: i64,
    /// Current participant count, including the host.
    pub participants: i64,
    /// Whether the host currently holds an open relay connection.
    pub relay_connected: bool,
    /// Whether a host-side question is waiting for an answer a writable guest
    /// could give.
    pub input_required: bool,
    /// Whether the session is running a turn: streaming a response or
    /// executing tools.
    ///
    /// `None` when the host does not report it (an omp older than this field;
    /// registry.ts:255-260 — a missing field reads as unknown, not idle).
    /// Serde maps absent AND `null` to `None`, and a non-bool to a parse
    /// error — exactly the TS `parseSnapshot` accept/reject set.
    #[serde(default)]
    pub busy: Option<bool>,
    /// Highest access the registry will hand out for this host.
    pub access: CollabAccess,
}

/// Live host state served over the IPC endpoint (registry.ts:99-105). Both
/// methods are sync like the TS source; the host computes them from in-memory
/// state. Errors are deliberately opaque: the server converts any failure to
/// the fixed `snapshot_unavailable` wire error so URLs and internals never
/// leak (registry.ts:322-327, 352-356).
pub trait CollabHostRegistrySource: Send + Sync + 'static {
    /// Current metadata; errors when the host can no longer vouch for its
    /// session.
    ///
    /// # Errors
    /// Anything the host cannot vouch for; the server collapses every error
    /// to the fixed `snapshot_unavailable` wire error (registry.ts:321-327).
    fn snapshot(&self) -> anyhow::Result<CollabHostSnapshot>;
    /// Browser URL for `access`, or `Ok(None)` when that access is not
    /// published.
    ///
    /// # Errors
    /// Anything the host cannot produce a URL for; collapsed to
    /// `snapshot_unavailable` on the wire (registry.ts:350-356).
    fn link(&self, access: CollabAccess) -> anyhow::Result<Option<String>>;
}

/// One resolved capability returned by [`resolve_collab_host_link`]
/// (registry.ts:107-113).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollabResolvedLink {
    pub instance_id: String,
    pub generation: u64,
    pub access: CollabAccess,
    pub url: String,
}

/// Stable failure codes for [`resolve_collab_host_link`]; never carry URLs
/// (registry.ts:149).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollabLinkErrorCode {
    NotFound,
    Ambiguous,
    StaleGeneration,
    AccessUnavailable,
    Unreachable,
}

impl std::fmt::Display for CollabLinkErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotFound => "not_found",
            Self::Ambiguous => "ambiguous",
            Self::StaleGeneration => "stale_generation",
            Self::AccessUnavailable => "access_unavailable",
            Self::Unreachable => "unreachable",
        })
    }
}

/// Typed link-resolution failure (registry.ts:151-158). The message strings
/// mirror the TS constructor call sites exactly; the machine-readable
/// [`CollabLinkErrorCode`] is what callers match on. TS throws this as a
/// named error class; here it travels inside `anyhow::Error`, so callers
/// recover the code with `err.downcast_ref::<CollabLinkError>()`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct CollabLinkError {
    pub code: CollabLinkErrorCode,
    pub message: String,
}

impl CollabLinkError {
    fn new(code: CollabLinkErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Options for [`list_collab_hosts`] / [`resolve_collab_host_link`]
/// (registry.ts:143-147).
#[derive(Debug, Clone, Default)]
pub struct CollabListOptions {
    /// Override the discovery metadata directory (tests).
    pub dir: Option<PathBuf>,
    /// Per-entry query deadline in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Options for [`publish_collab_host`] (registry.ts:115-141).
#[derive(Debug, Clone, Default)]
pub struct CollabPublishOptions {
    /// Override the discovery metadata directory (tests).
    pub dir: Option<PathBuf>,
    /// Identity recorded in the metadata and matched by
    /// `omp collab link <id>`. Defaults to a fresh random ID; a host that
    /// rotates rooms passes its process-lifetime instance ID so the
    /// replacement room keeps the same id (registry.ts:130-141).
    pub instance_id: Option<String>,
    /// Base for the short socket directory used when the canonical socket
    /// path would overflow `sun_path`. Defaults to `/tmp`; tests point it
    /// elsewhere (registry.ts:135-139).
    pub socket_fallback_base: Option<PathBuf>,
}

/// Discovery metadata persisted beside the endpoint (registry.ts:170-177,
/// wire shape `version/instanceId/pid/endpoint/createdAt/token` in that
/// order). Never carries URLs or room secrets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollabHostRecord {
    /// Discovery metadata / IPC protocol version.
    pub version: u64,
    /// Publisher-chosen instance identity (`^[a-z0-9-]{8,64}$`).
    pub instance_id: String,
    /// Host process ID.
    pub pid: i64,
    /// Absolute socket path the host listens on.
    pub endpoint: String,
    /// Epoch milliseconds when the entry was written.
    pub created_at: i64,
    /// Per-publication bearer token required by every request.
    pub token: String,
}

/// Discovery metadata directory. Deliberately under the profile-independent
/// config root — unlike the launch broker's profile-scoped runtime dir — so
/// hosts started under any profile are discoverable from any other
/// (registry.ts:160-167; TS `getBaseConfigRoot()` is `~/.omp` via
/// utils/src/dirs.ts:115-117 and :306-308).
#[must_use]
pub fn collab_hosts_runtime_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from);
    let name = std::env::var("PI_CONFIG_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| ".omp".into());
    home.join(name).join("run").join("collab-hosts")
}

/// `^[a-z0-9-]{8,64}$` (registry.ts:179) without a regex dependency.
fn instance_id_is_valid(instance_id: &str) -> bool {
    let chars = instance_id.chars().count();
    (8..=64).contains(&chars)
        && instance_id
            .chars()
            .all(|c| matches!(c, 'a'..='z' | '0'..='9' | '-'))
}

/// Parse one discovery metadata file (registry.ts:181-204). Unknown fields
/// are ignored; any missing or wrongly-typed required field rejects the whole
/// record so a partial write can never be listed (the writer renames into
/// place, so this only fires on tampering).
fn parse_collab_host_record(text: &str) -> Option<CollabHostRecord> {
    let raw: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = raw.as_object()?;
    let version = obj.get("version")?.as_u64()?;
    let instance_id = obj.get("instanceId")?.as_str()?;
    if !instance_id_is_valid(instance_id) {
        return None;
    }
    let pid = obj.get("pid")?.as_i64()?;
    if pid <= 0 {
        return None;
    }
    let endpoint = obj.get("endpoint")?.as_str()?;
    if endpoint.is_empty() {
        return None;
    }
    let created_at = obj.get("createdAt")?.as_i64()?;
    let token = obj.get("token")?.as_str()?;
    if token.is_empty() {
        return None;
    }
    Some(CollabHostRecord {
        version,
        instance_id: instance_id.to_owned(),
        pid,
        endpoint: endpoint.to_owned(),
        created_at,
        token: token.to_owned(),
    })
}

/// Constant-time bearer-token comparison (registry.ts:206-212). Length
/// mismatch returns `false` up front, exactly like the TS length check ahead
/// of `crypto.timingSafeEqual`; the fold accumulates the XOR of every byte
/// pair so no early exit depends on token content. A presented non-string
/// (missing/wrong JSON type) never matches.
fn token_matches(expected: &str, presented: Option<&str>) -> bool {
    let Some(presented) = presented else {
        return false;
    };
    let expected = expected.as_bytes();
    let presented = presented.as_bytes();
    expected.len() == presented.len()
        && expected
            .iter()
            .zip(presented)
            .fold(0_u8, |diff, (a, b)| diff | (a ^ b))
            == 0
}

/// Parse a snapshot value received over the wire (registry.ts:218-260).
/// `busy` is `Option` because the field postdates the protocol version and a
/// hard version gate would make every host invisible to a differently
/// versioned lister on the same machine — an older host simply omits it and a
/// missing one reads as unknown (`None`), not idle (registry.ts:255-260).
/// Post-parse checks re-apply the two structural invariants serde cannot
/// express: the instance-id pattern and `generation >= 1`.
fn parse_snapshot(raw: &serde_json::Value) -> Option<CollabHostSnapshot> {
    let snapshot: CollabHostSnapshot = serde_json::from_value(raw.clone()).ok()?;
    if !instance_id_is_valid(&snapshot.instance_id) || snapshot.generation < 1 {
        return None;
    }
    Some(snapshot)
}

/// Truncate to [`MAX_SNAPSHOT_FIELD_CHARS`] UTF-16 code units
/// (registry.ts:262-264). TS slices UTF-16 units and can split a surrogate
/// pair; we stop before the pair instead because Rust strings cannot carry a
/// lone surrogate. Identical for BMP-only strings (session names, cwds).
fn bound_field(value: &str) -> &str {
    let mut units = 0_usize;
    for (i, c) in value.char_indices() {
        if units + c.len_utf16() > MAX_SNAPSHOT_FIELD_CHARS {
            return &value[..i];
        }
        units += c.len_utf16();
    }
    value
}

/// The snapshot as sent on the wire: every free-form string bounded to
/// [`MAX_SNAPSHOT_FIELD_CHARS`] (registry.ts:267-278). Key order is untouched
/// (the TS spread keeps the interface order), so `serde_json` with
/// `preserve_order` serializes byte-identically.
fn bound_snapshot(snapshot: &CollabHostSnapshot) -> CollabHostSnapshot {
    let mut out = snapshot.clone();
    out.session_id = bound_field(&out.session_id).to_owned();
    out.session_name = out
        .session_name
        .as_deref()
        .map(bound_field)
        .map(str::to_owned);
    out.cwd = bound_field(&out.cwd).to_owned();
    out.model = out.model.map(|mut model| {
        model.provider = bound_field(&model.provider).to_owned();
        model.id = bound_field(&model.id).to_owned();
        model
    });
    out
}

/// Read one newline-terminated line with a byte cap.
///
/// Mirrors the TS accumulate-then-check order (server registry.ts:292-299,
/// client :562-566): the size check runs on every chunk BEFORE the newline
/// scan, so a line that carries its `\n` but exceeds `cap` is still rejected.
/// Returns `Ok(Some(line))` (without the `\n`), `Ok(None)` on EOF before any
/// newline, or `Err` when the cap is exceeded (the caller destroys the
/// connection without responding, registry.ts:294).
async fn read_line_capped<S: AsyncRead + Unpin>(
    stream: &mut S,
    cap: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut buf = Vec::with_capacity(512);
    let mut chunk = [0_u8; 1024];
    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > cap {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "line exceeds cap",
            ));
        }
        if let Some(newline) = buf.iter().position(|&b| b == b'\n') {
            buf.truncate(newline);
            return Ok(Some(buf));
        }
    }
}

/// One request per connection: authenticate, dispatch the op, respond, close
/// (registry.ts:280-364). Data after the first newline is ignored (`handled`
/// flag, registry.ts:286-287); an over-cap line destroys the connection
/// without a response.
async fn handle_connection(
    mut stream: tokio::net::UnixStream,
    token: Arc<String>,
    source: Arc<dyn CollabHostRegistrySource>,
) {
    let Ok(Some(line)) = read_line_capped(&mut stream, MAX_REQUEST_BYTES).await else {
        // EOF before a newline, or cap exceeded: destroy without responding
        // (registry.ts:289, :294).
        return;
    };
    let line = String::from_utf8_lossy(&line);
    let line = line.trim();
    if let Some(payload) = process_request(line, &token, source.as_ref()) {
        // `socket.end(...)` parity: write the frame then FIN (registry.ts:284).
        let _ = stream.write_all(payload.as_bytes()).await;
        let _ = stream.shutdown().await;
    }
}

/// Build the single response frame for one request line. Wire error strings
/// are the TS fixed vocabulary; every failure path returns before any URL or
/// source error can leak into the envelope.
fn process_request(
    line: &str,
    token: &str,
    source: &dyn CollabHostRegistrySource,
) -> Option<String> {
    let respond = |payload: serde_json::Value| -> Option<String> {
        let mut out = serde_json::to_string(&payload).expect("registry response serializes");
        out.push('\n');
        Some(out)
    };
    let fail = |error: &str| -> Option<String> {
        respond(serde_json::json!({ "ok": false, "v": COLLAB_REGISTRY_VERSION, "error": error }))
    };
    let request: serde_json::Value = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(_) => return fail("malformed_request"), // registry.ts:302-306
    };
    let Some(request) = request.as_object() else {
        return fail("malformed_request"); // registry.ts:307-310
    };
    // Protocol version is checked BEFORE authentication: a mixed-version peer
    // learns only that, never whether its token was plausible
    // (registry.ts:312-315).
    if request.get("v").and_then(serde_json::Value::as_u64) != Some(COLLAB_REGISTRY_VERSION) {
        return fail("unsupported_protocol");
    }
    if !token_matches(
        token,
        request.get("token").and_then(serde_json::Value::as_str),
    ) {
        return fail("authentication_failed"); // registry.ts:316-319
    }
    // The source snapshot is taken for EVERY op (even before the op is
    // validated) — a host that cannot vouch for its session answers
    // `snapshot_unavailable` whatever was asked (registry.ts:321-327).
    let Ok(snapshot) = source.snapshot() else {
        return fail("snapshot_unavailable");
    };
    match request.get("op").and_then(serde_json::Value::as_str) {
        Some("snapshot") => respond(serde_json::json!({
            "ok": true,
            "v": COLLAB_REGISTRY_VERSION,
            "snapshot": bound_snapshot(&snapshot),
        })),
        Some("link") => {
            let Some(access) = request
                .get("access")
                .and_then(|a| serde_json::from_value::<CollabAccess>(a.clone()).ok())
            else {
                return fail("invalid_access"); // registry.ts:335-338
            };
            // A capability is bound to the exact generation the caller listed:
            // a room that rotated underneath a stale card must not hand out
            // its successor (registry.ts:340-344). A non-number generation is
            // strictly unequal in TS too.
            if request
                .get("generation")
                .and_then(serde_json::Value::as_u64)
                != Some(snapshot.generation)
            {
                return fail("stale_generation");
            }
            if access == CollabAccess::Control && snapshot.access != CollabAccess::Control {
                return fail("access_unavailable"); // registry.ts:345-348
            }
            let Ok(url) = source.link(access) else {
                return fail("snapshot_unavailable"); // registry.ts:350-356
            };
            // TS falsy check: `null` AND `""` both read as "not published"
            // (registry.ts:357-359).
            match url {
                Some(url) if !url.is_empty() => respond(
                    serde_json::json!({ "ok": true, "v": COLLAB_REGISTRY_VERSION, "url": url }),
                ),
                _ => fail("access_unavailable"),
            }
        }
        _ => fail("invalid_operation"), // registry.ts:331-334
    }
}

/// Current uid; `0` where the platform has no uid concept, matching the TS
/// `process.getuid?.() ?? 0` fallback.
#[cfg(unix)]
fn current_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

#[cfg(not(unix))]
fn current_uid() -> u32 {
    0
}

/// Liveness probe: signal 0 succeeds iff the process exists and is
/// signalable. TS treats EPERM the same as ESRCH (`pidAlive` returns false —
/// registry.ts:595-601); mirrored, though with owner-only registry
/// permissions the EPERM case cannot arise in practice.
#[cfg(unix)]
fn pid_alive(pid: i64) -> bool {
    // `kill(pid, None)` is signal 0: existence/signalability probe only.
    // Out-of-i32-range pids cannot exist on this kernel.
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
}

#[cfg(not(unix))]
fn pid_alive(_pid: i64) -> bool {
    false
}

/// The registry must be a real directory; POSIX also verifies its owner.
/// Both publication and listing check this: listing prunes malformed entries,
/// so following a symlink into an unrelated directory would let a planted
/// link turn `omp collab list` into a deletion tool (registry.ts:366-388).
fn assert_private_dir(dir: &Path) -> std::io::Result<()> {
    let stat = fs::symlink_metadata(dir)?;
    validate_private_dir_meta(dir, &stat)
}

/// Shared validation for [`assert_private_dir`]/[`ensure_private_dir`]:
/// reject symlinks, non-directories, and foreign-owned dirs
/// (registry.ts:370-387).
fn validate_private_dir_meta(dir: &Path, stat: &fs::Metadata) -> std::io::Result<()> {
    if stat.is_symlink() {
        return Err(std::io::Error::other(format!(
            "collab registry directory is a symlink: {}",
            dir.display()
        )));
    }
    if !stat.is_dir() {
        return Err(std::io::Error::other(format!(
            "collab registry path is not a directory: {}",
            dir.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let uid = current_uid();
        if stat.uid() != uid {
            return Err(std::io::Error::other(format!(
                "collab registry directory is not owned by the current user: {}",
                dir.display()
            )));
        }
    }
    Ok(())
}

/// Create the directory with owner-only POSIX permissions. `mkdir` with a
/// mode leaves an existing directory's permissions alone, so an
/// already-present directory is tightened explicitly; a symlink or a
/// directory owned by another user is refused rather than published into
/// (registry.ts:390-395).
fn ensure_private_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        // node's recursive mkdir resolves when the path exists (whatever it
        // is) and leaves the verdict to the lstat validation below — a
        // planted file yields "not a directory", not a raw EEXIST
        // (registry.ts:390-395 + :370-374).
        if let Err(err) = builder.create(dir) {
            if err.kind() != ErrorKind::AlreadyExists {
                return Err(err);
            }
        }
    }
    #[cfg(not(unix))]
    fs::create_dir_all(dir)?;
    let stat = fs::symlink_metadata(dir)?;
    validate_private_dir_meta(dir, &stat)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if stat.mode() & 0o077 != 0 {
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        }
    }
    Ok(())
}

/// Lowercase hex, two chars per byte.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Short owner-private socket directory for registries whose canonical path
/// would overflow `sun_path`: the relocation the SSH control sockets use
/// (#9070), keyed by uid and the canonical registry directory
/// (registry.ts:400-411).
fn socket_fallback_dir(dir: &Path, base: &Path) -> PathBuf {
    use ring::digest::{Context, SHA256};
    let uid = current_uid().to_string();
    let mut ctx = Context::new(&SHA256);
    ctx.update(uid.as_bytes());
    ctx.update(b"\0");
    ctx.update(dir.to_string_lossy().as_bytes());
    let key = hex(&ctx.finish().as_ref()[..10]);
    base.join(format!("omp-collab-{key}"))
}

/// Where this publication's Unix socket lives. The canonical location is next
/// to the metadata, but a deep config root (long home directory, nested
/// `PI_CONFIG_DIR`) can push that past `sun_path`, and a host that cannot
/// bind would silently stay absent from `omp collab list`. Listers never
/// guess the relocated path; the metadata records the endpoint
/// (registry.ts:413-437).
fn resolve_socket_endpoint(
    dir: &Path,
    entry_id: &str,
    fallback_base: &Path,
) -> std::io::Result<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let canonical = dir.join(format!("{entry_id}.sock"));
    if canonical.as_os_str().as_bytes().len() < SUN_PATH_LIMIT {
        return Ok(canonical);
    }
    let short_dir = socket_fallback_dir(dir, fallback_base);
    ensure_private_dir(&short_dir)?;
    Ok(short_dir.join(format!("{entry_id}.sock")))
}

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0_u8; bytes];
    rand::rngs::OsRng.fill_bytes(&mut buf);
    hex(&buf)
}

fn now_ms() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(0)
}

/// Remove the metadata file and the endpoint socket, best-effort (registry.ts
/// `removeArtifactsSync`, :515-528). A survivor is pruned by the next list.
fn remove_artifacts(meta_path: &Path, endpoint: &str) {
    let _ = fs::remove_file(meta_path);
    let _ = fs::remove_file(endpoint);
}

/// Handle returned by [`publish_collab_host`]; closing withdraws the host
/// (registry.ts:115-124). Dropping the handle without `close` also withdraws
/// (best-effort equivalent of the TS `process.once("exit")` hook); an abrupt
/// kill leaves artifacts that the next list prunes.
pub struct CollabHostPublication {
    endpoint: String,
    meta_path: PathBuf,
    accept_task: JoinHandle<()>,
    conns: Arc<tokio::sync::Mutex<Vec<JoinHandle<()>>>>,
    closed: AtomicBool,
}

impl CollabHostPublication {
    /// Endpoint the host listens on (test/diagnostic use; not secret).
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Stop serving requests and remove the discovery metadata. Consumes the
    /// handle, which makes the TS idempotence guarantee (registry.ts:119-123)
    /// a compile-time property.
    pub async fn close(self) {
        self.teardown();
        // Abort is synchronous; give the aborted accept loop one scheduler
        // turn to wind down, mirroring the TS await of the server-close
        // callback (registry.ts:522-530) without moving the `JoinHandle` out
        // of a `Drop` type.
        tokio::task::yield_now().await;
    }

    fn teardown(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.accept_task.abort();
        if let Ok(mut live) = self.conns.try_lock() {
            for handle in live.drain(..) {
                handle.abort();
            }
        }
        remove_artifacts(&self.meta_path, &self.endpoint);
    }
}

impl std::fmt::Debug for CollabHostPublication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollabHostPublication")
            .field("endpoint", &self.endpoint)
            .field("meta_path", &self.meta_path)
            .field("accept_task", &self.accept_task)
            .field("conns", &self.conns)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .finish()
    }
}

impl Drop for CollabHostPublication {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// Publish a live Collab host to the local registry.
///
/// Creates the owner-only runtime dir, starts a private IPC endpoint backed
/// by `source`, and writes discovery metadata (never URLs or room secrets)
/// (registry.ts:439-531). Call [`CollabHostPublication::close`] on every
/// teardown path; the OS closing the endpoint covers crashes.
///
/// # Errors
/// Propagates directory creation/validation failures, bind failures, and
/// metadata write failures; on any post-bind failure the listener is closed
/// and the socket file removed again (registry.ts:498-503).
///
/// # Panics
/// Only if `serde_json` cannot serialize the metadata record, which is
/// statically impossible (all fields are strings and integers).
pub fn publish_collab_host(
    source: Arc<dyn CollabHostRegistrySource>,
    options: Option<&CollabPublishOptions>,
) -> anyhow::Result<CollabHostPublication> {
    let options = options.cloned().unwrap_or_default();
    let dir = options.dir.unwrap_or_else(collab_hosts_runtime_dir);
    ensure_private_dir(&dir).map_err(anyhow::Error::from)?;

    let instance_id = options.instance_id.unwrap_or_else(|| random_hex(8));
    if !instance_id_is_valid(&instance_id) {
        anyhow::bail!("invalid collab registry instance id"); // registry.ts:446-448
    }
    // Unpredictable per-publication entry ID names the endpoint and the
    // metadata file: PID reuse cannot attach stale metadata to an unrelated
    // process, and a room that rotates never shares artifact names with its
    // predecessor, so a lister pruning the stale entry can never remove the
    // live successor's (registry.ts:449-454).
    let entry_id = random_hex(8);
    let token = random_hex(32);
    let fallback_base = options
        .socket_fallback_base
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET_FALLBACK_BASE));
    let endpoint =
        resolve_socket_endpoint(&dir, &entry_id, &fallback_base).map_err(anyhow::Error::from)?;
    let meta_path = dir.join(format!("{entry_id}.json"));

    let listener = UnixListener::bind(&endpoint).map_err(|err| {
        anyhow::Error::new(err).context(format!(
            "bind collab registry socket {}",
            endpoint.display()
        ))
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&endpoint, fs::Permissions::from_mode(0o600)) // registry.ts:471
            .map_err(anyhow::Error::from)?;
    }

    let meta = CollabHostRecord {
        version: COLLAB_REGISTRY_VERSION,
        pid: i64::from(std::process::id()),
        instance_id,
        endpoint: endpoint.to_string_lossy().into_owned(),
        created_at: now_ms(),
        token: token.clone(),
    };
    // Write-then-rename so a concurrent list never observes a partial file
    // (it would classify the entry as malformed and prune it, leaving this
    // host published but undiscoverable). The temp suffix keeps it outside
    // the `*.json` listing filter; the entry ID makes the name unique. Any
    // failure after the exclusive create removes the temp file again
    // (registry.ts:500-512).
    let tmp_path = dir.join(format!("{entry_id}.json.tmp"));
    let write_meta = || -> std::io::Result<()> {
        use std::{io::Write as _, os::unix::fs::OpenOptionsExt};
        let mut handle = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp_path)?;
        handle.write_all(
            serde_json::to_string(&meta)
                .expect("registry metadata serializes")
                .as_bytes(),
        )?;
        drop(handle);
        fs::rename(&tmp_path, &meta_path)
    };
    if let Err(err) = write_meta() {
        let _ = fs::remove_file(&tmp_path);
        drop(listener);
        let _ = fs::remove_file(&endpoint);
        return Err(anyhow::Error::new(err).context("write collab registry metadata"));
    }

    let token = Arc::new(token);
    let conns: Arc<tokio::sync::Mutex<Vec<JoinHandle<()>>>> = Arc::default();
    let accept_task = tokio::spawn(accept_loop(listener, token, source, Arc::clone(&conns)));

    Ok(CollabHostPublication {
        endpoint: meta.endpoint,
        meta_path,
        accept_task,
        conns,
        closed: AtomicBool::new(false),
    })
}

/// Accept loop for one publication. Accept errors abort the loop except the
/// transient per-connection resets (TS keeps the server alive across accept
/// errors; a persistently failing accept is fatal here rather than spinning).
async fn accept_loop(
    listener: UnixListener,
    token: Arc<String>,
    source: Arc<dyn CollabHostRegistrySource>,
    conns: Arc<tokio::sync::Mutex<Vec<JoinHandle<()>>>>,
) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let handle = tokio::spawn(handle_connection(
                    stream,
                    Arc::clone(&token),
                    Arc::clone(&source),
                ));
                let mut live = conns.lock().await;
                live.retain(|handle| !handle.is_finished());
                live.push(handle);
            }
            // Transient per-connection resets: keep serving (TS keeps the
            // server alive across accept errors).
            Err(err)
                if matches!(
                    err.kind(),
                    ErrorKind::ConnectionAborted | ErrorKind::Interrupted
                ) => {}
            Err(_) => break,
        }
    }
}

/// Query outcome for one endpoint (registry.ts:532). `Dead` is reserved for
/// the two errno classes that mean "the host is gone"; everything else is a
/// non-committal `Skip` so a live host is never pruned on a fluke.
enum QueryResult {
    Ok(serde_json::Value),
    Dead,
    Skip { error: Option<String> },
}

/// Query one endpoint: connect, authenticate, send one request, read one
/// bounded response line (registry.ts:535-586). The deadline covers
/// connect+write+read, like the single TS timer.
async fn query(
    meta: &CollabHostRecord,
    request: serde_json::Value,
    timeout_ms: u64,
) -> QueryResult {
    match timeout(Duration::from_millis(timeout_ms), query_once(meta, request)).await {
        Ok(result) => result,
        Err(_) => QueryResult::Skip { error: None },
    }
}

async fn query_once(meta: &CollabHostRecord, request: serde_json::Value) -> QueryResult {
    let mut stream = match tokio::net::UnixStream::connect(&meta.endpoint).await {
        Ok(stream) => stream,
        // Endpoints die with their host process: a refused or missing socket
        // means the host is gone. Any other error (EMFILE, EACCES, EAGAIN, …)
        // says nothing about liveness and must not prune a live host
        // (registry.ts:548-552).
        Err(err)
            if matches!(
                err.kind(),
                ErrorKind::NotFound | ErrorKind::ConnectionRefused
            ) =>
        {
            return QueryResult::Dead;
        }
        Err(_) => return QueryResult::Skip { error: None },
    };
    // `{ v, token, ...request }` in TS insertion order (registry.ts:556-558).
    let mut frame = serde_json::Map::new();
    frame.insert("v".into(), serde_json::json!(COLLAB_REGISTRY_VERSION));
    frame.insert("token".into(), serde_json::json!(meta.token));
    if let serde_json::Value::Object(rest) = request {
        frame.extend(rest);
    }
    let mut line = serde_json::to_string(&serde_json::Value::Object(frame))
        .expect("registry request serializes");
    line.push('\n');
    if stream.write_all(line.as_bytes()).await.is_err() {
        return QueryResult::Skip { error: None };
    }
    let Ok(Some(response)) = read_line_capped(&mut stream, MAX_RESPONSE_BYTES).await else {
        return QueryResult::Skip { error: None };
    };
    let response: serde_json::Value = match serde_json::from_slice(&response) {
        Ok(response) => response,
        Err(_) => return QueryResult::Skip { error: None },
    };
    let Some(response) = response.as_object() else {
        return QueryResult::Skip { error: None };
    };
    if response.get("ok") != Some(&serde_json::Value::Bool(true)) {
        // Authentication failure or structured error: an unrelated endpoint
        // cannot satisfy stale metadata without the matching token
        // (registry.ts:573-578).
        return QueryResult::Skip {
            error: response
                .get("error")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        };
    }
    QueryResult::Ok(serde_json::Value::Object(response.clone()))
}

async fn query_snapshot(
    meta: &CollabHostRecord,
    timeout_ms: u64,
) -> Result<CollabHostSnapshot, QueryResult> {
    let result = query(meta, serde_json::json!({ "op": "snapshot" }), timeout_ms).await;
    let QueryResult::Ok(value) = result else {
        return Err(result);
    };
    match value.get("snapshot").and_then(parse_snapshot) {
        Some(snapshot) => Ok(snapshot),
        None => Err(QueryResult::Skip { error: None }),
    }
}

/// Remove one stale entry. Artifact names are unique per publication, so the
/// metadata and endpoint observed dead can only belong to that publication; a
/// room that rotated meanwhile lives under different names and is untouched
/// (registry.ts:603-622). Only sockets this registry could have created are
/// unlinked: beside the metadata, or in its own relocated socket directory.
fn prune_entry(dir: &Path, name: &str, meta: Option<&CollabHostRecord>) {
    let _ = fs::remove_file(dir.join(name));
    let Some(meta) = meta else { return };
    let endpoint = Path::new(&meta.endpoint);
    // `Path::starts_with` compares whole components, the strict form of the
    // TS `startsWith(dir + path.sep)` string check (registry.ts:616-619).
    let owns_endpoint = endpoint.starts_with(dir)
        || endpoint.starts_with(socket_fallback_dir(
            dir,
            Path::new(DEFAULT_SOCKET_FALLBACK_BASE),
        ));
    if owns_endpoint {
        let _ = fs::remove_file(endpoint);
    }
}

/// One listed publication: its metadata plus the snapshot it served
/// (registry.ts:625-628).
struct LiveEntry {
    meta: CollabHostRecord,
    snapshot: CollabHostSnapshot,
}

/// Read and classify one discovery entry (registry.ts:630-654). Malformed
/// metadata is pruned immediately; a foreign protocol version is shown to
/// nobody and pruned only once its owning process is gone so newer versions
/// keep their state; a dead endpoint is pruned; unreachable is left alone.
async fn list_entry(dir: &Path, name: &str, timeout_ms: u64) -> Option<LiveEntry> {
    let text = fs::read_to_string(dir.join(name)).ok()?;
    let Some(meta) = parse_collab_host_record(&text) else {
        // Malformed metadata can never become listable; remove it.
        prune_entry(dir, name, None);
        return None;
    };
    if meta.version != COLLAB_REGISTRY_VERSION {
        // A different omp version owns this entry. Never show it; prune only
        // once the owning process is gone so newer versions keep their state.
        if !pid_alive(meta.pid) {
            prune_entry(dir, name, Some(&meta));
        }
        return None;
    }
    match query_snapshot(&meta, timeout_ms).await {
        Ok(snapshot) => Some(LiveEntry { meta, snapshot }),
        Err(QueryResult::Dead) => {
            prune_entry(dir, name, Some(&meta));
            None
        }
        Err(_) => None,
    }
}

/// List every live publication under `dir` (registry.ts:655-694): bounded
/// worker pool, short independent deadlines, sorted by start time, PID, then
/// instance ID.
async fn list_live_entries(options: Option<&CollabListOptions>) -> anyhow::Result<Vec<LiveEntry>> {
    let options = options.cloned().unwrap_or_default();
    let dir = options.dir.unwrap_or_else(collab_hosts_runtime_dir);
    let timeout_ms = options.timeout_ms.unwrap_or(DEFAULT_QUERY_TIMEOUT_MS);
    if let Err(err) = assert_private_dir(&dir) {
        if err.kind() == ErrorKind::NotFound {
            return Ok(Vec::new());
        }
        return Err(anyhow::Error::from(err));
    }
    let mut names: Vec<String> = fs::read_dir(&dir)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    #[expect(
        clippy::case_sensitive_file_extension_comparisons,
        reason = "TS parity: registry.ts:666 filters `*.json` case-sensitively"
    )]
    names.retain(|name| name.ends_with(".json"));
    let mut tasks: JoinSet<Option<LiveEntry>> = JoinSet::new();
    let permits = Arc::new(tokio::sync::Semaphore::new(LIST_CONCURRENCY));
    for name in names {
        let dir = dir.clone();
        let permits = Arc::clone(&permits);
        tasks.spawn(async move {
            let _permit = permits.acquire().await;
            list_entry(&dir, &name, timeout_ms).await
        });
    }
    let mut live = Vec::new();
    while let Some(joined) = tasks.join_next().await {
        if let Ok(Some(entry)) = joined {
            live.push(entry);
        }
    }
    // registry.ts:679-684 — `localeCompare` reduced to byte order (ids are
    // `[a-z0-9-]`).
    live.sort_by(|a, b| {
        a.snapshot
            .started_at
            .cmp(&b.snapshot.started_at)
            .then(a.snapshot.pid.cmp(&b.snapshot.pid))
            .then(a.snapshot.instance_id.cmp(&b.snapshot.instance_id))
    });
    Ok(live)
}

/// List live Collab hosts under this config root.
///
/// Reads every discovery entry, queries the live hosts concurrently (bounded,
/// short independent deadlines), prunes stale or malformed entries
/// best-effort, and returns healthy hosts sorted by start time, PID, then
/// instance ID. Unreachable, unauthenticated, malformed, or version-mismatched
/// entries are omitted without failing the listing. The result carries no
/// URLs (registry.ts:696-704).
///
/// # Errors
/// Only directory validation/read failures other than a missing registry dir
/// (which lists empty) propagate.
pub async fn list_collab_hosts(
    options: Option<&CollabListOptions>,
) -> anyhow::Result<Vec<CollabHostSnapshot>> {
    Ok(list_live_entries(options)
        .await?
        .into_iter()
        .map(|entry| entry.snapshot)
        .collect())
}

/// Resolve one browser URL for the host selected by `selector` — an exact
/// instance ID, or a PID when no instance matches. The link request carries
/// the generation observed while listing, so a host that rotated rooms in
/// between answers `stale_generation` instead of leaking its successor's
/// capability (registry.ts:706-762).
///
/// # Errors
/// [`CollabLinkError`] for every classified failure (recover the code with
/// `err.downcast_ref::<CollabLinkError>()`); directory I/O failures other
/// than a missing registry dir propagate as plain errors.
pub async fn resolve_collab_host_link(
    selector: &str,
    access: CollabAccess,
    options: Option<&CollabListOptions>,
) -> anyhow::Result<CollabResolvedLink> {
    let wanted = selector.trim();
    let live = list_live_entries(options).await?;
    let mut matches: Vec<&LiveEntry> = live
        .iter()
        .filter(|entry| entry.snapshot.instance_id == wanted)
        .collect();
    if matches.is_empty() {
        // `/^[1-9][0-9]*$/` selector: exact PID match, no leading zeros.
        let is_pid = wanted
            .as_bytes()
            .first()
            .is_some_and(|&b| b.is_ascii_digit() && b != b'0')
            && wanted.bytes().all(|b| b.is_ascii_digit());
        if is_pid {
            if let Ok(pid) = wanted.parse::<i64>() {
                matches = live
                    .iter()
                    .filter(|entry| entry.snapshot.pid == pid)
                    .collect();
            }
        }
    }
    if matches.is_empty() {
        return Err(anyhow::Error::new(CollabLinkError::new(
            CollabLinkErrorCode::NotFound,
            format!("no active Collab host matches {wanted}"),
        )));
    }
    if matches.len() > 1 {
        let ids = matches
            .iter()
            .map(|entry| entry.snapshot.instance_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow::Error::new(CollabLinkError::new(
            CollabLinkErrorCode::Ambiguous,
            format!("{wanted} matches more than one Collab host; use an instance id: {ids}"),
        )));
    }
    let entry = matches[0];
    // No local access precheck: the host decides, and it checks the generation
    // before the access level, so a room that rotated underneath the listing
    // reports `stale_generation` rather than a verdict about its predecessor
    // (registry.ts:735-737).
    let timeout_ms = options
        .and_then(|options| options.timeout_ms)
        .unwrap_or(DEFAULT_QUERY_TIMEOUT_MS);
    let request = serde_json::json!({
        "op": "link",
        "access": access,
        "generation": entry.snapshot.generation,
    });
    let snapshot = &entry.snapshot;
    let instance_id = snapshot.instance_id.as_str();
    match query(&entry.meta, request, timeout_ms).await {
        QueryResult::Ok(value) => {
            // TS falsy check on `url`: empty string is an invalid response
            // (registry.ts:739-746).
            return match value.get("url").and_then(serde_json::Value::as_str) {
                Some(url) if !url.is_empty() => Ok(CollabResolvedLink {
                    instance_id: instance_id.to_owned(),
                    generation: snapshot.generation,
                    access,
                    url: url.to_owned(),
                }),
                _ => Err(anyhow::Error::new(CollabLinkError::new(
                    CollabLinkErrorCode::Unreachable,
                    format!("host {instance_id} returned an invalid link response"),
                ))),
            };
        }
        QueryResult::Skip { error: Some(error) } if error == "stale_generation" => {
            return Err(anyhow::Error::new(CollabLinkError::new(
                CollabLinkErrorCode::StaleGeneration,
                format!("host {instance_id} started a new room since it was listed; list again and retry"),
            )));
        }
        QueryResult::Skip { error: Some(error) } if error == "access_unavailable" => {
            return Err(anyhow::Error::new(CollabLinkError::new(
                CollabLinkErrorCode::AccessUnavailable,
                format!("host {instance_id} does not publish {access} access"),
            )));
        }
        QueryResult::Dead => {
            // Every room generation has its own endpoint, so a host that
            // rotated since the listing is simply gone from this one rather
            // than answering `stale_generation` itself. Look the instance up
            // again before giving up (registry.ts:749-761).
            let rotated = list_live_entries(options).await?.iter().any(|later| {
                later.snapshot.instance_id == instance_id
                    && later.snapshot.generation > snapshot.generation
            });
            if rotated {
                return Err(anyhow::Error::new(CollabLinkError::new(
                    CollabLinkErrorCode::StaleGeneration,
                    format!("host {instance_id} started a new room since it was listed; list again and retry"),
                )));
            }
        }
        QueryResult::Skip { .. } => {}
    }
    Err(anyhow::Error::new(CollabLinkError::new(
        CollabLinkErrorCode::Unreachable,
        format!("host {instance_id} did not answer the link request"),
    )))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    const FIXTURE_SNAPSHOT_JSON: &str = "{\"instanceId\":\"fixture-host-01\",\"generation\":3,\
         \"pid\":4242,\"sessionId\":\"sess-abc\",\"sessionName\":\"Fix Session\",\
         \"cwd\":\"/tmp/proj\",\"model\":{\"provider\":\"prov\",\"id\":\"m-1\"},\
         \"startedAt\":1700000000000,\"participants\":2,\"relayConnected\":true,\
         \"inputRequired\":false,\"busy\":true,\"access\":\"control\"}";

    /// In-memory [`CollabHostRegistrySource`] standing in for a live host
    /// (host.ts is T3's; the registry only sees this trait).
    struct StubSource {
        state: std::sync::Mutex<StubState>,
    }

    struct StubState {
        snapshot: CollabHostSnapshot,
        fail_snapshot: bool,
        fail_link: bool,
        links: HashMap<CollabAccess, String>,
        unpublished: HashSet<CollabAccess>,
    }

    impl StubSource {
        fn new(snapshot: CollabHostSnapshot) -> Self {
            Self {
                state: std::sync::Mutex::new(StubState {
                    snapshot,
                    fail_snapshot: false,
                    fail_link: false,
                    links: HashMap::new(),
                    unpublished: HashSet::new(),
                }),
            }
        }

        fn with_link(self, access: CollabAccess, url: &str) -> Self {
            self.state
                .lock()
                .expect("stub lock")
                .links
                .insert(access, url.to_owned());
            self
        }

        fn set_access(&self, access: CollabAccess) {
            self.state.lock().expect("stub lock").snapshot.access = access;
        }

        fn unpublish(&self, access: CollabAccess) {
            self.state
                .lock()
                .expect("stub lock")
                .unpublished
                .insert(access);
        }

        fn fail_snapshot(&self, on: bool) {
            self.state.lock().expect("stub lock").fail_snapshot = on;
        }

        fn fail_link(&self, on: bool) {
            self.state.lock().expect("stub lock").fail_link = on;
        }
    }

    impl CollabHostRegistrySource for StubSource {
        fn snapshot(&self) -> anyhow::Result<CollabHostSnapshot> {
            let state = self.state.lock().expect("stub lock");
            if state.fail_snapshot {
                anyhow::bail!("host cannot vouch for its session");
            }
            Ok(state.snapshot.clone())
        }

        fn link(&self, access: CollabAccess) -> anyhow::Result<Option<String>> {
            let state = self.state.lock().expect("stub lock");
            if state.fail_link {
                anyhow::bail!("url builder unavailable");
            }
            if state.unpublished.contains(&access) {
                return Ok(None);
            }
            Ok(state.links.get(&access).cloned())
        }
    }

    fn fixture_snapshot() -> CollabHostSnapshot {
        CollabHostSnapshot {
            instance_id: "fixture-host-01".into(),
            generation: 3,
            pid: 4242,
            session_id: "sess-abc".into(),
            session_name: Some("Fix Session".into()),
            cwd: "/tmp/proj".into(),
            model: Some(CollabHostModel {
                provider: "prov".into(),
                id: "m-1".into(),
            }),
            started_at: 1_700_000_000_000,
            participants: 2,
            relay_connected: true,
            input_required: false,
            busy: Some(true),
            access: CollabAccess::Control,
        }
    }

    /// Publish a fixture host into `dir` under the fixture instance id.
    fn publish_fixture(dir: &Path) -> (Arc<StubSource>, CollabHostPublication) {
        let source = Arc::new(
            StubSource::new(fixture_snapshot())
                .with_link(CollabAccess::Control, "https://relay.test/#ctl")
                .with_link(CollabAccess::View, "https://relay.test/#view"),
        );
        let options = CollabPublishOptions {
            dir: Some(dir.to_path_buf()),
            instance_id: Some("fixture-host-01".into()),
            ..CollabPublishOptions::default()
        };
        let publication = publish_collab_host(source.clone(), Some(&options)).expect("publish");
        (source, publication)
    }

    fn list_opts(dir: &Path, timeout_ms: u64) -> CollabListOptions {
        CollabListOptions {
            dir: Some(dir.to_path_buf()),
            timeout_ms: Some(timeout_ms),
        }
    }

    /// One raw request over the IPC socket; `None` when the connection closes
    /// without emitting a line (the TS destroy semantics).
    async fn raw_request(endpoint: &Path, line: &str) -> Option<String> {
        let mut stream = tokio::net::UnixStream::connect(endpoint)
            .await
            .expect("connect");
        stream.write_all(line.as_bytes()).await.expect("write");
        // Ok(None) = connection closed without a line (the TS destroy
        // semantics the caller asserts on); Err likewise counts as no line.
        let response = timeout(
            Duration::from_secs(5),
            read_line_capped(&mut stream, MAX_RESPONSE_BYTES),
        )
        .await
        .expect("no hang")
        .ok()
        .flatten()?;
        Some(String::from_utf8_lossy(&response).into_owned())
    }

    fn write_record(dir: &Path, name: &str, record: &CollabHostRecord) {
        fs::write(
            dir.join(name),
            serde_json::to_string(record).expect("record serializes"),
        )
        .expect("write record");
    }

    fn record(
        instance_id: &str,
        endpoint: &Path,
        token: &str,
        pid: i64,
        version: u64,
    ) -> CollabHostRecord {
        CollabHostRecord {
            version,
            instance_id: instance_id.into(),
            pid,
            endpoint: endpoint.to_string_lossy().into_owned(),
            created_at: now_ms(),
            token: token.into(),
        }
    }

    /// A pid that is definitely reaped: run `true` and wait for it.
    fn dead_pid() -> i64 {
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn true");
        let pid = i64::from(child.id());
        child.wait().expect("reap true");
        pid
    }

    #[tokio::test]
    async fn metadata_and_socket_layout() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("hosts");
        let (_source, publication) = publish_fixture(&dir);

        let meta_path = fs::read_dir(&dir)
            .expect("readdir")
            .filter_map(std::result::Result::ok)
            .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
            .expect("metadata file")
            .path();
        let name = meta_path
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        // Per-publication unpredictable entry id, 16 hex chars + `.json`
        // (registry.ts:449-454).
        assert_eq!(name.len(), 21, "entry name {name}");
        assert!(name.bytes().take(16).all(|b| b.is_ascii_hexdigit()));

        let text = fs::read_to_string(&meta_path).expect("read meta");
        // Wire field order: version, instanceId, pid, … (registry.ts:170-177).
        assert!(
            text.starts_with("{\"version\":1,\"instanceId\":\"fixture-host-01\",\"pid\":"),
            "meta bytes: {text}"
        );
        let parsed = parse_collab_host_record(&text).expect("parsed record");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.instance_id, "fixture-host-01");
        assert_eq!(parsed.endpoint, publication.endpoint());
        assert_eq!(parsed.token.len(), 64, "32 random bytes as hex");

        // Owner-only permissions on dir, socket, and metadata.
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let dir_mode = fs::metadata(&dir).expect("dir").mode() & 0o777;
            assert_eq!(dir_mode, 0o700, "registry dir mode {dir_mode:o}");
            let sock_mode = fs::metadata(publication.endpoint()).expect("sock").mode() & 0o777;
            assert_eq!(sock_mode, 0o600, "socket mode {sock_mode:o}");
            let meta_mode = fs::metadata(&meta_path).expect("meta").mode() & 0o777;
            assert_eq!(meta_mode, 0o600, "meta mode {meta_mode:o}");
        }
        // Write-then-rename leaves no temp file behind (registry.ts:500-512).
        assert!(
            !fs::read_dir(&dir).expect("readdir").any(|e| e
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")),
            "no tmp survivor"
        );
    }

    #[tokio::test]
    async fn snapshot_roundtrip_via_list() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());
        let listed = list_collab_hosts(Some(&list_opts(tmp.path(), 1_000)))
            .await
            .expect("list");
        assert_eq!(listed, vec![fixture_snapshot()]);
        publication.close().await;
    }

    #[tokio::test]
    async fn snapshot_wire_response_golden() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());
        let record = {
            let text = fs::read_dir(tmp.path())
                .expect("readdir")
                .filter_map(std::result::Result::ok)
                .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
                .expect("meta")
                .path();
            parse_collab_host_record(&fs::read_to_string(text).expect("text")).expect("record")
        };
        let response = raw_request(
            Path::new(publication.endpoint()),
            &format!(
                "{{\"v\":1,\"token\":\"{}\",\"op\":\"snapshot\"}}\n",
                record.token
            ),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            format!("{{\"ok\":true,\"v\":1,\"snapshot\":{FIXTURE_SNAPSHOT_JSON}}}")
        );
        publication.close().await;
    }

    #[tokio::test]
    async fn link_grant_by_instance_and_pid() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());

        // Instance selector, with surrounding whitespace (TS trims,
        // registry.ts:709).
        let resolved = resolve_collab_host_link(
            "  fixture-host-01  ",
            CollabAccess::Control,
            Some(&list_opts(tmp.path(), 1_000)),
        )
        .await
        .expect("resolve");
        assert_eq!(
            resolved,
            CollabResolvedLink {
                instance_id: "fixture-host-01".into(),
                generation: 3,
                access: CollabAccess::Control,
                url: "https://relay.test/#ctl".into(),
            }
        );

        // Numeric selector falls back to an exact snapshot PID match — the
        // stub advertises pid 4242, not the real process id.
        let resolved = resolve_collab_host_link(
            "4242",
            CollabAccess::View,
            Some(&list_opts(tmp.path(), 1_000)),
        )
        .await
        .expect("resolve by pid");
        assert_eq!(resolved.url, "https://relay.test/#view");
        publication.close().await;
    }

    #[tokio::test]
    async fn stale_generation_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());

        // Host-side: a link request carrying a stale generation is rejected
        // before the source is consulted (registry.ts:340-344).
        let record_token = parse_collab_host_record(
            &fs::read_to_string(
                fs::read_dir(tmp.path())
                    .expect("readdir")
                    .filter_map(std::result::Result::ok)
                    .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
                    .expect("meta")
                    .path(),
            )
            .expect("text"),
        )
        .expect("record")
        .token;
        let response = raw_request(
            Path::new(publication.endpoint()),
            &format!(
                "{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"link\",\"access\":\"control\",\
                 \"generation\":99}}\n"
            ),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"stale_generation\"}"
        );

        // The resolver-level stale path — a host whose listed generation
        // dies MID-RESOLVE and whose successor answers — is covered by
        // `resolve_reports_rotation_after_dead_endpoint`; a plain rotate
        // between two resolve calls is invisible because resolve re-lists
        // fresh on every call (registry.ts:709-733).
        publication.close().await;
    }

    #[tokio::test]
    async fn access_unavailable_when_not_published() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (source, publication) = publish_fixture(tmp.path());
        // The view-only host: snapshot says the ceiling is `view`, even though
        // a control URL technically exists in the source (registry.ts:345-348
        // gates before consulting the source).
        source.set_access(CollabAccess::View);

        let record_token = parse_collab_host_record(
            &fs::read_to_string(
                fs::read_dir(tmp.path())
                    .expect("readdir")
                    .filter_map(std::result::Result::ok)
                    .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
                    .expect("meta")
                    .path(),
            )
            .expect("text"),
        )
        .expect("record")
        .token;
        let response = raw_request(
            Path::new(publication.endpoint()),
            &format!(
                "{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"link\",\"access\":\"control\",\
                 \"generation\":3}}\n"
            ),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"access_unavailable\"}"
        );

        let err = resolve_collab_host_link(
            "fixture-host-01",
            CollabAccess::Control,
            Some(&list_opts(tmp.path(), 1_000)),
        )
        .await
        .expect_err("unavailable");
        assert_eq!(
            err.downcast_ref::<CollabLinkError>().expect("typed").code,
            CollabLinkErrorCode::AccessUnavailable
        );

        // View access still resolves against a view-only host.
        let resolved = resolve_collab_host_link(
            "fixture-host-01",
            CollabAccess::View,
            Some(&list_opts(tmp.path(), 1_000)),
        )
        .await
        .expect("view ok");
        assert_eq!(resolved.url, "https://relay.test/#view");
        publication.close().await;
    }

    #[tokio::test]
    async fn bearer_rejected_and_version_precedence() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());
        let endpoint = Path::new(publication.endpoint());

        // Wrong token.
        let response = raw_request(
            endpoint,
            "{\"v\":1,\"token\":\"deadbeef\",\"op\":\"snapshot\"}\n",
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"authentication_failed\"}"
        );

        // Missing token (undefined never matches, registry.ts:206-212).
        let response = raw_request(endpoint, "{\"v\":1,\"op\":\"snapshot\"}\n")
            .await
            .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"authentication_failed\"}"
        );

        // Version gate runs BEFORE authentication (registry.ts:312-319): a
        // wrong version AND wrong token answers unsupported_protocol.
        let response = raw_request(
            endpoint,
            "{\"v\":2,\"token\":\"deadbeef\",\"op\":\"snapshot\"}\n",
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"unsupported_protocol\"}"
        );
        publication.close().await;
    }

    #[tokio::test]
    async fn malformed_and_unknown_ops() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());
        let record_token = parse_collab_host_record(
            &fs::read_to_string(
                fs::read_dir(tmp.path())
                    .expect("readdir")
                    .filter_map(std::result::Result::ok)
                    .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
                    .expect("meta")
                    .path(),
            )
            .expect("text"),
        )
        .expect("record")
        .token;
        let endpoint = Path::new(publication.endpoint());

        let response = raw_request(endpoint, "not json\n").await.expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"malformed_request\"}"
        );

        // Valid JSON, not an object (registry.ts:307-310).
        let response = raw_request(endpoint, "[1,2]\n").await.expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"malformed_request\"}"
        );

        let response = raw_request(
            endpoint,
            &format!("{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"zzz\"}}\n"),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"invalid_operation\"}"
        );

        // Link without a usable access value (registry.ts:335-338).
        let response = raw_request(
            endpoint,
            &format!("{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"link\",\"generation\":3}}\n"),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"invalid_access\"}"
        );

        // Second request on the SAME connection is ignored: one request per
        // connection (registry.ts:280-287, `handled` flag).
        let mut stream = tokio::net::UnixStream::connect(endpoint)
            .await
            .expect("connect");
        let two = format!(
            "{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"snapshot\"}}\n\
             {{\"v\":1,\"token\":\"{record_token}\",\"op\":\"snapshot\"}}\n"
        );
        stream.write_all(two.as_bytes()).await.expect("write");
        let first = timeout(
            Duration::from_secs(5),
            read_line_capped(&mut stream, MAX_RESPONSE_BYTES),
        )
        .await
        .expect("no hang")
        .expect("read")
        .expect("one response");
        assert!(String::from_utf8_lossy(&first).contains("\"snapshot\""));
        assert!(
            timeout(
                Duration::from_millis(300),
                read_line_capped(&mut stream, MAX_RESPONSE_BYTES)
            )
            .await
            .expect("no hang")
            .expect("eof")
            .is_none(),
            "no second response"
        );
        publication.close().await;
    }

    #[tokio::test]
    async fn oversized_request_destroyed_silently() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (_source, publication) = publish_fixture(tmp.path());
        // 4 KiB of line with the newline attached: still destroyed, no
        // response, connection dropped (registry.ts:292-299).
        let line = format!("{}\n", "a".repeat(MAX_REQUEST_BYTES + 8));
        assert_eq!(
            raw_request(Path::new(publication.endpoint()), &line).await,
            None
        );
        // A valid request on a fresh connection still works afterwards.
        let record_token = parse_collab_host_record(
            &fs::read_to_string(
                fs::read_dir(tmp.path())
                    .expect("readdir")
                    .filter_map(std::result::Result::ok)
                    .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
                    .expect("meta")
                    .path(),
            )
            .expect("text"),
        )
        .expect("record")
        .token;
        let response = raw_request(
            Path::new(publication.endpoint()),
            &format!("{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"snapshot\"}}\n"),
        )
        .await
        .expect("still serving");
        assert!(response.contains("\"ok\":true"));
        publication.close().await;
    }

    #[tokio::test]
    async fn dead_endpoints_pruned() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        // ECONNREFUSED: a socket file whose listener is gone.
        let refused = dir.join("refused.sock");
        drop(UnixListener::bind(&refused).expect("bind"));
        write_record(
            dir,
            "refused.json",
            &record("dead-host-01", &refused, "tok", dead_pid(), 1),
        );

        // ENOENT: metadata pointing at a socket path that never existed.
        let missing = dir.join("missing.sock");
        write_record(
            dir,
            "missing.json",
            &record("dead-host-02", &missing, "tok", dead_pid(), 1),
        );

        let listed = list_collab_hosts(Some(&list_opts(dir, 1_000)))
            .await
            .expect("list");
        assert!(listed.is_empty(), "dead entries are not hosts");
        // Both metadata files pruned; the owned socket file unlinked too
        // (registry.ts:603-622).
        assert!(!dir.join("refused.json").exists());
        assert!(!dir.join("missing.json").exists());
        assert!(!refused.exists());
    }

    #[tokio::test]
    async fn malformed_metadata_pruned() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        fs::write(dir.join("a.json"), "not json").expect("write");
        // Valid JSON but instance id violates `^[a-z0-9-]{8,64}$`.
        let endpoint = dir.join("never.sock");
        write_record(
            dir,
            "b.json",
            &record("BAD_INSTANCE", &endpoint, "tok", dead_pid(), 1),
        );

        let listed = list_collab_hosts(Some(&list_opts(dir, 1_000)))
            .await
            .expect("list");
        assert!(listed.is_empty());
        assert!(!dir.join("a.json").exists());
        assert!(!dir.join("b.json").exists());
        // A malformed record prunes only its metadata file, never a socket it
        // does not own (registry.ts:609-615 — meta === null skips the unlink).
        // (endpoint never existed here, so just assert the directory has no
        // leftovers besides nothing.)
    }

    #[tokio::test]
    async fn foreign_version_kept_while_pid_alive_pruned_when_dead() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let endpoint = dir.join("foreign.sock");

        // A newer omp owns this entry: never listed, never pruned while its
        // process lives (registry.ts:641-648).
        write_record(
            dir,
            "alive.json",
            &record(
                "future-host-01",
                &endpoint,
                "tok",
                i64::from(std::process::id()),
                2,
            ),
        );
        // Same version, but its process is gone → safe to prune.
        write_record(
            dir,
            "dead.json",
            &record("future-host-02", &endpoint, "tok", dead_pid(), 2),
        );

        let listed = list_collab_hosts(Some(&list_opts(dir, 1_000)))
            .await
            .expect("list");
        assert!(listed.is_empty(), "foreign versions are invisible");
        assert!(
            dir.join("alive.json").exists(),
            "live owner keeps its state"
        );
        assert!(!dir.join("dead.json").exists(), "dead owner's entry pruned");
    }

    #[tokio::test]
    async fn unreachable_entry_skipped_but_kept() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        // A listener that accepts but never answers: unreachable is not dead
        // (registry.ts:548-552) — the entry must survive the listing.
        let silent = dir.join("silent.sock");
        let listener = UnixListener::bind(&silent).expect("bind");
        let parker = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_mins(1)).await;
                    drop(stream);
                });
            }
        });
        write_record(
            dir,
            "slow.json",
            &record(
                "slow-host-01",
                &silent,
                "tok",
                i64::from(std::process::id()),
                1,
            ),
        );

        let started = std::time::Instant::now();
        let listed = list_collab_hosts(Some(&list_opts(dir, 300)))
            .await
            .expect("list");
        assert!(listed.is_empty());
        assert!(dir.join("slow.json").exists(), "skip must not prune");
        assert!(
            started.elapsed() >= Duration::from_millis(290),
            "timeout bounded the listing"
        );
        parker.abort();
    }

    #[tokio::test]
    async fn symlinked_registry_dir_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real");
        fs::create_dir(&real).expect("mkdir");
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");

        // Publication refuses a symlinked registry dir (registry.ts:370-375).
        let err = ensure_private_dir(&link).expect_err("symlink refused");
        assert!(err.to_string().contains("symlink"), "{err}");

        // Listing refuses too — the prune-on-list behavior would otherwise
        // follow the link and delete inside the target (registry.ts:366-369).
        let err = list_collab_hosts(Some(&list_opts(&link, 1_000)))
            .await
            .expect_err("list refused");
        assert!(err.to_string().contains("symlink"), "{err}");

        // A plain file where a directory belongs.
        let file = tmp.path().join("file");
        fs::write(&file, "x").expect("write");
        let err = ensure_private_dir(&file).expect_err("file refused");
        assert!(err.to_string().contains("not a directory"), "{err}");
        // The foreign-owner check (registry.ts:379-386) needs a directory
        // owned by a different uid; arranging one requires root, so it is
        // exercised only on root CI rigs and skipped here.
    }

    #[tokio::test]
    async fn ensure_private_dir_tightens_mode() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("existing");
        fs::create_dir(&dir).expect("mkdir");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // An already-present directory keeps its permissions on mkdir
            // (registry.ts:390-395 comment), so ensure tightens explicitly.
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        ensure_private_dir(&dir).expect("ensure");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&dir).expect("stat").permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "tightened to owner-only, got {mode:o}");
        }
    }

    #[tokio::test]
    async fn long_path_socket_relocated_to_fallback() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // Canonical path well past every sun_path limit.
        let deep = tmp.path().join("d".repeat(120));
        let source = Arc::new(
            StubSource::new(CollabHostSnapshot {
                instance_id: "long-path-host-01".into(),
                ..fixture_snapshot()
            })
            .with_link(CollabAccess::Control, "https://relay.test/#ctl"),
        );
        let options = CollabPublishOptions {
            dir: Some(deep.clone()),
            instance_id: Some("long-path-host-01".into()),
            socket_fallback_base: Some(tmp.path().to_path_buf()),
        };
        let publication = publish_collab_host(source, Some(&options)).expect("publish");
        let expected_dir = socket_fallback_dir(&deep, tmp.path());
        assert_eq!(
            Path::new(publication.endpoint()).parent().expect("parent"),
            expected_dir,
            "socket relocated next to the metadata"
        );
        assert!(
            Path::new(publication.endpoint()).as_os_str().len() < SUN_PATH_LIMIT,
            "relocated path fits sun_path"
        );
        let listed = list_collab_hosts(Some(&list_opts(&deep, 1_000)))
            .await
            .expect("list");
        assert_eq!(listed.len(), 1, "discovered via the recorded endpoint");
        assert_eq!(listed[0].instance_id, "long-path-host-01");
        drop(publication);
    }

    #[test]
    fn socket_fallback_dir_hash_shape() {
        // SHA-256 sanity for the hex helper against the canonical "abc"
        // vector, then the streaming Context against a one-shot digest of the
        // exact TS byte sequence `String(uid) + "\0" + dir`
        // (registry.ts:405-411).
        use ring::digest::{digest, SHA256};
        assert!(hex(digest(&SHA256, b"abc").as_ref()).starts_with("ba7816bf"));
        let key = socket_fallback_dir(Path::new("/x"), Path::new("/tmp"));
        let expected = {
            let bytes = format!("{}\0/x", nix::unistd::getuid().as_raw());
            hex(&digest(&SHA256, bytes.as_bytes()).as_ref()[..10])
        };
        assert_eq!(
            key,
            PathBuf::from("/tmp").join(format!("omp-collab-{expected}"))
        );
        let name = key
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        assert_eq!(name.len(), 11 + 20, "{name}");
        assert!(name[11..]
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    }

    #[tokio::test]
    async fn instance_id_pattern_enforced() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let publish_with = |instance_id: String| {
            publish_collab_host(
                Arc::new(StubSource::new(fixture_snapshot())),
                Some(&CollabPublishOptions {
                    dir: Some(dir.to_path_buf()),
                    instance_id: Some(instance_id),
                    ..CollabPublishOptions::default()
                }),
            )
        };
        let err = publish_with("short".into()).expect_err("too short");
        assert!(
            err.to_string()
                .contains("invalid collab registry instance id"),
            "{err}"
        );
        assert!(
            publish_with("ABCDEFGH".into()).is_err(),
            "uppercase rejected"
        );
        assert!(publish_with("abcdefg".into()).is_err(), "7 chars rejected");
        assert!(publish_with("a".repeat(65)).is_err(), "65 chars rejected");
        assert!(
            publish_with("ab-cd-ef".into()).is_ok(),
            "8 chars with dashes ok"
        );
        assert!(publish_with("b".repeat(64)).is_ok(), "64 chars ok");
        // A default (random hex) instance id is always valid.
        assert!(publish_collab_host(
            Arc::new(StubSource::new(fixture_snapshot())),
            Some(&CollabPublishOptions {
                dir: Some(dir.to_path_buf()),
                ..CollabPublishOptions::default()
            })
        )
        .is_ok());
    }

    #[tokio::test]
    async fn resolve_selector_not_found_and_ambiguous() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();

        let err = resolve_collab_host_link(
            "ghost-host-01",
            CollabAccess::View,
            Some(&list_opts(dir, 1_000)),
        )
        .await
        .expect_err("not found");
        let err = err.downcast_ref::<CollabLinkError>().expect("typed");
        assert_eq!(err.code, CollabLinkErrorCode::NotFound);
        assert_eq!(err.message, "no active Collab host matches ghost-host-01");

        // Two publications with the SAME instance id: ambiguous by id.
        let mut twins: Vec<CollabHostPublication> = Vec::new();
        for _ in 0..2 {
            twins.push(
                publish_collab_host(
                    Arc::new(
                        StubSource::new(CollabHostSnapshot {
                            instance_id: "twin-host-01".into(),
                            ..fixture_snapshot()
                        })
                        .with_link(CollabAccess::View, "https://relay.test/#v"),
                    ),
                    Some(&CollabPublishOptions {
                        dir: Some(dir.to_path_buf()),
                        instance_id: Some("twin-host-01".into()),
                        ..CollabPublishOptions::default()
                    }),
                )
                .expect("publish twin"),
            );
        }
        // Hold both publications: a dropped handle withdraws its entry.
        let _twins: Vec<CollabHostPublication> = Vec::new();
        let err = resolve_collab_host_link(
            "twin-host-01",
            CollabAccess::View,
            Some(&list_opts(dir, 1_000)),
        )
        .await
        .expect_err("ambiguous");
        let err = err.downcast_ref::<CollabLinkError>().expect("typed");
        assert_eq!(err.code, CollabLinkErrorCode::Ambiguous);
        assert_eq!(
            err.message,
            "twin-host-01 matches more than one Collab host; use an instance id: twin-host-01, twin-host-01"
        );

        // Two publications under the same real pid but distinct ids:
        // ambiguous by pid.
        fs::remove_dir_all(dir).expect("reset");
        fs::create_dir(dir).expect("mkdir");
        let mut pid_twins: Vec<CollabHostPublication> = Vec::new();
        for instance in ["pidtwin-a1", "pidtwin-b2"] {
            pid_twins.push(
                publish_collab_host(
                    Arc::new(
                        StubSource::new(CollabHostSnapshot {
                            pid: 4243,
                            ..fixture_snapshot()
                        })
                        .with_link(CollabAccess::View, "https://relay.test/#v"),
                    ),
                    Some(&CollabPublishOptions {
                        dir: Some(dir.to_path_buf()),
                        instance_id: Some(instance.into()),
                        ..CollabPublishOptions::default()
                    }),
                )
                .expect("publish"),
            );
        }
        let err =
            resolve_collab_host_link("4243", CollabAccess::View, Some(&list_opts(dir, 1_000)))
                .await
                .expect_err("ambiguous pid");
        assert_eq!(
            err.downcast_ref::<CollabLinkError>().expect("typed").code,
            CollabLinkErrorCode::Ambiguous
        );

        // Leading-zero numerics are not pid selectors (registry.ts:713).
        let err =
            resolve_collab_host_link("04243", CollabAccess::View, Some(&list_opts(dir, 1_000)))
                .await
                .expect_err("leading zero");
        assert_eq!(
            err.downcast_ref::<CollabLinkError>().expect("typed").code,
            CollabLinkErrorCode::NotFound
        );
        drop(pid_twins);
    }

    /// The registry-side rotation lookback: the listed generation dies before
    /// the link request lands, and a higher generation of the same instance
    /// is live → `stale_generation`, not `unreachable` (registry.ts:749-761).
    #[tokio::test]
    async fn resolve_reports_rotation_after_dead_endpoint() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let instance = "rotate-host-01";

        // Generation 1: a hand-rolled endpoint that answers exactly one
        // snapshot, then unbinds so the link request finds ENOENT.
        let gen1_endpoint = dir.join("gen1.sock");
        let gen1_json = serde_json::to_string(&CollabHostSnapshot {
            instance_id: instance.into(),
            generation: 1,
            ..fixture_snapshot()
        })
        .expect("json");
        let gen1_task = tokio::spawn({
            let endpoint = gen1_endpoint.clone();
            let json = gen1_json.clone();
            async move {
                let listener = UnixListener::bind(&endpoint).expect("bind gen1");
                let (mut stream, _) = listener.accept().await.expect("gen1 conn");
                let _ = read_line_capped(&mut stream, MAX_REQUEST_BYTES).await;
                let frame = format!("{{\"ok\":true,\"v\":1,\"snapshot\":{json}}}\n");
                let _ = stream.write_all(frame.as_bytes()).await;
                let _ = stream.shutdown().await;
                // Unbind: the next connect finds the socket file gone.
                drop(listener);
                let _ = fs::remove_file(&endpoint);
            }
        });

        // Generation 2 of the SAME instance: answers snapshots from its
        // second connection on; the first is parked so the initial listing
        // skips it (its metadata survives — skip never prunes).
        let gen2_endpoint = dir.join("gen2.sock");
        let gen2_json = serde_json::to_string(&CollabHostSnapshot {
            instance_id: instance.into(),
            generation: 2,
            ..fixture_snapshot()
        })
        .expect("json");
        let gen2_task = tokio::spawn({
            let endpoint = gen2_endpoint.clone();
            let json = gen2_json.clone();
            async move {
                let listener = UnixListener::bind(&endpoint).expect("bind gen2");
                let mut parked_first = false;
                loop {
                    let (mut stream, _) = listener.accept().await.expect("gen2 conn");
                    if !parked_first {
                        parked_first = true;
                        tokio::spawn(async move {
                            tokio::time::sleep(Duration::from_mins(1)).await;
                            drop(stream);
                        });
                        continue;
                    }
                    let _ = read_line_capped(&mut stream, MAX_REQUEST_BYTES).await;
                    let frame = format!("{{\"ok\":true,\"v\":1,\"snapshot\":{json}}}\n");
                    let _ = stream.write_all(frame.as_bytes()).await;
                    let _ = stream.shutdown().await;
                }
            }
        });

        write_record(
            dir,
            "gen1.json",
            &record(
                instance,
                &gen1_endpoint,
                "tok1",
                i64::from(std::process::id()),
                1,
            ),
        );
        write_record(
            dir,
            "gen2.json",
            &record(
                instance,
                &gen2_endpoint,
                "tok2",
                i64::from(std::process::id()),
                1,
            ),
        );

        let err =
            resolve_collab_host_link(instance, CollabAccess::Control, Some(&list_opts(dir, 400)))
                .await
                .expect_err("stale after rotation");
        let err = err.downcast_ref::<CollabLinkError>().expect("typed");
        assert_eq!(err.code, CollabLinkErrorCode::StaleGeneration, "{err}");
        assert_eq!(
            err.message,
            "host rotate-host-01 started a new room since it was listed; list again and retry"
        );
        gen1_task.abort();
        gen2_task.abort();
    }

    #[tokio::test]
    async fn bound_snapshot_truncates_to_utf16_limit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // 1020 BMP chars + one astral char (2 UTF-16 units) + 10 more chars:
        // the TS UTF-16 bound keeps the elephant, drops the tail.
        let long = format!("{}🐘{}", "a".repeat(1020), "b".repeat(10));
        let mut snapshot = fixture_snapshot();
        snapshot.session_id = long;
        let source = Arc::new(StubSource::new(snapshot));
        let options = CollabPublishOptions {
            dir: Some(tmp.path().to_path_buf()),
            instance_id: Some("bounder-host-01".into()),
            ..CollabPublishOptions::default()
        };
        let publication = publish_collab_host(source, Some(&options)).expect("publish");
        let listed = list_collab_hosts(Some(&list_opts(tmp.path(), 1_000)))
            .await
            .expect("list");
        assert_eq!(listed.len(), 1);
        let session_id = listed[0].session_id.clone();
        // 1020 units of `a`, then the astral pair, then as many `b` units as
        // still fit: exactly 1024 UTF-16 units on the wire, like the TS
        // UTF-16 slice (registry.ts:262-264).
        assert_eq!(session_id.encode_utf16().count(), MAX_SNAPSHOT_FIELD_CHARS);
        assert!(
            session_id.contains('🐘'),
            "boundary never splits a surrogate pair"
        );
        assert!(
            session_id.ends_with("bb"),
            "units after the pair still count"
        );
        assert_eq!(
            session_id.chars().count(),
            1023,
            "1020 + 1 astral char + 2 ascii"
        );
        publication.close().await;
    }

    #[tokio::test]
    async fn source_errors_never_leak() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (source, publication) = publish_fixture(tmp.path());
        let record_token = parse_collab_host_record(
            &fs::read_to_string(
                fs::read_dir(tmp.path())
                    .expect("readdir")
                    .filter_map(std::result::Result::ok)
                    .find(|entry| entry.file_name().to_string_lossy().ends_with(".json"))
                    .expect("meta")
                    .path(),
            )
            .expect("text"),
        )
        .expect("record")
        .token;
        let endpoint = Path::new(publication.endpoint());

        // A failing snapshot source answers the fixed error for BOTH ops —
        // snapshot() is consulted before op dispatch (registry.ts:321-327).
        source.fail_snapshot(true);
        for op in ["snapshot", "link"] {
            let line = if op == "link" {
                format!("{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"link\",\"access\":\"control\",\"generation\":3}}\n")
            } else {
                format!("{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"{op}\"}}\n")
            };
            let response = raw_request(endpoint, &line).await.expect("response");
            assert_eq!(
                response, "{\"ok\":false,\"v\":1,\"error\":\"snapshot_unavailable\"}",
                "op {op}"
            );
        }
        source.fail_snapshot(false);

        // A failing link() collapses to snapshot_unavailable, never
        // access_unavailable (registry.ts:350-356).
        source.fail_link(true);
        let response = raw_request(
            endpoint,
            &format!(
                "{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"link\",\"access\":\"control\",\
                 \"generation\":3}}\n"
            ),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"snapshot_unavailable\"}"
        );
        source.fail_link(false);

        // TS falsy semantics: Ok(None) and Ok(Some("")) are both
        // "not published" (registry.ts:357-359).
        source.unpublish(CollabAccess::Control);
        let response = raw_request(
            endpoint,
            &format!(
                "{{\"v\":1,\"token\":\"{record_token}\",\"op\":\"link\",\"access\":\"control\",\
                 \"generation\":3}}\n"
            ),
        )
        .await
        .expect("response");
        assert_eq!(
            response,
            "{\"ok\":false,\"v\":1,\"error\":\"access_unavailable\"}"
        );
        publication.close().await;
    }

    #[tokio::test]
    async fn close_withdraws_and_drop_withdraws() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let (_source, publication) = publish_fixture(dir);
        let endpoint = publication.endpoint().to_owned();
        publication.close().await;
        assert!(!dir.join("endpoint-check").exists());
        assert!(
            fs::read_dir(dir).expect("readdir").next().is_none(),
            "close removed metadata and socket"
        );
        assert!(
            tokio::net::UnixStream::connect(&endpoint).await.is_err(),
            "socket gone"
        );
        let listed = list_collab_hosts(Some(&list_opts(dir, 1_000)))
            .await
            .expect("list");
        assert!(listed.is_empty());

        // Dropping without close() also withdraws — the Rust stand-in for the
        // TS process-exit hook (registry.ts:513).
        let (_source, publication) = publish_fixture(dir);
        let endpoint = publication.endpoint().to_owned();
        drop(publication);
        assert!(
            fs::read_dir(dir).expect("readdir").next().is_none(),
            "drop removed artifacts"
        );
        assert!(tokio::net::UnixStream::connect(&endpoint).await.is_err());
    }

    #[test]
    fn missing_busy_field_parses_as_unknown() {
        // An omp older than the busy field omits it; the snapshot must parse
        // with busy = None rather than being rejected (registry.ts:255-260).
        let raw: serde_json::Value = serde_json::from_str(FIXTURE_SNAPSHOT_JSON).expect("json");
        let mut without_busy = raw.clone();
        without_busy.as_object_mut().expect("obj").remove("busy");
        let parsed = parse_snapshot(&without_busy).expect("older host accepted");
        assert_eq!(parsed.busy, None);

        // Explicit null reads as unknown too; a wrong type still rejects.
        let mut null_busy = raw.clone();
        null_busy
            .as_object_mut()
            .expect("obj")
            .insert("busy".into(), serde_json::Value::Null);
        assert_eq!(parse_snapshot(&null_busy).expect("null busy").busy, None);
        let mut bad_busy = raw;
        bad_busy
            .as_object_mut()
            .expect("obj")
            .insert("busy".into(), serde_json::json!(1));
        assert!(
            parse_snapshot(&bad_busy).is_none(),
            "non-bool busy rejected"
        );

        // Emission parity: a None busy serializes as an explicit null field,
        // exactly like TS JSON.stringify of the interface-shaped object.
        let mut unknown = fixture_snapshot();
        unknown.busy = None;
        let text = serde_json::to_string(&bound_snapshot(&unknown)).expect("serialize");
        assert!(text.contains("\"busy\":null"), "{text}");
    }

    #[test]
    fn collab_hosts_runtime_dir_honours_env() {
        // Guarded by a process-wide lock: only this test mutates env.
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().expect("env lock");
        let tmp = tempfile::tempdir().expect("tempdir");
        let saved_home = std::env::var_os("HOME");
        let saved_config = std::env::var_os("PI_CONFIG_DIR");
        std::env::set_var("HOME", tmp.path());
        std::env::set_var("PI_CONFIG_DIR", "custom-root");
        assert_eq!(
            collab_hosts_runtime_dir(),
            tmp.path().join("custom-root/run/collab-hosts")
        );
        std::env::remove_var("PI_CONFIG_DIR");
        assert_eq!(
            collab_hosts_runtime_dir(),
            tmp.path().join(".omp/run/collab-hosts")
        );
        // An EMPTY PI_CONFIG_DIR falls back to .omp, like TS `env || ".omp"`
        // (dirs.ts:306-308).
        std::env::set_var("PI_CONFIG_DIR", "");
        assert_eq!(
            collab_hosts_runtime_dir(),
            tmp.path().join(".omp/run/collab-hosts")
        );
        if let Some(home) = saved_home {
            std::env::set_var("HOME", home);
        }
        match saved_config {
            Some(config) => std::env::set_var("PI_CONFIG_DIR", config),
            None => std::env::remove_var("PI_CONFIG_DIR"),
        }
    }

    #[test]
    fn record_and_snapshot_reject_bad_shapes() {
        // Discovery metadata (registry.ts:181-204).
        let good = record("good-host-01", Path::new("/tmp/x.sock"), "tok", 5, 1);
        let good_text = serde_json::to_string(&good).expect("serialize");
        assert!(parse_collab_host_record(&good_text).is_some());
        // Unknown fields are ignored, exactly like the TS field picks.
        let with_extra = format!("{{\"extra\":true,{}", &good_text[1..]);
        assert!(parse_collab_host_record(&with_extra).is_some());
        for broken in [
            "[]",
            "{}",
            "{\"version\":1}",
            &serde_json::to_string(&CollabHostRecord {
                pid: 0,
                ..good.clone()
            })
            .expect("pid 0"),
            &serde_json::to_string(&CollabHostRecord {
                token: String::new(),
                ..good.clone()
            })
            .expect("empty token"),
            &serde_json::to_string(&CollabHostRecord {
                endpoint: String::new(),
                ..good.clone()
            })
            .expect("empty endpoint"),
            &serde_json::to_string(&CollabHostRecord {
                instance_id: "Bad-Host!".into(),
                ..good.clone()
            })
            .expect("bad instance"),
            &serde_json::to_string(&CollabHostRecord { version: 0, ..good })
                .expect("version 0 fine to parse"),
        ] {
            let parsed = parse_collab_host_record(broken);
            if broken.contains("\"version\":0") {
                // version 0 parses; the version GATE happens later
                // (registry.ts:641-648).
                assert!(parsed.is_some());
            } else {
                assert!(parsed.is_none(), "{broken}");
            }
        }

        // Snapshots (registry.ts:218-260).
        let good_snapshot = serde_json::to_value(fixture_snapshot()).expect("snapshot value");
        assert!(parse_snapshot(&good_snapshot).is_some());
        for (key, value) in [
            ("generation", serde_json::json!(0)),
            ("access", serde_json::json!("admin")),
            ("sessionName", serde_json::json!(3)),
            ("relayConnected", serde_json::json!("yes")),
            ("instanceId", serde_json::json!("short")),
        ] {
            let mut broken = good_snapshot.clone();
            broken
                .as_object_mut()
                .expect("obj")
                .insert(key.into(), value);
            assert!(parse_snapshot(&broken).is_none(), "{key}");
        }
        // Non-object snapshots are rejected.
        assert!(parse_snapshot(&serde_json::json!("host")).is_none());
    }

    #[test]
    fn token_matches_unit() {
        assert!(token_matches("abc", Some("abc")));
        assert!(!token_matches("abc", Some("abd")));
        assert!(!token_matches("abc", Some("abcd")));
        assert!(!token_matches("abc", Some("")));
        assert!(!token_matches("", Some("x")));
        assert!(token_matches("", Some("")));
        assert!(!token_matches("abc", None), "absent token never matches");
    }

    #[tokio::test]
    async fn list_sorts_by_started_at_pid_and_id() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path();
        let make = |instance: &str, started_at: i64, pid: i64| {
            let mut snapshot = fixture_snapshot();
            snapshot.instance_id = instance.into();
            snapshot.started_at = started_at;
            snapshot.pid = pid;
            publish_collab_host(
                Arc::new(StubSource::new(snapshot)),
                Some(&CollabPublishOptions {
                    dir: Some(dir.to_path_buf()),
                    instance_id: Some(instance.into()),
                    ..CollabPublishOptions::default()
                }),
            )
            .expect("publish")
        };
        // Published in mixed order; listed sorted by (startedAt, pid, id)
        // (registry.ts:679-684).
        let _c = make("sort-host-c3", 300, 30);
        let _a = make("sort-host-a1", 100, 30);
        let _b2 = make("sort-host-b2", 200, 20);
        let _b1 = make("sort-host-b1", 200, 10);
        let listed = list_collab_hosts(Some(&list_opts(dir, 1_000)))
            .await
            .expect("list");
        let ids: Vec<&str> = listed.iter().map(|s| s.instance_id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "sort-host-a1",
                "sort-host-b1",
                "sort-host-b2",
                "sort-host-c3"
            ]
        );
    }

    #[tokio::test]
    async fn missing_registry_dir_lists_empty() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let listed = list_collab_hosts(Some(&list_opts(&tmp.path().join("absent"), 1_000)))
            .await
            .expect("empty, not error");
        assert!(listed.is_empty());
    }
}
