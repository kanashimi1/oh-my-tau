//! Agent-to-agent bus between a parent session and its children (`task`,
//! eval `agent()`, `workpool` — one channel for every spawner).
//!
//! The parent owns a unix-socket server for as long as at least one
//! [`BusLease`] is alive (a `task` call or an eval subagent holds one). Every
//! child it starts gets [`SOCK_ENV`] (the socket) and [`SELF_ENV`] (its own
//! agent id). A child's `write agent://<id>` that does not name one of its
//! own local handles is sent over the socket and routed by the parent:
//!
//! - `agent://parent` -> the parent's delivery queue ([`take_deliveries`]);
//!   the runtime drains it into a turn at the next turn boundary, so a
//!   message sent while the parent is busy is never lost.
//! - `agent://<sibling-id>` / `agent://all` -> a steer on the run(s)
//!   registered in the shared agent registry, delivered at the sibling's next
//!   turn boundary, prefixed with the sender's id.
//! - an id unknown locally -> forwarded up to this process's own parent bus
//!   ([`Client::from_env`]): the message reaches the sender's grandparent,
//!   which routes it through its own subtree or forwards it further up. The
//!   delivery receipt flows back as the reply.
//! - an id nobody in the tree knows (or a herdr pane id like `w4:pXX`) ->
//!   [`try_route_external`], the seam where the upcoming hub client will take
//!   over; until then it returns today's "not running" error.
//!
//! Ids stay unique within a subtree: every process mints generated ids from
//! its own counter (`agent-N`, `agt-N`, pool worker ids) and the spawn-time
//! registry rejects duplicate names, so an address always resolves to the
//! nearest registry up the tree.
//!
//! Wire format: the SPEC envelope (`runs/a2a-hub-20261026/SPEC.md`), one JSON
//! request line, one JSON reply line `{"ok":true,"text","message_id"}` /
//! `{"ok":false,"error","message_id"}`. Envelopes carry `message_id`
//! (uuid v7, sender-generated) — the server dedups resends by it — and
//! `payload` inline or `payload_ref {path, sha256, bytes}` when the body is
//! larger than [`PAYLOAD_INLINE_MAX`].

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

pub(crate) use omp_task::runner::{A2A_SELF_ENV as SELF_ENV, A2A_SOCK_ENV as SOCK_ENV};

/// Bound for one request/reply exchange (either side).
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Largest accepted request line.
const MAX_LINE: usize = 1 << 20;
/// Bodies larger than this go by `payload_ref`, never inline (SPEC).
pub(crate) const PAYLOAD_INLINE_MAX: usize = 8 * 1024;
/// Largest payload_ref body the server will read back.
const PAYLOAD_REF_MAX: usize = 64 * 1024 * 1024;
/// Recently-seen `message_id`s kept for dedup (SPEC: at-least-once, receivers
/// dedup by message_id); the oldest entries fall off the bound.
const SEEN_CAP: usize = 4096;

/// SPEC envelope kinds the bus speaks today (`agent.question` rides the wire
/// so a hub can flag threads later; delivery is identical to `agent.message`).
pub(crate) mod kind {
    pub(crate) const MESSAGE: &str = "agent.message";
    /// Steer envelopes are accepted on the wire and delivered like messages
    /// (a running recipient injects them at its next turn boundary); the hub
    /// uses the kind for its urgent-priority queue.
    #[allow(dead_code)]
    pub(crate) const STEER: &str = "agent.steer";
    pub(crate) const QUESTION: &str = "agent.question";
}

/// Offloaded body pointer (SPEC `payload_ref`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PayloadRef {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

/// The one envelope type for every channel (SPEC): hub stores it, the
/// in-process bus carries it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct A2aEnvelope {
    /// uuid v7, sender-generated; the dedup key.
    pub message_id: String,
    pub sender: String,
    pub recipient: String,
    /// SPEC `kind` (see [`kind`] for what the bus handles today).
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<PayloadRef>,
}

fn default_kind() -> String {
    kind::MESSAGE.to_string()
}

impl A2aEnvelope {
    /// Mint a fresh envelope, spilling an oversized payload to a ref file.
    pub(crate) fn new(sender: &str, recipient: &str, body: &str) -> std::io::Result<Self> {
        let message_id = crate::persist::uuid_v7();
        let (payload, payload_ref) = if body.len() > PAYLOAD_INLINE_MAX {
            let path = spill_payload(&message_id, body.as_bytes())?;
            (
                None,
                Some(PayloadRef {
                    path: path.to_string_lossy().into_owned(),
                    sha256: sha256_hex(body.as_bytes()),
                    bytes: body.len() as u64,
                }),
            )
        } else {
            (Some(body.to_string()), None)
        };
        Ok(Self {
            message_id,
            sender: sender.to_string(),
            recipient: recipient.to_string(),
            kind: kind::MESSAGE.to_string(),
            payload,
            payload_ref,
        })
    }
}

/// Write an oversized payload body to a private ref file for the server to
/// read back (single delivery: the server removes it after reading).
fn spill_payload(message_id: &str, body: &[u8]) -> std::io::Result<PathBuf> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "omt-a2a-payload-{}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        message_id
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&path)?;
    file.write_all(body)?;
    Ok(path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Read back a spilled body: verify size + hash, remove the file (a ref is
/// single-use; a resend is deduped before this runs, so the file is gone
/// only when the original delivery already happened).
fn read_payload_ref(reference: &PayloadRef) -> Result<String, String> {
    if reference.bytes > PAYLOAD_REF_MAX as u64 {
        return Err(format!(
            "payload_ref too large ({} bytes; max {PAYLOAD_REF_MAX})",
            reference.bytes
        ));
    }
    let bytes = std::fs::read(&reference.path).map_err(|e| {
        format!(
            "payload_ref {} unreadable: {e} (the body was already consumed or never spilled)",
            reference.path
        )
    })?;
    let _ = std::fs::remove_file(&reference.path);
    if bytes.len() as u64 != reference.bytes {
        return Err(format!(
            "payload_ref size mismatch: {} != {} bytes",
            bytes.len(),
            reference.bytes
        ));
    }
    let digest = sha256_hex(&bytes);
    if digest != reference.sha256 {
        return Err("payload_ref sha256 mismatch".to_string());
    }
    String::from_utf8(bytes).map_err(|_| "payload_ref is not UTF-8".to_string())
}

/// Resolve the inline payload (or read back the ref) of a request envelope.
fn envelope_body(envelope: &A2aEnvelope) -> Result<String, String> {
    match (&envelope.payload, &envelope.payload_ref) {
        (Some(payload), _) => Ok(payload.clone()),
        (None, Some(reference)) => read_payload_ref(reference),
        (None, None) => Err("a2a envelope needs a payload or a payload_ref".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Parent-side delivery queue
// ---------------------------------------------------------------------------

/// Identifies the session a delivery/child belongs to: the runtime registers
/// its token at `Runtime::start`; spawns and child messages stamp it, and
/// only that runtime's park/wake/wait legs consume them. `None` = no live
/// runtime (unit tests driving the bus directly).
pub(crate) type Owner = Option<std::sync::Arc<str>>;

static OWNER: LazyLock<Mutex<Owner>> = LazyLock::new(|| Mutex::new(None));

/// Register the live session's owner token (the latest Runtime wins; the
/// previous one is a finished session in tests).
pub(crate) fn set_owner(owner: std::sync::Arc<str>) {
    *OWNER.lock().expect("a2a owner lock") = Some(owner);
}

/// Drop the owner token if it is still `owner` (a newer Runtime::start wins).
pub(crate) fn clear_owner(owner: &std::sync::Arc<str>) {
    let mut slot = OWNER.lock().expect("a2a owner lock");
    if slot.as_ref().is_some_and(|current| current.as_ptr() == owner.as_ptr()) {
        *slot = None;
    }
}

/// The currently registered session owner (stamped on new spawns/deliveries).
#[must_use]
pub(crate) fn current_owner() -> Owner {
    OWNER.lock().expect("a2a owner lock").clone()
}

/// One item for the parent's next turn boundary: a child result
/// ([`crate::tools::task`]) or a child message (`agent://parent`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PendingDelivery {
    /// The session this delivery belongs to ([`Owner`]).
    pub owner: Owner,
    /// Address of the child that produced this (`agent://<id>` without the scheme).
    pub from: String,
    /// SPEC kind of what arrived (`agent.message` / `agent.question`) or
    /// `task.result` for a settled child run.
    pub kind: String,
    /// Model-facing rendered text (result envelope, `<agent-messages>`, ...).
    pub text: String,
}

/// `task.result` pseudo-kind for settled child runs (the envelope kinds cover
/// inter-agent traffic; the runner publishes results on the same queue).
pub(crate) const KIND_TASK_RESULT: &str = "task.result";

static DELIVERIES: LazyLock<Mutex<VecDeque<PendingDelivery>>> =
    LazyLock::new(|| Mutex::new(VecDeque::new()));
static DELIVERIES_NOTIFY: Notify = Notify::const_new();
/// Best-effort wake hook the runtime registers ([`set_delivery_wake`]): lets
/// an idle session start a delivery turn for a message that arrived after the
/// last settle.
static DELIVERY_WAKE: LazyLock<Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>>> =
    LazyLock::new(|| Mutex::new(None));

/// Queue one delivery for the parent's next turn boundary and wake waiters.
pub(crate) fn queue_delivery(delivery: PendingDelivery) {
    let wake = {
        let mut queue = DELIVERIES.lock().expect("a2a deliveries lock");
        let was_empty = queue.is_empty();
        queue.push_back(delivery);
        if was_empty {
            DELIVERY_WAKE.lock().expect("a2a wake lock").clone()
        } else {
            None
        }
    };
    DELIVERIES_NOTIFY.notify_waiters();
    if let Some(wake) = wake {
        wake();
    }
}

/// Drain every pending delivery (tests and unscoped readers).
#[cfg(test)]
pub(crate) fn take_deliveries() -> Vec<PendingDelivery> {
    DELIVERIES.lock().expect("a2a deliveries lock").drain(..).collect()
}

/// Drain the deliveries belonging to one session ([`Owner`]: `None` matches
/// unowned deliveries).
pub(crate) fn take_deliveries_for(owner: &Owner) -> Vec<PendingDelivery> {
    let mut queue = DELIVERIES.lock().expect("a2a deliveries lock");
    let (mine, rest): (Vec<_>, Vec<_>) = queue.drain(..).partition(|d| &d.owner == owner);
    *queue = rest.into();
    mine
}

/// Whether any delivery is waiting (any session).
#[cfg(test)]
#[must_use]
pub(crate) fn has_pending() -> bool {
    !DELIVERIES.lock().expect("a2a deliveries lock").is_empty()
}

/// Whether a delivery is waiting for one session.
#[must_use]
pub(crate) fn has_pending_for(owner: &Owner) -> bool {
    DELIVERIES
        .lock()
        .expect("a2a deliveries lock")
        .iter()
        .any(|d| &d.owner == owner)
}

/// Park until the queue may have changed (single notification, no stored
/// permit — callers re-check the queue and liveness).
pub(crate) async fn wait_delivery_notify() {
    DELIVERIES_NOTIFY.notified().await;
}

/// Register the runtime's idle-wake hook (best effort; replacing any
/// previous one). Called once from `Runtime::start`.
pub(crate) fn set_delivery_wake(wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
    *DELIVERY_WAKE.lock().expect("a2a wake lock") = Some(wake);
}

/// Render drained deliveries as the block a turn starts with.
pub(crate) fn render_deliveries(deliveries: &[PendingDelivery]) -> String {
    let mut messages = String::new();
    let mut results = String::new();
    for delivery in deliveries {
        if delivery.kind == KIND_TASK_RESULT {
            use std::fmt::Write as _;
            let _ = writeln!(results, "{}", delivery.text);
        } else {
            use std::fmt::Write as _;
            let _ = writeln!(
                messages,
                "from agent://{}{}: {}",
                delivery.from,
                question_marker(&delivery.kind),
                delivery.text
            );
        }
    }
    let mut out = String::new();
    if !results.is_empty() {
        out.push_str("<task-results>\n");
        out.push_str(results.trim_end());
        out.push_str("\n</task-results>");
    }
    if !messages.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("<agent-messages>\n");
        out.push_str(messages.trim_end());
        out.push_str("\n</agent-messages>");
    }
    out
}

/// Marker a question delivery carries so the parent can tell a asked
/// question from a plain note (SPEC `agent.question`).
fn question_marker(kind: &str) -> &'static str {
    if kind == kind::QUESTION {
        " (question — answer it in this turn or `write agent://<id>` back)"
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Bus server (parent side)
// ---------------------------------------------------------------------------

/// One session's bus: children of that session connect here. Keyed by the
/// session owner so concurrent sessions (unit tests, later hub-routed panes)
/// never share a socket or a dedup set; production runs one session.
struct BusInstance {
    leases: usize,
    sock: PathBuf,
    dir: PathBuf,
    server: JoinHandle<()>,
    /// Recently accepted `message_id`s (SPEC dedup on resend).
    seen: HashSet<String>,
    seen_order: VecDeque<String>,
}

static BUSES: LazyLock<Mutex<std::collections::HashMap<String, BusInstance>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// Map key for a session owner.
fn owner_key(owner: &Owner) -> String {
    owner.as_ref().map_or_else(String::new, |token| token.to_string())
}

/// Remember a delivered `message_id` on the owner's bus; `false` when it was
/// seen before. No bus instance (a direct `route` call in tests) accepts
/// everything.
fn seen_or_insert(owner: &Owner, message_id: &str) -> bool {
    let mut buses = BUSES.lock().expect("a2a bus lock");
    let Some(bus) = buses.get_mut(&owner_key(owner)) else {
        return true;
    };
    if bus.seen.contains(message_id) {
        return false;
    }
    bus.seen.insert(message_id.to_string());
    bus.seen_order.push_back(message_id.to_string());
    while bus.seen_order.len() > SEEN_CAP {
        if let Some(oldest) = bus.seen_order.pop_front() {
            bus.seen.remove(&oldest);
        }
    }
    true
}

/// Keeps one session's bus server alive; the last drop stops it and removes
/// the socket.
pub(crate) struct BusLease {
    key: String,
    sock: PathBuf,
}

impl BusLease {
    /// Socket path to hand to children as [`SOCK_ENV`].
    pub(crate) fn sock(&self) -> &Path {
        &self.sock
    }

    /// Start (or join) the bus of `owner` (the current session when called
    /// via [`BusLease::acquire`]). Must run inside a tokio runtime.
    pub(crate) fn acquire_for(owner: &Owner) -> std::io::Result<Self> {
        let key = owner_key(owner);
        let mut buses = BUSES.lock().expect("a2a bus lock");
        if !buses.contains_key(&key) {
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "omt-a2a-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            {
                use std::os::unix::fs::DirBuilderExt;
                std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
            }
            let sock = dir.join("bus.sock");
            let listener = match UnixListener::bind(&sock) {
                Ok(listener) => listener,
                Err(e) => {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err(e);
                }
            };
            buses.insert(
                key.clone(),
                BusInstance {
                    leases: 0,
                    sock: sock.clone(),
                    dir,
                    server: tokio::spawn(serve(listener, owner.clone())),
                    seen: HashSet::new(),
                    seen_order: VecDeque::new(),
                },
            );
        }
        let bus = buses.get_mut(&key).expect("bus instance just inserted");
        bus.leases += 1;
        Ok(Self {
            key,
            sock: bus.sock.clone(),
        })
    }

    /// Start (or join) the current session's bus.
    pub(crate) fn acquire() -> std::io::Result<Self> {
        Self::acquire_for(&current_owner())
    }
}

impl Drop for BusLease {
    fn drop(&mut self) {
        let mut buses = BUSES.lock().expect("a2a bus lock");
        let Some(bus) = buses.get_mut(&self.key) else {
            return;
        };
        bus.leases = bus.leases.saturating_sub(1);
        if bus.leases == 0 {
            let bus = buses.remove(&self.key).expect("leased bus exists");
            bus.server.abort();
            let _ = std::fs::remove_dir_all(bus.dir);
        }
    }
}

async fn serve(listener: UnixListener, owner: Owner) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let owner = owner.clone();
        tokio::spawn(async move {
            let _ = tokio::time::timeout(IO_TIMEOUT, handle_conn(stream, owner)).await;
        });
    }
}

async fn handle_conn(stream: UnixStream, owner: Owner) {
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    let mut reader = BufReader::new(read).take(MAX_LINE as u64);
    let reply = match reader.read_line(&mut line).await {
        Ok(n) if n > 0 => match serde_json::from_str::<A2aEnvelope>(&line) {
            Ok(envelope) => route(&envelope, &owner).await,
            Err(e) => Err(format!("bad a2a envelope: {e}")),
        },
        _ => Err("empty a2a request".to_string()),
    };
    let frame = match reply {
        Ok(text) => json!({"ok": true, "text": text}),
        Err(error) => json!({"ok": false, "error": error}),
    };
    let mut out = frame.to_string();
    out.push('\n');
    let _ = write.write_all(out.as_bytes()).await;
    let _ = write.shutdown().await;
}

/// Route one child envelope inside the parent process.
async fn route(envelope: &A2aEnvelope, owner: &Owner) -> Result<String, String> {
    if envelope.sender.is_empty() || envelope.recipient.is_empty() {
        return Err("a2a envelope needs sender and recipient".to_string());
    }
    // SPEC at-least-once: a resent message_id acks without re-delivering.
    if !seen_or_insert(owner, &envelope.message_id) {
        return Ok(format!(
            "Duplicate {}; already delivered.",
            envelope.message_id
        ));
    }
    let body = envelope_body(envelope)?;
    if body.trim().is_empty() {
        return Err("a2a envelope needs a non-empty payload".to_string());
    }
    let to = envelope.recipient.as_str();
    if to == "parent" {
        queue_delivery(PendingDelivery {
            owner: owner.clone(),
            from: envelope.sender.clone(),
            kind: envelope.kind.clone(),
            text: body,
        });
        return Ok("Delivered to parent.".to_string());
    }
    if to == envelope.sender {
        return Err("an agent cannot message itself".to_string());
    }
    let text = format!("[message from agent://{}]\n{body}", envelope.sender);
    if to == "all" {
        // Local subtree first (excluding the sender), then fan out to the rest
        // of the tree above (TS `agent://all` is every running agent).
        let local = super::eval_agents::deliver_from_peer(to, &text, &envelope.sender)
            .await
            .unwrap_or_default();
        return match forward_up(envelope, &body).await {
            Some(Ok(upstream)) => Ok(if local.is_empty() {
                upstream
            } else {
                format!("{local}\n{upstream}")
            }),
            Some(Err(error)) if local.is_empty() => Err(error),
            Some(Err(_)) | None => Ok(if local.is_empty() {
                "No live peers to broadcast to.".to_string()
            } else {
                local
            }),
        };
    }
    let local_error = match super::eval_agents::deliver_from_peer(to, &text, &envelope.sender).await {
        Ok(receipt) => return Ok(receipt),
        Err(error) => error,
    };
    // Unknown locally: forward up to the grandparent bus, which routes the id
    // through its own subtree or further up; the receipt flows back.
    if let Some(reply) = forward_up(envelope, &body).await {
        return reply;
    }
    // Not in this tree: the hub seam takes over here once crates/omp-hub
    // lands (pane ids like `w4:pXX`, then cross-tree addresses). Until then
    // the local error keeps today's text (e.g. "already finished").
    if let Some(reply) = try_route_external(envelope, &body).await {
        return reply;
    }
    Err(local_error)
}

/// Forward an envelope to this process's own parent bus (grandchild routing)
/// verbatim — sender, recipient, kind and `message_id` are preserved, so the
/// grandparent dedups by the original id and receipts name the real sender.
/// `None` when this process is a tree root (no parent socket inherited).
async fn forward_up(envelope: &A2aEnvelope, body: &str) -> Option<Result<String, String>> {
    let up = Client::from_env()?;
    // The recipient may live in the grandparent's subtree even when this
    // level resolved the id locally as unsettled; the body is re-spilled by
    // the client when oversized.
    Some(up.forward(envelope, body).await)
}

/// Hub seam: recipients this tree cannot route (herdr pane ids like `w4:pXX`
/// today, cross-tree addresses via the hub client later). Returns `None`
/// while no hub client exists, so the caller keeps the current
/// "not running" error; the eventual `crates/omp-hub` client plugs in here.
async fn try_route_external(
    envelope: &A2aEnvelope,
    body: &str,
) -> Option<Result<String, String>> {
    // Every recipient reaching this point is external to this tree: either a
    // pane address (`agent://w4:pXX`) or an id no ancestor registered. The hub
    // client will deliver these; until it exists nothing routes them.
    let _pane_address = looks_like_pane_id(&envelope.recipient);
    let _ = body;
    None
}

/// Whether `recipient` is a herdr pane address (`w4:pXX`) — the class of
/// recipients the hub seam will own.
#[must_use]
pub(crate) fn looks_like_pane_id(recipient: &str) -> bool {
    let Some((pane, rest)) = recipient.split_once(':') else {
        return false;
    };
    pane.starts_with('w')
        && pane[1..].chars().all(|c| c.is_ascii_digit())
        && rest.starts_with('p')
        && rest[1..].chars().all(|c| c.is_ascii_alphanumeric())
}

// ---------------------------------------------------------------------------
// Bus client (child side)
// ---------------------------------------------------------------------------

/// Child-side handle on the parent's bus (from the inherited environment).
pub(crate) struct Client {
    sock: PathBuf,
    from: String,
}

impl Client {
    /// `Some` iff this process was started by a parent that runs a bus.
    pub(crate) fn from_env() -> Option<Self> {
        let sock = std::env::var_os(SOCK_ENV).filter(|v| !v.is_empty())?;
        let from = std::env::var(SELF_ENV).ok().filter(|v| !v.is_empty())?;
        Some(Self {
            sock: PathBuf::from(sock),
            from,
        })
    }

    /// A client for an explicit socket/sender (tests drive the bus directly).
    #[cfg(test)]
    pub(crate) fn for_test(sock: &Path, from: &str) -> Self {
        Self {
            sock: sock.to_path_buf(),
            from: from.to_string(),
        }
    }

    /// Send `body` to `to` (`parent`, a sibling id, or `all`) via the parent.
    pub(crate) async fn send(&self, to: &str, body: &str) -> Result<String, String> {
        tokio::time::timeout(IO_TIMEOUT, self.exchange(to, body))
            .await
            .map_err(|_| "parent bus timed out".to_string())?
    }

    async fn exchange(&self, to: &str, body: &str) -> Result<String, String> {
        let envelope = A2aEnvelope::new(&self.from, to, body)
            .map_err(|e| format!("cannot build a2a envelope: {e}"))?;
        self.write_envelope(&envelope).await
    }

    /// Forward an envelope from a descendant (grandchild routing): the
    /// original sender/recipient/kind/message_id are preserved; only the
    /// body is re-spilled when oversized.
    async fn forward(&self, envelope: &A2aEnvelope, body: &str) -> Result<String, String> {
        let forward = A2aEnvelope {
            message_id: envelope.message_id.clone(),
            sender: envelope.sender.clone(),
            recipient: envelope.recipient.clone(),
            kind: envelope.kind.clone(),
            payload: None,
            payload_ref: None,
        };
        let forward = if body.len() > PAYLOAD_INLINE_MAX {
            let path = spill_payload(&forward.message_id, body.as_bytes())
                .map_err(|e| format!("cannot spill forwarded payload: {e}"))?;
            A2aEnvelope {
                payload_ref: Some(PayloadRef {
                    path: path.to_string_lossy().into_owned(),
                    sha256: sha256_hex(body.as_bytes()),
                    bytes: body.len() as u64,
                }),
                ..forward
            }
        } else {
            A2aEnvelope {
                payload: Some(body.to_string()),
                ..forward
            }
        };
        self.write_envelope(&forward).await
    }

    async fn write_envelope(&self, envelope: &A2aEnvelope) -> Result<String, String> {
        let stream = UnixStream::connect(&self.sock)
            .await
            .map_err(|e| format!("parent bus unreachable ({e}); the parent may have finished"))?;
        let (read, mut write) = stream.into_split();
        let mut req = serde_json::to_string(envelope)
            .map_err(|e| format!("cannot encode a2a envelope: {e}"))?;
        req.push('\n');
        write
            .write_all(req.as_bytes())
            .await
            .map_err(|e| format!("parent bus write failed: {e}"))?;
        let mut line = String::new();
        BufReader::new(read)
            .read_line(&mut line)
            .await
            .map_err(|e| format!("parent bus read failed: {e}"))?;
        let reply: Value =
            serde_json::from_str(&line).map_err(|e| format!("bad parent bus reply: {e}"))?;
        if reply.get("ok") == Some(&Value::Bool(true)) {
            Ok(reply
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("Delivered.")
                .to_string())
        } else {
            Err(reply
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("delivery failed")
                .to_string())
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The bus is process-global: serialize the tests that lease it.
    pub(crate) static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Run a block with no inherited parent bus and no registered delivery
    /// owner (both restored after, even on panic): tests driving the bus
    /// directly must not leak traffic up to a real parent, nor let a
    /// concurrent test runtime consume this test's deliveries.
    pub(crate) async fn without_parent_bus<T>(f: impl std::future::Future<Output = T>) -> T {
        struct Guard {
            sock: Option<std::ffi::OsString>,
            self_env: Option<std::ffi::OsString>,
            owner: Owner,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                if let Some(owner) = &self.owner {
                    set_owner(owner.clone());
                }
                match &self.sock {
                    Some(value) => std::env::set_var(SOCK_ENV, value),
                    None => std::env::remove_var(SOCK_ENV),
                }
                match &self.self_env {
                    Some(value) => std::env::set_var(SELF_ENV, value),
                    None => std::env::remove_var(SELF_ENV),
                }
            }
        }
        let guard = Guard {
            sock: std::env::var_os(SOCK_ENV),
            self_env: std::env::var_os(SELF_ENV),
            owner: current_owner(),
        };
        std::env::remove_var(SOCK_ENV);
        std::env::remove_var(SELF_ENV);
        if let Some(owner) = &guard.owner {
            clear_owner(owner);
        }
        f.await
    }

    /// Deliver one envelope straight to the leased bus (raw wire access for
    /// dedup/payload_ref tests).
    pub(crate) async fn raw_send(sock: &Path, envelope: &A2aEnvelope) -> Value {
        let stream = UnixStream::connect(sock).await.unwrap();
        let (read, mut write) = stream.into_split();
        let mut req = serde_json::to_string(envelope).unwrap();
        req.push('\n');
        write.write_all(req.as_bytes()).await.unwrap();
        let mut line = String::new();
        BufReader::new(read).read_line(&mut line).await.unwrap();
        serde_json::from_str(&line).unwrap()
    }

    #[tokio::test]
    async fn parent_messages_queue_and_drain_once() {
            without_parent_bus(async {
        let _serial = LOCK.lock().await;
        // A private owner: deliveries on this bus can never be consumed
        // by a concurrent test runtime's park/wake legs.
        let owner: Owner = Some(crate::persist::uuid_v7().into());
        let lease = BusLease::acquire_for(&owner).unwrap();
        let client = Client {
            sock: lease.sock().to_path_buf(),
            from: "kid-a2a-test".into(),
        };
        let reply = client.send("parent", "need the schema").await.unwrap();
        assert_eq!(reply, "Delivered to parent.");
        let drained = take_deliveries();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].from, "kid-a2a-test");
        assert_eq!(drained[0].kind, kind::MESSAGE);
        assert_eq!(drained[0].text, "need the schema");
        assert!(!has_pending());
        let rendered = render_deliveries(&drained);
        assert!(rendered.contains("from agent://kid-a2a-test: need"));
            })
            .await;
}

    #[tokio::test]
    async fn dedup_acks_resend_without_redelivering() {
            without_parent_bus(async {
        let _serial = LOCK.lock().await;
        // A private owner: deliveries on this bus can never be consumed
        // by a concurrent test runtime's park/wake legs.
        let owner: Owner = Some(crate::persist::uuid_v7().into());
        let lease = BusLease::acquire_for(&owner).unwrap();
        let envelope = A2aEnvelope {
            message_id: "018f-test-dup-00000000000000".to_string(),
            sender: "kid-dup".into(),
            recipient: "parent".into(),
            kind: kind::MESSAGE.to_string(),
            payload: Some("same body".to_string()),
            payload_ref: None,
        };
        let first = raw_send(lease.sock(), &envelope).await;
        assert_eq!(first["ok"], Value::Bool(true));
        let again = raw_send(lease.sock(), &envelope).await;
        assert_eq!(again["ok"], Value::Bool(true));
        assert!(again["text"].as_str().unwrap().contains("Duplicate"), "{again}");
        // Exactly one delivery despite two sends.
        let drained = take_deliveries();
        assert_eq!(drained.len(), 1, "{drained:?}");
        assert_eq!(drained[0].text, "same body");
            })
            .await;
}

    #[tokio::test]
    async fn oversized_payload_travels_by_ref() {
            without_parent_bus(async {
        let _serial = LOCK.lock().await;
        // A private owner: deliveries on this bus can never be consumed
        // by a concurrent test runtime's park/wake legs.
        let owner: Owner = Some(crate::persist::uuid_v7().into());
        let lease = BusLease::acquire_for(&owner).unwrap();
        let client = Client {
            sock: lease.sock().to_path_buf(),
            from: "kid-big".into(),
        };
        let big = format!("{}\nTAIL-MARKER", "x".repeat(PAYLOAD_INLINE_MAX + 5000));
        let reply = client.send("parent", &big).await.unwrap();
        assert_eq!(reply, "Delivered to parent.");
        let drained = take_deliveries();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].text, big);
        assert!(drained[0].text.ends_with("TAIL-MARKER"));
            })
            .await;
}

    #[tokio::test]
    async fn payload_ref_verifies_hash_and_is_single_use() {
            without_parent_bus(async {
        let _serial = LOCK.lock().await;
        // A private owner: deliveries on this bus can never be consumed
        // by a concurrent test runtime's park/wake legs.
        let owner: Owner = Some(crate::persist::uuid_v7().into());
        let lease = BusLease::acquire_for(&owner).unwrap();
        let body = "y".repeat(PAYLOAD_INLINE_MAX + 100);
        let message_id = crate::persist::uuid_v7();
        let path = spill_payload(&message_id, body.as_bytes()).unwrap();
        let envelope = A2aEnvelope {
            message_id,
            sender: "kid-ref".into(),
            recipient: "parent".into(),
            kind: kind::MESSAGE.to_string(),
            payload: None,
            payload_ref: Some(PayloadRef {
                path: path.to_string_lossy().into_owned(),
                sha256: sha256_hex(body.as_bytes()),
                bytes: body.len() as u64,
            }),
        };
        let reply = raw_send(lease.sock(), &envelope).await;
        assert_eq!(reply["ok"], Value::Bool(true), "{reply}");
        assert!(!path.exists(), "server removes the ref file after reading");
        let drained = take_deliveries();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].text, body);
        // A tampered ref body is rejected.
        let body2 = "z".repeat(PAYLOAD_INLINE_MAX + 100);
        let message_id2 = crate::persist::uuid_v7();
        let path2 = spill_payload(&message_id2, body2.as_bytes()).unwrap();
        let bad = A2aEnvelope {
            message_id: message_id2,
            sender: "kid-ref".into(),
            recipient: "parent".into(),
            kind: kind::MESSAGE.to_string(),
            payload: None,
            payload_ref: Some(PayloadRef {
                path: path2.to_string_lossy().into_owned(),
                sha256: sha256_hex(body.as_bytes()), // wrong hash for body2
                bytes: body2.len() as u64,
            }),
        };
        let reply = raw_send(lease.sock(), &bad).await;
        assert_eq!(reply["ok"], Value::Bool(false), "{reply}");
        let _ = std::fs::remove_file(&path2);
            })
            .await;
}

    #[tokio::test]
    async fn unknown_peer_self_and_pane_ids_are_refused() {
            let _serial = LOCK.lock().await;
            without_parent_bus(async {
                // A private owner: deliveries on this bus can never be consumed
                // by a concurrent test runtime's park/wake legs.
                let owner: Owner = Some(crate::persist::uuid_v7().into());
                let lease = BusLease::acquire_for(&owner).unwrap();
                let client = Client {
                    sock: lease.sock().to_path_buf(),
                    from: "kid-x".into(),
                };
                let err = client.send("no-such-peer", "hi").await.unwrap_err();
                assert!(err.contains("no-such-peer"), "{err}");
                let err = client.send("kid-x", "hi").await.unwrap_err();
                assert!(err.contains("itself"), "{err}");
                assert!(looks_like_pane_id("w4:p2P"));
                assert!(looks_like_pane_id("w12:p9Zz"));
                assert!(!looks_like_pane_id("agent-1"));
                assert!(!looks_like_pane_id("w4"));
                let err = client.send("w4:p2P", "hi").await.unwrap_err();
                assert!(err.contains("w4:p2P"), "{err}");
            })
            .await;
    }

    #[tokio::test]
    async fn grandchild_unknown_id_forwards_up_and_receipt_flows_back() {
        let _serial = LOCK.lock().await;
        // The "grandparent": a raw stand-in socket whose reply the forwarding
        // parent must return to the grandchild.
        let dir = std::env::temp_dir().join(format!("omt-a2a-gp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let gp_sock = dir.join("gp.sock");
        let listener = UnixListener::bind(&gp_sock).unwrap();
        // Panic-safe env restore: a leaked OMP_A2A_SOCK would point every
        // later test's `write agent://<unknown>` at this dead socket.
        struct EnvGuard {
            sock: Option<std::ffi::OsString>,
            self_env: Option<std::ffi::OsString>,
        }
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                match &self.sock {
                    Some(value) => std::env::set_var(SOCK_ENV, value),
                    None => std::env::remove_var(SOCK_ENV),
                }
                match &self.self_env {
                    Some(value) => std::env::set_var(SELF_ENV, value),
                    None => std::env::remove_var(SELF_ENV),
                }
            }
        }
        let _guard = EnvGuard {
            sock: std::env::var_os(SOCK_ENV),
            self_env: std::env::var_os(SELF_ENV),
        };
        // SAFETY of the env dance: tests touching OMP_A2A_SOCK are serialized
        // on LOCK, and the guard restores the vars on any exit.
        std::env::set_var(SOCK_ENV, &gp_sock);
        std::env::set_var(SELF_ENV, "middle-parent-1");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut line = String::new();
            BufReader::new(read).read_line(&mut line).await.unwrap();
            let env: A2aEnvelope = serde_json::from_str(&line).unwrap();
            let mut out = json!({"ok": true, "text": format!("routed-by-grandparent:{}", env.recipient)}).to_string();
            out.push('\n');
            write.write_all(out.as_bytes()).await.unwrap();
            env
        });
        let envelope = A2aEnvelope {
            message_id: crate::persist::uuid_v7(),
            sender: "grandchild-1".into(),
            recipient: "uncle-a".into(),
            kind: kind::MESSAGE.to_string(),
            payload: Some("hello up there".to_string()),
            payload_ref: None,
        };
        // route() in the role of the middle parent: id unknown locally and no
        // local registry hit -> forwarded up.
        let reply = route(&envelope, &None).await.unwrap();
        assert_eq!(reply, "routed-by-grandparent:uncle-a");
        let forwarded = server.await.unwrap();
        assert_eq!(forwarded.sender, "grandchild-1");
        assert_eq!(forwarded.kind, kind::MESSAGE);
        assert_eq!(forwarded.payload.as_deref(), Some("hello up there"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn socket_is_removed_after_last_lease() {
        let _serial = LOCK.lock().await;
        // A private owner: other tests may hold leases on the shared buses.
        let owner: Owner = Some(crate::persist::uuid_v7().into());
        let lease = BusLease::acquire_for(&owner).unwrap();
        let second = BusLease::acquire_for(&owner).unwrap();
        let sock = lease.sock().to_path_buf();
        assert!(sock.exists());
        drop(lease);
        assert!(sock.exists(), "still leased");
        drop(second);
        assert!(!sock.exists());
    }
}
