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

use parking_lot::Mutex;
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
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
/// Largest `payload_ref` body the server will read back.
const PAYLOAD_REF_MAX: usize = 64 * 1024 * 1024;
/// Recently-seen `message_id`s kept for dedup (SPEC: at-least-once, receivers
/// dedup by `message_id`); the oldest entries fall off the bound.
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

// invariant: write!/writeln! to a String is infallible.
#[allow(clippy::expect_used)]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    use std::fmt::Write as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    let mut hex = String::new();
    for b in hasher.finalize() {
        write!(hex, "{b:02x}").expect("writing to a String is infallible");
    }
    hex
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

/// Identifies the session a delivery/child belongs to. Two carriers:
/// per-session, the owning runtime stamps its token on the tools' Envs
/// (`crate::tools::Env::a2a_owner`) and `task`/`wait`/`agent()` stamp and
/// consume children/deliveries under it, so concurrent sessions in one
/// process never see each other's traffic; process-global ([`OWNER`], for
/// the Env-less hub endpoint / `BusLease` paths, where the latest Runtime
/// wins — production runs one Runtime per process). `None` = no live
/// runtime (unit tests driving the bus directly).
pub(crate) type Owner = Option<std::sync::Arc<str>>;

static OWNER: LazyLock<Mutex<Owner>> = LazyLock::new(|| Mutex::new(None));

/// Register the live session's owner token for the Env-less paths (hub
/// endpoint lookup, `BusLease::acquire`); per-session traffic goes through
/// the tools' `crate::tools::Env::a2a_owner` instead. The latest Runtime
/// wins; the previous one is a finished session in tests. Production runs
/// one runtime per process.
pub(crate) fn set_owner(owner: std::sync::Arc<str>) {
    *OWNER.lock() = Some(owner);
}

/// Drop the owner token if it is still `owner` (a newer `Runtime::start` wins).
pub(crate) fn clear_owner(owner: &std::sync::Arc<str>) {
    let mut slot = OWNER.lock();
    if slot
        .as_ref()
        .is_some_and(|current| current.as_ptr() == owner.as_ptr())
    {
        *slot = None;
    }
}

/// The process-global session owner (see [`set_owner`]); only for paths with
/// no tool `Env` at hand. Tools must read `crate::tools::Env::a2a_owner`
/// instead.
#[must_use]
pub(crate) fn current_owner() -> Owner {
    OWNER.lock().clone()
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
/// Shareable form of the delivery wake hook.
type WakeHook = std::sync::Arc<dyn Fn() + Send + Sync>;

/// Best-effort wake hook the runtime registers ([`set_delivery_wake`]): lets
/// an idle session start a delivery turn for a message that arrived after the
/// last settle.
static DELIVERY_WAKE: LazyLock<Mutex<Option<WakeHook>>> = LazyLock::new(|| Mutex::new(None));

/// Queue one delivery for the parent's next turn boundary and wake waiters.
pub(crate) fn queue_delivery(delivery: PendingDelivery) {
    let wake = {
        let mut queue = DELIVERIES.lock();
        let was_empty = queue.is_empty();
        queue.push_back(delivery);
        if was_empty {
            DELIVERY_WAKE.lock().clone()
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
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
pub(crate) fn take_deliveries() -> Vec<PendingDelivery> {
    DELIVERIES.lock().drain(..).collect()
}

/// Drain the deliveries belonging to one session ([`Owner`]: `None` matches
/// unowned deliveries). Hub deliveries drained here start their turn (or a
/// `wait` result): ack `read` and advance the endpoint's last acked seq.
pub(crate) fn take_deliveries_for(owner: &Owner) -> Vec<PendingDelivery> {
    let mut queue = DELIVERIES.lock();
    let (mine, rest): (Vec<_>, Vec<_>) = queue.drain(..).partition(|d| &d.owner == owner);
    *queue = rest.into();
    drop(queue);
    let read_acks: Vec<(String, u64)> = {
        let mut inflight = HUB_INFLIGHT.lock();
        let (mine, rest): (Vec<_>, Vec<_>) = inflight.drain(..).partition(|(o, _, _)| o == owner);
        *inflight = rest;
        mine.into_iter()
            .map(|(_, message_id, seq)| (message_id, seq))
            .collect()
    };
    if !read_acks.is_empty() {
        if let Some(ep) = hub_endpoint_for_owner(owner) {
            for (message_id, seq) in read_acks {
                let _ = ep.cmd_tx.send(HubCmd::Ack {
                    message_id,
                    seq,
                    state: omp_hub::State::READ,
                });
            }
        }
    }
    mine
}

/// Whether any delivery is waiting (any session).
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
#[must_use]
pub(crate) fn has_pending() -> bool {
    !DELIVERIES.lock().is_empty()
}

/// Whether a delivery is waiting for one session.
#[must_use]
pub(crate) fn has_pending_for(owner: &Owner) -> bool {
    DELIVERIES.lock().iter().any(|d| &d.owner == owner)
}

/// Park until the queue may have changed (single notification, no stored
/// permit — callers re-check the queue and liveness).
pub(crate) async fn wait_delivery_notify() {
    DELIVERIES_NOTIFY.notified().await;
}

/// Register the runtime's idle-wake hook (best effort; replacing any
/// previous one). Called once from `Runtime::start`.
pub(crate) fn set_delivery_wake(wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
    *DELIVERY_WAKE.lock() = Some(wake);
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
    owner.as_ref().map_or_else(String::new, ToString::to_string)
}

/// Remember a delivered `message_id` on the owner's bus; `false` when it was
/// seen before. No bus instance (a direct `route` call in tests) accepts
/// everything.
fn seen_or_insert(owner: &Owner, message_id: &str) -> bool {
    let mut buses = BUSES.lock();
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
    // invariant: the bus entry for key was inserted in the match arm above.
    #[allow(clippy::expect_used)]
    pub(crate) fn acquire_for(owner: &Owner) -> std::io::Result<Self> {
        let key = owner_key(owner);
        let mut buses = BUSES.lock();
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
                    sock,
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
    // invariant: a live BusLease always has its entry in the buses map.
    #[allow(clippy::expect_used)]
    fn drop(&mut self) {
        let mut buses = BUSES.lock();
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
    let local_error = match super::eval_agents::deliver_from_peer(to, &text, &envelope.sender).await
    {
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

/// Hub seam: recipients this tree cannot route (herdr pane ids like `w4:pXX`,
/// ids unknown to every ancestor). They become SPEC envelopes through the
/// session's live hub endpoint: a pending question from that sender is
/// answered (`answer{}` closes the thread, validated against the original's
/// `reply_schema`); anything else is a `send{}` whose receipt reports the hub
/// state (`delivered`/`queued`) and the `message_id`. `None` while opted out
/// (`OMP_HUB=0`) or when no endpoint is registered, so the caller keeps its
/// local "not running" error; a registered endpoint whose hub is unreachable
/// is a clear error, never silent loss.
async fn try_route_external(envelope: &A2aEnvelope, body: &str) -> Option<Result<String, String>> {
    if hub_opted_out() {
        return None;
    }
    let ep = hub_endpoint_for_owner(&current_owner())?;
    if !ep.connected.load(AtomicOrdering::SeqCst) {
        return Some(Err(format!(
            "hub unreachable for {}: endpoint {} is not connected (the hub link retries in the background; OMP_HUB=0 opts out)",
            envelope.recipient,
            ep.name
        )));
    }
    // Reply to a pending hub question: `write agent://<sender>` after a
    // reply-expected envelope landed here is an `answer{}` with
    // `causation_id` = the original message, so the hub can close the
    // thread / validate the reply schema.
    let answer_payload = {
        if let Some(q) = ep.question_from(&envelope.recipient) {
            let from = ep.name.clone();
            let (reply, rx) = oneshot::channel();
            let _ = ep.cmd_tx.send(HubCmd::Answer {
                causation_id: q.message_id.clone(),
                sender: from,
                payload: body.to_string(),
                reply,
            });
            Some(rx.await)
        } else {
            None
        }
    };
    if let Some(outcome) = answer_payload {
        return Some(match outcome {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("hub link closed before the answer was sent".to_string()),
        });
    }
    // Fresh envelope: sender is this session's endpoint so replies resolve
    // here; a child-originated write keeps its attribution in the body.
    let sender = hub_sender_name(envelope);
    let payload = if sender.1 {
        format!("[from agent://{}]\n{body}", envelope.sender)
    } else {
        body.to_string()
    };
    let mut hub_env = omp_hub::Envelope {
        message_id: envelope.message_id.clone(),
        sender: sender.0,
        recipient: envelope.recipient.clone(),
        kind: if envelope.kind.is_empty() {
            kind::MESSAGE.to_string()
        } else {
            envelope.kind.clone()
        },
        payload: None,
        ..omp_hub::Envelope::default()
    };
    if payload.len() > PAYLOAD_INLINE_MAX {
        match spill_payload(&hub_env.message_id, payload.as_bytes()) {
            Ok(path) => {
                hub_env.payload_ref = Some(omp_hub::PayloadRef {
                    path: path.to_string_lossy().into_owned(),
                    sha256: sha256_hex(payload.as_bytes()),
                    bytes: payload.len() as u64,
                });
            }
            Err(e) => return Some(Err(format!("cannot spill hub payload: {e}"))),
        }
    } else {
        hub_env.payload = Some(payload);
    }
    Some(hub_send_envelope(&ep, hub_env).await)
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
// Hub link (SPEC batch 2): endpoint registration, external routing, claims
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Env opt-outs / modes.
const HUB_OPT_OUT_ENV: &str = "OMP_HUB";
const CLAIMS_MODE_ENV: &str = "OMP_CLAIMS";
/// Auto-claim ttl after an agent writes a repo file (SPEC: 15 min).
const CLAIM_TTL_S: u64 = 15 * 60;
/// Renew an existing auto-claim at most this often (writes are frequent).
const CLAIM_RENEW_EVERY: Duration = Duration::from_mins(1);
/// Negative claim checks are trusted this long (SPEC: cache briefly).
const CLAIM_NEG_CACHE: Duration = Duration::from_secs(2);
/// Bound for one hub RPC issued through the endpoint command channel.
const HUB_CMD_TIMEOUT: Duration = Duration::from_secs(10);
/// Reconnect cadence after the hub dropped or refused a connection.
const HUB_RECONNECT_EVERY: Duration = Duration::from_secs(5);
/// Recently-injected hub `message_id`s (at-least-once replay dedup).
const HUB_SEEN_CAP: usize = 4096;

fn hub_opted_out() -> bool {
    std::env::var(HUB_OPT_OUT_ENV).ok().as_deref() == Some("0")
}

fn claims_block_mode() -> bool {
    std::env::var(CLAIMS_MODE_ENV).ok().as_deref() == Some("block")
}

/// This process's hub address: `$HERDR_PANE_ID` when running on a herdr
/// pane, else a per-process `session:<uuid>` (SPEC item 2).
fn hub_process_name() -> String {
    static NAME: LazyLock<String> = LazyLock::new(|| {
        std::env::var("HERDR_PANE_ID")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| format!("session:{}", crate::persist::uuid_v7()))
    });
    NAME.clone()
}

/// Wire sender for externally routed envelopes: `(address, attributed)` —
/// `attributed` when the original sender differed (a child id) and the body
/// carries a `[from agent://…]` attribution line.
fn hub_sender_name(envelope: &A2aEnvelope) -> (String, bool) {
    let process = hub_process_name();
    if envelope.sender.is_empty() || envelope.sender == process {
        (process, false)
    } else {
        (process, true)
    }
}

/// One command for the endpoint's hub task.
#[allow(clippy::large_enum_variant)] // commands are sent one at a time; boxing each payload adds noise for no gain
enum HubCmd {
    /// `delivered` on injection; `read` when the turn containing the message
    /// starts (also advances the persisted `last_seq`).
    Ack {
        message_id: String,
        seq: u64,
        state: &'static str,
    },
    /// Route one envelope through `send{}`.
    Send {
        envelope: omp_hub::Envelope,
        reply: oneshot::Sender<Result<String, String>>,
    },
    /// Answer a pending question (`answer{}`, schema-validated).
    Answer {
        causation_id: String,
        sender: String,
        payload: String,
        reply: oneshot::Sender<Result<String, String>>,
    },
    /// Who holds a resource (claims guard).
    Check {
        resource: String,
        reply: oneshot::Sender<Option<omp_hub::ClaimRow>>,
    },
    /// Auto-claim a resource after a successful write.
    Claim {
        resource: String,
        reply: oneshot::Sender<bool>,
    },
    /// Session end: release every auto-claim, then signal completion.
    ReleaseAll { done: ReleaseDone },
}

/// How the `ReleaseAll` handler signals completion back to the shutdown caller.
enum ReleaseDone {
    /// Sync path (`Runtime::drop` / `exit()`): a parking std thread.
    Std(std::sync::mpsc::Sender<()>),
    /// Async path (main's final `block_on`): a tokio oneshot, awaited inside
    /// the runtime so a current-thread runtime keeps driving the endpoint.
    Tokio(tokio::sync::oneshot::Sender<()>),
}

impl ReleaseDone {
    fn fire(self) {
        match self {
            Self::Std(sender) => {
                let _ = sender.send(());
            }
            Self::Tokio(sender) => {
                let _ = sender.send(());
            }
        }
    }
}

/// A reply-expected envelope this endpoint owes an answer to.
#[derive(Debug, Clone)]
struct PendingQuestion {
    message_id: String,
    sender: String,
}

/// One auto-claim held by this endpoint.
#[derive(Debug, Clone)]
struct HeldClaim {
    fencing_token: u64,
    acquired: std::time::Instant,
}

/// Live registrations keyed by the session owner they deliver into.
static HUB_ENDPOINTS: LazyLock<Mutex<HashMap<String, Arc<HubEndpoint>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Hub deliveries waiting for the turn that contains them to start: the
/// endpoint acks `read` (and advances `last_seq`) when
/// [`take_deliveries_for`] drains them into a turn or a `wait` result.
static HUB_INFLIGHT: LazyLock<Mutex<Vec<(Owner, String, u64)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// This session's hub endpoint (the latest registered wins, mirroring
/// [`OWNER`]; production runs one runtime per process).
fn hub_endpoint_for_owner(owner: &Owner) -> Option<Arc<HubEndpoint>> {
    let endpoints = HUB_ENDPOINTS.lock();
    if let Some(owner) = owner {
        if let Some(ep) = endpoints.get(owner.as_ref()) {
            return Some(ep.clone());
        }
    }
    None
}

/// Endpoint metadata + command surface shared with the tools.
pub(crate) struct HubEndpoint {
    name: String,
    owner: Owner,
    hub: omp_hub::HubDir,
    cmd_tx: mpsc::UnboundedSender<HubCmd>,
    /// Whether the background task currently holds a hub connection.
    connected: AtomicBool,
    stop: AtomicBool,
    /// Reply-expected envelopes awaiting an answer (any recipient we owe).
    questions: Mutex<Vec<PendingQuestion>>,
    /// Resource -> held claim (fencing token), for renewal and release.
    held: Mutex<HashMap<String, HeldClaim>>,
    /// Negative claim checks, trusted for [`CLAIM_NEG_CACHE`].
    claim_neg: Mutex<HashMap<String, std::time::Instant>>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl HubEndpoint {
    /// The oldest unanswered question owed to `sender`, if any.
    fn question_from(&self, sender: &str) -> Option<PendingQuestion> {
        self.questions
            .lock()
            .iter()
            .find(|q| q.sender == sender)
            .cloned()
    }
}

/// Register this session with the hub as a live endpoint (`hello` with its
/// last acked seq on a background task). `OMP_HUB=0` opts out; a hub that
/// cannot start or be reached logs once and the session continues without
/// it. Returns the endpoint (also stored for [`hub_endpoint_for_owner`]).
pub(crate) fn start_hub_endpoint(owner: &std::sync::Arc<str>) -> Option<Arc<HubEndpoint>> {
    if hub_opted_out() {
        return None;
    }
    Some(start_hub_endpoint_with(
        owner,
        &hub_process_name(),
        &omp_hub::HubDir::resolve(),
    ))
}

fn start_hub_endpoint_with(
    owner: &std::sync::Arc<str>,
    name: &str,
    hub: &omp_hub::HubDir,
) -> Arc<HubEndpoint> {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<HubCmd>();
    let ep = Arc::new(HubEndpoint {
        name: name.to_string(),
        owner: Some(owner.clone()),
        hub: hub.clone(),
        cmd_tx,
        connected: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        questions: Mutex::new(Vec::new()),
        held: Mutex::new(HashMap::new()),
        claim_neg: Mutex::new(HashMap::new()),
        task: Mutex::new(None),
    });
    let task = tokio::spawn(hub_endpoint_task(ep.clone(), cmd_rx));
    *ep.task.lock() = Some(task);
    HUB_ENDPOINTS.lock().insert(owner.to_string(), ep.clone());
    tracing::info!(endpoint = %name, dir = %hub.dir.display(), "hub endpoint registered");
    ep
}

/// Session end for every live hub endpoint (process exit path: `omt`'s
/// `exit()` bypasses `Runtime::drop`, so claims must be released here —
/// same pattern as the herdr pane release). Best-effort and bounded.
pub(crate) fn shutdown_all_hub_endpoints() {
    let keys: Vec<String> = HUB_ENDPOINTS.lock().keys().cloned().collect();
    for key in keys {
        shutdown_hub_endpoint_key(&key);
    }
}

/// Async variant of [`shutdown_all_hub_endpoints`] for the process
/// termination hook: awaited inside the runtime (a final `block_on`), so a
/// current-thread runtime keeps driving the endpoint task and the release
/// round-trip actually completes before the process exits.
pub(crate) async fn shutdown_all_hub_endpoints_async() {
    let keys: Vec<String> = HUB_ENDPOINTS.lock().keys().cloned().collect();
    for key in keys {
        let Some(ep) = HUB_ENDPOINTS.lock().remove(&key) else {
            continue;
        };
        ep.stop.store(true, AtomicOrdering::SeqCst);
        let task = ep.task.lock().take();
        if task.is_some() && ep.connected.load(AtomicOrdering::SeqCst) {
            let (done_tx, done_rx) = oneshot::channel::<()>();
            if ep
                .cmd_tx
                .send(HubCmd::ReleaseAll {
                    done: ReleaseDone::Tokio(done_tx),
                })
                .is_ok()
            {
                let _ = tokio::time::timeout(Duration::from_secs(2), done_rx).await;
            }
        }
        if let Some(task) = task {
            task.abort();
        }
    }
}

/// Session end: stop the background task, releasing every auto-claim first.
/// Best-effort: bounded so a dead hub never stalls teardown.
pub(crate) fn shutdown_hub_endpoint(owner: &std::sync::Arc<str>) {
    shutdown_hub_endpoint_key(owner.as_ref());
}

fn shutdown_hub_endpoint_key(owner_key: &str) {
    let Some(ep) = HUB_ENDPOINTS.lock().remove(owner_key) else {
        return;
    };
    ep.stop.store(true, AtomicOrdering::SeqCst);
    let task = ep.task.lock().take();
    // Release whatever this endpoint holds: a claim granted microseconds
    // before shutdown (the pre-check below raced the grant) must not leak —
    // the command queue is FIFO, so ReleaseAll drains every Claim that
    // preceded it. A connected endpoint always gets the command; the wait
    // runs on a dedicated std thread because `Drop` may execute on a
    // single-threaded runtime (parking the caller would starve the endpoint
    // task). The dropping thread still parks briefly so a clean process exit
    // gives the release a chance to land.
    let has_task = task.is_some();
    if has_task && ep.connected.load(AtomicOrdering::SeqCst) {
        let cmd_tx = ep.cmd_tx.clone();
        let released = Arc::new(AtomicBool::new(false));
        let released_thread = released.clone();
        std::thread::spawn(move || {
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            if cmd_tx
                .send(HubCmd::ReleaseAll {
                    done: ReleaseDone::Std(done_tx),
                })
                .is_ok()
            {
                let _ = done_rx.recv_timeout(Duration::from_secs(2));
            }
            released_thread.store(true, AtomicOrdering::SeqCst);
            if let Some(task) = task {
                task.abort();
            }
        });
        for _ in 0..60 {
            if released.load(AtomicOrdering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    } else if let Some(task) = task {
        task.abort();
    }
}

/// The background task: connect (auto-start), `hello` with the last acked
/// seq, inject `deliver` pushes at the session's next turn boundary, ack
/// `delivered` on injection / `read` on drain, answer and claim commands;
/// reconnects from the last acked seq after a drop.
#[allow(clippy::too_many_lines)] // one hub loop: commands, reconnect with seq resume, delivery acks
async fn hub_endpoint_task(ep: Arc<HubEndpoint>, mut cmd_rx: mpsc::UnboundedReceiver<HubCmd>) {
    let mut last_seq = read_last_seq(&ep.hub.dir, &ep.name);
    let mut seen: HashSet<String> = HashSet::new();
    let mut seen_order: VecDeque<String> = VecDeque::new();
    let mut logged_unreachable = false;
    // The first read-ack persists immediately (no 1 s debounce on a fresh
    // session: crash-recovery replay must start from it right away).
    let mut inflight_persist = std::time::Instant::now()
        .checked_sub(Duration::from_secs(2))
        .unwrap_or_else(std::time::Instant::now);
    while !ep.stop.load(AtomicOrdering::SeqCst) {
        if hub_opted_out() {
            return;
        }
        let mut client = match omp_hub::HubClient::connect_with(ep.hub.clone(), true).await {
            Ok(client) => {
                if !ep.connected.swap(true, AtomicOrdering::SeqCst) {
                    tracing::info!(endpoint = %ep.name, "hub endpoint connected");
                }
                logged_unreachable = false;
                client
            }
            Err(e) => {
                ep.connected.store(false, AtomicOrdering::SeqCst);
                if !logged_unreachable {
                    // Once per outage, not per retry (SPEC: log once).
                    tracing::warn!(
                        endpoint = %ep.name,
                        error = %format!("{e:#}"),
                        "hub unreachable; continuing without it (OMP_HUB=0 opts out)"
                    );
                    logged_unreachable = true;
                }
                tokio::time::sleep(HUB_RECONNECT_EVERY).await;
                continue;
            }
        };
        let mut stream = client.take_deliver_stream();
        let hello = client.hello(&ep.name, last_seq).await;
        match hello {
            Ok(h) => {
                let replayed = h.get("replayed").and_then(Value::as_u64).unwrap_or(0);
                tracing::info!(endpoint = %ep.name, replayed, "hub endpoint hello");
            }
            Err(e) => {
                tracing::warn!(endpoint = %ep.name, error = %format!("{e:#}"), "hub hello failed");
                ep.connected.store(false, AtomicOrdering::SeqCst);
                tokio::time::sleep(HUB_RECONNECT_EVERY).await;
                continue;
            }
        }
        // Connection phase: inject pushes and serve commands until the
        // connection drops (then reconnect from the last acked seq).
        let connection_dead = loop {
            if ep.stop.load(AtomicOrdering::SeqCst) {
                return;
            }
            tokio::select! {
                envelope = async {
                    match stream.as_mut() {
                        Some(s) => s.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    let Some(envelope) = envelope else {
                        break true;
                    };
                    inject_hub_envelope(&ep, &envelope, &mut seen, &mut seen_order);
                    // Ack `delivered` on injection; `read` follows when the
                    // turn that contains the message starts.
                    if let Err(e) = client.ack(&envelope.message_id, omp_hub::State::DELIVERED).await {
                        tracing::debug!(error = %format!("{e:#}"), "hub delivered-ack failed");
                        break true;
                    }
                }
                cmd = cmd_rx.recv() => {
                    let Some(cmd) = cmd else { return; };
                    match cmd {
                        HubCmd::Ack { message_id, seq, state } => {
                            match client.ack(&message_id, state).await {
                                Ok(()) => {
                                    if state == omp_hub::State::READ {
                                        last_seq = last_seq.max(seq);
                                        if inflight_persist.elapsed() > Duration::from_secs(1) {
                                            write_last_seq(&ep.hub.dir, &ep.name, last_seq);
                                            inflight_persist = std::time::Instant::now();
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::debug!(error = %format!("{e:#}"), "hub ack failed");
                                    break true;
                                }
                            }
                        }
                        HubCmd::Send { envelope, reply } => {
                            let out = client.send(envelope).await.map(receipt_text)
                                .map_err(|e| format!("{e:#}"));
                            let _ = reply.send(out);
                        }
                        HubCmd::Answer { causation_id, sender, payload, reply } => {
                            let out = client.answer_text(&causation_id, &sender, &payload).await
                                .map(|_| format!("Answered (question {causation_id} is now answered)."))
                                .map_err(|e| format!("answer failed: {e:#}"));
                            if out.is_ok() {
                                ep.questions.lock()
                                    .retain(|q| q.message_id != causation_id);
                            }
                            let _ = reply.send(out);
                        }
                        HubCmd::Check { resource, reply } => {
                            let row = client.check_claim(&resource).await
                                .unwrap_or(None)
                                .filter(|row| row.expires_ms > omp_hub::now_ms());
                            let _ = reply.send(row);
                        }
                        HubCmd::Claim { resource, reply } => {
                            let granted = match client.claim(&resource, &ep.name, CLAIM_TTL_S, Some("editing")).await {
                                Ok(outcome) if outcome.granted => {
                                    ep.held.lock().insert(resource.clone(), HeldClaim {
                                        fencing_token: outcome.fencing_token.unwrap_or(0),
                                        acquired: std::time::Instant::now(),
                                    });
                                    true
                                }
                                Ok(_) => false,
                                Err(e) => {
                                    tracing::debug!(error = %format!("{e:#}"), "hub claim failed");
                                    break true;
                                }
                            };
                            let _ = reply.send(granted);
                        }
                        HubCmd::ReleaseAll { done } => {
                            let claims: Vec<(String, u64)> = {
                                let mut held = ep.held.lock();
                                let drained: Vec<_> = held.drain().collect();
                                drained.into_iter()
                                    .map(|(r, c)| (r, c.fencing_token))
                                    .collect()
                            };
                            for (resource, token) in claims {
                                if let Err(e) = client.release(&resource, &ep.name, token).await {
                                    tracing::debug!(resource = %resource, error = %format!("{e:#}"), "hub claim release failed");
                                }
                            }
                            done.fire();
                        }
                    }
                }
            }
        };
        if connection_dead {
            ep.connected.store(false, AtomicOrdering::SeqCst);
            write_last_seq(&ep.hub.dir, &ep.name, last_seq);
            tokio::time::sleep(HUB_RECONNECT_EVERY).await;
        }
    }
    write_last_seq(&ep.hub.dir, &ep.name, last_seq);
}

/// Inject one hub `deliver` push into the session's next turn boundary
/// through the same delivery mechanism task results use; a reply-expected
/// envelope is registered as a pending question and its text tells the model
/// to answer with `write agent://<sender>`.
fn inject_hub_envelope(
    ep: &Arc<HubEndpoint>,
    envelope: &omp_hub::Envelope,
    seen: &mut HashSet<String>,
    seen_order: &mut VecDeque<String>,
) {
    // At-least-once: replayed pushes dedup on message_id.
    if seen.contains(&envelope.message_id) {
        return;
    }
    seen.insert(envelope.message_id.clone());
    seen_order.push_back(envelope.message_id.clone());
    while seen_order.len() > HUB_SEEN_CAP {
        if let Some(oldest) = seen_order.pop_front() {
            seen.remove(&oldest);
        }
    }
    let body = match (&envelope.payload, &envelope.payload_ref) {
        (Some(payload), _) => payload.clone(),
        (None, Some(reference)) => read_payload_ref(&PayloadRef {
            path: reference.path.clone(),
            sha256: reference.sha256.clone(),
            bytes: reference.bytes,
        })
        .unwrap_or_else(|e| format!("(unreadable payload_ref: {e})")),
        (None, None) => String::new(),
    };
    let mut text = body;
    let reply_expected = envelope.is_reply_expected();
    if reply_expected {
        ep.questions.lock().push(PendingQuestion {
            message_id: envelope.message_id.clone(),
            sender: envelope.sender.clone(),
        });
        let schema_hint = envelope
            .reply_schema
            .as_ref()
            .map_or(String::new(), |s| format!("\nThe reply is validated against this JSON Schema — a violating reply comes back as the write error:\n{}", serde_json::to_string(s).unwrap_or_default()));
        let expected_hint = envelope
            .expected
            .as_deref()
            .filter(|e| !e.is_empty() && *e != "none")
            .map_or(String::new(), |e| format!("\nExpected: {e}"));
        text = format!(
            "{text}\n\n---\nThis message expects a reply. Answer it in this turn with the `write` tool: write agent://{}{}{}",
            envelope.sender, expected_hint, schema_hint
        );
    }
    queue_delivery(PendingDelivery {
        owner: ep.owner.clone(),
        from: envelope.sender.clone(),
        kind: if envelope.kind == "agent.question" {
            kind::QUESTION.to_string()
        } else {
            kind::MESSAGE.to_string()
        },
        text,
    });
    HUB_INFLIGHT.lock().push((
        ep.owner.clone(),
        envelope.message_id.clone(),
        envelope.seq.unwrap_or(0),
    ));
}

/// Render one `send{}` receipt for the model-facing write result: the hub
/// state (`delivered`/`queued`/`failed`) and the `message_id`, so a queued
/// envelope is never mistaken for a lost one.
// invariant: write!/writeln! to a String is infallible.
#[allow(clippy::expect_used)]
fn receipt_text(receipt: omp_hub::SendReceipt) -> String {
    use std::fmt::Write as _;
    let state = match receipt.state.as_str() {
        s if s == omp_hub::State::SENT => "queued",
        s if s == omp_hub::State::DELIVERED => "delivered",
        s if s == omp_hub::State::FAILED => "failed",
        other => other,
    };
    let mut text = format!(
        "Sent via hub: state={state} (message_id {}, seq {})",
        receipt.message_id, receipt.seq
    );
    if let Some(error) = receipt.error {
        write!(text, " — {error}").expect("writing to a String is infallible");
    }
    if receipt.dedup {
        text.push_str(" (dedup)");
    }
    text
}

/// Send one envelope through a live endpoint's hub client. The hub itself is
/// auto-started (and reconnected) by the endpoint task, never here: a write
/// must not block on a spawn.
async fn hub_send_envelope(
    ep: &Arc<HubEndpoint>,
    envelope: omp_hub::Envelope,
) -> Result<String, String> {
    let recipient = envelope.recipient.clone();
    if !ep.connected.load(AtomicOrdering::SeqCst) {
        return Err(format!(
            "hub unreachable for {recipient}: endpoint {} is not connected (the hub link retries in the background; OMP_HUB=0 opts out)",
            ep.name
        ));
    }
    let (reply, rx) = oneshot::channel();
    if ep.cmd_tx.send(HubCmd::Send { envelope, reply }).is_err() {
        return Err(format!("hub link closed for {recipient}"));
    }
    match tokio::time::timeout(HUB_CMD_TIMEOUT, rx).await {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(_)) => Err(format!("hub link closed for {recipient}")),
        Err(_) => Err(format!("hub send to {recipient} timed out")),
    }
}

/// Route a model-issued `write agent://<to>` whose recipient this tree does
/// not know: answers a pending hub question when one exists, otherwise sends
/// a fresh envelope through the session's live hub endpoint. `None` while
/// opted out (`OMP_HUB=0`) or when no endpoint is registered (the caller
/// keeps its local "not running" error); a registered endpoint whose hub is
/// unreachable yields a clear error, never silent loss.
pub(crate) async fn route_external_message(
    to: &str,
    content: &str,
) -> Option<Result<String, String>> {
    if hub_opted_out() {
        return None;
    }
    let ep = hub_endpoint_for_owner(&current_owner())?;
    Some(route_external_message_via(&ep, to, content).await)
}

/// The endpoint-explicit core of [`route_external_message`]: resolves the
/// pending-question answer or sends a fresh envelope through this endpoint.
/// Tests call this directly so they do not depend on the process-global
/// session owner (production resolves it once, one runtime per process).
async fn route_external_message_via(
    ep: &Arc<HubEndpoint>,
    to: &str,
    content: &str,
) -> Result<String, String> {
    if !ep.connected.load(AtomicOrdering::SeqCst) {
        return Err(format!(
            "hub unreachable for {to}: endpoint {} is not connected (the hub link retries in the background; OMP_HUB=0 opts out)",
            ep.name
        ));
    }
    // Reply to a pending hub question: `write agent://<sender>` after a
    // reply-expected envelope landed here is an `answer{}` with
    // `causation_id` = the original message, so the hub can close the
    // thread / validate the reply schema.
    if let Some(q) = ep.question_from(to) {
        let (reply, rx) = oneshot::channel();
        if ep
            .cmd_tx
            .send(HubCmd::Answer {
                causation_id: q.message_id.clone(),
                sender: ep.name.clone(),
                payload: content.to_string(),
                reply,
            })
            .is_err()
        {
            return Err(format!("hub link closed for {to}"));
        }
        return match tokio::time::timeout(HUB_CMD_TIMEOUT, rx).await {
            Ok(Ok(Ok(text))) => Ok(text),
            Ok(Ok(Err(e))) => Err(e),
            Ok(Err(_)) => Err(format!("hub link closed for {to}")),
            Err(_) => Err(format!("hub answer to {to} timed out")),
        };
    }
    let mut hub_env = omp_hub::Envelope {
        message_id: crate::persist::uuid_v7(),
        sender: ep.name.clone(),
        recipient: to.to_string(),
        kind: kind::MESSAGE.to_string(),
        ..omp_hub::Envelope::default()
    };
    if content.len() > PAYLOAD_INLINE_MAX {
        match spill_payload(&hub_env.message_id, content.as_bytes()) {
            Ok(path) => {
                hub_env.payload_ref = Some(omp_hub::PayloadRef {
                    path: path.to_string_lossy().into_owned(),
                    sha256: sha256_hex(content.as_bytes()),
                    bytes: content.len() as u64,
                });
            }
            Err(e) => return Err(format!("cannot spill hub payload: {e}")),
        }
    } else {
        hub_env.payload = Some(content.to_string());
    }
    hub_send_envelope(ep, hub_env).await
}

// --- last-acked-seq persistence (hello replays from it after a restart) ---

fn last_seq_path(dir: &Path, name: &str) -> PathBuf {
    dir.join("endpoints").join(format!("{name}.json"))
}

fn read_last_seq(dir: &Path, name: &str) -> u64 {
    let path = last_seq_path(dir, name);
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|v| v.get("last_seq").and_then(Value::as_u64))
        .unwrap_or(0)
}

fn write_last_seq(dir: &Path, name: &str, last_seq: u64) {
    let path = last_seq_path(dir, name);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, json!({ "last_seq": last_seq }).to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

// --- claims guard (SPEC item 3) ---

/// Repo-relative posix path when `path` lives inside a git work tree
/// (`.git` dir or file at the anchor), else `None` — outside a repo the
/// claims guard stays silent.
fn git_repo_relative(path: &Path) -> Option<String> {
    let mut anchor = path;
    loop {
        if anchor.join(".git").exists() {
            let rel = path.strip_prefix(anchor).ok()?;
            return Some(rel.to_string_lossy().replace('\\', "/"));
        }
        match anchor.parent() {
            Some(parent) => anchor = parent,
            None => return None,
        }
    }
}

/// Best-effort check of every path's repo-relative resource: warning lines
/// naming the holder when a *different* live endpoint holds the claim. The
/// apply itself proceeds — the warning rides the tool result; with
/// `OMP_CLAIMS=block` the first conflict is an error instead. Hub down or
/// opted out: no warnings, no errors, no added latency.
pub(crate) async fn claims_check_files(paths: &[PathBuf]) -> Vec<String> {
    let Some(ep) = hub_endpoint_for_owner(&current_owner()) else {
        return Vec::new();
    };
    claims_check_files_via(&ep, paths).await
}

/// Endpoint-explicit core of [`claims_check_files`] (tests call this so they
/// do not depend on the process-global session owner).
async fn claims_check_files_via(ep: &Arc<HubEndpoint>, paths: &[PathBuf]) -> Vec<String> {
    if !ep.connected.load(AtomicOrdering::SeqCst) || hub_opted_out() {
        return Vec::new();
    }
    let mut warnings = Vec::new();
    let mut checked: Vec<String> = Vec::new();
    for path in paths {
        let Some(resource) = git_repo_relative(path) else {
            continue;
        };
        if checked.contains(&resource) {
            continue;
        }
        checked.push(resource.clone());
        let now = std::time::Instant::now();
        {
            let neg = ep.claim_neg.lock();
            if neg
                .get(&resource)
                .is_some_and(|at| now.duration_since(*at) < CLAIM_NEG_CACHE)
            {
                continue;
            }
        }
        let (reply, rx) = oneshot::channel();
        let Ok(()) = ep.cmd_tx.send(HubCmd::Check {
            resource: resource.clone(),
            reply,
        }) else {
            return warnings;
        };
        let holder = match tokio::time::timeout(HUB_CMD_TIMEOUT, rx).await {
            Ok(Ok(holder)) => holder,
            _ => None,
        };
        match holder {
            Some(row) if row.owner != ep.name => {
                let intent = row.intent.as_deref().unwrap_or("no intent");
                let left = row.expires_ms.saturating_sub(omp_hub::now_ms()) / 1000;
                let line = format!(
                    "[hub claim] {resource} is claimed by {} (intent: {intent}; expires in {left}s)"
                , row.owner);
                if claims_block_mode() {
                    return vec![format!("{line} — blocked by OMP_CLAIMS=block")];
                }
                warnings.push(line);
            }
            _ => {
                ep.claim_neg
                    .lock()
                    .insert(resource, std::time::Instant::now());
            }
        }
    }
    warnings
}

/// `OMP_CLAIMS=block` is set (the tools gate their pre-apply checks on it).
pub(crate) fn claims_block_mode_enabled() -> bool {
    claims_block_mode()
}

/// `OMP_CLAIMS=block` pre-apply gate: `Some(reason)` when another endpoint
/// holds a live claim on one of the paths. (The warning-mode check runs
/// after the apply and therefore cannot block it.)
pub(crate) async fn claims_block_reason(paths: &[PathBuf]) -> Option<String> {
    if !claims_block_mode() {
        return None;
    }
    claims_check_files(paths).await.first().map(|line| {
        line.trim_end_matches(" — blocked by OMP_CLAIMS=block")
            .to_string()
    })
}

/// After a successful write by an agent: auto-claim every repo-relative path
/// for this endpoint (ttl [`CLAIM_TTL_S`], renewed on further writes at most
/// every [`CLAIM_RENEW_EVERY`], released at session end). Fire-and-forget.
pub(crate) fn claims_note_writes(paths: &[PathBuf]) {
    let Some(ep) = hub_endpoint_for_owner(&current_owner()) else {
        return;
    };
    claims_note_writes_via(&ep, paths);
}

/// Endpoint-explicit core of [`claims_note_writes`].
fn claims_note_writes_via(ep: &Arc<HubEndpoint>, paths: &[PathBuf]) {
    if !ep.connected.load(AtomicOrdering::SeqCst) || hub_opted_out() {
        return;
    }
    for path in paths {
        let Some(resource) = git_repo_relative(path) else {
            continue;
        };
        let fresh = ep
            .held
            .lock()
            .get(&resource)
            .is_some_and(|c| c.acquired.elapsed() < CLAIM_RENEW_EVERY);
        if fresh {
            continue;
        }
        let (reply, rx) = oneshot::channel();
        if ep.cmd_tx.send(HubCmd::Claim { resource, reply }).is_ok() {
            // Fire-and-forget: a slow hub must never stall a write result.
            tokio::spawn(async move {
                let _ = rx.await;
            });
        }
    }
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
    #[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
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
    /// original sender/recipient/kind/`message_id` are preserved; only the
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
#[allow(clippy::disallowed_methods)] // tests: throwaway runtimes drive async code
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
    /// `dedup`/`payload_ref` tests).
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
            assert!(
                again["text"].as_str().unwrap().contains("Duplicate"),
                "{again}"
            );
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
        // Bus-level refusal semantics: with the hub link opted out the
        // local "not running" error stays the last word (hub routing of
        // these ids has its own tests below).
        let old_hub = std::env::var_os("OMP_HUB");
        std::env::set_var("OMP_HUB", "0");
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
        match old_hub {
            Some(v) => std::env::set_var("OMP_HUB", v),
            None => std::env::remove_var("OMP_HUB"),
        }
    }

    #[tokio::test]
    async fn grandchild_unknown_id_forwards_up_and_receipt_flows_back() {
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
        let _serial = LOCK.lock().await;
        // The "grandparent": a raw stand-in socket whose reply the forwarding
        // parent must return to the grandchild.
        let dir = std::env::temp_dir().join(format!("omt-a2a-gp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let gp_sock = dir.join("gp.sock");
        let listener = UnixListener::bind(&gp_sock).unwrap();
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
            let mut out =
                json!({"ok": true, "text": format!("routed-by-grandparent:{}", env.recipient)})
                    .to_string();
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

    // --- Hub link tests (SPEC batch 2): every test runs a real hub server
    // on an explicit temp dir (no process-env mutation; tests in this file
    // run concurrently with non-a2a tests) and drives the endpoint/command
    // surface directly.

    /// Serializes the hub tests with the bus tests (global owner/endpoint
    /// registration) and pins `OMP_CLAIMS` to its default for the duration.
    struct HubTestGuard {
        _serial: tokio::sync::MutexGuard<'static, ()>,
        claims: Option<std::ffi::OsString>,
    }

    impl HubTestGuard {
        async fn new() -> Self {
            let guard = Self {
                _serial: LOCK.lock().await,
                claims: std::env::var_os("OMP_CLAIMS"),
            };
            std::env::remove_var("OMP_CLAIMS");
            guard
        }
    }

    impl Drop for HubTestGuard {
        fn drop(&mut self) {
            match &self.claims {
                Some(v) => std::env::set_var("OMP_CLAIMS", v),
                None => std::env::remove_var("OMP_CLAIMS"),
            }
        }
    }

    /// Start a hub server on a fresh tempdir; wait until it answers.
    async fn start_test_hub() -> (tempfile::TempDir, omp_hub::HubDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let hub = omp_hub::HubDir::explicit(dir.path().to_path_buf(), None);
        let serve_hub = hub.clone();
        tokio::spawn(async move {
            let _ = omp_hub::serve(&serve_hub).await;
        });
        for _ in 0..100 {
            if UnixStream::connect(&hub.sock).await.is_ok() {
                return (dir, hub);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("test hub did not come up");
    }

    async fn connect_test_hub(hub: &omp_hub::HubDir) -> omp_hub::HubClient {
        for _ in 0..100 {
            if let Ok(c) = omp_hub::HubClient::connect_with(hub.clone(), false).await {
                return c;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("cannot connect to test hub");
    }

    async fn wait_for<F: Fn() -> bool>(what: &str, f: F) {
        for _ in 0..200 {
            if f() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("timed out waiting for {what}");
    }

    /// Poll a message's hub state until it reaches `state`.
    async fn wait_for_state(client: &omp_hub::HubClient, message_id: &str, state: &str) {
        for _ in 0..200 {
            if let Ok(v) = client.status_message(message_id).await {
                if v["state"] == state {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("timed out waiting for {state} on {message_id}");
    }

    /// Start an endpoint on an explicit hub and wait until its `hello`
    /// registered it.
    async fn start_test_endpoint(
        owner: &Owner,
        name: &str,
        hub: &omp_hub::HubDir,
    ) -> Arc<HubEndpoint> {
        let ep = start_hub_endpoint_with(&owner.as_ref().expect("test owner").clone(), name, hub);
        wait_for("endpoint connected", || {
            ep.connected.load(AtomicOrdering::SeqCst)
        })
        .await;
        ep
    }

    #[tokio::test]
    async fn external_pane_id_routes_through_hub_with_receipt() {
        let (dir, hub) = start_test_hub().await;
        let _guard = HubTestGuard::new().await;
        let owner: std::sync::Arc<str> = "ext-pane-owner".into();
        let ep = start_test_endpoint(&Some(owner.clone()), "session:test-external", &hub).await;
        let text = route_external_message_via(&ep, "w4:pZZ", "pane mail")
            .await
            .expect("receipt");
        // The pane is dead in test envs (no herdr): fail-fast state, clearly
        // reported with the message id — never silent loss.
        assert!(text.contains("state=failed"), "{text}");
        assert!(text.contains("message_id"), "{text}");
        assert!(text.contains("dead pane"), "{text}");
        let client = connect_test_hub(&hub).await;
        let inbox = client.inbox("w4:pZZ", None, None).await.unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].payload.as_deref(), Some("pane mail"));
        shutdown_hub_endpoint(&owner);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn external_unknown_id_routes_through_hub_queued() {
        let (dir, hub) = start_test_hub().await;
        let _guard = HubTestGuard::new().await;
        let owner: std::sync::Arc<str> = "ext-unknown-owner".into();
        let ep = start_test_endpoint(&Some(owner.clone()), "session:test-external", &hub).await;
        let text = route_external_message_via(&ep, "ghost-agent", "anybody there?")
            .await
            .expect("receipt");
        assert!(text.contains("state=queued"), "{text}");
        assert!(text.contains("message_id"), "{text}");
        let client = connect_test_hub(&hub).await;
        let inbox = client.inbox("ghost-agent", None, None).await.unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].payload.as_deref(), Some("anybody there?"));
        shutdown_hub_endpoint(&owner);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn endpoint_delivery_injected_at_turn_boundary_and_acked() {
        let (dir, hub) = start_test_hub().await;
        let _guard = HubTestGuard::new().await;
        let owner: std::sync::Arc<str> = "inject-owner".into();
        let owner: Owner = Some(owner);
        let _ep = start_test_endpoint(&owner, "session:test-inject", &hub).await;
        without_parent_bus(async {
            let sender = connect_test_hub(&hub).await;
            let receipt = sender
                .send_note(
                    "w4:p8T",
                    "session:test-inject",
                    kind::MESSAGE,
                    "hello endpoint",
                )
                .await
                .unwrap();
            assert_eq!(receipt.state, omp_hub::State::DELIVERED, "{receipt:?}");
            // The deliver push lands in the session's delivery queue and
            // fires the same mechanism task results use.
            wait_for("hub delivery", || has_pending_for(&owner)).await;
            let drained = take_deliveries_for(&owner);
            assert_eq!(drained.len(), 1, "{drained:?}");
            assert_eq!(drained[0].from, "w4:p8T");
            assert!(
                drained[0].text.contains("hello endpoint"),
                "{}",
                drained[0].text
            );
            // The drain starts the turn: the endpoint acks `read` and the hub
            // state follows (poll — the ack command is processed async).
            wait_for("read ack", || {
                !HUB_INFLIGHT.lock().iter().any(|(o, _, _)| o == &owner)
            })
            .await;
            wait_for_state(&sender, &receipt.message_id, omp_hub::State::READ).await;
            let status = sender.status_message(&receipt.message_id).await.unwrap();
            assert_eq!(
                status["state"],
                Value::from(omp_hub::State::READ),
                "{status}"
            );
            // The read ack advanced the persisted last acked seq.
            let persisted = read_last_seq(dir.path(), "session:test-inject");
            assert!(
                persisted >= receipt.seq,
                "last_seq {persisted} >= {}",
                receipt.seq
            );
        })
        .await;
        shutdown_hub_endpoint(owner.as_ref().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn reply_expected_question_answered_and_schema_violation_is_write_error() {
        let (dir, hub) = start_test_hub().await;
        let _guard = HubTestGuard::new().await;
        let owner: std::sync::Arc<str> = "answer-owner".into();
        let owner: Owner = Some(owner);
        let ep = start_test_endpoint(&owner, "session:test-answer", &hub).await;
        without_parent_bus(async {
            let asker = connect_test_hub(&hub).await;
            let schema = json!({
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": false
            });
            let mut question = omp_hub::Envelope {
                message_id: String::new(),
                sender: "asker".to_string(),
                recipient: "session:test-answer".to_string(),
                kind: "agent.question".to_string(),
                reply_schema: Some(schema.clone()),
                expected: Some("ack".to_string()),
                payload: Some("what is 6*7?".to_string()),
                ..omp_hub::Envelope::default()
            };
            let receipt = asker.send(question.clone()).await.unwrap();
            assert_eq!(receipt.state, omp_hub::State::DELIVERED);
            wait_for("question injected", || has_pending_for(&owner)).await;
            let drained = take_deliveries_for(&owner);
            assert_eq!(drained.len(), 1);
            // The injected text tells the model how to reply and carries the schema.
            assert!(
                drained[0].text.contains("write agent://asker"),
                "{}",
                drained[0].text
            );
            assert!(
                drained[0].text.contains("JSON Schema"),
                "{}",
                drained[0].text
            );
            assert!(ep.question_from("asker").is_some());
            // Valid reply: `write agent://asker` becomes an `answer{}` that
            // closes the thread.
            let answered = route_external_message_via(&ep, "asker", "{\"answer\":42}")
                .await
                .expect("valid answer");
            assert!(answered.contains("answered"), "{answered}");
            assert!(ep.question_from("asker").is_none(), "question consumed");
            let status = asker.status_message(&receipt.message_id).await.unwrap();
            assert_eq!(
                status["state"],
                Value::from(omp_hub::State::ANSWERED),
                "{status}"
            );
            // Second question, schema-violating reply: the write surfaces the
            // hub's validation error.
            question.message_id = String::new();
            question.payload = Some("now?".to_string());
            let receipt2 = asker.send(question).await.unwrap();
            wait_for("second question injected", || has_pending_for(&owner)).await;
            let _ = take_deliveries_for(&owner);
            let bad = route_external_message_via(&ep, "asker", "forty-two")
                .await
                .expect_err("schema violation must error");
            assert!(bad.contains("reply_schema"), "{bad}");
            let status2 = asker.status_message(&receipt2.message_id).await.unwrap();
            assert_ne!(
                status2["state"],
                Value::from(omp_hub::State::ANSWERED),
                "{status2}"
            );
        })
        .await;
        shutdown_hub_endpoint(ep.owner.as_ref().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn claims_warn_between_endpoints_auto_claim_released_at_end() {
        let (dir, hub) = start_test_hub().await;
        let _guard = HubTestGuard::new().await;
        without_parent_bus(async {
            // A git work tree on disk: both endpoints edit the same file.
            let repo = tempfile::TempDir::new().unwrap();
            std::fs::create_dir_all(repo.path().join(".git")).unwrap();
            let file = repo.path().join("src/lib.rs");
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, "pub fn a() {}\n").unwrap();

            let owner_a: std::sync::Arc<str> = "owner-a".into();
            let owner_b: std::sync::Arc<str> = "owner-b".into();
            let ep_a =
                start_test_endpoint(&Some(owner_a.clone()), "session:test-claims-a", &hub).await;
            let ep_b =
                start_test_endpoint(&Some(owner_b.clone()), "session:test-claims-b", &hub).await;

            // A writes: auto-claims the repo-relative path.
            claims_note_writes_via(&ep_a, std::slice::from_ref(&file));
            wait_for("auto-claim granted", || {
                ep_a.held.lock().contains_key("src/lib.rs")
            })
            .await;
            let client = connect_test_hub(&hub).await;
            let check = client
                .check_claim("src/lib.rs")
                .await
                .unwrap()
                .expect("held");
            assert_eq!(check.owner, "session:test-claims-a");

            // A's own re-check is silent (negative-cacheable, no warning).
            let own = claims_check_files_via(&ep_a, std::slice::from_ref(&file)).await;
            assert!(own.is_empty(), "{own:?}");
            // B (the other session) gets the warning line with holder+intent.
            let warnings = claims_check_files_via(&ep_b, std::slice::from_ref(&file)).await;
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert!(
                warnings[0].contains("session:test-claims-a"),
                "{warnings:?}"
            );
            assert!(warnings[0].contains("src/lib.rs"), "{warnings:?}");
            assert!(warnings[0].contains("intent"), "{warnings:?}");
            // B still proceeds (warning, not block) and the check re-fires
            // (positive results are not cached away).

            // Session A ends: its claims are released (the release handshake
            // completes on a dedicated thread right after shutdown returns).
            shutdown_hub_endpoint(&owner_a);
            for _ in 0..200 {
                if client.check_claim("src/lib.rs").await.unwrap().is_none() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            let free = client.check_claim("src/lib.rs").await.unwrap();
            assert!(free.is_none(), "{free:?}");
        })
        .await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn claims_silent_and_routes_clear_when_hub_down() {
        let _guard = HubTestGuard::new().await;
        without_parent_bus(async {
            // No endpoint registered: the claims guard is silent.
            let repo = tempfile::TempDir::new().unwrap();
            std::fs::create_dir_all(repo.path().join(".git")).unwrap();
            let file = repo.path().join("lib.rs");
            std::fs::write(&file, "x").unwrap();
            assert!(claims_check_files(std::slice::from_ref(&file))
                .await
                .is_empty());
            claims_note_writes(std::slice::from_ref(&file));
        })
        .await;
        // A registered endpoint whose hub can never come up (the "dir" is a
        // file): a write gets a clear hub-unreachable error, never silence.
        let blocker = tempfile::NamedTempFile::new().unwrap();
        let dead_hub = omp_hub::HubDir::explicit(blocker.path().to_path_buf(), None);
        let owner: std::sync::Arc<str> = "down-owner".into();
        let ep = start_hub_endpoint_with(&owner, "session:test-down", &dead_hub);
        let err = route_external_message_via(&ep, "w4:pZZ", "nowhere")
            .await
            .expect_err("hub down");
        assert!(err.contains("hub unreachable"), "{err}");
        shutdown_hub_endpoint(&owner);
    }

    #[tokio::test]
    async fn external_route_opted_out_returns_none() {
        let _guard = HubTestGuard::new().await;
        std::env::set_var("OMP_HUB", "0");
        assert!(route_external_message("ghost", "hi").await.is_none());
        assert!(start_hub_endpoint(&crate::persist::uuid_v7().into()).is_none());
        std::env::remove_var("OMP_HUB");
    }
}
