//! Tokio UDS JSON-RPC 2.0 server (`SPEC.md`): newline-delimited requests,
//! `deliver{envelope}` push to live endpoints over bounded per-endpoint
//! channels (a slow endpoint never blocks the others), urgent priority
//! first on replay, at-least-once with client-side dedup on `message_id`.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;

use crate::envelope::{
    looks_like_pane, new_message_id, normalize_address, Envelope, Kind, Priority, State,
    DEFAULT_REPLY_DEADLINE_MS, SENDER_HUB,
};
use crate::pane;
use crate::schema::{validate_reply, ReplyValidation};
use crate::store::{ClaimOutcome, StatusReport, Store};

/// Largest accepted request line.
const MAX_LINE: usize = 1 << 20;
/// Per-endpoint deliver channel bound (backpressure for slow endpoints).
const DELIVER_CAP: usize = 512;
/// Deadline sweep cadence.
const SWEEP_INTERVAL: Duration = Duration::from_secs(1);
/// Idle/orphan watchdog cadence (a shorter idle limit ticks faster).
const WATCH_INTERVAL: Duration = Duration::from_secs(2);
/// Default idle limit: no client connection and no in-flight pane delivery
/// for this long and the hub exits; clients auto-start it again on demand.
pub const DEFAULT_IDLE_EXIT: Duration = Duration::from_mins(15);
/// Overrides [`DEFAULT_IDLE_EXIT`] in seconds; `0` disables the idle exit.
pub const IDLE_EXIT_ENV: &str = "OMP_HUB_IDLE_EXIT_SECS";

/// JSON-RPC error codes (app range -32xxx).
const E_PARSE: i64 = -32700;
const E_INVALID: i64 = -32600;
const E_NOT_FOUND: i64 = -32004;
const E_SCHEMA: i64 = -32022;
const E_ANSWERED: i64 = -32023;

/// Resolved hub locations.
#[derive(Debug, Clone)]
pub struct HubDir {
    pub dir: PathBuf,
    pub sock: PathBuf,
}

impl HubDir {
    /// Resolve from `OMP_HUB_DIR` / `OMP_HUB_SOCK` or the defaults.
    #[must_use]
    pub fn resolve() -> Self {
        let dir = crate::default_dir();
        let sock = crate::socket_path(&dir);
        Self { dir, sock }
    }

    /// Explicit locations (what `omt hub serve --dir/--sock` uses).
    #[must_use]
    pub fn explicit(dir: PathBuf, sock: Option<PathBuf>) -> Self {
        let sock = sock.unwrap_or_else(|| crate::socket_path(&dir));
        Self { dir, sock }
    }
}

/// Live endpoint registration: one bounded deliver channel per agent.
#[derive(Debug)]
struct Endpoint {
    deliver_tx: mpsc::Sender<Arc<Envelope>>,
    gen: u64,
}

#[derive(Default)]
struct Registry {
    endpoints: Mutex<HashMap<String, Endpoint>>,
    gens: AtomicU64,
}

struct Server {
    store: Arc<Store>,
    registry: Registry,
    activity: Arc<Activity>,
}

/// Live work that keeps the hub up: client connections (every registered
/// agent holds one) and in-flight pane deliveries.
struct Activity {
    busy: AtomicUsize,
    /// When `busy` last dropped to zero (or the server started).
    idle_since: Mutex<Instant>,
}

impl Activity {
    fn enter(self: &Arc<Self>) -> BusyGuard {
        self.busy.fetch_add(1, Ordering::SeqCst);
        BusyGuard(self.clone())
    }

    /// How long nothing has been live; `None` while anything is.
    fn idle_for(&self) -> Option<Duration> {
        let since = self.idle_since.lock();
        (self.busy.load(Ordering::SeqCst) == 0).then(|| since.elapsed())
    }
}

/// Holds one unit of [`Activity`] until dropped.
struct BusyGuard(Arc<Activity>);

impl Drop for BusyGuard {
    fn drop(&mut self) {
        let mut since = self.0.idle_since.lock();
        if self.0.busy.fetch_sub(1, Ordering::SeqCst) == 1 {
            *since = Instant::now();
        }
    }
}

/// Why [`serve`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeExit {
    /// No client and no in-flight pane delivery for the idle limit.
    Idle,
    /// The hub dir or socket was deleted or replaced and nothing is
    /// connected: no client can reach this instance any more.
    Orphaned,
}

/// The idle limit from [`IDLE_EXIT_ENV`] (default [`DEFAULT_IDLE_EXIT`];
/// `0` means never).
#[must_use]
pub fn idle_exit_from_env() -> Option<Duration> {
    match std::env::var(IDLE_EXIT_ENV).map(|v| v.trim().parse::<u64>()) {
        Ok(Ok(0)) => None,
        Ok(Ok(secs)) => Some(Duration::from_secs(secs)),
        _ => Some(DEFAULT_IDLE_EXIT),
    }
}

/// Run the hub server with the idle limit from [`idle_exit_from_env`].
///
/// # Errors
/// See [`serve_with`].
pub async fn serve(hub: &HubDir) -> Result<ServeExit> {
    serve_with(hub, idle_exit_from_env()).await
}

/// Run the hub server until killed or until it has nothing left to serve.
/// Holds a single-instance lock on `<dir>/hub.lock`; a second `serve`
/// refuses to start. While no client is connected (so no agent is
/// registered) and no pane delivery is in flight, it returns
/// [`ServeExit::Orphaned`] as soon as its dir or socket file is gone or
/// replaced, and [`ServeExit::Idle`] after `idle_exit` (`None`: never).
///
/// # Errors
/// Lock held, socket bind, or store open failures.
pub async fn serve_with(hub: &HubDir, idle_exit: Option<Duration>) -> Result<ServeExit> {
    std::fs::create_dir_all(&hub.dir).with_context(|| format!("create {}", hub.dir.display()))?;
    let lock_path = hub.dir.join("hub.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("open {}", lock_path.display()))?;
    lock_file
        .try_lock()
        .map_err(|e| anyhow::anyhow!("hub already running ({}: {e})", lock_path.display()))?;

    if hub.sock.exists() {
        std::fs::remove_file(&hub.sock).context("remove stale hub socket")?;
    }
    if let Some(parent) = hub.sock.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener =
        UnixListener::bind(&hub.sock).with_context(|| format!("bind {}", hub.sock.display()))?;
    let sock_ino = std::fs::symlink_metadata(&hub.sock).ok().map(|m| m.ino());
    let server = Arc::new(Server {
        store: Arc::new(Store::open(&hub.dir)?),
        registry: Registry::default(),
        activity: Arc::new(Activity {
            busy: AtomicUsize::new(0),
            idle_since: Mutex::new(Instant::now()),
        }),
    });
    tracing::info!(dir = %hub.dir.display(), sock = %hub.sock.display(), "hub serving");

    // Deadline sweeper: expire reply-expected envelopes and notify senders.
    let sweeper = {
        let server = server.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(SWEEP_INTERVAL);
            loop {
                tick.tick().await;
                if let Err(e) = sweep_deadlines(&server) {
                    tracing::warn!(error = %e, "deadline sweep failed");
                }
            }
        })
    };

    let mut watch = tokio::time::interval(
        idle_exit
            .map_or(WATCH_INTERVAL, |d| d.min(WATCH_INTERVAL))
            .max(Duration::from_millis(10)),
    );
    watch.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let exit = loop {
        tokio::select! {
            accepted = listener.accept() => {
                if let Ok((stream, _)) = accepted {
                    spawn_conn(&server, stream);
                }
            }
            _ = watch.tick() => {
                let Some(reason) = exit_reason(&server, hub, sock_ino, idle_exit) else {
                    continue;
                };
                // A client that connected right before the decision: serve
                // it and re-check on the next tick.
                if let Ok(Ok((stream, _))) =
                    tokio::time::timeout(Duration::ZERO, listener.accept()).await
                {
                    spawn_conn(&server, stream);
                    continue;
                }
                break reason;
            }
        }
    };
    sweeper.abort();
    if sock_ino.is_some() && std::fs::symlink_metadata(&hub.sock).ok().map(|m| m.ino()) == sock_ino
    {
        let _ = std::fs::remove_file(&hub.sock);
    }
    drop(listener);
    tracing::info!(?exit, dir = %hub.dir.display(), "hub exiting");
    Ok(exit)
}

fn spawn_conn(server: &Arc<Server>, stream: UnixStream) {
    let busy = server.activity.enter();
    let server = server.clone();
    tokio::spawn(async move {
        let _busy = busy;
        if let Err(e) = handle_conn(server, stream).await {
            tracing::debug!(error = %e, "connection closed");
        }
    });
}

/// The idle-exit rule. Never while a connection (so a registered agent) or
/// a pane delivery is live; otherwise exit when this instance became
/// unreachable (dir gone, socket gone or replaced) or stayed idle for
/// `idle_exit`.
fn exit_reason(
    server: &Server,
    hub: &HubDir,
    sock_ino: Option<u64>,
    idle_exit: Option<Duration>,
) -> Option<ServeExit> {
    let idle_for = server.activity.idle_for()?;
    if !server.registry.endpoints.lock().is_empty() {
        return None;
    }
    let sock_lost = sock_ino
        .is_some_and(|ino| std::fs::symlink_metadata(&hub.sock).ok().map(|m| m.ino()) != Some(ino));
    if sock_lost || !hub.dir.is_dir() {
        return Some(ServeExit::Orphaned);
    }
    idle_exit
        .is_some_and(|limit| idle_for >= limit)
        .then_some(ServeExit::Idle)
}

/// Mark overdue envelopes `expired` and send their senders a delivery
/// notice (kind `delivery.ack`, `correlation_id` = original message).
fn sweep_deadlines(server: &Arc<Server>) -> Result<()> {
    let now = crate::now_ms();
    let due = server.store.due_expired(now)?;
    for row in due {
        let env = row.envelope;
        if !server
            .store
            .transition(&env.message_id, State::EXPIRED, now)?
        {
            continue;
        }
        let mut notice = Envelope {
            message_id: new_message_id(),
            correlation_id: Some(env.message_id.clone()),
            causation_id: Some(env.message_id.clone()),
            sender: SENDER_HUB.to_string(),
            recipient: env.sender.clone(),
            kind: Kind::DELIVERY_ACK.to_string(),
            priority: Priority::URGENT,
            payload: Some(
                json!({
                    "notice": "reply_deadline_expired",
                    "message_id": env.message_id,
                    "kind": env.kind,
                    "recipient": env.recipient,
                    "deadline_ms": env.deadline,
                    "default_reply_deadline_ms": DEFAULT_REPLY_DEADLINE_MS,
                })
                .to_string(),
            ),
            ..Envelope::default()
        };
        if let Err(e) = server.store.insert(&mut notice, now) {
            tracing::warn!(error = %e, "notice insert failed");
            continue;
        }
        dispatch(server, notice);
    }
    Ok(())
}

impl Registry {
    /// Whether `agent` currently holds a live registered endpoint.
    fn has_endpoint(&self, agent: &str) -> bool {
        self.endpoints.lock().contains_key(agent)
    }
}

/// Route one stored envelope to its recipient: live push when registered,
/// pane delivery for `w4:pXX`, else it stays in the store until the
/// recipient reconnects or pulls its inbox. Returns `true` when a live
/// endpoint took it (the store marks it `delivered`); `false` when it stays
/// queued (store-only or handed to the async pane delivery).
fn dispatch(server: &Arc<Server>, env: Envelope) -> bool {
    let recipient = env.recipient.clone();
    if let Some(ep) = server.registry.endpoints.lock().get(&recipient) {
        match ep.deliver_tx.try_send(Arc::new(env.clone())) {
            Ok(()) => {
                let _ = server
                    .store
                    .transition(&env.message_id, State::DELIVERED, crate::now_ms());
                true
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!(recipient, "endpoint channel full; leaving message in store");
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                drop_ep(&server.registry, &recipient, ep.gen);
                false
            }
        }
    } else if looks_like_pane(&recipient) {
        pane::spawn_delivery(server.store.clone(), env, server.activity.enter());
        false
    } else {
        false
    }
}

fn drop_ep(registry: &Registry, agent: &str, gen: u64) {
    let mut eps = registry.endpoints.lock();
    if eps.get(agent).is_some_and(|e| e.gen == gen) {
        eps.remove(agent);
    }
}

enum OutMsg {
    Response(Value),
    Deliver(Arc<Envelope>),
}

/// One client connection: read requests, write responses + deliver pushes.
async fn handle_conn(server: Arc<Server>, stream: UnixStream) -> Result<()> {
    let (read, mut write) = stream.into_split();
    let (resp_tx, mut resp_rx) = mpsc::unbounded_channel::<Value>();
    let (deliver_tx, mut deliver_rx) = mpsc::channel::<Arc<Envelope>>(DELIVER_CAP);
    let mut registered: Option<(String, u64)> = None;

    let writer = tokio::spawn(async move {
        loop {
            let msg = tokio::select! {
                r = resp_rx.recv() => match r {
                    Some(v) => OutMsg::Response(v),
                    None => return,
                },
                d = deliver_rx.recv() => match d {
                    Some(e) => OutMsg::Deliver(e),
                    None => return,
                },
            };
            let line = match msg {
                OutMsg::Response(v) => serde_json::to_string(&v),
                OutMsg::Deliver(e) => serde_json::to_string(&json!({
                    "jsonrpc": "2.0",
                    "method": "deliver",
                    "params": {"envelope": *e},
                })),
            };
            let Ok(line) = line else { return };
            if write.write_all(line.as_bytes()).await.is_err()
                || write.write_all(b"\n").await.is_err()
                || write.flush().await.is_err()
            {
                return;
            }
        }
    });

    let mut reader = BufReader::new(read).take(MAX_LINE as u64);
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let Ok(req) = serde_json::from_str::<Value>(&line) else {
            let _ = resp_tx.send(err_msg(&Value::Null, E_PARSE, "parse error"));
            continue;
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);
        let resp = dispatch_request(&server, method, &params, &deliver_tx, &mut registered).await;
        if !id.is_null() {
            let _ = resp_tx.send(match resp {
                Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                Err(e) => err_msg(&id, e.code, &e.message),
            });
        }
    }
    drop(resp_tx);
    drop(deliver_tx);
    if let Some((agent, gen)) = registered {
        drop_ep(&server.registry, &agent, gen);
    }
    writer.abort();
    Ok(())
}

/// A handler error rendered as a JSON-RPC error object.
struct RpcError {
    code: i64,
    message: String,
}

fn err_msg(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn invalid(message: impl Into<String>) -> RpcError {
    RpcError {
        code: E_INVALID,
        message: message.into(),
    }
}

fn not_found(message: impl Into<String>) -> RpcError {
    RpcError {
        code: E_NOT_FOUND,
        message: message.into(),
    }
}

fn bad_schema(errors: &[String]) -> RpcError {
    RpcError {
        code: E_SCHEMA,
        message: format!(
            "answer failed reply_schema validation ({} errors)",
            errors.len()
        ),
    }
}

fn str_param<'v>(params: &'v Value, name: &str) -> Option<&'v str> {
    params.get(name).and_then(Value::as_str)
}

#[allow(clippy::too_many_lines)] // one arm per wire method; each arm is short.
async fn dispatch_request(
    server: &Arc<Server>,
    method: &str,
    params: &Value,
    deliver_tx: &mpsc::Sender<Arc<Envelope>>,
    registered: &mut Option<(String, u64)>,
) -> std::result::Result<Value, RpcError> {
    match method {
        "hello" => {
            let agent = normalize_address(
                str_param(params, "agent").ok_or_else(|| invalid("hello: missing agent"))?,
            );
            let last_seq = params.get("last_seq").and_then(Value::as_u64).unwrap_or(0);
            server
                .store
                .hello(&agent, last_seq, crate::now_ms())
                .map_err(|e| invalid(format!("hello: {e}")))?;
            let gen = server.registry.gens.fetch_add(1, Ordering::Relaxed);
            server.registry.endpoints.lock().insert(
                agent.clone(),
                Endpoint {
                    deliver_tx: deliver_tx.clone(),
                    gen,
                },
            );
            *registered = Some((agent.clone(), gen));
            // Replay undelivered/unacked from last_seq, urgent first.
            let backlog = server
                .store
                .inbox(&agent, last_seq, DELIVER_CAP, true)
                .map_err(|e| invalid(format!("hello replay: {e}")))?;
            let replayed = backlog.len();
            for row in backlog {
                if deliver_tx.try_send(Arc::new(row.envelope)).is_err() {
                    break;
                }
            }
            let last = server
                .store
                .last_seq_for(&agent)
                .map_err(|e| invalid(format!("hello: {e}")))?;
            Ok(json!({"agent": agent, "last_seq": last, "replayed": replayed}))
        }
        "send" => {
            let mut env: Envelope = serde_json::from_value(
                params
                    .get("envelope")
                    .cloned()
                    .ok_or_else(|| invalid("send: missing envelope"))?,
            )
            .map_err(|e| invalid(format!("send: bad envelope: {e}")))?;
            if env.message_id.is_empty() || env.sender.is_empty() || env.recipient.is_empty() {
                return Err(invalid("send: message_id, sender, recipient are required"));
            }
            env.sender = normalize_address(&env.sender);
            env.recipient = normalize_address(&env.recipient);
            let now = crate::now_ms();
            let outcome = server
                .store
                .insert(&mut env, now)
                .map_err(|e| invalid(format!("send: {e}")))?;
            if outcome.dedup {
                // Report the freshest stored state: a dedup resend may find
                // the original already delivered/answered since.
                let state = server
                    .store
                    .get(&env.message_id)
                    .map_err(|e| invalid(format!("send: {e}")))?
                    .and_then(|row| row.envelope.state)
                    .unwrap_or(outcome.state);
                return Ok(json!({
                    "message_id": env.message_id,
                    "seq": outcome.seq,
                    "state": state,
                    "dedup": true,
                }));
            }
            // Dead-pane guard: fail fast, state failed. A live registered
            // endpoint wins over herdr typing: an omt pane serving this id
            // takes hub mail through its endpoint, so skip the guard (and
            // the pane delivery) for registered recipients.
            if looks_like_pane(&env.recipient)
                && !server.registry.has_endpoint(&env.recipient)
                && !pane::pane_exists(&env.recipient).await.unwrap_or(false)
            {
                let _ = server
                    .store
                    .transition(&env.message_id, State::FAILED, crate::now_ms());
                return Ok(json!({
                    "message_id": env.message_id,
                    "seq": outcome.seq,
                    "state": State::FAILED,
                    "error": format!("dead pane: {} not in `herdr agent list`", env.recipient),
                }));
            }
            let delivered = dispatch(server, env.clone());
            let state = if delivered {
                State::DELIVERED.to_string()
            } else {
                outcome.state
            };
            Ok(json!({
                "message_id": env.message_id,
                "seq": outcome.seq,
                "state": state,
            }))
        }
        "ack" => {
            let id = str_param(params, "message_id")
                .ok_or_else(|| invalid("ack: missing message_id"))?;
            let state = str_param(params, "state")
                .ok_or_else(|| invalid("ack: state must be delivered|read"))?;
            if !matches!(state, State::DELIVERED | State::READ) {
                return Err(invalid("ack: state must be delivered|read"));
            }
            let ok = server
                .store
                .transition(id, state, crate::now_ms())
                .map_err(|e| invalid(format!("ack: {e}")))?;
            if !ok {
                return Err(not_found(format!(
                    "no such message or bad transition: {id}"
                )));
            }
            Ok(json!({"message_id": id, "state": state, "ok": true}))
        }
        "answer" => {
            let causation_id = str_param(params, "causation_id")
                .ok_or_else(|| invalid("answer: missing causation_id"))?
                .to_string();
            let mut answer: Envelope = serde_json::from_value(
                params
                    .get("envelope")
                    .cloned()
                    .ok_or_else(|| invalid("answer: missing envelope"))?,
            )
            .map_err(|e| invalid(format!("answer: bad envelope: {e}")))?;
            let original = server
                .store
                .get(&causation_id)
                .map_err(|e| invalid(format!("answer: {e}")))?
                .ok_or_else(|| not_found(format!("no such message: {causation_id}")))?;
            let original = original.envelope;
            if original.state.as_deref() == Some(State::ANSWERED) {
                return Err(RpcError {
                    code: E_ANSWERED,
                    message: format!("{causation_id} already answered"),
                });
            }
            match validate_reply(
                original.reply_schema.as_ref(),
                answer.payload.as_deref().unwrap_or(""),
            ) {
                ReplyValidation::Valid | ReplyValidation::NoSchema => {}
                other => {
                    let errors = other.error_detail().unwrap_or_default();
                    return Err(bad_schema(&errors));
                }
            }
            if answer.message_id.is_empty() {
                answer.message_id = new_message_id();
            }
            answer.sender = normalize_address(&answer.sender);
            answer.recipient.clone_from(&original.sender);
            if answer.correlation_id.is_none() {
                answer.correlation_id = Some(
                    original
                        .correlation_id
                        .clone()
                        .unwrap_or_else(|| original.message_id.clone()),
                );
            }
            if answer.kind.is_empty() {
                answer.kind = Kind::ANSWER.to_string();
            }
            let now = crate::now_ms();
            server
                .store
                .set_reply(&causation_id, &answer.message_id)
                .map_err(|e| invalid(format!("answer: {e}")))?;
            if !server
                .store
                .transition(&causation_id, State::ANSWERED, now)
                .map_err(|e| invalid(format!("answer: {e}")))?
            {
                return Err(not_found(format!("cannot transition {causation_id}")));
            }
            let outcome = server
                .store
                .insert(&mut answer, now)
                .map_err(|e| invalid(format!("answer: {e}")))?;
            let answer_id = answer.message_id.clone();
            if !outcome.dedup {
                let delivered = dispatch(server, answer.clone());
                if delivered {
                    if let Some(row) = server
                        .store
                        .get(&answer_id)
                        .map_err(|e| invalid(format!("answer: {e}")))?
                    {
                        return Ok(json!({
                            "question": {"message_id": causation_id, "state": State::ANSWERED},
                            "answer": {
                                "message_id": answer_id,
                                "seq": outcome.seq,
                                "state": row.envelope.state.as_deref().unwrap_or(State::DELIVERED),
                            },
                        }));
                    }
                }
            }
            Ok(json!({
                "question": {"message_id": causation_id, "state": State::ANSWERED},
                "answer": {
                    "message_id": answer_id,
                    "seq": outcome.seq,
                    "state": outcome.state,
                },
            }))
        }
        "inbox" => {
            let recipient = normalize_address(
                str_param(params, "recipient")
                    .ok_or_else(|| invalid("inbox: missing recipient"))?,
            );
            let since = params.get("since_seq").and_then(Value::as_u64).unwrap_or(0);
            let limit = usize::try_from(params.get("limit").and_then(Value::as_u64).unwrap_or(50))
                .unwrap_or(50);
            let rows = server
                .store
                .inbox(&recipient, since, limit, false)
                .map_err(|e| invalid(format!("inbox: {e}")))?;
            let envelopes: Vec<Value> = rows.into_iter().map(|r| json!(r.envelope)).collect();
            Ok(json!({"recipient": recipient, "envelopes": envelopes}))
        }
        "status" => {
            let now = crate::now_ms();
            if let Some(id) = str_param(params, "message_id") {
                let row = server
                    .store
                    .get(id)
                    .map_err(|e| invalid(format!("status: {e}")))?
                    .ok_or_else(|| not_found(format!("no such message: {id}")))?;
                return Ok(json!(row.envelope));
            }
            let report: StatusReport = match str_param(params, "recipient") {
                Some(r) => server
                    .store
                    .status_recipient(&normalize_address(r), now)
                    .map_err(|e| invalid(format!("status: {e}")))?,
                None => server
                    .store
                    .status_all(now)
                    .map_err(|e| invalid(format!("status: {e}")))?,
            };
            serde_json::to_value(report).map_err(|e| invalid(format!("status: {e}")))
        }
        "claim" => {
            let resource =
                str_param(params, "resource").ok_or_else(|| invalid("claim: missing resource"))?;
            let owner =
                str_param(params, "owner").ok_or_else(|| invalid("claim: missing owner"))?;
            let ttl = params.get("ttl_s").and_then(Value::as_u64).unwrap_or(300);
            let intent = str_param(params, "intent");
            let outcome: ClaimOutcome = server
                .store
                .claim(resource, owner, ttl, intent, crate::now_ms())
                .map_err(|e| invalid(format!("claim: {e}")))?;
            serde_json::to_value(outcome).map_err(|e| invalid(format!("claim: {e}")))
        }
        "release" => {
            let resource = str_param(params, "resource")
                .ok_or_else(|| invalid("release: missing resource"))?;
            let owner =
                str_param(params, "owner").ok_or_else(|| invalid("release: missing owner"))?;
            let token = params
                .get("fencing_token")
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid("release: missing fencing_token"))?;
            let ok = server
                .store
                .release(resource, owner, token, crate::now_ms())
                .map_err(|e| invalid(format!("release: {e}")))?;
            if !ok {
                return Err(not_found(format!(
                    "release refused for {resource} (wrong owner/token or expired)"
                )));
            }
            Ok(json!({"resource": resource, "released": true}))
        }
        "claims" => {
            let prefix = str_param(params, "prefix");
            let owner = str_param(params, "owner").map(normalize_address);
            let rows = server
                .store
                .claims(prefix, owner.as_deref(), crate::now_ms())
                .map_err(|e| invalid(format!("claims: {e}")))?;
            Ok(json!({"claims": rows}))
        }
        "check" => {
            let resource =
                str_param(params, "resource").ok_or_else(|| invalid("check: missing resource"))?;
            let holder = server
                .store
                .check(resource, crate::now_ms())
                .map_err(|e| invalid(format!("check: {e}")))?;
            Ok(json!({"resource": resource, "held": holder.is_some(), "holder": holder}))
        }
        _ => Err(RpcError {
            code: -32601,
            message: format!("method not found: {method}"),
        }),
    }
}

/// Start the hub as a detached child (auto-start helper for clients; same
/// pattern as the ext-host shared daemon in `crates/omp-ext/src/host.rs`):
/// under a spawn lock, drop a stale socket, spawn, then wait for the
/// socket to answer.
///
/// # Errors
/// Directory creation or daemon spawn failures.
pub fn spawn_detached(hub: &HubDir, exe: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&hub.dir)?;
    let mut lock_path = hub.sock.as_os_str().to_owned();
    lock_path.push(".spawn.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    for _ in 0..300 {
        if lock_file.try_lock().is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if hub.sock.exists() {
        std::fs::remove_file(&hub.sock).ok();
    }
    let log_out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(hub.dir.join("hub.log"))?;
    let log_err = log_out.try_clone()?;
    let child = std::process::Command::new(exe)
        .args(["hub", "serve", "--dir"])
        .arg(&hub.dir)
        .arg("--sock")
        .arg(&hub.sock)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_out))
        .stderr(Stdio::from(log_err))
        .spawn()
        .with_context(|| format!("spawn {} hub serve", exe.display()))?;
    // Keep the child handle alive so the client does not reap it.
    SPAWNED_CHILDREN.lock().push(child);
    Ok(())
}

static SPAWNED_CHILDREN: Mutex<Vec<std::process::Child>> = Mutex::new(Vec::new());

/// Wait (up to `timeout`) for the hub socket to accept a connection.
pub async fn wait_for_socket(sock: &Path, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if UnixStream::connect(sock).await.is_ok() {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
