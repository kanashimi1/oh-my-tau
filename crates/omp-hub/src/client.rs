//! Client library (`SPEC.md` item 7): what `omt hub` and the task tool
//! call. `connect` auto-starts the hub (detached `omt hub serve` when the
//! socket does not answer — same pattern as the ext-host shared daemon);
//! `hello` + the `deliver` stream is the subscribe surface.

use parking_lot::{Mutex, MutexGuard};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};

use crate::envelope::Envelope;
use crate::server::{self, HubDir};
use crate::store::ClaimOutcome;

/// Default bound for one RPC round trip.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// How long `connect` waits for a freshly spawned hub to answer.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);

/// Result of `send`: idempotent on `message_id`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SendReceipt {
    pub message_id: String,
    pub seq: u64,
    pub state: String,
    pub dedup: bool,
    /// Set when delivery failed fast (e.g. dead pane).
    pub error: Option<String>,
}

/// Stream of `deliver` notifications after `hello`.
pub type DeliverStream = mpsc::UnboundedReceiver<Envelope>;

struct PendingTable {
    waiters: HashMap<u64, oneshot::Sender<std::result::Result<Value, String>>>,
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
    m.lock()
}

/// One hub connection. Cheap to drop; the reader/writer tasks end with it.
pub struct HubClient {
    out: mpsc::UnboundedSender<Value>,
    pending: Pending,
    deliver_rx: Option<DeliverStream>,
    next_id: AtomicU64,
    pub hub: HubDir,
}

impl HubClient {
    /// Connect to the default (or `OMP_HUB_*`) hub, auto-starting a
    /// detached `omt hub serve` when the socket does not answer.
    ///
    /// # Errors
    /// When the socket is unreachable and the spawn/startup fails.
    pub async fn connect() -> Result<Self> {
        Self::connect_with(HubDir::resolve(), true).await
    }

    /// Connect to an explicit hub directory/socket.
    ///
    /// # Errors
    /// Connection or auto-start failures.
    pub async fn connect_with(hub: HubDir, auto_start: bool) -> Result<Self> {
        match UnixStream::connect(&hub.sock).await {
            Ok(stream) => return Ok(Self::on_stream(hub, stream)),
            Err(e) if !auto_start => {
                return Err(e).with_context(|| format!("connect {}", hub.sock.display()));
            }
            Err(_) => {}
        }
        let exe = std::env::current_exe().context("current_exe for hub auto-start")?;
        server::spawn_detached(&hub, &exe)?;
        if !server::wait_for_socket(&hub.sock, STARTUP_TIMEOUT).await {
            anyhow::bail!(
                "hub did not start on {} (see {})",
                hub.sock.display(),
                hub.dir.join("hub.log").display()
            );
        }
        let stream = UnixStream::connect(&hub.sock)
            .await
            .with_context(|| format!("connect {}", hub.sock.display()))?;
        Ok(Self::on_stream(hub, stream))
    }

    fn on_stream(hub: HubDir, stream: UnixStream) -> Self {
        let (read, mut write) = stream.into_split();
        let (out, mut out_rx) = mpsc::unbounded_channel::<Value>();
        let pending: Pending = Arc::new(Mutex::new(PendingTable {
            waiters: HashMap::new(),
            closed: None,
        }));
        let (deliver_tx, deliver_rx) = mpsc::unbounded_channel::<Envelope>();

        tokio::spawn({
            let pending = pending.clone();
            async move {
                while let Some(v) = out_rx.recv().await {
                    if write.write_all(v.to_string().as_bytes()).await.is_err()
                        || write.write_all(b"\n").await.is_err()
                        || write.flush().await.is_err()
                    {
                        lock(&pending).fail_all("connection closed");
                        return;
                    }
                }
            }
        });

        tokio::spawn({
            let pending = pending.clone();
            async move {
                let mut reader = BufReader::new(read);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) | Err(_) => {
                            lock(&pending).fail_all("connection closed");
                            return;
                        }
                        Ok(_) => {}
                    }
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                        let w = lock(&pending).waiters.remove(&id);
                        if let Some(w) = w {
                            let result = match msg.get("error") {
                                Some(e) => Err(e
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("hub error")
                                    .to_owned()),
                                None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
                            };
                            let _ = w.send(result);
                        }
                    } else if msg.get("method").and_then(Value::as_str) == Some("deliver") {
                        if let Some(env) = msg
                            .get("params")
                            .and_then(|p| p.get("envelope"))
                            .and_then(|e| serde_json::from_value::<Envelope>(e.clone()).ok())
                        {
                            let _ = deliver_tx.send(env);
                        }
                    }
                }
            }
        });

        Self {
            out,
            pending,
            deliver_rx: Some(deliver_rx),
            next_id: AtomicU64::new(1),
            hub,
        }
    }

    /// Take the `deliver` notification stream (call before `hello`).
    pub fn take_deliver_stream(&mut self) -> Option<DeliverStream> {
        self.deliver_rx.take()
    }

    /// Raw JSON-RPC request with timeout.
    ///
    /// # Errors
    /// Hub-side errors (message string) and transport failures.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            let mut p = lock(&self.pending);
            if let Some(why) = p.closed {
                anyhow::bail!("hub connection closed: {why}");
            }
            p.waiters.insert(id, tx);
        }
        let req = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.out
            .send(req)
            .map_err(|e| anyhow::anyhow!("hub connection closed: {e}"))?;
        match tokio::time::timeout(RPC_TIMEOUT, rx).await {
            Ok(Ok(Ok(v))) => Ok(v),
            Ok(Ok(Err(message))) => anyhow::bail!("hub error: {message}"),
            Ok(Err(_)) => anyhow::bail!("hub connection closed"),
            Err(_) => {
                lock(&self.pending).waiters.remove(&id);
                anyhow::bail!("hub request {method} timed out after {RPC_TIMEOUT:?}")
            }
        }
    }

    /// Register a live endpoint and replay undelivered/unacked mail from
    /// `last_seq`; the returned stream (see [`Self::take_deliver_stream`])
    /// carries `deliver` pushes, urgent first.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn hello(&self, agent: &str, last_seq: u64) -> Result<Value> {
        self.request("hello", json!({"agent": agent, "last_seq": last_seq}))
            .await
    }

    /// Send one envelope; idempotent on its `message_id`.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn send(&self, mut envelope: Envelope) -> Result<SendReceipt> {
        if envelope.message_id.is_empty() {
            envelope.message_id = crate::envelope::new_message_id();
        }
        let id = envelope.message_id.clone();
        let v = self
            .request("send", json!({"envelope": envelope}))
            .await
            .with_context(|| format!("send {id}"))?;
        Ok(SendReceipt {
            message_id: v
                .get("message_id")
                .and_then(Value::as_str)
                .unwrap_or(&id)
                .to_string(),
            seq: v.get("seq").and_then(Value::as_u64).unwrap_or(0),
            state: v
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("sent")
                .to_string(),
            dedup: v.get("dedup").and_then(Value::as_bool).unwrap_or(false),
            error: v
                .get("error")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
        })
    }

    /// Acknowledge delivery or reading.
    ///
    /// # Errors
    /// Hub-side errors (unknown message, bad state, transport).
    pub async fn ack(&self, message_id: &str, state: &str) -> Result<()> {
        self.request("ack", json!({"message_id": message_id, "state": state}))
            .await?;
        Ok(())
    }

    /// Answer a message (validated against its `reply_schema` by the hub).
    ///
    /// # Errors
    /// Hub-side errors: unknown question, validation failure (message
    /// carries the schema errors), transport.
    pub async fn answer(&self, causation_id: &str, envelope: Envelope) -> Result<Value> {
        self.request(
            "answer",
            json!({"causation_id": causation_id, "envelope": envelope}),
        )
        .await
    }

    /// Pull an inbox (at-least-once; dedup client-side on `message_id`).
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn inbox(
        &self,
        recipient: &str,
        since_seq: Option<u64>,
        limit: Option<u64>,
    ) -> Result<Vec<Envelope>> {
        let v = self
            .request(
                "inbox",
                json!({
                    "recipient": recipient,
                    "since_seq": since_seq.unwrap_or(0),
                    "limit": limit.unwrap_or(50),
                }),
            )
            .await?;
        Ok(v.get("envelopes")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|e| serde_json::from_value::<Envelope>(e.clone()).ok())
                    .collect()
            })
            .unwrap_or_default())
    }

    /// One envelope by id.
    ///
    /// # Errors
    /// Hub-side errors (unknown message, transport).
    pub async fn status_message(&self, message_id: &str) -> Result<Value> {
        self.request("status", json!({"message_id": message_id}))
            .await
    }

    /// Per-recipient or global status aggregation.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn status(&self, recipient: Option<&str>) -> Result<Value> {
        match recipient {
            Some(r) => self.request("status", json!({"recipient": r})).await,
            None => self.request("status", json!({})).await,
        }
    }

    /// Claim a resource with a ttl lease.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn claim(
        &self,
        resource: &str,
        owner: &str,
        ttl_s: u64,
        intent: Option<&str>,
    ) -> Result<ClaimOutcome> {
        let v = self
            .request(
                "claim",
                json!({"resource": resource, "owner": owner, "ttl_s": ttl_s, "intent": intent}),
            )
            .await?;
        serde_json::from_value(v).context("claim outcome")
    }

    /// Release a claim; requires the fencing token.
    ///
    /// # Errors
    /// Hub-side errors: refused release (wrong owner/token or expired),
    /// transport.
    pub async fn release(&self, resource: &str, owner: &str, fencing_token: u64) -> Result<()> {
        self.request(
            "release",
            json!({"resource": resource, "owner": owner, "fencing_token": fencing_token}),
        )
        .await?;
        Ok(())
    }

    /// Active claims, optionally by resource prefix or owner.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn claims(&self, prefix: Option<&str>, owner: Option<&str>) -> Result<Value> {
        self.request("claims", json!({"prefix": prefix, "owner": owner}))
            .await
    }

    /// Who holds a resource right now.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn check(&self, resource: &str) -> Result<Value> {
        self.request("check", json!({"resource": resource})).await
    }

    /// Typed variant of [`Self::check`]: the live claim row, `None` when the
    /// resource is free or the lease expired (`expiry frees it`).
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn check_claim(&self, resource: &str) -> Result<Option<crate::store::ClaimRow>> {
        let v = self.check(resource).await?;
        Ok(v.get("holder")
            .and_then(|h| serde_json::from_value::<crate::store::ClaimRow>(h.clone()).ok()))
    }

    /// Send one plain-text envelope (`payload` inline), minting the
    /// `message_id` when empty — the convenience surface for notes/questions
    /// a session routes externally.
    ///
    /// # Errors
    /// Hub-side errors (transport or handler failure).
    pub async fn send_note(
        &self,
        sender: &str,
        recipient: &str,
        kind: &str,
        payload: &str,
    ) -> Result<SendReceipt> {
        self.send(Envelope {
            sender: sender.to_string(),
            recipient: recipient.to_string(),
            kind: kind.to_string(),
            payload: Some(payload.to_string()),
            ..Envelope::default()
        })
        .await
    }

    /// Answer a question with a plain-text payload envelope (kind
    /// `agent.answer`, urgent priority), validated against the original's
    /// `reply_schema` by the hub; on violation the returned error carries
    /// the schema failure so the caller can surface it verbatim.
    ///
    /// # Errors
    /// Hub-side errors: unknown/expired question, schema validation failure.
    pub async fn answer_text(
        &self,
        causation_id: &str,
        sender: &str,
        payload: &str,
    ) -> Result<Value> {
        self.answer(
            causation_id,
            Envelope {
                sender: sender.to_string(),
                kind: crate::envelope::KIND_ANSWER.to_string(),
                priority: crate::envelope::Priority::URGENT,
                payload: Some(payload.to_string()),
                ..Envelope::default()
            },
        )
        .await
    }
}

/// Path override for tests and non-default installs.
#[must_use]
pub fn hub_dir_from(dir: PathBuf) -> HubDir {
    HubDir::explicit(dir, None)
}
