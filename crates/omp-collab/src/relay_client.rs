//! Client-side WebSocket wrapper for collab live-session sharing.
//!
//! Rust port of `packages/coding-agent/src/collab/relay-client.ts`, at the transport level the
//! port contract fixes: [`connect_relay`] owns the room URL (`…/r/<roomId>?role=…`), the
//! reconnect policy (exponential backoff with ±25% jitter), relay close-code classification
//! ([`RelayEvent::Closed`] carries `will_reconnect`), the ping/silence heartbeat, the host
//! duplicate-connection reclaim window, and the queued-send backlog that survives reconnects.
//!
//! Frame sealing stays with the callers (the `crate::protocol` envelope plus the cipher live
//! above this layer), so [`RelayHandle`] moves opaque payloads; the only envelope knowledge here
//! is the 4-byte target-peer header, needed to drop targeted work across a room recreation
//! exactly like the TS send queue does (relay-client.ts:268).

use std::{
    collections::VecDeque,
    pin::Pin,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use futures::{SinkExt, Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async, tungstenite::protocol::CloseFrame, tungstenite::Message, MaybeTlsStream,
    WebSocketStream,
};

use crate::protocol::unpack_envelope;

/// Initial reconnect backoff (relay-client.ts:25).
pub const BACKOFF_BASE_MS: u64 = 1_000;
/// Reconnect backoff cap (relay-client.ts:26).
pub const BACKOFF_MAX_MS: u64 = 30_000;
/// How long after a transient drop a host keeps retrying the relay's duplicate-host close (4009):
/// the client can see its connection end before the relay does, and the relay refuses the
/// reconnect until its liveness check retires the old socket (relay-client.ts:35).
pub const HOST_RECLAIM_WINDOW_MS: u64 = 150_000;
/// Backoff cap for reclaim retries: each is a single upgrade the relay refuses at once
/// (relay-client.ts:41).
pub const HOST_RECLAIM_BACKOFF_MAX_MS: u64 = 5_000;
/// How long a reclaim-time open stays provisional before it is committed without the relay's
/// verdict (relay-client.ts:50).
pub const HOST_RECLAIM_CONFIRM_MS: u64 = 1_000;
/// How often an open relay socket is pinged (relay-client.ts:57).
pub const RELAY_PING_INTERVAL_MS: u64 = 15_000;
/// A socket that has heard nothing, not even a pong, for this long is dropped and retried
/// (relay-client.ts:59).
pub const RELAY_SILENCE_LIMIT_MS: u64 = 40_000;
/// Queued-send bound mirroring the TS seal queue (relay-client.ts:60).
pub const MAX_PENDING_SENDS: usize = 256;
/// Queued-send byte bound mirroring the TS seal queue (relay-client.ts:61).
pub const MAX_PENDING_SEND_BYTES: usize = 16 * 1024 * 1024;

/// Fixed relay codes whose meaning never reconnects (relay-client.ts:18).
fn relay_close_reason(code: u16) -> Option<&'static str> {
    match code {
        4001 => Some("room closed"),
        4004 => Some("no such room"),
        4009 => Some("a host is already connected for this room"),
        4029 => Some("room is full"),
        _ => None,
    }
}

/// The reason a [`RelayEvent::Closed`] reports: the fixed relay wording when the code is one of
/// the known ones, else the wire reason, else a synthesized lost-connection string
/// (relay-client.ts:719).
fn reported_close_reason(code: u16, wire_reason: &str) -> String {
    match relay_close_reason(code) {
        Some(reason) => reason.to_string(),
        None if wire_reason.is_empty() => format!("connection lost (code {code})"),
        None => wire_reason.to_string(),
    }
}

/// What [`connect_relay`]'s close handling does with a close code (relay-client.ts:707
/// `#handleClose`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloseKind {
    /// 4009 while the host's reclaim window is open: the relay may still hold this host's own
    /// dropped socket; retry fast instead of failing.
    HostReclaim,
    /// Guest retry after its room closed (4001) or a mid-recreation 4004: keep retrying until the
    /// host recreates the room.
    GuestRoomRetry,
    /// Fixed relay code with no reconnect.
    Fatal,
    /// Anything else: transient drop.
    Transient,
}

fn classify_close(
    code: u16,
    role: CollabRole,
    retry_missing_room: bool,
    reclaim_window_open: bool,
) -> CloseKind {
    if code == 4009 && role == CollabRole::Host && reclaim_window_open {
        return CloseKind::HostReclaim;
    }
    if role == CollabRole::Guest && (code == 4001 || (code == 4004 && retry_missing_room)) {
        return CloseKind::GuestRoomRetry;
    }
    if relay_close_reason(code).is_some() {
        return CloseKind::Fatal;
    }
    CloseKind::Transient
}

/// `min(base·2^attempt, max)·(0.75 + jitter·0.5)` — the TS backoff with its ±25% jitter
/// (relay-client.ts:765). `jitter01` must be in `[0, 1)`.
// The arithmetic is deliberately the TS f64 expression (`2 ** attempt` overflows no realistic
// attempt and the multiply keeps sub-ms jitter), so the casts mirror JS numerics.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn backoff_delay_ms(attempt: u32, base_ms: u64, max_ms: u64, jitter01: f64) -> u64 {
    let base = (base_ms as f64 * 2.0f64.powi(attempt.min(1_023) as i32)).min(max_ms as f64);
    (base * (0.75 + jitter01 * 0.5)) as u64
}

/// Which side of the relay room this socket joins (the `?role=` query value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollabRole {
    /// Creates the room; a second host is closed with 4009.
    Host,
    /// Joins an existing room; a missing room is closed with 4004.
    Guest,
}

impl CollabRole {
    fn as_str(self) -> &'static str {
        match self {
            CollabRole::Host => "host",
            CollabRole::Guest => "guest",
        }
    }
}

/// Reconnect and heartbeat timing, mirroring the relay-client.ts constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayOptions {
    /// Initial reconnect backoff.
    pub backoff_base_ms: u64,
    /// Reconnect backoff cap.
    pub backoff_max_ms: u64,
    /// How long after a transient drop a host treats 4009 as its own dropped socket.
    pub host_reclaim_window_ms: u64,
    /// Backoff cap for reclaim retries.
    pub host_reclaim_backoff_max_ms: u64,
    /// How long a reclaim-time open stays provisional without the relay's verdict.
    pub host_reclaim_confirm_ms: u64,
    /// How often an open socket is pinged.
    pub ping_interval_ms: u64,
    /// A socket silent this long is dropped and retried.
    pub silence_limit_ms: u64,
}

impl Default for RelayOptions {
    fn default() -> Self {
        Self {
            backoff_base_ms: BACKOFF_BASE_MS,
            backoff_max_ms: BACKOFF_MAX_MS,
            host_reclaim_window_ms: HOST_RECLAIM_WINDOW_MS,
            host_reclaim_backoff_max_ms: HOST_RECLAIM_BACKOFF_MAX_MS,
            host_reclaim_confirm_ms: HOST_RECLAIM_CONFIRM_MS,
            ping_interval_ms: RELAY_PING_INTERVAL_MS,
            silence_limit_ms: RELAY_SILENCE_LIMIT_MS,
        }
    }
}

/// One event from a relay session. `Opened` fires after every successful (re)connect — the first
/// one joins the room, and any later one follows a `Closed { will_reconnect: true }`, i.e. the
/// relay recreated the room and reissued peer ids from 1 (the TS `onRoomRecreated` folds into
/// it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayEvent {
    /// The socket is committed: writable and reported open.
    Opened,
    /// A binary frame from the relay: an envelope `[4B BE peer id][sealed payload]`.
    Binary(Vec<u8>),
    /// A relay control frame (`peer-joined` / `peer-left` / `room-closed` JSON), unencrypted.
    Text(String),
    /// The socket ended. `will_reconnect` distinguishes retries from terminal shutdown.
    Closed {
        code: u16,
        reason: String,
        will_reconnect: bool,
    },
}

/// One queued outbound frame waiting for a writable socket.
struct QueuedSend {
    data: QueuedData,
    bytes: usize,
}

enum QueuedData {
    Binary(Vec<u8>),
    Text(String),
}

impl QueuedData {
    fn message(&self) -> Message {
        match self {
            QueuedData::Binary(data) => Message::binary(data.clone()),
            QueuedData::Text(text) => Message::text(text.clone()),
        }
    }
}

enum Command {
    Binary(Vec<u8>),
    Text(String),
    Close,
}

/// Clonable handle to a live relay session. Sends queue until a socket accepts them and survive
/// reconnects (broadcast work; targeted work is dropped across a room recreation, whose peer ids
/// are reissued). Sends are no-ops once the session ended, like the TS socket's.
#[derive(Debug, Clone)]
pub struct RelayHandle {
    tx: mpsc::UnboundedSender<Command>,
}

impl RelayHandle {
    /// Queue one binary frame (an envelope: `[4B BE peer id][sealed payload]`). Returns `false`
    /// — dropping the frame — once the session is closed, like the TS no-op on a closed socket.
    #[must_use = "a dropped frame is silently lost; decide what that means for the caller"]
    pub fn send_binary(&self, data: Vec<u8>) -> bool {
        self.tx.send(Command::Binary(data)).is_ok()
    }

    /// Queue one text frame. Returns `false` once the session is closed.
    #[must_use = "a dropped frame is silently lost; decide what that means for the caller"]
    pub fn send_text(&self, text: impl Into<String>) -> bool {
        self.tx.send(Command::Text(text.into())).is_ok()
    }

    /// Intentional close: queued frames still flush to the live socket ahead of the close frame,
    /// reconnect is suppressed, and one final
    /// `Closed { code: 1000, reason: "closed", will_reconnect: false }` is emitted. Idempotent.
    pub fn close(&self) {
        let _ = self.tx.send(Command::Close);
    }

    /// Resolves once the session task has ended (and so flushed everything queued ahead of an
    /// intentional [`close`](Self::close)). Callers that own the runtime the session runs on must
    /// await this before dropping it, or the queued goodbye is cancelled with the task
    /// (TS `socket.flush()`, host.ts:554).
    pub async fn closed(&self) {
        self.tx.closed().await;
    }
}

/// [`Stream`] over a session's events. A thin wrapper around the session's mpsc receiver
/// (tokio's own receivers grow a `Stream` impl only via the separate tokio-stream crate).
struct EventStream {
    rx: mpsc::UnboundedReceiver<RelayEvent>,
}

impl futures::Stream for EventStream {
    type Item = RelayEvent;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<RelayEvent>> {
        self.rx.poll_recv(cx)
    }
}

/// Connect to a relay room and run the reconnect loop on a background task.
///
/// `url` is the relay origin (`ws://host[:port]` or `wss://host`); the room path and the role
/// query are appended here, matching the TS `wsUrl` construction (protocol.ts:338,
/// relay-client.ts:506). Returns immediately; the caller drives the event stream and sees
/// [`RelayEvent::Opened`] once the relay accepted the join.
///
/// # Errors
/// When `url` carries no `ws:`/`wss:` scheme, or no tokio runtime is running to host the session
/// task.
// The contract fixes this signature as `async fn`; the body only validates and spawns.
#[allow(clippy::unused_async)]
pub async fn connect_relay(
    url: &str,
    room_id: &str,
    role: CollabRole,
    opts: &RelayOptions,
) -> Result<(RelayHandle, impl Stream<Item = RelayEvent>)> {
    let origin = url.trim_end_matches('/');
    anyhow::ensure!(!origin.is_empty(), "relay url is empty");
    anyhow::ensure!(
        origin.starts_with("ws://") || origin.starts_with("wss://"),
        "relay url must use the ws: or wss: scheme: {url}"
    );
    let ws_url = format!("{origin}/r/{room_id}?role={}", role.as_str());
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let session = Session {
        events: event_tx,
        cmd_rx,
        ws_url,
        role,
        opts: *opts,
        rng: initial_rng_state(),
        attempt: 0,
        reclaim_attempt: 0,
        retry_missing_room: false,
        rejoining: false,
        host_reclaim_until: None,
        queued: VecDeque::new(),
        queued_bytes: 0,
    };
    tokio::spawn(run_session(session));
    Ok((RelayHandle { tx: cmd_tx }, EventStream { rx: event_rx }))
}

/// Splitmix64 state, seeded from the clock, the pid and a process-wide counter so sessions born
/// in the same tick still de-correlate their jitter.
fn initial_rng_state() -> u64 {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::from(d.subsec_nanos()) ^ d.as_secs());
    let sequence = SEQUENCE.fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed);
    nanos.rotate_left(32) ^ u64::from(std::process::id()) ^ sequence
}

fn next_jitter01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 53 random mantissa bits: exactly a uniform f64 in [0, 1).
    #[allow(clippy::cast_precision_loss)]
    {
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How the live socket loop ended.
enum SocketEnd {
    /// Owner closed (or abandoned) the session, or a fatal failure: already reported.
    Terminal,
    /// The wire closed. `code` 1006 with an empty reason stands in for a failed upgrade or an
    /// aborted connection, like Bun's close event; the heartbeat's silent drop reports its own
    /// reason (relay-client.ts:581).
    Dropped { code: u16, reason: String },
}

/// Session state for the reconnect loop.
struct Session {
    events: mpsc::UnboundedSender<RelayEvent>,
    cmd_rx: mpsc::UnboundedReceiver<Command>,
    ws_url: String,
    role: CollabRole,
    opts: RelayOptions,
    rng: u64,
    /// Retries since the last confirmed open; reset on open unless a guest is surviving a room
    /// recreation, where the TS never resets it (relay-client.ts:595).
    attempt: u32,
    /// Reclaim retries since the last transient drop (relay-client.ts:122).
    reclaim_attempt: u32,
    /// Set once a guest starts retrying after a room close, making a following 4004 retryable
    /// (relay-client.ts:112).
    retry_missing_room: bool,
    /// Set while a drop is being retried; the next open lands in a recreated room
    /// (relay-client.ts:114).
    rejoining: bool,
    /// Until when a host takes 4009 to be its own dropped connection (relay-client.ts:120).
    host_reclaim_until: Option<Instant>,
    /// Outbound frames waiting for a writable socket; the reconnect backlog. Across a room
    /// recreation every *targeted* envelope must be dropped, since the relay reissues peer ids
    /// (see [`Session::commit_open`]).
    queued: VecDeque<QueuedSend>,
    queued_bytes: usize,
}

impl Session {
    fn next_jitter01(&mut self) -> f64 {
        next_jitter01(&mut self.rng)
    }

    fn enqueue(&mut self, command: Command) -> Result<(), String> {
        let (data, bytes) = match command {
            Command::Binary(data) => {
                let bytes = data.len();
                (QueuedData::Binary(data), bytes)
            }
            Command::Text(text) => {
                let bytes = text.len();
                (QueuedData::Text(text), bytes)
            }
            Command::Close => return Ok(()), // handled by the loops; never queued
        };
        if self.queued.len() >= MAX_PENDING_SENDS
            || self.queued_bytes + bytes > MAX_PENDING_SEND_BYTES
        {
            return Err(self.overload_reason());
        }
        self.queued_bytes += bytes;
        self.queued.push_back(QueuedSend { data, bytes });
        Ok(())
    }

    fn dequeue(&mut self) {
        if let Some(sent) = self.queued.pop_front() {
            self.queued_bytes -= sent.bytes;
        }
    }

    fn overload_reason(&self) -> String {
        let recovery = if self.role == CollabRole::Host {
            "restart sharing and rejoin"
        } else {
            "rejoin"
        };
        format!(
            "collab send backlog exceeded its limit; {recovery} to resync and check whether \
             pending commands ran before retrying"
        )
    }

    /// Terminal send failure (relay-client.ts:317-321): the socket is torn down and nothing
    /// reconnects. `code` 0 — no wire close was involved.
    fn fail_send(&mut self, err: &tokio_tungstenite::tungstenite::Error) {
        let _ = self.events.send(RelayEvent::Closed {
            code: 0,
            reason: format!("collab send failed: {err}; rejoin to resync"),
            will_reconnect: false,
        });
    }

    /// Terminal overload failure (relay-client.ts:306 `#failOverload`): the socket is torn down
    /// and nothing reconnects. `code` 0 — no wire close was involved.
    fn fail_overload(&mut self) {
        let _ = self.events.send(RelayEvent::Closed {
            code: 0,
            reason: self.overload_reason(),
            will_reconnect: false,
        });
    }

    /// Room recreation: targeted work is undeliverable and must go — the relay reissues peer ids
    /// from 1, so an old id may already name somebody else (relay-client.ts:268
    /// `#resetForRecreatedRoom`). Broadcast work survives, which is the reconnect backlog
    /// contract.
    fn discard_targeted(&mut self) {
        self.queued.retain(|send| match &send.data {
            QueuedData::Text(_) => true,
            QueuedData::Binary(data) => {
                unpack_envelope(data).is_none_or(|envelope| envelope.peer_id == 0)
            }
        });
        self.queued_bytes = self.queued.iter().map(|send| send.bytes).sum();
    }

    /// A sealed frame arrived on the live socket and decoded cleanly: this is a real room again.
    /// A guest's room-recreation retry mode ends here, and the backoff restarts
    /// (relay-client.ts:659-662).
    fn on_live_frame(&mut self) {
        self.retry_missing_room = false;
        self.attempt = 0;
    }

    /// The relay accepted this socket (relay-client.ts:592 `#commitOpen`): writable, open, and —
    /// after a retry — facing a recreated room.
    fn commit_open(&mut self) {
        if !self.retry_missing_room {
            self.attempt = 0;
        }
        if self.rejoining {
            self.discard_targeted();
        }
        self.rejoining = false;
        let _ = self.events.send(RelayEvent::Opened);
    }

    /// Apply the close classification (relay-client.ts:707 `#handleClose`). `None` means
    /// terminal — the session ends here.
    fn handle_close(&mut self, code: u16, wire_reason: &str) -> Option<Duration> {
        let reclaim_window_open = self
            .host_reclaim_until
            .is_some_and(|until| Instant::now() < until);
        match classify_close(
            code,
            self.role,
            self.retry_missing_room,
            reclaim_window_open,
        ) {
            CloseKind::HostReclaim => {
                self.rejoining = true;
                let _ = self.events.send(RelayEvent::Closed {
                    code,
                    reason: "the relay still holds this host's previous connection".to_string(),
                    will_reconnect: true,
                });
                let delay = backoff_delay_ms(
                    self.reclaim_attempt,
                    self.opts.backoff_base_ms,
                    self.opts.host_reclaim_backoff_max_ms,
                    self.next_jitter01(),
                );
                self.reclaim_attempt += 1;
                Some(Duration::from_millis(delay))
            }
            CloseKind::GuestRoomRetry => {
                self.retry_missing_room = true;
                self.rejoining = true;
                let _ = self.events.send(RelayEvent::Closed {
                    code,
                    reason: reported_close_reason(code, wire_reason),
                    will_reconnect: true,
                });
                let delay = backoff_delay_ms(
                    self.attempt,
                    self.opts.backoff_base_ms,
                    self.opts.backoff_max_ms,
                    self.next_jitter01(),
                );
                self.attempt += 1;
                Some(Duration::from_millis(delay))
            }
            CloseKind::Fatal => {
                self.queued.clear();
                self.queued_bytes = 0;
                let _ = self.events.send(RelayEvent::Closed {
                    code,
                    reason: reported_close_reason(code, wire_reason),
                    will_reconnect: false,
                });
                None
            }
            CloseKind::Transient => {
                self.rejoining = true;
                if self.role == CollabRole::Host {
                    self.host_reclaim_until = Some(
                        Instant::now() + Duration::from_millis(self.opts.host_reclaim_window_ms),
                    );
                    self.reclaim_attempt = 0;
                }
                let _ = self.events.send(RelayEvent::Closed {
                    code,
                    reason: reported_close_reason(code, wire_reason),
                    will_reconnect: true,
                });
                let delay = backoff_delay_ms(
                    self.attempt,
                    self.opts.backoff_base_ms,
                    self.opts.backoff_max_ms,
                    self.next_jitter01(),
                );
                self.attempt += 1;
                Some(Duration::from_millis(delay))
            }
        }
    }
}

fn close_frame(code: u16, reason: &str) -> Message {
    Message::Close(Some(CloseFrame {
        code: code.into(),
        reason: reason.into(),
    }))
}

async fn run_session(mut session: Session) {
    loop {
        let Ok((ws, _)) = connect_async(session.ws_url.as_str()).await else {
            // A failed upgrade surfaces as close 1006 with an empty reason, like Bun's.
            if !retry_after_close(&mut session, 1006, "").await {
                return;
            }
            continue;
        };
        // After a transient drop a host's open is provisional: the relay may still
        // refuse it as a duplicate (relay-client.ts:516).
        let provisional = session.rejoining
            && session.role == CollabRole::Host
            && session
                .host_reclaim_until
                .is_some_and(|until| Instant::now() < until);
        let SocketEnd::Dropped { code, reason } = run_socket(&mut session, ws, provisional).await
        else {
            return; // Terminal: owner closed or a fatal failure — already reported.
        };
        if !retry_after_close(&mut session, code, &reason).await {
            return;
        }
    }
}

/// Classify a wire close and, when it is retryable, wait out the backoff. `false` ends the
/// session.
async fn retry_after_close(session: &mut Session, code: u16, reason: &str) -> bool {
    let Some(delay) = session.handle_close(code, reason) else {
        return false;
    };
    sleep_between_retries(session, delay).await
}

/// Wait out a retry delay. A close command (or an abandoned session) short-circuits it and ends
/// the session with the same `closed` report an open socket would produce
/// (relay-client.ts:474 `close()` while a retry timer runs). Returns `false` to stop.
async fn sleep_between_retries(session: &mut Session, delay: Duration) -> bool {
    tokio::select! {
        () = tokio::time::sleep(delay) => true,
        command = session.cmd_rx.recv() => match command {
            Some(Command::Close) | None => {
                let _ = session.events.send(RelayEvent::Closed {
                    code: 1000,
                    reason: "closed".to_string(),
                    will_reconnect: false,
                });
                false
            }
            Some(command) => {
                match session.enqueue(command) {
                    Ok(()) => true,
                    Err(reason) => {
                        let _ = session.events.send(RelayEvent::Closed {
                            code: 0,
                            reason,
                            will_reconnect: false,
                        });
                        false
                    }
                }
            }
        },
    }
}

/// Drive one live socket until it ends: commit (immediately, or after the reclaim confirm), pump
/// the queued backlog, interleave reads with writes, and ping against silence.
// One select loop mirrors the TS onopen/onmessage/onclose/ping handlers; splitting it would
// hide the interleaving the protocol depends on.
#[allow(clippy::too_many_lines)]
async fn run_socket(
    session: &mut Session,
    ws: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    provisional: bool,
) -> SocketEnd {
    let (mut sink, mut stream) = ws.split();
    // A reclaim-time open stays provisional until the confirm elapses or the relay speaks
    // (relay-client.ts:519): not writable, not reported open, backlog untouched.
    let mut committed = !provisional;
    if committed {
        session.commit_open();
    }
    let mut confirm: Pin<Box<tokio::time::Sleep>> = Box::pin(tokio::time::sleep(
        Duration::from_millis(session.opts.host_reclaim_confirm_ms),
    ));
    let mut ping = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_millis(session.opts.ping_interval_ms),
        Duration::from_millis(session.opts.ping_interval_ms),
    );
    let mut last_heard = Instant::now();

    loop {
        // Write exactly one queued frame per iteration while writable, so reads keep being
        // served between writes (the transport analogue of the TS bufferedAmount check).
        if committed {
            if let Some(send) = session.queued.front() {
                let message = send.data.message();
                match sink.send(message).await {
                    Ok(()) => session.dequeue(),
                    // A failed write is a dead socket; the TS treats it as fatal rather than
                    // transient (relay-client.ts:317-321).
                    Err(err) => {
                        session.fail_send(&err);
                        return SocketEnd::Terminal;
                    }
                }
                continue;
            }
        }
        tokio::select! {
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    last_heard = Instant::now();
                    if !committed {
                        committed = true;
                        session.commit_open();
                    }
                    let _ = session.events.send(RelayEvent::Text(text.to_string()));
                }
                Some(Ok(Message::Binary(data))) => {
                    last_heard = Instant::now();
                    if !committed {
                        committed = true;
                        session.commit_open();
                    }
                    // Reset before dispatch, like the TS recvChain (relay-client.ts:659-662).
                    session.on_live_frame();
                    let _ = session.events.send(RelayEvent::Binary(data.to_vec()));
                }
                // Pongs count as "heard" but do not commit a provisional open, exactly like the
                // TS pong listener (relay-client.ts:533).
                Some(Ok(Message::Pong(_) | Message::Ping(_))) => {
                    last_heard = Instant::now();
                }
                Some(Ok(Message::Close(frame))) => {
                    let (code, reason) = frame.map_or((1005, String::new()), |frame| {
                        (u16::from(frame.code), frame.reason.to_string())
                    });
                    return SocketEnd::Dropped { code, reason };
                }
                Some(Err(_)) | None => {
                    return SocketEnd::Dropped { code: 1006, reason: String::new() };
                }
                Some(Ok(Message::Frame(_))) => {}
            },
            command = session.cmd_rx.recv() => match command {
                Some(Command::Close) | None => {
                    // Closing is terminal, so backpressure no longer matters: everything queued
                    // enters the socket buffer ahead of the close frame
                    // (relay-client.ts:484-496). A provisional socket is not writable and its
                    // backlog stays untouched, like the TS queue.
                    if committed {
                        while let Some(send) = session.queued.pop_front() {
                            session.queued_bytes -= send.bytes;
                            if sink.send(send.data.message()).await.is_err() {
                                break;
                            }
                        }
                    }
                    let _ = sink.send(close_frame(1000, "closed")).await;
                    let _ = session.events.send(RelayEvent::Closed {
                        code: 1000,
                        reason: "closed".to_string(),
                        will_reconnect: false,
                    });
                    return SocketEnd::Terminal;
                }
                Some(command) => {
                    if session.enqueue(command).is_err() {
                        let _ = sink.send(close_frame(1000, "closed")).await;
                        session.fail_overload();
                        return SocketEnd::Terminal;
                    }
                }
            },
            _ = ping.tick(), if committed => {
                if last_heard.elapsed() > Duration::from_millis(session.opts.silence_limit_ms) {
                    // Nothing reveals a dead network until the next write; drop and retry here
                    // rather than wait for a close that may never come (relay-client.ts:555).
                    return SocketEnd::Dropped {
                        code: 1006,
                        reason: "relay stopped responding".to_string(),
                    };
                }
                // A failed ping write surfaces as the socket's own close event
                // (relay-client.ts:566-568); ignore it here.
                let _ = sink.send(Message::Ping(Vec::new().into())).await;
            }
            () = &mut confirm, if !committed => {
                committed = true;
                session.commit_open();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_close_reasons_match_ts_table() {
        assert_eq!(relay_close_reason(4001), Some("room closed"));
        assert_eq!(relay_close_reason(4004), Some("no such room"));
        assert_eq!(
            relay_close_reason(4009),
            Some("a host is already connected for this room")
        );
        assert_eq!(relay_close_reason(4029), Some("room is full"));
        assert_eq!(relay_close_reason(1006), None);
    }

    #[test]
    fn reported_reason_prefers_fixed_wording_then_wire_then_fallback() {
        assert_eq!(reported_close_reason(4004, "ignored"), "no such room");
        assert_eq!(reported_close_reason(1006, "boom"), "boom");
        assert_eq!(
            reported_close_reason(1006, ""),
            "connection lost (code 1006)"
        );
        assert_eq!(
            reported_close_reason(1001, "relay shutting down"),
            "relay shutting down"
        );
    }

    #[test]
    fn classification_first_connects() {
        // Fresh connects: only a guest whose room closed retries.
        assert_eq!(
            classify_close(1006, CollabRole::Host, false, false),
            CloseKind::Transient
        );
        assert_eq!(
            classify_close(1006, CollabRole::Guest, false, false),
            CloseKind::Transient
        );
        assert_eq!(
            classify_close(4001, CollabRole::Host, false, false),
            CloseKind::Fatal
        );
        assert_eq!(
            classify_close(4001, CollabRole::Guest, false, false),
            CloseKind::GuestRoomRetry
        );
        // A first 4004 is a real missing room, even for a guest.
        assert_eq!(
            classify_close(4004, CollabRole::Guest, false, false),
            CloseKind::Fatal
        );
        // …but after a room-close retry started, a 4004 is the recreation race.
        assert_eq!(
            classify_close(4004, CollabRole::Guest, true, false),
            CloseKind::GuestRoomRetry
        );
        // Second host / full room never reconnect, window or not.
        assert_eq!(
            classify_close(4009, CollabRole::Host, false, false),
            CloseKind::Fatal
        );
        assert_eq!(
            classify_close(4009, CollabRole::Guest, false, true),
            CloseKind::Fatal
        );
        assert_eq!(
            classify_close(4029, CollabRole::Host, false, true),
            CloseKind::Fatal
        );
    }

    #[test]
    fn classification_host_reclaim_window() {
        // 4009 inside the window is a reclaim retry; outside it a real second host.
        assert_eq!(
            classify_close(4009, CollabRole::Host, false, true),
            CloseKind::HostReclaim
        );
        assert_eq!(
            classify_close(4009, CollabRole::Host, false, false),
            CloseKind::Fatal
        );
        // Guests never reclaim.
        assert_eq!(
            classify_close(4009, CollabRole::Guest, true, true),
            CloseKind::Fatal
        );
    }

    #[test]
    fn backoff_bounds_and_caps() {
        for attempt in 0..12u32 {
            let low = backoff_delay_ms(attempt, 1_000, 30_000, 0.0);
            let high = backoff_delay_ms(attempt, 1_000, 30_000, 0.999_999);
            let base = (1_000_f64 * 2.0f64.powi(attempt as i32)).min(30_000.0);
            assert!(low as f64 >= base * 0.75 - 1.0, "attempt {attempt}: {low}");
            assert!(
                high as f64 <= base * 1.25 + 1.0,
                "attempt {attempt}: {high}"
            );
        }
        // The cap wins before jitter: 30_000·0.75 .. 30_000·1.25.
        assert_eq!(backoff_delay_ms(20, 1_000, 30_000, 0.0), 22_500);
        assert_eq!(backoff_delay_ms(20, 1_000, 30_000, 0.999_999), 37_499);
        // Reclaim retries cap much lower.
        assert_eq!(backoff_delay_ms(20, 1_000, 5_000, 0.999_999), 6_249);
    }

    fn test_session(role: CollabRole) -> (Session, mpsc::UnboundedReceiver<RelayEvent>) {
        let (_cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let session = Session {
            events: event_tx,
            cmd_rx,
            ws_url: "ws://127.0.0.1:1/r/roomtest12345?role=guest".to_string(),
            role,
            opts: RelayOptions::default(),
            rng: initial_rng_state(),
            attempt: 0,
            reclaim_attempt: 0,
            retry_missing_room: false,
            rejoining: false,
            host_reclaim_until: None,
            queued: VecDeque::new(),
            queued_bytes: 0,
        };
        (session, event_rx)
    }

    #[test]
    fn live_frame_ends_the_guests_room_retry_mode() {
        let (mut session, _events) = test_session(CollabRole::Guest);
        // Room closed: the guest starts retrying and tolerates a following 4004.
        assert!(session.handle_close(4001, "room closed").is_some());
        assert!(session.retry_missing_room);
        assert!(
            session.handle_close(4004, "").is_some(),
            "mid-recreation 4004 retries"
        );
        // One live frame later the room is real again: 4004 must be fatal.
        session.on_live_frame();
        assert!(!session.retry_missing_room);
        assert_eq!(session.attempt, 0);
        assert!(
            session.handle_close(4004, "").is_none(),
            "4004 after a live frame is fatal"
        );
    }

    #[test]
    fn live_frame_restarts_the_backoff_ladder() {
        let (mut session, _events) = test_session(CollabRole::Guest);
        for _ in 0..4 {
            assert!(session.handle_close(1006, "").is_some());
        }
        assert!(session.attempt > 0);
        session.on_live_frame();
        assert_eq!(
            session.attempt, 0,
            "a decoded frame restarts the backoff ladder"
        );
    }

    #[test]
    fn send_failure_is_fatal_not_transient() {
        let (mut session, mut events) = test_session(CollabRole::Host);
        session.fail_send(&tokio_tungstenite::tungstenite::Error::AlreadyClosed);
        let reported = events.try_recv().expect("fatal send emits a Closed event");
        assert_eq!(
            reported,
            RelayEvent::Closed {
                code: 0,
                reason:
                    "collab send failed: Trying to work with closed connection; rejoin to resync"
                        .to_string(),
                will_reconnect: false,
            }
        );
    }

    #[test]
    fn jitter_is_in_range_and_seeded_differently() {
        let mut a = initial_rng_state();
        std::thread::sleep(Duration::from_millis(2));
        let mut b = initial_rng_state();
        let mut all_in_range = true;
        let mut distinct = false;
        for _ in 0..100 {
            let ja = next_jitter01(&mut a);
            let jb = next_jitter01(&mut b);
            all_in_range &= (0.0..1.0).contains(&ja) && (0.0..1.0).contains(&jb);
            distinct |= ja != jb;
        }
        assert!(all_in_range);
        assert!(distinct, "two seeds produced identical jitter streams");
    }
}
