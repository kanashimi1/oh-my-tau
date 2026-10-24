//! Guest side of a collab live session.
//!
//! Port of `vendor-src/oh-my-pi/packages/coding-agent/src/collab/guest.ts`
//! (965 lines). [`join`] parses the collab link, connects to the relay as
//! `role=guest`, says `hello` (carrying the room's protocol version, a
//! display name, and the base64url write token when the link was a full
//! link), and waits for the host's `welcome` + `snapshot-chunk` train. The
//! train is accumulated into a local mirror of the host's replicated session
//! (replica header, transcript entries, agent ecosystem); afterwards live
//! `entry`/`event`/`state`/`agents`/`bus` frames apply strictly in arrival
//! order and host asks (`ui-request`) are answered.
//!
//! Everything renders through the embedder: [`GuestEvent`] carries the wire
//! data (the TS guest pipes the same frames into its TUI, guest.ts:1-27),
//! and [`GuestHandle`] is the control surface (`send_prompt`, `send_abort`,
//! agent commands, `respond_ui_request`, `fetch_transcript`, `leave`).
//!
//! Deviations from the TS guest, each forced by the crate boundary:
//!
//! 1. The replica *file* write + `/resume` machinery stay with the embedder
//!    (`getConfigRootDir()` + session storage live outside this crate).
//!    [`GuestSync`] exposes the replica header (rewritten to a stable
//!    replica id, guest.ts:447-452) and the transcript entries so the
//!    embedder persists them.
//! 2. TUI-only surfaces (status line, `renderInitialMessages`, todos,
//!    terminal title, extension lifecycle mirror, observer registry) have no
//!    crate-level counterpart; [`GuestEvent`] carries their inputs.
//! 3. A relay `room-closed` control message terminates the guest session
//!    (this port's spec: "host room-closed -> guest terminates"). The TS
//!    guest instead keeps retrying until a host recreates the room
//!    (relay-client.ts:726-733 classifies a guest close 4001 as a room
//!    retry); here the session is over once the room is gone. A graceful
//!    host exit still arrives as a sealed `bye` first (host.ts:553), which
//!    is what the TS guest actually acts on (guest.ts:688-694).
//!
//! # Handshake sequence (as implemented)
//!
//! 1. [`join`] parses the link ([`crate::link::parse_collab_link`]) and
//!    imports the 32-byte room key as the AES-256-GCM seal key
//!    ([`crate::crypto::CollabCipher`]).
//! 2. `connect_relay(url, room, Guest)`; on every [`RelayEvent::Opened`] the
//!    guest (re)sends `{"t":"hello","proto":3,"name":...,"writeToken":...}`
//!    sealed in a peer-0 envelope — the relay rewrites the peer id
//!    (guest.ts:312-324).
//! 3. The host verifies the write token against the link secret (host.ts:
//!    747-753, timing-safe) and answers with a targeted `welcome` carrying
//!    `{proto, header, state, agents, entryCount, readOnly?}` plus
//!    `snapshot-chunk` frames until `final` (host.ts:772-824).
//! 4. The guest finalizes the mirror once `entryCount` entries arrived or a
//!    chunk is tagged `final` (guest.ts:476), and the join resolves.
//!
//! Frames are sealed with the room key; the relay only ever sees the
//! plaintext envelope prefix and TEXT control JSON.

use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures::{Stream, StreamExt};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use crate::{
    crypto::CollabCipher,
    link::parse_collab_link,
    protocol::{encode_base64_url, pack_envelope, unpack_envelope},
    relay_client::{self, CollabRole, RelayEvent, RelayHandle, RelayOptions},
};

/// Wire protocol version carried in `hello`; the host rejects mismatches
/// (wire/src/index.ts:424, protocol.ts:75).
pub const COLLAB_PROTO: u32 = 3;

/// How long the guest waits for the host's small `welcome` frame before
/// giving up on the join (guest.ts:73-78). The welcome carries metadata
/// only, so it lands well under a second on any working relay.
pub const WELCOME_TIMEOUT_MS: u64 = 30_000;

/// How long the guest waits between `snapshot-chunk` frames during the
/// initial sync; resets on each chunk arrival, so a multi-MB snapshot only
/// fails when the relay genuinely stalls (guest.ts:79-88).
pub const SNAPSHOT_PROGRESS_TIMEOUT_MS: u64 = 30_000;

/// How long a `fetch-transcript` reply may take (guest.ts:89).
pub const TRANSCRIPT_TIMEOUT_MS: u64 = 20_000;

// ═══════════════════════════════════════════════════════════════════════════
// Wire frames (`CollabFrame`, protocol.ts:44-104; sealed as JSON inside the
// AES-GCM envelope — crypto.ts:36-58). Field order mirrors the TS object
// literals exactly: `JSON.stringify` emits declaration order, and the goldens
// in the tests pin the bytes against the TS encoders.
// ═══════════════════════════════════════════════════════════════════════════

/// One collab frame, both directions of the guest-host grammar
/// (wire/src/index.ts:326-383). Session-typed fields (`header`, `state`,
/// `entry`, `event`, ...) travel as opaque JSON here: the Rust session/agent
/// model does not exist yet, and [`crate::replication`] is Value-based.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "camelCase")]
pub enum CollabFrame {
    /// guest -> host, on every (re)connect (guest.ts:313-320).
    #[serde(rename_all = "camelCase")]
    Hello {
        /// Wire protocol version; the host rejects mismatches.
        proto: u32,
        /// Display name for participant lists.
        name: String,
        /// base64url write token proving full-link possession; absent for
        /// view links (wire/src/index.ts:332-337).
        #[serde(skip_serializing_if = "Option::is_none")]
        write_token: Option<String>,
    },
    /// guest -> host (guest.ts:411-413).
    #[serde(rename_all = "camelCase")]
    Prompt {
        /// Prompt text.
        text: String,
        /// Optional image attachments; omitted when empty, exactly like the
        /// TS `images: images && images.length > 0 ? images : undefined`.
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        images: Vec<Value>,
    },
    /// guest -> host (guest.ts:416-418).
    Abort {},
    /// guest -> host, answer to a `ui-request` (guest.ts:846).
    #[serde(rename = "ui-response", rename_all = "camelCase")]
    UiResponse {
        /// The request being answered.
        req_id: u64,
        /// `None` mirrors the TS `value: undefined` (the web client's
        /// Cancel button); the key is omitted, not null (guest.ts:843-845).
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<Value>,
    },
    /// guest -> host, Agent Hub actions routed to the host (guest.ts:227-238).
    #[serde(rename = "agent-cmd", rename_all = "camelCase")]
    AgentCmd {
        /// `chat` | `kill` | `revive`.
        cmd: String,
        /// Target agent id.
        agent_id: String,
        /// Chat text, chat only.
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    },
    /// guest -> host, remote transcript read (guest.ts:251).
    #[serde(rename = "fetch-transcript", rename_all = "camelCase")]
    FetchTranscript {
        /// Caller-side request id.
        req_id: u64,
        /// Agent whose transcript to read.
        agent_id: String,
        /// Byte offset into the transcript JSONL.
        from_byte: u64,
    },
    /// host -> guest, opens the snapshot train (guest.ts:368-384).
    #[serde(rename_all = "camelCase")]
    Welcome {
        /// Echoed protocol version (informational for guests).
        proto: u32,
        /// The host's session header JSON.
        header: Value,
        /// The host's session state JSON (`CollabSessionState`).
        state: Value,
        /// Full agent-registry snapshot.
        agents: Vec<Value>,
        /// Total entries the following chunks will deliver; the guest stays
        /// in the loading phase until all arrived or a chunk is `final`
        /// (guest.ts:361-366, protocol.ts:71-77).
        entry_count: u64,
        /// True when this peer joined through a view link (host.ts:811);
        /// omitted for writable peers.
        #[serde(skip_serializing_if = "std::ops::Not::not", default)]
        read_only: bool,
    },
    /// host -> guest, one byte-bounded batch of the snapshot
    /// (guest.ts:453-480).
    #[serde(rename = "snapshot-chunk", rename_all = "camelCase")]
    SnapshotChunk {
        /// Pre-serialized session entries, in snapshot order.
        entries: Vec<Value>,
        /// `final` on the wire — always present; keyword-escaped in Rust.
        #[serde(rename = "final")]
        is_final: bool,
    },
    /// host -> guest, one live session entry (guest.ts:656-672).
    Entry {
        /// The replicated session entry JSON.
        entry: Value,
    },
    /// host -> guest, one mirrored agent-session event (guest.ts:673-675).
    Event {
        /// The mirrored `AgentSessionEvent` JSON.
        event: Value,
    },
    /// host -> guest, debounced session-state snapshot (guest.ts:676-687).
    State {
        /// The `CollabSessionState` JSON.
        state: Value,
    },
    /// host -> guest, mirrored `EventBus` traffic (guest.ts:688-690).
    #[serde(rename_all = "camelCase")]
    Bus {
        /// `task:subagent:progress` | `task:subagent:lifecycle`.
        channel: String,
        /// The bus payload.
        data: Value,
    },
    /// host -> guest, full agent-registry snapshot (guest.ts:691-693).
    Agents {
        /// Agent snapshots.
        agents: Vec<Value>,
    },
    /// host -> guest, host ask presented locally (guest.ts:694-696).
    #[serde(rename = "ui-request")]
    UiRequest {
        /// The `CollabUiRequest` JSON.
        request: Value,
    },
    /// host -> guest, dismiss a pending ask without responding
    /// (guest.ts:697-698).
    #[serde(rename = "ui-request-end", rename_all = "camelCase")]
    UiRequestEnd {
        /// The settled request id.
        req_id: u64,
    },
    /// host -> guest, targeted reply to `fetch-transcript` (guest.ts:699-707).
    #[serde(rename_all = "camelCase")]
    Transcript {
        /// The request this answers.
        req_id: u64,
        /// Decoded JSONL text from `fromByte`.
        text: String,
        /// Next read offset.
        new_size: u64,
        /// Terminal read failure; guests surface it without hot retrying.
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// host -> guest, the session is over (guest.ts:708-709).
    Bye {
        /// Why the host ended the session.
        reason: String,
    },
    /// host -> guest, targeted error (protocol mismatch, read-only refusal,
    /// guest.ts:710-711).
    Error {
        /// The host's message.
        message: String,
    },
}

/// What [`open_frame`] made of a received envelope payload.
enum DecodedFrame {
    /// A parsed frame.
    Frame(CollabFrame),
    /// Authenticated but not a frame — skipped like the TS recvChain catch
    /// (relay-client.ts:677-681).
    Malformed,
    /// Decryption/auth failure — fatal for a guest
    /// (relay-client.ts:645-656).
    AuthFailed,
}

/// Open a received envelope payload into a frame.
fn open_frame(cipher: &CollabCipher, sealed: &[u8]) -> DecodedFrame {
    let Some(plaintext) = cipher.open(sealed) else {
        return DecodedFrame::AuthFailed;
    };
    match serde_json::from_slice(&plaintext) {
        Ok(frame) => DecodedFrame::Frame(frame),
        Err(_) => DecodedFrame::Malformed,
    }
}

/// Seal one frame into a peer-0 envelope (guest send path; the relay
/// rewrites the peer id, protocol.ts:179-181).
fn seal_frame(cipher: &CollabCipher, frame: &CollabFrame) -> Vec<u8> {
    let json = serde_json::to_string(frame).expect("CollabFrame serializes infallibly");
    pack_envelope(0, &cipher.seal(json.as_bytes()))
}

// ═══════════════════════════════════════════════════════════════════════════
// Options, errors, mirror types
// ═══════════════════════════════════════════════════════════════════════════

/// Join options. Defaults mirror the TS constants; tests shrink the timers.
#[derive(Debug, Clone)]
pub struct GuestOptions {
    /// Display name sent in `hello` (TS `collabDisplayName`: settings -> OS
    /// username -> "anonymous", display-name.ts:8-15).
    pub name: String,
    /// First `welcome` budget (guest.ts:73-78).
    pub welcome_timeout_ms: u64,
    /// Inter-`snapshot-chunk` budget during the initial sync (guest.ts:79-88).
    pub snapshot_progress_timeout_ms: u64,
    /// `fetch-transcript` budget (guest.ts:89).
    pub transcript_timeout_ms: u64,
    /// Transport tuning (reconnect backoff, heartbeat). Defaults are the TS
    /// relay-client constants.
    pub relay: RelayOptions,
}

impl Default for GuestOptions {
    fn default() -> Self {
        Self {
            name: current_username(),
            welcome_timeout_ms: WELCOME_TIMEOUT_MS,
            snapshot_progress_timeout_ms: SNAPSHOT_PROGRESS_TIMEOUT_MS,
            transcript_timeout_ms: TRANSCRIPT_TIMEOUT_MS,
            relay: RelayOptions::default(),
        }
    }
}

/// OS username, or "anonymous" when it cannot be determined — the TS
/// `os.userInfo().username` fallback (display-name.ts:11-14).
fn current_username() -> String {
    #[cfg(unix)]
    if let Ok(Some(user)) = nix::unistd::User::from_uid(nix::unistd::getuid()) {
        return user.name;
    }
    "anonymous".to_string()
}

/// Why a mutating guest action was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GuestSendError {
    /// The link was a read-only (view) link; the TS guest shows
    /// "This collab link is read-only" and drops the action
    /// (guest.ts:247-253).
    #[error("This collab link is read-only")]
    ReadOnly,
    /// The session already ended; the TS socket's sends are no-ops once
    /// closed (relay-client.ts:199-201) — surfaced here so callers can tell.
    #[error("collab session has ended")]
    Ended,
}

/// Local mirror of one host agent (wire/src/index.ts:238-248), diffed from
/// `welcome`/`agents` snapshots like the TS `AgentRegistry` updates
/// (guest.ts:763-790).
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRef {
    /// Host agent id.
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// `"main"` or `"sub"`.
    pub kind: String,
    /// Parent subagent id, if any.
    pub parent_id: Option<String>,
    /// `"running" | "idle" | "parked" | "aborted"`.
    pub status: String,
    /// Whether the host has a transcript file for this agent; gates
    /// `fetch-transcript` (guest.ts:227-231).
    pub has_session_file: bool,
    /// Host-side creation time (ms epoch).
    pub created_at: f64,
    /// Host-side last activity (ms epoch).
    pub last_activity: f64,
}

/// The replicated session as of a `welcome` + chunk train — the mirror the
/// TS guest writes to its replica file and resumes into (guest.ts:437-602).
#[derive(Debug, Clone, PartialEq)]
pub struct GuestSnapshot {
    /// The replica header: the host's header with `id` replaced by the
    /// stable replica session id, `parentSession` pointed at the host's
    /// session, and `providerPromptCacheKey` defaulted to the host's id
    /// (guest.ts:447-452 — a child of the host's session, as a sibling move
    /// is).
    pub replica_header: Value,
    /// Transcript entries, in snapshot order.
    pub entries: Vec<Value>,
}

/// Payload of [`GuestEvent::Synced`]: the freshly synced mirror.
#[derive(Debug, Clone, PartialEq)]
pub struct GuestSync {
    /// False on the initial join sync, true after a reconnect resync — the
    /// TS `Joined collab session` vs `Reconnected to collab session` status
    /// distinction (guest.ts:599-602).
    pub is_resync: bool,
    /// True when the host marked this peer read-only (view link).
    pub read_only: bool,
    /// The replica snapshot (rewritten header + entries).
    pub snapshot: GuestSnapshot,
    /// The host's session state at sync time.
    pub state: Option<Value>,
    /// The diffed agent mirror.
    pub agents: Vec<AgentRef>,
}

/// One resolved `fetch-transcript` (TS `AgentHubRemoteTranscript`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteTranscript {
    /// Decoded JSONL from the requested offset.
    pub text: String,
    /// Next read offset.
    pub new_size: u64,
    /// Terminal read failure, if any.
    pub error: Option<String>,
}

/// Everything that happens on a joined guest, in wire order. The TS guest
/// funnels the same frames into its TUI (guest.ts:1-27); an embedder renders
/// from these.
#[derive(Debug, Clone, PartialEq)]
pub enum GuestEvent {
    /// The initial snapshot (or a reconnect resync) landed; the mirror is
    /// current as of this event.
    Synced {
        /// Full payload: replica header, entries, state, agents.
        sync: GuestSync,
    },
    /// One live session entry; append it to the replica
    /// (guest.ts:656-672 — entries are never rendered directly, they keep
    /// the replica file and message array current).
    Entry {
        /// The replicated entry JSON.
        entry: Value,
    },
    /// One mirrored agent-session event. A `message_update` that arrives
    /// without a prior assistant `message_start` since (re)sync is preceded
    /// by a synthesized `message_start` (guest.ts:738-750).
    Event {
        /// The `AgentSessionEvent` JSON.
        event: Value,
    },
    /// Host session state updated (guest.ts:676-687).
    State {
        /// The `CollabSessionState` JSON.
        state: Value,
    },
    /// Mirrored host `EventBus` traffic (guest.ts:688-690).
    Bus {
        /// Bus channel.
        channel: String,
        /// Bus payload.
        data: Value,
    },
    /// Agent ecosystem updated (guest.ts:691-693).
    Agents {
        /// The full diffed mirror.
        agents: Vec<AgentRef>,
    },
    /// A host ask to present; answer with
    /// [`GuestHandle::respond_ui_request`] (guest.ts:829-853).
    UiRequest {
        /// The `CollabUiRequest` JSON.
        request: Value,
    },
    /// A pending ask was settled elsewhere; dismiss it without responding
    /// (guest.ts:696-698, 859-869).
    UiRequestEnd {
        /// The settled request id.
        req_id: u64,
    },
    /// Post-welcome targeted host error; the session stays up
    /// (guest.ts:710-711).
    Error {
        /// The host's message.
        message: String,
    },
    /// Transient disconnect; the transport is retrying and a fresh `hello`
    /// follows on the next `Opened` (guest.ts:682-684).
    Reconnecting {
        /// Why the socket dropped.
        reason: String,
    },
    /// Relay control: a peer joined the room. Today's relay sends these to
    /// hosts only; surfaced for wire parity (wire/src/index.ts:445).
    PeerJoined {
        /// The new peer id.
        peer: u32,
    },
    /// Relay control: a peer left (wire/src/index.ts:445).
    PeerLeft {
        /// The departed peer id.
        peer: u32,
    },
    /// Terminal: host `bye`, room closed, fatal transport error, or local
    /// leave. Exactly one per session; the stream ends after it.
    Ended {
        /// Why the session ended.
        reason: String,
    },
}

/// A [`GuestEvent`] stream; ends after [`GuestEvent::Ended`].
#[derive(Debug)]
pub struct GuestEventStream {
    rx: mpsc::UnboundedReceiver<GuestEvent>,
}

impl Stream for GuestEventStream {
    type Item = GuestEvent;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// The guest's shared mirror, readable from the [`GuestHandle`] without
/// touching the driver task.
#[derive(Debug)]
struct GuestCore {
    /// Stable replica session id (TS `#replicaId`, minted once per guest —
    /// never the host's id, never changed on a resync, guest.ts:171-181).
    session_id: String,
    /// True once the host marked this peer read-only (view link), latched at
    /// finalize like the TS `#readOnly` (guest.ts:598).
    read_only: bool,
    /// True after a snapshot finalized; reset on every reconnect so live
    /// frames are dropped until the fresh welcome (guest.ts:201, 315).
    welcomed: bool,
    /// Rewritten replica header, set at finalize.
    replica_header: Option<Value>,
    /// Transcript entries (snapshot train, then live appends).
    entries: Vec<Value>,
    /// Last host session state.
    state: Option<Value>,
    /// Diffed agent mirror in registration order.
    agents: Vec<AgentRef>,
    /// Per-agent `hasSessionFile` from the last snapshot; gates remote
    /// transcript fetches (guest.ts:203, 795-804).
    agent_has_transcript: HashMap<String, bool>,
}

/// Accumulator for an in-flight chunked welcome (guest.ts:97-111).
#[derive(Clone)]
struct PendingSnapshot {
    header: Value,
    state: Value,
    agents: Vec<Value>,
    read_only: bool,
    entry_count: u64,
    entries: Vec<Value>,
    is_resync: bool,
}

impl PendingSnapshot {
    /// Complete when every entry the welcome promised arrived; the host's
    /// `final` tag is checked by the caller (guest.ts:476).
    fn is_complete(&self) -> bool {
        u64::try_from(self.entries.len()).is_ok_and(|len| len >= self.entry_count)
    }
}

/// Which side of the driver loop woke this iteration.
enum DriverEvent {
    Relay(Option<RelayEvent>),
    Command(Option<GuestCommand>),
    Deadline,
}

/// Which join-phase budget the armed deadline enforces (guest.ts:585-604).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncDeadlineKind {
    /// Waiting for the first `welcome` (also covers a blackholed handshake).
    Welcome,
    /// Waiting for the next `snapshot-chunk`.
    Snapshot,
}

struct SyncDeadline {
    sleep: Pin<Box<tokio::time::Sleep>>,
    kind: SyncDeadlineKind,
}

// ═══════════════════════════════════════════════════════════════════════════
// Handle
// ═══════════════════════════════════════════════════════════════════════════

/// Commands the handle forwards to the driver task (the driver owns the
/// cipher and the socket, so sends stay serialized behind the frame
/// application order — the TS `#applyChain` invariant, guest.ts:186-188).
enum GuestCommand {
    Prompt {
        text: String,
        images: Vec<Value>,
    },
    Abort,
    AgentCmd {
        cmd: String,
        agent_id: String,
        text: Option<String>,
    },
    UiResponse {
        req_id: u64,
        value: Option<Value>,
    },
    FetchTranscript {
        agent_id: String,
        from_byte: u64,
        reply: oneshot::Sender<Option<RemoteTranscript>>,
    },
    Leave,
}

/// Clonable control surface for a joined guest.
#[derive(Debug, Clone)]
pub struct GuestHandle {
    cmd: mpsc::UnboundedSender<GuestCommand>,
    core: Arc<Mutex<GuestCore>>,
    transcript_timeout_ms: u64,
}

impl GuestHandle {
    /// Stable replica session id (TS `#replicaId`).
    #[must_use]
    ///
    /// # Panics
    /// When the shared mirror's mutex is poisoned (a driver task panicked
    /// while holding it).
    pub fn session_id(&self) -> String {
        self.core.lock().expect("guest core").session_id.clone()
    }

    /// True when the host marked this peer read-only (view link). Only
    /// meaningful after the first [`GuestEvent::Synced`]; before it the
    /// guest has no welcome yet (TS `#readOnly` latches at finalize,
    /// guest.ts:598).
    #[must_use]
    ///
    /// # Panics
    /// When the shared mirror's mutex is poisoned (a driver task panicked
    /// while holding it).
    pub fn is_read_only(&self) -> bool {
        self.core.lock().expect("guest core").read_only
    }

    /// Last host session state, if any has arrived.
    #[must_use]
    ///
    /// # Panics
    /// When the shared mirror's mutex is poisoned (a driver task panicked
    /// while holding it).
    pub fn state(&self) -> Option<Value> {
        self.core.lock().expect("guest core").state.clone()
    }

    /// The replicated session mirror, if a snapshot train has completed.
    #[must_use]
    ///
    /// # Panics
    /// When the shared mirror's mutex is poisoned (a driver task panicked
    /// while holding it).
    pub fn snapshot(&self) -> Option<GuestSnapshot> {
        let core = self.core.lock().expect("guest core");
        Some(GuestSnapshot {
            replica_header: core.replica_header.clone()?,
            entries: core.entries.clone(),
        })
    }

    /// Current agent mirror (TS `agentRegistry.list()`).
    #[must_use]
    ///
    /// # Panics
    /// When the shared mirror's mutex is poisoned (a driver task panicked
    /// while holding it).
    pub fn agents(&self) -> Vec<AgentRef> {
        self.core.lock().expect("guest core").agents.clone()
    }

    fn send_cmd(&self, cmd: GuestCommand) -> Result<(), GuestSendError> {
        self.cmd.send(cmd).map_err(|_| GuestSendError::Ended)
    }

    /// Send a prompt to the host. Refused locally on a read-only link —
    /// nothing reaches the wire, mirroring the TS `#rejectReadOnly` status
    /// + drop (guest.ts:247-253, 411-413).
    ///
    /// # Errors
    /// [`GuestSendError::ReadOnly`] on a view link,
    /// [`GuestSendError::Ended`] once the session is over.
    pub fn send_prompt(&self, text: impl Into<String>) -> Result<(), GuestSendError> {
        self.send_prompt_inner(text.into(), Vec::new())
    }

    /// [`GuestHandle::send_prompt`] with image attachments; an empty
    /// `images` omits the key, exactly like the TS send (guest.ts:412).
    ///
    /// # Errors
    /// Same as [`GuestHandle::send_prompt`].
    pub fn send_prompt_with_images(
        &self,
        text: impl Into<String>,
        images: Vec<Value>,
    ) -> Result<(), GuestSendError> {
        self.send_prompt_inner(text.into(), images)
    }

    fn send_prompt_inner(&self, text: String, images: Vec<Value>) -> Result<(), GuestSendError> {
        if self.is_read_only() {
            return Err(GuestSendError::ReadOnly);
        }
        self.send_cmd(GuestCommand::Prompt { text, images })
    }

    /// Interrupt the host's current run (guest.ts:416-418).
    ///
    /// # Errors
    /// [`GuestSendError::ReadOnly`] on a view link,
    /// [`GuestSendError::Ended`] once the session is over.
    pub fn send_abort(&self) -> Result<(), GuestSendError> {
        if self.is_read_only() {
            return Err(GuestSendError::ReadOnly);
        }
        self.send_cmd(GuestCommand::Abort)
    }

    /// Agent Hub action routed to the host (guest.ts:227-238): chat with,
    /// kill, or revive a mirrored subagent. `cmd` is `"chat"`, `"kill"`, or
    /// `"revive"`; `text` applies to chat only.
    ///
    /// # Errors
    /// [`GuestSendError::ReadOnly`] on a view link,
    /// [`GuestSendError::Ended`] once the session is over.
    pub fn send_agent_cmd(
        &self,
        cmd: &str,
        agent_id: &str,
        text: Option<String>,
    ) -> Result<(), GuestSendError> {
        if self.is_read_only() {
            return Err(GuestSendError::ReadOnly);
        }
        self.send_cmd(GuestCommand::AgentCmd {
            cmd: cmd.to_string(),
            agent_id: agent_id.to_string(),
            text,
        })
    }

    /// Answer a [`GuestEvent::UiRequest`]. `None` is the TS
    /// `value: undefined` (web Cancel button, guest.ts:843-845). Ownership
    /// is enforced by the driver: once a request was dismissed
    /// (`ui-request-end`) the answer frame is suppressed, like the TS
    /// identity check (guest.ts:839-846). Returns `false` only when the
    /// session already ended.
    #[must_use]
    pub fn respond_ui_request(&self, req_id: u64, value: Option<Value>) -> bool {
        self.send_cmd(GuestCommand::UiResponse { req_id, value })
            .is_ok()
    }

    /// Read a host agent's transcript from `from_byte`. Resolves `None`
    /// without wire traffic when the agent has no transcript file
    /// (guest.ts:227-231), on timeout after [`TRANSCRIPT_TIMEOUT_MS`]
    /// (guest.ts:89-90), or once the session ended (guest.ts:891,
    /// `#flushPendingTranscripts`).
    ///
    /// # Panics
    /// When the shared mirror's mutex is poisoned (a driver task panicked
    /// while holding it).
    pub async fn fetch_transcript(
        &self,
        agent_id: &str,
        from_byte: u64,
    ) -> Option<RemoteTranscript> {
        if matches!(
            self.core
                .lock()
                .expect("guest core")
                .agent_has_transcript
                .get(agent_id),
            Some(false)
        ) {
            return None;
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_cmd(GuestCommand::FetchTranscript {
            agent_id: agent_id.to_string(),
            from_byte,
            reply: reply_tx,
        })
        .ok()?;
        // The TS timer resolves null at TRANSCRIPT_TIMEOUT_MS
        // (guest.ts:240-247); the driver flushes earlier on disconnect.
        match tokio::time::timeout(Duration::from_millis(self.transcript_timeout_ms), reply_rx)
            .await
        {
            Ok(Ok(transcript)) => transcript,
            _ => None,
        }
    }

    /// User-initiated leave: close the socket and end the session (TS
    /// `leave`, guest.ts:406-409 — the host observes a `peer-left`).
    pub fn leave(&self) {
        let _ = self.cmd.send(GuestCommand::Leave);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Join + driver
// ═══════════════════════════════════════════════════════════════════════════

/// Join a collab session as a guest. Resolves after the initial snapshot
/// sync completes (the TS `join` awaits `firstWelcome`, which the host's
/// final `snapshot-chunk` settles, guest.ts:369-383, 426-431). Events that
/// arrive during the join are queued and delivered through the returned
/// stream.
///
/// # Errors
/// With the TS error strings when the link is invalid, the host rejects the
/// `hello` (targeted pre-welcome `error` frame, guest.ts:367-383), the room
/// is missing or the key is bad, or the welcome/snapshot budgets elapse
/// (guest.ts:586, 600).
pub async fn join(
    link: &str,
    options: GuestOptions,
) -> anyhow::Result<(GuestHandle, GuestEventStream)> {
    let parsed = parse_collab_link(link).map_err(anyhow::Error::msg)?;
    let cipher = CollabCipher::new(&parsed.key);
    // base64url of the raw token bytes — exactly the TS `#writeToken` string
    // put on the wire (guest.ts:293-295).
    let write_token = parsed.write_token.as_deref().map(encode_base64_url);
    let (relay, relay_events) = relay_client::connect_relay(
        &parsed.relay_url,
        &parsed.room_id,
        CollabRole::Guest,
        &options.relay,
    )
    .await?;
    let transcript_timeout_ms = options.transcript_timeout_ms;
    let core = Arc::new(Mutex::new(GuestCore {
        session_id: mint_session_id(),
        read_only: false,
        welcomed: false,
        replica_header: None,
        entries: Vec::new(),
        state: None,
        agents: Vec::new(),
        agent_has_transcript: HashMap::new(),
    }));
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (join_tx, join_rx) = oneshot::channel::<Result<(), String>>();
    let mut driver = GuestDriver {
        core: Arc::clone(&core),
        cipher,
        write_token,
        name: options.name.clone(),
        opts: options,
        relay,
        relay_events: Box::pin(relay_events),
        cmd_rx,
        events: event_tx,
        join_tx: Some(join_tx),
        pending: None,
        pending_transcripts: HashMap::new(),
        pending_ui: HashSet::new(),
        next_req_id: 1,
        assistant_stream_synced: false,
        synced_once: false,
        ended: false,
        sync_deadline: None,
    };
    // Cover a blackholed WebSocket handshake too; `Opened` re-arms it
    // (guest.ts:424-426).
    driver.arm_sync_deadline(SyncDeadlineKind::Welcome);
    driver.run();
    match join_rx.await {
        Ok(Ok(())) => Ok((
            GuestHandle {
                cmd: cmd_tx,
                core,
                transcript_timeout_ms,
            },
            GuestEventStream { rx: event_rx },
        )),
        Ok(Err(message)) => Err(anyhow::Error::msg(message)),
        // The driver exited without settling the join (task loss).
        Err(_) => Err(anyhow::Error::msg("collab guest driver exited")),
    }
}

/// `Bun.randomUUIDv7()`-shaped session id (session-manager.ts:129-131):
/// 48-bit big-endian millisecond timestamp, version 7, RFC 4122 variant.
fn mint_session_id() -> String {
    use std::fmt::Write as _;
    let millis = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_millis()),
    )
    .unwrap_or(u64::MAX)
        & 0xffff_ffff_ffff;
    let mut bytes = [0_u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    bytes[..6].copy_from_slice(&millis.to_be_bytes()[2..]);
    bytes[6] = (bytes[6] & 0x0f) | 0x70;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let mut hex = String::with_capacity(32);
    for byte in bytes {
        let _ = write!(hex, "{byte:02x}");
    }
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// The guest session state machine. One task means frames apply strictly in
/// arrival order — the TS `#applyChain` invariant (guest.ts:186-188) — and
/// cancellation/snapshot work have one owner, like `#restoration`
/// (guest.ts:191).
struct GuestDriver {
    core: Arc<Mutex<GuestCore>>,
    cipher: CollabCipher,
    /// base64url write token from the link; `None` on view links.
    write_token: Option<String>,
    name: String,
    opts: GuestOptions,
    relay: RelayHandle,
    relay_events: Pin<Box<dyn Stream<Item = RelayEvent> + Send>>,
    cmd_rx: mpsc::UnboundedReceiver<GuestCommand>,
    events: mpsc::UnboundedSender<GuestEvent>,
    /// Settles the pending [`join`]; `None` once joined or failed — later
    /// deadline arming is a no-op, so reconnect-time stalls fall through to
    /// the normal socket close handling (guest.ts:159-168).
    join_tx: Option<oneshot::Sender<Result<(), String>>>,
    /// In-flight chunked welcome (guest.ts:158-167).
    pending: Option<PendingSnapshot>,
    pending_transcripts: HashMap<u64, oneshot::Sender<Option<RemoteTranscript>>>,
    /// reqIds with a locally presented ask (guest.ts:225-226).
    pending_ui: HashSet<u64>,
    next_req_id: u64,
    /// False until the first assistant `message_start` (real or
    /// synthesized) since (re)sync (guest.ts:201-202).
    assistant_stream_synced: bool,
    /// True after the first completed sync; a later welcome is a resync.
    synced_once: bool,
    ended: bool,
    sync_deadline: Option<SyncDeadline>,
}

impl GuestDriver {
    /// Spawn the event loop; it ends once the session does.
    fn run(self) {
        tokio::spawn(self.run_inner());
    }

    async fn run_inner(mut self) {
        while !self.ended {
            // Take the deadline out for the poll: the select future borrows
            // it, so every other mutation has to wait for the poll to
            // settle. It is put back before the arm handlers run.
            let mut deadline = self.sync_deadline.take();
            let armed = deadline.is_some();
            let deadline_wait = async {
                match deadline.as_mut() {
                    Some(pending) => pending.sleep.as_mut().await,
                    None => std::future::pending().await,
                }
            };
            let outcome = tokio::select! {
                biased;
                relay_event = self.relay_events.next() => DriverEvent::Relay(relay_event),
                cmd = self.cmd_rx.recv() => DriverEvent::Command(cmd),
                () = deadline_wait, if armed => DriverEvent::Deadline,
            };
            self.sync_deadline = deadline;
            match outcome {
                DriverEvent::Relay(Some(event)) => self.on_relay_event(event),
                DriverEvent::Relay(None) => self.on_relay_event(RelayEvent::Closed {
                    code: 1006,
                    reason: "relay event stream ended".to_string(),
                    will_reconnect: false,
                }),
                DriverEvent::Command(Some(cmd)) => self.on_command(cmd),
                DriverEvent::Command(None) => self.shutdown("handle dropped"),
                DriverEvent::Deadline => {
                    let kind = self
                        .sync_deadline
                        .take()
                        .expect("deadline fired while armed")
                        .kind;
                    // Only reachable while joining: arming after the join
                    // resolves is a no-op (guest.ts:159-168).
                    let message = match kind {
                        SyncDeadlineKind::Welcome => "timed out waiting for the host's welcome",
                        SyncDeadlineKind::Snapshot => {
                            "timed out waiting for the host's session snapshot"
                        }
                    };
                    self.fail_join(message);
                }
            }
        }
    }

    fn on_relay_event(&mut self, event: RelayEvent) {
        if self.ended {
            return;
        }
        match event {
            RelayEvent::Opened => {
                // (Re)connect: re-introduce ourselves; the host answers with
                // a fresh welcome which (re)syncs the replica. Discard any
                // partially-streamed snapshot from a prior connection — the
                // host resends the full chunk train (guest.ts:312-322).
                {
                    let mut core = self.lock_core();
                    core.welcomed = false;
                    core.replica_header = None;
                }
                self.pending = None;
                self.clear_sync_deadline();
                if self.join_tx.is_some() {
                    self.arm_sync_deadline(SyncDeadlineKind::Welcome);
                }
                self.send(&CollabFrame::Hello {
                    proto: COLLAB_PROTO,
                    name: self.name.clone(),
                    write_token: self.write_token.clone(),
                });
            }
            RelayEvent::Binary(data) => {
                let Some(envelope) = unpack_envelope(&data) else {
                    return;
                };
                match open_frame(&self.cipher, &envelope.payload) {
                    DecodedFrame::Frame(frame) => self.on_frame(frame),
                    DecodedFrame::Malformed => {}
                    DecodedFrame::AuthFailed => {
                        // Wrong key or corrupted frame: never reconnect
                        // (relay-client.ts:645-656, guest branch).
                        self.fail_join("bad key or corrupted frame");
                        self.shutdown("bad key or corrupted frame");
                    }
                }
            }
            RelayEvent::Text(text) => {
                // TEXT relay control: `room-closed` for guests; `peer-joined`
                // / `peer-left` are host-directed but parsed for wire parity
                // (wire/src/index.ts:444-447).
                let Ok(control) = serde_json::from_str::<Value>(&text) else {
                    return;
                };
                match control.get("t").and_then(Value::as_str) {
                    Some("room-closed") => {
                        // This port: the room is gone, so is the session
                        // (module docs cover the TS retry deviation).
                        self.shutdown("room closed");
                    }
                    Some("peer-joined") => {
                        if let Some(peer) = control
                            .get("peer")
                            .and_then(Value::as_u64)
                            .and_then(|peer| u32::try_from(peer).ok())
                        {
                            let _ = self.events.send(GuestEvent::PeerJoined { peer });
                        }
                    }
                    Some("peer-left") => {
                        if let Some(peer) = control
                            .get("peer")
                            .and_then(Value::as_u64)
                            .and_then(|peer| u32::try_from(peer).ok())
                        {
                            let _ = self.events.send(GuestEvent::PeerLeft { peer });
                        }
                    }
                    _ => {}
                }
            }
            RelayEvent::Closed {
                code: _,
                reason,
                will_reconnect,
            } => {
                // Pending transcripts flush on any socket end, like the TS
                // onClose (guest.ts:386-390).
                self.flush_pending_transcripts();
                if will_reconnect {
                    if self.join_tx.is_some() {
                        // Pre-join drop: fail the join with the transport's
                        // reason, like the TS firstWelcome.reject
                        // (guest.ts:386-392).
                        self.fail_join(&reason);
                    } else {
                        let _ = self.events.send(GuestEvent::Reconnecting { reason });
                    }
                    return;
                }
                self.shutdown(reason);
            }
        }
    }

    fn on_frame(&mut self, frame: CollabFrame) {
        if self.ended {
            return;
        }
        match frame {
            CollabFrame::Welcome {
                header,
                state,
                agents,
                entry_count,
                read_only,
                ..
            } => self.on_welcome(header, state, agents, entry_count, read_only),
            CollabFrame::SnapshotChunk { entries, is_final } => {
                self.on_snapshot_chunk(entries, is_final);
            }
            CollabFrame::Error { message } => {
                if self.join_tx.is_some() {
                    // Pre-welcome errors are the host's targeted reply to
                    // our hello (e.g. protocol mismatch): no welcome will
                    // follow — fail the join with the host's message instead
                    // of hanging until the welcome timeout
                    // (guest.ts:367-383).
                    self.fail_join(&message);
                } else {
                    // Post-welcome: surface, stay connected
                    // (guest.ts:371-372, 710-711).
                    let _ = self.events.send(GuestEvent::Error { message });
                }
            }
            _frame if !self.lock_core().welcomed => {
                // Live frames before the first (or a resync) welcome are
                // ignored — the fresh snapshot covers them (guest.ts:375).
            }
            CollabFrame::Entry { entry } => {
                // Entries are never rendered directly — they keep the
                // replica file and message array current. The compaction /
                // branch-summary message-array rebuild is the embedder's
                // render step (guest.ts:656-672).
                self.lock_core().entries.push(entry.clone());
                let _ = self.events.send(GuestEvent::Entry { entry });
            }
            CollabFrame::Event { event } => self.apply_event(event),
            CollabFrame::State { state } => {
                self.lock_core().state = Some(state.clone());
                let _ = self.events.send(GuestEvent::State { state });
            }
            CollabFrame::Bus { channel, data } => {
                let _ = self.events.send(GuestEvent::Bus { channel, data });
            }
            CollabFrame::Agents { agents } => {
                apply_agent_snapshots(&mut self.lock_core(), &agents);
                let agents = self.lock_core().agents.clone();
                let _ = self.events.send(GuestEvent::Agents { agents });
            }
            CollabFrame::UiRequest { request } => {
                // The host only targets writable peers; drop defensively on
                // a read-only link, and never double-present a reqId
                // (guest.ts:833-838).
                if self.lock_core().read_only {
                    return;
                }
                let Some(req_id) = request.get("reqId").and_then(Value::as_u64) else {
                    return;
                };
                if !self.pending_ui.insert(req_id) {
                    return;
                }
                let _ = self.events.send(GuestEvent::UiRequest { request });
            }
            CollabFrame::UiRequestEnd { req_id } => {
                // Host settled the request (answered elsewhere or aborted):
                // dismiss without responding (guest.ts:859-869).
                self.pending_ui.remove(&req_id);
                let _ = self.events.send(GuestEvent::UiRequestEnd { req_id });
            }
            CollabFrame::Transcript {
                req_id,
                text,
                new_size,
                error,
            } => {
                if let Some(reply) = self.pending_transcripts.remove(&req_id) {
                    let _ = reply.send(Some(RemoteTranscript {
                        text,
                        new_size,
                        error,
                    }));
                }
            }
            CollabFrame::Bye { reason } => {
                // `Collab session ended (reason)` + local restoration
                // (guest.ts:688-691).
                self.shutdown(reason);
            }
            CollabFrame::Hello { .. }
            | CollabFrame::Prompt { .. }
            | CollabFrame::Abort {}
            | CollabFrame::UiResponse { .. }
            | CollabFrame::AgentCmd { .. }
            | CollabFrame::FetchTranscript { .. } => {
                // Hosts never send these; ignoring unexpected frames is the
                // TS default arm (guest.ts:713-715).
            }
        }
    }

    /// Latch the welcome metadata and prime the snapshot accumulator; the
    /// heavy work happens at finalize, so the small welcome frame clears the
    /// join timeout immediately (guest.ts:433-451).
    fn on_welcome(
        &mut self,
        header: Value,
        state: Value,
        agents: Vec<Value>,
        entry_count: u64,
        read_only: bool,
    ) {
        self.clear_sync_deadline();
        self.pending = Some(PendingSnapshot {
            header,
            state,
            agents,
            read_only,
            entry_count,
            entries: Vec::new(),
            is_resync: self.synced_once,
        });
        if self.join_tx.is_some() {
            self.arm_sync_deadline(SyncDeadlineKind::Snapshot);
        }
        // An empty snapshot (entryCount 0) finalizes immediately
        // (guest.ts:376-379).
        if self
            .pending
            .as_ref()
            .is_some_and(PendingSnapshot::is_complete)
        {
            self.finalize_snapshot();
        }
    }

    /// Append a chunk to the pending snapshot; complete when the host's
    /// `final` tag landed or the welcome's `entryCount` arrived
    /// (guest.ts:453-480).
    fn on_snapshot_chunk(&mut self, entries: Vec<Value>, is_final: bool) {
        let complete = match self.pending.as_mut() {
            // Orphan chunk: dropping (guest.ts:459-462).
            None => return,
            Some(pending) => {
                pending.entries.extend(entries);
                pending.is_complete() || is_final
            }
        };
        if complete {
            self.clear_sync_deadline();
            self.finalize_snapshot();
        } else if self.join_tx.is_some() {
            // Each chunk resets the progress budget (guest.ts:470-477).
            self.arm_sync_deadline(SyncDeadlineKind::Snapshot);
        }
    }

    fn on_command(&mut self, cmd: GuestCommand) {
        if self.ended {
            return;
        }
        match cmd {
            GuestCommand::Prompt { text, images } => {
                if self.lock_core().read_only {
                    return; // the handle already refuses; belt and braces
                }
                self.send(&CollabFrame::Prompt { text, images });
            }
            GuestCommand::Abort => {
                if self.lock_core().read_only {
                    return;
                }
                self.send(&CollabFrame::Abort {});
            }
            GuestCommand::AgentCmd {
                cmd,
                agent_id,
                text,
            } => {
                if self.lock_core().read_only {
                    return;
                }
                self.send(&CollabFrame::AgentCmd {
                    cmd,
                    agent_id,
                    text,
                });
            }
            GuestCommand::UiResponse { req_id, value } => {
                // Only the presentation that still owns the reqId may
                // respond (guest.ts:839-846).
                if self.pending_ui.remove(&req_id) {
                    self.send(&CollabFrame::UiResponse { req_id, value });
                }
            }
            GuestCommand::FetchTranscript {
                agent_id,
                from_byte,
                reply,
            } => {
                if matches!(
                    self.lock_core().agent_has_transcript.get(&agent_id),
                    Some(false)
                ) {
                    let _ = reply.send(None);
                    return;
                }
                let req_id = self.next_req_id;
                self.next_req_id += 1;
                self.pending_transcripts.insert(req_id, reply);
                self.send(&CollabFrame::FetchTranscript {
                    req_id,
                    agent_id,
                    from_byte,
                });
            }
            GuestCommand::Leave => self.shutdown("left"),
        }
    }

    // ── snapshot finalization ──────────────────────────────────────────

    /// Fold the accumulated welcome snapshot into the mirror (TS writes the
    /// replica file + resumes; the file write is the embedder's step here —
    /// guest.ts:437-602).
    fn finalize_snapshot(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        self.clear_sync_deadline();
        let replica_header = rewrite_replica_header(&pending.header, &self.lock_core().session_id);
        {
            let mut core = self.lock_core();
            core.replica_header = Some(replica_header);
            core.entries = pending.entries;
            core.state = Some(pending.state.clone());
            core.read_only = pending.read_only;
        }
        apply_agent_snapshots(&mut self.lock_core(), &pending.agents);
        // A resync clears locally presented asks — the host replays the ones
        // still pending after the fresh welcome (guest.ts:855-865) — and
        // re-latches the assistant stream (guest.ts:569).
        self.pending_ui.clear();
        self.assistant_stream_synced = false;
        self.synced_once = true;
        let sync = {
            let core = self.lock_core();
            GuestSync {
                is_resync: pending.is_resync,
                read_only: core.read_only,
                snapshot: GuestSnapshot {
                    replica_header: core.replica_header.clone().unwrap_or(Value::Null),
                    entries: core.entries.clone(),
                },
                state: core.state.clone(),
                agents: core.agents.clone(),
            }
        };
        let _ = self.events.send(GuestEvent::Synced { sync });
        self.lock_core().welcomed = true;
        if let Some(join_tx) = self.join_tx.take() {
            let _ = join_tx.send(Ok(()));
        }
    }

    // ── event application ──────────────────────────────────────────────

    /// Mirror one host agent-session event with the orphan-delta guard
    /// (guest.ts:738-750).
    fn apply_event(&mut self, event: Value) {
        if let Some(start) =
            synthesize_missing_message_start(&mut self.assistant_stream_synced, &event)
        {
            let _ = self.events.send(GuestEvent::Event { event: start });
        }
        let _ = self.events.send(GuestEvent::Event { event });
    }

    // ── lifecycle ──────────────────────────────────────────────────────

    /// Resolve every in-flight transcript with `None` (guest.ts:826-832).
    fn flush_pending_transcripts(&mut self) {
        for (_, reply) in self.pending_transcripts.drain() {
            let _ = reply.send(None);
        }
    }

    /// Terminal teardown, exactly once: flush transcripts, dismiss pending
    /// asks without responding, close the socket, emit `Ended` — the TS
    /// `#restoreLocalSession` shape (guest.ts:893-911).
    fn shutdown(&mut self, reason: impl Into<String>) {
        if self.ended {
            return;
        }
        self.ended = true;
        self.clear_sync_deadline();
        self.flush_pending_transcripts();
        self.pending_ui.clear();
        let reason = reason.into();
        if let Some(join_tx) = self.join_tx.take() {
            let _ = join_tx.send(Err(reason.clone()));
        }
        let _ = self.events.send(GuestEvent::Ended { reason });
        self.relay.close();
    }

    /// Fail a pending join (and end the session — the TS join throws and
    /// restores the local session, guest.ts:426-440).
    fn fail_join(&mut self, message: &str) {
        if self.join_tx.is_some() {
            self.shutdown(message.to_string());
        }
    }

    fn send(&mut self, frame: &CollabFrame) {
        let _ = self.relay.send_binary(seal_frame(&self.cipher, frame));
    }

    fn lock_core(&self) -> std::sync::MutexGuard<'_, GuestCore> {
        self.core.lock().expect("guest core")
    }

    fn arm_sync_deadline(&mut self, kind: SyncDeadlineKind) {
        // Arming after the join settled is a no-op (guest.ts:583-585,
        // 596-598: `if (this.#joinReject === null) return`).
        if self.join_tx.is_none() {
            return;
        }
        let ms = match kind {
            SyncDeadlineKind::Welcome => self.opts.welcome_timeout_ms,
            SyncDeadlineKind::Snapshot => self.opts.snapshot_progress_timeout_ms,
        };
        self.sync_deadline = Some(SyncDeadline {
            sleep: Box::pin(tokio::time::sleep(Duration::from_millis(ms))),
            kind,
        });
    }

    fn clear_sync_deadline(&mut self) {
        self.sync_deadline = None;
    }
}

/// Orphan-delta guard (guest.ts:738-750): a `message_update` for an
/// assistant message without a prior `message_start` since (re)sync yields
/// the synthesized start and latches the stream. Returns `None` for events
/// that need no synthesis.
fn synthesize_missing_message_start(synced: &mut bool, event: &Value) -> Option<Value> {
    let kind = event.get("type").and_then(Value::as_str);
    let role = event
        .get("message")
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str);
    match (kind, role) {
        (Some("message_start"), Some("assistant")) => {
            *synced = true;
            None
        }
        (Some("message_update"), Some("assistant")) if !*synced => {
            *synced = true;
            Some(serde_json::json!({
                "type": "message_start",
                "message": event.get("message").cloned().unwrap_or(Value::Null),
            }))
        }
        _ => None,
    }
}

/// Diff a host agent snapshot into the mirror (guest.ts:763-790):
/// unregister missing ids, register new ones, update status on known ones,
/// and patch host timestamps/display names on the refs.
fn apply_agent_snapshots(core: &mut GuestCore, agents: &[Value]) {
    let seen: HashSet<&str> = agents
        .iter()
        .filter_map(|snap| snap.get("id").and_then(Value::as_str))
        .collect();
    // Unregister refs the snapshot no longer carries, with their transcript
    // gates (guest.ts:766-773).
    let departed: Vec<String> = core
        .agents
        .iter()
        .filter(|agent| !seen.contains(agent.id.as_str()))
        .map(|agent| agent.id.clone())
        .collect();
    for id in &departed {
        core.agents.retain(|agent| &agent.id != id);
        core.agent_has_transcript.remove(id);
    }
    for snap in agents {
        let Some(id) = snap.get("id").and_then(Value::as_str) else {
            continue;
        };
        let display_name = snap
            .get("displayName")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_string();
        let status = snap
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("idle")
            .to_string();
        let has_session_file = snap
            .get("hasSessionFile")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let created_at = snap.get("createdAt").and_then(Value::as_f64).unwrap_or(0.0);
        let last_activity = snap
            .get("lastActivity")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        // Known ref: only the status changes (guest.ts:781-783); a new
        // registration carries kind/parentId (guest.ts:784-793).
        if let Some(agent) = core.agents.iter_mut().find(|agent| agent.id == id) {
            agent.status = status;
        } else {
            core.agents.push(AgentRef {
                id: id.to_string(),
                display_name: display_name.clone(),
                kind: snap
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("sub")
                    .to_string(),
                parent_id: snap
                    .get("parentId")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                status,
                has_session_file,
                created_at,
                last_activity,
            });
        }
        // Refs are returned by reference: patch host timestamps and display
        // name directly so hub age/activity columns reflect the host, not
        // local registration (guest.ts:795-803).
        let agent = core
            .agents
            .iter_mut()
            .find(|agent| agent.id == id)
            .expect("registered or updated above");
        agent.created_at = created_at;
        agent.last_activity = last_activity;
        agent.display_name.clone_from(&display_name);
        agent.has_session_file = has_session_file;
        core.agent_has_transcript
            .insert(id.to_string(), has_session_file);
    }
}

/// The TS `#finalizeSnapshot` header rewrite (guest.ts:447-452): the
/// replica gets its own stable id, points `parentSession` at the host's
/// session, and defaults `providerPromptCacheKey` to the host's id (`??`).
fn rewrite_replica_header(host_header: &Value, replica_id: &str) -> Value {
    let mut header = host_header.clone();
    let Some(map) = header.as_object_mut() else {
        return header;
    };
    map.insert("id".to_string(), Value::String(replica_id.to_string()));
    if let Some(host_id @ Value::String(_)) = host_header.get("id") {
        // TS writes both fields unconditionally; `JSON.stringify` then drops
        // the undefined ones — so only present when the host header had an
        // id.
        map.insert("parentSession".to_string(), host_id.clone());
        if host_header.get("providerPromptCacheKey").is_none() {
            map.insert("providerPromptCacheKey".to_string(), host_id.clone());
        }
    }
    header
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        link::format_collab_link, relay_client::connect_relay, relay_server::start_local_relay,
        replication::snapshot_chunks,
    };
    use std::net::SocketAddr;
    use tokio::{
        io,
        net::{TcpListener, TcpStream},
        task::{AbortHandle, JoinHandle},
    };

    const ROOM: &str = "collabguestroom01";
    const FIVE_SECONDS: Duration = Duration::from_secs(5);
    /// Fixed 32-byte room key shared by the scripted host + guest.
    const KEY: [u8; 32] = [7; 32];
    /// Fixed 16-byte write token for full links.
    const TOKEN: [u8; 16] = [0xf0; 16];

    // ── frame golden vectors (TS oracles generated with bun against the
    //    real protocol.ts encoders) ──────────────────────────────────────

    #[test]
    fn hello_frame_goldens() {
        // bun oracle: JSON.stringify of the guest.ts:313-320 literal.
        let writable = CollabFrame::Hello {
            proto: 3,
            name: "kaneli".to_string(),
            write_token: Some("ABAE".to_string()),
        };
        assert_eq!(
            serde_json::to_string(&writable).unwrap(),
            r#"{"t":"hello","proto":3,"name":"kaneli","writeToken":"ABAE"}"#
        );
        let view = CollabFrame::Hello {
            proto: 3,
            name: "kaneli".to_string(),
            write_token: None,
        };
        assert_eq!(
            serde_json::to_string(&view).unwrap(),
            r#"{"t":"hello","proto":3,"name":"kaneli"}"#
        );
        // Non-ASCII passes through unescaped, like JSON.stringify.
        let utf8 = CollabFrame::Hello {
            proto: 3,
            name: "眉 guest".to_string(),
            write_token: None,
        };
        assert_eq!(
            serde_json::to_string(&utf8).unwrap(),
            "{\"t\":\"hello\",\"proto\":3,\"name\":\"\u{7709} guest\"}"
        );
    }

    #[test]
    fn prompt_abort_response_frame_goldens() {
        let plain = CollabFrame::Prompt {
            text: "hello room".to_string(),
            images: Vec::new(),
        };
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            r#"{"t":"prompt","text":"hello room"}"#
        );
        let with_images = CollabFrame::Prompt {
            text: "look".to_string(),
            images: vec![serde_json::json!({
                "type": "image",
                "url": "data:image/png;base64,AAA",
                "mime": "image/png",
            })],
        };
        assert_eq!(
            serde_json::to_string(&with_images).unwrap(),
            r#"{"t":"prompt","text":"look","images":[{"type":"image","url":"data:image/png;base64,AAA","mime":"image/png"}]}"#
        );
        assert_eq!(
            serde_json::to_string(&CollabFrame::Abort {}).unwrap(),
            r#"{"t":"abort"}"#
        );
        let silent = CollabFrame::UiResponse {
            req_id: 7,
            value: None,
        };
        assert_eq!(
            serde_json::to_string(&silent).unwrap(),
            r#"{"t":"ui-response","reqId":7}"#
        );
        let valued = CollabFrame::UiResponse {
            req_id: 7,
            value: Some(Value::String("second option".to_string())),
        };
        assert_eq!(
            serde_json::to_string(&valued).unwrap(),
            r#"{"t":"ui-response","reqId":7,"value":"second option"}"#
        );
    }

    #[test]
    fn host_frame_goldens() {
        let bye = CollabFrame::Bye {
            reason: "host shutting down".to_string(),
        };
        assert_eq!(
            serde_json::to_string(&bye).unwrap(),
            r#"{"t":"bye","reason":"host shutting down"}"#
        );
        let error = CollabFrame::Error {
            message: "prompting is disabled on a read-only link".to_string(),
        };
        assert_eq!(
            serde_json::to_string(&error).unwrap(),
            r#"{"t":"error","message":"prompting is disabled on a read-only link"}"#
        );
        // welcome key order: t, proto, header, state, agents, entryCount,
        // readOnly (host.ts:800-812 literal).
        let welcome = CollabFrame::Welcome {
            proto: 3,
            header: serde_json::json!({
                "id": "host-session", "title": "Fix the flange",
                "timestamp": "2026-10-21T00:00:00.000Z", "cwd": "/tmp",
            }),
            state: serde_json::json!({
                "isStreaming": false, "queuedMessageCount": 0,
                "sessionName": "Fix the flange", "cwd": "/tmp",
                "participants": [{"name": "host", "role": "host"}],
            }),
            agents: vec![serde_json::json!({
                "id": "main", "displayName": "main", "kind": "main",
                "status": "idle", "hasSessionFile": true,
                "createdAt": 1, "lastActivity": 2,
            })],
            entry_count: 2,
            read_only: true,
        };
        assert_eq!(
            serde_json::to_string(&welcome).unwrap(),
            r#"{"t":"welcome","proto":3,"header":{"id":"host-session","title":"Fix the flange","timestamp":"2026-10-21T00:00:00.000Z","cwd":"/tmp"},"state":{"isStreaming":false,"queuedMessageCount":0,"sessionName":"Fix the flange","cwd":"/tmp","participants":[{"name":"host","role":"host"}]},"agents":[{"id":"main","displayName":"main","kind":"main","status":"idle","hasSessionFile":true,"createdAt":1,"lastActivity":2}],"entryCount":2,"readOnly":true}"#
        );
        // A writable welcome omits readOnly entirely (host.ts:811,
        // `canWrite ? undefined : true`).
        let writable = CollabFrame::Welcome {
            proto: 3,
            header: Value::Object(serde_json::Map::new()),
            state: Value::Object(serde_json::Map::new()),
            agents: Vec::new(),
            entry_count: 0,
            read_only: false,
        };
        assert_eq!(
            serde_json::to_string(&writable).unwrap(),
            r#"{"t":"welcome","proto":3,"header":{},"state":{},"agents":[],"entryCount":0}"#
        );
    }

    #[test]
    fn snapshot_chunk_frame_golden() {
        // TS encoder: encodeSnapshotChunk (protocol.ts:143-146).
        let chunk = CollabFrame::SnapshotChunk {
            entries: vec![
                serde_json::json!({"type": "message", "id": 1}),
                serde_json::json!({"type": "message", "id": 2}),
            ],
            is_final: true,
        };
        assert_eq!(
            serde_json::to_string(&chunk).unwrap(),
            r#"{"t":"snapshot-chunk","entries":[{"type":"message","id":1},{"type":"message","id":2}],"final":true}"#
        );
        let empty = CollabFrame::SnapshotChunk {
            entries: Vec::new(),
            is_final: false,
        };
        assert_eq!(
            serde_json::to_string(&empty).unwrap(),
            r#"{"t":"snapshot-chunk","entries":[],"final":false}"#
        );
    }

    #[test]
    fn frames_round_trip_through_parse() {
        let cases = [
            (
                r#"{"t":"hello","proto":3,"name":"k","writeToken":"AB"}"#,
                "hello",
            ),
            (r#"{"t":"prompt","text":"hi"}"#, "prompt"),
            (r#"{"t":"abort"}"#, "abort"),
            (
                r#"{"t":"welcome","proto":3,"header":{},"state":{},"agents":[],"entryCount":4}"#,
                "welcome",
            ),
            (
                r#"{"t":"snapshot-chunk","entries":[],"final":true}"#,
                "snapshot-chunk",
            ),
            (r#"{"t":"entry","entry":{"type":"message"}}"#, "entry"),
            (r#"{"t":"event","event":{"type":"agent_end"}}"#, "event"),
            (r#"{"t":"state","state":{"isStreaming":true}}"#, "state"),
            (
                r#"{"t":"bus","channel":"task:subagent:progress","data":1}"#,
                "bus",
            ),
            (r#"{"t":"agents","agents":[]}"#, "agents"),
            (
                r#"{"t":"ui-request","request":{"reqId":3,"kind":"select","title":"t"}}"#,
                "ui-request",
            ),
            (r#"{"t":"ui-request-end","reqId":3}"#, "ui-request-end"),
            (
                r#"{"t":"transcript","reqId":9,"text":"","newSize":0}"#,
                "transcript",
            ),
            (r#"{"t":"bye","reason":"done"}"#, "bye"),
            (r#"{"t":"error","message":"boom"}"#, "error"),
        ];
        for (case, tag) in cases {
            let parsed: CollabFrame = serde_json::from_str(case).unwrap();
            let round_tripped = serde_json::to_string(&parsed).unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&round_tripped)
                    .unwrap()
                    .get("t")
                    .and_then(Value::as_str),
                Some(tag),
                "{case} lost its tag"
            );
        }
        // Unknown tags are rejected at parse — the driver skips them like
        // the TS default arm (guest.ts:713-715).
        assert!(serde_json::from_str::<CollabFrame>(r#"{"t":"huh"}"#).is_err());
    }

    // ── mirror logic (no transport) ─────────────────────────────────────

    fn test_core() -> GuestCore {
        GuestCore {
            session_id: "replica-test".to_string(),
            read_only: false,
            welcomed: false,
            replica_header: None,
            entries: Vec::new(),
            state: None,
            agents: Vec::new(),
            agent_has_transcript: HashMap::new(),
        }
    }

    #[test]
    fn replica_header_rewrite() {
        let host = serde_json::json!({
            "id": "host-1", "title": "t", "cwd": "/w",
            "providerPromptCacheKey": "keep-me",
        });
        let replica = rewrite_replica_header(&host, "replica-9");
        assert_eq!(replica.get("id"), Some(&Value::String("replica-9".into())));
        assert_eq!(
            replica.get("parentSession"),
            Some(&Value::String("host-1".into()))
        );
        // An existing cache key is preserved, not defaulted
        // (guest.ts:450-451 `??`).
        assert_eq!(
            replica.get("providerPromptCacheKey"),
            Some(&Value::String("keep-me".into()))
        );
        let bare = serde_json::json!({ "id": "host-2", "title": "t" });
        let replica = rewrite_replica_header(&bare, "replica-9");
        assert_eq!(
            replica.get("providerPromptCacheKey"),
            Some(&Value::String("host-2".into()))
        );
        assert_eq!(
            replica.get("parentSession"),
            Some(&Value::String("host-2".into()))
        );
        assert_eq!(replica.get("title"), Some(&Value::String("t".into())));
    }

    #[test]
    fn orphan_delta_guard_synthesizes_message_start() {
        let update = serde_json::json!({
            "type": "message_update",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "par"}]},
        });
        let mut synced = false;
        // The first orphaned update synthesizes the missing start
        // (guest.ts:738-750).
        let synthesized = synthesize_missing_message_start(&mut synced, &update);
        assert!(synced);
        assert_eq!(
            synthesized,
            Some(serde_json::json!({
                "type": "message_start",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "par"}]},
            }))
        );
        // A second update does not re-synthesize.
        assert_eq!(synthesize_missing_message_start(&mut synced, &update), None);
        // A real start latches without synthesis.
        let mut synced = false;
        let start = serde_json::json!({
            "type": "message_start",
            "message": {"role": "assistant"},
        });
        assert_eq!(synthesize_missing_message_start(&mut synced, &start), None);
        assert!(synced);
        // Non-assistant updates never synthesize.
        let mut synced = false;
        let user_update = serde_json::json!({
            "type": "message_update",
            "message": {"role": "user"},
        });
        assert_eq!(
            synthesize_missing_message_start(&mut synced, &user_update),
            None
        );
        assert!(!synced);
    }

    #[test]
    fn pending_snapshot_completes_on_entry_count() {
        let base = PendingSnapshot {
            header: Value::Null,
            state: Value::Null,
            agents: Vec::new(),
            read_only: false,
            entry_count: 2,
            entries: Vec::new(),
            is_resync: false,
        };
        assert!(!base.is_complete());
        let one = PendingSnapshot {
            entries: vec![Value::Null],
            ..base.clone()
        };
        assert!(!one.is_complete());
        let two = PendingSnapshot {
            entries: vec![Value::Null, Value::Null],
            ..base.clone()
        };
        assert!(two.is_complete());
        // An empty snapshot (entryCount 0) is complete on arrival.
        let empty = PendingSnapshot {
            entry_count: 0,
            entries: Vec::new(),
            ..base
        };
        assert!(empty.is_complete());
    }

    #[test]
    // Timestamps are host-authored JSON numbers round-tripped as f64.
    #[allow(clippy::float_cmp)]
    fn agent_snapshots_diff_like_the_ts_registry() {
        let mut core = test_core();
        let round1 = vec![
            serde_json::json!({
                "id": "main", "displayName": "main", "kind": "main",
                "status": "running", "hasSessionFile": true,
                "createdAt": 100.0, "lastActivity": 110.0,
            }),
            serde_json::json!({
                "id": "sub-1", "displayName": "scout", "kind": "sub",
                "parentId": "main", "status": "idle", "hasSessionFile": false,
                "createdAt": 120.0, "lastActivity": 121.0,
            }),
        ];
        apply_agent_snapshots(&mut core, &round1);
        assert_eq!(core.agents.len(), 2);
        assert_eq!(core.agents[0].id, "main");
        assert_eq!(core.agents[1].parent_id.as_deref(), Some("main"));
        assert_eq!(core.agent_has_transcript.get("sub-1"), Some(&false));
        assert_eq!(core.agent_has_transcript.get("main"), Some(&true));
        // Round 2: status + timestamp + display patch on known refs; sub-1
        // unregistered with its gate (guest.ts:766-773).
        let round2 = vec![serde_json::json!({
            "id": "main", "displayName": "renamed", "kind": "main",
            "status": "idle", "hasSessionFile": true,
            "createdAt": 100.0, "lastActivity": 200.0,
        })];
        apply_agent_snapshots(&mut core, &round2);
        assert_eq!(core.agents.len(), 1);
        let main = &core.agents[0];
        assert_eq!(main.status, "idle");
        assert_eq!(main.display_name, "renamed");
        assert_eq!(main.last_activity, 200.0);
        assert_eq!(main.kind, "main", "kind is registration-time only");
        assert!(!core.agent_has_transcript.contains_key("sub-1"));
    }

    #[test]
    fn minted_session_ids_are_uuidv7_shaped() {
        let id = mint_session_id();
        let parts: Vec<&str> = id.split('-').collect();
        assert_eq!(
            parts.iter().map(|part| part.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12]
        );
        let hex: String = id.chars().filter(|c| *c != '-').collect();
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
        // Version 7 nibble and RFC 4122 variant bits.
        assert_eq!(&hex[12..13], "7");
        assert!(matches!(&hex[16..17], "8" | "9" | "a" | "b"));
        assert_ne!(mint_session_id(), mint_session_id());
    }

    // ── e2e scaffolding: scripted fake host over the real relay ─────────

    /// Scripted host behavior, mutated by the test between phases.
    // Test scaffolding: each bool is one scripted knob.
    #[allow(clippy::struct_excessive_bools)]
    struct HostScript {
        /// Session entries the snapshot train delivers (grows as prompts are
        /// answered).
        entries: Vec<Value>,
        /// `welcome.readOnly`.
        read_only: bool,
        /// `hasSessionFile` on the mirrored main agent.
        main_has_transcript: bool,
        /// Targeted pre-welcome error reply to a hello (protocol mismatch).
        hello_error: Option<String>,
        /// Never respond to a hello (welcome-timeout test).
        quiet: bool,
        /// Seal replies with this key instead of the room key (bad-key test).
        wrong_seal_key: Option<Vec<u8>>,
        /// Reply to prompts with entry + event + state.
        prompt_reply: bool,
        /// base64url write tokens seen per hello.
        hello_tokens: Vec<Option<String>>,
        /// Prompt texts received.
        prompts: Vec<String>,
        /// ui-responses received.
        ui_responses: Vec<(u64, Option<Value>)>,
        next_entry_id: u64,
    }

    impl HostScript {
        fn new(entries: Vec<Value>) -> Mutex<Self> {
            Mutex::new(Self {
                entries,
                read_only: false,
                main_has_transcript: true,
                hello_error: None,
                quiet: false,
                wrong_seal_key: None,
                prompt_reply: true,
                hello_tokens: Vec::new(),
                prompts: Vec::new(),
                ui_responses: Vec::new(),
                next_entry_id: 100,
            })
        }

        fn main_agent(&self) -> Value {
            serde_json::json!({
                "id": "main", "displayName": "main", "kind": "main",
                "status": "idle", "hasSessionFile": self.main_has_transcript,
                "createdAt": 1_000.0, "lastActivity": 2_000.0,
            })
        }

        fn header() -> Value {
            serde_json::json!({
                "id": "host-session-1",
                "title": "Collab test",
                "timestamp": "2026-10-21T00:00:00.000Z",
                "cwd": "/tmp/room",
            })
        }

        fn state(is_streaming: bool) -> Value {
            serde_json::json!({
                "isStreaming": is_streaming,
                "queuedMessageCount": 0,
                "sessionName": "Collab test",
                "cwd": "/tmp/room",
                "participants": [{"name": "host", "role": "host"}],
            })
        }
    }

    /// Spawn the scripted host peer: `relay_client` for the transport,
    /// `replication::snapshot_chunks` for the snapshot train.
    async fn spawn_fake_host(
        relay_url: &str,
        key: &'static [u8],
        script: Arc<Mutex<HostScript>>,
    ) -> (RelayHandle, JoinHandle<()>) {
        let (relay, events) = connect_relay(relay_url, ROOM, CollabRole::Host, &fast_relay_opts())
            .await
            .expect("fake host connects");
        let mut events = Box::pin(events);
        // connect_relay returns before the socket lands; the room only
        // exists once the host's upgrade completes, so gate the guest join
        // on the host's Opened.
        while let Some(event) = events.next().await {
            if matches!(event, RelayEvent::Opened) {
                break;
            }
        }
        let task_relay = relay.clone();
        let task = tokio::spawn(async move {
            let relay = task_relay;
            let room_cipher = CollabCipher::new(key);
            while let Some(event) = events.next().await {
                match event {
                    RelayEvent::Binary(data) => {
                        let Some(envelope) = unpack_envelope(&data) else {
                            continue;
                        };
                        let Some(plaintext) = room_cipher.open(&envelope.payload) else {
                            continue;
                        };
                        let Ok(frame) = serde_json::from_slice::<CollabFrame>(&plaintext) else {
                            continue;
                        };
                        // Replies go to the peer that talked to us, sealed
                        // with the wrong key when the script says so.
                        let reply = |frame: &CollabFrame| -> bool {
                            let wrong = script.lock().unwrap().wrong_seal_key.clone();
                            let sealing = CollabCipher::new(wrong.as_deref().unwrap_or(key));
                            let json = serde_json::to_string(frame).expect("frame json");
                            relay.send_binary(pack_envelope(
                                envelope.peer_id,
                                &sealing.seal(json.as_bytes()),
                            ))
                        };
                        match frame {
                            CollabFrame::Hello { write_token, .. } => {
                                if host_on_hello(&script, write_token, &reply) {
                                    send_snapshot(&script, &reply);
                                }
                            }
                            CollabFrame::Prompt { text, .. } => {
                                host_on_prompt(&script, text, &reply);
                            }
                            CollabFrame::UiResponse { req_id, value } => {
                                script.lock().unwrap().ui_responses.push((req_id, value));
                            }
                            _ => {}
                        }
                    }
                    RelayEvent::Closed { will_reconnect, .. } => {
                        if !will_reconnect {
                            break;
                        }
                    }
                    RelayEvent::Opened | RelayEvent::Text(_) => {}
                }
            }
        });
        (relay, task)
    }

    /// Record the hello and answer it. Returns false when the script says
    /// to stay quiet or to fail the join with a pre-welcome error.
    fn host_on_hello(
        script: &Arc<Mutex<HostScript>>,
        write_token: Option<String>,
        reply: &impl Fn(&CollabFrame) -> bool,
    ) -> bool {
        let (error, quiet) = {
            let mut script = script.lock().unwrap();
            script.hello_tokens.push(write_token);
            (script.hello_error.clone(), script.quiet)
        };
        if quiet {
            return false;
        }
        if let Some(message) = error {
            assert!(reply(&CollabFrame::Error { message }));
            return false;
        }
        true
    }

    /// Answer a prompt like the host's live traffic: a message entry
    /// appended to the session, the closing event, settled state.
    fn host_on_prompt(
        script: &Arc<Mutex<HostScript>>,
        text: String,
        reply: &impl Fn(&CollabFrame) -> bool,
    ) {
        let do_reply = {
            let mut script = script.lock().unwrap();
            script.prompts.push(text);
            script.prompt_reply
        };
        if !do_reply {
            return;
        }
        let entry = {
            let mut script = script.lock().unwrap();
            let id = script.next_entry_id;
            script.next_entry_id += 1;
            let entry = serde_json::json!({
                "type": "message",
                "id": id,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "reply"}],
                },
            });
            script.entries.push(entry.clone());
            entry
        };
        assert!(reply(&CollabFrame::Entry { entry }));
        assert!(reply(&CollabFrame::Event {
            event: serde_json::json!({"type": "agent_end"}),
        }));
        assert!(reply(&CollabFrame::State {
            state: HostScript::state(false),
        }));
    }

    /// Send welcome + chunk train for the scripted entries, targeted at the
    /// hello's peer (host.ts:800-824 shape, chunked via the real
    /// replication port).
    fn send_snapshot(script: &Arc<Mutex<HostScript>>, reply: &impl Fn(&CollabFrame) -> bool) {
        let (read_only, entry_jsons) = {
            let script = script.lock().unwrap();
            let jsons: Vec<String> = script
                .entries
                .iter()
                .map(|entry| serde_json::to_string(entry).expect("entry json"))
                .collect();
            (script.read_only, jsons)
        };
        let bytes: Vec<usize> = entry_jsons.iter().map(String::len).collect();
        let agents = vec![script.lock().unwrap().main_agent()];
        assert!(reply(&CollabFrame::Welcome {
            proto: COLLAB_PROTO,
            header: HostScript::header(),
            state: HostScript::state(false),
            agents,
            entry_count: u64::try_from(entry_jsons.len()).unwrap_or(0),
            read_only,
        }));
        for chunk in snapshot_chunks(&entry_jsons, &bytes) {
            let entries: Vec<Value> = chunk
                .entries
                .iter()
                .map(|json| serde_json::from_str(json).expect("entry json"))
                .collect();
            assert!(reply(&CollabFrame::SnapshotChunk {
                entries,
                is_final: chunk.is_final,
            }));
        }
    }

    /// Direct host→guest frame delivery (envelope peer id N).
    fn host_send_frame(host: &RelayHandle, peer: u32, frame: &CollabFrame) {
        let json = serde_json::to_string(frame).expect("frame json");
        let cipher = CollabCipher::new(&KEY);
        assert!(host.send_binary(pack_envelope(peer, &cipher.seal(json.as_bytes()))));
    }

    fn fast_relay_opts() -> RelayOptions {
        RelayOptions {
            backoff_base_ms: 30,
            backoff_max_ms: 250,
            ..RelayOptions::default()
        }
    }

    fn guest_opts() -> GuestOptions {
        GuestOptions {
            name: "guest-user".to_string(),
            welcome_timeout_ms: 5_000,
            snapshot_progress_timeout_ms: 5_000,
            relay: fast_relay_opts(),
            ..GuestOptions::default()
        }
    }

    fn full_link(relay: &crate::relay_server::LocalRelay) -> String {
        format_collab_link(&relay.url, ROOM, &KEY, Some(&TOKEN)).expect("link formats")
    }

    fn view_link(relay: &crate::relay_server::LocalRelay) -> String {
        format_collab_link(&relay.url, ROOM, &KEY, None).expect("link formats")
    }

    async fn next_guest(stream: &mut GuestEventStream) -> GuestEvent {
        tokio::time::timeout(FIVE_SECONDS, stream.next())
            .await
            .expect("guest event timeout")
            .expect("guest stream ended early")
    }

    /// Drain until the next `Synced`, collecting everything else.
    async fn next_synced(
        stream: &mut GuestEventStream,
        collected: &mut Vec<GuestEvent>,
    ) -> GuestSync {
        loop {
            match next_guest(stream).await {
                GuestEvent::Synced { sync } => return sync,
                event => collected.push(event),
            }
        }
    }

    /// Poll until `probe` yields, with a bounded budget.
    async fn wait_for<T>(probe: impl FnMut() -> Option<T>) -> T {
        let deadline = tokio::time::Instant::now() + FIVE_SECONDS;
        let mut probe = probe;
        loop {
            if let Some(value) = probe() {
                return value;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "condition not met within the budget"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn initial_entries() -> Vec<Value> {
        vec![
            serde_json::json!({
                "type": "message", "id": 1,
                "message": {"role": "user", "content": [{"type": "text", "text": "first"}]},
            }),
            serde_json::json!({"type": "model-change", "id": 2, "model": "test-model"}),
        ]
    }

    // ── e2e: the acceptance scenarios ───────────────────────────────────

    /// Full sync on join + prompt round-trip: the sealed prompt opens at the
    /// host, and the host's reply train reaches the guest mirror.
    #[tokio::test]
    async fn e2e_join_sync_then_prompt_round_trip() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        let (host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let (guest, mut stream) = join(&full_link(&relay), guest_opts())
            .await
            .expect("guest joins");
        let mut collected = Vec::new();
        let sync = next_synced(&mut stream, &mut collected).await;
        assert!(!sync.is_resync);
        assert!(!sync.read_only);
        // Replica header: own stable id, parentSession = the host's session
        // (guest.ts:447-452).
        assert_eq!(
            sync.snapshot.replica_header.get("id"),
            Some(&Value::String(guest.session_id()))
        );
        assert_eq!(
            sync.snapshot.replica_header.get("parentSession"),
            Some(&Value::String("host-session-1".into()))
        );
        assert_eq!(
            sync.snapshot.replica_header.get("providerPromptCacheKey"),
            Some(&Value::String("host-session-1".into()))
        );
        assert_eq!(sync.snapshot.entries, script.lock().unwrap().entries);
        assert_eq!(guest.snapshot().expect("synced").entries.len(), 2);
        assert_eq!(sync.agents.len(), 1);
        assert_eq!(sync.agents[0].id, "main");
        assert!(collected.is_empty(), "no stray events on a clean join");

        // The hello carried the write token, base64url of the raw bytes.
        assert_eq!(
            script.lock().unwrap().hello_tokens.clone(),
            vec![Some(encode_base64_url(&TOKEN))]
        );

        // Prompt round-trip: the sealed write reaches the host, the reply
        // train updates the mirror.
        guest
            .send_prompt("hello room")
            .expect("writable guest prompts");
        let mut events = Vec::new();
        loop {
            match next_guest(&mut stream).await {
                GuestEvent::State { state } => {
                    assert_eq!(state.get("isStreaming"), Some(&Value::Bool(false)));
                    break;
                }
                event => events.push(event),
            }
        }
        assert!(
            events
                .iter()
                .any(|event| matches!(event, GuestEvent::Entry { .. })),
            "reply entry reached the guest: {events:?}"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                GuestEvent::Event { event } if event["type"] == "agent_end"
            )),
            "reply event reached the guest: {events:?}"
        );
        {
            let script = script.lock().unwrap();
            assert_eq!(script.prompts, vec!["hello room"]);
            assert_eq!(guest.snapshot().expect("synced").entries.len(), 3);
        }
        // The host's session state is mirrored.
        assert!(guest.state().is_some());

        guest.leave();
        host.close();
        host_task.abort();
        relay.stop();
    }

    /// Read-only guest: view link (no write token), host marks readOnly,
    /// mutating sends are refused locally with nothing on the wire, and a
    /// host-targeted error still surfaces.
    #[tokio::test]
    async fn e2e_view_link_read_only_refusal() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        {
            let mut script = script.lock().unwrap();
            script.read_only = true;
            script.main_has_transcript = false;
        }
        let (host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let (guest, mut stream) = join(&view_link(&relay), guest_opts())
            .await
            .expect("guest joins");
        let sync = next_synced(&mut stream, &mut Vec::new()).await;
        assert!(sync.read_only);
        assert!(guest.is_read_only());
        // The view-link hello carried no token (guest.ts:293-295).
        assert_eq!(script.lock().unwrap().hello_tokens.clone(), vec![None]);

        // Graceful refusal path: local errors, no wire traffic.
        assert_eq!(guest.send_prompt("nope"), Err(GuestSendError::ReadOnly));
        assert_eq!(guest.send_abort(), Err(GuestSendError::ReadOnly));
        assert_eq!(
            guest.send_agent_cmd("chat", "main", Some("nope".to_string())),
            Err(GuestSendError::ReadOnly)
        );
        assert_eq!(
            guest.send_prompt_with_images("nope", vec![serde_json::json!({"mime": "image/png"})]),
            Err(GuestSendError::ReadOnly)
        );
        // fetch_transcript for an agent without a transcript file resolves
        // None without wire traffic (guest.ts:227-231).
        assert!(guest.fetch_transcript("main", 0).await.is_none());
        tokio::time::sleep(Duration::from_millis(100)).await;
        {
            let script = script.lock().unwrap();
            assert!(script.prompts.is_empty(), "no prompt reached the host");
            assert!(script.ui_responses.is_empty());
        }

        // The host's targeted refusal (defensive path) surfaces as an Error
        // event while the session stays up.
        host_send_frame(
            &host,
            1,
            &CollabFrame::Error {
                message: "prompting is disabled on a read-only link".to_string(),
            },
        );
        match next_guest(&mut stream).await {
            GuestEvent::Error { message } => {
                assert_eq!(message, "prompting is disabled on a read-only link");
            }
            event => panic!("expected Error event, got {event:?}"),
        }
        guest.leave();
        host.close();
        host_task.abort();
        relay.stop();
    }

    /// Host `bye` ends the session and the stream.
    #[tokio::test]
    async fn e2e_bye_terminates_guest() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        let (host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let (guest, mut stream) = join(&full_link(&relay), guest_opts())
            .await
            .expect("guest joins");
        next_synced(&mut stream, &mut Vec::new()).await;

        // Graceful host exit: goodbye broadcast, then the socket closes
        // (host.ts:549-556).
        host_send_frame(
            &host,
            0,
            &CollabFrame::Bye {
                reason: "host shutting down".to_string(),
            },
        );
        match next_guest(&mut stream).await {
            GuestEvent::Ended { reason } => assert_eq!(reason, "host shutting down"),
            event => panic!("expected Ended, got {event:?}"),
        }
        // The stream ends after Ended (driver exits, sender dropped).
        let ended = tokio::time::timeout(FIVE_SECONDS, stream.next()).await;
        assert!(
            matches!(ended, Ok(None)),
            "stream should end, got {ended:?}"
        );
        // Sends after the end are refused, not silently dropped.
        assert_eq!(guest.send_prompt("late"), Err(GuestSendError::Ended));
        host.close();
        host_task.abort();
        relay.stop();
    }

    /// Host drop: the relay tears the room down (`room-closed` TEXT + 4001)
    /// and the guest terminates instead of retrying the dead room.
    #[tokio::test]
    async fn e2e_room_closed_terminates_guest() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        let (host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let (_guest, mut stream) = join(&full_link(&relay), guest_opts())
            .await
            .expect("guest joins");
        next_synced(&mut stream, &mut Vec::new()).await;

        // The host peer disappears without a goodbye: relay GCs the room.
        host.close();
        host_task.abort();
        match next_guest(&mut stream).await {
            GuestEvent::Ended { reason } => assert_eq!(reason, "room closed"),
            event => panic!("expected Ended, got {event:?}"),
        }
        let ended = tokio::time::timeout(FIVE_SECONDS, stream.next()).await;
        assert!(
            matches!(ended, Ok(None)),
            "stream should end, got {ended:?}"
        );
        relay.stop();
    }

    /// An abrupt guest-side TCP cut (no WebSocket close frame) classifies as
    /// a transient drop: the guest emits `Reconnecting`, re-hellos on the
    /// next `Opened`, and the fresh welcome resyncs the same replica
    /// session — no rejoin, and live traffic continues.
    #[tokio::test]
    async fn e2e_reconnect_resumes_session() {
        let relay = start_local_relay(0).expect("relay start");
        let port: u16 = relay
            .url
            .rsplit(':')
            .next()
            .expect("relay port")
            .parse()
            .expect("port u16");
        let proxy = spawn_kill_proxy(SocketAddr::from(([127, 0, 0, 1], port))).await;
        let script = Arc::new(HostScript::new(initial_entries()));
        let (host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let link = format_collab_link(
            &format!("ws://127.0.0.1:{}", proxy.addr.port()),
            ROOM,
            &KEY,
            Some(&TOKEN),
        )
        .expect("link formats");
        let (guest, mut stream) = join(&link, guest_opts()).await.expect("guest joins");
        let first = next_synced(&mut stream, &mut Vec::new()).await;
        assert!(!first.is_resync);
        let session_id = guest.session_id();

        // Round-trip once; the host appends the reply entry to its session.
        guest.send_prompt("before drop").expect("prompt");
        loop {
            if matches!(next_guest(&mut stream).await, GuestEvent::State { .. }) {
                break;
            }
        }
        assert_eq!(script.lock().unwrap().entries.len(), 3);

        // Cut the guest's TCP pair abruptly: the transport retries (1006,
        // transient) while the room stays alive behind the still-connected
        // host.
        proxy.cut();
        let mut collected = Vec::new();
        let second = next_synced(&mut stream, &mut collected).await;
        assert!(
            collected
                .iter()
                .any(|event| matches!(event, GuestEvent::Reconnecting { .. })),
            "expected a Reconnecting event, got {collected:?}"
        );
        assert!(second.is_resync);
        assert_eq!(
            guest.session_id(),
            session_id,
            "the replica id is stable across a resync"
        );
        assert_eq!(second.snapshot.entries.len(), 3);
        assert_eq!(second.snapshot.entries, script.lock().unwrap().entries);
        assert_eq!(
            script.lock().unwrap().hello_tokens.len(),
            2,
            "the guest re-introduced itself on the fresh socket"
        );

        // Live again on the resumed session.
        guest.send_prompt("after resume").expect("prompt");
        loop {
            if matches!(next_guest(&mut stream).await, GuestEvent::State { .. }) {
                break;
            }
        }
        assert_eq!(script.lock().unwrap().prompts.len(), 2);

        guest.leave();
        host.close();
        host_task.abort();
        relay.stop();
    }

    /// An empty snapshot (entryCount 0) finalizes on the welcome alone.
    #[tokio::test]
    async fn e2e_empty_snapshot_sync() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(Vec::new()));
        let (_host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let (_guest, mut stream) = join(&full_link(&relay), guest_opts())
            .await
            .expect("guest joins");
        let sync = next_synced(&mut stream, &mut Vec::new()).await;
        assert!(!sync.is_resync);
        assert!(sync.snapshot.entries.is_empty());
        assert_eq!(sync.agents.len(), 1);
        host_task.abort();
        relay.stop();
    }

    /// The host's pre-welcome error reply (protocol mismatch) fails the
    /// join with the host's message (guest.ts:367-383).
    #[tokio::test]
    async fn e2e_protocol_mismatch_fails_join() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        script.lock().unwrap().hello_error =
            Some("protocol mismatch: host speaks v3, guest sent v9".to_string());
        let (_host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let error = join(&full_link(&relay), guest_opts())
            .await
            .expect_err("mismatched join fails");
        assert_eq!(
            error.to_string(),
            "protocol mismatch: host speaks v3, guest sent v9"
        );
        host_task.abort();
        relay.stop();
    }

    /// A host that never welcomes fails the join on the welcome budget
    /// (guest.ts:586).
    #[tokio::test]
    async fn e2e_welcome_timeout_fails_join() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        script.lock().unwrap().quiet = true;
        let (_host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let opts = GuestOptions {
            welcome_timeout_ms: 250,
            ..guest_opts()
        };
        let error = join(&full_link(&relay), opts)
            .await
            .expect_err("quiet host times the join out");
        assert_eq!(
            error.to_string(),
            "timed out waiting for the host's welcome"
        );
        host_task.abort();
        relay.stop();
    }

    /// A host answering with frames sealed under a different key fails the
    /// join fatally (relay-client.ts:645-656, guest branch).
    #[tokio::test]
    async fn e2e_bad_key_fails_join() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        script.lock().unwrap().wrong_seal_key = Some(vec![9_u8; 32]);
        let (_host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let error = join(&full_link(&relay), guest_opts())
            .await
            .expect_err("wrong-key host fails the join");
        assert_eq!(error.to_string(), "bad key or corrupted frame");
        host_task.abort();
        relay.stop();
    }

    /// Host asks round-trip (`ui-request` presented, answered, dismissed)
    /// and `fetch-transcript` resolves from the targeted reply.
    #[tokio::test]
    async fn e2e_ui_request_and_transcript_roundtrip() {
        let relay = start_local_relay(0).expect("relay start");
        let script = Arc::new(HostScript::new(initial_entries()));
        let (host, host_task) = spawn_fake_host(&relay.url, &KEY, Arc::clone(&script)).await;

        let (guest, mut stream) = join(&full_link(&relay), guest_opts())
            .await
            .expect("guest joins");
        next_synced(&mut stream, &mut Vec::new()).await;

        // The host asks; the guest presents it.
        host_send_frame(
            &host,
            1,
            &CollabFrame::UiRequest {
                request: serde_json::json!({
                    "reqId": 5, "kind": "select", "title": "pick one",
                    "options": ["a", "b"],
                }),
            },
        );
        match next_guest(&mut stream).await {
            GuestEvent::UiRequest { request } => {
                assert_eq!(request.get("reqId"), Some(&serde_json::json!(5)));
            }
            event => panic!("expected UiRequest, got {event:?}"),
        }
        let _ = guest.respond_ui_request(5, Some(Value::String("b".to_string())));
        let answered = wait_for(|| {
            let script = script.lock().unwrap();
            (!script.ui_responses.is_empty()).then(|| script.ui_responses.clone())
        })
        .await;
        assert_eq!(answered, vec![(5, Some(Value::String("b".to_string())))]);

        // fetch-transcript: the driver numbers requests from 1; the reply
        // resolves the awaiting caller.
        let transcript_guest = guest.clone();
        let pending =
            tokio::spawn(async move { transcript_guest.fetch_transcript("main", 0).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        host_send_frame(
            &host,
            1,
            &CollabFrame::Transcript {
                req_id: 1,
                text: "{\"x\":1}".to_string(),
                new_size: 7,
                error: None,
            },
        );
        let transcript = pending
            .await
            .expect("fetch task joins")
            .expect("transcript resolves");
        assert_eq!(transcript.text, "{\"x\":1}");
        assert_eq!(transcript.new_size, 7);
        assert_eq!(transcript.error, None);

        // The host settles a second ask elsewhere: dismissed without
        // responding (guest.ts:859-869).
        host_send_frame(
            &host,
            1,
            &CollabFrame::UiRequest {
                request: serde_json::json!({"reqId": 6, "kind": "select", "title": "other"}),
            },
        );
        match next_guest(&mut stream).await {
            GuestEvent::UiRequest { request } => {
                assert_eq!(request.get("reqId"), Some(&serde_json::json!(6)));
            }
            event => panic!("expected UiRequest, got {event:?}"),
        }
        host_send_frame(&host, 1, &CollabFrame::UiRequestEnd { req_id: 6 });
        match next_guest(&mut stream).await {
            GuestEvent::UiRequestEnd { req_id } => assert_eq!(req_id, 6),
            event => panic!("expected UiRequestEnd, got {event:?}"),
        }
        // Ownership is enforced driver-side: the late answer is suppressed.
        let _ = guest.respond_ui_request(6, Some(Value::String("late".to_string())));
        let _ = guest.respond_ui_request(99, None);
        tokio::time::sleep(Duration::from_millis(100)).await;
        let responses = script.lock().unwrap().ui_responses.clone();
        assert_eq!(responses.len(), 1, "dismissed/unknown answers stay silent");

        guest.leave();
        host.close();
        host_task.abort();
        relay.stop();
    }

    /// TCP proxy in front of the relay; [`KillProxy::cut`] aborts the
    /// current forwarding pair with an abrupt TCP close (no WebSocket close
    /// frame), so the guest's transport sees a 1006-style drop and retries.
    struct KillProxy {
        addr: SocketAddr,
        pairs: Arc<Mutex<Vec<AbortHandle>>>,
    }

    impl KillProxy {
        fn cut(&self) {
            for handle in self.pairs.lock().expect("pairs").drain(..) {
                handle.abort();
            }
        }
    }

    async fn spawn_kill_proxy(target: SocketAddr) -> KillProxy {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("proxy binds");
        let addr = listener.local_addr().expect("proxy addr");
        let pairs: Arc<Mutex<Vec<AbortHandle>>> = Arc::new(Mutex::new(Vec::new()));
        let loop_pairs = Arc::clone(&pairs);
        tokio::spawn(async move {
            loop {
                let Ok((client, _)) = listener.accept().await else {
                    break;
                };
                let Ok(server) = TcpStream::connect(target).await else {
                    continue;
                };
                let pairs = Arc::clone(&loop_pairs);
                tokio::spawn(async move {
                    let (mut client_read, mut client_write) = io::split(client);
                    let (mut server_read, mut server_write) = io::split(server);
                    let up = tokio::spawn(async move {
                        io::copy(&mut client_read, &mut server_write).await.ok();
                    });
                    let down = tokio::spawn(async move {
                        io::copy(&mut server_read, &mut client_write).await.ok();
                    });
                    {
                        let mut pending = pairs.lock().expect("pairs");
                        pending.push(up.abort_handle());
                        pending.push(down.abort_handle());
                    }
                    let _ = tokio::join!(up, down);
                });
            }
        });
        KillProxy { addr, pairs }
    }
}
