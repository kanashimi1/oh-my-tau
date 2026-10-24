//! Offline stand-in for the public collab relay (`wss://my.omp.sh`).
//!
//! Rust port of `packages/collab-web/scripts/local-relay.ts` — it speaks the exact relay
//! contract the real clients expect:
//! - `GET /r/<roomId>?role=host|guest` upgrades to a WebSocket (rooms live under a
//!   `[A-Za-z0-9_-]{10,64}` id; anything else that is not a static request is a 404).
//! - The host creates the room; a second host is rejected with close 4009 and a guest joining a
//!   missing room with close 4004 (both after a completed upgrade, so clients classify them).
//! - Host binary frames: envelope peerId 0 broadcasts to every guest, peerId N targets that
//!   guest only — forwarded unchanged either way.
//! - Guest binary frames: the first 4 envelope bytes are rewritten to the sender's peerId, then
//!   forwarded to the host.
//! - TEXT control to the host: `{"t":"peer-joined","peer":N}` / `{"t":"peer-left","peer":N}`.
//! - Host disconnect: TEXT `{"t":"room-closed"}` to every guest, then close 4001, and the room
//!   is garbage-collected.
//!
//! The relay never sees plaintext: payloads stay sealed end to end. [`serve_collab_client`]
//! additionally serves the collab-web static build for every non-`/r/` GET, so one port hosts
//! both the relay and the browser client.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use tokio::{
    io::AsyncWriteExt,
    net::{TcpListener, TcpStream},
    sync::{mpsc, watch},
    task::AbortHandle,
};
use tokio_tungstenite::{accept_async, tungstenite::protocol::CloseFrame, tungstenite::Message};

use crate::protocol::{rewrite_envelope_peer, unpack_envelope};

/// Default port of the TS script (local-relay.ts:22).
pub const DEFAULT_PORT: u16 = 7466;

/// Room ids: `/r/<id>` with 10..=64 url-safe chars (local-relay.ts:20).
fn is_room_id(text: &str) -> bool {
    (10..=64).contains(&text.len())
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// The WS route: `/r/<roomId>` plus `?role=host|guest` (local-relay.ts:52-57).
fn parse_room_request(path: &str, query: &str) -> Option<(String, bool)> {
    let room_id = path.strip_prefix("/r/")?;
    if !is_room_id(room_id) {
        return None;
    }
    for pair in query.split('&') {
        if let Some(role) = pair.strip_prefix("role=") {
            return match role {
                "host" => Some((room_id.to_string(), true)),
                "guest" => Some((room_id.to_string(), false)),
                _ => None,
            };
        }
    }
    None
}

/// Split a request target into path and query, percent-decoding the path.
fn split_target(target: &str) -> (String, String) {
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (target, ""),
    };
    (percent_decode(path), query.to_string())
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or_default();
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The control frames the relay emits. Byte-exact `JSON.stringify` shapes
/// (`{"t":"peer-joined","peer":1}`) — built as literals rather than serde so the wire bytes do
/// not depend on map-ordering features (local-relay.ts:81,109,118).
fn control_frame(kind: &str, peer: Option<u32>) -> Message {
    match peer {
        Some(peer) => Message::text(format!(r#"{{"t":"{kind}","peer":{peer}}}"#)),
        None => Message::text(format!(r#"{{"t":"{kind}"}}"#)),
    }
}

fn close_frame(code: u16, reason: &str) -> Message {
    Message::Close(Some(CloseFrame {
        code: code.into(),
        reason: reason.into(),
    }))
}

/// One side of a room: the forwarding address for a connected socket.
type PeerSender = mpsc::UnboundedSender<Message>;

struct Room {
    /// Connection id of the socket that created the room; a closing socket that does not match
    /// is a rejected second host, and the live room is not its to tear down (local-relay.ts:107).
    host_conn: u64,
    host: PeerSender,
    guests: HashMap<u32, PeerSender>,
    next_peer_id: u32,
}

struct RelayShared {
    rooms: Mutex<HashMap<String, Room>>,
    /// Flips once on [`LocalRelay::stop`]; connection loops close instead of idling.
    shutdown_tx: watch::Sender<bool>,
    static_dir: Option<PathBuf>,
    next_conn: std::sync::atomic::AtomicU64,
    accept_task: Mutex<Option<AbortHandle>>,
}

/// Handle to a running local relay.
///
/// `url` is `ws://localhost:<port>` — append `/r/<roomId>?role=…` to connect
/// (local-relay.ts:40-44).
pub struct LocalRelay {
    pub url: String,
    shared: Arc<RelayShared>,
}

impl LocalRelay {
    /// Close every room and stop the server. Idempotent (local-relay.ts:42-43).
    ///
    /// # Panics
    /// When the shared room table's mutex is poisoned by a panicking connection task.
    pub fn stop(&self) {
        let closure = control_frame("room-closed", None);
        {
            let mut rooms = self.shared.rooms.lock().expect("rooms mutex poisoned");
            for room in rooms.values() {
                for guest in room.guests.values() {
                    let _ = guest.send(closure.clone());
                    let _ = guest.send(close_frame(4001, "room closed"));
                }
                let _ = room.host.send(close_frame(1001, "relay shutting down"));
            }
            rooms.clear();
        }
        let _ = self.shared.shutdown_tx.send(true);
        if let Some(handle) = self.shared.accept_task.lock().expect("accept mutex").take() {
            handle.abort();
        }
    }
}

/// Start the offline relay on `port` (0 picks a free one). Must be called inside a tokio
/// runtime.
///
/// # Errors
/// When the port cannot be bound, or no tokio runtime is running.
pub fn start_local_relay(port: u16) -> Result<LocalRelay> {
    start_relay(port, None)
}

/// Start the relay and additionally serve static files from `dist_dir` for every non-`/r/` GET,
/// so one port hosts the relay and the collab-web browser client. Must be called inside a tokio
/// runtime.
///
/// # Errors
/// When the port cannot be bound, or no tokio runtime is running.
pub fn serve_collab_client(port: u16, dist_dir: &Path) -> Result<LocalRelay> {
    start_relay(port, Some(dist_dir.to_path_buf()))
}

fn start_relay(port: u16, static_dir: Option<PathBuf>) -> Result<LocalRelay> {
    let std_listener = std::net::TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("failed to bind local collab relay on port {port}"))?;
    let addr = std_listener
        .local_addr()
        .context("bound listener has no address")?;
    // from_std requires a nonblocking fd (tokio#7172).
    std_listener
        .set_nonblocking(true)
        .context("failed to set the listener nonblocking")?;
    let listener = tokio::net::TcpListener::from_std(std_listener)
        .context("failed to register listener with tokio")?;
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let shared = Arc::new(RelayShared {
        rooms: Mutex::new(HashMap::new()),
        shutdown_tx,
        static_dir,
        next_conn: std::sync::atomic::AtomicU64::new(1),
        accept_task: Mutex::new(None),
    });
    let handle = tokio::spawn(accept_loop(listener, Arc::clone(&shared), shutdown_rx));
    *shared.accept_task.lock().expect("accept mutex") = Some(handle.abort_handle());
    Ok(LocalRelay {
        url: format!("ws://localhost:{}", addr.port()),
        shared,
    })
}

async fn accept_loop(
    listener: TcpListener,
    shared: Arc<RelayShared>,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let shared = Arc::clone(&shared);
                    tokio::spawn(handle_connection(stream, shared));
                }
                Err(_) => break,
            },
        }
    }
}

async fn handle_connection(mut stream: TcpStream, shared: Arc<RelayShared>) {
    let Some(request_line) = peek_request_line(&mut stream).await else {
        return;
    };
    let Some((method, target)) = parse_request_line(&request_line) else {
        return;
    };
    let (path, query) = split_target(&target);
    if let Some((room_id, is_host)) = parse_room_request(&path, &query) {
        handle_relay_socket(stream, shared, room_id, is_host).await;
        return;
    }
    if method == "GET" {
        if let Some(dist_dir) = shared.static_dir.clone() {
            serve_static(stream, &dist_dir, &path).await;
            return;
        }
    }
    let _ = write_http_response(
        &mut stream,
        404,
        "Not Found",
        "text/plain; charset=utf-8",
        b"not found",
    )
    .await;
    graceful_close(&mut stream).await;
}

/// Read just the request line without consuming it, so a WS route can hand the untouched stream
/// to tungstenite's handshake.
async fn peek_request_line(stream: &mut TcpStream) -> Option<String> {
    let mut buf = [0u8; 1024];
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let n = stream.peek(&mut buf).await.ok()?;
            if let Some(pos) = buf[..n].iter().position(|&b| b == b'\n') {
                let line = &buf[..pos];
                return Some(String::from_utf8_lossy(line).trim_end().to_string());
            }
            if n == buf.len() {
                return None; // absurdly long request line
            }
        }
    })
    .await
    .ok()
    .flatten()
}

fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    Some((method, target))
}

async fn handle_relay_socket(
    stream: TcpStream,
    shared: Arc<RelayShared>,
    room_id: String,
    is_host: bool,
) {
    // The upgrade completes before any room decision: a rejected host/guest is closed with its
    // relay code after the handshake, exactly like the Bun open handlers (local-relay.ts:63-77).
    let Ok(ws) = accept_async(stream).await else {
        return; // tungstenite already answered (e.g. non-upgrade request)
    };
    let (mut sink, mut stream) = ws.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
    let conn = shared
        .next_conn
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut peer_id = 0u32;

    // ── open handler (local-relay.ts:63-82) ─────────────────────────────────────────
    let join = {
        let mut rooms = shared.rooms.lock().expect("rooms mutex poisoned");
        if is_host {
            if rooms.contains_key(&room_id) {
                Join::Rejected(4009, "a host is already connected for this room")
            } else {
                rooms.insert(
                    room_id.clone(),
                    Room {
                        host_conn: conn,
                        host: tx.clone(),
                        guests: HashMap::new(),
                        next_peer_id: 1,
                    },
                );
                Join::Host
            }
        } else {
            match rooms.get_mut(&room_id) {
                Some(room) => {
                    let guest_peer_id = room.next_peer_id;
                    room.next_peer_id += 1;
                    room.guests.insert(guest_peer_id, tx.clone());
                    let _ = room
                        .host
                        .send(control_frame("peer-joined", Some(guest_peer_id)));
                    Join::Guest(guest_peer_id)
                }
                None => Join::Rejected(4004, "no such room"),
            }
        }
    };
    match join {
        Join::Rejected(code, reason) => {
            let _ = sink.send(close_frame(code, reason)).await;
            return;
        }
        Join::Guest(guest_peer_id) => peer_id = guest_peer_id,
        Join::Host => {}
    }

    // subscribe() only sees changes after it; a stop that already happened is caught by the
    // rooms table being empty, which closes this socket on its first relay access anyway.
    let mut shutdown = shared.shutdown_tx.subscribe();
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                // stop() routes the room teardown (room-closed TEXT + close 4001/1001) through
                // the room senders before signalling; deliver whatever already queued, and give
                // sockets the teardown never reached (strays) their own shutdown close.
                let mut saw_close = false;
                while let Ok(message) = rx.try_recv() {
                    saw_close |= matches!(message, Message::Close(_));
                    if sink.send(message).await.is_err() {
                        break;
                    }
                }
                if !saw_close {
                    let _ = sink.send(close_frame(1001, "relay shutting down")).await;
                }
                break;
            }
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Binary(data))) => {
                    relay_binary(&shared, &room_id, is_host, peer_id, &data);
                }
                // Clients never send TEXT (local-relay.ts:84); the protocol layer answers
                // pings on the next read.
                Some(Ok(Message::Text(_) | Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {}
                Some(Ok(Message::Close(_)) | Err(_)) | None => break,
            },
            outgoing = rx.recv() => match outgoing {
                Some(message) => {
                    let is_close = matches!(message, Message::Close(_));
                    if sink.send(message).await.is_err() || is_close {
                        break;
                    }
                }
                None => break,
            },
        }
    }
    drop(sink);
    // ── close handler (local-relay.ts:101-120) ──────────────────────────────────────
    relay_close(&shared, &room_id, is_host, conn, peer_id);
}

/// Outcome of a room join computed under the room-table lock, before any await.
enum Join {
    /// Guest: assigned peer id.
    Guest(u32),
    /// Closed right after the completed upgrade, with its relay code.
    Rejected(u16, &'static str),
    /// Host: created the room.
    Host,
}

/// Host broadcast/target forwarding and guest peer-rewrite (local-relay.ts:83-100). Frames the
/// TS relay ignores (short guest frames, malformed envelopes, unknown rooms) are ignored here.
fn relay_binary(shared: &RelayShared, room_id: &str, is_host: bool, peer_id: u32, data: &[u8]) {
    if is_host {
        let Some(envelope) = unpack_envelope(data) else {
            return;
        };
        let targets: Vec<PeerSender> = {
            let rooms = shared.rooms.lock().expect("rooms mutex poisoned");
            let Some(room) = rooms.get(room_id) else {
                return;
            };
            if envelope.peer_id == 0 {
                room.guests.values().cloned().collect()
            } else {
                room.guests
                    .get(&envelope.peer_id)
                    .cloned()
                    .into_iter()
                    .collect()
            }
        };
        for target in targets {
            let _ = target.send(Message::binary(data.to_vec()));
        }
    } else {
        if data.len() < 4 {
            return;
        }
        let mut forwarded = data.to_vec();
        rewrite_envelope_peer(&mut forwarded, peer_id);
        let host = {
            let rooms = shared.rooms.lock().expect("rooms mutex poisoned");
            rooms.get(room_id).map(|room| room.host.clone())
        };
        if let Some(host) = host {
            let _ = host.send(Message::binary(forwarded));
        }
    }
}

fn relay_close(shared: &RelayShared, room_id: &str, is_host: bool, conn: u64, peer_id: u32) {
    let mut rooms = shared.rooms.lock().expect("rooms mutex poisoned");
    let Some(room) = rooms.get_mut(room_id) else {
        return;
    };
    if is_host {
        // Rejected second host: the live room is not ours to tear down.
        if room.host_conn != conn {
            return;
        }
        let room = rooms.remove(room_id).expect("room checked above");
        let closure = control_frame("room-closed", None);
        for guest in room.guests.values() {
            let _ = guest.send(closure.clone());
            let _ = guest.send(close_frame(4001, "room closed"));
        }
    } else if room.guests.remove(&peer_id).is_some() {
        let _ = room.host.send(control_frame("peer-left", Some(peer_id)));
    }
}

// ─── static file serving (the serve_collab_client extension) ─────────────────────────

/// Mime table for the collab-web build, following the TS-relay practice
/// (html/js/css/svg/png/ico/webmanifest/woff2).
fn mime_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "webmanifest" => "application/manifest+json",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

/// Map a request path to a file below `dist_dir`: `""`/`"/"` and trailing-slash directories map
/// to `index.html`; any `..` segment is rejected.
fn static_relative_path(path: &str) -> Option<String> {
    if !path.starts_with('/') {
        return None;
    }
    let mut parts: Vec<&str> = path
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if parts.contains(&"..") {
        return None;
    }
    if parts.is_empty() || path.ends_with('/') {
        parts.push("index.html");
    }
    Some(parts.join("/"))
}

async fn serve_static(mut stream: TcpStream, dist_dir: &Path, path: &str) {
    let Some(relative) = static_relative_path(path) else {
        let _ = write_http_response(
            &mut stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"not found",
        )
        .await;
        graceful_close(&mut stream).await;
        return;
    };
    match tokio::fs::read(dist_dir.join(&relative)).await {
        Ok(body) => {
            let _ = write_http_response(&mut stream, 200, "OK", mime_type(&relative), &body).await;
        }
        Err(_) => {
            let _ = write_http_response(
                &mut stream,
                404,
                "Not Found",
                "text/plain; charset=utf-8",
                b"not found",
            )
            .await;
        }
    }
    graceful_close(&mut stream).await;
}

/// Close a plain-HTTP connection cleanly: FIN first, then drain whatever the peer still has in
/// flight, so the socket ends with FIN/ACK instead of a RST (the request head beyond the peeked
/// line is never consumed by the routing fast path).
async fn graceful_close(stream: &mut TcpStream) {
    use tokio::io::AsyncReadExt;
    let _ = stream.shutdown().await;
    let mut drain = [0u8; 1024];
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            match stream.read(&mut drain).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    })
    .await;
}

async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::pack_envelope;
    use crate::relay_client::{connect_relay, CollabRole, RelayEvent, RelayHandle, RelayOptions};
    use std::time::Duration;

    /// 22 chars of `[A-Za-z0-9_-]` — the shape of a real base64url room id.
    const TEST_ROOM: &str = "collabtestroom0123456";
    const FIVE_SECONDS: Duration = Duration::from_secs(5);

    fn fast_opts() -> RelayOptions {
        RelayOptions {
            backoff_base_ms: 30,
            backoff_max_ms: 250,
            ..RelayOptions::default()
        }
    }

    /// Guest opts that never get far enough to retry, so teardown tests stay quiet.
    fn slow_opts() -> RelayOptions {
        RelayOptions {
            backoff_base_ms: 60_000,
            backoff_max_ms: 60_000,
            ..RelayOptions::default()
        }
    }

    async fn next_event<S>(events: &mut S) -> RelayEvent
    where
        S: futures::Stream<Item = RelayEvent> + Unpin,
    {
        tokio::time::timeout(FIVE_SECONDS, events.next())
            .await
            .expect("timed out waiting for a relay event")
            .expect("relay event stream ended")
    }

    async fn expect_opened<S>(events: &mut S)
    where
        S: futures::Stream<Item = RelayEvent> + Unpin,
    {
        assert_eq!(next_event(events).await, RelayEvent::Opened);
    }

    async fn drain_until<S>(
        events: &mut S,
        wanted: Duration,
        pred: impl Fn(&RelayEvent) -> bool,
    ) -> RelayEvent
    where
        S: futures::Stream<Item = RelayEvent> + Unpin,
    {
        let deadline = tokio::time::Instant::now() + wanted;
        loop {
            let event = tokio::time::timeout_at(deadline, next_event(events))
                .await
                .expect("timed out draining relay events");
            if pred(&event) {
                return event;
            }
        }
    }

    async fn expect_no_binary<S>(events: &mut S)
    where
        S: futures::Stream<Item = RelayEvent> + Unpin,
    {
        let leaked = tokio::time::timeout(Duration::from_millis(200), async {
            loop {
                if let RelayEvent::Binary(_) = next_event(events).await {
                    return true;
                }
            }
        })
        .await;
        assert!(
            leaked.is_err(),
            "guest received a frame that was not addressed to it"
        );
    }

    async fn connect(
        url: &str,
        room: &str,
        role: CollabRole,
        opts: &RelayOptions,
    ) -> (RelayHandle, impl futures::Stream<Item = RelayEvent> + Unpin) {
        connect_relay(url, room, role, opts)
            .await
            .expect("connect_relay failed")
    }

    #[tokio::test]
    async fn request_parsing() {
        assert_eq!(
            parse_room_request("/r/collabtestroom0123456", "role=host"),
            Some(("collabtestroom0123456".to_string(), true))
        );
        assert_eq!(
            parse_room_request("/r/collabtestroom0123456", "x=1&role=guest"),
            Some(("collabtestroom0123456".to_string(), false))
        );
        // Wrong role, missing role, short id, bad chars, wrong path.
        assert_eq!(
            parse_room_request("/r/collabtestroom0123456", "role=chief"),
            None
        );
        assert_eq!(parse_room_request("/r/collabtestroom0123456", ""), None);
        assert_eq!(parse_room_request("/r/short", "role=host"), None);
        assert_eq!(
            parse_room_request("/r/has.dot0123456789", "role=host"),
            None
        );
        assert_eq!(
            parse_room_request("/room/collabtestroom0123456", "role=host"),
            None
        );
    }

    #[tokio::test]
    async fn traversal_rejected_and_index_mapped() {
        assert_eq!(static_relative_path("/"), Some("index.html".to_string()));
        assert_eq!(static_relative_path(""), None);
        assert_eq!(static_relative_path("/a/../../etc/passwd"), None);
        assert_eq!(
            static_relative_path("/assets/app.js"),
            Some("assets/app.js".to_string())
        );
        assert_eq!(
            static_relative_path("/sub/"),
            Some("sub/index.html".to_string())
        );
    }

    #[tokio::test]
    async fn e2e_full_relay_contract() {
        let relay = start_local_relay(0).expect("relay start");
        assert!(relay.url.starts_with("ws://localhost:"));

        // ── host creates the room ──
        let (host, mut host_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Host, &fast_opts()).await;
        expect_opened(&mut host_events).await;

        // ── second host → upgrade completes, then 4009, terminal (the TS onopen fires on the
        // upgrade response, before the close frame is read — relay-client.ts:513) ──
        let (host2, mut host2_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Host, &fast_opts()).await;
        expect_opened(&mut host2_events).await;
        assert_eq!(
            next_event(&mut host2_events).await,
            RelayEvent::Closed {
                code: 4009,
                reason: "a host is already connected for this room".to_string(),
                will_reconnect: false,
            }
        );
        drop(host2);

        // ── guests join; host sees peer-joined TEXT ──
        let (guest1, mut guest1_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Guest, &slow_opts()).await;
        expect_opened(&mut guest1_events).await;
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Text(r#"{"t":"peer-joined","peer":1}"#.to_string())
        );
        let (guest2, mut guest2_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Guest, &slow_opts()).await;
        expect_opened(&mut guest2_events).await;
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Text(r#"{"t":"peer-joined","peer":2}"#.to_string())
        );

        // ── host broadcast (peer 0) reaches every guest unchanged ──
        let broadcast = pack_envelope(0, b"broadcast-payload");
        assert!(host.send_binary(broadcast.clone()));
        for guest_events in [&mut guest1_events, &mut guest2_events] {
            assert_eq!(
                next_event(guest_events).await,
                RelayEvent::Binary(broadcast.clone())
            );
        }

        // ── targeted peer 2 reaches only guest 2 ──
        let targeted = pack_envelope(2, b"only-for-guest-2");
        assert!(host.send_binary(targeted.clone()));
        assert_eq!(
            next_event(&mut guest2_events).await,
            RelayEvent::Binary(targeted.clone())
        );
        expect_no_binary(&mut guest1_events).await;

        // ── guest frame is rewritten to its peer id and reaches the host ──
        let guest_frame = pack_envelope(0, b"from-guest-1");
        assert!(guest1.send_binary(guest_frame));
        let host_received = drain_until(&mut host_events, FIVE_SECONDS, |event| {
            matches!(event, RelayEvent::Binary(_))
        })
        .await;
        let RelayEvent::Binary(bytes) = host_received else {
            unreachable!("drain_until matched a Binary")
        };
        let envelope = crate::protocol::unpack_envelope(&bytes).expect("host received an envelope");
        assert_eq!(envelope.peer_id, 1, "relay rewrote the guest's peer id");
        assert_eq!(envelope.payload, b"from-guest-1");

        // ── guest 2 leaves; host sees peer-left ──
        guest2.close();
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Text(r#"{"t":"peer-left","peer":2}"#.to_string())
        );
        drop(guest2);

        // ── host disconnect: room-closed TEXT then close 4001 to the guest ──
        host.close();
        assert_eq!(
            next_event(&mut guest1_events).await,
            RelayEvent::Text(r#"{"t":"room-closed"}"#.to_string())
        );
        assert_eq!(
            next_event(&mut guest1_events).await,
            RelayEvent::Closed {
                code: 4001,
                reason: "room closed".to_string(),
                will_reconnect: true,
            }
        );
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Closed {
                code: 1000,
                reason: "closed".to_string(),
                will_reconnect: false,
            }
        );
        drop(host);
        // Silence the guest's retry loop for the rest of the test.
        guest1.close();

        // ── missing room → upgrade completes, then 4004, terminal ──
        let (late, mut late_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Guest, &slow_opts()).await;
        expect_opened(&mut late_events).await;
        assert_eq!(
            next_event(&mut late_events).await,
            RelayEvent::Closed {
                code: 4004,
                reason: "no such room".to_string(),
                will_reconnect: false,
            }
        );
        drop(late);

        relay.stop();
    }

    #[tokio::test]
    async fn e2e_guest_survives_room_recreation() {
        let relay = start_local_relay(0).expect("relay start");

        let (host, mut host_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Host, &fast_opts()).await;
        expect_opened(&mut host_events).await;
        let (guest, mut guest_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Guest, &fast_opts()).await;
        expect_opened(&mut guest_events).await;
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Text(r#"{"t":"peer-joined","peer":1}"#.to_string())
        );

        // Host leaves; the room dies; the guest gets 4001 and keeps retrying.
        host.close();
        assert_eq!(
            next_event(&mut guest_events).await,
            RelayEvent::Text(r#"{"t":"room-closed"}"#.to_string())
        );
        let closed = next_event(&mut guest_events).await;
        let RelayEvent::Closed {
            code: 4001,
            will_reconnect: true,
            ..
        } = closed
        else {
            panic!("expected a retrying room-closed event, got {closed:?}")
        };
        drop(host);

        // A new host recreates the room; the guest's retry lands in it.
        let (host2, mut host2_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Host, &fast_opts()).await;
        expect_opened(&mut host2_events).await;
        let reopened = drain_until(&mut guest_events, FIVE_SECONDS, |event| {
            matches!(event, RelayEvent::Opened)
        })
        .await;
        assert_eq!(reopened, RelayEvent::Opened);
        assert_eq!(
            next_event(&mut host2_events).await,
            RelayEvent::Text(r#"{"t":"peer-joined","peer":1}"#.to_string()),
            "the relay reissued peer ids from 1 for the recreated room"
        );

        guest.close();
        host2.close();
        relay.stop();
    }

    #[tokio::test]
    async fn e2e_relay_stop_tears_down_rooms() {
        let relay = start_local_relay(0).expect("relay start");
        let (host, mut host_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Host, &fast_opts()).await;
        expect_opened(&mut host_events).await;
        let (guest, mut guest_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Guest, &fast_opts()).await;
        expect_opened(&mut guest_events).await;
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Text(r#"{"t":"peer-joined","peer":1}"#.to_string())
        );

        relay.stop();

        assert_eq!(
            next_event(&mut guest_events).await,
            RelayEvent::Text(r#"{"t":"room-closed"}"#.to_string())
        );
        assert_eq!(
            next_event(&mut guest_events).await,
            RelayEvent::Closed {
                code: 4001,
                reason: "room closed".to_string(),
                will_reconnect: true,
            }
        );
        let host_closed = next_event(&mut host_events).await;
        let RelayEvent::Closed {
            code: 1001,
            will_reconnect: true,
            ..
        } = host_closed
        else {
            panic!("expected a retrying relay-shutdown event, got {host_closed:?}")
        };
        // The relay is gone, so the next retry fails to connect: 1006, still retrying.
        let failed = drain_until(&mut host_events, FIVE_SECONDS, |event| {
            matches!(event, RelayEvent::Closed { code: 1006, .. })
        })
        .await;
        let RelayEvent::Closed {
            reason,
            will_reconnect: true,
            ..
        } = failed
        else {
            unreachable!("drain_until matched a 1006 close")
        };
        assert_eq!(reason, "connection lost (code 1006)");

        host.close();
        guest.close();
    }

    #[tokio::test]
    async fn e2e_heartbeat_keeps_idle_sockets_alive() {
        let relay = start_local_relay(0).expect("relay start");
        // Pings every 25 ms, drop after 80 ms of silence: with working pong handling the socket
        // survives many cycles; if pongs stopped counting as "heard", it would die at ~80 ms.
        let opts = RelayOptions {
            ping_interval_ms: 25,
            silence_limit_ms: 80,
            ..RelayOptions::default()
        };
        let (host, mut host_events) = connect(&relay.url, TEST_ROOM, CollabRole::Host, &opts).await;
        expect_opened(&mut host_events).await;
        let (guest, mut guest_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Guest, &slow_opts()).await;
        expect_opened(&mut guest_events).await;
        assert_eq!(
            next_event(&mut host_events).await,
            RelayEvent::Text(r#"{"t":"peer-joined","peer":1}"#.to_string())
        );

        // Idle past several silence limits, then prove the session is still the same one.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(host.send_binary(pack_envelope(0, b"still-alive")));
        assert_eq!(
            next_event(&mut guest_events).await,
            RelayEvent::Binary(pack_envelope(0, b"still-alive"))
        );

        host.close();
        guest.close();
        relay.stop();
    }

    #[tokio::test]
    async fn e2e_static_files_served_alongside_relay() {
        let dist = std::env::temp_dir().join(format!("relaycheck-dist-{}", std::process::id()));
        std::fs::create_dir_all(dist.join("assets")).expect("create dist dir");
        std::fs::write(dist.join("index.html"), b"<html>collab</html>").expect("write index");
        std::fs::write(dist.join("assets/app.js"), b"console.log(1)").expect("write js");
        std::fs::write(dist.join("favicon.ico"), b"ico").expect("write ico");

        let relay = serve_collab_client(0, &dist).expect("relay start");
        let mut socket = TcpStream::connect(relay.url.trim_start_matches("ws://"))
            .await
            .expect("connect");

        let (status, content_type, body) = http_get(&mut socket, "/").await;
        assert_eq!(status, 200);
        assert_eq!(content_type, "text/html; charset=utf-8");
        assert_eq!(body, b"<html>collab</html>");

        let mut socket = TcpStream::connect(relay.url.trim_start_matches("ws://"))
            .await
            .expect("connect");
        let (status, content_type, body) = http_get(&mut socket, "/assets/app.js").await;
        assert_eq!(status, 200);
        assert_eq!(content_type, "text/javascript; charset=utf-8");
        assert_eq!(body, b"console.log(1)");

        for path in ["/missing.js", "/../etc/passwd", "/r/short"] {
            let mut socket = TcpStream::connect(relay.url.trim_start_matches("ws://"))
                .await
                .expect("connect");
            let (status, _, body) = http_get(&mut socket, path).await;
            assert_eq!(status, 404, "path {path}");
            assert_eq!(body, b"not found");
        }

        // The relay still works next to the static handler.
        let (host, mut host_events) =
            connect(&relay.url, TEST_ROOM, CollabRole::Host, &fast_opts()).await;
        expect_opened(&mut host_events).await;
        host.close();

        relay.stop();
        let _ = std::fs::remove_dir_all(&dist);
    }

    /// Minimal HTTP/1.1 GET against the raw socket (Connection: close responses).
    async fn http_get(socket: &mut TcpStream, path: &str) -> (u16, String, Vec<u8>) {
        use tokio::io::AsyncReadExt;
        socket
            .write_all(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes())
            .await
            .expect("write request");
        let mut buf = Vec::new();
        tokio::time::timeout(FIVE_SECONDS, socket.read_to_end(&mut buf))
            .await
            .expect("read response")
            .expect("read response body");
        let text = String::from_utf8_lossy(&buf);
        let status_line = text.lines().next().unwrap_or_default();
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let content_type = text
            .lines()
            .find_map(|line| line.strip_prefix("Content-Type: "))
            .unwrap_or_default()
            .to_string();
        let body = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|pos| buf[pos + 4..].to_vec())
            .unwrap_or_default();
        (status, content_type, body)
    }
}
