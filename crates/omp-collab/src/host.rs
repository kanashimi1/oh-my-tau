//! Host side of a collab live session.
//!
//! Rust port of `vendor-src/oh-my-pi/packages/coding-agent/src/collab/host.ts`
//! (1262 lines). The host taps the local omp session (via [`CollabHostBackend`])
//! and mirrors entries/events/state to guests through the relay; guests prompt
//! and abort through the host, which injects them into the local session as user
//! turns.
//!
//! Wire formats are byte-identical to the TS source.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::crypto::CollabCipher;
use crate::link::{format_collab_link, format_collab_web_link, parse_collab_link};
use crate::protocol::{generate_room_id, pack_envelope, unpack_envelope};
use crate::registry::{
    publish_collab_host, CollabAccess, CollabHostModel, CollabHostPublication,
    CollabHostRegistrySource, CollabHostSnapshot, CollabPublishOptions,
};
use crate::relay_client::{connect_relay, CollabRole, RelayEvent, RelayHandle, RelayOptions};
use crate::replication::{
    oversized_entry_notice, serialize_replicated_event, snapshot_chunks,
    COLLAB_ENTRY_OMITTED_CUSTOM_TYPE,
};

/// Wire protocol version (`wire/src/index.ts:401`).
pub const COLLAB_PROTO: u64 = 3;

/// Relay connect timeout (host.ts:123).
const CONNECT_TIMEOUT_MS: u64 = 15_000;
/// Grace between the goodbye and closing the relay socket; see [`CollabHost::stop`].
const BYE_DRAIN: std::time::Duration = std::time::Duration::from_millis(300);

/// Events that change the footer state guests render (host.ts:69-78).
const STATE_TRIGGER_EVENTS: &[&str] = &[
    "agent_start",
    "agent_end",
    "message_end",
    "tool_execution_end",
    "thinking_level_changed",
    "model_changed",
    "advisor_cost_changed",
    "auto_compaction_end",
];

/// The local omp session a host mirrors.
pub trait CollabHostBackend: Send + Sync {
    fn session_id(&self) -> String;
    fn session_name(&self) -> Option<String> {
        None
    }
    fn cwd(&self) -> String;
    fn model(&self) -> Option<(String, String)> {
        None
    }
    fn is_streaming(&self) -> bool {
        false
    }
    fn is_aborting(&self) -> bool {
        false
    }
    fn queued_message_count(&self) -> i64 {
        0
    }
    fn context_usage(&self) -> (u64, u64) {
        (0, 0)
    }
    fn thinking_level(&self) -> Option<String> {
        None
    }
    fn snapshot_header(&self) -> Value {
        Value::Null
    }
    fn snapshot_entries(&self) -> Vec<String> {
        Vec::new()
    }
    fn inject_guest_prompt(&self, _text: &str, _images: Vec<Value>) -> Result<()> {
        Err(anyhow!("guest prompts not supported"))
    }
    fn abort_turn(&self) -> Result<()> {
        Err(anyhow!("guest abort not supported"))
    }
}

/// Options passed to [`CollabHost::new`].
pub struct CollabHostOptions {
    pub instance_id: String,
    pub generation: u64,
    pub access: CollabAccess,
    pub guest_actions_ready: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub on_ended: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for CollabHostOptions {
    fn default() -> Self {
        Self {
            instance_id: String::new(),
            generation: 1,
            access: CollabAccess::Control,
            guest_actions_ready: None,
            on_ended: None,
        }
    }
}

#[derive(Debug, Clone)]
struct Peer {
    name: String,
    can_write: bool,
}

struct HostInner {
    backend: Arc<dyn CollabHostBackend>,
    instance_id: String,
    generation: u64,
    access: CollabAccess,
    guest_actions_ready: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    on_ended: Option<Arc<dyn Fn() + Send + Sync>>,
    session_id: String,
    handle: StdMutex<Option<RelayHandle>>,
    peers: StdMutex<std::collections::HashMap<u32, Peer>>,
    last_state_json: StdMutex<Option<String>>,
    publication: StdMutex<Option<CollabHostPublication>>,
    full_link: StdMutex<Option<String>>,
    web_link: StdMutex<Option<String>>,
    view_link: StdMutex<Option<String>>,
    web_view_link: StdMutex<Option<String>>,
    relay_connected: std::sync::atomic::AtomicBool,
    stopped: std::sync::atomic::AtomicBool,
    stopping: std::sync::atomic::AtomicBool,
    abort_start: StdMutex<Option<mpsc::Sender<()>>>,
    stop_done: StdMutex<Option<mpsc::Receiver<()>>>,
}

/// The host peer for one collab room.
pub struct CollabHost {
    inner: Arc<HostInner>,
}

impl CollabHost {
    pub fn new(backend: Arc<dyn CollabHostBackend>, options: CollabHostOptions) -> Self {
        assert!(
            !options.instance_id.is_empty(),
            "collab host needs an instance id"
        );
        let session_id = backend.session_id();
        let inner = Arc::new(HostInner {
            backend,
            instance_id: options.instance_id,
            generation: options.generation,
            access: options.access,
            guest_actions_ready: options.guest_actions_ready,
            on_ended: options.on_ended,
            session_id,
            handle: StdMutex::new(None),
            peers: StdMutex::new(std::collections::HashMap::new()),
            last_state_json: StdMutex::new(None),
            publication: StdMutex::new(None),
            full_link: StdMutex::new(None),
            web_link: StdMutex::new(None),
            view_link: StdMutex::new(None),
            web_view_link: StdMutex::new(None),
            relay_connected: std::sync::atomic::AtomicBool::new(false),
            stopping: std::sync::atomic::AtomicBool::new(false),
            stopped: std::sync::atomic::AtomicBool::new(false),
            abort_start: StdMutex::new(None),
            stop_done: StdMutex::new(None),
        });
        Self { inner }
    }

    #[must_use]
    pub fn instance_id(&self) -> &str {
        &self.inner.instance_id
    }
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.inner.generation
    }
    #[must_use]
    pub fn access(&self) -> CollabAccess {
        self.inner.access
    }
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }
    #[must_use]
    pub fn stopped(&self) -> bool {
        self.inner.stopped.load(std::sync::atomic::Ordering::SeqCst)
    }
    #[must_use]
    pub fn ending(&self) -> bool {
        self.inner
            .stopping
            .load(std::sync::atomic::Ordering::SeqCst)
            || self.inner.stopped.load(std::sync::atomic::Ordering::SeqCst)
    }
    #[must_use]
    pub fn relay_connected(&self) -> bool {
        self.inner
            .relay_connected
            .load(std::sync::atomic::Ordering::SeqCst)
    }
    #[must_use]
    pub fn participant_count(&self) -> usize {
        self.inner.peers.lock().unwrap().len()
    }

    /// Live links for the current (or last) room — `None` before start / after stop.
    #[must_use]
    pub fn full_link(&self) -> Option<String> {
        self.inner.full_link.lock().unwrap().clone()
    }
    #[must_use]
    pub fn web_link(&self) -> Option<String> {
        self.inner.web_link.lock().unwrap().clone()
    }
    #[must_use]
    pub fn view_link(&self) -> Option<String> {
        self.inner.view_link.lock().unwrap().clone()
    }
    #[must_use]
    pub fn web_view_link(&self) -> Option<String> {
        self.inner.web_view_link.lock().unwrap().clone()
    }

    /// Open the room on `relay_url` and publish to the local registry.
    pub async fn start(&self, relay_url: &str, web_url: &str) -> Result<()> {
        if self.ending() {
            return Err(anyhow!("collab host already stopped"));
        }
        let inner = &self.inner;

        let raw_key = crate::crypto::generate_room_key();
        let write_token = crate::crypto::generate_write_token();
        let room_id = generate_room_id();
        let full_link = format_collab_link(relay_url, &room_id, &raw_key, Some(&write_token))
            .map_err(|e| anyhow!(e))?;
        let web_link = format_collab_web_link(
            relay_url,
            &room_id,
            &raw_key,
            Some(&write_token),
            Some(web_url),
        )
        .map_err(|e| anyhow!(e))?;
        let view_link =
            format_collab_link(relay_url, &room_id, &raw_key, None).map_err(|e| anyhow!(e))?;
        let web_view_link =
            format_collab_web_link(relay_url, &room_id, &raw_key, None, Some(web_url))
                .map_err(|e| anyhow!(e))?;
        let parsed = parse_collab_link(&full_link).map_err(|e| anyhow!(e))?;

        let (open_tx, mut open_rx) = mpsc::channel(1);
        *inner.abort_start.lock().unwrap() = Some(open_tx.clone());

        let cipher = CollabCipher::new(&parsed.key);
        let (handle, mut events) = connect_relay(
            &parsed.relay_url,
            &parsed.room_id,
            CollabRole::Host,
            &RelayOptions::default(),
        )
        .await?;
        *inner.handle.lock().unwrap() = Some(handle.clone());

        let opened = tokio::time::timeout(Duration::from_millis(CONNECT_TIMEOUT_MS), async {
            loop {
                tokio::select! {
                    ev = events.next() => match ev {
                        Some(RelayEvent::Opened) => break Ok(()),
                        Some(RelayEvent::Closed { code, reason, will_reconnect }) => {
                            break Err(anyhow!(
                                "relay closed room: {reason} (code {code}, reconnect={will_reconnect})"
                            ));
                        }
                        Some(_) => {}
                        None => break Err(anyhow!("relay session ended before open")),
                    },
                    _ = open_rx.recv() => {
                        break Err(anyhow!("collab host stopped before connecting"));
                    }
                }
            }
        })
        .await
        .map_err(|_| anyhow!("timed out connecting to relay"))?;

        opened?;
        inner
            .relay_connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
        (*inner.abort_start.lock().unwrap()) = None;

        // Publish to the local registry after the relay connection succeeded.
        let source = Arc::new(HostRegistrySource {
            host: inner.clone(),
        });
        let opts = CollabPublishOptions {
            instance_id: Some(inner.instance_id.clone()),
            ..Default::default()
        };
        let publication = match publish_collab_host(source, Some(&opts)) {
            Ok(p) => Some(p),
            Err(err) => {
                eprintln!("[collab] registry publication failed: {err:#}");
                None
            }
        };
        if self.ending() {
            if let Some(p) = publication {
                p.close().await;
            }
            return Err(anyhow!("collab host stopped during startup"));
        }
        *inner.publication.lock().unwrap() = publication;

        // Publish the live links (controller/cli read them via the getters).
        *inner.full_link.lock().unwrap() = Some(full_link.clone());
        *inner.web_link.lock().unwrap() = Some(web_link.clone());
        *inner.view_link.lock().unwrap() = Some(view_link.clone());
        *inner.web_view_link.lock().unwrap() = Some(web_view_link.clone());

        // Install the cipher for outbound sealing, then drive the event loop.
        *CIPHER_CELL.lock().unwrap() = Some(cipher);
        self.run_event_loop(events, full_link, web_link, view_link, web_view_link)
            .await;
        *inner.full_link.lock().unwrap() = None;
        *inner.web_link.lock().unwrap() = None;
        *inner.view_link.lock().unwrap() = None;
        *inner.web_view_link.lock().unwrap() = None;
        *CIPHER_CELL.lock().unwrap() = None;
        Ok(())
    }

    /// Broadcast a goodbye, withdraw the registry entry, and close the socket.
    pub async fn stop(&self, reason: &str) {
        let inner = &self.inner;
        if inner.stopped.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        {
            let mut done = inner.stop_done.lock().unwrap();
            if done.is_some() {
                return;
            }
            let (tx, rx) = mpsc::channel(1);
            *done = Some(rx);
            drop(tx);
        }
        inner
            .stopping
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(tx) = inner.abort_start.lock().unwrap().take() {
            let _ = tx.send(()).await;
        }
        self.send_frame(&json!({ "t": "bye", "reason": reason }), 0);
        // Guests open frames asynchronously and drop any frame still decrypting when their
        // socket closes (collab-web socket.ts `#handleMessage`: `this.#ws !== ws`). The relay
        // tears the room down (close 4001) as soon as the host socket closes, so closing right
        // behind the goodbye would lose it and leave guests reconnecting instead of ended. The
        // TS host gets this gap for free from awaiting the registry shutdown before closing.
        if !inner.peers.lock().unwrap().is_empty() {
            tokio::time::sleep(BYE_DRAIN).await;
        }
        self.teardown().await;
        let _ = inner.stop_done.lock().unwrap().take();
        let _ = std::thread::current();
    }

    async fn teardown(&self) {
        let inner = &self.inner;
        if inner
            .stopped
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        if let Some(pub_) = inner.publication.lock().unwrap().take() {
            pub_.close().await;
        }
        inner.peers.lock().unwrap().clear();
        let handle = inner.handle.lock().unwrap().take();
        if let Some(handle) = handle {
            handle.close();
            // Queued frames (the goodbye) are flushed by the session task ahead of the close
            // frame; wait for it so a caller that drops the runtime right after `stop` does not
            // cancel the flush (host.ts #runStop `await socket.flush()`).
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), handle.closed()).await;
        }
        inner
            .relay_connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Mirror one session entry to all guests (host.ts:477-489).
    pub fn broadcast_entry(&self, entry_json: String) {
        if !self.broadcast_allowed() {
            return;
        }
        if let Ok(entry) = serde_json::from_str::<Value>(&entry_json) {
            if entry.get("type").and_then(Value::as_str) == Some("custom_message")
                && entry.get("customType").and_then(Value::as_str)
                    == Some(COLLAB_ENTRY_OMITTED_CUSTOM_TYPE)
            {
                let notice = oversized_entry_notice("custom_message");
                let event_json = serialize_replicated_event(&notice).json;
                self.send_frame(&json!({ "t": "event", "event": event_json }), 0);
            }
        }
        self.send_raw(format!(r#"{{"t":"entry","entry":{entry_json}}}"#), 0);
    }

    /// Mirror one session event to all guests (host.ts:696-700).
    pub fn broadcast_event(&self, event_json: String) {
        if !self.broadcast_allowed() {
            return;
        }
        self.send_raw(format!(r#"{{"t":"event","event":{event_json}}}"#), 0);
        if let Ok(event) = serde_json::from_str::<Value>(&event_json) {
            let kind = event.get("type").and_then(Value::as_str);
            if kind.is_some_and(|k| STATE_TRIGGER_EVENTS.contains(&k)) {
                self.schedule_state_broadcast();
            }
        }
    }

    /// Recompute and broadcast the footer state, deduped (host.ts:1244-1255).
    pub fn schedule_state_broadcast(&self) {
        if self.ending() || !self.broadcast_allowed() {
            return;
        }
        let state = self.build_state();
        let json = serde_json::to_string(&state).unwrap_or_default();
        let mut last = self.inner.last_state_json.lock().unwrap();
        if last.as_deref() == Some(json.as_str()) {
            return;
        }
        *last = Some(json.clone());
        drop(last);
        self.send_frame(&json!({ "t": "state", "state": state }), 0);
    }

    fn broadcast_allowed(&self) -> bool {
        !self.ending() && self.inner.peers.lock().unwrap().len() > 0 && self.session_still_current()
    }

    fn session_still_current(&self) -> bool {
        self.inner.backend.session_id() == self.inner.session_id
    }

    fn build_state(&self) -> Value {
        let backend = &self.inner.backend;
        let (tokens, window) = backend.context_usage();
        let percent = if window > 0 {
            (tokens as f64 / window as f64) * 100.0
        } else {
            0.0
        };
        let display_name = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "anonymous".to_string());
        let mut participants = vec![json!({ "name": display_name, "role": "host" })];
        for (_, p) in self.inner.peers.lock().unwrap().iter() {
            let mut pj = json!({ "name": p.name, "role": "guest" });
            if !p.can_write {
                pj["readOnly"] = Value::Bool(true);
            }
            participants.push(pj);
        }
        json!({
            "isStreaming": backend.is_streaming(),
            "isAborting": backend.is_aborting(),
            "queuedMessageCount": backend.queued_message_count(),
            "sessionName": backend.session_name(),
            "cwd": backend.cwd(),
            "model": backend.model().map(|(provider, id)| json!({ "provider": provider, "id": id })),
            "thinkingLevel": backend.thinking_level(),
            "contextUsage": {
                "tokens": tokens,
                "contextWindow": window,
                "percent": percent,
            },
            "participants": participants,
        })
    }

    fn snapshot_chunks(&self, entries: Vec<String>) -> Vec<String> {
        let bytes: Vec<usize> = entries.iter().map(|e| e.len()).collect();
        snapshot_chunks(&entries, &bytes)
            .into_iter()
            .map(|chunk| {
                let json = chunk.entries.join(",");
                format!(
                    r#"{{"t":"snapshot-chunk","entries":[{json}],"final":{}}}"#,
                    chunk.is_final
                )
            })
            .collect()
    }

    fn handle_frame(&self, frame: Value, from_peer: u32) {
        let t = frame.get("t").and_then(Value::as_str).unwrap_or("");
        {
            let peers = self.inner.peers.lock().unwrap();
            // `hello` is the frame that registers a peer (host.ts #handleHello), so it
            // must be admitted from unknown senders; everything else still requires it.
            if !peers.contains_key(&from_peer) && t != "ui-response" && t != "hello" {
                return;
            }
        }
        if t == "ui-response" {
            self.send_frame(
                &json!({ "t": "ui-request-end", "reqId": frame["reqId"] }),
                from_peer,
            );
            return;
        }
        if !self.guest_traffic_allowed() {
            return;
        }
        match t {
            "hello" => self.handle_hello(&frame, from_peer),
            "prompt" => self.handle_prompt(&frame, from_peer),
            "abort" => self.handle_abort(from_peer),
            "fetch-transcript" => self.handle_fetch_transcript(&frame, from_peer),
            _ => {}
        }
    }

    fn handle_hello(&self, frame: &Value, from_peer: u32) {
        let proto = frame.get("proto").and_then(Value::as_u64).unwrap_or(0);
        if proto != COLLAB_PROTO {
            self.send_frame(
                &json!({ "t": "error", "message": format!("protocol mismatch: host speaks v{COLLAB_PROTO}, guest sent v{proto}") }),
                from_peer,
            );
            return;
        }
        let raw_name = frame.get("name").and_then(Value::as_str).unwrap_or("");
        let clean_name = {
            let trimmed = raw_name.trim();
            let slice: String = trimmed.chars().take(64).collect();
            if slice.is_empty() {
                format!("guest-{from_peer}")
            } else {
                slice
            }
        };
        let write_token = frame.get("writeToken").and_then(Value::as_str);
        let can_write = write_token.is_some();
        let first_peer = self.inner.peers.lock().unwrap().is_empty();
        self.inner.peers.lock().unwrap().insert(
            from_peer,
            Peer {
                name: clean_name.clone(),
                can_write,
            },
        );

        let header = self.inner.backend.snapshot_header();
        let state = self.build_state();
        if first_peer {
            *self.inner.last_state_json.lock().unwrap() =
                Some(serde_json::to_string(&state).unwrap_or_default());
        }
        let entries = self.inner.backend.snapshot_entries();
        let entry_count = entries.len();
        let mut welcome = json!({
            "t": "welcome",
            "proto": COLLAB_PROTO,
            "header": header,
            "state": state,
            "agents": [],
            "entryCount": entry_count,
        });
        if !can_write {
            welcome["readOnly"] = Value::Bool(true);
        }
        self.send_frame(&welcome, from_peer);
        for chunk in self.snapshot_chunks(entries) {
            self.send_raw(chunk, from_peer);
        }
        self.schedule_state_broadcast();
    }

    fn handle_prompt(&self, frame: &Value, from_peer: u32) {
        if self.reject_while_starting("prompting", from_peer) {
            return;
        }
        let can_write = self
            .inner
            .peers
            .lock()
            .unwrap()
            .get(&from_peer)
            .map(|p| p.can_write)
            .unwrap_or(false);
        if !can_write {
            self.reject_read_only("prompting", from_peer);
            return;
        }
        let text = frame.get("text").and_then(Value::as_str).unwrap_or("");
        let images: Vec<Value> = frame
            .get("images")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if let Err(err) = self.inner.backend.inject_guest_prompt(text, images) {
            self.send_frame(
                &json!({ "t": "error", "message": format!("prompt failed: {err}") }),
                from_peer,
            );
            return;
        }
        if self.inner.backend.is_streaming() {
            self.schedule_state_broadcast();
        }
    }

    fn handle_abort(&self, from_peer: u32) {
        if self.reject_while_starting("interrupting", from_peer) {
            return;
        }
        let can_write = self
            .inner
            .peers
            .lock()
            .unwrap()
            .get(&from_peer)
            .map(|p| p.can_write)
            .unwrap_or(false);
        if !can_write {
            self.reject_read_only("interrupting", from_peer);
            return;
        }
        let _ = self.inner.backend.abort_turn();
    }

    fn handle_fetch_transcript(&self, frame: &Value, from_peer: u32) {
        let req_id = frame.get("reqId").cloned().unwrap_or(Value::Null);
        let from_byte = frame.get("fromByte").and_then(Value::as_u64).unwrap_or(0);
        let reply = json!({
            "t": "transcript",
            "reqId": req_id,
            "text": "",
            "newSize": from_byte,
            "error": "no transcript available",
        });
        self.send_frame(&reply, from_peer);
    }

    fn reject_read_only(&self, action: &str, from_peer: u32) {
        self.send_frame(&json!({ "t": "error", "message": format!("{action} is disabled on a read-only link") }), from_peer);
    }

    fn reject_while_starting(&self, action: &str, from_peer: u32) -> bool {
        let ready = self
            .inner
            .guest_actions_ready
            .as_ref()
            .map(|f| f())
            .unwrap_or(true);
        if ready {
            return false;
        }
        self.send_frame(
            &json!({ "t": "error", "message": format!("{action} is unavailable until the host finishes starting up") }),
            from_peer,
        );
        true
    }

    fn guest_traffic_allowed(&self) -> bool {
        !self.ending() && self.session_still_current()
    }

    fn send_frame(&self, frame: &Value, to_peer: u32) {
        let json = serde_json::to_string(frame).unwrap_or_default();
        self.send_raw(json, to_peer);
    }

    fn send_raw(&self, json: String, to_peer: u32) {
        let handle = self.inner.handle.lock().unwrap().clone();
        let Some(handle) = handle else { return };
        let cell = CIPHER_CELL.lock().unwrap();
        if let Some(cipher) = cell.as_ref() {
            let sealed = cipher.seal(json.as_bytes());
            let env = pack_envelope(to_peer, &sealed);
            let _ = handle.send_binary(env);
        }
    }

    async fn run_event_loop(
        &self,
        mut events: impl futures::Stream<Item = RelayEvent> + Unpin,
        _full_link: String,
        _web_link: String,
        _view_link: String,
        _web_view_link: String,
    ) {
        while let Some(ev) = events.next().await {
            match ev {
                RelayEvent::Opened => {}
                RelayEvent::Binary(data) => {
                    let Some(envelope) = unpack_envelope(&data) else {
                        continue;
                    };
                    let plain = {
                        let cell = CIPHER_CELL.lock().unwrap();
                        cell.as_ref().and_then(|c| c.open(&envelope.payload))
                    };
                    let Some(plain) = plain else { continue };
                    if let Ok(frame) =
                        serde_json::from_str::<Value>(&String::from_utf8_lossy(&plain))
                    {
                        self.handle_frame(frame, envelope.peer_id);
                    }
                }
                RelayEvent::Text(text) => {
                    if let Ok(msg) = serde_json::from_str::<Value>(&text) {
                        if msg.get("t").and_then(Value::as_str) == Some("peer-left") {
                            if let Some(peer) = msg.get("peer").and_then(Value::as_u64) {
                                if self.guest_traffic_allowed() {
                                    self.inner.peers.lock().unwrap().remove(&(peer as u32));
                                    self.schedule_state_broadcast();
                                }
                            }
                        }
                    }
                }
                RelayEvent::Closed {
                    code,
                    reason,
                    will_reconnect,
                } => {
                    if will_reconnect {
                        self.inner.peers.lock().unwrap().clear();
                        self.schedule_state_broadcast();
                    } else {
                        if !self.ending() {
                            self.teardown().await;
                            if let Some(on_ended) = &self.inner.on_ended {
                                on_ended();
                            }
                        }
                        let _ = (code, reason);
                        break;
                    }
                }
            }
        }
    }
}

static CIPHER_CELL: std::sync::LazyLock<StdMutex<Option<CollabCipher>>> =
    std::sync::LazyLock::new(|| StdMutex::new(None));

/// Registry source backed by a live host.
struct HostRegistrySource {
    host: Arc<HostInner>,
}

impl CollabHostRegistrySource for HostRegistrySource {
    fn snapshot(&self) -> Result<CollabHostSnapshot> {
        let host = &self.host;
        let model = host.backend.model();
        let participants = host.peers.lock().unwrap().len() as i64 + 1;
        Ok(CollabHostSnapshot {
            instance_id: host.instance_id.clone(),
            generation: host.generation,
            pid: std::process::id() as i64,
            session_id: host.session_id.clone(),
            session_name: host.backend.session_name(),
            cwd: host.backend.cwd(),
            model: model.map(|(provider, id)| CollabHostModel { provider, id }),
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0),
            participants,
            relay_connected: host
                .relay_connected
                .load(std::sync::atomic::Ordering::SeqCst),
            input_required: false,
            busy: Some(host.backend.is_streaming()),
            access: host.access,
        })
    }

    fn link(&self, _access: CollabAccess) -> Result<Option<String>> {
        Ok(None)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CollabHostStoppedError(pub String);

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct CollabRelayUnavailableError(pub String);
