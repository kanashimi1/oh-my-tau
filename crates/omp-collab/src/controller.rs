//! CollabController — owns collaboration hosting for one interactive process.
//!
//! Rust port of `vendor-src/oh-my-pi/packages/coding-agent/src/collab/controller.ts`
//! (357 lines). The controller owns the host lifecycle for a live session:
//! manual `/collab` start/stop, the opt-in `collab.autoStart` policy, and
//! room rotation when the active session changes.
//!
//! Settings are read lazily at start time through a caller-supplied getter
//! (the CLI wires it to `omp_settings::Settings::get_string`, reusing the
//! omp-settings getter). This keeps the controller decoupled from the CLI's
//! settings plumbing and testable without a config file.
//!
//! # Known limitation (wiring TODO)
//!
//! `CollabHost::start` generates room creds it never exposes (T3 host.rs), and
//! [`crate::registry`] publications never carry links by design (registry.ts).
//! Until the host exposes its real room credentials, this controller computes
//! links via `crate::link` from its own fresh credentials: the getters return
//! well-formed links for UI display, but a guest following one joins an EMPTY
//! relay room — the link does not admit to the host's actual room. Wiring the
//! host's real creds into these getters is the follow-up.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};

use crate::crypto::{generate_room_key, generate_write_token};
use crate::host::{CollabHost, CollabHostBackend, CollabHostOptions};
use crate::link::{format_collab_link, format_collab_web_link};
use crate::protocol::generate_room_id;
use crate::registry::CollabAccess;

pub const DEFAULT_RELAY_URL: &str = "http://127.0.0.1:7466";
const SCHEMA_DEFAULT_RELAY_URL: &str = "http://127.0.0.1:9224";
/// How long `start_host` waits for the relay `Opened` event before returning
/// (the room keeps connecting in the background past this budget).
const CONNECT_WAIT_MS: u64 = 2_000;
/// How long stop/start reaping waits for the host thread to finish teardown.
const TEARDOWN_WAIT_MS: u64 = 5_000;
pub type SettingsGetter = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollabAutoStart {
    Off,
    View,
    Control,
}

#[derive(Debug, Clone)]
pub struct CollabStartOptions {
    pub access: CollabAccess,
    pub relay: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollabLinks {
    pub full_link: String,
    pub view_link: String,
    pub web_link: String,
    pub web_view_link: String,
}

/// Handle to the dedicated host thread: the room's start/event-loop future is
/// not `Send` (host.rs holds std MutexGuards across awaits), so it — and every
/// `CollabHost::stop` — must run on that thread, not on the caller's runtime.
struct HostTask {
    /// Signal the thread to run `CollabHost::stop` and end (tokio channel so
    /// the thread's `recv()` is selectable next to `CollabHost::start`).
    stop_tx: tokio::sync::mpsc::UnboundedSender<String>,
    /// Settles when the thread finished (start failed, room ended, or stop
    /// completed). `None` between spawning the thread and storing the task.
    done_rx: Option<std::sync::mpsc::Receiver<anyhow::Result<()>>>,
}

pub struct CollabController {
    instance_id: String,
    backend: Arc<dyn CollabHostBackend>,
    get: SettingsGetter,
    relay_default: String,
    generation: AtomicU64,
    host: StdMutex<Option<Arc<CollabHost>>>,
    host_task: StdMutex<Option<HostTask>>,
    links: StdMutex<Option<CollabLinks>>,
    shutdown: AtomicBool,
}

impl CollabController {
    /// Create a controller for `backend` with settings getter `get`.
    #[must_use]
    pub fn new(backend: Arc<dyn CollabHostBackend>, get: SettingsGetter) -> Self {
        Self {
            // 64 random bits: unique per process on one machine (controller.ts:68-69).
            instance_id: random_instance_id(),
            backend,
            get,
            relay_default: DEFAULT_RELAY_URL.to_string(),
            generation: AtomicU64::new(0),
            host: StdMutex::new(None),
            host_task: StdMutex::new(None),
            links: StdMutex::new(None),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Process-scoped identity, stable across room generations
    /// (controller.ts `instanceId`).
    #[must_use]
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// Registry generation of the most recently started room; 0 before the
    /// first (controller.ts `generation`).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Live `collab.autoStart` policy, read fresh each call.
    #[must_use]
    pub fn auto_start_mode(&self) -> CollabAutoStart {
        match (self.get)("collab.autoStart").as_deref() {
            Some("control") => CollabAutoStart::Control,
            Some("view") => CollabAutoStart::View,
            _ => CollabAutoStart::Off,
        }
    }

    /// The live room for the current session; stale or ending rooms are
    /// absent (controller.ts `host` getter).
    #[must_use]
    pub fn current_host(&self) -> Option<Arc<CollabHost>> {
        let host = self.host.lock().unwrap().clone()?;
        if host.ending() {
            return None;
        }
        if host.session_id() != self.backend.session_id() {
            return None;
        }
        Some(host)
    }

    /// The live room's links, if a room is currently hosted.
    #[must_use]
    pub fn links(&self) -> Option<CollabLinks> {
        self.links.lock().unwrap().clone()
    }

    /// Full control link (`<relay>/r/<room>.<key+token>`), when hosted.
    #[must_use]
    pub fn full_link(&self) -> Option<String> {
        self.links().map(|l| l.full_link)
    }

    /// View-only link (`<relay>/r/<room>.<key>`), when hosted.
    #[must_use]
    pub fn view_link(&self) -> Option<String> {
        self.links().map(|l| l.view_link)
    }

    /// Browser deep link granting control (`<web>/#<full>`), when hosted.
    #[must_use]
    pub fn web_link(&self) -> Option<String> {
        self.links().map(|l| l.web_link)
    }

    /// Browser deep link granting view-only access (`<web>/#<view>`), when hosted.
    #[must_use]
    pub fn web_view_link(&self) -> Option<String> {
        self.links().map(|l| l.web_view_link)
    }

    /// Start (or reuse) a room for `/collab` (controller.ts `start`). A live
    /// room already granting at least the requested access is reused; a
    /// view-only room is replaced when control is requested.
    ///
    /// # Errors
    /// When the controller is shut down, the relay is unconfigured, or the
    /// host fails to start.
    pub async fn start_host(&self, options: CollabStartOptions) -> Result<Arc<CollabHost>> {
        if self.shutdown.load(Ordering::SeqCst) {
            return Err(anyhow!("collab controller shut down"));
        }
        if let Some(existing) = self.current_host() {
            if matches!(existing.access(), CollabAccess::Control)
                || options.access == CollabAccess::View
            {
                return Ok(existing);
            }
        }
        // Stop any non-current or lower-access room before starting the new
        // one; strict sequencing also keeps the process-wide cipher and the
        // registry publication consistent (TS stops the old room first too).
        self.teardown_task("session switched").await;

        let relay_url = self.resolve_relay_url(options.relay.as_deref())?;
        let web_url = self.resolve_web_url();
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;

        let host = Arc::new(CollabHost::new(
            self.backend.clone(),
            CollabHostOptions {
                instance_id: self.instance_id.clone(),
                generation,
                access: options.access,
                guest_actions_ready: None,
                on_ended: None,
            },
        ));

        // Compute the controller's view of the room's links. See the module
        // doc for the known wiring limitation (these are NOT the host's actual
        // room creds — display only until host.rs exposes them).
        let room_id = generate_room_id();
        let key = generate_room_key();
        let token = generate_write_token();
        let links = CollabLinks {
            full_link: format_collab_link(&relay_url, &room_id, &key, Some(&token))
                .map_err(|e| anyhow!(e))?,
            view_link: format_collab_link(&relay_url, &room_id, &key, None)
                .map_err(|e| anyhow!(e))?,
            web_link: format_collab_web_link(
                &relay_url,
                &room_id,
                &key,
                Some(&token),
                web_url.as_deref(),
            )
            .map_err(|e| anyhow!(e))?,
            web_view_link: format_collab_web_link(
                &relay_url,
                &room_id,
                &key,
                None,
                web_url.as_deref(),
            )
            .map_err(|e| anyhow!(e))?,
        };

        // CollabHost::start owns the relay event loop and returns only when the
        // room ends (host.rs `run_event_loop`); the TS host is callback-driven,
        // which is why controller.ts can `await host.start()`. Its future — and
        // `CollabHost::stop` — are not `Send` (host.rs holds std MutexGuards
        // across awaits), so start AND stop run on a dedicated single-thread
        // runtime thread: one OS thread per hosted room, ending with the room.
        // The controller signals stop through `stop_tx` and reaps the thread
        // via `done_rx`.
        let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let start_host = Arc::clone(&host);
        let relay_for_start = relay_url.clone();
        let web_for_start = web_url.clone().unwrap_or_default();
        std::thread::Builder::new()
            .name("collab-host".to_string())
            .spawn(move || {
                let run = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                match run {
                    Err(err) => {
                        let _ = done_tx.send(Err(anyhow!("collab host runtime: {err}")));
                    }
                    Ok(rt) => rt.block_on(async move {
                        tokio::select! {
                            res = start_host.start(&relay_for_start, &web_for_start) => {
                                let res = res.map_err(|err| anyhow!("collab host stopped: {err:#}"));
                                if let Err(err) = &res {
                                    eprintln!("[collab] {err:#}");
                                }
                                let _ = done_tx.send(res);
                            }
                            reason = stop_rx.recv() => {
                                let reason =
                                    reason.unwrap_or_else(|| "host stopped".to_string());
                                start_host.stop(&reason).await;
                                let _ = done_tx.send(Ok(()));
                            }
                        }
                    }),
                }
            })
            .map_err(|err| anyhow!("spawn collab host thread: {err}"))?;

        // Register the task before waiting so a stop arriving during the
        // connect window still reaches the thread.
        *self.host.lock().unwrap() = Some(Arc::clone(&host));
        *self.host_task.lock().unwrap() = Some(HostTask {
            stop_tx: stop_tx.clone(),
            done_rx: Some(done_rx),
        });
        let done_rx = self
            .host_task
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|t| t.done_rx.take());

        // Surface immediate connect failures without burning the relay
        // handshake budget (host.rs CONNECT_TIMEOUT_MS = 15s): return as soon
        // as the relay reports Opened, when the start task settles, or after a
        // short grace period (slow remote handshake keeps hosting in
        // background, matching TS's fire-and-forget UX).
        let deadline = std::time::Instant::now() + Duration::from_millis(CONNECT_WAIT_MS);
        let start_result = loop {
            if host.relay_connected() || std::time::Instant::now() >= deadline {
                break None;
            }
            let Some(rx) = done_rx.as_ref() else {
                break None;
            };
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(res) => break Some(res),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    break Some(Err(anyhow!("collab host thread died")))
                }
            }
        };
        if let Some(res) = start_result {
            match res {
                Ok(()) => {
                    // The room ended immediately after connecting; keep the
                    // (now dead) task registered so a later stop is a no-op.
                }
                Err(err) => {
                    *self.host.lock().unwrap() = None;
                    *self.host_task.lock().unwrap() = None;
                    return Err(err);
                }
            }
        }
        if let Some(task) = self.host_task.lock().unwrap().as_mut() {
            task.done_rx = done_rx;
        }
        // Prefer the host's real room creds now that host.rs exposes them;
        // fall back to the controller's own (display-only) computation.
        let links = CollabLinks {
            full_link: host.full_link().unwrap_or(links.full_link),
            view_link: host.view_link().unwrap_or(links.view_link),
            web_link: host.web_link().unwrap_or(links.web_link),
            web_view_link: host.web_view_link().unwrap_or(links.web_view_link),
        };
        *self.links.lock().unwrap() = Some(links);
        Ok(host)
    }

    /// Stop the current room (controller.ts `stop`; this port has no pending
    /// relaunch launches to cancel). Runs on the host thread and is reaped
    /// bounded — a wedged teardown gives up waiting and lets the thread finish
    /// in the background.
    pub async fn stop_host(&self, reason: &str) {
        self.teardown_task(reason).await;
    }

    /// Signal the host thread to stop and wait (bounded) for it to finish.
    async fn teardown_task(&self, reason: &str) {
        // Take out of the lock first: a std MutexGuard must not live across
        // the waits (the dispatch chain runs inside `tokio::spawn`).
        let task = self.host_task.lock().unwrap().take();
        *self.host.lock().unwrap() = None;
        let Some(task) = task else { return };
        let _ = task.stop_tx.send(reason.to_string());
        let Some(done_rx) = task.done_rx else {
            return;
        };
        let deadline = std::time::Instant::now() + Duration::from_millis(TEARDOWN_WAIT_MS);
        loop {
            match done_rx.recv_timeout(Duration::from_millis(20)) {
                Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                }
            }
        }
    }

    /// Stop hosting for good; no further rooms are started for this process.
    pub async fn shutdown(&self, reason: &str) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.stop_host(reason).await;
    }

    /// Resolve the relay URL to connect to: an explicit `relay` override wins,
    /// then `collab.relayUrl` (treating the schema default as unset), then the
    /// local relay [`DEFAULT_RELAY_URL`]. Scheme-less values default to `wss://`
    /// (controller.ts `#resolveRelayUrl`).
    ///
    /// # Errors
    /// When no relay is configured at all.
    fn resolve_relay_url(&self, relay: Option<&str>) -> Result<String> {
        let input = relay
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| {
                let v = (self.get)("collab.relayUrl").unwrap_or_default();
                let v = v.trim().to_string();
                if v.is_empty() || v == SCHEMA_DEFAULT_RELAY_URL {
                    None
                } else {
                    Some(v)
                }
            });
        let input = input.unwrap_or_else(|| self.relay_default.clone());
        if input.is_empty() {
            bail!(
                "No relay configured. Set collab.relayUrl in /settings or pass one: /collab relay.example.com"
            );
        }
        if input.contains("://") {
            Ok(input)
        } else {
            Ok(format!("wss://{input}"))
        }
    }

    /// Resolve the browser web base (`collab.webUrl`); empty derives from the
    /// relay inside `format_collab_web_link` (settings.ts `cfgCollabWebUrl`).
    fn resolve_web_url(&self) -> Option<String> {
        (self.get)("collab.webUrl").filter(|s| !s.trim().is_empty())
    }
}

/// 64 random bits as lowercase hex (controller.ts `randomBytes(8).toString("hex")`).
fn random_instance_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    let mut out = String::with_capacity(16);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::parse_collab_link;
    use std::sync::Arc;

    /// Minimal backend standing in for the CLI's Runtime.
    struct FakeBackend;
    impl CollabHostBackend for FakeBackend {
        fn session_id(&self) -> String {
            "sess-123".to_string()
        }
        fn cwd(&self) -> String {
            "/tmp".to_string()
        }
    }

    fn getter(pairs: &[(&str, &str)]) -> SettingsGetter {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Arc::new(move |id: &str| map.get(id).cloned())
    }

    #[test]
    fn auto_start_mode_parses() {
        let ctl = CollabController::new(
            Arc::new(FakeBackend),
            getter(&[("collab.autoStart", "control")]),
        );
        assert_eq!(ctl.auto_start_mode(), CollabAutoStart::Control);
        let ctl = CollabController::new(
            Arc::new(FakeBackend),
            getter(&[("collab.autoStart", "view")]),
        );
        assert_eq!(ctl.auto_start_mode(), CollabAutoStart::View);
        let ctl = CollabController::new(Arc::new(FakeBackend), getter(&[]));
        assert_eq!(ctl.auto_start_mode(), CollabAutoStart::Off);
        let ctl = CollabController::new(
            Arc::new(FakeBackend),
            getter(&[("collab.autoStart", "off")]),
        );
        assert_eq!(ctl.auto_start_mode(), CollabAutoStart::Off);
    }

    #[test]
    fn relay_default_falls_back_to_local_relay() {
        let ctl = CollabController::new(
            Arc::new(FakeBackend),
            getter(&[("collab.relayUrl", "http://127.0.0.1:9224")]),
        );
        assert_eq!(
            ctl.resolve_relay_url(None).unwrap(),
            "http://127.0.0.1:7466"
        );
        let ctl = CollabController::new(Arc::new(FakeBackend), getter(&[]));
        assert_eq!(
            ctl.resolve_relay_url(None).unwrap(),
            "http://127.0.0.1:7466"
        );
    }

    #[test]
    fn relay_override_wins_and_scheme_less_defaults_to_wss() {
        let ctl = CollabController::new(Arc::new(FakeBackend), getter(&[]));
        assert_eq!(
            ctl.resolve_relay_url(Some("relay.example.com")).unwrap(),
            "wss://relay.example.com"
        );
        assert_eq!(
            ctl.resolve_relay_url(Some("ws://127.0.0.1:9999")).unwrap(),
            "ws://127.0.0.1:9999"
        );
        let ctl = CollabController::new(
            Arc::new(FakeBackend),
            getter(&[("collab.relayUrl", "wss://my.omp.sh")]),
        );
        assert_eq!(ctl.resolve_relay_url(None).unwrap(), "wss://my.omp.sh");
    }

    #[test]
    fn computed_links_parse_round_trip() {
        let ctl = CollabController::new(Arc::new(FakeBackend), getter(&[]));
        let relay = ctl.resolve_relay_url(None).unwrap();
        let room_id = generate_room_id();
        let key = generate_room_key();
        let token = generate_write_token();
        let full = format_collab_link(&relay, &room_id, &key, Some(&token)).unwrap();
        let view = format_collab_link(&relay, &room_id, &key, None).unwrap();
        let parsed_full = parse_collab_link(&full).unwrap();
        assert_eq!(parsed_full.room_id, room_id);
        assert!(parsed_full.write_token.is_some());
        let parsed_view = parse_collab_link(&view).unwrap();
        assert_eq!(parsed_view.room_id, room_id);
        assert!(parsed_view.write_token.is_none());
        let web = format_collab_web_link(&relay, &room_id, &key, Some(&token), None).unwrap();
        assert!(web.starts_with("http://127.0.0.1:7466/#"));
        let parsed_web = parse_collab_link(&web).unwrap();
        assert_eq!(parsed_web.room_id, room_id);
    }

    #[tokio::test]
    async fn start_and_stop_host_with_local_relay() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("PI_CONFIG_DIR", tmp.path().join(".omp"));
        let relay = crate::relay_server::start_local_relay(0).unwrap();

        let ctl = CollabController::new(Arc::new(FakeBackend), getter(&[]));
        assert!(ctl.current_host().is_none());

        let host = ctl
            .start_host(CollabStartOptions {
                access: CollabAccess::Control,
                relay: Some(relay.url.clone()),
            })
            .await
            .expect("host starts against local relay");
        assert_eq!(host.session_id(), "sess-123");
        assert_eq!(
            ctl.current_host().map(|h| h.instance_id().to_string()),
            Some(ctl.instance_id().to_string())
        );
        let links = ctl.links().expect("links stored after start");
        assert!(!links.full_link.is_empty());
        assert!(!links.web_link.is_empty());
        let web = ctl.web_link().unwrap();
        assert!(web.contains("/r/"));
        assert_eq!(ctl.generation(), 1);

        ctl.stop_host("test").await;
        ctl.stop_host("test").await;
        let _ = relay;
    }
}
