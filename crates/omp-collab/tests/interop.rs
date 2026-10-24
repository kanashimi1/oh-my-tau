//! Cross-language interop: the RUST collab stack against the REAL TS
//! collab-web guest (`vendor-src/oh-my-pi/packages/collab-web/src/lib/client.ts`)
//! driven by bun.
//!
//! Two tests share one TS guest script (`tests/ts_guest.ts`):
//!
//! * [`ts_guest_interops_with_rust_host`] hosts the room with the real
//!   [`CollabHost`]. This is the regression test for the join path and it
//!   currently FAILS: `host.rs:417` drops the first `hello` from any peer
//!   (peers are only registered *by* hello), so no welcome is ever sent.
//!   See the interop report, mismatch #1.
//!
//! * [`ts_guest_interops_with_wire_host`] hosts the room with a wire-level
//!   rust host driver (relay_client + crypto + replication, hello gate fixed)
//!   and proves the whole protocol end to end with the same real TS guest:
//!   link parse, hello, welcome, snapshot chunks, prompt → user-turn
//!   injection, transcript reply, live entry broadcast, view-link read-only
//!   rejection, and bye/room-closed propagation.
//!
//! Run: `cargo test -p omp-collab --test interop -- --nocapture`
//!
//! Requires `bun` on PATH and an oh-my-pi checkout: `$OMP_TS_SRC` (root of the
//! checkout) or `<repo root>/vendor-src/oh-my-pi`. `$INTEROP_TS_GUEST`
//! overrides the guest script. When any prerequisite is missing the tests
//! print a `skipping: ...` line to stderr and pass without running.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc::UnboundedReceiver;

use omp_collab::crypto::{generate_room_key, generate_write_token, CollabCipher};
use omp_collab::guest::CollabFrame;
use omp_collab::host::{CollabHost, CollabHostBackend, CollabHostOptions, COLLAB_PROTO};
use omp_collab::protocol::{encode_base64_url, generate_room_id, pack_envelope, unpack_envelope};
use omp_collab::registry::CollabAccess;
use omp_collab::relay_client::{connect_relay, CollabRole, RelayEvent, RelayOptions};
use omp_collab::relay_server::start_local_relay;
use omp_collab::replication::snapshot_chunks;

/// Prompt text the TS guest sends (must match ts_guest.ts PROMPT_TEXT).
const PROMPT_TEXT: &str = "interop-prompt-77";
/// Stop reason the host hands the guests (they must surface it as ended reason).
const BYE_REASON: &str = "interop-bye";
/// Overall watchdog for one bun interaction.
const GUEST_BUDGET: Duration = Duration::from_secs(90);

/// Session entry shape the TS replica accepts verbatim (`SessionEntry`).
fn sample_entry(id: u64, text: &str) -> Value {
    json!({
        "type": "message",
        "id": id,
        "message": {
            "role": "assistant",
            "content": [{ "type": "text", "text": text }],
        },
    })
}

/// The two snapshot entries every host here mirrors.
fn snapshot_entries() -> Vec<String> {
    vec![
        serde_json::to_string(&sample_entry(1, "snapshot entry one")).expect("entry json"),
        serde_json::to_string(&sample_entry(2, "snapshot entry two")).expect("entry json"),
    ]
}

/// The `CollabSessionState` shape `CollabHost::build_state` produces.
fn host_state() -> Value {
    json!({
        "isStreaming": false,
        "isAborting": false,
        "queuedMessageCount": 0,
        "sessionName": "interop session",
        "cwd": "/tmp/room",
        "model": { "provider": "test-provider", "id": "test-model" },
        "thinkingLevel": null,
        "contextUsage": { "tokens": 0, "contextWindow": 0, "percent": 0.0 },
        "participants": [{ "name": "host", "role": "host" }],
    })
}

fn host_header() -> Value {
    json!({
        "id": "host-session-1",
        "title": "interop",
        "timestamp": "2026-10-21T00:00:00.000Z",
        "cwd": "/tmp/room",
    })
}

/// Minimal `CollabHostBackend` recording what the host injects.
struct InteropBackend {
    prompts: Mutex<Vec<(String, Vec<Value>)>>,
    aborts: AtomicUsize,
}

impl InteropBackend {
    fn new() -> Self {
        Self {
            prompts: Mutex::new(Vec::new()),
            aborts: AtomicUsize::new(0),
        }
    }
}

impl CollabHostBackend for InteropBackend {
    fn session_id(&self) -> String {
        "sess-interop".to_string()
    }
    fn session_name(&self) -> Option<String> {
        Some("interop session".to_string())
    }
    fn cwd(&self) -> String {
        "/tmp/room".to_string()
    }
    fn model(&self) -> Option<(String, String)> {
        Some(("test-provider".to_string(), "test-model".to_string()))
    }
    fn snapshot_header(&self) -> Value {
        host_header()
    }
    fn snapshot_entries(&self) -> Vec<String> {
        snapshot_entries()
    }
    fn inject_guest_prompt(&self, text: &str, images: Vec<Value>) -> anyhow::Result<()> {
        self.prompts
            .lock()
            .unwrap()
            .push((text.to_string(), images));
        Ok(())
    }
    fn abort_turn(&self) -> anyhow::Result<()> {
        self.aborts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Root of the TS oh-my-pi checkout: `$OMP_TS_SRC`, else `<repo>/vendor-src/oh-my-pi`
/// (same resolution as `ts_guest.ts`).
fn ts_source_root() -> std::path::PathBuf {
    match std::env::var_os("OMP_TS_SRC") {
        Some(root) if !root.is_empty() => std::path::PathBuf::from(root),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor-src/oh-my-pi"),
    }
}

/// Path of the bun guest script, or `None` (after printing a `skipping:` line
/// to stderr) when the interop prerequisites are missing: `bun` on PATH and
/// the TS collab-web client source.
fn guest_script_path() -> Option<std::path::PathBuf> {
    let script = match std::env::var_os("INTEROP_TS_GUEST") {
        Some(path) if !path.is_empty() => std::path::PathBuf::from(path),
        _ => std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/ts_guest.ts")),
    };
    if !script.is_file() {
        eprintln!("skipping: guest script {} not found", script.display());
        return None;
    }
    let bun_ok = std::process::Command::new("bun")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !bun_ok {
        eprintln!("skipping: `bun` not found on PATH");
        return None;
    }
    let client = ts_source_root().join("packages/collab-web/src/lib/client.ts");
    if !client.is_file() {
        eprintln!(
            "skipping: TS client {} not found (set OMP_TS_SRC to an oh-my-pi checkout)",
            client.display()
        );
        return None;
    }
    Some(script)
}

/// Markers the TS guest printed, one bool per protocol phase.
#[derive(Default, Debug)]
struct GuestMarkers {
    full_live_snapshot: bool,
    prompt_sent: bool,
    transcript_reply: bool,
    live_entry_broadcast: bool,
    view_live_readonly: bool,
    view_prompt_rejected: bool,
    full_ended: bool,
    view_ended: bool,
}

/// Side effects the test injects into the guest marker loop.
struct GuestHooks {
    /// Called once the full-link guest reached `live` on the snapshot.
    on_full_live: Box<dyn Fn() + Send>,
    /// Called once the read-only prompt rejection notice was observed.
    on_view_notice: Box<dyn Fn() + Send>,
}

/// Spawn `bun ts_guest.ts <fullLink> <viewLink>` and drive it through its
/// stdout markers. Returns the observed markers; asserts the child succeeded.
async fn drive_ts_guest(
    script: &std::path::Path,
    full_link: &str,
    view_link: &str,
    hooks: GuestHooks,
) -> GuestMarkers {
    let mut child = tokio::process::Command::new("bun")
        .arg(script)
        .arg(full_link)
        .arg(view_link)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn bun ts_guest.ts");

    let stdout = child.stdout.take().expect("child stdout");
    let mut lines = BufReader::new(stdout).lines();
    let mut markers = GuestMarkers::default();
    let mut stop_sent = false;

    let interaction = async {
        while let Some(line) = lines.next_line().await.expect("read child stdout") {
            println!("[ts-guest] {line}");
            if line.starts_with("FATAL") {
                panic!(
                    "TS guest failed: {line}\n\
                     hint: `timeout waiting for full reaches live` with no prior frames means the \
                     rust host never answered the hello — see the interop report, mismatch #1 \
                     (crates/omp-collab/src/host.rs:417 drops the first hello)"
                );
            }
            if !line.starts_with("MARK ") {
                continue;
            }
            if line.contains("live guest=full entries=2") {
                assert!(
                    line.contains("readonly=false"),
                    "full-link guest must be writable: {line}"
                );
                assert!(
                    line.contains("header_id=host-session-1"),
                    "welcome header echoed: {line}"
                );
                markers.full_live_snapshot = true;
                (hooks.on_full_live)();
            } else if line.contains("prompt_sent") {
                assert!(
                    line.contains(PROMPT_TEXT),
                    "prompt marker carries the text: {line}"
                );
                markers.prompt_sent = true;
            } else if line.contains("transcript kind=error") {
                assert!(
                    line.contains("message=no transcript available"),
                    "host's transcript error reply: {line}"
                );
                markers.transcript_reply = true;
            } else if line.contains("live guest=full entries=3") {
                markers.live_entry_broadcast = true;
            } else if line.contains("live guest=view entries=2") {
                assert!(
                    line.contains("readonly=true"),
                    "view-link guest must be read-only: {line}"
                );
                markers.view_live_readonly = true;
            } else if line.contains("notice guest=view") {
                assert!(
                    line.contains("prompting is disabled on a read-only link"),
                    "read-only rejection notice: {line}"
                );
                markers.view_prompt_rejected = true;
                if !stop_sent {
                    stop_sent = true;
                    (hooks.on_view_notice)();
                }
            } else if line.contains("ended guest=full") {
                assert!(
                    line.contains(BYE_REASON),
                    "full guest ended with the bye reason: {line}"
                );
                markers.full_ended = true;
            } else if line.contains("ended guest=view") {
                assert!(
                    line.contains(BYE_REASON),
                    "view guest ended with the bye reason: {line}"
                );
                markers.view_ended = true;
            }
        }
    };

    match tokio::time::timeout(GUEST_BUDGET, interaction).await {
        Ok(()) => {}
        Err(_) => {
            let _ = child.kill().await;
            panic!("TS guest interaction exceeded {GUEST_BUDGET:?}");
        }
    }

    let output = child.wait_with_output().await.expect("reap bun child");
    assert!(
        output.status.success(),
        "bun exited {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    markers
}

/// Test A — the real [`CollabHost`] vs the real TS guest.
///
/// Regression test for the join path. FAILS until host.rs:417 exempts
/// `hello` from the unknown-peer gate (mismatch #1 in the interop report).
#[tokio::test(flavor = "multi_thread")]
async fn ts_guest_interops_with_rust_host() {
    let Some(script) = guest_script_path() else {
        return;
    };

    // Registry publication needs a config dir (same pattern as the
    // controller tests).
    let tmp = tempfile::tempdir().expect("tempdir");
    std::env::set_var("PI_CONFIG_DIR", tmp.path().join(".omp"));

    let relay = start_local_relay(0).expect("local relay starts");
    let port = relay
        .url
        .rsplit(':')
        .next()
        .and_then(|p| p.parse::<u16>().ok())
        .expect("relay url carries a port");
    let web_url = format!("http://127.0.0.1:{port}");

    let backend = Arc::new(InteropBackend::new());
    let host = Arc::new(CollabHost::new(
        Arc::clone(&backend) as Arc<dyn CollabHostBackend>,
        CollabHostOptions {
            instance_id: "interop-host".to_string(),
            access: CollabAccess::Control,
            ..Default::default()
        },
    ));

    // `CollabHost::start` owns the event loop until the room ends and its
    // future is !Send — drive it on a dedicated thread like
    // controller.start_host does, and signal stop through a channel.
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let host_for_thread = Arc::clone(&host);
    let relay_url = relay.url.clone();
    let web_for_start = web_url.clone();
    let host_thread = std::thread::Builder::new()
        .name("collab-host".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("host runtime");
            rt.block_on(async move {
                tokio::select! {
                    res = host_for_thread.start(&relay_url, &web_for_start) => {
                        if let Err(err) = res {
                            eprintln!("[interop] host ended with error: {err:#}");
                        }
                    }
                    reason = stop_rx.recv() => {
                        host_for_thread
                            .stop(&reason.unwrap_or_else(|| "host stopped".to_string()))
                            .await;
                    }
                }
            });
        })
        .expect("spawn collab-host thread");

    // The room exists (and the links are live) once the relay connection is open.
    tokio::time::timeout(Duration::from_secs(15), async {
        while !host.relay_connected() || host.full_link().is_none() {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("rust host connects to the rust relay");
    let full_link = host.full_link().expect("full link published");
    let view_link = host.view_link().expect("view link published");

    let broadcast_host = Arc::clone(&host);
    let stop_for_hooks = stop_tx.clone();
    let markers = drive_ts_guest(
        &script,
        &full_link,
        &view_link,
        GuestHooks {
            on_full_live: Box::new(move || {
                assert_eq!(
                    broadcast_host.participant_count(),
                    1,
                    "rust host registered the TS guest"
                );
                // Mirror a live entry: host→guest broadcast through the relay.
                broadcast_host.broadcast_entry(
                    serde_json::to_string(&sample_entry(3, "live broadcast entry"))
                        .expect("entry json"),
                );
            }),
            on_view_notice: Box::new(move || {
                let _ = stop_for_hooks.send(BYE_REASON.to_string());
            }),
        },
    )
    .await;

    // ── assertions: every phase of the round trip happened ────────────────
    assert!(
        markers.full_live_snapshot,
        "TS guest reached live on the rust host snapshot"
    );
    assert!(markers.prompt_sent, "TS guest sent the prompt");
    assert!(
        markers.transcript_reply,
        "TS guest got the rust host's transcript reply"
    );
    assert!(
        markers.live_entry_broadcast,
        "live entry broadcast reached the TS guest"
    );
    assert!(
        markers.view_live_readonly,
        "view-link guest reached live read-only"
    );
    assert!(
        markers.view_prompt_rejected,
        "read-only prompt was rejected by the rust host"
    );
    assert!(
        markers.full_ended && markers.view_ended,
        "both guests observed the room close"
    );

    // The rust host received exactly the guest's prompt as a user-turn injection.
    let prompts = backend.prompts.lock().unwrap().clone();
    assert_eq!(
        prompts,
        vec![(PROMPT_TEXT.to_string(), Vec::<Value>::new())],
        "rust host backend must record exactly the TS guest's prompt"
    );
    assert_eq!(
        backend.aborts.load(Ordering::SeqCst),
        0,
        "no aborts were sent"
    );
    // `stop()` sends the goodbye first and finishes teardown afterwards (the guests
    // observe the room close before the host reports `stopped`), so wait for the
    // host thread — which returns once `stop()` resolves — before asserting on it.
    host_thread.join().expect("collab-host thread joins");
    assert!(host.stopped(), "rust host is stopped after the room closed");

    relay.stop();
    println!("[interop] PASS: TS collab-web guest interops with the rust CollabHost");
}

// ═══════════════ Test B: wire-level rust host vs the same TS guest ═══════════════

/// What the wire host saw, for protocol-level assertions.
#[derive(Default)]
struct WireHostLog {
    /// `(peer, name, writeToken present, writeToken value)` per hello.
    hellos: Mutex<Vec<(u32, String, bool, Option<String>)>>,
    prompts: Mutex<Vec<(u32, String)>>,
    transcript_requests: Mutex<Vec<u32>>,
}

/// Wire-level rust host: the protocol half of `CollabHost` driven directly
/// over [`connect_relay`] (the guest.rs fake-host pattern), with the one
/// deviation under test: hello from a not-yet-registered peer is admitted,
/// exactly like TS `host.ts:707` (transport-liveness gate, not a peer-list
/// gate). Every outbound frame is the typed [`CollabFrame`] serde encoding;
/// every inbound frame must parse as [`CollabFrame`].
async fn run_wire_host(
    relay_url: String,
    room_id: String,
    key: Vec<u8>,
    write_token_b64: String,
    backend: Arc<InteropBackend>,
    log: Arc<WireHostLog>,
    mut broadcast_rx: UnboundedReceiver<Value>,
    mut stop_rx: UnboundedReceiver<String>,
    opened_tx: tokio::sync::oneshot::Sender<()>,
) {
    let (relay, mut events) = connect_relay(
        &relay_url,
        &room_id,
        CollabRole::Host,
        &RelayOptions::default(),
    )
    .await
    .expect("wire host connects to relay");
    let cipher = CollabCipher::new(&key);
    let mut peers: HashMap<u32, bool> = HashMap::new(); // peer -> can_write

    let send_frame = |relay: &omp_collab::relay_client::RelayHandle,
                      cipher: &CollabCipher,
                      peer: u32,
                      frame: &CollabFrame| {
        let json = serde_json::to_string(frame).expect("frame json");
        assert!(
            relay.send_binary(pack_envelope(peer, &cipher.seal(json.as_bytes()))),
            "relay send queued"
        );
    };

    let mut opened_tx = Some(opened_tx);
    loop {
        tokio::select! {
            ev = events.next() => match ev {
                Some(RelayEvent::Opened) => {
                    if let Some(tx) = opened_tx.take() {
                        let _ = tx.send(());
                    }
                }
                Some(RelayEvent::Binary(data)) => {
                    let Some(envelope) = unpack_envelope(&data) else { continue };
                    let peer = envelope.peer_id;
                    let Some(plain) = cipher.open(&envelope.payload) else {
                        panic!("wire host could not open a guest frame (peer {peer})");
                    };
                    let frame: CollabFrame = match serde_json::from_slice(&plain) {
                        Ok(frame) => frame,
                        Err(err) => panic!("wire host: guest frame did not parse as CollabFrame: {err}"),
                    };
                    match frame {
                        CollabFrame::Hello { proto, name, write_token } => {
                            assert_eq!(proto, COLLAB_PROTO as u32, "TS guest speaks proto 3");
                            match &write_token {
                                Some(token) => assert_eq!(
                                    token, &write_token_b64,
                                    "TS guest must echo the link's write token verbatim"
                                ),
                                None => {}
                            }
                            let can_write = write_token.is_some();
                            log.hellos.lock().unwrap().push((
                                peer,
                                name,
                                can_write,
                                write_token,
                            ));
                            peers.insert(peer, can_write);
                            let entries = snapshot_entries();
                            let entry_count = u64::try_from(entries.len()).unwrap_or(0);
                            let bytes: Vec<usize> = entries.iter().map(String::len).collect();
                            send_frame(&relay, &cipher, peer, &CollabFrame::Welcome {
                                proto: COLLAB_PROTO as u32,
                                header: host_header(),
                                state: host_state(),
                                agents: Vec::new(),
                                entry_count,
                                read_only: !can_write,
                            });
                            for chunk in snapshot_chunks(&entries, &bytes) {
                                let values: Vec<Value> = chunk
                                    .entries
                                    .iter()
                                    .map(|e| serde_json::from_str(e).expect("entry json"))
                                    .collect();
                                send_frame(&relay, &cipher, peer, &CollabFrame::SnapshotChunk {
                                    entries: values,
                                    is_final: chunk.is_final,
                                });
                            }
                        }
                        CollabFrame::Prompt { text, images } => match peers.get(&peer) {
                            Some(true) => {
                                log.prompts.lock().unwrap().push((peer, text.clone()));
                                backend.inject_guest_prompt(&text, images)
                                    .expect("prompt recorded");
                            }
                            Some(false) => send_frame(&relay, &cipher, peer, &CollabFrame::Error {
                                message: "prompting is disabled on a read-only link".to_string(),
                            }),
                            None => {} // unknown peer, like the TS gate
                        },
                        CollabFrame::FetchTranscript { req_id, from_byte, .. } => {
                            log.transcript_requests.lock().unwrap().push(peer);
                            send_frame(&relay, &cipher, peer, &CollabFrame::Transcript {
                                req_id,
                                text: String::new(),
                                new_size: from_byte,
                                error: Some("no transcript available".to_string()),
                            });
                        }
                        CollabFrame::Abort {} => {
                            backend.abort_turn().expect("abort recorded");
                        }
                        CollabFrame::UiResponse { .. } | CollabFrame::AgentCmd { .. } => {}
                        _ => panic!("wire host received a host→guest frame: {frame:?}"),
                    }
                }
                Some(RelayEvent::Text(text)) => {
                    if let Ok(msg) = serde_json::from_str::<Value>(&text) {
                        if msg.get("t").and_then(Value::as_str) == Some("peer-left") {
                            if let Some(peer) = msg.get("peer").and_then(Value::as_u64) {
                                peers.remove(&(peer as u32));
                            }
                        }
                    }
                }
                Some(RelayEvent::Closed { .. }) | None => break,
            },
            entry = broadcast_rx.recv() => {
                if let Some(entry) = entry {
                    send_frame(&relay, &cipher, 0, &CollabFrame::Entry { entry });
                }
            },
            reason = stop_rx.recv() => {
                let reason = reason.unwrap_or_else(|| "host stopped".to_string());
                eprintln!("[wire-host] sending bye: {reason}");
                send_frame(&relay, &cipher, 0, &CollabFrame::Bye { reason: reason.clone() });
                tokio::time::sleep(Duration::from_millis(300)).await;
                relay.close();
                break;
            },
        }
    }
}

/// Test B — wire-level rust host (relay_client + crypto + replication) vs
/// the same real TS guest: proves the rust wire format, sealing, link
/// encoding, and every frame type interop with the browser client.
#[tokio::test(flavor = "multi_thread")]
async fn ts_guest_interops_with_wire_host() {
    let Some(script) = guest_script_path() else {
        return;
    };
    let relay = start_local_relay(0).expect("local relay starts");

    let room_id = generate_room_id();
    let key = generate_room_key();
    let token = generate_write_token();
    let full_link = omp_collab::link::format_collab_link(&relay.url, &room_id, &key, Some(&token))
        .expect("full link formats");
    let view_link = omp_collab::link::format_collab_link(&relay.url, &room_id, &key, None)
        .expect("view link formats");
    let write_token_b64 = encode_base64_url(&token);

    let backend = Arc::new(InteropBackend::new());
    let log = Arc::new(WireHostLog::default());
    let (broadcast_tx, broadcast_rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let (stop_tx, stop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (opened_tx, opened_rx) = tokio::sync::oneshot::channel::<()>();

    let host_task = tokio::spawn(run_wire_host(
        relay.url.clone(),
        room_id.clone(),
        key.to_vec(),
        write_token_b64.clone(),
        Arc::clone(&backend),
        Arc::clone(&log),
        broadcast_rx,
        stop_rx,
        opened_tx,
    ));
    tokio::time::timeout(Duration::from_secs(10), opened_rx)
        .await
        .expect("wire host opened")
        .expect("opened signal");

    let stop_for_hooks = stop_tx.clone();
    let broadcast_for_hooks = broadcast_tx.clone();
    let markers = drive_ts_guest(
        &script,
        &full_link,
        &view_link,
        GuestHooks {
            on_full_live: Box::new(move || {
                broadcast_for_hooks
                    .send(sample_entry(3, "live broadcast entry"))
                    .expect("queue entry broadcast");
            }),
            on_view_notice: Box::new(move || {
                let _ = stop_for_hooks.send(BYE_REASON.to_string());
            }),
        },
    )
    .await;

    // ── rust-side protocol assertions ─────────────────────────────────────
    let hellos = log.hellos.lock().unwrap().clone();
    assert_eq!(
        hellos.len(),
        2,
        "two hellos (full + view link), got {hellos:?}"
    );
    let full_hello = hellos.iter().find(|(_, _, with_token, _)| *with_token);
    let view_hello = hellos.iter().find(|(_, _, with_token, _)| !*with_token);
    let full_hello = full_hello.expect("full-link hello");
    let view_hello = view_hello.expect("view-link hello");
    assert_eq!(full_hello.1, "bun-guest", "display name crosses the wire");
    assert_eq!(
        full_hello.3.as_deref(),
        Some(write_token_b64.as_str()),
        "write token arrives as base64url, byte-exact"
    );
    assert_eq!(view_hello.1, "bun-viewer", "view guest display name");
    assert_eq!(view_hello.3, None, "view link carries no write token");

    let prompts = log.prompts.lock().unwrap().clone();
    assert_eq!(
        prompts.len(),
        1,
        "only the writable guest's prompt is accepted: {prompts:?}"
    );
    assert_eq!(prompts[0].1, PROMPT_TEXT, "prompt text byte-exact");
    assert_eq!(
        backend.prompts.lock().unwrap().clone(),
        vec![(PROMPT_TEXT.to_string(), Vec::<Value>::new())],
        "backend injection records the guest prompt"
    );
    assert_eq!(
        log.transcript_requests.lock().unwrap().len(),
        1,
        "one fetch-transcript request"
    );

    assert!(
        markers.full_live_snapshot,
        "guest went live on the snapshot"
    );
    assert!(markers.prompt_sent, "guest sent the prompt");
    assert!(markers.transcript_reply, "guest got the transcript reply");
    assert!(markers.live_entry_broadcast, "entry broadcast applied");
    assert!(markers.view_live_readonly, "view guest joined read-only");
    assert!(markers.view_prompt_rejected, "view prompt rejected");
    assert!(markers.full_ended && markers.view_ended, "bye propagated");

    let _ = host_task.await;
    relay.stop();
    println!("[interop] PASS: TS collab-web guest interops with the rust wire protocol");
}
