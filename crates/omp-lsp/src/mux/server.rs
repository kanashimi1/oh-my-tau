//! The broker-shared LSP mux daemon core (omp `src/lsp/mux/server.ts`,
//! v18.6.1): one language-server child per identity key, multiplexed over
//! Content-Length framed JSON-RPC links on a Unix socket.
//!
//! Semantics ported 1:1 from TS: `omp/muxConnect` binds a link to an idle
//! server or spawns one; `initialize` is executed once and cached; per-session
//! `didOpen`/`didClose` isolation closes a dropped session's documents;
//! `publishDiagnostics` are cached and replayed; servers linger
//! `SERVER_LINGER_MS` after their last session and the mux itself exits after
//! `MUX_IDLE_MS` with no links. Both timers are overridable through the
//! environment for tests (`OMP_LSP_MUX_SERVER_LINGER_MS`,
//! `OMP_LSP_MUX_IDLE_MS`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tracing::{debug, warn};

use super::protocol::{
    lsp_mux_ready_banner, mux_server_key, MuxConnectParams, MuxConnectResult, MUX_CONNECT_METHOD,
    MUX_PING_METHOD, MUX_PING_RESULT, MUX_RESTART_METHOD,
};
use crate::client::Framer;

/// How long a server with zero attached sessions is kept warm
/// (omp `SERVER_LINGER_MS`). Overridable via `OMP_LSP_MUX_SERVER_LINGER_MS`.
fn server_linger_ms() -> u64 {
    std::env::var("OMP_LSP_MUX_SERVER_LINGER_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5 * 60 * 1_000)
}

/// How long the mux runs with zero connected links before it exits
/// (omp `MUX_IDLE_MS`). Overridable via `OMP_LSP_MUX_IDLE_MS`.
fn mux_idle_ms() -> u64 {
    std::env::var("OMP_LSP_MUX_IDLE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15 * 60 * 1_000)
}

/// Graceful `shutdown` → `exit` budget before the child is killed
/// (omp `SHUTDOWN_BUDGET_MS`).
const SHUTDOWN_BUDGET_MS: u64 = 2_000;

fn rpc_result(id: &Value, result: &Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn rpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn frame(message: &Value) -> Vec<u8> {
    let body = serde_json::to_string(message).expect("serialize mux frame");
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

/// `Content-Length`-framed write of one message.
async fn write_framed(
    writer: &mut (impl AsyncWriteExt + Unpin),
    message: &Value,
) -> std::io::Result<()> {
    writer.write_all(&frame(message)).await?;
    writer.flush().await
}

fn has_method(message: &Value) -> bool {
    message.get("method").and_then(Value::as_str).is_some()
}

fn has_request_id(message: &Value) -> bool {
    matches!(message.get("id"), Some(Value::String(_) | Value::Number(_)))
}

/// Stable string key for a client request id (`Value` is not `Hash`).
fn id_key(id: &Value) -> String {
    serde_json::to_string(id).unwrap_or_default()
}

fn parse_connect_params(params: Option<&Value>) -> Option<MuxConnectParams> {
    let params = params?;
    serde_json::from_value::<MuxConnectParams>(params.clone()).ok()
}

fn parse_document_params(params: Option<&Value>) -> Option<(String, i64)> {
    let params = params?;
    let doc = params.get("textDocument")?;
    let uri = doc.get("uri")?.as_str()?.to_string();
    let version = doc.get("version")?.as_i64()?;
    Some((uri, version))
}

fn parse_uri(params: Option<&Value>) -> Option<String> {
    Some(
        params?
            .get("textDocument")?
            .get("uri")?
            .as_str()?
            .to_string(),
    )
}

fn parse_diagnostics(params: Option<&Value>) -> Option<Value> {
    let params = params?;
    params.get("uri")?.as_str()?;
    Some(params.clone())
}

fn parse_progress(params: Option<&Value>) -> Option<Value> {
    let params = params?;
    match params.get("token") {
        Some(Value::String(_) | Value::Number(_)) => Some(params.clone()),
        _ => None,
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A server request/notification destined for one specific link, tracked so a
/// client response can be relayed back under the server's original id.
struct PendingClient {
    relay: bool,
    server_id: Option<Value>,
}

/// A forwarded client request awaiting the server's response.
enum PendingForward {
    Forward {
        session: u64,
        original_id: Value,
        drop: bool,
    },
    Initialize,
    Resolve(oneshot::Sender<()>),
}

/// One mux link. `writer` is `None` once the socket has been destroyed.
struct Session {
    id: u64,
    writer: AsyncMutex<Option<tokio::net::unix::OwnedWriteHalf>>,
    open_uris: Mutex<HashSet<String>>,
    /// Client request id (keyed canonically) → mux id allocated on the server.
    forwarded_ids: Mutex<HashMap<String, i64>>,
    /// `mux:N` client-side ids → relay bookkeeping.
    pending_client: Mutex<HashMap<String, PendingClient>>,
    server: Mutex<Option<Arc<ServerInstance>>>,
    initialized: AtomicBool,
    closed: AtomicBool,
    last_activity: AtomicI64,
}

impl Session {
    fn new(id: u64, writer: tokio::net::unix::OwnedWriteHalf) -> Self {
        Self {
            id,
            writer: AsyncMutex::new(Some(writer)),
            open_uris: Mutex::new(HashSet::new()),
            forwarded_ids: Mutex::new(HashMap::new()),
            pending_client: Mutex::new(HashMap::new()),
            server: Mutex::new(None),
            initialized: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            last_activity: AtomicI64::new(0),
        }
    }

    fn touch(&self, clock: i64) {
        self.last_activity.store(clock, Ordering::SeqCst);
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    async fn send(&self, message: &Value) {
        let mut guard = self.writer.lock().await;
        if let Some(writer) = guard.as_mut() {
            if write_framed(writer, message).await.is_err() {
                *guard = None;
            }
        }
    }

    /// Destroy the socket write half (the daemon side of
    /// `session.socket.destroy()`). The read task observes EOF once the
    /// peer closes and runs `close_session`, which is what severs the link
    /// bookkeeping; setting `closed` here would make that cleanup a no-op.
    async fn destroy(&self) {
        *self.writer.lock().await = None;
    }
}

/// One shared language-server child and its multiplexed state.
pub(crate) struct ServerInstance {
    key: String,
    state: Weak<MuxState>,
    stdin: AsyncMutex<Option<ChildStdin>>,
    child: Mutex<Option<Child>>,
    pid: u32,
    sessions: Mutex<Vec<Arc<Session>>>,
    documents: Mutex<HashSet<String>>,
    diagnostics: Mutex<BTreeMap<String, Value>>,
    /// `client/registerCapability` batches, pruned by unregistrations.
    registrations: Mutex<Vec<Value>>,
    /// Live `$/progress` params keyed by token string.
    progress: Mutex<BTreeMap<String, Value>>,
    pending: Mutex<HashMap<i64, PendingForward>>,
    initialize_waiters: Mutex<Vec<(u64, Value)>>,
    init_result: Mutex<Option<Value>>,
    init_cached: AtomicBool,
    init_in_flight: AtomicBool,
    initialized_sent: AtomicBool,
    next_mux_id: AtomicI64,
    next_client_id: AtomicI64,
    stopping: AtomicBool,
}

impl ServerInstance {
    fn mux_id(&self) -> i64 {
        self.next_mux_id.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn client_id(&self) -> String {
        format!(
            "mux:{}",
            self.next_client_id.fetch_add(1, Ordering::SeqCst) + 1
        )
    }

    fn session_snapshot(&self) -> Vec<Arc<Session>> {
        lock(&self.sessions).clone()
    }

    async fn write(&self, message: &Value) {
        let mut guard = self.stdin.lock().await;
        if let Some(stdin) = guard.as_mut() {
            if write_framed(stdin, message).await.is_err() {
                *guard = None;
            }
        }
    }
}

/// Daemon-wide registry.
pub(crate) struct MuxState {
    endpoint: PathBuf,
    servers: Mutex<HashMap<String, Arc<ServerInstance>>>,
    sessions: Mutex<HashMap<u64, Arc<Session>>>,
    next_session: AtomicU64,
    clock: AtomicI64,
    idle: Mutex<Option<tokio::task::JoinHandle<()>>>,
    shutting_down: AtomicBool,
}

impl MuxState {
    fn new(endpoint: PathBuf) -> Self {
        Self {
            endpoint,
            servers: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
            next_session: AtomicU64::new(1),
            clock: AtomicI64::new(0),
            idle: Mutex::new(None),
            shutting_down: AtomicBool::new(false),
        }
    }

    fn tick(&self) -> i64 {
        self.clock.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn disarm_idle(&self) {
        if let Some(handle) = lock(&self.idle).take() {
            handle.abort();
        }
    }

    fn arm_idle(self: &Arc<Self>) {
        self.disarm_idle();
        if self.shutting_down.load(Ordering::SeqCst) || !lock(&self.sessions).is_empty() {
            return;
        }
        let state = Arc::clone(self);
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(mux_idle_ms())).await;
            if lock(&state.sessions).is_empty() && !state.shutting_down.load(Ordering::SeqCst) {
                debug!("omp lsp mux idle; exiting");
                state.shutting_down.store(true, Ordering::SeqCst);
                let _ = std::fs::remove_file(&state.endpoint);
                std::process::exit(0);
            }
        });
        *lock(&self.idle) = Some(handle);
    }
}

/// Run the mux daemon on `endpoint` until it idles out (omp
/// `startLspMuxFromEnvironment`'s serve loop).
///
/// # Errors
///
/// Socket bind/accept failures and a live mux already on the endpoint.
pub async fn run(endpoint: &Path, _project_dir: &Path) -> std::io::Result<()> {
    if endpoint.exists() {
        if probe_unix(endpoint).await {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AddrInUse,
                format!("LSP mux already listening on {}", endpoint.display()),
            ));
        }
        std::fs::remove_file(endpoint)?;
    }
    if let Some(parent) = endpoint.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(endpoint)?;
    let state = Arc::new(MuxState::new(endpoint.to_path_buf()));
    state.arm_idle();
    println!("{}", lsp_mux_ready_banner(&endpoint.to_string_lossy()));
    std::io::stdout().flush().ok();
    loop {
        let (stream, _) = listener.accept().await?;
        let (reader, writer) = stream.into_split();
        let id = state.next_session.fetch_add(1, Ordering::SeqCst);
        let session = Arc::new(Session::new(id, writer));
        lock(&state.sessions).insert(id, Arc::clone(&session));
        state.disarm_idle();
        let st = Arc::clone(&state);
        tokio::spawn(async move {
            handle_connection(st, session, reader).await;
        });
    }
}

/// True when something is accepting connections at `endpoint`.
async fn probe_unix(endpoint: &Path) -> bool {
    matches!(
        tokio::time::timeout(Duration::from_millis(200), UnixStream::connect(endpoint)).await,
        Ok(Ok(_))
    )
}

async fn handle_connection(
    state: Arc<MuxState>,
    session: Arc<Session>,
    mut reader: tokio::net::unix::OwnedReadHalf,
) {
    let mut framer = Framer::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        if session.is_closed() {
            break;
        }
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => framer.push(&chunk[..n]),
        }
        while let Some(text) = framer.next() {
            let Ok(message) = serde_json::from_str::<Value>(&text) else {
                warn!("LSP mux client sent malformed JSON");
                continue;
            };
            if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
                warn!("LSP mux client sent invalid JSON-RPC message");
                continue;
            }
            from_session(&state, &session, message).await;
        }
    }
    close_session(&state, &session).await;
}

async fn from_session(state: &Arc<MuxState>, session: &Arc<Session>, message: Value) {
    if session.is_closed() {
        return;
    }
    let clock = state.tick();
    session.touch(clock);
    if !has_method(&message) {
        handle_client_response(session, &message);
        return;
    }
    let request = has_request_id(&message);
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let id = message.get("id").cloned().unwrap_or(Value::Null);

    if method == MUX_PING_METHOD && request {
        session
            .send(&rpc_result(&id, &json!(MUX_PING_RESULT)))
            .await;
        return;
    }
    if method == MUX_CONNECT_METHOD && request {
        handle_mux_connect(state, session, &message, id).await;
        return;
    }
    let bound = lock(&session.server).clone();
    let Some(server) = bound else {
        if request {
            session
                .send(&rpc_error(&id, -32002, "muxConnect must be first"))
                .await;
        }
        return;
    };
    match (method, request) {
        (MUX_RESTART_METHOD, false) => {
            kill_server(state, &server);
        }
        ("initialize", true) => {
            initialize_session(session, &server, &message, id).await;
        }
        ("initialized", false) => {
            if server.initialized_sent.swap(true, Ordering::SeqCst) {
                replay_state(session, &server).await;
            } else {
                server.write(&message).await;
            }
            session.initialized.store(true, Ordering::SeqCst);
        }
        ("shutdown", true) => {
            // Swallowed like TS: the shared server outlives any one session.
            session.send(&rpc_result(&id, &Value::Null)).await;
        }
        ("exit", false) => {
            session.destroy().await;
            // The client keeps its read half open, so no EOF arrives; tear the
            // link down here or the daemon would never idle out.
            close_session(state, session).await;
        }
        ("textDocument/didOpen", false) => {
            if let Some((uri, _version)) = parse_document_params(message.get("params")) {
                lock(&session.open_uris).insert(uri.clone());
                lock(&server.documents).insert(uri);
            }
            server.write(&message).await;
        }
        ("textDocument/didClose", false) => {
            if let Some(uri) = parse_uri(message.get("params")) {
                lock(&session.open_uris).remove(&uri);
                lock(&server.documents).remove(&uri);
            }
            server.write(&message).await;
        }
        ("$/cancelRequest", false) => {
            let original = message.get("params").and_then(|p| p.get("id"));
            if let Some(original) = original {
                let mux_id = lock(&session.forwarded_ids).get(&id_key(original)).copied();
                if let Some(mux_id) = mux_id {
                    let mut cancelled = message.clone();
                    cancelled["params"]["id"] = json!(mux_id);
                    server.write(&cancelled).await;
                }
            }
        }
        (_, true) => {
            let mux_id = server.mux_id();
            lock(&server.pending).insert(
                mux_id,
                PendingForward::Forward {
                    session: session.id,
                    original_id: id.clone(),
                    drop: false,
                },
            );
            lock(&session.forwarded_ids).insert(id_key(&id), mux_id);
            let mut forwarded = message.clone();
            forwarded["id"] = json!(mux_id);
            server.write(&forwarded).await;
        }
        (_, false) => {
            server.write(&message).await;
        }
    }
}

async fn handle_mux_connect(
    state: &Arc<MuxState>,
    session: &Arc<Session>,
    message: &Value,
    id: Value,
) {
    if lock(&session.server).is_some() {
        session
            .send(&rpc_error(&id, -32600, "session already bound"))
            .await;
        return;
    }
    let Some(params) = parse_connect_params(message.get("params")) else {
        session
            .send(&rpc_error(&id, -32602, "invalid mux connect params"))
            .await;
        return;
    };
    let key = mux_server_key(&params);
    let mut server = lock(&state.servers)
        .values()
        .find(|c| c.key == key)
        .cloned();
    if let Some(candidate) = server.clone() {
        // A crashed child must not be re-adopted.
        let dead = lock(&candidate.child)
            .as_mut()
            .is_some_and(|c| matches!(c.try_wait(), Ok(Some(_))));
        if dead || candidate.stopping.load(Ordering::SeqCst) {
            server = None;
        }
    }
    let mut spawned = false;
    let server = if let Some(existing) = server {
        existing
    } else {
        spawned = true;
        spawn_server(state, &key, &params)
    };
    lock(&server.sessions).push(Arc::clone(session));
    *lock(&session.server) = Some(Arc::clone(&server));
    let result = MuxConnectResult {
        key,
        spawned,
        pid: Some(server.pid),
    };
    session
        .send(&rpc_result(
            &id,
            &serde_json::to_value(result).expect("serialize connect result"),
        ))
        .await;
}

fn spawn_server(
    state: &Arc<MuxState>,
    key: &str,
    params: &MuxConnectParams,
) -> Arc<ServerInstance> {
    let mut command = Command::new(&params.command);
    command
        .args(&params.args)
        .current_dir(&params.cwd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    if let Some(env) = &params.env {
        command.envs(env);
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            // Keep a placeholder so the connect result can still be formed
            // with pid `None`; the first real request fails on the dead pipe.
            warn!("LSP mux failed to spawn server {key}: {error}");
            let server = ServerInstance {
                key: key.to_string(),
                state: Arc::downgrade(state),
                stdin: AsyncMutex::new(None),
                child: Mutex::new(None),
                pid: 0,
                sessions: Mutex::new(Vec::new()),
                documents: Mutex::new(HashSet::new()),
                diagnostics: Mutex::new(BTreeMap::new()),
                registrations: Mutex::new(Vec::new()),
                progress: Mutex::new(BTreeMap::new()),
                pending: Mutex::new(HashMap::new()),
                initialize_waiters: Mutex::new(Vec::new()),
                init_result: Mutex::new(None),
                init_cached: AtomicBool::new(false),
                init_in_flight: AtomicBool::new(false),
                initialized_sent: AtomicBool::new(false),
                next_mux_id: AtomicI64::new(0),
                next_client_id: AtomicI64::new(0),
                stopping: AtomicBool::new(true),
            };
            let server = Arc::new(server);
            lock(&state.servers).insert(key.to_string(), Arc::clone(&server));
            return server;
        }
    };
    let pid = child.id().unwrap_or(0);
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let server = ServerInstance {
        key: key.to_string(),
        state: Arc::downgrade(state),
        stdin: AsyncMutex::new(stdin),
        child: Mutex::new(Some(child)),
        pid,
        sessions: Mutex::new(Vec::new()),
        documents: Mutex::new(HashSet::new()),
        diagnostics: Mutex::new(BTreeMap::new()),
        registrations: Mutex::new(Vec::new()),
        progress: Mutex::new(BTreeMap::new()),
        pending: Mutex::new(HashMap::new()),
        initialize_waiters: Mutex::new(Vec::new()),
        init_result: Mutex::new(None),
        init_cached: AtomicBool::new(false),
        init_in_flight: AtomicBool::new(false),
        initialized_sent: AtomicBool::new(false),
        next_mux_id: AtomicI64::new(0),
        next_client_id: AtomicI64::new(0),
        stopping: AtomicBool::new(false),
    };
    let server = Arc::new(server);
    lock(&state.servers).insert(key.to_string(), Arc::clone(&server));
    if let Some(stdout) = stdout {
        let st = Arc::clone(&server);
        tokio::spawn(async move {
            read_server(&st, stdout).await;
        });
    }
    let st = Arc::clone(&server);
    tokio::spawn(async move {
        // Reap the child so a server exit is observed even when its stdout
        // pipe outlives the process.
        let mut child = lock(&st.child).take();
        if let Some(child) = child.as_mut() {
            let _ = child.wait().await;
        }
        if let Some(state) = st.state.upgrade() {
            server_exited(&state, &st);
        }
    });
    server
}

async fn initialize_session(
    session: &Arc<Session>,
    server: &Arc<ServerInstance>,
    message: &Value,
    id: Value,
) {
    if server.init_cached.load(Ordering::SeqCst) {
        let result = lock(&server.init_result).clone().unwrap_or(Value::Null);
        session.send(&rpc_result(&id, &result)).await;
        return;
    }
    lock(&server.initialize_waiters).push((session.id, id));
    if server.init_in_flight.swap(true, Ordering::SeqCst) {
        return;
    }
    let mux_id = server.mux_id();
    lock(&server.pending).insert(mux_id, PendingForward::Initialize);
    let mut forwarded = message.clone();
    forwarded["id"] = json!(mux_id);
    // Language servers self-terminate with their advertised client pid; pin
    // the mux so the shared server never dies with one session (omp comment).
    forwarded["params"]["processId"] = json!(std::process::id());
    server.write(&forwarded).await;
}

/// Read one server's stdout and route every framed message.
async fn read_server(server: &Arc<ServerInstance>, mut stdout: tokio::process::ChildStdout) {
    let mut framer = Framer::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        match stdout.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => framer.push(&chunk[..n]),
        }
        while let Some(text) = framer.next() {
            let Ok(message) = serde_json::from_str::<Value>(&text) else {
                warn!("LSP mux server sent malformed JSON (key {})", server.key);
                continue;
            };
            if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
                continue;
            }
            from_server(server, &message).await;
        }
    }
    // stdout EOF means the server is gone; the child-reap task confirms it.
    if let Some(state) = server.state.upgrade() {
        if !server.stopping.load(Ordering::SeqCst) {
            kill_server(&state, server);
        }
    }
}

async fn from_server(server: &Arc<ServerInstance>, message: &Value) {
    if !has_method(message) {
        handle_server_response(server, message).await;
        return;
    }
    if has_request_id(message) {
        handle_server_request(server, message).await;
        return;
    }
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    if method == "textDocument/publishDiagnostics" {
        let Some(params) = parse_diagnostics(message.get("params")) else {
            return;
        };
        let uri = params
            .get("uri")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        lock(&server.diagnostics).insert(uri, params.clone());
        for session in server.session_snapshot() {
            if session.initialized.load(Ordering::SeqCst) {
                send_diagnostics(&session, &params).await;
            }
        }
        return;
    }
    if method == "$/progress" {
        if let Some(params) = parse_progress(message.get("params")) {
            let token_key = serde_json::to_string(params.get("token").unwrap_or(&Value::Null))
                .unwrap_or_default();
            match params
                .get("value")
                .and_then(|v| v.get("kind"))
                .and_then(Value::as_str)
            {
                Some("begin") => {
                    lock(&server.progress).insert(token_key, params.clone());
                }
                Some("end") => {
                    lock(&server.progress).remove(&token_key);
                }
                _ => {}
            }
        }
        // Falls through to the broadcast, like TS.
    }
    for session in server.session_snapshot() {
        if session.initialized.load(Ordering::SeqCst) {
            session.send(message).await;
        }
    }
}

async fn send_diagnostics(session: &Arc<Session>, params: &Value) {
    session
        .send(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": params,
        }))
        .await;
}

/// Replies go out inline, in server order, like diagnostics/progress on this same read loop: a
/// task per reply cost a spawn each and could reorder a reply against those inline sends.
async fn handle_server_response(server: &Arc<ServerInstance>, message: &Value) {
    let Some(id) = message.get("id").and_then(Value::as_i64) else {
        return;
    };
    let pending = lock(&server.pending).remove(&id);
    let Some(pending) = pending else { return };
    match pending {
        PendingForward::Resolve(tx) => {
            let _ = tx.send(());
        }
        PendingForward::Initialize => {
            server.init_in_flight.store(false, Ordering::SeqCst);
            if message.get("error").is_some() {
                let waiters = std::mem::take(&mut *lock(&server.initialize_waiters));
                for (session_id, client_id) in waiters {
                    if let Some(session) = find_session(server, session_id) {
                        let mut reply = message.clone();
                        reply["id"] = client_id;
                        session.send(&reply).await;
                    }
                }
                if let Some(state) = server.state.upgrade() {
                    let st = Arc::clone(server);
                    kill_server(&state, &st);
                }
                return;
            }
            let result = message.get("result").cloned().unwrap_or(Value::Null);
            *lock(&server.init_result) = Some(result.clone());
            server.init_cached.store(true, Ordering::SeqCst);
            let waiters = std::mem::take(&mut *lock(&server.initialize_waiters));
            for (session_id, client_id) in waiters {
                if let Some(session) = find_session(server, session_id) {
                    session.send(&rpc_result(&client_id, &result)).await;
                }
            }
        }
        PendingForward::Forward {
            session,
            original_id,
            drop,
        } => {
            if drop {
                return;
            }
            let Some(session) = find_session(server, session) else {
                return;
            };
            if session.is_closed() {
                return;
            }
            lock(&session.forwarded_ids).remove(&id_key(&original_id));
            let mut reply = message.clone();
            reply["id"] = original_id;
            session.send(&reply).await;
        }
    }
}

fn find_session(server: &Arc<ServerInstance>, session_id: u64) -> Option<Arc<Session>> {
    lock(&server.sessions)
        .iter()
        .find(|s| s.id == session_id)
        .cloned()
}

async fn handle_server_request(server: &Arc<ServerInstance>, message: &Value) {
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let server_id = message.get("id").cloned().unwrap_or(Value::Null);
    if matches!(
        method,
        "client/registerCapability"
            | "client/unregisterCapability"
            | "window/workDoneProgress/create"
    ) {
        if method == "client/registerCapability" {
            if let Some(registrations) = message
                .get("params")
                .and_then(|p| p.get("registrations"))
                .and_then(Value::as_array)
            {
                let valid: Vec<Value> = registrations
                    .iter()
                    .filter(|r| r.get("id").and_then(Value::as_str).is_some())
                    .cloned()
                    .collect();
                if !valid.is_empty() {
                    lock(&server.registrations).push(json!({ "registrations": valid }));
                }
            }
        } else {
            let raw = message
                .get("params")
                .and_then(|p| {
                    p.get("unregisterations")
                        .or_else(|| p.get("unregistrations"))
                })
                .and_then(Value::as_array);
            if let Some(items) = raw {
                let ids: HashSet<String> = items
                    .iter()
                    .filter_map(|item| item.get("id").and_then(Value::as_str).map(str::to_string))
                    .collect();
                for batch in lock(&server.registrations).iter_mut() {
                    if let Some(regs) = batch.get_mut("registrations").and_then(Value::as_array_mut)
                    {
                        regs.retain(|r| {
                            r.get("id")
                                .and_then(Value::as_str)
                                .is_none_or(|id| !ids.contains(id))
                        });
                    }
                }
            }
        }
        server.write(&rpc_result(&server_id, &Value::Null)).await;
        for session in server.session_snapshot() {
            if session.initialized.load(Ordering::SeqCst) {
                forward_no_relay(&session, server, message).await;
            }
        }
        return;
    }
    // Client-side effects (applyEdit et al) reach exactly one, most recently
    // active session.
    let focus = server
        .session_snapshot()
        .into_iter()
        .max_by_key(|s| s.last_activity.load(Ordering::SeqCst));
    let Some(focus) = focus else {
        server
            .write(&rpc_error(&server_id, -32601, "no client attached"))
            .await;
        return;
    };
    let client_id = server.client_id();
    lock(&focus.pending_client).insert(
        client_id.clone(),
        PendingClient {
            relay: true,
            server_id: Some(server_id),
        },
    );
    let mut forwarded = message.clone();
    forwarded["id"] = json!(client_id);
    focus.send(&forwarded).await;
}

async fn forward_no_relay(session: &Arc<Session>, server: &Arc<ServerInstance>, message: &Value) {
    let client_id = server.client_id();
    lock(&session.pending_client).insert(
        client_id.clone(),
        PendingClient {
            relay: false,
            server_id: None,
        },
    );
    let mut forwarded = message.clone();
    forwarded["id"] = json!(client_id);
    session.send(&forwarded).await;
}

fn handle_client_response(session: &Arc<Session>, message: &Value) {
    let Some(id) = message.get("id").and_then(Value::as_str) else {
        return;
    };
    let Some(pending) = lock(&session.pending_client).remove(id) else {
        return;
    };
    if !pending.relay {
        return;
    }
    let (Some(server_id), Some(server)) = (pending.server_id, lock(&session.server).clone()) else {
        return;
    };
    let mut reply = message.clone();
    reply["id"] = server_id;
    tokio::spawn(async move {
        server.write(&reply).await;
    });
}

/// Replay cached registrations, diagnostics and progress to a freshly
/// initialized session (omp `#replayState`).
async fn replay_state(session: &Arc<Session>, server: &Arc<ServerInstance>) {
    let batches: Vec<Value> = lock(&server.registrations).clone();
    for batch in &batches {
        let Some(regs) = batch.get("registrations").and_then(Value::as_array) else {
            continue;
        };
        if regs.is_empty() {
            continue;
        }
        forward_no_relay(
            session,
            server,
            &json!({
                "jsonrpc": "2.0",
                "id": "replaced",
                "method": "client/registerCapability",
                "params": batch,
            }),
        )
        .await;
    }
    let diagnostics: Vec<Value> = lock(&server.diagnostics).values().cloned().collect();
    for params in &diagnostics {
        send_diagnostics(session, params).await;
    }
    let progress: Vec<Value> = lock(&server.progress).values().cloned().collect();
    for params in &progress {
        session
            .send(&json!({ "jsonrpc": "2.0", "method": "$/progress", "params": params }))
            .await;
    }
}

async fn close_session(state: &Arc<MuxState>, session: &Arc<Session>) {
    if session.closed.swap(true, Ordering::SeqCst) {
        return;
    }
    *session.writer.lock().await = None;
    lock(&state.sessions).remove(&session.id);
    let server = lock(&session.server).take();
    if let Some(server) = server {
        // Teardown writes are best-effort: the server may already be gone.
        let open_uris: Vec<String> = lock(&session.open_uris).drain().collect();
        for uri in &open_uris {
            lock(&server.documents).remove(uri);
            server
                .write(&json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/didClose",
                    "params": { "textDocument": { "uri": uri } },
                }))
                .await;
        }
        // Drop this session's pending forwards and cancel them upstream.
        let mut kept = HashMap::new();
        let pending = std::mem::take(&mut *lock(&server.pending));
        for (mux_id, mut entry) in pending {
            let drop_it = match &mut entry {
                PendingForward::Forward {
                    session: sid, drop, ..
                } if *sid == session.id => {
                    *drop = true;
                    true
                }
                _ => false,
            };
            if drop_it {
                server
                    .write(&json!({ "jsonrpc": "2.0", "method": "$/cancelRequest", "params": { "id": mux_id } }))
                    .await;
            } else {
                kept.insert(mux_id, entry);
            }
        }
        *lock(&server.pending) = kept;
        lock(&server.initialize_waiters).retain(|(sid, _)| *sid != session.id);
        lock(&server.sessions).retain(|s| s.id != session.id);
        if lock(&server.sessions).is_empty() && !server.stopping.load(Ordering::SeqCst) {
            let st = Arc::clone(&server);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(server_linger_ms())).await;
                if lock(&st.sessions).is_empty() && !st.stopping.load(Ordering::SeqCst) {
                    stop_server(&st).await;
                }
            });
        }
    }
    if lock(&state.sessions).is_empty() {
        state.arm_idle();
    }
}

/// The server process exited (crash or kill): sever its links.
fn server_exited(state: &Arc<MuxState>, server: &Arc<ServerInstance>) {
    if server.stopping.swap(true, Ordering::SeqCst) {
        // Still sever sessions if this is the first observation.
    }
    lock(&state.servers).remove(&server.key);
    let pending = std::mem::take(&mut *lock(&server.pending));
    for (_, entry) in pending {
        if let PendingForward::Resolve(tx) = entry {
            let _ = tx.send(());
        }
    }
    for session in lock(&server.sessions).drain(..) {
        let session = Arc::clone(&session);
        tokio::spawn(async move {
            session.destroy().await;
        });
    }
}

/// Graceful `shutdown` + `exit`, then kill (omp `#stopServer`).
async fn stop_server(server: &Arc<ServerInstance>) {
    if server.stopping.swap(true, Ordering::SeqCst) {
        return;
    }
    let (tx, rx) = oneshot::channel::<()>();
    let mux_id = server.mux_id();
    lock(&server.pending).insert(mux_id, PendingForward::Resolve(tx));
    server
        .write(
            &json!({ "jsonrpc": "2.0", "id": mux_id, "method": "shutdown", "params": Value::Null }),
        )
        .await;
    let _ = tokio::time::timeout(Duration::from_millis(SHUTDOWN_BUDGET_MS), rx).await;
    server
        .write(&json!({ "jsonrpc": "2.0", "method": "exit" }))
        .await;
    if let Ok(mut guard) = server.child.lock() {
        if let Some(child) = guard.as_mut() {
            let _ = child.start_kill();
        }
    }
}

fn kill_server(state: &Arc<MuxState>, server: &Arc<ServerInstance>) {
    server.stopping.store(true, Ordering::SeqCst);
    if let Ok(mut guard) = server.child.lock() {
        if let Some(child) = guard.as_mut() {
            let _ = child.start_kill();
        }
    }
    server_exited(state, server);
}
