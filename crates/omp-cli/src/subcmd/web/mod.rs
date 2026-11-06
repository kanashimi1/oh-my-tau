//! `omp-rs web` — the browser UI: a local HTTP server whose single-page app
//! drives real `omp-rs --mode rpc` agents (one child process per open
//! session) over a WebSocket bridge, so everything the rpc protocol offers —
//! streaming turns, tools, slash commands, dialogs, model/thinking control,
//! session switching — is usable from the browser.
//!
//! Agents are detached children: own process group, stdout/stderr logged to
//! `~/.omp-rs/agents/<id>/log`, control frames over `~/.omp-rs/agents/<id>.sock`.
//! They survive a hub restart — the next hub adopts them from the
//! `registry.json` scan (see `detach.rs`).
//!
//! Routes:
//! - `GET /`, `/assets/*`: the embedded app.
//! - `GET /api/info`, `/api/projects`, `/api/sessions?cwd=`, `/api/fs?path=`.
//! - `GET /api/maps`: graph map data from `~/.config/omt/maps.json`.
//! - `GET /api/agents`, `POST /api/agents {cwd, resume?, model?, worktree?}`,
//!   `DELETE /api/agents/<id>`.
//! - `GET /api/export?session=<file>`: HTML export of a stored session.
//! - `POST /api/transcribe?ext=&lang=` (raw audio): local dictation, see `stt.rs`.
//! - `GET /preview/<abs path>` (Range-aware file serving), `GET /api/preview/info?path=`,
//!   `POST /api/preview/office?path=` (Office → PDF), `POST /api/open?path=&reveal=`:
//!   file previews, see `preview.rs`.
//! - `POST /api/uploads?name=` (raw file): composer attachments, see `uploads.rs`.
//! - `GET /api/wallpapers`, `POST /api/wallpapers?name=` (raw image body),
//!   `DELETE /api/wallpapers/<name>`, `GET /wallpapers/custom/<name>`.
//! - `GET /ws/<agentId>` (WebSocket): rpc frames both ways.
//!
//! Local-only: binds loopback by default and rejects cross-origin requests
//! and foreign `Host` headers (a web page must not drive the agent).

mod agent;
mod assets;
mod changes;
mod desktop;
pub(crate) mod detach;
mod preview;
mod search;
mod sessions;
mod stt;
mod terminal;
mod uploads;
mod wallpapers;

use parking_lot::Mutex;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context as _, Result};
use futures::{SinkExt as _, StreamExt as _};
use http_body_util::{BodyExt as _, Full};
use hyper::body::{Bytes, Incoming};
use hyper::header::{HeaderValue, CONNECTION, CONTENT_TYPE, UPGRADE};
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use agent::{Agent, Spawn};

const USAGE: &str = "Usage: omt web [--port <port>] [--host <host>] [--cwd <dir>] [--open]

Start the omt browser UI (default http://127.0.0.1:7467). Each open
session runs as its own `omt --mode rpc` agent.";
const DEFAULT_PORT: u16 = 7467;
/// Request bodies are small JSON objects; images travel over the socket.
const MAX_BODY: usize = 1 << 20;

type Body = Full<Bytes>;

struct Options {
    port: u16,
    host: String,
    cwd: PathBuf,
    open: bool,
}

fn parse_args(args: &[String]) -> Result<Options> {
    let mut opts = Options {
        port: DEFAULT_PORT,
        host: "127.0.0.1".into(),
        cwd: std::env::current_dir().context("current dir")?,
        open: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (arg.as_str(), None),
        };
        let mut value = |name: &str| -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it
                    .next()
                    .cloned()
                    .with_context(|| format!("{name} needs a value")),
            }
        };
        match flag {
            "-p" | "--port" => {
                let v = value("--port")?;
                opts.port = v.parse().with_context(|| format!("Invalid port: {v}"))?;
            }
            "--host" => opts.host = value("--host")?,
            "--cwd" => opts.cwd = PathBuf::from(value("--cwd")?),
            "--open" => opts.open = true,
            other => bail!("Unknown option: {other}\n{USAGE}"),
        }
    }
    opts.cwd = std::fs::canonicalize(&opts.cwd)
        .with_context(|| format!("--cwd {}", opts.cwd.display()))?;
    Ok(opts)
}

struct Hub {
    exe: PathBuf,
    default_cwd: PathBuf,
    port: u16,
    agents: Mutex<HashMap<String, Arc<Agent>>>,
    /// Last "show this agent" request (`POST /api/agents/<id>/focus`, from
    /// `omt` `/desktop`): `{id, ms}`; open pages navigate to newer ones.
    focus: Mutex<Option<Value>>,
}

impl Hub {
    fn agents(&self) -> Vec<Arc<Agent>> {
        let mut list: Vec<Arc<Agent>> = lock(&self.agents).values().cloned().collect();
        list.sort_by_key(|a| a.started_ms);
        list
    }

    fn agent(&self, id: &str) -> Option<Arc<Agent>> {
        lock(&self.agents).get(id).cloned()
    }

    async fn spawn(self: &Arc<Self>, spec: Spawn) -> Result<Arc<Agent>> {
        // Resuming a session that is already open attaches to it instead.
        if let Some(resume) = &spec.resume {
            let canon = std::fs::canonicalize(resume).ok();
            if let Some(a) = self.agents().into_iter().find(|a| {
                a.is_alive()
                    && a.summary().session_file.is_some_and(|f| {
                        f == *resume || std::fs::canonicalize(&f).ok() == canon && canon.is_some()
                    })
            }) {
                return Ok(a);
            }
        }
        let id = crate::persist::uuid_v7()[..13].replace('-', "");
        let hub = Arc::downgrade(self);
        let agent = Agent::spawn(id.clone(), &self.exe, &spec, move |id| {
            if let Some(hub) = hub.upgrade() {
                lock(&hub.agents).remove(id);
            }
            // The child is gone: its registry record and control socket are
            // stale (this also cleans up after DELETE).
            detach::unregister(id);
        })
        .await?;
        // The registry record is what makes the child adoptable after this
        // hub dies (and what DELETE cleans up).
        detach::upsert(detach::Entry {
            id: id.clone(),
            pid: agent.pid,
            pgid: agent.pgid,
            session: spec.resume.clone(),
            approval_mode: spec.approval_mode.clone(),
            socket: detach::socket_path(&id),
            started_ms: agent.started_ms,
            hub_pid: Some(std::process::id()),
        });
        lock(&self.agents).insert(id, Arc::clone(&agent));
        Ok(agent)
    }
}

fn lock<T>(m: &Mutex<T>) -> parking_lot::MutexGuard<'_, T> {
    m.lock()
}

/// Entry point: `omp-rs web [options]`.
///
/// # Errors
/// Bad flags or a failed bind.
#[allow(clippy::disallowed_methods)] // entry point: dedicated runtime on its own thread
pub fn run(args: &[String]) -> Result<i32> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(0);
    }
    let opts = parse_args(args)?;
    // Called from inside the CLI's runtime: serve on a dedicated thread with
    // its own runtime (serve_collab.rs pattern).
    std::thread::spawn(move || -> Result<i32> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("create tokio runtime")?;
        rt.block_on(serve(opts))
    })
    .join()
    .map_err(|_| anyhow::anyhow!("web server thread panicked"))?
}

async fn serve(opts: Options) -> Result<i32> {
    let addr: SocketAddr = format!("{}:{}", opts.host, opts.port)
        .parse()
        .or_else(|_| {
            use std::net::ToSocketAddrs as _;
            (opts.host.as_str(), opts.port)
                .to_socket_addrs()?
                .next()
                .context("no address")
        })
        .with_context(|| format!("resolve {}", opts.host))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr} (is another `omt web` running?)"))?;
    let port = listener.local_addr()?.port();
    let hub = Arc::new(Hub {
        exe: std::env::current_exe().context("current exe")?,
        default_cwd: opts.cwd.clone(),
        port,
        agents: Mutex::new(HashMap::new()),
        focus: Mutex::new(None),
    });
    let adopted = adopt_detached(&hub).await;
    let shown_host = if addr.ip().is_unspecified() {
        "127.0.0.1".to_string()
    } else {
        opts.host.clone()
    };
    let url = format!("http://{shown_host}:{port}/");
    println!("omt web UI: {url}");
    println!("Working directory: {}", opts.cwd.display());
    if adopted > 0 {
        println!("Adopted {adopted} detached agent(s) from the previous run");
    }
    println!("Press Ctrl+C to stop");
    if opts.open {
        open_browser(&url);
    }
    #[cfg(unix)]
    let mut terminate: Option<tokio::signal::unix::Signal> = Some(
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .context("install SIGTERM handler")?,
    );
    #[cfg(not(unix))]
    let mut terminate: Option<()> = None;
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let hub = Arc::clone(&hub);
                tokio::spawn(async move {
                    let service = hyper::service::service_fn(move |req| {
                        let hub = Arc::clone(&hub);
                        async move { Ok::<_, Infallible>(route(hub, req).await) }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(stream), service)
                        .with_upgrades()
                        .await;
                });
            }
            _ = tokio::signal::ctrl_c() => break,
            () = terminated(&mut terminate) => break,
        }
    }
    println!("\nShutting down.");
    // Detached children survive the hub: release ownership (hubPid -> none)
    // and leave the processes, registry records and sockets for the next
    // `omp-rs web` to adopt. Nothing is killed here.
    for agent in hub.agents() {
        detach::set_hub_pid(&agent.id, None);
    }
    println!("Detached agents keep running; restart `omp-rs web` to reattach.");
    Ok(0)
}

/// Startup scan of `~/.omp-rs/agents/registry.json`: live, reachable children
/// whose owning hub is gone (or us) are adopted into the roster; dead ones
/// have their record and socket reclaimed; children owned by another *live*
/// hub are logged and left alone (never two writers to one session).
async fn adopt_detached(hub: &Arc<Hub>) -> usize {
    let my_pid = std::process::id();
    let mut entries = detach::load_registry();
    let mut adopted = 0;
    for i in (0..entries.len()).rev() {
        let id = entries[i].id.clone();
        if let Some(owner) = detach::foreign_owner(&entries[i], my_pid) {
            eprintln!("agent {id} owned by hub {owner}");
            continue;
        }
        if !(detach::pid_alive(entries[i].pid) && detach::socket_connects(&entries[i].socket)) {
            detach::remove_entry(&mut entries, &id);
            continue;
        }
        let weak = Arc::downgrade(hub);
        let on_exit = move |id: &str| {
            if let Some(hub) = weak.upgrade() {
                lock(&hub.agents).remove(id);
            }
            detach::unregister(id);
        };
        match Agent::adopt(&entries[i], on_exit).await {
            Ok(agent) => {
                entries[i].hub_pid = Some(my_pid);
                lock(&hub.agents).insert(id, agent);
                adopted += 1;
            }
            Err(e) => {
                eprintln!("agent {id} not adoptable: {e:#}");
                detach::remove_entry(&mut entries, &id);
            }
        }
    }
    detach::save_registry(&entries);
    adopted
}

/// Resolve on SIGTERM; never resolves where there is no SIGTERM to catch.
#[cfg(unix)]
async fn terminated(terminate: &mut Option<tokio::signal::unix::Signal>) {
    if let Some(signal) = terminate.as_mut() {
        signal.recv().await;
    }
}
#[cfg(not(unix))]
async fn terminated(terminate: &mut Option<()>) {
    let _ = terminate;
    std::future::pending::<()>().await;
}

fn open_browser(url: &str) {
    let candidates: &[(&str, &[&str])] = if std::env::var_os("WSL_DISTRO_NAME").is_some() {
        &[("cmd.exe", &["/c", "start", ""]), ("xdg-open", &[])]
    } else if cfg!(target_os = "macos") {
        &[("open", &[])]
    } else {
        &[("xdg-open", &[])]
    };
    for (bin, pre) in candidates {
        let ok = std::process::Command::new(bin)
            .args(*pre)
            .arg(url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .is_ok();
        if ok {
            return;
        }
    }
}

fn respond(status: StatusCode, content_type: &str, body: impl Into<Bytes>) -> Response<Body> {
    let mut res = Response::new(Full::new(body.into()));
    *res.status_mut() = status;
    let headers = res.headers_mut();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_str(content_type).unwrap_or(HeaderValue::from_static("text/plain")),
    );
    headers.insert("cache-control", HeaderValue::from_static("no-store"));
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    res
}

fn json_response(status: StatusCode, value: &Value) -> Response<Body> {
    respond(status, "application/json; charset=utf-8", value.to_string())
}

fn error(status: StatusCode, message: &str) -> Response<Body> {
    json_response(status, &json!({"error": message}))
}

/// Path of the graph-map data file (`~/.config/omt/maps.json`).
fn maps_path() -> PathBuf {
    std::env::var_os("HOME")
        .map_or_else(|| PathBuf::from("/"), PathBuf::from)
        .join(".config/omt/maps.json")
}

/// `GET /api/maps`: serve `~/.config/omt/maps.json`, if present and valid.
fn maps_json() -> Response<Body> {
    let raw = match std::fs::read(maps_path()) {
        Ok(raw) => raw,
        Err(_) => {
            return error(
                StatusCode::NOT_FOUND,
                "maps.json not found — запусти hypertrail/compose_maps.py",
            )
        }
    };
    match serde_json::from_slice::<Value>(&raw) {
        Ok(v) => json_response(StatusCode::OK, &v),
        Err(e) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("maps.json is not valid JSON: {e}"),
        ),
    }
}

fn query_param(req: &Request<Incoming>, key: &str) -> Option<String> {
    req.uri().query()?.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        (k == key).then(|| percent_decode(v))
    })
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                match hex.map(|h| u8::from_str_radix(h, 16)) {
                    Some(Ok(b)) => {
                        out.push(b);
                        i += 2;
                    }
                    _ => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Same-origin guard: the `Host` must be ours (DNS-rebinding) and a browser
/// `Origin`, when sent, must match it (cross-site pages).
fn same_origin(host: Option<&str>, origin: Option<&str>, port: u16) -> bool {
    let Some(host) = host else { return false };
    let host_name = host.rsplit_once(':').map_or(host, |(h, _)| h);
    let local = matches!(host_name, "127.0.0.1" | "localhost" | "[::1]");
    let host_ok = local || host.ends_with(&format!(":{port}"));
    let origin_ok =
        origin.is_none_or(|o| o == format!("http://{host}") || o == format!("https://{host}"));
    host_ok && origin_ok
}

async fn read_json(req: Request<Incoming>) -> Result<Value> {
    let body = http_body_util::Limited::new(req.into_body(), MAX_BODY)
        .collect()
        .await
        .map_err(|e| anyhow::anyhow!("read body: {e}"))?
        .to_bytes();
    if body.is_empty() {
        return Ok(json!({}));
    }
    Ok(serde_json::from_slice(&body)?)
}

async fn route(hub: Arc<Hub>, req: Request<Incoming>) -> Response<Body> {
    let header = |name: &str| req.headers().get(name).and_then(|v| v.to_str().ok());
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    if method == Method::GET && path.starts_with("/preview/") {
        return serve_preview(&req, &path, hub.port);
    }
    if !same_origin(header("host"), header("origin"), hub.port) {
        return error(StatusCode::FORBIDDEN, "cross-origin request rejected");
    }
    if let Some(id) = path.strip_prefix("/ws/") {
        return websocket(&hub, req, id);
    }
    match (&method, path.as_str()) {
        (&Method::GET, "/api/info") => json_response(
            StatusCode::OK,
            &json!({
                "version": env!("CARGO_PKG_VERSION"),
                "defaultCwd": hub.default_cwd.display().to_string(),
                "home": std::env::var("HOME").unwrap_or_default(),
            }),
        ),
        (&Method::GET, "/api/projects") => {
            json_response(StatusCode::OK, &sessions::list_projects().await)
        }
        (&Method::GET, "/api/maps") => maps_json(),
        (&Method::GET, "/api/sessions") => {
            let cwd =
                query_param(&req, "cwd").map_or_else(|| hub.default_cwd.clone(), PathBuf::from);
            json_response(StatusCode::OK, &sessions::list_sessions(&cwd).await)
        }
        (&Method::POST, "/api/sessions/rename") => rename_session_route(req).await,
        (&Method::GET, "/api/fs") => json_response(
            StatusCode::OK,
            &sessions::browse(query_param(&req, "path").as_deref()),
        ),
        (&Method::GET, "/api/agents") => {
            let list: Vec<Value> = hub.agents().iter().map(|a| a.info_json()).collect();
            let focus = lock(&hub.focus).clone();
            json_response(StatusCode::OK, &json!({"agents": list, "focus": focus}))
        }
        (&Method::GET, "/api/search") => search::api(req).await,
        (_, p) if p.starts_with("/api/changes") => changes::api(req).await,
        (_, p) if p.starts_with("/api/terminal/") => terminal::api(req).await,
        (&Method::POST, "/api/agents") => create_agent(&hub, req).await,
        (&Method::POST, p) if p.starts_with("/api/agents/") => {
            agent_action(&hub, &p["/api/agents/".len()..]).await
        }
        (&Method::DELETE, p) if p.starts_with("/api/agents/") => {
            match hub.agent(&p["/api/agents/".len()..]) {
                Some(agent) => {
                    // SIGTERM the child's process group, then drop its
                    // registry record and control socket (idempotent with the
                    // exit watcher's cleanup).
                    agent.kill().await;
                    detach::unregister(&agent.id);
                    json_response(StatusCode::OK, &json!({"ok": true}))
                }
                None => error(StatusCode::NOT_FOUND, "no such agent"),
            }
        }
        (&Method::GET, "/api/export") => export(query_param(&req, "session")),
        (&Method::POST, "/api/transcribe") => transcribe(req).await,
        (_, "/api/preview/info" | "/api/preview/office" | "/api/open") => preview_api(req).await,
        (&Method::POST, "/api/uploads") => upload_file(req).await,
        (&Method::GET, "/api/wallpapers") => json_response(StatusCode::OK, &wallpapers::list()),
        (&Method::POST, "/api/wallpapers") => upload_wallpaper(req).await,
        (&Method::DELETE, p) if p.starts_with("/api/wallpapers/") => {
            match wallpapers::delete(&percent_decode(&p["/api/wallpapers/".len()..])) {
                Ok(()) => json_response(StatusCode::OK, &json!({"ok": true})),
                Err(e) => error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
            }
        }
        (&Method::GET, p) if p.starts_with("/wallpapers/custom/") => {
            match wallpapers::read(&percent_decode(&p["/wallpapers/custom/".len()..])) {
                Some((mime, bytes)) => {
                    let mut res = respond(StatusCode::OK, mime, bytes);
                    res.headers_mut().insert(
                        "cache-control",
                        HeaderValue::from_static("private, max-age=86400"),
                    );
                    res
                }
                None => error(StatusCode::NOT_FOUND, "not found"),
            }
        }
        (&Method::GET, p) => match assets::get(p) {
            Some((mime, body)) => respond(StatusCode::OK, mime, body),
            None => error(StatusCode::NOT_FOUND, "not found"),
        },
        _ => error(StatusCode::METHOD_NOT_ALLOWED, "method not allowed"),
    }
}

async fn rename_session_route(req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let path = body.get("path").and_then(Value::as_str).map(PathBuf::from);
    let name = body.get("name").and_then(Value::as_str);
    let (Some(path), Some(name)) = (path, name) else {
        return error(StatusCode::BAD_REQUEST, "path and name are required");
    };
    match sessions::rename_session(&path, name) {
        Ok(()) => json_response(StatusCode::OK, &json!({"ok": true})),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}

async fn create_agent(hub: &Arc<Hub>, req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let text = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let cwd = text("cwd").map_or_else(|| hub.default_cwd.clone(), PathBuf::from);
    if !cwd.is_dir() {
        return error(
            StatusCode::BAD_REQUEST,
            &format!("not a directory: {}", cwd.display()),
        );
    }
    let resume = text("resume");
    if let Some(r) = &resume {
        if Path::new(r).is_absolute() && !sessions::is_session_path(Path::new(r)) {
            return error(StatusCode::BAD_REQUEST, "not a session file");
        }
    }
    let approval_mode = match text("approvalMode") {
        Some(m) => match crate::approval::parse_mode(&m) {
            Some(_) => Some(m),
            None => {
                return error(
                    StatusCode::BAD_REQUEST,
                    &format!("invalid approvalMode {m:?}: expected always-ask, write or yolo"),
                )
            }
        },
        None => None,
    };
    let spec = Spawn {
        cwd,
        resume,
        model: text("model"),
        approval_mode,
        role: text("role"),
        worktree: body.get("worktree").and_then(Value::as_bool).unwrap_or(false),
    };
    match hub.spawn(spec).await {
        Ok(agent) => json_response(StatusCode::OK, &agent.info_json()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

/// `POST /api/agents/<id>/{focus,terminal,abort}`: show the agent in the
/// desktop app, continue it in a terminal (`omt attach <id>`), or abort its
/// run (also cancels its background `task` children).
async fn agent_action(hub: &Hub, rest: &str) -> Response<Body> {
    let Some((id, action)) = rest.split_once('/') else {
        return error(StatusCode::NOT_FOUND, "not found");
    };
    let Some(agent) = hub.agent(id) else {
        return error(StatusCode::NOT_FOUND, "no such agent");
    };
    let result = match action {
        // Show the agent in the desktop app: open pages pick the
        // request up from `GET /api/agents`, the window is raised.
        "focus" => {
            *lock(&hub.focus) = Some(json!({"id": agent.id, "ms": pa_agent::now_ms()}));
            desktop::raise()
        }
        // Continue the agent in a terminal (`omt attach <id>`).
        "terminal" => desktop::open_terminal(&hub.exe, &agent.cwd, &agent.id),
        // A parked parent waiting on a stuck subagent: the rpc `abort` kills
        // the session's `task` children and ends the park.
        "abort" => {
            agent
                .send_line(&json!({"id": "web-abort", "type": "abort"}).to_string())
                .await
        }
        _ => return error(StatusCode::NOT_FOUND, "not found"),
    };
    match result {
        Ok(()) => json_response(StatusCode::OK, &json!({"ok": true})),
        Err(e) => error(StatusCode::BAD_GATEWAY, &format!("{e:#}")),
    }
}

/// `GET /preview/<path>`: sandboxed preview pages load their assets with
/// `Origin: null`. Read-only and without CORS headers, so a foreign page
/// still cannot read the files.
fn serve_preview(req: &Request<Incoming>, path: &str, port: u16) -> Response<Body> {
    let header = |name: &str| req.headers().get(name).and_then(|v| v.to_str().ok());
    let host_ok = same_origin(header("host"), None, port);
    let origin_ok = header("origin").is_none_or(|o| o == "null")
        || same_origin(header("host"), header("origin"), port);
    if !(host_ok && origin_ok) {
        return error(StatusCode::FORBIDDEN, "cross-origin request rejected");
    }
    let file = preview::host_path(&percent_decode(&path["/preview".len()..]));
    let embed = query_param(req, "embed").is_some_and(|v| v == "1");
    preview::serve(&file, header("range"), embed)
}

/// `/api/preview/info`, `/api/preview/office`, `/api/open` (see `preview.rs`).
async fn preview_api(req: Request<Incoming>) -> Response<Body> {
    let Some(raw) = query_param(&req, "path") else {
        return error(StatusCode::BAD_REQUEST, "missing path");
    };
    let file = preview::host_path(&raw);
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/api/preview/info") => match std::fs::metadata(&file) {
            Ok(m) if m.is_file() => json_response(
                StatusCode::OK,
                &json!({"path": file.display().to_string(), "size": m.len(),
                        "mtimeMs": m.modified().ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map_or(0, |d| d.as_millis()),
                        "mime": preview::mime(&file), "office": preview::is_office(&file)}),
            ),
            Ok(_) => error(StatusCode::BAD_REQUEST, "not a file"),
            Err(e) => error(StatusCode::NOT_FOUND, &format!("{}: {e}", file.display())),
        },
        (&Method::POST, "/api/preview/office") => match preview::office_pdf(&file).await {
            Ok(v) => json_response(StatusCode::OK, &v),
            Err(e) => error(StatusCode::BAD_GATEWAY, &format!("{e:#}")),
        },
        (&Method::POST, "/api/open") => {
            let reveal = query_param(&req, "reveal").is_some_and(|v| v == "1");
            match preview::open_in_windows(&file, reveal) {
                Ok(()) => json_response(StatusCode::OK, &json!({"ok": true})),
                Err(e) => error(StatusCode::BAD_GATEWAY, &format!("{e:#}")),
            }
        }
        _ => error(StatusCode::METHOD_NOT_ALLOWED, "method not allowed"),
    }
}

/// Dictation: raw audio body (`?ext=webm&lang=ru`) → `{text, language}`.
async fn transcribe(req: Request<Incoming>) -> Response<Body> {
    let ext = query_param(&req, "ext").unwrap_or_else(|| "webm".to_owned());
    let lang = query_param(&req, "lang").filter(|l| !l.is_empty() && l != "auto");
    let body = match http_body_util::Limited::new(req.into_body(), stt::MAX_AUDIO)
        .collect()
        .await
    {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, &format!("recording: {e}")),
    };
    if body.is_empty() {
        return error(StatusCode::BAD_REQUEST, "empty recording");
    }
    match stt::transcribe(&body, &ext, lang.as_deref()).await {
        Ok(v) => json_response(StatusCode::OK, &v),
        Err(e) => error(StatusCode::BAD_GATEWAY, &format!("{e:#}")),
    }
}

/// Composer attachment: raw file body (`?name=`) → `{path, name, size}`.
async fn upload_file(req: Request<Incoming>) -> Response<Body> {
    let name = query_param(&req, "name").unwrap_or_default();
    let body = match http_body_util::Limited::new(req.into_body(), uploads::MAX_UPLOAD)
        .collect()
        .await
    {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, &format!("upload: {e}")),
    };
    match uploads::save(&name, &body) {
        Ok(v) => json_response(StatusCode::OK, &v),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

async fn upload_wallpaper(req: Request<Incoming>) -> Response<Body> {
    let name = query_param(&req, "name").unwrap_or_default();
    let body = match http_body_util::Limited::new(req.into_body(), wallpapers::MAX_UPLOAD)
        .collect()
        .await
    {
        Ok(b) => b.to_bytes(),
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, &format!("upload: {e}")),
    };
    match wallpapers::save(&name, &body) {
        Ok(entry) => json_response(StatusCode::OK, &entry),
        Err(e) => error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    }
}

fn export(session: Option<String>) -> Response<Body> {
    let Some(session) = session else {
        return error(StatusCode::BAD_REQUEST, "missing session");
    };
    let path = PathBuf::from(&session);
    if !sessions::is_session_path(&path) {
        return error(StatusCode::BAD_REQUEST, "not a session file");
    }
    match crate::subcmd::export::render_html(&path) {
        Ok(html) => respond(StatusCode::OK, "text/html; charset=utf-8", html),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

fn websocket(hub: &Hub, req: Request<Incoming>, id: &str) -> Response<Body> {
    let Some(agent) = hub.agent(id) else {
        return error(StatusCode::NOT_FOUND, "no such agent");
    };
    let upgrade = req
        .headers()
        .get(UPGRADE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
    let Some(key) = req
        .headers()
        .get("sec-websocket-key")
        .map(|k| derive_accept_key(k.as_bytes()))
    else {
        return error(StatusCode::BAD_REQUEST, "missing sec-websocket-key");
    };
    if !upgrade {
        return error(StatusCode::BAD_REQUEST, "expected a websocket upgrade");
    }
    tokio::spawn(async move {
        let Ok(upgraded) = hyper::upgrade::on(req).await else {
            return;
        };
        let ws = WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, None).await;
        bridge(agent, ws).await;
    });
    let mut res = Response::new(Full::new(Bytes::new()));
    *res.status_mut() = StatusCode::SWITCHING_PROTOCOLS;
    let headers = res.headers_mut();
    headers.insert(UPGRADE, HeaderValue::from_static("websocket"));
    headers.insert(CONNECTION, HeaderValue::from_static("Upgrade"));
    if let Ok(v) = HeaderValue::from_str(&key) {
        headers.insert("sec-websocket-accept", v);
    }
    res
}

/// Server ping interval for browser sockets. Browsers answer with a pong at
/// the network layer, so an otherwise idle socket (agent parked waiting for
/// subagents) keeps traffic flowing and a half-open peer is noticed on send.
const WS_PING_EVERY: std::time::Duration = std::time::Duration::from_secs(20);

/// Pump one browser socket <-> one agent until either side ends. Every close
/// is logged to stderr (journal) with its reason, age and frame counts.
async fn bridge(agent: Arc<Agent>, ws: WebSocketStream<TokioIo<hyper::upgrade::Upgraded>>) {
    let started = std::time::Instant::now();
    let (mut out_frames, mut in_frames, mut lagged_total) = (0u64, 0u64, 0u64);
    eprintln!("omt web: ws open agent={}", agent.id);
    let mut rx = agent.subscribe();
    let (mut sink, mut stream) = ws.split();
    let hello = json!({"type": "web_attached", "agent": agent.info_json()});
    if sink.send(Message::text(hello.to_string())).await.is_err() {
        eprintln!("omt web: ws close agent={} reason=hello send failed", agent.id);
        return;
    }
    for frame in agent.replay_frames() {
        if sink.send(Message::text(frame.to_string())).await.is_err() {
            eprintln!("omt web: ws close agent={} reason=replay send failed", agent.id);
            return;
        }
    }
    if !agent.is_alive() {
        let _ = sink
            .send(Message::text(
                json!({"type": "web_agent_exit", "code": null}).to_string(),
            ))
            .await;
    }
    let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + WS_PING_EVERY, WS_PING_EVERY);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let reason: String = loop {
        tokio::select! {
            frame = rx.recv() => match frame {
                Ok(line) => {
                    if sink.send(Message::text(line.to_string())).await.is_err() {
                        break "send to browser failed".to_string();
                    }
                    out_frames += 1;
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    lagged_total += n;
                    let note = json!({"type": "web_lagged", "skipped": n});
                    if sink.send(Message::text(note.to_string())).await.is_err() {
                        break "send to browser failed (lag note)".to_string();
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    break "agent event channel closed".to_string();
                }
            },
            _ = ping.tick() => {
                if sink.send(Message::Ping(Default::default())).await.is_err() {
                    break "ping send failed".to_string();
                }
            }
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    in_frames += 1;
                    if let Err(e) = agent.send_line(text.as_str()).await {
                        let note = json!({"type": "web_error", "message": format!("agent stdin: {e}")});
                        let _ = sink.send(Message::text(note.to_string())).await;
                    }
                }
                Some(Ok(Message::Ping(p))) => {
                    let _ = sink.send(Message::Pong(p)).await;
                }
                Some(Ok(Message::Close(frame))) => {
                    break match frame {
                        Some(f) => format!("browser sent close code={} reason={:?}", u16::from(f.code), f.reason.as_str()),
                        None => "browser sent close".to_string(),
                    };
                }
                Some(Err(e)) => break format!("socket error: {e}"),
                None => break "browser socket ended without close frame".to_string(),
                Some(Ok(_)) => {}
            },
        }
    };
    eprintln!(
        "omt web: ws close agent={} reason={} age={}s out={} in={} lagged={}",
        agent.id,
        reason,
        started.elapsed().as_secs(),
        out_frames,
        in_frames,
        lagged_total
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_site_pages_cannot_drive_the_agent() {
        let host = Some("127.0.0.1:7467");
        assert!(same_origin(host, None, 7467));
        assert!(same_origin(host, Some("http://127.0.0.1:7467"), 7467));
        assert!(!same_origin(host, Some("https://evil.example"), 7467));
        assert!(!same_origin(host, Some("http://localhost:7467"), 7467));
        // DNS rebinding: a foreign name resolving to loopback.
        assert!(!same_origin(
            Some("evil.example:80"),
            Some("http://evil.example:80"),
            7467
        ));
        assert!(!same_origin(None, None, 7467));
    }

    #[test]
    fn query_values_are_percent_decoded() {
        assert_eq!(percent_decode("%2Fhome%2Fme%20x+y"), "/home/me x y");
        assert_eq!(percent_decode("bad%zz%"), "bad%zz%");
        assert_eq!(percent_decode("%D1%8F"), "я");
    }
}
