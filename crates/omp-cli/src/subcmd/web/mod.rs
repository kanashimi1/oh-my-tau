//! `omp-rs web` — the browser UI: a local HTTP server whose single-page app
//! drives real `omp-rs --mode rpc` agents (one child process per open
//! session) over a WebSocket bridge, so everything the rpc protocol offers —
//! streaming turns, tools, slash commands, dialogs, model/thinking control,
//! session switching — is usable from the browser.
//!
//! Routes:
//! - `GET /`, `/assets/*`: the embedded app.
//! - `GET /api/info`, `/api/projects`, `/api/sessions?cwd=`, `/api/fs?path=`.
//! - `GET /api/agents`, `POST /api/agents {cwd, resume?, model?}`,
//!   `DELETE /api/agents/<id>`.
//! - `GET /api/export?session=<file>`: HTML export of a stored session.
//! - `GET /api/wallpapers`, `POST /api/wallpapers?name=` (raw image body),
//!   `DELETE /api/wallpapers/<name>`, `GET /wallpapers/custom/<name>`.
//! - `GET /ws/<agentId>` (WebSocket): rpc frames both ways.
//!
//! Local-only: binds loopback by default and rejects cross-origin requests
//! and foreign `Host` headers (a web page must not drive the agent).

mod agent;
mod assets;
mod sessions;
mod wallpapers;

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
                None => it.next().cloned().with_context(|| format!("{name} needs a value")),
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
        })
        .await?;
        lock(&self.agents).insert(id, Arc::clone(&agent));
        Ok(agent)
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Entry point: `omp-rs web [options]`.
///
/// # Errors
/// Bad flags or a failed bind.
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
    });
    let shown_host = if addr.ip().is_unspecified() { "127.0.0.1".to_string() } else { opts.host.clone() };
    let url = format!("http://{shown_host}:{port}/");
    println!("omt web UI: {url}");
    println!("Working directory: {}", opts.cwd.display());
    println!("Press Ctrl+C to stop");
    if opts.open {
        open_browser(&url);
    }
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
        }
    }
    println!("\nShutting down.");
    for agent in hub.agents() {
        agent.kill().await;
    }
    Ok(0)
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
    headers.insert(CONTENT_TYPE, HeaderValue::from_str(content_type).unwrap_or(HeaderValue::from_static("text/plain")));
    headers.insert("cache-control", HeaderValue::from_static("no-store"));
    headers.insert("x-content-type-options", HeaderValue::from_static("nosniff"));
    res
}

fn json_response(status: StatusCode, value: &Value) -> Response<Body> {
    respond(status, "application/json; charset=utf-8", value.to_string())
}

fn error(status: StatusCode, message: &str) -> Response<Body> {
    json_response(status, &json!({"error": message}))
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
    let origin_ok = origin.is_none_or(|o| o == format!("http://{host}") || o == format!("https://{host}"));
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
    if !same_origin(header("host"), header("origin"), hub.port) {
        return error(StatusCode::FORBIDDEN, "cross-origin request rejected");
    }
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    if let Some(id) = path.strip_prefix("/ws/") {
        return websocket(hub, req, id.to_string());
    }
    match (&method, path.as_str()) {
        (&Method::GET, "/api/info") => json_response(StatusCode::OK, &json!({
            "version": env!("CARGO_PKG_VERSION"),
            "defaultCwd": hub.default_cwd.display().to_string(),
            "home": std::env::var("HOME").unwrap_or_default(),
        })),
        (&Method::GET, "/api/projects") => json_response(StatusCode::OK, &sessions::list_projects()),
        (&Method::GET, "/api/sessions") => {
            let cwd = query_param(&req, "cwd").map_or_else(|| hub.default_cwd.clone(), PathBuf::from);
            json_response(StatusCode::OK, &sessions::list_sessions(&cwd))
        }
        (&Method::GET, "/api/fs") => {
            json_response(StatusCode::OK, &sessions::browse(query_param(&req, "path").as_deref()))
        }
        (&Method::GET, "/api/agents") => {
            let list: Vec<Value> = hub.agents().iter().map(|a| a.info_json()).collect();
            json_response(StatusCode::OK, &json!({"agents": list}))
        }
        (&Method::POST, "/api/agents") => create_agent(&hub, req).await,
        (&Method::DELETE, p) if p.starts_with("/api/agents/") => {
            match hub.agent(&p["/api/agents/".len()..]) {
                Some(agent) => {
                    agent.kill().await;
                    json_response(StatusCode::OK, &json!({"ok": true}))
                }
                None => error(StatusCode::NOT_FOUND, "no such agent"),
            }
        }
        (&Method::GET, "/api/export") => export(query_param(&req, "session")),
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
                    res.headers_mut().insert("cache-control", HeaderValue::from_static("private, max-age=86400"));
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

async fn create_agent(hub: &Arc<Hub>, req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("{e:#}")),
    };
    let text = |k: &str| body.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_owned);
    let cwd = text("cwd").map_or_else(|| hub.default_cwd.clone(), PathBuf::from);
    if !cwd.is_dir() {
        return error(StatusCode::BAD_REQUEST, &format!("not a directory: {}", cwd.display()));
    }
    let resume = text("resume");
    if let Some(r) = &resume {
        if Path::new(r).is_absolute() && !sessions::is_session_path(Path::new(r)) {
            return error(StatusCode::BAD_REQUEST, "not a session file");
        }
    }
    let spec = Spawn { cwd, resume, model: text("model") };
    match hub.spawn(spec).await {
        Ok(agent) => json_response(StatusCode::OK, &agent.info_json()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

async fn upload_wallpaper(req: Request<Incoming>) -> Response<Body> {
    let name = query_param(&req, "name").unwrap_or_default();
    let body = match http_body_util::Limited::new(req.into_body(), wallpapers::MAX_UPLOAD).collect().await {
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

fn websocket(hub: Arc<Hub>, req: Request<Incoming>, id: String) -> Response<Body> {
    let Some(agent) = hub.agent(&id) else {
        return error(StatusCode::NOT_FOUND, "no such agent");
    };
    let upgrade = req
        .headers()
        .get(UPGRADE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
    let Some(key) = req.headers().get("sec-websocket-key").map(|k| derive_accept_key(k.as_bytes())) else {
        return error(StatusCode::BAD_REQUEST, "missing sec-websocket-key");
    };
    if !upgrade {
        return error(StatusCode::BAD_REQUEST, "expected a websocket upgrade");
    }
    tokio::spawn(async move {
        let Ok(upgraded) = hyper::upgrade::on(req).await else { return };
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

/// Pump one browser socket <-> one agent until either side ends.
async fn bridge(agent: Arc<Agent>, ws: WebSocketStream<TokioIo<hyper::upgrade::Upgraded>>) {
    let mut rx = agent.subscribe();
    let (mut sink, mut stream) = ws.split();
    let hello = json!({"type": "web_attached", "agent": agent.info_json()});
    if sink.send(Message::text(hello.to_string())).await.is_err() {
        return;
    }
    for frame in agent.replay_frames() {
        if sink.send(Message::text(frame.to_string())).await.is_err() {
            return;
        }
    }
    if !agent.is_alive() {
        let _ = sink.send(Message::text(json!({"type": "web_agent_exit", "code": null}).to_string())).await;
    }
    loop {
        tokio::select! {
            frame = rx.recv() => match frame {
                Ok(line) => {
                    if sink.send(Message::text(line.to_string())).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    let note = json!({"type": "web_lagged", "skipped": n});
                    if sink.send(Message::text(note.to_string())).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    if let Err(e) = agent.send_line(text.as_str()).await {
                        let note = json!({"type": "web_error", "message": format!("agent stdin: {e}")});
                        let _ = sink.send(Message::text(note.to_string())).await;
                    }
                }
                Some(Ok(Message::Ping(p))) => {
                    let _ = sink.send(Message::Pong(p)).await;
                }
                Some(Ok(Message::Close(_)) | Err(_)) | None => break,
                Some(Ok(_)) => {}
            },
        }
    }
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
        assert!(!same_origin(Some("evil.example:80"), Some("http://evil.example:80"), 7467));
        assert!(!same_origin(None, None, 7467));
    }

    #[test]
    fn query_values_are_percent_decoded() {
        assert_eq!(percent_decode("%2Fhome%2Fme%20x+y"), "/home/me x y");
        assert_eq!(percent_decode("bad%zz%"), "bad%zz%");
        assert_eq!(percent_decode("%D1%8F"), "я");
    }
}
