//! Integrated terminal for the web UI: real interactive shells (WSL/bash and
//! PowerShell) spawned as our own children with piped stdio, driven over
//! small REST endpoints (`/api/terminal/*`, all routed here by `mod.rs`).
//!
//! Memory is bounded per terminal: reader threads fold output into lines and
//! a ~2 MiB ring retains the tail; older bytes are dropped (the client gets a
//! `truncated` flag and resynchronizes from `nextCursor`). Only shells this
//! module spawned are ever killed — the registry holds the sole `Child`
//! handles. No user input ever reaches a command line: shells are started
//! with fixed argv, user text flows through stdin only.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode};
use parking_lot::Mutex;
use serde_json::{json, Value};

use super::{error, json_response, query_param, read_json, Body};

/// Retained output tail per terminal (about 2 MiB).
const RING_BYTES: usize = 2 << 20;
/// Lines longer than this are flushed anyway (protects memory on binaries).
const MAX_LINE: usize = 64 << 10;
/// Single `input` writes above this are rejected.
const MAX_INPUT: usize = 64 << 10;
/// Upper bound on live registry entries; the oldest exited ones are pruned.
const MAX_TERMINALS: usize = 24;

fn lock<T>(m: &Mutex<T>) -> parking_lot::MutexGuard<'_, T> {
    m.lock()
}

fn registry() -> &'static Mutex<HashMap<String, Arc<Terminal>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Terminal>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Bounded output tail: complete UTF-8-safe byte chunks with absolute
/// offsets, oldest dropped once `retained` would exceed [`RING_BYTES`].
struct Ring {
    chunks: std::collections::VecDeque<Vec<u8>>,
    retained: usize,
    produced: u64,
}

impl Ring {
    fn new() -> Self {
        Self {
            chunks: std::collections::VecDeque::new(),
            retained: 0,
            produced: 0,
        }
    }

    fn push(&mut self, bytes: Vec<u8>) {
        self.produced += bytes.len() as u64;
        self.retained += bytes.len();
        self.chunks.push_back(bytes);
        while self.retained > RING_BYTES {
            let Some(front) = self.chunks.front_mut() else {
                break;
            };
            let drop = (self.retained - RING_BYTES).min(front.len());
            front.drain(..drop);
            self.retained -= drop;
            if front.is_empty() {
                self.chunks.pop_front();
            }
        }
    }

    /// Offset of the oldest retained byte.
    fn base(&self) -> u64 {
        self.produced - self.retained as u64
    }

    /// Bytes from `cursor` (clamped to what is still retained), whether the
    /// requested range was partially dropped, and the cursor to resume from.
    fn since(&self, cursor: u64) -> (Vec<u8>, bool, u64) {
        let base = self.base();
        let truncated = cursor < base;
        let cursor = cursor.clamp(base, self.produced);
        let mut out = Vec::new();
        // Walk chunks with their absolute start offsets.
        let mut offset = base;
        for chunk in &self.chunks {
            let start = offset;
            offset += chunk.len() as u64;
            if start + chunk.len() as u64 <= cursor {
                continue;
            }
            let skip = cursor.saturating_sub(start) as usize;
            out.extend_from_slice(&chunk[skip..]);
        }
        (out, truncated, self.produced)
    }
}

#[cfg(test)]
mod ring_tests {
    use super::*;

    #[test]
    fn cursor_is_clamped_and_tail_stays_bounded() {
        let mut ring = Ring::new();
        ring.push(b"hello".to_vec());
        let (out, truncated, next) = ring.since(u64::MAX);
        assert!(out.is_empty());
        assert!(!truncated);
        assert_eq!(next, 5);

        ring.push(b"world".to_vec());
        assert_eq!(ring.since(2).0, b"lloworld");
        ring.chunks.clear();
        ring.retained = 0;
        ring.produced = 0;
        ring.push(b"hello".to_vec());

        ring.push(vec![b'x'; RING_BYTES]);
        let (out, truncated, next) = ring.since(0);
        assert!(truncated);
        assert_eq!(out.len(), RING_BYTES);
        assert_eq!(next, RING_BYTES as u64 + 5);
        assert!(ring.since(next).0.is_empty());
    }
}

struct Terminal {
    shell: String,
    title: Mutex<String>,
    cwd: PathBuf,
    stdin: Mutex<Option<ChildStdin>>,
    child: Mutex<Option<Child>>,
    ring: Mutex<Ring>,
    running: AtomicBool,
    exit: Mutex<Option<i32>>,
    created_ms: u64,
}

impl Terminal {
    fn status(&self, id: &str) -> Value {
        json!({
            "id": id,
            "shell": self.shell,
            "title": lock(&self.title).clone(),
            "cwd": self.cwd.display().to_string(),
            "running": self.running.load(Ordering::SeqCst),
            "exitCode": *lock(&self.exit),
            "createdMs": self.created_ms,
        })
    }
}

/// Absolute offset walking helper shared by `Ring::since` (kept trivial so the
/// ring stays provably bounded).
fn ring_since(ring: &Ring, cursor: u64) -> (Vec<u8>, bool, u64) {
    ring.since(cursor)
}

/// Resolve a shell kind to `(program, args)` without any user-controlled argv.
/// All discovery is by fixed candidate paths; user text never reaches here.
fn resolve(shell: &str) -> (String, Vec<&'static str>) {
    // systemd user services do not necessarily inherit WSL_DISTRO_NAME.
    let in_wsl = std::env::var_os("WSL_DISTRO_NAME").is_some()
        || std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| s.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false);
    match shell {
        "powershell" => {
            const UTF8_ARGS: &[&str] = &[
                "-NoLogo",
                "-NoExit",
                "-Command",
                "[Console]::InputEncoding=[Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)",
            ];
            let candidates: &[&str] = if cfg!(windows) {
                &["powershell.exe", "pwsh.exe"]
            } else if in_wsl {
                &[
                    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    "pwsh.exe",
                    "powershell.exe",
                    "pwsh",
                ]
            } else {
                &["pwsh", "powershell"]
            };
            for c in candidates {
                let path = PathBuf::from(c);
                if path.is_absolute() && path.exists() {
                    return (c.to_string(), UTF8_ARGS.to_vec());
                }
                if !path.is_absolute() {
                    // Relative: let Command's PATH lookup try it; prefer an
                    // existing file check for the interop path only.
                    return (c.to_string(), UTF8_ARGS.to_vec());
                }
            }
            (candidates[0].to_string(), UTF8_ARGS.to_vec())
        }
        "wsl" => {
            if in_wsl {
                // Already inside WSL: a "wsl" tab is just the user's shell.
                let sh = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".into());
                (sh, vec!["-i"])
            } else {
                ("wsl.exe".into(), vec![])
            }
        }
        // "bash" and anything else: the user's POSIX shell.
        _ => {
            let sh = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".into());
            (sh, vec!["-i"])
        }
    }
}

fn valid_cwd(raw: &str) -> Option<PathBuf> {
    let path = PathBuf::from(raw);
    let canon = std::fs::canonicalize(&path).ok()?;
    canon.is_dir().then_some(canon)
}

fn spawn(id: &str, shell: &str, cwd: PathBuf, title: &str) -> Result<Arc<Terminal>, String> {
    let (program, args) = resolve(shell);
    let mut cmd = Command::new(&program);
    cmd.args(&args)
        .current_dir(&cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("spawn {program}: {e}"))?;
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let term = Arc::new(Terminal {
        shell: shell.to_string(),
        title: Mutex::new(title.to_string()),
        cwd,
        stdin: Mutex::new(stdin),
        child: Mutex::new(Some(child)),
        ring: Mutex::new(Ring::new()),
        running: AtomicBool::new(true),
        exit: Mutex::new(None),
        created_ms: now_ms(),
    });

    let streams: Vec<Box<dyn Read + Send>> = vec![
        stdout.map(|s| Box::new(s) as Box<dyn Read + Send>),
        stderr.map(|s| Box::new(s) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    .collect();
    for stream in streams {
        let term = Arc::clone(&term);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stream);
            let mut buf = Vec::with_capacity(4096);
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        // Split overly long runs so memory stays bounded.
                        while buf.len() > MAX_LINE {
                            let rest = buf.split_off(MAX_LINE);
                            lock(&term.ring).push(std::mem::take(&mut buf));
                            buf = rest;
                        }
                        lock(&term.ring).push(buf.clone());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        });
    }

    // Exit watcher: only this module's `Child` handle is polled/killed.
    let watched = Arc::clone(&term);
    let id = id.to_string();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        let mut guard = lock(&watched.child);
        if let Some(child) = guard.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let code = status.code().unwrap_or(-1);
                    *lock(&watched.exit) = Some(code);
                    watched.running.store(false, Ordering::SeqCst);
                    drop(guard);
                    lock(registry()).remove(&id);
                    break;
                }
                Ok(None) => {}
                Err(_) => {
                    watched.running.store(false, Ordering::SeqCst);
                    drop(guard);
                    lock(registry()).remove(&id);
                    break;
                }
            }
        } else {
            break;
        }
    });

    Ok(term)
}

fn prune(max: usize) {
    let mut reg = lock(registry());
    if reg.len() <= max {
        return;
    }
    let mut exited: Vec<(String, u64)> = reg
        .iter()
        .filter(|(_, t)| !t.running.load(Ordering::SeqCst))
        .map(|(id, t)| (id.clone(), t.created_ms))
        .collect();
    exited.sort_by_key(|(_, ms)| *ms);
    for (id, _) in exited {
        if reg.len() <= max {
            break;
        }
        reg.remove(&id);
    }
}

/// POST /api/terminal/create `{shell, cwd, title?}`.
async fn create(req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad body: {e}")),
    };
    let shell = body
        .get("shell")
        .and_then(Value::as_str)
        .unwrap_or("bash")
        .to_string();
    if !matches!(shell.as_str(), "bash" | "wsl" | "powershell") {
        return error(
            StatusCode::BAD_REQUEST,
            "shell must be bash, wsl or powershell",
        );
    }
    let cwd_raw = body
        .get("cwd")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let Some(cwd) = valid_cwd(&cwd_raw) else {
        return error(StatusCode::BAD_REQUEST, "cwd must be an existing directory");
    };
    let title = body
        .get("title")
        .and_then(Value::as_str)
        .filter(|t| !t.trim().is_empty() && t.chars().count() <= 80)
        .unwrap_or(&shell)
        .to_string();

    prune(MAX_TERMINALS - 1);
    let id = crate::persist::uuid_v7()[..12].replace('-', "");
    match spawn(&id, &shell, cwd, &title) {
        Ok(term) => {
            lock(registry()).insert(id.clone(), Arc::clone(&term));
            json_response(StatusCode::OK, &term.status(&id))
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e),
    }
}

/// GET /api/terminal/list.
fn list() -> Response<Body> {
    prune(MAX_TERMINALS);
    let reg = lock(registry());
    let mut ids: Vec<&String> = reg.keys().collect();
    ids.sort();
    let items: Vec<Value> = ids
        .iter()
        .filter_map(|id| reg.get(*id).map(|t| t.status(id)))
        .collect();
    json_response(StatusCode::OK, &json!({ "terminals": items }))
}

/// GET /api/terminal/output?id=&cursor= → `{data, nextCursor, truncated, running, exitCode}`.
fn output(req: &Request<Incoming>) -> Response<Body> {
    let Some(id) = query_param(req, "id") else {
        return error(StatusCode::BAD_REQUEST, "id required");
    };
    let cursor = query_param(req, "cursor")
        .and_then(|c| c.parse::<u64>().ok())
        .unwrap_or(0);
    let reg = lock(registry());
    let Some(term) = reg.get(&id) else {
        return error(StatusCode::NOT_FOUND, "unknown terminal");
    };
    let (bytes, truncated, next_cursor) = {
        let ring = lock(&term.ring);
        ring_since(&ring, cursor)
    };
    json_response(
        StatusCode::OK,
        &json!({
            "data": String::from_utf8_lossy(&bytes),
            "nextCursor": next_cursor,
            "truncated": truncated,
            "running": term.running.load(Ordering::SeqCst),
            "exitCode": *lock(&term.exit),
        }),
    )
}

/// POST /api/terminal/input `{id, data}` — text goes to stdin only.
async fn input(req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad body: {e}")),
    };
    let Some(id) = body.get("id").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "id required");
    };
    let data = body.get("data").and_then(Value::as_str).unwrap_or("");
    if data.len() > MAX_INPUT {
        return error(StatusCode::BAD_REQUEST, "input too large");
    }
    let term = lock(registry()).get(id).cloned();
    let Some(term) = term else {
        return error(StatusCode::NOT_FOUND, "unknown terminal");
    };
    let mut guard = lock(&term.stdin);
    match guard.as_mut() {
        Some(stdin) => match stdin.write_all(data.as_bytes()).and_then(|_| stdin.flush()) {
            Ok(()) => json_response(StatusCode::OK, &json!({"ok": true})),
            Err(e) => {
                *guard = None; // broken pipe: the shell is gone
                error(StatusCode::CONFLICT, &format!("write: {e}"))
            }
        },
        None => error(StatusCode::CONFLICT, "terminal stdin is closed"),
    }
}

/// POST /api/terminal/kill `{id}` — kills only a shell we spawned ourselves.
async fn kill(req: Request<Incoming>) -> Response<Body> {
    let body = match read_json(req).await {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad body: {e}")),
    };
    let Some(id) = body.get("id").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "id required");
    };
    let term = lock(registry()).get(id).cloned();
    let Some(term) = term else {
        return error(StatusCode::NOT_FOUND, "unknown terminal");
    };
    // Closing stdin first lets interactive shells exit on their own; then the
    // owned child is killed outright.
    *lock(&term.stdin) = None;
    let mut guard = lock(&term.child);
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
        *lock(&term.exit) = Some(code);
    }
    term.running.store(false, Ordering::SeqCst);
    drop(guard);
    lock(registry()).remove(id);
    json_response(StatusCode::OK, &json!({"ok": true}))
}

/// `ls /proc/<pid>` snapshot: processes (with a "ours" flag for descendants of
/// this server) and listening TCP ports. Fixed file reads — nothing user
/// supplied reaches a command line.
fn system() -> Response<Body> {
    #[cfg(target_os = "linux")]
    {
        let my_pid = std::process::id();
        let mut procs: Vec<(u32, u32, String)> = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for entry in entries.flatten() {
                let Ok(name) = entry.file_name().into_string() else {
                    continue;
                };
                let Ok(pid) = name.parse::<u32>() else {
                    continue;
                };
                let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
                    continue;
                };
                // comm is parenthesized and may contain spaces: cut after the
                // last ')'.
                let Some(close) = stat.rfind(')') else {
                    continue;
                };
                let comm = stat[1..close].trim_matches(|c| c == '(').to_string();
                let mut rest = stat[close + 1..].split_whitespace();
                // After state (field 3) comes ppid (field 4).
                let ppid = rest.nth(1).and_then(|p| p.parse::<u32>().ok()).unwrap_or(0);
                procs.push((pid, ppid, comm));
            }
        }
        let ppid_of: HashMap<u32, u32> = procs.iter().map(|(p, pp, _)| (*p, *pp)).collect();
        let is_ours = |mut pid: u32| -> bool {
            for _ in 0..16 {
                if pid == my_pid {
                    return true;
                }
                match ppid_of.get(&pid) {
                    Some(&pp) if pp != 0 => pid = pp,
                    _ => return false,
                }
            }
            false
        };
        procs.sort_by_key(|(p, _, _)| *p);
        let processes: Vec<Value> = procs
            .iter()
            .map(|(pid, ppid, comm)| {
                json!({"pid": pid, "ppid": ppid, "comm": comm, "ours": is_ours(*pid)})
            })
            .collect();

        let mut ports: Vec<Value> = Vec::new();
        for (file, v6) in [("/proc/net/tcp", false), ("/proc/net/tcp6", true)] {
            let Ok(text) = std::fs::read_to_string(file) else {
                continue;
            };
            for line in text.lines().skip(1) {
                let cols: Vec<&str> = line.split_whitespace().collect();
                if cols.len() < 4 || cols[3] != "0A" {
                    continue; // only LISTEN
                }
                let Some((_, port_hex)) = cols[1].split_once(':') else {
                    continue;
                };
                let Ok(port) = u16::from_str_radix(port_hex, 16) else {
                    continue;
                };
                ports.push(json!({"port": port, "v6": v6, "inode": cols[9]}));
            }
        }
        ports.sort_by_key(|p| p.get("port").and_then(Value::as_u64).unwrap_or(0));
        json_response(
            StatusCode::OK,
            &json!({"processes": processes, "ports": ports}),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Fixed argv only.
        let ps = Command::new("ps")
            .args(["-eo", "pid,ppid,comm"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        let processes: Vec<Value> = ps
            .lines()
            .skip(1)
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let pid = it.next()?.parse::<u32>().ok()?;
                let ppid = it.next()?.parse::<u32>().ok()?;
                let comm = it.next()?.to_string();
                Some(json!({"pid": pid, "ppid": ppid, "comm": comm, "ours": false}))
            })
            .collect();
        json_response(
            StatusCode::OK,
            &json!({"processes": processes, "ports": []}),
        )
    }
}

/// Entry point: the parent router forwards every `/api/terminal/*` request.
pub async fn api(req: Request<Incoming>) -> Response<Body> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    let rest = path.strip_prefix("/api/terminal/").unwrap_or("");
    match (&method, rest) {
        (&Method::POST, "create") => create(req).await,
        (&Method::GET, "list") => list(),
        (&Method::GET, "output") => output(&req),
        (&Method::POST, "input") => input(req).await,
        (&Method::POST, "kill") => kill(req).await,
        (&Method::GET, "system") => system(),
        _ => error(StatusCode::NOT_FOUND, "unknown terminal endpoint"),
    }
}
