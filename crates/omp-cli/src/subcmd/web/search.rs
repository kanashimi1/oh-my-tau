//! `GET /api/search?q=&cwd=` — backend for the command palette: searches
//! stored session titles, message content in bounded JSONL history, and
//! project file names. All disk work happens in `spawn_blocking` with hard
//! caps on files scanned, lines read and results returned, so a huge history
//! can never stall the hub.

use std::path::{Path, PathBuf};

use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{json, Value};

use super::{error, json_response, Body};

/// Session files scanned per store (newest first).
const MAX_SESSION_FILES: usize = 150;
/// Lines read per session file while hunting for matches.
const MAX_LINES_PER_FILE: usize = 2000;
/// Lines longer than this are skipped (they are tool blobs, not prose).
const MAX_LINE_BYTES: u64 = 256 * 1024;
/// Bytes of a file's head read to find the session header (id/cwd/title).
const HEADER_SCAN_LINES: usize = 40;
/// Results per category.
const MAX_RESULTS: usize = 12;
/// Snippet length around a match.
const SNIPPET_CHARS: usize = 200;
/// Directory-walk budget (entries visited) and depth for the file search.
const MAX_WALKED: usize = 20_000;
const MAX_WALK_DEPTH: usize = 10;
/// Directory names never descended into during the file walk.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "dist",
    "build",
    ".next",
    ".cache",
    ".venv",
    "venv",
    "__pycache__",
];

pub async fn api(req: Request<Incoming>) -> Response<Body> {
    if req.method() != Method::GET {
        return error(StatusCode::METHOD_NOT_ALLOWED, "GET only");
    }
    let Some(q) = query_param(&req, "q").filter(|q| !q.trim().is_empty()) else {
        return json_response(
            StatusCode::OK,
            &json!({"sessions": [], "messages": [], "files": []}),
        );
    };
    let cwd = query_param(&req, "cwd")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let needle = q.to_lowercase();
    let result = tokio::task::spawn_blocking(move || search(&cwd, &needle)).await;
    match result {
        Ok(v) => json_response(StatusCode::OK, &v),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    }
}

fn query_param(req: &Request<Incoming>, key: &str) -> Option<String> {
    let query = req.uri().query()?;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k == key {
            return Some(percent_decode(v));
        }
    }
    None
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hex = bytes.get(i + 1..i + 3).and_then(|h| {
                    std::str::from_utf8(h)
                        .ok()
                        .and_then(|h| u8::from_str_radix(h, 16).ok())
                });
                if let Some(b) = hex {
                    out.push(b);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn mtime_ms(path: &Path) -> i64 {
    use std::time::UNIX_EPOCH;
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// First text of a message `content` (string or block array).
fn content_text(content: &Value) -> Option<String> {
    if let Some(s) = content.as_str() {
        return Some(s.to_owned());
    }
    let mut joined = String::new();
    for block in content.as_array()? {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(t) = block.get("text").and_then(Value::as_str) {
                if !joined.is_empty() {
                    joined.push(' ');
                }
                joined.push_str(t);
            }
        }
    }
    Some(joined).filter(|s| !s.is_empty())
}

/// Session files under one store dir, newest first.
fn jsonl_files(dir: &Option<PathBuf>) -> Vec<PathBuf> {
    let Some(dir) = dir else { return Vec::new() };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(i64, PathBuf)> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "jsonl"))
        .map(|p| (mtime_ms(&p), p))
        .collect();
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    files.into_iter().map(|(_, p)| p).collect()
}

/// Header facts from a file's first lines: (id, cwd, title).
fn scan_header(path: &Path) -> (Option<String>, Option<String>, Option<String>) {
    use std::io::{BufRead, BufReader};
    let Ok(file) = std::fs::File::open(path) else {
        return (None, None, None);
    };
    let reader = BufReader::with_capacity(16 * 1024, file);
    let (mut id, mut cwd, mut title) = (None, None, None);
    for line in reader.lines().take(HEADER_SCAN_LINES) {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("session") => {
                id = v.get("id").and_then(Value::as_str).map(str::to_owned);
                cwd = v.get("cwd").and_then(Value::as_str).map(str::to_owned);
                title = v
                    .get("title")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                    .map(str::to_owned)
                    .or(title);
            }
            Some("title" | "session_info") => {
                title = v
                    .get("title")
                    .or_else(|| v.get("name"))
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                    .map(str::to_owned)
                    .or(title);
            }
            _ => {}
        }
    }
    (id, cwd, title)
}

/// A snippet of `text` centered on the (lowercased) match at `idx`.
fn snippet(text: &str, idx: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let idx = chars.len().min(idx);
    let half = SNIPPET_CHARS / 2;
    let start = idx.saturating_sub(half);
    let end = (idx + half).min(chars.len());
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(&chars[start..end]);
    if end < chars.len() {
        out.push('…');
    }
    out
}

/// Match messages of one session file against `needle`; returns at most
/// `MAX_RESULTS` snippets.
fn scan_messages(path: &Path, needle: &str) -> Vec<(usize, String, String)> {
    use std::io::{BufRead, BufReader};
    let mut hits = Vec::new();
    let Ok(file) = std::fs::File::open(path) else {
        return hits;
    };
    let reader = BufReader::with_capacity(64 * 1024, file);
    for line in reader.lines().take(MAX_LINES_PER_FILE) {
        if hits.len() >= MAX_RESULTS {
            break;
        }
        let Ok(line) = line else { break };
        if line.len() as u64 > MAX_LINE_BYTES {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if v.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let msg = v.get("message");
        let role = msg
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_owned();
        let Some(text) = msg.and_then(|m| m.get("content")).and_then(content_text) else {
            continue;
        };
        let lower = text.to_lowercase();
        let Some(idx) = lower.find(needle) else {
            continue;
        };
        hits.push((idx, snippet(&text, idx), role));
    }
    hits
}

fn search(cwd: &Path, needle: &str) -> Value {
    let mut sessions = Vec::new();
    let mut messages = Vec::new();

    let stores = [
        crate::persist::sessions_dir(cwd),
        crate::persist::ts_sessions_dir(cwd),
    ];
    'stores: for dir in stores {
        for path in jsonl_files(&dir).into_iter().take(MAX_SESSION_FILES) {
            if sessions.len() >= MAX_RESULTS && messages.len() >= MAX_RESULTS {
                break 'stores;
            }
            let (id, file_cwd, title) = scan_header(&path);
            let title = title.unwrap_or_default();
            let id = id.unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            let title_hit = title.to_lowercase().contains(needle);
            let id_hit = id.to_lowercase().contains(needle);
            if (title_hit || id_hit) && sessions.len() < MAX_RESULTS {
                sessions.push(json!({
                    "id": id,
                    "title": if title.is_empty() { Value::Null } else { json!(title) },
                    "sessionFile": path.display().to_string(),
                    "cwd": file_cwd,
                    "modifiedMs": mtime_ms(&path),
                }));
            }
            if messages.len() < MAX_RESULTS {
                for (idx, snip, role) in scan_messages(&path, needle) {
                    if messages.len() >= MAX_RESULTS {
                        break;
                    }
                    messages.push(json!({
                        "sessionId": id,
                        "agentId": Value::Null,
                        "sessionFile": path.display().to_string(),
                        "cwd": file_cwd,
                        "role": role,
                        "snippet": snip,
                        "matchIndex": idx,
                        "modifiedMs": mtime_ms(&path),
                    }));
                }
            }
        }
    }

    let files = search_files(cwd, needle);
    json!({"sessions": sessions, "messages": messages, "files": files})
}

/// Case-insensitive file-name search under `cwd`, bounded walk.
fn search_files(cwd: &Path, needle: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut visited = 0usize;
    let mut stack = vec![(cwd.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        if out.len() >= MAX_RESULTS || visited >= MAX_WALKED {
            break;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.flatten() {
            visited += 1;
            if visited >= MAX_WALKED || out.len() >= MAX_RESULTS {
                break;
            }
            let Ok(ft) = entry.file_type() else { continue };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if ft.is_dir() {
                if !SKIP_DIRS.contains(&name.as_ref()) && depth < MAX_WALK_DEPTH {
                    stack.push((entry.path(), depth + 1));
                }
                continue;
            }
            if name.to_lowercase().contains(needle) {
                out.push(json!({
                    "name": name,
                    "path": entry.path().display().to_string(),
                    "dir": dir.display().to_string(),
                }));
            }
        }
    }
    out
}
