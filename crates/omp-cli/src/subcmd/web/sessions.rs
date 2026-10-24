//! Read-only views over the on-disk session stores for the web sidebar:
//! projects (one per session dir), the sessions of one folder, and a
//! directory browser for picking a new session's working directory.

use std::io::{BufRead as _, BufReader};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde_json::{json, Value};

/// Lines read per file looking for the title / first user prompt.
const SCAN_LINES: usize = 60;
/// Sessions listed per store.
const MAX_SESSIONS: usize = 300;
const PREVIEW_CHARS: usize = 160;

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn mtime_ms(path: &Path) -> i64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

fn truncate(text: &str, max: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        let mut out: String = flat.chars().take(max).collect();
        out.push('…');
        out
    }
}

/// First text of a message `content` (string or block array).
fn content_text(content: &Value) -> Option<String> {
    if let Some(s) = content.as_str() {
        return Some(s.to_owned());
    }
    content.as_array()?.iter().find_map(|b| {
        (b.get("type").and_then(Value::as_str) == Some("text"))
            .then(|| b.get("text").and_then(Value::as_str).map(str::to_owned))
            .flatten()
    })
}

/// Header facts of one session file: id, cwd, title, first user prompt.
fn scan_file(path: &Path) -> Option<Value> {
    let file = std::fs::File::open(path).ok()?;
    let reader = BufReader::with_capacity(16 * 1024, file);
    let (mut id, mut cwd, mut title, mut first) = (None, None, None, None);
    for line in reader.lines().take(SCAN_LINES) {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        match v.get("type").and_then(Value::as_str) {
            Some("session") => {
                id = v.get("id").and_then(Value::as_str).map(str::to_owned);
                cwd = v.get("cwd").and_then(Value::as_str).map(str::to_owned);
                if title.is_none() {
                    title = v.get("title").and_then(Value::as_str).map(str::to_owned);
                }
            }
            Some("title" | "session_info") => {
                if let Some(t) = v
                    .get("title")
                    .or_else(|| v.get("name"))
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                {
                    title = Some(t.to_owned());
                }
            }
            Some("message") if first.is_none() => {
                let msg = v.get("message");
                if msg.and_then(|m| m.get("role")).and_then(Value::as_str) == Some("user") {
                    first = msg.and_then(|m| m.get("content")).and_then(content_text);
                }
            }
            _ => {}
        }
        if id.is_some() && title.is_some() && first.is_some() {
            break;
        }
    }
    let id = id.or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))?;
    Some(json!({
        "id": id,
        "path": path.display().to_string(),
        "cwd": cwd,
        "title": title,
        "preview": first.map(|f| truncate(&f, PREVIEW_CHARS)),
        "modifiedMs": mtime_ms(path),
        "bytes": std::fs::metadata(path).map_or(0, |m| m.len()),
    }))
}

fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<(i64, PathBuf)> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "jsonl"))
        .map(|p| (mtime_ms(&p), p))
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    files.into_iter().map(|(_, p)| p).collect()
}

/// Sessions of `cwd` across the omp-rs store and TS omp's store, newest first.
pub fn list_sessions(cwd: &Path) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    let stores = [
        (crate::persist::sessions_dir(cwd), "omt"),
        (crate::persist::ts_sessions_dir(cwd), "omp"),
    ];
    for (dir, store) in stores {
        let Some(dir) = dir else { continue };
        for path in jsonl_files(&dir).into_iter().take(MAX_SESSIONS) {
            if let Some(mut row) = scan_file(&path) {
                row["store"] = json!(store);
                rows.push(row);
            }
        }
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r["modifiedMs"].as_i64().unwrap_or(0)));
    json!({"cwd": cwd.display().to_string(), "sessions": rows})
}

/// Folders that have omp-rs sessions, newest activity first (cwd read from
/// the newest session header, since the dir slug is lossy).
pub fn list_projects() -> Value {
    let Some(root) = home().map(|h| h.join(".omp-rs/sessions")) else {
        return json!({"projects": []});
    };
    let Ok(rd) = std::fs::read_dir(&root) else {
        return json!({"projects": []});
    };
    let mut rows: Vec<Value> = Vec::new();
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let files = jsonl_files(&dir);
        let Some(newest) = files.first() else { continue };
        let cwd = files
            .iter()
            .take(5)
            .find_map(|f| scan_file(f).and_then(|v| v["cwd"].as_str().map(str::to_owned)));
        let Some(cwd) = cwd else { continue };
        rows.push(json!({
            "cwd": cwd,
            "sessions": files.len(),
            "modifiedMs": mtime_ms(newest),
            "exists": Path::new(&cwd).is_dir(),
        }));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r["modifiedMs"].as_i64().unwrap_or(0)));
    json!({"projects": rows})
}

/// Sub-directories of `path` (default `$HOME`) for the folder picker.
pub fn browse(path: Option<&str>) -> Value {
    let dir = path
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(home)
        .unwrap_or_else(|| PathBuf::from("/"));
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    let mut dirs: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    dirs.sort_by_key(|d| d.to_lowercase());
    json!({
        "path": dir.display().to_string(),
        "parent": dir.parent().map(|p| p.display().to_string()),
        "dirs": dirs,
        "isGit": dir.join(".git").exists(),
    })
}

/// Only session files under a known store may be exported/opened by path.
pub fn is_session_path(path: &Path) -> bool {
    let Some(home) = home() else { return false };
    let Ok(canon) = std::fs::canonicalize(path) else { return false };
    canon.extension().is_some_and(|e| e == "jsonl")
        && [home.join(".omp-rs"), omp_stats::paths::sessions_dir()]
            .iter()
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .any(|root| canon.starts_with(root))
}
