//! Read-only views over the on-disk session stores for the web sidebar:
//! projects (one per session dir), the sessions of one folder, and a
//! directory browser for picking a new session's working directory.

use std::collections::HashMap;
use std::io::{BufRead as _, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::UNIX_EPOCH;

use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

/// Metadata cache: scan results never re-read while a file/dir keeps its
/// mtime+size; per-key single-flight so concurrent requests for the same
/// project/cwd share one scan (the loser re-checks the cache after waiting).
/// Cache entry: (mtime ms, size, payload).
type Stamp<T> = (i64, u64, T);

struct Cache {
    files: AsyncMutex<HashMap<PathBuf, Stamp<Option<Value>>>>,
    dirs: AsyncMutex<HashMap<PathBuf, Stamp<Vec<PathBuf>>>>,
    flights: AsyncMutex<HashMap<String, Arc<AsyncMutex<()>>>>,
    scans: std::sync::atomic::AtomicU64,
}

static CACHE: OnceLock<Cache> = OnceLock::new();

fn cache() -> &'static Cache {
    CACHE.get_or_init(|| Cache {
        files: AsyncMutex::new(HashMap::new()),
        dirs: AsyncMutex::new(HashMap::new()),
        flights: AsyncMutex::new(HashMap::new()),
        scans: std::sync::atomic::AtomicU64::new(0),
    })
}

/// (mtime ms, size) of `path`, or `None` if it cannot be stated.
fn stamp(path: &Path) -> Option<(i64, u64)> {
    let m = std::fs::metadata(path).ok()?;
    let ms = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    Some((ms, m.len()))
}

/// Run `f` under a single per-key flight: concurrent callers of the same key
/// serialize, and the latecomer re-checks the cache inside `f` instead of
/// rescanning.
async fn with_flight<T>(key: String, f: impl std::future::Future<Output = T>) -> T {
    let flight = {
        let mut flights = cache().flights.lock().await;
        Arc::clone(flights.entry(key).or_default())
    };
    let _guard = flight.lock().await;
    f.await
}

/// Scan results of `paths`, re-reading only files whose mtime+size changed.
/// All misses of one store/directory are scanned in ONE blocking-pool job:
/// hundreds of tiny `spawn_blocking` calls cost far more than the JSONL reads.
/// `None` results (unreadable files) are cached too, keyed by their stamp.
async fn cached_scan_files(paths: &[PathBuf]) -> Vec<Option<Value>> {
    let mut out = vec![None; paths.len()];
    let mut misses = Vec::new();
    {
        let files = cache().files.lock().await;
        for (i, path) in paths.iter().enumerate() {
            let Some(st) = stamp(path) else { continue };
            if let Some((m, s, v)) = files.get(path) {
                if (*m, *s) == st {
                    out[i].clone_from(v);
                    continue;
                }
            }
            misses.push((i, path.clone(), st));
        }
    }
    if misses.is_empty() {
        return out;
    }
    cache()
        .scans
        .fetch_add(misses.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let scanned = tokio::task::spawn_blocking(move || {
        misses
            .into_iter()
            .map(|(i, path, st)| {
                let value = scan_file(&path);
                (i, path, st, value)
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    let mut files = cache().files.lock().await;
    for (i, path, st, value) in scanned {
        files.insert(path, (st.0, st.1, value.clone()));
        out[i] = value;
    }
    out
}

/// `.jsonl` files of `dir` (newest first). The directory entry list is cached,
/// but ordering is deliberately recomputed: appending to a session changes the
/// file's mtime, not the directory's mtime.
async fn cached_jsonl_files(dir: PathBuf) -> Vec<PathBuf> {
    let stamp = stamp(&dir);
    {
        let dirs = cache().dirs.lock().await;
        if let (Some((m, s, v)), Some(st)) = (dirs.get(&dir), stamp) {
            if (*m, *s) == st {
                let mut files = v.clone();
                files.sort_by_key(|p| std::cmp::Reverse(mtime_ms(p)));
                return files;
            }
        }
    }
    let list_dir = dir.clone();
    let mut listed = tokio::task::spawn_blocking(move || jsonl_files(&list_dir))
        .await
        .ok()
        .unwrap_or_default();
    if let Some(st) = stamp {
        cache()
            .dirs
            .lock()
            .await
            .insert(dir, (st.0, st.1, listed.clone()));
    }
    listed.sort_by_key(|p| std::cmp::Reverse(mtime_ms(p)));
    listed
}

/// Sub-directories of `root` (project dirs), same dir-cache invalidation.
async fn cached_subdirs(root: PathBuf) -> Vec<PathBuf> {
    let stamp = stamp(&root);
    {
        let dirs = cache().dirs.lock().await;
        if let (Some((m, s, v)), Some(st)) = (dirs.get(&root), stamp) {
            if (*m, *s) == st {
                return v.clone();
            }
        }
    }
    let list_root = root.clone();
    let listed = tokio::task::spawn_blocking(move || {
        std::fs::read_dir(&list_root)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    })
    .await
    .ok()
    .unwrap_or_default();
    if let Some(st) = stamp {
        cache()
            .dirs
            .lock()
            .await
            .insert(root, (st.0, st.1, listed.clone()));
    }
    listed
}

/// Lines read per file looking for the title / first user prompt.
const SCAN_LINES: usize = 60;
/// Sessions listed per store.
const MAX_SESSIONS: usize = 300;
/// Session rows returned by the explorer (empty-cwd) sweep, per project dir
/// and for the merged list.
const MAX_ALL_SESSIONS: usize = 2000;
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
    let (mut id, mut cwd, mut hdr_title, mut first) = (None, None, None, None);
    for line in reader.lines().take(SCAN_LINES) {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("session") => {
                id = v.get("id").and_then(Value::as_str).map(str::to_owned);
                cwd = v.get("cwd").and_then(Value::as_str).map(str::to_owned);
                if hdr_title.is_none() {
                    hdr_title = v.get("title").and_then(Value::as_str).map(str::to_owned);
                }
            }
            Some("title" | "session_info") => {
                if hdr_title.is_none() {
                    if let Some(t) = v
                        .get("title")
                        .or_else(|| v.get("name"))
                        .and_then(Value::as_str)
                        .filter(|t| !t.is_empty())
                    {
                        hdr_title = Some(t.to_owned());
                    }
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
        if id.is_some() && first.is_some() {
            break;
        }
    }
    // A rename appends a `session_info`/`title` entry at the tail; prefer the
    // newest one over the header slot so a stored-session rename always shows.
    let title = tail_title(path).or(hdr_title);
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

/// Last `title`/`session_info` title in the file tail (a `rename_session`
/// append). Scanning the tail keeps renames visible even for long sessions.
fn tail_title(path: &Path) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    const TAIL: u64 = 8192;
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let start = len.saturating_sub(TAIL);
    if start > 0 {
        f.seek(SeekFrom::Start(start)).ok()?;
    }
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf);
    let mut title = None;
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some("title" | "session_info") = v.get("type").and_then(Value::as_str) {
            if let Some(t) = v
                .get("title")
                .or_else(|| v.get("name"))
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
            {
                title = Some(t.to_owned());
            }
        }
    }
    title
}

fn jsonl_files(dir: &Path) -> Vec<PathBuf> {
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

/// Sessions of `cwd` across the omp-rs store and TS omp's store, newest
/// first. An empty `cwd` (`GET /api/sessions?cwd=`, the sidebar explorer's
/// "every project" mode) sweeps both stores' project dirs instead and fills
/// each row's `cwd` so the UI can group by project.
pub async fn list_sessions(cwd: &Path) -> Value {
    if cwd.as_os_str().is_empty() {
        return list_all_sessions().await;
    }
    let stores = [
        (crate::persist::sessions_dir(cwd), "omt"),
        (crate::persist::ts_sessions_dir(cwd), "omp"),
    ];
    sessions_in(&stores, cwd).await
}

/// Store dirs → session rows; the async core of [`list_sessions`] (testable
/// with explicit dirs). Wrapped in a per-dirs single-flight: concurrent
/// callers serialize and the latecomer re-checks the file/dir caches instead
/// of rescanning. Blocking disk work runs on the blocking pool.
async fn sessions_in(stores: &[(Option<PathBuf>, &'static str)], cwd: &Path) -> Value {
    let dirs: Vec<PathBuf> = stores.iter().filter_map(|(d, _)| d.clone()).collect();
    let key = format!(
        "sessions:{}",
        dirs.iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join("|")
    );
    with_flight(key, scan_stores(stores, cwd)).await
}

async fn scan_stores(stores: &[(Option<PathBuf>, &'static str)], cwd: &Path) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    for (dir, store) in stores {
        let Some(dir) = dir else { continue };
        let paths: Vec<PathBuf> = cached_jsonl_files(dir.clone())
            .await
            .into_iter()
            .take(MAX_SESSIONS)
            .collect();
        let scanned = cached_scan_files(&paths).await;
        for mut row in scanned.into_iter().flatten() {
            row["store"] = json!(store);
            rows.push(row);
        }
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r["modifiedMs"].as_i64().unwrap_or(0)));
    json!({"cwd": cwd.display().to_string(), "sessions": rows})
}

/// Store roots that hold one sub-directory per project: the omt slug dirs and
/// TS omp's `sessionDirForCwd` dirs.
fn store_roots() -> Vec<(PathBuf, &'static str)> {
    let mut roots = Vec::new();
    if let Some(home) = home() {
        roots.push((home.join(".omp-rs/sessions"), "omt"));
    }
    roots.push((omp_stats::paths::sessions_dir(), "omp"));
    roots
}

/// Every project's sessions across both stores, newest first — the empty-cwd
/// mode of [`list_sessions`]. One single-flight for the whole sweep; the
/// per-dir listings and file scans reuse the same caches as single-project
/// listing, so an explorer refresh only stats and re-sorts.
async fn list_all_sessions() -> Value {
    with_flight("sessions:all".to_string(), all_sessions_in(store_roots())).await
}

/// Async core of [`list_all_sessions`] over explicit roots (testable).
/// `cwd` per row comes from the session header; rows whose header omits it
/// inherit the newest cwd found in their project dir, so grouping by project
/// never loses them.
async fn all_sessions_in(roots: Vec<(PathBuf, &'static str)>) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    for (root, store) in roots {
        for dir in cached_subdirs(root).await {
            let paths: Vec<PathBuf> = cached_jsonl_files(dir.clone())
                .await
                .into_iter()
                .take(MAX_ALL_SESSIONS)
                .collect();
            let scanned = cached_scan_files(&paths).await;
            let dir_cwd = scanned
                .iter()
                .flatten()
                .find_map(|r| r["cwd"].as_str().map(str::to_owned));
            for mut row in scanned.into_iter().flatten() {
                if row["cwd"].as_str().is_none() {
                    row["cwd"] = json!(dir_cwd);
                }
                row["store"] = json!(store);
                rows.push(row);
            }
        }
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r["modifiedMs"].as_i64().unwrap_or(0)));
    rows.truncate(MAX_ALL_SESSIONS);
    json!({"sessions": rows})
}

/// Folders that have omp-rs sessions, newest activity first (cwd read from
/// the newest session header, since the dir slug is lossy).
pub async fn list_projects() -> Value {
    let Some(root) = home().map(|h| h.join(".omp-rs/sessions")) else {
        return json!({"projects": []});
    };
    with_flight("projects".to_string(), projects_in(root)).await
}

/// Async core of [`list_projects`] over an explicit root (testable). Rows are
/// rebuilt from cached dir listings and cached file scans on every call —
/// no JSONL is re-read unless a file's mtime+size changed; only a cheap stat
/// of each project's newest file (`modifiedMs`) stays uncached so appended
/// sessions stay fresh.
async fn projects_in(root: PathBuf) -> Value {
    let dirs = cached_subdirs(root.clone()).await;
    let mut rows: Vec<Value> = Vec::new();
    for dir in dirs {
        let files = cached_jsonl_files(dir.clone()).await;
        let Some(newest) = files.first() else {
            continue;
        };
        let mut cwd = None;
        let candidates: Vec<PathBuf> = files.iter().take(5).cloned().collect();
        for v in cached_scan_files(&candidates).await.into_iter().flatten() {
            if let Some(c) = v["cwd"].as_str().map(str::to_owned) {
                cwd = Some(c);
                break;
            }
        }
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
    let Ok(canon) = std::fs::canonicalize(path) else {
        return false;
    };
    canon.extension().is_some_and(|e| e == "jsonl")
        && [home.join(".omp-rs"), omp_stats::paths::sessions_dir()]
            .iter()
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .any(|root| canon.starts_with(root))
}

/// Append a `session_info`/`title` entry to a stored session file so the
/// sidebar's header/tail scan picks the new name up. Stored sessions only —
/// running sessions rename via the `set_session_name` rpc instead.
pub fn rename_session(path: &Path, name: &str) -> Result<(), String> {
    if !is_session_path(path) {
        return Err("not a session file".to_string());
    }
    let name = name.trim();
    if name.is_empty() {
        return Err("session name cannot be empty".to_string());
    }
    let line = serde_json::json!({
        "type": "session_info",
        "title": name,
        "source": "web-rename",
    });
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|e| format!("{e}"))?;
    writeln!(f, "{line}").map_err(|e| format!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// Tests share the process-global cache and its scan counter, so run
    /// them one at a time. Test entry points: the runtime macro blocks.
    static TEST_GUARD: AsyncMutex<()> = AsyncMutex::const_new(());

    fn write_session(path: &Path, title: &str) {
        std::fs::write(
            path,
            format!(
                "{{\"type\":\"session\",\"id\":\"s1\",\"cwd\":\"{}\"}}\n\
                 {{\"type\":\"message\",\"message\":{{\"role\":\"user\",\"content\":\"hello\"}}}}\n",
                path.parent().unwrap().display()
            ),
        )
        .unwrap();
        let _ = title;
    }

    fn stores(dir: &Path) -> Vec<(Option<PathBuf>, &'static str)> {
        vec![(Some(dir.to_path_buf()), "omt")]
    }

    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    #[tokio::test(flavor = "multi_thread")]
    async fn unchanged_file_is_not_rescanned() {
        let _g = TEST_GUARD.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        write_session(&tmp.path().join("a.jsonl"), "t");
        let before = cache().scans.load(Ordering::Relaxed);
        let first = sessions_in(&stores(tmp.path()), tmp.path()).await;
        let second = sessions_in(&stores(tmp.path()), tmp.path()).await;
        let scanned = cache().scans.load(Ordering::Relaxed) - before;
        assert_eq!(scanned, 1, "second request must hit the file cache");
        assert_eq!(first, second);
    }

    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    #[tokio::test(flavor = "multi_thread")]
    async fn changed_mtime_or_size_invalidates() {
        let _g = TEST_GUARD.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("a.jsonl");
        write_session(&file, "t");
        sessions_in(&stores(tmp.path()), tmp.path()).await;
        let before = cache().scans.load(Ordering::Relaxed);
        // mtime change without size change: rewrite the same bytes (the
        // original is two lines of known length).
        let bytes = std::fs::read(&file).unwrap();
        let mut same_len = bytes.clone();
        same_len[0] = b' ';
        std::fs::write(&file, &same_len).unwrap();
        assert_eq!(std::fs::read(&file).unwrap().len(), bytes.len());
        drop(same_len);
        std::fs::write(&file, bytes).unwrap();
        sessions_in(&stores(tmp.path()), tmp.path()).await;
        assert_eq!(
            cache().scans.load(Ordering::Relaxed) - before,
            1,
            "mtime bump must invalidate"
        );
        // size change
        let before = cache().scans.load(Ordering::Relaxed);
        std::fs::write(
            &file,
            "{\"type\":\"session\",\"id\":\"s1\",\"cwd\":\"/x\"}\n",
        )
        .unwrap();
        sessions_in(&stores(tmp.path()), tmp.path()).await;
        assert_eq!(
            cache().scans.load(Ordering::Relaxed) - before,
            1,
            "size change must invalidate"
        );
    }

    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    #[tokio::test(flavor = "multi_thread")]
    async fn tail_rename_becomes_visible() {
        let _g = TEST_GUARD.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("a.jsonl");
        write_session(&file, "t");
        let before = sessions_in(&stores(tmp.path()), tmp.path()).await;
        assert_ne!(before["sessions"][0]["title"], json!("renamed"));
        // append a rename entry (what `rename_session` does); the file's
        // mtime+size change must invalidate the cached scan.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&file)
            .unwrap()
            .write_all(b"{\"type\":\"session_info\",\"title\":\"renamed\"}\n")
            .unwrap();
        let after = sessions_in(&stores(tmp.path()), tmp.path()).await;
        assert_eq!(after["sessions"][0]["title"], json!("renamed"));
    }

    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_callers_share_one_scan() {
        let _g = TEST_GUARD.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        for n in 0..5 {
            write_session(&tmp.path().join(format!("{n}.jsonl")), "t");
        }
        let stores = stores(tmp.path());
        let before = cache().scans.load(Ordering::Relaxed);
        let (a, b, c) = tokio::join!(
            sessions_in(&stores, tmp.path()),
            sessions_in(&stores, tmp.path()),
            sessions_in(&stores, tmp.path()),
        );
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a["sessions"].as_array().unwrap().len(), 5);
        assert_eq!(
            cache().scans.load(Ordering::Relaxed) - before,
            5,
            "one scan per file"
        );
    }

    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    #[tokio::test(flavor = "multi_thread")]
    async fn append_to_older_file_refreshes_newest_order() {
        let _g = TEST_GUARD.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let older = tmp.path().join("a.jsonl");
        let newer = tmp.path().join("b.jsonl");
        write_session(&older, "a");
        std::thread::sleep(std::time::Duration::from_millis(5));
        write_session(&newer, "b");
        let first = sessions_in(&stores(tmp.path()), tmp.path()).await;
        assert_eq!(
            first["sessions"][0]["path"],
            json!(newer.display().to_string())
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
        std::fs::OpenOptions::new()
            .append(true)
            .open(&older)
            .unwrap()
            .write_all(b"{\"type\":\"session_info\",\"title\":\"active again\"}\n")
            .unwrap();
        let second = sessions_in(&stores(tmp.path()), tmp.path()).await;
        assert_eq!(
            second["sessions"][0]["path"],
            json!(older.display().to_string())
        );
    }

    #[allow(clippy::disallowed_methods)] // test entry point: the runtime macro blocks on the body
    #[tokio::test(flavor = "multi_thread")]
    async fn all_projects_mode_scans_every_dir_and_fills_cwd() {
        let _g = TEST_GUARD.lock().await;
        let tmp = tempfile::tempdir().unwrap();
        let proj_a = tmp.path().join("proj-a");
        let proj_b = tmp.path().join("proj-b");
        std::fs::create_dir(&proj_a).unwrap();
        std::fs::create_dir(&proj_b).unwrap();
        write_session(&proj_a.join("a.jsonl"), "a");
        write_session(&proj_b.join("b.jsonl"), "b");
        // A header-less session: its row must still get the project cwd,
        // inherited from the dir's newest headed session.
        std::fs::write(
            &proj_b.join("bare.jsonl"),
            "{\"type\":\"message\",\"message\":{\"role\":\"user\",\"content\":\"hi\"}}\n",
        )
        .unwrap();
        let all = all_sessions_in(vec![(tmp.path().to_path_buf(), "omt")]).await;
        let rows = all["sessions"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        let cwd_of = |suffix: &str| {
            rows.iter()
                .find(|r| r["path"].as_str().is_some_and(|s| s.ends_with(suffix)))
                .unwrap()["cwd"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(cwd_of("a.jsonl"), proj_a.display().to_string());
        assert_eq!(cwd_of("b.jsonl"), proj_b.display().to_string());
        assert_eq!(cwd_of("bare.jsonl"), proj_b.display().to_string());
        assert!(rows.iter().all(|r| r["store"] == json!("omt")));
    }
}
