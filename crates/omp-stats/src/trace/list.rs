//! Session list for the Traces section (`listSessionSummaries` in trace.ts):
//! fold every synced child transcript (subagents, advisors) into its root row
//! and merge in on-disk roots the sync pass hasn't indexed yet.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde_json::Value;

use crate::parser::{extract_folder_from_path, util::date_parse};
use crate::paths::{path_relative, session_root_of, session_roots};
use crate::rollup::get_session_rollups;
use crate::types::SessionSummary;

use super::{mtime_ms, LIST_FOLD_LIMIT, TITLE_SCAN_BYTES};

/// Session list metadata read from the fixed title slot and session header.
#[derive(Debug, Clone)]
struct SessionMetadata {
    title: Option<String>,
    cwd: Option<String>,
}

static METADATA_CACHE: std::sync::LazyLock<Mutex<HashMap<String, (f64, SessionMetadata)>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));
static DISK_ROOTS_MEMO: Mutex<Option<DiskRootsMemo>> = Mutex::new(None);

struct DiskRootsMemo {
    at_ms: f64,
    sessions_dir: String,
    limit: i64,
    roots: Vec<DiskRoot>,
}

#[derive(Debug, Clone)]
struct DiskRoot {
    file: String,
    mtime_ms: f64,
    started_at: i64,
}

/// Read at most `TITLE_SCAN_BYTES` decompressed bytes: a bounded `take` for
/// plain files, and a `GzDecoder` stopped after 4 KB for gc-compressed ones
/// (TS streams a `DecompressionStream` and cancels it after 4 KB — never
/// decompressing the whole file, and using whatever decoded before a corrupt
/// tail).
fn read_session_metadata_head(file: &str) -> Option<String> {
    use std::io::Read;
    let file_handle = std::fs::File::open(file).ok()?;
    let mut head = Vec::new();
    if file.ends_with(".gz") {
        let mut decoder = flate2::read::GzDecoder::new(file_handle).take(TITLE_SCAN_BYTES as u64);
        let _ = decoder.read_to_end(&mut head); // partial head kept on a corrupt tail
    } else {
        let _ = file_handle
            .take(TITLE_SCAN_BYTES as u64)
            .read_to_end(&mut head);
    }
    Some(String::from_utf8_lossy(&head).into_owned())
}

/// Read list metadata from the fixed title slot and session header. The cache
/// lock is only held for the lookup/insert — the transcript read happens with
/// the lock released.
fn read_session_metadata(file: &str) -> SessionMetadata {
    let Ok(meta) = std::fs::metadata(file) else {
        return SessionMetadata {
            title: None,
            cwd: None,
        };
    };
    let mtime = mtime_ms(&meta);
    {
        let cache = METADATA_CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((cached_mtime, metadata)) = cache.get(file) {
            if *cached_mtime == mtime {
                return metadata.clone();
            }
        }
    }
    let mut title: Option<String> = None;
    let mut cwd: Option<String> = None;
    if let Some(head) = read_session_metadata_head(file) {
        for line in head.split('\n') {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(parsed) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if parsed.get("type").and_then(Value::as_str) == Some("title") {
                if let Some(t) = parsed.get("title").and_then(Value::as_str) {
                    title = Some(t.to_string());
                }
                continue;
            }
            if parsed.get("type").and_then(Value::as_str) == Some("session") {
                if title.is_none() {
                    title = parsed
                        .get("title")
                        .and_then(Value::as_str)
                        .map(ToString::to_string);
                }
                cwd = parsed
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                break;
            }
        }
    }
    let metadata = SessionMetadata { title, cwd };
    let mut cache = METADATA_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.insert(file.to_string(), (mtime, metadata.clone()));
    metadata
}

/// Fold accumulator: a SessionSummary plus the model set.
#[derive(Debug)]
struct SummaryFold {
    summary: SessionSummary,
    model_set: std::collections::BTreeSet<String>,
}

/// Parse the ms timestamp from a session basename
/// (`2026-09-01T10-24-49-741Z_<uuid>.jsonl`). Byte-checked to match the TS
/// regex `/^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d{3})Z_/` — string
/// slicing here would panic on non-ASCII names.
fn basename_timestamp(base: &str) -> Option<i64> {
    let b = base.as_bytes();
    let digit = |i: usize| b[i].is_ascii_digit();
    if b.len() < 25
        || !digit(0)
        || !digit(1)
        || !digit(2)
        || !digit(3)
        || b[4] != b'-'
        || !digit(5)
        || !digit(6)
        || b[7] != b'-'
        || !digit(8)
        || !digit(9)
        || b[10] != b'T'
        || !digit(11)
        || !digit(12)
        || b[13] != b'-'
        || !digit(14)
        || !digit(15)
        || b[16] != b'-'
        || !digit(17)
        || !digit(18)
        || b[19] != b'-'
        || !digit(20)
        || !digit(21)
        || !digit(22)
        || &b[23..25] != b"Z_"
    {
        return None;
    }
    let iso = format!(
        "{}T{}:{}:{}.{}Z",
        std::str::from_utf8(&b[0..10]).ok()?,
        std::str::from_utf8(&b[11..13]).ok()?,
        std::str::from_utf8(&b[14..16]).ok()?,
        std::str::from_utf8(&b[17..19]).ok()?,
        std::str::from_utf8(&b[20..23]).ok()?
    );
    date_parse(&iso)
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}

/// Root session transcripts found on disk, newest-first by mtime. Lets the
/// Traces list include sessions the sync pass hasn't indexed yet (most
/// importantly the currently running one).
fn scan_disk_roots(limit: i64) -> Result<Vec<DiskRoot>> {
    // Short-TTL memo (30s poll): the session list changes only on
    // create/edit, so readdir+stat of every file per poll is pure syscall
    // churn. Only the newest `limit` roots are kept.
    const DISK_ROOTS_TTL_MS: f64 = 5_000.0;
    let sessions_dir_str = session_roots()
        .iter()
        .map(|r| r.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("\n");
    {
        let memo = DISK_ROOTS_MEMO
            .lock()
            .map_err(|_| anyhow::anyhow!("disk roots memo poisoned"))?;
        if let Some(memo) = memo.as_ref() {
            if memo.sessions_dir == sessions_dir_str
                && memo.limit >= limit
                && now_ms() - memo.at_ms < DISK_ROOTS_TTL_MS
            {
                return Ok(memo
                    .roots
                    .iter()
                    .take(limit.max(0) as usize)
                    .cloned()
                    .collect());
            }
        }
    }
    let mut roots: Vec<DiskRoot> = Vec::new();
    let mut projects = Vec::new();
    for root in session_roots() {
        match std::fs::read_dir(&root) {
            Ok(entries) => projects.extend(entries),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    for project in projects {
        let Ok(project) = project else { continue };
        let Ok(names) = read_dir_sorted(&project.path()) else {
            continue; // Not a directory or unreadable — skip.
        };
        for name in names {
            if !name.ends_with(".jsonl") && !name.ends_with(".jsonl.gz") {
                continue;
            }
            let file = project.path().join(&name);
            let Ok(meta) = std::fs::metadata(&file) else {
                continue; // Raced with gc — skip.
            };
            let mtime = mtime_ms(&meta);
            roots.push(DiskRoot {
                file: file.to_string_lossy().into_owned(),
                mtime_ms: mtime,
                started_at: basename_timestamp(&name).unwrap_or(mtime as i64),
            });
        }
    }
    roots.sort_by(|a, b| {
        b.mtime_ms
            .partial_cmp(&a.mtime_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    roots.truncate(limit.max(0) as usize);
    let memo = DiskRootsMemo {
        at_ms: now_ms(),
        sessions_dir: sessions_dir_str,
        limit,
        roots: roots.clone(),
    };
    let mut slot = DISK_ROOTS_MEMO
        .lock()
        .map_err(|_| anyhow::anyhow!("disk roots memo poisoned"))?;
    *slot = Some(memo);
    Ok(roots)
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

/**
 * List root sessions for the Traces section, folding every synced child
 * transcript (subagents, advisors) into its root row. `limit` and `q` mirror
 * the `/api/sessions` query params.
 */
pub fn list_session_summaries(limit: i64, q: Option<&str>) -> Result<Vec<SessionSummary>> {
    // The Traces page may be the first thing a dashboard serves; get_session_rollups
    // goes through db::with_db, which initializes the database first.
    // TS `Map` preserves insertion order — the tie-break for equal endedAt in
    // the sort below — so keep folds in a Vec with an index on the side.
    let mut by_root: Vec<SummaryFold> = Vec::new();
    let mut by_root_index: HashMap<String, usize> = HashMap::new();

    for row in get_session_rollups()? {
        let Some(sessions_dir) = session_root_of(Path::new(&row.session_file)) else {
            continue;
        };
        let sessions_dir_str = sessions_dir.to_string_lossy().into_owned();
        let rel = path_relative(&sessions_dir, Path::new(&row.session_file));
        let rel_str = rel.to_string_lossy();
        if rel_str.is_empty() || rel.is_absolute() || rel_str.starts_with("..") {
            continue;
        }
        let segments: Vec<String> = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        if segments.len() < 2 {
            continue;
        }
        let is_child = segments.len() > 2;
        let root_file = if is_child {
            format!(
                "{sessions_dir_str}/{}/{segments1}.jsonl",
                segments[0],
                segments1 = segments[1]
            )
        } else {
            row.session_file.clone()
        };

        let fold_idx = *by_root_index.entry(root_file.clone()).or_insert_with(|| {
            by_root.push(SummaryFold {
                summary: SessionSummary {
                    file: root_file.clone(),
                    folder: extract_folder_from_path(&root_file),
                    title: None,
                    started_at: row.started_at,
                    ended_at: row.ended_at as f64,
                    requests: 0,
                    tool_calls: 0,
                    subagents: 0,
                    total_tokens: 0,
                    cost_total: 0.0,
                    unpriced_requests: 0,
                    models: Vec::new(),
                },
                model_set: std::collections::BTreeSet::new(),
            });
            by_root.len() - 1
        });
        let fold = &mut by_root[fold_idx];
        let s = &mut fold.summary;
        s.requests += row.requests;
        s.tool_calls += row.tool_calls;
        if is_child {
            s.subagents += 1;
        }
        s.started_at = s.started_at.min(row.started_at);
        s.ended_at = s.ended_at.max(row.ended_at as f64);
        s.total_tokens += row.total_tokens;
        s.cost_total += row.cost_total;
        s.unpriced_requests += row.unpriced_requests;
        if let Some(models) = &row.models {
            for model in models.split(',') {
                if !model.is_empty() {
                    fold.model_set.insert(model.to_string());
                }
            }
        }
    }

    // Fold in on-disk roots the sync pass hasn't indexed yet, so the currently
    // running session shows up without a manual Sync.
    for root in scan_disk_roots(LIST_FOLD_LIMIT as i64)? {
        match by_root_index.get(&root.file).copied() {
            Some(idx) => {
                // Live sessions keep writing after the last sync; sort by real recency.
                by_root[idx].summary.ended_at = by_root[idx].summary.ended_at.max(root.mtime_ms);
            }
            None => {
                by_root_index.insert(root.file.clone(), by_root.len());
                by_root.push(SummaryFold {
                    summary: SessionSummary {
                        file: root.file.clone(),
                        folder: extract_folder_from_path(&root.file),
                        title: None,
                        started_at: root.started_at,
                        ended_at: root.mtime_ms,
                        requests: 0,
                        tool_calls: 0,
                        subagents: 0,
                        total_tokens: 0,
                        cost_total: 0.0,
                        unpriced_requests: 0,
                        models: Vec::new(),
                    },
                    model_set: std::collections::BTreeSet::new(),
                });
            }
        }
    }

    by_root.sort_by(|a, b| {
        b.summary
            .ended_at
            .partial_cmp(&a.summary.ended_at)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    by_root.truncate(LIST_FOLD_LIMIT);

    let mut rows: Vec<SessionSummary> = by_root
        .into_iter()
        .map(|mut fold| {
            let metadata = read_session_metadata(&fold.summary.file);
            fold.summary.title = metadata.title;
            fold.summary.folder = metadata
                .cwd
                .unwrap_or_else(|| extract_folder_from_path(&fold.summary.file));
            fold.summary.models = fold.model_set.into_iter().collect();
            fold.summary
        })
        .collect();

    if let Some(needle) = q.map(str::trim).filter(|s| !s.is_empty()) {
        let needle = needle.to_lowercase();
        rows.retain(|row| {
            row.title
                .as_deref()
                .unwrap_or("")
                .to_lowercase()
                .contains(&needle)
                || row.folder.to_lowercase().contains(&needle)
                || PathBuf::from(&row.file)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_lowercase().contains(&needle))
                    .unwrap_or(false)
        });
    }
    rows.truncate(limit.max(0) as usize);
    Ok(rows)
}
