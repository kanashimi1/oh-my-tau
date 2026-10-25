//! Port of `trace.ts`: session trace assembly for the Traces dashboard
//! section. Builds a span tree (turns, model calls, tool calls, subagents,
//! background jobs) directly from raw session JSONL — the only source with
//! tool start/end times — while stats.db supplies the session-list rollups.
//! Synchronous `std::fs` like every other port; the TS async memo becomes a
//! process-global `Mutex`.

mod list;
mod scan;
#[cfg(test)]
mod tests;
mod tree;

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::paths::session_roots;
use crate::types::{SessionEntry, SessionTrace};

pub use list::list_session_summaries;

const MAX_TRACK_DEPTH: usize = 6;
const LABEL_MAX: usize = 80;
const DETAIL_MAX: usize = 160;
const TITLE_SCAN_BYTES: usize = 4096;
const LIST_FOLD_LIMIT: usize = 300;

/// Client-supplied path escaped the sessions root or is not a transcript.
/// Maps to HTTP 400 (the server downcasts the anyhow error to this type).
#[derive(Debug, Clone)]
pub struct TracePathError(String);

impl std::fmt::Display for TracePathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TracePathError {}

/// Trace payload schema/assembly revision, folded into the HTTP ETag. Bump
/// when span assembly changes so browsers don't revalidate stale cached
/// bodies against an unchanged transcript mtime.
pub const TRACE_ETAG_VERSION: i64 = 3;

/// Freshness of a root trace: root mtime plus the child-transcript set
/// fingerprint.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceFingerprint {
    pub root_mtime_ms: f64,
    pub child_fingerprint: String,
}

/// Parse every well-formed JSONL entry in `bytes` as a loose [`Value`]
/// (mirrors `parser::parse_all_session_entries`, but trace.ts re-validates
/// every field with `typeof`, so the loose view is what it actually uses).
fn parse_entry_values(bytes: &[u8]) -> Vec<Value> {
    let mut entries = Vec::new();
    for line in bytes.split(|b| *b == b'\n') {
        let mut end = line.len();
        while end > 0 && line[end - 1] == b'\r' {
            end -= 1;
        }
        if end == 0 {
            continue;
        }
        if let Ok(entry) = serde_json::from_slice::<Value>(&line[..end]) {
            entries.push(entry);
        }
    }
    entries
}

/// Resolve a client-supplied transcript path, enforcing sessions-root
/// containment (`path.resolve` + `path.relative` in TS — lexical, so a
/// best-effort normalization is enough: non-existent `..` segments must not
/// be symlink-resolved away).
pub(crate) fn resolve_session_path(file_param: &str) -> Result<PathBuf> {
    let raw = Path::new(file_param);
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(raw)
    };
    let resolved = normalize_lexically(&joined);
    // Accept a file under ANY session root (TS root and omp-rs root).
    let root = session_roots().into_iter().find(|root| {
        let rel = crate::paths::path_relative(&normalize_lexically(root), &resolved);
        let rel_str = rel.to_string_lossy();
        !(rel_str.is_empty() || rel.is_absolute() || rel_str.starts_with(".."))
    });
    let Some(root) = root else {
        anyhow::bail!(TracePathError(format!(
            "Path is outside the sessions directory: {file_param}"
        )));
    };
    let base = resolved
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !base.ends_with(".jsonl") && !base.ends_with(".jsonl.gz") {
        anyhow::bail!(TracePathError(format!(
            "Not a session transcript: {file_param}"
        )));
    }
    // Hardening beyond TS (m6): the lexical check above matches TS, but a
    // symlinked project/file can still escape the root. When the target
    // exists, re-check against the canonicalized root.
    if let (Ok(croot), Ok(ctarget)) = (
        std::fs::canonicalize(&root),
        std::fs::canonicalize(&resolved),
    ) {
        if !ctarget.starts_with(&croot) {
            anyhow::bail!(TracePathError(format!(
                "Path is outside the sessions directory: {file_param}"
            )));
        }
    }
    Ok(resolved)
}

/// Lexical `path.resolve`/`path.normalize`: drop `.` and apply `..` without
/// touching the filesystem.
fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Read a transcript, transparently gunzipping gc-compressed sessions.
pub(crate) fn read_transcript(resolved: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(resolved)?;
    if resolved.to_string_lossy().ends_with(".gz") {
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        Ok(std::fs::read(resolved)?)
    }
}

/// Node `stat.mtimeMs`: `sec * 1e3 + nsec / 1e6`.
#[cfg(unix)]
pub(crate) fn mtime_ms(meta: &std::fs::Metadata) -> f64 {
    use std::os::unix::fs::MetadataExt;
    meta.mtime() as f64 * 1000.0 + (meta.mtime_nsec() as f64) / 1_000_000.0
}

/// Portable `mtime_ms` via `metadata.modified()` (same formula, same values).
#[cfg(not(unix))]
pub(crate) fn mtime_ms(meta: &std::fs::Metadata) -> f64 {
    use std::time::UNIX_EPOCH;
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0.0, |d| {
            (d.as_secs() as f64) * 1000.0 + f64::from(d.subsec_nanos()) / 1_000_000.0
        })
}

/// ISO string → ms epoch, `None` when absent/unparseable (`Date.parse`).
pub(crate) fn to_ms(iso: Option<&Value>) -> Option<i64> {
    crate::parser::util::date_parse(iso?.as_str()?)
}

/// JS `\s` whitespace set (differs from `char::is_whitespace`: no U+0085,
/// yes U+FEFF).
fn js_whitespace(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t' | '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

/// Whitespace-collapsed head of a string, capped at `max` UTF-16 code units
/// like TS `headText` (`length`/`slice` count UTF-16 units; truncation backs
/// off to a char boundary rather than emitting a lone surrogate).
pub(crate) fn head_text(value: &str, max: usize) -> String {
    let collapsed: String = value
        .split(js_whitespace)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let units: Vec<usize> = collapsed
        .chars()
        .map(|c| c.len_utf16())
        .scan(0usize, |acc, n| {
            *acc += n;
            Some(*acc)
        })
        .collect();
    if units.last().map_or(0, |u| *u) > max {
        let head_len = units.partition_point(|&u| u < max); // chars fitting in max-1 units
        let head_len = head_len.min(max - 1);
        let head: String = collapsed.chars().take(head_len).collect();
        format!("{head}…")
    } else {
        collapsed
    }
}

/// First text-block head of a message content payload.
pub(crate) fn content_head(content: Option<&Value>, max: usize) -> Option<String> {
    let content = content?;
    if let Some(s) = content.as_str() {
        return if s.trim().is_empty() {
            None
        } else {
            Some(head_text(s, max))
        };
    }
    let blocks = content.as_array()?;
    for block in blocks {
        if block.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            if !text.trim().is_empty() {
                return Some(head_text(text, max));
            }
        }
    }
    None
}

/// Compact token count for marker labels (142_000 → "142k").
pub(crate) fn compact_tokens(value: Option<&Value>) -> String {
    match value.and_then(Value::as_f64) {
        Some(v) if v.is_finite() => {
            if v >= 1000.0 {
                format!("{}k", (v / 1000.0).round() as i64)
            } else {
                // JS String(number): "999.5" stays fractional, "999" drops ".0".
                format!("{v}")
            }
        }
        _ => "?".to_string(),
    }
}

/// JS truthiness (used for `msg.errorMessage` etc.).
pub(crate) fn js_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n
            .as_f64()
            .map(|f| f != 0.0 && f.is_finite())
            .unwrap_or(false),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(_)) | Some(Value::Object(_)) => true,
    }
}

/// Merge intervals and return covered length (mirrors `unionLength`).
pub(crate) fn union_length(mut intervals: Vec<(i64, i64)>) -> i64 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_unstable();
    let mut total = 0i64;
    let (mut cur_start, mut cur_end) = intervals[0];
    for (s, e) in intervals.into_iter().skip(1) {
        if s > cur_end {
            total = total.saturating_add(cur_end.saturating_sub(cur_start));
            cur_start = s;
            cur_end = e;
        } else if e > cur_end {
            cur_end = e;
        }
    }
    total.saturating_add(cur_end.saturating_sub(cur_start))
}

/// Strip the transcript extension (`.jsonl` / `.jsonl.gz`).
pub(crate) fn transcript_stem(file: &str) -> String {
    if let Some(s) = file.strip_suffix(".jsonl.gz") {
        s.to_string()
    } else {
        file.strip_suffix(".jsonl")
            .map_or_else(|| file.to_string(), ToString::to_string)
    }
}

/// Single-entry memo: the dashboard polls one open trace every 15s. Keyed on
/// the root file's mtime PLUS a fingerprint of the child-transcript SET under
/// the root's artifacts tree (one cheap readdir+stat walk, no parse). A
/// high-water mark alone is blind to deletions and to older-mtime
/// replacements, so the fingerprint encodes every child's (name, mtimeMs)
/// pair: any append, delete, rename, or content swap changes the key and the
/// memo misses.
struct TraceMemoEntry {
    mtime_ms: f64,
    child_fingerprint: String,
    trace: SessionTrace,
}

static TRACE_MEMO: Mutex<Option<(String, TraceMemoEntry)>> = Mutex::new(None);

#[cfg(test)]
pub fn trace_memo_for_tests() -> Option<(String, f64)> {
    let memo = TRACE_MEMO.lock().ok()?;
    let (file, entry) = memo.as_ref()?;
    Some((file.clone(), entry.mtime_ms))
}

/// Fingerprint of the child-transcript set: per-directory file COUNT plus
/// sorted `name@mtimeMs` pairs, mirroring buildTrackTree's discovery (stats
/// only, no parse). Best-effort: a raced gc deletion is recorded as
/// `name@gone` (which itself changes the fingerprint and misses the memo —
/// the safe direction).
pub(crate) fn child_transcripts_fingerprint(root_file: &str, depth: usize) -> String {
    if depth > MAX_TRACK_DEPTH {
        return String::new();
    }
    let dir = transcript_stem(root_file);
    let Ok(mut names) = read_dir_names(Path::new(&dir)) else {
        return String::new();
    };
    names.sort();
    let transcripts: Vec<&String> = names
        .iter()
        .filter(|n| n.ends_with(".jsonl") || n.ends_with(".jsonl.gz"))
        .collect();
    let mut parts = vec![format!("n={}", transcripts.len())];
    for name in transcripts {
        let child_file = format!("{dir}/{name}");
        match std::fs::metadata(&child_file) {
            Ok(meta) => {
                let nested = child_transcripts_fingerprint(&child_file, depth + 1);
                if nested.is_empty() {
                    parts.push(format!("{name}@{}", mtime_ms(&meta)));
                } else {
                    parts.push(format!("{name}@{},{name}/{{{nested}}}", mtime_ms(&meta)));
                }
            }
            Err(_) => parts.push(format!("{name}@gone")),
        }
    }
    parts.join(",")
}

pub(crate) fn read_dir_names(dir: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

/// Stat the root and walk its child transcripts (readdir+stat, no parse).
pub(crate) fn read_trace_fingerprint(resolved: &Path) -> Result<TraceFingerprint> {
    let root_mtime_ms = mtime_ms(&std::fs::metadata(resolved)?);
    let child_fingerprint = child_transcripts_fingerprint(&resolved.to_string_lossy(), 0);
    Ok(TraceFingerprint {
        root_mtime_ms,
        child_fingerprint,
    })
}

/**
 * Freshness fingerprint for the trace ETag pre-check; `SessionTrace.etag` is
 * `${rootMtimeMs}:${childFingerprint}` for an unraced build. Returns `None`
 * when the path is rejected or missing (the caller falls through to the full
 * build, which maps those to 400/404).
 */
pub fn trace_fingerprint_for_etag(file_param: &str) -> Option<TraceFingerprint> {
    let resolved = resolve_session_path(file_param).ok()?;
    read_trace_fingerprint(&resolved).ok()
}

/// Fetch one full journal entry by id from a transcript, for the span drawer.
/// ENOENT passes through as `Ok(None)` (the caller maps that to 404).
pub fn get_trace_entry(file_param: &str, entry_id: &str) -> Result<Option<SessionEntry>> {
    let resolved = resolve_session_path(file_param)?;
    let bytes = match read_transcript(&resolved) {
        Ok(bytes) => bytes,
        Err(err)
            if err.downcast_ref::<std::io::Error>().map(|e| e.kind())
                == Some(ErrorKind::NotFound) =>
        {
            return Ok(None)
        }
        Err(err) => return Err(err),
    };
    for entry in parse_entry_values(&bytes) {
        if entry.get("id").and_then(Value::as_str) == Some(entry_id) {
            // TS returns the parsed entry unchanged; `Other` serializes the
            // raw JSON verbatim (untagged), keeping fields like compaction
            // payloads or unknown keys intact.
            return Ok(Some(SessionEntry::Other(entry)));
        }
    }
    Ok(None)
}

/**
 * Assemble the full span tree for one root session transcript.
 * Errors with [`TracePathError`] for paths outside the sessions root; ENOENT
 * passes through for the caller's 404 mapping. `fingerprint` may carry the
 * [`trace_fingerprint_for_etag`] value the caller just computed for the same
 * file (its ETag pre-check), sparing a second walk.
 */
pub fn build_session_trace(
    file_param: &str,
    fingerprint: Option<&TraceFingerprint>,
) -> Result<SessionTrace> {
    let resolved = resolve_session_path(file_param)?;
    let resolved_str = resolved.to_string_lossy().into_owned();
    let (root_mtime_ms, pre_child_fingerprint) = match fingerprint {
        Some(fp) => (fp.root_mtime_ms, fp.child_fingerprint.clone()),
        None => {
            let fp = read_trace_fingerprint(&resolved)?;
            (fp.root_mtime_ms, fp.child_fingerprint)
        }
    };
    // Pre-build child fingerprint on EVERY path (hit check and rebuild alike):
    // the post-build race guard compares against this. Memo equality, not <=:
    // a LOWER mark (delete/older-mtime replace) is also stale.
    {
        let memo = TRACE_MEMO
            .lock()
            .map_err(|_| anyhow::anyhow!("trace memo poisoned"))?;
        if let Some((file, entry)) = memo.as_ref() {
            if file == &resolved_str
                && entry.mtime_ms == root_mtime_ms
                && entry.child_fingerprint == pre_child_fingerprint
            {
                return Ok(entry.trace.clone());
            }
        }
    }

    let mut tracks = Vec::new();
    let mut scans = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let root_scan_idx = tree::build_track_tree(
        &resolved_str,
        "main",
        None,
        0,
        &mut visited,
        &mut tracks,
        &mut scans,
    )
    .with_context(|| format!("build track tree for {resolved_str}"))?;
    let Some(root_scan_idx) = root_scan_idx else {
        anyhow::bail!(std::io::Error::new(
            ErrorKind::NotFound,
            format!("session not found: {resolved_str}")
        ));
    };
    let root_index = tracks
        .iter()
        .position(|t| t.id == "main")
        .expect("root track id main");
    tracks[root_index].label = "Main".to_string();

    let mut started_at: Option<i64> = None;
    let mut ended_at: Option<i64> = None;
    let mut intervals: Vec<(i64, i64)> = Vec::new();
    let mut tool_calls = 0i64;
    let mut requests = 0i64;
    let mut total_tokens = 0i64;
    let mut cost_total = 0f64;
    let mut unpriced_requests = 0i64;
    // TS Map preserves insertion order; the final sort is stable on totalMs
    // desc only, so first-appearance order is the tie-break. `scans` already
    // holds the root at index 0 (pushed before recursion, TS DFS order).
    let mut tool_stats: Vec<crate::types::TraceToolStat> = Vec::new();
    let mut tool_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for scan in &scans {
        requests += scan.requests;
        tool_calls += scan.tool_calls;
        unpriced_requests += scan.unpriced_requests;
        for span in &scan.track.spans {
            intervals.push((span.start, span.end));
            started_at = Some(started_at.map_or(span.start, |s| s.min(span.start)));
            ended_at = Some(ended_at.map_or(span.end, |e| e.max(span.end)));
            match span.kind {
                crate::types::TraceSpanKind::Model => {
                    total_tokens += span.tokens.unwrap_or(0);
                    cost_total += span.cost.unwrap_or(0.0);
                }
                crate::types::TraceSpanKind::Tool => {
                    let duration = span.end.saturating_sub(span.start);
                    let idx = *tool_index.entry(span.label.clone()).or_insert_with(|| {
                        tool_stats.push(crate::types::TraceToolStat {
                            tool: span.label.clone(),
                            calls: 0,
                            errors: 0,
                            total_ms: 0,
                            max_ms: 0,
                        });
                        tool_stats.len() - 1
                    });
                    let agg = &mut tool_stats[idx];
                    agg.calls += 1;
                    if span.is_error == Some(true) {
                        agg.errors += 1;
                    }
                    agg.total_ms = agg.total_ms.saturating_add(duration);
                    agg.max_ms = agg.max_ms.max(duration);
                }
                _ => {}
            }
        }
    }

    let mut model_ms = 0i64;
    let mut tool_ms = 0i64;
    let mut turns = 0i64;
    for span in &tracks[root_index].spans {
        match span.kind {
            crate::types::TraceSpanKind::Model => {
                model_ms = model_ms.saturating_add(span.end.saturating_sub(span.start))
            }
            crate::types::TraceSpanKind::Tool => {
                tool_ms = tool_ms.saturating_add(span.end.saturating_sub(span.start))
            }
            crate::types::TraceSpanKind::Turn => turns += 1,
            _ => {}
        }
    }

    let start = started_at
        .or(scans[root_scan_idx].header_ts)
        .unwrap_or(root_mtime_ms as i64);
    let end = ended_at.unwrap_or(start);
    let wall_ms = end.saturating_sub(start);

    tool_stats.sort_by_key(|s| std::cmp::Reverse(s.total_ms));

    let summary = crate::types::TraceSummary {
        wall_ms,
        model_ms,
        tool_ms,
        idle_ms: wall_ms.saturating_sub(union_length(intervals)).max(0),
        turns,
        requests,
        tool_calls,
        subagents: tracks.len() as i64 - 1,
        total_tokens,
        cost_total,
        unpriced_requests,
        tool_stats,
    };

    // Re-collect the fingerprint AFTER the parse: a child append landing
    // between the pre-build check and now would otherwise be baked into
    // neither the trace nor the key. Raced builds keep the PRE-build
    // fingerprint as their ETag so the next conditional poll misses and
    // rebuilds — at most one stale response per race, never an indefinitely
    // cached one.
    let child_fingerprint = child_transcripts_fingerprint(&resolved_str, 0);
    let root_changed = std::fs::metadata(&resolved)
        .ok()
        .map(|m| mtime_ms(&m) != root_mtime_ms)
        .unwrap_or(true);
    let raced = child_fingerprint != pre_child_fingerprint;
    let etag_fingerprint = if raced || root_changed {
        pre_child_fingerprint.clone()
    } else {
        child_fingerprint.clone()
    };
    let trace = SessionTrace {
        file: resolved_str.clone(),
        title: scans[root_scan_idx].title.clone(),
        cwd: scans[root_scan_idx].cwd.clone(),
        started_at: start,
        ended_at: end,
        mtime_ms: root_mtime_ms,
        etag: format!("{root_mtime_ms}:{etag_fingerprint}"),
        tracks,
        summary,
    };
    if !raced && !root_changed {
        let mut memo = TRACE_MEMO
            .lock()
            .map_err(|_| anyhow::anyhow!("trace memo poisoned"))?;
        *memo = Some((
            resolved_str,
            TraceMemoEntry {
                mtime_ms: root_mtime_ms,
                child_fingerprint,
                trace: trace.clone(),
            },
        ));
    }
    Ok(trace)
}
