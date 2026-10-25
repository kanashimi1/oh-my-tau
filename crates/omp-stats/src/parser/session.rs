//! File access layer from `parser.ts`: byte-level lenient JSONL scanning and
//! the incremental `parse_session_file` with the wyhash checkpoint protocol.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use anyhow::Result;
use serde_json::Value;
use walkdir::WalkDir;

use crate::types::{AgentType, ServiceTierByFamily, SessionEntry};

use super::entries::{
    extract_model_usage_stats, extract_stats, extract_tool_calls, extract_tool_result_link,
    extract_user_stats, is_assistant_message, is_model_usage, is_service_tier_change,
    is_tool_result_message, is_user_message,
};
use super::state::{ParseSessionResult, SessionParserState};
use super::tier::coerce_service_tier_by_family;
use super::wyhash::wyhash_hex;

/// Basename of an advisor agent's transcript inside a session artifacts dir.
const LF: u8 = 0x0a;
const CR: u8 = 0x0d;

/// Bytes hashed for the rewrite-detection checkpoint (`CHECKPOINT_BYTES`).
pub(crate) const CHECKPOINT_BYTES: i64 = 256;

/// Classify which agent produced a transcript from its path within the
/// sessions directory (wire-string core in [`crate::paths`]).
#[must_use]
pub fn classify_agent_type(session_path: &str) -> AgentType {
    match crate::paths::classify_agent_type_str(session_path) {
        "advisor" => AgentType::Advisor,
        "subagent" => AgentType::Subagent,
        _ => AgentType::Main,
    }
}

/// Extract folder name from session filename. Session files are named like
/// `--work--pi--/timestamp_uuid.jsonl`; the folder part uses `--` as path
/// separator, so `--work--pi--` becomes `/work/pi`.
#[must_use]
pub fn extract_folder_from_path(session_path: &str) -> String {
    // Relative to the containing root; omp-rs slugs are mapped to the TS
    // project dir name of the same cwd so a project merges across agents.
    let project_dir = crate::paths::session_project_dir(session_path);
    project_dir.replacen("--", "/", 1).replace("--", "/")
}

/// Parse one JSONL line from `bytes[start..end]`; trailing CRs trimmed.
fn parse_json_line(bytes: &[u8], start: i64, end: i64) -> Option<Value> {
    let (start, mut end) = (start as usize, end as usize);
    while end > start && bytes[end - 1] == CR {
        end -= 1;
    }
    if end <= start {
        return None;
    }
    serde_json::from_slice(&bytes[start..end]).ok()
}

/// Walk whole lines of `bytes`, visiting well-formed entries; malformed
/// lines are skipped. Returns the offset after the last complete entry
/// (a partial final line is not counted).
pub(crate) fn visit_session_entries_lenient(bytes: &[u8], visit: &mut dyn FnMut(&Value)) -> i64 {
    let mut cursor = 0usize;
    let mut read = 0i64;

    while cursor < bytes.len() {
        let newline = bytes[cursor..]
            .iter()
            .position(|b| *b == LF)
            .map(|p| p + cursor);
        let has_newline = newline.is_some();
        let line_end = newline.unwrap_or(bytes.len());
        if let Some(entry) = parse_json_line(bytes, cursor as i64, line_end as i64) {
            visit(&entry);
            read = if has_newline {
                line_end as i64 + 1
            } else {
                line_end as i64
            };
        } else if has_newline {
            read = line_end as i64 + 1;
        } else {
            break;
        }
        cursor = if has_newline { line_end + 1 } else { line_end };
    }

    read
}

/// Parse every well-formed entry in a transcript buffer (malformed lines
/// skipped).
#[must_use]
pub fn parse_all_session_entries(bytes: &[u8]) -> Vec<SessionEntry> {
    let mut entries = Vec::new();
    visit_session_entries_lenient(bytes, &mut |entry| {
        if let Ok(e) = serde_json::from_value::<SessionEntry>(entry.clone()) {
            entries.push(e);
        }
    });
    entries
}

/// Fold the last persisted `service_tier_change` entry of the prefix into the
/// live tier state (offset-only callers reconstruct it once).
fn scan_last_service_tier(bytes: &[u8]) -> Option<ServiceTierByFamily> {
    let mut current = None;
    visit_session_entries_lenient(bytes, &mut |entry| {
        if is_service_tier_change(entry) {
            current = entry
                .get("serviceTier")
                .and_then(coerce_service_tier_by_family);
        }
    });
    current
}

/// Node `stat.mtimeMs`: `sec * 1e3 + nsec / 1e6`.
#[cfg(unix)]
fn mtime_ms(meta: &std::fs::Metadata) -> f64 {
    use std::os::unix::fs::MetadataExt;
    (meta.mtime() as f64) * 1000.0 + (meta.mtime_nsec() as f64) / 1_000_000.0
}

/// Portable `mtime_ms` via `metadata.modified()` (same formula, same values).
#[cfg(not(unix))]
fn mtime_ms(meta: &std::fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0.0, |d| {
            (d.as_secs() as f64) * 1000.0 + f64::from(d.subsec_nanos()) / 1_000_000.0
        })
}

/// Node/Bun `stat.birthtimeMs`: `sec * 1e3 + nsec / 1e6` (fractional, so a
/// TS-written parser state compares equal).
fn metadata_birthtime_ms(meta: &std::fs::Metadata) -> f64 {
    meta.created()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0.0, |d| {
            (d.as_secs() as f64) * 1000.0 + f64::from(d.subsec_nanos()) / 1_000_000.0
        })
}

/// File identity `(dev, ino)` as `f64`, matching TS `stat.dev`/`stat.ino`.
#[cfg(unix)]
fn file_identity(meta: &std::fs::Metadata, _path: &std::path::Path) -> (f64, f64) {
    use std::os::unix::fs::MetadataExt;
    (meta.dev() as f64, meta.ino() as f64)
}

/// Windows identity key: `std::os::windows::fs::MetadataExt` handle fields are
/// unstable, so use a stable hash of the canonical path as the `ino` stand-in
/// (`dev` = 0). A moved/renamed file re-parses from scratch — safe, only
/// redundant; the path is stable for the lifetime of a transcript.
#[cfg(windows)]
fn file_identity(meta: &std::fs::Metadata, path: &std::path::Path) -> (f64, f64) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let _ = meta;
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    (0.0, hasher.finish() as f64)
}

/// Same file identity check as TS `matchesSessionFile`: dev, ino and
/// birthtime must all match (size/mtime changes are the rewrite signal).
/// `session_path` is unused on Unix (dev/ino come from the metadata) and is
/// the identity key on Windows.
#[must_use]
pub fn matches_session_file(
    state: &SessionParserState,
    session_path: &str,
    info: &std::fs::Metadata,
) -> bool {
    let (dev, ino) = file_identity(info, std::path::Path::new(session_path));
    state.dev == dev && state.ino == ino && state.birthtime_ms == metadata_birthtime_ms(info)
}

/// Positional read of the last `CHECKPOINT_BYTES` bytes before `end`.
fn read_checkpoint(handle: &mut File, end: i64) -> Result<Vec<u8>> {
    let start = (end - CHECKPOINT_BYTES).max(0);
    let mut bytes = vec![0u8; (end - start) as usize];
    handle.seek(SeekFrom::Start(start as u64))?;
    handle.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn empty_result(from_offset: i64) -> ParseSessionResult {
    ParseSessionResult {
        new_offset: from_offset,
        ..ParseSessionResult::default()
    }
}

/// Incrementally parse a session file from `from_offset`. With a persisted
/// `state` whose offset matches, only the appended tail is read and the
/// rewrite check (size/mtime + checkpoint hash) decides between resume and
/// full replay. `replay` forces a full rescan without changing `reset`.
///
/// # Errors
/// Propagates I/O errors other than ENOENT (which yields an empty result at
/// the same offset, like TS).
pub fn parse_session_file(
    session_path: &str,
    from_offset: i64,
    state: Option<&SessionParserState>,
    replay: bool,
) -> Result<ParseSessionResult> {
    let mut handle = match File::open(session_path) {
        Ok(h) => h,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(empty_result(from_offset)),
        Err(e) => return Err(e.into()),
    };
    let info = handle.metadata()?;

    let folder = extract_folder_from_path(session_path);
    let agent_type = classify_agent_type(session_path);
    let mut result = ParseSessionResult::default();

    // Reduce each entry immediately so full replays do not retain
    // tool-output/message bodies.
    let visit = |entry: &Value,
                 result: &mut ParseSessionResult,
                 tier: &mut Option<ServiceTierByFamily>| {
        if is_service_tier_change(entry) {
            *tier = entry
                .get("serviceTier")
                .and_then(coerce_service_tier_by_family);
            return;
        }
        if is_user_message(entry) {
            if let Some(user_stats) = extract_user_stats(session_path, &folder, entry) {
                result.user_stats.push(user_stats);
            }
            return;
        }
        if is_tool_result_message(entry) {
            if let Some(link) = extract_tool_result_link(session_path, entry) {
                result.tool_results.push(link);
            }
            return;
        }
        if is_model_usage(entry) {
            if let Some(stats) = extract_model_usage_stats(session_path, &folder, entry, agent_type)
            {
                result.stats.push(stats);
            }
            return;
        }
        if is_assistant_message(entry) {
            if let Some(msg_stats) =
                extract_stats(session_path, &folder, entry, tier.as_ref(), agent_type)
            {
                result.stats.push(msg_stats);
            }
            let calls = extract_tool_calls(session_path, &folder, entry, agent_type);
            result.tool_calls.extend(calls);
            // Persist links even when the user entry was ingested in an
            // earlier tail read.
            let msg = entry.get("message").expect("assistant entry has message");
            let parent_id = entry.get("parentId").and_then(Value::as_str);
            let model = msg.get("model").and_then(Value::as_str);
            let provider = msg.get("provider").and_then(Value::as_str);
            if let (Some(parent_id), Some(model), Some(provider)) = (parent_id, model, provider) {
                if !parent_id.is_empty() && !model.is_empty() && !provider.is_empty() {
                    result.user_links.push(crate::types::UserMessageLink {
                        session_file: session_path.to_owned(),
                        entry_id: parent_id.to_owned(),
                        model: model.to_owned(),
                        provider: provider.to_owned(),
                    });
                }
            }
        }
    };

    let file_size = info.len() as i64;
    let mut resume = state.is_some_and(|s| s.version == 1 && s.offset == from_offset);
    let mut reset = false;
    if let (true, Some(state)) = (resume, state) {
        reset = !matches_session_file(state, session_path, &info)
            || file_size < state.size
            || (file_size == state.size && mtime_ms(&info) != state.mtime_ms);
        if !reset {
            let previous = read_checkpoint(&mut handle, from_offset)?;
            reset = wyhash_hex(&previous) != state.checkpoint;
        }
        resume = !reset;
    }
    if from_offset > file_size {
        reset = true;
    }
    if replay {
        resume = false;
    }
    let start = if reset || replay {
        0
    } else {
        from_offset.max(0)
    };
    let read_start = if resume { start } else { 0 };

    handle.seek(SeekFrom::Start(read_start as u64))?;
    let mut bytes = Vec::with_capacity((file_size - read_start).max(0) as usize);
    handle.read_to_end(&mut bytes)?;

    let mut tier = if resume {
        state.and_then(|s| s.service_tier.clone())
    } else {
        scan_last_service_tier(&bytes[..start as usize])
    };
    let region = &bytes[(start - read_start) as usize..];
    let read = visit_session_entries_lenient(region, &mut |entry| {
        visit(entry, &mut result, &mut tier);
    });
    let current_service_tier = tier;
    let new_offset = start + read;
    let checkpoint_start = (new_offset - CHECKPOINT_BYTES).max(0);
    let checkpoint = if checkpoint_start >= read_start {
        let lo = (checkpoint_start - read_start) as usize;
        let hi = (new_offset - read_start) as usize;
        wyhash_hex(&bytes[lo..hi])
    } else {
        wyhash_hex(&read_checkpoint(&mut handle, new_offset)?)
    };

    result.new_offset = new_offset;
    result.reset = Some(reset);
    let (dev, ino) = file_identity(&info, std::path::Path::new(session_path));
    result.parser_state = Some(SessionParserState {
        version: 1,
        offset: new_offset,
        dev,
        ino,
        birthtime_ms: metadata_birthtime_ms(&info),
        size: file_size,
        mtime_ms: mtime_ms(&info),
        checkpoint,
        service_tier: current_service_tier,
    });
    Ok(result)
}

/// List all session directories (folders).
#[must_use]
pub fn list_session_folders() -> Vec<String> {
    let mut out = Vec::new();
    for root in crate::paths::session_roots() {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                out.push(entry.path().to_string_lossy().into_owned());
            }
        }
    }
    out
}

/// List all session files in a folder (recursive, `*.jsonl`).
#[must_use]
pub fn list_session_files(folder_path: &str) -> Vec<String> {
    let mut out = Vec::new();
    for entry in WalkDir::new(folder_path).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file() {
            let name = entry.file_name().to_string_lossy();
            if name.ends_with(".jsonl") {
                out.push(entry.path().to_string_lossy().into_owned());
            }
        }
    }
    out
}

/// List all session files across all folders.
#[must_use]
pub fn list_all_session_files() -> Vec<String> {
    list_session_folders()
        .iter()
        .flat_map(|folder| list_session_files(folder))
        .collect()
}

/// Find a specific entry in a session file. Returns `Ok(None)` for ENOENT.
///
/// # Errors
/// Propagates non-ENOENT I/O errors.
pub fn get_session_entry(session_path: &str, entry_id: &str) -> Result<Option<SessionEntry>> {
    let file = match File::open(session_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let mut reader = BufReader::new(file);
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(LF, &mut line)? == 0 {
            return Ok(None);
        }
        while line.last() == Some(&LF) || line.last() == Some(&CR) {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        let Ok(entry) = serde_json::from_slice::<Value>(&line) else {
            continue;
        };
        if entry.get("id").and_then(Value::as_str) == Some(entry_id) {
            return Ok(serde_json::from_value::<SessionEntry>(entry).ok());
        }
    }
}
