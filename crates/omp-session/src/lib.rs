//! omp session files (JSONL, format v3) on top of `pa_types::session::FileEntry`.
//!
//! The entry model is prime-agent's (MIT, derived from the same pi lineage as omp); this crate adds
//! what omp needs around it: a line-preserving reader, a writer that chains `id`/`parentId` the way
//! omp's session manager does, and a round-trip checker used against real `~/.omp/agent/sessions`.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pa_types::session::{AgentMessage, EntryBase, FileEntry};
use serde_json::{json, Map, Value};

pub use pa_types::session;

pub const SESSION_VERSION: u32 = 3;

/// One line of a session file: the parsed entry plus the raw text it came from.
pub struct Line {
    pub raw: String,
    pub entry: FileEntry,
}

/// Read every non-empty line. Lines that are not JSON objects are an error (omp never writes them).
///
/// # Errors
/// I/O failure, or a line that is not a JSON object.
pub fn read(path: &Path) -> Result<Vec<Line>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let raw = line.with_context(|| format!("{}:{}", path.display(), n + 1))?;
        if raw.trim().is_empty() {
            continue;
        }
        let entry: FileEntry = serde_json::from_str(&raw).with_context(|| format!("{}:{}", path.display(), n + 1))?;
        out.push(Line { raw, entry });
    }
    Ok(out)
}

/// Append-only writer producing files omp itself can open: v3 header first, then entries whose
/// `parentId` is the previous entry's `id` (a linear branch, as omp writes without /tree navigation).
pub struct SessionWriter {
    file: File,
    path: PathBuf,
    last_id: Option<String>,
}

impl SessionWriter {
    /// Create `<dir>/<iso-timestamp>_<id>.jsonl` with a v3 header (omp's naming scheme).
    ///
    /// # Errors
    /// I/O failure creating the directory or file.
    pub fn create(dir: &Path, cwd: &str, session_id: &str, timestamp_iso: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let stamp = timestamp_iso.replace([':', '.'], "-");
        let path = dir.join(format!("{stamp}_{session_id}.jsonl"));
        let file = OpenOptions::new().create_new(true).append(true).open(&path)
            .with_context(|| format!("create {}", path.display()))?;
        let mut w = Self { file, path, last_id: None };
        w.write_value(&json!({"type": "session", "version": SESSION_VERSION, "id": session_id, "timestamp": timestamp_iso, "cwd": cwd}))?;
        Ok(w)
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a `message` entry; returns the new entry id.
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_message(&mut self, message: &AgentMessage, timestamp_iso: &str) -> Result<String> {
        // Serialize through the typed entry (not a Value detour) so field order matches omp's writer.
        let id = new_entry_id();
        let entry = FileEntry::Message {
            message: message.clone(),
            base: EntryBase { id: Some(id.clone()), parent_id: self.last_id.clone(), timestamp: Some(timestamp_iso.into()), rest: Map::new() },
        };
        let mut line = serde_json::to_string(&entry)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.last_id = Some(id.clone());
        Ok(id)
    }

    /// Append any entry kind (`compaction`, `custom_message`, `model_change`, ...). `payload` holds
    /// the kind-specific fields; `type`, `id`, `parentId` and `timestamp` are filled in here.
    ///
    /// # Errors
    /// `payload` is not a JSON object, or I/O failure.
    pub fn append_entry(&mut self, kind: &str, payload: Value, timestamp_iso: &str) -> Result<String> {
        let Value::Object(fields) = payload else { anyhow::bail!("entry payload must be an object") };
        let id = new_entry_id();
        let mut obj = Map::new();
        obj.insert("type".into(), Value::String(kind.into()));
        obj.insert("id".into(), Value::String(id.clone()));
        obj.insert("parentId".into(), self.last_id.clone().map_or(Value::Null, Value::String));
        obj.insert("timestamp".into(), Value::String(timestamp_iso.into()));
        obj.extend(fields);
        self.write_value(&Value::Object(obj))?;
        self.last_id = Some(id.clone());
        Ok(id)
    }

    fn write_value(&mut self, v: &Value) -> Result<()> {
        let mut line = serde_json::to_string(v)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        Ok(())
    }
}

/// omp entry ids are 8 lowercase hex chars.
fn new_entry_id() -> String {
    let n: u32 = rand_u32();
    format!("{n:08x}")
}

fn rand_u32() -> u32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    #[allow(clippy::cast_possible_truncation)]
    let v = h.finish() as u32;
    v
}

/// How a line survives parse → serialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundTrip {
    /// Byte-identical.
    Exact,
    /// Same JSON value, different bytes (key order / number formatting / escaping).
    SameValue,
    /// Parsed into a modeled kind but the value changed (fields dropped or rewritten).
    Changed,
    /// Kept verbatim as `Unknown` (type not modeled, or payload failed validation).
    Unknown,
}

#[must_use]
pub fn round_trip(line: &Line) -> RoundTrip {
    let Ok(back) = serde_json::to_string(&line.entry) else { return RoundTrip::Changed };
    if back == line.raw {
        return RoundTrip::Exact;
    }
    if matches!(line.entry, FileEntry::Unknown { .. }) {
        return RoundTrip::Unknown;
    }
    let a: Option<Value> = serde_json::from_str(&line.raw).ok();
    let b: Option<Value> = serde_json::from_str(&back).ok();
    if a.is_some() && a == b { RoundTrip::SameValue } else { RoundTrip::Changed }
}

/// For an `Unknown` line whose `type` IS a modeled kind: why the typed parse failed.
#[must_use]
pub fn unknown_reason(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    let kind = v.get("type")?.as_str()?.to_owned();
    if kind == "message" {
        let msg = v.get("message")?.clone();
        return serde_json::from_value::<AgentMessage>(msg).err().map(|e| format!("message: {e}"));
    }
    Some(format!("type={kind}"))
}

/// Paths of JSON values that differ between two documents (first `limit`), for Changed lines.
#[must_use]
pub fn diff_paths(a: &Value, b: &Value, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    walk(a, b, String::new(), &mut out, limit);
    out
}

fn walk(a: &Value, b: &Value, path: String, out: &mut Vec<String>, limit: usize) {
    if out.len() >= limit || a == b {
        return;
    }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                let p = format!("{path}.{k}");
                match (x.get(k), y.get(k)) {
                    (Some(u), Some(v)) => walk(u, v, p, out, limit),
                    (Some(_), None) => out.push(format!("{p} dropped")),
                    (None, Some(_)) => out.push(format!("{p} added")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                walk(u, v, format!("{path}[{i}]"), out, limit);
            }
        }
        _ => out.push(format!("{path} {a:.60} -> {b:.60}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_output_reads_back_and_chains_parents() {
        let dir = std::env::temp_dir().join(format!("omp-session-test-{}", rand_u32()));
        let mut w = SessionWriter::create(&dir, "/tmp", "01a00000-0000-7000-8000-000000000000", "2026-10-11T00:00:00.000Z").unwrap();
        let msg: AgentMessage = serde_json::from_value(json!({"role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 1})).unwrap();
        let first = w.append_message(&msg, "2026-10-11T00:00:01.000Z").unwrap();
        w.append_entry("custom_message", json!({"customType": "note", "content": "x", "display": false}), "2026-10-11T00:00:02.000Z").unwrap();
        let lines = read(w.path()).unwrap();
        assert_eq!(lines.len(), 3);
        assert!(matches!(lines[0].entry, FileEntry::Header { .. }));
        assert!(matches!(lines[1].entry, FileEntry::Message { .. }));
        assert!(matches!(lines[2].entry, FileEntry::CustomMessage { .. }));
        let v: Value = serde_json::from_str(&lines[2].raw).unwrap();
        assert_eq!(v["parentId"], json!(first));
        for l in &lines {
            assert_eq!(round_trip(l), RoundTrip::Exact, "{}", l.raw);
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
