//! Lazy session index: one cheap pass over the journal records
//! `(line, offset, len, id, parentId, kind)` per entry; full entries are parsed
//! on demand and only for the active branch.
//!
//! omp (TS) resumes with the whole parsed journal in memory
//! (`SessionManager.#setSessionFile` → `#applyEntries`), but everything the
//! continuation needs — the model context (`buildSessionContext`), the todo
//! rehydration (`getLatestTodoPhasesFromEntries(getBranch())`), the seeded
//! continuation file — derives from the leaf branch only
//! (`SessionEntryIndex.pathTo(leaf)`). This index keeps just the links, so a
//! 40 MB journal costs O(entries) small fixed records until a consumer parses
//! (part of) the branch.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::{entries, Malformed};

/// The entry kinds the lazy loader distinguishes (everything else is [`EntryKind::Other`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// `session` header.
    Header,
    /// Fixed-width first-line title slot (`{"type":"title","v":1,...}`).
    Title,
    Message,
    Compaction,
    ResetBoundary,
    BranchSummary,
    CustomMessage,
    Custom,
    Label,
    ThinkingLevelChange,
    ModelChange,
    ServiceTierChange,
    TtsrInjection,
    ModeChange,
    /// Any other (or missing) `type`.
    Other,
}

fn kind_of(type_field: Option<&str>) -> EntryKind {
    match type_field {
        Some("session") => EntryKind::Header,
        Some("title") => EntryKind::Title,
        Some("message") => EntryKind::Message,
        Some("compaction") => EntryKind::Compaction,
        Some("reset_boundary") => EntryKind::ResetBoundary,
        Some("branch_summary") => EntryKind::BranchSummary,
        Some("custom_message") => EntryKind::CustomMessage,
        Some("custom") => EntryKind::Custom,
        Some("label") => EntryKind::Label,
        Some("thinking_level_change") => EntryKind::ThinkingLevelChange,
        Some("model_change") => EntryKind::ModelChange,
        Some("service_tier_change") => EntryKind::ServiceTierChange,
        Some("ttsr_injection") => EntryKind::TtsrInjection,
        Some("mode_change") => EntryKind::ModeChange,
        _ => EntryKind::Other,
    }
}

/// The cheap per-line parse: top-level `type`/`id`/`parentId` only.
#[derive(serde::Deserialize)]
struct Head {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "parentId", default)]
    parent_id: Option<String>,
    #[serde(default)]
    timestamp: Option<String>,
}

/// One journal entry's index record.
#[derive(Debug)]
pub struct EntryRef {
    /// 1-based line number in the file.
    pub line_no: u32,
    /// Byte offset of the line start.
    pub offset: u64,
    /// Byte length of the line (newline excluded).
    pub len: u32,
    /// Entry id (`""` when the record has none).
    pub id: Box<str>,
    /// `parentId` (`""` when null / missing — a root).
    pub parent_id: Box<str>,
    /// Entry timestamp (`""` when missing).
    pub timestamp: Box<str>,
    /// The entry kind, from its `type` field.
    pub kind: EntryKind,
    /// The exact `type` string (`""` when missing) — shims must reproduce it.
    pub type_name: Box<str>,
}

/// A lazily-loaded session: the link index plus loader diagnostics, like
/// [`crate::Loaded`] but without any parsed entry.
pub struct SessionIndex {
    /// Records in file order; `refs[0]` is the title slot or the header.
    pub refs: Vec<EntryRef>,
    /// Records skipped by the lenient scan (non-JSON or non-object lines).
    pub malformed: Vec<Malformed>,
    /// The fixed-width title slot, when the first line is one.
    pub title_slot: Option<entries::TitleSlot>,
    /// Non-empty data without a valid leading `session` header (title slot excluded).
    pub invalid_header: bool,
    /// File size in bytes.
    pub size: u64,
}

/// One branch entry loaded on demand: the raw line plus, when requested, the parsed value.
pub struct BranchLine {
    /// The line text (lossily decoded, `\r` stripped), exactly as written.
    pub raw: String,
    /// The entry kind from the index.
    pub kind: EntryKind,
    /// The parsed entry, present only when `load_branch` was asked to parse values.
    pub value: Option<Value>,
}

impl SessionIndex {
    /// Scan `path` once, recording links only — no entry is fully parsed here.
    ///
    /// # Errors
    /// I/O failure.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut refs = Vec::new();
        let mut malformed = Vec::new();
        let mut offset = 0u64;
        for (n, chunk) in BufReader::new(file).split(b'\n').enumerate() {
            let bytes = chunk.with_context(|| format!("{}:{}", path.display(), n + 1))?;
            let line_len = bytes.len();
            let text = String::from_utf8_lossy(&bytes);
            let raw = text.strip_suffix('\r').unwrap_or(&text);
            if raw.trim().is_empty() {
                offset += line_len as u64 + 1;
                continue;
            }
            match head_of(&bytes) {
                Some(head) => refs.push(EntryRef {
                    line_no: n as u32 + 1,
                    offset,
                    len: line_len as u32,
                    id: head.id.unwrap_or_default().into_boxed_str(),
                    parent_id: head.parent_id.unwrap_or_default().into_boxed_str(),
                    timestamp: head.timestamp.unwrap_or_default().into_boxed_str(),
                    kind: kind_of(head.kind.as_deref()),
                    type_name: head.kind.unwrap_or_default().into_boxed_str(),
                }),
                None => malformed.push(Malformed {
                    line_no: n + 1,
                    raw: raw.to_owned(),
                }),
            }
            offset += line_len as u64 + 1;
        }

        // First line: fixed-width title slot?
        let mut title_slot = None;
        if let Some(first) = refs.first().filter(|r| r.kind == EntryKind::Title) {
            if let Some(raw) = read_line(path, first.offset, first.len).ok().flatten() {
                if let Some(entries::OmpEntry::Title(slot)) = entries::OmpEntry::parse(&raw) {
                    if slot.v == 1 {
                        title_slot = Some(slot);
                    }
                }
            }
        }

        // Header validation, mirroring `load`: first non-slot record must be a
        // `session` header with an id; a file with no records at all is invalid
        // only when something malformed was skipped.
        let slot_lines = u32::from(title_slot.is_some());
        let mut body = refs.iter().skip(slot_lines as usize);
        let invalid_header = match body.next() {
            Some(r) => r.kind != EntryKind::Header || r.id.is_empty(),
            None => !malformed.is_empty(),
        };

        Ok(Self {
            refs,
            malformed,
            title_slot,
            invalid_header,
            size,
        })
    }

    /// Ids of the active branch, root → leaf (TS `SessionEntryIndex.pathTo(leafId)`).
    ///
    /// The leaf is the last journaled entry (`#applyEntries` inserts in file
    /// order and leaves `#leaf` on the last one); the walk follows `parentId`
    /// links and stops at an unknown parent or the first repeated id, exactly
    /// like the TS memoized branch. Indices index into [`Self::refs`].
    #[must_use]
    pub fn branch(&self) -> Vec<u32> {
        let is_structural = |i: usize| {
            let r = &self.refs[i];
            r.kind == EntryKind::Header || (r.kind == EntryKind::Title && i == 0)
        };
        let mut by_id: HashMap<&str, u32> = HashMap::with_capacity(self.refs.len());
        for (i, r) in self.refs.iter().enumerate() {
            if !is_structural(i) && !r.id.is_empty() {
                by_id.insert(&r.id, i as u32);
            }
        }
        let Some(mut cur) = (0..self.refs.len()).rposition(|i| !is_structural(i)) else {
            return Vec::new();
        };
        let mut path = Vec::new();
        let mut seen: HashSet<u32> = HashSet::new();
        loop {
            if !seen.insert(cur as u32) {
                break;
            }
            path.push(cur as u32);
            let parent = &self.refs[cur].parent_id;
            if parent.is_empty() {
                break;
            }
            match by_id.get(parent.as_ref()) {
                Some(&p) => cur = p as usize,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Load the raw lines (and, when `parse_values`, the parsed entries) of the
    /// branch `indices` from `path`, in branch order. Non-branch entries are
    /// never parsed.
    ///
    /// # Errors
    /// I/O or JSON failure on a requested line.
    pub fn load_branch(
        &self,
        path: &Path,
        indices: &[u32],
        parse_values: bool,
    ) -> Result<Vec<BranchLine>> {
        let mut file = LineReader::open(path)?;
        let mut out = Vec::with_capacity(indices.len());
        for &i in indices {
            let r = &self.refs[i as usize];
            let Some(raw) = file.read_at(r.offset, r.len)? else {
                anyhow::bail!("{}:{}: entry vanished", path.display(), r.line_no);
            };
            let value =
                if parse_values {
                    Some(serde_json::from_str::<Value>(&raw).with_context(|| {
                        format!("{}:{}: entry parse", path.display(), r.line_no)
                    })?)
                } else {
                    None
                };
            out.push(BranchLine {
                raw,
                kind: r.kind,
                value,
            });
        }
        Ok(out)
    }

    /// Stream the raw lines of `indices` (in the order given), one at a time —
    /// no line survives the callback. Use for passes that consume the branch
    /// without materializing it (seeding, scans).
    ///
    /// # Errors
    /// I/O failure, a vanished line, or the callback's error.
    pub fn for_each_line(
        &self,
        path: &Path,
        indices: &[u32],
        mut f: impl FnMut(&EntryRef, &str) -> Result<()>,
    ) -> Result<()> {
        self.scan_lines(path, indices, |r, raw| {
            f(r, raw).map(|()| std::ops::ControlFlow::Continue(()))
        })
    }

    /// Like [`Self::for_each_line`], but the callback can stop the scan early
    /// (`ControlFlow::Break`): lines after the break are never read.
    ///
    /// # Errors
    /// I/O failure, a vanished line, or the callback's error.
    pub fn scan_lines(
        &self,
        path: &Path,
        indices: &[u32],
        mut f: impl FnMut(&EntryRef, &str) -> Result<std::ops::ControlFlow<()>>,
    ) -> Result<()> {
        let mut file = LineReader::open(path)?;
        for &i in indices {
            let r = &self.refs[i as usize];
            let Some(raw) = file.read_at(r.offset, r.len)? else {
                anyhow::bail!("{}:{}: entry vanished", path.display(), r.line_no);
            };
            if f(r, &raw)?.is_break() {
                break;
            }
        }
        Ok(())
    }
}

/// `Head` from a raw line: the fast path is a borrowed typed parse; lines that
/// are JSON objects with unusual field shapes fall back to a `Value` parse so
/// they still index (with empty links).
fn head_of(bytes: &[u8]) -> Option<Head> {
    if let Ok(head) = serde_json::from_slice::<Head>(bytes) {
        return Some(head);
    }
    let Ok(Value::Object(obj)) = serde_json::from_slice::<Value>(bytes) else {
        return None;
    };
    Some(Head {
        kind: obj.get("type").and_then(Value::as_str).map(str::to_owned),
        id: obj.get("id").and_then(Value::as_str).map(str::to_owned),
        parent_id: obj
            .get("parentId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        timestamp: obj
            .get("timestamp")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

/// Buffered positional reader over a session journal. Branch passes walk
/// entries mostly in file order, so a forward jump inside the buffer is a
/// `seek_relative` (no syscall) and a sequential run is one `read` per buffer
/// fill instead of a `seek`+`read` per line.
struct LineReader {
    inner: BufReader<File>,
    pos: u64,
}

impl LineReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(Self {
            inner: BufReader::with_capacity(256 * 1024, file),
            pos: 0,
        })
    }

    /// Read the `len` bytes at `offset` and decode like the streaming loader
    /// (lossy UTF-8, trailing `\r` stripped).
    fn read_at(&mut self, offset: u64, len: u32) -> Result<Option<String>> {
        if offset >= self.pos {
            let delta = i64::try_from(offset - self.pos).unwrap_or(i64::MAX);
            self.inner.seek_relative(delta)?;
        } else {
            self.inner.seek(SeekFrom::Start(offset))?;
        }
        self.pos = offset;
        let mut buf = vec![0u8; len as usize];
        match self.inner.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Position is unknown after a short read: resync on the next call.
                self.inner.seek(SeekFrom::Start(offset))?;
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
        self.pos = offset + u64::from(len);
        let mut text = match String::from_utf8(buf) {
            Ok(t) => t,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };
        if text.ends_with('\r') {
            text.pop();
        }
        Ok(Some(text))
    }
}

/// Convenience: read one line of `path` at `offset`/`len` (opens the file).
fn read_line(path: &Path, offset: u64, len: u32) -> Result<Option<String>> {
    LineReader::open(path)?.read_at(offset, len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpfile(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("omp-rs-index-{name}-{}", std::process::id()));
        let mut f = File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn index_links_match_load() {
        let body = concat!(
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            "\n",
            r#"{"type":"message","id":"a","parentId":"hdr","timestamp":"t","message":{"role":"user"}}"#,
            "\n",
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"t","message":{"role":"assistant"}}"#,
            "\n",
        );
        let path = tmpfile("links", body);
        let idx = SessionIndex::load(&path).unwrap();
        assert_eq!(idx.refs.len(), 3);
        assert!(!idx.invalid_header, "valid header");
        assert_eq!(idx.refs[0].kind, EntryKind::Header);
        assert_eq!(&*idx.refs[1].id, "a");
        assert_eq!(&*idx.refs[2].parent_id, "a");
        assert_eq!(idx.branch(), vec![1, 2]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn branch_follows_parent_not_file_order() {
        // c branches off a; the file ends on the a→b line, but the leaf (last
        // entry) is b — TS leaves `#leaf` on the last journaled entry.
        let body = concat!(
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            "\n",
            r#"{"type":"message","id":"a","parentId":"hdr","timestamp":"t"}"#,
            "\n",
            r#"{"type":"message","id":"c","parentId":"a","timestamp":"t"}"#,
            "\n",
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"t"}"#,
            "\n",
        );
        let path = tmpfile("branch", body);
        let idx = SessionIndex::load(&path).unwrap();
        assert_eq!(idx.branch(), vec![1, 3]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn branch_stops_on_cycle_and_unknown_parent() {
        let body = concat!(
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            "\n",
            r#"{"type":"message","id":"a","parentId":"b","timestamp":"t"}"#,
            "\n",
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"t"}"#,
            "\n",
        );
        let path = tmpfile("cycle", body);
        let idx = SessionIndex::load(&path).unwrap();
        assert_eq!(idx.branch().len(), 2, "cycle visits each id once");
        std::fs::remove_file(&path).ok();

        let body2 = concat!(
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            "\n",
            r#"{"type":"message","id":"a","parentId":"missing","timestamp":"t"}"#,
            "\n",
        );
        let path2 = tmpfile("orphan", body2);
        let idx2 = SessionIndex::load(&path2).unwrap();
        assert_eq!(idx2.branch(), vec![1]);
        std::fs::remove_file(&path2).ok();
    }

    #[test]
    fn malformed_and_blank_lines_are_skipped() {
        let body = concat!(
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            "\n\nnot json\n",
            r#"{"type":"message","id":"a","parentId":"hdr","timestamp":"t"}"#,
            "\n",
        );
        let path = tmpfile("malformed", body);
        let idx = SessionIndex::load(&path).unwrap();
        assert_eq!(idx.refs.len(), 2);
        assert_eq!(idx.malformed.len(), 1);
        assert_eq!(idx.malformed[0].line_no, 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_branch_parses_only_requested_lines() {
        let big = "x".repeat(4096);
        let msg = format!(
            r#"{{"type":"message","id":"a","parentId":"hdr","timestamp":"t","message":{{"role":"user","content":[{{"type":"text","text":"{big}"}}]}}}}"#
        );
        let body = format!(
            "{}\n{}\n{}\n",
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            msg,
            r#"{"type":"message","id":"b","parentId":"a","timestamp":"t"}"#,
        );
        let path = tmpfile("load-branch", &body);
        let idx = SessionIndex::load(&path).unwrap();
        let branch = idx.branch();
        let lines = idx.load_branch(&path, &branch, false).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().all(|l| l.value.is_none()));
        assert!(lines[0].raw.contains(&big));
        let parsed = idx.load_branch(&path, &branch, true).unwrap();
        assert_eq!(
            parsed[1]
                .value
                .as_ref()
                .and_then(|v| v.get("id"))
                .and_then(Value::as_str),
            Some("b")
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn title_slot_and_invalid_header() {
        let body = concat!(
            r#"{"type":"title","v":1,"title":"hi","updatedAt":"t","pad":""}"#,
            "\n",
            r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#,
            "\n",
            r#"{"type":"message","id":"a","parentId":"hdr","timestamp":"t"}"#,
            "\n",
        );
        let path = tmpfile("slot", body);
        let idx = SessionIndex::load(&path).unwrap();
        assert!(idx.title_slot.is_some());
        assert!(!idx.invalid_header);
        assert_eq!(idx.branch(), vec![2]);
        std::fs::remove_file(&path).ok();

        let bad = tmpfile("bad-header", "{\"type\":\"message\",\"id\":\"a\"}\n");
        let idx = SessionIndex::load(&bad).unwrap();
        assert!(idx.invalid_header);
        std::fs::remove_file(&bad).ok();
    }
}
