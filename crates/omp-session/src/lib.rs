//! omp session files (JSONL, format v3) on top of `pa_types::session::FileEntry`.
//!
//! The entry model is prime-agent's (MIT, derived from the same pi lineage as omp); this crate adds
//! what omp needs around it: typed models for the omp-only entry kinds ([`entries`]), a
//! line-preserving reader, a writer that chains `id`/`parentId` the way omp's session manager does,
//! and a round-trip checker used against real `~/.omp/agent/sessions`.

pub mod entries;
pub mod index;
pub mod scan;
pub mod tree;

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pa_types::session::{AgentMessage, FileEntry};
use serde_json::{json, Map, Value};

pub use entries::{OmpEntry, OmpMessage};
pub use pa_types::session;

pub const SESSION_VERSION: u32 = 3;

/// One line of a session file: the parsed entry plus the raw text it came from.
pub struct Line {
    pub raw: String,
    pub entry: FileEntry,
}

impl Line {
    /// The line in omp's own typed shape, when it is one of the kinds [`OmpEntry`] models.
    #[must_use]
    pub fn omp(&self) -> Option<OmpEntry> {
        OmpEntry::parse(&self.raw)
    }
}

/// A record skipped by the lenient loader (omp's `parseJsonlLenient`): not a JSON object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Malformed {
    /// 1-based line number in the file.
    pub line_no: usize,
    /// The line, lossily decoded.
    pub raw: String,
}

/// A loaded session file, like omp's `SessionLoadResult`.
pub struct Loaded {
    /// Parsed lines in file order, including the title slot line when present.
    pub lines: Vec<Line>,
    /// Skipped records. omp counts these, and rewrites the file when any exist.
    pub malformed: Vec<Malformed>,
    /// The fixed-width title slot, when the first line is one.
    pub title_slot: Option<entries::TitleSlot>,
    /// Non-empty data without a valid leading `session` header (title slot excluded); with no
    /// parsable record at all, true when something malformed was skipped.
    pub invalid_header: bool,
}

/// Load a session file the way omp's loader does: blank lines are ignored, records that are not
/// JSON objects (NUL-padded tails, truncated writes, binary noise) are skipped and reported, and
/// invalid UTF-8 is decoded lossily.
///
/// # Errors
/// I/O failure.
pub fn load(path: &Path) -> Result<Loaded> {
    let buf = scan::FileBuf::read(path)?;
    /// One line's parse result, in file order after assembly.
    enum Item {
        Line(Line),
        Malformed(Malformed),
        Blank,
    }
    let parse = |n: usize, bytes: &[u8]| -> Item {
        let raw = scan::decode_line(bytes);
        if raw.trim().is_empty() {
            return Item::Blank;
        }
        let raw = raw.into_owned();
        match serde_json::from_str::<FileEntry>(&raw) {
            Ok(entry) => Item::Line(Line { raw, entry }),
            Err(_) => Item::Malformed(Malformed { line_no: n, raw }),
        }
    };
    let items: Vec<Item> = if buf.as_slice().len() > scan::PAR_THRESHOLD {
        use rayon::prelude::*;
        let buf = buf.as_slice();
        let spans = scan::line_spans(buf);
        let threads = rayon::current_num_threads().max(1);
        let chunk = (spans.len() / (threads * 8)).clamp(256, 4096).max(1);
        spans
            .par_chunks(chunk)
            .map(|c| {
                c.iter()
                    .map(|&(n, o, l)| parse(n, &buf[o as usize..o as usize + l as usize]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    } else {
        scan::line_spans(buf.as_slice())
            .into_iter()
            .map(|(n, o, l)| parse(n, &buf.as_slice()[o as usize..o as usize + l as usize]))
            .collect()
    };
    let mut lines = Vec::new();
    let mut malformed = Vec::new();
    for item in items {
        match item {
            Item::Line(l) => lines.push(l),
            Item::Malformed(m) => malformed.push(m),
            Item::Blank => {}
        }
    }
    let title_slot = lines.first().and_then(|l| match l.omp() {
        Some(OmpEntry::Title(slot)) if slot.v == 1 => Some(slot),
        _ => None,
    });
    let mut body = lines.iter().skip(usize::from(title_slot.is_some()));
    let invalid_header = match body.next() {
        Some(l) => !matches!(&l.entry, FileEntry::Header { header } if !header.id.is_empty()),
        None => !malformed.is_empty(),
    };
    Ok(Loaded {
        lines,
        malformed,
        title_slot,
        invalid_header,
    })
}

/// [`load`] without the diagnostics: the parsed lines, malformed records skipped.
///
/// # Errors
/// I/O failure.
pub fn read(path: &Path) -> Result<Vec<Line>> {
    Ok(load(path)?.lines)
}

/// One parse per line straight into `Value` (object records only, malformed
/// lines skipped), for consumers that need the entry tree itself (resume/rebuild):
/// [`load`]/`read` parse every line twice (typed entry + caller's `Value`).
///
/// # Errors
/// I/O failure.
pub fn read_values(path: &Path) -> Result<Vec<Value>> {
    let buf = scan::FileBuf::read(path)?;
    let parse = |bytes: &[u8]| -> Option<Value> {
        let raw = scan::decode_line(bytes);
        if raw.trim().is_empty() {
            return None;
        }
        serde_json::from_str::<Value>(&raw)
            .ok()
            .filter(Value::is_object)
    };
    let mut items: Vec<Option<Value>> = if buf.as_slice().len() > scan::PAR_THRESHOLD {
        use rayon::prelude::*;
        let buf = buf.as_slice();
        let spans = scan::line_spans(buf);
        let threads = rayon::current_num_threads().max(1);
        let chunk = (spans.len() / (threads * 8)).clamp(256, 4096).max(1);
        spans
            .par_chunks(chunk)
            .map(|c| {
                c.iter()
                    .map(|&(_, o, l)| parse(&buf[o as usize..o as usize + l as usize]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    } else {
        scan::line_spans(buf.as_slice())
            .into_iter()
            .map(|(_, o, l)| parse(&buf.as_slice()[o as usize..o as usize + l as usize]))
            .collect()
    };
    Ok(items.drain(..).flatten().collect())
}

/// Append-only writer producing files omp itself can open: v3 header first, then entries whose
/// `parentId` is the previous entry's `id` (a linear branch, as omp writes without /tree navigation).
pub struct SessionWriter {
    file: File,
    path: PathBuf,
    last_id: Option<String>,
    /// The append point: the parent the next entry chains to. Advances with every append
    /// (`SessionEntryIndex.insert` sets the leaf to the recorded entry); `/tree` navigation
    /// retargets it without rewriting the file.
    leaf: Option<String>,
    ids: HashSet<String>,
}

impl SessionWriter {
    /// Open `<dir>/<iso-timestamp>_<id>.jsonl` for append (omp's naming scheme and flag `'a'`). A new
    /// or empty file gets the v3 header; an existing one is continued: its ids are loaded (for
    /// collision checks) and the last entry becomes the parent of the next append.
    ///
    /// # Errors
    /// I/O failure creating the directory or opening the file.
    pub fn create(dir: &Path, cwd: &str, session_id: &str, timestamp_iso: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let stamp = timestamp_iso.replace([':', '.'], "-");
        let path = dir.join(format!("{stamp}_{session_id}.jsonl"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        let mut w = Self {
            file,
            path,
            last_id: None,
            leaf: None,
            ids: HashSet::new(),
        };
        if w.file.metadata()?.len() == 0 {
            w.write_value(&json!({"type": "session", "version": SESSION_VERSION, "id": session_id, "timestamp": timestamp_iso, "cwd": cwd}))?;
        } else {
            w.load_existing()?;
        }
        Ok(w)
    }

    /// Write a fresh session header at exactly `path` (a caller-chosen file name,
    /// unlike [`Self::create`]); the file must not exist yet.
    ///
    /// # Errors
    /// The file exists, or I/O failure.
    pub fn create_at(path: &Path, cwd: &str, session_id: &str, timestamp_iso: &str) -> Result<()> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create {}", path.display()))?;
        let header = json!({"type": "session", "version": SESSION_VERSION, "id": session_id, "timestamp": timestamp_iso, "cwd": cwd});
        std::io::Write::write_all(&mut file, format!("{header}\n").as_bytes())?;
        Ok(())
    }

    /// Reopen an existing session file for append (TS `SessionManager.open`: the file and its
    /// header stay untouched, new entries are appended). The append point is `leaf` when given
    /// (the active branch tip, which need not be the last line), else the last entry.
    ///
    /// # Errors
    /// I/O failure, a missing file, or `leaf` is not an entry of the file.
    pub fn open(path: &Path, leaf: Option<&str>) -> Result<Self> {
        anyhow::ensure!(path.is_file(), "session file not found: {}", path.display());
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let mut w = Self {
            file,
            path: path.to_path_buf(),
            last_id: None,
            leaf: None,
            ids: HashSet::new(),
        };
        w.load_existing()?;
        if std::fs::read(path)
            .map(|b| b.last().is_some_and(|&c| c != b'\n'))
            .unwrap_or(false)
        {
            w.file.write_all(b"\n")?;
        }
        if let Some(leaf) = leaf {
            w.branch_to(Some(leaf.to_owned()))?;
        }
        Ok(w)
    }

    /// Record ids of an existing file (skipping lines that are not JSON objects) and its last entry.
    fn load_existing(&mut self) -> Result<()> {
        let buf = scan::FileBuf::read(&self.path)?;
        scan::for_each_line(buf.as_slice(), |_n, bytes| {
            // Borrowed scan: only `type`/`id` are needed, no Value tree.
            let Some(head) = scan::head_of(bytes) else {
                return Ok(std::ops::ControlFlow::Continue(()));
            };
            let Some(id) = head.id else {
                return Ok(std::ops::ControlFlow::Continue(()));
            };
            self.ids.insert(id.to_string());
            if head.kind.as_deref() != Some("session") {
                self.last_id = Some(id.to_string());
                self.leaf = Some(id.to_string());
            }
            Ok(std::ops::ControlFlow::Continue(()))
        })?;
        Ok(())
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a `message` entry; returns the new entry id.
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_message(
        &mut self,
        message: &AgentMessage,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_omp_message(OmpMessage::from_agent(message)?, timestamp_iso)
    }

    /// Append a `message` entry carrying any omp message shape (`developer`, `pythonExecution`, or a
    /// message with extra `usage` counters).
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_omp_message(
        &mut self,
        message: OmpMessage,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::Message(Box::new(entries::MessageEntry {
                id,
                parent_id,
                timestamp,
                message,
                rest: Map::new(),
            }))
        })
    }

    /// Append `model_change` (`model` is `provider/modelId`).
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_model_change(
        &mut self,
        model: &str,
        role: Option<&str>,
        resolved_model_is_fallback: bool,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::ModelChange(entries::ModelChange {
                id,
                parent_id,
                timestamp,
                model: model.to_owned(),
                role: role.map(str::to_owned),
                resolved_model_is_fallback: Some(resolved_model_is_fallback),
                rest: Map::new(),
            })
        })
    }

    /// Append `thinking_level_change`; absent levels are written as explicit nulls, as omp does.
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_thinking_level_change(
        &mut self,
        thinking_level: Option<&str>,
        configured: Option<&str>,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::ThinkingLevelChange(entries::ThinkingLevelChange {
                id,
                parent_id,
                timestamp,
                thinking_level: Some(thinking_level.map(str::to_owned)),
                configured: Some(configured.map(str::to_owned)),
                rest: Map::new(),
            })
        })
    }

    /// Append `mode_change` (`mode` is `"none"` when leaving a mode).
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_mode_change(
        &mut self,
        mode: &str,
        data: Option<Value>,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::ModeChange(entries::ModeChange {
                id,
                parent_id,
                timestamp,
                mode: mode.to_owned(),
                data,
                rest: Map::new(),
            })
        })
    }

    /// Append `title_change` (`source` is `"auto"` or `"user"`).
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_title_change(
        &mut self,
        title: &str,
        source: &str,
        previous_title: Option<&str>,
        trigger: Option<&str>,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::TitleChange(entries::TitleChange {
                id,
                parent_id,
                timestamp,
                title: title.to_owned(),
                previous_title: previous_title.filter(|p| !p.is_empty()).map(str::to_owned),
                source: source.to_owned(),
                trigger: trigger.filter(|t| !t.is_empty()).map(str::to_owned),
                rest: Map::new(),
            })
        })
    }

    /// Patch the fixed-width 256-byte title slot at line 1 (`FileSessionStorage.updateSessionTitle`).
    ///
    /// When the file has no title slot yet (sessions created before titling), one is
    /// prepended by an atomic rewrite, so line 1 is always the slot afterwards — the same
    /// on-disk shape omp's `SessionManager` produces.
    ///
    /// # Errors
    /// I/O or serialization failure.
    pub fn update_title_slot(
        &mut self,
        title: &str,
        source: Option<&str>,
        timestamp_iso: &str,
    ) -> Result<()> {
        let slot = entries::serialize_title_slot(title, source, timestamp_iso)?;
        let slot_bytes = slot.as_bytes();

        let first_line_is_slot = {
            let mut f = BufReader::new(File::open(&self.path)?);
            let mut line = Vec::new();
            // A slot line is exactly TITLE_SLOT_BYTES including the newline.
            (f.read_until(b'\n', &mut line)? == entries::TITLE_SLOT_BYTES)
                && serde_json::from_slice::<Value>(&line)
                    .ok()
                    .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_owned))
                    .as_deref()
                    == Some("title")
        };

        if first_line_is_slot {
            let f = OpenOptions::new().write(true).open(&self.path)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                f.write_all_at(slot_bytes, 0)?;
            }
            #[cfg(not(unix))]
            {
                // Windows: fixed-offset write via seek+write on a fresh handle
                // opened without append, so the pointer position is irrelevant.
                use std::io::{Seek, SeekFrom};
                let mut f = &f;
                f.seek(SeekFrom::Start(0))?;
                f.write_all(slot_bytes)?;
            }
            f.sync_data()?;
            return Ok(());
        }

        // No slot yet: prepend one, atomically, and re-open the append handle.
        let body = std::fs::read_to_string(&self.path)?;
        let tmp = self.path.with_extension("jsonl.title-tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(slot_bytes)?;
            f.write_all(body.as_bytes())?;
            f.sync_data()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        Ok(())
    }

    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_ttsr_injection(
        &mut self,
        injected_rules: &[String],
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::TtsrInjection(entries::TtsrInjection {
                id,
                parent_id,
                timestamp,
                injected_rules: injected_rules.to_vec(),
                rest: Map::new(),
            })
        })
    }

    /// Append `credential_pin`.
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_credential_pin(
        &mut self,
        provider: &str,
        hash: &str,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::CredentialPin(entries::CredentialPin {
                id,
                parent_id,
                timestamp,
                provider: provider.to_owned(),
                hash: hash.to_owned(),
                rest: Map::new(),
            })
        })
    }

    /// Append `reset_boundary` (the `/clear` marker).
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_reset_boundary(&mut self, timestamp_iso: &str) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            OmpEntry::ResetBoundary(entries::ResetBoundary {
                id,
                parent_id,
                timestamp,
                rest: Map::new(),
            })
        })
    }

    /// Append `model_usage` for a call outside the transcript. `entry.id`/`parent_id`/`timestamp` are
    /// overwritten with generated values.
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_model_usage(
        &mut self,
        mut entry: entries::ModelUsage,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            entry.id = id;
            entry.parent_id = parent_id;
            entry.timestamp = timestamp;
            OmpEntry::ModelUsage(Box::new(entry))
        })
    }

    /// Append `session_init` (subagent initial context). `entry.id`/`parent_id`/`timestamp` are
    /// overwritten with generated values.
    ///
    /// # Errors
    /// Serialization or I/O failure.
    pub fn append_session_init(
        &mut self,
        mut entry: entries::SessionInit,
        timestamp_iso: &str,
    ) -> Result<String> {
        self.append_typed(timestamp_iso, |id, parent_id, timestamp| {
            entry.id = id;
            entry.parent_id = parent_id;
            entry.timestamp = timestamp;
            OmpEntry::SessionInit(Box::new(entry))
        })
    }

    /// Append any entry kind (`compaction`, `custom_message`, `model_change`, ...). `payload` holds
    /// the kind-specific fields; `type`, `id`, `parentId` and `timestamp` are generated here and win
    /// over same-named payload keys (as in omp, where the fresh fields are spread first).
    ///
    /// # Errors
    /// `payload` is not a JSON object, an omp-modeled kind whose payload does not match its schema,
    /// or I/O failure.
    pub fn append_entry(
        &mut self,
        kind: &str,
        payload: Value,
        timestamp_iso: &str,
    ) -> Result<String> {
        let Value::Object(fields) = payload else {
            anyhow::bail!("entry payload must be an object")
        };
        let id = self.fresh_id();
        let mut obj = Map::new();
        obj.insert("type".into(), Value::String(kind.into()));
        obj.insert("id".into(), Value::String(id.clone()));
        obj.insert(
            "parentId".into(),
            self.leaf.clone().map_or(Value::Null, Value::String),
        );
        obj.insert("timestamp".into(), Value::String(timestamp_iso.into()));
        obj.extend(
            fields
                .into_iter()
                .filter(|(k, _)| !matches!(k.as_str(), "type" | "id" | "parentId" | "timestamp")),
        );
        let obj = Value::Object(obj);
        // Through a typed entry (never a Value detour) so field order matches omp's writer: omp
        // kinds via `OmpEntry`; the rest via `FileEntry` (unmodeled kinds keep construction order).
        // The typed parse borrows (`Deserialize for &Value`): seeding a resumed session clones
        // nothing here — a consuming `from_value` would force a deep clone of every payload.
        let line = match <OmpEntry as serde::Deserialize>::deserialize(&obj) {
            Ok(entry) => entries::to_js_string(&entry)?,
            Err(e) if entries::STRICT_KINDS.contains(&kind) => {
                return Err(anyhow::Error::from(e)
                    .context(format!("{kind} entry payload does not match its schema")));
            }
            Err(_) => {
                let entry: FileEntry = serde_json::from_value(obj)
                    .with_context(|| format!("{kind} entry payload does not match its schema"))?;
                entries::to_js_string(&entry)?
            }
        };
        self.push_line(&line, &id)?;
        Ok(id)
    }

    /// Seed from a raw entry line of another session file (resume): the payload
    /// bytes stay verbatim — only the top-level `id` (fresh), `parentId` (chained
    /// to the current leaf) and, through `remap`, `firstKeptEntryId` /
    /// `providerReplayThroughEntryId` are rewritten. This is what makes resuming
    /// a large session cheap: no `Value` DOM is ever built for the seeded branch.
    ///
    /// # Errors
    /// `raw` is not a JSON object line whose top-level fields can be rewritten.
    pub fn append_seeded_line(
        &mut self,
        raw: &str,
        remap: &std::collections::HashMap<String, String>,
    ) -> Result<String> {
        let members = top_level_string_members(raw)
            .ok_or_else(|| anyhow::anyhow!("entry line is not a rewriteable JSON object"))?;
        let new_id = self.fresh_id();
        let parent = self
            .leaf
            .clone()
            .map_or_else(|| "null".to_owned(), |l| format!("\"{l}\""));
        let mut replacements: Vec<(usize, usize, String)> = Vec::with_capacity(3);
        for (key, vs, ve, escaped) in members {
            if escaped {
                continue; // an escaped id field is not worth guessing: caller falls back to the DOM path
            }
            match key.as_str() {
                "id" => replacements.push((vs, ve, format!("\"{new_id}\""))),
                "parentId" => replacements.push((vs, ve, parent.clone())),
                "firstKeptEntryId" | "providerReplayThroughEntryId" => {
                    if let Some(mapped) = remap.get(&raw[vs + 1..ve - 1]) {
                        replacements.push((vs, ve, format!("\"{mapped}\"")));
                    }
                }
                _ => {}
            }
        }
        replacements.sort_by_key(|(start, _, _)| *start);
        let mut line = String::with_capacity(raw.len() + replacements.len() * 8);
        let mut pos = 0;
        for (vs, ve, text) in &replacements {
            line.push_str(&raw[pos..*vs]);
            line.push_str(text);
            pos = *ve;
        }
        line.push_str(&raw[pos..]);
        self.push_line(&line, &new_id)?;
        Ok(new_id)
    }

    /// Build an entry with a fresh id chained to the previous one, write it, advance the chain.
    fn append_typed(
        &mut self,
        timestamp_iso: &str,
        build: impl FnOnce(String, Option<String>, String) -> OmpEntry,
    ) -> Result<String> {
        let id = self.fresh_id();
        let entry = build(id.clone(), self.leaf.clone(), timestamp_iso.to_owned());
        let line = entry.to_line()?;
        self.push_line(&line, &id)?;
        Ok(id)
    }

    fn fresh_id(&self) -> String {
        generate_id(|id| self.ids.contains(id), rand_u32)
    }

    fn push_line(&mut self, line: &str, id: &str) -> Result<()> {
        let mut buf = String::with_capacity(line.len() + 1);
        buf.push_str(line);
        buf.push('\n');
        self.file.write_all(buf.as_bytes())?;
        self.ids.insert(id.to_owned());
        self.last_id = Some(id.to_owned());
        self.leaf = Some(id.to_owned());
        Ok(())
    }

    /// Move the append point to an earlier entry (omp `SessionManager.branch`, the `/tree`
    /// rewind): the next appended entry chains to `id` instead of the tail; `None` makes the
    /// next entry a new root (`resetLeaf`). Nothing already written is modified — like omp,
    /// the move becomes durable only once an entry lands on the new position.
    ///
    /// # Errors
    /// `id` is not an entry of this session file.
    pub fn branch_to(&mut self, id: Option<String>) -> Result<()> {
        if let Some(id) = &id {
            anyhow::ensure!(self.ids.contains(id), "Entry {id} not found");
        }
        self.leaf = id;
        Ok(())
    }

    /// Append a `branch_summary` entry parented at `branch_from` (omp `branchWithSummary`) and
    /// land the append point on it, so the journal durably marks the abandoned branch.
    ///
    /// # Errors
    /// `branch_from` is not an entry of this session file, or the write failed.
    pub fn append_branch_summary(
        &mut self,
        branch_from: Option<&str>,
        summary: &str,
        details: Option<Value>,
        from_extension: Option<bool>,
        timestamp_iso: &str,
    ) -> Result<String> {
        if let Some(id) = &branch_from {
            anyhow::ensure!(self.ids.contains(*id), "Entry {id} not found");
        }
        self.branch_to(branch_from.map(str::to_owned))?;
        let id = self.fresh_id();
        let entry = tree::branch_summary_entry(
            &id,
            branch_from,
            timestamp_iso,
            summary,
            details,
            from_extension,
        );
        let line = entries::to_js_string(&entry)?;
        self.push_line(&line, &id)?;
        Ok(id)
    }

    fn write_value(&mut self, v: &Value) -> Result<()> {
        let mut line = serde_json::to_string(v)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        if let Some(id) = v.get("id").and_then(Value::as_str) {
            self.ids.insert(id.to_owned());
        }
        Ok(())
    }
}

/// omp entry ids are 8 lowercase hex chars (`crypto.randomUUID().slice(-8)`), collision-checked
/// against `has` up to 100 times, then a longer 16-hex id (omp falls back to a full snowflake id).
fn generate_id(has: impl Fn(&str) -> bool, mut next: impl FnMut() -> u32) -> String {
    for _ in 0..100 {
        let id = format!("{:08x}", next());
        if !has(&id) {
            return id;
        }
    }
    format!("{:08x}{:08x}", next(), next())
}

/// Scan one JSON object line for its top-level string members: returns
/// `(key, value_start, value_end, value_has_escapes)` byte spans (`value_*`
/// include the quotes). Returns `None` when the line is not exactly one JSON
/// object. Used by [`SessionWriter::append_seeded_line`] to rewrite id links
/// in place without building a `Value`.
fn top_level_string_members(line: &str) -> Option<Vec<(String, usize, usize, bool)>> {
    let b = line.as_bytes();
    let mut i = 0usize;
    let skip_ws = |i: &mut usize| {
        while b.get(*i).is_some_and(u8::is_ascii_whitespace) {
            *i += 1;
        }
    };
    skip_ws(&mut i);
    if b.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    let mut out = Vec::new();
    loop {
        skip_ws(&mut i);
        match b.get(i) {
            Some(b'}') => {
                i += 1;
                skip_ws(&mut i);
                return (i == b.len()).then_some(out);
            }
            Some(b'"') => {
                let (ks, ke) = scan_json_string(b, i)?;
                let key = unescape(&line[ks..ke - 1]);
                i = ke;
                skip_ws(&mut i);
                if b.get(i) != Some(&b':') {
                    return None;
                }
                i += 1;
                skip_ws(&mut i);
                match b.get(i) {
                    Some(b'"') => {
                        let (vs, ve) = scan_json_string(b, i)?;
                        let escaped = line[vs..ve - 1].contains('\\');
                        out.push((key, vs - 1, ve, escaped));
                        i = ve;
                    }
                    Some(b'{' | b'[') => {
                        i = skip_json_value(b, i)?;
                    }
                    Some(_) => {
                        // number / true / false / null: up to the next , } ] at any depth
                        let start = i;
                        while let Some(&c) = b.get(i) {
                            if matches!(c, b',' | b'}' | b']') {
                                break;
                            }
                            i += 1;
                        }
                        if i == start {
                            return None;
                        }
                    }
                    None => return None,
                }
                skip_ws(&mut i);
                match b.get(i) {
                    Some(b',') => i += 1,
                    Some(b'}') => {}
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

/// Byte span of a JSON string starting at `b[start] == b'"'`: returns
/// `(content_start, end_inclusive_past_quote)`.
fn scan_json_string(b: &[u8], start: usize) -> Option<(usize, usize)> {
    if b.get(start) != Some(&b'"') {
        return None;
    }
    let mut j = start + 1;
    while let Some(&c) = b.get(j) {
        match c {
            b'\\' => j += 2,
            b'"' => return Some((start + 1, j + 1)),
            _ => j += 1,
        }
    }
    None
}

/// Position past the balanced JSON value starting at `b[start]` (object or array).
fn skip_json_value(b: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = start;
    while let Some(&c) = b.get(i) {
        match c {
            b'"' => {
                let (_, end) = scan_json_string(b, i)?;
                i = end;
                continue;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Unescape a JSON string body (the content between quotes). Unknown escapes
/// keep their character, matching `serde_json`'s lossy handling.
fn unescape(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(
                    u32::from_str_radix(&hex, 16)
                        .ok()
                        .and_then(char::from_u32)
                        .unwrap_or('\u{fffd}'),
                );
            }
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

fn rand_u32() -> u32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
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

/// Re-serialize a line through its typed model: omp's own ([`OmpEntry`]) when it has one, otherwise
/// prime-agent's `FileEntry`.
#[must_use]
pub fn reserialize(line: &Line) -> Option<String> {
    match line.omp() {
        Some(omp) => omp.to_line().ok(),
        None => entries::to_js_string(&line.entry).ok(),
    }
}

#[must_use]
pub fn round_trip(line: &Line) -> RoundTrip {
    let Some(back) = reserialize(line) else {
        return RoundTrip::Changed;
    };
    if back == line.raw {
        return RoundTrip::Exact;
    }
    if line.omp().is_none() && matches!(line.entry, FileEntry::Unknown { .. }) {
        return RoundTrip::Unknown;
    }
    let a: Option<Value> = serde_json::from_str(&line.raw).ok();
    let b: Option<Value> = serde_json::from_str(&back).ok();
    if a.is_some() && a == b {
        RoundTrip::SameValue
    } else {
        RoundTrip::Changed
    }
}

/// For an `Unknown` line whose `type` IS a modeled kind: why the typed parse failed.
#[must_use]
pub fn unknown_reason(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    let kind = v.get("type")?.as_str()?.to_owned();
    if kind == "message" {
        let msg = v.get("message")?.clone();
        return serde_json::from_value::<OmpMessage>(msg)
            .err()
            .map(|e| format!("message: {e}"));
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
    use std::cell::Cell;

    const T0: &str = "2026-10-11T00:00:00.000Z";
    const T1: &str = "2026-10-11T00:00:01.000Z";

    fn tmp() -> PathBuf {
        std::env::temp_dir().join(format!("omp-session-test-{}", rand_u32()))
    }

    fn writer(dir: &Path) -> SessionWriter {
        SessionWriter::create(dir, "/tmp", "01a00000-0000-7000-8000-000000000000", T0).unwrap()
    }

    fn raws(w: &SessionWriter) -> Vec<String> {
        std::fs::read_to_string(w.path())
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn line(raw: &str) -> Line {
        Line {
            raw: raw.to_owned(),
            entry: serde_json::from_str(raw).unwrap(),
        }
    }

    #[test]
    fn writer_output_reads_back_and_chains_parents() {
        let dir = tmp();
        let mut w = writer(&dir);
        let msg: AgentMessage = serde_json::from_value(
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 1}),
        )
        .unwrap();
        let first = w.append_message(&msg, T1).unwrap();
        w.append_entry(
            "custom_message",
            json!({"customType": "note", "content": "x", "display": false}),
            "2026-10-11T00:00:02.000Z",
        )
        .unwrap();
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

    #[test]
    fn typed_helpers_write_omp_field_order() {
        let dir = tmp();
        let mut w = writer(&dir);
        let a = w
            .append_model_change("zai/glm-5.3-flash", None, false, T1)
            .unwrap();
        w.append_thinking_level_change(None, None, T1).unwrap();
        w.append_mode_change("plan", Some(json!({"planFilePath": "/p.md"})), T1)
            .unwrap();
        w.append_title_change("Name", "user", Some("Old"), None, T1)
            .unwrap();
        w.append_ttsr_injection(&["ts-no-any".to_owned()], T1)
            .unwrap();
        w.append_credential_pin("anthropic", "ab12", T1).unwrap();
        w.append_reset_boundary(T1).unwrap();
        let got = raws(&w);
        let id = |i: usize| {
            serde_json::from_str::<Value>(&got[i]).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(
            got[1],
            format!(
                r#"{{"type":"model_change","id":"{a}","parentId":null,"timestamp":"{T1}","model":"zai/glm-5.3-flash","resolvedModelIsFallback":false}}"#
            )
        );
        assert_eq!(
            got[2],
            format!(
                r#"{{"type":"thinking_level_change","id":"{}","parentId":"{a}","timestamp":"{T1}","thinkingLevel":null,"configured":null}}"#,
                id(2)
            )
        );
        assert!(
            got[3].ends_with(r#""mode":"plan","data":{"planFilePath":"/p.md"}}"#),
            "{}",
            got[3]
        );
        assert!(
            got[4].ends_with(r#""title":"Name","previousTitle":"Old","source":"user"}"#),
            "{}",
            got[4]
        );
        assert!(
            got[5].ends_with(r#""injectedRules":["ts-no-any"]}"#),
            "{}",
            got[5]
        );
        assert!(
            got[6].ends_with(r#""provider":"anthropic","hash":"ab12"}"#),
            "{}",
            got[6]
        );
        assert!(
            got[7].ends_with(&format!(r#""timestamp":"{T1}"}}"#)),
            "{}",
            got[7]
        );
        for l in read(w.path()).unwrap() {
            assert!(
                l.omp().is_some() || matches!(l.entry, FileEntry::Header { .. }),
                "{}",
                l.raw
            );
            assert_eq!(round_trip(&l), RoundTrip::Exact, "{}", l.raw);
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn omp_shapes_round_trip_without_loss() {
        let cases = [
            r#"{"type":"title","v":1,"title":"x","source":"auto","updatedAt":"2026-10-06T16:15:06.242Z","pad":"   "}"#,
            r#"{"type":"model_change","id":"f30ba933","parentId":null,"timestamp":"t","model":"zai/glm-5.3-flash","role":"smol","resolvedModelIsFallback":true}"#,
            r#"{"type":"thinking_level_change","id":"a","parentId":"b","timestamp":"t","thinkingLevel":null,"configured":"high"}"#,
            r#"{"type":"model_usage","id":"a","parentId":"b","timestamp":"t","purpose":"p","role":"judge","api":"x","provider":"y","model":"m","usage":{"input":1,"output":2,"cacheRead":0,"cacheWrite":0,"totalTokens":3,"reasoningTokens":1,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"error","errorMessage":"boom","futureField":[1]}"#,
            r#"{"type":"session_init","id":"a","parentId":"b","timestamp":"t","systemPrompt":"s","task":"t","tools":["bash"],"agent":"scout","outputSchema":null,"isolated":false}"#,
            r#"{"type":"mode_change","id":"a","parentId":"b","timestamp":"t","mode":"none"}"#,
            r#"{"type":"message","id":"a","parentId":"b","timestamp":"t","message":{"role":"developer","content":[{"type":"text","text":"hi"}],"attribution":"agent","timestamp":1791303322752,"synthetic":true}}"#,
            r#"{"type":"message","id":"a","parentId":"b","timestamp":"t","message":{"role":"pythonExecution","code":"1","output":"1","exitCode":0,"cancelled":false,"truncated":false,"timestamp":5,"excludeFromContext":true}}"#,
            r#"{"type":"message","id":"a","parentId":"b","timestamp":"t","message":{"role":"assistant","content":[],"api":"x","provider":"y","model":"m","usage":{"input":7,"output":3,"cacheRead":0,"cacheWrite":16,"totalTokens":26,"reasoningTokens":354,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0},"cttl":{"ephemeral1h":16587}},"stopReason":"stop","timestamp":2}}"#,
            r#"{"type":"custom","customType":"x","data":{"a":3.75e-7,"b":0.00000375},"id":"a","parentId":"b","timestamp":"t"}"#,
            r#"{"type":"custom_message","customType":"x","content":"c","display":true,"details":{"k":1},"attribution":"agent","id":"a","parentId":null,"timestamp":"t"}"#,
            r#"{"type":"compaction","id":"a","parentId":"b","timestamp":"t","summary":"s","shortSummary":"ss","firstKeptEntryId":"f","tokensBefore":10,"tokensAfter":5,"method":"m","details":{"x":1},"fromExtension":false,"preserveData":{"y":2}}"#,
            r#"{"type":"branch_summary","id":"a","parentId":"b","timestamp":"t","fromId":"f","summary":"s","fromExtension":true}"#,
            r#"{"type":"service_tier_change","id":"a","parentId":"b","timestamp":"t","serviceTier":null}"#,
            r#"{"type":"message","id":"a","parentId":"b","timestamp":"t","message":{"role":"assistant","content":[{"type":"text","text":"x"}],"api":"x","provider":"y","model":"m","usage":{"input":1,"output":1,"cacheRead":0,"cacheWrite":0,"totalTokens":2,"cost":{"input":0,"output":0.00000375,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":2,"responseId":"r","duration":5}}"#,
            r#"{"type":"message","id":"a","parentId":"b","timestamp":"t","message":{"role":"toolResult","toolCallId":"c","toolName":"bash","content":[{"type":"text","text":"x"}],"isError":false,"useless":true,"timestamp":2}}"#,
        ];
        for raw in cases {
            let l = line(raw);
            assert!(l.omp().is_some(), "not modeled: {raw}");
            assert_eq!(round_trip(&l), RoundTrip::Exact, "{raw}");
        }
    }

    #[test]
    fn update_title_slot_prepends_then_patches_in_place() {
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        // Fresh omp-rs file: header at line 1, no slot yet -> prepend path.
        let mut w = SessionWriter::create(&dir, "/", "s1", "2026-10-11T00:00:00.000Z").unwrap();
        w.append_title_change("First", "auto", None, None, "2026-10-11T00:00:01.000Z")
            .unwrap();
        w.update_title_slot("First", Some("auto"), "2026-10-11T00:00:01.000Z")
            .unwrap();
        let loaded = load(&w.path).unwrap();
        let slot = loaded.title_slot.expect("slot prepended");
        assert_eq!(slot.title, "First");
        assert_eq!(slot.source.as_deref(), Some("auto"));
        // Line 1 is exactly 256 bytes; the header follows, then the entry.
        let raw = std::fs::read_to_string(&w.path).unwrap();
        let mut lines = raw.lines();
        assert_eq!(lines.next().unwrap().len() + 1, entries::TITLE_SLOT_BYTES);
        assert!(lines.next().unwrap().contains("\"type\":\"session\""));
        // Appending after the rewrite still chains to the last entry.
        w.append_title_change(
            "Second",
            "user",
            Some("First"),
            None,
            "2026-10-11T00:00:02.000Z",
        )
        .unwrap();
        // Second update patches the slot in place without growing the file.
        let before = std::fs::metadata(&w.path).unwrap().len();
        w.update_title_slot("Second", Some("user"), "2026-10-11T00:00:02.000Z")
            .unwrap();
        let after = std::fs::metadata(&w.path).unwrap().len();
        assert_eq!(before, after, "in-place patch keeps the file size");
        let loaded = load(&w.path).unwrap();
        assert_eq!(
            loaded.title_slot.as_ref().map(|s| s.title.as_str()),
            Some("Second")
        );
        assert_eq!(
            loaded.title_slot.as_ref().and_then(|s| s.source.as_deref()),
            Some("user")
        );
        let raw = std::fs::read_to_string(&w.path).unwrap();
        assert!(
            raw.lines()
                .nth(2)
                .unwrap()
                .ends_with("\"title\":\"First\",\"source\":\"auto\"}"),
            "{raw}"
        );
        assert!(
            raw.lines()
                .nth(3)
                .unwrap()
                .ends_with("\"title\":\"Second\",\"previousTitle\":\"First\",\"source\":\"user\"}"),
            "{raw}"
        );
    }

    #[test]
    fn load_skips_malformed_records_like_omp() {
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.jsonl");
        let slot =
            entries::serialize_title_slot("T", Some("auto"), "2026-10-11T00:00:00.000Z").unwrap();
        let header = r#"{"type":"session","version":3,"id":"s1","timestamp":"t","cwd":"/"}"#;
        let ok =
            r#"{"type":"model_change","id":"a","parentId":null,"timestamp":"t","model":"zai/glm"}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(slot.as_bytes());
        bytes.extend_from_slice(format!("{header}\n\n").as_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        bytes.extend_from_slice(format!("\n{ok}\n   \n{{\"type\":\"mess\n[1,2]\n").as_bytes());
        bytes.extend_from_slice(&[0xff, 0xfe]);
        bytes.extend_from_slice(format!(" broken\n{ok}").as_bytes());
        std::fs::write(&path, bytes).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.lines.len(), 4, "slot + header + 2 valid");
        assert_eq!(
            loaded
                .malformed
                .iter()
                .map(|m| m.line_no)
                .collect::<Vec<_>>(),
            vec![4, 7, 8, 9]
        );
        assert_eq!(
            loaded.title_slot.as_ref().map(|s| s.title.as_str()),
            Some("T")
        );
        assert!(!loaded.invalid_header);
        assert_eq!(read(&path).unwrap().len(), 4);
        std::fs::write(&path, "garbage\n").unwrap();
        let bad = load(&path).unwrap();
        assert!(bad.lines.is_empty() && bad.invalid_header);
        std::fs::write(&path, format!("{ok}\n")).unwrap();
        assert!(
            load(&path).unwrap().invalid_header,
            "first record is not a session header"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_agent_writes_omp_key_order() {
        let msg: AgentMessage = serde_json::from_value(
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 1}),
        )
        .unwrap();
        let line = OmpEntry::Message(Box::new(entries::MessageEntry {
            id: "a".into(),
            parent_id: None,
            timestamp: "t".into(),
            message: OmpMessage::from_agent(&msg).unwrap(),
            rest: Map::new(),
        }))
        .to_line()
        .unwrap();
        assert_eq!(
            line,
            r#"{"type":"message","id":"a","parentId":null,"timestamp":"t","message":{"role":"user","content":[{"type":"text","text":"hi"}],"timestamp":1}}"#
        );
        assert_eq!(OmpEntry::parse(&line).unwrap().to_line().unwrap(), line);
        let Some(OmpEntry::Message(m)) = OmpEntry::parse(&line) else {
            panic!("not a message")
        };
        assert_eq!(m.message.to_agent(), Some(msg));
    }

    #[test]
    fn title_slot_is_256_bytes() {
        let s =
            entries::serialize_title_slot("Заголовок", Some("auto"), "2026-10-09T22:09:38.413Z")
                .unwrap();
        assert_eq!(s.len(), entries::TITLE_SLOT_BYTES);
        assert!(s.ends_with('\n'));
        let long =
            entries::serialize_title_slot(&"я".repeat(500), None, "2026-10-09T22:09:38.413Z")
                .unwrap();
        assert_eq!(long.len(), entries::TITLE_SLOT_BYTES);
        assert!(OmpEntry::parse(long.trim_end_matches('\n')).is_some());
    }

    #[test]
    fn strict_kind_with_bad_payload_is_an_error() {
        let dir = tmp();
        let mut w = writer(&dir);
        assert!(w
            .append_entry("model_change", json!({"provider": "zai"}), T1)
            .is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn append_entry_generated_fields_win_over_payload() {
        let dir = tmp();
        let mut w = writer(&dir);
        let first = w
            .append_entry("mode_change", json!({"mode": "plan"}), T1)
            .unwrap();
        let id = w
            .append_entry("mode_change", json!({"type": "message", "id": "evil", "parentId": "evil", "timestamp": "evil", "mode": "none"}), "2026-10-11T00:00:02.000Z")
            .unwrap();
        let v: Value = serde_json::from_str(raws(&w).last().unwrap()).unwrap();
        assert_eq!(v["type"], "mode_change");
        assert_eq!(v["id"], json!(id));
        assert_ne!(id, "evil");
        assert_eq!(v["parentId"], json!(first));
        assert_eq!(v["timestamp"], "2026-10-11T00:00:02.000Z");
        assert_eq!(v["mode"], "none");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn append_seeded_line_rewrites_links_and_keeps_payload_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "omp-rs-seed-{}-{:08x}",
            std::process::id(),
            rand_u32()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut w =
            SessionWriter::create(&dir, "/tmp", "sess", "2026-10-12T00-00-00-000Z").unwrap();
        let first = w
            .append_seeded_line(
                r#"{"type":"message","id":"old1","parentId":"hdr","timestamp":"t1","message":{"role":"user","content":[{"type":"text","text":"kept \u00e9 verbatim"}]}}"#,
                &Default::default(),
            )
            .unwrap();
        let mut remap = std::collections::HashMap::new();
        remap.insert("old1".to_owned(), first.clone());
        remap.insert("nope".to_owned(), "x".to_owned());
        let second = w
            .append_seeded_line(
                r#"{"type":"compaction","id":"old2","parentId":"old1","timestamp":"t2","summary":"s","firstKeptEntryId":"old1","providerReplayThroughEntryId":"nope","preserveData":{"nested":{"id":"not-top-level"}}}"#,
                &remap,
            )
            .unwrap();
        let path = w.path().to_path_buf();
        drop(w);
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            r#"{"type":"session","version":3,"id":"sess","timestamp":"2026-10-12T00-00-00-000Z","cwd":"/tmp"}"#
        );
        let a: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(a["id"], first);
        assert_eq!(
            a["parentId"],
            Value::Null,
            "first seeded entry is a root, like the old DOM seeder"
        );
        assert_eq!(a["timestamp"], "t1");
        assert_eq!(a["message"]["content"][0]["text"], "kept é verbatim");
        let b: Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(b["id"], second);
        assert_eq!(b["parentId"], first);
        assert_eq!(
            b["firstKeptEntryId"], first,
            "remapped through the copy map"
        );
        assert_eq!(
            b["providerReplayThroughEntryId"], "x",
            "mapped links follow the copy map"
        );
        assert_eq!(
            b["preserveData"]["nested"]["id"], "not-top-level",
            "nested ids untouched"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn top_level_string_members_scans_nested_and_escaped_shapes() {
        let m = top_level_string_members(
            r#"{"type":"custom","id":"a","data":{"k":"v"},"list":[{"id":"inner"}],"note":"q\"x","n":42,"b":true,"z":null}"#,
        )
        .expect("scans");
        let by_key: std::collections::HashMap<_, _> = m
            .iter()
            .map(|(k, s, e, _)| (k.as_str(), (*s, *e)))
            .collect();
        assert_eq!(by_key["id"], (22, 25));
        assert_eq!(by_key["note"], (74, 80));
        assert!(top_level_string_members("[1,2]").is_none());
        assert!(top_level_string_members("{\"a\":1} trailing").is_none());
        assert!(top_level_string_members("").is_none());
    }

    #[test]
    fn generate_id_retries_on_collision_then_falls_back_to_longer_id() {
        let calls = Cell::new(0u32);
        // First two draws collide, the third is free.
        let id = generate_id(
            |id| id == "00000001" || id == "00000002",
            || {
                calls.set(calls.get() + 1);
                calls.get()
            },
        );
        assert_eq!(id, "00000003");
        // Every 8-hex id is taken: 100 draws, then a 16-hex fallback.
        let draws = Cell::new(0u32);
        let id = generate_id(
            |id| id.len() == 8,
            || {
                draws.set(draws.get() + 1);
                7
            },
        );
        assert_eq!(id.len(), 16);
        assert_eq!(draws.get(), 102);
    }

    #[test]
    fn writer_never_reuses_an_existing_id() {
        let dir = tmp();
        let mut w = writer(&dir);
        let mut seen = HashSet::new();
        for _ in 0..200 {
            assert!(seen.insert(w.append_reset_boundary(T1).unwrap()));
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn create_continues_an_existing_file_like_open_with_flag_a() {
        let dir = tmp();
        let mut w = writer(&dir);
        let last = w.append_reset_boundary(T1).unwrap();
        let before = raws(&w);
        let mut again = writer(&dir);
        assert_eq!(again.path(), w.path());
        assert_eq!(raws(&again), before, "header must not be rewritten");
        let next = again
            .append_reset_boundary("2026-10-11T00:00:02.000Z")
            .unwrap();
        assert!(again.ids.contains(&last));
        let v: Value = serde_json::from_str(raws(&again).last().unwrap()).unwrap();
        assert_eq!(v["parentId"], json!(last));
        assert_eq!(v["id"], json!(next));
        assert_eq!(read(again.path()).unwrap().len(), 3);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_handles_crlf_truncated_and_empty_files() {
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"type":"session","version":3,"id":"hdr","timestamp":"t","cwd":"/tmp"}"#;
        let ok = r#"{"type":"reset_boundary","id":"rb","parentId":"hdr","timestamp":"t"}"#;

        // CRLF line endings: \r stripped, entries parse.
        let p = dir.join("crlf.jsonl");
        std::fs::write(&p, format!("{header}\r\n{ok}\r\n")).unwrap();
        let loaded = load(&p).unwrap();
        assert_eq!(loaded.lines.len(), 2);
        assert!(loaded.lines.iter().all(|l| !l.raw.ends_with('\r')));
        assert!(!loaded.invalid_header);
        assert!(loaded.malformed.is_empty());

        // Truncated last line: kept as a line, reported as malformed.
        let p = dir.join("truncated.jsonl");
        std::fs::write(
            &p,
            format!("{header}\n{ok}\n{{\"type\":\"reset_boundary\",\"id\":\"cut"),
        )
        .unwrap();
        let loaded = load(&p).unwrap();
        assert_eq!(loaded.lines.len(), 2);
        assert_eq!(loaded.malformed.len(), 1);
        assert_eq!(loaded.malformed[0].line_no, 3);
        assert_eq!(
            loaded.malformed[0].raw,
            "{\"type\":\"reset_boundary\",\"id\":\"cut"
        );

        // Empty file: no lines, nothing malformed, no header.
        let p = dir.join("empty.jsonl");
        std::fs::write(&p, b"").unwrap();
        let loaded = load(&p).unwrap();
        assert!(loaded.lines.is_empty());
        assert!(loaded.malformed.is_empty());
        assert!(loaded.title_slot.is_none());
        assert!(!loaded.invalid_header);

        // Same content through the mmap path (file > 1 MiB): identical results.
        let p = dir.join("big.jsonl");
        let mut body = format!("{header}\n").into_bytes();
        // Each line is the same entry with a long (valid JSON) id, sized so the
        // file crosses MMAP_THRESHOLD and the auto-selected reader mmaps it.
        let big_line = ok.replace("rb", &"x".repeat(8192));
        let mut big_line_count = 1usize; // header
        while (body.len() as u64) <= scan::MMAP_THRESHOLD {
            body.extend_from_slice(big_line.as_bytes());
            body.push(b'\n');
            big_line_count += 1;
        }
        assert!(body.len() as u64 > scan::MMAP_THRESHOLD);
        std::fs::write(&p, &body).unwrap();
        let big = load(&p).unwrap();
        assert!(big.malformed.is_empty());
        assert!(!big.invalid_header);
        assert_eq!(big.lines.len(), big_line_count);
        // Same content through the buffered path: identical results.
        let small_path = dir.join("small.jsonl");
        std::fs::write(&small_path, &body).unwrap();
        let small = load(&small_path).unwrap();
        assert_eq!(small.lines.len(), big.lines.len());
        assert_eq!(
            big.lines.iter().map(|l| l.raw.as_str()).collect::<Vec<_>>(),
            small
                .lines
                .iter()
                .map(|l| l.raw.as_str())
                .collect::<Vec<_>>()
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
