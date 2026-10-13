//! Prompt history — the port of the editor's history ring and
//! `HistoryStorage` (`components/editor.ts:513,988-1060`).
//!
//! - In-memory ring capped at 100 entries, most-recent-dedup on push
//!   (TS `#pushHistory`).
//! - Up from an empty buffer starts browsing at the newest entry; editing
//!   a recalled draft exits browsing and restores the saved draft
//!   (TS `#exitHistoryForEditing`).
//! - [`HistoryStore`] is the persistence seam (TS `HistoryStorage.add` /
//!   `getRecent`); [`MemoryHistory`] is the default, [`JsonlHistory`]
//!   appends one JSON object per line (`{"prompt","cwd","ts"}`).

use std::io;
use std::path::{Path, PathBuf};

/// Maximum in-memory entries (TS history cap = 100).
const MAX_ENTRIES: usize = 100;

/// Persistent-history seam (TS `HistoryStorage`).
pub trait HistoryStore {
    /// Record a submitted prompt in a working directory.
    fn add(&mut self, prompt: &str, cwd: &Path);
    /// Most recent prompts, newest last.
    fn get_recent(&self, limit: usize) -> Vec<String>;
}

/// In-memory ring (TS editor-local list seeded by the storage).
#[derive(Debug, Clone, Default)]
pub struct MemoryHistory {
    entries: Vec<String>,
}

impl MemoryHistory {
    /// An empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push with most-recent dedup + cap (TS `#pushHistory`).
    pub fn push(&mut self, prompt: &str) {
        if prompt.is_empty() {
            return;
        }
        if self.entries.last().is_some_and(|last| last == prompt) {
            return;
        }
        self.entries.push(prompt.to_owned());
        if self.entries.len() > MAX_ENTRIES {
            self.entries.remove(0);
        }
    }

    /// Number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there is nothing stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl HistoryStore for MemoryHistory {
    fn add(&mut self, prompt: &str, _cwd: &Path) {
        self.push(prompt);
    }

    fn get_recent(&self, limit: usize) -> Vec<String> {
        self.entries
            .iter()
            .rev()
            .take(limit)
            .rev()
            .cloned()
            .collect()
    }
}

/// JSONL-file history: one `{"prompt": "...", "cwd": "...", "ts": ...}` per
/// line. Reads are best-effort; corrupt lines are skipped.
#[derive(Debug, Clone)]
pub struct JsonlHistory {
    path: PathBuf,
    mem: MemoryHistory,
}

impl JsonlHistory {
    /// Load (or start) the history file at `path`.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        let mut mem = MemoryHistory::new();
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(p) = v["prompt"].as_str() {
                        mem.push(p);
                    }
                }
            }
        }
        Self {
            path: path.to_owned(),
            mem,
        }
    }

    /// Flush in-memory entries to disk (append). Called by the app on
    /// submit and on exit.
    ///
    /// # Errors
    /// Returns the underlying [`io::Error`] when the file cannot be
    /// created or written.
    pub fn flush(&self) -> io::Result<()> {
        use std::io::Write;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        // Histories are small (<100 lines): simplest correct scheme is a
        // full rewrite on every add.
        let mut out = Vec::new();
        for e in &self.mem.entries {
            let line = serde_json::json!({"prompt": e, "cwd": "", "ts": 0});
            writeln!(out, "{line}")?;
        }
        f.write_all(&out)
    }
}

impl HistoryStore for JsonlHistory {
    fn add(&mut self, prompt: &str, cwd: &Path) {
        self.mem.add(prompt, cwd);
        let _ = self.flush();
    }

    fn get_recent(&self, limit: usize) -> Vec<String> {
        self.mem.get_recent(limit)
    }
}

/// Browsing state over a [`MemoryHistory`]: which entry is shown and what
/// the user's pre-browse draft was (TS `historyIndex` + draft recall).
#[derive(Debug, Clone, Default)]
pub struct HistoryNav {
    /// `None` = not browsing (editing a draft); `Some(i)` = browsing,
    /// `i` counts from the end: 0 = most recent.
    index: Option<usize>,
    /// Draft saved when browsing started; restored on exit.
    draft: String,
}

impl HistoryNav {
    /// Whether a history entry (not the live draft) is on screen.
    #[must_use]
    pub fn browsing(&self) -> bool {
        self.index.is_some()
    }

    /// Up: start browsing at the newest entry, or walk one entry older.
    /// `current` is the text currently in the editor. Returns the entry to
    /// display (or the unchanged draft when history is empty).
    #[must_use]
    pub fn up(&mut self, history: &MemoryHistory, current: &str) -> String {
        match self.index {
            None => {
                if history.is_empty() {
                    return current.to_owned();
                }
                current.clone_into(&mut self.draft);
                self.index = Some(0);
                history
                    .entries
                    .last()
                    .cloned()
                    .unwrap_or_else(|| current.to_owned())
            }
            Some(i) if i + 1 < history.entries.len() => {
                self.index = Some(i + 1);
                let pos = history.entries.len() - 1 - (i + 1);
                history.entries[pos].clone()
            }
            Some(_) => current.to_owned(),
        }
    }

    /// Down: walk one entry newer; at the newest, exit browsing and restore
    /// the saved draft.
    #[must_use]
    pub fn down(&mut self, history: &MemoryHistory, current: &str) -> String {
        match self.index {
            None => current.to_owned(),
            Some(0) => self.exit_draft(),
            Some(i) => {
                self.index = Some(i - 1);
                let pos = history.entries.len() - 1 - (i - 1);
                history.entries[pos].clone()
            }
        }
    }

    /// Exit browsing: the draft text is what the editor should show.
    #[must_use]
    pub fn exit_draft(&mut self) -> String {
        self.index = None;
        std::mem::take(&mut self.draft)
    }

    /// Reset on submit (TS resets `historyIndex` and clears the draft).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(entries: &[&str]) -> MemoryHistory {
        let mut h = MemoryHistory::new();
        for e in entries {
            h.push(e);
        }
        h
    }

    #[test]
    fn push_dedupes_most_recent_and_caps() {
        let mut h = MemoryHistory::new();
        h.push("a");
        h.push("a");
        assert_eq!(h.len(), 1);
        h.push("b");
        assert_eq!(h.get_recent(10), vec!["a", "b"]);
        for i in 0..120 {
            h.push(&i.to_string());
        }
        assert_eq!(h.len(), 100);
    }

    #[test]
    fn nav_up_down_restores_draft() {
        let h = hist(&["one", "two", "three"]);
        let mut nav = HistoryNav::default();
        assert_eq!(nav.up(&h, ""), "three");
        assert_eq!(nav.up(&h, "three"), "two");
        assert_eq!(nav.up(&h, "two"), "one");
        assert_eq!(nav.up(&h, "one"), "one"); // clamped at oldest
        assert_eq!(nav.down(&h, "one"), "two");
        assert_eq!(nav.down(&h, "two"), "three");
        assert_eq!(nav.down(&h, "three"), ""); // exit -> draft restored
        assert!(!nav.browsing());
    }

    #[test]
    fn editing_exits_browsing_with_draft() {
        let h = hist(&["one", "two"]);
        let mut nav = HistoryNav::default();
        assert_eq!(nav.up(&h, "draft"), "two");
        assert_eq!(nav.exit_draft(), "draft");
    }
}
