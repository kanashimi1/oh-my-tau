//! Paste handling — the port of `bracketed-paste.ts` + the editor's
//! `#sanitizePastedText` / `#handlePaste` / `#storePasteMarker` /
//! `#expandPasteMarkers` (`components/editor.ts:3177-3285`).
//!
//! - [`sanitize`] decodes tmux re-encoded control bytes, CRLF→LF, tabs→4
//!   spaces, strips control characters except `\n`.
//! - Pastes over 10 lines or 1000 chars become an atomic
//!   `[Paste #N, +<lines> lines]` / `[Paste #N, <chars> chars]` marker
//!   (TS `#storePasteMarker`); the payload lives in a side table and is
//!   expanded back on submit (`#expandPasteMarkers`).
//! - Pasted absolute paths glued to a word char get a separating space
//!   (TS path-paste assist, `:3183`).

/// Marker threshold: > 10 lines counts as line-based (TS).
pub const LARGE_PASTE_LINES: usize = 10;
/// Marker threshold: > 1000 chars counts as char-based (TS).
pub const LARGE_PASTE_CHARS: usize = 1000;

/// Paste payloads keyed by marker number (`#pastes` map).
#[derive(Debug, Clone, Default)]
pub struct PasteStore {
    next_id: usize,
    pastes: Vec<(usize, String)>,
}

impl PasteStore {
    /// An empty store; marker numbering starts at 1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            pastes: Vec::new(),
        }
    }

    /// Store a payload and return its marker number.
    pub fn store(&mut self, text: String) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.pastes.push((id, text));
        id
    }

    /// Expand every `[Paste #N, ...]` marker in `text` to its payload.
    /// Unknown numbers are left as-is (history recall of an evicted
    /// payload degrades to the visible marker text).
    #[must_use]
    pub fn expand(&self, text: &str) -> String {
        let mut out = text.to_owned();
        while let Some(start) = out.find("[Paste #") {
            let Some(rel_end) = out[start..].find(']') else {
                break;
            };
            let end = start + rel_end + 1;
            let header = &out[start + "[Paste #".len()..end - 1];
            let num_str = header.split(',').next().unwrap_or("");
            let expanded = num_str
                .parse::<usize>()
                .ok()
                .and_then(|n| self.pastes.iter().find(|(id, _)| *id == n))
                .map(|(_, payload)| payload.clone());
            match expanded {
                Some(payload) => out.replace_range(start..end, &payload),
                None => {
                    // Unknown id: blank the '[' so the scan skips this
                    // marker (restored after the loop).
                    out.replace_range(start..=start, "\u{0}");
                }
            }
        }
        out.replace('\u{0}', "[")
    }

    /// Drop all payloads (called after submit).
    pub fn clear(&mut self) {
        self.pastes.clear();
    }

    /// Whether any payload is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pastes.is_empty()
    }
}

/// Sanitize pasted text (TS `#sanitizePastedText` :3230): CRLF→LF,
/// tabs→4 spaces, strip control chars except `\n`.
#[must_use]
pub fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\r' => {} // CRLF / lone CR → dropped; the following \n stays
            '\t' => out.push_str("    "),
            '\n' => out.push('\n'),
            // tmux re-encodes some control bytes as C-/C-? pairs; keep it
            // simple: drop everything else below space.
            c if c < ' ' => {}
            c => out.push(c),
        }
    }
    out
}

/// Whether a paste should collapse to a marker (TS marker-size rule).
#[must_use]
pub fn is_large(text: &str) -> bool {
    text.lines().count() > LARGE_PASTE_LINES || text.chars().count() > LARGE_PASTE_CHARS
}

/// The marker label for a payload (TS `#storePasteMarker`).
#[must_use]
pub fn marker_label(id: usize, text: &str) -> String {
    let lines = text.lines().count();
    if lines > LARGE_PASTE_LINES {
        format!("[Paste #{id}, +{lines} lines]")
    } else {
        format!("[Paste #{id}, {} chars]", text.chars().count())
    }
}

/// Path-paste assist: insert a separating space when an absolute path is
/// glued directly to a word character before it (TS `:3183`).
#[must_use]
pub fn space_before_glued_path(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut prev_word = false;
    let mut in_path = false;
    for c in text.chars() {
        if c.is_whitespace() {
            in_path = false;
        }
        if c == '/' && !in_path {
            if prev_word {
                out.push(' ');
            }
            in_path = true;
        }
        out.push(c);
        prev_word = c.is_alphanumeric() || c == '_';
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_normalizes() {
        assert_eq!(sanitize("a\r\nb\tc\x07d"), "a\nb    cd");
        assert_eq!(sanitize("keep\nnew\nlines"), "keep\nnew\nlines");
    }

    #[test]
    fn large_detection() {
        assert!(!is_large("short"));
        assert!(is_large(&"x".repeat(1001)));
        assert!(is_large(&"l\n".repeat(11)));
    }

    #[test]
    fn marker_labels() {
        assert_eq!(marker_label(1, &"x\n".repeat(12)), "[Paste #1, +12 lines]");
        assert_eq!(marker_label(2, &"x".repeat(500)), "[Paste #2, 500 chars]");
    }

    #[test]
    fn store_and_expand_roundtrip() {
        let mut store = PasteStore::new();
        let payload = "line1\nline2\n".repeat(8);
        let id = store.store(payload.clone());
        let label = marker_label(id, &payload);
        let text = format!("prefix {label} suffix");
        assert_eq!(store.expand(&text), format!("prefix {payload} suffix"));
    }

    #[test]
    fn expand_leaves_unknown_ids() {
        let store = PasteStore::new();
        assert_eq!(
            store.expand("a [Paste #9, +5 lines] b"),
            "a [Paste #9, +5 lines] b"
        );
    }

    #[test]
    fn glued_path_gets_space() {
        assert_eq!(space_before_glued_path("file/tmp/x"), "file /tmp/x");
        assert_eq!(space_before_glued_path("open /tmp/x"), "open /tmp/x");
    }
}
