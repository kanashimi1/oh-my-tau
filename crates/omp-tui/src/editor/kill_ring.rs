//! The kill ring — the port of `pi-tui/src/kill-ring.ts`.
//!
//! A ring of killed text entries (cap 60). Consecutive kill commands
//! accumulate into one entry (backward kills prepend, forward kills append);
//! [`KillRing::rotate`] backs through the ring for yank-pop (`alt+y`).

/// Maximum number of entries kept (TS `KillRing.MAX_ENTRIES` = 60).
const MAX_ENTRIES: usize = 60;

/// Ring of killed text with accumulation and rotation.
#[derive(Debug, Clone, Default)]
pub struct KillRing {
    entries: Vec<String>,
    /// Index into `entries` for the next `rotate()` (yank-pop).
    /// `None` until a yank starts a rotation sequence.
    rotate_index: Option<usize>,
}

impl KillRing {
    /// An empty ring.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a kill. When `accumulate` is true (the previous editor action
    /// was also a kill), the text merges into the newest entry — backward
    /// kills (`prepend == true`) prepend, forward kills append.
    pub fn push(&mut self, text: &str, prepend: bool, accumulate: bool) {
        if text.is_empty() {
            return;
        }
        if accumulate {
            if let Some(last) = self.entries.last_mut() {
                if prepend {
                    last.insert_str(0, text);
                } else {
                    last.push_str(text);
                }
                return;
            }
        }
        self.entries.push(text.to_owned());
        if self.entries.len() > MAX_ENTRIES {
            self.entries.remove(0);
        }
    }

    /// The entry a plain yank (`ctrl+y`) would insert: the most recent kill.
    #[must_use]
    pub fn yank(&self) -> Option<&str> {
        self.entries.last().map(String::as_str)
    }

    /// Begin a yank-pop sequence after a yank: the next `rotate` returns the
    /// second-newest entry.
    pub fn start_pop(&mut self) {
        self.rotate_index = Some(self.entries.len().saturating_sub(1));
    }

    /// The next-older entry for yank-pop, wrapping around the ring.
    #[must_use]
    pub fn rotate(&mut self) -> Option<&str> {
        let len = self.entries.len();
        if len == 0 {
            return None;
        }
        let idx = self
            .rotate_index
            .unwrap_or(len - 1)
            .checked_sub(1)
            .unwrap_or(len - 1);
        self.rotate_index = Some(idx);
        Some(&self.entries[idx])
    }

    /// Whether a yank-pop sequence is in progress (the last editor action
    /// was a yank/yank-pop).
    #[must_use]
    pub fn popping(&self) -> bool {
        self.rotate_index.is_some()
    }

    /// End the pop sequence (any non-yank editor action calls this).
    pub fn end_pop(&mut self) {
        self.rotate_index = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_yank_most_recent() {
        let mut ring = KillRing::new();
        ring.push("hello", false, false);
        ring.push("world", false, false);
        assert_eq!(ring.yank(), Some("world"));
    }

    #[test]
    fn consecutive_backward_kills_accumulate_prepended() {
        let mut ring = KillRing::new();
        ring.push("bar", true, false);
        ring.push("foo", true, true);
        assert_eq!(ring.yank(), Some("foobar"));
    }

    #[test]
    fn consecutive_forward_kills_accumulate_appended() {
        let mut ring = KillRing::new();
        ring.push("foo", false, false);
        ring.push("bar", false, true);
        assert_eq!(ring.yank(), Some("foobar"));
    }

    #[test]
    fn rotate_walks_older_entries_and_wraps() {
        let mut ring = KillRing::new();
        ring.push("one", false, false);
        ring.push("two", false, false);
        ring.push("three", false, false);
        assert_eq!(ring.yank(), Some("three"));
        ring.start_pop();
        assert_eq!(ring.rotate(), Some("two"));
        assert_eq!(ring.rotate(), Some("one"));
        assert_eq!(ring.rotate(), Some("three")); // wraps
    }

    #[test]
    fn end_pop_resets_sequence() {
        let mut ring = KillRing::new();
        ring.push("x", false, false);
        ring.start_pop();
        assert!(ring.popping());
        ring.end_pop();
        assert!(!ring.popping());
    }

    #[test]
    fn ring_capped_at_sixty() {
        let mut ring = KillRing::new();
        for i in 0..70 {
            ring.push(&i.to_string(), false, false);
        }
        // Cap = 60: the 10 oldest entries were evicted.
        assert_eq!(ring.yank(), Some("69"));
        ring.start_pop();
        for _ in 0..58 {
            let _ = ring.rotate();
        }
        assert_eq!(ring.rotate(), Some("10"));
    }
}
