//! Undo for the editor — the port of the editor's `#undoStack` /
//! `#recordUndoState` / `#applyUndo` (`components/editor.ts:747,3594-3660`).
//!
//! Snapshots are whole-buffer `(lines, cursor)` pairs taken before each
//! mutating operation; composite operations (paste, yank-pop) bracket
//! themselves with suspend/resume so only one snapshot lands.

use crate::editor::buffer::{Cursor, TextBuffer};

/// How many snapshots to keep (TS keeps 100).
const MAX_UNDO: usize = 100;

/// A stack of buffer snapshots.
#[derive(Debug, Clone, Default)]
pub struct UndoStack {
    snapshots: Vec<(TextBuffer,)>,
    /// Suspension counter: while `> 0` (a composite operation), snapshots
    /// are not recorded (`#withUndoSuspended`).
    suspended: usize,
}

impl UndoStack {
    /// An empty stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the pre-mutation state of `buffer`.
    pub fn record(&mut self, buffer: &TextBuffer) {
        if self.suspended > 0 {
            return;
        }
        self.snapshots.push((buffer.clone(),));
        if self.snapshots.len() > MAX_UNDO {
            self.snapshots.remove(0);
        }
    }

    /// Suspend recording for a composite operation; call [`UndoStack::resume`]
    /// when done.
    pub fn suspend(&mut self) {
        self.suspended += 1;
    }

    /// Resume recording after [`UndoStack::suspend`].
    pub fn resume(&mut self) {
        self.suspended = self.suspended.saturating_sub(1);
    }

    /// Pop the newest snapshot into `buffer`, returning its cursor.
    #[must_use]
    pub fn undo(&mut self, buffer: &mut TextBuffer) -> Option<Cursor> {
        let (snapshot,) = self.snapshots.pop()?;
        let cursor = snapshot.cursor();
        *buffer = snapshot;
        Some(cursor)
    }

    /// Drop all snapshots (TS clears the undo stack on submit).
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Whether no snapshots exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_restores_previous_buffer() {
        let mut stack = UndoStack::new();
        let mut buf = TextBuffer::new();
        stack.record(&buf);
        buf.insert_str("hello");
        stack.record(&buf);
        buf.insert_char('!');
        assert_eq!(buf.text(), "hello!");
        let cursor = stack.undo(&mut buf).unwrap();
        buf.set_cursor(cursor);
        assert_eq!(buf.text(), "hello");
        let _ = stack.undo(&mut buf);
        assert!(buf.is_empty());
        assert!(stack.undo(&mut buf).is_none());
    }

    #[test]
    fn suspended_ops_record_one_snapshot() {
        let mut stack = UndoStack::new();
        let mut buf = TextBuffer::new();
        stack.record(&buf);
        stack.suspend();
        buf.insert_str("abc");
        stack.record(&buf); // ignored while suspended
        buf.insert_str("def");
        stack.record(&buf); // ignored
        stack.resume();
        stack.record(&buf);
        buf.insert_char('!');
        let _ = stack.undo(&mut buf);
        assert_eq!(buf.text(), "abcdef");
        let _ = stack.undo(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn clear_empties_stack() {
        let mut stack = UndoStack::new();
        let buf = TextBuffer::new();
        stack.record(&buf);
        stack.clear();
        assert!(stack.is_empty());
    }
}
