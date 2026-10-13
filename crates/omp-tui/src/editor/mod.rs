//! The prompt editor — the port of `pi-tui/src/editor-component.ts` plus
//! the editing core in `pi-edit` (already vendored as a Rust crate).
//!
//! MVP scope implemented here: a multi-line text buffer with a cursor,
//! printable input, backspace/delete, left/right/home/end, Ctrl+J newline,
//! Enter to submit, Ctrl+C abort-or-quit handoff (the [`crate::app::App`]
//! decides). Everything fancier — kill ring (`kill-ring.ts`), vim modes
//! (`vim.ts`), bracketed paste (`bracketed-paste.ts`), space-hold
//! (`space-hold.ts`) — lives behind the same [`Editor`] API in `vim.rs` /
//! `kill_ring` stubs to be filled by the editor implementer.
//!
//! The editor never talks to the runtime directly; it returns
//! [`EditorOutcome`] and the app turns submissions into [`Action`]s.

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;
use ratatui::layout::Rect;
use ratatui::Frame;

pub mod vim;

/// What the editor wants to happen after handling a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorOutcome {
    /// Nothing special; buffer/cursor changed (or key ignored).
    Edited,
    /// Enter pressed: submit the buffer contents.
    Submit,
    /// Ctrl+C pressed.
    Interrupt,
}

/// Multi-line prompt buffer with a byte-index cursor.
#[derive(Debug, Clone, Default)]
pub struct Editor {
    /// Full buffer text (logical newlines `\n`).
    text: String,
    /// Cursor position as a byte index into `text`.
    cursor: usize,
}

impl Editor {
    /// An empty editor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current buffer text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Cursor byte offset.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Insert one character at the cursor.
    pub fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Insert a newline at the cursor.
    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    /// Delete the char before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i);
        self.text.replace_range(prev..self.cursor, "");
        self.cursor = prev;
    }

    /// Move the cursor one char left.
    pub fn move_left(&mut self) {
        self.cursor = self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i);
    }

    /// Move the cursor one char right.
    pub fn move_right(&mut self) {
        self.cursor = self.text[self.cursor..]
            .chars()
            .next()
            .map_or(self.cursor, |c| self.cursor + c.len_utf8());
    }

    /// Move to the start of the buffer.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Move to the end of the buffer.
    pub fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Take the buffer contents, leaving it empty (on submit).
    #[must_use]
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    /// Handle one key event, mutating the buffer.
    #[must_use]
    pub fn handle_key(&mut self, key: KeyEvent) -> EditorOutcome {
        match (key.code, key.modifiers) {
            (KeyCode::Enter, m) if !m.contains(KeyModifiers::CONTROL) && !m.contains(KeyModifiers::ALT) => {
                EditorOutcome::Submit
            }
            (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => EditorOutcome::Interrupt,
            (KeyCode::Char('j'), m) if m.contains(KeyModifiers::CONTROL) => {
                self.insert_newline();
                EditorOutcome::Edited
            }
            (KeyCode::Char(c), m) if !m.contains(KeyModifiers::CONTROL) => {
                self.insert_char(c);
                EditorOutcome::Edited
            }
            (KeyCode::Backspace, _) => {
                self.backspace();
                EditorOutcome::Edited
            }
            (KeyCode::Delete, _) => {
                let saved = self.cursor;
                self.move_right();
                let end = self.cursor;
                self.text.replace_range(saved..end, "");
                self.cursor = saved;
                EditorOutcome::Edited
            }
            (KeyCode::Left, _) => {
                self.move_left();
                EditorOutcome::Edited
            }
            (KeyCode::Right, _) => {
                self.move_right();
                EditorOutcome::Edited
            }
            (KeyCode::Home, _) => {
                self.move_home();
                EditorOutcome::Edited
            }
            (KeyCode::End, _) => {
                self.move_end();
                EditorOutcome::Edited
            }
            _ => EditorOutcome::Edited,
        }
    }
}

impl Component for Editor {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::widgets::{Paragraph, Wrap};
        let prompt = theme.glyph("prompt");
        let line = if self.text.is_empty() {
            ratatui::text::Line::from(format!("{prompt} "))
        } else {
            ratatui::text::Line::from(format!("{prompt} {}", self.text))
        };
        let p = Paragraph::new(line).wrap(Wrap { trim: false });
        frame.render_widget(p, area);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        match self.handle_key(key) {
            EditorOutcome::Edited => EventOutcome::Consumed,
            _ => EventOutcome::Ignored, // app owns submit/interrupt semantics
        }
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn typing_and_editing() {
        let mut e = Editor::new();
        for c in "ab".chars() {
            e.insert_char(c);
        }
        e.move_left();
        e.insert_char('x');
        assert_eq!(e.text(), "axb");
        e.backspace();
        assert_eq!(e.text(), "ab");
        e.move_home();
        e.insert_char('!');
        assert_eq!(e.text(), "!ab");
        assert_eq!(e.take(), "!ab");
        assert!(e.is_empty());
    }

    #[test]
    fn enter_submits_ctrl_j_newlines() {
        let mut e = Editor::new();
        assert_eq!(e.handle_key(key(KeyCode::Enter)), EditorOutcome::Submit);
        let ctrl_j = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL);
        assert_eq!(e.handle_key(ctrl_j), EditorOutcome::Edited);
        assert_eq!(e.text(), "\n");
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(e.handle_key(ctrl_c), EditorOutcome::Interrupt);
    }
}
