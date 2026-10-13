//! The editor's text model — the port of `EditorState` /
//! `components/editor.ts` state core.
//!
//! Text is a list of logical lines joined by `\n` (TS `lines: string[]`
//! with `cursorLine` / `cursorCol`). The cursor column is a **byte** offset
//! within the line; all motion/insert/delete operations are grapheme-aware
//! (TS `Intl.Segmenter`, `editor.ts:393-425`) via
//! [`unicode_segmentation`], so combining sequences and wide characters
//! move as one unit.
//!
//! The buffer is pure data + operations; editing policy (bindings, kill
//! ring, history, undo) lives in [`super::Editor`].

use unicode_segmentation::UnicodeSegmentation;

/// One cursor position: line index + byte column within that line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cursor {
    /// Logical line (0-based).
    pub line: usize,
    /// Byte offset within the line, always on a `char` boundary.
    pub col: usize,
}

/// A multi-line text buffer with a grapheme-aware cursor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextBuffer {
    lines: Vec<String>,
    cursor: Cursor,
}

impl TextBuffer {
    /// An empty buffer with one empty line.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            cursor: Cursor::default(),
        }
    }

    /// Build a buffer from text (logical newlines split into lines).
    #[must_use]
    pub fn from_text(text: &str) -> Self {
        let mut lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
        if lines.is_empty() {
            lines.push(String::new());
        }
        Self {
            lines,
            cursor: Cursor::default(),
        }
    }

    /// Full text with logical `\n` separators.
    #[must_use]
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Borrow the logical lines.
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Current cursor.
    #[must_use]
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    /// Set the cursor, clamping to valid boundaries.
    pub fn set_cursor(&mut self, cursor: Cursor) {
        let line = cursor.line.min(self.lines.len() - 1);
        let col = self.clamp_col(line, cursor.col);
        self.cursor = Cursor { line, col };
    }

    fn clamp_col(&self, line: usize, col: usize) -> usize {
        let len = self.lines[line].len();
        if col > len {
            len
        } else {
            col
        }
    }

    /// Whether the buffer holds no characters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(String::is_empty)
    }

    /// The line the cursor is on.
    #[must_use]
    pub fn current_line(&self) -> &str {
        &self.lines[self.cursor.line]
    }

    /// Byte offset of the cursor in the whole-buffer text.
    #[must_use]
    pub fn flat_cursor(&self) -> usize {
        self.lines[..self.cursor.line]
            .iter()
            .map(|l| l.len() + 1)
            .sum::<usize>()
            + self.cursor.col
    }

    /// Replace the whole buffer, cursor to end.
    pub fn set_text(&mut self, text: &str) {
        *self = Self::from_text(text);
        self.move_to_end();
    }

    /// Insert one character at the cursor.
    pub fn insert_char(&mut self, c: char) {
        if c == '\n' {
            self.split_line();
            return;
        }
        self.lines[self.cursor.line].insert(self.cursor.col, c);
        self.cursor.col += c.len_utf8();
    }

    /// Insert a string (may contain `\n`, which splits lines).
    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() {
            self.insert_char(c);
        }
    }

    /// Split the current line at the cursor (newline insertion).
    pub fn split_line(&mut self) {
        let line = &mut self.lines[self.cursor.line];
        let tail: String = line[self.cursor.col..].to_owned();
        line.truncate(self.cursor.col);
        self.lines.insert(self.cursor.line + 1, tail);
        self.cursor = Cursor {
            line: self.cursor.line + 1,
            col: 0,
        };
    }

    /// Delete the grapheme before the cursor. Returns the removed text.
    pub fn delete_backward(&mut self) -> Option<String> {
        if self.cursor.col == 0 {
            if self.cursor.line == 0 {
                return None;
            }
            // Join with the previous line.
            let line = self.cursor.line;
            let tail = self.lines.remove(line);
            self.cursor = Cursor {
                line: line - 1,
                col: self.lines[line - 1].len(),
            };
            self.lines[self.cursor.line].push_str(&tail);
            return Some("\n".to_owned());
        }
        let (start, grapheme) = prev_grapheme(&self.lines[self.cursor.line], self.cursor.col)?;
        let removed = grapheme.to_owned();
        self.lines[self.cursor.line].replace_range(start..self.cursor.col, "");
        self.cursor.col = start;
        Some(removed)
    }

    /// Delete the grapheme at/after the cursor. Returns the removed text.
    pub fn delete_forward(&mut self) -> Option<String> {
        let line = &self.lines[self.cursor.line];
        if self.cursor.col >= line.len() {
            if self.cursor.line + 1 >= self.lines.len() {
                return None;
            }
            let next = self.lines.remove(self.cursor.line + 1);
            self.lines[self.cursor.line].push_str(&next);
            return Some("\n".to_owned());
        }
        let (end, grapheme) = next_grapheme(line, self.cursor.col)?;
        let removed = grapheme.to_owned();
        self.lines[self.cursor.line].replace_range(self.cursor.col..end, "");
        Some(removed)
    }

    /// Move one grapheme left.
    pub fn move_left(&mut self) {
        if let Some((col, _)) = prev_grapheme(self.current_line(), self.cursor.col) {
            self.cursor.col = col;
        } else if self.cursor.line > 0 {
            self.cursor = Cursor {
                line: self.cursor.line - 1,
                col: self.lines[self.cursor.line - 1].len(),
            };
        }
    }

    /// Move one grapheme right.
    pub fn move_right(&mut self) {
        let line = self.current_line();
        if let Some((end, _)) = next_grapheme(line, self.cursor.col) {
            self.cursor.col = end;
        } else if self.cursor.line + 1 < self.lines.len() {
            self.cursor = Cursor {
                line: self.cursor.line + 1,
                col: 0,
            };
        }
    }

    /// Move to the start of the current line.
    pub fn move_line_start(&mut self) {
        self.cursor.col = 0;
    }

    /// Move to the end of the current line.
    pub fn move_line_end(&mut self) {
        self.cursor.col = self.current_line().len();
    }

    /// Move to the very start of the buffer.
    pub fn move_to_start(&mut self) {
        self.cursor = Cursor::default();
    }

    /// Move to the very end of the buffer.
    pub fn move_to_end(&mut self) {
        let line = self.lines.len() - 1;
        self.cursor = Cursor {
            line,
            col: self.lines[line].len(),
        };
    }

    /// Move up one line, keeping the column (clamped).
    pub fn move_up(&mut self) {
        if self.cursor.line > 0 {
            let col = self.cursor.col;
            self.cursor.line -= 1;
            self.cursor.col = self.clamp_col(self.cursor.line, col);
        }
    }

    /// Move down one line, keeping the column (clamped).
    pub fn move_down(&mut self) {
        if self.cursor.line + 1 < self.lines.len() {
            let col = self.cursor.col;
            self.cursor.line += 1;
            self.cursor.col = self.clamp_col(self.cursor.line, col);
        }
    }

    /// Readline word boundaries: the previous word start at/before `col`.
    /// Words are runs of alphanumeric/underscore chars; punctuation and
    /// whitespace are separate runs (`utils.moveWordLeft`).
    #[must_use]
    pub fn word_left(&self, mut col: usize) -> usize {
        let line = self.current_line();
        col = col.min(line.len());
        let mut chars: Vec<(usize, char)> = line.char_indices().collect();
        chars.retain(|(i, _)| *i < col);
        // Skip trailing whitespace/punct of the current word, then skip the
        // word itself: land on the first char of the previous word.
        let mut seen_word_char = false;
        while let Some((i, c)) = chars.pop() {
            if is_word_char(c) {
                seen_word_char = true;
            } else if seen_word_char {
                return i + c.len_utf8();
            }
        }
        0
    }

    /// Readline word end at/after `col` (`utils.moveWordRight`).
    #[must_use]
    pub fn word_right(&self, mut col: usize) -> usize {
        let line = self.current_line();
        col = col.min(line.len());
        let mut it = line[col..].char_indices();
        // Skip the current word, then any non-word chars: land on the first
        // char of the next word.
        let mut seen_word_char = false;
        for (i, c) in &mut it {
            if is_word_char(c) {
                seen_word_char = true;
            } else if seen_word_char {
                return col + i;
            }
        }
        line.len()
    }

    /// Move to the previous word start.
    pub fn move_word_left(&mut self) {
        self.cursor.col = self.word_left(self.cursor.col);
    }

    /// Move to the next word end.
    pub fn move_word_right(&mut self) {
        self.cursor.col = self.word_right(self.cursor.col);
    }

    /// Delete the word before the cursor. Returns the removed text.
    pub fn delete_word_backward(&mut self) -> String {
        if self.cursor.col == 0 {
            if self.cursor.line == 0 {
                return String::new();
            }
            self.move_left();
            return self.delete_forward().unwrap_or_default();
        }
        let start = self.word_left(self.cursor.col);
        let removed: String = self.lines[self.cursor.line][start..self.cursor.col].to_owned();
        self.lines[self.cursor.line].replace_range(start..self.cursor.col, "");
        self.cursor.col = start;
        removed
    }

    /// Delete the word at/after the cursor. Returns the removed text.
    pub fn delete_word_forward(&mut self) -> String {
        let end = self.word_right(self.cursor.col);
        if end == self.cursor.col {
            return String::new();
        }
        let removed: String = self.lines[self.cursor.line][self.cursor.col..end].to_owned();
        self.lines[self.cursor.line].replace_range(self.cursor.col..end, "");
        removed
    }

    /// Delete from cursor to line start. Returns the removed text.
    pub fn delete_to_line_start(&mut self) -> String {
        let removed: String = self.lines[self.cursor.line][..self.cursor.col].to_owned();
        self.lines[self.cursor.line].replace_range(..self.cursor.col, "");
        self.cursor.col = 0;
        removed
    }

    /// Delete from cursor to line end. Returns the removed text.
    pub fn delete_to_line_end(&mut self) -> String {
        let removed: String = self.lines[self.cursor.line][self.cursor.col..].to_owned();
        self.lines[self.cursor.line].truncate(self.cursor.col);
        removed
    }

    /// Replace a byte range on one line with new text; cursor lands at the
    /// end of the inserted text. Used by yank/kill restores.
    pub fn replace_range(&mut self, line: usize, start: usize, end: usize, text: &str) {
        self.lines[line].replace_range(start..end, text);
        let mut cursor = Cursor { line, col: start };
        for c in text.chars() {
            if c == '\n' {
                cursor.line += 1;
                cursor.col = 0;
            } else {
                cursor.col += c.len_utf8();
            }
        }
        self.set_cursor(cursor);
    }
}

/// Whether a char belongs to a "word" for readline motions.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The grapheme cluster ending at byte offset `col` -> `(start, cluster)`.
fn prev_grapheme(line: &str, col: usize) -> Option<(usize, &str)> {
    line[..col].grapheme_indices(true).next_back()
}

/// The grapheme cluster starting at byte offset `col` -> `(end, cluster)`.
fn next_grapheme(line: &str, col: usize) -> Option<(usize, &str)> {
    line[col..]
        .grapheme_indices(true)
        .next()
        .map(|(i, g)| (col + i + g.len(), g))
}
