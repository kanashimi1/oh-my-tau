//! Modal editing state machine for the prompt editor — the port of
//! `pi-tui/src/vim.ts`.
//!
//! Deliberately a *pure* state machine: it reads a snapshot of the buffer
//! ([`VimBuffer`]) and returns the edits it wants applied ([`VimCommand`]), so
//! motions can be unit-tested without a terminal and the editor keeps sole
//! ownership of undo, atomic placeholder tokens, the kill ring, and change
//! notifications. Enabled by the host via the editor's `set_vim_mode`; off by
//! default.
//!
//! This is a usable subset of Vim, not a reimplementation of it: Normal,
//! Insert, Visual and Visual-Line modes, operators `d`/`y`/`c` composed with
//! motions (`h l j k 0 ^ $ w b e gg G`), `f`/`t`-style scanning is *not*
//! included, text objects (`iw aw i" a" i( a{ ip ap` …), counts, `x D C`,
//! `o O i a I A v V p P u`, and Vim's desired-column semantics for `j`/`k`.
//!
//! Columns are **char indices** into a line (Unicode scalar values), the
//! closest self-consistent analogue of the TS port's grapheme offsets. Ranges
//! in commands are always **end-exclusive** on `to`, so inclusive motions (`e`,
//! `cw`) expand `to` by one char when composing.

/// Vim editing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VimMode {
    /// Normal (command) mode; the initial mode.
    #[default]
    Normal,
    /// Insert mode — keys fall through to the editor's regular handling.
    Insert,
    /// Charwise Visual selection.
    Visual,
    /// Linewise Visual selection.
    VisualLine,
}

/// A composed operator waiting for a motion or text object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VimOperator {
    /// Delete.
    Delete,
    /// Yank (copy).
    Yank,
    /// Change (delete + insert).
    Change,
}

impl VimOperator {
    fn as_str(self) -> &'static str {
        match self {
            VimOperator::Delete => "d",
            VimOperator::Yank => "y",
            VimOperator::Change => "c",
        }
    }
}

/// A buffer position: line index plus char-index column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VimPosition {
    /// Line index.
    pub line: usize,
    /// Char index within the line.
    pub col: usize,
}

/// Read-only view of the editor buffer that motions resolve against.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VimBuffer {
    /// Logical lines (no trailing newlines).
    pub lines: Vec<String>,
    /// Cursor line index.
    pub cursor_line: usize,
    /// Cursor column as a char index into `lines[cursor_line]`.
    pub cursor_col: usize,
}

impl VimBuffer {
    /// A snapshot of a whole-document string split on `\n`, with the cursor
    /// given as a byte offset into that string (the flat editor's model).
    #[must_use]
    pub fn from_flat(text: &str, cursor_byte: usize) -> Self {
        let lines: Vec<String> = text.split('\n').map(str::to_string).collect();
        let mut cursor_line = 0usize;
        let mut cursor_col = 0usize;
        let mut offset = 0usize;
        for (idx, line) in lines.iter().enumerate() {
            let line_len = line.len();
            if cursor_byte <= offset + line_len {
                cursor_line = idx;
                cursor_col = line[..(cursor_byte - offset).min(line_len)].chars().count();
                break;
            }
            offset += line_len + 1;
        }
        Self {
            lines,
            cursor_line,
            cursor_col,
        }
    }

    /// The line under the cursor, or `""` past the end.
    #[must_use]
    pub fn line(&self) -> &str {
        self.lines.get(self.cursor_line).map_or("", String::as_str)
    }

    fn cursor(&self) -> VimPosition {
        VimPosition {
            line: self.cursor_line,
            col: self.cursor_col,
        }
    }
}

/// An edit the editor should apply. `to` is always an *exclusive* end offset,
/// so ranges compose the same way regardless of whether the originating motion
/// was inclusive (`e`) or exclusive (`w`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VimCommand {
    /// Move the cursor.
    Move { to: VimPosition },
    /// Switch mode.
    Mode(VimMode),
    /// Copy the range into the kill ring.
    Yank {
        from: VimPosition,
        to: VimPosition,
        linewise: bool,
    },
    /// Delete the range; `insert` distinguishes change-style deletes (which
    /// keep the lines for `c` on whole lines) from plain deletes.
    Delete {
        from: VimPosition,
        to: VimPosition,
        linewise: bool,
        insert: bool,
    },
    /// `o`/`O`: open a line below/above and park the cursor there.
    OpenLine { below: bool },
    /// `p`/`P`: paste the last yank after/before the cursor, `count` times.
    Paste { after: bool, count: usize },
    /// `u`: undo.
    Undo,
}

/// End-exclusive charwise span of a text object resolved on a single line.
type LineSpan = (usize, usize);

/// Full resolved extent of a text object.
struct TextObjectRange {
    from: VimPosition,
    to: VimPosition,
    linewise: bool,
}

/// A resolved motion.
struct Motion {
    to: VimPosition,
    /// Inclusive motions cover the char under `to` when used with an operator.
    inclusive: bool,
    linewise: bool,
}

/// Char length of a line.
fn line_len(line: &str) -> usize {
    line.chars().count()
}

/// The `i`-th char of a line, or `' '` past the end.
fn char_at(line: &str, col: usize) -> char {
    line.chars().nth(col).unwrap_or(' ')
}

/// Start index of the char after `col`, clamped to `line_len`.
fn next_char_start(line: &str, col: usize) -> usize {
    (col + 1).min(line_len(line))
}

/// Start index of the char before `col`, clamped to 0.
fn prev_char_start(line: &str, col: usize) -> usize {
    col.saturating_sub(1).min(line_len(line))
}

/// Index of the final char — where a Normal-mode cursor rests on a non-empty line.
fn last_char_start(line: &str) -> usize {
    line_len(line).saturating_sub(1)
}

fn first_non_blank(line: &str) -> usize {
    line.chars().position(|c| !c.is_whitespace()).unwrap_or(0)
}

/// Vim's `w`: stop at the end of the current word, then skip the gap that follows.
fn word_forward(line: &str, col: usize) -> usize {
    let mut next = move_word_right(line, col);
    let len = line_len(line);
    while next < len && char_at(line, next).is_whitespace() {
        next += 1;
    }
    next
}

/// Vim's `e`: the last char of the word the cursor is about to run into.
fn word_end(line: &str, col: usize) -> usize {
    let mut from = next_char_start(line, col);
    let len = line_len(line);
    while from < len && char_at(line, from).is_whitespace() {
        from += 1;
    }
    let end = move_word_right(line, from);
    col.max(prev_char_start(line, end))
}

/// `0` whitespace, `1` keyword, `2` punctuation — the classes `w` groups by
/// (`W` folds 2 into 1).
fn char_class(c: char, big: bool) -> u8 {
    if c.is_whitespace() {
        0
    } else if big || c.is_alphanumeric() || c == '_' {
        1
    } else {
        2
    }
}

/// A maximal same-class run — the atoms a counted `iw`/`aw` walks over.
struct Chunk {
    start: usize,
    end: usize,
    space: bool,
}

/// Split a line into maximal same-class runs.
fn chunk_line(line: &str, big: bool) -> Vec<Chunk> {
    let chars: Vec<char> = line.chars().collect();
    let mut chunks = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let cls = char_class(chars[i], big);
        let mut j = i + 1;
        while j < chars.len() && char_class(chars[j], big) == cls {
            j += 1;
        }
        chunks.push(Chunk {
            start: i,
            end: j,
            space: cls == 0,
        });
        i = j;
    }
    chunks
}

/// readline-style word start: back over the whitespace run, then the word run.
fn move_word_left(line: &str, col: usize) -> usize {
    let chars: Vec<char> = line.chars().collect();
    let mut i = col.min(chars.len());
    while i > 0 && chars[i - 1].is_whitespace() {
        i -= 1;
    }
    if i == 0 {
        return 0;
    }
    let cls = char_class(chars[i - 1], false);
    while i > 0 && char_class(chars[i - 1], false) == cls {
        i -= 1;
    }
    i
}

/// readline-style word end: over the word run, then the trailing whitespace.
fn move_word_right(line: &str, col: usize) -> usize {
    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut i = col.min(len);
    while i < len && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= len {
        return len;
    }
    let cls = char_class(chars[i], false);
    while i < len && char_class(chars[i], false) == cls {
        i += 1;
    }
    i
}

// vim-part3
