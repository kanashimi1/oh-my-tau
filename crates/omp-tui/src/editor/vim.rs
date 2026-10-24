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
//! Insert and Visual-Line modes, operators `d`/`y`/`c` composed with
//! motions (`h l j k 0 ^ $ w b e W B E gg G`, `f`/`F`/`t`/`T` char search),
//! text objects (`iw aw i" a" i( a{ ip ap` …), counts, `x D C`,
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

use crate::keys::{KeyCode, KeyEvent, KeyModifiers};

/// `f`/`F`/`t`/`T` char search awaiting its target character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharSearch {
    /// `f`: forward, land *on* the match.
    ForwardFind,
    /// `F`: backward, land *on* the match.
    BackwardFind,
    /// `t`: forward, land one char before the match.
    ForwardTo,
    /// `T`: backward, land one char after the match.
    BackwardTo,
}

/// The modal editing state machine. Pure: it snapshots the buffer via
/// [`VimBuffer`] and returns the edits it wants applied as [`VimCommand`]s.
#[derive(Debug, Clone, Default)]
pub struct VimState {
    /// Current editing mode.
    pub mode: VimMode,
    /// Fixed end of a Visual selection; the cursor is the moving end.
    pub anchor: Option<VimPosition>,
    /// The last text yanked by a Vim command, written back by the host that
    /// applies the commands (the state machine itself never touches buffer
    /// text). `p`/`P` paste this register.
    pub register: Option<String>,
    /// Composed operator waiting for a motion or text object.
    operator: Option<VimOperator>,
    /// Digits of a pending count ("" when none).
    count: String,
    /// `g` typed, waiting for the second `g`.
    pending_g: bool,
    /// `i`/`a` typed after an operator or in Visual mode — waiting for the object key.
    text_object: Option<char>,
    /// `f`/`F`/`t`/`T` typed — waiting for the target character.
    pending_char: Option<CharSearch>,
    /// Vim's "desired column": `j`/`k` remember the column you started from,
    /// so descending through a short line and back out returns to it instead
    /// of collapsing permanently. `None` means the next vertical motion
    /// anchors it to the live cursor column; `usize::MAX` is `$`'s sticky
    /// end-of-line, which keeps `$j` on the end of each line. Every
    /// non-vertical command clears it.
    desired_col: Option<usize>,
}

/// Map a key event onto the single-character key Vim understands, normalizing
/// arrows/Home/End/Backspace/Delete/Space onto their Vim equivalents (TS
/// `VIM_NAV_KEYS`). Chords with ctrl/alt are not Vim's — the caller falls
/// through to its regular handling.
fn vim_key_of(key: &KeyEvent) -> Option<char> {
    let mods = key.modifiers;
    if mods.contains(KeyModifiers::CONTROL) || mods.contains(KeyModifiers::ALT) {
        return None;
    }
    match key.code {
        KeyCode::Char(c) => Some(c),
        KeyCode::Left | KeyCode::Backspace => Some('h'),
        KeyCode::Right => Some('l'),
        KeyCode::Down => Some('j'),
        KeyCode::Up => Some('k'),
        KeyCode::Home => Some('0'),
        KeyCode::End => Some('$'),
        KeyCode::Delete => Some('x'),
        _ => None,
    }
}

/// `W`'s forward step: end of the current WORD run, then the gap after it.
fn word_forward_big(line: &str, col: usize) -> usize {
    let len = line_len(line);
    let mut i = col.min(len);
    let start_cls = char_class(char_at(line, i), true);
    while i < len && char_class(char_at(line, i), true) == start_cls {
        i += 1;
    }
    while i < len && char_at(line, i).is_whitespace() {
        i += 1;
    }
    i
}

/// `E`: the last char of the WORD the cursor is about to run into.
fn word_end_big(line: &str, col: usize) -> usize {
    let mut from = next_char_start(line, col);
    let len = line_len(line);
    while from < len && char_at(line, from).is_whitespace() {
        from += 1;
    }
    let cls = char_class(char_at(line, from), true);
    let mut end = from;
    while end < len && char_class(char_at(line, end), true) == cls {
        end += 1;
    }
    col.max(prev_char_start(line, end))
}

/// `B`: the start of the current WORD, or of the previous one.
fn word_backward_big(line: &str, col: usize) -> usize {
    let chars: Vec<char> = line.chars().collect();
    let mut i = col.min(chars.len());
    while i > 0 && chars[i - 1].is_whitespace() {
        i -= 1;
    }
    if i == 0 {
        return 0;
    }
    let cls = char_class(chars[i - 1], true);
    while i > 0 && char_class(chars[i - 1], true) == cls {
        i -= 1;
    }
    i
}

/// `iw`/`aw`/`iW`/`aW`, line-local like Vim's. `aw` takes the trailing
/// whitespace run, or the leading one when the word ends the line; starting
/// on whitespace instead takes it plus the following word.
fn word_object(line: &str, col: usize, around: bool, big: bool, count: usize) -> Option<LineSpan> {
    let chunks = chunk_line(line, big);
    if chunks.is_empty() {
        return None;
    }
    let last = chunks.len() - 1;
    let mut first = chunks.iter().position(|c| col < c.end).unwrap_or(last);
    let mut end = first;
    if !around {
        end = (first + count - 1).min(last);
    } else if chunks[first].space {
        end = (first + 2 * count - 1).min(last);
    } else {
        for n in 0..count {
            if n > 0 && end < last {
                end += 1;
            }
            if end < last && chunks[end + 1].space {
                end += 1;
            }
        }
        if !chunks[end].space && first > 0 && chunks[first - 1].space {
            first -= 1;
        }
    }
    Some((chunks[first].start, chunks[end].end))
}

/// `i"`/`a"` and friends, line-local like Vim's: quotes pair off left to
/// right, and the cursor selects the first pair that ends at or after it.
/// `a` swallows the trailing whitespace, or the leading run when there is none.
fn quote_object(line: &str, col: usize, quote: char, around: bool) -> Option<LineSpan> {
    let chars: Vec<char> = line.chars().collect();
    let mut marks = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '\\' {
            i += 1;
        } else if chars[i] == quote {
            marks.push(i);
        }
        i += 1;
    }
    let mut p = 0usize;
    while p + 1 < marks.len() {
        let open = marks[p];
        let close = marks[p + 1];
        if col > close {
            p += 2;
            continue;
        }
        if !around {
            return Some((open + 1, close));
        }
        let mut end = close + 1;
        let mut start = open;
        while end < chars.len() && chars[end].is_whitespace() {
            end += 1;
        }
        if end == close + 1 {
            while start > 0 && chars[start - 1].is_whitespace() {
                start -= 1;
            }
        }
        return Some((start, end));
    }
    None
}

/// Buffer flattened to one string plus the cursor's offset in it, so scans
/// can cross lines. Offsets are char indices.
fn flatten(buf: &VimBuffer) -> (String, usize) {
    let text = buf.lines.join("\n");
    let mut cursor = buf.cursor_col;
    for line in buf.lines.iter().take(buf.cursor_line.min(buf.lines.len())) {
        cursor += line_len(line) + 1;
    }
    (text, cursor)
}

fn offset_to_pos(lines: &[String], offset: usize) -> VimPosition {
    let mut remaining = offset;
    for (line, text) in lines.iter().enumerate() {
        let width = line_len(text);
        if remaining <= width {
            return VimPosition {
                line,
                col: remaining,
            };
        }
        remaining -= width + 1;
    }
    let line = lines.len().saturating_sub(1);
    VimPosition {
        line,
        col: lines.get(line).map_or(0, |l| line_len(l)),
    }
}

/// `i(`/`a{`… — the innermost pair enclosing the cursor, counting nesting
/// and spanning lines. A cursor sitting on either delimiter counts as being
/// on that pair. Charwise in both variants; Vim's linewise-ish `i{`
/// reshaping is deliberately not reproduced.
fn bracket_object(
    buf: &VimBuffer,
    open: char,
    close: char,
    around: bool,
) -> Option<TextObjectRange> {
    let (text, cursor) = flatten(buf);
    let chars: Vec<char> = text.chars().collect();
    let mut depth = 0usize;
    let mut open_at: Option<usize> = None;
    let start = cursor.min(chars.len().saturating_sub(1));
    for i in (0..=start).rev() {
        let ch = chars[i];
        if ch == close && i != cursor {
            depth += 1;
        } else if ch == open {
            if depth == 0 {
                open_at = Some(i);
                break;
            }
            depth -= 1;
        }
    }
    let open_at = open_at?;
    depth = 0;
    let mut close_at: Option<usize> = None;
    for (i, &ch) in chars.iter().enumerate().skip(open_at + 1) {
        if ch == open {
            depth += 1;
        } else if ch == close {
            if depth == 0 {
                close_at = Some(i);
                break;
            }
            depth -= 1;
        }
    }
    let close_at = close_at?;
    let from = offset_to_pos(&buf.lines, if around { open_at } else { open_at + 1 });
    let to = offset_to_pos(&buf.lines, if around { close_at + 1 } else { close_at });
    Some(TextObjectRange {
        from,
        to,
        linewise: false,
    })
}

/// `ip`/`ap`: the run of lines matching the cursor line's blankness. `ap`
/// also takes the run that follows, or the one before it when the paragraph
/// ends the buffer. Always linewise.
fn paragraph_object(buf: &VimBuffer, around: bool) -> TextObjectRange {
    let last = buf.lines.len().saturating_sub(1);
    let blank = |line: usize| buf.lines.get(line).map_or(true, |l| l.trim().is_empty());
    let target = blank(buf.cursor_line);
    let mut first = buf.cursor_line;
    let mut end = buf.cursor_line;
    while first > 0 && blank(first - 1) == target {
        first -= 1;
    }
    while end < last && blank(end + 1) == target {
        end += 1;
    }
    if around {
        let stop = end;
        while end < last && blank(end + 1) != target {
            end += 1;
        }
        if end == stop {
            while first > 0 && blank(first - 1) != target {
                first -= 1;
            }
        }
    }
    TextObjectRange {
        from: VimPosition {
            line: first,
            col: 0,
        },
        to: VimPosition {
            line: end,
            col: buf.lines.get(end).map_or(0, |l| line_len(l)),
        },
        linewise: true,
    }
}

/// Normalized, end-exclusive span covered by a Visual selection.
///
/// Charwise selections include the char under the cursor (Vim semantics);
/// linewise selections span whole lines, with `to.col` at the end of the
/// last line so the caller can decide whether the trailing newline goes too.
#[must_use]
pub fn visual_range(
    buf: &VimBuffer,
    anchor: VimPosition,
    linewise: bool,
) -> (VimPosition, VimPosition) {
    let cursor = buf.cursor();
    let (from, to) = if pos_less(cursor, anchor) {
        (cursor, anchor)
    } else {
        (anchor, cursor)
    };
    if linewise {
        let last_line = to.line.min(buf.lines.len().saturating_sub(1));
        return (
            VimPosition {
                line: from.line,
                col: 0,
            },
            VimPosition {
                line: last_line,
                col: buf.lines.get(last_line).map_or(0, |l| line_len(l)),
            },
        );
    }
    let col = next_char_start(buf.lines.get(to.line).map_or("", String::as_str), to.col);
    (from, VimPosition { line: to.line, col })
}

fn pos_less(a: VimPosition, b: VimPosition) -> bool {
    a.line < b.line || (a.line == b.line && a.col < b.col)
}

/// Find the next occurrence of `ch` for an `f`/`F`/`t`/`T` motion. Returns
/// the *landing* column (already adjusted for `t`/`T`).
fn find_char(line: &str, col: usize, ch: char, search: CharSearch) -> Option<usize> {
    let chars: Vec<char> = line.chars().collect();
    match search {
        CharSearch::ForwardFind | CharSearch::ForwardTo => {
            let mut i = col + 1;
            while i < chars.len() {
                if chars[i] == ch {
                    return Some(if search == CharSearch::ForwardTo {
                        i.saturating_sub(1)
                    } else {
                        i
                    });
                }
                i += 1;
            }
            None
        }
        CharSearch::BackwardFind | CharSearch::BackwardTo => {
            let mut i = col as isize - 1;
            while i >= 0 {
                if chars[i as usize] == ch {
                    return Some(if search == CharSearch::BackwardTo {
                        (i as usize).saturating_add(1)
                    } else {
                        i as usize
                    });
                }
                i -= 1;
            }
            None
        }
    }
}

impl VimState {
    /// A fresh state in Normal mode.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// True while a count, operator, `g`, text-object, or char-search prefix
    /// is half-typed — Escape cancels it.
    #[must_use]
    pub fn pending(&self) -> bool {
        !self.count.is_empty()
            || self.operator.is_some()
            || self.pending_g
            || self.text_object.is_some()
            || self.pending_char.is_some()
    }

    /// The half-typed command as Vim would echo it (`"2"`, `"d"`, `"2d"`,
    /// `"di"`) — empty when nothing is pending.
    #[must_use]
    pub fn pending_text(&self) -> String {
        let mut text = self.count.clone();
        if let Some(op) = self.operator {
            text.push_str(op.as_str());
        }
        if self.pending_g {
            text.push('g');
        }
        if let Some(to) = self.text_object {
            text.push(to);
        }
        if let Some(search) = self.pending_char {
            text.push(match search {
                CharSearch::ForwardFind => 'f',
                CharSearch::BackwardFind => 'F',
                CharSearch::ForwardTo => 't',
                CharSearch::BackwardTo => 'T',
            });
        }
        text
    }

    /// Whether the mode is one of the Visual selection modes.
    #[must_use]
    pub fn visual(&self) -> bool {
        self.mode == VimMode::Visual || self.mode == VimMode::VisualLine
    }

    /// Drop back to Normal mode, discarding any half-typed command.
    pub fn reset(&mut self) {
        self.mode = VimMode::Normal;
        self.anchor = None;
        self.desired_col = None;
        self.clear_pending();
    }

    fn clear_pending(&mut self) {
        self.count.clear();
        self.operator = None;
        self.pending_g = false;
        self.text_object = None;
        self.pending_char = None;
    }

    fn peek_count(&self) -> usize {
        self.count.parse::<usize>().unwrap_or(1).max(1)
    }

    fn take_count(&mut self) -> usize {
        let count = self.peek_count();
        self.count.clear();
        count
    }

    /// Clamp a position so the Normal-mode cursor rests *on* a char rather
    /// than past the last.
    fn clamp_normal(&self, buf: &VimBuffer, pos: VimPosition) -> VimPosition {
        let line = pos.line.min(buf.lines.len().saturating_sub(1));
        let text = buf.lines.get(line).map_or("", String::as_str);
        VimPosition {
            line,
            col: pos.col.min(last_char_start(text)),
        }
    }

    /// Handle one key. Returns `(consumed, commands)`: `consumed == false`
    /// means the key is not Vim's — the caller falls through to its regular
    /// handling (and, for Escape in idle Normal mode, to the app's interrupt).
    #[must_use]
    pub fn handle_key(&mut self, key: KeyEvent, buf: &VimBuffer) -> (bool, Vec<VimCommand>) {
        if key.code == KeyCode::Esc {
            return self.handle_escape(buf);
        }
        if self.mode == VimMode::Insert {
            return (false, Vec::new());
        }
        let Some(k) = vim_key_of(&key) else {
            return (false, Vec::new());
        };

        // Count prefix. `0` is the line-start motion unless it extends a
        // count already being typed.
        if k.is_ascii_digit() && (k != '0' || !self.count.is_empty()) {
            self.count.push(k);
            return (true, Vec::new());
        }

        if self.pending_g {
            self.pending_g = false;
            if k != 'g' {
                self.clear_pending();
                return (true, Vec::new());
            }
            // `gg` goes to the first line; `5gg` to line 5.
            self.desired_col = None;
            let line = self
                .take_count()
                .saturating_sub(1)
                .min(buf.lines.len().saturating_sub(1));
            return (
                true,
                self.apply_motion(
                    buf,
                    Motion {
                        to: VimPosition { line, col: 0 },
                        inclusive: false,
                        linewise: true,
                    },
                ),
            );
        }

        if self.text_object.is_some() {
            self.desired_col = None;
            return (true, self.apply_text_object(k, buf));
        }

        if let Some(search) = self.pending_char.take() {
            self.desired_col = None;
            return (true, self.apply_char_search(search, k, buf));
        }

        // `i`/`a` only introduce a text object where they cannot mean
        // "insert": after an operator, or in Visual mode. Bare `i` in Normal
        // mode still enters Insert.
        if (k == 'i' || k == 'a') && (self.operator.is_some() || self.visual()) {
            self.text_object = Some(k);
            return (true, Vec::new());
        }

        // Only consecutive `j`/`k` carry the desired column; anything else
        // re-anchors it. Counts and the `g` prefix returned above, so `2j`
        // still continues an established column.
        if k != 'j' && k != 'k' {
            self.desired_col = None;
        }

        if let Some(motion) = self.resolve_motion(k, buf) {
            return (true, self.apply_motion(buf, motion));
        }
        // `f`/`F`/`t`/`T` set `pending_char` and defer to the next key.
        if self.pending_char.is_some() {
            return (true, Vec::new());
        }

        if self.visual() {
            (true, self.handle_visual_key(k, buf))
        } else {
            (true, self.handle_normal_key(k, buf))
        }
    }

    fn handle_escape(&mut self, buf: &VimBuffer) -> (bool, Vec<VimCommand>) {
        if self.pending() {
            self.clear_pending();
            return (true, Vec::new());
        }
        if self.visual() {
            self.mode = VimMode::Normal;
            self.anchor = None;
            return (
                true,
                vec![
                    VimCommand::Mode(VimMode::Normal),
                    VimCommand::Move {
                        to: self.clamp_normal(buf, buf.cursor()),
                    },
                ],
            );
        }
        if self.mode == VimMode::Insert {
            self.mode = VimMode::Normal;
            return (
                true,
                vec![
                    VimCommand::Mode(VimMode::Normal),
                    VimCommand::Move {
                        to: VimPosition {
                            line: buf.cursor_line,
                            col: prev_char_start(buf.line(), buf.cursor_col),
                        },
                    },
                ],
            );
        }
        // Normal mode with nothing pending: leave Escape to the app.
        (false, Vec::new())
    }

    fn resolve_motion(&mut self, k: char, buf: &VimBuffer) -> Option<Motion> {
        let count = self.peek_count();
        let line = buf.line();
        let at = |col| VimPosition {
            line: buf.cursor_line,
            col,
        };
        match if k == ' ' { 'l' } else { k } {
            'h' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = prev_char_start(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: false,
                    linewise: false,
                })
            }
            'l' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = next_char_start(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: false,
                    linewise: false,
                })
            }
            'j' | 'k' => {
                let delta = count as isize;
                let delta = if k == 'j' { delta } else { -delta };
                let target = (buf.cursor_line as isize + delta)
                    .clamp(0, buf.lines.len().saturating_sub(1) as isize)
                    as usize;
                // Anchor the desired column on the first vertical move, then
                // keep reusing it; the host clamps to each line's length.
                self.desired_col.get_or_insert(buf.cursor_col);
                Some(Motion {
                    to: VimPosition {
                        line: target,
                        col: self.desired_col.unwrap_or(buf.cursor_col),
                    },
                    inclusive: false,
                    linewise: true,
                })
            }
            '0' => Some(Motion {
                to: at(0),
                inclusive: false,
                linewise: false,
            }),
            '^' => Some(Motion {
                to: at(first_non_blank(line)),
                inclusive: false,
                linewise: false,
            }),
            '$' => {
                // Sticky end-of-line, so `$j` lands on the end of each line.
                self.desired_col = Some(usize::MAX);
                Some(Motion {
                    to: at(line_len(line)),
                    inclusive: false,
                    linewise: false,
                })
            }
            'w' => {
                let mut col = buf.cursor_col;
                // Vim's `cw` quirk: standing on a non-blank, it changes to the
                // end of the word like `ce` rather than swallowing the
                // whitespace that follows it.
                if self.operator == Some(VimOperator::Change) && !char_at(line, col).is_whitespace()
                {
                    for _ in 0..count {
                        col = word_end(line, col);
                    }
                    return Some(Motion {
                        to: at(col),
                        inclusive: true,
                        linewise: false,
                    });
                }
                for _ in 0..count {
                    col = word_forward(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: false,
                    linewise: false,
                })
            }
            'b' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = move_word_left(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: false,
                    linewise: false,
                })
            }
            'e' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = word_end(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: true,
                    linewise: false,
                })
            }
            'W' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = word_forward_big(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: false,
                    linewise: false,
                })
            }
            'B' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = word_backward_big(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: false,
                    linewise: false,
                })
            }
            'E' => {
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    col = word_end_big(line, col);
                }
                Some(Motion {
                    to: at(col),
                    inclusive: true,
                    linewise: false,
                })
            }
            'G' => {
                let target = if self.count.is_empty() {
                    buf.lines.len().saturating_sub(1)
                } else {
                    count.saturating_sub(1)
                };
                Some(Motion {
                    to: VimPosition {
                        line: target.min(buf.lines.len().saturating_sub(1)),
                        col: 0,
                    },
                    inclusive: false,
                    linewise: true,
                })
            }
            'f' | 'F' | 't' | 'T' => {
                self.pending_char = Some(match k {
                    'f' => CharSearch::ForwardFind,
                    'F' => CharSearch::BackwardFind,
                    't' => CharSearch::ForwardTo,
                    _ => CharSearch::BackwardTo,
                });
                None
            }
            _ => None,
        }
    }

    /// Resolve an `f`/`F`/`t`/`T` target character into a motion application.
    fn apply_char_search(
        &mut self,
        search: CharSearch,
        ch: char,
        buf: &VimBuffer,
    ) -> Vec<VimCommand> {
        let count = self.take_count();
        let line = buf.line();
        let mut cur = buf.cursor_col;
        for _ in 0..count {
            let Some(next) = find_char(line, cur, ch, search) else {
                // No match: the pending operator (if any) is cancelled.
                self.operator = None;
                return Vec::new();
            };
            cur = next;
        }
        let inclusive = matches!(search, CharSearch::ForwardFind | CharSearch::BackwardFind);
        self.apply_motion(
            buf,
            Motion {
                to: VimPosition {
                    line: buf.cursor_line,
                    col: cur,
                },
                inclusive,
                linewise: false,
            },
        )
    }

    /// Turn a resolved motion into either a cursor move, a selection
    /// extension, or an operator range.
    fn apply_motion(&mut self, buf: &VimBuffer, motion: Motion) -> Vec<VimCommand> {
        let operator = self.operator.take();
        self.take_count();

        if operator.is_none() {
            // `$` parks on the last char in Normal mode but must still be
            // able to select the final character in Visual mode, where the
            // cursor is allowed one past it.
            let to = if self.visual() {
                VimPosition {
                    line: motion.to.line,
                    col: motion
                        .to
                        .col
                        .min(buf.lines.get(motion.to.line).map_or(0, |l| line_len(l))),
                }
            } else {
                self.clamp_normal(buf, motion.to)
            };
            return vec![VimCommand::Move { to }];
        }

        let from = buf.cursor();
        let (start, mut end) = if pos_less(motion.to, from) {
            (motion.to, from)
        } else {
            (from, motion.to)
        };
        if motion.inclusive {
            end = VimPosition {
                line: end.line,
                col: next_char_start(buf.lines.get(end.line).map_or("", String::as_str), end.col),
            };
        }
        self.operate(
            operator.unwrap_or(VimOperator::Delete),
            start,
            end,
            motion.linewise,
        )
    }

    fn operate(
        &mut self,
        operator: VimOperator,
        from: VimPosition,
        to: VimPosition,
        linewise: bool,
    ) -> Vec<VimCommand> {
        match operator {
            VimOperator::Yank => vec![
                VimCommand::Yank { from, to, linewise },
                VimCommand::Move { to: from },
            ],
            VimOperator::Delete => vec![VimCommand::Delete {
                from,
                to,
                linewise,
                insert: false,
            }],
            VimOperator::Change => {
                // `c` always lands in Insert mode. The leading move matters
                // when the range is empty (`ci"` between bare quotes): the
                // delete is a no-op, so nothing else would park the cursor.
                // `cc`/`cj` clear the lines but keep them, so a linewise
                // change stays linewise-shaped.
                self.mode = VimMode::Insert;
                vec![
                    VimCommand::Move { to: from },
                    VimCommand::Delete {
                        from,
                        to,
                        linewise,
                        insert: true,
                    },
                    VimCommand::Mode(VimMode::Insert),
                ]
            }
        }
    }

    /// `iw`, `a(`, `i"`, `ap`, … — resolved around the cursor rather than
    /// from a motion endpoint.
    fn resolve_text_object(
        &self,
        k: char,
        around: bool,
        buf: &VimBuffer,
        count: usize,
    ) -> Option<TextObjectRange> {
        let line = buf.line();
        let on_line = |range: Option<LineSpan>| {
            range.map(|(start, end)| TextObjectRange {
                from: VimPosition {
                    line: buf.cursor_line,
                    col: start,
                },
                to: VimPosition {
                    line: buf.cursor_line,
                    col: end,
                },
                linewise: false,
            })
        };
        match k {
            'w' | 'W' => on_line(word_object(line, buf.cursor_col, around, k == 'W', count)),
            '"' | '\'' | '`' => on_line(quote_object(line, buf.cursor_col, k, around)),
            '(' | ')' | 'b' => bracket_object(buf, '(', ')', around),
            '[' | ']' => bracket_object(buf, '[', ']', around),
            '{' | '}' | 'B' => bracket_object(buf, '{', '}', around),
            '<' | '>' => bracket_object(buf, '<', '>', around),
            'p' => Some(paragraph_object(buf, around)),
            _ => None,
        }
    }

    /// Consume the object key that follows a pending `i`/`a`. Under an
    /// operator the object becomes the operated range (`diw`); in Visual
    /// mode it becomes the selection instead (`viw`).
    fn apply_text_object(&mut self, k: char, buf: &VimBuffer) -> Vec<VimCommand> {
        let around = self.text_object == Some('a');
        let operator = self.operator.take();
        self.text_object = None;
        let count = self.take_count();
        let Some(range) = self.resolve_text_object(k, around, buf, count) else {
            return Vec::new();
        };
        if let Some(operator) = operator {
            return self.operate(operator, range.from, range.to, range.linewise);
        }

        self.anchor = Some(range.from);
        let mut commands = Vec::new();
        // A linewise object in charwise Visual mode promotes the selection,
        // as Vim's `vip` does.
        if range.linewise && self.mode == VimMode::Visual {
            self.mode = VimMode::VisualLine;
            commands.push(VimCommand::Mode(self.mode));
        }
        // The selection's moving end sits *on* the object's last char, not
        // one past it.
        let to = if range.to.col > 0 {
            VimPosition {
                line: range.to.line,
                col: prev_char_start(
                    buf.lines.get(range.to.line).map_or("", String::as_str),
                    range.to.col,
                ),
            }
        } else if range.to.line > range.from.line {
            let line = range.to.line - 1;
            VimPosition {
                line,
                col: last_char_start(buf.lines.get(line).map_or("", String::as_str)),
            }
        } else {
            range.from
        };
        commands.push(VimCommand::Move { to });
        commands
    }

    fn handle_normal_key(&mut self, k: char, buf: &VimBuffer) -> Vec<VimCommand> {
        let line = buf.line();
        let count = self.peek_count();
        match k {
            'g' => {
                self.pending_g = true;
                Vec::new()
            }
            'i' => {
                self.take_count();
                self.mode = VimMode::Insert;
                vec![VimCommand::Mode(VimMode::Insert)]
            }
            'a' => {
                self.take_count();
                self.mode = VimMode::Insert;
                vec![
                    VimCommand::Mode(VimMode::Insert),
                    VimCommand::Move {
                        to: VimPosition {
                            line: buf.cursor_line,
                            col: next_char_start(line, buf.cursor_col),
                        },
                    },
                ]
            }
            'I' => {
                self.take_count();
                self.mode = VimMode::Insert;
                vec![
                    VimCommand::Mode(VimMode::Insert),
                    VimCommand::Move {
                        to: VimPosition {
                            line: buf.cursor_line,
                            col: first_non_blank(line),
                        },
                    },
                ]
            }
            'A' => {
                self.take_count();
                self.mode = VimMode::Insert;
                vec![
                    VimCommand::Mode(VimMode::Insert),
                    VimCommand::Move {
                        to: VimPosition {
                            line: buf.cursor_line,
                            col: line_len(line),
                        },
                    },
                ]
            }
            'o' | 'O' => {
                self.take_count();
                self.mode = VimMode::Insert;
                vec![
                    VimCommand::OpenLine { below: k == 'o' },
                    VimCommand::Mode(VimMode::Insert),
                ]
            }
            'v' | 'V' => {
                self.take_count();
                self.mode = if k == 'v' {
                    VimMode::Visual
                } else {
                    VimMode::VisualLine
                };
                self.anchor = Some(buf.cursor());
                vec![VimCommand::Mode(self.mode)]
            }
            'x' => {
                self.take_count();
                let len = line_len(line);
                let mut col = buf.cursor_col;
                for _ in 0..count {
                    if col >= len {
                        break;
                    }
                    col = next_char_start(line, col);
                }
                if col == buf.cursor_col {
                    return Vec::new();
                }
                vec![VimCommand::Delete {
                    from: buf.cursor(),
                    to: VimPosition {
                        line: buf.cursor_line,
                        col,
                    },
                    linewise: false,
                    insert: false,
                }]
            }
            'D' | 'C' => {
                // Like Vim, `D`/`C` take a count: `2D` deletes to the end of
                // the next line, not just this one (`:h D`).
                let span = self.take_count();
                let last = (buf.cursor_line + span - 1).min(buf.lines.len().saturating_sub(1));
                self.operate(
                    if k == 'C' {
                        VimOperator::Change
                    } else {
                        VimOperator::Delete
                    },
                    buf.cursor(),
                    VimPosition {
                        line: last,
                        col: buf.lines.get(last).map_or(0, |l| line_len(l)),
                    },
                    false,
                )
            }
            'd' | 'y' | 'c' => {
                // A doubled operator (`dd`, `yy`, `cc`) is linewise over `count` lines.
                let operator = if k == 'd' {
                    VimOperator::Delete
                } else if k == 'y' {
                    VimOperator::Yank
                } else {
                    VimOperator::Change
                };
                if self.operator == Some(operator) {
                    let span = self.take_count();
                    self.operator = None;
                    let last = (buf.cursor_line + span - 1).min(buf.lines.len().saturating_sub(1));
                    return self.operate(
                        operator,
                        VimPosition {
                            line: buf.cursor_line,
                            col: 0,
                        },
                        VimPosition {
                            line: last,
                            col: buf.lines.get(last).map_or(0, |l| line_len(l)),
                        },
                        true,
                    );
                }
                self.operator = Some(operator);
                Vec::new()
            }
            'p' | 'P' => vec![VimCommand::Paste {
                after: k == 'p',
                count: self.take_count(),
            }],
            'u' => {
                self.take_count();
                vec![VimCommand::Undo]
            }
            // Normal mode swallows unknown printable keys rather than typing
            // them into the buffer.
            _ => {
                self.clear_pending();
                Vec::new()
            }
        }
    }

    fn handle_visual_key(&mut self, k: char, buf: &VimBuffer) -> Vec<VimCommand> {
        let anchor = self.anchor.unwrap_or_else(|| buf.cursor());
        let linewise = self.mode == VimMode::VisualLine;
        match k {
            'v' | 'V' => {
                let next = if k == 'v' {
                    VimMode::Visual
                } else {
                    VimMode::VisualLine
                };
                if self.mode == next {
                    self.mode = VimMode::Normal;
                    self.anchor = None;
                    return vec![
                        VimCommand::Mode(VimMode::Normal),
                        VimCommand::Move {
                            to: self.clamp_normal(buf, buf.cursor()),
                        },
                    ];
                }
                self.mode = next;
                vec![VimCommand::Mode(next)]
            }
            'o' => {
                self.anchor = Some(buf.cursor());
                vec![VimCommand::Move { to: anchor }]
            }
            'y' | 'd' | 'x' | 'c' | 's' => {
                let operator = match k {
                    'y' => VimOperator::Yank,
                    'c' | 's' => VimOperator::Change,
                    _ => VimOperator::Delete,
                };
                let (from, to) = visual_range(buf, anchor, linewise);
                self.clear_pending();
                self.anchor = None;
                // `operate` switches to Insert itself for `c`; everything
                // else drops back to Normal.
                if operator != VimOperator::Change {
                    self.mode = VimMode::Normal;
                }
                let mut commands = self.operate(operator, from, to, linewise);
                if operator != VimOperator::Change {
                    commands.push(VimCommand::Mode(VimMode::Normal));
                }
                commands
            }
            _ => {
                self.clear_pending();
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn esc() -> KeyEvent {
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)
    }

    /// Run a string of plain keys, returning the last key's commands.
    fn send(state: &mut VimState, b: &VimBuffer, keys: &str) -> Vec<VimCommand> {
        let mut commands = Vec::new();
        for c in keys.chars() {
            let (_, cmds) = state.handle_key(key(c), b);
            commands = cmds;
        }
        commands
    }

    fn pos(line: usize, col: usize) -> VimPosition {
        VimPosition { line, col }
    }

    // -- motions -----------------------------------------------------------

    #[test]
    fn h_l_0_dollar_motions_clamp_to_last_char() {
        let b = VimBuffer::from_flat("abc", 1);
        let mut s = VimState::new();
        let (_, cmds) = s.handle_key(key('l'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 2) }]);
        let (_, cmds) = s.handle_key(key('h'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 0) }]);
        let (_, cmds) = s.handle_key(key('$'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 2) }]);
        let (_, cmds) = s.handle_key(key('0'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 0) }]);
    }

    #[test]
    fn word_motions_w_b_e_and_big_variants() {
        let b = VimBuffer::from_flat("foo bar-baz qux", 0);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "w"),
            vec![VimCommand::Move { to: pos(0, 4) }]
        );
        // `W` folds punctuation into the word, so the WORD ends at the gap.
        let b = VimBuffer::from_flat("foo bar-baz qux", 4);
        assert_eq!(
            send(&mut s, &b, "W"),
            vec![VimCommand::Move { to: pos(0, 12) }]
        );
        assert_eq!(
            send(&mut s, &b, "E"),
            vec![VimCommand::Move { to: pos(0, 10) }]
        );
        assert_eq!(
            send(&mut s, &b, "B"),
            vec![VimCommand::Move { to: pos(0, 0) }]
        );
        let b = VimBuffer::from_flat("foo bar", 6);
        assert_eq!(
            send(&mut s, &b, "e"),
            vec![VimCommand::Move { to: pos(0, 6) }]
        );
        assert_eq!(
            send(&mut s, &b, "b"),
            vec![VimCommand::Move { to: pos(0, 4) }]
        );
    }

    #[test]
    fn gg_g_and_counts() {
        let b = VimBuffer::from_flat("one\ntwo\nthree", 0);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "G"),
            vec![VimCommand::Move { to: pos(2, 0) }]
        );
        assert_eq!(
            send(&mut s, &b, "gg"),
            vec![VimCommand::Move { to: pos(0, 0) }]
        );
        assert_eq!(
            send(&mut s, &b, "2G"),
            vec![VimCommand::Move { to: pos(1, 0) }]
        );
        let b = VimBuffer::from_flat("aaa bbb ccc", 0);
        assert_eq!(
            send(&mut s, &b, "2w"),
            vec![VimCommand::Move { to: pos(0, 8) }]
        );
    }

    #[test]
    fn f_f_t_t_char_search() {
        let b = VimBuffer::from_flat("a.b.c", 0);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "f."),
            vec![VimCommand::Move { to: pos(0, 1) }]
        );
        assert_eq!(
            send(&mut s, &b, "2f."),
            vec![VimCommand::Move { to: pos(0, 3) }]
        );
        let b = VimBuffer::from_flat("a.b.c", 4);
        assert_eq!(
            send(&mut s, &b, "F."),
            vec![VimCommand::Move { to: pos(0, 3) }]
        );
        let b = VimBuffer::from_flat("a.b.c", 0);
        assert_eq!(
            send(&mut s, &b, "t."),
            vec![VimCommand::Move { to: pos(0, 0) }]
        );
        let b = VimBuffer::from_flat("xa;b", 0);
        assert_eq!(
            send(&mut s, &b, "t;"),
            vec![VimCommand::Move { to: pos(0, 1) }]
        );
        // `df,` on "a,b,c" at col 2: lands on the comma at 3, inclusive.
        let b = VimBuffer::from_flat("a,b,c", 2);
        let mut s = VimState::new();
        send(&mut s, &b, "df");
        let (_, cmds) = s.handle_key(key(','), &b);
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 2),
                to: pos(0, 4),
                linewise: false,
                insert: false,
            }]
        );
    }

    #[test]
    fn f_without_match_cancels_operator() {
        let b = VimBuffer::from_flat("abc", 0);
        let mut s = VimState::new();
        send(&mut s, &b, "df");
        assert!(s.pending());
        let (_, cmds) = s.handle_key(key('z'), &b);
        assert!(cmds.is_empty());
        assert!(!s.pending());
    }

    #[test]
    fn desired_column_survives_short_lines() {
        let b = VimBuffer::from_flat("ab\nx\nabcd", 1);
        let mut s = VimState::new();
        let (_, cmds) = s.handle_key(key('j'), &b);
        // Line 1 has one char; the Normal-mode cursor clamps onto it…
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(1, 0) }]);
        // …but `k` returns to the anchored column 1.
        let (_, cmds) = s.handle_key(key('k'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 1) }]);
    }

    #[test]
    fn dollar_j_sticks_to_line_ends() {
        let b = VimBuffer::from_flat("abc\nx\npq", 0);
        let mut s = VimState::new();
        let (_, cmds) = s.handle_key(key('$'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 2) }]);
        let (_, cmds) = s.handle_key(key('j'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(1, 0) }]);
    }

    // -- operators ----------------------------------------------------------

    #[test]
    fn dw_de_and_cw_quirk() {
        // Two spaces after "foo": `dw` swallows the gap, `de`/`cw` don't.
        let b = VimBuffer::from_flat("foo  bar", 0);
        let mut s = VimState::new();
        send(&mut s, &b, "d");
        let (_, cmds) = s.handle_key(key('w'), &b);
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 0),
                to: pos(0, 5),
                linewise: false,
                insert: false,
            }]
        );
        let mut s = VimState::new();
        send(&mut s, &b, "c");
        let (_, cmds) = s.handle_key(key('w'), &b);
        assert_eq!(
            cmds,
            vec![
                VimCommand::Move { to: pos(0, 0) },
                VimCommand::Delete {
                    from: pos(0, 0),
                    to: pos(0, 3),
                    linewise: false,
                    insert: true,
                },
                VimCommand::Mode(VimMode::Insert),
            ]
        );
        assert_eq!(s.mode, VimMode::Insert);
    }

    #[test]
    fn doubled_operators_are_linewise() {
        let b = VimBuffer::from_flat("one\ntwo\nthree", 4);
        let mut s = VimState::new();
        send(&mut s, &b, "d");
        let (_, cmds) = s.handle_key(key('d'), &b);
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(1, 0),
                to: pos(1, 3),
                linewise: true,
                insert: false,
            }]
        );
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "2yy"),
            vec![
                VimCommand::Yank {
                    from: pos(1, 0),
                    to: pos(2, 5),
                    linewise: true,
                },
                VimCommand::Move { to: pos(1, 0) },
            ]
        );
    }

    #[test]
    fn x_d_and_c_counts() {
        let b = VimBuffer::from_flat("abcdef", 1);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "2x"),
            vec![VimCommand::Delete {
                from: pos(0, 1),
                to: pos(0, 3),
                linewise: false,
                insert: false,
            }]
        );
        let b = VimBuffer::from_flat("ab\ncd", 1);
        assert_eq!(
            send(&mut s, &b, "2D"),
            vec![VimCommand::Delete {
                from: pos(0, 1),
                to: pos(1, 2),
                linewise: false,
                insert: false,
            }]
        );
        let b = VimBuffer::from_flat("ab\ncd", 1);
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "C");
        assert_eq!(s.mode, VimMode::Insert);
        assert!(matches!(
            cmds.as_slice(),
            [
                VimCommand::Move { .. },
                VimCommand::Delete { insert: true, .. },
                VimCommand::Mode(VimMode::Insert)
            ]
        ));
    }

    #[test]
    fn paste_and_undo_commands() {
        let b = VimBuffer::from_flat("abc", 0);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "p"),
            vec![VimCommand::Paste {
                after: true,
                count: 1
            }]
        );
        assert_eq!(
            send(&mut s, &b, "3P"),
            vec![VimCommand::Paste {
                after: false,
                count: 3
            }]
        );
        assert_eq!(send(&mut s, &b, "u"), vec![VimCommand::Undo]);
    }

    #[test]
    fn open_line_and_entry_commands() {
        let b = VimBuffer::from_flat("ab", 0);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "o"),
            vec![
                VimCommand::OpenLine { below: true },
                VimCommand::Mode(VimMode::Insert),
            ]
        );
        assert_eq!(s.mode, VimMode::Insert);
        s.reset();
        assert_eq!(send(&mut s, &b, "A")[0], VimCommand::Mode(VimMode::Insert));
        s.reset();
        let cmds = send(&mut s, &b, "I");
        assert_eq!(
            cmds,
            vec![
                VimCommand::Mode(VimMode::Insert),
                VimCommand::Move { to: pos(0, 0) },
            ]
        );
    }

    // -- text objects ---------------------------------------------------------

    #[test]
    fn word_text_objects() {
        let b = VimBuffer::from_flat("foo bar", 5);
        let mut s = VimState::new();
        send(&mut s, &b, "d");
        let (_, cmds) = s.handle_key(key('i'), &b);
        assert!(cmds.is_empty());
        let (_, cmds) = s.handle_key(key('w'), &b);
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 4),
                to: pos(0, 7),
                linewise: false,
                insert: false,
            }]
        );
        // `aw` on the last word takes the leading whitespace.
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "daw");
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 3),
                to: pos(0, 7),
                linewise: false,
                insert: false,
            }]
        );
    }

    #[test]
    fn quote_text_objects() {
        let b = VimBuffer::from_flat("say \"hi there\" ok", 6);
        let mut s = VimState::new();
        send(&mut s, &b, "di");
        let (_, cmds) = s.handle_key(key('"'), &b);
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 5),
                to: pos(0, 13),
                linewise: false,
                insert: false,
            }]
        );
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "da\"");
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 4),
                to: pos(0, 15),
                linewise: false,
                insert: false,
            }]
        );
    }

    #[test]
    fn bracket_text_object_spans_lines() {
        let b = VimBuffer::from_flat("f(a,\n  b(c)\n)", 3);
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "di(");
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 2),
                to: pos(2, 0),
                linewise: false,
                insert: false,
            }]
        );
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "da{");
        // No braces in the buffer: nothing happens.
        assert!(cmds.is_empty());
    }

    #[test]
    fn paragraph_text_object() {
        let b = VimBuffer::from_flat("one\ntwo\n\nfour", 0);
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "dip");
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 0),
                to: pos(1, 3),
                linewise: true,
                insert: false,
            }]
        );
        let mut s = VimState::new();
        let cmds = send(&mut s, &b, "dap");
        assert_eq!(
            cmds,
            vec![VimCommand::Delete {
                from: pos(0, 0),
                to: pos(2, 0),
                linewise: true,
                insert: false,
            }]
        );
    }

    // -- visual ----------------------------------------------------------------

    #[test]
    fn visual_select_and_delete() {
        let b = VimBuffer::from_flat("abcdef", 1);
        let mut s = VimState::new();
        assert_eq!(
            send(&mut s, &b, "v"),
            vec![VimCommand::Mode(VimMode::Visual)]
        );
        let (_, cmds) = s.handle_key(key('l'), &b);
        // Visual mode may sit one past the last selected char.
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 2) }]);
        // The machine reads buffer snapshots: after `l` the host cursor is
        // at col 2, so the delete covers the anchor..cursor span.
        let b = VimBuffer::from_flat("abcdef", 3);
        let (_, cmds) = s.handle_key(key('d'), &b);
        assert_eq!(
            cmds,
            vec![
                VimCommand::Delete {
                    from: pos(0, 1),
                    to: pos(0, 4),
                    linewise: false,
                    insert: false,
                },
                VimCommand::Mode(VimMode::Normal),
            ]
        );
        assert_eq!(s.mode, VimMode::Normal);
    }

    #[test]
    fn visual_line_yank() {
        let b = VimBuffer::from_flat("one\ntwo", 0);
        let mut s = VimState::new();
        send(&mut s, &b, "V");
        let (_, cmds) = s.handle_key(key('y'), &b);
        assert_eq!(
            cmds,
            vec![
                VimCommand::Yank {
                    from: pos(0, 0),
                    to: pos(0, 3),
                    linewise: true,
                },
                VimCommand::Move { to: pos(0, 0) },
                VimCommand::Mode(VimMode::Normal),
            ]
        );
    }

    #[test]
    fn visual_text_object_promotes_selection() {
        let b = VimBuffer::from_flat("one\ntwo", 1);
        let mut s = VimState::new();
        send(&mut s, &b, "v");
        let (_, cmds) = s.handle_key(key('i'), &b);
        assert!(cmds.is_empty());
        let (_, cmds) = s.handle_key(key('p'), &b);
        assert_eq!(
            cmds,
            vec![
                VimCommand::Mode(VimMode::VisualLine),
                VimCommand::Move { to: pos(1, 2) },
            ]
        );
        assert_eq!(s.mode, VimMode::VisualLine);
    }

    #[test]
    fn visual_o_swaps_ends() {
        let b = VimBuffer::from_flat("abcdef", 1);
        let mut s = VimState::new();
        send(&mut s, &b, "v");
        // Two `l`s moved the host cursor to col 3; `o` swaps the ends.
        let b = VimBuffer::from_flat("abcdef", 3);
        let (_, cmds) = s.handle_key(key('o'), &b);
        assert_eq!(cmds, vec![VimCommand::Move { to: pos(0, 1) }]);
        assert_eq!(s.anchor, Some(pos(0, 3)));
    }

    // -- escape ----------------------------------------------------------------

    #[test]
    fn escape_cancels_pending_and_leaves_visual() {
        let b = VimBuffer::from_flat("abc", 0);
        let mut s = VimState::new();
        send(&mut s, &b, "d");
        assert!(s.pending());
        assert_eq!(s.pending_text(), "d");
        let (consumed, cmds) = s.handle_key(esc(), &b);
        assert!(consumed);
        assert!(cmds.is_empty());
        assert!(!s.pending());
        // Idle Normal mode: Escape is the app's interrupt, not ours.
        let (consumed, _) = s.handle_key(esc(), &b);
        assert!(!consumed);
        // Visual: Escape drops back to Normal and clamps the cursor. (`vl`
        // moved the host cursor, but the machine reads the static snapshot
        // here, so the clamp is against col 0.)
        send(&mut s, &b, "vl");
        let (consumed, cmds) = s.handle_key(esc(), &b);
        assert!(consumed);
        assert_eq!(
            cmds,
            vec![
                VimCommand::Mode(VimMode::Normal),
                VimCommand::Move { to: pos(0, 0) },
            ]
        );
    }

    #[test]
    fn escape_leaves_insert_on_last_char() {
        let b = VimBuffer::from_flat("abc", 3);
        let mut s = VimState::new();
        s.mode = VimMode::Insert;
        let (consumed, cmds) = s.handle_key(esc(), &b);
        assert!(consumed);
        assert_eq!(
            cmds,
            vec![
                VimCommand::Mode(VimMode::Normal),
                VimCommand::Move { to: pos(0, 2) },
            ]
        );
        assert_eq!(s.mode, VimMode::Normal);
    }

    #[test]
    fn insert_mode_ignores_keys() {
        let b = VimBuffer::from_flat("abc", 0);
        let mut s = VimState::new();
        s.mode = VimMode::Insert;
        let (consumed, cmds) = s.handle_key(key('x'), &b);
        assert!(!consumed);
        assert!(cmds.is_empty());
    }

    #[test]
    fn count_then_escape_only_clears_count() {
        let b = VimBuffer::from_flat("abc", 0);
        let mut s = VimState::new();
        send(&mut s, &b, "12");
        assert_eq!(s.pending_text(), "12");
        let (consumed, _) = s.handle_key(esc(), &b);
        assert!(consumed);
        assert!(!s.pending());
        assert_eq!(s.mode, VimMode::Normal);
    }
}
