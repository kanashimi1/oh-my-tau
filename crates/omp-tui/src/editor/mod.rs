//! The prompt editor — the port of `pi-tui/src/editor-component.ts` +
//! `pi-tui/src/components/editor.ts` + the composer parts of
//! `pi-tui/src/prompt/custom-editor.ts`.
//!
//! Submodules:
//! - [`buffer`] — line-based text model with a grapheme-aware cursor
//! - [`dispatch`] — the key pipeline (motions, kills, yank, history)
//! - [`kill_ring`] — `kill-ring.ts` (60 entries, accumulate, yank-pop)
//! - [`history`] — the 100-entry ring + persistent store seam
//! - [`undo`] — snapshot undo stack
//! - [`paste`] — bracketed-paste sanitize + large-paste markers
//! - [`attachments`] — `[Image #N]`/`[Video #N]`/`[Text #N]` chips
//! - [`layout`] — wide-char widths, soft wrap, atomic-marker scanning
//! - [`render`] — band layout, `╰─ ` gutter, right hint, bash `!` gutter color
//! - [`submit`] — submit classification (follow-up, queue, slash, continue)
//! - [`vim`] — modal editing behind the same API (separate implementer)
//!
//! The editor never talks to the runtime directly; it returns
//! [`EditorOutcome`] and the app turns submissions into [`Action`]s.

mod attachments;
mod buffer;
mod dispatch;
mod history;
mod kill_ring;
pub(crate) mod layout;
pub mod paste;
mod render;
mod submit;
mod undo;

pub mod vim;

pub use attachments::{Attachment, AttachmentKind, Attachments};
pub use buffer::{Cursor, TextBuffer};
pub use history::{HistoryNav, HistoryStore, JsonlHistory, MemoryHistory};
pub use paste::PasteStore;
pub use submit::{SubmitKind, SubmitRequest};

use crate::keys::KeyEvent;
use dispatch::LastAction;
use render::ScrollOffset;

/// Char index → byte offset within a line (`line.len()` when past the end).
fn byte_col(line: &str, char_col: usize) -> usize {
    line.char_indices()
        .nth(char_col)
        .map_or(line.len(), |(byte, _)| byte)
}

/// Char-index slice of a line (end-exclusive; `usize::MAX` means the rest).
fn char_slice(line: &str, start: usize, end: usize) -> String {
    line.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// What the editor wants to happen after handling a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorOutcome {
    /// Nothing special; buffer/cursor changed (or key ignored).
    Edited,
    /// A submit key was pressed. Call [`Editor::take`] (plain text) or
    /// [`Editor::take_submit`] (routed request) to consume the buffer.
    Submit,
    /// Ctrl+C pressed.
    Interrupt,
}

/// Multi-line prompt editor: text model + kill ring + history + undo +
/// paste markers + attachment chips, behind the pi-tui `Editor` API.
#[derive(Debug, Clone)]
pub struct Editor {
    pub(crate) buffer: TextBuffer,
    kill_ring: kill_ring::KillRing,
    history: history::MemoryHistory,
    history_nav: history::HistoryNav,
    undo: undo::UndoStack,
    pastes: paste::PasteStore,
    attachments: attachments::Attachments,
    pub(crate) scroll_offset: ScrollOffset,
    pub(crate) page_scroll_step: std::cell::Cell<usize>,
    pub(crate) max_height: u16,
    /// Right-aligned composer hint `(key, label)` shown while the draft is empty.
    pub(crate) hint: Option<(String, String)>,
    /// Active composer shape (`composer.shape`).
    pub(crate) shape: crate::composer::ComposerShape,
    /// Status content for the shape's top chrome (set by the app per frame).
    pub(crate) top_border: Option<crate::composer::TopBorder>,
    pub(crate) disable_submit: bool,
    submit_kind: submit::SubmitKind,
    last_action: LastAction,
    yank_span: Option<(usize, usize, usize)>,
    vim: Option<vim::VimState>,
    /// Ghost-text word completion (`spelling.autocomplete`); `None` = off.
    word: Option<crate::word_completion::WordCompletion>,
    /// Magic-keyword shimmer phase for the next render (`0` = static).
    pub(crate) magic_phase: f64,
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Editor {
    /// An empty editor with one empty line.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: TextBuffer::new(),
            kill_ring: kill_ring::KillRing::new(),
            history: history::MemoryHistory::new(),
            history_nav: history::HistoryNav::default(),
            undo: undo::UndoStack::new(),
            pastes: paste::PasteStore::new(),
            attachments: attachments::Attachments::new(),
            scroll_offset: ScrollOffset::new(0),
            page_scroll_step: std::cell::Cell::new(1),
            max_height: 10,
            hint: None,
            shape: crate::composer::ComposerShape::default(),
            top_border: None,
            disable_submit: false,
            submit_kind: submit::SubmitKind::Normal,
            last_action: LastAction::Other,
            yank_span: None,
            vim: None,
            word: None,
            magic_phase: 0.0,
        }
    }

    /// Set the magic-keyword gradient phase the next render paints with
    /// (the app passes a clock-derived value while a keyword shimmers).
    pub fn set_magic_phase(&mut self, phase: f64) {
        self.magic_phase = phase;
    }

    // -- text model passthroughs (pi-tui EditorComponent API) --------------

    /// Current buffer text (markers *not* expanded).
    #[must_use]
    pub fn text(&self) -> String {
        self.buffer.text()
    }

    /// Whether the buffer holds no characters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Cursor as a byte offset in the whole-buffer text.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.buffer.flat_cursor()
    }

    /// The structured cursor (line + byte column).
    #[must_use]
    pub fn structured_cursor(&self) -> Cursor {
        self.buffer.cursor()
    }

    /// Borrow the text model (autocomplete providers read this).
    #[must_use]
    pub fn text_buffer(&self) -> &TextBuffer {
        &self.buffer
    }

    /// Replace the whole buffer and move the cursor to the end.
    pub fn set_text(&mut self, text: &str) {
        self.record_undo();
        self.buffer.set_text(text);
        self.note_action(LastAction::Other);
    }

    /// Apply an autocomplete pick: replace the buffer with `lines` and
    /// place the cursor at (`line`, byte `col`). Undoable.
    pub fn apply_completion(&mut self, lines: Vec<String>, line: usize, col: usize) {
        self.record_undo();
        self.replace_lines(lines, Cursor { line, col });
    }

    /// Insert one character at the cursor.
    pub fn insert_char(&mut self, c: char) {
        if self.history_nav.browsing() {
            // Editing a recalled history entry exits browsing and restores
            // the draft (`#exitHistoryForEditing`).
            let draft = self.history_nav.exit_draft();
            self.buffer.set_text(&draft);
        }
        // The space a Tab word accept added is provisional until the next
        // keystroke: a typed space is swallowed, closing punctuation or a
        // hyphen replaces it, anything else keeps it.
        let provisional = self.has_provisional_space();
        if provisional && c == ' ' {
            self.note_action(LastAction::Other);
            return;
        }
        self.record_undo();
        self.report_typed_past_word_completion(c);
        if provisional && matches!(c, '-' | '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}') {
            self.buffer.delete_backward();
        }
        self.buffer.insert_char(c);
        self.note_action(LastAction::Other);
    }

    // -- ghost-text word completion (TS `WordCompletionProvider`) -------------

    /// Install (or clear) the word-completion engine host behind the ghost text.
    pub fn set_word_completion(
        &mut self,
        host: Option<std::sync::Arc<crate::word_completion::host::WordHost>>,
    ) {
        self.word = host.map(crate::word_completion::WordCompletion::new);
    }

    /// Whether ghost-text word completion is on.
    #[must_use]
    pub fn word_completion_enabled(&self) -> bool {
        self.word.is_some()
    }

    /// Learn a submitted prompt (no-op while completion is off).
    pub fn observe_prompt(&self, prompt: &str) {
        if let Some(word) = &self.word {
            word.observe(prompt);
        }
    }

    /// Flush the engine's learned state (no-op while completion is off).
    pub fn persist_word_completion(&self) {
        if let Some(word) = &self.word {
            word.host().persist();
        }
    }

    /// The ghost suffix painted after the cursor: only at the end of a line
    /// (the only place ghost text renders), never while browsing history.
    #[must_use]
    pub fn word_ghost(&self) -> Option<String> {
        let word = self.word.as_ref()?;
        if self.history_nav.browsing() {
            return None;
        }
        let cursor = self.buffer.cursor();
        let lines = self.buffer.lines();
        if cursor.col < lines.get(cursor.line)?.len() {
            return None;
        }
        word.ghost(lines, cursor.line, cursor.col)
    }

    /// Insert the shown ghost completion at the cursor, if any. Tab passes
    /// `space: true` for a trailing space that stays provisional until the
    /// next keystroke; Right passes `false` so a suffix can follow the word.
    pub fn accept_word_completion(&mut self, space: bool) -> bool {
        let Some(ghost) = self.word_ghost() else {
            return false;
        };
        let cursor = self.buffer.cursor();
        if let Some(word) = &self.word {
            word.feedback(self.buffer.lines(), cursor.line, cursor.col, &ghost, true);
        }
        self.record_undo();
        self.buffer.insert_str(&ghost);
        if space {
            self.buffer.insert_char(' ');
        }
        self.note_action(if space {
            LastAction::AcceptWord
        } else {
            LastAction::Other
        });
        true
    }

    /// Typing `typed` where it diverges from the shown ghost rejects it.
    fn report_typed_past_word_completion(&self, typed: char) {
        let Some(ghost) = self.word_ghost() else {
            return;
        };
        let first = ghost
            .chars()
            .next()
            .into_iter()
            .flat_map(char::to_lowercase);
        if first.eq(typed.to_lowercase()) {
            return;
        }
        if let Some(word) = &self.word {
            let cursor = self.buffer.cursor();
            word.feedback(self.buffer.lines(), cursor.line, cursor.col, &ghost, false);
        }
    }

    fn has_provisional_space(&self) -> bool {
        if self.last_action != LastAction::AcceptWord {
            return false;
        }
        let cursor = self.buffer.cursor();
        self.buffer.lines()[cursor.line][..cursor.col].ends_with(' ')
    }

    /// A newline drops the provisional space a Tab accept added.
    pub(crate) fn drop_provisional_space(&mut self) {
        if self.has_provisional_space() {
            self.buffer.delete_backward();
        }
    }

    /// Insert a string at the cursor (may contain newlines).
    pub fn insert_str_at_cursor(&mut self, s: &str) {
        self.record_undo();
        self.buffer.insert_str(s);
        self.note_action(LastAction::Other);
    }

    /// Insert a newline at the cursor.
    pub fn insert_newline(&mut self) {
        self.newline();
    }

    /// Delete the char before the cursor (marker-widened via keys;
    /// plain here for API compat).
    pub fn backspace(&mut self) {
        self.record_undo();
        self.buffer.delete_backward();
        self.note_action(LastAction::Other);
    }

    /// Move the cursor one grapheme left.
    pub fn move_left(&mut self) {
        self.buffer.move_left();
    }

    /// Move the cursor one grapheme right.
    pub fn move_right(&mut self) {
        self.buffer.move_right();
    }

    /// Move to the start of the current line.
    pub fn move_home(&mut self) {
        self.buffer.move_line_start();
    }

    /// Move to the start of the current line (`Home` / `ctrl+a`).
    pub fn move_line_start(&mut self) {
        self.buffer.move_line_start();
    }

    /// Move to the end of the current line (`End` / `ctrl+e`).
    pub fn move_line_end(&mut self) {
        self.buffer.move_line_end();
    }

    /// Move to the end of the current line.
    pub fn move_end(&mut self) {
        self.buffer.move_line_end();
    }

    /// Move to the previous word start.
    pub fn move_word_left(&mut self) {
        self.buffer.move_word_left();
    }

    /// Move to the next word end.
    pub fn move_word_right(&mut self) {
        self.buffer.move_word_right();
    }

    // -- configuration ------------------------------------------------------

    /// Set (or clear) the right-aligned composer hint shown on an empty
    /// buffer (TS `editor.placeholder`, `composer-hints.ts`).
    pub fn set_hint(&mut self, hint: Option<(String, String)>) {
        self.hint = hint;
    }

    /// Switch the composer shape (TS `editor.setBorderStyle`).
    pub fn set_shape(&mut self, shape: crate::composer::ComposerShape) {
        self.shape = shape;
    }

    /// The active composer shape.
    #[must_use]
    pub fn shape(&self) -> crate::composer::ComposerShape {
        self.shape
    }

    /// Status content drawn into the shape's top chrome (TS
    /// `setTopBorderProvider` result for this frame).
    pub fn set_top_border(&mut self, top: Option<crate::composer::TopBorder>) {
        self.top_border = top;
    }

    /// Cap the editor height (TS `setMaxHeight`).
    pub fn set_max_height(&mut self, height: u16) {
        self.max_height = height.max(3);
    }

    /// Temporarily block submits (used while an async paste materializes).
    pub fn set_disable_submit(&mut self, disable: bool) {
        self.disable_submit = disable;
    }

    // -- history ------------------------------------------------------------

    /// Borrow the prompt history ring.
    #[must_use]
    pub fn history(&self) -> &history::MemoryHistory {
        &self.history
    }

    /// Mutably borrow the history ring (app seeds persistence on startup).
    pub fn history_mut(&mut self) -> &mut history::MemoryHistory {
        &mut self.history
    }

    /// Up: browse older history entries (TS `:2160`).
    pub(crate) fn history_up(&mut self) {
        let current = self.buffer.text();
        let next = self.history_nav.up(&self.history, &current);
        if next != current || self.history_nav.browsing() {
            self.buffer.set_text(&next);
        }
    }

    /// Down: browse newer entries; at the newest, restore the draft.
    pub(crate) fn history_down(&mut self) {
        let current = self.buffer.text();
        let next = self.history_nav.down(&self.history, &current);
        if next != current || !self.history_nav.browsing() {
            self.buffer.set_text(&next);
        }
    }

    // -- undo ---------------------------------------------------------------

    pub(crate) fn record_undo(&mut self) {
        self.undo.record(&self.buffer);
    }

    pub(crate) fn apply_undo(&mut self) {
        if let Some(cursor) = self.undo.undo(&mut self.buffer) {
            self.buffer.set_cursor(cursor);
            self.note_action(LastAction::Other);
        }
    }

    /// Whether the undo stack has entries.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    // -- kill ring ----------------------------------------------------------

    pub(crate) fn push_kill(&mut self, text: &str, backward: bool) {
        let accumulate = matches!(
            self.last_action,
            LastAction::Kill { backward: prev } if prev == backward
        );
        self.kill_ring.push(text, backward, accumulate);
        self.note_action(LastAction::Kill { backward });
    }

    pub(crate) fn kill_ring_yank(&mut self) -> Option<&str> {
        self.kill_ring.end_pop();
        self.kill_ring.yank()
    }

    pub(crate) fn kill_ring_rotate(&mut self) -> Option<&str> {
        if !self.kill_ring.popping() {
            self.kill_ring.start_pop();
        }
        self.kill_ring.rotate()
    }

    pub(crate) fn note_action(&mut self, action: LastAction) {
        if action != LastAction::Yank {
            self.yank_span = None;
        }
        self.last_action = action;
    }

    pub(crate) fn last_action(&self) -> LastAction {
        self.last_action
    }

    /// Insert a yanked string, remembering the span for yank-pop.
    pub(crate) fn insert_yanked(&mut self, text: &str) {
        let line = self.buffer.cursor().line;
        let col = self.buffer.cursor().col;
        self.buffer.insert_str(text);
        self.yank_span = Some((line, col, self.buffer.flat_cursor()));
        self.kill_ring.start_pop();
    }

    /// Replace the last yanked span with the next kill-ring entry.
    pub(crate) fn replace_yanked(&mut self, text: &str) {
        let Some((line, start, end)) = self.yank_span else {
            return;
        };
        // Flat offsets -> same-line replacement (yanked text is one line
        // in the MVP; multi-line yanks fall back to insert).
        let line_text = &self.buffer.lines()[line];
        let line_start: usize = self.buffer.lines()[..line]
            .iter()
            .map(|l| l.len() + 1)
            .sum();
        if end <= line_start + line_text.len() {
            self.buffer
                .replace_range(line, start - line_start, end - line_start, text);
        } else {
            self.buffer.insert_str(text);
        }
    }
    // -- vim modal editing (TS editor.ts:901-940) ----------------------------

    /// Enable Vim-style modal editing. Toggling always drops back to Insert
    /// mode so the editor is immediately editable (`setVimMode` :903).
    pub fn set_vim_mode(&mut self, enabled: bool) {
        if enabled == self.vim.is_some() {
            return;
        }
        if enabled {
            let mut state = vim::VimState::new();
            state.mode = vim::VimMode::Insert;
            self.vim = Some(state);
        } else {
            self.vim = None;
        }
    }

    /// Whether Vim mode is active (`vimEnabled` :916).
    #[must_use]
    pub fn vim_enabled(&self) -> bool {
        self.vim.is_some()
    }

    /// Current modal state; `None` when Vim mode is off (the host renders no
    /// mode badge then) (`vimMode` :911).
    #[must_use]
    pub fn vim_mode(&self) -> Option<vim::VimMode> {
        self.vim.as_ref().map(|v| v.mode)
    }

    /// Half-typed Vim command (`"2d"`, `"g"`), or `""` when nothing is
    /// pending (`vimPending` :921).
    #[must_use]
    pub fn vim_pending(&self) -> String {
        self.vim
            .as_ref()
            .map_or(String::new(), vim::VimState::pending_text)
    }

    /// Hosts bind Escape to interrupt/clear; in Vim mode it first has to
    /// mean "leave Insert mode" or "cancel the half-typed command"
    /// (`vimConsumesEscape` :937).
    #[must_use]
    pub fn vim_consumes_escape(&self) -> bool {
        match &self.vim {
            None => false,
            Some(v) => v.mode != vim::VimMode::Normal || v.pending(),
        }
    }

    /// Route one key through the Vim state machine. Returns `Some(outcome)`
    /// when Vim consumed the key; `None` means the caller's regular pipeline
    /// handles it (TS `#handleVimInput` :2234).
    pub(crate) fn vim_handle_key(&mut self, key: KeyEvent) -> Option<EditorOutcome> {
        if self.vim.is_none() {
            return None;
        }
        let snapshot = vim::VimBuffer::from_flat(&self.buffer.text(), self.buffer.flat_cursor());
        let (consumed, commands) = {
            let state = self.vim.as_mut().expect("checked above");
            state.handle_key(key, &snapshot)
        };
        if !consumed {
            return None;
        }
        self.apply_vim_commands(commands);
        Some(EditorOutcome::Edited)
    }

    fn apply_vim_commands(&mut self, commands: Vec<vim::VimCommand>) {
        use vim::VimCommand as C;
        for command in commands {
            match command {
                C::Move { to } => self.move_vim_cursor(to),
                C::Mode(_) => {
                    self.note_action(LastAction::Other);
                }
                C::Yank { from, to, linewise } => {
                    let body = self.slice_vim_range(from, to, linewise);
                    // The trailing newline is what marks a register linewise,
                    // so `p` puts it back as whole lines rather than splicing
                    // it mid-line (TS `#applyVimCommands` :2293).
                    let text = if linewise { format!("{body}\n") } else { body };
                    if !text.is_empty() {
                        self.push_kill(&text, false);
                        if let Some(state) = self.vim.as_mut() {
                            state.register = Some(text);
                        }
                    }
                    self.move_vim_cursor(from);
                }
                C::Delete {
                    from,
                    to,
                    linewise,
                    insert: _,
                } => self.delete_vim_range(from, to, linewise),
                C::OpenLine { below } => self.open_vim_line(below),
                C::Paste { after, count } => self.paste_vim_register(after, count),
                C::Undo => self.apply_undo(),
            }
        }
        self.clamp_vim_cursor();
    }

    fn move_vim_cursor(&mut self, to: vim::VimPosition) {
        let line = to.line.min(self.buffer.lines().len().saturating_sub(1));
        let col = byte_col(self.buffer.lines()[line].as_str(), to.col);
        self.buffer.set_cursor(Cursor { line, col });
    }

    /// Normal mode rests the cursor *on* a grapheme; Insert and Visual may
    /// sit one past the end (TS `#clampVimCursor` :2327).
    fn clamp_vim_cursor(&mut self) {
        if self
            .vim
            .as_ref()
            .map_or(true, |v| v.mode != vim::VimMode::Normal)
        {
            return;
        }
        let line = self.buffer.cursor().line;
        let text = self.buffer.lines()[line].clone();
        let chars = text.chars().count();
        if chars > 0 {
            let last = byte_col(&text, chars - 1);
            if self.buffer.cursor().col > last {
                self.buffer.set_cursor(Cursor { line, col: last });
            }
        }
    }

    /// Text covered by a half-open `[from, to)` range. Linewise ranges take
    /// whole lines. Positions are char indices.
    fn slice_vim_range(
        &self,
        from: vim::VimPosition,
        to: vim::VimPosition,
        linewise: bool,
    ) -> String {
        let lines = self.buffer.lines();
        if linewise {
            let last = to.line.min(lines.len().saturating_sub(1));
            return lines[from.line..=last].join("\n");
        }
        if from.line == to.line {
            let line = &lines[from.line.min(lines.len() - 1)];
            return char_slice(line, from.col, to.col);
        }
        let mut parts = vec![char_slice(&lines[from.line], from.col, usize::MAX)];
        for line in lines.iter().take(to.line).skip(from.line + 1) {
            parts.push(line.clone());
        }
        parts.push(char_slice(&lines[to.line.min(lines.len() - 1)], 0, to.col));
        parts.join("\n")
    }

    fn delete_vim_range(&mut self, from: vim::VimPosition, to: vim::VimPosition, linewise: bool) {
        let mut lines: Vec<String> = self.buffer.lines().to_vec();
        if linewise {
            let last = to.line.min(lines.len() - 1);
            let removed = lines[from.line..=last].join("\n");
            if removed.is_empty() && from.line == last && lines.len() == 1 {
                return;
            }
            self.record_undo();
            self.push_kill(&format!("{removed}\n"), false);
            lines.drain(from.line..=last);
            if lines.is_empty() {
                lines.push(String::new());
            }
            let line = from.line.min(lines.len() - 1);
            self.replace_lines(lines, Cursor { line, col: 0 });
            return;
        }
        let removed = self.slice_vim_range(from, to, false);
        if removed.is_empty() {
            return;
        }
        self.record_undo();
        self.push_kill(&removed, false);
        let last = to.line.min(lines.len() - 1);
        let head = char_slice(&lines[from.line], 0, from.col);
        let tail = char_slice(&lines[last], to.col, usize::MAX);
        lines.splice(from.line..=last, [format!("{head}{tail}")]);
        self.replace_lines(
            lines,
            Cursor {
                line: from.line,
                col: head.len(),
            },
        );
    }

    fn open_vim_line(&mut self, below: bool) {
        self.record_undo();
        let mut lines: Vec<String> = self.buffer.lines().to_vec();
        let at = if below {
            self.buffer.cursor().line + 1
        } else {
            self.buffer.cursor().line
        };
        let at = at.min(lines.len());
        lines.insert(at, String::new());
        self.replace_lines(lines, Cursor { line: at, col: 0 });
    }

    fn paste_vim_register(&mut self, after: bool, count: usize) {
        let Some(state) = self.vim.as_ref() else {
            return;
        };
        let Some(entry) = state.register.clone() else {
            return;
        };
        if entry.is_empty() {
            return;
        }
        self.record_undo();
        let linewise = entry.ends_with('\n');
        let mut lines: Vec<String> = self.buffer.lines().to_vec();
        let cur = self.buffer.cursor();
        if linewise {
            let body: Vec<String> = entry
                .trim_end_matches('\n')
                .split('\n')
                .map(str::to_owned)
                .collect();
            let at = (if after { cur.line + 1 } else { cur.line }).min(lines.len());
            let mut payload = Vec::with_capacity(body.len() * count);
            for _ in 0..count {
                payload.extend(body.iter().cloned());
            }
            lines.splice(at..at, payload);
            self.replace_lines(lines, Cursor { line: at, col: 0 });
        } else {
            let line_text = lines[cur.line].clone();
            let cursor_char = line_text[..cur.col.min(line_text.len())].chars().count();
            let at = if after {
                byte_col(&line_text, cursor_char + 1)
            } else {
                cur.col
            };
            let payload = entry.repeat(count);
            let last_char = payload.char_indices().last().map_or(0, |(i, _)| i);
            lines[cur.line] = format!(
                "{}{payload}{}",
                &line_text[..at.min(line_text.len())],
                &line_text[at.min(line_text.len())..]
            );
            self.replace_lines(
                lines,
                Cursor {
                    line: cur.line,
                    col: at + last_char,
                },
            );
        }
    }

    /// Rebuild the buffer from logical lines and park the cursor.
    fn replace_lines(&mut self, lines: Vec<String>, cursor: Cursor) {
        self.buffer.set_text(&lines.join("\n"));
        self.buffer.set_cursor(cursor);
        self.history_nav.reset();
        self.note_action(LastAction::Other);
    }

    // -- submit --------------------------------------------------------------

    pub(crate) fn set_submit_kind(&mut self, kind: submit::SubmitKind) {
        self.submit_kind = kind;
    }

    /// Text with paste and attachment markers expanded (`getExpandedText`).
    #[must_use]
    pub fn expanded_text(&self) -> String {
        let with_pastes = self.pastes.expand(&self.buffer.text());
        self.attachments.expand(&with_pastes)
    }

    /// Take the buffer contents for submit: expands markers, resets the
    /// buffer/history-nav/undo/paste state (`#submitValue` :3329), records
    /// the submission in history, and returns the expanded text.
    #[must_use]
    pub fn take(&mut self) -> String {
        self.take_submit().text
    }

    /// Like [`Editor::take`] but returns the full classified request for
    /// the app's submit router (`input-controller.ts:942`).
    #[must_use]
    pub fn take_submit(&mut self) -> SubmitRequest {
        let visible = self.buffer.text();
        let expanded = self.expanded_text();
        let kind = std::mem::take(&mut self.submit_kind);
        let request = SubmitRequest::classify(&visible, expanded, kind);
        // Reset state (TS `#submitValue`).
        self.buffer = TextBuffer::new();
        self.history_nav.reset();
        self.undo.clear();
        self.pastes.clear();
        self.scroll_offset.set(0);
        self.note_action(LastAction::Other);
        let trimmed = request.text.trim().to_owned();
        if !trimmed.is_empty() && request.slash.is_none() {
            self.history.push(&trimmed);
        }
        request
    }

    // -- paste ----------------------------------------------------------------

    /// Handle a (bracketed) paste payload: sanitize, collapse large pastes
    /// to an atomic marker (`#handlePaste` :3177), path-assist spacing.
    pub fn handle_paste(&mut self, text: &str) {
        let sanitized = paste::sanitize(text);
        let assisted = paste::space_before_glued_path(&sanitized);
        // A paste is one composite undo step (`#withUndoSuspended`).
        self.record_undo();
        self.undo.suspend();
        if paste::is_large(&assisted) {
            let id = self.pastes.store(assisted.clone());
            let label = paste::marker_label(id, &assisted);
            if !self.buffer.is_empty() && !self.buffer.current_line().is_empty() {
                self.buffer.insert_char('\n');
            }
            self.buffer.insert_str(&label);
        } else {
            self.buffer.insert_str(&assisted);
        }
        self.undo.resume();
        self.note_action(LastAction::Other);
    }

    /// Borrow the paste store (expansion of `[Paste #N]` markers).
    #[must_use]
    pub fn pastes(&self) -> &paste::PasteStore {
        &self.pastes
    }

    // -- attachments -----------------------------------------------------------

    /// Queue an image attachment; the marker label is inserted at the
    /// cursor as an atomic chip.
    pub fn attach_image(&mut self, source: &str, width: Option<u32>, height: Option<u32>) {
        let (_, label) = self.attachments.add_image(source, width, height);
        self.record_undo();
        self.buffer.insert_str(&label);
        self.note_action(LastAction::Other);
    }

    /// Queue a video attachment and insert its marker chip.
    pub fn attach_video(&mut self, source: &str) {
        let (_, label) = self.attachments.add_video(source);
        self.record_undo();
        self.buffer.insert_str(&label);
        self.note_action(LastAction::Other);
    }

    /// Queue a text attachment and insert its marker chip.
    pub fn attach_text(&mut self, name: &str, content: &str) {
        let (_, label) = self.attachments.add_text(name, content);
        self.record_undo();
        self.buffer.insert_str(&label);
        self.note_action(LastAction::Other);
    }

    /// Borrow the attachment table (chips).
    #[must_use]
    pub fn attachments(&self) -> &attachments::Attachments {
        &self.attachments
    }
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;
    use crate::keys::{KeyCode, KeyEvent};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, crate::keys::KeyModifiers::NONE)
    }

    #[test]
    fn legacy_typing_and_editing_api() {
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
    fn legacy_enter_submits_ctrl_j_newlines() {
        let mut e = Editor::new();
        assert_eq!(e.handle_key(key(KeyCode::Enter)), EditorOutcome::Submit);
        let ctrl_j = KeyEvent::new(KeyCode::Char('j'), crate::keys::KeyModifiers::CONTROL);
        assert_eq!(e.handle_key(ctrl_j), EditorOutcome::Edited);
        assert_eq!(e.text(), "\n");
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), crate::keys::KeyModifiers::CONTROL);
        assert_eq!(e.handle_key(ctrl_c), EditorOutcome::Interrupt);
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::keys::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }
    fn shift(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::SHIFT)
    }
    fn alt(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::ALT)
    }
    fn type_str(e: &mut Editor, s: &str) {
        for c in s.chars() {
            let _ = e.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    #[test]
    fn word_kills_accumulate_and_yank_pop_rotates() {
        let mut e = Editor::new();
        type_str(&mut e, "foo");
        let _ = e.handle_key(ctrl(KeyCode::Char('u'))); // kill "foo"
        type_str(&mut e, "bar");
        let _ = e.handle_key(ctrl(KeyCode::Char('u'))); // kill "bar"
        assert_eq!(e.text(), "");
        type_str(&mut e, "x");
        let _ = e.handle_key(key(KeyCode::Backspace));
        let _ = e.handle_key(ctrl(KeyCode::Char('y'))); // yank "bar"
        assert_eq!(e.text(), "bar");
        let _ = e.handle_key(alt(KeyCode::Char('y'))); // pop -> "foo"
        assert_eq!(e.text(), "foo");
    }

    #[test]
    fn consecutive_backward_word_kills_merge() {
        let mut e = Editor::new();
        type_str(&mut e, "hello world");
        let _ = e.handle_key(ctrl(KeyCode::Char('w'))); // kill "world"
        let _ = e.handle_key(ctrl(KeyCode::Char('w'))); // accumulates "hello"
        assert_eq!(e.text(), "");
        let _ = e.handle_key(ctrl(KeyCode::Char('y')));
        assert_eq!(e.text(), "hello world");
    }

    #[test]
    fn ctrl_k_kills_to_line_end() {
        let mut e = Editor::new();
        type_str(&mut e, "abcdef");
        for _ in 0..3 {
            let _ = e.handle_key(key(KeyCode::Left));
        }
        let _ = e.handle_key(ctrl(KeyCode::Char('k')));
        assert_eq!(e.text(), "abc");
        let _ = e.handle_key(ctrl(KeyCode::Char('y')));
        assert_eq!(e.text(), "abcdef");
    }

    #[test]
    fn undo_via_ctrl_minus() {
        let mut e = Editor::new();
        type_str(&mut e, "ab");
        let _ = e.handle_key(ctrl(KeyCode::Char('-')));
        assert_eq!(e.text(), "a");
        let _ = e.handle_key(ctrl(KeyCode::Char('_')));
        assert_eq!(e.text(), "");
    }

    #[test]
    fn shift_enter_newlines_enter_submits() {
        let mut e = Editor::new();
        type_str(&mut e, "ab");
        assert_eq!(e.handle_key(shift(KeyCode::Enter)), EditorOutcome::Edited);
        assert_eq!(e.text(), "ab\n");
        assert_eq!(e.handle_key(key(KeyCode::Enter)), EditorOutcome::Submit);
    }

    #[test]
    fn backslash_enter_converts_to_newline_plus_submit() {
        let mut e = Editor::new();
        type_str(&mut e, "abc\\");
        assert_eq!(e.handle_key(key(KeyCode::Enter)), EditorOutcome::Submit);
        assert_eq!(e.text(), "abc\n");
    }

    #[test]
    fn ctrl_enter_is_follow_up_submit() {
        let mut e = Editor::new();
        type_str(&mut e, "later");
        assert_eq!(e.handle_key(ctrl(KeyCode::Enter)), EditorOutcome::Submit);
        let req = e.take_submit();
        assert_eq!(req.kind, SubmitKind::FollowUp);
        assert_eq!(req.text, "later");
    }

    #[test]
    fn history_navigation_via_up_down() {
        let mut e = Editor::new();
        type_str(&mut e, "first");
        let _ = e.handle_key(key(KeyCode::Enter));
        let _ = e.take();
        type_str(&mut e, "second");
        let _ = e.handle_key(key(KeyCode::Enter));
        let _ = e.take();
        let _ = e.handle_key(key(KeyCode::Up));
        assert_eq!(e.text(), "second");
        let _ = e.handle_key(key(KeyCode::Up));
        assert_eq!(e.text(), "first");
        let _ = e.handle_key(key(KeyCode::Down));
        assert_eq!(e.text(), "second");
        let _ = e.handle_key(key(KeyCode::Down));
        assert_eq!(e.text(), "");
        // Draft recall: type while browsing, draft is restored on exit.
        let _ = e.handle_key(key(KeyCode::Up)); // "second"
        let _ = e.handle_key(key(KeyCode::Up)); // "first"
        type_str(&mut e, "!");
        assert_eq!(e.text(), "!");
    }

    #[test]
    fn word_motion_keys() {
        let mut e = Editor::new();
        type_str(&mut e, "foo bar");
        let _ = e.handle_key(alt(KeyCode::Char('b')));
        assert_eq!(e.structured_cursor().col, 4);
        let _ = e.handle_key(alt(KeyCode::Char('f')));
        assert_eq!(e.structured_cursor().col, 7);
    }

    #[test]
    fn large_paste_collapses_to_marker_and_expands_on_take() {
        let mut e = Editor::new();
        let payload = "x".repeat(1200);
        e.handle_paste(&payload);
        assert_eq!(e.text(), "[Paste #1, 1200 chars]");
        // Small paste inserts raw.
        e.handle_paste("small");
        assert_eq!(e.text(), "[Paste #1, 1200 chars]small");
        assert_eq!(e.take(), format!("{payload}small"));
        assert!(e.is_empty());
    }

    #[test]
    fn attachment_chip_expands_on_take() {
        let mut e = Editor::new();
        type_str(&mut e, "look ");
        e.attach_image("/tmp/a.png", Some(640), Some(480));
        assert_eq!(e.text(), "look [Image #1, 640x480]");
        assert_eq!(e.take(), "look /tmp/a.png");
    }

    #[test]
    fn submit_classifies_queue_and_slash_and_continue() {
        let mut e = Editor::new();
        type_str(&mut e, "-> do it");
        let _ = e.handle_key(key(KeyCode::Enter));
        let req = e.take_submit();
        assert_eq!(req.queue_body.as_deref(), Some("do it"));

        type_str(&mut e, "/model gpt");
        let _ = e.handle_key(key(KeyCode::Enter));
        let req = e.take_submit();
        assert_eq!(req.slash, Some(("model".to_string(), "gpt".to_string())));
        // Slash commands do not pollute prompt history.
        assert_eq!(e.history().len(), 1);

        type_str(&mut e, ".");
        let _ = e.handle_key(key(KeyCode::Enter));
        assert!(e.take_submit().is_continue);
    }

    #[test]
    fn empty_submit_while_streaming_flagged() {
        let mut e = Editor::new();
        assert_eq!(e.handle_key(key(KeyCode::Enter)), EditorOutcome::Submit);
        assert!(e.take_submit().was_empty);
    }

    #[test]
    fn escape_and_dequeue_keys_bubble_to_app() {
        use crate::component::{Component, EventOutcome};
        let mut e = Editor::new();
        assert_eq!(
            Component::handle_key(&mut e, key(KeyCode::Esc)),
            EventOutcome::Ignored
        );
        assert_eq!(
            Component::handle_key(&mut e, alt(KeyCode::Up)),
            EventOutcome::Ignored
        );
        // ...while plain keys are consumed.
        assert_eq!(
            Component::handle_key(&mut e, key(KeyCode::Char('a'))),
            EventOutcome::Consumed
        );
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::component::Component;
    use crate::keys::KeyModifiers;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_row(e: &Editor, y: u16, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| e.render(f, f.area(), &crate::theme::Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..width).map(|x| buf[(x, y)].symbol()).collect()
    }

    #[test]
    fn band_editor_renders_gutter_and_right_hint() {
        let mut e = Editor::new();
        e.set_hint(Some(("⇧⇥".into(), "to change thinking effort".into())));
        // Row 0 is the band's status row; the prompt sits under it.
        let row = render_row(&e, 1, 60, 2);
        assert!(row.starts_with("╰─ "), "gutter: {row:?}");
        assert!(
            row.ends_with("⇧⇥ to change thinking effort"),
            "hint flush right: {row:?}"
        );
        assert!(!row.contains('│') && !row.contains('╭'), "no box: {row:?}");
        e.set_hint(None);
        assert_eq!(render_row(&e, 1, 60, 2).trim_end(), "╰─");
    }

    #[test]
    fn typed_text_renders_after_gutter() {
        let mut e = Editor::new();
        for c in "hello".chars() {
            let _ = e.handle_key(crate::keys::KeyEvent::new(
                crate::keys::KeyCode::Char(c),
                KeyModifiers::NONE,
            ));
        }
        let content = render_row(&e, 1, 40, 2);
        assert!(content.starts_with("╰─ hello"), "content: {content:?}");
    }

    fn type_str(e: &mut Editor, s: &str) {
        for c in s.chars() {
            let _ = e.handle_key(crate::keys::KeyEvent::new(
                crate::keys::KeyCode::Char(c),
                KeyModifiers::NONE,
            ));
        }
    }

    /// Column of the inverse-video cursor cell on row `y`.
    fn cursor_col(e: &Editor, y: u16, width: u16, height: u16) -> Option<u16> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|f| e.render(f, f.area(), &crate::theme::Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..width).find(|&x| {
            buf[(x, y)]
                .modifier
                .contains(ratatui::style::Modifier::REVERSED)
        })
    }

    #[test]
    fn cursor_sits_after_last_typed_char_past_the_gutter() {
        // Regression: the cursor ignored the 2-cell `❯ ` gutter and was drawn
        // two cells left (on the last-but-one char).
        let mut e = Editor::new();
        type_str(&mut e, "hello world");
        // gutter(3) + 11 chars => column 14.
        assert_eq!(cursor_col(&e, 1, 40, 2), Some(14));
    }

    #[test]
    fn wrapped_rows_keep_gutter_indent_and_cursor() {
        // Regression: continuation rows were indented by their byte offset,
        // pushing text off-screen. Width 20 => 17 text cells per row.
        let mut e = Editor::new();
        type_str(&mut e, "abcdefghijklmnopqrstu");
        let height = e.desired_height(20);
        assert_eq!(height, 3, "status band row + two wrapped rows");
        let r0 = render_row(&e, 1, 20, height);
        let r1 = render_row(&e, 2, 20, height);
        assert_eq!(r0, "╰─ abcdefghijklmnopq");
        assert_eq!(r1.trim_end(), "   rstu");
        assert_eq!(cursor_col(&e, 2, 20, height), Some(7));
    }

    #[test]
    fn bash_mode_uses_accent_gutter() {
        let mut e = Editor::new();
        type_str(&mut e, "!ls");
        assert!(e.is_bash_mode());
        let mut terminal = Terminal::new(TestBackend::new(40, 2)).unwrap();
        let theme = crate::theme::Theme::default();
        terminal.draw(|f| e.render(f, f.area(), &theme)).unwrap();
        assert_eq!(
            terminal.backend().buffer()[(0, 1)].fg,
            theme.color("accent")
        );
    }

    /// All rows of a `width`x`height` render, trailing blanks trimmed.
    fn render_all(e: &Editor, width: u16, height: u16) -> Vec<String> {
        (0..height)
            .map(|y| render_row(e, y, width, height).trim_end().to_owned())
            .collect()
    }

    #[test]
    fn shapes_change_the_editor_look() {
        use crate::composer::ComposerShape;
        let mut e = Editor::new();
        type_str(&mut e, "hi");
        let expect = |shape: ComposerShape, rows: u16| {
            let mut e = e.clone();
            e.set_shape(shape);
            assert_eq!(e.desired_height(40), rows, "{shape:?} height");
            render_all(&e, 40, rows)
        };
        let box_rows = expect(ComposerShape::Box, 2);
        assert!(
            box_rows[0].starts_with("╭──") && box_rows[0].ends_with("──╮"),
            "{box_rows:?}"
        );
        assert!(
            box_rows[1].starts_with("╰─ hi") && box_rows[1].ends_with(" ─╯"),
            "{box_rows:?}"
        );
        let claude = expect(ComposerShape::Claude, 3);
        assert_eq!(claude[0], "─".repeat(40), "{claude:?}");
        assert_eq!(claude[1], "❯ hi");
        assert_eq!(claude[2], "─".repeat(40));
        let pi = expect(ComposerShape::Pi, 3);
        assert_eq!(pi[1], " hi");
        let borderless = expect(ComposerShape::Borderless, 1);
        assert_eq!(borderless[0], "❯ hi");
        let rule = expect(ComposerShape::Rule, 2);
        assert_eq!(rule[0], "─".repeat(40));
        assert_eq!(rule[1], "❯ hi");
        let field = expect(ComposerShape::Field, 1);
        assert!(
            field[0].starts_with("▐ hi") && field[0].ends_with('▌'),
            "{field:?}"
        );
        let rail = expect(ComposerShape::Rail, 1);
        assert!(rail[0].starts_with("▎ hi"), "{rail:?}");
    }

    #[test]
    fn box_shape_cursor_follows_chrome() {
        let mut e = Editor::new();
        e.set_shape(crate::composer::ComposerShape::Box);
        type_str(&mut e, "abc");
        // `╰─ ` (3 cells) + 3 chars.
        assert_eq!(cursor_col(&e, 1, 40, 2), Some(6));
    }

    #[test]
    fn desired_height_grows_with_content_and_caps() {
        let mut e = Editor::new();
        assert_eq!(e.desired_height(80), 2, "band row + one prompt row");
        for i in 0..20 {
            let _ = e.handle_key(crate::keys::KeyEvent::new(
                crate::keys::KeyCode::Char(char::from(b'a' + i as u8 % 26)),
                KeyModifiers::NONE,
            ));
            e.insert_newline();
        }
        let h = e.desired_height(80);
        assert!(h > 1, "grows: {h}");
        assert!(h <= 10, "capped: {h}");
    }
}
