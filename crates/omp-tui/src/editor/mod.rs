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
//! - [`render`] — boxed layout, gutter, placeholder, bash `!` border
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
mod layout;
mod paste;
mod render;
mod submit;
mod undo;

pub mod vim;

pub use attachments::{Attachment, AttachmentKind, Attachments};
pub use buffer::{Cursor, TextBuffer};
pub use history::{HistoryNav, HistoryStore, JsonlHistory, MemoryHistory};
pub use paste::PasteStore;
pub use submit::{SubmitKind, SubmitRequest};

use dispatch::LastAction;
use render::ScrollOffset;

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
    pub(crate) placeholder: String,
    pub(crate) disable_submit: bool,
    submit_kind: submit::SubmitKind,
    last_action: LastAction,
    yank_span: Option<(usize, usize, usize)>,
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
            placeholder: render::PLACEHOLDER.to_owned(),
            disable_submit: false,
            submit_kind: submit::SubmitKind::Normal,
            last_action: LastAction::Other,
            yank_span: None,
        }
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

    /// Insert one character at the cursor.
    pub fn insert_char(&mut self, c: char) {
        if self.history_nav.browsing() {
            // Editing a recalled history entry exits browsing and restores
            // the draft (`#exitHistoryForEditing`).
            let draft = self.history_nav.exit_draft();
            self.buffer.set_text(&draft);
        }
        self.record_undo();
        self.buffer.insert_char(c);
        self.note_action(LastAction::Other);
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

    /// Set the placeholder hint shown on an empty buffer.
    pub fn set_placeholder(&mut self, text: &str) {
        text.clone_into(&mut self.placeholder);
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
    fn boxed_editor_renders_border_gutter_placeholder() {
        let e = Editor::new();
        let top = render_row(&e, 0, 40, 5);
        assert!(top.starts_with('╭'), "top border: {top:?}");
        assert!(top.ends_with('╮'));
        let content = render_row(&e, 1, 40, 5);
        assert!(content.contains('❯'), "gutter: {content:?}");
        assert!(content.contains("Ask omp"), "placeholder: {content:?}");
        let bottom = render_row(&e, 4, 40, 5);
        assert!(bottom.starts_with('╰'), "bottom border: {bottom:?}");
    }

    #[test]
    fn typed_text_renders_inside_box() {
        let mut e = Editor::new();
        for c in "hello".chars() {
            e.handle_key(crate::keys::KeyEvent::new(
                crate::keys::KeyCode::Char(c),
                KeyModifiers::NONE,
            ));
        }
        let content = render_row(&e, 1, 40, 5);
        assert!(content.contains("❯ hello"), "content: {content:?}");
    }

    #[test]
    fn bash_mode_uses_accent_border_and_title() {
        let mut e = Editor::new();
        for c in "!ls".chars() {
            e.handle_key(crate::keys::KeyEvent::new(
                crate::keys::KeyCode::Char(c),
                KeyModifiers::NONE,
            ));
        }
        let top = render_row(&e, 0, 40, 5);
        assert!(top.contains("bash"), "title: {top:?}");
    }

    #[test]
    fn desired_height_grows_with_content_and_caps() {
        let mut e = Editor::new();
        assert_eq!(e.desired_height(80), 3);
        for i in 0..20 {
            e.handle_key(crate::keys::KeyEvent::new(
                crate::keys::KeyCode::Char(char::from(b'a' + i as u8 % 26)),
                KeyModifiers::NONE,
            ));
            e.insert_newline();
        }
        let h = e.desired_height(80);
        assert!(h > 3, "grows: {h}");
        assert!(h <= 10, "capped: {h}");
    }
}
