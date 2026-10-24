//! Key dispatch for the editor — the port of the base Editor's
//! `handleInput` pipeline (`components/editor.ts:1755-2233`) restricted to
//! the P0 binding set (`keybindings.ts:70-166` `TUI_KEYBINDINGS`).
//!
//! Order (mirrors TS): undo → autocomplete popup nav (owned by the
//! autocomplete overlay, not here) → kills → yank/yank-pop → newline group
//! (shift+enter / ctrl+j; backslash-Enter) → submit (plain Enter;
//! ctrl+enter / ctrl+q = follow-up) → backspace (marker-widening) →
//! motions → printable chars → history up/down.
//!
//! Keys the app owns (escape interrupt, `alt+up`/`shift+up` dequeue,
//! `ctrl+d` exit on an empty buffer, all `app.*` chords) are returned as
//! [`EditorOutcome::Ignored`] by [`Editor::dispatch_app_keys`] before this
//! pipeline runs.

use crate::editor::layout::marker_covering;
use crate::editor::submit::SubmitKind;
use crate::editor::{Editor, EditorOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};

/// What the last mutating action was, for kill accumulation / yank-pop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LastAction {
    /// Anything that is not a kill or yank (resets both sequences).
    Other,
    /// A kill (`ctrl+u/k/w`, `alt+d`, ...): the next kill accumulates.
    Kill {
        /// Backward kills prepend, forward kills append.
        backward: bool,
    },
    /// A yank or yank-pop: the next `alt+y` replaces the yanked span.
    Yank,
    /// A Tab word-completion accept: the trailing space it added is
    /// provisional until the next keystroke (see [`Editor::insert_char`]).
    AcceptWord,
}

impl Editor {
    /// Handle one key event (the full P0 pipeline).
    ///
    /// The match is intentionally one flat table (TS `handleInput` order);
    /// splitting it would obscure the binding precedence.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn handle_key(&mut self, key: KeyEvent) -> EditorOutcome {
        use KeyCode::{
            Backspace, Char, Delete, Down, End, Enter, Home, Left, PageDown, PageUp, Right, Up,
        };
        let mods = key.modifiers;
        let ctrl = mods.contains(KeyModifiers::CONTROL);
        let alt = mods.contains(KeyModifiers::ALT);
        let shift = mods.contains(KeyModifiers::SHIFT);

        // -- vim modal editing (TS editor.ts:1813, :2234) -------------------
        // Escape is the one key Vim owns in every mode; everything else in
        // Insert mode and every unmapped chord falls through to the pipeline
        // below, so behavior with Vim off is untouched.
        if let Some(outcome) = self.vim_handle_key(key) {
            return outcome;
        }

        match key.code {
            // -- app-level -------------------------------------------------
            Char('c') if ctrl => EditorOutcome::Interrupt,

            // -- submit / newline group (TS :2085-2118) ---------------------
            Enter | Char('q') if ctrl && !alt => self.follow_up_submit(),
            Enter if shift => {
                self.newline();
                EditorOutcome::Edited
            }
            Char('j') if ctrl && !alt && !shift => {
                self.newline();
                EditorOutcome::Edited
            }
            Enter => {
                // Backslash-Enter: a trailing `\` on the line converts the
                // Enter into newline+submit (`#shouldSubmitOnBackslashEnter`).
                if self.current_line_ends_with_backslash() {
                    self.strip_trailing_backslash();
                    self.newline();
                }
                self.try_submit()
            }

            // -- undo (`tui.editor.undo` = ctrl+- / ctrl+_) ----------------
            Char('-' | '_') if ctrl && !alt => {
                self.apply_undo();
                EditorOutcome::Edited
            }

            // -- kill group (TS :2020-2083) ---------------------------------
            Char('u') if ctrl && !alt => self.kill_to_line_start(),
            Char('k') if ctrl && !alt => self.kill_to_line_end(),
            Char('w') if ctrl && !alt => self.kill_word_backward(),
            Char('d') if ctrl && !alt => self.forward_delete(),
            Backspace if alt || ctrl => self.kill_word_backward(),
            Delete if alt => self.kill_word_forward(),
            Char('d') if alt && !ctrl => self.kill_word_forward(),

            // -- yank group --------------------------------------------------
            Char('y') if ctrl && !alt => self.yank(),
            Char('y') if alt && !ctrl => self.yank_pop(),

            // -- history / vertical movement (TS :2160-2188) -----------------
            Up if !ctrl && !alt && !shift => {
                self.history_up();
                EditorOutcome::Edited
            }
            Down if !ctrl && !alt && !shift => {
                self.history_down();
                EditorOutcome::Edited
            }

            // -- horizontal movement ------------------------------------------
            Char('b') if ctrl && !alt => {
                self.move_left();
                EditorOutcome::Edited
            }
            Char('f') if ctrl && !alt => {
                self.move_right();
                EditorOutcome::Edited
            }
            Char('a') if ctrl && !alt => {
                self.move_line_start();
                EditorOutcome::Edited
            }
            Char('e') if ctrl && !alt => {
                self.move_line_end();
                EditorOutcome::Edited
            }
            // alt+arrows / ctrl+arrows = word motion (TS `cursorWordLeft/Right`).
            Left if alt || ctrl => {
                self.move_word_left();
                EditorOutcome::Edited
            }
            Right if alt || ctrl => {
                self.move_word_right();
                EditorOutcome::Edited
            }
            Left => {
                self.move_left();
                EditorOutcome::Edited
            }
            Right => {
                self.accept_or_move_right();
                EditorOutcome::Edited
            }
            Home => {
                self.move_line_start();
                EditorOutcome::Edited
            }
            End => {
                self.move_line_end();
                EditorOutcome::Edited
            }

            // -- word movement (alt+arrows, alt+b/f) ---------------------------
            Char('b') if alt && !ctrl => {
                self.move_word_left();
                EditorOutcome::Edited
            }
            Char('f') if alt && !ctrl => {
                self.move_word_right();
                EditorOutcome::Edited
            }
            // -- page scroll (viewport only, never history; TS :2141) ---------
            PageUp => {
                self.scroll_page_up();
                EditorOutcome::Edited
            }
            PageDown => {
                self.scroll_page_down();
                EditorOutcome::Edited
            }

            // -- delete ---------------------------------------------------------
            Backspace => {
                self.backspace_widened();
                EditorOutcome::Edited
            }
            Delete => self.forward_delete(),

            // -- printable input -------------------------------------------------
            Char(c) if !ctrl && !alt => {
                self.insert_char(c);
                EditorOutcome::Edited
            }
            // Everything else — Tab (owned by the autocomplete popup in the
            // full composer), unknown chords — is consumed without effect.
            _ => EditorOutcome::Edited,
        }
    }

    // -- operation helpers (each records undo + action sequences) -----------

    pub(crate) fn newline(&mut self) {
        self.record_undo();
        self.drop_provisional_space();
        self.buffer.insert_char('\n');
        self.note_action(LastAction::Other);
    }

    fn try_submit(&mut self) -> EditorOutcome {
        if self.disable_submit {
            return EditorOutcome::Edited;
        }
        EditorOutcome::Submit
    }

    fn follow_up_submit(&mut self) -> EditorOutcome {
        if self.disable_submit {
            return EditorOutcome::Edited;
        }
        self.set_submit_kind(SubmitKind::FollowUp);
        EditorOutcome::Submit
    }

    fn kill_to_line_start(&mut self) -> EditorOutcome {
        self.record_undo();
        let killed = self.buffer.delete_to_line_start();
        self.push_kill(&killed, true);
        EditorOutcome::Edited
    }

    fn kill_to_line_end(&mut self) -> EditorOutcome {
        self.record_undo();
        let killed = self.buffer.delete_to_line_end();
        self.push_kill(&killed, false);
        EditorOutcome::Edited
    }

    fn kill_word_backward(&mut self) -> EditorOutcome {
        self.record_undo();
        let killed = self.buffer.delete_word_backward();
        self.push_kill(&killed, true);
        EditorOutcome::Edited
    }

    fn kill_word_forward(&mut self) -> EditorOutcome {
        self.record_undo();
        let killed = self.buffer.delete_word_forward();
        self.push_kill(&killed, false);
        EditorOutcome::Edited
    }

    fn yank(&mut self) -> EditorOutcome {
        let Some(text) = self.kill_ring_yank().map(str::to_owned) else {
            return EditorOutcome::Edited;
        };
        self.record_undo();
        self.insert_yanked(&text);
        self.note_action(LastAction::Yank);
        EditorOutcome::Edited
    }

    fn yank_pop(&mut self) -> EditorOutcome {
        if self.last_action() != LastAction::Yank {
            return EditorOutcome::Edited;
        }
        let Some(text) = self.kill_ring_rotate().map(str::to_owned) else {
            return EditorOutcome::Edited;
        };
        self.record_undo();
        self.replace_yanked(&text);
        EditorOutcome::Edited
    }

    fn backspace_widened(&mut self) {
        // Atomic markers: backspace deletes the whole marker the cursor
        // sits at/after (`#widenOverAtomicTokens`).
        let line = self.buffer.cursor().line;
        let col = self.buffer.cursor().col;
        let line_text = self.buffer.lines()[line].clone();
        if let Some(span) = marker_covering(&line_text, col) {
            if span.start < col {
                self.record_undo();
                self.buffer.replace_range(line, span.start, span.end, "");
                self.note_action(LastAction::Other);
                return;
            }
        }
        self.record_undo();
        self.buffer.delete_backward();
        self.note_action(LastAction::Other);
    }

    fn forward_delete(&mut self) -> EditorOutcome {
        self.record_undo();
        self.buffer.delete_forward();
        self.note_action(LastAction::Other);
        EditorOutcome::Edited
    }

    fn accept_or_move_right(&mut self) {
        // Right walks over the ghost word completion as if it were typed: no
        // trailing space (TS `#acceptWordCompletion({ space: false })`).
        if self.accept_word_completion(false) {
            return;
        }
        self.move_right();
    }

    fn current_line_ends_with_backslash(&self) -> bool {
        self.buffer.current_line().ends_with('\\')
    }

    fn strip_trailing_backslash(&mut self) {
        self.record_undo();
        let line = self.buffer.cursor().line;
        let len = self.buffer.lines()[line].len();
        self.buffer.replace_range(line, len - 1, len, "");
    }
}

#[cfg(test)]
mod vim_dispatch_tests {
    use super::super::vim::VimMode;
    use super::*;
    use crate::keys::KeyCode;

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn esc() -> KeyEvent {
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)
    }

    fn type_str(e: &mut Editor, s: &str) {
        for c in s.chars() {
            let _ = e.handle_key(key(c));
        }
    }

    #[test]
    fn vim_mode_roundtrip_insert_normal_delete() {
        let mut e = Editor::new();
        e.set_vim_mode(true);
        assert!(e.vim_enabled());
        assert_eq!(e.vim_mode(), Some(VimMode::Insert));
        // Insert mode owns Escape (it leaves insert first).
        assert!(e.vim_consumes_escape());

        type_str(&mut e, "hello world");
        assert_eq!(e.text(), "hello world");

        let _ = e.handle_key(esc());
        assert_eq!(e.vim_mode(), Some(VimMode::Normal));
        assert_eq!(e.structured_cursor().col, 10);
        // Idle Normal mode: Escape is not consumed (the app's interrupt).
        let outcome = e.vim_handle_key(esc());
        assert!(outcome.is_none());

        // `0dw` deletes the first word plus the gap.
        let _ = e.handle_key(key('0'));
        let _ = e.handle_key(key('d'));
        assert_eq!(e.vim_pending(), "d");
        let _ = e.handle_key(key('w'));
        assert_eq!(e.text(), "world");
        assert_eq!(e.vim_pending(), "");
    }

    #[test]
    fn vim_dd_yank_and_paste_linewise() {
        let mut e = Editor::new();
        e.set_vim_mode(true);
        type_str(&mut e, "foo");
        let _ = e.handle_key(esc());
        let _ = e.handle_key(key('y'));
        let _ = e.handle_key(key('y'));
        assert_eq!(e.text(), "foo"); // yank does not change the text
        let _ = e.handle_key(key('p'));
        assert_eq!(e.text(), "foo\nfoo");
        // Cursor parked on the pasted line, still Normal.
        assert_eq!(e.vim_mode(), Some(VimMode::Normal));
        assert_eq!(e.structured_cursor().line, 1);
        // Undo restores the single line.
        let _ = e.handle_key(key('u'));
        assert_eq!(e.text(), "foo");
    }

    #[test]
    fn vim_visual_delete_and_escape_pending() {
        let mut e = Editor::new();
        e.set_vim_mode(true);
        type_str(&mut e, "abcdef");
        let _ = e.handle_key(esc()); // cursor on 'f' (col 5)
        let _ = e.handle_key(key('0'));
        let _ = e.handle_key(key('v'));
        assert_eq!(e.vim_mode(), Some(VimMode::Visual));
        let _ = e.handle_key(key('l'));
        let _ = e.handle_key(key('l'));
        let _ = e.handle_key(key('d'));
        assert_eq!(e.text(), "def");
        assert_eq!(e.vim_mode(), Some(VimMode::Normal));

        // Half-typed operator + Escape: nothing changes.
        let before = e.text();
        let _ = e.handle_key(key('d'));
        assert_eq!(e.vim_pending(), "d");
        let _ = e.handle_key(esc());
        assert_eq!(e.vim_pending(), "");
        assert_eq!(e.text(), before);
    }

    #[test]
    fn vim_off_keeps_plain_typing_semantics() {
        let mut e = Editor::new();
        assert!(!e.vim_enabled());
        assert_eq!(e.vim_mode(), None);
        assert!(!e.vim_consumes_escape());
        // Keys type through; Escape is a no-op like before vim existed.
        type_str(&mut e, "dw");
        let outcome = e.handle_key(esc());
        assert_eq!(outcome, EditorOutcome::Edited);
        assert_eq!(e.text(), "dw");
    }
}
