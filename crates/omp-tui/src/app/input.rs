use super::{
    Action, App, Component, EditorOutcome, Event, Instant, KeyCode, KeyEvent, KeyModifiers,
    Maintenance, OpenOverlay, QueueKind, RuntimePort, SubmitBehavior, TreeMode, DOUBLE_TAP_WINDOW,
};

impl App {
    /// Handle one terminal event. Returns `false` when the app should quit.
    pub fn handle_terminal_event(&mut self, event: &Event, port: &impl RuntimePort) -> bool {
        if let Some(mouse) = crate::mouse::sgr_mouse_of(event) {
            self.handle_mouse(&mouse, port);
            return true;
        }
        // The user started a prompt: let the runtime open the provider
        // connection now, so DNS + TLS are done by the time Enter is pressed.
        let was_empty = self.editor.is_empty();
        let keep_running = if let Event::Key(key) = event {
            self.handle_key(*key, port, Instant::now()).unwrap_or(false)
        } else {
            // Bracketed paste: unwrap the payload (crossterm turns the
            // `\x1b[200~…\x1b[201~` bytes into `Event::Paste`) and run the
            // paste pipeline. Everything between the markers is paste
            // *content* — an Enter that shares the paste burst arrives as a
            // separate `Event::Key` after the end marker (TS `stdin-buffer.ts`
            // `leadingEnter` splits bytes *after* `\x1b[201~`), so there is
            // nothing submit-shaped inside the payload.
            if let Some(payload) = crate::bracketed_paste::paste_of(event) {
                self.route_paste(payload, false);
            }
            true // resize: redraw-only
        };
        if was_empty && !self.editor.is_empty() {
            port.prewarm_connection();
        }
        keep_running
    }

    /// Route one SGR mouse report. Reporting is only on while a full-screen
    /// overlay is on top (routed to it, then its intents drain like after a
    /// key) or while inline `tui.mouse` capture is on with no overlay open
    /// — TS consumes every report there so button/wheel bytes never reach
    /// the editor; omp-rs has no inline click targets (live subagent cards)
    /// yet, so inline reports are swallowed.
    pub fn handle_mouse(&mut self, event: &crate::mouse::SgrMouseEvent, port: &impl RuntimePort) {
        if self.overlays.wants_focus() && self.overlays.handle_mouse(event).consumed() {
            self.drain_overlay_actions(port);
        }
    }

    /// Whether terminal mouse reporting should be on right now (TS
    /// `tui.ts` `wantMouse`): the overlay layer decides while any overlay is
    /// open (on only for a TS `fullscreen: true` top overlay); otherwise the
    /// `tui.mouse` inline capture setting.
    #[must_use]
    pub fn wants_mouse(&self) -> bool {
        if self.overlays.wants_focus() {
            self.overlays.wants_mouse()
        } else {
            self.mouse_capture
        }
    }

    /// Dispatch one [`Action`] to the port (the single emit point).
    pub(super) fn dispatch(&mut self, action: Action, port: &impl RuntimePort) {
        match action {
            Action::Submit { id, text, behavior } => port.submit(id, text, behavior),
            Action::Steer { text } => port.steer(text),
            Action::FollowUp { text } => port.follow_up(text),
            Action::RemoveQueued { text, queue } => port.remove_queued(text, queue),
            Action::PromoteQueued { text } => port.promote_queued(text),
            Action::Abort => port.abort(),
            Action::Continue => port.continue_run(),
            Action::RefreshModels => port.refresh_models(),
            Action::ResumeSession { path, fork } => {
                // A resumed session replaces the transcript: drop the
                // journal feeding the rewind/tree overlays.
                self.journal.clear();
                port.resume_session(path, fork);
            }
            Action::Rewind { entry_id } => port.rewind_entry(entry_id),
            Action::SetModel { provider, id } => port.set_model(provider, id),
            Action::SetThinkingLevel(level) => port.set_thinking_level(level),
            Action::Slash { name, args } => port.run_slash(name, args),
            Action::SettingChanged { id, value } => {
                if let Err(e) = port.settings_set(id.clone(), value.clone()) {
                    self.push_system(&format!("settings: failed to save {id}: {e}"));
                } else {
                    self.apply_setting_live(&id, &value, port);
                    // Re-seed the overlay so layered values (env/project)
                    // reflect what actually won after the write.
                    let overlay = self.overlays.settings_mut();
                    let ids = overlay.ids();
                    let values = port.settings_get(&ids);
                    overlay.seed(&values);
                }
            }
            Action::SettingPreview { id, value } => self.preview_setting(&id, &value),
            Action::ExtensionUiResponse { id, value } => port.extension_ui_response(id, value),
            Action::Scroll(dir) => self.apply_scroll(dir),
            Action::Quit => {}
        }
    }

    /// Live preview of a highlighted `/settings` choice (TS `onThemePreview`):
    /// the theme hot-swaps while the select field is open; the previous theme
    /// returns on cancel.
    fn preview_setting(&mut self, id: &str, value: &str) {
        if matches!(id, "theme.dark" | "theme.light") {
            if let Ok(theme) = crate::theme::load(value) {
                self.set_theme(theme);
            }
        }
    }

    fn apply_scroll(&mut self, dir: crate::action::ScrollDir) {
        use crate::action::ScrollDir::{Bottom, Down, PageDown, PageUp, Up};
        match dir {
            Up => self.scroll = self.scroll.saturating_add(1),
            Down => self.scroll = self.scroll.saturating_sub(1),
            PageUp => self.scroll = self.scroll.saturating_add(10),
            PageDown => self.scroll = self.scroll.saturating_sub(10),
            Bottom => self.scroll = 0,
        }
    }

    /// Handle a key event; `None` means quit.
    // One dispatcher over overlays, editor and chords, mirroring the TS key
    // router; splitting it would scatter the ordering rules.
    #[allow(clippy::too_many_lines)]
    pub(super) fn handle_key(
        &mut self,
        key: KeyEvent,
        port: &impl RuntimePort,
        now: Instant,
    ) -> Option<bool> {
        // Global keys, active even under overlays. `?` toggles help (TS
        // hard-codes this chord outside the keybindings table); everything
        // else resolves through the KeybindingsManager so user overrides
        // from ~/.omp/agent/keybindings.yml apply (review2 §11 #15).
        if key.code == KeyCode::Char('?')
            && !key.modifiers.contains(KeyModifiers::CONTROL)
            && (self.editor.is_empty() || self.overlays.open() == Some(OpenOverlay::Help))
        {
            self.overlays.toggle(OpenOverlay::Help);
            return Some(true);
        }
        if self.keys.matches_event(&key, "app.exit") {
            return None; // exit (§6 handleCtrlD)
        }
        if self.keys.matches_event(&key, "app.tools.expand") {
            self.messages.toggle_expanded(); // app.tools.expand (chat.md)
            if self.theme.clean() {
                // Clean rows collapse in scrollback too: re-commit the whole
                // transcript in the new state.
                self.clear_scrollback = true;
            }
            return Some(true);
        }

        // Overlay focus: route keys to the stack, top first. Esc closes the
        // top overlay only when it ignored the key (help/palette ignore esc;
        // the tree navigator consumes esc as "cancel", so it is closed by
        // the host explicitly).
        if self.overlays.wants_focus() {
            if key.code == KeyCode::Esc {
                if self.overlays.open() == Some(OpenOverlay::Tree) {
                    self.overlays.pop();
                    return Some(true);
                }
                if !self.overlays.handle_key(key).consumed() {
                    self.overlays.pop();
                }
                return Some(true);
            }
            let consumed = self.overlays.handle_key(key).consumed();
            self.drain_overlay_actions(port);
            return Some(consumed || self.overlays.wants_focus());
        }

        // Ctrl+C discipline: first tap clears the editor draft, double-tap
        // within 500 ms exits (§6 handleCtrlC; exit code 130 comes from the
        // CLI). Bound via `app.clear`.
        if self.keys.matches_event(&key, "app.clear") {
            if self
                .last_ctrl_c
                .is_some_and(|t| now.duration_since(t) <= DOUBLE_TAP_WINDOW)
            {
                return None;
            }
            self.last_ctrl_c = Some(now);
            if !self.editor.is_empty() {
                let _ = self.editor.take();
            }
            return Some(true);
        }
        self.last_ctrl_c = None;

        // Pending extension dialogs own keys before the editor (ExtUi
        // contract): they render modally above the live region.
        if self.extension_ui.handle_key(key, port) {
            return Some(true);
        }

        // Shift+Tab: cycle the thinking effort (`app.thinking.cycle`).
        if self.keys.matches_event(&key, "app.thinking.cycle") {
            self.cycle_thinking_level(port);
            return Some(true);
        }

        // Restore the most recently queued message to the editor
        // (steering first, then follow-up; §5.3 handleDequeue); the chord
        // is the `app.message.dequeue` binding (default alt+up, shift+up).
        if self.keys.matches_event(&key, "app.message.dequeue") {
            self.dequeue_to_editor(port);
            return Some(true);
        }

        // Slash / @file popup owns navigation keys while open (editor.md
        // §9): ↑/↓/PgUp/PgDn select, Tab accepts, Enter accepts and — for a
        // leading slash command — submits, Esc dismisses before the ladder.
        if self.autocomplete.open {
            let enter = key.code == KeyCode::Enter && key.modifiers.is_empty();
            if self.autocomplete.handle_key(key).consumed() {
                if let Some((item, prefix)) = self.autocomplete.take_confirmed() {
                    let cur = self.editor.structured_cursor();
                    let applied = self.autocomplete.apply_completion(
                        self.editor.text_buffer().lines(),
                        cur.line,
                        cur.col,
                        &item,
                        &prefix,
                    );
                    self.editor.apply_completion(
                        &applied.lines,
                        applied.cursor_line,
                        applied.cursor_col,
                    );
                    let is_slash = crate::autocomplete::find_leading_slash_command_start(&prefix)
                        .is_some()
                        && !item.value.starts_with('/');
                    if enter && is_slash {
                        return self.submit_editor(port);
                    }
                    self.refresh_autocomplete();
                }
                return Some(true);
            }
        }

        // System-clipboard pastes (custom-editor.ts `app.clipboard.*`):
        // ctrl+v smart-pastes (image first, then text), ctrl+shift+v /
        // alt+shift+v paste the clipboard text raw. The read shells out, so
        // it runs on a worker thread and lands via `poll_deadlines`.
        if self.keys.matches_event(&key, "app.clipboard.pasteImage") {
            self.begin_clipboard_paste(crate::clipboard::PasteRequest::Smart);
            return Some(true);
        }
        if self.keys.matches_event(&key, "app.clipboard.pasteTextRaw") {
            self.begin_clipboard_paste(crate::clipboard::PasteRequest::TextRaw);
            return Some(true);
        }

        if self.keys.matches_event(&key, "app.interrupt") {
            return Some(self.esc_ladder(now, port));
        }

        // Tab accepts the ghost word completion plus a space; the popup branch
        // above already owns Tab while a slash/@ menu is open.
        if key.code == KeyCode::Tab
            && key.modifiers.is_empty()
            && !self.autocomplete.open
            && self.editor.accept_word_completion(true)
        {
            self.refresh_autocomplete();
            return Some(true);
        }

        match self.editor.handle_key(key) {
            EditorOutcome::Submit => self.submit_editor(port),
            EditorOutcome::Interrupt => Some(true), // app.clear handled above
            EditorOutcome::Edited => {
                self.refresh_autocomplete();
                Some(true)
            }
        }
    }

    /// Submit the editor draft through the submit pipeline; `None` = quit.
    fn submit_editor(&mut self, port: &impl RuntimePort) -> Option<bool> {
        self.autocomplete.close();
        let text = self.editor.take();
        self.editor.observe_prompt(&text);
        if !self.submit_pipeline(&text, port) || self.quit_requested {
            self.quit_requested = false;
            return None;
        }
        Some(true)
    }

    /// `paste.largeMenuThreshold` from settings (TS `cfgPasteLargeMenuThreshold`,
    /// default 100; `0` disables the menu so large pastes always collapse).
    fn large_paste_menu_threshold(&self) -> usize {
        self.load_settings()
            .get_number("paste.largeMenuThreshold")
            // Settings numbers are small positive counts here.
            .map_or(100, |v| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    if v <= 0.0 {
                        0
                    } else {
                        v as usize
                    }
                }
            })
    }

    /// Route an incoming paste payload through the TS paste pipeline
    /// (`#handlePaste` + `handleLargePaste`): small pastes insert inline,
    /// marker-sized pastes below the menu threshold stage as a
    /// text-attachment chip (`insertTextAttachment`), and a paste at/above
    /// the threshold opens the large-paste menu instead. `submit_after_paste`
    /// skips the menu (nobody can answer it before the submit lands, TS
    /// `options.submitAfterPaste`).
    fn route_paste(&mut self, text: &str, submit_after_paste: bool) {
        if !crate::editor::paste::is_large(text) {
            self.editor.handle_paste(text);
            return;
        }
        let line_count = crate::editor::paste::sanitize(text).lines().count();
        let threshold = self.large_paste_menu_threshold();
        if threshold > 0 && line_count >= threshold && !submit_after_paste {
            self.pending_large_paste = Some((text.to_owned(), line_count));
            self.overlays.open_large_paste(line_count);
            return;
        }
        // Below the gate: stage as a text-attachment chip (TS default in
        // `handleLargePaste`).
        self.editor.attach_text("paste.md", text);
    }

    /// Kick off a system-clipboard read on a worker thread (`handleImagePaste`
    /// / `handleClipboardTextRawPaste`); the result is applied when
    /// [`App::poll_deadlines`] drains it.
    fn begin_clipboard_paste(&mut self, request: crate::clipboard::PasteRequest) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.clipboard_rx = Some(rx);
        std::thread::spawn(move || {
            // A panicking bridge (bad stub command, odd locale) must not kill
            // the app: report it as a failed read.
            let outcome = std::panic::catch_unwind(|| crate::clipboard::perform(request))
                .unwrap_or(crate::clipboard::ClipboardOutcome::Failed);
            let _ = tx.send(outcome);
        });
    }

    /// Apply one drained clipboard read (TS `handleImagePaste` smart order:
    /// image first — a resolved text payload is discarded unused when an
    /// image is present — then text, with "Clipboard is empty" only when
    /// both miss).
    pub(super) fn apply_clipboard_outcome(&mut self, outcome: crate::clipboard::ClipboardOutcome) {
        use crate::clipboard::ClipboardOutcome;
        match outcome {
            ClipboardOutcome::Image(png) => self.attach_clipboard_image(&png),
            ClipboardOutcome::TextRaw(text) => {
                self.editor.insert_str_at_cursor(&text);
            }
            ClipboardOutcome::Text(text) if text.is_empty() => {
                self.push_system("Clipboard is empty");
            }
            ClipboardOutcome::Text(text) => self.route_paste(&text, false),
            ClipboardOutcome::Empty => {
                self.push_system("Clipboard is empty");
            }
            ClipboardOutcome::Failed => {
                self.push_system("Failed to read clipboard");
            }
        }
    }

    /// Save a clipboard PNG to a temp file and queue it as an image
    /// attachment (TS `sink.attach({type:"image", …})`; omp-rs attachments
    /// reference a file path, so the payload is materialized first).
    fn attach_clipboard_image(&mut self, png: &[u8]) {
        let path = std::env::temp_dir().join(format!(
            "omp-clipboard-{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis())
        ));
        if std::fs::write(&path, png).is_err() {
            self.push_system("Failed to save clipboard image");
            return;
        }
        self.editor
            .attach_image(&path.display().to_string(), None, None);
    }

    /// Apply the large-paste menu choice (`presentLargePasteMenu` switch):
    /// wrapped `<attachment>` block, file reference, inline marker, or the
    /// Esc fallback (stage as a text-attachment chip). The content is never
    /// lost.
    pub(super) fn apply_large_paste_pick(
        &mut self,
        pick: crate::overlays::large_paste::LargePastePick,
    ) {
        use crate::overlays::large_paste::LargePastePick;
        let Some((text, line_count)) = self.pending_large_paste.take() else {
            return;
        };
        self.overlays.pop(); // close the LargePaste panel (it is on top)
        match pick {
            LargePastePick::WrappedBlock => {
                let wrapped = format!("<attachment>\n{text}\n</attachment>");
                self.editor.attach_text("paste.md", &wrapped);
            }
            LargePastePick::LocalFile => {
                // TS saves to the session's `local://` store; omp-rs has no
                // session-local store on the UI side yet, so the paste lands
                // in a temp file and the path is referenced inline. Failure
                // falls back to the text chip like TS `#attachPasteAsFile`.
                let path = std::env::temp_dir().join(format!(
                    "omp-paste-{}.md",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_millis())
                ));
                if std::fs::write(&path, &text).is_ok() {
                    self.editor
                        .insert_str_at_cursor(&format!("{} ", path.display()));
                    self.push_system(&format!(
                        "Saved {line_count} pasted lines to {}",
                        path.display()
                    ));
                } else {
                    self.editor.attach_text("paste.md", &text);
                    self.push_system(
                        "Failed to save paste to a file — attached as a text chip instead",
                    );
                }
            }
            LargePastePick::Inline => self.editor.handle_paste(&text),
            LargePastePick::Cancelled => self.editor.attach_text("paste.md", &text),
        }
    }

    /// Recompute popup suggestions for the text before the cursor (opens,
    /// updates or closes the popup).
    fn refresh_autocomplete(&mut self) {
        let cur = self.editor.structured_cursor();
        if self
            .autocomplete
            .suggestions(self.editor.text_buffer().lines(), cur.line, cur.col)
            .is_none()
        {
            self.autocomplete.close();
        }
    }

    /// The Esc priority ladder (§6), P0 steps 4, 9, 12-14. First match wins.
    fn esc_ladder(&mut self, now: Instant, port: &impl RuntimePort) -> bool {
        // Step 4: maintenance advertised as "(esc to cancel)".
        if let Some(kind) = self.maintenance.take() {
            match kind {
                Maintenance::Retry => port.abort_retry(),
                Maintenance::Compaction => port.abort_compaction(),
            }
            let label = match kind {
                Maintenance::Retry => "retry",
                Maintenance::Compaction => "compaction",
            };
            self.push_system(&format!("[{label} aborted]"));
            return true;
        }

        // Step 9: a submission was accepted but not yet dispatched — cancel
        // it and restore all queued messages into the editor (draft
        // protection applies to queued text too, §5.3).
        if self.pending_submission {
            self.pending_submission = false;
            self.restore_queued_to_editor();
            return true;
        }

        // Step 12: abort the live run. TS `restoreQueuedMessagesToEditor({abort})`:
        // queued steering/follow-up is pulled out of the runtime queue back into
        // the editor first; otherwise the post-abort drain ran it under the user
        // interrupt and every queued message ended as "Request was aborted".
        if self.messages.is_streaming() || self.status.streaming {
            for text in self.steering.clone() {
                port.remove_queued(text, crate::action::QueueKind::Steering);
            }
            for text in self.follow_up.clone() {
                port.remove_queued(text, crate::action::QueueKind::FollowUp);
            }
            self.restore_queued_to_editor();
            self.dispatch(Action::Abort, port);
            return true;
        }

        // Step 13: never destroy a draft.
        if !self.editor.is_empty() {
            self.last_esc = None;
            return true;
        }

        // Step 14: double-Esc within 500 ms opens the rewind selector
        // (overlays-b P0 #4 + interactive P0 #4): the user-message column
        // along the live branch, fed by the journal accumulated from
        // `message_end` frames. Enter confirms → `Action::Rewind`.
        if self
            .last_esc
            .is_some_and(|t| now.duration_since(t) <= DOUBLE_TAP_WINDOW)
        {
            self.last_esc = None;
            if self.journal.is_empty() {
                self.push_system("nothing to rewind yet");
            } else {
                self.overlays
                    .open_tree(&self.journal.clone(), TreeMode::Rewind);
            }
        } else {
            self.last_esc = Some(now);
        }
        true
    }

    /// Pop the newest queued message back into the editor (§5.3).
    fn dequeue_to_editor(&mut self, port: &impl RuntimePort) {
        if let Some(text) = self.steering.pop() {
            self.dispatch(
                Action::RemoveQueued {
                    text: text.clone(),
                    queue: QueueKind::Steering,
                },
                port,
            );
            self.set_editor_text(&text);
            self.status.queued = self.steering.len() + self.follow_up.len();
        } else if let Some(text) = self.follow_up.pop() {
            self.dispatch(
                Action::RemoveQueued {
                    text: text.clone(),
                    queue: QueueKind::FollowUp,
                },
                port,
            );
            self.set_editor_text(&text);
            self.status.queued = self.steering.len() + self.follow_up.len();
        }
    }

    /// Concatenate every queued message into the editor (Esc during a
    /// pending submission, §5.3 `restoreQueuedMessagesToEditor`).
    fn restore_queued_to_editor(&mut self) {
        let mut all = std::mem::take(&mut self.steering);
        all.append(&mut self.follow_up);
        self.status.queued = 0;
        if all.is_empty() {
            return;
        }
        // Queued entries go ahead of the current draft (TS `#restoreEntriesToEditor`).
        let draft = self.editor.text();
        let mut joined = all.join("\n");
        if !draft.is_empty() {
            joined.push('\n');
            joined.push_str(&draft);
        }
        self.set_editor_text(&joined);
    }

    /// Replace the editor buffer with `text` (cursor to end).
    pub(super) fn set_editor_text(&mut self, text: &str) {
        let _ = self.editor.take();
        for c in text.chars() {
            self.editor.insert_char(c);
        }
    }

    pub(super) fn push_system(&mut self, text: &str) {
        self.messages.push_system(text);
    }

    /// Forward queued overlay intents (model picks, palette commands,
    /// session picks) to the port. The model browser stays open after an
    /// assignment (TS model hub); Esc closes it.
    fn drain_overlay_actions(&mut self, port: &impl RuntimePort) {
        let actions = self.overlays.take_actions();
        if actions.is_empty() {
            return;
        }
        for action in actions {
            self.dispatch(action, port);
        }
    }

    /// The submit pipeline (§4.1, P0 steps 1, 3, 4, 6, 7, 8, 11, 13).
    /// Returns `true` (always consumed).
    fn submit_pipeline(&mut self, text: &str, port: &impl RuntimePort) -> bool {
        // Step 1: trim. (Emoticon expansion is P1.)
        let text = text.trim().to_owned();

        if text.is_empty() {
            // Step 7 (confirm): a bare slash command armed with the first
            // Enter executes on the second, confirming, Enter.
            if let Some(name) = self.armed_bare_command.take() {
                return self.execute_slash(&name, "", port);
            }
            // Step 3: empty submit while streaming aborts the turn so the
            // post-unwind drain delivers queued messages (§5.4).
            if self.messages.is_streaming() || self.status.streaming {
                self.dispatch(Action::Abort, port);
            }
            return true;
        }

        // Step 4: continue shortcuts — a hidden synthetic directive, no
        // visible user row (TS `manualContinuePrompt`, port `continue_run`).
        if text == "." || text == "c" {
            self.dispatch(Action::Continue, port);
            self.pending_submission = true;
            return true;
        }

        // Step 6: queue shorthand — `>> rest` queues `rest` as follow-up.
        if let Some(rest) = text.strip_prefix(">>") {
            self.queue_message(rest.trim().to_owned(), SubmitBehavior::FollowUp, port);
            return true;
        }

        // Step 7+8: slash commands.
        if let Some(body) = text.strip_prefix('/') {
            return self.slash_pipeline(body, port);
        }

        // Step 11: local execution commands (`!cmd`, `>>>` python) — the
        // runtime has no executeBash/eval equivalent yet (deferred P1); the
        // shortcuts stay hidden behind a notice rather than misfiring as
        // free-text prompts.
        if text.starts_with('!') || text.starts_with(">>>") {
            self.push_system("local commands (! / >>>) are not available yet");
            return true;
        }

        // Step 13: free-text prompt; steered into the live run when streaming
        // (the optimistic queue chip shows in the band immediately, §5.2).
        self.queue_message(text, SubmitBehavior::Steer, port);
        true
    }

    /// `Some(Steer)` while streaming (the submit is steered, not queued).
    fn streaming_behavior(&self) -> Option<SubmitBehavior> {
        if self.messages.is_streaming() || self.status.streaming {
            Some(SubmitBehavior::Steer)
        } else {
            None
        }
    }

    /// Submit a user message. Idle: echo it into the chat immediately.
    /// Streaming: the behavior is forced to `steer`/`followUp` and the message
    /// shows only in the pending band (TS); it enters the chat when the runtime
    /// delivers it (`message_start` user). Echoing it here as well showed the
    /// prompt twice (chat + queue) and cut the live assistant block short.
    fn queue_message(&mut self, text: String, behavior: SubmitBehavior, port: &impl RuntimePort) {
        if self.streaming_behavior().is_none() {
            self.messages.push_user(&text);
        }
        match self.streaming_behavior() {
            Some(_) => {
                // Optimistic band update (TS refreshes at local call sites);
                // the next `queue_update` frame replaces it authoritatively.
                match behavior {
                    SubmitBehavior::Steer => self.steering.push(text.clone()),
                    SubmitBehavior::FollowUp => self.follow_up.push(text.clone()),
                }
                self.status.queued = self.steering.len() + self.follow_up.len();
                self.dispatch(
                    Action::Submit {
                        id: None,
                        text,
                        behavior: Some(behavior),
                    },
                    port,
                );
            }
            None => self.dispatch(
                Action::Submit {
                    id: None,
                    text,
                    behavior: None,
                },
                port,
            ),
        }
        self.pending_submission = true;
    }
}
