//! The root application state: owns every component, applies runtime
//! events through a serialized coalescing pump, routes input through the
//! submit pipeline and the Esc ladder, and lays out the live region.
//!
//! Layout (root vertical split, matching pi-tui's `resetDisplay()` order,
//! interactive.md §1):
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │ chat: committed messages + tool cards  │  fill (committed to
//! │                                        │   scrollback by ShellRender)
//! ├────────────────────────────────────────┤
//! │ queued band (Steering·n / After yield) │  0..n rows
//! │ ⠋ Working… (esc to cancel)             │  0..1 row
//! ├────────────────────────────────────────┤
//! │ ❯ prompt editor                        │  3 rows
//! ├────────────────────────────────────────┤
//! │ model  cwd  state · hints              │  1 row
//! └────────────────────────────────────────┘
//! ```
//!
//! Input routing: an open overlay owns keys first (`?` toggles help); then
//! the Esc ladder (maintenance cancel → pending-submission restore →
//! abort turn → draft protection → double-Esc, interactive.md §6), the
//! Ctrl+C double-tap / Ctrl+D exit discipline, `alt+up` dequeue, and the
//! submit pipeline (§4.1).
//!
//! Event flow: runtime frames enter through [`App::push_frame`] and are
//! applied by [`App::pump_events`] — dispatch is serialized through a
//! single queue and `message_update` snapshots are coalesced over a 33 ms
//! window (TS `EventController` `#MESSAGE_UPDATE_COALESCE_MS`), so the
//! newest cumulative snapshot wins.

use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::Event;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::Frame;
use serde_json::Value;

use crate::action::{Action, QueueKind, SubmitBehavior};
use crate::chat::{ChatRole, Messages};
use crate::component::Component;
use crate::editor::{Editor, EditorOutcome};
use crate::events::{parse_frame, AppEvent};
use crate::extension_ui::ExtensionUi;
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::overlays::{OpenOverlay, Overlays};
use crate::port::RuntimePort;
use crate::status_line::StatusLine;
use crate::theme::Theme;

/// `message_update` coalesce window (event-controller.ts L300).
const MESSAGE_UPDATE_COALESCE: Duration = Duration::from_millis(33);
/// Double-Esc / double-Ctrl+C window (input-controller L585, L1527).
const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(500);
/// Spinner frames for the "Working…" loader (TS loading animation set).
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// P0 builtin slash commands dispatched through `Action::Slash`
/// (§4.2 MVP cut: `exit help model effort compact clear context export`).
const P0_BUILTINS: &[&str] = &[
    "exit", "help", "model", "effort", "compact", "clear", "context", "export",
];
/// Bare-slash commands that require a confirming second Enter (§4.1 step 7).
const CONFIRM_BARE: &[&str] = &["exit", "clear", "compact", "export"];

/// Maintenance the runtime advertised as esc-to-cancel (§6 step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Maintenance {
    /// `auto_compaction_start` is in flight.
    Compaction,
    /// `auto_retry_start` is in flight.
    Retry,
}

/// Serialized frame pump: single queue, 33 ms `message_update` coalescing,
/// run epoch bumped at every `agent_start` (TS `#runSerialized`/`#runEpoch`).
#[derive(Debug, Default)]
pub struct EventPump {
    /// Frames waiting for their coalesce window / dispatch, in arrival order.
    queue: VecDeque<(Instant, AppEvent)>,
    /// The newest `message_update` snapshot held back for coalescing.
    pending_update: Option<(Instant, Value)>,
    /// Bumped at every `agent_start`; stale async watches stand down (TS epoch).
    pub epoch: u64,
}

impl EventPump {
    /// A fresh, empty pump.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue one raw frame. `message_update` snapshots replace any pending
    /// one (only the latest within the window is kept); any other frame
    /// flushes the held-back snapshot first so ordering is preserved.
    pub fn push_frame(&mut self, frame: &Value, now: Instant) {
        let event = parse_frame(frame);
        if let AppEvent::MessageUpdate(_) = event {
            self.pending_update = Some((now, frame.clone()));
            return;
        }
        // Ordering: any other frame flushes the held-back snapshot first,
        // even inside the coalesce window (the snapshot is cumulative, so
        // applying it before the newer event preserves stream order).
        if let Some((at, update)) = self.pending_update.take() {
            self.queue.push_back((at, AppEvent::MessageUpdate(update)));
        }
        self.queue.push_back((now, event));
    }

    /// Apply every due event to `app`, oldest first (serialized dispatch).
    /// A held-back `message_update` is applied once its window elapsed, or
    /// when a newer frame forces the flush inside [`Self::push_frame`].
    /// Returns the number of events applied.
    pub fn drain(&mut self, app: &mut App, now: Instant) -> usize {
        self.flush_pending_update(now);
        let mut applied = 0;
        while let Some((_, event)) = self.queue.pop_front() {
            if event == AppEvent::AgentStart {
                self.epoch += 1;
                app.run_epoch = self.epoch;
            }
            app.apply_event(&event);
            applied += 1;
        }
        applied
    }

    /// Whether a coalesced `message_update` is still waiting for its window.
    #[must_use]
    pub fn has_pending_update(&self) -> bool {
        self.pending_update.is_some()
    }

    /// Move the held-back snapshot into the dispatch queue if it is due.
    fn flush_pending_update(&mut self, now: Instant) {
        if let Some((at, frame)) = self.pending_update.take() {
            if now.saturating_duration_since(at) >= MESSAGE_UPDATE_COALESCE {
                self.queue.push_back((at, AppEvent::MessageUpdate(frame)));
            } else {
                self.pending_update = Some((at, frame));
            }
        }
    }
}
/// Root state; single owner, lives on the controller loop.
#[derive(Debug)]
pub struct App {
    messages: Messages,
    editor: Editor,
    status: StatusLine,
    overlays: Overlays,
    extension_ui: ExtensionUi,
    theme: Theme,
    /// Backlog scroll offset (PgUp/PgDn), reused once the shell renderer
    /// owns scrolling; kept for the full-screen fallback renderer.
    scroll: u16,
    /// The serialized, coalescing event pump (frames enter via
    /// [`Self::push_frame`], are applied by [`Self::pump_events`]).
    pump: EventPump,
    /// Run epoch, mirrored from the pump at every `agent_start`.
    pub run_epoch: u64,
    /// Steering queue previews (`queue_update.steering`), newest last.
    steering: Vec<String>,
    /// Follow-up queue previews (`queue_update.followUp`), newest last.
    follow_up: Vec<String>,
    /// Maintenance advertised as "(esc to cancel)" (Esc ladder step 4).
    maintenance: Option<Maintenance>,
    /// A submission was accepted locally but the run has not started yet
    /// (TS `loadingAnimation`; Esc ladder step 9).
    pending_submission: bool,
    /// Bare-slash command armed for a confirming second Enter (§4.1 step 7).
    armed_bare_command: Option<String>,
    /// Last Esc on an empty, idle editor (double-Esc, §6 step 14).
    last_esc: Option<Instant>,
    /// Last Ctrl+C (double-tap exit, §6).
    last_ctrl_c: Option<Instant>,
    /// `/exit` requested a quit; surfaced by the next submit return.
    quit_requested: bool,
}

impl App {
    /// Build the initial app from the run configuration.
    #[must_use]
    pub fn new(model: String, cwd: &Path) -> Self {
        let dir = cwd.file_name().map_or_else(
            || cwd.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        Self {
            messages: Messages::new(),
            editor: Editor::new(),
            status: StatusLine::new(model, dir),
            overlays: Overlays::default(),
            extension_ui: ExtensionUi::new(),
            theme: Theme::default(),
            scroll: 0,
            pump: EventPump::new(),
            run_epoch: 0,
            steering: Vec::new(),
            follow_up: Vec::new(),
            maintenance: None,
            pending_submission: false,
            armed_bare_command: None,
            last_esc: None,
            last_ctrl_c: None,
            quit_requested: false,
        }
    }

    /// Replace the app-wide theme (ThemeLoader startup + hot-swap).
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Enqueue one raw runtime frame for serialized, coalesced dispatch.
    pub fn push_frame(&mut self, frame: &Value, now: Instant) {
        self.pump.push_frame(frame, now);
    }

    /// Apply all due events. Returns `true` when anything was applied
    /// (the controller redraws after a pump that returns `true`).
    pub fn pump_events(&mut self, now: Instant) -> bool {
        // The pump lends itself out while draining to satisfy the borrow
        // checker (it calls back into `apply_event`).
        let mut pump = std::mem::take(&mut self.pump);
        let applied = pump.drain(self, now);
        self.pump = pump;
        applied > 0
    }

    /// Extension dialogs can time out; the controller calls this every
    /// tick. Expired dialogs are answered here (`cancelled` + `timedOut`).
    pub fn poll_deadlines(&mut self, now: Instant, port: &impl RuntimePort) {
        for (id, value) in self.extension_ui.poll_deadlines(now) {
            port.extension_ui_response(id, value);
        }
    }

    /// Whether a `message_update` is still inside its coalesce window; the
    /// controller uses this to schedule the next pump tick.
    #[must_use]
    pub fn has_pending_update(&self) -> bool {
        self.pump.has_pending_update()
    }

    /// Apply one classified runtime event directly (tests + compat path).
    pub fn apply_event(&mut self, event: &AppEvent) {
        match &event {
            AppEvent::AgentStart | AppEvent::TurnStart => {
                self.status.streaming = true;
                self.pending_submission = false;
            }
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) | AppEvent::SessionSettled => {
                self.status.streaming = false;
            }
            AppEvent::MessageStart(_) => self.pending_submission = false,
            AppEvent::ModelChanged(frame) => {
                if let Some(m) = frame
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        let p = frame.get("provider").and_then(Value::as_str)?;
                        let id = frame.get("id").and_then(Value::as_str)?;
                        Some(format!("{p}/{id}"))
                    })
                {
                    self.status.model = m;
                }
            }
            AppEvent::QueueUpdate {
                steering,
                follow_up,
            } => {
                self.steering = steering.clone();
                self.follow_up = follow_up.clone();
                self.status.queued = steering.len() + follow_up.len();
            }
            AppEvent::AutoCompaction(frame) => {
                let start =
                    frame.get("type").and_then(Value::as_str) == Some("auto_compaction_start");
                self.maintenance = if start {
                    Some(Maintenance::Compaction)
                } else {
                    None
                };
                self.messages.apply(event);
            }
            AppEvent::Retry(frame) => {
                if frame.get("type").and_then(Value::as_str) == Some("auto_retry_start") {
                    self.maintenance = Some(Maintenance::Retry);
                } else if self.maintenance == Some(Maintenance::Retry) {
                    self.maintenance = None;
                }
                self.messages.apply(event);
            }
            AppEvent::ExtensionUiRequest(frame) => self.extension_ui.push(frame),
            _ => {}
        }
        self.messages.apply(event);
    }

    /// Handle one terminal event. Returns `false` when the app should quit.
    pub fn handle_terminal_event(&mut self, event: &Event, port: &impl RuntimePort) -> bool {
        let Event::Key(key) = event else {
            return true; // resize/paste/mouse: redraw-only for now
        };
        self.handle_key(*key, port, Instant::now()).unwrap_or(false)
    }

    /// Dispatch one [`Action`] to the port (the single emit point).
    fn dispatch(&mut self, action: Action, port: &impl RuntimePort) {
        match action {
            Action::Submit { id, text, behavior } => port.submit(id, text, behavior),
            Action::Steer { text } => port.steer(text),
            Action::FollowUp { text } => port.follow_up(text),
            Action::RemoveQueued { text, queue } => port.remove_queued(text, queue),
            Action::PromoteQueued { text } => port.promote_queued(text),
            Action::Abort => port.abort(),
            Action::Continue => port.continue_run(),
            Action::ResumeSession { path, fork } => port.resume_session(path, fork),
            Action::Rewind { entry_id } => port.rewind_entry(entry_id),
            Action::SetModel { provider, id } => port.set_model(provider, id),
            Action::SetThinkingLevel(level) => port.set_thinking_level(level),
            Action::Slash { name, args } => port.run_slash(name, args),
            Action::ExtensionUiResponse { id, value } => port.extension_ui_response(id, value),
            Action::Scroll(dir) => self.apply_scroll(dir),
            Action::Quit => {}
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
    fn handle_key(&mut self, key: KeyEvent, port: &impl RuntimePort, now: Instant) -> Option<bool> {
        // Global keys, active even under overlays.
        match key.code {
            KeyCode::Char('?') => {
                self.overlays.toggle(OpenOverlay::Help);
                return Some(true);
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return None; // Ctrl+D: immediate exit (§6 handleCtrlD)
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.messages.toggle_expanded(); // app.tools.expand (chat.md)
                return Some(true);
            }
            _ => {}
        }

        // Overlay focus: route keys to the stack, top first. Esc closes the
        // top overlay only when it ignored the key (help/palette ignore esc).
        if self.overlays.wants_focus() {
            if key.code == KeyCode::Esc {
                if !self.overlays.handle_key(key).consumed() {
                    self.overlays.pop();
                }
                return Some(true);
            }
            let consumed = self.overlays.handle_key(key).consumed();
            self.drain_overlay_actions(port);
            return Some(consumed || self.overlays.wants_focus());
        }

        // Ctrl+C: first tap clears the editor draft, double-tap within
        // 500 ms exits (§6 handleCtrlC; exit code 130 comes from the CLI).
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
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

        // alt+up: restore the most recently queued message to the editor
        // (steering first, then follow-up; §5.3 handleDequeue).
        if key.code == KeyCode::Up && key.modifiers.contains(KeyModifiers::ALT) {
            self.dequeue_to_editor(port);
            return Some(true);
        }

        if key.code == KeyCode::Esc {
            return Some(self.esc_ladder(now, port));
        }

        match self.editor.handle_key(key) {
            EditorOutcome::Submit => {
                let text = self.editor.take();
                if !self.submit_pipeline(text, port) || self.quit_requested {
                    self.quit_requested = false;
                    return None;
                }
                Some(true)
            }
            EditorOutcome::Interrupt => Some(true), // ctrl+c handled above
            EditorOutcome::Edited => Some(true),
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
            self.push_system(format!("[{label} aborted]"));
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

        // Step 12: abort the live run; queued steering/follow-up drains in
        // the post-unwind turn (§5.4).
        if self.messages.is_streaming() || self.status.streaming {
            self.dispatch(Action::Abort, port);
            return true;
        }

        // Step 13: never destroy a draft.
        if !self.editor.is_empty() {
            self.last_esc = None;
            return true;
        }

        // Step 14: double-Esc within 500 ms runs the configured action
        // (default `rewind` — the rewind selector is P1, so this is a
        // repaint-only no-op for now, keeping the timing state correct).
        if self
            .last_esc
            .is_some_and(|t| now.duration_since(t) <= DOUBLE_TAP_WINDOW)
        {
            self.last_esc = None;
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
        let mut all = self.steering.clone();
        all.append(&mut self.follow_up.clone());
        self.follow_up.clear();
        self.status.queued = 0;
        if all.is_empty() {
            return;
        }
        let joined = all.join("\n");
        self.set_editor_text(&joined);
    }

    /// Replace the editor buffer with `text` (cursor to end).
    fn set_editor_text(&mut self, text: &str) {
        let _ = self.editor.take();
        for c in text.chars() {
            self.editor.insert_char(c);
        }
    }

    fn push_system(&mut self, text: String) {
        self.messages.push_system(&text);
    }

    /// Forward queued overlay intents (model picks, palette commands,
    /// session picks) to the port. A drained pick is terminal: the picker
    /// overlay closes (OverlayModels contract).
    fn drain_overlay_actions(&mut self, port: &impl RuntimePort) {
        let actions = self.overlays.take_actions();
        if actions.is_empty() {
            return;
        }
        if self.overlays.open() == Some(OpenOverlay::Models) {
            self.overlays.pop();
        }
        for action in actions {
            self.dispatch(action, port);
        }
    }

    /// The submit pipeline (§4.1, P0 steps 1, 3, 4, 6, 7, 8, 11, 13).
    /// Returns `true` (always consumed).
    fn submit_pipeline(&mut self, text: String, port: &impl RuntimePort) -> bool {
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
            self.push_system("local commands (! / >>>) are not available yet".into());
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

    /// Echo a user message and submit it; while streaming the behavior is
    /// forced to `steer`/`followUp` and the band is updated optimistically.
    fn queue_message(&mut self, text: String, behavior: SubmitBehavior, port: &impl RuntimePort) {
        self.messages.push_user(&text);
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

    /// Steps 7-8: bare-slash confirmation arming and builtin dispatch.
    fn slash_pipeline(&mut self, body: &str, port: &impl RuntimePort) -> bool {
        let mut parts = body.splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let args = parts.next().unwrap_or("").trim();

        if !P0_BUILTINS.contains(&name) {
            self.push_system(format!("unknown command: /{name}"));
            return true;
        }

        // Step 7: bare confirm-on-first-enter commands.
        if args.is_empty() && CONFIRM_BARE.contains(&name) {
            if self.armed_bare_command.as_deref() == Some(name) {
                self.armed_bare_command = None;
            } else {
                self.armed_bare_command = Some(name.to_owned());
                self.push_system(format!(
                    "Press Enter again to run /{name}; add a leading space to send \"/{name}\" as a message"
                ));
                return true;
            }
        }
        self.execute_slash(name, args, port)
    }

    /// Step 8: run a P0 builtin slash command (`exit` locally quits, `help`
    /// opens the keyhints overlay, the rest dispatch to the runtime's
    /// slash module via [`Action::Slash`]).
    fn execute_slash(&mut self, name: &str, args: &str, port: &impl RuntimePort) -> bool {
        self.armed_bare_command = None;
        match name {
            "exit" => self.quit_requested = true,
            "help" => self.overlays.toggle(OpenOverlay::Help),
            _ => self.dispatch(
                Action::Slash {
                    name: name.to_owned(),
                    args: args.to_owned(),
                },
                port,
            ),
        }
        true
    }

    // ---- Live-region layout (ShellRender commit/live API) ----------------

    /// Height of the queued band in rows: a leading spacer, one heading +
    /// one row per message per non-empty group, and the trailing key-hint
    /// row (§5.2). Zero when nothing is queued.
    #[must_use]
    pub fn band_height(&self) -> u16 {
        let groups =
            usize::from(!self.steering.is_empty()) + usize::from(!self.follow_up.is_empty());
        if groups == 0 {
            0
        } else {
            (1 + groups + self.steering.len() + self.follow_up.len() + 1) as u16
        }
    }

    /// Whether the "Working…" loader row is visible.
    #[must_use]
    pub fn show_loader(&self) -> bool {
        self.status.streaming || self.maintenance.is_some()
    }

    /// Height of the whole live region below the committed chat: band +
    /// loader + editor (3) + status line (1). `ShellRender` sizes the
    /// inline viewport with this.
    #[must_use]
    pub fn live_height(&self) -> u16 {
        self.band_height() + u16::from(self.show_loader()) + 3 + 1
    }

    /// Whether the top overlay is full-screen and the shell renderer must
    /// borrow the alternate screen (ShellRender contract).
    #[must_use]
    pub fn has_fullscreen_overlay(&self) -> bool {
        self.overlays.is_fullscreen()
    }

    /// Rows that settled since the last call and must enter terminal
    /// scrollback (`ShellRender::insert_before` source). Advances the chat
    /// watermark: committed entries are owned by the scrollback afterwards
    /// and are no longer drawn by [`Self::render_chat`]. With `rebuild =
    /// /// true` (width-change resize) returns the FULL transcript re-rendered
    /// at `width` and resets the watermark.
    pub fn take_scrollback_lines(&mut self, width: u16, rebuild: bool) -> Vec<Line<'static>> {
        if rebuild {
            self.messages.all_rows(width)
        } else {
            self.messages.take_committed_rows(width)
        }
    }

    /// Render the live region (band + loader + editor + status) into
    /// `area`; `ShellRender` calls this each frame inside the inline
    /// viewport, below the committed scrollback rows.
    pub fn render_live(&mut self, frame: &mut Frame<'_>, area: Rect) {
        self.status.tick(now_millis());
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(self.band_height()),
                Constraint::Length(u16::from(self.show_loader())),
                Constraint::Length(3),
                Constraint::Min(1),
            ])
            .split(area);
        self.render_band(frame, chunks[0]);
        self.render_loader(frame, chunks[1]);
        self.editor.render(frame, chunks[2], &self.theme);
        self.status.render(frame, chunks[3], &self.theme);
        self.overlays.render(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
    }

    /// Render the full screen (full-frame fallback renderer, used until the
    /// inline shell renderer replaces it; the fallback never commits, so
    /// the watermark stays 0 and the whole transcript renders).
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        self.status.tick(now_millis());
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(self.band_height()),
                Constraint::Length(u16::from(self.show_loader())),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(frame.area());
        self.render_chat(frame, chunks[0]);
        self.render_band(frame, chunks[1]);
        self.render_loader(frame, chunks[2]);
        self.editor.render(frame, chunks[3], &self.theme);
        self.status.render(frame, chunks[4], &self.theme);
        self.overlays.render(frame, frame.area(), &self.theme);
        self.extension_ui.render(frame, frame.area(), &self.theme);
    }

    /// The queued-messages band (§5.2 ANSI visuals): a spacer, per-group
    /// headings `Steering·n` / `After yield·n`, dim numbered rows, and a
    /// trailing `└ alt+up to edit` hint.
    fn render_band(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.height == 0 {
            return;
        }
        use ratatui::widgets::Paragraph;
        let muted = Style::default().fg(self.theme.color("text_muted"));
        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::default());
        for (label, msgs) in [
            ("Steering", &self.steering),
            ("After yield", &self.follow_up),
        ] {
            if msgs.is_empty() {
                continue;
            }
            lines.push(Line::from(Span::styled(
                format!("{label}·{}", msgs.len()),
                muted,
            )));
            for (i, m) in msgs.iter().enumerate() {
                let text: String = m.replace('\t', " ").lines().collect::<Vec<_>>().join(" ↵ ");
                lines.push(Line::from(Span::styled(
                    format!("  {}. {text}", i + 1),
                    Style::default().fg(self.theme.color("text_muted")),
                )));
            }
        }
        lines.push(Line::from(Span::styled(
            format!("  └ {} to edit", self.dequeue_hint()),
            muted,
        )));
        frame.render_widget(Paragraph::new(lines), area);
    }

    /// The `app.message.dequeue` binding, formatted for the band hint
    /// (TS `formatKeyHint`; the keybindings loader will make this dynamic).
    fn dequeue_hint(&self) -> &'static str {
        "alt+up"
    }

    /// The "Working…" loader row (spinner + optional maintenance label).
    fn render_loader(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.height == 0 {
            return;
        }
        use ratatui::widgets::Paragraph;
        let frame_idx = (Instant::now().elapsed().as_millis() / 80) as usize % SPINNER.len();
        let label = match self.maintenance {
            Some(Maintenance::Compaction) => " compacting… (esc to cancel)",
            Some(Maintenance::Retry) => " retrying… (esc to cancel)",
            None => " Working…",
        };
        let line = Line::from(Span::styled(
            format!("{} {label}", SPINNER[frame_idx]),
            Style::default().fg(self.theme.color("accent")),
        ));
        frame.render_widget(Paragraph::new(line), area);
    }

    /// One chat entry as a single committed row (`take_scrollback_lines`
    /// fallback rows).
    fn entry_line(&self, e: &crate::chat::ChatEntry, width: u16) -> Line<'static> {
        let theme = &self.theme;
        let (glyph, color) = match e.role {
            ChatRole::User => (theme.glyph("prompt"), theme.color("accent")),
            ChatRole::Assistant => ("", theme.color("text")),
            ChatRole::Tool => ("●", theme.color("warning")),
            ChatRole::System => ("!", theme.color("text_muted")),
        };
        let text: String = e.text.lines().collect::<Vec<_>>().join(" ⏎ ");
        let text = truncate(&text, width as usize);
        if glyph.is_empty() {
            Line::from(Span::styled(text, Style::default().fg(color)))
        } else {
            Line::from(vec![
                Span::styled(format!("{glyph} "), Style::default().fg(color)),
                Span::raw(text),
            ])
        }
    }

    /// The uncommitted chat tail: entries at/after the commit watermark
    /// only — committed rows live in terminal scrollback and must never be
    /// re-drawn here (ShellRender invariant).
    fn render_chat(&self, frame: &mut Frame<'_>, area: Rect) {
        use ratatui::widgets::Paragraph;
        let height = area.height as usize;
        let tail = self.messages.live_tail();
        let skip = tail.len().saturating_sub(height + self.scroll as usize);
        let take = tail.len().saturating_sub(skip);
        let lines: Vec<Line<'static>> = tail[skip..take]
            .iter()
            .map(|e| self.entry_line(e, area.width))
            .collect();
        frame.render_widget(Paragraph::new(lines), area);
    }
}

/// Wall-clock milliseconds (drives status-line animation ticks).
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Char-truncate to `width` display columns (approximation: chars, not
/// graphemes; TS uses `TruncatedText` with the same intent).
fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    text.chars()
        .take(width.saturating_sub(1))
        .collect::<String>()
        + "…"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::FakePort;
    use crossterm::event::{KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn app() -> App {
        App::new("zai/glm-5.3-flash".into(), Path::new("/repo"))
    }

    fn port() -> FakePort {
        FakePort::default()
    }

    fn key(code: CtKeyCode) -> KeyEvent {
        CtKeyEvent::new(code, CtMods::NONE)
    }

    fn ctrl(code: char) -> KeyEvent {
        CtKeyEvent::new(CtKeyCode::Char(code), CtMods::CONTROL)
    }

    fn type_text(app: &mut App, port: &FakePort, text: &str) {
        for c in text.chars() {
            assert!(app
                .handle_key(
                    KeyEvent::new(CtKeyCode::Char(c), CtMods::NONE),
                    port,
                    Instant::now()
                )
                .is_some());
        }
    }

    fn enter(app: &mut App, port: &FakePort) -> bool {
        app.handle_terminal_event(&Event::Key(key(CtKeyCode::Enter)), port)
    }

    fn press(app: &mut App, port: &FakePort, k: KeyEvent) -> bool {
        app.handle_terminal_event(&Event::Key(k), port)
    }

    fn log(port: &FakePort) -> Vec<String> {
        port.log.lock().clone()
    }

    fn start_streaming(app: &mut App) {
        app.apply_event(&AppEvent::AgentStart);
    }

    // ---- submit pipeline -------------------------------------------------

    #[test]
    fn typing_then_enter_submits_and_echoes() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "hi");
        assert!(enter(&mut app, &port));
        assert_eq!(app.messages.entries().len(), 1);
        assert_eq!(app.messages.entries()[0].role, ChatRole::User);
        assert_eq!(log(&port), vec!["submit(None, \"hi\", None)".to_string()]);
    }

    #[test]
    fn empty_enter_while_streaming_aborts_for_drain() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        assert!(enter(&mut app, &port));
        assert_eq!(log(&port), vec!["abort()".to_string()]);
    }

    #[test]
    fn dot_and_c_send_continue_directive() {
        for shortcut in [".", "c"] {
            let mut app = app();
            let port = port();
            type_text(&mut app, &port, shortcut);
            assert!(enter(&mut app, &port));
            assert_eq!(log(&port), vec!["continue_run()".to_string()]);
            assert!(app.messages.entries().is_empty()); // hidden, no user row
        }
    }

    #[test]
    fn queue_shorthand_follows_up() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        type_text(&mut app, &port, ">> later");
        assert!(enter(&mut app, &port));
        assert_eq!(
            log(&port),
            vec!["submit(None, \"later\", Some(FollowUp))".to_string()]
        );
        assert_eq!(app.follow_up, vec!["later".to_string()]);
        assert_eq!(app.band_height(), 1 + 1 + 1 + 1); // spacer+heading+msg+hint
    }

    #[test]
    fn steering_while_streaming_updates_band_optimistically() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        type_text(&mut app, &port, "faster");
        assert!(enter(&mut app, &port));
        assert_eq!(
            log(&port),
            vec!["submit(None, \"faster\", Some(Steer))".to_string()]
        );
        assert_eq!(app.steering, vec!["faster".to_string()]);
    }

    #[test]
    fn free_text_while_streaming_steers() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        type_text(&mut app, &port, "side note");
        assert!(enter(&mut app, &port));
        assert_eq!(
            log(&port),
            vec!["submit(None, \"side note\", Some(Steer))".to_string()]
        );
    }

    #[test]
    fn builtin_slash_dispatches_and_exit_quits() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/model zai/glm");
        assert!(enter(&mut app, &port));
        assert_eq!(
            log(&port),
            vec!["run_slash(model, \"zai/glm\")".to_string()]
        );
    }

    #[test]
    fn slash_exit_quits_after_bare_confirmation() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/exit");
        // Bare /exit must be confirmed with a second Enter (§4.1 step 7).
        // Bare /exit must be confirmed: first Enter arms, second quits.
        assert!(enter(&mut app, &port));
        assert!(!enter(&mut app, &port));
    }

    #[test]
    fn slash_help_opens_help_overlay() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/help");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Help));
    }

    #[test]
    fn unknown_slash_shows_notice_not_prompt() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/frobnicate");
        assert!(enter(&mut app, &port));
        assert!(log(&port).is_empty());
        assert!(app
            .messages
            .entries()
            .iter()
            .any(|e| e.text.contains("unknown command")));
    }

    #[test]
    fn local_command_shortcut_stays_hidden() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "!ls -la");
        assert!(enter(&mut app, &port));
        assert!(log(&port).is_empty());
        assert!(app
            .messages
            .entries()
            .iter()
            .any(|e| e.text.contains("not available yet")));
    }

    // ---- Esc ladder ------------------------------------------------------

    #[test]
    fn esc_aborts_streaming_turn() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(log(&port), vec!["abort()".to_string()]);
    }

    #[test]
    fn esc_never_destroys_a_draft() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "draft");
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(app.editor.text(), "draft");
        assert!(log(&port).is_empty());
    }

    #[test]
    fn esc_aborts_pending_submission_and_restores_queue() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        type_text(&mut app, &port, ">> queued one");
        assert!(enter(&mut app, &port));
        assert!(app.pending_submission);
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert!(!app.pending_submission);
        assert_eq!(app.editor.text(), "queued one");
        assert!(app.follow_up.is_empty());
    }

    #[test]
    fn esc_cancels_maintenance_retry() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::Retry(json!({"type":"auto_retry_start"})));
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(log(&port), vec!["abort_retry()".to_string()]);
        assert!(app
            .messages
            .entries()
            .iter()
            .any(|e| e.text.contains("retry aborted")));
    }

    // ---- Ctrl+C / Ctrl+D -------------------------------------------------

    #[test]
    fn ctrl_c_double_tap_quits_single_tap_clears_editor() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "draft");
        let t0 = Instant::now();
        assert_eq!(app.handle_key(ctrl('c'), &port, t0), Some(true));
        assert!(app.editor.is_empty());
        assert_eq!(
            app.handle_key(ctrl('c'), &port, t0 + Duration::from_millis(100)),
            None
        );
    }

    #[test]
    fn ctrl_c_taps_beyond_window_do_not_quit() {
        let mut app = app();
        let port = port();
        let t0 = Instant::now();
        assert!(app.handle_key(ctrl('c'), &port, t0).is_some());
        assert!(app
            .handle_key(ctrl('c'), &port, t0 + Duration::from_millis(600))
            .is_some());
    }

    #[test]
    fn ctrl_d_quits_immediately() {
        let mut app = app();
        let port = port();
        assert!(!press(&mut app, &port, ctrl('d')));
    }

    // ---- dequeue ----------------------------------------------------------

    #[test]
    fn alt_up_dequeues_newest_message_to_editor() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::QueueUpdate {
            steering: vec!["s1".into()],
            follow_up: vec!["f1".into(), "f2".into()],
        });
        let alt_up = CtKeyEvent::new(CtKeyCode::Up, CtMods::ALT);
        // Steering pops before follow-up (§5.3): newest steering first.
        assert!(press(&mut app, &port, alt_up));
        assert_eq!(app.editor.text(), "s1");
        assert!(app.steering.is_empty());
        assert_eq!(
            log(&port),
            vec!["remove_queued(\"s1\", Steering)".to_string()]
        );
        assert!(press(&mut app, &port, alt_up));
        assert_eq!(app.editor.text(), "f2"); // then the newest follow-up
        assert_eq!(app.follow_up, vec!["f1".to_string()]);
    }

    // ---- event pump -------------------------------------------------------

    #[test]
    fn pump_coalesces_message_updates_within_33ms() {
        let mut app = app();
        let t0 = Instant::now();
        let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
        app.push_frame(&upd("hel"), t0);
        app.push_frame(&upd("hello"), t0 + Duration::from_millis(5));
        assert!(!app.pump_events(t0 + Duration::from_millis(10)));
        assert!(app.messages.entries().is_empty());
        assert!(app.pump_events(t0 + Duration::from_millis(40)));
        assert_eq!(app.messages.entries().len(), 1);
        assert_eq!(app.messages.entries()[0].text, "hello");
    }

    #[test]
    fn pump_holds_updates_until_window_elapses() {
        let mut app = app();
        let t0 = Instant::now();
        let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
        app.push_frame(&upd("one"), t0);
        assert!(!app.pump_events(t0 + Duration::from_millis(10)));
        assert!(app.messages.entries().is_empty());
        assert!(app.pump_events(t0 + Duration::from_millis(40)));
        assert_eq!(app.messages.entries()[0].text, "one");
    }

    #[test]
    fn pump_bumps_epoch_at_agent_start() {
        let mut app = app();
        let t0 = Instant::now();
        app.push_frame(&json!({"type":"agent_start"}), t0);
        app.push_frame(&json!({"type":"agent_end"}), t0);
        app.pump_events(t0);
        assert_eq!(app.run_epoch, 1);
    }

    #[test]
    fn non_update_frame_flushes_pending_update_first() {
        let mut app = app();
        let t0 = Instant::now();
        let upd = |t: &str| json!({"type":"message_update","message":{"role":"assistant","content":[{"type":"text","text":t}]}});
        app.push_frame(&upd("hi"), t0);
        app.push_frame(&json!({"type":"agent_end"}), t0);
        app.pump_events(t0);
        assert_eq!(app.messages.entries()[0].text, "hi");
        assert!(!app.status.streaming);
    }

    // ---- rendering --------------------------------------------------------

    #[test]
    fn renders_chat_editor_status() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "hello");
        assert!(enter(&mut app, &port));
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let bottom: String = (0..80).map(|x| buf[(x, 23)].symbol()).collect();
        assert!(bottom.contains("zai/glm-5.3-flash"));
        let row: String = (0..80).map(|x| buf[(x, 21)].symbol()).collect();
        assert!(row.contains('❯'));
    }

    #[test]
    fn working_loader_shows_while_streaming() {
        let mut app = app();
        let _port = port();
        start_streaming(&mut app);
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        // rows: 10 - 3 editor - 1 status - 1 loader = 5 chat
        let row: String = (0..80).map(|x| buf[(x, 5)].symbol()).collect();
        assert!(row.contains("Working…"), "row: {row:?}");
    }

    #[test]
    fn queued_band_renders_groups_and_hint() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::QueueUpdate {
            steering: vec!["left".into()],
            follow_up: vec!["after".into()],
        });
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let band: Vec<String> = (15..22)
            .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect())
            .collect();
        let text = band.join("\n");
        assert!(text.contains("Steering·1"), "{text}");
        assert!(text.contains("After yield·1"), "{text}");
        assert!(text.contains("alt+up to edit"), "{text}");
    }

    #[test]
    fn maintenance_loader_advertises_esc_to_cancel() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::AutoCompaction(
            json!({"type":"auto_compaction_start","reason":"threshold"}),
        ));
        let backend = TestBackend::new(80, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        // 6 - 3 editor - 1 status - 1 loader = 1 chat row; loader at row 1
        let row: String = (0..80).map(|x| buf[(x, 1)].symbol()).collect();
        assert!(row.contains("esc to cancel"), "row: {row:?}");
    }

    #[test]
    fn take_scrollback_lines_advances_watermark() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "done");
        assert!(enter(&mut app, &port));
        let committed = app.messages.committed_len();
        let lines = app.take_scrollback_lines(80, false);
        assert_eq!(lines.len(), app.messages.committed_len() - committed);
        // The live tail must exclude everything committed so far.
        assert_eq!(
            app.messages.live_tail().len() + app.messages.committed_len(),
            app.messages.entries().len()
        );
        // Rebuild returns the full transcript and re-bases the watermark.
        let full = app.take_scrollback_lines(80, true);
        assert_eq!(full.len(), app.messages.entries().len());
    }

    #[test]
    fn render_live_draws_editor_and_status() {
        let mut app = app();
        let _port = port();
        let backend = TestBackend::new(80, 5);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| app.render_live(f, Rect::new(0, 0, 80, 5)))
            .unwrap();
        let buf = terminal.backend().buffer();
        // The editor renders as a boxed component: border row, then the
        // prompt glyph on the first inner row.
        let editor_row: String = (0..80).map(|x| buf[(x, 1)].symbol()).collect();
        assert!(editor_row.contains('❯'), "row: {editor_row:?}");
        let status_row: String = (0..80).map(|x| buf[(x, 3)].symbol()).collect();
        assert!(
            status_row.contains("zai/glm-5.3-flash"),
            "row: {status_row:?}"
        );
    }
}
