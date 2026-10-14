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
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::Event;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::Frame;
use serde_json::Value;

use crate::action::{Action, QueueKind, SubmitBehavior};
use crate::autocomplete::Autocomplete;
use crate::chat::Messages;
use crate::chrome::notify::{send_notification, NotifyProtocol, TerminalNotification};
use crate::chrome::title::{TerminalTitle, TitleState};
use crate::component::Component;
use crate::editor::{Editor, EditorOutcome};
use crate::events::{parse_frame, AppEvent};
use crate::extension_ui::ExtensionUi;
use crate::keybindings::KeybindingsManager;
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::overlays::models::{parse_model_string, ModelEntry};
use crate::overlays::tree::TreeMode;
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
/// (§4.2 MVP cut: `exit help model effort compact clear context export`
/// plus the overlay-backed `resume`/`tree`/`theme`/`settings`/`hotkeys`).
const P0_BUILTINS: &[&str] = &[
    "exit", "help", "hotkeys", "model", "effort", "compact", "clear", "context", "export",
    "resume", "tree", "theme", "settings",
];
/// Bare-slash commands that require a confirming second Enter (§4.1 step 7).
const CONFIRM_BARE: &[&str] = &["exit", "clear", "compact", "export"];

/// TUI-local slash entries merged into the autocomplete catalog: commands
/// the app handles itself and the runtime catalog does not announce (TS
/// `/settings` is registered by the interactive host, not the agent).
const LOCAL_COMMANDS: &[(&str, &str)] = &[
    ("settings", "Browse and edit settings"),
    ("hotkeys", "Show keybindings (alias of /help)"),
];

/// Maintenance the runtime advertised as esc-to-cancel (§6 step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Maintenance {
    /// `auto_compaction_start` is in flight.
    Compaction,
    /// `auto_retry_start` is in flight.
    Retry,
}

/// Turn-settle notification dispatcher (chrome.md §13 P0: Bell + OSC +
/// notify-send fallback honoring `PI_NO_DESKTOP_NOTIFY`).
///
/// Every fired notification is recorded (headless-observable for tests);
/// delivery goes through [`send_notification`] unless the host installs a
/// hook (tests capture instead of writing escape sequences / spawning
/// `notify-send`).
#[derive(Clone, Default)]
pub struct NotifySink {
    hook: Option<Arc<dyn Fn(&TerminalNotification) + Send + Sync>>,
    record: Arc<parking_lot::Mutex<Vec<String>>>,
}

impl fmt::Debug for NotifySink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NotifySink")
            .field("hook", &self.hook.is_some())
            .field("record", &self.record.lock().clone())
            .finish()
    }
}

impl NotifySink {
    /// Capture deliveries through `hook` instead of the real terminal path.
    pub fn set_hook(&mut self, hook: Arc<dyn Fn(&TerminalNotification) + Send + Sync>) {
        self.hook = Some(hook);
    }

    /// Record + deliver one notification.
    pub fn fire(&self, n: &TerminalNotification) {
        use crate::chrome::notify::notification_to_line;
        self.record.lock().push(notification_to_line(n));
        match &self.hook {
            Some(hook) => hook(n),
            None => send_notification(NotifyProtocol::Bell, n),
        }
    }

    /// Recorded `notification_to_line` strings, in fire order.
    #[must_use]
    pub fn record(&self) -> Vec<String> {
        self.record.lock().clone()
    }
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
    /// Slash / `@file` / path completion popup above the editor.
    autocomplete: Autocomplete,
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
    /// Resolved keybindings (defaults + `~/.omp/agent/keybindings.yml`);
    /// the app input path matches against these instead of hard-coded
    /// chords (interactive P0 #15 / review2 §11 #15).
    keys: KeybindingsManager,
    /// Terminal title run-state owner (chrome P0 §8): Working on
    /// `agent_start`, Idle on settle; disposed with the app.
    title: TerminalTitle,
    /// Title label: session name once known, else the cwd basename.
    title_label: String,
    /// Turn-settle notifications (chrome P0 notifications).
    notify: NotifySink,
    /// Raw journal entries (`message_end` frames normalized to
    /// `{type:"message",id,parentId,message}`) feeding the rewind column /
    /// tree overlay; reset when a session is resumed.
    journal: Vec<Value>,
    /// The current folder's session dir (`/resume` embedded picker source).
    sessions_dir: PathBuf,
    /// Terminal appearance from the startup OSC 11 / COLORFGBG probe:
    /// decides whether a `theme.dark`/`theme.light` edit hot-swaps now.
    appearance: crate::theme::Appearance,
    /// Working directory the session runs in (settings re-reads on edits).
    cwd: PathBuf,
}

impl App {
    /// Build the initial app from the run configuration.
    #[must_use]
    pub fn new(model: String, cwd: &Path) -> Self {
        let dir = cwd.file_name().map_or_else(
            || cwd.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let title = TerminalTitle::new();
        title.init(TitleState::Idle, Some(&dir));
        Self {
            messages: Messages::new(),
            editor: Editor::new(),
            autocomplete: Autocomplete::new(cwd.to_path_buf()),
            status: StatusLine::new(model, dir.clone()),
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
            keys: KeybindingsManager::defaults(),
            title,
            title_label: dir,
            notify: NotifySink::default(),
            journal: Vec::new(),
            sessions_dir: PathBuf::new(),
            appearance: crate::theme::Appearance::Dark,
            cwd: cwd.to_path_buf(),
        }
    }

    /// Record the terminal appearance resolved at startup (OSC 11 /
    /// COLORFGBG); `/settings` theme edits hot-swap the matching variant.
    pub fn set_appearance(&mut self, appearance: crate::theme::Appearance) {
        self.appearance = appearance;
    }

    /// Replace the app-wide theme (ThemeLoader startup + hot-swap). The
    /// transcript renderer is synced here — not per frame — so a hot-swap
    /// lands everywhere on the next paint (tui-ChatRenderWire caveat).
    pub fn set_theme(&mut self, theme: Theme) {
        self.messages.set_theme(theme.clone());
        self.theme = theme;
    }

    /// Seed the `/model` picker catalogue + the embedded `/resume` session
    /// dir from the run configuration (called once by the controller).
    pub fn configure(&mut self, models: Vec<ModelEntry>, sessions_dir: PathBuf) {
        let current = Some(self.status.model.clone());
        let picker = self.overlays.models_mut();
        picker.models = models;
        picker.current = current;
        self.sessions_dir = sessions_dir;
    }

    /// Load user key overrides from `~/.omp/agent/keybindings.yml`
    /// (`OMP_AGENT_DIR` aware). Returns a human-readable error when the file
    /// exists but cannot be parsed (defaults stay in effect).
    pub fn load_user_keybindings(&mut self) -> Option<String> {
        let path = crate::keybindings::user_keybindings_path();
        match crate::keybindings::load_user_bindings(&path) {
            Ok(user) => {
                let conflicts = {
                    let km = KeybindingsManager::new(user.clone());
                    km.get_conflicts().len()
                };
                self.keys = KeybindingsManager::new(user);
                (conflicts > 0).then(|| {
                    format!("keybindings: {conflicts} conflicting override(s) in {path:?}; first binding wins on the input path")
                })
            }
            Err(e) => Some(format!("keybindings: {e}; defaults in effect")),
        }
    }

    /// The terminal title owner (tests + controller dispose).
    #[must_use]
    pub fn title(&self) -> &TerminalTitle {
        &self.title
    }

    /// Stop the title spinner and latch teardown (controller calls this on
    /// exit; [`TerminalTitle`]'s `Drop` also covers panic paths).
    pub fn dispose_title(&mut self) {
        self.title.dispose();
    }

    /// The turn-settle notification sink (delivery hook + fired record).
    #[must_use]
    pub fn notify_sink(&self) -> &NotifySink {
        &self.notify
    }

    /// Install a custom notification delivery hook (tests).
    pub fn set_notify_hook(&mut self, hook: Arc<dyn Fn(&TerminalNotification) + Send + Sync>) {
        self.notify.set_hook(hook);
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
            AppEvent::AgentStart => {
                self.status.streaming = true;
                self.pending_submission = false;
                self.title.set_state(TitleState::Working, Some(&self.title_label));
            }
            AppEvent::TurnStart => {
                self.status.streaming = true;
                self.pending_submission = false;
            }
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) => {
                self.status.streaming = false;
                self.title.set_state(TitleState::Idle, Some(&self.title_label));
            }
            AppEvent::SessionSettled => {
                self.status.streaming = false;
                self.title.set_state(TitleState::Idle, Some(&self.title_label));
                self.notify_turn_settled(None);
            }
            AppEvent::PromptResult {
                success, frame, ..
            } => {
                if !success {
                    let error = frame
                        .get("error")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                    // Surface the failure in the transcript (e.g. an unknown
                    // or unported slash command), not only as a notification.
                    if let Some(e) = &error {
                        self.push_system(format!("error: {e}"));
                    }
                    self.title.set_state(TitleState::Idle, Some(&self.title_label));
                    self.notify_turn_settled(error);
                }
            }
            AppEvent::MessageEnd(frame) => {
                if let Some(entry) = journal_entry(frame) {
                    self.journal.push(entry);
                }
            }
            AppEvent::SessionNameChanged(name) => {
                self.title_label = name.clone();
                // Re-emit the current state with the new label.
                let state = if self.status.streaming {
                    TitleState::Working
                } else {
                    TitleState::Idle
                };
                self.title.set_state(state, Some(name));
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
                    self.status.model = m.clone();
                    self.overlays.models_mut().current = Some(m);
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
            AppEvent::AvailableCommands(frame) => {
                self.autocomplete.update_commands(frame);
                // Merge the TUI-local entries (`/settings`, `/hotkeys`) into
                // the announced catalog so the `/` popup offers them even
                // though the runtime does not register them (TS registers
                // both host-side).
                let mut commands = self.autocomplete.commands().to_vec();
                for (name, description) in LOCAL_COMMANDS {
                    if !commands.iter().any(|cmd| cmd.name == *name) {
                        commands.push(crate::autocomplete::SlashCommand {
                            name: (*name).to_string(),
                            description: Some((*description).to_string()),
                            source: Some(crate::autocomplete::CommandSource::Builtin),
                            ..crate::autocomplete::SlashCommand::default()
                        });
                    }
                }
                self.autocomplete.set_commands(commands);
            }
            AppEvent::CommandOutput(text) => self.push_system(text.clone()),
            _ => {}
        }
        self.messages.apply(event);
    }

    /// Fire the turn-settle notification (chrome P0): error body on failure,
    /// plain completion on success. Delivery honors `PI_NO_DESKTOP_NOTIFY`
    /// inside [`send_notification`].
    fn notify_turn_settled(&mut self, error: Option<String>) {
        let body = match error {
            Some(e) => format!("Turn failed: {e}"),
            None => "Turn complete".to_owned(),
        };
        let mut n = TerminalNotification::body(body);
        n.title = Some("omp".to_owned());
        self.notify.fire(&n);
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
                    self.push_system(format!("settings: failed to save {id}: {e}"));
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
                    self.editor
                        .apply_completion(applied.lines, applied.cursor_line, applied.cursor_col);
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

        if self.keys.matches_event(&key, "app.interrupt") {
            return Some(self.esc_ladder(now, port));
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
        if !self.submit_pipeline(text, port) || self.quit_requested {
            self.quit_requested = false;
            return None;
        }
        Some(true)
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
                self.push_system("nothing to rewind yet".into());
            } else {
                self.overlays.open_tree(&self.journal.clone(), TreeMode::Rewind);
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
        if self.overlays.open() == Some(OpenOverlay::Models)
            && actions.iter().any(|a| {
                matches!(
                    a,
                    Action::SetModel { .. } | Action::SetThinkingLevel(_)
                )
            })
        {
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

        // Local P0 builtins plus anything the runtime announced in its
        // catalog (builtins, extension and file commands, incl. aliases) go
        // through dispatch; the runtime reports commands it cannot run.
        let in_catalog = self
            .autocomplete
            .commands()
            .iter()
            .any(|c| c.name == name || c.aliases.iter().any(|a| a == name));
        if !P0_BUILTINS.contains(&name) && !in_catalog {
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
    /// opens the keyhints overlay, `model`/`resume`/`tree` open their
    /// overlays, `theme` hot-swaps locally, the rest dispatch to the
    /// runtime's slash module via [`Action::Slash`]).
    fn execute_slash(&mut self, name: &str, args: &str, port: &impl RuntimePort) -> bool {
        self.armed_bare_command = None;
        match name {
            "exit" => self.quit_requested = true,
            "help" | "hotkeys" => self.overlays.toggle(OpenOverlay::Help),
            "settings" => self.slash_settings(port),
            "model" => self.slash_model(args, port),
            "resume" => self.slash_resume(),
            "tree" => {
                if self.journal.is_empty() {
                    self.push_system("no session tree yet".into());
                } else {
                    self.overlays.open_tree(&self.journal.clone(), TreeMode::Tree);
                }
            }
            "theme" => self.slash_theme(args),
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

    /// `/model`: bare or query form opens the picker (overlays-b P0 #1);
    /// an exact `provider/id` that exists in the catalogue still goes
    /// through the runtime slash dispatch (TS `setModel` path). Picking a
    /// row drains `Action::SetModel` → `port.set_model`.
    fn slash_model(&mut self, args: &str, port: &impl RuntimePort) {
        let exact = parse_model_string(args).is_some_and(|(provider, id, _)| {
            let selector = format!("{provider}/{id}").to_lowercase();
            self.overlays
                .models_mut()
                .models
                .iter()
                .any(|m| m.selector().to_lowercase() == selector)
        });
        if exact {
            self.dispatch(
                Action::Slash {
                    name: "model".to_owned(),
                    args: args.to_owned(),
                },
                port,
            );
            return;
        }
        // Picker path: open (or re-open) and pre-fill the search query.
        if self.overlays.open() == Some(OpenOverlay::Models) {
            self.overlays.pop();
        }
        self.overlays.toggle(OpenOverlay::Models);
        if !args.is_empty() {
            let picker = self.overlays.models_mut();
            picker.query = args.to_owned();
            picker.selected = 0;
        }
    }

    /// `/settings`: open the tabbed settings overlay, seeded with the
    /// effective values from the port (TS opens the settings selector with a
    /// `cfg://` snapshot). The overlay stays open; edits drain as
    /// `Action::SettingChanged`.
    fn slash_settings(&mut self, port: &impl RuntimePort) {
        self.overlays.open_settings();
        let overlay = self.overlays.settings_mut();
        let ids = overlay.ids();
        let values = port.settings_get(&ids);
        overlay.seed(&values);
    }

    /// `/resume`: open the embedded session picker (overlays-b P0 #10); the
    /// pick resolves to `Action::ResumeSession` via the overlay drain.
    fn slash_resume(&mut self) {
        if self.sessions_dir.as_os_str().is_empty() {
            self.push_system("session storage is not available in this session".into());
            return;
        }
        self.overlays.open_sessions(&self.sessions_dir);
    }

    /// Live half of a persisted settings edit: apply what omp-rs can right
    /// now (TS applies everything through the cfg:// host; here only theme
    /// and thinking level have live consumers, everything else is saved for
    /// the next runtime read and flagged "no live effect" in the overlay).
    fn apply_setting_live(&mut self, id: &str, value: &Value, port: &impl RuntimePort) {
        match id {
            "theme.dark" | "theme.light" => {
                // Re-read the layered settings and hot-swap the theme for
                // the terminal's current appearance.
                let settings = omp_settings::Settings::load(&self.cwd);
                let dark = settings.get_string("theme.dark");
                let light = settings.get_string("theme.light");
                match crate::theme::load_configured(
                    dark.as_deref(),
                    light.as_deref(),
                    self.appearance,
                ) {
                    Ok(theme) => {
                        self.set_theme(theme);
                        self.push_system(format!("{id} applied"));
                    }
                    Err(e) => self.push_system(format!("{id} saved, but theme reload failed: {e}")),
                }
            }
            "defaultThinkingLevel" => {
                if let Some(level) = value.as_str().filter(|level| *level != "auto") {
                    port.set_thinking_level(level.to_string());
                }
            }
            _ => {}
        }
    }

    /// `/theme`: bare lists the loadable themes; with a name it hot-swaps
    /// the app theme immediately (chrome.md §7, settings hot-swap half).
    fn slash_theme(&mut self, args: &str) {
        if args.is_empty() {
            let names: Vec<String> = crate::theme::available_themes()
                .into_iter()
                .map(|t| t.name)
                .collect();
            self.push_system(format!("available themes: {}", names.join(", ")));
            return;
        }
        match crate::theme::load(args) {
            Ok(theme) => {
                self.set_theme(theme);
                self.push_system(format!("theme set to {args}"));
            }
            Err(e) => self.push_system(format!("theme error: {e}")),
        }
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
        self.render_autocomplete(frame, area, chunks[2]);
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
        self.render_autocomplete(frame, frame.area(), chunks[3]);
        self.overlays.render(frame, frame.area(), &self.theme);
        self.extension_ui.render(frame, frame.area(), &self.theme);
    }

    /// Draw the completion popup bottom-anchored just above the editor box.
    fn render_autocomplete(&self, frame: &mut Frame<'_>, area: Rect, editor: Rect) {
        let above = Rect {
            x: editor.x,
            y: area.y,
            width: editor.width,
            height: editor.y.saturating_sub(area.y),
        };
        self.autocomplete.render(frame, above, &self.theme);
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
    /// (TS `formatKeyHint`): first resolved key, static fallback.
    fn dequeue_hint(&self) -> String {
        self.keys
            .first_key("app.message.dequeue")
            .unwrap_or_else(|| "alt+up".to_owned())
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

    /// The uncommitted chat tail: entries at/after the commit watermark
    /// only — committed rows live in terminal scrollback and must never be
    /// re-drawn here (ShellRender invariant). Rows come from the chat block
    /// renderer (`chat::blocks`); the streaming reveal controller is ticked
    /// once per frame here in the draw loop (chat.md §8).
    fn render_chat(&mut self, frame: &mut Frame<'_>, area: Rect) {
        use ratatui::widgets::Paragraph;
        // The transcript renderer's theme is synced by `set_theme` only
        // (startup load + `/theme` hot-swap), not per frame.
        self.messages.tick_reveal();
        let height = area.height as usize;
        let base = self.messages.committed_len();
        let tail_len = self.messages.entries().len() - base;
        let width = usize::from(area.width).max(8);
        let mut lines: Vec<Line<'static>> = Vec::new();
        for idx in base..base + tail_len {
            lines.extend(self.messages.entry_lines_at(idx, width));
        }
        let skip = lines.len().saturating_sub(height + self.scroll as usize);
        frame.render_widget(Paragraph::new(lines[skip..].to_vec()), area);
    }
}

/// Wall-clock milliseconds (drives status-line animation ticks).
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Normalize a `message_end` frame into the journal shape the session tree
/// expects (`{type:"message", id, parentId?, message}`; omp-session `tree`
/// helpers read exactly these fields). `None` when the message has no id.
fn journal_entry(frame: &Value) -> Option<Value> {
    let message = frame.get("message")?;
    let id = message.get("id").and_then(Value::as_str)?;
    let mut entry = serde_json::json!({
        "type": "message",
        "id": id,
        "message": message,
    });
    if let Some(parent) = message.get("parentId").and_then(Value::as_str) {
        entry["parentId"] = Value::from(parent);
    }
    Some(entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatRole;
    use crate::port::FakePort;
    use crossterm::event::{KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;
    use std::sync::LazyLock;

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
    fn slash_model_exact_id_dispatches_to_runtime() {
        let mut app = app();
        let port = port();
        app.configure(
            vec![ModelEntry::new("zai", "glm", Some(128_000))],
            PathBuf::new(),
        );
        type_text(&mut app, &port, "/model zai/glm");
        assert!(enter(&mut app, &port));
        // An exact catalogue id goes through the runtime slash dispatch.
        assert_eq!(
            log(&port),
            vec!["run_slash(model, \"zai/glm\")".to_string()]
        );
        assert_eq!(app.overlays.open(), None);
    }

    #[test]
    fn slash_model_bare_or_query_opens_picker() {
        for args in ["", "glm"] {
            let mut app = app();
            let port = port();
            app.configure(
                vec![ModelEntry::new("zai", "glm", Some(128_000))],
                PathBuf::new(),
            );
            type_text(&mut app, &port, &format!("/model {args}"));
            assert!(enter(&mut app, &port), "args: {args:?}");
            assert_eq!(
                app.overlays.open(),
                Some(OpenOverlay::Models),
                "args: {args:?}"
            );
            assert!(log(&port).is_empty(), "no port calls for picker");
        }
    }

    #[test]
    fn model_picker_pick_dispatches_set_model_and_closes() {
        let mut app = app();
        let port = port();
        app.configure(
            vec![ModelEntry::new("zai", "glm", Some(128_000))],
            PathBuf::new(),
        );
        type_text(&mut app, &port, "/model");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Models));
        // Enter on the highlighted row drains SetModel → port.set_model.
        assert!(enter(&mut app, &port));
        assert_eq!(log(&port), vec!["set_model(zai, glm)".to_string()]);
        assert_eq!(app.overlays.open(), None);
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
    fn slash_hotkeys_opens_help_overlay() {
        // TS alias: `/hotkeys` is `/help`.
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/hotkeys");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Help));
    }

    #[test]
    fn slash_settings_opens_overlay_listing_sections() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
        let s = screen(&mut app, 150, 30);
        // Tab strip and the first tab's section headings.
        assert!(s.contains("appearance"), "{s}");
        assert!(s.contains("model") && s.contains("providers"), "{s}");
        assert!(s.contains("─ Theme "), "{s}");
        // Current values render next to labels.
        assert!(s.contains("dark"), "{s}");
    }

    #[test]
    fn settings_popup_lists_local_commands() {
        let mut app = app();
        let port = port();
        with_catalog(&mut app);
        type_text(&mut app, &port, "/se");
        let s = screen(&mut app, 60, 14);
        assert!(s.contains("settings"), "{s}");
        // New query: drop the old draft and type the other local command.
        for _ in 0..3 {
            assert!(press(&mut app, &port, key(CtKeyCode::Backspace)));
        }
        type_text(&mut app, &port, "/hot");
        let s = screen(&mut app, 60, 14);
        assert!(s.contains("hotkeys"), "{s}");
    }

    /// Serialize the HOME-mutating settings tests (env is process-global).
    static HOME_LOCK: LazyLock<parking_lot::Mutex<()>> = LazyLock::new(|| parking_lot::Mutex::new(()));

    fn with_temp_home(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omp-tui-settings-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("HOME", &dir);
        std::env::remove_var("OMP_PROFILE");
        dir
    }

    fn restore_home(old: Option<std::ffi::OsString>) {
        match old {
            Some(h) => std::env::set_var("HOME", h),
            None => std::env::remove_var("HOME"),
        }
    }

    #[test]
    fn settings_bool_toggle_writes_config_and_rereads() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let home = with_temp_home("bool");
        {
            let mut app = app();
            let port = port();
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            // Defaults before the edit.
            assert_eq!(
                app.overlays.settings_mut().value_of("colorBlindMode"),
                Some(json!(false))
            );
            // Cycle the bool row with → (same as Enter in the overlay).
            assert!(app.overlays.settings_mut().select("colorBlindMode"));
            assert!(press(&mut app, &port, key(CtKeyCode::Right)));
            // The overlay shows the new value and stays open.
            assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
            assert_eq!(
                app.overlays.settings_mut().value_of("colorBlindMode"),
                Some(json!(true))
            );
            // The write landed in ~/.omp/agent/config.yml…
            let config = home.join(".omp/agent/config.yml");
            let text = std::fs::read_to_string(&config).unwrap();
            assert!(text.contains("colorBlindMode: true"), "{text}");
            // …and a fresh layered read sees it (port round trip).
            assert_eq!(
                port.settings_get(&["colorBlindMode".to_string()])[0].1,
                json!(true)
            );
            let logged = log(&port);
            assert!(
                logged.iter().any(|entry| entry.starts_with("settings_set(colorBlindMode")),
                "{logged:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old_home);
    }

    #[test]
    fn settings_theme_change_applies_live() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let home = with_temp_home("theme");
        {
            let mut app = app();
            let port = port();
            app.set_appearance(crate::theme::Appearance::Dark);
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            assert!(app.overlays.settings_mut().select("theme.dark"));
            // Text row: Enter opens the inline editor pre-filled with the
            // current value ("titanium"); clear it, then type + save.
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            for _ in 0.."titanium".len() {
                assert!(press(&mut app, &port, key(CtKeyCode::Backspace)));
            }
            type_text(&mut app, &port, "light");
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            // Saved…
            assert_eq!(
                app.overlays.settings_mut().value_of("theme.dark"),
                Some(json!("light"))
            );
            let config = home.join(".omp/agent/config.yml");
            assert!(std::fs::read_to_string(&config).unwrap().contains("light"));
            // …and applied: the app theme is now the light theme.
            let expected = crate::theme::load("light").unwrap();
            assert_eq!(app.theme.color("text"), expected.color("text"));
            assert_eq!(app.theme.color("accent"), expected.color("accent"));
            // A notice confirms the live apply.
            let s = screen(&mut app, 150, 30);
            assert!(s.contains("theme.dark applied"), "{s}");
        }
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old_home);
    }

    fn with_catalog(app: &mut App) {
        app.apply_event(&AppEvent::AvailableCommands(json!({
            "type": "available_commands_update",
            "commands": [
                {"name": "help", "description": "Show help", "source": "builtin"},
                {"name": "model", "description": "Pick model", "source": "builtin"},
                {"name": "compact", "description": "Compact context", "source": "builtin"}
            ]
        })));
    }

    fn screen(app: &mut App, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| app.render(f)).unwrap();
        let buf = t.backend().buffer().clone();
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn typing_slash_shows_command_popup_above_editor() {
        // Regression: `/` showed nothing — the popup was never wired.
        let mut app = app();
        let port = port();
        with_catalog(&mut app);
        type_text(&mut app, &port, "/");
        let s = screen(&mut app, 60, 14);
        assert!(s.contains("help") && s.contains("model") && s.contains("compact"), "{s}");
        type_text(&mut app, &port, "mo");
        let s = screen(&mut app, 60, 14);
        // Best match first and selected (lower-ranked fuzzy hits may follow).
        assert!(s.contains("❯ model"), "filtered: {s}");
        assert!(!s.contains("help"), "non-matching rows dropped: {s}");
    }

    #[test]
    fn enter_in_slash_popup_completes_and_runs_command() {
        let mut app = app();
        let port = port();
        with_catalog(&mut app);
        type_text(&mut app, &port, "/he");
        assert!(app.autocomplete.open);
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Help));
        assert!(app.editor.is_empty());
    }

    #[test]
    fn esc_closes_popup_and_keeps_draft() {
        let mut app = app();
        let port = port();
        with_catalog(&mut app);
        type_text(&mut app, &port, "/co");
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert!(!app.autocomplete.open);
        assert_eq!(app.editor.text(), "/co");
    }

    #[test]
    fn catalog_command_dispatches_to_runtime() {
        // Regression: every command outside the 11 local builtins was
        // rejected as "unknown command" although the popup offered it.
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::AvailableCommands(json!({
            "commands": [{"name": "session", "aliases": ["info"], "source": "builtin"}]
        })));
        type_text(&mut app, &port, "/info");
        app.autocomplete.close();
        assert!(enter(&mut app, &port));
        assert!(log(&port).iter().any(|l| l.contains("info")), "{:?}", log(&port));
    }

    #[test]
    fn command_output_and_prompt_errors_reach_the_transcript() {
        let mut app = app();
        app.apply_event(&parse_frame(
            &json!({"type": "command_output", "text": "Session: abc\nTitle: t\nCWD: /tmp"}),
        ));
        app.apply_event(&parse_frame(
            &json!({"type": "prompt_result", "success": false, "error": "Unknown command: /zz"}),
        ));
        let s = screen(&mut app, 60, 14);
        assert!(s.contains("Session: abc") && !s.contains("abcTitle"), "{s}");
        assert!(s.lines().any(|l| l.trim_start().starts_with("CWD: /tmp")), "line breaks kept: {s}");
        assert!(s.contains("Unknown command: /zz"), "{s}");
    }

    #[test]
    fn question_mark_is_text_inside_a_draft() {
        // Regression: `?` toggled help even mid-sentence.
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "why?");
        assert_eq!(app.editor.text(), "why?");
        assert_eq!(app.overlays.open(), None);
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
        // Rich rendering emits several rows per entry; the watermark must
        // advance over every finished entry.
        assert!(!lines.is_empty());
        assert!(app.messages.committed_len() > committed);
        // The live tail must exclude everything committed so far.
        assert_eq!(
            app.messages.live_tail().len() + app.messages.committed_len(),
            app.messages.entries().len()
        );
        // Rebuild returns the full transcript and re-bases the watermark.
        let full = app.take_scrollback_lines(80, true);
        assert!(!full.is_empty());
        assert_eq!(app.messages.committed_len(), app.messages.finished_len());
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

    // ---- /resume, /tree, /theme overlay wiring ----------------------------

    #[test]
    fn slash_resume_opens_embedded_session_picker() {
        let mut app = app();
        let port = port();
        let dir = std::env::temp_dir().join(format!("omp-resume-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A minimal journal: title line + session header.
        let file = dir.join("sess.jsonl");
        std::fs::write(
            &file,
            concat!(
                "{\"type\":\"title\",\"title\":\"demo session\"}\n",
                "{\"type\":\"session\",\"id\":\"s1\",\"cwd\":\"/repo\"}\n",
            ),
        )
        .unwrap();
        app.configure(Vec::new(), dir.clone());
        type_text(&mut app, &port, "/resume");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Sessions));
        // Enter resumes the highlighted session through the port.
        assert!(enter(&mut app, &port));
        let calls = log(&port);
        assert!(
            calls
                .iter()
                .any(|c| c.starts_with("resume_session(") && c.contains("sess.jsonl")),
            "calls: {calls:?}"
        );
        assert_eq!(app.overlays.open(), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn slash_tree_opens_tree_overlay_from_journal() {
        let mut app = app();
        let port = port();
        push_user_turn(&mut app, "u1", None, "first question");
        push_user_turn(&mut app, "u2", Some("u1"), "follow up");
        type_text(&mut app, &port, "/tree");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Tree));
        assert!(app.has_fullscreen_overlay());
    }

    #[test]
    fn slash_theme_lists_and_hot_swaps() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/theme");
        assert!(enter(&mut app, &port));
        assert!(app
            .messages
            .entries()
            .iter()
            .any(|e| e.text.contains("available themes:")));
        type_text(&mut app, &port, "/theme light");
        assert!(enter(&mut app, &port));
        // The transcript renderer follows the app theme (set_theme is the
        // single sync point, not a per-frame copy).
        assert_eq!(
            app.theme.color("accent"),
            app.messages.theme().color("accent")
        );
        assert_ne!(
            app.theme.color("accent"),
            Theme::default().color("accent")
        );
    }

    #[test]
    fn theme_setter_syncs_transcript_renderer_once() {
        let mut app = app();
        let before = app.messages.theme().color("accent");
        let t = crate::theme::load("dark").expect("builtin dark theme");
        app.set_theme(t.clone());
        assert_eq!(app.messages.theme().color("accent"), t.color("accent"));
        assert_ne!(before, t.color("accent"));
    }

    // ---- double-Esc rewind --------------------------------------------------

    #[test]
    fn double_esc_opens_rewind_tree_and_confirms() {
        let mut app = app();
        let port = port();
        push_user_turn(&mut app, "u1", None, "first question");
        push_user_turn(&mut app, "u2", Some("u1"), "follow up");
        let t0 = Instant::now();
        let esc = || key(CtKeyCode::Esc);
        assert!(app.handle_key(esc(), &port, t0).is_some());
        assert_eq!(app.overlays.open(), None); // first Esc is timing state
        assert!(app
            .handle_key(esc(), &port, t0 + Duration::from_millis(100))
            .is_some());
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Tree));
        // Enter confirms the rewind target → port.rewind_entry.
        assert!(enter(&mut app, &port));
        let calls = log(&port);
        assert!(
            calls.iter().any(|c| c.starts_with("rewind_entry(")),
            "calls: {calls:?}"
        );
        assert_eq!(app.overlays.open(), None);
    }

    #[test]
    fn double_esc_without_history_shows_notice() {
        let mut app = app();
        let port = port();
        let t0 = Instant::now();
        assert!(app.handle_key(key(CtKeyCode::Esc), &port, t0).is_some());
        assert!(app
            .handle_key(key(CtKeyCode::Esc), &port, t0 + Duration::from_millis(50))
            .is_some());
        assert_eq!(app.overlays.open(), None);
        assert!(app
            .messages
            .entries()
            .iter()
            .any(|e| e.text.contains("nothing to rewind")));
    }

    // ---- terminal title -----------------------------------------------------

    #[test]
    fn title_follows_agent_start_and_end() {
        let mut app = app();
        let title = app.title().current_title().unwrap_or_default();
        assert!(title.contains("repo"), "initial title: {title:?}");
        app.apply_event(&AppEvent::AgentStart);
        let working = app.title().current_title().unwrap_or_default();
        assert!(working.starts_with('π'), "working title: {working:?}");
        assert!(working.contains("repo"), "working title: {working:?}");
        app.apply_event(&AppEvent::AgentEnd(json!({})));
        let idle = app.title().current_title().unwrap_or_default();
        assert!(idle.contains('>'), "idle title: {idle:?}");
    }

    // ---- notifications ------------------------------------------------------

    #[test]
    fn session_settled_fires_turn_notification() {
        let mut app = app();
        app.apply_event(&AppEvent::SessionSettled);
        assert_eq!(
            app.notify_sink().record(),
            vec!["omp: Turn complete".to_string()]
        );
    }

    #[test]
    fn failed_prompt_result_fires_error_notification() {
        let mut app = app();
        app.apply_event(&AppEvent::PromptResult {
            id: None,
            success: false,
            frame: json!({"type":"prompt_result","success":false,"error":"kaboom"}),
        });
        assert_eq!(
            app.notify_sink().record(),
            vec!["omp: Turn failed: kaboom".to_string()]
        );
    }

    #[test]
    fn notify_hook_intercepts_delivery() {
        let mut app = app();
        let got = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink_got = got.clone();
        app.set_notify_hook(std::sync::Arc::new(move |n| {
            sink_got.lock().push(n.body.clone());
        }));
        app.apply_event(&AppEvent::SessionSettled);
        assert_eq!(got.lock().clone(), vec!["Turn complete".to_string()]);
    }

    // ---- keymap engine on the input path ---------------------------------

    #[test]
    fn input_path_uses_keymap_overrides() {
        let mut app = app();
        let port = port();
        // Rebind app.exit to ctrl+q; ctrl+d must stop quitting.
        let mut user = crate::keybindings::UserBindings::new();
        user.insert("app.exit".into(), vec!["ctrl+q".into()]);
        let mut km = crate::keybindings::KeybindingsManager::new(user);
        std::mem::swap(&mut app.keys, &mut km);
        assert!(app.handle_key(ctrl('d'), &port, Instant::now()).is_some());
        assert_eq!(app.handle_key(ctrl('q'), &port, Instant::now()), None);
    }

    #[test]
    fn input_path_rebound_dequeue_chord() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::QueueUpdate {
            steering: vec!["one".into()],
            follow_up: Vec::new(),
        });
        // Rebind dequeue to ctrl+n.
        let mut user = crate::keybindings::UserBindings::new();
        user.insert("app.message.dequeue".into(), vec!["ctrl+n".into()]);
        let mut km = crate::keybindings::KeybindingsManager::new(user);
        std::mem::swap(&mut app.keys, &mut km);
        assert!(press(&mut app, &port, ctrl('n')));
        assert_eq!(app.editor.text(), "one");
        assert_eq!(
            log(&port),
            vec!["remove_queued(\"one\", Steering)".to_string()]
        );
    }

    #[test]
    fn band_hint_reflects_dequeue_binding() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::QueueUpdate {
            steering: vec!["x".into()],
            follow_up: Vec::new(),
        });
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let band: Vec<String> = (15..22)
            .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect())
            .collect();
        let text = band.join("\n");
        assert!(text.contains("alt+up to edit"), "{text}");
    }

    /// Push one user turn through `message_end` so the rewind journal has a
    /// realistic entry (`{id, parentId?, role, content}` inside `message`).
    fn push_user_turn(app: &mut App, id: &str, parent: Option<&str>, text: &str) {
        let mut message = serde_json::json!({
            "id": id,
            "role": "user",
            "content": [{"type": "text", "text": text}],
        });
        if let Some(p) = parent {
            message["parentId"] = Value::from(p);
        }
        app.apply_event(&AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": message,
        })));
    }
}
