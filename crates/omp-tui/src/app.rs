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
use crate::chrome::welcome;
use crate::component::Component;
use crate::editor::{Editor, EditorOutcome};
use crate::events::{parse_frame, parse_frame_owned, AppEvent};
use crate::extension_ui::ExtensionUi;
use crate::keybindings::KeybindingsManager;
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::overlays::models::{parse_model_string, ModelEntry};
use crate::overlays::sessions::SessionSource;
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
/// Loader spinner glyph step (TS `SPINNER_ADVANCE_MS`).
const SPINNER_ADVANCE_MS: u64 = 80;
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
    ("resume", "Resume a different session"),
    ("tree", "Navigate session tree (switch branches)"),
    ("theme", "Switch the color theme"),
    ("clear", "Start a new session (clears the conversation)"),
    ("help", "Show keybindings and commands"),
    ("exit", "Exit the application"),
];

/// Startup welcome banner (`WelcomeComponent`): the first transcript block.
#[derive(Debug, Clone)]
struct Welcome {
    version: String,
    /// Tip latched once per session.
    tip: String,
    /// Rows already committed to scrollback.
    emitted: bool,
    /// First paint of the welcome: the clean welcome shimmers for
    /// [`welcome::SHIMMER_MS`] from here, held in the live region (`None`
    /// until drawn, so a slow startup does not eat the animation).
    started: Option<Instant>,
}

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

    /// Owned-frame variant of [`Self::push_frame`]; identical queueing, but
    /// `session_loaded` moves its (large) message array into the event.
    pub fn push_frame_owned(&mut self, frame: Value, now: Instant) {
        if frame.get("type").and_then(Value::as_str) == Some("session_loaded") {
            if let Some((at, update)) = self.pending_update.take() {
                self.queue.push_back((at, AppEvent::MessageUpdate(update)));
            }
            self.queue.push_back((now, parse_frame_owned(frame)));
        } else {
            self.push_frame(&frame, now);
        }
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
    /// Startup welcome banner state (first transcript block); `None` once
    /// disabled. Rows are committed to scrollback on the first drain.
    welcome: Option<Welcome>,
    messages: Messages,
    editor: Editor,
    /// Slash / `@file` / path completion popup above the editor.
    autocomplete: Autocomplete,
    status: StatusLine,
    /// Efforts the active model supports (without `off`); empty = the model
    /// cannot think.
    thinking_levels: Vec<String>,
    overlays: Overlays,
    extension_ui: ExtensionUi,
    theme: Theme,
    /// Backlog scroll offset (PgUp/PgDn), reused once the shell renderer
    /// owns scrolling; kept for the full-screen fallback renderer.
    scroll: u16,
    /// The serialized, coalescing event pump (frames enter via
    /// [`Self::push_frame`], are applied by [`Self::pump_events`]).
    pump: EventPump,
    /// Spinner clock origin (TS derives the glyph from wall time).
    spinner_start: Instant,
    /// Live per-turn token tally for the loader row (reset each turn).
    turn_usage: TurnUsage,
    /// Active model's `$/M` pricing `(input, output)` for the cost suffix.
    model_pricing: Option<(f64, f64)>,
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
    /// Mirror of `tui.mouse` ("Mouse Click-to-Focus"): inline SGR mouse
    /// capture on the normal buffer while no overlay is open (TS
    /// `InteractiveMode#mouseCapture` → `setInlineMouseTrackingProvider`).
    mouse_capture: bool,
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
    /// Session name once known (TS `sessionManager.getSessionName()`): the notification title.
    session_name: Option<String>,
    /// `stopReason` of the turn's last assistant message (TS reads `agent_end.messages`).
    last_stop_reason: Option<String>,
    /// Raw journal entries (`message_end` frames normalized to
    /// `{type:"message",id,parentId,message}`) feeding the rewind column /
    /// tree overlay; reset when a session is resumed.
    journal: Vec<Value>,
    /// Set by `session_reset`: the controller erases the terminal scrollback
    /// and re-commits the (now empty) transcript before the next paint.
    clear_scrollback: bool,
    /// Session stores for the `/resume` embedded picker (native first).
    session_sources: Vec<SessionSource>,
    /// Terminal appearance from the startup OSC 11 / COLORFGBG probe:
    /// decides whether a `theme.dark`/`theme.light` edit hot-swaps now.
    appearance: crate::theme::Appearance,
    /// Working directory the session runs in (settings re-reads on edits).
    cwd: PathBuf,
    /// Presentation prefs layered onto every theme set after
    /// [`App::load_settings_theme`] (`symbolPreset`, `tui.style`); `None`
    /// keeps each theme's own (tests, `/theme` before startup).
    presentation: Option<(String, crate::theme::UiStyle)>,
    /// In-flight clipboard read (paste chord → worker thread → drained in
    /// [`App::poll_deadlines`]); the read shells out and must not block the
    /// render loop (TS spawns off the event loop, clipboard.ts `spawnCapture`).
    clipboard_rx: Option<std::sync::mpsc::Receiver<crate::clipboard::ClipboardOutcome>>,
    /// Paste staged behind the open large-paste menu (`paste.largeMenuThreshold`).
    pending_large_paste: Option<(String, usize)>,
    /// Word-completion engine host, spawned once by
    /// [`App::start_word_completion`] and reused across setting flips.
    word_host: Option<std::sync::Arc<crate::word_completion::host::WordHost>>,
    /// Set by [`App::start_word_completion`]; tests and embedders that never
    /// call it never spawn an engine (or touch the agent dir).
    word_started: bool,
    /// Magic keywords disabled by `magicKeywords.*` (read at startup and on
    /// settings edits, never per keystroke); layered onto every theme.
    magic_off: u32,
    /// Pinned global settings file (see [`App::set_settings_global`]); `None`
    /// means the user's own config resolved through `HOME`.
    settings_global: Option<Option<PathBuf>>,
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
            welcome: None,
            messages: Messages::new(),
            editor: Editor::new(),
            autocomplete: Autocomplete::new(cwd.to_path_buf()),
            status: StatusLine::new(model, cwd.display().to_string()),
            thinking_levels: Vec::new(),
            overlays: Overlays::default(),
            extension_ui: ExtensionUi::new(),
            theme: Theme::default(),
            scroll: 0,
            pump: EventPump::new(),
            spinner_start: Instant::now(),
            turn_usage: TurnUsage::default(),
            model_pricing: None,
            run_epoch: 0,
            steering: Vec::new(),
            follow_up: Vec::new(),
            maintenance: None,
            pending_submission: false,
            armed_bare_command: None,
            last_esc: None,
            last_ctrl_c: None,
            quit_requested: false,
            mouse_capture: false,
            keys: KeybindingsManager::defaults(),
            title,
            title_label: dir,
            notify: NotifySink::default(),
            session_name: None,
            last_stop_reason: None,
            journal: Vec::new(),
            clear_scrollback: false,
            session_sources: Vec::new(),
            appearance: crate::theme::Appearance::Dark,
            cwd: cwd.to_path_buf(),
            settings_global: None,
            presentation: None,
            clipboard_rx: None,
            pending_large_paste: None,
            word_host: None,
            word_started: false,
            magic_off: 0,
        }
    }

    /// Pin the global settings file this app reads (`None` = built-in
    /// defaults only), instead of resolving the user's config through the
    /// process-global `HOME`. Lets embedders and tests stay independent of
    /// ambient environment; project settings under `<cwd>/.omp` still layer on.
    pub fn set_settings_global(&mut self, global: Option<PathBuf>) {
        self.settings_global = Some(global);
    }

    /// Load settings from the pinned source, else the user's config.
    fn load_settings(&self) -> omp_settings::Settings {
        match &self.settings_global {
            None => omp_settings::Settings::load(&self.cwd),
            Some(global) => {
                omp_settings::Settings::load_from(global.as_deref(), &self.cwd.join(".omp"))
            }
        }
    }

    /// Turn on ghost-text word completion per `spelling.autocomplete`: the
    /// engine loads on a background thread, so the first frame never waits on it.
    pub fn start_word_completion(&mut self) {
        self.word_started = true;
        let settings = self.load_settings();
        self.apply_word_completion(&settings);
    }

    /// Flush the word-completion engine's learned state (called on exit).
    pub fn persist_word_completion(&self) {
        if let Some(host) = &self.word_host {
            host.persist();
        }
    }

    fn apply_word_completion(&mut self, settings: &omp_settings::Settings) {
        if !self.word_started {
            return;
        }
        let value = settings
            .get_string("spelling.autocomplete")
            .unwrap_or_else(|| "auto".to_owned());
        let Some(method) = crate::word_completion::host::method_for_setting(&value) else {
            self.editor.set_word_completion(None);
            return;
        };
        if self.word_host.is_none() {
            let Some(dir) = crate::word_completion::host::agent_dir() else {
                return;
            };
            self.word_host = Some(crate::word_completion::host::WordHost::spawn(&dir, method));
        }
        if !self.editor.word_completion_enabled() {
            self.editor.set_word_completion(self.word_host.clone());
        }
    }

    /// Enable the startup welcome banner (logo + version + a rotating tip)
    /// as the first transcript block.
    pub fn set_welcome(&mut self, version: &str) {
        self.welcome = Some(Welcome {
            version: version.to_owned(),
            tip: welcome::random_tip(),
            emitted: false,
            started: None,
        });
    }

    /// Set the model display name shown by the status line (models.yml `name`).
    pub fn set_model_name(&mut self, name: Option<String>) {
        self.status.model_name = name;
    }

    /// Mirror the runtime's thinking state (`thinkingLevel`, supported
    /// `thinkingLevels`, `autoThinking`) into the status line; a model
    /// without thinking shows no level (TS `thinkingLevelWord`).
    fn apply_thinking_state(&mut self, frame: &Value) {
        if let Some(levels) = frame.get("thinkingLevels").and_then(Value::as_array) {
            self.thinking_levels = levels
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
        }
        if let Some(auto) = frame.get("autoThinking").and_then(Value::as_bool) {
            self.status.auto_thinking = auto;
        }
        if let Some(level) = frame.get("thinkingLevel").and_then(Value::as_str) {
            self.set_thinking_display(level);
        }
        self.apply_ultra(frame);
        if self.thinking_levels.is_empty() {
            self.status.thinking = None;
            self.status.auto_thinking = false;
        }
    }

    /// Mirror the runtime's ultra mode (`"ultra": "<name>" | null`, wire
    /// contract 1); frames without the key leave it unchanged.
    fn apply_ultra(&mut self, frame: &Value) {
        if let Some(ultra) = frame.get("ultra") {
            self.status.ultra = ultra.as_str().and_then(crate::magic::Ultra::parse);
        }
    }

    fn set_thinking_display(&mut self, level: &str) {
        self.status.thinking = (!self.thinking_levels.is_empty()).then(|| level.to_owned());
    }

    /// Selector values for the active model in cycle order
    /// (`getAvailableEffortSelectors`): `off`, `auto`, then the efforts, then
    /// the omp-rs `ultracode` mode.
    fn effort_selectors(&self) -> Vec<String> {
        if self.thinking_levels.is_empty() {
            return Vec::new();
        }
        ["off", "auto"]
            .iter()
            .map(|s| (*s).to_owned())
            .chain(self.thinking_levels.iter().cloned())
            .chain(std::iter::once(crate::magic::Ultra::Code.name().to_owned()))
            .collect()
    }

    /// The configured selector (`configuredThinkingLevel`): the ultra mode
    /// while one is active, `auto` while auto thinking is on, else the
    /// effective level. Any ultra mode (incl. the hybrids, which are not in
    /// the cycle) is followed by `off`.
    fn current_effort_selector(&self) -> String {
        if let Some(ultra) = self.status.ultra {
            ultra.name().to_owned()
        } else if self.status.auto_thinking {
            "auto".to_owned()
        } else {
            self.status
                .thinking
                .clone()
                .unwrap_or_else(|| "off".to_owned())
        }
    }

    /// Shift+Tab (`app.thinking.cycle`): off → auto → minimal..max →
    /// ultracode → off (TS `cycleThinkingLevel` + omp-rs ultra).
    fn cycle_thinking_level(&mut self, port: &impl RuntimePort) {
        let levels = self.effort_selectors();
        if levels.is_empty() {
            self.push_system("Current model does not support thinking".into());
            return;
        }
        let current = self.current_effort_selector();
        let next = levels
            .iter()
            .position(|l| *l == current)
            .map_or(0, |idx| (idx + 1) % levels.len());
        self.dispatch(Action::SetThinkingLevel(levels[next].clone()), port);
    }

    /// Bare `/effort`: open the thinking-level selector (TS
    /// `showThinkingSelector`).
    fn open_thinking_selector(&mut self) {
        let mut levels = self.effort_selectors();
        if levels.is_empty() {
            self.push_system("Current model does not support thinking".into());
            return;
        }
        // TS lists `off, auto, …`; the selector shows the same order.
        let current = self.current_effort_selector();
        self.overlays
            .open_thinking(std::mem::take(&mut levels), &current);
    }

    /// Apply a `context_usage` frame: model identity plus the context gauge
    /// inputs (percent, window, compaction boundaries).
    fn apply_context_usage(&mut self, frame: &Value) {
        if let Some(m) = frame.get("model").and_then(Value::as_str) {
            if self.status.model != m {
                self.status.model = m.to_owned();
                self.overlays.models_mut().current = Some(m.to_owned());
            }
        }
        if let Some(name) = frame.get("modelName") {
            self.status.model_name = name.as_str().filter(|n| !n.is_empty()).map(str::to_owned);
        }
        self.apply_thinking_state(frame);
        self.status.context_percent = frame.get("percent").and_then(Value::as_f64);
        self.status.context_tokens = frame.get("tokens").and_then(Value::as_u64);
        self.status.context_window = frame
            .get("contextWindow")
            .and_then(Value::as_u64)
            .filter(|w| *w > 0);
        self.status.compaction_pct = frame.get("thresholdPercent").and_then(Value::as_f64);
        self.status.speculation_pct = frame.get("speculationPercent").and_then(Value::as_f64);
        self.status.boundaries_known = self.status.compaction_pct.is_some();
    }

    /// The right-aligned composer hint (`resolveComposerHint`): the
    /// thinking-effort gesture until the conversation starts.
    fn composer_hint(&self) -> Option<(String, String)> {
        if !self.messages.entries().is_empty() {
            return None;
        }
        let key = self.keys.first_key("app.thinking.cycle")?;
        Some((
            welcome::format_key_hint(&key),
            "to change thinking effort".to_owned(),
        ))
    }

    /// Record the terminal appearance resolved at startup (OSC 11 /
    /// COLORFGBG); `/settings` theme edits hot-swap the matching variant.
    pub fn set_appearance(&mut self, appearance: crate::theme::Appearance) {
        self.appearance = appearance;
    }

    /// Replace the app-wide theme (ThemeLoader startup + hot-swap). The
    /// transcript renderer is synced here — not per frame — so a hot-swap
    /// lands everywhere on the next paint (tui-ChatRenderWire caveat).
    pub fn set_theme(&mut self, mut theme: Theme) {
        if let Some((glyphs, ui)) = &self.presentation {
            theme.glyph_set.clone_from(glyphs);
            theme.ui = *ui;
        }
        theme.magic_off = self.magic_off;
        self.title.set_brand(if theme.clean() {
            crate::chrome::title::CLEAN_TERMINAL_TITLE
        } else {
            crate::chrome::title::DEFAULT_TERMINAL_TITLE
        });
        self.messages.set_theme(theme.clone());
        self.theme = theme;
    }

    /// Load the configured theme for the current appearance (`theme.dark` /
    /// `theme.light`) with the `symbolPreset` and `tui.style` settings
    /// applied (TS `symbolPresetOverride`: the setting beats the theme
    /// file's `symbols.preset`).
    ///
    /// # Errors
    ///
    /// The configured theme failed to load; the current theme is kept.
    pub fn load_settings_theme(&mut self) -> Result<(), crate::theme::ThemeError> {
        let settings = self.load_settings();
        let glyphs = settings
            .get_string("symbolPreset")
            .filter(|p| crate::theme::loader::is_valid_symbol_preset(p))
            .unwrap_or_else(|| "unicode".to_owned());
        let ui = crate::theme::UiStyle::resolve(settings.get_string("tui.style").as_deref());
        self.presentation = Some((glyphs, ui));
        self.magic_off = crate::magic::disabled_mask(&settings);
        let theme = crate::theme::load_configured(
            settings.get_string("theme.dark").as_deref(),
            settings.get_string("theme.light").as_deref(),
            self.appearance,
        );
        match theme {
            Ok(theme) => {
                self.set_theme(theme);
                Ok(())
            }
            Err(e) => {
                // Keep the current palette but still apply the presentation.
                let current = self.theme.clone();
                self.set_theme(current);
                Err(e)
            }
        }
    }

    /// Seed the `/model` picker catalogue + the embedded `/resume` session
    /// stores from the run configuration (called once by the controller).
    pub fn configure(&mut self, models: Vec<ModelEntry>, session_sources: Vec<SessionSource>) {
        let current = Some(self.status.model.clone());
        let picker = self.overlays.models_mut();
        picker.models = models;
        picker.current = current;
        self.session_sources = session_sources;
        self.refresh_model_pricing();
        self.reload_display_prefs();
    }

    /// Re-read the transcript display settings the chat model honours
    /// (`display.showTokenUsage`, `display.showTurnTime`,
    /// `read.toolResultPreview`; TS reads them per event from settings).
    fn reload_display_prefs(&mut self) {
        let settings = self.load_settings();
        let flag = |id: &str| settings.get_bool(id).unwrap_or(false);
        self.messages.set_display_prefs(crate::chat::DisplayPrefs {
            show_token_usage: flag("display.showTokenUsage"),
            show_turn_time: flag("display.showTurnTime"),
            read_preview: flag("read.toolResultPreview"),
        });
        self.mouse_capture = flag("tui.mouse");
        self.reload_composer(&settings);
    }

    /// Apply the composer shape and `statusLine.*` settings (TS
    /// `syncComposerShape` + the status line's live settings read).
    fn reload_composer(&mut self, settings: &omp_settings::Settings) {
        let shape = settings
            .get_string("composer.shape")
            .map_or(crate::composer::ComposerShape::default(), |name| {
                crate::composer::ComposerShape::from_name(&name)
            });
        self.editor.set_shape(shape);
        self.status
            .apply_config(crate::status_line::StatusLineConfig::from_settings(
                settings,
            ));
        // Input settings with a live consumer here (TS `#applyUiSettingChanges`).
        self.editor
            .set_vim_mode(settings.get_bool("tui.vimMode").unwrap_or(false));
        self.status.vim_text_display = settings
            .get_string("tui.vimModeDisplay")
            .is_none_or(|mode| mode == "text");
        self.apply_word_completion(settings);
        if let Some(max) = settings.get("autocompleteMaxVisible").as_u64() {
            self.autocomplete.max_visible = usize::try_from(max).unwrap_or(8).clamp(3, 20);
        }
    }

    /// Mirror the editor's Vim state into the status line's `vim` segment.
    fn sync_vim_status(&mut self) {
        use crate::editor::vim::VimMode as Editor;
        use crate::status_line::VimMode as Status;
        self.status.vim = self.editor.vim_mode().map(|mode| match mode {
            Editor::Normal => Status::Normal,
            Editor::Insert => Status::Insert,
            Editor::Visual => Status::Visual,
            Editor::VisualLine => Status::VisualLine,
        });
        self.status.vim_pending = self.editor.vim_pending();
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

    /// Owned-frame variant of [`Self::push_frame`]: a `session_loaded`
    /// history is moved into its event instead of cloned.
    pub fn push_frame_owned(&mut self, frame: Value, now: Instant) {
        self.pump.push_frame_owned(frame, now);
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
        self.poll_paste_results();
    }

    /// Apply finished paste results: a clipboard read (paste chord) and a
    /// resolved large-paste menu pick. The controller wakes on
    /// [`App::has_pending_paste_results`] so this runs — and the following
    /// paint lands — even when the loop would otherwise block with no
    /// terminal events.
    pub fn poll_paste_results(&mut self) {
        if let Some(rx) = &self.clipboard_rx {
            if let Ok(outcome) = rx.try_recv() {
                self.clipboard_rx = None;
                self.apply_clipboard_outcome(outcome);
            }
        }
        if let Some(pick) = self.overlays.large_paste_mut().take_pick() {
            self.apply_large_paste_pick(pick);
        }
    }

    /// Whether paste results are pending: a clipboard read in flight or an
    /// open large-paste menu whose pick still needs draining.
    #[must_use]
    pub fn has_pending_paste_results(&self) -> bool {
        self.clipboard_rx.is_some() || self.overlays.stack().contains(&OpenOverlay::LargePaste)
    }

    /// Whether a `message_update` is still inside its coalesce window; the
    /// controller uses this to schedule the next pump tick.
    #[must_use]
    pub fn has_pending_update(&self) -> bool {
        self.pump.has_pending_update()
    }

    /// The frame-timer period the controller loop must arm right now, or
    /// `None` when nothing animates (idle loops must not wake): 30 fps
    /// while the streaming reveal has a backlog (TS
    /// `STREAMING_REVEAL_FRAME_MS`), the 80 ms spinner step while the
    /// loader row is visible (TS `SPINNER_ADVANCE_MS`).
    #[must_use]
    pub fn animation_interval(&self) -> Option<Duration> {
        if self.messages.reveal_animating() || self.welcome_shimmer(Instant::now()).is_some() {
            Some(Duration::from_millis(crate::chat::reveal::FRAME_MS))
        } else if self.editor_shimmers() {
            Some(Duration::from_millis(crate::magic_paint::SHIMMER_FRAME_MS))
        } else if self.show_loader() {
            Some(Duration::from_millis(SPINNER_ADVANCE_MS))
        } else {
            None
        }
    }

    /// Startup shimmer phase of the clean welcome while it is still held in
    /// the live region (not yet committed), `None` once it froze.
    fn welcome_shimmer(&self, now: Instant) -> Option<f64> {
        let w = self
            .welcome
            .as_ref()
            .filter(|w| !w.emitted && self.theme.clean())?;
        welcome::shimmer_phase(
            w.started
                .map_or(Duration::ZERO, |s| now.saturating_duration_since(s)),
        )
    }

    /// Welcome block rows at `width` (clean: OMT logo at `phase`).
    fn welcome_rows(&self, width: u16, phase: f64) -> Vec<Line<'static>> {
        let Some(w) = &self.welcome else {
            return Vec::new();
        };
        if self.theme.clean() {
            welcome::clean_welcome_lines(&w.version, &w.tip, usize::from(width), &self.theme, phase)
        } else {
            welcome::welcome_lines(&w.version, &w.tip, usize::from(width), &self.theme)
        }
    }

    /// Whether the editor's magic keywords shimmer (TS: focused editor with
    /// a keyword on screen); no keyword → no animation timer.
    fn editor_shimmers(&self) -> bool {
        self.overlays.open().is_none()
            && !self.extension_ui.is_active()
            && crate::magic_paint::any_keyword(self.editor.buffer.lines(), self.theme.magic_off)
    }

    /// The clean status line's τ state: blocked on the user (dialog / ask)
    /// beats running, which beats a pinned turn error, else idle.
    fn brand_state(&self) -> crate::status_line::Brand {
        use crate::status_line::Brand;
        let asking = self.messages.live_tail().iter().any(|e| {
            matches!(&e.kind, crate::chat::ChatKind::Tool(card) if card.name == "ask" && !card.status.settled())
        });
        if self.extension_ui.pending_len() > 0 || asking {
            Brand::Blocked
        } else if self.status.streaming || self.pending_submission {
            Brand::Running
        } else if self.messages.pinned_error().is_some() {
            Brand::Error
        } else {
            Brand::Idle
        }
    }

    /// Per-frame live state the status line and editor render from.
    fn sync_frame_state(&mut self) {
        self.status.tick(now_millis());
        self.status.brand = self.brand_state();
        let phase = if self.editor_shimmers() {
            crate::magic_paint::shimmer_phase(now_millis())
        } else {
            0.0
        };
        self.editor.set_magic_phase(phase);
    }

    /// Apply one classified runtime event directly (tests + compat path).
    pub fn apply_event(&mut self, event: &AppEvent) {
        match &event {
            AppEvent::AgentStart => {
                self.last_stop_reason = None;
                self.spinner_start = Instant::now();
                self.turn_usage.reset();
                self.status.streaming = true;
                self.pending_submission = false;
                self.title
                    .set_state(TitleState::Working, Some(&self.title_label));
            }
            AppEvent::TurnStart => {
                self.turn_usage.reset();
                self.status.streaming = true;
                self.pending_submission = false;
            }
            AppEvent::AgentEnd(_) | AppEvent::TurnEnd(_) => {
                self.status.streaming = false;
                self.title
                    .set_state(TitleState::Idle, Some(&self.title_label));
            }
            AppEvent::SessionSettled => {
                self.status.streaming = false;
                self.title
                    .set_state(TitleState::Idle, Some(&self.title_label));
                self.notify_turn_settled(false);
            }
            AppEvent::PromptResult { success, .. } => {
                if !success {
                    // The transcript row is drawn by `messages.apply` below;
                    // here only the title and the settle notification.
                    self.title
                        .set_state(TitleState::Idle, Some(&self.title_label));
                    self.notify_turn_settled(true);
                }
            }
            AppEvent::MessageEnd(frame) => {
                if let Some(message) = frame.get("message") {
                    self.turn_usage.observe_message(message);
                }
                let message = frame.get("message");
                if message.and_then(|m| m.get("role")).and_then(Value::as_str) == Some("assistant")
                {
                    self.last_stop_reason = message
                        .and_then(|m| m.get("stopReason"))
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                }
                if let Some(entry) = journal_entry(frame) {
                    self.journal.push(entry);
                }
            }
            AppEvent::SessionNameChanged(name) => {
                self.title_label = name.clone();
                self.session_name = Some(name.clone()).filter(|n| !n.is_empty());
                // Re-emit the current state with the new label.
                let state = if self.status.streaming {
                    TitleState::Working
                } else {
                    TitleState::Idle
                };
                self.title.set_state(state, Some(name));
            }
            AppEvent::MessageStart(_) => self.pending_submission = false,
            AppEvent::MessageUpdate(frame) => {
                if let Some(message) = frame.get("message") {
                    self.turn_usage.observe_message(message);
                }
                // `delta_updates` mode carries only `assistantMessageEvent.delta`
                // (no cumulative `message` snapshot); fold it here.
                self.turn_usage.observe_delta(frame);
            }
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
                self.refresh_model_pricing();
            }
            AppEvent::ContextUsage(frame) => {
                self.apply_context_usage(frame);
                self.refresh_model_pricing();
            }
            AppEvent::ModelsChanged { models } => {
                let picker = self.overlays.models_mut();
                picker.models = models.clone();
            }
            AppEvent::ThinkingLevelChanged(frame) => {
                let configured_auto =
                    frame.get("configured").and_then(Value::as_str) == Some("auto");
                self.status.auto_thinking = configured_auto;
                if let Some(level) = frame.get("thinkingLevel").and_then(Value::as_str) {
                    self.set_thinking_display(level);
                }
                self.apply_ultra(frame);
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
            AppEvent::ExtensionUiRequest(frame) => {
                // TS rpc `set_editor_text` (fire-and-forget): replace the prompt draft.
                if frame.get("method").and_then(Value::as_str) == Some("set_editor_text") {
                    let text = frame
                        .get("text")
                        .or_else(|| frame.pointer("/params/text"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    self.set_editor_text(&text);
                } else {
                    self.extension_ui.push(frame);
                }
            }
            AppEvent::ClipboardCopy(text) => {
                // TS `copyToClipboard`: OSC 52 + platform tools, best effort, off the
                // render thread.
                {
                    use std::io::{IsTerminal as _, Write as _};
                    let mut out = std::io::stdout();
                    if out.is_terminal() {
                        let _ = out.write_all(crate::clipboard::osc52(text).as_bytes());
                        let _ = out.flush();
                    }
                }
                let text = text.clone();
                std::thread::spawn(move || crate::clipboard::write_text(&text));
            }
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
            AppEvent::Notice { level, message } => {
                let glyph = match level.as_str() {
                    "warning" => self.theme.symbol("status.warning"),
                    "error" => self.theme.symbol("status.error"),
                    _ => self.theme.symbol("status.info"),
                };
                self.push_system(format!("{glyph} {message}"));
            }
            AppEvent::SessionLoaded { messages, name } => {
                // The resumed history replaces the transcript; rows already
                // in terminal scrollback stay, the history is appended below.
                self.journal.clear();
                self.status.streaming = false;
                self.messages.rebuild(messages);
                if let Some(name) = name {
                    self.title_label.clone_from(name);
                    self.session_name = Some(name.clone()).filter(|n| !n.is_empty());
                }
                self.title
                    .set_state(TitleState::Idle, Some(&self.title_label));
            }
            AppEvent::SessionReset => {
                self.journal.clear();
                self.status.streaming = false;
                self.messages.rebuild(&[]);
                self.clear_scrollback = true;
            }
            AppEvent::SessionInfo { title, lines } => {
                self.overlays
                    .open_session_info(title.as_deref(), lines.clone());
            }
            _ => {}
        }
        self.messages.apply(event);
    }

    /// Turn-settle notifications (TS `sendErrorNotification` / `sendCompletionNotification`).
    /// A failed turn (an errored assistant message or a failed prompt) rings `request` with
    /// "Stopped with error" when `error.notify` is on (default off); any other non-aborted
    /// settle rings `done` with "Complete" unless `completion.notify` is off. The title is the
    /// session name, else `omp`.
    fn notify_turn_settled(&mut self, failed: bool) {
        let settings = self.load_settings();
        let on = |id: &str| settings.get_string(id).as_deref() != Some("off");
        let Some((body, kind)) = turn_notification(
            failed,
            self.last_stop_reason.as_deref(),
            on("error.notify"),
            on("completion.notify"),
        ) else {
            return;
        };
        let mut n = TerminalNotification::body(body);
        n.title = Some(
            self.session_name
                .clone()
                .unwrap_or_else(|| "omp".to_owned()),
        );
        n.kind = kind;
        self.notify.fire(&n);
    }

    /// Handle one terminal event. Returns `false` when the app should quit.
    pub fn handle_terminal_event(&mut self, event: &Event, port: &impl RuntimePort) -> bool {
        if let Some(mouse) = crate::mouse::sgr_mouse_of(event) {
            self.handle_mouse(&mouse, port);
            return true;
        }
        let Event::Key(key) = event else {
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
            return true; // resize: redraw-only
        };
        self.handle_key(*key, port, Instant::now()).unwrap_or(false)
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
    fn dispatch(&mut self, action: Action, port: &impl RuntimePort) {
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
                        applied.lines,
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
        if !self.submit_pipeline(text, port) || self.quit_requested {
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
            .map_or(100, |v| if v <= 0.0 { 0 } else { v as usize })
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
    fn apply_clipboard_outcome(&mut self, outcome: crate::clipboard::ClipboardOutcome) {
        use crate::clipboard::ClipboardOutcome;
        match outcome {
            ClipboardOutcome::Image(png) => self.attach_clipboard_image(png),
            ClipboardOutcome::TextRaw(text) => {
                self.editor.insert_str_at_cursor(&text);
            }
            ClipboardOutcome::Text(text) if text.is_empty() => {
                self.push_system("Clipboard is empty".to_owned());
            }
            ClipboardOutcome::Text(text) => self.route_paste(&text, false),
            ClipboardOutcome::Empty => {
                self.push_system("Clipboard is empty".to_owned());
            }
            ClipboardOutcome::Failed => {
                self.push_system("Failed to read clipboard".to_owned());
            }
        }
    }

    /// Save a clipboard PNG to a temp file and queue it as an image
    /// attachment (TS `sink.attach({type:"image", …})`; omp-rs attachments
    /// reference a file path, so the payload is materialized first).
    fn attach_clipboard_image(&mut self, png: Vec<u8>) {
        let path = std::env::temp_dir().join(format!(
            "omp-clipboard-{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis())
        ));
        if std::fs::write(&path, &png).is_err() {
            self.push_system("Failed to save clipboard image".to_owned());
            return;
        }
        self.editor
            .attach_image(&path.display().to_string(), None, None);
    }

    /// Apply the large-paste menu choice (`presentLargePasteMenu` switch):
    /// wrapped `<attachment>` block, file reference, inline marker, or the
    /// Esc fallback (stage as a text-attachment chip). The content is never
    /// lost.
    fn apply_large_paste_pick(&mut self, pick: crate::overlays::large_paste::LargePastePick) {
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
                    self.push_system(format!(
                        "Saved {line_count} pasted lines to {}",
                        path.display()
                    ));
                } else {
                    self.editor.attach_text("paste.md", &text);
                    self.push_system(
                        "Failed to save paste to a file — attached as a text chip instead"
                            .to_owned(),
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
                self.push_system("nothing to rewind yet".into());
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
            // TS `/switch` opens the same model selector (session-only pick).
            "model" | "switch" => self.slash_model(name, args, port),
            "effort" if args.is_empty() => self.open_thinking_selector(),
            "resume" => self.slash_resume(),
            "tree" => {
                if self.journal.is_empty() {
                    self.push_system("no session tree yet".into());
                } else {
                    self.overlays
                        .open_tree(&self.journal.clone(), TreeMode::Tree);
                }
            }
            "theme" => self.slash_theme(args),
            // TS `/session` (bare or `info`) opens the session info panel;
            // other verbs (delete, pin, …) stay with the runtime.
            "session" if matches!(args.trim(), "" | "info") => port.request_session_info(),
            // TS `/usage` (bare or `show`) opens the usage panel; `reset`
            // stays with the runtime.
            "usage" if matches!(args.trim(), "" | "show") => port.request_usage(),
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
    fn slash_model(&mut self, name: &str, args: &str, port: &impl RuntimePort) {
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
                    name: name.to_owned(),
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
        overlay.set_available_themes(
            crate::theme::available_themes()
                .into_iter()
                .map(|t| t.name)
                .collect(),
        );
        overlay.set_thinking_levels(port.available_thinking_levels());
    }

    /// `/resume`: open the embedded session picker (overlays-b P0 #10); the
    /// pick resolves to `Action::ResumeSession` via the overlay drain.
    fn slash_resume(&mut self) {
        if self.session_sources.is_empty() {
            self.push_system("session storage is not available in this session".into());
            return;
        }
        self.overlays.open_sessions(&self.session_sources);
    }

    /// Live half of a persisted settings edit: apply what omp-rs can right
    /// now (TS applies everything through the cfg:// host; here only theme
    /// and thinking level have live consumers, everything else is saved for
    /// the next runtime read and flagged "no live effect" in the overlay).
    fn apply_setting_live(&mut self, id: &str, value: &Value, port: &impl RuntimePort) {
        match id {
            "theme.dark" | "theme.light" | "symbolPreset" | "tui.style" => {
                // Re-read the layered settings and hot-swap the theme for
                // the terminal's current appearance.
                match self.load_settings_theme() {
                    Ok(()) => self.push_system(format!("{id} applied")),
                    Err(e) => self.push_system(format!("{id} saved, but theme reload failed: {e}")),
                }
            }
            "defaultThinkingLevel" => {
                if let Some(level) = value.as_str().filter(|level| *level != "auto") {
                    port.set_thinking_level(level.to_string());
                }
            }
            "display.showTokenUsage"
            | "display.showTurnTime"
            | "read.toolResultPreview"
            | "tui.mouse" => {
                self.reload_display_prefs();
            }
            id if id.starts_with("magicKeywords.") => {
                self.magic_off = crate::magic::disabled_mask(&self.load_settings());
                let theme = self.theme.clone();
                self.set_theme(theme);
            }
            "composer.shape"
            | "statusLine.preset"
            | "statusLine.separator"
            | "statusLine.contextLine"
            | "statusLine.sessionAccent"
            | "statusLine.transparent"
            | "statusLine.compactThinkingLevel"
            | "statusLine.showHookStatus"
            | "statusLine.leftSegments"
            | "statusLine.rightSegments"
            | "tui.vimMode"
            | "tui.vimModeDisplay"
            | "spelling.autocomplete"
            | "autocompleteMaxVisible" => {
                let settings = self.load_settings();
                self.reload_composer(&settings);
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

    /// Minimum height of the live region below the committed chat: band +
    /// loader + status line (1) + one editor row (the band composer has no
    /// frame). `ShellRender` sizes the inline viewport with this.
    #[must_use]
    pub fn live_height(&self) -> u16 {
        let width = crossterm::terminal::size().map_or(80, |(w, _)| w);
        let shape = self.editor.shape();
        // One content row inside the shape's chrome, plus the standalone bar.
        let editor = 1 + u16::from(shape.has_top()) + u16::from(shape.has_bottom());
        self.band_height()
            + self.error_banner_height(width)
            + u16::from(self.show_loader())
            + editor
            + self.bottom_bar_rows(width)
    }

    /// Rows of the standalone status bar under the editor (gap + bar) for
    /// the active composer shape (`ComposerStyle.bottomBar`).
    fn bottom_bar_rows(&self, width: u16) -> u16 {
        let shape = self.editor.shape();
        if crate::composer::bottom_bar(shape, &self.status, &self.theme, usize::from(width))
            .is_none()
        {
            return 0;
        }
        1 + u16::from(shape.bottom_bar_gap())
    }

    /// Push this frame's status content into the editor's top chrome (TS
    /// `attachToEditor` provider) for a composer `width` columns wide.
    fn sync_composer_chrome(&mut self, width: u16) {
        let shape = self.editor.shape();
        let top = crate::composer::top_border(shape, &self.status, &self.theme, usize::from(width));
        self.editor.set_top_border(top);
    }

    /// Paint the standalone bar (and its spacer row) under the editor.
    fn render_bottom_bar(&self, frame: &mut Frame<'_>, area: Rect) {
        let shape = self.editor.shape();
        let Some(bar) =
            crate::composer::bottom_bar(shape, &self.status, &self.theme, usize::from(area.width))
        else {
            return;
        };
        let y = area.y + u16::from(shape.bottom_bar_gap());
        if y < area.y + area.height {
            frame.render_widget(
                ratatui::widgets::Paragraph::new(bar),
                Rect {
                    y,
                    height: 1,
                    ..area
                },
            );
        }
    }

    /// Rows of the pinned provider-error banner (TS `ErrorBannerComponent`,
    /// between the queued-messages band and the loader); 0 when none.
    fn error_banner_height(&self, width: u16) -> u16 {
        self.messages.pinned_error().map_or(0, |message| {
            let lines =
                crate::chat::blocks::error_banner_lines(message, usize::from(width), &self.theme);
            u16::try_from(lines.len()).unwrap_or(u16::MAX)
        })
    }

    fn render_error_banner(&self, frame: &mut Frame<'_>, area: Rect) {
        if area.height == 0 {
            return;
        }
        if let Some(message) = self.messages.pinned_error() {
            let lines = crate::chat::blocks::error_banner_lines(
                message,
                usize::from(area.width),
                &self.theme,
            );
            frame.render_widget(ratatui::widgets::Paragraph::new(lines), area);
        }
    }

    /// Whether an open overlay (or a `fullscreen` extension dialog) is full-screen and the shell renderer must
    /// hold the alternate screen (ShellRender contract); while true the
    /// controller paints [`Self::render_fullscreen`] instead of the live
    /// region.
    #[must_use]
    pub fn has_fullscreen_overlay(&self) -> bool {
        self.overlays.needs_alt_screen() || self.extension_ui.wants_fullscreen()
    }

    /// Paint the alternate-screen frame: the topmost full-screen overlay
    /// (plus popups stacked over it) into the whole terminal area at origin
    /// (0,0), then any pending extension dialog on top.
    pub fn render_fullscreen(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        self.overlays.render_fullscreen(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
    }

    /// Rows that settled since the last call and must enter terminal
    /// scrollback (`ShellRender::insert_before` source). Advances the chat
    /// watermark: committed entries are owned by the scrollback afterwards
    /// and are no longer drawn by [`Self::render_chat`]. With `rebuild =
    /// /// true` (width-change resize) returns the FULL transcript re-rendered
    /// at `width` and resets the watermark.
    /// Consume a pending scrollback erase (see [`AppEvent::SessionReset`]).
    pub fn take_clear_scrollback(&mut self) -> bool {
        std::mem::take(&mut self.clear_scrollback)
    }

    /// Drain rows for terminal scrollback (see the doc above
    /// [`Self::take_clear_scrollback`]: incremental, or the whole transcript
    /// when `rebuild`).
    ///
    /// The clean welcome is held back (drawn live by [`Self::render`]) while
    /// its startup shimmer runs; it commits with its static rows once the
    /// shimmer ends, or as soon as anything else retires or a rebuild runs.
    pub fn take_scrollback_lines(&mut self, width: u16, rebuild: bool) -> Vec<Line<'static>> {
        let transcript = if rebuild {
            self.messages.all_rows(width)
        } else {
            self.messages.take_committed_rows(width)
        };
        let hold =
            !rebuild && transcript.is_empty() && self.welcome_shimmer(Instant::now()).is_some();
        let mut rows = Vec::new();
        if !hold && self.welcome.as_ref().is_some_and(|w| rebuild || !w.emitted) {
            rows = self.welcome_rows(width, 0.0);
            if !rows.is_empty() {
                rows.push(Line::default());
            }
            if let Some(w) = self.welcome.as_mut() {
                w.emitted = true;
            }
        }
        rows.extend(transcript);
        rows
    }

    /// Render the live region (band + loader + editor + status) into
    /// `area`; `ShellRender` calls this each frame inside the inline
    /// viewport, below the committed scrollback rows.
    pub fn render_live(&mut self, frame: &mut Frame<'_>, area: Rect) {
        self.sync_frame_state();
        self.editor.set_hint(self.composer_hint());
        self.sync_vim_status();
        self.sync_composer_chrome(area.width);
        let bar_h = self.bottom_bar_rows(area.width);
        let editor_h = self
            .editor
            .desired_height(area.width)
            .min(area.height.saturating_sub(bar_h))
            .max(1);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(self.band_height()),
                Constraint::Length(self.error_banner_height(area.width)),
                Constraint::Length(u16::from(self.show_loader())),
                Constraint::Length(editor_h),
                Constraint::Length(bar_h),
                Constraint::Min(0),
            ])
            .split(area);
        self.render_band(frame, chunks[0]);
        self.render_error_banner(frame, chunks[1]);
        self.render_loader(frame, chunks[2]);
        self.editor.render(frame, chunks[3], &self.theme);
        self.render_bottom_bar(frame, chunks[4]);
        self.render_autocomplete(frame, area, chunks[3]);
        self.overlays.render(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
    }

    /// Render the full screen (full-frame fallback renderer, used until the
    /// inline shell renderer replaces it; the fallback never commits, so
    /// the watermark stays 0 and the whole transcript renders).
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        self.sync_frame_state();
        self.editor.set_hint(self.composer_hint());
        let area = frame.area();
        self.sync_composer_chrome(area.width);
        let bar_h = self.bottom_bar_rows(area.width);
        let editor_h = self
            .editor
            .desired_height(area.width)
            .min(area.height.saturating_sub(bar_h))
            .max(1);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(self.band_height()),
                Constraint::Length(self.error_banner_height(area.width)),
                Constraint::Length(u16::from(self.show_loader())),
                Constraint::Length(editor_h),
                Constraint::Length(bar_h),
            ])
            .split(area);
        self.render_chat(frame, chunks[0]);
        self.render_band(frame, chunks[1]);
        self.render_error_banner(frame, chunks[2]);
        self.render_loader(frame, chunks[3]);
        self.editor.render(frame, chunks[4], &self.theme);
        self.render_bottom_bar(frame, chunks[5]);
        self.render_autocomplete(frame, area, chunks[4]);
        self.overlays.render(frame, area, &self.theme);
        self.extension_ui.render(frame, area, &self.theme);
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
    fn render_loader(&mut self, frame: &mut Frame<'_>, area: Rect) {
        if area.height == 0 {
            return;
        }
        use ratatui::widgets::Paragraph;
        let frame_idx = (self.spinner_start.elapsed().as_millis() / u128::from(SPINNER_ADVANCE_MS))
            as usize
            % SPINNER.len();
        // Live token usage only while streaming (not during maintenance) and
        // only once there is something to show; dropped on narrow terminals.
        let usage = if area.width >= 60 {
            self.loader_usage_text()
        } else {
            None
        };
        if self.theme.clean() && self.maintenance.is_none() {
            let line = self.clean_loader_line(frame_idx, usage.as_deref());
            frame.render_widget(Paragraph::new(line), area);
            return;
        }
        let label = match self.maintenance {
            Some(Maintenance::Compaction) => " compacting… (esc to cancel)",
            Some(Maintenance::Retry) => " retrying… (esc to cancel)",
            None => " Working…",
        };
        let text = match &usage {
            Some(u) => format!("{} {label} · {u}", SPINNER[frame_idx]),
            None => format!("{} {label}", SPINNER[frame_idx]),
        };
        let line = Line::from(Span::styled(
            text,
            Style::default().fg(self.theme.color("accent")),
        ));
        frame.render_widget(Paragraph::new(line), area);
    }

    /// The live usage suffix for the loader row (`↑in ↓out (think t) · $c`),
    /// `None` until any token count is known this turn.
    fn loader_usage_text(&self) -> Option<String> {
        if !self.status.streaming {
            return None;
        }
        let input = self.turn_usage.input_tokens(self.status.context_tokens);
        let output = self.turn_usage.output_tokens();
        if input == 0 && output == 0 {
            return None;
        }
        let mut parts = vec![
            format!("↑{}", format_tokens(input)),
            format!("↓{}", format_tokens(output)),
        ];
        let think = self.turn_usage.thinking_tokens();
        if think > 0 {
            parts.push(format!("(think {})", format_tokens(think)));
        }
        if let Some((in_per_m, out_per_m)) = self.model_pricing {
            if let Some(cost) = self.turn_usage.cost(input, output, in_per_m, out_per_m) {
                parts.push(format!("${cost:.3}"));
            }
        }
        Some(parts.join(" "))
    }

    /// Re-resolve the active model's pricing (loader cost suffix) from the
    /// `/model` catalogue; `None` when the model is unknown or unpriced.
    fn refresh_model_pricing(&mut self) {
        let model = self.status.model.clone();
        self.model_pricing = self
            .overlays
            .models_mut()
            .models
            .iter()
            .find(|m| format!("{}/{}", m.provider, m.id) == model)
            .map(|m| (m.cost_input, m.cost_output))
            .filter(|(i, o)| *i > 0.0 || *o > 0.0);
    }

    /// Clean loader: spinner, what is happening, Omi's face, elapsed time.
    fn clean_loader_line(&self, frame_idx: usize, usage: Option<&str>) -> Line<'static> {
        use crate::chrome::mascot::{face, Mood};
        let tool_running = self.messages.live_tail().iter().any(|e| match &e.kind {
            crate::chat::ChatKind::Tool(card) => !card.status.settled(),
            crate::chat::ChatKind::Bash(x) | crate::chat::ChatKind::Python(x) => {
                x.status == crate::chat::ExecStatus::Running
            }
            crate::chat::ChatKind::ReadGroup(g) => g.reads.iter().any(|r| !r.status.settled()),
            _ => false,
        });
        let (mood, label) = if tool_running {
            (Mood::Working, "Working…")
        } else {
            (Mood::Thinking, "Thinking…")
        };
        let elapsed = self.spinner_start.elapsed();
        let tick = (elapsed.as_millis() / 700) as u64;
        let accent = Style::default().fg(self.theme.color("accent"));
        let dim = Style::default().fg(self.theme.color("dim"));
        let pink = Style::default().fg(ratatui::style::Color::Rgb(236, 112, 214));
        let tail = match usage {
            Some(u) => format!("{}s · {u} · esc to interrupt", elapsed.as_secs()),
            None => format!("  {}s · esc to interrupt", elapsed.as_secs()),
        };
        Line::from(vec![
            Span::styled(format!("{} ", SPINNER[frame_idx]), accent),
            Span::styled(
                label.to_owned(),
                accent.add_modifier(ratatui::style::Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(face(mood, tick).to_owned(), pink),
            Span::styled(tail, dim),
        ])
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
        let mut area = area;
        let now = Instant::now();
        if let Some(w) = self
            .welcome
            .as_mut()
            .filter(|w| !w.emitted && w.started.is_none())
        {
            w.started = Some(now);
        }
        if let Some(phase) = self.welcome_shimmer(now) {
            // Startup shimmer: the welcome sits at the top of the live
            // region — exactly where its rows land when they commit.
            let rows = self.welcome_rows(area.width, phase);
            let h = u16::try_from(rows.len())
                .unwrap_or(u16::MAX)
                .min(area.height);
            frame.render_widget(Paragraph::new(rows), Rect { height: h, ..area });
            area.y += h;
            area.height -= h;
        }
        let height = area.height as usize;
        let width = usize::from(area.width).max(8);
        // Bottom-anchored: the last `height + scroll` rows of the tail.
        let lines = self
            .messages
            .tail_rows(width, height + self.scroll as usize);
        frame.render_widget(Paragraph::new(lines), area);
    }
}

/// Live per-turn token accounting for the loader row (TS mirrors the
/// event-controller's usage tally, but reported while streaming).
///
/// Output/thinking tokens are estimated from the streamed deltas (chars/4)
/// until the provider's final `usage` arrives on the message frame, then
/// exact. Input comes from the request usage or the `context_usage`
/// estimate.
#[derive(Debug, Default, Clone)]
struct TurnUsage {
    /// Streamed char high-water marks per message id: `(text, thinking)`.
    /// `message_update` frames carry the accumulated text, so the max per
    /// id is the message total; summing across ids spans a whole turn.
    streamed: std::collections::HashMap<String, (u64, u64)>,
    /// Same marks for frames whose message has no id.
    anonymous: (u64, u64),
    /// Additive delta tallies per message id: `message_update` frames sent
    /// without a full snapshot (`delta_updates` mode) carry only the
    /// `assistantMessageEvent` delta string, so those are summed, not maxed.
    delta_streamed: std::collections::HashMap<String, (u64, u64)>,
    /// Same additive tallies for delta frames whose message has no id.
    delta_anonymous: (u64, u64),
    /// Provider-reported usage from the latest assistant frame.
    exact: Option<ExactUsage>,
}

/// Provider-reported usage (`message.usage` on assistant frames).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ExactUsage {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

const CHARS_PER_TOKEN: u64 = 4;

impl TurnUsage {
    /// Clear all per-turn tallies (turn start).
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fold one assistant message frame (streaming partial or settled).
    fn observe_message(&mut self, message: &Value) {
        let mut text = 0u64;
        let mut think = 0u64;
        if let Some(content) = message.get("content").and_then(Value::as_array) {
            for part in content {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => text += str_chars(part.get("text")),
                    Some("thinking") => think += str_chars(part.get("thinking")),
                    _ => {}
                }
            }
        }
        let slot = match message.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => self.streamed.entry(id.to_owned()).or_default(),
            _ => &mut self.anonymous,
        };
        slot.0 = slot.0.max(text);
        slot.1 = slot.1.max(think);
        // A cumulative snapshot supersedes any deltas tallied for this message
        // (`delta_updates` mode never sends both on one frame, and the settled
        // `message_end` snapshot that closes a stream is the complete total).
        self.delta_streamed.clear();
        self.delta_anonymous = (0, 0);
        // Streaming partials carry the provider's running `usage` from the first
        // chunk (e.g. output 9) and would freeze the live count; only a settled
        // message (`stopReason`) or a bare usage frame is authoritative.
        let settled = message.get("stopReason").is_some() || message.get("content").is_none();
        if let Some(usage) = message.get("usage").filter(|_| settled) {
            let get = |k: &str| usage.get(k).and_then(Value::as_u64).unwrap_or(0);
            let exact = ExactUsage {
                input: get("input"),
                output: get("output"),
                cache_read: get("cacheRead"),
                cache_write: get("cacheWrite"),
            };
            if exact.input > 0 || exact.output > 0 {
                self.exact = Some(exact);
            }
        }
    }
    /// Fold one streaming delta frame (`delta_updates` mode): the runtime
    /// stripped the cumulative `message` snapshot, leaving only
    /// `assistantMessageEvent.delta`. These add (never max) into per-message
    /// tallies; a later cumulative snapshot resets them (see `observe_message`).
    fn observe_delta(&mut self, frame: &Value) {
        let mut text = 0u64;
        let mut think = 0u64;
        if let Some(ev) = frame.get("assistantMessageEvent") {
            match ev.get("type").and_then(Value::as_str) {
                Some("text_delta") => text = str_chars(ev.get("delta")),
                Some("thinking_delta") => think = str_chars(ev.get("delta")),
                _ => return,
            }
        } else {
            return;
        }
        let slot = match frame.get("messageId").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => self.delta_streamed.entry(id.to_owned()).or_default(),
            _ => &mut self.delta_anonymous,
        };
        slot.0 += text;
        slot.1 += think;
    }

    /// Estimated output tokens so far (text + thinking chars/4) or the exact
    /// provider count. Thinking is included so the live `↓` ticks up while
    /// the model reasons (the `(think N)` suffix still breaks it out).
    fn output_tokens(&self) -> u64 {
        if let Some(exact) = self.exact {
            return exact.output;
        }
        self.estimated_output()
    }

    /// Estimated thinking tokens (chars/4); providers do not report these.
    fn thinking_tokens(&self) -> u64 {
        self.estimated_thinking()
    }

    /// Input tokens: the provider's request usage, else the `context_usage`
    /// estimate for the active turn.
    fn input_tokens(&self, context_estimate: Option<u64>) -> u64 {
        match self.exact {
            Some(exact) if exact.input > 0 => exact.input + exact.cache_read + exact.cache_write,
            _ => context_estimate.unwrap_or(0),
        }
    }

    /// Estimated turn cost in dollars from $/M pricing, `None` when free.
    fn cost(&self, input: u64, output: u64, in_per_m: f64, out_per_m: f64) -> Option<f64> {
        let cost = input as f64 / 1e6 * in_per_m + output as f64 / 1e6 * out_per_m;
        (cost > 0.0).then_some(cost)
    }

    fn estimated_output(&self) -> u64 {
        let (text, think) = self.totals();
        (text + think) / CHARS_PER_TOKEN
    }

    fn estimated_thinking(&self) -> u64 {
        let (_, think) = self.totals();
        think / CHARS_PER_TOKEN
    }

    fn totals(&self) -> (u64, u64) {
        let mut acc = self.anonymous;
        for (text, think) in self.streamed.values() {
            acc.0 += *text;
            acc.1 += *think;
        }
        acc.0 += self.delta_anonymous.0;
        acc.1 += self.delta_anonymous.1;
        for (text, think) in self.delta_streamed.values() {
            acc.0 += *text;
            acc.1 += *think;
        }
        acc
    }
}

/// Character count of an optional JSON string field.
fn str_chars(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_str)
        .map_or(0, |s| s.chars().count() as u64)
}

/// Compact token count (`1234` → `1.2k`, `18200` → `18.2k`).
fn format_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
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
/// Which notification a settled turn raises (TS `sendErrorNotification` /
/// `sendCompletionNotification`): an errored turn rings `request` ("Stopped with error") only
/// when `error.notify` is on; any other turn that was not aborted rings `done` ("Complete")
/// unless `completion.notify` is off.
fn turn_notification(
    failed: bool,
    stop_reason: Option<&str>,
    error_notify: bool,
    completion_notify: bool,
) -> Option<(&'static str, crate::chrome::notify::NotifyKind)> {
    use crate::chrome::notify::NotifyKind;
    if failed || stop_reason == Some("error") {
        return error_notify.then_some(("Stopped with error", NotifyKind::Error));
    }
    (completion_notify && stop_reason != Some("aborted"))
        .then_some(("Complete", NotifyKind::Completion))
}

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

    /// An app pinned to built-in default settings: independent of the
    /// process-global `HOME`, which the temp-HOME tests mutate.
    fn app() -> App {
        let mut app = App::new("zai/glm-5.3-flash".into(), Path::new("/repo"));
        app.set_settings_global(None);
        // Capture deliveries instead of ringing the real terminal / desktop.
        app.set_notify_hook(Arc::new(|_| {}));
        app
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
            Vec::new(),
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
                Vec::new(),
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
    fn model_picker_default_chip_dispatches_set_model_and_stays_open() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let home = with_temp_home("model-pick");
        {
            let mut app = home_app();
            let port = port();
            app.configure(
                vec![ModelEntry::new("zai", "glm", Some(128_000))],
                Vec::new(),
            );
            type_text(&mut app, &port, "/model");
            assert!(enter(&mut app, &port));
            assert_eq!(app.overlays.open(), Some(OpenOverlay::Models));
            // Enter focuses the model rows, Enter opens the role strip,
            // Enter on the first chip (default) switches the live model.
            assert!(enter(&mut app, &port));
            assert!(enter(&mut app, &port));
            assert!(enter(&mut app, &port));
            let calls = log(&port);
            assert_eq!(calls[0], "set_model(zai, glm)", "{calls:?}");
            assert!(
                calls
                    .iter()
                    .any(|c| c.starts_with("settings_set(modelRoles")),
                "{calls:?}"
            );
            // The browser stays open (TS model hub); Esc closes it.
            assert_eq!(app.overlays.open(), Some(OpenOverlay::Models));
            assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
            assert_eq!(app.overlays.open(), None);
        }
        restore_home(old_home);
        let _ = std::fs::remove_dir_all(home);
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
    fn slash_session_info_requests_report_and_opens_overlay() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::AvailableCommands(json!({
            "commands": [{"name": "session", "source": "builtin"}]
        })));
        type_text(&mut app, &port, "/session");
        assert!(enter(&mut app, &port));
        assert!(
            log(&port).contains(&"request_session_info()".to_owned()),
            "{:?}",
            log(&port)
        );
        assert!(
            !log(&port).iter().any(|c| c.starts_with("run_slash")),
            "{:?}",
            log(&port)
        );
        // Other verbs still reach the runtime.
        type_text(&mut app, &port, "/session delete");
        assert!(enter(&mut app, &port));
        assert!(
            log(&port)
                .iter()
                .any(|c| c.starts_with("run_slash(session")),
            "{:?}",
            log(&port)
        );
        // The runtime's report frame opens the full-screen panel; Esc closes it.
        app.apply_event(&crate::events::parse_frame(&json!({
            "type": "session_info",
            "lines": [{"k": "ID", "v": "abc"}, {"head": "Provider"}, {"text": ""}]
        })));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::SessionInfo));
        assert!(app.has_fullscreen_overlay());
        let s = screen(&mut app, 120, 40);
        assert!(s.contains("Session Info") && s.contains("ID: abc"), "{s}");
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(app.overlays.open(), None);
    }

    #[test]
    fn slash_usage_requests_report_and_opens_panel() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::AvailableCommands(json!({
            "commands": [{"name": "usage", "source": "builtin"}]
        })));
        type_text(&mut app, &port, "/usage");
        assert!(enter(&mut app, &port));
        assert!(
            log(&port).contains(&"request_usage()".to_owned()),
            "{:?}",
            log(&port)
        );
        assert!(
            !log(&port).iter().any(|c| c.starts_with("run_slash")),
            "{:?}",
            log(&port)
        );
        type_text(&mut app, &port, "/usage reset");
        assert!(enter(&mut app, &port));
        assert!(
            log(&port).iter().any(|c| c.starts_with("run_slash(usage")),
            "{:?}",
            log(&port)
        );
        app.apply_event(&crate::events::parse_frame(&json!({
            "type": "session_info", "title": "Usage",
            "lines": [{"head": "Provider limits (anthropic)"}, {"k": "5h", "v": "6%"}]
        })));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::SessionInfo));
        let s = screen(&mut app, 120, 40);
        assert!(
            s.contains("Usage") && !s.contains("Session Info") && s.contains("5h: 6%"),
            "{s}"
        );
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(app.overlays.open(), None);
    }

    #[test]
    fn slash_settings_opens_overlay_listing_sections() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
        let s = screen(&mut app, 150, 30);
        // Fullscreen tab strip (TS labels) and the first tab's section
        // headings (sidebar + pane).
        assert!(s.contains("Appearance"), "{s}");
        assert!(s.contains("Model") && s.contains("Providers"), "{s}");
        assert!(s.contains("Theme"), "{s}");
        // Row labels render with their values (values are seeded from the
        // layered config, so assert the stable label half).
        assert!(s.contains("Dark Theme") && s.contains("Light Theme"), "{s}");
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

    use crate::test_support::HOME_LOCK;

    /// An app reading the user's config through `HOME`, for the temp-HOME
    /// tests (call only while holding [`HOME_LOCK`]).
    fn home_app() -> App {
        App::new("zai/glm-5.3-flash".into(), Path::new("/repo"))
    }

    fn with_temp_home(test: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omp-tui-settings-{test}-{}", std::process::id()));
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
            let mut app = home_app();
            let port = port();
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            // Defaults before the edit.
            assert_eq!(
                app.overlays.settings_mut().value_of("colorBlindMode"),
                Some(json!(false))
            );
            // Cycle the bool row with ⏎ (TS confirm; → switches tabs).
            assert!(app.overlays.settings_mut().select("colorBlindMode"));
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
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
                logged
                    .iter()
                    .any(|entry| entry.starts_with("settings_set(colorBlindMode")),
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
            let mut app = home_app();
            let port = port();
            app.set_appearance(crate::theme::Appearance::Dark);
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            assert!(app.overlays.settings_mut().select("theme.dark"));
            // Submenu row: Enter opens the theme select field with the
            // current theme highlighted; ↓ to "light", Enter picks.
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            let names: Vec<String> = crate::theme::available_themes()
                .into_iter()
                .map(|t| t.name)
                .collect();
            let from = names.iter().position(|n| n == "titanium").unwrap();
            let to = names.iter().position(|n| n == "light").unwrap();
            for _ in 0..(to + names.len() - from) % names.len() {
                assert!(press(&mut app, &port, key(CtKeyCode::Down)));
            }
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
            // A notice confirms the live apply; it renders in the chat once
            // the fullscreen overlay closes.
            assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
            assert_eq!(app.overlays.open(), None);
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

    /// Every row of every tab is editable like TS and the edit persists to
    /// config.yml (temp HOME): bool toggle, enum cycle, submenu pick, text /
    /// JSON input, multiselect toggle.
    #[test]
    fn every_settings_row_is_editable_and_persists() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let home = with_temp_home("all-rows");
        {
            let mut app = home_app();
            let port = port();
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            let rows = app.overlays.settings_mut().shown_rows();
            assert!(
                rows.len() > 300,
                "TS panel lists ~380 rows, got {}",
                rows.len()
            );
            let kinds: std::collections::BTreeSet<&str> = rows.iter().map(|(_, k)| *k).collect();
            for kind in ["bool", "enum", "submenu", "text", "multi", "limits"] {
                assert!(kinds.contains(kind), "no {kind} row in the panel");
            }
            let mut unchanged: Vec<String> = Vec::new();
            for (id, kind) in &rows {
                let overlay = app.overlays.settings_mut();
                assert!(overlay.select(id), "{id}: not selectable");
                let before = overlay.value_of(id);
                assert!(press(&mut app, &port, key(CtKeyCode::Enter)), "{id}: enter");
                match *kind {
                    // Enter/Space flip the bool / cycle the enum in place.
                    "bool" | "enum" => {}
                    // Submenu: ↓ moves to another choice, Enter picks it.
                    "submenu" => {
                        assert!(
                            app.overlays.settings_mut().is_editing(),
                            "{id}: submenu did not open"
                        );
                        assert!(press(&mut app, &port, key(CtKeyCode::Down)));
                        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
                    }
                    // Text field: type a value of the right shape, Enter saves.
                    "text" | "limits" => {
                        assert!(
                            app.overlays.settings_mut().is_editing(),
                            "{id}: text field did not open"
                        );
                        let schema = if *kind == "limits" {
                            "record"
                        } else {
                            text_schema(id)
                        };
                        // The field opens pre-filled: clear it first.
                        for _ in 0..400 {
                            press(&mut app, &port, key(CtKeyCode::Backspace));
                        }
                        let input = match schema {
                            "array" => "[\"e2e\"]",
                            "record" if *kind == "limits" => "{\"openai\": 3}",
                            "record" => "{\"e2e\": \"1\"}",
                            _ => "e2e-value",
                        };
                        type_text(&mut app, &port, input);
                        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
                    }
                    // Multiselect: Enter toggles the first choice, Esc goes back.
                    "multi" => {
                        assert!(
                            app.overlays.settings_mut().is_editing(),
                            "{id}: multiselect did not open"
                        );
                        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
                        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
                    }
                    other => panic!("{id}: unknown kind {other}"),
                }
                let overlay = app.overlays.settings_mut();
                assert!(!overlay.is_editing(), "{id} ({kind}): field still open");
                let after = overlay.value_of(id);
                if after == before {
                    unchanged.push(format!("{id} ({kind})"));
                    continue;
                }
                // Persisted: a fresh layered read (config.yml + env) agrees.
                let stored = port.settings_get(std::slice::from_ref(id))[0].1.clone();
                assert_eq!(Some(stored), after, "{id} ({kind}): not persisted");
            }
            assert!(
                unchanged.is_empty(),
                "rows whose edit did not change the value: {unchanged:?}"
            );
            let config = std::fs::read_to_string(home.join(".omp/agent/config.yml")).unwrap();
            assert!(config.contains("composer:"), "{config}");
        }
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old_home);
    }

    /// Registry value type of a text row (`array` JSON lists vs plain strings).
    fn text_schema(id: &str) -> &'static str {
        match omp_settings::lookup(id).map(|d| d.ty) {
            Some(omp_settings::SettingType::Array) => "array",
            Some(omp_settings::SettingType::Record) => "record",
            _ => "string",
        }
    }

    /// Composer Shape edit changes the live editor look at once and persists
    /// (TS `syncComposerShape`).
    #[test]
    fn composer_shape_edit_changes_editor_live_and_persists() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let home = with_temp_home("shape");
        {
            let mut app = home_app();
            let port = port();
            assert_eq!(app.editor.shape(), crate::composer::ComposerShape::Band);
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            assert!(app.overlays.settings_mut().select("composer.shape"));
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            // band → box (second choice).
            assert!(press(&mut app, &port, key(CtKeyCode::Down)));
            let s = screen(&mut app, 120, 40);
            assert!(s.contains("Rounded Box"), "{s}");
            assert!(s.contains("Preview:"), "{s}");
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            assert_eq!(app.editor.shape(), crate::composer::ComposerShape::Box);
            let config = std::fs::read_to_string(home.join(".omp/agent/config.yml")).unwrap();
            assert!(config.contains("shape: box"), "{config}");
            assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
            // The live region now draws the box frame around the prompt.
            let s = screen(&mut app, 80, 12);
            assert!(s.contains('╭') && s.contains('╯'), "{s}");
            // Re-seeding from disk (a fresh start) restores the same shape.
            let mut fresh = App::new("p/m".into(), Path::new("/nonexistent"));
            fresh.configure(Vec::new(), Vec::new());
            assert_eq!(fresh.editor.shape(), crate::composer::ComposerShape::Box);
        }
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old_home);
    }

    /// Status-line settings apply live to the real status line.
    #[test]
    fn status_line_settings_apply_live() {
        let _guard = HOME_LOCK.lock();
        let old_home = std::env::var_os("HOME");
        let home = with_temp_home("statusline");
        {
            let mut app = home_app();
            let port = port();
            type_text(&mut app, &port, "/settings");
            assert!(enter(&mut app, &port));
            assert!(app
                .overlays
                .settings_mut()
                .select("statusLine.compactThinkingLevel"));
            assert!(app.status.config.compact_thinking);
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            assert!(!app.status.config.compact_thinking);
            assert!(app.overlays.settings_mut().select("statusLine.separator"));
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            assert!(press(&mut app, &port, key(CtKeyCode::Down)));
            assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
            assert_ne!(
                app.status.config.separator,
                crate::status_line::SeparatorStyle::PowerlineThin
            );
        }
        let _ = std::fs::remove_dir_all(&home);
        restore_home(old_home);
    }

    /// Shift+Tab cycles off → auto → efforts → ultracode → off and updates
    /// the status line.
    #[test]
    fn shift_tab_cycles_thinking_effort() {
        let mut app = app();
        let port = port();
        // A model without thinking: nothing to cycle.
        let shift_tab = KeyEvent::new(CtKeyCode::BackTab, CtMods::SHIFT);
        assert!(press(&mut app, &port, shift_tab));
        assert!(!log(&port)
            .iter()
            .any(|c| c.starts_with("set_thinking_level")));
        // Reasoning model: context_usage announces the supported efforts.
        app.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "p/m", "thinkingLevel": "high",
            "thinkingLevels": ["minimal", "low", "medium", "high"], "autoThinking": false,
            "contextWindow": 1000,
        })));
        assert_eq!(app.status.thinking.as_deref(), Some("high"));
        let mut seen = Vec::new();
        for _ in 0..7 {
            assert!(press(&mut app, &port, shift_tab));
            let last = log(&port).last().cloned().unwrap();
            let level = last
                .trim_start_matches("set_thinking_level(")
                .trim_end_matches(')')
                .to_owned();
            // The runtime echoes the change (ultra keeps the effort level).
            let configured = if level == "auto" {
                json!("auto")
            } else {
                json!(level)
            };
            let ultra = (level == "ultracode").then_some("ultracode");
            app.apply_event(&AppEvent::ThinkingLevelChanged(json!({
                "type": "thinking_level_changed",
                "thinkingLevel": match level.as_str() { "auto" => "medium", "ultracode" => "high", l => l },
                "configured": configured,
                "ultra": ultra,
            })));
            seen.push(level);
        }
        assert_eq!(
            seen,
            [
                "ultracode",
                "off",
                "auto",
                "minimal",
                "low",
                "medium",
                "high"
            ]
        );
        assert_eq!(app.status.thinking.as_deref(), Some("high"));
    }

    /// Bare `/effort` opens the thinking selector; Enter applies the pick.
    #[test]
    fn effort_selector_opens_and_applies_pick() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "p/m", "thinkingLevel": "low",
            "thinkingLevels": ["minimal", "low", "medium"], "contextWindow": 1000,
        })));
        type_text(&mut app, &port, "/effort");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Thinking));
        assert!(press(&mut app, &port, key(CtKeyCode::Down)));
        assert!(press(&mut app, &port, key(CtKeyCode::Enter)));
        assert!(
            log(&port).iter().any(|c| c == "set_thinking_level(medium)"),
            "{:?}",
            log(&port)
        );
        assert_eq!(app.overlays.open(), None);
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
    fn fullscreen_overlays_want_alt_screen_and_mouse_then_restore_on_esc() {
        use crossterm::event::{MouseEvent, MouseEventKind};
        let mut app = app();
        let port = port();
        assert!(!app.has_fullscreen_overlay());
        assert!(!app.wants_mouse(), "tui.mouse defaults off");
        type_text(&mut app, &port, "/settings");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Settings));
        assert!(app.has_fullscreen_overlay());
        assert!(
            app.wants_mouse(),
            "TS fullscreen overlays enable 1000/1003/1006"
        );
        // The alt-screen frame paints the overlay over the whole terminal.
        let mut t = Terminal::new(TestBackend::new(60, 20)).unwrap();
        t.draw(|f| app.render_fullscreen(f)).unwrap();
        let buf = t.backend().buffer().clone();
        let used = (0..20u16)
            .filter(|&y| (0..60u16).any(|x| buf[(x, y)].symbol() != " "))
            .count();
        assert!(
            used >= 18,
            "fullscreen overlay should fill the screen, used {used}/20"
        );
        // Mouse reports go to the overlay, never to the editor as input.
        let wheel = Event::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 3,
            row: 3,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        assert!(app.handle_terminal_event(&wheel, &port));
        assert_eq!(app.editor.text(), "");
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        assert_eq!(app.overlays.open(), None);
        assert!(!app.has_fullscreen_overlay());
        assert!(!app.wants_mouse());
        // Popups (help) stay inline without mouse tracking.
        app.overlays.toggle(OpenOverlay::Help);
        assert!(!app.has_fullscreen_overlay());
        assert!(!app.wants_mouse());
    }

    #[test]
    fn typing_slash_shows_command_popup_above_editor() {
        // Regression: `/` showed nothing — the popup was never wired.
        let mut app = app();
        let port = port();
        with_catalog(&mut app);
        type_text(&mut app, &port, "/");
        let s = screen(&mut app, 60, 14);
        assert!(
            s.contains("help") && s.contains("model") && s.contains("compact"),
            "{s}"
        );
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
        assert!(
            log(&port).iter().any(|l| l.contains("info")),
            "{:?}",
            log(&port)
        );
    }

    #[test]
    fn command_output_and_prompt_errors_reach_the_transcript() {
        let mut app = app();
        app.apply_event(&parse_frame(
            &json!({"type": "command_output", "text": "Session: abc\nTitle: t\nCWD: /tmp"}),
        ));
        app.apply_event(&parse_frame(
            &json!({"type": "prompt_result", "status": "error", "error": {"message": "Unknown command: /zz"}}),
        ));
        let s = screen(&mut app, 60, 14);
        assert!(s.contains("Session: abc") && !s.contains("abcTitle"), "{s}");
        assert!(
            s.lines().any(|l| l.trim_start().starts_with("CWD: /tmp")),
            "line breaks kept: {s}"
        );
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

    /// Regression: a prompt submitted while streaming shows only in the
    /// steering band, not also as a chat message (it entered both, so the user
    /// saw the prompt "sent and queued" at once).
    #[test]
    fn steer_while_streaming_is_not_echoed_into_chat() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        let users_before = app
            .messages
            .entries()
            .iter()
            .filter(|e| e.role == ChatRole::User)
            .count();
        type_text(&mut app, &port, "steer me");
        assert!(enter(&mut app, &port));
        assert_eq!(app.steering, vec!["steer me".to_string()]);
        let users_after = app
            .messages
            .entries()
            .iter()
            .filter(|e| e.role == ChatRole::User)
            .count();
        assert_eq!(
            users_after, users_before,
            "queued prompt must not be echoed into the chat"
        );
        assert!(
            log(&port).iter().any(|c| c.contains("steer me")),
            "{:?}",
            log(&port)
        );
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

    /// Regression: Esc during a run with an already-dispatched queued steer pulls it
    /// out of the runtime queue into the editor before aborting (TS
    /// `restoreQueuedMessagesToEditor({abort: true})`). Left queued, the post-abort
    /// drain ran it under the interrupt and it ended as "Request was aborted".
    #[test]
    fn esc_while_streaming_restores_dispatched_queue_before_abort() {
        let mut app = app();
        let port = port();
        start_streaming(&mut app);
        app.steering.push("как успехи".into());
        type_text(&mut app, &port, "draft");
        assert!(press(&mut app, &port, key(CtKeyCode::Esc)));
        let calls = log(&port);
        assert_eq!(
            calls
                .iter()
                .filter(|c| c.as_str() != "abort()")
                .cloned()
                .collect::<Vec<_>>(),
            vec![format!("remove_queued({:?}, Steering)", "как успехи")]
        );
        assert_eq!(calls.last().map(String::as_str), Some("abort()"));
        assert_eq!(app.editor.text(), "как успехи\ndraft");
        assert!(app.steering.is_empty());
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
        // Band composer: status line on top, then the `╰─ ` prompt row.
        let status: String = (0..80).map(|x| buf[(x, 22)].symbol()).collect();
        assert!(status.contains("zai/glm-5.3-flash"), "status: {status:?}");
        let editor: String = (0..80).map(|x| buf[(x, 23)].symbol()).collect();
        assert!(editor.starts_with("╰─"), "editor: {editor:?}");
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
        // rows: 10 - 1 editor - 1 status - 1 loader = 7 chat
        let row: String = (0..80).map(|x| buf[(x, 7)].symbol()).collect();
        assert!(row.contains("Working…"), "row: {row:?}");
    }

    #[test]
    fn format_tokens_compacts() {
        assert_eq!(format_tokens(0), "0");
        assert_eq!(format_tokens(999), "999");
        assert_eq!(format_tokens(1234), "1.2k");
        assert_eq!(format_tokens(18_200), "18.2k");
        assert_eq!(format_tokens(1_400), "1.4k");
        assert_eq!(format_tokens(1_500_000), "1.5M");
    }

    #[test]
    fn turn_usage_estimates_from_deltas_then_exact() {
        let mut usage = TurnUsage::default();
        // 1200 text chars + 3600 thinking chars → (1200+3600)/4 = 1200 out.
        let text = "a".repeat(1200);
        let think = "t".repeat(3600);
        usage.observe_message(&json!({
            "id": "m1", "role": "assistant",
            "content": [{"type": "text", "text": text}, {"type": "thinking", "thinking": think}]
        }));
        assert_eq!(usage.output_tokens(), 1200);
        assert_eq!(usage.thinking_tokens(), 900);
        // No provider usage yet: input falls back to the context estimate.
        assert_eq!(usage.input_tokens(Some(18_200)), 18_200);
        // Second message accumulates per turn.
        usage.observe_message(&json!({
            "id": "m2", "role": "assistant",
            "content": [{"type": "text", "text": "b".repeat(400)}]
        }));
        assert_eq!(usage.estimated_output(), 1300);
        // Provider usage wins over the estimate once it arrives.
        usage.observe_message(&json!({
            "id": "m1", "role": "assistant", "usage": {"input": 100, "output": 500, "cacheRead": 50}
        }));
        assert_eq!(usage.output_tokens(), 500);
        assert_eq!(usage.input_tokens(None), 150);
        // Pricing math: 150 in @ $2/M + 500 out @ $8/M = $0.0043.
        let cost = usage.cost(150, 500, 2.0, 8.0).unwrap();
        assert!((cost - 0.0043).abs() < 1e-9, "cost: {cost}");
    }
    #[test]
    fn turn_usage_deltas_add_thinking_and_text() {
        let mut usage = TurnUsage::default();
        // Thinking deltas (`delta_updates` mode: no cumulative `message`).
        usage.observe_delta(&json!({
            "type": "message_update", "messageId": "m1",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "t".repeat(3600)}
        }));
        assert_eq!(usage.output_tokens(), 900);
        // Text deltas add into the same message.
        usage.observe_delta(&json!({
            "type": "message_update", "messageId": "m1",
            "assistantMessageEvent": {"type": "text_delta", "delta": "a".repeat(1200)}
        }));
        assert_eq!(usage.output_tokens(), 1200);
        assert_eq!(usage.thinking_tokens(), 900);
        // A settled cumulative snapshot resets the deltas (no double count).
        usage.observe_message(&json!({
            "id": "m1", "role": "assistant",
            "content": [{"type": "thinking", "thinking": "t".repeat(3600)},
                        {"type": "text", "text": "a".repeat(1200)}]
        }));
        assert_eq!(usage.output_tokens(), 1200);
        assert_eq!(usage.thinking_tokens(), 900);
    }

    #[test]
    fn loader_row_counts_thinking_deltas_live() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::AgentStart);
        app.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "zai/glm-5.3-flash", "tokens": 1234
        })));
        let render_row = |app: &mut App| -> String {
            let backend = TestBackend::new(80, 10);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal.draw(|f| app.render(f)).unwrap();
            (0..80)
                .map(|x| terminal.backend().buffer()[(x, 7)].symbol())
                .collect()
        };
        // Thinking-only deltas: `↓` grows with each delta.
        app.apply_event(&AppEvent::MessageUpdate(json!({
            "type": "message_update", "messageId": "m1",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "t".repeat(1800)}
        })));
        let row1 = render_row(&mut app);
        assert!(row1.contains("↓450"), "first row: {row1:?}");
        assert!(row1.contains("(think 450)"), "first row: {row1:?}");
        app.apply_event(&AppEvent::MessageUpdate(json!({
            "type": "message_update", "messageId": "m1",
            "assistantMessageEvent": {"type": "thinking_delta", "delta": "t".repeat(1800)}
        })));
        let row2 = render_row(&mut app);
        assert!(row2.contains("↓900"), "second row: {row2:?}");
        assert!(row2.contains("(think 900)"), "second row: {row2:?}");
    }

    #[test]
    fn loader_row_shows_live_token_usage() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::AgentStart);
        app.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "zai/glm-5.3-flash", "tokens": 1234
        })));
        // Stream ~1200 output tokens (~300 text + ~900 thinking).
        app.apply_event(&AppEvent::MessageUpdate(json!({
            "message": {"id": "m1", "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "t".repeat(3600)},
                                    {"type": "text", "text": "a".repeat(1200)}]}
        })));
        let backend = TestBackend::new(80, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let row: String = (0..80).map(|x| buf[(x, 7)].symbol()).collect();
        assert!(row.contains("↑1.2k"), "row: {row:?}");
        assert!(row.contains("↓1.2k"), "row: {row:?}");
        assert!(row.contains("(think 900)"), "row: {row:?}");
    }

    #[test]
    fn loader_usage_drops_on_narrow_terminals() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::AgentStart);
        app.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "zai/glm-5.3-flash", "tokens": 1234
        })));
        app.apply_event(&AppEvent::MessageUpdate(json!({
            "message": {"id": "m1", "role": "assistant",
                        "content": [{"type": "text", "text": "a".repeat(1200)}]}
        })));
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer();
        let row: String = (0..40).map(|x| buf[(x, 7)].symbol()).collect();
        assert!(!row.contains("↑1.2k"), "row: {row:?}");
        assert!(row.contains("Working…"), "row: {row:?}");
    }

    #[test]
    fn turn_usage_resets_each_turn() {
        let mut app = app();
        let _port = port();
        app.apply_event(&AppEvent::AgentStart);
        app.apply_event(&AppEvent::MessageUpdate(json!({
            "message": {"id": "m1", "role": "assistant",
                        "content": [{"type": "text", "text": "a".repeat(1200)}]}
        })));
        assert_eq!(app.turn_usage.output_tokens(), 300);
        app.apply_event(&AppEvent::TurnStart);
        assert_eq!(app.turn_usage.output_tokens(), 0);
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
        // 6 - 1 editor - 1 status - 1 loader = 3 chat rows; loader at row 3
        let row: String = (0..80).map(|x| buf[(x, 3)].symbol()).collect();
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
        // Band composer: the status line row, then the `╰─ ` prompt row.
        let editor_row: String = (0..80).map(|x| buf[(x, 1)].symbol()).collect();
        assert!(editor_row.starts_with("╰─"), "row: {editor_row:?}");
        let status_row: String = (0..80).map(|x| buf[(x, 0)].symbol()).collect();
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
        app.configure(Vec::new(), vec![SessionSource::new(&dir)]);
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
    fn slash_resume_beats_description_matches_in_the_popup() {
        // Regression: `/resume` was not in the popup catalog, so Enter
        // accepted `/pin` (its description mentions the resume list).
        let mut app = app();
        let port = port();
        let dir = std::env::temp_dir().join(format!("omp-resume-popup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        app.configure(Vec::new(), vec![SessionSource::new(&dir)]);
        app.apply_event(&parse_frame(&json!({
            "type": "available_commands_update",
            "commands": [{"name": "pin", "description": "Pin this session to the top of the resume list",
                          "source": "builtin"}],
        })));
        type_text(&mut app, &port, "/resume");
        assert!(enter(&mut app, &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::Sessions));
        assert!(
            !log(&port).iter().any(|c| c.contains("pin")),
            "calls: {:?}",
            log(&port)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn session_loaded_rebuilds_transcript_from_history() {
        let mut app = app();
        app.push_system("stale notice".into());
        app.apply_event(&parse_frame(&json!({
            "type": "session_loaded",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "old question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "old answer"}],
                 "stopReason": "stop"},
            ],
        })));
        let texts: Vec<String> = app
            .messages
            .entries()
            .iter()
            .map(|e| e.text.clone())
            .collect();
        assert_eq!(texts.len(), 2, "history replaces the transcript: {texts:?}");
        assert!(texts[0].contains("old question"), "{texts:?}");
        assert!(texts[1].contains("old answer"), "{texts:?}");
        // The loaded history is settled and flows into scrollback.
        let rows = app.take_scrollback_lines(80, false);
        let joined: String = rows
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
            .collect();
        assert!(joined.contains("old answer"), "{joined}");
    }

    /// Regression: `/new` left the old conversation on screen. `session_reset`
    /// empties the transcript and asks the controller to erase scrollback once.
    #[test]
    fn session_reset_clears_transcript_and_requests_scrollback_erase() {
        let mut app = app();
        app.apply_event(&parse_frame(&json!({
            "type": "session_loaded",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "old question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "old answer"}],
                 "stopReason": "stop"},
            ],
        })));
        let _ = app.take_scrollback_lines(80, false);
        assert!(!app.take_clear_scrollback());
        app.apply_event(&parse_frame(&json!({"type": "session_reset"})));
        assert!(
            app.messages.entries().is_empty(),
            "transcript must be empty after /new"
        );
        assert!(app.take_clear_scrollback(), "scrollback erase requested");
        assert!(!app.take_clear_scrollback(), "erase is one-shot");
        let joined: String = app
            .take_scrollback_lines(80, true)
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
            .collect();
        assert!(!joined.contains("old answer"), "{joined}");
    }

    #[test]
    fn session_loaded_name_sets_title_label_and_owned_path_matches() {
        let mut app = app();
        let before = app.title_label.clone();
        // No name: label unchanged.
        app.push_frame_owned(
            json!({"type": "session_loaded", "messages": [], "name": null}),
            Instant::now(),
        );
        app.pump_events(Instant::now());
        assert_eq!(app.title_label, before);
        // Named: label follows the restored name; history moved, not lost.
        app.push_frame_owned(
            json!({"type": "session_loaded", "name": "Old title", "messages": [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
            ]}),
            Instant::now(),
        );
        app.pump_events(Instant::now());
        assert_eq!(app.title_label, "Old title");
        assert_eq!(app.messages.entries().len(), 1);
        // Borrowed parse agrees with the owned parse.
        let frame = json!({"type": "session_loaded", "name": "n", "messages": [{"role": "user"}]});
        assert_eq!(parse_frame(&frame), parse_frame_owned(frame.clone()));
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
        assert_ne!(app.theme.color("accent"), Theme::default().color("accent"));
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
            vec!["omp: Complete".to_string()]
        );
    }

    #[test]
    fn turn_notification_follows_ts_gates() {
        use crate::chrome::notify::NotifyKind;
        assert_eq!(
            turn_notification(false, Some("stop"), false, true),
            Some(("Complete", NotifyKind::Completion))
        );
        assert_eq!(
            turn_notification(false, None, false, true),
            Some(("Complete", NotifyKind::Completion))
        );
        assert_eq!(turn_notification(false, Some("aborted"), true, true), None);
        assert_eq!(turn_notification(false, Some("stop"), true, false), None);
        assert_eq!(
            turn_notification(false, Some("error"), true, true),
            Some(("Stopped with error", NotifyKind::Error))
        );
        assert_eq!(
            turn_notification(true, None, true, true),
            Some(("Stopped with error", NotifyKind::Error))
        );
        // error.notify defaults to off: an errored turn is silent, and never reads as "Complete".
        assert_eq!(turn_notification(true, Some("error"), false, true), None);
    }

    #[test]
    fn failed_prompt_result_fires_error_notification() {
        let mut app = app();
        app.apply_event(&AppEvent::PromptResult {
            id: None,
            success: false,
            error: Some("kaboom".into()),
            frame: json!({"type":"prompt_result","status":"error","error":{"message":"kaboom"}}),
        });
        // `error.notify` defaults to off (TS): silent, and never a "Complete" toast.
        assert!(app
            .notify_sink()
            .record()
            .iter()
            .all(|l| l != "omp: Complete"));
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
        assert_eq!(got.lock().clone(), vec!["Complete".to_string()]);
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

    // ---- omp-rs: OMT welcome, τ, ultra, magic keywords -------------------

    fn clean_app() -> App {
        let mut app = app();
        let mut theme = Theme::default();
        theme.ui = crate::theme::UiStyle::Clean;
        app.set_theme(theme);
        app
    }

    fn reasoning(app: &mut App, ultra: &Value) {
        app.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "p/m", "thinkingLevel": "high",
            "thinkingLevels": ["minimal", "low", "medium", "high"], "autoThinking": false,
            "contextWindow": 1000, "ultra": ultra,
        })));
    }

    /// Any ultra mode (hybrids too) is followed by `off`; snapshots carry
    /// `ultra` and `null` clears it.
    #[test]
    fn ultra_state_mirrors_frames_and_cycles_to_off() {
        let mut app = app();
        let port = port();
        reasoning(&mut app, &json!("ultradeep"));
        assert_eq!(app.status.ultra, Some(crate::magic::Ultra::Deep));
        assert_eq!(app.current_effort_selector(), "ultradeep");
        assert!(press(
            &mut app,
            &port,
            KeyEvent::new(CtKeyCode::BackTab, CtMods::SHIFT)
        ));
        assert_eq!(
            log(&port).last().map(String::as_str),
            Some("set_thinking_level(off)")
        );
        // Frames without the key leave it; `null` clears it.
        app.apply_event(&AppEvent::ThinkingLevelChanged(json!({
            "type": "thinking_level_changed", "thinkingLevel": "high"
        })));
        assert_eq!(app.status.ultra, Some(crate::magic::Ultra::Deep));
        app.apply_event(&AppEvent::ThinkingLevelChanged(json!({
            "type": "thinking_level_changed", "thinkingLevel": "off", "ultra": null
        })));
        assert_eq!(app.status.ultra, None);
        // A model that cannot think offers no ultracode either.
        let mut plain = self::app();
        plain.apply_event(&AppEvent::ContextUsage(json!({
            "type": "context_usage", "model": "p/m", "thinkingLevels": []
        })));
        assert!(plain.effort_selectors().is_empty());
        assert_eq!(
            app.effort_selectors().last().map(String::as_str),
            Some("ultracode")
        );
    }

    /// `/ultra …` is a runtime slash command once the catalog lists it.
    #[test]
    fn slash_ultra_reaches_the_runtime() {
        let mut app = app();
        let port = port();
        app.apply_event(&AppEvent::AvailableCommands(json!({
            "commands": [{"name": "ultra", "description": "Ultra mode", "source": "builtin"}]
        })));
        assert!(app
            .autocomplete
            .commands()
            .iter()
            .any(|c| c.name == "ultra"));
        type_text(&mut app, &port, "/ultra deep");
        app.autocomplete.close();
        assert!(enter(&mut app, &port));
        assert!(
            log(&port).iter().any(|l| l == "run_slash(ultra, \"deep\")"),
            "{:?}",
            log(&port)
        );
    }

    /// Clean style renders the ultra mode in the status line.
    #[test]
    fn clean_status_shows_ultra_and_tau() {
        let mut app = clean_app();
        reasoning(&mut app, &json!("ultracode"));
        let s = screen(&mut app, 120, 12);
        assert!(s.contains("◉ ultracode"), "{s}");
        assert!(s.contains("τ∿"), "{s}");
        assert!(s.lines().any(|l| l.trim_start().starts_with("τ ·")), "{s}");
    }

    /// τ colour follows the run state: idle, running, blocked, error.
    #[test]
    fn brand_state_tracks_run_dialogs_and_errors() {
        use crate::status_line::Brand;
        let mut app = clean_app();
        assert_eq!(app.brand_state(), Brand::Idle);
        app.apply_event(&AppEvent::AgentStart);
        assert_eq!(app.brand_state(), Brand::Running);
        app.apply_event(&AppEvent::ExtensionUiRequest(json!({
            "type": "extension_ui_request", "id": "q1", "method": "confirm",
            "title": "Allow?", "message": "run it?"
        })));
        assert_eq!(app.brand_state(), Brand::Blocked);
        app.extension_ui.complete("q1");
        app.apply_event(&AppEvent::MessageEnd(json!({
            "type": "message_end",
            "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "boom"}
        })));
        app.apply_event(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
        assert_eq!(app.brand_state(), Brand::Error);
        app.apply_event(&AppEvent::AgentStart);
        assert_eq!(app.brand_state(), Brand::Running);
        app.apply_event(&AppEvent::AgentEnd(json!({"type": "agent_end"})));
        assert_eq!(app.brand_state(), Brand::Idle);
        // Painted into the status line on the next frame.
        let _ = screen(&mut app, 100, 10);
        assert_eq!(app.status.brand, Brand::Idle);
    }

    /// The clean welcome stays live while it shimmers, then commits its
    /// static rows; anything retiring earlier commits it immediately.
    #[test]
    fn clean_welcome_shimmers_live_then_commits_static_rows() {
        let mut app = clean_app();
        app.set_welcome("9.9.9");
        assert!(
            app.take_scrollback_lines(100, false).is_empty(),
            "held while shimmering"
        );
        assert_eq!(
            app.animation_interval(),
            Some(Duration::from_millis(crate::chat::reveal::FRAME_MS))
        );
        let s = screen(&mut app, 100, 14);
        assert!(
            s.lines().nth(1).is_some_and(|l| l.contains("OMT")),
            "drawn at the top of the live region:\n{s}"
        );
        // Time is up: the static rows commit, the live region drops them.
        let w = app.welcome.as_mut().unwrap();
        w.started = w
            .started
            .map(|s| s - Duration::from_millis(welcome::SHIMMER_MS));
        assert!(w.started.is_some(), "the clock starts at the first paint");
        assert_eq!(app.animation_interval(), None);
        let rows = app.take_scrollback_lines(100, false);
        let tip = app.welcome.as_ref().unwrap().tip.clone();
        let mut expected = welcome::clean_welcome_lines("9.9.9", &tip, 100, &app.theme, 0.0);
        expected.push(Line::default());
        assert_eq!(rows, expected);
        assert!(!screen(&mut app, 100, 14).contains("OMT"));
        assert!(app.take_scrollback_lines(100, false).is_empty());

        // A prompt retiring mid-shimmer commits the welcome first.
        let mut app = clean_app();
        let port = port();
        app.set_welcome("9.9.9");
        type_text(&mut app, &port, "hello");
        assert!(enter(&mut app, &port));
        let text: Vec<String> = app
            .take_scrollback_lines(100, false)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(text.iter().any(|r| r.contains("OMT")), "{text:#?}");
        let omt = text.iter().position(|r| r.contains("OMT")).unwrap();
        let hello = text.iter().position(|r| r.contains("hello")).unwrap();
        assert!(omt < hello);

        // `/new` (session_reset) rebuild re-commits the static welcome.
        let mut app = clean_app();
        app.set_welcome("9.9.9");
        app.apply_event(&AppEvent::SessionReset);
        assert!(app.take_clear_scrollback());
        let rows = app.take_scrollback_lines(100, true);
        assert!(rows
            .iter()
            .any(|l| l.spans.iter().any(|s| s.content == "O")));
    }

    /// Classic keeps committing the TS lockup at once (no shimmer).
    #[test]
    fn classic_welcome_commits_immediately() {
        let mut app = app();
        app.set_welcome("1.0.0");
        assert!(!app.take_scrollback_lines(120, false).is_empty());
        assert_eq!(app.animation_interval(), None);
    }

    /// Clean style brands the terminal title with τ, classic with π.
    #[test]
    fn title_brand_follows_the_style() {
        let app = clean_app();
        assert!(
            app.title()
                .current_title()
                .is_some_and(|t| t.starts_with("τ > ")),
            "{:?}",
            app.title().current_title()
        );
        let classic = self::app();
        assert!(classic
            .title()
            .current_title()
            .is_some_and(|t| t.starts_with("π > ")));
    }

    /// The editor paints magic keywords and shimmers only while one is typed.
    #[test]
    fn editor_paints_magic_keywords_and_shimmers_only_with_one() {
        let mut app = app();
        let port = port();
        type_text(&mut app, &port, "plain text");
        assert_eq!(app.animation_interval(), None, "no keyword, no timer");
        app.set_editor_text("please ultrathink now");
        assert_eq!(
            app.animation_interval(),
            Some(Duration::from_millis(crate::magic_paint::SHIMMER_FRAME_MS))
        );
        let mut t = Terminal::new(TestBackend::new(60, 10)).unwrap();
        t.draw(|f| app.render(f)).unwrap();
        let buf = t.backend().buffer().clone();
        let (x, y) = (0..10u16)
            .find_map(|y| {
                let row: String = (0..60u16).map(|x| buf[(x, y)].symbol()).collect();
                row.find("ultrathink")
                    .map(|b| (u16::try_from(row[..b].chars().count()).unwrap(), y))
            })
            .expect("keyword on screen");
        assert!(matches!(buf[(x, y)].fg, ratatui::style::Color::Rgb(..)));
        assert!(matches!(buf[(x + 9, y)].fg, ratatui::style::Color::Rgb(..)));
        assert!(!matches!(
            buf[(x - 2, y)].fg,
            ratatui::style::Color::Rgb(..)
        ));
        // Disabled by settings: no paint, no timer.
        app.magic_off = 1;
        let theme = app.theme.clone();
        app.set_theme(theme);
        assert_eq!(app.animation_interval(), None);
    }

    /// Hidden notices (`display: false`) never render, live or rebuilt.
    #[test]
    fn hidden_custom_notices_never_render() {
        let notice = json!({
            "role": "custom", "customType": "ultracode-notice",
            "content": [{"type": "text", "text": "SECRET-NOTICE"}],
            "display": false, "attribution": "user", "timestamp": 1
        });
        let mut app = app();
        app.apply_event(&AppEvent::MessageStart(
            json!({"type": "message_start", "message": notice}),
        ));
        app.apply_event(&AppEvent::MessageEnd(
            json!({"type": "message_end", "message": notice}),
        ));
        let live: Vec<String> = app
            .take_scrollback_lines(80, true)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(
            live.iter()
                .all(|r| !r.contains("SECRET") && !r.contains("ultracode-notice")),
            "{live:#?}"
        );
        assert!(!screen(&mut app, 80, 12).contains("SECRET"));
        app.apply_event(&AppEvent::SessionLoaded {
            messages: vec![
                json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
                notice,
            ],
            name: None,
        });
        let rebuilt: Vec<String> = app
            .take_scrollback_lines(80, true)
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(rebuilt.iter().any(|r| r.contains("hi")));
        assert!(
            rebuilt
                .iter()
                .all(|r| !r.contains("SECRET") && !r.contains("ultracode-notice")),
            "{rebuilt:#?}"
        );
    }
}

#[cfg(test)]
mod paste_tests {
    use super::*;
    use crate::clipboard::ClipboardOutcome;
    use crossterm::event::{
        Event, KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods,
    };
    use std::path::Path;
    use std::time::Instant;

    /// An app pinned to built-in default settings: independent of the
    /// process-global `HOME`, which the temp-HOME tests mutate.
    fn app() -> App {
        let mut app = App::new("zai/glm-5.3-flash".into(), Path::new("/repo"));
        app.set_settings_global(None);
        // Capture deliveries instead of ringing the real terminal / desktop.
        app.set_notify_hook(Arc::new(|_| {}));
        app
    }

    fn port() -> super::super::port::FakePort {
        crate::port::FakePort::default()
    }

    fn key(code: CtKeyCode) -> KeyEvent {
        CtKeyEvent::new(code, CtMods::NONE)
    }

    fn log(port: &crate::port::FakePort) -> Vec<String> {
        port.log.lock().clone()
    }

    fn shift_ctrl(code: char) -> KeyEvent {
        CtKeyEvent::new(CtKeyCode::Char(code), CtMods::CONTROL | CtMods::SHIFT)
    }

    #[test]
    fn paste_chords_arm_clipboard_read() {
        let mut app = app();
        let port = port();
        let ctrl_v = CtKeyEvent::new(CtKeyCode::Char('v'), CtMods::CONTROL);
        assert!(app.handle_key(ctrl_v, &port, Instant::now()).is_some());
        assert!(app.clipboard_rx.is_some(), "ctrl+v arms the clipboard read");
        app.clipboard_rx = None;
        let csi_u = CtKeyEvent::new(CtKeyCode::Char('v'), CtMods::CONTROL | CtMods::SHIFT);
        assert!(app.handle_key(csi_u, &port, Instant::now()).is_some());
        assert!(
            app.clipboard_rx.is_some(),
            "ctrl+shift+v arms the clipboard read"
        );
    }

    #[test]
    fn bracketed_paste_small_inserts_verbatim() {
        let mut app = app();
        let port = port();
        assert!(app.handle_terminal_event(&Event::Paste("hello\nworld".into()), &port));
        assert_eq!(app.editor.text(), "hello\nworld");
    }

    #[test]
    fn bracketed_paste_large_collapses_to_marker() {
        let mut app = app();
        let port = port();
        let payload = "x".repeat(1200);
        assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
        // TS input-controller stages a marker-sized paste as a text chip
        // (`insertTextAttachment`); the raw-text marker stays the editor
        // default for hosts without the large-paste gate.
        assert_eq!(app.editor.text(), "[Text #1, paste.md]");
    }

    #[test]
    fn bracketed_paste_keeps_payload_verbatim_including_newlines() {
        let mut app = app();
        let port = port();
        // A trailing newline inside the markers is paste content; a real
        // Enter sharing the burst arrives as its own Event::Key (TS
        // `leadingEnter` splits bytes after `\x1b[201~`).
        assert!(app.handle_terminal_event(&Event::Paste("hi there\n".into()), &port));
        assert_eq!(app.editor.text(), "hi there\n");
        assert!(
            !log(&port).iter().any(|l| l.contains("hi there")),
            "no auto-submit"
        );
    }

    #[test]
    fn paste_text_raw_inserts_verbatim_without_collapse() {
        let mut app = app();
        let port = port();
        assert!(app.handle_terminal_event(&Event::Key(shift_ctrl('v')), &port));
        // The read is in flight; nothing inserted yet.
        assert_eq!(app.editor.text(), "");
        app.apply_clipboard_outcome(ClipboardOutcome::TextRaw("a\nb\nc".into()));
        assert_eq!(app.editor.text(), "a\nb\nc");
    }

    #[test]
    fn clipboard_smart_text_routes_through_paste_pipeline() {
        let mut app = app();
        app.apply_clipboard_outcome(ClipboardOutcome::Text("x".repeat(1200)));
        assert_eq!(app.editor.text(), "[Text #1, paste.md]");
        // Discard the pasted chip so the next outcome starts from an empty editor.
        let _ = app.editor.take();
        app.apply_clipboard_outcome(ClipboardOutcome::Empty);
        assert_eq!(app.editor.text(), "", "empty clipboard inserts nothing");
    }

    #[test]
    fn clipboard_image_attaches_chip() {
        let mut app = app();
        app.apply_clipboard_outcome(ClipboardOutcome::Image(b"png".to_vec()));
        assert!(
            app.editor.text().starts_with("[Image #1]"),
            "{}",
            app.editor.text()
        );
        assert_eq!(app.editor.attachments().len(), 1);
    }

    #[test]
    fn large_paste_menu_opens_at_threshold_and_inline_collapses() {
        let mut app = app();
        let port = port();
        let mut cfg_home = std::env::temp_dir();
        cfg_home.push(format!("omp-paste-menu-test-{}", std::process::id()));
        std::fs::create_dir_all(&cfg_home).unwrap();
        // Default settings file location derives from cwd/env; keep the
        // default threshold (100 lines) and use a 120-line paste.
        let payload = "line\n".repeat(119) + "line";
        assert!(app.handle_terminal_event(&Event::Paste(payload.clone()), &port));
        assert_eq!(
            app.overlays.open(),
            Some(OpenOverlay::LargePaste),
            "menu gate"
        );
        assert_eq!(
            app.editor.text(),
            "",
            "insert deferred while the menu is open"
        );
        // Choose "Paste inline" (third row) → collapse to the inline marker.
        app.overlays.handle_key(key(CtKeyCode::Down));
        app.overlays.handle_key(key(CtKeyCode::Down));
        app.overlays.handle_key(key(CtKeyCode::Enter));
        app.poll_deadlines(Instant::now(), &port);
        assert_eq!(app.overlays.open(), None);
        assert!(
            app.editor.text().contains("[Paste #1,") || app.editor.text().contains("[Text #1,"),
            "{}",
            app.editor.text()
        );
        let _ = std::fs::remove_dir_all(cfg_home);
    }

    #[test]
    fn large_paste_menu_cancel_stages_text_chip() {
        let mut app = app();
        let port = port();
        let payload = "line\n".repeat(119) + "line";
        assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
        assert_eq!(app.overlays.open(), Some(OpenOverlay::LargePaste));
        // Esc bubbles out of the panel; the host then applies the fallback.
        assert!(app.handle_terminal_event(&Event::Key(key(CtKeyCode::Esc)), &port));
        app.poll_deadlines(Instant::now(), &port);
        assert_eq!(app.overlays.open(), None);
        assert!(
            app.editor.text().starts_with("[Text #1, paste.md]"),
            "{}",
            app.editor.text()
        );
    }

    #[test]
    fn large_paste_below_threshold_stages_chip_without_menu() {
        let mut app = app();
        let port = port();
        // Marker-sized (>10 lines) but under the 100-line menu gate.
        let payload = "line\n".repeat(20);
        assert!(app.handle_terminal_event(&Event::Paste(payload), &port));
        assert_eq!(app.overlays.open(), None, "below gate: no menu");
        assert!(
            app.editor.text().starts_with("[Text #1, paste.md]"),
            "{}",
            app.editor.text()
        );
    }
}

#[cfg(test)]
mod paste_key_tests {
    use super::*;
    use crossterm::event::{KeyCode as CtKeyCode, KeyEvent as CtKeyEvent, KeyModifiers as CtMods};
    use std::path::Path;

    #[test]
    fn ctrl_shift_v_csi_u_matches_paste_text_raw() {
        let app = App::new("m".into(), Path::new("/repo"));
        let key = CtKeyEvent::new(CtKeyCode::Char('v'), CtMods::CONTROL | CtMods::SHIFT);
        assert_eq!(
            crate::keybindings::key_id_from_event(&key).as_deref(),
            Some("ctrl+shift+v"),
            "CSI-u ctrl+shift+v canonicalizes to the binding id"
        );
        assert!(app.keys.matches_event(&key, "app.clipboard.pasteTextRaw"));
    }
}
