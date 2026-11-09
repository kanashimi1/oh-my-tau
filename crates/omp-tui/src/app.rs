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

pub(crate) use std::collections::VecDeque;
pub(crate) use std::fmt;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Arc;
pub(crate) use std::time::{Duration, Instant};

pub(crate) use crossterm::event::Event;
pub(crate) use ratatui::layout::{Constraint, Direction, Layout, Rect};
pub(crate) use ratatui::style::Style;
pub(crate) use ratatui::text::{Line, Span};
pub(crate) use ratatui::Frame;
pub(crate) use serde_json::Value;

pub(crate) use crate::action::{Action, QueueKind, SubmitBehavior};
pub(crate) use crate::autocomplete::Autocomplete;
pub(crate) use crate::chat::Messages;
pub(crate) use crate::chrome::notify::{send_notification, NotifyProtocol, TerminalNotification};
pub(crate) use crate::chrome::title::{TerminalTitle, TitleState};
pub(crate) use crate::chrome::welcome;
pub(crate) use crate::component::Component;
pub(crate) use crate::editor::{Editor, EditorOutcome};
pub(crate) use crate::events::{parse_frame, parse_frame_owned, AppEvent};
pub(crate) use crate::extension_ui::ExtensionUi;
pub(crate) use crate::keybindings::KeybindingsManager;
pub(crate) use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
pub(crate) use crate::overlays::models::{parse_model_string, ModelEntry};
pub(crate) use crate::overlays::sessions::SessionSource;
pub(crate) use crate::overlays::tree::TreeMode;
pub(crate) use crate::overlays::{OpenOverlay, Overlays};
pub(crate) use crate::port::RuntimePort;
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

impl App {
    /// Record the terminal appearance resolved at startup (OSC 11 /
    /// COLORFGBG); `/settings` theme edits hot-swap the matching variant.
    pub fn set_appearance(&mut self, appearance: crate::theme::Appearance) {
        self.appearance = appearance;
    }

    /// Replace the app-wide theme (`ThemeLoader` startup + hot-swap). The
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
                    // `Path`'s `Debug` (quoted) is the established message form.
                    #[allow(clippy::unnecessary_debug_formatting)]
                    {
                        format!("keybindings: {conflicts} conflicting override(s) in {path:?}; first binding wins on the input path")
                    }
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
type NotifyHook = Arc<dyn Fn(&TerminalNotification) + Send + Sync>;

#[derive(Clone, Default)]
pub struct NotifySink {
    hook: Option<NotifyHook>,
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
    /// one (only the latest within the window is kept) but keep the pending
    /// one's arrival time, so a continuous stream still shows a snapshot every
    /// window (throttle) instead of nothing until the stream pauses; any other
    /// frame flushes the held-back snapshot first so ordering is preserved.
    pub fn push_frame(&mut self, frame: &Value, now: Instant) {
        let event = parse_frame(frame);
        if let AppEvent::MessageUpdate(_) = event {
            let at = self.pending_update.as_ref().map_or(now, |(at, _)| *at);
            self.pending_update = Some((at, frame.clone()));
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
// The mode flags mirror the TS client's flat session state; grouping them
// would only add indirection over the same five booleans.
#[allow(clippy::struct_excessive_bools)]
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
    turn_usage: turn_usage::TurnUsage,
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
    /// means the user's own config resolved through `HOME`. The inner
    /// `Option` distinguishes "pinned to the global file" from "unpinned".
    #[allow(clippy::option_option)]
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
            turn_usage: turn_usage::TurnUsage::default(),
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
}

mod events;
mod input;
mod render;
mod slash;
mod turn_usage;

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
mod paste_key_tests;
#[cfg(test)]
mod paste_tests;
#[cfg(test)]
mod tests;
