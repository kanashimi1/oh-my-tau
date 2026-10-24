//! Terminal title run states — port of
//! `pi-coding-agent/src/utils/title-generator.ts` §8 (composed run-state
//! titles) on top of omp-cli's static `π: <label>` title.
//!
//! Shape (spec §8.1): `π <separator> <label>` where the separator is `>`
//! idle, `!` attention and a spinner frame while working. Under WSL (stdout
//! crosses ConPTY at wslhost) and on Windows after a native-title failure the
//! working separator degrades to static `:`; this port targets WSL panes
//! (herdr) and therefore defaults to the static separator there.
//!
//! Emission: OSC 0 (`\x1b]0;<title>\x07`) via crossterm [`SetTitle`], deduped
//! against the last written title. A teardown latch (`dispose`) stops the
//! spinner thread and forbids further emission until [`TerminalTitle::init`]
//! is called again.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Brand used in titles: plain π, never a PUA nerd glyph (OS tab UIs lack
/// coverage).
pub const DEFAULT_TERMINAL_TITLE: &str = "π";

/// Brand of the omp-rs clean style (`τ > label`).
pub const CLEAN_TERMINAL_TITLE: &str = "τ";

/// Tick rate for the working-state spinner (spec §8.2).
const SPINNER_ADVANCE: Duration = Duration::from_millis(80);

/// Run state shown in the title separator (TS `setTerminalTitleState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TitleState {
    /// User's turn — `π > label`.
    #[default]
    Idle,
    /// Agent running — `π <frame> label`.
    Working,
    /// Agent blocked on the user — `π ! label`.
    Attention,
}

/// Spinner frame sets for the working state (spec §8.2, `tui.titleSpinner`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TitleSpinner {
    /// `⠋⠙⠹…` (unicode activity frames).
    #[default]
    Braille,
    /// `○◔◑◕●◕◑◔`.
    Pulse,
    /// `⠁⠂⠄⠠⠐⠈`.
    Dots,
    /// `- \ | /`.
    Line,
}

impl TitleSpinner {
    /// Frame list for this style.
    #[must_use]
    pub fn frames(self) -> &'static [&'static str] {
        match self {
            Self::Braille => &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            Self::Pulse => &["○", "◔", "◑", "◕", "●", "◕", "◑", "◔"],
            Self::Dots => &["⠁", "⠂", "⠄", "⠠", "⠐", "⠈"],
            Self::Line => &["-", "\\", "|", "/"],
        }
    }

    /// Frame at `tick` (80 ms clock, index mod len; style switch resets the
    /// counter — callers pass the tick from `TerminalTitle` which resets on
    /// `set_spinner`).
    #[must_use]
    pub fn frame_at(self, tick: usize) -> &'static str {
        let frames = self.frames();
        frames[tick % frames.len()]
    }

    /// Parse a `tui.titleSpinner` setting value; unknown → [`Self::Braille`].
    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value {
            "pulse" => Self::Pulse,
            "dots" => Self::Dots,
            "line" => Self::Line,
            _ => Self::Braille,
        }
    }
}

/// Strip control characters the way TS `sanitize` does
/// (`[\u{0000}-\u{001f}\u{007f}-\u{009f}]`).
#[must_use]
pub fn sanitize_title(raw: &str) -> String {
    raw.chars()
        .filter(|c| !matches!(c, '\u{0}'..='\u{1f}' | '\u{7f}' | '\u{80}'..='\u{9f}'))
        .collect()
}

/// Whether stdout crosses a WSL ConPTY boundary (static `:` separator).
#[must_use]
pub fn is_wsl() -> bool {
    std::env::var_os("WSL_DISTRO_NAME").is_some()
        || std::env::var_os("WSL_INTEROP").is_some()
        || std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|v| v.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false)
}

/// Compose a run-state title (TS `buildTerminalTitleWithState`, :800).
///
/// `separator` is the pre-rendered state separator (`>`, `!`, a spinner
/// frame or the static `:`); `label` the sanitized session name or cwd
/// basename. No label → `π <separator>` (trailing space kept, as TS).
#[must_use]
pub fn build_terminal_title_with_state(separator: &str, label: Option<&str>) -> String {
    build_branded_title(DEFAULT_TERMINAL_TITLE, separator, label)
}

/// [`build_terminal_title_with_state`] with an explicit brand (`π` classic,
/// `τ` clean).
#[must_use]
pub fn build_branded_title(brand: &str, separator: &str, label: Option<&str>) -> String {
    match label {
        Some(l) if !l.is_empty() => format!("{brand} {separator} {l}"),
        _ => format!("{brand} {separator}"),
    }
}

/// Shared teardown/emission state between the owner and its spinner thread.
#[derive(Debug, Default)]
struct Shared {
    /// Teardown latch: once set, nothing may emit again until `init`.
    disposed: AtomicBool,
    /// Last title written (dedup, TS `lastTerminalTitle`).
    last: parking_lot::Mutex<Option<String>>,
    /// Current working-frame tick (bumped by the spinner thread).
    tick: parking_lot::Mutex<usize>,
    /// Title brand (`π` / `τ`); read by the spinner thread every frame.
    brand: parking_lot::Mutex<Option<&'static str>>,
    /// Last state + sanitized label, re-emitted when the brand changes.
    current: parking_lot::Mutex<Option<(TitleState, Option<String>)>>,
}

/// A running spinner thread plus its own stop flag. `disposed` alone cannot
/// end it: a Working -> Idle transition stops the spinner without disposing,
/// and joining a thread nobody told to exit hangs the UI thread forever.
#[derive(Debug)]
struct Spinner {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl Spinner {
    fn stop(self) {
        self.stop.store(true, Ordering::SeqCst);
        self.handle.thread().unpark();
        let _ = self.handle.join();
    }
}

impl Shared {
    fn brand(&self) -> &'static str {
        self.brand.lock().unwrap_or(DEFAULT_TERMINAL_TITLE)
    }
}

/// Owner of the terminal title run-state. Cloned handles share one spinner
/// thread; state changes recompose and emit immediately (dedup makes idle
/// transitions cheap).
#[derive(Debug, Clone)]
pub struct TerminalTitle {
    shared: Arc<Shared>,
    spinner: Arc<parking_lot::Mutex<Option<Spinner>>>,
    /// Spinner style in use.
    style: Arc<parking_lot::Mutex<TitleSpinner>>,
    /// `true` when the working separator must be static (`:`) — WSL or a
    /// failed native title (TS `nativeTitleFailed` latch; this port has no
    /// native Win32 path, so WSL alone decides).
    static_working: bool,
}

impl Default for TerminalTitle {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalTitle {
    /// Create a title owner. Emits nothing until `init` (and never on a
    /// non-TTY stdout).
    #[must_use]
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Shared::default()),
            spinner: Arc::new(parking_lot::Mutex::new(None)),
            style: Arc::new(parking_lot::Mutex::new(TitleSpinner::Braille)),
            static_working: is_wsl(),
        }
    }

    /// Release the teardown latch and write the current title (TS
    /// `initTerminalTitleState`).
    pub fn init(&self, state: TitleState, label: Option<&str>) {
        self.shared.disposed.store(false, Ordering::SeqCst);
        self.set_state(state, label);
    }

    /// Latch teardown: stop the spinner thread; no further emission until
    /// [`Self::init`] (TS `disposeTerminalTitleState`).
    pub fn dispose(&self) {
        self.shared.disposed.store(true, Ordering::SeqCst);
        self.stop_spinner();
    }

    /// Set the spinner style (`tui.titleSpinner`); resets the frame counter.
    pub fn set_spinner(&self, style: TitleSpinner) {
        *self.style.lock() = style;
        *self.shared.tick.lock() = 0;
    }

    /// Recompose and emit for a new run state (TS `setTerminalTitleState`).
    /// Starts/stops the 80 ms spinner thread as needed.
    pub fn set_state(&self, state: TitleState, label: Option<&str>) {
        let sanitized = label.map(sanitize_title).filter(|l| !l.trim().is_empty());
        *self.shared.current.lock() = Some((state, sanitized.clone()));
        let spinner_wanted = state == TitleState::Working && !self.static_working;
        if spinner_wanted {
            self.start_spinner(sanitized.clone());
        } else {
            self.stop_spinner();
        }
        let separator = self.separator_for(state);
        self.emit(&build_branded_title(
            self.shared.brand(),
            separator,
            sanitized.as_deref(),
        ));
    }

    /// Switch the brand (`π` classic, `τ` clean) and re-emit the current
    /// state with it.
    pub fn set_brand(&self, brand: &'static str) {
        if self.shared.brand() == brand {
            return;
        }
        *self.shared.brand.lock() = Some(brand);
        let current = self.shared.current.lock().clone();
        if let Some((state, label)) = current {
            self.set_state(state, label.as_deref());
        }
    }

    /// Static title (`π: <name|basename>`); used for authoritative session
    /// renames and teardown restore (TS `formatSessionTerminalTitle` shape is
    /// omp-cli's; this emits the same OSC).
    pub fn set_static(&self, label: Option<&str>) {
        self.stop_spinner();
        let brand = self.shared.brand();
        let title = match label.map(sanitize_title).filter(|l| !l.is_empty()) {
            Some(l) => format!("{brand}: {l}"),
            None => brand.to_string(),
        };
        self.emit(&title);
    }

    /// The last composed title (deduped), or `None` before the first emit.
    /// On a non-TTY the OSC write is skipped but the composition is still
    /// recorded, so tests and headless hosts can observe title state.
    #[must_use]
    pub fn current_title(&self) -> Option<String> {
        self.shared.last.lock().clone()
    }

    fn separator_for(&self, state: TitleState) -> &'static str {
        match state {
            TitleState::Idle => ">",
            TitleState::Attention => "!",
            // Static under WSL / failed native title (spec §8.1).
            TitleState::Working if self.static_working => ":",
            TitleState::Working => {
                let tick = *self.shared.tick.lock();
                self.style.lock().frame_at(tick)
            }
        }
    }

    fn start_spinner(&self, label: Option<String>) {
        let mut handle = self.spinner.lock();
        if handle.is_some() {
            return;
        }
        let shared = Arc::clone(&self.shared);
        let style = Arc::clone(&self.style);
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        *shared.tick.lock() = 0;
        let halted =
            move || thread_stop.load(Ordering::SeqCst) || shared.disposed.load(Ordering::SeqCst);
        let shared = Arc::clone(&self.shared);
        let thread = thread::spawn(move || loop {
            if halted() {
                return;
            }
            // Parked, not slept: `Spinner::stop` unparks so the join is prompt.
            thread::park_timeout(SPINNER_ADVANCE);
            if halted() {
                return;
            }
            let mut tick = shared.tick.lock();
            *tick += 1;
            let frame = style.lock().frame_at(*tick);
            drop(tick);
            let title = build_branded_title(shared.brand(), frame, label.as_deref());
            write_title(&shared, &title);
        });
        *handle = Some(Spinner {
            stop,
            handle: thread,
        });
    }

    fn stop_spinner(&self) {
        // Take first so the mutex is not held across the join.
        let spinner = self.spinner.lock().take();
        if let Some(spinner) = spinner {
            spinner.stop();
        }
    }

    fn emit(&self, title: &str) {
        if self.shared.disposed.load(Ordering::SeqCst) {
            return;
        }
        write_title(&self.shared, title);
    }
}

impl Drop for TerminalTitle {
    fn drop(&mut self) {
        self.dispose();
    }
}

/// OSC 0 write with dedup; silent no-op when stdout is not a TTY.
fn write_title(shared: &Shared, title: &str) {
    {
        let mut last = shared.last.lock();
        if last.as_deref() == Some(title) {
            return;
        }
        // Record the composed title even when the write is skipped (non-TTY:
        // tests and headless consumers observe state through `current`).
        *last = Some(title.to_string());
    }
    if !std::io::stdout().is_terminal() {
        return;
    }
    // crossterm queues the sequence; flush is left to the terminal loop, and
    // the controller draws at ≥ the spinner tick anyway. Emit via SetTitle so
    // the sequence stays well-formed alongside ratatui output.
    let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::SetTitle(title));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_shapes() {
        assert_eq!(
            build_terminal_title_with_state(">", Some("Продолжить")),
            "π > Продолжить"
        );
        assert_eq!(build_terminal_title_with_state("!", None), "π !");
        assert_eq!(
            build_terminal_title_with_state("⠋", Some("omt")),
            "π ⠋ omt"
        );
    }

    #[test]
    fn clean_titles_use_tau() {
        assert_eq!(
            build_branded_title(CLEAN_TERMINAL_TITLE, ">", Some("omt")),
            "τ > omt"
        );
        assert_eq!(
            build_branded_title(CLEAN_TERMINAL_TITLE, "⠋", Some("x")),
            "τ ⠋ x"
        );
        assert_eq!(build_branded_title(CLEAN_TERMINAL_TITLE, "!", None), "τ !");
        let title = TerminalTitle::new();
        title.init(TitleState::Idle, Some("proj"));
        assert_eq!(title.current_title().as_deref(), Some("π > proj"));
        // Switching the brand re-emits the current state.
        title.set_brand(CLEAN_TERMINAL_TITLE);
        assert_eq!(title.current_title().as_deref(), Some("τ > proj"));
        title.set_state(TitleState::Attention, Some("proj"));
        assert_eq!(title.current_title().as_deref(), Some("τ ! proj"));
        title.set_static(Some("proj"));
        assert_eq!(title.current_title().as_deref(), Some("τ: proj"));
        title.dispose();
    }

    #[test]
    fn sanitize_strips_control_chars() {
        assert_eq!(sanitize_title("bad\u{7}title"), "badtitle");
        assert_eq!(sanitize_title("ok"), "ok");
    }

    #[test]
    fn spinner_styles_parse_and_frame() {
        assert_eq!(TitleSpinner::parse("pulse"), TitleSpinner::Pulse);
        assert_eq!(TitleSpinner::parse("nonsense"), TitleSpinner::Braille);
        assert_eq!(TitleSpinner::Line.frame_at(0), "-");
        assert_eq!(TitleSpinner::Line.frame_at(4), "-");
    }

    /// Regression: Working -> Idle joined a spinner thread that only exited on
    /// `dispose`, freezing the TUI after the first reply (non-WSL hosts).
    #[test]
    fn spinner_stops_on_state_change() {
        let mut title = TerminalTitle::new();
        title.static_working = false;
        let (tx, rx) = std::sync::mpsc::channel();
        let t = title.clone();
        thread::spawn(move || {
            t.init(TitleState::Working, Some("p"));
            thread::sleep(Duration::from_millis(200));
            t.set_state(TitleState::Idle, Some("p"));
            t.set_state(TitleState::Working, Some("p"));
            t.set_state(TitleState::Idle, Some("p"));
            let _ = tx.send(());
        });
        rx.recv_timeout(Duration::from_secs(5))
            .expect("set_state(Idle) hung joining the spinner");
        assert_eq!(title.current_title().as_deref(), Some("π > p"));
    }

    #[test]
    fn wsl_detect_does_not_panic() {
        let _ = is_wsl();
    }
}
