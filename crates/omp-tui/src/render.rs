//! Shell render: inline main-screen rendering with scrollback commit
//! (port of pi-tui's normal-screen provider paint; see
//! `runs/omp-rust/tui-spec/core.md` §1–§5).
//!
//! The terminal's normal buffer is the canvas: finished transcript rows are
//! *committed* — printed above the live region via
//! [`Terminal::insert_before`], which pushes them into the terminal's native
//! scrollback and is never redrawn. Only the live region (streaming text,
//! pending tools, editor, status line) is repainted each frame, in an inline
//! viewport anchored to the bottom of the screen. Full-screen overlays
//! temporarily borrow the alternate screen (see [`Shell::with_fullscreen`]).
//!
//! Every paint is bracketed with hide-cursor + DEC 2026 synchronized output
//! (when supported) + autowrap-off … autowrap-on, matching pi-tui's
//! `PAINT_BEGIN`/`PAINT_END` (tui.ts:62-105). The hardware cursor is hidden
//! and parked on the live region's top row — the anchor invariant resize
//! recovery relies on.

use std::io::{self, Write};

use ratatui::backend::{Backend, CrosstermBackend, WindowSize};
use ratatui::buffer::Cell;
use ratatui::layout::{Position, Rect, Size};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{Terminal, TerminalOptions, Viewport};

// Paint sequences (core.md §1).
pub const SEGMENT_RESET: &str = "\x1b[0m";
pub const HIDE_CURSOR: &str = "\x1b[?25l";
pub const SYNC_OUTPUT_BEGIN: &str = "\x1b[?2026h";
pub const SYNC_OUTPUT_END: &str = "\x1b[?2026l";
pub const DISABLE_AUTOWRAP: &str = "\x1b[?7l";
pub const ENABLE_AUTOWRAP: &str = "\x1b[?7h";
pub const BRACKETED_PASTE_ON: &str = "\x1b[?2004h";
pub const BRACKETED_PASTE_OFF: &str = "\x1b[?2004l";
/// Force normal cursor keys + numeric keypad (terminfo `rmkx`; a parent app
/// may have left the TTY in application mode; core.md §6.1 step 2).
pub const DECCKM_NORMAL: &str = "\x1b[?1l\x1b>";
pub const ENTER_ALT: &str = "\x1b[?1049h";
pub const LEAVE_ALT: &str = "\x1b[?1049l";
pub const MOUSE_TRACKING_ON: &str = "\x1b[?1000h\x1b[?1003h\x1b[?1006h";
pub const MOUSE_TRACKING_OFF: &str = "\x1b[?1006l\x1b[?1003l\x1b[?1000l";
/// Destructive reset: ED2 **before** ED3 (tmux implements ED2 by scrolling the
/// live screen into pane history; ED3-first would duplicate it, core.md §2.5).
pub const CLEAR_SCREEN_AND_SCROLLBACK: &str = "\x1b[H\x1b[2J\x1b[3J";
/// OSC 133 shell-integration prompt-zone marks (TS `user-message.ts`
/// `OSC133_*`): a committed batch is wrapped zone-start → zone-end so
/// multiplexers group the transcript block as one prompt zone. The zone is
/// always closed inside the same paint (`133;B` latches a sticky `.input`
/// cursor semantic in Ghostty unless `133;C`/`133;D` follow immediately).
pub const OSC133_ZONE_START: &str = "\x1b]133;A\x07";
/// Zone end (`133;B`).
pub const OSC133_ZONE_END: &str = "\x1b]133;B\x07";
/// Command start (`133;C`) — clears the sticky input state.
pub const OSC133_COMMAND_START: &str = "\x1b]133;C\x07";
/// Command done (`133;D;0`).
pub const OSC133_COMMAND_DONE: &str = "\x1b]133;D;0\x07";
/// The zone-close trailer: end + command start + command done.
pub const OSC133_ZONE_CLOSE: &str = "\x1b]133;B\x07\x1b]133;C\x07\x1b]133;D;0\x07";
/// xterm `modifyOtherKeys` fallback when the kitty protocol is unavailable
/// (core.md §6.1 step 4). Env-gated (`OMP_TUI_MODIFY_OTHER_KEYS=1`).
pub const MODIFY_OTHER_KEYS_ON: &str = "\x1b[>4;2m";
pub const MODIFY_OTHER_KEYS_OFF: &str = "\x1b[>4;1m";

const PAINT_BEGIN: &str = concat!("\x1b[?25l", "\x1b[?2026h", "\x1b[?7l");
const PAINT_BEGIN_NO_SYNC: &str = concat!("\x1b[?25l", "\x1b[?7l");
const PAINT_END: &str = concat!("\x1b[?7h", "\x1b[?2026l");
const PAINT_END_NO_SYNC: &str = "\x1b[?7h";

/// Detected terminal multiplexer (port of `terminal-multiplexer.ts`).
/// Inside a mux the mux owns the grid: resize uses the "rebuild" scrollback
/// strategy instead of in-place reflow (core.md §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Multiplexer {
    Herdr,
    Tmux,
    Screen,
    Zellij,
    Cmux,
    None,
}

impl Multiplexer {
    /// Classify from the process environment.
    #[must_use]
    pub fn detect() -> Self {
        let env = std::env::var_os("HERDR_ENV").is_some_and(|v| v == "1")
            || std::env::var_os("HERDR_PANE_ID").is_some()
            || std::env::var_os("HERDR_TAB_ID").is_some()
            || std::env::var_os("HERDR_WORKSPACE_ID").is_some();
        if env {
            return Self::Herdr;
        }
        if std::env::var_os("TMUX").is_some()
            || std::env::var("TERM").is_ok_and(|t| t.starts_with("tmux"))
        {
            return Self::Tmux;
        }
        if std::env::var_os("STY").is_some()
            || std::env::var("TERM").is_ok_and(|t| t.starts_with("screen"))
        {
            return Self::Screen;
        }
        if std::env::var_os("ZELLIJ").is_some() {
            return Self::Zellij;
        }
        if std::env::var_os("CMUX_WORKSPACE_ID").is_some()
            || std::env::var_os("CMUX_SURFACE_ID").is_some()
            || std::env::var_os("CMUX_REMOTE_TRANSPORT").is_some()
        {
            return Self::Cmux;
        }
        Self::None
    }

    /// Whether resize must replay/rebuild scrollback instead of relying on
    /// the host to reflow the grid in place.
    #[must_use]
    pub fn owns_grid(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Resolve DEC 2026 synchronized-output support (core.md §1).
///
/// Precedence: env override (`OMP_TUI_SYNC_OUTPUT=0|1`) > static terminal
/// identity. Inside Herdr, sync stays **on** even where a DECRQM probe would
/// report unrecognized (Ghostty honors 2026 with status 0). Runtime DECRQM
/// adjustment is a P1 item; this is the P0 static gate.
#[must_use]
pub fn sync_output_enabled() -> bool {
    if let Some(v) = std::env::var_os("OMP_TUI_SYNC_OUTPUT") {
        return v != "0" && v != "false";
    }
    let term = std::env::var("TERM").unwrap_or_default();
    let program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    if Multiplexer::detect().owns_grid() {
        // tmux ≥ 3.3 implements synchronized updates itself.
        return true;
    }
    term.contains("kitty")
        || term.contains("ghostty")
        || term.contains("wezterm")
        || term.contains("foot")
        || term.contains("rio")
        || term.contains("contour")
        || program == "ghostty"
        || program == "WezTerm"
        || program == "kitty"
        || program == "iTerm.app"
        || program == "warpterminal"
}

/// Whether the environment plausibly speaks the kitty keyboard protocol, and
/// a probe is worth its latency on startup (core.md §6.1 step 4).
fn kitty_probe_worthwhile() -> bool {
    if std::env::var_os("OMP_TUI_KITTY_KEYS").is_some_and(|v| v == "0") {
        return false;
    }
    let term = std::env::var("TERM").unwrap_or_default();
    let program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    std::env::var_os("KITTY_WINDOW_ID").is_some()
        || term.contains("kitty")
        || term.contains("wezterm")
        || term.contains("ghostty")
        || term.contains("rio")
        || term.contains("contour")
        || program == "ghostty"
        || program == "WezTerm"
        || program == "kitty"
}

/// Vertical layout of the live region: keep the bottom `height` rows
/// (pi-tui paints the bottom rows of the composed tree) and top-pad with
/// blanks so the frame fills the viewport exactly.
#[must_use]
pub fn plan_live_frame<'a>(lines: &[Line<'a>], height: usize) -> Vec<Line<'a>> {
    let start = lines.len().saturating_sub(height);
    let visible = &lines[start..];
    let pad = height - visible.len();
    let mut out = Vec::with_capacity(height);
    out.extend(std::iter::repeat_with(Line::default).take(pad));
    out.extend_from_slice(visible);
    out
}

/// Kitty-keyboard state (push/pop levels, core.md §6.1 step 4 / §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KittyKeys {
    /// No protocol: either unsupported or not probed.
    Inactive,
    /// One `PushKeyboardEnhancementFlags` level is on the stack; pop exactly
    /// once at shutdown.
    Pushed,
    /// `modifyOtherKeys` fallback engaged (kitty unavailable).
    ModifyOtherKeys,
}

/// What a settled resize requires of the app (core.md §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeAction {
    /// The live region was repainted at the new size.
    Repaint,
    /// The mux owns the grid: scrollback was erased and must be re-committed
    /// at the new width ("rebuild" mode; caller re-commits its transcript).
    RebuildScrollback,
}

/// Sink for escape sequences written outside ratatui's diff engine.
pub trait RawOut {
    /// Write one raw sequence; implementations flush.
    ///
    /// # Errors
    ///
    /// Fails if the underlying write fails.
    fn raw(&mut self, s: &str) -> io::Result<()>;
}

/// Render a crossterm command to its ANSI string form.
fn cmd_ansi(cmd: &impl crossterm::Command) -> String {
    let mut s = String::new();
    let _ = cmd.write_ansi(&mut s);
    s
}

impl RawOut for io::Stdout {
    fn raw(&mut self, s: &str) -> io::Result<()> {
        self.write_all(s.as_bytes())?;
        self.flush()
    }
}

/// Crossterm backend with a cursor-probe fallback: creating an inline
/// viewport queries the cursor position (CPR), which dumb PTYs and some
/// multiplexers never answer. Rather than failing startup, assume the
/// cursor sits on the last row — the viewport anchors at the screen bottom,
/// which is where a shell prompt lives anyway.
pub struct ShellBackend {
    inner: CrosstermBackend<io::Stdout>,
}

impl ShellBackend {
    fn new() -> Self {
        Self {
            inner: CrosstermBackend::new(io::stdout()),
        }
    }
}

impl Backend for ShellBackend {
    fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        self.inner.draw(content)
    }
    fn hide_cursor(&mut self) -> io::Result<()> {
        self.inner.hide_cursor()
    }
    fn show_cursor(&mut self) -> io::Result<()> {
        self.inner.show_cursor()
    }
    fn get_cursor_position(&mut self) -> io::Result<Position> {
        if let Ok(pos) = self.inner.get_cursor_position() {
            Ok(pos)
        } else {
            let size = self.inner.size()?;
            Ok(Position {
                x: 0,
                y: size.height.saturating_sub(1),
            })
        }
    }
    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
        self.inner.set_cursor_position(position)
    }
    fn clear(&mut self) -> io::Result<()> {
        self.inner.clear()
    }
    fn clear_region(&mut self, clear_type: ratatui::backend::ClearType) -> io::Result<()> {
        self.inner.clear_region(clear_type)
    }
    fn append_lines(&mut self, n: u16) -> io::Result<()> {
        self.inner.append_lines(n)
    }
    fn size(&self) -> io::Result<Size> {
        self.inner.size()
    }
    fn window_size(&mut self) -> io::Result<WindowSize> {
        self.inner.window_size()
    }
    fn flush(&mut self) -> io::Result<()> {
        ratatui::backend::Backend::flush(&mut self.inner)
    }
}

/// The inline rendering shell. Concrete on a real terminal via
/// [`Shell::enter`]; testable with ratatui's `TestBackend` through
/// [`Shell::new_for_backend`].
pub struct Shell<B: Backend, R: RawOut> {
    term: Terminal<B>,
    raw: R,
    live_height: u16,
    sync: bool,
    kitty: KittyKeys,
    mux: Multiplexer,
    fullscreen: bool,
    /// Last known full-screen area (tracked from completed frames).
    screen: Rect,
}

/// Probe the kitty keyboard protocol: send `CSI ? u`; any reply means the
/// terminal speaks the protocol, then push exactly one flag level. On a
/// confirmed unsupported terminal, optionally engage the `modifyOtherKeys`
/// fallback when `OMP_TUI_MODIFY_OTHER_KEYS=1` (core.md §6.1 step 4).
fn probe_kitty_keys() -> io::Result<KittyKeys> {
    use crossterm::event::{KeyboardEnhancementFlags, PushKeyboardEnhancementFlags};
    if !kitty_probe_worthwhile() {
        return Ok(KittyKeys::Inactive);
    }
    // `supports_keyboard_enhancement` sends the query and consumes the reply
    // from crossterm's shared event source (a late reply therefore never
    // reaches the app's EventStream as a phantom keypress).
    let supported = crossterm::terminal::supports_keyboard_enhancement()?;
    if supported {
        // Default TS level `>5u`: disambiguate escape codes + report
        // alternate keys (no event types — release events would confuse the
        // editor; core.md §6.4/§6.5).
        let push = cmd_ansi(&PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS,
        ));
        io::stdout().write_all(push.as_bytes())?;
        return Ok(KittyKeys::Pushed);
    }
    if std::env::var_os("OMP_TUI_MODIFY_OTHER_KEYS").is_some_and(|v| v == "1") {
        return Ok(KittyKeys::ModifyOtherKeys);
    }
    Ok(KittyKeys::Inactive)
}

impl Shell<ShellBackend, io::Stdout> {
    /// Set up the real terminal: raw mode, bracketed paste, DECCKM
    /// normalization, capability probes, inline viewport.
    ///
    /// # Errors
    ///
    /// Fails if raw mode or the initial size query fails.
    pub fn enter(live_height: u16) -> io::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        let term = Terminal::with_options(
            ShellBackend::new(),
            TerminalOptions {
                viewport: Viewport::Inline(live_height.max(1)),
            },
        )?;
        // Entry writes ride the same stream before any paint: bracketed
        // paste on + force normal cursor keys/keypad (core.md §6.1).
        let mut entry = io::stdout();
        entry.write_all(concat!("\x1b[?2004h", "\x1b[?1l\x1b>").as_bytes())?;

        let kitty = probe_kitty_keys()?;
        if kitty == KittyKeys::ModifyOtherKeys {
            entry.write_all(MODIFY_OTHER_KEYS_ON.as_bytes())?;
        }
        entry.flush()?;
        let screen = Rect::from((Position::ORIGIN, term.size()?));
        Ok(Self {
            term,
            raw: io::stdout(),
            live_height: live_height.max(1),
            sync: sync_output_enabled(),
            kitty,
            mux: Multiplexer::detect(),
            fullscreen: false,
            screen,
        })
    }

    /// Borrow the alternate screen for a full-screen overlay: enter alt,
    /// re-push kitty flags (they are per-screen; Ghostty/kitty/iTerm2 revert
    /// to legacy encoding inside overlays otherwise, core.md §3), enable
    /// mouse tracking, run `draw` (once per overlay frame), then restore the
    /// inline layout.
    ///
    /// # Errors
    ///
    /// Fails if the alt-screen paint or restore writes fail.
    pub fn with_fullscreen(&mut self, draw: impl FnMut(&mut ratatui::Frame<'_>)) -> io::Result<()> {
        use crossterm::event::{
            KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
        };
        if self.fullscreen {
            return Ok(());
        }
        self.fullscreen = true;
        let mut out = String::with_capacity(64);
        out.push_str(ENTER_ALT);
        out.push_str(MOUSE_TRACKING_ON);
        out.push_str(HIDE_CURSOR);
        if self.kitty == KittyKeys::Pushed {
            out.push_str(&cmd_ansi(&PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS,
            )));
        }
        self.raw.raw(&out)?;
        // A throwaway fullscreen terminal paints the modal; the inline
        // terminal's diff state over the main screen stays frozen.
        let mut alt = Terminal::new(ShellBackend::new())?;
        let _ = alt.draw(draw)?;
        ratatui::backend::Backend::flush(alt.backend_mut())?;
        drop(alt);
        let mut restore = String::new();
        if self.kitty == KittyKeys::Pushed {
            restore.push_str(&cmd_ansi(&PopKeyboardEnhancementFlags));
        }
        restore.push_str(MOUSE_TRACKING_OFF);
        restore.push_str(LEAVE_ALT);
        self.raw.raw(&restore)?;
        self.fullscreen = false;
        // Force a full repaint of the inline viewport next frame.
        self.term.resize(self.screen)
    }

    /// Restore the terminal: leave any fullscreen borrow, pop kitty /
    /// modifyOtherKeys, bracketed paste off, DECCKM normal, show cursor,
    /// raw mode off. Idempotent; safe on panic via the controller's hook.
    ///
    /// # Errors
    ///
    /// Fails if a restore write fails; raw-mode disable errors are reported
    /// too.
    pub fn leave(&mut self) -> io::Result<()> {
        use crossterm::event::PopKeyboardEnhancementFlags;
        if self.fullscreen {
            self.raw.raw(&format!("{MOUSE_TRACKING_OFF}{LEAVE_ALT}"))?;
            self.fullscreen = false;
        }
        let mut out = String::new();
        if self.kitty == KittyKeys::Pushed {
            out.push_str(&cmd_ansi(&PopKeyboardEnhancementFlags));
        } else if self.kitty == KittyKeys::ModifyOtherKeys {
            out.push_str(MODIFY_OTHER_KEYS_OFF);
        }
        out.push_str(BRACKETED_PASTE_OFF);
        out.push_str(DECCKM_NORMAL);
        out.push_str("\x1b[?25h");
        self.raw.raw(&out)?;
        ratatui::backend::Backend::flush(self.term.backend_mut())?;
        crossterm::terminal::disable_raw_mode()
    }
}

impl<B: Backend, R: RawOut> Shell<B, R> {
    /// Test/support constructor over an arbitrary backend.
    #[cfg(test)]
    fn new_for_backend(term: Terminal<B>, raw: R, live_height: u16, sync: bool) -> Self {
        Self {
            screen: Rect::from((Position::ORIGIN, term.size().unwrap_or(Size::new(80, 24)))),
            term,
            raw,
            live_height: live_height.max(1),
            sync,
            kitty: KittyKeys::Inactive,
            mux: Multiplexer::None,
            fullscreen: false,
        }
    }

    /// Number of rows in the live region.
    #[must_use]
    pub fn live_height(&self) -> u16 {
        self.live_height
    }

    /// Current terminal width in columns (committed rows are wrapped at this
    /// width; after a width change a rebuild re-commits at the new one).
    #[must_use]
    pub fn width(&self) -> u16 {
        self.screen.width
    }

    fn paint_begin(&mut self) -> io::Result<()> {
        let seq = if self.sync {
            PAINT_BEGIN
        } else {
            PAINT_BEGIN_NO_SYNC
        };
        self.raw.raw(seq)
    }

    fn paint_end(&mut self) -> io::Result<()> {
        let seq = if self.sync {
            PAINT_END
        } else {
            PAINT_END_NO_SYNC
        };
        self.raw.raw(seq)
    }

    /// Commit finished transcript rows: print them above the live region
    /// (pushing them into native scrollback) and never redraw them. Empty
    /// batches are a no-op so callers can drain unconditionally.
    ///
    /// The batch is bracketed with OSC 133 prompt-zone marks written raw
    /// around the insert (inside the same synchronized paint): embedding the
    /// escapes in row content would clip them against the cell-grid width.
    /// The zone opens and closes inside this one paint, matching the TS
    /// `user-message.ts` discipline (Ghostty's sticky `133;B` input semantic
    /// is cleared by the immediate `133;C`/`133;D;0`).
    ///
    /// # Errors
    ///
    /// Fails if the scroll/insert writes fail.
    pub fn commit(&mut self, lines: &[Line]) -> io::Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        self.paint_begin()?;
        self.raw.raw(OSC133_ZONE_START)?;
        let frame = self.term.insert_before(u16::try_from(lines.len()).unwrap_or(u16::MAX), |buf| {
            let area = buf.area;
            Paragraph::new(lines.to_vec()).render(area, buf);
        });
        let result = frame;
        self.raw.raw(OSC133_ZONE_CLOSE)?;
        self.paint_end()?;
        result
    }

    /// Repaint the live region from pre-rendered rows (bottom-aligned, top
    /// blank-padded). Width cropping is the widget's job; rows must not
    /// contain newlines.
    ///
    /// # Errors
    ///
    /// Fails if the paint writes fail.
    pub fn set_live(&mut self, lines: &[Line]) -> io::Result<()> {
        self.paint_begin()?;
        let height = self.live_height as usize;
        let frame = plan_live_frame(lines, height);
        let drawn = self.term.draw(|f| {
            let area = f.area();
            f.render_widget(Paragraph::new(frame), area);
            // Park the hardware cursor on the viewport's top row (the resize
            // anchor invariant, core.md §2.9) — it is hidden right after.
            f.set_cursor_position(Position {
                x: area.x,
                y: area.y,
            });
        });
        let result = drawn.map(|_| ());
        // The draw above *shows* the cursor at the parked position; hide it
        // again before ending the synchronized update.
        self.raw.raw(HIDE_CURSOR)?;
        self.paint_end()?;
        result
    }

    /// Convenience: draw the live region through a ratatui closure (the
    /// pre-`set_live` path while components still paint via `Frame`).
    ///
    /// # Errors
    ///
    /// Fails if the paint writes fail.
    pub fn draw_live(&mut self, draw: impl FnOnce(&mut ratatui::Frame<'_>)) -> io::Result<()> {
        self.paint_begin()?;
        let mut parked = None;
        let result = self
            .term
            .try_draw(|f| {
                draw(f);
                let area = f.area();
                parked = Some(Position {
                    x: area.x,
                    y: area.y,
                });
                io::Result::Ok(())
            })
            .map(|f| {
                self.screen = f.area;
            });
        if let Some(pos) = parked {
            // Park cursor at the live region's top row, then hide it.
            let _ = self.term.set_cursor_position(pos);
        }
        self.raw.raw(HIDE_CURSOR)?;
        self.paint_end()?;
        result
    }

    /// Terminal resized. Repaints the live region; when the width changed
    /// inside a mux (tmux/herdr/…, which owns the grid and does not reflow
    /// the app's anchor in place), erases the scrollback and reports
    /// [`ResizeAction::RebuildScrollback`] so the caller re-commits the
    /// transcript at the new width (core.md §5 "rebuild" mode).
    ///
    /// # Errors
    ///
    /// Fails if the repaint writes fail.
    pub fn on_resize(&mut self) -> io::Result<ResizeAction> {
        let before = self.term.size()?;
        self.term.autoresize()?;
        let after = self.term.size()?;
        let width_changed = before.width != after.width;
        self.screen = Rect::from((Position::ORIGIN, after));
        if width_changed && self.mux.owns_grid() {
            self.clear_history()?;
            return Ok(ResizeAction::RebuildScrollback);
        }
        Ok(ResizeAction::Repaint)
    }

    /// Erase the committed scrollback and invalidate the live region's diff
    /// state (ED2 before ED3, core.md §2.5).
    ///
    /// # Errors
    ///
    /// Fails if the erase or repaint writes fail.
    pub fn clear_history(&mut self) -> io::Result<()> {
        self.raw.raw(CLEAR_SCREEN_AND_SCROLLBACK)?;
        // ratatui diffs against its cached buffers; after an external erase
        // that state is stale, so force a full repaint.
        self.term.resize(self.screen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::text::{Line, Span};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Shared [`RawOut`] that records escape sequences written outside ratatui.
    #[derive(Clone, Default)]
    struct Recorder(Rc<RefCell<Vec<String>>>);
    impl RawOut for Recorder {
        fn raw(&mut self, s: &str) -> io::Result<()> {
            self.0.borrow_mut().push(s.to_owned());
            Ok(())
        }
    }
    impl Recorder {
        fn take(&self) -> Vec<String> {
            std::mem::take(&mut *self.0.borrow_mut())
        }
    }

    fn line(text: &str) -> Line<'static> {
        Line::from(vec![Span::raw(text.to_owned())])
    }

    fn row(buf: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf[(x, y)].symbol().to_owned())
            .collect()
    }

    #[test]
    fn plan_live_frame_keeps_bottom_rows() {
        let lines: Vec<Line> = (0..5).map(|i| line(&format!("l{i}"))).collect();
        let planned = plan_live_frame(&lines, 3);
        assert_eq!(planned.len(), 3);
        // Exactly fills the height: bottom three rows, no padding.
        assert_eq!(planned[0], lines[2]);
        assert_eq!(planned[1], lines[3]);
        assert_eq!(planned[2], lines[4]);
        // One extra row forces a top crop.
        let planned = plan_live_frame(&lines, 4);
        assert_eq!(planned[0], lines[1]);
    }

    #[test]
    fn plan_live_frame_pads_short_frames() {
        let lines = [line("only")];
        let planned = plan_live_frame(&lines, 4);
        assert_eq!(planned.len(), 4);
        assert_eq!(planned[0], Line::default());
        assert_eq!(planned[3], lines[0]);
    }

    #[test]
    fn sync_mode_env_override_wins() {
        // Set/restore around the global env access.
        std::env::set_var("OMP_TUI_SYNC_OUTPUT", "0");
        std::env::set_var("TERM", "xterm-kitty");
        assert!(!sync_output_enabled());
        std::env::set_var("OMP_TUI_SYNC_OUTPUT", "1");
        assert!(sync_output_enabled());
        std::env::remove_var("OMP_TUI_SYNC_OUTPUT");
        std::env::remove_var("TERM");
    }

    #[test]
    fn set_live_paints_bottom_aligned_and_brackets_sync() {
        let backend = TestBackend::new(20, 10);
        let term = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(4),
            },
        )
        .unwrap();
        let rec = Recorder::default();
        let mut shell = Shell::new_for_backend(term, rec.clone(), 4, true);
        shell.set_live(&[line("editor"), line("status")]).unwrap();
        let buf = shell.term.backend().buffer();
        // Bottom-aligned in the 4-row viewport. (A fresh TestBackend screen
        // keeps the inline viewport at the top — no scrollback pressure; a
        // real terminal anchors it at the screen bottom.)
        assert_eq!(row(buf, 0), "                    ");
        assert_eq!(row(buf, 1), "                    ");
        assert_eq!(row(buf, 2), "editor              ");
        assert_eq!(row(buf, 3), "status              ");
        let joined = rec.take().join("|");
        assert!(joined.contains(SYNC_OUTPUT_BEGIN), "sync begin: {joined}");
        assert!(joined.contains(SYNC_OUTPUT_END), "sync end: {joined}");
        assert!(joined.contains(DISABLE_AUTOWRAP), "decawm off: {joined}");
        assert!(joined.contains(ENABLE_AUTOWRAP), "decawm on: {joined}");
        assert!(joined.contains(HIDE_CURSOR), "hide cursor: {joined}");
    }

    #[test]
    fn commit_inserts_rows_above_viewport() {
        let backend = TestBackend::new(20, 10);
        let term = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(3),
            },
        )
        .unwrap();
        let rec = Recorder::default();
        let mut shell = Shell::new_for_backend(term, rec, 3, false);
        shell.set_live(&[line("live")]).unwrap();
        shell
            .commit(&[line("history-1"), line("history-2")])
            .unwrap();
        // A real terminal scrolls its grid on insert_before; the live
        // region is repainted on the next frame (the controller draws after
        // every event). Mirror that here before asserting the buffer.
        shell.set_live(&[line("live")]).unwrap();
        let buf = shell.term.backend().buffer();
        // Committed rows are printed above the live viewport (on a real
        // terminal the scroll pushes prior rows into native scrollback; the
        // TestBackend buffer keeps everything on the 10-row screen).
        let rows: Vec<String> = (0..10).map(|y| row(buf, y)).collect();
        assert!(
            rows.contains(&"history-1           ".to_owned()),
            "history-1 missing: {rows:?}"
        );
        assert!(
            rows.contains(&"history-2           ".to_owned()),
            "history-2 missing: {rows:?}"
        );
        assert!(
            rows.contains(&"live                ".to_owned()),
            "live row missing: {rows:?}"
        );
        // The committed rows are above the live row on screen.
        let h1 = rows
            .iter()
            .position(|r| r.starts_with("history-1"))
            .unwrap();
        let live = rows.iter().position(|r| r.starts_with("live")).unwrap();
        assert!(h1 < live, "history must be above the live region: {rows:?}");
    }

    #[test]
    fn commit_wraps_batch_in_osc133_prompt_zone() {
        let backend = TestBackend::new(20, 10);
        let term = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(3),
            },
        )
        .unwrap();
        let rec = Recorder::default();
        let mut shell = Shell::new_for_backend(term, rec.clone(), 3, false);
        shell.set_live(&[line("live")]).unwrap();
        rec.take();
        shell.commit(&[line("history-1")]).unwrap();
        shell.set_live(&[line("live")]).unwrap();
        // The zone marks are written raw, inside the same paint, in order.
        let written = rec.take().join("");
        let start_at = written.find(OSC133_ZONE_START).expect("zone start emitted");
        let close_at = written.find(OSC133_ZONE_CLOSE).expect("zone close emitted");
        assert!(start_at < close_at, "{written:?}");
        assert!(written[..start_at].contains(HIDE_CURSOR), "{written:?}");
        assert!(written[close_at..].contains(ENABLE_AUTOWRAP), "{written:?}");
        // The committed row itself lands in the buffer above the live row.
        let buf = shell.term.backend().buffer();
        let rows: Vec<String> = (0..10).map(|y| row(buf, y)).collect();
        let h1 = rows.iter().position(|r| r.starts_with("history-1")).unwrap();
        let live = rows.iter().position(|r| r.starts_with("live")).unwrap();
        assert!(h1 < live, "{rows:?}");
    }

    #[test]
    fn commit_empty_is_noop() {
        let backend = TestBackend::new(20, 10);
        let term = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(3),
            },
        )
        .unwrap();
        let rec = Recorder::default();
        let mut shell = Shell::new_for_backend(term, rec.clone(), 3, true);
        shell.set_live(&[line("live")]).unwrap();
        rec.take();
        shell.commit(&[]).unwrap();
        assert!(rec.take().is_empty(), "no paint sequences expected");
    }

    #[test]
    fn resize_without_mux_is_repaint() {
        let backend = TestBackend::new(20, 10);
        let term = Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Inline(3),
            },
        )
        .unwrap();
        let rec = Recorder::default();
        let mut shell = Shell::new_for_backend(term, rec, 3, false);
        shell.set_live(&[line("live")]).unwrap();
        // Same size: autoresize is a no-op, plain repaint.
        assert_eq!(shell.on_resize().unwrap(), ResizeAction::Repaint);
    }
}
