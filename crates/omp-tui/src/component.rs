//! The component contract every on-screen widget implements.
//!
//! This is the ratatui-native analogue of pi-tui's
//! `TuiNode`/`TerminalFrameProvider` model (`pi-tui/src/tui.ts`): instead of
//! owning a differential inline renderer, each widget paints itself into a
//! ratatui [`Frame`] region each tick, and the terminal loop diffs full
//! frames. Interactivity is layered on top: widgets return
//! [`EventOutcome`] from `handle_event` and the [`crate::app::App`] routes
//! keys according to focus order (editor > focused overlay > chat > status).
//!
//! ## Implementing a component
//!
//! - Take `&self` in `render`; keep mutation in `handle_event`.
//! - Never hardcode colors: resolve everything through the [`Theme`].
//! - Never assume a size: lay out with `ratatui::layout` constraints.
//! - Keep per-frame work proportional to the visible area (chat keeps a
//!   scrollback, render only the window).

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::keys::KeyEvent;
use crate::theme::Theme;

/// What a component did with an input event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventOutcome {
    /// The event was consumed; stop routing.
    Consumed,
    /// The component ignored it; keep routing to the next focus target.
    Ignored,
}

impl EventOutcome {
    /// Whether the event was consumed.
    #[must_use]
    pub fn consumed(self) -> bool {
        matches!(self, Self::Consumed)
    }
}

/// A renderable, optionally interactive widget.
pub trait Component {
    /// Paint this widget into `area` of `frame` using `theme`.
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme);

    /// Handle a key event. The default ignores everything; interactive
    /// components override this. `app` may inspect `actions` pushed by the
    /// component via the return of the owning container — see
    /// [`crate::app::App`] for the dispatch convention.
    fn handle_key(&mut self, _key: KeyEvent) -> EventOutcome {
        EventOutcome::Ignored
    }

    /// Handle a decoded SGR mouse event (click, wheel, hover/drag). Mouse
    /// reporting is only enabled while a fullscreen overlay owns the
    /// alternate screen (or inline `tui.mouse` capture is on), and the event
    /// is routed to the topmost overlay only. `event.row`/`event.col` are
    /// 0-based screen cells; fullscreen overlays render into
    /// `Rect { x: 0, y: 0, .. }`, so they map 1:1 onto the last render.
    /// The default ignores everything.
    fn handle_mouse(&mut self, _event: &crate::mouse::SgrMouseEvent) -> EventOutcome {
        EventOutcome::Ignored
    }

    /// Whether this component wants keyboard focus right now (overlays that
    /// are open return `true`; the editor returns `true` unless an overlay
    /// owns focus).
    fn wants_focus(&self) -> bool {
        false
    }

    /// Drain intents produced by the most recent `handle_key` calls.
    /// Interactive components (overlays, editor) push [`Action`]s here
    /// instead of talking to the runtime directly; the app drains after each
    /// routing step and dispatches them through [`crate::RuntimePort`].
    /// Default: none.
    fn take_actions(&mut self) -> Vec<crate::action::Action> {
        Vec::new()
    }
}
