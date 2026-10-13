//! Modal overlays — ports of `pi-tui/src/overlays/` and the
//! interactive-mode controllers in `pi-coding-agent/src/modes/controllers/`.
//!
//! An overlay is a [`Component`](crate::component::Component) that, when
//! open, owns keyboard focus and paints over a centered area of the screen
//! (TS: bordered box at a percentage of the terminal, theme `"border"`
//! token). Each overlay lives in its own module so implementers never
//! collide.
//!
//! [`Overlays`] is the focus stack: at most one overlay owns focus at a
//! time, but overlays can stack (help over the model picker, per the TS
//! overlay-manager comment that most overlays are exclusive while help may
//! stack over others). Key events go to the top overlay first and bubble
//! downward while ignored; [`Overlays::open`] reports the top for callers
//! that predate the stack.
//!
//! Popup overlays render into the live region (centered, `Clear`ed). A
//! full-screen overlay ([`OpenOverlay::is_fullscreen`], e.g. the diff view)
//! instead borrows the alternate screen from the shell renderer for the
//! duration it is on top.

pub mod command_palette;
pub mod diff;
pub mod help;
pub mod models;
pub mod sessions;
pub mod tree;

use crate::component::{Component, EventOutcome};
use crate::keys::KeyEvent;
use crate::theme::Theme;
use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// A centered rectangle of `percent_x`/`percent_y` of `area`, shared by all
/// popups (TS overlays use the same percentage-of-terminal rule).
#[must_use]
pub fn centered(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    let horiz = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vert[1]);
    horiz[1]
}

/// Which overlay (if any) is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenOverlay {
    /// Help/keyhints (`?`).
    Help,
    /// Model picker (`/model` or Tab).
    Models,
    /// Session picker (`/resume`).
    Sessions,
    /// Command palette (Cmd/Ctrl+K).
    CommandPalette,
    /// Full-screen diff view (from an edit tool entry).
    Diff,
    /// Session tree / rewind navigator (`/tree`, double-Esc rewind column).
    Tree,
}

impl OpenOverlay {
    /// Whether this overlay renders full-screen (alternate screen) rather
    /// than as a centered popup in the live region.
    #[must_use]
    pub fn is_fullscreen(self) -> bool {
        matches!(self, Self::Diff | Self::Tree)
    }
}

/// The focus stack: bottom-to-top order of open overlays. Empty means the
/// editor owns focus.
#[derive(Debug, Clone, Default)]
pub struct Overlays {
    help: help::Help,
    models: models::ModelPicker,
    sessions: sessions::SessionPicker,
    palette: command_palette::CommandPalette,
    diff: diff::DiffView,
    tree: tree::TreeNavigator,
    stack: Vec<OpenOverlay>,
}

impl Overlays {
    /// Toggle one overlay: if it is anywhere in the stack, remove it;
    /// otherwise push it on top. (TS: opening another overlay stacks it;
    /// re-toggling closes.)
    pub fn toggle(&mut self, which: OpenOverlay) {
        if let Some(pos) = self.stack.iter().position(|o| *o == which) {
            self.stack.remove(pos);
        } else {
            if which == OpenOverlay::Models {
                self.models.on_open();
            }
            self.stack.push(which);
        }
    }

    /// The overlay that currently owns focus (top of stack).
    #[must_use]
    pub fn open(&self) -> Option<OpenOverlay> {
        self.stack.last().copied()
    }

    /// The full stack, bottom-to-top.
    #[must_use]
    pub fn stack(&self) -> &[OpenOverlay] {
        &self.stack
    }

    /// Mutable access to the session picker (key routing + pick drain).
    pub fn sessions_mut(&mut self) -> &mut sessions::SessionPicker {
        &mut self.sessions
    }

    /// Close the top overlay (esc routing).
    pub fn pop(&mut self) {
        self.stack.pop();
    }

    /// Whether any overlay is open (i.e. the editor does not have focus).
    #[must_use]
    pub fn wants_focus(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Whether the topmost open overlay renders full-screen and thus needs
    /// the shell renderer's alternate screen (core.md §3). The controller
    /// borrows the alt screen per frame while this is true.
    #[must_use]
    pub fn is_fullscreen(&self) -> bool {
        self.open().is_some_and(OpenOverlay::is_fullscreen)
    }

    /// Back-compat alias for [`Self::is_fullscreen`].
    #[must_use]
    pub fn top_is_fullscreen(&self) -> bool {
        self.is_fullscreen()
    }

    /// Mutable access to the tree navigator (populate entries, drain picks).
    pub fn tree_mut(&mut self) -> &mut tree::TreeNavigator {
        &mut self.tree
    }

    fn component(&self, which: OpenOverlay) -> &dyn Component {
        match which {
            OpenOverlay::Help => &self.help,
            OpenOverlay::Models => &self.models,
            OpenOverlay::Sessions => &self.sessions,
            OpenOverlay::CommandPalette => &self.palette,
            OpenOverlay::Diff => &self.diff,
            OpenOverlay::Tree => &self.tree,
        }
    }

    fn component_mut(&mut self, which: OpenOverlay) -> &mut dyn Component {
        match which {
            OpenOverlay::Help => &mut self.help,
            OpenOverlay::Models => &mut self.models,
            OpenOverlay::Sessions => &mut self.sessions,
            OpenOverlay::CommandPalette => &mut self.palette,
            OpenOverlay::Diff => &mut self.diff,
            OpenOverlay::Tree => &mut self.tree,
        }
    }

    /// Route a key event to the open overlays, top first; bubbling stops at
    /// the first consumer. Returns `Ignored` when nothing is open or every
    /// overlay ignored the event.
    pub fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        for i in (0..self.stack.len()).rev() {
            let which = self.stack[i];
            if self.component_mut(which).handle_key(key).consumed() {
                return EventOutcome::Consumed;
            }
        }
        EventOutcome::Ignored
    }

    /// Drain every child's queued intents into `Action`s for the app to
    /// dispatch (TS: the interactive controller polls each overlay after
    /// key handling). Covers: model picker's SetModel/SetThinkingLevel,
    /// palette confirmations (`/cmd` → [`Action::Slash`]), session picker
    /// resolutions ([`Action::ResumeSession`]), tree selections
    /// ([`Action::Rewind`]). Resolved overlays (palette, sessions, tree) are
    /// popped automatically; Models stays open. The session picker is popped
    /// from the stack whenever its pick resolves (either variant).
    pub fn take_actions(&mut self) -> Vec<crate::action::Action> {
        use crate::action::Action;
        let mut out = self.models.take_actions();
        if let Some(label) = self.palette.take_picked() {
            let trimmed = label.trim_start_matches('/');
            let (name, args) = trimmed
                .split_once(char::is_whitespace)
                .map_or((trimmed, ""), |(n, a)| (n, a.trim_start()));
            if !name.is_empty() {
                out.push(Action::Slash {
                    name: name.to_string(),
                    args: args.to_string(),
                });
            }
            self.palette_close();
        }
        if let Some(sel) = self.tree.take_selection() {
            out.push(Action::Rewind {
                entry_id: sel.entry_id,
            });
            self.close(OpenOverlay::Tree);
        }
        if let Some(pick) = self.sessions.take_pick() {
            if let sessions::SessionPick::Resume(path) = pick {
                out.push(Action::ResumeSession { path, fork: false });
            }
            self.close(OpenOverlay::Sessions);
        }
        out
    }

    fn close(&mut self, which: OpenOverlay) {
        if let Some(pos) = self.stack.iter().position(|o| *o == which) {
            self.stack.remove(pos);
        }
    }

    fn palette_close(&mut self) {
        self.close(OpenOverlay::CommandPalette);
    }

    /// Render every open overlay, bottom-to-top, each over the full area
    /// (popups clear and center themselves; the full-screen top overlay is
    /// expected to have been handed the alternate screen by the shell).
    pub fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        theme: &Theme,
    ) {
        for &which in &self.stack {
            self.component(which).render(frame, area, theme);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::KeyCode;

    #[test]
    fn centered_is_centered() {
        let area = Rect::new(0, 0, 100, 40);
        let rect = centered(area, 60, 50);
        assert_eq!(rect.width, 60);
        assert_eq!(rect.height, 20);
        assert!(rect.x > 0 && rect.y > 0);
        assert!(rect.x + rect.width <= 100);
    }

    #[test]
    fn toggle_pushes_and_removes() {
        let mut o = Overlays::default();
        assert_eq!(o.open(), None);
        o.toggle(OpenOverlay::Help);
        assert_eq!(o.open(), Some(OpenOverlay::Help));
        o.toggle(OpenOverlay::Models);
        assert_eq!(o.stack(), &[OpenOverlay::Help, OpenOverlay::Models]);
        // Re-toggling help removes it from the middle, models stays top.
        o.toggle(OpenOverlay::Help);
        assert_eq!(o.stack(), &[OpenOverlay::Models]);
        o.toggle(OpenOverlay::Models);
        assert_eq!(o.open(), None);
    }

    #[test]
    fn esc_close_routing() {
        let mut o = Overlays::default();
        o.toggle(OpenOverlay::Help);
        let outcome = o.handle_key(KeyEvent::from(KeyCode::Esc));
        // Help ignores esc (the host owns open/close state) so it bubbles.
        assert!(!outcome.consumed());
        o.pop();
        assert_eq!(o.open(), None);
    }

    #[test]
    fn fullscreen_marker() {
        assert!(OpenOverlay::Diff.is_fullscreen());
        assert!(!OpenOverlay::Help.is_fullscreen());
        let mut o = Overlays::default();
        assert!(!o.top_is_fullscreen());
        o.toggle(OpenOverlay::Diff);
        assert!(o.top_is_fullscreen());
    }
}
