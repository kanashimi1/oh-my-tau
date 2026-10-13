//! Modal overlays — ports of `pi-tui/src/overlays/` and the
//! interactive-mode controllers in `pi-coding-agent/src/modes/controllers/`.
//!
//! An overlay is a [`Component`](crate::component::Component) that, when
//! open, owns keyboard focus and paints over a centered area of the screen
//! (TS: bordered box at a percentage of the terminal, theme `"border"`
//! token). Each overlay lives in its own module so implementers never
//! collide; [`Overlays`] tracks which one is open (TS: a stack — most
//! overlays are exclusive, help can stack over others in the future).

pub mod command_palette;
pub mod diff;
pub mod help;
pub mod models;
pub mod sessions;

use crate::component::Component;
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

/// Which overlay (if any) currently owns focus.
#[derive(Debug, Clone, Default)]
pub struct Overlays {
    help: help::Help,
    models: models::ModelPicker,
    sessions: sessions::SessionPicker,
    palette: command_palette::CommandPalette,
    diff: diff::DiffView,
    open: OpenOverlay,
}

/// The open-overlay selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenOverlay {
    /// None open; editor has focus.
    #[default]
    None,
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
}

impl Overlays {
    /// Toggle one overlay; opening another closes the current (TS behavior).
    pub fn toggle(&mut self, which: OpenOverlay) {
        self.open = if self.open == which {
            OpenOverlay::None
        } else {
            which
        };
    }

    /// The currently open overlay.
    #[must_use]
    pub fn open(&self) -> OpenOverlay {
        self.open
    }

    /// Render the open overlay (if any) centered over `area`.
    pub fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, theme: &Theme) {
        match self.open {
            OpenOverlay::None => {}
            OpenOverlay::Help => self.help.render(frame, area, theme),
            OpenOverlay::Models => self.models.render(frame, area, theme),
            OpenOverlay::Sessions => self.sessions.render(frame, area, theme),
            OpenOverlay::CommandPalette => self.palette.render(frame, area, theme),
            OpenOverlay::Diff => self.diff.render(frame, area, theme),
        }
    }
}
