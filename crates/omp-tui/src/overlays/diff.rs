//! Full-screen diff overlay — port of `pi-tui/src/overlays/diff-overlay.ts`.
//!
//! Stub: placeholder box. Planned: side-by-side or unified diff of a pending
//! edit with accept/reject affordances (TS uses the vendored `pi-diff`
//! engine and the pager from `render/output-pane.ts`).

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// The diff overlay.
#[derive(Debug, Clone, Default)]
pub struct DiffView {
    /// Title (path under edit).
    pub title: String,
}

impl Component for DiffView {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::widgets::{Block, Borders, Clear, Paragraph};
        let popup = super::centered(area, 90, 90);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(format!(" diff: {} ", self.title))
            .borders(Borders::ALL)
            .border_style(ratatui::style::Style::default().fg(theme.color("border")));
        frame.render_widget(Paragraph::new("(diff view not implemented)").block(block), popup);
    }
}
