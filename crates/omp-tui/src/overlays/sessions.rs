//! Session picker overlay — port of `pi-tui/src/overlays/session-picker.ts`.
//!
//! Stub: empty bordered box. Planned: session list from the sessions dir
//! (newest first, title + relative time), fuzzy filter, resume/fork actions.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// The session picker overlay.
#[derive(Debug, Clone, Default)]
pub struct SessionPicker {
    /// Session rows (display strings).
    pub sessions: Vec<String>,
}

impl Component for SessionPicker {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::text::Line;
        use ratatui::widgets::{Block, Borders, Clear, List};
        let popup = super::centered(area, 60, 60);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(" sessions ")
            .borders(Borders::ALL)
            .border_style(ratatui::style::Style::default().fg(theme.color("border")));
        let items: Vec<Line<'_>> = if self.sessions.is_empty() {
            vec![Line::from("(session picker not implemented)")]
        } else {
            self.sessions.iter().map(|s| Line::from(s.clone())).collect()
        };
        frame.render_widget(List::new(items).block(block), popup);
    }
}
