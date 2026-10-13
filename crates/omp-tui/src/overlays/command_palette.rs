//! Command palette overlay — port of
//! `pi-coding-agent/src/modes/components/command-palette.ts`.
//!
//! Stub: empty bordered box. Planned: fuzzy-filtered list of slash commands
//! (`available_commands_update` frames) and UI actions, ranked with
//! [`crate::autocomplete::fuzzy_score`].

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// The command palette overlay.
#[derive(Debug, Clone, Default)]
pub struct CommandPalette {
    /// Current filter text.
    pub query: String,
    /// Candidate commands (display strings).
    pub commands: Vec<String>,
}

impl Component for CommandPalette {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::widgets::{Block, Borders, Clear, Paragraph};
        let popup = super::centered(area, 60, 20);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(" command ")
            .borders(Borders::ALL)
            .border_style(ratatui::style::Style::default().fg(theme.color("border")));
        frame.render_widget(Paragraph::new(self.query.clone()).block(block), popup);
    }
}
