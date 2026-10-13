//! Help / keyhints overlay — port of `pi-tui/src/overlays/help-overlay.ts`.
//!
//! Stub: renders a bordered centered box listing the MVP key hints.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// The help overlay.
#[derive(Debug, Clone, Default)]
pub struct Help;

impl Component for Help {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::widgets::{Block, Borders, Clear, Paragraph};
        let popup = crate::overlays::centered(area, 60, 70);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(" help ")
            .borders(Borders::ALL)
            .border_style(ratatui::style::Style::default().fg(theme.color("border")));
        let text = "enter: send prompt\nctrl+j: newline\nctrl+c: abort run / quit\nscroll: pgup/pgdown";
        frame.render_widget(Paragraph::new(text).block(block), popup);
    }
}
