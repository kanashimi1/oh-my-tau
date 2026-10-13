//! Centered-popup helper shared by overlays, plus the model picker — port of
//! `pi-tui/src/overlays/model-picker.ts`.
//!
//! Stub: renders an empty bordered box; the picker implementer will populate
//! it from `Runtime::available_models` via a new `RuntimePort` method.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// The model picker overlay.
#[derive(Debug, Clone, Default)]
pub struct ModelPicker {
    /// Providers flattened to `provider/id` rows.
    pub models: Vec<String>,
    /// Selected row.
    pub selected: usize,
}

impl Component for ModelPicker {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::text::Line;
        use ratatui::widgets::{Block, Borders, Clear, List};
        let popup = super::centered(area, 50, 50);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(" model ")
            .borders(Borders::ALL)
            .border_style(ratatui::style::Style::default().fg(theme.color("border")));
        let items: Vec<Line<'_>> = if self.models.is_empty() {
            vec![Line::from("(no models yet)")]
        } else {
            self.models.iter().map(|m| Line::from(m.clone())).collect()
        };
        frame.render_widget(List::new(items).block(block), popup);
    }
}
