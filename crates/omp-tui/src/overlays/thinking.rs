//! Thinking-level selector — port of `pi-tui/src/overlays/thinking-selector.ts`
//! (`ThinkingSelectorComponent`, opened by a bare `/effort`).
//!
//! A bordered "Thinking Level" panel anchored over the composer with one row
//! per selectable effort (`off`, then the model's supported efforts); the
//! current level is preselected, Enter picks, Esc goes back.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent};
use crate::theme::Theme;

/// Label and description of a thinking selector value
/// (`getConfiguredThinkingLevelMetadata`).
#[must_use]
pub fn level_metadata(level: &str) -> (&'static str, &'static str) {
    match level {
        "off" => ("off", "No reasoning"),
        "auto" => ("auto", "Auto-detect per prompt"),
        "minimal" => ("min", "Very brief reasoning (~1k tokens)"),
        "low" => ("low", "Light reasoning (~2k tokens)"),
        "medium" => ("medium", "Moderate reasoning (~8k tokens)"),
        "high" => ("high", "Deep reasoning (~16k tokens)"),
        "xhigh" => ("xhigh", "Extended reasoning (~32k tokens)"),
        "max" => ("max", "Maximum reasoning the model supports"),
        _ => ("", ""),
    }
}

/// The selector panel state.
#[derive(Debug, Clone, Default)]
pub struct ThinkingSelector {
    /// Selector values in cycle order.
    levels: Vec<String>,
    selected: usize,
    picked: Option<String>,
}

impl ThinkingSelector {
    /// Open over `levels` (e.g. `off, minimal, …`) with `current` preselected.
    pub fn open(&mut self, levels: Vec<String>, current: &str) {
        self.selected = levels.iter().position(|l| l == current).unwrap_or(0);
        self.levels = levels;
        self.picked = None;
    }

    /// The level the user picked, once.
    pub fn take_pick(&mut self) -> Option<String> {
        self.picked.take()
    }

    /// Height of the panel (rows + borders).
    #[must_use]
    pub fn height(&self) -> u16 {
        u16::try_from(self.levels.len() + 2).unwrap_or(u16::MAX)
    }
}

impl Component for ThinkingSelector {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let height = self.height().min(area.height);
        if height < 3 || area.width < 20 {
            return;
        }
        // Anchored to the bottom: the panel replaces the composer (TS mounts
        // it in the editor container).
        let panel = Rect {
            x: area.x,
            y: area.y + area.height - height,
            width: area.width,
            height,
        };
        frame.render_widget(Clear, panel);
        let block = Block::default()
            .title("─ Thinking Level ")
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(panel);
        frame.render_widget(block, panel);
        let accent = Style::default().fg(theme.color("accent"));
        let text = Style::default().fg(theme.color("text"));
        let dim = Style::default().fg(theme.color("dim"));
        let lines: Vec<Line<'_>> = self
            .levels
            .iter()
            .enumerate()
            .map(|(i, level)| {
                let (label, desc) = level_metadata(level);
                let current = i == self.selected;
                let prefix = if current { "❯ " } else { "  " };
                let style = if current { accent } else { text };
                Line::from(vec![
                    Span::styled(format!(" {prefix}"), style),
                    Span::styled(format!("{label:<32}"), style),
                    Span::styled(desc.to_owned(), if current { accent } else { dim }),
                ])
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        let len = self.levels.len();
        match key.code {
            KeyCode::Up if len > 0 => self.selected = (self.selected + len - 1) % len,
            KeyCode::Down if len > 0 => self.selected = (self.selected + 1) % len,
            KeyCode::Enter => {
                self.picked = self.levels.get(self.selected).cloned();
            }
            // Esc bubbles so the host pops the panel.
            KeyCode::Esc => return EventOutcome::Ignored,
            _ => {}
        }
        EventOutcome::Consumed
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn selector() -> ThinkingSelector {
        let mut s = ThinkingSelector::default();
        s.open(
            ["off", "minimal", "low", "medium", "high"]
                .map(String::from)
                .to_vec(),
            "medium",
        );
        s
    }

    #[test]
    fn preselects_current_and_picks_with_enter() {
        let mut s = selector();
        assert_eq!(s.selected, 3);
        s.handle_key(KeyEvent::from(KeyCode::Down));
        s.handle_key(KeyEvent::from(KeyCode::Enter));
        assert_eq!(s.take_pick().as_deref(), Some("high"));
        assert_eq!(s.take_pick(), None);
    }

    #[test]
    fn navigation_wraps_and_esc_bubbles() {
        let mut s = selector();
        s.selected = 0;
        s.handle_key(KeyEvent::from(KeyCode::Up));
        assert_eq!(s.selected, 4);
        assert!(!s.handle_key(KeyEvent::from(KeyCode::Esc)).consumed());
    }

    #[test]
    fn renders_titled_panel_with_ts_labels() {
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        let s = selector();
        terminal
            .draw(|f| s.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let rows: Vec<String> = (0..12)
            .map(|y| (0..80).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect();
        let text = rows.join("\n");
        assert!(text.contains("Thinking Level"), "{text}");
        assert!(
            text.contains("min") && text.contains("Very brief reasoning (~1k tokens)"),
            "{text}"
        );
        assert!(text.contains("❯ medium"), "{text}");
    }
}
