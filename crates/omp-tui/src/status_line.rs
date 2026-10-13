//! The bottom status line — the port of `pi-tui/src/status-line/`
//! (`status-line.ts`, `status-line-host.ts` in pi-coding-agent).
//!
//! MVP: one line showing the model, the working directory, the streaming
//! state and the key hints for the current context (TS renders context
//! usage, todo phases, mode indicators and extension items; those slots
//! arrive with their submodules). Rendered as the last row of the root
//! layout by [`crate::app::App`].

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// What the status line knows about the session right now.
#[derive(Debug, Clone)]
pub struct StatusLine {
    /// `provider/id` of the active model.
    pub model: String,
    /// Working directory basename.
    pub cwd: String,
    /// Whether the agent is streaming.
    pub streaming: bool,
    /// Queued steer/follow-up counts.
    pub queued: usize,
}

impl StatusLine {
    /// Build the initial status from [`crate::RunConfig`] data.
    #[must_use]
    pub fn new(model: String, cwd: impl Into<String>) -> Self {
        Self {
            model,
            cwd: cwd.into(),
            streaming: false,
            queued: 0,
        }
    }
}

impl Component for StatusLine {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        use ratatui::style::{Modifier, Style};
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Paragraph;
        let state = if self.streaming { "streaming" } else { "idle" };
        let queue = if self.queued > 0 {
            format!(" (+{} queued)", self.queued)
        } else {
            String::new()
        };
        let line = Line::from(vec![
            Span::styled(
                self.model.clone(),
                Style::default()
                    .fg(theme.color("accent"))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}  {state}{queue}", self.cwd),
                Style::default().fg(theme.color("text_muted")),
            ),
            Span::styled(
                "  enter: send · ctrl+c: abort/quit",
                Style::default().fg(theme.color("text_muted")),
            ),
        ]);
        frame.render_widget(Paragraph::new(line), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn renders_model_and_hints() {
        let backend = TestBackend::new(80, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        let status = StatusLine::new("zai/glm".into(), "/repo");
        terminal
            .draw(|f| status.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let row: String = (0..80).map(|x| buf[(x, 0)].symbol()).collect();
        assert!(row.contains("zai/glm"));
        assert!(row.contains("idle"));
        assert!(row.contains("enter: send"));
    }
}
