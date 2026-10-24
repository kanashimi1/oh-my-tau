//! Header banner — the top chrome row (spec §5.1 box layout's top border
//! content, simplified for the inline viewport): brand icon, model, project
//! dir, session name and run state, left-aligned with a themed rule filling
//! the remainder. Renders into any ratatui area of height ≥ 1.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::Frame;

use crate::component::Component;
use crate::theme::Theme;

/// Stable state label used by the banner and the terminal title: `>` idle,
/// `!` attention (the banner only distinguishes working/idle; the full
/// spinner lives in [`super::title`]).
#[must_use]
pub fn title_state_label(working: bool) -> &'static str {
    if working {
        "·"
    } else {
        ">"
    }
}

/// What the banner shows. Kept mirror-of-`App`-minimal; the app shell feeds
/// it each frame until the status-line box composer lands.
#[derive(Debug, Clone)]
pub struct Banner {
    /// `provider/id` of the active model.
    pub model: String,
    /// Project directory basename.
    pub dir: String,
    /// Session name (herdr-pane-names / `/rename`); accent-colored when set.
    pub session_name: Option<String>,
    /// Whether a turn is running.
    pub working: bool,
}

impl Banner {
    /// Build a banner from run-config data.
    #[must_use]
    pub fn new(model: String, dir: impl Into<String>) -> Self {
        Self {
            model,
            dir: dir.into(),
            session_name: None,
            working: false,
        }
    }

    /// Elapsed-time label for the working state (compact `12s` / `3m` / `7h`,
    /// capped at 99h — TS `brandTimer`).
    #[must_use]
    pub fn timer_label(elapsed_ms: u64) -> String {
        let s = elapsed_ms / 1000;
        if s < 60 {
            format!("{s}s")
        } else if s < 3600 {
            format!("{}m", s / 60)
        } else {
            format!("{}h", (s / 3600).min(99))
        }
    }
}

impl Component for Banner {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height == 0 {
            return;
        }
        let mut spans: Vec<Span<'static>> = Vec::new();
        spans.push(Span::styled(
            format!("π {} ", title_state_label(self.working)),
            Style::default()
                .fg(theme.color("accent"))
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            self.model.clone(),
            Style::default().fg(theme.color("text")),
        ));
        spans.push(Span::styled(
            format!(" {}", self.dir),
            Style::default().fg(theme.color("text_muted")),
        ));
        if let Some(name) = &self.session_name {
            spans.push(Span::styled(
                format!(" · {name}"),
                Style::default().fg(theme.color("accent")),
            ));
        }
        frame.render_widget(
            ratatui::widgets::Paragraph::new(Line::from(spans)),
            Rect::new(area.x, area.y, area.width, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn banner_shows_brand_model_dir_session() {
        let banner = Banner {
            session_name: Some("conv modpack".into()),
            ..Banner::new("zai/glm-5.3-flash".into(), "omt")
        };
        let backend = TestBackend::new(60, 1);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| banner.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let row: String = (0..60).map(|x| buf[(x, 0)].symbol()).collect();
        assert!(row.starts_with("\u{3c0} > "));
        assert!(row.contains("zai/glm-5.3-flash"));
        assert!(row.contains("omt"));
        assert!(row.contains("conv modpack"));
    }

    #[test]
    fn timer_label_units() {
        assert_eq!(Banner::timer_label(12_000), "12s");
        assert_eq!(Banner::timer_label(180_000), "3m");
        assert_eq!(Banner::timer_label(25_200_000), "7h");
        assert_eq!(Banner::timer_label(u64::from(u32::MAX) * 1000), "99h");
    }
}
