//! Full-screen diff overlay — port of `pi-tui/src/overlays/diff-overlay.ts`
//! (unified diff, styled with the toolDiff* theme tokens, scrollable).
//!
//! Data in: a raw unified diff (`Path/Text` `rawDiff` from an edit tool
//! entry, or a `git diff` blob). Parsing is pure so it is unit-tested
//! independently of rendering.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent};
use crate::theme::Theme;

/// One rendered row of a unified diff (TS `ReviewDiffRow`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffRow {
    /// `---`/`+++` file header.
    Header(String),
    /// `@@ -a,b +c,d @@` hunk header.
    Hunk(String),
    /// Unchanged context line.
    Context(String),
    /// `+` line.
    Added(String),
    /// `-` line.
    Removed(String),
    /// `\ No newline at end of file`.
    NoNewline,
}

impl DiffRow {
    /// The plain text of the row (empty for `NoNewline`).
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Header(t)
            | Self::Hunk(t)
            | Self::Context(t)
            | Self::Added(t)
            | Self::Removed(t) => t,
            Self::NoNewline => "",
        }
    }

    /// Theme token used to color the row.
    #[must_use]
    pub fn token(&self) -> &'static str {
        match self {
            Self::Header(_) | Self::Hunk(_) => "dim",
            Self::Context(_) => "text",
            Self::Added(_) => "success",
            Self::Removed(_) => "error",
            Self::NoNewline => "warning",
        }
    }
}

/// Parse a unified diff into styled rows. Unknown lines become context so a
/// malformed diff still renders (TS vendored `pi-diff` is lenient too).
#[must_use]
pub fn parse_unified_diff(raw: &str) -> Vec<DiffRow> {
    raw.lines()
        .map(|line| {
            if line.starts_with("--- ") || line.starts_with("+++ ") {
                DiffRow::Header(line.to_owned())
            } else {
                match line.as_bytes().first() {
                    Some(b'+') => DiffRow::Added(line.to_owned()),
                    Some(b'-') => DiffRow::Removed(line.to_owned()),
                    Some(b'@') if line.starts_with("@@") => DiffRow::Hunk(line.to_owned()),
                    _ if line.starts_with("\\ No newline") => DiffRow::NoNewline,
                    _ => DiffRow::Context(line.to_owned()),
                }
            }
        })
        .collect()
}

/// Added/removed line counts (excluding headers and hunks).
#[must_use]
pub fn count_changes(rows: &[DiffRow]) -> (usize, usize) {
    rows.iter().fold((0, 0), |(a, r), row| match row {
        DiffRow::Added(_) => (a + 1, r),
        DiffRow::Removed(_) => (a, r + 1),
        _ => (a, r),
    })
}

/// Style one row for the viewer.
#[must_use]
pub fn row_span(row: &DiffRow, theme: &Theme) -> Span<'static> {
    Span::styled(
        row.text().to_owned(),
        Style::default().fg(theme.color(row.token())),
    )
}

/// Scrollable viewer for one file's unified diff.
#[derive(Debug, Clone, Default)]
pub struct DiffView {
    /// Title (path under edit).
    pub title: String,
    rows: Vec<DiffRow>,
    scroll: u16,
}

impl DiffView {
    /// Load a diff and reset scroll.
    pub fn set_diff(&mut self, title: &str, raw: &str) {
        self.title = title.to_owned();
        self.rows = parse_unified_diff(raw);
        self.scroll = 0;
    }

    /// (`+added`, `-removed`) counts for the title line.
    #[must_use]
    pub fn counts(&self) -> (usize, usize) {
        count_changes(&self.rows)
    }

    /// Clamp the scroll so the last page is reachable but no further.
    #[cfg(test)]
    fn clamp_scroll(&mut self, body: u16) {
        let max = self.rows.len().saturating_sub(body as usize) as u16;
        self.scroll = self.scroll.min(max);
    }
}

impl Component for DiffView {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let popup = super::centered(area, 92, 90);
        if popup.width < 4 || popup.height < 3 {
            return;
        }
        frame.render_widget(Clear, popup);
        let (added, removed) = self.counts();
        let title = format!(
            " diff: {}  +{added} -{removed} ",
            if self.title.is_empty() {
                "(no file)"
            } else {
                &self.title
            }
        );
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);

        let body = self.rows.len().min(inner.height as usize);
        let start = (self.scroll as usize).min(self.rows.len().saturating_sub(body));
        let lines: Vec<Line<'_>> = self.rows[start..start + body]
            .iter()
            .map(|row| Line::from(vec![row_span(row, theme)]))
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        let step = match key.code {
            KeyCode::Up | KeyCode::Char('k') => Some(-1),
            KeyCode::Down | KeyCode::Char('j') => Some(1),
            KeyCode::PageUp => Some(-20),
            KeyCode::PageDown => Some(20),
            KeyCode::Home => {
                self.scroll = 0;
                return EventOutcome::Consumed;
            }
            KeyCode::End => {
                self.scroll = u16::MAX;
                return EventOutcome::Consumed;
            }
            _ => return EventOutcome::Ignored,
        };
        if let Some(step) = step {
            self.scroll = self.scroll.saturating_add_signed(step);
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
    use crate::keys::KeyModifiers;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    const SAMPLE: &str = "--- a/main.rs\n+++ b/main.rs\n@@ -1,3 +1,4 @@\n fn main() {\n-    old();\n+    new();\n+    more();\n }";

    #[test]
    fn parse_counts_and_kinds() {
        let rows = parse_unified_diff(SAMPLE);
        assert_eq!(rows[2], DiffRow::Hunk("@@ -1,3 +1,4 @@".into()));
        assert_eq!(rows[3], DiffRow::Context(" fn main() {".into()));
        assert_eq!(rows[4], DiffRow::Removed("-    old();".into()));
        assert_eq!(rows[5], DiffRow::Added("+    new();".into()));
        assert_eq!(count_changes(&rows), (2, 1));
    }

    #[test]
    fn scroll_keys_and_clamp() {
        let mut v = DiffView::default();
        v.set_diff("main.rs", SAMPLE);
        v.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
        // max scroll = rows(6) - body; body depends on frame, just ensure clamped later
        v.clamp_scroll(10);
        assert_eq!(v.scroll, 0);
        v.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(v.scroll, 1);
        v.handle_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert_eq!(v.scroll, 0);
    }

    #[test]
    fn render_snapshot() {
        let mut v = DiffView::default();
        v.set_diff("main.rs", SAMPLE);
        let theme = Theme::default();
        let mut term = Terminal::new(TestBackend::new(40, 10)).unwrap();
        term.draw(|f| v.render(f, f.area(), &theme)).unwrap();
        let out = term.backend().to_string();
        assert!(out.contains("diff: main.rs"), "{out}");
        assert!(out.contains("+    new();"), "{out}");
        assert!(out.contains("-    old();"), "{out}");
    }
}
