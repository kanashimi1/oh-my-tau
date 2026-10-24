//! Help / keyhints overlay — port of `pi-tui/src/overlays/hotkeys-markdown.ts`
//! (`HotkeysSheetComponent`) and `apps/git/help.ts` (`GitHelpSheet`).
//!
//! Renders the live keybindings as a scrollable table: groups of
//! `{keys, action}` rows inside a bordered centered box. Key glyphs are
//! plain-text for now — the dynamic `formatKeyHint` theming seam lands with
//! the keybindings loader (editor spec §12). Close keys: `?`, `q`, `esc`,
//! `ctrl+c` (TS: "Keys: `?`/q/esc/ctrl+c close").

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// One row of the sheet: a key list plus what it does.
#[derive(Debug, Clone)]
pub struct HotkeyRow {
    /// Displayed keys, e.g. `"ctrl+c"` (already platform-formatted).
    pub keys: String,
    /// The action label.
    pub action: String,
}

/// A titled section of the sheet.
#[derive(Debug, Clone)]
pub struct HotkeyGroup {
    /// Group title, e.g. `"General"`.
    pub title: String,
    /// Rows in display order.
    pub rows: Vec<HotkeyRow>,
}

/// Build the MVP keyhint table (TS `buildHotkeysMarkdown` with the default
/// app bindings; unbound actions are omitted, not shown as "Disabled").
#[must_use]
pub fn default_groups() -> Vec<HotkeyGroup> {
    let g = |title: &str, rows: &[(&str, &str)]| HotkeyGroup {
        title: title.into(),
        rows: rows
            .iter()
            .map(|(k, a)| HotkeyRow {
                keys: (*k).into(),
                action: (*a).into(),
            })
            .collect(),
    };
    vec![
        g(
            "General",
            &[
                ("enter", "send prompt"),
                ("ctrl+j", "insert newline"),
                ("ctrl+c", "abort run / quit when idle"),
                ("?", "toggle this help"),
                ("esc", "close overlay / cancel"),
                ("ctrl+k", "command palette"),
                ("pgup/pgdown", "scroll chat"),
            ],
        ),
        g(
            "Selection",
            &[
                ("up/down", "move selection"),
                ("enter/space", "confirm"),
                ("esc", "cancel"),
            ],
        ),
    ]
}

/// The help/keyhints overlay.
#[derive(Debug, Clone)]
pub struct Help {
    groups: Vec<HotkeyGroup>,
    /// Index of the first visible row (flat across groups).
    scroll: usize,
}

impl Default for Help {
    fn default() -> Self {
        Self {
            groups: default_groups(),
            scroll: 0,
        }
    }
}

impl Help {
    /// Flatten groups into `(is_header, keys, action)` display rows.
    fn flat_rows(&self) -> Vec<(bool, &str, &str)> {
        let mut out = Vec::new();
        for group in &self.groups {
            out.push((true, "", group.title.as_str()));
            for row in &group.rows {
                out.push((false, row.keys.as_str(), row.action.as_str()));
            }
        }
        out
    }

    /// Width of the keys column: the widest key list (TS `GitHelpSheet` rule).
    fn keys_width(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| g.rows.iter())
            .map(|r| r.keys.chars().count())
            .max()
            .unwrap_or(0)
            .max(3)
    }
}

impl Component for Help {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let popup = crate::overlays::centered(area, 60, 70);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(" keyboard shortcuts ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(popup);
        frame.render_widget(Clear, inner);
        frame.render_widget(block, popup);

        let keys_w = self.keys_width();
        let rows = self.flat_rows();
        let visible = inner.height as usize;
        let window: Vec<Line<'_>> = rows
            .iter()
            .skip(self.scroll)
            .take(visible.saturating_sub(1))
            .map(|(header, keys, action)| {
                if *header {
                    Line::from(Span::styled(
                        action.to_string(),
                        Style::default()
                            .fg(theme.color("accent"))
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(vec![
                        Span::styled(
                            format!("{keys:>keys_w$}  "),
                            Style::default().fg(theme.color("text")),
                        ),
                        Span::styled(
                            (*action).to_string(),
                            Style::default().fg(theme.color("text_muted")),
                        ),
                    ])
                }
            })
            .chain(std::iter::once(Line::from(Span::styled(
                "esc close · ? close · q close".to_string(),
                Style::default().fg(theme.color("text_muted")),
            ))))
            .collect();
        frame.render_widget(Paragraph::new(window).wrap(Wrap { trim: false }), inner);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        let rows = self.flat_rows().len();
        let page = 10usize;
        match key.code {
            // Close keys (`?`/q/esc/ctrl+c per the TS sheet): intentionally
            // ignored so they bubble to the host, which owns open/close
            // state. Esc/q/? are unmodified; `c` requires ctrl.
            KeyCode::Esc | KeyCode::Char('q' | '?') => EventOutcome::Ignored,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                EventOutcome::Ignored
            }
            KeyCode::Up => {
                self.scroll = self.scroll.saturating_sub(1);
                EventOutcome::Consumed
            }
            KeyCode::Down => {
                self.scroll = (self.scroll + 1).min(rows.saturating_sub(1));
                EventOutcome::Consumed
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(page);
                EventOutcome::Consumed
            }
            KeyCode::PageDown => {
                self.scroll = (self.scroll + page).min(rows.saturating_sub(1));
                EventOutcome::Consumed
            }
            _ => EventOutcome::Ignored,
        }
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

    fn render_help(help: &Help, width: u16, height: u16) -> Vec<String> {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| help.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..height)
            .map(|y| (0..width).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect()
    }

    #[test]
    fn renders_bordered_sheet_with_groups() {
        let rows = render_help(&Help::default(), 80, 24);
        let body = rows.join("\n");
        assert!(body.contains("keyboard shortcuts"), "title in {:?}", body);
        assert!(body.contains("General"), "group header");
        assert!(body.contains("enter"), "keys column");
        assert!(body.contains("send prompt"), "action column");
        assert!(body.contains("esc close"), "footer");
        // Bordered: border token color used on the frame (top row of the popup).
        let popup_top = rows
            .iter()
            .find(|r| r.contains('┌'))
            .cloned()
            .unwrap_or_default();
        assert!(!popup_top.is_empty(), "border drawn: {:?}", rows);
    }

    #[test]
    fn scrolling_moves_window() {
        let mut help = Help::default();
        let before = render_help(&help, 80, 12).join("\n");
        help.handle_key(KeyEvent::from(KeyCode::PageDown));
        let after = render_help(&help, 80, 12).join("\n");
        assert_ne!(before, after);
        help.handle_key(KeyEvent::from(KeyCode::PageUp));
        let back = render_help(&help, 80, 12).join("\n");
        assert_eq!(before, back);
    }

    #[test]
    fn arrows_scroll_and_other_keys_bubble() {
        let mut help = Help::default();
        assert!(help.handle_key(KeyEvent::from(KeyCode::Down)).consumed());
        assert_eq!(help.scroll, 1);
        assert!(!help
            .handle_key(KeyEvent::from(KeyCode::Char('x')))
            .consumed());
        assert!(!help.handle_key(KeyEvent::from(KeyCode::Esc)).consumed());
    }
}
