//! Command palette overlay — port of the fuzzy-filtered picker pattern used
//! across omp overlays (`SelectList` + `fuzzy.ts` word-local scoring; the
//! TS host is the Ctrl/Cmd+K palette in the app keybindings).
//!
//! A top input line filters the candidate list with
//! [`crate::autocomplete::fuzzy_score`]; up/down move the selection,
//! enter/space confirm, esc cancels. Confirmation is observed by the host
//! via [`CommandPalette::take_picked`] (the `Component` trait has no action
//! channel; the app pops the palette and dispatches the command string).

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// One palette entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteCommand {
    /// Display label, e.g. `/model`.
    pub label: String,
    /// Optional dim description shown after the label.
    pub hint: String,
}

impl PaletteCommand {
    /// A label-only entry.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            hint: String::new(),
        }
    }
}

/// The command palette overlay.
#[derive(Debug, Clone, Default)]
pub struct CommandPalette {
    /// Current filter text (editable).
    pub query: String,
    /// Full candidate list (set by `available_commands_update`-style frames).
    pub commands: Vec<PaletteCommand>,
    /// Index into the *filtered* list.
    pub selected: usize,
    /// Set on enter; drained by the host with [`Self::take_picked`].
    picked: Option<String>,
}

impl CommandPalette {
    /// Replace the candidate list, keeping the selection in range.
    pub fn set_commands(&mut self, commands: Vec<PaletteCommand>) {
        self.commands = commands;
        self.clamp_selection();
    }

    /// Indices of candidates matching `query`, best fuzzy score first.
    /// An empty query keeps declaration order (TS `SelectList` behavior).
    #[must_use]
    pub fn filtered(&self) -> Vec<usize> {
        if self.query.is_empty() {
            return (0..self.commands.len()).collect();
        }
        let mut ranked: Vec<(usize, i64)> = self
            .commands
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let score = crate::autocomplete::fuzzy_score(&self.query, &c.label)
                    + crate::autocomplete::fuzzy_score(&self.query, &c.hint) / 2;
                (i, score)
            })
            .filter(|(_, s)| *s > 0)
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.into_iter().map(|(i, _)| i).collect()
    }

    /// Drain the confirmed command label, if enter was pressed.
    #[must_use]
    pub fn take_picked(&mut self) -> Option<String> {
        self.picked.take()
    }

    fn clamp_selection(&mut self) {
        let len = self.filtered().len();
        if len == 0 {
            self.selected = 0;
        } else {
            self.selected = self.selected.min(len - 1);
        }
    }
}

impl Component for CommandPalette {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let popup = super::centered(area, 60, 30);
        frame.render_widget(Clear, popup);
        let block = Block::default()
            .title(" command ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);

        // Input row (the live query).
        let input = Paragraph::new(Line::from(vec![
            Span::styled("❯ ", Style::default().fg(theme.color("accent"))),
            Span::raw(self.query.clone()),
        ]));
        let input_area = Rect { height: 1, ..inner };
        frame.render_widget(input, input_area);

        // Filtered candidates below.
        let list_area = Rect {
            y: inner.y + 1,
            height: inner.height.saturating_sub(1),
            ..inner
        };
        let matches = self.filtered();
        let rows: Vec<Line<'_>> = matches
            .iter()
            .take(list_area.height as usize)
            .enumerate()
            .map(|(row, &idx)| {
                let cmd = &self.commands[idx];
                let selected = row == self.selected;
                let style = if selected {
                    Style::default()
                        .fg(theme.color("selected"))
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.color("text"))
                };
                let mut spans = vec![Span::styled(format!("{} ", theme.glyph("prompt")), style)];
                spans.push(Span::styled(cmd.label.clone(), style));
                if !cmd.hint.is_empty() {
                    spans.push(Span::styled(
                        format!(" — {}", cmd.hint),
                        Style::default().fg(theme.color("text_muted")),
                    ));
                }
                Line::from(spans)
            })
            .collect();
        if rows.is_empty() {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "no matching commands",
                    Style::default().fg(theme.color("text_muted")),
                )),
                list_area,
            );
        } else {
            frame.render_widget(Paragraph::new(rows), list_area);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        match key.code {
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                EventOutcome::Consumed
            }
            KeyCode::Down => {
                let len = self.filtered().len();
                if len > 0 {
                    self.selected = (self.selected + 1).min(len - 1);
                }
                EventOutcome::Consumed
            }
            KeyCode::Enter => {
                if let Some(&idx) = self.filtered().get(self.selected) {
                    self.picked = Some(self.commands[idx].label.clone());
                }
                EventOutcome::Consumed
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.clamp_selection();
                EventOutcome::Consumed
            }
            KeyCode::Char(c) => {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    return EventOutcome::Ignored;
                }
                self.query.push(c);
                self.selected = 0;
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

    pub(crate) fn palette() -> CommandPalette {
        let mut p = CommandPalette::default();
        p.set_commands(vec![
            PaletteCommand::new("/model"),
            PaletteCommand::new("/help"),
            PaletteCommand::new("/exit"),
        ]);
        p
    }

    #[test]
    fn empty_query_lists_all_in_order() {
        let p = palette();
        assert_eq!(p.filtered(), vec![0, 1, 2]);
    }

    #[test]
    fn fuzzy_filter_ranks_prefix_higher() {
        let mut p = palette();
        p.query = "mod".into();
        assert_eq!(p.filtered(), vec![0]);
        p.query = "e".into();
        let hits = p.filtered();
        assert!(hits.contains(&1)); // /help
        assert!(hits.contains(&2)); // /exit
    }

    #[test]
    fn enter_picks_selected_label() {
        let mut p = palette();
        p.selected = 2;
        let outcome = p.handle_key(KeyEvent::from(KeyCode::Enter));
        assert!(outcome.consumed());
        assert_eq!(p.take_picked().as_deref(), Some("/exit"));
        assert_eq!(p.take_picked(), None);
    }

    #[test]
    fn typing_updates_query_and_resets_selection() {
        let mut p = palette();
        p.selected = 2;
        p.handle_key(KeyEvent::from(KeyCode::Char('m')));
        assert_eq!(p.query, "m");
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn backspace_edits_query() {
        let mut p = palette();
        p.handle_key(KeyEvent::from(KeyCode::Char('m')));
        p.handle_key(KeyEvent::from(KeyCode::Backspace));
        assert!(p.query.is_empty());
    }
}

#[cfg(test)]
mod render_tests {
    use super::tests::palette;
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_palette(p: &CommandPalette, width: u16, height: u16) -> Vec<String> {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| p.render(f, f.area(), &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..height)
            .map(|y| (0..width).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect()
    }

    #[test]
    fn renders_input_and_candidates() {
        let p = palette();
        let rows = render_palette(&p, 80, 20);
        let body = rows.join("\n");
        assert!(body.contains("command"), "title in {:?}", body);
        assert!(body.contains("❯"), "input row");
        assert!(body.contains("/model"));
        assert!(body.contains("/exit"));
    }

    #[test]
    fn empty_filter_shows_no_matches_line() {
        let mut p = palette();
        p.query = "zzz".into();
        let body = render_palette(&p, 80, 20).join("\n");
        assert!(body.contains("no matching commands"));
    }
}
