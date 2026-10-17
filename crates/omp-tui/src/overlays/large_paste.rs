//! Large-paste menu — port of the TS large-paste flow
//! (`input-controller.ts` `handleLargePaste` / `presentLargePasteMenu`, the
//! `paste.largeMenuThreshold` gate in `modes/settings.ts:1041`).
//!
//! When a bracketed paste reaches the configured line threshold, the host
//! opens this panel instead of collapsing immediately; the choice comes back
//! through [`LargePasteMenu::take_pick`] and the host applies it to the
//! editor draft (wrap in `<attachment>` tags, save to a file, paste inline,
//! or the Esc fallback — stage as a text-attachment chip).

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::component::{Component, EventOutcome};
use crate::keys::{KeyCode, KeyEvent};
use crate::theme::Theme;

/// What the user chose (TS `presentLargePasteMenu` switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LargePastePick {
    /// Wrap the text in `<attachment>` tags, collapsed to a text chip.
    WrappedBlock,
    /// Save the text to a file and reference its path.
    LocalFile,
    /// Collapse the text to an inline `[Paste]` marker.
    Inline,
    /// Esc / cancel — TS fallback: stage as a text-attachment chip.
    Cancelled,
}

/// Menu rows in TS order: `(label, description)`.
const ITEMS: &[(LargePastePick, &str, &str)] = &[
    (
        LargePastePick::WrappedBlock,
        "Attach as a wrapped block",
        "Wrap the text in <attachment> tags, collapsed to a marker",
    ),
    (
        LargePastePick::LocalFile,
        "Attach as local file",
        "Save the text to a file and reference its path",
    ),
    (
        LargePastePick::Inline,
        "Paste inline",
        "Collapse the text to an inline paste marker",
    ),
];

/// The large-paste selector panel.
#[derive(Debug, Clone, Default)]
pub struct LargePasteMenu {
    /// Header, e.g. `Pasted 128 lines` (TS `showHookSelector` title).
    title: String,
    selected: usize,
    picked: Option<LargePastePick>,
}

impl LargePasteMenu {
    /// Open the menu for a paste of `line_count` lines.
    pub fn open(&mut self, line_count: usize) {
        self.title = format!("Pasted {line_count} lines");
        self.selected = 0;
        self.picked = None;
    }

    /// The user's choice, once.
    pub fn take_pick(&mut self) -> Option<LargePastePick> {
        self.picked.take()
    }

    /// Panel height (rows + borders).
    #[must_use]
    pub fn height(&self) -> u16 {
        u16::try_from(ITEMS.len() + 3).unwrap_or(u16::MAX)
    }
}

impl Component for LargePasteMenu {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let height = self.height().min(area.height);
        if height < 3 || area.width < 30 {
            return;
        }
        // Anchored to the bottom like the thinking selector (TS mounts the
        // selector over the composer area).
        let panel = Rect {
            x: area.x,
            y: area.y + area.height - height,
            width: area.width,
            height,
        };
        frame.render_widget(Clear, panel);
        let block = Block::default()
            .title(format!("─ {} ", self.title))
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(theme.color("border")));
        let inner = block.inner(panel);
        frame.render_widget(block, panel);
        let accent = Style::default().fg(theme.color("accent"));
        let text = Style::default().fg(theme.color("text"));
        let dim = Style::default().fg(theme.color("dim"));
        let mut lines: Vec<Line<'_>> = Vec::new();
        for (i, (pick, label, desc)) in ITEMS.iter().enumerate() {
            let current = i == self.selected;
            let prefix = if current { "❯ " } else { "  " };
            let style = if current { accent } else { text };
            lines.push(Line::from(vec![
                Span::styled(format!(" {prefix}"), style),
                Span::styled((*label).to_owned(), style),
            ]));
            let _ = pick;
            lines.push(Line::from(Span::styled(
                format!("    {desc}"),
                if current { accent } else { dim },
            )));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn handle_key(&mut self, key: KeyEvent) -> EventOutcome {
        match key.code {
            KeyCode::Up => self.selected = (self.selected + ITEMS.len() - 1) % ITEMS.len(),
            KeyCode::Down => self.selected = (self.selected + 1) % ITEMS.len(),
            KeyCode::Enter => {
                self.picked = ITEMS.get(self.selected).map(|(pick, _, _)| *pick);
            }
            // Esc bubbles so the host pops the panel, then applies the
            // cancel fallback (TS: "esc to paste inline").
            KeyCode::Esc => {
                self.picked = Some(LargePastePick::Cancelled);
                return EventOutcome::Ignored;
            }
            _ => {}
        }
        EventOutcome::Consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{KeyCode, KeyEvent};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, crate::keys::KeyModifiers::NONE)
    }

    #[test]
    fn navigation_and_pick() {
        let mut m = LargePasteMenu::default();
        m.open(120);
        m.handle_key(key(KeyCode::Down));
        m.handle_key(key(KeyCode::Enter));
        assert_eq!(m.take_pick(), Some(LargePastePick::LocalFile));
        assert_eq!(m.take_pick(), None);
    }

    #[test]
    fn enter_defaults_to_wrapped_block() {
        let mut m = LargePasteMenu::default();
        m.open(5);
        m.handle_key(key(KeyCode::Enter));
        assert_eq!(m.take_pick(), Some(LargePastePick::WrappedBlock));
    }

    #[test]
    fn up_wraps_around_to_last_item() {
        let mut m = LargePasteMenu::default();
        m.open(1);
        m.handle_key(key(KeyCode::Up));
        m.handle_key(key(KeyCode::Enter));
        assert_eq!(m.take_pick(), Some(LargePastePick::Inline));
    }

    #[test]
    fn esc_signals_cancel_but_bubbles() {
        let mut m = LargePasteMenu::default();
        m.open(3);
        assert!(!m.handle_key(key(KeyCode::Esc)).consumed());
        assert_eq!(m.take_pick(), Some(LargePastePick::Cancelled));
    }
}
