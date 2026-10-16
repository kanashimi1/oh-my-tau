//! Rendering for the editor — the port of `Editor.render`
//! (`components/editor.ts:1337`) and the composer chrome around it.
//!
//! - Band composer (`components/composer/band.ts`, the default style): no
//!   frame — the status line sits above as the "top band" and the rows carry
//!   the `╰─ ` prompt gutter (accent when the draft is a bash `!` command).
//! - Right-aligned composer hint on an empty composer (`composer-hints.ts`:
//!   `⇧⇥ to change thinking effort`), flush right with a 2-cell minimum gap.
//! - Soft wrap to the content width (`layout::layout`), internal scroll
//!   (`#scrollOffset`) with the cursor kept visible, page step scrolling
//!   (`#getPageScroll`).
//! - Cursor: inverse-video cell (styled cursor, `#getStyledInputCursor`).

use std::cell::Cell;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::component::Component;
use crate::editor::layout::{display_width, layout, LayoutLine};
use crate::editor::{Editor, EditorOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

/// Prompt gutter of the band composer (`bandComposerStyle.defaultPromptGutter`).
pub const PROMPT_GUTTER: &str = "╰─ ";

/// Minimum free cells between the end-of-line cursor and the right hint
/// (`PLACEHOLDER_MIN_GAP`).
const HINT_MIN_GAP: usize = 2;

/// First visual row shown in the viewport (`#scrollOffset`).
pub(crate) type ScrollOffset = Cell<usize>;

impl Editor {
    /// Whether the draft is a bash-mode `!`/`!!` command (TS shell sigils,
    /// `SHELL_SIGIL_RE`): first line starts with `!`.
    #[must_use]
    pub(crate) fn is_bash_mode(&self) -> bool {
        self.buffer
            .lines()
            .first()
            .is_some_and(|l| l.starts_with('!'))
    }

    /// Scroll up one page (`pageUp`, viewport-only).
    pub(crate) fn scroll_page_up(&mut self) {
        let step = self.page_scroll_step.get().max(1);
        let cur = self.scroll_offset.get();
        self.scroll_offset.set(cur.saturating_sub(step));
    }

    /// Scroll down one page (`pageDown`).
    pub(crate) fn scroll_page_down(&mut self) {
        let step = self.page_scroll_step.get().max(1);
        let cur = self.scroll_offset.get();
        self.scroll_offset.set(cur + step);
    }

    /// The height the editor wants for `width` columns: wrapped content
    /// rows (the band composer has no vertical chrome), capped at
    /// [`Editor::max_height`] minus the band row (TS `setMaxHeight`).
    #[must_use]
    pub fn desired_height(&self, width: u16) -> u16 {
        let content_width = width
            .saturating_sub(display_width(PROMPT_GUTTER) as u16)
            .max(1) as usize;
        let rows = if self.buffer.is_empty() {
            1
        } else {
            layout(self.buffer.lines(), content_width).len().max(1)
        };
        u16::try_from(rows.min(self.max_height.saturating_sub(1).max(2) as usize)).unwrap_or(u16::MAX)
    }

    /// Keep the scroll offset showing the cursor (called from render).
    fn sync_scroll(&self, rows: &[LayoutLine], content_height: usize) {
        let Some(cursor_row) = rows.iter().position(|r| {
            r.line == self.buffer.cursor().line
                && self.buffer.cursor().col >= r.start
                && self.buffer.cursor().col <= r.start + r.text.len()
        }) else {
            return;
        };
        let mut offset = self.scroll_offset.get();
        if cursor_row < offset {
            offset = cursor_row;
        } else if content_height > 0 && cursor_row >= offset + content_height {
            offset = cursor_row + 1 - content_height;
        }
        if offset >= rows.len() {
            offset = rows.len().saturating_sub(1);
        }
        self.scroll_offset.set(offset);
        self.page_scroll_step.set(content_height);
    }
}

impl Component for Editor {
    fn render(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height < 1 || area.width < 4 {
            return;
        }
        let inner = area;
        let content_height = area.height as usize;
        let gutter_color = if self.is_bash_mode() {
            theme.color("accent")
        } else {
            theme.color("border")
        };
        let gutter_style = Style::default().fg(gutter_color);
        let gutter_w = display_width(PROMPT_GUTTER);

        // Empty composer: gutter, cursor, and the right-aligned hint.
        if self.buffer.is_empty() {
            let mut spans = vec![Span::styled(PROMPT_GUTTER, gutter_style)];
            if let Some((key, label)) = &self.hint {
                let hint_w = display_width(key) + 1 + display_width(label);
                let avail = (inner.width as usize).saturating_sub(gutter_w);
                // The cursor cell sits at the start of the free area.
                if avail >= hint_w + HINT_MIN_GAP {
                    spans.push(Span::raw(" ".repeat(avail - hint_w)));
                    spans.push(Span::styled(key.clone(), Style::default().fg(theme.color("accent"))));
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled(
                        label.clone(),
                        Style::default()
                            .fg(theme.color("text_muted"))
                            .add_modifier(Modifier::ITALIC),
                    ));
                }
            }
            frame.render_widget(Paragraph::new(Line::from(spans)), inner);
            let cursor_area = Rect {
                x: inner.x + u16::try_from(gutter_w).unwrap_or(0),
                width: inner.width.saturating_sub(u16::try_from(gutter_w).unwrap_or(0)),
                ..inner
            };
            Self::draw_cursor(frame.buffer_mut(), cursor_area, "", 0, 0, theme);
            return;
        }

        // Every content row reserves the prompt-gutter width: the `╰─ ` cue
        // on the first row, blank indent on the rest — wrap width and
        // cursor column both account for it.
        let rows = layout(
            self.buffer.lines(),
            (inner.width as usize).saturating_sub(gutter_w).max(1),
        );
        self.sync_scroll(&rows, content_height);
        let offset = self.scroll_offset.get();
        let visible: Vec<Line<'_>> = rows
            .iter()
            .enumerate()
            .skip(offset)
            .take(content_height)
            .enumerate()
            .map(|(vis, (_, r))| {
                let lead: Span<'_> = if vis == 0 {
                    Span::styled(PROMPT_GUTTER, gutter_style)
                } else {
                    Span::raw(" ".repeat(gutter_w))
                };
                Line::from(vec![lead, Span::raw(r.text.as_str())])
            })
            .collect();
        frame.render_widget(Paragraph::new(visible), inner);

        // Cursor: find its visual row/column and invert the cell.
        let cursor = self.buffer.cursor();
        let mut vis_row = 0usize;
        let mut col_in_row = 0usize;
        for (i, r) in rows.iter().enumerate() {
            if r.line == cursor.line
                && cursor.col >= r.start
                && cursor.col <= r.start + r.text.len()
            {
                vis_row = i;
                col_in_row = cursor.col - r.start;
                break;
            }
        }
        let before = rows
            .get(vis_row)
            .map_or("", |r| &r.text[..col_in_row.min(r.text.len())]);
        let cursor_area = Rect {
            x: inner.x + u16::try_from(gutter_w).unwrap_or(0),
            width: inner.width.saturating_sub(u16::try_from(gutter_w).unwrap_or(0)),
            ..inner
        };
        Self::draw_cursor(
            frame.buffer_mut(),
            cursor_area,
            before,
            vis_row.saturating_sub(offset),
            col_in_row,
            theme,
        );
    }

    fn handle_key(&mut self, key: KeyEvent) -> crate::component::EventOutcome {
        use crate::component::EventOutcome;
        // App-owned keys must bubble: escape (interrupt), alt/shift+up/down
        // (dequeue), ctrl+d on an empty buffer (exit).
        let mods = key.modifiers;
        let app_key = match key.code {
            KeyCode::Esc => true,
            KeyCode::Up | KeyCode::Down if mods.contains(KeyModifiers::ALT) => true,
            KeyCode::Up | KeyCode::Down if mods.contains(KeyModifiers::SHIFT) => true,
            KeyCode::Char('d') if mods.contains(KeyModifiers::CONTROL) && self.is_empty() => true,
            _ => false,
        };
        if app_key {
            return EventOutcome::Ignored;
        }
        match self.handle_key(key) {
            EditorOutcome::Edited => EventOutcome::Consumed,
            // The app owns submit/interrupt semantics.
            EditorOutcome::Submit | EditorOutcome::Interrupt => EventOutcome::Ignored,
        }
    }

    fn wants_focus(&self) -> bool {
        true
    }
}

impl Editor {
    /// Invert the cursor cell (styled cursor, `#getStyledInputCursor`).
    fn draw_cursor(
        buf: &mut Buffer,
        inner: Rect,
        before_cursor: &str,
        vis_row: usize,
        _col_in_row: usize,
        theme: &Theme,
    ) {
        if vis_row >= inner.height as usize {
            return;
        }
        let cell_w = u16::try_from(display_width(before_cursor))
            .unwrap_or(inner.width)
            .min(inner.width.saturating_sub(1));
        let x = inner.x + cell_w;
        let y = inner.y + u16::try_from(vis_row).unwrap_or(inner.height);
        let cell = &mut buf[(x, y)];
        cell.set_style(
            Style::default()
                .fg(theme.color("background"))
                .bg(theme.color("text"))
                .add_modifier(Modifier::REVERSED),
        );
        if cell.symbol().is_empty() {
            cell.set_symbol(" ");
        }
    }
}
