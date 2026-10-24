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
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::component::Component;
use crate::editor::layout::{display_width, layout, LayoutLine};
use crate::editor::{Editor, EditorOutcome};
use crate::keys::{KeyCode, KeyEvent, KeyModifiers};
use crate::theme::Theme;

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

    /// Geometry of the active composer shape for `width` columns:
    /// `(padding_x, side chrome per side, prompt gutter)`.
    fn shape_geometry(&self) -> (usize, usize, Option<&'static str>) {
        let shape = self.shape;
        let padding_x = shape.default_padding_x();
        let chrome = shape.side_chrome_width(padding_x);
        // The box frame never renders a gutter (TS `#getEffectivePromptGutter`).
        let gutter = if shape.side_borders() {
            None
        } else {
            shape.default_prompt_gutter()
        };
        (padding_x, chrome, gutter)
    }

    /// Width available to wrapped text for a composer `width` columns wide.
    fn text_width(&self, width: usize) -> usize {
        let (_, chrome, gutter) = self.shape_geometry();
        let gutter_w = gutter
            .map_or(0, display_width)
            .min(width.saturating_sub(chrome * 2));
        width.saturating_sub(chrome * 2 + gutter_w).max(1)
    }

    /// Chrome rows the shape draws around the content (box merges its bottom
    /// border into the last content row).
    fn chrome_rows(&self) -> u16 {
        u16::from(self.shape.has_top()) + u16::from(self.shape.has_bottom())
    }

    /// The height the editor wants for `width` columns: wrapped content rows
    /// capped at [`Editor::max_height`] minus the shape's vertical chrome
    /// (TS `setMaxHeight` / `#getVisibleContentHeight`), plus the chrome rows.
    #[must_use]
    pub fn desired_height(&self, width: u16) -> u16 {
        let content_width = self.text_width(usize::from(width));
        let rows = if self.buffer.is_empty() {
            1
        } else {
            layout(self.buffer.lines(), content_width).len().max(1)
        };
        let max_content = usize::from(
            self.max_height
                .saturating_sub(self.shape.vertical_chrome())
                .max(1),
        );
        u16::try_from(rows.min(max_content)).unwrap_or(u16::MAX) + self.chrome_rows()
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
        let shape = self.shape;
        let (padding_x, chrome_w, gutter) = self.shape_geometry();
        let gutter_color = if self.is_bash_mode() {
            theme.color("accent")
        } else {
            theme.color("border")
        };
        let gutter_style = Style::default().fg(gutter_color);
        let width = usize::from(area.width);
        let gutter_w = gutter
            .map_or(0, display_width)
            .min(width.saturating_sub(chrome_w * 2));
        let text_w = width.saturating_sub(chrome_w * 2 + gutter_w).max(1);
        let (accent, surface) = crate::composer::chrome_styles(theme, gutter_style);
        let ctx = crate::composer::Chrome {
            width,
            padding_x,
            border: gutter_style,
            accent,
            surface,
            top: self.top_border.clone(),
        };

        // Top chrome row.
        let mut y = area.y;
        if let Some(top) = shape.render_top(&ctx) {
            frame.render_widget(
                Paragraph::new(top),
                Rect {
                    y,
                    height: 1,
                    ..area
                },
            );
            y += 1;
        }
        let bottom_rows = u16::from(shape.has_bottom());
        let content_height = (area.y + area.height)
            .saturating_sub(y + bottom_rows)
            .max(1) as usize;
        let content_top = y;

        // Visual rows (an empty draft shows one row carrying the hint).
        let empty = self.buffer.is_empty();
        let rows = if empty {
            Vec::new()
        } else {
            layout(self.buffer.lines(), text_w)
        };
        if !empty {
            self.sync_scroll(&rows, content_height);
        }
        let offset = if empty { 0 } else { self.scroll_offset.get() };
        let visible_count = if empty {
            1
        } else {
            rows.len().saturating_sub(offset).min(content_height)
        };
        // Magic keywords: ranges over the whole draft (prose masking needs
        // the full text), painted per visual row by global byte offset.
        let magic_hits =
            if !empty && crate::magic_paint::any_keyword(self.buffer.lines(), theme.magic_off) {
                crate::magic::enabled_ranges(&self.buffer.lines().join("\n"), theme.magic_off)
            } else {
                Vec::new()
            };
        let line_offsets: Vec<usize> = if magic_hits.is_empty() {
            Vec::new()
        } else {
            self.buffer
                .lines()
                .iter()
                .scan(0usize, |off, l| {
                    let start = *off;
                    *off += l.len() + 1;
                    Some(start)
                })
                .collect()
        };
        let gutter_first = gutter.map(|g| Span::styled(g.to_owned(), gutter_style));
        let gutter_rest = gutter.map(|_| Span::raw(" ".repeat(gutter_w)));
        for vis in 0..visible_count {
            let (text, text_cells) = if empty {
                let mut spans = Vec::new();
                let mut used = 0usize;
                if let Some((key, label)) = &self.hint {
                    let hint_w = display_width(key) + 1 + display_width(label);
                    // The cursor cell sits at the start of the free area.
                    if text_w >= hint_w + HINT_MIN_GAP {
                        spans.push(Span::raw(" ".repeat(text_w - hint_w)));
                        spans.push(Span::styled(
                            key.clone(),
                            Style::default().fg(theme.color("accent")),
                        ));
                        spans.push(Span::raw(" "));
                        spans.push(Span::styled(
                            label.clone(),
                            Style::default()
                                .fg(theme.color("text_muted"))
                                .add_modifier(Modifier::ITALIC),
                        ));
                        used = text_w;
                    }
                }
                (spans, used)
            } else {
                let row = &rows[offset + vis];
                let cells = display_width(&row.text);
                let spans = match line_offsets.get(row.line) {
                    Some(line_off) => crate::magic_paint::paint_segment(
                        &row.text,
                        line_off + row.start,
                        &magic_hits,
                        Style::default(),
                        self.magic_phase,
                    ),
                    None => vec![Span::raw(row.text.clone())],
                };
                (spans, cells)
            };
            let row = crate::composer::Row {
                text,
                pad: text_w.saturating_sub(text_cells),
                gutter: if vis == 0 {
                    gutter_first.clone()
                } else {
                    gutter_rest.clone()
                },
                is_last: vis + 1 == visible_count,
            };
            let line = shape.render_row(&ctx, &row);
            frame.render_widget(
                Paragraph::new(line),
                Rect {
                    y: content_top + vis as u16,
                    height: 1,
                    ..area
                },
            );
        }

        // Bottom chrome row (rules); the box merges its border into the last
        // content row.
        if let Some(bottom) = shape.render_bottom(&ctx) {
            let by = content_top + visible_count as u16;
            if by < area.y + area.height {
                frame.render_widget(
                    Paragraph::new(bottom),
                    Rect {
                        y: by,
                        height: 1,
                        ..area
                    },
                );
            }
        }

        // Cursor: find its visual row/column and invert the cell.
        let cursor_area = Rect {
            x: area.x + u16::try_from(chrome_w + gutter_w).unwrap_or(0),
            y: content_top,
            width: u16::try_from(text_w).unwrap_or(area.width),
            height: u16::try_from(visible_count).unwrap_or(1),
        };
        if empty {
            Self::draw_cursor(frame.buffer_mut(), cursor_area, "", 0, 0, theme);
            return;
        }
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
        if let Some(ghost) = self.word_ghost() {
            let at_row_end = rows
                .get(vis_row)
                .is_some_and(|r| col_in_row == r.text.len());
            if at_row_end && vis_row >= offset && vis_row - offset < visible_count {
                Self::draw_ghost(
                    frame.buffer_mut(),
                    cursor_area,
                    vis_row - offset,
                    display_width(before),
                    &ghost,
                    theme,
                );
            }
        }
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
    /// Dim ghost text starting at the end-of-line cursor cell (TS `hintTail`
    /// follows the zero-width cursor marker, so the cursor sits on the ghost's
    /// first character), truncated to the free cells of the row. Call before
    /// [`Self::draw_cursor`], which restyles the first cell.
    fn draw_ghost(
        buf: &mut Buffer,
        inner: Rect,
        vis_row: usize,
        text_cells: usize,
        ghost: &str,
        theme: &Theme,
    ) {
        let y = inner.y + u16::try_from(vis_row).unwrap_or(0);
        let style = Style::default()
            .fg(theme.color("text_muted"))
            .add_modifier(Modifier::DIM);
        let mut x = inner.x + u16::try_from(text_cells).unwrap_or(inner.width);
        let end = inner.x + inner.width;
        for c in ghost.chars() {
            let w =
                u16::try_from(unicode_width::UnicodeWidthChar::width(c).unwrap_or(0)).unwrap_or(0);
            if w == 0 {
                continue;
            }
            if x + w > end {
                break;
            }
            let cell = &mut buf[(x, y)];
            cell.set_symbol(c.encode_utf8(&mut [0; 4]));
            cell.set_style(style);
            x += w;
        }
    }

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
