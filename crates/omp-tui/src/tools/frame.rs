//! Framed tool card — port of `pi-tui/src/render/output-block.ts`
//! (`renderOutputBlock`) as used by `framedToolCard`.
//!
//! Layout (width `W`): a top bar `╭─── header ───╮`, labelled section bars
//! `├─── Label ───┤`, content rows `│ text │` (one column of padding each
//! side, rows word-wrapped to `W - 4`), and a closing `╰───╯`.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

use crate::theme::Theme;

/// One body section of a framed card.
pub(crate) struct CardSection {
    /// Titled separator bar drawn above the section (`├─── Output ───┤`).
    pub label: Option<Line<'static>>,
    /// Content rows (wrapped to the inner width).
    pub lines: Vec<Line<'static>>,
}

impl CardSection {
    /// Unlabelled section.
    pub fn plain(lines: Vec<Line<'static>>) -> Self {
        Self { label: None, lines }
    }

    /// Section under a bold `toolTitle` label.
    pub fn labelled(theme: &Theme, label: &str, lines: Vec<Line<'static>>) -> Self {
        Self {
            label: Some(Line::from(Span::styled(
                label.to_owned(),
                Style::default().fg(token(theme, "toolTitle", "accent")),
            ))),
            lines,
        }
    }
}

/// Theme color with a fallback token (TS tokens unknown to a custom theme
/// resolve to [`Color::Reset`]).
pub(crate) fn token(theme: &Theme, name: &str, fallback: &str) -> Color {
    let c = theme.color(name);
    if c == Color::Reset {
        theme.color(fallback)
    } else {
        c
    }
}

/// Language glyph for a file path (TS `theme.getLangIcon(getLanguageFromPath(path))`).
pub(crate) fn lang_icon(theme: &Theme, path: &str) -> &'static str {
    let lang = crate::chat::code::lang_from_path(path).unwrap_or("default");
    let icon = theme.symbol(&format!("lang.{lang}"));
    if icon.is_empty() {
        theme.symbol("lang.default")
    } else {
        icon
    }
}

/// `renderStatusLine`: `icon title: description meta · meta`.
///
/// `description` spans are muted by the caller's own styles; `meta` entries
/// join with the dim ` · ` separator after one space.
pub(crate) fn status_header(
    theme: &Theme,
    icon: Option<Span<'static>>,
    title: &str,
    title_token: &str,
    description: Vec<Span<'static>>,
    meta: Vec<Span<'static>>,
) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    if let Some(icon) = icon {
        spans.push(icon);
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled(
        title.to_owned(),
        Style::default().fg(token(theme, title_token, "accent")),
    ));
    if !description.is_empty() {
        spans.push(Span::raw(": "));
        spans.extend(description);
    }
    if !meta.is_empty() {
        spans.push(Span::raw(" "));
        let sep = Span::styled(
            theme.symbol("sep.dot").to_owned(),
            Style::default().fg(token(theme, "dim", "text_muted")),
        );
        for (i, m) in meta.into_iter().enumerate() {
            if i > 0 {
                spans.push(sep.clone());
            }
            spans.push(m);
        }
    }
    Line::from(spans)
}

fn cell_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

/// Visible width of a line.
pub(crate) fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|s| s.content.chars().map(cell_width).sum::<usize>())
        .sum()
}

/// Cut a line to `max` display cells (no ellipsis).
pub(crate) fn clip_line(line: &Line<'static>, max: usize) -> Line<'static> {
    let mut used = 0;
    let mut spans: Vec<Span<'static>> = Vec::new();
    for span in &line.spans {
        let mut text = String::new();
        for ch in span.content.chars() {
            let w = cell_width(ch);
            if used + w > max {
                break;
            }
            used += w;
            text.push(ch);
        }
        let done = text.chars().count() < span.content.chars().count();
        if !text.is_empty() {
            spans.push(Span::styled(text, span.style));
        }
        if done {
            break;
        }
    }
    Line::from(spans)
}

/// Word-wrap one styled line to `width` cells.
fn wrap_line(line: &Line<'static>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    if line_width(line) <= width {
        return vec![line.clone()];
    }
    let cells: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|s| s.content.chars().map(move |c| (c, s.style)))
        .collect();
    let mut rows: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_w = 0;
    for (ch, style) in cells {
        let w = cell_width(ch);
        if cur_w + w > width {
            // Break at the last space when one exists in the row.
            if let Some(at) = cur.iter().rposition(|(c, _)| *c == ' ') {
                let rest = cur.split_off(at + 1);
                while cur.last().is_some_and(|(c, _)| *c == ' ') {
                    cur.pop();
                }
                rows.push(std::mem::take(&mut cur));
                cur = rest;
                cur_w = cur.iter().map(|(c, _)| cell_width(*c)).sum();
            } else {
                rows.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
        }
        cur.push((ch, style));
        cur_w += w;
    }
    rows.push(cur);
    rows.into_iter()
        .map(|row| {
            let mut spans: Vec<Span<'static>> = Vec::new();
            let mut text = String::new();
            let mut style: Option<Style> = None;
            for (ch, st) in row {
                if style != Some(st) {
                    if let Some(prev) = style {
                        spans.push(Span::styled(std::mem::take(&mut text), prev));
                    }
                    style = Some(st);
                }
                text.push(ch);
            }
            if let Some(prev) = style {
                spans.push(Span::styled(text, prev));
            }
            Line::from(spans)
        })
        .collect()
}

/// `╭─── label ───╮` style bar.
fn bar(
    left: &str,
    right: &str,
    label: Option<&Line<'static>>,
    width: usize,
    border: Style,
) -> Line<'static> {
    let left_glyphs = format!("{left}───");
    let left_w = 4;
    let mut spans = vec![Span::styled(left_glyphs, border)];
    let fill = |n: usize| Span::styled("─".repeat(n), border);
    match label.filter(|l| line_width(l) > 0) {
        None => {
            spans.push(fill(width.saturating_sub(left_w + 1)));
        }
        Some(label) => {
            let max_label = width.saturating_sub(left_w + 1);
            let mut padded = Line::from(
                std::iter::once(Span::raw(" "))
                    .chain(label.spans.iter().cloned())
                    .chain(std::iter::once(Span::raw(" ")))
                    .collect::<Vec<_>>(),
            );
            if line_width(&padded) > max_label {
                padded = clip_line(&padded, max_label);
            }
            let used = line_width(&padded);
            spans.extend(padded.spans);
            spans.push(fill(width.saturating_sub(left_w + used + 1)));
        }
    }
    spans.push(Span::styled(right.to_owned(), border));
    Line::from(spans)
}

/// Render a bordered card with one column of content padding per side.
/// `header` rides in the top bar.
pub(crate) fn framed_card(
    header: Option<Line<'static>>,
    sections: Vec<CardSection>,
    width: usize,
    border_color: Color,
) -> Vec<Line<'static>> {
    framed_card_padded(header, sections, width, border_color, 1)
}

/// [`framed_card`] with explicit left/right content padding (`0` = flush
/// borders, as the edit diff card draws them).
pub(crate) fn framed_card_padded(
    header: Option<Line<'static>>,
    sections: Vec<CardSection>,
    width: usize,
    border_color: Color,
    pad: usize,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    let border = Style::default().fg(border_color);
    let inner = width.saturating_sub(2 + 2 * pad).max(1);
    let gap = " ".repeat(pad);
    let mut out = vec![bar("╭", "╮", header.as_ref(), width, border)];
    for section in sections {
        if let Some(label) = &section.label {
            out.push(bar("├", "┤", Some(label), width, border));
        }
        for line in &section.lines {
            for row in wrap_line(line, inner) {
                let fill = inner.saturating_sub(line_width(&row));
                let mut spans = vec![Span::styled(format!("│{gap}"), border)];
                spans.extend(row.spans);
                if fill > 0 {
                    spans.push(Span::raw(" ".repeat(fill)));
                }
                spans.push(Span::styled(format!("{gap}│"), border));
                out.push(Line::from(spans));
            }
        }
    }
    out.push(Line::from(vec![
        Span::styled("╰───".to_owned(), border),
        Span::styled("─".repeat(width.saturating_sub(5)), border),
        Span::styled("╯".to_owned(), border),
    ]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn frames_header_labelled_sections() {
        let theme = Theme::default();
        let lines = framed_card(
            Some(Line::from("✎ Write: a.txt")),
            vec![
                CardSection::plain(vec![Line::from("$ echo hi")]),
                CardSection::labelled(&theme, "Output", vec![Line::from("hi")]),
            ],
            24,
            Color::Reset,
        );
        let t = text(&lines);
        assert_eq!(t[0], format!("╭─── ✎ Write: a.txt {}╮", "─".repeat(3)));
        assert_eq!(t[1], format!("│ $ echo hi{} │", " ".repeat(11)));
        assert_eq!(t[2], format!("├─── Output {}┤", "─".repeat(11)));
        assert_eq!(t[3], format!("│ hi{} │", " ".repeat(18)));
        assert_eq!(t[4], format!("╰{}╯", "─".repeat(22)));
    }

    #[test]
    fn long_rows_word_wrap_inside_the_frame() {
        let lines = framed_card(
            None,
            vec![CardSection::plain(vec![Line::from(
                "alpha beta gamma delta",
            )])],
            15,
            Color::Reset,
        );
        let t = text(&lines);
        assert_eq!(t[0], format!("╭{}╮", "─".repeat(13)));
        assert_eq!(t[1], "│ alpha beta  │");
        assert_eq!(t[2], "│ gamma delta │");
        assert!(t.iter().all(|r| r.chars().count() == 15), "{t:?}");
    }
}
