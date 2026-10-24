//! Clean-style ("rail") tool rows — the omp-rs presentation
//! ([`crate::theme::UiStyle::Clean`]).
//!
//! The tool families render their TS-parity output (boxed `╭───╮` cards or
//! `icon Title: description` status rows plus tree bodies); [`restyle`]
//! rewrites that output into the calm Claude Code / Codex layout:
//!
//! ```text
//! ● Write e2e-new.txt · 3 lines
//!   ⎿ 1 alpha
//!     2 beta
//! ```
//!
//! Hierarchy: the status dot carries the state (accent = running, success,
//! error), the title is the only bold text, and settled bodies are dimmed so
//! the eye lands on what is still moving. Collapsed (no ctrl+O) settled
//! bodies keep only their first rows behind a `… +N lines` trailer.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

use crate::theme::Theme;

/// Rail status of one tool row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailStatus {
    /// Waiting for / streaming a result.
    Running,
    /// Settled successfully.
    Ok,
    /// Settled with an error.
    Error,
}

/// Body elbow under the header (Claude Code `⎿`).
pub const ELBOW: &str = "  ⎿ ";
/// Continuation indent for body rows after the first.
pub const BODY_INDENT: &str = "    ";

/// Body rows kept for a collapsed settled card.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Rows shown when collapsed and settled.
    pub collapsed: usize,
}

impl Budget {
    /// Budget for a tool family: diffs keep more context than logs.
    #[must_use]
    pub fn for_tool(name: &str) -> Self {
        let collapsed = match name {
            "edit" | "apply_patch" => 14,
            "write" => 0,
            "todo" | "todo_write" | "ask" | "goal" | "task" => 12,
            "bash" | "eval" => 4,
            _ => 3,
        };
        Self { collapsed }
    }
}

fn cell_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn text_of(line: &Line<'_>) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

fn is_box_glyph(c: char) -> bool {
    matches!(c, '╭' | '╮' | '╰' | '╯' | '─' | '├' | '┤' | '│')
}

/// Styled cells of a line.
fn cells(line: &Line<'static>) -> Vec<(char, Style)> {
    line.spans
        .iter()
        .flat_map(|s| {
            let style = line.style.patch(s.style);
            s.content.chars().map(move |c| (c, style))
        })
        .collect()
}

/// Fold styled cells back into spans.
fn spans_of(cells: &[(char, Style)]) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut text = String::new();
    let mut cur: Option<Style> = None;
    for &(c, st) in cells {
        if cur != Some(st) {
            if let Some(prev) = cur {
                spans.push(Span::styled(std::mem::take(&mut text), prev));
            }
            cur = Some(st);
        }
        text.push(c);
    }
    if let Some(prev) = cur {
        spans.push(Span::styled(text, prev));
    }
    spans
}

fn trim_cells(mut cells: Vec<(char, Style)>) -> Vec<(char, Style)> {
    while cells.last().is_some_and(|(c, _)| c.is_whitespace()) {
        cells.pop();
    }
    let lead = cells.iter().take_while(|(c, _)| c.is_whitespace()).count();
    cells.drain(..lead);
    cells
}

/// Label cells of a box bar (`╭─── label ───╮` / `├─── label ───┤`).
fn bar_label(line: &Line<'static>) -> Vec<(char, Style)> {
    let cells: Vec<(char, Style)> = cells(line);
    // Strip the leading corner + `───` and the trailing fill + corner.
    let start = cells
        .iter()
        .position(|(c, _)| !is_box_glyph(*c))
        .unwrap_or(cells.len());
    let end = cells
        .iter()
        .rposition(|(c, _)| !is_box_glyph(*c))
        .map_or(start, |i| i + 1);
    trim_cells(cells[start..end.max(start)].to_vec())
}

/// Content cells of a framed row (`│ text   │`).
fn row_content(line: &Line<'static>) -> Vec<(char, Style)> {
    let mut cells = cells(line);
    if cells.first().is_some_and(|(c, _)| *c == '│') {
        cells.remove(0);
        if cells.first().is_some_and(|(c, _)| *c == ' ') {
            cells.remove(0);
        }
    }
    while cells.last().is_some_and(|(c, _)| c.is_whitespace()) {
        cells.pop();
    }
    if cells.last().is_some_and(|(c, _)| *c == '│') {
        cells.pop();
    }
    while cells.last().is_some_and(|(c, _)| c.is_whitespace()) {
        cells.pop();
    }
    cells
}

/// A parsed family render: header cells and body rows.
struct Parsed {
    header: Vec<(char, Style)>,
    body: Vec<Vec<(char, Style)>>,
}

fn parse(lines: &[Line<'static>]) -> Parsed {
    let mut rows = lines.iter().skip_while(|l| text_of(l).trim().is_empty());
    let Some(first) = rows.next() else {
        return Parsed {
            header: Vec::new(),
            body: Vec::new(),
        };
    };
    let first_text = text_of(first);
    if first_text.starts_with('╭') {
        let header = bar_label(first);
        let mut body = Vec::new();
        for line in rows {
            let t = text_of(line);
            if t.starts_with('╰') {
                break;
            }
            if t.starts_with('├') {
                let label = bar_label(line);
                let label_text: String = label.iter().map(|(c, _)| *c).collect();
                // The `⎿` elbow already reads as "output"; other labels stay
                // as a dim sub-heading.
                if !label.is_empty() && label_text != "Output" {
                    body.push(label);
                }
                continue;
            }
            body.push(row_content(line));
        }
        return Parsed { header, body };
    }
    let header = trim_cells(cells(first));
    let body = rows
        .map(|l| {
            let mut c = cells(l);
            if c.first().is_some_and(|(ch, _)| *ch == ' ') {
                c.remove(0);
            }
            while c.last().is_some_and(|(ch, _)| ch.is_whitespace()) {
                c.pop();
            }
            c
        })
        .collect();
    Parsed { header, body }
}

/// Split a family header into `(title, description)` cells: drops the
/// leading status/tool icon and the `Title:` colon.
fn split_header(cells: &[(char, Style)]) -> (Vec<(char, Style)>, Vec<(char, Style)>) {
    let mut cells: Vec<(char, Style)> = cells.to_vec();
    // Leading icon: one non-alphanumeric glyph (or `$`) followed by a space.
    let icon_len = cells.iter().take_while(|(c, _)| *c != ' ').count();
    let icon_is_glyph = icon_len > 0
        && icon_len <= 2
        && cells[..icon_len].iter().all(|(c, _)| !c.is_alphanumeric());
    if icon_is_glyph && cells.get(icon_len).is_some_and(|(c, _)| *c == ' ') {
        cells.drain(..=icon_len);
    }
    // Title: leading word run up to `:` (within the first 32 cells).
    let colon = cells
        .iter()
        .take(32)
        .position(|(c, _)| *c == ':')
        .filter(|&i| i > 0 && cells.get(i + 1).is_none_or(|(c, _)| *c == ' '));
    match colon {
        Some(i) => {
            let title = cells[..i].to_vec();
            let desc = trim_cells(cells[i + 1..].to_vec());
            (title, desc)
        }
        None => {
            let end = cells
                .iter()
                .position(|(c, _)| *c == ' ')
                .unwrap_or(cells.len());
            let title = cells[..end].to_vec();
            let desc = trim_cells(cells[end..].to_vec());
            (title, desc)
        }
    }
}

fn dot_color(status: RailStatus, theme: &Theme) -> Color {
    let pick = |a: &str, b: &str| {
        let c = theme.color(a);
        if c == Color::Reset {
            theme.color(b)
        } else {
            c
        }
    };
    match status {
        RailStatus::Running => pick("accent", "text"),
        RailStatus::Ok => pick("success", "text"),
        RailStatus::Error => pick("error", "text"),
    }
}

/// Title-case a raw tool name (`web_search` → `Web search`).
#[must_use]
pub fn pretty_name(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.replace('_', " ").chars().enumerate() {
        if i == 0 {
            out.extend(ch.to_uppercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// The rail header row: `● Title description`.
#[must_use]
pub fn header_line(
    status: RailStatus,
    title: &str,
    desc: Vec<Span<'static>>,
    theme: &Theme,
) -> Line<'static> {
    let dot = Style::default().fg(dot_color(status, theme));
    let title_style = Style::default()
        .fg(if status == RailStatus::Error {
            theme.color("error")
        } else {
            theme.color("text")
        })
        .add_modifier(Modifier::BOLD);
    let mut spans = vec![
        Span::styled("● ", dot),
        Span::styled(title.to_owned(), title_style),
    ];
    if !desc.is_empty() {
        spans.push(Span::raw(" "));
        spans.extend(desc);
    }
    Line::from(spans)
}

fn dim_style(theme: &Theme) -> Style {
    let c = theme.color("dim");
    Style::default().fg(if c == Color::Reset {
        theme.color("text_muted")
    } else {
        c
    })
}

/// Drop one level of `├─ `/`└─ ` tree connectors (and their `│  `
/// continuation indent): the `⎿` elbow already groups the rows.
fn strip_tree(body: &mut [Vec<(char, Style)>]) {
    let starts = |row: &[(char, Style)], pat: &str| {
        row.len() >= 3 && row.iter().take(3).map(|(c, _)| *c).eq(pat.chars())
    };
    if !body.iter().any(|r| starts(r, "├─ ") || starts(r, "└─ ")) {
        return;
    }
    for row in body.iter_mut() {
        if ["├─ ", "└─ ", "│  ", "   "].iter().any(|p| starts(row, p)) {
            row.drain(..3);
        }
    }
}

/// [`clip`] with a trailing `…` when anything was cut.
fn clip_ellipsis(cells: &[(char, Style)], max: usize) -> Vec<(char, Style)> {
    let total: usize = cells.iter().map(|(c, _)| cell_width(*c)).sum();
    if total <= max {
        return cells.to_vec();
    }
    let mut out = clip(cells, max.saturating_sub(1));
    let style = out.last().map_or_else(Style::default, |(_, s)| *s);
    out.push(('…', style));
    out
}

/// Clip cells to `max` display cells.
fn clip(cells: &[(char, Style)], max: usize) -> Vec<(char, Style)> {
    let mut used = 0;
    let mut out = Vec::new();
    for &(c, st) in cells {
        let w = cell_width(c);
        if used + w > max {
            break;
        }
        used += w;
        out.push((c, st));
    }
    out
}

/// `(title, description)` of a family render's header row (icon and
/// `Title:` colon dropped) — used for one-line group rows.
#[must_use]
pub fn header_parts(lines: &[Line<'static>]) -> (String, Vec<Span<'static>>) {
    let Parsed { header, .. } = parse(lines);
    let (title, desc) = split_header(&header);
    (title.iter().map(|(c, _)| *c).collect(), spans_of(&desc))
}

/// Rewrite a family render into rail form.
///
/// `fallback_title` names the tool when the family drew no header (the bash
/// card's top bar is empty; its `$ command` row is promoted instead).
#[must_use]
pub fn restyle(
    lines: &[Line<'static>],
    status: RailStatus,
    tool: &str,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let Parsed { header, mut body } = parse(lines);
    let (title, mut desc) = if header.is_empty() {
        (
            pretty_name(tool)
                .chars()
                .map(|c| (c, Style::default()))
                .collect(),
            Vec::new(),
        )
    } else {
        split_header(&header)
    };
    // Bash/eval: the `$ command` row becomes the description.
    if desc.is_empty() {
        if let Some(first) = body.first() {
            let t: String = first.iter().map(|(c, _)| *c).collect();
            if t.starts_with("$ ") {
                desc = first[2..].to_vec();
                body.remove(0);
            }
        }
    }
    // `⟦Wall: 0.12s | Timeout: 300s⟧` meta rows: the wall time rides in the
    // header as a dim suffix; the rest is noise in the clean layout.
    let mut wall: Option<String> = None;
    body.retain(|row| {
        let s: String = row.iter().map(|(c, _)| *c).collect();
        let s = s.trim();
        if s.starts_with('⟦') && s.ends_with('⟧') && (s.contains("Wall:") || s.contains("Timeout:"))
        {
            if let Some(rest) = s.split("Wall: ").nth(1) {
                wall = rest
                    .split(|c| c == ' ' || c == '|' || c == '⟧')
                    .next()
                    .map(str::to_owned);
            }
            return false;
        }
        true
    });
    let suffix: Vec<(char, Style)> = wall
        .map(|w| {
            format!(" · {w}")
                .chars()
                .map(|c| (c, dim_style(theme)))
                .collect()
        })
        .unwrap_or_default();
    let title_text: String = title.iter().map(|(c, _)| *c).collect();
    let title_text = if title_text.is_empty() {
        pretty_name(tool)
    } else {
        title_text
    };
    // Generic one-liners repeat the name (`todo Todo init …`): keep one.
    let desc_text: String = desc.iter().map(|(c, _)| *c).collect();
    let title_text = match desc_text.split_once(' ') {
        Some((first, _)) if first.eq_ignore_ascii_case(&title_text) => {
            desc.drain(..=first.chars().count());
            pretty_name(first)
        }
        _ if desc_text.eq_ignore_ascii_case(&title_text) => {
            desc.clear();
            pretty_name(&title_text)
        }
        _ => title_text,
    };
    let width = width.max(12);
    let room = width.saturating_sub(3 + title_text.chars().count() + suffix.len());
    let mut desc = clip_ellipsis(&desc, room);
    desc.extend(suffix);
    let mut out = vec![header_line(status, &title_text, spans_of(&desc), theme)];

    while body.last().is_some_and(Vec::is_empty) {
        body.pop();
    }
    strip_tree(&mut body);
    let budget = Budget::for_tool(tool).collapsed;
    let settled = status != RailStatus::Running;
    let hidden = if settled && !expanded && status == RailStatus::Ok && body.len() > budget {
        let hidden = body.len() - budget;
        body.truncate(budget);
        hidden
    } else {
        0
    };
    let dim = dim_style(theme);
    let inner = width.saturating_sub(BODY_INDENT.len());
    for (i, row) in body.iter().enumerate() {
        let prefix = if i == 0 { ELBOW } else { BODY_INDENT };
        let mut spans = vec![Span::styled(prefix, dim)];
        let mut row = clip(row, inner);
        if settled && status == RailStatus::Ok {
            for (_, st) in &mut row {
                *st = st.add_modifier(Modifier::DIM);
            }
        }
        spans.extend(spans_of(&row));
        out.push(Line::from(spans));
    }
    if hidden > 0 && budget > 0 {
        let prefix = if body.is_empty() { ELBOW } else { BODY_INDENT };
        let noun = if hidden == 1 { "line" } else { "lines" };
        out.push(Line::from(vec![
            Span::styled(prefix, dim),
            Span::styled(format!("… +{hidden} {noun} (ctrl+o to expand)"), dim),
        ]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::UiStyle;

    fn theme() -> Theme {
        let mut t = Theme::default();
        t.ui = UiStyle::Clean;
        t
    }

    fn rows(lines: &[Line<'_>]) -> Vec<String> {
        lines.iter().map(|l| text_of(l)).collect()
    }

    #[test]
    fn framed_card_becomes_rail() {
        let t = theme();
        let card = crate::tools::frame::framed_card(
            Some(Line::from("✎ Write: ◦ a.txt · 2 lines")),
            vec![crate::tools::frame::CardSection::plain(vec![
                Line::from("1 alpha"),
                Line::from("2 beta"),
            ])],
            40,
            Color::Gray,
        );
        let out = restyle(&card, RailStatus::Running, "write", false, 40, &t);
        assert_eq!(
            rows(&out),
            vec!["● Write ◦ a.txt · 2 lines", "  ⎿ 1 alpha", "    2 beta"]
        );
        // Settled write collapses to the header.
        let out = restyle(&card, RailStatus::Ok, "write", false, 40, &t);
        assert_eq!(rows(&out), vec!["● Write ◦ a.txt · 2 lines"]);
        // ctrl+O shows everything.
        let out = restyle(&card, RailStatus::Ok, "write", true, 40, &t);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn bash_command_row_promotes_to_header() {
        let t = theme();
        let card = crate::tools::frame::framed_card(
            None,
            vec![
                crate::tools::frame::CardSection::plain(vec![Line::from("$ echo hi")]),
                crate::tools::frame::CardSection::labelled(&t, "Output", vec![Line::from("hi")]),
            ],
            40,
            Color::Gray,
        );
        let out = restyle(&card, RailStatus::Ok, "bash", false, 40, &t);
        assert_eq!(rows(&out), vec!["● Bash echo hi", "  ⎿ hi"]);
        assert_eq!(out[0].spans[0].style.fg, Some(Color::Green));
        assert!(out[1].spans[1].style.add_modifier.contains(Modifier::DIM));
    }

    #[test]
    fn status_row_family_drops_icon_and_colon() {
        let t = theme();
        let lines = vec![
            Line::from(" ⌕ Grep: beta 1 match · 1 file"),
            Line::from(" └─ # a.txt"),
            Line::from("    *2│beta"),
        ];
        let out = restyle(&lines, RailStatus::Error, "grep", false, 60, &t);
        assert_eq!(
            rows(&out),
            vec!["● Grep beta 1 match · 1 file", "  ⎿ # a.txt", "    *2│beta"]
        );
        assert_eq!(out[0].spans[1].style.fg, Some(Color::Red));
    }
}
