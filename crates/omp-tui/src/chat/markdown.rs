//! Markdown rendering for assistant text — the port of
//! `pi-tui/src/chat/markdown.ts` (plus `render/` helpers).
//!
//! A self-contained CommonMark-subset renderer: block parsing (headings,
//! paragraphs, fenced code, tables, blockquotes, lists, thematic breaks,
//! display math), inline styling (emphasis, code, links, strikethrough,
//! `$…$` / `\(...\)` math via [`latex`]), and width-aware word wrapping with
//! unicode-aware cell widths. Output is a ratatui [`Text`] of styled
//! [`Line`]s ready for the inline renderer's `insert_before` path.
//!
//! Streaming: [`MarkdownStream`] commits every top-level block whose parse
//! can no longer change and re-renders only the open tail, so appending a
//! chunk costs a tail re-layout, not a full one.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use unicode_width::UnicodeWidthChar;

use crate::theme::Theme;

use super::code;
#[path = "latex.rs"]
pub mod latex;

/// Resolved style handles for one render pass (TS `MarkdownTheme`).
#[derive(Debug, Clone, Copy)]
struct Styles {
    text: Style,
    bold: Style,
    italic: Style,
    strike: Style,
    code: Style,
    code_block: Style,
    code_block_border: Style,
    link: Style,
    link_url: Style,
    quote: Style,
    quote_border: Style,
    hr: Style,
    heading: Style,
    list_bullet: Style,
}

impl Styles {
    fn new(theme: &Theme) -> Self {
        let text = Style::default().fg(theme.color("text"));
        Self {
            text,
            bold: text.add_modifier(Modifier::BOLD),
            italic: text.add_modifier(Modifier::ITALIC),
            strike: text.add_modifier(Modifier::CROSSED_OUT),
            code: Style::default().fg(theme.color("accent")),
            code_block: text.fg(theme.color("text")),
            code_block_border: Style::default().fg(theme.color("border")),
            link: Style::default()
                .fg(theme.color("accent"))
                .add_modifier(Modifier::UNDERLINED),
            link_url: Style::default().fg(theme.color("text_muted")),
            quote: text.add_modifier(Modifier::ITALIC),
            quote_border: Style::default().fg(theme.color("accent")),
            hr: Style::default().fg(theme.color("text_muted")),
            heading: Style::default()
                .fg(theme.color("text"))
                .add_modifier(Modifier::BOLD),
            list_bullet: Style::default().fg(theme.color("accent")),
        }
    }
}

/// Render markdown `src` laid out to `width` columns.
#[must_use]
pub fn render(src: &str, width: u16, theme: &Theme) -> Text<'static> {
    let st = Styles::new(theme);
    render_st(src, usize::from(width).max(1), theme, &st)
}

/// Core layout over pre-resolved styles (recursion shares the theme).
fn render_st(src: &str, width: usize, theme: &Theme, st: &Styles) -> Text<'static> {
    let mut lines = Vec::new();
    for block in parse_blocks(src) {
        render_block(&block, width, theme, st, &mut lines);
    }
    Text::from(lines)
}

/// Render one-line inline markdown (chips/labels); wraps nothing.
#[must_use]
pub fn render_inline(src: &str, theme: &Theme) -> Line<'static> {
    let st = Styles::new(theme);
    Line::from(inline_spans(src, &st))
}

// ---------------------------------------------------------------------------
// Inline parsing
// ---------------------------------------------------------------------------

/// Tokenize inline markdown into styled spans.
fn inline_spans(text: &str, st: &Styles) -> Vec<Span<'static>> {
    parse_inline(text, st, st.text)
}

#[allow(clippy::too_many_lines)] // delimiter-heavy inline parser
fn parse_inline(text: &str, st: &Styles, base: Style) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;

    macro_rules! flush {
        () => {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), base));
            }
        };
    }

    while i < text.len() {
        let rest = &text[i..];
        // Code span: `code`
        if let Some(after) = rest.strip_prefix('`') {
            if let Some(end) = after.find('`') {
                let inner = &after[..end];
                flush!();
                let content = inner
                    .strip_prefix(' ')
                    .and_then(|s| s.strip_suffix(' '))
                    .unwrap_or(inner);
                spans.push(Span::styled(content.to_owned(), base.patch(st.code)));
                i += end + 2;
                continue;
            }
        }
        // Bold: **x** or __x__
        if (rest.starts_with("**") || rest.starts_with("__"))
            && rest[2..].find(&rest[..2]).is_some_and(|e| e > 0)
        {
            let inner_end = rest[2..].find(&rest[..2]).unwrap();
            flush!();
            spans.extend(parse_inline(
                &rest[2..2 + inner_end],
                st,
                base.patch(st.bold),
            ));
            i += inner_end + 4;
            continue;
        }
        // Strikethrough: ~~x~~
        if rest.starts_with("~~") && rest[2..].find("~~").is_some_and(|e| e > 0) {
            let inner_end = rest[2..].find("~~").unwrap();
            flush!();
            spans.extend(parse_inline(
                &rest[2..2 + inner_end],
                st,
                base.patch(st.strike),
            ));
            i += inner_end + 4;
            continue;
        }
        // Italic: *x* or _x_ (the latter not inside words, so snake_case is safe).
        let delim = if rest.starts_with('*') {
            Some('*')
        } else if rest.starts_with('_') {
            Some('_')
        } else {
            None
        };
        if let Some(d) = delim {
            if let Some(inner_end) = rest[1..].find(d).filter(|e| *e > 0) {
                let prev_ok = d != '_' || i == 0 || !text.as_bytes()[i - 1].is_ascii_alphanumeric();
                let next_ok = !rest.as_bytes()[1].is_ascii_whitespace();
                if prev_ok && next_ok {
                    flush!();
                    spans.extend(parse_inline(
                        &rest[1..=inner_end],
                        st,
                        base.patch(st.italic),
                    ));
                    i += inner_end + 2;
                    continue;
                }
            }
        }
        // Link: [label](url)
        if rest.starts_with('[')
            && rest.find(']').is_some_and(|close| {
                rest[close..].starts_with("](") && rest[close + 2..].find(')').is_some()
            })
        {
            let close = rest.find(']').unwrap();
            let url_end = rest[close + 2..].find(')').unwrap();
            let label = &rest[1..close];
            let url = &rest[close + 2..close + 2 + url_end];
            flush!();
            spans.push(Span::styled(label.to_owned(), base.patch(st.link)));
            spans.push(Span::styled(format!(" ({url})"), base.patch(st.link_url)));
            i += close + 2 + url_end + 1;
            continue;
        }
        // Autolink: <http://...>
        if rest.starts_with("<http") && rest.find('>').is_some() {
            let end = rest.find('>').unwrap();
            flush!();
            spans.push(Span::styled(rest[1..end].to_owned(), base.patch(st.link)));
            i += end + 1;
            continue;
        }
        // Inline math: $...$ or \(...\)
        if let Some((math, len)) = take_math(rest) {
            flush!();
            spans.push(Span::styled(latex::to_unicode(&math), base));
            i += len;
            continue;
        }
        // Common HTML entities.
        if let Some(entity) = ["&amp;", "&lt;", "&gt;", "&quot;", "&#39;"]
            .iter()
            .find(|e| rest.starts_with(**e))
        {
            buf.push(match *entity {
                "&amp;" => '&',
                "&lt;" => '<',
                "&gt;" => '>',
                "&quot;" => '"',
                _ => '\'',
            });
            i += entity.len();
            continue;
        }
        let ch_len = rest.chars().next().map_or(1, char::len_utf8);
        buf.push_str(&rest[..ch_len]);
        i += ch_len;
    }
    flush!();
    spans
}

/// Extract the math source from a leading `$…$` or `\(...\)` span, returning
/// the source and the total byte length the span occupies.
fn take_math(rest: &str) -> Option<(String, usize)> {
    if let Some(after) = rest.strip_prefix('$') {
        if !after.starts_with('$') && !after.starts_with(' ') {
            if let Some(end) = after.find('$') {
                if end > 0 {
                    return Some((after[..end].to_owned(), end + 2));
                }
            }
        }
    }
    if let Some(after) = rest.strip_prefix("\\(") {
        if let Some(end) = after.find("\\)") {
            if end > 0 {
                return Some((after[..end].to_owned(), end + 4));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Width-aware wrapping
// ---------------------------------------------------------------------------

/// Display width of a string (zero-width chars count as 0).
fn str_width(s: &str) -> usize {
    s.chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum()
}

/// A styled word: run of non-space chars with their styles.
type Word = Vec<(char, Style)>;

/// Split spans into words (space-separated, spaces dropped) and hard-split
/// any word longer than `width` into `width`-sized pieces.
fn to_words(spans: &[Span], width: usize) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut cur: Word = Vec::new();
    for span in spans {
        for c in span.content.chars() {
            if c == ' ' || c == '\t' {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            } else {
                cur.push((c, span.style));
            }
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    // Hard-split overlong words.
    let mut out: Vec<Word> = Vec::new();
    for word in words {
        let mut w = 0;
        let mut piece: Word = Vec::new();
        for (c, st) in word {
            let cw = UnicodeWidthChar::width(c).unwrap_or(0);
            if w + cw > width && !piece.is_empty() {
                out.push(std::mem::take(&mut piece));
                w = 0;
            }
            w += cw;
            piece.push((c, st));
        }
        if !piece.is_empty() {
            out.push(piece);
        }
    }
    out
}

/// Greedy word wrap of styled spans into lines of at most `width` cells.
/// Word separators take `space_style` (the ambient text style) so styled
/// words remain their own clean span runs.
fn wrap_spans(spans: &[Span], width: usize, space_style: Style) -> Vec<Line<'static>> {
    let words = to_words(spans, width);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut cur: Word = Vec::new();
    let mut cur_w = 0;
    for word in words {
        let ww: usize = word
            .iter()
            .map(|(c, _)| UnicodeWidthChar::width(*c).unwrap_or(0))
            .sum();
        if !cur.is_empty() && cur_w + 1 + ww > width {
            lines.push(line_from_word(&cur));
            cur = Vec::new();
            cur_w = 0;
        }
        if !cur.is_empty() {
            cur.push((' ', space_style));
            cur_w += 1;
        }
        cur.extend(word);
        cur_w += ww;
    }
    if !cur.is_empty() {
        lines.push(line_from_word(&cur));
    }
    lines
}

/// Group a flat char run into minimal spans (style-run boundaries only).
fn line_from_word(word: &Word) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (c, st) in word {
        match spans.last_mut() {
            Some(last) if last.style == *st => last.content.to_mut().push(*c),
            _ => spans.push(Span::styled(c.to_string(), *st)),
        }
    }
    Line::from(spans)
}

// ---------------------------------------------------------------------------
// Block parsing
// ---------------------------------------------------------------------------

/// A parsed block.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Block {
    /// `#`..`######` heading plus raw inline source.
    Heading(u8, String),
    /// Blank-line-separated inline source lines.
    Paragraph(Vec<String>),
    /// Fenced code: info string, body, whether the closing fence arrived.
    Code {
        lang: String,
        body: String,
        closed: bool,
    },
    /// Pipe table: header cells and row cell lines.
    Table {
        header: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    /// Lines of `>`-quoted markdown (parsed recursively at render time).
    Quote(Vec<String>),
    /// Consecutive list items; each item is raw markdown lines at one indent.
    List(Vec<ListItem>),
    /// Thematic break; the source char drives the glyph.
    Hr(char),
    /// Display math: raw source (without the `$$`/`\[ \]` delimiters).
    Math(String),
}

/// One list entry: ordered number if any, checkbox state, raw content lines
/// (indent stripped), and nested items.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ListItem {
    ordered: Option<u32>,
    checkbox: Option<bool>,
    lines: Vec<String>,
    children: Vec<ListItem>,
}

/// Split cells on `|`, trimming; tolerates a leading/trailing empty cell.
fn split_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let trimmed = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix('|').unwrap_or(trimmed);
    trimmed.split('|').map(|c| c.trim().to_owned()).collect()
}

/// Whether the line is a table delimiter row like `| --- | :---: |`.
fn is_table_delim(line: &str) -> bool {
    let t = line.trim();
    t.contains('|') || t.contains('-')
}

fn fence_marker(line: &str) -> Option<(&str, String)> {
    let t = line.trim_start();
    for marker in ["```", "~~~"] {
        if let Some(rest_info) = t.strip_prefix(marker) {
            let info = rest_info.trim().to_owned();
            return Some((marker, info));
        }
    }
    None
}

/// Parse markdown into blocks. Streaming-safe: an unterminated fence,
/// comment, or math block simply becomes a still-open block.
fn parse_blocks(src: &str) -> Vec<Block> {
    let lines: Vec<&str> = src.lines().collect();
    parse_block_spans(&lines)
        .into_iter()
        .map(|(block, _)| block)
        .collect()
}

/// [`parse_blocks`] over pre-split lines, pairing every block with its end:
/// the index of the first line it did not consume. Deciding where a block
/// ends looks at most one line past that end (see [`block_is_final`]).
#[allow(clippy::too_many_lines)] // block-kind dispatch table
fn parse_block_spans(lines: &[&str]) -> Vec<(Block, usize)> {
    let mut blocks = Vec::new();
    let mut ends = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        // Every iteration pushes at most one block and then moves `i` past
        // it, so the cursor here is the previous block's end.
        if ends.len() < blocks.len() {
            ends.push(i);
        }
        let line = lines[i];
        let trimmed = line.trim();
        // Blank / space token.
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        // HTML comment: strip through `-->` (unterminated → strip to EOF).
        if trimmed.starts_with("<!--") {
            i += 1;
            while i < lines.len() && !lines[i - 1].contains("-->") {
                i += 1;
            }
            continue;
        }
        // Code fence.
        if let Some((marker, lang)) = fence_marker(line) {
            let mut body_lines: Vec<String> = Vec::new();
            let mut closed = false;
            i += 1;
            while i < lines.len() {
                let l = lines[i];
                if l.trim_start().starts_with(marker) {
                    closed = true;
                    i += 1;
                    break;
                }
                body_lines.push(l.to_owned());
                i += 1;
            }
            blocks.push(Block::Code {
                lang,
                body: body_lines.join("\n"),
                closed,
            });
            continue;
        }
        // Heading: 1-6 `#` + space (or EOL).
        {
            let mut n = 0;
            let bytes = trimmed.as_bytes();
            while n < bytes.len() && bytes[n] == b'#' {
                n += 1;
            }
            if (1..=6).contains(&n) && trimmed[n..].starts_with(' ') {
                let level = u8::try_from(n).expect("n is bounded by 1..=6");
                blocks.push(Block::Heading(level, trimmed[n..].trim().to_owned()));
                i += 1;
                continue;
            }
        }
        // Thematic break: >=3 identical chars from the HR family.
        if let Some(c) = hr_char(trimmed) {
            blocks.push(Block::Hr(c));
            i += 1;
            continue;
        }
        // HTML <hr>.
        if trimmed.eq_ignore_ascii_case("<hr>")
            || trimmed.eq_ignore_ascii_case("<hr/>")
            || trimmed.eq_ignore_ascii_case("<hr />")
        {
            blocks.push(Block::Hr('-'));
            i += 1;
            continue;
        }
        // Blockquote.
        if trimmed.starts_with('>') {
            let mut quote: Vec<String> = Vec::new();
            while i < lines.len()
                && (lines[i].trim().starts_with('>') || lines[i].trim().is_empty())
            {
                let t = lines[i].trim();
                if t.is_empty() {
                    // End of quote unless the following line continues it.
                    if i + 1 < lines.len() && lines[i + 1].trim().starts_with('>') {
                        quote.push(String::new());
                        i += 1;
                        continue;
                    }
                    break;
                }
                let after = t.strip_prefix('>').unwrap_or(t);
                quote.push(after.strip_prefix(' ').unwrap_or(after).to_owned());
                i += 1;
            }
            blocks.push(Block::Quote(quote));
            continue;
        }
        // Table: header line with pipes followed by a delimiter row.
        if trimmed.contains('|')
            && i + 1 < lines.len()
            && lines[i + 1].trim().contains('-')
            && split_table_row(lines[i + 1])
                .iter()
                .all(|c| !c.is_empty() && c.chars().all(|ch| "-: ".contains(ch)))
        {
            let header = split_table_row(trimmed);
            let mut rows = Vec::new();
            i += 2;
            while i < lines.len() && !lines[i].trim().is_empty() && lines[i].trim().contains('|') {
                rows.push(split_table_row(lines[i]));
                i += 1;
            }
            blocks.push(Block::Table { header, rows });
            continue;
        }
        // List.
        if list_marker(line).is_some() {
            let (items, next) = parse_list(lines, i);
            blocks.push(Block::List(items));
            i = next;
            continue;
        }
        // Display math: `$$…$$`, `\[…\]`, or bare `\begin{env}…\end{env}`.
        if trimmed.starts_with("$$")
            || trimmed.starts_with("\\[")
            || (trimmed.starts_with("\\begin{") && !trimmed.contains("\\end{"))
        {
            let mut math = String::new();
            let closer: &str = if trimmed.starts_with("$$") {
                "$$"
            } else if trimmed.starts_with("\\[") {
                "\\]"
            } else {
                "\\end{"
            };
            // Single-line form? (bare \begin needs at least one more line)
            if !trimmed.starts_with("\\begin{") && trimmed[closer.len()..].contains(closer) {
                let start = closer.len();
                let end = trimmed[start..].find(closer).unwrap() + start;
                math = trimmed[start..end].trim().to_string();
                i += 1;
            } else {
                let offset = if closer.is_empty() { 0 } else { closer.len() };
                math.push_str(trimmed[offset..].trim());
                i += 1;
                while i < lines.len() && !lines[i].trim().contains(closer) {
                    math.push(' ');
                    math.push_str(lines[i].trim());
                    i += 1;
                }
                if i < lines.len() {
                    let tail = lines[i].trim();
                    let end = tail.find(closer).unwrap_or(tail.len());
                    let piece = tail[..end].trim();
                    if !piece.is_empty() {
                        math.push(' ');
                        math.push_str(piece);
                    }
                    i += 1;
                }
            }
            blocks.push(Block::Math(math));
            continue;
        }
        // Paragraph: everything until a blank line or a new block start.
        let mut para: Vec<String> = Vec::new();
        while i < lines.len() && !lines[i].trim().is_empty() {
            let l = lines[i];
            let t = l.trim();
            if !para.is_empty() && starts_block(l, lines.get(i + 1).copied()) {
                break;
            }
            // <br> → hard line break.
            let t = t
                .strip_suffix("<br>")
                .or_else(|| t.strip_suffix("<br/>"))
                .unwrap_or(t);
            para.push(t.to_owned());
            i += 1;
        }
        blocks.push(Block::Paragraph(para));
    }
    if ends.len() < blocks.len() {
        ends.push(i);
    }
    blocks.into_iter().zip(ends).collect()
}

/// Whether a trimmed line is a thematic break; returns the source char.
fn hr_char(trimmed: &str) -> Option<char> {
    const FAMILY: &[char] = &['-', '*', '_', '─', '━', '═', '=', '–', '—'];
    let mut chars = trimmed.chars();
    let first = chars.next()?;
    if !FAMILY.contains(&first) || trimmed.len() < 3 {
        return None;
    }
    if chars.all(|c| c == first || c == ' ') {
        let count = trimmed.chars().filter(|&c| c == first).count();
        if count >= 3 {
            return Some(first);
        }
    }
    None
}

/// Whether `line` begins a new block (used to end a paragraph early).
fn starts_block(line: &str, next: Option<&str>) -> bool {
    let t = line.trim();
    t.starts_with("```")
        || t.starts_with("~~~")
        || t.starts_with('#')
        || t.starts_with('>')
        || t.starts_with("<!--")
        || hr_char(t).is_some()
        || t.starts_with("$$")
        || fence_marker(line).is_some()
        || list_marker(line).is_some()
        || (t.contains('|') && next.is_some_and(|n| is_table_delim(n.trim())))
}

/// Match a list item start; returns (indent, ordered number, checkbox, text).
fn list_marker(line: &str) -> Option<(usize, Option<u32>, Option<bool>, String)> {
    let indent = line.len() - line.trim_start().len();
    let t = line.trim_start();
    let (ordered, rest) = if let Some(t2) = t
        .strip_prefix("- [x] ")
        .or_else(|| t.strip_prefix("- [X] "))
    {
        (None, (Some(true), t2))
    } else if let Some(t2) = t.strip_prefix("- [ ] ") {
        (None, (Some(false), t2))
    } else if let Some(t2) = t
        .strip_prefix("- ")
        .or_else(|| t.strip_prefix("* "))
        .or_else(|| t.strip_prefix("+ "))
    {
        (None, (None, t2))
    } else {
        let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
        if !digits.is_empty() && t[digits.len()..].starts_with(". ") {
            (Some(digits.parse().ok()?), (None, &t[digits.len() + 2..]))
        } else {
            return None;
        }
    };
    let _ = ordered;
    let (checkbox, text) = rest;
    Some((indent, ordered, checkbox, text.to_owned()))
}

/// Parse consecutive list lines starting at `lines[start]` (which carries a
/// marker at the outermost indent). Returns the items and the next unconsumed
/// line index. Nested markers at greater indent become `children`.
fn parse_list(lines: &[&str], start: usize) -> (Vec<ListItem>, usize) {
    let base_indent = list_marker(lines[start]).map_or(0, |(indent, _, _, _)| indent);
    let mut items: Vec<ListItem> = Vec::new();
    let mut i = start;
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim();
        if t.is_empty() {
            // A blank line ends the list unless the next line is a list item
            // or an indented continuation.
            if i + 1 < lines.len()
                && (list_marker(lines[i + 1]).is_some()
                    || lines[i + 1].len() - lines[i + 1].trim_start().len() > base_indent)
            {
                i += 1;
                continue;
            }
            break;
        }
        if let Some((indent, ordered, checkbox, text)) = list_marker(line) {
            if indent < base_indent {
                break;
            }
            let mut item = ListItem {
                ordered,
                checkbox,
                lines: vec![text],
                children: Vec::new(),
            };
            i += 1;
            // Continuation lines (indented more than the marker).
            while i < lines.len() {
                let cont = lines[i];
                let ct = cont.trim();
                if ct.is_empty() {
                    break;
                }
                let cont_indent = cont.len() - cont.trim_start().len();
                if let Some((cindent, _, _, _ctext)) = list_marker(cont) {
                    if cindent > indent {
                        // Nested list: hand the remaining lines to a recursive
                        // parse and attach as children.
                        let (children, next) = parse_list(lines, i);
                        item.children = children;
                        i = next;
                        continue;
                    }
                    if cindent == indent {
                        break; // sibling; outer loop will consume it
                    }
                    // cindent < indent: belongs to an outer list.
                    break;
                }
                if cont_indent > indent {
                    item.lines.push(ct.to_owned());
                    i += 1;
                } else {
                    break;
                }
            }
            items.push(item);
            continue;
        }
        // Not a list line at all: if it is indented, treat as lazy
        // continuation of the previous item; otherwise the list is done.
        let indent = line.len() - line.trim_start().len();
        let continuation = indent > base_indent;
        match items.last_mut() {
            Some(last) if continuation => {
                last.lines.push(t.to_owned());
                i += 1;
            }
            _ => break,
        }
    }
    (items, i)
}

// ---------------------------------------------------------------------------
// Block rendering
// ---------------------------------------------------------------------------

fn render_block(
    block: &Block,
    width: usize,
    theme: &Theme,
    st: &Styles,
    out: &mut Vec<Line<'static>>,
) {
    match block {
        Block::Heading(level, text) => {
            let mut spans = inline_spans(text, st);
            if *level == 1 {
                for s in &mut spans {
                    s.style = s.style.add_modifier(Modifier::UNDERLINED);
                }
            } else if *level >= 3 {
                spans.insert(
                    0,
                    Span::styled("#".repeat(usize::from(*level)) + " ", st.heading),
                );
            }
            for s in &mut spans {
                if *level <= 2 {
                    s.style = s.style.add_modifier(Modifier::BOLD);
                }
            }
            out.push(Line::from(spans));
            out.push(Line::default());
        }
        Block::Paragraph(lines) => {
            let src = lines.join(" ");
            out.extend(wrap_spans(&inline_spans(&src, st), width, st.text));
            out.push(Line::default());
        }
        Block::Code { lang, body, closed } => {
            let info = if lang.is_empty() {
                String::new()
            } else {
                format!(" {lang}")
            };
            out.push(Line::from(Span::styled(
                format!("```{info} "),
                st.code_block_border,
            )));
            if *closed {
                // Delegate highlighting to chat::code; indent by the block indent.
                for line in code::highlight(body, Some(lang), theme).lines {
                    out.push(indent_line(line, 2, st.code_block_border));
                }
            } else {
                // Streaming: flat body styling until the fence closes.
                for l in body.lines() {
                    out.push(Line::from(Span::styled(format!("  {l}"), st.code_block)));
                }
            }
            if *closed {
                out.push(Line::from(Span::styled("``` ", st.code_block_border)));
            }
            out.push(Line::default());
        }
        Block::Table { header, rows } => {
            render_table(header, rows, width, st, out);
            out.push(Line::default());
        }
        Block::Quote(lines) => {
            let inner_src = lines.join("\n");
            let mut inner = render_st(&inner_src, width.saturating_sub(2).max(1), theme, st);
            // TS re-applies quote(italic(...)) around the inner render.
            for line in &mut inner.lines {
                for sp in &mut line.spans {
                    sp.style = st.quote.patch(sp.style);
                }
            }
            for line in inner.lines {
                let mut spans = vec![Span::styled("│ ", st.quote_border)];
                spans.extend(line.spans);
                out.push(Line::from(spans));
            }
            out.push(Line::default());
        }
        Block::List(items) => {
            render_list(items, width, st, 0, out);
            out.push(Line::default());
        }
        Block::Hr(c) => {
            let ch = match c {
                '─' | '━' | '═' => *c,
                '=' => '═',
                _ => '─',
            };
            let n = width.min(80);
            out.push(Line::from(Span::styled(ch.to_string().repeat(n), st.hr)));
            out.push(Line::default());
        }
        Block::Math(src) => {
            // 2-D display engine is P1-fancy; per spec a unicode inline
            // rendering of display math is the accepted fallback.
            let converted = latex::to_unicode(src);
            out.extend(wrap_spans(
                &[Span::styled(converted, st.text)],
                width,
                st.text,
            ));
            out.push(Line::default());
        }
    }
}

/// Prefix every span of `line` with `n` spaces in style `st`.
fn indent_line(mut line: Line<'static>, n: usize, st: Style) -> Line<'static> {
    let mut spans = vec![Span::styled(" ".repeat(n), st)];
    spans.append(&mut line.spans);
    Line::from(spans)
}

fn render_list(
    items: &[ListItem],
    width: usize,
    st: &Styles,
    base: usize,
    out: &mut Vec<Line<'static>>,
) {
    for item in items {
        let marker = match (&item.ordered, &item.checkbox) {
            (_, Some(true)) => "[✓] ".to_owned(),
            (_, Some(false)) => "[ ] ".to_owned(),
            (Some(n), _) => format!("{n}. "),
            (None, None) => "• ".to_owned(),
        };
        let marker_width = str_width(&marker);
        let content_indent = base + marker_width;
        let content_width = width.saturating_sub(content_indent).max(1);
        let para_src = item.lines.join(" ");
        let mut lines = wrap_spans(&inline_spans(&para_src, st), content_width, st.text);
        if lines.is_empty() {
            lines.push(Line::default());
        }
        let mut first = lines.remove(0);
        let mut spans = vec![Span::styled(
            " ".repeat(base) + &marker,
            if item.ordered.is_some() {
                st.text
            } else {
                st.list_bullet
            },
        )];
        spans.append(&mut first.spans);
        out.push(Line::from(spans));
        for mut cont in lines {
            let mut spans = vec![Span::styled(" ".repeat(content_indent), st.text)];
            spans.append(&mut cont.spans);
            out.push(Line::from(spans));
        }
        if !item.children.is_empty() {
            render_list(&item.children, width, st, content_indent, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines, clippy::needless_range_loop)] // grid layouter indexes widths/heights/grid together
fn render_table(
    header: &[String],
    rows: &[Vec<String>],
    width: usize,
    st: &Styles,
    out: &mut Vec<Line<'static>>,
) {
    let ncols = header
        .len()
        .max(rows.iter().map(Vec::len).max().unwrap_or(0));
    if ncols == 0 {
        return;
    }
    let border_overhead = 3 * ncols + 1;
    let available = width.saturating_sub(border_overhead);
    if available < ncols {
        let mut raw = header.join(" | ");
        for r in rows {
            raw.push('\n');
            raw.push_str(&r.join(" | "));
        }
        out.extend(wrap_spans(&[Span::styled(raw, st.text)], width, st.text));
        return;
    }
    let cell = |r: usize, c: usize| -> String {
        if r == 0 {
            header.get(c).cloned().unwrap_or_default()
        } else {
            rows.get(r - 1)
                .and_then(|row| row.get(c))
                .cloned()
                .unwrap_or_default()
        }
    };
    let nrows = 1 + rows.len();
    let mut natural = vec![0usize; ncols];
    let mut minw = vec![1usize; ncols];
    for c in 0..ncols {
        for r in 0..nrows {
            let t = cell(r, c);
            natural[c] = natural[c].max(str_width(&t).min(30));
            let longest_word = t.split_whitespace().map(str_width).max().unwrap_or(1);
            minw[c] = minw[c].max(longest_word.min(30)).max(1);
        }
    }
    let mut widths = natural.clone();
    let total: usize = widths.iter().sum();
    if total > available {
        let mut sum: usize = widths.iter().sum();
        while sum > available {
            // Columns may be squeezed below their min word width when the
            // terminal is narrow — cell wrapping hard-splits the words.
            let candidate = (0..ncols)
                .filter(|&c| widths[c] > 1)
                .max_by_key(|&c| widths[c].saturating_sub(minw[c]));
            match candidate {
                Some(c) => {
                    widths[c] -= 1;
                    sum -= 1;
                }
                None => break,
            }
        }
    } else if total < available {
        let surplus = available - total;
        let denom: usize = natural.iter().sum::<usize>().max(1);
        let mut given = 0usize;
        for c in 0..ncols {
            let share = surplus * natural[c] / denom;
            widths[c] += share;
            given += share;
        }
        let mut c = 0;
        while given < surplus {
            widths[c % ncols] += 1;
            given += 1;
            c += 1;
        }
    }
    let mut grid: Vec<Vec<Vec<String>>> = Vec::with_capacity(nrows);
    let mut heights = vec![1usize; nrows];
    for r in 0..nrows {
        let mut row_cells = Vec::with_capacity(ncols);
        let mut h = 1;
        for c in 0..ncols {
            let wrapped = wrap_plain(&cell(r, c), widths[c]);
            h = h.max(wrapped.len());
            row_cells.push(wrapped);
        }
        heights[r] = h;
        grid.push(row_cells);
    }
    let border = st.code_block_border;
    for r in 0..nrows {
        for line_idx in 0..heights[r] {
            let mut spans = vec![Span::styled("│ ".to_owned(), border)];
            for c in 0..ncols {
                let text = grid[r][c].get(line_idx).cloned().unwrap_or_default();
                let pad = widths[c].saturating_sub(str_width(&text));
                let style = if r == 0 { st.bold } else { st.text };
                spans.push(Span::styled(text, style));
                spans.push(Span::styled(" ".repeat(pad), style));
                spans.push(Span::styled(
                    if c + 1 == ncols {
                        " │".to_owned()
                    } else {
                        " │ ".to_owned()
                    },
                    border,
                ));
            }
            out.push(Line::from(spans));
        }
    }
}

/// Word-wrap plain text to `width` (hard-splitting overlong words).
fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0;
    for word in text.split_whitespace() {
        let ww = str_width(word);
        let sep = usize::from(!cur.is_empty());
        if !cur.is_empty() && cur_w + sep + ww > width {
            lines.push(std::mem::take(&mut cur));
            cur_w = 0;
            continue;
        }
        if !cur.is_empty() {
            cur.push(' ');
            cur_w += 1;
        }
        cur.push_str(word);
        cur_w += ww;
        while cur_w > width {
            let mut split_at = 0;
            let mut acc = 0;
            for (idx, ch) in cur.char_indices() {
                let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                if acc + cw > width {
                    break;
                }
                acc += cw;
                split_at = idx + ch.len_utf8();
            }
            if split_at == 0 {
                break;
            }
            let tail: String = cur[split_at..].to_owned();
            lines.push(cur[..split_at].to_owned());
            cur = tail;
            cur_w = str_width(&cur);
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Incremental renderer for one growing markdown document (a streaming
/// assistant message): `render(src)` output always equals
/// [`render`]`(src)`, but a block whose parse can no longer change is laid
/// out once and kept, so a frame only re-parses and re-renders the open
/// tail (normally the last block or two).
///
/// Every call validates its cache: the source must extend the committed
/// prefix and the width must match, otherwise the stream starts over. The
/// theme is not tracked — drop the stream when it changes.
#[derive(Debug, Clone, Default)]
pub struct MarkdownStream {
    width: usize,
    /// Source of the committed blocks (ends on a line boundary).
    committed_src: String,
    /// Rendered rows: the committed blocks' first, then the open tail's.
    lines: Vec<Line<'static>>,
    /// Rows of `lines` owned by committed blocks.
    committed_rows: usize,
    /// Uncommitted source the tail rows were rendered from (`None` until
    /// the first render or after a reset).
    tail_src: Option<String>,
}

impl MarkdownStream {
    /// Create an empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Rows of `src` laid out to `width` columns — identical to
    /// [`render`]`(src, width, theme).lines` — re-rendering only what the
    /// text appended since the previous call can still affect.
    pub fn render(&mut self, src: &str, width: u16, theme: &Theme) -> &[Line<'static>] {
        let width = usize::from(width).max(1);
        if width != self.width || !src.starts_with(self.committed_src.as_str()) {
            *self = Self {
                width,
                ..Self::default()
            };
        }
        let rest = &src[self.committed_src.len()..];
        if self.tail_src.as_deref() == Some(rest) {
            return &self.lines;
        }
        let st = Styles::new(theme);
        let lines: Vec<&str> = rest.lines().collect();
        let complete = rest.bytes().filter(|&b| b == b'\n').count();
        let blocks = parse_block_spans(&lines);
        let settled = blocks
            .iter()
            .take_while(|(block, end)| block_is_final(block, *end, complete))
            .count();
        self.lines.truncate(self.committed_rows);
        if settled > 0 {
            let end_line = blocks[settled - 1].1;
            let end_byte: usize = rest
                .split_inclusive('\n')
                .take(end_line)
                .map(str::len)
                .sum();
            for (block, _) in &blocks[..settled] {
                render_block(block, width, theme, &st, &mut self.lines);
            }
            self.committed_rows = self.lines.len();
            self.committed_src.push_str(&rest[..end_byte]);
        }
        for (block, _) in &blocks[settled..] {
            render_block(block, width, theme, &st, &mut self.lines);
        }
        self.tail_src = Some(src[self.committed_src.len()..].to_owned());
        &self.lines
    }

    /// Rows of the last [`Self::render`].
    #[must_use]
    pub fn lines(&self) -> &[Line<'static>] {
        &self.lines
    }
}

/// Whether a block parsed from a growing document can still change: its
/// parse read lines up to its `end` (one past for the blocks that end on
/// lookahead), and only the first `complete` lines (`\n`-terminated) are
/// immutable — the last line may still grow, and new lines may follow.
fn block_is_final(block: &Block, end: usize, complete: usize) -> bool {
    match block {
        // Decided by their own lines alone: the closing fence / the line.
        Block::Code { closed: true, .. } | Block::Heading(..) | Block::Hr(_) => end <= complete,
        // Paragraphs, lists, quotes, and tables end on (and may peek past)
        // the following line; open fences and math run to the end of input.
        _ => end + 1 < complete,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        Theme::default()
    }

    /// Whether a span carries the modifier (via `Style::add_modifier`).
    fn has(s: &Span, m: Modifier) -> bool {
        s.style.add_modifier.contains(m)
    }

    /// Plain concatenated text of every line (spans joined).
    fn plain(text: &Text) -> Vec<String> {
        text.lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    /// Paint the rendered text into a plain buffer to prove it draws.
    fn snapshot(src: &str, width: u16) -> String {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        let text = render(src, width, &theme());
        let height = (text.lines.len() as u16 + 2).max(1);
        let mut buffer = Buffer::empty(Rect::new(0, 0, width, height));
        for (y, line) in text.lines.iter().enumerate() {
            buffer.set_line(0, y as u16, line, width);
        }
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn heading_levels() {
        let t = render("# Title\n\n## Sub\n\n### Deep", 40, &theme());
        let lines = plain(&t);
        assert_eq!(lines[0], "Title");
        assert!(t.lines[0]
            .spans
            .iter()
            .all(|s| has(s, Modifier::BOLD) && has(s, Modifier::UNDERLINED)));
        assert_eq!(lines[2], "Sub");
        assert!(t.lines[2].spans.iter().all(|s| has(s, Modifier::BOLD)));
        assert!(t.lines[2]
            .spans
            .iter()
            .all(|s| !has(s, Modifier::UNDERLINED)));
        assert_eq!(lines[4], "### Deep");
    }

    #[test]
    fn paragraph_wraps_to_width() {
        let t = render("one two three four five six", 10, &theme());
        let lines: Vec<String> = plain(&t).into_iter().filter(|l| !l.is_empty()).collect();
        assert!(lines.iter().all(|l| str_width(l) <= 10), "{lines:?}");
        assert_eq!(lines.join(" "), "one two three four five six");
    }

    #[test]
    fn bold_italic_strike_code_spans() {
        let t = render("a **b** _c_ `d` ~~e~~", 40, &theme());
        let spans: Vec<_> = t.lines[0].spans.iter().collect();
        let raw: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(raw, "a b c d e");
        let has =
            |txt: &str, m: Modifier| spans.iter().any(|s| s.content.as_ref() == txt && has(s, m));
        assert!(has("b", Modifier::BOLD), "{raw}");
        assert!(has("c", Modifier::ITALIC), "{raw}");
        assert!(has("e", Modifier::CROSSED_OUT), "{raw}");
        assert!(spans
            .iter()
            .any(|s| s.content.as_ref() == "d" && s.style.fg.is_some()));
    }

    #[test]
    fn link_renders_label_and_url() {
        let t = render("[docs](https://x.dev)", 40, &theme());
        let raw: String = t.lines[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(raw, "docs (https://x.dev)");
        assert!(has(&t.lines[0].spans[0], Modifier::UNDERLINED));
        assert!(t.lines[0]
            .spans
            .iter()
            .any(|sp| sp.content.as_ref() == "(https://x.dev)" && sp.style.fg.is_some()));
    }

    #[test]
    fn snake_case_underscores_untouched() {
        let t = render("use snake_case_name here", 40, &theme());
        let raw: String = t.lines[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(raw, "use snake_case_name here");
        assert!(t.lines[0].spans.iter().all(|s| !has(s, Modifier::ITALIC)));
    }

    #[test]
    fn fenced_code_with_lang_and_borders() {
        let src = "```rust\nfn main() {}\n```";
        let t = render(src, 40, &theme());
        let lines = plain(&t);
        assert!(lines[0].starts_with("``` rust"), "{lines:?}");
        assert!(lines[1].starts_with("  fn main() {}"), "{lines:?}");
        assert!(lines[2].starts_with("```"), "{lines:?}");
    }

    #[test]
    fn streaming_fence_renders_flat_until_closed() {
        let t = render("```py\nprint(1)\n", 40, &theme());
        let lines = plain(&t);
        assert!(lines[0].starts_with("``` py"));
        assert!(lines.iter().all(|l| l.trim() != "```"));
    }

    #[test]
    fn table_borders_and_fallback() {
        let src = "| Name | Age |\n| --- | --- |\n| alice | 30 |\n| bob | 5 |";
        let t = render(src, 40, &theme());
        let lines = plain(&t);
        // Weighted surplus distribution grows both columns (spec §6).
        assert!(lines[0].starts_with("│ Name"), "{lines:?}");
        assert!(lines[0].contains("│ Age"), "{lines:?}");
        assert!(lines[1].starts_with("│ alice"), "{lines:?}");
        assert!(lines.iter().all(|l| str_width(l) <= 40));
        assert!(t.lines[0].spans.iter().any(|sp| has(sp, Modifier::BOLD)));
        let narrow = render(src, 8, &theme());
        assert!(plain(&narrow)[0].contains("Name"));
    }

    #[test]
    fn table_shrinks_to_width() {
        let src = "| aaaaaaaaaaaaaaaa | bbbbbbbbbbbbbbbb | cccccccccccccccc |\n| --- | --- | --- |\n| x | y | z |";
        let t = render(src, 30, &theme());
        for l in plain(&t) {
            assert!(str_width(&l) <= 30, "{l}");
        }
    }

    #[test]
    fn quote_has_border_and_italic() {
        let t = render("> hello world", 40, &theme());
        let line = &t.lines[0];
        assert!(line.spans[0].content.starts_with('│'));
        let inner: String = line.spans[1..].iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(inner.trim(), "hello world");
        assert!(line.spans[1..].iter().all(|s| has(s, Modifier::ITALIC)));
    }

    #[test]
    fn quote_wraps_at_width_minus_border() {
        let t = render("> one two three four", 12, &theme());
        let lines = plain(&t);
        assert!(lines.iter().all(|l| str_width(l) <= 12), "{lines:?}");
        assert!(lines[0].starts_with('│'));
        assert!(lines.len() > 2);
    }

    #[test]
    fn unordered_ordered_checkbox_lists() {
        let src = "- a\n- b\n1. one\n2. two\n- [x] done\n- [ ] todo";
        let t = render(src, 40, &theme());
        let lines = plain(&t);
        assert_eq!(lines[0], "• a");
        assert_eq!(lines[1], "• b");
        assert_eq!(lines[2], "1. one");
        assert_eq!(lines[3], "2. two");
        assert_eq!(lines[4], "[✓] done");
        assert_eq!(lines[5], "[ ] todo");
    }

    #[test]
    fn nested_list_indents() {
        let src = "- outer\n  - inner";
        let t = render(src, 40, &theme());
        let lines = plain(&t);
        assert_eq!(lines[0], "• outer");
        assert!(lines[1].starts_with("  •"), "{lines:?}");
    }

    #[test]
    fn hr_uses_width_cap() {
        let t = render("---", 100, &theme());
        let line = &plain(&t)[0];
        assert_eq!(line.chars().count(), 80);
        assert!(line.starts_with('─'));
        let eq = render("===", 20, &theme());
        assert!(plain(&eq)[0].starts_with('═'));
    }

    #[test]
    fn html_comments_stripped() {
        let t = render("a\n<!-- hidden -->\nb", 40, &theme());
        let lines = plain(&t);
        assert!(!lines.iter().any(|l| l.contains("hidden")), "{lines:?}");
        assert!(lines.iter().any(|l| l == "a"));
        assert!(lines.iter().any(|l| l == "b"));
    }

    #[test]
    fn unterminated_comment_stripped_streaming_safe() {
        let t = render("a\n<!-- still streaming", 40, &theme());
        assert!(!plain(&t).iter().any(|l| l.contains("still streaming")));
    }

    #[test]
    fn inline_math_uses_latex_unicode() {
        let t = render("energy $E = mc^2$ today", 40, &theme());
        let raw: String = t.lines[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(raw, "energy E = mc² today");
    }

    #[test]
    fn display_math_block() {
        let t = render("$$\\frac{1}{2} + x$$", 40, &theme());
        let lines = plain(&t);
        assert!(lines.iter().any(|l| l.contains('½')), "{lines:?}");
    }

    #[test]
    fn render_inline_for_chips() {
        let line = render_inline("**bold** and `code`", &theme());
        let raw: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(raw, "bold and code");
    }

    #[test]
    fn snapshot_paints_into_buffer() {
        let buf = snapshot("# Hi\n\n- a\n- b\n\n```rs\nx\n```", 30);
        assert!(buf.contains("Hi"), "{buf}");
        assert!(buf.contains('•'), "{buf}");
    }

    /// Deterministic xorshift PRNG for the randomized stream tests.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Fragments exercising every block kind and their boundaries.
    const FRAGMENTS: &[&str] = &[
        "# Title\n",
        "### Deep *heading* `x`\n",
        "Plain paragraph with **bold**, _it_, ~~gone~~ and a [link](https://e.x/a).\n",
        "Wide 日本語テキスト and emoji 😀👍🏽 naïve café that wraps around the edge.\n",
        "continued lazily on the next line\n",
        "\n",
        "\n\n",
        "```rust\nfn main() {\n    let x = 1;\n}\n```\n",
        "~~~\nraw ~~~ text\n~~~\n",
        "```py\nprint('unterminated')\n",
        "- one\n- two **b**\n  - nested a\n  - nested b\n- three\n",
        "1. first\n2. second\n   lazy continuation\n",
        "- [ ] todo\n- [x] done\n",
        "\n- after blank item\n",
        "| a | b |\n|---|:-:|\n| 1 | 22 |\n| 日本 | 😀 |\n",
        "| not | a table\n",
        "> quoted *text*\n> more\n>\n> after gap\n",
        "---\n",
        "***\n",
        "<hr>\n",
        "$$x^2 + y_1$$\n",
        "$$\n\\frac{a}{b}\n$$\n",
        "<!-- a comment\nspanning -->\n",
        "Inline $\\alpha$ and \\(\\beta\\) math.<br>\n",
        "Trailing text without newline",
        "CRLF line one\r\nCRLF line two\r\n\r\n",
        "    indented line\n",
    ];

    fn random_doc(rng: &mut Rng) -> String {
        let n = 1 + rng.below(14);
        (0..n)
            .map(|_| FRAGMENTS[rng.below(FRAGMENTS.len())])
            .collect()
    }

    /// Random chunk boundaries on char boundaries (1..=12 chars per chunk).
    fn random_cuts(rng: &mut Rng, doc: &str) -> Vec<usize> {
        let bounds: Vec<usize> = doc.char_indices().map(|(i, _)| i).skip(1).collect();
        let mut cuts = Vec::new();
        let mut k = 0;
        loop {
            k += 1 + rng.below(12);
            match bounds.get(k) {
                Some(&b) => cuts.push(b),
                None => break,
            }
        }
        cuts.push(doc.len());
        cuts
    }

    #[test]
    fn stream_matches_full_render_at_every_chunk() {
        let theme = theme();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for _ in 0..400 {
            let doc = random_doc(&mut rng);
            let mut width = [12u16, 30, 41, 80][rng.below(4)];
            let mut stream = MarkdownStream::new();
            for end in random_cuts(&mut rng, &doc) {
                if rng.below(16) == 0 {
                    // Resize mid-stream: the cache must not leak old widths.
                    width = [9u16, 25, 60, 100][rng.below(4)];
                }
                let prefix = &doc[..end];
                assert_eq!(
                    stream.render(prefix, width, &theme),
                    render(prefix, width, &theme).lines.as_slice(),
                    "doc {doc:?} prefix {end} width {width}"
                );
            }
        }
    }

    #[test]
    fn stream_rewritten_source_restarts() {
        let theme = theme();
        let mut stream = MarkdownStream::new();
        let _ = stream.render("# A\n\npara\n\n- x\n\nmore\n", 40, &theme);
        let other = "# B\n\nother text\n";
        assert_eq!(
            stream.render(other, 40, &theme),
            render(other, 40, &theme).lines.as_slice()
        );
    }

    #[test]
    fn stream_commits_settled_blocks() {
        let theme = theme();
        let mut stream = MarkdownStream::new();
        let mut doc = String::new();
        for n in 0..50 {
            doc.push_str(&format!(
                "## S{n}\n\nText {n}.\n\n```rs\nlet v = {n};\n```\n\n"
            ));
            let _ = stream.render(&doc, 60, &theme);
        }
        doc.push_str("tail");
        let _ = stream.render(&doc, 60, &theme);
        // Everything but the open paragraph is committed: a frame re-parses
        // only `tail`.
        assert_eq!(&doc[stream.committed_src.len()..], "\ntail");
        assert_eq!(
            stream.render(&doc, 60, &theme),
            render(&doc, 60, &theme).lines.as_slice()
        );
    }
}
