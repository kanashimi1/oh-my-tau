//! `read` tool renderer — port of `pi-tui/src/tools/read-tool.ts` (TS §1 of the
//! pi-tui tools spec) plus the read-group data model from
//! `chat/read-tool-group.ts` (spec `chat.md` §7.1).
//!
//! The omp-cli read tool returns plain text: an optional `[path#TAG]`
//! hashline header, `N:line` rows, and trailing `[…]` continuation notices.
//! This module parses that shape (or the structured `details.displayContent`
//! when present) into styled transcript lines:
//!
//! - pending head: `⧗ Read <path>:<start>-<end>`
//! - error card: `✗ Read <path>:<range>` + error body
//! - success card: title, hashline header, line-numbered snippet, warnings
//! - grouped reads: consecutive fs reads fold into one compact block
//!
//! Line-building functions return `Vec<Line<'static>>` so the chat transcript
//! (inline viewport, per `core.md`) can append them to scrollback; the
//! [`ToolRenderer`] impl covers the dispatch contract in `tools/mod.rs`.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use serde_json::Value;

use crate::theme::Theme;

use super::{error_detail_lines, ToolRenderer};

/// Collapsed snippet height (`PREVIEW_LIMITS.OUTPUT_COLLAPSED`).
pub const COLLAPSED_PREVIEW_LINES: usize = 10;
/// Expanded snippet clamp (`read` code cell budget).
pub const EXPANDED_PREVIEW_LINES: usize = 1000;

/// Parsed read-tool call arguments (`ReadRenderArgs`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReadRenderArgs {
    /// Target path or URL (`file_path` legacy alias folded in).
    pub path: String,
    /// 1-based first line (args `offset` is 0-based).
    pub offset: Option<u64>,
    /// Max rows requested.
    pub limit: Option<u64>,
    /// `raw: true` — no hashline prefixes.
    pub raw: bool,
}

impl ReadRenderArgs {
    /// Extract args from a tool-call payload (`arguments` or `input`).
    #[must_use]
    pub fn from_call(call: &Value) -> Self {
        let args = call
            .get("arguments")
            .or_else(|| call.get("input"))
            .or_else(|| call.get("args"))
            .filter(|a| a.get("path").is_some() || a.get("file_path").is_some())
            .or(Some(call));
        let get_str = |k: &str| {
            args.and_then(|a| a.get(k))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        };
        let path = get_str("file_path")
            .or_else(|| get_str("path"))
            .unwrap_or_default();
        Self {
            path: path.to_owned(),
            offset: args.and_then(|a| a.get("offset")).and_then(Value::as_u64),
            limit: args.and_then(|a| a.get("limit")).and_then(Value::as_u64),
            raw: args
                .and_then(|a| a.get("raw"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }

    /// Display range suffix `:<start>-<end>` implied by offset/limit args.
    #[must_use]
    pub fn arg_range(&self) -> Option<String> {
        let start = self.offset? + 1;
        Some(match self.limit {
            Some(limit) if limit > 0 => format!(":{start}-{}", start + limit - 1),
            _ => format!(":{start}"),
        })
    }

    /// Peel the trailing selector from the target path, returning
    /// `(display path, selector)` (TS `splitReadRenderPath`).
    #[must_use]
    pub fn split_selector(&self) -> (String, Option<String>) {
        split_path_and_sel(&self.path)
    }
}

/// A single parsed selector chunk: 1-based inclusive line range or tail count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelChunk {
    /// `N`, `N-M`, `N+M` — inclusive 1-based lines.
    Range(u64, u64),
    /// `-N` — last N lines.
    Tail(u64),
}

/// Parse one selector chunk: `50`, `50-200`, `50+150`, `-60` (tail).
/// Incomplete counts (`50-`, `50+`) fail like TS `parseLineRanges`.
#[must_use]
pub fn parse_selector_chunk(chunk: &str) -> Option<SelChunk> {
    // Tail selector `-N` (last N lines): parse the count after the dash.
    if let Some(count) = chunk.strip_prefix('-') {
        let (count, rest) = digits_prefix(count)?;
        if !rest.is_empty() || count == 0 {
            return None;
        }
        return Some(SelChunk::Tail(count));
    }
    let (num, rest) = digits_prefix(chunk)?;
    if rest.is_empty() {
        return Some(SelChunk::Range(num, num));
    }
    if let Some(count) = rest.strip_prefix('+') {
        let (count, tail) = digits_prefix(count)?;
        if !tail.is_empty() || count == 0 {
            return None;
        }
        return Some(SelChunk::Range(num, num.saturating_add(count - 1)));
    }
    if let Some(end) = rest.strip_prefix('-') {
        let (end, tail) = digits_prefix(end)?;
        if !tail.is_empty() || end == 0 {
            return None;
        }
        return Some(SelChunk::Range(num, end));
    }
    None
}

fn digits_prefix(s: &str) -> Option<(u64, &str)> {
    let digits: usize = s.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let (n, rest) = s.split_at(digits);
    Some((n.parse().ok()?, rest))
}

/// Split a filesystem path from its trailing read selector
/// (TS `splitPathAndSel`): `f.rs:50`, `f.rs:50-200`, `f.rs:50+150`,
/// `f.rs:-60`, comma lists, `raw`/`conflicts`/`img`, and one compound
/// `f.rs:raw:1-50` / `f.rs:1-50:raw`.
#[must_use]
pub fn split_path_and_sel(raw_path: &str) -> (String, Option<String>) {
    let Some((base, sel)) = peel_one(raw_path) else {
        return (raw_path.to_owned(), None);
    };
    // Compound: exactly one of the two chunks is `raw`, the other a range.
    if let Some((inner_base, inner_sel)) = peel_one(base) {
        let outer_raw = sel.eq_ignore_ascii_case("raw");
        let inner_raw = inner_sel.eq_ignore_ascii_case("raw");
        let outer_range = is_range_selector(sel);
        let inner_range = is_range_selector(inner_sel);
        if (inner_raw && outer_range) || (outer_raw && inner_range) {
            // Canonical order: the range chunk first, `raw` second.
            let (range, raw) = if inner_raw {
                (sel, inner_sel)
            } else {
                (inner_sel, sel)
            };
            return (inner_base.to_owned(), Some(format!("{range}:{raw}")));
        }
    }
    (base.to_owned(), Some(sel.to_owned()))
}

/// Peel one trailing `:chunk` when it is a complete selector.
fn peel_one(raw_path: &str) -> Option<(&str, &str)> {
    let colon = raw_path.rfind(':')?;
    if colon == 0 {
        return None;
    }
    let candidate = &raw_path[colon + 1..];
    if is_selector(candidate) {
        Some((&raw_path[..colon], candidate))
    } else {
        None
    }
}

/// Complete selector chunk grammar (`FILE_LINE_RANGE_RE`).
#[must_use]
pub fn is_selector(chunk: &str) -> bool {
    chunk.eq_ignore_ascii_case("raw")
        || chunk.eq_ignore_ascii_case("conflicts")
        || chunk.eq_ignore_ascii_case("img")
        || is_range_selector(chunk)
}

/// Range list or tail chunk (`FILE_LINE_RANGE_ONLY_RE`).
#[must_use]
pub fn is_range_selector(chunk: &str) -> bool {
    if let Some(tail) = chunk.strip_prefix('-') {
        return digits_prefix(tail).is_some_and(|(_, rest)| rest.is_empty());
    }
    chunk.split(',').all(|c| parse_selector_chunk(c).is_some())
}

/// HTTP(S) or `www.` read target (`isReadableUrlPath`).
#[must_use]
pub fn is_readable_url_path(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("https://")
        || lower.starts_with("http://")
        || lower.starts_with("http:/")
        || lower.starts_with("https:/")
        || lower.starts_with("www.")
}

// ---------------------------------------------------------------------------
// Pending head
// ---------------------------------------------------------------------------

/// Pending status head: `⧗ Read <path>[:<start>-<end>]` (TS `renderCall`).
///
/// The range comes from the peeled selector when present, else from
/// `offset`/`limit` args.
#[must_use]
pub fn pending_head(args: &ReadRenderArgs) -> String {
    let (path, sel) = args.split_selector();
    let range = sel
        .as_deref()
        .and_then(first_read_selector_line)
        .map(|start| {
            let end = sel
                .as_deref()
                .and_then(selector_end_line)
                .map_or_else(|| format!("{start}"), |e| format!("{start}-{e}"));
            format!(":{end}")
        })
        .or_else(|| args.arg_range());
    match range {
        Some(range) => format!("Read {path}{range}"),
        None => format!("Read {path}"),
    }
}

/// First selected line of a selector (tail selectors have no start).
#[must_use]
pub fn first_read_selector_line(sel: &str) -> Option<u64> {
    sel.split(':')
        .find(|c| !c.eq_ignore_ascii_case("raw"))
        .and_then(|c| c.split(',').next())
        .and_then(parse_selector_chunk)
        .and_then(|c| match c {
            SelChunk::Range(start, _) => Some(start),
            SelChunk::Tail(_) => None,
        })
}

/// Last selected line (None for open ranges and tails with unknown end).
#[must_use]
fn selector_end_line(sel: &str) -> Option<u64> {
    sel.split(':')
        .find(|c| !c.eq_ignore_ascii_case("raw"))
        .and_then(|c| c.split(',').next_back())
        .and_then(parse_selector_chunk)
        .and_then(|c| match c {
            SelChunk::Range(_, end) => Some(end),
            SelChunk::Tail(_) => None,
        })
}

// ---------------------------------------------------------------------------
// Result parsing
// ---------------------------------------------------------------------------

/// One parsed snippet row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetRow {
    /// 1-based line number when the row carried one (hashline mode).
    pub line: Option<u64>,
    /// Row text without the gutter.
    pub text: String,
}

/// Parsed read result body (`ReadResult`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReadResult {
    /// Whether the tool reported `isError`.
    pub is_error: bool,
    /// Error/notice text when `is_error`.
    pub error_text: String,
    /// `[path]` or `[path#TAG]` hashline header, when present.
    pub header: Option<String>,
    /// Hashline tag (`#0793`), when present.
    pub tag: Option<String>,
    /// Parsed snippet rows.
    pub rows: Vec<SnippetRow>,
    /// True when rows carried `N:` numbers (hashline display mode).
    pub numbered: bool,
    /// Trailing `[…]` continuation/truncation notices.
    pub notices: Vec<String>,
    /// Structured image content blocks (base64 + mime), when any.
    pub images: Vec<(String, String)>,
}

/// Parse a `tool_execution_end` frame (`{"result": {...}, "isError": ...}`)
/// into display data. Notices are the trailing `[…]` paragraphs; rows are
/// `N:text` when the first content row is numbered, else raw lines.
#[must_use]
pub fn parse_result(frame: &Value) -> ReadResult {
    let result = frame.get("result").unwrap_or(&Value::Null);
    let is_error = frame
        .get("isError")
        .and_then(Value::as_bool)
        .or_else(|| result.get("isError").and_then(Value::as_bool))
        .unwrap_or(false);
    let text = result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(Value::as_str) == Some("text") {
                        b.get("text").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let images = result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(Value::as_str) == Some("image") {
                        Some((
                            b.get("mimeType")
                                .and_then(Value::as_str)
                                .unwrap_or("image")
                                .to_owned(),
                            b.get("data")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_owned(),
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let mut out = ReadResult {
        is_error,
        images,
        ..ReadResult::default()
    };
    if is_error {
        text.trim_start_matches("Error: ")
            .clone_into(&mut out.error_text);
        return out;
    }
    parse_body(&text, &mut out);
    out
}

fn parse_body(text: &str, out: &mut ReadResult) {
    let mut lines = text.lines().peekable();
    // Hashline header `[path]` / `[path#TAG]` on its own first line.
    if let Some(first) = lines.peek().copied() {
        if first.starts_with('[') && first.ends_with(']') && !first.contains(": ") {
            let inner = &first[1..first.len() - 1];
            match inner.split_once('#') {
                Some((path, tag)) => {
                    out.header = Some(path.to_owned());
                    out.tag = Some(tag.to_owned());
                }
                None => out.header = Some(inner.to_owned()),
            }
            lines.next();
        }
    }
    let body: Vec<&str> = lines.collect();
    // Trailing notices: paragraphs after a blank line wrapped in `[...]`.
    let mut body_end = body.len();
    let mut notices = Vec::new();
    let mut i = body.len();
    while i > 0 {
        i -= 1;
        let line = body[i];
        if line.is_empty() {
            if let Some(notice) = body.get(i + 1) {
                let trimmed = notice.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    notices.insert(0, trimmed.to_owned());
                    body_end = i;
                    continue;
                }
            }
            break;
        }
    }
    out.notices = notices;
    let rows: Vec<&str> = body[..body_end].to_vec();
    out.numbered = rows.first().is_some_and(|r| {
        let (num, rest) = digits_prefix(r).unwrap_or((0, ""));
        num > 0 && rest.starts_with(':')
    });
    for row in rows {
        if out.numbered {
            if let Some((num, rest)) = digits_prefix(row) {
                if let Some(text) = rest.strip_prefix(':') {
                    out.rows.push(SnippetRow {
                        line: Some(num),
                        text: text.to_owned(),
                    });
                    continue;
                }
            }
        }
        out.rows.push(SnippetRow {
            line: None,
            text: row.to_owned(),
        });
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Small theme handle for span construction.
struct Painter<'a> {
    theme: &'a Theme,
}

impl Painter<'_> {
    fn span(&self, token: &str, text: impl Into<String>) -> Span<'static> {
        Span::styled(text.into(), Style::default().fg(self.theme.color(token)))
    }
    fn muted(&self, text: impl Into<String>) -> Span<'static> {
        self.span("text_muted", text)
    }
}

/// Title head for the result card: `Read <path>[:range]` plus correction
/// suffix; conflicts get a warning badge.
#[must_use]
pub fn result_title(args: &ReadRenderArgs, result: &ReadResult) -> String {
    let (mut path, sel) = args.split_selector();
    if path.is_empty() {
        path = result.header.clone().unwrap_or_default();
    }
    let mut title = format!("Read {path}");
    if let Some(sel) = sel.as_deref() {
        if !sel.eq_ignore_ascii_case("raw") {
            title.push(':');
            title.push_str(sel);
        }
    }
    if result.tag.is_some() {
        // hashline snapshot: surface the tag so edits can anchor to it
        if let Some(tag) = &result.tag {
            title.push_str(" (snapshot ");
            title.push_str(tag);
            title.push(')');
        }
    }
    title
}

/// Build the transcript lines for a settled read result.
///
/// `expanded` toggles snippet height; notices and warnings always follow the
/// body (spec §0.4).
#[must_use]
pub fn result_lines(
    args: &ReadRenderArgs,
    result: &ReadResult,
    expanded: bool,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let p = Painter { theme };
    let mut lines = Vec::new();
    if result.is_error {
        let (path, sel) = args.split_selector();
        let range = sel.map_or_else(String::new, |s| format!(":{s}"));
        lines.push(Line::from(vec![p.span(
            "error",
            format!("{} Read {path}{range}", theme.glyph("failure")),
        )]));
        lines.extend(error_detail_lines(&result.error_text, usize::MAX, theme));
        return lines;
    }
    let title = result_title(args, result);
    lines.push(Line::from(vec![
        p.span("accent", format!("{} ", theme.glyph("collapsed"))),
        p.span("accent", title),
    ]));
    if let Some(header) = &result.header {
        lines.push(Line::from(p.muted(format!("[{header}]"))));
    }
    let budget = if expanded {
        EXPANDED_PREVIEW_LINES
    } else {
        COLLAPSED_PREVIEW_LINES
    };
    let gutter = result
        .rows
        .iter()
        .filter_map(|r| r.line)
        .map(|n| n.to_string().len())
        .max()
        .unwrap_or(0);
    for (i, row) in result.rows.iter().enumerate() {
        if i == budget && result.rows.len() > budget {
            break;
        }
        let spans = if let Some(n) = row.line.filter(|_| result.numbered) {
            vec![
                p.muted(format!("{n:>gutter$}:")),
                p.span("text", row.text.clone()),
            ]
        } else {
            vec![p.span("text", row.text.clone())]
        };
        lines.push(Line::from(spans));
    }
    if result.rows.len() > budget {
        lines.push(Line::from(p.muted(format!(
            "… +{} more lines (ctrl+o to expand)",
            result.rows.len() - budget
        ))));
    }
    for notice in &result.notices {
        lines.push(Line::from(p.span("warning", notice.clone())));
    }
    for (mime, _) in &result.images {
        lines.push(Line::from(p.muted(format!("(image {mime})"))));
    }
    lines
}

/// Convenience: pending + result for one call frame.
///
/// `frame` is a `tool_execution_start` / `tool_execution_end` JSON payload.
#[must_use]
pub fn render_frame(frame: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let args = ReadRenderArgs::from_call(frame);
    if frame.get("result").is_some() || frame.get("isError").is_some() {
        let result = parse_result(frame);
        result_lines(&args, &result, expanded, theme)
    } else {
        vec![Line::from(vec![
            Painter { theme }.span("text_muted", format!("⧗ {}", pending_head(&args)))
        ])]
    }
}

// ---------------------------------------------------------------------------
// Grouped reads (chat.md §7.1 — `ReadToolGroupComponent`)
// ---------------------------------------------------------------------------

/// Status of one grouped read entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStatus {
    /// Still executing.
    Pending,
    /// Completed.
    Success,
    /// Completed with soft-error content (truncation etc.).
    Warning,
    /// Failed.
    Error,
}

impl ReadStatus {
    fn glyph(self, theme: &Theme) -> &'static str {
        match self {
            Self::Pending => "⧗",
            Self::Success | Self::Warning => theme.glyph("success"),
            Self::Error => theme.glyph("failure"),
        }
    }
    fn token(self) -> &'static str {
        match self {
            Self::Pending => "text_muted",
            Self::Success => "success",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// One folded read entry (`ReadGroupEntry`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadGroupEntry {
    /// Display path (selector peeled).
    pub path: String,
    /// Entry status.
    pub status: ReadStatus,
    /// Snippet rows for the expanded view / collapsed preview.
    pub rows: Vec<SnippetRow>,
    /// Whether rows carry `N:` line numbers.
    pub numbered: bool,
    /// Trailing notices (truncation etc.) → warning tone.
    pub notices: Vec<String>,
}

impl ReadGroupEntry {
    /// Build an entry from a tool-execution frame; `None` for non-fs targets
    /// (URLs and internal schemes never collapse into the group).
    #[must_use]
    pub fn from_frame(frame: &Value) -> Option<Self> {
        let name = frame
            .get("toolName")
            .or_else(|| frame.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if name != "read" {
            return None;
        }
        let args = ReadRenderArgs::from_call(frame);
        let (path, sel) = args.split_selector();
        if sel.is_none() && is_readable_url_path(&args.path) {
            return None;
        }
        if args.path.contains("://") {
            return None;
        }
        let result = parse_result(frame);
        let status = if result.is_error {
            ReadStatus::Error
        } else if result.notices.is_empty() {
            ReadStatus::Success
        } else {
            ReadStatus::Warning
        };
        let path = if path.is_empty() {
            result.header.clone().unwrap_or_default()
        } else {
            path
        };
        Some(Self {
            path,
            status,
            rows: result.rows,
            numbered: result.numbered,
            notices: result.notices,
        })
    }
}

/// Fold consecutive read frames into group entries (caller feeds the frame
/// list of one assistant turn; TS `readArgsCollapseIntoGroup`).
#[must_use]
pub fn collapse_into_group(frames: &[Value]) -> Vec<ReadGroupEntry> {
    frames
        .iter()
        .filter_map(ReadGroupEntry::from_frame)
        .collect()
}

/// Lines for a read group: collapsed shows one row per file with a dim
/// preview; expanded shows full numbered snippets per file.
#[must_use]
pub fn group_lines(
    entries: &[ReadGroupEntry],
    expanded: bool,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let p = Painter { theme };
    let mut lines = Vec::new();
    for entry in entries {
        lines.push(Line::from(vec![
            p.span(entry.status.token(), entry.status.glyph(theme)),
            Span::raw(" "),
            p.span(entry.status.token(), entry.path.clone()),
        ]));
        let budget = if expanded { usize::MAX } else { 3 };
        for (i, row) in entry.rows.iter().enumerate() {
            if i >= budget {
                lines.push(Line::from(p.muted(format!(
                    "    … +{} more lines (ctrl+o to expand)",
                    entry.rows.len() - budget
                ))));
                break;
            }
            let text = if entry.numbered {
                row.line
                    .map_or_else(|| row.text.clone(), |n| format!("{n}:{}", row.text))
            } else {
                row.text.clone()
            };
            lines.push(Line::from(vec![p.muted("    "), p.span("text", text)]));
        }
        for notice in &entry.notices {
            lines.push(Line::from(vec![
                p.muted("    "),
                p.span("warning", notice.clone()),
            ]));
        }
    }
    lines
}

// ---------------------------------------------------------------------------
// ToolRenderer contract
// ---------------------------------------------------------------------------

/// Renderer for file reads.
pub struct Read;

impl ToolRenderer for Read {
    fn summary(&self, call: &Value) -> String {
        pending_head(&ReadRenderArgs::from_call(call))
    }

    fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        call: &Value,
        theme: &Theme,
    ) {
        // The dispatch contract carries the call only; the result-aware view
        // lives in [`render_frame`] (used by the chat transcript). Here we
        // draw the pending head as a placeholder row.
        let line = Line::from(Painter { theme }.span(
            "text_muted",
            format!("⧗ {}", pending_head(&ReadRenderArgs::from_call(call))),
        ));
        frame.render_widget(ratatui::widgets::Paragraph::new(line), area);
    }

    fn render_frame(&self, frame: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
        render_frame(frame, expanded, theme)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Theme {
        Theme::default()
    }

    #[test]
    fn selector_peeling_matches_ts() {
        assert_eq!(
            split_path_and_sel("src/a.rs:50-200"),
            ("src/a.rs".into(), Some("50-200".into()))
        );
        assert_eq!(
            split_path_and_sel("src/a.rs:50+150"),
            ("src/a.rs".into(), Some("50+150".into()))
        );
        assert_eq!(
            split_path_and_sel("f.rs:-60"),
            ("f.rs".into(), Some("-60".into()))
        );
        assert_eq!(
            split_path_and_sel("f.rs:1-5,20-30"),
            ("f.rs".into(), Some("1-5,20-30".into()))
        );
        assert_eq!(
            split_path_and_sel("a:1-50:raw"),
            ("a".into(), Some("1-50:raw".into()))
        );
        assert_eq!(
            split_path_and_sel("f.rs:raw:1-50"),
            ("f.rs".into(), Some("1-50:raw".into()))
        );
        // Trailing `+` stays in the path (incomplete selector).
        assert_eq!(split_path_and_sel("f.rs:50+"), ("f.rs:50+".into(), None));
        assert_eq!(split_path_and_sel("f.rs"), ("f.rs".into(), None));
    }

    #[test]
    fn chunk_semantics() {
        assert_eq!(parse_selector_chunk("50"), Some(SelChunk::Range(50, 50)));
        assert_eq!(
            parse_selector_chunk("50-200"),
            Some(SelChunk::Range(50, 200))
        );
        assert_eq!(
            parse_selector_chunk("50+150"),
            Some(SelChunk::Range(50, 199))
        );
        assert_eq!(parse_selector_chunk("-60"), Some(SelChunk::Tail(60)));
        assert_eq!(parse_selector_chunk("50+"), None);
        assert_eq!(parse_selector_chunk("50-0"), None);
    }

    #[test]
    fn url_detection() {
        assert!(is_readable_url_path("https://example.com"));
        assert!(is_readable_url_path("http:/example.com"));
        assert!(is_readable_url_path("www.example.com"));
        assert!(!is_readable_url_path("src/main.rs"));
        assert!(!is_readable_url_path("artifact://3"));
    }

    #[test]
    fn pending_head_includes_ranges() {
        let mk = |v: Value| ReadRenderArgs::from_call(&v);
        assert_eq!(
            pending_head(&mk(serde_json::json!({"path": "a.rs"}))),
            "Read a.rs"
        );
        assert_eq!(
            pending_head(&mk(
                serde_json::json!({"path": "a.rs", "offset": 12, "limit": 5})
            )),
            "Read a.rs:13-17"
        );
        assert_eq!(
            pending_head(&mk(serde_json::json!({"path": "a.rs:50-200"}))),
            "Read a.rs:50-200"
        );
        assert_eq!(
            pending_head(&mk(serde_json::json!({"file_path": "a.rs"}))),
            "Read a.rs"
        );
    }

    fn end_frame(body: &str) -> Value {
        serde_json::json!({
            "toolName": "read",
            "result": {"content": [{"type": "text", "text": body}]},
            "isError": false
        })
    }

    #[test]
    fn parses_hashline_body_and_notices() {
        let r = parse_result(&end_frame(
            "[a.txt#0793]\n1:hello\n2:world\n\n[3 more lines in file. Use a.txt:3 to continue]",
        ));
        assert_eq!(r.header.as_deref(), Some("a.txt"));
        assert_eq!(r.tag.as_deref(), Some("0793"));
        assert!(r.numbered);
        assert_eq!(
            r.rows[0],
            SnippetRow {
                line: Some(1),
                text: "hello".into()
            }
        );
        assert_eq!(r.notices.len(), 1);
        assert!(!r.is_error);
    }

    #[test]
    fn parses_raw_body() {
        let r = parse_result(&end_frame("just a raw line\nsecond"));
        assert_eq!(r.header, None);
        assert!(!r.numbered);
        assert_eq!(r.rows.len(), 2);
        assert_eq!(r.rows[1].text, "second");
    }

    #[test]
    fn error_result_renders_error_card() {
        let theme = theme();
        let frame = serde_json::json!({
            "toolName": "read", "isError": true,
            "result": {"content": [{"type": "text", "text": "Error: file not found"}]}
        });
        let args = ReadRenderArgs::from_call(&frame);
        let result = parse_result(&frame);
        let lines = result_lines(&args, &result, false, &theme);
        let flat: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        assert!(flat[0].contains("Read"), "{flat:?}");
        assert!(flat[1].contains("file not found"));
    }

    #[test]
    fn error_body_strips_prefix_and_indents_two_columns() {
        let theme = theme();
        let frame = serde_json::json!({
            "toolName": "read", "isError": true,
            "result": {"content": [{"type": "text", "text": "Error: file not found"}]}
        });
        let args = ReadRenderArgs::from_call(&frame);
        let result = parse_result(&frame);
        let lines = result_lines(&args, &result, false, &theme);
        let flat: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        assert_eq!(flat[1], "  file not found", "{flat:?}");
        assert!(lines[1]
            .spans
            .iter()
            .all(|s| s.style.fg == Some(theme.color("error"))));
    }

    #[test]
    fn success_renders_gutter_and_collapse_hint() {
        let theme = theme();
        let mut body = String::from("[a.txt#0abc]\n");
        for i in 1..=30 {
            body.push_str(&format!("{i}:line {i}\n"));
        }
        let frame = end_frame(body.trim_end());
        let args = ReadRenderArgs::from_call(&frame);
        let result = parse_result(&frame);
        let collapsed = result_lines(&args, &result, false, &theme);
        assert_eq!(collapsed.len(), 1 + 1 + COLLAPSED_PREVIEW_LINES + 1);
        assert!(collapsed
            .last()
            .unwrap()
            .to_string()
            .contains("+20 more lines"));
        let expanded = result_lines(&args, &result, true, &theme);
        assert_eq!(expanded.len(), 1 + 1 + 30);
        assert!(expanded[2].to_string().contains("1:line 1"));
    }

    #[test]
    fn images_surface_as_placeholder_rows() {
        let theme = theme();
        let frame = serde_json::json!({
            "toolName": "read",
            "result": {"content": [
                {"type": "text", "text": ""},
                {"type": "image", "mimeType": "image/png", "data": "aGk="}
            ]},
            "isError": false
        });
        let args = ReadRenderArgs::from_call(&frame);
        let result = parse_result(&frame);
        let lines = result_lines(&args, &result, false, &theme);
        assert!(lines
            .iter()
            .any(|l| l.to_string().contains("(image image/png)")));
    }

    #[test]
    fn snapshot_collapsed_read_card() {
        let theme = theme();
        let frame = end_frame("[a.txt#0793]\n1:hello\n2:world\n3:tail");
        let args = ReadRenderArgs::from_call(&frame);
        let result = parse_result(&frame);
        let lines = result_lines(&args, &result, false, &theme);
        let backend = ratatui::backend::TestBackend::new(40, (lines.len() as u16) + 1);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal
            .draw(|f| {
                let area = f.area();
                for (i, line) in lines.iter().enumerate() {
                    let y = area.y + i as u16;
                    if y < area.height {
                        f.buffer_mut()[(area.x, y)].set_symbol(&line.to_string());
                    }
                }
            })
            .expect("draw");
        let (w, h) = {
            let size = terminal.size().expect("size");
            (size.width, size.height)
        };
        let buf = terminal.backend().buffer().clone();
        let output: String = (0..h)
            .map(|y| {
                let mut row = String::new();
                for x in 0..w {
                    row.push_str(buf[(x, y)].symbol());
                }
                format!("{}\n", row.trim_end())
            })
            .collect();
        let expected = "▸ Read a.txt (snapshot 0793)\n[a.txt]\n1:hello\n2:world\n3:tail\n";
        assert_eq!(output.trim_end(), expected.trim_end());
    }

    #[test]
    fn group_collapses_fs_reads_only() {
        let theme = theme();
        let frames = vec![
            end_frame("[a.txt#0793]\n1:hi"),
            end_frame("# Title\n"),
            serde_json::json!({
                "toolName": "read", "args": {"path": "https://x"}, "isError": true,
                "result": {"content": [{"type": "text", "text": "Error: boom"}]}
            }),
        ];
        let entries = collapse_into_group(&frames);
        assert_eq!(entries.len(), 2); // URL read skipped
        assert_eq!(entries[0].path, "a.txt");
        assert_eq!(entries[0].status, ReadStatus::Success);
        assert_eq!(entries[1].status, ReadStatus::Success);
        let lines = group_lines(&entries, false, &theme);
        assert!(lines[0].to_string().contains("a.txt"));
    }
}
