//! Transcript block rendering: turns one [`ChatEntry`] into styled rows.
//!
//! Ports the pi-tui chat component bodies (`chat/user-message.ts`,
//! `chat/assistant-message.ts`, `chat/read-tool-group.ts`,
//! `chrome/error-block.ts`, `overlays/usage-row.ts`) onto the ratatui
//! [`Line`] model so both the scrollback commit path
//! ([`Messages::take_committed_rows`]) and the live tail
//! ([`App::render_chat`]) share one renderer.
//!
//! Tool cards dispatch through the [`crate::tools`] family renderers
//! (review2 §11 #1): the card's stored payload is rebuilt into a synthetic
//! `tool_execution_*`-shaped call and handed to `ToolRenderer::render`
//! (full card, drawn onto a scratch [`TestBackend`] and read back as rows)
//! or the families' native line APIs (`edit::build_card_lines`,
//! `read::render_frame`). Collapsed/expanded state is the transcript-wide
//! ctrl+O flag, exactly as the data model already tracks it.

use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::{backend::TestBackend, layout::Rect, Terminal};
use serde_json::{json, Value};

use crate::chat::markdown;
use crate::chat::reveal::RevealBlock;
use crate::chat::thinking::{ThinkingBlockView, ThinkingDisplay};
use crate::chat::{
    AssistantMsg, ChatEntry, ChatKind, Compaction, Custom, ExecStatus, Execution, ReadGroup,
    ToolCard, ToolStatus, Usage,
};
use crate::theme::Theme;
use crate::tools;

/// Collapsed error-block row cap (TS `MAX_BANNER_ROWS` / chat.md §6).
pub const ERROR_ROW_CAP: usize = 8;
/// Continuation indent for wrapped error rows (TS `CONTINUATION_INDENT`).
const CONTINUATION_INDENT: &str = "  ";
/// Scratch-card height for the frame-render readback (families self-cap
/// their budgets; trailing blank rows are trimmed afterwards).
const CARD_MAX_ROWS: u16 = 48;
/// Collapsed compaction-summary budget.
const SUMMARY_COLLAPSED_ROWS: usize = 8;

/// Render one transcript entry to rows at `width` columns.
#[must_use]
pub fn entry_lines(
    entry: &ChatEntry,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    match &entry.kind {
        ChatKind::User {
            text,
            synthetic,
            live_steered,
        } => user_lines(text, *synthetic, *live_steered, width, theme),
        ChatKind::Assistant(a) => assistant_lines(a, expanded, width, theme),
        ChatKind::Tool(card) => tool_card_lines(card, expanded, width, theme),
        ChatKind::ReadGroup(group) => read_group_lines(group, expanded, width, theme),
        ChatKind::Bash(exec) => exec_lines(exec, "bash", width, theme),
        ChatKind::Python(exec) => exec_lines(exec, "eval", width, theme),
        ChatKind::Compaction(c) => compaction_lines(c, expanded, width, theme),
        ChatKind::Branch { summary } => branch_lines(summary, expanded, width, theme),
        ChatKind::Custom(c) => custom_lines(c, theme),
        ChatKind::Usage(usage) => usage_block_lines(usage, width, theme),
        ChatKind::Notice(text) => notice_lines(text, expanded, width, theme),
    }
}

/// Render a reveal-snapshot of assistant blocks (live tail, smooth reveal).
#[must_use]
pub fn revealed_lines(blocks: &[RevealBlock], width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let settings = ThinkingDisplay::default();
    let mut lines = Vec::new();
    for block in blocks {
        match block {
            RevealBlock::Text(text) => lines.extend(prose_lines(text, width, theme)),
            RevealBlock::Thinking { display, .. } => {
                let view = ThinkingBlockView {
                    text: display.clone(),
                    live: true,
                    preformatted: true,
                    ..ThinkingBlockView::default()
                };
                lines.extend(view.render(&settings, theme).lines);
            }
        }
    }
    lines
}

// ─── User bubble (chat.md §4) ──────────────────────────────────────────────

/// Collapse composer attachments into chips: image markdown becomes a
/// `🖼 label` chip (TS `UserMessageComponent` image/mention chips).
#[must_use]
pub fn collapse_chips(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("![") {
        out.push_str(&rest[..start]);
        let Some(close) = rest[start..].find("](") else {
            out.push_str(&rest[start..]);
            return out;
        };
        let alt = &rest[start + 2..start + close];
        let after = start + close + 2;
        let Some(end) = rest[after..].find(')') else {
            out.push_str(&rest[start..]);
            return out;
        };
        let label = if alt.is_empty() { "image" } else { alt };
        out.push_str(&format!("[🖼 {label}]"));
        rest = &rest[after + end + 1..];
    }
    out.push_str(rest);
    out
}

fn user_lines(
    text: &str,
    synthetic: bool,
    live_steered: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    // `UserMessageComponent`: Markdown with paddingX 1 / paddingY 1 on the
    // `userMessageBg` fill — a flush-left bubble, no glyph, full-width rows.
    let chipped = collapse_chips(text);
    let inner = width.saturating_sub(2).max(1);
    let mut text = markdown::render(&chipped, inner as u16, theme);
    let bg = theme.color("userMessageBg");
    let fg = theme.color("userMessageText");
    let fill = Style::default().bg(bg);
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(text.lines.len() + 2);
    // Top padding row; a live-steered message carries `*` at its left.
    let top = if live_steered {
        Line::from(vec![Span::styled(
            "* ",
            Style::default().fg(theme.color("warning")).bg(bg),
        )])
    } else {
        Line::default()
    };
    lines.push(pad_row(top, width, fill));
    for line in &mut text.lines {
        for span in &mut line.spans {
            let mut style = span.style;
            style.bg = Some(bg);
            if synthetic {
                style.fg = Some(theme.color("text_muted"));
            } else if style.fg.is_none() && fg != ratatui::style::Color::Reset {
                style.fg = Some(fg);
            }
            span.style = style;
        }
        let mut spans = vec![Span::styled(" ", fill)];
        spans.append(&mut line.spans);
        lines.push(pad_row(Line::from(spans), width, fill));
    }
    lines.push(pad_row(Line::default(), width, fill));
    lines
}

/// Pad `line` with the bubble fill out to `width` cells.
fn pad_row(mut line: Line<'static>, width: usize, fill: Style) -> Line<'static> {
    let used: usize = line
        .spans
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
        .sum();
    if used < width {
        line.spans.push(Span::styled(" ".repeat(width - used), fill));
    }
    line
}

// ─── Assistant (chat.md §3) ────────────────────────────────────────────────

/// Assistant prose: Markdown at paddingX 1 (`AssistantMessageComponent`
/// builds `new Markdown(text, 1, 0, ...)`) — a one-cell left indent.
fn prose_lines(text: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let inner = width.saturating_sub(2).max(1);
    markdown::render(text, inner as u16, theme)
        .lines
        .into_iter()
        .map(|mut line| {
            if !line.spans.is_empty() {
                line.spans.insert(0, Span::raw(" "));
            }
            line
        })
        .collect()
}

fn assistant_lines(
    a: &AssistantMsg,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let settings = ThinkingDisplay::default();
    let mut lines = Vec::new();
    for segment in &a.segments {
        match segment {
            crate::chat::AssistantSegment::Text { text } => {
                lines.extend(prose_lines(text, width, theme));
            }
            crate::chat::AssistantSegment::Thinking { text } => {
                let view = ThinkingBlockView {
                    text: text.clone(),
                    live: a.live,
                    elapsed: Duration::ZERO,
                    streamed: true,
                    ..ThinkingBlockView::default()
                };
                lines.extend(view.render(&settings, theme).lines);
            }
        }
    }
    // While the error is pinned above the editor the inline block is
    // suppressed unless expanded (TS `#errorPinned`); the caller-owned
    // Spacer precedes it.
    if let Some(error) = a.error_message.as_ref().filter(|_| !a.error_pinned || expanded) {
        lines.push(Line::default());
        lines.extend(format_error_block(error, width, expanded, theme));
    }
    lines
}

/// Inline turn-ending error (TS `AssistantMessageComponent#appendErrorBlock`):
/// `formatErrorBlock` rows at paddingX 1, `Error: ` on the first logical
/// line, everything error-tinted; capped at [`ERROR_ROW_CAP`] rows unless
/// `expanded`, with a dim `… +N more lines (ctrl+o to expand)` trailer.
#[must_use]
pub fn format_error_block(
    message: &str,
    width: usize,
    expanded: bool,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let cap = if expanded { usize::MAX } else { ERROR_ROW_CAP };
    let error = Style::default().fg(theme.color("error"));
    error_block_rows(message, width.saturating_sub(2), cap, "Error: ")
        .into_iter()
        .map(|row| match row {
            ErrorRow::Text(text) => Line::from(vec![Span::raw(" "), Span::styled(text, error)]),
            ErrorRow::More(text) => Line::from(vec![
                Span::raw(" "),
                Span::styled(text, Style::default().fg(theme.color("dim"))),
            ]),
        })
        .collect()
}

/// Max wrapped message rows in the pinned banner (TS `MAX_BANNER_ROWS`).
pub const BANNER_ROW_CAP: usize = 4;

/// TS `ErrorBannerComponent` (`overlays/error-banner.ts`): the turn-ending
/// provider error pinned above the editor — a spacer, an error-tinted
/// full-width rule, the message (`✘ ` + line 0 bold, wrapped, capped at
/// [`BANNER_ROW_CAP`] rows), the dim dismissal hint, and a closing rule.
#[must_use]
pub fn error_banner_lines(message: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let error = Style::default().fg(theme.color("error"));
    let dim = Style::default().fg(theme.color("dim"));
    let rule = Line::from(Span::styled("─".repeat(width), error));
    let prefix = format!("{} ", theme.symbol("status.error"));
    let mut lines = vec![Line::default(), rule.clone()];
    for (index, row) in error_block_rows(message, width.saturating_sub(2), BANNER_ROW_CAP, &prefix)
        .into_iter()
        .enumerate()
    {
        let span = match row {
            ErrorRow::Text(text) if index == 0 => {
                Span::styled(text, error.add_modifier(Modifier::BOLD))
            }
            ErrorRow::Text(text) => Span::styled(text, error),
            ErrorRow::More(text) => Span::styled(text, dim),
        };
        lines.push(Line::from(vec![Span::raw(" "), span]));
    }
    lines.push(Line::from(vec![
        Span::raw(" "),
        Span::styled("Dismissed when you send your next message.", dim),
    ]));
    lines.push(rule);
    lines
}

/// One row of a [`error_block_rows`] block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorRow {
    /// A wrapped message row (continuations carry the hanging indent).
    Text(String),
    /// The dim `… +N more lines (ctrl+o to expand)` overflow row.
    More(String),
}

/// TS `formatErrorBlock` text: logical lines trimmed, blank ones dropped,
/// `first_prefix` glued to line 0 before wrapping at `content_width - 2`,
/// every row after the first hung under [`CONTINUATION_INDENT`], at most
/// `max_rows` rows plus an overflow row.
#[must_use]
pub fn error_block_rows(
    message: &str,
    content_width: usize,
    max_rows: usize,
    first_prefix: &str,
) -> Vec<ErrorRow> {
    let mut logical: Vec<String> = message
        .replace("\r\n", "\n")
        .replace('\t', "   ")
        .split('\n')
        .map(|l| l.trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect();
    if logical.is_empty() {
        logical.push("Unknown error".to_owned());
    }
    let wrap_width = content_width.saturating_sub(CONTINUATION_INDENT.len()).max(1);
    let mut rows: Vec<String> = Vec::new();
    for (index, line) in logical.iter().enumerate() {
        let styled = if index == 0 {
            format!("{first_prefix}{line}")
        } else {
            line.clone()
        };
        for row in wrap_text(&styled, wrap_width) {
            rows.push(if rows.is_empty() {
                row
            } else {
                format!("{CONTINUATION_INDENT}{row}")
            });
        }
    }
    let hidden = rows.len().saturating_sub(max_rows);
    rows.truncate(max_rows);
    let mut out: Vec<ErrorRow> = rows.into_iter().map(ErrorRow::Text).collect();
    if hidden > 0 {
        out.push(ErrorRow::More(format!(
            "{CONTINUATION_INDENT}… +{hidden} more line{} (ctrl+o to expand)",
            if hidden == 1 { "" } else { "s" }
        )));
    }
    out
}

/// Greedy word wrap to `width` display columns (chars approximation).
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut rows = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            rows.push(std::mem::take(&mut line));
        } else if !line.is_empty() {
            line.push(' ');
        }
        let mut chars = word.chars();
        while line.chars().count() < width {
            match chars.next() {
                Some(c) => line.push(c),
                None => break,
            }
        }
        let rest: String = chars.collect();
        if !rest.is_empty() {
            rows.push(std::mem::take(&mut line));
            let mut rest = rest;
            loop {
                let len = rest.chars().count();
                if len <= width {
                    line = rest;
                    break;
                }
                let tail: String = rest.chars().skip(width).collect();
                rows.push(rest.chars().take(width).collect());
                rest = tail;
            }
        }
    }
    if !line.is_empty() {
        rows.push(line);
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
}

// ─── Usage row (overlays/usage-row.ts) ─────────────────────────────────────

/// Below this the rate is nonsense (cached/instant responses).
const MIN_RATE_DURATION_MS: u64 = 100;

/// TS `formatUsageRow`: the metric specs (`usageRowSpecs`) joined by two
/// spaces — local `YYYY-MM-DD HH:MM:SS`, `Δ <turn time>`, `⤵ input`,
/// `⤴ output`, `💾 cache read`, `⏱ ttft`, `⚡ tok/s`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // display-only rounding
pub fn usage_metrics(usage: &Usage, theme: &Theme) -> String {
    let mut specs: Vec<String> = Vec::new();
    if let Some(ts) = usage.timestamp_ms.filter(|t| *t > 0) {
        specs.push(crate::chat::local_time::format_local_timestamp(ts));
    }
    if let Some(elapsed) = usage.turn_elapsed_ms.filter(|e| *e > 0) {
        specs.push(format!("Δ {}", format_duration(elapsed)));
    }
    let metric = |icon: &str, value: String| format!("{} {value}", theme.symbol(icon));
    specs.push(metric("icon.input", format_number(usage.input + usage.cache_write)));
    specs.push(metric("icon.output", format_number(usage.output)));
    if usage.cache_read > 0 {
        specs.push(metric("icon.cache", format_number(usage.cache_read)));
    }
    if let Some(ttft) = usage.ttft_ms.filter(|t| *t > 0) {
        specs.push(metric("icon.time", format!("{}s", to_fixed1(ttft as f64 / 1000.0))));
    }
    if let Some(duration) = usage.duration_ms.filter(|d| *d > MIN_RATE_DURATION_MS) {
        if usage.output > 0 {
            let rate = usage.output as f64 * 1000.0 / duration as f64;
            specs.push(metric("icon.throughput", format!("{}/s", to_fixed1(rate))));
        }
    }
    specs.join("  ")
}

/// TS `createUsageRowBlock`: a spacer row, then the dim metric strip at
/// paddingX 1 (word-wrapped when wider than the row).
#[must_use]
pub fn usage_block_lines(usage: &Usage, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let dim = Style::default().fg(theme.color("dim"));
    let metrics = usage_metrics(usage, theme);
    let inner = width.saturating_sub(2).max(1);
    let rows = if unicode_width::UnicodeWidthStr::width(metrics.as_str()) <= inner {
        vec![metrics]
    } else {
        wrap_text(&metrics, inner)
    };
    let mut lines = vec![Line::default()];
    lines.extend(
        rows.into_iter()
            .map(|row| Line::from(vec![Span::raw(" "), Span::styled(row, dim)])),
    );
    lines
}

/// JS `Number#toFixed(1)` (half away from zero, unlike `{:.1}`'s
/// half-to-even).
fn to_fixed1(v: f64) -> String {
    format!("{:.1}", (v * 10.0).round() / 10.0)
}

/// pi-utils `formatNumber`: `999`, `1.2K`, `19K`, `1.5M`, `12M`, `2B`.
#[must_use]
#[allow(clippy::cast_precision_loss)] // display-only rounding of token counts
pub fn format_number(n: u64) -> String {
    let trim1 = |v: f64| {
        let s = to_fixed1(v);
        s.strip_suffix(".0").map_or_else(|| s.clone(), str::to_owned)
    };
    let f = n as f64;
    match n {
        0..=999 => n.to_string(),
        1_000..=9_999 => format!("{}K", trim1(f / 1e3)),
        10_000..=999_999 => format!("{}K", (f / 1e3).round()),
        1_000_000..=9_999_999 => format!("{}M", trim1(f / 1e6)),
        10_000_000..=999_999_999 => format!("{}M", (f / 1e6).round()),
        1_000_000_000..=9_999_999_999 => format!("{}B", trim1(f / 1e9)),
        _ => format!("{}B", (f / 1e9).round()),
    }
}

/// pi-utils `formatDuration`: `850ms`, `6.2s`, `3m12s`, `2h5m`, `1d4h`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_duration(ms: u64) -> String {
    const SEC: u64 = 1_000;
    const MIN: u64 = 60 * SEC;
    const HOUR: u64 = 60 * MIN;
    const DAY: u64 = 24 * HOUR;
    let unit = |major: u64, major_s: &str, minor: u64, minor_s: &str| {
        if minor > 0 {
            format!("{major}{major_s}{minor}{minor_s}")
        } else {
            format!("{major}{major_s}")
        }
    };
    match ms {
        0 => "0ms".to_owned(),
        1..SEC => format!("{ms}ms"),
        SEC..MIN => {
            let tenths = (ms as f64 / 100.0).round() as u64;
            if tenths < 600 {
                format!("{:.1}s", tenths as f64 / 10.0)
            } else {
                "1m".to_owned()
            }
        }
        MIN..HOUR => unit(ms / MIN, "m", (ms % MIN) / SEC, "s"),
        HOUR..DAY => unit(ms / HOUR, "h", (ms % HOUR) / MIN, "m"),
        _ => unit(ms / DAY, "d", (ms % DAY) / HOUR, "h"),
    }
}

// ─── Tool cards (tools-a/tools-b dispatch, review2 §11 #1) ─────────────────

/// Rebuild the stored card into a synthetic `tool_execution_*`-shaped call
/// payload for the family renderers.
fn tool_call_value(card: &ToolCard, expanded: bool) -> Value {
    let mut call = json!({
        "name": card.name,
        "arguments": card.args,
        "toolCallId": card.call_id,
    });
    if let Some(text) = &card.result_text {
        call["result"] = json!({
            "text": text,
            "isError": card.status == ToolStatus::Error,
            "details": card.details,
        });
        // The bash/eval families read the raw stream from the top-level
        // `output` key (`BashCard::parse`).
        if matches!(card.name.as_str(), "bash" | "eval") {
            call["output"] = json!(text);
        }
    }
    call["isPartial"] = json!(card.partial || !card.status.settled());
    call["expanded"] = json!(expanded);
    call
}

fn tool_card_lines(
    card: &ToolCard,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let call = tool_call_value(card, expanded);
    let lines = tool_card_body(card, &call, expanded, width, theme);
    // TS `ToolExecutionComponent` opens with a one-row Spacer (read cards
    // keep their own grouping, see `read_group_lines`).
    if card.name == "read" || lines.is_empty() {
        return lines;
    }
    std::iter::once(Line::default()).chain(lines).collect()
}

fn tool_card_body(
    card: &ToolCard,
    call: &Value,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    match card.name.as_str() {
        // Families with a native line API render their settled card
        // directly (edit: numbered unified diff; read: snippets).
        "edit" | "apply_patch" => {
            tools::edit::build_card_lines(call, call.get("result"), expanded, width, theme)
        }
        "read" => tools::read::render_frame(call, expanded, theme),
        "bash" => tools::bash::card_lines(call, width, theme),
        "grep" | "search" => tools::search::grep_lines(call, expanded, theme),
        "glob" => tools::search::glob_lines(call, expanded, theme),
        "find" => tools::find::find_lines(call, expanded, theme),
        "write" => tools::write::build_card_lines(call, call.get("result"), expanded, width, theme),
        _ => family_frame_lines(&card.name, &call, width, theme),
    }
}

/// Draw a family card through `ToolRenderer::render` onto a scratch
/// [`TestBackend`] and read the visible rows back as styled lines.
fn family_frame_lines(name: &str, call: &Value, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let width = width.max(8);
    let Ok(mut term) = Terminal::new(TestBackend::new(width as u16, CARD_MAX_ROWS)) else {
        return Vec::new();
    };
    let _ = term.draw(|frame| {
        let area = Rect::new(0, 0, width as u16, CARD_MAX_ROWS);
        tools::family(name).render(frame, area, call, theme);
    });
    buffer_lines(term.backend().buffer())
}

/// Extract visible rows from a scratch buffer: trailing blank rows dropped,
/// per-row trailing whitespace trimmed, runs of equally-styled cells folded
/// into spans.
fn buffer_lines(buf: &Buffer) -> Vec<Line<'static>> {
    let area = buf.area;
    let mut lines = Vec::new();
    for y in area.top()..area.bottom() {
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut cur_style: Option<Style> = None;
        let mut cur_text = String::new();
        for x in area.left()..area.right() {
            let cell = &buf[(x, y)];
            let style = cell.style();
            if cur_style != Some(style) {
                if let Some(prev) = cur_style.take() {
                    spans.push(Span::styled(std::mem::take(&mut cur_text), prev));
                }
                cur_style = Some(style);
            }
            cur_text.push_str(cell.symbol());
        }
        if let Some(prev) = cur_style.take() {
            spans.push(Span::styled(std::mem::take(&mut cur_text), prev));
        }
        while let Some(last) = spans.last_mut() {
            let trimmed = last.content.trim_end().to_owned();
            if trimmed.is_empty() && last.content.len() > trimmed.len() {
                // Pure-padding span: pop trailing spaces entirely.
                if spans.len() == 1 {
                    spans.pop();
                    break;
                }
                spans.pop();
                continue;
            }
            last.content = trimmed.into();
            break;
        }
        lines.push(Line::from(spans));
    }
    while lines.last().is_some_and(|l| l.spans.is_empty()) {
        lines.pop();
    }
    lines
}

// ─── Read group (chat.md §7.1) ─────────────────────────────────────────────

fn read_group_lines(
    group: &ReadGroup,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    if group.preview {
        return read_group_preview_lines(group, expanded, theme);
    }
    read_group_summary_lines(group, width, theme)
}

/// TS `ReadToolGroupComponent#updateDisplay` without content previews: one
/// read is ` ● Read <path>`; several are ` • Read (N)` over a `├─`/`└─`
/// tree (status glyph only for non-success rows). Attached usage rows nest
/// under their anchor read.
fn read_group_summary_lines(group: &ReadGroup, _width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let dim = Style::default().fg(theme.color("dim"));
    let title = Span::styled(
        "Read",
        Style::default()
            .fg(theme.color("toolTitle"))
            .add_modifier(Modifier::BOLD),
    );
    let usage_rows = |call_id: &str, prefix: &str, lines: &mut Vec<Line<'static>>| {
        for (anchor, usage) in &group.usages {
            if anchor == call_id {
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{}", usage_metrics(usage, theme)),
                    dim,
                )));
            }
        }
    };
    let mut lines = Vec::new();
    match group.reads.as_slice() {
        [] => lines.push(Line::from(vec![
            Span::raw(format!(" {} ", theme.symbol("format.bullet"))),
            title,
        ])),
        [read] => {
            let mut spans = vec![
                Span::raw(" "),
                read_status_span(read.status, theme),
                Span::raw(" "),
                title,
            ];
            let path = read_display_path(&read.path);
            if !path.is_empty() {
                spans.push(Span::raw(" "));
                spans.push(Span::styled(path, Style::default().fg(theme.color("text"))));
            }
            lines.push(Line::from(spans));
            usage_rows(&read.call_id, "   ", &mut lines);
        }
        reads => {
            lines.push(Line::from(vec![
                Span::raw(format!(" {} ", theme.symbol("format.bullet"))),
                title,
                Span::styled(format!(" ({})", reads.len()), dim),
            ]));
            for (index, read) in reads.iter().enumerate() {
                let last = index + 1 == reads.len();
                let connector = theme.symbol(if last { "tree.last" } else { "tree.branch" });
                let mut spans = vec![Span::raw("   "), Span::styled(connector, dim), Span::raw(" ")];
                if read.status != ToolStatus::Ok {
                    spans.push(read_status_span(read.status, theme));
                    spans.push(Span::raw(" "));
                }
                spans.push(Span::styled(
                    read_display_path(&read.path),
                    Style::default().fg(theme.color("text")),
                ));
                lines.push(Line::from(spans));
                let connector_width = unicode_width::UnicodeWidthStr::width(connector);
                let continuation = if last {
                    " ".repeat(connector_width)
                } else {
                    let vertical = theme.symbol("tree.vertical");
                    let pad = connector_width
                        .saturating_sub(unicode_width::UnicodeWidthStr::width(vertical));
                    format!("{vertical}{}", " ".repeat(pad))
                };
                usage_rows(&read.call_id, &format!("   {continuation} "), &mut lines);
            }
        }
    }
    lines
}

/// TS `#formatStatus`: `●` success (text), `⚠` warning, `✘` error,
/// `⏳` pending (dim).
fn read_status_span(status: ToolStatus, theme: &Theme) -> Span<'static> {
    match status {
        ToolStatus::Ok => Span::styled(
            theme.symbol("status.enabled"),
            Style::default().fg(theme.color("text")),
        ),
        ToolStatus::Error => Span::styled(
            theme.symbol("status.error"),
            Style::default().fg(theme.color("error")),
        ),
        ToolStatus::Pending | ToolStatus::Running => Span::styled(
            theme.symbol("status.pending"),
            Style::default().fg(theme.color("dim")),
        ),
    }
}

/// TS `shortenPath`: the home directory as `~`.
fn read_display_path(path: &str) -> String {
    if let Some(home) = std::env::var_os("HOME").and_then(|h| h.into_string().ok()) {
        if !home.is_empty() && home != "/" {
            if path == home {
                return "~".to_owned();
            }
            if let Some(rest) = path.strip_prefix(&home).filter(|r| r.starts_with('/')) {
                return format!("~{rest}");
            }
        }
    }
    path.to_owned()
}

/// `read.toolResultPreview`: each read as a snippet card (family renderer).
fn read_group_preview_lines(group: &ReadGroup, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let mut entries = Vec::new();
    let mut plain: Vec<Line<'static>> = Vec::new();
    for read in &group.reads {
        let mut frame = json!({
            "toolName": "read",
            "args": {"path": read.path},
        });
        if let Some(preview) = &read.preview {
            frame["result"] = json!({"text": preview});
        }
        match tools::read::ReadGroupEntry::from_frame(&frame) {
            Some(mut entry) => {
                // The stored settlement state wins over frame inference
                // (pending reads have no result to infer from).
                entry.status = match read.status {
                    ToolStatus::Pending | ToolStatus::Running => tools::read::ReadStatus::Pending,
                    ToolStatus::Ok => entry.status,
                    ToolStatus::Error => tools::read::ReadStatus::Error,
                };
                entries.push(entry);
            }
            None => {
                // URL / non-fs target: a plain head row + first preview line.
                let glyph = if read.status.settled() {
                    theme.glyph("success")
                } else {
                    "⧗"
                };
                plain.push(Line::from(vec![
                    Span::styled(glyph, Style::default().fg(theme.color("text_muted"))),
                    Span::raw(format!(" {}", read.path)),
                ]));
                if let Some(preview) = &read.preview {
                    plain.push(Line::from(Span::styled(
                        format!("    {}", preview.lines().next().unwrap_or("")),
                        Style::default().fg(theme.color("text_muted")),
                    )));
                }
            }
        }
    }
    let mut lines = tools::read::group_lines(&entries, expanded, theme);
    lines.extend(plain);
    // TS `#addPreviewUsage`: usage rows under the previews at paddingX 3.
    let dim = Style::default().fg(theme.color("dim"));
    for (_, usage) in &group.usages {
        lines.push(Line::from(Span::styled(
            format!("   {}", usage_metrics(usage, theme)),
            dim,
        )));
    }
    lines
}

// ─── Bash/python executions (chat.md §7.2) ─────────────────────────────────

fn exec_lines(exec: &Execution, tool: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut call = json!({
        "name": tool,
        "arguments": {
            "command": exec.command,
            "excludeFromContext": exec.excluded,
        },
        "output": exec.output,
        "isPartial": exec.status == ExecStatus::Running,
    });
    if exec.status != ExecStatus::Running {
        let mut details = if exec.details.is_object() {
            exec.details.clone()
        } else {
            json!({})
        };
        if details.get("exitCode").is_none() {
            details["exitCode"] = json!(exec.exit_code);
        }
        call["result"] = json!({
            "isError": exec.status == ExecStatus::Error,
            "details": details,
        });
    }
    let mut lines = if tool == "bash" {
        tools::bash::card_lines(&call, width, theme)
    } else {
        family_frame_lines(tool, &call, width, theme)
    };
    // `BashExecutionComponent` opens with a one-row Spacer as well.
    lines.insert(0, Line::default());
    if exec.excluded {
        for line in &mut lines {
            for span in &mut line.spans {
                span.style = Style::default().fg(theme.color("text_muted"));
            }
        }
    }
    lines
}

// ─── Dividers and notices (chat.md §10–11) ─────────────────────────────────

fn compaction_lines(
    c: &Compaction,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!("── {} ──", c.label()),
        Style::default()
            .fg(theme.color("text_muted"))
            .add_modifier(Modifier::ITALIC),
    ))];
    if let Some(warning) = &c.warning {
        lines.push(Line::from(Span::styled(
            format!("{} {warning}", theme.glyph("failure")),
            Style::default().fg(theme.color("warning")),
        )));
    }
    let summary = markdown::render(&c.summary, width as u16, theme);
    let cap = if expanded { usize::MAX } else { SUMMARY_COLLAPSED_ROWS };
    for line in summary.lines.into_iter().take(cap) {
        lines.push(line);
    }
    lines
}

fn branch_lines(summary: &str, expanded: bool, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        "⎇ branch",
        Style::default().fg(theme.color("text_muted")),
    ))];
    if expanded {
        lines.extend(markdown::render(summary, width as u16, theme).lines);
    } else {
        lines.push(Line::from(Span::styled(
            summary.lines().next().unwrap_or("").to_owned(),
            Style::default().fg(theme.color("text_muted")),
        )));
    }
    lines
}

fn custom_lines(c: &Custom, theme: &Theme) -> Vec<Line<'static>> {
    if !c.display {
        return Vec::new();
    }
    let mut lines = vec![Line::from(Span::styled(
        format!("[{}]", c.custom_type),
        Style::default()
            .fg(theme.color("accent"))
            .add_modifier(Modifier::ITALIC),
    ))];
    for line in c.content_text.lines() {
        lines.push(Line::from(Span::styled(
            line.to_owned(),
            Style::default().fg(theme.color("text")),
        )));
    }
    lines
}

fn notice_lines(text: &str, expanded: bool, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let trimmed = text.trim();
    if trimmed.to_ascii_lowercase().starts_with("error") {
        let body = trimmed.split_once(':').map_or(trimmed, |(_, rest)| rest).trim();
        return format_error_block(body, width, expanded, theme);
    }
    // Multi-line command output (`/session`, `/tools`, …): keep its line
    // breaks and wrap each line to the width.
    let style = Style::default().fg(theme.color("text_muted"));
    trimmed
        .lines()
        .flat_map(|l| {
            if l.is_empty() {
                vec![String::new()]
            } else {
                wrap_text(l, width)
            }
        })
        .map(|l| Line::from(Span::styled(l, style)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatRole, ChatEntry};

    fn theme() -> Theme {
        Theme::default()
    }

    fn rows(lines: &[Line<'static>]) -> Vec<String> {
        lines.iter().map(|l| l.to_string()).collect()
    }

    #[test]
    fn bash_tool_call_renders_card_rows() {
        let card = ToolCard {
            call_id: "t1".into(),
            name: "bash".into(),
            args: json!({"command": "cargo test -p omp-tui"}),
            status: ToolStatus::Ok,
            result_text: Some("test result: ok. 42 passed".into()),
            details: json!(null),
            partial: false,
        };
        let entry = ChatEntry::new_for_test(
            ChatRole::Tool,
            ChatKind::Tool(card),
        );
        let lines = entry_lines(&entry, true, 60, &theme());
        let rows = rows(&lines);
        assert!(rows.len() > 1, "bash card must be multiple rows: {rows:?}");
        let joined = rows.join("\n");
        assert!(joined.contains("cargo test -p omp-tui"), "{joined}");
        assert!(joined.contains("42 passed"), "{joined}");
    }

    #[test]
    fn edit_tool_call_renders_diff_rows() {
        let diff = "--- a/src/x.rs\n+++ b/src/x.rs\n@@ -1,2 +1,2 @@\n-old\n+new";
        let card = ToolCard {
            call_id: "t2".into(),
            name: "edit".into(),
            args: json!({
                "file_path": "src/x.rs",
                "old_string": "old",
                "new_string": "new",
            }),
            status: ToolStatus::Ok,
            result_text: Some(diff.into()),
            details: json!(null),
            partial: false,
        };
        let entry = ChatEntry::new_for_test(ChatRole::Tool, ChatKind::Tool(card));
        let lines = entry_lines(&entry, true, 60, &theme());
        let joined = rows(&lines).join("\n");
        assert!(joined.contains("src/x.rs"), "{joined}");
        assert!(joined.contains("new"), "{joined}");
    }

    #[test]
    fn assistant_markdown_heading_and_fence_render_as_rows() {
        let a = AssistantMsg {
            segments: vec![crate::chat::AssistantSegment::Text {
                text: "# Title\n\n```rust\nfn main() {}\n```\n".into(),
            }],
            stop_reason: Some("stop".into()),
            error_message: None,
            error_pinned: false,
            live: false,
        };
        let entry = ChatEntry::new_for_test(ChatRole::Assistant, ChatKind::Assistant(a));
        let lines = entry_lines(&entry, false, 60, &theme());
        let rows = rows(&lines);
        assert!(rows.iter().any(|r| r.contains("Title")), "{rows:?}");
        assert!(rows.iter().any(|r| r.contains("fn main()")), "{rows:?}");
    }

    #[test]
    fn thinking_block_renders_head_row() {
        let a = AssistantMsg {
            segments: vec![crate::chat::AssistantSegment::Thinking {
                text: "let me think".into(),
            }],
            stop_reason: None,
            error_message: None,
            error_pinned: false,
            live: false,
        };
        let entry = ChatEntry::new_for_test(ChatRole::Assistant, ChatKind::Assistant(a));
        let lines = entry_lines(&entry, false, 60, &theme());
        let rows = rows(&lines);
        assert!(
            rows.iter().any(|r| r.to_lowercase().contains("thought")),
            "{rows:?}"
        );
    }

    #[test]
    fn error_block_caps_at_eight_rows_with_prefix() {
        let message = (1..=20).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let lines = format_error_block(&message, 40, false, &theme());
        assert_eq!(lines.len(), ERROR_ROW_CAP + 1);
        // TS inline block: paddingX 1, `Error: ` on line 0, hanging indent.
        assert_eq!(lines[0].to_string(), " Error: line 1");
        assert_eq!(lines[1].to_string(), "   line 2");
        assert_eq!(
            lines.last().unwrap().to_string(),
            "   … +12 more lines (ctrl+o to expand)"
        );
        let expanded = format_error_block(&message, 40, true, &theme());
        assert_eq!(expanded.len(), 20);
    }

    #[test]
    fn error_banner_matches_ts_layout() {
        let message = "400 E2E-PROVIDER-FAILURE\nE2E-PROVIDER-FAILURE (type=invalid_request_error param=e2e_failure)\nraw-http-request=/tmp/h/.omp/logs/http-400-requests/1-x.json";
        let rows = rows(&error_banner_lines(message, 100, &theme()));
        assert_eq!(rows[0], "");
        assert_eq!(rows[1], "─".repeat(100));
        assert_eq!(rows[2], " ✘ 400 E2E-PROVIDER-FAILURE");
        assert_eq!(rows[3], "   E2E-PROVIDER-FAILURE (type=invalid_request_error param=e2e_failure)");
        assert_eq!(rows[4], "   raw-http-request=/tmp/h/.omp/logs/http-400-requests/1-x.json");
        assert_eq!(rows[5], " Dismissed when you send your next message.");
        assert_eq!(rows[6], "─".repeat(100));
        assert_eq!(rows.len(), 7);
    }

    #[test]
    fn pinned_error_is_suppressed_inline_until_expanded() {
        let a = AssistantMsg {
            segments: Vec::new(),
            stop_reason: Some("error".into()),
            error_message: Some("400 boom".into()),
            error_pinned: true,
            live: false,
        };
        let entry = ChatEntry::new_for_test(ChatRole::Assistant, ChatKind::Assistant(a));
        assert!(entry_lines(&entry, false, 60, &theme()).is_empty());
        let shown = rows(&entry_lines(&entry, true, 60, &theme()));
        assert_eq!(shown, vec!["".to_owned(), " Error: 400 boom".to_owned()]);
    }

    #[test]
    fn usage_row_matches_ts_metric_strip() {
        let usage = Usage {
            input: 900,
            output: 480,
            cache_read: 100,
            cache_write: 200,
            duration_ms: Some(12_000),
            ttft_ms: Some(400),
            timestamp_ms: None,
            turn_elapsed_ms: Some(6_200),
        };
        assert_eq!(
            usage_metrics(&usage, &theme()),
            "Δ 6.2s  ⤵ 1.1K  ⤴ 480  💾 100  ⏱ 0.4s  ⚡ 40.0/s"
        );
        let block = rows(&usage_block_lines(&usage, 120, &theme()));
        assert_eq!(block[0], "");
        assert!(block[1].starts_with(" Δ 6.2s  ⤵ 1.1K"), "{block:?}");
        // Instant responses carry no rate.
        let quick = Usage { duration_ms: Some(40), turn_elapsed_ms: None, ..usage };
        assert_eq!(usage_metrics(&quick, &theme()), "⤵ 1.1K  ⤴ 480  💾 100  ⏱ 0.4s");
    }

    #[test]
    fn number_and_duration_formats_match_pi_utils() {
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_000), "1K");
        assert_eq!(format_number(1_250), "1.3K");
        assert_eq!(format_number(19_400), "19K");
        assert_eq!(format_number(1_500_000), "1.5M");
        assert_eq!(format_duration(850), "850ms");
        assert_eq!(format_duration(59_960), "1m");
        assert_eq!(format_duration(192_000), "3m12s");
        assert_eq!(format_duration(7_500_000), "2h5m");
    }

    fn read_group(reads: &[(&str, &str, ToolStatus)], usages: Vec<(String, Usage)>) -> ChatEntry {
        ChatEntry::new_for_test(
            ChatRole::Tool,
            ChatKind::ReadGroup(ReadGroup {
                reads: reads
                    .iter()
                    .map(|(id, path, status)| crate::chat::ReadItem {
                        call_id: (*id).into(),
                        path: (*path).into(),
                        status: *status,
                        preview: Some("body".into()),
                    })
                    .collect(),
                usages,
                open: false,
                preview: false,
            }),
        )
    }

    #[test]
    fn read_group_summary_matches_ts_rows() {
        let usage = Usage { input: 100, output: 7, ..Usage::default() };
        let one = read_group(&[("r1", "e2e-read-me.txt", ToolStatus::Ok)], vec![("r1".into(), usage)]);
        assert_eq!(
            rows(&entry_lines(&one, false, 80, &theme())),
            vec![" ● Read e2e-read-me.txt".to_owned(), "   ⤵ 100  ⤴ 7".to_owned()]
        );
        let many = read_group(
            &[("r1", "a.rs", ToolStatus::Ok), ("r2", "b.rs", ToolStatus::Error)],
            vec![("r1".into(), usage)],
        );
        assert_eq!(
            rows(&entry_lines(&many, false, 80, &theme())),
            vec![
                " • Read (2)".to_owned(),
                "   ├─ a.rs".to_owned(),
                "   │  ⤵ 100  ⤴ 7".to_owned(),
                "   └─ ✘ b.rs".to_owned(),
            ]
        );
    }

    #[test]
    fn user_bubble_wraps_rows_with_background() {
        let entry = ChatEntry::new_for_test(
            ChatRole::User,
            ChatKind::User {
                text: "hello ![shot](file:///tmp/a.png) world".into(),
                synthetic: false,
                live_steered: false,
            },
        );
        let lines = entry_lines(&entry, false, 60, &theme());
        let joined = rows(&lines).join("\n");
        assert!(joined.contains("🖼 shot"), "{joined}");
        // Flush-left bubble on the `userMessageBg` fill: padding rows above
        // and below, one-space indent, rows padded to the full width.
        assert!(lines.iter().all(|l| {
            l.spans.iter().all(|s| s.style.bg == Some(theme().color("userMessageBg")))
        }));
        let plain = rows(&lines);
        assert!(plain.first().is_some_and(|r| r.trim().is_empty()));
        assert!(plain.last().is_some_and(|r| r.trim().is_empty()));
        assert!(plain.iter().any(|r| r.starts_with(" hello")), "{plain:?}");
        assert!(plain.iter().all(|r| unicode_width::UnicodeWidthStr::width(r.as_str()) == 60), "{plain:?}");
    }
}
