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
        ChatKind::Usage(usage) => vec![usage_row(usage, theme)],
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
            RevealBlock::Text(text) => lines.extend(markdown::render(text, width as u16, theme).lines),
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
    let chipped = collapse_chips(text);
    let mut text = markdown::render(&chipped, width as u16, theme);
    let bg = theme.color("border");
    for line in &mut text.lines {
        for span in &mut line.spans {
            let mut style = span.style;
            style.bg = Some(bg);
            if synthetic {
                style.fg = Some(theme.color("text_muted"));
            }
            span.style = style;
        }
    }
    let mut lines = text.lines;
    let bg_style = Style::default().fg(theme.color("accent")).bg(bg);
    if let Some(first) = lines.first_mut() {
        let mut spans = vec![Span::styled(format!("{} ", theme.glyph("prompt")), bg_style)];
        if live_steered {
            spans.push(Span::styled(
                "* ",
                Style::default().fg(theme.color("warning")).bg(bg),
            ));
        }
        spans.append(&mut first.spans);
        *first = Line::from(spans);
    } else {
        lines.push(Line::from(Span::styled(
            format!("{} ", theme.glyph("prompt")),
            bg_style,
        )));
    }
    lines
}

// ─── Assistant (chat.md §3) ────────────────────────────────────────────────

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
                lines.extend(markdown::render(text, width as u16, theme).lines);
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
    if let Some(error) = &a.error_message {
        lines.extend(format_error_block(error, width, expanded, theme));
    }
    if let Some(usage) = &a.usage {
        lines.push(usage_row(usage, theme));
    }
    lines
}

/// `chrome/error-block.ts` port: wrap at `width`, cap at
/// [`ERROR_ROW_CAP`] rows unless `expanded`, `Error:` prefix on the first
/// row, dim `… +N more lines (ctrl+o to expand)` trailer when capped.
#[must_use]
pub fn format_error_block(
    message: &str,
    width: usize,
    expanded: bool,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let width = width.max(CONTINUATION_INDENT.len() + 8);
    let mut logical: Vec<String> = message
        .replace("\r\n", "\n")
        .split('\n')
        .map(|l| l.trim().split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect();
    if logical.is_empty() {
        logical.push("Unknown error".to_owned());
    }
    let wrap_width = width - CONTINUATION_INDENT.len();
    let mut rows: Vec<String> = Vec::new();
    for line in &logical {
        for row in wrap_text(line, wrap_width) {
            rows.push(if rows.is_empty() {
                row
            } else {
                format!("{CONTINUATION_INDENT}{row}")
            });
        }
    }
    let cap = if expanded { usize::MAX } else { ERROR_ROW_CAP };
    let hidden = rows.len().saturating_sub(cap);
    rows.truncate(cap);
    let mut lines: Vec<Line<'static>> = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            let spans = if index == 0 {
                vec![
                    Span::styled(
                        format!("{} Error: ", theme.glyph("failure")),
                        Style::default().fg(theme.color("error")),
                    ),
                    Span::styled(row, Style::default().fg(theme.color("error"))),
                ]
            } else {
                vec![Span::styled(
                    row,
                    Style::default().fg(theme.color("error")),
                )]
            };
            Line::from(spans)
        })
        .collect();
    if hidden > 0 {
        lines.push(Line::from(Span::styled(
            format!(
                "{CONTINUATION_INDENT}… +{hidden} more line{} (ctrl+o to expand)",
                if hidden == 1 { "" } else { "s" }
            ),
            Style::default().fg(theme.color("text_muted")),
        )));
    }
    lines
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

/// The settled-turn usage line: duration · tok/s · ttft · token totals.
#[must_use]
pub fn usage_row(usage: &Usage, theme: &Theme) -> Line<'static> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(duration_ms) = usage.duration_ms.filter(|d| *d > 0) {
        parts.push(format!("{:.1}s", duration_ms as f64 / 1000.0));
        // Below 100 ms the rate is nonsense (cached/instant responses).
        if duration_ms > 100 && usage.output > 0 {
            parts.push(format!(
                "{:.0} tok/s",
                usage.output as f64 * 1000.0 / duration_ms as f64
            ));
        }
    }
    if let Some(ttft_ms) = usage.ttft_ms.filter(|t| *t > 0) {
        parts.push(format!("ttft {:.2}s", ttft_ms as f64 / 1000.0));
    }
    let total = usage.input + usage.output + usage.cache_read + usage.cache_write;
    if total > 0 {
        parts.push(format!("{} tok", compact(total)));
    }
    Line::from(Span::styled(
        format!("⏱ {} this turn", parts.join(" · ")),
        Style::default().fg(theme.color("text_muted")),
    ))
}

/// `formatNumber`-style compact amounts (`19K`, `1.2M`).
#[allow(clippy::cast_precision_loss)] // display-only rounding of token counts
fn compact(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
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
    match card.name.as_str() {
        // Families with a native line API render their settled card
        // directly (edit: numbered unified diff; read: snippets).
        "edit" | "apply_patch" => {
            tools::edit::build_card_lines(&call, call.get("result"), expanded, width, theme)
        }
        "read" => tools::read::render_frame(&call, expanded, theme),
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
    _width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
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
    // Turn usage attaches beneath the group when the turn was reads-only
    // (TS `ReadToolGroupComponent.attachUsage`).
    if let Some(usage) = &group.usage {
        lines.push(usage_row(usage, theme));
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
        call["result"] = json!({
            "isError": exec.status == ExecStatus::Error,
            "details": {"exitCode": exec.exit_code},
        });
    }
    let mut lines = family_frame_lines(tool, &call, width, theme);
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
            usage: None,
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
            usage: None,
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
        assert!(lines.len() <= ERROR_ROW_CAP + 1);
        assert!(lines[0].to_string().starts_with('✗'));
        assert!(lines[0].to_string().contains("Error: line 1"));
        assert!(lines.last().unwrap().to_string().contains("more line"));
        let expanded = format_error_block(&message, 40, true, &theme());
        assert!(expanded.len() > lines.len());
    }

    #[test]
    fn usage_row_shows_rate_duration_and_ttft() {
        let usage = Usage {
            input: 900,
            output: 480,
            cache_read: 100,
            cache_write: 0,
            duration_ms: Some(12_000),
            ttft_ms: Some(400),
        };
        let line = usage_row(&usage, &theme());
        let s = line.to_string();
        assert!(s.contains("12.0s"), "{s}");
        assert!(s.contains("40 tok/s"), "{s}");
        assert!(s.contains("ttft 0.40s"), "{s}");
        assert!(s.contains("1K tok"), "{s}");
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
        assert!(lines.iter().all(|l| {
            l.spans.iter().all(|s| s.style.bg == Some(theme().color("border")))
        }));
    }
}
