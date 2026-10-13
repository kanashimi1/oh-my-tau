//! `write` tool renderer — port of `pi-tui/src/tools/write.ts` (P0).
//!
//! Covers the plain-file shapes: the pending streaming card (tail-anchored
//! gutter preview), the settled success card (first-lines preview with the
//! line-count suffix), and the error card. Internal-URL routing
//! (`proc://`, `cfg://`, `xd://` device writes) is out of scope for this
//! module until the URL host registry lands.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::Value;

use crate::theme::Theme;

use super::edit::{
    arg_str, args_of, error_detail_lines, frame_card, phase_of, result_text,
    streaming_preview_lines, truncate_width, Phase,
};
use super::ToolRenderer;

/// Rows shown in the collapsed settled preview (TS `WRITE_PREVIEW_LINES`).
pub const WRITE_PREVIEW_LINES: usize = 6;
/// Rows shown in the expanded settled preview.
pub const WRITE_PREVIEW_LINES_EXPANDED: usize = 12;

/// Renderer for the `write` tool.
pub struct Write;

impl ToolRenderer for Write {
    fn summary(&self, call: &Value) -> String {
        let path = write_path(call).unwrap_or_else(|| "…".into());
        format!("Write {path}")
    }

    fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        call: &Value,
        theme: &Theme,
    ) {
        let width = area.width as usize;
        let lines = build_card_lines(call, None, false, width, theme);
        frame.render_widget(Paragraph::new(lines), area);
    }
}

/// Target path of a write call (`file_path`, then `path`).
#[must_use]
pub fn write_path(call: &Value) -> Option<String> {
    arg_str(args_of(call), &["file_path", "filePath", "path"]).map(str::to_owned)
}

/// Full content of a write call (only present once args are complete).
fn write_content(call: &Value) -> Option<&str> {
    arg_str(args_of(call), &["content"])
}

/// Header row. Pending carries NO status icon (liveness rides the trailing
/// streaming content); settled success gets the success glyph and the
/// line-count suffix; errors get the failure glyph.
fn header_line(
    path: &str,
    phase: Phase,
    line_count: Option<usize>,
    made_executable: bool,
    width: usize,
    theme: &Theme,
) -> Line<'static> {
    let inner = width.saturating_sub(4).max(1);
    let (icon, icon_color) = match phase {
        Phase::Pending => ("", theme.color("text")),
        Phase::Partial => ("◐", theme.color("accent")),
        Phase::Success => (theme.glyph("success"), theme.color("success")),
        Phase::Error => (theme.glyph("failure"), theme.color("error")),
    };
    let suffix = match (line_count, made_executable) {
        (Some(n), true) => format!(" ({n} lines) · made executable!"),
        (Some(n), false) => format!(" ({n} lines)"),
        (None, true) => " · made executable!".to_owned(),
        (None, false) => String::new(),
    };
    let prefix_width =
        icon.chars().count() + usize::from(!icon.is_empty()) + 1 + suffix.chars().count();
    let path_width = inner.saturating_sub(prefix_width + 6);
    let mut spans = Vec::new();
    if !icon.is_empty() {
        spans.push(Span::styled(
            format!("{icon} "),
            Style::default().fg(icon_color),
        ));
    }
    spans.push(Span::styled(
        "Write ",
        Style::default().fg(theme.color("text")),
    ));
    spans.push(Span::styled(
        truncate_width(path, path_width),
        Style::default().fg(theme.color("accent")),
    ));
    spans.push(Span::styled(
        suffix,
        Style::default()
            .fg(theme.color("success"))
            .add_modifier(Modifier::BOLD),
    ));
    Line::from(spans)
}

/// Settled preview: the FIRST rows of the file (TS `renderContentPreview`),
/// collapsed to [`WRITE_PREVIEW_LINES`] with a `… N more lines` hint.
fn content_preview_lines(
    content: &str,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let inner = width.saturating_sub(4).max(1);
    let rows: Vec<&str> = content.lines().collect();
    let budget = if expanded {
        WRITE_PREVIEW_LINES_EXPANDED
    } else {
        WRITE_PREVIEW_LINES
    };
    let mut lines = Vec::new();
    let gutter = rows.len().to_string().len().max(2);
    for (i, row) in rows.iter().enumerate().take(budget) {
        let gutter_text = format!("{:>gutter$} ", i + 1);
        lines.push(Line::from(vec![
            Span::styled(gutter_text, Style::default().fg(theme.color("text_muted"))),
            Span::raw(truncate_width(row, inner.saturating_sub(gutter + 1))),
        ]));
    }
    let remaining = rows.len().saturating_sub(budget);
    if remaining > 0 {
        let hint = if expanded {
            format!("… {remaining} more lines")
        } else {
            format!("… {remaining} more lines [ctrl+o: Expand]")
        };
        lines.push(Line::from(Span::styled(
            truncate_width(&hint, inner),
            Style::default().fg(theme.color("text_muted")),
        )));
    }
    lines
}

/// Build the full write card. `result` is the settled `toolResult` message
/// when present.
#[must_use]
pub fn build_card_lines(
    call: &Value,
    result: Option<&Value>,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    let phase = phase_of(call, result);
    let path = write_path(call).unwrap_or_else(|| "…".into());
    let made_executable = result
        .and_then(|r| r.get("details"))
        .and_then(|d| d.get("madeExecutable").or_else(|| d.get("made_executable")))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // Content: args.content while streaming; settled result content wins
    // (the file as actually written, notices stripped by the host).
    let content = match phase {
        Phase::Pending | Phase::Partial => write_content(call).map(str::to_owned),
        Phase::Success | Phase::Error => result
            .map(result_text)
            .filter(|t| !t.is_empty())
            .or_else(|| write_content(call).map(str::to_owned)),
    };
    let line_count = content.as_ref().map(|c| c.lines().count());

    let mut body = vec![header_line(
        &path,
        phase,
        line_count,
        made_executable,
        width,
        theme,
    )];
    let mut border_token = "border";
    match phase {
        Phase::Pending => match &content {
            Some(content) => body.extend(streaming_preview_lines(
                content,
                super::edit::PREVIEW_TAIL_LINES,
                width,
                theme,
                true,
            )),
            None => body.push(Line::from(Span::styled(
                "…",
                Style::default().fg(theme.color("text_muted")),
            ))),
        },
        Phase::Partial => {
            if let Some(content) = &content {
                body.extend(streaming_preview_lines(
                    content,
                    super::edit::PREVIEW_TAIL_LINES,
                    width,
                    theme,
                    true,
                ));
            }
        }
        Phase::Error => {
            border_token = "error";
            let text = result.map(result_text).unwrap_or_default();
            if text.is_empty() {
                body.push(Line::from(Span::styled(
                    "  Unknown error",
                    Style::default().fg(theme.color("error")),
                )));
            } else {
                body.extend(error_detail_lines(&text, width, theme));
            }
        }
        Phase::Success => {
            if let Some(content) = &content {
                body.extend(content_preview_lines(content, expanded, width, theme));
            }
        }
    }
    frame_card(body, width, theme, border_token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn render_to_string(lines: Vec<Line<'static>>, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                let area = ratatui::layout::Rect::new(0, 0, width, height);
                f.render_widget(Paragraph::new(lines), area);
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..height {
            for x in 0..width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn summary_shows_path() {
        let call = json!({"name":"write","arguments":{"path":"src/lib.rs"}});
        assert_eq!(Write.summary(&call), "Write src/lib.rs");
    }

    #[test]
    fn pending_card_streams_tail_with_gutter() {
        let mut content = String::new();
        for i in 1..=20 {
            content.push_str("line ");
            content.push_str(&i.to_string());
            content.push('\n');
        }
        let call = json!({"name":"write","arguments":{"path":"out.txt","content":content}});
        let theme = Theme::default();
        let lines = build_card_lines(&call, None, false, 40, &theme);
        let text = render_to_string(lines, 40, 20);
        assert!(text.contains("Write out.txt"), "got:\n{text}");
        assert!(text.contains("earlier lines"), "got:\n{text}");
        assert!(text.contains("20 line"), "got:\n{text}");
        assert!(text.contains("(streaming)"), "got:\n{text}");
        assert!(
            !text.contains("line 1\n"),
            "head should not render while streaming:\n{text}"
        );
    }

    #[test]
    fn success_card_shows_first_lines_and_count() {
        let content = "a\nb\nc\nd\ne\nf\ng\nh\n";
        let call = json!({"name":"write","arguments":{"path":"out.txt","content":content}});
        let result = json!({"isError": false, "content":[{"type":"text","text":content}]});
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 16);
        assert!(text.contains("Write out.txt (8 lines)"), "got:\n{text}");
        assert!(text.contains("a "), "got:\n{text}");
        assert!(text.contains("more lines [ctrl+o: Expand]"), "got:\n{text}");
    }
    #[test]
    fn expanded_success_shows_twelve_and_count_hint() {
        let mut content = String::new();
        for i in 1..=30 {
            content.push('l');
            content.push_str(&i.to_string());
            content.push('\n');
        }
        let call = json!({"name":"write","arguments":{"path":"o.txt","content":content}});
        let result = json!({"isError": false, "content":[{"type":"text","text":content}]});
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), true, 40, &theme);
        let text = render_to_string(lines, 40, 20);
        assert!(text.contains("l12"), "got:\n{text}");
        assert!(text.contains("18 more lines"), "got:\n{text}");
    }

    #[test]
    fn made_executable_suffix() {
        let call = json!({"name":"write","arguments":{"path":"run.sh","content":"#!/bin/sh\n"}});
        let result = json!({"isError": false, "details": {"madeExecutable": true},
                            "content":[{"type":"text","text":"#!/bin/sh\n"}]});
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 8);
        assert!(text.contains("made executable!"), "got:\n{text}");
    }

    #[test]
    fn error_card_red_detail() {
        let call = json!({"name":"write","arguments":{"path":"/x","content":"y"}});
        let result = json!({"isError": true,
            "content":[{"type":"text","text":"Error: permission denied"}]});
        let theme = Theme::default();
        let lines = build_card_lines(&call, Some(&result), false, 40, &theme);
        let text = render_to_string(lines, 40, 8);
        assert!(text.contains("permission denied"), "got:\n{text}");
        assert!(
            !text.contains("Error: permission denied"),
            "prefix stripped:\n{text}"
        );
    }
}
