//! `write` tool renderer — port of `pi-tui/src/tools/write.ts` (P0).
//!
//! Covers the plain-file shapes: the pending streaming card (tail-anchored
//! gutter preview), the settled success card (first-lines preview with the
//! line-count suffix), and the error card. Internal-URL routing
//! (`proc://`, `cfg://`, `xd://` device writes) is out of scope for this
//! module until the URL host registry lands.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::Value;

use crate::theme::Theme;

use super::bash::replace_tabs;
use super::edit::{arg_str, args_of, phase_of, result_text, Phase};
use super::frame::{framed_card, lang_icon, status_header, token, CardSection};
use super::{error_detail_lines, ToolRenderer};

/// Rows shown in the collapsed settled preview (TS `WRITE_PREVIEW_LINES`).
pub const WRITE_PREVIEW_LINES: usize = 6;
/// Rows of the streaming tail window (TS `PREVIEW_LIMITS.EXPANDED_LINES`).
pub const WRITE_PREVIEW_LINES_EXPANDED: usize = 12;
/// Minimum gutter width (TS `WRITE_GUTTER_MIN_WIDTH`).
const GUTTER_MIN_WIDTH: usize = 3;

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

/// `formatLineCountSuffix`: dim ` · N lines`.
fn line_count_suffix(n: usize, theme: &Theme) -> Option<Span<'static>> {
    (n > 0).then(|| {
        Span::styled(
            format!(" · {n} line{}", if n == 1 { "" } else { "s" }),
            Style::default().fg(token(theme, "dim", "text_muted")),
        )
    })
}

/// Gutter + content row (`  1 alpha`; gutter width ≥ 3).
fn gutter_row(num: usize, width: usize, row: &str, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{num:>width$} "),
            Style::default().fg(token(theme, "dim", "text_muted")),
        ),
        Span::styled(
            replace_tabs(row),
            Style::default().fg(token(theme, "toolOutput", "text")),
        ),
    ])
}

fn dim_row(text: String, theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        text,
        Style::default().fg(token(theme, "dim", "text_muted")),
    ))
}

/// Streaming preview (TS `formatStreamingContent`): tail window of the last
/// [`WRITE_PREVIEW_LINES_EXPANDED`] rows (all when expanded), a dim
/// `… (N earlier lines)` row, and the trailing `… (streaming)` row.
fn streaming_lines(content: &str, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let rows: Vec<&str> = content.split('\n').collect();
    let total = rows.len();
    if total == 1 && rows[0].is_empty() {
        return Vec::new();
    }
    let start = if expanded {
        0
    } else {
        total.saturating_sub(WRITE_PREVIEW_LINES_EXPANDED)
    };
    let width = GUTTER_MIN_WIDTH.max(total.to_string().len());
    let mut out = Vec::new();
    if start > 0 {
        out.push(dim_row(
            format!(
                "… ({start} earlier line{})",
                if start == 1 { "" } else { "s" }
            ),
            theme,
        ));
    }
    for (i, row) in rows.iter().enumerate().skip(start) {
        out.push(gutter_row(i + 1, width, row, theme));
    }
    out.push(dim_row("… (streaming)".to_owned(), theme));
    out
}

/// Settled preview (TS `renderContentPreview`): the FIRST rows of the file,
/// collapsed to [`WRITE_PREVIEW_LINES`] with a `… N more lines ⟦ctrl+o: Expand⟧` row.
fn content_preview_lines(content: &str, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let rows: Vec<&str> = content.split('\n').collect();
    let total = rows.len();
    let shown = if expanded {
        total
    } else {
        total.min(WRITE_PREVIEW_LINES)
    };
    let width = GUTTER_MIN_WIDTH.max(total.to_string().len());
    let mut out: Vec<Line<'static>> = rows
        .iter()
        .take(shown)
        .enumerate()
        .map(|(i, row)| gutter_row(i + 1, width, row, theme))
        .collect();
    let hidden = total - shown;
    if !expanded && hidden > 0 {
        out.push(dim_row(
            format!(
                "… {hidden} more line{} {}ctrl+o: Expand{}",
                if hidden == 1 { "" } else { "s" },
                theme.symbol("format.bracketLeft"),
                theme.symbol("format.bracketRight"),
            ),
            theme,
        ));
    }
    out
}

/// Build the full framed write card. `result` is the settled `toolResult`
/// message when present. The previewed content is always the call's
/// `content` argument (TS `normalizeDisplayText(args.content)`).
#[must_use]
pub fn build_card_lines(
    call: &Value,
    result: Option<&Value>,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let phase = phase_of(call, result);
    let path = write_path(call).unwrap_or_default();
    let made_executable = result
        .and_then(|r| r.get("details"))
        .and_then(|d| d.get("madeExecutable").or_else(|| d.get("made_executable")))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let content = write_content(call).unwrap_or("");

    let accent = |text: String| Span::styled(text, Style::default().fg(theme.color("accent")));
    let muted = |text: String| {
        Span::styled(
            text,
            Style::default().fg(token(theme, "muted", "text_muted")),
        )
    };
    let path_span = if path.is_empty() {
        Span::styled(
            "…".to_owned(),
            Style::default().fg(token(theme, "toolOutput", "text")),
        )
    } else {
        accent(path.clone())
    };
    let mut description = vec![muted(format!("{} ", lang_icon(theme, &path))), path_span];

    let (icon, border) = match phase {
        Phase::Pending => (None, token(theme, "borderMuted", "text_muted")),
        Phase::Partial => (
            Some(Span::styled(
                theme.symbol("status.running").to_owned(),
                Style::default().fg(theme.color("accent")),
            )),
            token(theme, "borderMuted", "text_muted"),
        ),
        Phase::Success => (
            Some(Span::styled(
                theme.symbol("tool.write").to_owned(),
                Style::default().fg(theme.color("accent")),
            )),
            token(theme, "borderMuted", "text_muted"),
        ),
        Phase::Error => (
            Some(Span::styled(
                theme.symbol("status.error").to_owned(),
                Style::default().fg(theme.color("error")),
            )),
            theme.color("error"),
        ),
    };
    if matches!(phase, Phase::Success | Phase::Partial) {
        let lines = if content.is_empty() {
            0
        } else {
            content.split('\n').count()
        };
        description.extend(line_count_suffix(lines, theme));
        if phase == Phase::Success && made_executable {
            description.push(dim_row(" · ".to_owned(), theme).spans.remove(0));
            description.push(Span::styled(
                "made executable!".to_owned(),
                Style::default().fg(theme.color("success")),
            ));
        }
    }
    let header = status_header(theme, icon, "Write", "accent", description, Vec::new());

    let mut body: Vec<Line<'static>> = match phase {
        Phase::Pending | Phase::Partial => streaming_lines(content, expanded, theme),
        Phase::Success => {
            if content.is_empty() {
                Vec::new()
            } else {
                content_preview_lines(content, expanded, theme)
            }
        }
        Phase::Error => {
            let text = result.map(result_text).unwrap_or_default();
            if text.is_empty() {
                vec![Line::from(Span::styled(
                    "  Unknown error",
                    Style::default().fg(theme.color("error")),
                ))]
            } else {
                error_detail_lines(&text, width, theme)
            }
        }
    };
    if phase == Phase::Pending && content.is_empty() {
        body.clear();
    }
    let sections = if body.is_empty() {
        Vec::new()
    } else {
        vec![CardSection::plain(body)]
    };
    framed_card(Some(header), sections, width.max(8), border)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rows(lines: Vec<Line<'static>>) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
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
            content.push_str(&format!("line {i}\n"));
        }
        let call = json!({"name":"write","arguments":{"path":"out.txt","content":content}});
        let theme = Theme::default();
        let text = rows(build_card_lines(&call, None, false, 40, &theme)).join("\n");
        assert!(
            text.starts_with("╭─── Write: "),
            "no status icon while pending:\n{text}"
        );
        assert!(text.contains("earlier lines"), "got:\n{text}");
        assert!(text.contains(" 20 line 20"), "got:\n{text}");
        assert!(text.contains("… (streaming)"), "got:\n{text}");
        assert!(
            !text.contains("line 1\n"),
            "head hidden while streaming:\n{text}"
        );
    }

    #[test]
    fn success_card_title_row_and_preview() {
        let call =
            json!({"name":"write","arguments":{"path":"e2e-new.txt","content":"alpha\nbeta\n"}});
        let result = json!({"isError": false, "content":[{"type":"text","text":"Successfully wrote 11 bytes"}]});
        let theme = Theme::default();
        let t = rows(build_card_lines(&call, Some(&result), false, 60, &theme));
        assert!(
            t[0].starts_with("╭─── ✎ Write: ◦ e2e-new.txt · 3 lines "),
            "{:?}",
            t[0]
        );
        assert_eq!(t[1], format!("│   1 alpha{} │", " ".repeat(60 - 4 - 9)));
        assert!(t[2].starts_with("│   2 beta"), "{:?}", t[2]);
        assert!(t[3].starts_with("│   3 "), "{:?}", t[3]);
        assert!(t[4].starts_with("╰───") && t[4].ends_with('╯'));
        assert!(t.iter().all(|r| r.chars().count() == 60), "{t:?}");
    }

    #[test]
    fn error_card_shows_the_flattened_result_text() {
        let call = json!({"name":"write","arguments":{"path":"a.txt","content":"x"}});
        let result = json!({"isError": true, "text": "Tool call denied by user: write"});
        let theme = Theme::default();
        let text = rows(build_card_lines(&call, Some(&result), false, 60, &theme)).join("\n");
        assert!(
            text.contains("Tool call denied by user: write"),
            "got:\n{text}"
        );
        assert!(!text.contains("Unknown error"), "got:\n{text}");
    }

    #[test]
    fn long_file_collapses_with_more_lines_hint() {
        let content = "a\nb\nc\nd\ne\nf\ng\nh\n";
        let call = json!({"name":"write","arguments":{"path":"out.txt","content":content}});
        let result = json!({"isError": false, "content":[{"type":"text","text":"ok"}]});
        let theme = Theme::default();
        let text = rows(build_card_lines(&call, Some(&result), false, 60, &theme)).join("\n");
        assert!(text.contains("· 9 lines"), "got:\n{text}");
        assert!(
            text.contains("… 3 more lines ⟦ctrl+o: Expand⟧"),
            "got:\n{text}"
        );
        let expanded = rows(build_card_lines(&call, Some(&result), true, 60, &theme)).join("\n");
        assert!(!expanded.contains("more line"), "got:\n{expanded}");
        assert!(expanded.contains("  9 "), "got:\n{expanded}");
    }

    #[test]
    fn made_executable_suffix() {
        let call = json!({"name":"write","arguments":{"path":"run.sh","content":"#!/bin/sh\n"}});
        let result = json!({"isError": false, "details": {"madeExecutable": true},
                            "content":[{"type":"text","text":"ok"}]});
        let theme = Theme::default();
        let text = rows(build_card_lines(&call, Some(&result), false, 70, &theme)).join("\n");
        assert!(text.contains("made executable!"), "got:\n{text}");
    }

    #[test]
    fn error_card_red_detail() {
        let call = json!({"name":"write","arguments":{"path":"/x","content":"y"}});
        let result = json!({"isError": true,
            "content":[{"type":"text","text":"Error: permission denied"}]});
        let theme = Theme::default();
        let text = rows(build_card_lines(&call, Some(&result), false, 60, &theme)).join("\n");
        assert!(text.contains("permission denied"), "got:\n{text}");
        assert!(
            !text.contains("Error: permission denied"),
            "prefix stripped:\n{text}"
        );
        assert!(text.starts_with("╭─── ✗ Write: "), "got:\n{text}");
    }
}
