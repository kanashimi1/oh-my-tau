//! `ask` tool renderer — port of `pi-tui/src/tools/ask.ts` (§15 of
//! tools-a.md).
//!
//! Questions are normalized (`normalizeRenderQuestions`: bare strings become
//! labels, `\r` sanitized). Pending: framed-card-style rows — question rows
//! plus options as flat `•` bullets. Result: chosen answers with `✔` icons;
//! timed-out questions warn; errors render an error line.
//!
//! The interactive dialog itself is driven by `extension_ui.rs` (host
//! `AskUi`); this module only renders the transcript card.

use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for the ask tool.
pub struct Ask;

/// Fallback renderer for not-yet-styled tools.
pub struct Generic;

/// Synthesize a `call` view from a tool-execution frame (`args` holds the
/// arguments; `toolName` is the fallback name).
/// Fallback line width when the frame carries none (chat pane minimum).
const LINE_WIDTH: usize = 100;

fn json_call(frame: &Value) -> Value {
    let name = frame.get("toolName").cloned().unwrap_or(Value::Null);
    let args = frame.get("args").cloned().unwrap_or(Value::Null);
    serde_json::json!({"name": name, "arguments": args})
}

impl ToolRenderer for Ask {
    fn summary(&self, call: &Value) -> String {
        first_question(args(call)).unwrap_or_else(|| "Ask".to_owned())
    }

    fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        call: &Value,
        theme: &Theme,
    ) {
        super::render(frame, area, call, theme);
    }

    /// Frame seam: pending question card before a result, answers after.
    fn render_frame(
        &self,
        frame: &Value,
        expanded: bool,
        theme: &Theme,
    ) -> Vec<ratatui::text::Line<'static>> {
        let call = json_call(frame);
        match frame.get("result").filter(|r| !r.is_null()) {
            Some(result) => result_lines(&call, result, theme, expanded, LINE_WIDTH),
            None => pending_lines(&call, theme, LINE_WIDTH),
        }
    }
}

impl ToolRenderer for Generic {
    fn summary(&self, call: &Value) -> String {
        first_question(args(call)).unwrap_or_else(|| "Ask".to_owned())
    }

    fn render(
        &self,
        frame: &mut ratatui::Frame<'_>,
        area: ratatui::layout::Rect,
        call: &Value,
        theme: &Theme,
    ) {
        super::render(frame, area, call, theme);
    }
}

fn args(call: &Value) -> &Value {
    call.get("arguments")
        .or_else(|| call.get("input"))
        .unwrap_or(&Value::Null)
}

/// `questions` is the schema array; tolerate bare-string entries (TS
/// `normalizeRenderQuestions`) and sanitize `\r`-ridden text.
fn questions(a: &Value) -> Vec<(String, Vec<String>, bool)> {
    a.get("questions")
        .and_then(Value::as_array)
        .map(|qs| {
            qs.iter()
                .filter_map(|q| {
                    if let Some(s) = q.as_str() {
                        return Some((sanitize(s), Vec::new(), false));
                    }
                    let text = q.get("question").and_then(Value::as_str)?;
                    let options = q
                        .get("options")
                        .and_then(Value::as_array)
                        .map(|os| {
                            os.iter()
                                .filter_map(|o| {
                                    o.as_str().map(sanitize).or_else(|| {
                                        o.get("label").and_then(Value::as_str).map(sanitize)
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let multi = q.get("multi").and_then(Value::as_bool).unwrap_or(false);
                    Some((sanitize(text), options, multi))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn first_question(a: &Value) -> Option<String> {
    questions(a)
        .into_iter()
        .next()
        .map(|(q, _, _)| truncate_str(&q, 60))
}

/// Collapse control/format chars to spaces (TS `forDisplay`; also drops
/// `\r`-overwrite segments by keeping only their final text).
fn sanitize(s: &str) -> String {
    let last = s.rsplit('\r').next().unwrap_or(s);
    let mut out = String::with_capacity(last.len());
    for c in last.chars() {
        if c.is_control() {
            if !out.ends_with(' ') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out.trim().to_owned()
}

fn truncate_str(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

// ─── Pending ────────────────────────────────────────────────────────────────

/// Pending card rows: `⧗ Ask` head, per-question rows with `•` option
/// bullets (flat, TS `renderQuestionOptionLines`).
#[must_use]
pub fn pending_lines(call: &Value, theme: &Theme, width: usize) -> Vec<Line<'static>> {
    let mut lines = vec![head_line(theme, "Ask", "⧗", "text_muted", "text")];
    for (q, options, multi) in questions(args(call)) {
        let mode = if multi { " [multi]" } else { "" };
        lines.push(Line::from(Span::styled(
            format!("  {}{mode}", truncate_str(&q, width.saturating_sub(12))),
            Style::new().fg(theme.color("text")),
        )));
        for o in options {
            lines.push(Line::from(Span::styled(
                format!("    • {}", truncate_str(&o, width.saturating_sub(10))),
                Style::new().fg(theme.color("text_muted")),
            )));
        }
    }
    lines
}

// ─── Result ────────────────────────────────────────────────────────────────

/// Result card rows: `✔ <question>` + selected option lines per question
/// (TS `details.results[]`, built by omp-cli `result_json`).
pub fn result_lines(
    _call: &Value,
    result: &Value,
    theme: &Theme,
    _expanded: bool,
    width: usize,
) -> Vec<Line<'static>> {
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        let reason = result
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("ask failed");
        return vec![Line::from(vec![
            Span::styled(
                format!("{} ", theme.glyph("failure")),
                Style::new().fg(theme.color("error")),
            ),
            Span::styled(
                format!("Ask — {}", truncate_str(reason, width.saturating_sub(8))),
                Style::new().fg(theme.color("error")),
            ),
        ])];
    }

    let d = result.get("details").unwrap_or(result);
    let chat_redirect = d.get("chatRedirect").is_some();
    if chat_redirect {
        return vec![Line::from(Span::styled(
            "ℹ Ask [chat redirect]".to_owned(),
            Style::new().fg(theme.color("text_muted")),
        ))];
    }

    let mut lines = vec![head_line(
        theme,
        "Ask",
        theme.glyph("success"),
        "success",
        "accent",
    )];
    let results = d.get("results").and_then(Value::as_array);
    match results {
        None => lines.push(Line::from(Span::styled(
            "  (no answers recorded)".to_owned(),
            Style::new().fg(theme.color("text_muted")),
        ))),
        Some(rs) => {
            for r in rs {
                let q = r.get("question").and_then(Value::as_str).unwrap_or("");
                lines.push(answer_row(theme, q, width));
                let selected = r.get("selectedOptions").and_then(Value::as_array);
                let custom = r
                    .get("customInput")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty());
                let timed_out = r.get("timedOut").and_then(Value::as_bool).unwrap_or(false);
                let any = selected.map_or(0, Vec::len) + usize::from(custom.is_some());
                if timed_out && any == 0 {
                    lines.push(Line::from(Span::styled(
                        format!(
                            "    {} timed out — no answer",
                            theme.symbol("status.warning")
                        ),
                        Style::new().fg(theme.color("warning")),
                    )));
                    continue;
                }
                for s in selected.map_or(vec![], |s| {
                    s.iter()
                        .map(|v| {
                            sanitize(&match v {
                                Value::String(t) => t.clone(),
                                other => other.to_string(),
                            })
                        })
                        .collect::<Vec<_>>()
                }) {
                    lines.push(Line::from(Span::styled(
                        format!("    → {}", truncate_str(&s, width.saturating_sub(8))),
                        Style::new().fg(theme.color("success")),
                    )));
                }
                if let Some(c) = custom {
                    lines.push(Line::from(Span::styled(
                        format!("    → {}", truncate_str(c, width.saturating_sub(8))),
                        Style::new().fg(theme.color("accent")),
                    )));
                }
            }
        }
    }
    lines
}

fn answer_row(theme: &Theme, question: &str, width: usize) -> Line<'static> {
    Line::from(Span::styled(
        format!(
            "  {} {}",
            theme.symbol("status.success"),
            truncate_str(&sanitize(question), width.saturating_sub(8))
        ),
        Style::new().fg(theme.color("text")),
    ))
}

fn head_line<'a>(
    theme: &Theme,
    title: &'a str,
    icon: &'a str,
    icon_token: &'a str,
    title_token: &'a str,
) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{icon} "), Style::new().fg(theme.color(icon_token))),
        Span::styled(
            title.to_owned(),
            Style::new().fg(theme.color(title_token)).bold(),
        ),
    ])
}

/// Status-line summary for an ask result: selected count or `asked`.
#[must_use]
pub fn status_summary(result: &Value) -> String {
    let d = result.get("details").unwrap_or(result);
    let n = d.get("results").and_then(Value::as_array).map_or(0, |rs| {
        rs.iter()
            .map(|r| {
                r.get("selectedOptions")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len)
                    + usize::from(
                        r.get("customInput")
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty()),
                    )
            })
            .sum::<usize>()
    });
    if n == 0 {
        "asked".into()
    } else {
        format!("{n} answered")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn theme() -> Theme {
        Theme::default()
    }

    fn text_of(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn pending_lists_questions_and_option_bullets() {
        let t = theme();
        let call = json!({"arguments": {"questions": [
            {"id": "q1", "question": "Deploy now?", "options": [
                {"label": "Yes"}, {"label": "No"}
            ], "multi": false}
        ]}});
        let text = text_of(&pending_lines(&call, &t, 100));
        assert_eq!(text[0], "⧗ Ask");
        assert!(text[1].contains("Deploy now?"));
        assert_eq!(text[2], "    • Yes");
        assert_eq!(text[3], "    • No");
    }

    #[test]
    fn pending_normalizes_bare_string_questions() {
        let t = theme();
        let call = json!({"arguments": {"questions": ["Ship it?\rold"]}});
        let text = text_of(&pending_lines(&call, &t, 100));
        // \r overwrite collapses to the final segment (TS sanitizeDisplayLines).
        assert!(text[1].contains("old"), "{text:?}");
        assert!(!text[1].contains("Ship it?"), "{text:?}");
    }

    #[test]
    fn result_shows_checkmarks_and_selections() {
        let t = theme();
        let call = json!({"arguments": {"questions": [{"id": "q1", "question": "Deploy now?"}]}});
        let result = json!({"details": {"results": [
            {"id": "q1", "question": "Deploy now?", "selectedOptions": ["Yes"], "multi": false}
        ]}});
        let text = text_of(&result_lines(&call, &result, &t, false, 100));
        assert!(text[1].contains("✓ Deploy now?"));
        assert_eq!(text[2], "    → Yes");
    }

    #[test]
    fn result_warns_on_timeout_and_renders_custom_input() {
        let t = theme();
        let call = json!({"arguments": {"questions": [{"id": "q1", "question": "Pick"}]}});
        let result = json!({"details": {"results": [
            {"id": "q1", "question": "Pick", "selectedOptions": [], "timedOut": true},
            {"id": "q2", "question": "Other", "selectedOptions": [], "customInput": "my own"}
        ]}});
        let text = text_of(&result_lines(&call, &result, &t, false, 100));
        assert!(text.iter().any(|l| l.contains("timed out")));
        assert!(text.iter().any(|l| l == "    → my own"));
    }

    #[test]
    fn result_renders_error_and_chat_redirect() {
        let t = theme();
        let call = json!({"arguments": {}});
        let err = text_of(&result_lines(
            &call,
            &json!({"isError": true, "error": "cancelled"}),
            &t,
            false,
            100,
        ));
        assert!(err[0].contains("Ask — cancelled"));
        let redir = text_of(&result_lines(
            &call,
            &json!({"details": {"chatRedirect": true}}),
            &t,
            false,
            100,
        ));
        assert_eq!(redir[0], "ℹ Ask [chat redirect]");
    }

    #[test]
    fn summary_uses_first_question() {
        let call = json!({"arguments": {"questions": [{"id": "a", "question": "First?"}, {"id": "b", "question": "Second?"}]}});
        assert_eq!(Ask.summary(&call), "First?");
        assert_eq!(Generic.summary(&call), "First?");
    }

    #[test]
    fn status_summary_counts_selections() {
        let result = json!({"details": {"results": [
            {"selectedOptions": ["a", "b"]},
            {"customInput": "c"}
        ]}});
        assert_eq!(status_summary(&result), "3 answered");
        assert_eq!(
            status_summary(&json!({"details": {"results": []}})),
            "asked"
        );
    }
}
