//! `todo` tool renderer — port of `pi-tui/src/tools/todo.ts` (the phased
//! task list that also feeds the status line).
//!
//! Pending (`todo.ts:505-530`) is ONE bare status line, no box: pending icon
//! plus `Todo <op>[ <task>][ N items]` per op. Result (`todo.ts:531-667`) is a
//! header done-glyph plus `closed/total` counts and phase-grouped task rows
//! (`renderTreeList`, `itemType:"todo"`). Collapsed keeps the last completed
//! task, the active one and the following pendings.
//!
//! [`status_summary`] exposes the `closed/total` counter for the status line.

use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Parsed phase list: `(name, tasks)` with `(content, status, blocker)`.
type Phases = Vec<(String, Vec<(String, String, Option<String>)>)>;

/// Collapsed preview cap (TS `PREVIEW_LIMITS.COLLAPSED_ITEMS`).
const COLLAPSED_ITEMS: usize = 8;

/// Renderer for the todo tool.
pub struct Todo;

/// Synthesize a `call` view from a tool-execution frame (`args` holds the
/// arguments; `toolName` is the fallback name).
/// Fallback line width when the frame carries none (chat pane minimum).
const LINE_WIDTH: usize = 100;

fn json_call(frame: &Value) -> Value {
    let name = frame.get("toolName").cloned().unwrap_or(Value::Null);
    let args = frame.get("args").cloned().unwrap_or(Value::Null);
    serde_json::json!({"name": name, "arguments": args})
}

impl ToolRenderer for Todo {
    fn summary(&self, call: &Value) -> String {
        pending_label(args(call))
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

    /// Frame seam: bare pending line before a result, task list after.
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

/// Tool-arguments accessor tolerant of both `arguments` and `input` shapes.
pub(crate) fn args(call: &Value) -> &Value {
    call.get("arguments")
        .or_else(|| call.get("input"))
        .unwrap_or(&Value::Null)
}

fn pending_label(a: &Value) -> String {
    let mut parts = vec!["Todo".to_string()];
    if let Some(op) = a.get("op").and_then(Value::as_str) {
        parts.push(sanitize(op));
    }
    for key in ["task", "phase"] {
        if let Some(v) = a.get(key).and_then(Value::as_str) {
            parts.push(sanitize(v));
        }
    }
    let n = items_in_args(a);
    if n > 0 {
        parts.push(format!("{n} items"));
    }
    parts.join(" ")
}

/// Count `items`/`tasks`/list entries in the raw args (init/batch shapes).
fn items_in_args(a: &Value) -> usize {
    for key in ["items", "tasks"] {
        if let Some(v) = a.get(key).and_then(Value::as_array) {
            return v.len();
        }
    }
    a.get("list").and_then(Value::as_array).map_or(0, |l| {
        l.iter()
            .map(|p| p.get("items").and_then(Value::as_array).map_or(0, Vec::len))
            .sum()
    })
}

/// Collapse control/format chars and whitespace (TS `forDisplay`).
pub(crate) fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_control() || c.is_whitespace() {
            if !out.ends_with(' ') {
                out.push(' ');
            }
        } else if !c.is_control() {
            out.push(c);
        }
    }
    out.trim().to_owned()
}

// ─── Pending ────────────────────────────────────────────────────────────────

/// One bare pending status line (no frame), TS `renderCall`.
#[must_use]
pub fn pending_lines(call: &Value, theme: &Theme, width: usize) -> Vec<Line<'static>> {
    let mut spans = vec![
        Span::styled("⧗ ".to_owned(), Style::new().fg(theme.color("text_muted"))),
        Span::styled(
            pending_label(args(call)),
            Style::new().fg(theme.color("text")),
        ),
    ];
    truncate_spans(&mut spans, width);
    vec![Line::from(spans)]
}

// ─── Result ────────────────────────────────────────────────────────────────

/// Phases as `[{"name":…,"tasks":[{"content","status","blocker"?}]}]`.
pub fn result_lines(
    _call: &Value,
    result: &Value,
    theme: &Theme,
    expanded: bool,
    width: usize,
) -> Vec<Line<'static>> {
    let details = result.get("details").unwrap_or(result);
    let phases: Phases = details
        .get("phases")
        .and_then(Value::as_array)
        .map(|ps| {
            ps.iter()
                .filter_map(|p| {
                    let name = p.get("name").and_then(Value::as_str)?.to_owned();
                    let tasks = p
                        .get("tasks")
                        .and_then(Value::as_array)
                        .map(|ts| {
                            ts.iter()
                                .filter_map(|t| {
                                    Some((
                                        t.get("content").and_then(Value::as_str)?.to_owned(),
                                        t.get("status").and_then(Value::as_str)?.to_owned(),
                                        t.get("blocker").and_then(Value::as_str).map(str::to_owned),
                                    ))
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    Some((name, tasks))
                })
                .collect()
        })
        .unwrap_or_default();

    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return vec![error_line(theme, width)];
    }

    let total: usize = phases.iter().map(|(_, ts)| ts.len()).sum();
    let closed_count = phases
        .iter()
        .flat_map(|(_, ts)| ts.iter())
        .filter(|(_, s, _)| is_closed(s))
        .count();
    let multi = phases.len() > 1;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{} ", theme.glyph("success")),
            Style::new().fg(theme.color("success")),
        ),
        Span::styled("Todo ", Style::new().fg(theme.color("accent")).bold()),
        Span::styled(
            format!("{closed_count}/{total}"),
            Style::new().fg(if total > 0 && closed_count == total {
                theme.color("success")
            } else {
                theme.color("text")
            }),
        ),
    ])];

    if total == 0 {
        lines.push(dim(theme, "Todo list is empty."));
        return lines;
    }

    let selection = collapsed_selection(&phases, expanded, COLLAPSED_ITEMS);
    let mut hidden = 0usize;
    for (pi, (name, tasks)) in phases.iter().enumerate() {
        if multi {
            let done = tasks.iter().filter(|(_, s, _)| is_closed(s)).count();
            lines.push(Line::from(Span::styled(
                format!("{name} ({done}/{})", tasks.len()),
                Style::new().fg(theme.color("accent")).bold(),
            )));
        }
        for (ti, (content, status, blocker)) in tasks.iter().enumerate() {
            if !expanded && !selection.contains(&(pi, ti)) {
                hidden += 1;
                continue;
            }
            let mut spans = vec![Span::raw("  ".to_owned())];
            spans.extend(todo_glyph(theme, status));
            spans.push(Span::raw(" ".to_owned()));
            spans.push(Span::styled(
                sanitize(content),
                Style::new().fg(theme.color(text_token(status))),
            ));
            if let Some(b) = blocker {
                spans.push(Span::styled(
                    format!(" ⛒ {b}"),
                    Style::new().fg(theme.color("warning")),
                ));
            }
            truncate_spans(&mut spans, width);
            lines.push(Line::from(spans));
        }
    }
    if hidden > 0 {
        lines.push(dim(
            theme,
            &format!("… {hidden} more tasks [ctrl+o: Expand]"),
        ));
    }
    lines
}

/// Status-line summary: `"3/7"`, or `"idle"` when no tasks exist.
#[must_use]
pub fn status_summary(result: &Value) -> String {
    let details = result.get("details").unwrap_or(result);
    let phases = details.get("phases").and_then(Value::as_array);
    let (closed, total) = phases.map_or((0, 0), |ps| {
        ps.iter()
            .filter_map(|p| p.get("tasks").and_then(Value::as_array))
            .flatten()
            .fold((0usize, 0usize), |(c, t), task| {
                let s = task.get("status").and_then(Value::as_str).unwrap_or("");
                (
                    c + usize::from(matches!(s, "completed" | "abandoned")),
                    t + 1,
                )
            })
    });
    if total == 0 {
        "idle".into()
    } else {
        format!("{closed}/{total}")
    }
}

fn collapsed_selection(phases: &Phases, expanded: bool, cap: usize) -> Vec<(usize, usize)> {
    if expanded {
        return (0..phases.len())
            .flat_map(|pi| (0..phases[pi].1.len()).map(move |ti| (pi, ti)))
            .collect();
    }
    let mut picked: Vec<(usize, usize)> = Vec::new();
    let flat: Vec<(usize, usize, &str)> = phases
        .iter()
        .enumerate()
        .flat_map(|(pi, (_, ts))| {
            ts.iter()
                .enumerate()
                .map(move |(ti, (_, s, _))| (pi, ti, s.as_str()))
        })
        .collect();
    // TS `selectCollapsedTodos`: last completed first, then the active
    // (in_progress) task, then the following pendings.
    if let Some((pi, ti, _)) = flat.iter().rev().find(|(_, _, s)| is_closed(s)) {
        picked.push((*pi, *ti));
    }
    if let Some(pos) = flat.iter().position(|(_, _, s)| *s == "in_progress") {
        let (pi, ti, _) = flat[pos];
        if !picked.contains(&(pi, ti)) {
            picked.push((pi, ti));
        }
        for (pi, ti, s) in flat.iter().skip(pos + 1) {
            if picked.len() >= cap {
                break;
            }
            if *s == "pending" && !picked.contains(&(*pi, *ti)) {
                picked.push((*pi, *ti));
            }
        }
    }
    picked
}

/// Truncate the final text span so the rendered line fits WIDTH columns
/// (approximated as char count; TS truncates with `truncateToWidth`).
fn truncate_spans(spans: &mut Vec<Span<'static>>, width: usize) {
    let total: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if total <= width {
        return;
    }
    let mut budget = width.saturating_sub(1);
    for span in spans.iter_mut() {
        let n = span.content.chars().count();
        if n > budget {
            let text: String = span.content.chars().take(budget).collect();
            span.content = format!("{text}…").into();
            budget = 0;
        } else {
            budget -= n;
        }
    }
}

fn is_closed(status: &str) -> bool {
    matches!(status, "completed" | "abandoned")
}

fn todo_glyph(theme: &Theme, status: &str) -> Vec<Span<'static>> {
    let (g, token) = match status {
        "completed" => (theme.glyph("success"), "success"),
        "in_progress" => ("◐", "accent"),
        "blocked" => ("⛒", "warning"),
        "abandoned" => (theme.glyph("failure"), "text_muted"),
        _ => ("○", "text_muted"),
    };
    vec![Span::styled(
        g.to_owned(),
        Style::new().fg(theme.color(token)),
    )]
}

fn text_token(status: &str) -> &'static str {
    match status {
        "completed" => "success",
        "in_progress" => "text",
        "blocked" => "warning",
        _ => "text_muted",
    }
}

fn dim(theme: &Theme, text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_owned(),
        Style::new().fg(theme.color("text_muted")),
    ))
}

fn error_line(theme: &Theme, width: usize) -> Line<'static> {
    let mut spans = vec![
        Span::styled(
            format!("{} ", theme.glyph("failure")),
            Style::new().fg(theme.color("error")),
        ),
        Span::styled(
            "Todo".to_owned(),
            Style::new().fg(theme.color("error")).bold(),
        ),
    ];
    truncate_spans(&mut spans, width);
    Line::from(spans)
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

    fn phases() -> Value {
        json!([
            {"name": "Setup", "tasks": [
                {"content": "clone repo", "status": "completed"},
                {"content": "install deps", "status": "in_progress"}
            ]},
            {"name": "Build", "tasks": [
                {"content": "cargo build", "status": "pending"}
            ]}
        ])
    }

    #[test]
    fn pending_is_one_bare_line_with_op_and_counts() {
        let t = theme();
        let call =
            json!({"arguments": {"op": "init", "list": [{"phase": "P", "items": ["a", "b"]}]}});
        let text = text_of(&pending_lines(&call, &t, 100));
        assert_eq!(text.len(), 1);
        assert_eq!(text[0], "⧗ Todo init 2 items");
    }

    #[test]
    fn result_header_shows_closed_over_total_and_phase_groups() {
        let t = theme();
        let result = json!({"details": {"op": "start", "phases": phases()}});
        let text = text_of(&result_lines(&json!({}), &result, &t, true, 100));
        assert!(text[0].contains("Todo 1/3"), "{text:?}");
        assert!(text.iter().any(|l| l == "Setup (1/2)"));
        assert!(text.iter().any(|l| l == "Build (0/1)"));
        assert!(text.iter().any(|l| l.contains("clone repo")));
        assert!(!text.iter().any(|l| l.contains("Expand")));
    }

    #[test]
    fn collapsed_keeps_active_and_next_pending_with_hint() {
        let t = theme();
        let result = json!({"details": {"op": "start", "phases": phases()}});
        let text = text_of(&result_lines(&json!({}), &result, &t, false, 100));
        // Collapsed keeps last completed + active + following pendings.
        assert!(text.iter().any(|l| l.contains("clone repo")), "{text:?}");
        assert!(text.iter().any(|l| l.contains("install deps")));
        assert!(text.iter().any(|l| l.contains("cargo build")));
        assert!(!text.iter().any(|l| l.contains("Expand")));
    }

    #[test]
    fn empty_list_renders_fallback() {
        let t = theme();
        let result = json!({"details": {"phases": []}});
        let text = text_of(&result_lines(&json!({}), &result, &t, false, 80));
        assert!(text.iter().any(|l| l.contains("Todo list is empty.")));
    }

    #[test]
    fn error_result_renders_error_icon() {
        let t = theme();
        let result = json!({"isError": true});
        let text = text_of(&result_lines(&json!({}), &result, &t, false, 80));
        assert_eq!(text[0], "✗ Todo");
    }

    #[test]
    fn blocked_task_shows_blocker_badge() {
        let t = theme();
        let result = json!({"details": {"phases": [{"name": "P", "tasks": [
            {"content": "wait for ci", "status": "blocked", "blocker": "flaky test"}
        ]}]}});
        let text = text_of(&result_lines(&json!({}), &result, &t, true, 100));
        assert!(text.iter().any(|l| l.contains("⛒ flaky test")));
    }

    #[test]
    fn status_summary_feeds_status_line() {
        assert_eq!(
            status_summary(&json!({"details": {"phases": phases()}})),
            "1/3"
        );
        assert_eq!(status_summary(&json!({"details": {"phases": []}})), "idle");
    }
}
