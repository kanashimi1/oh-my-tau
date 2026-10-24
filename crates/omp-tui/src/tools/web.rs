//! `web_search`/`fetch` renderer — port of `pi-tui/src/tools/web-search.ts`
//! (§7 of tools-b.md) and fetch.ts (§9 of tools-a.md, `Read <url>` cards).
//!
//! Pending: `⧗ Web Search <query trunc 80>` / `⧗ Read <url>`. Result: head
//! glyph + `provider · N sources` meta, `Answer:` section (markdown rendered
//! as styled text), `Sources:` rows (accent title + dim domain + muted age),
//! `Metadata:` usage line. No sources → warning row. Fetch results show
//! metadata + a 3/12-line content preview.

use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};

use serde_json::Value;

use crate::theme::Theme;

use super::{error_detail_lines, result_text, ToolRenderer};

/// Collapsed source cap (TS `MAX_COLLAPSED_ITEMS`).
const MAX_COLLAPSED_SOURCES: usize = 5;

/// Preview caps for fetch content (TS fetch.ts: 3 collapsed / 12 expanded).
const PREVIEW_COLLAPSED: usize = 3;
const PREVIEW_EXPANDED: usize = 12;

/// Renderer for web tools (`web_search` and `fetch`).
pub struct Web;

/// Synthesize a `call` view from a tool-execution frame (`args` holds the
/// arguments; `toolName` is the fallback name).
/// Fallback line width when the frame carries none (chat pane minimum).
const LINE_WIDTH: usize = 100;

fn json_call(frame: &Value) -> Value {
    let name = frame.get("toolName").cloned().unwrap_or(Value::Null);
    let args = frame.get("args").cloned().unwrap_or(Value::Null);
    serde_json::json!({"name": name, "arguments": args})
}

impl ToolRenderer for Web {
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

    /// Frame seam: pending lines before a result, result lines after.
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

/// Tool-arguments accessor tolerant of `arguments`/`input` shapes.
fn args(call: &Value) -> &Value {
    call.get("arguments")
        .or_else(|| call.get("input"))
        .unwrap_or(&Value::Null)
}

fn query(a: &Value) -> &str {
    a.get("query").and_then(Value::as_str).unwrap_or("")
}

fn url(a: &Value) -> &str {
    ["url", "path"]
        .iter()
        .find_map(|k| a.get(*k).and_then(Value::as_str))
        .unwrap_or("")
}

fn is_fetch(a: &Value) -> bool {
    !url(a).is_empty() && query(a).is_empty()
}

fn pending_label(a: &Value) -> String {
    if is_fetch(a) {
        format!("Read {}", truncate_str(url(a), 80))
    } else {
        format!("Web Search {}", truncate_str(query(a), 80))
    }
}

// ─── Pending ────────────────────────────────────────────────────────────────

/// One bare pending status line, TS `renderCall`.
#[must_use]
pub fn pending_lines(call: &Value, theme: &Theme, _width: usize) -> Vec<Line<'static>> {
    vec![Line::from(vec![
        Span::styled("⧗ ".to_owned(), Style::new().fg(theme.color("text_muted"))),
        Span::styled(
            pending_label(args(call)),
            Style::new().fg(theme.color("text")),
        ),
    ])]
}

// ─── Result ────────────────────────────────────────────────────────────────

/// Result card lines. `result` is the tool-result JSON (`details` holds the
/// `SearchResponse` for `web_search`, fetch metadata + content for fetch);
/// the value itself is accepted when there is no `details` wrapper.
#[must_use]
pub fn result_lines(
    call: &Value,
    result: &Value,
    theme: &Theme,
    expanded: bool,
    width: usize,
) -> Vec<Line<'static>> {
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        let mut lines = vec![Line::from(vec![
            Span::styled(
                format!("{} ", theme.glyph("failure")),
                Style::new().fg(theme.color("error")),
            ),
            Span::styled(
                pending_label(args(call)),
                Style::new().fg(theme.color("error")).bold(),
            ),
        ])];
        // Error body per TS `formatErrorDetail`: `Error: ` prefix stripped,
        // two-column indent, error color.
        if let Some(text) = result_text(result) {
            if !text.trim().is_empty() {
                lines.extend(error_detail_lines(&text, width, theme));
            }
        }
        return lines;
    }
    if is_fetch(args(call)) {
        return fetch_result_lines(result, theme, expanded, width);
    }
    search_result_lines(result, theme, expanded, width)
}

fn search_result_lines(
    result: &Value,
    theme: &Theme,
    expanded: bool,
    width: usize,
) -> Vec<Line<'static>> {
    let d = result.get("details").unwrap_or(result);
    let provider = d.get("provider").and_then(Value::as_str).unwrap_or("web");
    let sources = d.get("sources").and_then(Value::as_array);
    let n_sources = sources.map_or(0, Vec::len);

    let mut lines = vec![Line::from(vec![
        Span::styled("⌕ ".to_owned(), Style::new().fg(theme.color("accent"))),
        Span::styled(
            "Web Search".to_owned(),
            Style::new().fg(theme.color("accent")).bold(),
        ),
        Span::styled(
            format!("  {provider} · {n_sources} sources"),
            Style::new().fg(theme.color("text_muted")),
        ),
    ])];

    if n_sources == 0 {
        lines.push(Line::from(Span::styled(
            "No sources returned".to_owned(),
            Style::new().fg(theme.color("warning")),
        )));
    }

    if let Some(answer) = d
        .get("answer")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        lines.push(label_line(theme, "Answer:"));
        for l in answer.lines().flat_map(str::lines) {
            lines.push(wrap_span(theme, "text", l, width.saturating_sub(2)));
        }
    }

    if let Some(srcs) = sources.filter(|s| !s.is_empty()) {
        lines.push(label_line(theme, "Sources:"));
        let shown: Vec<&Value> = if expanded {
            srcs.iter().collect()
        } else {
            srcs.iter().take(MAX_COLLAPSED_SOURCES).collect()
        };
        for s in &shown {
            let title = s
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("(untitled)");
            let link = s.get("url").and_then(Value::as_str).unwrap_or("");
            let domain = link
                .split("://")
                .nth(1)
                .unwrap_or(link)
                .split(['/', '?', '#'])
                .next()
                .unwrap_or(link);
            let mut spans = vec![
                Span::raw("  • ".to_owned()),
                Span::styled(
                    truncate_str(title, width.saturating_sub(8)),
                    Style::new().fg(theme.color("accent")),
                ),
                Span::styled(
                    format!(" ({domain})"),
                    Style::new().fg(theme.color("text_muted")),
                ),
            ];
            if let Some(age) = age_label(s) {
                spans.push(Span::styled(
                    format!(" {theme_sep}{age}", theme_sep = "·"),
                    Style::new().fg(theme.color("text_muted")),
                ));
            }
            truncate_spans(&mut spans, width);
            lines.push(Line::from(spans));
        }
        if srcs.len() > shown.len() {
            lines.push(dim_line(
                theme,
                &format!(
                    "… {} more sources [ctrl+o: Expand]",
                    srcs.len() - shown.len()
                ),
            ));
        }
    }

    let mut meta = vec![];
    if let Some(model) = d.get("model").and_then(Value::as_str) {
        meta.push(format!("model {model} @ {provider}"));
    }
    if let Some(u) = d.get("usage") {
        let f = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        meta.push(format!(
            "in {} · out {} · search {}",
            f("inputTokens"),
            f("outputTokens"),
            f("searchRequests")
        ));
    }
    if !meta.is_empty() {
        lines.push(label_line(theme, "Metadata:"));
        lines.push(dim_line(theme, &format!("  {}", meta.join(" · "))));
    }
    lines
}

fn fetch_result_lines(
    result: &Value,
    theme: &Theme,
    expanded: bool,
    width: usize,
) -> Vec<Line<'static>> {
    let d = result.get("details").unwrap_or(result);
    let mut lines = vec![Line::from(vec![
        Span::styled("⌕ ".to_owned(), Style::new().fg(theme.color("accent"))),
        Span::styled(
            "Read".to_owned(),
            Style::new().fg(theme.color("accent")).bold(),
        ),
        Span::styled(
            format!("  {}", url(args_of_fetch(d))),
            Style::new().fg(theme.color("text_muted")),
        ),
    ])];

    let mut meta_rows: Vec<String> = Vec::new();
    for (key, label) in [
        ("contentType", "Content-Type"),
        ("method", "Method"),
        ("finalUrl", "Final URL"),
        ("lines", "Lines"),
        ("chars", "Chars"),
    ] {
        if let Some(v) = d.get(key).map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }) {
            meta_rows.push(format!("{label}: {v}"));
        }
    }
    let truncated = d.get("truncated").and_then(Value::as_bool).unwrap_or(false);
    if truncated {
        meta_rows.push(format!(
            "{} Output truncated",
            theme.symbol("status.warning")
        ));
    }
    if !meta_rows.is_empty() {
        lines.push(label_line(theme, "Metadata:"));
        for row in meta_rows {
            lines.push(dim_line(theme, &format!("  {row}")));
        }
    }

    lines.push(label_line(theme, "Content Preview:"));
    let content = d
        .get("content")
        .or_else(|| d.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let non_empty: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if non_empty.is_empty() {
        lines.push(dim_line(theme, "  (no content)"));
        return lines;
    }
    let cap = if expanded {
        PREVIEW_EXPANDED
    } else {
        PREVIEW_COLLAPSED
    };
    for l in non_empty.iter().take(cap) {
        lines.push(wrap_span(theme, "text_muted", l, width.saturating_sub(2)));
    }
    if non_empty.len() > cap {
        lines.push(dim_line(
            theme,
            &format!("… {} more lines [ctrl+o: Expand]", non_empty.len() - cap),
        ));
    }
    lines
}

fn args_of_fetch(d: &Value) -> &Value {
    d
}

fn age_label(source: &Value) -> Option<String> {
    if let Some(date) = source.get("publishedDate").and_then(Value::as_str) {
        return Some(date.to_owned());
    }
    let secs = source.get("ageSeconds").and_then(Value::as_f64)?;
    Some(if secs < 3600.0 {
        format!("{:.0}m ago", secs / 60.0)
    } else if secs < 86_400.0 {
        format!("{:.0}h ago", secs / 3600.0)
    } else {
        format!("{:.0}d ago", secs / 86_400.0)
    })
}

fn label_line(theme: &Theme, label: &str) -> Line<'static> {
    Line::from(Span::styled(
        label.to_owned(),
        Style::new().fg(theme.color("text_muted")),
    ))
}

fn dim_line(theme: &Theme, text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_owned(),
        Style::new().fg(theme.color("text_muted")),
    ))
}

fn wrap_span(theme: &Theme, token: &str, text: &str, width: usize) -> Line<'static> {
    Line::from(Span::styled(
        truncate_str(text, width.max(4)),
        Style::new().fg(theme.color(token)),
    ))
}

/// Char-count truncation with a unicode ellipsis (approximates TS
/// `truncateToWidth`, which needs `unicode-width`).
fn truncate_str(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.replace('\n', " ");
    }
    let mut out: String = s.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Clamp the final span so the line fits `width` columns.
fn truncate_spans(spans: &mut [Span<'static>], width: usize) {
    let total: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if total <= width || spans.is_empty() {
        return;
    }
    let others: usize = spans
        .iter()
        .rev()
        .skip(1)
        .map(|s| s.content.chars().count())
        .sum();
    let last = spans.last_mut().expect("non-empty");
    let text: String = last
        .content
        .chars()
        .take(width.saturating_sub(others + 1))
        .collect();
    last.content = format!("{text}…").into();
}

/// Status-line summary: `"<provider> · N sources"`.
#[must_use]
pub fn status_summary(result: &Value) -> String {
    let d = result.get("details").unwrap_or(result);
    let provider = d.get("provider").and_then(Value::as_str).unwrap_or("web");
    let n = d
        .get("sources")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    format!("{provider} · {n} sources")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn theme() -> Theme {
        Theme::default()
    }

    #[test]
    fn pending_uses_fetch_label_for_url_args() {
        let t = theme();
        let fetch = json!({"name":"fetch","arguments":{"url":"https://example.com/a"}});
        let lines = pending_lines(&fetch, &t, 100);
        assert_eq!(
            lines[0]
                .spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>(),
            "⧗ Read https://example.com/a"
        );
        let search = json!({"name":"web_search","arguments":{"query":"rust ratatui"}});
        let lines = pending_lines(&search, &t, 100);
        assert_eq!(
            lines[0]
                .spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>(),
            "⧗ Web Search rust ratatui"
        );
    }

    #[test]
    fn search_result_lists_sources_and_usage() {
        let t = theme();
        let result = json!({
            "details": {
                "provider": "anthropic",
                "answer": "Ratatui is a Rust TUI library.",
                "sources": [
                    {"title": "Ratatui docs", "url": "https://docs.rs/ratatui", "ageSeconds": 7200.0},
                    {"title": "GitHub", "url": "https://github.com/ratatui/ratatui", "publishedDate": "2026-01-02"}
                ],
                "usage": {"inputTokens": 10, "outputTokens": 20, "searchRequests": 1}
            }
        });
        let lines = result_lines(&json!({"arguments":{"query":"q"}}), &result, &t, false, 120);
        let text: Vec<String> = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(text[0].contains("anthropic · 2 sources"), "{text:?}");
        assert!(text
            .iter()
            .any(|l| l.contains("Ratatui is a Rust TUI library.")));
        assert!(text.iter().any(|l| l.contains("docs.rs")));
        assert!(text.iter().any(|l| l.contains("2h ago")));
        assert!(text.iter().any(|l| l.contains("in 10 · out 20 · search 1")));
    }

    #[test]
    fn search_result_without_sources_warns() {
        let t = theme();
        let result = json!({"details": {"provider": "anthropic", "sources": []}});
        let lines = result_lines(&json!({"arguments":{"query":"q"}}), &result, &t, false, 80);
        let text: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(text.contains("No sources returned"));
    }

    #[test]
    fn search_result_collapses_sources_with_expand_hint() {
        let t = theme();
        let sources: Vec<Value> = (0..8)
            .map(|i| json!({"title": format!("s{i}"), "url": format!("https://h{i}.example")}))
            .collect();
        let result = json!({"details": {"provider": "p", "sources": sources}});
        let lines = result_lines(&json!({"arguments":{"query":"q"}}), &result, &t, false, 120);
        let text: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(text.contains("… 3 more sources [ctrl+o: Expand]"));
        let expanded = result_lines(&json!({"arguments":{"query":"q"}}), &result, &t, true, 120);
        assert_eq!(expanded.len(), 1 + 1 + 8);
    }

    #[test]
    fn fetch_result_previews_content() {
        let t = theme();
        let result = json!({
            "details": {
                "url": "https://example.com",
                "contentType": "text/plain",
                "lines": 6,
                "truncated": true,
                "content": "one\ntwo\nthree\nfour\nfive\nsix"
            }
        });
        let lines = result_lines(
            &json!({"arguments":{"url":"https://example.com"}}),
            &result,
            &t,
            false,
            100,
        );
        let text: Vec<String> = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(text.iter().any(|l| l.contains("Content-Type: text/plain")));
        assert!(text.iter().any(|l| l.contains("▲ Output truncated")));
        assert!(text.iter().any(|l| l.contains("three")));
        assert!(!text.iter().any(|l| l.contains("four")));
        assert!(text
            .iter()
            .any(|l| l.contains("… 3 more lines [ctrl+o: Expand]")));
    }

    #[test]
    fn error_result_renders_error_icon() {
        let t = theme();
        let result = json!({"isError": true});
        let lines = result_lines(&json!({"arguments":{"query":"q"}}), &result, &t, false, 80);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn error_result_renders_stripped_indented_detail_body() {
        let t = theme();
        let result = json!({
            "isError": true,
            "content": [{"type": "text", "text": "Error: rate limited, retry later"}]
        });
        let lines = result_lines(&json!({"arguments":{"query":"q"}}), &result, &t, false, 80);
        assert_eq!(lines.len(), 2);
        let flat: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect();
        assert!(flat[0].contains('✗') || flat[0].contains('x'), "{flat:?}");
        assert_eq!(flat[1], "  rate limited, retry later");
    }

    #[test]
    fn status_summary_reports_provider_and_count() {
        let result = json!({"details": {"provider": "anthropic", "sources": [{}, {}]}});
        assert_eq!(status_summary(&result), "anthropic · 2 sources");
    }
}
