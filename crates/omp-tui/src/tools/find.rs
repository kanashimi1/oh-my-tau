//! `find` tool renderer — port of `pi-tui/src/tools/find.ts`.
//!
//! Each hit renders as a score gauge + linked relative path + coverage note,
//! followed by its strongest code ranges. Collapsed keeps the first
//! [`COLLAPSED_HITS`] hits with 1 range each; `… N more hits` folds the rest.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::Frame;
use serde_json::Value;

use crate::theme::Theme;

use super::frame::{status_header, token};
use super::search::{
    args_of, details_of, empty_row, error_row, format_count, pad_rows, paint, search_icon,
    status_icon, warn,
};
use super::ToolRenderer;
use crate::chat::blocks::{format_duration, format_number};

/// Cells in the per-hit score gauge (`GAUGE_WIDTH`).
const GAUGE_WIDTH: usize = 6;
/// Hits shown collapsed.
const COLLAPSED_HITS: usize = 5;
/// Code ranges per hit: collapsed / expanded.
const RANGES_COLLAPSED: usize = 1;
/// Used by the ctrl+o expanded view once the expand layer lands.
#[allow(dead_code)]
const RANGES_EXPANDED: usize = 3;
/// Score thresholds mapping to theme colors (`scoreColor`).
const STRONG: f64 = 0.7;
const PLAUSIBLE: f64 = 0.4;

/// Renderer for semantic file search.
pub struct Find;

fn score_token(score: f64) -> &'static str {
    if score >= STRONG {
        "success"
    } else if score >= PLAUSIBLE {
        "warning"
    } else {
        "text_muted"
    }
}

/// `renderProgressBar` at `GAUGE_WIDTH`: filled/empty cells, score-colored.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn gauge(score: f64, theme: &Theme) -> Span<'static> {
    let filled = ((score * GAUGE_WIDTH as f64).round() as usize).min(GAUGE_WIDTH);
    let text = format!("{}{}", "█".repeat(filled), "░".repeat(GAUGE_WIDTH - filled));
    Span::styled(text, Style::default().fg(theme.color(score_token(score))))
}

fn dim(theme: &Theme, text: impl Into<String>) -> Span<'static> {
    Span::styled(
        text.into(),
        Style::default().fg(token(theme, "dim", "text_muted")),
    )
}

fn score_style(score: f64, theme: &Theme) -> Style {
    Style::default().fg(token(
        theme,
        if score >= STRONG {
            "success"
        } else if score >= PLAUSIBLE {
            "warning"
        } else {
            "muted"
        },
        "text_muted",
    ))
}

/// One hit item (TS `renderHit`): head row + strongest range rows.
fn hit_item(hit: &Value, range_limit: usize, theme: &Theme) -> Vec<Line<'static>> {
    let score = hit
        .get("contentScore")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let rel = hit
        .get("rel")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_owned();
    let seen = hit.get("linesSeen").and_then(Value::as_u64).unwrap_or(0);
    let coverage = if hit.get("truncated").and_then(Value::as_bool) == Some(true) {
        format!("{seen} lines judged, partial")
    } else {
        format!("{seen} lines judged")
    };
    let mut rows = vec![Line::from(vec![
        gauge(score, theme),
        Span::raw(" "),
        Span::styled(format!("{score:.2}"), score_style(score, theme)),
        Span::raw(" "),
        Span::styled(rel, Style::default().fg(theme.color("accent"))),
        Span::raw(" "),
        dim(theme, coverage),
    ])];
    let mut ranges: Vec<&Value> = hit
        .get("ranges")
        .and_then(Value::as_array)
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    ranges.sort_by(|a, b| {
        let pa = a.get("p").and_then(Value::as_f64).unwrap_or(0.0);
        let pb = b.get("p").and_then(Value::as_f64).unwrap_or(0.0);
        pb.partial_cmp(&pa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let sa = a.get("start").and_then(Value::as_u64).unwrap_or(0);
                let sb = b.get("start").and_then(Value::as_u64).unwrap_or(0);
                sa.cmp(&sb)
            })
    });
    ranges.truncate(range_limit);
    let n = ranges.len();
    for (i, r) in ranges.into_iter().enumerate() {
        let start = r.get("start").and_then(Value::as_u64).unwrap_or(0);
        let end = r.get("end").and_then(Value::as_u64).unwrap_or(start);
        let span_txt = if start == end {
            format!(":{start}")
        } else {
            format!(":{start}-{end}")
        };
        let p = r.get("p").and_then(Value::as_f64).unwrap_or(0.0);
        let snippet = r
            .get("snippet")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .replace('\t', "    ");
        rows.push(Line::from(vec![
            Span::raw("  "),
            dim(
                theme,
                format!(
                    "{} ",
                    theme.symbol(if i + 1 == n {
                        "tree.last"
                    } else {
                        "tree.branch"
                    })
                ),
            ),
            Span::styled(
                span_txt,
                Style::default().fg(token(theme, "muted", "text_muted")),
            ),
            Span::raw(" "),
            dim(theme, format!("{p:.2}")),
            Span::raw(" "),
            Span::styled(
                snippet,
                Style::default().fg(token(theme, "toolOutput", "text")),
            ),
        ]));
    }
    rows
}

/// Find transcript rows (TS `findToolRenderer.renderCall/renderResult`).
#[must_use]
pub fn find_lines(call: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let args = args_of(call);
    let details = details_of(call);
    let quote = |q: Option<&str>| q.map(|q| format!("\"{q}\""));
    let query_arg = args.and_then(|a| a.get("query")).and_then(Value::as_str);
    let muted =
        |t: String| Span::styled(t, Style::default().fg(token(theme, "muted", "text_muted")));
    let pending = call.get("result").is_none() && call.get("isError").is_none();
    if pending {
        let keywords: Vec<&str> = args
            .and_then(|a| a.get("grep_keywords"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let mut meta = Vec::new();
        if !keywords.is_empty() {
            meta.push(dim(theme, keywords.join(" ")));
        }
        if let Some(path) = args
            .and_then(|a| a.get("path"))
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
        {
            meta.push(dim(theme, format!("in {path}")));
        }
        let header = status_header(
            theme,
            Some(status_icon(theme, "status.pending", "muted")),
            "Find",
            "toolTitle",
            quote(query_arg).map(muted).into_iter().collect(),
            meta,
        );
        return pad_rows(vec![header]);
    }
    let flagged = call
        .get("result")
        .and_then(|r| r.get("isError"))
        .or_else(|| call.get("isError"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if flagged {
        let text = super::search::result_text(call).unwrap_or_else(|| "Unknown error".to_owned());
        return pad_rows(vec![error_row(theme, &text)]);
    }
    let hits = details
        .and_then(|d| d.get("hits"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let stats = details.and_then(|d| d.get("stats"));
    let num = |key: &str| {
        stats
            .and_then(|s| s.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let cost = stats
        .and_then(|s| s.get("cost"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let elapsed = details
        .and_then(|d| d.get("elapsedMs"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let query = details.and_then(|d| d.get("query")).and_then(Value::as_str);
    let scope = details
        .and_then(|d| d.get("scopePath"))
        .and_then(Value::as_str)
        .map(|p| format!("in {p}"));
    let threshold = detail_f64(details, "threshold").unwrap_or(0.0);
    let errors = num("errors");
    let failures: Vec<String> = stats
        .and_then(|s| s.get("failures"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let desc: Vec<Span<'static>> = quote(query).map(muted).into_iter().collect();
    let failed = |theme: &Theme| warn(theme, format!("{errors} failed"));
    if hits.is_empty() {
        let mut meta = vec![dim(theme, "0 hits")];
        meta.extend(scope.map(|s| dim(theme, s)));
        meta.push(dim(theme, format!("${cost:.4}")));
        meta.push(dim(theme, format_duration(elapsed)));
        if errors > 0 {
            meta.push(failed(theme));
        }
        let header = status_header(
            theme,
            Some(status_icon(theme, "status.warning", "warning")),
            "Find",
            "toolTitle",
            desc,
            meta,
        );
        let mut rows = vec![header, empty_row(theme, "No relevant passages found")];
        rows.extend(failures.iter().map(|f| Line::from(warn(theme, f.clone()))));
        return pad_rows(rows);
    }
    let mut meta = vec![dim(theme, format_count("hit", hits.len()))];
    meta.extend(scope.map(|s| dim(theme, s)));
    meta.push(dim(theme, format!("{} files read", num("filesRead"))));
    meta.push(dim(theme, format!("τ {threshold:.2}")));
    meta.push(dim(
        theme,
        format!("{} tokens", format_number(num("inputTokens"))),
    ));
    meta.push(dim(theme, format!("${cost:.4}")));
    meta.push(dim(theme, format_duration(elapsed)));
    if errors > 0 {
        meta.push(failed(theme));
    }
    let header = status_header(
        theme,
        Some(search_icon(theme)),
        "Find",
        "toolTitle",
        desc,
        meta,
    );
    let max = if expanded { hits.len() } else { COLLAPSED_HITS };
    let range_limit = if expanded {
        RANGES_EXPANDED
    } else {
        RANGES_COLLAPSED
    };
    let shown = hits.len().min(max);
    let mut rows = vec![header];
    let n = shown;
    let has_summary = hits.len() > shown;
    for (i, hit) in hits.iter().take(shown).enumerate() {
        let last = !has_summary && i + 1 == n;
        let branch = format!(
            "{} ",
            theme.symbol(if last { "tree.last" } else { "tree.branch" })
        );
        let cont = if last {
            "   ".to_owned()
        } else {
            format!("{}  ", theme.symbol("tree.vertical"))
        };
        for (j, mut row) in hit_item(hit, range_limit, theme).into_iter().enumerate() {
            row.spans.insert(
                0,
                dim(theme, if j == 0 { branch.clone() } else { cont.clone() }),
            );
            rows.push(row);
        }
    }
    if has_summary {
        rows.push(Line::from(vec![
            dim(theme, format!("{} ", theme.symbol("tree.last"))),
            muted(format!(
                "… {} more {}",
                hits.len() - shown,
                format_count("hit", hits.len() - shown)
                    .split_once(' ')
                    .map_or("hits", |(_, w)| w)
            )),
        ]));
    }
    if expanded {
        let keywords: Vec<&str> = details
            .and_then(|d| d.get("keywords"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        rows.push(Line::from(dim(
            theme,
            format!("keywords: {}", keywords.join(", ")),
        )));
        rows.extend(failures.iter().map(|f| Line::from(warn(theme, f.clone()))));
    }
    pad_rows(rows)
}

fn detail_f64(details: Option<&Value>, key: &str) -> Option<f64> {
    details.and_then(|d| d.get(key)).and_then(Value::as_f64)
}

impl ToolRenderer for Find {
    fn summary(&self, call: &Value) -> String {
        let query = args_of(call)
            .and_then(|a| a.get("query"))
            .or_else(|| details_of(call).and_then(|d| d.get("query")))
            .and_then(Value::as_str)
            .unwrap_or("?");
        query.to_owned()
    }

    fn render(&self, frame: &mut Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        paint(frame, area, find_lines(call, false, theme));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_lines(call: &Value, w: u16, h: u16) -> Vec<String> {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).unwrap();
        let theme = Theme::default();
        term.draw(|f| Find.render(f, f.area(), call, &theme))
            .unwrap();
        let buffer = term.backend().buffer().clone();
        (0..buffer.area.height)
            .filter_map(|y| {
                let row: String = (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_owned())
                    .collect();
                let trimmed = row.trim_end().to_owned();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
            .collect()
    }

    fn hit(rel: &str, score: f64) -> Value {
        serde_json::json!({
            "rel": rel,
            "contentScore": score,
            "linesSeen": 40,
            "ranges": [{"start": 3, "end": 7, "p": 0.9, "snippet": "let x = find();"}]
        })
    }

    #[test]
    fn gauge_fills_proportionally() {
        let theme = Theme::default();
        let g = gauge(0.83, &theme);
        assert_eq!(
            g.content,
            "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2591}"
        ); // round(0.83*6)=5
        let g0 = gauge(0.0, &theme);
        assert_eq!(
            g0.content,
            "\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}"
        );
    }

    #[test]
    fn score_tokens_map_thresholds() {
        assert_eq!(score_token(0.9), "success");
        assert_eq!(score_token(0.5), "warning");
        assert_eq!(score_token(0.1), "text_muted");
    }

    #[test]
    fn hits_capped_with_more_line() {
        let hits: Vec<Value> = (0..7).map(|i| hit(&format!("f{i}.rs"), 0.8)).collect();
        let call = serde_json::json!({
            "name": "find",
            "arguments": {"query": "needle"},
            "result": {"details": {
                "query": "needle",
                "hits": hits,
                "threshold": 0.3,
                "stats": {"filesRead": 12, "errors": 0}
            }}
        });
        let out = render_lines(&call, 100, 40);
        assert!(out[0].contains("7 hits"));
        assert!(out[0].contains("12 files read"));
        assert_eq!(
            out.iter()
                .filter(|l| l.contains(".rs") && l.contains("40 lines judged"))
                .count(),
            5
        );
        assert!(out.last().unwrap().contains("\u{2026} 2 more hits"));
        // collapsed shows one range per hit
        assert_eq!(out.iter().filter(|l| l.contains(":3-7")).count(), 5);
    }

    #[test]
    fn empty_hits_warn() {
        let call = serde_json::json!({
            "name": "find",
            "arguments": {"query": "q"},
            "result": {"details": {"query": "q", "hits": [], "threshold": 0.3, "stats": {"filesRead": 0, "errors": 0}}}
        });
        let out = render_lines(&call, 80, 10);
        assert!(out[0].contains("0 hits"));
        assert!(out.iter().any(|l| l.contains("No relevant passages found")));
    }

    #[test]
    fn summary_is_query() {
        let call = serde_json::json!({"name":"find","arguments":{"query":"borrow checker"}});
        assert_eq!(Find.summary(&call), "borrow checker");
    }
}
