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

use super::search::{args_of, details_of, more_items, paint, status_line, styled, warn};
use super::ToolRenderer;

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

/// One hit row: `████░░ 0.83 src/lib.rs (40 lines judged)`.
fn hit_head(hit: &Value, theme: &Theme) -> Line<'static> {
    let score = hit
        .get("contentScore")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let rel = hit
        .get("rel")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_owned();
    let coverage = match (
        hit.get("truncated").and_then(Value::as_bool),
        hit.get("linesSeen").and_then(Value::as_u64),
    ) {
        (Some(true), Some(n)) => format!(" ({n} lines judged, partial)"),
        (_, Some(n)) => format!(" ({n} lines judged)"),
        _ => String::new(),
    };
    Line::from(vec![
        gauge(score, theme),
        Span::raw(" "),
        styled(theme, score_token(score), format!("{score:.2}")),
        Span::raw(" "),
        styled(theme, "accent", rel),
        styled(theme, "text_muted", coverage),
    ])
}

/// Range rows for a hit, strongest first: `├─ :12-18 <snippet>`.
fn hit_ranges(hit: &Value, limit: usize, theme: &Theme) -> Vec<Line<'static>> {
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
    ranges
        .into_iter()
        .take(limit)
        .map(|r| {
            let start = r.get("start").and_then(Value::as_u64).unwrap_or(0);
            let end = r.get("end").and_then(Value::as_u64).unwrap_or(start);
            let span_txt = if start == end {
                format!(":{start}")
            } else {
                format!(":{start}-{end}")
            };
            let snippet = r
                .get("snippet")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .replace('\t', "    ");
            Line::from(vec![
                styled(theme, "text_muted", "\u{251c}\u{2500} "), // ├─
                styled(theme, "text_muted", span_txt),
                Span::raw(" "),
                styled(theme, "text", snippet),
            ])
        })
        .collect()
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
        let details = details_of(call);
        let hits = details
            .and_then(|d| d.get("hits"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let scope = details
            .and_then(|d| d.get("scopePath"))
            .and_then(Value::as_str)
            .map(|p| format!("in {p}"));
        let files_read = details
            .and_then(|d| d.get("stats"))
            .and_then(|s| s.get("filesRead"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let errors = details
            .and_then(|d| d.get("stats"))
            .and_then(|s| s.get("errors"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let threshold = detail_f64(details, "threshold").unwrap_or(0.0);

        let mut meta = vec![styled(theme, "text_muted", format!("{} hits", hits.len()))];
        if let Some(s) = &scope {
            meta.push(styled(theme, "text_muted", s.clone()));
        }
        meta.push(styled(
            theme,
            "text_muted",
            format!("{files_read} files read"),
        ));
        meta.push(styled(theme, "text_muted", format!("τ {threshold:.2}")));
        if errors > 0 {
            meta.push(warn(theme, format!("{errors} failed")));
        }

        let mut lines = Vec::new();
        if hits.is_empty() {
            lines.push(status_line(theme, "⚠", "Find", &self.summary(call), meta));
            lines.push(Line::from(vec![warn(
                theme,
                "⚠ No relevant passages found",
            )]));
            paint(frame, area, lines);
            return;
        }

        lines.push(status_line(
            theme,
            "\u{1f50d}",
            "Find",
            &self.summary(call),
            meta,
        ));
        let shown = hits.len().min(COLLAPSED_HITS);
        for (i, hit) in hits.iter().enumerate().take(shown) {
            let last_group = i + 1 == shown && hits.len() <= COLLAPSED_HITS;
            lines.push({
                let mut head = hit_head(hit, theme);
                head.spans.insert(
                    0,
                    styled(theme, "text_muted", if last_group { "└─ " } else { "├─ " }),
                );
                head
            });
            for range in hit_ranges(hit, RANGES_COLLAPSED, theme) {
                lines.push(range);
            }
        }
        if hits.len() > COLLAPSED_HITS {
            lines.push(Line::from(vec![
                styled(theme, "text_muted", "└─ "),
                more_items(theme, hits.len() - COLLAPSED_HITS, "hit"),
            ]));
        }
        paint(frame, area, lines);
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
                .filter(|l| l.contains("f") && l.contains(".rs (40 lines judged)"))
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
