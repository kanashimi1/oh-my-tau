//! `wait` tool renderer — port of `pi-tui/src/tools/wait.ts` (spec
//! `tools-b.md` §6): the jobs/agents roster herdr panes show constantly.
//!
//! `inline: true`, `mergeCallAndResult: true`. Pending: `⧗ Wait`. Result
//! header `waiting on N of M jobs` / `M jobs settled` / `N running agents —
//! no jobs`, meta `n done · n failed · n cancelled · n agents`, rows sorted
//! running → failed → cancelled → completed, then duration desc.

use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use serde_json::Value;

use crate::theme::Theme;

use super::{details_of, is_error, paint, status_line, styled, truncate, ToolRenderer};

/// Max width of a job label (`LABEL_MAX_WIDTH` in wait.ts).
pub(crate) const LABEL_MAX_WIDTH: usize = 60;

/// Renderer for the wait tool.
pub struct Wait;

impl ToolRenderer for Wait {
    fn summary(&self, call: &Value) -> String {
        let jobs = jobs_of(call);
        match jobs.len() {
            0 => String::new(),
            n => format!("{n} jobs"),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let title = head_title(call, "wait");
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if jobs_of(call).iter().any(|j| status(j) == "failed") {
            "⚠"
        } else if jobs_of(call).iter().any(|j| status(j) == "running") {
            "ℹ"
        } else {
            theme.glyph("success")
        };
        let mut lines = vec![status_line(
            theme,
            icon,
            &title,
            "",
            count_meta(call, theme),
        )];
        lines.extend(roster_rows(call, theme));
        paint(frame, area, lines);
    }
}

/// Head title for the roster (`jobsRenderResult` header wording).
pub(crate) fn head_title(call: &Value, op: &str) -> String {
    let jobs = jobs_of(call);
    let running = jobs.iter().filter(|j| status(j) == "running").count();
    let agents = details_of(call)
        .and_then(|d| d.get("agents"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    match (jobs.len(), running, agents) {
        (0, 0, 0) => format!("{op}: no jobs"),
        (0, _, a) => format!("{a} running agents — no jobs"),
        (m, 0, _) => format!("{m} jobs settled"),
        (m, r, _) => format!("waiting on {r} of {m} jobs"),
    }
}

/// Count badges: `n done / n failed / n cancelled / n agents`.
pub(crate) fn count_meta(call: &Value, theme: &Theme) -> Vec<Span<'static>> {
    let jobs = jobs_of(call);
    let count = |s: &str| jobs.iter().filter(|j| status(j) == s).count();
    let mut meta = Vec::new();
    if count("completed") > 0 {
        meta.push(styled(
            theme,
            "success",
            format!("{} done", count("completed")),
        ));
    }
    if count("failed") > 0 {
        meta.push(styled(
            theme,
            "error",
            format!("{} failed", count("failed")),
        ));
    }
    if count("cancelled") > 0 {
        meta.push(styled(
            theme,
            "warning",
            format!("{} cancelled", count("cancelled")),
        ));
    }
    if let Some(agents) = details_of(call)
        .and_then(|d| d.get("agents"))
        .and_then(Value::as_array)
    {
        if !agents.is_empty() {
            meta.push(styled(theme, "accent", format!("{} agents", agents.len())));
        }
    }
    meta
}

/// Sorted roster rows, one per job (`renderTreeList` itemType "job").
pub(crate) fn roster_rows(call: &Value, theme: &Theme) -> Vec<Line<'static>> {
    let mut jobs = jobs_of(call);
    jobs.sort_by_key(|j| (sort_rank(status(j)), std::cmp::Reverse(duration_ms(j))));
    jobs.iter()
        .map(|j| {
            let (icon, token) = match status(j) {
                "running" => ("●", "accent"),
                "failed" => (theme.glyph("failure"), "error"),
                "cancelled" => ("⊘", "warning"),
                _ => (theme.glyph("success"), "success"),
            };
            let label = truncate(
                j.get("label").and_then(Value::as_str).unwrap_or(""),
                LABEL_MAX_WIDTH,
            );
            let mut spans = vec![
                Span::raw("  "),
                styled(theme, token, format!("{icon} {label}")),
            ];
            if let Some(kind) = j.get("type").and_then(Value::as_str) {
                spans.push(styled(theme, "text_muted", format!("  [{kind}]")));
            }
            if duration_ms(j) > 0 {
                spans.push(styled(
                    theme,
                    "text_muted",
                    format!("  {}", format_duration(duration_ms(j))),
                ));
            }
            if let Some(code) = j.get("exitCode").and_then(Value::as_u64) {
                spans.push(styled(theme, "error", format!("  exit {code}")));
            }
            Line::from(spans)
        })
        .collect()
}

/// Borrow the job snapshot list from result details.
pub(crate) fn jobs_of(call: &Value) -> Vec<Value> {
    details_of(call)
        .and_then(|d| d.get("jobs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn status(job: &Value) -> &str {
    job.get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed")
}

fn duration_ms(job: &Value) -> u64 {
    job.get("durationMs").and_then(Value::as_u64).unwrap_or(0)
}

/// Sort rank: running → failed → cancelled → completed.
fn sort_rank(status: &str) -> u8 {
    match status {
        "running" => 0,
        "failed" => 1,
        "cancelled" => 2,
        _ => 3,
    }
}

/// `formatDuration`: ms → `1m 02s` / `3.5s` / `120ms`.
#[must_use]
pub(crate) fn format_duration(ms: u64) -> String {
    if ms >= 60_000 {
        format!("{}m {:02}s", ms / 60_000, (ms % 60_000) / 1000)
    } else if ms >= 10_000 {
        format!("{}s", (ms + 500) / 1000)
    } else if ms >= 1_000 {
        format!("{}.{:.1}s", ms / 1000, (ms % 1000) / 100)
    } else {
        format!("{ms}ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn call_with(jobs: Value) -> Value {
        json!({"name": "wait", "arguments": {},
            "result": {"content": [], "details": {"jobs": jobs}}})
    }

    #[test]
    fn summary_counts_jobs() {
        let call = call_with(json!([
            {"id": "bg_1", "status": "running", "label": "cargo test"},
            {"id": "bg_2", "status": "completed", "label": "done"},
        ]));
        assert_eq!(Wait.summary(&call), "2 jobs");
        assert_eq!(Wait.summary(&json!({"name": "wait"})), "");
    }

    #[test]
    fn header_waits_on_running_subset() {
        let call = call_with(json!([
            {"id": "a", "status": "running", "label": "x", "durationMs": 500},
            {"id": "b", "status": "running", "label": "y", "durationMs": 900},
            {"id": "c", "status": "completed", "label": "z", "durationMs": 100},
        ]));
        assert_eq!(head_title(&call, "wait"), "waiting on 2 of 3 jobs");
        let settled =
            call_with(json!([{"id": "a", "status": "completed", "label": "x", "durationMs": 1}]));
        assert_eq!(head_title(&settled, "wait"), "1 jobs settled");
    }

    #[test]
    fn rows_sort_running_first_then_duration_desc() {
        let call = call_with(json!([
            {"id": "done", "status": "completed", "label": "old", "durationMs": 5000},
            {"id": "run-slow", "status": "running", "label": "slow", "durationMs": 9000},
            {"id": "run-fast", "status": "running", "label": "fast", "durationMs": 1000},
        ]));
        let rows = roster_rows(&call, &Theme::default());
        let text: Vec<String> = rows
            .iter()
            .map(|r| r.spans.iter().map(|s| s.content.to_string()).collect())
            .collect();
        assert!(text[0].contains("slow"));
        assert!(text[1].contains("fast"));
        assert!(text[2].contains("old"));
    }

    #[test]
    fn render_shows_meta_and_rows() {
        let call = call_with(json!([
            {"id": "a", "status": "failed", "label": "build", "durationMs": 1500, "exitCode": 1},
        ]));
        let backend = TestBackend::new(80, 6);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Wait.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("1 jobs settled") || text.contains("jobs"));
        assert!(text.contains("1 failed"));
        assert!(text.contains("build"));
        assert!(text.contains("exit 1"));
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(1_500), "1.5s");
        assert_eq!(format_duration(65_000), "1m 05s");
    }
}
