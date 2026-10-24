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
        if let Some(waited) = waited_of(call) {
            paint(frame, area, message_card_lines(&waited, false, theme));
            return;
        }
        let title = head_title(call, "wait");
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if jobs_of(call).iter().any(|j| status(j) == "failed") {
            theme.symbol("status.warning")
        } else if jobs_of(call).iter().any(|j| status(j) == "running") {
            theme.symbol("status.info")
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

    fn render_frame(&self, frame: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
        // Pending: a single `⧗ Wait` line (TS `waitRenderCall`).
        let Some(result) = frame.get("result") else {
            return vec![status_line(theme, "⧗", "Wait", "", Vec::new())];
        };
        let details = result.get("details");
        // Message waited: inline IRC message card (`wait.ts:748-769`).
        if let Some(waited) = details.and_then(|d| d.get("waited")) {
            return message_card_lines(waited, expanded, theme);
        }
        // Interrupted: `ⓘ Wait — interrupted by message`.
        if details
            .and_then(|d| d.get("interrupted"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return vec![status_line(
                theme,
                "ⓘ",
                "Wait",
                "interrupted by message",
                Vec::new(),
            )];
        }
        // Jobs/agents roster, transcript-shaped frame (`result.details.jobs`).
        let icon = if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            theme.glyph("failure")
        } else if jobs_of(frame).iter().any(|j| status(j) == "failed") {
            theme.symbol("status.warning")
        } else if jobs_of(frame).iter().any(|j| status(j) == "running") {
            theme.symbol("status.info")
        } else {
            theme.glyph("success")
        };
        let mut lines = vec![status_line(
            theme,
            icon,
            &head_title(frame, "wait"),
            "",
            count_meta(frame, theme),
        )];
        lines.extend(roster_rows(frame, theme));
        lines
    }
}

/// Collapsed/expanded body line caps for a message card (TS wait.ts
/// `BODY_LINES_COLLAPSED` / `BODY_LINES_EXPANDED`).
pub(crate) const BODY_LINES_COLLAPSED: usize = 2;
/// Expanded body line cap.
pub(crate) const BODY_LINES_EXPANDED: usize = 12;
/// Per-line width cap for message bodies (TS `BODY_LINE_WIDTH`).
pub(crate) const BODY_LINE_WIDTH: usize = 100;

/// Borrow the waited peer message (`details.waited`), if any.
pub(crate) fn waited_of(call: &Value) -> Option<Value> {
    details_of(call)?
        .get("waited")
        .filter(|w| w.is_object())
        .cloned()
}

/// Transcript lines for a waited IRC message (TS `wait.ts` message branch +
/// `createIrcMessageCard`): `IRC ← <from>` head with `age` / `reply` meta and
/// a quote-bordered, width-capped body.
#[must_use]
pub fn message_card_lines(waited: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let from = waited.get("from").and_then(Value::as_str).unwrap_or("?");
    let body = waited.get("body").and_then(Value::as_str).unwrap_or("");
    let mut meta: Vec<String> = Vec::new();
    if let Some(ts) = waited.get("ts").and_then(Value::as_u64) {
        let now = now_ms();
        if now > ts {
            meta.push(age_label(now - ts));
        }
    }
    if waited.get("replyTo").is_some() {
        meta.push("reply".to_owned());
    }
    let mut spans = vec![
        styled(theme, "accent", "IRC"),
        Span::raw(" "),
        styled(theme, "text", format!("← {from}")),
    ];
    for m in meta {
        spans.push(Span::raw(" "));
        spans.push(styled(theme, "text_muted", m));
    }
    let mut lines = vec![Line::from(spans)];
    lines.extend(message_body_lines(body, expanded, theme));
    lines
}

/// Quote-bordered message body rows (`  │ <line>`), blank lines dropped,
/// capped to the collapsed/expanded budget with a dim `… +N more lines` tail.
#[must_use]
pub fn message_body_lines(body: &str, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
    let cap = if expanded {
        BODY_LINES_EXPANDED
    } else {
        BODY_LINES_COLLAPSED
    };
    let rows: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    let shown = cap.min(rows.len());
    let mut lines: Vec<Line<'static>> = rows[..shown]
        .iter()
        .map(|l| {
            Line::from(vec![
                Span::raw("  "),
                styled(theme, "text_muted", "│ "),
                styled(theme, "toolOutput", truncate(l.trim(), BODY_LINE_WIDTH)),
            ])
        })
        .collect();
    let hidden = rows.len() - shown;
    if hidden > 0 {
        lines.push(Line::from(vec![
            Span::raw("  "),
            styled(theme, "text_muted", "│ "),
            styled(
                theme,
                "text_muted",
                format!(
                    "… +{hidden} more {}",
                    if hidden == 1 { "line" } else { "lines" }
                ),
            ),
        ]));
    }
    lines
}

/// Compact age label for a millisecond delta (TS `formatAge`): `45s`/`12m`/`3h`.
#[must_use]
pub fn age_label(elapsed_ms: u64) -> String {
    let secs = (elapsed_ms / 1000).max(1);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3_600 {
        format!("{}m", secs / 60)
    } else {
        format!("{}h", secs / 3_600)
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
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

    fn flat(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect()
    }

    #[test]
    fn message_card_renders_irc_head_reply_meta_and_quoted_body() {
        let waited = json!({"from": "alice", "body": "hello\n\nworld", "replyTo": "agent://bob"});
        let lines = message_card_lines(&waited, false, &Theme::default());
        let flat = flat(&lines);
        assert_eq!(flat[0], "IRC ← alice reply", "{flat:?}");
        assert_eq!(flat[1], "  │ hello", "{flat:?}");
        assert_eq!(flat[2], "  │ world", "{flat:?}");
    }

    #[test]
    fn message_body_collapses_with_more_lines_hint() {
        let theme = Theme::default();
        let body = "l1\nl2\nl3\nl4";
        let collapsed = message_body_lines(body, false, &theme);
        assert_eq!(collapsed.len(), 3);
        let flat = flat(&collapsed);
        assert_eq!(flat[2], "  │ … +2 more lines", "{flat:?}");
        let expanded = message_body_lines(body, true, &theme);
        assert_eq!(expanded.len(), 4);
    }

    #[test]
    fn render_frame_waited_branch_renders_message_card() {
        let frame = json!({"toolCallId": "1", "toolName": "wait", "args": {},
            "result": {"content": [], "details": {"waited": {"from": "alice", "body": "hi"}}}});
        let flat = flat(&Wait.render_frame(&frame, false, &Theme::default()));
        assert!(flat[0].starts_with("IRC ← alice"), "{flat:?}");
        assert!(flat.iter().any(|l| l == "  │ hi"), "{flat:?}");
    }

    #[test]
    fn render_frame_pending_and_interrupted_rows() {
        let theme = Theme::default();
        let pending = flat(&Wait.render_frame(&json!({"toolName": "wait"}), false, &theme));
        assert_eq!(pending.len(), 1);
        assert!(pending[0].contains('⧗'), "{pending:?}");
        assert!(pending[0].contains("Wait"), "{pending:?}");
        let interrupted = json!({"toolCallId": "1", "toolName": "wait", "args": {},
            "result": {"content": [], "details": {"interrupted": true}}});
        let flat = flat(&Wait.render_frame(&interrupted, false, &theme));
        assert_eq!(flat.len(), 1);
        assert!(flat[0].contains('ⓘ'), "{flat:?}");
        assert!(flat[0].contains("interrupted by message"), "{flat:?}");
    }

    #[test]
    fn render_frame_roster_branch_keeps_rows() {
        let frame = json!({"toolCallId": "1", "toolName": "wait", "args": {},
        "result": {"content": [], "details": {"jobs": [
            {"id": "a", "status": "failed", "label": "build", "durationMs": 10, "exitCode": 1}
        ]}}});
        let flat = flat(&Wait.render_frame(&frame, false, &Theme::default()));
        let text = flat.join("\n");
        assert!(text.contains("1 jobs settled"), "{text}");
        assert!(text.contains("1 failed"), "{text}");
        assert!(text.contains("build"), "{text}");
    }

    #[test]
    fn age_label_buckets() {
        assert_eq!(age_label(500), "1s");
        assert_eq!(age_label(65_000), "1m");
        assert_eq!(age_label(3_700_000), "1h");
    }
}
