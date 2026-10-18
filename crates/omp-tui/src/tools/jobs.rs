//! `jobs` tool renderer — roster view over the background-job registry
//! (omp-cli `src/tools/jobs.rs`), sharing the `wait.ts` row chrome.
//!
//! Same snapshot shape as the wait roster (`details.jobs[]`), but the head
//! is a flat `N jobs` summary since the tool is a query, not a wait.

use ratatui::layout::Rect;
use serde_json::Value;

use crate::theme::Theme;

use super::{is_error, paint, status_line, wait, ToolRenderer};

/// Renderer for the jobs tool.
pub struct Jobs;

impl ToolRenderer for Jobs {
    fn summary(&self, call: &Value) -> String {
        match wait::jobs_of(call).len() {
            0 => String::new(),
            n => format!("{n} jobs"),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let n = wait::jobs_of(call).len();
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if wait::jobs_of(call)
            .iter()
            .any(|j| j.get("status").and_then(Value::as_str) == Some("failed"))
        {
            theme.symbol("status.warning")
        } else {
            theme.glyph("success")
        };
        let title = if n == 0 {
            "no jobs".to_owned()
        } else {
            format!("{n} jobs")
        };
        let mut lines = vec![status_line(
            theme,
            icon,
            &title,
            "",
            wait::count_meta(call, theme),
        )];
        lines.extend(wait::roster_rows(call, theme));
        paint(frame, area, lines);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn call_with(jobs: Value) -> Value {
        json!({"name": "jobs", "arguments": {},
            "result": {"content": [], "details": {"jobs": jobs}}})
    }

    #[test]
    fn summary_counts() {
        let call = call_with(json!([{"id": "bg_1", "status": "running", "label": "t"}]));
        assert_eq!(Jobs.summary(&call), "1 jobs");
    }

    #[test]
    fn empty_registry_renders_no_jobs() {
        let backend = TestBackend::new(60, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Jobs.render(f, f.area(), &call_with(json!([])), &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("no jobs"));
    }

    #[test]
    fn mixed_status_renders_counts_and_rows() {
        let call = call_with(json!([
            {"id": "a", "status": "running", "label": "server", "type": "bash", "durationMs": 2000},
            {"id": "b", "status": "completed", "label": "index", "type": "task", "durationMs": 42_000},
        ]));
        let backend = TestBackend::new(80, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Jobs.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("2 jobs"));
        assert!(text.contains("1 done"));
        assert!(text.contains("server"));
        assert!(text.contains("42s"));
    }
}
