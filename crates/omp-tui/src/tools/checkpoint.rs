//! `checkpoint` / `rewind` tool renderers — context-save/restore chrome
//! (omp-cli `src/tools/checkpoint.rs`: `checkpoint {goal}`, `rewind {report}`).
//!
//! Both are lightweight inline rows: `⧗ Checkpoint: <goal>` while pending,
//! done-glyph on success, ✘ on error; `rewind` previews the report text.

use ratatui::layout::Rect;
use ratatui::text::Line;
use serde_json::Value;

use crate::theme::Theme;

use super::{args_of, is_error, paint, result_text, status_line, styled, truncate, ToolRenderer};

/// Summary budget for the goal/report text.
const SUMMARY_BUDGET: usize = 60;
/// Result preview rows.
const PREVIEW_LINES: usize = 3;

/// Renderer for the checkpoint and rewind tools.
pub struct Checkpoint;

impl ToolRenderer for Checkpoint {
    fn summary(&self, call: &Value) -> String {
        let Some(args) = args_of(call) else {
            return String::new();
        };
        let text = args
            .get("goal")
            .or_else(|| args.get("report"))
            .and_then(Value::as_str)
            .unwrap_or("");
        truncate(text, SUMMARY_BUDGET)
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let is_rewind = call.get("name").and_then(Value::as_str) == Some("rewind");
        let title = if is_rewind { "Rewind" } else { "Checkpoint" };
        let settled = call.get("result").is_some();
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if settled {
            theme.glyph("success")
        } else {
            "⧗"
        };
        let mut lines = vec![status_line(
            theme,
            icon,
            title,
            &self.summary(call),
            Vec::new(),
        )];
        if let Some(text) = result_text(call) {
            for line in text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .take(PREVIEW_LINES)
            {
                lines.push(Line::from(styled(theme, "text_muted", line.to_owned())));
            }
        }
        if !lines.iter().any(|l| !l.spans.is_empty()) {
            lines.push(Line::from(styled(
                theme,
                "text_muted",
                "checkpoint recorded",
            )));
        }
        paint(frame, area, lines);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    #[test]
    fn summary_prefers_goal_then_report() {
        let cp = json!({"name": "checkpoint", "arguments": {"goal": "investigate flake"}});
        assert_eq!(Checkpoint.summary(&cp), "investigate flake");
        let rw = json!({"name": "rewind", "arguments": {"report": "root cause: race"}});
        assert_eq!(Checkpoint.summary(&rw), "root cause: race");
    }

    #[test]
    fn settled_renders_success_glyph() {
        let backend = TestBackend::new(60, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let call = json!({"name": "checkpoint", "arguments": {"goal": "g"},
            "result": {"content": [{"type": "text", "text": "checkpoint saved"}]}});
        terminal
            .draw(|f| Checkpoint.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains('✓'));
        assert!(text.contains("Checkpoint"));
        assert!(text.contains("checkpoint saved"));
    }

    #[test]
    fn error_renders_failure_glyph() {
        let backend = TestBackend::new(60, 2);
        let mut terminal = Terminal::new(backend).unwrap();
        let call = json!({"name": "rewind", "arguments": {"report": "r"},
            "result": {"isError": true, "content": [{"type": "text", "text": "boom"}]}});
        terminal
            .draw(|f| Checkpoint.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains('✗'));
    }
}
