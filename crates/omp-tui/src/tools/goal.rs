//! `goal` tool renderer — port of the goal chrome in
//! `pi-tui/src/tools/goal.ts` (spec `tools-a.md` §16).
//!
//! Pending: `⧗ Goal: <text> [meta]`; error ✘; "no active goal" → warning;
//! success → done-glyph override. `inline` status line, no frame.

use ratatui::layout::Rect;
use ratatui::text::Line;
use serde_json::Value;

use crate::theme::Theme;

use super::{args_of, is_error, paint, result_text, status_line, styled, truncate, ToolRenderer};

/// Summary budget for the objective text.
const SUMMARY_BUDGET: usize = 60;

/// Renderer for the goal tool (`create|get|complete|resume|drop`).
pub struct Goal;

impl ToolRenderer for Goal {
    fn summary(&self, call: &Value) -> String {
        let Some(args) = args_of(call) else {
            return String::new();
        };
        let op = args.get("op").and_then(Value::as_str).unwrap_or("");
        let objective = args
            .get("objective")
            .and_then(Value::as_str)
            .map_or(String::new(), |o| truncate(o, SUMMARY_BUDGET));
        match (op.is_empty(), objective.is_empty()) {
            (true, _) => objective,
            (false, true) => op.to_owned(),
            (false, false) => format!("{op} {objective}"),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let op = args_of(call)
            .and_then(|a| a.get("op"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let settled = call.get("result").is_some();
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if settled {
            theme.glyph("success")
        } else {
            "⧗"
        };
        let mut lines = vec![status_line(theme, icon, "Goal", op, Vec::new())];
        if let Some(text) = result_text(call) {
            let text = text.trim();
            if !text.is_empty() {
                if text.to_lowercase().contains("no active goal") {
                    lines.push(Line::from(styled(
                        theme,
                        "warning",
                        format!("{} no active goal", theme.symbol("status.warning")),
                    )));
                } else {
                    for line in text.lines().take(3) {
                        lines.push(Line::from(styled(theme, "text_muted", line.to_owned())));
                    }
                }
            }
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
    fn summary_joins_op_and_objective() {
        let call =
            json!({"name": "goal", "arguments": {"op": "create", "objective": "port the TUI"}});
        assert_eq!(Goal.summary(&call), "create port the TUI");
    }

    #[test]
    fn pending_renders_hourglass_line() {
        let backend = TestBackend::new(60, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let call = json!({"name": "goal", "arguments": {"op": "get"}});
        terminal
            .draw(|f| Goal.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains('⧗'));
        assert!(text.contains("Goal"));
    }

    #[test]
    fn no_active_goal_warns() {
        let backend = TestBackend::new(60, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        let call = json!({"name": "goal", "arguments": {"op": "get"},
            "result": {"content": [{"type": "text", "text": "no active goal"}]}});
        terminal
            .draw(|f| Goal.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("no active goal"));
    }
}
