//! `task` (subagent) renderer — port of `pi-tui/src/tools/task-tool.ts` and
//! the subagent tree in `chat/`.
//!
//! Stub: summary only (subagent type + subject). Planned: nested agent tree,
//! per-agent status spinner, collapsed transcript, and the agent-end summary
//! badge.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for subagent spawns.
pub struct Task;

impl ToolRenderer for Task {
    fn summary(&self, call: &Value) -> String {
        let args = call.get("arguments").or_else(|| call.get("input"));
        let subject = args
            .and_then(|a| a.get("subject"))
            .or_else(|| args.and_then(|a| a.get("prompt")))
            .and_then(Value::as_str)
            .unwrap_or("");
        let subject: String = subject.chars().take(60).collect();
        subject
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}
