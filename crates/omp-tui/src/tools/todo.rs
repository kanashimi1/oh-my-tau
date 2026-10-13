//! `todo` tool renderer — port of `pi-tui/src/tools/todo-tool.ts` (the
//! phased task list that also feeds the status line).
//!
//! Stub: summary only (counts). Planned: checkbox rows with phase headers,
//! streaming updates while the tool streams, and `todo_phases` extraction
//! for the status line slot.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for the todo tool.
pub struct Todo;

impl ToolRenderer for Todo {
    fn summary(&self, call: &Value) -> String {
        let args = call.get("arguments").or_else(|| call.get("input"));
        let n = args
            .and_then(|a| a.get("todos"))
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        format!("{n} items")
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}
