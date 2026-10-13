//! `read` tool renderer — port of `pi-tui/src/tools/read-tool.ts`.
//!
//! Stub: summary only (path + line range). Planned: line-numbered snippet
//! with line-range collapse and the "read outside cwd" badge.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for file reads.
pub struct Read;

impl ToolRenderer for Read {
    fn summary(&self, call: &Value) -> String {
        let args = call.get("arguments").or_else(|| call.get("input"));
        let path = args
            .and_then(|a| a.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let range = args
            .and_then(|a| a.get("offset"))
            .and_then(Value::as_u64)
            .map_or(String::new(), |o| format!(" @{}", o + 1));
        format!("{path}{range}")
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}
