//! Search-family renderer (`search`/`glob`/`grep`/`find`) — port of
//! `pi-tui/src/tools/search-tool.ts` and `render/file-list.ts`.
//!
//! Stub: summary only (pattern). Planned: hit counts, file list with match
//! rows, and the truncation notice when results exceed the display budget.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for search tools.
pub struct Search;

impl ToolRenderer for Search {
    fn summary(&self, call: &Value) -> String {
        let args = call.get("arguments").or_else(|| call.get("input"));
        let pattern = args
            .and_then(|a| a.get("pattern"))
            .or_else(|| args.and_then(|a| a.get("regex")))
            .and_then(Value::as_str)
            .unwrap_or("");
        let path = args
            .and_then(|a| a.get("path"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if path.is_empty() {
            pattern.to_owned()
        } else {
            format!("{pattern} in {path}")
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}
