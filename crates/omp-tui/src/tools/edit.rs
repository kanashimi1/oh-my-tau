//! `edit`/`write` tool renderer — port of `pi-tui/src/tools/edit-tool.ts`
//! (+ `write-tool.ts`), backed by the diff view in `render/` and the
//! vendored `pi-diff`/`pi-edit` crates.
//!
//! Stub: summary only (path). Planned: unified diff with syntax highlighting,
//! per-hunk collapse, and the read-permission prompt chrome.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for file edits and writes.
pub struct Edit;

impl ToolRenderer for Edit {
    fn summary(&self, call: &Value) -> String {
        let path = call
            .get("arguments")
            .and_then(|a| a.get("path"))
            .or_else(|| call.get("input").and_then(|i| i.get("path")))
            .and_then(Value::as_str)
            .unwrap_or("");
        path.to_owned()
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}
