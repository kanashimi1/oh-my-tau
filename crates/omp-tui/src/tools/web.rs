//! `web_search`/`fetch` renderer — port of `pi-tui/src/tools/web-search-tool.ts`.
//!
//! Stub: summary only (query). Planned: result list with sources, fetched-page
//! truncation notice, and answer citation markers.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for web tools.
pub struct Web;

impl ToolRenderer for Web {
    fn summary(&self, call: &Value) -> String {
        let args = call.get("arguments").or_else(|| call.get("input"));
        args.and_then(|a| a.get("query"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}
