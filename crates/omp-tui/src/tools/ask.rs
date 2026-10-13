//! `ask` tool renderer — port of the ask-dialog chrome in
//! `pi-tui/src/tools/` (`ask-dialog.ts`) wired to `extension_ui.rs`.
//!
//! Stub: summary only (question). Planned: interactive option list when the
//! dialog is interactive, timeout countdown, and the answered-value badge.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for the ask tool.
pub struct Ask;

/// Fallback renderer for not-yet-styled tools.
pub struct Generic;

impl ToolRenderer for Ask {
    fn summary(&self, call: &Value) -> String {
        question(call)
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}

impl ToolRenderer for Generic {
    fn summary(&self, call: &Value) -> String {
        question(call)
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme);
    }
}

fn question(call: &Value) -> String {
    let args = call.get("arguments").or_else(|| call.get("input"));
    args.and_then(|a| a.get("question"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .chars()
        .take(60)
        .collect()
}
