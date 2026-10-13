//! `bash` tool renderer — port of `pi-tui/src/tools/bash-tool.ts`.
//!
//! Stub: summary only (command + running/finished state). Planned: the
//! collapsed line with the spinner while running, exit-code badge
//! (green ✓ / red ✗ + code), expanded output pane with the pager
//! (`render/output-pane.ts`) and the running-bash abort affordance.

use serde_json::Value;

use crate::theme::Theme;

use super::ToolRenderer;

/// Renderer for shell commands.
pub struct Bash;

impl ToolRenderer for Bash {
    fn summary(&self, call: &Value) -> String {
        let cmd = call
            .get("arguments")
            .and_then(|a| a.get("command"))
            .or_else(|| call.get("input").and_then(|i| i.get("command")))
            .and_then(Value::as_str)
            .unwrap_or("");
        let cmd = cmd.lines().next().unwrap_or("");
        cmd.chars().take(80).collect()
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: ratatui::layout::Rect, call: &Value, theme: &Theme) {
        super::render(frame, area, call, theme); // generic one-liner for now
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn summary_shows_first_command_line() {
        let call = json!({"name":"bash","arguments":{"command":"cargo test\n"}});
        assert_eq!(Bash.summary(&call), "cargo test");
    }
}
