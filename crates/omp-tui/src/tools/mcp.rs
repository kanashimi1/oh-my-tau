//! MCP tool renderer — port of `pi-tui/src/tools/mcp.ts` (spec `tools-b.md`
//! §9) for `mcp__<server>_<tool>` tool names.
//!
//! Call: `⧗ server/tool` status line plus one dim `└─ <inline args>` row.
//! Result: static MCP glyph on success, ✗ on error; output parsed as JSON →
//! pretty tree, raw text pane otherwise (4 collapsed / 12 expanded rows,
//! `(no output)` when empty).

use ratatui::layout::Rect;
use ratatui::text::Line;
use serde_json::Value;

use crate::theme::Theme;

use super::{
    args_inline, args_of, is_error, paint, result_text, status_line, styled, ToolRenderer,
};

/// Tool-name prefix for MCP tool calls (`MCP_TOOL_NAME_PREFIX`).
pub(crate) const MCP_NAME_PREFIX: &str = "mcp__";
/// Collapsed output pane rows (`PREVIEW_LIMITS.OUTPUT_COLLAPSED`).
const OUTPUT_COLLAPSED: usize = 4;
/// Expanded output pane rows (`PREVIEW_LIMITS.OUTPUT_EXPANDED`).
const OUTPUT_EXPANDED: usize = 12;
/// Inline args row budget (`max(20, width − tree.last − 2)` is settled later).
const INLINE_BUDGET: usize = 60;

/// Renderer for all `mcp__*` tool calls.
pub struct Mcp;

impl ToolRenderer for Mcp {
    fn summary(&self, call: &Value) -> String {
        let label = label_of(call);
        let args = args_inline(args_of(call), INLINE_BUDGET);
        if args.is_empty() || args == "{}" {
            label
        } else {
            format!("{label} {args}")
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let settled = call.get("result").is_some();
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if settled {
            "⚙"
        } else {
            "⧗"
        };
        let mut lines = vec![status_line(theme, icon, &label_of(call), "", Vec::new())];
        if !settled {
            let args = args_inline(args_of(call), INLINE_BUDGET);
            if !args.is_empty() && args != "{}" {
                lines.push(Line::from(styled(
                    theme,
                    "text_muted",
                    format!(" └─ {args}"),
                )));
            }
            paint(frame, area, lines);
            return;
        }
        match result_text(call) {
            None => lines.push(Line::from(styled(theme, "text_muted", "(no output)"))),
            Some(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    lines.push(Line::from(styled(theme, "text_muted", "(no output)")));
                } else if let Some(parsed) = parse_json_tree(trimmed) {
                    let cap = OUTPUT_COLLAPSED.min(parsed.len());
                    for row in &parsed[..cap] {
                        lines.push(Line::from(styled(theme, "toolOutput", row.clone())));
                    }
                    if parsed.len() > OUTPUT_EXPANDED {
                        lines.push(Line::from(styled(theme, "text_muted", "…")));
                    } else if parsed.len() > cap {
                        lines.push(Line::from(styled(
                            theme,
                            "text_muted",
                            format!("… {} more lines [ctrl+o: Expand]", parsed.len() - cap),
                        )));
                    }
                } else {
                    let body: Vec<&str> = trimmed.lines().collect();
                    let cap = OUTPUT_COLLAPSED.min(body.len());
                    for line in &body[..cap] {
                        lines.push(Line::from(styled(theme, "toolOutput", (*line).to_owned())));
                    }
                    if body.len() > cap {
                        lines.push(Line::from(styled(
                            theme,
                            "text_muted",
                            format!("… {} more lines [ctrl+o: Expand]", body.len() - cap),
                        )));
                    }
                }
            }
        }
        paint(frame, area, lines);
    }
}

/// `server/tool` label from the `mcp__<server>_<tool>` name.
#[must_use]
pub(crate) fn label_of(call: &Value) -> String {
    let name = call.get("name").and_then(Value::as_str).unwrap_or("MCP");
    parse_mcp_name(name).unwrap_or_else(|| "MCP".to_owned())
}

/// Split `mcp__<server>_<tool>` into `server/tool` (omp-rs mints single
/// underscore separators; `mcp__<server>__<tool>` double-underscore names
/// from the TS convention are handled first).
pub(crate) fn parse_mcp_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix(MCP_NAME_PREFIX)?;
    let (server, tool) = if let Some((s, t)) = rest.split_once("__") {
        (s, t)
    } else {
        rest.split_once('_')?
    };
    if server.is_empty() || tool.is_empty() {
        None
    } else {
        Some(format!("{server}/{tool}"))
    }
}

/// Pretty-print a JSON payload to shallow tree rows; `None` when not JSON.
fn parse_json_tree(text: &str) -> Option<Vec<String>> {
    if !text.starts_with('{') && !text.starts_with('[') {
        return None;
    }
    let value: Value = serde_json::from_str(text).ok()?;
    let pretty = serde_json::to_string_pretty(&value).ok()?;
    Some(pretty.lines().map(ToOwned::to_owned).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    #[test]
    fn parses_omp_minted_names() {
        assert_eq!(
            parse_mcp_name("mcp__fs__read_text_file").as_deref(),
            Some("fs/read_text_file")
        );
        assert_eq!(
            parse_mcp_name("mcp__hub_status").as_deref(),
            Some("hub/status")
        );
        assert_eq!(parse_mcp_name("bash"), None);
    }

    #[test]
    fn summary_includes_inline_args() {
        let call = json!({"name": "mcp__fs__read", "arguments": {"path": "/tmp/x"}});
        assert_eq!(Mcp.summary(&call), "fs/read {\"path\":\"/tmp/x\"}");
    }

    #[test]
    fn pending_renders_dim_args_row() {
        let call = json!({"name": "mcp__fs__read", "arguments": {"path": "/tmp/x"}});
        let backend = TestBackend::new(60, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Mcp.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains('⧗'));
        assert!(text.contains("fs/read"));
        assert!(text.contains("└─"));
    }

    #[test]
    fn json_output_renders_tree() {
        let call = json!({"name": "mcp__hub__status", "arguments": {},
            "result": {"content": [{"type": "text", "text": "{\"ok\":true,\"n\":2}"}]}});
        let backend = TestBackend::new(60, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Mcp.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("\"ok\": true"));
    }

    #[test]
    fn empty_output_renders_placeholder() {
        let call = json!({"name": "mcp__hub__ping", "arguments": {},
            "result": {"content": [{"type": "text", "text": ""}]}});
        let backend = TestBackend::new(60, 3);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Mcp.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("(no output)"));
    }
}
