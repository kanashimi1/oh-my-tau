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
    args_inline, args_of, error_detail_lines, is_error, json_tree_lines, paint, result_text,
    status_line, styled, ToolRenderer, JSON_TREE_MAX_DEPTH_EXPANDED, JSON_TREE_MAX_LINES_EXPANDED,
    JSON_TREE_SCALAR_LEN_EXPANDED,
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
            theme.symbol("cmd.gear")
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
            Some(text) => lines.extend(output_pane_lines(theme, &text, false)),
        }
        paint(frame, area, lines);
    }

    fn render_frame(&self, frame: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
        let name = frame
            .get("toolName")
            .and_then(Value::as_str)
            .unwrap_or("MCP");
        let label = parse_mcp_name(name).unwrap_or_else(|| "MCP".to_owned());
        let result = frame.get("result");
        let settled = result.is_some();
        let error = result
            .and_then(|r| r.get("isError"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let icon = if error {
            theme.glyph("failure")
        } else if settled {
            theme.symbol("cmd.gear")
        } else {
            "⧗"
        };
        let mut lines = vec![status_line(theme, icon, &label, "", Vec::new())];
        let args = frame
            .get("args")
            .filter(|a| a.is_object() && !a.as_object().is_some_and(serde_json::Map::is_empty));
        if !settled {
            let inline = args_inline(args, INLINE_BUDGET);
            if !inline.is_empty() && inline != "{}" {
                lines.push(Line::from(styled(
                    theme,
                    "text_muted",
                    format!(" └─ {inline}"),
                )));
            }
            return lines;
        }
        if let Some(args) = args.filter(|_| expanded) {
            // Expanded Args section (TS `mcp.ts` `buildMcpArgsSection`).
            lines.push(Line::from(styled(theme, "text_muted", "Args")));
            let (tree, truncated) = json_tree_lines(
                args,
                JSON_TREE_MAX_DEPTH_EXPANDED,
                JSON_TREE_MAX_LINES_EXPANDED,
                JSON_TREE_SCALAR_LEN_EXPANDED,
            );
            for row in tree {
                lines.push(Line::from(styled(theme, "toolOutput", row)));
            }
            if truncated {
                lines.push(Line::from(styled(theme, "text_muted", "…")));
            }
        }
        if error {
            // Error results render the error detail (TS `formatErrorDetail`):
            // `Error: ` prefix stripped, two-column indent, error color.
            match result_text(frame) {
                Some(text) if !text.trim().is_empty() => {
                    lines.extend(error_detail_lines(&text, usize::MAX, theme));
                }
                _ => lines.push(Line::from(styled(theme, "text_muted", "(no output)"))),
            }
            return lines;
        }
        match result_text(frame) {
            None => lines.push(Line::from(styled(theme, "text_muted", "(no output)"))),
            Some(text) => lines.extend(output_pane_lines(theme, &text, expanded)),
        }
        lines
    }
}

/// Capped output pane shared by the frame and transcript renderers: JSON
/// text pretty-prints as a tree, raw text is capped to the collapsed budget
/// with a dim expand hint (`formatOutputPaneLines` simplified to hard caps).
fn output_pane_lines(theme: &Theme, text: &str, expanded: bool) -> Vec<Line<'static>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![Line::from(styled(theme, "text_muted", "(no output)"))];
    }
    if let Some(parsed) = parse_json_tree(trimmed) {
        let cap = if expanded {
            OUTPUT_EXPANDED.min(parsed.len())
        } else {
            OUTPUT_COLLAPSED.min(parsed.len())
        };
        let mut lines: Vec<Line<'static>> = parsed[..cap]
            .iter()
            .map(|row| Line::from(styled(theme, "toolOutput", row.clone())))
            .collect();
        if parsed.len() > cap {
            lines.push(Line::from(styled(
                theme,
                "text_muted",
                format!("… {} more lines [ctrl+o: Expand]", parsed.len() - cap),
            )));
        }
        return lines;
    }
    let body: Vec<&str> = trimmed.lines().collect();
    let cap = if expanded {
        OUTPUT_EXPANDED.min(body.len())
    } else {
        OUTPUT_COLLAPSED.min(body.len())
    };
    let mut lines: Vec<Line<'static>> = body[..cap]
        .iter()
        .map(|line| Line::from(styled(theme, "toolOutput", (*line).to_owned())))
        .collect();
    if body.len() > cap {
        lines.push(Line::from(styled(
            theme,
            "text_muted",
            format!("… {} more lines [ctrl+o: Expand]", body.len() - cap),
        )));
    }
    lines
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

    fn flat(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
            .collect()
    }

    #[test]
    fn error_result_renders_error_detail_rows() {
        let frame = json!({
            "toolCallId": "1", "toolName": "mcp__fs__read", "args": {"path": "/x"},
            "result": {"content": [{"type": "text", "text": "Error: no such file"}],
                       "isError": true}
        });
        let lines = Mcp.render_frame(&frame, false, &Theme::default());
        let flat = flat(&lines);
        assert!(flat[0].contains("fs/read"), "{flat:?}");
        assert!(flat.iter().any(|l| l == "  no such file"), "{flat:?}");
        assert!(
            !flat.iter().any(|l| l.contains("Error: no such")),
            "{flat:?}"
        );
    }

    #[test]
    fn expanded_result_renders_args_json_tree() {
        let frame = json!({
            "toolCallId": "1", "toolName": "mcp__fs__write", "args": {"path": "/x", "opts": {"deep": true}},
            "result": {"content": [{"type": "text", "text": "ok"}], "isError": false}
        });
        let theme = Theme::default();
        let expanded = flat(&Mcp.render_frame(&frame, true, &theme));
        assert!(expanded.iter().any(|l| l == "Args"), "{expanded:?}");
        assert!(
            expanded.iter().any(|l| l.contains("├─ path: \"/x\"")),
            "{expanded:?}"
        );
        assert!(expanded.iter().any(|l| l.contains("opts")), "{expanded:?}");
        // Collapsed: no Args section, inline row instead.
        let collapsed = flat(&Mcp.render_frame(&frame, false, &theme));
        assert!(!collapsed.iter().any(|l| l == "Args"), "{collapsed:?}");
    }

    #[test]
    fn pending_frame_renders_dim_args_row() {
        let frame = json!({"toolCallId": "1", "toolName": "mcp__fs__read", "args": {"path": "/x"}});
        let flat = flat(&Mcp.render_frame(&frame, false, &Theme::default()));
        assert!(flat[0].contains('⧗'), "{flat:?}");
        assert!(flat.iter().any(|l| l.contains("└─")), "{flat:?}");
    }
}
