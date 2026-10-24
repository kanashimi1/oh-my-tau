//! Tool call renderers — ports of `pi-tui/src/tools/` and `pi-tui/src/render/`.
//!
//! Every tool family gets its own module with one [`ToolRenderer`]
//! implementation; [`render`] dispatches on the tool name from a settled
//! assistant message (`content[]` entries with `type: "toolCall"` /
//! `toolcall`, TS `pi-wire` shapes) or a `tool_execution_*` frame. The omp
//! tool inventory lives in omp-cli `src/tools/` (`bash`, `edit`, `read`,
//! `write`, `search` (glob+grep), `find`, `task`, `web_search`, `ask`,
//! `todo`, `think`, `mcp`, `lsp`, `checkpoint`, `jobs`, `eval`, `ast`,
//! `wait`, `goal`); spec details per family follow the pi-tui
//! tool renderers.
//!
//! Shared chrome (collapse/expand glyphs, elapsed-time spinner, exit-code
//! badge) lives here: [`status_line`] is the `renderStatusLine` port and
//! [`args_inline`] is the `formatArgsInline` port used by the default
//! renderer and the MCP renderer.

pub mod ask;
pub mod ast;
pub mod bash;
pub mod checkpoint;
pub mod edit;
pub mod eval;
pub mod find;
pub(crate) mod frame;
pub mod goal;
pub mod jobs;
pub mod lsp;
pub mod mcp;
pub mod read;
pub mod search;
pub mod task;
pub mod think;
pub mod todo;
pub mod wait;
pub mod web;
pub mod write;

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use serde_json::Value;

use crate::theme::Theme;

/// Render one tool call's visual into the chat stream.
pub trait ToolRenderer {
    /// Short one-line summary used in the collapsed state, e.g.
    /// `● bash cargo test` (TS `tool-summary` line).
    fn summary(&self, call: &Value) -> String;

    /// Full render for the expanded state.
    fn render(&self, frame: &mut Frame<'_>, area: Rect, call: &Value, theme: &Theme);

    /// Render a call + its settled result as transcript lines. Default:
    /// merge into a synthetic `tool_execution_end`-shaped frame and defer to
    /// [`ToolRenderer::render_frame`], so families only implement one seam.
    fn render_call_result(
        &self,
        call: &Value,
        result: &Value,
        expanded: bool,
        theme: &Theme,
    ) -> Vec<ratatui::text::Line<'static>> {
        let frame = serde_json::json!({
            "toolCallId": result.get("toolCallId").unwrap_or(&serde_json::Value::Null),
            "toolName": call.get("name").unwrap_or(&serde_json::Value::Null),
            "args": call.get("arguments").unwrap_or(&serde_json::Value::Null),
            "result": result,
        });
        self.render_frame(&frame, expanded, theme)
    }

    /// Render a settled/streaming tool-execution frame
    /// (`tool_execution_start|update|end` JSON payload: `{toolCallId,
    /// toolName, args}` / `{toolCallId, toolName, result}`) into transcript
    /// lines. Default: a pending one-liner via [`ToolRenderer::summary`].
    fn render_frame<'a>(
        &self,
        frame: &Value,
        expanded: bool,
        theme: &'a Theme,
    ) -> Vec<ratatui::text::Line<'static>> {
        let _ = expanded;
        let name = frame
            .get("toolName")
            .and_then(Value::as_str)
            .unwrap_or("tool");
        let call = frame
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"name": name}));
        vec![ratatui::text::Line::from(format!(
            "{} {name} {}",
            theme.glyph("collapsed"),
            self.summary(&call)
        ))]
    }
}

/// Dispatch a tool call payload to its family's renderer. MVP: a themed
/// one-line summary for every tool; expanded views come with each family.
pub fn render(frame: &mut Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
    let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
    let summary = family(name).summary(call);
    let line = Line::from(vec![
        Span::styled(
            theme.glyph("collapsed"),
            Style::default().fg(theme.color("accent")),
        ),
        Span::raw(" "),
        Span::styled(name.to_owned(), Style::default().fg(theme.color("accent"))),
        Span::raw(format!(" {summary}")),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// The renderer for a tool family name (unknown names fall back to the
/// generic default renderer, §17 of `tools-a.md`).
#[must_use]
pub fn family(name: &str) -> &'static dyn ToolRenderer {
    match name {
        "bash" => &bash::Bash,
        "edit" | "apply_patch" => &edit::Edit,
        "read" => &read::Read,
        "write" => &write::Write,
        "search" | "grep" => &search::Grep,
        "glob" => &search::Glob,
        "find" => &find::Find,
        "task" | "subprocess" => &task::Task,
        "web_search" | "fetch" => &web::Web,
        "ask" => &ask::Ask,
        "todo" => &todo::Todo,
        "think" => &think::Think,
        "lsp" => &lsp::Lsp,
        "checkpoint" | "rewind" => &checkpoint::Checkpoint,
        "jobs" => &jobs::Jobs,
        "eval" => &eval::Eval,
        "ast_grep" | "ast_edit" => &ast::Ast,
        "wait" => &wait::Wait,
        "goal" => &goal::Goal,
        _ if name.starts_with(mcp::MCP_NAME_PREFIX) => &mcp::Mcp,
        _ => &default_renderer::Default,
    }
}

/// Generic fallback renderer (`default-renderer.ts`) for tools without a
/// bespoke renderer: status line + inline args + capped output pane.
mod default_renderer {
    use ratatui::layout::Rect;
    use ratatui::text::Line;
    use serde_json::Value;

    use crate::theme::Theme;

    use super::{
        args_inline, args_of, json_tree_lines, paint, result_text, status_line, styled,
        ToolRenderer, JSON_TREE_MAX_DEPTH_EXPANDED, JSON_TREE_MAX_LINES_EXPANDED,
        JSON_TREE_SCALAR_LEN_EXPANDED,
    };

    /// Collapsed output pane rows (`PREVIEW_LIMITS.OUTPUT_COLLAPSED`).
    const OUTPUT_COLLAPSED: usize = 3;
    /// Inline-args budget for the collapsed head row.
    const INLINE_BUDGET: usize = 60;

    /// Fallback renderer for unknown tool families.
    pub struct Default;

    impl ToolRenderer for Default {
        fn summary(&self, call: &Value) -> String {
            args_inline(args_of(call), INLINE_BUDGET)
        }

        fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
            let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
            let is_error = call
                .get("result")
                .and_then(|r| r.get("isError"))
                .or_else(|| call.get("isError"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let icon = if is_error {
                theme.glyph("failure")
            } else {
                theme.glyph("success")
            };
            let mut lines = vec![status_line(
                theme,
                icon,
                name,
                &args_inline(args_of(call), INLINE_BUDGET),
                Vec::new(),
            )];
            lines.extend(output_pane_lines(
                theme,
                result_text(call).as_deref(),
                false,
            ));
            paint(frame, area, lines);
        }

        fn render_frame(&self, frame: &Value, expanded: bool, theme: &Theme) -> Vec<Line<'static>> {
            let name = frame
                .get("toolName")
                .and_then(Value::as_str)
                .unwrap_or("tool");
            let result = frame.get("result");
            let settled = result.is_some();
            let is_error = result
                .and_then(|r| r.get("isError"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let icon = if is_error {
                theme.glyph("failure")
            } else if settled {
                theme.glyph("success")
            } else {
                "⧗"
            };
            let args = frame
                .get("args")
                .filter(|a| a.is_object() && !a.as_object().is_some_and(serde_json::Map::is_empty));
            let description = if settled {
                String::new()
            } else {
                args_inline(args, INLINE_BUDGET)
            };
            let mut lines = vec![status_line(theme, icon, name, &description, Vec::new())];
            if let Some(args) = args {
                if expanded {
                    // Expanded Args section: dim header + JSON tree
                    // (TS `default-renderer.ts` / result-card §17, §22).
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
                } else {
                    let inline = args_inline(Some(args), INLINE_BUDGET);
                    if !inline.is_empty() && inline != "{}" {
                        lines.push(Line::from(styled(
                            theme,
                            "text_muted",
                            format!(" └─ {inline}"),
                        )));
                    }
                }
            }
            if settled {
                lines.extend(output_pane_lines(
                    theme,
                    result_text(frame).as_deref(),
                    expanded,
                ));
            }
            lines
        }
    }

    /// Capped output pane shared by the frame and transcript renderers:
    /// `(no output)` placeholder, JSON-tree passthrough is left to callers
    /// that pre-parse; here the raw text rows are capped to the collapsed
    /// budget with a dim expand hint.
    fn output_pane_lines(theme: &Theme, text: Option<&str>, _expanded: bool) -> Vec<Line<'static>> {
        let Some(text) = text else {
            return Vec::new();
        };
        let body: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        if body.is_empty() {
            return vec![Line::from(styled(theme, "text_muted", "(no output)"))];
        }
        let cap = OUTPUT_COLLAPSED.min(body.len());
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
}

// ─── Shared payload helpers (generic infra used by every family) ───────────

/// Borrow the arguments object of a tool-call payload.
pub(crate) fn args_of(call: &Value) -> Option<&Value> {
    call.get("arguments").or_else(|| call.get("input"))
}

/// Borrow the settled details object, if any.
pub(crate) fn details_of(call: &Value) -> Option<&Value> {
    call.get("result")
        .and_then(|r| r.get("details"))
        .or_else(|| call.get("details"))
}

/// Whether the settled result is flagged as an error.
pub(crate) fn is_error(call: &Value) -> bool {
    call.get("result")
        .and_then(|r| r.get("isError"))
        .or_else(|| call.get("isError"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Collect the model-facing text of a settled result.
pub(crate) fn result_text(call: &Value) -> Option<String> {
    let result = call.get("result").or(Some(call));
    if let Some(text) = result.and_then(|r| r.get("text")).and_then(Value::as_str) {
        return Some(text.to_owned());
    }
    if let Some(out) = result.and_then(|r| r.get("output")).and_then(Value::as_str) {
        return Some(out.to_owned());
    }
    let content = result?.get("content")?;
    let mut joined = String::new();
    for piece in content.as_array()? {
        if piece.get("type").and_then(Value::as_str) == Some("text") {
            if !joined.is_empty() {
                joined.push('\n');
            }
            joined.push_str(piece.get("text").and_then(Value::as_str).unwrap_or(""));
        }
    }
    Some(joined).filter(|s| !s.is_empty())
}

/// One styled, owned display span.
pub(crate) fn styled(theme: &Theme, token: &str, text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(theme.color(token)))
}

/// Render accumulated lines into the frame, truncated by the area.
pub(crate) fn paint(frame: &mut ratatui::Frame<'_>, area: Rect, lines: Vec<Line<'static>>) {
    frame.render_widget(Paragraph::new(lines), area);
}

/// `renderStatusLine` equivalent: `icon title: description · meta…`.
pub(crate) fn status_line(
    theme: &Theme,
    icon: &str,
    title: &str,
    description: &str,
    meta: Vec<Span<'static>>,
) -> Line<'static> {
    let mut spans = vec![styled(theme, "accent", format!("{icon} {title}:"))];
    if !description.is_empty() {
        spans.push(Span::raw(" "));
        spans.push(styled(theme, "text", description));
    }
    for m in meta {
        spans.push(Span::raw(" "));
        spans.push(m);
    }
    Line::from(spans)
}

/// `formatArgsInline` port: compact single-line JSON preview of the args,
/// truncated to `budget` chars with a Unicode ellipsis.
pub(crate) fn args_inline(args: Option<&Value>, budget: usize) -> String {
    let Some(args) = args else {
        return String::new();
    };
    let compact = serde_json::to_string(args).unwrap_or_default();
    let compact = compact.replace('\\', "");
    truncate(&compact, budget)
}

/// Error-detail body lines (TS `formatErrorDetail` semantics): the leading
/// `Error: ` prefix that `formatErrorMessage` adds for standalone single-line
/// errors is stripped, every remaining line is indented two columns to sit
/// under the card header, colored with the `error` token, and width-truncated.
#[must_use]
pub fn error_detail_lines(text: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let inner = width.saturating_sub(4).max(1);
    let style = Style::default().fg(theme.color("error"));
    text.lines()
        .map(|l| {
            let l = l.strip_prefix("Error: ").unwrap_or(l);
            Line::from(Span::styled(
                format!("  {}", truncate_width(l, inner)),
                style,
            ))
        })
        .collect()
}

/// Truncate to `width` chars with a Unicode ellipsis (no whitespace
/// flattening — unlike [`truncate`], meant for already-single-line rows).
fn truncate_width(s: &str, width: usize) -> String {
    let count = s.chars().count();
    if count <= width {
        return s.to_owned();
    }
    match width {
        0 => String::new(),
        1 => "…".to_owned(),
        _ => {
            let kept: String = s.chars().take(width - 1).collect();
            format!("{kept}…")
        }
    }
}

// ─── JSON tree (TS `json-tree.ts`, used by the expanded Args sections) ─────

/// Max depth for a collapsed JSON tree (`JSON_TREE_MAX_DEPTH_COLLAPSED`).
pub const JSON_TREE_MAX_DEPTH_COLLAPSED: usize = 2;
/// Max depth for an expanded JSON tree (`JSON_TREE_MAX_DEPTH_EXPANDED`).
pub const JSON_TREE_MAX_DEPTH_EXPANDED: usize = 6;
/// Max rows for a collapsed JSON tree (`JSON_TREE_MAX_LINES_COLLAPSED`).
pub const JSON_TREE_MAX_LINES_COLLAPSED: usize = 6;
/// Max rows for an expanded JSON tree (`JSON_TREE_MAX_LINES_EXPANDED`).
pub const JSON_TREE_MAX_LINES_EXPANDED: usize = 200;
/// Max scalar width, collapsed (`JSON_TREE_SCALAR_LEN_COLLAPSED`).
pub const JSON_TREE_SCALAR_LEN_COLLAPSED: usize = 60;
/// Max scalar width, expanded (`JSON_TREE_SCALAR_LEN_EXPANDED`).
pub const JSON_TREE_SCALAR_LEN_EXPANDED: usize = 2000;

/// Render a JSON value as tree rows (TS `renderJsonTreeLines`): objects and
/// arrays branch with `├─`/`└─` connectors, scalars are summarized inline,
/// and nodes past `max_depth` collapse to `[n items]` / `{n keys}`.
/// Returns the rows plus a `truncated` flag (row or scalar budget hit).
#[must_use]
pub fn json_tree_lines(
    value: &Value,
    max_depth: usize,
    max_lines: usize,
    max_scalar_len: usize,
) -> (Vec<String>, bool) {
    let mut lines = Vec::new();
    let mut truncated = false;
    let roots: Vec<(String, &Value)> = match value {
        Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v)).collect(),
        Value::Array(arr) => arr
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("[{i}]"), v))
            .collect(),
        other => vec![(String::new(), other)],
    };
    let last = roots.len().saturating_sub(1);
    for (i, (key, child)) in roots.iter().enumerate() {
        if lines.len() >= max_lines {
            truncated = true;
            break;
        }
        walk_json(
            &mut lines,
            &mut truncated,
            key,
            child,
            0,
            i == last,
            "",
            max_depth,
            max_lines,
            max_scalar_len,
        );
    }
    (lines, truncated)
}

#[allow(clippy::too_many_arguments)]
fn walk_json(
    lines: &mut Vec<String>,
    truncated: &mut bool,
    key: &str,
    value: &Value,
    depth: usize,
    last: bool,
    prefix: &str,
    max_depth: usize,
    max_lines: usize,
    max_scalar_len: usize,
) {
    if lines.len() >= max_lines {
        *truncated = true;
        return;
    }
    let connector = if last { "└─" } else { "├─" };
    let header = if key.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}{connector} {key}")
    };
    let summary = |v: &Value| -> String {
        match v {
            Value::Null => "null".to_owned(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => {
                let escaped = s.replace('\n', "\\n").replace('\t', "\\t");
                format!("\"{}\"", truncate_width(&escaped, max_scalar_len))
            }
            Value::Array(a) => format!("[{} items]", a.len()),
            Value::Object(o) => format!("{{{} keys}}", o.len()),
        }
    };
    let children: Vec<(String, &Value)> = match value {
        Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v)).collect(),
        Value::Array(arr) => arr
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("[{i}]"), v))
            .collect(),
        _ => Vec::new(),
    };
    let expandable = !children.is_empty() && depth + 1 < max_depth;
    if !expandable {
        lines.push(format!("{header}: {}", summary(value)));
        return;
    }
    lines.push(header);
    let child_prefix = format!("{prefix}{}", if last { "   " } else { "│  " });
    let child_last = children.len().saturating_sub(1);
    for (i, (ck, cv)) in children.iter().enumerate() {
        if lines.len() >= max_lines {
            *truncated = true;
            break;
        }
        walk_json(
            lines,
            truncated,
            ck,
            cv,
            depth + 1,
            i == child_last,
            &child_prefix,
            max_depth,
            max_lines,
            max_scalar_len,
        );
    }
}

/// Truncate to `budget` chars (Unicode-ellipsis), flattening newlines.
pub(crate) fn truncate(text: &str, budget: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= budget {
        return flat;
    }
    let kept: String = flat.chars().take(budget.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn family_dispatches_every_omp_tool_without_panic() {
        // Every name in the omp inventory must resolve to a renderer whose
        // summary tolerates an empty payload.
        for name in [
            "bash",
            "edit",
            "apply_patch",
            "read",
            "write",
            "search",
            "glob",
            "grep",
            "find",
            "task",
            "subprocess",
            "web_search",
            "fetch",
            "ask",
            "todo",
            "think",
            "lsp",
            "checkpoint",
            "rewind",
            "jobs",
            "eval",
            "ast_grep",
            "ast_edit",
            "wait",
            "goal",
            "mcp__fs__read_text_file",
            "mcp__hub_status",
            "totally_unknown_tool",
        ] {
            let _ = family(name).summary(&json!({"name": name, "arguments": {}}));
            let _ = family(name).summary(&json!({"name": name}));
        }
    }

    #[test]
    fn mcp_family_uses_server_tool_label() {
        let call = json!({"name": "mcp__fs__read_text_file", "arguments": {}});
        assert_eq!(
            family("mcp__fs__read_text_file").summary(&call),
            "fs/read_text_file"
        );
    }

    #[test]
    fn unknown_tool_uses_default_renderer() {
        let call = json!({"name": "frobnicate", "arguments": {"x": 1}});
        assert_eq!(family("frobnicate").summary(&call), "{\"x\":1}");
    }

    #[test]
    fn args_inline_truncates_with_ellipsis() {
        let args = json!({"thoughts": "a".repeat(200)});
        let s = args_inline(Some(&args), 60);
        assert!(s.ends_with('…'));
        assert!(s.chars().count() <= 60);
    }

    #[test]
    fn result_text_joins_content_pieces() {
        let call = json!({"result": {"content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]}});
        assert_eq!(result_text(&call).as_deref(), Some("one\ntwo"));
    }

    #[test]
    fn error_detail_lines_strip_prefix_indent_and_color() {
        let theme = Theme::default();
        let lines = error_detail_lines("Error: boom\nsecond line", 80, &theme);
        assert_eq!(lines.len(), 2);
        for (i, want) in ["  boom", "  second line"].iter().enumerate() {
            let got: String = lines[i]
                .spans
                .iter()
                .map(|s| s.content.to_string())
                .collect();
            assert_eq!(&got, want);
            assert!(lines[i]
                .spans
                .iter()
                .all(|s| s.style.fg == Some(theme.color("error"))));
        }
    }

    #[test]
    fn error_detail_lines_truncates_to_width() {
        let theme = Theme::default();
        let long = "x".repeat(200);
        let lines = error_detail_lines(&format!("Error: {long}"), 40, &theme);
        let got: String = lines[0]
            .spans
            .iter()
            .map(|s| s.content.to_string())
            .collect();
        assert!(got.ends_with('…'), "{got}");
        assert!(got.chars().count() <= 40, "{got}");
    }

    #[test]
    fn json_tree_lines_render_branches_and_scalars() {
        let value = json!({"a": 1, "b": {"c": [1, 2], "d": "x"}, "e": null});
        let (rows, truncated) = json_tree_lines(
            &value,
            JSON_TREE_MAX_DEPTH_EXPANDED,
            JSON_TREE_MAX_LINES_EXPANDED,
            JSON_TREE_SCALAR_LEN_EXPANDED,
        );
        assert!(!truncated, "{rows:?}");
        assert_eq!(rows[0], "├─ a: 1");
        assert!(rows.iter().any(|r| r == "├─ b"), "{rows:?}");
        assert!(rows.iter().any(|r| r == "│  ├─ c"), "{rows:?}");
        assert!(rows.iter().any(|r| r == "│  │  ├─ [0]: 1"), "{rows:?}");
        assert!(rows.iter().any(|r| r == "│  └─ d: \"x\""), "{rows:?}");
        assert!(rows.iter().any(|r| r == "└─ e: null"), "{rows:?}");
    }

    #[test]
    fn json_tree_lines_depth_cap_summarizes_containers() {
        let value = json!({"b": {"c": 1}, "arr": [true]});
        let (rows, truncated) = json_tree_lines(&value, 1, 200, 60);
        assert!(!truncated);
        assert_eq!(rows[0], "├─ b: {1 keys}");
        assert_eq!(rows[1], "└─ arr: [1 items]");
    }

    #[test]
    fn json_tree_lines_truncates_on_line_budget() {
        let value = json!({"a": 1, "b": 2, "c": 3});
        let (rows, truncated) = json_tree_lines(&value, 6, 2, 60);
        assert!(truncated);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn default_renderer_expanded_args_is_json_tree() {
        let theme = Theme::default();
        let frame = json!({
            "toolCallId": "1",
            "toolName": "frobnicate",
            "args": {"x": 1, "nested": {"y": [true, null]}},
            "result": {"content": [{"type": "text", "text": "done"}], "isError": false},
        });
        let flat = |lines: &[ratatui::text::Line<'static>]| -> Vec<String> {
            lines
                .iter()
                .map(|l| l.spans.iter().map(|s| s.content.to_string()).collect())
                .collect()
        };
        let expanded = flat(&family("frobnicate").render_frame(&frame, true, &theme));
        assert!(expanded.iter().any(|l| l == "Args"), "{expanded:?}");
        assert!(
            expanded.iter().any(|l| l.contains("├─ x: 1")),
            "{expanded:?}"
        );
        assert!(
            expanded.iter().any(|l| l.contains("nested")),
            "{expanded:?}"
        );
        let collapsed = flat(&family("frobnicate").render_frame(&frame, false, &theme));
        assert!(!collapsed.iter().any(|l| l == "Args"), "{collapsed:?}");
        assert!(collapsed.iter().any(|l| l.contains("└─")), "{collapsed:?}");
    }
}
