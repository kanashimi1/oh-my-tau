//! `ast_grep` / `ast_edit` tool renderers — port of
//! `pi-tui/src/tools/ast-grep.ts` and `ast-edit.ts` (spec `tools-a.md` §14).
//!
//! `ast_grep` (`inline`): pending `⧗ AST Grep: <pattern> [lang]`; error ✘;
//! empty → warning `No matches found`; hits capped at
//! `COLLAPSED_MATCH_LIMIT` collapsed with a `… N more` summary; truncated →
//! warning icon + `truncated` meta. `ast_edit`: header badge
//! `N replacements`, body = change groups with linked paths.

use ratatui::layout::Rect;
use ratatui::text::Line;
use serde_json::Value;

use crate::theme::Theme;

use super::{args_of, is_error, paint, result_text, status_line, styled, truncate, ToolRenderer};

/// Collapsed hit/change cap (`COLLAPSED_MATCH_LIMIT` / `COLLAPSED_CHANGE_LIMIT`).
pub(crate) const COLLAPSED_MATCH_LIMIT: usize = 6;
/// Summary budget for the pattern.
const SUMMARY_BUDGET: usize = 60;

/// Renderer for the `ast_grep` and `ast_edit` tools.
pub struct Ast;

impl ToolRenderer for Ast {
    fn summary(&self, call: &Value) -> String {
        let Some(args) = args_of(call) else {
            return String::new();
        };
        if call.get("name").and_then(Value::as_str) == Some("ast_edit") {
            let ops = args
                .get("ops")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            let paths = args
                .get("paths")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            return format!("{ops} ops → {paths} paths");
        }
        let pat = args.get("pat").and_then(Value::as_str).unwrap_or("");
        let lang = args.get("lang").and_then(Value::as_str).unwrap_or("");
        match (pat.is_empty(), lang.is_empty()) {
            (true, _) => String::new(),
            (false, true) => truncate(pat, SUMMARY_BUDGET),
            (false, false) => format!("{} [{lang}]", truncate(pat, SUMMARY_BUDGET)),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let is_edit = call.get("name").and_then(Value::as_str) == Some("ast_edit");
        let title = if is_edit { "AST Edit" } else { "AST Grep" };
        let settled = call.get("result").is_some();
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if settled {
            theme.glyph("success")
        } else {
            "⧗"
        };
        let mut meta = Vec::new();
        if is_edit {
            if let Some(n) = replacement_count(call) {
                meta.push(styled(theme, "accent", format!("{n} replacements")));
            }
        } else if truncated(call) {
            meta.push(styled(theme, "warning", "truncated"));
        }
        let mut lines = vec![status_line(theme, icon, title, &self.summary(call), meta)];
        match result_text(call) {
            Some(text) => {
                let body: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
                if body.is_empty() || body.iter().any(|l| l.contains("No matches found")) {
                    lines.push(Line::from(styled(
                        theme,
                        "warning",
                        format!("{} No matches found", theme.symbol("status.warning")),
                    )));
                } else {
                    let cap = COLLAPSED_MATCH_LIMIT.min(body.len());
                    for line in &body[..cap] {
                        lines.push(Line::from(styled(theme, "toolOutput", (*line).to_owned())));
                    }
                    if body.len() > cap {
                        lines.push(Line::from(styled(
                            theme,
                            "text_muted",
                            format!("… {} more matches [ctrl+o: Expand]", body.len() - cap),
                        )));
                    }
                }
            }
            None => lines.push(Line::from(styled(theme, "text_muted", "…"))),
        }
        paint(frame, area, lines);
    }
}

/// Replacement count from result details (`ast_edit` details).
fn replacement_count(call: &Value) -> Option<u64> {
    call.get("result")?
        .get("details")
        .and_then(|d| d.get("replacements").or_else(|| d.get("replacementCount")))
        .and_then(Value::as_u64)
}

/// Whether the grep results were cut short by the tool.
fn truncated(call: &Value) -> bool {
    call.get("result")
        .and_then(|r| r.get("details"))
        .and_then(|d| d.get("truncated"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    #[test]
    fn grep_summary_includes_lang() {
        let call =
            json!({"name": "ast_grep", "arguments": {"pat": "fn $F() { $$ }", "lang": "rs"}});
        assert_eq!(Ast.summary(&call), "fn $F() { $$ } [rs]");
    }

    #[test]
    fn edit_summary_counts_ops_and_paths() {
        let call = json!({"name": "ast_edit", "arguments": {
            "ops": [{"pat": "a", "out": "b"}], "paths": ["src", "tests"]}});
        assert_eq!(Ast.summary(&call), "1 ops → 2 paths");
    }

    #[test]
    fn empty_result_warns() {
        let call = json!({"name": "ast_grep", "arguments": {"pat": "x"},
            "result": {"content": [{"type": "text", "text": "No matches found"}]}});
        let backend = TestBackend::new(60, 4);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Ast.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("No matches found"));
    }

    #[test]
    fn collapsed_results_cap_with_more_line() {
        let text: String = (0..10).map(|i| format!("src/f{i}.rs:1: match\n")).collect();
        let call = json!({"name": "ast_grep", "arguments": {"pat": "x"},
            "result": {"content": [{"type": "text", "text": text}]}});
        let backend = TestBackend::new(60, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Ast.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let shown: usize = buf.content().iter().filter(|c| c.symbol() == "f").count();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("4 more matches"));
        assert!(shown <= 6);
    }
}
