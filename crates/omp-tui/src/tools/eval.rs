//! `eval` tool renderer — port of `pi-tui/src/tools/eval.ts` (spec
//! `tools-a.md` §11): Python/JS computer cells.
//!
//! The cell head carries `Eval <lang>[: title]`; the output pane is
//! tail-capped to `EVAL_DEFAULT_PREVIEW_LINES` (10) rows, styled
//! `toolOutput`, with an error glyph + red tint on failure.

use ratatui::layout::Rect;
use ratatui::text::Line;
use serde_json::Value;

use crate::theme::Theme;

use super::{args_of, is_error, paint, result_text, status_line, styled, truncate, ToolRenderer};

/// Output pane rows (`EVAL_DEFAULT_PREVIEW_LINES = DEFAULT_TERMINAL_PREVIEW_LINES`).
pub(crate) const EVAL_DEFAULT_PREVIEW_LINES: usize = 10;
/// Summary budget for the code snippet.
const SUMMARY_BUDGET: usize = 60;

/// Renderer for the eval tool.
pub struct Eval;

impl ToolRenderer for Eval {
    fn summary(&self, call: &Value) -> String {
        let Some(args) = args_of(call) else {
            return String::new();
        };
        let lang = args.get("language").and_then(Value::as_str).unwrap_or("");
        let title = args.get("title").and_then(Value::as_str).unwrap_or("");
        let head = match (lang.is_empty(), title.is_empty()) {
            (true, true) => String::new(),
            (true, false) => title.to_owned(),
            (false, true) => lang.to_owned(),
            (false, false) => format!("{lang}: {title}"),
        };
        let code = args
            .get("code")
            .and_then(Value::as_str)
            .map(|c| c.lines().next().unwrap_or("").trim().to_owned())
            .filter(|c| !c.is_empty());
        match (head.is_empty(), code) {
            (true, None) => String::new(),
            (true, Some(c)) => truncate(&c, SUMMARY_BUDGET),
            (false, None) => head,
            (false, Some(c)) => format!("{head} {}", truncate(&c, SUMMARY_BUDGET)),
        }
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let settled = call.get("result").is_some();
        let icon = if is_error(call) {
            theme.glyph("failure")
        } else if settled {
            theme.glyph("success")
        } else {
            "⧗"
        };
        let mut lines = vec![status_line(
            theme,
            icon,
            "Eval",
            &self.summary(call),
            Vec::new(),
        )];
        match result_text(call) {
            None => lines.push(Line::from(styled(theme, "text_muted", "…"))),
            Some(text) => {
                let body: Vec<&str> = text.lines().collect();
                if body.is_empty() {
                    lines.push(Line::from(styled(theme, "text_muted", "(no output)")));
                } else {
                    // Output panes keep the live tail (spec: tail-capped).
                    let start = body.len().saturating_sub(EVAL_DEFAULT_PREVIEW_LINES);
                    for line in &body[start..] {
                        lines.push(Line::from(styled(theme, "toolOutput", (*line).to_owned())));
                    }
                    if start > 0 {
                        lines.push(Line::from(styled(
                            theme,
                            "text_muted",
                            format!("… {start} earlier lines [ctrl+o: Expand]"),
                        )));
                    }
                }
            }
        }
        paint(frame, area, lines);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    #[test]
    fn summary_uses_title_and_first_code_line() {
        let call = json!({"name": "eval", "arguments": {
            "language": "py", "title": "plot", "code": "import numpy\nprint(1)"}});
        assert_eq!(Eval.summary(&call), "py: plot import numpy");
    }

    #[test]
    fn output_keeps_live_tail() {
        let lines: String = (0..20).map(|i| format!("line {i}\n")).collect();
        let call = json!({"name": "eval", "arguments": {"language": "js", "code": "x"},
            "result": {"content": [{"type": "text", "text": lines}]}});
        let backend = TestBackend::new(60, 14);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Eval.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("line 19"));
        assert!(!text.contains("line 0 "));
        assert!(text.contains("10 earlier lines"));
    }

    #[test]
    fn error_result_renders_failure_glyph() {
        let call = json!({"name": "eval", "arguments": {"language": "py", "code": "1/0"},
            "result": {"isError": true, "content": [{"type": "text", "text": "ZeroDivisionError"}]}});
        let backend = TestBackend::new(60, 4);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Eval.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains('✗'));
        assert!(text.contains("ZeroDivisionError"));
    }
}
