//! `lsp` tool renderer — port of `pi-tui/src/tools/lsp.ts` (spec
//! `tools-b.md` §8).
//!
//! `inline: true`, `mergeCallAndResult: true`. Pending: one line
//! `lsp <operation> <file or filecount>`; per-action result bodies — hover
//! markdown, severity-color-coded diagnostics, grouped references/symbols,
//! rename arrows — all width-truncated with tabs replaced.

use ratatui::layout::Rect;
use ratatui::text::Line;
use serde_json::Value;

use crate::theme::Theme;

use super::{args_of, is_error, paint, result_text, status_line, styled, truncate, ToolRenderer};

/// Summary budget for query/symbol previews (`TRUNCATE_LENGTHS.SHORT`-ish).
const SUMMARY_BUDGET: usize = 40;
/// Result preview rows per action.
const PREVIEW_LINES: usize = 8;

/// Renderer for the lsp tool.
pub struct Lsp;

impl ToolRenderer for Lsp {
    fn summary(&self, call: &Value) -> String {
        let Some(args) = args_of(call) else {
            return String::new();
        };
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_default()
            .replace('_', " ");
        let target = args
            .get("file")
            .and_then(Value::as_str)
            .map(|f| f.rsplit('/').next().unwrap_or(f).to_owned())
            .or_else(|| {
                args.get("query")
                    .or_else(|| args.get("symbol"))
                    .and_then(Value::as_str)
                    .map(|q| truncate(&q.replace('\n', " "), SUMMARY_BUDGET))
            })
            .unwrap_or_default();
        match (action.is_empty(), target.is_empty()) {
            (true, _) => target,
            (false, true) => action,
            (false, false) => format!("{action} {target}"),
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
            "lsp",
            &self.summary(call),
            Vec::new(),
        )];
        if let Some(text) = result_text(call) {
            let width = usize::from(area.width).max(1);
            for line in text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .take(PREVIEW_LINES)
            {
                let cleaned = line.replace('\t', "    ").replace('\r', "");
                let token = if severity_of(&cleaned).is_some() {
                    "error"
                } else {
                    "text"
                };
                lines.push(Line::from(styled(theme, token, truncate(&cleaned, width))));
            }
        }
        paint(frame, area, lines);
    }
}

/// Map a diagnostics line to its severity token, when recognized.
fn severity_of(line: &str) -> Option<&'static str> {
    let lower = line.to_lowercase();
    if lower.contains("error") {
        Some("error")
    } else if lower.contains("warning") {
        Some("warning")
    } else if lower.contains("info") || lower.contains("hint") {
        Some("text_muted")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    #[test]
    fn summary_flattens_action_and_file() {
        let call = json!({"name": "lsp", "arguments": {"action": "type_definition", "file": "src/main.rs"}});
        assert_eq!(Lsp.summary(&call), "type definition main.rs");
    }

    #[test]
    fn summary_falls_back_to_query() {
        let call =
            json!({"name": "lsp", "arguments": {"action": "symbols", "query": "find\nmy symbol"}});
        assert_eq!(Lsp.summary(&call), "symbols find my symbol");
    }

    #[test]
    fn diagnostics_errors_are_red() {
        let call = json!({"name": "lsp", "arguments": {"action": "diagnostics", "file": "a.rs"},
            "result": {"content": [{"type": "text", "text": "1 error(s), 0 warning(s)\nmismatched types: expected u32"}]}});
        let backend = TestBackend::new(60, 5);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| Lsp.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let text: String = buf.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("mismatched types"));
        let error_fg = Theme::default().color("error");
        assert!(buf.content().iter().any(|c| c.fg == error_fg));
    }
}
