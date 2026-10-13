//! `think` tool renderer — port of `pi-tui/src/tools/think.ts`.
//!
//! `inline: true`, call-only: the streamed `thoughts` render as italic
//! `thinkingText`-colored markdown; there is never a result block
//! (`renderResult` returns `undefined` in TS).

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;

use crate::theme::Theme;

use super::{args_of, paint, truncate, ToolRenderer};

/// Summary/inline budget for the collapsed transcript line.
const SUMMARY_BUDGET: usize = 72;

/// Renderer for the think tool.
pub struct Think;

impl ToolRenderer for Think {
    fn summary(&self, call: &Value) -> String {
        thoughts_of(call).map_or_else(String::new, |t| truncate(&t, SUMMARY_BUDGET))
    }

    fn render(&self, frame: &mut ratatui::Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
        let Some(thoughts) = thoughts_of(call) else {
            return;
        };
        let style = Style::default()
            .fg(theme.color("thinkingText"))
            .add_modifier(Modifier::ITALIC);
        let width = usize::from(area.width).max(1);
        let lines: Vec<Line<'static>> = thoughts
            .lines()
            .flat_map(|line| wrap(line, width))
            .map(|l| Line::from(Span::styled(l, style)))
            .collect();
        paint(frame, area, lines);
    }
}

/// Borrow the streamed thoughts argument.
fn thoughts_of(call: &Value) -> Option<String> {
    let args = args_of(call)?;
    let thoughts = args.get("thoughts").and_then(Value::as_str)?;
    Some(thoughts.replace('\r', ""))
}

/// Greedy word-wrap to `width` chars (TS Markdown component wraps inline).
fn wrap(line: &str, width: usize) -> Vec<String> {
    if line.chars().count() <= width {
        return vec![line.to_owned()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in line.split(' ') {
        let extra = word.chars().count() + usize::from(!cur.is_empty());
        if cur.chars().count() + extra > width && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    #[test]
    fn summary_flattens_and_truncates() {
        let call = json!({"name": "think", "arguments": {"thoughts": "line one\nline two"}});
        assert_eq!(Think.summary(&call), "line one line two");
        let long = json!({"name": "think", "arguments": {"thoughts": "x".repeat(200)}});
        assert!(Think.summary(&long).ends_with('…'));
    }

    #[test]
    fn render_paints_italic_thoughts() {
        let backend = TestBackend::new(40, 5);
        let mut terminal = Terminal::new(backend).unwrap();
        let call = json!({"name": "think", "arguments": {"thoughts": "pondering the port"}});
        terminal
            .draw(|f| Think.render(f, f.area(), &call, &Theme::default()))
            .unwrap();
        let buffer = terminal.backend().buffer();
        assert!(buffer.content().iter().any(|c| c.symbol() == "p"));
        let text: String = buffer.content().iter().map(|c| c.symbol()).collect();
        assert!(text.contains("pondering"));
    }

    #[test]
    fn wrap_splits_long_lines() {
        assert_eq!(wrap("aaa bbb ccc", 7), vec!["aaa bbb", "ccc"]);
    }
}
