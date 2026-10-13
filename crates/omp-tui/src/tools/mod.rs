//! Tool call renderers — ports of `pi-tui/src/tools/` and `pi-tui/src/render/`.
//!
//! Every tool family gets its own module with one [`ToolRenderer`]
//! implementation; [`render`] dispatches on the tool name from a settled
//! assistant message (`content[]` entries with `type: "toolCall"` /
//! `toolcall`, TS `pi-wire` shapes) or a `tool_execution_*` frame. The omp
//! tool inventory lives in omp-cli `src/tools/` (`bash`, `edit`, `read`,
//! `write`, `search` (glob+grep), `find`, `task`, `web_search`, `ask`,
//! `todo`, `think`, `mcp`, `lsp`, `checkpoint`, `jobs`, `eval`, `ast`,
//! `wait`, `goal`); spec details per family are in
//! `runs/omp-rust/tui-spec/tools*.md`.
//!
//! Shared chrome (collapse/expand glyphs, elapsed-time spinner, exit-code
//! badge) will live here in `render` once the first family lands.

pub mod ask;
pub mod bash;
pub mod edit;
pub mod read;
pub mod search;
pub mod task;
pub mod todo;
pub mod web;

use ratatui::layout::Rect;
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
}

/// Dispatch a tool call payload to its family's renderer. MVP: a themed
/// one-line summary for every tool; expanded views come with each family.
pub fn render(frame: &mut Frame<'_>, area: Rect, call: &Value, theme: &Theme) {
    use ratatui::style::Style;
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;
    let name = call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("tool");
    let summary = family(name).summary(call);
    let line = Line::from(vec![
        Span::styled(theme.glyph("collapsed"), Style::default().fg(theme.color("accent"))),
        Span::raw(" "),
        Span::styled(name.to_owned(), Style::default().fg(theme.color("accent"))),
        Span::raw(format!(" {summary}")),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// The renderer for a tool family name (unknown names fall back to a
/// generic passthrough).
#[must_use]
pub fn family(name: &str) -> &'static dyn ToolRenderer {
    match name {
        "bash" => &bash::Bash,
        "edit" => &edit::Edit,
        "read" => &read::Read,
        "write" => &write_placeholder::Write,
        "search" | "glob" | "grep" | "find" => &search::Search,
        "task" => &task::Task,
        "web_search" | "fetch" => &web::Web,
        "ask" => &ask::Ask,
        "todo" => &todo::Todo,
        _ => &ask::Generic,
    }
}

/// Helper for families not yet split into their own module: `write` renders
/// through the edit family until its implementer lands.
mod write_placeholder {
    use serde_json::Value;

    use crate::theme::Theme;

    /// Passthrough to the edit renderer.
    pub struct Write;

    impl super::ToolRenderer for Write {
        fn summary(&self, call: &Value) -> String {
            super::edit::Edit.summary(call)
        }
        fn render(
            &self,
            frame: &mut ratatui::Frame<'_>,
            area: ratatui::layout::Rect,
            call: &Value,
            theme: &Theme,
        ) {
            super::edit::Edit.render(frame, area, call, theme);
        }
    }
}
