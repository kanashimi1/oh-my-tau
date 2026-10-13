//! Inline code highlighting — the port of `pi-tui/src/lang-from-path.ts`
//! plus the highlighter behind `render/code-cell.ts`.
//!
//! Stub. Planned: language detection from fence info strings / file paths and
//! syntactic highlighting into themed spans. Dependency choice (syntext vs
//! tree-sitter) is recorded in `runs/omp-rust/tui-spec/ARCHITECTURE.md`;
//! until then [`highlight`] returns unstyled text.

use ratatui::text::Text;

/// Highlight `code` written in `lang`. MVP: unstyled pass-through.
#[must_use]
pub fn highlight(code: &str, lang: Option<&str>) -> Text<'static> {
    let _ = lang;
    Text::from(code.to_owned())
}
