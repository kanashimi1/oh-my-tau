//! Markdown rendering for assistant text — the port of
//! `pi-tui/src/chat/markdown.ts` (plus `render/` helpers).
//!
//! Stub. Planned: a `CommonMark` subset renderer that turns a chat entry's
//! text into ratatui `Text`/`Line` spans themed via the [`Theme`], with
//! headers, bold/italic/code, lists, blockquotes and links (OSC 8 via
//! ratatui hyperlink support). Until then [`plain`] passes text through
//! unchanged.

use ratatui::text::Text;

/// Render markdown source to display text. MVP: pass-through.
#[must_use]
pub fn plain(src: &str) -> Text<'static> {
    Text::from(src.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_through_keeps_text() {
        let t = plain("hello **world**");
        assert_eq!(t.lines.len(), 1);
    }
}
