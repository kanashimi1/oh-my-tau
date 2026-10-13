//! Inline terminal images — port of `pi-tui/src/kitty-graphics.ts`,
//! `render/sixel.ts`, `glyph-protocol.ts` and `components/image.ts`.
//!
//! Stub: protocol detection only. Planned: detect kitty/sixel/iTerm2 support
//! (`terminal-capabilities.ts`), budget placements per message
//! (`ImageBudget`, `DEFAULT_MAX_INLINE_IMAGES`), and fall back to the
//! placeholder glyph on unsupported terminals.

/// A detected terminal image protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageProtocol {
    /// No inline images; render placeholders.
    #[default]
    None,
    /// Kitty graphics protocol.
    Kitty,
    /// Sixel.
    Sixel,
    /// iTerm2 inline images.
    Iterm2,
}

/// How many inline images one message may show (TS `DEFAULT_MAX_INLINE_IMAGES`).
pub const DEFAULT_MAX_INLINE_IMAGES: usize = 5;

/// Detect the image protocol from environment variables the way
/// `terminal-capabilities.ts` does (`KITTY_WINDOW_ID`, `TERM` containing
/// `kitty`, `LS_COLORS`-style sixel hints, `TERM_PROGRAM`).
#[must_use]
pub fn detect() -> ImageProtocol {
    let term = std::env::var("TERM").unwrap_or_default();
    if std::env::var_os("KITTY_WINDOW_ID").is_some() || term.contains("kitty") {
        return ImageProtocol::Kitty;
    }
    if term.contains("sixel") {
        return ImageProtocol::Sixel;
    }
    if std::env::var("TERM_PROGRAM").as_deref() == Ok("iTerm.app") {
        return ImageProtocol::Iterm2;
    }
    ImageProtocol::None
}
