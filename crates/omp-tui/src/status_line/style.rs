//! Fallback palette + icon table for the status line (chrome.md §7.2/§7.4).
//!
//! Every color resolves through the [`Theme`] first; these are the built-in
//! `dark.json` values used until `ThemeLoader` registers the tokens, and as
//! the defaults for custom themes that omit them.

use ratatui::style::Color;

use crate::theme::Theme;

/// Resolve a TS token name to a color, falling back to the dark-theme value.
#[must_use]
pub fn color(theme: &Theme, token: &str) -> Color {
    let resolved = theme.color(token);
    if resolved != Color::Reset {
        return resolved;
    }
    fallback(token, theme)
}

fn fallback(token: &str, theme: &Theme) -> Color {
    match token {
        "statusLineSep" => Color::Indexed(244),
        "statusLineModel" => Color::Rgb(0xd7, 0x87, 0xaf),
        "statusLinePath" => Color::Rgb(0x00, 0xaf, 0xaf),
        "statusLineGitClean" => Color::Rgb(0x5f, 0xaf, 0x5f),
        "statusLineGitDirty" => Color::Rgb(0xd7, 0xaf, 0x5f),
        "statusLineContext" => Color::Rgb(0x87, 0x87, 0xaf),
        "statusLineSpend" => Color::Rgb(0x5f, 0xaf, 0xaf),
        "statusLineStaged" => Color::Indexed(70),
        "statusLineDirty" => Color::Indexed(178),
        "statusLineUntracked" => Color::Indexed(39),
        "statusLineOutput" | "statusLineCost" => Color::Indexed(205),
        "statusLineSubagents" => theme.color("accent"),
        "thinkingHigh" => Color::Rgb(0xb2, 0x81, 0xd6),
        "muted" | "dim" => Color::DarkGray,
        _ => Color::Reset,
    }
}

/// The status line background color (token `statusLineBg`).
#[must_use]
pub fn bg(theme: &Theme) -> Color {
    match theme.color("statusLineBg") {
        Color::Reset => Color::Rgb(0x12, 0x12, 0x12),
        resolved => resolved,
    }
}

/// Icon from the theme's symbol preset (`symbols.ts:389+`), then the legacy
/// `Theme::glyph` table, then `default`.
#[must_use]
pub fn icon(theme: &Theme, name: &str, default: &'static str) -> &'static str {
    let s = theme.symbol(name);
    if !s.is_empty() {
        return s;
    }
    let g = theme.glyph(name);
    if g == "?" {
        default
    } else {
        g
    }
}

/// Clean-style muted chrome color: `dim`, then `text_muted`, then the
/// dark-theme fallback.
#[must_use]
pub fn dim(theme: &Theme) -> Color {
    for token in ["dim", "text_muted"] {
        let c = theme.color(token);
        if c != Color::Reset {
            return c;
        }
    }
    fallback("dim", theme)
}

/// Clean-style token resolution: `dim` routes through [`dim`].
#[must_use]
pub fn clean_color(theme: &Theme, token: &str) -> Color {
    if token == "dim" {
        dim(theme)
    } else {
        color(theme, token)
    }
}

/// Braille activity spinner frame for `now_ms` (`segments.ts:235`,
/// `SPINNER_ADVANCE_MS` = 80).
#[must_use]
pub fn activity_frame(now_ms: u64) -> &'static str {
    const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[((now_ms / 80) as usize) % FRAMES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_palette_covers_status_tokens() {
        let t = Theme::default();
        assert_eq!(color(&t, "statusLineSep"), Color::Indexed(244));
        assert_eq!(color(&t, "statusLineModel"), Color::Rgb(0xd7, 0x87, 0xaf));
        assert_eq!(bg(&t), Color::Rgb(0x12, 0x12, 0x12));
        assert_eq!(color(&t, "accent"), t.color("accent")); // present in theme
    }

    #[test]
    fn spinner_frames_rotate() {
        assert_eq!(activity_frame(0), "⠋");
        assert_eq!(activity_frame(80), "⠙");
        assert_eq!(activity_frame(80 * 10), "⠋");
    }
}
