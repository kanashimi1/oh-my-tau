//! Omi — the omp-rs mascot: a chibi anime girl drawn in truecolor
//! half-blocks (two pixels per cell: `▀` with fg = top pixel, bg = bottom
//! pixel), plus one-line kaomoji moods for the working indicator.
//!
//! The sprite is pixel art on a 26×26 grid (13 terminal rows). `.` is
//! transparent; every other character is a palette key.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

/// Sprite width in cells.
pub const WIDTH: usize = 26;

/// Pixel rows (two per terminal row).
const SPRITE: [&str; 26] = [
    "............hh............",
    "...........hHh............",
    "..........hhhhhh..........",
    ".......hhhHHHHHHhhh.......",
    ".....hhHHHLLLHHHHHHhh.....",
    "....hHHHHLLHHHHHHHRHHh....",
    "...hHHHHHHHHHHHHHRYRHHh...",
    "..hHHHHHHHHHHHHHHHRHHHHh..",
    "..hHHHHhHHHHHHhHHHHHHHHh..",
    ".hHHHHhSHHHHHhSSHHHHhHHHh.",
    ".hHHHhSSShHHhSSSShHHhSHHh.",
    ".hHHhKKKKKSSSSSSKKKKKhHHh.",
    ".hHHhKweeeSSSSSSweeeKhHHh.",
    ".hHHhSeKKeSSSSSSeKKeShHHh.",
    ".hHHhSEKKESSSSSSEKKEShHHh.",
    ".hHHhSlEElSSSSSSlEElShHHh.",
    ".hHhSBBSSSSSSSSSSSSBBShHh.",
    ".hHhSSSSSSSSMMSSSSSSSShHh.",
    ".hHHhSSSSSSSSSSSSSSSShHHh.",
    ".hHHHhssSSSSSSSSSSsshHHHh.",
    ".hHHHHhh.sSSSSSSs.hhHHHHh.",
    ".hHHHHh..CCsSSsCC..hHHHHh.",
    "..hHHh..OCCCCCCCCO..hHHh..",
    "..hHh..OOOCCRRCCOOO..hHh..",
    "..hh..OOOOOORROOOOOO..hh..",
    "..h..OOOOOOORROOOOOOO..h..",
];

/// Pixel color for a palette key (`None` = transparent).
fn color(key: char) -> Option<Color> {
    let (r, g, b) = match key {
        'H' => (236, 112, 214),
        'h' => (150, 70, 170),
        'L' => (255, 196, 240),
        'S' => (255, 231, 220),
        's' => (232, 186, 182),
        'E' => (40, 200, 226),
        'e' => (20, 110, 160),
        'w' => (255, 255, 255),
        'B' => (255, 150, 175),
        'M' => (205, 80, 115),
        'O' => (44, 48, 92),
        'C' => (238, 240, 250),
        'R' => (0, 219, 228),
        'l' => (150, 240, 250),
        'K' => (40, 20, 52),
        'Y' => (255, 224, 120),
        _ => return None,
    };
    Some(Color::Rgb(r, g, b))
}

/// Height in terminal rows.
#[must_use]
pub const fn height() -> usize {
    SPRITE.len().div_ceil(2)
}

/// The sprite as styled rows (`WIDTH` cells each, transparent cells are
/// plain spaces so the terminal background shows through).
#[must_use]
pub fn lines() -> Vec<Line<'static>> {
    let px = |row: usize, x: usize| -> Option<Color> {
        SPRITE.get(row).and_then(|r| r.chars().nth(x)).and_then(color)
    };
    (0..height())
        .map(|y| {
            let mut spans: Vec<Span<'static>> = Vec::with_capacity(WIDTH);
            for x in 0..WIDTH {
                let (top, bottom) = (px(2 * y, x), px(2 * y + 1, x));
                spans.push(match (top, bottom) {
                    (None, None) => Span::raw(" "),
                    (Some(t), None) => Span::styled("▀", Style::default().fg(t)),
                    (None, Some(b)) => Span::styled("▄", Style::default().fg(b)),
                    (Some(t), Some(b)) => Span::styled("▀", Style::default().fg(t).bg(b)),
                });
            }
            Line::from(spans)
        })
        .collect()
}

/// What Omi is doing (working indicator kaomoji).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    /// Model is thinking / streaming.
    Thinking,
    /// A tool is running.
    Working,
    /// Turn finished.
    Happy,
    /// Turn failed.
    Sad,
}

/// One-line face for `mood`; `tick` animates (blink / glance).
#[must_use]
pub fn face(mood: Mood, tick: u64) -> &'static str {
    match mood {
        Mood::Thinking => ["(•_• )", "( •_•)", "(•_• )", "(-_- )"][(tick % 4) as usize],
        Mood::Working => ["(•ᴗ•)~", "(•ᴗ•)~", "(-ᴗ-)~"][(tick % 3) as usize],
        Mood::Happy => "(◕ᴗ◕)✧",
        Mood::Sad => "(╥_╥)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sprite_rows_are_rectangular_and_use_known_colors() {
        for row in SPRITE {
            assert_eq!(row.chars().count(), WIDTH, "{row}");
            for c in row.chars().filter(|c| *c != '.') {
                assert!(color(c).is_some(), "unknown palette key {c}");
            }
        }
    }

    #[test]
    fn half_block_rows_are_width_cells() {
        let rows = lines();
        assert_eq!(rows.len(), 13);
        for row in &rows {
            let w: usize = row.spans.iter().map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref())).sum();
            assert_eq!(w, WIDTH);
        }
        // Both pixels set → upper half block with a background.
        assert!(rows[6].spans.iter().any(|s| s.content == "▀" && s.style.bg.is_some()));
    }
}
