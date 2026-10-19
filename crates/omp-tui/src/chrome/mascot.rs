//! Omi — the omp-rs mascot: a chibi anime girl drawn in truecolor
//! half-blocks (two pixels per cell: `▀` with fg = top pixel, bg = bottom
//! pixel), plus one-line kaomoji moods for the working indicator.
//!
//! The sprite is pixel art on a 36×40 grid (20 terminal rows). `.` is
//! transparent; every other character is a palette key.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

/// Sprite width in cells.
pub const WIDTH: usize = 36;

/// Pixel rows (two per terminal row): pink hair, cat-ear headphones with
/// cyan cups, navy hoodie. Outline pixels are darker shades of their fill.
const SPRITE: [&str; 40] = [
    "........a..................a........",
    "........aa................aa........",
    "........aaa..............aaa........",
    "........abaa....aaaa....aaba........",
    "........abbaaaaa....aaaaabba........",
    ".......aaaaaa.cccccccc.aaaaaa.......",
    ".......a.aacccddddddddcccaa.a.......",
    "........a.cddddddddddddddc.a........",
    ".......a.cddddddddddddddddc.a.......",
    "........edddddfffddddffdddde........",
    "......acddddffdddddddddffdddca......",
    "......cddddffddddddddddfdddddc......",
    ".....aeddddddddddddddddddddddea.....",
    ".....cddddddddddddddddddddddddc.....",
    ".....cddddddddddddddddddddddddc.....",
    ".....edddddddddddddddddddddddde.....",
    ".....edddddddddddddddddddddddde.....",
    "...aaaaddddddddddddddddddddddaaaa...",
    "...abbadddddddddgddgdddddddddabba...",
    "...abbaddddgdggggddggggdgddddabba...",
    "...abbaddddgdggggddggggdgddddabba...",
    "...abbaddhghhhhgggggghhhhghddabba...",
    "...abbaddhhhhhhhgggghhhhhhhddabba...",
    "...abbadddijjiiiggggijjiiidddabba...",
    "...abbadddijhhiiggggijhhiidddabba...",
    "...aaaadddkkhhkkggggkkhhkkdddaaaa...",
    ".....cddddkkhhkkggggkkhhkkddddc.....",
    ".....cddddllllmlggggllllmlddddc.....",
    ".....cddddnllllggggggllllnddddc.....",
    ".....cddooonggggggggggggnoooddc.....",
    ".....cdddnnnnggggppggggnnnndddc.....",
    ".....cdddnnnnnngggggqnnnnnndddc.....",
    "....cndddnnnnnnnrrrs.cnnnnndddnc....",
    "....cnnddnnnnnnnrrrs.cnnnnnddnnc....",
    "....cnndnnnnnnnnrrrs.cnnnnnndnnc....",
    "....cnndnnnnttnnrrrs.cttnnnndnnc....",
    "....cnnnnuuuuttuuuuuvttuuuunnnnc....",
    "....cnnnuuuuuuttuuuuttuuuuuunnnc....",
    "....cnnnuuuuuuuttuuttuuuuuuunnnc....",
    ".....cnuuuuuuuuutbbtuuuuuuuuunc.....",
];

/// Pixel color for a palette key (`None` = transparent).
fn color(key: char) -> Option<Color> {
    let (r, g, b) = match key {
        'a' => (60, 60, 80), // p
        'b' => (0, 219, 228), // a
        'c' => (110, 40, 120), // h (outline)
        'd' => (236, 112, 214), // H
        'e' => (150, 60, 150), // H (outline)
        'f' => (255, 196, 240), // L
        'g' => (255, 233, 222), // S
        'h' => (36, 18, 48), // K
        'i' => (30, 90, 170), // E1
        'j' => (255, 255, 255), // w
        'k' => (40, 170, 226), // E2
        'l' => (140, 235, 250), // E3
        'm' => (220, 250, 255), // w2
        'n' => (170, 72, 170), // h
        'o' => (255, 160, 180), // B
        'p' => (200, 70, 110), // M
        'q' => (214, 170, 170), // S (outline)
        'r' => (238, 196, 190), // S2
        's' => (190, 150, 150), // S2 (outline)
        't' => (232, 236, 250), // c2
        'u' => (44, 48, 92), // c
        'v' => (26, 28, 60), // c (outline)
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
        assert_eq!(rows.len(), 20);
        for row in &rows {
            let w: usize = row.spans.iter().map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref())).sum();
            assert_eq!(w, WIDTH);
        }
        // Both pixels set → upper half block with a background.
        assert!(rows[10].spans.iter().any(|s| s.content == "▀" && s.style.bg.is_some()));
    }
}
