//! Startup welcome banner — port of `pi-tui/src/prompt/welcome.ts`
//! (classic renderer): the gradient block-grid logo beside the `omp`
//! wordmark with the version under it, and one rotating `Tip:` line, both
//! centered in the terminal width. It is the first transcript block and
//! retires into scrollback with the rest of the history.
//!
//! The clean style (omp-rs default) shows the OMT τ logo instead
//! ([`clean_welcome_lines`]), shimmering for [`SHIMMER_MS`] after startup.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::theme::Theme;

/// Block-grid brand mark (`PI_LOGO`).
const PI_LOGO: [&str; 5] = [
    "████████████",
    "   ██  ██   ",
    "   ██  ██   ",
    "   ▒▒  ██   ",
    "       ██   ",
];
/// The `omp` wordmark in half-blocks, set beside the logo from its second row.
const WORDMARK: [&str; 3] = ["▄▀▀▄ █▀▄▀▄ █▀▀▄", "▀▄▄▀ █ █ █ █▄▄▀", "           █"];
const LOGO_WIDTH: usize = 12;
const WORDMARK_WIDTH: usize = 15;
const LOCKUP_GAP: usize = 4;
/// Widest a tip wraps.
const TIP_MEASURE: usize = 72;
/// Narrowest terminal that still shows the tip.
const TIP_MIN_COLUMNS: usize = 50;
/// Weight of a `[NEW]` tip in the random pick.
const NEW_TIP_WEIGHT: usize = 4;

const GRADIENT_STOPS: [(f64, f64, f64); 3] = [
    (248.0, 79.0, 204.0),
    (147.0, 98.0, 244.0),
    (0.0, 219.0, 228.0),
];

const TIPS_TXT: &str = include_str!("tips.txt");

/// Tips with the `[NO-TERN]` marker stripped (classic renderer shows all).
fn tips() -> Vec<String> {
    TIPS_TXT
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.strip_prefix("[NO-TERN]")
                .map_or(l, str::trim_start)
                .to_owned()
        })
        .collect()
}

fn is_new(tip: &str) -> bool {
    tip.trim_end().ends_with("[NEW]")
}

/// Weighted pick (`pickWeightedTip`); `r` uniform in `[0, 1)`.
#[must_use]
pub fn pick_weighted_tip(tips: &[String], r: f64) -> String {
    if tips.is_empty() {
        return String::new();
    }
    let weight = |t: &String| if is_new(t) { NEW_TIP_WEIGHT } else { 1 };
    let total: usize = tips.iter().map(weight).sum();
    let mut acc = r * total as f64;
    for t in tips {
        acc -= weight(t) as f64;
        if acc < 0.0 {
            return t.clone();
        }
    }
    tips[tips.len() - 1].clone()
}

/// Pick the session's tip from wall-clock entropy (TS latches
/// `Math.random()` once per component).
#[must_use]
pub fn random_tip() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    pick_weighted_tip(&tips(), f64::from(nanos % 1_000_000) / 1_000_000.0)
}

/// `formatKeyHint` with the unicode symbol preset.
#[must_use]
pub fn format_key_hint(key: &str) -> String {
    let parts: Vec<&str> = key.split('+').collect();
    let (base, mods) = parts.split_last().map_or(("", &[][..]), |(b, m)| (*b, m));
    let mut out = String::new();
    for (name, label) in [
        ("ctrl", "Ctrl+"),
        ("alt", "Alt+"),
        ("shift", "⇧"),
        ("super", "Super+"),
    ] {
        if mods.contains(&name) {
            out.push_str(label);
        }
    }
    let lower = base.to_lowercase();
    let glyph = match lower.as_str() {
        "esc" | "escape" => Some("⎋"),
        "enter" | "return" => Some("⏎"),
        "space" => Some("␣"),
        "tab" => Some("⇥"),
        "backspace" => Some("⌫"),
        "delete" => Some("⌦"),
        "up" => Some("↑"),
        "down" => Some("↓"),
        "left" => Some("←"),
        "right" => Some("→"),
        "shift" => Some("⇧"),
        "ctrl" => Some("Ctrl"),
        "alt" => Some("Alt"),
        _ => None,
    };
    if let Some(g) = glyph {
        out.push_str(g);
    } else if base.chars().count() == 1 {
        out.push_str(&if mods.is_empty() {
            base.to_owned()
        } else {
            base.to_uppercase()
        });
    } else {
        let mut cs = base.chars();
        if let Some(f) = cs.next() {
            out.extend(f.to_uppercase());
            out.push_str(cs.as_str());
        }
    }
    out
}

/// Expand `{key:..}` / `{keys:a,b}` / `{tap:..}` placeholders; links
/// `[label](url)` become `label (url)`.
fn expand_tip(tip: &str) -> String {
    let mut out = String::new();
    let mut rest = tip;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let Some(end) = rest[start..].find('}') else {
            out.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let inner = &rest[start + 1..start + end];
        match inner.split_once(':') {
            Some(("key", k)) => out.push_str(&format_key_hint(k)),
            Some(("keys", ks)) => {
                out.push_str(
                    &ks.split(',')
                        .map(format_key_hint)
                        .collect::<Vec<_>>()
                        .join("/"),
                );
            }
            Some(("tap", k)) => {
                let h = format_key_hint(k);
                out.push_str(&format!("{h}{h}"));
            }
            _ => out.push_str(&rest[start..=start + end]),
        }
        rest = &rest[start + end + 1..];
    }
    out.push_str(rest);
    expand_links(&out)
}

fn expand_links(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(open) = rest.find('[') {
        let tail = &rest[open..];
        let parsed = tail.find("](").and_then(|mid| {
            let close = tail[mid + 2..].find(')')?;
            Some((
                &tail[1..mid],
                &tail[mid + 2..mid + 2 + close],
                mid + 2 + close + 1,
            ))
        });
        match parsed {
            Some((label, url, consumed)) => {
                out.push_str(&rest[..open]);
                out.push_str(&format!("{label} ({url})"));
                rest = &rest[open + consumed..];
            }
            None => {
                out.push_str(&rest[..=open]);
                rest = &rest[open + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Greedy word wrap at `width` cells, lines right-trimmed.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        let w = UnicodeWidthStr::width(word);
        if !cur.is_empty() && UnicodeWidthStr::width(cur.as_str()) + 1 + w > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Gradient color at diagonal position `t` (`gradientEscape`, truecolor).
fn gradient(t: f64) -> Color {
    let seg = t * (GRADIENT_STOPS.len() - 1) as f64;
    let i = (seg.floor() as usize).min(GRADIENT_STOPS.len() - 2);
    let f = seg - i as f64;
    let (a, b) = (GRADIENT_STOPS[i], GRADIENT_STOPS[i + 1]);
    let ch = |x: f64, y: f64| (x + (y - x) * f).round().clamp(0.0, 255.0) as u8;
    Color::Rgb(ch(a.0, b.0), ch(a.1, b.1), ch(a.2, b.2))
}

fn logo_spans(row: usize) -> Vec<Span<'static>> {
    let x_span = (LOGO_WIDTH - 1) as f64;
    let y_span = (PI_LOGO.len() - 1) as f64;
    PI_LOGO[row]
        .chars()
        .enumerate()
        .map(|(x, c)| {
            if c == ' ' {
                Span::raw(" ")
            } else {
                let t = (x as f64 / x_span + row as f64 / y_span) / 2.0;
                Span::styled(c.to_string(), Style::default().fg(gradient(t)))
            }
        })
        .collect()
}

fn indent(widest: usize, width: usize) -> Span<'static> {
    Span::raw(" ".repeat(width.saturating_sub(widest) / 2))
}

/// The welcome block rows at `width` columns: lockup, blank, tip.
#[must_use]
pub fn welcome_lines(version: &str, tip: &str, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    let room = width.saturating_sub(2);
    if room < 4 {
        return Vec::new();
    }
    let version_text = format!("v{version}");
    let version_w = UnicodeWidthStr::width(version_text.as_str());
    let lockup_w = LOGO_WIDTH + LOCKUP_GAP + WORDMARK_WIDTH.max(version_w);
    let mut lines: Vec<Line<'static>> = Vec::new();
    if room >= lockup_w {
        let dim = Style::default().fg(theme.color("text_muted"));
        let text_style = Style::default()
            .fg(theme.color("text"))
            .add_modifier(Modifier::BOLD);
        for row in 0..PI_LOGO.len() {
            let mut spans = vec![indent(lockup_w, width)];
            spans.extend(logo_spans(row));
            let beside: Option<Span<'static>> = match row {
                1..=3 => Some(Span::styled(WORDMARK[row - 1], text_style)),
                4 => Some(Span::styled(version_text.clone(), dim)),
                _ => None,
            };
            if let Some(b) = beside {
                spans.push(Span::raw(" ".repeat(LOCKUP_GAP)));
                spans.push(b);
            }
            lines.push(Line::from(spans));
        }
    } else if room >= LOGO_WIDTH {
        for row in 0..PI_LOGO.len() {
            let mut spans = vec![indent(LOGO_WIDTH, width)];
            spans.extend(logo_spans(row));
            lines.push(Line::from(spans));
        }
    }
    if width >= TIP_MIN_COLUMNS && !tip.is_empty() {
        let label = "Tip: ";
        let body = expand_tip(tip.trim_end().trim_end_matches("[NEW]").trim_end());
        let measure = room.min(TIP_MEASURE);
        if measure >= label.len() + 8 {
            let wrapped = wrap(&format!("{label}{body}"), measure);
            if !wrapped.is_empty() {
                lines.push(Line::default());
                let muted = Style::default()
                    .fg(theme.color("text_muted"))
                    .add_modifier(Modifier::ITALIC);
                let label_style = Style::default()
                    .fg(theme.color("accent"))
                    .add_modifier(Modifier::ITALIC);
                for (i, line) in wrapped.iter().enumerate() {
                    let w = UnicodeWidthStr::width(line.as_str());
                    let mut spans = vec![indent(w, width)];
                    match line.strip_prefix(label).filter(|_| i == 0) {
                        Some(rest) => {
                            spans.push(Span::styled(label, label_style));
                            spans.push(Span::styled(rest.to_owned(), muted));
                        }
                        None => spans.push(Span::styled(line.clone(), muted)),
                    }
                    lines.push(Line::from(spans));
                }
            }
        }
    }
    lines
}

/// The OMT τ glyph, 10 × 8 pixels (`#` lit), drawn as 4 half-block rows.
const TAU: [&str; 8] = [
    "##########",
    "##########",
    "   ###    ",
    "   ###    ",
    "   ###    ",
    "   ###    ",
    "    ###   ",
    "     #####",
];
const TAU_WIDTH: usize = 10;
/// OMT gradient ends (also the idle τ of the clean status line).
pub const PINK: (u8, u8, u8) = (255, 92, 168);
/// See [`PINK`].
pub const CYAN: (u8, u8, u8) = (80, 220, 255);
const KATAKANA: &str = "オー・マイ・タウ";
/// Startup shimmer length: the logo animates this long, then freezes.
pub const SHIMMER_MS: u64 = 1500;
/// One colour cycle of the shimmer; [`SHIMMER_MS`] is a whole number of
/// cycles, so the last frame lands on the static palette.
const SHIMMER_CYCLE_MS: u64 = 750;
/// Narrowest terminal that gets the logo block (else the one-line form).
const CLEAN_MIN_COLUMNS: usize = 60;
const CLEAN_MARGIN: usize = 2;
const CLEAN_GAP: usize = 3;

/// Shimmer phase in `[0, 1)` at `elapsed` since startup, or `None` once
/// the shimmer is over and the welcome shows its static palette.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn shimmer_phase(elapsed: std::time::Duration) -> Option<f64> {
    let ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    (ms < SHIMMER_MS).then(|| (ms % SHIMMER_CYCLE_MS) as f64 / SHIMMER_CYCLE_MS as f64)
}

/// `s` cut to `width` cells, with `…` when anything was dropped.
fn fit(s: &str, width: usize) -> String {
    if UnicodeWidthStr::width(s) <= width {
        return s.to_owned();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in s.chars() {
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}

fn rgb((r, g, b): (u8, u8, u8)) -> Color {
    Color::Rgb(r, g, b)
}

/// Sunset stripe colour of pixel row `row` (0..8). Phase 0 is the static
/// palette (row `r` = stop `r`); the phase rotates the stops cyclically.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn stripe(row: usize, phase: f64) -> Color {
    use crate::magic::SUNSET;
    let n = SUNSET.len();
    let x = (row as f64 + phase * n as f64).rem_euclid(n as f64);
    let i = x.floor() as usize % n;
    let f = x - x.floor();
    rgb(crate::magic::gradient_at(
        &[SUNSET[i], SUNSET[(i + 1) % n]],
        f,
    ))
}

/// Gradient position of `OMT` char `i`, ping-ponged by the phase so the
/// sweep runs out and back and phase 0 / 1 are the static spread.
#[allow(clippy::cast_precision_loss)]
fn omt_color(i: usize, phase: f64) -> Color {
    let t = i as f64 / 2.0 + 2.0 * phase;
    let f = t.rem_euclid(2.0);
    rgb(crate::magic::gradient_at(
        &[PINK, CYAN],
        if f > 1.0 { 2.0 - f } else { f },
    ))
}

fn omt_spans(phase: f64) -> Vec<Span<'static>> {
    "OMT"
        .chars()
        .enumerate()
        .map(|(i, c)| {
            Span::styled(
                c.to_string(),
                Style::default()
                    .fg(omt_color(i, phase))
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

/// The τ glyph as 4 rows of [`TAU_WIDTH`] half-block cells (`▀` fg = top
/// pixel, bg = bottom pixel; unlit cells are plain spaces).
fn tau_rows(phase: f64) -> Vec<Vec<Span<'static>>> {
    let lit = |row: usize, x: usize| TAU[row].as_bytes().get(x) == Some(&b'#');
    (0..TAU.len() / 2)
        .map(|y| {
            let (top, bottom) = (2 * y, 2 * y + 1);
            (0..TAU_WIDTH)
                .map(|x| match (lit(top, x), lit(bottom, x)) {
                    (false, false) => Span::raw(" "),
                    (true, false) => Span::styled("▀", Style::default().fg(stripe(top, phase))),
                    (false, true) => Span::styled("▄", Style::default().fg(stripe(bottom, phase))),
                    (true, true) => Span::styled(
                        "▀",
                        Style::default()
                            .fg(stripe(top, phase))
                            .bg(stripe(bottom, phase)),
                    ),
                })
                .collect()
        })
        .collect()
}

/// Clean style welcome: the OMT τ logo (sunset half-block stripes) with
/// the `OMT` wordmark, version + katakana reading and the tip beside it;
/// one line (`τ OMT v…`) on narrow terminals. `phase` 0 is the static
/// palette, other values come from [`shimmer_phase`].
#[must_use]
pub fn clean_welcome_lines(
    version: &str,
    tip: &str,
    width: usize,
    theme: &Theme,
    phase: f64,
) -> Vec<Line<'static>> {
    let side = width.saturating_sub(CLEAN_MARGIN + TAU_WIDTH + CLEAN_GAP + 1);
    let muted = Style::default().fg(theme.color("text_muted"));
    if side < 24 || width < CLEAN_MIN_COLUMNS {
        let mut spans = vec![
            Span::raw(" ".repeat(CLEAN_MARGIN)),
            Span::styled(
                "τ",
                Style::default().fg(rgb(PINK)).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
        ];
        spans.extend(omt_spans(phase));
        spans.push(Span::styled(format!(" v{version}"), muted));
        return vec![Line::from(spans)];
    }
    let mut beside: Vec<Vec<Span<'static>>> = vec![Vec::new(); TAU.len() / 2];
    beside[0] = omt_spans(phase);
    beside[1] = vec![Span::styled(
        fit(&format!("v{version} · {KATAKANA}"), side),
        muted,
    )];
    if !tip.is_empty() {
        let body = expand_tip(tip.trim_end().trim_end_matches("[NEW]").trim_end());
        let measure = side.min(TIP_MEASURE);
        let mut wrapped = wrap(&format!("Tip: {body}"), measure);
        if wrapped.len() > 2 {
            wrapped.truncate(2);
            let last = &mut wrapped[1];
            *last = fit(&format!("{last} …"), measure);
        }
        let label = Style::default()
            .fg(theme.color("accent"))
            .add_modifier(Modifier::ITALIC);
        let body_style = muted.add_modifier(Modifier::ITALIC);
        for (i, line) in wrapped.iter().enumerate() {
            let row = &mut beside[2 + i];
            match line.strip_prefix("Tip: ").filter(|_| i == 0) {
                Some(rest) => {
                    row.push(Span::styled("Tip: ", label));
                    row.push(Span::styled(rest.to_owned(), body_style));
                }
                None => row.push(Span::styled(line.clone(), body_style)),
            }
        }
    }
    let mut lines = vec![Line::default()];
    for (row, art) in tau_rows(phase).into_iter().enumerate() {
        let mut spans = vec![Span::raw(" ".repeat(CLEAN_MARGIN))];
        spans.extend(art);
        if !beside[row].is_empty() {
            spans.push(Span::raw(" ".repeat(CLEAN_GAP)));
            spans.extend(std::mem::take(&mut beside[row]));
        }
        lines.push(Line::from(spans));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_welcome_is_the_omt_logo_with_text_beside() {
        let theme = Theme::default();
        let rows = clean_welcome_lines("0.10.0", "Type `/` for commands", 100, &theme, 0.0);
        let t: Vec<String> = rows.iter().map(text).collect();
        assert_eq!(rows.len(), 5, "{t:#?}");
        assert!(t[0].is_empty());
        assert_eq!(t[1], "  ▀▀▀▀▀▀▀▀▀▀   OMT");
        assert_eq!(t[2], "     ▀▀▀       v0.10.0 · オー・マイ・タウ");
        assert_eq!(t[3], "     ▀▀▀       Tip: Type `/` for commands");
        assert_eq!(t[4], "      ▀▀▀▄▄▄");
        assert!(t.iter().all(|r| !r.contains("Omi") && !r.contains("/w")));
        // Pixel row r carries sunset stop r: row 0 = fg stop 0 / bg stop 1.
        let first = &rows[1].spans[1];
        assert_eq!(first.style.fg, Some(rgb(crate::magic::SUNSET[0])));
        assert_eq!(first.style.bg, Some(rgb(crate::magic::SUNSET[1])));
        // OMT: bold pink → cyan.
        let o = rows[1].spans.iter().find(|s| s.content == "O").unwrap();
        assert_eq!(o.style.fg, Some(rgb(PINK)));
        assert!(o.style.add_modifier.contains(Modifier::BOLD));
        let t = rows[1].spans.iter().find(|s| s.content == "T").unwrap();
        assert_eq!(t.style.fg, Some(rgb(CYAN)));
    }

    #[test]
    fn clean_welcome_wraps_tip_to_two_rows_and_narrow_is_one_line() {
        let theme = Theme::default();
        let long = "word ".repeat(40);
        let rows = clean_welcome_lines("1.2.3", &long, 70, &theme, 0.0);
        assert_eq!(rows.len(), 5);
        let t: Vec<String> = rows.iter().map(text).collect();
        assert!(t[3].contains("Tip: word"), "{t:#?}");
        assert!(t[4].trim_end().ends_with('…'), "{t:#?}");
        assert!(t.iter().all(|r| UnicodeWidthStr::width(r.as_str()) <= 70));
        let narrow = clean_welcome_lines("1.2.3", "tip", 50, &theme, 0.0);
        assert_eq!(narrow.len(), 1);
        assert_eq!(text(&narrow[0]), "  τ OMT v1.2.3");
        assert_eq!(narrow[0].spans[1].style.fg, Some(rgb(PINK)));
    }

    #[test]
    fn shimmer_runs_then_freezes_on_the_static_rows() {
        use std::time::Duration;
        let theme = Theme::default();
        assert_eq!(shimmer_phase(Duration::ZERO), Some(0.0));
        let mid = shimmer_phase(Duration::from_millis(375)).unwrap();
        assert!((mid - 0.5).abs() < 1e-9);
        assert_eq!(shimmer_phase(Duration::from_millis(SHIMMER_MS)), None);
        assert_eq!(shimmer_phase(Duration::from_mins(1)), None);
        let static_rows = clean_welcome_lines("1.0.0", "tip", 100, &theme, 0.0);
        let moving = clean_welcome_lines("1.0.0", "tip", 100, &theme, mid);
        assert_ne!(moving, static_rows);
        // The phase only recolours: same text.
        assert_eq!(
            moving.iter().map(text).collect::<Vec<_>>(),
            static_rows.iter().map(text).collect::<Vec<_>>()
        );
        // After 1.5 s the rows are the static rows.
        let done = shimmer_phase(Duration::from_millis(1600)).unwrap_or(0.0);
        assert_eq!(
            clean_welcome_lines("1.0.0", "tip", 100, &theme, done),
            static_rows
        );
        // Just before the end the cycle is nearly closed (no jump at freeze).
        let late = shimmer_phase(Duration::from_millis(SHIMMER_MS - 1)).unwrap();
        assert!(late > 0.99);
        assert_eq!(stripe(3, 0.0), rgb(crate::magic::SUNSET[3]));
        assert_eq!(stripe(3, 1.0), stripe(3, 0.0));
        assert_eq!(omt_color(2, 1.0), omt_color(2, 0.0));
    }

    fn text(l: &Line<'_>) -> String {
        l.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn lockup_is_centered_like_ts() {
        let rows = welcome_lines("18.6.1", "Type `/` for commands", 120, &Theme::default());
        assert_eq!(
            text(&rows[0]).trim_end(),
            format!("{}████████████", " ".repeat(44))
        );
        assert_eq!(
            text(&rows[4]).trim_end(),
            format!("{}       ██       v18.6.1", " ".repeat(44))
        );
        assert!(text(&rows[6]).trim_start().starts_with("Tip: Type"));
    }

    #[test]
    fn key_hints_use_keycaps() {
        assert_eq!(format_key_hint("shift+tab"), "⇧⇥");
        assert_eq!(format_key_hint("ctrl+d"), "Ctrl+D");
        assert_eq!(expand_tip("Press {key:alt+p} now"), "Press Alt+P now");
    }

    #[test]
    fn narrow_terminal_drops_tip() {
        let rows = welcome_lines("1.0.0", "tip", 40, &Theme::default());
        assert!(rows.iter().all(|r| !text(r).contains("Tip")));
    }
}
