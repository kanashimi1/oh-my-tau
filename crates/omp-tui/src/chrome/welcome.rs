//! Startup welcome banner — port of `pi-tui/src/prompt/welcome.ts`
//! (classic renderer): the gradient block-grid logo beside the `omp`
//! wordmark with the version under it, and one rotating `Tip:` line, both
//! centered in the terminal width. It is the first transcript block and
//! retires into scrollback with the rest of the history.

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
        .map(|l| l.strip_prefix("[NO-TERN]").map_or(l, str::trim_start).to_owned())
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
    for (name, label) in [("ctrl", "Ctrl+"), ("alt", "Alt+"), ("shift", "⇧"), ("super", "Super+")] {
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
        out.push_str(&if mods.is_empty() { base.to_owned() } else { base.to_uppercase() });
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
                out.push_str(&ks.split(',').map(format_key_hint).collect::<Vec<_>>().join("/"));
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
            Some((&tail[1..mid], &tail[mid + 2..mid + 2 + close], mid + 2 + close + 1))
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
        let text_style = Style::default().fg(theme.color("text")).add_modifier(Modifier::BOLD);
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

/// Clean style welcome: Omi (the pixel mascot) on the left; the gradient
/// wordmark, version, greeting, working directory and the tip beside her.
/// Falls back to the classic lockup when the terminal is too narrow.
#[must_use]
pub fn clean_welcome_lines(
    version: &str,
    tip: &str,
    cwd: &str,
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    use super::mascot;
    const MARGIN: usize = 2;
    const GAP: usize = 3;
    let side = width.saturating_sub(MARGIN + mascot::WIDTH + GAP + 1);
    if side < 24 {
        return welcome_lines(version, tip, width, theme);
    }
    let muted = Style::default().fg(theme.color("text_muted"));
    let text = Style::default().fg(theme.color("text"));
    let accent = Style::default().fg(theme.color("accent")).add_modifier(Modifier::BOLD);

    let mut beside: Vec<Vec<Span<'static>>> = vec![Vec::new(); mascot::height()];
    // Text column (wordmark … tip, 11 rows) centred beside the sprite.
    let top = mascot::height().saturating_sub(11) / 2;
    // Wordmark rows 2..=4, gradient left → right.
    for (i, row) in WORDMARK.iter().enumerate() {
        let n = WORDMARK_WIDTH.max(1) as f64;
        beside[top + i] = row
            .chars()
            .enumerate()
            .map(|(x, c)| {
                if c == ' ' {
                    Span::raw(" ")
                } else {
                    Span::styled(c.to_string(), Style::default().fg(gradient(x as f64 / n)))
                }
            })
            .collect();
    }
    beside[top].push(Span::styled("  rs", muted));
    beside[top + 3] = vec![Span::styled(format!("v{version}"), muted)];
    beside[top + 5] = vec![
        Span::styled("Omi", accent),
        Span::styled(" · hi! ready when you are.", text),
    ];
    let cwd_text = super::super::tools::truncate(cwd, side);
    beside[top + 6] = vec![Span::styled(cwd_text, muted)];
    if !tip.is_empty() {
        let body = expand_tip(tip.trim_end().trim_end_matches("[NEW]").trim_end());
        let wrapped = wrap(&format!("Tip: {body}"), side.min(TIP_MEASURE));
        let label = Style::default().fg(theme.color("accent")).add_modifier(Modifier::ITALIC);
        let body_style = muted.add_modifier(Modifier::ITALIC);
        for (i, line) in wrapped.iter().take(3).enumerate() {
            let row = &mut beside[top + 8 + i];
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
    for (row, art) in mascot::lines().into_iter().enumerate() {
        let mut spans = vec![Span::raw(" ".repeat(MARGIN))];
        spans.extend(art.spans);
        if !beside[row].is_empty() {
            spans.push(Span::raw(" ".repeat(GAP)));
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
    fn clean_welcome_puts_text_beside_the_mascot() {
        let rows = clean_welcome_lines("0.10.0", "Type `/` for commands", "/w/proj", 100, &Theme::default());
        let text: Vec<String> = rows
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(rows.len(), 1 + super::super::mascot::height());
        assert!(text.iter().any(|r| r.contains("v0.10.0")));
        assert!(text.iter().any(|r| r.contains("Omi · hi!")));
        assert!(text.iter().any(|r| r.contains("/w/proj")));
        assert!(text.iter().any(|r| r.contains("Tip: Type `/` for commands")));
        // Too narrow for the side column: classic lockup.
        let narrow = clean_welcome_lines("0.10.0", "", "/w", 40, &Theme::default());
        assert_eq!(narrow, welcome_lines("0.10.0", "", 40, &Theme::default()));
    }

    fn text(l: &Line<'_>) -> String {
        l.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn lockup_is_centered_like_ts() {
        let rows = welcome_lines("18.6.1", "Type `/` for commands", 120, &Theme::default());
        assert_eq!(text(&rows[0]).trim_end(), format!("{}████████████", " ".repeat(44)));
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
